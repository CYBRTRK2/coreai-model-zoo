#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""E6: how close the fp16 audio encoder runs to fp16's ceiling (65,504) — fp32 torch, every fixture clip.

The encoder bundle computes in fp16; any activation past 65,504 becomes inf. This runs the fp32
re-authored encoder (``funasr_encoder.py``) exactly as the bundle is fed (NumPy front end, zero-pad
to 500 frames + mask) and records the absolute maximum over the valid rows of:

    input      feats * sqrt(512) + PE
    <layer>.mid       the residual after the attention branch (input of norm2)
    <layer>.out       the layer output (residual after the FFN)
    <layer>.attn / .ffn   the two branch outputs
    adaptor.linear1 / adaptor.linear2 / adaptor.blocks.* as above

for every SAN-M layer (encoders0, encoders.0-48, tp_encoders.0-19), the norms and the adaptor, on
the 155 fixtures plus ``fixtures_extra/concat30_en.wav`` (exactly 30 s). Then the loudest clip again
with the waveform scaled x0.25, x1 and x4 (clipped to [-1, 1]). -> ``logs/r2_fp16_headroom.json``.

    ~/code/coreai/coreai-models/.venv/bin/python conversion/funasr_nano/fp16_headroom.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import work_path  # noqa: E402
from frontend import fbank_lfr  # noqa: E402
from funasr_encoder import D, L_MAX, FunASRNanoAudioEncoder, load_weights  # noqa: E402

WORK = work_path("_funasr_nano")
FP16_MAX = 65504.0


class Taps:
    """Forward hooks recording |activation| max over the valid rows (mask set per call)."""

    def __init__(self, enc: FunASRNanoAudioEncoder) -> None:
        self.valid = 0
        self.vals: dict[str, float] = {}
        e, a = enc.audio_encoder, enc.audio_adaptor
        layers = ([("encoders0", e.encoders0[0])] + [(f"encoders.{i}", m) for i, m in enumerate(e.encoders)]
                  + [(f"tp_encoders.{i}", m) for i, m in enumerate(e.tp_encoders)]
                  + [(f"adaptor.blocks.{i}", m) for i, m in enumerate(a.blocks)])
        for name, layer in layers:
            layer.register_forward_hook(self._out(f"{name}.out"))
            layer.norm2.register_forward_pre_hook(self._in(f"{name}.mid"))
            layer.self_attn.register_forward_hook(self._out(f"{name}.attn"))
            layer.feed_forward.register_forward_hook(self._out(f"{name}.ffn"))
        e.encoders0[0].norm1.register_forward_pre_hook(self._in("input"))
        e.after_norm.register_forward_hook(self._out("after_norm"))
        e.tp_norm.register_forward_hook(self._out("tp_norm"))
        a.linear1.register_forward_hook(self._out("adaptor.linear1"))
        a.linear2.register_forward_hook(self._out("adaptor.linear2"))

    def _record(self, name: str, t: torch.Tensor) -> None:
        self.vals[name] = float(t[0, : self.valid].abs().max())

    def _out(self, name: str):
        return lambda _m, _i, out: self._record(name, out)

    def _in(self, name: str):
        return lambda _m, args: self._record(name, args[0])


def features(wav: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, int]:
    f = fbank_lfr(wav)
    L = f.shape[0]
    x = np.zeros((1, L_MAX, f.shape[1]), np.float32)
    x[0, :L] = f
    m = np.zeros((1, L_MAX), np.float32)
    m[0, :L] = 1
    return torch.from_numpy(x), torch.from_numpy(m), L


@torch.no_grad()
def run(enc, taps: Taps, wav: np.ndarray) -> dict:
    x, m, L = features(wav)
    taps.valid, taps.vals = L, {}
    enc(x, m)
    worst = max(taps.vals, key=taps.vals.get)
    return {"L": L, "absmax": taps.vals[worst], "where": worst, "values": dict(taps.vals)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    enc = load_weights(FunASRNanoAudioEncoder(), WORK / "hf" / "model.safetensors", torch.float32)
    taps = Taps(enc)
    meta = json.loads((WORK / "fixtures" / "meta.json").read_text())["clips"]
    clips = [(c["name"], WORK / "fixtures" / c["path"]) for c in meta if not args.only or c["name"] in args.only]
    clips.append(("concat30_en (fixtures_extra)", WORK / "fixtures_extra" / "concat30_en.wav"))

    rows, per_point = [], {}
    t0 = time.perf_counter()
    for i, (name, path) in enumerate(clips):
        wav, sr = sf.read(str(path), dtype="float32")
        assert sr == 16000
        r = run(enc, taps, wav)
        for k, v in r.pop("values").items():
            if v > per_point.get(k, (0.0, ""))[0]:
                per_point[k] = (v, name)
        rows.append({"name": name, **r, "wav_peak": float(np.abs(wav).max())})
        if i < 3 or i % 30 == 0:
            print(f"[{i + 1}/{len(clips)}] {name}: L={r['L']} absmax {r['absmax']:.0f} at {r['where']}", flush=True)

    worst = max(rows, key=lambda r: r["absmax"])
    wpath = dict(clips)[worst["name"]]
    wav, _ = sf.read(str(wpath), dtype="float32")
    gains = {}
    for g in (0.25, 1.0, 4.0):
        w = np.clip(wav * g, -1.0, 1.0)
        r = run(enc, taps, w)
        r.pop("values")
        gains[f"x{g:g}"] = {**r, "clipped_samples": int((np.abs(wav * g) > 1.0).sum()),
                            "headroom": FP16_MAX / r["absmax"]}
        print(f"[gain x{g:g}] {worst['name']}: absmax {r['absmax']:.0f} at {r['where']} "
              f"(headroom {FP16_MAX / r['absmax']:.2f}x)", flush=True)

    top = sorted(per_point.items(), key=lambda kv: -kv[1][0])
    summary = {
        "fp16_max": FP16_MAX, "clips": len(rows),
        "absmax": worst["absmax"], "absmax_clip": worst["name"], "absmax_where": worst["where"],
        "headroom": FP16_MAX / worst["absmax"],
        "clip_absmax_quantiles": {q: float(np.quantile([r["absmax"] for r in rows], q)) for q in (0.5, 0.9, 1.0)},
        "top_points": [{"point": k, "absmax": v, "clip": c} for k, (v, c) in top[:12]],
        "gain_sweep_on_loudest": gains,
        "note": f"absmax over valid rows of every tapped activation, fp32 torch; D={D}, sqrt(D)={math.sqrt(D):.3f}",
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    out = {"summary": summary, "per_point_max": {k: {"absmax": v, "clip": c} for k, (v, c) in top}, "clips": rows}
    (WORK / "logs" / "r2_fp16_headroom.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
