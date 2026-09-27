#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""Gate A for the Fun-ASR-Nano audio encoder ``.aimodel`` on the Mac GPU, every fixture clip.

Host path exactly as the app will run it: wav -> ``frontend.fbank_lfr`` (NumPy) -> zero-pad to
``[1, 500, 560]`` + ``mask [1, 500]`` -> bundle (GPU preferred explicitly, not ``default()``) ->
``audio_embeds[:N]``. Compared with the fp32 oracle's ``adaptor_out[:N]`` per row.

PASS = per-row cosine mean >= 0.999 and min >= 0.99 on every clip (the qwen3_asr bar).

Also measured: the first call after load, then 5 warm calls on the longest clip (median); the Mac
GPU is shared with other sessions, so every time is marked contended. A red arm proves the gate
can fail: the shortest clip with ``mask`` all ones (padding treated as audio).

Run with the shared venv:
    ~/code/coreai/coreai-models/.venv/bin/python conversion/funasr_nano/gate_encoder.py \
        --bundle ~/code/coreai/_funasr_nano/exports/funasr_nano_audio_encoder_fp16_l500.aimodel
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import work_path  # noqa: E402
from frontend import fake_token_len, fbank_lfr  # noqa: E402
from funasr_encoder import L_MAX  # noqa: E402

import coreai.runtime as rt  # noqa: E402

WORK = work_path("_funasr_nano")


def row_stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)
    return {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()), "max_abs": float(np.abs(a - b).max()),
            "nan": bool(np.isnan(a).any())}


def inputs(feats: np.ndarray, dtype, all_valid: bool = False) -> dict:
    L = feats.shape[0]
    x = np.zeros((1, L_MAX, feats.shape[1]), dtype=dtype)
    x[0, :L] = feats
    m = np.ones((1, L_MAX), dtype=dtype) if all_valid else np.zeros((1, L_MAX), dtype=dtype)
    m[0, :L] = 1
    return {"feats": rt.NDArray(np.ascontiguousarray(x)), "mask": rt.NDArray(np.ascontiguousarray(m))}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--input-dtype", choices=("fp16", "fp32"), default="fp16")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    bundle = Path(args.bundle).expanduser()
    dtype = np.float16 if args.input_dtype == "fp16" else np.float32
    out_json = Path(args.out) if args.out else WORK / "logs" / f"r1_gate_{bundle.stem}.json"

    meta = json.loads((WORK / "fixtures" / "meta.json").read_text())
    clips = [c for c in meta["clips"] if not args.only or c["name"] in args.only]
    feats = {}
    for c in clips:
        wav, sr = sf.read(str(WORK / "fixtures" / c["path"]), dtype="float32")
        feats[c["name"]] = fbank_lfr(wav)

    t0 = time.perf_counter()
    model = await rt.AIModel.load(bundle, rt.SpecializationOptions.from_preferred_compute_unit_kind(
        rt.ComputeUnitKind.gpu()))
    fn = model.load_function("main")
    load_s = time.perf_counter() - t0

    rows, first_call_s = [], None
    for i, c in enumerate(clips):
        f = feats[c["name"]]
        L, N = f.shape[0], fake_token_len(f.shape[0])
        o = np.load(WORK / "oracle" / f"{c['name']}.npz")
        assert int(o["fake_token_len"]) == N and o["speech"].shape[0] == L
        t1 = time.perf_counter()
        res = await asyncio.wait_for(fn(inputs(f, dtype)), timeout=600)
        dt = time.perf_counter() - t1
        if first_call_s is None:
            first_call_s = dt
        emb = res["audio_embeds"].numpy().astype(np.float32)
        st = row_stats(emb[:N], o["adaptor_out"][:N])
        rows.append({"name": c["name"], "L": L, "N": N, **st, "call_s": round(dt, 4)})
        if i < 5 or i % 25 == 0:
            print(f"[{i + 1}/{len(clips)}] {c['name']}: L={L} N={N} cos {st['cos_mean']:.6f}/{st['cos_min']:.6f} "
                  f"max|Δ| {st['max_abs']:.3e} ({dt * 1000:.1f} ms)", flush=True)

    # warm timing on the longest clip (the bundle is fixed-shape, so every clip costs the same)
    longest = max(clips, key=lambda c: feats[c["name"]].shape[0])
    warm = []
    for _ in range(5):
        t1 = time.perf_counter()
        await fn(inputs(feats[longest["name"]], dtype))
        warm.append(time.perf_counter() - t1)

    # red arm: padding treated as audio on the shortest clip
    shortest = min(clips, key=lambda c: feats[c["name"]].shape[0])
    f = feats[shortest["name"]]
    N = fake_token_len(f.shape[0])
    o = np.load(WORK / "oracle" / f"{shortest['name']}.npz")
    red = await fn(inputs(f, dtype, all_valid=True))
    red_st = row_stats(red["audio_embeds"].numpy().astype(np.float32)[:N], o["adaptor_out"][:N])

    worst_mean = min(rows, key=lambda r: r["cos_mean"])
    worst_min = min(rows, key=lambda r: r["cos_min"])
    worst_abs = max(rows, key=lambda r: r["max_abs"])
    size = subprocess.run(["du", "-sh", str(bundle)], capture_output=True, text=True).stdout.split()[0]
    summary = {
        "bundle": str(bundle), "du_sh": size, "compute_unit": "gpu (preferred, explicit)",
        "clips": len(rows),
        "cos_mean_worst": worst_mean["cos_mean"], "cos_mean_worst_clip": worst_mean["name"],
        "cos_min_worst": worst_min["cos_min"], "cos_min_worst_clip": worst_min["name"],
        "max_abs_worst": worst_abs["max_abs"], "max_abs_worst_clip": worst_abs["name"],
        "any_nan": any(r["nan"] for r in rows),
        "timing_contended": {"load_s": round(load_s, 3), "first_call_ms": round(first_call_s * 1000, 2),
                             "warm5_median_ms": round(statistics.median(warm) * 1000, 2),
                             "warm5_ms": [round(w * 1000, 2) for w in warm], "warm_clip": longest["name"],
                             "note": "Mac GPU shared with other sessions (contended)"},
        "red_arm_mask_all_ones": {"clip": shortest["name"], "L": int(f.shape[0]), "N": N, **red_st},
        "pass": all(r["cos_mean"] >= 0.999 and r["cos_min"] >= 0.99 and not r["nan"] for r in rows),
    }
    out_json.write_text(json.dumps({"summary": summary, "clips": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print("PASS" if summary["pass"] else "FAIL")


if __name__ == "__main__":
    asyncio.run(main())
