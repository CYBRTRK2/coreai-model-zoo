#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""Gate: the re-authored fixed-shape audio path (fp32 torch) vs the funasr oracle, every clip.

Input = the oracle's own features (``oracle/<name>.npz['speech']``, ``[L, 560]``) zero-padded to the
export contract ``[1, 500, 560]`` with ``mask [1, 500]`` — so this isolates the encoder + adaptor
from the host front end. Compared per row (float64 cosine, max|Δ|):

- ``audio_embeds[:N]`` vs ``adaptor_out[:N]`` — what the LLM consumes (N = fake_token_len(L));
- ``encoder_out[:L]`` vs the oracle's ``encoder_out`` (SAN-M output after tp_norm);
- and ``N`` itself: ``fake_token_len(L)`` must equal the oracle's value on every clip.

PASS = audio_embeds per-row cos mean >= 0.9999 and min >= 0.999 on every clip, N equal everywhere.
Also checks that the shift-accumulate FSMN (the fallback if grouped conv1d does not lower) matches
the conv1d FSMN on the first clip.

Run with the shared venv:
    ~/code/coreai/coreai-models/.venv/bin/python conversion/funasr_nano/parity_encoder.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import work_path  # noqa: E402
from funasr_encoder import L_MAX, N_MAX, FunASRNanoAudioEncoder, fake_token_len, load_weights  # noqa: E402

WORK = work_path("_funasr_nano")
SAFETENSORS = WORK / "hf" / "model.safetensors"


def row_stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)
    return {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
            "max_abs": float(np.abs(a - b).max()), "ref_absmax": float(np.abs(b).max())}


def pad(speech: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    L = speech.shape[0]
    feats = np.zeros((1, L_MAX, speech.shape[1]), dtype=np.float32)
    feats[0, :L] = speech
    mask = np.zeros((1, L_MAX), dtype=np.float32)
    mask[0, :L] = 1.0
    return torch.from_numpy(feats), torch.from_numpy(mask)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=str(WORK / "logs" / "r1_parity_encoder.json"))
    args = ap.parse_args()

    meta = json.loads((WORK / "fixtures" / "meta.json").read_text())
    clips = [c for c in meta["clips"] if not args.only or c["name"] in args.only]
    t0 = time.perf_counter()
    enc = load_weights(FunASRNanoAudioEncoder(return_encoder_out=True), SAFETENSORS, torch.float32)
    print(f"[load] {time.perf_counter() - t0:.1f} s", flush=True)

    rows = []
    for i, c in enumerate(clips):
        o = np.load(WORK / "oracle" / f"{c['name']}.npz")
        speech = o["speech"]
        L = speech.shape[0]
        N = fake_token_len(L)
        feats, mask = pad(speech)
        t1 = time.perf_counter()
        audio_embeds, encoder_out, _ = enc(feats, mask)
        dt = time.perf_counter() - t1
        emb = row_stats(audio_embeds[:N].numpy(), o["adaptor_out"][:N])
        eo = row_stats(encoder_out[:L].numpy(), o["encoder_out"])
        rows.append({"name": c["name"], "L": L, "N": N, "N_oracle": int(o["fake_token_len"]),
                     "audio_embeds": emb, "encoder_out": eo, "forward_s": round(dt, 3)})
        if i < 5 or i % 25 == 0:
            print(f"[{i + 1}/{len(clips)}] {c['name']}: L={L} N={N} emb cos {emb['cos_mean']:.7f}/{emb['cos_min']:.7f} "
                  f"max|Δ| {emb['max_abs']:.3e} | enc cos {eo['cos_min']:.7f} max|Δ| {eo['max_abs']:.3e} ({dt:.2f} s)",
                  flush=True)

    # shift-accumulate FSMN == conv1d FSMN (first clip)
    shift = load_weights(FunASRNanoAudioEncoder(fsmn="shift"), SAFETENSORS, torch.float32)
    o = np.load(WORK / "oracle" / f"{clips[0]['name']}.npz")
    feats, mask = pad(o["speech"])
    a_conv = enc(feats, mask)[0].numpy()
    a_shift = shift(feats, mask).numpy()
    fsmn_check = {"clip": clips[0]["name"], "max_abs_shift_vs_conv": float(np.abs(a_conv - a_shift).max()),
                  "ref_absmax": float(np.abs(a_conv).max())}

    n_ok = all(r["N"] == r["N_oracle"] for r in rows)
    worst_mean = min(rows, key=lambda r: r["audio_embeds"]["cos_mean"])
    worst_min = min(rows, key=lambda r: r["audio_embeds"]["cos_min"])
    worst_abs = max(rows, key=lambda r: r["audio_embeds"]["max_abs"])
    worst_enc = min(rows, key=lambda r: r["encoder_out"]["cos_min"])
    summary = {
        "clips": len(rows),
        "N_formula_matches_all": n_ok,
        "audio_embeds_cos_mean_worst": worst_mean["audio_embeds"]["cos_mean"], "cos_mean_worst_clip": worst_mean["name"],
        "audio_embeds_cos_min_worst": worst_min["audio_embeds"]["cos_min"], "cos_min_worst_clip": worst_min["name"],
        "audio_embeds_max_abs_worst": worst_abs["audio_embeds"]["max_abs"], "max_abs_worst_clip": worst_abs["name"],
        "encoder_out_cos_min_worst": worst_enc["encoder_out"]["cos_min"], "encoder_out_worst_clip": worst_enc["name"],
        "encoder_out_max_abs_worst": max(r["encoder_out"]["max_abs"] for r in rows),
        "fsmn_shift_vs_conv": fsmn_check,
        "pass": n_ok and all(r["audio_embeds"]["cos_mean"] >= 0.9999 and r["audio_embeds"]["cos_min"] >= 0.999
                             for r in rows),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "clips": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print("PASS" if summary["pass"] else "FAIL")


if __name__ == "__main__":
    main()
