#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""Gate B for Fun-ASR-Nano: the NumPy front end (``frontend.fbank_lfr``) vs the oracle's features.

For every clip in ``fixtures/meta.json``: read the wav (float32 = int16 / 32768, the path funasr
takes through soundfile), compute ``fbank_lfr``, and compare with ``oracle/<name>.npz['speech']``
(the fp32 torch features captured at the encoder input). Also checks the frame count ``L`` and
the audio-slot count ``N = fake_token_len(L)`` against the oracle.

PASS = same L and N on every clip and max|Δ| < 1e-2 everywhere. Writes a per-clip JSON table.

Run with the shared venv:
    ~/code/coreai/coreai-models/.venv/bin/python conversion/funasr_nano/gate_frontend.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import work_path  # noqa: E402
from frontend import fake_token_len, fbank_lfr  # noqa: E402

WORK = work_path("_funasr_nano")
THRESHOLD = 1e-2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(WORK / "logs" / "r1_gate_frontend.json"))
    args = ap.parse_args()

    meta = json.loads((WORK / "fixtures" / "meta.json").read_text())
    rows = []
    for c in meta["clips"]:
        wav, sr = sf.read(str(WORK / "fixtures" / c["path"]), dtype="float32")
        assert sr == 16000 and wav.ndim == 1
        feats = fbank_lfr(wav)
        o = np.load(WORK / "oracle" / f"{c['name']}.npz")
        ref = o["speech"]
        n_ref = int(o["fake_token_len"])
        same_shape = feats.shape == ref.shape
        d = np.abs(feats.astype(np.float64) - ref.astype(np.float64)) if same_shape else None
        rows.append({
            "name": c["name"], "L": feats.shape[0], "L_oracle": ref.shape[0],
            "N": fake_token_len(feats.shape[0]), "N_oracle": n_ref,
            "max_abs": float(d.max()) if same_shape else None,
            "mean_abs": float(d.mean()) if same_shape else None,
            "argmax_frame": int(np.unravel_index(d.argmax(), d.shape)[0]) if same_shape else None,
        })
    ok_shape = all(r["L"] == r["L_oracle"] and r["N"] == r["N_oracle"] for r in rows)
    maxes = [r["max_abs"] for r in rows if r["max_abs"] is not None]
    worst = max(rows, key=lambda r: -1 if r["max_abs"] is None else r["max_abs"])
    summary = {
        "clips": len(rows),
        "L_and_N_match_all": ok_shape,
        "max_abs_worst": max(maxes), "worst_clip": worst["name"],
        "max_abs_median": float(np.median(maxes)),
        "mean_abs_mean": float(np.mean([r["mean_abs"] for r in rows if r["mean_abs"] is not None])),
        "threshold": THRESHOLD,
        "pass": ok_shape and max(maxes) < THRESHOLD,
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "clips": rows}, indent=1))
    for r in rows[:5]:
        print(f"  {r['name']:<6} L={r['L']} N={r['N']} max|Δ|={r['max_abs']:.3e} mean|Δ|={r['mean_abs']:.3e}")
    print(json.dumps(summary))
    print("PASS" if summary["pass"] else "FAIL")


if __name__ == "__main__":
    main()
