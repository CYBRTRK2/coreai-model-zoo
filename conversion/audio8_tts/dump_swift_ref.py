#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""Write the reference files the CoreAIKit smoke test (`Audio8SmokeTests`) and the gate app read.

    <swift_ref>/manifest.json         fixtures: name, lang, text, seed, voice (reference text + codes) or null,
                                      prompt row 0 ids and the packed [11,P] prompt, oracle frames, the Python
                                      engine run's codes (`--tag`, the arm that ships) and its frame count
    <swift_ref>/<name>.noise_slow.f32  [T, 2, 4097] float32 — the oracle's uniform draws (normal, RAS-high)
    <swift_ref>/<name>.noise_fast.f32  [T, 9, 4096] float32
    <swift_ref>/<name>.python_codes.json  the Python engine free run's [10][T] codes on the same bundles
    <swift_ref>/voices/<name>.json     Audio8Voice for the clone fixtures

The Swift host, fed the same draws, must make the same choices as the Python engine run on the same bundles
(frame for frame until the fp16 GPU noise flips a knife-edge — the identical prefix and the count are what the test
reports), and the same prompt ids as the publisher's processor (asserted here already: prompt.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
from _paths import work_path  # noqa: E402

WORK = work_path("_audio8_tts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="gate tag whose port codes ship (e.g. slowint8_fastfp16_cl2048_w32_t160_gpu)")
    ap.add_argument("--out", default=str(WORK / "swift_ref"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "voices").mkdir(parents=True, exist_ok=True)
    fx = json.loads((HERE / "fixtures.json").read_text())
    meta = json.loads((WORK / "oracle" / "oracle.json").read_text())["fixtures"]
    gate_dir = WORK / "gate" / args.tag
    rows = []
    for f in fx["fixtures"]:
        name = f["name"]
        orc = np.load(WORK / "oracle" / f"{name}.npz")
        info = meta[name]
        orc["noise_slow"].astype(np.float32).tofile(out / f"{name}.noise_slow.f32")
        orc["noise_fast"].astype(np.float32).tofile(out / f"{name}.noise_fast.f32")
        codes_py = np.load(gate_dir / f"{name}.port_codes.npy")
        (out / f"{name}.python_codes.json").write_text(json.dumps(codes_py.astype(int).tolist()))
        voice = None
        if info.get("reference"):
            voice = {"referenceText": info["reference"]["reference_text"], "codes": orc["ref_codes"].astype(int).tolist()}
            (out / "voices" / f"{name}.json").write_text(json.dumps(voice, ensure_ascii=False))
        rows.append({
            "name": name, "lang": f["lang"], "text": f["text"], "seed": f["seed"],
            "voice": f"voices/{name}.json" if voice else None,
            "prompt_length": int(orc["prompt"].shape[1]),
            "prompt_rows": orc["prompt"].astype(int).reshape(-1).tolist(),
            "oracle_frames": int(orc["codes"].shape[1]),
            "oracle_steps": int(orc["semantic"].shape[0]),
            "python_frames": int(codes_py.shape[1]),
            "python_codes": f"{name}.python_codes.json",
            "noise_slow": f"{name}.noise_slow.f32", "noise_fast": f"{name}.noise_fast.f32",
            "noise_steps": int(orc["noise_slow"].shape[0]),
        })
    (out / "manifest.json").write_text(json.dumps({"tag": args.tag, "generation": fx["generation"], "fixtures": rows},
                                                  ensure_ascii=False, indent=1))
    print(f"[swift_ref] {len(rows)} fixtures -> {out}")


if __name__ == "__main__":
    main()
