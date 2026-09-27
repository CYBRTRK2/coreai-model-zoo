#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""Speaker similarity for the voice-clone fixtures: WavLM-Base-Plus-SV x-vector cosine.

`microsoft/wavlm-base-plus-sv` (Microsoft UniSpeech, its LICENSE file governs) run through transformers'
`WavLMForXVector` on 16 kHz audio. For each row the cosine between the x-vector of `wav` and of `reference`; the
oracle's own output against the same reference is the yardstick the port's output is read against (the metric is
relative: an x-vector cosine is not a probability that two clips share a speaker).

    python speaker_sim.py --manifest gate/spk_manifest.json --out gate/speaker.json
manifest = [{"id", "arm", "wav", "reference"}]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MODEL = "microsoft/wavlm-base-plus-sv"


def load_16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        from torchaudio.functional import resample

        audio = resample(torch.from_numpy(audio), sr, 16000).numpy()
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = json.loads(Path(args.manifest).read_text())

    from transformers import AutoFeatureExtractor, WavLMForXVector

    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = WavLMForXVector.from_pretrained(MODEL).eval()
    cache: dict[str, np.ndarray] = {}

    @torch.inference_mode()
    def embed(path: str) -> np.ndarray:
        if path not in cache:
            inputs = fe(load_16k(path), sampling_rate=16000, return_tensors="pt")
            cache[path] = torch.nn.functional.normalize(model(**inputs).embeddings, dim=-1)[0].numpy()
        return cache[path]

    for r in rows:
        r["cos"] = float(embed(r["wav"]) @ embed(r["reference"]))
    by_arm: dict[str, list[float]] = {}
    for r in rows:
        by_arm.setdefault(r.get("arm", "all"), []).append(r["cos"])
    summary = {arm: {"n": len(v), "mean": float(np.mean(v)), "min": float(np.min(v))} for arm, v in by_arm.items()}
    Path(args.out).write_text(json.dumps({"model": MODEL, "rows": rows, "summary": summary}, ensure_ascii=False, indent=1))
    for arm, s in summary.items():
        print(f"[speaker] {arm}: n {s['n']} cos mean {s['mean']:.4f} min {s['min']:.4f}")


if __name__ == "__main__":
    main()
