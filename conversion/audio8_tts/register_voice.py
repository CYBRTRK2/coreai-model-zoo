#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""Register a reference voice: a 0.5–30 s recording + its exact transcript -> `voice.json` for the Swift host.

    python register_voice.py --wav ref.wav --text "The exact transcript." --out voice.json [--encoder <.aimodel>]

Without `--encoder` the codes come from the publisher's fp32 codec in torch (the oracle path); with it, from the
exported `audio8_codec_encoder_fp16_t<T>.aimodel` on the Core AI GPU (the on-device path). Audio is converted to
mono 44.1 kHz and right-padded to the encoder bucket; the first ceil(len / 2048) frames are kept.

`voice.json` = {"referenceText": ..., "codes": [[...] × 10]} — `Audio8Voice` in CoreAIKit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from _paths import hf_snapshot  # noqa: E402

HF_ID = "Edge0/Audio8-TTS-Preview-0.6b"
REVISION = "f07040f3d151f1ba0253bfb92cb2f5dd38b44594"
FRAME = 2048
SR = 44100


def load_44k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        from torchaudio.functional import resample

        audio = resample(torch.from_numpy(audio), sr, SR).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default=None, help="audio8_codec_encoder_fp16_t<T>.aimodel (default: fp32 torch codec)")
    args = ap.parse_args()
    wav = load_44k(args.wav)
    seconds = wav.size / SR
    if not 0.5 <= seconds <= 30.0:
        raise SystemExit(f"reference must be 0.5–30 s, got {seconds:.2f} s")
    frames = math.ceil(wav.size / FRAME)
    if args.encoder:
        import coreai.runtime as rt

        path = Path(args.encoder)
        bucket = int(path.name.split("_t")[-1].split(".")[0])
        if frames > bucket:
            raise SystemExit(f"{seconds:.1f} s needs {frames} frames; the encoder bucket is {bucket}")
        padded = np.zeros(bucket * FRAME, np.float32)
        padded[: wav.size] = wav

        async def run():
            m = await rt.AIModel.load(str(path), rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
            fn = m.load_function("main")
            r = await fn(inputs={"audio": rt.NDArray(padded.reshape(1, 1, -1))})
            return r["codes"].numpy()[0, :, :frames]

        codes = asyncio.run(run())
        source = path.name
    else:
        from transformers import AutoConfig

        import audio8_model as M

        snap = Path(hf_snapshot(HF_ID, revision=REVISION))
        cfg = AutoConfig.from_pretrained(str(snap), trust_remote_code=True)
        pub = M.load_publisher_codec(snap, cfg)
        with torch.inference_mode():
            codes, lengths = pub.encode(torch.from_numpy(wav).reshape(1, 1, -1))
        codes = codes[0, :, : int(lengths[0])].numpy()
        source = "publisher codec fp32 (torch)"
    voice = {"referenceText": " ".join(args.text.strip().split()), "codes": codes.astype(int).tolist(),
             "source": {"wav": Path(args.wav).name, "seconds": round(seconds, 3), "frames": int(codes.shape[1]), "encoder": source}}
    Path(args.out).write_text(json.dumps(voice, ensure_ascii=False))
    print(f"[voice] {codes.shape[1]} frames ({seconds:.2f} s) -> {args.out} via {source}")


if __name__ == "__main__":
    main()
