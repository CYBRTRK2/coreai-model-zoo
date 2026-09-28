#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""Eager parity: the plain-torch re-author (fp32, CPU) against the fp32 oracle, teacher-forced.

For each fixture: windowed prefill (default 32) + one decode per frame on the oracle's own tokens, the fast AR on
the oracle's codebooks, and the codec decoder on the oracle's codes. Everything is fp32 on both sides, so the
bar is near-exactness (cos >= 0.99999, every argmax equal, every replayed sample equal); a miss here is a
re-authoring bug, not precision.

    ~/code/coreai/coreai-models/.venv/bin/python conversion/audio8_tts/parity_audio8.py --only ja_1 en_1 clone_en_1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from _paths import hf_snapshot, work_path  # noqa: E402

import audio8_model as M  # noqa: E402
from replay import TorchPorts, codec_gate, summarize, teacher_forced  # noqa: E402

HF_ID = "Edge0/Audio8-TTS-Preview-0.6b"
REVISION = "f07040f3d151f1ba0253bfb92cb2f5dd38b44594"
WORK = work_path("_audio8_tts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["ja_1", "en_1", "clone_en_1"])
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--cl", type=int, default=M.MAX_SEQ_LEN)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--out", default=str(WORK / "logs" / "parity_eager.json"))
    args = ap.parse_args()
    torch.set_num_threads(8)
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    snap = Path(hf_snapshot(HF_ID, revision=REVISION))
    t0 = time.time()
    slow, fast = M.load_slow_fast(snap / "model.safetensors", cl=args.cl, dtype=dtype)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(snap), trust_remote_code=True)
    pub = M.load_publisher_codec(snap, cfg)
    codec = M.build_codec_decoder(pub).to(dtype)
    print(f"[load] re-author {args.dtype} in {time.time() - t0:.1f}s", flush=True)
    ports = TorchPorts(slow, fast, codec, args.cl, dtype)

    gen = json.loads((HERE / "fixtures.json").read_text())["generation"]
    results = {}
    for name in args.only:
        orc = np.load(WORK / "oracle" / f"{name}.npz")
        t1 = time.time()
        tf = teacher_forced(ports, orc, gen, window=args.window, max_frames=args.max_frames)
        cg = codec_gate(ports, orc)
        s = summarize(tf)
        s["codec"] = cg
        s["wall_s"] = round(time.time() - t1, 1)
        results[name] = s
        print(f"[{name}] prefill cos {s['prefill']['logits_cos']:.7f} argmax {s['prefill']['argmax_eq']} | "
              f"slow {s['slow_steps']} steps cos min {s['slow_cos_min']:.7f} max|Δ| {s['slow_maxabs_max']:.2e} "
              f"argmax {s['slow_argmax_eq']}/{s['slow_steps']} sample {s['slow_sample_eq']}/{s['slow_steps']} miss {s['slow_sample_miss_margins']} | "
              f"fast {s['fast_logits']} cos min {s['fast_cos_min']:.7f} argmax {s['fast_argmax_eq']}/{s['fast_logits']} "
              f"sample {s['fast_sample_eq']}/{s['fast_logits']} miss {s['fast_sample_miss_margins']} | "
              f"codec cos {cg['cos']:.7f} max|Δ| {cg['maxabs']:.2e} logmel {cg['logmel_cos']:.6f} ({cg['seconds']:.1f}s) | {s['wall_s']}s",
              flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"dtype": args.dtype, "window": args.window, "cl": args.cl, "results": results}, indent=1))
    ok = all(r["slow_sample_eq"] == r["slow_steps"] and r["fast_sample_eq"] == r["fast_logits"]
             and r["slow_argmax_eq"] == r["slow_steps"]
             and r["codec"]["cos"] > 0.9999 for r in results.values())
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
