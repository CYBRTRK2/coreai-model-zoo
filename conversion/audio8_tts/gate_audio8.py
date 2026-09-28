#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""The engine gate: the three `.aimodel`s through `coreai.runtime` against the fp32 oracle, on every fixture.

Three ladders per fixture, all on the GPU (the compute unit that ships; `--unit cpu` is the parity control):

  1. teacher-forced  the oracle's tokens are fed; every slow-logits vector (4097), slow hidden and fast-logits
                     vector (9 × 4096 per frame) is compared: cosine, argmax equality, and whether the publisher's
                     sampler with the oracle's recorded noise would have made the oracle's choice from the port's
                     logits. A miss is reported with the oracle's own Gumbel margin at that draw (a knife-edge
                     when small).
  2. codec           the oracle's codes, right-padded to the bucket, decoded by the codec graph: wav cosine, max
                     abs difference and log-mel cosine against the oracle wav.
  3. free run        the port's own loop with the oracle's noise: frames identical to the oracle until the first
                     divergence (a divergence is a sampling flip at a knife-edge, not an error); the audio is
                     decoded and written, then transcribed by the fp32 Fun-ASR oracle (`asr_judge.py`, WER / CER
                     against the fixture text, the oracle's own audio scored the same way) and, for the voice-clone
                     fixtures, scored for speaker similarity against the reference clip (`speaker_sim.py`).

    ~/code/coreai/coreai-models/.venv/bin/python conversion/audio8_tts/gate_audio8.py --mode int8 --codec-frames 160
-> `$ZOO_WORK_ROOT/_audio8_tts/gate/<tag>/gate.json` (+ wavs), and a one-screen summary.

Each fixture runs in its own child interpreter (`--worker`): the Python `coreai.runtime` bindings leak one IOSurface
per call and a fixture costs ~2,000 calls (10 per frame, two passes), so a single process dies after ~7 fixtures
(`Failed to allocate storage for NDArray ... sk: ioSurface`) — pocket-tts-port.md, defect 2. The Swift framework does
not leak.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from _paths import work_path  # noqa: E402

import audio8_model as M  # noqa: E402
from export_audio8 import EnginePorts  # noqa: E402
from replay import codec_gate, free_run, summarize, teacher_forced  # noqa: E402

WORK = work_path("_audio8_tts")
ORACLE_VENV = work_path("_funasr_nano", "venv-oracle", "bin", "python")


def decode_in_buckets(ports, codes: np.ndarray, bucket: int) -> np.ndarray:
    """Whole-utterance decode through a fixed bucket: windows of `bucket` frames, the first `bucket - 128`
    context frames of each later window are re-decoded and dropped (every op is causal; window 128 is the post
    transformer's attention span, so a frame's output depends on at most the 127 frames before it)."""
    T = codes.shape[1]
    ctx = M.CODEC_WINDOW
    out = []
    start = 0
    while start < T:
        win_start = max(0, start - ctx) if start else 0
        win = codes[:, win_start:win_start + bucket]
        real = win.shape[1]
        padded = np.zeros((M.NUM_CODEBOOKS, bucket), np.int32)
        padded[:, :real] = win
        wav = np.asarray(ports.codec_decode(padded[None]), np.float32).reshape(-1)[: real * M.CODEC_FRAME]
        keep_from = (start - win_start) * M.CODEC_FRAME
        out.append(wav[keep_from:])
        start = win_start + real
    return np.concatenate(out)[: T * M.CODEC_FRAME]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="int8", help="slow AR mode (int8 | fp16)")
    ap.add_argument("--fast-mode", default=None, help="fast AR mode (default: same as --mode)")
    ap.add_argument("--cl", type=int, default=M.MAX_SEQ_LEN)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--codec-frames", type=int, default=160)
    ap.add_argument("--exports", default=str(WORK / "exports"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--unit", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--worker", action="store_true", help="(internal) run the fixtures of --only in this process")
    return ap.parse_args()


def bundle_paths(args):
    exports = Path(args.exports)
    fast_mode = args.fast_mode or args.mode
    slow = exports / f"audio8_slow_ar_{args.mode}_cl{args.cl}_w{args.window}.aimodel"
    fast = exports / f"audio8_fast_ar_{fast_mode}.aimodel"
    codec = exports / f"audio8_codec_decoder_fp16_t{args.codec_frames}.aimodel"
    for p in (slow, fast, codec):
        assert p.exists(), p
    tag = args.tag or f"slow{args.mode}_fast{fast_mode}_cl{args.cl}_w{args.window}_t{args.codec_frames}_{args.unit}"
    return slow, fast, codec, tag


def run_fixture(ports, name: str, orc, info: dict, gen: dict, args, out_dir: Path) -> dict:
    t1 = time.time()
    tf = summarize(teacher_forced(ports, orc, gen, window=args.window))
    cg = codec_gate(ports, orc, bucket=args.codec_frames) if orc["codes"].shape[1] <= args.codec_frames else \
        {"note": f"{orc['codes'].shape[1]} frames > bucket; whole-utterance decode below"}
    fr = free_run(ports, orc, gen, window=args.window)
    wav = decode_in_buckets(ports, fr["codes"], args.codec_frames) if fr["frames"] else np.zeros(0, np.float32)
    wav_path = out_dir / f"{name}.port.wav"
    sf.write(wav_path, wav, M.CODEC_SR, subtype="FLOAT")
    np.save(out_dir / f"{name}.port_codes.npy", fr["codes"])
    fr_out = {k: v for k, v in fr.items() if k not in ("codes", "semantics")}
    fr_out["seconds"] = wav.size / M.CODEC_SR
    res = {"teacher_forced": tf, "codec": cg, "free_run": fr_out, "wall_s": round(time.time() - t1, 1)}
    print(f"[{name}] slow {tf['slow_sample_eq']}/{tf['slow_steps']} (argmax {tf['slow_argmax_eq']}, cos min {tf['slow_cos_min']:.5f}, "
          f"miss {tf['slow_sample_miss_margins']}) | fast {tf['fast_sample_eq']}/{tf['fast_logits']} (argmax {tf['fast_argmax_eq']}, "
          f"cos min {tf['fast_cos_min']:.5f}, miss {tf['fast_sample_miss_margins'][:6]}{'…' if len(tf['fast_sample_miss_margins']) > 6 else ''}) | "
          f"codec cos {cg.get('cos', float('nan')):.5f} logmel {cg.get('logmel_cos', float('nan')):.4f} | "
          f"free run {fr['frames']} frames (oracle {fr['oracle_frames']}), identical prefix {fr['identical_frames_prefix']}, eos {fr['ended_with_eos']} | "
          f"slow {tf['slow_decode_ms_median']:.1f} ms fast {tf['fast_frame_ms_median']:.1f} ms/frame | {res['wall_s']}s", flush=True)
    return res


def worker(args):
    slow, fast, codec, tag = bundle_paths(args)
    out_dir = WORK / "gate" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = json.loads((HERE / "fixtures.json").read_text())["generation"]
    oracle_meta = json.loads((WORK / "oracle" / "oracle.json").read_text())
    t0 = time.time()
    ports = EnginePorts(slow, fast, codec, args.cl, unit=args.unit)
    load_s = time.time() - t0
    for name in args.only:
        orc = np.load(WORK / "oracle" / f"{name}.npz")
        res = run_fixture(ports, name, orc, oracle_meta["fixtures"][name], gen, args, out_dir)
        res["load_s"] = load_s
        (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1))


def main():
    args = parse_args()
    if args.worker:
        worker(args)
        return
    slow, fast, codec, tag = bundle_paths(args)
    out_dir = WORK / "gate" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fx = json.loads((HERE / "fixtures.json").read_text())
    names = [f["name"] for f in fx["fixtures"]] if args.only is None else args.only
    oracle_meta = json.loads((WORK / "oracle" / "oracle.json").read_text())
    print(f"[gate] {slow.name} + {fast.name} + {codec.name} on {args.unit} -> {out_dir}", flush=True)

    base = [sys.executable, str(Path(__file__).resolve()), "--worker", "--mode", args.mode, "--cl", str(args.cl),
            "--window", str(args.window), "--codec-frames", str(args.codec_frames), "--exports", args.exports,
            "--unit", args.unit, "--tag", tag] + (["--fast-mode", args.fast_mode] if args.fast_mode else [])
    results = {}
    for name in names:
        if not (out_dir / f"{name}.json").exists():
            subprocess.run(base + ["--only", name], check=True)
        results[name] = json.loads((out_dir / f"{name}.json").read_text())
    load_s = float(np.median([r["load_s"] for r in results.values()]))

    asr_rows, spk_rows = [], []
    for name in names:
        info = oracle_meta["fixtures"][name]
        wav_path, orc_wav = out_dir / f"{name}.port.wav", WORK / "oracle" / f"{name}.wav"
        asr_rows.append({"id": name, "arm": "port", "wav": str(wav_path), "lang": info["lang"], "text": info["text"]})
        asr_rows.append({"id": name, "arm": "oracle", "wav": str(orc_wav), "lang": info["lang"], "text": info["text"]})
        if info.get("reference"):
            ref = info["reference"]["path"]
            spk_rows.append({"id": name, "arm": "port", "wav": str(wav_path), "reference": ref})
            spk_rows.append({"id": name, "arm": "oracle", "wav": str(orc_wav), "reference": ref})

    summary = {
        "fixtures": len(results),
        "slow_steps": sum(r["teacher_forced"]["slow_steps"] for r in results.values()),
        "slow_sample_eq": sum(r["teacher_forced"]["slow_sample_eq"] for r in results.values()),
        "slow_argmax_eq": sum(r["teacher_forced"]["slow_argmax_eq"] for r in results.values()),
        "slow_cos_min": min(r["teacher_forced"]["slow_cos_min"] for r in results.values()),
        "slow_miss_margins": sorted(m for r in results.values() for m in r["teacher_forced"]["slow_sample_miss_margins"]),
        "fast_logits": sum(r["teacher_forced"]["fast_logits"] for r in results.values()),
        "fast_sample_eq": sum(r["teacher_forced"]["fast_sample_eq"] for r in results.values()),
        "fast_argmax_eq": sum(r["teacher_forced"]["fast_argmax_eq"] for r in results.values()),
        "fast_cos_min": min(r["teacher_forced"]["fast_cos_min"] for r in results.values()),
        "fast_miss_margins_max": max([m for r in results.values() for m in r["teacher_forced"]["fast_sample_miss_margins"]] or [0.0]),
        "codec_cos_min": min(r["codec"]["cos"] for r in results.values() if "cos" in r["codec"]),
        "codec_logmel_min": min(r["codec"]["logmel_cos"] for r in results.values() if "logmel_cos" in r["codec"]),
        "free_run_identical": sum(int(r["free_run"]["identical"]) for r in results.values()),
        "free_run_eos": sum(int(r["free_run"]["ended_with_eos"]) for r in results.values()),
        "free_run_frames": sum(r["free_run"]["frames"] for r in results.values()),
        "oracle_frames": sum(r["free_run"]["oracle_frames"] for r in results.values()),
        "slow_decode_ms_median": float(np.median([r["teacher_forced"]["slow_decode_ms_median"] for r in results.values()])),
        "fast_frame_ms_median": float(np.median([r["teacher_forced"]["fast_frame_ms_median"] for r in results.values()])),
        "load_s_median": load_s,
    }
    if not args.skip_asr:
        (out_dir / "asr_manifest.json").write_text(json.dumps(asr_rows, ensure_ascii=False, indent=1))
        print("[asr] transcribing with the fp32 Fun-ASR oracle ...", flush=True)
        subprocess.run([str(ORACLE_VENV), str(HERE / "asr_judge.py"), "--manifest", str(out_dir / "asr_manifest.json"),
                        "--out", str(out_dir / "asr.json")], check=True)
        summary["asr"] = json.loads((out_dir / "asr.json").read_text())["summary"]
        if spk_rows:
            (out_dir / "spk_manifest.json").write_text(json.dumps(spk_rows, ensure_ascii=False, indent=1))
            subprocess.run([sys.executable, str(HERE / "speaker_sim.py"), "--manifest", str(out_dir / "spk_manifest.json"),
                            "--out", str(out_dir / "speaker.json")], check=True)
            summary["speaker"] = json.loads((out_dir / "speaker.json").read_text())["summary"]
    (out_dir / "gate.json").write_text(json.dumps({"tag": tag, "bundles": [slow.name, fast.name, codec.name], "unit": args.unit,
                                                    "summary": summary, "results": results}, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"-> {out_dir / 'gate.json'}")


if __name__ == "__main__":
    main()
