#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""The engine gate of the one-call-per-frame asset (audio8_dualar_*) + the codec decoder, on every fixture.

Per fixture, on the GPU:
  1. teacher-forced  the oracle's tokens are fed (`use_forced` = 1); per frame the slow logits (4097), the slow hidden and
                     the nine fast-logits vectors are compared with the oracle's (cosine, argmax), and the in-graph
                     sampler's own draws from those logits with the oracle's noise are compared with the oracle's tokens
  2. codec           the oracle's codes through the codec graph: wav cosine / max abs / log-mel cosine vs the oracle wav
  3. free run        the port's own loop (the oracle's noise while it lasts, then a seeded stream): frames identical to
                     the oracle until the first knife-edge flip; the wav written and transcribed (asr_judge.py, Fun-ASR
                     fp32, WER/CER vs the fixture text, the oracle's own audio scored alike); the clone fixtures scored
                     for speaker similarity to the reference (speaker_sim.py)

Fixtures run in child interpreters (the Python bindings leak an IOSurface per call).

    ~/code/coreai/coreai-models/.venv/bin/python conversion/audio8_tts/gate_audio8_frame.py --mode int8
-> `$ZOO_WORK_ROOT/_audio8_tts/gate/<tag>/gate.json` (+ wavs), and a one-screen summary.
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
from export_audio8_frame import EngineFramePorts, free_run_frames, summarize_tf, teacher_forced_frames  # noqa: E402
from gate_audio8 import decode_in_buckets  # noqa: E402
from replay import codec_gate  # noqa: E402

WORK = work_path("_audio8_tts")
ORACLE_VENV = work_path("_funasr_nano", "venv-oracle", "bin", "python")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="int8")
    ap.add_argument("--cl", type=int, default=M.MAX_SEQ_LEN)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--codec-frames", type=int, default=160)
    ap.add_argument("--exports", default=str(WORK / "exports"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--unit", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--worker", action="store_true")
    return ap.parse_args()


def bundle_paths(args):
    exports = Path(args.exports)
    dualar = exports / f"audio8_dualar_{args.mode}_cl{args.cl}_w{args.window}.aimodel"
    codec = exports / f"audio8_codec_decoder_fp16_t{args.codec_frames}.aimodel"
    for p in (dualar, codec):
        assert p.exists(), p
    tag = args.tag or f"dualar{args.mode}_cl{args.cl}_w{args.window}_t{args.codec_frames}_{args.unit}"
    return dualar, codec, tag


def worker(args):
    dualar, codec, tag = bundle_paths(args)
    out_dir = WORK / "gate" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ports = EngineFramePorts(dualar, args.cl, unit=args.unit)
    cports = EnginePorts(None, None, codec, args.cl, unit=args.unit)
    load_s = time.time() - t0
    for name in args.only:
        orc = np.load(WORK / "oracle" / f"{name}.npz")
        t1 = time.time()
        tf = summarize_tf(teacher_forced_frames(ports, orc, window=args.window))
        cg = codec_gate(cports, orc, bucket=args.codec_frames) if orc["codes"].shape[1] <= args.codec_frames else {"note": "longer than the bucket"}
        fr = free_run_frames(ports, orc, window=args.window)
        wav = decode_in_buckets(cports, fr["codes"], args.codec_frames) if fr["frames"] else np.zeros(0, np.float32)
        sf.write(out_dir / f"{name}.port.wav", wav, M.CODEC_SR, subtype="FLOAT")
        np.save(out_dir / f"{name}.port_codes.npy", fr["codes"])
        fr_out = {k: v for k, v in fr.items() if k != "codes"}
        fr_out["seconds"] = wav.size / M.CODEC_SR
        res = {"teacher_forced": tf, "codec": cg, "free_run": fr_out, "load_s": load_s, "wall_s": round(time.time() - t1, 1)}
        print(f"[{name}] slow {tf['slow_sample_eq']}/{tf['steps']} (argmax {tf['slow_argmax_eq']}, cos min {tf['slow_cos_min']:.5f}) | "
              f"fast {tf['fast_sample_eq']}/{tf['fast_n']} (argmax {tf['fast_argmax_eq']}, cos min {tf['fast_cos_min']:.5f}) | "
              f"codec cos {cg.get('cos', float('nan')):.5f} logmel {cg.get('logmel_cos', float('nan')):.4f} | "
              f"free run {fr['frames']} frames (oracle {fr['oracle_frames']}), identical prefix {fr['identical_frames_prefix']}, eos {fr['ended_with_eos']}, "
              f"{fr['ms_per_frame']:.1f} ms/frame | frame {tf['frame_ms_median']:.1f} ms, prefill {tf['prefill_ms']:.0f} ms | {res['wall_s']}s", flush=True)
        (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1))


def main():
    args = parse_args()
    if args.worker:
        worker(args)
        return
    dualar, codec, tag = bundle_paths(args)
    out_dir = WORK / "gate" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fx = json.loads((HERE / "fixtures.json").read_text())
    names = [f["name"] for f in fx["fixtures"]] if args.only is None else args.only
    oracle_meta = json.loads((WORK / "oracle" / "oracle.json").read_text())
    print(f"[gate] {dualar.name} + {codec.name} on {args.unit} -> {out_dir}", flush=True)
    base = [sys.executable, str(Path(__file__).resolve()), "--worker", "--mode", args.mode, "--cl", str(args.cl), "--window", str(args.window),
            "--codec-frames", str(args.codec_frames), "--exports", args.exports, "--unit", args.unit, "--tag", tag]
    results = {}
    for name in names:
        if not (out_dir / f"{name}.json").exists():
            subprocess.run(base + ["--only", name], check=True)
        results[name] = json.loads((out_dir / f"{name}.json").read_text())

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
    R = list(results.values())
    summary = {
        "fixtures": len(R),
        "slow_steps": sum(r["teacher_forced"]["steps"] for r in R),
        "slow_sample_eq": sum(r["teacher_forced"]["slow_sample_eq"] for r in R),
        "slow_argmax_eq": sum(r["teacher_forced"]["slow_argmax_eq"] for r in R),
        "slow_cos_min": min(r["teacher_forced"]["slow_cos_min"] for r in R),
        "hidden_cos_min": min(r["teacher_forced"]["hidden_cos_min"] for r in R),
        "fast_n": sum(r["teacher_forced"]["fast_n"] for r in R),
        "fast_sample_eq": sum(r["teacher_forced"]["fast_sample_eq"] for r in R),
        "fast_argmax_eq": sum(r["teacher_forced"]["fast_argmax_eq"] for r in R),
        "fast_cos_min": min(r["teacher_forced"]["fast_cos_min"] for r in R),
        "codec_cos_min": min(r["codec"]["cos"] for r in R if "cos" in r["codec"]),
        "codec_logmel_min": min(r["codec"]["logmel_cos"] for r in R if "logmel_cos" in r["codec"]),
        "free_run_eos": sum(int(r["free_run"]["ended_with_eos"]) for r in R),
        "free_run_identical": sum(int(r["free_run"]["identical"]) for r in R),
        "free_run_frames": sum(r["free_run"]["frames"] for r in R),
        "oracle_frames": sum(r["free_run"]["oracle_frames"] for r in R),
        "frame_ms_median": float(np.median([r["teacher_forced"]["frame_ms_median"] for r in R])),
        "free_run_ms_per_frame_median": float(np.median([r["free_run"]["ms_per_frame"] for r in R])),
        "prefill_ms_median": float(np.median([r["teacher_forced"]["prefill_ms"] for r in R])),
        "load_s_median": float(np.median([r["load_s"] for r in R])),
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
    (out_dir / "gate.json").write_text(json.dumps({"tag": tag, "bundles": [dualar.name, codec.name], "unit": args.unit,
                                                    "summary": summary, "results": results}, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"-> {out_dir / 'gate.json'}")


if __name__ == "__main__":
    main()
