#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""E4: end-to-end on the Core AI engine (Mac GPU): wav -> text, every fixture clip, vs the fp32 oracle.

The host path the app will run, with the ``gate_static.py --unified`` driver shape from
``conversion/qwen3_asr``:

    wav -> frontend.fbank_lfr (NumPy) -> zero-pad [1, 500, 560] + mask -> encoder .aimodel
        -> audio_embeds[:N] -> prompt ids (prefix + V+slot + suffix)
        -> unified decoder ``prefill`` (writes KV [0, Sp), last-token logits)
        -> ``decode`` greedy loop (``pos`` = absolute position) until EOS or 512 tokens
        -> tokenizer.decode(skip_special_tokens) -> funasr's whitespace clean-up

One arm = one encoder bundle x one decoder bundle. Per clip: gen_ids, text, token-exact vs the
oracle, first divergence step with the oracle's top-2 margin there (``logs/r2_margins.json``, floor
0.1 = knife-edge), cap hit, encoder / prefill (first and repeated call) / decode ms. The KV buffers
are created once per process and reused: every slot a query can see (``j <= pos``) is rewritten for
the current clip before it is read, so stale slots are masked out (``--fresh-state`` allocates new
ones per clip, for an A/B of that claim).

The Python runtime leaks per call, so the driver runs the clips in chunks, one subprocess each,
writing ``<work>/gate_e2e/<arm>/<clip>.json`` (a rerun resumes), then aggregates
``logs/r2_gate_e2e_<arm>.json``. Everything runs on the GPU explicitly and is timed contended.

    python gate_e2e.py --enc fp16 --dec int8hu            # one arm, all clips
    python gate_e2e.py --enc fp16 --dec int8hu --red      # red arm: zero audio_embeds on one clip
    python gate_e2e.py --enc fp16w32 --dec int8lin --only zh --hotwords 开放时间   # hotword prompt

``--hotwords`` builds funasr's hotword prompt (``prompt.user_text``) and compares against the oracle run
with the same hotwords (``oracle_hotwords/<clip>_hotwords.json``, from ``make_oracle.py --hotwords``);
those runs have no per-step margins. Results go to ``gate_e2e/hw_<arm>/`` and
``logs/r3_gate_e2e_hotwords_<arm>.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from _paths import work_path  # noqa: E402
from frontend import fake_token_len, fbank_lfr  # noqa: E402
from prompt import EOS_IDS, MAX_NEW_TOKENS, V, build_prompt_ids  # noqa: E402

WORK = work_path("_funasr_nano")
QWEN_DIR = WORK / "official" / "Qwen3-0.6B"
L_MAX = 500
MARGIN_FLOOR = 0.1
N_LAYERS, N_KV, HEAD_DIM, CACHE_LEN = 28, 8, 128, 1024
ENC = {"fp16": ("funasr_nano_audio_encoder_fp16_l500.aimodel", np.float16),
       "fp32": ("funasr_nano_audio_encoder_fp32_l500.aimodel", np.float32),
       # fp16 weights, fp32 activations (the ship candidate: fp32-grade parity at fp16 size)
       "fp16w32": ("funasr_nano_audio_encoder_fp16w32_l500.aimodel", np.float32)}
FILTER = re.compile(r"Redirects are currently|scikit-learn version|coremltools|cpp extensions")


def clean(text: str) -> str:
    """``FunASRNano.inference_llm``'s post-processing of the decoded string."""
    return re.sub(r"\s+", " ", text.replace("/sil", " "))


def arm_name(enc: str, dec: str) -> str:
    return f"enc{enc[2:]}_{dec}"


def arm_id(args) -> str:
    """Result directory / JSON name: the arm, marked when the run is the red arm or the fresh-state A/B."""
    return (("red_" if args.red else "") + ("hw_" if args.hotwords else "") + arm_name(args.enc, args.dec)
            + ("_fresh" if args.fresh_state else ""))


def dec_bundle(dec: str) -> Path:
    name = f"funasr_nano_2512_{dec}_unified_cl{CACHE_LEN}"
    return WORK / "exports" / name / f"{name}.aimodel"


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def top2_gap(logits: np.ndarray) -> tuple[float, int, int]:
    x = logits.astype(np.float64)
    i2 = np.argpartition(-x, 2)[:2]
    i2 = i2[np.argsort(-x[i2])]
    p = np.exp(x[i2] - x.max()) / np.exp(x - x.max()).sum()
    return float(p[0] - p[1]), int(i2[0]), int(i2[1])


def row_cos_min(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)
    return float(cos.min())


async def worker(args) -> None:
    import soundfile as sf
    from transformers import AutoTokenizer

    import coreai.runtime as rt

    arm = arm_id(args)
    out_dir = WORK / "gate_e2e" / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(QWEN_DIR))
    meta = {c["name"]: c for c in json.loads((WORK / "fixtures" / "meta.json").read_text())["clips"]}
    oracle = {c["name"]: c for c in json.loads((WORK / "oracle" / "oracle.json").read_text())["clips"]}
    margins = json.loads((WORK / "logs" / "r2_margins.json").read_text())["clips"]
    variants = {}
    for p in sorted((WORK / "oracle_hotwords").glob("*.json")) if args.hotwords else ():
        v = json.loads(p.read_text())
        if v["hotwords"] == list(args.hotwords) and v["language"] is None and v["itn"]:
            variants[v["name"]] = v
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())

    enc_file, enc_dtype = ENC[args.enc]
    t0 = time.perf_counter()
    enc_model = await rt.AIModel.load(WORK / "exports" / enc_file, gpu)
    efn = enc_model.load_function("main")
    dec_model = await rt.AIModel.load(dec_bundle(args.dec), gpu)
    pfn, dfn = dec_model.load_function("prefill"), dec_model.load_function("decode")
    load_s = time.perf_counter() - t0

    def new_state() -> dict:
        return {"keyCache": rt.NDArray(np.zeros((N_LAYERS, 1, N_KV, CACHE_LEN, HEAD_DIM), np.float16)),
                "valueCache": rt.NDArray(np.zeros((N_LAYERS, 1, N_KV, CACHE_LEN, HEAD_DIM), np.float16))}

    state = new_state()

    async def call(fn, inputs: dict, **kw) -> np.ndarray:
        res = await asyncio.wait_for(fn(inputs={k: rt.NDArray(np.ascontiguousarray(v)) for k, v in inputs.items()},
                                        **kw), timeout=600)
        return res

    for name in args.only:
        path = out_dir / f"{name}.json"
        if path.exists() and not args.force:
            continue
        if args.fresh_state:
            state = new_state()
        o = np.load(WORK / "oracle" / f"{name}.npz")
        if args.hotwords:
            v = variants[name]
            golden, oracle_text, om, runner = v["gen_ids"], v["text"], None, None
        else:
            golden, oracle_text = o["gen_ids"].astype(np.int64).tolist(), oracle[name]["text"]
            om, runner = margins[name]["margins_pos0"], margins[name]["runner_up_pos0"]
        wav, sr = sf.read(str(WORK / "fixtures" / meta[name]["path"]), dtype="float32")
        assert sr == 16000
        feats = fbank_lfr(wav)
        L, N = feats.shape[0], fake_token_len(feats.shape[0])
        assert N == int(o["fake_token_len"]) and L <= L_MAX
        x = np.zeros((1, L_MAX, feats.shape[1]), enc_dtype)
        x[0, :L] = feats
        mask = np.zeros((1, L_MAX), enc_dtype)
        mask[0, :L] = 1

        t1 = time.perf_counter()
        eres = await call(efn, {"feats": x, "mask": mask})
        enc_ms = (time.perf_counter() - t1) * 1000
        emb = eres["audio_embeds"].numpy()[:N]
        cos_min = row_cos_min(emb, o["adaptor_out"][:N])
        audio = np.zeros_like(emb, dtype=np.float16) if args.red else emb.astype(np.float16)

        ids, _ = build_prompt_ids(tok, N, hotwords=args.hotwords or ())
        if args.hotwords:   # the oracle's source_ids with its placeholders written as V + slot
            src, beg = list(v["source_ids"]), v["fbank_beg"]
            assert v["N"] == N and ids == src[:beg] + [V + s for s in range(N)] + src[beg + N:], name
        prompt = np.array([ids], np.int32)
        t1 = time.perf_counter()
        pres = await call(pfn, {"input_ids": prompt, "audio_embeds": audio}, state=state)
        pre_first_ms = (time.perf_counter() - t1) * 1000
        logits = pres["logits"].numpy().astype(np.float32).reshape(-1)
        t1 = time.perf_counter()
        pres2 = await call(pfn, {"input_ids": prompt, "audio_embeds": audio}, state=state)   # same K/V rewritten
        pre_warm_ms = (time.perf_counter() - t1) * 1000
        repeat_equal = bool(np.array_equal(pres2["logits"].numpy().astype(np.float32).reshape(-1), logits))
        if not np.isfinite(logits).all() or not logits.any():
            raise RuntimeError(f"{name}: prefill logits are NaN/inf or all zero — engine output is broken")

        gen, step_ms, gaps = [], [], []
        pos = len(ids)
        while True:
            gap, top1, _ = top2_gap(logits)
            nxt = int(np.argmax(logits))
            gen.append(nxt)
            gaps.append(round(gap, 6))
            if nxt in EOS_IDS or len(gen) >= MAX_NEW_TOKENS:
                break
            t1 = time.perf_counter()
            dres = await call(dfn, {"input_ids": np.array([[nxt]], np.int32), "pos": np.array([pos], np.int32)},
                              state=state)
            step_ms.append((time.perf_counter() - t1) * 1000)
            logits = dres["logits"].numpy().astype(np.float32).reshape(-1)
            if not np.isfinite(logits).all():
                raise RuntimeError(f"{name}: decode logits NaN/inf at step {len(gen)}")
            pos += 1

        text = clean(tok.decode(gen, skip_special_tokens=True))
        div = first_divergence(gen, golden)
        row = {"name": name, "arm": arm, "N": N, "L": L, "Sp": len(ids), "gen_ids": gen, "text": text,
               "oracle_text": oracle_text, "exact": gen == golden, "text_equal": text == oracle_text,
               "hotwords": list(args.hotwords or []),
               "first_divergence": div, "hit_cap": len(gen) >= MAX_NEW_TOKENS and gen[-1] not in EOS_IDS,
               "gen_len": len(gen), "oracle_len": len(golden), "encoder_row_cos_min": cos_min,
               "enc_ms": round(enc_ms, 2), "prefill_ms_first": round(pre_first_ms, 2),
               "prefill_ms_warm": round(pre_warm_ms, 2), "prefill_repeat_equal": repeat_equal,
               "decode_ms_median": round(statistics.median(step_ms), 2) if step_ms else None,
               "decode_steps": len(step_ms), "port_top2_gaps": gaps, "fresh_state": bool(args.fresh_state),
               "load_s_this_process": round(load_s, 2)}
        if div is not None:
            k = min(div, len(golden) - 1)
            row.update({"oracle_margin_at_div": om[k] if om else None,
                        "knife_edge": om[k] < MARGIN_FLOOR if om else None,
                        "ours_at_div": gen[div] if div < len(gen) else None, "oracle_at_div": golden[k],
                        "oracle_runner_up_at_div": runner[k] if runner else None,
                        "port_gap_at_div": gaps[div] if div < len(gaps) else None,
                        "ours_tail": tok.decode(gen[div:div + 6]), "oracle_tail": tok.decode(golden[k:k + 6])})
        path.write_text(json.dumps(row, ensure_ascii=False, indent=1))
        flag = "OK  " if row["exact"] else ("KNIFE" if row.get("knife_edge") else "DIFF")
        print(f"[{arm}] {flag} {name}: N={N} gen {len(gen)}/{len(golden)} enc {enc_ms:.0f} ms prefill "
              f"{pre_first_ms:.0f}/{pre_warm_ms:.0f} ms decode {row['decode_ms_median']} ms/tok cos_min {cos_min:.5f}"
              + (f" div@{div}" if div is not None else "")
              + (f" margin {row['oracle_margin_at_div']:.4f}" if row.get("oracle_margin_at_div") is not None else "")
              + (f"  {text!r}" if args.red or args.hotwords else ""), flush=True)


def aggregate(arm: str, names: list[str]) -> dict:
    rows = [json.loads((WORK / "gate_e2e" / arm / f"{n}.json").read_text()) for n in names]
    exact = [r for r in rows if r["exact"]]
    knife = [r for r in rows if not r["exact"] and r.get("knife_edge")]
    bad = [r for r in rows if not r["exact"] and not r.get("knife_edge")]
    med = lambda xs: round(statistics.median(xs), 2) if xs else None  # noqa: E731
    summary = {
        "arm": arm, "clips": len(rows), "exact": len(exact), "exact_or_knife_edge": len(exact) + len(knife),
        "knife_edge": len(knife), "mismatch": len(bad), "hit_cap": sum(r["hit_cap"] for r in rows),
        "text_equal": sum(r["text_equal"] for r in rows),
        "prefill_repeat_equal": sum(r["prefill_repeat_equal"] for r in rows),
        "encoder_row_cos_min": min(r["encoder_row_cos_min"] for r in rows),
        "enc_ms_median": med([r["enc_ms"] for r in rows]),
        "prefill_ms_warm_median": med([r["prefill_ms_warm"] for r in rows]),
        "prefill_ms_first_median": med([r["prefill_ms_first"] for r in rows]),
        "decode_ms_per_token_median": med([r["decode_ms_median"] for r in rows if r["decode_ms_median"]]),
        "timing_note": "Mac GPU shared with other sessions (contended); prefill first = first call of a "
                       "(Sp, N) shape in the process, warm = the identical call repeated",
        "non_exact": [{k: r.get(k) for k in ("name", "first_divergence", "oracle_margin_at_div", "knife_edge",
                                              "port_gap_at_div", "ours_at_div", "oracle_at_div", "ours_tail",
                                              "oracle_tail", "hit_cap")} for r in knife + bad],
    }
    (WORK / "logs" / f"r2_gate_e2e_{arm}.json").write_text(
        json.dumps({"summary": summary, "clips": rows}, ensure_ascii=False, indent=1))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--enc", choices=sorted(ENC), default="fp16")
    ap.add_argument("--dec", choices=("int8lin", "int8hu", "fp16"), default="int8lin")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--chunk", type=int, default=40, help="clips per worker process")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--red", action="store_true", help="zero audio_embeds on one clip (the gate must go red)")
    ap.add_argument("--fresh-state", action="store_true", help="new KV buffers for every clip")
    ap.add_argument("--force", action="store_true", help="recompute clips that already have a result")
    ap.add_argument("--hotwords", nargs="+", help="funasr hotword prompt; compares against oracle_hotwords/")
    args = ap.parse_args()

    if args.worker:
        asyncio.run(worker(args))
        return
    names = [c["name"] for c in json.loads((WORK / "oracle" / "oracle.json").read_text())["clips"]]
    if args.red:
        args.only = args.only or ["en"]
    names = [n for n in names if not args.only or n in args.only]
    arm = arm_id(args)
    for i in range(0, len(names), args.chunk):
        cmd = [sys.executable, __file__, "--worker", "--enc", args.enc, "--dec", args.dec,
               "--only", *names[i:i + args.chunk]]
        cmd += ["--red"] * args.red + ["--fresh-state"] * args.fresh_state + ["--force"] * args.force
        cmd += ["--hotwords", *args.hotwords] if args.hotwords else []
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            if not FILTER.search(line):
                print(line, end="", flush=True)
        if proc.wait() != 0:
            raise SystemExit(f"worker for clips {i}..{i + args.chunk - 1} exited {proc.returncode}")
    if args.hotwords:
        rows = [json.loads(p.read_text()) for p in sorted((WORK / "gate_e2e" / arm).glob("*.json"))]
        (WORK / "logs" / f"r3_gate_e2e_hotwords_{arm_name(args.enc, args.dec)}.json").write_text(
            json.dumps({"clips": rows}, ensure_ascii=False, indent=1))
        for r in rows:
            print(f"  {r['name']} hotwords {r['hotwords']}: exact {r['exact']} text_equal {r['text_equal']} "
                  f"{r['text']!r} (oracle {r['oracle_text']!r})", flush=True)
        return
    summary = aggregate(arm, names)
    print(json.dumps({k: v for k, v in summary.items() if k != "non_exact"}, ensure_ascii=False, indent=1))
    for r in summary["non_exact"]:
        print(f"  non-exact {r['name']}: step {r['first_divergence']} oracle margin {r['oracle_margin_at_div']:.4f} "
              f"({'knife-edge' if r['knife_edge'] else 'ABOVE floor'}) ours {r['ours_tail']!r} vs oracle "
              f"{r['oracle_tail']!r}", flush=True)


if __name__ == "__main__":
    main()
