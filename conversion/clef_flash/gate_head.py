#!/usr/bin/env python3
"""Gates of the clef-flash head graph: H0 (the matrix form, eager), H1 (the graph alone), H2 (decoder hidden -> probabilities).

H0  `ClefHeadGraph` in eager fp32 (CPU) vs the author's `JointSchemaHead` (fp32, CPU, same process,
    parity_decoder_torch.AuthorHead) on every oracle run's fp32 hidden: once at the exact sizes
    (T, Q, O) and once padded the way a host pads (T to a multiple of 64, Q to 16, O to 128). Both
    read lexical from the same fp32 table (the bf16 values), so the difference is the matrix form
    alone. Bar (fixed before running): max |d logit| <= 1e-4 on every run.
H1  the exported head graph (AOT `.aimodelc`, Python runtime, `SpecializationOptions.default()`, no
    JIT) on every oracle run's fp32 hidden, lexical from the shipped fp16 table
    (`exports/host/lm_head_fp16.bin`), vs the oracle's logits and probabilities. Bar: max |dp| <= 1e-3
    (fp16 weights, fp32 compute; expected <= 1e-4), argmax equal on every question.
H2  end to end in Python: the decoder bundle's fp16 hidden rows of a readout-gate transcript (its shard
    npz, cast to fp32; the decoder is not run again), the host's span matrices from the oracle's spans,
    lexical from the fp16 table, the head graph, the per-question softmax -> vs the oracle's
    probabilities. Bar = the decoder gate's: argmax equal on every question whose oracle top-2 margin
    is above 0.02 (near-ties listed apart), max |dp| <= 0.02, mean of run means |dp| <= 0.002. Side by
    side: the same hidden rows read through the author's fp32 head (the readout transcript's values),
    so what the head graph adds is visible. `--red` adds a separate process with three arms on a few
    runs: the question / option span rows shifted one token right, lexical zeroed, the option ->
    question membership rotated by one option; each must move the answers.

Every Core AI call runs in a worker process (at most 40 runs + a re-run of its first run, which must
reproduce its logits bit for bit: the state-free graph's determinism / reset check). Times are
contended reference values (the GPU is shared; no _GPU_LOCK held): every run is called twice in a row
and both calls are kept (the first call of a new input shape includes any re-specialization).

    cd conversion/clef_flash
    PY=<coreai-models venv>/bin/python
    $PY gate_head.py h0 --out $ZOO_WORK_ROOT/_clefflash/results/head_h0.json
    $PY gate_head.py h1 --head <head bundle dir> --out .../results/head_h1.json
    $PY gate_head.py h2 --head <head bundle dir> --readout .../results/readout_fp16_pf64_full.json --red \\
        --out .../results/head_h2_fixture.json
    $PY gate_head.py h2 --head <head bundle dir> --oracle .../oracle/heldout \\
        --readout .../results/readout_heldout_fp16_pf64.json --out .../results/head_h2_heldout.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import work_path  # noqa: E402

LANE = work_path("_clefflash")
os.environ.setdefault("HF_HOME", str(LANE / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ORACLE = LANE / "oracle"
TABLE = LANE / "exports" / "host" / "lm_head_fp16.bin"
VOCAB, HIDDEN = 248320, 4096
T_MULTIPLE = 64
PAD_Q, PAD_O = 16, 128
BUCKETS = (512, 1024, 2048, 4096)
H0_BAR = {"max_abs_dlogit": 1e-4}
H1_BAR = {"max_abs_dp": 1e-3, "argmax": "every question"}
H2_BAR = {"max_abs_dp": 0.02, "mean_of_run_mean_abs_dp": 0.002, "near_tie_top2_margin": 0.02}
RUNS_PER_PROCESS = 40
RED_ARMS = ("span_shift_right_1", "lexical_zero", "member_roll_1")
RED_RUNS = (("own_t01", "text"), ("own_j08", "text"), ("img_01", "g256"), ("img_07", "g448"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_oracle(oracle: Path) -> tuple[dict, list[dict]]:
    doc = json.loads((oracle / "records_oracle.json").read_text())
    assert doc["complete"], f"{oracle}: oracle incomplete"
    return doc, doc["rows"]


def oracle_npz(oracle: Path, row: dict):
    return np.load(oracle / "npz" / f"{row['id']}__{row['arm']}.npz")


def table_memmap(path: Path = TABLE):
    return np.memmap(path, dtype="<f2", mode="r", shape=(VOCAB, HIDDEN))


def padded_sizes(row: dict, mode: str, q_min: int = 1) -> tuple[int | None, int | None, int | None]:
    """(t_pad, q_pad, o_pad) of a run: 'exact' = none; 'host' = T to a multiple of 64, Q to at least
    `q_min` (the dynamic export's smallest Q); 'bucket' = T to the smallest bucket that holds it, Q 16, O 128."""
    from clef_head import ceil_to
    T = row["tokens"]
    Q = len(row["questions"])
    if mode == "exact":
        return None, None, None
    if mode == "host":
        return ceil_to(T, T_MULTIPLE), max(Q, q_min), None
    if mode == "bucket":
        return next(b for b in BUCKETS if b >= T), PAD_Q, PAD_O
    raise ValueError(mode)


def probs_and_scores(row: dict, logits: np.ndarray, layout) -> dict:
    """Per-question fp32 softmax of the graph's logits vs the oracle's logits / probabilities."""
    from clef_head import question_probs
    probs = question_probs(logits, layout)
    qs, deltas = [], []
    for q, (a, b), p in zip(row["questions"], layout, probs):
        po = np.asarray(q["probs"], np.float64)
        lo = np.asarray(q["logits"], np.float64)
        dp = np.abs(p.astype(np.float64) - po)
        deltas.append(dp)
        qs.append({"question_id": q["question_id"], "type": q["type"], "n_options": int(b - a),
                   "argmax": int(np.argmax(p)), "argmax_oracle": q["argmax_index"],
                   "argmax_equal": int(np.argmax(p)) == q["argmax_index"],
                   "max_abs_dp": float(dp.max()), "mean_abs_dp": float(dp.mean()),
                   "max_abs_dlogit": float(np.abs(logits[a:b].astype(np.float64) - lo).max()),
                   "oracle_top2_margin": q["top2_margin"], "near_tie": bool(q["near_tie"]),
                   "probs": [float(v) for v in p], "probs_oracle": [float(v) for v in po]})
    flat = np.concatenate(deltas)
    return {"questions": qs, "argmax_all_equal": all(q["argmax_equal"] for q in qs),
            "argmax_equal_non_near_tie": all(q["argmax_equal"] for q in qs if not q["near_tie"]),
            "max_abs_dp": float(flat.max()), "mean_abs_dp": float(flat.mean()),
            "max_abs_dlogit": max(q["max_abs_dlogit"] for q in qs)}


def summarize(runs: list[dict]) -> dict:
    qs = [q for r in runs for q in r["questions"]]
    far = [q for q in qs if not q["near_tie"]]
    near = [q for q in qs if q["near_tie"]]
    worst = max(runs, key=lambda r: r["max_abs_dp"]) if runs else None
    return {"runs": len(runs), "questions": len(qs), "argmax_equal": sum(q["argmax_equal"] for q in qs),
            "questions_non_near_tie": len(far), "argmax_equal_non_near_tie": sum(q["argmax_equal"] for q in far),
            "near_tie_questions": len(near), "argmax_equal_near_tie": sum(q["argmax_equal"] for q in near),
            "max_abs_dp": max((q["max_abs_dp"] for q in qs), default=None),
            "mean_of_run_mean_abs_dp": float(np.mean([r["mean_abs_dp"] for r in runs])) if runs else None,
            "max_abs_dlogit": max((q["max_abs_dlogit"] for q in qs), default=None),
            "worst_run": None if worst is None else f"{worst['id']}/{worst['arm']}"}


def env_record() -> dict:
    import importlib.metadata as md
    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    for p in ("coreai-core", "coreai-torch", "coreai-models", "torch", "safetensors"):
        try:
            v[p] = md.version(p)
        except Exception as e:  # noqa: BLE001
            v[p] = repr(e)
    return {"versions": v, "platform": platform.platform(),
            "macos_build": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
            "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()}


# --------------------------------------------------------------------------- H0
def h0(args) -> int:
    import torch
    from clef_head import ClefHeadGraph, head_inputs
    from parity_decoder_torch import AuthorHead

    torch.set_num_threads(args.threads)
    oracle = Path(args.oracle)
    doc, rows = load_oracle(oracle)
    t0 = time.monotonic()
    author = AuthorHead()
    graph = ClefHeadGraph(author.head).eval()               # the same module object: the same weights
    table32 = author.lm_head_w.numpy()                      # [V, 4096] fp32 = the bf16 values
    load_s = time.monotonic() - t0
    runs = []
    for k, row in enumerate(rows):
        z = oracle_npz(oracle, row)
        ids = [int(x) for x in z["input_ids"]]
        assert ids == row["ids"]
        hid = np.asarray(z["last_hidden"], np.float32)
        ref = author.logits(torch.from_numpy(hid), ids, row)                # the author's head, this process
        ref_flat = torch.cat(ref).double().numpy()
        oracle_flat = np.concatenate([np.asarray(q["logits"], np.float64) for q in row["questions"]])
        rec = {"id": row["id"], "arm": row["arm"], "source": row["source"], "tokens": row["tokens"],
               "questions": len(row["questions"]), "options": int(ref_flat.size),
               "author_rerun_vs_oracle_max_abs_dlogit": float(np.abs(ref_flat - oracle_flat).max())}
        for mode in ("exact", "bucket"):
            tp, qp, op = padded_sizes(row, mode)
            arrays, layout = head_inputs(hid, ids, row["questions"], table32, t_pad=tp, q_pad=qp, o_pad=op)
            with torch.inference_mode():
                got = graph(**{n: torch.from_numpy(a) for n, a in arrays.items()}).double().numpy()
            O = ref_flat.size
            pads = got[O:]
            pr = np.concatenate([np.exp(got[a:b] - got[a:b].max()) / np.exp(got[a:b] - got[a:b].max()).sum()
                                 for a, b in layout])
            prr = np.concatenate([torch.softmax(t.float(), -1).double().numpy() for t in ref])
            rec[mode] = {"shape": {"T": int(arrays["hidden"].shape[0]), "Q": int(arrays["member"].shape[0]),
                                   "O": int(arrays["member"].shape[1])},
                         "max_abs_dlogit_vs_author": float(np.abs(got[:O] - ref_flat).max()),
                         "max_abs_dlogit_vs_oracle": float(np.abs(got[:O] - oracle_flat).max()),
                         "max_abs_dp_vs_author": float(np.abs(pr - prr).max()),
                         "padding_logits_zero": bool(np.all(pads == 0.0)),
                         "finite": bool(np.isfinite(got).all())}
        runs.append(rec)
        if k % 20 == 0:
            print(f"  {k:3d} {row['id']}/{row['arm']} T={row['tokens']} exact {rec['exact']['max_abs_dlogit_vs_author']:.2e} "
                  f"bucket {rec['bucket']['max_abs_dlogit_vs_author']:.2e}", flush=True)
    worst = {m: max(runs, key=lambda r: r[m]["max_abs_dlogit_vs_author"]) for m in ("exact", "bucket")}
    summary = {m: {"runs": len(runs),
                   "max_abs_dlogit_vs_author": worst[m][m]["max_abs_dlogit_vs_author"],
                   "worst_run": f"{worst[m]['id']}/{worst[m]['arm']}",
                   "max_abs_dlogit_vs_oracle": max(r[m]["max_abs_dlogit_vs_oracle"] for r in runs),
                   "max_abs_dp_vs_author": max(r[m]["max_abs_dp_vs_author"] for r in runs),
                   "padding_logits_zero_all": all(r[m]["padding_logits_zero"] for r in runs),
                   "finite_all": all(r[m]["finite"] for r in runs)} for m in ("exact", "bucket")}
    ok = all(summary[m]["max_abs_dlogit_vs_author"] <= H0_BAR["max_abs_dlogit"] and summary[m]["finite_all"]
             and summary[m]["padding_logits_zero_all"] for m in summary)
    rec = {"schema": "clef-flash-head-h0/1",
           "gate": "ClefHeadGraph eager fp32 (CPU) vs the author's JointSchemaHead fp32 (CPU, same process, same "
                   "module weights), on the oracle's fp32 hidden; lexical from the fp32 lm_head (bf16 values) for both",
           "bar": H0_BAR, "modes": {"exact": "T, Q, O as the record has them",
                                    "bucket": f"T padded to the smallest bucket {BUCKETS} that holds it (zero rows, "
                                              f"key_valid 0; T > 2048 runs the key-block attention), Q to {PAD_Q}, O to "
                                              f"{PAD_O} (zero rows, member 0)"},
           "oracle": {"path": str(oracle / "records_oracle.json"), "sha256": sha256_file(oracle / "records_oracle.json"),
                      "runs": len(rows)},
           "author_rerun_vs_oracle_max_abs_dlogit": max(r["author_rerun_vs_oracle_max_abs_dlogit"] for r in runs),
           "summary": summary, "result": "PASS" if ok else "FAIL",
           "script": {"gate_head.py": sha256_file(Path(__file__).resolve()),
                      "clef_head.py": sha256_file(HERE / "clef_head.py")},
           "environment": env_record(), "seconds": {"load": load_s, "total": time.monotonic() - t0},
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "runs": runs}
    Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(f"H0 {rec['result']}: {json.dumps(summary)}; author re-run vs oracle {rec['author_rerun_vs_oracle_max_abs_dlogit']:.2e}")
    print(f"wrote {args.out}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- Core AI worker
async def maybe(x):
    return await x if inspect.isawaitable(x) else x


def dsc(d) -> list:
    return [[int(x) for x in d.shape], str(d.dtype).split(".")[-1]]


def fn_desc(fn) -> dict:
    d = fn.desc
    return {"function": getattr(d, "name", None),
            "inputs": {n: dsc(d.input_descriptor(n)) for n in d.input_names},
            "outputs": {n: dsc(d.output_descriptor(n)) for n in d.output_names}}


def hidden_of(src: dict) -> np.ndarray:
    z = np.load(src["npz"])
    return np.asarray(z[src["key"]])


def red_variant(arrays: dict, variant: str, n_options: int) -> dict:
    a = dict(arrays)
    if variant == "span_shift_right_1":
        a["q_avg"] = np.roll(arrays["q_avg"], 1, axis=1)
        a["o_avg"] = np.roll(arrays["o_avg"], 1, axis=1)
    elif variant == "lexical_zero":
        a["lexical"] = np.zeros_like(arrays["lexical"])
    elif variant == "member_roll_1":
        m = arrays["member"].copy()
        m[:, :n_options] = np.roll(arrays["member"][:, :n_options], 1, axis=1)
        a["member"] = m
    elif variant != "base":
        raise ValueError(variant)
    return a


def worker(spec_path: Path) -> int:
    import coreai.runtime as rt
    from clef_head import head_inputs

    spec = json.loads(spec_path.read_text())
    oracle = Path(spec["oracle"])
    _, rows = load_oracle(oracle)
    orc = {(r["id"], r["arm"]): r for r in rows}
    table = table_memmap(Path(spec["table"]))
    out: dict = {"spec": spec, "pid": os.getpid(), "runs": [], "started": time.time()}
    store: dict[str, np.ndarray] = {}

    def nd(a):
        return rt.NDArray(np.ascontiguousarray(a))

    async def go() -> None:
        t0 = time.perf_counter()
        model = await maybe(rt.AIModel.load(spec["aimodelc"], rt.SpecializationOptions.default()))
        out["load_model_seconds"] = time.perf_counter() - t0
        names = list(getattr(model, "function_names", []) or [])
        out["function_names"] = names
        fns, descs = {}, {}
        for fname in spec["functions"]:
            t1 = time.perf_counter()
            fns[fname] = await maybe(model.load_function(fname))
            descs[fname] = fn_desc(fns[fname])
            out.setdefault("load_function_seconds", {})[fname] = time.perf_counter() - t1
        out["descriptors"] = descs

        def pick(T: int) -> tuple[str, int | None]:
            """The function for a run of T tokens and the T it takes (None = the run's own padded T)."""
            if spec["shape"] == "dynamic":
                return spec["functions"][0], None
            for fname, tb in spec["buckets"]:
                if T <= tb:
                    return fname, tb
            raise SystemExit(f"T {T} exceeds every bucket")

        async def one(run: list) -> tuple[dict, dict]:
            rid, arm, variant, src = run
            row = orc[(rid, arm)]
            hid = hidden_of(src) if src else np.asarray(np.load(oracle / "npz" / f"{rid}__{arm}.npz")["last_hidden"])
            fname, tb = pick(row["tokens"])
            tp, qp, op = padded_sizes(row, spec["pad"], (spec.get("dims") or {}).get("Q", [1])[0])
            if tb is not None:
                tp = tb
            t_host = time.perf_counter()
            arrays, layout = head_inputs(hid, row["ids"], row["questions"], table, t_pad=tp, q_pad=qp, o_pad=op)
            n_opt = layout[-1][1]
            arrays = red_variant(arrays, variant, n_opt)
            host_ms = (time.perf_counter() - t_host) * 1e3
            feed = {n: nd(a) for n, a in arrays.items()}
            ms, outs = [], []
            for _ in range(spec["calls_per_run"]):
                t1 = time.perf_counter()
                res = await maybe(fns[fname](inputs=feed))
                lg = np.asarray(res["logits"].numpy())
                ms.append((time.perf_counter() - t1) * 1e3)
                outs.append(lg)
            same = all(np.array_equal(outs[0], o) for o in outs[1:])
            lg = outs[0]
            if lg.shape != (arrays["member"].shape[1],) or lg.dtype != np.float32:
                raise SystemExit(f"{rid}/{arm}: logits {lg.shape} {lg.dtype}")
            rec = {"id": rid, "arm": arm, "variant": variant, "tokens": row["tokens"], "function": fname,
                   "shape": {"T": int(arrays["hidden"].shape[0]), "Q": int(arrays["member"].shape[0]),
                             "O": int(arrays["member"].shape[1])},
                   "options": n_opt, "layout": layout, "call_ms": ms, "host_ms": host_ms,
                   "calls_bit_equal": same, "finite": bool(np.isfinite(lg).all()),
                   "padding_logits_zero": bool(np.all(lg[n_opt:] == 0.0))}
            return rec, {"logits": lg[:n_opt].copy()}

        runs = [list(r) for r in spec["runs"]]
        first = None
        for i, run in enumerate(runs + [runs[0]]):
            rec, arrs = await one(run)
            if i < len(runs):
                rec["index"] = i
                out["runs"].append(rec)
                store[f"{i:02d}__logits"] = arrs["logits"]
                if i == 0:
                    first = arrs
                print(f"  [{os.getpid()}] {run[0]}/{run[1]}/{run[2]} T={rec['shape']['T']} Q={rec['shape']['Q']} "
                      f"O={rec['shape']['O']} {rec['function']}: {', '.join(f'{x:.1f}' for x in rec['call_ms'])} ms",
                      flush=True)
            else:
                same = bool(np.array_equal(first["logits"], arrs["logits"]))
                out["reset_check"] = {"run": run[:3], "bit_equal": same,
                                      "max_abs_diff": float(np.max(np.abs(first["logits"] - arrs["logits"])))}
                print(f"  [{os.getpid()}] re-run {run[0]}/{run[1]}: bit-equal {same}", flush=True)

    asyncio.run(go())
    out["finished"] = time.time()
    prefix = Path(spec["out"])
    np.savez(prefix.with_suffix(".npz"), **store)
    prefix.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


def split(runs: list, per: int = RUNS_PER_PROCESS) -> list[list]:
    n = math.ceil(len(runs) / per)
    size = math.ceil(len(runs) / n)
    return [runs[i:i + size] for i in range(0, len(runs), size)]


def run_workers(shards: list[dict]) -> list[dict]:
    got = []
    for sp in shards:
        spec_path = Path(sp["out"]).with_suffix(".spec.json")
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(sp, indent=1) + "\n")
        print(f"{Path(sp['out']).name}: {len(sp['runs'])} runs + re-run", flush=True)
        t0 = time.monotonic()
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--spec", str(spec_path)])
        js = Path(sp["out"]).with_suffix(".json")
        if proc.returncode != 0 or not js.exists():
            raise SystemExit(f"{Path(sp['out']).name}: worker failed (exit {proc.returncode})")
        rec = json.loads(js.read_text())
        rec["process_wall_seconds"] = time.monotonic() - t0
        z = np.load(Path(sp["out"]).with_suffix(".npz"))
        for r in rec["runs"]:
            r["_logits"] = z[f"{r['index']:02d}__logits"]
        got.append(rec)
    return got


def head_bundle(path: Path) -> dict:
    """The head bundle's record: its metadata, the .aimodel and the AOT asset next to it."""
    meta = json.loads((path / "metadata.json").read_text())
    aot = Path(meta["aot"]["aimodelc"]) if meta.get("aot") else None
    return {"dir": str(path), "metadata": meta, "metadata_sha256": sha256_file(path / "metadata.json"),
            "aimodelc": str(aot) if aot else None}


def resolve_readout(readout: Path) -> dict:
    """(id, arm) -> {npz, key, author_head: the transcript's per-run values}, for a plain or merged transcript."""
    d = json.loads(readout.read_text())
    out = {}
    for r in d["runs"]:
        if r.get("variant", "base") != "base":
            continue
        npz = r.get("npz") or next((p["npz"] for p in d["processes"] if p.get("shard") == r["shard"] and p.get("npz")), None)
        if not npz or not Path(npz).exists():
            raise SystemExit(f"{r['id']}/{r['arm']}: shard npz {npz} not found")
        out[(r["id"], r["arm"])] = {"npz": npz, "key": f"{r['npz_key']}__hidden",
                                    "author_head": {"max_abs_dp": r["max_abs_dp"], "mean_abs_dp": r["mean_abs_dp"],
                                                    "argmax": [q["argmax"] for q in r["questions"]],
                                                    "probs": [q["probs"] for q in r["questions"]]}}
    return {"doc": d, "runs": out}


def timing_by_t(runs: list[dict]) -> dict:
    """Median ms of the head call by T (first call = a new shape, second = the same shape again)."""
    groups = {"T<=300": (0, 300), "300<T<=700": (300, 700), "700<T<=1500": (700, 1500), "T>1500": (1500, 10 ** 9)}
    out = {}
    for g, (lo, hi) in groups.items():
        rs = [r for r in runs if lo < r["tokens"] <= hi]
        if not rs:
            continue
        out[g] = {"runs": len(rs), "median_tokens": float(np.median([r["tokens"] for r in rs])),
                  "median_padded_T": float(np.median([r["shape"]["T"] for r in rs])),
                  "first_call_ms_median": float(np.median([r["call_ms"][0] for r in rs])),
                  "second_call_ms_median": float(np.median([r["call_ms"][1] for r in rs])) if len(rs[0]["call_ms"]) > 1 else None,
                  "host_ms_median": float(np.median([r["host_ms"] for r in rs]))}
    return out


def gate_graph(args, stage: str) -> int:
    oracle = Path(args.oracle)
    doc, rows = load_oracle(oracle)
    orc = {(r["id"], r["arm"]): r for r in rows}
    hb = head_bundle(Path(args.head))
    meta = hb["metadata"]["head"]
    work = Path(args.work_dir) / args.tag
    work.mkdir(parents=True, exist_ok=True)
    readout = resolve_readout(Path(args.readout)) if stage == "h2" else None
    if stage == "h1":
        keys = [(r["id"], r["arm"]) for r in rows]
        runs = [[i, a, "base", None] for i, a in keys]
    else:
        keys = [k for k in [(r["id"], r["arm"]) for r in rows] if k in readout["runs"]]
        runs = [[i, a, "base", {"npz": readout["runs"][(i, a)]["npz"], "key": readout["runs"][(i, a)]["key"]}]
                for i, a in keys]
    common = {"aimodelc": hb["aimodelc"], "oracle": str(oracle), "table": str(Path(args.table)),
              "shape": meta["shape"], "functions": meta["functions"], "buckets": meta.get("buckets"), "dims": meta.get("dims"),
              "pad": meta["host_pad"], "calls_per_run": 2}
    t0 = time.monotonic()
    recs = run_workers([{**common, "runs": part, "out": str(work / f"shard_{k:02d}")} for k, part in enumerate(split(runs))])
    red_recs = []
    if stage == "h2" and args.red:
        red_keys = [k for k in RED_RUNS if k in readout["runs"]]
        red_runs = [[i, a, v, {"npz": readout["runs"][(i, a)]["npz"], "key": readout["runs"][(i, a)]["key"]}]
                    for i, a in red_keys for v in ("base",) + RED_ARMS]
        red_recs = run_workers([{**common, "runs": red_runs, "out": str(work / "red")}])
    gpu_s = time.monotonic() - t0
    scored = []
    for rec in recs:
        for r in rec["runs"]:
            row = orc[(r["id"], r["arm"])]
            sc = probs_and_scores(row, r["_logits"], r["layout"])
            item = {**{k: v for k, v in r.items() if k != "_logits"}, "source": row["source"], **sc,
                    "logits": r["_logits"].tolist()}
            if stage == "h2":
                ah = readout["runs"][(r["id"], r["arm"])]["author_head"]
                item["author_head_fp32_same_hidden"] = {"max_abs_dp": ah["max_abs_dp"], "mean_abs_dp": ah["mean_abs_dp"]}
                item["vs_author_head_same_hidden_max_abs_dp"] = max(
                    float(np.max(np.abs(np.asarray(q["probs"]) - np.asarray(p)))) for q, p in zip(sc["questions"], ah["probs"]))
                item["argmax_equal_author_head_same_hidden"] = [q["argmax"] for q in sc["questions"]] == ah["argmax"]
            scored.append(item)
    s = summarize(scored)
    resets = all(rec.get("reset_check", {}).get("bit_equal", False) for rec in recs + red_recs)
    calls_same = all(r["calls_bit_equal"] for rec in recs + red_recs for r in rec["runs"])
    finite = all(r["finite"] and r["padding_logits_zero"] for rec in recs + red_recs for r in rec["runs"])
    if stage == "h1":
        checks = {"all_runs": s["runs"] == len(rows), "argmax_every_question": s["argmax_equal"] == s["questions"],
                  "max_abs_dp": s["max_abs_dp"] <= H1_BAR["max_abs_dp"], "finite_and_padding_zero": finite,
                  "rerun_bit_equal": resets and calls_same}
        bar = H1_BAR
    else:
        checks = {"all_runs": s["runs"] == len(readout["runs"]),
                  "argmax_non_near_tie": s["argmax_equal_non_near_tie"] == s["questions_non_near_tie"],
                  "max_abs_dp": s["max_abs_dp"] <= H2_BAR["max_abs_dp"],
                  "mean_of_run_mean_abs_dp": s["mean_of_run_mean_abs_dp"] <= H2_BAR["mean_of_run_mean_abs_dp"],
                  "finite_and_padding_zero": finite, "rerun_bit_equal": resets and calls_same}
        bar = H2_BAR
    ok = all(checks.values())
    red = None
    if red_recs:
        red = {"arms": {"span_shift_right_1": "q_avg and o_avg rolled one column right (every span one token later)",
                        "lexical_zero": "lexical = 0 for every option",
                        "member_roll_1": "member columns rotated by one option (option o counted under option o-1's question)"},
               "runs": []}
        base = {}
        for rec in red_recs:
            for r in rec["runs"]:
                if r["variant"] == "base":
                    base[(r["id"], r["arm"])] = r
        for rec in red_recs:
            for r in rec["runs"]:
                if r["variant"] == "base":
                    continue
                row = orc[(r["id"], r["arm"])]
                b = base[(r["id"], r["arm"])]
                from clef_head import question_probs
                pb = question_probs(b["_logits"], b["layout"])
                pv = question_probs(r["_logits"], r["layout"])
                gate_run = next(x for x in scored if (x["id"], x["arm"]) == (r["id"], r["arm"]))
                red["runs"].append({"id": r["id"], "arm": r["arm"], "arm_name": r["variant"],
                                    "questions": len(row["questions"]),
                                    "argmax_changed": int(sum(int(np.argmax(x) != np.argmax(y)) for x, y in zip(pv, pb))),
                                    "max_abs_dp_vs_base": float(max(np.max(np.abs(x - y)) for x, y in zip(pv, pb))),
                                    "base_equals_gate_run": bool(np.array_equal(b["_logits"], np.asarray(gate_run["logits"], np.float32)))})
        red["by_arm"] = {a: {"runs": len([x for x in red["runs"] if x["arm_name"] == a]),
                             "argmax_changed_total": sum(x["argmax_changed"] for x in red["runs"] if x["arm_name"] == a),
                             "questions_total": sum(x["questions"] for x in red["runs"] if x["arm_name"] == a),
                             "max_abs_dp_vs_base": max(x["max_abs_dp_vs_base"] for x in red["runs"] if x["arm_name"] == a),
                             "moves": all(x["max_abs_dp_vs_base"] > 0.02 for x in red["runs"] if x["arm_name"] == a)}
                         for a in RED_ARMS}
        red["result"] = "RED (every arm moves every run by more than 0.02)" if all(
            v["moves"] for v in red["by_arm"].values()) else "NOT RED on every run"
    procs = [{k: v for k, v in rec.items() if k not in ("runs", "spec")} | {"runs": len(rec["runs"])}
             for rec in recs + red_recs]
    descs = [rec["descriptors"] for rec in recs]
    record = {
        "schema": f"clef-flash-head-{stage}/1",
        "gate": ("head graph alone on the oracle's fp32 hidden (lexical from the fp16 table)" if stage == "h1" else
                 "end to end in Python: the decoder bundle's fp16 hidden (readout shards, not re-run) -> host span "
                 "matrices + fp16-table lexical -> head graph -> per-question fp32 softmax"),
        "bar": bar, "checks": checks, "result": "PASS" if ok else "FAIL", "summary": s,
        "by_source_arm": {g: summarize([x for x in scored if f"{x['source']}/{x['arm']}" == g])
                          for g in sorted({f"{x['source']}/{x['arm']}" for x in scored})},
        "head_bundle": {k: v for k, v in hb.items() if k != "metadata"} | {"head_metadata": meta},
        "descriptors": descs[0] if descs else None,
        "descriptors_same_every_process": all(d == descs[0] for d in descs),
        "oracle": {"path": str(oracle / "records_oracle.json"), "sha256": sha256_file(oracle / "records_oracle.json"),
                   "runs": len(rows)},
        "table": {"path": str(args.table), "json": str(Path(args.table).with_suffix(".json"))},
        "near_ties": [{"id": x["id"], "arm": x["arm"], "question_id": q["question_id"], "oracle_top2_margin": q["oracle_top2_margin"],
                       "argmax_equal": q["argmax_equal"], "max_abs_dp": q["max_abs_dp"], "probs": q["probs"],
                       "probs_oracle": q["probs_oracle"]} for x in scored for q in x["questions"] if q["near_tie"]],
        "timing": {"contended": True, "by_T": timing_by_t(scored),
                   "all_first_call_ms_median": float(np.median([x["call_ms"][0] for x in scored])),
                   "all_second_call_ms_median": float(np.median([x["call_ms"][1] for x in scored]))},
        "processes": procs, "red_arm": red,
        "environment": env_record(),
        "script": {"gate_head.py": sha256_file(Path(__file__).resolve()), "clef_head.py": sha256_file(HERE / "clef_head.py")},
        "seconds": {"gpu_processes": gpu_s},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": scored,
    }
    if stage == "h2":
        record["readout"] = {"path": str(args.readout), "sha256": sha256_file(Path(args.readout)),
                             "bundle": readout["doc"]["bundle"]["name"], "result": readout["doc"]["result"],
                             "author_head_summary": {k: readout["doc"]["summary"][k] for k in (
                                 "max_abs_dp", "mean_of_run_mean_abs_dp", "argmax_equal_non_near_tie",
                                 "questions_non_near_tie", "argmax_equal_near_tie", "near_tie_questions")},
                             "vs_author_head_same_hidden": {
                                 "max_abs_dp": max(x["vs_author_head_same_hidden_max_abs_dp"] for x in scored),
                                 "argmax_equal_runs": sum(x["argmax_equal_author_head_same_hidden"] for x in scored)}}
    if args.note:
        record["note"] = args.note
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    print(f"{stage.upper()} {record['result']}: {json.dumps(s)}")
    for k, v in checks.items():
        print(f"  {k}: {'ok' if v else 'FAIL'}")
    if stage == "h2":
        print(f"  author head (same hidden): {json.dumps(record['readout']['author_head_summary'])}; graph vs author head "
              f"{json.dumps(record['readout']['vs_author_head_same_hidden'])}")
    if red:
        print(f"  red: {json.dumps(red['by_arm'])} -> {red['result']}")
    print(f"  timing: {json.dumps(record['timing'])}")
    print(f"wrote {args.out}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a0 = sub.add_parser("h0")
    a0.add_argument("--oracle", default=str(ORACLE))
    a0.add_argument("--threads", type=int, default=8)
    a0.add_argument("--out", required=True)
    for st in ("h1", "h2"):
        a = sub.add_parser(st)
        a.add_argument("--head", required=True, help="the head bundle dir (metadata.json names the AOT asset)")
        a.add_argument("--oracle", default=str(ORACLE))
        a.add_argument("--table", default=str(TABLE))
        a.add_argument("--work-dir", default=str(LANE / "readout"))
        a.add_argument("--tag", default=f"head_{st}")
        a.add_argument("--out", required=True)
        a.add_argument("--note")
        if st == "h2":
            a.add_argument("--readout", required=True, help="a readout_gate.py transcript (its shards hold the hidden rows)")
            a.add_argument("--red", action="store_true")
    aw = sub.add_parser("worker")
    aw.add_argument("--spec", required=True)
    args = ap.parse_args()
    if args.cmd == "worker":
        return worker(Path(args.spec))
    if args.cmd == "h0":
        return h0(args)
    return gate_graph(args, args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
