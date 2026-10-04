#!/usr/bin/env python3
"""Kev decision latency on the Mac GPU (AOT `.aimodelc`, decide.py's path), in one _GPU_LOCK window.

Protocol:
  * the lock (`_paths.gpu_lock()`, ~/code/coreai/_GPU_LOCK): when it is empty, write `kev timing r5 pid <pid> since
    <time>`, measure, and empty it (0 B) at the end, a failed run included; when another session holds it, poll for up
    to 15 minutes, then measure without it and mark the run contended; with the lock taken, a GPU job another session
    started before it (a readout gate, ...) is waited out, up to 15 minutes, before the first process;
  * the other GPU-capable processes (agent CLIs and shells left out), `top`'s load line and the busiest processes,
    before and after the window, and between processes;
  * four processes in the window, Kev-0.8B, Kev-4B, Kev-0.8B, Kev-4B (A B A B); each loads the AOT asset twice
    (cold = the first load after the process starts, warm = the second) and then measures, each item after one warm-up:
      (a) one decision = one question = one row: tv4_000 q0 (94 tokens), own_j03 q0 (380), own_L02 q0 (1,518) and the
          fixture's longest row, own_L01 q2 (1,802); 10 decisions each;
      (b) own_m01's first 5 questions as one request (state 137 tokens), (c) all 8 of them, (d) own_L02's 4 questions
          (state 1,477 tokens): direct and shared alternated (D S, then S D), 10 each.
    A decision's `latency_ms` is decide.py's: the graph calls and the head, state allocations and copies included,
    tokenizing and the answers not; `e2e_ms` is request in -> response out.
Medians with p10 / p90 over the 20 decisions of the two processes of a model (and per process). The numbers are this
machine's in this window and are not compared with any other runtime.

    cd conversion/kev
    $PY timing.py run                    # -> $ZOO_WORK_ROOT/_kev/timing/<tag>/ (raw) + results/timing_r5.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
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
import decide  # noqa: E402
import host  # noqa: E402
from _paths import gpu_lock  # noqa: E402

LANE = decide.LANE
FIXTURES = LANE / "fixtures" / "records.json"
ORDER = ("kev-0.8b", "kev-4b", "kev-0.8b", "kev-4b")
REPS = 10
LOCK_WAIT_MINUTES = 15
SINGLE = (("tv4_000", 0), ("own_j03", 0), ("own_L02", 0), ("own_L01", 2))
MULTI = (("b_m01_first5", "own_m01", 5), ("c_m01_all8", "own_m01", 8), ("d_L02_all4", "own_L02", 4))


def stats(a) -> dict:
    a = np.asarray(a, np.float64)
    return {"n": int(a.size), "median": float(np.median(a)), "p10": float(np.quantile(a, 0.1)),
            "p90": float(np.quantile(a, 0.9)), "min": float(a.min()), "max": float(a.max())}


def sub_request(request: dict, keep: list[int]) -> dict:
    qs = list(request["questions"].items())
    return {**request, "questions": {qid: q for i, (qid, q) in enumerate(qs) if i in keep}}


# --------------------------------------------------------------------------- worker: one model, one process
def worker(model: str, slot: int, out: Path) -> int:
    t_start = time.time()
    recs = {r["id"]: r for r in json.loads(FIXTURES.read_text())["records"]}
    e = decide.Kev(decide.BUNDLES[model])
    doc: dict = {"model": model, "slot": slot, "pid": os.getpid(), "bundle": e.name, "aimodelc": str(e.aimodelc),
                 "started": t_start}

    async def one(req: dict, shared: bool) -> tuple[dict, dict]:
        tr: dict = {}
        t0 = time.perf_counter()
        body = await e.decide(req, shared=shared, trace=tr)
        e2e = (time.perf_counter() - t0) * 1e3
        rec = {"latency_ms": tr["decide_ms"], "graph_ms": tr["graph_ms"], "head_ms": tr["head_ms"], "host_ms": tr["host_ms"],
               "e2e_ms": e2e, "calls": tr["calls"], "padded_tokens": tr["padded_tokens"]}
        return rec, {"body": body, "probs": tr["_probs"], "trace": tr}

    async def go():
        doc["load_cold"] = await e.load()
        doc["load_warm"] = await e.load()
        doc["single"] = []
        for rid, k in SINGLE:
            req = sub_request(recs[rid]["request"], [k])
            warm, _ = await one(req, False)
            reps = [(await one(req, False))[0] for _ in range(REPS)]
            T = len(host.build_rows(req, e.tok)["rows"][0]["row_ids"])
            lat = [r["latency_ms"] for r in reps]
            item = {"item": f"{rid}:q{k}", "row_tokens": T, "calls": reps[0]["calls"], "padded_tokens": reps[0]["padded_tokens"],
                    "warmup_latency_ms": warm["latency_ms"], "latency_ms": stats(lat), "graph_ms": stats([r["graph_ms"] for r in reps]),
                    "e2e_ms": stats([r["e2e_ms"] for r in reps]), "head_ms": stats([r["head_ms"] for r in reps]),
                    "tokens_per_s_at_median": T / (float(np.median(lat)) / 1e3), "reps": reps}
            doc["single"].append(item)
            print(f"  [{os.getpid()} {model}] {item['item']}: T {T}, {item['calls']} calls, median {item['latency_ms']['median']:.1f} ms "
                  f"(p10 {item['latency_ms']['p10']:.1f}, p90 {item['latency_ms']['p90']:.1f})", flush=True)
        doc["multi"] = []
        for name, rid, n in MULTI:
            req = sub_request(recs[rid]["request"], list(range(n)))
            b = host.build_rows(req, e.tok)
            (await one(req, False))
            (await one(req, True))
            reps = {"direct": [], "shared": []}
            same = True
            for i in range(REPS):
                outs = {}
                for mode in (("direct", "shared") if i % 2 == 0 else ("shared", "direct")):
                    r, o = await one(req, mode == "shared")
                    reps[mode].append(r)
                    outs[mode] = o
                same = same and all(np.array_equal(a, c) for a, c in zip(outs["direct"]["probs"], outs["shared"]["probs"]))
            plan = host.shared_prefix_plan(b["state_len"], e.S, e.q, e.qmin)
            item = {"item": name, "record": rid, "questions": n, "state_len": b["state_len"], "input_tokens": b["input_tokens"],
                    "row_tokens": [len(r["row_ids"]) for r in b["rows"]], "shared_plan": plan,
                    "direct_and_shared_p_bit_equal_every_rep": bool(same)}
            for mode in ("direct", "shared"):
                lat = [r["latency_ms"] for r in reps[mode]]
                item[mode] = {"calls": reps[mode][0]["calls"], "padded_tokens": reps[mode][0]["padded_tokens"],
                              "latency_ms": stats(lat), "graph_ms": stats([r["graph_ms"] for r in reps[mode]]),
                              "e2e_ms": stats([r["e2e_ms"] for r in reps[mode]]),
                              "input_tokens_per_s_at_median": b["input_tokens"] / (float(np.median(lat)) / 1e3),
                              "reps": reps[mode]}
            doc["multi"].append(item)
            print(f"  [{os.getpid()} {model}] {name}: direct {item['direct']['latency_ms']['median']:.1f} ms / "
                  f"{item['direct']['calls']} calls, shared {item['shared']['latency_ms']['median']:.1f} ms / "
                  f"{item['shared']['calls']} calls, p bit-equal {same}", flush=True)

    asyncio.run(go())
    doc["finished"] = time.time()
    out.write_text(json.dumps(doc, indent=1) + "\n")
    return 0


# --------------------------------------------------------------------------- driver
def top_line() -> list[str]:
    t = subprocess.run(["top", "-l", "1", "-n", "0"], capture_output=True, text=True).stdout.splitlines()
    return [ln for ln in t if ln.startswith(("Load Avg", "CPU usage", "PhysMem"))]


def busiest() -> list[str]:
    ps = subprocess.run(["ps", "-Ao", "%cpu,command"], capture_output=True, text=True).stdout.splitlines()[1:]
    rows = sorted(ps, key=lambda s: -float(s.split(None, 1)[0]))[:5]
    return [r.strip()[:160] for r in rows]


def snapshot() -> dict:
    return {"at": datetime.now().astimezone().isoformat(timespec="seconds"), "lock": decide.lock_state(),
            "other_gpu_processes": decide.other_gpu_processes(), "gpu_jobs": gpu_jobs(), "top": top_line(),
            "busiest": busiest(), "memory": memory()}


def take_lock(tag: str) -> dict:
    lock = gpu_lock()
    t0 = time.monotonic()
    seen = None
    while True:
        st = decide.lock_state()
        if not st.get("bytes"):
            content = f"kev timing r5 pid {os.getpid()} since {datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            lock.write_text(content)
            return {"taken": True, "content": content.strip(), "waited_seconds": round(time.monotonic() - t0, 1),
                    "held_by_before": seen, "contended": False}
        seen = st.get("content")
        if time.monotonic() - t0 > LOCK_WAIT_MINUTES * 60:
            return {"taken": False, "content": None, "waited_seconds": round(time.monotonic() - t0, 1),
                    "held_by_before": seen, "contended": True}
        print(f"_GPU_LOCK held ({seen!r}); waiting", flush=True)
        time.sleep(30)


GPU_JOBS = re.compile(r"readout_gate\.py|/gate_[a-z0-9_]*\.py|coreai_verify|llm-bench|yardstick|mlx_lm|"
                      r"decide\.py (worker|check|run)|--accel gpu|gpu_gate_subprocess|litert_parity\.py .*gpu")


def gpu_jobs() -> list[str]:
    """GPU work another session started before our lock (a gate in progress is not stopped by a lock it never read;
    the LiteRT lane's correctness gates run on the Mac GPU without the lock by their own rule)."""
    return [p for p in decide.other_gpu_processes() if GPU_JOBS.search(p)]


def memory() -> dict:
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout.splitlines()
    pages = {ln.split(":")[0].strip(): ln.split(":")[1].strip().rstrip(".") for ln in vm[1:8] if ":" in ln}
    return {"swapusage": sw, "vm_stat_pages_16k": pages}


def wait_gpu_quiet(max_minutes: float = LOCK_WAIT_MINUTES) -> dict:
    t0 = time.monotonic()
    seen: list = []
    while True:
        jobs = gpu_jobs()
        if not jobs:
            return {"waited_seconds": round(time.monotonic() - t0, 1), "jobs_seen": seen, "quiet": True}
        seen = sorted(set(seen) | set(jobs))
        if time.monotonic() - t0 > max_minutes * 60:
            return {"waited_seconds": round(time.monotonic() - t0, 1), "jobs_seen": seen, "quiet": False}
        print(f"GPU jobs running ({len(jobs)}): {jobs[0][:120]}; waiting", flush=True)
        time.sleep(15)


def release_lock(info: dict) -> dict:
    if not info.get("taken"):
        return {"emptied": False, "why": "not ours"}
    lock = gpu_lock()
    now = lock.read_text() if lock.exists() else ""
    if now.strip() != info["content"]:
        return {"emptied": False, "why": f"the lock holds {now[:200]!r}, not ours"}
    with open(lock, "w"):
        pass
    return {"emptied": True, "bytes_after": lock.stat().st_size}


GATE_REFERENCE = {"kev-0.8b": "readout_fp16_pf16.json", "kev-4b": "readout_fp16_pf16_4b.json"}


def per_call_check(model: str, ps: list[dict]) -> dict:
    """Each process's median graph ms per call over its one-row decisions, beside the readout gate's warm median of
    the same asset (rounds 2 / 4, no lock window): a process far above it ran on a busy GPU."""
    ref = json.loads((LANE / "results" / GATE_REFERENCE[model]).read_text())["timing"]["ms_per_call_warm_runs"]
    mine = [float(np.median([r["graph_ms"] / r["calls"] for it in p["single"] for r in it["reps"]])) for p in ps]
    return {"ms_per_call_median_per_process": mine, "gate_warm_median": ref["median"],
            "gate_transcript": GATE_REFERENCE[model], "ratio_per_process": [m / ref["median"] for m in mine]}


def summarize(procs: list[dict]) -> dict:
    out = {}
    for model in sorted({p["model"] for p in procs}):
        ps = [p for p in procs if p["model"] == model]
        m = {"processes": [p["slot"] for p in ps],
             "load_seconds": {"cold": [p["load_cold"]["seconds"] for p in ps], "warm": [p["load_warm"]["seconds"] for p in ps]},
             "per_call_check": per_call_check(model, ps), "single": [], "multi": []}
        for k, item in enumerate(ps[0]["single"]):
            reps = [r for p in ps for r in p["single"][k]["reps"]]
            lat = [r["latency_ms"] for r in reps]
            m["single"].append({"item": item["item"], "row_tokens": item["row_tokens"], "calls": item["calls"],
                                "padded_tokens": item["padded_tokens"], "latency_ms": stats(lat),
                                "latency_ms_median_per_process": [p["single"][k]["latency_ms"]["median"] for p in ps],
                                "e2e_ms": stats([r["e2e_ms"] for r in reps]), "graph_ms": stats([r["graph_ms"] for r in reps]),
                                "tokens_per_s_at_median": item["row_tokens"] / (float(np.median(lat)) / 1e3),
                                "padded_tokens_per_s_at_median": item["padded_tokens"] / (float(np.median(lat)) / 1e3)})
        for k, item in enumerate(ps[0]["multi"]):
            row = {"item": item["item"], "record": item["record"], "questions": item["questions"], "state_len": item["state_len"],
                   "input_tokens": item["input_tokens"], "row_tokens": item["row_tokens"], "shared_plan": item["shared_plan"],
                   "p_bit_equal_every_rep": all(p["multi"][k]["direct_and_shared_p_bit_equal_every_rep"] for p in ps)}
            for mode in ("direct", "shared"):
                reps = [r for p in ps for r in p["multi"][k][mode]["reps"]]
                lat = [r["latency_ms"] for r in reps]
                row[mode] = {"calls": item[mode]["calls"], "padded_tokens": item[mode]["padded_tokens"], "latency_ms": stats(lat),
                             "latency_ms_median_per_process": [p["multi"][k][mode]["latency_ms"]["median"] for p in ps],
                             "e2e_ms": stats([r["e2e_ms"] for r in reps]),
                             "input_tokens_per_s_at_median": item["input_tokens"] / (float(np.median(lat)) / 1e3)}
            row["shared_over_direct_median"] = row["shared"]["latency_ms"]["median"] / row["direct"]["latency_ms"]["median"]
            m["multi"].append(row)
        out[model] = m
    return out


def cmd_run(args) -> int:
    tag = args.tag or f"r5_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = LANE / "timing" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    window = {"tag": tag, "order": list(ORDER), "reps": REPS}
    window["lock"] = take_lock(tag)
    procs, between = [], []
    try:
        window["gpu_quiet_wait"] = wait_gpu_quiet()
        if not window["gpu_quiet_wait"]["quiet"]:
            window["lock"]["contended"] = True
        window["before"] = snapshot()
        print(f"window: lock {window['lock']}", flush=True)
        for slot, model in enumerate(ORDER):
            out = out_dir / f"p{slot}_{model}.json"
            log = out_dir / f"p{slot}_{model}.log"
            t0 = time.monotonic()
            with open(log, "w") as fh:
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--model", model,
                                       "--slot", str(slot), "--out", str(out)], stdout=fh, stderr=subprocess.STDOUT)
            wall = time.monotonic() - t0
            print(f"[{slot}] {model}: exit {proc.returncode}, {wall:.0f} s", flush=True)
            if proc.returncode != 0 or not out.exists():
                raise SystemExit(f"process {slot} ({model}) failed; see {log}")
            d = json.loads(out.read_text())
            d["process_wall_seconds"] = wall
            procs.append(d)
            between.append(snapshot())
        window["after"] = snapshot()
    finally:
        window["release"] = release_lock(window["lock"])
        print(f"lock released: {window['release']}", flush=True)
        window["between_processes"] = between
        (out_dir / "window.json").write_text(json.dumps(window, indent=1) + "\n")
    return write_summary(out_dir, procs, window, Path(args.out))


def write_summary(out_dir: Path, procs: list[dict], window: dict, res: Path) -> int:
    doc = {"schema": "kev-timing/1",
           "what": "decide.py's decision latency (graph + head) per model, A B A B processes in one _GPU_LOCK window",
           "machine": {"chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
                       "memory_bytes": int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout),
                       "macos_build": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
                       "python": platform.python_version()},
           "contended": window["lock"]["contended"], "window": window,
           "raw": {"dir": str(out_dir), "files": sorted(p.name for p in out_dir.iterdir())},
           "scripts": {n: decide.sha256_file(HERE / n) for n in ("timing.py", "decide.py", "host.py")},
           "summary": summarize(procs),
           "note": "latency_ms = graph calls + head (decide.py's latency_ms); the times are this machine's in this window only",
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    res.write_text(json.dumps(doc, indent=1) + "\n")
    (out_dir / "timing_r5.copy.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {res}")
    return 0


def cmd_resummarize(args) -> int:
    """Rebuild the summary from a run's raw directory (window.json + p<slot>_<model>.json)."""
    out_dir = Path(args.dir)
    window = json.loads((out_dir / "window.json").read_text())
    procs = []
    for slot, model in enumerate(window["order"]):
        d = json.loads((out_dir / f"p{slot}_{model}.json").read_text())
        procs.append(d)
    return write_summary(out_dir, procs, window, Path(args.out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default=str(LANE / "results" / "timing_r5.json"))
    r.add_argument("--tag")
    w = sub.add_parser("worker")
    w.add_argument("--model", required=True, choices=sorted(decide.BUNDLES))
    w.add_argument("--slot", type=int, required=True)
    w.add_argument("--out", required=True)
    s = sub.add_parser("resummarize")
    s.add_argument("--dir", required=True)
    s.add_argument("--out", default=str(LANE / "results" / "timing_r5.json"))
    args = ap.parse_args()
    if args.cmd == "worker":
        return worker(args.model, args.slot, Path(args.out))
    if args.cmd == "resummarize":
        return cmd_resummarize(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
