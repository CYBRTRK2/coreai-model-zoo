#!/usr/bin/env python3
"""Kev Python reference: a SystemOne request -> the SystemOne response, on the Core AI graph (Mac GPU, AOT only).

The whole read-out, in the order a Swift host repeats it (`host.py` is the specification of every host step):

    request JSON --host.build_rows--> one row per question: ids, the <decide> / </opt> indices, keys (and legend)
    host.graph_context_check: the padded end of every row <= max_context_length - 1
    graph (the bundle's AOT `.aimodelc`, function `main`, SpecializationOptions.default(); never the JIT), its calls
    cut by host.plan with the bundle's call max L / q (host.graph_shape: a static-S bundle is L = q = S; a dynamic-S
    one, round 14's query_len_range, has L = query_len_call_max and q = query_len_multiple; --call-max / --multiple
    override them):
        direct   every row from fresh zero states in host.plan(T) calls (position_ids 0..p+c-1 for a call of c ids
                 after p, the last call padded with <|endoftext|> up to a multiple of q) -> hidden [T, d] fp16
        shared   (--shared) the first k row tokens (k = floor(Ls / q) * q: whole multiples of q of state tokens) once;
                 its four states copied per question (the KV sequence axis up to k, the rest zero as in a direct
                 run); each row's remaining tokens in host.plan calls from position k. Static S: the direct run's
                 calls, so its hidden rows bit for bit; dynamic S: other cuts, other last bits (p within ~1e-3)
    warm_up (run --warm): every call length of the plan (q, 2q, ..., L) once from zero states, so no request pays a
        length's first-call specialization
    prepared state (round 15): `prepare(state)` runs the state's first k = floor(Ls / q) * q tokens once and keeps the
        four states (and those rows); `decide_prepared(prepared, questions)` answers questions on that state later, each
        from a copy of the kept states, only the row's tokens from k on: the shared prefix split in two, the same calls,
        so its hidden rows and p equal the `shared` request's bit for bit. The caller holds the prepared value (no cache).
    host.head_logits + host.head_probs at <decide> and at every </opt> (float64, p rounded to fp32)
    host.to_answers -> host.response {"model": the bundle name, "answers", "usage", "latency_ms"}

`latency_ms` is the graph calls + the head (the state allocations and copies included; tokenizing and the answers
are not), as the author's server times its model pass. Everything model-specific comes from the bundle:
metadata.json (S, max_context_length, the delimiter and pad ids, the head files), head/ (weights, scale,
temperature) and tokenizer/tokenizer.json; the `.aimodelc` is `<bundles>/../bundles_aotc/<name>.h16c.aimodelc`.

    cd conversion/kev
    PY=<coreai-models venv>/bin/python
    $PY decide.py run --model kev-0.8b --request req.json [--shared] --out resp.json [--trace trace.json]
    $PY decide.py check --model kev-0.8b --rows all --shared --out $ZOO_WORK_ROOT/_kev/results/e2e_kev-0.8b.json
    $PY decide.py check --model kev-4b --rows heldout --shared --out $ZOO_WORK_ROOT/_kev/results/e2e_kev-4b_heldout.json

`check` answers every fixture record from its raw request, direct and (--shared) shared, at most 40 records per
process plus a reset re-run of the process's first record (the Python runtime leaks an IOSurface per call), and
scores it against the round's gate transcript of the same asset and the author's oracle:
  (i)   the direct hidden rows == the gate's (its shard npz) bit for bit; the gate's own fp32 torch head
        (parity_decoder_torch.KevHead) on them == the transcript's probs bit for bit; the host head's p against the
        transcript (max |dp|, bit-equal rows: float64 vs PyTorch fp32 arithmetic);
  (ii)  static S: shared == direct (hidden rows and p, bit for bit; max |dp|). Dynamic S (round 15): the shared p
        against the oracle's bar (iv), max |dp| and the hidden difference against direct recorded, and with
        --shared-gate-transcript (readout_gate.py --shared of the same asset) the shared hidden rows == that gate's;
  (iii) the response from the host p against the oracle's response: answers, every rounded field, the choice /
        noul / score values, usage (the oracle body's `model` is the request's echo, ours the bundle name);
  (iv)  the readout gate's bar recomputed against the oracle from the host p, the torch-head p and the shared p.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import re
import resource
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
import host  # noqa: E402
from _paths import gpu_lock, work_path  # noqa: E402

LANE = work_path("_kev")
BUNDLES = {"kev-0.8b": LANE / "exports" / "bundles" / "kev_0_8b_decode_fp16_pf16",
           "kev-4b": LANE / "exports" / "bundles" / "kev_4b_decode_fp16_pf16"}
ORACLES = {"kev-0.8b": LANE / "oracle", "kev-4b": LANE / "oracle_4b"}
GATES = {("kev-0.8b", "all"): LANE / "results" / "readout_fp16_pf16.json",
         ("kev-0.8b", "heldout"): LANE / "results" / "readout_heldout_fp16_pf16.json",
         ("kev-4b", "all"): LANE / "results" / "readout_fp16_pf16_4b.json",
         ("kev-4b", "heldout"): LANE / "results" / "readout_heldout_fp16_pf16_4b.json"}
FIXTURES = {"all": LANE / "fixtures" / "records.json", "heldout": LANE / "fixtures" / "heldout.json"}
WORK = LANE / "host"
RECORDS_PER_PROCESS = 40
BAR = {"max_abs_dp": 0.02, "mean_of_run_mean_abs_dp": 0.002, "near_tie_top2_margin": 0.02}
SHARED_MAX_ABS_DP = 1e-4
OTHER_GPU = re.compile(r"yardstick|litert|llm-bench|coreai_verify|coreai-build|readout_gate|gate_|parity_|mlx|decide\.py|timing\.py")


async def maybe(x):
    return await x if inspect.isawaitable(x) else x


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(path: Path) -> dict:
    """readout_gate.tree_digest: the per-file sha256 of a directory and one hash over them."""
    files = sorted(p for p in path.rglob("*") if p.is_file())
    per = {str(p.relative_to(path)): sha256_file(p) for p in files}
    tree = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in per.items()).encode()).hexdigest()
    return {"path": str(path), "bytes": sum(p.stat().st_size for p in files), "tree_sha256": tree}


# --------------------------------------------------------------------------- the engine
class Kev:
    """One bundle's graph, head and tokenizer, loaded once; `decide()` answers one request."""

    def __init__(self, bundle: Path, aimodelc: Path | None = None, call_max: int | None = None, multiple: int | None = None):
        self.bundle = Path(bundle).expanduser().resolve()
        self.meta = json.loads((self.bundle / "metadata.json").read_text())
        lang, dec = self.meta["language"], self.meta["decision"]
        self.name = self.meta["name"]
        self.shape = host.graph_shape(lang, call_max, multiple)   # L (cap) / q / qmin; a static-S bundle is L = q = S
        self.dynamic = self.shape["dynamic"]
        self.S, self.q, self.qmin = self.shape["cap"], self.shape["q"], self.shape["qmin"]   # S = the call max L
        self.max_ctx = int(lang["max_context_length"])
        row = dec["row"]
        self.delims = {n: int(v["id"]) for n, v in row["delimiters"].items()}
        self.delims["pad"] = int(row["pad"]["id"])
        self.tok = host.load_tokenizer(self.bundle / "tokenizer" / "tokenizer.json")
        mine = host.delimiter_ids(self.tok)
        if mine != self.delims:
            raise SystemExit(f"{self.bundle}: the tokenizer's delimiter ids {mine} != metadata {self.delims}")
        head_files = [self.bundle / f for f in dec["head"]["files"]]
        if not all(p.exists() for p in head_files):
            raise SystemExit(f"{self.bundle}: missing head files {head_files}")
        self.head_dir = head_files[0].parent
        self.head = host.load_head(self.head_dir)
        if self.head["scale"] != float(dec["head"]["scale"]):
            raise SystemExit(f"head scale {self.head['scale']} != metadata {dec['head']['scale']}")
        self.T = self.head["temperature"]
        self.d = self.head["hidden_size"]
        self.aimodelc = (Path(aimodelc).expanduser().resolve() if aimodelc
                         else self.bundle.parent.parent / "bundles_aotc" / f"{self.name}.h16c.aimodelc")
        if not self.aimodelc.exists():
            raise SystemExit(f"no AOT asset {self.aimodelc} (the JIT is never used)")
        self.fn = None

    async def load(self) -> dict:
        import coreai.runtime as rt
        self.rt = rt
        t0 = time.perf_counter()
        model = await maybe(rt.AIModel.load(str(self.aimodelc), rt.SpecializationOptions.default()))
        t1 = time.perf_counter()
        fn = await maybe(model.load_function("main"))
        t2 = time.perf_counter()
        self._model, self.fn = model, fn
        d = fn.desc
        self.state_desc = {n: ([int(x) for x in d.state_descriptor(n).shape], str(d.state_descriptor(n).dtype).split(".")[-1])
                           for n in d.state_names}
        out = [int(x) for x in d.output_descriptor("hidden").shape]
        ids = [int(x) for x in d.input_descriptor("input_ids").shape]
        n = -1 if self.dynamic else self.S                  # a dynamic-S graph: input_ids [1, -1], hidden [1, -1, d]
        if out != [1, n, self.d] or ids != [1, n]:
            raise SystemExit(f"the graph's contract (input_ids {ids}, hidden {out}) != {'dynamic' if self.dynamic else 'S'} "
                             f"{n}, d {self.d}")
        return {"model_seconds": t1 - t0, "main_seconds": t2 - t1, "seconds": t2 - t0}

    def plan(self, n: int) -> list[tuple[int, int]]:
        return host.plan(n, self.S, self.q, self.qmin)

    # states ---------------------------------------------------------------
    def zero_arrays(self) -> dict[str, np.ndarray]:
        return {n: np.zeros([self.max_ctx if s < 0 else s for s in shape], np.dtype(dt))
                for n, (shape, dt) in self.state_desc.items()}

    def to_state(self, arrays: dict[str, np.ndarray]) -> dict:
        return {n: self.rt.NDArray(np.ascontiguousarray(a)) for n, a in arrays.items()}   # the runtime copies

    def snapshot(self, state: dict) -> dict[str, np.ndarray]:
        return {n: np.array(v.numpy(), copy=True) for n, v in state.items()}

    def branch_arrays(self, snap: dict[str, np.ndarray], kS: int) -> dict[str, np.ndarray]:
        """A question's states after the shared prefix: KV up to kS on its sequence axis (beyond it zero, as a
        direct run has it after kS tokens), the conv / recurrent states as they are."""
        out = self.zero_arrays()
        for n, (shape, _) in self.state_desc.items():
            if -1 in shape:
                ax = shape.index(-1)
                sl = tuple(slice(0, kS) if i == ax else slice(None) for i in range(len(shape)))
                out[n][sl] = snap[n][sl]
            else:
                out[n][...] = snap[n]
        return out

    # graph calls ------------------------------------------------------------
    async def call(self, x: np.ndarray, p: int, state: dict) -> np.ndarray:
        """One call: ids x [c] after p earlier ids (position_ids 0..p+c-1) -> hidden [c, d] fp16."""
        c = len(x)
        res = await maybe(self.fn(inputs={"input_ids": self.rt.NDArray(np.ascontiguousarray(x.reshape(1, c))),
                                          "position_ids": self.rt.NDArray(np.arange(p + c, dtype=np.int32)[None])},
                                  state=state))
        h = np.asarray(res["hidden"].numpy())
        if h.shape != (1, c, self.d) or h.dtype != np.float16:
            raise SystemExit(f"hidden {h.shape} {h.dtype} != (1, {c}, {self.d}) float16")
        return h[0]

    async def run_plan(self, ids: list[int], p0: int, state: dict, call_ms: list | None = None,
                       call_lens: list | None = None) -> np.ndarray:
        """ids from position p0 on, in host.plan calls (the last padded with <|endoftext|>) -> hidden fp16 [len, d]."""
        out = np.empty((len(ids), self.d), np.float16)
        q0 = 0
        for c, r in self.plan(len(ids)):
            x = np.full(c, self.delims["pad"], np.int32)
            x[:r] = ids[q0:q0 + r]
            t = time.perf_counter()
            out[q0:q0 + r] = (await self.call(x, p0 + q0, state))[:r]
            if call_ms is not None:
                call_ms.append((time.perf_counter() - t) * 1e3)
            if call_lens is not None:
                call_lens.append(c)
            q0 += r
        return out

    async def hidden_direct(self, rows: list[dict], call_ms: list, call_lens: list | None = None) -> list[np.ndarray]:
        hs = []
        for r in rows:
            hs.append(await self.run_plan(r["row_ids"], 0, self.to_state(self.zero_arrays()), call_ms, call_lens))
        return hs

    async def hidden_shared(self, rows: list[dict], state_len: int, call_ms: list,
                            call_lens: list | None = None) -> tuple[list[np.ndarray], dict]:
        plan = host.shared_prefix_plan(state_len, self.S, self.q, self.qmin)
        k = plan["shared_tokens"]
        if plan["k"] == 0:
            return await self.hidden_direct(rows, call_ms, call_lens), plan
        prefix = rows[0]["row_ids"][:k]
        if any(r["row_ids"][:k] != prefix for r in rows):
            raise SystemExit("rows do not share their first k tokens")
        state = self.to_state(self.zero_arrays())
        h_pre = await self.run_plan(prefix, 0, state, call_ms, call_lens)
        snap = self.snapshot(state)
        hs = []
        for r in rows:
            h = await self.run_plan(r["row_ids"][k:], k, self.to_state(self.branch_arrays(snap, k)), call_ms, call_lens)
            hs.append(np.concatenate([h_pre, h]))
        return hs, plan

    # prepared state (round 15) ------------------------------------------------
    def state_ids(self, state) -> list[int]:
        """[<state>] + user_tokens(render(state)) within the serving limit (host.encode's first check and message)."""
        toks = host.user_tokens(self.tok, host.render(state))
        if len(toks) + 1 > host.SERVE_MAX_STATE:
            raise host.ContextOverflow(f"state is {len(toks) + 1:,} tokens, over the {host.SERVE_MAX_STATE:,}-token limit (the "
                                       "<state> token included): shorten the document or split it across requests",
                                       state_tokens=len(toks) + 1, max_state=host.SERVE_MAX_STATE)
        return [self.delims["state"]] + toks

    async def prepare(self, state, trace: dict | None = None) -> dict:
        """The state run once: its first k = floor(Ls / q) * q ids (host.shared_prefix_plan; k = 0 keeps nothing to reuse)
        from zero states -> {"state", "state_ids", "plan", "snapshot" (the four states after k ids), "hidden" (those k rows),
        "calls", "call_lens", "call_ms", "ms"}. The caller keeps it; nothing is cached here."""
        t0 = time.perf_counter()
        ids = self.state_ids(state)
        plan = host.shared_prefix_plan(len(ids), self.S, self.q, self.qmin)
        k = plan["shared_tokens"]
        call_ms: list = []
        call_lens: list = []
        state_nd = self.to_state(self.zero_arrays())
        h = await self.run_plan(ids[:k], 0, state_nd, call_ms, call_lens) if k else np.empty((0, self.d), np.float16)
        snap = self.snapshot(state_nd)
        out = {"state": state, "state_ids": ids, "plan": plan, "snapshot": snap, "hidden": h, "calls": len(call_ms),
               "call_lens": call_lens, "call_ms": call_ms, "ms": (time.perf_counter() - t0) * 1e3}
        if trace is not None:
            trace.update({k_: v for k_, v in out.items() if k_ not in ("snapshot", "hidden", "state")})
        return out

    async def decide_prepared(self, prepared: dict, questions: dict, trace: dict | None = None, model: str | None = None) -> dict:
        """The request {state: prepared state, questions} answered on the prepared state: per question its row's ids from k
        on, from a copy of the kept states (the shared path's second half). The rows must start with the prepared state
        ids. latency_ms = these graph calls + the head (not the prepare)."""
        tr = trace if trace is not None else {}
        t0 = time.perf_counter()
        request = {"state": prepared["state"], "questions": questions}
        if model is not None:
            request["model"] = model
        b = host.build_rows(request, self.tok, delims=self.delims)
        if b["state_len"] != len(prepared["state_ids"]) or any(r["row_ids"][:b["state_len"]] != prepared["state_ids"] for r in b["rows"]):
            raise SystemExit("the questions' rows do not start with the prepared state's ids")
        host.graph_context_check(b["rows"], self.max_ctx, self.S, self.q, self.qmin)
        t1 = time.perf_counter()
        k = prepared["plan"]["shared_tokens"]
        call_ms: list = []
        call_lens: list = []
        hs = []
        for r in b["rows"]:
            st = self.to_state(self.branch_arrays(prepared["snapshot"], k) if k else self.zero_arrays())
            h = await self.run_plan(r["row_ids"][k:], k, st, call_ms, call_lens)
            hs.append(np.concatenate([prepared["hidden"], h]))
        t2 = time.perf_counter()
        ps = [self.probs(r, h) for r, h in zip(b["rows"], hs)]
        t3 = time.perf_counter()
        answers = host.to_answers([[float(x) for x in p] for p in ps], b["meta"])
        body = host.response(self.name, answers, b["input_tokens"], host.output_tokens(self.tok, answers), (t3 - t1) * 1e3)
        tr.update({"mode": "prepared", "S": self.S, "graph": self.shape, "state_len": b["state_len"],
                   "row_tokens": [len(r["row_ids"]) for r in b["rows"]], "input_tokens": b["input_tokens"], "calls": len(call_ms),
                   "padded_tokens": int(sum(call_lens)), "call_lens": call_lens, "graph_ms": float(sum(call_ms)),
                   "call_ms": call_ms, "host_ms": (t1 - t0) * 1e3, "decide_ms": (t3 - t1) * 1e3, "head_ms": (t3 - t2) * 1e3,
                   "shared_plan": prepared["plan"], "_hidden": hs, "_probs": ps, "_rows": b})
        return body

    async def warm_up(self) -> list[dict]:
        """Every call length of the plan (host.call_lengths: q, 2q, ..., cap; one length S for a static-S bundle), one
        call each of <|endoftext|> ids from zero states, the hidden read back and dropped -> [{"s", "ms"}]."""
        out = []
        for s in host.call_lengths(self.S, self.q, self.qmin):
            state = self.to_state(self.zero_arrays())
            t = time.perf_counter()
            await self.call(np.full(s, self.delims["pad"], np.int32), 0, state)
            out.append({"s": s, "ms": (time.perf_counter() - t) * 1e3})
        return out

    def probs(self, row: dict, h: np.ndarray) -> np.ndarray:
        return host.head_probs(host.head_logits(h[row["decide"]], h[row["opts"]], self.head), self.T)

    async def decide(self, request: dict, shared: bool = False, trace: dict | None = None) -> dict:
        tr = trace if trace is not None else {}
        t0 = time.perf_counter()
        b = host.build_rows(request, self.tok, delims=self.delims)
        host.graph_context_check(b["rows"], self.max_ctx, self.S, self.q, self.qmin)
        t1 = time.perf_counter()
        call_ms: list = []
        call_lens: list = []
        plan = None
        if shared:
            hs, plan = await self.hidden_shared(b["rows"], b["state_len"], call_ms, call_lens)
        else:
            hs = await self.hidden_direct(b["rows"], call_ms, call_lens)
        t2 = time.perf_counter()
        ps = [self.probs(r, h) for r, h in zip(b["rows"], hs)]
        t3 = time.perf_counter()
        answers = host.to_answers([[float(x) for x in p] for p in ps], b["meta"])
        body = host.response(self.name, answers, b["input_tokens"], host.output_tokens(self.tok, answers), (t3 - t1) * 1e3)
        t4 = time.perf_counter()
        T = [len(r["row_ids"]) for r in b["rows"]]
        tr.update({"mode": "shared" if shared else "direct", "S": self.S, "graph": self.shape, "state_len": b["state_len"],
                   "row_tokens": T, "input_tokens": b["input_tokens"], "calls": len(call_ms),
                   "padded_tokens": int(sum(call_lens)), "call_lens": call_lens,
                   "graph_ms": float(sum(call_ms)), "call_ms": call_ms, "host_ms": (t1 - t0) * 1e3,
                   "decide_ms": (t3 - t1) * 1e3, "head_ms": (t3 - t2) * 1e3, "answers_ms": (t4 - t3) * 1e3,
                   "shared_plan": plan, "_hidden": hs, "_probs": ps, "_rows": b})
        return body


def public(trace: dict) -> dict:
    return {k: v for k, v in trace.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- run
def cmd_run(args) -> int:
    request = json.loads(Path(args.request).read_text())
    e = Kev(args.bundle or BUNDLES[args.model], args.aimodelc, args.call_max, args.multiple)
    trace: dict = {}

    async def go():
        trace["load"] = await e.load()
        if args.warm:
            trace["warm_up"] = await e.warm_up()
        return await e.decide(request, shared=args.shared, trace=trace)

    resp = asyncio.run(go())
    Path(args.out).write_text(json.dumps(resp, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(resp, ensure_ascii=False))
    if args.trace:
        tr = public(trace)
        tr["probs"] = [[float(x) for x in p] for p in trace["_probs"]]
        tr["bundle"], tr["aimodelc"] = str(e.bundle), str(e.aimodelc)
        Path(args.trace).write_text(json.dumps(tr, indent=1) + "\n")
    return 0


# --------------------------------------------------------------------------- check: worker
def worker(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text())
    recs = {r["id"]: r for r in json.loads(Path(spec["fixtures"]).read_text())["records"]}
    e = Kev(Path(spec["bundle"]), Path(spec["aimodelc"]), spec.get("call_max"), spec.get("multiple"))
    gate = {tuple(k.split("|")): v for k, v in spec["gate"].items()}
    shared_gate = {tuple(k.split("|")): v for k, v in (spec.get("shared_gate") or {}).items()}
    npzs: dict = {}
    out = {"spec": spec, "pid": os.getpid(), "started": time.time(), "records": []}
    readout: dict[str, np.ndarray] = {}

    def gate_hidden(rid: str, q: int, which: dict | None = None) -> np.ndarray:
        path, key = (which or gate)[(rid, str(q))]
        z = npzs.setdefault(path, np.load(path))
        return z[f"{key}__hidden"]

    async def go():
        out["load"] = await e.load()
        first = None
        order = spec["records"] + [spec["records"][0]]
        for i, rid in enumerate(order):
            request = recs[rid]["request"]
            if i == len(spec["records"]):                     # the reset re-run: the first record again, direct
                tr: dict = {}
                await e.decide(request, shared=False, trace=tr)
                same = all(np.array_equal(a, b) for a, b in zip(first, tr["_hidden"]))
                out["reset_check"] = {"record": rid, "bit_equal": bool(same)}
                print(f"  [{os.getpid()}] reset re-run {rid}: bit-equal {same}", flush=True)
                continue
            modes = ("direct", "shared") if spec["shared"] else ("direct",)
            prepared_res = None
            if spec["shared"] and i % 2:
                modes = ("shared", "direct")                  # alternate which mode runs first
            res = {}
            for mode in modes:
                tr = {}
                body = await e.decide(request, shared=mode == "shared", trace=tr)
                res[mode] = (body, tr)
            body_d, tr_d = res["direct"]
            if first is None:
                first = tr_d["_hidden"]
            if spec.get("prepared") and "shared" in res:   # round 15: prepare once, then all questions, then one at a time
                pr = await e.prepare(request["state"])
                tr_all: dict = {}
                body_all = await e.decide_prepared(pr, request["questions"], trace=tr_all)
                singles = []
                for qid, q in request["questions"].items():
                    tr1: dict = {}
                    await e.decide_prepared(pr, {qid: q}, trace=tr1)
                    singles.append(tr1)
                prepared_res = (pr, body_all, tr_all, singles)
            b = tr_d["_rows"]
            rows = []
            for k, (r, h) in enumerate(zip(b["rows"], tr_d["_hidden"])):
                g = gate_hidden(rid, k)
                row = {"qid": r["qid"], "q": k, "T": len(r["row_ids"]), "calls": len(e.plan(len(r["row_ids"]))),
                       "hidden_bit_equal_gate": bool(g.shape == h.shape and np.array_equal(g, h)),
                       "hidden_max_abs_diff_gate": float(np.max(np.abs(g.astype(np.float32) - h.astype(np.float32))))
                       if g.shape == h.shape else None,
                       "finite": bool(np.isfinite(h.astype(np.float32)).all()), "all_zero": bool(not np.any(h)),
                       "p_host": [float(x) for x in tr_d["_probs"][k]]}
                readout[f"{rid}__{k}"] = np.concatenate([h[[r["decide"]]], h[r["opts"]]], 0)
                if "shared" in res:
                    hs = res["shared"][1]["_hidden"][k]
                    ps = res["shared"][1]["_probs"][k]
                    row.update({"shared_hidden_bit_equal_direct": bool(np.array_equal(hs, h)),
                                "shared_hidden_max_abs_diff": float(np.max(np.abs(hs.astype(np.float32) - h.astype(np.float32)))),
                                "p_host_shared": [float(x) for x in ps],
                                "shared_p_bit_equal_direct": bool(np.array_equal(ps, tr_d["_probs"][k]))})
                    if (rid, str(k)) in shared_gate:      # round 15: the shared gate of the same asset, same plan
                        gs = gate_hidden(rid, k, shared_gate)
                        row["shared_hidden_bit_equal_shared_gate"] = bool(gs.shape == hs.shape and np.array_equal(gs, hs))
                    if prepared_res:                       # round 15: prepared = shared, bit for bit
                        _, _, tr_all, singles = prepared_res
                        hp, pp = tr_all["_hidden"][k], tr_all["_probs"][k]
                        p1 = singles[k]["_probs"][0]
                        row.update({"prepared_hidden_bit_equal_shared": bool(np.array_equal(hp, hs)),
                                    "prepared_p_bit_equal_shared": bool(np.array_equal(pp, ps)),
                                    "prepared_single_p_bit_equal_shared": bool(np.array_equal(p1, ps)),
                                    "p_host_prepared": [float(x) for x in pp]})
                rows.append(row)
            item = {"id": rid, "source": recs[rid]["source"], "order": list(modes), "rows": rows,
                    "state_len": b["state_len"], "input_tokens": b["input_tokens"],
                    "direct": {"response": body_d, **{k: tr_d[k] for k in ("calls", "padded_tokens", "graph_ms", "decide_ms", "head_ms", "host_ms")}}}
            if "shared" in res:
                body_s, tr_s = res["shared"]
                item["shared"] = {"response": body_s, "plan": tr_s["shared_plan"],
                                  **{k: tr_s[k] for k in ("calls", "padded_tokens", "graph_ms", "decide_ms", "head_ms", "host_ms")}}
            if prepared_res:
                pr, body_all, tr_all, singles = prepared_res
                item["prepared"] = {"prepare_calls": pr["calls"], "prepare_ms": pr["ms"], "k": pr["plan"]["shared_tokens"],
                                    "calls": tr_all["calls"], "graph_ms": tr_all["graph_ms"],
                                    "answers_equal_shared": json.dumps(body_all["answers"]) == json.dumps(res["shared"][0]["answers"]),
                                    "usage_equal_shared": body_all["usage"] == res["shared"][0]["usage"]}
            out["records"].append(item)
            ok = all(x["hidden_bit_equal_gate"] for x in rows)
            sh = all(x.get("shared_hidden_bit_equal_direct", True) for x in rows)
            print(f"  [{os.getpid()}] {rid}: {len(rows)} rows, hidden = gate {ok}, shared = direct {sh}, "
                  f"direct {tr_d['graph_ms']:.0f} ms / {tr_d['calls']} calls"
                  + (f", shared {res['shared'][1]['graph_ms']:.0f} ms / {res['shared'][1]['calls']} calls" if "shared" in res else ""),
                  flush=True)

    asyncio.run(go())
    out["finished"] = time.time()
    out["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    prefix = Path(spec["out"])
    np.savez(prefix.with_suffix(".npz"), **readout)
    prefix.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


# --------------------------------------------------------------------------- check: driver
def other_gpu_processes() -> list[str]:
    ps = subprocess.run(["ps", "-axo", "pid=,etime=,command="], capture_output=True, text=True).stdout
    me = os.getpid()
    skip = ("/.local/bin/claude", "claude --", "shell-snapshots", "until grep", "zsh -c", "/bin/zsh", "/bin/bash")
    return [ln.strip()[:200] for ln in ps.splitlines()
            if OTHER_GPU.search(ln) and not ln.strip().startswith(f"{me} ") and not any(s in ln for s in skip)
            and "decide.py worker" not in ln]


def lock_state() -> dict:
    p = gpu_lock()
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {"path": str(p), "bytes": st.st_size, "content": p.read_text()[:300],
            "mtime": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds")}


def wait_for_lock_release(max_minutes: float = 20.0) -> dict:
    """Another session's timing window (a non-empty _GPU_LOCK) is not disturbed: wait for it to clear before the
    next GPU process starts (checked before every shard). -> what was waited for."""
    t0 = time.monotonic()
    seen = None
    while True:
        st = lock_state()
        if not st.get("bytes"):
            return {"waited_seconds": round(time.monotonic() - t0, 1), "held_by": seen}
        seen = st.get("content")
        if time.monotonic() - t0 > max_minutes * 60:
            raise SystemExit(f"_GPU_LOCK held for {max_minutes} min by: {seen!r}; not starting GPU work")
        print(f"  _GPU_LOCK held ({seen!r}); waiting", flush=True)
        time.sleep(20)


def bar_summary(items: list[dict], key: str) -> dict:
    """readout_gate.summarize + verdict on one set of p (items carry the oracle's probs / argmax / near_tie)."""
    runs = []
    for it in items:
        p = np.asarray(it[key], np.float64)
        po = np.asarray(it["probs_oracle"], np.float64)
        dp = np.abs(p - po)
        runs.append({"argmax_equal": int(p.argmax()) == it["argmax_oracle"], "near_tie": it["near_tie"],
                     "max_abs_dp": float(dp.max()), "mean_abs_dp": float(dp.mean()), "row": it["row"]})
    far = [r for r in runs if not r["near_tie"]]
    near = [r for r in runs if r["near_tie"]]
    worst = max(runs, key=lambda r: r["max_abs_dp"])
    s = {"questions": len(runs), "questions_non_near_tie": len(far),
         "argmax_equal_non_near_tie": sum(r["argmax_equal"] for r in far), "near_tie_questions": len(near),
         "argmax_equal_near_tie": sum(r["argmax_equal"] for r in near),
         "max_abs_dp": worst["max_abs_dp"], "worst_row": worst["row"],
         "mean_of_run_mean_abs_dp": float(np.mean([r["mean_abs_dp"] for r in runs]))}
    s["pass"] = (s["argmax_equal_non_near_tie"] == s["questions_non_near_tie"] and s["max_abs_dp"] <= BAR["max_abs_dp"]
                 and s["mean_of_run_mean_abs_dp"] <= BAR["mean_of_run_mean_abs_dp"])
    return s


def compare_response(mine: dict, ref: dict) -> dict:
    """Every answer field, the main values, usage; 4-decimal values as the bodies carry them."""
    fields, diffs, main_eq, main_n = 0, [], 0, 0
    by_type: dict = {}
    for qid, a in ref["answers"].items():
        b = mine["answers"].get(qid)
        main = {"noul": "noul", "choice": "choice", "score": "score"}[a["type"]]
        t = by_type.setdefault(a["type"], {"questions": 0, "main_equal": 0})
        t["questions"] += 1
        main_n += 1
        if b is None or b["type"] != a["type"]:
            diffs.append({"qid": qid, "field": "missing or type"})
            continue
        if a[main] == b[main]:
            main_eq += 1
            t["main_equal"] += 1
        pairs = [(k, a[k], b[k]) for k in ("noul", "score", "confidence") if k in a]
        pairs += [(f"p[{k}]", v, b["probabilities"][k]) for k, v in (a.get("probabilities") or {}).items()]
        for name, x, y in pairs:
            fields += 1
            if x != y:
                diffs.append({"qid": qid, "field": name, "oracle": x, "mine": y, "abs_diff": round(abs(x - y), 4)})
        if a["type"] == "choice" and a["choice"] != b["choice"]:
            diffs.append({"qid": qid, "field": "choice", "oracle": a["choice"], "mine": b["choice"]})
        if a["type"] == "score" and a["legend"] != b["legend"]:
            diffs.append({"qid": qid, "field": "legend"})
    num = [d for d in diffs if "abs_diff" in d]
    return {"answers_equal": json.dumps(mine["answers"]) == json.dumps(ref["answers"]), "questions": main_n,
            "main_values_equal": main_eq, "by_type": by_type, "fields": fields, "fields_equal": fields - len(num),
            "fields_off_by_1e-4": sum(d["abs_diff"] <= 0.0001 for d in num),
            "fields_off_more": sum(d["abs_diff"] > 0.0001 for d in num),
            "max_field_abs_diff": max((d["abs_diff"] for d in num), default=0.0),
            "input_tokens_equal": mine["usage"]["input_tokens"] == ref["usage"]["input_tokens"],
            "output_tokens_equal": mine["usage"]["output_tokens"] == ref["usage"]["output_tokens"],
            "model": [ref["model"], mine["model"]], "diffs": diffs}


def cmd_check(args) -> int:
    model = args.model
    bundle = Path(args.bundle or BUNDLES[model]).expanduser().resolve()
    e = Kev(bundle, args.aimodelc, args.call_max, args.multiple)   # resolves and checks the bundle; no graph load here
    oracle_dir = Path(args.oracle).expanduser().resolve() if args.oracle else (
        ORACLES[model] / "heldout" if args.rows == "heldout" else ORACLES[model])
    oracle_path = oracle_dir / "records_oracle.json"
    gate_path = Path(args.gate_transcript or GATES[(model, args.rows)]).expanduser().resolve()
    fx_path = FIXTURES[args.rows]
    oracle = {r["id"]: r for r in json.loads(oracle_path.read_text())["records"]}
    gate_t = json.loads(gate_path.read_text())
    if gate_t["bundle"]["name"] != e.name:
        raise SystemExit(f"{gate_path} is a gate of {gate_t['bundle']['name']}, not {e.name}")
    fx = json.loads(fx_path.read_text())["records"]
    ids = [r["id"] for r in fx if r["id"] in oracle]
    if args.records:
        want = args.records.split(",")
        ids = [i for i in ids if i in want]
    if len(ids) != (len(oracle) if not args.records else len(ids)):
        raise SystemExit("the fixture file and the oracle do not hold the same records")
    gate_rows = {(r["id"], r["q"]): r for r in gate_t["runs"]}

    def npz_map(t: dict) -> dict:
        out = {}
        for r in t["runs"]:
            npz = next(p["npz"] for p in t["processes"] if p.get("shard") == r["shard"])
            out[f"{r['id']}|{r['q']}"] = [npz, r["npz_key"]]
        return out
    gate_npz = npz_map(gate_t)
    shared_gate_npz, sgate_t = {}, None
    if args.shared_gate_transcript:   # round 15: readout_gate.py --shared of the same asset (the same shared plan)
        sgate_t = json.loads(Path(args.shared_gate_transcript).read_text())
        if sgate_t["bundle"]["name"] != e.name or sgate_t.get("mode") != "shared":
            raise SystemExit(f"{args.shared_gate_transcript}: not a shared gate of {e.name}")
        shared_gate_npz = npz_map(sgate_t)
    tag = args.tag or f"e2e_{model}_{args.rows}"
    work = WORK / tag
    work.mkdir(parents=True, exist_ok=True)
    t_start = datetime.now().astimezone().isoformat(timespec="seconds")
    others0, lock0 = other_gpu_processes(), lock_state()
    print(f"asset digest {e.aimodelc.name} ...", flush=True)
    t0 = time.monotonic()
    digest = tree_digest(e.aimodelc)
    print(f"  {digest['tree_sha256'][:12]} vs the gate's {gate_t['bundle']['aimodelc']['tree_sha256'][:12]} "
          f"({time.monotonic() - t0:.1f} s)", flush=True)
    n = math.ceil(len(ids) / RECORDS_PER_PROCESS)
    size = math.ceil(len(ids) / n)
    parts = [ids[i:i + size] for i in range(0, len(ids), size)]
    shards, lock_waits = [], []
    t_gpu = time.monotonic()
    for k, part in enumerate(parts):
        w = wait_for_lock_release()
        if w["held_by"]:
            lock_waits.append({"shard": k, **w})
        spec = {"model": model, "bundle": str(bundle), "aimodelc": str(e.aimodelc), "fixtures": str(fx_path),
                "records": part, "shared": bool(args.shared), "out": str(work / f"shard_{k:02d}"),
                "call_max": args.call_max, "multiple": args.multiple, "prepared": bool(args.prepared),
                "gate": {f"{i}|{q}": gate_npz[f"{i}|{q}"] for i in part for q in range(len(oracle[i]["questions"]))},
                "shared_gate": {key: v for key, v in shared_gate_npz.items() if key.split("|")[0] in part}}
        sp = work / f"shard_{k:02d}.spec.json"
        sp.write_text(json.dumps(spec, indent=1) + "\n")
        print(f"[{tag}] shard {k:02d}: {len(part)} records + reset re-run", flush=True)
        t1 = time.monotonic()
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--spec", str(sp)])
        js = work / f"shard_{k:02d}.json"
        if proc.returncode != 0 or not js.exists():
            raise SystemExit(f"shard {k:02d}: worker failed (exit {proc.returncode})")
        rec = json.loads(js.read_text())
        rec["process_wall_seconds"] = time.monotonic() - t1
        shards.append(rec)
    gpu_seconds = time.monotonic() - t_gpu
    others1, lock1 = other_gpu_processes(), lock_state()

    # score: the gate's torch head on the direct readout rows, the transcript's p, the oracle
    import torch
    from parity_decoder_torch import KevHead
    kh = KevHead(e.head_dir, e.d)
    items, records = [], []
    for sh in shards:
        z = np.load(Path(sh["spec"]["out"]).with_suffix(".npz"))
        for rec in sh["records"]:
            o = oracle[rec["id"]]
            for row in rec["rows"]:
                q = o["questions"][row["q"]]
                g = gate_rows[(rec["id"], row["q"])]
                x = torch.from_numpy(z[f"{rec['id']}__{row['q']}"].astype(np.float32))
                lt = kh.logits_T(x, 0, list(range(1, x.shape[0])))
                p_t = torch.softmax(lt, -1).double().numpy()
                p_g = np.asarray(g["probs"], np.float64)
                p_h = np.asarray(row["p_host"], np.float64)
                row["p_torch_head_bit_equal_gate"] = bool(np.array_equal(p_t, p_g))
                row["p_host_bit_equal_gate"] = bool(np.array_equal(p_h, p_g))
                row["p_host_max_abs_dp_gate"] = float(np.abs(p_h - p_g).max())
                it = {"row": f"{rec['id']}:q{row['q']}", "probs_oracle": q["probs"], "near_tie": bool(q["near_tie"]),
                      "argmax_oracle": q["keys"].index(q["argmax"]), "p_host": row["p_host"], "p_torch": p_t.tolist(),
                      "p_gate": g["probs"], "type": q["type"]}
                row["argmax_equal_oracle"] = int(np.argmax(p_h)) == it["argmax_oracle"]
                row["type"], row["near_tie"] = q["type"], bool(q["near_tie"])
                if "p_host_shared" in row:
                    it["p_shared"] = row["p_host_shared"]
                    row["shared_max_abs_dp"] = float(np.abs(np.asarray(row["p_host_shared"]) - p_h).max())
                items.append(it)
            rec["response_vs_oracle"] = compare_response(rec["direct"]["response"], o["response"])
            meta = host.to_record(host.validate_request(json.loads(json.dumps(
                next(r for r in fx if r["id"] == rec["id"])["request"]))))[1]
            torch_answers = host.to_answers([[float(v) for v in items[-len(rec["rows"]) + k]["p_torch"]]
                                             for k in range(len(rec["rows"]))], meta)
            rec["torch_head_answers_equal_host"] = json.dumps(torch_answers) == json.dumps(rec["direct"]["response"]["answers"])
            if "shared" in rec:
                rec["shared_response_equal_direct"] = (json.dumps(rec["shared"]["response"]["answers"])
                                                       == json.dumps(rec["direct"]["response"]["answers"])
                                                       and rec["shared"]["response"]["usage"] == rec["direct"]["response"]["usage"])
            records.append(rec)
    rows = [r for rec in records for r in rec["rows"]]
    bars = {"host": bar_summary(items, "p_host"), "torch_head": bar_summary(items, "p_torch"),
            "gate_transcript_p": bar_summary(items, "p_gate")}
    if args.shared:
        bars["shared_host"] = bar_summary(items, "p_shared")
    gs = gate_t["summary"]
    gate_bar = {k: gs[k] for k in ("questions_non_near_tie", "argmax_equal_non_near_tie", "near_tie_questions",
                                   "argmax_equal_near_tie", "max_abs_dp", "mean_of_run_mean_abs_dp")}
    rv = [rec["response_vs_oracle"] for rec in records]
    summary = {
        "records": len(records), "rows": len(rows),
        "i_hidden_bit_equal_gate": sum(r["hidden_bit_equal_gate"] for r in rows),
        "i_hidden_max_abs_diff_gate": max(r["hidden_max_abs_diff_gate"] or 0.0 for r in rows),
        "i_p_torch_head_bit_equal_gate": sum(r["p_torch_head_bit_equal_gate"] for r in rows),
        "i_p_host_bit_equal_gate": sum(r["p_host_bit_equal_gate"] for r in rows),
        "i_p_host_max_abs_dp_gate": max(r["p_host_max_abs_dp_gate"] for r in rows),
        "finite_all": all(r["finite"] for r in rows), "all_zero_rows": sum(r["all_zero"] for r in rows),
        "reset_bit_equal_all_processes": all(sh.get("reset_check", {}).get("bit_equal") for sh in shards),
        "iii_records_answers_equal_oracle": sum(x["answers_equal"] for x in rv),
        "iii_questions": sum(x["questions"] for x in rv), "iii_main_values_equal": sum(x["main_values_equal"] for x in rv),
        "iii_by_type": {t: {k: sum(x["by_type"].get(t, {}).get(k, 0) for x in rv) for k in ("questions", "main_equal")}
                        for t in ("noul", "choice", "score")},
        "iii_argmax_equal_by_type": {t: {"questions": sum(r["type"] == t for r in rows),
                                         "argmax_equal": sum(r["type"] == t and r["argmax_equal_oracle"] for r in rows),
                                         "near_tie_questions": sum(r["type"] == t and r["near_tie"] for r in rows),
                                         "argmax_differs": [f"{rec['id']}:q{r['q']}" for rec in records for r in rec["rows"]
                                                            if r["type"] == t and not r["argmax_equal_oracle"]]}
                                     for t in ("noul", "choice", "score")},
        "iii_fields": sum(x["fields"] for x in rv), "iii_fields_equal": sum(x["fields_equal"] for x in rv),
        "iii_fields_off_by_1e-4": sum(x["fields_off_by_1e-4"] for x in rv),
        "iii_fields_off_more": sum(x["fields_off_more"] for x in rv),
        "iii_max_field_abs_diff": max(x["max_field_abs_diff"] for x in rv),
        "iii_input_tokens_equal": sum(x["input_tokens_equal"] for x in rv),
        "iii_output_tokens_equal": sum(x["output_tokens_equal"] for x in rv),
        "iii_model_field": "the oracle body echoes the request's model; ours is the bundle name",
        "torch_head_answers_equal_host_records": sum(r["torch_head_answers_equal_host"] for r in records),
        "iv_bar": bars, "iv_gate_transcript_summary": gate_bar,
        "iv_torch_head_equals_gate": all(bars["torch_head"][k] == gate_bar[k] for k in gate_bar),
    }
    if args.shared:
        sg = [r for r in rows if "shared_hidden_bit_equal_shared_gate" in r]
        summary.update({
            "ii_dynamic": e.dynamic, "ii_graph": e.shape,
            "ii_shared_hidden_max_abs_diff_direct": max(r["shared_hidden_max_abs_diff"] for r in rows),
            "ii_shared_hidden_bit_equal_shared_gate": (f"{sum(r['shared_hidden_bit_equal_shared_gate'] for r in sg)}/{len(sg)}"
                                                       if sgate_t else None),
            "ii_shared_hidden_bit_equal_direct": sum(r["shared_hidden_bit_equal_direct"] for r in rows),
            "ii_shared_p_bit_equal_direct": sum(r["shared_p_bit_equal_direct"] for r in rows),
            "ii_shared_max_abs_dp": max(r["shared_max_abs_dp"] for r in rows),
            "ii_shared_responses_equal_direct": sum(r["shared_response_equal_direct"] for r in records),
            "ii_calls": {"direct": sum(r["direct"]["calls"] for r in records), "shared": sum(r["shared"]["calls"] for r in records)},
            "ii_graph_seconds": {"direct": sum(r["direct"]["graph_ms"] for r in records) / 1e3,
                                 "shared": sum(r["shared"]["graph_ms"] for r in records) / 1e3}})
    summary["pass_i"] = (summary["i_hidden_bit_equal_gate"] == len(rows) and summary["i_p_torch_head_bit_equal_gate"] == len(rows)
                         and summary["finite_all"] and summary["all_zero_rows"] == 0 and summary["reset_bit_equal_all_processes"])
    pr_rows = [r for r in rows if "prepared_hidden_bit_equal_shared" in r]
    if pr_rows:   # round 15: the prepared state = the shared request, bit for bit
        summary.update({"v_prepared_rows": len(pr_rows),
                        "v_prepared_hidden_bit_equal_shared": sum(r["prepared_hidden_bit_equal_shared"] for r in pr_rows),
                        "v_prepared_p_bit_equal_shared": sum(r["prepared_p_bit_equal_shared"] for r in pr_rows),
                        "v_prepared_single_question_p_bit_equal_shared": sum(r["prepared_single_p_bit_equal_shared"] for r in pr_rows)})
        summary["pass_v"] = all(r["prepared_hidden_bit_equal_shared"] and r["prepared_p_bit_equal_shared"]
                                and r["prepared_single_p_bit_equal_shared"] for r in pr_rows)
    if args.shared and not e.dynamic:
        summary["pass_ii"] = summary["ii_shared_max_abs_dp"] <= SHARED_MAX_ABS_DP
    elif args.shared:   # round 15: other call cuts, other last bits -> the shared p on the oracle's bar
        summary["pass_ii"] = bool(bars["shared_host"]["pass"] and (not sg or all(r["shared_hidden_bit_equal_shared_gate"] for r in sg)))
    doc = {"schema": "kev-decide-e2e/1",
           "what": "decide.py from the raw request (host.build_rows, the AOT graph direct and shared, the float64 host "
                   "head, host.to_answers / response) vs the round-2/4 readout gate of the same asset and the author's oracle",
           "model": model, "rows_set": args.rows, "started": t_start,
           "bundle": {"path": str(bundle), "name": e.name, "metadata_sha256": sha256_file(bundle / "metadata.json"),
                      "head_sha256": {f.name: sha256_file(f) for f in sorted(e.head_dir.iterdir())},
                      "tokenizer_json_sha256": sha256_file(bundle / "tokenizer" / "tokenizer.json")},
           "aimodelc": {**digest, "equals_gate_asset": digest["tree_sha256"] == gate_t["bundle"]["aimodelc"]["tree_sha256"]},
           "gate_transcript": {"path": str(gate_path), "sha256": sha256_file(gate_path), "head": gate_t.get("head")},
           "shared_gate_transcript": ({"path": str(args.shared_gate_transcript),
                                       "sha256": sha256_file(Path(args.shared_gate_transcript))} if sgate_t else None),
           "graph": e.shape,
           "oracle": {"path": str(oracle_path), "sha256": sha256_file(oracle_path)},
           "fixtures": {"path": str(fx_path), "sha256": sha256_file(fx_path)},
           "scripts": {"decide.py": sha256_file(Path(__file__).resolve()), "host.py": sha256_file(HERE / "host.py")},
           "gpu": {"other_gpu_processes_start": others0, "other_gpu_processes_end": others1, "gpu_lock_start": lock0,
                   "gpu_lock_end": lock1, "lock_waits": lock_waits, "seconds": gpu_seconds,
                   "note": "contended reference times (no _GPU_LOCK window of our own); a held lock is waited out"},
           "processes": [{"shard": Path(sh["spec"]["out"]).name, "pid": sh["pid"], "records": len(sh["records"]),
                          "load": sh["load"], "reset_check": sh.get("reset_check"), "max_rss_bytes": sh["max_rss_bytes"],
                          "process_wall_seconds": sh["process_wall_seconds"]} for sh in shards],
           "summary": summary, "records": records,
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("iv_bar",)}, indent=None)[:2500])
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("max_abs_dp", "mean_of_run_mean_abs_dp", "pass")}
                      for k, v in bars.items()}))
    print(f"wrote {args.out}")
    ok = summary["pass_i"] and summary.get("pass_ii", True) and summary.get("pass_v", True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "check"):
        a = sub.add_parser(name)
        a.add_argument("--model", choices=sorted(BUNDLES), required=True)
        a.add_argument("--bundle", help="bundle directory (default: the lane's fp16 pf16 bundle of --model)")
        a.add_argument("--aimodelc", help="AOT asset (default: <bundles>/../bundles_aotc/<name>.h16c.aimodelc)")
        a.add_argument("--shared", action="store_true", help="run: the shared prefix; check: also the shared prefix")
        a.add_argument("--call-max", type=int, help="round 15: the longest call (default: metadata query_len_call_max)")
        a.add_argument("--multiple", type=int, help="round 15: every call a multiple of this (default: metadata "
                                                    "query_len_multiple)")
        a.add_argument("--out", required=True)
        if name == "run":
            a.add_argument("--request", required=True)
            a.add_argument("--trace")
            a.add_argument("--warm", action="store_true",
                           help="round 15: run every call length of the plan once before the request (Kev.warm_up)")
        else:
            a.add_argument("--rows", choices=("all", "heldout"), default="all")
            a.add_argument("--oracle", help="oracle directory (default: the model's, heldout/ for --rows heldout)")
            a.add_argument("--gate-transcript", help="the readout gate transcript of the same asset and rows")
            a.add_argument("--shared-gate-transcript",
                           help="round 15: readout_gate.py --shared of the same asset (its shared hidden rows == ours)")
            a.add_argument("--prepared", action="store_true",
                           help="round 15 (with --shared): also prepare(state) once and answer the questions on it, all "
                                "together and one at a time; both must equal the shared run bit for bit")
            a.add_argument("--records", help="comma-separated record ids (default: every record)")
            a.add_argument("--tag", help="work directory name under <lane>/host/")
    w = sub.add_parser("worker")
    w.add_argument("--spec", required=True)
    args = ap.parse_args()
    if args.cmd == "worker":
        return worker(Path(args.spec))
    return cmd_run(args) if args.cmd == "run" else cmd_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
