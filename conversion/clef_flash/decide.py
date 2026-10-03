#!/usr/bin/env python3
"""clef-flash Python reference: a SystemOne-shaped request (+ an optional image) -> the SystemOne-shaped response, on Core AI.

The whole read-out, in the order a Swift host repeats it (every step is a function another file gated):

    request JSON ──host.build_ids──> ids, question / option spans, the decoder's static inputs
    image ──host.preprocess(tile)──> patches [4 G^2, 1536] ──tower (AOT)──> image_embeds [G^2, 4096]
    decoder (AOT, one static-S function `main`): fresh zero states, ceil(T / S) calls of S ids
        (position_ids 0..kS+S-1, the last call padded with <|endoftext|>) -> hidden [T, 4096] fp16
    clef_head.head_inputs: span-mean rows, the last-token row, lexical = the mean of the lm_head table's
        fp16 rows (lm_head_fp16.bin) over each option span, membership, types, masks, padded to the head's
        bucket (T 512 / 1024 / 2048 / 4096, Q 16, O 128)
    head (AOT, function t<bucket>) -> one logit per option -> per question an fp32 softmax
    -> {"model", "answers": {question_id: answer}, "usage": {"input_tokens": T, "output_tokens": 0}}

The answer of each question is the checkpoint's own `systemone_answer()` written out (joint_schema_model.py
at the pinned revision): noul -> p(true); choice -> the argmax id, its p, every p; score -> sum(level * p),
max p, the legend, every p; all rounded to 4 decimals. Images: one image per request, sent at a fixed grid
(`--grid 256` = 8 x 8 = 64 image tokens, `448` = 14 x 14 = 196); the request's own `images` field is not
read (pass the file with `--image`). Every asset is the AOT `.aimodelc` for the Mac GPU, loaded with
`SpecializationOptions.default()` (no JIT).

    cd conversion/clef_flash
    PY=<coreai-models venv>/bin/python
    $PY decide.py run --request req.json [--image x.png --grid 448] --out resp.json [--trace trace.json]
    $PY decide.py check --runs own_t01:text,img_04:g448,... --out e2e.json   # vs the oracle's systemone_response

Asset defaults are the lane's ($ZOO_WORK_ROOT/_clefflash/exports/...): the fp16 S=64 decoder bundle and
its AOT asset, the head bucket bundle (its metadata names its AOT asset), lm_head_fp16.bin, the two
tower AOT assets.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import inspect
import json
import os
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
EXPORTS = LANE / "exports"
DEFAULTS = {
    "bundle": EXPORTS / "bundles" / "clef_flash_decode_fp16_pf64",
    "decoder_aot": EXPORTS / "bundles_aotc" / "clef_flash_decode_fp16_pf64.h16c.aimodelc",
    "head": EXPORTS / "head" / "clef_flash_head_bucket_fp16w32",
    "table": EXPORTS / "host" / "lm_head_fp16.bin",
    "tower_g256": EXPORTS / "clef_flash_g256_vision_fp16w32_aotc" / "clef_flash_g256_vision_fp16w32.h16c.aimodelc",
    "tower_g448": EXPORTS / "clef_flash_g448_vision_fp16w32_aotc" / "clef_flash_g448_vision_fp16w32.h16c.aimodelc",
}
GRIDS = {256: 8, 448: 14}
VOCAB, HIDDEN, PAD_ID = 248320, 4096, 248044


async def maybe(x):
    return await x if inspect.isawaitable(x) else x


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- the response
def systemone_answer(question: dict, probabilities: dict) -> dict:
    """The checkpoint's systemone_answer(): per-option probabilities of one question -> its SystemOne answer."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = [str(option) for option in question["criteria"]]
        choice = max(options, key=probabilities.__getitem__)
        return {"type": "choice", "choice": choice, "confidence": round(probabilities[choice], 4),
                "probabilities": {option: round(probabilities[option], 4) for option in options}}
    levels = [str(index) for index in range(len(question["criteria"]))]
    return {"type": "score",
            "score": round(sum(index * probabilities[level] for index, level in enumerate(levels)), 4),
            "confidence": round(max(probabilities[level] for level in levels), 4),
            "legend": dict(zip(levels, question["criteria"])),
            "probabilities": {level: round(probabilities[level], 4) for level in levels}}


def response(request: dict, questions: list[dict], probs: list[np.ndarray], tokens: int) -> dict:
    answers = {q["question_id"]: systemone_answer(request["questions"][q["question_id"]],
                                                  dict(zip(q["option_ids"], p.astype(np.float32).tolist())))
               for q, p in zip(questions, probs)}
    return {"model": request["model"], "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}


# --------------------------------------------------------------------------- the engine
class ClefFlash:
    """The four Core AI assets + the host files, loaded once; `decide()` answers one request."""

    def __init__(self, bundle: Path, decoder_aot: Path, head: Path, table: Path, towers: dict[int, Path]):
        from tokenizers import Tokenizer
        self.bundle = Path(bundle)
        self.meta = json.loads((self.bundle / "metadata.json").read_text())
        self.S = int(self.meta["language"]["prefill_chunk"])
        self.max_ctx = int(self.meta["language"]["max_context_length"])
        self.n_image_max = int(self.meta["vision"]["n_image_max"])
        self.tok = Tokenizer.from_file(str(self.bundle / "tokenizer" / "tokenizer.json"))
        self.decoder_aot = Path(decoder_aot)
        self.head_dir = Path(head)
        self.head_meta = json.loads((self.head_dir / "metadata.json").read_text())
        hm = self.head_meta["head"]
        if hm["shape"] != "bucket":
            raise SystemExit(f"{head}: this reference runs the bucket head (got {hm['shape']})")
        self.buckets = [(f, int(t)) for f, t in hm["buckets"]]
        self.head_aot = Path(self.head_meta["aot"]["aimodelc"])
        self.table_path = Path(table)
        self.table = np.memmap(self.table_path, dtype="<f2", mode="r", shape=(VOCAB, HIDDEN))
        self.tower_paths = {g: Path(p) for g, p in towers.items() if p}
        self.loaded = False

    async def load(self) -> dict:
        import coreai.runtime as rt
        self.rt = rt
        opts = rt.SpecializationOptions.default()
        t = {}
        t0 = time.perf_counter()
        dm = await maybe(rt.AIModel.load(str(self.decoder_aot), opts))
        self.decoder = await maybe(dm.load_function("main"))
        d = self.decoder.desc
        self.state_desc = {n: ([int(x) for x in d.state_descriptor(n).shape], str(d.state_descriptor(n).dtype).split(".")[-1])
                           for n in d.state_names}
        t["decoder_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        hmodel = await maybe(rt.AIModel.load(str(self.head_aot), opts))
        self.head_fns = {f: await maybe(hmodel.load_function(f)) for f, _ in self.buckets}
        t["head_s"] = time.perf_counter() - t0
        self.towers = {}
        for g, p in self.tower_paths.items():
            t0 = time.perf_counter()
            tm = await maybe(rt.AIModel.load(str(p), opts))
            self.towers[g] = await maybe(tm.load_function(tm.function_names[0]))
            t[f"tower_g{g}_s"] = time.perf_counter() - t0
        self.loaded = True
        return t

    def nd(self, a):
        return self.rt.NDArray(np.ascontiguousarray(a))

    async def decide(self, request: dict, image=None, grid: int | None = None, trace: dict | None = None) -> dict:
        import host
        from clef_head import head_inputs, question_probs
        tr = trace if trace is not None else {}
        t_all = time.perf_counter()
        G = None
        emb = np.zeros((self.n_image_max, HIDDEN), np.float16)
        if image is not None:
            if grid not in GRIDS or grid not in self.towers:
                raise SystemExit(f"--grid must be one of {sorted(self.towers)} with an image")
            G = GRIDS[grid]
            t0 = time.perf_counter()
            patches = host.preprocess(image, grid)
            t1 = time.perf_counter()
            res = await maybe(self.towers[grid](inputs={"patches": self.nd(patches)}))
            tower_out = np.asarray(res["image_embeds"].numpy()).astype(np.float32)
            tr["tower"] = {"grid": grid, "preprocess_ms": (t1 - t0) * 1e3, "tower_ms": (time.perf_counter() - t1) * 1e3,
                           "rows": int(tower_out.shape[0])}
            if tower_out.shape != (G * G, HIDDEN):
                raise SystemExit(f"tower output {tower_out.shape} != {(G * G, HIDDEN)}")
            emb[:G * G] = tower_out.astype(np.float16)
            tr["_image_embeds"] = tower_out
        t0 = time.perf_counter()
        b = host.build_ids(request, self.tok, grid=(G, G) if G else None, n_image_max=self.n_image_max)
        ids, qs, st = b["ids"], b["questions"], b["static"]
        T = len(ids)
        n_calls = -(-T // self.S)
        if n_calls * self.S > self.max_ctx:
            raise SystemExit(f"{T} tokens (padded {n_calls * self.S}) > the decoder's context {self.max_ctx}")
        tr["host_ids_ms"] = (time.perf_counter() - t0) * 1e3
        tr["tokens"], tr["ids"], tr["questions"] = T, ids, qs
        ids_p = np.full(n_calls * self.S, PAD_ID, np.int32)
        ids_p[:T] = st["input_ids"]
        static = {"image_embeds": self.nd(emb), "image_rc": self.nd(st["image_rc"]),
                  "rope_shift_start": self.nd(st["rope_shift_start"]), "rope_shift_amount": self.nd(st["rope_shift_amount"])}
        state = {n: self.nd(np.zeros([self.max_ctx if s < 0 else s for s in shape], np.dtype(dt)))
                 for n, (shape, dt) in self.state_desc.items()}
        hid = np.zeros((n_calls * self.S, HIDDEN), np.float16)
        call_ms = []
        for c in range(n_calls):
            t1 = time.perf_counter()
            res = await maybe(self.decoder(inputs={"input_ids": self.nd(ids_p[c * self.S:(c + 1) * self.S].reshape(1, self.S)),
                                                   "position_ids": self.nd(np.arange((c + 1) * self.S, dtype=np.int32)[None]),
                                                   **static}, state=state))
            hid[c * self.S:(c + 1) * self.S] = np.asarray(res["hidden"].numpy())[0]
            call_ms.append((time.perf_counter() - t1) * 1e3)
        tr["decoder"] = {"calls": n_calls, "S": self.S, "call_ms": call_ms, "total_ms": float(sum(call_ms))}
        tr["_hidden"] = hid[:T]
        Q = len(qs)
        O = sum(len(q["option_spans"]) for q in qs)
        fname, tb = next(((f, t) for f, t in self.buckets if t >= T), (None, None))
        if fname is None or Q > 16 or O > 128:
            raise SystemExit(f"T {T} / Q {Q} / O {O} exceed the head's buckets (T <= 4096, Q <= 16, O <= 128)")
        t0 = time.perf_counter()
        arrays, layout = head_inputs(hid, ids, qs, self.table, t_pad=tb, q_pad=16, o_pad=128)
        t1 = time.perf_counter()
        res = await maybe(self.head_fns[fname](inputs={n: self.nd(a) for n, a in arrays.items()}))
        logits = np.asarray(res["logits"].numpy())
        t2 = time.perf_counter()
        probs = question_probs(logits, layout)
        tr["head"] = {"function": fname, "host_ms": (t1 - t0) * 1e3, "head_ms": (t2 - t1) * 1e3}
        tr["_logits"] = logits[:O]
        out = response(request, qs, probs, T)
        tr["total_ms"] = (time.perf_counter() - t_all) * 1e3
        return out


def engine_from(args) -> ClefFlash:
    return ClefFlash(args.bundle, args.decoder_aot, args.head, args.table,
                     {256: args.tower_g256, 448: args.tower_g448})


def asset_record(e: ClefFlash) -> dict:
    return {"bundle": str(e.bundle), "decoder_aot": str(e.decoder_aot), "head_bundle": str(e.head_dir),
            "head_aot": str(e.head_aot), "head_buckets": e.buckets, "table": str(e.table_path),
            "table_sha256_recorded": json.loads(e.table_path.with_suffix(".json").read_text())["sha256"]
            if e.table_path.with_suffix(".json").exists() else None,
            "towers": {str(g): str(p) for g, p in e.tower_paths.items()},
            "decoder_bundle_name": e.meta.get("name"), "decoder_prefill_chunk": e.S}


def cmd_run(args) -> int:
    request = json.loads(Path(args.request).read_text())
    e = engine_from(args)
    trace: dict = {}

    async def go():
        load = await e.load()
        trace["load_seconds"] = load
        return await e.decide(request, image=args.image, grid=args.grid, trace=trace)

    resp = asyncio.run(go())
    Path(args.out).write_text(json.dumps(resp, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(resp, ensure_ascii=False))
    if args.trace:
        Path(args.trace).write_text(json.dumps({k: v for k, v in trace.items() if not k.startswith("_")}, indent=1) + "\n")
    return 0


# --------------------------------------------------------------------------- e2e check vs the oracle
def compare_answers(mine: dict, ref: dict) -> dict:
    """Every numeric field and the choice of every answer, 4-decimal values as the responses carry them."""
    fields, equal, diffs = 0, 0, []
    choice_equal = []
    for qid, a in ref["answers"].items():
        b = mine["answers"].get(qid)
        if b is None or b["type"] != a["type"]:
            diffs.append({"question_id": qid, "field": "missing or type", "abs_diff": None})
            continue
        if a["type"] == "choice":
            choice_equal.append(a["choice"] == b["choice"])
        pairs = [(k, a[k], b[k]) for k in ("noul", "score", "confidence") if k in a]
        pairs += [(f"p[{k}]", v, b["probabilities"][k]) for k, v in (a.get("probabilities") or {}).items()]
        for name, x, y in pairs:
            fields += 1
            if x == y:
                equal += 1
            else:
                diffs.append({"question_id": qid, "field": name, "oracle": x, "mine": y, "abs_diff": abs(x - y)})
    main = {"noul": "noul", "choice": "choice", "score": "score"}
    head_values = []
    for qid, a in ref["answers"].items():
        b = mine["answers"].get(qid)
        if b is None:
            continue
        k = main[a["type"]]
        head_values.append({"question_id": qid, "type": a["type"], "oracle": a[k], "mine": b[k], "equal": a[k] == b[k]})
    return {"fields": fields, "fields_equal": equal, "answer_values": head_values,
            "answer_values_equal": sum(v["equal"] for v in head_values), "answers": len(head_values),
            "choices_equal": f"{sum(choice_equal)}/{len(choice_equal)}",
            "model_equal": mine["model"] == ref["model"], "usage_equal": mine["usage"] == ref["usage"],
            "max_abs_diff": max((d["abs_diff"] for d in diffs if d["abs_diff"] is not None), default=0.0),
            "diffs": diffs}


def cmd_check(args) -> int:
    from parity_decoder_torch import cos_rows
    oracle = Path(args.oracle)
    fixtures = Path(args.fixtures)
    rows = {(r["id"], r["arm"]): r for r in json.loads((oracle / "records_oracle.json").read_text())["rows"]}
    records = {r["id"]: r for r in json.loads((fixtures / "records.json").read_text())["records"]}
    runs = [tuple(x.split(":")) for x in args.runs.split(",")]
    if len(runs) > 40:
        raise SystemExit("at most 40 requests per process")
    e = engine_from(args)
    results = []

    async def go():
        load = await e.load()
        for rid, arm in runs:
            rec, row = records[rid], rows[(rid, arm)]
            request = copy.deepcopy(rec["request"])          # the raw request, no oracle ids or spans
            image = (fixtures / "images" / rec["image_files"][0]) if arm != "text" else None
            grid = int(arm[1:]) if arm != "text" else None
            tr: dict = {}
            t0 = time.perf_counter()
            resp = await e.decide(request, image=image, grid=grid, trace=tr)
            wall = time.perf_counter() - t0
            cmp = compare_answers(resp, row["systemone_response"])
            z = np.load(oracle / "npz" / f"{rid}__{arm}.npz")
            item = {"id": rid, "arm": arm, "source": rec["source"], "tokens": tr["tokens"], "wall_seconds": wall,
                    "ids_equal_oracle": tr["ids"] == row["ids"],
                    "spans_equal_oracle": [[q["question_span"], q["option_spans"]] for q in tr["questions"]]
                    == [[q["question_span"], q["option_spans"]] for q in row["questions"]],
                    "hidden_min_pos_cos_vs_oracle": float(cos_rows(tr["_hidden"].astype(np.float32), z["last_hidden"]).min()),
                    "response": resp, "oracle_response": row["systemone_response"], "compare": cmp,
                    "trace": {k: v for k, v in tr.items() if not k.startswith("_") and k not in ("ids", "questions")}}
            if "_image_embeds" in tr:
                c = cos_rows(tr["_image_embeds"], z["image_embeds"])
                item["tower_min_row_cos_vs_oracle"] = float(c.min())
            results.append(item)
            print(f"{rid}/{arm}: T={tr['tokens']} ids {item['ids_equal_oracle']} answer values {cmp['answer_values_equal']}/"
                  f"{cmp['answers']} fields {cmp['fields_equal']}/{cmp['fields']} choices {cmp['choices_equal']} "
                  f"max|d| {cmp['max_abs_diff']:.4f} ({wall:.2f} s)", flush=True)
        return load

    load = asyncio.run(go())
    tot = {"requests": len(results),
           "answer_values_equal": sum(r["compare"]["answer_values_equal"] for r in results),
           "answers": sum(r["compare"]["answers"] for r in results),
           "fields_equal": sum(r["compare"]["fields_equal"] for r in results),
           "fields": sum(r["compare"]["fields"] for r in results),
           "choices_equal": f"{sum(int(r['compare']['choices_equal'].split('/')[0]) for r in results)}/"
                            f"{sum(int(r['compare']['choices_equal'].split('/')[1]) for r in results)}",
           "max_abs_diff": max(r["compare"]["max_abs_diff"] for r in results),
           "ids_equal_all": all(r["ids_equal_oracle"] for r in results),
           "spans_equal_all": all(r["spans_equal_oracle"] for r in results),
           "model_and_usage_equal_all": all(r["compare"]["model_equal"] and r["compare"]["usage_equal"] for r in results)}
    doc = {"schema": "clef-flash-decide-e2e/1",
           "what": "decide.py from the raw request (host ids, host image preprocess + tower AOT, decoder AOT, head AOT, "
                   "fp16 lm_head table) vs the author's fp32 systemone() response of the same record",
           "assets": asset_record(e), "load_seconds": load, "oracle": str(oracle / "records_oracle.json"),
           "fixtures": str(fixtures / "records.json"), "summary": tot,
           "script_sha256": {"decide.py": sha256_file(Path(__file__).resolve()), "clef_head.py": sha256_file(HERE / "clef_head.py"),
                             "host.py": sha256_file(HERE / "host.py")},
           "timing_note": "contended GPU (other sessions); first request of the process includes warm-up",
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "runs": results}
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(tot))
    print(f"wrote {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "check"):
        a = sub.add_parser(name)
        a.add_argument("--bundle", default=str(DEFAULTS["bundle"]))
        a.add_argument("--decoder-aot", default=str(DEFAULTS["decoder_aot"]))
        a.add_argument("--head", default=str(DEFAULTS["head"]), help="the head bundle dir (its metadata names the AOT asset)")
        a.add_argument("--table", default=str(DEFAULTS["table"]))
        a.add_argument("--tower-g256", default=str(DEFAULTS["tower_g256"]))
        a.add_argument("--tower-g448", default=str(DEFAULTS["tower_g448"]))
        a.add_argument("--out", required=True)
        if name == "run":
            a.add_argument("--request", required=True)
            a.add_argument("--image")
            a.add_argument("--grid", type=int, choices=sorted(GRIDS))
            a.add_argument("--trace")
        else:
            a.add_argument("--runs", required=True, help="id:arm list (arm = text | g256 | g448)")
            a.add_argument("--oracle", default=str(LANE / "oracle"))
            a.add_argument("--fixtures", default=str(LANE / "fixtures"))
    args = ap.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
