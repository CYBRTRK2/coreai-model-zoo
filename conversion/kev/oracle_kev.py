#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "kev",            # the author's repository at tag kev-1.0, installed with its uv.lock (see README)
#     "safetensors>=0.8.0",
#     "numpy",
# ]
# ///
"""fp32 oracle for Kev-0.8B, from the author's own code (github.com/jaredpalmer/kev at tag kev-1.0).

jaredpalmer/kev-0.8b (Hub tag v1.0, Apache-2.0) is a rank-16 LoRA and a pointer head on Qwen/Qwen3.5-0.8B-Base
(revision dc7cdfe2). This script imports the author's package unchanged and loads the checkpoint the way every
reported number is computed: `Checkpoint(...).load("cpu", LoadOptions(dtype=torch.float32))` (the LoRA merged in fp32
by peft's merge_and_unload), then answers every fixture record through the serving encoder and the row-form forward:

    req = SystemOneRequest.model_validate(request); rec, meta = kev.api.to_record(req)
    enc = kev.model.admit(model, tok, rec)            # the serving limits, strict (a state is never truncated)
    logits = model.forward(enc)                       # one causal row per question = state + its branch, T applied
    answers = kev.api.to_answers(softmax(logits), meta)

A wrapper on `model._rows_hidden` (the call that returns each row's `last_hidden_state`, after the final RMSNorm,
fp32) records what every later Core AI stage is compared against. Asserted on every record: the row the hidden
states belong to is state ids + branch ids (kev.model.rows_of), the head re-run on the recorded hidden states equals
forward()'s logits bit for bit, and the raw logits (head at temperature 1) divided by T equal them bit for bit.

Per question the oracle keeps: the row's ids, its length, the <decide> and </opt> indices within the row, the logits
after and before the temperature, the probabilities (in kev.api.question_keys order), the argmax key, the top-2
margin, the gold key. Per record: the packed ids and segments, the state token count, the TypeSafe response body
(kev.api.to_answers + kev.api.output_tokens, assembled as kev.serve.Server._body does; kev.serve itself needs
fastapi and is not imported), and the forward time.

Also recorded:
* `prefix_check` - for a fixed subset, the serving path `model.probs(enc)` (the state once, the question rows on
  its cache) against forward()'s probabilities (the author: equal to fp32 rounding);
* `hidden/<id>.npz` - every row's hidden states [L, hidden] fp32 for six records (round 2's decoder parity);
* `determinism` - record 0 run again after the loop, logits bit-equal;
* `--phase nomerge` (a separate process) - the same subset with `LoadOptions(dtype=fp32, merge=False)` (the adapter
  applied by peft at run time, nothing folded) against the merged run's probabilities: the floor of the merge.

    HF_HOME=$ZOO_WORK_ROOT/_kev/hf HF_HUB_OFFLINE=1 python conversion/kev/oracle_kev.py
    HF_HOME=$ZOO_WORK_ROOT/_kev/hf HF_HUB_OFFLINE=1 python conversion/kev/oracle_kev.py --phase nomerge

-> $ZOO_WORK_ROOT/_kev/oracle/records_oracle.json, oracle/hidden/<id>.npz, results/oracle_summary.json,
results/oracle_nomerge_check.json.

Kev-4B (round 4) is the same code on another checkpoint and output directory:

    HF_HOME=$ZOO_WORK_ROOT/_kev/hf HF_HUB_OFFLINE=1 python conversion/kev/oracle_kev.py --threads 1 \
        --checkpoint jaredpalmer/kev-4b@591dcb5bd6d05eb0b5131ea6608f93f10243335c \
        --base Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b \
        --out-dir $ZOO_WORK_ROOT/_kev/oracle_4b --summary-name oracle_summary_4b.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import work_path  # noqa: E402

from kev.api import SystemOneRequest, output_tokens, question_keys, to_answers, to_record  # noqa: E402
from kev.checkpoint import Checkpoint, LoadOptions  # noqa: E402
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, SPECIAL, admit, rows_of  # noqa: E402

CHECKPOINT = "jaredpalmer/kev-0.8b@788ddbdd65715bb03a56788c822f6c632c9a551d"   # Hub tag v1.0 (resolves to commit bf75a6a8)
BASE = ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
HIDDEN_RECORDS = ["tv4_000", "semif_a3f18f3a63d45345942b", "own_t01", "own_j03", "own_L02", "own_m01"]
SUBSET_OWN = ["own_t01", "own_t03", "own_t06", "own_t09", "own_j01", "own_j03", "own_a01", "own_L01", "own_L02", "own_L03"]
TV4X_SOURCES = ["emotion", "tweet_offensive", "qnli", "paws", "sciq", "legacy_holdout", "composition_holdout"]
NEAR_TIE = 0.02


def subset_ids(records):
    """The fixed subset of the prefix and no-merge checks: tv4 5, tv4x 1 per source (7), tv4s 3, semif 5, own 10."""
    ids = [r["id"] for r in records]
    semif = [i for i in ids if i.startswith("semif_")][:5]
    out = [f"tv4_{k:03d}" for k in range(5)] + [f"tv4x_{s}_00" for s in TV4X_SOURCES] + [f"tv4s_{k:02d}" for k in range(3)] + semif + SUBSET_OWN
    assert all(i in ids for i in out), [i for i in out if i not in ids]
    return out


def has_subset(records) -> bool:
    """The prefix / no-merge subset is defined on records.json; another fixture file (heldout.json) has none of it."""
    ids = {r["id"] for r in records}
    return all(i in ids for i in [f"tv4_{k:03d}" for k in range(5)] + SUBSET_OWN)


def fixtures_file(arg: str) -> Path:
    """--fixtures: a directory (its records.json) or a fixture file (heldout.json)."""
    p = Path(arg)
    return p / "records.json" if p.is_dir() else p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def floats(t: torch.Tensor) -> list[float]:
    return [float(x) for x in t.detach().cpu().reshape(-1).tolist()]


class RowsHook:
    """Wraps model._rows_hidden: every call's list of per-row hidden states [L_i, d] (fp32, after the final norm)."""

    def __init__(self, model):
        self.model, self.calls = model, []
        self.orig = model._rows_hidden

        def hooked(rows, cache=None, prefix_len=0):
            out = self.orig(rows, cache=cache, prefix_len=prefix_len)
            self.calls.append({"rows": [list(map(int, ids)) for ids, _ in rows], "pos": [list(map(int, p)) for _, p in rows],
                               "prefix_len": prefix_len, "hidden": out})
            return out
        model._rows_hidden = hooked

    def reset(self):
        self.calls = []


def load(opts, checkpoint=CHECKPOINT, base=BASE):
    t0 = time.perf_counter()
    ck = Checkpoint(checkpoint)
    tok, model = ck.load("cpu", opts)
    secs = time.perf_counter() - t0
    assert (ck.meta.base, ck.meta.base_revision) == tuple(base), (ck.meta.base, ck.meta.base_revision)
    assert model.hybrid, "Qwen3.5 must run the row form"
    info = {"checkpoint": checkpoint, "resolved_path": ck.path, "load_options": repr(opts), "load_seconds": round(secs, 1),
            "lm_class": type(model.lm).__name__, "model_type": model.lm.config.model_type, "dtype": model.dtype,
            "hybrid": model.hybrid, "backend": model.backend, "attn": getattr(model.lm.config, "_attn_implementation", None),
            "temperature": model.head.temperature, "meta_temperature": ck.meta.temperature, "head_dim": ck.meta.head_dim,
            "torch_threads": torch.get_num_threads(), "pad_id": model.pad_id,
            "special_ids": {t: tok.convert_tokens_to_ids(t) for t in SPECIAL},
            "head_pt_sha256": sha256_file(Path(ck.path) / "head.pt"),
            "adapter_sha256": sha256_file(Path(ck.path) / "adapter_model.safetensors")}
    return tok, model, info


def run_record(model, tok, hook, r, keep_hidden=False):
    """One record through the author's serving encoder and row-form forward. -> (entry, per-row hidden or None)."""
    req = SystemOneRequest.model_validate(r["request"])
    rec, meta = to_record(req)
    enc = admit(model, tok, rec)
    assert not enc["state_truncated"]
    S, Sp, rows = rows_of(enc)
    hook.reset()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model.forward(enc)
    secs = time.perf_counter() - t0
    assert len(hook.calls) == 1, len(hook.calls)
    call = hook.calls[0]
    hs = call["hidden"]
    assert len(hs) == len(rows) == len(logits) == len(meta)
    T = model.head.temperature
    questions, probs_all = [], []
    for k, (r_k, m, z_T, h) in enumerate(zip(rows, meta, logits, hs)):
        row_ids = S + r_k["ids"]
        assert call["rows"][k] == row_ids and call["pos"][k] == list(range(len(row_ids))), "row != state + branch at 0..L-1"
        assert h.shape == (len(row_ids), model.lm.config.hidden_size) and h.dtype == torch.float32
        d = len(S) + r_k["decide"]
        oi = [len(S) + o for o in r_k["opts"]]
        assert row_ids[d] == tok.convert_tokens_to_ids(SPECIAL[4]) and all(row_ids[o] == tok.convert_tokens_to_ids(SPECIAL[3]) for o in oi)
        with torch.no_grad():
            again = model.head(h[d], h[torch.tensor(oi)])
            model.head.temperature = 1.0
            z_raw = model.head(h[d], h[torch.tensor(oi)])
            model.head.temperature = T
        assert torch.equal(again, z_T), "head on the recorded hidden states != forward logits"
        assert torch.equal(z_raw / T, z_T), "raw logits / T != forward logits"
        p = F.softmax(z_T, -1)
        probs_all.append(p)
        keys = m["keys"]
        assert keys == question_keys(m["type"], r["request"]["questions"][m["id"]].get("criteria"))
        order = torch.argsort(p, descending=True)
        top = keys[int(order[0])]
        margin = float(p[order[0]] - p[order[1]]) if len(keys) > 1 else 1.0
        gold = r["gold"].get(m["id"])
        questions.append({
            "qid": m["id"], "type": m["type"], "keys": keys, "options": rec["questions"][k]["options"],
            "instr": rec["questions"][k]["instr"], "row_ids": row_ids, "row_len": len(row_ids), "state_len": len(S),
            "decide": d, "opts": oi, "logits_T": floats(z_T), "logits_raw": floats(z_raw), "probs": floats(p),
            "argmax": top, "top2_margin": margin, "near_tie": margin <= NEAR_TIE, "gold": gold,
            "correct": None if gold is None else top == gold})
    answers = to_answers([floats(p) for p in probs_all], meta)
    body = {"model": req.model, "answers": answers, "usage": {"input_tokens": len(enc["ids"]), "output_tokens": output_tokens(tok, answers)},
            "latency_ms": round(secs * 1000, 1)}
    entry = {"id": r["id"], "source": r["source"], "ids": enc["ids"], "seg": enc["seg"], "pos": enc["pos"], "state_tokens": enc["state_tokens"],
             "state_truncated": enc["state_truncated"], "questions": questions, "response": body, "seconds": round(secs, 4)}
    hidden = [h.detach().clone() for h in hs] if keep_hidden else None
    return entry, hidden, logits


def save_hidden(path: Path, entry, hidden):
    arrays = {}
    for k, (q, h) in enumerate(zip(entry["questions"], hidden)):
        arrays[f"q{k}_hidden"] = h.numpy().astype(np.float32)
        arrays[f"q{k}_ids"] = np.asarray(q["row_ids"], dtype=np.int32)
        arrays[f"q{k}_decide"] = np.asarray([q["decide"]], dtype=np.int32)
        arrays[f"q{k}_opts"] = np.asarray(q["opts"], dtype=np.int32)
        arrays[f"q{k}_logits_T"] = np.asarray(q["logits_T"], dtype=np.float32)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return {k: list(v.shape) for k, v in arrays.items() if k.endswith("_hidden")}


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def group_of(rid):
    for g in ("tv4x", "tv4s", "tv4", "semif", "own"):
        if rid.startswith(g + "_"):
            return g
    return "other"


def summarize(entries, header):
    groups = {}
    for e in entries:
        g = groups.setdefault(group_of(e["id"]), {"records": 0, "questions": 0, "types": {}, "gold": 0, "correct": 0, "near_ties": []})
        g["records"] += 1
        for q in e["questions"]:
            g["questions"] += 1
            g["types"][q["type"]] = g["types"].get(q["type"], 0) + 1
            if q["gold"] is not None:
                g["gold"] += 1
                g["correct"] += bool(q["correct"])
            if q["near_tie"]:
                g["near_ties"].append(f"{e['id']}/{q['qid']}")
    for g in groups.values():
        g["accuracy_our_subset"] = round(g["correct"] / g["gold"], 4) if g["gold"] else None
    rows = [q["row_len"] for e in entries for q in e["questions"]]
    secs = [e["seconds"] for e in entries]
    ties = [f"{e['id']}/{q['qid']}" for e in entries for q in e["questions"] if q["near_tie"]]
    return {"records": len(entries), "questions": len(rows), "by_group": groups,
            "row_len": {"min": min(rows), "p50": pct(rows, 0.5), "p99": pct(rows, 0.99), "max": max(rows)},
            "options_max": max(len(q["keys"]) for e in entries for q in e["questions"]),
            "near_ties": {"threshold_top2": NEAR_TIE, "count": len(ties), "ids": ties},
            "accuracy_note": "gold agreement on our fixture subset only (not the author's benchmark numbers)",
            "seconds_per_record": {"p50": round(pct(secs, 0.5), 3), "max": round(max(secs), 3), "total_forward": round(sum(secs), 1)},
            "threads": header["model"]["torch_threads"]}


def main_phase(args):
    out_dir, res_dir = Path(args.out_dir), Path(args.results_dir)
    (out_dir / "hidden").mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    fx_path = fixtures_file(args.fixtures)
    doc = json.loads(fx_path.read_text())
    records = doc["records"]
    if args.records:
        want = set(args.records.split(","))
        records = [r for r in records if r["id"] in want]
    torch.manual_seed(0)
    if args.threads:
        torch.set_num_threads(args.threads)
    tok, model, info = load(LoadOptions(dtype=torch.float32), args.checkpoint, base_of(args.base))
    from kev.predictors import kernel_environment
    header = {"schema": "kev-oracle/1", "fixtures": {"path": str(fx_path), "sha256": sha256_file(fx_path)}, "model": info,
              "kernel_environment": kernel_environment(model, "cpu"),
              "serving_limits": {"max_state": SERVE_MAX_STATE, "max_branch": SERVE_MAX_BRANCH},
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           **{m: __import__(m).__version__ for m in ("transformers", "peft", "numpy", "safetensors")}},
              "platform": platform.platform(), "started": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    print(json.dumps(info, indent=1), flush=True)
    hook = RowsHook(model)
    partial = out_dir / "records_oracle.partial.jsonl"
    done = {}
    if args.resume and partial.exists():
        for line in partial.read_text().splitlines():
            e = json.loads(line)
            done[e["id"]] = e
    entries, t_all = [], time.perf_counter()
    hidden_index = {}
    with open(partial, "a" if args.resume else "w") as pf:
        for n, r in enumerate(records):
            if r["id"] in done and r["id"] not in HIDDEN_RECORDS:
                entries.append(done[r["id"]])
                continue
            entry, hidden, _ = run_record(model, tok, hook, r, keep_hidden=r["id"] in HIDDEN_RECORDS)
            if hidden is not None:
                hidden_index[r["id"]] = save_hidden(out_dir / "hidden" / f"{r['id']}.npz", entry, hidden)
            entries.append(entry)
            if r["id"] not in done:
                pf.write(json.dumps(entry) + "\n")
                pf.flush()
            if n % 20 == 0 or entry["seconds"] > 5:
                print(f"[{n + 1}/{len(records)}] {r['id']} q={len(entry['questions'])} rows={[q['row_len'] for q in entry['questions']]} "
                      f"{entry['seconds']:.2f}s", flush=True)
    loop_secs = time.perf_counter() - t_all
    by_id = {e["id"]: e for e in entries}

    # determinism: record 0 again, bit-equal logits
    r0 = records[0]
    e0, _, _ = run_record(model, tok, hook, r0)
    det = all(a["logits_T"] == b["logits_T"] for a, b in zip(e0["questions"], by_id[r0["id"]]["questions"]))
    assert det, "record 0 re-run differs"

    # prefix (serving) path vs forward on the fixed subset
    prefix = []
    if not args.records and has_subset(doc["records"]):
        for rid in subset_ids(doc["records"]):
            r = next(x for x in records if x["id"] == rid)
            rec, _ = to_record(SystemOneRequest.model_validate(r["request"]))
            enc = admit(model, tok, rec)
            hook.reset()
            t0 = time.perf_counter()
            with torch.no_grad():
                ps = model.probs(enc)
            secs = time.perf_counter() - t0
            ref = [q["probs"] for q in by_id[rid]["questions"]]
            d = max(max(abs(a - b) for a, b in zip(floats(p), q)) for p, q in zip(ps, ref))
            prefix.append({"id": rid, "max_abs_dp": d, "argmax_equal": all(int(torch.argmax(p)) == int(np.argmax(q)) for p, q in zip(ps, ref)),
                           "calls": [{"rows": len(c["rows"]), "prefix_len": c["prefix_len"]} for c in hook.calls], "seconds": round(secs, 3)})
            print(f"prefix {rid}: max|dp| {d:.3g}", flush=True)
    header["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    header["loop_seconds"] = round(loop_secs, 1)
    header["determinism"] = {"record": r0["id"], "logits_bit_equal": det}
    header["hidden_records"] = {"dir": str(out_dir / "hidden"), "shapes": hidden_index}
    header["checks"] = ("per question: row == state + branch at positions 0..L-1; <decide> / </opt> ids at the readout indices; "
                        "head(recorded hidden) == forward logits (torch.equal); head at T=1 / T == forward logits (torch.equal); "
                        "keys == kev.api.question_keys")
    out = {**header, "records": entries,
           "prefix_check": {"subset": len(prefix), "max_abs_dp": max((p["max_abs_dp"] for p in prefix), default=None),
                            "argmax_equal_all": all(p["argmax_equal"] for p in prefix) if prefix else None,
                            "per_record": prefix,
                            "path": "model.probs(enc): DecisionModel.probs -> probs_and_prefix (state once, rows on its cache)",
                            **({} if prefix else {"skipped": "not run: the subset is records.json's"})}}
    write_atomic(out_dir / "records_oracle.json", (json.dumps(out) + "\n").encode())
    summary = summarize(entries, header)
    summary["prefix_check"] = {k: v for k, v in out["prefix_check"].items() if k != "per_record"}
    summary["determinism"] = header["determinism"]
    summary["load_seconds"] = info["load_seconds"]
    summary["loop_seconds"] = header["loop_seconds"]
    summary["oracle_sha256"] = sha256_file(out_dir / "records_oracle.json")
    if not args.records:
        write_atomic(res_dir / args.summary_name, (json.dumps(summary, indent=1) + "\n").encode())
    print(json.dumps({k: v for k, v in summary.items() if k != "by_group"}, indent=1))
    for g, v in summary["by_group"].items():
        print(g, {k: v[k] for k in ("records", "questions", "types", "accuracy_our_subset")}, "near ties", len(v["near_ties"]))
    return 0


def nomerge_phase(args):
    out_dir, res_dir = Path(args.out_dir), Path(args.results_dir)
    ref = json.loads((out_dir / "records_oracle.json").read_text())
    by_id = {e["id"]: e for e in ref["records"]}
    doc = json.loads(fixtures_file(args.fixtures).read_text())
    if args.threads:
        torch.set_num_threads(args.threads)
    tok, model, info = load(LoadOptions(dtype=torch.float32, merge=False), args.checkpoint, base_of(args.base))
    info["lm_class_unmerged"] = type(model.lm).__name__
    hook = RowsHook(model)
    rows = []
    for rid in subset_ids(doc["records"]):
        r = next(x for x in doc["records"] if x["id"] == rid)
        e, _, _ = run_record(model, tok, hook, r)
        qs = []
        for a, b in zip(e["questions"], by_id[rid]["questions"]):
            assert a["row_ids"] == b["row_ids"]
            qs.append({"qid": a["qid"], "max_abs_dp": max(abs(x - y) for x, y in zip(a["probs"], b["probs"])),
                       "max_abs_dlogit_T": max(abs(x - y) for x, y in zip(a["logits_T"], b["logits_T"])),
                       "argmax_equal": a["argmax"] == b["argmax"], "near_tie": b["near_tie"]})
        rows.append({"id": rid, "questions": qs, "seconds": e["seconds"]})
        print(f"nomerge {rid}: max|dp| {max(q['max_abs_dp'] for q in qs):.3g}", flush=True)
    allq = [q for r in rows for q in r["questions"]]
    out = {"what": "LoadOptions(dtype=fp32, merge=False) (peft LoRA applied at run time) vs the merged fp32 oracle, same encoder and rows",
           "model": info, "records": len(rows), "questions": len(allq),
           "max_abs_dp": max(q["max_abs_dp"] for q in allq), "mean_abs_dp_max_per_question": statistics.mean(q["max_abs_dp"] for q in allq),
           "max_abs_dlogit_T": max(q["max_abs_dlogit_T"] for q in allq),
           "argmax_equal": sum(q["argmax_equal"] for q in allq), "per_record": rows}
    write_atomic(res_dir / "oracle_nomerge_check.json", (json.dumps(out, indent=1) + "\n").encode())
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_record", "model")}, indent=1))
    return 0


def base_of(arg: str) -> tuple[str, str]:
    """--base 'repo@revision' -> (repo, revision)."""
    repo, _, rev = arg.partition("@")
    return repo, rev


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=["main", "nomerge"], default="main")
    ap.add_argument("--checkpoint", default=CHECKPOINT,
                    help="adapter checkpoint id@revision (Kev-4B: jaredpalmer/kev-4b@591dcb5bd6d05eb0b5131ea6608f93f10243335c)")
    ap.add_argument("--base", default=f"{BASE[0]}@{BASE[1]}",
                    help="the base repo@revision the checkpoint must name (Kev-4B: Qwen/Qwen3.5-4B-Base@1001bb4d...)")
    ap.add_argument("--summary-name", default="oracle_summary.json", help="the summary file under --results-dir")
    ap.add_argument("--fixtures", default=str(work_path("_kev", "fixtures")))
    ap.add_argument("--out-dir", default=str(work_path("_kev", "oracle")))
    ap.add_argument("--results-dir", default=str(work_path("_kev", "results")))
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = torch's default)")
    ap.add_argument("--records", default=None, help="comma-separated record ids (probe; no summary written)")
    ap.add_argument("--resume", action="store_true", help="keep records already in records_oracle.partial.jsonl")
    args = ap.parse_args()
    return main_phase(args) if args.phase == "main" else nomerge_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
