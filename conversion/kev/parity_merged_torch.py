#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "kev",            # the author's repository at tag kev-1.0, installed with its uv.lock (see README)
#     "safetensors>=0.8.0",
#     "numpy",
# ]
# ///
"""Merged Kev-0.8B weights vs the adapter checkpoint, both through the author's loader, fp32 on CPU.

The merged checkpoint is written by the author's `scripts/merge_lora_checkpoint.py` (tag kev-1.0): a full-weight
checkpoint (a flat `qwen3_5_text` config.json, the backbone's safetensors with keys `layers.N...` / `embed_tokens` /
`norm`, and head.pt with weights "full"), every adapted weight = fp32(W) + fp32(peft get_delta_weight). Every later
Core AI stage loads that directory, so before any export it must answer exactly as the checkpoint the oracle used
(the LoRA adapter on the base, merged at load by peft in fp32).

`--phase answers` loads `Checkpoint(<merged>).load("cpu", LoadOptions(dtype=torch.float32))` (the full-weight path:
nothing to merge) and runs every fixture record through the same encoder and row-form forward as oracle_kev.py,
then compares with oracle/records_oracle.json question by question. Bar, fixed before the run: the argmax equal on
every question (near-ties included) and max |dp| <= 1e-4; the logits' max |d| and the hidden states' max |d| (for
the oracle's six hidden records) are recorded too.

`--phase formula` checks the merge arithmetic on three tensors read straight from the safetensors files (no model):
a Gated DeltaNet `in_proj_qkv`, an attention `q_proj` and an MLP `down_proj`, W (base, bf16 -> fp32) + (B @ A) *
lora_alpha / r against the merged tensor.

    HF_HOME=$ZOO_WORK_ROOT/_kev/hf HF_HUB_OFFLINE=1 python conversion/kev/parity_merged_torch.py --phase formula
    HF_HOME=$ZOO_WORK_ROOT/_kev/hf HF_HUB_OFFLINE=1 python conversion/kev/parity_merged_torch.py --phase answers

-> $ZOO_WORK_ROOT/_kev/results/merge_formula_check.json, results/parity_merged_vs_adapter.json.

Kev-4B (round 4): `--model kev-4b --merged $ZOO_WORK_ROOT/_kev/merged/kev-4b-v1.0/checkpoint --oracle-dir
$ZOO_WORK_ROOT/_kev/oracle_4b` (results gain the suffix `_4b`). The base and the merged checkpoint are sharded there:
every tensor is read from the shard model.safetensors.index.json names.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import work_path  # noqa: E402

BAR = {"argmax": "equal on every question, near-ties included", "max_abs_dp": 1e-4}
BASE_SNAPSHOT = ("models--Qwen--Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
                 "model.safetensors-00001-of-00001.safetensors")
ADAPTER_SNAPSHOT = ("models--jaredpalmer--kev-0.8b", "788ddbdd65715bb03a56788c822f6c632c9a551d")
MODELS = {"kev-0.8b": {"base": BASE_SNAPSHOT[:2], "adapter": ADAPTER_SNAPSHOT, "suffix": ""},
          "kev-4b": {"base": ("models--Qwen--Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"),
                     "adapter": ("models--jaredpalmer--kev-4b", "591dcb5bd6d05eb0b5131ea6608f93f10243335c"), "suffix": "_4b"}}
FORMULA_TENSORS = ["layers.0.linear_attn.in_proj_qkv", "layers.3.self_attn.q_proj", "layers.9.mlp.down_proj"]


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


def shard_of(directory: Path, key: str, single: str) -> Path:
    """The file holding `key`: the shard model.safetensors.index.json names, else the directory's one file."""
    index = directory / "model.safetensors.index.json"
    if index.exists():
        return directory / json.loads(index.read_text())["weight_map"][key]
    return directory / single


def read_tensor(path: Path, key: str):
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        return f.get_tensor(key)


def formula_phase(args):
    hub = Path(args.hf_home) / "hub"
    m = MODELS[args.model]
    base_dir = hub / m["base"][0] / "snapshots" / m["base"][1]
    ad_dir = hub / m["adapter"][0] / "snapshots" / m["adapter"][1]
    cfg = json.loads((ad_dir / "adapter_config.json").read_text())
    scaling = cfg["lora_alpha"] / cfg["r"]
    assert not cfg.get("use_rslora") and not cfg.get("use_dora") and not cfg.get("fan_in_fan_out")
    rows, files = [], {"adapter": str(ad_dir / "adapter_model.safetensors"), "base": {}, "merged": {}}
    for name in FORMULA_TENSORS:
        keys = {"base": f"model.language_model.{name}.weight", "A": f"base_model.model.{name}.lora_A.weight",
                "B": f"base_model.model.{name}.lora_B.weight", "merged": f"{name}.weight"}
        base_file = shard_of(base_dir, keys["base"], BASE_SNAPSHOT[2])
        merged_file = shard_of(Path(args.merged), keys["merged"], "model.safetensors")
        files["base"][name], files["merged"][name] = str(base_file), str(merged_file)
        ad_file = ad_dir / "adapter_model.safetensors"
        W, A, B = read_tensor(base_file, keys["base"]), read_tensor(ad_file, keys["A"]), read_tensor(ad_file, keys["B"])
        Mg = read_tensor(merged_file, keys["merged"])
        expect = W.float() + (B.float() @ A.float()) * scaling
        d = (expect - Mg).abs()
        delta = (B.float() @ A.float()) * scaling
        rows.append({"tensor": name, "keys": keys, "dtypes": {"base": str(W.dtype), "A": str(A.dtype), "B": str(B.dtype), "merged": str(Mg.dtype)},
                     "shapes": {"base": list(W.shape), "A": list(A.shape), "B": list(B.shape), "merged": list(Mg.shape)},
                     "max_abs_diff": float(d.max()), "max_rel_diff": float((d / expect.abs().clamp_min(1e-12)).max()),
                     "bit_equal": bool(torch.equal(expect, Mg)), "delta_abs_max": float(delta.abs().max()),
                     "delta_rel_to_W": float(delta.norm() / W.float().norm()),
                     "merged_differs_from_base": bool(not torch.equal(W.float(), Mg))})
    out = {"formula": f"merged = fp32(W_base) + (fp32(B) @ fp32(A)) * lora_alpha / r = ... * {scaling}", "scaling": scaling,
           "key_map": "adapter base_model.model.<name>.lora_{A,B}.weight -> base model.language_model.<name>.weight -> merged <name>.weight",
           "files": files, "tensors": rows, "max_abs_diff": max(r["max_abs_diff"] for r in rows)}
    write_json(Path(args.results_dir) / f"merge_formula_check{m['suffix']}.json", out)
    for r in rows:
        print(r["tensor"], r["shapes"]["merged"], "max|d|", r["max_abs_diff"], "bit_equal", r["bit_equal"], "delta/W", round(r["delta_rel_to_W"], 5))
    return 0


def answers_phase(args):
    from kev.checkpoint import Checkpoint, LoadOptions
    from oracle_kev import HIDDEN_RECORDS, RowsHook, run_record
    ref = json.loads((Path(args.oracle_dir) / "records_oracle.json").read_text())
    by_id = {e["id"]: e for e in ref["records"]}
    doc = json.loads((Path(args.fixtures) / "records.json").read_text())
    if args.threads:
        torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    ck = Checkpoint(args.merged)
    assert ck.full and ck.meta.weights == "full" and ck.meta.lora == 0
    tok, model = ck.load("cpu", LoadOptions(dtype=torch.float32))
    load_secs = time.perf_counter() - t0
    info = {"checkpoint": args.merged, "full": ck.full, "lm_class": type(model.lm).__name__, "dtype": model.dtype, "hybrid": model.hybrid,
            "temperature": model.head.temperature, "oracle_temperature": ref["model"]["temperature"], "threads": torch.get_num_threads(),
            "oracle_threads": ref["model"]["torch_threads"], "load_seconds": round(load_secs, 1),
            "merged_lora": ck.meta.extra.get("merged_lora")}
    assert info["temperature"] == info["oracle_temperature"]
    hook = RowsHook(model)
    per, qs_all = [], []
    t_loop = time.perf_counter()
    for n, r in enumerate(doc["records"]):
        keep = r["id"] in HIDDEN_RECORDS
        e, hidden, _ = run_record(model, tok, hook, r, keep_hidden=keep)
        o = by_id[r["id"]]
        qs = []
        for k, (a, b) in enumerate(zip(e["questions"], o["questions"])):
            assert a["row_ids"] == b["row_ids"] and a["decide"] == b["decide"] and a["opts"] == b["opts"]
            q = {"qid": a["qid"], "argmax_equal": a["argmax"] == b["argmax"], "near_tie": b["near_tie"],
                 "max_abs_dp": max(abs(x - y) for x, y in zip(a["probs"], b["probs"])),
                 "max_abs_dlogit_T": max(abs(x - y) for x, y in zip(a["logits_T"], b["logits_T"])),
                 "bit_equal_logits": a["logits_T"] == b["logits_T"]}
            qs.append(q)
        rec = {"id": r["id"], "questions": qs, "seconds": e["seconds"]}
        if keep:
            z = np.load(Path(args.oracle_dir) / "hidden" / f"{r['id']}.npz")
            rec["hidden_max_abs_diff"] = max(float(np.abs(h.numpy() - z[f"q{k}_hidden"]).max()) for k, h in enumerate(hidden))
        per.append(rec)
        qs_all += qs
        if n % 50 == 0:
            print(f"[{n + 1}/{len(doc['records'])}] {r['id']} max|dp| {max(q['max_abs_dp'] for q in qs):.3g}", flush=True)
    dp = [q["max_abs_dp"] for q in qs_all]
    out = {"what": "full-weight merged checkpoint (author's merge script) vs the adapter checkpoint merged at load (the oracle), "
                   "both kev.checkpoint fp32 CPU, same encoder (kev.model.admit) and row-form forward",
           "bar": BAR, "model": info, "records": len(per), "questions": len(qs_all),
           "argmax_equal": sum(q["argmax_equal"] for q in qs_all),
           "near_ties": sum(q["near_tie"] for q in qs_all), "near_ties_argmax_equal": sum(q["argmax_equal"] for q in qs_all if q["near_tie"]),
           "max_abs_dp": max(dp), "mean_of_question_max_abs_dp": statistics.mean(dp),
           "max_abs_dlogit_T": max(q["max_abs_dlogit_T"] for q in qs_all),
           "bit_equal_logits_questions": sum(q["bit_equal_logits"] for q in qs_all),
           "hidden_max_abs_diff": {r["id"]: r["hidden_max_abs_diff"] for r in per if "hidden_max_abs_diff" in r},
           "loop_seconds": round(time.perf_counter() - t_loop, 1), "per_record": per}
    out["pass"] = out["argmax_equal"] == out["questions"] and out["max_abs_dp"] <= BAR["max_abs_dp"]
    write_json(Path(args.results_dir) / f"parity_merged_vs_adapter{MODELS[args.model]['suffix']}.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_record", "model")}, indent=1))
    return 0 if out["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=["answers", "formula"], required=True)
    ap.add_argument("--merged", default=str(work_path("_kev", "merged", "kev-0.8b-v1.0", "checkpoint")))
    ap.add_argument("--fixtures", default=str(work_path("_kev", "fixtures")))
    ap.add_argument("--oracle-dir", default=str(work_path("_kev", "oracle")))
    ap.add_argument("--results-dir", default=str(work_path("_kev", "results")))
    ap.add_argument("--hf-home", default=str(work_path("_kev", "hf")))
    ap.add_argument("--threads", type=int, default=1, help="torch threads (the oracle ran with 1)")
    ap.add_argument("--model", default="kev-0.8b", choices=sorted(MODELS),
                    help="which adapter / base snapshots the formula phase reads, and the results' file suffix")
    args = ap.parse_args()
    return formula_phase(args) if args.phase == "formula" else answers_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
