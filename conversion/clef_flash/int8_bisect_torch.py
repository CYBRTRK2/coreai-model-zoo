#!/usr/bin/env python3
"""Which layers carry the int8 body's error? fp32 torch with the exported int8 weights, per config (clef-flash).

decider_vision/int8_bisect_torch.py's instrument on this port. The module is
`qwen3_5_clef_decoder.Qwen3_5ClefDecoder` in fp32, driven by `parity_decoder_torch.Runner` the
way the graph runs: fresh zero states, the prompt in S = 64 chunks (the shipped width), oracle ids,
the oracle's fp32 image_embeds; the hidden rows at every position go through the checkpoint's own
fp32 JointSchemaHead (`parity_decoder_torch.AuthorHead`) and the per-question softmax is compared
with the oracle (`oracle/records_oracle.json`). What changes per config is which linear weights are
int8: every module the exporter's int8lin config quantizes (`export_decoder.linear_quant_config`:
all 248 decoder linears; embedding, conv1d, norms excluded) holds either the checkpoint's own weight
(bf16 in the file, exact in fp32) or the weight the exported op computes — the exporter's own
`quantize_pytorch_model` call on the fp16 model (same config, the S = 64 export spec's reference
inputs, the same GDN flags), read back through the finalized module's dequantization
(`coreai::constexpr_blockwise_shift_scale` on the int8 codes and fp16 block scales, fp16 out),
then cast to fp32. Only the weights differ between configs; activations stay fp32.

    dump   quantize the fp16 model, save every quantized module's dequantized fp16 weight
           (<work>/dump/int8_clipping/layer_NN.safetensors + int8_clipping.json)
    rule   write the selection rule and the bisect runs (<work>/rule.json) before any result
    run    evaluate configs on the bisect runs; one process, one part file, resumable; the exact
           weights go to <work>/dump/exact_bf16/ once (read back per swap, so only the fp32 module
           stays resident)
    merge  collect the part files, apply the rule -> one transcript

Configs: `int8lin` (all int8), `exact` (the instrument's floor vs the oracle), `fp16_L00-15`,
`fp16_L16-31`, `only_gdn` / `only_attn` / `only_mlp` (that kind int8, the rest exact), `fp16_LNN`
(layer NN exact, every other int8; 32), and `fp16_set_...` candidate sets (`--candidates`: from the
single-layer ranking, the procedure in RULE, run after every single is in the part file).

    cd conversion/clef_flash
    PY=~/code/coreai/coreai-models/.venv/bin/python
    $PY int8_bisect_torch.py dump
    $PY int8_bisect_torch.py rule --runs img_07:g256,own_t03:text,...
    $PY int8_bisect_torch.py run --part <work>/parts/part_a.json [--configs ..] [--candidates]
    $PY int8_bisect_torch.py merge --out $ZOO_WORK_ROOT/_clefflash/results/int8_bisect.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
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

WORK = LANE / "bisect"
DUMP = WORK / "dump" / "int8_clipping"
EXACT = WORK / "dump" / "exact_bf16"
RULE_PATH = WORK / "rule.json"
N_LAYERS = 32
CHUNK = 64
GDN = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
RULE = {
    "1": "worst bisect run max|dp| <= 0.010 (every option of every question, near-ties included)",
    "2": "no bisect run's max|dp| above its own int8lin value (this instrument)",
    "3": "at most 6 fp16 layers",
    "4": "the smallest layer set meeting 1-3; between sets of one size, the lexicographically smallest sorted "
         "layer list",
    "search": "int8lin, exact, fp16 L00-15, fp16 L16-31, only GDN int8, only attention int8, only MLP int8, one "
              "layer fp16 at a time (32); then candidate sets: layers ranked by their single-layer worst-run "
              "max|dp| (ties: the lower index); for k = 2..6 the top-k set; at the first k whose top-k set meets "
              "1-3, also every other k-subset of the top k+1; stop there. A single layer meeting 1-3 ends the "
              "search at size 1 (every single is evaluated).",
    "limits": {"worst_max_abs_dp": 0.010, "max_fp16_layers": 6},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def layer_of(name: str) -> int:
    m = re.match(r"model\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def kind_of(name: str) -> str:
    leaf = name.rsplit(".", 1)[-1]
    if ".mlp." in name and leaf in MLP:
        return "mlp"
    if ".self_attn." in name and leaf in ATTN:
        return "attn"
    if ".linear_attn." in name and leaf in GDN:
        return "gdn"
    return "other"


def set_name(layers) -> str:
    return "fp16_set_" + "_".join(f"L{i:02d}" for i in sorted(layers))


def fp16_set_config(every: list[str], layers) -> dict:
    keep = set(layers)
    return {"int8": [n for n in every if layer_of(n) not in keep], "fp16_layers": sorted(keep),
            "why": f"layers {sorted(keep)} exact, every other layer int8"}


def configs(names: list[str]) -> dict[str, dict]:
    """name -> {"int8": [module names], "fp16_layers": [..] | None, "why": ..}."""
    every = sorted(names)
    for k in (kind_of(n) for n in every):
        assert k != "other", "a quantized module outside mlp / attn / gdn"
    out = {
        "int8lin": {"int8": every, "fp16_layers": [], "why": "the export's int8lin body"},
        "exact": {"int8": [], "fp16_layers": None, "why": "no int8: the instrument's floor vs the oracle"},
        "fp16_L00-15": {**fp16_set_config(every, range(16)), "why": "first 16 layers exact, last 16 int8"},
        "fp16_L16-31": {**fp16_set_config(every, range(16, 32)), "why": "last 16 layers exact, first 16 int8"},
        "only_gdn": {"int8": [n for n in every if kind_of(n) == "gdn"], "fp16_layers": None,
                     "why": "GDN in_proj_qkv/z/b/a + out_proj int8, the rest exact"},
        "only_attn": {"int8": [n for n in every if kind_of(n) == "attn"], "fp16_layers": None,
                      "why": "full-attention q/k/v/o int8, the rest exact"},
        "only_mlp": {"int8": [n for n in every if kind_of(n) == "mlp"], "fp16_layers": None,
                     "why": "MLP gate/up/down int8, the rest exact"},
    }
    for i in range(N_LAYERS):
        out[f"fp16_L{i:02d}"] = {**fp16_set_config(every, [i]), "why": f"layer {i} exact, every other layer int8"}
    return out


def informative_configs(names: list[str]) -> dict[str, dict]:
    """Outside the rule's search (`run --informative`, written after the fixed search found no set):
    contiguous fp16 blocks inside layers 0-15, and one kind at a time kept exact inside layers 0-15.
    They map the error; the rule never selects from them."""
    every = sorted(names)
    out = {}
    for lo, hi in ((0, 3), (0, 5), (0, 7), (0, 11), (4, 15), (8, 15), (12, 15)):
        out[f"fp16_L{lo:02d}-{hi:02d}"] = {**fp16_set_config(every, range(lo, hi + 1)),
                                           "why": f"layers {lo}-{hi} exact, every other layer int8"}
    for kind in ("gdn", "mlp", "attn"):
        out[f"first16_{kind}_exact"] = {
            "int8": [n for n in every if not (layer_of(n) < 16 and kind_of(n) == kind)], "fp16_layers": None,
            "why": f"the {kind} linears of layers 0-15 exact, every other linear int8"}
    return out


def meets(rec: dict, base: dict) -> dict:
    """Rule 1-3 for one evaluated config against the int8lin record (same runs)."""
    runs = rec["runs"]
    worst = max(v["max_abs_dp"] for v in runs.values())
    not_worse = all(v["max_abs_dp"] <= base["runs"][k]["max_abs_dp"] for k, v in runs.items())
    n = len(rec["fp16_layers"]) if rec.get("fp16_layers") is not None else None
    return {"worst_max_abs_dp": worst, "rule_1": worst <= RULE["limits"]["worst_max_abs_dp"],
            "rule_2": not_worse, "rule_3": n is not None and 0 < n <= RULE["limits"]["max_fp16_layers"],
            "fp16_layer_count": n}


def ranking(part: dict) -> list[int]:
    singles = [(max(v["max_abs_dp"] for v in part["configs"][f"fp16_L{i:02d}"]["runs"].values()), i)
               for i in range(N_LAYERS)]
    return [i for _, i in sorted(singles)]


# --------------------------------------------------------------------------- #
# dump
# --------------------------------------------------------------------------- #
def dump(args) -> None:
    import torch
    import torch.nn.utils.parametrize as P
    from export_decoder import HF_ID, linear_quant_config
    from qwen3_5_clef_decoder import Qwen3_5ClefDecoder
    from safetensors.torch import save_file

    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    from coreai_models.export.compression import quantize_pytorch_model

    torch.set_num_threads(args.threads)
    t0 = time.monotonic()
    model = Qwen3_5ClefDecoder.from_hf(HF_ID, target_dtype=torch.float16, max_context_length=4096,
                                       n_image_max=1024)
    for layer in model.model.layers:          # the exporter's flags, before quantizing
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True
            layer.linear_attn.use_loopfree_unroll = True
    spec = model.build_export_spec(torch.float16, 4096, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, query_len=CHUNK)
    cfg = linear_quant_config("int8")
    qspec = json.loads(json.dumps(cfg["global_config"]["op_state_spec"]["weight"]))
    before = {n: m.weight.detach().clone() for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    model = quantize_pytorch_model(model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg)
    t_q = time.monotonic() - t0
    DUMP.mkdir(parents=True, exist_ok=True)
    per_layer: dict[int, dict] = {}
    info = {}
    for name, mod in model.named_modules():
        if not P.is_parametrized(mod, "weight"):
            continue
        assert isinstance(mod, torch.nn.Linear), (name, type(mod).__name__)
        deq = [p for p in mod.parametrizations["weight"] if hasattr(p, "quantized_data")]
        assert len(deq) == 1, (name, [type(p).__name__ for p in mod.parametrizations["weight"]])
        d = deq[0]
        w = mod.weight.detach().clone()                    # the finalized module's own dequantization
        again = torch.ops.coreai.constexpr_blockwise_shift_scale(
            d.quantized_data, d.scale, zero_point=d.zero_point, minval=d.minval,
            input_dtype=d.input_dtype, output_dtype=d.output_dtype)
        assert torch.equal(w, again), name
        q = d.quantized_data
        info[name] = {"shape": list(w.shape), "dtype": str(w.dtype), "codes_dtype": str(q.dtype),
                      "codes_min": int(q.min()), "codes_max": int(q.max()),
                      "scale_shape": list(d.scale.shape), "scale_dtype": str(d.scale.dtype),
                      "zero_point": None if d.zero_point is None else int(d.zero_point.abs().max()),
                      "rel_err": float((w.float() - before[name].float()).norm() / before[name].float().norm())}
        per_layer.setdefault(layer_of(name), {})[name] = w.contiguous()
    files = {}
    for li, ws in sorted(per_layer.items()):
        p = DUMP / f"layer_{li:02d}.safetensors"
        save_file(ws, str(p))
        files[str(li)] = {"file": str(p), "sha256": sha256(p), "modules": sorted(ws)}
    missing = sorted(set(before) - set(info))
    meta = {"weight_spec": qspec, "hf_id": HF_ID, "export_spec_query_len": CHUNK,
            "quantized_modules": len(info), "linear_modules_not_quantized": missing,
            "params": int(sum(int(np.prod(v["shape"])) for v in info.values())), "quantize_seconds": t_q,
            "files": files, "modules": info, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (DUMP.parent / "int8_clipping.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"dumped {len(info)} modules ({meta['params']:,} params) to {DUMP}; not quantized: {missing}; "
          f"rel err median {np.median([v['rel_err'] for v in info.values()]):.4e}; codes "
          f"{min(v['codes_min'] for v in info.values())}..{max(v['codes_max'] for v in info.values())}", flush=True)


# --------------------------------------------------------------------------- #
# rule
# --------------------------------------------------------------------------- #
def rule(args) -> None:
    if RULE_PATH.exists() and not args.force:
        sys.exit(f"{RULE_PATH} exists (the rule is written once, before any result)")
    runs = [k.split(":") for k in args.runs.split(",")]
    if not 1 <= len(runs) <= 6:
        sys.exit("1..6 bisect runs")
    doc = {"rule": RULE, "runs": runs, "why_runs": args.why, "chunk": CHUNK,
           "script_sha256": sha256(Path(__file__).resolve()),
           "written_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    RULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RULE_PATH.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"rule -> {RULE_PATH} ({sha256(RULE_PATH)})")


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
class Weights:
    """Swaps the quantizable modules' weights between exact (bf16-exact fp32) and int8 (the dump),
    reading both from per-layer safetensors so only the fp32 module stays resident."""

    def __init__(self, model, device):
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        self.torch, self.device = torch, device
        meta = json.loads((DUMP.parent / "int8_clipping.json").read_text())
        self.names = sorted(meta["modules"])
        mods = dict(model.named_modules())
        self.mods = {n: mods[n] for n in self.names}
        assert all(isinstance(m, torch.nn.Linear) for m in self.mods.values())
        EXACT.mkdir(parents=True, exist_ok=True)
        for li in sorted({layer_of(n) for n in self.names}):
            p = EXACT / f"layer_{li:02d}.safetensors"
            if p.exists():
                continue
            ws = {}
            for n in self.names:
                if layer_of(n) == li:
                    w = self.mods[n].weight.detach()
                    b = w.to(torch.bfloat16)
                    assert torch.equal(b.float(), w), f"{n}: fp32 weight is not bf16-exact"
                    ws[n] = b.contiguous()
            save_file(ws, str(p))
        self.files = {"int8": {li: safe_open(str(DUMP / f"layer_{li:02d}.safetensors"), framework="pt", device="cpu")
                               for li in sorted({layer_of(n) for n in self.names})},
                      "exact": {li: safe_open(str(EXACT / f"layer_{li:02d}.safetensors"), framework="pt", device="cpu")
                                for li in sorted({layer_of(n) for n in self.names})}}
        self.state = {n: "exact" for n in self.names}

    def apply(self, int8: list[str]) -> int:
        want = set(int8)
        changed = 0
        with self.torch.no_grad():
            for n, m in self.mods.items():
                s = "int8" if n in want else "exact"
                if self.state[n] == s:
                    continue
                src = self.files[s][layer_of(n)].get_tensor(n)
                m.weight.copy_(src.to(self.torch.float32).to(m.weight.device))
                self.state[n] = s
                changed += 1
        return changed


def run(args) -> None:
    import torch
    from parity_decoder_torch import AuthorHead, Runner, load_oracle, score_run

    rdoc = json.loads(RULE_PATH.read_text())
    runs = [tuple(k) for k in rdoc["runs"]]
    doc, rows, o = load_oracle()
    by_key = o["by_key"]
    part_path = Path(args.part)
    part = json.loads(part_path.read_text()) if part_path.exists() else {"configs": {}}
    t0 = time.monotonic()
    head = AuthorHead()
    runner = Runner(args.threads)
    W = Weights(runner.model, args.device)
    if args.device != "cpu":
        runner.to(args.device)
    load_s = time.monotonic() - t0
    cfgs = configs(W.names)
    if args.informative:
        cfgs = informative_configs(W.names)
        part["informative"] = True
    todo = args.configs.split(",") if args.configs else list(cfgs)
    for extra in (args.sets or "").split(";"):
        if extra.strip():
            layers = [int(x) for x in extra.split(",")]
            cfgs[set_name(layers)] = fp16_set_config(W.names, layers)
            todo.append(set_name(layers))
    unknown = [c for c in todo if c not in cfgs]
    if unknown:
        sys.exit(f"unknown configs {unknown}")
    inputs = {k: (by_key[k], np.load(LANE / "oracle" / "npz" / f"{k[0]}__{k[1]}.npz")) for k in runs}
    part.update({"pid": os.getpid(), "threads": args.threads, "device": args.device, "chunk": CHUNK,
                 "runs": [list(k) for k in runs], "rule_sha256": sha256(RULE_PATH),
                 "script_sha256": sha256(Path(__file__).resolve()), "load_seconds": load_s,
                 "dump_meta_sha256": sha256(DUMP.parent / "int8_clipping.json")})

    def save() -> None:
        tmp = part_path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(part, indent=1) + "\n")
        os.replace(tmp, part_path)

    def evaluate(name: str, c: dict) -> dict:
        t1 = time.monotonic()
        changed = W.apply(c["int8"])
        t_swap = time.monotonic() - t1
        rec = {"why": c["why"], "fp16_layers": c.get("fp16_layers"), "int8_modules": len(c["int8"]),
               "exact_modules": len(W.names) - len(c["int8"]),
               "int8_params": int(sum(W.mods[n].weight.numel() for n in c["int8"])),
               "modules_swapped": changed, "swap_seconds": t_swap, "runs": {}}
        for k in runs:
            row, npz = inputs[k]
            t2 = time.monotonic()
            res = runner.run(runner.static_inputs(row, npz), S=CHUNK, conv_check=False)
            sc = score_run(head, row, npz, res)
            far = [q for q in sc["questions"] if not q["near_tie"]]
            rec["runs"][f"{k[0]}:{k[1]}"] = {
                "max_abs_dp": sc["max_abs_dp"], "mean_abs_dp": sc["mean_abs_dp"],
                "argmax_all_equal": sc["argmax_all_equal"],
                "argmax_equal_non_near_tie": all(q["argmax_equal"] for q in far),
                "min_pos_cos": sc["hidden"]["min_pos_cos"], "hidden_max_abs_diff": sc["hidden"]["max_abs_diff"],
                "questions": [{"question_id": q["question_id"], "max_abs_dp": q["max_abs_dp"],
                               "argmax_equal": q["argmax_equal"], "near_tie": q["near_tie"], "probs": q["probs"]}
                              for q in sc["questions"]],
                "tokens": res["T"], "chunks": res["chunks"], "seconds": time.monotonic() - t2}
        rec["seconds"] = time.monotonic() - t1
        part["configs"][name] = rec
        save()
        msg = " ".join(f"{k}={v['max_abs_dp']:.4f}" for k, v in rec["runs"].items())
        print(f"[{os.getpid()}] {name}: worst {max(v['max_abs_dp'] for v in rec['runs'].values()):.4f} | {msg} "
              f"({rec['seconds']:.0f}s, {changed} swapped in {t_swap:.1f}s)", flush=True)
        return rec

    print(f"[{os.getpid()}] loaded in {load_s:.0f}s on {args.device}; runs {runs}; configs {len(todo)}", flush=True)
    for name in todo:
        if name in part["configs"] and not args.redo:
            print(f"[{os.getpid()}] {name}: done already", flush=True)
            continue
        evaluate(name, cfgs[name])
    if args.candidates:
        base = part["configs"]["int8lin"]
        order = ranking(part)
        part["ranking"] = order
        save()
        print(f"[{os.getpid()}] single-layer ranking {order}", flush=True)
        if any(all(meets(part["configs"][f"fp16_L{i:02d}"], base)[r] for r in ("rule_1", "rule_2", "rule_3"))
               for i in range(N_LAYERS)):
            print(f"[{os.getpid()}] a single layer meets the rule: no candidate sets", flush=True)
            return
        for k in range(2, RULE["limits"]["max_fp16_layers"] + 1):
            top = order[:k]
            name = set_name(top)
            rec = part["configs"].get(name) or evaluate(name, fp16_set_config(W.names, top))
            m = meets(rec, base)
            if m["rule_1"] and m["rule_2"] and m["rule_3"]:
                for sub in itertools.combinations(sorted(order[:k + 1]), k):
                    sn = set_name(sub)
                    if sn not in part["configs"]:
                        evaluate(sn, fp16_set_config(W.names, sub))
                print(f"[{os.getpid()}] size {k} reached: stop", flush=True)
                return
        print(f"[{os.getpid()}] no top-k set (k <= 6) met the rule", flush=True)


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def merge(args) -> None:
    rdoc = json.loads(RULE_PATH.read_text())
    parts = sorted((WORK / "parts").glob("part_*.json"))
    table, procs = {}, []
    for p in parts:
        d = json.loads(p.read_text())
        procs.append({"part": str(p), "pid": d.get("pid"), "device": d.get("device"), "threads": d.get("threads"),
                      "load_seconds": d.get("load_seconds"), "rule_sha256": d.get("rule_sha256"),
                      "script_sha256": d.get("script_sha256"), "configs": list(d["configs"]),
                      "ranking": d.get("ranking"), "informative": bool(d.get("informative"))})
        for name, rec in d["configs"].items():
            rec["outside_rule_search"] = bool(d.get("informative"))
            if name in table:
                a = {k: v["max_abs_dp"] for k, v in table[name]["runs"].items()}
                b = {k: v["max_abs_dp"] for k, v in rec["runs"].items()}
                rec["repeat_equal"] = a == b
            table[name] = rec
    base = table.get("int8lin")
    rows = []
    for name, rec in table.items():
        r = {"config": name, "outside_rule_search": rec["outside_rule_search"],
             "fp16_layers": rec.get("fp16_layers"), "int8_params": rec["int8_params"],
             "argmax_all_equal": all(v["argmax_all_equal"] for v in rec["runs"].values()),
             "argmax_equal_non_near_tie": all(v["argmax_equal_non_near_tie"] for v in rec["runs"].values()),
             "seconds": rec["seconds"], **{k: v["max_abs_dp"] for k, v in rec["runs"].items()}}
        if base:
            r.update(meets(rec, base))
        rows.append(r)
    ok = [r for r in rows if r.get("rule_1") and r.get("rule_2") and r.get("rule_3") and not r["outside_rule_search"]]
    chosen = min(ok, key=lambda r: (len(r["fp16_layers"]), sorted(r["fp16_layers"]))) if ok else None
    out = {"schema": "clef-flash-int8-bisect/1", "rule": rdoc["rule"], "rule_file": str(RULE_PATH),
           "rule_sha256": sha256(RULE_PATH), "rule_written_at": rdoc["written_at"], "runs": rdoc["runs"],
           "why_runs": rdoc["why_runs"],
           "instrument": ("fp32 torch (parity_decoder_torch.Runner), S = 64 chunks from fresh zero states, oracle ids + "
                          "oracle fp32 image_embeds, the author's fp32 head; per config the listed modules carry the "
                          "exporter's int8 weights (quantize_pytorch_model on the fp16 model, dequantized by the "
                          "finalized module), every other quantizable module its exact checkpoint weight"),
           "script": {"path": "conversion/clef_flash/int8_bisect_torch.py", "sha256": sha256(Path(__file__).resolve())},
           "oracle": {"path": str(LANE / "oracle" / "records_oracle.json"),
                      "sha256": sha256(LANE / "oracle" / "records_oracle.json")},
           "dump": {"meta": str(DUMP.parent / "int8_clipping.json"), "sha256": sha256(DUMP.parent / "int8_clipping.json")},
           "processes": procs, "chosen": chosen, "candidates_meeting_rule": ok, "table": rows, "configs": table,
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(table)} configs -> {args.out}; chosen {None if chosen is None else chosen['config']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dump")
    a.add_argument("--threads", type=int, default=8)
    r = sub.add_parser("rule")
    r.add_argument("--runs", required=True, help="comma list id:arm (1..6)")
    r.add_argument("--why", required=True, help="how the runs were picked")
    r.add_argument("--force", action="store_true")
    b = sub.add_parser("run")
    b.add_argument("--part", required=True)
    b.add_argument("--configs", help="comma list (default: every fixed config)")
    b.add_argument("--sets", help="extra fp16 layer sets, ';'-separated lists of layer indices")
    b.add_argument("--candidates", action="store_true", help="after the configs, the rule's candidate-set search")
    b.add_argument("--informative", action="store_true",
                   help="the informative_configs() map instead of the rule's search (never selected)")
    b.add_argument("--device", default="mps", choices=["mps", "cpu"])
    b.add_argument("--threads", type=int, default=8)
    b.add_argument("--redo", action="store_true")
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    args = ap.parse_args()
    {"dump": dump, "rule": rule, "run": run, "merge": merge}[args.cmd](args)


if __name__ == "__main__":
    main()
