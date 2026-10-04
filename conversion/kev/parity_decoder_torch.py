#!/usr/bin/env python3
"""fp32 torch parity: the Kev decoder module vs the author's fp32 oracle, read through the author's pointer head.

`qwen3_5_kev_decoder.Qwen3_5KevDecoder` (fp32, every GDN layer on `use_loopfree_unroll`) is driven the way
its Core AI graph will run: one row per question (the oracle's `row_ids` = state + branch, positions
0..L-1), fresh zero states per row, the row in static S-token chunks (position_ids = the cache ramp
0..cS+S-1), the last chunk padded with <|endoftext|> 248044 and the padded positions' outputs discarded.
The hidden state at every position goes through the pointer head read from `oracle/head/head.safetensors`
(fp32 q / k Linear: z = (k(h_opts) @ q(h_decide)) * 0.0625, then z / T, softmax over the question's
options; the author's `PointerHead.forward` op for op) and is compared with the oracle
(`oracle/records_oracle.json`, read only):

  probe  tv4_000 q0 / own_t01 q0 / own_j03 q0 under F.conv1d and a bmm depthwise conv at 1 / 4 / 12
         threads: seconds per token, hidden max |d| against F.conv1d at 1 thread. P1 uses the fastest.
  P1     every question (434 rows), S = 16. Per question argmax, max |dp| and max |dlogit| (after the
         temperature) vs the oracle; the six records with oracle hidden states (`oracle/hidden/<id>.npz`)
         also get per-position cosine and max |d| of the hidden rows.
         Bar (fixed before running): argmax equal on every question (near-ties included),
         max |dp| <= 1e-4, min position cos >= 0.9999 on the six hidden records.
  P2     tv4_000 q0, semif_a3f18f3a63d45345942b q0, own_L02 q0: this module vs the overlay's plain
         `Qwen3_5StatefulForCausalLM.model.forward_stateful` (loaded in the same process with the same
         loader arguments, same unroll, same chunks, same conv): hidden bit-equal at every position and
         the sha256 of every decoder layer's output equal per chunk.
  P3     the 21 rows of the six hidden records at S = 16 / 32 / 64 / 128: hidden max |d| and p max |d|
         between widths and against the oracle (the chunk-width sensitivity of fp32 torch; recorded,
         not a bar). S = 16 is re-run here and compared bit for bit with P1's run of the same rows.
  scan   (round 11) P3's 21 rows with the GDN scan on the overlay's in-graph chunk form at --p3-widths
         (default 16,32) against P1's unroll S=16 hidden rows and the oracle, with the unroll S=16 control
         in the same process -> results/r11_torch_chunk.json. The Metal kernel form has no torch values.

CPU fp32 is the reference. Harness-only substitution (the module is untouched): `--conv bmm` evaluates the
GDN's valid depthwise conv (`F.conv1d`, groups = 6144, 3 + S columns) as one `torch.bmm` over the same
windows, and at the last chunk of every row all 18 layers' results are checked against `F.conv1d` itself
(`torch.equal`, every channel); the counts go into the transcript. The probe measures whether it pays.

    cd conversion/kev
    export HF_HOME=$ZOO_WORK_ROOT/_kev/hf HF_HUB_OFFLINE=1
    PY=<coreai-models venv>/bin/python
    $PY parity_decoder_torch.py probe
    for k in 0 1 2; do $PY parity_decoder_torch.py p1 --shard $k --shards 3 --threads 1 --conv bmm & done; wait
    $PY parity_decoder_torch.py p2 --threads 1 --conv bmm
    $PY parity_decoder_torch.py p3 --threads 1 --conv bmm
    $PY parity_decoder_torch.py merge          # -> results/parity_decoder_torch.json

Kev-4B (round 4): `--model kev-4b` on every stage reads `kev-local/kev-4b-v1.0-merged`, `oracle_4b/` (and its
`head/`), writes `parity_4b/` and `results/parity_decoder_torch_4b.json`, and expects hidden 2560.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

LANE = work_path("_kev")
os.environ.setdefault("HF_HOME", str(LANE / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

HF_ID = "kev-local/kev-0.8b-v1.0-merged"     # L/hf: the author's merge_lora_checkpoint.py output, symlinked
ORACLE = LANE / "oracle"
HEAD_DIR = ORACLE / "head"
PARITY = LANE / "parity"
PAD_ID = 248044
HIDDEN = 1024
CHUNK = 16
WIDTHS = (16, 32, 64, 128)
BAR = {"argmax_equal": "every question, near-ties included", "max_abs_dp": 1e-4, "min_pos_cos": 0.9999}
HIDDEN_RECORDS = ("tv4_000", "semif_a3f18f3a63d45345942b", "own_t01", "own_j03", "own_L02", "own_m01")
PROBE_ROWS = (("tv4_000", 0), ("own_t01", 0), ("own_j03", 0))
P2_ROWS = (("tv4_000", 0), ("semif_a3f18f3a63d45345942b", 0), ("own_L02", 0))
MODELS = {"kev-0.8b": {"hf_id": HF_ID, "oracle": "oracle", "parity": "parity", "hidden": 1024, "suffix": ""},
          "kev-4b": {"hf_id": "kev-local/kev-4b-v1.0-merged", "oracle": "oracle_4b", "parity": "parity_4b", "hidden": 2560,
                     "suffix": "_4b"}}


def configure(model: str) -> None:
    """--model: the merged checkpoint's local id, the oracle directory (and its head/), the parity directory, the hidden size."""
    global HF_ID, ORACLE, HEAD_DIR, PARITY, HIDDEN
    m = MODELS[model]
    HF_ID, HIDDEN = m["hf_id"], m["hidden"]
    ORACLE = LANE / m["oracle"]
    HEAD_DIR = ORACLE / "head"
    PARITY = LANE / m["parity"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_oracle() -> tuple[dict, dict]:
    doc = json.loads((ORACLE / "records_oracle.json").read_text())
    return doc, {r["id"]: r for r in doc["records"]}


def all_rows(recs: dict) -> list[tuple[str, int]]:
    return [(rid, k) for rid, r in recs.items() for k in range(len(r["questions"]))]


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def compare_hidden(mine: np.ndarray, ref: np.ndarray) -> dict:
    d = np.abs(mine.astype(np.float64) - ref.astype(np.float64))
    c = cos_rows(mine, ref)
    i = int(c.argmin())
    ref_max = float(np.abs(ref).max())
    return {"max_abs_diff": float(d.max()), "ref_absmax": ref_max, "rel_max_abs_diff": float(d.max()) / ref_max,
            "min_pos_cos": float(c[i]), "min_pos_cos_index": i, "mean_pos_cos": float(c.mean()),
            "positions_below_0.9999": int((c < 0.9999).sum())}


class DepthwiseConvBmm:
    """F.conv1d replacement for the GDN's valid depthwise conv on CPU tensors (batch 1, no bias, stride 1),
    evaluated as one bmm; every other call goes to the original. With `check` set, the original runs too
    and the two are compared (torch.equal). The type of conversion/clef_flash/parity_decoder_torch.py."""

    def __init__(self, original):
        self.original = original
        self.check = False
        self.fast_calls = self.checked = self.mismatched = 0
        self.max_abs_diff = 0.0

    def __call__(self, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        C = weight.shape[0]
        if not (input.device.type == "cpu" and bias is None and input.dim() == 3 and input.shape[0] == 1
                and input.shape[1] == C and groups == C and weight.shape[1] == 1 and stride in (1, (1,))
                and padding in (0, (0,)) and dilation in (1, (1,))):
            return self.original(input, weight, bias, stride, padding, dilation, groups)
        import torch
        K = weight.shape[-1]
        win = input[0].unfold(-1, K, 1)                                  # [C, L, K]
        out = torch.bmm(weight, win.transpose(1, 2)).reshape(1, C, -1)   # [C,1,K] @ [C,K,L]
        self.fast_calls += 1
        if self.check:
            ref = self.original(input, weight, bias, stride, padding, dilation, groups)
            self.checked += 1
            if not torch.equal(ref, out):
                self.mismatched += 1
                self.max_abs_diff = max(self.max_abs_diff, float((ref - out).abs().max()))
        return out

    def report(self) -> dict:
        return {"fast_calls": self.fast_calls, "checked_vs_F_conv1d": self.checked,
                "mismatched": self.mismatched, "max_abs_diff": self.max_abs_diff}


class KevHead:
    """The pointer head from head.safetensors in fp32 on the CPU: `PointerHead.forward` op for op
    (z = (k(h_opts) @ q(h_decide)) * scale, then z / temperature)."""

    def __init__(self, head_dir: Path | None = None, hidden: int | None = None):
        import torch
        from safetensors.torch import load_file

        self.torch = torch
        head_dir = Path(head_dir) if head_dir else HEAD_DIR
        hidden = hidden or HIDDEN
        info = json.loads((head_dir / "kev_head.json").read_text())
        sd = load_file(str(head_dir / "head.safetensors"))
        self.qw, self.qb = sd["q.weight"].float(), sd["q.bias"].float()
        self.kw, self.kb = sd["k.weight"].float(), sd["k.bias"].float()
        self.scale, self.temperature = float(info["scale"]), float(info["temperature"])
        assert self.qw.shape == (info["head_dim"], hidden) and abs(self.scale - info["head_dim"] ** -0.5) < 1e-12
        self.provenance = {"head_safetensors": str(head_dir / "head.safetensors"),
                           "head_safetensors_sha256": sha256_file(head_dir / "head.safetensors"),
                           "kev_head_json_sha256": sha256_file(head_dir / "kev_head.json"),
                           "scale": self.scale, "temperature": self.temperature, "dtype": "float32", "device": "cpu"}

    def logits_T(self, hidden, decide: int, opts: list[int]):
        """hidden [T, 1024] fp32 torch -> temperature-scaled logits [K] (fp32 torch)."""
        F = self.torch.nn.functional
        q = F.linear(hidden[decide], self.qw, self.qb)
        k = F.linear(hidden[self.torch.tensor(opts)], self.kw, self.kb)
        z = (k @ q) * self.scale
        return z / self.temperature


def score_question(head: KevHead, hidden: np.ndarray, q: dict) -> dict:
    import torch
    lt = head.logits_T(torch.from_numpy(np.ascontiguousarray(hidden, dtype=np.float32)), q["decide"], q["opts"])
    p = torch.softmax(lt, -1).double().numpy()
    po = np.asarray(q["probs"], np.float64)
    dp = np.abs(p - po)
    am = int(p.argmax())
    am_o = q["keys"].index(q["argmax"])
    return {"argmax": am, "argmax_oracle": am_o, "argmax_equal": am == am_o,
            "max_abs_dp": float(dp.max()), "mean_abs_dp": float(dp.mean()),
            "max_abs_dlogit_T": float(np.abs(lt.double().numpy() - np.asarray(q["logits_T"], np.float64)).max()),
            "near_tie": bool(q["near_tie"]), "oracle_top2_margin": q["top2_margin"], "n_options": len(po),
            "probs": [float(v) for v in p]}


class Runner:
    """The module, the chunk loop and the taps."""

    def __init__(self, threads: int, conv: str):
        import torch
        import torch.nn.functional as F

        torch.set_num_threads(threads)
        self.torch = torch
        from qwen3_5_kev_decoder import Qwen3_5KevDecoder, set_unrolled_scan

        from coreai_models.models.macos import qwen3_5 as q

        self.q = q
        t0 = time.monotonic()
        self.model = Qwen3_5KevDecoder.from_hf(HF_ID, torch.float32, max_context_length=4096)
        self.load_seconds = time.monotonic() - t0
        self.cfg = self.model.config
        assert self.cfg.hidden_size == HIDDEN
        self.unrolled = set_unrolled_scan(self.model)
        self.conv_kind = conv
        self.conv = DepthwiseConvBmm(F.conv1d)
        self.original_conv1d = F.conv1d
        self.set_conv(conv)

    def set_conv(self, conv: str) -> None:
        import torch.nn.functional as F

        self.conv_kind = conv
        F.conv1d = self.conv if conv == "bmm" else self.original_conv1d

    def run(self, ids: list[int], S: int = CHUNK, *, model=None, hashes: bool = False, conv_check: bool = True) -> dict:
        """One row from fresh zero states in S-token chunks -> hidden [T, 1024] fp32 + extras."""
        torch = self.torch
        model = model or self.model
        T = len(ids)
        n = -(-T // S)
        Tp = n * S
        ids_p = torch.full((Tp,), PAD_ID, dtype=torch.int32)
        ids_p[:T] = torch.tensor(ids, dtype=torch.int32)
        st = self.q.build_decode_state(self.cfg, max_seq_len=Tp, dtype=torch.float32)
        layers = model.model.layers
        cur: list = []
        hooks = [layer.register_forward_hook(lambda mod, args, out: cur.append(out.detach().clone()))
                 for layer in layers] if hashes else []
        outs, layer_hashes = [], []
        t0 = time.monotonic()
        try:
            with torch.inference_mode():
                for c in range(n):
                    self.conv.check = conv_check and c == n - 1
                    cur.clear()
                    tok = ids_p[c * S:(c + 1) * S].reshape(1, S)
                    pos = torch.arange((c + 1) * S, dtype=torch.int32).unsqueeze(0)
                    if model is self.model:
                        h = model(tok, pos, st["k_cache"], st["v_cache"], st["conv_state"], st["rec_state"])
                    else:   # the plain overlay model: its text graph's own entry
                        from coreai_models.primitives.macos.cache import KVCache, SSMState
                        h = model.model.forward_stateful(tok, pos, KVCache(st["k_cache"], st["v_cache"]),
                                                         SSMState(st["conv_state"]), SSMState(st["rec_state"]))
                    outs.append(h[0])
                    if hashes:
                        layer_hashes.append([hashlib.sha256(x.numpy().tobytes()).hexdigest()[:16] for x in cur]
                                            + [hashlib.sha256(h[0].numpy().tobytes()).hexdigest()[:16]])
            hidden = torch.cat(outs)[:T].numpy().astype(np.float32)
        finally:
            for hk in hooks:
                hk.remove()
            self.conv.check = False
        out = {"hidden": hidden, "T": T, "Tp": Tp, "chunks": n, "S": S, "seconds": time.monotonic() - t0}
        if hashes:
            out["layer_hashes"] = layer_hashes
        return out


def hidden_npz(rid: str, k: int, S: int) -> Path:
    return PARITY / "hidden" / f"{rid}__q{k}__s{S}.npz"


def save_hidden(rid: str, k: int, S: int, hidden: np.ndarray) -> str:
    p = hidden_npz(rid, k, S)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, hidden=hidden)
    os.replace(tmp, p)
    return str(p)


def oracle_hidden(rid: str, k: int) -> np.ndarray | None:
    p = ORACLE / "hidden" / f"{rid}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z[f"q{k}_hidden"] if f"q{k}_hidden" in z.files else None


def head_floor(head: KevHead, rid: str, k: int, q: dict) -> dict | None:
    """The harness head on the ORACLE hidden rows vs the oracle's logits (the head's own floor)."""
    import torch
    ref = oracle_hidden(rid, k)
    if ref is None:
        return None
    lt = head.logits_T(torch.from_numpy(ref), q["decide"], q["opts"]).numpy()
    o = np.asarray(q["logits_T"], np.float32)
    return {"bit_equal": bool(np.array_equal(lt.astype(np.float32), o)),
            "max_abs_dlogit_T": float(np.abs(lt.astype(np.float64) - o.astype(np.float64)).max())}


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_probe(args) -> None:
    import torch
    _, recs = load_oracle()
    runner = Runner(1, "f_conv1d")
    rows = [(rid, k, recs[rid]["questions"][k]) for rid, k in PROBE_ROWS]
    ref: dict = {}
    configs, best = [], None
    for conv in ("f_conv1d", "bmm"):
        for threads in (1, 4, 12):
            runner.set_conv(conv)
            torch.set_num_threads(threads)
            per = []
            for i, (rid, k, q) in enumerate(rows):
                res = runner.run(q["row_ids"], CHUNK, conv_check=False)
                key = (rid, k)
                if conv == "f_conv1d" and threads == 1:
                    ref[key] = res["hidden"]
                d = float(np.abs(res["hidden"].astype(np.float64) - ref[key]).max()) if key in ref else None
                per.append({"id": rid, "q": k, "T": res["T"], "seconds": res["seconds"],
                            "ms_per_token": res["seconds"] * 1e3 / res["T"], "hidden_max_abs_diff_vs_conv1d_t1": d,
                            "bit_equal_vs_conv1d_t1": bool(d == 0.0) if d is not None else None})
                print(f"probe conv={conv} threads={threads} {rid} q{k} T={res['T']} {res['seconds']:.2f}s "
                      f"d={d}", flush=True)
                # a config 3x slower than the best on the first row is not finished (it cannot win)
                if i == 0 and best is not None and res["seconds"] > 3 * best["first_row_seconds"]:
                    break
            tot_t = sum(p["T"] for p in per)
            tot_s = sum(p["seconds"] for p in per)
            cfg = {"conv": conv, "threads": threads, "rows": per, "complete": len(per) == len(rows),
                   "ms_per_token": tot_s * 1e3 / tot_t, "first_row_seconds": per[0]["seconds"]}
            configs.append(cfg)
            if cfg["complete"] and (best is None or cfg["ms_per_token"] < best["ms_per_token"]):
                best = cfg
    out = {"schema": "kev-decoder-torch-parity-probe/1", "rows": [f"{r}:q{k}" for r, k in PROBE_ROWS],
           "chunk": CHUNK, "configs": configs, "fastest": {"conv": best["conv"], "threads": best["threads"],
                                                            "ms_per_token": best["ms_per_token"]},
           "load_seconds": runner.load_seconds, "conv_report": runner.conv.report(), "finished": now()}
    PARITY.mkdir(parents=True, exist_ok=True)
    (PARITY / "probe.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"fastest: {json.dumps(out['fastest'])}", flush=True)


def shard_rows(rows: list[tuple[str, int]], recs: dict, shard: int, shards: int) -> list[tuple[str, int]]:
    """Balance by padded tokens: longest first, each to the lightest shard (deterministic)."""
    load = [0] * shards
    own: list[list] = [[] for _ in range(shards)]
    for rid, k in sorted(rows, key=lambda x: (-recs[x[0]]["questions"][x[1]]["row_len"], x[0], x[1])):
        i = min(range(shards), key=lambda j: (load[j], j))
        own[i].append((rid, k))
        load[i] += -(-recs[rid]["questions"][k]["row_len"] // CHUNK) * CHUNK
    return own[shard]


def stage_p1(args) -> None:
    _, recs = load_oracle()
    rows = shard_rows(all_rows(recs), recs, args.shard, args.shards)
    out_path = PARITY / f"p1_shard{args.shard}of{args.shards}.jsonl"
    done = set()
    if out_path.exists():
        done = {(x["id"], x["q"]) for x in map(json.loads, out_path.read_text().splitlines()) if x.get("kind") == "p1"}
    head = KevHead()
    runner = Runner(args.threads, args.conv)
    PARITY.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps({"kind": "meta", "pid": os.getpid(), "started": now(), "shard": args.shard,
                            "shards": args.shards, "rows": len(rows), "threads": args.threads, "conv": args.conv,
                            "load_seconds": runner.load_seconds, "load_report": runner.model.load_report,
                            "unrolled_linear_layers": runner.unrolled, "head": head.provenance}) + "\n")
        f.flush()
        for rid, k in rows:
            if (rid, k) in done:
                continue
            q = recs[rid]["questions"][k]
            res = runner.run(q["row_ids"], CHUNK, conv_check=args.conv == "bmm")
            sc = score_question(head, res["hidden"], q)
            rec = {"kind": "p1", "id": rid, "q": k, "qid": q["qid"], "type": q["type"], "source": recs[rid]["source"],
                   "T": res["T"], "padded": res["Tp"], "chunks": res["chunks"], "seconds": res["seconds"], **sc}
            ref = oracle_hidden(rid, k) if rid in HIDDEN_RECORDS else None
            if ref is not None:
                rec["hidden"] = compare_hidden(res["hidden"], ref)
                rec["hidden_npz"] = save_hidden(rid, k, CHUNK, res["hidden"])
                rec["head_floor"] = head_floor(head, rid, k, q)
            rec["conv"] = runner.conv.report()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[p1 {args.shard}/{args.shards}] {rid} q{k} T={res['T']} argmax {'ok' if sc['argmax_equal'] else 'NO'} "
                  f"max|dp| {sc['max_abs_dp']:.2e} {res['seconds']:.1f}s"
                  + (f" cos {rec['hidden']['min_pos_cos']:.7f}" if "hidden" in rec else ""), flush=True)
        f.write(json.dumps({"kind": "end", "pid": os.getpid(), "finished": now(), "conv": runner.conv.report()}) + "\n")


def stage_p2(args) -> None:
    import torch

    from coreai_models.models.macos import qwen3_5 as qm

    _, recs = load_oracle()
    runner = Runner(args.threads, args.conv)
    t0 = time.monotonic()
    plain = qm.Qwen3_5StatefulForCausalLM.from_hf_memory_efficient(
        HF_ID, max_context_length=4096, target_dtype=torch.float32, hf_config_attr=None, hf_state_dict_prefix="")
    plain.eval()
    plain_load = time.monotonic() - t0
    n_plain = 0
    for layer in plain.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True
            layer.linear_attn.use_loopfree_unroll = True
            n_plain += 1
    rows = []
    for rid, k in P2_ROWS:
        q = recs[rid]["questions"][k]
        a = runner.run(q["row_ids"], CHUNK, hashes=True, conv_check=args.conv == "bmm")
        b = runner.run(q["row_ids"], CHUNK, model=plain, hashes=True, conv_check=args.conv == "bmm")
        first = None
        for c, (x, y) in enumerate(zip(a["layer_hashes"], b["layer_hashes"])):
            if x != y:
                li = next(i for i, (u, v) in enumerate(zip(x, y)) if u != v)
                first = {"chunk": c, "layer": li if li < len(plain.model.layers) else "final norm"}
                break
        rec = {"id": rid, "q": k, "T": a["T"], "chunks": a["chunks"],
               "hidden_bit_equal": bool(np.array_equal(a["hidden"], b["hidden"])),
               "max_abs_diff": float(np.abs(a["hidden"].astype(np.float64) - b["hidden"]).max()),
               "chunk_layer_hashes_equal": first is None and len(a["layer_hashes"]) == len(b["layer_hashes"]),
               "first_difference": first, "seconds": [a["seconds"], b["seconds"]]}
        rows.append(rec)
        print(f"P2 {rid} q{k} T={a['T']} bit-equal {rec['hidden_bit_equal']} hashes {rec['chunk_layer_hashes_equal']} "
              f"max|d| {rec['max_abs_diff']:.2e}", flush=True)
    out = {"kind": "p2", "pid": os.getpid(), "finished": now(), "threads": args.threads, "conv": args.conv,
           "module_load_seconds": runner.load_seconds, "plain_load_seconds": plain_load,
           "plain_class": "coreai_models.models.macos.qwen3_5.Qwen3_5StatefulForCausalLM",
           "plain_entry": "model.forward_stateful (embed_tokens + forward_stateful_core, in-graph plain RoPE)",
           "plain_loader": "from_hf_memory_efficient(HF_ID, max_context_length=4096, target_dtype=float32, "
                           "hf_config_attr=None, hf_state_dict_prefix='')",
           "plain_unrolled_linear_layers": n_plain, "rows": rows, "conv_report": runner.conv.report()}
    PARITY.mkdir(parents=True, exist_ok=True)
    (PARITY / "p2.json").write_text(json.dumps(out, indent=1) + "\n")


def stage_p3(args) -> None:
    _, recs = load_oracle()
    head = KevHead()
    runner = Runner(args.threads, args.conv)
    rows = [(rid, k) for rid in HIDDEN_RECORDS for k in range(len(recs[rid]["questions"]))]
    if args.p3_rows:   # e.g. P2's three rows: tv4_000:0,semif_a3f18f3a63d45345942b:0,own_L02:0
        want = [(x.rsplit(":", 1)[0], int(x.rsplit(":", 1)[1])) for x in args.p3_rows.split(",")]
        assert all(w in rows for w in want), want
        rows = want
    widths = tuple(args.p3_widths) if args.p3_widths else WIDTHS
    assert CHUNK in widths, "S = 16 is the reference width of P3"
    out_path = PARITY / "p3.jsonl"
    done = set()
    if out_path.exists():
        done = {(x["id"], x["q"], x["S"]) for x in map(json.loads, out_path.read_text().splitlines()) if x.get("kind") == "p3"}
    with open(out_path, "a") as f:
        f.write(json.dumps({"kind": "meta", "pid": os.getpid(), "started": now(), "threads": args.threads,
                            "conv": args.conv, "load_seconds": runner.load_seconds}) + "\n")
        for S in widths:
            for rid, k in rows:
                if (rid, k, S) in done:
                    continue
                q = recs[rid]["questions"][k]
                res = runner.run(q["row_ids"], S, conv_check=args.conv == "bmm")
                sc = score_question(head, res["hidden"], q)
                rec = {"kind": "p3", "id": rid, "q": k, "S": S, "T": res["T"], "padded": res["Tp"],
                       "chunks": res["chunks"], "seconds": res["seconds"],
                       "hidden": compare_hidden(res["hidden"], oracle_hidden(rid, k)),
                       **{x: sc[x] for x in ("argmax", "argmax_oracle", "argmax_equal", "max_abs_dp",
                                             "max_abs_dlogit_T", "probs")},
                       "npz": save_hidden(rid, k, S, res["hidden"]) if S != CHUNK else None}
                if S == CHUNK:   # P1's run of the same row, another process: bit for bit
                    p = hidden_npz(rid, k, CHUNK)
                    if p.exists():
                        rec["bit_equal_p1_run"] = bool(np.array_equal(np.load(p)["hidden"], res["hidden"]))
                    else:
                        rec["npz"] = save_hidden(rid, k, S, res["hidden"])
                        rec["bit_equal_p1_run"] = None
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"[p3] S={S} {rid} q{k} T={res['T']} max|dp| {sc['max_abs_dp']:.2e} "
                      f"cos {rec['hidden']['min_pos_cos']:.7f} {res['seconds']:.1f}s", flush=True)
        f.write(json.dumps({"kind": "end", "pid": os.getpid(), "finished": now(), "conv": runner.conv.report()}) + "\n")


def stage_scan(args) -> None:
    """Round 11: P3's 21 rows with the GDN scan in another form (`--gdn-scan chunk`: the overlay's in-graph chunk scan,
    ceil(log2 S) doublings) at each of `--p3-widths` (default 16,32), fp32 torch, against P1's S = 16 unroll run of the
    same row (parity/hidden/<row>__s16.npz: hidden max |d| and p through the head) and the oracle (max |dp|, argmax,
    position cos). The control: the unroll form at S = 16 in the same process, which must reproduce P1's npz bit for
    bit (so a zero difference of the other form would be visible as the harness not switching). The Metal kernel form
    has no torch values (its torch_defn returns zeros of the right shape) and is not run here."""
    from qwen3_5_kev_decoder import set_chunk_scan, set_unrolled_scan

    _, recs = load_oracle()
    head = KevHead()
    runner = Runner(args.threads, args.conv)
    rows = [(rid, k) for rid in HIDDEN_RECORDS for k in range(len(recs[rid]["questions"]))]
    widths = tuple(args.p3_widths) if args.p3_widths else (16, 32)
    configs = [("unroll", CHUNK)] + [(args.gdn_scan, S) for S in widths]
    u16 = {(rid, k): np.load(hidden_npz(rid, k, CHUNK))["hidden"] for rid, k in rows}
    p_u16 = {(rid, k): score_question(head, u16[(rid, k)], recs[rid]["questions"][k])["probs"] for rid, k in rows}
    out_rows, table = [], {}
    for scan, S in configs:
        info = {"gdn_scan": "unroll", "linear_layers": set_unrolled_scan(runner.model)}
        for layer in runner.model.model.layers:
            if not layer.is_full:
                layer.linear_attn.use_loopfree_chunk = False
        if scan == "chunk":
            info = set_chunk_scan(runner.model, S)
        res_rows = []
        for rid, k in rows:
            q = recs[rid]["questions"][k]
            res = runner.run(q["row_ids"], S, conv_check=args.conv == "bmm")
            sc = score_question(head, res["hidden"], q)
            ref = u16[(rid, k)]
            d = np.abs(res["hidden"].astype(np.float64) - ref.astype(np.float64))
            row = {"scan": scan, "S": S, "id": rid, "q": k, "T": res["T"], "chunks": res["chunks"],
                   "seconds": res["seconds"], "finite": bool(np.isfinite(res["hidden"]).all()),
                   "vs_u16_hidden_max_abs_diff": float(d.max()), "vs_u16_bit_equal": bool(np.array_equal(res["hidden"], ref)),
                   "vs_u16_max_abs_dp": float(np.abs(np.asarray(sc["probs"]) - np.asarray(p_u16[(rid, k)])).max()),
                   "vs_oracle": compare_hidden(res["hidden"], oracle_hidden(rid, k)),
                   **{x: sc[x] for x in ("argmax", "argmax_oracle", "argmax_equal", "max_abs_dp", "max_abs_dlogit_T",
                                         "near_tie", "probs")}}
            res_rows.append(row)
            print(f"[scan {scan} S={S}] {rid} q{k} T={res['T']} vs U16 max|dh| {row['vs_u16_hidden_max_abs_diff']:.2e} "
                  f"max|dp| {row['vs_u16_max_abs_dp']:.2e}; vs oracle max|dp| {sc['max_abs_dp']:.2e} "
                  f"cos {row['vs_oracle']['min_pos_cos']:.7f} {res['seconds']:.1f}s", flush=True)
        out_rows += res_rows
        table[f"{scan}{S}"] = {
            **info, "S": S, "rows": len(res_rows), "finite_rows": sum(r["finite"] for r in res_rows),
            "vs_u16_bit_equal_rows": sum(r["vs_u16_bit_equal"] for r in res_rows),
            "vs_u16_hidden_max_abs_diff": max(r["vs_u16_hidden_max_abs_diff"] for r in res_rows),
            "vs_u16_max_abs_dp": max(r["vs_u16_max_abs_dp"] for r in res_rows),
            "vs_oracle_argmax_equal": sum(r["argmax_equal"] for r in res_rows),
            "vs_oracle_max_abs_dp": max(r["max_abs_dp"] for r in res_rows),
            "vs_oracle_min_pos_cos": min(r["vs_oracle"]["min_pos_cos"] for r in res_rows),
            "vs_oracle_hidden_max_abs_diff": max(r["vs_oracle"]["max_abs_diff"] for r in res_rows),
            "seconds": float(sum(r["seconds"] for r in res_rows)), "chunks": int(sum(r["chunks"] for r in res_rows))}
        print(f"[scan {scan} S={S}] {json.dumps(table[f'{scan}{S}'])}", flush=True)
    here = Path(__file__).resolve()
    record = {"schema": "kev-decoder-torch-scan/1", "round": 11,
              "what": "fp32 torch, P3's 21 rows (the six oracle-hidden records), the GDN scan's form x S against P1's "
                      "unroll S=16 run and the author's fp32 oracle; the unroll S=16 row set is the in-process control "
                      "(bit-equal to P1 expected)",
              "harness": {"path": "conversion/kev/parity_decoder_torch.py", "sha256": sha256_file(here),
                          "threads": args.threads, "conv": args.conv, "conv_report": runner.conv.report()},
              "module": {"path": "conversion/kev/qwen3_5_kev_decoder.py", "sha256": sha256_file(here.parent / "qwen3_5_kev_decoder.py"),
                         "load_report": runner.model.load_report, "load_seconds": runner.load_seconds},
              "reference": {"u16": str(PARITY / "hidden"), "oracle": str(ORACLE / "records_oracle.json"),
                            "oracle_sha256": sha256_file(ORACLE / "records_oracle.json")},
              "head": head.provenance, "versions": versions(), "table": table,
              "rows": [{k: v for k, v in r.items() if k != "probs"} for r in out_rows],
              "not_run": {"metal": "the fp32 Metal chunk kernel has no torch values (torch_defn returns zeros of the "
                                   "right shape); its numerics exist only on the GPU (readout_gate.py)"},
              "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def versions() -> dict:
    import torch
    import transformers

    import coreai_models
    out = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
           "numpy": np.__version__, "platform": platform.platform(), "mkldnn": torch.backends.mkldnn.is_available()}
    try:
        import importlib.metadata as md
        out["coreai_torch"] = md.version("coreai-torch")
    except Exception:  # noqa: BLE001
        pass
    overlay = Path(coreai_models.__file__).resolve().parents[3]

    def git(*a):  # read-only; --no-optional-locks keeps `status` from rewriting the index
        return subprocess.run(["git", "--no-optional-locks", "-C", str(overlay), *a],
                              capture_output=True, text=True).stdout.rstrip("\n")
    src = overlay / "python/src/coreai_models"
    out["overlay"] = {"path": str(overlay), "branch": git("branch", "--show-current"),
                      "rev": git("rev-parse", "--short", "HEAD"),
                      "uncommitted_python": git("status", "--porcelain", "--", "python/src").splitlines(),
                      "sha256": {f: sha256_file(src / f) for f in ("models/macos/qwen3_5.py", "models/base.py",
                                                                   "primitives/macos/cache.py", "primitives/_ops.py")}}
    return out


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def summarize(qs: list[dict]) -> dict:
    near = [q for q in qs if q["near_tie"]]
    return {"questions": len(qs), "argmax_equal": sum(q["argmax_equal"] for q in qs),
            "near_tie_questions": len(near), "argmax_equal_near_tie": sum(q["argmax_equal"] for q in near),
            "max_abs_dp": max((q["max_abs_dp"] for q in qs), default=None),
            "mean_question_mean_abs_dp": float(np.mean([q["mean_abs_dp"] for q in qs])) if qs else None,
            "max_abs_dlogit_T": max((q["max_abs_dlogit_T"] for q in qs), default=None),
            "worst": (lambda w: {"id": w["id"], "q": w["q"], "max_abs_dp": w["max_abs_dp"]})(
                max(qs, key=lambda q: q["max_abs_dp"])) if qs else None,
            "tokens": int(sum(q["T"] for q in qs)), "padded_tokens": int(sum(q["padded"] for q in qs)),
            "seconds": float(sum(q["seconds"] for q in qs))}


def merge(args) -> None:
    doc, recs = load_oracle()
    expected = all_rows(recs)
    shard_files = sorted(PARITY.glob("p1_shard*of*.jsonl"))
    parts = [x for p in shard_files for x in read_jsonl(p)]
    metas = [x for x in parts if x["kind"] == "meta"]
    p1_by = {(x["id"], x["q"]): x for x in parts if x["kind"] == "p1"}
    p1 = [p1_by[k] for k in expected if k in p1_by]
    missing = [f"{r}:q{k}" for r, k in expected if (r, k) not in p1_by]
    s1 = summarize(p1)
    hid = [x for x in p1 if "hidden" in x]
    s1["hidden_records"] = {
        "rows": len(hid), "records": sorted({x["id"] for x in hid}),
        "min_pos_cos": min((x["hidden"]["min_pos_cos"] for x in hid), default=None),
        "max_abs_diff": max((x["hidden"]["max_abs_diff"] for x in hid), default=None),
        "max_rel_diff": max((x["hidden"]["rel_max_abs_diff"] for x in hid), default=None),
        "positions_below_0.9999": int(sum(x["hidden"]["positions_below_0.9999"] for x in hid)),
        "per_row": [{"id": x["id"], "q": x["q"], "T": x["T"], **{k: x["hidden"][k] for k in
                    ("min_pos_cos", "min_pos_cos_index", "max_abs_diff", "ref_absmax")}} for x in hid]}
    floors = [x["head_floor"] for x in hid if x.get("head_floor")]
    s1["head_floor"] = {"rows": len(floors), "bit_equal": sum(f["bit_equal"] for f in floors),
                        "max_abs_dlogit_T": max((f["max_abs_dlogit_T"] for f in floors), default=None)}
    by_src: dict = {}
    for x in p1:
        by_src.setdefault(x["source"], []).append(x)
    p1_pass = (not missing and s1["argmax_equal"] == s1["questions"] and s1["max_abs_dp"] <= BAR["max_abs_dp"]
               and len(hid) > 0 and s1["hidden_records"]["min_pos_cos"] >= BAR["min_pos_cos"]
               and s1["hidden_records"]["records"] == sorted(HIDDEN_RECORDS))
    conv_total = {k: 0 for k in ("fast_calls", "checked_vs_F_conv1d", "mismatched")}
    conv_max = 0.0
    for p in shard_files:
        ends = [x for x in read_jsonl(p) if x["kind"] in ("p1", "end")]
        if ends:
            c = ends[-1]["conv"]
            for k in conv_total:
                conv_total[k] += c[k]
            conv_max = max(conv_max, c["max_abs_diff"])
    p2 = json.loads((PARITY / "p2.json").read_text()) if (PARITY / "p2.json").exists() else None
    p2_pass = bool(p2) and len(p2["rows"]) == len(P2_ROWS) and all(
        r["hidden_bit_equal"] and r["chunk_layer_hashes_equal"] for r in p2["rows"])
    p3_parts = read_jsonl(PARITY / "p3.jsonl")
    p3 = {(x["id"], x["q"], x["S"]): x for x in p3_parts if x["kind"] == "p3"}
    p3_rows, p3_table = [], {}
    for rid in HIDDEN_RECORDS:
        for k in range(len(recs[rid]["questions"])):
            base = p3.get((rid, k, CHUNK))
            if base is None:
                continue
            h16 = np.load(hidden_npz(rid, k, CHUNK))["hidden"].astype(np.float64)
            row = {"id": rid, "q": k, "T": base["T"], "widths": {}}
            for S in WIDTHS:
                x = p3.get((rid, k, S))
                if x is None:
                    continue
                w = {"vs_oracle_max_abs_dp": x["max_abs_dp"], "vs_oracle_hidden_max_abs_diff": x["hidden"]["max_abs_diff"],
                     "vs_oracle_min_pos_cos": x["hidden"]["min_pos_cos"], "argmax_equal_oracle": x["argmax_equal"],
                     "chunks": x["chunks"], "seconds": x["seconds"]}
                if S != CHUNK:
                    hs = np.load(hidden_npz(rid, k, S))["hidden"]
                    w["vs_s16_hidden_max_abs_diff"] = float(np.abs(h16 - hs).max())
                    w["vs_s16_max_abs_dp"] = float(np.abs(np.asarray(x["probs"]) - np.asarray(base["probs"])).max())
                else:
                    w["bit_equal_p1_run"] = x.get("bit_equal_p1_run")
                row["widths"][S] = w
            p3_rows.append(row)
    for S in WIDTHS:
        ws = [r["widths"][S] for r in p3_rows if S in r["widths"]]
        if not ws:
            continue
        p3_table[S] = {"rows": len(ws), "argmax_equal_oracle": sum(w["argmax_equal_oracle"] for w in ws),
                       "vs_oracle_max_abs_dp": max(w["vs_oracle_max_abs_dp"] for w in ws),
                       "vs_oracle_min_pos_cos": min(w["vs_oracle_min_pos_cos"] for w in ws),
                       "vs_oracle_hidden_max_abs_diff": max(w["vs_oracle_hidden_max_abs_diff"] for w in ws),
                       "seconds": float(sum(w["seconds"] for w in ws)), "chunks": int(sum(w["chunks"] for w in ws))}
        if S != CHUNK:
            p3_table[S]["vs_s16_hidden_max_abs_diff"] = max(w["vs_s16_hidden_max_abs_diff"] for w in ws)
            p3_table[S]["vs_s16_max_abs_dp"] = max(w["vs_s16_max_abs_dp"] for w in ws)
        else:
            p3_table[S]["bit_equal_p1_run"] = sum(bool(w.get("bit_equal_p1_run")) for w in ws)
    probe = json.loads((PARITY / "probe.json").read_text()) if (PARITY / "probe.json").exists() else None
    here = Path(__file__).resolve()
    mod_path = here.parent / "qwen3_5_kev_decoder.py"
    record = {
        "schema": "kev-decoder-torch-parity/1",
        "purpose": "the hidden-output decoder module (fp32 torch, S-token chunks from fresh zero states, unrolled GDN step "
                   "scan, ids input, one row per question) vs the author's fp32 oracle, read through the author's pointer "
                   "head (head.safetensors, fp32)",
        "module": {"path": "conversion/kev/qwen3_5_kev_decoder.py", "sha256": sha256_file(mod_path),
                   "class": "Qwen3_5KevDecoder", "parent": "coreai_models.models.macos.qwen3_5.Qwen3_5StatefulForCausalLM",
                   "hf_id": HF_ID, "chunk": CHUNK, "load_report": metas[-1]["load_report"] if metas else None},
        "harness": {"path": "conversion/kev/parity_decoder_torch.py", "sha256": sha256_file(here),
                    "depthwise_conv": "bmm (checked against F.conv1d at the last chunk of every row, torch.equal, all "
                                      "channels, every linear-attention layer)" if metas and metas[-1]["conv"] == "bmm"
                                      else "F.conv1d",
                    "conv_check_p1": {**conv_total, "max_abs_diff": conv_max},
                    "invocations": [{k: m[k] for k in ("pid", "started", "shard", "shards", "rows", "threads", "conv",
                                                       "load_seconds")} for m in metas]},
        "head": metas[-1]["head"] if metas else None,
        "oracle": {"path": str(ORACLE / "records_oracle.json"), "sha256": sha256_file(ORACLE / "records_oracle.json"),
                   "checkpoint": doc["model"]["checkpoint"], "versions": doc["versions"],
                   "fixtures_sha256": doc["fixtures"]["sha256"]},
        "contract": {"input_names": ["input_ids", "position_ids"], "output": f"hidden [1, S, {HIDDEN}] (final norm, every position)",
                     "row": "oracle row_ids (state + branch), positions 0..L-1",
                     "chunks": "ceil(T / S) chunks from zero states, position_ids = 0..cS+S-1, the last chunk padded with "
                               "248044, padded outputs discarded"},
        "bar": BAR,
        "versions": versions(),
        "probe": probe,
        "p1": {"result": "PASS" if p1_pass else "FAIL", "summary": s1, "missing_rows": missing,
               "by_source": {k: summarize(v) for k, v in sorted(by_src.items())},
               "near_ties": [{"id": x["id"], "q": x["q"], "qid": x["qid"], "oracle_top2_margin": x["oracle_top2_margin"],
                              "argmax_equal": x["argmax_equal"], "max_abs_dp": x["max_abs_dp"]} for x in p1 if x["near_tie"]],
               "questions": [{k: v for k, v in x.items() if k not in ("conv", "probs")} for x in p1]},
        "p2": {"result": "PASS" if p2_pass else ("FAIL" if p2 else "NOT RUN"), **(p2 or {})},
        "p3": {"widths": list(WIDTHS), "table": p3_table, "rows": p3_rows,
               "note": "fp32 torch chunk-width sensitivity (recorded, not a bar); S = 16 re-run in the P3 process and "
                       "compared bit for bit with P1's run"},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"P1 {record['p1']['result']}: questions {s1['questions']}/{len(expected)} argmax {s1['argmax_equal']} "
          f"(near-tie {s1['argmax_equal_near_tie']}/{s1['near_tie_questions']}) max|dp| {s1['max_abs_dp']:.3e} "
          f"max|dlogit_T| {s1['max_abs_dlogit_T']:.3e} hidden min cos {s1['hidden_records']['min_pos_cos']} "
          f"max|d| {s1['hidden_records']['max_abs_diff']} head floor {json.dumps(s1['head_floor'])} conv {json.dumps(conv_total)}")
    for k, v in sorted(by_src.items()):
        s = summarize(v)
        print(f"  {k:26s} q {s['questions']:3d} argmax {s['argmax_equal']}/{s['questions']} max|dp| {s['max_abs_dp']:.2e} "
              f"mean {s['mean_question_mean_abs_dp']:.2e}")
    print(f"P2 {record['p2']['result']}")
    for S, t in p3_table.items():
        print(f"P3 S={S}: {json.dumps(t)}")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["probe", "p1", "p2", "p3", "scan", "merge"])
    ap.add_argument("--gdn-scan", default="chunk", choices=["chunk"],
                    help="scan only (round 11): the GDN scan form run beside the unroll control")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--conv", default="bmm", choices=["bmm", "f_conv1d"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--model", default="kev-0.8b", choices=sorted(MODELS))
    ap.add_argument("--p3-rows", help="p3 only: comma list of id:q (default: every row of the six hidden records)")
    ap.add_argument("--p3-widths", type=lambda s: [int(x) for x in s.split(",")],
                    help="p3 only: comma list of chunk widths (default 16,32,64,128; 16 must be in it)")
    ap.add_argument("--out", help="merge's record (default <work>/_kev/results/parity_decoder_torch[_4b].json)")
    args = ap.parse_args()
    configure(args.model)
    if args.stage == "scan":
        args.out = args.out or str(LANE / "results" / f"r11_torch_{args.gdn_scan}{MODELS[args.model]['suffix']}.json")
        if Path(args.out).exists():
            raise SystemExit(f"{args.out} exists: records are never overwritten")
    args.out = args.out or str(LANE / "results" / f"parity_decoder_torch{MODELS[args.model]['suffix']}.json")
    {"probe": stage_probe, "p1": stage_p1, "p2": stage_p2, "p3": stage_p3, "scan": stage_scan,
     "merge": merge}[args.stage](args)


if __name__ == "__main__":
    main()
