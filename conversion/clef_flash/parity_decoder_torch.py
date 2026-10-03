#!/usr/bin/env python3
"""fp32 torch parity: the clef-flash decoder module vs the author's fp32 oracle, read through the author's head.

`qwen3_5_clef_decoder.Qwen3_5ClefDecoder` (fp32, every GDN layer on `use_loopfree_unroll`) is driven
the way its Core AI graph will run: fresh zero states per run, the prompt in static S-token chunks
(position_ids = the cache ramp 0..cursor+S-1, the last chunk padded with <|endoftext|> 248044 and
the padded positions' outputs discarded), ids in (image tokens rewritten to V + k by
`host_static_inputs`, i0 = 36), the image rows = the oracle's own HF fp32 tower output
(`oracle/npz/<id>__<arm>.npz image_embeds`, zero-padded to 1024 rows) — so this isolates the
decoder. The hidden state at every position goes through the checkpoint's own `JointSchemaHead`
(path import of the snapshot's joint_schema_model.py, sha256 pinned; joint_head.safetensors and
the untied lm_head.weight both read in fp32) and the per-question softmax is compared with the
oracle (`oracle/records_oracle.json` + npz, read only):

  P1  every oracle run (214): module hidden vs the oracle `last_hidden` (per-position cos,
      max |d|, max |d| / max |ref|), the in-graph rope planes vs the planes the oracle's rotary
      received, and per question argmax / max |dp| / mean |dp| vs the oracle `probs`.
      Bar (fixed before running): argmax equal on every question (near-ties included),
      max |dp| <= 1e-4, min position cos >= 0.9999. Also recorded per run: the author's head
      re-run on the ORACLE hidden in this venv vs the oracle logits (the head's own floor).
  P2  text runs own_t01 / own_j01 / the first SemIf record: this module (image inputs zero,
      start 1 << 30, amount 0) vs the overlay's plain `Qwen3_5StatefulForCausalLM`
      (`from_hf_memory_efficient(..., hf_config_attr="text_config")`, loaded in a separate
      invocation after this module is freed, same unroll, same chunks, CPU both):
      `model.forward_stateful` hidden bit-equal at every position; else the first chunk and
      layer that differ (per chunk x layer sha256 of the layer outputs).
  P3  own_t01/text, img_01/g256: the layer-0 input and every decoder layer's output (all
      positions) vs the oracle `hidden_states` [33, T, 4096], and the final norm vs `last_hidden`.
  P4  img_07/g448, img_08/g448, img_01/g256, red arms vs the unperturbed run: image_embeds zero /
      image_rc row-col swapped / 1-D positions (all three planes = the cache index, no shift) /
      rope_shift_amount + 1 — argmax changes and max |dp|.
  P5  own_t01, own_t14, img_04/g448, photo_02/native, the first two SemIf records: S = 16 vs 32
      vs 64 (hidden max |d| and p max |d| between widths; recorded, not a bar).
  device  own_t01, own_j01, img_01/g256, img_04/g448, the first SemIf record on CPU and on MPS:
      max |dp| <= 1e-5 and hidden max |d| <= 1e-2 before MPS carries the rest.

CPU fp32 is the reference. Harness-only substitution (the module is untouched): this torch build
has no oneDNN, so on the CPU the GDN depthwise conv (`F.conv1d`, groups = 8192) runs as 8192
per-channel convolutions; it is evaluated as one `torch.bmm` over the same windows instead, and at
the last chunk of a CPU run all 24 layers' results are checked against `F.conv1d` itself
(`torch.equal`, all channels): every run by default, every Nth with `--conv-check-every N` plus every
run that keeps its npz (the check costs ~8 s a run on this build; each record says whether its run
was checked). MPS runs use `F.conv1d`, and call the module inside
`with torch.device("mps")`: the parent's forward builds its 0/1 M-RoPE frequency masks with
`torch.tensor` (default device), which on the CPU default meets the MPS `inv_freq` (the exported
graph bakes the same 0/1 constants; nothing numeric changes).

One run exceeds the shipped 1024-row image buffer: photo_01/native (N = 40 x 30 = 1200). The
module is shape-generic (`n_image_max` only sizes the static buffer and the slot clamp), so that
run uses a 1200-row buffer and says so (`n_image_max_used`); the shipped path sends a fixed grid.

    cd <worktree>/conversion/clef_flash
    PY=~/code/coreai/coreai-models/.venv/bin/python
    $PY parity_decoder_torch.py --stages device,p1,p4,p5 --device mps   # one process, one module load
    $PY parity_decoder_torch.py --stages p2                               # plain overlay model, CPU
    $PY parity_decoder_torch.py --merge                                   # -> results/parity_decoder.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

# No .pyc next to the author's path-imported file in the snapshot, nor in the shared conversion/ dirs.
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import hf_snapshot, work_path  # noqa: E402

LANE = work_path("_clefflash")
os.environ.setdefault("HF_HOME", str(LANE / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

HF_ID = "Cloudflare/clef-flash"
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
JSM_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
ORACLE = LANE / "oracle"
VOCAB = 248320
VISION_START, IMAGE_PAD, PAD_ID = 248053, 248056, 248044
HIDDEN = 4096
MERGE = 2
N_IMAGE_MAX = 1024
CHUNK = 16
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
BAR = {"argmax_equal": "every question", "max_abs_dp": 1e-4, "min_pos_cos": 0.9999}
DEVICE_BAR = {"max_abs_dp": 1e-5, "hidden_max_abs": 1e-2}
P3_RUNS = [("own_t01", "text"), ("img_01", "g256")]
P4_RUNS = [("img_07", "g448"), ("img_08", "g448"), ("img_01", "g256")]
P4_ARMS = ("image_embeds_zero", "image_rc_row_col_swapped", "pos1d_index_all_planes", "rope_shift_amount_plus_1")
P5_WIDTHS = (16, 32, 64)
PARTS = LANE / "parity" / "parts.jsonl"
NPZ_DIR = LANE / "parity"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_oracle() -> tuple[dict, list[dict], dict]:
    doc = json.loads((ORACLE / "records_oracle.json").read_text())
    assert doc["complete"] and doc["source"]["revision"] == REVISION, "oracle incomplete or another revision"
    rows = doc["rows"]
    by_key = {(r["id"], r["arm"]): r for r in rows}
    semif = [r["id"] for r in rows if r["source"] == "semif_authored144"]
    named = {
        "semif_first": (semif[0], "text"), "semif_second": (semif[1], "text"),
    }
    return doc, rows, {"by_key": by_key, **named}


def run_sets(o: dict) -> dict:
    s1, s2 = o["semif_first"], o["semif_second"]
    return {
        "device": [("own_t01", "text"), ("own_j01", "text"), ("img_01", "g256"), ("img_04", "g448"), s1],
        "p2": [("own_t01", "text"), ("own_j01", "text"), s1],
        "p5": [("own_t01", "text"), ("own_t14", "text"), ("img_04", "g448"), ("photo_02", "native"), s1, s2],
    }


def append_part(rec: dict) -> None:
    PARTS.parent.mkdir(parents=True, exist_ok=True)
    with open(PARTS, "a") as f:
        f.write(json.dumps(rec) + "\n")


def read_parts() -> list[dict]:
    return [json.loads(x) for x in PARTS.read_text().splitlines()] if PARTS.exists() else []


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def compare(mine: np.ndarray, ref: np.ndarray) -> dict:
    d = np.abs(mine.astype(np.float64) - ref.astype(np.float64))
    c = cos_rows(mine, ref)
    i = int(c.argmin())
    ref_max = float(np.abs(ref).max())
    return {"max_abs_diff": float(d.max()), "ref_absmax": ref_max, "rel_max_abs_diff": float(d.max()) / ref_max,
            "min_pos_cos": float(c[i]), "min_pos_cos_index": i, "mean_pos_cos": float(c.mean())}


class DepthwiseConvBmm:
    """F.conv1d replacement for the GDN's valid depthwise conv on CPU tensors (batch 1, no bias,
    stride 1), evaluated as one bmm; every other call goes to the original. `check` compares
    against the original on the next calls."""

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


class AuthorHead:
    """The checkpoint's JointSchemaHead in fp32 on the CPU, with the untied lm_head table (fp32)."""

    def __init__(self):
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file

        self.torch = torch
        snap = Path(hf_snapshot(HF_ID, revision=REVISION))
        path = snap / "joint_schema_model.py"
        assert sha256_file(path) == JSM_SHA256, "joint_schema_model.py is not the pinned file"
        spec = importlib.util.spec_from_file_location("joint_schema_model", path)
        jsm = importlib.util.module_from_spec(spec)
        sys.modules["joint_schema_model"] = jsm
        spec.loader.exec_module(jsm)                                        # author's code, unchanged
        self.jsm = jsm
        head = jsm.JointSchemaHead(**json.loads((snap / "joint_head_config.json").read_text()))
        head.load_state_dict(load_file(snap / "joint_head.safetensors"), strict=True)
        self.head = head.to(dtype=torch.float32).eval()
        index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
        with safe_open(snap / index["lm_head.weight"], framework="pt", device="cpu") as f:
            self.lm_head_w = f.get_tensor("lm_head.weight").to(torch.float32)
        self.provenance = {"joint_schema_model_sha256": JSM_SHA256,
                           "joint_head_safetensors_sha256": sha256_file(snap / "joint_head.safetensors"),
                           "lm_head_file": index["lm_head.weight"], "lm_head_shape": list(self.lm_head_w.shape),
                           "dtype": "float32 (bf16 checkpoint values cast up)", "device": "cpu"}

    def logits(self, hidden, ids: list[int], row: dict):
        """hidden [T, 4096] fp32 CPU, ids = the processor-form ids -> per-question logits (fp32)."""
        torch, jsm = self.torch, self.jsm
        qs = tuple(jsm.EncodedQuestion(question_id=q["question_id"], question_type=QUESTION_TYPES[q["type"]],
                                       question_span=tuple(q["question_span"]),
                                       option_spans=tuple(tuple(s) for s in q["option_spans"]),
                                       option_ids=tuple(q["option_ids"])) for q in row["questions"])
        enc = jsm.EncodedRecord(input_ids=tuple(ids), questions=qs, record_id=row["id"])
        ids_t = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            out = self.head(hidden.unsqueeze(0), ids_t, torch.ones_like(ids_t), [enc], self.lm_head_w)[0]
        return [t.float() for t in out]


class Runner:
    """The module, the chunk loop and the taps."""

    def __init__(self, threads: int):
        import torch
        import torch.nn.functional as F

        torch.set_num_threads(threads)
        self.torch = torch
        from qwen3_5_clef_decoder import Qwen3_5ClefDecoder, host_static_inputs

        from coreai_models.models.macos import qwen3_5 as q

        self.q = q
        self.host_static_inputs = host_static_inputs
        self.cls = Qwen3_5ClefDecoder
        t0 = time.monotonic()
        self.model = Qwen3_5ClefDecoder.from_hf(HF_ID, torch.float32, max_context_length=4096,
                                                n_image_max=N_IMAGE_MAX)
        self.load_seconds = time.monotonic() - t0
        self.cfg = self.model.config
        assert self.cfg.vocab_size == VOCAB and self.cfg.hidden_size == HIDDEN
        self.unrolled = 0
        for layer in self.model.model.layers:
            if not layer.is_full:
                layer.linear_attn.use_loopfree_step = True
                layer.linear_attn.use_loopfree_unroll = True
                self.unrolled += 1
        self.conv = DepthwiseConvBmm(F.conv1d)
        F.conv1d = self.conv
        self.planes_override = None
        self._planes_log = None
        self.model._rope_planes = self._rope_planes
        self.device = torch.device("cpu")

    def to(self, device: str) -> float:
        t0 = time.monotonic()
        self.model.to(device)
        self.device = self.torch.device(device)
        return time.monotonic() - t0

    def _rope_planes(self, is_img, slot, p, image_rc, start, amount):
        planes = self.cls._rope_planes(self.model, is_img, slot, p, image_rc, start, amount)
        if self.planes_override is not None:
            planes = self.planes_override(is_img, slot, p, image_rc, start, amount, planes)
        if self._planes_log is not None:
            self._planes_log.append(self.torch.stack([x.reshape(-1) for x in planes]).cpu())
        return planes

    def static_inputs(self, row: dict, npz) -> dict:
        torch = self.torch
        mh = row.get("merged_hw")
        hw = tuple(mh) if mh else None
        n = hw[0] * hw[1] if hw else 0
        nmax = max(N_IMAGE_MAX, n)
        ids, rc, start, amount = self.host_static_inputs(row["ids"], hw, VOCAB, IMAGE_PAD, VISION_START, nmax)
        emb = torch.zeros(nmax, HIDDEN, dtype=torch.float32)
        if hw:
            e = torch.from_numpy(np.asarray(npz["image_embeds"], dtype=np.float32))
            assert e.shape == (n, HIDDEN), (e.shape, hw)
            emb[:n] = e
            assert int(start) == row["token_offset"] + 1 + n and int(amount) == row["rope_shift_amount"]
        return {"ids": ids, "emb": emb, "rc": rc, "start": start, "amount": amount, "nmax": nmax, "hw": hw}

    def run(self, si: dict, S: int = CHUNK, *, taps: bool = False, hashes: bool = False,
            emb=None, rc=None, start=None, amount=None, conv_check: bool = True) -> dict:
        """One prompt from fresh zero states in S-token chunks -> hidden [T, 4096] fp32 CPU + extras."""
        torch, dev = self.torch, self.device
        ids = si["ids"]
        T = int(ids.shape[0])
        n_chunks = -(-T // S)
        Tp = n_chunks * S
        ids_p = torch.full((Tp,), PAD_ID, dtype=torch.int32)
        ids_p[:T] = ids
        emb = (si["emb"] if emb is None else emb).to(dev)
        rc = (si["rc"] if rc is None else rc).to(dev)
        start = (si["start"] if start is None else start).to(dev)
        amount = (si["amount"] if amount is None else amount).to(dev)
        st = {k: v.to(dev) for k, v in self.q.build_decode_state(self.cfg, max_seq_len=Tp,
                                                                  dtype=torch.float32).items()}
        self.model.n_image_max = si["nmax"]
        layers = self.model.model.layers
        cur: list = []
        tap_layers = [[] for _ in layers] if taps else None
        tap_embed: list = []
        layer_hashes: list = []
        hooks = []
        if taps or hashes:
            for li, layer in enumerate(layers):
                hooks.append(layer.register_forward_hook(
                    lambda mod, args, out, li=li: cur.append((li, out[0].detach().to("cpu", copy=True)))))
        if taps:
            hooks.append(layers[0].register_forward_pre_hook(
                lambda mod, args: tap_embed.append(args[0][0].detach().to("cpu", copy=True))))
        self._planes_log = []
        outs = []
        t0 = time.monotonic()
        try:
            with torch.inference_mode():
                for c in range(n_chunks):
                    self.conv.check = conv_check and c == n_chunks - 1
                    cur.clear()
                    tok = ids_p[c * S:(c + 1) * S].reshape(1, S).to(dev)
                    pos = torch.arange((c + 1) * S, dtype=torch.int32).unsqueeze(0).to(dev)
                    with torch.device(dev):   # the parent builds its 0/1 M-RoPE masks with torch.tensor
                        h = self.model(tok, pos, emb, rc, start, amount, st["k_cache"], st["v_cache"],
                                       st["conv_state"], st["rec_state"])
                    outs.append(h[0])
                    if taps:
                        for li, x in cur:
                            tap_layers[li].append(x)
                    if hashes:
                        layer_hashes.append([hashlib.sha256(x.numpy().tobytes()).hexdigest()[:16] for _, x in cur]
                                            + [hashlib.sha256(h[0].cpu().numpy().tobytes()).hexdigest()[:16]])
                hidden = torch.cat(outs)[:T].to("cpu").numpy().astype(np.float32)
        finally:
            for hk in hooks:
                hk.remove()
            self.conv.check = False
            self.model.n_image_max = N_IMAGE_MAX
        secs = time.monotonic() - t0
        planes = torch.cat(self._planes_log, dim=1)[:, :T].numpy().astype(np.int64)
        self._planes_log = None
        out = {"hidden": hidden, "planes": planes, "T": T, "Tp": Tp, "chunks": n_chunks, "S": S, "seconds": secs}
        if taps:
            out["layers"] = [torch.cat(x)[:T].numpy() for x in tap_layers]
            out["embed"] = torch.cat(tap_embed)[:T].numpy()
        if hashes:
            out["layer_hashes"] = layer_hashes
        return out


def probs_of(logits) -> list[np.ndarray]:
    return [t.float().softmax(-1).double().numpy() for t in logits]


def score_run(head: AuthorHead, row: dict, npz, res: dict) -> dict:
    """P1 metrics of one module run vs the oracle."""
    import torch
    ids = [int(x) for x in npz["input_ids"]]
    assert ids == row["ids"]
    hid = compare(res["hidden"], npz["last_hidden"])
    logits = head.logits(torch.from_numpy(res["hidden"]), ids, row)
    probs = probs_of(logits)
    qrecs = []
    for q, lg, p in zip(row["questions"], logits, probs):
        po = np.asarray(q["probs"], np.float64)
        lo = np.asarray(q["logits"], np.float64)
        dp = np.abs(p - po)
        qrecs.append({"question_id": q["question_id"], "type": q["type"], "n_options": len(po),
                      "argmax": int(p.argmax()), "argmax_oracle": q["argmax_index"],
                      "argmax_equal": int(p.argmax()) == q["argmax_index"],
                      "max_abs_dp": float(dp.max()), "mean_abs_dp": float(dp.mean()),
                      "max_abs_dlogit": float(np.abs(lg.double().numpy() - lo).max()),
                      "oracle_top2_margin": q["top2_margin"], "near_tie": q["near_tie"],
                      "probs": [float(v) for v in p]})
    rope_eq = bool(res["planes"].shape == npz["rope_pos"].shape and np.array_equal(res["planes"], npz["rope_pos"]))
    return {"hidden": hid, "rope_planes_equal_oracle": rope_eq, "questions": qrecs,
            "argmax_all_equal": all(x["argmax_equal"] for x in qrecs),
            "max_abs_dp": max(x["max_abs_dp"] for x in qrecs),
            "mean_question_max_abs_dp": float(np.mean([x["max_abs_dp"] for x in qrecs])),
            "mean_abs_dp": float(np.mean(np.concatenate([np.abs(np.asarray(x["probs"]) - np.asarray(q["probs"]))
                                                         for x, q in zip(qrecs, row["questions"])])))}


def head_floor(head: AuthorHead, row: dict, npz) -> dict:
    """The author's head re-run on the ORACLE hidden in this venv vs the oracle logits."""
    import torch
    ids = [int(x) for x in npz["input_ids"]]
    logits = head.logits(torch.from_numpy(np.asarray(npz["last_hidden"], np.float32)), ids, row)
    d = [float(np.abs(lg.double().numpy() - np.asarray(q["logits"], np.float64)).max())
         for lg, q in zip(logits, row["questions"])]
    eq = [lg.numpy().astype(np.float32).tolist() == [float(np.float32(v)) for v in q["logits"]]
          for lg, q in zip(logits, row["questions"])]
    return {"max_abs_dlogit": max(d), "bit_equal": all(eq)}


def p3_record(row: dict, npz, res: dict) -> dict:
    hs = npz["hidden_states"]                                        # [33, T, 4096]
    stages = []

    def cmp(name, a, b):
        c = compare(a, b)
        stages.append({"stage": name, **{k: c[k] for k in ("max_abs_diff", "ref_absmax", "min_pos_cos", "mean_pos_cos")}})

    cmp("layer-0 input (hs[0])", res["embed"], hs[0])
    for li, x in enumerate(res["layers"]):
        cmp(f"layer {li} output (hs[{li + 1}])", x, hs[li + 1])
    cmp("final norm (last_hidden)", res["hidden"], npz["last_hidden"])
    worst_d = max(stages, key=lambda s: s["max_abs_diff"])
    worst_c = min(stages, key=lambda s: s["min_pos_cos"])
    return {"id": row["id"], "arm": row["arm"], "tokens": row["tokens"], "stages": stages,
            "worst_max_abs_diff": {"value": worst_d["max_abs_diff"], "stage": worst_d["stage"]},
            "worst_min_pos_cos": {"value": worst_c["min_pos_cos"], "stage": worst_c["stage"]}}


def save_npz(key, device: str, S: int, res: dict, extra: dict | None = None) -> str:
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    path = NPZ_DIR / f"{key[0]}__{key[1]}__{device}_s{S}.npz"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, hidden=res["hidden"], planes=res["planes"], **(extra or {}))
    os.replace(tmp, path)
    return str(path)


def stage_runs(args) -> None:
    import torch

    doc, rows, o = load_oracle()
    by_key = o["by_key"]
    sets = run_sets(o)
    stages = args.stages.split(",")
    done = {(p["kind"], p["id"], p["arm"], p["device"], p["S"], p.get("perturb")) for p in read_parts()}
    print(f"loading module + head (threads {args.threads}) ...", flush=True)
    t0 = time.monotonic()
    head = AuthorHead()
    runner = Runner(args.threads)
    print(f"module loaded in {runner.load_seconds:.0f}s (head + module {time.monotonic() - t0:.0f}s); "
          f"unrolled GDN layers {runner.unrolled}; load_report {json.dumps(runner.model.load_report)}", flush=True)
    meta = {"kind": "meta", "id": None, "arm": None, "device": None, "S": None, "pid": os.getpid(),
            "started": datetime.now().astimezone().isoformat(timespec="seconds"), "stages": stages,
            "threads": args.threads, "load_seconds": runner.load_seconds, "load_report": runner.model.load_report,
            "unrolled_linear_layers": runner.unrolled, "head": head.provenance}
    append_part(meta)
    floors: dict = {}
    cpu_runs = [0]

    def do_run(key, device: str, S: int = CHUNK, kind: str = "p1", perturb: str | None = None,
               taps: bool = False, hashes: bool = False, keep_npz: bool = False):
        tag = (kind, key[0], key[1], device, S, perturb)
        if tag in done:
            return None
        if runner.device.type != device:
            secs = runner.to(device)
            print(f"moved module to {device} in {secs:.0f}s", flush=True)
        row = by_key[key]
        npz = np.load(ORACLE / "npz" / f"{key[0]}__{key[1]}.npz")
        si = runner.static_inputs(row, npz)
        kw = {}
        if perturb == "image_embeds_zero":
            kw["emb"] = torch.zeros_like(si["emb"])
        elif perturb == "image_rc_row_col_swapped":
            kw["rc"] = si["rc"][:, [1, 0]].contiguous()
        elif perturb == "pos1d_index_all_planes":
            runner.planes_override = lambda is_img, slot, p, rc_, s_, a_, planes: (p, p, p)
            kw["start"] = torch.tensor([1 << 30], dtype=torch.int32)
            kw["amount"] = torch.tensor([0], dtype=torch.int32)
        elif perturb == "rope_shift_amount_plus_1":
            kw["amount"] = si["amount"] + 1
        check = device == "cpu" and (cpu_runs[0] % args.conv_check_every == 0 or keep_npz or taps or hashes)
        cpu_runs[0] += device == "cpu"
        try:
            res = runner.run(si, S, taps=taps, hashes=hashes, conv_check=check, **kw)
        finally:
            runner.planes_override = None
        sc = score_run(head, row, npz, res)
        if key not in floors:
            floors[key] = head_floor(head, row, npz)
        rec = {"kind": kind, "id": key[0], "arm": key[1], "device": device, "S": S, "perturb": perturb,
               "source": row["source"], "tokens": res["T"], "padded_tokens": res["Tp"], "chunks": res["chunks"],
               "n_image_tokens": row["n_image_tokens"], "merged_hw": row.get("merged_hw"),
               "n_image_max_used": si["nmax"], "seconds": res["seconds"],
               "host": {"start": int(si["start"][0]), "amount": int(si["amount"][0]),
                        "image_tokens": int((si["ids"] >= VOCAB).sum())},
               "head_floor": floors[key], **sc, "conv": runner.conv.report(), "conv_checked_this_run": check}
        if taps:
            rec["p3"] = p3_record(row, npz, res)
        if hashes:
            rec["layer_hashes"] = res["layer_hashes"]
        bar_ok = sc["argmax_all_equal"] and sc["max_abs_dp"] <= BAR["max_abs_dp"] and \
            sc["hidden"]["min_pos_cos"] >= BAR["min_pos_cos"]
        if keep_npz or (kind == "p1" and not bar_ok):
            rec["npz"] = save_npz(key, device, S, res, {"probs": np.concatenate([np.asarray(x["probs"]) for x in sc["questions"]])})
        append_part(rec)
        done.add(tag)
        print(f"[{kind}{'/' + perturb if perturb else ''}] {key[0]}:{key[1]} {device} S={S} T={res['T']} "
              f"argmax {'ok' if sc['argmax_all_equal'] else 'NO'} max|dp| {sc['max_abs_dp']:.2e} "
              f"min cos {sc['hidden']['min_pos_cos']:.7f} max|d| {sc['hidden']['max_abs_diff']:.2e} "
              f"rope {'ok' if sc['rope_planes_equal_oracle'] else 'NO'} {res['seconds']:.1f}s", flush=True)
        return rec

    main_dev = args.device
    if "device" in stages:
        for key in sets["device"]:
            do_run(key, "cpu", kind="p1", taps=key in P3_RUNS, hashes=key in sets["p2"], keep_npz=True)
        if torch.backends.mps.is_available():
            for key in sets["device"]:
                do_run(key, "mps", kind="p1", keep_npz=True)
        dc = device_check(read_parts(), sets["device"])
        (LANE / "results").mkdir(exist_ok=True)
        (LANE / "results" / "parity_device_check.json").write_text(json.dumps(dc, indent=1) + "\n")
        print(f"device check: {dc['result']} (max|dp| {dc['max_abs_dp']:.2e}, hidden max|d| "
              f"{dc['hidden_max_abs_diff']:.2e})", flush=True)
        if main_dev == "mps" and dc["result"] != "PASS":
            print("MPS does not match the CPU within the device bar: main stages fall back to the CPU", flush=True)
            main_dev = "cpu"
    if "p1" in stages:
        keys = [(r["id"], r["arm"]) for r in rows]
        if args.runs:
            keys = [tuple(k.split(":")) for k in args.runs.split(",")]
        for key in keys:
            do_run(key, main_dev, kind="p1", taps=(main_dev == "cpu" and key in P3_RUNS),
                   hashes=(main_dev == "cpu" and key in sets["p2"]),
                   keep_npz=key in P3_RUNS or key in sets["p5"])
    if "p4" in stages:
        for key in P4_RUNS:
            do_run(key, main_dev, kind="p1")
            for arm in P4_ARMS:
                do_run(key, main_dev, kind="p4", perturb=arm)
    if "p5" in stages:
        for key in sets["p5"]:
            for S in P5_WIDTHS:
                do_run(key, main_dev, S=S, kind="p1" if S == CHUNK else "p5", keep_npz=True)
    append_part({"kind": "end", "id": None, "arm": None, "device": None, "S": None, "pid": os.getpid(),
                 "finished": datetime.now().astimezone().isoformat(timespec="seconds"), "conv": runner.conv.report()})
    print(f"done; conv {json.dumps(runner.conv.report())}", flush=True)


def device_check(parts: list[dict], keys) -> dict:
    by = {(p["id"], p["arm"], p["device"]): p for p in parts if p["kind"] == "p1" and p["S"] == CHUNK}
    runs = []
    for key in keys:
        c, m = by.get((key[0], key[1], "cpu")), by.get((key[0], key[1], "mps"))
        if not (c and m):
            runs.append({"id": key[0], "arm": key[1], "missing": [d for d, x in (("cpu", c), ("mps", m)) if not x]})
            continue
        hc = np.load(c["npz"])["hidden"]
        hm = np.load(m["npz"])["hidden"]
        dh = float(np.abs(hc.astype(np.float64) - hm).max())
        cs = cos_rows(hc, hm)
        dps = [float(np.abs(np.asarray(a["probs"]) - np.asarray(b["probs"])).max())
               for a, b in zip(c["questions"], m["questions"])]
        runs.append({"id": key[0], "arm": key[1], "tokens": c["tokens"], "hidden_max_abs_diff": dh,
                     "hidden_min_pos_cos": float(cs.min()), "max_abs_dp": max(dps),
                     "argmax_equal": all(a["argmax"] == b["argmax"] for a, b in zip(c["questions"], m["questions"])),
                     "cpu_vs_oracle": {"max_abs_dp": c["max_abs_dp"], "min_pos_cos": c["hidden"]["min_pos_cos"]},
                     "mps_vs_oracle": {"max_abs_dp": m["max_abs_dp"], "min_pos_cos": m["hidden"]["min_pos_cos"]},
                     "seconds": {"cpu": c["seconds"], "mps": m["seconds"]}, "chunks": c["chunks"]})
    ok = [r for r in runs if "missing" not in r]
    res = {"schema": "clef-flash-decoder-device-check/1",
           "what": "the module in fp32, S=16 chunks, fresh zero states: CPU (bmm depthwise conv, checked against "
                   "F.conv1d) vs MPS (F.conv1d), hidden at every position and the author's head on each",
           "bar": DEVICE_BAR, "runs": runs,
           "max_abs_dp": max((r["max_abs_dp"] for r in ok), default=None),
           "hidden_max_abs_diff": max((r["hidden_max_abs_diff"] for r in ok), default=None),
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    res["result"] = "PASS" if (len(ok) == len(keys) and res["max_abs_dp"] <= DEVICE_BAR["max_abs_dp"]
                               and res["hidden_max_abs_diff"] <= DEVICE_BAR["hidden_max_abs"]
                               and all(r["argmax_equal"] for r in ok)) else "FAIL"
    return res


def stage_p2(args) -> None:
    """The overlay's plain text decoder, loaded separately, vs this module's CPU runs (bit equality)."""
    import torch
    import torch.nn.functional as F

    from coreai_models.models.macos import qwen3_5 as q
    from coreai_models.primitives.macos.cache import KVCache, SSMState

    torch.set_num_threads(args.threads)
    doc, rows, o = load_oracle()
    by_key = o["by_key"]
    parts = read_parts()
    mod = {(p["id"], p["arm"]): p for p in parts if p["kind"] == "p1" and p["device"] == "cpu" and p["S"] == CHUNK
           and p.get("layer_hashes")}
    conv = DepthwiseConvBmm(F.conv1d)
    F.conv1d = conv
    t0 = time.monotonic()
    plain = q.Qwen3_5StatefulForCausalLM.from_hf_memory_efficient(
        HF_ID, max_context_length=4096, target_dtype=torch.float32, hf_config_attr="text_config")
    plain.eval()
    load_s = time.monotonic() - t0
    n = 0
    for layer in plain.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True
            layer.linear_attn.use_loopfree_unroll = True
            n += 1
    print(f"plain overlay model loaded in {load_s:.0f}s, unrolled {n}", flush=True)
    out = []
    for key in run_sets(o)["p2"]:
        m = mod.get(key)
        if m is None:
            out.append({"id": key[0], "arm": key[1], "error": "no module CPU run with layer hashes"})
            continue
        row = by_key[key]
        ids = torch.tensor(row["ids"], dtype=torch.int32)
        T = int(ids.shape[0])
        S = CHUNK
        n_chunks = -(-T // S)
        Tp = n_chunks * S
        ids_p = torch.full((Tp,), PAD_ID, dtype=torch.int32)
        ids_p[:T] = ids
        st = q.build_decode_state(plain.config, max_seq_len=Tp, dtype=torch.float32)
        cur: list = []
        hooks = [layer.register_forward_hook(lambda mod_, a, out_: cur.append(out_[0].detach().clone()))
                 for layer in plain.model.layers]
        hashes, outs = [], []
        t1 = time.monotonic()
        try:
            with torch.inference_mode():
                for c in range(n_chunks):
                    conv.check = c == n_chunks - 1
                    cur.clear()
                    tok = ids_p[c * S:(c + 1) * S].reshape(1, S)
                    pos = torch.arange((c + 1) * S, dtype=torch.int32).unsqueeze(0)
                    h = plain.model.forward_stateful(tok, pos, KVCache(st["k_cache"], st["v_cache"]),
                                                     SSMState(st["conv_state"]), SSMState(st["rec_state"]))
                    outs.append(h[0])
                    hashes.append([hashlib.sha256(x.numpy().tobytes()).hexdigest()[:16] for x in cur]
                                  + [hashlib.sha256(h[0].numpy().tobytes()).hexdigest()[:16]])
        finally:
            for hk in hooks:
                hk.remove()
            conv.check = False
        hidden = torch.cat(outs)[:T].numpy()
        mine = np.load(m["npz"])["hidden"]
        eq = bool(np.array_equal(hidden, mine))
        first = None
        for c, (a, b) in enumerate(zip(m["layer_hashes"], hashes)):
            if a != b:
                li = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
                first = {"chunk": c, "layer": li if li < len(plain.model.layers) else "final norm"}
                break
        rec = {"id": key[0], "arm": key[1], "tokens": T, "chunks": n_chunks, "hidden_bit_equal": eq,
               "max_abs_diff": float(np.abs(hidden.astype(np.float64) - mine).max()),
               "chunk_layer_hashes_equal": first is None and len(hashes) == len(m["layer_hashes"]),
               "first_difference": first, "seconds": time.monotonic() - t1,
               "module_run": {"npz": m["npz"], "device": m["device"]}}
        out.append(rec)
        print(f"P2 {key[0]}:{key[1]} T={T} bit-equal {eq} hashes {rec['chunk_layer_hashes_equal']} "
              f"max|d| {rec['max_abs_diff']:.2e} first {first}", flush=True)
    append_part({"kind": "p2", "id": None, "arm": None, "device": "cpu", "S": CHUNK, "runs": out,
                 "load_seconds": load_s, "unrolled_linear_layers": n, "conv": conv.report(),
                 "plain_class": "coreai_models.models.macos.qwen3_5.Qwen3_5StatefulForCausalLM",
                 "plain_entry": "model.forward_stateful (embed_tokens + forward_stateful_core, in-graph plain RoPE)"})


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


def conv_by_process(parts: list[dict]) -> dict:
    """The bmm-vs-F.conv1d checks per process (a stopped process has no end record: its last run's
    cumulative report counts) and in total, P2's plain-model process included."""
    segs, seg = [], None
    for p in parts:
        if p["kind"] == "meta":
            seg = {"pid": p["pid"], "stages": p["stages"], "cpu_runs": 0, "checked_runs": 0, "conv": None}
            segs.append(seg)
        elif p["kind"] in ("p1", "p4", "p5") and seg is not None:
            seg["conv"] = p["conv"]
            if p["device"] == "cpu":
                seg["cpu_runs"] += 1
                seg["checked_runs"] += bool(p.get("conv_checked_this_run", True))
        elif p["kind"] == "end" and seg is not None:
            seg["conv"] = p["conv"]
    p2 = [p for p in parts if p["kind"] == "p2"]
    if p2:
        segs.append({"pid": None, "stages": ["p2"], "cpu_runs": len(p2[-1]["runs"]), "checked_runs": len(p2[-1]["runs"]),
                     "conv": p2[-1]["conv"]})
    total = {k: sum((s["conv"] or {}).get(k, 0) for s in segs) for k in ("fast_calls", "checked_vs_F_conv1d", "mismatched")}
    total["checked_runs"] = sum(s["checked_runs"] for s in segs)
    return {"processes": segs, "total": total}


def merge(args) -> None:
    doc, rows, o = load_oracle()
    parts = read_parts()
    metas = [p for p in parts if p["kind"] == "meta"]
    p1_dev = args.p1_device
    p1_by = {}
    for p in parts:
        if p["kind"] == "p1" and p["S"] == CHUNK and p["perturb"] is None:
            p1_by.setdefault((p["id"], p["arm"]), {})[p["device"]] = p
    p1 = []
    for r in rows:
        d = p1_by.get((r["id"], r["arm"]), {})
        rec = d.get(p1_dev) or d.get("cpu") or d.get("mps")
        if rec:
            p1.append(rec)
    expected = {(r["id"], r["arm"]) for r in rows}
    got = {(p["id"], p["arm"]) for p in p1}
    qs = [q for p in p1 for q in p["questions"]]

    def summarize(ps):
        qq = [q for p in ps for q in p["questions"]]
        return {"runs": len(ps), "questions": len(qq), "argmax_equal": sum(q["argmax_equal"] for q in qq),
                "near_tie_questions": sum(q["near_tie"] for q in qq),
                "max_abs_dp": max((q["max_abs_dp"] for q in qq), default=None),
                "worst_run": max(ps, key=lambda p: p["max_abs_dp"])["id"] if ps else None,
                "mean_run_mean_abs_dp": float(np.mean([p["mean_abs_dp"] for p in ps])) if ps else None,
                "mean_question_max_abs_dp": float(np.mean([q["max_abs_dp"] for q in qq])) if qq else None,
                "max_abs_dlogit": max((q["max_abs_dlogit"] for q in qq), default=None),
                "min_pos_cos": min((p["hidden"]["min_pos_cos"] for p in ps), default=None),
                "hidden_max_abs_diff": max((p["hidden"]["max_abs_diff"] for p in ps), default=None),
                "hidden_max_rel_diff": max((p["hidden"]["rel_max_abs_diff"] for p in ps), default=None),
                "rope_planes_equal_runs": sum(p["rope_planes_equal_oracle"] for p in ps),
                "devices": sorted({p["device"] for p in ps}),
                "seconds_total": float(sum(p["seconds"] for p in ps))}

    by_sa = {}
    for p in p1:
        by_sa.setdefault(f"{p['source']}/{p['arm']}", []).append(p)
    s1 = summarize(p1)
    s1.update(expected_runs=len(expected), missing_runs=sorted(":".join(k) for k in expected - got),
              head_floor_bit_equal_runs=sum(p["head_floor"]["bit_equal"] for p in p1),
              head_floor_max_abs_dlogit=max((p["head_floor"]["max_abs_dlogit"] for p in p1), default=None),
              runs_by_device={d: sum(p["device"] == d for p in p1) for d in ("cpu", "mps")},
              n_image_max_over_1024=[f"{p['id']}:{p['arm']} ({p['n_image_max_used']})" for p in p1
                                     if p["n_image_max_used"] > N_IMAGE_MAX])
    p1_pass = (s1["runs"] == len(expected) and s1["argmax_equal"] == s1["questions"]
               and s1["max_abs_dp"] <= BAR["max_abs_dp"] and s1["min_pos_cos"] >= BAR["min_pos_cos"])
    p2 = next((p for p in reversed(parts) if p["kind"] == "p2"), None)
    p2_pass = bool(p2) and len(p2["runs"]) == 3 and all(r.get("hidden_bit_equal") and r.get("chunk_layer_hashes_equal")
                                                         for r in p2["runs"])
    p3 = [p["p3"] for p in parts if p.get("p3")]
    p4 = []
    for key in P4_RUNS:
        base = p1_by.get(key, {}).get(p1_dev) or p1_by.get(key, {}).get("cpu")
        arms = {}
        for p in parts:
            if p["kind"] == "p4" and (p["id"], p["arm"]) == key and base and p["device"] == base["device"]:
                bq, pq = base["questions"], p["questions"]
                arms[p["perturb"]] = {
                    "argmax_changed": sum(a["argmax"] != b["argmax"] for a, b in zip(pq, bq)),
                    "questions": len(bq),
                    "max_abs_dp_vs_base": max(float(np.abs(np.asarray(a["probs"]) - np.asarray(b["probs"])).max())
                                              for a, b in zip(pq, bq)),
                    "rope_planes_equal_oracle": p["rope_planes_equal_oracle"],
                    "argmax_equal_oracle": sum(a["argmax_equal"] for a in pq)}
        p4.append({"id": key[0], "arm": key[1], "device": base["device"] if base else None, "arms": arms})
    p4_zero_moves = all(r["arms"].get("image_embeds_zero", {}).get("argmax_changed", 0) > 0 for r in p4)
    p5 = []
    for key in run_sets(o)["p5"]:
        recs = {p["S"]: p for p in parts if (p["id"], p["arm"]) == key and p["perturb"] is None
                and p["kind"] in ("p1", "p5") and p["device"] == p1_dev}
        if CHUNK not in recs:
            recs = {p["S"]: p for p in parts if (p["id"], p["arm"]) == key and p["perturb"] is None
                    and p["kind"] in ("p1", "p5") and p["device"] == "cpu"}
        row = {"id": key[0], "arm": key[1], "tokens": recs[CHUNK]["tokens"] if CHUNK in recs else None,
               "device": recs[CHUNK]["device"] if CHUNK in recs else None, "widths": {}}
        for S in P5_WIDTHS:
            if S in recs:
                row["widths"][S] = {"vs_oracle_max_abs_dp": recs[S]["max_abs_dp"],
                                    "vs_oracle_hidden_max_abs_diff": recs[S]["hidden"]["max_abs_diff"],
                                    "seconds": recs[S]["seconds"], "chunks": recs[S]["chunks"]}
        for S in P5_WIDTHS[1:]:
            if S in recs and CHUNK in recs and recs[S].get("npz") and recs[CHUNK].get("npz"):
                a = np.load(recs[CHUNK]["npz"])["hidden"].astype(np.float64)
                b = np.load(recs[S]["npz"])["hidden"]
                row["widths"][S]["vs_s16_hidden_max_abs_diff"] = float(np.abs(a - b).max())
            if S in recs and CHUNK in recs:
                row["widths"][S]["vs_s16_max_abs_dp"] = max(
                    float(np.abs(np.asarray(x["probs"]) - np.asarray(y["probs"])).max())
                    for x, y in zip(recs[S]["questions"], recs[CHUNK]["questions"]))
        p5.append(row)
    here = Path(__file__).resolve()
    mod_path = here.parent / "qwen3_5_clef_decoder.py"
    record = {
        "schema": "clef-flash-decoder-torch-parity/1",
        "purpose": "the hidden-output decoder module (fp32 torch, S-token chunks from fresh zero states, unrolled GDN "
                   "step scan, in-graph M-RoPE, ids input with image rows on a static input) vs the author's fp32 "
                   "oracle, read through the author's JointSchemaHead",
        "module": {"path": "conversion/clef_flash/qwen3_5_clef_decoder.py", "sha256": sha256_file(mod_path),
                   "class": "Qwen3_5ClefDecoder", "parent": "conversion/decider_vision/qwen3_5_vl_pipelined.py",
                   "parent_sha256": sha256_file(here.parents[1] / "decider_vision" / "qwen3_5_vl_pipelined.py"),
                   "n_image_max": N_IMAGE_MAX, "chunk": CHUNK,
                   "load_report": metas[-1]["load_report"] if metas else None},
        "harness": {"path": "conversion/clef_flash/parity_decoder_torch.py", "sha256": sha256_file(here),
                    "depthwise_conv": "CPU: one torch.bmm over the conv windows (no oneDNN in this torch build), "
                                      "checked against F.conv1d (torch.equal, all channels, all 24 layers) at the "
                                      "last chunk of every CPU run; MPS: F.conv1d",
                    "conv_check": conv_by_process(parts),
                    "invocations": [{k: m[k] for k in ("pid", "started", "stages", "threads", "load_seconds")}
                                    for m in metas]},
        "head": metas[-1]["head"] if metas else None,
        "oracle": {"path": str(ORACLE / "records_oracle.json"), "sha256": sha256_file(ORACLE / "records_oracle.json"),
                   "hf_id": HF_ID, "revision": REVISION, "versions": doc["versions"],
                   "image_embeds": "oracle npz image_embeds (HF fp32 tower merger output), zero-padded to the buffer"},
        "contract": {"input_names": ["input_ids", "position_ids", "image_embeds", "image_rc", "rope_shift_start",
                                     "rope_shift_amount"], "output": "hidden [1, S, 4096] (final norm, every position)",
                     "image": "<|image_pad|> -> V + k (row-major), image_rc[k] = (k // W, k % W), start = 37 + N, "
                              "amount = N - max(H, W) (i0 = 36)",
                     "text": "image_embeds 0, image_rc 0, start 1 << 30, amount 0",
                     "chunks": "ceil(T / S) chunks from zero states, position_ids = 0..cursor+S-1, the last chunk padded "
                               "with 248044, padded outputs discarded"},
        "bar": BAR,
        "versions": versions(),
        "p1": {"result": "PASS" if p1_pass else "FAIL", "device_of_record": p1_dev, "summary": s1,
               "by_source_arm": {k: summarize(v) for k, v in sorted(by_sa.items())}, "runs": p1},
        "p2": {"result": "PASS" if p2_pass else ("FAIL" if p2 else "NOT RUN"), **(p2 or {})},
        "p3": {"runs": p3},
        "p4": {"zero_embeds_moves_argmax_every_run": p4_zero_moves, "runs": p4},
        "p5": {"runs": p5},
        "device_check": json.loads((LANE / "results" / "parity_device_check.json").read_text())
        if (LANE / "results" / "parity_device_check.json").exists() else None,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"P1 {record['p1']['result']}: runs {s1['runs']}/{len(expected)}, argmax {s1['argmax_equal']}/"
          f"{s1['questions']}, max|dp| {s1['max_abs_dp']:.3e}, min pos cos {s1['min_pos_cos']:.8f}, "
          f"rope {s1['rope_planes_equal_runs']}/{s1['runs']}, devices {s1['runs_by_device']}")
    for k, v in sorted(by_sa.items()):
        s = summarize(v)
        print(f"  {k:28s} runs {s['runs']:3d} q {s['questions']:3d} argmax {s['argmax_equal']}/{s['questions']} "
              f"max|dp| {s['max_abs_dp']:.2e} mean {s['mean_run_mean_abs_dp']:.2e} min cos {s['min_pos_cos']:.8f}")
    print(f"P2 {record['p2']['result']}")
    for x in p3:
        print(f"P3 {x['id']}:{x['arm']} worst max|d| {x['worst_max_abs_diff']} worst min cos {x['worst_min_pos_cos']}")
    for x in p4:
        print(f"P4 {x['id']}:{x['arm']} " + "; ".join(f"{a}: changed {v['argmax_changed']}/{v['questions']} "
                                                      f"max|dp| {v['max_abs_dp_vs_base']:.3f}" for a, v in x["arms"].items()))
    for x in p5:
        print(f"P5 {x['id']}:{x['arm']} {json.dumps(x['widths'])}")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stages", default="device,p1,p4,p5", help="device,p1,p4,p5 (one module load) | p2")
    ap.add_argument("--device", default="mps", choices=["cpu", "mps"], help="device of p1/p4/p5 after the check")
    ap.add_argument("--runs", help="p1 subset: comma list id:arm")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--conv-check-every", type=int, default=1,
                    help="CPU runs: check the bmm conv against F.conv1d on every Nth run (and on every kept run)")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--p1-device", default="mps", help="merge: the device whose runs form the P1 table")
    ap.add_argument("--out", default=str(LANE / "results" / "parity_decoder.json"))
    args = ap.parse_args()
    if args.merge:
        merge(args)
    elif args.stages == "p2":
        stage_p2(args)
    else:
        stage_runs(args)


if __name__ == "__main__":
    main()
