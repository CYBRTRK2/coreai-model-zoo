#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "torch",
#     "safetensors>=0.8.0",
#     "numpy",
# ]
# ///
"""Kev-0.8B pointer head as plain files: head.safetensors + kev_head.json (what the host needs besides the decoder).

The head is two Linear layers over the decoder's final-norm hidden states (author: kev.model.PointerHead):

    z_k = ((W_k h_opt_k + b_k) . (W_q h_decide + b_q)) / sqrt(head_dim)      # raw logit of option k
    p   = softmax_k(z / T)                                                     # T = the calibrated temperature

h_decide is the hidden state at the question's <decide> token (the last token of its row) and h_opt_k the hidden
state at option k's </opt> token. This script copies `q.weight`, `q.bias`, `k.weight`, `k.bias` (fp32) out of the
checkpoint's head.pt into head.safetensors and writes the constants into kev_head.json: head_dim, scale, temperature,
the delimiter token ids, the pad id, and the checkpoint and adapter provenance with their sha256.

Asserted: the file round-trips bit for bit, and the head rebuilt from the files reproduces the oracle's logits for
record 0 bit for bit (kev.model.PointerHead on the oracle's recorded hidden states; needs the author's package).

    HF_HOME=$ZOO_WORK_ROOT/_kev/hf python conversion/kev/export_head.py   # -> $ZOO_WORK_ROOT/_kev/oracle/head/
    HF_HOME=$ZOO_WORK_ROOT/_kev/hf python conversion/kev/export_head.py --model kev-4b \
        --out-dir $ZOO_WORK_ROOT/_kev/oracle_4b/head --oracle-dir $ZOO_WORK_ROOT/_kev/oracle_4b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import work_path  # noqa: E402

ADAPTER = {"repo": "jaredpalmer/kev-0.8b", "tag": "v1.0", "revision": "788ddbdd65715bb03a56788c822f6c632c9a551d",
           "resolved_commit": "bf75a6a8848ea6960ff2ed108d9ed44c2941174f",
           "adapter_sha256": "9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a",
           "head_pt_sha256": "f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6"}
BASE = {"repo": "Qwen/Qwen3.5-0.8B-Base", "revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"}
# Kev-4B (round 4): Hub tag v1.0 -> 591dcb5b, resolved by the Hub to commit 6cfce5c2 (the Kev 1.0 card commit; the
# weights are those of 139fdd94, the card's revision: same LFS oids).
ADAPTER_4B = {"repo": "jaredpalmer/kev-4b", "tag": "v1.0", "revision": "591dcb5bd6d05eb0b5131ea6608f93f10243335c",
              "resolved_commit": "6cfce5c2fa4b4bd64026336ab649c5ca78857d52",
              "adapter_sha256": "90e817356246e7f18bfa7ca3d31794cd4fbeb3332a66a84cb51d9ceae925f2b2",
              "head_pt_sha256": "dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c"}
BASE_4B = {"repo": "Qwen/Qwen3.5-4B-Base", "revision": "1001bb4d826a52d1f399e183466143f4da7b741b"}
MODELS = {"kev-0.8b": (ADAPTER, BASE), "kev-4b": (ADAPTER_4B, BASE_4B)}
DELIMITERS = {"state": ("<|fim_prefix|>", 248060), "q": ("<|fim_middle|>", 248061), "opt": ("<|box_start|>", 248049),
              "opt_end": ("<|box_end|>", 248050), "decide": ("<|fim_suffix|>", 248062)}
PAD = ("<|endoftext|>", 248044)
KEYS = ("q.weight", "q.bias", "k.weight", "k.bias")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf-home", default=str(work_path("_kev", "hf")))
    ap.add_argument("--out-dir", default=str(work_path("_kev", "oracle", "head")))
    ap.add_argument("--oracle-dir", default=str(work_path("_kev", "oracle")))
    ap.add_argument("--model", default="kev-0.8b", choices=sorted(MODELS),
                    help="kev-4b: pass --out-dir <work>/_kev/oracle_4b/head --oracle-dir <work>/_kev/oracle_4b")
    args = ap.parse_args()
    adapter, base = MODELS[args.model]
    snap = Path(args.hf_home) / "hub" / ("models--" + adapter["repo"].replace("/", "--")) / "snapshots" / adapter["revision"]
    head_pt = snap / "head.pt"
    assert sha256(head_pt) == adapter["head_pt_sha256"], "head.pt changed"
    assert sha256(snap / "adapter_model.safetensors") == adapter["adapter_sha256"], "adapter changed"
    meta = torch.load(head_pt, map_location="cpu")
    assert (meta["base"], meta["base_revision"]) == (base["repo"], base["revision"])
    head_dim, T = int(meta["head_dim"]), float(meta["temperature"])
    tensors = {k: meta["head"][k].detach().to(torch.float32).contiguous() for k in KEYS}
    d = tensors["q.weight"].shape[1]
    assert tensors["q.weight"].shape == tensors["k.weight"].shape == (head_dim, d) and tensors["q.bias"].shape == (head_dim,)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out / "head.safetensors", metadata={"format": "pt", "source": f"{adapter['repo']}@{adapter['tag']} head.pt"})
    back = load_file(out / "head.safetensors")
    assert all(torch.equal(back[k], tensors[k]) for k in KEYS), "head.safetensors round trip"
    info = {
        "model": args.model, "hidden_size": d, "head_dim": head_dim, "scale": 1 / math.sqrt(head_dim), "temperature": T,
        "formula": "z_k = ((k.weight @ h_opt_k + k.bias) . (q.weight @ h_decide + q.bias)) * scale; p = softmax_k(z / temperature)",
        "readout": {"decide": "hidden state at the row's last token (<decide>)", "option_k": "hidden state at option k's </opt> token",
                    "hidden": "the decoder's last_hidden_state (after the final RMSNorm), fp32"},
        "row": ("one causal row per question: [state] + user_tokens(render(state)) + [q] + user_tokens(render(instructions)) + "
                "for each option ([opt] + user_tokens(option) + [opt_end]) + [decide]; positions 0..L-1; fresh state per row"),
        "user_tokens": "re.sub(r'<\\|([A-Za-z0-9_]+)\\|>', r'<¦\\1¦>', text), tokenized without special tokens",
        "delimiters": {role: {"token": t, "id": i} for role, (t, i) in DELIMITERS.items()},
        "pad": {"token": PAD[0], "id": PAD[1]},
        "base": base, "adapter": adapter,
        "head_safetensors_sha256": sha256(out / "head.safetensors"),
        "temperature_fit": {k: v for k, v in (meta.get("temperature_fit") or {}).items() if k in ("n", "method", "rows")},
    }
    # The head rebuilt from these two files must reproduce the oracle's record-0 logits bit for bit.
    from kev.model import PointerHead
    head = PointerHead(d, dp=head_dim)
    head.load_state_dict({k: back[k] for k in KEYS})
    head.eval()
    head.temperature = info["temperature"]
    oracle = json.loads((Path(args.oracle_dir) / "records_oracle.json").read_text())
    r0 = oracle["records"][0]
    z = np.load(Path(args.oracle_dir) / "hidden" / f"{r0['id']}.npz")
    checks = []
    for k, q in enumerate(r0["questions"]):
        h = torch.from_numpy(z[f"q{k}_hidden"])
        with torch.no_grad():
            logits = head(h[q["decide"]], h[torch.tensor(q["opts"])])
        ref = torch.tensor(q["logits_T"], dtype=torch.float32)
        checks.append({"record": r0["id"], "qid": q["qid"], "bit_equal": bool(torch.equal(logits, ref)),
                       "max_abs_diff": float((logits - ref).abs().max())})
    assert all(c["bit_equal"] for c in checks), checks
    info["record0_check"] = checks
    (out / "kev_head.json").write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: info[k] for k in ("hidden_size", "head_dim", "scale", "temperature", "head_safetensors_sha256", "record0_check")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
