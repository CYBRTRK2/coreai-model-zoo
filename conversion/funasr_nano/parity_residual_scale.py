#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""The residual scale is an exact reparametrization: fp32 teacher-forced check on every fixture clip.

For each clip, one forward of prompt (``V + slot`` ids, oracle ``adaptor_out[:N]``) + oracle
``gen_ids[:-1]`` through the engine-shaped decoder (``funasr_decoder.py``, fp32 eager, CPU) in three
variants:

    unscaled    s = 1 (the checkpoint as is)
    scaled      s = 1/4 with the residual RMSNorm eps scaled by s**2 (what the bundles carry)
    scaled_raw  s = 1/4 with eps left at 1e-6 (the RMSNorm eps is effectively 16x larger)

Per generated step: argmax of each scaled variant vs unscaled and vs the oracle token, and the
logits' max |difference| from unscaled. Also the residual-stream absmax of the scaled model (every
decoder layer's output) at position 0 and at every other position, against fp16's 65,504.
-> ``logs/r2_residual_scale.json``.

    ~/code/coreai/coreai-models/.venv/bin/python conversion/funasr_nano/parity_residual_scale.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import work_path  # noqa: E402
from funasr_decoder import FunASRNanoDecoderPipelined, prompt_tensors  # noqa: E402

WORK = work_path("_funasr_nano")
QWEN_DIR = WORK / "official" / "Qwen3-0.6B"
CKPT = WORK / "hf" / "model.safetensors"
FP16_MAX = 65504.0
SCALE = 0.25


@torch.no_grad()
def teacher_forced(model, ids: torch.Tensor, audio: torch.Tensor, golden: list[int], taps: list | None = None):
    """Logits at every generated step ``[len(golden), V]``; ``taps`` collects each layer's output."""
    seq = torch.cat([ids, torch.tensor([golden[:-1]], dtype=torch.int32)], dim=1)
    t = seq.shape[1]
    cfg = model.config
    kc = torch.zeros(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, t, cfg.head_dim)
    vc = torch.zeros_like(kc)
    hooks = []
    if taps is not None:
        hooks = [layer.register_forward_hook(lambda _m, _i, o: taps.append(o[0].abs()))
                 for layer in model.model.layers]
    logits = model(seq, torch.arange(t, dtype=torch.int32)[None], audio, kc, vc)[0]   # audio rows scaled in-graph
    for h in hooks:
        h.remove()
    sp = ids.shape[1]
    return logits[sp - 1: sp - 1 + len(golden)].float()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(QWEN_DIR))
    oracle = json.loads((WORK / "oracle" / "oracle.json").read_text())
    names = [c["name"] for c in oracle["clips"] if not args.only or c["name"] in args.only]

    load = lambda s: FunASRNanoDecoderPipelined.from_safetensors(  # noqa: E731
        CKPT, QWEN_DIR / "config.json", target_dtype=torch.float32, residual_scale=s)
    unscaled, scaled, scaled_raw = load(1.0), load(SCALE), load(SCALE)
    for layer in scaled_raw.model.layers:           # undo the eps compensation on this copy only
        layer.input_layernorm.rmsnorm_impl.eps /= SCALE * SCALE
        layer.post_attention_layernorm.rmsnorm_impl.eps /= SCALE * SCALE
    scaled_raw.model.norm.rmsnorm_impl.eps /= SCALE * SCALE
    qk_eps = {m.model.layers[0].self_attn.qk_norm.rmsnorm_impl.eps for m in (unscaled, scaled, scaled_raw)}
    assert len(qk_eps) == 1, qk_eps   # q/k norms sit off the residual stream: untouched
    eps = {"unscaled": unscaled.model.norm.rmsnorm_impl.eps, "scaled": scaled.model.norm.rmsnorm_impl.eps,
           "scaled_raw": scaled_raw.model.norm.rmsnorm_impl.eps, "qk_norm (all three)": qk_eps.pop()}

    rows = []
    t0 = time.perf_counter()
    for i, name in enumerate(names):
        ids, audio, N, golden = prompt_tensors(name, tok, WORK, torch.float32)
        ref = teacher_forced(unscaled, ids, audio, golden)
        taps: list = []
        a = teacher_forced(scaled, ids, audio, golden, taps)
        b = teacher_forced(scaled_raw, ids, audio, golden)
        res = torch.stack(taps)                              # [28, T, 1024] |residual| of the scaled model
        row = {"name": name, "steps": len(golden),
               "unscaled_argmax_eq_oracle": int((ref.argmax(-1) == torch.tensor(golden)).sum()),
               "scaled_argmax_eq_unscaled": int((a.argmax(-1) == ref.argmax(-1)).sum()),
               "scaled_raw_argmax_eq_unscaled": int((b.argmax(-1) == ref.argmax(-1)).sum()),
               "scaled_logits_max_abs": float((a - ref).abs().max()),
               "scaled_raw_logits_max_abs": float((b - ref).abs().max()),
               "scaled_residual_absmax_pos0": float(res[:, 0].max()),
               "scaled_residual_absmax_other": float(res[:, 1:].max())}
        rows.append(row)
        if i < 5 or i % 25 == 0:
            print(f"[{i + 1}/{len(names)}] {name}: argmax eq {row['scaled_argmax_eq_unscaled']}/"
                  f"{row['scaled_raw_argmax_eq_unscaled']}/{row['steps']}  max|Δlogits| "
                  f"{row['scaled_logits_max_abs']:.2e} / raw {row['scaled_raw_logits_max_abs']:.2e}  residual "
                  f"pos0 {row['scaled_residual_absmax_pos0']:.0f} other {row['scaled_residual_absmax_other']:.0f}",
                  flush=True)

    steps = sum(r["steps"] for r in rows)
    pos0 = max(r["scaled_residual_absmax_pos0"] for r in rows)
    other = max(rows, key=lambda r: r["scaled_residual_absmax_other"])
    summary = {
        "clips": len(rows), "steps": steps, "scale": SCALE, "eps": eps,
        "unscaled_argmax_eq_oracle": sum(r["unscaled_argmax_eq_oracle"] for r in rows),
        "scaled_argmax_eq_unscaled": sum(r["scaled_argmax_eq_unscaled"] for r in rows),
        "scaled_raw_argmax_eq_unscaled": sum(r["scaled_raw_argmax_eq_unscaled"] for r in rows),
        "scaled_logits_max_abs": max(r["scaled_logits_max_abs"] for r in rows),
        "scaled_raw_logits_max_abs": max(r["scaled_raw_logits_max_abs"] for r in rows),
        "scaled_raw_logits_max_abs_clip": max(rows, key=lambda r: r["scaled_raw_logits_max_abs"])["name"],
        "scaled_residual_absmax_pos0": pos0, "fp16_headroom_pos0": FP16_MAX / pos0,
        "scaled_residual_absmax_other": other["scaled_residual_absmax_other"],
        "scaled_residual_absmax_other_clip": other["name"],
        "fp16_headroom_other": FP16_MAX / other["scaled_residual_absmax_other"],
        "unscaled_residual_absmax_pos0": pos0 / SCALE,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    tag = "" if not args.only else "_subset"
    (WORK / "logs" / f"r2_residual_scale{tag}.json").write_text(json.dumps({"summary": summary, "clips": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
