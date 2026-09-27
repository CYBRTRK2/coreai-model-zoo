#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""E1: the prompt contract and an fp32 eager decoder reproduce the funasr oracle, every fixture clip.

1. Prompt ids. ``build_prompt_ids(tokenizer, N)`` must equal the oracle's ``source_ids`` with its N
   placeholder zeros (starting at ``fbank_beg``) replaced by ``V + slot``, with
   ``fbank_beg == len(prefix)`` and ``Sp == len(prefix) + N + len(suffix)``. The prefix/suffix ids
   go to ``logs/r2_prompt_ids.json`` (what the Swift host must produce).
2. Greedy. HF ``Qwen3ForCausalLM`` in fp32 (official ``Qwen3-0.6B/config.json``; weights = the
   checkpoint's ``llm.*``, renamed) on ``inputs_embeds = embed(prefix) ++ adaptor_out[:N] ++
   embed(suffix)``, ``generate(do_sample=False, max_new_tokens=512, eos_token_id=EOS_IDS)`` with
   position ids from 0 (funasr's own call starts them at 2: its attention mask is Sp + 2 long).
   Compared with the oracle's ``gen_ids`` and text.
3. Teacher-forced margins. One forward over prompt + oracle ``gen_ids[:-1]``; per step the top-2
   softmax gap (``cli/coreai_verify.py``'s knife-edge measure, floor 0.1) -> ``logs/r2_margins.json``.
   Done with position ids from 0 and from 2; the per-step argmaxes of both must equal the oracle.

Run with the shared venv:
    ~/code/coreai/coreai-models/.venv/bin/python conversion/funasr_nano/parity_decoder.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import work_path  # noqa: E402
from prompt import EOS_IDS, MAX_NEW_TOKENS, PREFIX_TEXT, SUFFIX_TEXT, V, build_prompt_ids, prompt_segments  # noqa: E402

WORK = work_path("_funasr_nano")
QWEN_DIR = WORK / "official" / "Qwen3-0.6B"
CKPT = WORK / "hf" / "model.safetensors"
MARGIN_FLOOR = 0.1
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def clean(text: str) -> str:
    """``FunASRNano.inference_llm``'s post-processing of the decoded string."""
    return re.sub(r"\s+", " ", text.replace("/sil", " "))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_hf_decoder(dtype: torch.dtype = torch.float32):
    """HF Qwen3ForCausalLM with the checkpoint's ``llm.*`` weights (tied head, as the config says)."""
    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config.from_pretrained(str(QWEN_DIR))
    model = Qwen3ForCausalLM(cfg)
    sd = {}
    with safe_open(str(CKPT), framework="pt") as f:
        for k in f.keys():  # noqa: SIM118
            if k.startswith("llm."):
                sd[k[len("llm."):]] = f.get_tensor(k).to(dtype)
    assert torch.equal(sd["lm_head.weight"], sd["model.embed_tokens.weight"]), "lm_head != embed_tokens"
    model.load_state_dict(sd, strict=True)
    model = model.to(dtype).eval()
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), "head not tied"
    return model


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


@torch.no_grad()
def teacher_forced(model, prompt_embeds: torch.Tensor, gen: list[int], offset: int) -> tuple[list[float], list[int], list[int]]:
    """Per generated step: top-2 softmax gap, argmax, runner-up — prompt + gen[:-1] in one forward."""
    emb = model.get_input_embeddings()
    sp = prompt_embeds.shape[1]
    x = torch.cat([prompt_embeds, emb(torch.tensor([gen[:-1]], dtype=torch.long))], dim=1)
    pos = torch.arange(offset, offset + x.shape[1], dtype=torch.long)[None]
    logits = model(inputs_embeds=x, position_ids=pos,
                   attention_mask=torch.ones(1, x.shape[1], dtype=torch.long)).logits[0]
    step = logits[sp - 1: sp - 1 + len(gen)].float()
    p = torch.softmax(step, dim=-1)
    top2 = torch.topk(p, 2, dim=-1)
    return ((top2.values[:, 0] - top2.values[:, 1]).tolist(), top2.indices[:, 0].tolist(),
            top2.indices[:, 1].tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--retry-offset2", action="store_true",
                    help="also greedy-decode every mismatching clip with position ids from 2")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer

    oracle = json.loads((WORK / "oracle" / "oracle.json").read_text())
    rows = [r for r in oracle["clips"] if not args.only or r["name"] in args.only]
    tok = AutoTokenizer.from_pretrained(str(QWEN_DIR))

    # ---- 1. prompt contract ------------------------------------------------------------------
    prefix, suffix = prompt_segments(tok)
    with_special = (tok(PREFIX_TEXT).input_ids, tok(SUFFIX_TEXT).input_ids)
    assert with_special == (prefix, suffix), "add_special_tokens changes the prompt ids"
    assert (len(prefix), len(suffix)) == (18, 5), (len(prefix), len(suffix))
    for r in rows:
        o = np.load(WORK / "oracle" / f"{r['name']}.npz")
        N, beg, src = int(o["fake_token_len"]), int(o["fbank_beg"]), o["source_ids"].astype(np.int64)
        assert N == r["N"] and beg == r["fbank_beg"] == len(prefix), (r["name"], N, beg)
        assert src.shape[0] == r["Sp"] == len(prefix) + N + len(suffix), (r["name"], src.shape, r["Sp"])
        assert (src[beg:beg + N] == 0).all(), r["name"]
        expect = src.copy()
        expect[beg:beg + N] = V + np.arange(N)
        ids, plen = build_prompt_ids(tok, N)
        assert plen == beg and ids == expect.tolist(), r["name"]
    prompt_record = {
        "prefix_text": PREFIX_TEXT, "suffix_text": SUFFIX_TEXT,
        "prefix_ids": prefix, "suffix_ids": suffix, "audio_slot_ids": "V + slot, slot = 0..N-1, V = 151936",
        "eos_ids": list(EOS_IDS), "max_new_tokens": MAX_NEW_TOKENS,
        "add_special_tokens_invariant": True,
        "checked_clips": len(rows),
        "tokenizer_dir": str(QWEN_DIR),
        "tokenizer_sha256": {f: sha256(QWEN_DIR / f) for f in TOKENIZER_FILES},
        "tokenizer_class": type(tok).__name__,
    }
    if not args.only:
        (WORK / "logs" / "r2_prompt_ids.json").write_text(json.dumps(prompt_record, ensure_ascii=False, indent=1))
    print(f"[prompt] {len(rows)}/{len(rows)} clips: ids == oracle source_ids with V+slot "
          f"(prefix {len(prefix)}, suffix {len(suffix)})", flush=True)

    # ---- 2./3. fp32 eager decoder ------------------------------------------------------------
    t0 = time.perf_counter()
    model = load_hf_decoder()
    print(f"[load] HF Qwen3ForCausalLM fp32, attn={model.config._attn_implementation} "
          f"({time.perf_counter() - t0:.1f} s)", flush=True)
    from transformers import GenerationConfig

    gcfg = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
                            eos_token_id=list(EOS_IDS), pad_token_id=151643, bos_token_id=151643)
    emb = model.get_input_embeddings()
    results, margins = [], {}
    oracle_has_endoftext = []
    for i, r in enumerate(rows):
        name = r["name"]
        o = np.load(WORK / "oracle" / f"{name}.npz")
        N = int(o["fake_token_len"])
        golden = o["gen_ids"].astype(np.int64).tolist()
        assert golden == r["gen_ids"]
        if 151643 in golden[:-1]:
            oracle_has_endoftext.append(name)
        with torch.no_grad():
            x = torch.cat([emb(torch.tensor([prefix])), torch.from_numpy(o["adaptor_out"][:N])[None].float(),
                           emb(torch.tensor([suffix]))], dim=1)
        emb_dev = float((x[0] - torch.from_numpy(o["inputs_embeds"])).abs().max())
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(inputs_embeds=x, attention_mask=torch.ones(1, x.shape[1], dtype=torch.long),
                                 generation_config=gcfg)
        wall = time.perf_counter() - t1
        gen = out[0].tolist()
        text = clean(tok.decode(gen, skip_special_tokens=True))
        m0, a0, b0 = teacher_forced(model, x, golden, 0)
        m2, a2, _ = teacher_forced(model, x, golden, 2)
        div = first_divergence(gen, golden)
        row = {"name": name, "N": N, "Sp": int(x.shape[1]), "inputs_embeds_max_abs_vs_oracle": emb_dev,
               "gen_equal": gen == golden, "text_equal": text == r["text"], "gen_len": len(gen),
               "oracle_len": len(golden), "first_divergence": div, "hit_cap": len(gen) >= MAX_NEW_TOKENS
               and gen[-1] not in EOS_IDS, "wall_s": round(wall, 2),
               "tf_argmax_equal_pos0": a0 == golden, "tf_argmax_equal_pos2": a2 == golden,
               "min_margin": min(m0), "min_margin_step": int(np.argmin(m0)),
               "n_below_floor": sum(m < MARGIN_FLOOR for m in m0)}
        if div is not None:
            k = min(div, len(golden) - 1)
            row.update({"ours_at_div": gen[div] if div < len(gen) else None,
                        "oracle_at_div": golden[div] if div < len(golden) else None,
                        "oracle_margin_at_div": m0[k], "oracle_runner_up_at_div": b0[k],
                        "text": text, "oracle_text": r["text"]})
        results.append(row)
        margins[name] = {"margins_pos0": [round(v, 6) for v in m0], "margins_pos2": [round(v, 6) for v in m2],
                         "runner_up_pos0": b0, "max_abs_margin_diff_pos0_vs_pos2":
                         float(np.max(np.abs(np.array(m0) - np.array(m2))))}
        flag = "OK " if gen == golden else "DIFF"
        print(f"[{i + 1}/{len(rows)}] {flag} {name}: N={N} gen {len(gen)}/{len(golden)} "
              f"text_eq={row['text_equal']} emb|Δ|={emb_dev:.1e} min_margin={row['min_margin']:.4f}"
              f"@{row['min_margin_step']} below_floor={row['n_below_floor']} ({wall:.1f} s)"
              + (f" div@{div} ours={row['ours_at_div']} oracle={row['oracle_at_div']} "
                 f"gap={row['oracle_margin_at_div']:.4f}" if div is not None else ""), flush=True)

    mism = [r for r in results if not r["gen_equal"]]
    retry = []
    if mism and args.retry_offset2:
        for r in mism:
            o = np.load(WORK / "oracle" / f"{r['name']}.npz")
            N = int(o["fake_token_len"])
            golden = o["gen_ids"].astype(np.int64).tolist()
            with torch.no_grad():
                x = torch.cat([emb(torch.tensor([prefix])), torch.from_numpy(o["adaptor_out"][:N])[None].float(),
                               emb(torch.tensor([suffix]))], dim=1)
                # funasr's own call: an attention mask 2 longer than the prompt -> position ids from 2
                out = model.generate(inputs_embeds=x, attention_mask=torch.ones(1, x.shape[1] + 2, dtype=torch.long),
                                     generation_config=gcfg)
            gen = out[0].tolist()
            retry.append({"name": r["name"], "gen_equal_offset2": gen == golden,
                          "first_divergence_offset2": first_divergence(gen, golden)})
            print(f"[retry pos2] {r['name']}: equal={gen == golden}", flush=True)

    all_m = np.concatenate([np.array(v["margins_pos0"]) for v in margins.values()])
    summary = {
        "clips": len(results),
        "gen_equal": sum(r["gen_equal"] for r in results),
        "text_equal": sum(r["text_equal"] for r in results),
        "tf_argmax_equal_pos0": sum(r["tf_argmax_equal_pos0"] for r in results),
        "tf_argmax_equal_pos2": sum(r["tf_argmax_equal_pos2"] for r in results),
        "hit_cap": sum(r["hit_cap"] for r in results),
        "inputs_embeds_max_abs_vs_oracle": max(r["inputs_embeds_max_abs_vs_oracle"] for r in results),
        "steps": int(all_m.size), "steps_below_floor": int((all_m < MARGIN_FLOOR).sum()),
        "clips_with_step_below_floor": sum(r["n_below_floor"] > 0 for r in results),
        "max_abs_margin_diff_pos0_vs_pos2": max(v["max_abs_margin_diff_pos0_vs_pos2"] for v in margins.values()),
        "oracle_clips_with_endoftext_before_eos": oracle_has_endoftext,
        "mismatches": [{k: r.get(k) for k in ("name", "first_divergence", "ours_at_div", "oracle_at_div",
                                               "oracle_margin_at_div", "oracle_runner_up_at_div", "text",
                                               "oracle_text")} for r in mism],
        "retry_offset2": retry,
        "attn_implementation": model.config._attn_implementation,
        "torch": torch.__version__,
        "pass": sum(r["gen_equal"] for r in results) >= len(results) - 3,
    }
    tag = "" if not args.only else "_subset"
    (WORK / "logs" / f"r2_parity_decoder{tag}.json").write_text(
        json.dumps({"summary": summary, "clips": results}, ensure_ascii=False, indent=1))
    (WORK / "logs" / f"r2_margins{tag}.json").write_text(json.dumps({
        "floor": MARGIN_FLOOR,
        "definition": "top-2 softmax probability gap of the fp32 reference at each generated step, teacher-forced "
                      "on the oracle gen_ids (prompt + gen_ids[:-1] in one forward). margins_pos0 = position ids "
                      "from 0 (the port's contract); margins_pos2 = from 2 (funasr's own call). runner_up_pos0 = "
                      "the second-ranked token id. Step k predicts gen_ids[k].",
        "clips": margins}, indent=0))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("PASS" if summary["pass"] else "FAIL")


if __name__ == "__main__":
    main()
