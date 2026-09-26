#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""E3: export the Fun-ASR-Nano decoder (Qwen3-0.6B) to Core AI — ship ``_s1`` + gate ``unified`` bundles.

One module, quantized once (``--mode int8lin`` / ``int8hu``) or not at all (``--mode fp16``), exported twice:

* ``funasr_nano_2512_decode_<mode>_n63_s1/`` — the ship shape (``conversion/qwen3_asr/export_decoder.py``):
  static ``[1, 1]`` query, dynamic KV, ``input_ids, position_ids, audio_embeds [63, 1024], keyCache,
  valueCache -> logits``, the pipelined-engine contract CoreAIKit drives.
* ``funasr_nano_2512_<mode>_unified_cl<C>/`` — the gate shape (``conversion/qwen3_asr/export_unified.py``):
  ``prefill`` (``input_ids [1, Sp]`` with Sp dynamic 16..128, ``audio_embeds [N, 1024]`` with N dynamic
  1..64, writes KV slots [0, Sp), returns last-token logits) + ``decode`` (``input_ids [1, 1]``,
  ``pos [1]`` int32 value) sharing one weight copy and one KV state of C slots. Fully static decode
  runs in the Python runtime. ``rope`` and ``scaled_dot_product_attention`` are NOT externalized here:
  the engine-native RoPE mishandles a baked-constant ``position_ids`` and the SDPA composite cannot
  take the explicit causal-buffer mask (qwen3_asr STATE.md, "P3 SOLVED").

int8lin = ``linear_quant_config("int8")`` as is: weight-only per-block-32 symmetric_with_clipping on
every decoder linear; embedding, norms, RoPE, SDPA and the tied head stay fp16 (one shared table).
int8hu = the qwen3.5 ship recipe: ``linear_quant_config("int8")`` plus an untied int8 head at
per-block-32 **plain symmetric (absmax)** — ``head_quant_spec("block32", True)``. The qwen3_asr driver
calls ``head_quant_spec()``, which meant absmax when it was written and has meant clipping since the
helper moved to ``_bundle.py`` (default ``sym=False``); the absmax head is the recipe.

Both shapes carry the residual scale (``funasr_decoder.py``, default s = 1/4): without it the
first token's residual peaks at ~125k, past fp16, and every logit is NaN on the engine. The unified
``prefill`` scales its ``audio_embeds`` input in-graph (``FunASRNanoStaticPrefill``); text rows are
scaled by the embedding wrapper in every graph. The host passes the encoder output unchanged.

Each bundle directory gets ``metadata.json`` (with ``residual_scale`` and ``rmsnorm_eps_residual``)
and ``tokenizer/`` = the official ``Qwen3-0.6B/`` tokenizer files copied byte for byte
(``save_pretrained`` would re-serialize them).

Run with the shared venv:
    ~/code/coreai/coreai-models/.venv/bin/python conversion/funasr_nano/export_decoder.py --mode int8hu
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[0] / "qwen3_asr"))   # qwen3_asr_static; last, so our names win

from _bundle import head_quant_spec, write_bundle_metadata  # noqa: E402
from _paths import work_path  # noqa: E402
from funasr_decoder import N_AUDIO_MAX, PIPELINED_STATE_NAMES, FunASRNanoDecoderPipelined, greedy, prompt_tensors  # noqa: E402
from prompt import EOS_IDS  # noqa: E402

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN  # noqa: E402
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai  # noqa: E402

WORK = work_path("_funasr_nano")
QWEN_DIR = WORK / "official" / "Qwen3-0.6B"
CKPT = WORK / "hf" / "model.safetensors"
EXPORTS = WORK / "exports"
HF_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
REVISION = "272c57b82523ada6fd87095e955f8e29100979ab"
WEIGHTS = ("FunAudioLLM/Fun-ASR-Nano-2512-vllm@a4362c943d48951f98ca2a62181cc028970270c5 model.safetensors "
           "(bit-identical to the official model.pt), llm.* tensors")
DTYPE = torch.float16
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
_DROP = {"scaled_dot_product_attention", "rope"}
EXAMPLES = ("zh", "en", "ja", "ko", "yue")



def linear_quant_config(dtype: str = "int8") -> dict:
    """Weight-only linear per-block-32 (the qwen3.5 ship recipe).

    Verbatim from ``conversion/export_qwen3_vl_pipelined.py``, which cannot be imported in the shared
    venv (its ``coreai_models.models.macos.qwen3_vl`` import needs a ``PIPELINED_STATE_NAMES`` the
    installed package does not have).
    """
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": dtype,
                    "qscheme": "symmetric_with_clipping",
                    "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
                }
            },
            "op_input_spec": None,
            "op_output_spec": None,
        },
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rope.RoPE": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "torch.nn.modules.sparse.Embedding": None,
        },
        "module_name_configs": {r".*lm_head$": None},
    }

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_tokenizer(out_dir: Path) -> dict:
    dst = out_dir / "tokenizer"
    dst.mkdir()
    shas = {}
    for f in TOKENIZER_FILES:
        shutil.copyfile(QWEN_DIR / f, dst / f)   # follows the hub-cache symlink, copies bytes
        shas[f] = sha256(dst / f)
        assert shas[f] == sha256(QWEN_DIR / f), f
    return shas


def du(path: Path) -> str:
    return subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True).stdout.split()[0]


@torch.no_grad()
def eager_sanity(model, tok, label: str) -> list[dict]:
    """First-token argmax + full greedy vs the oracle on the five examples, eager (CPU)."""
    rows = []
    for clip in EXAMPLES:
        ids, audio, N, golden = prompt_tensors(clip, tok, WORK, DTYPE)
        gen, first = greedy(model, ids, audio, len(golden) + 8, EOS_IDS)
        top2 = torch.topk(torch.softmax(first, -1), 2)
        rows.append({"clip": clip, "first_argmax": int(first.argmax()), "golden_first": golden[0],
                     "first_ok": int(first.argmax()) == golden[0],
                     "first_top2_gap": float(top2.values[0] - top2.values[1]),
                     "greedy_equal": gen == golden, "gen_len": len(gen), "golden_len": len(golden)})
        print(f"[eager {label}] {clip}: first {rows[-1]['first_argmax']} golden {golden[0]} "
              f"{'OK' if rows[-1]['first_ok'] else 'MISMATCH'}; greedy "
              f"{'== oracle' if gen == golden else '!= oracle'} ({len(gen)}/{len(golden)})", flush=True)
    return rows


def save(prog, out_dir: Path, name: str) -> Path:
    import coreai.runtime as rt

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aimodel = out_dir / f"{name}.aimodel"
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    return aimodel


def scale_metadata(model) -> dict:
    s = model.residual_scale
    eps = model.model.norm.rmsnorm_impl.eps
    assert all(l.input_layernorm.rmsnorm_impl.eps == eps == l.post_attention_layernorm.rmsnorm_impl.eps
               for l in model.model.layers)
    return {"residual_scale": s, "rmsnorm_eps_residual": eps}


def export_s1(model, mode: str, max_ctx: int) -> dict:
    name = f"funasr_nano_2512_decode_{mode}_n{model.n_audio_tokens}_s1"
    spec = model.build_export_spec(DTYPE, max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, trace_query=1)
    t0 = time.perf_counter()
    print(f"exporting {name} ...", flush=True)
    prog = export_to_coreai(
        model, spec["reference_inputs"], dynamic_shapes=spec["dynamic_shapes"],
        input_names=spec["input_names"], output_names=spec["output_names"],
        state_names=PIPELINED_STATE_NAMES, externalize_modules=list(_EXTERNALIZE_SPECS))
    prog.optimize()
    out_dir = EXPORTS / name
    aimodel = save(prog, out_dir, name)
    write_bundle_metadata(out_dir, name, HF_ID, model.config.vocab_size, max_ctx,
                          revision=REVISION, weights=WEIGHTS, mode=mode, extra=scale_metadata(model))
    shas = copy_tokenizer(out_dir)
    rec = {"name": name, "dir": str(out_dir), "aimodel_du": du(aimodel), "dir_du": du(out_dir),
           "tokenizer_sha256": shas, "export_s": round(time.perf_counter() - t0, 1)}
    print(f"bundle ready: {rec}", flush=True)
    return rec


def export_unified(base, mode: str, cache_len: int, max_sp: int, max_n: int, max_ctx: int) -> dict:
    import coreai_torch
    from coreai_models.export.mlir_ops import register_custom_torch_lowering, remove_functionalization
    from qwen3_asr_static import STATE_NAMES, Qwen3ASRStaticDecode, Qwen3ASRStaticPrefill, build_kv_state

    def mk_export_fn(ref: dict, dyn: dict | None = None):
        def export_fn(m):
            with torch.no_grad():
                ep = torch.export.export(m, args=(), kwargs=ref, dynamic_shapes=dyn)
            ep = ep.run_decompositions(coreai_torch.get_decomp_table())
            remove_functionalization(ep)
            return ep
        return export_fn

    class FunASRNanoStaticPrefill(Qwen3ASRStaticPrefill):
        """qwen3_asr's static prefill; the audio rows join the residual stream at ``base.residual_scale``."""

        def forward(self, input_ids, audio_embeds, k_cache, v_cache):
            if base.residual_scale != 1.0:
                audio_embeds = audio_embeds * base.residual_scale
            return super().forward(input_ids, audio_embeds, k_cache, v_cache)

    cfg = base.config
    prefill = FunASRNanoStaticPrefill(base).eval()
    decode = Qwen3ASRStaticDecode(base).eval()
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name not in _DROP]
    st = build_kv_state(cfg, cache_len, DTYPE)
    sp_trace, n_trace = 55, 32   # any interior point of the dynamic ranges
    pref_ref = {"input_ids": torch.zeros(1, sp_trace, dtype=torch.int32),
                "audio_embeds": torch.zeros(n_trace, cfg.hidden_size, dtype=DTYPE),
                "k_cache": st["k_cache"].clone(), "v_cache": st["v_cache"].clone()}
    dec_ref = {"input_ids": torch.zeros(1, 1, dtype=torch.int32),
               "pos": torch.tensor([sp_trace], dtype=torch.int32),
               "k_cache": st["k_cache"].clone(), "v_cache": st["v_cache"].clone()}
    pref_dyn = {"input_ids": {1: torch.export.Dim("sp", min=16, max=max_sp)},
                "audio_embeds": {0: torch.export.Dim("n", min=1, max=max_n)},
                "k_cache": None, "v_cache": None}

    name = f"funasr_nano_2512_{mode}_unified_cl{cache_len}"
    t0 = time.perf_counter()
    print(f"exporting {name} (prefill dynamic Sp/N + decode static, shared weights + state) ...", flush=True)
    conv = coreai_torch.TorchConverter()
    conv.add_pytorch_module(
        prefill, export_fn=mk_export_fn(pref_ref, pref_dyn), externalize_modules=specs,
        input_names=("input_ids", "audio_embeds"), output_names=("logits",),
        state_names=STATE_NAMES, entrypoint_name="prefill")
    conv.add_pytorch_module(
        decode, export_fn=mk_export_fn(dec_ref), externalize_modules=specs,
        input_names=("input_ids", "pos"), output_names=("logits",),
        state_names=STATE_NAMES, entrypoint_name="decode")
    register_custom_torch_lowering(conv)
    prog = conv.to_coreai()
    prog.optimize()
    out_dir = EXPORTS / name
    aimodel = save(prog, out_dir, name)
    write_bundle_metadata(out_dir, name, HF_ID, cfg.vocab_size, max_ctx, revision=REVISION, weights=WEIGHTS,
                          mode=mode, extra={**scale_metadata(base),
                                            "gate_only": "prefill/decode static pair for the Python-runtime "
                                                         "e2e gate; not the CoreAIKit ship shape"})
    meta = json.loads((out_dir / "metadata.json").read_text())
    meta["language"]["function_map"] = {"prefill": ["prefill"], "decode": ["decode"]}
    meta["language"]["kv_cache_len"] = cache_len
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    shas = copy_tokenizer(out_dir)
    rec = {"name": name, "dir": str(out_dir), "aimodel_du": du(aimodel), "dir_du": du(out_dir),
           "tokenizer_sha256": shas, "export_s": round(time.perf_counter() - t0, 1),
           "prefill_sp_range": [16, max_sp], "prefill_n_range": [1, max_n], "cache_len": cache_len}
    print(f"bundle ready: {rec}", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default="int8lin", choices=["fp16", "int8lin", "int8hu"])
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--cache-len", type=int, default=1024)
    ap.add_argument("--max-sp", type=int, default=128)
    ap.add_argument("--max-n", type=int, default=64)
    ap.add_argument("--skip-s1", action="store_true")
    ap.add_argument("--skip-unified", action="store_true")
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--residual-scale", type=float, default=0.25,
                    help="run the residual stream at this scale (power of two; 1 = off, which overflows fp16)")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(QWEN_DIR))
    t0 = time.perf_counter()
    model = FunASRNanoDecoderPipelined.from_safetensors(CKPT, QWEN_DIR / "config.json", N_AUDIO_MAX, DTYPE,
                                                        residual_scale=args.residual_scale)
    print(f"loaded decoder fp16 (N={model.n_audio_tokens}, residual scale {model.residual_scale}) "
          f"in {time.perf_counter() - t0:.1f} s", flush=True)
    record = {"mode": args.mode, "torch": torch.__version__, **scale_metadata(model)}

    if args.mode in ("int8lin", "int8hu"):
        from coreai_models.export.compression import quantize_pytorch_model

        spec = model.build_export_spec(DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)
        cfg_q = linear_quant_config("int8")          # int8lin: lm_head -> None, head stays tied fp16
        if args.mode == "int8hu":
            cfg_q["module_name_configs"] = {r".*lm_head$": head_quant_spec("block32", True)}
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())   # untie
        record["quant_config"] = json.loads(json.dumps(cfg_q))
        t1 = time.perf_counter()
        print(f"quantizing ({args.mode}) ...", flush=True)
        model = quantize_pytorch_model(model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg_q)
        record["quantize_s"] = round(time.perf_counter() - t1, 1)
    if not args.skip_sanity:
        record["eager_sanity"] = eager_sanity(model, tok, args.mode)

    record["bundles"] = []
    if not args.skip_s1:
        record["bundles"].append(export_s1(model, args.mode, args.max_ctx))
    if not args.skip_unified:
        record["bundles"].append(export_unified(model, args.mode, args.cache_len, args.max_sp, args.max_n,
                                                args.max_ctx))
    out = WORK / "logs" / f"r2_export_{args.mode}.json"
    out.write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(record, indent=1, default=str))


if __name__ == "__main__":
    main()
