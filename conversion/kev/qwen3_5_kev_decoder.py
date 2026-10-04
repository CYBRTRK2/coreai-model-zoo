# Kev decoder: the Qwen3.5-0.8B hybrid backbone (LoRA merged) returning the final-norm hidden
# state at every position of a static-S chunk (no vocabulary head).
#
# Community port — NOT an Apple model.
#
# jaredpalmer/kev-0.8b never generates text: its pointer head reads the backbone's
# `last_hidden_state` (after the final RMSNorm) at the row's <decide> token and at every </opt>
# token. So the graph is the overlay's text decode graph (`Qwen3_5StatefulForCausalLM`,
# coreai_models/models/macos/qwen3_5.py: ids input, in-graph embedding gather and plain partial
# RoPE, four hybrid states) with three changes and nothing else:
#
#   (a) no `lm_head`: the forward returns `self.model.forward_stateful(...)` itself, the
#       final-norm hidden at EVERY query position ([1, S, hidden]; `last_token_only` is refused).
#       The tied [248320, 1024] table stays in the graph only as the embedding the ids gather
#       from; the parent's head slot is an Identity, so a weight-only int8 pass sees decoder
#       linears and nothing else (the tied Linear would alias the embedding table).
#   (b) `from_hf` reads the author's full-weight checkpoint (`scripts/merge_lora_checkpoint.py`
#       output: flat `qwen3_5_text` config, keys without a prefix) through the overlay's
#       `from_hf_memory_efficient(..., hf_config_attr=None, hf_state_dict_prefix="")` and
#       records every checkpoint key against the module's tensors in `load_report` (the overlay
#       loader is strict=False).
#   (c) `build_export_spec`: decider_vision's static-S spec (`Qwen3_5VLPipelinedForCausalLM.
#       build_export_spec`, conversion/decider_vision/qwen3_5_vl_pipelined.py) without the four
#       image inputs: static `input_ids [1, S]`, dynamic `position_ids` and KV sequence dims,
#       output `hidden`.
#
# Why the overlay's text class and not clef-flash's VL child (conversion/clef_flash/
# qwen3_5_clef_decoder.py) with the image inputs held constant: Kev reads text only, and the
# text class's forward IS the overlay's `forward_stateful`, so the graph is the text decode graph
# bit for bit (P2 in parity_decoder_torch.py checks it) with no constant image buffer or
# M-RoPE blend to carry.
#
# Graph (static S; states mutate in place):
#
#   inputs  input_ids     [1, S]    int32  static
#           position_ids  [1, seq]  int32  dynamic; the cache ramp 0..seq-1 (offset = seq - S)
#   states  keyCache / valueCache [6, 1, 2, ctx, 256] (ctx dynamic), convState [18, 1, 6144, 3],
#           recState [18, 1, 16, 128, 128]
#   output  hidden        [1, S, 1024]  final-norm hidden at every position
#
# A row of T tokens runs as ceil(T / S) calls from fresh zero states; call c gets ids[cS : cS + S]
# with position_ids 0..cS+S-1, the last call is padded with <|endoftext|> (248044) and the padded
# positions' outputs are discarded (causal, so they cannot reach a real position). Every GDN
# layer runs `use_loopfree_unroll` (the overlay's `_gated_delta_step_unroll`: S single steps
# unrolled in-graph, fp32 inside the call, no doubling inverse); set it on every linear-attention
# layer before tracing.
#
# Round 14 (`build_dynamic_export_spec`, Metal-kernel scan only): the query length S is a Dim(2..cap) and the output is
# hidden [1, S, hidden]; the kernel is built with chunk_max = cap and reads S from its input's extent, so a host calls
# with any length in 2..cap. The AOT stub of this graph (`main-h16c.mlirb`, the function signature) does not carry the
# cap: two caps compile to the same stub and so to the same coreai-cache entry name (round 14 trap).
#
# Round 16 (`set_metal_scan(..., static_io=)`, dynamic-S graph only): the same graph contract with every edge of the
# kernel a static shape — the kernel's three S-long inputs zero-padded in-graph to the cap, the call's length S passed
# as a 1-element int32 input ("slen") or the padded steps run as no-ops ("pad"); the kernel and its module are
# conversion/kev/qwen3_5_gdn_metal_static.py (a copy of the overlay's). Without static_io nothing changes. Measured: each
# call 0.9-1.4 ms slower and no less memory per call length than the dynamic-edge graph (not shipped).
from __future__ import annotations

import math

import torch

from coreai_models.models.macos.qwen3_5 import (
    DECODE_STATE_NAMES,
    Qwen3_5StatefulForCausalLM,
    build_decode_state,
)
from coreai_models.primitives.macos.cache import KVCache, SSMState

PREFILL_CHUNK = 16
INPUT_NAMES = ("input_ids", "position_ids")
OUTPUT_NAMES = ("hidden",)
GDN_SCANS = ("unroll", "chunk", "metal")
QUERY_MIN = 2   # round 14: the dynamic-S graph's smallest call (no S=1 call)
METAL_KERNEL_SOURCE = "coreai_models.models.macos.qwen3_5_gdn_metal.build_gdn_chunk_kernel (overlay, unchanged)"
# Round 16: the static-edge copy, by static_io; the name suffix of the kernel and the bundle.
STATIC_KERNEL_SOURCE = "conversion/kev/qwen3_5_gdn_metal_static.build_gdn_chunk_kernel_static (copy of the overlay's)"
STATIC_IO_SUFFIX = {"slen": "s", "pad": "p"}
STATIC_IO_TEXT = {
    "slen": "static: the kernel's conv / g / beta inputs zero-padded in-graph to chunk_max, the call's length as a "
            "1-element int32 input SLEN, the scan runs t < SLEN[0]",
    "pad": "static: the kernel's conv / g / beta inputs zero-padded in-graph to chunk_max, no length input, the scan runs "
           "all chunk_max steps (a padded step leaves the state as it is)",
}

__all__ = ["GDN_SCANS", "INPUT_NAMES", "OUTPUT_NAMES", "PREFILL_CHUNK", "QUERY_MIN", "Qwen3_5KevDecoder", "STATIC_IO_SUFFIX",
           "metal_kernel_name", "metal_scan_record", "set_chunk_scan", "set_metal_scan", "set_unrolled_scan"]


def set_unrolled_scan(model: torch.nn.Module) -> int:
    """Put every linear-attention layer on the static-S unrolled step scan; returns the count."""
    n = 0
    for layer in model.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True
            layer.linear_attn.use_loopfree_unroll = True   # S is a static int at trace time
            n += 1
    return n


# Round 11: two more forms of the GDN scan, same graph contract (the scan is internal to each linear-attention layer).
def set_chunk_scan(model: torch.nn.Module, chunk: int) -> dict:
    """Every linear-attention layer on the overlay's in-graph chunk scan (`_gated_delta_chunk`: the S tokens in
    parallel, the triangular inverse as a fixed product of ceil(log2 S) doublings, fp32 inside the call). The unroll
    flag is cleared because it takes precedence in the overlay's forward."""
    doublings = max(1, math.ceil(math.log2(chunk)))
    n = 0
    for layer in model.model.layers:
        if not layer.is_full:
            la = layer.linear_attn
            la.use_loopfree_unroll = False
            la.use_loopfree_chunk = True
            la.chunk_doublings = doublings
            n += 1
    return {"gdn_scan": "chunk", "linear_layers": n, "chunk_doublings": doublings}


def metal_kernel_name(chunk: int, dynamic: bool = False, static_io: str | None = None) -> str:
    """The kernel's name: `qwen3_5_gdn_chunk_s<S>` for a static S, `qwen3_5_gdn_chunk_dyn<cap>` for round 14's
    dynamic-S graph, `..._dyn<cap>s` / `..._dyn<cap>p` for round 16's static-edge kernel (slen / pad)."""
    return f"qwen3_5_gdn_chunk_{'dyn' if dynamic else 's'}{chunk}{STATIC_IO_SUFFIX[static_io] if static_io else ''}"


def metal_scan_record(linear_layers: int, num_v: int, dv: int, chunk: int, name: str,
                      static_io: str | None = None) -> dict:
    """The record of the Metal-kernel scan (`metadata.json`'s `gdn_scan` block without `linear_layers`): the grid is
    one thread per value channel (dv) and value head (num_v). Round 15 writes a bundle's metadata from the checkpoint's
    config.json with it (export_decoder.py --metadata-only), so it takes the config's numbers, not the module. Round 16
    adds `kernel_io` for the static-edge kernel (absent = the overlay's kernel, its edges as long as the call)."""
    rec = {"gdn_scan": "metal", "linear_layers": linear_layers, "kernel": name, "chunk_max": chunk,
           "kernel_source": STATIC_KERNEL_SOURCE if static_io else METAL_KERNEL_SOURCE,
           "threads_per_grid": [dv, num_v, 1], "threads_per_thread_group": [dv, 1, 1]}
    if static_io:
        rec["kernel_io"] = STATIC_IO_TEXT[static_io]
    return rec


def set_metal_scan(model: torch.nn.Module, chunk: int, name: str | None = None, static_io: str | None = None):
    """Every linear-attention layer on the overlay's fp32 Metal chunk kernel (`qwen3_5_gdn_metal`: the step recurrence
    over the whole chunk in one GPU dispatch per layer, qk l2-norm and q-scale inside the kernel), built here with
    chunk_max = S (`metalize_gdn_chunk` fixes 64). Returns (the shared kernel to register with the converter, record).
    The kernel's torch_defn returns zeros of the right shapes: in torch this form has shapes, not values.
    Round 16: `static_io` "slen" / "pad" puts the layers on the static-edge copy (qwen3_5_gdn_metal_static.py)."""
    from coreai_models.models.macos.qwen3_5_gdn_metal import MetalGDNChunk, build_gdn_chunk_kernel

    lins = [layer.linear_attn for layer in model.model.layers if not layer.is_full]
    la0 = lins[0]
    name = name or metal_kernel_name(chunk, static_io=static_io)
    geometry = dict(num_k=la0.num_k, num_v=la0.num_v, dk=la0.dk, dv=la0.dv, use_qk_l2_norm=la0.gdu.use_qk_l2_norm,
                    chunk_max=chunk)
    if static_io:
        from qwen3_5_gdn_metal_static import MetalGDNChunkStatic, build_gdn_chunk_kernel_static

        kernel = build_gdn_chunk_kernel_static(name=name, static_io=static_io, **geometry)
    else:
        kernel = build_gdn_chunk_kernel(name=name, **geometry)
    for la in lins:
        la.metal_chunk = (MetalGDNChunkStatic(kernel, chunk_max=chunk, static_io=static_io) if static_io
                          else MetalGDNChunk(kernel, chunk_max=chunk))
        la.use_metal_chunk = True
        la.use_loopfree_unroll = False   # the kernel branch comes first in the forward either way
    return kernel, metal_scan_record(len(lins), la0.num_v, la0.dv, chunk, name, static_io)


class Qwen3_5KevDecoder(Qwen3_5StatefulForCausalLM):
    """The overlay's stateful Qwen3.5 text decoder without a vocabulary head; contract in the header."""

    def _init_model(self, config) -> None:
        super()._init_model(config)
        del self.lm_head                      # the tied table is the embedding; no head in the graph
        self.lm_head = torch.nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,     # [1, S] int32
        position_ids: torch.Tensor,  # [1, seq] int32 cache ramp
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
        rec_state: torch.Tensor,
    ) -> torch.Tensor:
        """-> hidden [1, S, hidden]: the final-norm hidden state at every query position."""
        if self.last_token_only or self.emit_hidden is not None:
            raise ValueError("the Kev decoder returns the final-norm hidden at every position only")
        h = self.model.forward_stateful(
            input_ids, position_ids, KVCache(k_cache, v_cache), SSMState(conv_state), SSMState(rec_state))
        cap = getattr(self, "pad_output_to", None)
        if cap:   # round 14 fallback: a static [1, cap, hidden] output, rows s..cap-1 zero (the host reads [:s])
            h = torch.cat([h, h.new_zeros(1, cap - h.shape[1], h.shape[2])], dim=1)
        return h

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_hf(
        cls,
        hf_id: str,
        target_dtype: torch.dtype = torch.float16,
        max_context_length: int | None = 4096,
    ) -> "Qwen3_5KevDecoder":
        """Load a full-weight Qwen3.5 text checkpoint whose keys carry no prefix (`layers.N...`,
        `embed_tokens.weight`, `norm.weight`) and whose config.json is the flat text config.
        `load_report` records every checkpoint key against the module's tensors."""
        import glob
        import os

        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        model = cls.from_hf_memory_efficient(
            hf_id, max_context_length=max_context_length, target_dtype=target_dtype,
            hf_config_attr=None, hf_state_dict_prefix="")
        # The overlay loader re-ties `lm_head.weight` to the embedding; on the Identity that
        # registers a parameter alias. Drop it: this module has no head.
        model.lm_head = torch.nn.Identity()
        model.eval()

        model_dir = snapshot_download(
            hf_id, allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"])
        ckpt, files = set(), []
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            files.append(os.path.basename(path))
            with safe_open(path, framework="pt", device="cpu") as f:
                ckpt |= {"model." + k for k in f.keys()}  # noqa: SIM118
        own = set(model.state_dict(keep_vars=True).keys())
        model.load_report = {
            "checkpoint_files": files,
            "checkpoint_keys": len(ckpt),
            "module_tensors": len(own),
            "unread_checkpoint_keys": sorted(ckpt - own),
            "module_tensors_not_in_checkpoint": sorted(own - ckpt),
            "checkpoint_has_lm_head_key": "model.lm_head.weight" in ckpt,
            "tie_word_embeddings": bool(model.config.tie_word_embeddings),
            "module_has_lm_head_weight": any(k.startswith("lm_head.") for k in own),
            "meta_params": [n for n, p in model.named_parameters() if p.is_meta],
            "dtype": str(target_dtype).replace("torch.", ""),
        }
        return model

    # -- export -------------------------------------------------------------

    def build_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        trace_past: int = 64,
        query_len: int = PREFILL_CHUNK,
    ) -> dict:
        """Static-S entrypoint: static `input_ids` [1, S]; `position_ids` and the KV sequence dim
        stay dynamic, the conv / rec states are fixed-shape. Trace with every GDN layer on
        `use_loopfree_unroll` (`set_unrolled_scan`)."""
        if query_len < 2:
            raise ValueError("this graph has no S=1 function: query_len must be >= 2")
        cfg = self.config
        S = query_len
        if trace_past + S > trace_kv_len:
            raise ValueError(f"trace_past + S = {trace_past + S} exceeds trace_kv_len {trace_kv_len}")
        state = build_decode_state(cfg, max_seq_len=trace_kv_len, dtype=target_dtype)
        reference_inputs = {
            "input_ids": torch.randint(1, cfg.vocab_size, (1, S), dtype=torch.int32),
            "position_ids": torch.arange(trace_past + S, dtype=torch.int32).unsqueeze(0),
            "k_cache": state["k_cache"],
            "v_cache": state["v_cache"],
            "conv_state": state["conv_state"],
            "rec_state": state["rec_state"],
        }
        # The first chunk of a row runs at seq_len == S (offset 0), so the dim admits it.
        seq_pos = torch.export.Dim("seq_pos", min=max(2, S), max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        dynamic_shapes = {
            "input_ids": None,  # static [1, S]: no scan, no while_loop
            "position_ids": {1: seq_pos},
            "k_cache": {KVCache.seq_len_dim(): k_seq},
            "v_cache": {KVCache.seq_len_dim(): v_seq},
            "conv_state": None,
            "rec_state": None,
        }
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": INPUT_NAMES,
            "output_names": OUTPUT_NAMES,
            "state_names": DECODE_STATE_NAMES,
        }

    # Round 14: one function whose query length S is dynamic (QUERY_MIN..cap), for the Metal kernel scan only.
    def build_dynamic_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        cap: int,
        trace_query_len: int = 24,
        trace_past: int = 64,
    ) -> dict:
        """`input_ids` [1, S] with S a Dim(QUERY_MIN..cap) (a call carries any length in that range);
        `position_ids` [1, seq] (the cache ramp, seq = offset + S) and the KV sequence dim dynamic as before; the conv /
        rec states fixed-shape. The output is `hidden` [1, S, hidden] (or [1, cap, hidden] with `pad_output_to`). Trace
        with every GDN layer on the Metal kernel built with chunk_max = cap (`set_metal_scan(model, cap)`): the kernel
        reads S from its input's extent and writes rows [0:S] of a static [cap, h*dv] buffer the module slices.
        The trace length is chosen apart from every static extent of the graph (16 heads, 128, 3, 256 ...)."""
        if not (QUERY_MIN <= trace_query_len <= cap):
            raise ValueError(f"trace_query_len {trace_query_len} outside [{QUERY_MIN}, {cap}]")
        cfg = self.config
        S = trace_query_len
        if trace_past + S > trace_kv_len:
            raise ValueError(f"trace_past + S = {trace_past + S} exceeds trace_kv_len {trace_kv_len}")
        state = build_decode_state(cfg, max_seq_len=trace_kv_len, dtype=target_dtype)
        reference_inputs = {
            "input_ids": torch.randint(1, cfg.vocab_size, (1, S), dtype=torch.int32),
            "position_ids": torch.arange(trace_past + S, dtype=torch.int32).unsqueeze(0),
            "k_cache": state["k_cache"],
            "v_cache": state["v_cache"],
            "conv_state": state["conv_state"],
            "rec_state": state["rec_state"],
        }
        q_ids = torch.export.Dim("q_ids", min=QUERY_MIN, max=cap)
        seq_pos = torch.export.Dim("seq_pos", min=QUERY_MIN, max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        dynamic_shapes = {
            "input_ids": {1: q_ids},
            "position_ids": {1: seq_pos},
            "k_cache": {KVCache.seq_len_dim(): k_seq},
            "v_cache": {KVCache.seq_len_dim(): v_seq},
            "conv_state": None,
            "rec_state": None,
        }
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": INPUT_NAMES,
            "output_names": OUTPUT_NAMES,
            "state_names": DECODE_STATE_NAMES,
        }
