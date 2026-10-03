# clef-flash decoder: the Qwen3.5-9B hybrid backbone on the ids-input + static `image_embeds`
# contract, returning the final-norm hidden state at every position (no vocabulary head).
#
# Community port — NOT an Apple model.
#
# Cloudflare/clef-flash never generates text: its joint schema head reads the backbone's
# `last_hidden_state` (after the final RMSNorm) at every position of one prefill. So this module is
# decider-2b-vision's `Qwen3_5VLPipelinedForCausalLM` (conversion/decider_vision/
# qwen3_5_vl_pipelined.py: ids input, image rows on a static input, interleaved M-RoPE derived
# in-graph from (ids, position)) with four changes and nothing else:
#
#   (a) no `lm_head`: the untied [248320, 4096] table (2 GB at fp16) stays on the host, where the
#       head's lexical anchor gathers option-token rows from it. The parent's forward ends in
#       `self.lm_head(h)`; here that slot is the identity, so the forward runs the parent's code
#       verbatim and returns h. `n_image_max` defaults to 1024 rows (any merged grid up to 32x32).
#   (b) output = the final-norm hidden at EVERY query position, [1, S, hidden] (`last_token_only`
#       is refused: the head needs all of them).
#   (c) `from_hf` records `lm_head.weight` as a checkpoint key this module reads on purpose not
#       at all (`intentionally_unread_keys`), so `unread_checkpoint_keys` stays a real check.
#   (d) `build_export_spec`: output name `hidden`, `image_embeds [n_image_max, hidden]`, S = 16
#       by default (one static-S "prefill" function; there is no S=1 decode function).
#
# The M-RoPE derivation (`_rope_planes`), the embedding / image-row gather, the four states and
# the dynamic dims are the parent's. Host side: `host_static_inputs` (re-exported) with
# n_image_max = 1024, the image block right after the 36-token prefix (i0 = 36,
# rope_shift_start = 37 + N, rope_shift_amount = N - max(H, W)).
#
# Graph (static S; states mutate in place):
#
#   inputs  input_ids          [1, S]       int32  image token k (row-major) = V + k
#           position_ids       [1, seq]     int32  cache ramp 0..seq-1 (offset = seq - S)
#           image_embeds       [1024, h]    float  tower rows 0..N-1, rows N.. zero
#           image_rc           [1024, 2]    int32  (row, col) of slot k on the merged grid
#           rope_shift_start   [1]          int32  index of <|vision_end|> (text: 1 << 30)
#           rope_shift_amount  [1]          int32  N - max(H, W) (text: 0)
#   states  keyCache / valueCache [8, 1, 4, ctx, 256], convState [24, 1, 8192, 3],
#           recState [24, 1, 32, 128, 128]
#   output  hidden             [1, S, h]    final-norm hidden at every position
#
# A prompt of T tokens runs as ceil(T / S) chunks from fresh zero states; the last chunk is
# padded with <|endoftext|> (248044) and the padded positions' outputs are discarded (causal,
# so they cannot reach a real position). Every GDN layer runs `use_loopfree_unroll` (the S
# single steps unrolled in-graph); set it on every linear-attention layer before tracing.
from __future__ import annotations

import sys
from pathlib import Path

import torch

_DECIDER_VISION = Path(__file__).resolve().parents[1] / "decider_vision"
if str(_DECIDER_VISION) not in sys.path:
    sys.path.insert(0, str(_DECIDER_VISION))

from qwen3_5_vl_pipelined import (  # noqa: E402
    INPUT_NAMES,
    TEXT_ONLY_SHIFT_START,
    Qwen3_5VLPipelinedForCausalLM,
    host_static_inputs,
)

N_IMAGE_MAX = 1024
PREFILL_CHUNK = 16
OUTPUT_NAMES = ("hidden",)
INTENTIONALLY_UNREAD = ("lm_head.weight",)

__all__ = [
    "INPUT_NAMES", "INTENTIONALLY_UNREAD", "N_IMAGE_MAX", "OUTPUT_NAMES", "PREFILL_CHUNK",
    "Qwen3_5ClefDecoder", "TEXT_ONLY_SHIFT_START", "host_static_inputs",
]


class Qwen3_5ClefDecoder(Qwen3_5VLPipelinedForCausalLM):
    """`Qwen3_5VLPipelinedForCausalLM` without a vocabulary head; contract in the header."""

    def _init_model(self, config) -> None:
        super()._init_model(config)
        del self.lm_head                      # the untied table stays on the host
        self.lm_head = torch.nn.Identity()    # the parent's final call returns h unchanged
        self.n_image_max = N_IMAGE_MAX

    def forward(
        self,
        input_ids: torch.Tensor,          # [1, S] int32; image tokens = V + slot
        position_ids: torch.Tensor,       # [1, seq] int32 cache ramp
        image_embeds: torch.Tensor,       # [n_image_max, h]
        image_rc: torch.Tensor,           # [n_image_max, 2] int32 (row, col) per slot
        rope_shift_start: torch.Tensor,   # [1] int32
        rope_shift_amount: torch.Tensor,  # [1] int32
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
        rec_state: torch.Tensor,
    ) -> torch.Tensor:
        """-> hidden [1, S, hidden]: the final-norm hidden state at every query position."""
        if self.last_token_only:
            raise ValueError("the clef decoder returns every position; last_token_only is not used")
        return super().forward(input_ids, position_ids, image_embeds, image_rc, rope_shift_start,
                               rope_shift_amount, k_cache, v_cache, conv_state, rec_state)

    @classmethod
    def from_hf(
        cls,
        hf_id: str,
        target_dtype: torch.dtype = torch.float16,
        max_context_length: int | None = 4096,
        n_image_max: int = N_IMAGE_MAX,
    ) -> "Qwen3_5ClefDecoder":
        """The parent's loader (`from_hf_memory_efficient(..., hf_config_attr="text_config")` +
        `load_report`), with `lm_head.weight` moved from `unread_checkpoint_keys` to
        `intentionally_unread_keys` and the untied head's checkpoint entry recorded."""
        import glob
        import json
        import os

        from huggingface_hub import snapshot_download

        model = super().from_hf(hf_id, target_dtype=target_dtype,
                                max_context_length=max_context_length, n_image_max=n_image_max)
        rep = model.load_report
        unread = [k for k in rep["unread_checkpoint_keys"] if k not in INTENTIONALLY_UNREAD]
        skipped = [k for k in rep["unread_checkpoint_keys"] if k in INTENTIONALLY_UNREAD]
        model_dir = snapshot_download(
            hf_id, allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"])
        head_entry = None
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with open(path, "rb") as f:           # header only: the table itself is not read
                n = int.from_bytes(f.read(8), "little")
                header = json.loads(f.read(n))
            if "lm_head.weight" in header:
                head_entry = {"file": os.path.basename(path), "dtype": header["lm_head.weight"]["dtype"],
                              "shape": header["lm_head.weight"]["shape"]}
        rep.update(
            unread_checkpoint_keys=unread,
            intentionally_unread_keys=skipped,
            lm_head_untied=not model.config.tie_word_embeddings,
            checkpoint_lm_head=head_entry,
            module_has_lm_head_weight=any(k.startswith("lm_head.") for k in model.state_dict()),
        )
        return model

    def build_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        trace_past: int = 64,
        query_len: int = PREFILL_CHUNK,
    ) -> dict:
        """The parent's static-S spec (static ids [1, S] and image inputs, dynamic position and
        KV sequence dims) with the `hidden` output; `last_token_only` stays off."""
        spec = super().build_export_spec(target_dtype, max_context_length, trace_kv_len,
                                         trace_past=trace_past, query_len=query_len)
        spec["output_names"] = OUTPUT_NAMES
        return spec
