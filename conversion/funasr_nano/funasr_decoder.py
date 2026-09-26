# Community port — NOT an Apple model.
"""Fun-ASR-Nano text decoder (Qwen3-0.6B) for the Core AI pipelined engine.

Copied from ``conversion/qwen3_asr/qwen3_asr_decoder.py`` (``Qwen3ASRDecoderPipelined``): the zoo
overlay ``Qwen3Model`` blocks, the ``V + slot`` audio injection and the export spec are unchanged.
What differs for Fun-ASR-Nano-2512:

- config = the official ``Qwen3-0.6B/config.json`` inside ``FunAudioLLM/Fun-ASR-Nano-2512`` — plain
  Qwen3 (28 layers, hidden 1024, 16 / 8 heads, head_dim 128, rope_theta 1e6, no rope_scaling, tied
  head, vocab 151936), so no RoPE rewrite is needed;
- weights = the checkpoint's ``llm.model.*``. ``llm.lm_head.weight`` is stored too but is
  bit-identical to ``llm.model.embed_tokens.weight`` (asserted at load), so the head stays tied;
- ``audio_embeds [63, 1024]`` = the audio encoder bundle's output; the first
  ``N = fake_token_len(L) <= 63`` rows are the audio slots (``prompt.py`` has the id contract).

Residual scale (``apply_residual_scale``, s = 1/4 in every exported bundle). This checkpoint's
Qwen3 puts a massive activation on the first token (``<|im_start|>``, channel 35): layer 2's
``mlp.down_proj`` writes 125,074 there and the residual stream stays at 125,052-125,719 through layer
27 — past fp16's 65,504, so an fp16 graph turns it into inf and every logit into NaN. Every other
position stays under ~8k. The fix is an exact reparametrization: multiply the
embedding output (text and audio rows) by s, fold s into every ``o_proj`` and ``down_proj`` weight,
and scale the eps of every residual-stream RMSNorm (input / post-attention / final) by s**2. Each
RMSNorm then sees ``s * x`` with ``s**2 * eps`` and returns exactly what it returned for ``x``; the
q/k norms and the tied head are untouched, so the logits are unchanged (bit-identical in fp32 for
s = 1/4) while the residual peak drops to ~31.4k.

Bundle: ``input_ids, position_ids, audio_embeds, keyCache, valueCache -> logits``.

``python funasr_decoder.py --selftest`` (shared venv, eager, CPU; fp32 unscaled, fp32 s=1/4, fp16 s=1/4): on
the five model-repo examples the first generated token's argmax must equal the oracle's, and a full
greedy decode through the KV cache should reproduce the oracle's ``gen_ids``.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from coreai_models.models.macos.qwen3 import Qwen3Model
from coreai_models.primitives.macos.cache import KVCache

PIPELINED_STATE_NAMES = ("keyCache", "valueCache")
N_AUDIO_MAX = 63   # fake_token_len(500 LFR frames) = 30 s of audio


def qwen3_config(config_json: str | Path) -> Qwen3Config:
    d = json.loads(Path(config_json).read_text())
    for k in ("torch_dtype", "dtype", "architectures", "transformers_version"):
        d.pop(k, None)
    assert d.get("rope_scaling") is None and d["model_type"] == "qwen3", d
    return Qwen3Config(**d)


class _ScaledEmbedding(nn.Module):
    """``embed(ids) * scale`` — the residual stream starts scaled; the table itself (tied head) is untouched."""

    def __init__(self, embedding: nn.Embedding, scale: float) -> None:
        super().__init__()
        self.emb = embedding
        self.scale = float(scale)

    @property
    def weight(self) -> torch.Tensor:
        return self.emb.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids) * self.scale


class FunASRNanoDecoderPipelined(nn.Module):
    """Engine-shaped Fun-ASR-Nano decoder; audio embeds injected via ``V + slot`` ids."""

    coreai_externalize_specs: tuple = ()

    def __init__(self, config: Qwen3Config, n_audio_tokens: int = N_AUDIO_MAX) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.n_audio_tokens = int(n_audio_tokens)
        self.residual_scale = 1.0

    def apply_residual_scale(self, scale: float) -> "FunASRNanoDecoderPipelined":
        """Run the residual stream at ``scale`` times its size (module docstring). Call once, before quantizing."""
        assert self.residual_scale == 1.0, "residual scale already applied"
        with torch.no_grad():
            for layer in self.model.layers:
                layer.self_attn.o_proj.weight.mul_(scale)
                layer.mlp.down_proj.weight.mul_(scale)
                layer.input_layernorm.rmsnorm_impl.eps *= scale * scale
                layer.post_attention_layernorm.rmsnorm_impl.eps *= scale * scale
            self.model.norm.rmsnorm_impl.eps *= scale * scale
        self.model.embed_tokens = _ScaledEmbedding(self.model.embed_tokens, scale)
        self.residual_scale = float(scale)
        return self

    def forward(
        self,
        input_ids: torch.Tensor,     # [1, s] int32; audio tokens = V + slot
        position_ids: torch.Tensor,  # [1, total] int32 sequential ramp from 0
        audio_embeds: torch.Tensor,  # [N, h] static input (audio encoder output)
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model
        V = self.config.vocab_size
        N = self.n_audio_tokens
        b, s = input_ids.shape

        ids = input_ids
        is_aud = ids >= V
        slot = (ids - V).clamp(0, N - 1)
        e_txt = m.embed_tokens(ids.clamp(0, V - 1))
        e_aud = audio_embeds.index_select(0, slot.reshape(-1)).reshape(b, s, -1)
        if self.residual_scale != 1.0:
            e_aud = e_aud * self.residual_scale       # text rows are scaled by _ScaledEmbedding
        x = torch.where(is_aud.unsqueeze(-1), e_aud.to(e_txt.dtype), e_txt)

        cache = KVCache(k_cache, v_cache)
        for layer in m.layers:
            x = layer(x, position_ids, cache)
        return self.lm_head(m.norm(x))

    # -- loading ------------------------------------------------------------
    @classmethod
    def from_safetensors(cls, path: str | Path, config_json: str | Path, n_audio_tokens: int = N_AUDIO_MAX,
                         target_dtype: torch.dtype = torch.float16,
                         residual_scale: float = 1.0) -> "FunASRNanoDecoderPipelined":
        from safetensors import safe_open

        cfg = qwen3_config(config_json)
        model = cls(cfg, n_audio_tokens).to(dtype=target_dtype)

        sd = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in f.keys():  # noqa: SIM118
                if k.startswith("llm.model.") or k == "llm.lm_head.weight":
                    sd[k[len("llm."):]] = f.get_tensor(k).to(target_dtype)
        head = sd.pop("lm_head.weight")
        if not torch.equal(head, sd["model.embed_tokens.weight"]):
            raise RuntimeError("llm.lm_head.weight differs from llm.model.embed_tokens.weight; the tied "
                               "decoder would be wrong for this checkpoint")

        # remap to the zoo qwen3 tree: fuse qkv + qk_norm (USE_FUSED_KV recipe)
        out = {}
        hd, nh, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        for i in range(cfg.num_hidden_layers):
            pre = f"model.layers.{i}.self_attn."
            qw = sd.pop(pre + "q_proj.weight"); kw = sd.pop(pre + "k_proj.weight"); vw = sd.pop(pre + "v_proj.weight")
            out[pre + "qkv_proj.weight"] = torch.cat([qw, kw, vw], dim=0)
            qn = sd.pop(pre + "q_norm.weight"); kn = sd.pop(pre + "k_norm.weight")
            out[pre + "qk_norm.weight"] = torch.cat(
                [qn.view(1, 1, hd).expand(nh, 1, hd), kn.view(1, 1, hd).expand(nkv, 1, hd)], dim=0).contiguous()
        for k, v in sd.items():
            out[k] = v  # model.layers.*.{o_proj,mlp.*,*layernorm}, model.embed_tokens, model.norm

        missing, unexpected = model.load_state_dict(out, strict=False, assign=True)
        missing = [k for k in missing if k != "lm_head.weight"]
        if missing or unexpected:
            raise RuntimeError(f"load mismatch: missing={missing} unexpected={unexpected}")
        if cfg.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
        if residual_scale != 1.0:
            model.apply_residual_scale(residual_scale)
        return model.eval()

    def build_export_spec(self, target_dtype: torch.dtype, max_context_length: int,
                          trace_kv_len: int, trace_query: int = 8, trace_past: int = 64) -> dict:
        cfg = self.config
        N, h = self.n_audio_tokens, cfg.hidden_size
        input_ids = torch.randint(1, cfg.vocab_size, (1, trace_query), dtype=torch.int32)
        position_ids = torch.arange(trace_past + trace_query, dtype=torch.int32).unsqueeze(0)
        k_cache = torch.zeros(cfg.num_hidden_layers, 1, cfg.num_key_value_heads,
                              trace_kv_len, cfg.head_dim, dtype=target_dtype)
        v_cache = torch.zeros_like(k_cache)
        reference_inputs = {
            "input_ids": input_ids, "position_ids": position_ids,
            "audio_embeds": torch.zeros(N, h, dtype=target_dtype),
            "k_cache": k_cache, "v_cache": v_cache,
        }
        seq_pos = torch.export.Dim("seq_pos", min=2, max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        ids_shape = None if trace_query == 1 else {1: torch.export.Dim("seq_ids", min=1, max=max_context_length - 2)}
        dynamic_shapes = {
            "input_ids": ids_shape, "position_ids": {1: seq_pos}, "audio_embeds": None,
            "k_cache": {KVCache.seq_len_dim(): k_seq}, "v_cache": {KVCache.seq_len_dim(): v_seq},
        }
        return {
            "reference_inputs": reference_inputs, "dynamic_shapes": dynamic_shapes,
            "input_names": ("input_ids", "position_ids", "audio_embeds"),
            "output_names": ("logits",), "state_names": PIPELINED_STATE_NAMES,
        }


def prompt_tensors(clip: str, tokenizer, work: Path, dtype: torch.dtype, n_audio: int = N_AUDIO_MAX):
    """Oracle tensors for one fixture clip: ``(ids [1,Sp] int32, audio [n_audio,h], N, golden gen_ids)``."""
    import numpy as np

    from prompt import build_prompt_ids

    o = np.load(work / "oracle" / f"{clip}.npz")
    N = int(o["fake_token_len"])
    ids, _ = build_prompt_ids(tokenizer, N)
    audio = torch.zeros(n_audio, o["adaptor_out"].shape[1], dtype=dtype)
    audio[:N] = torch.from_numpy(o["adaptor_out"][:N]).to(dtype)
    return (torch.tensor([ids], dtype=torch.int32), audio, N, o["gen_ids"].astype(np.int64).tolist())


@torch.no_grad()
def greedy(model: FunASRNanoDecoderPipelined, ids: torch.Tensor, audio: torch.Tensor, max_new: int,
           eos: tuple[int, ...], cache_len: int = 1024) -> tuple[list[int], torch.Tensor]:
    """Eager greedy decode through the KV cache: prefill the prompt, then one token per call."""
    cfg = model.config
    dtype = audio.dtype
    kc = torch.zeros(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, cache_len, cfg.head_dim, dtype=dtype)
    vc = torch.zeros_like(kc)
    sp = ids.shape[1]
    logits = model(ids, torch.arange(sp, dtype=torch.int32)[None], audio, kc, vc)[0, -1].float()
    first = logits.clone()
    gen = []
    for p in range(sp, sp + max_new):
        nxt = int(logits.argmax())
        gen.append(nxt)
        if nxt in eos:
            break
        logits = model(torch.tensor([[nxt]], dtype=torch.int32), torch.arange(p + 1, dtype=torch.int32)[None],
                       audio, kc, vc)[0, -1].float()
    return gen, first


def _selftest() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _paths import work_path
    from prompt import EOS_IDS
    from transformers import AutoTokenizer

    work = work_path("_funasr_nano")
    qwen_dir = work / "official" / "Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(str(qwen_dir))
    ok = True
    for scale, dtype in ((1.0, torch.float32), (0.25, torch.float32), (0.25, torch.float16)):
        model = FunASRNanoDecoderPipelined.from_safetensors(work / "hf" / "model.safetensors", qwen_dir / "config.json",
                                                            target_dtype=dtype, residual_scale=scale)
        label = f"{str(dtype).split('.')[-1]} s={scale}"
        first_ok = full_ok = 0
        for clip in ("zh", "en", "ja", "ko", "yue"):
            ids, audio, N, golden = prompt_tensors(clip, tok, work, dtype)
            gen, first = greedy(model, ids, audio, len(golden) + 8, EOS_IDS)
            top2 = torch.topk(torch.softmax(first, -1), 2)
            am = int(first.argmax())
            first_ok += am == golden[0]
            full_ok += gen == golden
            print(f"[{label} eager] {clip}: N={N} Sp={ids.shape[1]} first argmax {am} golden {golden[0]} "
                  f"{'OK' if am == golden[0] else 'MISMATCH'} (top-2 gap {float(top2.values[0] - top2.values[1]):.4f}); "
                  f"greedy {len(gen)} tokens {'== oracle' if gen == golden else '!= oracle'}", flush=True)
        print(f"[{label}] first-token argmax {first_ok}/5, full greedy {full_ok}/5", flush=True)
        ok = ok and first_ok == 5
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        _selftest()
