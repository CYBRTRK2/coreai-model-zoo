# Community port — NOT an Apple model.
"""Fun-ASR-Nano audio path — SAN-M encoder + adaptor — re-authored in plain torch, fixed shape.

Weights come straight from ``FunAudioLLM/Fun-ASR-Nano-2512-vllm`` ``model.safetensors`` (BF16,
bit-identical to the official ``model.pt``). The math follows funasr 1.4.16:

``SenseVoiceEncoderSmall`` (``funasr/models/sense_voice/model.py``)
    x = feats * sqrt(512) + PE, PE = SinusoidalPositionEncoder: positions 1..L, depth 560,
        cat[sin, cos] with inv_timescale_k = exp(-k ln(10000) / (280 - 1))
    encoders0 (1 layer, 560 -> 512, no attention residual) -> encoders (49) -> after_norm
    -> tp_encoders (20) -> tp_norm. Every layer is pre-LN (eps 1e-5):
        h = LN1(x); q, k, v = split(linear_q_k_v(h), 512)
        fsmn = (depthwise_conv11(pad5(v * m)) + v * m) * m          # m = valid-frame mask
        attn = linear_out(softmax(q k^T / sqrt(128) + key_mask) v)  # 4 heads x 128
        x = x + attn + fsmn        (encoders0: x = attn + fsmn)
        x = x + w_2(relu(w_1(LN2(x))))
``Transformer`` adaptor (``funasr/models/llm_asr/adaptor.py``, downsample_rate 1)
    linear1 (512 -> 2048) -> ReLU -> linear2 (2048 -> 1024) -> 2 x EncoderLayer:
        pre-LN (``funasr.models.transformer.layer_norm.LayerNorm``, eps 1e-12),
        8-head MHA with separate q/k/v/out projections, FFN 1024 -> 256 -> 1024 (ReLU)

Static-shape contract for export: ``feats [1, 500, 560]`` (LFR frames zero-padded to 30 s) and
``mask [1, 500]`` (1 valid / 0 pad) -> ``audio_embeds [63, 1024]``. The LLM consumes only the first
``N = fake_token_len(L)`` adaptor rows (``N <= 63`` for L <= 500); the host trims to N.

Padding: the valid rows of the padded run equal the unpadded run because pad rows only reach
valid rows through the FSMN input (zeroed by ``v * m``) and attention keys (masked). Pad rows are
zeroed after every layer so they stay finite in fp16 (a 0 * inf in either path would be NaN).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

D_IN = 560
D = 512
HEADS = 4
D_K = 128
FFN = 2048
KERNEL = 11
N_ENCODERS = 49
N_TP = 20
ENC_EPS = 1e-5

D_LLM = 1024
A_HIDDEN = 2048
A_HEADS = 8
A_D_K = 128
A_FFN = 256
N_BLOCKS = 2
ADAPTOR_EPS = 1e-12

L_MAX = 500          # LFR frames in 30 s (480000 samples -> 2998 fbank frames -> 500)
N_MAX = 63           # fake_token_len(500)
MASK_NEG = 65504.0   # largest finite fp16; additive key mask


def fake_token_len(num_lfr: int) -> int:
    olens = 1 + (num_lfr - 3 + 2) // 2
    olens = 1 + (olens - 3 + 2) // 2
    return (olens - 1) // 2 + 1


def sinusoid_pe(length: int, depth: int = D_IN, first_position: int = 1) -> torch.Tensor:
    """``SinusoidalPositionEncoder.encode`` evaluated in fp32, as the fp32 oracle does: ``[1, L, depth]``."""
    positions = torch.arange(first_position, first_position + length)[None, :].type(torch.float32)
    increment = torch.log(torch.tensor([10000], dtype=torch.float32)) / (depth / 2 - 1)
    inv_timescales = torch.exp(torch.arange(depth / 2).type(torch.float32) * (-increment))
    inv_timescales = torch.reshape(inv_timescales, [1, -1])
    scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(inv_timescales, [1, 1, -1])
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)


class _Norm(nn.Module):
    """LayerNorm; ``fp32=True`` computes it in fp32 like funasr's SAN-M ``LayerNorm`` does."""

    def __init__(self, dim: int, eps: float, fp32: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.fp32 = fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp32 and x.dtype != torch.float32:
            y = F.layer_norm(x.float(), (x.shape[-1],), self.weight.float(), self.bias.float(), self.eps)
            return y.to(x.dtype)
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class _FFN(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w_1 = nn.Linear(dim, hidden)
        self.w_2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(torch.relu(self.w_1(x)))


class _SANMAttention(nn.Module):
    def __init__(self, in_dim: int, fsmn: str) -> None:
        super().__init__()
        self.linear_q_k_v = nn.Linear(in_dim, 3 * D)
        self.linear_out = nn.Linear(D, D)
        self.fsmn_block = nn.Conv1d(D, D, KERNEL, groups=D, bias=False)
        self.fsmn = fsmn

    def _fsmn(self, v: torch.Tensor) -> torch.Tensor:
        pad = (KERNEL - 1) // 2
        if self.fsmn == "conv":
            y = self.fsmn_block(F.pad(v.transpose(1, 2), (pad, pad))).transpose(1, 2)
        else:  # exact shift-accumulate: y[t] = sum_j w[:, j] * v_pad[t + j]
            t = v.shape[1]
            vp = F.pad(v, (0, 0, pad, pad))
            w = self.fsmn_block.weight[:, 0, :]                 # [512, 11]
            y = vp[:, 0:t, :] * w[:, 0]
            for j in range(1, KERNEL):
                y = y + vp[:, j:j + t, :] * w[:, j]
        return y

    def forward(self, x: torch.Tensor, m: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = torch.split(self.linear_q_k_v(x), D, dim=-1)
        vm = v * m
        fsmn = (self._fsmn(vm) + vm) * m
        qh = q.reshape(b, t, HEADS, D_K).transpose(1, 2) * (D_K ** -0.5)
        kh = k.reshape(b, t, HEADS, D_K).transpose(1, 2)
        vh = v.reshape(b, t, HEADS, D_K).transpose(1, 2)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) + key_bias
        ctx = torch.matmul(torch.softmax(scores, dim=-1), vh)
        attn = self.linear_out(ctx.transpose(1, 2).reshape(b, t, D))
        return attn + fsmn


class _SANMLayer(nn.Module):
    def __init__(self, in_dim: int, fsmn: str, ln_fp32: bool) -> None:
        super().__init__()
        self.norm1 = _Norm(in_dim, ENC_EPS, ln_fp32)
        self.self_attn = _SANMAttention(in_dim, fsmn)
        self.norm2 = _Norm(D, ENC_EPS, ln_fp32)
        self.feed_forward = _FFN(D, FFN)
        self.residual = in_dim == D

    def forward(self, x: torch.Tensor, m: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
        a = self.self_attn(self.norm1(x), m, key_bias)
        x = x + a if self.residual else a
        x = x + self.feed_forward(self.norm2(x))
        return x * m


class _AdaptorAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_q = nn.Linear(D_LLM, D_LLM)
        self.linear_k = nn.Linear(D_LLM, D_LLM)
        self.linear_v = nn.Linear(D_LLM, D_LLM)
        self.linear_out = nn.Linear(D_LLM, D_LLM)

    def forward(self, x: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.linear_q(x).reshape(b, t, A_HEADS, A_D_K).transpose(1, 2)
        k = self.linear_k(x).reshape(b, t, A_HEADS, A_D_K).transpose(1, 2)
        v = self.linear_v(x).reshape(b, t, A_HEADS, A_D_K).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(A_D_K) + key_bias
        ctx = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self.linear_out(ctx.transpose(1, 2).reshape(b, t, D_LLM))


class _AdaptorBlock(nn.Module):
    def __init__(self, ln_fp32: bool) -> None:
        super().__init__()
        self.norm1 = _Norm(D_LLM, ADAPTOR_EPS, ln_fp32)
        self.self_attn = _AdaptorAttention()
        self.norm2 = _Norm(D_LLM, ADAPTOR_EPS, ln_fp32)
        self.feed_forward = _FFN(D_LLM, A_FFN)

    def forward(self, x: torch.Tensor, m: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), key_bias)
        x = x + self.feed_forward(self.norm2(x))
        return x * m


class _Encoder(nn.Module):
    def __init__(self, fsmn: str, ln_fp32: bool) -> None:
        super().__init__()
        self.encoders0 = nn.ModuleList([_SANMLayer(D_IN, fsmn, ln_fp32)])
        self.encoders = nn.ModuleList([_SANMLayer(D, fsmn, ln_fp32) for _ in range(N_ENCODERS)])
        self.after_norm = _Norm(D, ENC_EPS, ln_fp32)
        self.tp_encoders = nn.ModuleList([_SANMLayer(D, fsmn, ln_fp32) for _ in range(N_TP)])
        self.tp_norm = _Norm(D, ENC_EPS, ln_fp32)


class _Adaptor(nn.Module):
    def __init__(self, ln_fp32: bool) -> None:
        super().__init__()
        self.linear1 = nn.Linear(D, A_HIDDEN)
        self.linear2 = nn.Linear(A_HIDDEN, D_LLM)
        self.blocks = nn.ModuleList([_AdaptorBlock(ln_fp32) for _ in range(N_BLOCKS)])


class FunASRNanoAudioEncoder(nn.Module):
    """``feats [1, L, 560]`` + ``mask [1, L]`` -> ``audio_embeds [n_out, 1024]`` (first rows of the adaptor)."""

    def __init__(self, length: int = L_MAX, n_out: int = N_MAX, fsmn: str = "conv",
                 ln_fp32: bool = False, pe_first_position: int = 1,
                 return_encoder_out: bool = False) -> None:
        super().__init__()
        self.length = length
        self.n_out = n_out
        self.return_encoder_out = return_encoder_out
        self.audio_encoder = _Encoder(fsmn, ln_fp32)
        self.audio_adaptor = _Adaptor(ln_fp32)
        # fp32 table, baked as a plain buffer: no in-graph trig on large arguments.
        self.register_buffer("pe", sinusoid_pe(length, D_IN, pe_first_position), persistent=False)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor):
        m = mask[:, :, None]                                     # [1, L, 1]
        key_bias = ((mask - 1.0) * MASK_NEG)[:, None, None, :]   # [1, 1, 1, L]
        x = feats * math.sqrt(D) + self.pe.to(feats.dtype)
        enc = self.audio_encoder
        for layer in enc.encoders0:
            x = layer(x, m, key_bias)
        for layer in enc.encoders:
            x = layer(x, m, key_bias)
        x = enc.after_norm(x)
        for layer in enc.tp_encoders:
            x = layer(x, m, key_bias)
        encoder_out = enc.tp_norm(x) * m
        ad = self.audio_adaptor
        y = ad.linear2(torch.relu(ad.linear1(encoder_out))) * m
        for block in ad.blocks:
            y = block(y, m, key_bias)
        audio_embeds = y[0, : self.n_out, :]
        if self.return_encoder_out:
            return audio_embeds, encoder_out[0], y[0]
        return audio_embeds


def load_weights(model: FunASRNanoAudioEncoder, safetensors_path: str | Path,
                 dtype: torch.dtype = torch.float32) -> FunASRNanoAudioEncoder:
    """Copy ``audio_encoder.*`` / ``audio_adaptor.*`` from the checkpoint; every key must match."""
    from safetensors import safe_open

    state = model.state_dict()
    wanted = {k for k in state if k.startswith(("audio_encoder.", "audio_adaptor."))}
    loaded = set()
    with safe_open(str(safetensors_path), framework="pt") as f:
        for name in f.keys():
            if not name.startswith(("audio_encoder.", "audio_adaptor.")):
                continue
            if name not in state:
                raise KeyError(f"checkpoint tensor {name} has no slot in the re-authored module")
            t = f.get_tensor(name)
            if tuple(t.shape) != tuple(state[name].shape):
                raise ValueError(f"{name}: checkpoint {tuple(t.shape)} vs module {tuple(state[name].shape)}")
            state[name] = t.to(torch.float32)
            loaded.add(name)
    missing = wanted - loaded
    if missing:
        raise KeyError(f"{len(missing)} module tensors not in the checkpoint, e.g. {sorted(missing)[:5]}")
    model.load_state_dict(state, strict=True)
    return model.to(dtype).eval()
