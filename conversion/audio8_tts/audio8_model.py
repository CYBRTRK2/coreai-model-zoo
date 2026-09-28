# Community port — NOT an Apple model.
"""Audio8-TTS-Preview-0.6b re-authored in plain torch for Core AI: slow AR, fast AR, codec decoder.

Weights come straight from the checkpoint (`model.safetensors`, bf16, 226 tensors; `codec.pth`, fp32),
not from the publisher's `nn.Module`s — except that the codec is *instantiated* from the publisher's
class once, to fold its weight-norm parametrizations and copy tensors by object traversal instead of by
string key. Every graph is static-shape with in-graph Core AI state where the model has a cache.

Three graphs (four with the optional codec encoder in `audio8_encoder.py`):

  slow AR   `prefill(codes [1,11,S], pos [1])` and `decode(codes [1,11,1], pos [1])`, one KV state pair
            `k_cache / v_cache [24, 1, 2, CL, 64]`, output `logits [S, 4097]` (semantic 4096 + eos, the only
            rows the sampler ever reads) and `hidden [S, 896]` (the post-norm slow hidden the fast AR takes).
  fast AR   `first(hidden [1,1,896], code0 [1])` (rows 0 and 1: the slow hidden, then codebook 0) and
            `step(code [1], pos [1])` (rows 2..9), one KV pair `[4, 1, 2, 10, 64]`, output `logits [1, 4096]`.
  codec     `decode(codes [1,10,T])` -> `wav [1, T*2048]` at 44.1 kHz, stateless; everything in it is causal,
            so a right-padded window decodes its real frames exactly.

Numerics kept from the publisher's code, because the fp32 oracle has them:
  * RoPE tables are the publisher's `_precompute_rope` rounded to **bfloat16** (slow/fast base 1e6,
    codec base 1e4), then held as fp32 constants. Interleaved-pair rotation (GPT-J style), not rotate-half.
  * RMSNorm computes in fp32 with eps inside the rsqrt (slow/fast 1e-6, codec 1e-5); ConvNeXt LayerNorm 1e-6.
  * The semantic mask on the embedding sum is float arithmetic on the id (exact on integers), never an
    int64 comparison chain (conversion-guide: those clobber live buffers on the GPU delegate).
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from coreai_models.primitives._ops import mutable_slice_update

# ----------------------------------------------------------------------------- constants (config.json)
DIM = 896
N_HEAD = 14
N_KV = 2
HEAD_DIM = 64
FFN = 4864
N_LAYER = 24
N_FAST_LAYER = 4
VOCAB = 155776
CODEBOOK_SIZE = 4096
NUM_CODEBOOKS = 10
SEMANTIC_BEGIN = 151678
SEMANTIC_END = 155773
EOS = 151645
PAD = 151643
ROPE_BASE = 1_000_000.0
NORM_EPS = 1e-6
MAX_SEQ_LEN = 2048
N_ALLOWED = SEMANTIC_END - SEMANTIC_BEGIN + 1 + 1     # 4097: semantic ids then eos
CODEC_SR = 44100
CODEC_FRAME = 2048
CODEC_DIM = 1024
CODEC_POST_LAYERS = 8
CODEC_POST_HEADS = 16
CODEC_POST_KV = 8
CODEC_POST_FFN = 1216
CODEC_WINDOW = 128
CODEC_ROPE_BASE = 10_000.0
CODEC_NORM_EPS = 1e-5


def allowed_ids() -> torch.Tensor:
    return torch.cat([torch.arange(SEMANTIC_BEGIN, SEMANTIC_END + 1), torch.tensor([EOS])])


def rope_table(length: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The publisher's `_precompute_rope`, rounded to bf16 as the checkpoint code does, returned as fp32
    cos / sin `[length, head_dim // 2]`."""
    frequencies = 1.0 / (base ** (torch.arange(0, head_dim, 2).float()[: head_dim // 2] / head_dim))
    phases = torch.outer(torch.arange(length), frequencies)
    values = torch.polar(torch.ones_like(phases), phases)
    packed = torch.stack((values.real, values.imag), dim=-1).to(torch.bfloat16).float()
    return packed[..., 0].contiguous(), packed[..., 1].contiguous()


def apply_rope_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, S, H, D] with cos/sin [S, D/2] (or [1, S, 1, D/2]): interleaved-pair rotation in fp32."""
    dt = x.dtype
    b, s, h, d = x.shape
    xr = x.float().reshape(b, s, h, d // 2, 2)
    re, im = xr[..., 0], xr[..., 1]
    if cos.dim() == 2:
        cos = cos.reshape(1, s, 1, d // 2)
        sin = sin.reshape(1, s, 1, d // 2)
    out = torch.stack((re * cos - im * sin, im * cos + re * sin), dim=-1)
    return out.reshape(b, s, h, d).to(dt)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def write_kv(cache: torch.Tensor, layer: int, start: torch.Tensor, x: torch.Tensor) -> None:
    """cache[layer, :, :, start:start+S] = x  with x [1, H_kv, S, D] and `start` a runtime int32 [1]."""
    dev = cache.device
    s = x.shape[2]
    li = torch.tensor([layer], dtype=torch.int32, device=dev)
    z = torch.zeros(1, dtype=torch.int32, device=dev)
    one = torch.ones(1, dtype=torch.int32, device=dev)
    nkv = torch.tensor([cache.shape[2]], dtype=torch.int32, device=dev)
    hd = torch.tensor([cache.shape[4]], dtype=torch.int32, device=dev)
    sn = torch.tensor([s], dtype=torch.int32, device=dev)
    p = start.reshape(1).to(torch.int32)
    begin = torch.cat([li, z, z, p, z])
    end = torch.cat([li + 1, one, nkv, p + sn, hd])
    mutable_slice_update(cache, x.unsqueeze(0), begin, end)


def integer_range_mask(x: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """1.0 where lo <= x <= hi else 0.0, for integer-valued float x, in float arithmetic only."""
    return 1.0 - (x - x.clamp(lo, hi)).abs().clamp(max=1.0)


# ----------------------------------------------------------------------------- transformer blocks
class Attention(nn.Module):
    def __init__(self, dim, n_head, n_kv, head_dim, qkv_bias: bool):
        super().__init__()
        self.n_head, self.n_kv, self.head_dim = n_head, n_kv, head_dim
        self.wqkv = nn.Linear(dim, (n_head + 2 * n_kv) * head_dim, bias=qkv_bias)
        self.wo = nn.Linear(n_head * head_dim, dim, bias=False)

    def forward(self, x, cos, sin, start, mask, k_cache, v_cache, layer):
        b, s, _ = x.shape
        qkv = self.wqkv(x)
        q, k, v = qkv.split((self.n_head * self.head_dim, self.n_kv * self.head_dim, self.n_kv * self.head_dim), dim=-1)
        q = apply_rope_pairs(q.reshape(b, s, self.n_head, self.head_dim), cos, sin).transpose(1, 2)
        k = apply_rope_pairs(k.reshape(b, s, self.n_kv, self.head_dim), cos, sin).transpose(1, 2)
        v = v.reshape(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        write_kv(k_cache, layer, start, k)
        write_kv(v_cache, layer, start, v)
        full_k = k_cache.narrow(0, layer, 1).squeeze(0)      # [1, n_kv, CL, D]
        full_v = v_cache.narrow(0, layer, 1).squeeze(0)
        cl = full_k.shape[2]
        rep = self.n_head // self.n_kv
        full_k = full_k.unsqueeze(2).expand(b, self.n_kv, rep, cl, self.head_dim).reshape(b, self.n_head, cl, self.head_dim)
        full_v = full_v.unsqueeze(2).expand(b, self.n_kv, rep, cl, self.head_dim).reshape(b, self.n_head, cl, self.head_dim)
        out = F.scaled_dot_product_attention(q, full_k, full_v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(b, s, self.n_head * self.head_dim)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, dim, inter):
        super().__init__()
        self.w1 = nn.Linear(dim, inter, bias=False)
        self.w2 = nn.Linear(inter, dim, bias=False)
        self.w3 = nn.Linear(dim, inter, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, dim, inter, n_head, n_kv, head_dim, qkv_bias, eps):
        super().__init__()
        self.attention = Attention(dim, n_head, n_kv, head_dim, qkv_bias)
        self.feed_forward = FeedForward(dim, inter)
        self.attention_norm = RMSNorm(dim, eps)
        self.ffn_norm = RMSNorm(dim, eps)

    def forward(self, x, cos, sin, start, mask, k_cache, v_cache, layer):
        x = x + self.attention(self.attention_norm(x), cos, sin, start, mask, k_cache, v_cache, layer)
        return x + self.feed_forward(self.ffn_norm(x))


def causal_cache_mask(start: torch.Tensor, s: int, cl: int) -> torch.Tensor:
    """bool [1, 1, s, cl]: cache slot j is visible to query row i iff j <= start + i."""
    k_idx = torch.arange(cl, dtype=torch.float32, device=start.device)
    q_pos = torch.arange(s, dtype=torch.float32, device=start.device) + start.to(torch.float32)
    return k_idx.reshape(1, 1, 1, cl) <= q_pos.reshape(1, 1, s, 1)


# ----------------------------------------------------------------------------- slow AR
class SlowARCore(nn.Module):
    def __init__(self, cl: int = MAX_SEQ_LEN):
        super().__init__()
        self.cl = cl
        self.embeddings = nn.Embedding(VOCAB, DIM)
        self.codebook_embeddings = nn.Embedding(CODEBOOK_SIZE * NUM_CODEBOOKS, DIM)
        self.layers = nn.ModuleList([Block(DIM, FFN, N_HEAD, N_KV, HEAD_DIM, True, NORM_EPS) for _ in range(N_LAYER)])
        self.norm = RMSNorm(DIM, NORM_EPS)
        self.head = nn.Linear(DIM, N_ALLOWED, bias=False)           # embeddings.weight[allowed_ids]
        cos, sin = rope_table(MAX_SEQ_LEN, HEAD_DIM, ROPE_BASE)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("codebook_offsets", (torch.arange(NUM_CODEBOOKS) * CODEBOOK_SIZE).to(torch.int32).reshape(1, NUM_CODEBOOKS, 1),
                             persistent=False)

    def embed(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [1, 11, S] int32 -> [1, S, DIM]: token embedding + (semantic rows only) the 10 codebook embeddings."""
        tok = codes[:, 0]                                             # [1, S]
        cb_idx = codes[:, 1:] + self.codebook_offsets                 # [1, 10, S]
        e = self.embeddings(tok)                                      # [1, S, D]
        cb = self.codebook_embeddings(cb_idx).sum(dim=1)              # [1, S, D]
        is_sem = integer_range_mask(tok.to(torch.float32), SEMANTIC_BEGIN, SEMANTIC_END).unsqueeze(-1).to(e.dtype)
        return e + cb * is_sem

    def forward(self, codes, pos, k_cache, v_cache):
        s = codes.shape[2]
        x = self.embed(codes)
        idx = (pos.reshape(1) + torch.arange(s, dtype=torch.int32, device=pos.device)).to(torch.int64)
        cos = self.rope_cos.index_select(0, idx).to(x.dtype)
        sin = self.rope_sin.index_select(0, idx).to(x.dtype)
        mask = causal_cache_mask(pos, s, self.cl)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, pos, mask, k_cache, v_cache, i)
        normalized = self.norm(x)
        logits = self.head(normalized)
        return logits.reshape(s, N_ALLOWED), normalized.reshape(s, DIM)


class SlowARPrefill(nn.Module):
    def __init__(self, core: SlowARCore):
        super().__init__()
        self.core = core

    def forward(self, codes, pos, k_cache, v_cache):
        return self.core(codes, pos, k_cache, v_cache)


class SlowARDecode(nn.Module):
    def __init__(self, core: SlowARCore):
        super().__init__()
        self.core = core

    def forward(self, codes, pos, k_cache, v_cache):
        return self.core(codes, pos, k_cache, v_cache)


def slow_kv_state(cl: int = MAX_SEQ_LEN, dtype=torch.float32):
    shape = (N_LAYER, 1, N_KV, cl, HEAD_DIM)
    return {"k_cache": torch.zeros(shape, dtype=dtype), "v_cache": torch.zeros(shape, dtype=dtype)}


# ----------------------------------------------------------------------------- fast AR
class FastARCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.fast_embeddings = nn.Embedding(CODEBOOK_SIZE, DIM)
        self.fast_layers = nn.ModuleList([Block(DIM, FFN, N_HEAD, N_KV, HEAD_DIM, False, NORM_EPS) for _ in range(N_FAST_LAYER)])
        self.fast_norm = RMSNorm(DIM, NORM_EPS)
        self.fast_output = nn.Linear(DIM, CODEBOOK_SIZE, bias=False)
        cos, sin = rope_table(NUM_CODEBOOKS, HEAD_DIM, ROPE_BASE)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, pos, k_cache, v_cache):
        s = x.shape[1]
        idx = (pos.reshape(1) + torch.arange(s, dtype=torch.int32, device=pos.device)).to(torch.int64)
        cos = self.rope_cos.index_select(0, idx).to(x.dtype)
        sin = self.rope_sin.index_select(0, idx).to(x.dtype)
        mask = causal_cache_mask(pos, s, NUM_CODEBOOKS)
        for i, layer in enumerate(self.fast_layers):
            x = layer(x, cos, sin, pos, mask, k_cache, v_cache, i)
        return self.fast_output(self.fast_norm(x[:, -1:])).reshape(1, CODEBOOK_SIZE)


class FastARFirst(nn.Module):
    """Rows 0 (the slow hidden) and 1 (codebook 0 = the semantic token): logits for codebook 1."""

    def __init__(self, core: FastARCore):
        super().__init__()
        self.core = core

    def forward(self, hidden, code0, k_cache, v_cache):
        e = self.core.fast_embeddings(code0.reshape(1, 1))            # [1, 1, D]
        x = torch.cat([hidden.reshape(1, 1, DIM).to(e.dtype), e], dim=1)
        pos = torch.zeros(1, dtype=torch.int32, device=hidden.device)
        return self.core(x, pos, k_cache, v_cache)


class FastARStep(nn.Module):
    """Row `pos` in 2..9 (codebook pos-1 just sampled): logits for codebook `pos`."""

    def __init__(self, core: FastARCore):
        super().__init__()
        self.core = core

    def forward(self, code, pos, k_cache, v_cache):
        x = self.core.fast_embeddings(code.reshape(1, 1))
        return self.core(x, pos, k_cache, v_cache)


def fast_kv_state(dtype=torch.float32):
    shape = (N_FAST_LAYER, 1, N_KV, NUM_CODEBOOKS, HEAD_DIM)
    return {"k_cache": torch.zeros(shape, dtype=dtype), "v_cache": torch.zeros(shape, dtype=dtype)}


# ----------------------------------------------------------------------------- loading (safetensors)
def load_slow_fast(safetensors_path: Path, cl: int = MAX_SEQ_LEN, dtype=torch.float32):
    sd = load_file(str(safetensors_path))
    slow = SlowARCore(cl)
    fast = FastARCore()
    s_sd, f_sd = {}, {}
    for k, v in sd.items():
        v = v.to(dtype)
        if k.startswith("fast_") :
            f_sd[k] = v
        else:
            s_sd[k] = v
    s_sd["head.weight"] = sd["embeddings.weight"][allowed_ids()].to(dtype).clone()
    missing, unexpected = slow.load_state_dict(s_sd, strict=False)
    missing = [m for m in missing if not m.startswith(("rope_", "codebook_offsets"))]
    assert not missing and not unexpected, (missing, unexpected)
    missing, unexpected = fast.load_state_dict(f_sd, strict=False)
    missing = [m for m in missing if not m.startswith("rope_")]
    assert not missing and not unexpected, (missing, unexpected)
    return slow.to(dtype).eval(), fast.to(dtype).eval()


# ----------------------------------------------------------------------------- codec decoder
def _fold_weight_norm(module: nn.Module) -> None:
    """Fold every weight-norm parametrization (new-style `parametrizations.weight` and legacy `weight_g/v`)
    into plain `.weight` tensors, in place."""
    from torch.nn.utils import parametrize, remove_weight_norm

    for m in list(module.modules()):
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
        elif hasattr(m, "weight_g") and hasattr(m, "weight_v"):
            remove_weight_norm(m)


def load_publisher_codec(snapshot_dir: Path, config) -> nn.Module:
    """Instantiate the publisher's `ArkttsCodec` from `codec.pth` (its own load rules), weight norm folded."""
    spec = importlib.util.spec_from_file_location("arktts_codec_pub", str(Path(snapshot_dir) / "modeling_arktts_codec.py"))
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod          # `@dataclass` looks the module up in sys.modules
    spec.loader.exec_module(mod)
    codec = mod.ArkttsCodec(config)
    state = torch.load(str(Path(snapshot_dir) / "codec.pth"), map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if any("generator." in k for k in state):
        state = {k.replace("generator.", ""): v for k, v in state.items() if "generator." in k}
    state = {k: v for k, v in state.items() if not k.endswith(("freqs_cis", "causal_mask"))}
    codec.load_state_dict(state, strict=True)
    codec.eval()
    _fold_weight_norm(codec)
    return codec


def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)


class Snake(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class CausalConv(nn.Module):
    """Conv1d with left padding (k-1)*d; stride 1 only (every conv on the decode path is stride 1)."""

    def __init__(self, cin, cout, k, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation, groups=groups)
        self.pad = (k - 1) * dilation

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class CausalConvT(nn.Module):
    """ConvTranspose1d(k, s) with the right (k - s) samples cropped: L -> L*s."""

    def __init__(self, cin, cout, k, s):
        super().__init__()
        self.conv = nn.ConvTranspose1d(cin, cout, k, stride=s)
        self.crop = k - s

    def forward(self, x):
        y = self.conv(x)
        return y[..., : y.shape[-1] - self.crop] if self.crop else y


class ResUnit(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.s1 = Snake(dim)
        self.c1 = CausalConv(dim, dim, 7, dilation=dilation)
        self.s2 = Snake(dim)
        self.c2 = CausalConv(dim, dim, 1)

    def forward(self, x):
        return x + self.c2(self.s2(self.c1(self.s1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.snake = Snake(cin)
        self.up = CausalConvT(cin, cout, 2 * stride, stride)
        self.res = nn.ModuleList([ResUnit(cout, d) for d in (1, 3, 9)])

    def forward(self, x):
        x = self.up(self.snake(x))
        for r in self.res:
            x = r(x)
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = CausalConv(dim, dim, 7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = self.dwconv(x).permute(0, 2, 1)
        y = self.pwconv2(F.gelu(self.pwconv1(self.norm(y))))
        return x + (self.gamma * y).permute(0, 2, 1)


class CodecAttention(nn.Module):
    def __init__(self, dim, n_head, n_kv, head_dim):
        super().__init__()
        self.n_head, self.n_kv, self.head_dim = n_head, n_kv, head_dim
        self.wqkv = nn.Linear(dim, (n_head + 2 * n_kv) * head_dim, bias=False)
        self.wo = nn.Linear(n_head * head_dim, dim, bias=False)

    def forward(self, x, cos, sin, mask):
        b, t, _ = x.shape
        q, k, v = self.wqkv(x).split((self.n_head * self.head_dim, self.n_kv * self.head_dim, self.n_kv * self.head_dim), dim=-1)
        q = apply_rope_pairs(q.reshape(b, t, self.n_head, self.head_dim), cos, sin).transpose(1, 2)
        k = apply_rope_pairs(k.reshape(b, t, self.n_kv, self.head_dim), cos, sin).transpose(1, 2)
        v = v.reshape(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        rep = self.n_head // self.n_kv
        k = k.unsqueeze(2).expand(b, self.n_kv, rep, t, self.head_dim).reshape(b, self.n_head, t, self.head_dim)
        v = v.unsqueeze(2).expand(b, self.n_kv, rep, t, self.head_dim).reshape(b, self.n_head, t, self.head_dim)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(out.transpose(1, 2).reshape(b, t, self.n_head * self.head_dim))


class CodecBlock(nn.Module):
    def __init__(self, dim, n_head, n_kv, head_dim, inter, eps):
        super().__init__()
        self.attention = CodecAttention(dim, n_head, n_kv, head_dim)
        self.w1 = nn.Linear(dim, inter, bias=False)
        self.w3 = nn.Linear(dim, inter, bias=False)
        self.w2 = nn.Linear(inter, dim, bias=False)
        self.attention_norm = RMSNorm(dim, eps)
        self.ffn_norm = RMSNorm(dim, eps)
        self.attention_gamma = nn.Parameter(torch.ones(dim))
        self.ffn_gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x, cos, sin, mask):
        x = x + self.attention_gamma * self.attention(self.attention_norm(x), cos, sin, mask)
        h = self.ffn_norm(x)
        return x + self.ffn_gamma * self.w2(F.silu(self.w1(h)) * self.w3(h))


class CodecWindowTransformer(nn.Module):
    """The publisher's `ArkttsCodecWindowTransformer` for lengths up to `t_max` (channels-first in/out).

    RoPE (bf16-rounded, base 1e4) and the causal window-128 mask are baked as buffers: no trig and no integer
    comparison chain in the graph."""

    def __init__(self, n_layer, n_head, n_kv, dim, inter, window, eps, t_max, head_dim=64):
        super().__init__()
        self.layers = nn.ModuleList([CodecBlock(dim, n_head, n_kv, head_dim, inter, eps) for _ in range(n_layer)])
        self.norm = RMSNorm(dim, eps)
        self.window = window
        self.head_dim = head_dim
        cos, sin = rope_table(t_max, head_dim, CODEC_ROPE_BASE)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        row = torch.arange(t_max).reshape(t_max, 1)
        col = torch.arange(t_max).reshape(1, t_max)
        mask = (col <= row) & (col >= (row - window + 1).clamp_min(0))
        self.register_buffer("mask", mask.reshape(1, 1, t_max, t_max), persistent=False)

    def forward(self, x):                                   # x [B, C, T]
        x = x.transpose(1, 2)
        t = x.shape[1]
        cos, sin = self.rope_cos[:t].to(x.dtype), self.rope_sin[:t].to(x.dtype)
        mask = self.mask[:, :, :t, :t]
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.norm(x).transpose(1, 2)


class CodecDecoder(nn.Module):
    """codes [1, 10, T] int32 -> wav [1, T*2048] fp32/fp16 in [-1, 1]."""

    def __init__(self, t_max: int = 512):
        super().__init__()
        self.codebooks = nn.ParameterList([nn.Parameter(torch.zeros(CODEBOOK_SIZE if i == 0 else 1024, 8)) for i in range(NUM_CODEBOOKS)])
        self.out_proj = nn.ModuleList([nn.Conv1d(8, CODEC_DIM, 1) for _ in range(NUM_CODEBOOKS)])
        self.post = CodecWindowTransformer(CODEC_POST_LAYERS, CODEC_POST_HEADS, CODEC_POST_KV, CODEC_DIM, CODEC_POST_FFN,
                                           CODEC_WINDOW, CODEC_NORM_EPS, t_max)
        self.upsample = nn.ModuleList([nn.ModuleList([CausalConvT(CODEC_DIM, CODEC_DIM, 2, 2), ConvNeXtBlock(CODEC_DIM)]) for _ in range(2)])
        self.first = CausalConv(CODEC_DIM, 1536, 7)
        self.blocks = nn.ModuleList([DecoderBlock(1536 // (2 ** i), 1536 // (2 ** (i + 1)), s) for i, s in enumerate((8, 8, 4, 2))])
        self.last_snake = Snake(96)
        self.last = CausalConv(96, 1, 7)

    def forward(self, codes):
        b, _, t = codes.shape
        z = None
        for i in range(NUM_CODEBOOKS):
            # the publisher's decode() clamps: codebook 0 to 0..4095, codebooks 1..9 to 0..1023 (the fast head is
            # 4096 wide for every codebook; the codec's residual books have 1024 rows)
            idx = codes[:, i].clamp(0, self.codebooks[i].shape[0] - 1).to(torch.int64)
            proj = F.embedding(idx, self.codebooks[i]).transpose(1, 2)        # [B, 8, T]
            y = self.out_proj[i](proj)
            z = y if z is None else z + y
        z = self.post(z)
        for up, cnx in self.upsample:
            z = cnx(up(z))
        x = self.first(z)
        for blk in self.blocks:
            x = blk(x)
        x = torch.tanh(self.last(self.last_snake(x)))
        return x.reshape(b, t * CODEC_FRAME)


def _copy_conv(dst: nn.Conv1d | nn.ConvTranspose1d, src) -> None:
    dst.weight.data.copy_(src.weight.detach())
    dst.bias.data.copy_(src.bias.detach())


def build_codec_decoder(pub, t_max: int = 512) -> CodecDecoder:
    """Copy the decode-path tensors out of the publisher's (weight-norm-folded) codec by object traversal."""
    dec = CodecDecoder(t_max)
    q = pub.quantizer
    vqs = [q.semantic_quantizer.quantizers[0]] + list(q.quantizer.quantizers)
    for i, vq in enumerate(vqs):
        dec.codebooks[i].data.copy_(vq.codebook.weight.detach())
        _copy_conv(dec.out_proj[i], vq.out_proj)
    for li, layer in enumerate(q.post_module.layers):
        d = dec.post.layers[li]
        d.attention.wqkv.weight.data.copy_(layer.attention.wqkv.weight.detach())
        d.attention.wo.weight.data.copy_(layer.attention.wo.weight.detach())
        d.w1.weight.data.copy_(layer.feed_forward.w1.weight.detach())
        d.w2.weight.data.copy_(layer.feed_forward.w2.weight.detach())
        d.w3.weight.data.copy_(layer.feed_forward.w3.weight.detach())
        d.attention_norm.weight.data.copy_(layer.attention_norm.weight.detach())
        d.ffn_norm.weight.data.copy_(layer.ffn_norm.weight.detach())
        d.attention_gamma.data.copy_(layer.attention_layer_scale.gamma.detach())
        d.ffn_gamma.data.copy_(layer.ffn_layer_scale.gamma.detach())
    dec.post.norm.weight.data.copy_(q.post_module.norm.weight.detach())
    for i in range(2):
        up_src, cnx_src = q.upsample[i][0], q.upsample[i][1]
        _copy_conv(dec.upsample[i][0].conv, up_src.conv)
        cnx = dec.upsample[i][1]
        _copy_conv(cnx.dwconv.conv, cnx_src.dwconv.conv)
        cnx.norm.weight.data.copy_(cnx_src.norm.weight.detach()); cnx.norm.bias.data.copy_(cnx_src.norm.bias.detach())
        cnx.pwconv1.weight.data.copy_(cnx_src.pwconv1.weight.detach()); cnx.pwconv1.bias.data.copy_(cnx_src.pwconv1.bias.detach())
        cnx.pwconv2.weight.data.copy_(cnx_src.pwconv2.weight.detach()); cnx.pwconv2.bias.data.copy_(cnx_src.pwconv2.bias.detach())
        cnx.gamma.data.copy_(cnx_src.gamma.detach())
    m = pub.decoder.model
    _copy_conv(dec.first.conv, m[0].conv)
    for bi in range(4):
        src_blk = m[1 + bi].block
        blk = dec.blocks[bi]
        blk.snake.alpha.data.copy_(src_blk[0].alpha.detach())
        _copy_conv(blk.up.conv, src_blk[1].conv)
        for ri in range(3):
            ru_src = src_blk[2 + ri].block
            ru = blk.res[ri]
            ru.s1.alpha.data.copy_(ru_src[0].alpha.detach())
            _copy_conv(ru.c1.conv, ru_src[1].conv)
            ru.s2.alpha.data.copy_(ru_src[2].alpha.detach())
            _copy_conv(ru.c2.conv, ru_src[3].conv)
    dec.last_snake.alpha.data.copy_(m[5].alpha.detach())
    _copy_conv(dec.last.conv, m[6].conv)
    return dec.eval()


__all__ = [
    "SlowARCore", "SlowARPrefill", "SlowARDecode", "slow_kv_state", "FastARCore", "FastARFirst", "FastARStep",
    "fast_kv_state", "load_slow_fast", "load_publisher_codec", "build_codec_decoder", "CodecDecoder", "allowed_ids",
]
