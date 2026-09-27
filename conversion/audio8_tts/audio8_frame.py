# Community port — NOT an Apple model.
"""One graph call per frame: slow decode + semantic sampling (RAS) + the unrolled fast AR with in-graph codebook sampling.

The split graphs (audio8_model.py: slow decode, fast first, fast step ×8) cost ten runtime calls per 46 ms frame, and a
call is ~2–3 ms of fixed cost on the Core AI GPU path before any arithmetic — the frame ran at real time on an M4 Max.
This module folds the frame into one call:

  frame(codes [1,11,1], pos [1], noise_slow [2,4097], window [10], noise_fast [9,4096], forced [11], use_forced [1])
      -> semantic [1] i32, codes [10] i32, logits [4097] f16, hidden [896] f16, fast_logits [9,4096] f16,
         sampled_semantic [1] i32, sampled_codes [10] i32
  first_frame(logits [4097], hidden [896], noise_slow, window, noise_fast, forced, use_forced)  -> the same outputs
      (the frame right after the prefill, whose logits/hidden the prefill graph returned)

Sampling in the graph is the publisher's processor written without a sort: `topk(50)` bounds the top-p candidates
(nothing outside the top 50 survives top-k anyway), the full-vocabulary softmax normaliser comes from `logsumexp`,
the cumulative sum over the 50 sorted probabilities gives the top-p cut (the best is always kept), the kept scores
are divided by the temperature, and the draw is `argmax(softmax(kept) / -log(u))` with `u` gathered at the
candidates' indices — the Gumbel-max form `ArkttsModel._sample` uses. The RAS rule takes the 10-token window as an
input (-1 = the publisher's `previous is None`). Every integer decision is float arithmetic on integer-valued floats
(no int comparison chains: conversion-guide, RF-DETR finding 2).

`forced` / `use_forced` teacher-force the frame's tokens (forced[0] = semantic id, forced[1..10] = codebooks) so the
same graph yields, in one pass, the logits the oracle's tokens produce AND what the in-graph sampler would have drawn
from them with the oracle's noise — the gate's two questions.

The fast AR is unrolled over its ten rows with the keys/values concatenated in-graph: no state, no per-row call.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from audio8_model import (CODEBOOK_SIZE, DIM, EOS, N_ALLOWED, NUM_CODEBOOKS, SEMANTIC_BEGIN, FastARCore, SlowARCore,
                          apply_rope_pairs, causal_cache_mask)

TOP_K = 50
TOP_P = 0.9
TEMPERATURE = 0.7
RAS_TOP_P = 0.9
RAS_TEMPERATURE = 1.0
RAS_WINDOW = 10
EOS_INDEX = N_ALLOWED - 1


def sample_topk_topp(logits: torch.Tensor, noise: torch.Tensor, top_k: int, top_p: float, temperature: float) -> torch.Tensor:
    """logits [1, N] (any float dtype), noise [1, N] f32 -> the chosen index [1] (int64), publisher's semantics."""
    x = logits.float()
    lse = torch.logsumexp(x, dim=-1, keepdim=True)
    vals, idx = torch.topk(x, top_k, dim=-1)                          # sorted descending
    p = torch.exp(vals - lse)                                         # full-softmax probabilities of the candidates
    cum = torch.cumsum(p, dim=-1)
    keep_f = 1.0 - (cum - top_p).clamp(min=0.0, max=1e-6) * 1e6     # 1 where cum <= top_p, else 0 (float, no bool chain)
    keep_f = torch.cat([torch.ones_like(keep_f[:, :1]), keep_f[:, 1:]], dim=-1)   # the best is always kept
    scores = vals / max(temperature, 1e-5)
    masked = scores + (keep_f - 1.0) * 1e9                            # dropped -> -1e9 (softmax weight exactly 0 in fp32)
    q = torch.softmax(masked, dim=-1)
    u = torch.gather(noise, 1, idx)
    g = q / (-torch.log(u))
    j = torch.argmax(g, dim=-1, keepdim=True)
    return torch.gather(idx, 1, j).reshape(1)


def index_to_id(idx_f: torch.Tensor) -> torch.Tensor:
    """allowed index (float, 0..4096) -> token id (float): semantic ids, or eos for index 4096."""
    is_eos = (idx_f - (EOS_INDEX - 1)).clamp(0.0, 1.0)
    return SEMANTIC_BEGIN + idx_f + is_eos * (EOS - SEMANTIC_BEGIN - EOS_INDEX)


def sample_semantic(logits4097: torch.Tensor, noise_slow: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """RAS: normal draw, high draw, replace a repeated semantic by the high draw. Returns the token id [1] (float)."""
    normal = sample_topk_topp(logits4097, noise_slow[0:1], TOP_K, TOP_P, TEMPERATURE).float()
    high = sample_topk_topp(logits4097, noise_slow[1:2], TOP_K, RAS_TOP_P, RAS_TEMPERATURE).float()
    normal_id, high_id = index_to_id(normal), index_to_id(high)
    is_sem = 1.0 - (normal - (EOS_INDEX - 1)).clamp(0.0, 1.0)         # 1 for a semantic index, 0 for eos
    d = (window.float() - normal_id).abs().min()                       # 0 when the id is in the window
    repeated = (1.0 - d.clamp(0.0, 1.0)) * is_sem                      # 1 when a repeated semantic id
    return normal_id + repeated * (high_id - normal_id)


class FastUnrolled(nn.Module):
    """The fast AR over its ten rows in one pass, sampling codebooks 1..9 in-graph (or taking them forced)."""

    def __init__(self, core: FastARCore):
        super().__init__()
        self.core = core

    def attend(self, layer, x, cos, sin, keys, values, li):
        """x [1,1,D] at row i; keys/values: per-layer lists of [1, n_kv, 1, D] from the rows before."""
        att = layer.attention
        h = layer.attention_norm(x)
        qkv = att.wqkv(h)
        q, k, v = qkv.split((att.n_head * att.head_dim, att.n_kv * att.head_dim, att.n_kv * att.head_dim), dim=-1)
        q = apply_rope_pairs(q.reshape(1, 1, att.n_head, att.head_dim), cos, sin).transpose(1, 2)
        k = apply_rope_pairs(k.reshape(1, 1, att.n_kv, att.head_dim), cos, sin).transpose(1, 2)
        v = v.reshape(1, 1, att.n_kv, att.head_dim).transpose(1, 2)
        keys[li].append(k)
        values[li].append(v)
        K = torch.cat(keys[li], dim=2)                                   # [1, n_kv, i+1, D]
        V = torch.cat(values[li], dim=2)
        rep = att.n_head // att.n_kv
        n = K.shape[2]
        K = K.unsqueeze(2).expand(1, att.n_kv, rep, n, att.head_dim).reshape(1, att.n_head, n, att.head_dim)
        V = V.unsqueeze(2).expand(1, att.n_kv, rep, n, att.head_dim).reshape(1, att.n_head, n, att.head_dim)
        out = F.scaled_dot_product_attention(q, K, V)                    # the query is the last row: every key is visible
        out = out.transpose(1, 2).reshape(1, 1, att.n_head * att.head_dim)
        x = x + att.wo(out)
        return x + layer.feed_forward(layer.ffn_norm(x))

    def row(self, x, i, keys, values):
        cos = self.core.rope_cos[i:i + 1].to(x.dtype)
        sin = self.core.rope_sin[i:i + 1].to(x.dtype)
        for li, layer in enumerate(self.core.fast_layers):
            x = self.attend(layer, x, cos, sin, keys, values, li)
        return self.core.fast_output(self.core.fast_norm(x)).reshape(1, CODEBOOK_SIZE)

    def forward(self, hidden, code0_f, noise_fast, forced_f, use_forced):
        """hidden [1,896]; code0_f [1] float; noise_fast [9,4096]; forced_f [10] float (codebooks); use_forced [1] float.
        -> codes_f [10] float, fast_logits [9, 4096], sampled_f [10] float (the sampler's own choices)."""
        keys = [[] for _ in self.core.fast_layers]
        values = [[] for _ in self.core.fast_layers]
        dt = self.core.fast_embeddings.weight.dtype
        x = hidden.reshape(1, 1, DIM).to(dt)
        _ = self.row(x, 0, keys, values)                                 # row 0: the slow hidden (its logits are unused)
        f = use_forced.reshape(1)
        cur_f = code0_f.reshape(1)
        codes = [cur_f]
        sampled = [cur_f]
        logits_all = []
        for i in range(1, NUM_CODEBOOKS):
            x = self.core.fast_embeddings(cur_f.to(torch.int32)).reshape(1, 1, DIM)
            logits = self.row(x, i, keys, values)                        # logits for codebook i
            logits_all.append(logits.reshape(1, CODEBOOK_SIZE))
            drawn = sample_topk_topp(logits, noise_fast[i - 1:i], TOP_K, TOP_P, TEMPERATURE).float()
            sampled.append(drawn)
            cur_f = drawn + f * (forced_f[i:i + 1] - drawn)
            codes.append(cur_f)
        return torch.cat(codes), torch.cat(logits_all, dim=0), torch.cat(sampled)


def _frame_tail(fast: FastUnrolled, logits4097, hidden, noise_slow, window, noise_fast, forced, use_forced):
    f = use_forced.reshape(1).float()
    forced_f = forced.float()
    sem_sampled = sample_semantic(logits4097.reshape(1, N_ALLOWED), noise_slow, window)             # [1] float id
    sem = sem_sampled + f * (forced_f[0:1] - sem_sampled)
    code0 = (sem - SEMANTIC_BEGIN).clamp(0.0, CODEBOOK_SIZE - 1)
    codes_f, fast_logits, sampled_f = fast(hidden, code0, noise_fast, forced_f[1:], use_forced)
    return (sem.to(torch.int32), codes_f.to(torch.int32), logits4097.reshape(N_ALLOWED), hidden.reshape(DIM),
            fast_logits, sem_sampled.to(torch.int32), sampled_f.to(torch.int32))


class FrameGraph(nn.Module):
    """slow decode (with the KV state) + the frame tail."""

    def __init__(self, slow: SlowARCore, fast: FastUnrolled):
        super().__init__()
        self.slow = slow
        self.fast = fast

    def forward(self, codes, pos, noise_slow, window, noise_fast, forced, use_forced, k_cache, v_cache):
        logits, hidden = self.slow(codes, pos, k_cache, v_cache)         # [1,4097], [1,896]
        return _frame_tail(self.fast, logits, hidden, noise_slow, window, noise_fast, forced, use_forced)


class FirstFrameGraph(nn.Module):
    """the frame tail on the prefill's last-row logits/hidden (no slow decode, no state)."""

    def __init__(self, fast: FastUnrolled):
        super().__init__()
        self.fast = fast

    def forward(self, logits, hidden, noise_slow, window, noise_fast, forced, use_forced):
        return _frame_tail(self.fast, logits, hidden, noise_slow, window, noise_fast, forced, use_forced)


FRAME_OUTPUTS = ("semantic", "codes", "logits", "hidden", "fast_logits", "sampled_semantic", "sampled_codes")
