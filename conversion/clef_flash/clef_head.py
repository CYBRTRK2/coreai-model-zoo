#!/usr/bin/env python3
"""The clef-flash joint schema head as a tensor-only graph, and the host arrays that feed it.

The checkpoint's `JointSchemaHead` (joint_schema_model.py at the pinned revision) loops over records,
questions and options in Python and reads the untied lm_head table inside its forward, so it cannot be
exported as is (`torch.export` stops at `int(attention_mask.sum().item())`). `ClefHeadGraph` holds the
author's module itself (`load_state_dict(strict=True)` on joint_head.safetensors, every key used) and
computes the same forward for one record with the loops written as matrices:

    hidden    [T, 4096] f32   the decoder's final-norm hidden rows (rows T_real.. are padding)
    key_valid [T]       f32   1 = a real token, 0 = padding
    q_avg     [Q, T]    f32   row q = 1/len over question q's span (a padding question: zeros)
    o_avg     [O, T]    f32   row o = 1/len over option o's span (a padding option: zeros)
    g_avg     [1, T]    f32   1 at the last real token (the author's global vector)
    lexical   [O, 4096] f32   the mean of the lm_head rows of option o's span ids (the host's gather)
    member    [Q, O]    f32   1 where option o belongs to question q
    type_ids  [Q]       i32   noul 0, choice 1, score 2 (padding questions: 0)
    q_valid   [Q]       f32   1 = a real question
    o_valid   [O]       f32   1 = a real option
    -> logits [O]       f32   one logit per option, in the author's option order; padding options 0

    x       = hidden_norm(hidden);   memory = memory_projection(x)
    qv      = q_avg @ x;   ctx = o_avg @ x;   global = g_avg @ x
    queries = option_context_projection(ctx) + option_lexical_projection(lexical)
              + member^T @ option_question_projection(qv)
    2 x evidence routing (pre-LN cross-attention of the options to the memory, then the FFN)
    summary = softmax over each question's options of routed . question_projection(qv) / 32, weighted sum
    fields  = question_projection(qv) + option_summary_norm(summary) + global_projection(global)
              + type_embedding(type_ids)
    4 x decoder layer (norm_first: self-attention over the questions, cross-attention to the memory,
              FFN); field_norm
    prior   = e^min(prior_logit_scale, ln 100) * cos(lexical, member^T @ (qv + global))
    joint   = e^min(joint_logit_scale, ln 100) * cos(member^T @ fields, option_norm(routed))
              + residual_scorer([f, o, f*o, |f-o|])
    logits  = (prior + sigmoid(residual_gate) * joint) * o_valid

The attention blocks use the modules' own weights (in_proj rows [0:E] / [E:2E] / [2E:3E] = q / k / v,
16 heads of 64) in plain matmul / softmax form, so the graph carries no `nn.MultiheadAttention` fast
path; padding keys get an additive -1e9 (exp underflows to exactly 0 in fp32, so a padded record scores
like the unpadded one). Dropout is skipped (eval). The per-question softmax, the lexical gather and the
SystemOne response stay on the host (`question_probs`, `lexical_rows`, `decide.py`).

Key blocks: an attention over more than KEY_CHUNK (2048) keys is written as blocks of at most 2048
keys sharing one max (exp(s - max) per block, the denominators and the block products summed), the
same softmax. Measured 2026-10-03 (macOS 27 26A428, coreai-build of Xcode 27.0 RC, AOT h16c, Python
runtime): the plain `softmax(q k^T * s [+ bias]) v` chain compiled for the GPU returns wrong values that
change from call to call once the key length reaches 4032 (3584 is exact; 16 / 8 / 1 heads, 16 / 64 /
128 queries, with or without the bias all fail at 4096), while each op alone is exact at 4096 and the
blocked form is exact (max |d| 5.7e-7) at 4032 and 4096 (L/logs/r5_attn_probe.log,
r5_op_probe.log, r5_attn_fix_probe.log). A static T <= 2048 traces to the plain form.

`head_inputs()` builds the ten arrays from the hidden rows, the processor-form ids, the spans
(`host.build_ids()` questions or the oracle's) and the lm_head table, padded to a T multiple and to
(Q, O) bucket sizes when asked. It is the spec the Swift host copies.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True   # no .pyc next to the author's path-imported file in the snapshot
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot  # noqa: E402

HF_ID = "Cloudflare/clef-flash"
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
JSM_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
HIDDEN = 4096
VOCAB = 248320
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
INPUT_NAMES = ("hidden", "key_valid", "q_avg", "o_avg", "g_avg", "lexical", "member", "type_ids", "q_valid", "o_valid")
OUTPUT_NAMES = ("logits",)
MASK_NEG = -1.0e9   # additive bias of a masked key / a non-member option (fp32 graph)
KEY_CHUNK = 2048    # keys per attention block (see "Key blocks" above)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsm():
    """The checkpoint's joint_schema_model.py, path-imported unchanged (sha256 pinned)."""
    if "joint_schema_model" in sys.modules:
        return sys.modules["joint_schema_model"]
    path = Path(hf_snapshot(HF_ID, "joint_schema_model.py", revision=REVISION))
    assert sha256_file(path) == JSM_SHA256, "joint_schema_model.py is not the pinned file"
    spec = importlib.util.spec_from_file_location("joint_schema_model", path)
    jsm = importlib.util.module_from_spec(spec)
    sys.modules["joint_schema_model"] = jsm
    spec.loader.exec_module(jsm)
    return jsm


def load_author_head(dtype=None):
    """The author's JointSchemaHead with joint_head.safetensors loaded strictly (bf16 values, cast to `dtype`)."""
    from safetensors.torch import load_file

    jsm = load_jsm()
    snap = Path(hf_snapshot(HF_ID, revision=REVISION))
    head = jsm.JointSchemaHead(**json.loads((snap / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(str(snap / "joint_head.safetensors")), strict=True)
    return head.to(dtype=dtype or torch.float32).eval()


def _attention(mha, q_in, kv_in, key_bias):
    """nn.MultiheadAttention (batch_first, eval, key padding only) as matmul / softmax.
    q_in [Lq, E], kv_in [Lk, E], key_bias [Lk] (0 or MASK_NEG) -> [Lq, E]."""
    E, H = mha.embed_dim, mha.num_heads
    D = E // H
    w, b = mha.in_proj_weight, mha.in_proj_bias
    q = F.linear(q_in, w[:E], b[:E]).reshape(q_in.shape[0], H, D).transpose(0, 1)          # [H, Lq, D]
    k = F.linear(kv_in, w[E:2 * E], b[E:2 * E]).reshape(kv_in.shape[0], H, D).transpose(0, 1)
    v = F.linear(kv_in, w[2 * E:], b[2 * E:]).reshape(kv_in.shape[0], H, D).transpose(0, 1)
    scale = 1.0 / math.sqrt(D)
    Lk = kv_in.shape[0]
    if isinstance(Lk, int) and Lk > KEY_CHUNK:                                              # key blocks
        bounds = [(a, min(a + KEY_CHUNK, Lk)) for a in range(0, Lk, KEY_CHUNK)]
        s = [torch.matmul(q, k[:, a:z].transpose(1, 2)) * scale + key_bias[a:z] for a, z in bounds]
        m = s[0].amax(dim=-1, keepdim=True)
        for x in s[1:]:
            m = torch.maximum(m, x.amax(dim=-1, keepdim=True))
        e = [torch.exp(x - m) for x in s]
        den = e[0].sum(dim=-1, keepdim=True)
        num = torch.matmul(e[0], v[:, bounds[0][0]:bounds[0][1]])
        for x, (a, z) in zip(e[1:], bounds[1:]):
            den = den + x.sum(dim=-1, keepdim=True)
            num = num + torch.matmul(x, v[:, a:z])
        o = num / den
    else:
        s = torch.matmul(q, k.transpose(1, 2)) * scale + key_bias                          # [H, Lq, Lk]
        o = torch.matmul(torch.softmax(s, dim=-1), v)
    o = o.transpose(0, 1).reshape(q_in.shape[0], E)
    return F.linear(o, mha.out_proj.weight, mha.out_proj.bias)


def _normalize(x, eps):
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)


class ClefHeadGraph(torch.nn.Module):
    """The author's JointSchemaHead for one record, tensor inputs only (see the module docstring)."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, hidden, key_valid, q_avg, o_avg, g_avg, lexical, member, type_ids, q_valid, o_valid):
        h = self.head
        key_bias = (key_valid - 1.0) * (-MASK_NEG)            # 0 for a real token, -1e9 for padding
        question_bias = (q_valid - 1.0) * (-MASK_NEG)
        member_t = member.transpose(0, 1)                       # [O, Q]

        x = h.hidden_norm(hidden)                               # [T, 4096]
        memory = h.memory_projection(x)                         # [T, 1024]
        qv = torch.matmul(q_avg, x)                             # [Q, 4096]
        ctx = torch.matmul(o_avg, x)                            # [O, 4096]
        global_vector = torch.matmul(g_avg, x)                  # [1, 4096]

        queries = (h.option_context_projection(ctx) + h.option_lexical_projection(lexical)
                   + torch.matmul(member_t, h.option_question_projection(qv)))             # [O, 1024]
        for layer in h.evidence_layers:                         # pre-LN cross-attention to the memory + FFN
            queries = queries + _attention(layer.attention, layer.query_norm(queries), layer.memory_norm(memory),
                                           key_bias)
            ff = layer.feedforward                              # Linear, GELU, Dropout, Linear, Dropout
            queries = queries + ff[3](ff[1](ff[0](layer.feedforward_norm(queries))))
        routed = queries                                        # [O, 1024]

        base = h.question_projection(qv)                        # [Q, 1024]
        scores = torch.matmul(base, routed.transpose(0, 1)) * (1.0 / math.sqrt(routed.shape[-1]))
        weights = torch.softmax(scores + (member - 1.0) * (-MASK_NEG), dim=-1)              # [Q, O]
        summaries = torch.matmul(weights, routed)               # [Q, 1024]
        fields = (base + h.option_summary_norm(summaries) + h.global_projection(global_vector)
                  + h.type_embedding(type_ids))                 # [Q, 1024]
        for layer in h.layers:                                  # TransformerDecoderLayer: norm_first, gelu
            f1 = layer.norm1(fields)
            fields = fields + _attention(layer.self_attn, f1, f1, question_bias)
            fields = fields + _attention(layer.multihead_attn, layer.norm2(fields), memory, key_bias)
            fields = fields + layer.linear2(F.gelu(layer.linear1(layer.norm3(fields))))
        fields = h.field_norm(fields)

        anchor = _normalize(qv + global_vector, 1e-12)          # F.normalize
        prior_scale = h.prior_logit_scale.clamp(max=math.log(100.0)).exp()
        prior = prior_scale * (_normalize(lexical, 1e-12) * torch.matmul(member_t, anchor)).sum(-1)
        options = h.option_norm(routed)                         # [O, 1024]
        f_o = torch.matmul(member_t, fields)                    # [O, 1024]: each option's question field
        cosine = (_normalize(f_o, 1e-8) * _normalize(options, 1e-8)).sum(-1)                 # F.cosine_similarity
        rs = h.residual_scorer                                  # Linear, GELU, Dropout, Linear
        feats = torch.cat([f_o, options, f_o * options, torch.abs(f_o - options)], dim=-1)
        residual = rs[3](rs[1](rs[0](feats))).squeeze(-1)
        joint = h.joint_logit_scale.clamp(max=math.log(100.0)).exp() * cosine + residual
        logits = prior + torch.sigmoid(h.residual_gate) * joint
        return logits * o_valid


# --------------------------------------------------------------------------- host arrays
def lexical_rows(table, ids, spans) -> np.ndarray:
    """[n_spans, 4096] f32: the mean of the table rows of each span's ids (accumulated in f64).
    `table` = any [V, 4096] array (the fp16 raw file as a memmap, or f32 rows)."""
    ids = np.asarray(ids, dtype=np.int64)
    out = np.zeros((len(spans), HIDDEN), np.float32)
    for i, (s, e) in enumerate(spans):
        rows = np.asarray(table[ids[s:e]], dtype=np.float64)
        out[i] = rows.mean(axis=0).astype(np.float32)
    return out


def ceil_to(n: int, m: int) -> int:
    return -(-n // m) * m


def head_inputs(hidden, ids, questions, table=None, *, lexical=None, t_pad: int | None = None,
                q_pad: int | None = None, o_pad: int | None = None) -> tuple[dict, list[tuple[int, int]]]:
    """One record -> the graph's ten inputs (NumPy, graph dtypes) + each question's [start, end) option slice.

    hidden [T, 4096] (any float dtype; rows past the real tokens may be present and are masked),
    ids = the processor-form ids (len T_real), questions = [{type, question_span, option_spans}, ...]
    (host.build_ids() questions or the oracle's rows). `t_pad` / `q_pad` / `o_pad` = the padded sizes
    (None = exact). `lexical` [O_real, 4096] overrides the table gather."""
    T = len(ids)
    hidden = np.asarray(hidden)
    Tp = t_pad or T
    if hidden.shape[0] < T or Tp < T:
        raise ValueError(f"hidden rows {hidden.shape[0]} / t_pad {Tp} < tokens {T}")
    Q = len(questions)
    O = sum(len(q["option_spans"]) for q in questions)
    Qp, Op = q_pad or Q, o_pad or O
    if Qp < Q or Op < O:
        raise ValueError(f"{Q} questions / {O} options do not fit q_pad {Qp} / o_pad {Op}")
    h = np.zeros((Tp, HIDDEN), np.float32)
    n = min(hidden.shape[0], Tp)
    h[:n] = hidden[:n].astype(np.float32)
    h[T:] = 0.0                                           # rows past the real tokens are masked; keep them finite
    key_valid = np.zeros(Tp, np.float32)
    key_valid[:T] = 1.0
    q_avg = np.zeros((Qp, Tp), np.float32)
    o_avg = np.zeros((Op, Tp), np.float32)
    g_avg = np.zeros((1, Tp), np.float32)
    g_avg[0, T - 1] = 1.0
    member = np.zeros((Qp, Op), np.float32)
    type_ids = np.zeros(Qp, np.int32)
    q_valid = np.zeros(Qp, np.float32)
    o_valid = np.zeros(Op, np.float32)
    spans, layout, o = [], [], 0
    for qi, q in enumerate(questions):
        s, e = q["question_span"]
        if not 0 <= s < e <= T:
            raise ValueError(f"question span {s, e} outside [0, {T})")
        q_avg[qi, s:e] = np.float32(1.0) / np.float32(e - s)
        type_ids[qi] = QUESTION_TYPES[q["type"]]
        q_valid[qi] = 1.0
        start = o
        for (a, b) in q["option_spans"]:
            if not 0 <= a < b <= T:
                raise ValueError(f"option span {a, b} outside [0, {T})")
            o_avg[o, a:b] = np.float32(1.0) / np.float32(b - a)
            member[qi, o] = 1.0
            o_valid[o] = 1.0
            spans.append((a, b))
            o += 1
        layout.append((start, o))
    lex = np.zeros((Op, HIDDEN), np.float32)
    if lexical is not None:
        lex[:O] = np.asarray(lexical, np.float32)
    else:
        lex[:O] = lexical_rows(table, ids, spans)
    arrays = {"hidden": h, "key_valid": key_valid, "q_avg": q_avg, "o_avg": o_avg, "g_avg": g_avg, "lexical": lex,
              "member": member, "type_ids": type_ids, "q_valid": q_valid, "o_valid": o_valid}
    return arrays, layout


def question_probs(logits, layout) -> list[np.ndarray]:
    """Per question, the fp32 softmax over its options (the author's `question_logits.float().softmax(-1)`)."""
    out = []
    for a, b in layout:
        z = np.asarray(logits[a:b], np.float32)
        e = np.exp(z - z.max())
        out.append(e / e.sum(dtype=np.float32))
    return out
