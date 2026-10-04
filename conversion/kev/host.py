#!/usr/bin/env python3
"""Kev host reference: a SystemOne request -> one decoder row per question -> pointer head -> the SystemOne response.

NumPy, `tokenizers` and json only; the author's `kev` package is not imported. This file is the specification a
Swift host copies: every rule below is the author's code (github.com/jaredpalmer/kev tag kev-1.0, `kev/api.py`,
`kev/model.py`, `kev/serve.py`) written out, and `test_host.py` gates it against the author's fp32 oracle (ids,
readout indices, keys, answers, usage) before any Swift exists. `decide.py` runs it on the Core AI graph.

1. Request (`kev.api.SystemOneRequest`, pydantic v2; `validate_request` makes the same accept / reject decisions)

     {"model": str = "kev-latest", "state": <JSON>, "questions": {qid: question, ...}}
     question = {"type": "noul",   "instructions"?: <JSON>, "criteria"?: {str: <JSON>} | null}
              | {"type": "choice", "instructions"?: <JSON>, "criteria":  {str: <JSON>}}      1..255 entries
              | {"type": "score",  "instructions"?: <JSON>, "criteria":  [<JSON>, ...]}      1..255 items

   <JSON> is any JSON value (string, number, true / false, null, object, array). `state` is required (null is
   allowed); `questions` needs at least one entry; `model`, when present, must be a string. A noul `criteria` object
   may carry any keys: only "true" and "false" are read. Unknown fields are ignored (pydantic's default). Anything
   else raises ValueError (the author's server answers 422).

2. Text (`kev.api.render`, `option_text`, `to_record`)

     render(None) = ""; render(str | int | float | bool) = Python str(): "True" / "False" for booleans, the
       shortest round-trip repr for floats ("1.0", "0.1", "1e-07"), integers in full
     render(array, indent) = "\\n".join(pad + "- " + render(item, indent + 1).lstrip()) for each item
     render(object, indent) = "\\n".join(pad + key + ":\\n" + render(value, indent + 1)   if value is an object or array
                                         pad + key + ": " + render(value)               otherwise)
       pad = two spaces per indent level; lstrip = Python str.lstrip() (Unicode whitespace)
     option_text(name, desc) = name if desc is None or desc == "" else name + ": " + render(desc)
     options:  noul   = [option_text("no", criteria.false), option_text("yes", criteria.true)]   (false first)
               choice = [option_text(name, desc) for name, desc in criteria]                     (request order)
               score  = [render(level) for level in criteria]
     keys (`question_keys`, the order probabilities are reported in):
               noul ["false", "true"]; choice the criteria names; score "0" .. "n-1"
     legend (score only) = {"i": render(level_i)}
     record = {state: render(state), questions: [{instr: render(instructions), options}]}   (request order)

3. Tokens (`kev.model.user_tokens`, `encode`, `admit`, `rows_of`)

     tokenizer = the bundle's tokenizer/tokenizer.json (Qwen3.5 base, sha256 fe000e3e...; no bos)
     user_tokens(text) = encode(re.sub(r"<\\|([A-Za-z0-9_]+)\\|>", r"<\\u00a6\\1\\u00a6>", text), no special tokens)
       (caller text can never produce a delimiter; "" -> no tokens)
     delimiters (looked up by token text): <|fim_prefix|> state 248060, <|fim_middle|> q 248061,
       <|box_start|> opt 248049, <|box_end|> /opt 248050, <|fim_suffix|> decide 248062; pad <|endoftext|> 248044
     S_ids = [state] + user_tokens(record.state)                                  Ls = len(S_ids)
     branch_k = [q] + user_tokens(instr_k) + for each option ([opt] + user_tokens(option) + [/opt]) + [decide]
     packed ids = S_ids + branch_1 + ... + branch_Q          usage.input_tokens = len(packed ids)
     limits (serving, strict, nothing is ever cut): Ls <= 65,536, else ContextOverflow
       "state is {Ls:,} tokens, over the 65,536-token limit (the <state> token included): shorten the document or
       split it across requests"; Ls + len(branch_k) <= 73,728, else ContextOverflow
       "branch too long: {len(branch_k)} tokens with a {Ls}-token state (row limit 73728)"
     row_k = S_ids + branch_k, positions 0 .. L_k - 1; decide_k = L_k - 1 (the <decide> token);
       opts_k[j] = the index of option j's </opt> token in row_k

4. Graph (`decide.py`; the bundle's metadata.json `decision.readout`)

     one function `main`; states keyCache / valueCache / convState / recState, fp16, all zero for a new row (the KV
       sequence axis allocated at max_context_length 4096); output hidden [1, s, d] fp16 = the final-norm hidden
       state of every position of the call. Its call lengths (`graph_shape` on metadata.json's language block):
         static S  language.prefill_chunk = S: input_ids [1, S], every call S ids     cap = L = q = S
         dynamic   language.query_len_range = [qmin, cap] (round 14, input_ids [1, -1]: the graph takes qmin..cap),
                   language.query_len_call_max = L (round 15; absent = cap: the longest call a host makes, <= cap) and
                   language.query_len_multiple = q (round 15; absent = 1): every call a multiple of q in qmin..L
     `plan(n, L, q, qmin)`: an n-id run is cut into pieces of L ids, the remainder last; the last piece is padded
       with 248044 up to the next multiple of q and its padded positions' rows are dropped (causal: they cannot reach
       a real position). At q = 1 nothing is padded: a 1-id remainder takes one id from the piece before it (no call
       shorter than qmin = 2). A static-S bundle is the case L = q = S: ceil(T / S) calls of S, the last padded.
     a call of c ids (r of them real) after p earlier real ids gets input_ids [1, c] and position_ids [1, p + c] =
       0 .. p + c - 1
     graph limit: the padded end ceil(T / q) * q <= max_context_length - 1 = 4,095 (the position axis' upper bound),
       so T <= 4,080 at q = 16 and T <= 4,095 at q = 1 (`graph_context_check`; the author's serving limits above are
       far larger)
     shared prefix (`shared_prefix_plan`): the first k = floor(Ls / q) * q row tokens are state tokens for every
       question; run them once (`plan(k)`: no pad, k and L being multiples of q), copy the four states per question
       (the KV axis is only written up to k), then run each question's remaining tokens with `plan` from position k
       (k = 0: every row direct; q = 1 and Ls < qmin: direct). At cap = q = S the calls and their inputs are the
       direct run's, so the hidden rows are the same; on a dynamic bundle the shared calls are cut at other places
       than the direct run's, so its hidden rows (and p, by ~1e-3) can differ from a direct run's in the last bits.
     `call_lengths(L, q, qmin)`: every length a plan can produce (q, 2q, ..., L; at q = 1 qmin..L): a host may
       run each once from zero states before its first request (`warm_up`), so no request pays a length's one-time
       specialization.

5. Head (`kev.model.PointerHead`; the bundle's head/head.safetensors + head/kev_head.json)

     q = Wq h_decide + bq; k_j = Wk h_opt_j + bk  (Wq, Wk [256, d], biases [256], fp32 in the file)
     z_j = (k_j . q) * 0.0625;  p = softmax_j(z / T)  (T = kev_head.json temperature: 0.8B 2.3510958125672174,
       4B 2.406050072164233); h = the graph's fp16 hidden rows at decide / opts
     This file computes the head and the softmax in float64 (the fp16 hidden and the fp32 weights converted
     exactly) and rounds p to fp32 once at the end, so two hosts agree bit for bit whatever their summation order.
     The author's PyTorch head is fp32; the two differ by fp32 rounding (decide.py check records it).

6. Answers (`kev.api.to_answers`, `choice_confidence`, `score_confidence`, `round_prob`)

     noul   {"type": "noul", "noul": round4(p[1])}
     choice {"type": "choice", "choice": keys[first argmax p], "confidence": round4((max(n) - 1/K) / (1 - 1/K)),
             "probabilities": {key: round4(p)}}                                     (confidence 1.0 when K = 1)
     score  {"type": "score", "score": round4(sum_i i * p_i), "legend": legend, "probabilities": {"i": round4(p_i)},
             "confidence": round4(max(0, 1 - sum_i n_i |i - mode| / D))}             (confidence 1.0 when L = 1)
       n = p / sum(p) (all zeros -> uniform); mode = the first argmax of n; D = sum_i |i - (L - 1) / 2| / L
       p = the fp32 probabilities as doubles; round4 = Python round(x, 4) (the exact binary value, ties to even)
       every sum over floats is Python 3.12's sum(): left to right with Neumaier compensation (`py_sum`; the
       author's oracle ran Python 3.12, and Python 3.11's plain sum can differ in the last bit)

7. Response (`kev.serve.Server._body`)

     {"model": <name>, "answers": {qid: answer} (request order),
      "usage": {"input_tokens": len(packed ids), "output_tokens": len(encode(json.dumps(answers)))},
      "latency_ms": <graph + head wall ms, 1 decimal>}
     output_tokens encodes Python's json.dumps(answers) with its defaults (", " and ": " separators, ensure_ascii:
     non-ASCII as \\uXXXX, floats as repr) with the plain tokenizer (special tokens matched, no user_tokens rewrite).
     The author's body echoes the request's `model`; `decide.py` puts the bundle name there.
"""
from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

QUESTION_TYPES = ("noul", "choice", "score")
MAX_OPTIONS = 255
DEFAULT_MODEL = "kev-latest"
SERVE_MAX_STATE = 65536                     # kev.model.SERVE_MAX_STATE (the <state> token included)
SERVE_MAX_BRANCH = SERVE_MAX_STATE + 8192   # kev.model.SERVE_MAX_BRANCH: one row = state + its branch
DELIMITER_TOKENS = {"state": "<|fim_prefix|>", "q": "<|fim_middle|>", "opt": "<|box_start|>",
                    "opt_end": "<|box_end|>", "decide": "<|fim_suffix|>"}
PAD_TOKEN = "<|endoftext|>"
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


# --------------------------------------------------------------------------- #
# 1. Request
# --------------------------------------------------------------------------- #
def _is_json(v: Any) -> bool:
    """kev.api.JSONContent = Union[str, dict, list, int, float, bool, None] (containers are not looked into)."""
    return v is None or isinstance(v, (str, int, float, bool, dict, list))


def validate_request(request: Any) -> dict:
    """The decisions `SystemOneRequest.model_validate` makes -> {"model", "state", "questions": [(qid, q)]}.

    Raises ValueError for: a request that is not an object; no `state`; a `model` that is not a string; `questions`
    not an object or empty; a question that is not an object or whose `type` is not noul / choice / score;
    `instructions` or a criteria value that is not JSON; a noul `criteria` that is neither an object nor null; a
    choice without an object `criteria` of 1..255 entries; a score without an array `criteria` of 1..255 items.
    """
    if not isinstance(request, dict):
        raise ValueError("the request must be a JSON object")
    if "state" not in request:
        raise ValueError("state: field required")
    if not _is_json(request["state"]):
        raise ValueError("state: not a JSON value")
    model = request.get("model", DEFAULT_MODEL)
    if not isinstance(model, str):
        raise ValueError("model: must be a string")
    questions = request.get("questions")
    if not isinstance(questions, dict):
        raise ValueError("questions: must be an object of question id -> question")
    if len(questions) < 1:
        raise ValueError("questions: at least one question is required")
    out = []
    for qid, q in questions.items():
        if not isinstance(qid, str):
            raise ValueError(f"questions: key {qid!r} is not a string")
        if not isinstance(q, dict):
            raise ValueError(f"questions.{qid}: must be an object")
        kind = q.get("type")
        if not isinstance(kind, str) or kind not in QUESTION_TYPES:
            raise ValueError(f"questions.{qid}.type: must be one of 'noul', 'choice', 'score'")
        instructions = q.get("instructions")
        if not _is_json(instructions):
            raise ValueError(f"questions.{qid}.instructions: not a JSON value")
        criteria = q.get("criteria")
        if kind == "noul":
            if criteria is not None and not isinstance(criteria, dict):
                raise ValueError(f"questions.{qid}.criteria: a noul question takes an object or null")
        elif kind == "choice":
            if "criteria" not in q:
                raise ValueError(f"questions.{qid}.criteria: field required")
            if not isinstance(criteria, dict):
                raise ValueError(f"questions.{qid}.criteria: a choice question takes an object")
            if not 1 <= len(criteria) <= MAX_OPTIONS:
                raise ValueError(f"questions.{qid}.criteria: must have 1..{MAX_OPTIONS} options")
        else:
            if "criteria" not in q:
                raise ValueError(f"questions.{qid}.criteria: field required")
            if not isinstance(criteria, list):
                raise ValueError(f"questions.{qid}.criteria: a score question takes an array")
            if not 1 <= len(criteria) <= MAX_OPTIONS:
                raise ValueError(f"questions.{qid}.criteria: must have 1..{MAX_OPTIONS} levels")
        if isinstance(criteria, dict):
            for name, value in criteria.items():
                if not isinstance(name, str) or not _is_json(value):
                    raise ValueError(f"questions.{qid}.criteria: {name!r} is not a string key with a JSON value")
        elif isinstance(criteria, list) and not all(_is_json(v) for v in criteria):
            raise ValueError(f"questions.{qid}.criteria: an item is not a JSON value")
        out.append((qid, {"type": kind, "instructions": instructions, "criteria": criteria}))
    return {"model": model, "state": request["state"], "questions": out}


# --------------------------------------------------------------------------- #
# 2. Text
# --------------------------------------------------------------------------- #
def render(v: Any, indent: int = 0) -> str:
    """kev.api.render: str | object | array -> the text the model reads (field names kept as labels)."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
                     for k, x in v.items())


def option_text(name: str, desc: Any) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def question_keys(qtype: str, criteria: Any) -> list[str]:
    """The keys a question's probabilities are reported under, in option order."""
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def to_record(req: dict) -> tuple[dict, list[dict]]:
    """kev.api.to_record on a validated request -> (record for encode, per-question meta: id, type, keys, legend)."""
    qs, meta = [], []
    for qid, q in req["questions"]:
        m = {"id": qid, "type": q["type"], "keys": question_keys(q["type"], q["criteria"])}
        if q["type"] == "noul":
            c = q["criteria"] or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        elif q["type"] == "choice":
            opts = [option_text(k, v) for k, v in q["criteria"].items()]
        else:
            opts = [render(x) for x in q["criteria"]]
            m["legend"] = dict(zip(m["keys"], opts))
        qs.append({"instr": render(q["instructions"]), "options": opts})
        meta.append(m)
    return {"state": render(req["state"]), "questions": qs}, meta


# --------------------------------------------------------------------------- #
# 3. Tokens
# --------------------------------------------------------------------------- #
def load_tokenizer(path: str | Path):
    """The bundle's tokenizer.json as a `tokenizers.Tokenizer`."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def token_ids(tok, text: str) -> list[int]:
    """`text` -> ids without special tokens, for a `tokenizers.Tokenizer` or a transformers tokenizer."""
    if hasattr(tok, "encode_batch") and hasattr(tok, "token_to_id"):        # tokenizers.Tokenizer
        return list(tok.encode(text, add_special_tokens=False).ids)
    return list(tok(text, add_special_tokens=False).input_ids)


def special_id(tok, token: str) -> int:
    i = tok.token_to_id(token) if hasattr(tok, "token_to_id") else tok.convert_tokens_to_ids(token)
    if i is None or i < 0:
        raise ValueError(f"{token} is not in the tokenizer")
    return int(i)


def delimiter_ids(tok) -> dict:
    """The five delimiter ids and the pad id, looked up by token text (the author's convert_tokens_to_ids)."""
    out = {name: special_id(tok, t) for name, t in DELIMITER_TOKENS.items()}
    out["pad"] = special_id(tok, PAD_TOKEN)
    return out


def user_tokens(tok, text: str) -> list[int]:
    """kev.model.user_tokens: `<|name|>` -> `<¦name¦>`, then tokenized without special tokens."""
    return token_ids(tok, _SPECIAL_RE.sub("<¦\\1¦>", text))


class ContextOverflow(ValueError):
    """kev.model.ContextOverflow: a record over the serving limits (the author's server answers 422)."""

    def __init__(self, message: str, state_tokens: int | None = None, max_state: int | None = None):
        super().__init__(message)
        self.state_tokens, self.max_state = state_tokens, max_state


def encode(tok, rec: dict, max_state: int = SERVE_MAX_STATE, max_branch: int = SERVE_MAX_BRANCH,
           delims: dict | None = None) -> dict:
    """kev.model.encode, strict (a state is never cut): the packed record.

    -> ids, seg (0 = state, k = question k), pos (each branch's positions continue the state's), decide_idx [Q],
    opt_idx [Q][K] (packed indices of <decide> and of every </opt>), state_tokens (Ls, the <state> token included).
    """
    d = delims or delimiter_ids(tok)
    state_tokens = user_tokens(tok, rec["state"])
    if len(state_tokens) + 1 > max_state:
        raise ContextOverflow(f"state exceeds {max_state} tokens: {len(state_tokens) + 1}",
                              state_tokens=len(state_tokens) + 1, max_state=max_state)
    S = [d["state"]] + state_tokens
    ids, seg, pos = list(S), [0] * len(S), list(range(len(S)))
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [d["q"]] + user_tokens(tok, q["instr"])
        spans = [[d["opt"]] + user_tokens(tok, o) + [d["opt_end"]] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d["decide"]]
        if len(br) > max_branch - len(S):
            raise ContextOverflow(f"branch too long: {len(br)} tokens with a {len(S)}-token state (row limit {max_branch})")
        base = len(ids)
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp)
            ends.append(cursor - 1)
        ids += br
        seg += [k] * len(br)
        pos += list(range(len(S), len(S) + len(br)))
        decide_idx.append(base + len(br) - 1)
        opt_idx.append([base + e for e in ends])
    return {"ids": ids, "seg": seg, "pos": pos, "decide_idx": decide_idx, "opt_idx": opt_idx,
            "state_tokens": len(S)}


def admit(tok, rec: dict, delims: dict | None = None) -> dict:
    """kev.model.admit: encode within the serving limits, the state's overflow message reworded as the author's."""
    try:
        return encode(tok, rec, delims=delims)
    except ContextOverflow as e:
        if e.max_state is None:
            raise
        raise ContextOverflow(f"state is {e.state_tokens:,} tokens, over the {e.max_state:,}-token limit (the <state> "
                              "token included): shorten the document or split it across requests",
                              state_tokens=e.state_tokens, max_state=e.max_state) from None


def rows_of(enc: dict) -> tuple[list[int], list[int], list[dict]]:
    """kev.model.rows_of: (state ids, state positions, per question {ids, pos, decide, opts} within its branch)."""
    seg = enc["seg"]
    Ls = seg.count(0)
    rows, start = [], Ls
    for k, (d, oi) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
        end = d + 1
        if seg[start] != k or seg[end - 1] != k:
            raise ValueError("branch layout mismatch")
        rows.append({"ids": enc["ids"][start:end], "pos": enc["pos"][start:end], "decide": d - start,
                     "opts": [o - start for o in oi]})
        start = end
    return enc["ids"][:Ls], enc["pos"][:Ls], rows


def build_rows(request: Any, tok, delims: dict | None = None) -> dict:
    """request -> {"model", "ids" (packed), "state_len", "rows": [{qid, type, keys, row_ids, decide, opts, legend?}],
    "input_tokens", "meta"}. decide / opts index row_ids (the row = the state ids + the question's branch)."""
    req = validate_request(request)
    rec, meta = to_record(req)
    enc = admit(tok, rec, delims=delims)
    state_ids, _, branches = rows_of(enc)
    Ls = len(state_ids)
    rows = []
    for m, b in zip(meta, branches):
        row = {"qid": m["id"], "type": m["type"], "keys": m["keys"], "row_ids": state_ids + b["ids"],
               "decide": Ls + b["decide"], "opts": [Ls + o for o in b["opts"]]}
        if "legend" in m:
            row["legend"] = m["legend"]
        rows.append(row)
    return {"model": req["model"], "ids": enc["ids"], "state_len": Ls, "rows": rows,
            "input_tokens": len(enc["ids"]), "meta": meta}


# --------------------------------------------------------------------------- #
# 4. Graph limits and chunk plans
# --------------------------------------------------------------------------- #
def chunk_calls(T: int, chunk: int) -> int:
    return -(-T // chunk)


def graph_shape(language: dict, call_max: int | None = None, multiple: int | None = None) -> dict:
    """metadata.json's language block -> {"dynamic", "graph_max", "cap", "q", "qmin"}; `cap` is the plan's longest call
    L. A static-S bundle (`prefill_chunk` S) is graph_max = cap = q = qmin = S. A dynamic one (`query_len_range`
    [qmin, graph_max]) has cap = `query_len_call_max` (absent: graph_max) and q = `query_len_multiple` (absent: 1);
    `call_max` / `multiple` override the metadata's (a host's choice within the graph's range)."""
    if "query_len_range" in language:
        qmin, graph_max = (int(x) for x in language["query_len_range"])
        cap = int(call_max if call_max is not None else language.get("query_len_call_max", graph_max))
        q = int(multiple if multiple is not None else language.get("query_len_multiple", 1))
        if not (1 <= qmin <= cap <= graph_max) or q < 1 or cap % q or (q > 1 and q < qmin):
            raise ValueError(f"language block: query_len_range {language['query_len_range']}, call max {cap}, multiple {q} "
                             f"(want qmin <= call max <= the range's max, the call max a multiple of q, q = 1 or >= qmin)")
        return {"dynamic": True, "graph_max": graph_max, "cap": cap, "q": q, "qmin": qmin}
    if call_max is not None or multiple is not None:
        raise ValueError("call_max / multiple: a dynamic-S bundle's (query_len_range) only")
    S = int(language["prefill_chunk"])
    return {"dynamic": False, "graph_max": S, "cap": S, "q": S, "qmin": S}


def plan(n: int, cap: int, q: int, qmin: int = 1) -> list[tuple[int, int]]:
    """An n-id run as graph calls -> [(call length, real ids in it)]: pieces of cap ids (the call max L), the remainder
    last, the last piece padded up to the next multiple of q (q = 1: no padding, a remainder below qmin takes ids from
    the piece before it). Every call's length is a multiple of q in max(q, qmin)..cap; only the last one holds a pad."""
    if n < 1:
        raise ValueError(f"a run of {n} ids")
    if cap % q or (q > 1 and q < qmin):
        raise ValueError(f"cap {cap}, q {q}, qmin {qmin}: cap must be a multiple of q, and q > 1 at least qmin")
    full, rem = divmod(n, cap)
    if q == 1:
        sizes = [cap] * full + ([rem] if rem else [])
        if sizes[-1] < qmin:
            if len(sizes) == 1:
                raise ValueError(f"a run of {n} ids is shorter than the graph's smallest call ({qmin})")
            sizes[-2] -= qmin - sizes[-1]
            sizes[-1] = qmin
        return [(x, x) for x in sizes]
    out = [(cap, cap)] * full
    if rem:
        out.append((-(-rem // q) * q, rem))
    return out


def call_lengths(cap: int, q: int, qmin: int = 1) -> list[int]:
    """Every call length `plan` can produce: q, 2q, ..., cap (at q = 1: qmin..cap)."""
    return list(range(max(qmin, 1), cap + 1)) if q == 1 else list(range(q, cap + 1, q))


def padded_end(T: int, cap: int, q: int, qmin: int = 1) -> int:
    """The last position + 1 a row of T ids writes, pad included (= the position_ids length of its last call)."""
    return sum(c for c, _ in plan(T, cap, q, qmin))


def graph_context_check(rows: list[dict], max_ctx: int = 4096, chunk: int = 16, q: int | None = None,
                        qmin: int | None = None) -> None:
    """Every row fits the graph: its padded end <= max_ctx - 1 (the position axis' upper bound); `chunk` is the
    cap, q and qmin default to it (a static-S bundle). The author's serving limits (65,536 / 73,728) are checked by
    `admit`; this is the exported graph's own limit."""
    q = chunk if q is None else q
    qmin = q if qmin is None else qmin
    for r in rows:
        T = len(r["row_ids"])
        padded = padded_end(T, chunk, q, qmin)
        if padded > max_ctx - 1:
            raise ValueError(f"question {r['qid']!r}: a row of {T} tokens runs {padded} padded positions, over the "
                             f"graph's {max_ctx - 1} (rows of at most {(max_ctx - 1) // q * q} tokens)")


def shared_prefix_plan(state_len: int, chunk: int, q: int | None = None, qmin: int | None = None) -> dict:
    """The tokens every question's row shares, run once: k = floor(Ls / q) whole multiples of q (q defaults to the
    cap `chunk`: whole chunks of a static-S bundle), no pad; k = 0 when they would be shorter than the graph's
    smallest call."""
    q = chunk if q is None else q
    qmin = q if qmin is None else qmin
    k = state_len // q
    if k * q < qmin:
        k = 0
    return {"state_len": state_len, "k": k, "shared_tokens": k * q}


# --------------------------------------------------------------------------- #
# 5. Head
# --------------------------------------------------------------------------- #
_SAFETENSORS_DTYPES = {"F32": "<f4", "F16": "<f2", "BF16": None, "F64": "<f8", "I32": "<i4", "I64": "<i8"}


def read_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """A .safetensors file -> {name: array}: u64 little-endian header size, the JSON header, then the raw data
    (offsets relative to the end of the header)."""
    raw = Path(path).read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dt = _SAFETENSORS_DTYPES.get(info["dtype"])
        if dt is None:
            raise ValueError(f"{path}: {name} has dtype {info['dtype']}")
        a, b = info["data_offsets"]
        out[name] = np.frombuffer(raw[base + a:base + b], dtype=dt).reshape(info["shape"]).copy()
    return out


def load_head(head_dir: str | Path) -> dict:
    """head.safetensors (q / k weight [256, d] and bias [256], fp32) + kev_head.json (scale, temperature)."""
    head_dir = Path(head_dir)
    info = json.loads((head_dir / "kev_head.json").read_text())
    t = read_safetensors(head_dir / "head.safetensors")
    head = {"scale": float(info["scale"]), "temperature": float(info["temperature"]),
            "head_dim": int(info["head_dim"]), "hidden_size": int(info["hidden_size"])}
    for n in ("q.weight", "q.bias", "k.weight", "k.bias"):
        a = t[n]
        if a.dtype != np.float32:
            raise ValueError(f"{n}: {a.dtype}, the head is fp32")
        head[n] = a.astype(np.float64)
    if head["q.weight"].shape != (head["head_dim"], head["hidden_size"]) or head["k.weight"].shape != head["q.weight"].shape:
        raise ValueError("head weights do not match kev_head.json")
    if abs(head["scale"] - head["head_dim"] ** -0.5) > 1e-12:
        raise ValueError("scale != 1 / sqrt(head_dim)")
    return head


def head_logits(h_dec: np.ndarray, h_opts: np.ndarray, head: dict) -> np.ndarray:
    """z [K] (float64, before the temperature): ((Wk h_opt + bk) . (Wq h_decide + bq)) * scale."""
    hd = np.asarray(h_dec).astype(np.float64)
    ho = np.asarray(h_opts).astype(np.float64)
    q = head["q.weight"] @ hd + head["q.bias"]
    k = ho @ head["k.weight"].T + head["k.bias"]
    return (k @ q) * head["scale"]


def head_probs(z: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(z / T) over one question's options, float64, rounded to fp32 once."""
    zt = np.asarray(z, np.float64) / float(temperature)
    e = np.exp(zt - zt.max())
    return (e / e.sum()).astype(np.float32)


# --------------------------------------------------------------------------- #
# 6. Answers
# --------------------------------------------------------------------------- #
def py_sum(values) -> float:
    """Python 3.12's sum() of floats (bltinmodule.c): start 0, then left to right with Neumaier's compensation,
    the compensation added at the end when it is nonzero and finite. Python 3.11's sum() adds plainly."""
    it = iter(values)
    try:
        first = next(it)
    except StopIteration:
        return 0
    f = 0 + first
    c = 0.0
    for x in it:
        x = float(x)
        t = f + x
        if abs(f) >= abs(x):
            c += (f - t) + x
        else:
            c += (x - t) + f
        f = t
    if c and math.isfinite(c):
        f += c
    return f


def _normalize(p: list[float]) -> list[float]:
    t = py_sum(p)
    return [1 / len(p)] * len(p) if t == 0 else [x / t for x in p]


def choice_confidence(p: list[float]) -> float:
    """(p_max - 1/K) / (1 - 1/K): 0 at uniform, 1 at certainty; 1 when K = 1."""
    K = len(p)
    return 1.0 if K == 1 else (max(_normalize(p)) - 1 / K) / (1 - 1 / K)


def score_confidence(p: list[float]) -> float:
    """max(0, 1 - E|level - mode| / D), D = the mean |i - (L-1)/2| of a uniform distribution; 1 when L = 1."""
    L = len(p)
    if L == 1:
        return 1.0
    p = _normalize(p)
    mode = max(range(L), key=p.__getitem__)
    D = py_sum(abs(i - (L - 1) / 2) for i in range(L)) / L
    return max(0.0, 1.0 - py_sum(pi * abs(i - mode) for i, pi in enumerate(p)) / D)


def round_prob(x: float) -> float:
    return round(float(x), 4)


def to_answers(probs: list[list[float]], meta: list[dict]) -> dict[str, Any]:
    """kev.api.to_answers: per question its SystemOne answer (probs = each question's p in option order)."""
    out = {}
    for p, m in zip(probs, meta):
        p = [float(v) for v in p]
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": round_prob(p[1])}
        elif m["type"] == "choice":
            dist = {k: round_prob(v) for k, v in zip(m["keys"], p)}
            out[m["id"]] = {"type": "choice", "choice": m["keys"][max(range(len(p)), key=lambda i: p[i])],
                            "confidence": round_prob(choice_confidence(p)), "probabilities": dist}
        else:
            score = py_sum(i * pi for i, pi in enumerate(p))
            out[m["id"]] = {"type": "score", "score": round_prob(score), "legend": m["legend"],
                            "probabilities": {str(i): round_prob(v) for i, v in enumerate(p)},
                            "confidence": round_prob(score_confidence(p))}
    return out


# --------------------------------------------------------------------------- #
# 7. Response
# --------------------------------------------------------------------------- #
def output_tokens(tok, answers: dict) -> int:
    """kev.api.output_tokens: tokens of json.dumps(answers) (Python's defaults), plain tokenizer."""
    return len(token_ids(tok, json.dumps(answers)))


def response(model_name: str, answers: dict, input_tokens: int, output_tokens_: int, latency_ms: float) -> dict:
    """The /v1/systemone body (kev.serve.Server._body without the truncation fields a default server never sends)."""
    return {"model": model_name, "answers": answers,
            "usage": {"input_tokens": int(input_tokens), "output_tokens": int(output_tokens_)},
            "latency_ms": round(float(latency_ms), 1)}
