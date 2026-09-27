# Community port — NOT an Apple model.
"""The Audio8-TTS host sampler in NumPy — the spec the Swift host is written from, gated against the oracle.

Mirrors `modeling_arktts.py` step for step (fp32):
  * `ArkttsLegacyTopKTopPLogitsProcessor`: sort descending, cumulative softmax, drop `cum > top_p` or rank >= top_k,
    always keep the best; then divide by `max(temperature, 1e-5)`.
  * `ArkttsModel._sample`: `argmax(softmax(scores) / -log(u))` with u ~ U(0,1) — a Gumbel-max draw whose randomness is
    the vector `u`, so a recorded `u` replays the publisher's choice exactly on identical logits.
  * `_sample_semantic` (RAS): draw `normal` at (top_p, temperature) and `high` at (ras_top_p 0.9, ras_temperature 1.0);
    if `normal` is a semantic id already in the 10-token window, emit `high`. The window is the publisher's: the
    first frame's semantic never enters it (the window is created as zeros after step 0), later frames roll in.
  * codebooks 1..9: plain top-k/top-p/temperature + the same Gumbel draw over 4096 logits.

Layout: the slow logits are the 4097 rows the sampler can ever pick — semantic ids 151678..155773 as indices
0..4095, eos 151645 as index 4096 (`allowed_ids()` in audio8_model.py; the same layout as the publisher's ONNX
`slow_logits_layout = semantic_then_eos`).
"""
from __future__ import annotations

import numpy as np

SEMANTIC_BEGIN = 151678
SEMANTIC_END = 155773
EOS = 151645
EOS_INDEX = 4096
RAS_TOP_P = 0.9
RAS_TEMPERATURE = 1.0
RAS_WINDOW = 10


def softmax32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = np.max(x)
    if not np.isfinite(m):
        m = np.float32(0.0)
    e = np.exp(x - m, dtype=np.float32)
    return e / np.sum(e, dtype=np.float32)


def top_k_top_p(scores: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")
    probs = softmax32(scores[order])
    cumulative = np.cumsum(probs, dtype=np.float32)
    remove = (cumulative > np.float32(top_p)) | (np.arange(scores.size) >= int(top_k))
    remove[0] = False
    out = scores.copy()
    out[order[remove]] = -np.inf
    return out


def processed(scores: np.ndarray, top_k: int, top_p: float, temperature: float) -> np.ndarray:
    return top_k_top_p(scores, top_k, top_p) / np.float32(max(float(temperature), 1e-5))


def gumbel_argmax(scores: np.ndarray, u: np.ndarray) -> int:
    probs = softmax32(scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        noise = -np.log(np.asarray(u, dtype=np.float32))
        return int(np.argmax(probs / noise))


def gumbel_margin(scores: np.ndarray, u: np.ndarray) -> float:
    """How decisive the draw was: (best - runner-up) / best of `softmax / -log(u)`; small = knife-edge."""
    probs = softmax32(scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = probs / -np.log(np.asarray(u, dtype=np.float32))
    v = np.where(np.isfinite(v), v, 0.0)
    top = np.sort(v)[-2:]
    return float((top[1] - top[0]) / top[1]) if top[1] > 0 else 0.0


def index_to_id(index: int) -> int:
    return EOS if index == EOS_INDEX else SEMANTIC_BEGIN + int(index)


def sample_semantic(logits4097: np.ndarray, u_normal: np.ndarray, u_high: np.ndarray, window: list[int] | None,
                    top_k: int, top_p: float, temperature: float) -> int:
    """Returns the sampled token id (a semantic id or EOS)."""
    normal = index_to_id(gumbel_argmax(processed(logits4097, top_k, top_p, temperature), u_normal))
    high = index_to_id(gumbel_argmax(processed(logits4097, top_k, RAS_TOP_P, RAS_TEMPERATURE), u_high))
    if window is None:
        return normal
    if SEMANTIC_BEGIN <= normal <= SEMANTIC_END and normal in window:
        return high
    return normal


def sample_codebook(logits4096: np.ndarray, u: np.ndarray, top_k: int, top_p: float, temperature: float) -> int:
    return gumbel_argmax(processed(logits4096, top_k, top_p, temperature), u)


class RASWindow:
    """The publisher's `previous` tensor: None before step 0, zeros(10) after it, then a rolling window."""

    def __init__(self):
        self.values: list[int] | None = None

    def push(self, semantic: int) -> None:
        if self.values is None:
            self.values = [0] * RAS_WINDOW
        else:
            self.values = self.values[1:] + [int(semantic)]
