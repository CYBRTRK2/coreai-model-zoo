#!/usr/bin/env python3
"""clef-flash host reference: a SystemOne-shaped request -> decoder ids, spans and static inputs; image -> tower patches.

NumPy + Pillow + a tokenizer only (a `tokenizers.Tokenizer` from the snapshot's `tokenizer.json`, or a
transformers tokenizer built from the same file). This is the spec a Swift host reproduces, gated
before any Swift exists by `test_host.py` against the round-1 oracle (ids, spans, option order,
M-RoPE planes, the decoder's static inputs, the processor's `pixel_values`).

Token path = the checkpoint's own `encode_record()` (joint_schema_model.py) written out on the
tokenizer alone. Each piece is tokenized on its own, without special tokens, and concatenated:

    prefix   "<|im_start|>system\\n{SYSTEM_PROMPT}<|im_end|>\\n<|im_start|>user\\nSTATE:\\n"     36 ids
    [image]  "<|vision_start|>" + N x "<|image_pad|>" + "<|vision_end|>\\n"                 N + 3 ids
    state    render(state): a string as is, anything else compact JSON (sorted keys, ensure_ascii off)
    schema   "\\n\\nSCHEMA FIELDS:\\n", then per question
             "\\nFIELD {i}\\nID: {id}\\nTYPE: {type}\\nINSTRUCTION: " + [render(instructions or id)]
             + "\\nALLOWED OPTIONS:\\n" + per option "OPTION {j}: " + [render({"option_id", "description"})]
             + "\\n", and "END FIELD\\n"                                  [..] = the head's spans
    suffix   "\\n<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nJOINT SCHEMA DECISIONS:"  18 ids

Option order (the head scores options in this order; probabilities are a softmax per question):
noul = (true, false) with the author's default criteria overridable; choice = option ids sorted as
strings; score = levels 0..n-1. The state is cut to fit `max_length` (the author's 16384) exactly as
`encode_record` cuts it; the decoder graph's own context (4096) is the caller's check.

Decoder static inputs (`static_inputs`, the NumPy form of `host_static_inputs`): the N image-pad ids
become V + k (k = 0..N-1, row-major over the merged grid), image_rc[k] = (k // W, k % W),
rope_shift_start = i0 + 1 + N (i0 = 36, the <|vision_start|> index), rope_shift_amount = N - max(H, W);
text-only: image_rc 0, start 1 << 30, amount 0.

Image path (one fixed tile per tower graph: 256 / 448 / 672 / 896 px = 8 / 14 / 21 / 28 merged rows):

    RGB -> BICUBIC resize to tile x tile (aspect NOT kept) -> /255 -> (x - 0.5) / 0.5
        -> merge-block-major patchify -> patches [4 G^2, 1536] float32, vector layout (C, T=2, 16, 16)

`resize="pil"` is Pillow's own resize (what the oracle's fixed-grid arms fed the processor);
`resize="numpy"` is the same filter written out in Pillow's pass order (horizontal first, uint8
between the passes) — the form a non-Pillow host copies. `resize_bicubic` and `patchify` are copied
from conversion/decider_vision/host.py and _smoke/lfm25vl_preprocess.py (same repository).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

PATCH = 16
MERGE = 2
TEMPORAL = 2
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5

VOCAB = 248320                      # text_config.vocab_size; image rows are ids VOCAB + k
VISION_START, VISION_END, IMAGE_PAD = 248053, 248054, 248056
PAD_ID = 248044                     # <|endoftext|>: fills the last decoder chunk
N_IMAGE_MAX = 1024                  # rows of the decoder's static image buffer
NO_SHIFT = 1 << 30                  # rope_shift_start of a text-only row
MAX_LENGTH = 16384                  # the author's encode_record / systemone default
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
SCHEMA_HEADER = "\n\nSCHEMA FIELDS:\n"
NOUL_CRITERIA = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}


# --------------------------------------------------------------------------- #
# Request -> ids
# --------------------------------------------------------------------------- #
def render(value: Any) -> str:
    """A string as is; anything else as compact JSON (the author's `render`)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: dict) -> list[tuple[str, Any]]:
    """(option id, description) in the order the head scores them (the author's `question_options`)."""
    kind = str(question["type"])
    if kind == "noul":
        criteria = dict(NOUL_CRITERIA)
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if kind == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


def validate_request(request: dict) -> None:
    """The checks the author's `systemone()` makes before encoding."""
    questions = request.get("questions")
    if not isinstance(request.get("model"), str) or "state" not in request:
        raise ValueError("model and state are required")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("at least one question is required")
    for qid, q in questions.items():
        if q.get("type") not in QUESTION_TYPES:
            raise ValueError(f"{qid}: type must be noul, choice, or score")
        if q["type"] != "noul" and not q.get("criteria"):
            raise ValueError(f"{qid}: criteria must not be empty")


def tokens(tok, text: str) -> list[int]:
    """`text` -> ids without special tokens, for a `tokenizers.Tokenizer` or a transformers tokenizer."""
    if hasattr(tok, "encode_batch"):                       # tokenizers.Tokenizer
        return list(tok.encode(text, add_special_tokens=False).ids)
    return list(tok(text, add_special_tokens=False).input_ids)


def image_block(tok, n: int) -> list[int]:
    """The processor's media ids for one image of n merged tokens: <|vision_start|>, n pads, <|vision_end|>, \\n."""
    ids = tokens(tok, "<|vision_start|>" + "<|image_pad|>" * n + "<|vision_end|>\n")
    if ids[:1] != [VISION_START] or ids[1:n + 1] != [IMAGE_PAD] * n or ids[n + 1] != VISION_END or len(ids) != n + 3:
        raise ValueError("the tokenizer does not keep the image placeholder tokens whole")
    return ids


def build_ids(request: dict, tok, grid: tuple[int, int] | None = None, max_length: int = MAX_LENGTH,
              n_image_max: int = N_IMAGE_MAX) -> dict:
    """One request -> the decoder row and what the head needs.

    `grid` = the merged (H, W) of the tower that produces the image rows (g256: (8, 8); a native
    image: the processor's grid // 2), or None for a text-only request. Returns
      ids          the processor-form ids (N x <|image_pad|>), what the head's lexical gather reads
      token_offset index of <|vision_start|> (= len(prefix) = 36), None without an image
      questions    [{question_id, type, type_id, question_span, option_spans, option_ids}] (spans [start, end))
      grid_hw      `grid`
      static       the decoder's four inputs: input_ids (V + k mapped), image_rc, rope_shift_start,
                   rope_shift_amount (int32 NumPy arrays; image_rc has n_image_max rows)
    """
    validate_request(request)
    schema = tokens(tok, SCHEMA_HEADER)
    questions = []
    for qi, (qid, q) in enumerate(request["questions"].items()):
        schema += tokens(tok, f"\nFIELD {qi + 1}\nID: {qid}\nTYPE: {q['type']}\nINSTRUCTION: ")
        q0 = len(schema)
        instructions = q.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(qid)
        schema += tokens(tok, render(instructions))
        q1 = len(schema)
        schema += tokens(tok, "\nALLOWED OPTIONS:\n")
        spans, oids = [], []
        for oi, (oid, desc) in enumerate(question_options(q)):
            schema += tokens(tok, f"OPTION {oi + 1}: ")
            o0 = len(schema)
            sem = {"option_id": oid}
            if desc is not None:
                sem["description"] = desc
            schema += tokens(tok, render(sem))
            spans.append([o0, len(schema)])
            oids.append(oid)
            schema += tokens(tok, "\n")
        schema += tokens(tok, "END FIELD\n")
        questions.append({"question_id": str(qid), "type": str(q["type"]), "type_id": QUESTION_TYPES[str(q["type"])],
                          "question_span": [q0, q1], "option_spans": spans, "option_ids": oids})
    prefix = tokens(tok, PREFIX)
    suffix = tokens(tok, SUFFIX)
    token_offset = None
    if grid is not None:
        h, w = (int(v) for v in grid)
        token_offset = len(prefix)
        prefix = prefix + image_block(tok, h * w)
    state = tokens(tok, render(request["state"]))
    fixed = len(prefix) + len(schema) + len(suffix)
    if fixed > max_length:
        raise ValueError(f"schema requires {fixed} tokens before state; maximum is {max_length}")
    state = state[: max_length - fixed]
    shift = len(prefix) + len(state)
    for q in questions:
        q["question_span"] = [q["question_span"][0] + shift, q["question_span"][1] + shift]
        q["option_spans"] = [[a + shift, b + shift] for a, b in q["option_spans"]]
    ids = prefix + state + schema + suffix
    return {"ids": ids, "token_offset": token_offset, "questions": questions,
            "grid_hw": tuple(grid) if grid is not None else None,
            "static": static_inputs(ids, grid, n_image_max=n_image_max)}


def static_inputs(ids, grid_hw, vocab: int = VOCAB, n_image_max: int = N_IMAGE_MAX) -> dict:
    """The decoder's input_ids / image_rc / rope_shift_start / rope_shift_amount for one row (NumPy
    form of `qwen3_5_vl_pipelined.host_static_inputs`; at most one image, one contiguous block)."""
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    rc = np.zeros((n_image_max, 2), dtype=np.int32)
    pads = np.flatnonzero(ids == IMAGE_PAD)
    if grid_hw is None:
        if pads.size:
            raise ValueError("image tokens in a text-only row")
        return {"input_ids": ids.astype(np.int32), "image_rc": rc,
                "rope_shift_start": np.array([NO_SHIFT], np.int32), "rope_shift_amount": np.array([0], np.int32)}
    h, w = (int(v) for v in grid_hw)
    n = h * w
    if n > n_image_max:
        raise ValueError(f"{h}x{w} = {n} image tokens > n_image_max {n_image_max}")
    starts = np.flatnonzero(ids == VISION_START)
    if starts.size != 1 or pads.size != n:
        raise ValueError(f"expected one image block of {n} tokens, got {starts.size} <|vision_start|> "
                         f"and {pads.size} <|image_pad|>")
    i0 = int(starts[0])
    if not np.array_equal(pads, np.arange(i0 + 1, i0 + 1 + n)):
        raise ValueError("image tokens are not one contiguous block after <|vision_start|>")
    k = np.arange(n)
    mapped = ids.copy()
    mapped[i0 + 1:i0 + 1 + n] = vocab + k
    rc[:n, 0] = k // w
    rc[:n, 1] = k % w
    return {"input_ids": mapped.astype(np.int32), "image_rc": rc,
            "rope_shift_start": np.array([i0 + 1 + n], np.int32),
            "rope_shift_amount": np.array([n - max(h, w)], np.int32)}


def rope_positions(mapped_ids, start: int, amount: int, merged_w: int, vocab: int = VOCAB) -> np.ndarray:
    """The three M-RoPE planes [3, T] the decoder derives in-graph from (ids, start, amount)."""
    ids = np.asarray(mapped_ids, dtype=np.int64)
    i = np.arange(len(ids), dtype=np.int64)
    pos = np.broadcast_to(i - amount * (i >= start), (3, len(ids))).copy()
    img = ids >= vocab
    k = ids[img] - vocab
    s0 = i[img] - k
    pos[0, img] = s0
    pos[1, img] = s0 + k // merged_w
    pos[2, img] = s0 + k % merged_w
    return pos


# --------------------------------------------------------------------------- #
# Image -> patches
# --------------------------------------------------------------------------- #
def tile_grid(tile: int) -> int:
    """Merged rows per side of a square tile (256 -> 8, 448 -> 14, 672 -> 21, 896 -> 28)."""
    if tile % (PATCH * MERGE):
        raise ValueError(f"tile {tile} is not a multiple of {PATCH * MERGE}")
    return tile // (PATCH * MERGE)


def patchify(x: np.ndarray) -> np.ndarray:
    """Normalized [H, W, C] -> [gh*gw, C*T*P*P] in Qwen's merge-block-major order: patches iterate
    (block_row, block_col, y-in-block, x-in-block), channel outermost inside the vector, the still
    frame repeated at both temporal slots (the Qwen2VLImageProcessor reshape/permute)."""
    h, w, c = x.shape
    if h % (PATCH * MERGE) or w % (PATCH * MERGE):
        raise ValueError(f"{h}x{w} not divisible by patch*merge {PATCH * MERGE}")
    gh, gw = h // PATCH, w // PATCH
    t = x.transpose(2, 0, 1).reshape(c, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    t = t.transpose(1, 4, 2, 5, 0, 3, 6).reshape(gh * gw, 1, c, PATCH, PATCH)
    t = np.broadcast_to(t[:, :, :, None], (gh * gw, 1, c, TEMPORAL, PATCH, PATCH))
    return t.reshape(gh * gw, c * TEMPORAL * PATCH * PATCH)


def _bicubic_kernel(x: np.ndarray) -> np.ndarray:
    """Pillow's bicubic weight (a = -0.5) at distance x (already divided by the filter scale)."""
    x = np.abs(x)
    a = -0.5
    w = np.zeros_like(x)
    m1 = x < 1.0
    w[m1] = ((a + 2.0) * x[m1] - (a + 3.0)) * x[m1] * x[m1] + 1.0
    m2 = (x >= 1.0) & (x < 2.0)
    w[m2] = (((x[m2] - 5.0) * x[m2] + 8.0) * x[m2] - 4.0) * a
    return w


def _resample_axis(img: np.ndarray, out_size: int, axis: int) -> np.ndarray:
    """Pillow's antialiased bicubic along one axis of a float [H, W, C] image (support 2 x scale)."""
    img = np.moveaxis(img, axis, 0)
    in_size = img.shape[0]
    scale = in_size / out_size
    filterscale = max(1.0, scale)
    support = 2.0 * filterscale
    kmax = int(np.ceil(support) * 2) + 1
    starts = np.zeros(out_size, dtype=np.int64)
    weights = np.zeros((out_size, kmax), dtype=np.float64)
    for i in range(out_size):
        center = (i + 0.5) * scale
        xmin = int(max(0, np.floor(center - support + 0.5)))
        xmax = int(min(in_size, np.ceil(center + support + 0.5)))
        xs = np.arange(xmin, xmax)
        w = _bicubic_kernel((xs + 0.5 - center) / filterscale)
        total = w.sum()
        if total > 0:
            w = w / total
        starts[i] = xmin
        weights[i, : xs.size] = w
    padded = np.concatenate([img, np.zeros((kmax,) + img.shape[1:], img.dtype)], axis=0)
    taps = padded[starts[:, None] + np.arange(kmax)[None, :]]       # [out, k, ...]
    out = np.einsum("okyc,ok->oyc", taps, weights.astype(img.dtype))
    return np.moveaxis(out, 0, axis)


def resize_bicubic(u8: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Pillow's uint8 BICUBIC resize in NumPy, in Pillow's order: the HORIZONTAL pass first, the
    intermediate rounded and clipped to uint8, then the vertical pass."""
    x = np.asarray(u8, dtype=np.float64)
    if x.shape[1] != out_w:
        x = np.clip(np.floor(_resample_axis(x, out_w, axis=1) + 0.5), 0, 255)
    if x.shape[0] != out_h:
        x = np.clip(np.floor(_resample_axis(x, out_h, axis=0) + 0.5), 0, 255)
    return x.astype(np.uint8)


def preprocess(image, tile: int, resize: str = "pil") -> np.ndarray:
    """Image (path, PIL image or uint8 [H, W, 3]) -> patches [4 G^2, 1536] float32, G = tile // 32."""
    from PIL import Image

    tile_grid(tile)
    if isinstance(image, (str, Path)):
        im = Image.open(image)
    elif isinstance(image, Image.Image):
        im = image
    else:
        im = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if resize == "pil":
        px = np.asarray(im.convert("RGB").resize((tile, tile), Image.Resampling.BICUBIC))
    elif resize == "numpy":
        px = resize_bicubic(np.asarray(im.convert("RGB")), tile, tile)
    else:
        raise ValueError(f"resize {resize!r} (pil | numpy)")
    x = (px.astype(np.float64) / 255.0 - IMAGE_MEAN) / IMAGE_STD
    return patchify(x).astype(np.float32)
