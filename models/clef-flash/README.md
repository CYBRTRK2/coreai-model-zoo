# clef-flash — Core AI

[🤗 mlboydaisuke/clef-flash-CoreAI](https://huggingface.co/mlboydaisuke/clef-flash-CoreAI) · Apache-2.0 · source [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash/tree/17f0b0ad64efb65d273590632833508766b2aae6) (revision `17f0b0a`) · base Qwen/Qwen3.5-9B

A **decision model**. Give it a state (text, JSON or an image) and typed questions: `noul` (true /
false), `choice` (named options) or `score` (ordered levels). It returns a probability for every
allowed option of every question from one prefill. It never generates text. Requests and responses
use the SystemOne-compatible request shape: `{model, state, questions}` in, `{model, answers, usage}`
out, one typed decision per question.

Cloudflare post-trained it from Qwen3.5-9B. Their announcement: "By freezing Qwen3.8-27B for Clef and
Qwen3.5-9B for Clef-flash, we jointly optimized the routing head alongside rank-256 low-rank
adapters." The checkpoint ships the backbone (24 Gated DeltaNet + 8 full-attention layers, a
27-block vision tower) as sharded safetensors, plus a 121.8M-parameter joint schema head that reads
the backbone's final hidden states. The author's card reports benchmark results; none of them are
re-measured here.

This port runs that readout as three Core AI graphs and one host table. The **decoder** is the
Qwen3.5 hybrid with token ids in, the image rows as a static input and M-RoPE derived inside the
graph. It returns the final-norm hidden state at every position and has no vocabulary head. One
call takes 64 tokens. Two bundles: fp16 (15.9 GB, the default) and int8mix (int8 per block of 32
except layers 0–11, 11.8 GB). The **vision tower** is baked at a fixed grid, `g256` (64 image
tokens) or `g448` (196), fp16 weights read through a cast with fp32 math (909 / 912 MB). The
**head** is the author's joint schema head written as plain matrices, 244 MB, with four static key
lengths. The untied `lm_head.weight` stays on the host as a 2.03 GB fp16 gather table for the head's
lexical term. The gate is probability parity with the author's own fp32 code, on every option of
every question, on a self-made fixture and on a held-out record set written after the int8 layers
were chosen.

The port runs on the Mac GPU. iPhone is not a target: the decoder alone is 15.9 GB (fp16) or
11.8 GB (int8mix), and an iOS app had about 6.4 GB available on an iPhone 18 Pro with the
increased-memory-limit entitlement ([decider-2b-vision](../decider-2b-vision/README.md)).

## Readout contract

One record per request, every question answered in the same pass. The ids are the checkpoint's
`encode_record()`: prefix + [image block] + state + schema + suffix, each piece tokenized on its own
without special tokens, then concatenated:

```
<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n[<|vision_start|><N image tokens><|vision_end|>\n]<state>\n\nSCHEMA FIELDS:\n\nFIELD 1\nID: <question id>\nTYPE: <type>\nINSTRUCTION: <instructions>\nALLOWED OPTIONS:\nOPTION 1: {"description":"…","option_id":"…"}\nOPTION 2: …\nEND FIELD\n\nFIELD 2\n…END FIELD\n\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:
```

- The prefix is 36 tokens for every record and the suffix 18, ending in `:` (25). The image block
  starts at index 36: `<|vision_start|>` (248053), N image tokens, `<|vision_end|>` (248054), `\n`.
- `render()`: a string as is, anything else `json.dumps(sort_keys=True, ensure_ascii=False,
  separators=(",", ":"))`. The state is rendered this way; so is each question's instructions (the
  question id when there are none) and each option's `{"option_id", "description"}`.
- A question's span is its rendered instructions; an option's span is its rendered object. The head
  averages the hidden rows over each span, reads the last token as a global vector, and averages the
  `lm_head.weight` rows of each option span's ids (the lexical term).
- Option order: `noul` = (true, false), with the default descriptions "The proposition is true or the
  answer is yes." / "…false or the answer is no." unless the request overrides them; `choice` = option
  ids sorted as strings; `score` = levels 0..n−1.
- Probabilities: a float32 softmax per question over that question's options. Response: `noul` →
  p(true); `choice` → the argmax id, its p, every p; `score` → Σ level·p, the largest p, the legend,
  every p; all rounded to 4 decimals. `usage` = `{input_tokens: T, output_tokens: 0}`.
- Image: one image per request. RGB → Pillow BICUBIC resize to 256×256 or 448×448 (the aspect ratio is
  not kept) → /255 → (x − 0.5) / 0.5 → merge-block-major patches `[4G², 1536]`, the frame repeated at
  both temporal slots. The author's processor picks a grid per image (at least 256×256);
  [what a fixed grid costs](#what-a-fixed-grid-costs-the-authors-code-only) is measured below. The
  graph takes the image tokens as ids V + k (V = 248,320, k row-major over the merged grid), so each
  token names its tower row.
- Text-only record: no image block, the same bundle, zero image rows and `rope_shift_start` = 2³⁰.
- Length: a record and its padded last call must fit 4,096 tokens, the image block included. The
  author's own cut at 16,384 tokens is never reached.

`conversion/clef_flash/oracle_clef.py` runs the checkpoint's own `joint_schema_model.py`
(`load_release_model()` + `systemone()`, fp32, CPU) and records the reference:
[`fixtures-clef-flash.json`](fixtures-clef-flash.json). It holds 216 records with their requests,
ids, spans, option logits and probabilities: 186 written for the port (16 text states, three of
them about 2,000 tokens; 12 JSON states; 10 images drawn by `make_fixtures.py`; 4 CC0 photographs;
the 144 `choice` items of SemIf authored144) and 30 written later as a held-out set. Each image
record runs at `g256`, `g448` and `native` (the processor's own grid): 214 runs, 405 questions. The
gates use 213 runs and 401 questions: `photo_01` at `native` has 1,200 image rows, more than the
graph's 1,024-row buffer. Four questions have an oracle top-2 margin of 0.02 or less (near-ties) and
are listed apart. The held-out set runs text, `g256` and `g448`: 40 runs, 188 questions, 2 near-ties.

## Core AI shape

**Decoder.** `conversion/clef_flash/qwen3_5_clef_decoder.py`: decider-2b-vision's ids-input Qwen3.5
VL decoder (`conversion/decider_vision/qwen3_5_vl_pipelined.py`) at 9B, without the vocabulary head,
returning the hidden state at every position. One function, `main`, at a static 64 tokens; the Gated
DeltaNet recurrence is unrolled in the graph.

| | name | shape, type |
|---|---|---|
| inputs | `input_ids` | [1, 64] int32 |
| | `position_ids` | [1, seq] int32, the ramp 0 .. seq − 1 |
| static inputs | `image_embeds` | [1024, 4096] fp16: tower rows 0..N−1, the rest zero |
| | `image_rc` | [1024, 2] int32: (k // W, k % W) for image token k |
| | `rope_shift_start` | [1] int32: 37 + N, the `<\|vision_end\|>` index |
| | `rope_shift_amount` | [1] int32: N − max(H, W) |
| states | `keyCache`, `valueCache` | [8, 1, 4, ctx, 256] fp16, ctx up to 4,096 |
| | `convState`, `recState` | [24, 1, 8192, 3], [24, 1, 32, 128, 128] fp16 |
| output | `hidden` | [1, 64, 4096] fp16, every position |

Per record, from zeroed states: ⌈T / 64⌉ calls, call k with ids[64k : 64k + 64] and position_ids
0..64k + 63; the last call is padded with `<|endoftext|>` (248044) and its padded rows are dropped.
The rows of every call, cut to T, are the backbone's final-norm `last_hidden_state [T, 4096]`. The
bundle's `metadata.json` says `kind: decision-backbone` and carries this order, the prompt pieces,
the span rules, the option order and the response shape.

**Towers.** The overlay's `qwen3_5_vision.Qwen3_5VisionEncoder` through
`conversion/export_qwen38vl_pipelined.py --skip-decoder --vision-dtype fp16w32`, baked at 8×8 or
14×14 merged tokens: patches `[4G², 1536]` f32 → `image_embeds [G², 4096]` f32.

**Head.** `conversion/clef_flash/clef_head.py`, `ClefHeadGraph`: the author's `JointSchemaHead` (its
own submodules, `load_state_dict(strict=True)`) for one record, its Python loops over questions and
options written as matrices, fp16 weights read through a cast with fp32 math. Ten inputs: `hidden
[T, 4096]` f32, `key_valid [T]`, `q_avg [Q, T]` and `o_avg [O, T]` (1/len over each span), `g_avg
[1, T]` (the last real token), `lexical [O, 4096]`, `member [Q, O]`, `type_ids [Q]` int32, `q_valid
[Q]`, `o_valid [O]`; output `logits [O]` f32. Four functions share the weights, `t512`, `t1024`,
`t2048`, `t4096`; the host pads T to the smallest that holds it, Q to 16 and O to 128, with zero rows
and valid flags 0. Its `metadata.json` says `kind: decision-head`. Above 2,048 keys the head's
attention runs in key blocks of at most 2,048 (see [Precision](#precision)).

**Host.** Builds the ids and spans, preprocesses the image, runs the decoder calls, converts the
hidden rows to f32 and builds the ten head arrays (the lexical rows from `host/lm_head_fp16.bin`,
`[248320, 4096]` fp16, row t at byte offset t × 8192), then the per-question softmax and the
response. `conversion/clef_flash/decide.py` is the Python reference; `apps/ClefFlash` is the Swift
one.

No engine is involved: the hosts drive the low-level runtime (`AIModel` + `loadFunction`, four
zeroed states per record). No runtime patch.

## Measured (Apple M4 Max, macOS 27.0 26A428, 2026-10-03)

The bar, fixed before any graph ran: the argmax equal to the oracle's on every question whose oracle
top-2 margin is above 0.02 (near-ties listed apart), max |Δp| ≤ 0.02 over every option of every
question, the mean over runs of each run's mean |Δp| ≤ 0.002, and every process re-running its first
run bit for bit. The Python gates load AOT `.aimodelc` assets (`coreai-build compile … --platform
macOS --preferred-compute gpu --architecture h16c`, `--expect-frequent-reshapes` for the decoder) with
`SpecializationOptions.default()`; their times are on a GPU shared with other work.

### Before any graph: fp32 torch

The decoder module in fp32 on the CPU, driven like the graph (chunks from zeroed states, the unrolled
recurrence, M-RoPE in the module, the oracle's own tower rows), read through the author's head: 214
runs, argmax 405/405, max |Δp| 9.6e-6, lowest per-position hidden cosine 0.99999939; its M-RoPE
planes equal the oracle's on 214/214 runs. With zero image inputs its hidden state equals the
overlay's plain Qwen3.5 text decoder bit for bit (3/3 runs). Zeroing the image rows changes 2 of 3
answers on `img_07` at `g448` and on `img_01` at `g256` (max |Δp| 0.66 / 0.75)
([`gate-clef-flash-torch-parity.json`](gate-clef-flash-torch-parity.json)).

The host spec (`conversion/clef_flash/host.py`) rebuilds every oracle run's ids, image offset, spans,
option order and M-RoPE planes: 242/242 runs with two tokenizer implementations. Its Pillow-resized
patches equal the processor's `pixel_values` bit for bit on all 56 fixed-grid runs.

### Vision tower

The shipped fp16w32 tower, AOT for the Mac GPU, fed the host's patches, against the oracle's fp32
tower rows: `g256` 14/14 images, worst image cosine 0.9999999994, worst row 0.99999995; `g448`
14/14, 0.9999999994 and 0.99999995; 31 / 79 ms per image on the shared GPU. Patches in raster order
instead of merge-block order miss on every image ([`gate-clef-flash-tower.json`](gate-clef-flash-tower.json)).

### Decoder alone on the Mac GPU: 213 fixture runs

The oracle's ids and fp32 image rows in, the fp16 hidden read through the author's fp32 head:

| decoder | chunk | argmax (margin > 0.02) | near-ties agreeing | max \|Δp\| | mean of run means | bar |
|---|---:|---:|---:|---:|---:|---|
| **fp16 (default)** | 64 | **397/397** | **4/4** | **0.0112** | **0.00054** | **PASS** |
| **int8mix: int8 except layers 0–11** | 64 | **397/397** | **4/4** | **0.0153** | **0.00089** | **PASS** |
| int8lin: every decoder linear int8 | 64 | 397/397 | 3/4 | 0.0748 | 0.00192 | FAIL |
| fp16 | 16 | 397/397 | 4/4 | 0.0127 | 0.00053 | PASS |

int8 is per block of 32 (`symmetric_with_clipping`, weights only); the embedding table, the Gated
DeltaNet conv1d and every norm stay fp16. int8lin misses on 9 of 213 runs, worst a SemIf record at
0.0748 (fp16 on the same record: 0.0105), and flips the `img_06` `g448` near-tie (margin 0.0175).
The red arm, the image rows zeroed on `img_07` `g448` and `img_01` `g256`, moves p by 0.66 and 0.75
and changes one or two of the three answers on every bundle that ran it. Transcripts:
`gate-clef-flash-readout-<bundle>.json`.

### The head graph

The head graph alone on the oracle's fp32 hidden: 214 runs, argmax 405/405, max |Δp| 1.8e-7; held
out, 40 runs, 188/188, 1.8e-7. End to end in Python from the fp16 decoder's hidden through the host
arrays and the head graph: the same values as the author's fp32 head on that hidden (0.0112 fixture,
0.0085 held out), within 1.8e-7 / 2.4e-7. Three perturbed arms on 4 runs each move p: span rows one
token late by 0.003–0.056, the lexical term zeroed by 0.017–0.055, option membership rotated by
0.051–0.348. `decide.py` from the raw request, on 12 fixture and 12 held-out requests: ids and spans
equal the oracle's 24/24, every choice equal (18/18, 32/32), the 4-decimal values within 0.0082
([`gate-clef-flash-head.json`](gate-clef-flash-head.json)).

### Held out: 40 runs that no choice was made on

30 records written after the int8 layers were chosen, with new names, drawings and photographs and a
novelty check against the fixture (no shared record, image, name or 8-word run), and their own fp32
oracle run:

| decoder | argmax (margin > 0.02) | near-ties agreeing | max \|Δp\| | mean of run means | bar |
|---|---:|---:|---:|---:|---|
| fp16 (default) | 186/186 | 2/2 | 0.0085 | 0.00036 | PASS |
| int8mix | 186/186 | 1/2 | 0.0171 | 0.00063 | PASS |

int8mix flips the `ho_t07` near-tie (oracle margin 0.0117: p 0.4102 / 0.3985, int8mix 0.3979 /
0.4059) ([`gate-clef-flash-heldout.json`](gate-clef-flash-heldout.json)).

### What a fixed grid costs (the author's code only)

`conversion/clef_flash/grid_price.py` runs the checkpoint's own code on the 14 image records (50
questions), the image resized to a square grid first, against the processor's own grid (`native`),
fp32 on the CPU:

| arm | image tokens | argmax = native | max \|Δp\| vs native | mean \|Δp\| vs native | gold correct |
|---|---:|---:|---:|---:|---:|
| native | 412 (mean) | — | — | — | 35/38 |
| `g256` | 64 | 46/50 | 0.959 | 0.039 | 33/38 |
| `g448` | 196 | 48/50 | 0.436 | 0.018 | 36/38 |
| 672×672 | 441 | 48/50 | 0.350 | 0.016 | 36/38 |
| 896×896 | 784 | 48/50 | 0.244 | 0.013 | 36/38 |

The two larger grids agree with `native` no more often than `g448`, so only `g256` and `g448` ship
([`gate-clef-flash-variants.json`](gate-clef-flash-variants.json)).

### Swift ([`apps/ClefFlash`](../../apps/ClefFlash/), Mac, Release)

Images through the Swift tower; the 13 `native` fixture runs fed the oracle's rows:

| decoder asset | fixture 213: argmax / near-ties / max \|Δp\| / mean | held-out 40 |
|---|---|---|
| fp16, AOT `.aimodelc` | 397/397, 4/4, 0.0122, 0.00054 | 186/186, 2/2, 0.0082, 0.00037 |
| fp16, JIT: the `.aimodel` specialized by the Swift runtime | the same values | the same values |
| int8mix, AOT | 397/397, 4/4, 0.0153, 0.00089 | 186/186, 1/2, 0.0171, 0.00065 |
| int8mix, JIT | 397/397, 4/4, 0.0148, 0.00089 | 186/186, 1/2, 0.0173, 0.00065 |

- On identical inputs Swift computes what Python computes: ids and spans from the raw request
  253/253 runs; the renderer = CPython's `json.dumps` / `repr` / `round(x, 4)` on 2,204 values and
  199,994 doubles; tiles and patches = Pillow and the oracle's `pixel_values` on 48/48; tower rows =
  the Python runtime's on 48/48; with fp16 AOT, hidden rows and head logits bit-equal on 185 + 20
  runs and on the 48 runs fed the oracle's image rows (int8mix AOT: hidden rows bit-equal on the
  same runs), and responses = `decide.py`'s on 24/24. The image runs differ from the Python table
  above only through the tower rows (the Python readout fed the oracle's).
- JIT against AOT: fp16 bit-equal on 253/253 runs (hidden, logits, responses). int8mix is not
  bit-equal on any run: max |Δp| 0.0014, max |Δlogit| 0.020, argmax 589/589.
- The JIT's first specialization: fp16 56.1 s (the decoder 50.0 s), the runtime's cache +31.8 GB,
  4.5 s from the cache; int8mix 42.3 s (39.2 s), +17.0 GB, 3.0 s.
- Changing one word of one question moves the ids and spans off the oracle's and moves p (up to
  0.0056 on `own_t01`).

Transcript: [`gate-clef-flash-swift.json`](gate-clef-flash-swift.json).

### Time per decision

Swift, Release CLI, AOT assets, the machine-wide GPU lock taken with the GPU at 0 %. Two passes per
bundle, each in its own process, one warm-up decision, then 11 decisions × 3; wall time from the file
read to the response, ms:

| record | T | decoder calls | head | fp16 median (min–max) | int8mix median | tokenize | image decode + resize + patches | tower | decoder | head |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| SemIf `a3f18f3a` | 185 | 3 | t512 | 411 (411–416) | 412 | 0.8 | – | – | 401 | 7.4 |
| SemIf `6149a17b` | 200 | 4 | t512 | 545 (544–547) | 547 | 0.9 | – | – | 535 | 7.4 |
| `own_t01` | 436 | 7 | t512 | 949 (947–949) | 951 | 1.9 | – | – | 938 | 7.4 |
| `own_t11` | 497 | 8 | t512 | 1,084 (1,082–1,085) | 1,085 | 2.2 | – | – | 1,073 | 7.4 |
| `own_t12` | 967 | 16 | t1024 | 2,165 (2,157–2,167) | 2,168 | 3.9 | – | – | 2,147 | 12.0 |
| `own_t16` | 2,494 | 39 | t4096 | 5,289 (5,280–5,293) | 5,291 | 10.8 | – | – | 5,249 | 27.7 |
| `own_t15` | 2,603 | 41 | t4096 | 5,563 (5,551–5,576) | 5,572 | 11.7 | – | – | 5,522 | 27.5 |
| `photo_02` `g256` | 499 | 8 | t512 | 1,138 (1,136–1,142) | 1,139 | 1.7 | 11.6 | 39.6 | 1,073 | 10.0 |
| `img_04` `g256` | 558 | 9 | t1024 | 1,273 (1,271–1,276) | 1,276 | 1.7 | 8.9 | 40.0 | 1,207 | 12.3 |
| `photo_02` `g448` | 631 | 10 | t1024 | 1,457 (1,455–1,459) | 1,459 | 1.8 | 13.4 | 87.1 | 1,342 | 12.1 |
| `img_04` `g448` | 690 | 11 | t1024 | 1,586 (1,584–1,600) | 1,587 | 2.0 | 11.3 | 87.2 | 1,475 | 9.9 |

The stage columns are the fp16 medians. One decoder call (64 tokens) takes 134.2 ms with fp16 (936
calls, 132.6–145.3) and 134.3 ms with int8mix (132.7–149.8). Load from the runtime's cache 3.3 s
(fp16); int8mix 13.3 s in its first pass, right after the fp16 passes, and 2.9 s in the second. The
first decision of a process takes 3.4–3.7 s, its first decoder call 2.6–2.8 s. Zeroing the four
states takes 1.3 ms. Transcript: [`gate-clef-flash-timing.json`](gate-clef-flash-timing.json).

## Precision

**int8.** int8 over every decoder linear misses the bar (int8lin above). An fp32 torch bisect with
the exporter's own int8 weights (`conversion/clef_flash/int8_bisect_torch.py`, the six runs int8lin
misses worst) puts the error in layers 0–15: those 16 layers in fp16 leave the worst run at 0.0023,
layers 16–31 in fp16 change nothing (0.0845, all int8: 0.0842). The error is spread out and partly
cancels: one kind of linear alone in int8 is already large (Gated DeltaNet projections 0.0752, MLP
0.0342, attention 0.0230), no single fp16 layer brings the worst run below 0.0492, and some single
layers make it worse (layer 3: 0.1164).

The rule was written before any bisect result: worst run ≤ 0.010, no run worse than int8lin, at most
six fp16 layers, the smallest set. No set met it; the ranked sets of two to six layers stayed at
0.0260–0.0350. Contiguous fp16 blocks, run after the rule's search and outside it:

| fp16 layers | 0–7 | 0–8 | 0–9 | 0–10 | 0–11 | 4–15 | 8–15 | 12–15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| worst run max \|Δp\| | 0.0568 | 0.0509 | 0.0232 | 0.0116 | 0.0069 | 0.0207 | 0.0296 | 0.0939 |

int8mix keeps layers 0–11 fp16, twelve layers, outside the rule's cap. It was chosen on the
fixture; the held-out records above are its test (186/186, max 0.0171). It saves 4.05 GB against
fp16 (`main.mlirb` 11,826,520,241 vs 15,880,253,283 bytes) and no time: on the same 202 runs a call
takes 134.4 ms (int8mix), 134.3 (int8lin) and 134.2 (fp16); why int8 buys no time is not isolated
([`gate-clef-flash-int8-bisect.json`](gate-clef-flash-int8-bisect.json)).

**fp16.** The fp16 decoder passes with room, but a few hidden rows drift: 50 of 73,343 positions
have a cosine below 0.99 against the oracle (34 of them image rows), 6 below 0.9, one below 0.5,
0.41 at an image row of `photo_04` `g448`, a run whose max |Δp| is 1.1e-4. The fp16 module in torch
on MPS finds the same row hardest but keeps it at 0.877. Computing the full-attention linears and SDPA
in fp32 does not lift it (38 runs: lowest 0.378 vs 0.407, 53 vs 48 positions below 0.99, max |Δp|
0.0075 in both, 7% slower per call). Which op opens the gap is not isolated
([`gate-clef-flash-variants.json`](gate-clef-flash-variants.json)).

**Chunk width.** 64 tokens per call: 134.2 ms per call, 2.26 ms per token, against 58.8 ms and 3.77
ms per token at 16 (202 runs, shared GPU), the argmax equal on all 213 runs and p within 0.0043. 128
tokens per call bought nothing on 19 runs: 2.14 vs 2.19 ms per token, 1.227 vs 1.210 s per run with
the longer padding.

**Head attention over long inputs.** On this GPU stack the plain `softmax(q kᵀ · s + b) v` chain
compiled for the head gave wrong values, different on every call, from 4,032 keys on (exact at
3,584), while each op alone was exact at 4,096. Keys in blocks of at most 2,048 with one shared max
are exact at 4,032 and 4,096 (max |d| 5.7e-7), so the `t4096` function uses that form. The decoder's
own attention showed nothing of the kind on a 4,050-token row whose last calls attend 4,032 and 4,096
keys: one position of 4,050 is below a cosine of 0.99 (index 1,300, 0.9585), max |Δp| 6.1e-4
([`gate-clef-flash-head.json`](gate-clef-flash-head.json)).

**The lm_head table.** fp16 rows move p by at most 1.8e-7 (the fp32 table's own re-run floor); int8 per
row would move it by 4.4e-4 and save 1.0 GB. The table ships in fp16.

## ⬇️ Bundle

[mlboydaisuke/clef-flash-CoreAI](https://huggingface.co/mlboydaisuke/clef-flash-CoreAI), five folders
under `gpu-pipelined/` and the table under `host/`. A decision needs one decoder, the head and the
table; an image adds one tower.

| folder | what | main.mlirb bytes | main.mlirb sha256 |
|---|---|---:|---|
| `gpu-pipelined/clef_flash_decode_fp16_pf64/` | decoder, fp16 (default): `.aimodel` + `metadata.json` + `tokenizer/` | 15,880,253,283 | `0b7dfa30…b9bea8` |
| `gpu-pipelined/clef_flash_decode_int8mix_pf64/` | decoder, int8 except layers 0–11, same layout | 11,826,520,241 | `2c19a0e7…6a67b9` |
| `gpu-pipelined/clef_flash_g256_vision_fp16w32/` | tower, 256×256 → 64 rows (`.aimodel`) | 909,207,876 | `b2adb039…64f31d` |
| `gpu-pipelined/clef_flash_g448_vision_fp16w32/` | tower, 448×448 → 196 rows (`.aimodel`) | 911,945,013 | `04d8629b…3a1eaa` |
| `gpu-pipelined/clef_flash_head_bucket_fp16w32/` | joint schema head, `t512`–`t4096` (`.aimodel` + `metadata.json`) | 243,978,965 | `51e512ad…6856fe` |

| file | what | bytes | sha256 |
|---|---|---:|---|
| `host/lm_head_fp16.bin` | the untied `lm_head.weight`, `[248320, 4096]` fp16, row-major, no header | 2,034,237,440 | `bfccf00b…1ba36d` |
| `host/lm_head_fp16.json` | its shape, dtype, source tensor and read-back check | 2,939 | `ef89e2fb…b75e86a` |

`SHA256SUMS` in the repository lists every file. Uploaded 2026-10-03 as Hub revision `3d9e0ae355a52fc98b9e819b80c9c996734dc840` (the bundle upload); after the upload every file's size and sha256 matched the staged copy (the 8 LFS files by the Hub's own hash, the 23 small files by download). A later commit replaced only `README.md` and `SHA256SUMS`.

No AOT asset ships: the Swift runtime specializes the `.aimodel` files correctly on the Mac (the JIT
rows above); the decoder's AOT compile is 29.7 GB (fp16) / 25.7 GB (int8mix). The author's
`joint_head.safetensors` is not included, because the head graph carries its weights. The source's
`config.json` and `joint_head_config.json` are in the repository verbatim, with its Apache-2.0
`LICENSE`.

## Use it

Swift, with the [`ClefFlash`](../../apps/ClefFlash/) package (macOS 27; the system CoreAI framework
and swift-transformers' tokenizer), on a download of the repository:

```swift
import CoreAI
import ClefFlash

let root = URL(filePath: "clef-flash-CoreAI")                        // a download of the HF repo
let gp = root.appending(path: "gpu-pipelined")
var decoderOptions = SpecializationOptions(preferredComputeUnitKind: .gpu)
decoderOptions.expectFrequentReshapes = true
let gpu = SpecializationOptions(preferredComputeUnitKind: .gpu)
let decider = try await ClefDecider(
    assets: .init(decoderBundle: gp.appending(path: "clef_flash_decode_fp16_pf64"),
                  head: gp.appending(path: "clef_flash_head_bucket_fp16w32"),
                  table: root.appending(path: "host/lm_head_fp16.bin"),
                  towers: [.g448: gp.appending(path: "clef_flash_g448_vision_fp16w32/clef_flash_g448_vision_fp16w32.aimodel")]),
    decoderOptions: decoderOptions, headOptions: gpu, towerOptions: gpu)   // nil decoder / headAsset = the .aimodel (JIT)
let request = try SystemOneRequest(data: try Data(contentsOf: requestURL))
let response = try await decider.decide(request: request, image: cgImage, grid: .g448)   // image: nil = text only
print(PythonJSON.dumps(response, sortKeys: false))
// {"model":"clef-flash","answers":{"department":{"type":"choice","choice":…,"confidence":…,"probabilities":{…}},…},
//  "usage":{"input_tokens":…,"output_tokens":0}}
```

The same from the CLI (`swift build -c release --package-path apps/ClefFlash` builds `clef-flash`):

```bash
R=clef-flash-CoreAI; G=$R/gpu-pipelined
clef-flash ask --decoder-asset jit --decoder $G/clef_flash_decode_fp16_pf64 --head $G/clef_flash_head_bucket_fp16w32 \
    --table $R/host/lm_head_fp16.bin \
    --tower-g448 $G/clef_flash_g448_vision_fp16w32/clef_flash_g448_vision_fp16w32.aimodel \
    --request req.json --image x.png --grid 448 --out resp.json
```

A request, in the SystemOne-compatible request shape:

```json
{"model": "clef-flash",
 "state": "Our checkout started returning errors and orders are blocked.",
 "questions": {
   "department": {"type": "choice", "instructions": "Which team should handle the message?",
                  "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
   "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
   "outage": {"type": "noul", "instructions": "Is a service down?"}}}
```

Python: `conversion/clef_flash/decide.py run --request req.json [--image x.png --grid 448] --out
resp.json` is the gates' own read-out with `coreai.runtime`. It loads AOT `.aimodelc` assets only
(compile each `.aimodel` with the `aot` line of [`recipe.toml`](recipe.toml)); the Python gates never
used the Python runtime's JIT.

## Reproduce

Environment: the zoo overlay venv (coreai-core 1.0.0b2, coreai-torch 0.4.1, torch 2.9.0,
transformers 4.57.6), the source snapshot pinned at `17f0b0a` in `HF_HOME`, `HF_HUB_OFFLINE=1`,
Xcode 27.0 RC. The author's code runs in its own venv (transformers 5.17.0, torch 2.9.0). The steps,
in order, with every flag, are in [`conversion/clef_flash/README.md`](../../conversion/clef_flash/README.md).

```bash
# towers: the Qwen3.8-27B vision exporter with the HF id and the grid set
python conversion/export_qwen38vl_pipelined.py --hf-id Cloudflare/clef-flash --name clef_flash_g256 \
    --grid-h 8 --grid-w 8 --skip-decoder --vision-dtype fp16w32 --out-dir <work>/_clefflash/exports
python conversion/export_qwen38vl_pipelined.py --hf-id Cloudflare/clef-flash --name clef_flash_g448 \
    --grid-h 14 --grid-w 14 --skip-decoder --vision-dtype fp16w32 --out-dir <work>/_clefflash/exports
# decoders (--aot adds the h16c .aimodelc the Python gates load)
python conversion/clef_flash/export_decoder.py fp16 --prefill-chunk 64 --aot
python conversion/clef_flash/export_decoder.py int8mix --fp16-layers 0,1,2,3,4,5,6,7,8,9,10,11 --prefill-chunk 64 --aot
# head and host table
python conversion/clef_flash/export_head.py --shape bucket --weights fp16w32 --aot
python conversion/clef_flash/export_lm_head_table.py
```

Port notes: [`knowledge/clef-flash-port.md`](../../knowledge/clef-flash-port.md).

## Other formats

On the Hub (2026-10-03), not run here: MLX conversions with the joint head,
`mlx-community/clef-flash-4bit` and `-8bit` and `TrevorJS/clef-flash-mlx-4bit` and `-8bit`; an
OpenVINO conversion with the head, `meossistant/clef-flash-openvino`; GGUF conversions such as
`bartowski/Cloudflare_clef-flash-GGUF`, which carry no joint head file. mlx-community's card on the
backbone alone: "It is not a chat model — `mlx_vlm.generate`, `mlx_lm.generate`, and LM Studio will
load the backbone but produce meaningless text." No other Core AI conversion of clef-flash was
listed on the Hub on 2026-10-03.

## License

Source Apache-2.0 (Cloudflare/clef-flash, following its base Qwen/Qwen3.5-9B, also Apache-2.0); the
bundles inherit it, and the Hugging Face repository carries the source's `LICENSE`. The author's
`joint_schema_model.py` runs only in the oracle and the reference scripts at gate time; it is not
part of the bundles. The fixture's drawn images are generated by `make_fixtures.py` (CC0-1.0); its
photographs are Wikimedia Commons files marked CC0 (each file page, thumbnail URL and sha256 is in
the fixture); its 144 SemIf authored144 items are MIT (github.com/TheoLeeCJ/SemIf at `ca3ba65f`), and
the fixture carries their copyright and permission notice. The records written for the port use
invented people, organisations and places.
