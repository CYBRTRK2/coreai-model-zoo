# ClefFlash: the Swift host of clef-flash

A Swift package that answers a SystemOne-shaped request — a state, typed questions (`noul`, `choice`, `score`) and
optionally one image — with the typed decisions of the clef-flash Core AI port, on the system CoreAI framework and
swift-transformers' tokenizer only. The library `ClefFlash` and the Mac CLI `clef-flash` build for macOS 27; the iPhone
is out of scope (the decoder alone is ≥ 10 GB).

```swift
import ClefFlash

let decider = try await ClefDecider(
    assets: .init(decoderBundle: exports.appendingPathComponent("bundles/clef_flash_decode_fp16_pf64"),
                  decoder: exports.appendingPathComponent("bundles_aotc/clef_flash_decode_fp16_pf64.h16c.aimodelc"),
                  head: exports.appendingPathComponent("head/clef_flash_head_bucket_fp16w32"),
                  headAsset: exports.appendingPathComponent(
                      "head/clef_flash_head_bucket_fp16w32/clef_flash_head_bucket_fp16w32.h16c.aimodelc"),
                  table: exports.appendingPathComponent("host/lm_head_fp16.bin"),
                  towers: [.g448: towerURL]))                       // decoder / headAsset nil = the .aimodel (JIT)
let request = try SystemOneRequest(data: try Data(contentsOf: requestURL))
let response = try await decider.decide(request: request, image: cgImage, grid: .g448)   // image: nil = text only
print(PythonJSON.dumps(response, sortKeys: false))
// {"model":"clef-flash","answers":{"department":{"type":"choice","choice":"technical","confidence":0.9783,...}},
//  "usage":{"input_tokens":436,"output_tokens":0}}
```

`trace(request:image:grid:)` returns the ids, the question and option spans, the hidden rows, the head's bucket, every
option's logit and float32 probability, the response and every step's time; `trace(…, embeds:, embedsGrid:)` replaces
the image and the tower with given image rows (the gate feeds the oracle's rows at a native grid this way).

## What each part copies

| file | the contract it reproduces |
|---|---|
| `JSONValue.swift` | the request as written: members in order, number literals kept (`1250` ≠ `1250.0` in the prompt); `json.loads`' duplicate-key and NaN/Infinity rules |
| `Renderer.swift` | the author's `render`: a string as is, anything else `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)` — keys by code point, Python's float `repr`, Python's string escapes; `round(x, 4)` |
| `SystemOne.swift` | `host.validate_request`, the option order (`noul` true/false with the default criteria, `choice` ids by code point, `score` levels), `decide.systemone_answer` / `response` |
| `PromptBuilder.swift` | `host.build_ids`: prefix (36) + image block + state + schema (FIELD blocks, spans) + suffix (18), each piece tokenized alone; `static_inputs` (`V + k`, `image_rc`, rope shift) |
| `ImagePreprocess.swift` | `host.preprocess`: Pillow's own 8-bit BICUBIC (`Resample.c`: 22-bit fixed-point weights, horizontal pass first) and the merge-block-major patches |
| `VisionTower.swift` | patches f32 `[4 G², 1536]` → image_embeds f32 `[G², 4096]` (G = 8 / 14) |
| `Decoder.swift` | `main` (S = 64): four zeroed states per row, the static inputs, ⌈T / 64⌉ calls, the last padded with `<\|endoftext\|>`, hidden `[T, 4096]` fp16 |
| `Head.swift` | `clef_head.head_inputs` (span means, last token, lexical rows of the mmapped fp16 table in float64, membership, types, masks, bucket padding), the bucket function, `question_probs` (NumPy's float32 softmax order) |
| `ClefDecider.swift` | the glue, in decide.py's order |

## Contract checks at load

An asset that differs fails at load, not in a probability:

- the decoder bundle's `metadata.json`: vocabulary 248,320, `max_context_length`, `prefill_chunk`, `n_image_max` 1,024, prompt lengths 36 / 18;
- the tokenizer: `<|endoftext|>` 248044, `<|vision_start|>` 248053, `<|vision_end|>` 248054, `<|image_pad|>` 248056, `<|im_end|>` 248046, `"\n"` → [198], the prefix encodes to 36 ids ending in 198 and the suffix to 18 ids ending in `:` (25);
- the decoder's `main`: input / output / state names, shapes and types (`input_ids [1, 64]`, `position_ids [1, -1]`, `image_embeds [1024, 4096]` f16, `image_rc [1024, 2]`, the rope shift, `hidden [1, 64, 4096]` f16, the four states);
- the head: `head.shape == "bucket"`, every bucket function present with its ten inputs (`[T, 4096]`, `[Q 16, T]`, `[O 128, T]`, …) and `logits [128]` f32;
- each tower: `patches [4 G², 1536]` f32 → `image_embeds [G², 4096]` f32, no states;
- the lm_head table: 2,034,237,440 bytes (`[248320, 4096]` fp16).

## Build and run (Mac)

```sh
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer   # CoreAI.framework is in the Xcode 27 SDK
# $ZOO_WORK_ROOT: the lane work root, by default the parent directory of this repository (conversion/_paths.py)
swift build -c release --package-path apps/ClefFlash --scratch-path $ZOO_WORK_ROOT/_clefflash/swift/.build
BIN=$ZOO_WORK_ROOT/_clefflash/swift/.build/release/clef-flash
EX=$ZOO_WORK_ROOT/_clefflash/exports
$BIN ask --assets $EX --request req.json [--image x.png --grid 256|448] --out resp.json [--trace trace.json] \
    [--bundle-name clef_flash_decode_int8mix_pf64] [--decoder-asset aot|jit]
$BIN fixture --assets $EX --records records.json --images <dir> --arms text,g256,g448,native \
    --embeds manifest.json --out out.json [--reload] [--warmup 1 --repeat 3] [--dump-dir d --dump tiles,embeds,hidden]
$BIN check-ids --records records.json --expected records_oracle.json --tokenizer <bundle>/tokenizer --out ids.json
$BIN preprocess --images <dir> --out-dir <dir>      # the host half alone: tiles + patches' sha256
$BIN render-test --in values.json --out out.json    # the renderer alone
```

`--decoder-asset aot` (default) loads the AOT `.aimodelc` assets with `SpecializationOptions.default`, as the Python
gates do. `--decoder-asset jit` specializes the `.aimodel` files here: the decoder GPU-preferred with
`expectFrequentReshapes` (its AOT flags), the head and the towers GPU-preferred. Each asset can be named on its own
(`--decoder`, `--decoder-path`, `--head`, `--head-path`, `--table`, `--tower-g256`, `--tower-g448`).

`conversion/clef_flash/gate_swift.py` scores the CLI's JSON against the author's fp32 oracle and the Python reference of
the same assets; `_time_mac.sh` times a fixed set of decisions under the machine-wide GPU lock.

## Measured (M4 Max, macOS 27.0 26A428, Xcode 27.0 RC, Release)

- Against the author's fp32 oracle (fixture 213 runs + held-out 40, every option of every question): fp16 S = 64
  argmax 397/397 + 186/186 (near-ties 4/4, 2/2), max |dp| 0.0122 / 0.0082; int8mix 397/397 + 186/186 (near-ties 4/4,
  1/2), 0.0153 / 0.0171. AOT and JIT both pass; the fp16 JIT result is bit-equal to the AOT one.
- Against the Python reference of the same assets: on identical inputs the hidden rows and the logits are bit-equal
  (every text run, every run fed the oracle's image rows), the tower rows are bit-equal (48/48), and the responses
  equal `decide.py`'s (24/24). ids and spans equal the oracle's on all 254 runs, from the raw request.
- A decision (AOT, the GPU lock held): 0.41 s at 185 tokens, 0.95 s at 436, 2.17 s at 967, 5.56 s at 2,603; an image at
  g256 1.14–1.27 s, at g448 1.46–1.59 s. The decoder takes 134 ms per 64 tokens; load 2.9–3.3 s from the runtime's
  cache; the first decision of a process 3.4–3.6 s.

Transcripts: `$ZOO_WORK_ROOT/_clefflash/results/swift_gate_{fp16,int8mix}_{aot,jit}.json`, `swift_timing.json`.

## Notes

- `InferenceFunction.MutableViews` borrows what it holds until `run` returns: `ClefDecoder` keeps the four states and
  the hidden buffer in a class property and moves them into locals for a row's whole pass (the DeciderVision form).
- The states are allocated once at `max_context_length` (4,096) and zeroed per row (≈ 160 MB of fp16).
- Images: one image per request at a fixed grid (`g256` = 64 image tokens, `g448` = 196). The request's own `images`
  field is not read. PNG decodes to Pillow's bytes; a JPEG decoded by ImageIO is not guaranteed to equal libjpeg's.
- A row longer than the decoder's context (4,096 tokens with the image block) is refused; the author's own cut at
  16,384 tokens is copied but never reached.
