# Fun-ASR-Nano-2512 — Core AI

[🤗 mlboydaisuke/Fun-ASR-Nano-2512-CoreAI](https://huggingface.co/mlboydaisuke/Fun-ASR-Nano-2512-CoreAI) · Apache-2.0 · base [FunAudioLLM/Fun-ASR-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)

[`FunAudioLLM/Fun-ASR-Nano-2512`](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) (Tongyi Lab,
Apache-2.0, 985M) transcribes **Chinese, English and Japanese**, and Chinese **dialects and accents**
(Cantonese included), with punctuation and inverse text normalization built in and an optional
**hotword list** in the prompt. It is a SenseVoice **SAN-M encoder** (50 + 20 layers with FSMN memory
blocks) and a 2-block adaptor feeding a **fine-tuned Qwen3-0.6B** decoder, so it is the same phase as
the zoo's [Qwen3-ASR](../qwen3-asr/README.md) at a third of the size: encoder + decoder **1.2 GB** on
disk. This is the zoo's fifth speech-to-text family and the first SAN-M / FSMN encoder and the first
hotword-prompted ASR on Core AI; no Core AI or Core ML port of Fun-ASR-Nano was on the Hub on
2026-09-26 (MLX: `mlx-community/Fun-ASR-Nano-2512-*`; ONNX / sherpa-onnx: `csukuangfj/*funasr-nano*`;
GGUF: `FunAudioLLM/Fun-ASR-Nano-GGUF` with the FunASR llama.cpp runtime).

## Pipeline

```
16 kHz mono ──(host: kaldi fbank 80 · hamming 25/10 ms · pre-emphasis 0.97 · ×32768 · LFR 7/6)──▶ feats[L, 560]
  1. encoder.aimodel : feats[1,500,560] f32 (zero-padded to 30 s) · mask[1,500] f32
                       ─▶ audio_embeds[63,1024] f32            (host keeps rows [:N], N = ceil(L/8))
  host: prompt ids = prefix(18) + N ids of `vocab+slot` + suffix(5)   — no audio marker token
  2. decoder bundle  : Qwen3-0.6B _s1 (static [1,1] query, dynamic KV, `audio_embeds` bound as a static
                       input buffer) ─▶ logits ─ greedy ─▶ text        (EOS 151645 / 151643, ≤ 512 tokens)
```

The encoder is one fixed-shape graph for a 30 s window: L = 500 LFR frames (2,998 fbank frames), and the
LLM consumes only the first `ceil(L/8)` adaptor rows (63 at 30 s) — the training-time low-frame-rate
contract. Clips longer than 30 s are transcribed window by window on the host. The prompt is funasr's
ChatML turn (`You are a helpful assistant.` / `语音转写：` + audio); hotwords, a target language and
`itn=false` are text edits to the user turn, verified id-for-id against funasr.

### Graph contracts

```
encoder  in  feats[1,500,560] f32 · mask[1,500] f32 (1 valid / 0 pad)      fp16 weights, fp32 arithmetic
         out audio_embeds[63,1024] f32
decoder  in  input_ids[1,1] i32 (audio = 151936 + slot) · position_ids · audio_embeds[63,1024] (static buffer)
         out logits[1,151936]                            KV state: keyCache / valueCache; int8 linears, tied fp16 head
metadata residual_scale = 0.25, rmsnorm_eps_residual = 6.25e-8   (inside the graph; the host does nothing)
```

## What made it work

1. **The decoder is not stock Qwen3-0.6B, and it overflows float16.** All 311 LLM tensors differ from
   `Qwen/Qwen3-0.6B`. Qwen3's massive activation at position 0 (channel 35) is 8,141 in the stock model
   and **125,067** here — above float16's 65,504 — so the first fp16 and int8 bundles returned NaN.
   Fix: run the residual stream at 1/4 (embedding output ×¼ in-graph, ¼ folded into `o_proj` /
   `down_proj`, residual RMSNorm eps × 1/16). It is exact: 4,890 teacher-forced steps match the
   unscaled fp32 model at every argmax with max |Δlogit| = 0.0. Details:
   [`knowledge/funasr-nano-port.md`](../../knowledge/funasr-nano-port.md).
2. **The encoder ships as float16 weights with float32 arithmetic (450 MB).** The all-fp16 graph has only
   1.39× headroom in the SAN-M residual stream (47,227 on one FLEURS clip) and dips to row cosine 0.9916
   on it; the fp16w32 export matches the 894 MB fp32 bundle (cos min 0.9999982) at 39 ms vs 34 ms.
3. **Keep the tied head tied.** `int8lin` (759 MB) agrees with the fp32 oracle on 150/155 clips;
   untying the head for `int8hu` costs 156 MB and drops to 144/155. Every miss on every arm is a
   knife-edge: the oracle's own top-2 gap is under 0.05 and the port emits its runner-up.
4. **Two epsilons, positions from 1, dither off.** Encoder LayerNorm eps 1e-5, adaptor blocks 1e-12;
   sinusoidal positions start at 1 with depth 560 (bake the table in fp32); funasr's `AutoModel` only
   honours `frontend_conf={"dither": 0.0}` at construction.

## Numerics gate (155 clips: the 5 upstream `example/*.mp3` + FLEURS test en_us / cmn_hans_cn / ja_jp, 50 each, CC BY 4.0)

Oracle = the publisher's `funasr` 1.4.16 in fp32 (dither 0, greedy, itn on). Agreement is counted with
the zoo's margin rule: a divergence at a step where the oracle's top-2 softmax gap is below 0.1 is a
knife-edge, reported but not a failure.

| stage (Mac GPU) | result |
| --- | --- |
| NumPy front end vs oracle LFR features | max abs Δ 1.39e-3, frame counts equal on every clip |
| torch fp32 re-author vs oracle adaptor output | per-row cos min 0.999999999 |
| encoder `.aimodel` (fp16w32) vs oracle | per-row cos mean 0.99999988 · min 0.9999982 |
| HF fp32 decoder with the port's prompt ids | gen_ids == oracle 155/155 |
| **end-to-end, ship bundles** (encoder fp16w32 + decoder int8lin), Python engine | **150/155 token-exact**, 5 knife-edge, **0 above the floor**, 0 clips at the 512 cap |
| end-to-end, decoder fp16 (control, 1.1 GB) | 154/155 (encoder fp16) · 155/155 (encoder fp32) |
| Swift host (CoreAIKit `KitFunASRModel`, Mac) | ids equal to the Python engine on 155/155 |

| WER en / CER zh / CER ja | oracle | port (int8lin) |
| --- | --- | --- |
| vs the oracle | — | 0.34 % / 0.00 % / 0.07 % |
| vs FLEURS `transcription` (50 utterances per language, one normalizer) | 5.08 % / 6.86 % / 6.95 % | 5.34 % / 6.86 % / 6.92 % |

The FLEURS rows are our own measurement of the fp32 model and of the port under the same normalizer
(`conversion/funasr_nano/metrics.py`); they are not the publisher's table and are not comparable to it.

## Speed

Measured with [`apps/FunASRGate`](../../apps/FunASRGate) (the kit's `KitFunASRModel` in a Release build,
the same 155 clips, 1,836 s of audio; medians over the clips; 2026-09-27). Both bundles are JIT: the
first load on a fresh install specializes them once, later loads read the cache.

| | encoder (30 s window) | prefill (prompt, S=1 steps) | decode | RTF median / p90 | first load (cold) → later loads |
|---|---|---|---|---|---|
| M4 Max (GPU, macOS 27 26A428, GPU lock held) | 40 ms | 123 ms | 2.78 ms/token | **0.022** / 0.028 | 2.3 s + 1.3 s → 0.55 s |
| iPhone 18 Pro (GPU, iOS 27 24A437, device JIT, h19p) | 146 ms | 393 ms | 8.2 ms/token | **0.075** / 0.094 | 2.5 s + 2.7 s → 0.9 s |

A 13.6 s Japanese clip transcribes in 0.91 s on the phone at a nominal thermal state (0.29 s on the
Mac). The phone heats under back-to-back transcription: the state reaches *fair* after about two
minutes of continuous clips and the encoder slows from 146 to 225 ms; peak footprint 385–411 MB. The
first transcription after a fresh install is 2.3 s (cache still warming), 0.55 s after a relaunch. No
AOT compile was needed on this device; the container's Core AI cache grows to 1.56 GB. Longest-RTF clip
in both runs: the Korean example (0.137 phone / 0.043 Mac).

## Use it

CoreAIKit `KitFunASRModel` (catalog enrollment as `fun-asr-nano-2512` pending):

```swift
import CoreAIKit

let asr = try await KitFunASRModel(decoderBundleAt: decoderDir, encoderModelAt: encoderModel)
let samples = try AudioFile.pcm16kMono(url)                  // any wav/m4a/mp3 → 16 kHz mono
let result = try await asr.transcribe(samples: samples)      // ≤ 30 s per window, longer clips windowed
// options: transcribe(samples:, hotwords: ["开放时间"], language: "中文", itn: false)
```

## ⬇️ Bundle

**[mlboydaisuke/Fun-ASR-Nano-2512-CoreAI](https://huggingface.co/mlboydaisuke/Fun-ASR-Nano-2512-CoreAI)** —
`gpu-pipelined/funasr_nano_2512_decode_int8lin_n63_s1/` (decoder, 759 MB, tokenizer included) +
`gpu-pipelined/funasr_nano_audio_encoder_fp16w32_l500/…aimodel` (encoder, 450 MB); one subtree for
macOS and iOS (both are JIT `.aimodel`s). Apache-2.0: LICENSE (the text the official vLLM packaging
ships; the official repo has none) + NOTICE. Mirror: `coreai-community/Fun-ASR-Nano-2512-CoreAI`.

Convert yourself — [`conversion/funasr_nano/`](../../conversion/funasr_nano/): `export_encoder.py --dtype fp16w32`,
`export_decoder.py --mode int8lin`, gated by `gate_encoder.py`, `parity_decoder.py` and `gate_e2e.py`
(recipe: [`recipe.toml`](recipe.toml)). The CTC decoder configured upstream has no weights in the
checkpoint, so there are no timestamps in this port; the 31-language sibling `Fun-ASR-MLT-Nano-2512` is
the same architecture and is not covered here.
