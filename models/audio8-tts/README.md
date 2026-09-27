# Audio8-TTS-Preview-0.6b — Core AI

[🤗 mlboydaisuke/Audio8-TTS-Preview-0.6b-CoreAI](https://huggingface.co/mlboydaisuke/Audio8-TTS-Preview-0.6b-CoreAI) · Apache-2.0 · base [Edge0/Audio8-TTS-Preview-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.6b)

[`Edge0/Audio8-TTS-Preview-0.6b`](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.6b) (Edge0, Apache-2.0, 601M + a
337M codec) is a **DualAR text-to-speech** model of the Fish Audio S2 Pro design: a Qwen2.5-shaped **slow AR** (24
layers, 896 wide) predicts one *semantic* token per 46 ms frame, a 4-layer **fast AR** predicts the frame's nine other
codec codebooks one after another, and a 44.1 kHz **DAC-style codec** turns the ten codebooks into audio. **Eleven
languages** (Cantonese, Chinese, Dutch, English, French, German, Italian, Japanese, Korean, Polish, Spanish) and
**zero-shot voice cloning** from a 0.5–30 s reference recording and its transcript. This is the zoo's first DualAR /
semantic-token TTS and the first port with the sampler inside the graph. On 2026-09-28 the Hub had MLX builds
(`mlx-community/Audio8-TTS-Preview-0.6b-bf16`, 8-bit and 4-bit conversions), the publisher's ONNX INT4 build for CPUs
and a GGUF, and no Core AI or Core ML port of the TTS (a Core ML port exists of the publisher's ASR sibling); this port
runs the model on iPhone and Mac through Apple's runtime.

## Pipeline

```
text (+ reference transcript + reference codes) ──(host: the publisher's prompt segments, encoded one at a time)──▶ prompt [11, P]
  1. dualar.aimodel  prefill(codes [1,11,32], pos) in windows                            ─▶ logits [32,4097] · hidden [32,896]
                     frame(codes [1,11,1], pos, noise_slow [2,4097], window [10], noise_fast [9,4096], forced, use_forced)
                       = the slow step + the semantic draw (top-k 50 / top-p 0.9 / T 0.7, Gumbel-max, RAS) + the fast AR's
                         ten rows with nine codebook draws — ONE call per frame                 ─▶ semantic · codes [10]
                     first_frame(logits, hidden, …) for the frame right after the prefill
  host: the 10-token RAS window, the uniform draws (a seeded stream), stop at eos 151645 or 512 frames
  2. codec.aimodel   codes [1,10,160] ─▶ wav [1, 327680] at 44.1 kHz                     (every op causal: 160-frame windows, keep the last 32)
voice registration:  wav 44.1 kHz [1,1,442368] ─▶ encoder.aimodel ─▶ codes [1,10,216]    (0.5–10 s reference; the codes + transcript are the voice)
```

A frame is 2,048 samples (21.5 frames/s). The prompt is the publisher's: `<|im_start|>system\n` · "convert the provided
text to speech" (or, with a voice, "… reference to the following:\n\nText:\n" · `<|speaker:0|>` + transcript ·
"\n\nSpeech:\n" · the reference's codebook-0 codes as semantic ids) · `<|im_end|>\n<|im_start|>user\n` · text ·
`<|im_end|>\n<|im_start|>assistant\n<|voice|>` — segments encoded separately, asserted id for id against the publisher's
processor on all 18 fixtures, in Python and in the Swift host.

### Graph contracts

```
dualar   prefill     in  codes[1,11,32] i32 (row 0 ids, rows 1-10 codebooks; pad 151643 / 0) · pos[1] i32   out logits[32,4097] f16 · hidden[32,896] f16
         frame       in  codes[1,11,1] i32 (the previous frame) · pos[1] · noise_slow[2,4097] f32 · window[10] i32 (-1 = none)
                         noise_fast[9,4096] f32 · forced[11] i32 · use_forced[1] f32           out semantic[1] i32 · codes[10] i32 · logits · hidden · fast_logits[9,4096] · sampled_*
         first_frame in  logits[4097] f16 · hidden[896] f16 · the same noise / window / forced   out as frame
         state       k_cache / v_cache [24,1,2,2048,64] f16 (prefill and frame)
         precision   slow AR linears int8 (weight-only, per-block-32, symmetric with clipping); embeddings, the 4,097-row head, norms, the fast AR fp16
codec    main        in  codes[1,10,160] i32 (right-pad 0; codebook 0 < 4096, codebooks 1-9 < 1024, clamped in-graph)   out wav[1,327680] f16
encoder  main        in  audio[1,1,442368] f32 mono 44.1 kHz (right-pad 0)                                          out codes[1,10,216] i32 (keep ceil(samples/2048) frames)
```

The head is the embedding table's 4,097 rows (the semantic ids and eos) instead of 155,776: the publisher's sampler sets
every other logit to -inf before drawing, so nothing is lost (7 MB instead of 279 MB).

## What made it work

1. **One graph call per frame, the sampler inside.** The publisher's ONNX runtime — and this port's first export —
   runs a frame as a slow-AR call plus nine fast-AR calls with sampling on the host; on the raw Core AI runtime path
   that is ~48 ms of engine time per 46 ms frame on an M4 Max, a call costing milliseconds before any arithmetic. The
   shipped `frame` function does the slow step, the semantic draw and the fast AR's ten rows in one call, with the
   publisher's top-k / top-p / temperature / Gumbel-max / RAS sampler written without a sort (`topk(50)` +
   `logsumexp` + a cumulative sum); the uniform draws are inputs, so a recorded draw replays the oracle's choice exactly.
   Details: [`knowledge/audio8-tts-port.md`](../../knowledge/audio8-tts-port.md).
2. **int8 on the slow AR, fp16 on the fast AR.** On 15,093 teacher-forced codebook draws the fp16 fast AR reproduces the
   oracle's draw 99.5 % of the time; int8 drops to 95.8 % for 53 MB. The slow AR takes int8 well (1,671 / 1,695 semantic
   draws, 21 of the 24 misses at a Gumbel margin under 0.13; fp16: 1,688).
3. **The codec encoder needs fp32 arithmetic.** Its vector quantizers pick nearest codebook entries; the whole-fp16
   export flips those choices and the nine residual books cascade (44–121 of ~200 frames exact). fp16 weights with fp32
   arithmetic (the Fun-ASR encoder recipe) gives 691 / 712 frames exact and codebook 0 exact on every frame.
4. **Kept from the publisher's code:** bfloat16-rounded RoPE tables (interleaved pairs), RMSNorm eps 1e-6 in fp32, the
   RAS window that skips the first frame's token, the clamp of codebooks 1–9 to 1,023. The slow AR's residual peaks at
   12,827 — a fifth of float16's range — so no rescale was needed.

## Numerics gate (18 fixtures: 6 ja + 6 en + 2 zh sentences of our own, 4 voice-clone fixtures on FLEURS test clips)

Oracle = the publisher's `modeling_arktts.py` in fp32 on the CPU, seeded (`generate` reproduced call for call, every
uniform draw recorded). Teacher-forced = the oracle's tokens fed to the graphs; a *draw* comparison asks whether the
in-graph sampler, given the port's logits and the oracle's noise, makes the oracle's choice.

| stage (Mac GPU) | result |
| --- | --- |
| tokenizer-only prompt vs the publisher's processor | 18 / 18 prompts identical (Python and Swift) |
| plain-torch re-author, eager fp32, teacher-forced | every argmax and every replayed draw equal to the oracle; codec wav cos 1.000000 |
| **dualar int8 (ship)**, teacher-forced, 1,695 slow steps | logits cos min 0.99985, argmax 1,668, **draw 1,671 / 1,695** |
| same, 15,093 fast-AR codebook draws | logits cos min 0.99805, argmax 14,705, **draw 14,711 / 15,093** |
| codec decoder on the oracle's codes, 18 utterances | wav cos min 0.99978, log-mel cos min 0.99876 |
| codec encoder (fp16w32) on the 4 reference clips, 712 frames | 691 frames exact; codebook 0 exact on every frame, every codebook ≥ 97.7 % |
| free run (the port's own loop, the oracle's draws) | reached eos 18 / 18; Swift host == Python engine on 1,637 / 1,688 frames (10 / 18 utterances identical end to end) |
| ASR round trip vs the fixture text (Fun-ASR fp32): ja CER / en WER / zh CER | port **2.3 % / 0.9 % / 0.0 %** · the oracle's own audio 6.5 / 0.0 / 0.0 |
| speaker cosine to the reference clip (WavLM-Base-Plus-SV), 4 clone fixtures | port mean 0.855 / min 0.561 · oracle 0.854 / 0.582 |

A miss is fp16 GPU or int8 arithmetic moving a draw that sat within a hair of the runner-up; under sampling, one flip
changes every later frame, so end-to-end identity is not the bar — the ASR and speaker rows are. They are eight to
fourteen sentences per language, our own measurement with one normalizer: the port's speech is as intelligible and as
speaker-faithful as the publisher's fp32 output on these fixtures, not more.

## Speed

Measured with [`apps/Audio8Gate`](../../apps/Audio8Gate) (the kit's `Audio8TTS` in a Release build, the 18 fixtures,
78 s of audio; medians; 2026-09-28) **while another model conversion ran on the same Mac (load average 5–7)**. Both
assets are JIT: the first load specializes them, later loads read the cache.

| | frame (slow step + sampling + fast AR) | codec (160-frame window) | time to first audio (32-frame chunk) | RTF median / p90 | first load (cold) → later loads | footprint |
|---|---|---|---|---|---|---|
| M4 Max (GPU, macOS 27 26A428, GPU lock held) | 36–40 ms | ~0.7 s per utterance | 1.5 s | **1.05** / 1.09 (bench on one 5 s sentence: 0.99) | 5.4 s (cache 0 → 1.27 GB) → 0.8 s | 0.85–0.95 GB |
| iPhone 18 Pro | not yet measured (the device was held by another lane on 2026-09-28) | | | | | |

A 46 ms frame costs ~36 ms of engine time, so the Mac synthesizes at about real time and streams the first 1.5 s of
audio after 1.5 s. The per-frame cost is the raw runtime path's per-layer dispatch (a 512-slot cache, AOT compilation
and Apple's composite RMSNorm/RoPE change it by 0–10 %); Apple's pipelined engine drives a same-sized Qwen3-0.6B at
2.8 ms per token, and moving the slow step onto it is the lever for a several-times faster frame — a separate round.

## Use it

CoreAIKit `Audio8TTS` (catalog enrollment pending):

```swift
import CoreAIKit

let tts = try await Audio8TTS(paths: .standard(root: modelDir))      // dualar + codec .aimodel, tokenizer/
let audio = try await tts.synthesize("明日の午後、駅前の喫茶店で待ち合わせましょう。")   // 44.1 kHz mono [Float]
// streaming: tts.synthesizeStreaming(text) { chunk in play(chunk) }   ~1.5 s chunks
// voice cloning: Audio8Voice(referenceText:codes:) from register_voice.py / the encoder graph
let cloned = try await tts.synthesize("Turn left at the second traffic light.", voice: voice)
```

## ⬇️ Bundle

**[mlboydaisuke/Audio8-TTS-Preview-0.6b-CoreAI](https://huggingface.co/mlboydaisuke/Audio8-TTS-Preview-0.6b-CoreAI)** —
`audio8_dualar_int8_cl2048_w32.aimodel/` (876 MB) + `audio8_codec_decoder_fp16_t160.aimodel/` (261 MB) +
`audio8_codec_encoder_fp16w32_t216.aimodel/` (416 MB, voice registration only) + `tokenizer/`; one subtree for macOS and
iOS (JIT `.aimodel`s). Apache-2.0: LICENSE + NOTICE from the publisher's repository. Mirror:
`coreai-community/Audio8-TTS-Preview-0.6b-CoreAI`.

Convert yourself — [`conversion/audio8_tts/`](../../conversion/audio8_tts/): `oracle_audio8.py` → `parity_audio8.py` →
`export_audio8_frame.py --mode int8`, `export_audio8.py --part codec`, `audio8_encoder.py --frames 216 --dtype fp16w32`,
gated by `gate_audio8_frame.py` (recipe: [`recipe.toml`](recipe.toml)). Generated speech can be misused for
impersonation; the publisher asks for consent before cloning a voice and disclosure of synthetic audio, and so does
this port.
