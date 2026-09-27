---
license: apache-2.0
library_name: coreai
pipeline_tag: automatic-speech-recognition
base_model: FunAudioLLM/Fun-ASR-Nano-2512
language:
  - zh
  - en
  - ja
  - yue
tags: [core-ai, coreaikit, funasr, fun-asr, sensevoice, san-m, qwen3, asr, speech-recognition, hotwords, on-device, apple]
base_model_relation: quantized
---

Core AI is Apple's on-device ML runtime in iOS 27 / macOS 27 and the successor to Core ML: PyTorch models are exported with Apple's `coreai-torch` (LLMs: `coreai.llm.export`) into `.aimodel` bundles that run on the GPU or the Neural Engine, e.g. Qwen3-8B 4-bit decodes at 94 tok/s on an M4 Max GPU, MLX 90 under the same protocol ([apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench), macOS 27 beta, 2026-06).

# Fun-ASR-Nano-2512 — Core AI

[`FunAudioLLM/Fun-ASR-Nano-2512`](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) (Tongyi Lab,
Apache-2.0, 985M) converted to Apple **Core AI** `.aimodel` bundles: speech-to-text for **Chinese
(with dialects and accents, Cantonese included), English and Japanese**, punctuation and inverse text
normalization built in, an optional **hotword list** in the prompt. Encoder + decoder **1.2 GB**.

- SenseVoice **SAN-M encoder** (50 + 20 layers, FSMN memory) + 2-block adaptor → `audio_embeds` rows
- **fine-tuned Qwen3-0.6B** decoder (int8 linears, tied fp16 head), driven by the Core AI high-level
  engine with the audio rows bound as a static input
- one 30 s window per call (500 LFR frames → 63 audio rows); longer clips are windowed on the host

## Numbers (2026-09, macOS 27, M4 Max GPU; fixtures = the 5 upstream `example/*.mp3` + FLEURS test en_us / cmn_hans_cn / ja_jp, 50 each, CC BY 4.0)

Oracle = the publisher's `funasr` 1.4.16 in fp32, dither 0, greedy.

| | result |
| --- | --- |
| end-to-end token agreement with the fp32 oracle, 155 clips | 150/155 exact; the 5 others diverge only where the oracle's own top-2 gap is 0.007–0.033 (knife-edge, runner-up chosen); 0 divergences above the 0.1 floor |
| WER en / CER zh / CER ja, port vs oracle | 0.34 % / 0.00 % / 0.07 % |
| WER en / CER zh / CER ja vs FLEURS reference, oracle → port | 5.08 → 5.34 % / 6.86 → 6.86 % / 6.95 → 6.92 % |
| encoder graph vs oracle (per-row cosine, 155 clips) | mean 0.99999988, min 0.9999982 |
| Swift host (CoreAIKit) vs the Python engine | identical ids on 155/155 |

| speed (CoreAIKit, Release, medians over the 155 clips) | encoder / prefill / decode | RTF median / p90 | first load → later loads |
| --- | --- | --- | --- |
| M4 Max, GPU, macOS 27 (26A428) | 40 ms / 123 ms / 2.78 ms per token | 0.022 / 0.028 | 3.6 s → 0.55 s |
| iPhone 18 Pro, GPU, iOS 27 (24A437), device JIT | 146 ms / 393 ms / 8.2 ms per token | 0.075 / 0.094 | 5.1 s → 0.9 s |

On the phone a 13.6 s clip takes 0.91 s (nominal thermal state); two minutes of back-to-back clips
raise the state to *fair* and the encoder to 225 ms. Peak footprint 385–411 MB. No AOT compile needed.

The FLEURS rows are our measurement of the fp32 model and of this port on 50 utterances per language
under one normalizer; they are not the publisher's benchmark table.

## Files

```
gpu-pipelined/funasr_nano_2512_decode_int8lin_n63_s1/     decoder: <name>.aimodel + metadata.json + tokenizer/   (759 MB)
gpu-pipelined/funasr_nano_audio_encoder_fp16w32_l500/     encoder: funasr_nano_audio_encoder_fp16w32_l500.aimodel (450 MB)
config.json · config.yaml · preprocessor_config.json      the source model's configuration
LICENSE · NOTICE                                           Apache-2.0 text + attribution
```

Both bundles are JIT `.aimodel`s: macOS and iPhone specialize them on the first load, so one subtree
serves both platforms. The encoder stores float16 weights and computes in float32 (its `feats` /
`mask` inputs and `audio_embeds` output are float32). The decoder graph runs its residual stream at
1/4 with the matching RMSNorm eps (`metadata.json`: `residual_scale`, `rmsnorm_eps_residual`) — an
exact transformation that keeps the fine-tuned Qwen3's 125k activation at position 0 inside float16.

## Host contract

```
wav 16 kHz mono → kaldi fbank 80 (hamming 25/10 ms, pre-emphasis 0.97, DC removal, ×32768, log floor FLT_EPSILON,
                  dither 0) → LFR 7/6 → feats[L,560], zero-padded to [1,500,560] + mask[1,500]
encoder → audio_embeds[63,1024]; keep rows [:N], N = ceil(L/8)
prompt ids = enc("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写：")   (18 ids)
           + N ids of 151936 + slot
           + enc("<|im_end|>\n<|im_start|>assistant\n")                                                        (5 ids)
greedy to EOS (151645 or 151643), ≤ 512 tokens; text = decode(skip_special_tokens), "/sil" → " ", spaces collapsed
hotwords: prepend "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n热词列表：[a, b]\n"
language: "语音转写成{中文|英文|日文}："   itn off: "语音转写，不进行文本规整："
```

## Use it (CoreAIKit)

```swift
import CoreAIKit

let asr = try await KitFunASRModel(catalog: "fun-asr-nano-2512")        // downloads this repo
let samples = try AudioFile.pcm16kMono(url)
let result = try await asr.transcribe(samples: samples)                  // result.text
let biased = try await asr.transcribe(samples: samples, hotwords: ["开放时间"])
```

Conversion code, oracle, fixtures and gates: [coreai-model-zoo `conversion/funasr_nano`](https://github.com/john-rocky/coreai-model-zoo/tree/main/conversion/funasr_nano);
card with the lessons: [`models/funasr-nano`](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/funasr-nano/README.md).
Other runtimes for this model: MLX (`mlx-community/Fun-ASR-Nano-2512-*`), ONNX / sherpa-onnx
(`csukuangfj/*funasr-nano*`), GGUF (`FunAudioLLM/Fun-ASR-Nano-GGUF`, FunASR's llama.cpp runtime).

## License

Apache-2.0 (the source model's card states it; the LICENSE text is the one the official vLLM packaging
`FunAudioLLM/Fun-ASR-Nano-2512-vllm` ships, whose weights are bit-identical to the official `model.pt`).
See NOTICE for attribution. The CTC decoder configured upstream has no weights in the released
checkpoint, so this port has no timestamps.
