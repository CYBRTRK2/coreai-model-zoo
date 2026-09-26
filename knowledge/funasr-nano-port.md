# Fun-ASR-Nano-2512: a SAN-M speech encoder feeding a fine-tuned Qwen3-0.6B, ported from the weights

Lessons from porting [`FunAudioLLM/Fun-ASR-Nano-2512`](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)
(Apache-2.0, 985M parameters: SenseVoice SAN-M encoder 70 layers + adaptor 2 blocks + Qwen3-0.6B) to Core AI.
Same phase as Qwen3-ASR ([`qwen3_asr`](../conversion/qwen3_asr/)): audio encoder → embedding rows → Qwen3
decoder, so the export shapes were copied from that port. The findings below are the ones that port did
not have. Code and gates: [`conversion/funasr_nano`](../conversion/funasr_nano); card:
[`models/funasr-nano`](../models/funasr-nano/README.md).

## 1. The LLM is not Qwen3-0.6B, and that is why float16 overflows

The released `config.yaml` says `llm_conf.freeze: true`, and the decoder's shapes are Qwen3-0.6B's. All 311
`llm.*` tensors nevertheless differ from `Qwen/Qwen3-0.6B`. The difference matters for float16: Qwen3
models carry a massive activation in the residual stream at position 0 (`<|im_start|>`), channel 35, from
layer 2 on. In stock Qwen3-0.6B it is 8,141 and fits float16 (65,504). In Fun-ASR-Nano's fine-tuned decoder it
is **125,067**, and the rest of the residual stream is 7,851 against the stock 399. The fp16 and int8 bundles
therefore returned all-NaN logits; CPU float16 eager overflows at layer 2 as well. S1-mini (another Qwen3-0.6B
finetune) and Apple's own qwen3-0.6b bundle never hit this, which is what made the fine-tune visible.

The fix is an exact algebraic transformation, not a clamp: run the residual stream at `s = 1/4`.

- multiply the embedding output by `s` in the graph (text rows and the injected audio rows),
- fold `s` into every `o_proj.weight` and `down_proj.weight`,
- set the residual-side RMSNorm eps (`input_layernorm`, `post_attention_layernorm`, final `norm`) to `eps · s²`
  (1e-6 → 6.25e-8); `q_norm` / `k_norm` act on projections and stay at 1e-6,
- leave the tied embedding / head weights alone.

RMSNorm(x·s) with eps·s² equals RMSNorm(x) with eps, so the logits are unchanged: teacher-forced on the 155
fixture clips (4,890 steps) the scaled fp32 model matches the unscaled one at every argmax with
max |Δlogit| = 0.0; without the eps correction the argmax still matches but logits move by up to 0.46. Position
0 now peaks at 31,430 (2.08× headroom), every other position at 1,963. Per-block symmetric int8 quantization
commutes with a power-of-two scale, so `int8lin` quantizes to the same integers. The scale and the eps are
recorded in the bundle's `metadata.json` (`residual_scale`, `rmsnorm_eps_residual`); the host does nothing.

## 2. Keep the tied head tied: int8lin, not int8hu

Fun-ASR-Nano ties `lm_head` to `embed_tokens` (the checkpoint stores both, bit-equal). The zoo's `int8hu`
recipe unties the head to quantize it, which for a 0.6B model adds 156 MB rather than saving any (S1-mini
found the same). Here it also cost agreement: int8hu 144/155 token-exact against the fp32 oracle, int8lin
150/155, fp16 154/155 — every miss on every arm at a step where the oracle's own top-2 probability gap is
below 0.05 and the port emits the oracle's runner-up. The FLEURS error rates say the same thing in task
terms: en WER oracle 5.08 %, int8lin 5.34 %, int8hu 5.42 %; zh and ja CER within 0.05 pt. int8lin ships
(759 MB); fp16 (1.1 GB) is the control.

## 3. The encoder's float16 headroom is 1.39×; ship float16 weights with float32 arithmetic

The SAN-M residual stream reaches 47,227 (fp32 reference, `encoders.42`, clip en_us_1688; median clip 38,229)
against the float16 ceiling of 65,504. The all-fp16 bundle passes the row-cosine gate on all 155 clips but
one is at 0.9916, and the margin depends on the audio. Log-mel input makes gain irrelevant (×0.25 / ×1 / ×4
changed the peak by 0.6 %), but 1.39× is not a shipping margin. `--dtype fp16w32` stores every parameter in
float16 and computes in float32: 450 MB (the casts are not folded into fp32 constants), per-row cos min
0.9999982 — the same as the 894 MB fp32 bundle — at 39 ms instead of 34 ms per 30 s window on an M4 Max.
Its inputs and output are float32; a float16 `feats` is refused at load (`invalid scalar type`).

## 4. Two LayerNorm epsilons, one input-scale, positions from 1

`SenseVoiceEncoderSmall` uses funasr's SAN-M `LayerNorm` (eps 1e-5, computed in fp32). The adaptor's two
`EncoderLayer` blocks use `funasr.models.transformer.layer_norm.LayerNorm`, eps **1e-12**. The vLLM port
uses 1e-12 for both; copying it moves the encoder output slightly. The encoder input is multiplied by
√512 before a sinusoidal position encoding of **depth 560 with positions starting at 1**; bake the table in
fp32 (a float16 trace of `sin(500 · ω)` is wrong). FSMN is a depthwise Conv1d (kernel 11, pad 5, no bias) on
the masked value projection plus a residual; Core AI lowers the grouped conv directly (an 11-tap
shift-accumulate is kept as the fallback and matches to 6e-4). Padding rows only reach valid rows through the
masked FSMN input and the masked attention keys, so a zero-padded 500-frame window equals the unpadded run;
the graph re-zeroes pad rows after every layer so nothing non-finite can appear in float16.

## 5. The LLM sees ceil(L/8) rows, not the whole adaptor output

With `use_low_frame_rate`, the model consumes only the first `fake_token_len` adaptor rows,
`1 + (L−1)//2 → 1 + (·−1)//2 → (·−1)//2 + 1` for L LFR frames — algebraically `ceil(L/8)`, 63 for the
30 s window (480,000 samples → 2,998 fbank frames → 500 LFR frames). The adaptor's attention still runs over
all L rows, so the graph computes 500 and emits 63; the host reads N. Feeding all rows loops the LLM
(llama.cpp's port found the same).

## 6. No audio marker token; the prompt is fixed ids around the rows

funasr splits the user turn at `<|startofspeech|>…<|endofspeech|>`, tokenizes the two text halves, and
inserts N placeholder ids (0) whose embeddings it overwrites. Nothing is tokenized for the audio itself, and
`<|AUDIO|>` is not in `tokenizer.json` (the vLLM wrapper adds it at load). The port's contract is the same
thing in the id-space recipe: `enc("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写：")`
= 18 ids, N ids `vocab + slot`, `enc("<|im_end|>\n<|im_start|>assistant\n")` = 5 ids. Hotwords, a
language and `itn=false` are text edits to the user half (`get_prompt`), verified id-for-id against funasr
for five cases. The reference's `generate` runs positions from 2 (its attention mask is two tokens longer
than the embeds); positions from 0 give the same argmax at every step (margin Δ ≤ 3.6e-5) because RoPE is
relative.

## 7. The oracle: dither is a constructor argument, and the first call is bf16

funasr's `WavFrontend` dithers by default (1.0), so the reference is not deterministic. Setting
`frontend.dither = 0.0` on the built model does nothing: `AutoModel` deep-copies its kwargs at construction
and restores them on every `generate`. Only `AutoModel(..., frontend_conf={"dither": 0.0})` reaches
inference (deep-merged with `config.yaml`; the first oracle pass ran dithered and the NumPy front-end gate
caught it at max |Δ| 25). The LLM is built in bf16 and cast to fp32 inside the first `inference_llm` call,
after that call's `inputs_embeds` were built from the bf16 table — the first transcription in a process is a
bf16 one. The oracle is defined as the fp32 steady state (second call on); the audio path is fp32 always.
The configured CTC decoder has no weights in the checkpoint (funasr disables it at load), so there are no
CTC timestamps to port.

## 8. Fixtures and the gate vocabulary

155 clips: the 5 upstream `example/*.mp3` and FLEURS test `en_us` / `cmn_hans_cn` / `ja_jp`, 50 each
(CC BY 4.0, first 50 ids ≤ 30 s). The headline is token agreement with the fp32 oracle, counted with the
zoo's margin rule: a divergence at a step whose oracle top-2 softmax gap is below 0.1 is a knife-edge and is
reported separately, never as a failure; a divergence above the floor fails. WER (en) and CER (zh, ja) are
computed twice, port vs oracle and both vs the FLEURS reference, under one normalizer (NFKC, lowercase,
Unicode P*/S* removed, whitespace collapsed; CER over characters with whitespace removed), so the port's
distance to the model and the model's distance to the truth stay separate. FLEURS `transcription` is
lowercased and unpunctuated, spaces every Chinese character, and writes Japanese numbers in digits where the
model writes them in kanji, so the vs-FLEURS rate carries normalization noise the vs-oracle rate does not.

## 9. Swift: the DFT in Double

The Swift front end (`FunASRFbankPreprocessor`, CoreAIKit) mirrors `frontend.py` line by line. With the
400-term DFT dot products in Float, the worst clip sat 2.3e-3 from the oracle's float32 torchaudio fbank,
and one knife-edge step (oracle gap 0.0036) flipped relative to the Python engine. DFT, mel projection and
log in Double: 1.3e-3, ids equal to the Python engine on all 155 clips, +1.7 ms per 13.6 s clip.
