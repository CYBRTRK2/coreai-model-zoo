# Audio8-TTS-Preview-0.6b → Core AI — conversion, oracle and gates

Port of [`Edge0/Audio8-TTS-Preview-0.6b`](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.6b) (Apache-2.0): a
DualAR text-to-speech model (Fish Audio S2 Pro design) — a 24-layer Qwen2.5-shaped **slow AR** predicting one semantic
token per 46 ms frame, a 4-layer **fast AR** predicting the frame's nine other codebooks, and a 44.1 kHz **codec**
(DAC-style, 10 codebooks: one semantic book of 4,096 and nine residual books of 1,024). Eleven languages, zero-shot
voice cloning from a 0.5–30 s reference. Card: [`models/audio8-tts/README.md`](../../models/audio8-tts/README.md);
lessons: [`knowledge/audio8-tts-port.md`](../../knowledge/audio8-tts-port.md).

One venv: the shared coreai-torch venv runs the publisher's remote code as the fp32 oracle (it targets
transformers 4.57) and re-authors, exports and gates. The ASR round trip runs in the Fun-ASR port's oracle venv.

```
text (+ reference text + reference codes)
 └─ host: prompt = segments encoded one at a time → [11, P] (prompt.py)
     └─ dualar .aimodel   prefill(codes [1,11,32], pos) in windows                                → logits [32,4097] + hidden [32,896]
                          frame(codes [1,11,1], pos, noise_slow [2,4097], window [10], noise_fast [9,4096], forced, use_forced)
                            = slow step + semantic draw (top-k 50 / top-p 0.9 / T 0.7, Gumbel-max, RAS window 10)
                              + the fast AR's 10 rows with 9 codebook draws — ONE call per frame   → semantic, codes [10], logits, hidden
                          first_frame(logits, hidden, …) for the frame after the prefill
         └─ host: RAS window, the uniform draws (a seeded stream; the gate replays the oracle's)
             └─ codec decoder .aimodel  codes [1,10,160] → wav [1, 327680] @ 44.1 kHz             (160-frame windows, keep the last 32)
voice registration: wav [1,1,442368] → codec encoder .aimodel (fp16 weights, fp32 arithmetic) → codes [1,10,216]
```

The first export split the frame the way the publisher's ONNX runtime does (a slow-AR call + nine fast-AR calls,
sampling on the host: `export_audio8.py --part slow|fast`, `gate_audio8.py`, `sampler.py`, `replay.py`) and ran at
~48 ms of engine time per 46 ms frame on an M4 Max — a Core AI call costs a few milliseconds before any arithmetic.
Those scripts stay as the diagnostic path (they gate the slow and fast graphs *separately*, logits vector by logits
vector); the shipped asset is the fused one (`audio8_frame.py`, `export_audio8_frame.py`, `gate_audio8_frame.py`).

## Scripts (run order)

| step | script | writes |
| --- | --- | --- |
| fixtures | `fixtures.json` | 14 no-reference sentences (6 ja, 6 en, 2 zh, our own text) + 4 voice-clone fixtures whose reference clips are FLEURS test utterances (CC BY 4.0, from the Fun-ASR port's fixture set) |
| oracle | `oracle_audio8.py --check-generate` | `_audio8_tts/oracle/<name>.npz` — the fp32 CPU run of the publisher's code: prompt, every slow-logits vector (the 4,097 rows the sampler reads), slow hidden, every fast-logits vector, the uniform draws of every sample, codes, wav; `--check-generate` asserts the recorded loop equals `model.generate` with the same seed |
| prompt | `prompt.py` | the prompt from the tokenizer alone, asserted id for id against the 18 oracle prompts (the Swift host's spec) |
| re-author | `audio8_model.py`, `parity_audio8.py` | plain-torch slow AR / fast AR / codec decoder from `model.safetensors` + `codec.pth`; fp32 eager teacher-forced replay vs the oracle (argmax and replayed sample equal at every step, codec cos 1.000000) |
| fused frame | `audio8_frame.py`, `export_audio8_frame.py --mode int8` | the in-graph sampler + unrolled fast AR; eager fp32 check (every draw equals the oracle's), then `_audio8_tts/exports/audio8_dualar_int8_cl2048_w32.aimodel` (prefill / frame / first_frame), quick-gated on the GPU |
| codec | `export_audio8.py --part codec --codec-frames 160` | `audio8_codec_decoder_fp16_t160.aimodel`, quick-gated on the oracle codes |
| encoder | `audio8_encoder.py --frames 216 --dtype fp16w32` | `audio8_codec_encoder_fp16w32_t216.aimodel` (voice registration): eager fp32 codes exact on the four reference clips (712/712 frames); on the GPU 691/712 frames exact, codebook 0 exact on every frame, every codebook ≥ 97.7 % (the whole-fp16 export: 44–121 of ~200 frames — the nearest-neighbour choice flips on fp16 noise and the residual books cascade) |
| gate | `gate_audio8_frame.py --mode int8` | `_audio8_tts/gate/<tag>/gate.json` + wavs: teacher-forced replay (per frame the slow logits, hidden and nine fast-logits vectors vs the oracle, and the in-graph sampler's draws vs the oracle's tokens), codec decode of the oracle codes, free run with the oracle's draws, then `asr_judge.py` (Fun-ASR fp32 WER/CER on every wav) and `speaker_sim.py` (WavLM-SV x-vector cosine vs the reference clip) |
| split-graph diagnostics | `export_audio8.py --part slow|fast --mode int8|fp16`, `gate_audio8.py`, `sampler.py`, `replay.py` | the slow and fast graphs as separate assets with the sampler in NumPy: the arm table below, and where the per-call cost was measured |
| Swift reference | `dump_swift_ref.py --tag <ship arm>` | `_audio8_tts/swift_ref/` — the recorded draws, prompts and the Python engine run's codes for `Audio8SmokeTests` (CoreAIKit) and `apps/Audio8Gate` |
| voices | `register_voice.py` | `voice.json` (reference text + codes) from a wav, through the fp32 codec or the exported encoder |
| ship | `stage_ship.py --with-encoder` | the Hugging Face repository layout in `_audio8_tts/ship/` (no upload) |

Paths resolve through [`../_paths.py`](../_paths.py): the work directory is `$ZOO_WORK_ROOT/_audio8_tts/` (default
`~/code/coreai/_audio8_tts/`); the checkpoint is read from the Hugging Face cache at the pinned revision.

## Gate results (Mac, M4 Max, macOS 27, GPU; 18 fixtures = 1,695 slow steps, 15,093 fast draws)

The shipped asset (`gate_audio8_frame.py`, sampler in the graph), then the split-graph arms it was chosen from
(`gate_audio8.py`, sampler in NumPy, the same fixtures):

| stage | **ship: dualar int8 (slow int8, fast fp16), one call per frame** | split: int8 slow + fp16 fast | split: int8 slow + int8 fast | split: fp16 slow + fp16 fast |
| --- | --- | --- | --- | --- |
| slow logits: the sampler's draw == oracle | **1,671 / 1,695** (21 of the 24 misses at a Gumbel margin < 0.13) | 1,671 / 1,695 | 1,671 / 1,695 | 1,688 / 1,695 |
| slow logits cos min · hidden cos min | 0.99985 · 0.99991 | 0.99985 | 0.99985 | 1.00000 |
| fast logits: the sampler's draw == oracle | **14,711 / 15,093** | 15,019 / 15,093 | 14,457 / 15,093 | 15,019 / 15,093 |
| fast logits cos min | 0.99805 | 0.99989 | 0.98428 | 0.99989 |
| codec decoder (oracle codes): wav cos min / log-mel cos min | 0.99978 / 0.99876 | same | same | same |
| free run reached eos | 18 / 18 | 18 / 18 | 18 / 18 | 18 / 18 |
| ASR round trip vs the fixture text (Fun-ASR fp32): ja CER / en WER / zh CER | 2.3 % / 0.9 % / 0.0 % (oracle's own audio: 6.5 / 0.0 / 0.0) | 2.3 / 0.0 / 0.0 | 2.3 / 0.0 / 0.0 | 4.2 / 0.0 / 0.0 |
| speaker cosine to the reference (WavLM-SV), 4 clone fixtures: mean / min | 0.855 / 0.561 (oracle 0.854 / 0.582) | 0.873 / 0.611 | 0.865 / 0.594 | 0.858 / 0.585 |
| engine time per frame, Python runtime, machine under other load | **32.9 ms** (one call) | 16.6 + 1.6 + 8 × 3.3 = 44.6 ms (10 calls) | — | — |

The fp32 eager re-author (and the fused module in fp32) matches the oracle at every draw; every miss above is fp16 GPU
or int8 arithmetic moving a draw that sat within a hair of the runner-up. The fused asset's fast-AR agreement is two
points under the split fp16 graph's (its ten rows attend over concatenated keys instead of a state; the difference is
in the fp16 attention arithmetic, cos min 0.998) and its audio lands where the oracle's does. The ASR and speaker rows
are eight to fourteen sentences per language, our own measurement with one normalizer — they say the port's speech is
as intelligible and as speaker-faithful as the publisher's fp32 output on these fixtures, not more; the one English
"error" is `calendars` → `calendar` on a clone fixture.
