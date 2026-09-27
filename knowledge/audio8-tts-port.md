# Audio8-TTS-Preview-0.6b on Core AI — port notes

Engineering notes from porting [`Edge0/Audio8-TTS-Preview-0.6b`](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.6b)
(Apache-2.0; 601M + a 337M codec) to Core AI: a **DualAR** text-to-speech model of the Fish Audio S2 Pro design — a
Qwen2.5-shaped **slow AR** (24 layers, 896 wide, 14 heads / 2 KV heads) that predicts one *semantic* token per 46 ms
frame, a 4-layer **fast AR** that predicts the frame's nine other codebooks one after another, and a 44.1 kHz
**DAC-style codec** (semantic book 4,096 × 8, nine residual books 1,024 × 8, a window-128 transformer, a SEANet decoder
to 2,048 samples per frame). Eleven languages, zero-shot voice cloning from a 0.5–30 s reference. The zoo's first
DualAR / semantic-token TTS and its first sampler inside a graph. Card:
[`models/audio8-tts/README.md`](../models/audio8-tts/README.md); scripts:
[`conversion/audio8_tts/`](../conversion/audio8_tts/README.md).

## The shape of the port

| stage | Core AI form |
|---|---|
| prompt | host: the publisher's segments encoded one at a time (`<\|im_start\|>system\n`, the system line, …, `<\|voice\|>`), packed as `[11, P]` — row 0 ids, rows 1..10 the reference's codebooks under its semantic ids |
| slow AR prefill | `prefill(codes [1,11,32], pos)` in windows over one KV state pair `[24,1,2,2048,64]`; the last real row's logits (4,097 rows: the semantic ids then eos — the only rows the sampler can pick) and post-norm hidden |
| one frame | `frame(codes [1,11,1], pos, noise_slow [2,4097], window [10], noise_fast [9,4096], forced [11], use_forced)`: the slow step, the semantic draw with the RAS rule, the fast AR's ten rows and nine codebook draws — **one runtime call per frame** |
| codec | `main(codes [1,10,160]) -> wav [1, 327680]`, stateless; every op causal, so a stream decodes in 160-frame windows and keeps the last 32 |
| voice registration | `main(audio [1,1,442368]) -> codes [1,10,216]` (the encoder, 10 s bucket) |

## What made it work, and what did not

1. **The sampler belongs in the graph.** The first export split the frame the way the publisher's ONNX runtime does —
   a slow-AR call, then nine fast-AR calls, sampling on the host — and ran at **~48 ms of engine time per 46 ms
   frame** on an M4 Max (slow 16.6 ms, fast step 3.3 ms × 9), real time at best. A Core AI call on the raw runtime
   path costs a few milliseconds before any arithmetic, in Python and in Swift alike. Folding the frame into one
   call needs the sampler inside the graph, and the publisher's processor can be written without a sort: `topk(50)`
   bounds the top-p candidates (top-k removes everything else anyway), `logsumexp` gives the full-vocabulary softmax
   normaliser, a cumulative sum over the 50 sorted probabilities gives the top-p cut with the best always kept, the
   kept scores are divided by the temperature, and the draw is `argmax(softmax(kept) / -log(u))` with `u` gathered at
   the candidates' indices — the Gumbel-max form `ArkttsModel._sample` uses. The randomness is the vector `u`, an
   input, so a recorded `u` replays the publisher's choice exactly on identical logits: the graph's own draws match
   the NumPy sampler 20/20 on random logits and the oracle's tokens frame for frame under teacher forcing. The fused
   frame runs in **~32 ms** on the same machine under the same load (about a third of it the slow step); the fast
   AR's ten rows are unrolled with their keys concatenated in-graph, so it needs no state and no call of its own.
2. **Where the remaining milliseconds are not.** Tested one variable at a time on the slow step: a 512-slot cache
   costs the same as 2,048 (22.45 vs 22.46 ms under load); int8 linears save ~4 ms over fp16 (weight bandwidth is
   a fifth of it); Apple's composite RMSNorm + RoPE externalized save ~10 %; AOT compilation for the Mac GPU (h16c)
   changes nothing (32.2 vs 32.2 ms per frame). What is left is the per-layer dispatch of a 24-layer S = 1 graph on
   the raw runtime path (~0.4–0.6 ms per transformer layer, the unrolled fast AR's 40 layer-rows included). Apple's
   pipelined engine drives a same-sized Qwen3-0.6B at 2.8 ms per token (the zoo's Fun-ASR decoder), so the lever
   for a 3–4× faster frame is that engine — its per-token inputs would have to carry the 11-row code column and its
   outputs the 4,097 logits and the hidden — a separate round, not this port.
3. **The 4,097-row head is exact, not an approximation.** The publisher's `ArkttsSemanticLogitsProcessor` sets every
   logit outside the semantic range and eos to -inf before sampling, so the head can be the embedding table's 4,097
   rows (7 MB) instead of 155,776 (279 MB fp16). Over 1,695 oracle steps the full-vocabulary argmax never fell outside
   those rows either.
4. **The publisher's RoPE tables are bfloat16.** `_precompute_rope` rounds cos / sin to bf16 in the fp32 model too, so
   the fp32 oracle carries ~3-digit angles. The port bakes the same bf16-rounded tables as fp32 constants (interleaved
   pairs, GPT-J style — `apply_rope_pairs`, not rotate-half); swapping in exact fp32 angles moves the fp16 logits to
   cos 0.9991 against the bf16-table build. Apple's `RoPE(interleaved=True)` composite implements the same pairing
   (max |Δ| 0.0 against the publisher's rotation on the `[B, H, S, D]` layout).
5. **No massive-activation trouble this time — measure anyway.** The slow AR's residual stream peaks at 12,827 from
   layer 3 on (Qwen2.5's usual position-0 spike), a fifth of float16's 65,504: the fp16 graph needs no residual
   rescale, unlike the zoo's Fun-ASR decoder (125k). The fast AR peaks at 115, the codec at 672.
6. **int8 on the fast AR is the wrong 53 MB.** On 15,093 teacher-forced codebook draws the fp16 fast AR reproduces
   the oracle's draw 15,019 times (99.5 %, logits cos min 0.99989); int8 drops to 14,457 (95.8 %, cos min 0.984).
   The slow AR takes int8 well: 1,671 / 1,695 semantic draws (fp16: 1,688), and 21 of its 24 misses sit at a Gumbel
   margin below 0.13. The fp16 arm misses one draw at a margin of 0.95 — under sampling, an fp16 GPU pass alone can
   move a decisive draw when the top-p cut shifts by one entry, so the sampled-agreement count is read together with
   the end-to-end audio, never alone.
7. **The RAS window has a quirk worth reproducing.** `generate` creates `previous` as zeros *after* step 0, so the
   first frame's semantic never enters the 10-token window; from step 2 the window rolls. The publisher's ONNX runtime
   appends from step 0 — it does not reproduce `generate`. The oracle here is `generate` (asserted identical to the
   recorded loop with the same seed), and the graph takes the window as an input (-1s for "none").
8. **The fast head is wider than the codec.** `fast_output` has 4,096 rows for every codebook, but codebooks 1..9 of
   the codec have 1,024 entries; the publisher's `decode()` clamps. In 8,000 oracle draws one codebook sample exceeded
   1,023, so the clamp is in the codec graph too.
9. **Gate the audio, not the tensors.** A TTS port's tensors can match while the speech does not, so the ladder ends
   in the zoo's Fun-ASR fp32 oracle transcribing every generated wav (ja CER / en WER / zh CER against the fixture
   text, the publisher's own audio scored alike) and, for the clone fixtures, a WavLM-SV x-vector cosine to the
   reference clip — the port lands where the oracle lands on both.
10. **The Python bindings leak an IOSurface per call.** A fixture is ~2,000 calls on the split graphs; the gate died
    after seven fixtures (`Failed to allocate storage for NDArray … sk: ioSurface`). One child interpreter per fixture
    (pocket-tts-port.md, defect 2). The fused graph cuts a fixture to ~200 calls; the workers stay.

## Reproduce

```bash
V=~/code/coreai/coreai-models/.venv/bin/python; C=conversion/audio8_tts
$V $C/oracle_audio8.py --check-generate          # fp32 oracle (publisher's code), every draw recorded
$V $C/prompt.py                                   # tokenizer-only prompt == the processor's, 18/18
$V $C/parity_audio8.py                            # plain-torch re-author vs oracle, eager fp32
$V $C/export_audio8_frame.py --mode int8          # the fused asset (+ eager check, quick GPU gate)
$V $C/export_audio8.py --part codec               # the codec decoder
$V $C/audio8_encoder.py --frames 216              # the encoder (voice registration)
$V $C/gate_audio8_frame.py --mode int8            # 18 fixtures: teacher-forced, codec, free run, ASR, speaker
```
