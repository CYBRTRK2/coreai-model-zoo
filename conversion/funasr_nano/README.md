# Fun-ASR-Nano-2512 → Core AI — conversion, oracle and gates

Port of [`FunAudioLLM/Fun-ASR-Nano-2512`](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)
(Apache-2.0): a 70-layer SAN-M speech encoder + 2-block adaptor feeding a fine-tuned Qwen3-0.6B
decoder. Card: [`models/funasr-nano/README.md`](../../models/funasr-nano/README.md); lessons:
[`knowledge/funasr-nano-port.md`](../../knowledge/funasr-nano-port.md).

Two venvs, as in `nemotron_asr/`: the **oracle venv** runs the publisher's `funasr` package
(1.4.16) in fp32; the **shared coreai-torch venv** re-authors, exports and gates. No funasr
import anywhere in the export path — the encoder and decoder are plain torch over
`model.safetensors`.

```
wav 16 kHz mono
 └─ host: kaldi fbank 80 (hamming 25/10 ms, pre-emphasis 0.97, ×32768, dither 0) → LFR 7/6 → [L, 560]
     └─ encoder .aimodel (static L=500): feats[1,500,560] f32 + mask[1,500] f32 → audio_embeds[63,1024] f32
         └─ host: rows [:N], N = ceil(L/8); prompt = prefix(18) + [vocab+slot]×N + suffix(5)
             └─ decoder _s1 bundle (Qwen3-0.6B, int8 linears, tied fp16 head, residual ×1/4 in-graph) → greedy → text
```

## Scripts (run order)

| step | script | venv | writes |
| --- | --- | --- | --- |
| fixtures | `make_fixtures.py` | shared | `_funasr_nano/fixtures/` — the 5 upstream `example/*.mp3` as 16 kHz wav + FLEURS test `en_us` / `cmn_hans_cn` / `ja_jp`, 50 clips each (CC BY 4.0), `meta.json` with source, license, sha256, reference text |
| oracle | `make_oracle.py` | oracle | `_funasr_nano/oracle/<clip>.npz` (LFR features, encoder_out, adaptor_out, N, prompt ids with 0 placeholders, gen_ids) + `oracle.json`; `--case`/`--hotwords`/`--language`/`--no-itn` variants into `oracle_hotwords/` |
| front end | `frontend.py`, `gate_frontend.py` | shared | NumPy kaldi fbank + LFR (the Swift spec), gated vs the oracle features (max abs Δ 1.39e-3 over 155 clips) |
| encoder | `funasr_encoder.py`, `parity_encoder.py` | shared | plain-torch SAN-M + adaptor from safetensors; fp32 parity vs the oracle (cos min 0.999999999) |
| encoder export | `export_encoder.py --dtype fp16w32`, `gate_encoder.py` | shared | `_funasr_nano/exports/funasr_nano_audio_encoder_fp16w32_l500.aimodel` (450 MB); per-row cos vs the oracle on the Mac GPU (min 0.9999982); `--dtype fp16` / `fp32` are the evidence variants |
| prompt / decoder | `prompt.py`, `parity_decoder.py`, `funasr_decoder.py` | shared | prompt ids asserted against the oracle's on 155 clips; HF fp32 greedy == oracle gen_ids 155/155; per-step top-2 margins (`logs/r2_margins.json`) |
| decoder export | `export_decoder.py --mode int8lin`, `parity_residual_scale.py` | shared | `_funasr_nano/exports/funasr_nano_2512_decode_int8lin_n63_s1/` (ship, 759 MB) + `funasr_nano_2512_int8lin_unified_cl1024/` (gate twin); the residual-scale equivalence check |
| e2e gate | `gate_e2e.py --enc fp16w32 --dec int8lin`, `metrics.py` | shared | wav → text on the Core AI engine, 155 clips: exact / knife-edge / mismatch vs the oracle, WER (en) and CER (zh, ja) vs the oracle and vs FLEURS |
| evidence | `fp16_headroom.py` | shared | residual-stream abs-max of the fp32 encoder over the fixtures (fp16 headroom 1.39×) |
| Swift reference | `dump_features.py` | shared | `_funasr_nano/swift_ref/` — LFR features per clip, expected texts/ids, prompt variants for `Tests/CoreAIKitTests/FunASRSmokeTests.swift` in CoreAIKit |
| ship | `stage_ship.py` | shared | lays out the Hugging Face repo in `_funasr_nano/ship/` (no upload) |

Paths resolve through [`../_paths.py`](../_paths.py): the work directory is
`$ZOO_WORK_ROOT/_funasr_nano/` (default `~/code/coreai/_funasr_nano/`) with `hf/model.safetensors`
(the vLLM repo's file, sha256 `96dfbec4…`), `official/` (the official repo snapshot: `config.yaml`,
`Qwen3-0.6B/` tokenizer + config, `example/`, `model.pt` for the oracle only), `venv-oracle/`.

## Gate results (Mac, M4 Max, macOS 27, GPU; 155 clips = 5 upstream examples + 150 FLEURS)

| stage | result |
| --- | --- |
| NumPy front end vs oracle LFR features | max abs Δ 1.39e-3 (log-mel scale), L and N equal on every clip |
| torch fp32 encoder + adaptor vs oracle | per-row cos min 0.999999999, max abs Δ 8.3e-3 (ref abs-max ~1300) |
| encoder `.aimodel` fp16w32 (ship) vs oracle | per-row cos mean 0.99999988 / min 0.9999982, max abs Δ 0.27 |
| encoder `.aimodel` fp16 (evidence) | cos min 0.9916 on one clip (en_us_1664), residual headroom 1.39× — not shipped |
| HF fp32 decoder, our prompt, greedy | gen_ids == oracle 155/155 |
| end-to-end, encoder fp16w32 + decoder int8lin | 150/155 token-exact; the other 5 diverge only at steps where the oracle's top-2 gap is 0.007–0.033 and pick the oracle's runner-up; 0 divergences above the 0.1 floor; 0 clips hit the 512-token cap |
| same, decoder fp16 (control, 1.1 GB) | 154/155 (encoder fp16) · 155/155 (encoder fp32) |
| same, decoder int8hu (untied int8 head, evidence) | 144/155, all knife-edge |
| WER en / CER zh / CER ja, port vs oracle (int8lin) | 0.34 % / 0.00 % / 0.07 % |
| WER en / CER zh / CER ja vs FLEURS `transcription`, oracle → port (int8lin) | 5.08 → 5.34 % / 6.86 → 6.86 % / 6.95 → 6.92 % (n = 50 per language; normalizer in `metrics.py`) |

The FLEURS numbers are our own measurement of the publisher's fp32 model on 50 test utterances
per language with a stated normalizer; they are not the publisher's benchmark table.
