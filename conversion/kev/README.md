# kev — Kev-0.8B and Kev-4B on Core AI

Export and gate scripts for [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) and
[jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) (Hub tag `v1.0`, Apache-2.0; code at
[github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev) tag `kev-1.0`). Cards:
[`models/kev-0.8b/README.md`](../../models/kev-0.8b/README.md), [`models/kev-4b/README.md`](../../models/kev-4b/README.md);
port notes: [`knowledge/kev-port.md`](../../knowledge/kev-port.md); Swift host: [`apps/Kev`](../../apps/Kev/); device gate:
[`apps/KevGate`](../../apps/KevGate/).

Kev is a decision model: a rank-16 LoRA and a pointer head on Qwen3.5-0.8B-Base / Qwen3.5-4B-Base. It reads a state and
typed questions (`noul`, `choice`, `score`), one row per question, and returns a probability per option; it never
generates text. Requests and responses use the SystemOne-compatible request shape. The port is the author's own merge,
the overlay's text decoder returning the final-norm hidden state at every position (a static number of tokens a call,
`--prefill-chunk`; no vocabulary head), and the pointer head on the host. The published Kev-0.8B bundle takes 128 tokens
a call and runs each linear-attention layer's recurrence in the overlay's fp32 Metal kernel (§10, §12). Every gate compares with the author's own fp32 code.

| file | what it does |
|---|---|
| `make_fixtures.py` | the fixture: transfer-v4 development (tag `kev-1.0`) first 60 lines, the first 20 records of each of its seven other sources, its first 20 `score` records; SemIf authored144 as 144 `choice` questions (MIT notice in `LICENSE-SemIf`); 20 records written for the port. `--heldout`: the next 130 transfer-v4 records The 20 own records come from `own_records_public.json` (11 with text, 9 as stubs: an invented name in each turned up in use on the web); the 9 full records are a private file outside the repository, so a fresh clone builds 375 records. |
| `oracle_kev.py` | the author's package unchanged: `Checkpoint.load("cpu", fp32)`, `kev.model.admit` + the row-form `forward` per record, with asserts on the readout; `--phase nomerge` the unmerged adapter on a subset; `--checkpoint` / `--base` for Kev-4B, `--fixtures` / `--out-dir` / `--results-dir` for other files |
| `export_head.py` | the pointer head as `head.safetensors` (q / k weight and bias, fp32) + `kev_head.json` (scale, temperature, delimiter ids, provenance); `--model kev-4b` |
| `parity_merged_torch.py` | the author's merged checkpoint through the author's loader against the oracle on every question (`--phase answers`), and `W + (B @ A) · α / r` on three tensors (`--phase formula`) |
| `qwen3_5_kev_decoder.py` | the decoder module: the overlay's stateful Qwen3.5 text decoder on the merged weights, `lm_head` = identity, the final-norm hidden state at every position of a static-S chunk; `from_hf` reads the author's full-weight checkpoint and reports every key (`load_report`) |
| `parity_decoder_torch.py` | that module in fp32 torch (CPU) driven like the graph, through the pointer head, against the oracle: `probe` (threads, conv form), `p1` every question, `p2` the overlay's plain `forward_stateful` bit for bit, `p3` chunk widths, `merge` |
| `export_decoder.py` | the bundle: `fp16`, `int8lin`, `int8mix --fp16-layers`; one function `main` at a static `--prefill-chunk` S; `--model kev-4b`; `--fp16-kinds`, `--quant-block`, `--quant-scheme`, `--embed-int8`; `--gdn-scan unroll\|chunk\|metal` (round 11), `--dynamic-query` (round 14: the query length 2..cap); `--query-call-max` / `--query-multiple` (round 15: the host's call lengths in `metadata.json`), `--metadata-only` / `--from-aimodel` / `--bundle-dir` (round 15: a bundle around an existing, gated `.aimodel`); `--aot` the h16c `.aimodelc`; `--record` a JSON log. Checks the merged weights' sha256 first |
| `readout_gate.py` | the AOT graph on the Mac GPU (Python runtime, never the JIT), its fp16 hidden rows through the fp32 pointer head, against the oracle: `run` (`--red`, `--oracle`, `--rows`, `--aimodelc`, `--compare-with`; a dynamic-S bundle: `--shared`, `--records`, the call plan from its metadata, `--host-q` / `--call-max` to override), `red` (a fixed rows file), `merge` |
| `int8_bisect_torch.py` | which linears carry the int8 error, in fp32 torch with the exporter's own int8 weights: `rule` (written before any result), `dump`, `run`, `plan`, `merge`; `--model kev-4b` |
| `host.py` | the host specification (NumPy, `tokenizers`, json): the request checks, `render` / `option_text` / the keys, `user_tokens`, the rows and their readout indices, the call plan (`graph_shape`, `plan`, `call_lengths`), the graph's row limit, the shared-prefix plan, the float64 head, `to_answers` and the response |
| `test_host.py` | `host.py` against every oracle record of both checkpoints with two tokenizers, a table of valid and invalid requests, two negative controls; the call plan for every row length and the plan settings, against `readout_gate.py`'s own |
| `decide.py` | the Python reference read-out on the AOT graph: `run` (a request → a response, `--shared`, `--warm`), `check` (every record from its raw request against the readout gate and the oracle; `--shared-gate-transcript` for a dynamic-S bundle); `--call-max` / `--multiple` override the bundle's call plan |
| `timing.py` | decision latency in one `_GPU_LOCK` window (Python runtime) |
| `gate_swift.py` | scores the Swift CLI (`apps/Kev`): `render`, `score` (`--records`, `--shared-gate-*` for a dynamic-S bundle), `negative`, `jit`, `longrow-*`, `pytime` (`--warm`), `timing`; `--swift-dir` keeps a round's files apart |

## Environment

- **The author's code** (`make_fixtures.py`, `oracle_kev.py`, `export_head.py`, `parity_merged_torch.py`, the merge, the
  host test in the author's venv): the author's repository at tag `kev-1.0` and an environment from its own `uv.lock`:

  ```bash
  git clone --branch kev-1.0 --depth 1 https://github.com/jaredpalmer/kev.git $ZOO_WORK_ROOT/_kev/kev-src
  uv export --project $ZOO_WORK_ROOT/_kev/kev-src --frozen --no-dev --no-emit-project --no-hashes -o req.txt
  uv venv --python 3.12 venv-oracle && uv pip install --python venv-oracle/bin/python -r req.txt
  uv pip install --python venv-oracle/bin/python --no-deps -e $ZOO_WORK_ROOT/_kev/kev-src
  # torch 2.8.0, transformers 5.17.0, peft 0.21.0, numpy 2.5.3, huggingface_hub 1.32.0
  ```

- **Export and gates:** the zoo's overlay venv (coreai-core 1.0.0b2, coreai-torch 0.4.1, torch 2.9.0, transformers 4.57.6),
  called `$PY` below; Xcode 27.0 RC as `DEVELOPER_DIR`.
- **Weights:** `HF_HOME=$ZOO_WORK_ROOT/_kev/hf`, `HF_HUB_DISABLE_XET=1`, then `HF_HUB_OFFLINE=1`. huggingface_hub 1.32
  resolves a 40-hex revision offline only after one online `snapshot_download` with that venv (it writes the tree
  listing); the Hub resolves the tag `v1.0` (`788ddbdd` / `591dcb5b`) to commits `bf75a6a8` / `6cfce5c2`.

Run from `conversion/kev`. `K=$ZOO_WORK_ROOT/_kev`; everything the scripts write outside the repository goes under it
(`conversion/_paths.py`).

## 1. Fixture and oracle

```bash
python make_fixtures.py                    # -> $K/fixtures/records.json (384 records, 434 questions), LICENSE-SemIf
python make_fixtures.py --heldout          # -> $K/fixtures/heldout.json (130); never used to choose anything
HF_HUB_OFFLINE=1 python oracle_kev.py --threads 1
HF_HUB_OFFLINE=1 python oracle_kev.py --phase nomerge --threads 1
HF_HUB_OFFLINE=1 python oracle_kev.py --threads 1 --fixtures $K/fixtures/heldout.json \
    --out-dir $K/oracle/heldout --results-dir $K/oracle/heldout
# Kev-4B: the same with
#   --checkpoint jaredpalmer/kev-4b@591dcb5bd6d05eb0b5131ea6608f93f10243335c
#   --base Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b --out-dir $K/oracle_4b --summary-name oracle_summary_4b.json
```

The oracle asserts, per record, that each row is state + branch at positions 0..L − 1, that the delimiter ids sit at the
readout indices, that the head re-run on the recorded hidden state equals `forward()`'s logits bit for bit, and that the
raw logits divided by T equal them; record 0 runs again and must be bit-identical; 30 records also run the serving path
(`model.probs`). Run it with `--threads 1`: transformers' reference causal conv is slow and changes its last bits with
more threads. On another fixture file pass `--out-dir` as well as `--results-dir`: the default output directory is the
fixture oracle's, whose `records_oracle.json` it would replace.

## 2. The merge, the head and merged parity

```bash
(cd $K/kev-src && HF_HUB_OFFLINE=1 python scripts/merge_lora_checkpoint.py \
    --lora jaredpalmer/kev-0.8b@788ddbdd65715bb03a56788c822f6c632c9a551d --out $K/merged/kev-0.8b-v1.0)
HF_HUB_OFFLINE=1 python export_head.py                                   # -> $K/oracle/head/
HF_HUB_OFFLINE=1 python parity_merged_torch.py --phase formula
HF_HUB_OFFLINE=1 python parity_merged_torch.py --phase answers
# Kev-4B: --lora jaredpalmer/kev-4b@591dcb5bd6d05eb0b5131ea6608f93f10243335c --out $K/merged/kev-4b-v1.0;
#   export_head.py --model kev-4b --out-dir $K/oracle_4b/head --oracle-dir $K/oracle_4b;
#   parity_merged_torch.py --phase formula|answers --model kev-4b --merged $K/merged/kev-4b-v1.0/checkpoint --oracle-dir $K/oracle_4b
```

The exporter reads the merged checkpoint through a local HF-cache id: `$K/hf/hub/models--kev-local--kev-0.8b-v1.0-merged/
snapshots/ab6bd41853ef/` with symlinks to the merged `config.json`, `model.safetensors`, `tokenizer.json` and
`tokenizer_config.json`, and `refs/main` = `ab6bd41853ef` (Kev-4B: `models--kev-local--kev-4b-v1.0-merged/snapshots/
904380cbf0be/` with the config, the index and the four shards). Transcripts: `gate-kev-<size>-merge.json`.

## 3. The decoder module in fp32 torch

```bash
export HF_HOME=$K/hf HF_HUB_OFFLINE=1 DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
$PY parity_decoder_torch.py probe
for k in 0 1 2; do $PY parity_decoder_torch.py p1 --shard $k --shards 3 --threads 1 --conv bmm & done; wait
$PY parity_decoder_torch.py p2 --threads 1 --conv bmm
$PY parity_decoder_torch.py p3 --threads 1 --conv bmm
$PY parity_decoder_torch.py merge
# Kev-4B: --model kev-4b on each; p3 --p3-rows tv4_000:0,semif_a3f18f3a63d45345942b:0,own_L02:0 --p3-widths 16,64
```

The harness evaluates the Gated DeltaNet's depthwise conv as one `bmm` over the windows and checks it against
`F.conv1d` at the last chunk of every row; one thread is the fastest setting. Transcripts: `gate-kev-<size>-torch-parity.json`.

## 4. Export and the Mac readout gate

```bash
$PY export_decoder.py fp16 --prefill-chunk 16 --aot --record $K/results/export_fp16_pf16.json
$PY readout_gate.py run $K/exports/bundles/kev_0_8b_decode_fp16_pf16 --red --transcript $K/results/readout_fp16_pf16.json
$PY readout_gate.py red $K/exports/bundles/kev_0_8b_decode_fp16_pf16 --red-file $K/readout/red_arms_v2.json \
    --gate-transcript $K/results/readout_fp16_pf16.json --transcript $K/results/readout_fp16_pf16_red_v2.json
$PY readout_gate.py run $K/exports/bundles/kev_0_8b_decode_fp16_pf16 --oracle $K/oracle/heldout \
    --tag heldout_fp16_pf16 --transcript $K/results/readout_heldout_fp16_pf16.json
# Kev-4B: export_decoder.py fp16 --model kev-4b ...; readout_gate.py run|red ... --model kev-4b --tag ..._4b
```

The bundle lands in `$K/exports/bundles/<name>/` (`.aimodel`, `metadata.json`, `tokenizer/`, `head/`), the AOT asset in
`$K/exports/bundles_aotc/`. Each gate process takes at most 40 rows plus a re-run of its first, which must reproduce its
hidden rows bit for bit. The Python runtime caches every AOT asset it loads under `~/Library/Caches/coreai-cache/<build>/
python/<sha256 of main-h16c.mlirb>/`; delete the entry after the gate. Two exports of the same S are not byte-identical
(the compiled `main-h16c.mlirb` is): ship the bytes you gated. The chunk widths 32 / 64 / 128 ran the same way
(`--prefill-chunk`, `--subset s80`). Transcripts: `gate-kev-<size>-readout-fp16_pf16.json`, `-heldout.json`,
`gate-kev-0.8b-readout-s.json`.

## 5. int8, and how the fp16 sets were chosen

```bash
$PY export_decoder.py int8lin --prefill-chunk 16 --aot --record $K/results/export_int8lin_pf16.json
$PY readout_gate.py run $K/exports/bundles/kev_0_8b_decode_int8lin_pf16 --tag int8lin_pf16 \
    --compare-with $K/results/readout_fp16_pf16.json --transcript $K/results/readout_int8lin_pf16.json   # FAILS the bar
$PY int8_bisect_torch.py rule --gate $K/results/readout_int8lin_pf16.json
$PY int8_bisect_torch.py dump
$PY int8_bisect_torch.py run --part $K/bisect/parts/part_a.json --configs all_int8,exact,layer_00_fp16,..
$PY int8_bisect_torch.py plan          # the candidate sets the rule asks for next
$PY int8_bisect_torch.py merge --out $K/results/r3_int8_bisect.json
$PY export_decoder.py int8mix --fp16-layers 0,1,2,3,4,5,6,7,8,9,10,11 --prefill-chunk 16 --aot \
    --record $K/results/export_int8mix_pf16.json
# Kev-4B: int8_bisect_torch.py --model kev-4b rule|dump --variant b32|b16|aff32|embed32|run --wait-dumps|plan|merge;
#   export_decoder.py int8mix --model kev-4b --quant-scheme asymmetric --fp16-kinds qkv,out,v --prefill-chunk 16 --aot \
#     --name kev_4b_decode_int8mix_aff32_k_qkv_out_v_pf16
# iOS bytes: xcrun coreai-build compile <bundle>.aimodel --output <dir> --platform iOS --architecture h19p \
#     --preferred-compute gpu [--expect-frequent-reshapes]   (compile only; never load an iOS asset on a Mac)
```

The rule is written by `rule` before any bisect result (`$K/bisect/rule.json`, `$K/bisect_4b/rule.json`). Neither size's
rule chose a set and neither int8 bundle ships. Rank layers by the bisect rows' mean, not the worst row, and pick the
bisect rows again when the int8 variant changes. Transcripts: `gate-kev-<size>-int8.json`.

## 6. The Python host

```bash
$K/venv-oracle/bin/python test_host.py --out $K/results/host_test_oracle_venv.json   # with the author's pydantic models
$PY test_host.py
$PY decide.py run --model kev-0.8b --request req.json --shared --out resp.json --trace trace.json
for m in kev-0.8b kev-4b; do
  $PY decide.py check --model $m --rows all --shared --out $K/results/e2e_$m.json
  $PY decide.py check --model $m --rows heldout --shared --out $K/results/e2e_${m}_heldout.json
done
$PY timing.py run                       # takes the machine-wide GPU lock
```

The host's head runs in float64 and rounds p to fp32 once; the author's sums are Python 3.12's compensated `sum()`.
Transcripts: `gate-kev-<size>-host.json`.

## 7. Swift

```bash
swift build -c release --package-path ../../apps/Kev --scratch-path $K/swift/.build
BIN=$K/swift/.build/release/kev
for m in 0_8b 4b; do
  B=$K/exports/bundles/kev_${m}_decode_fp16_pf16
  $BIN fixture --bundle $B --records $K/fixtures/records.json --shared --out fixture_$m.json
  $BIN fixture --bundle $B --records $K/fixtures/heldout.json --shared --heldout --out heldout_$m.json
done
$PY gate_swift.py render
$PY gate_swift.py score --model kev-0.8b --fixture fixture_0_8b.json --heldout heldout_0_8b.json
$PY gate_swift.py negative
$BIN fixture --bundle $B --asset aot --records $K/fixtures/records.json --ids <10 records> --out aot.json
$BIN fixture --bundle $B --asset jit --records $K/fixtures/records.json --ids <10 records> --out jit.json
$PY gate_swift.py jit --aot aot.json --jit jit.json        # Kev-0.8B; Kev-4B was scored by a lane copy of it
../../apps/Kev/_time_mac.sh && $PY gate_swift.py timing --run-dir $K/swift/timing/<run id>
```

`--asset aot` loads `<bundle>/../../bundles_aotc/<name>.h16c.aimodelc` with `SpecializationOptions.default`; `--asset
jit` specializes the bundle's `.aimodel` (GPU preferred, `expectFrequentReshapes`). The JIT check used the records
`tv4_000, tv4_001, tv4x_emotion_00, tv4x_qnli_00, tv4s_00, semif_a3f18f3a63d45345942b, own_t01, own_j03, own_m01,
own_L02`; delete the JIT's cache entry (`~/Library/Caches/coreai-cache/<build>/kev/<main.mlirb sha256>`) after the
check. `_time_mac.sh` takes the machine-wide GPU lock (a tag in the file and a `flock(2)` held for the whole window).
Transcripts: `gate-kev-<size>-swift.json`, `-timing-mac.json`.

## 8. The iPhone (Kev-0.8B)

[`apps/KevGate`](../../apps/KevGate/README.md) lists the steps: `_build.sh`, `_stage.sh`, `_run_mac.sh` (the harness
on the Mac first, and once with one oracle row reversed, which must fail), `_gate.sh <udid>` under the device hold,
then `$K/scripts/r8_score.py device` re-scores every run from the p bits the app wrote. Kev-4B's h19p asset was staged
apart (`_stage.sh --4b`) and loaded once with `KEV_TRY_4B=1`. Transcript: `gate-kev-0.8b-iphone.json`.

## 9. The published files

`models/kev-<size>/fixtures-kev-<size>.json` and the `gate-kev-<size>-*.json` transcripts were assembled from the lane
files above without re-running anything (round 9: `$K/scripts/r9_build_public.py`; round 17 rebuilt both sizes' for the
published graphs with `$K/scripts/r17_build_public.py`): each names its source files with their bytes and sha256 and lists
what it leaves out (`trimmed`). The transfer-v4 records are references (the
author's file, line and row hashes) — `make_fixtures.py` rebuilds their requests and `row_ids_sha256` checks the ids —
and nine records written for the port are withheld because an invented name in them was found in use on the web. The
Hugging Face staging trees (`$K/hf_staging/`) were written by `$K/scripts/r9_stage.py` and, for the published Kev-0.8B
graph, `$K/scripts/r17_stage.py` (APFS clones of the gated bundle, the final `metadata.json`, the card with its front
matter, `SHA256SUMS`). Every number in the cards and in `knowledge/kev-port.md` is listed with its source file and key
in `$K/results/r17_numbers.md` (`$K/scripts/r17_numbers.py --check <file>` lists the numbers a text has that the table
does not).

## 10. The GDN scan's form and S (round 11, Kev-0.8B, Mac GPU)

Every linear-attention layer runs its recurrence inside the call in one of three forms; the graph's inputs, outputs
and states stay the same, so the hosts read S from `metadata.json` and nothing else changes:

| form | `--gdn-scan` | what runs inside the call |
|---|---|---|
| U<S> | `unroll` (default, every earlier bundle) | `_gated_delta_step_unroll`: the S single steps unrolled in the graph, fp32 |
| C<S> | `chunk` | the overlay's `_gated_delta_chunk`: the S tokens in parallel, the triangular inverse as ceil(log2 S) doublings, fp32 |
| K<S> | `metal` | the overlay's fp32 Metal chunk kernel (`qwen3_5_gdn_metal`), one GPU dispatch per layer for the whole chunk, built here with chunk_max = S |

```bash
$PY export_decoder.py fp16 --prefill-chunk 64 --gdn-scan metal --aot --record $K/results/r11_export_fp16_metal_pf64.json
$PY readout_gate.py run $K/exports/bundles/kev_0_8b_decode_fp16_metal_pf64 --subset s80 \
    --compare-with $K/results/readout_fp16_pf16.json --tag r11_s80_K64 --transcript $K/results/r11_s80_K64.json
$PY parity_decoder_torch.py scan --gdn-scan chunk --p3-widths 16,32 --threads 1 --conv bmm   # fp32 torch, C only
```

The kernel's `torch_defn` returns zeros of the right shapes, so K has no values in torch: its numerics exist only on
the GPU, and the readout gate is its parity check. Every form exported in 26–101 s and compiled (h16c, efr) in 5–19 s.
The screen (s80, 130 rows) and the full gates ran in one `_GPU_LOCK` window, with U16 at both ends of the screen (its
hidden rows and p equal to round 2's on 130/130 rows both times); the times there are contended (other lanes' exports
and compiles ran), the decision latencies are in the timing window below.

| form | s80 | full gate: fixture 434 (non-near-tie, max\|Δp\|, mean) | held-out 130 max\|Δp\| | red arms v2 | ms per call (screen) |
|---|---|---|---|---|---:|
| U16 | PASS | round 2: 420/420, 0.0117, 0.00097 | 0.0057 | red | 18.26 (quiet end of the window) – 27.03 |
| U32 | PASS | 420/420, 0.0118, 0.00095 | 0.0058 | red | 34.94 |
| U64 | PASS | 420/420, 0.0100, 0.00095 | 0.0061 | red | 68.06 |
| C16 | PASS | 420/420, 0.0135, 0.00098 | 0.0057 | red | 10.90 |
| C32 | FAIL (non-near-tie argmax 112/127, max\|Δp\| 0.125, non-finite rows) | not run | — | — | 12.56 |
| K16 | PASS | 420/420, 0.0115, 0.00096 | 0.0062 | red | 10.62 |
| K32 | PASS | 420/420, 0.0134, 0.00096 | 0.0060 | red | 13.60 |
| K64 | PASS | 420/420, 0.0115, 0.00097 | 0.0063 | red | 18.87 |
| K128 | PASS | 420/420, 0.0124, 0.00096 | 0.0058 | red | 30.01 |

- The in-graph chunk form breaks at S = 32 already in fp32 torch (P3's 21 rows: hidden max |Δ| 6.4 against U16, max
  |Δp| 0.12, position cosine down to 0.845; C16 is 4.4e-4 and 3e-6), so its GPU FAIL is the algorithm, not fp16. The
  overlay's notes put the break at chunk ≥ 64; for this model it is between 16 and 32.
- The kernel holds the state in fp32 for the whole chunk and computes the qk l2-norm in fp32 (the unrolled form does the
  l2-norm in fp16), so K and U are not bit-equal (p within 0.0015–0.0021 of U16 on s80); both are inside the bar.
- Call cost against S, least squares over the screen window's ms per call: U ≈ 1.7 + 1.04·S ms, K ≈ 7.9 + 0.17·S ms
  (C: the two points give 9.2 + 0.10·S ms). An unrolled call grows with S almost entirely; a kernel call is mostly a
  fixed cost with a sixth of the per-token cost. What the fixed ~8 ms is made of was not isolated.

Decision latency (`apps/Kev/_time_mac.sh` with `KEV_FORMS`, the Swift Release CLI on the AOT assets, the forms A B C …
in rounds of one process each, 10 decisions per item after one warm-up; `gate_swift.py timing-r11`). The Mac was shared
with three other lanes, so a process counted only when no other GPU job ran during it and the 1-minute load average was
≤ 12 at its start and its end; the rule was fixed before round 3, rounds 1–3 lost processes to load 15–43, another
lane's GPU probe and GPU jobs the process list did not show (U16's four processes: 177.3, 106.5, 493.0 and 115.2 ms for
94 tokens), and the medians below are over the clean processes only (`swift/timing/<run>/clean.json` lists every process):

| form | clean processes | 94 tokens, 1 question (p10–p90) | 380 | 1,518 | 5 questions shared | ms per call |
|---|---:|---|---:|---:|---:|---:|
| U16 | 1 | 115.2 (108.6–127.2) | 452.4 | 2310.9 | 495.6 | 22.53 |
| U32 | 2 | 139.0 (131.3–142.8) | 555.0 | 2071.0 | 446.1 | 44.73 |
| K16 | 2 | 63.6 (61.4–64.3) | 255.0 | 1028.5 | 211.4 | 10.63 |
| K32 | 2 | 42.7 (40.6–44.5) | 171.2 | 711.1 | 154.5 | 14.30 |
| K64 | 2 | 41.5 (37.0–46.4) | 124.4 | 537.7 | 145.0 | 20.76 |
| K128 | 2 | 30.0 (29.6–30.1) | 89.7 | 355.9 | 189.1 | 29.53 |

Ranked by the 94-token median (ties within 3 % broken by 5 questions shared, then the smaller S; a form 5 % or more
slower than U16 on any of the ten items is out): K128, K64, K32, K16, U16; U32 is out (×1.21 on 94 tokens). K128 answers
a one-question row of up to 128 tokens in one call; K64 is the faster form for the multi-question requests (5 and 8
questions shared). In the same turn, K128's `.aimodel` specialized by Swift (GPU preferred + efr) took 3.3 s the first
time and decides 94 tokens in 29.7 ms, its hidden rows and p bit-equal to the AOT asset on 25 rows
(`gate_swift.py jit`); the Python reference host on the AOT asset takes 33.4 ms; the Swift host's fixture and held-out
passes give the readout gate's bar to six decimals (fixture max|Δp| 0.012426, mean 0.000962; the float64 host head and
the gate's fp32 torch head part at the seventh), with hidden rows equal to the gate's and p bit-equal to the Python
host's on 434 + 130 rows (`gate_swift.py score --bundle …`).

## 11. The dynamic-S graph and the host's call plan (rounds 14–16, Kev-0.8B, Mac; measured, not shipped)

Round 14 exported the Metal-kernel form with a dynamic query length (`--gdn-scan metal --dynamic-query --prefill-chunk
<cap>`): `input_ids [1, -1]` and `hidden [1, -1, 1024]`, any call length 2..cap, the kernel built with chunk_max = cap. A
call costs what the static form of the same length costs (D128 ≈ 7.99 + 0.184·s ms against K ≈ 8.52 + 0.169·S ms in one
window), so a row pays for its real tokens instead of whole chunks. What it can add is a first-call cost per length: in
the process that had just made the runtime's cache entry, a new length's first call took 30–2,458 ms (round 14, D128),
and later processes ran every length at its steady cost from the first call; where that is kept was not found. Round
15, on the shipped D512, saw only a process's first call slow (150–170 ms; 1.3 s in the process that made the AOT
asset's entry) and every other length's first call at its steady cost. The host therefore fixes the lengths, from the
bundle's `metadata.json`:

| key | what the host does |
|---|---|
| `language.query_len_range` [2, cap] | the graph's range (round 14; a static-S bundle has `prefill_chunk` instead) |
| `language.query_len_call_max` L | the longest call it makes (≤ cap; absent = cap) |
| `language.query_len_multiple` q | every call length is a multiple of q (absent = 1) |

`host.plan(n, L, q)` cuts a row into pieces of L ids, the remainder last, and pads the last piece with `<|endoftext|>`
up to the next multiple of q; the padded rows are dropped. A process then sees at most L / q lengths, and `warm_up` (the
Python and Swift hosts, `--warm` on the CLIs) runs each once at load. The shared prefix runs the state's first
⌊Ls / q⌋·q tokens once; its calls are cut at other places than a direct run's, so on this graph shared and direct differ
in the last bits (both are gated against the oracle). A static-S bundle is the case L = q = S and reads exactly as before
(test_host.py checks the plan against the gate's own `host_pieces` for every row length 1..4,095).

The exporter writes a bundle around a gated `.aimodel` without exporting again (`--metadata-only`; the same path
rewrites the metadata of U16, K128 and round 14's D128 / D256 / D512 byte for byte, the date aside). For a dynamic-S
graph:

```bash
$PY export_decoder.py fp16 --gdn-scan metal --dynamic-query --prefill-chunk <cap> --query-call-max <L> --query-multiple <q> \
    --metadata-only --from-aimodel $K/exports/bundles/kev_0_8b_decode_fp16_metal_dyn<cap>/kev_0_8b_decode_fp16_metal_dyn<cap>.aimodel \
    --bundle-dir $K/exports/ship/kev_0_8b_decode_fp16_metal_dyn<cap> --record $K/results/r15_ship_bundle_dyn<cap>.json
$PY readout_gate.py run $K/exports/ship/kev_0_8b_decode_fp16_metal_dyn<cap> --transcript ...        # L and q from the metadata
$PY readout_gate.py run $K/exports/ship/kev_0_8b_decode_fp16_metal_dyn<cap> --shared --transcript ...
$PY decide.py check --model kev-0.8b --bundle <ship> --rows all --shared --prepared --gate-transcript <direct> \
    --shared-gate-transcript <shared> --out ...
$BIN fixture --bundle <ship> --records $K/fixtures/records.json --shared --prepared --warm --out ...
$PY gate_swift.py --swift-dir <dir> score --model kev-0.8b --bundle <ship> --fixture ... --gate-fixture <direct> \
    --e2e-fixture <decide.py check> --shared-gate-fixture <shared> --transcript ...
```

The AOT asset of a dynamic-S graph has a cache trap: its `main-h16c.mlirb` holds the function's type and the source
file's path and hash, not the cap, so two caps exported from the same version of `qwen3_5_kev_decoder.py` get the same
coreai-cache entry name, and loading the second reuses the first's delegates. Check which asset an entry holds (its
`manifest.plist` hash) before a gate; the `.aimodel` JIT's entries are named by the graph and do not collide.

Round 15 gated and timed `kev_0_8b_decode_fp16_metal_dyn512` (round 14's D512 `.aimodel`; L = 512, q = 16, chosen from
round 14's timing) on the Mac as the form to ship. The Python host against the oracle: fixture 434 argmax 420/420
outside near-ties (near-ties 12/14), max |Δp| 0.0112, mean 0.00096; held-out 130 125/125 (5/5), 0.0062, 0.00087; shared
420/420, 0.0128, 0.00095, at most 0.0022 from the direct p; the red arms red; four rows near the graph limit 0.0017; the
p equal to round 14's gate of the same bytes on all 564 rows. The Swift host equals the Python host on every row (ids,
hidden sha256, p bits, answers' bytes; direct, shared, and the prepared state, which equals the shared prefix bit for
bit), its `.aimodel` specialized by Swift equals the AOT asset on 25 rows, the negative control fails, and the U16 bundle
still gives round 7's numbers. One `_GPU_LOCK` window, the Swift Release CLI's `kev time --warm --prepared` (the lane
driver `$K/logs/r15/r15_timing_window.py`; `_time_mac.sh` with `KEV_FORMS` and `KEV_CLEAN=1` runs the same processes
without the prepared items), two clean processes per form, median of 20 decisions (`$K/results/r15_timing_0_8b.json`):

| form | 94 tokens, 1 question (p10–p90) | 380 | 1,518 | 1,802 | 5 questions shared | 5 direct | 8 shared | 4 on 1,477 tokens, shared |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D512 L512 q16, AOT | 25.4 (25.0–25.7) | 76.9 | 296.8 | 359.1 | 111.3 | 188.8 | 162.0 | 365.3 |
| D512 L512 q16, JIT | 25.2 (24.9–25.3) | 76.9 | 297.1 | 359.1 | 111.7 | 189.0 | 162.2 | 365.7 |
| D512 L256 q16, AOT | 25.1 (24.9–25.3) | 83.4 | 317.1 | 384.3 | 111.3 | 188.2 | 162.0 | 385.0 |
| D512 L128 q16, AOT | 25.0 (24.8–25.2) | 90.3 | 360.9 | 434.5 | 111.4 | 225.6 | 162.6 | 429.3 |
| K128, AOT | 29.9 (29.6–30.1) | 89.2 | 357.0 | 447.9 | 186.4 | 296.6 | 278.8 | 485.4 |
| U16, AOT | 100.1 (98.4–103.6) | 404.4 | 1600.1 | 1897.6 | 327.7 | 845.7 | 461.0 | 1749.0 |

L256 and L128 are the same bundle with `--call-max` / `--multiple`. By round 12's rank rule v2 (a form no more than 3 %
slower on all ten items and more than 3 % faster on one dominates; `$K/logs/r12_drivers/r12_rank_v2.py`), L512
dominates U16, K128, L256 and L128. A prepared state (`prepare` once, then each question's branch) costs 34.3 ms and then
14.2 ms a question on a 137-token state, 295 ms and 18.7 ms on 1,477 tokens (U16: 137.8 / 34.1 ms and 1,543 / 51.5 ms).
The Python host in the same window: 29.5 ms for 94 tokens, 136.7 ms for 5 questions shared. Load with the runtime's cache
warm: 0.2 s; the `.aimodel` specialized by Swift the first time: 3.2–3.3 s and a 2.50 GB cache entry.

`export_decoder.py --kernel-io static` (round 16) builds the same graph with the GDN kernel on static edges (its inputs
zero-padded in-graph to the cap, the call's length as a 1-element int32 input); it exports, compiles and runs. Each call
is 0.9–1.4 ms slower than the dynamic-edge graph's and each new call length adds as much memory or more (28–29 MB at 96
and 128 tokens against 19–20 MB, 75–77 MB at 512 for both), so it is no answer to the dynamic graph's footprint
(`$K/ROUND16.md`, `$K/results/r16_static_kernel_0_8b.json`).

The dynamic-S graph does not ship. A process that keeps changing its call length keeps growing: cycling four call
lengths (128, 256, 384, 512) three times added 467.1, 166.9 and 168.5 MB per lap in Swift and 460.7, 169.5 and 169.6 MB
in the Python runtime; alternating between two lengths did not grow after the opening calls; on the iPhone the gate run
reached 3,497 MB and stopped writing. Loading the function again gave nothing back; creating the `AIModel` again gave
it all back. Which layer keeps the memory was not isolated (`$K/results/r15_leak_0_8b.json` `leak2` and `leak3`,
`$K/results/r13_c_memory.json`). The hosts still read a dynamic-S bundle as above.

## 12. The published bundles (round 17)

The published Kev-0.8B bundle is K128 (§10): round 11's `.aimodel` (`main.mlirb` sha256 `19d5a480…`), a static
128-token call, with its metadata written again by the same path (round 15):

```bash
$PY export_decoder.py fp16 --gdn-scan metal --prefill-chunk 128 --metadata-only \
    --from-aimodel $K/exports/bundles/kev_0_8b_decode_fp16_metal_pf128/kev_0_8b_decode_fp16_metal_pf128.aimodel \
    --bundle-dir $K/exports/ship/kev_0_8b_decode_fp16_metal_pf128 --record $K/results/r15_ship_bundle_pf128.json
```

Its gates are round 11's on the same bytes (§10) and the iPhone 18 Pro run of round 15 through
[`apps/KevGate`](../../apps/KevGate/README.md) (`$K/results/r15_device_gate_K128.json`). The hosts read it as a
static-S bundle (L = q = 128, the last call padded). Shared and direct runs cut the state at the same places: the Swift
host's shared hidden rows equal its direct ones on 434/434 and 130/130 rows (`$K/results/r11_swift_gate_K128.json`). The
staged `metadata.json` changes only the keys the hosts do not read (`metadata_of_record` in
the file lists them); the Swift CLI's `kev rows` on it gives the gated ids on 434 + 130 rows
(`$K/results/r17_staging_rows_0_8b.json`).

The published Kev-4B bundle is the same form at 4B: round 12's K128 export (`--model kev-4b --gdn-scan metal
--prefill-chunk 128 --aot`, `main.mlirb` sha256 `da529dd4…`), with its metadata written again by the same path
(`$K/results/r12_ship_bundle_pf128_4b.json`). Its gates are round 12's on the same bytes: the readout gate
(`$K/results/r12_fixture_K128_4b.json`, `r12_heldout_K128_4b.json`, `r12_red_K128_4b.json`), `decide.py check`
(`r12_e2e_K128_4b.json`, `r12_e2e_heldout_K128_4b.json`) and the Swift host (`r12_swift_gate_K128_4b.json`,
`r12_swift_jit_K128_4b.json`). Round 12 timed U16, K64, K128 and the cap-512 dynamic graph in one lock window
(`$K/swift/timing/r12_swift_timing_20261004_132821/`); K128 decides one question in the least time of the static
forms, and K64 answers several questions on one state in less. `kev rows` on the staged metadata gives the gated ids
on 434 + 130 rows (`$K/results/r17_staging_rows_4b.json`). Kev-4B ships for the Mac only.

## 13. Kev-4B: the GDN scan's form and S (round 12, Mac GPU)

The same choice on Kev-4B (§10's table of forms; C was not built for 4B: it breaks at S = 32 on 0.8B). Kev-4B's GDN
layers have 16 key heads and 32 value heads, so the Metal kernel runs its grouped-value path (value head h reads key
head h / 2, `__R__` = 2 in the kernel source) for the first time in this lane; the readout gate is its only parity check.

```bash
$PY export_decoder.py fp16 --model kev-4b --prefill-chunk 64 --gdn-scan metal --aot --record $K/results/r12_export_fp16_metal_pf64_4b.json
$PY readout_gate.py run $K/exports/bundles/kev_4b_decode_fp16_metal_pf64 --model kev-4b --subset s80 \
    --compare-with $K/results/readout_fp16_pf16_4b.json --tag r12_s80_K64 --transcript $K/results/r12_s80_K64_4b.json
```

Exports took 34–37 s for K and 94–112 s for U (the unrolled form puts S steps per layer in the graph), the AOT compiles
(h16c, efr) 31–42 s; every compiled asset is 15.55–15.57 GB, like U16's. The screen ran the six forms in one
`_GPU_LOCK` window with U16 at both ends (its hidden rows and p equal to round 4's on 130/130 rows both times); the
LiteRT lane's Mac GPU parity jobs, which do not read the lock, overlapped every step, so the screen's times are
contended reference values.

| form | s80 (non-near-tie, max\|Δp\|, mean) | ms per call (screen) | ms per padded token | full gate: fixture 434 / held-out 130 / red arms v2 |
|---|---|---:|---:|---|
| U16 | 124/124, 0.0087, 0.00076 | 53.51 / 48.19 (first / last of the window) | 3.34 / 3.01 | round 4: 422/422, 0.0154, 0.00078 / 0.0109 / red |
| U32 | 124/124, 0.0078, 0.00072 | 104.85 | 3.28 | not run (projected fixture time 1.09 × U16's) |
| U64 | 124/124, 0.0070, 0.00071 | 174.06 | 2.72 | not run (1.02 × U16's) |
| K32 | 124/124, 0.0074, 0.00077 | 48.42 | 1.51 | not run (0.50 × U16's; the round ran the two fastest) |
| K64 | 124/124, 0.0068, 0.00071 | 62.90 | 0.98 | 422/422, 0.0170, 0.00077 / 0.0100 / red |
| K128 | 124/124, 0.0071, 0.00074 | 114.04 | 0.89 | 422/422, 0.0153, 0.00077 / 0.0096 / red |

- Call cost against S (least squares over the screen's ms per call): U ≈ 16.2 + 2.51·S ms, K ≈ 22.8 + 0.70·S ms. On 4B
  the kernel's call has a larger fixed part than on 0.8B (§10: 7.9 ms) and a per-token part 0.28 times the unrolled
  one's; the unrolled call is still nearly proportional to S, so U32 and U64 only trade calls for padding.
- No form is bit-equal to U16: on s80 p is within 0.0015–0.0040 of round 4's U16 (K32 0.0015, K128 0.0018, U64 0.0018, U32 0.0021, K64 0.0040).

### The dynamic-S kernel graph on Kev-4B (D512)

§11's form (`--gdn-scan metal --dynamic-query`: one function whose query length is dynamic, 2..cap, the kernel built
with chunk_max = cap) at cap 512, with `--trace-query-len 20` (the default 24 equals Kev-4B's 24 linear layers, the
first dim of the conv / rec states). Its bundle for the hosts is the gated `.aimodel` cloned into
`$K/exports/ship/kev_4b_decode_fp16_metal_dyn512` with round 15's metadata (`--metadata-only`, L 512, q 16):

```bash
$PY export_decoder.py fp16 --model kev-4b --gdn-scan metal --dynamic-query --prefill-chunk 512 --trace-query-len 20 --aot \
    --record $K/results/r12_export_fp16_metal_dyn512_4b.json
$PY export_decoder.py fp16 --model kev-4b --gdn-scan metal --dynamic-query --prefill-chunk 512 --query-multiple 16 \
    --query-call-max 512 --metadata-only --from-aimodel $K/exports/bundles/kev_4b_decode_fp16_metal_dyn512/kev_4b_decode_fp16_metal_dyn512.aimodel \
    --bundle-dir $K/exports/ship/kev_4b_decode_fp16_metal_dyn512 --record $K/results/r12_ship_bundle_dyn512_4b.json
```

- Export 44 s, AOT 30 s, compiled asset 15.55 GB. Its AOT `main-h16c.mlirb` (the function's signature and debug
  locations, 2,886 B) is byte-equal to the cap-128 graph's from the same source = one coreai-cache entry name for both;
  the cap-128 graph ran once, its entry swapped in and out around that run.
- Call cost, one process, the same calls: D512 = 22.2 + 0.620·s ms (s = 16..512), K64 / K128 = 18.6 + 0.623·S; at the same
  length D512 / K = 1.014 (64) and 1.009 (128); the cap-128 graph 1.014 / 1.009 too. A call on a dynamic graph costs what
  the static graph of that length costs.
- A process's first call pays a one-time cost (13.2 s on a cold runtime cache, 1.05 s in the next process; 8.9 s when the
  first call is s = 512); after it, each new length's first call is close to its warm time. Hidden rows repeat bit for bit
  across resets and across two processes (12 / 12 lengths).
- Gates (host policy = call cap L and multiple q, from the metadata or `--call-max` / `--host-q`):

| policy | rows | non-near-tie | max\|Δp\| / mean |
|---|---|---|---|
| L512 q16 fixture 434 direct / shared | 434 | 422/422 | 0.0132 / 0.00077, shared 0.0168 / 0.00078 |
| L512 q16 held-out 130 direct / shared | 130 | 126/126 | 0.0092 / 0.00070, shared 0.0124 / 0.00070 |
| L512 q16 red arms v2 | 10 | — | red (both arms) |
| L256 q16 s80 / own shared | 130 / 70 | 124/124, 67/67 | 0.0070 / 0.00073, 0.0062 / 0.00062 |
| L128 q16 s80 / own shared | 130 / 70 | 124/124, 67/67 | 0.0070 / 0.00074, 0.0059 / 0.00062 |
| L256 q8 s80 / own shared | 130 / 70 | 124/124, 67/67 | 0.0070 / 0.00073, 0.0062 / 0.00063 |

`decide.py check` on the L512 q16 bundle (raw request → ids → graph → head): hidden = the gate's on 434 + 130 rows, the
shared run's hidden = the shared gate's (`$K/results/r12_e2e_D512L512q16_4b.json`, `r12_e2e_heldout_D512L512q16_4b.json`).

### The timing window and the choice

Round 15's Swift CLI (`kev time`; the lane's copy `$K/swift_r15/kev_pregate`, sha256 `5c966633…`) timed four forms in one
`_GPU_LOCK` window (10-04 14:05–14:30): U16, K64, K128 and D512 at L 512 / q 16, two rounds of the four, each process
clean (no other GPU job, load ≤ 12 at its start and its end), each item the median of 6 decisions × 2 processes
(`$K/results/r12_timing_4b.json`, the processes in `$K/swift/timing/r12_swift_timing_20261004_132821/`):

| form | 94 tokens, 1 decision (ms) | ms per call there (calls × padded tokens) | 5 questions shared | 8 questions shared | 1,518 tokens | footprint max |
|---|---:|---:|---:|---:|---:|---:|
| U16 | 269.5 | 44.5 (6 × 16) | 894.7 | 1,262.4 | 4,341.9 | 0.93–0.95 GiB |
| K64 | 118.0 | 58.1 (2 × 64) | 435.9 | 617.9 | 1,409.5 | 0.86–0.87 GiB |
| K128 | 100.2 | 98.4 (1 × 128) | 615.6 | 920.0 | 1,189.0 | 0.89 GiB |
| D512 L512 q16 | 83.3 | 81.6 (1 × 96) | 353.5 | 515.7 | 1,009.6 | 5.77–5.78 GiB |

- Rank rule v2 (fixed before any D timing: drop a form another is not slower than by more than 3 % on all 10 items and
  faster by more than 3 % on one; then the 94-token decision): D512 dominates the other three. It is not shipped: its
  footprint grows with every new call length (round 15) and stood at 5.8 GiB here against K128's 0.89 GiB.
- Over the shippable forms U16 dominated by both K forms; K64 and K128 do not dominate each other (K64 is faster on the
  requests of several questions on one state, K128 on single questions and long rows) and the 94-token decision picks
  K128 (100.2 against 118.0 ms) = §12's Kev-4B bundle.
- K128 through the Swift host: hidden rows equal to the Python gate's on 434 + 130 rows (sha256), shared = direct, reset
  bit-equal; the `.aimodel` specialized by the Swift runtime (JIT) equals the AOT asset on 25 rows (hidden sha256 and p;
  `$K/results/r12_swift_gate_K128_4b.json`, `r12_swift_jit_K128_4b.json`). D512's JIT decides the 94 tokens in 83.1 ms.
- The runtime's cache directory for the CLI is `<os build>/kev-pregate` (the executable's name with `-`); the CLI's records
  print `kev_pregate`, a path that does not exist, so their cache fields read 0. The real listing
  (`$K/results/r12_cache_kev_pregate_4b.json`): one 14.5 GiB entry per AOT asset, named by its `main.hash`, and one per
  JIT-specialized `.aimodel`, named by the sha256 of its `main.mlirb`.

## Rounds

The port ran in rounds; their notes are in the lane (`$K/ROUND<N>.md`). Round 1 = steps 1–2, round 2 = steps 3–4 (S
chosen), round 3 = step 5 (Kev-0.8B), round 4 = Kev-4B through steps 1–5, round 5 = step 6, round 6 = step 5 (Kev-4B's
int8 variants and the compiled assets), round 7 = step 7, round 8 = step 8, round 9 = step 9 and Kev-4B's JIT check, round 11 = §10 (Kev-0.8B, Mac),
rounds 14–16 = §11 (Kev-0.8B, Mac), round 17 = §12 and the published files.
