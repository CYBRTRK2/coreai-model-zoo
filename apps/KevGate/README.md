# KevGate: the device gate for Kev-0.8B

A headless app that runs the Kev-0.8B port ([`conversion/kev`](../../conversion/kev)) through the Swift host
[`apps/Kev`](../Kev) on sideloaded assets, over the fixture the port was gated on (384 records, 434 questions) and the
held-out set (130), on the iPhone or on the Mac. It links the library by path, so the decision path is exactly the one an
app gets from `KevDecider`: the author's request checks and text, swift-transformers' tokenizer, one row per question,
the decoder `main` in its call order from zero states (the call length from the bundle's `metadata.json`), the float64
pointer head, the SystemOne answers. The gate
starts at launch with no UI input. It writes `result.json` (rewritten when a stage starts, after every stage and every 5
records), `result.log` (every line with the thermal state and the battery) and `memory.tsv` (every 100 ms memory reading,
written as taken, so a killed process leaves its series): in `Documents/kev_gate/` on the iPhone, in `KEV_OUT` on the Mac.
`p_ref_jit.json` / `p_ref_aot.json` beside them keep every e2e row's p bits and hidden sha256 across launches.

What it loads (staged by `_stage.sh`):

| path in the assets directory | what |
|---|---|
| `decoder/` | the bundle `kev_0_8b_decode_fp16_pf16`: `metadata.json`, `kev_0_8b_decode_fp16_pf16.aimodel` (`main`, S = 16, fp16, final-norm hidden output), `tokenizer/`, `head/` |
| `aot/kev_0_8b_decode_fp16_pf16.h19p.aimodelc` | the iPhone 18 Pro's AOT of the same graph (`coreai-build compile --platform iOS --architecture h19p --preferred-compute gpu --expect-frequent-reshapes`) |
| `decoder_4b/` | Kev-4B's `metadata.json`, `tokenizer/`, `head/` (no `.aimodel`): `load_4b` opens an AOT asset pushed apart |
| `fixtures/` | `requests.json` (514 records), `oracle_slim.json` (the author's fp32 oracle per question: row ids, readout indices, keys, probabilities, argmax, top-2 margin), `mac_ref.json` (the Mac's Swift run of the same bundle, round 7, AOT h16c: hidden sha256 and p bits per question), `bench.json`, and the 4B pair for one record |

`_stage.sh` stages the earlier upload's graph (U16) as `decoder/`. Rounds 13 and 15 staged the Metal-kernel forms
beside it as `decoder_k16` … `decoder_k128`; `KEV_DECODER=decoder_k128` runs the published graph
(`kev_0_8b_decode_fp16_metal_pf128.aimodel`, `main.mlirb` sha256 `19d5a480…`, S = 128).

A dynamic-S bundle (round 14's graph, `language.query_len_range` in its metadata) loads the same way with
`KEV_DECODER=<its directory>`: the host cuts each run into calls of at most `L` ids, the last one padded up to a multiple
of `q` (`language.query_len_call_max` / `query_len_multiple`, or the launch's `KEV_CALL_MAX` / `KEV_MULTIPLE`).
On the iPhone 18 Pro (iOS 27.0, round 13) such a graph's footprint grows each time a call's length differs from the
previous call's (13.8 MB on average over 20 fixture records at L 512 / q 16; 36 MB per call when every call is a new
length), part of it comes back about a second after the work stops, and a long e2e run reached the app's memory limit
and stopped; with one call length (`KEV_CALL_MAX` = `KEV_MULTIPLE`, e.g. 128) it stays flat. `KEV_RECORD_PAUSE=<s>`
rests that many seconds after every e2e record, so `memory.tsv` shows what the footprint does while idle.

## Stages

| stage | what it records |
|---|---|
| `assets` | every file in `MD5SUMS` present, md5 of every file up to 16 MB (the model files wait for `md5`), the free space (stops below `KEV_MIN_FREE_GB`, default 8) |
| `load_jit` | `KevDecider` on the bundle's `.aimodel`, GPU preferred with `expectFrequentReshapes` (the specialization happens on the device): wall, the library's split (tokenizer, `AIModel`, `main`), the 100 ms memory series (peak footprint, least `os_proc_available_memory`), the Core AI cache before and after, the descriptor (checked against the contract by the library) |
| `load_aot` | the same with `KEV_AOT` (default the h19p `.aimodelc`) and `SpecializationOptions.default` |
| `warm` | one decision (`tv4_000`): the first call after the load |
| `warm_up` | `KevDecider.warmUp`: every call length of the plan once from zero states, each length's ms (a dynamic-S graph specializes each length on its first call in a process; on the iPhone the result lands in the app's Metal cache, `Library/Caches/<bundle id>/com.apple.metal`, and survives a relaunch), the app's caches before and after |
| `e2e_fixture`, `e2e_heldout` | every record from its raw request, direct: per question the ids / readout indices / keys against the oracle, p against the oracle and against the Mac's Swift run (p bits, \|dp\|, hidden sha256), calls with every call's length and ms, thermal, battery, footprint; the 20 own records (multi-question) also with the shared prefix and with the prepared state (`prepare(state:)` then `trace(prepared:)`, p and hidden bit-equal to shared). Shared vs direct: a static-S bundle runs the same calls (bit-equal required); a dynamic-S graph cuts them elsewhere, so its shared p must pass the bar against the oracle and the \|dp\| to direct is recorded |
| `reset` | the first e2e record again: hidden rows and p bit-equal |
| `bench` | per item (`bench.json`): rest `KEV_BENCH_REST` s (60), wait up to `KEV_WAIT_NOMINAL` s (300) for the thermal state nominal, 1 warm-up per mode, then `reps` decisions with the modes alternating; every decision with its start offset, latency, calls and their lengths, thermal, battery, footprint. The items in `KEV_BENCH_PREPARED` (default `own_m01:first5,own_L02:all4`) also run the mode `prepared`: the state prepared once (`prepare_ms`), then the questions on it (`latency_ms` = their graph calls + head) |
| `e2e_aot` | on the AOT decider: the fixture's first 60 and the held-out set's first 30 records, scored as e2e and against `p_ref_jit.json` |
| `bench_aot` | `bench.json`'s `bench_aot` items on the decider in use |
| `load_4b` | only with `KEV_TRY_4B=1`: Kev-4B's AOT asset (`KEV_4B_ASSET`) with `SpecializationOptions.default`, one decision, dropped |
| `delete` | `KEV_DELETE`: paths under the assets directory, or `cache:<hex>` (this app's Core AI cache entries of that hash) |
| `md5` | md5 of the files `assets` left, or of every file of `KEV_MD5SUMS` |

A stage writes its start into `result.json` before it runs. A launch that finds `result.json` still `running` records
the stage the previous launch died in (a crash or a jetsam kill) and skips that stage if it is planned again
(`KEV_RETRY_DIED=1` overrides): a configuration that died is not retried by accident. The bar (the port's, unchanged:
argmax on every question whose oracle top-2 margin is above 0.02, max |dp| <= 0.02, mean of the questions' mean |dp| <=
0.002, plus the ids and finite hidden rows) is computed in the app and again on the Mac from the p bits in
`result.json` (`_kev/scripts/r8_score.py`, `conversion/kev/decide.py`'s `bar_summary`).

## Signing and the memory limit

The bundle id `com.daisukemajima.kevgate` is new, and the only profile of team MFN25KNUGJ that lists the iPhone 18 Pro and
fits a new id is the wildcard one, which cannot carry `com.apple.developer.kernel.increased-memory-limit`
([DeciderVisionGate](../DeciderVisionGate/README.md#signing-and-the-memory-limit)). The app has no entitlements and runs at
the default memory limit; it records `os_proc_available_memory` at launch and through every load.

## Steps

1. `./_build.sh` (generic iOS, Release) and `./_build.sh --mac` (macOS arm64, Release). The `.app` paths go to
   `_work/app_path.txt` and `_work/app_path_mac.txt`. The script puts `../Kev/Package.resolved` back if package resolution
   rewrote it, and compares the md5 of every other file of `../Kev` before and after.
2. `./_stage.sh` gathers the assets into `_work/device_stage/KevAssets/` (3.8 GB, APFS clones) and writes `MD5SUMS`;
   `./_stage.sh --4b [efr|noefr]` stages Kev-4B's iPhone AOT apart with `MD5SUMS_4B`. `KEV_LANE` (default
   `~/code/coreai/_kev`) is the port's working directory.
3. The Mac: `./_run_mac.sh` runs the macOS build on the stage directory with the Mac's h16c asset (`KEV_AOT`), 20 records,
   only while the machine-wide GPU lock (`~/code/coreai/_GPU_LOCK`) is free — it never takes it; `./_run_mac.sh --red`
   runs the same with one oracle row's probabilities reversed, and the bar must fail. Results land in `_work/mac_runs/`.
4. The phone: `KEV_SESSION=<name> KEV_ALLOWED_DEVICES=<udid>,<coredevice id> ./_gate.sh <udid> [JSON members]` takes the
   device hold (`~/code/coreai/ondevice/.device_hold`; a JSON hold is never removed while its keeper pid lives), installs
   the app, pushes `KevAssets/` to `Library/Application Support/`, pulls it back and re-pushes any file whose md5 differs,
   launches the app with no console, polls `result.json` until it is done, and releases the hold on any exit.
   `KEV_ALLOWED_DEVICES` is required: the scripts refuse any other phone. Knobs go in as JSON members, e.g.
   `./_gate.sh <udid> '"KEV_STAGES":"load_aot,e2e_aot,bench_aot"'` (all of them are listed at the top of
   `Sources/GateRunner.swift`). `KEV_SKIP_INSTALL=1` runs again on what is installed; `KEV_FRESH=1` uninstalls first (the
   container and its Core AI cache go with it). The poll cap is `KEV_CAP` polls of 10 s (default 360).
5. The 4B asset: `KEV_STAGE_DIR=_work/device_stage_4b/KevAssets KEV_PUSH_ONLY=aot/<name>.h19p.aimodelc,MD5SUMS_4B
   KEV_SKIP_APP=1 ./_install.sh <udid>` under the hold (sizes checked from the phone's listing), then the stages
   `md5,load_4b` with `KEV_MD5SUMS=MD5SUMS_4B KEV_TRY_4B=1`, then `delete` with `KEV_DELETE=aot/<name>.h19p.aimodelc`.
6. Exit codes: 0 = the gate passed, 3 = a stage failed, 1 = no result, 2 = the device is busy, held or refused. A run that
   does not end `done` leaves the phone's crash-log listing, and copies of today's KevGate / jetsam reports, in the run
   directory. Reprint a result: `./_run.sh --summary <result.json>`.

`_work/` is git-ignored. The scripts never pass `--remove-existing-content` (it wipes the whole app container) or
`--console`, and the app refuses to open an iPhone AOT bundle (`.h18p.`, `.h19p.`) on a Mac.

## Runs (2026-10-04, iPhone 18 Pro, iOS 27.0 24A437)

At the default memory limit (`os_proc_available_memory` 3,529 MB at launch) every Kev-0.8B stage passed:
`20261004-003915` (assets, cold JIT load 6.65 s, the fixture and the held-out set, reset), `20261004-004248` (warm JIT
load 0.61 s, bench), `20261004-005157` (cold AOT load 4.64 s, the AOT subset bit-equal to the JIT run, bench_aot). Kev-4B's
h19p AOT crashed the app inside `AIModel(contentsOf:)` (`20261004-010831`, SIGSEGV in the on-device compile for
delegates); `20261004-011119` recorded that and deleted the asset. The numbers and the tables are in
[`conversion/kev/README.md`](../../conversion/kev/README.md) (step 8, the device gate).

Round 13 ran the Metal-kernel forms (K16 … K128) and the dynamic-S graph on the same phone. Round 15's
`20261004-140048` ran the published graph (`KEV_DECODER=decoder_k128`, stages `load_jit,warm,e2e_fixture,e2e_heldout,reset`)
and passed; its numbers are in the [card](../../models/kev-0.8b/README.md) (iPhone 18 Pro).
