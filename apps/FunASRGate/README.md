# FunASRGate: the device gate for Fun-ASR-Nano-2512

A headless app that runs the Fun-ASR-Nano-2512 port ([`conversion/funasr_nano`](../../conversion/funasr_nano)) through
CoreAIKit's FunASR host on sideloaded bundles, over the 155 fixture clips the port was gated on, on the iPhone or on the
Mac. It links CoreAIKit by path, so the transcription path is exactly the one an app gets from `KitFunASRModel`: the
kaldi fbank + LFR front end (Accelerate), the SAN-M encoder + adaptor graph, and the Qwen3-0.6B decoder on the
pipelined engine. The gate starts at launch with no UI input. It writes `result.json` (rewritten after every stage and
every 10 clips) and `result.log`: in `Documents/funasr_gate/` on the iPhone, in `FUNASR_OUT` on the Mac.

What it loads (staged by `_stage.sh`, the shipped pair):

| path in the assets directory | what |
|---|---|
| `decoder/` | the decoder bundle directory: `metadata.json`, `funasr_nano_2512_decode_int8lin_n63_s1.aimodel` (int8 block-32 linears, fp16 tied head and embedding, static [1,1] query with a dynamic KV cache, the residual-stream scale folded in), `tokenizer/` |
| `encoder.aimodel/` | the audio encoder graph: fp16 weights, fp32 compute; `feats [1,500,560]` + `mask [1,500]` float32 in, `audio_embeds [63,1024]` float32 out |
| `fixtures/` | the 155 clips (5 examples of the model repo, 50 each of FLEURS en_us, cmn_hans_cn, ja_jp, ≤ 30 s) and their references: `meta.json`, `expected.json` (the fp32 oracle's ids and text, the Python engine run of the same bundles, the oracle's top-2 gap per step), `mac_swift_ref.json` (the kit's Mac run: `FunASRSmokeTests` test 3), `manifest.json` + `feats/` (the NumPy front end's features) |

Both graphs are JIT `.aimodel` bundles: the phone specializes them at the first load and caches the result.

## Stages

| stage | what it records |
|---|---|
| `assets` | every file in `MD5SUMS` present, md5 of every file up to 16 MB (the model files wait for `md5`, so the cold load does not start on a warm file cache) |
| `load1` | the encoder graph alone (its cold specialization), then `KitFunASRModel(decoderBundleAt:encoderModelAt:)` (the decoder cold, the encoder now cached), then the encoder alone again (warm): the model load split into decoder and encoder. Each step runs under a 100 ms memory sampler (peak footprint, least `os_proc_available_memory`), and the Core AI cache of the app is sized between the steps |
| `warmup` | one transcription (`zh`): the first call after the load |
| `e2e` | every clip in order: the generated ids against the fp32 oracle's (exact / knife-edge / mismatch), the Python engine's and the kit's Mac run's; the text against all three; per clip the front-end, encoder, prefill and decode times, the wall time and RTF; the Swift front end against the NumPy features; for a clip whose ids differ from the Mac's or the Python engine's, the decode from the NumPy features (which half moved). Thermal state, footprint and headroom every 20 s |
| `load2` | the model dropped and loaded again in the same process |
| `bench` | `ja_jp_1719` (13.6 s): wait for the thermal state nominal (up to `FUNASR_WAIT_NOMINAL` s, default 300), then one warm-up and 5 timed transcriptions back to back, each with its start offset (the iPhone 18 Pro's GPU slows after about 20 s of back-to-back work) |
| `md5` | md5 of the model files |

A clip passes when its ids equal the oracle's, or first differ at a step where the oracle's own top-2 softmax gap is below
0.1 (a knife-edge: the port took the oracle's runner-up where the oracle barely chose). The gate passes when every stage
passes: every clip exact or knife-edge, none at the 512-token cap, every file md5-equal to the stage.

The ids and the per-stage times come from `KitFunASRModel.transcribeWindows`, the window loop behind
`transcribe(samples:)`, reached through `@testable import CoreAIKit` as the kit's smoke tests do. `_build.sh` therefore
builds every module with `ENABLE_TESTABILITY=YES`. `_build.sh --mac --public` builds the same app without testability
anywhere, on the public entry point only (text and wall time): the control that shows what `-enable-testing` costs.

## Where the kit is

`project.yml` takes the kit by a path relative to this directory: `../../../../coreai-kit-funasr-wt/coreai-kit`, a kit
checkout that carries `Sources/CoreAIKit/FunASR`, beside the directory that holds this repository's checkout. If yours
is elsewhere, change `packages: coreai-kit: path:` in `project.yml` and pass `FUNASR_KIT=<path>` to `_build.sh` (it
checks the kit's `Package.resolved` before and after a build, and restores it when the build changed a clean file).

## Steps

1. `./_build.sh` (generic iOS, Release) and `./_build.sh --mac` (macOS arm64, Release). The `.app` paths go to
   `_work/app_path.txt` and `_work/app_path_mac.txt`.
2. `./_stage.sh` gathers the assets into `_work/device_stage/FunASRAssets/` (1.3 GB, APFS clones) and writes `MD5SUMS`.
   `FUNASR_DATA` (default `~/code/coreai/_funasr_nano`) is the port's working directory.
3. The Mac: `./_run_mac.sh` runs the macOS build on the stage directory under the machine-wide GPU lock
   (`~/code/coreai/_GPU_LOCK`; the script's header says how it reads the two conventions that share that file, and when
   it waits). Results land in `_work/mac_runs/<run id>/`, with the lock decision in `gpu_lock.json`.
4. The phone: unlock it, connect it by USB, get its ids from `xcrun devicectl list devices`. Then
   `FUNASR_SESSION=<name> FUNASR_ALLOWED_DEVICES=<udid>,<coredevice id> ./_gate.sh <udid>` takes the device hold
   (`~/code/coreai/ondevice/.device_hold`; while another session holds it, waits up to 60 min), installs the app,
   pushes `FunASRAssets/` to `Library/Application Support/`, pulls it back and re-pushes any file whose md5 differs,
   launches the app with no console, polls `result.json` until it is done, and releases the hold on any exit.
   `_work/device_runs/hold.log` records every take, wait and release; results land in `_work/device_runs/<run id>/`.
5. A cold first load: `FUNASR_FRESH=1 ./_gate.sh <udid>` uninstalls first (the data container goes with the app, and the
   container's Core AI cache with it), then installs and pushes everything again. A warm rerun on what is installed:
   `FUNASR_SKIP_INSTALL=1 ./_gate.sh <udid>`. Knobs go in as JSON members, e.g.
   `./_gate.sh <udid> '"FUNASR_STAGES":"load1,bench","FUNASR_WAIT_NOMINAL":"600"'` (all of them are listed at the top
   of `Sources/GateRunner.swift`). The poll cap is `FUNASR_CAP` polls of 10 s (default 180).
6. Exit codes: 0 = the gate passed, 3 = a stage failed, 1 = no result, 2 = the device is busy, held or refused. A run that
   does not end `done` leaves the phone's crash-log listing, and copies of today's FunASRGate / jetsam reports, in the
   run directory. Reprint a result: `./_run.sh --summary <result.json>`.

`_work/` is git-ignored. The scripts never pass `--remove-existing-content` (it wipes the whole app container) or
`--console`, and the app refuses to open an iPhone AOT bundle (`.h18p.`, `.h19p.`) on a Mac.
