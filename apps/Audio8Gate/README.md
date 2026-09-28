# Audio8Gate: the device gate for Audio8-TTS-Preview-0.6b

A headless app that runs the Audio8-TTS port ([`conversion/audio8_tts`](../../conversion/audio8_tts)) through
CoreAIKit's `Audio8TTS` host on sideloaded assets, over the 18 fixtures the port was gated on, on the iPhone or on the
Mac. It links CoreAIKit by path, so the synthesis path is exactly the one an app gets from `Audio8TTS`: the prompt
builder over swift-transformers, the `prefill` / `frame` / `first_frame` functions of the DualAR asset (slow AR step,
sampling and fast AR in one call per frame), the codec decoder in 160-frame windows. The gate starts at launch with no
UI input. It writes `result.json` (rewritten after every stage and every fixture), `result.log` and `wav/<fixture>.wav`:
in `Documents/audio8_gate/` on the iPhone, in `AUDIO8_OUT` on the Mac.

What it loads (staged by `_stage.sh`):

| path in the assets directory | what |
|---|---|
| `audio8_dualar_int8_cl2048_w32.aimodel/` | prefill + frame + first_frame over one weight set and one KV state pair (24 × 2 × 2048 × 64 fp16): int8 slow AR, fp16 fast AR, the sampler in the graph |
| `audio8_codec_decoder_fp16_t160.aimodel/` | codes `[1,10,160]` → wav `[1, 327680]` at 44.1 kHz |
| `tokenizer/` | tokenizer.json + tokenizer_config.json (class retagged `Qwen2Tokenizer` for swift-transformers) |
| `swift_ref/` | `manifest.json` (fixtures, packed prompts, the Python engine run's codes), the oracle's recorded uniform draws per fixture, `voices/` for the clone fixtures (`conversion/audio8_tts/dump_swift_ref.py`) |

Both assets are JIT `.aimodel`s: the phone specializes them at the first load and caches the result.

## Stages

| stage | what it records |
|---|---|
| `assets` | every file in `MD5SUMS` present, md5 of every file up to 16 MB |
| `load1` | `Audio8TTS(paths:)` — both assets cold — under a 100 ms memory sampler; the Core AI cache of the app sized before and after |
| `warmup` | one synthesis (`zh_2`, the oracle's draws replayed): the first call after the load |
| `e2e` | every fixture in manifest order, the oracle's recorded draws replayed through the host: the codes against the Python engine run of the same assets (identical prefix; a divergence is an fp16 knife-edge flip), eos reached, per fixture the prefill, frame and codec time, wall, RTF and time to first audio; the wav written; thermal, footprint and headroom every 20 s |
| `load2` | the model dropped and loaded again in the same process |
| `bench` | `en_2` (~5 s): wait for the thermal state nominal (up to `AUDIO8_WAIT_NOMINAL` s, default 300), then one warm-up and 5 timed syntheses with the host's own seeds |
| `md5` | md5 of the model files |

The gate passes when every stage passes: every fixture reaches eos, every file md5-equal to the stage. The identical-prefix
count is reported, not gated: sampling on fp16 GPU logits flips a knife-edge draw now and then, and the ASR round trip on
the written wavs (`conversion/audio8_tts/asr_judge.py`) is the quality check.

## Where the kit is

`project.yml` takes the kit by a path relative to this directory: `../../../../coreai-kit-audio8-wt/coreai-kit`, a kit
checkout that carries `Sources/CoreAIKit/Audio8TTS`. If yours is elsewhere, change `packages: coreai-kit: path:` in
`project.yml` and pass `AUDIO8_KIT=<path>` to `_build.sh`.

## Steps

1. `./_build.sh` (generic iOS, Release) and `./_build.sh --mac` (macOS arm64, Release). The `.app` paths go to
   `_work/app_path.txt` and `_work/app_path_mac.txt`.
2. `./_stage.sh` gathers the assets into `_work/device_stage/Audio8Assets/` (APFS clones) and writes `MD5SUMS`.
   `AUDIO8_DATA` (default `~/code/coreai/_audio8_tts`) is the port's working directory; `AUDIO8_TOKENIZER` the tokenizer
   directory (default `$AUDIO8_DATA/ship/tokenizer`).
3. The Mac: `./_run_mac.sh` runs the macOS build on the stage directory under the machine-wide GPU lock
   (`~/code/coreai/_GPU_LOCK`). Results land in `_work/mac_runs/<run id>/`.
4. The phone: `AUDIO8_SESSION=<name> AUDIO8_ALLOWED_DEVICES=<udid>,<coredevice id> ./_gate.sh <udid>` takes the device
   hold, installs the app, pushes `Audio8Assets/`, launches the gate and pulls `result.json`, `result.log` and `wav/`
   back into `_work/device_runs/<run id>/`.
