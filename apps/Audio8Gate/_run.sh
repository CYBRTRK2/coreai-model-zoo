#!/bin/zsh
# Launch Audio8Gate on the phone and collect its result. No --console (after an install a console launch can be refused
# as Busy for minutes); the app writes Documents/audio8_gate/result.log line by line and result.json after every stage
# (and every 10 clips), and this script pulls both every 10 s until result.json carries this run's id with status
# done / failed, the app's process is gone, or the cap passes. Then prints the summary.
#   ./_run.sh <udid> [extra env as JSON members]     e.g. ./_run.sh <udid> '"AUDIO8_STAGES":"load1,bench"'
#   ./_run.sh --summary <result.json>                print the summary of a pulled (or a Mac) result again
# Normally from ./_gate.sh, which holds the phone; refuses without this lane's hold, and (when AUDIO8_ALLOWED_DEVICES is
# set) on a device it does not list.
# App knobs (GateRunner.swift): AUDIO8_STAGES, AUDIO8_CLIPS, AUDIO8_WAIT_NOMINAL, AUDIO8_BENCH_RUNS, AUDIO8_BENCH_CLIP,
# AUDIO8_WARMUP_CLIP, AUDIO8_DECODER, AUDIO8_ENCODER, AUDIO8_ISOLATE.
# Poll cap: AUDIO8_CAP polls of 10 s (default 180 = 30 min). When the phone stops answering (unplugged, locked up), the
# script says so and keeps polling until the cap: reconnecting is a person's job.
# A run that does not end "done" (the app gone, or the cap) also lists the phone's crash logs and copies the ones of
# today that name Audio8Gate or a jetsam event into crash/.
# Output: _work/device_runs/<run id>/{result.json,result.log,run.out,launch.log[,crash/]}
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
BID=com.daisukemajima.audio8gate
DIR=${0:A:h}
W=${AUDIO8_WORK:-$DIR/_work}

summary() {  # summary <result.json> [run id]
  /usr/bin/python3 - "$@" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if len(sys.argv) > 2 and r.get("run_id") != sys.argv[2]:
    print(f"result.json is from run {r.get('run_id')}, not {sys.argv[2]}")
def num(v, f="%.2f"):
    return (f % v) if isinstance(v, (int, float)) and not isinstance(v, bool) else "-"
d = r.get("device", {})
print(f"run {r.get('run_id')}: status {r.get('status')}, verdict {r.get('verdict')}, {num(r.get('total_seconds'), '%.0f')} s | "
      f"{d.get('machine')} {d.get('hw_model')} {d.get('os')} (build {d.get('os_build')}, Core AI arch {d.get('coreai_architecture')}), "
      f"thermal {d.get('thermal')}, low power {d.get('low_power_mode')}")
env = r.get("config", {}).get("env", {})
if env.get("AUDIO8_GPU_LOCK"): print(f"GPU lock: {env['AUDIO8_GPU_LOCK']}")
S = r.get("stages", {})
for k in r.get("stage_order", []):
    s = S.get(k, {})
    tag = f"{k}: {'ok' if s.get('ok') else 'FAIL'}" + (" (partial)" if s.get("partial") else "")
    err = f" | ERROR {s['error']}"[:300] if "error" in s else ""
    if k == "assets":
        print(f"{tag} | {s.get('listed')} files listed, {s.get('checked')} md5-checked, {s.get('deferred')} deferred, missing {len(s.get('missing', []))}, "
              f"bad {len(s.get('bad', []))}, {s.get('fixtures')} fixtures (ref {s.get('ref_tag')}){err}")
    elif k in ("load1", "load2"):
        m = s.get("memory", {})
        print(f"{tag} | both assets {num(s.get('load_s'))} s | peak footprint {num(m.get('peak_footprint_mb'), '%.0f')} MB | Core AI cache MB "
              f"{num(s.get('coreai_cache_mb_before'), '%.0f')} -> {num(s.get('coreai_cache_mb_after'), '%.0f')} | thermal {s.get('thermal')}{err}")
    elif k == "warmup":
        print(f"{tag} | {s.get('fixture')}: {s.get('frames')} frames, {num(s.get('audio_s'))} s audio in {num(s.get('wall_s'))} s (rtf {num(s.get('rtf'), '%.3f')}), "
              f"eos {s.get('eos')}, first audio {num(s.get('first_audio_s'))} s{err}")
    elif k == "e2e":
        m = s.get("summary", {})
        mem = s.get("memory", {})
        print(f"{tag} | {m.get('fixtures')} fixtures, eos {m.get('eos_runs')}, codes == Python engine {m.get('identical_runs')} runs / "
              f"{m.get('identical_prefix_frames')} of {m.get('frames')} frames | RTF median {num(m.get('rtf_median'), '%.3f')} p90 {num(m.get('rtf_p90'), '%.3f')} "
              f"overall {num(m.get('rtf_overall'), '%.3f')} | frame {num(m.get('frame_ms_per_frame_median'), '%.1f')} ms, prefill {num(m.get('prefill_ms_median'), '%.0f')} ms, "
              f"first audio {num(m.get('first_audio_s_median'))} s | {num(m.get('audio_s'), '%.0f')} s audio in {num(m.get('wall_s'), '%.0f')} s | "
              f"peak footprint {num(mem.get('peak_footprint_mb'), '%.0f')} MB{err}")
    elif k == "bench":
        print(f"{tag} | {s.get('fixture')}: waited {num(s.get('waited_s'), '%.0f')} s ({s.get('thermal_start')}), RTF median {num(s.get('rtf_median'), '%.3f')} "
              f"(min {num(s.get('rtf_min'), '%.3f')} max {num(s.get('rtf_max'), '%.3f')}), frame {num(s.get('frame_ms_per_frame_median'), '%.1f')} ms, "
              f"first audio {num(s.get('first_audio_s_median'))} s{err}")
    elif k == "md5":
        print(f"{tag} | {s.get('checked')} large files checked, bad {len(s.get('bad', []))}{err}")
    else:
        print(f"{tag}{err}")
PY
}

if [ "${1:-}" = "--summary" ]; then summary "${2:?usage: _run.sh --summary <result.json>}"; exit 0; fi
UDID=${1:?usage: _run.sh <udid> [extra env JSON members]}
EXTRA=${2:-}
RUN_ID=$(date +%Y%m%d-%H%M%S)
OUT=$W/device_runs/$RUN_ID
mkdir -p $OUT
CAP=${AUDIO8_CAP:-180}
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/run.out; }

if [ -n "${AUDIO8_ALLOWED_DEVICES:-}" ] && [[ ",$AUDIO8_ALLOWED_DEVICES," != *",$UDID,"* ]]; then
  say "refusing device $UDID: not in AUDIO8_ALLOWED_DEVICES ($AUDIO8_ALLOWED_DEVICES)"; exit 2
fi
HOLD=${AUDIO8_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
if ! { [ -f $HOLD ] && head -1 $HOLD | grep -q "^Audio8-TTS device gate"; }; then
  say "the phone is not held by this lane ($HOLD: $(head -c 200 $HOLD 2>/dev/null || echo absent)); go through ./_gate.sh"; exit 2
fi
say "hold: $(head -1 $HOLD)"
IDS=($UDID ${(s:,:)${AUDIO8_ALLOWED_DEVICES:-}})
busy() { ps -axo pid,command | grep -E "^ *[0-9]+ +(/[^ ]*/)?(xcrun )?devicectl device (process launch|install app|copy (to|from))" \
  | grep -qE -- "--device (${(j:|:)IDS})( |\$)"; }
for w in 1 2 3 4 5 6; do busy || break; sleep 10; done
busy && { say "device busy: another devicectl launch / install / copy on $UDID is running"; exit 2; }

# the phone answers for this app's container (the list State alone flips while the phone is fine)
L=$(xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier $BID \
  --subdirectory "Library/Application Support/Audio8Assets" 2>&1)
if echo "$L" | grep -q ERROR; then
  say "container not reachable (installed? phone unlocked, on USB?): $(echo "$L" | grep -m1 ERROR | cut -c1-160)"; exit 2
fi

ENVJ="{\"AUDIO8_RUN_ID\":\"$RUN_ID\"${EXTRA:+,$EXTRA}}"
launched=0
for t in $(seq 1 12); do
  LO=$(xcrun devicectl device process launch --device $UDID --terminate-existing --environment-variables "$ENVJ" $BID 2>&1)
  echo "$LO" >> $OUT/launch.log
  if echo "$LO" | grep -q "Launched application"; then launched=1; break; fi
  say "launch retry $t: $(echo "$LO" | grep -m1 -iE 'error|busy' | cut -c1-140)"; sleep 15
done
[ $launched = 1 ] || { say "ERROR launch never accepted (see $OUT/launch.log)"; exit 1; }
say "launched run $RUN_ID (env $ENVJ); polling every 10 s, cap $((CAP * 10)) s"

state=""; last=""; gone=0; silent=0
for i in $(seq 1 $CAP); do
  sleep 10
  rm -f $OUT/result.log.pull $OUT/result.json.pull
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source Documents/audio8_gate/result.log --destination $OUT/result.log.pull >/dev/null 2>&1 \
    && [ -f $OUT/result.log.pull ] && mv $OUT/result.log.pull $OUT/result.log
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source Documents/audio8_gate/result.json --destination $OUT/result.json.pull >/dev/null 2>&1 \
    && [ -f $OUT/result.json.pull ] && mv $OUT/result.json.pull $OUT/result.json
  if [ -f $OUT/result.log ]; then
    cur=$(tail -1 $OUT/result.log)
    [ "$cur" != "$last" ] && { echo "  ${cur[1,230]}"; last=$cur; }
  fi
  if [ -f $OUT/result.json ]; then
    state=$(/usr/bin/python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r.get("status") if r.get("run_id")==sys.argv[2] else "other-run")' \
      $OUT/result.json $RUN_ID 2>/dev/null)
    [[ $state == done || $state == failed ]] && break
  fi
  # every minute: is the app still running? (two misses in a row = gone; a listing that fails says nothing about the
  # app, and three failures in a row say the phone stopped answering)
  if (( i % 6 == 0 )); then
    if ! P=$(xcrun devicectl device info processes --device $UDID 2>&1); then
      silent=$((silent + 1))
      (( silent == 3 )) && say "the phone has not answered for 3 min (unplugged? $(echo "$P" | grep -m1 -iE 'error' | cut -c1-120)); still polling until the cap"
      continue
    fi
    silent=0
    if echo "$P" | grep -q "Audio8Gate"; then gone=0; else gone=$((gone + 1)); fi
    [ $gone -ge 2 ] && { say "the app's process is gone and result.json says '${state:-none}' (crash? see result.log)"; break; }
  fi
done
say "state: ${state:-no result.json} after $((i * 10)) s"
if [[ $state != done ]]; then
  # crash reports: the listing, then today's files that name the app or a jetsam event
  CL=$(xcrun devicectl device info files --device $UDID --domain-type systemCrashLogs 2>&1)
  echo "$CL" > $OUT/crashlogs_listing.txt
  names=(${(f)"$(echo "$CL" | grep -oE "[A-Za-z0-9._+-]*(Audio8Gate|JetsamEvent)[A-Za-z0-9._+-]*$(date +%Y-%m-%d)[A-Za-z0-9._+-]*" | sort -u)"})
  if (( ${#names} )); then
    mkdir -p $OUT/crash
    for n in $names; do
      xcrun devicectl device copy from --device $UDID --domain-type systemCrashLogs --source "$n" --destination "$OUT/crash/$n" \
        >> $OUT/crash/copy.log 2>&1 && say "crash log: $OUT/crash/$n" || say "crash log $n: copy failed (see crash/copy.log)"
    done
  else
    say "no crash log of today names Audio8Gate or a jetsam event (listing: crashlogs_listing.txt)"
  fi
fi
# the generated wavs (the ASR round trip on the Mac reads them: conversion/audio8_tts/asr_judge.py)
rm -rf $OUT/wav
xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
  --source Documents/audio8_gate/wav --destination $OUT/wav >/dev/null 2>&1 && say "pulled $(ls $OUT/wav 2>/dev/null | wc -l | tr -d ' ') wavs" \
  || say "no wav/ pulled"
[ -f $OUT/result.log ] && { echo "--- last lines of result.log"; tail -8 $OUT/result.log | cut -c1-230; }
[ -f $OUT/result.json ] && { echo "--- summary"; summary $OUT/result.json $RUN_ID | tee -a $OUT/run.out; }
echo "files: $OUT"
[[ $state == done ]] || exit 1
/usr/bin/python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("pass") else 3)' $OUT/result.json
