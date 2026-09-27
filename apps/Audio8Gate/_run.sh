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
d, b = r.get("device", {}), r.get("build", {})
print(f"run {r.get('run_id')}: status {r.get('status')}, pass {r.get('pass')}, {num(r.get('elapsed_s'), '%.0f')} s | "
      f"{d.get('machine')} {d.get('hw_model')} {d.get('os')} (build {d.get('os_build')}, Core AI arch {d.get('coreai_architecture')}), "
      f"thermal {d.get('thermal')} -> {r.get('device_end', {}).get('thermal')}, battery {num(d.get('battery_level', -1) * 100, '%.0f')} % "
      f"{d.get('battery_state')}, low power {d.get('low_power_mode')} | {b.get('configuration')} build, testable {b.get('testable_import')}")
env = r.get("config", {}).get("env", {})
if env.get("AUDIO8_GPU_LOCK"): print(f"GPU lock: {env['AUDIO8_GPU_LOCK']}")
if r.get("fatal"): print("FATAL", r["fatal"])
S = r.get("stages", {})
for k in r.get("stage_order", []):
    s = S.get(k, {})
    tag = f"{k}: {'PASS' if s.get('pass') else 'FAIL'}" + (" (partial)" if s.get("partial") else "")
    err = f" | ERROR {s.get('error_step', '')} {s['error']}"[:300] if "error" in s else ""
    if k == "assets":
        print(f"{tag} | {s.get('md5sums_listed')} files, missing {len(s.get('missing', []))}, md5 checked {s.get('md5_checked')} "
              f"(different {len(s.get('md5_mismatch', []))}), {len(s.get('md5_deferred', []))} model files for the md5 stage, "
              f"free {num(s.get('free_gb'), '%.1f')} GB{err}")
    elif k == "load1":
        m = s.get("model_memory", {})
        print(f"{tag} | encoder alone {num(s.get('encoder_alone_s'))} s, model {num(s.get('model_s'))} s (decoder + cached "
              f"encoder), encoder warm {num(s.get('encoder_warm_s'))} s -> decoder ≈ {num(s.get('decoder_est_s'))} s, cold total ≈ "
              f"{num(s.get('cold_total_est_s'))} s | model peak footprint {num(m.get('peak_footprint_mb'), '%.0f')} MB, least available "
              f"{num(m.get('min_available_mb'), '%.0f')} MB | cache MB {num(s.get('cache_bytes_before', 0) / 1e6, '%.1f')} -> "
              f"{num(s.get('cache_bytes_after_encoder_alone', 0) / 1e6, '%.1f')} -> {num(s.get('cache_bytes_after_model', 0) / 1e6, '%.1f')} "
              f"-> {num(s.get('cache_bytes_after_encoder_warm', 0) / 1e6, '%.1f')}{err}")
    elif k == "warmup":
        print(f"{tag} | {s.get('clip')} {s.get('verdict')}, wall {num(s.get('wall_ms'), '%.1f')} ms, RTF {num(s.get('rtf'), '%.3f')}{err}")
    elif k == "e2e":
        m = s.get("summary", {})
        print(f"{tag} | {m.get('clips_done')}/{m.get('clips_planned')} clips, errors {m.get('errors')}, {num(m.get('audio_s_total'), '%.0f')} s "
              f"audio in {num(m.get('wall_s_total'), '%.1f')} s | RTF median {num(m.get('rtf_median'), '%.4f')}, p90 {num(m.get('rtf_p90'), '%.4f')}, "
              f"max {num(m.get('rtf_max'), '%.4f')} ({m.get('rtf_max_clip')}), aggregate {num(m.get('rtf_aggregate'), '%.4f')}; first 20 s "
              f"{num(m.get('rtf_median_first_20s'), '%.4f')} ({m.get('clips_first_20s')} clips), after {num(m.get('rtf_median_after_20s'), '%.4f')}{err}")
        if "exact_oracle" in m:
            print(f"  ids: oracle exact {m['exact_oracle']}, knife-edge {m['knife_edge']}, exact or knife-edge {m['exact_or_knife_edge']}"
                  f"/{m['clips_done']}, above the floor {m['mismatch_above_floor']}; == Python engine {m['exact_python_engine']}, == Mac Swift "
                  f"{m['exact_mac_swift']}/{m.get('mac_swift_reference')}; cap {m['hit_cap']}")
            print(f"  median: front end {num(m.get('frontend_ms_median'), '%.1f')} ms, encoder {num(m.get('encoder_ms_median'), '%.1f')} ms "
                  f"(p90 {num(m.get('encoder_ms_p90'), '%.1f')}), prefill {num(m.get('prefill_ms_median'), '%.1f')} ms (p90 "
                  f"{num(m.get('prefill_ms_p90'), '%.1f')}), decode {num(m.get('decode_ms_per_token_median'), '%.2f')} ms/token (p90 "
                  f"{num(m.get('decode_ms_per_token_p90'), '%.2f')}), {m.get('tokens_total')} tokens")
            for c in m.get("knife_edge_clips", []) + m.get("mismatch_clips", []):
                kind = "knife-edge" if c in m.get("knife_edge_clips", []) else "MISMATCH"
                print(f"  {kind} {c['clip']}: step {c['step']}, oracle margin {num(c['margin'], '%.4f')}, ours {c['ours']}, oracle "
                      f"{c['oracle']}, runner-up {c['oracle_runner_up']}")
            if m.get("ids_differ_mac_swift"): print(f"  ids differ from Mac Swift: {m['ids_differ_mac_swift']}")
            if m.get("ids_differ_python_engine"): print(f"  ids differ from the Python engine: {m['ids_differ_python_engine']}")
        print(f"  text: == oracle {m.get('text_equal_oracle')}, == Python engine {m.get('text_equal_python_engine')}, == Mac Swift "
              f"{m.get('text_equal_mac_swift')}/{m.get('mac_swift_reference')}; front end max|Δ| vs NumPy "
              f"{num(m.get('frontend_max_abs_delta'), '%.2e')} ({m.get('frontend_max_abs_delta_clip')})")
        mem = s.get("memory", {})
        tl = mem.get("timeline", [])
        if mem:
            print(f"  memory: peak footprint {num(mem.get('peak_footprint_mb'), '%.0f')} MB, least available "
                  f"{num(mem.get('min_available_mb'), '%.0f')} MB | thermal every 20 s: "
                  + ", ".join(f"{t[0]:.0f}s {t[1]}" for t in tl))
    elif k == "load2":
        m = s.get("model_memory", {})
        print(f"{tag} | model {num(s.get('model_s'))} s (again in this process), peak footprint {num(m.get('peak_footprint_mb'), '%.0f')} MB, "
              f"footprint after {num(s.get('footprint_mb_after'), '%.0f')} MB{err}")
    elif k == "bench":
        m = s.get("summary", {})
        w = s.get("wait_nominal")
        wt = (f" | waited {num(w.get('waited_s'), '%.0f')} s for nominal ({w.get('state_before')} -> {w.get('state_after')})" if w else "")
        print(f"{tag} | {s.get('clip')} {num(s.get('audio_s'))} s, {s.get('runs_timed')} runs after 1 warm-up: RTF median "
              f"{num(m.get('rtf_median'), '%.4f')}, p90 {num(m.get('rtf_p90'), '%.4f')}, wall median {num(m.get('wall_ms_median'), '%.1f')} ms | "
              f"encoder {num(m.get('encoder_ms_median'), '%.1f')} ms, prefill {num(m.get('prefill_ms_median'), '%.1f')} ms, decode "
              f"{num(m.get('decode_ms_per_token_median'), '%.2f')} ms/token | runs end at {num(m.get('timed_runs_end_s'), '%.1f')} s | thermal "
              f"{s.get('thermal_start')} -> {s.get('thermal_end')}{wt}{err}")
    elif k == "md5":
        print(f"{tag} | {len(s.get('files', []))} model files, {num(s.get('bytes', 0) / 1e6, '%.0f')} MB, different "
              f"{len(s.get('md5_mismatch', []))}{err}")
    else:
        print(f"{tag}{err}")
print("summary:", " ".join(r.get("summary", [])))
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
[ -f $OUT/result.log ] && { echo "--- last lines of result.log"; tail -8 $OUT/result.log | cut -c1-230; }
[ -f $OUT/result.json ] && { echo "--- summary"; summary $OUT/result.json $RUN_ID | tee -a $OUT/run.out; }
echo "files: $OUT"
[[ $state == done ]] || exit 1
/usr/bin/python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("pass") else 3)' $OUT/result.json
