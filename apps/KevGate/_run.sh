#!/bin/zsh
# Launch KevGate on the phone and collect its result. No --console (after an install a console launch can be refused as
# Busy for minutes); the app writes Documents/kev_gate/result.log line by line, memory.tsv reading by reading and
# result.json when a stage starts, after every stage (and every 5 records), and this script pulls the three every 10 s
# until result.json carries this run's id with status done / failed, the app's process is gone, or the cap passes.
# Copied from apps/DeciderVisionGate/_run.sh (zoo main 2d214b3); the summary is Kev's.
#   ./_run.sh <udid> [extra env as JSON members]     e.g. ./_run.sh <udid> '"KEV_STAGES":"assets,load_jit"'
#   ./_run.sh --summary <result.json>                print the summary of a pulled (or a Mac) result again
# Normally from ./_gate.sh, which holds the phone; refuses without this lane's hold, and on a device KEV_ALLOWED_DEVICES
# does not list. App knobs: the KEV_* list at the top of Sources/GateRunner.swift.
# Poll cap: KEV_CAP polls of 10 s (default 360 = 60 min). When the phone stops answering (unplugged, locked up), the
# script says so and keeps polling until the cap: reconnecting is a person's job.
# A run that does not end "done" (the app gone, or the cap) also lists the phone's crash logs and copies the ones of
# today that name KevGate or a jetsam event into crash/.
# Output: _work/device_runs/<run id>/{result.json,result.log,memory.tsv,run.out,launch.log[,crash/]}
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
BID=com.daisukemajima.kevgate
DIR=${0:A:h}
W=${KEV_WORK:-$DIR/_work}
RDIR=Documents/kev_gate

summary() {  # summary <result.json> [run id]
  /usr/bin/python3 - "$@" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if len(sys.argv) > 2 and r.get("run_id") != sys.argv[2]:
    print(f"result.json is from run {r.get('run_id')}, not {sys.argv[2]}")
def num(v, f="%.2f"):
    return (f % v) if isinstance(v, (int, float)) and not isinstance(v, bool) else "-"
d = r.get("device", {})
de = r.get("device_end", {})
print(f"run {r.get('run_id')}: status {r.get('status')}, pass {r.get('pass')}, {num(r.get('elapsed_s'), '%.0f')} s | "
      f"{d.get('machine')} {d.get('hw_model')} {d.get('os')} (build {d.get('os_build')}, Core AI arch {d.get('coreai_architecture')}), "
      f"thermal {d.get('thermal')} -> {de.get('thermal')}, battery {num(d.get('battery_level', -1) * 100, '%.0f')} % "
      f"{d.get('battery_state')} ({d.get('power_source')}) -> {num((de.get('battery_level') or -1) * 100, '%.0f')} %, low power "
      f"{d.get('low_power_mode')}, free {num(d.get('free_gb'), '%.1f')} GB, available {num(d.get('available_mb'), '%.0f')} MB at launch")
p = r.get("previous_launch")
if p:
    print(f"previous launch: run {p.get('run_id')} {p.get('status')}" + (f", died in {p['died_in_stage']}" if p.get("died_in_stage") else ""))
if r.get("fatal"): print("FATAL", r["fatal"])
S = r.get("stages", {})
def bar(s):
    m = s.get("mac", {})
    return (f"{s.get('questions')} q | argmax {s.get('argmax_equal_non_near_tie')}/{s.get('questions_non_near_tie')} + near-tie "
            f"{s.get('argmax_equal_near_tie')}/{s.get('near_tie_questions')}, ids {s.get('ids_equal_oracle')}, max|dp| "
            f"{num(s.get('max_abs_dp'), '%.6f')} ({s.get('worst_row')}), mean {num(s.get('mean_of_run_mean_abs_dp'), '%.6f')}, bar "
            f"{'PASS' if s.get('bar_pass') else 'FAIL'} | Mac: hidden {m.get('hidden_sha256_equal')}/{m.get('rows_compared')}, p bits "
            f"{m.get('p_bit_equal')}, argmax {m.get('argmax_equal')}, max|dp| {num(m.get('max_abs_dp'), '%.6f')}")
for k in r.get("stage_order", []):
    s = S.get(k, {})
    tag = f"{k}: {'SKIPPED' if s.get('skipped') else ('PASS' if s.get('pass') else 'FAIL')}" + (" (partial)" if s.get("partial") else "")
    err = f" | ERROR {s['error']}"[:300] if "error" in s else ""
    if k == "assets":
        print(f"{tag} | {s.get('md5sums_listed')} files, missing {len(s.get('missing', []))}, md5 checked {s.get('md5_checked')} "
              f"(different {len(s.get('md5_mismatch', []))}), {len(s.get('md5_deferred', []))} model files for md5, free "
              f"{num(s.get('free_gb'), '%.1f')} GB{err}")
    elif k == "md5":
        print(f"{tag} | {s.get('list')}: {len(s.get('files', []))} files, {num(s.get('bytes', 0) / 1e6, '%.0f')} MB in "
              f"{num(s.get('seconds'), '%.1f')} s, different {len(s.get('md5_mismatch', []))}{err}")
    elif k in ("load_jit", "load_aot", "load_4b"):
        ld, m = s.get("load", {}), s.get("memory", {})
        print(f"{tag} | {s.get('asset', '').split('/')[-1]} {num(s.get('asset_bytes', 0) / 1e6, '%.0f')} MB: wall "
              f"{num(s.get('wall_s'))} s (AIModel {num(ld.get('decoder_model_s'))} s, main {num(ld.get('decoder_function_s'))} s, "
              f"tokenizer {num(ld.get('tokenizer_s'))} s), peak footprint {num(m.get('peak_footprint_mb'), '%.0f')} MB, least available "
              f"{num(m.get('min_available_mb'), '%.0f')} MB, cache +{num(s.get('cache_bytes_added', (s.get('cache_bytes_after') or 0) - (s.get('cache_bytes_before') or 0)) / 1e6, '%.0f')} MB"
              + (f" | decide: {num(s.get('decide', {}).get('latency_ms'), '%.1f')} ms" if k == "load_4b" and s.get("decide") else "")
              + err)
    elif k == "warm":
        run = s.get("run", {})
        print(f"{tag} | {s.get('record')} ({s.get('kind')}): latency {num(run.get('latency_ms'), '%.1f')} ms, {run.get('calls')} calls{err}")
    elif k.startswith("e2e_"):
        sm, sh = s.get("summary", {}), s.get("shared_summary", {})
        line = f"{tag} | {s.get('set')} ({s.get('kind')}): {bar(sm)} | shared {sh.get('p_bit_equal_direct')}/{sh.get('records')} = direct"
        if s.get("ref_summary"):
            rs = s["ref_summary"]
            line += f" | vs JIT: p bits {rs.get('p_bit_equal')}/{rs.get('rows_compared')}, hidden {rs.get('hidden_sha256_equal')}, max|dp| {num(rs.get('max_abs_dp'), '%.6f')}"
        print(line + f" | {s.get('calls_total')} calls, call ms median {num(s.get('call_ms_median'))}, {num(s.get('seconds'), '%.0f')} s{err}")
    elif k == "reset":
        print(f"{tag} | {s.get('record')} hidden bit-equal {s.get('hidden_bit_equal')}, p bit-equal {s.get('p_bit_equal')}{err}")
    elif k.startswith("bench"):
        for it in s.get("items", []):
            sm = it.get("summary", {})
            wn = it.get("wait_nominal", {})
            modes = ", ".join(f"{m} median {num(v.get('median'), '%.1f')} ms [{' '.join(num(x, '%.0f') for x in v.get('latency_ms', []))}]"
                              for m, v in sm.items())
            print(f"{tag} | {it.get('item')} rows {it.get('row_tokens')}: {modes} | thermal {it.get('thermal_start')} -> "
                  f"{it.get('thermal_end')} (waited {num(wn.get('waited_s'), '%.0f')} s), battery "
                  f"{num(it.get('battery_start', {}).get('level', -1) * 100, '%.0f')} -> {num(it.get('battery_end', {}).get('level', -1) * 100, '%.0f')} % "
                  f"{it.get('battery_end', {}).get('power')}, end {num(it.get('timed_runs_end_s'), '%.1f')} s"
                  + (f" | ERROR {it['error']}" if 'error' in it else ""))
    elif k == "delete":
        print(f"{tag} | {[x.get('path') for x in s.get('deleted', [])]} free {num(s.get('free_gb_before'), '%.1f')} -> "
              f"{num(s.get('free_gb_after'), '%.1f')} GB{err}")
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
CAP=${KEV_CAP:-360}
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/run.out; }

[ -n "${KEV_ALLOWED_DEVICES:-}" ] || { say "refusing: KEV_ALLOWED_DEVICES is not set"; exit 2; }
if [[ ",$KEV_ALLOWED_DEVICES," != *",$UDID,"* ]]; then
  say "refusing device $UDID: not in KEV_ALLOWED_DEVICES ($KEV_ALLOWED_DEVICES)"; exit 2
fi
HOLD=${KEV_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
if ! { [ -f $HOLD ] && head -1 $HOLD | grep -q "^kev-0.8b device gate"; }; then
  say "the phone is not held by this lane ($HOLD: $(head -c 200 $HOLD 2>/dev/null || echo absent)); go through ./_gate.sh"; exit 2
fi
say "hold: $(head -1 $HOLD)"
IDS=($UDID ${(s:,:)${KEV_ALLOWED_DEVICES:-}})
busy() { ps -axo pid,command | grep -E "^ *[0-9]+ +(/[^ ]*/)?(xcrun )?devicectl device (process launch|install app|copy (to|from))" \
  | grep -qE -- "--device (${(j:|:)IDS})( |\$)"; }
for w in 1 2 3 4 5 6; do busy || break; sleep 10; done
busy && { say "device busy: another devicectl launch / install / copy on $UDID is running"; exit 2; }

# the phone answers for this app's container (the list State alone flips while the phone is fine)
L=$(xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier $BID \
  --subdirectory "Library/Application Support/KevAssets" 2>&1)
if echo "$L" | grep -q ERROR; then
  say "container not reachable (installed? phone unlocked, on USB?): $(echo "$L" | grep -m1 ERROR | cut -c1-160)"; exit 2
fi

ENVJ="{\"KEV_RUN_ID\":\"$RUN_ID\"${EXTRA:+,$EXTRA}}"
launched=0
for t in $(seq 1 12); do
  LO=$(xcrun devicectl device process launch --device $UDID --terminate-existing --environment-variables "$ENVJ" $BID 2>&1)
  echo "$LO" >> $OUT/launch.log
  if echo "$LO" | grep -q "Launched application"; then launched=1; break; fi
  say "launch retry $t: $(echo "$LO" | grep -m1 -iE 'error|busy' | cut -c1-160)"; sleep 15
done
[ $launched = 1 ] || { say "ERROR launch never accepted (see $OUT/launch.log)"; exit 1; }
say "launched run $RUN_ID (env $ENVJ); polling every 10 s, cap $((CAP * 10)) s"

pull() {  # pull <name>: Documents/kev_gate/<name> -> $OUT/<name> (kept when the pull fails)
  rm -f $OUT/$1.pull
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source $RDIR/$1 --destination $OUT/$1.pull >/dev/null 2>&1 && [ -f $OUT/$1.pull ] && mv $OUT/$1.pull $OUT/$1
}
state=""; last=""; gone=0; silent=0
for i in $(seq 1 $CAP); do
  sleep 10
  pull result.log; pull result.json; pull memory.tsv
  if [ -f $OUT/result.log ]; then
    cur=$(tail -1 $OUT/result.log)
    [ "$cur" != "$last" ] && { echo "  ${cur[1,260]}"; last=$cur; }
  fi
  if [ -f $OUT/result.json ]; then
    state=$(/usr/bin/python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r.get("status") if r.get("run_id")==sys.argv[2] else "other-run")' \
      $OUT/result.json $RUN_ID 2>/dev/null)
    [[ $state == done || $state == failed ]] && break
  fi
  # every minute: is the app still running? (two misses in a row = gone; a listing that fails says nothing about the app,
  # and three failures in a row say the phone stopped answering)
  if (( i % 6 == 0 )); then
    if ! P=$(xcrun devicectl device info processes --device $UDID 2>&1); then
      silent=$((silent + 1))
      (( silent == 3 )) && say "the phone has not answered for 3 min (unplugged? $(echo "$P" | grep -m1 -iE 'error' | cut -c1-120)); still polling until the cap"
      continue
    fi
    silent=0
    if echo "$P" | grep -q "KevGate"; then gone=0; else gone=$((gone + 1)); fi
    [ $gone -ge 2 ] && { say "the app's process is gone and result.json says '${state:-none}' (crash? see result.log)"; break; }
  fi
done
say "state: ${state:-no result.json} after $((i * 10)) s"
if [[ $state != done ]]; then
  # crash reports: the listing, then today's files that name the app or a jetsam event
  CL=$(xcrun devicectl device info files --device $UDID --domain-type systemCrashLogs 2>&1)
  echo "$CL" > $OUT/crashlogs_listing.txt
  names=(${(f)"$(echo "$CL" | grep -oE "[A-Za-z0-9._+-]*(KevGate|JetsamEvent)[A-Za-z0-9._+-]*$(date +%Y-%m-%d)[A-Za-z0-9._+-]*" | sort -u)"})
  if (( ${#names} )); then
    mkdir -p $OUT/crash
    for n in $names; do
      xcrun devicectl device copy from --device $UDID --domain-type systemCrashLogs --source "$n" --destination "$OUT/crash/$n" \
        >> $OUT/crash/copy.log 2>&1 && say "crash log: $OUT/crash/$n" || say "crash log $n: copy failed (see crash/copy.log)"
    done
  else
    say "no crash log of today names KevGate or a jetsam event (listing: crashlogs_listing.txt)"
  fi
fi
[ -f $OUT/result.log ] && { echo "--- last lines of result.log"; tail -8 $OUT/result.log | cut -c1-260; }
[ -f $OUT/result.json ] && { echo "--- summary"; summary $OUT/result.json $RUN_ID | tee -a $OUT/run.out; }
echo "files: $OUT"
[[ $state == done ]] || exit 1
/usr/bin/python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("pass") else 3)' $OUT/result.json
