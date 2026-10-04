#!/bin/zsh
# Run the macOS build of KevGate on this Mac on the Mac's own AOT asset (h16c), short, and print its summary: the check
# that the harness scores what the Mac's Swift gate (round 7) scored, before the phone. It does NOT take the machine-wide
# GPU lock: it runs only while the lock is free — no tag in ~/code/coreai/_GPU_LOCK, no process holding the file open
# (lsof: the kit's with-gpu-lock.py and the clef-flash scripts hold a flock(2) on it), no other GPU job of the lanes
# (readout gates, timing, decide workers, the LiteRT GPU parity, benches) and the GPU not busy (ioreg "Device
# Utilization %" >= 50 in 3 of 5 samples 1 s apart) — polled every 30 s for KEV_LOCK_WAIT s (default 1800), then it gives
# up (exit 4) rather than run contended. While the app runs, the same readings every 5 s go to gpu_lock.json.
# Adapted from apps/DeciderVisionGate/_run_mac.sh (zoo main 2d214b3), which takes the lock; this one only reads it.
#   ./_run_mac.sh                  KEV_STAGES=load_aot,e2e_fixture KEV_LIMIT=20 on the stage directory, KEV_AOT = the lane's
#                                  h16c asset (never the phone's h19p: the app refuses it on macOS)
#   ./_run_mac.sh --red            the same with an oracle_slim whose tv4_000 q0 probabilities are reversed: the bar must fail
#   any KEV_* knob of GateRunner.swift passes through the environment
# Output: _work/mac_runs/<run id>/{result.json,result.log,memory.tsv,app.stdout,gpu_lock.json,run.out}
set -u
DIR=${0:A:h}
W=${KEV_WORK:-$DIR/_work}
L=${KEV_LANE:-$HOME/code/coreai/_kev}
APP=$(cat $W/app_path_mac.txt 2>/dev/null)
[ -d "$APP" ] || { echo "no macOS build: run ./_build.sh --mac"; exit 1; }
BIN=$APP/Contents/MacOS/KevGate
S=${KEV_ASSETS:-$W/device_stage/KevAssets}
[ -f $S/MD5SUMS ] || { echo "nothing staged at $S: run ./_stage.sh"; exit 1; }
RED=0
[ "${1:-}" = "--red" ] && RED=1
RUN_ID=${KEV_RUN_ID:-mac-$(date +%Y%m%d-%H%M%S)$( (( RED )) && echo -red)}
OUT=$W/mac_runs/$RUN_ID
mkdir -p $OUT
LOCK=${COREAI_GPU_LOCK:-$HOME/code/coreai/_GPU_LOCK}
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/run.out; }
export KEV_ASSETS=$S KEV_OUT=$OUT KEV_RUN_ID=$RUN_ID
export KEV_AOT=${KEV_AOT:-$L/exports/bundles_aotc/kev_0_8b_decode_fp16_pf16.h16c.aimodelc}
export KEV_STAGES=${KEV_STAGES:-load_aot,e2e_fixture} KEV_LIMIT=${KEV_LIMIT:-20}
[[ $KEV_AOT == *.h16c.aimodelc ]] || { say "KEV_AOT $KEV_AOT: the Mac runs the h16c asset only"; exit 1; }
if (( RED )); then
  /usr/bin/python3 - $S/fixtures/oracle_slim.json $OUT/oracle_slim_red.json <<'PY'
import json, sys
o = json.load(open(sys.argv[1]))
q = o["records"]["tv4_000"]["questions"][0]
before = list(q["probs"])
q["probs"] = before[::-1]          # reversed: argmax moves (c -> b) and |dp| > 0.02
o["red_arm"] = {"row": "tv4_000:q0", "probs_before": before, "probs_after": q["probs"]}
json.dump(o, open(sys.argv[2], "w"))
print(f"red arm: tv4_000 q0 probs {before} -> {q['probs']}")
PY
  export KEV_ORACLE=$OUT/oracle_slim_red.json
fi
say "run $RUN_ID: $BIN on $S, KEV_AOT $KEV_AOT, stages $KEV_STAGES, limit $KEV_LIMIT${KEV_ORACLE:+, oracle $KEV_ORACLE}"

/usr/bin/python3 - "$LOCK" "$OUT" "$BIN" "${KEV_LOCK_WAIT:-1800}" "${KEV_MAC_CAP:-900}" <<'PY' 2>&1 | tee -a $OUT/run.out
import json, os, re, subprocess, sys, time
lock, out, binary, wait_cap, run_cap = sys.argv[1:6]
wait_cap, run_cap = float(wait_cap), float(run_cap)
GPU = re.compile(r"readout_gate\.py|timing\.py (run|worker)|decide\.py (check|worker|run)|--accel gpu|gpu_gate_subprocess|"
                 r"litert_parity\.py .*gpu|llm-bench|yardstick|coreai_verify|mlx_lm|/kev (fixture|time|run) |gate_swift\.py (pytime|longrow-python)|"
                 r"with-gpu-lock|parity_catalog")
me = os.getpid()

def reading():
    try:
        tag = open(lock).read().strip()
    except FileNotFoundError:
        tag = None
    lsof = subprocess.run(["lsof", lock], capture_output=True, text=True).stdout.strip().splitlines()[1:]
    ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout.splitlines()
    jobs = [l.strip()[:160] for l in ps if GPU.search(l) and "claude" not in l and "grep" not in l
            and int(l.split(None, 1)[0]) != me and "KevGate" not in l]
    vals = []
    for i in range(5):
        o = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
        m = re.findall(r'"Device Utilization %"=(\d+)', o)
        vals.append(max(map(int, m)) if m else -1)
        if i < 4: time.sleep(1)
    busy = sum(v >= 50 for v in vals) >= 3
    return {"t": time.strftime("%H:%M:%S"), "tag": tag, "lsof": lsof, "gpu_jobs": jobs, "gpu_util": vals, "gpu_busy": busy,
            "free": not tag and not lsof and not jobs and not busy}

rec = {"lock": lock, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "policy": "run only while the lock is free; never take it"}
t0 = time.time()
waits = []
while True:
    r = reading()
    waits.append(r)
    if r["free"]:
        break
    if time.time() - t0 >= wait_cap:
        rec.update({"state": f"not free after {time.time() - t0:.0f} s: not run", "waits": waits})
        json.dump(rec, open(f"{out}/gpu_lock.json", "w"), indent=1)
        print(f"GPU not free after {time.time() - t0:.0f} s (tag {r['tag']!r}, lsof {len(r['lsof'])}, jobs {r['gpu_jobs'][:2]}, util {r['gpu_util']}): not running", flush=True)
        sys.exit(4)
    print(f"GPU in use (tag {r['tag']!r}, lsof {len(r['lsof'])}, jobs {r['gpu_jobs'][:2]}, util {r['gpu_util']}); waiting 30 s "
          f"({time.time() - t0:.0f} s so far, cap {wait_cap:.0f} s)", flush=True)
    time.sleep(30)
rec.update({"state": f"free after {time.time() - t0:.0f} s", "waited_s": round(time.time() - t0, 1), "waits": waits})
print(f"GPU free ({rec['state']}): running the app", flush=True)
during = []
t1 = time.time()
with open(f"{out}/app.stdout", "w") as so:
    p = subprocess.Popen([binary], env=dict(os.environ), stdout=so, stderr=subprocess.STDOUT)
    while p.poll() is None:
        if time.time() - t1 > run_cap:
            print(f"the app ran past {run_cap:.0f} s: terminating pid {p.pid}", flush=True)
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
            break
        r = reading()
        r["t_s"] = round(time.time() - t1, 1)
        during.append(r)
        time.sleep(2)
rec.update({"app_exit": p.returncode, "app_s": round(time.time() - t1, 1), "during": during,
            "contended_samples": sum(not x["free"] for x in during), "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
json.dump(rec, open(f"{out}/gpu_lock.json", "w"), indent=1)
print(f"app exit {p.returncode} after {rec['app_s']} s; readings while it ran: {len(during)}, not free in {rec['contended_samples']} "
      f"(the app's own GPU work counts in the utilization)", flush=True)
sys.exit(0 if p.returncode == 0 else 1)
PY
rc=${pipestatus[1]}
if [ -f $OUT/result.json ]; then
  echo "--- last lines of result.log"; tail -6 $OUT/result.log | cut -c1-230
  echo "--- summary"; $DIR/_run.sh --summary $OUT/result.json $RUN_ID | tee -a $OUT/run.out
else
  say "no result.json (exit $rc; see $OUT/app.stdout)"
fi
echo "files: $OUT"
[ -f $OUT/result.json ] || exit ${rc:-1}
/usr/bin/python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("status")=="done" and r.get("pass") else 3)' $OUT/result.json
