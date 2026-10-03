#!/bin/zsh
# Time the Release clef-flash CLI on this Mac under the machine-wide GPU lock: a fixed set of decisions (text rows of
# about 200 / 500 / 1,000 / 2,500 tokens, two images at g256 and g448) through the AOT assets, CF_PASSES passes per
# bundle (default 2), each pass its own process: the first load in the process, one warm-up decision (the process's
# first decoder call pays the runtime's warm-up; recorded apart), every decision CF_REPEAT times (default 3), then
# everything dropped and loaded again (--reload).
#   ./_time_mac.sh
#   CF_BUNDLES="clef_flash_decode_int8mix_pf64" ./_time_mac.sh
# The lock protocol is apps/DeciderVision/_time_mac.sh's (apps/FunASRGate/_run_mac.sh's): ~/code/coreai/_GPU_LOCK
# (COREAI_GPU_LOCK overrides) is an advisory file two conventions share — the kit's scripts/with-gpu-lock.py holds
# flock(2) on it for the length of a command (and leaves the 0-byte file behind); the zoo's tools read its existence as
# "held", write a tag and remove it. This script takes the flock, writes a tag line while it runs, and leaves the file
# as it found it (a 0-byte file stays; a file that did not exist goes). Another session counts as using the GPU while
# it holds the flock, while the file carries a tag naming a live process, or while the GPU reads busy ("Device
# Utilization %" >= 50 in 3 of 5 samples 1 s apart); then this script waits in 30 s steps up to CF_LOCK_WAIT s
# (default 1800), and after that runs anyway and records "contended". The decision, the readings, a process snapshot
# before and after, and every 5 s the passes' scheduling priority with the load average and the GPU utilization go to
# <out>/gpu_lock.json.
# Output: ~/code/coreai/_clefflash/swift/timing/<run id>/{<bundle>_pass<k>.json,<bundle>_pass<k>.log,gpu_lock.json}
set -u
L=${CF_LANE:-$HOME/code/coreai/_clefflash}
BIN=${CF_BIN:-$L/swift/.build/release/clef-flash}
[ -x $BIN ] || { echo "no Release build at $BIN (swift build -c release --scratch-path $L/swift/.build)"; exit 1; }
RUN_ID=${CF_RUN_ID:-mac-$(date +%Y%m%d-%H%M%S)}
OUT=$L/swift/timing/$RUN_ID
mkdir -p $OUT
LOCK=${COREAI_GPU_LOCK:-$HOME/code/coreai/_GPU_LOCK}
# T ~ 200 (x2), ~ 450-500 (x2), ~ 970, ~ 2,500-2,600 (x2); two images at both grids
RUNS=${CF_RUNS:-semif_6149a17bc154f9c5b4a0:text,semif_a3f18f3a63d45345942b:text,own_t01:text,own_t11:text,own_t12:text,own_t16:text,own_t15:text,img_04:g256,img_04:g448,photo_02:g256,photo_02:g448}
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/run.out; }
say "run $RUN_ID: $BIN (GPU lock $LOCK)"

/usr/bin/python3 - "$LOCK" "$OUT" "$BIN" "$L" "$RUN_ID" "${CF_LOCK_WAIT:-1800}" "${CF_PASSES:-2}" \
  "${CF_BUNDLES:-clef_flash_decode_fp16_pf64 clef_flash_decode_int8mix_pf64}" "$RUNS" "${CF_REPEAT:-3}" <<'PY'
import fcntl, json, os, re, subprocess, sys, time
lock_path, out, binary, lane, run_id, wait_cap, passes, bundles, runs, repeat = sys.argv[1:11]
wait_cap, passes, bundles = float(wait_cap), int(passes), bundles.split()
rec = {"lock": lock_path, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "load_avg_start": os.getloadavg(),
       "runs": runs.split(","), "repeat": int(repeat)}
WATCH = re.compile(r"python|clef|decider|coreai|Gate|mlx|llama|ollama|Xcode|xcodebuild|swift-frontend|MTLCompiler|metal|"
                   r"yardstick|litert|llm-bench|coreai_verify|run_ddp", re.I)

def gpu_util():
    o = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
    m = re.findall(r'"Device Utilization %"=(\d+)', o)
    return max(map(int, m)) if m else -1

def gpu_busy():
    vals = []
    for i in range(5):
        vals.append(gpu_util())
        if i < 4: time.sleep(1)
    return sum(v >= 50 for v in vals) >= 3, vals

def snapshot(tag):
    o = subprocess.run(["ps", "-axo", "pid=,ppid=,pri=,%cpu=,rss=,etime=,command="], capture_output=True, text=True).stdout
    rows = []
    for line in o.splitlines():
        parts = line.split(None, 6)
        if len(parts) < 7: continue
        pid, ppid, pri, cpu, rss, etime, cmd = parts
        if int(pid) == os.getpid(): continue
        if float(cpu) >= 5.0 or WATCH.search(cmd):
            rows.append({"pid": int(pid), "ppid": int(ppid), "pri": int(pri), "cpu_pct": float(cpu), "rss_mb": int(rss) // 1024,
                         "etime": etime, "command": cmd[:240]})
    busy, vals = gpu_busy()
    return {"at": tag, "time": time.strftime("%H:%M:%S"), "load_avg": os.getloadavg(), "gpu_util_samples": vals,
            "gpu_busy": busy, "processes": rows}

def tag_holder():
    try:
        t = open(lock_path).read().strip()
    except FileNotFoundError:
        return None, None
    m = re.search(r"pid (\d+)", t)
    if m:
        try:
            os.kill(int(m.group(1)), 0)
            return t, True
        except (ProcessLookupError, PermissionError):
            return t, False
    return t, None

rec["snapshot_before_lock"] = snapshot("before lock")
existed = os.path.exists(lock_path)
size_before = os.path.getsize(lock_path) if existed else None
mtime_before = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(lock_path))) if existed else None
rec.update({"file_existed": existed, "file_bytes_before": size_before, "file_mtime_before": mtime_before})
f = open(lock_path, "a+")
t0 = time.time()
readings, state = [], None
while True:
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        have = True
    except BlockingIOError:
        have = False
    tag, live = tag_holder()
    busy, vals = gpu_busy()
    readings.append({"t_s": round(time.time() - t0, 1), "flock_free": have, "tag": tag, "tag_pid_live": live,
                     "gpu_util_samples": vals, "gpu_busy": busy})
    if have and live is not True and not busy:
        state = ("taken by this run (flock; file " + ("absent before" if not existed else
                 f"{size_before} bytes, last written {mtime_before}") + f"; GPU utilization {vals})")
        break
    if have:
        fcntl.flock(f, fcntl.LOCK_UN)
    if time.time() - t0 >= wait_cap:
        state = (f"contended: after {time.time() - t0:.0f} s, flock {'free' if have else 'held'}, tag {tag!r} (live {live}), "
                 f"GPU utilization {vals}; running anyway")
        break
    print(f"GPU in use by another session (flock {'free' if have else 'held'}, tag {tag!r}, live {live}, GPU {vals}); "
          f"waiting 30 s ({time.time() - t0:.0f} s so far, cap {wait_cap:.0f} s)", flush=True)
    time.sleep(30)
rec.update({"state": state, "waited_s": round(time.time() - t0, 1), "readings": readings})
print(f"GPU lock: {state}", flush=True)
ours = state.startswith("taken")
if ours:
    f.seek(0); f.truncate(); f.write(f"clef-flash timing {run_id} pid {os.getpid()} since {time.strftime('%Y-%m-%d %H:%M:%S')}\n"); f.flush()
rec["snapshot_after_lock"] = snapshot("after lock, before the passes")

ex = f"{lane}/exports"
rec["passes"] = []
for b in bundles:
    for k in range(1, passes + 1):
        js = f"{out}/{b.replace('clef_flash_decode_', '')}_pass{k}.json"
        cmd = [binary, "fixture", "--assets", ex, "--bundle-name", b, "--decoder-asset", "aot",
               "--records", f"{lane}/fixtures/records.json", "--images", f"{lane}/fixtures/images",
               "--arms", "text,g256,g448", "--runs", runs, "--warmup", "1", "--repeat", repeat, "--reload",
               "--out", js, "--label", f"timing {run_id} {b} pass {k} ({'lock taken' if ours else 'contended'})"]
        prio = []
        t1 = time.time()
        with open(js.replace(".json", ".log"), "w") as so:
            p = subprocess.Popen(cmd, stdout=so, stderr=subprocess.STDOUT)
            while p.poll() is None:
                pr = subprocess.run(["ps", "-o", "pri=", "-p", str(p.pid)], capture_output=True, text=True).stdout.strip()
                if pr:
                    prio.append([round(time.time() - t1, 1), int(pr), round(os.getloadavg()[0], 2), gpu_util()])
                time.sleep(5)
        low = [x for x in prio if x[1] < 20]
        rec["passes"].append({"bundle": b, "pass": k, "json": js, "exit": p.returncode, "s": round(time.time() - t1, 1),
                              "priority_every_5s": prio, "priority_min": min((x[1] for x in prio), default=None),
                              "demoted_samples": len(low)})
        print(f"{b} pass {k}: exit {p.returncode} after {time.time() - t1:.1f} s"
              + (f" (WARNING: background priority in {len(low)} samples)" if low else ""), flush=True)
rec["snapshot_after_passes"] = snapshot("after the passes, lock still held")
rec.update({"load_avg_end": os.getloadavg(), "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
if ours:
    f.seek(0); f.truncate(); f.flush()
    if not existed:
        os.remove(lock_path)
    fcntl.flock(f, fcntl.LOCK_UN)
f.close()
rec["file_after"] = ("absent" if not os.path.exists(lock_path) else f"{os.path.getsize(lock_path)} bytes")
json.dump(rec, open(f"{out}/gpu_lock.json", "w"), indent=1)
print(f"lock file after: {rec['file_after']}", flush=True)
sys.exit(0 if all(x["exit"] == 0 for x in rec["passes"]) else 1)
PY
rc=$?
say "done (exit $rc): $OUT"
exit $rc
