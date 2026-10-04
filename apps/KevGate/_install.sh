#!/bin/zsh
# Install KevGate on the phone and push _work/device_stage/KevAssets/ (./_stage.sh) into its data container as one
# directory (Library/Application Support/KevAssets). Then pull the directory back and check every file against the
# staged MD5SUMS; a file that is missing or differs is pushed again on its own (up to 3 rounds). `devicectl device copy
# to` can exit 0 with a file cut short, hence the pull. Copied from apps/DeciderVisionGate/_install.sh (zoo main 2d214b3).
#   ./_install.sh <udid>                   normally from ./_gate.sh, which holds the phone
#   KEV_SKIP_APP=1 ./_install.sh <udid>    assets only (the app is already installed)
#   KEV_SKIP_PUSH=1 ./_install.sh <udid>   no whole-directory push (the assets are already there): the md5 check below
#                                          still pulls everything back and re-pushes what differs
#   KEV_STAGE_DIR=<dir> KEV_PUSH_ONLY=<rel>,<rel> KEV_SKIP_APP=1 ./_install.sh <udid>
#                                          push only these paths of another stage directory into KevAssets/<rel> (the
#                                          4B asset: ./_stage.sh --4b); KEV_VERIFY=sizes then checks every pushed file's
#                                          size from the phone's listing instead of pulling it back (the app's md5 stage
#                                          reads the bytes: KEV_MD5SUMS=MD5SUMS_4B)
#   KEV_FRESH=1 ./_install.sh <udid>       uninstall first: the app and its whole data container go (assets, results,
#                                          and the container's Core AI cache), then install and push everything
# Never passes --remove-existing-content (it wipes the whole app container, not the destination). The destination names
# the pushed directory itself (.../KevAssets, and .../<file> on a re-push): a push onto a parent flattens a bundle's tree.
# Runs only while this lane holds the phone (the first line of ~/code/coreai/ondevice/.device_hold starts with
# "kev-0.8b device gate"; ./_gate.sh takes and releases it), and only on a device KEV_ALLOWED_DEVICES lists.
# Log: _work/device_install_<time>.log
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
UDID=${1:?usage: _install.sh <udid>}
BID=com.daisukemajima.kevgate
DIR=${0:A:h}
W=${KEV_WORK:-$DIR/_work}
S=${KEV_STAGE_DIR:-$W/device_stage/KevAssets}
DEST="Library/Application Support/KevAssets"
LOG=$W/device_install_$(date +%Y%m%d-%H%M%S).log
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

[ -n "${KEV_ALLOWED_DEVICES:-}" ] || { say "refusing: KEV_ALLOWED_DEVICES is not set (the 18 Pro's UDID and CoreDevice id)"; exit 2; }
if [[ ",$KEV_ALLOWED_DEVICES," != *",$UDID,"* ]]; then
  say "refusing device $UDID: not in KEV_ALLOWED_DEVICES ($KEV_ALLOWED_DEVICES)"; exit 2
fi
HOLD=${KEV_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
if ! { [ -f $HOLD ] && head -1 $HOLD | grep -q "^kev-0.8b device gate"; }; then
  say "the phone is not held by this lane ($HOLD: $(head -c 200 $HOLD 2>/dev/null || echo absent)); go through ./_gate.sh"; exit 2
fi
# a real devicectl process on this phone only (a session whose prompt quotes these words must not count; other phones do
# not count): the id given plus KEV_ALLOWED_DEVICES (the same phone by UDID and by CoreDevice id)
IDS=($UDID ${(s:,:)${KEV_ALLOWED_DEVICES:-}})
busy() { ps -axo pid,command | grep -E "^ *[0-9]+ +(/[^ ]*/)?(xcrun )?devicectl device (process launch|install app|copy (to|from))" \
  | grep -qE -- "--device (${(j:|:)IDS})( |\$)"; }
for w in 1 2 3 4 5 6; do busy || break; sleep 10; done
busy && { say "device busy: another devicectl launch / install / copy on $UDID is running"; exit 2; }

push() {  # push <local path> <container path>
  xcrun devicectl device copy to --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source "$1" --destination "$2" >> $LOG 2>&1 || say "copy to exited non-zero for ${1:t} (checked below)"
}

# the 4B (or any extra) asset: only the listed paths, size check from the phone's listing
if [ -n "${KEV_PUSH_ONLY:-}" ]; then
  [ "${KEV_SKIP_APP:-0}" = 1 ] || { say "KEV_PUSH_ONLY pushes into the installed app: set KEV_SKIP_APP=1"; exit 1; }
  rels=(${(s:,:)KEV_PUSH_ONLY})
  for rel in $rels; do [ -e "$S/$rel" ] || { say "nothing staged at $S/$rel"; exit 1; }; done
  t0=$SECONDS
  for rel in $rels; do
    say "pushing $S/$rel ($(du -sh "$S/$rel" | cut -f1)) -> $DEST/$rel"
    t1=$SECONDS
    push "$S/$rel" "$DEST/$rel"
    say "push of $rel returned after $((SECONDS - t1)) s"
  done
  check_sizes() {  # every staged file under the pushed paths, by size in bytes, from the phone's listing of each file's
    # directory (--json-output: the table rounds sizes to "2.33 GB")
    local bad=0 f rel dir name want got J=$W/device_files_listing.json
    for rel in $rels; do
      for f in ${(f)"$(cd $S && find $rel -type f | LC_ALL=C sort)"}; do
        dir=${f:h}; name=${f:t}
        want=$(stat -f %z "$S/$f")
        [ "$dir" = . ] && dir=""
        rm -f $J
        xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier $BID \
          --subdirectory "$DEST${dir:+/$dir}" --json-output $J >> $LOG 2>&1
        got=$(/usr/bin/python3 -c 'import json,sys
try:
    fs = json.load(open(sys.argv[1]))["result"]["files"]
except Exception:
    sys.exit(0)
print(next((str(x["metadata"]["size"]) for x in fs if x.get("relativePath") == sys.argv[2] or x.get("name") == sys.argv[2]), ""))' $J "$name")
        if [ "$got" != "$want" ]; then
          bad=$((bad + 1)); say "size differs: $f staged $want, phone '${got:-absent}'"
        fi
      done
    done
    return $bad
  }
  if [ "${KEV_VERIFY:-sizes}" = sizes ]; then
    for round in 1 2 3; do
      if check_sizes; then say "sizes equal on the phone for every file of ${KEV_PUSH_ONLY} (check $round)"; break; fi
      [ $round -eq 3 ] && { say "ERROR sizes still differ after 2 re-pushes"; exit 1; }
      for rel in $rels; do push "$S/$rel" "$DEST/$rel"; done
    done
  fi
  say "pushed ${KEV_PUSH_ONLY} in $((SECONDS - t0)) s (the bytes are md5-checked on the phone by the md5 stage)"
  exit 0
fi

[ -f $S/MD5SUMS ] || { say "nothing staged at $S: run ./_stage.sh"; exit 1; }

# 0. KEV_FRESH=1: uninstall (the data container goes with the app), confirmed by the app list, grepped for the id
if [ "${KEV_FRESH:-0}" = 1 ]; then
  [ "${KEV_SKIP_APP:-0}" = 1 ] && { say "KEV_FRESH=1 reinstalls the app: unset KEV_SKIP_APP"; exit 1; }
  [ "${KEV_SKIP_PUSH:-0}" = 1 ] && { say "KEV_FRESH=1 pushes the assets again: unset KEV_SKIP_PUSH"; exit 1; }
  installed() {  # 0 = listed, 1 = not listed, 2 = no answer
    local A; A=$(xcrun devicectl device info apps --device $UDID 2>&1) || { echo "$A" >> $LOG; return 2; }
    echo "$A" | grep -qF "$BID"
  }
  installed; st=$?
  [ $st -eq 2 ] && { say "ERROR the app list did not answer (see $LOG)"; exit 1; }
  if [ $st -eq 0 ]; then
    OUT=$(xcrun devicectl device uninstall app --device $UDID $BID 2>&1); echo "$OUT" >> $LOG
    installed; st=$?
    [ $st -eq 1 ] || { say "ERROR $BID still listed (or no answer) after uninstall: $(echo "$OUT" | grep -m1 -i error | cut -c1-160)"; exit 1; }
    say "uninstalled $BID with its data container (KEV_FRESH=1)"
  else
    say "$BID was not installed (KEV_FRESH=1): nothing to uninstall"
  fi
fi

# 1. the app: success = an installationURL line (a failed install leaves the previous app in place)
if [ "${KEV_SKIP_APP:-0}" != 1 ]; then
  APP=$(cat $W/app_path.txt 2>/dev/null)
  [ -d "$APP" ] || { say "no built app: run ./_build.sh"; exit 1; }
  ok=0
  for attempt in 1 2 3; do
    OUT=$(xcrun devicectl device install app --device $UDID "$APP" 2>&1); echo "$OUT" >> $LOG
    if echo "$OUT" | grep -q installationURL; then ok=1; say "installed $BID"; break; fi
    say "install attempt $attempt failed: $(echo "$OUT" | grep -m1 -i error | cut -c1-200)"; sleep 10
  done
  [ $ok = 1 ] || { say "ERROR install failed 3 times"; exit 1; }
fi

# 2. the assets, the whole directory in one push
if [ "${KEV_SKIP_PUSH:-0}" != 1 ]; then
  say "pushing $S ($(du -sh $S | cut -f1), $(grep -c . $S/MD5SUMS) files) -> $DEST"
  t0=$SECONDS
  push $S "$DEST"
  say "push returned after $((SECONDS - t0)) s"
else
  say "no whole-directory push (KEV_SKIP_PUSH=1): checking what is on the phone against $S/MD5SUMS"
fi

# 3. pull back, compare with MD5SUMS; re-push what is missing or differs, one file at a time
BAD=()
verify() {
  local P=$W/device_pull
  rm -rf $P; mkdir -p $P
  BAD=()
  # the exact path the app reads (a single-file pull lands at the destination path itself)
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source "$DEST/MD5SUMS" --destination $P/MD5SUMS.exact >> $LOG 2>&1
  cmp -s $P/MD5SUMS.exact $S/MD5SUMS || BAD+=(MD5SUMS)
  # every file, one directory pull; the tree may land at the destination or one level under it, so its root is wherever
  # the shallowest MD5SUMS is
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source "$DEST" --destination $P/tree >> $LOG 2>&1
  local M=$(find $P/tree -name MD5SUMS 2>/dev/null | awk '{ print gsub("/", "/"), $0 }' | sort -n | head -1 | cut -d' ' -f2-)
  local R=${M:h}
  while read -r sum rel; do
    if [ -z "$M" ] || [ ! -f "$R/$rel" ] || [ "$(md5 -q "$R/$rel")" != "$sum" ]; then BAD+=("$rel"); fi
  done < $S/MD5SUMS
}
for round in 1 2 3 4; do
  t0=$SECONDS
  verify
  say "check $round: pulled back and compared in $((SECONDS - t0)) s: ${#BAD} files missing or different"
  [ ${#BAD} -eq 0 ] && break
  [ $round -eq 4 ] && break
  say "check $round: ${BAD[1,4]}; pushing them one by one"
  for rel in $BAD; do push "$S/$rel" "$DEST/$rel"; done
done
if [ ${#BAD} -ne 0 ]; then
  say "ERROR after 3 re-pushes ${#BAD} files still missing or different: ${BAD[1,8]}"; exit 1
fi
say "assets verified on the phone: $(grep -c . $S/MD5SUMS) files md5-equal to the stage, MD5SUMS equal"
rm -rf $W/device_pull
if [ "${KEV_FRESH:-0}" = 1 ]; then
  # what the new container's Library/Caches holds before the first launch (a fresh one: no coreai-cache)
  C=$(xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --subdirectory Library/Caches 2>&1); echo "$C" >> $LOG
  say "fresh container, Library/Caches before the first launch: $(echo "$C" | grep -c coreai-cache) lines naming coreai-cache"
fi
