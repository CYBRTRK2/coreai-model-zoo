#!/bin/zsh
# Gather what FunASRGate reads into _work/device_stage/FunASRAssets/ (APFS clones: no extra disk), then write
# FunASRAssets/MD5SUMS (every file but itself). ./_install.sh pushes the directory as it is; the Mac run reads it in place
# (FUNASR_ASSETS, ./_run_mac.sh).
#   ./_stage.sh
#   FUNASR_DATA=<dir> ./_stage.sh     the port's working directory (default ~/code/coreai/_funasr_nano)
#   FUNASR_DECODER_SRC=<dir> FUNASR_ENCODER_SRC=<dir> ./_stage.sh     another bundle pair (default: the shipped pair)
#   FUNASR_EXTRA="<rel path>=<src path>,..." ./_stage.sh     more bundles cloned in beside them (e.g. an AOT fallback
#                                     <name>.h19p.aimodelc for FUNASR_ENCODER / FUNASR_DECODER to name at launch)
# Layout (GateRunner.swift reads it):
#   decoder/           <- $FUNASR_DATA/exports/funasr_nano_2512_decode_int8lin_n63_s1/ (metadata.json, <name>.aimodel/,
#                         tokenizer/): int8 block-32 linears, fp16 tied head, static [1,1] query + dynamic KV
#   encoder.aimodel/   <- $FUNASR_DATA/exports/funasr_nano_audio_encoder_fp16w32_l500.aimodel/ (fp16 weights, fp32 compute)
#   fixtures/          <- $FUNASR_DATA/fixtures/{examples,fleurs/*}/*.wav + meta.json (155 clips, 1836 s)
#                         $FUNASR_DATA/swift_ref/{expected.json,manifest.json,feats/} (the oracle's and the Python engine's
#                         ids and text; the NumPy front end's features)
#                         mac_swift_ref.json: name, gen_ids and text per clip of the kit's Mac run
#                         ($FUNASR_DATA/logs/sup_r3/r3_swift_e2e.json, FunASRSmokeTests test 3)
set -euo pipefail
DIR=${0:A:h}
W=${FUNASR_WORK:-$DIR/_work}
DATA=${FUNASR_DATA:-$HOME/code/coreai/_funasr_nano}
DEC=${FUNASR_DECODER_SRC:-$DATA/exports/funasr_nano_2512_decode_int8lin_n63_s1}
ENC=${FUNASR_ENCODER_SRC:-$DATA/exports/funasr_nano_audio_encoder_fp16w32_l500.aimodel}
MAC_REF=${FUNASR_MAC_REF:-$DATA/logs/sup_r3/r3_swift_e2e.json}
S=$W/device_stage/FunASRAssets

need() { [ -e "$1" ] || { echo "missing: $1"; exit 1; } }
for f in metadata.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json; do need $DEC/$f; done
DEC_MODEL=$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["assets"]["main"])' $DEC/metadata.json)
for f in main.mlirb main.hash metadata.json; do need $DEC/$DEC_MODEL/$f; need $ENC/$f; done
for f in fixtures/meta.json swift_ref/expected.json swift_ref/manifest.json swift_ref/feats $MAC_REF; do
  [[ $f == /* ]] && need $f || need $DATA/$f
done
junk=$(find $DEC $ENC \( -name '.*' -o -name '*.incomplete' -o -name '*.lock' -o -name '*.aria2' \) -print -quit)
[ -z "$junk" ] || { echo "$junk: a hidden or partial file (a download or export still running?)"; exit 1; }
typeset -A EXTRA
for e in ${(s:,:)${FUNASR_EXTRA:-}}; do
  rel=${e%%=*}; p=${e#*=}; p=${p%/}
  [[ $e == *=* && -n $rel && $rel != /* && $rel != *..* ]] || { echo "FUNASR_EXTRA entry '$e': want <rel path>=<src path>"; exit 1; }
  need $p
  EXTRA[$rel]=$p
done

[[ $S == */_work/device_stage/FunASRAssets ]] || { echo "refusing to clear $S"; exit 1; }
rm -rf $S
mkdir -p $S/fixtures
cp -cR $DEC $S/decoder
cp -cR $ENC $S/encoder.aimodel
# the wavs meta.json names (relative to fixtures/), then the reference files
/usr/bin/python3 - $DATA/fixtures/meta.json <<'PY' | while read -r rel; do mkdir -p $S/fixtures/${rel:h}; cp -c $DATA/fixtures/$rel $S/fixtures/$rel; done
import json, sys
for c in json.load(open(sys.argv[1]))["clips"]:
    print(c["path"])
PY
cp -c $DATA/fixtures/meta.json $DATA/swift_ref/expected.json $DATA/swift_ref/manifest.json $S/fixtures/
cp -cR $DATA/swift_ref/feats $S/fixtures/feats
/usr/bin/python3 - $MAC_REF $S/fixtures/mac_swift_ref.json <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
r = json.load(open(src))
clips = [{"name": c["name"], "gen_ids": c["gen_ids"], "text": c["text"]} for c in r["clips"]]
json.dump({"source": src.split("/_funasr_nano/")[-1] + " (kit FunASRSmokeTests test 3, Mac GPU, swift test debug build)",
           "summary": r.get("summary", {}), "clips": clips}, open(dst, "w"), ensure_ascii=False)
print(f"mac_swift_ref.json: {len(clips)} clips from {src}")
PY
for rel in ${(ok)EXTRA}; do mkdir -p $S/${rel:h}; cp -cR $EXTRA[$rel] $S/$rel; echo "extra: $rel <- $EXTRA[$rel]"; done

# every wav meta.json lists, one feature file per expected clip
/usr/bin/python3 - $S/fixtures <<'PY'
import json, os, sys
root = sys.argv[1]
meta = json.load(open(f"{root}/meta.json"))["clips"]
exp = json.load(open(f"{root}/expected.json"))["clips"]
miss = [c["path"] for c in meta if not os.path.isfile(f"{root}/{c['path']}")]
miss += [f"feats/{c['name']}.f32" for c in exp if not os.path.isfile(f"{root}/feats/{c['name']}.f32")]
assert not miss, f"missing in the stage: {miss[:5]}"
print(f"fixtures: {len(meta)} wavs, {len(exp)} expected clips, {sum(c['num_samples'] for c in meta) / 16000:.1f} s of audio")
PY

(cd $S && find . -type f ! -name MD5SUMS | sed 's|^\./||' | LC_ALL=C sort | while read -r f; do md5 -r "$f"; done > MD5SUMS)
echo "staged $S: $(grep -c . $S/MD5SUMS) files + MD5SUMS, $(du -sh $S | cut -f1)"
for d in decoder encoder.aimodel fixtures ${(k)EXTRA}; do
  printf "  %-40s %8.1f MB  %4d files\n" $d $(( $(find $S/$d -type f -exec stat -f %z {} + | paste -sd+ - | bc) / 1e6 )) \
    $(find $S/$d -type f | wc -l)
done
