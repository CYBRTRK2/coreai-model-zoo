#!/bin/zsh
# Gather what Audio8Gate reads into _work/device_stage/Audio8Assets/ (APFS clones: no extra disk), then write
# Audio8Assets/MD5SUMS (every file but itself). ./_install.sh pushes the directory as it is; the Mac run reads it in place
# (AUDIO8_ASSETS, ./_run_mac.sh).
#   ./_stage.sh
#   AUDIO8_DATA=<dir> ./_stage.sh          the port's working directory (default ~/code/coreai/_audio8_tts)
#   AUDIO8_DUALAR=<name> AUDIO8_CODEC=<name> ./_stage.sh    other bundle names under $AUDIO8_DATA/exports
#                                          (default: the shipped pair; the app takes the same names at launch)
# Layout (GateRunner.swift reads it):
#   <dualar>.aimodel/ <codec>.aimodel/                <- $AUDIO8_DATA/exports/  (int8 slow AR + fp16 fast AR + sampling in one asset; fp16 codec decoder)
#   tokenizer/                                        <- the checkpoint's tokenizer.json + tokenizer_config.json (+ special_tokens_map.json)
#   swift_ref/                                        <- $AUDIO8_DATA/swift_ref (dump_swift_ref.py: manifest, recorded draws, Python codes, voices)
set -euo pipefail
DIR=${0:A:h}
W=${AUDIO8_WORK:-$DIR/_work}
DATA=${AUDIO8_DATA:-$HOME/code/coreai/_audio8_tts}
DUALAR=${AUDIO8_DUALAR:-audio8_dualar_int8_cl2048_w32}
CODEC=${AUDIO8_CODEC:-audio8_codec_decoder_fp16_t160}
TOK=${AUDIO8_TOKENIZER:-$DATA/ship/tokenizer}
S=$W/device_stage/Audio8Assets

need() { [ -e "$1" ] || { echo "missing: $1"; exit 1; } }
for b in $DUALAR $CODEC; do for f in main.mlirb main.hash metadata.json; do need $DATA/exports/$b.aimodel/$f; done; done
need $TOK/tokenizer.json; need $TOK/tokenizer_config.json
need $DATA/swift_ref/manifest.json
junk=$(find $DATA/exports/$DUALAR.aimodel $DATA/exports/$CODEC.aimodel \( -name '.*' -o -name '*.incomplete' \) -print -quit)
[ -z "$junk" ] || { echo "$junk: a hidden or partial file (an export still running?)"; exit 1; }

[[ $S == */_work/device_stage/Audio8Assets ]] || { echo "refusing to clear $S"; exit 1; }
rm -rf $S
mkdir -p $S
for b in $DUALAR $CODEC; do cp -cR $DATA/exports/$b.aimodel $S/$b.aimodel; done
cp -cR $TOK $S/tokenizer
cp -cR $DATA/swift_ref $S/swift_ref
/usr/bin/python3 - $S/swift_ref <<'PY'
import json, os, sys
root = sys.argv[1]
m = json.load(open(f"{root}/manifest.json"))
miss = [f for fx in m["fixtures"] for f in (fx["noise_slow"], fx["noise_fast"], fx["python_codes"]) + ((fx["voice"],) if fx["voice"] else ())
        if not os.path.isfile(f"{root}/{f}")]
assert not miss, f"missing in the stage: {miss[:5]}"
print(f"swift_ref: {len(m['fixtures'])} fixtures, tag {m['tag']}")
PY

(cd $S && find . -type f ! -name MD5SUMS | sed 's|^\./||' | LC_ALL=C sort | while read -r f; do md5 -r "$f"; done > MD5SUMS)
echo "staged $S: $(grep -c . $S/MD5SUMS) files + MD5SUMS, $(du -sh $S | cut -f1)"
for d in $DUALAR.aimodel $CODEC.aimodel tokenizer swift_ref; do
  printf "  %-44s %8.1f MB  %4d files\n" $d $(( $(find $S/$d -type f -exec stat -f %z {} + | paste -sd+ - | bc) / 1e6 )) \
    $(find $S/$d -type f | wc -l)
done
