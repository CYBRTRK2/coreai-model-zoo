#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""ASR round trip + speaker cosine on the wavs a gate-app run wrote (the phone's or the Mac's), next to the oracle's.

    ~/code/coreai/coreai-models/.venv/bin/python conversion/audio8_tts/asr_device_wavs.py --wavs <run dir>/wav --tag iphone18pro-run3
-> `$ZOO_WORK_ROOT/_audio8_tts/gate/device_<tag>/{asr.json,speaker.json,summary.json}`

The transcriber is the Fun-ASR fp32 oracle (asr_judge.py, in the Fun-ASR port's oracle venv); the speaker model is
WavLM-Base-Plus-SV (speaker_sim.py). The oracle's own wavs are scored in the same call so the two arms share one run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
from _paths import work_path  # noqa: E402

WORK = work_path("_audio8_tts")
ORACLE_VENV = work_path("_funasr_nano", "venv-oracle", "bin", "python")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", required=True, help="directory of <fixture>.wav from Audio8Gate")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    wavs = Path(args.wavs)
    out = WORK / "gate" / f"device_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    fx = json.loads((HERE / "fixtures.json").read_text())["fixtures"]
    meta = json.loads((WORK / "oracle" / "oracle.json").read_text())["fixtures"]
    asr_rows, spk_rows = [], []
    for f in fx:
        name = f["name"]
        wav = wavs / f"{name}.wav"
        if not wav.exists():
            print(f"[skip] {name}: no wav")
            continue
        info = meta[name]
        asr_rows.append({"id": name, "arm": "device", "wav": str(wav), "lang": f["lang"], "text": f["text"]})
        asr_rows.append({"id": name, "arm": "oracle", "wav": str(WORK / "oracle" / f"{name}.wav"), "lang": f["lang"], "text": f["text"]})
        if info.get("reference"):
            spk_rows.append({"id": name, "arm": "device", "wav": str(wav), "reference": info["reference"]["path"]})
            spk_rows.append({"id": name, "arm": "oracle", "wav": str(WORK / "oracle" / f"{name}.wav"), "reference": info["reference"]["path"]})
    (out / "asr_manifest.json").write_text(json.dumps(asr_rows, ensure_ascii=False, indent=1))
    subprocess.run([str(ORACLE_VENV), str(HERE / "asr_judge.py"), "--manifest", str(out / "asr_manifest.json"), "--out", str(out / "asr.json")], check=True)
    summary = {"tag": args.tag, "wavs": len(asr_rows) // 2, "asr": json.loads((out / "asr.json").read_text())["summary"]}
    if spk_rows:
        (out / "spk_manifest.json").write_text(json.dumps(spk_rows, ensure_ascii=False, indent=1))
        subprocess.run([sys.executable, str(HERE / "speaker_sim.py"), "--manifest", str(out / "spk_manifest.json"), "--out", str(out / "speaker.json")], check=True)
        summary["speaker"] = json.loads((out / "speaker.json").read_text())["summary"]
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
