#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""Reference files for the Swift host (CoreAIKit ``FunASR``): features, expected transcripts, manifest.

Writes, under ``<work>/_funasr_nano/swift_ref/``:

``feats/<name>.f32``
    ``frontend.fbank_lfr(wav)`` for every fixture clip — float32, little-endian, row-major ``[L, 560]``.
    The Swift front end (``FunASRFbankPreprocessor``) is compared against these row by row.
``oracle_feats/<clip>.f32``
    the fp32 oracle's own features (``oracle/<name>.npz`` ``speech``: torchaudio kaldi fbank + funasr LFR in
    float32), same layout — so the Swift front end can be measured against the oracle as well as the spec.
``manifest.json``
    per clip: name, wav path (relative to ``fixtures/``), sample count, L, N = ceil(L / 8), and the
    sha256 of the ``.f32`` file.
``expected.json``
    per clip: the fp32 oracle's text and gen_ids (EOS included), the Python engine run of the ship arm
    (encoder fp16w32 + decoder int8lin, ``gate_e2e/enc16w32_int8lin/``; the fp16-encoder arm
    ``enc16_int8lin`` stands in for any clip the ship-arm run has not written), and the oracle's
    top-2 softmax margins per generated step (``logs/r2_margins.json``) with the runner-up ids.

``prompt_variants.json``
    the prompt token ids funasr built for each ``get_prompt`` option: the default case (``oracle/zh.npz``)
    and every ``oracle_hotwords/<case>.json`` (``make_oracle.py --case ...``): hotwords, language, itn,
    ``source_ids`` (audio placeholders are id 0 at ``fbank_beg`` .. ``fbank_beg + N``), text, gen_ids.

    python dump_features.py                 # shared venv (NumPy + soundfile)
    python dump_features.py --variants-only # rewrite prompt_variants.json only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from _paths import work_path  # noqa: E402
from frontend import FEAT_DIM, fake_token_len, fbank_lfr, num_lfr_frames  # noqa: E402

WORK = work_path("_funasr_nano")
OUT = WORK / "swift_ref"
SHIP_ARM = "enc16w32_int8lin"
FALLBACK_ARM = "enc16_int8lin"


def write_prompt_variants() -> None:
    oracle = {c["name"]: c for c in json.loads((WORK / "oracle" / "oracle.json").read_text())["clips"]}
    z = np.load(WORK / "oracle" / "zh.npz")
    cases = [{"case": "default", "name": "zh", "hotwords": [], "language": None, "itn": True,
              "source_ids": z["source_ids"].astype(np.int64).tolist(), "fbank_beg": int(z["fbank_beg"]),
              "N": int(z["fake_token_len"]), "text": oracle["zh"]["text"], "gen_ids": oracle["zh"]["gen_ids"]}]
    for p in sorted((WORK / "oracle_hotwords").glob("*.json")):
        v = json.loads(p.read_text())
        case = {k: v[k] for k in ("case", "name", "hotwords", "language", "itn", "source_ids", "fbank_beg",
                                  "N", "text", "gen_ids")}
        # The Python engine run of the ship arm with the same hotwords (gate_e2e.py --hotwords), if any.
        hw = WORK / "gate_e2e" / f"hw_{SHIP_ARM}" / f"{v['name']}.json"
        if v["hotwords"] and v["language"] is None and v["itn"] and hw.exists():
            r = json.loads(hw.read_text())
            assert r["hotwords"] == v["hotwords"], (hw, r["hotwords"], v["hotwords"])
            case["python_engine"] = {"arm": SHIP_ARM, "text": r["text"], "gen_ids": r["gen_ids"]}
        cases.append(case)
    for c in cases:
        ids, beg, n = c["source_ids"], c["fbank_beg"], c["N"]
        assert ids[beg:beg + n] == [0] * n and len(ids) == beg + n + 5, c["case"]
    (OUT / "prompt_variants.json").write_text(json.dumps({
        "source": "oracle/zh.npz (default) + oracle_hotwords/<case>.json (make_oracle.py --case): funasr 1.4.16 "
                  "data_load_speech source_ids; audio placeholders are id 0 at [fbank_beg, fbank_beg + N)",
        "cases": cases}, ensure_ascii=False, indent=1))
    print(f"[swift_ref] {len(cases)} prompt cases -> {OUT / 'prompt_variants.json'}: "
          + ", ".join(f"{c['case']} ({len(c['source_ids'])} ids)" for c in cases), flush=True)


def main() -> None:
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", help="clip names (default: every clip in fixtures/meta.json)")
    ap.add_argument("--variants-only", action="store_true", help="only rewrite prompt_variants.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    write_prompt_variants()
    if args.variants_only:
        return

    meta = json.loads((WORK / "fixtures" / "meta.json").read_text())["clips"]
    oracle = {c["name"]: c for c in json.loads((WORK / "oracle" / "oracle.json").read_text())["clips"]}
    margins = json.loads((WORK / "logs" / "r2_margins.json").read_text())["clips"]
    clips = [c for c in meta if not args.only or c["name"] in args.only]
    (OUT / "feats").mkdir(parents=True, exist_ok=True)
    (OUT / "oracle_feats").mkdir(parents=True, exist_ok=True)

    manifest, expected, sources = [], [], {SHIP_ARM: 0, FALLBACK_ARM: 0}
    for c in clips:
        name = c["name"]
        wav, sr = sf.read(str(WORK / "fixtures" / c["path"]), dtype="float32")
        assert sr == 16000 and wav.ndim == 1, (name, sr, wav.shape)
        feats = fbank_lfr(wav)
        L = feats.shape[0]
        N = math.ceil(L / 8)
        assert feats.dtype == np.float32 and feats.shape == (L, FEAT_DIM), (name, feats.shape)
        assert L == num_lfr_frames(wav.shape[0]) == oracle[name]["L"], (name, L, oracle[name]["L"])
        assert N == fake_token_len(L) == oracle[name]["N"], (name, N, oracle[name]["N"])
        blob = np.ascontiguousarray(feats).astype("<f4").tobytes()
        (OUT / "feats" / f"{name}.f32").write_bytes(blob)
        speech = np.load(WORK / "oracle" / f"{name}.npz")["speech"]
        assert speech.dtype == np.float32 and speech.shape == (L, FEAT_DIM), (name, speech.shape)
        (OUT / "oracle_feats" / f"{name}.f32").write_bytes(np.ascontiguousarray(speech).astype("<f4").tobytes())
        manifest.append({"name": name, "wav": c["path"], "num_samples": int(wav.shape[0]), "L": L, "N": N,
                         "sha256": hashlib.sha256(blob).hexdigest()})

        arm = SHIP_ARM if (WORK / "gate_e2e" / SHIP_ARM / f"{name}.json").exists() else FALLBACK_ARM
        port = json.loads((WORK / "gate_e2e" / arm / f"{name}.json").read_text())
        assert port["name"] == name and port["N"] == N and port["L"] == L, (name, arm)
        sources[arm] += 1
        o = oracle[name]
        m = margins[name]
        assert len(m["margins_pos0"]) == len(o["gen_ids"]), (name, len(m["margins_pos0"]), len(o["gen_ids"]))
        expected.append({"name": name, "N": N, "L": L,
                         "oracle_text": o["text"], "oracle_gen_ids": o["gen_ids"],
                         "ship_arm": arm, "ship_text": port["text"], "ship_gen_ids": port["gen_ids"],
                         "margins_pos0": m["margins_pos0"], "runner_up_pos0": m["runner_up_pos0"]})
        print(f"{name}: samples {wav.shape[0]} L {L} N {N} ship arm {arm}", flush=True)

    common = {"feature_file": "float32 little-endian row-major [L, 560] = frontend.fbank_lfr(wav)",
              "frontend": "conversion/funasr_nano/frontend.py (NumPy float64, cast to float32)",
              "fixtures_root": str(WORK / "fixtures")}
    (OUT / "manifest.json").write_text(json.dumps({**common, "clips": manifest}, ensure_ascii=False, indent=1))
    (OUT / "expected.json").write_text(json.dumps({
        "oracle": "funasr 1.4.16 AutoModel fp32 CPU, dither 0, greedy, itn True, language None, no hotwords "
                  "(oracle/oracle.json); gen_ids include the EOS",
        "ship_arm": f"Python engine run (Mac GPU) of encoder fp16w32 + decoder int8lin unified static: "
                    f"gate_e2e/{SHIP_ARM}/ ({sources[SHIP_ARM]} clips), {FALLBACK_ARM} for the rest "
                    f"({sources[FALLBACK_ARM]} clips)",
        "margins": "logs/r2_margins.json margins_pos0: oracle top-2 softmax gap per generated step (EOS step "
                   "included); knife-edge = gap < 0.1",
        "clips": expected}, ensure_ascii=False, indent=1))
    print(f"[swift_ref] {len(manifest)} clips -> {OUT} (ship arm {sources})", flush=True)


if __name__ == "__main__":
    main()
