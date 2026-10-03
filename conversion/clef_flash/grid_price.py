#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
# ]
# [tool.uv]
# index-url = "https://pypi.org/simple"
# ///
"""What a fixed image grid costs clef-flash on the fixture, from the author's code alone (no Core AI).

A Core AI tower bakes one grid; the checkpoint's processor picks a grid per image (dynamic
resolution, at least 65,536 pixels = 256x256). This compares, for every image record of the
fixture and every question, the CPU fp32 oracle's answer on the image as drawn (`native`, the
processor's own grid) with the answer after a PIL BICUBIC resize to 256x256 (`g256`, 64 image
tokens) and to 448x448 (`g448`, 196 tokens): argmax agreement, max |dp| and mean |dp| over the
options, per arm the sequence length and image-token count, and the native grid. Everything is
read from `records_oracle.json` (oracle_clef.py, which runs the checkpoint's joint_schema_model.py).
`--arms native,g256,g448,g672,g896` adds the two larger grids (672x672 = 441 tokens, 896x896 = 784) from
`records_oracle_large.json` (oracle_clef.py --arms g672,g896 --tag _large).

    python3 conversion/clef_flash/grid_price.py        # -> $ZOO_WORK_ROOT/_clefflash/results/grid_price.json
    python3 conversion/clef_flash/grid_price.py --arms native,g256,g448,g672,g896 \
        --out $ZOO_WORK_ROOT/_clefflash/results/grid_price_large.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import work_path  # noqa: E402

ARMS = ("native", "g256", "g448")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", default=str(work_path("_clefflash", "oracle", "records_oracle.json")))
    ap.add_argument("--oracle-large", default=str(work_path("_clefflash", "oracle", "records_oracle_large.json")))
    ap.add_argument("--out", default=str(work_path("_clefflash", "results", "grid_price.json")))
    ap.add_argument("--arms", default=",".join(ARMS), help="native first, then the fixed grids to price")
    args = ap.parse_args()
    arms = tuple(args.arms.split(","))
    assert arms[0] == "native", arms
    path = Path(args.oracle).expanduser()
    oracle = json.loads(path.read_text())
    rows = {(x["id"], x["arm"]): x for x in oracle["rows"]}
    sources = [path]
    if any(a not in ARMS for a in arms):
        large = Path(args.oracle_large).expanduser()
        doc = json.loads(large.read_text())
        assert doc.get("complete"), f"{large} is not complete"
        rows.update({(x["id"], x["arm"]): x for x in doc["rows"]})
        sources.append(large)
    image_ids = sorted({x["id"] for x in oracle["rows"] if x["arm"] in arms})
    per_record, per_question = [], []
    for rid in image_ids:
        missing = [a for a in arms if (rid, a) not in rows]
        assert not missing, (rid, missing)
        nat = rows[(rid, "native")]
        rec = {"id": rid, "image_files": nat.get("image_files"), "image_size_in": nat.get("image_size_in"),
               "native_grid_thw": nat["grid_thw"], "arms": {}}
        for arm in arms:
            x = rows[(rid, arm)]
            rec["arms"][arm] = {"tokens": x["tokens"], "n_image_tokens": x["n_image_tokens"], "grid_thw": x["grid_thw"],
                                "wall_s": x["wall_s"]}
        for qi, qn in enumerate(nat["questions"]):
            item = {"id": rid, "question_id": qn["question_id"], "type": qn["type"], "n_options": len(qn["option_ids"]),
                    "native_argmax": qn["argmax_id"], "native_top2_margin": qn["top2_margin"], "gold": qn["gold"]}
            pn = np.asarray(qn["probs"])
            for arm in arms[1:]:
                qa = rows[(rid, arm)]["questions"][qi]
                assert qa["question_id"] == qn["question_id"] and qa["option_ids"] == qn["option_ids"]
                pa = np.asarray(qa["probs"])
                item[arm] = {"argmax": qa["argmax_id"], "argmax_agree": qa["argmax_id"] == qn["argmax_id"],
                             "max_abs_dp": float(np.abs(pa - pn).max()), "mean_abs_dp": float(np.abs(pa - pn).mean()),
                             "top2_margin": qa["top2_margin"]}
            per_question.append(item)
        per_record.append(rec)

    def agg(arm):
        qs = per_question
        if arm == "native":
            gold = [q for q in qs if q["gold"] is not None]
            return {"questions": len(qs), "gold_questions": len(gold),
                    "gold_correct": sum(q["native_argmax"] == q["gold"] for q in gold),
                    "mean_image_tokens": float(np.mean([r["arms"][arm]["n_image_tokens"] for r in per_record])),
                    "mean_tokens": float(np.mean([r["arms"][arm]["tokens"] for r in per_record]))}
        gold = [q for q in qs if q["gold"] is not None]
        flips = [f"{q['id']}/{q['question_id']}" for q in qs if not q[arm]["argmax_agree"]]
        return {"questions": len(qs), "argmax_agree_with_native": sum(q[arm]["argmax_agree"] for q in qs),
                "argmax_agree_rate": float(np.mean([q[arm]["argmax_agree"] for q in qs])),
                "flips": flips,
                "max_abs_dp_vs_native": float(max(q[arm]["max_abs_dp"] for q in qs)),
                "mean_of_question_max_abs_dp": float(np.mean([q[arm]["max_abs_dp"] for q in qs])),
                "mean_abs_dp_vs_native": float(np.mean([q[arm]["mean_abs_dp"] for q in qs])),
                "gold_questions": len(gold), "gold_correct": sum(q[arm]["argmax"] == q["gold"] for q in gold),
                "mean_image_tokens": float(np.mean([r["arms"][arm]["n_image_tokens"] for r in per_record])),
                "mean_tokens": float(np.mean([r["arms"][arm]["tokens"] for r in per_record]))}

    what_default = ("native (processor grid) vs g256 / g448 (PIL BICUBIC to 256x256 / 448x448 first) on every image record of the "
                    "fixture, CPU fp32 oracle (the checkpoint's joint_schema_model.py); dp = per-option probability difference")
    what = what_default if arms == ARMS else (
        f"native (processor grid) vs {' / '.join(arms[1:])} (PIL BICUBIC to the square tile first) on every image record of "
        "the fixture, CPU fp32 oracle (the checkpoint's joint_schema_model.py); dp = per-option probability difference")
    out = {
        "schema": "clef-flash-grid-price/1",
        "what": what,
        "oracle": str(path), "oracle_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **({"oracle_large": str(sources[1]), "oracle_large_sha256": hashlib.sha256(sources[1].read_bytes()).hexdigest()}
           if len(sources) > 1 else {}),
        "image_records": len(per_record), "questions": len(per_question),
        "arms": {arm: agg(arm) for arm in arms},
        "records": per_record, "per_question": per_question,
    }
    dst = Path(args.out).expanduser()
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    os.replace(tmp, dst)
    for arm in arms:
        a = out["arms"][arm]
        if arm == "native":
            print(f"native  q={a['questions']} gold {a['gold_correct']}/{a['gold_questions']} img_tokens {a['mean_image_tokens']:.0f}")
        else:
            print(f"{arm}    agree {a['argmax_agree_with_native']}/{a['questions']} max|dp| {a['max_abs_dp_vs_native']:.4f} "
                  f"mean max|dp| {a['mean_of_question_max_abs_dp']:.4f} mean|dp| {a['mean_abs_dp_vs_native']:.4f} "
                  f"gold {a['gold_correct']}/{a['gold_questions']} img_tokens {a['mean_image_tokens']:.0f} flips {a['flips']}")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
