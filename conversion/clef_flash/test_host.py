#!/usr/bin/env python3
"""Test: `host.py` reproduces every oracle run — ids, spans, option order, static inputs, M-RoPE, pixels.

For every run in `oracle/records_oracle.json` (214) and, when present, `oracle/records_oracle_large.json`
(the g672 / g896 runs), rebuilt from `fixtures/records.json` with each tokenizer under test:

  * ids        `host.build_ids` == the oracle's input_ids (the ids `systemone()` fed the model), token for token;
  * offset     token_offset == the oracle's (image runs: the <|vision_start|> index, 36);
  * questions  per question: id, type, question span, option spans, option ids in order == the oracle's
               EncodedRecord (the spans and the option order the author's head read);
  * static     `host.static_inputs` == `qwen3_5_vl_pipelined.host_static_inputs` (the module's host function,
               torch) on the same ids: mapped ids, image_rc, rope_shift_start, rope_shift_amount;
  * rope       `host.rope_positions` on the mapped ids == the planes the oracle's text rotary received;
  * the two tokenizers give the same ids.

Grid per arm: text -> none; g256 / g448 / g672 / g896 -> 8 / 14 / 21 / 28 square; native -> the merged grid the
processor picked (the oracle's merged_hw). Negative control: one word of one question's instructions changed
must turn the run red. Pixels: `host.preprocess(image, tile)` vs the oracle's `pixel_values` on every fixed-grid
run (Pillow resize = the gated path; the NumPy resize in Pillow's pass order recorded in 0-255 levels).

Run from the worktree (shared venv, offline):
    PYTHONDONTWRITEBYTECODE=1 HF_HOME=~/code/coreai/_clefflash/hf HF_HUB_OFFLINE=1 \\
        ../../../coreai-models/.venv/bin/python test_host.py      # -> _clefflash/results/test_host.json
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import host  # noqa: E402
from _paths import hf_snapshot, work_path  # noqa: E402

HF_ID = "Cloudflare/clef-flash"
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
LANE = work_path("_clefflash")
TILES = {"g256": 256, "g448": 448, "g672": 672, "g896": 896}
LEVEL = 1.0 / (255.0 * host.IMAGE_STD)          # one 0-255 level after rescale + normalize = 2/255


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenizers_under_test(snapshot: Path) -> tuple[dict, dict]:
    import tokenizers

    toks = {f"tokenizers {tokenizers.__version__} Tokenizer.from_file(tokenizer.json)":
            tokenizers.Tokenizer.from_file(str(snapshot / "tokenizer.json"))}
    errors = {}
    try:
        import transformers
        from transformers import AutoTokenizer
        toks[f"transformers {transformers.__version__} AutoTokenizer"] = AutoTokenizer.from_pretrained(str(snapshot))
    except Exception as e:  # noqa: BLE001
        errors["transformers AutoTokenizer"] = f"{type(e).__name__}: {e}"[:400]
    return toks, errors


def grid_of(run: dict):
    arm = run["arm"]
    if arm == "text":
        return None
    if arm in TILES:
        g = host.tile_grid(TILES[arm])
        return (g, g)
    return tuple(run["merged_hw"])


def oracle_runs() -> tuple[list[dict], dict]:
    docs = {}
    runs = []
    for name in ("records_oracle.json", "records_oracle_large.json"):
        p = LANE / "oracle" / name
        if not p.exists():
            continue
        doc = json.loads(p.read_text())
        docs[name] = {"sha256": sha256_file(p), "runs": len(doc["rows"]), "complete": doc.get("complete")}
        runs += [dict(r, _file=name) for r in doc["rows"]]
    return runs, docs


def check_run(run: dict, rec: dict, tok, hsi) -> tuple[list[str], dict]:
    """Mismatch messages for one oracle run (empty = pass) and the host's output."""
    import torch

    grid = grid_of(run)
    # photo_01 native (40 x 30 = 1200) exceeds the shipped 1024-row buffer; the shipped path sends a fixed grid
    nmax = max(host.N_IMAGE_MAX, grid[0] * grid[1] if grid else 0)
    out = host.build_ids(rec["request"], tok, grid, n_image_max=nmax)
    bad = []
    if out["ids"] != run["ids"]:
        i = next((k for k, (a, b) in enumerate(zip(out["ids"], run["ids"])) if a != b),
                 min(len(out["ids"]), len(run["ids"])))
        bad.append(f"ids differ at {i} (len {len(out['ids'])} vs {len(run['ids'])})")
    if grid is not None and out["token_offset"] != run["token_offset"]:
        bad.append(f"token_offset {out['token_offset']} vs {run['token_offset']}")
    mine = [(q["question_id"], q["type"], q["question_span"], q["option_spans"], q["option_ids"])
            for q in out["questions"]]
    want = [(q["question_id"], q["type"], q["question_span"], q["option_spans"], q["option_ids"])
            for q in run["questions"]]
    if mine != want:
        k = next((i for i, (a, b) in enumerate(zip(mine, want)) if a != b), min(len(mine), len(want)))
        bad.append(f"question {k} differs: {mine[k] if k < len(mine) else None} vs {want[k] if k < len(want) else None}")
    st = out["static"]
    ids_t, rc_t, start_t, amount_t = hsi(out["ids"], grid, host.VOCAB, host.IMAGE_PAD, host.VISION_START, nmax)
    for name, a, b in (("input_ids", st["input_ids"], ids_t), ("image_rc", st["image_rc"], rc_t),
                       ("rope_shift_start", st["rope_shift_start"], start_t),
                       ("rope_shift_amount", st["rope_shift_amount"], amount_t)):
        if not (a.dtype == np.int32 and b.dtype == torch.int32 and np.array_equal(a, b.numpy())):
            bad.append(f"static {name} differs from host_static_inputs")
    z = np.load(LANE / "oracle" / "npz" / f"{run['id']}__{run['arm']}.npz")
    planes = host.rope_positions(st["input_ids"], int(st["rope_shift_start"][0]), int(st["rope_shift_amount"][0]),
                                 grid[1] if grid else 1)
    if planes.shape != z["rope_pos"].shape or not np.array_equal(planes, z["rope_pos"]):
        bad.append("rope planes differ from the oracle's rotary input")
    if grid is not None and int(st["rope_shift_amount"][0]) != run["rope_shift_amount"]:
        bad.append(f"amount {int(st['rope_shift_amount'][0])} vs {run['rope_shift_amount']}")
    if not np.array_equal(z["input_ids"], np.asarray(run["ids"])):
        bad.append("npz input_ids differ from the oracle json ids")
    return bad, out


def negative_control(runs: list[dict], recs: dict, tok, hsi) -> dict:
    """One word of the first question's instructions changed: the run must go red (ids and span)."""
    run = next(r for r in runs if r["id"] == "own_t01" and r["arm"] == "text")
    rec = copy.deepcopy(recs[run["id"]])
    qid = next(iter(rec["request"]["questions"]))
    q = rec["request"]["questions"][qid]
    words = q["instructions"].split(" ")
    words[1] = "zqxvortmund"                        # "Which team ..." -> "Which zqxvortmund ...": more tokens
    q["instructions"] = " ".join(words)
    bad, out = check_run(run, rec, tok, hsi)
    q0 = out["questions"][0]["question_span"]
    q0_oracle = run["questions"][0]["question_span"]
    return {"run": f"{run['id']}:{run['arm']}", "question": qid, "changed_instructions": q["instructions"],
            "red": bool(bad) and q0 != q0_oracle, "span_changed": q0 != q0_oracle, "messages": bad[:4],
            "question_span_host": q0, "question_span_oracle": q0_oracle}


def pixel_checks(runs: list[dict], recs: dict) -> dict:
    out = {}
    for run in runs:
        if run["arm"] not in TILES:
            continue
        name = recs[run["id"]]["image_files"][0]
        path = LANE / "fixtures" / "images" / name
        z = np.load(LANE / "oracle" / "npz" / f"{run['id']}__{run['arm']}.npz")
        want = z["pixel_values"]
        res = {}
        for mode in ("pil", "numpy"):
            got = host.preprocess(path, TILES[run["arm"]], resize=mode)
            if got.shape != want.shape:
                res[mode] = {"shape": [list(got.shape), list(want.shape)]}
                continue
            d = np.abs(got.astype(np.float64) - want.astype(np.float64))
            res[mode] = {"max_abs": float(d.max()), "max_levels": float(d.max() / LEVEL),
                         "n_diff": int((d > 0).sum()), "n_diff_gt_half_level": int((d > 0.5 * LEVEL).sum()),
                         "n": int(d.size)}
        out[f"{run['id']}:{run['arm']}"] = {"image": name, "tile": TILES[run["arm"]], **res}
    summ = {}
    for mode in ("pil", "numpy"):
        vals = [(k, v[mode]) for k, v in out.items() if "max_abs" in v.get(mode, {})]
        if vals:
            k, w = max(vals, key=lambda kv: kv[1]["max_abs"])
            summ[mode] = {"runs": len(vals), "max_abs": w["max_abs"], "max_levels": w["max_levels"], "worst": k,
                          "runs_bit_equal": sum(v["max_abs"] == 0.0 for _, v in vals),
                          "runs_with_diff_gt_half_level": sum(v["n_diff_gt_half_level"] > 0 for _, v in vals)}
    return {"summary": summ, "runs": out}


def main() -> int:
    from qwen3_5_clef_decoder import host_static_inputs

    snapshot = Path(hf_snapshot(HF_ID, revision=REVISION))
    fx_path = LANE / "fixtures" / "records.json"
    recs = {r["id"]: r for r in json.loads(fx_path.read_text())["records"]}
    runs, docs = oracle_runs()
    toks, tok_errors = tokenizers_under_test(snapshot)
    report = {"schema": "clef-flash-host-test/1",
              "what": "host.build_ids / static_inputs / rope_positions / preprocess vs every oracle run",
              "oracle": docs, "fixtures_records_json_sha256": sha256_file(fx_path),
              "tokenizer_json_sha256": sha256_file(snapshot / "tokenizer.json"),
              "host_py_sha256": sha256_file(HERE / "host.py"), "runs": len(runs),
              "arms": {a: sum(r["arm"] == a for r in runs) for a in sorted({r["arm"] for r in runs})},
              "tokenizers": {}, "tokenizer_errors": tok_errors}
    first_ids: dict = {}
    for name, tok in toks.items():
        fails = []
        for run in runs:
            bad, out = check_run(run, recs[run["id"]], tok, host_static_inputs)
            key = (run["id"], run["arm"])
            if key in first_ids and first_ids[key] != out["ids"]:
                bad.append("ids differ between tokenizers")
            first_ids.setdefault(key, out["ids"])
            if bad:
                fails.append(f"{run['id']}:{run['arm']}: {'; '.join(bad)}")
        report["tokenizers"][name] = {"pass": len(runs) - len(fails), "of": len(runs), "fail": fails}
        print(f"{name}: {len(runs) - len(fails)}/{len(runs)} runs match (ids, offset, spans, options, static, rope)")
        for f in fails[:8]:
            print("   ", f)
    first_tok = next(iter(toks.values()))
    report["negative_control"] = negative_control(runs, recs, first_tok, host_static_inputs)
    print("negative control:", json.dumps({k: report["negative_control"][k] for k in ("run", "red", "messages")}))
    report["pixel_values"] = pixel_checks(runs, recs)
    print("pixel_values:", json.dumps(report["pixel_values"]["summary"]))
    report["pass"] = (all(not v["fail"] for v in report["tokenizers"].values()) and report["negative_control"]["red"])
    report["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = LANE / "results" / "test_host.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out}\n{'PASS' if report['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
