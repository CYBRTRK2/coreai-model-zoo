#!/usr/bin/env python3
"""clef-flash vision tower at fixed grids vs the author's fp32 tower output.

The Core AI tower is the shipped `qwen3_5_vision.Qwen3_5VisionEncoder` (27 blocks, hidden 1152, merger
out 4096, no DeepStack) baked at ONE merged grid per graph, exported by
`conversion/export_qwen38vl_pipelined.py --skip-decoder --vision-dtype fp16w32`:

    g256 = 256 px tile -> 16 x 16 patches -> 8 x 8 merged   =  64 rows
    g448 = 448 px tile -> 28 x 28 patches -> 14 x 14 merged = 196 rows
    g672 = 672 px tile -> 42 x 42 patches -> 21 x 21 merged = 441 rows
    g896 = 896 px tile -> 56 x 56 patches -> 28 x 28 merged = 784 rows

The target is the oracle (`oracle_clef.py`): per (record, fixed-grid arm), the processor's `pixel_values`
[4 G^2, 1536] and the author's fp32 tower merger output `image_embeds` [G^2, 4096], captured inside
`systemone()`. The tower input here is the HOST's patches (`host.preprocess`, Pillow BICUBIC to the tile),
asserted bit-equal to the oracle's pixel_values first, so the gate measures the shipped host path.

  torch  the authored tower in fp32, eager CPU (`--stages torch` -> results/tower_torch.json)
         bar: image cos >= 0.9999 and min row cos >= 0.999 on every run
  aot    the fp16w32 `.aimodel` AOT-compiled for the Mac GPU (h16c), loaded by the Python runtime with
         `SpecializationOptions.default()` (`--stages aot` -> results/tower_aot.json); same bar; encode ms
         are CONTENDED reference values (shared GPU, no lock)

Negative control (both stages): the same pixels in raster patch order instead of merge-block-major must
miss the bar. Cosines are float64.

    PYTHONDONTWRITEBYTECODE=1 HF_HOME=~/code/coreai/_clefflash/hf HF_HUB_OFFLINE=1 \\
        ../../../coreai-models/.venv/bin/python gate_tower.py --stages torch
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import platform
import sys
import time
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
BAR = {"image_cos": 0.9999, "min_row_cos": 0.999}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cos_stats(got: np.ndarray, want: np.ndarray) -> dict:
    g = np.asarray(got, np.float64)
    w = np.asarray(want, np.float64)
    c = float(g.ravel() @ w.ravel() / (np.linalg.norm(g) * np.linalg.norm(w)))
    rows = (g * w).sum(-1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(w, axis=-1))
    i = int(rows.argmin())
    d = np.abs(g - w)
    return {"image_cos": c, "min_row_cos": float(rows[i]), "min_row_index": i, "max_abs": float(d.max()),
            "mean_abs": float(d.mean()), "ref_absmax": float(np.abs(w).max()), "finite": bool(np.isfinite(g).all())}


def passes(s: dict) -> bool:
    return s["image_cos"] >= BAR["image_cos"] and s["min_row_cos"] >= BAR["min_row_cos"]


def raster_perm(grid: int) -> np.ndarray:
    """raster-ordered patches = block_major_patches[perm] (the negative control)."""
    gp = grid * host.MERGE
    r, c = np.divmod(np.arange(gp * gp), gp)
    m = host.MERGE
    return (((r // m) * grid + c // m) * m + r % m) * m + c % m


def load_runs(arms: list[str]) -> tuple[dict, dict]:
    """{arm: [{id, image, patches (host), pixel_values, image_embeds}]} from the oracle files."""
    recs = {r["id"]: r for r in json.loads((LANE / "fixtures" / "records.json").read_text())["records"]}
    out, prov = {}, {}
    for name in ("records_oracle.json", "records_oracle_large.json"):
        p = LANE / "oracle" / name
        if not p.exists():
            continue
        doc = json.loads(p.read_text())
        prov[name] = {"sha256": sha256_file(p), "complete": doc.get("complete"), "runs": len(doc["rows"])}
        for r in doc["rows"]:
            if r["arm"] not in arms:
                continue
            g = host.tile_grid(TILES[r["arm"]])
            z = np.load(LANE / "oracle" / "npz" / f"{r['id']}__{r['arm']}.npz")
            assert z["image_grid_thw"].tolist() == [[1, 2 * g, 2 * g]], (r["id"], r["arm"])
            img = LANE / "fixtures" / "images" / recs[r["id"]]["image_files"][0]
            patches = host.preprocess(img, TILES[r["arm"]])
            out.setdefault(r["arm"], []).append({
                "id": r["id"], "image": img.name, "patches": patches, "pixel_values": z["pixel_values"],
                "image_embeds": z["image_embeds"],
                "host_patches_equal_oracle": bool(np.array_equal(patches, z["pixel_values"]))})
    return out, prov


def stage_torch(runs: dict, args) -> dict:
    import torch

    from coreai_models.models.macos.qwen3_5_vision import Qwen3_5VisionEncoder

    torch.set_num_threads(args.threads)
    res = {"device": "cpu", "dtype": "float32", "threads": args.threads, "arms": {}}
    for arm, items in runs.items():
        g = host.tile_grid(TILES[arm])
        t0 = time.perf_counter()
        vis = Qwen3_5VisionEncoder.from_hf(HF_ID, target_dtype=torch.float32, grid_h=g, grid_w=g).eval()
        load_s = time.perf_counter() - t0
        perm = raster_perm(g)
        rows, walls = [], []
        for it in items:
            t0 = time.perf_counter()
            with torch.inference_mode():
                got = vis(torch.from_numpy(it["patches"])).numpy()
            walls.append(time.perf_counter() - t0)
            s = cos_stats(got, it["image_embeds"])
            s["pass"] = passes(s)
            with torch.inference_mode():
                ctrl = vis(torch.from_numpy(np.ascontiguousarray(it["patches"][perm]))).numpy()
            cs = cos_stats(ctrl, it["image_embeds"])
            rows.append({"id": it["id"], "image": it["image"], "host_patches_equal_oracle": it["host_patches_equal_oracle"],
                         **s, "neg_raster_order": {"image_cos": cs["image_cos"], "min_row_cos": cs["min_row_cos"],
                                                   "red": not passes(cs)}})
        del vis
        res["arms"][arm] = summarize(arm, rows, {"load_s": load_s, "encode_s_median": float(np.median(walls))})
        a = res["arms"][arm]
        print(f"torch {arm}: runs {a['runs']} pass {a['n_pass']} min image cos {a['min_image_cos']['value']:.8f} "
              f"min row cos {a['min_row_cos']['value']:.8f} max|d| {a['max_abs']['value']:.3e} "
              f"neg all red {a['neg_all_red']}", flush=True)
    return res


def summarize(arm: str, rows: list[dict], timing: dict) -> dict:
    def worst(key, fn=min):
        r = fn(rows, key=lambda x: x[key])
        return {"value": r[key], "id": r["id"]}
    g = host.tile_grid(TILES[arm])
    return {"grid": [g, g], "tile": TILES[arm], "rows_per_image": g * g, "runs": len(rows),
            "n_pass": sum(r["pass"] for r in rows), "pass": all(r["pass"] for r in rows),
            "host_patches_equal_oracle": sum(r["host_patches_equal_oracle"] for r in rows),
            "min_image_cos": worst("image_cos"), "min_row_cos": worst("min_row_cos"),
            "max_abs": worst("max_abs", max), "neg_all_red": all(r["neg_raster_order"]["red"] for r in rows),
            "neg_max_image_cos": max(r["neg_raster_order"]["image_cos"] for r in rows),
            "timing": timing, "per_run": rows}


async def _maybe(x):
    return await x if inspect.isawaitable(x) else x


def asset_digest(path: Path) -> dict:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    per = {str(p.relative_to(path)): {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files}
    return {"path": str(path), "bytes": sum(v["bytes"] for v in per.values()), "files": per}


def stage_aot(runs: dict, args) -> dict:
    import coreai.runtime as rt

    res = {"runtime": "coreai.runtime (Python), AIModel.load(.aimodelc, SpecializationOptions.default())",
           "contended": "shared GPU, no lock: encode ms are reference values", "arms": {}}

    async def go(arm: str, items: list[dict]) -> dict:
        stem = f"clef_flash_{arm}_vision_fp16w32"
        aimodel = Path(args.exports) / stem / f"{stem}.aimodel"
        cands = sorted((Path(args.exports) / f"{stem}_aotc").glob("*.h16c.aimodelc"))
        assert len(cands) == 1, cands
        aimodelc = cands[0]
        info = {"aimodel": asset_digest(aimodel) if aimodel.exists() else None, "aimodelc": asset_digest(aimodelc)}
        t0 = time.perf_counter()
        m = await _maybe(rt.AIModel.load(str(aimodelc), rt.SpecializationOptions.default()))
        fn = await _maybe(m.load_function(m.function_names[0]))
        load_s = time.perf_counter() - t0
        info["function"] = {"name": fn.desc.name, "inputs": list(fn.desc.input_names),
                            "outputs": list(fn.desc.output_names)}

        async def encode(pt):
            out = await _maybe(fn(inputs={"patches": rt.NDArray(np.ascontiguousarray(pt.astype(np.float32)))}))
            return np.asarray(out["image_embeds"].numpy()).astype(np.float32)

        g = host.tile_grid(TILES[arm])
        perm = raster_perm(g)
        t0 = time.perf_counter()
        first = await encode(items[0]["patches"])
        first_call_s = time.perf_counter() - t0
        again = await encode(items[0]["patches"])
        rows = []
        for it in items:
            got = await encode(it["patches"])
            s = cos_stats(got, it["image_embeds"])
            s["pass"] = passes(s)
            ctrl = await encode(np.ascontiguousarray(it["patches"][perm]))
            cs = cos_stats(ctrl, it["image_embeds"])
            rows.append({"id": it["id"], "image": it["image"], "host_patches_equal_oracle": it["host_patches_equal_oracle"],
                         **s, "neg_raster_order": {"image_cos": cs["image_cos"], "min_row_cos": cs["min_row_cos"],
                                                   "red": not passes(cs)}})
        ms = []
        for _ in range(args.bench):
            t0 = time.perf_counter()
            await encode(items[0]["patches"])
            ms.append((time.perf_counter() - t0) * 1e3)
        timing = {"load_s": load_s, "first_call_s": first_call_s, "repeat_bit_identical": bool(np.array_equal(first, again)),
                  "encode_ms": {"n": len(ms), "median": float(np.median(ms)), "min": min(ms), "max": max(ms)},
                  "loadavg": list(os.getloadavg())}
        out = summarize(arm, rows, timing)
        out["assets"] = info
        del fn, m
        return out

    for arm, items in runs.items():
        res["arms"][arm] = asyncio.run(go(arm, items))
        a = res["arms"][arm]
        print(f"aot {arm}: runs {a['runs']} pass {a['n_pass']} min image cos {a['min_image_cos']['value']:.6f} "
              f"min row cos {a['min_row_cos']['value']:.6f} max|d| {a['max_abs']['value']:.3e} "
              f"encode median {a['timing']['encode_ms']['median']:.1f} ms neg all red {a['neg_all_red']}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stages", default="torch", choices=["torch", "aot"])
    ap.add_argument("--arms", default="g256,g448")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--exports", default=str(LANE / "exports"))
    ap.add_argument("--bench", type=int, default=10)
    args = ap.parse_args()
    import PIL
    import torch

    runs, prov = load_runs(args.arms.split(","))
    record = {"schema": f"clef-flash-tower-{args.stages}/1", "source": {"hf_id": HF_ID, "revision": REVISION},
              "bar": BAR, "oracle": prov, "host_py_sha256": sha256_file(HERE / "host.py"),
              "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pillow": PIL.__version__,
                           "torch": torch.__version__},
              "platform": {"machine": platform.machine(), "macos": platform.mac_ver()[0]}}
    try:
        import importlib.metadata as md
        for p in ("coreai-core", "coreai-torch", "coreai-models"):
            record["versions"][p] = md.version(p)
    except Exception as e:  # noqa: BLE001
        record["versions"]["error"] = repr(e)
    snap = Path(hf_snapshot(HF_ID, revision=REVISION))
    record["snapshot"] = str(snap)
    record.update(stage_torch(runs, args) if args.stages == "torch" else stage_aot(runs, args))
    record["pass"] = all(a["pass"] and a["neg_all_red"] for a in record["arms"].values())
    record["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = LANE / "results" / f"tower_{args.stages}.json"
    if out.exists() and args.arms != "g256,g448":
        prev = json.loads(out.read_text())
        prev["arms"].update(record["arms"])
        prev["pass"] = all(a["pass"] and a["neg_all_red"] for a in prev["arms"].values())
        prev.setdefault("updates", []).append({"arms": args.arms, "generated_at": record["generated_at"]})
        record = prev
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {out}: {'PASS' if record['pass'] else 'FAIL'}")
    return 0 if record["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
