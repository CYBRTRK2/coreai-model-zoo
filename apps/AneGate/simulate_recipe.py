"""fp32 simulation of an iOS k-means palettization recipe, judged by the fixture's teacher-forced rule (S5).

Splits "the recipe itself is lossy" from "the ANE path executes it wrong": the same yaml the exporter uses
(`kmeans_palettization_config`: global n_bits/granularity + module_name_configs regexes on the iOS module
names, `extend.model.layers.N.self_attn.q_proj` …) is applied to the HF fp32 weights with coreai-opt's OWN
palettizer (`_KMeansFakePalettize` — the exporter's optimal 1-D k-means, same values on coreai-opt 0.2.1 and 0.3.0; a home-made Lloyd
k-means was 50 % worse on these heavy-tailed tensors and useless as a proxy), then every fixture prompt is run
teacher-forced: argmax at each step vs the oracle token, a mismatch on a margin-clear step = FAIL (knife-edge
steps excluded), exactly as AneGateRunner judges the device. Embeddings stay fp32 here (the device has them
int8; S1 showed that is not where the 4-bit flips come from); the LUT is not rounded to fp16.

    # in the coreai-models-rebase venv (coreai-opt + transformers):
    ~/code/coreai/coreai-models-rebase/.venv/bin/python simulate_recipe.py --fixture fixtures/minicpm5_2b/fixture.json \
        --yaml minicpm5_mixed_4bit_8bit_g32.yaml [--yaml …] [--preset 4bit_weight_palettized_group32] --out sim.json
"""
import argparse, json, os, re, time, warnings
os.environ.setdefault("HF_HUB_OFFLINE", "1"); warnings.filterwarnings("ignore")
import torch, yaml
logging = __import__("logging"); logging.getLogger("coreai_opt").setLevel(logging.ERROR)
from transformers import AutoModelForCausalLM

PRESETS = {
    "4bit_weight_palettized_group32": {"n_bits": 4, "granularity": {"type": "per_grouped_channel", "axis": 0, "group_size": 32}},
    "4bit_weight_palettized_group8": {"n_bits": 4, "granularity": {"type": "per_grouped_channel", "axis": 0, "group_size": 8}},
}


def palettize(w: torch.Tensor, spec: dict) -> torch.Tensor:
    """coreai-opt's k-means palettization of one 2-D weight, exactly as the iOS exporter runs it."""
    from coreai_opt.palettization.kmeans.kmeans_fake_palettize import _KMeansFakePalettize
    from coreai_opt.palettization.spec.granularity import PerGroupedChannelGranularity, PerTensorGranularity
    g = spec["granularity"]
    if g["type"] == "per_tensor":
        gran = PerTensorGranularity()
    elif g["type"] == "per_grouped_channel":
        gran = PerGroupedChannelGranularity(axis=g.get("axis", 0), group_size=int(g["group_size"]))
    else:
        raise ValueError(g)
    fp = _KMeansFakePalettize(n_bits=int(spec["n_bits"]), lut_qspec=None, granularity=gran, cluster_dim=1,
                              enable_per_channel_scale=False)
    if hasattr(fp, "_calculate_centroids"):  # coreai-opt 0.2.x
        lut, idx = fp._calculate_centroids(w)
        return fp._palettize(lut, idx, w).to(w.dtype)
    # coreai-opt 0.3.0: centroids are clustered on first use; forward_enabled is the hard-assigned
    # reconstruction and is bit-identical to the 0.2.1 path (checked on MiniCPM5-2B k_proj, 4 recipes, 2026-09-27).
    fp.ensure_initialized(w)
    return fp.forward_enabled(w).to(w.dtype)


def int8_block32(w: torch.Tensor) -> torch.Tensor:
    """Symmetric int8 weight-only quantization, one scale per 32-wide block along the input dim, no clipping —
    the shipped GPU bundle's recipe (conversion/minicpm5_int8sym_b32.yaml), simulated in fp32."""
    out_ch, in_ch = w.shape
    x = w.float().reshape(out_ch, in_ch // 32, 32)
    scale = x.abs().amax(dim=2, keepdim=True) / 127.0
    scale[scale == 0] = 1.0
    return (torch.round(x / scale).clamp(-127, 127) * scale).reshape(out_ch, in_ch).to(w.dtype)


def load_recipe(path: str | None, preset: str | None):
    if preset == "int8_block32_linear":
        return {"name": preset, "global": {"linear_int8_block32": True}, "by_name": []}
    if preset:
        return {"name": preset, "global": PRESETS[preset], "by_name": []}
    d = yaml.safe_load(open(path))["kmeans_palettization_config"]
    glob = d["global_config"]["op_state_spec"]["weight"]
    by_name = [(re.compile(pat), cfg["op_state_spec"]["weight"] if cfg else None)  # a `null` module config = not palettized (kept fp16)
               for pat, cfg in (d.get("module_name_configs") or {}).items()]
    return {"name": os.path.basename(path), "global": glob, "by_name": by_name}


def spec_for(recipe, ios_name: str):
    for pat, spec in recipe["by_name"]:
        if pat.fullmatch(ios_name):
            return spec
    return recipe["global"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="openbmb/MiniCPM5-2B"); ap.add_argument("--fixture", required=True)
    ap.add_argument("--yaml", action="append", default=[]); ap.add_argument("--preset", action="append", default=[])
    ap.add_argument("--prompts", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fx = json.load(open(a.fixture)); floor = fx["floor"]
    prompts = [p for p in fx["prompts"] if not a.prompts or p["name"] in a.prompts.split(",")]
    model = AutoModelForCausalLM.from_pretrained(a.hf_id, dtype=torch.float32).eval()
    # the palettized modules: every Linear except the embedding (iOS names = "extend." + HF name)
    targets = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    saved = {n: m.weight.data.clone() for n, m in targets.items()}
    print(f"{len(targets)} linear modules (incl. lm_head); prompts={[p['name'] for p in prompts]}")

    @torch.no_grad()
    def judge():
        res = {}
        for p in prompts:
            ids = torch.tensor([p["prompt_ids"] + p["expected_ids"]]); P = len(p["prompt_ids"]); n = len(p["expected_ids"])
            lg = model(ids).logits[0, P - 1:P - 1 + n].float(); top = lg.argmax(-1)
            exp = torch.tensor(p["expected_ids"]); m = torch.tensor(p["margins"])
            fails = [(i, round(float(m[i]), 3), int(exp[i]), int(top[i])) for i in range(n) if top[i] != exp[i] and m[i] >= floor]
            knife = [i for i in range(n) if top[i] != exp[i] and m[i] < floor]
            res[p["name"]] = {"tf_ok": int((top == exp).sum()), "n": n, "fail_steps": fails, "knife_steps": knife,
                              "first_fail": fails[0][0] if fails else None, "verdict": "PASS" if not fails else "FAIL"}
            print(f"  {p['name']:8s} tf={res[p['name']]['tf_ok']}/{n} fails={len(fails)} first={res[p['name']]['first_fail']} "
                  f"knife={len(knife)} -> {res[p['name']]['verdict']}  {fails[:4]}", flush=True)
        return res

    out = {"hf_id": a.hf_id, "fixture": a.fixture, "floor": floor, "recipes": {}}
    print("baseline (fp32):"); out["baseline"] = judge()
    recipes = [load_recipe(None, p) for p in a.preset] + [load_recipe(y, None) for y in a.yaml]
    for r in recipes:
        t0 = time.time(); n8 = 0; n_other = 0
        for n, m in targets.items():
            spec = spec_for(r, "extend." + n)
            if spec.get("linear_int8_block32"):
                m.weight.data = int8_block32(saved[n]); n8 += 1; continue
            m.weight.data = palettize(saved[n], spec)
            if int(spec["n_bits"]) == 8: n8 += 1
            else: n_other += 1
        print(f"recipe {r['name']}: palettized {len(targets)} modules ({n8} at 8-bit) in {time.time() - t0:.0f} s", flush=True)
        out["recipes"][r["name"]] = judge()
        for n, m in targets.items():
            m.weight.data = saved[n]
        json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
