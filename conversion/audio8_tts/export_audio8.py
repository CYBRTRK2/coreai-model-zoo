#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""Export Audio8-TTS-Preview-0.6b to Core AI: slow AR (prefill + decode), fast AR (first + step), codec decoder.

    slow  audio8_slow_ar_<mode>_cl<CL>_w<W>.aimodel   functions prefill(codes [1,11,W] i32, pos [1] i32) / decode(codes [1,11,1], pos)
                                                      -> logits [S,4097] f16, hidden [S,896] f16; state k_cache/v_cache [24,1,2,CL,64] f16
    fast  audio8_fast_ar_<mode>.aimodel               functions first(hidden [1,1,896] f16, code0 [1] i32) / step(code [1] i32, pos [1] i32)
                                                      -> logits [1,4096] f16; state k_cache/v_cache [4,1,2,10,64] f16
    codec audio8_codec_decoder_fp16_t<T>.aimodel      main(codes [1,10,T] i32) -> wav [1, T*2048] f16

`--mode int8` = weight-only int8, symmetric-with-clipping, per-block-32 along the input axis on every Linear of the
slow and fast transformers (the zoo's LLM recipe); embeddings, the 4097-row head, `fast_output`, norms stay fp16.
`--mode fp16` is the uncompressed control. The codec decoder ships fp16 (its residual maxima are ~700).

Each part is gated right after saving: the `.aimodel` is loaded through `coreai.runtime` on the GPU and replayed
teacher-forced against the fp32 oracle on `--gate-fixture` (a few frames), so a lowering that returns wrong numbers
is caught here, not in the full gate.

Run (shared venv):
    ~/code/coreai/coreai-models/.venv/bin/python conversion/audio8_tts/export_audio8.py --part all --mode int8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from _paths import hf_snapshot, work_path  # noqa: E402

import audio8_model as M  # noqa: E402
from replay import codec_gate, summarize, teacher_forced  # noqa: E402

HF_ID = "Edge0/Audio8-TTS-Preview-0.6b"
REVISION = "f07040f3d151f1ba0253bfb92cb2f5dd38b44594"
WORK = work_path("_audio8_tts")
EXPORTS = WORK / "exports"
DT = torch.float16


def du(path: Path) -> str:
    return subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True).stdout.split()[0]


def quant_cfg(dtype: str, exclude_names: tuple[str, ...]) -> dict:
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {"weight": {"dtype": dtype, "qscheme": "symmetric_with_clipping",
                                         "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}},
            "op_input_spec": None, "op_output_spec": None,
        },
        "module_type_configs": {
            "audio8_model.RMSNorm": None,
            "torch.nn.modules.sparse.Embedding": None,
        },
        "module_name_configs": {name: None for name in exclude_names},
    }


def make_export_fn(ref: dict):
    import coreai_torch
    from coreai_models.export.mlir_ops import remove_functionalization

    def export_fn(m):
        with torch.no_grad():
            ep = torch.export.export(m, args=(), kwargs=ref, dynamic_shapes=None)
        ep = ep.run_decompositions(coreai_torch.get_decomp_table())
        remove_functionalization(ep)
        return ep

    return export_fn


def save(prog, path: Path) -> None:
    import coreai.runtime as rt

    shutil.rmtree(path, ignore_errors=True)
    meta = rt.AIModelAssetMetadata()
    meta.license = "apache-2.0"
    prog.save_asset(path, meta)
    print(f"[save] {path.name} ({du(path)})", flush=True)


def convert(entrypoints: list[dict]):
    """entrypoints: [{module, ref, input_names, output_names, state_names, name}] -> AIProgram (optimized)."""
    import coreai_torch
    from coreai_models.export.mlir_ops import register_custom_torch_lowering

    conv = coreai_torch.TorchConverter()
    for e in entrypoints:
        conv.add_pytorch_module(e["module"], export_fn=make_export_fn(e["ref"]), externalize_modules=[],
                                input_names=e["input_names"], output_names=e["output_names"],
                                state_names=e.get("state_names"), entrypoint_name=e["name"])
    register_custom_torch_lowering(conv)
    prog = conv.to_coreai()
    prog.optimize()
    return prog


def maybe_quantize(core, ref_tuple, mode: str, exclude: tuple[str, ...]):
    if mode == "fp16":
        return core
    from coreai_models.export.compression import quantize_pytorch_model

    names = ["codes", "pos", "k_cache", "v_cache"] if len(ref_tuple) == 4 else ["x", "pos", "k_cache", "v_cache"]
    q = quantize_pytorch_model(core, ref_tuple, {n: None for n in names}, quant_cfg(mode, exclude))
    return q


# ----------------------------------------------------------------------------- engine ports
class EnginePorts:
    def __init__(self, slow_path: Path | None, fast_path: Path | None, codec_path: Path | None, cl: int, unit: str = "gpu"):
        import coreai.runtime as rt

        self.rt = rt
        self.cl = cl
        self.opts = (rt.SpecializationOptions.cpu_only() if unit == "cpu"
                     else rt.SpecializationOptions.from_preferred_compute_unit_kind(getattr(rt.ComputeUnitKind, unit)()))
        self.models = {}
        self.fn = {}
        if slow_path:
            m = asyncio.run(rt.AIModel.load(str(slow_path), self.opts)); self.models["slow"] = m
            self.fn["prefill"], self.fn["decode"] = m.load_function("prefill"), m.load_function("decode")
        if fast_path:
            m = asyncio.run(rt.AIModel.load(str(fast_path), self.opts)); self.models["fast"] = m
            self.fn["first"], self.fn["step"] = m.load_function("first"), m.load_function("step")
        if codec_path:
            m = asyncio.run(rt.AIModel.load(str(codec_path), self.opts)); self.models["codec"] = m
            self.fn["codec"] = m.load_function("main")
        self.reset_slow(); self.reset_fast()

    def nd(self, a, dtype):
        return self.rt.NDArray(np.ascontiguousarray(np.asarray(a, dtype=dtype)))

    def reset_slow(self):
        z = np.zeros((M.N_LAYER, 1, M.N_KV, self.cl, M.HEAD_DIM), np.float16)
        self.ss = {"k_cache": self.rt.NDArray(z.copy()), "v_cache": self.rt.NDArray(z.copy())}

    def reset_fast(self):
        z = np.zeros((M.N_FAST_LAYER, 1, M.N_KV, M.NUM_CODEBOOKS, M.HEAD_DIM), np.float16)
        self.fs = {"k_cache": self.rt.NDArray(z.copy()), "v_cache": self.rt.NDArray(z.copy())}

    def _slow(self, name, codes, pos):
        r = asyncio.run(self.fn[name](inputs={"codes": self.nd(codes, np.int32), "pos": self.nd([pos], np.int32)}, state=self.ss))
        return r["logits"].numpy().astype(np.float32), r["hidden"].numpy().astype(np.float32)

    def slow_prefill(self, codes, pos):
        return self._slow("prefill", codes, pos)

    def slow_decode(self, codes, pos):
        return self._slow("decode", codes, pos)

    def fast_first(self, hidden, code0):
        r = asyncio.run(self.fn["first"](inputs={"hidden": self.nd(hidden, np.float16), "code0": self.nd([code0], np.int32)}, state=self.fs))
        return r["logits"].numpy().astype(np.float32)

    def fast_step(self, code, pos):
        r = asyncio.run(self.fn["step"](inputs={"code": self.nd([code], np.int32), "pos": self.nd([pos], np.int32)}, state=self.fs))
        return r["logits"].numpy().astype(np.float32)

    def codec_decode(self, codes):
        r = asyncio.run(self.fn["codec"](inputs={"codes": self.nd(codes, np.int32)}))
        return r["wav"].numpy().astype(np.float32).reshape(-1)


# ----------------------------------------------------------------------------- parts
def export_slow(args, snap: Path) -> Path:
    slow, _ = M.load_slow_fast(snap / "model.safetensors", cl=args.cl, dtype=DT)
    st = M.slow_kv_state(args.cl, DT)
    ref_p = {"codes": torch.zeros(1, M.NUM_CODEBOOKS + 1, args.window, dtype=torch.int32),
             "pos": torch.tensor([0], dtype=torch.int32), **{k: v.clone() for k, v in st.items()}}
    ref_p["codes"][:, 0] = M.PAD
    ref_d = {"codes": torch.zeros(1, M.NUM_CODEBOOKS + 1, 1, dtype=torch.int32),
             "pos": torch.tensor([args.window], dtype=torch.int32), **{k: v.clone() for k, v in st.items()}}
    ref_d["codes"][:, 0] = M.PAD
    t0 = time.time()
    core = maybe_quantize(slow, (ref_p["codes"], ref_p["pos"], ref_p["k_cache"], ref_p["v_cache"]), args.mode, (r".*head$",))
    if args.mode != "fp16":
        print(f"[quant] slow {args.mode} in {time.time() - t0:.0f}s", flush=True)
    prefill_m, decode_m = M.SlowARPrefill(core).eval(), M.SlowARDecode(core).eval()
    name = f"audio8_slow_ar_{args.mode}_cl{args.cl}_w{args.window}"
    t0 = time.time()
    prog = convert([
        {"module": prefill_m, "ref": ref_p, "input_names": ("codes", "pos"), "output_names": ("logits", "hidden"),
         "state_names": ("k_cache", "v_cache"), "name": "prefill"},
        {"module": decode_m, "ref": ref_d, "input_names": ("codes", "pos"), "output_names": ("logits", "hidden"),
         "state_names": ("k_cache", "v_cache"), "name": "decode"},
    ])
    print(f"[convert] slow in {time.time() - t0:.0f}s", flush=True)
    path = args.out / f"{name}.aimodel"
    save(prog, path)
    return path


def export_fast(args, snap: Path) -> Path:
    _, fast = M.load_slow_fast(snap / "model.safetensors", cl=16, dtype=DT)
    st = M.fast_kv_state(DT)
    ref_core = (torch.zeros(1, 2, M.DIM, dtype=DT), torch.tensor([0], dtype=torch.int32), st["k_cache"].clone(), st["v_cache"].clone())
    core = maybe_quantize(fast, ref_core, args.mode, (r".*fast_output$",))
    first_m, step_m = M.FastARFirst(core).eval(), M.FastARStep(core).eval()
    ref_f = {"hidden": torch.zeros(1, 1, M.DIM, dtype=DT), "code0": torch.tensor([0], dtype=torch.int32),
             **{k: v.clone() for k, v in st.items()}}
    ref_s = {"code": torch.tensor([0], dtype=torch.int32), "pos": torch.tensor([2], dtype=torch.int32),
             **{k: v.clone() for k, v in st.items()}}
    name = f"audio8_fast_ar_{args.mode}"
    prog = convert([
        {"module": first_m, "ref": ref_f, "input_names": ("hidden", "code0"), "output_names": ("logits",),
         "state_names": ("k_cache", "v_cache"), "name": "first"},
        {"module": step_m, "ref": ref_s, "input_names": ("code", "pos"), "output_names": ("logits",),
         "state_names": ("k_cache", "v_cache"), "name": "step"},
    ])
    path = args.out / f"{name}.aimodel"
    save(prog, path)
    return path


def export_codec(args, snap: Path, frames: int) -> Path:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(snap), trust_remote_code=True)
    pub = M.load_publisher_codec(snap, cfg)
    dec = M.build_codec_decoder(pub, t_max=frames).to(DT).eval()
    ref = {"codes": torch.zeros(1, M.NUM_CODEBOOKS, frames, dtype=torch.int32)}
    name = f"audio8_codec_decoder_fp16_t{frames}"
    t0 = time.time()
    prog = convert([{"module": dec, "ref": ref, "input_names": ("codes",), "output_names": ("wav",), "name": "main"}])
    print(f"[convert] codec t{frames} in {time.time() - t0:.0f}s", flush=True)
    path = args.out / f"{name}.aimodel"
    save(prog, path)
    return path


def quick_gate(args, slow_path, fast_path, codec_path, codec_frames):
    gen = json.loads((HERE / "fixtures.json").read_text())["generation"]
    orc = np.load(WORK / "oracle" / f"{args.gate_fixture}.npz")
    t0 = time.time()
    ports = EnginePorts(slow_path, fast_path, codec_path, args.cl, unit="gpu")
    print(f"[gate] loaded on gpu in {time.time() - t0:.1f}s", flush=True)
    res = {}
    if slow_path and fast_path:
        tf = teacher_forced(ports, orc, gen, window=args.window, max_frames=args.gate_frames)
        s = summarize(tf)
        res["teacher_forced"] = s
        print(f"[gate {args.gate_fixture}] prefill cos {s['prefill']['logits_cos']:.6f} | slow {s['slow_steps']} cos min {s['slow_cos_min']:.6f} "
              f"argmax {s['slow_argmax_eq']}/{s['slow_steps']} sample {s['slow_sample_eq']}/{s['slow_steps']} miss {s['slow_sample_miss_margins']} | "
              f"fast {s['fast_logits']} cos min {s['fast_cos_min']:.6f} argmax {s['fast_argmax_eq']}/{s['fast_logits']} "
              f"sample {s['fast_sample_eq']}/{s['fast_logits']} | slow decode {s['slow_decode_ms_median']:.1f} ms, fast frame {s['fast_frame_ms_median']:.1f} ms",
              flush=True)
    elif slow_path or fast_path:
        print("[gate] slow and fast are gated together; export both to gate them", flush=True)
    if codec_path:
        cg = codec_gate(ports, orc, bucket=codec_frames)
        res["codec"] = cg
        print(f"[gate {args.gate_fixture}] codec cos {cg['cos']:.6f} max|Δ| {cg['maxabs']:.3e} logmel {cg['logmel_cos']:.6f} "
              f"({cg['frames']} frames in bucket {codec_frames}, {cg['seconds']:.2f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["slow", "fast", "codec", "all"], default="all")
    ap.add_argument("--mode", choices=["fp16", "int8"], default="int8")
    ap.add_argument("--cl", type=int, default=M.MAX_SEQ_LEN)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--codec-frames", type=int, nargs="*", default=[160])
    ap.add_argument("--out", default=str(EXPORTS))
    ap.add_argument("--gate-fixture", default="ja_1")
    ap.add_argument("--gate-frames", type=int, default=12)
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    snap = Path(hf_snapshot(HF_ID, revision=REVISION))

    slow_path = fast_path = codec_path = None
    if args.part in ("slow", "all"):
        slow_path = export_slow(args, snap)
    if args.part in ("fast", "all"):
        fast_path = export_fast(args, snap)
    codec_paths = []
    if args.part in ("codec", "all"):
        for fr in args.codec_frames:
            codec_paths.append((fr, export_codec(args, snap, fr)))
    if args.no_gate:
        return
    if args.part == "all" or (slow_path and fast_path):
        quick_gate(args, slow_path, fast_path, codec_paths[0][1] if codec_paths else None,
                   codec_paths[0][0] if codec_paths else None)
    elif codec_paths:
        for fr, p in codec_paths:
            quick_gate(args, None, None, p, fr)
    elif slow_path or fast_path:
        # gate a single AR part against the other part's existing export, if present
        other = None
        if slow_path:
            cands = sorted(args.out.glob(f"audio8_fast_ar_{args.mode}.aimodel"))
            other = cands[0] if cands else None
            quick_gate(args, slow_path, other, None, None) if other else print("[gate] no fast export to pair with")
        else:
            cands = sorted(args.out.glob(f"audio8_slow_ar_{args.mode}_cl{args.cl}_w{args.window}.aimodel"))
            other = cands[0] if cands else None
            quick_gate(args, other, fast_path, None, None) if other else print("[gate] no slow export to pair with")


if __name__ == "__main__":
    main()
