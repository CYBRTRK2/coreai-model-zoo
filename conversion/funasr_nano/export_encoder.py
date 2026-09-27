#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""Export the Fun-ASR-Nano audio path (SAN-M encoder + adaptor) to a static Core AI ``.aimodel``.

Graph: ``feats [1, 500, 560]`` + ``mask [1, 500]`` -> ``audio_embeds [63, 1024]`` (see
``funasr_encoder.py`` for the contract). Weights come from the vLLM repo's ``model.safetensors``.
``--dtype fp16`` casts weights and activations to fp16; ``--dtype fp32`` keeps both in fp32;
``--dtype fp16w32`` stores every parameter in fp16 and computes in fp32 (each parameter is read
through a ``.float()`` parametrization, so the graph holds fp16 constants plus a cast — unless the
optimizer folds the cast, which would bring the size back to the fp32 bundle's).
Gate the result with ``gate_encoder.py``.

Run with the shared venv:
    ~/code/coreai/coreai-models/.venv/bin/python conversion/funasr_nano/export_encoder.py --dtype fp16
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import work_path  # noqa: E402
from funasr_encoder import D_IN, L_MAX, N_MAX, FunASRNanoAudioEncoder, load_weights  # noqa: E402

import coreai.runtime as rt  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402

WORK = work_path("_funasr_nano")
SAFETENSORS = WORK / "hf" / "model.safetensors"
DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "fp16w32": torch.float32}


class _Upcast(torch.nn.Module):
    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w.float()


def fp16_storage_fp32_compute(module: torch.nn.Module) -> torch.nn.Module:
    """Keep every parameter as fp16 and hand the forward an fp32 copy."""
    import torch.nn.utils.parametrize as P

    for m in list(module.modules()):
        for name, param in list(m.named_parameters(recurse=False)):
            setattr(m, name, torch.nn.Parameter(param.detach().half(), requires_grad=False))
            P.register_parametrization(m, name, _Upcast(), unsafe=True)
    return module


def bundle_name(dtype: str, fsmn: str, ln_fp32: bool) -> str:
    extra = ("_shift" if fsmn == "shift" else "") + ("_lnfp32" if ln_fp32 else "")
    return f"funasr_nano_audio_encoder_{dtype}{extra}_l{L_MAX}.aimodel"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    ap.add_argument("--fsmn", choices=("conv", "shift"), default="conv",
                    help="conv = depthwise Conv1d; shift = 11-tap shift-accumulate (same math)")
    ap.add_argument("--ln-fp32", action="store_true", help="compute every LayerNorm in fp32 inside an fp16 graph")
    ap.add_argument("--out-dir", default=str(WORK / "exports"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    out = Path(args.out_dir) / bundle_name(args.dtype, args.fsmn, args.ln_fp32)
    enc = load_weights(FunASRNanoAudioEncoder(L_MAX, N_MAX, fsmn=args.fsmn, ln_fp32=args.ln_fp32), SAFETENSORS, dtype)
    if args.dtype == "fp16w32":
        enc = fp16_storage_fp32_compute(enc)
    example = {"feats": torch.zeros(1, L_MAX, D_IN, dtype=dtype), "mask": torch.ones(1, L_MAX, dtype=dtype)}

    t0 = time.perf_counter()
    print(f"[export] {out.name}: torch.export -> Core AI ...", flush=True)
    prog = export_to_coreai(enc, example, dynamic_shapes=None, input_names=("feats", "mask"),
                            output_names=("audio_embeds",), state_names=None, externalize_modules=[])
    t1 = time.perf_counter()
    print(f"[export] converted in {t1 - t0:.1f} s; optimize ...", flush=True)
    prog.optimize()
    t2 = time.perf_counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)                 # save_asset does not overwrite
    prog.save_asset(out, rt.AIModelAssetMetadata())
    size = subprocess.run(["du", "-sh", str(out)], capture_output=True, text=True).stdout.split()[0]
    print(f"[save] {out}  optimize {t2 - t1:.1f} s, save {time.perf_counter() - t2:.1f} s, du -sh {size}", flush=True)


if __name__ == "__main__":
    main()
