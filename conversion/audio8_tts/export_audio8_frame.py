#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""Export the one-call-per-frame DualAR asset: prefill + frame + first_frame over one weight set and one KV state.

    audio8_dualar_<mode>_cl<CL>_w<W>.aimodel
      prefill    (codes [1,11,W] i32, pos [1] i32)                                  -> logits [W,4097] f16, hidden [W,896] f16
      frame      (codes [1,11,1] i32, pos [1] i32, noise_slow [2,4097] f32, window [10] i32, noise_fast [9,4096] f32,
                  forced [11] i32, use_forced [1] f32)                              -> semantic [1] i32, codes [10] i32, logits [4097] f16,
                                                                                      hidden [896] f16, fast_logits [9,4096] f16,
                                                                                      sampled_semantic [1] i32, sampled_codes [10] i32
      first_frame(logits [4097] f16, hidden [896] f16, noise_slow, window, noise_fast, forced, use_forced) -> the same outputs
      state      k_cache / v_cache [24, 1, 2, CL, 64] f16 (prefill and frame)

`--mode int8` quantizes the slow AR's 24 layers (weight-only int8, per-block-32, symmetric with clipping); the fast AR,
the embeddings, the head and the norms stay fp16. The codec decoder is a separate asset (export_audio8.py --part codec).

Ladder: 1. eager fp32 teacher-forced replay of the fused module on a fixture (logits, fast logits and the in-graph
sampler's draws against the oracle) — a re-authoring check; 2. export; 3. the same replay through the .aimodel on the
GPU for a few frames. The full gate is gate_audio8_frame.py.

    ~/code/coreai/coreai-models/.venv/bin/python conversion/audio8_tts/export_audio8_frame.py --mode int8
"""
from __future__ import annotations

import argparse
import asyncio
import json
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
from audio8_frame import FRAME_OUTPUTS, FastUnrolled, FirstFrameGraph, FrameGraph  # noqa: E402
from export_audio8 import convert, maybe_quantize, save  # noqa: E402
from replay import cos, frame_column, pad_window  # noqa: E402

HF_ID = "Edge0/Audio8-TTS-Preview-0.6b"
REVISION = "f07040f3d151f1ba0253bfb92cb2f5dd38b44594"
WORK = work_path("_audio8_tts")
EXPORTS = WORK / "exports"
DT = torch.float16
NO_WINDOW = np.full(10, -1, np.int32)


# ----------------------------------------------------------------------------- the host loop over "frame ports"
class FramePorts:
    """prefill(codes, pos) -> (logits [S,4097], hidden [S,896]); first_frame(...) / frame(...) -> dict of FRAME_OUTPUTS."""

    def reset(self): ...
    def prefill(self, codes, pos): ...
    def first_frame(self, logits, hidden, noise_slow, window, noise_fast, forced, use_forced): ...
    def frame(self, codes, pos, noise_slow, window, noise_fast, forced, use_forced): ...


def ras_window_after(window: np.ndarray | None, semantic: int) -> np.ndarray:
    if window is None:
        return np.zeros(10, np.int32)
    return np.concatenate([window[1:], np.array([semantic], np.int32)])


def prefill_windowed(ports: FramePorts, prompt: np.ndarray, window: int):
    P = prompt.shape[1]
    codes = prompt[None].astype(np.int32)
    last = None
    for start in range(0, P, window):
        chunk = codes[:, :, start:start + window]
        real = chunk.shape[2]
        logits, hidden = ports.prefill(pad_window(chunk, window), start)
        last = (np.asarray(logits)[real - 1].astype(np.float32), np.asarray(hidden)[real - 1].astype(np.float32))
    return last


def teacher_forced_frames(ports: FramePorts, orc, window: int = 32, max_frames: int | None = None) -> dict:
    """Feed the oracle's tokens (use_forced = 1); compare the logits and what the in-graph sampler drew."""
    prompt = orc["prompt"]
    P = prompt.shape[1]
    T = int(orc["semantic"].shape[0])
    if max_frames is not None:
        T = min(T, max_frames)
    n_emit = orc["codes"].shape[1]
    ports.reset()
    t0 = time.time()
    logits, hidden = prefill_windowed(ports, prompt, window)
    out = {"prefill_s": time.time() - t0, "slow_cos": [], "slow_argmax_eq": 0, "slow_sample_eq": 0, "hidden_cos": [],
           "fast_cos": [], "fast_argmax_eq": 0, "fast_sample_eq": 0, "fast_n": 0, "n": 0, "frame_s": []}
    ras = None
    for t in range(T):
        sem_orc = int(orc["semantic"][t])
        is_last = sem_orc == M.EOS or t >= n_emit
        forced = np.zeros(11, np.int32)
        forced[0] = sem_orc
        if not is_last:
            forced[1:] = orc["codes"][:, t]
        else:
            forced[1:] = 0
        wnd = NO_WINDOW if ras is None else ras
        t1 = time.time()
        if t == 0:
            r = ports.first_frame(logits, hidden, orc["noise_slow"][t], wnd, orc["noise_fast"][t], forced, 1.0)
        else:
            r = ports.frame(frame_column(int(orc["semantic"][t - 1]), orc["codes"][:, t - 1]), P + t - 1,
                            orc["noise_slow"][t], wnd, orc["noise_fast"][t], forced, 1.0)
        out["frame_s"].append(time.time() - t1)
        lg = np.asarray(r["logits"], np.float32).reshape(-1)
        ref = orc["slow_logits"][t]
        out["slow_cos"].append(cos(lg, ref))
        out["slow_argmax_eq"] += int(np.argmax(lg) == np.argmax(ref))
        out["slow_sample_eq"] += int(int(r["sampled_semantic"].reshape(-1)[0]) == sem_orc)
        out["hidden_cos"].append(cos(np.asarray(r["hidden"], np.float32).reshape(-1), orc["slow_hidden"][t]))
        out["n"] += 1
        if is_last:
            break
        fl = np.asarray(r["fast_logits"], np.float32).reshape(9, M.CODEBOOK_SIZE)
        sc = np.asarray(r["sampled_codes"]).reshape(-1)
        for k in range(9):
            ref_k = orc["fast_logits"][t][k].astype(np.float32)
            out["fast_cos"].append(cos(fl[k], ref_k))
            out["fast_argmax_eq"] += int(np.argmax(fl[k]) == np.argmax(ref_k))
            out["fast_sample_eq"] += int(int(sc[k + 1]) == int(orc["codes"][k + 1, t]))
            out["fast_n"] += 1
        ras = ras_window_after(ras, sem_orc)
    return out


def summarize_tf(o: dict) -> dict:
    return {"steps": o["n"], "slow_cos_min": min(o["slow_cos"]), "slow_argmax_eq": o["slow_argmax_eq"], "slow_sample_eq": o["slow_sample_eq"],
            "hidden_cos_min": min(o["hidden_cos"]), "fast_n": o["fast_n"], "fast_cos_min": min(o["fast_cos"]) if o["fast_cos"] else None,
            "fast_argmax_eq": o["fast_argmax_eq"], "fast_sample_eq": o["fast_sample_eq"],
            "frame_ms_median": float(np.median(o["frame_s"][1:]) * 1e3) if len(o["frame_s"]) > 1 else None,
            "prefill_ms": o["prefill_s"] * 1e3}


def free_run_frames(ports: FramePorts, orc, window: int = 32, max_new: int = 512, seed: int = 0) -> dict:
    prompt = orc["prompt"]
    P = prompt.shape[1]
    n_noise = orc["noise_slow"].shape[0]
    rng = np.random.default_rng(seed)
    ports.reset()
    logits, hidden = prefill_windowed(ports, prompt, window)
    ras = None
    frames, sems = [], []
    forced = np.zeros(11, np.int32)
    t1 = time.time()
    for t in range(max_new):
        ns = orc["noise_slow"][t] if t < n_noise else rng.random((2, M.N_ALLOWED), dtype=np.float32)
        nf = orc["noise_fast"][t] if t < n_noise else rng.random((9, M.CODEBOOK_SIZE), dtype=np.float32)
        wnd = NO_WINDOW if ras is None else ras
        if t == 0:
            r = ports.first_frame(logits, hidden, ns, wnd, nf, forced, 0.0)
        else:
            r = ports.frame(frame_column(sems[-1], frames[-1]), P + t - 1, ns, wnd, nf, forced, 0.0)
        sem = int(np.asarray(r["semantic"]).reshape(-1)[0])
        sems.append(sem)
        if sem == M.EOS:
            break
        frames.append(np.asarray(r["codes"], np.int32).reshape(-1))
        ras = ras_window_after(ras, sem)
    wall = time.time() - t1
    codes = np.stack(frames, axis=1) if frames else np.zeros((10, 0), np.int32)
    ref = orc["codes"]
    n = min(codes.shape[1], ref.shape[1])
    first_div = next((t for t in range(n) if not np.array_equal(codes[:, t], ref[:, t])), None)
    if first_div is None and codes.shape[1] != ref.shape[1]:
        first_div = n
    return {"codes": codes, "frames": int(codes.shape[1]), "oracle_frames": int(ref.shape[1]), "ended_with_eos": bool(sems and sems[-1] == M.EOS),
            "identical_frames_prefix": first_div if first_div is not None else n, "identical": first_div is None,
            "steps": len(sems), "wall_s": wall, "ms_per_frame": wall / max(len(sems), 1) * 1e3}


# ----------------------------------------------------------------------------- torch (eager) ports
class TorchFramePorts(FramePorts):
    def __init__(self, slow: M.SlowARCore, fast: M.FastARCore, cl: int, dtype=torch.float32):
        self.prefill_m = M.SlowARPrefill(slow).eval()
        self.frame_m = FrameGraph(slow, FastUnrolled(fast)).eval()
        self.first_m = FirstFrameGraph(FastUnrolled(fast)).eval()
        self.cl, self.dtype = cl, dtype
        self.reset()

    def reset(self):
        self.st = M.slow_kv_state(self.cl, self.dtype)

    @staticmethod
    def _t(a, dt):
        return torch.from_numpy(np.ascontiguousarray(np.asarray(a))).to(dt)

    @torch.inference_mode()
    def prefill(self, codes, pos):
        l, h = self.prefill_m(self._t(codes, torch.int32), torch.tensor([pos], dtype=torch.int32), self.st["k_cache"], self.st["v_cache"])
        return l.float().numpy(), h.float().numpy()

    def _pack(self, outs):
        return {k: (v.float().numpy() if v.dtype.is_floating_point else v.numpy()) for k, v in zip(FRAME_OUTPUTS, outs)}

    @torch.inference_mode()
    def first_frame(self, logits, hidden, noise_slow, window, noise_fast, forced, use_forced):
        outs = self.first_m(self._t(logits, self.dtype), self._t(hidden, self.dtype), self._t(noise_slow, torch.float32),
                            self._t(window, torch.int32), self._t(noise_fast, torch.float32), self._t(forced, torch.int32),
                            torch.tensor([use_forced], dtype=torch.float32))
        return self._pack(outs)

    @torch.inference_mode()
    def frame(self, codes, pos, noise_slow, window, noise_fast, forced, use_forced):
        outs = self.frame_m(self._t(codes, torch.int32), torch.tensor([pos], dtype=torch.int32), self._t(noise_slow, torch.float32),
                            self._t(window, torch.int32), self._t(noise_fast, torch.float32), self._t(forced, torch.int32),
                            torch.tensor([use_forced], dtype=torch.float32), self.st["k_cache"], self.st["v_cache"])
        return self._pack(outs)


# ----------------------------------------------------------------------------- engine ports
class EngineFramePorts(FramePorts):
    def __init__(self, path: Path, cl: int, unit: str = "gpu"):
        import coreai.runtime as rt

        self.rt, self.cl = rt, cl
        opts = (rt.SpecializationOptions.cpu_only() if unit == "cpu"
                else rt.SpecializationOptions.from_preferred_compute_unit_kind(getattr(rt.ComputeUnitKind, unit)()))
        t0 = time.time()
        self.model = asyncio.run(rt.AIModel.load(str(path), opts))
        self.load_s = time.time() - t0
        self.f_prefill = self.model.load_function("prefill")
        self.f_frame = self.model.load_function("frame")
        self.f_first = self.model.load_function("first_frame")
        self.reset()

    def nd(self, a, dtype):
        return self.rt.NDArray(np.ascontiguousarray(np.asarray(a, dtype=dtype)))

    def reset(self):
        z = np.zeros((M.N_LAYER, 1, M.N_KV, self.cl, M.HEAD_DIM), np.float16)
        self.st = {"k_cache": self.rt.NDArray(z.copy()), "v_cache": self.rt.NDArray(z.copy())}

    def prefill(self, codes, pos):
        r = asyncio.run(self.f_prefill(inputs={"codes": self.nd(codes, np.int32), "pos": self.nd([pos], np.int32)}, state=self.st))
        return r["logits"].numpy().astype(np.float32), r["hidden"].numpy().astype(np.float32)

    def _tail(self, noise_slow, window, noise_fast, forced, use_forced):
        return {"noise_slow": self.nd(noise_slow, np.float32), "window": self.nd(window, np.int32), "noise_fast": self.nd(noise_fast, np.float32),
                "forced": self.nd(forced, np.int32), "use_forced": self.nd([use_forced], np.float32)}

    @staticmethod
    def _pack(r):
        return {k: r[k].numpy() for k in FRAME_OUTPUTS}

    def first_frame(self, logits, hidden, noise_slow, window, noise_fast, forced, use_forced):
        inputs = {"logits": self.nd(logits, np.float16), "hidden": self.nd(hidden, np.float16), **self._tail(noise_slow, window, noise_fast, forced, use_forced)}
        return self._pack(asyncio.run(self.f_first(inputs=inputs)))

    def frame(self, codes, pos, noise_slow, window, noise_fast, forced, use_forced):
        inputs = {"codes": self.nd(codes, np.int32), "pos": self.nd([pos], np.int32), **self._tail(noise_slow, window, noise_fast, forced, use_forced)}
        return self._pack(asyncio.run(self.f_frame(inputs=inputs, state=self.st)))


def load_cores(cl: int, dtype):
    snap = Path(hf_snapshot(HF_ID, revision=REVISION))
    return M.load_slow_fast(snap / "model.safetensors", cl=cl, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp16", "int8"], default="int8")
    ap.add_argument("--cl", type=int, default=M.MAX_SEQ_LEN)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--out", default=str(EXPORTS))
    ap.add_argument("--fixture", default="ja_1")
    ap.add_argument("--eager-frames", type=int, default=12)
    ap.add_argument("--gate-frames", type=int, default=12)
    ap.add_argument("--skip-eager", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    orc = np.load(WORK / "oracle" / f"{args.fixture}.npz")

    if not args.skip_eager:
        slow32, fast32 = load_cores(args.cl, torch.float32)
        s = summarize_tf(teacher_forced_frames(TorchFramePorts(slow32, fast32, args.cl), orc, args.window, args.eager_frames))
        print(f"[eager fp32 {args.fixture}] {s['steps']} steps: slow cos min {s['slow_cos_min']:.7f} argmax {s['slow_argmax_eq']}/{s['steps']} "
              f"sample {s['slow_sample_eq']}/{s['steps']} | fast cos min {s['fast_cos_min']:.7f} argmax {s['fast_argmax_eq']}/{s['fast_n']} "
              f"sample {s['fast_sample_eq']}/{s['fast_n']} | hidden cos min {s['hidden_cos_min']:.7f}", flush=True)
        assert s["slow_sample_eq"] == s["steps"] and s["fast_sample_eq"] == s["fast_n"], "eager fused graph diverges from the oracle"
        del slow32, fast32
    if args.skip_export:
        return

    slow, fast = load_cores(args.cl, DT)
    st = M.slow_kv_state(args.cl, DT)
    ref_p = {"codes": torch.full((1, M.NUM_CODEBOOKS + 1, args.window), 0, dtype=torch.int32), "pos": torch.tensor([0], dtype=torch.int32),
             **{k: v.clone() for k, v in st.items()}}
    ref_p["codes"][:, 0] = M.PAD
    t0 = time.time()
    slow_q = maybe_quantize(slow, (ref_p["codes"], ref_p["pos"], ref_p["k_cache"], ref_p["v_cache"]), args.mode, (r".*head$",))
    if args.mode != "fp16":
        print(f"[quant] slow {args.mode} in {time.time() - t0:.0f}s", flush=True)
    fast_u = FastUnrolled(fast)
    prefill_m = M.SlowARPrefill(slow_q).eval()
    frame_m = FrameGraph(slow_q, fast_u).eval()
    first_m = FirstFrameGraph(fast_u).eval()
    tail = {"noise_slow": torch.rand(2, M.N_ALLOWED), "window": torch.full((10,), -1, dtype=torch.int32),
            "noise_fast": torch.rand(9, M.CODEBOOK_SIZE), "forced": torch.zeros(11, dtype=torch.int32),
            "use_forced": torch.zeros(1)}
    ref_f = {"codes": torch.full((1, M.NUM_CODEBOOKS + 1, 1), 0, dtype=torch.int32), "pos": torch.tensor([args.window], dtype=torch.int32),
             **tail, **{k: v.clone() for k, v in st.items()}}
    ref_f["codes"][:, 0] = M.PAD
    ref_first = {"logits": torch.zeros(M.N_ALLOWED, dtype=DT), "hidden": torch.zeros(M.DIM, dtype=DT), **tail}
    name = f"audio8_dualar_{args.mode}_cl{args.cl}_w{args.window}"
    t0 = time.time()
    prog = convert([
        {"module": prefill_m, "ref": ref_p, "input_names": ("codes", "pos"), "output_names": ("logits", "hidden"),
         "state_names": ("k_cache", "v_cache"), "name": "prefill"},
        {"module": frame_m, "ref": ref_f, "input_names": ("codes", "pos", "noise_slow", "window", "noise_fast", "forced", "use_forced"),
         "output_names": FRAME_OUTPUTS, "state_names": ("k_cache", "v_cache"), "name": "frame"},
        {"module": first_m, "ref": ref_first, "input_names": ("logits", "hidden", "noise_slow", "window", "noise_fast", "forced", "use_forced"),
         "output_names": FRAME_OUTPUTS, "name": "first_frame"},
    ])
    print(f"[convert] dualar in {time.time() - t0:.0f}s", flush=True)
    path = Path(args.out) / f"{name}.aimodel"
    save(prog, path)

    ports = EngineFramePorts(path, args.cl, "gpu")
    print(f"[gate] loaded on gpu in {ports.load_s:.1f}s", flush=True)
    s = summarize_tf(teacher_forced_frames(ports, orc, args.window, args.gate_frames))
    print(f"[gate gpu {args.fixture}] {s['steps']} steps: slow cos min {s['slow_cos_min']:.6f} argmax {s['slow_argmax_eq']}/{s['steps']} "
          f"sample {s['slow_sample_eq']}/{s['steps']} | fast cos min {s['fast_cos_min']:.6f} argmax {s['fast_argmax_eq']}/{s['fast_n']} "
          f"sample {s['fast_sample_eq']}/{s['fast_n']} | frame {s['frame_ms_median']:.1f} ms, prefill {s['prefill_ms']:.0f} ms", flush=True)
    fr = free_run_frames(ports, orc, args.window)
    print(f"[free run gpu {args.fixture}] {fr['frames']} frames (oracle {fr['oracle_frames']}), identical prefix {fr['identical_frames_prefix']}, "
          f"eos {fr['ended_with_eos']}, {fr['ms_per_frame']:.1f} ms/frame", flush=True)


if __name__ == "__main__":
    main()
