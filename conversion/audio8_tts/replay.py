# Community port — NOT an Apple model.
"""Replay harness shared by the eager parity check and the engine gate.

A `Ports` object is the port seen from the host: five calls plus two resets, whatever runs them (the torch
re-author in fp32, the `.aimodel`s through `coreai.runtime`, later the Swift host through a JSON dump).

    slow_prefill(codes [1,11,S] int32, pos int) -> (logits [S,4097] f32, hidden [S,896] f32)
    slow_decode (codes [1,11,1] int32, pos int) -> (logits [1,4097],   hidden [1,896])
    fast_first  (hidden [1,1,896] f32, code0 int) -> logits [1,4096]
    fast_step   (code int, pos int)               -> logits [1,4096]
    codec_decode(codes [1,10,T] int32)            -> wav [T*2048] f32
    reset_slow(), reset_fast()

`teacher_forced` feeds the oracle's own tokens and measures every logits vector against the oracle's;
`free_run` lets the port's sampler choose with the oracle's recorded noise and returns the codes and audio.
"""
from __future__ import annotations

import time

import numpy as np
import torch

import sampler as S
from audio8_model import CODEBOOK_SIZE, CODEC_FRAME, EOS, N_ALLOWED, NUM_CODEBOOKS, PAD, SEMANTIC_BEGIN


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def pad_window(codes: np.ndarray, width: int) -> np.ndarray:
    """codes [1, 11, s] -> [1, 11, width]: row 0 padded with the pad id, codebook rows with 0."""
    s = codes.shape[2]
    if s == width:
        return codes
    out = np.zeros((1, NUM_CODEBOOKS + 1, width), dtype=np.int32)
    out[:, 0] = PAD
    out[:, :, :s] = codes
    return out


def prefill_windowed(ports, prompt: np.ndarray, window: int):
    """Run the prompt [11, P] through `slow_prefill` in windows; returns the last real row's (logits, hidden)."""
    P = prompt.shape[1]
    codes = prompt[None].astype(np.int32)
    last = None
    for start in range(0, P, window):
        chunk = codes[:, :, start:start + window]
        real = chunk.shape[2]
        logits, hidden = ports.slow_prefill(pad_window(chunk, window), start)
        last = (np.asarray(logits)[real - 1], np.asarray(hidden)[real - 1])
    return last


def frame_column(semantic: int, codebooks: np.ndarray) -> np.ndarray:
    col = np.zeros((1, NUM_CODEBOOKS + 1, 1), dtype=np.int32)
    col[0, 0, 0] = semantic
    col[0, 1:, 0] = codebooks
    return col


def teacher_forced(ports, orc, gen: dict, window: int = 32, max_frames: int | None = None) -> dict:
    """Feed the oracle's tokens; compare every logits vector. Returns per-stage metrics."""
    top_k, top_p, temp = int(gen["top_k"]), float(gen["top_p"]), float(gen["temperature"])
    prompt = orc["prompt"]
    P = prompt.shape[1]
    T = int(orc["semantic"].shape[0])            # steps (the last may be eos)
    codes = orc["codes"]                          # [10, T_emit]
    n_emit = codes.shape[1]
    if max_frames is not None:
        T = min(T, max_frames)
    ports.reset_slow()
    t0 = time.time()
    logits, hidden = prefill_windowed(ports, prompt, window)
    t_prefill = time.time() - t0
    out = {
        "prefill": {"logits_cos": cos(logits, orc["prefill_logits"]), "logits_maxabs": float(np.abs(logits - orc["prefill_logits"]).max()),
                    "argmax_eq": int(np.argmax(logits) == np.argmax(orc["prefill_logits"])),
                    "hidden_cos": cos(hidden, orc["prefill_hidden"]), "seconds": t_prefill, "P": P, "window": window},
        "slow": {"cos": [], "maxabs": [], "argmax_eq": 0, "sample_eq": 0, "sample_miss_margin": [], "hidden_cos": [], "n": 0},
        "fast": {"cos": [], "argmax_eq": 0, "sample_eq": 0, "sample_miss_margin": [], "n": 0},
        "timing": {"slow_decode_s": [], "fast_frame_s": []},
    }
    window_ras = S.RASWindow()
    for t in range(T):
        # slow logits for step t: prefill for t = 0, else the decode after frame t-1 (already in `logits`)
        ref = orc["slow_logits"][t]
        out["slow"]["cos"].append(cos(logits, ref))
        out["slow"]["maxabs"].append(float(np.abs(logits - ref).max()))
        out["slow"]["argmax_eq"] += int(np.argmax(logits) == np.argmax(ref))
        out["slow"]["hidden_cos"].append(cos(hidden, orc["slow_hidden"][t]))
        u_n, u_h = orc["noise_slow"][t]
        sem_port = S.sample_semantic(logits, u_n, u_h, window_ras.values, top_k, top_p, temp)
        sem_orc = int(orc["semantic"][t])
        if sem_port == sem_orc:
            out["slow"]["sample_eq"] += 1
        else:
            out["slow"]["sample_miss_margin"].append(S.gumbel_margin(S.processed(ref, top_k, top_p, temp), u_n))
        out["slow"]["n"] += 1
        if sem_orc == EOS or t >= n_emit:
            break
        # fast AR, teacher-forced on the oracle's codebooks
        frame = codes[:, t]
        t1 = time.time()
        ports.reset_fast()
        fl = ports.fast_first(np.asarray(orc["slow_hidden"][t], np.float32).reshape(1, 1, -1), int(frame[0]))
        fast_logits = [np.asarray(fl).reshape(-1)]
        for k in range(1, NUM_CODEBOOKS - 1):
            fast_logits.append(np.asarray(ports.fast_step(int(frame[k]), k + 1)).reshape(-1))
        out["timing"]["fast_frame_s"].append(time.time() - t1)
        for k in range(NUM_CODEBOOKS - 1):
            ref_k = orc["fast_logits"][t][k].astype(np.float32)
            out["fast"]["cos"].append(cos(fast_logits[k], ref_k))
            out["fast"]["argmax_eq"] += int(np.argmax(fast_logits[k]) == np.argmax(ref_k))
            u = orc["noise_fast"][t][k]
            got = S.sample_codebook(fast_logits[k], u, top_k, top_p, temp)
            if got == int(frame[k + 1]):
                out["fast"]["sample_eq"] += 1
            else:
                out["fast"]["sample_miss_margin"].append(S.gumbel_margin(S.processed(ref_k, top_k, top_p, temp), u))
            out["fast"]["n"] += 1
        window_ras.push(sem_orc)
        if t + 1 >= T:
            break
        t2 = time.time()
        logits, hidden = ports.slow_decode(frame_column(sem_orc, frame), P + t)
        logits, hidden = np.asarray(logits).reshape(-1), np.asarray(hidden).reshape(-1)
        out["timing"]["slow_decode_s"].append(time.time() - t2)
    return out


def codec_gate(ports, orc, bucket: int | None = None) -> dict:
    """Decode the oracle's codes (right-padded to `bucket` frames if given) and compare with the oracle wav."""
    codes = orc["codes"].astype(np.int32)
    T = codes.shape[1]
    if bucket is not None:
        assert T <= bucket, (T, bucket)
        padded = np.zeros((NUM_CODEBOOKS, bucket), np.int32)
        padded[:, :T] = codes
        codes = padded
    t0 = time.time()
    wav = np.asarray(ports.codec_decode(codes[None]), np.float32).reshape(-1)[: T * CODEC_FRAME]
    dt = time.time() - t0
    ref = orc["wav"].astype(np.float32)
    n = min(wav.size, ref.size)
    return {"frames": T, "cos": cos(wav[:n], ref[:n]), "maxabs": float(np.abs(wav[:n] - ref[:n]).max()),
            "logmel_cos": logmel_cos(wav[:n], ref[:n]), "seconds": dt, "audio_s": n / 44100.0}


def logmel_cos(a: np.ndarray, b: np.ndarray, sr: int = 44100) -> float:
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=2048, hop_length=512, n_mels=128)
    la = torch.log(mel(torch.from_numpy(a)) + 1e-5)
    lb = torch.log(mel(torch.from_numpy(b)) + 1e-5)
    return cos(la.numpy(), lb.numpy())


def free_run(ports, orc, gen: dict, window: int = 32, max_new: int | None = None, seed: int = 0) -> dict:
    """The port's own loop: the oracle's recorded noise while it lasts, then a seeded NumPy stream, up to
    `max_new` steps (default the generation config's max_new_tokens) or eos."""
    top_k, top_p, temp = int(gen["top_k"]), float(gen["top_p"]), float(gen["temperature"])
    prompt = orc["prompt"]
    P = prompt.shape[1]
    n_noise = orc["noise_slow"].shape[0]
    steps = int(gen["max_new_tokens"]) if max_new is None else max_new
    rng = np.random.default_rng(seed)

    def noise_slow(t):
        if t < n_noise:
            return orc["noise_slow"][t]
        return rng.random((2, N_ALLOWED), dtype=np.float32)

    def noise_fast(t, k):
        if t < n_noise:
            return orc["noise_fast"][t][k]
        return rng.random(CODEBOOK_SIZE, dtype=np.float32)

    ports.reset_slow()
    logits, hidden = prefill_windowed(ports, prompt, window)
    window_ras = S.RASWindow()
    frames, semantics = [], []
    for t in range(steps):
        u_n, u_h = noise_slow(t)
        sem = S.sample_semantic(logits, u_n, u_h, window_ras.values, top_k, top_p, temp)
        semantics.append(sem)
        if sem == EOS:
            break
        code0 = min(max(sem - SEMANTIC_BEGIN, 0), CODEBOOK_SIZE - 1)
        ports.reset_fast()
        fl = np.asarray(ports.fast_first(np.asarray(hidden, np.float32).reshape(1, 1, -1), code0)).reshape(-1)
        cbs = [code0]
        cur = S.sample_codebook(fl, noise_fast(t, 0), top_k, top_p, temp)
        cbs.append(cur)
        for k in range(2, NUM_CODEBOOKS):
            fl = np.asarray(ports.fast_step(cur, k)).reshape(-1)
            cur = S.sample_codebook(fl, noise_fast(t, k - 1), top_k, top_p, temp)
            cbs.append(cur)
        frame = np.asarray(cbs, np.int32)
        frames.append(frame)
        window_ras.push(sem)
        if t + 1 >= steps:
            break
        logits, hidden = ports.slow_decode(frame_column(sem, frame), P + t)
        logits, hidden = np.asarray(logits).reshape(-1), np.asarray(hidden).reshape(-1)
    codes = np.stack(frames, axis=1) if frames else np.zeros((NUM_CODEBOOKS, 0), np.int32)
    ref = orc["codes"]
    n = min(codes.shape[1], ref.shape[1])
    first_div = None
    for t in range(n):
        if not np.array_equal(codes[:, t], ref[:, t]):
            first_div = t
            break
    if first_div is None and codes.shape[1] != ref.shape[1]:
        first_div = n
    return {"codes": codes, "semantics": semantics, "frames": int(codes.shape[1]), "oracle_frames": int(ref.shape[1]),
            "identical_frames_prefix": (first_div if first_div is not None else n), "identical": first_div is None,
            "ended_with_eos": bool(semantics and semantics[-1] == EOS), "steps": len(semantics),
            "noise_source": "oracle" if len(semantics) <= n_noise else f"oracle then numpy seed {seed} from step {n_noise}"}


# ----------------------------------------------------------------------------- eager torch ports
class TorchPorts:
    """The re-authored modules in eager torch (fp32 by default)."""

    def __init__(self, slow, fast, codec, cl: int, dtype=torch.float32):
        from audio8_model import SlowARDecode, SlowARPrefill, FastARFirst, FastARStep, fast_kv_state, slow_kv_state

        self.dtype = dtype
        self.slow_prefill_m = SlowARPrefill(slow).eval()
        self.slow_decode_m = SlowARDecode(slow).eval()
        self.fast_first_m = FastARFirst(fast).eval()
        self.fast_step_m = FastARStep(fast).eval()
        self.codec = codec
        self.cl = cl
        self._slow_state_fn = lambda: slow_kv_state(cl, dtype)
        self._fast_state_fn = lambda: fast_kv_state(dtype)
        self.reset_slow()
        self.reset_fast()

    def reset_slow(self):
        self.ss = self._slow_state_fn()

    def reset_fast(self):
        self.fs = self._fast_state_fn()

    @torch.inference_mode()
    def slow_prefill(self, codes, pos):
        l, h = self.slow_prefill_m(torch.from_numpy(np.ascontiguousarray(codes)), torch.tensor([pos], dtype=torch.int32),
                                   self.ss["k_cache"], self.ss["v_cache"])
        return l.float().numpy(), h.float().numpy()

    @torch.inference_mode()
    def slow_decode(self, codes, pos):
        l, h = self.slow_decode_m(torch.from_numpy(np.ascontiguousarray(codes)), torch.tensor([pos], dtype=torch.int32),
                                  self.ss["k_cache"], self.ss["v_cache"])
        return l.float().numpy(), h.float().numpy()

    @torch.inference_mode()
    def fast_first(self, hidden, code0):
        return self.fast_first_m(torch.from_numpy(np.ascontiguousarray(hidden)).to(self.dtype),
                                 torch.tensor([code0], dtype=torch.int32), self.fs["k_cache"], self.fs["v_cache"]).float().numpy()

    @torch.inference_mode()
    def fast_step(self, code, pos):
        return self.fast_step_m(torch.tensor([code], dtype=torch.int32), torch.tensor([pos], dtype=torch.int32),
                                self.fs["k_cache"], self.fs["v_cache"]).float().numpy()

    @torch.inference_mode()
    def codec_decode(self, codes):
        return self.codec(torch.from_numpy(np.ascontiguousarray(codes))).float().numpy().reshape(-1)


def summarize(tf: dict) -> dict:
    s, f = tf["slow"], tf["fast"]
    return {
        "prefill": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in tf["prefill"].items()},
        "slow_steps": s["n"], "slow_cos_min": min(s["cos"]) if s["cos"] else None, "slow_cos_mean": float(np.mean(s["cos"])) if s["cos"] else None,
        "slow_maxabs_max": max(s["maxabs"]) if s["maxabs"] else None, "slow_argmax_eq": s["argmax_eq"], "slow_sample_eq": s["sample_eq"],
        "slow_sample_miss_margins": [round(m, 4) for m in s["sample_miss_margin"]],
        "slow_hidden_cos_min": min(s["hidden_cos"]) if s["hidden_cos"] else None,
        "fast_logits": f["n"], "fast_cos_min": min(f["cos"]) if f["cos"] else None, "fast_cos_mean": float(np.mean(f["cos"])) if f["cos"] else None,
        "fast_argmax_eq": f["argmax_eq"], "fast_sample_eq": f["sample_eq"],
        "fast_sample_miss_margins": [round(m, 4) for m in f["sample_miss_margin"]],
        "slow_decode_ms_median": (float(np.median(tf["timing"]["slow_decode_s"]) * 1e3) if tf["timing"]["slow_decode_s"] else None),
        "fast_frame_ms_median": (float(np.median(tf["timing"]["fast_frame_s"]) * 1e3) if tf["timing"]["fast_frame_s"] else None),
    }
