# Community port — NOT an Apple model.
"""Fun-ASR-Nano host front end in NumPy: 16 kHz mono waveform -> LFR features ``[L, 560]``.

This is the line-for-line spec the Swift front end will port. It reproduces, for the settings
Fun-ASR-Nano ships with (``config.yaml`` ``frontend_conf``), what funasr 1.4.16 computes:

``funasr.frontends.wav_frontend.WavFrontend.forward``
    waveform * 32768 -> ``kaldi.fbank`` -> ``apply_lfr(m=7, n=6)`` (no CMVN: ``cmvn_file`` is null)
``funasr.utils.fbank.fbank``
    forwards to ``torchaudio.compliance.kaldi.fbank`` when torchaudio is installed (it is in the
    oracle venv), called with num_mel_bins 80, frame_length 25 ms, frame_shift 10 ms,
    dither 0 (the oracle pins it; funasr's default of 1.0 adds noise), energy_floor 0,
    window hamming, snip_edges True and the torchaudio defaults for everything else:
    remove_dc_offset, preemphasis 0.97, round_to_power_of_two (512-point FFT), use_power,
    use_log_fbank, low_freq 20 Hz, high_freq 0 (= Nyquist, 8000 Hz), no VTLN, no energy column.

Per frame (400 samples, hop 160, only frames that fit completely):
    1. subtract the frame mean                      (remove_dc_offset)
    2. x[j] -= 0.97 * x[max(j - 1, 0)]              (pre-emphasis inside the frame; x[0] -> 0.03 x[0])
    3. multiply by hamming 0.54 - 0.46 cos(2 pi n / 399)
    4. zero-pad to 512, power spectrum |rfft|^2     (257 bins)
    5. 80 triangular kaldi mel filters on bins 0..255 (bin 256 has weight 0),
       mel(f) = 1127 ln(1 + f / 700), 20..8000 Hz, filter edges spaced evenly in mel
    6. log(max(energy, FLT_EPSILON))

LFR (``apply_lfr``): output frame i stacks fbank frames 6i-3 .. 6i+3, each index clamped to
[0, T-1] (funasr left-pads 3 copies of frame 0 and repeats the last frame at the tail), so
``L = ceil(T / 6)``. ``lfr_reference`` below is funasr's code transcribed literally;
``fbank_lfr`` uses the clamp form, and ``--selftest`` checks the two agree for T = 1..3000.

Computed in float64 here; the oracle computes in float32, so expect ~1e-4-level differences.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

SAMPLE_RATE = 16000
FRAME_LEN = 400          # 25 ms
FRAME_SHIFT = 160        # 10 ms
N_FFT = 512              # next power of two >= 400
N_MELS = 80
LOW_FREQ = 20.0
HIGH_FREQ = SAMPLE_RATE / 2
PREEMPH = 0.97
LFR_M = 7
LFR_N = 6
FEAT_DIM = N_MELS * LFR_M  # 560
FLT_EPSILON = float(np.finfo(np.float32).eps)  # 1.1920929e-07, torchaudio's log floor


def _mel(f: np.ndarray | float) -> np.ndarray | float:
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def kaldi_mel_banks() -> np.ndarray:
    """``torchaudio.compliance.kaldi.get_mel_banks`` + the zero column fbank() appends: ``[80, 257]``."""
    num_fft_bins = N_FFT // 2
    fft_bin_width = SAMPLE_RATE / N_FFT
    mel_low, mel_high = _mel(LOW_FREQ), _mel(HIGH_FREQ)
    delta = (mel_high - mel_low) / (N_MELS + 1)
    b = np.arange(N_MELS, dtype=np.float64)[:, None]
    left = mel_low + b * delta
    center = mel_low + (b + 1.0) * delta
    right = mel_low + (b + 2.0) * delta
    mel = _mel(fft_bin_width * np.arange(num_fft_bins, dtype=np.float64))[None, :]
    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    bins = np.maximum(0.0, np.minimum(up, down))                 # [80, 256]
    return np.concatenate([bins, np.zeros((N_MELS, 1))], axis=1)  # [80, 257]


_MEL_BANKS = kaldi_mel_banks()
_HAMMING = 0.54 - 0.46 * np.cos(2.0 * math.pi * np.arange(FRAME_LEN) / (FRAME_LEN - 1))


def num_fbank_frames(num_samples: int) -> int:
    return 0 if num_samples < FRAME_LEN else 1 + (num_samples - FRAME_LEN) // FRAME_SHIFT


def num_lfr_frames(num_samples: int) -> int:
    return math.ceil(num_fbank_frames(num_samples) / LFR_N)


def fbank(wav: np.ndarray) -> np.ndarray:
    """``wav`` float in [-1, 1) at 16 kHz -> log-mel ``[T, 80]`` float64."""
    x = np.asarray(wav, dtype=np.float64).reshape(-1) * 32768.0
    t = num_fbank_frames(x.shape[0])
    if t == 0:
        return np.zeros((0, N_MELS))
    idx = np.arange(t)[:, None] * FRAME_SHIFT + np.arange(FRAME_LEN)[None, :]
    frames = x[idx]                                                  # [T, 400]
    frames = frames - frames.mean(axis=1, keepdims=True)
    prev = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames = (frames - PREEMPH * prev) * _HAMMING
    padded = np.zeros((t, N_FFT))
    padded[:, :FRAME_LEN] = frames
    spec = np.abs(np.fft.rfft(padded, axis=1)) ** 2                  # [T, 257]
    energies = spec @ _MEL_BANKS.T                                   # [T, 80]
    return np.log(np.maximum(energies, FLT_EPSILON))


def lfr(feats: np.ndarray) -> np.ndarray:
    """Clamp form of funasr ``apply_lfr(m=7, n=6)``: ``[T, 80]`` -> ``[ceil(T/6), 560]``."""
    t = feats.shape[0]
    t_lfr = math.ceil(t / LFR_N)
    src = np.arange(t_lfr)[:, None] * LFR_N + np.arange(LFR_M)[None, :] - (LFR_M - 1) // 2
    src = np.clip(src, 0, t - 1)                                     # [L, 7]
    return feats[src].reshape(t_lfr, LFR_M * feats.shape[1])


def lfr_reference(feats: np.ndarray) -> np.ndarray:
    """funasr ``apply_lfr`` transcribed literally (vstack padding + as_strided rows)."""
    t = feats.shape[0]
    t_lfr = int(np.ceil(t / LFR_N))
    inputs = np.vstack([np.repeat(feats[:1], (LFR_M - 1) // 2, axis=0), feats])
    t = t + (LFR_M - 1) // 2
    last_idx = (t - LFR_M) // LFR_N + 1
    num_padding = LFR_M - (t - last_idx * LFR_N)
    if num_padding > 0:
        num_padding = (2 * LFR_M - 2 * t + (t_lfr - 1 + last_idx) * LFR_N) / 2 * (t_lfr - last_idx)
        inputs = np.vstack([inputs] + [inputs[-1:]] * int(num_padding))
    flat = inputs.reshape(-1)
    d = feats.shape[1]
    rows = [flat[i * LFR_N * d: i * LFR_N * d + LFR_M * d] for i in range(t_lfr)]
    assert all(r.shape[0] == LFR_M * d for r in rows), "as_strided would read past the end"
    return np.stack(rows)


def fbank_lfr(wav: np.ndarray) -> np.ndarray:
    """16 kHz mono waveform (float, [-1, 1)) -> LFR features ``[L, 560]`` float32."""
    return lfr(fbank(wav)).astype(np.float32)


def fake_token_len(num_lfr: int) -> int:
    """Audio slots the LLM receives (funasr ``data_load_speech``, ``use_low_frame_rate``)."""
    olens = 1 + (num_lfr - 3 + 2) // 2
    olens = 1 + (olens - 3 + 2) // 2
    return (olens - 1) // 2 + 1


def _selftest() -> None:
    rng = np.random.default_rng(0)
    for t in range(1, 3001):
        f = rng.standard_normal((t, 4))
        a, b = lfr(f), lfr_reference(f)
        assert a.shape == b.shape and np.array_equal(a, b), t
    assert num_fbank_frames(480000) == 2998 and num_lfr_frames(480000) == 500
    assert fake_token_len(500) == 63
    print("selftest ok: clamp LFR == funasr apply_lfr for T=1..3000; 30 s -> T=2998, L=500, N=63")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
