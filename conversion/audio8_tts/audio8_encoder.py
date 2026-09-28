#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""The codec ENCODER (voice registration) re-authored, exported and gated: 44.1 kHz audio -> the 10 codebooks.

    audio [1, 1, N] f32  (N = T * 2048, right-padded)  ->  codes [1, 10, T] i32

Encoder (SEANet-style, 207M): causal convs with strides 2·4·8·8 = 512 (the last block carries a 4-layer window-512
transformer), a 2× downsample pair (stride-2 causal conv + ConvNeXt) to 2048 samples per frame, the 8-layer
window-128 "pre" transformer, then the vector quantizers: one semantic book (4096 × 8, cosine nearest neighbour on
the `in_proj` output) and nine residual books (1024 × 8), each subtracting its `out_proj` reconstruction from the
running residual. Every op is causal, so a clip shorter than the bucket is right-padded and its first
ceil(len / 2048) frames are exact.

Gate: the four clone fixtures' reference clips (`ref_wav` in the oracle npz, 44.1 kHz) -> codes vs the fp32
oracle's `ref_codes`, per codebook exact-match rate; then the same audio through the `.aimodel` on the GPU.

`--dtype`: `fp16` (whole graph fp16: the vector quantizers' nearest-neighbour decisions flip on fp16 noise and the
residual books cascade — 44–121 of ~200 frames exact on the GPU), `fp32`, or `fp16w32` (every parameter stored fp16,
handed to the forward as fp32: fp32 arithmetic at half the bytes — the Fun-ASR encoder's recipe).

    ~/code/coreai/coreai-models/.venv/bin/python conversion/audio8_tts/audio8_encoder.py --frames 216 --dtype fp16w32   # 10.0 s
"""
from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from _paths import hf_snapshot, work_path  # noqa: E402

import audio8_model as M  # noqa: E402
from audio8_model import (CODEC_FRAME, CODEC_NORM_EPS, NUM_CODEBOOKS, CodecWindowTransformer, ConvNeXtBlock, RMSNorm,  # noqa: E402
                          Snake, load_publisher_codec, snake)

HF_ID = "Edge0/Audio8-TTS-Preview-0.6b"
REVISION = "f07040f3d151f1ba0253bfb92cb2f5dd38b44594"
WORK = work_path("_audio8_tts")
EXPORTS = WORK / "exports"


class CausalConvS(nn.Module):
    """Causal Conv1d with stride: left pad (k - s), right pad so the output length is L / s (L divisible by s)."""

    def __init__(self, cin, cout, k, stride=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, stride=stride, dilation=dilation, groups=groups)
        self.stride = stride
        self.k_eff = (k - 1) * dilation + 1
        self.pad_left = self.k_eff - stride

    def forward(self, x):
        L = x.shape[-1]
        frames = (L - self.k_eff + self.pad_left) / self.stride + 1
        ideal = (math.ceil(frames) - 1) * self.stride + self.k_eff - self.pad_left
        right = ideal - L
        return self.conv(F.pad(x, (self.pad_left, right)))


class EncResUnit(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.s1 = Snake(dim)
        self.c1 = CausalConvS(dim, dim, 7, dilation=dilation)
        self.s2 = Snake(dim)
        self.c2 = CausalConvS(dim, dim, 1)

    def forward(self, x):
        y = self.c2(self.s2(self.c1(self.s1(x))))
        d = x.shape[-1] - y.shape[-1]
        if d > 0:
            x = x[..., :-d]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim, stride, transformer_layers, t_max):
        super().__init__()
        self.res = nn.ModuleList([EncResUnit(dim // 2, d) for d in (1, 3, 9)])
        self.snake = Snake(dim // 2)
        self.down = CausalConvS(dim // 2, dim, 2 * stride, stride=stride)
        self.transformer = (CodecWindowTransformer(transformer_layers, dim // 64, dim // 64, dim, dim * 3, 512, CODEC_NORM_EPS, t_max)
                            if transformer_layers else None)

    def forward(self, x):
        for r in self.res:
            x = r(x)
        x = self.down(self.snake(x))
        if self.transformer is not None:
            x = self.transformer(x)
        return x


class VQ(nn.Module):
    def __init__(self, codebook_size, dim=1024, cdim=8):
        super().__init__()
        self.in_proj = nn.Conv1d(dim, cdim, 1)
        self.out_proj = nn.Conv1d(cdim, dim, 1)
        self.codebook = nn.Parameter(torch.zeros(codebook_size, cdim))

    def forward(self, z):
        """z [B, 1024, T] -> (reconstruction [B, 1024, T], codes [B, T])."""
        proj = self.in_proj(z)                                  # [B, 8, T]
        flat = proj.transpose(1, 2)                             # [B, T, 8]
        # F.normalize without its eps clamp: explicit x * rsqrt(sum x^2 + eps)  (eps 1e-12 squared as F.normalize's 1e-12 floor)
        fn = flat * torch.rsqrt(flat.pow(2).sum(-1, keepdim=True).clamp_min(1e-24))
        cb = self.codebook * torch.rsqrt(self.codebook.pow(2).sum(-1, keepdim=True).clamp_min(1e-24))
        # distances = |f|^2 - 2 f·c + |c|^2 ; argmax(-dist) == argmax(f·c) since |f| = |c| = 1 up to the eps
        sims = fn @ cb.t()                                      # [B, T, K]
        codes = sims.argmax(-1)                                 # [B, T]
        q = F.embedding(codes, self.codebook).transpose(1, 2)   # [B, 8, T]
        return self.out_proj(q), codes


class CodecEncoder(nn.Module):
    """audio [1, 1, N] -> codes [1, 10, N/2048] int32."""

    def __init__(self, frames: int):
        super().__init__()
        self.frames = frames
        dim = 64
        self.first = CausalConvS(1, dim, 7)
        blocks = []
        t_at = frames * 4                                       # frames at the 512-sample rate (last encoder block)
        for stride, layers in zip((2, 4, 8, 8), (0, 0, 0, 4)):
            dim *= 2
            blocks.append(EncoderBlock(dim, stride, layers, t_at))
        self.blocks = nn.ModuleList(blocks)
        self.snake = Snake(dim)
        self.last = CausalConvS(dim, 1024, 3)
        self.downsample = nn.ModuleList([nn.ModuleList([CausalConvS(1024, 1024, 2, stride=2), ConvNeXtBlock(1024)]) for _ in range(2)])
        self.pre = CodecWindowTransformer(8, 16, 16, 1024, 3072, 128, CODEC_NORM_EPS, frames)
        self.semantic = VQ(4096)
        self.residual = nn.ModuleList([VQ(1024) for _ in range(NUM_CODEBOOKS - 1)])

    def forward(self, audio):
        x = self.first(audio)
        for b in self.blocks:
            x = b(x)
        z = self.last(self.snake(x))                            # [1, 1024, 4T]
        for down, cnx in self.downsample:
            z = cnx(down(z))                                    # [1, 1024, T]
        z = self.pre(z)
        rec, c0 = self.semantic(z)
        codes = [c0]
        residual = z - rec
        for vq in self.residual:
            rec, c = vq(residual)
            residual = residual - rec
            codes.append(c)
        return torch.stack(codes, dim=1).to(torch.int32)        # [1, 10, T]


def _copy_conv(dst, src):
    dst.weight.data.copy_(src.weight.detach())
    dst.bias.data.copy_(src.bias.detach())


def _copy_transformer(dst: CodecWindowTransformer, src):
    for d, s in zip(dst.layers, src.layers):
        d.attention.wqkv.weight.data.copy_(s.attention.wqkv.weight.detach())
        d.attention.wo.weight.data.copy_(s.attention.wo.weight.detach())
        d.w1.weight.data.copy_(s.feed_forward.w1.weight.detach())
        d.w2.weight.data.copy_(s.feed_forward.w2.weight.detach())
        d.w3.weight.data.copy_(s.feed_forward.w3.weight.detach())
        d.attention_norm.weight.data.copy_(s.attention_norm.weight.detach())
        d.ffn_norm.weight.data.copy_(s.ffn_norm.weight.detach())
        d.attention_gamma.data.copy_(s.attention_layer_scale.gamma.detach())
        d.ffn_gamma.data.copy_(s.ffn_layer_scale.gamma.detach())
    dst.norm.weight.data.copy_(src.norm.weight.detach())


def _copy_cnx(dst: ConvNeXtBlock, src):
    _copy_conv(dst.dwconv.conv, src.dwconv.conv)
    dst.norm.weight.data.copy_(src.norm.weight.detach()); dst.norm.bias.data.copy_(src.norm.bias.detach())
    dst.pwconv1.weight.data.copy_(src.pwconv1.weight.detach()); dst.pwconv1.bias.data.copy_(src.pwconv1.bias.detach())
    dst.pwconv2.weight.data.copy_(src.pwconv2.weight.detach()); dst.pwconv2.bias.data.copy_(src.pwconv2.bias.detach())
    dst.gamma.data.copy_(src.gamma.detach())


def build_codec_encoder(pub, frames: int) -> CodecEncoder:
    enc = CodecEncoder(frames)
    src = pub.encoder.block                                     # Sequential
    _copy_conv(enc.first.conv, src[0].conv)
    for bi in range(4):
        sb = src[1 + bi].block                                  # Sequential: 3 res, snake, conv, transformer|identity
        b = enc.blocks[bi]
        for ri in range(3):
            ru = sb[ri].block
            b.res[ri].s1.alpha.data.copy_(ru[0].alpha.detach()); _copy_conv(b.res[ri].c1.conv, ru[1].conv)
            b.res[ri].s2.alpha.data.copy_(ru[2].alpha.detach()); _copy_conv(b.res[ri].c2.conv, ru[3].conv)
        b.snake.alpha.data.copy_(sb[3].alpha.detach())
        _copy_conv(b.down.conv, sb[4].conv)
        if b.transformer is not None:
            _copy_transformer(b.transformer, sb[5])
    enc.snake.alpha.data.copy_(src[5].alpha.detach())
    _copy_conv(enc.last.conv, src[6].conv)
    q = pub.quantizer
    for i in range(2):
        _copy_conv(enc.downsample[i][0].conv, q.downsample[i][0].conv)
        _copy_cnx(enc.downsample[i][1], q.downsample[i][1])
    _copy_transformer(enc.pre, q.pre_module)
    vqs = [q.semantic_quantizer.quantizers[0]] + list(q.quantizer.quantizers)
    for dst, s in zip([enc.semantic] + list(enc.residual), vqs):
        _copy_conv(dst.in_proj, s.in_proj)
        _copy_conv(dst.out_proj, s.out_proj)
        dst.codebook.data.copy_(s.codebook.weight.detach())
    return enc.eval()


class _Upcast(nn.Module):
    """Parametrization: the stored fp16 parameter is read as fp32 by every op."""

    def forward(self, x):
        return x.float()


def fp16_storage_fp32_compute(module: nn.Module) -> nn.Module:
    """Keep every parameter as fp16 and hand the forward an fp32 copy (conversion/funasr_nano/export_encoder.py)."""
    import torch.nn.utils.parametrize as P

    for m in list(module.modules()):
        for name, param in list(m.named_parameters(recurse=False)):
            setattr(m, name, nn.Parameter(param.detach().half(), requires_grad=False))
            P.register_parametrization(m, name, _Upcast(), unsafe=True)
    return module


def pad_audio(wav: np.ndarray, frames: int) -> np.ndarray:
    n = frames * CODEC_FRAME
    assert wav.size <= n, (wav.size, n)
    out = np.zeros(n, np.float32)
    out[: wav.size] = wav
    return out


def compare_codes(got: np.ndarray, ref: np.ndarray) -> dict:
    T = ref.shape[1]
    got = got[:, :T]
    per = [(float((got[i] == ref[i]).mean())) for i in range(NUM_CODEBOOKS)]
    return {"frames": T, "exact_frames": int(np.all(got == ref, axis=0).sum()), "per_codebook_match": per,
            "codebook0_match": per[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=216, help="bucket in codec frames (216 = 10.03 s)")
    ap.add_argument("--dtype", choices=["fp16", "fp32", "fp16w32"], default="fp16w32")
    ap.add_argument("--out", default=str(EXPORTS))
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    snap = Path(hf_snapshot(HF_ID, revision=REVISION))
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(snap), trust_remote_code=True)
    pub = load_publisher_codec(snap, cfg)
    enc = build_codec_encoder(pub, args.frames)

    # ---- eager fp32 parity on the clone fixtures' reference clips ----
    fixtures = ["clone_en_1", "clone_ja_1", "clone_en_2", "clone_ja_2"]
    refs = {}
    for name in fixtures:
        orc = np.load(WORK / "oracle" / f"{name}.npz")
        wav, ref_codes = orc["ref_wav"].astype(np.float32), orc["ref_codes"]
        if wav.size > args.frames * CODEC_FRAME:
            print(f"[{name}] reference {wav.size / 44100:.1f} s exceeds the bucket, skipped")
            continue
        refs[name] = (wav, ref_codes)
        with torch.inference_mode():
            got = enc(torch.from_numpy(pad_audio(wav, args.frames)).reshape(1, 1, -1)).numpy()[0]
        r = compare_codes(got, ref_codes)
        print(f"[eager fp32 {name}] {r['exact_frames']}/{r['frames']} frames exact; per-codebook match "
              f"{[round(p, 3) for p in r['per_codebook_match']]}", flush=True)
    if args.skip_export:
        return

    # ---- export ----
    from export_audio8 import convert, save

    if args.dtype == "fp16":
        enc_x = enc.to(torch.float16)
        in_dtype = torch.float16
    elif args.dtype == "fp16w32":
        enc_x = fp16_storage_fp32_compute(enc)
        in_dtype = torch.float32
    else:
        enc_x = enc
        in_dtype = torch.float32

    class Wrap(nn.Module):
        def __init__(self, m, dt):
            super().__init__()
            self.m = m
            self.dt = dt

        def forward(self, audio):
            return self.m(audio.to(self.dt))

    w = Wrap(enc_x, in_dtype).eval()
    ref = {"audio": torch.zeros(1, 1, args.frames * CODEC_FRAME, dtype=torch.float32)}
    t0 = time.time()
    prog = convert([{"module": w, "ref": ref, "input_names": ("audio",), "output_names": ("codes",), "name": "main"}])
    print(f"[convert] encoder {args.dtype} t{args.frames} in {time.time() - t0:.0f}s", flush=True)
    path = Path(args.out) / f"audio8_codec_encoder_{args.dtype}_t{args.frames}.aimodel"
    save(prog, path)

    # ---- engine gate (GPU) ----
    import coreai.runtime as rt

    async def gate():
        opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
        m = await rt.AIModel.load(str(path), opts)
        fn = m.load_function("main")
        for name, (wav, ref_codes) in refs.items():
            t1 = time.time()
            r = await fn(inputs={"audio": rt.NDArray(np.ascontiguousarray(pad_audio(wav, args.frames).reshape(1, 1, -1)))})
            got = r["codes"].numpy()[0]
            res = compare_codes(got, ref_codes)
            print(f"[engine gpu {name}] {res['exact_frames']}/{res['frames']} frames exact; per-codebook "
                  f"{[round(p, 3) for p in res['per_codebook_match']]} ({time.time() - t1:.2f}s)", flush=True)

    asyncio.run(gate())


if __name__ == "__main__":
    main()
