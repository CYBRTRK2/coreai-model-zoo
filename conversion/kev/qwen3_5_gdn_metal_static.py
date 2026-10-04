# Copied from coreai_models/models/macos/qwen3_5_gdn_metal.py (the coreai-models overlay, unchanged there) for Kev round 16.
"""The fp32 GDN chunk-scan Metal kernel with every kernel edge a static shape (Kev round 16).

The overlay's kernel and module (`build_gdn_chunk_kernel` + `MetalGDNChunk`) take the conv-native activation
[conv_dim, S], g / beta [S, h] and the state; on the dynamic-S graph (round 14's D<cap>, query length 2..cap) the
first three carry the call length S. A Swift process on that graph grows its memory footprint on every call whose
length differs from the call before it, by about 3.5 MB + 0.12 MB per token of the call (round 15's probe), while the
static-S graph (K128) and calls of one length do not grow. Here the module zero-pads those three inputs in-graph to
chunk_max, so the kernel only ever sees [conv_dim, chunk_max] and [chunk_max, h]:

    static_io="slen" (V1)  the call's length rides along as SLEN [1] int32 and the MSL runs t < SLEN[0]
    static_io="pad"  (V2)  no length input: the MSL runs every one of the chunk_max steps. A padded step has g = 0
                           (decay exp(0) = 1), beta = 0 (the write term is 0) and q = k = v = 0, so it leaves the
                           state as it is

The padding is `torch.cat([x, zeros(chunk_max + 2 - S)])` cut back to chunk_max: the zeros' extent is never 0 or 1 for
S in 2..chunk_max (torch.export guards a 0 / 1 extent: `chunk_max + 1 - S` failed its `!= 1` guard at S = chunk_max), and
the cut is a slice whose static result shape the converter takes from the exported node (a concat of two dynamic
extents stays dynamic in the IR). A rank-1 SLEN, not a 0-d tensor: the runtime binds a 0-d tensor input
as a tensor handle that the generated `constant int&` parameter cannot read (coreai_torch/_torch_metal_kernel.py).

OUT stays [chunk_max, h*dv] (the module slices [:S]) and SNEW [h, dk, dv]: the graph's inputs, outputs and states are
the D graph's. The MSL body is the overlay's line for line except the first (the length S) and its comment;
`overlay_body_check` compares them.

Measured (round 16, Kev-0.8B "slen" at cap 512 against the dynamic-edge D512, Mac, Swift JIT): every call 0.9-1.4 ms
slower (16 tokens 13.35 vs 11.95 ms, 512 tokens 99.7 vs 98.85 ms), and a new call length adds 28-29 MB at 96 / 128 tokens
against 19-20 MB (75-77 MB at 512 for both): the static edges do not reduce the per-shape memory. Not shipped.
"""
from __future__ import annotations

import hashlib
import inspect

import torch
import torch.nn as nn

from coreai_torch import MetalParameter, TorchMetalKernel

STATIC_IO = ("slen", "pad")

_LENGTH_LINES = {
    "slen": "const uint S  = uint(SLEN[0]);              // the call's length (the inputs are zero-padded to chunk_max)",
    "pad": "const uint S  = MIXED.get_extent(0);       // = chunk_max: a padded step leaves the state as it is",
}

# DSL axes are reversed vs torch (DSL dim0 = torch innermost):
#   MIXED torch [conv_dim, chunk_max] -> DSL [chunk_max, conv_dim]:  MIXED[t, channel]
#   G/BETA torch [chunk_max, h]       -> DSL [h, chunk_max]:         G[hh, t]
#   S0/SNEW torch [h, dk, dv]         -> DSL [dv, dk, h]:            S0[c, d, hh]
#   OUT torch [chunk_max, h*dv]       -> DSL [h*dv, chunk_max]:      OUT[hh*dv + c, t]
#   SLEN torch [1] int32              -> DSL [1]:                    SLEN[0]
_GDN_CHUNK_SRC = """
    __LENGTH__
    const uint c  = gid.x;                      // value column (0..dv-1) — this thread's state column
    const uint hh = gid.y;                      // value head (one threadgroup per head)
    const uint kb  = (hh / __R__) * __DK__;     // k/q channel base (GVA: __R__ value heads per key head)
    const uint qch = kb + c;                    // this thread's staged q channel
    const uint kch = __KEYDIM__ + kb + c;       // this thread's staged k channel
    const uint vch = __VOFF__ + hh * __DV__ + c;   // this thread's v channel

    float st[__DK__];                            // state column [dk] for value-col c (fp32, persists over S)
    for (uint d = 0; d < __DK__; ++d) st[d] = float(S0[c, d, hh]);

    threadgroup float ksh[__DK__];               // raw k_t / q_t staged for the whole head each step
    threadgroup float qsh[__DK__];

    for (uint t = 0; t < S; ++t) {
        qsh[c] = float(MIXED[t, qch]);           // dv column-threads stage the dk-dim raw q_t / k_t
        ksh[c] = float(MIXED[t, kch]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float qsc = __QSCALE__;                  // dk^-0.5 (baked)
        float ksc = 1.0f;
        if (__L2__) {                            // qk l2-norm, in-kernel fp32 (every thread computes
            float qs = 0.0f, ks = 0.0f;          //  the same scalars from threadgroup memory)
            for (uint d = 0; d < __DK__; ++d) { qs += qsh[d] * qsh[d]; ks += ksh[d] * ksh[d]; }
            qsc *= rsqrt(qs + 1e-6f);
            ksc  = rsqrt(ks + 1e-6f);
        }

        float gt = float(G[hh, t]);              // per-head scalar decay logit
        float bt = float(BETA[hh, t]);
        float vc = float(MIXED[t, vch]);
        float ge = exp(gt);                      // g is the NEGATIVE log-decay -> multiplier exp(g)

        float kv = 0.0f;
        for (uint d = 0; d < __DK__; ++d) { st[d] *= ge; kv += st[d] * ksh[d]; }   // decay, then k^T state
        float kd = ksc * (vc - ksc * kv) * bt;   // ksc*delta: k_eff = ksc*ksh both in the dot and the write
        float oc = 0.0f;
        for (uint d = 0; d < __DK__; ++d) { st[d] += ksh[d] * kd; oc += st[d] * qsh[d]; }  // write, then q^T state
        OUT[hh * __DV__ + c, t] = TYPE(oc * qsc);
        threadgroup_barrier(mem_flags::mem_threadgroup);   // before next step overwrites ksh/qsh
    }
    for (uint d = 0; d < __DK__; ++d) SNEW[c, d, hh] = TYPE(st[d]);
"""


def overlay_body_check() -> dict:
    """The MSL body here against the overlay's `_GDN_CHUNK_SRC`: equal line for line after the first line (the length)."""
    from coreai_models.models.macos import qwen3_5_gdn_metal as overlay

    ours = _GDN_CHUNK_SRC.strip("\n").splitlines()
    theirs = overlay._GDN_CHUNK_SRC.strip("\n").splitlines()
    path = inspect.getsourcefile(overlay)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"overlay_file": path, "overlay_sha256": sha, "overlay_length_line": theirs[0].strip(),
            "body_equal_after_the_length_line": ours[1:] == theirs[1:], "body_lines": len(ours)}


def build_gdn_chunk_kernel_static(name: str = "qwen3_5_gdn_chunk_static", num_k: int = 16, num_v: int = 16,
                                  dk: int = 128, dv: int = 128, use_qk_l2_norm: bool = True, chunk_max: int = 512,
                                  static_io: str = "slen") -> TorchMetalKernel:
    """The overlay's kernel on static edges: MIXED [conv_dim, chunk_max], G / BETA [chunk_max, h] (+ SLEN [1] int32 for
    static_io="slen"), S0 [h, dk, dv] -> OUT [chunk_max, h*dv] (rows [0:S] written), SNEW [h, dk, dv]."""
    if static_io not in STATIC_IO:
        raise ValueError(f"static_io {static_io!r}: one of {STATIC_IO}")
    if dk > dv:
        raise ValueError(f"staging needs dk <= dv (dv column-threads stage dk dims): dk={dk} dv={dv}")
    if num_v % num_k:
        raise ValueError(f"GVA needs num_v % num_k == 0: num_v={num_v} num_k={num_k}")

    def _shapes(MIXED: torch.Tensor, S0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, dv_ = S0.shape[0], S0.shape[2]
        return MIXED.new_zeros(chunk_max, h * dv_), S0.clone()

    # Shape-inference references for torch.export (the numerics are the MSL on the engine); one per input list.
    def _torch_defn_slen(MIXED: torch.Tensor, G: torch.Tensor, BETA: torch.Tensor, S0: torch.Tensor,
                         SLEN: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _shapes(MIXED, S0)

    def _torch_defn_pad(MIXED: torch.Tensor, G: torch.Tensor, BETA: torch.Tensor,
                        S0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _shapes(MIXED, S0)

    src = (_GDN_CHUNK_SRC
           .replace("__LENGTH__", _LENGTH_LINES[static_io])
           .replace("__KEYDIM__", str(num_k * dk))
           .replace("__VOFF__", str(2 * num_k * dk))
           .replace("__R__", str(num_v // num_k))
           .replace("__DK__", str(dk))
           .replace("__DV__", str(dv))
           .replace("__QSCALE__", f"{dk ** -0.5!r}f")
           .replace("__L2__", "1" if use_qk_l2_norm else "0"))
    inputs = ["MIXED", "G", "BETA", "S0"] + (["SLEN"] if static_io == "slen" else [])
    return TorchMetalKernel(
        name,
        input_names=inputs,
        result_names=["OUT", "SNEW"],
        src=src,
        torch_defn=_torch_defn_slen if static_io == "slen" else _torch_defn_pad,
        metal_params=[MetalParameter("gid", "uint2", "thread_position_in_grid")],
        template_dtypes={"MIXED": "TYPE"},
    )


class MetalGDNChunkStatic(nn.Module):
    """`MetalGDNChunk` with the kernel's inputs zero-padded in-graph to chunk_max. ``forward(conv, g, beta, S0)`` with
    conv [b, conv_dim, S] (post-silu, channel-major), g / beta [b, S, h], S0 [b, h, dk, dv] -> (out [b, S, h, dv],
    Snew [b, h, dk, dv]), as the overlay's module."""

    coreai_externalize_specs: tuple = ()

    def __init__(self, kernel: TorchMetalKernel, chunk_max: int, static_io: str) -> None:
        super().__init__()
        if static_io not in STATIC_IO:
            raise ValueError(f"static_io {static_io!r}: one of {STATIC_IO}")
        self.kernel = kernel
        self.chunk_max = chunk_max
        self.static_io = static_io

    def forward(self, conv, g, beta, S0):
        b, cdim, S = conv.shape
        nh = g.shape[2]
        h, dk, dv = S0.shape[1], S0.shape[2], S0.shape[3]
        cap = self.chunk_max
        # zero-pad the time axis by cap + 2 - S steps (an extent of 2..cap), then cut to cap: static [cap] edges
        conv_p = torch.cat([conv, conv.new_zeros(b, cdim, cap + 2 - S)], dim=2)[0, :, :cap]   # [conv_dim, cap]
        g_p = torch.cat([g, g.new_zeros(b, cap + 2 - S, nh)], dim=1)[0, :cap]                 # [cap, h]
        beta_p = torch.cat([beta, beta.new_zeros(b, cap + 2 - S, nh)], dim=1)[0, :cap]
        args = [conv_p, g_p, beta_p, S0[0]]
        if self.static_io == "slen":
            args.append(torch.scalar_tensor(S, dtype=torch.int32).reshape(1))   # SLEN [1] int32 = the call's S
        out, snew = self.kernel(
            *args, threads_per_grid=(dv, h, 1), threads_per_thread_group=(dv, 1, 1),
            result_shapes=[[cap, h * dv], [h, dk, dv]])
        out = out[:S].reshape(1, S, h, dv)   # rows [0:S] valid (S dynamic); reshape, not transpose
        return out, snew.unsqueeze(0)        # [1,h,dk,dv]
