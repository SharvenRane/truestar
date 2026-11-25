"""Metal kernels for the chunked recurrence (docs/LINEAR_ATTENTION.md), Apple GPUs through MPS.

The same two kernel split as the Triton kernels (triton_kernels.py), written for Metal from the
identities in chunked.py:

* ``la_prep`` (delta rule only), one threadgroup per (chunk, batch and head): the chunk's unit
  lower triangular system ``(I + M)`` is inverted by forward substitution in threadgroup memory,
  and ``Wk = (I + M)^-1 beta (k * exp(c))``, ``Wv = (I + M)^-1 beta V`` are stored as float32.
* ``la_fwd``, one threadgroup per (block of value columns, batch and head): walks the chunks in
  order and keeps its [dk, BV] slice of the state in threadgroup memory. Per chunk: the masked and
  decayed scores ``A = q k^T``, ``U`` (the values, or ``Wv - Wk S`` for the delta rule), the output
  ``(q S) e^c + A U`` and the state update ``S e^l + k^T (U e^(l - c))``.

Every sum is a serial loop in index order by one thread (no atomics, no split along the sequence),
so the bits depend on the inputs and the chunk only: not on the batch, the number of groups or the
run. Inputs are read as float32 (16 bit inputs are widened once on the device); the output is
written in float32 and returned in v's dtype, the state in float32. Kernels are compiled once per
(rule, decay, chunk, head dims) with ``torch.mps.compile_shader``.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import torch

from .reference import LinAttnError

CHUNK = 16            # the chunk of the Metal kernels (also the deterministic chunk of the Triton ones)
THREADS = 128         # threads per threadgroup
MAX_DK = 128
MAX_DV = 256
TG_BYTES = 28 * 1024  # threadgroup memory the forward kernel may use (Metal allows 32 KB)
BV_CAP = 16           # value columns per group: fastest of 8, 16, 32, 64 on an M2 (more groups per head)

_lock = threading.Lock()
_libs: Dict[Tuple, object] = {}


def available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available() and hasattr(torch.mps, "compile_shader"))


def _bv(dk: int, dv: int) -> int:
    """Value columns per threadgroup: dv rounded up to a power of two, at most BV_CAP, halved until
    the state slice and the chunk tiles fit in threadgroup memory. A column of the state never mixes
    with another, so the width changes no bits; TRUESTAR_METAL_LA_BV sets the cap for measurements."""
    import os
    cap = int(os.environ.get("TRUESTAR_METAL_LA_BV", str(BV_CAP)))
    bv = 4
    while bv < min(cap, dv):
        bv *= 2
    while bv > 4 and 4 * (dk * bv + 2 * CHUNK * dk + CHUNK * CHUNK + CHUNK * bv + 2 * CHUNK) > TG_BYTES:
        bv //= 2
    return bv


def kernel_supports(kind: str, dk: int, dv: int, dtype: torch.dtype) -> Optional[str]:
    """None when the Metal kernels run this call, else the reason they do not."""
    if kind == "channel":
        return "per key channel decay runs in the chunked PyTorch form only (no Metal kernel)"
    if dk > MAX_DK or dv > MAX_DV:
        return f"head dims above dk {MAX_DK} or dv {MAX_DV} (dk={dk}, dv={dv}) are not compiled for Metal"
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return f"the Metal kernels take float32, float16 or bfloat16, got {dtype}"
    return None


def _source(delta: bool, has_g: bool, has_s0: bool, dk: int, dv: int, bv: int) -> str:
    C, NT = CHUNK, THREADS
    nvb = -(-dv // bv)
    flags = (f"#define DELTA {int(delta)}\n#define HAS_G {int(has_g)}\n#define HAS_S0 {int(has_s0)}\n"
             f"#define C {C}\n#define NT {NT}\n#define DK {dk}\n#define DV {dv}\n#define BV {bv}\n#define NVB {nvb}\n")
    return "#include <metal_stdlib>\nusing namespace metal;\n#pragma METAL fp math_mode(safe)\n" + flags + r"""
// index of element (b, t, h, c) of a contiguous [B, T, H, D] tensor
#define IX(b, t, h, c, T, H, D) ((((b) * (T) + (t)) * (H) + (h)) * (D) + (c))

kernel void la_prep(device const float* k [[buffer(0)]], device const float* v [[buffer(1)]],
                    device const float* g [[buffer(2)]], device const float* beta [[buffer(3)]],
                    device float* wk [[buffer(4)]], device float* wv [[buffer(5)]],
                    device const int* P [[buffer(6)]],
                    uint3 tg [[threadgroup_position_in_grid]], uint3 lt [[thread_position_in_threadgroup]]) {
    const int T = P[0], H = P[1], NCH = P[2];
    const int lid = (int)lt.x;
    const int n = (int)tg.x % NCH, bh = (int)tg.x / NCH, b = bh / H, h = bh % H;
    const int t0 = n * C;
    threadgroup float Kc[C * DK];
    threadgroup float Vc[C * DV];
    threadgroup float M[C * C];
    threadgroup float X[C * C];
    threadgroup float cum[C];
    threadgroup float bt[C];
    for (int e = lid; e < C * DK; e += NT) {
        const int t = t0 + e / DK;
        Kc[e] = t < T ? k[IX(b, t, h, e % DK, T, H, DK)] : 0.0f;
    }
    for (int e = lid; e < C * DV; e += NT) {
        const int t = t0 + e / DV;
        Vc[e] = t < T ? v[IX(b, t, h, e % DV, T, H, DV)] : 0.0f;
    }
    if (lid == 0) {
        float c = 0.0f;
        for (int r = 0; r < C; ++r) {
            const int t = t0 + r;
#if HAS_G
            c = c + (t < T ? g[(b * T + t) * H + h] : 0.0f);
#endif
            cum[r] = c;
            bt[r] = t < T ? beta[(b * T + t) * H + h] : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = lid; e < C * C; e += NT) {
        const int i = e / C, j = e % C;
        float m = 0.0f;
        if (j < i) {
            float s = 0.0f;
            for (int c = 0; c < DK; ++c) s = s + Kc[i * DK + c] * Kc[j * DK + c];
            m = s * precise::exp(cum[i] - cum[j]) * bt[i];
        }
        M[e] = m;
        X[e] = i == j ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // (I + M)^-1 by forward substitution: row i = e_i - sum_{j < i} M_ij row_j
    for (int i = 1; i < C; ++i) {
        if (lid < C) {
            float s = 0.0f;
            for (int j = 0; j < i; ++j) s = s + M[i * C + j] * X[j * C + lid];
            X[i * C + lid] = (i == lid ? 1.0f : 0.0f) - s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int base = (bh * NCH + n) * C;
    for (int e = lid; e < C * DK; e += NT) {
        const int r = e / DK, c = e % DK;
        float s = 0.0f;
        for (int j = 0; j < C; ++j) s = s + X[r * C + j] * (Kc[j * DK + c] * precise::exp(cum[j]) * bt[j]);
        wk[(base + r) * DK + c] = s;
    }
    for (int e = lid; e < C * DV; e += NT) {
        const int r = e / DV, d = e % DV;
        float s = 0.0f;
        for (int j = 0; j < C; ++j) s = s + X[r * C + j] * (Vc[j * DV + d] * bt[j]);
        wv[(base + r) * DV + d] = s;
    }
}

kernel void la_fwd(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                   device const float* v [[buffer(2)]], device const float* g [[buffer(3)]],
                   device const float* wk [[buffer(4)]], device const float* wv [[buffer(5)]],
                   device const float* s0 [[buffer(6)]], device float* o [[buffer(7)]],
                   device float* sf [[buffer(8)]], device const int* P [[buffer(9)]],
                   uint3 tg [[threadgroup_position_in_grid]], uint3 lt [[thread_position_in_threadgroup]]) {
    const int T = P[0], H = P[1], NCH = P[2];
    const float scale = as_type<float>(P[3]);
    const int lid = (int)lt.x;
    const int vb = (int)tg.x % NVB, bh = (int)tg.x / NVB, b = bh / H, h = bh % H;
    threadgroup float S[DK * BV];
    threadgroup float Qs[C * DK];
    threadgroup float Ks[C * DK];
    threadgroup float A[C * C];
    threadgroup float U[C * BV];
    threadgroup float cum[C];
    threadgroup float rsu[C];
    for (int e = lid; e < DK * BV; e += NT) {
        const int col = vb * BV + e % BV;
#if HAS_S0
        S[e] = col < DV ? s0[(bh * DK + e / BV) * DV + col] : 0.0f;
#else
        S[e] = 0.0f;
#endif
    }
    for (int n = 0; n < NCH; ++n) {
        const int t0 = n * C;
        for (int e = lid; e < C * DK; e += NT) {
            const int t = t0 + e / DK;
            Qs[e] = t < T ? q[IX(b, t, h, e % DK, T, H, DK)] : 0.0f;
            Ks[e] = t < T ? k[IX(b, t, h, e % DK, T, H, DK)] : 0.0f;
        }
        if (lid == 0) {
            float c = 0.0f;
            for (int r = 0; r < C; ++r) {
#if HAS_G
                const int t = t0 + r;
                c = c + (t < T ? g[(b * T + t) * H + h] : 0.0f);
#endif
                cum[r] = c;
            }
            for (int r = 0; r < C; ++r) rsu[r] = HAS_G ? precise::exp(c - cum[r]) : 1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = lid; e < C * BV; e += NT) {
            const int r = e / BV, cc = e % BV, col = vb * BV + cc, t = t0 + r;
            float u = 0.0f;
            if (t < T && col < DV) {
#if DELTA
                const int base = (bh * NCH + n) * C + r;
                float s = 0.0f;
                for (int c = 0; c < DK; ++c) s = s + wk[base * DK + c] * S[c * BV + cc];
                u = wv[base * DV + col] - s;
#else
                u = v[IX(b, t, h, col, T, H, DV)];
#endif
            }
            U[e] = u;
        }
        for (int e = lid; e < C * C; e += NT) {
            const int i = e / C, j = e % C;
            float a = 0.0f;
            if (j <= i) {
                float s = 0.0f;
                for (int c = 0; c < DK; ++c) s = s + Qs[i * DK + c] * Ks[j * DK + c];
                a = HAS_G ? s * precise::exp(cum[i] - cum[j]) : s;
            }
            A[e] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = lid; e < C * BV; e += NT) {
            const int i = e / BV, cc = e % BV, col = vb * BV + cc, t = t0 + i;
            if (t < T && col < DV) {
                float qs = 0.0f;
                for (int c = 0; c < DK; ++c) qs = qs + Qs[i * DK + c] * S[c * BV + cc];
                if (HAS_G) qs = qs * precise::exp(cum[i]);
                float au = 0.0f;
                for (int j = 0; j <= i; ++j) au = au + A[i * C + j] * U[j * BV + cc];
                o[IX(b, t, h, col, T, H, DV)] = (qs + au) * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float el = HAS_G ? precise::exp(cum[C - 1]) : 1.0f;
        for (int e = lid; e < DK * BV; e += NT) {
            const int c = e / BV, cc = e % BV;
            float s = 0.0f;
            for (int j = 0; j < C; ++j) s = s + Ks[j * DK + c] * (U[j * BV + cc] * rsu[j]);
            S[e] = (HAS_G ? S[e] * el : S[e]) + s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int e = lid; e < DK * BV; e += NT) {
        const int col = vb * BV + e % BV;
        if (col < DV) sf[(bh * DK + e / BV) * DV + col] = S[e];
    }
}
"""


def _lib(key):
    with _lock:
        lib = _libs.get(key)
        if lib is None:
            lib = torch.mps.compile_shader(_source(*key))
            _libs[key] = lib
        return lib


def config(dk: int, dv: int) -> Dict[str, int]:
    return {"CHUNK": CHUNK, "BV": _bv(dk, dv), "threads": THREADS}


def forward(q, k, v, g, beta, *, rule: str, scale: float, initial_state, store_state: bool = True):
    """q, k, v: [B, T, H, d] on MPS, g: [B, T, H] or None, beta: [B, T, H] for the delta rule.
    Returns (o [B, T, H, dv] in v's dtype, final state [B, H, dk, dv] float32 or None)."""
    if not available():
        raise LinAttnError("[TS1061] the Metal linear attention kernels need an MPS device")
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    odt = v.dtype
    f32 = lambda x: None if x is None else x.to(torch.float32).contiguous()  # noqa: E731
    q, k, v, g, beta = f32(q), f32(k), f32(v), f32(g), f32(beta)
    s0 = f32(initial_state)
    dev = q.device
    bv = _bv(DK, DV)
    delta = rule == "delta"
    lib = _lib((delta, g is not None, s0 is not None, DK, DV, bv))
    NCH = -(-T // CHUNK)
    o = torch.empty(B, T, H, DV, dtype=torch.float32, device=dev)
    sf = torch.empty(B, H, DK, DV, dtype=torch.float32, device=dev)
    dummy = o
    P = torch.tensor([T, H, NCH, int(torch.tensor([float(scale)], dtype=torch.float32).view(torch.int32))],
                     dtype=torch.int32).to(dev)
    if NCH:
        if delta:
            wk = torch.empty(B * H * NCH * CHUNK * DK, dtype=torch.float32, device=dev)
            wv = torch.empty(B * H * NCH * CHUNK * DV, dtype=torch.float32, device=dev)
            n = NCH * B * H
            lib.la_prep(k, v, g if g is not None else dummy, beta, wk, wv, P,
                        threads=[n * THREADS, 1, 1], group_size=[THREADS, 1, 1])
        else:
            wk = wv = dummy
        n = -(-DV // bv) * B * H
        lib.la_fwd(q, k, v, g if g is not None else dummy, wk, wv, s0 if s0 is not None else dummy, o, sf, P,
                   threads=[n * THREADS, 1, 1], group_size=[THREADS, 1, 1])
    else:
        sf.copy_(s0 if s0 is not None else torch.zeros_like(sf))
    out = o if odt == torch.float32 else o.to(odt)
    return out, (sf if store_state else None)
