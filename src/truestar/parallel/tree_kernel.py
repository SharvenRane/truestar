"""One Triton kernel for the exact leaf of the canonical reduction tree (tree.py) on CUDA.

The exact leaf sums a group of `gs` products a[m, k] * b[k, n] by the pairwise tree of `ptree`: level
by level, entries (2i, 2i + 1) added, an odd last one carried up. Written with torch ops it first
materializes every product ([rows, gs, N] at a time) and then launches one add per level, which made
an nn.Linear forward and backward 21x to 174x slower than F.linear (docs/PARALLEL.md).

Here one program computes a BM x BN tile of the sums of one aligned chunk of C consecutive products
(C a power of two that divides gs): it loads a [BM, C] piece of `a` and a [C, BN] piece of `b`, forms
the products in registers and adds them pairwise over log2(C) levels. Those are exactly the first
log2(C) levels of `ptree` over the group (every level count there is even, as C divides gs, so nothing
is carried), so `ptree` over the gs / C chunk sums finishes the same tree: same additions, same order,
same bits. Every multiply and add is a single IEEE round to nearest operation (PTX `mul.rn.f32`,
`add.rn.f32`, no flush to zero), never a fused multiply add, as in the torch arm.

Used by `contract_groups(..., leaf="exact")` for CUDA float32 tensors when Triton is importable and gs
is even; `TRUESTAR_TREE_KERNEL=0` turns it off (the torch arm then runs, for comparison).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

_KERNEL = None
_FAILED: Optional[str] = None


def chunk_width(gs: int, cap: int = 64) -> int:
    """Largest power of two up to `cap` that divides gs (1 when gs is odd: no kernel)."""
    c = 1
    while c * 2 <= cap and gs % (c * 2) == 0:
        c *= 2
    return c


def _build():
    import triton
    import triton.language as tl

    # PTX mul.rn.f32 and add.rn.f32: one rounding each, never contracted into an fma, and, without
    # .ftz, subnormal inputs and results kept (libdevice's __nv_fmul_rn follows Triton's flush to
    # zero setting, which flushed subnormal products to zero in a first version)
    @triton.jit
    def mul_rn(x, y):
        return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [x, y], dtype=tl.float32,
                                         is_pure=True, pack=1)

    @triton.jit
    def add_rn(x, y):
        return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f", [x, y], dtype=tl.float32,
                                         is_pure=True, pack=1)

    @triton.jit
    def chunk_sums(a_ptr, b_ptr, o_ptr, M, N, sam, sak, sbk, sbn, per, gs, groups, nch,
                   C: tl.constexpr, LOG: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        pid = tl.program_id(0)
        q = tl.program_id(1)                     # flat (part, group, chunk)
        tiles_n = tl.cdiv(N, BN)
        tm = pid // tiles_n
        tn = pid % tiles_n
        c = q % nch
        pg = q // nch
        g = pg % groups
        p = pg // groups
        k0 = p * per + g * gs + c * C
        rm = tm * BM + tl.arange(0, BM)
        rn = tn * BN + tl.arange(0, BN)
        rk = k0 + tl.arange(0, C)
        a = tl.load(a_ptr + rm[:, None] * sam + rk[None, :] * sak, mask=rm[:, None] < M, other=0.0)   # [BM, C]
        bt = tl.load(b_ptr + rk[None, :] * sbk + rn[:, None] * sbn, mask=rn[:, None] < N, other=0.0)  # [BN, C]
        a3 = tl.broadcast_to(a[:, None, :], (BM, BN, C))
        b3 = tl.broadcast_to(bt[None, :, :], (BM, BN, C))
        x = mul_rn(a3, b3)
        for i in tl.static_range(LOG):
            x = tl.reshape(x, (BM, BN, C >> (i + 1), 2))
            lo, hi = tl.split(x)
            x = add_rn(lo, hi)
        s = tl.reshape(x, (BM, BN))
        o = o_ptr + q * M * N + rm[:, None] * N + rn[None, :]
        tl.store(o, s, mask=(rm[:, None] < M) & (rn[None, :] < N))

    return chunk_sums


def available() -> bool:
    global _KERNEL, _FAILED
    if os.environ.get("TRUESTAR_TREE_KERNEL", "1") == "0":
        return False
    if _KERNEL is not None:
        return True
    if _FAILED is not None:
        return False
    try:
        _KERNEL = _build()
        return True
    except Exception as exc:  # noqa: BLE001 - no Triton, or one without libdevice: the torch arm runs
        _FAILED = f"{type(exc).__name__}: {exc}"
        return False


def usable(a: torch.Tensor, b: torch.Tensor, gs: int) -> bool:
    return (a.is_cuda and b.is_cuda and a.dtype == torch.float32 and b.dtype == torch.float32
            and chunk_width(gs) >= 2 and available())


def chunk_group_sums(a: torch.Tensor, b: torch.Tensor, parts: int, groups: int) -> torch.Tensor:
    """a [M, L], b [L, N] (any strides): the chunk sums [parts, groups, gs / C, M, N] of the canonical
    tree, C = chunk_width(gs). `ptree` over dim 2 gives the group sums."""
    m, length = a.shape
    n = b.shape[1]
    per = length // parts
    gs = per // groups
    c = chunk_width(gs)
    nch = gs // c
    log = c.bit_length() - 1
    out = torch.empty((parts, groups, nch, m, n), dtype=a.dtype, device=a.device)
    bm, bn = 16, 16
    warps = 4 if c <= 32 else 8
    grid = (((m + bm - 1) // bm) * ((n + bn - 1) // bn), parts * groups * nch)
    _KERNEL[grid](a, b, out, m, n, a.stride(0), a.stride(1), b.stride(0), b.stride(1), per, gs, groups, nch,
                  C=c, LOG=log, BM=bm, BN=bn, num_warps=warps, enable_fp_fusion=False)
    return out
