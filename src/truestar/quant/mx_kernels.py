"""MX block scaled matmul on CUDA through `tl.dot_scaled`, checked against the exact MX reference.

y[M, N] = dequant(A)[M, K] @ dequant(B)[N, K]^T for A and B in the same or different MX formats
(`truestar.quant.mx`): element codes one per byte, one UE8M0 scale per 32 inputs. The kernel is
one `tl.dot_scaled` per K tile with a float32 accumulator; Triton picks the MMA, and which formats
run natively on a GPU is measured, not assumed (`probe` and docs/QUANT.md).

Verdict (a) uses the bound of `truestar.quant.verify` (float32 accumulation of exact products).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from .formats import F64, FormatError
from .mx import BLOCK, MXTensor

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

TRITON_NAME = {"mxfp8_e4m3": "e4m3", "mxfp8_e5m2": "e5m2"}

if triton is not None:
    @triton.jit
    def _mx_kernel(a_ptr, sa_ptr, b_ptr, sb_ptr, y_ptr, M, N, K, KS,
                   FA: tl.constexpr, FB: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        rs = tl.arange(0, BK // 32)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(a_ptr + rm[:, None] * K + (k0 + rk[None, :]), mask=rm[:, None] < M, other=0)
            b = tl.load(b_ptr + rn[None, :] * K + (k0 + rk[:, None]), mask=rn[None, :] < N, other=0)
            sa = tl.load(sa_ptr + rm[:, None] * KS + (k0 // 32 + rs[None, :]), mask=rm[:, None] < M,
                         other=127)
            sb = tl.load(sb_ptr + rn[:, None] * KS + (k0 // 32 + rs[None, :]), mask=rn[:, None] < N,
                         other=127)
            acc = tl.dot_scaled(a, sa, FA, b, sb, FB, acc)
        tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a: MXTensor, b: MXTensor, BM: int = 64, BN: int = 64, BK: int = 128) -> torch.Tensor:
    """float32 y = dequant(a) @ dequant(b)^T on CUDA (a [M, K], b [N, K])."""
    if triton is None:
        raise FormatError("MX matmul needs Triton")
    for t in (a, b):
        if t.fmt not in TRITON_NAME:
            raise FormatError(f"MX matmul formats: {', '.join(TRITON_NAME)}; got {t.fmt}")
        if t.codes.device.type != "cuda":
            raise FormatError("MX matmul runs on CUDA tensors")
    M, K = a.codes.shape
    N, K2 = b.codes.shape
    if K != K2 or K % BK:
        raise FormatError(f"MX matmul needs equal K, a multiple of {BK}; got {K} and {K2}")
    ac = a.codes.to(torch.uint8).contiguous()
    bc = b.codes.to(torch.uint8).contiguous()
    sa = a.scale_codes.to(torch.uint8).contiguous()
    sb = b.scale_codes.to(torch.uint8).contiguous()
    y = torch.empty((M, N), device=ac.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _mx_kernel[grid](ac, sa, bc, sb, y, M, N, K, K // BLOCK, FA=TRITON_NAME[a.fmt],
                     FB=TRITON_NAME[b.fmt], BM=BM, BN=BN, BK=BK)
    return y


def verdict(a: MXTensor, b: MXTensor, y: torch.Tensor) -> float:
    """Worst |y - ref| over the verdict (a) bound (float32 output); <= 1 is a correct kernel."""
    from .verify import LAMBDA, U32

    A = a.dequantize()
    B = b.dequantize()
    ref = A @ B.T
    S = A.abs() @ B.abs().T
    K = A.shape[1]
    u = 2.0 ** -24
    tol = u * ref.abs() + LAMBDA * math.sqrt(K) * U32 * S * (1 + u) + 2.0 ** -150
    err = (y.to(F64) - ref).abs()
    return float((err / tol).max())


def probe(device: str = "cuda") -> Dict[str, Optional[str]]:
    """Which MX format pairs compile and pass verdict (a) here, and the MMA in the PTX."""
    from . import mx

    out: Dict[str, Optional[str]] = {}
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(64, 256, generator=g)
    w = torch.randn(64, 256, generator=g)
    for f in TRITON_NAME:
        try:
            qa = mx.quantize(x, f)
            qb = mx.quantize(w, f)
            qa = MXTensor(qa.fmt, qa.codes.to(device), qa.scale_codes.to(device), qa.rule)
            qb = MXTensor(qb.fmt, qb.codes.to(device), qb.scale_codes.to(device), qb.rule)
            y = matmul(qa, qb)
            r = verdict(qa, qb, y)
            out[f] = f"ratio {r:.3g}" + ("" if r <= 1 else " FAILS")
        except Exception as e:  # recorded, never hidden
            out[f] = f"error: {type(e).__name__}: {str(e).splitlines()[0][:160]}"
    return out
