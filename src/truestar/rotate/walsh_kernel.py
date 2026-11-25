"""Fast Walsh transform on the GPU: y = (x @ H) * scale along the last dim, H the Sylvester
Hadamard matrix of a power of two order D up to 256, for online rotations (decode queries and
outputs of a rotated KV cache).

One program per block of rows; each row runs the same log2(D) float32 butterflies (add and
subtract), so a row's bits never depend on the other rows of the call (batch invariant) and every
run gives the same bits. The first call per (D, input dtype, output dtype, scale) is compared with
float64 ``walsh`` of the same rows; a mismatch raises WalshKernelError (TS1023). Measured relative
error on float16 rows: 0 (the butterflies of float16 inputs are exact in float32 for D up to 256
unless a row's exponents spread over more than 2^13).
"""

from __future__ import annotations

import math

import torch

from .hadamard import RotationError, walsh


class WalshKernelError(AssertionError):
    """TS1023: the fast Walsh transform kernel disagrees with the float64 transform."""
    code = "TS1023"


_K = None
_VERIFIED = set()
BM = 16


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def stage(x, BM: tl.constexpr, D: tl.constexpr, H: tl.constexpr):
        # one butterfly level: pairs (i, i + H) inside blocks of 2 H become (a + b, a - b)
        if H < D:
            x = tl.reshape(x, (BM, D // (2 * H), 2, H))
            x = tl.permute(x, (0, 1, 3, 2))
            a, b = tl.split(x)
            x = tl.join(a + b, a - b)
            x = tl.permute(x, (0, 1, 3, 2))
            x = tl.reshape(x, (BM, D))
        return x

    @triton.jit
    def fwht_rows(x_ptr, y_ptr, M, sx, sy, scale, D: tl.constexpr, BM: tl.constexpr):
        pid = tl.program_id(0)
        rm = pid * BM + tl.arange(0, BM)
        rd = tl.arange(0, D)
        msk = rm[:, None] < M
        x = tl.load(x_ptr + rm[:, None] * sx + rd[None, :], mask=msk, other=0.0).to(tl.float32)
        x = stage(x, BM, D, 1)
        x = stage(x, BM, D, 2)
        x = stage(x, BM, D, 4)
        x = stage(x, BM, D, 8)
        x = stage(x, BM, D, 16)
        x = stage(x, BM, D, 32)
        x = stage(x, BM, D, 64)
        x = stage(x, BM, D, 128)
        y = x * scale
        tl.store(y_ptr + rm[:, None] * sy + rd[None, :], y.to(y_ptr.dtype.element_ty), mask=msk)

    return fwht_rows


def fwht(x: torch.Tensor, scale: float = 1.0, out_dtype=None, verify: bool = True) -> torch.Tensor:
    """(x @ H) * scale along the last dim on CUDA; H is the +1 / -1 Sylvester matrix of order
    D = x.shape[-1] (a power of two, 2 to 256). ``scale`` should be a power of two (1 / D undoes
    a transform exactly), so it adds no rounding."""
    global _K
    D = x.shape[-1]
    if x.device.type != "cuda":
        raise RotationError("[TS1022] fwht runs on a CUDA tensor")
    if D < 2 or D > 256 or D & (D - 1):
        raise RotationError(f"[TS1022] fwht needs a power of two last dim from 2 to 256, got {D}")
    out_dtype = out_dtype or x.dtype
    x2 = x.reshape(-1, D)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    M = x2.shape[0]
    y = torch.empty(M, D, dtype=out_dtype, device=x.device)
    if _K is None:
        _K = _kernel()
    if M:
        with torch.cuda.device(x.device):
            _K[(-(-M // BM),)](x2, y, M, x2.stride(0), y.stride(0), float(scale), D=D, BM=BM, num_warps=4)
    key = (D, str(x.dtype), str(out_dtype), float(scale))
    if verify and M and key not in _VERIFIED:
        ref = walsh(x2, normalize=False) * scale
        err = float((y.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-300))
        u = {torch.float16: 2.0 ** -11, torch.bfloat16: 2.0 ** -8}.get(out_dtype, 2.0 ** -24)
        tol = 4 * u + 2.0 ** -24 * math.log2(D)
        if not (err <= tol):
            raise WalshKernelError(f"[TS1023] fwht (D {D}, {x.dtype} to {out_dtype}): relative error "
                                   f"{err:.3e} against the float64 transform, tolerance {tol:.1e}")
        _VERIFIED.add(key)
    return y.reshape(*x.shape[:-1], D)
