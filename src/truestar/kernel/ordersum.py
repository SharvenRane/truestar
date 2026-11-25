"""Order free float32 sums: the same bits for any summation order, split, tile or SM count.

Docs: docs/ORDER_FREE_SUM.md. Author: Sharven Rane, 2026-11-24.

A float32 sum normally depends on the order of its additions, so a reduction gives other bits when
its block size, split count or number of programs changes. An order free sum (``osum``) instead
turns every term into an exact integer on one fixed point grid chosen from the row, adds the
integers (integer addition is associative, so every order gives the same total), and rounds the
total to float32 once.

The grid ("anchor"). Let ``emax`` be the largest biased exponent field among the finite terms of
the row (1 for an all zero or subnormal row). Every finite term is below ``2**(emax - 126)`` in
magnitude. The grid step is ``2**L`` with ``L = emax - 126 - PRECISION_BITS``, so each term is kept
to ``PRECISION_BITS`` (39) bits below the top of the row's range: a term ``x`` becomes the integer
``q = round_half_even(x / 2**L)``, ``|q| < 2**39``. ``emax`` is a maximum, which is exact and order
free too, so the grid itself does not depend on the order.

Exactness of the integer sum: ``MAX_TERMS`` (2**24 - 1) terms of magnitude below ``2**39`` add up to
less than ``2**63``, so the int64 total never overflows and is exact whatever the order. The only
roundings are one per term onto the grid (at most ``2**(L-1)`` each, so at most
``max|x| * 2**-39`` per term) and the final conversion to float32 (half an ulp of the result, plus a
second rounding only when the result is subnormal). For the same terms this is both independent of
the order and usually far more accurate than a float32 pairwise or sequential sum.

Special values: any NaN term, or both +inf and -inf, gives NaN; otherwise any +inf gives +inf and
any -inf gives -inf (counted as integers, so also order free). A zero total is +0.0 even when every
term is -0.0 (IEEE would give -0.0).

This module holds the definition. It is computed three ways that must agree bit for bit:

* :func:`order_free_sum` with torch integer ops on any device (CPU, CUDA), with optional chunking
  and permutation to show order freedom;
* :func:`order_free_sum_numpy` in NumPy (used by tests as an independent second implementation);
* the kernel IR reduction ``osum`` (``K.order_free_sum(axis, body)``), emitted by the Triton and
  numba backends (kernel/triton_backend.py, kernel/cpu_backend.py) with the same integer steps.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

PRECISION_BITS = 39            # bits kept below the top of the row's range
MAX_TERMS = (1 << 24) - 1      # n * 2**39 < 2**63: the int64 total cannot overflow
_SHIFT_UP = PRECISION_BITS - 24  # 15: a term at the row's top exponent is shifted left by this


class OrderFreeSumError(ValueError):
    """An order free sum cannot run as asked (too many terms, a dtype it does not take)."""

    code = "TS1080"


# --------------------------------------------------------------------------- torch implementation


def _fields(x):
    """(bits, biased exponent field, effective exponent >= 1, integer significand) of float32 x."""
    import torch

    bits = x.contiguous().view(torch.int32)
    e = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    ee = torch.clamp(e, min=1)
    m = torch.where(e > 0, mant | 0x800000, mant).to(torch.int64)
    return bits, e, ee, m


def anchor(x, dim: int = -1, keepdim: bool = True):
    """The row's anchor exponent emax (int32): largest exponent field among finite terms, >= 1."""
    import torch

    _, e, ee, _ = _fields(x)
    ee = torch.where(e == 0xFF, torch.ones_like(ee), ee)
    if x.shape[dim] == 0:
        shape = list(x.shape)
        if keepdim:
            shape[dim] = 1
        else:
            del shape[dim]
        return torch.ones(shape, dtype=torch.int32, device=x.device)
    return ee.amax(dim=dim, keepdim=keepdim)


def quantize(x, emax):
    """Integer grid values q (int64) of float32 terms x for the anchor emax (broadcast to x), and
    the special value masks (nan, +inf, -inf)."""
    import torch

    bits, e, ee, m = _fields(x)
    sh = (ee - emax + _SHIFT_UP).to(torch.int64)
    left = m << torch.clamp(sh, min=0)
    s = torch.clamp(-sh, min=1, max=62)
    hi = m >> s
    rem = m & ((torch.ones_like(m) << s) - 1)
    half = torch.ones_like(m) << (s - 1)
    up = (rem > half) | ((rem == half) & ((hi & 1) == 1))
    right = hi + up.to(torch.int64)
    q = torch.where(sh >= 0, left, right)
    fin = e != 0xFF
    q = torch.where(fin, q, torch.zeros_like(q))
    q = torch.where(bits < 0, -q, q)
    nan = (e == 0xFF) & ((bits & 0x7FFFFF) != 0)
    pinf = bits == 0x7F800000
    ninf = bits == -8388608      # 0xFF800000 as int32
    return q, nan, pinf, ninf


def finish(total, emax, nan, pinf, ninf):
    """Round the exact integer total on the grid of emax to float32 (counts are int64 tensors)."""
    import torch

    f = total.to(torch.float32)
    L = emax.to(torch.int64) - 126 - PRECISION_BITS
    low = L < -126
    a = torch.where(low, L + 64, L)
    scale_a = ((a + 127) << 23).to(torch.int32).view(torch.float32)
    r = f * scale_a
    r = torch.where(low, r * (2.0 ** -64), r)       # float32 times an exact power of two
    r = torch.where(total == 0, 0.0, r)
    r = torch.where(ninf > 0, float("-inf"), r)
    r = torch.where(pinf > 0, float("inf"), r)
    return torch.where((nan > 0) | ((pinf > 0) & (ninf > 0)), float("nan"), r)


def order_free_sum(x, dim: int = -1, keepdim: bool = False, chunks: Optional[int] = None,
                   perm=None):
    """Order free sum of a float32 tensor along dim, with torch integer ops on x's device.

    chunks and perm only exist to show order freedom: the terms are permuted by perm (an index
    tensor along dim), cut into `chunks` partial sums (each an exact int64 partial on the shared
    grid), and the partials added in reverse order. Every choice gives the same bits."""
    import torch

    if x.dtype != torch.float32:
        raise OrderFreeSumError(f"order free sums take float32 terms, got {x.dtype} "
                                "(widen half precision terms to float32 first, which is exact)")
    dim = dim % x.dim() if x.dim() else 0
    n = x.shape[dim]
    if n > MAX_TERMS:
        raise OrderFreeSumError(f"an order free sum takes at most {MAX_TERMS} terms per row "
                                f"(2**24 - 1), got {n}: the int64 total could overflow")
    if perm is not None:
        x = x.index_select(dim, perm.to(x.device))
    emax = anchor(x, dim, keepdim=True)
    q, nan, pinf, ninf = quantize(x, emax)
    if chunks and chunks > 1 and n:
        parts = [p.sum(dim=dim, keepdim=True) for p in torch.tensor_split(q, chunks, dim=dim)]
        total = parts[-1]
        for p in reversed(parts[:-1]):
            total = total + p
    else:
        total = q.sum(dim=dim, keepdim=True)
    cnt = lambda b: b.to(torch.int64).sum(dim=dim, keepdim=True)  # noqa: E731
    r = finish(total, emax, cnt(nan), cnt(pinf), cnt(ninf))
    return r if keepdim else r.squeeze(dim)


def error_bound(x, dim: int = -1):
    """Proven bound on |order_free_sum(x) - exact sum| per row, in float64, before the final
    rounding: n * 2**(L - 1) with L the grid exponent (the final rounding adds half an ulp of the
    float32 result, plus a second rounding only for subnormal results)."""
    import torch

    emax = anchor(x, dim, keepdim=False).to(torch.float64)
    n = x.shape[dim]
    return n * torch.pow(torch.tensor(2.0, dtype=torch.float64, device=x.device),
                         emax - 126 - PRECISION_BITS - 1)


# --------------------------------------------------------------------------- NumPy implementation


def order_free_sum_numpy(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """The same definition in NumPy (an independent implementation for the tests)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    axis = axis % x.ndim
    if x.shape[axis] > MAX_TERMS:
        raise OrderFreeSumError(f"at most {MAX_TERMS} terms per row")
    bits = x.view(np.int32).astype(np.int64)
    e = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    ee = np.maximum(e, 1)
    m = np.where(e > 0, mant | 0x800000, mant)
    fin = e != 0xFF
    if x.shape[axis] == 0:
        emax = np.ones(x.shape[:axis] + (1,) + x.shape[axis + 1:], np.int64)
    else:
        emax = np.where(fin, ee, 1).max(axis=axis, keepdims=True)
    sh = ee - emax + _SHIFT_UP
    q = np.zeros_like(m)
    up = sh >= 0
    q[up] = m[up] << sh[up]
    s = np.clip(-sh, 1, 62)
    hi = m >> s
    rem = m & ((np.int64(1) << s) - 1)
    half = np.int64(1) << (s - 1)
    rnd = hi + ((rem > half) | ((rem == half) & ((hi & 1) == 1)))
    q = np.where(up, q, rnd)
    q = np.where(fin, q, 0)
    q = np.where(bits < 0, -q, q)
    total = q.sum(axis=axis, keepdims=True)
    nan = ((e == 0xFF) & (mant != 0)).sum(axis=axis, keepdims=True)
    pinf = (bits == 0x7F800000).sum(axis=axis, keepdims=True)
    ninf = (bits == -8388608).sum(axis=axis, keepdims=True)
    f = total.astype(np.float32)
    L = emax - 126 - PRECISION_BITS
    low = L < -126
    a = np.where(low, L + 64, L)
    scale = ((a + 127) << 23).astype(np.int32).view(np.float32)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        r = f * scale
        r = np.where(low, r * np.float32(2.0 ** -64), r).astype(np.float32)
    r = np.where(total == 0, np.float32(0.0), r)
    r = np.where(ninf > 0, np.float32(-np.inf), r)
    r = np.where(pinf > 0, np.float32(np.inf), r)
    r = np.where((nan > 0) | ((pinf > 0) & (ninf > 0)), np.float32(np.nan), r)
    return np.squeeze(r.astype(np.float32), axis=axis)


# --------------------------------------------------------------------------- standalone CUDA sum

_CUDA = {}


def _cuda_kernels():
    if "anchor" in _CUDA:
        return _CUDA["anchor"], _CUDA["sum"]
    import triton
    import triton.language as tl

    @triton.jit
    def ts_osum_anchor(p_x, p_out, N, BLOCK: tl.constexpr, ITERS: tl.constexpr):
        pid = tl.program_id(0)
        best = tl.full((BLOCK,), 1, tl.int32)
        for it in range(ITERS):
            off = (pid * ITERS + it) * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(p_x + off, mask=off < N, other=0.0)
            bits = x.to(tl.int32, bitcast=True)
            e = (bits >> 23) & 255
            ee = tl.where(e == 255, 1, tl.maximum(e, 1))
            best = tl.maximum(best, ee)
        tl.store(p_out + pid, tl.max(best, axis=0))

    @triton.jit
    def ts_osum_part(p_x, p_anchor, p_out, N, NPROG, BLOCK: tl.constexpr, ITERS: tl.constexpr,
                     PB: tl.constexpr):
        pid = tl.program_id(0)
        pa = tl.arange(0, PB)
        emax = tl.max(tl.load(p_anchor + pa, mask=pa < NPROG, other=1), axis=0)
        acc = tl.zeros((BLOCK,), tl.int64)
        nan = tl.zeros((BLOCK,), tl.int32)
        pinf = tl.zeros((BLOCK,), tl.int32)
        ninf = tl.zeros((BLOCK,), tl.int32)
        for it in range(ITERS):
            off = (pid * ITERS + it) * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(p_x + off, mask=off < N, other=0.0)
            bits = x.to(tl.int32, bitcast=True)
            e = (bits >> 23) & 255
            mant = bits & 8388607
            ee = tl.maximum(e, 1)
            m = tl.where(e > 0, mant | 8388608, mant).to(tl.int64)
            sh = (ee - emax + 15).to(tl.int64)
            left = m << tl.maximum(sh, 0)
            s = tl.minimum(tl.maximum(-sh, 1), 62)
            hi = m >> s
            rem = m & ((tl.full(m.shape, 1, tl.int64) << s) - 1)
            half = tl.full(m.shape, 1, tl.int64) << (s - 1)
            up = (rem > half) | ((rem == half) & ((hi & 1) == 1))
            q = tl.where(sh >= 0, left, hi + up.to(tl.int64))
            q = tl.where(e == 255, 0, q)
            q = tl.where(bits < 0, -q, q)
            acc += q
            nan += ((e == 255) & (mant != 0)).to(tl.int32)
            pinf += (bits == 2139095040).to(tl.int32)
            ninf += (bits == -8388608).to(tl.int32)
        tl.store(p_out + pid * 4 + 0, tl.sum(acc, axis=0))
        tl.store(p_out + pid * 4 + 1, tl.sum(nan, axis=0).to(tl.int64))
        tl.store(p_out + pid * 4 + 2, tl.sum(pinf, axis=0).to(tl.int64))
        tl.store(p_out + pid * 4 + 3, tl.sum(ninf, axis=0).to(tl.int64))

    @triton.jit
    def ts_osum_finish(p_anchor, p_parts, p_out, NPROG, PB: tl.constexpr):
        pa = tl.arange(0, PB)
        ok = pa < NPROG
        emax = tl.max(tl.load(p_anchor + pa, mask=ok, other=1), axis=0)
        tot = tl.sum(tl.load(p_parts + pa * 4, mask=ok, other=0), axis=0)
        nan = tl.sum(tl.load(p_parts + pa * 4 + 1, mask=ok, other=0), axis=0)
        pinf = tl.sum(tl.load(p_parts + pa * 4 + 2, mask=ok, other=0), axis=0)
        ninf = tl.sum(tl.load(p_parts + pa * 4 + 3, mask=ok, other=0), axis=0)
        L = emax - 165
        low = L < -126
        sc = ((tl.where(low, L + 64, L) + 127) << 23).to(tl.float32, bitcast=True)
        r = tot.to(tl.float32) * sc
        r = tl.where(low, r * 5.421010862427522e-20, r)
        r = tl.where(tot == 0, 0.0, r)
        r = tl.where(ninf > 0, float("-inf"), r)
        r = tl.where(pinf > 0, float("inf"), r)
        r = tl.where((nan > 0) | ((pinf > 0) & (ninf > 0)), float("nan"), r)
        tl.store(p_out, r)

    _CUDA["anchor"], _CUDA["sum"], _CUDA["finish"] = ts_osum_anchor, ts_osum_part, ts_osum_finish
    return ts_osum_anchor, ts_osum_part


def _sm_count(device) -> int:
    import torch
    key = ("sms", str(device))
    if key not in _CUDA:
        _CUDA[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return _CUDA[key]


def order_free_sum_cuda(x, block: int = 1024, programs: Optional[int] = None):
    """Order free sum of every element of a float32 CUDA tensor with two Triton kernels: a max of
    exponent fields per program, then an int64 partial per program, added on the device. The
    number of programs (default: 4 per SM) and the block size change the order of every addition
    and never the bits."""
    import torch

    if x.dtype != torch.float32 or not x.is_cuda:
        raise OrderFreeSumError(f"order_free_sum_cuda takes a float32 CUDA tensor, got {x.dtype} on {x.device}; "
                                "order_free_sum(x.reshape(-1)) takes a float32 tensor on any device")
    x = x.reshape(-1).contiguous()
    n = x.numel()
    if n > MAX_TERMS:
        raise OrderFreeSumError(f"at most {MAX_TERMS} terms, got {n}")
    if programs is None:
        programs = 4 * _sm_count(x.device)
    programs = min(programs, 8192)
    tiles = max(1, -(-n // block))
    programs = max(1, min(programs, tiles))
    iters = -(-tiles // programs)
    programs = -(-tiles // iters)
    k_anchor, k_sum = _cuda_kernels()
    pb = 1 << max(0, programs - 1).bit_length()
    a = torch.empty(programs, dtype=torch.int32, device=x.device)
    k_anchor[(programs,)](x, a, n, BLOCK=block, ITERS=iters)
    parts = torch.empty(programs * 4, dtype=torch.int64, device=x.device)
    k_sum[(programs,)](x, a, parts, n, programs, BLOCK=block, ITERS=iters, PB=pb)
    out = torch.empty((), dtype=torch.float32, device=x.device)
    _CUDA["finish"][(1,)](a, parts, out, programs, PB=pb)
    return out


def describe() -> Tuple[str, ...]:
    """One line facts for reports."""
    return (f"order free sum: int64 total on a grid {PRECISION_BITS} bits below the row's top "
            f"exponent, at most {MAX_TERMS} terms per row",)
