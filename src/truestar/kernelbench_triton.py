"""Triton kernels for the R1 and R7 benchmarks: correct ones and ones with the bugs hand and AI
written GPU kernels carry. CUDA and Triton only; import this module lazily.

* matmul (tl.dot, float32 accumulator, masked tiles): correct; grid that floors N (the last
  partial column tile is never written); float16 accumulator; strides ignored (assumes row major).
* row softmax: correct; mask edge off by one (the last column is never loaded).
* row sum with tl.atomic_add of block partials: the order of the atomics varies run to run.
* for `verified_autotune` (R7): a matmul whose K loop has no mask unless KMASK (right only for
  configs whose BK divides K) and a row softmax, with their autotune config lists.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .kernelbench import Case, causal_attn_ref, mm_ref, rowsum_ref, softmax_ref


@triton.jit
def _mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, F16ACC: tl.constexpr,
               ROWMAJOR: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    if F16ACC:
        acc = tl.zeros((BM, BN), dtype=tl.float16)
    else:
        acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        if ROWMAJOR:  # the bug: strides assumed, not read
            pa = a_ptr + rm[:, None] * K + kk[None, :]
            pb = b_ptr + kk[:, None] * N + rn[None, :]
        else:
            pa = a_ptr + rm[:, None] * sam + kk[None, :] * sak
            pb = b_ptr + kk[:, None] * sbk + rn[None, :] * sbn
        a = tl.load(pa, mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(pb, mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        if F16ACC:
            acc += tl.dot(a, b, out_dtype=tl.float16)
        else:
            acc += tl.dot(a, b)
    pc = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(pc, acc.to(c_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


def _mm(a, b, *, floor_n=False, f16acc=False, rowmajor=False, BM=64, BN=64, BK=32):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    gn = N // BN if floor_n else triton.cdiv(N, BN)
    grid = (triton.cdiv(M, BM), max(gn, 1))
    _mm_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                     c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, F16ACC=f16acc,
                     ROWMAJOR=rowmajor)
    return c


def mm_triton(a, b):
    return _mm(a, b)


def mm_triton_floor_grid(a, b):
    """Wrong: grid N // BN, the last partial column tile is never written."""
    return _mm(a, b, floor_n=True)


def mm_triton_f16acc(a, b):
    """Wrong: float16 accumulator."""
    return _mm(a, b, f16acc=True)


def mm_triton_rowmajor(a, b):
    """Wrong: ignores strides (assumes row major operands)."""
    return _mm(a, b, rowmajor=True)


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, n, sx, sy, BLOCK: tl.constexpr, EDGE_BUG: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    lim = n - 1 if EDGE_BUG else n
    x = tl.load(x_ptr + row * sx + cols, mask=cols < lim, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row * sy + cols, y.to(y_ptr.dtype.element_ty), mask=cols < n)


def _softmax(x, edge_bug=False):
    m, n = x.shape
    if x.stride(1) != 1:
        x = x.contiguous()
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    _softmax_kernel[(m,)](x, y, n, x.stride(0), y.stride(0), BLOCK=triton.next_power_of_2(n),
                          EDGE_BUG=edge_bug)
    return y


def softmax_triton(x):
    return _softmax(x)


def softmax_triton_edge(x):
    """Wrong: the last column is masked out of the max and the sum (mask edge off by one)."""
    return _softmax(x, edge_bug=True)


@triton.jit
def _rowsum_atomic_kernel(x_ptr, y_ptr, n, sx, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * sx + cols, mask=cols < n, other=0.0).to(tl.float32)
    tl.atomic_add(y_ptr + row, tl.sum(x, axis=0))


def rowsum_triton_atomic(x):
    """Race prone: block partials added with float32 atomics in whatever order blocks finish, so
    the bits change from run to run."""
    m, n = x.shape
    y = torch.zeros(m, device=x.device, dtype=torch.float32)
    BLOCK = 128
    _rowsum_atomic_kernel[(m, triton.cdiv(n, BLOCK))](x, y, n, x.stride(0), BLOCK=BLOCK)
    return y.to(x.dtype)


def _mm_inputs(d, dtype, g):
    return (torch.randn(d["M"], d["K"], generator=g).to(dtype),
            torch.randn(d["K"], d["N"], generator=g).to(dtype))


def _rows(d, dtype, g, scale=1.0):
    return (torch.randn(d["M"], d["N"], generator=g).mul(scale).to(dtype),)


def cases():
    mm_d = dict(M=192, K=512, N=200)
    sm_d = dict(M=64, N=1000)
    rs_d = dict(M=64, N=8192)
    return [
        Case("triton_mm_fp16", mm_ref, mm_triton, True, _mm_inputs, mm_d, torch.float16),
        Case("triton_mm_bf16", mm_ref, mm_triton, True, _mm_inputs, mm_d, torch.bfloat16),
        Case("triton_mm_floor_grid_fp16", mm_ref, mm_triton_floor_grid, False, _mm_inputs, mm_d,
             torch.float16, bug="grid floors N: last column tile never written"),
        Case("triton_mm_floor_grid_even", mm_ref, mm_triton_floor_grid, False, _mm_inputs,
             dict(M=128, K=512, N=256), torch.float16,
             bug="grid floors N (only at sizes that do not divide 64)", needs=("shape",)),
        Case("triton_mm_f16acc", mm_ref, mm_triton_f16acc, False, _mm_inputs, mm_d,
             torch.float16, bug="float16 accumulator (16 steps: within the default reduced "
                                "precision reduction policy)"),
        Case("triton_mm_f16acc_long", mm_ref, mm_triton_f16acc, False, _mm_inputs,
             dict(M=192, K=4096, N=200), torch.float16, bug="float16 accumulator (128 steps)"),
        Case("triton_mm_rowmajor", mm_ref, mm_triton_rowmajor, False, _mm_inputs, mm_d,
             torch.float16, bug="ignores strides", needs=("strided",)),
        Case("triton_softmax_fp16", softmax_ref, softmax_triton, True,
             lambda d, dt, g: _rows(d, dt, g, 3.0), sm_d, torch.float16),
        Case("triton_softmax_fp32", softmax_ref, softmax_triton, True,
             lambda d, dt, g: _rows(d, dt, g, 3.0), sm_d, torch.float32),
        Case("triton_softmax_edge", softmax_ref, softmax_triton_edge, False,
             lambda d, dt, g: _rows(d, dt, g, 3.0), sm_d, torch.float16,
             bug="mask edge off by one (last column dropped)"),
        Case("triton_rowsum_atomic", rowsum_ref, rowsum_triton_atomic, False, _rows, rs_d,
             torch.float32, bug="atomic reduction: bits change run to run",
             nondeterministic=True),
    ]


# ----------------------------------------------------------------------------- autotune (R7)


@triton.jit
def autotune_mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KMASK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        if KMASK:
            ma = (rm[:, None] < M) & (kk[None, :] < K)
            mb = (kk[:, None] < K) & (rn[None, :] < N)
        else:  # assumes BK divides K
            ma = rm[:, None] < M
            mb = rn[None, :] < N
        a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak, mask=ma, other=0.0)
        b = tl.load(b_ptr + kk[:, None] * sbk + rn[None, :] * sbn, mask=mb, other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc.to(c_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def autotune_softmax_kernel(x_ptr, y_ptr, N, sx, sy, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * sx + cols, mask=cols < N, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(y_ptr + row * sy + cols, (e / tl.sum(e, axis=0)).to(y_ptr.dtype.element_ty),
             mask=cols < N)


AUTOTUNE_MM_CONFIGS = [triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=w, num_stages=s)
              for bm, bn, bk, w, s in [(32, 32, 16, 4, 2), (64, 64, 16, 4, 3), (64, 64, 32, 4, 3),
                                       (128, 64, 32, 4, 3), (64, 128, 64, 4, 3),
                                       (128, 128, 64, 8, 3)]]
AUTOTUNE_SM_CONFIGS = [triton.Config({}, num_warps=w) for w in (1, 2, 4, 8, 16)]
