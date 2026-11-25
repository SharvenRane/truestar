"""Float64 matmul on int8 tensor cores, for Truestar's own float64 checks (docs/SLICED_F64.md).

Author: Sharven Rane, 2026-11-24.

Consumer GPUs run float64 at 1/64 of float32 (RTX 5070 Ti: about 0.63 TFLOPS measured), so the
float64 reference of a large contraction is the slow part of verifying it. This module computes
``A @ B`` for float64 ``A`` (M x K) and ``B`` (K x N) from exact integer products on the int8
tensor cores, and proves, entry by entry, that the result is at least as accurate as a native
float64 matmul would be guaranteed to be. It is used only inside verification, never as a user
facing matmul.

Method (an integer slice scheme in the family of Ozaki's error free splitting, written from the
math, not from any implementation):

1. Scale. Each row of A is scaled by a power of two so its largest magnitude is in [1/2, 1), and
   each column of B likewise (exact).
2. Slice. Each scaled entry is cut into ``s`` signed 7 bit digits: ``a = sum_p a_p 2**(-7p) + da``
   with integer ``|a_p| <= 127``, all digits of one entry with the entry's sign, and ``|da| <
   2**(-7s)``. Every step is exact in float64 (power of two scaling and truncation).
3. Multiply. For every digit pair with ``p + q <= s + 1``, ``a_p @ b_q`` is computed exactly as an
   int32 matmul on the int8 tensor cores (exact while ``K * 127**2 < 2**31``, so K is cut into
   chunks of at most 133,000), and the products of equal weight ``d = p + q`` are added exactly in
   int64.
4. Combine. ``C' = sum_d S_d 2**(-7d)`` in float64 (s additions), then unscale (exact).

Error bound, per entry, in scaled units (``P = |A'| |B'|``, ``rA`` row sums of ``|A'|``, ``cB``
column sums of ``|B'|``, ``u = 2**-53``):

    |C' - A'B'| <= 2**(-7s) (cB_j + rA_i)              digits beyond s
                 + 2**(-7(s+1)) DA_i                   digit pairs with p + q > s + 1 skipped
                 + gamma_s * P_ij                      float64 combine
    with DA_i = sum_k sum_p |a_p,ik| and gamma_n = n u / (1 - n u).

A native float64 matmul (any summation order) is only guaranteed ``gamma_K * P_ij``. ``P`` is
bounded from below and above with one float32 matmul of ``|A'|`` and ``|B'|`` (float32 rounding
accounted for). The number of digits ``s`` is the smallest from 6 to ``MAX_SLICES`` for which the
bound is at or below the native guarantee on (almost) every entry; entries where it still is not
are recomputed as native float64 dot products. So every entry of the result carries a proven bound
no larger than native float64's.

Not trusted blindly: the first call for each shape class is also compared with a float64 matmul on
the CPU (sampled rows), and a result outside the sum of both bounds raises
:class:`SlicedF64Mismatch` (TS1085). CUDA 13.5's release notes fixed a silent wrong result in the
vendor's own float64 emulation path; that is why this check exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

DIGIT_BITS = 7
MIN_SLICES = 6
MAX_SLICES = 10
K_CHUNK = 133_120             # K * 127**2 < 2**31 for exact int32 products, a multiple of 8
U64 = 2.0 ** -53
U32 = 2.0 ** -24


class SlicedF64Error(RuntimeError):
    """The sliced float64 matmul cannot run here (no CUDA, no int8 matmul)."""

    code = "TS1086"


class SlicedF64Mismatch(AssertionError):
    """The sliced result disagreed with a CPU float64 matmul beyond both proven bounds."""

    code = "TS1085"


@dataclass
class SlicedInfo:
    slices: int
    gemms: int
    native_entries: int          # entries recomputed as native float64 dot products
    entries: int
    worst_bound_ratio: float     # max over entries of (our bound) / (native float64 guarantee)
    cpu_checked: bool = False
    cpu_worst_ratio: float = 0.0  # max |ours - cpu| / (our bound + native bound), sampled rows
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> Dict:
        return dict(self.__dict__, notes=list(self.notes))


_checked_shapes = set()


def gamma(n: int, u: float = U64) -> float:
    return n * u / (1 - n * u)


def _scale_exponents(x, dim):
    """Per row (dim=1) or column (dim=0) exponent e with max |x| in [2**(e-1), 2**e); 0 for zeros."""
    import torch

    mx = x.abs().amax(dim=dim, keepdim=True)
    _, e = torch.frexp(mx)
    return torch.where(mx > 0, e, torch.zeros_like(e))


def _digits(xs, s):
    """Signed 7 bit digits (int8, shape (s, ...)) of scaled values |xs| < 1, and the residual
    (None from the fused CUDA kernel, which gives the same digits)."""
    import torch

    if xs.is_cuda and xs.dim() == 2 and _triton_ok():
        return _digits_cuda(xs.contiguous(), s), None

    out = []
    r = xs
    for _ in range(s):
        r = r * 128.0
        d = torch.trunc(r)
        out.append(d.to(torch.int8))
        r = r - d
    return torch.stack(out), r


def _pad32(n: int) -> int:
    return max(32, (n + 31) // 32 * 32)


def _int_mm_exact(a8, b8):
    """Exact int32 product of int8 matrices on CUDA (padding to the shapes cuBLAS takes)."""
    import torch

    M, K = a8.shape
    N = b8.shape[1]
    Mp, Kp, Np = _pad32(M), _pad32(K), _pad32(N)   # cuBLASLt int8 on sm_89 refuses m 264, n 304
    if (Mp, Kp, Np) != (M, K, N):
        a = torch.zeros((Mp, Kp), dtype=torch.int8, device=a8.device)
        a[:M, :K] = a8
        b = torch.zeros((Kp, Np), dtype=torch.int8, device=b8.device)
        b[:K, :N] = b8
        return torch._int_mm(a, b)[:M, :N]
    return torch._int_mm(a8, b8)


def _triton_ok() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def _chunks(K: int, s: int) -> int:
    """K chunks per weight on the Triton path: s digit pairs of K * 127**2 must fit in int32."""
    kc = ((2 ** 31 - 1) // (s * 127 * 127)) // 64 * 64
    return max(1, -(-K // kc))


_KERNEL = {}


def _weight_kernel():
    if "k" in _KERNEL:
        return _KERNEL["k"]
    import triton
    import triton.language as tl

    @triton.jit
    def ts_sliced_weight(pa, pb, pc, M, N, K0, K1, s_ap, s_am, s_ak, s_bq, s_bk, s_bn, P0, P1, D,
                         scale, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        # C[tile] += 2**(-7 D) * sum over p of digits_A[p] @ digits_B[D - p], over K0 <= k < K1,
        # accumulated exactly in int32 (the caller keeps K1 - K0 small enough)
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.int32)
        for p in range(P0, P1 + 1):
            q = D - p
            for k0 in range(K0, K1, BK):
                rk = k0 + tl.arange(0, BK)
                a = tl.load(pa + (p - 1) * s_ap + rm[:, None] * s_am + rk[None, :] * s_ak,
                            mask=(rm[:, None] < M) & (rk[None, :] < K1), other=0)
                b = tl.load(pb + (q - 1) * s_bq + rk[:, None] * s_bk + rn[None, :] * s_bn,
                            mask=(rk[:, None] < K1) & (rn[None, :] < N), other=0)
                acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        msk = (rm[:, None] < M) & (rn[None, :] < N)
        off = rm[:, None] * N + rn[None, :]
        c = tl.load(pc + off, mask=msk, other=0.0)
        tl.store(pc + off, c + acc.to(tl.float64) * scale, mask=msk)

    _KERNEL["k"] = ts_sliced_weight
    return ts_sliced_weight


def _digits_kernel():
    if "d" in _KERNEL:
        return _KERNEL["d"]
    import triton
    import triton.language as tl

    @triton.jit
    def ts_sliced_digits(px, pout, total, S: tl.constexpr, BLOCK: tl.constexpr):
        # the same steps as the torch loop: v = 128 v, digit = trunc(v), v -= digit (all exact)
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        msk = off < total
        v = tl.load(px + off, mask=msk, other=0.0)
        for i in tl.static_range(S):
            v = v * 128.0
            d = v.to(tl.int32)
            v = v - d.to(tl.float64)
            tl.store(pout + i * total + off, d.to(tl.int8), mask=msk)

    _KERNEL["d"] = ts_sliced_digits
    return ts_sliced_digits


def _digits_cuda(xs, s):
    import torch
    import triton

    total = xs.numel()
    out = torch.empty((s,) + tuple(xs.shape), dtype=torch.int8, device=xs.device)
    _digits_kernel()[(triton.cdiv(total, 1024),)](xs, out, total, S=s, BLOCK=1024, num_warps=4)
    return out


def _triton_products(ad, bd, s, C, BM=128, BN=128, BK=64):
    """C += sum over weights d of 2**(-7d) sum_{p+q=d} ad[p] @ bd[q], smallest weight first."""
    import triton

    k = _weight_kernel()
    _, M, K = ad.shape
    N = bd.shape[2]
    kc = -(-K // _chunks(K, s))
    kc = -(-kc // BK) * BK
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    launches = 0
    for d in range(s + 1, 1, -1):
        p0, p1 = max(1, d - s), min(s, d - 1)
        for k0 in range(0, K, kc):
            k[grid](ad, bd, C, M, N, k0, min(K, k0 + kc), ad.stride(0), ad.stride(1), ad.stride(2),
                    bd.stride(0), bd.stride(1), bd.stride(2), p0, p1, d, 2.0 ** (-DIGIT_BITS * d),
                    BM=BM, BN=BN, BK=BK, num_warps=8, num_stages=3)
            launches += p1 - p0 + 1
    return launches


def _bounds(Ap, Bp, K):
    """Lower and upper bounds on P = |A'| |B'| from one float32 matmul (no TF32)."""
    import torch

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        P32 = (Ap.abs().to(torch.float32) @ Bp.abs().to(torch.float32)).to(torch.float64)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    g = gamma(K + 2, U32)        # K float32 roundings of the sum, two of the operands
    slack = K * 2.0 ** -126      # operands below float32's normal range may round up or flush
    return torch.clamp(P32 * (1 - g) - slack, min=0.0), P32 * (1 + g) / (1 - g) + slack


def sliced_matmul(a, b, slices: Optional[int] = None, check_cpu: Optional[bool] = None,
                  fix_entries: bool = True, engine: Optional[str] = None):
    """(A @ B in float64, SlicedInfo) computed on int8 tensor cores with a proven per entry bound
    at or below native float64's guarantee. a, b: float64 CUDA tensors (M x K, K x N).

    slices: digits per operand (default: the smallest from 6 to MAX_SLICES that meets the bound).
    check_cpu: compare sampled rows with a CPU float64 matmul (default: first call per shape).
    fix_entries: recompute the entries whose bound is above native float64's as native dots.
    engine: "triton" (one fused int8 kernel per digit weight, the default when Triton runs) or
    "int_mm" (torch._int_mm per digit pair, int64 sums)."""
    import torch

    if not (a.is_cuda and b.is_cuda):
        raise SlicedF64Error("the sliced float64 matmul needs CUDA tensors (int8 tensor cores)")
    if a.dim() != 2 or b.dim() != 2 or a.shape[1] != b.shape[0]:
        raise SlicedF64Error(f"needs M x K and K x N matrices, got {tuple(a.shape)} and {tuple(b.shape)}")
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise SlicedF64Error("the sliced float64 matmul takes finite operands only")
    a, b = a.to(torch.float64), b.to(torch.float64)
    M, K = a.shape
    N = b.shape[1]
    ea = _scale_exponents(a, 1)                      # (M, 1)
    eb = _scale_exponents(b, 0)                      # (1, N)
    Ap = torch.ldexp(a, -ea)
    Bp = torch.ldexp(b, -eb)
    P_lo, P_hi = _bounds(Ap, Bp, K)
    rA = Ap.abs().sum(1, keepdim=True)
    cB = Bp.abs().sum(0, keepdim=True)
    native = gamma(K) * P_lo

    use_triton = engine == "triton" or (engine is None and _triton_ok())

    def combine_terms(s):
        # float64 additions into C: one per weight (int_mm path: K chunks add exactly in int64),
        # one per weight and K chunk on the Triton path (int32 accumulators)
        return s * (_chunks(K, s) if use_triton else 1)

    def bound(s, da):
        return (2.0 ** (-DIGIT_BITS * s) * (cB + rA) + 2.0 ** (-DIGIT_BITS * (s + 1)) * da
                + gamma(combine_terms(s)) * P_hi)

    choices = [slices] if slices else list(range(MIN_SLICES, MAX_SLICES + 1))
    all_dig, _ = _digits(Ap, choices[-1])            # digits do not depend on how many are kept
    da_rows = all_dig.abs().sum(dim=2, dtype=torch.int64).to(torch.float64)   # (s_max, M)
    for s in choices:
        da_dig = all_dig[:s]
        DA = da_rows[:s].sum(0).unsqueeze(1)
        E = bound(s, DA)
        bad = E > native
        nbad = int(bad.sum())
        if nbad <= max(0, (M * N) // 1000) or s == choices[-1]:
            break
    db_dig, _ = _digits(Bp, s)                       # (s, K, N); B was scaled per column
    C = torch.zeros((M, N), dtype=torch.float64, device=a.device)
    gemms = 0
    if use_triton:
        gemms = _triton_products(da_dig.contiguous(), db_dig, s, C)
    for d in (range(s + 1, 1, -1) if not use_triton else ()):   # smallest weight first
        S = torch.zeros((M, N), dtype=torch.int64, device=a.device)
        for p in range(max(1, d - s), min(s, d - 1) + 1):
            q = d - p
            for k0 in range(0, K, K_CHUNK):
                k1 = min(K, k0 + K_CHUNK)
                S += _int_mm_exact(da_dig[p - 1][:, k0:k1].contiguous(), db_dig[q - 1][k0:k1]).to(torch.int64)
                gemms += 1
        C += S.to(torch.float64) * 2.0 ** (-DIGIT_BITS * d)
    C = torch.ldexp(C, ea + eb)
    ratio = (E / torch.where(native > 0, native, torch.full_like(native, float("inf"))))
    ratio = torch.where(native > 0, ratio, torch.where(E > 0, torch.full_like(E, float("inf")), torch.zeros_like(E)))
    notes = []
    nfix = 0
    if fix_entries and nbad:
        idx = bad.nonzero()
        step = max(1, (1 << 24) // max(K, 1))
        for c0 in range(0, idx.shape[0], step):
            ii, jj = idx[c0:c0 + step, 0], idx[c0:c0 + step, 1]
            C[ii, jj] = (a[ii] * b[:, jj].t()).sum(1)
        nfix = int(idx.shape[0])
        notes.append(f"{nfix} entries recomputed as native float64 dots (bound above native)")
        ratio = torch.where(bad, torch.ones_like(ratio), ratio)
    info = SlicedInfo(slices=s, gemms=gemms, native_entries=nfix, entries=M * N,
                      worst_bound_ratio=float(ratio.max()) if M * N else 0.0, notes=tuple(notes))
    key = (M, N, K, s)
    if check_cpu or (check_cpu is None and key not in _checked_shapes):
        _cpu_check(a, b, C, torch.ldexp(E, ea + eb), torch.ldexp(native, ea + eb), info)
        _checked_shapes.add(key)
    return C, info


def _cpu_check(a, b, C, E, native, info, rows: int = 64):
    """Compare sampled rows with a float64 CPU matmul: |ours - cpu| <= our bound + native's."""
    import torch

    M = a.shape[0]
    g = torch.Generator().manual_seed(M * 7919 + a.shape[1])
    pick = torch.randperm(M, generator=g)[:rows].to(a.device)
    ref = (a[pick].cpu() @ b.cpu())
    diff = (C[pick].cpu() - ref).abs()
    lim = (E[pick] + native[pick]).cpu()
    worst = float((diff / torch.where(lim > 0, lim, torch.ones_like(lim))).max())
    info.cpu_checked = True
    info.cpu_worst_ratio = worst
    bad = diff > lim
    if bool(bad.any()):
        i, j = (int(v) for v in bad.nonzero()[0])
        raise SlicedF64Mismatch(f"[TS1085] sliced float64 matmul {tuple(a.shape)} x {tuple(b.shape)}: row "
                                f"{int(pick[i])}, column {j}: sliced {float(C[pick][i, j])!r}, CPU float64 "
                                f"{float(ref[i, j])!r}, beyond both proven bounds {float(lim[i, j]):.3e}")


def sliced_einsum(spec: str, x, y):
    """torch.einsum(spec, x, y) in float64 through sliced_matmul when spec is two operands with one
    contracted letter (batch letters in both operands loop). None when the spec has another form."""
    import torch

    ins, out = spec.replace(" ", "").split("->")
    sa, sb = ins.split(",")
    if len(set(sa)) != len(sa) or len(set(sb)) != len(sb):
        return None
    contracted = [c for c in sa if c in sb and c not in out]
    if len(contracted) != 1 or any(c not in sa + sb for c in out):
        return None
    if any(c not in out and c not in contracted for c in sa + sb):
        return None                                  # a letter summed out of one operand only
    k = contracted[0]
    batch = [c for c in out if c in sa and c in sb]
    fa = [c for c in out if c in sa and c not in sb]
    fb = [c for c in out if c in sb and c not in sa]
    sizes = {c: x.shape[sa.index(c)] for c in sa}
    sizes.update({c: y.shape[sb.index(c)] for c in sb})
    xa = x.permute(*[sa.index(c) for c in batch + fa + [k]]).reshape(
        [sizes[c] for c in batch] + [math.prod(sizes[c] for c in fa), sizes[k]])
    yb = y.permute(*[sb.index(c) for c in batch + [k] + fb]).reshape(
        [sizes[c] for c in batch] + [sizes[k], math.prod(sizes[c] for c in fb)])
    nb = math.prod(sizes[c] for c in batch)
    xa = xa.reshape(nb, xa.shape[-2], xa.shape[-1])
    yb = yb.reshape(nb, yb.shape[-2], yb.shape[-1])
    outs = [sliced_matmul(xa[i].to(torch.float64), yb[i].to(torch.float64))[0] for i in range(nb)]
    r = torch.stack(outs).reshape([sizes[c] for c in batch + fa + fb])
    order = batch + fa + fb
    return r.permute(*[order.index(c) for c in out])
