"""Batch and position invariant Triton kernels shared by the RL trainer and sampler (P15 part b).

Every kernel here gives an output element the same bits whatever else runs with it: the batch
size, the row's place in the batch, how many query tokens run in one call (training forward,
prefill, chunked prefill or one token decode), the KV cache length and capacity, and left
padding. The rule that makes this hold is the same in every kernel:

* Tile shapes, block sizes and warp counts are constants. Nothing is chosen from the shapes of
  a call (no autotuning, no split K picked by size, no vendor library heuristics). A program
  computes an output element the same way whichever tile row or column it sits in, and rows
  never mix.
* Every reduction walks its axis in a fixed order of fixed size chunks: chunk results are
  accumulated one after another into a vector of fixed width, and the vector is summed once at
  the end. Extra chunks past the end, or masked entries inside a chunk, add exact zeros.
* Reductions whose leading entries can be masked (softmax rows of attention scores, attention
  over a KV cache) start their chunks at the first unmasked entry, the "anchor", not at index 0.
  A row left padded by p puts its first real key at column p; its chunks then hold the same
  logical keys as the trainer's unpadded row, so the online softmax sees the same chunk maxima.
  The attention kernel finds the anchor per query tile from the mask (an exact integer min).
* Contractions over a weight's input dim (`matmul` with `segments` > 1) split K into a fixed
  number of equal segments, each reduced from a zero accumulator, and add the segment results
  as a balanced binary tree in registers. A tensor parallel rank that holds a contiguous half of
  K computes exactly one subtree (`matmul_partials`), so adding the ranks' float32 results in
  the same tree order (`tp.all_reduce`) gives the TP 1 bits.
* One compiled binary per kernel and dtype. Masks and causality are runtime flags (as constexpr
  flags the masked decode compiled to another layout and gave other bits than the causal
  training forward, found while building this). Kernels that reduce loaded values are compiled
  without Triton's value and alignment specialization (an unaligned view otherwise got its own
  binary with another `tl.sum` order, found and tested); the contraction keeps alignment
  specialization, copies unaligned operands to aligned memory, and never specializes on M.
  Compiler chosen mul + add contraction is off (`enable_fp_fusion=False`); that one is
  defensive, no test failed with it on.

All of it runs on CUDA through Triton. There is no CPU path: a CPU tensor is an error, never a
quiet switch to another kernel.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

try:  # Triton is optional for the rest of Truestar
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only without Triton
    triton = None
    tl = None

# Fixed tile shapes. Never derived from a call's shapes.
ATT_BM, ATT_BN, ATT_WARPS = 32, 16, 2
ROW_BLOCK, ROW_WARPS = 1024, 4
SEGMENTS = 4  # default K segments of a weight contraction (supports TP 1, 2 and 4)

_FLOAT = (torch.float32, torch.float16, torch.bfloat16)


class RLKernelError(RuntimeError):
    """A call the RL kernel set cannot run with its invariance guarantee."""


def available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _need(t: torch.Tensor, what: str) -> None:
    if triton is None:
        raise RLKernelError("the RL kernel set needs Triton, which is not installed")
    if t.device.type != "cuda":
        raise RLKernelError(f"{what}: the RL kernel set runs on CUDA only, got a {t.device.type} "
                            "tensor (there is no CPU path and no fallback)")
    if t.dtype not in _FLOAT:
        raise RLKernelError(f"{what}: dtype {t.dtype} is not supported (float32, float16, bfloat16)")


def _prec(dtype: torch.dtype) -> str:
    return "ieee" if dtype == torch.float32 else "tf32"  # tf32 is ignored for half inputs


_LAUNCH = {"enable_fp_fusion": False}  # no compiler chosen mul + add contraction
_dummy_masks = {}


def _dummy_mask(device) -> torch.Tensor:
    key = str(device)
    if key not in _dummy_masks:
        _dummy_masks[key] = torch.ones(1, dtype=torch.bool, device=device)
    return _dummy_masks[key]


def _jit(fn):
    """triton.jit with no value or alignment specialization of any argument.

    Triton compiles a separate binary when an integer argument is 1 or a multiple of 16, or a
    pointer is 16 byte aligned, and the layouts it picks (and with them the order of every
    tl.sum) can differ between those binaries. A one token decode (Lq = 1) or an unaligned
    cache view must run the same binary as the training forward, so nothing is specialized."""
    import inspect

    names = [n for n, p in inspect.signature(fn).parameters.items()
             if "constexpr" not in str(p.annotation)]
    return triton.jit(do_not_specialize=names, do_not_specialize_on_alignment=names)(fn)


def _jit_keep_alignment(fn):
    """triton.jit with no specialization of M (the batch times sequence rows) but with Triton's
    alignment specialization of pointers and strides, so tiles load with vector instructions.

    Only for the contraction: its arithmetic is fixed by the tl.dot operand layouts, which do not
    depend on how a tile was loaded (tests/test_rlkernels.py checks unaligned views). Kernels
    that reduce loaded values with tl.sum use `_jit`."""
    return triton.jit(do_not_specialize=["M"], do_not_specialize_on_alignment=["M"])(fn)


if triton is not None:

    @triton.jit
    def _mm_seg(a_base, b_base, rm, rn, M, N, k0, seg_len, sam, sak, sbk, sbn,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        """One K segment [k0, k0 + seg_len), from a zero accumulator, BK at a time in order."""
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        k_end = k0 + seg_len
        for kk in range(0, seg_len, BK):
            rk = k0 + kk + tl.arange(0, BK)
            km = rk < k_end
            a = tl.load(a_base + rm[:, None] * sam + rk[None, :] * sak,
                        mask=(rm[:, None] < M) & km[None, :], other=0.0)
            b = tl.load(b_base + rk[:, None] * sbk + rn[None, :] * sbn,
                        mask=km[:, None] & (rn[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc, input_precision=PREC)
        return acc

    @_jit_keep_alignment
    def _mm_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, M, N, seg_len,
                   sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                   G: tl.constexpr, HAS_BIAS: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
        # G segments added as a balanced binary tree in registers: ((s0 + s1) + (s2 + s3)) ...
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        bat = tl.program_id(2).to(tl.int64)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        a_base = a_ptr + bat * sab
        b_base = b_ptr + bat * sbb
        acc = _mm_seg(a_base, b_base, rm, rn, M, N, 0, seg_len, sam, sak, sbk, sbn, BM, BN, BK, PREC)
        if G >= 2:
            acc = acc + _mm_seg(a_base, b_base, rm, rn, M, N, seg_len, seg_len, sam, sak, sbk, sbn,
                                BM, BN, BK, PREC)
        if G >= 4:
            t = _mm_seg(a_base, b_base, rm, rn, M, N, 2 * seg_len, seg_len, sam, sak, sbk, sbn,
                        BM, BN, BK, PREC)
            t = t + _mm_seg(a_base, b_base, rm, rn, M, N, 3 * seg_len, seg_len, sam, sak, sbk, sbn,
                            BM, BN, BK, PREC)
            acc = acc + t
        if HAS_BIAS:
            acc = acc + tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
        tl.store(c_ptr + bat * scb + rm[:, None] * scm + rn[None, :] * scn,
                 acc.to(c_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))

    @_jit
    def _row_sum_kernel(x_ptr, out_ptr, n, sx, SQUARE: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        base = x_ptr + r.to(tl.int64) * sx
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for c in range(0, n, BLOCK):
            idx = c + tl.arange(0, BLOCK)
            v = tl.load(base + idx, mask=idx < n, other=0.0).to(tl.float32)
            if SQUARE:
                v = v * v
            acc += v
        tl.store(out_ptr + r, tl.sum(acc, axis=0))

    @_jit
    def _softmax_kernel(x_ptr, out_ptr, n, sx, so, LOG: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        base = x_ptr + r.to(tl.int64) * sx
        obase = out_ptr + r.to(tl.int64) * so
        mx = tl.full((BLOCK,), float("-inf"), tl.float32)
        first = tl.full((BLOCK,), n, tl.int32)
        for c in range(0, n, BLOCK):
            idx = c + tl.arange(0, BLOCK)
            v = tl.load(base + idx, mask=idx < n, other=float("-inf")).to(tl.float32)
            mx = tl.maximum(mx, v)
            first = tl.minimum(first, tl.where(v > float("-inf"), idx, n))
        m = tl.max(mx, axis=0)
        a = tl.min(first, axis=0)
        a = tl.where(a >= n, 0, a)     # an all masked row: no anchor, plain order
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for c in range(a, n, BLOCK):   # chunks anchored at the first unmasked entry
            idx = c + tl.arange(0, BLOCK)
            v = tl.load(base + idx, mask=idx < n, other=float("-inf")).to(tl.float32)
            acc += tl.where(idx < n, tl.exp(v - m), 0.0)
        s = tl.sum(acc, axis=0)
        ls = tl.log(s)
        for c in range(0, n, BLOCK):
            idx = c + tl.arange(0, BLOCK)
            v = tl.load(base + idx, mask=idx < n, other=float("-inf")).to(tl.float32)
            if LOG:
                y = v - m - ls
            else:
                y = tl.exp(v - m) / s
            tl.store(obase + idx, y.to(out_ptr.dtype.element_ty), mask=idx < n)

    @_jit
    def _norm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, n, sx, so, eps,
                     RMS: tl.constexpr, HAS_W: tl.constexpr, HAS_B: tl.constexpr,
                     BLOCK: tl.constexpr):
        r = tl.program_id(0)
        base = x_ptr + r.to(tl.int64) * sx
        obase = out_ptr + r.to(tl.int64) * so
        if RMS:
            mean = 0.0
        else:
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            for c in range(0, n, BLOCK):
                idx = c + tl.arange(0, BLOCK)
                acc += tl.load(base + idx, mask=idx < n, other=0.0).to(tl.float32)
            mean = tl.sum(acc, axis=0) / n
        acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
        for c in range(0, n, BLOCK):
            idx = c + tl.arange(0, BLOCK)
            v = tl.load(base + idx, mask=idx < n, other=0.0).to(tl.float32)
            d = tl.where(idx < n, v - mean, 0.0)
            acc2 += d * d
        rstd = tl.rsqrt(tl.sum(acc2, axis=0) / n + eps)
        for c in range(0, n, BLOCK):
            idx = c + tl.arange(0, BLOCK)
            v = tl.load(base + idx, mask=idx < n, other=0.0).to(tl.float32)
            y = (v - mean) * rstd
            if HAS_W:
                y = y * tl.load(w_ptr + idx, mask=idx < n, other=0.0).to(tl.float32)
            if HAS_B:
                y = y + tl.load(b_ptr + idx, mask=idx < n, other=0.0).to(tl.float32)
            tl.store(obase + idx, y.to(out_ptr.dtype.element_ty), mask=idx < n)

    @_jit
    def _attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, m_ptr, Lq, Lk, HQ, GROUP, scale,
                     sqb, sqh, sqm, sqd, skb, skh, skn, skd, svb, svh, svn, svd,
                     sob, soh, som, sod, smb, smh, smq, smk, has_mask, causal, anchor,
                     D: tl.constexpr, BD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                     PREC: tl.constexpr):
        # Mask and causality are runtime flags, not constexpr: one compiled binary serves every
        # call, so the compiler cannot pick another layout (and another reduction order) for
        # the masked decode than for the causal training forward.
        pid_m = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // HQ
        h = bh % HQ
        hk = h // GROUP
        rm = pid_m * BM + tl.arange(0, BM)
        rd = tl.arange(0, BD)
        b64 = b.to(tl.int64)
        q = tl.load(q_ptr + b64 * sqb + h * sqh + rm[:, None] * sqm + rd[None, :] * sqd,
                    mask=(rm[:, None] < Lq) & (rd[None, :] < D), other=0.0)
        # The anchor: the first key column any query row of this tile attends (integer min, exact).
        # Key chunks start there, so a row left padded by p reads the same logical keys per chunk
        # as the unpadded row. Fully masked rows (pad queries) do not count. The same scan finds
        # the last attended column; chunks past it would only add exact zeros and are skipped.
        first = Lk
        last = -1
        for n0 in range(0, tl.where(has_mask != 0, Lk, 0), BN):
            rn = n0 + tl.arange(0, BN)
            mk = tl.load(m_ptr + b64 * smb + h * smh + rm[:, None] * smq + rn[None, :] * smk,
                         mask=(rm[:, None] < Lq) & (rn[None, :] < Lk), other=0)
            first = tl.minimum(first, tl.min(tl.min(tl.where(mk != 0, rn[None, :], Lk), axis=1), axis=0))
            last = tl.maximum(last, tl.max(tl.max(tl.where(mk != 0, rn[None, :], -1), axis=1), axis=0))
        start = tl.where((has_mask != 0) & (anchor != 0) & (first < Lk), first, 0)
        end = tl.where(has_mask != 0, last + 1, Lk)
        # top left causal: keys past the tile's last row are masked for every row
        end = tl.where(causal != 0, tl.minimum(end, (pid_m + 1) * BM), end)
        m_i = tl.full((BM,), float("-inf"), tl.float32)
        l_i = tl.zeros((BM,), dtype=tl.float32)
        acc = tl.zeros((BM, BD), dtype=tl.float32)
        for n0 in range(start, end, BN):
            rn = n0 + tl.arange(0, BN)
            inb = rn < Lk
            kt = tl.load(k_ptr + b64 * skb + hk * skh + rn[None, :] * skn + rd[:, None] * skd,
                         mask=inb[None, :] & (rd[:, None] < D), other=0.0)
            s = tl.dot(q, kt, input_precision=PREC) * scale
            valid = inb[None, :] & (rm[:, None] < Lq)
            valid = valid & ((causal == 0) | (rn[None, :] <= rm[:, None]))
            mk = tl.load(m_ptr + b64 * smb + h * smh + rm[:, None] * smq + rn[None, :] * smk,
                         mask=valid & (has_mask != 0), other=1)
            valid = valid & (mk != 0)
            s = tl.where(valid, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.where(m_new == m_i, 1.0, tl.exp(m_i - m_new))
            p = tl.where(valid, tl.exp(s - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            used = tl.max(valid.to(tl.int32), axis=0) > 0     # keys some row of the tile reads
            vt = tl.load(v_ptr + b64 * svb + hk * svh + rn[:, None] * svn + rd[None, :] * svd,
                         mask=used[:, None] & (rd[None, :] < D), other=0.0)
            acc = tl.dot(p.to(vt.dtype), vt, acc * alpha[:, None], input_precision=PREC)
            m_i = m_new
        out = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
        tl.store(o_ptr + b64 * sob + h * soh + rm[:, None] * som + rd[None, :] * sod,
                 out.to(o_ptr.dtype.element_ty), mask=(rm[:, None] < Lq) & (rd[None, :] < D))


# ----------------------------------------------------------------------------- matmul


def tree_sum(parts: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Balanced pairwise sum over `dim` (a power of two long): ((p0 + p1) + (p2 + p3)) ...
    Elementwise adds, so every element gets the same bits on every backend."""
    n = parts.shape[dim]
    if n & (n - 1):
        raise RLKernelError(f"tree_sum needs a power of two count, got {n}")
    while parts.shape[dim] > 1:
        parts = parts.narrow(dim, 0, parts.shape[dim]).unflatten(dim, (-1, 2))
        parts = parts.select(dim + 1, 0) + parts.select(dim + 1, 1)
    return parts.squeeze(dim)


def _segments_for(K: int, segments: Optional[int]) -> int:
    g = SEGMENTS if segments is None else int(segments)
    if g < 1 or g & (g - 1):
        raise RLKernelError(f"segments must be a power of two, got {g}")
    if K % g:
        if segments is not None:
            raise RLKernelError(f"K = {K} does not split into {g} equal segments")
        g = 1
        while g * 2 <= SEGMENTS and K % (g * 2) == 0:
            g *= 2
    return g


MM_CFG = {  # (BM, BN, BK, warps, stages) per operand dtype; constants, never chosen per call
    torch.float32: (64, 64, 32, 4, 3),
    torch.float16: (64, 64, 64, 4, 3),
    torch.bfloat16: (64, 64, 64, 4, 3),
}


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """A copy when the data does not start on a 16 byte boundary, so every call compiles to the
    aligned binary (an unaligned view would otherwise get its own)."""
    return t if t.data_ptr() % 16 == 0 else t.clone()


def _mm(a: torch.Tensor, b: torch.Tensor, G: int, bias: Optional[torch.Tensor],
        out_dtype: torch.dtype) -> torch.Tensor:
    _need(a, "matmul")
    _need(b, "matmul")
    if a.dtype != b.dtype:
        raise RLKernelError(f"matmul: operand dtypes differ ({a.dtype}, {b.dtype})")
    if a.dim() < 2 or b.dim() < 2:
        raise RLKernelError("matmul: both operands need at least 2 dims")
    M, K = a.shape[-2:]
    K2, N = b.shape[-2:]
    if K != K2:
        raise RLKernelError(f"matmul: inner dims differ ({K} and {K2})")
    if G not in (1, 2, 4) or K % G:
        raise RLKernelError(f"matmul: K = {K} does not split into {G} segments (1, 2 or 4)")
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a3 = _aligned(a.expand(*batch, M, K).reshape(-1, M, K))
    b3 = _aligned(b.expand(*batch, K, N).reshape(-1, K, N))
    nb = a3.shape[0]
    out = torch.empty((nb, M, N), device=a.device, dtype=out_dtype)
    if out.numel():
        BM, BN, BK, warps, stages = MM_CFG[a.dtype]
        bias_arg = _aligned(bias.contiguous()) if bias is not None else out
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), nb)
        _mm_kernel[grid](a3, b3, bias_arg, out, M, N, K // G,
                         a3.stride(0), a3.stride(1), a3.stride(2),
                         b3.stride(0), b3.stride(1), b3.stride(2),
                         out.stride(0), out.stride(1), out.stride(2),
                         G=G, HAS_BIAS=bias is not None, BM=BM, BN=BN, BK=BK, PREC=_prec(a.dtype),
                         num_warps=warps, num_stages=stages, **_LAUNCH)
    return out.view(*batch, M, N)


def matmul_partials(a: torch.Tensor, b: torch.Tensor, segments: int) -> torch.Tensor:
    """float32 a @ b with K reduced in `segments` equal segments added as a balanced tree, before
    any cast: what one tensor parallel rank contributes to a row parallel layer."""
    return _mm(a, b, segments, None, torch.float32)


def matmul(a: torch.Tensor, b: torch.Tensor, segments: Optional[int] = 1,
           bias: Optional[torch.Tensor] = None, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """a @ b with a fixed reduction order. `segments` None picks the default weight split for K."""
    G = _segments_for(a.shape[-1], segments)
    return _mm(a, b, G, bias, out_dtype or a.dtype)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
           segments: Optional[int] = None) -> torch.Tensor:
    """F.linear with the weight contraction split into `segments` (default SEGMENTS) K segments."""
    return matmul(x, weight.t(), segments=segments, bias=bias)


# ----------------------------------------------------------------------------- rows


def _rows(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    return x2, n


def row_sum(x: torch.Tensor, square: bool = False) -> torch.Tensor:
    """Sum over the last dim in float32, fixed chunk order. Shape [...]."""
    _need(x, "sum")
    x2, n = _rows(x)
    out = torch.empty(x2.shape[0], device=x.device, dtype=torch.float32)
    if x2.shape[0]:
        _row_sum_kernel[(x2.shape[0],)](x2, out, n, x2.stride(0), SQUARE=square, BLOCK=ROW_BLOCK,
                                        num_warps=ROW_WARPS, **_LAUNCH)
    return out.view(x.shape[:-1])


def sum_last(x: torch.Tensor, keepdim: bool = False, mean: bool = False,
             dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    s = row_sum(x)
    if mean:
        s = s / x.shape[-1]
    if keepdim:
        s = s.unsqueeze(-1)
    return s.to(dtype or x.dtype)


def _softmax(x: torch.Tensor, dim: int, log: bool, dtype: Optional[torch.dtype]) -> torch.Tensor:
    _need(x, "log_softmax" if log else "softmax")
    if dtype is not None and dtype != x.dtype:
        x = x.to(dtype)
    dim = dim % x.dim() if x.dim() else 0
    moved = x.movedim(dim, -1) if dim != x.dim() - 1 else x
    x2, n = _rows(moved)
    out = torch.empty_like(x2)
    if x2.shape[0] and n:
        _softmax_kernel[(x2.shape[0],)](x2, out, n, x2.stride(0), out.stride(0), LOG=log,
                                        BLOCK=ROW_BLOCK, num_warps=ROW_WARPS, **_LAUNCH)
    out = out.view(moved.shape)
    return out.movedim(-1, dim) if dim != x.dim() - 1 else out


def softmax(x: torch.Tensor, dim: int = -1, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    return _softmax(x, dim, False, dtype)


def log_softmax(x: torch.Tensor, dim: int = -1, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    return _softmax(x, dim, True, dtype)


def _norm(x, weight, bias, eps, rms: bool, n_norm: int) -> torch.Tensor:
    _need(x, "rms_norm" if rms else "layer_norm")
    if n_norm != 1:
        # normalized over the last k dims: flatten them into one row
        lead = x.shape[:-n_norm]
        x = x.reshape(*lead, -1)
        weight = weight.reshape(-1) if weight is not None else None
        bias = bias.reshape(-1) if bias is not None else None
        out = _norm(x, weight, bias, eps, rms, 1)
        return out
    x2, n = _rows(x)
    out = torch.empty_like(x2)
    w = weight.contiguous() if weight is not None else x2
    b = bias.contiguous() if bias is not None else x2
    if x2.shape[0]:
        _norm_kernel[(x2.shape[0],)](x2, w, b, out, n, x2.stride(0), out.stride(0), float(eps),
                                     RMS=rms, HAS_W=weight is not None, HAS_B=bias is not None,
                                     BLOCK=ROW_BLOCK, num_warps=ROW_WARPS, **_LAUNCH)
    return out.view(x.shape)


def rms_norm(x: torch.Tensor, normalized_shape: Sequence[int], weight: Optional[torch.Tensor] = None,
             eps: Optional[float] = None) -> torch.Tensor:
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    out = _norm(x, weight, None, eps, True, len(tuple(normalized_shape)))
    return out.view(x.shape)


def layer_norm(x: torch.Tensor, normalized_shape: Sequence[int], weight: Optional[torch.Tensor] = None,
               bias: Optional[torch.Tensor] = None, eps: float = 1e-5) -> torch.Tensor:
    out = _norm(x, weight, bias, eps, False, len(tuple(normalized_shape)))
    return out.view(x.shape)


# ----------------------------------------------------------------------------- attention


ANCHOR = True  # False only in tests that break the kernel on purpose


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
              is_causal: bool = False, scale: Optional[float] = None,
              enable_gqa: bool = False) -> torch.Tensor:
    """scaled_dot_product_attention with one kernel for the training forward, prefill and decode.

    q [B, Hq, Lq, D], k and v [B, Hkv, Lk, D]; `attn_mask` bool, broadcastable to [B, Hq, Lq, Lk]
    (True = attend); `is_causal` is PyTorch's top left causal mask. A fully masked query row
    gives zeros (PyTorch's math path gives zeros too). Key chunks start at the anchor, so a row
    gets the same bits whatever its padding, the cache capacity or the number of queries."""
    _need(q, "attention")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise RLKernelError("attention: q, k and v must have one dtype")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise RLKernelError("attention: q, k, v must be [batch, heads, seq, head_dim]")
    B, Hq, Lq, D = q.shape
    Bk, Hk, Lk, Dk = k.shape
    if Bk != B or v.shape[:3] != k.shape[:3] or Dk != D or v.shape[-1] != D:
        raise RLKernelError(f"attention: shapes q {tuple(q.shape)}, k {tuple(k.shape)}, "
                            f"v {tuple(v.shape)} are not supported (value head dim must equal D)")
    if Hk != Hq and not (enable_gqa and Hq % Hk == 0):
        raise RLKernelError(f"attention: {Hq} query heads on {Hk} key heads needs enable_gqa")
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        raise RLKernelError("attention: only boolean masks are supported (an additive float mask "
                            "is not, convert it to bool)")
    if is_causal and attn_mask is not None:
        raise RLKernelError("attention: pass either attn_mask or is_causal")
    scale = float(scale) if scale is not None else 1.0 / (D ** 0.5)
    out = torch.empty((B, Hq, Lq, D), device=q.device, dtype=q.dtype)
    mask = attn_mask.expand(B, Hq, Lq, Lk) if attn_mask is not None else None
    if out.numel() == 0:
        return out
    BD = max(16, triton.next_power_of_2(D))
    if mask is None:
        mask_arg, ms = _dummy_mask(q.device), (0, 0, 0, 0)
    else:
        mask_arg, ms = mask, mask.stride()
    grid = (triton.cdiv(Lq, ATT_BM), B * Hq)
    _attn_kernel[grid](q, k, v, out, mask_arg, Lq, Lk, Hq, Hq // Hk, scale,
                       *q.stride(), *k.stride(), *v.stride(), *out.stride(), *ms,
                       int(mask is not None), int(bool(is_causal)), int(ANCHOR),
                       D=D, BD=BD, BM=ATT_BM, BN=ATT_BN, PREC=_prec(q.dtype),
                       num_warps=ATT_WARPS, **_LAUNCH)
    return out
