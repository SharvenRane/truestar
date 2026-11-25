"""Compiled ``attention_grad`` (docs/TRAINING.md, Attention backward; the IR op is ir/attention_grad.py).

CUDA: three Triton kernels, written for Truestar.

1. ``_rowstats``: one program per (query block, batch row). An online pass over the key blocks
   gives each query row's log2 sum of exponentials ``lse`` of the scaled scores, and
   ``delta = rowsum(do * o)``. Both float32, to a workspace.
2. ``_dkdv``: one program per (key block, batch row). It loops over the query blocks that can see
   its keys (from its own block on when causal) in increasing order, recomputes the probabilities
   ``p = exp2(s - lse)`` and accumulates ``dv += p^T do`` and ``dk += ds^T q`` with
   ``ds = p (do v^T - delta)``.
3. ``_dq``: one program per (query block, batch row), looping over the key blocks it sees in
   increasing order: ``dq += ds k``.

Every output element is written by exactly one program and every sum runs in a fixed order with
block sizes chosen from the shapes alone (no atomics, no autotuning, no split), so the result has
the same bits on every call and after a recompile. float32 products use IEEE float32 (``tl.dot``
with ``input_precision="ieee"``) unless the program allows TF32; 16 bit operands multiply in their
dtype with float32 accumulation, like the forward kernel.

CPU: the same three passes in float32 torch over query blocks of 64 rows (the same formula).
"""

from __future__ import annotations

import math
from typing import List

import torch

LOG2E = 1.4426950408889634

try:  # Triton is optional (CPU only installs)
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None


def configs(d: int, dtype: torch.dtype):
    """(BM, BN, warps, stages) per kernel: a function of the head dim and dtype only, never of the
    data, the GPU or its load, so the summation order is fixed by the shapes. One stage and tiles
    that fit 64 KiB of shared memory (the smallest of the GPUs Truestar runs on) so no GPU needs a
    different table; picked with scripts/bench_attention_grad.py --sweep on an L4."""
    if d <= 64:
        return {"rowstats": (64, 64, 4, 1), "dkdv": (32, 32, 4, 1), "dq": (32, 32, 4, 1)}
    return {"rowstats": (32, 32, 4, 1), "dkdv": (32, 32, 4, 1), "dq": (32, 32, 4, 1)}


if triton is not None:

    @triton.jit
    def _rowstats(Q, K, O, DO, LSE, DELTA, sb, sh, st, ob, oh, ot, H, T, D, qk_scale,
                  CAUSAL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
                  PREC: tl.constexpr):
        pm = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        base = b.to(tl.int64) * sb + h.to(tl.int64) * sh
        obase = b.to(tl.int64) * ob + h.to(tl.int64) * oh
        rm = pm * BM + tl.arange(0, BM)
        rd = tl.arange(0, BD)
        mm = rm < T
        md = rd < D
        q = tl.load(Q + base + rm[:, None] * st + rd[None, :], mask=mm[:, None] & md[None, :], other=0.0)
        o = tl.load(O + obase + rm[:, None] * ot + rd[None, :], mask=mm[:, None] & md[None, :], other=0.0)
        do = tl.load(DO + obase + rm[:, None] * ot + rd[None, :], mask=mm[:, None] & md[None, :], other=0.0)
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        m = tl.full((BM,), float("-inf"), tl.float32)
        l = tl.zeros((BM,), tl.float32)
        hi = tl.minimum((pm + 1) * BM, T) if CAUSAL else T
        for n0 in range(0, hi, BN):
            rn = n0 + tl.arange(0, BN)
            mn = rn < T
            kt = tl.load(K + base + rn[None, :] * st + rd[:, None], mask=mn[None, :] & md[:, None], other=0.0)
            s = tl.dot(q, kt, input_precision=PREC) * qk_scale
            keep = mn[None, :]
            if CAUSAL:
                keep = keep & (rn[None, :] <= rm[:, None])
            s = tl.where(keep, s, float("-inf"))
            mnew = tl.maximum(m, tl.max(s, axis=1))
            l = l * tl.exp2(m - mnew) + tl.sum(tl.exp2(s - mnew[:, None]), axis=1)
            m = mnew
        lse = m + tl.log2(l)
        tl.store(LSE + bh.to(tl.int64) * T + rm, lse, mask=mm)
        tl.store(DELTA + bh.to(tl.int64) * T + rm, delta, mask=mm)

    @triton.jit
    def _dkdv_block(Q, DO, LSE, DELTA, base, obase, st, ot, bh, T, m0, rn, rd, mn, md, k, v, dk, dv, qk_scale,
                    CAUSAL_BLOCK: tl.constexpr, BM: tl.constexpr, PREC: tl.constexpr):
        rm = m0 + tl.arange(0, BM)
        mm = rm < T
        qt = tl.load(Q + base + rm[None, :] * st + rd[:, None], mask=mm[None, :] & md[:, None], other=0.0)
        q = tl.load(Q + base + rm[:, None] * st + rd[None, :], mask=mm[:, None] & md[None, :], other=0.0)
        dot = tl.load(DO + obase + rm[None, :] * ot + rd[:, None], mask=mm[None, :] & md[:, None], other=0.0)
        do = tl.load(DO + obase + rm[:, None] * ot + rd[None, :], mask=mm[:, None] & md[None, :], other=0.0)
        lse = tl.load(LSE + bh.to(tl.int64) * T + rm, mask=mm, other=0.0)
        delta = tl.load(DELTA + bh.to(tl.int64) * T + rm, mask=mm, other=0.0)
        st_ = tl.dot(k, qt, input_precision=PREC) * qk_scale                 # [BN, BM]
        keep = mm[None, :] & mn[:, None]
        if CAUSAL_BLOCK:
            keep = keep & (rn[:, None] <= rm[None, :])
        pt = tl.where(keep, tl.exp2(st_ - lse[None, :]), 0.0)
        dv += tl.dot(pt.to(do.dtype), do, input_precision=PREC)
        dpt = tl.dot(v, dot, input_precision=PREC)
        dst = pt * (dpt - delta[None, :])
        dk += tl.dot(dst.to(q.dtype), q, input_precision=PREC)
        return dk, dv

    @triton.jit
    def _dkdv(Q, K, V, DO, LSE, DELTA, DK, DV, sb, sh, st, ob, oh, ot, gb, gh, gt, H, T, D, qk_scale, scale,
              CAUSAL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
              PREC: tl.constexpr):
        pn = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        base = b.to(tl.int64) * sb + h.to(tl.int64) * sh
        obase = b.to(tl.int64) * ob + h.to(tl.int64) * oh
        rn = pn * BN + tl.arange(0, BN)
        rd = tl.arange(0, BD)
        mn = rn < T
        md = rd < D
        k = tl.load(K + base + rn[:, None] * st + rd[None, :], mask=mn[:, None] & md[None, :], other=0.0)
        v = tl.load(V + base + rn[:, None] * st + rd[None, :], mask=mn[:, None] & md[None, :], other=0.0)
        dk = tl.zeros((BN, BD), tl.float32)
        dv = tl.zeros((BN, BD), tl.float32)
        # computed transposed (scores as [keys, queries]) so no product result is transposed in
        # registers; the query blocks on the diagonal take the causal mask, the later ones do not
        lo = (pn * BN // BM) * BM if CAUSAL else 0
        full = tl.minimum(((pn * BN + BN - 1) // BM + 1) * BM, T) if CAUSAL else lo
        for m0 in range(lo, full, BM):
            dk, dv = _dkdv_block(Q, DO, LSE, DELTA, base, obase, st, ot, bh, T, m0, rn, rd, mn, md, k, v, dk, dv,
                                 qk_scale, True, BM, PREC)
        for m0 in range(full, T, BM):
            dk, dv = _dkdv_block(Q, DO, LSE, DELTA, base, obase, st, ot, bh, T, m0, rn, rd, mn, md, k, v, dk, dv,
                                 qk_scale, False, BM, PREC)
        gbase = b.to(tl.int64) * gb + h.to(tl.int64) * gh
        msk = mn[:, None] & md[None, :]
        tl.store(DK + gbase + rn[:, None] * gt + rd[None, :], (dk * scale).to(DK.dtype.element_ty), mask=msk)
        tl.store(DV + gbase + rn[:, None] * gt + rd[None, :], dv.to(DV.dtype.element_ty), mask=msk)

    @triton.jit
    def _dq_block(K, V, base, st, T, n0, rm, rd, mm, md, q, do, lse, delta, dq, qk_scale,
                  CAUSAL_BLOCK: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr):
        rn = n0 + tl.arange(0, BN)
        mn = rn < T
        kt = tl.load(K + base + rn[None, :] * st + rd[:, None], mask=mn[None, :] & md[:, None], other=0.0)
        k = tl.load(K + base + rn[:, None] * st + rd[None, :], mask=mn[:, None] & md[None, :], other=0.0)
        vt = tl.load(V + base + rn[None, :] * st + rd[:, None], mask=mn[None, :] & md[:, None], other=0.0)
        s = tl.dot(q, kt, input_precision=PREC) * qk_scale
        keep = mm[:, None] & mn[None, :]
        if CAUSAL_BLOCK:
            keep = keep & (rn[None, :] <= rm[:, None])
        p = tl.where(keep, tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, vt, input_precision=PREC)
        ds = p * (dp - delta[:, None])
        return dq + tl.dot(ds.to(k.dtype), k, input_precision=PREC)

    @triton.jit
    def _dq(Q, K, V, DO, LSE, DELTA, DQ, sb, sh, st, ob, oh, ot, gb, gh, gt, H, T, D, qk_scale, scale,
            CAUSAL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
            PREC: tl.constexpr):
        pm = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        base = b.to(tl.int64) * sb + h.to(tl.int64) * sh
        obase = b.to(tl.int64) * ob + h.to(tl.int64) * oh
        rm = pm * BM + tl.arange(0, BM)
        rd = tl.arange(0, BD)
        mm = rm < T
        md = rd < D
        q = tl.load(Q + base + rm[:, None] * st + rd[None, :], mask=mm[:, None] & md[None, :], other=0.0)
        do = tl.load(DO + obase + rm[:, None] * ot + rd[None, :], mask=mm[:, None] & md[None, :], other=0.0)
        lse = tl.load(LSE + bh.to(tl.int64) * T + rm, mask=mm, other=0.0)
        delta = tl.load(DELTA + bh.to(tl.int64) * T + rm, mask=mm, other=0.0)
        dq = tl.zeros((BM, BD), tl.float32)
        hi = tl.minimum((pm + 1) * BM, T) if CAUSAL else T
        full = (pm * BM // BN) * BN if CAUSAL else hi     # key blocks wholly below the diagonal
        for n0 in range(0, full, BN):
            dq = _dq_block(K, V, base, st, T, n0, rm, rd, mm, md, q, do, lse, delta, dq, qk_scale, False, BN, PREC)
        for n0 in range(full, hi, BN):
            dq = _dq_block(K, V, base, st, T, n0, rm, rd, mm, md, q, do, lse, delta, dq, qk_scale, CAUSAL, BN, PREC)
        gbase = b.to(tl.int64) * gb + h.to(tl.int64) * gh
        tl.store(DQ + gbase + rm[:, None] * gt + rd[None, :], (dq * scale).to(DQ.dtype.element_ty),
                 mask=mm[:, None] & md[None, :])


def _as4(x: torch.Tensor) -> torch.Tensor:
    """[B, H, T, D] with unit stride on D and one shared (B, H, T) stride pattern."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    elif x.dim() > 4:
        x = x.reshape((-1,) + tuple(x.shape[-3:]))
    if x.stride(-1) != 1:
        x = x.contiguous()
    return x


def _triton(q, k, v, o, do, scale: float, causal: bool, tf32: bool, cfg=None, events=None) -> torch.Tensor:
    """``cfg`` overrides the block table and ``events`` (a list) collects one CUDA event before and
    after each kernel: both for scripts/bench_attention_grad.py only."""
    shape = q.shape
    q4, k4, v4, o4, do4 = (_as4(t) for t in (q, k, v, o, do))
    if not (q4.stride() == k4.stride() == v4.stride()):
        q4, k4, v4 = q4.contiguous(), k4.contiguous(), v4.contiguous()
    if o4.stride() != do4.stride():
        o4, do4 = o4.contiguous(), do4.contiguous()
    B, H, T, D = q4.shape
    out = torch.empty((3,) + tuple(q4.shape), dtype=q.dtype, device=q.device)
    lse = torch.empty((B * H, T), dtype=torch.float32, device=q.device)
    delta = torch.empty_like(lse)
    BD = max(16, 1 << (D - 1).bit_length())
    cfg = cfg or configs(D, q.dtype)
    prec = "tf32" if tf32 and q.dtype == torch.float32 else "ieee"
    sb, sh, st, _ = q4.stride()
    ob, oh, ot, _ = o4.stride()
    gb, gh, gt, _ = out[0].stride()
    qk_scale = float(scale) * LOG2E
    BM, BN, w, ns = cfg["rowstats"]
    _mark(events)
    _rowstats[(triton.cdiv(T, BM), B * H)](q4, k4, o4, do4, lse, delta, sb, sh, st, ob, oh, ot, H, T, D, qk_scale,
                                          CAUSAL=causal, BM=BM, BN=BN, BD=BD, PREC=prec,
                                          num_warps=w, num_stages=ns)
    BM, BN, w, ns = cfg["dkdv"]
    _mark(events)
    _dkdv[(triton.cdiv(T, BN), B * H)](q4, k4, v4, do4, lse, delta, out[1], out[2], sb, sh, st, ob, oh, ot,
                                      gb, gh, gt, H, T, D, qk_scale, float(scale), CAUSAL=causal, BM=BM, BN=BN,
                                      BD=BD, PREC=prec, num_warps=w, num_stages=ns)
    BM, BN, w, ns = cfg["dq"]
    _mark(events)
    _dq[(triton.cdiv(T, BM), B * H)](q4, k4, v4, do4, lse, delta, out[0], sb, sh, st, ob, oh, ot, gb, gh, gt,
                                    H, T, D, qk_scale, float(scale), CAUSAL=causal, BM=BM, BN=BN, BD=BD,
                                    PREC=prec, num_warps=w, num_stages=ns)
    _mark(events)
    return out.reshape((3,) + tuple(shape))


def _mark(events) -> None:
    if events is not None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        events.append(e)


def _blocked_torch(q, k, v, o, do, scale: float, causal: bool, block: int = 64) -> torch.Tensor:
    """The three passes in float32 torch, query blocks of ``block`` rows in increasing order."""
    dt = q.dtype
    q, k, v, o, do = (t.float() for t in (q, k, v, o, do))
    T = q.shape[-2]
    delta = (o * do).sum(-1, keepdim=True)
    dq = torch.empty_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    for m0 in range(0, T, block):
        m1 = min(T, m0 + block)
        hi = m1 if causal else T
        s = torch.matmul(q[..., m0:m1, :], k[..., :hi, :].transpose(-1, -2)) * scale
        if causal:
            rows = torch.arange(m0, m1, device=q.device)[:, None]
            cols = torch.arange(hi, device=q.device)[None, :]
            s = s.masked_fill(cols > rows, -math.inf)
        p = torch.softmax(s, dim=-1)
        dob = do[..., m0:m1, :]
        dv[..., :hi, :] += torch.matmul(p.transpose(-1, -2), dob)
        ds = p * (torch.matmul(dob, v[..., :hi, :].transpose(-1, -2)) - delta[..., m0:m1, :])
        dq[..., m0:m1, :] = torch.matmul(ds, k[..., :hi, :]) * scale
        dk[..., :hi, :] += torch.matmul(ds.transpose(-1, -2), q[..., m0:m1, :]) * scale
    return torch.stack([dq, dk, dv]).to(dt)


def attention_grad(q, k, v, o, do, scale: float, causal: bool, tf32: bool = False) -> torch.Tensor:
    if q.device.type == "meta":
        return torch.empty((3,) + tuple(q.shape), dtype=q.dtype, device="meta")
    if q.device.type == "cuda" and triton is not None and q.shape[-1] <= 128 and q.numel() > 0:
        return _triton(q, k, v, o, do, scale, causal, tf32)
    return _blocked_torch(q, k, v, o, do, scale, causal)


def tolerance_length(shape) -> int:
    """The longest sum of the op (for the kernel layer's growth factor)."""
    dims: List[int] = [d if isinstance(d, int) else 1024 for d in shape[-2:]]
    return max(dims)
