"""Triton kernels for the chunked recurrence (docs/LINEAR_ATTENTION.md), CUDA only.

Two kernels, both written from the identities in chunked.py:

* ``_la_prep`` (delta rule only), one program per (chunk, batch and head): the chunk's unit lower
  triangular system ``(I + M)`` is inverted by forward substitution in registers, and
  ``Wv = (I + M)^-1 beta V`` and ``Wk = (I + M)^-1 beta (k * exp(c))`` are stored as float32. These
  depend on the chunk's own inputs only, so every chunk runs in parallel.
* ``_la_fwd``, one program per (block of value columns, batch and head): walks the chunks in order
  0, 1, 2, ... and keeps its [dk, BV] slice of the state in float32 registers. Per chunk it does
  five tensor core or FMA dots and nothing else. A column of the state never mixes with another
  column (both rules act on columns independently), so splitting the value width into blocks
  changes no arithmetic.

Determinism. No atomics and no split along the sequence: every output element and state entry is
computed by exactly one program, in a fixed order. The order of the sums is set by the chunk size
(the reduction order parameter), the key block (always the whole key width, never split) and the
dot layouts the launch config gives. ``config(..., deterministic=True)`` pins all of them to values
that depend on the call's dims only, never on the GPU's SM count, so a 26 SM and a 70 SM GPU of one
architecture run the same binary on the same data in the same order.

Decay kinds: ``none`` and ``scalar`` (one log decay per head and step). The per key channel decay
of gated linear attention and Kimi Delta Attention runs in chunked.py only (``kernel_supports``).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .reference import LinAttnError

try:  # Triton is optional for the rest of Truestar
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only without Triton
    triton = None
    tl = None

CHUNKS = (16, 32, 64)        # chunk sizes the kernels compile (tl.dot needs at least 16)
DET_CHUNK = 16               # the deterministic chunk: every GPU, every batch, every length
BV_DEFAULT = 32              # value columns per program (capped by dv); never changes the bits
PREP_WARPS = 4               # the prep kernel's warps set its reduction order: fixed in every mode
MAX_DK = 256
MAX_DV = 256


def available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _pow2(n: int) -> int:
    return 1 << max(4, (int(n) - 1).bit_length())


def kernel_supports(kind: str, dk: int, dv: int) -> Optional[str]:
    """None when the Triton kernels run this call, else the reason they do not."""
    if kind == "channel":
        return "per key channel decay runs in the chunked PyTorch form only (no Triton kernel yet)"
    if dk > MAX_DK or dv > MAX_DV:
        return f"head dims above {MAX_DK} (dk={dk}, dv={dv}) are not compiled"
    return None


def _warps(BK: int) -> int:
    return 4 if BK <= 64 else 8


def config(B: int, H: int, dk: int, dv: int, *, deterministic: bool, chunk: Optional[int] = None) -> Dict[str, int]:
    """Launch config, from the call's dims only (never the SM count, the batch or the length).

    Measured 2026-11-24 on the RTX 5070 Ti (T 4096, strict float32, scripts/bench_linattn.py
    --sweep): chunk 16 with 32 value columns per program and 4 warps (key width up to 64) or 8
    warps (128) was the fastest of 18 configs in 4 of 8 cases and within 1.71x in all 8, while chunk
    64 with 64 columns was 11x to 21x slower than the fastest (register spills of the FMA dots). The value block and the forward
    kernel's warps do not change the bits (tests/test_linattn.py); the chunk does, so deterministic
    mode pins it to DET_CHUNK and refuses another (TS1063)."""
    BK, BDV = _pow2(dk), _pow2(dv)
    if deterministic and chunk not in (None, DET_CHUNK):
        raise LinAttnError(f"[TS1063] deterministic mode pins the chunk to {DET_CHUNK} (the reduction order "
                           f"must not depend on the caller or the GPU); got chunk={chunk}")
    c = DET_CHUNK if chunk is None else chunk
    if c not in CHUNKS:
        raise LinAttnError(f"[TS1061] the Triton kernels compile chunk sizes {CHUNKS}, got {c}")
    return {"CHUNK": c, "BK": BK, "BV": min(BV_DEFAULT, BDV), "num_warps": _warps(BK)}


def check_config(cfg: Dict[str, int], dk: int, dv: int, deterministic: bool) -> None:
    if deterministic and cfg["CHUNK"] != DET_CHUNK:
        raise LinAttnError(f"[TS1063] launch config {cfg} does not use the pinned deterministic chunk {DET_CHUNK}")
    if cfg["CHUNK"] not in CHUNKS or cfg["BK"] < dk or cfg["BV"] < 16 or cfg["BV"] & (cfg["BV"] - 1):
        raise LinAttnError(f"[TS1061] invalid launch config {cfg} for dk={dk}, dv={dv}")


if triton is not None:

    @triton.jit
    def _la_prep(k_ptr, v_ptr, g_ptr, b_ptr, wk_ptr, wv_ptr,
                 T, H, DK, DV,
                 skb, skt, skh, svb, svt, svh, sgb, sgt, sgh, sbb, sbt, sbh,
                 HAS_G: tl.constexpr, C: tl.constexpr, BK: tl.constexpr, BDV: tl.constexpr):
        n = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        r = tl.arange(0, C)
        t = n * C + r
        tm = t < T
        dk = tl.arange(0, BK)
        dvv = tl.arange(0, BDV)
        k = tl.load(k_ptr + b * skb + h * skh + t[:, None] * skt + dk[None, :],
                    mask=tm[:, None] & (dk[None, :] < DK), other=0.0).to(tl.float32)
        v = tl.load(v_ptr + b * svb + h * svh + t[:, None] * svt + dvv[None, :],
                    mask=tm[:, None] & (dvv[None, :] < DV), other=0.0).to(tl.float32)
        beta = tl.load(b_ptr + b * sbb + h * sbh + t * sbt, mask=tm, other=0.0).to(tl.float32)
        if HAS_G:
            g = tl.load(g_ptr + b * sgb + h * sgh + t * sgt, mask=tm, other=0.0).to(tl.float32)
            cum = tl.cumsum(g, 0)
        else:
            cum = tl.zeros([C], dtype=tl.float32)
        kk = tl.dot(k, tl.trans(k), input_precision="ieee")
        strict = r[:, None] > r[None, :]
        decay = tl.exp(tl.where(strict, cum[:, None] - cum[None, :], float("-inf")))
        M = kk * decay * beta[:, None]
        # (I + M)^-1 by forward substitution: row i = e_i - sum_{j<i} M_ij row_j
        X = tl.where(r[:, None] == r[None, :], 1.0, 0.0)
        for i in range(1, C):
            m_row = tl.sum(tl.where(r[:, None] == i, M, 0.0), axis=0)
            upd = tl.sum(m_row[:, None] * X, axis=0)
            X = tl.where(r[:, None] == i, X - upd[None, :], X)
        kc = k * tl.exp(cum)[:, None] * beta[:, None]
        wk = tl.dot(X, kc, input_precision="ieee")
        wv = tl.dot(X, v * beta[:, None], input_precision="ieee")
        base = bh * (tl.num_programs(0) * C)
        tl.store(wk_ptr + (base + t[:, None]) * BK + dk[None, :], wk)
        tl.store(wv_ptr + (base + t[:, None]) * BDV + dvv[None, :], wv)

    @triton.jit
    def _la_fwd(q_ptr, k_ptr, v_ptr, g_ptr, wk_ptr, wv_ptr, s0_ptr, o_ptr, sf_ptr,
                T, H, DK, DV, NCH, scale,
                sqb, sqt, sqh, skb, skt, skh, svb, svt, svh, sgb, sgt, sgh, sob, sot, soh,
                HAS_G: tl.constexpr, DELTA: tl.constexpr, HAS_S0: tl.constexpr, STORE_S: tl.constexpr,
                C: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BDV: tl.constexpr,
                PREC: tl.constexpr):
        vb = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        r = tl.arange(0, C)
        dk = tl.arange(0, BK)
        cv = vb * BV + tl.arange(0, BV)
        cm = cv < DV
        km = dk < DK
        lower = r[:, None] >= r[None, :]
        if HAS_S0:
            S = tl.load(s0_ptr + bh * DK * DV + dk[:, None] * DV + cv[None, :],
                        mask=km[:, None] & cm[None, :], other=0.0).to(tl.float32)
        else:
            S = tl.zeros([BK, BV], dtype=tl.float32)
        for n in range(0, NCH):
            t = n * C + r
            tm = t < T
            q = tl.load(q_ptr + b * sqb + h * sqh + t[:, None] * sqt + dk[None, :],
                        mask=tm[:, None] & km[None, :], other=0.0).to(tl.float32)
            k = tl.load(k_ptr + b * skb + h * skh + t[:, None] * skt + dk[None, :],
                        mask=tm[:, None] & km[None, :], other=0.0).to(tl.float32)
            if HAS_G:
                g = tl.load(g_ptr + b * sgb + h * sgh + t * sgt, mask=tm, other=0.0).to(tl.float32)
                cum = tl.cumsum(g, 0)
                last = tl.sum(tl.where(r == C - 1, cum, 0.0), axis=0)
                # the decays are row scales: (q * e^c) S = e^c (q S) and (k * e^(l - c))^T U =
                # k^T (e^(l - c) U), so q and k stay the only key width operands (shared memory)
                rs_q = tl.exp(cum)
                rs_u = tl.exp(last - cum)
                A = tl.dot(q, tl.trans(k), input_precision=PREC) * \
                    tl.exp(tl.where(lower, cum[:, None] - cum[None, :], float("-inf")))
            else:
                A = tl.where(lower, tl.dot(q, tl.trans(k), input_precision=PREC), 0.0)
            if DELTA:
                base = bh * (NCH * C)
                wk = tl.load(wk_ptr + (base + t[:, None]) * BK + dk[None, :])
                wv = tl.load(wv_ptr + (base + t[:, None]) * BDV + cv[None, :], mask=cm[None, :], other=0.0)
                U = wv - tl.dot(wk, S, input_precision=PREC)
            else:
                U = tl.load(v_ptr + b * svb + h * svh + t[:, None] * svt + cv[None, :],
                            mask=tm[:, None] & cm[None, :], other=0.0).to(tl.float32)
            qS = tl.dot(q, S, input_precision=PREC)
            if HAS_G:
                qS = qS * rs_q[:, None]
            o = (qS + tl.dot(A, U, input_precision=PREC)) * scale
            tl.store(o_ptr + b * sob + h * soh + t[:, None] * sot + cv[None, :],
                     o.to(o_ptr.dtype.element_ty), mask=tm[:, None] & cm[None, :])
            if HAS_G:
                S = S * tl.exp(last) + tl.dot(tl.trans(k), U * rs_u[:, None], input_precision=PREC)
            else:
                S = S + tl.dot(tl.trans(k), U, input_precision=PREC)
        if STORE_S:
            tl.store(sf_ptr + bh * DK * DV + dk[:, None] * DV + cv[None, :], S,
                     mask=km[:, None] & cm[None, :])


def forward(q, k, v, g, beta, *, rule: str, scale: float, initial_state, cfg: Dict[str, int],
            allow_tf32: bool = False, store_state: bool = True):
    """Run the kernels. q, k, v: [B, T, H, d] (any strides with a unit last stride), g: [B, T, H]
    or None, beta: [B, T, H] for the delta rule. Returns (o [B, T, H, dv] in v's dtype, final state
    [B, H, dk, dv] float32 or None)."""
    if not available():
        raise LinAttnError("[TS1061] the linear attention kernels need Triton and a CUDA GPU")
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    C, BK, BV = cfg["CHUNK"], cfg["BK"], cfg["BV"]
    BDV = _pow2(DV)
    dev = q.device
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    o = torch.empty(B, T, H, DV, dtype=v.dtype, device=dev)
    NCH = -(-T // C)
    sf = torch.empty(B, H, DK, DV, dtype=torch.float32, device=dev) if store_state else o
    s0 = initial_state.contiguous() if initial_state is not None else o
    has_g = g is not None
    gg = g if has_g else q
    gs = (g.stride(0), g.stride(1), g.stride(2)) if has_g else (0, 0, 0)
    delta = rule == "delta"
    if delta:
        wk = torch.empty(B * H * NCH * C * BK, dtype=torch.float32, device=dev)
        wv = torch.empty(B * H * NCH * C * BDV, dtype=torch.float32, device=dev)
        if NCH:
            _la_prep[(NCH, B * H)](
                k, v, gg, beta, wk, wv, T, H, DK, DV,
                k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
                *gs, beta.stride(0), beta.stride(1), beta.stride(2),
                HAS_G=has_g, C=C, BK=BK, BDV=BDV, num_warps=PREP_WARPS, num_stages=1)
    else:
        wk = wv = o
    prec = "tf32" if allow_tf32 else "ieee"
    _la_fwd[(-(-DV // BV), B * H)](
        q, k, v, gg, wk, wv, s0, o, sf, T, H, DK, DV, NCH, float(scale),
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), *gs, o.stride(0), o.stride(1), o.stride(2),
        HAS_G=has_g, DELTA=delta, HAS_S0=initial_state is not None, STORE_S=store_state,
        C=C, BK=BK, BV=BV, BDV=BDV, PREC=prec, num_warps=cfg["num_warps"], num_stages=1)
    return o, (sf if store_state else None)
