"""Quantized KV cache for decode attention: int8 or fp8 (e4m3) keys and values with a float32
scale per head and per group of tokens, dequantized inside a split KV decode kernel.

    from truestar import kvquant as KQ
    kc = KQ.quantize(k, "int8", group=32)        # k [B, Hkv, N, D] float16 or bfloat16
    vc = KQ.quantize(v, "int8", group=32)
    out = KQ.decode_attention(q, kc, vc)         # q [B, Hq, T, D], T up to 16, Hq a multiple of Hkv

Formats (``FORMATS``): ``int8`` stores round(x / s) with s = amax / 127 of each group of ``group``
tokens of one head (symmetric, per channel of nothing: one scale per head and group); ``fp8``
stores x / s as float8_e4m3fn with s = amax / 448, so the largest magnitude of a group maps to
e4m3's largest finite value. The scale tensors are float32, [B, Hkv, ceil(N / group)].

The kernel (one program per batch, query head and chunk of keys, as the generated split KV
attention) loads int8 or fp8 blocks, multiplies by the group scale in float32, accumulates scores,
the running max and sum and the output in float32, and stores each chunk's partial; a second
kernel merges the chunks in the fixed order 0, 1, 2, ... Grouped query heads read their KV head
(h // (Hq / Hkv)) directly, so a GQA cache is never expanded. Deterministic mode
(``deterministic=True``, or ``torch.use_deterministic_algorithms``) uses chunks of
``DET_CHUNK`` keys on every GPU; otherwise the chunk follows the shapes (chunk_for). No atomics: every run
gives the same bits.

Verification: the first call for each (shape, chunk, format) is compared with float64 attention
over the dequantized cache (the exact values the kernel reads); a mismatch raises
``truestar.kernel.KernelMismatchError``. ``error_bound`` gives the documented worst case error
against attention over the unquantized cache, and tests/test_kvquant.py measures that it holds.

Below 8 bits (``LOW_FORMATS``, kvquant/low.py): ``int4`` and ``nvfp4`` pack two channels a byte
with an E4M3 scale per token and 16 channels and a float32 scale per head and group of tokens;
``rot=True`` stores the cache Hadamard rotated along the head dim (the query and output are
rotated online by the fast Walsh kernel). The same first call check and the same proven
``error_bound`` hold, computed from the values in the unrotated basis (``QuantKV.original``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

FORMATS = ("int8", "fp8")      # 8 bit formats (the compiled decode also takes LOW_FORMATS, with block scales)
LOW_FORMATS = ("int4", "nvfp4")  # below 8 bits, kvquant/low.py: decode_attention and kvq_attention
QMAX = {"int8": 127.0, "fp8": 448.0}
# worst rounding error of one stored value: int8 rounds to the nearest of 255 levels, half a step
# (0.5 / 127 of the group's amax) plus float32 rounding of x / s (below 2^-16 of a step); e4m3
# keeps 3 mantissa bits, so a value in its normal range is within 2^-4 of itself (relative), and
# below it within half the smallest subnormal step (2^-10 times the group scale).
UNIT = {"int8": 0.5 / 127.0, "fp8": 2.0 ** -4}
DET_CHUNK = 256          # keys per chunk in deterministic mode, on every GPU
MIN_SERIAL_KEYS = DET_CHUNK + 1   # deterministic mode splits from this many keys (2048 before 2026-11-25)
TARGET_PROGRAMS = 256    # otherwise: chunks for about this many programs (a constant, not the SM count)
BLOCK_N = 64             # keys per inner step
MAX_ROWS = 16            # query rows per call (decode)


class KVQuantError(ValueError):
    """TS701: a quantized KV cache was asked for in a format, layout or shape that the kernel
    cannot run; nothing falls back to another path."""
    code = "TS701"


@dataclass
class QuantKV:
    """A quantized K or V cache: ``data`` [B, Hkv, N, D] int8 or float8_e4m3fn, float32 ``scale``
    [B, Hkv, ceil(N / group)] (one per head and group of tokens) or, with ``channel``,
    [B, Hkv, ceil(N / group), D] (one per head, channel and block of ``group`` tokens)."""
    data: torch.Tensor
    scale: torch.Tensor
    fmt: str
    group: int
    channel: bool = False
    bscale: Optional[torch.Tensor] = None   # 4 bit formats: E4M3 per token and 16 channels
    rot: bool = False                       # 4 bit formats: stored Hadamard rotated along D

    @property
    def low(self) -> bool:
        return self.fmt in LOW_FORMATS

    @property
    def shape(self):
        """[B, Hkv, N, D] of the cache (a 4 bit cache stores D / 2 bytes per token)."""
        B, H, N, X = self.data.shape
        return (B, H, N, 2 * X) if self.low else (B, H, N, X)

    def dequantize(self, dtype=torch.float64) -> torch.Tensor:
        """The values the kernel reads, exactly: data times its scale (in ``dtype``); for a
        rotated cache, in the rotated basis."""
        if self.low:
            from . import low
            return low.dequantize(self.data, self.bscale, self.scale, self.fmt, self.group, dtype)
        n = self.data.shape[2]
        s = self.scale.to(dtype).repeat_interleave(self.group, dim=2)[:, :, :n]
        return self.data.to(dtype) * (s if self.channel else s[..., None])

    def original(self) -> torch.Tensor:
        """The cache's values in the basis of the unquantized cache, float64: ``dequantize()``,
        turned back by H / D when the cache is rotated."""
        d = self.dequantize()
        if self.rot:
            from . import low
            d = low.unrotate(d)
        return d


def quantize_pages(pool: torch.Tensor, fmt: str = "int8", group: int = 32,
                   channel: bool = False, rot: bool = False) -> QuantKV:
    """Quantize a paged cache pool [P, Hkv, page, D] (P pages of ``page`` tokens); ``page`` must be
    a multiple of ``group`` so no scale group spans two pages. The result's data is
    [P, Hkv, page, D] and its scale [P, Hkv, page / group]."""
    _need(pool, "the page pool")
    if pool.dim() != 4 or pool.shape[2] % group:
        raise KVQuantError(f"[TS701] a page pool must be [P, Hkv, page, D] with page a multiple of "
                           f"the group {group}, got {tuple(pool.shape)}")
    return quantize(pool, fmt, group, channel, rot)


def gather_pages(pool: QuantKV, table: torch.Tensor, length: int) -> QuantKV:
    """The dense cache [B, Hkv, length, D] of the sequences whose pages ``table`` [B, pages] lists,
    gathered from a quantized pool with its scales: the same values and scales a dense cache
    quantized with the same group would hold, so paged and dense decode give the same bits."""
    P, H, page, X = pool.data.shape
    B, npg = table.shape
    if npg * page < length:
        raise KVQuantError(f"[TS701] {npg} pages of {page} tokens hold fewer than {length}")

    def tok(t):    # [P, H, page, X] -> [B, H, length, X] in table order
        return t[table].permute(0, 2, 1, 3, 4).reshape(B, H, npg * page, t.shape[3])[:, :, :length]

    data = tok(pool.data)
    bscale = None if pool.bscale is None else tok(pool.bscale).contiguous()
    per = page // pool.group
    sc = pool.scale[table]                      # [B, pages, H, per (, D)]
    if pool.channel:
        scale = sc.permute(0, 2, 1, 3, 4).reshape(B, H, npg * per, X)
    else:
        scale = sc.permute(0, 2, 1, 3).reshape(B, H, npg * per)
    scale = scale[:, :, : -(-length // pool.group)]
    return QuantKV(data.contiguous(), scale.contiguous(), pool.fmt, pool.group, pool.channel,
                   bscale, pool.rot)


def bytes_per_token(fmt: str, D: int, group: int, heads: int = 1, k_channel_block: int = 0) -> float:
    """KV memory per token for ``heads`` KV heads of head dim D: K and V values plus their share of
    the scales: V (and K unless ``k_channel_block``) one float32 per ``group`` tokens, per channel K
    D float32 per ``k_channel_block`` tokens. float16 caches store 2 bytes per value, no scales."""
    if fmt in ("float16", "bfloat16"):
        return 2 * heads * D * 2.0
    if fmt in LOW_FORMATS:
        from . import low
        return low.bytes_per_token(fmt, D, group, heads)
    k_scale = 4.0 * D / k_channel_block if k_channel_block else 4.0 / group
    return heads * (2 * D * 1.0 + 4.0 / group + k_scale)


def _need(t: torch.Tensor, what: str) -> None:
    if not isinstance(t, torch.Tensor) or t.device.type != "cuda":
        raise KVQuantError(f"[TS701] {what} must be a CUDA tensor")


def quantize(x: torch.Tensor, fmt: str = "int8", group: int = 32, channel: bool = False,
             rot: bool = False) -> QuantKV:
    """Quantize a K or V cache [B, Hkv, N, D] per head and per group of ``group`` tokens, or with
    ``channel`` per head, channel and block of ``group`` tokens (keys whose outliers sit in a few
    channels lose less there). One scale per channel per block of D x g tokens costs the same
    memory as one scale per group of g tokens.

    ``int4`` and ``nvfp4`` (kvquant/low.py) add an E4M3 scale per token and 16 channels;
    ``rot=True`` (4 bit formats only) stores the cache Hadamard rotated along the head dim."""
    _need(x, "the cache")
    if fmt not in FORMATS + LOW_FORMATS:
        raise KVQuantError(f"[TS701] unknown KV format {fmt!r}; use one of "
                           f"{', '.join(FORMATS + LOW_FORMATS)}")
    if fmt in LOW_FORMATS:
        from . import low
        if channel:
            raise KVQuantError(f"[TS701] {fmt} already scales every 16 channels; channel=True is "
                               f"for int8 and fp8 keys")
        low.check_shape(x, fmt, group, rot, KVQuantError)
        data, bscale, scale = low.codes(x, fmt, group, rot)
        return QuantKV(data, scale, fmt, group, False, bscale, rot)
    if rot:
        raise KVQuantError("[TS701] rot=True is for the 4 bit formats (int4, nvfp4); the int8 and "
                           "fp8 kernels read unrotated caches")
    if fmt == "fp8" and not hasattr(torch, "float8_e4m3fn"):
        raise KVQuantError("[TS701] this torch has no float8_e4m3fn")
    if channel:
        if group < 16 or group > 65536 or group & (group - 1):
            raise KVQuantError(f"[TS701] a per channel block must be a power of two of 16 to "
                               f"65536 tokens, got {group}")
    elif group not in (16, 32, 64, 128):
        raise KVQuantError(f"[TS701] group must be 16, 32, 64 or 128 tokens, got {group}")
    if x.dim() != 4:
        raise KVQuantError(f"[TS701] the cache must be [B, Hkv, N, D], got {tuple(x.shape)}")
    B, H, N, D = x.shape
    ng = -(-N // group)
    pad = ng * group - N
    xf = x.float()
    if pad:
        xf = torch.cat([xf, xf.new_zeros(B, H, pad, D)], dim=2)
    g = xf.view(B, H, ng, group, D)
    amax = g.abs().amax(dim=3) if channel else g.abs().amax(dim=(3, 4))
    scale = torch.where(amax > 0, amax / QMAX[fmt], torch.ones_like(amax))
    y = g / (scale[:, :, :, None, :] if channel else scale[..., None, None])
    if fmt == "int8":
        q = torch.round(y).clamp(-127, 127).to(torch.int8)
    else:
        q = y.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    q = q.view(B, H, ng * group, D)[:, :, :N].contiguous()
    return QuantKV(q, scale.contiguous(), fmt, group, channel)


def _sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def chunk_for(B: int, Hq: int, N: int, T: int = 1, deterministic: Optional[bool] = None,
              sms: Optional[int] = None, Hkv: Optional[int] = None) -> int:
    """Keys per chunk for a call, or N (one chunk: no split). Shapes decide, never the GPU.

    Deterministic mode: DET_CHUNK keys per chunk from MIN_SERIAL_KEYS keys (the context length
    decides, nothing else). Otherwise the smallest power of two chunk (at least BLOCK_N keys) that
    gives about TARGET_PROGRAMS programs: one per KV head with grouped query heads (``Hkv``, the
    grouped kernel), else one per query head and row. ``sms`` is accepted and ignored (before
    2026-11-25 the chunk followed the SM count, and short contexts below 2048 keys never split)."""
    if deterministic is None:
        deterministic = torch.are_deterministic_algorithms_enabled()
    if deterministic:
        return DET_CHUNK if N >= MIN_SERIAL_KEYS else N
    progs = B * Hkv if Hkv else B * Hq * T
    if progs >= TARGET_PROGRAMS or N <= BLOCK_N:
        return N
    chunks = -(-TARGET_PROGRAMS // progs)
    c = BLOCK_N
    while c * chunks < N:
        c *= 2
    return min(c, N)


def error_bound(q: torch.Tensor, kc: QuantKV, vc: QuantKV, k: torch.Tensor, v: torch.Tensor,
                scale: Optional[float] = None) -> torch.Tensor:
    """Worst case absolute error, per query row and output column, of attention over the quantized
    cache against attention over the unquantized cache (k, v), from the dequantized values:

        |o' - o| <= (exp(2 E) - 1) * sum_j w_j |v_j - o| + max_j |v'_j - v_j|

    with w and o the exact (unquantized) weights and output, E = max_j |s'_j - s_j| the largest
    change of a scaled score. Why: o' - o = sum_j (w'_j - w_j)(v_j - o) + sum_j w'_j (v'_j - v_j),
    since both weight vectors sum to one; a score change of at most E moves each weight by a factor
    within [exp(-2 E), exp(2 E)], so |w'_j - w_j| <= (exp(2 E) - 1) w_j; and the w' sum to one.
    Computed in float64 from this cache's actual rounding errors: an upper bound, not an estimate.
    Returns a tensor [B, Hq, T, D]."""
    return error_bound_values(q, kc.original(), vc.original(), k, v, scale)


def error_bound_values(q: torch.Tensor, kd: torch.Tensor, vd: torch.Tensor, k: torch.Tensor,
                       v: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    """``error_bound`` from the values a kernel reads (kd, vd [B, Hkv, N, D]) and the unquantized
    ones (k, v): for caches whose newest tokens stay unquantized (a float tail), pass both with the
    tail appended; its exact tokens add nothing to the bound."""
    B, Hq, T, D = q.shape
    G = Hq // kd.shape[1]
    sc = D ** -0.5 if scale is None else scale
    kd, vd = kd.double(), vd.double()
    k64, v64 = k.double(), v.double()
    rep = lambda t: t.repeat_interleave(G, dim=1)  # noqa: E731
    qd = q.double()
    kr, vr = rep(k64), rep(v64)
    w = torch.softmax((qd @ kr.transpose(-1, -2)) * sc, dim=-1)          # [B, Hq, T, N]
    o = w @ vr                                                            # [B, Hq, T, D]
    E = ((qd @ rep(kd - k64).transpose(-1, -2)).abs() * sc).amax(dim=-1, keepdim=True)
    spread = torch.einsum("bhtn,bhtnd->bhtd", w, (vr[:, :, None] - o[:, :, :, None]).abs())
    dv = rep(vd - v64).abs().amax(dim=2, keepdim=True)
    return torch.expm1(2 * E) * spread + dv


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def kvq_decode(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, ws_ptr, tk_ptr, tv_ptr, tlen_ptr, plen_ptr,
                   T, N, HQ, G, NG, NQ, sqb, sqh, sqt, skb, skh, skn, ssb, ssh,
                   svb, svh, stb, sth, stn, qk_scale, GROUP: tl.constexpr, VGROUP: tl.constexpr,
                   K_CHANNEL: tl.constexpr, HAS_TAIL: tl.constexpr, HAS_PLEN: tl.constexpr,
                   CHUNK: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
        pbh = tl.program_id(0)            # batch * query heads + query head
        c = tl.program_id(1)              # chunk
        t = tl.program_id(2)              # query row
        b = pbh // HQ
        h = pbh % HQ
        hk = h // G                       # the KV head this query head reads
        rd = tl.arange(0, D)
        q = tl.load(q_ptr + b * sqb + h * sqh + t * sqt + rd).to(tl.float32) * qk_scale
        m = tl.full((), float("-inf"), tl.float32)
        l = tl.zeros((), tl.float32)
        acc = tl.zeros((D,), tl.float32)
        lo = c * CHUNK
        if HAS_PLEN:          # a fixed capacity prefix: only its first plen keys are live
            hi = tl.minimum(lo + CHUNK, tl.minimum(N, tl.load(plen_ptr)))
        else:
            hi = tl.minimum(lo + CHUNK, N)
        if HAS_TAIL:          # the last program of a row reads the float tail instead
            if c == NQ:
                hi = lo
                tlen = tl.load(tlen_ptr)
                tkb = tk_ptr + b * stb + hk * sth
                tvb = tv_ptr + b * stb + hk * sth
                for n0 in range(0, tlen, BN):
                    rn = n0 + tl.arange(0, BN)
                    mn = rn < tlen
                    kk = tl.load(tkb + rn[:, None] * stn + rd[None, :], mask=mn[:, None], other=0.0)
                    s = tl.sum(kk.to(tl.float32) * q[None, :], axis=1)
                    s = tl.where(mn, s, float("-inf"))
                    mnew = tl.maximum(m, tl.max(s, axis=0))
                    msafe = tl.where(mnew == float("-inf"), 0.0, mnew)
                    alpha = tl.exp2(m - msafe)
                    p = tl.exp2(s - msafe)
                    vv = tl.load(tvb + rn[:, None] * stn + rd[None, :], mask=mn[:, None], other=0.0)
                    l = l * alpha + tl.sum(p, axis=0)
                    acc = acc * alpha + tl.sum(p[:, None] * vv.to(tl.float32), axis=0)
                    m = mnew
        kb = k_ptr + b * skb + hk * skh
        vb = v_ptr + b * skb + hk * skh
        sb = b * ssb + hk * ssh
        vsb = b * svb + hk * svh
        for n0 in range(lo, hi, BN):
            rn = n0 + tl.arange(0, BN)
            mn = rn < hi
            vsc = tl.load(vs_ptr + vsb + rn // VGROUP, mask=mn, other=0.0)
            kk = tl.load(kb + rn[:, None] * skn + rd[None, :], mask=mn[:, None], other=0.0)
            if K_CHANNEL:     # one scale per channel and block of GROUP tokens: [blocks, D]
                ksc = tl.load(ks_ptr + sb + (rn // GROUP)[:, None] * D + rd[None, :],
                              mask=mn[:, None], other=0.0)
                kk = kk.to(tl.float32) * ksc
            else:
                ksc = tl.load(ks_ptr + sb + rn // GROUP, mask=mn, other=0.0)
                kk = kk.to(tl.float32) * ksc[:, None]
            s = tl.sum(kk * q[None, :], axis=1)
            s = tl.where(mn, s, float("-inf"))
            mnew = tl.maximum(m, tl.max(s, axis=0))
            msafe = tl.where(mnew == float("-inf"), 0.0, mnew)
            alpha = tl.exp2(m - msafe)
            p = tl.exp2(s - msafe)
            vv = tl.load(vb + rn[:, None] * skn + rd[None, :], mask=mn[:, None], other=0.0)
            vv = vv.to(tl.float32) * vsc[:, None]
            l = l * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * vv, axis=0)
            m = mnew
        nch = tl.num_programs(1)
        base = ((pbh * T + t) * nch + c) * (D + 2)
        tl.store(ws_ptr + base, m)
        tl.store(ws_ptr + base + 1, l)
        tl.store(ws_ptr + base + 2 + rd, acc)

    @triton.jit
    def kvq_merge(ws_ptr, o_ptr, T, NCH, sob, soh, sot, HQ, D: tl.constexpr):
        pbh = tl.program_id(0)
        t = tl.program_id(1)
        b = pbh // HQ
        h = pbh % HQ
        rd = tl.arange(0, D)
        m = tl.full((), float("-inf"), tl.float32)
        l = tl.zeros((), tl.float32)
        acc = tl.zeros((D,), tl.float32)
        for c in range(0, NCH):           # the fixed order 0, 1, 2, ...
            base = ((pbh * T + t) * NCH + c) * (D + 2)
            mc = tl.load(ws_ptr + base)
            lc = tl.load(ws_ptr + base + 1)
            ac = tl.load(ws_ptr + base + 2 + rd)
            mnew = tl.maximum(m, mc)
            msafe = tl.where(mnew == float("-inf"), 0.0, mnew)
            wa = tl.exp2(m - msafe)
            wb = tl.exp2(mc - msafe)
            l = l * wa + lc * wb
            acc = acc * wa + ac * wb
            m = mnew
        out = acc / l
        tl.store(o_ptr + b * sob + h * soh + t * sot + rd, out.to(o_ptr.dtype.element_ty))

    @triton.jit
    def kvq_decode_grp(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, ws_ptr, tk_ptr, tv_ptr, tlen_ptr, plen_ptr,
                       T, N, HQ, HK, G, NQ, sqb, sqh, sqt, skb, skh, skn, ssb, ssh,
                       svb, svh, stb, sth, stn, qk_scale, GROUP: tl.constexpr, VGROUP: tl.constexpr,
                       HAS_TAIL: tl.constexpr, HAS_PLEN: tl.constexpr, CHUNK: tl.constexpr,
                       BN: tl.constexpr, RP: tl.constexpr, D: tl.constexpr):
        """All G * T query rows of one KV head and chunk in one program: the rows of a tile."""
        pbk = tl.program_id(0)            # batch * KV heads + KV head
        c = tl.program_id(1)              # chunk
        b = pbk // HK
        hk = pbk % HK
        rr = tl.arange(0, RP)             # row r = (query head g of this KV head, query row t)
        live = rr < G * T
        g = rr // T
        t = rr % T
        rd = tl.arange(0, D)
        qrow = q_ptr + b * sqb + (hk * G + g) * sqh + t * sqt
        q = tl.load(qrow[:, None] + rd[None, :], mask=live[:, None], other=0.0)   # q's own dtype
        dt = q.dtype
        m = tl.full((RP,), float("-inf"), tl.float32)
        l = tl.zeros((RP,), tl.float32)
        acc = tl.zeros((RP, D), tl.float32)
        lo = c * CHUNK
        if HAS_PLEN:
            hi = tl.minimum(lo + CHUNK, tl.minimum(N, tl.load(plen_ptr)))
        else:
            hi = tl.minimum(lo + CHUNK, N)
        if HAS_TAIL:
            if c == NQ:
                hi = lo
                tlen = tl.load(tlen_ptr)
                tkb = tk_ptr + b * stb + hk * sth
                tvb = tv_ptr + b * stb + hk * sth
                for n0 in range(0, tlen, BN):
                    rn = n0 + tl.arange(0, BN)
                    mn = rn < tlen
                    kk = tl.load(tkb + rn[:, None] * stn + rd[None, :], mask=mn[:, None], other=0.0)
                    s = tl.dot(q, tl.trans(kk.to(dt))) * qk_scale
                    s = tl.where(mn[None, :], s, float("-inf"))
                    mnew = tl.maximum(m, tl.max(s, axis=1))
                    msafe = tl.where(mnew == float("-inf"), 0.0, mnew)
                    alpha = tl.exp2(m - msafe)
                    p = tl.exp2(s - msafe[:, None])
                    vv = tl.load(tvb + rn[:, None] * stn + rd[None, :], mask=mn[:, None], other=0.0)
                    l = l * alpha + tl.sum(p, axis=1)
                    acc = acc * alpha[:, None] + tl.dot(p.to(dt), vv.to(dt))
                    m = mnew
        kb = k_ptr + b * skb + hk * skh
        vb = v_ptr + b * skb + hk * skh
        sb = b * ssb + hk * ssh
        vsb = b * svb + hk * svh
        for n0 in range(lo, hi, BN):
            rn = n0 + tl.arange(0, BN)
            mn = rn < hi
            # int8 and e4m3 codes are exact in float16 and bfloat16: the dot multiplies codes, the
            # group scales are applied to its float32 result (K per key, V per block: BN divides
            # VGROUP, so a block lies in one V group)
            kk = tl.load(kb + rn[:, None] * skn + rd[None, :], mask=mn[:, None], other=0.0)
            ksc = tl.load(ks_ptr + sb + rn // GROUP, mask=mn, other=0.0)
            s = tl.dot(q, tl.trans(kk.to(tl.float32).to(dt))) * (ksc * qk_scale)[None, :]
            s = tl.where(mn[None, :], s, float("-inf"))
            mnew = tl.maximum(m, tl.max(s, axis=1))
            msafe = tl.where(mnew == float("-inf"), 0.0, mnew)
            alpha = tl.exp2(m - msafe)
            p = tl.exp2(s - msafe[:, None])
            vv = tl.load(vb + rn[:, None] * skn + rd[None, :], mask=mn[:, None], other=0.0)
            vsc = tl.load(vs_ptr + vsb + n0 // VGROUP)
            l = l * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(dt), vv.to(tl.float32).to(dt)) * vsc
            m = mnew
        nch = tl.num_programs(1)
        h = hk * G + g
        base = (((b * HQ + h) * T + t) * nch + c) * (D + 2)
        tl.store(ws_ptr + base, m, mask=live)
        tl.store(ws_ptr + base + 1, l, mask=live)
        tl.store(ws_ptr + base[:, None] + 2 + rd[None, :], acc, mask=live[:, None])

    return kvq_decode, kvq_merge, kvq_decode_grp


GRP_MAX_ROWS = 64   # the grouped kernel takes up to this many query rows (heads x rows) per KV head


def grouped_block(G: int, T: int, vgroup: int, channel: bool) -> int:
    """Keys per step of the grouped query kernel for this call, or 0 for the one row kernel.

    The grouped kernel runs the G query heads of a KV head times the T query rows as the rows of
    one tile (tl.dot on the int8 or e4m3 codes in the query's dtype, float32 accumulation), so a
    KV block is read once for all of them. It needs a block that lies in one V group (the V scale
    is applied after the dot) and keys scaled per token, not per channel."""
    if channel or G * T < 2 or G * T > GRP_MAX_ROWS:
        return 0
    for bn in ((64, 32, 16) if G * T <= 32 else (32, 16)):   # 64 rows x 64 keys: over 99 KB of shared memory
        if vgroup % bn == 0:
            return bn
    return 0


_KERNELS = None
_VERIFIED = set()


def decode_attention(q: torch.Tensor, kc: QuantKV, vc: QuantKV, scale: Optional[float] = None,
                     deterministic: Optional[bool] = None, chunk: Optional[int] = None,
                     verify: bool = True, tail: Optional[tuple] = None,
                     prefix_len: Optional[torch.Tensor] = None) -> torch.Tensor:
    """softmax(q k^T * scale) v over a quantized cache, for up to MAX_ROWS query rows.

    q [B, Hq, T, D] float16 or bfloat16 (the output dtype); kc, vc from ``quantize`` with the same
    shape and format, Hq a multiple of their head count. ``chunk`` forces the keys per chunk (a
    multiple of BLOCK_N, or N); by default ``chunk_for`` chooses it.

    ``tail`` = (tk, tv, tlen): the newest tokens not quantized yet, tk and tv [B, Hkv, C, D] in
    float16 or bfloat16 of which the first ``tlen`` (an int32 CUDA tensor of one element, read on
    the device) are live. They follow the quantized keys: one more program per row reads them, and
    the merge takes their partial last, after the quantized chunks in order.

    ``prefix_len``: an int32 CUDA tensor of one element, the number of live quantized keys of a
    cache allocated to a fixed capacity N (read on the device, so the call's shapes do not change as
    the cache fills). Chunks are laid over the capacity; a chunk past the live keys stores max -inf
    and sum 0, and the merge adds exactly nothing for it, so with the same chunk size the result is
    bitwise that of a cache of exactly the live keys."""
    global _KERNELS
    for t, what in ((q, "q"), (kc.data, "the K cache"), (vc.data, "the V cache")):
        _need(t, what)
    if kc.shape != vc.shape or kc.fmt != vc.fmt:
        raise KVQuantError("[TS701] K and V caches must have the same shape and format")
    if kc.low:
        from . import low
        low.check_pair(kc, vc, KVQuantError)
    if vc.channel:
        raise KVQuantError("[TS701] per channel scales are for keys; V takes one per token group")
    B, Hq, T, D = q.shape
    Bk, Hk, N, Dk = kc.shape
    if Bk != B or Dk != D or Hq % Hk:
        raise KVQuantError(f"[TS701] q {tuple(q.shape)} does not match the cache {kc.shape}")
    if T > MAX_ROWS or D not in (16, 32, 64, 128, 256):
        raise KVQuantError(f"[TS701] decode only: up to {MAX_ROWS} query rows and a power of two "
                           f"head dim up to 256 (got T {T}, D {D})")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise KVQuantError("[TS701] q must be float16 or bfloat16")
    det = torch.are_deterministic_algorithms_enabled() if deterministic is None else deterministic
    gbn = 0 if kc.low else grouped_block(Hq // Hk, T, vc.group, kc.channel)
    ch = chunk_for(B, Hq, N, T, det, Hkv=Hk if gbn else None) if chunk is None else chunk
    if chunk is None and det and prefix_len is not None and N > DET_CHUNK:
        # the live length is on the device, so the chunks cannot follow it: a fixed capacity cache
        # chunks by DET_CHUNK always (bitwise an exact length cache from MIN_SERIAL_KEYS live keys)
        ch = DET_CHUNK
    if ch != N and (ch % BLOCK_N or ch <= 0):
        raise KVQuantError(f"[TS701] chunk must be a multiple of {BLOCK_N} or N, got {ch}")
    if det and ch not in (DET_CHUNK, N) or det and ch == N and N >= MIN_SERIAL_KEYS:
        raise KVQuantError(f"[TS701] a deterministic call uses chunks of {DET_CHUNK} from "
                           f"{MIN_SERIAL_KEYS} keys")
    if tail is not None:
        tk, tv, tlen = tail
        for t, what in ((tk, "the K tail"), (tv, "the V tail"), (tlen, "the tail length")):
            _need(t, what)
        if tk.shape != tv.shape or tk.shape[:2] != (B, Hk) or tk.shape[3] != D:
            raise KVQuantError(f"[TS701] the tail {tuple(tk.shape)} does not match the cache {kc.shape}")
        if tlen.numel() != 1 or tlen.dtype != torch.int32:
            raise KVQuantError("[TS701] the tail length must be one int32 element")
        if tk.stride(-1) != 1 or tv.stride() != tk.stride():
            tk, tv = tk.contiguous(), tv.contiguous()
    if prefix_len is not None:
        _need(prefix_len, "the prefix length")
        if prefix_len.numel() != 1 or prefix_len.dtype != torch.int32:
            raise KVQuantError("[TS701] the prefix length must be one int32 element")
    sc = D ** -0.5 if scale is None else scale
    nq = -(-N // ch) if N else 0
    nch = nq + (1 if tail is not None else 0)
    if nch == 0:
        raise KVQuantError("[TS701] an empty cache and no tail: nothing to attend to")
    if kc.low:
        from . import low
        out = low.decode(q, kc, vc, sc, ch, nch, tail, prefix_len)
        key = (tuple(q.shape), kc.shape, kc.fmt, kc.group, kc.rot, ch, str(q.dtype), sc,
               None if tail is None else tuple(tk.shape), prefix_len is None)
        if verify and key not in _VERIFIED:
            _check(q, kc, vc, sc, out, tail, prefix_len)
            _VERIFIED.add(key)
        return out
    if _KERNELS is None:
        _KERNELS = _kernels()
    kvq_decode, kvq_merge, kvq_decode_grp = _KERNELS
    ks, vs = kc.data, vc.data
    if ks.stride(-1) != 1 or vs.stride() != ks.stride():
        ks, vs = ks.contiguous(), vs.contiguous()
    qq = q if q.stride(-1) == 1 else q.contiguous()
    ws = torch.empty(B * Hq * T * nch * (D + 2), dtype=torch.float32, device=q.device)
    out = torch.empty(B, Hq, T, D, dtype=q.dtype, device=q.device)
    kscale, vscale = kc.scale.contiguous(), vc.scale.contiguous()
    plen = prefix_len if prefix_len is not None else ws
    if tail is None:
        tk = tv = qq
        tlen = ws
    with torch.cuda.device(q.device):
        if gbn:
            G = Hq // Hk
            kvq_decode_grp[(B * Hk, nch)](
                qq, ks, vs, kscale, vscale, ws, tk, tv, tlen, plen, T, N, Hq, Hk, G, nq,
                qq.stride(0), qq.stride(1), qq.stride(2), ks.stride(0), ks.stride(1), ks.stride(2),
                kscale.stride(0), kscale.stride(1), vscale.stride(0), vscale.stride(1),
                tk.stride(0), tk.stride(1), tk.stride(2),
                sc * 1.4426950408889634, GROUP=kc.group, VGROUP=vc.group,
                HAS_TAIL=tail is not None, HAS_PLEN=prefix_len is not None, CHUNK=max(ch, 1), BN=gbn,
                RP=max(16, 1 << (G * T - 1).bit_length()), D=D, num_warps=4)
        else:
            kvq_decode[(B * Hq, nch, T)](
                qq, ks, vs, kscale, vscale, ws, tk, tv, tlen, plen, T, N, Hq, Hq // Hk, kscale.shape[2], nq,
                qq.stride(0), qq.stride(1), qq.stride(2), ks.stride(0), ks.stride(1), ks.stride(2),
                kscale.stride(0), kscale.stride(1), vscale.stride(0), vscale.stride(1),
                tk.stride(0), tk.stride(1), tk.stride(2),
                sc * 1.4426950408889634, GROUP=kc.group, VGROUP=vc.group, K_CHANNEL=kc.channel,
                HAS_TAIL=tail is not None, HAS_PLEN=prefix_len is not None, CHUNK=max(ch, 1), BN=BLOCK_N,
                D=D, num_warps=4)
        kvq_merge[(B * Hq, T)](ws, out, T, nch, out.stride(0), out.stride(1), out.stride(2), Hq,
                               D=D, num_warps=4)
    key = (tuple(q.shape), kc.shape, kc.fmt, kc.group, kc.channel, vc.group, ch, str(q.dtype), sc,
           None if tail is None else tuple(tk.shape))
    if verify and key not in _VERIFIED:
        _check(q, kc, vc, sc, out, tail, prefix_len)
        _VERIFIED.add(key)
    return out


def reference(q: torch.Tensor, kc: QuantKV, vc: QuantKV, scale: Optional[float] = None,
              tail: Optional[tuple] = None, prefix_len: Optional[torch.Tensor] = None) -> torch.Tensor:
    """float64 attention over the dequantized cache (and the live tail after it): what the kernel
    computes, exactly rounded."""
    D = q.shape[-1]
    G = q.shape[1] // kc.shape[1]
    sc = D ** -0.5 if scale is None else scale
    kd, vd = kc.original(), vc.original()
    if prefix_len is not None:
        p = int(prefix_len.reshape(-1)[0])
        kd, vd = kd[:, :, :p], vd[:, :, :p]
    if tail is not None:
        tk, tv, tlen = tail
        n = int(tlen.reshape(-1)[0])
        kd = torch.cat([kd, tk[:, :, :n].double()], dim=2)
        vd = torch.cat([vd, tv[:, :, :n].double()], dim=2)
    kd, vd = kd.repeat_interleave(G, dim=1), vd.repeat_interleave(G, dim=1)
    w = torch.softmax((q.double() @ kd.transpose(-1, -2)) * sc, dim=-1)
    return w @ vd


def _check(q, kc, vc, sc, out, tail=None, prefix_len=None):
    from ..kernel.verify import KernelMismatchError
    ref = reference(q, kc, vc, sc, tail, prefix_len)
    err = ((out.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
    tol = (2e-3 if q.dtype == torch.float16 else 1.6e-2) * max(1.0, math.log2(max(2, kc.shape[2])) / 4)
    if not (err <= tol):
        raise KernelMismatchError(f"kvquant decode attention ({kc.fmt}"
                                  f"{' rotated' if kc.rot else ''}, K group {kc.group}"
                                  f"{' per channel' if kc.channel else ''}, V group {vc.group}): relative "
                                  f"error {err:.3e} against float64 over the dequantized cache, "
                                  f"tolerance {tol:.1e}")


# ----------------------------------------------------------------------------- torch op (model opt in)
# A model that opts into a quantized cache calls torch.ops.truestar.kvq_attention; Truestar's
# importer turns it into the IR op kvq_attention (float64 reference: ir/reference.py), and the
# compiled executor runs it through decode_attention above. Eager calls compute it in float64.

@torch.library.custom_op("truestar::kvq_attention", mutates_args=())
def kvq_attention(q: torch.Tensor, k_data: torch.Tensor, k_scale: torch.Tensor, v_data: torch.Tensor,
                  v_scale: torch.Tensor, k_tail: torch.Tensor, v_tail: torch.Tensor,
                  tail_len: torch.Tensor, fmt: str, group: int, vgroup: int, channel: bool,
                  scale: float, prefix_len: Optional[torch.Tensor] = None,
                  k_bscale: Optional[torch.Tensor] = None, v_bscale: Optional[torch.Tensor] = None,
                  rot: bool = False) -> torch.Tensor:
    """Attention of q [B, Hq, T, D] over a quantized cache (data int8, or fp8 as uint8 bits, and
    float32 scales) followed by the first ``tail_len`` tokens of a float tail. ``scale`` 0 means
    1 / sqrt(D). The 4 bit formats (``int4``, ``nvfp4``, kvquant/low.py) store data uint8
    [B, Hkv, N, D / 2] and add ``k_bscale`` and ``v_bscale``, the E4M3 block scales as uint8 bits
    [B, Hkv, N, D / 16]; ``rot`` says the cache holds x H (the tail stays in the natural basis)."""
    from ..ir.reference import IMPLS
    attrs = dict(fmt=fmt, group=group, vgroup=vgroup, channel=channel, scale=scale, rot=rot)
    args = [q.double(), k_data, k_scale.double(), v_data, v_scale.double(), k_tail.double(),
            v_tail.double(), tail_len, prefix_len, k_bscale, v_bscale]
    return IMPLS["kvq_attention"](attrs, args, None).to(q.dtype)


@kvq_attention.register_fake
def _(q, k_data, k_scale, v_data, v_scale, k_tail, v_tail, tail_len, fmt, group, vgroup, channel, scale,
      prefix_len=None, k_bscale=None, v_bscale=None, rot=False):
    return torch.empty_like(q)
