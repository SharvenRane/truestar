"""KV cache below 8 bits: 4 bit integer and NVFP4 keys and values, optionally Hadamard rotated
along the head dim, with the dequantization inside a split KV decode kernel.

Layout of one K or V cache [B, Hkv, N, D] (D a multiple of 32):

* ``data`` uint8 [B, Hkv, N, D / 2]: byte j of a token holds channel j in its low nibble and
  channel j + D / 2 in its high nibble, so the kernel reads the two halves of the head dim with
  no interleave.
* ``bscale`` float8_e4m3fn [B, Hkv, N, D / 16]: one E4M3 scale per token and block of 16
  channels (the NVFP4 block).
* ``scale`` float32 [B, Hkv, ceil(N / group)]: one scale per head and group of tokens, so the
  E4M3 block scales of a group use E4M3's range (the NVFP4 tensor scale, kept per token group so a
  growing cache never rescales what it already stored, and so a paged and a dense cache agree).

Element codes: ``nvfp4`` E2M1 (values 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives), block
scale amax / 6; ``int4`` integers -7 to 7 (two's complement nibbles), block scale amax / 7.
Both round to nearest, ties to even, against the stored scales (all computed in float64 with
the format tables of truestar.quant.formats).

Rotation (``rot=True``): the cache stores x H with H the +1 / -1 Sylvester matrix of order D
(H H = D I), computed in float64 from the float16 or bfloat16 cache, which is exact unless a
row's exponents spread over more than 2^38. Decode rotates the query the same way (q H, float32,
by the fast Walsh kernel), scales the scores by 1 / D and turns the output back (o H / D, an
exact power of two). In exact arithmetic the attention is unchanged: (q H)(k H)^T / D = q k^T and
sum_j w_j v_j H H / D = sum_j w_j v_j. ``QuantKV.original()`` returns the values in the
unrotated basis, and ``error_bound`` is computed from those, so the proven bound needs no
transfer argument: it is the same theorem on the values the cache really holds.
"""

from __future__ import annotations


import torch

from ..quant import formats as QF
from ..rotate.hadamard import walsh

F64 = torch.float64
LOW_FORMATS = ("int4", "nvfp4")
QM = {"int4": 7.0, "nvfp4": 6.0}      # largest element magnitude
BLOCK = 16                            # channels per E4M3 block scale


def bytes_per_token(fmt: str, D: int, group: int, heads: int = 1) -> float:
    """K and V bytes per token: D / 2 packed bytes, D / 16 E4M3 block scales, one float32 per
    ``group`` tokens, each for K and for V."""
    return heads * 2 * (D / 2 + D / BLOCK + 4.0 / group)


def check_shape(x: torch.Tensor, fmt: str, group: int, rot: bool, err) -> None:
    if x.dim() != 4:
        raise err(f"[TS701] the cache must be [B, Hkv, N, D], got {tuple(x.shape)}")
    D = x.shape[3]
    if D % 32:
        raise err(f"[TS701] {fmt} packs two channels a byte and 16 a block scale: the head dim "
                  f"must be a multiple of 32, got {D}")
    if rot and (D & (D - 1) or D > 256):
        raise err(f"[TS701] a rotated cache needs a power of two head dim up to 256, got {D}")
    if group not in (16, 32, 64, 128):
        raise err(f"[TS701] group must be 16, 32, 64 or 128 tokens, got {group}")


def codes(x: torch.Tensor, fmt: str, group: int, rot: bool):
    """(data, bscale, scale) for a cache x [B, H, N, D] on any device (float64 arithmetic)."""
    B, H, N, D = x.shape
    x64 = x.to(F64).contiguous()
    if rot:
        x64 = walsh(x64, normalize=False).contiguous()
    ng = -(-N // group)
    pad = ng * group - N
    if pad:
        x64 = torch.cat([x64, x64.new_zeros(B, H, pad, D)], dim=2)
    amax = x64.view(B, H, ng, group * D).abs().amax(dim=3)
    gs = (amax / (QF.E4M3.max_finite * QM[fmt])).to(torch.float32)
    gs = torch.where(gs > 0, gs, torch.ones_like(gs))
    gs64 = gs.to(F64).repeat_interleave(group, dim=2)[:, :, :, None]          # [B, H, Np, 1]
    bam = x64.view(B, H, ng * group, D // BLOCK, BLOCK).abs().amax(dim=4)     # [B, H, Np, D/16]
    bcode = QF.E4M3.encode(bam / QM[fmt] / gs64)
    bsv = QF.E4M3.decode(bcode) * gs64
    sv = bsv.repeat_interleave(BLOCK, dim=3)
    y = torch.where(sv > 0, x64 / torch.where(sv > 0, sv, torch.ones_like(sv)), torch.zeros_like(x64))
    if fmt == "nvfp4":
        c = QF.E2M1.encode(y)
    else:
        c = QF.round_half_even_int(y, -7, 7) & 15
    c = c[:, :, :N]
    data = (c[..., : D // 2] | (c[..., D // 2:] << 4)).to(torch.uint8).contiguous()
    bscale = bcode[:, :, :N].to(torch.uint8).view(torch.float8_e4m3fn).contiguous()
    return data, bscale, gs.contiguous()


def element_values(data: torch.Tensor, fmt: str) -> torch.Tensor:
    """Decoded elements (before scaling) [..., D] in float64, channels in natural order."""
    b = data.long()
    c = torch.cat([b & 15, b >> 4], dim=-1)
    if fmt == "nvfp4":
        return QF.E2M1.decode(c)
    return torch.where(c >= 8, c - 16, c).to(F64)


def dequantize(data, bscale, scale, fmt, group, dtype=F64) -> torch.Tensor:
    """Exactly the values the kernel reads (in the stored basis): element x block scale x group
    scale, in float64 (then ``dtype``)."""
    N = data.shape[2]
    e = element_values(data, fmt)
    bs = QF.E4M3.decode(bscale.view(torch.uint8)).repeat_interleave(BLOCK, dim=3)
    gs = scale.to(F64).repeat_interleave(group, dim=2)[:, :, :N, None]
    return (e * bs * gs).to(dtype)


def unrotate(x64: torch.Tensor) -> torch.Tensor:
    """x H / D in float64: the inverse of the cache rotation."""
    return walsh(x64, normalize=False) / x64.shape[-1]


def fake_quantize(x: torch.Tensor, fmt: str, group: int = 32, rot: bool = False) -> torch.Tensor:
    """The values a 4 bit cache of x would hold, in x's basis, as float64, on any device: the same
    codes ``quantize`` stores (used by model studies on the CPU)."""
    from . import KVQuantError
    check_shape(x, fmt, group, rot, KVQuantError)
    d, b, s = codes(x, fmt, group, rot)
    v = dequantize(d, b, s, fmt, group)
    return unrotate(v) if rot else v


# ----------------------------------------------------------------------------- kernel

def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def dec(c, FP4: tl.constexpr):
        if FP4:            # E2M1: sign, 2 exponent bits, 1 mantissa bit
            m = c & 7
            e = m >> 1
            f = (m & 1).to(tl.float32)
            p = tl.where(e == 1, 1.0, tl.where(e == 2, 2.0, 4.0))
            mag = tl.where(e == 0, f * 0.5, (1.0 + f * 0.5) * p)
            v = tl.where((c >> 3) == 1, -mag, mag)
        else:              # two's complement nibble
            v = ((c ^ 8) - 8).to(tl.float32)
        return v

    @triton.jit
    def kvq4_decode(q_ptr, k_ptr, v_ptr, kb_ptr, vb_ptr, ks_ptr, vs_ptr, ws_ptr,
                    tk_ptr, tv_ptr, tlen_ptr, plen_ptr, NQ, stb, sth, stn,
                    T, N, HQ, G, sqb, sqh, sqt, skb, skh, skn, sbb, sbh, sbn, ssb, ssh,
                    qk_scale, GROUP: tl.constexpr, FP4: tl.constexpr, CHUNK: tl.constexpr,
                    BN: tl.constexpr, DH: tl.constexpr, HAS_TAIL: tl.constexpr, HAS_PLEN: tl.constexpr):
        pbh = tl.program_id(0)
        c = tl.program_id(1)
        t = tl.program_id(2)
        b = pbh // HQ
        h = pbh % HQ
        hk = h // G
        rh = tl.arange(0, DH)
        qb = q_ptr + b * sqb + h * sqh + t * sqt
        qlo = tl.load(qb + rh).to(tl.float32) * qk_scale
        qhi = tl.load(qb + DH + rh).to(tl.float32) * qk_scale
        m = tl.full((), float("-inf"), tl.float32)
        l = tl.zeros((), tl.float32)
        alo = tl.zeros((DH,), tl.float32)
        ahi = tl.zeros((DH,), tl.float32)
        lo = c * CHUNK
        if HAS_PLEN:          # a fixed capacity prefix: only its first plen keys are live
            hi = tl.minimum(lo + CHUNK, tl.minimum(N, tl.load(plen_ptr)))
        else:
            hi = tl.minimum(lo + CHUNK, N)
        if HAS_TAIL:          # the last program of a row reads the float tail (in the cache's basis)
            if c == NQ:
                hi = lo
                tlen = tl.load(tlen_ptr)
                tkb = tk_ptr + b * stb + hk * sth
                tvb = tv_ptr + b * stb + hk * sth
                for n0 in range(0, tlen, BN):
                    rn = n0 + tl.arange(0, BN)
                    mn = rn < tlen
                    m2 = mn[:, None]
                    t_lo = tl.load(tkb + rn[:, None] * stn + rh[None, :], mask=m2, other=0.0).to(tl.float32)
                    t_hi = tl.load(tkb + rn[:, None] * stn + DH + rh[None, :], mask=m2, other=0.0).to(tl.float32)
                    s = tl.sum(t_lo * qlo[None, :], axis=1) + tl.sum(t_hi * qhi[None, :], axis=1)
                    s = tl.where(mn, s, float("-inf"))
                    mnew = tl.maximum(m, tl.max(s, axis=0))
                    msafe = tl.where(mnew == float("-inf"), 0.0, mnew)
                    alpha = tl.exp2(m - msafe)
                    p = tl.exp2(s - msafe)
                    u_lo = tl.load(tvb + rn[:, None] * stn + rh[None, :], mask=m2, other=0.0).to(tl.float32)
                    u_hi = tl.load(tvb + rn[:, None] * stn + DH + rh[None, :], mask=m2, other=0.0).to(tl.float32)
                    l = l * alpha + tl.sum(p, axis=0)
                    alo = alo * alpha + tl.sum(p[:, None] * u_lo, axis=0)
                    ahi = ahi * alpha + tl.sum(p[:, None] * u_hi, axis=0)
                    m = mnew
        kb = k_ptr + b * skb + hk * skh
        vb = v_ptr + b * skb + hk * skh
        kbs = kb_ptr + b * sbb + hk * sbh
        vbs = vb_ptr + b * sbb + hk * sbh
        sb = b * ssb + hk * ssh
        blo = rh // 16
        bhi = DH // 16 + rh // 16
        for n0 in range(lo, hi, BN):
            rn = n0 + tl.arange(0, BN)
            mn = rn < hi
            m2 = mn[:, None]
            gk = tl.load(ks_ptr + sb + rn // GROUP, mask=mn, other=0.0)
            gv = tl.load(vs_ptr + sb + rn // GROUP, mask=mn, other=0.0)
            kk = tl.load(kb + rn[:, None] * skn + rh[None, :], mask=m2, other=0).to(tl.int32)
            k_lo = dec(kk & 15, FP4) * tl.load(kbs + rn[:, None] * sbn + blo[None, :], mask=m2,
                                               other=0.0).to(tl.float32) * gk[:, None]
            k_hi = dec(kk >> 4, FP4) * tl.load(kbs + rn[:, None] * sbn + bhi[None, :], mask=m2,
                                               other=0.0).to(tl.float32) * gk[:, None]
            s = tl.sum(k_lo * qlo[None, :], axis=1) + tl.sum(k_hi * qhi[None, :], axis=1)
            s = tl.where(mn, s, float("-inf"))
            mnew = tl.maximum(m, tl.max(s, axis=0))
            msafe = tl.where(mnew == float("-inf"), 0.0, mnew)
            alpha = tl.exp2(m - msafe)
            p = tl.exp2(s - msafe)
            vv = tl.load(vb + rn[:, None] * skn + rh[None, :], mask=m2, other=0).to(tl.int32)
            v_lo = dec(vv & 15, FP4) * tl.load(vbs + rn[:, None] * sbn + blo[None, :], mask=m2,
                                               other=0.0).to(tl.float32) * gv[:, None]
            v_hi = dec(vv >> 4, FP4) * tl.load(vbs + rn[:, None] * sbn + bhi[None, :], mask=m2,
                                               other=0.0).to(tl.float32) * gv[:, None]
            l = l * alpha + tl.sum(p, axis=0)
            alo = alo * alpha + tl.sum(p[:, None] * v_lo, axis=0)
            ahi = ahi * alpha + tl.sum(p[:, None] * v_hi, axis=0)
            m = mnew
        nch = tl.num_programs(1)
        base = ((pbh * T + t) * nch + c) * (2 * DH + 2)
        tl.store(ws_ptr + base, m)
        tl.store(ws_ptr + base + 1, l)
        tl.store(ws_ptr + base + 2 + rh, alo)
        tl.store(ws_ptr + base + 2 + DH + rh, ahi)

    return kvq4_decode


_K4 = None


def decode(q, kc, vc, sc, ch, nch, tail=None, prefix_len=None):
    """Launch the 4 bit decode and the shared merge; returns the output in q's dtype.

    ``tail`` = (tk, tv, tlen): float tokens after the quantized ones, in the natural basis; for a
    rotated cache they are rotated here (x H in float32, one launch each) so that every key the
    kernel reads is in the cache's basis. ``prefix_len``: live quantized keys of a fixed capacity
    cache (an int32 CUDA tensor of one element)."""
    global _K4
    from . import _kernels
    import truestar.kvquant as KQ
    from ..rotate.walsh_kernel import fwht
    if _K4 is None:
        _K4 = _kernel()
    if KQ._KERNELS is None:
        KQ._KERNELS = _kernels()
    merge = KQ._KERNELS[1]
    B, Hq, T, D = q.shape
    Hk, N = kc.shape[1], kc.shape[2]
    if kc.rot:
        qq = fwht(q, out_dtype=torch.float32)
        qk_scale = sc * 1.4426950408889634 / D
        out = torch.empty(B, Hq, T, D, dtype=torch.float32, device=q.device)
    else:
        qq = q if q.stride(-1) == 1 else q.contiguous()
        qk_scale = sc * 1.4426950408889634
        out = torch.empty(B, Hq, T, D, dtype=q.dtype, device=q.device)
    kd, vd = kc.data.contiguous(), vc.data.contiguous()
    kb, vb = kc.bscale.contiguous(), vc.bscale.contiguous()
    ks, vs = kc.scale.contiguous(), vc.scale.contiguous()
    ws = torch.empty(B * Hq * T * nch * (D + 2), dtype=torch.float32, device=q.device)
    nq = nch - (1 if tail is not None else 0)
    if tail is not None:
        tk, tv, tlen = tail
        if kc.rot:
            tk, tv = fwht(tk, out_dtype=torch.float32), fwht(tv, out_dtype=torch.float32)
        elif tk.stride(-1) != 1 or tv.stride() != tk.stride():
            tk, tv = tk.contiguous(), tv.contiguous()
    else:
        tk = tv = qq
        tlen = ws
    plen = ws if prefix_len is None else prefix_len
    with torch.cuda.device(q.device):
        _K4[(B * Hq, nch, T)](
            qq, kd, vd, kb, vb, ks, vs, ws, tk, tv, tlen, plen, nq, tk.stride(0), tk.stride(1), tk.stride(2),
            T, N, Hq, Hq // Hk,
            qq.stride(0), qq.stride(1), qq.stride(2), kd.stride(0), kd.stride(1), kd.stride(2),
            kb.stride(0), kb.stride(1), kb.stride(2), ks.stride(0), ks.stride(1), qk_scale,
            GROUP=kc.group, FP4=kc.fmt == "nvfp4", CHUNK=max(ch, 1), BN=64, DH=D // 2,
            HAS_TAIL=tail is not None, HAS_PLEN=prefix_len is not None, num_warps=4)
        merge[(B * Hq, T)](ws, out, T, nch, out.stride(0), out.stride(1), out.stride(2), Hq,
                           D=D, num_warps=4)
    if kc.rot:
        out = fwht(out, scale=1.0 / D, out_dtype=q.dtype)
    return out


def check_pair(kc, vc, err) -> None:
    if vc.fmt != kc.fmt or vc.rot != kc.rot or vc.group != kc.group:
        raise err("[TS701] a 4 bit K and V cache must share format, group and rotation")
    if kc.bscale is None or vc.bscale is None:
        raise err("[TS701] a 4 bit cache needs its E4M3 block scales")
