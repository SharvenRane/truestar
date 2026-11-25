"""Fused latent attention for the MLA decode (mode ``"fused"``, docs/MLA.md).

The absorbed decode of model.py runs, per layer: the latent query ``q_nope W_uk``, two concats
(the query parts and the whole cache ``[c, k_rope]``), an attention over the concatenated cache,
the output projection ``o W_uv^T``, and an ``index_copy`` of the layer's whole cache before it.
``mla_attention`` is that computation as one op, and on CUDA one Triton kernel per (sequence, head,
query row):

* ``q_lat = q_nope W_uk[h]`` is formed in registers (never written);
* scores ``(q_lat . c_p + q_rope . r_p) * scale`` over the cached positions ``p < pos[0]`` read
  the latent cache ``c`` [B, S, r] and the RoPE key cache ``r`` [B, S, dr] where they are (no concat,
  no copy), then over this step's own new tokens ``j <= t`` (``lat`` [B, T, r], ``kr`` [B, T, dr]);
* an online softmax in a fixed key order, float32 accumulation, then ``o = (sum_p w_p c_p) W_uv[h]^T``.

So the step does not write the new token into the cache before attending; it writes every
layer's new latents once at the end (model.py, ``Step`` in mode ``"fused"``). Key ``p`` of the cache
is visible to query row ``t`` when ``p < pos[0]``, new token ``j`` when ``j <= t``: for consecutive
positions ``pos = pos[0] .. pos[0] + T - 1`` (what ``generate`` passes) that is exactly the absorbed
decode's mask ``p <= pos[t]`` over the cache after its ``index_copy``, and ``prove_absorption``
checks the two in float64 before anything is compiled.

The float64 reference (``reference``) is the IR op's reference; the compiled executor runs the
kernel as a library step, region checked against that reference on its first call. On the CPU the
library step runs ``composite`` (the same math with torch ops in the working precision). The kernel
has no atomics and no split over keys: its bits depend on the inputs only.
"""

from __future__ import annotations

import torch

F64 = torch.float64
MAX_R = 1024       # latent rank the kernel keeps in registers (DeepSeek V3: 512)
MAX_D = 256        # dn, dr and dv


class MLAKernelError(ValueError):
    """TS1460: the fused latent attention was given shapes it does not run."""
    code = "TS1460"


def _check(q_nope, q_rope, cc, rc, lat, kr, pos, w_uk, w_uv):
    B, H, T, dn = q_nope.shape
    S, r = cc.shape[1], cc.shape[2]
    dr, dv = q_rope.shape[3], w_uv.shape[1]
    want = {"q_rope": (q_rope, (B, H, T, dr)), "cc": (cc, (B, S, r)), "rc": (rc, (B, S, dr)),
            "lat": (lat, (B, T, r)), "kr": (kr, (B, T, dr)), "pos": (pos, (T,)), "w_uk": (w_uk, (H, dn, r)),
            "w_uv": (w_uv, (H, dv, r))}
    for name, (x, shape) in want.items():
        if tuple(x.shape) != shape:
            raise MLAKernelError(f"[TS1460] mla_attention: {name} must be {list(shape)}, got {list(x.shape)}")
    return B, H, T, dn, S, r, dr, dv


def reference(q_nope, q_rope, cc, rc, lat, kr, pos, w_uk, w_uv, scale: float) -> torch.Tensor:
    """float64 attention of the absorbed decode over ``cc[:, :pos[0]]`` then the new tokens."""
    return _math(*(x.to(F64) if x.is_floating_point() else x for x in (q_nope, q_rope, cc, rc, lat, kr, pos,
                                                                       w_uk, w_uv)), scale)


def composite(q_nope, q_rope, cc, rc, lat, kr, pos, w_uk, w_uv, scale: float) -> torch.Tensor:
    """The same math with torch ops in the inputs' dtype (the CPU library step)."""
    return _math(q_nope, q_rope, cc, rc, lat, kr, pos, w_uk, w_uv, scale)


def _math(q_nope, q_rope, cc, rc, lat, kr, pos, w_uk, w_uv, scale):
    B, H, T, dn, S, r, dr, dv = _check(q_nope, q_rope, cc, rc, lat, kr, pos, w_uk, w_uv)
    n = max(0, min(int(pos.reshape(-1)[0]), S))
    k = torch.cat([cc[:, :n], lat], dim=1)                       # [B, n + T, r]
    kq = torch.cat([rc[:, :n], kr], dim=1)                       # [B, n + T, dr]
    q_lat = torch.einsum("bhtd,hdr->bhtr", q_nope, w_uk)
    s = (q_lat @ k[:, None].transpose(-1, -2) + q_rope @ kq[:, None].transpose(-1, -2)) * scale
    t = torch.arange(T, device=q_nope.device)
    j = torch.arange(n + T, device=q_nope.device)
    keep = (j[None, :] < n) | ((j[None, :] - n) <= t[:, None])   # [T, n + T]
    s = s.masked_fill(~keep, float("-inf"))
    w = torch.softmax(s, dim=-1)
    o = w @ k[:, None]                                           # [B, H, T, r]
    return torch.einsum("bhtr,hvr->bhtv", o, w_uv)


# ----------------------------------------------------------------------------- kernel

_K = None


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mla_decode(qn_ptr, qr_ptr, c_ptr, r_ptr, l_ptr, k_ptr, pos_ptr, wk_ptr, wv_ptr, o_ptr,
                   S, T, H, DN, DR, R, DV,
                   sqn_b, sqn_h, sqn_t, sqr_b, sqr_h, sqr_t, sc_b, sc_s, sr_b, sr_s,
                   sl_b, sl_t, sk_b, sk_t, swk_h, swk_d, swv_h, swv_v, so_b, so_h, so_t,
                   qk_scale, BR: tl.constexpr, BDR: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr,
                   BN: tl.constexpr):
        pid = tl.program_id(0)
        t = tl.program_id(1)
        b = pid // H
        h = pid % H
        rr = tl.arange(0, BR)
        mr = rr < R
        # latent query: q_nope W_uk[h], kept in registers
        qlat = tl.zeros((BR,), tl.float32)
        for d0 in range(0, DN, BD):
            rd = d0 + tl.arange(0, BD)
            md = rd < DN
            qv = tl.load(qn_ptr + b * sqn_b + h * sqn_h + t * sqn_t + rd, mask=md, other=0.0).to(tl.float32)
            w = tl.load(wk_ptr + h * swk_h + rd[:, None] * swk_d + rr[None, :], mask=md[:, None] & mr[None, :],
                        other=0.0).to(tl.float32)
            qlat += tl.sum(qv[:, None] * w, axis=0)
        qlat = qlat * qk_scale
        rq = tl.arange(0, BDR)
        mq = rq < DR
        qrope = tl.load(qr_ptr + b * sqr_b + h * sqr_h + t * sqr_t + rq, mask=mq, other=0.0).to(tl.float32) * qk_scale
        n = tl.minimum(tl.maximum(tl.load(pos_ptr).to(tl.int32), 0), S)
        m = tl.full((), float("-inf"), tl.float32)
        l = tl.zeros((), tl.float32)
        acc = tl.zeros((BR,), tl.float32)
        for n0 in range(0, n, BN):                 # the cache, positions < pos[0], in order
            rn = n0 + tl.arange(0, BN)
            mn = rn < n
            kc = tl.load(c_ptr + b * sc_b + rn[:, None] * sc_s + rr[None, :], mask=mn[:, None] & mr[None, :],
                         other=0.0).to(tl.float32)
            kq = tl.load(r_ptr + b * sr_b + rn[:, None] * sr_s + rq[None, :], mask=mn[:, None] & mq[None, :],
                         other=0.0).to(tl.float32)
            s = tl.sum(kc * qlat[None, :], axis=1) + tl.sum(kq * qrope[None, :], axis=1)
            s = tl.where(mn, s, float("-inf"))
            mnew = tl.maximum(m, tl.max(s, axis=0))
            msafe = tl.where(mnew == float("-inf"), 0.0, mnew)
            alpha = tl.exp2(m - msafe)
            p = tl.exp2(s - msafe)
            l = l * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * kc, axis=0)
            m = mnew
        for j0 in range(0, t + 1, BN):             # this step's new tokens j <= t, in order
            rn = j0 + tl.arange(0, BN)
            mn = rn <= t
            kc = tl.load(l_ptr + b * sl_b + rn[:, None] * sl_t + rr[None, :], mask=mn[:, None] & mr[None, :],
                         other=0.0).to(tl.float32)
            kq = tl.load(k_ptr + b * sk_b + rn[:, None] * sk_t + rq[None, :], mask=mn[:, None] & mq[None, :],
                         other=0.0).to(tl.float32)
            s = tl.sum(kc * qlat[None, :], axis=1) + tl.sum(kq * qrope[None, :], axis=1)
            s = tl.where(mn, s, float("-inf"))
            mnew = tl.maximum(m, tl.max(s, axis=0))
            msafe = tl.where(mnew == float("-inf"), 0.0, mnew)
            alpha = tl.exp2(m - msafe)
            p = tl.exp2(s - msafe)
            l = l * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * kc, axis=0)
            m = mnew
        ol = acc / l
        for v0 in range(0, DV, BV):                # o = o_lat W_uv[h]^T
            rv = v0 + tl.arange(0, BV)
            mv = rv < DV
            w = tl.load(wv_ptr + h * swv_h + rv[:, None] * swv_v + rr[None, :], mask=mv[:, None] & mr[None, :],
                        other=0.0).to(tl.float32)
            o = tl.sum(w * ol[None, :], axis=1)
            tl.store(o_ptr + b * so_b + h * so_h + t * so_t + rv, o, mask=mv)

    return mla_decode


def _pow2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def latent_attention(q_nope, q_rope, cc, rc, lat, kr, pos, w_uk, w_uv, scale: float) -> torch.Tensor:
    """The fused kernel on CUDA tensors; float32 math, the output in q_nope's dtype."""
    global _K
    B, H, T, dn, S, r, dr, dv = _check(q_nope, q_rope, cc, rc, lat, kr, pos, w_uk, w_uv)
    if r > MAX_R or max(dn, dr, dv) > MAX_D:
        raise MLAKernelError(f"[TS1460] mla_attention runs latent ranks up to {MAX_R} and head dims up to {MAX_D} "
                             f"(got r {r}, dn {dn}, dr {dr}, dv {dv}); use decode='absorb'")
    for x, name in ((q_nope, "q_nope"), (cc, "cc"), (pos, "pos")):
        if x.device.type != "cuda":
            raise MLAKernelError(f"[TS1460] latent_attention runs on CUDA tensors ({name} is on {x.device})")

    def last1(x):
        return x if x.stride(-1) == 1 else x.contiguous()
    q_nope, q_rope, cc, rc, lat, kr, w_uk, w_uv = (last1(x) for x in (q_nope, q_rope, cc, rc, lat, kr, w_uk, w_uv))
    if _K is None:
        _K = _kernel()
    out = torch.empty(B, H, T, dv, dtype=q_nope.dtype, device=q_nope.device)
    BR = _pow2(r)
    BN = 32 if BR <= 128 else 16
    with torch.cuda.device(q_nope.device):
        _K[(B * H, T)](q_nope, q_rope, cc, rc, lat, kr, pos, w_uk, w_uv, out,
                       S, T, H, dn, dr, r, dv,
                       q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
                       q_rope.stride(0), q_rope.stride(1), q_rope.stride(2),
                       cc.stride(0), cc.stride(1), rc.stride(0), rc.stride(1),
                       lat.stride(0), lat.stride(1), kr.stride(0), kr.stride(1),
                       w_uk.stride(0), w_uk.stride(1), w_uv.stride(0), w_uv.stride(1),
                       out.stride(0), out.stride(1), out.stride(2),
                       float(scale) * 1.4426950408889634, BR=BR, BDR=max(16, _pow2(dr)), BD=16, BV=16, BN=BN,
                       num_warps=4 if BR <= 128 else 8)
    return out


# ----------------------------------------------------------------------------- torch op

@torch.library.custom_op("truestar::mla_attention", mutates_args=())
def mla_attention(q_nope: torch.Tensor, q_rope: torch.Tensor, cc: torch.Tensor, rc: torch.Tensor,
                  lat: torch.Tensor, kr: torch.Tensor, pos: torch.Tensor, w_uk: torch.Tensor,
                  w_uv: torch.Tensor, scale: float) -> torch.Tensor:
    """Eager calls compute the float64 reference and return it in q_nope's dtype; the importer
    turns the op into the IR op ``mla_attention``."""
    return reference(q_nope, q_rope, cc, rc, lat, kr, pos, w_uk, w_uv, scale).to(q_nope.dtype)


@mla_attention.register_fake
def _(q_nope, q_rope, cc, rc, lat, kr, pos, w_uk, w_uv, scale):
    return q_nope.new_empty((q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], w_uv.shape[1]))


__all__ = ["mla_attention", "latent_attention", "reference", "composite", "MLAKernelError", "MAX_R", "MAX_D"]
