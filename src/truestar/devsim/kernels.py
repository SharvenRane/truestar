"""Float64 kernels that reproduce a profile's arithmetic: rounding, accumulation, formulations.

Every function takes float64 tensors whose values are already representable in the dtype being
simulated, and returns float64 tensors. ``rnd`` is the only place a value is rounded, so the
rounding rules of a profile (round to nearest even, flush to zero, saturation, NaN to +inf) live in
one function.

Accumulation models (``Profile.acc``):

* wide, no tile: the sum is exact here, then rounded once (a float32 class accumulator rounds
  far below the storage format, so float64 stands for it);
* wide with ``tile``: groups of ``tile`` products are summed and rounded to storage first;
* storage: blocks of ``acc_block`` products are summed (and rounded when the block is wider than
  one element), then added to a running sum that is rounded after every block, in order. With
  ``acc_block`` 1 each step is a fused multiply add rounded once.

Memory: products are materialised in chunks of at most ``MAX_TEMP`` float64 elements.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from .profiles import Profile

MAX_TEMP = 1 << 22  # float64 elements per temporary (32 MB)
F64 = torch.float64


def rnd(x: torch.Tensor, dtype: Optional[torch.dtype], p: Profile) -> torch.Tensor:
    """Round float64 ``x`` to ``dtype`` under the profile's rules (returns float64)."""
    if dtype is None or dtype == F64:
        y = x
    else:
        y = x.to(dtype).to(F64)
        fi = torch.finfo(dtype)
        if p.saturate:
            y = torch.where(torch.isinf(y) & torch.isfinite(x),
                            torch.copysign(torch.full_like(y, fi.max), x), y)
        if p.ftz:
            y = torch.where((y != 0) & (y.abs() < fi.tiny), y * 0.0, y)
    if p.nan_to_inf:
        y = torch.where(torch.isnan(y), torch.full_like(y, math.inf), y)
    return y


def stepwise(p: Profile) -> bool:
    """True when composite ops (norms, softmax, attention) round between their steps on this
    profile. Otherwise they run as one fused kernel with a wide internal type and round once."""
    return p.acc == "storage" or p.tile > 0 or p.decompose or p.saturate


# ----------------------------------------------------------------------------- sums


def _pad_last(x: torch.Tensor, multiple: int) -> torch.Tensor:
    k = x.shape[-1]
    extra = (-k) % multiple
    return F.pad(x, (0, extra)) if extra else x


def _sum_rows(x: torch.Tensor, p: Profile, sdt: Optional[torch.dtype]) -> torch.Tensor:
    """Sum the last dim of a [R, K] float64 tensor under the profile's accumulation model."""
    r, k = x.shape
    if k == 0:
        return x.new_zeros(r)
    if p.acc == "wide":
        if not p.tile:
            return x.sum(-1)
        g = _pad_last(x, p.tile).reshape(r, -1, p.tile).sum(-1)
        return rnd(g, sdt, p).sum(-1)
    b = p.acc_block
    g = _pad_last(x, b).reshape(r, -1, b)
    g = g.sum(-1) if b > 1 else g[..., 0]
    if b > 1:
        g = rnd(g, sdt, p)
    acc = g[:, 0]
    for i in range(1, g.shape[1]):
        acc = rnd(acc + g[:, i], sdt, p)
    return acc


def acc_sum(x: torch.Tensor, dims: Optional[Sequence[int]], p: Profile, sdt: Optional[torch.dtype],
            keepdim: bool = False) -> torch.Tensor:
    """Sum over ``dims`` (None or empty: all dims) with the profile's accumulator."""
    nd = x.dim()
    if nd == 0:
        return x.clone()
    dims = list(range(nd)) if not dims else sorted({d % nd for d in dims})
    if p.acc == "wide" and not p.tile:
        return x.sum(dims, keepdim=keepdim)
    keep = [d for d in range(nd) if d not in dims]
    xt = x.permute(keep + dims).reshape(math.prod(x.shape[d] for d in keep) if keep else 1, -1)
    s = _sum_rows(xt, p, sdt).reshape([x.shape[d] for d in keep])
    if keepdim:
        for d in dims:
            s = s.unsqueeze(d)
    return s


# ----------------------------------------------------------------------------- contractions


def contract(a: torch.Tensor, b: torch.Tensor, p: Profile, sdt: Optional[torch.dtype]) -> torch.Tensor:
    """Batched matmul [B, M, K] x [B, K, N] -> [B, M, N] under the accumulation model. The result
    is not rounded to storage here (callers add the bias first or round it)."""
    if p.acc == "wide" and not p.tile:
        return torch.matmul(a, b)
    bsz, m, k = a.shape
    n = b.shape[-1]
    if k == 0:
        return a.new_zeros(bsz, m, n)
    if p.acc == "wide":
        t = p.tile
        a4 = _pad_last(a, t).reshape(bsz, m, -1, t)
        b4 = _pad_last(b.transpose(1, 2), t).reshape(bsz, n, -1, t)
        groups = a4.shape[2]
        step = max(1, MAX_TEMP // max(1, bsz * m * n))
        total = a.new_zeros(bsz, m, n)
        for g0 in range(0, groups, step):
            part = torch.einsum("bmgt,bngt->bmgn", a4[:, :, g0:g0 + step], b4[:, :, g0:g0 + step])
            total = total + rnd(part, sdt, p).sum(2)
        return total
    blk = p.acc_block
    acc = None
    for k0 in range(0, k, blk):
        part = torch.matmul(a[:, :, k0:k0 + blk], b[:, k0:k0 + blk, :])
        if blk > 1:
            part = rnd(part, sdt, p)
        acc = rnd(part if acc is None else acc + part, sdt, p)
    return acc


def matmul_like(name: str, a: dict, p: Profile, sdt, notes: set) -> torch.Tensor:
    """mm, bmm, addmm, baddbmm, mv, dot with the profile's accumulator; bias added after the
    accumulator is rounded (ANE: 'added in the fp16 datapath')."""
    if name in ("mm", "addmm"):
        x, w = a["mat1" if name == "addmm" else "self"], a["mat2"]
        out = contract(x.unsqueeze(0), w.unsqueeze(0), p, sdt)[0]
    elif name in ("bmm", "baddbmm"):
        x, w = a["batch1" if name == "baddbmm" else "self"], a["batch2" if name == "baddbmm" else "mat2"]
        out = contract(x, w, p, sdt)
    elif name == "mv":
        out = contract(a["self"].unsqueeze(0), a["vec"].reshape(1, -1, 1), p, sdt)[0, :, 0]
    elif name == "dot":
        out = contract(a["self"].reshape(1, 1, -1), a["tensor"].reshape(1, -1, 1), p, sdt).reshape(())
    else:
        raise KeyError(name)
    out = rnd(out, sdt, p)
    if name in ("addmm", "baddbmm"):
        alpha, beta = float(a.get("alpha", 1)), float(a.get("beta", 1))
        if alpha != 1:
            out = rnd(out * alpha, sdt, p)
        bias = a["self"]
        out = rnd(out + (bias * beta if beta != 1 else bias), sdt, p) if beta != 0 else out
    return out


def convolution(a: dict, p: Profile, sdt, notes: set) -> torch.Tensor:
    x, w, bias = a["input"], a["weight"], a["bias"]
    stride, padding, dilation = list(a["stride"]), list(a["padding"]), list(a["dilation"])
    groups, transposed = int(a["groups"]), bool(a["transposed"])
    plain = p.acc == "wide" and not p.tile
    if plain or transposed or x.dim() not in (3, 4):
        if not plain:
            notes.add("convolution (transposed or 3d): accumulated exactly, the profile's "
                      "accumulator is not modelled for it")
        out = torch.ops.aten.convolution.default(x, w, None, stride, padding, dilation, transposed,
                                                 list(a["output_padding"]), groups)
    else:
        one_d = x.dim() == 3
        if one_d:
            x, w = x.unsqueeze(-2), w.unsqueeze(-2)
            stride, padding, dilation = [1] + stride, [0] + padding, [1] + dilation
        n, c, h, wd = x.shape
        o, cg, kh, kw = w.shape
        og = o // groups
        ho = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
        wo = (wd + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
        outs = []
        for g in range(groups):
            cols = F.unfold(x[:, g * cg:(g + 1) * cg], (kh, kw), dilation=tuple(dilation),
                            padding=tuple(padding), stride=tuple(stride))
            wg = w[g * og:(g + 1) * og].reshape(og, -1)
            outs.append(contract(wg.unsqueeze(0).expand(n, -1, -1), cols, p, sdt))
        out = torch.cat(outs, 1).reshape(n, o, ho, wo)
        if one_d:
            out = out.squeeze(-2)
    out = rnd(out, sdt, p)
    if bias is not None:
        out = rnd(out + bias.reshape([1, -1] + [1] * (out.dim() - 2)), sdt, p)
    return out


# ----------------------------------------------------------------------------- reductions


def _dims(a: dict, key: str = "dim") -> Optional[List[int]]:
    d = a.get(key)
    if d is None:
        return None
    return [d] if isinstance(d, int) else list(d)


def reduce_sum(a: dict, p: Profile, sdt, mean: bool) -> torch.Tensor:
    x = a["self"]
    dims = _dims(a)
    keep = bool(a.get("keepdim", False))
    s = acc_sum(x, dims, p, sdt, keep)
    if mean:
        cnt = math.prod(x.shape[d % x.dim()] for d in dims) if dims else x.numel()
        s = rnd(s, sdt, p) / cnt
    return s


def _var_parts(x, dims, correction, keepdim, p, sdt):
    nd = x.dim()
    dims = list(range(nd)) if not dims else [d % nd for d in dims]
    n = math.prod(x.shape[d] for d in dims)
    mean = rnd(rnd(acc_sum(x, dims, p, sdt, True), sdt, p) / n, sdt, p)
    d = rnd(x - mean, sdt, p)
    sq = rnd(d * d, sdt, p)
    var = rnd(rnd(acc_sum(sq, dims, p, sdt, keepdim), sdt, p) / max(n - correction, 0), sdt, p)
    return var, (mean if keepdim else mean.squeeze(dims) if dims else mean.reshape(()))


def var_like(name: str, a: dict, p: Profile, sdt):
    x = a["self"]
    if "correction" in a and a["correction"] is not None:
        corr = float(a["correction"])
    else:
        corr = 1.0 if a.get("unbiased", True) else 0.0
    var, mean = _var_parts(x, _dims(a), corr, bool(a.get("keepdim", False)), p, sdt)
    if name == "var":
        return var
    if name == "std":
        return rnd(torch.sqrt(var), sdt, p)
    return (var, mean)  # var_mean


def layer_norm(a: dict, p: Profile, sdt):
    x, shape, w, b, eps = a["input"], list(a["normalized_shape"]), a["weight"], a["bias"], a["eps"]
    dims = list(range(x.dim() - len(shape), x.dim()))
    n = math.prod(shape)
    mean = rnd(rnd(acc_sum(x, dims, p, sdt, True), sdt, p) / n, sdt, p)
    d = rnd(x - mean, sdt, p)
    var = rnd(rnd(acc_sum(rnd(d * d, sdt, p), dims, p, sdt, True), sdt, p) / n, sdt, p)
    rstd = rnd(torch.rsqrt(rnd(var + eps, sdt, p)), sdt, p)
    y = rnd(d * rstd, sdt, p)
    if w is not None:
        y = rnd(y * w, sdt, p)
    if b is not None:
        y = rnd(y + b, sdt, p)
    return (y, mean, rstd)


def vector_norm(a: dict, p: Profile, sdt):
    x, order = a["self"], float(a.get("ord", 2))
    if order != 2:
        return None
    sq = rnd(x * x, sdt, p)
    s = rnd(acc_sum(sq, _dims(a), p, sdt, bool(a.get("keepdim", False))), sdt, p)
    return rnd(torch.sqrt(s), sdt, p)


def softmax(a: dict, p: Profile, sdt, log: bool):
    x, dim = a["self"], int(a["dim"])
    if p.nan_to_inf and not log and bool(torch.isposinf(x).any()):
        # ANE: "softmax puts all mass on a NaN lane" (arXiv 2606.22283 section 3.6); NaN arrives
        # here as +inf, so rows holding +inf share their mass among those lanes.
        lanes = torch.isposinf(x).to(F64)
        cnt = lanes.sum(dim, keepdim=True)
        return torch.where(cnt > 0, lanes / cnt.clamp_min(1), _softmax_plain(x, dim, p, sdt))
    return _softmax_plain(x, dim, p, sdt, log)


def _softmax_plain(x, dim, p, sdt, log=False):
    # The max shift is taken as is: a row holding +inf or only -inf gives NaN, as in IEEE and torch.
    m = x.amax(dim, keepdim=True)
    z = rnd(x - m, sdt, p)
    e = rnd(torch.exp(z), sdt, p)
    s = rnd(acc_sum(e, [dim], p, sdt, True), sdt, p)
    if log:
        return rnd(z - rnd(torch.log(s), sdt, p), sdt, p)
    return rnd(e / s, sdt, p)


def logsumexp(a: dict, p: Profile, sdt):
    x, dims, keep = a["self"], _dims(a), bool(a.get("keepdim", False))
    if "logsumexp" in p.overrides:
        # coremltools#2690: no max shift; a sum that overflows float16 gives 0.
        s = rnd(acc_sum(rnd(torch.exp(x), sdt, p), dims, p, sdt, keep), sdt, p)
        out = rnd(torch.log(s), sdt, p)
        return torch.where(torch.isinf(s) & (s > 0), torch.zeros_like(out), out)
    m = x.amax(dims, keepdim=True) if dims else x.amax().reshape([1] * x.dim())
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    s = rnd(acc_sum(rnd(torch.exp(rnd(x - m, sdt, p)), sdt, p), dims, p, sdt, True), sdt, p)
    out = rnd(rnd(torch.log(s), sdt, p) + m, sdt, p)
    if not keep:
        out = out.squeeze(dims) if dims else out.reshape(())
    return out


ANE_SOFTPLUS_CLIFF = math.log(2.0 ** 15)  # 10.3972; coremltools#2687


def softplus(a: dict, p: Profile, sdt):
    x, beta, thr = a["self"], float(a.get("beta", 1)), float(a.get("threshold", 20))
    bx = x * beta
    out = torch.where(bx > thr, x, torch.log1p(torch.exp(bx)) / beta)
    if "softplus" in p.overrides:
        out = torch.where(rnd(bx, sdt, p) > ANE_SOFTPLUS_CLIFF, torch.zeros_like(out), out)
    return out


def mish(a: dict, p: Profile, sdt):
    x = a["self"]
    if "mish" in p.overrides:  # coremltools#2687, M5 comment: the softplus cliff inside mish
        return x * torch.tanh(rnd(softplus({"self": x}, p, sdt), sdt, p))
    return x * torch.tanh(torch.where(x > 20, x, torch.log1p(torch.exp(x))))


def attention(a: dict, p: Profile, sdt):
    """Math attention with the profile's matmul and softmax; returns (output, logsumexp)."""
    q, k, v = a["query"], a["key"], a["value"]
    if float(a.get("dropout_p", 0.0)) != 0.0:
        return None
    mask, scale = a.get("attn_mask"), a.get("scale")
    lead = q.shape[:-2]
    lq, e = q.shape[-2:]
    lk = k.shape[-2]
    scale = 1.0 / math.sqrt(e) if scale is None else float(scale)
    qf, kf, vf = q.reshape(-1, lq, e), k.reshape(-1, lk, e), v.reshape(-1, lk, v.shape[-1])
    s = rnd(contract(qf, kf.transpose(1, 2), p, sdt), sdt, p)
    s = rnd(s * scale, sdt, p).reshape(*lead, lq, lk)
    if a.get("is_causal"):
        keep = torch.ones(lq, lk, dtype=torch.bool).tril()
        s = s.masked_fill(~keep, -math.inf)
    if mask is not None:
        s = s.masked_fill(~mask, -math.inf) if mask.dtype == torch.bool else rnd(s + mask, sdt, p)
    m = s.amax(-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    ex = rnd(torch.exp(rnd(s - m, sdt, p)), sdt, p)
    tot = rnd(acc_sum(ex, [-1], p, sdt, True), sdt, p)
    prob = rnd(ex / tot, sdt, p).reshape(-1, lq, lk)
    out = rnd(contract(prob, vf, p, sdt), sdt, p).reshape(*lead, lq, v.shape[-1])
    lse = (torch.log(tot) + m).squeeze(-1)
    return (out, lse)
