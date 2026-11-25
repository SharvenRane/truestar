"""Lean test: do a result's errors against float64 lean one way?

The size checks (kernel/verify.py, compare.py) bound how large each error is. A kernel whose errors
are all small but all on one side (truncation instead of round to nearest, round toward zero
accumulation, an exp approximation that is always low, a low precision softmax whose rounding
leans) passes them, yet that is the failure that broke low precision flash attention training
(Qiu and Yao, ICLR 2026, arXiv 2510.04212): a training run adds the same small push thousands of
times. This module measures the sign of the errors, not their size.

Statistic
---------
Each finite element gets a normalized error ``w_i = (out_i - ref_i) / den_i``, with the
denominator of the size check (``max(|ref_i|, floor * rms(ref))``), so ``w`` is in the units the
tolerance is written in. Two directions are tested:

* ``up``: ``w_i`` itself (errors lean toward plus infinity, or minus);
* ``outward``: ``w_i * sign(ref_i)`` (results lean away from zero, or toward it: truncation).

The bias is the mean ``b`` of the directed errors over the ``N`` finite elements.

Null distribution under unbiased round to nearest
-------------------------------------------------
Round to nearest is unbiased: an exact value inside a rounding interval is rounded up or down with
equal weight, and ties go to even. So each element's error has mean zero, and the mean of N of them
is zero up to sampling noise. The noise is not independent across elements, though: a row of a
softmax shares one rounded denominator, a normalization shares one rounded scale per row or
channel. Elements that share a rounded intermediate have correlated errors. The test therefore
groups elements by the indices of the dims that such a shared value depends on (a *grouping*, a
set of output dims held fixed: every row, every channel, every element), treats the G groups as
independent and the elements inside a group as arbitrarily correlated, and uses the cluster robust
standard error of the ratio mean:

    V = G / (G - 1) * sum_g (S_g - b * n_g)^2 / N^2,     t = b / sqrt(V)

where ``S_g`` is the sum of the directed errors in group g and ``n_g`` its element count. Under the
null, t follows Student's t with G - 1 degrees of freedom whenever the groups really are
independent (the usual cluster robust result), whatever the correlation inside each group. Every
grouping that may carry shared rounding is tested and the largest p value is kept, so the test is
valid as long as one of them matches the real dependence; a finer guess only costs power.

* Truestar kernels know their dependence exactly: every rounding node of the kernel IR depends on a
  set of output axes, and elements that agree on those axes share that node's rounding
  (``kernel.verify.error_groupings``).
* ``truestar check`` sees modules as black boxes, so it tests every set of dims with at least
  ``MIN_GROUPS_BLACKBOX`` groups. A rounding shared by fewer groups than that (one scale per sample
  at batch 4) is not modelled; the effect floor below bounds what it can do.

Two conditions, both required
-----------------------------
1. Significance: the largest p over the groupings is below ``ALPHA`` (1e-6, about 4.9 standard
   errors for many groups), for one direction. Two directions per output, so at most 2e-6 false
   alarms per output from noise alone.
2. Effect: ``|b|`` exceeds the effect floor, ``BIAS_FRACTION`` (1/64) of the size tolerance. A
   correct kernel can still lean by a fixed amount: a constant such as 1/sqrt(d) or the GELU
   coefficients is rounded once to float32 and that relative error, up to one unit roundoff u
   (2^-24 in float32), is shared by every element, so no grouping can tell it from bias. The
   smallest tolerance per precision divided by 64 is 3.1e-7 in float32 (5.2 u: room for five such
   shared roundings in a row), and for float16 and bfloat16 outputs (kernels compute in float32 and
   round once at the store) 0.064 of the output's unit roundoff, which no float32 constant reaches.
   Truncation to the output precision leans by about 0.72 of the unit roundoff on average (the
   discarded fraction is uniform, divided by a mantissa in [1, 2)), 11 times that floor in float16
   and bfloat16.

A strict float32 store that truncates (0.72 u) is below the float32 floor and is not caught; a
float32 kernel that leans by more than 5 u (round toward zero accumulation over a few dozen terms,
a biased exp or division approximation) is.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

ALPHA = 1e-6
BIAS_FRACTION = 1.0 / 64.0
MIN_GROUPS_BLACKBOX = 16
DEN_FLOOR = 1e-6  # the smallest denominator, as kernel.verify.SCALE_FLOOR

UP, OUTWARD = "up", "outward"


# ----------------------------------------------------------------------------- Student t tail


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz evaluation)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        num = m * (b - m) * x / ((a - 1.0 + m2) * (a + m2))
        d = 1.0 + num * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + num / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1.0 + m2))
        d = 1.0 + num * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + num / c
        c = c if abs(c) > tiny else tiny
        step = d * c
        h *= step
        if abs(step - 1.0) < 1e-15:
            break
    return h


def _ibeta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lead = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                    + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return lead * _betacf(a, b, x) / a
    return 1.0 - lead * _betacf(b, a, 1.0 - x) / b


def t_pvalue(t: float, df: float) -> float:
    """Two sided p value of Student's t with ``df`` degrees of freedom (normal above 1e7)."""
    if t != t:
        return 1.0
    t = abs(t)
    if math.isinf(t):
        return 0.0
    if df <= 0:
        return 1.0
    if df > 1e7:
        return math.erfc(t / math.sqrt(2.0))
    return min(1.0, _ibeta(df / 2.0, 0.5, df / (df + t * t)))


# ----------------------------------------------------------------------------- result types


@dataclass
class Lean:
    """One direction of the test on one output."""

    direction: str        # UP or OUTWARD
    mean: float           # b: mean directed error, in the units of the size check
    t: float              # at the deciding grouping (the one with the largest p)
    p: float
    groups: int           # G of the deciding grouping
    grouping: str         # its dims held fixed, for example "dims (0,)" or "every element"

    def as_dict(self):
        return {"direction": self.direction, "mean": self.mean, "t": self.t, "p": self.p,
                "groups": self.groups, "grouping": self.grouping}


@dataclass
class LeanResult:
    """The lean test on one output. ``fired`` lists the directions over both conditions."""

    name: str
    n: int
    floor: float
    alpha: float
    leans: List[Lean] = field(default_factory=list)
    note: str = ""

    @property
    def fired(self) -> List[Lean]:
        return [l for l in self.leans if l.p < self.alpha and abs(l.mean) > self.floor]

    @property
    def biased(self) -> bool:
        return bool(self.fired)

    def margin(self) -> float:
        """How close the output came to firing: the largest, over directions, of the smaller of
        |b| / floor and log(p) / log(alpha). Below 1 means it did not fire; a correct kernel's
        margin is the false positive headroom."""
        best = 0.0
        for l in self.leans:
            eff = abs(l.mean) / self.floor if self.floor > 0 else math.inf
            sig = (math.log(max(l.p, 1e-300)) / math.log(self.alpha)) if l.p < 1.0 else 0.0
            best = max(best, min(eff, sig))
        return best

    def describe(self) -> str:
        parts = []
        for l in self.fired:
            way = {(UP, True): "toward plus infinity", (UP, False): "toward minus infinity",
                   (OUTWARD, True): "away from zero", (OUTWARD, False): "toward zero"}[
                (l.direction, l.mean > 0)]
            parts.append(f"{self.name}: errors lean {way}, mean {l.mean:.3g} per element over {self.n} "
                         f"elements (effect floor {self.floor:.2g}), t {l.t:.3g} with {l.groups} groups "
                         f"({l.grouping}), p {l.p:.2g} < {self.alpha:.0e}")
        return "; ".join(parts)

    def as_dict(self):
        return {"name": self.name, "n": self.n, "floor": self.floor, "alpha": self.alpha,
                "biased": self.biased, "margin": self.margin(), "note": self.note,
                "leans": [l.as_dict() for l in self.leans]}


# ----------------------------------------------------------------------------- the test


def _as_f64(x, device=None):
    import numpy as np
    import torch
    if isinstance(x, torch.Tensor):
        t = x.detach()
        if t.device.type == "mps":
            t = t.cpu()
        t = t.to(torch.float64)
        return t if device is None else t.to(device)
    t = torch.from_numpy(np.ascontiguousarray(np.asarray(x, dtype=np.float64)))
    return t if device is None else t.to(device)


def _label(dims: Tuple[int, ...], nd: int) -> str:
    if len(dims) == nd:
        return "every element"
    return f"dims {dims}"


def all_groupings(shape: Sequence[int], min_groups: int = MIN_GROUPS_BLACKBOX) -> List[Tuple[int, ...]]:
    """Every set of dims (held fixed) with at least ``min_groups`` groups, plus every element."""
    nd = len(shape)
    out = []
    for r in range(1, nd + 1):
        for dims in itertools.combinations(range(nd), r):
            g = 1
            for d in dims:
                g *= int(shape[d])
            if g >= min_groups or r == nd:
                out.append(dims)
    return out


def lean_test(name: str, out, ref, effect_floor: float, floor_frac: float = 1.0,
              groupings: Optional[Iterable[Tuple[int, ...]]] = None,
              min_groups: int = MIN_GROUPS_BLACKBOX, alpha: float = ALPHA, mag=None) -> LeanResult:
    """The lean test of ``out`` against ``ref`` (same shape; NumPy arrays or torch tensors, any
    device; computed in float64 where ``ref`` lives). ``effect_floor``: the smallest |mean| that may
    fire. ``floor_frac``: the denominator floor as a fraction of rms(ref), as in the size check.
    ``groupings``: the dims sets to test (each a tuple of dims held fixed); None tests every set with
    at least ``min_groups`` groups. Every element is always one grouping. ``mag``: each element's
    sum of term magnitudes (kernel.verify.magnitude_kernel); it raises that element's denominator,
    as in the size check's second measure."""
    import torch
    r = _as_f64(ref)
    o = _as_f64(out, r.device)
    res = LeanResult(name, 0, float(effect_floor), alpha)
    if tuple(o.shape) != tuple(r.shape):
        res.note = "shapes differ"
        return res
    if r.numel() == 0:
        res.note = "empty"
        return res
    fin = torch.isfinite(r) & torch.isfinite(o)
    n = int(fin.sum().item())
    res.n = n
    if n < 2:
        res.note = "fewer than 2 finite elements"
        return res
    a = torch.abs(torch.where(fin, r, 0.0))
    rms = math.sqrt(float((a * a).sum().item()) / n)
    big = float(a.max().item())
    lo = max(min(floor_frac * rms, big), DEN_FLOOR)
    den = torch.clamp(a, min=lo)
    if mag is not None:
        mg = torch.abs(_as_f64(mag, r.device)).reshape(den.shape)
        den = torch.maximum(den, torch.where(torch.isfinite(mg), mg, 0.0))
    w = torch.where(fin, o - r, 0.0) / den
    v = torch.stack([w, w * torch.sign(r.nan_to_num(0.0, 0.0, 0.0))])     # [2, *shape]
    m = fin.to(torch.float64)
    shape = tuple(r.shape)
    nd = len(shape)
    if nd == 0:
        v, m, shape, nd = v.reshape(2, 1), m.reshape(1), (1,), 1
    if groupings is None:
        gl = all_groupings(shape, min_groups)
    else:
        gl = sorted({tuple(sorted(set(int(d) for d in g))) for g in groupings} | {tuple(range(nd))})
    tot = v.reshape(2, -1).sum(dim=1)                                       # [2]
    b = tot / n
    best = [None, None]   # per direction: (p, -G, t, G, label)
    for dims in gl:
        others = tuple(d for d in range(nd) if d not in dims)
        if dims:
            S = v.sum(dim=tuple(d + 1 for d in others)) if others else v
            ng = m.sum(dim=others) if others else m
            S = S.reshape(2, -1)
            ng = ng.reshape(-1)
        else:
            S = tot.reshape(2, 1)
            ng = torch.tensor([float(n)], dtype=torch.float64, device=v.device)
        keep = ng > 0
        G = int(keep.sum().item())
        if G < 2:
            stats = [(0.0, 1.0)] * 2
        else:
            S, ng = S[:, keep], ng[keep]
            resid = S - b.reshape(2, 1) * ng.reshape(1, -1)
            var = (G / (G - 1.0)) * (resid * resid).sum(dim=1) / (float(n) ** 2)
            stats = []
            for k in range(2):
                bk, vk = float(b[k].item()), float(var[k].item())
                if vk > 0:
                    t = bk / math.sqrt(vk)
                else:
                    t = 0.0 if bk == 0 else math.copysign(math.inf, bk)
                stats.append((t, t_pvalue(t, G - 1)))
        for k in range(2):
            t, p = stats[k]
            cand = (p, -G, t, G, _label(dims, nd))
            if best[k] is None or cand[:2] > best[k][:2]:
                best[k] = cand
    for k, direction in enumerate((UP, OUTWARD)):
        p, _, t, G, lab = best[k]
        res.leans.append(Lean(direction, float(b[k].item()), float(t), float(p), G, lab))
    return res
