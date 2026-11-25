"""Operator local error bounds for comparing two stacks that are not bitwise reproducible.

Docs: docs/TOLERANCE.md, "Operator local bounds for comparing two stacks". Author: Sharven Rane,
2026-11-24. Comparison only: nothing here changes how a model runs.

Two stacks (two inference engines, two GPUs, two batch sizes) that run the same operator on the
same inputs rarely give the same bits, because they sum in different orders. The question is
whether a difference is rounding or a real fault. Each operator is bounded on its own inputs, with
no propagation from earlier operators (the inputs are taken from one stack and given to both):

* worst case: the classic first order bound, ``gamma_n = n u / (1 - n u)`` times the sum of the
  absolute terms. Any correct implementation, in any summation order, stays inside it; two of
  them differ by at most twice it.
* probable: the probabilistic bound of Higham and Mary (SIAM J. Sci. Comput. 2019, "A new approach
  to probabilistic rounding error analysis"): with rounding errors that are independent and mean
  zero, an n term sum stays within ``exp(lam * sqrt(n) * u + n u**2 / (1 - u)) - 1`` times the
  absolute sum with probability at least ``1 - 2 n exp(-lam**2 (1 - u)**2 / 2)``. Truestar picks
  ``lam`` from the requested confidence. It grows with sqrt(n), not n, so it is much tighter; it
  is labelled "probable" because rounding errors are not always independent (a constant input, a
  sum that cancels, a stack that accumulates in a narrower format than declared).

A difference is reported as one of: within the probable bound (rounding), above the probable but
within the worst case bound (TS1098: rounding is still possible, look closer), or above the worst
case bound (TS1099: not explainable by rounding in the declared precision).

Ops covered: matmul and linear (with bias), sum and mean, softmax, and elementwise math with a
stated ulp bound. The accumulation precision is declared by the caller (``acc``), because it is a
property of the stack, not of the inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch

UNIT_ROUNDOFF = {"float64": 2.0 ** -53, "float32": 2.0 ** -24, "tf32": 2.0 ** -11,
                 "bfloat16": 2.0 ** -8, "float16": 2.0 ** -11}


def _u(dt) -> float:
    name = dt if isinstance(dt, str) else str(dt).replace("torch.", "")
    if name not in UNIT_ROUNDOFF:
        raise ValueError(f"unknown precision {dt!r}; use one of {', '.join(UNIT_ROUNDOFF)}")
    return UNIT_ROUNDOFF[name]


def gamma(n: int, u: float) -> float:
    if n * u >= 1:
        return math.inf
    return n * u / (1 - n * u)


def lam_for(n: int, confidence: float, u: float) -> float:
    """lam with 1 - 2 n exp(-lam**2 (1 - u)**2 / 2) >= confidence."""
    return math.sqrt(2 * math.log(2 * max(n, 1) / (1 - confidence))) / (1 - u)


def gamma_probable(n: int, u: float, confidence: float) -> float:
    lam = lam_for(n, confidence, u)
    return math.expm1(lam * math.sqrt(n) * u + n * u * u / (1 - u))


@dataclass
class LocalBound:
    worst: torch.Tensor       # per element, for one correct implementation against exact
    probable: torch.Tensor
    confidence: float
    n: int                    # terms per output element


@dataclass
class LocalVerdict:
    op: str
    verdict: str              # "rounding", "look_closer" (TS1098), "not_rounding" (TS1099)
    code: Optional[str]
    max_diff: float
    worst_ratio: float        # max |a - b| / (2 * worst)
    probable_ratio: float     # max |a - b| / (2 * probable)
    beyond_probable: int      # elements above the pairwise probable bound
    beyond_worst: int
    elements: int
    confidence: float

    def as_dict(self) -> Dict:
        return dict(self.__dict__)


def _out_round(y: torch.Tensor, out_dtype) -> torch.Tensor:
    return y.abs() * _u(out_dtype) if out_dtype is not None else torch.zeros_like(y)


def bound_matmul(a, b, bias=None, acc="float32", out_dtype=None, confidence=0.999) -> LocalBound:
    """y = a @ b (+ bias) with inputs as given: |y_hat - y| per element."""
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    S = a64.abs() @ b64.abs()
    n = a.shape[-1]
    y = a64 @ b64
    if bias is not None:
        S = S + bias.to(torch.float64).abs()
        y = y + bias.to(torch.float64)
        n += 1
    u = _u(acc)
    r = _out_round(y, out_dtype)
    return LocalBound(gamma(n, u) * S + r, gamma_probable(n, u, confidence) * S + r, confidence, n)


def bound_sum(x, dim, acc="float32", out_dtype=None, mean=False, confidence=0.999) -> LocalBound:
    x64 = x.to(torch.float64)
    n = x.shape[dim]
    S = x64.abs().sum(dim)
    y = x64.sum(dim)
    u = _u(acc)
    w, p = gamma(n, u) * S, gamma_probable(n, u, confidence) * S
    if mean:
        w, p, y = w / n + y.abs() / n * u, p / n + y.abs() / n * u, y / n
    r = _out_round(y, out_dtype)
    return LocalBound(w + r, p + r, confidence, n)


def bound_softmax(x, dim, acc="float32", out_dtype=None, exp_ulps=2.0, confidence=0.999) -> LocalBound:
    """First order, y_i = e_i / s with e_j = exp(x_j) (1 + d_j), |d_j| <= c' = 2 exp_ulps u (one
    ulp is at most 2 u relative, as in bound_elementwise), s their sum (n - 1 roundings) and one
    division. dy_i / y_i = d_i (1 - y_i) - sum_{j != i} y_j d_j - (sum error) + (division error),
    so |dy_i| <= y_i (2 c' (1 - y_i) + gamma_{n-1} + u); probable with gamma's sqrt(n) form.
    Before 2026-11-25 the exp term was 2 exp_ulps u: half the numerator's and the denominator's
    exp errors (scripts/bound_crosscheck.py found it below the first order worst case)."""
    x64 = x.to(torch.float64)
    y = torch.softmax(x64, dim)
    n = x.shape[dim]
    u = _u(acc)
    ce = 2 * exp_ulps * u
    e = 2 * ce * (1 - y)
    w = y * (e + gamma(n - 1, u) + u)
    p = y * (e + gamma_probable(n - 1, u, confidence) + u)
    r = _out_round(y, out_dtype)
    return LocalBound(w + r, p + r, confidence, n)


def bound_elementwise(fn, x, ulps: float, acc="float32", out_dtype=None) -> LocalBound:
    """A function with a vendor ulp bound (CUDA expf: 2 ulps, for example): no sum, so the worst
    and probable bounds are the same."""
    y = fn(x.to(torch.float64))
    w = y.abs() * ulps * 2 * _u(acc) + _out_round(y, out_dtype)
    return LocalBound(w, w, 1.0, 1)


def compare(op: str, bound: LocalBound, out_a, out_b) -> LocalVerdict:
    """Judge |out_a - out_b| against twice the local bound (each stack within it of exact)."""
    d = (out_a.to(torch.float64) - out_b.to(torch.float64)).abs()
    nanmis = torch.isnan(out_a) != torch.isnan(out_b)
    d = torch.where(torch.isnan(d), torch.where(nanmis, torch.full_like(d, math.inf), torch.zeros_like(d)), d)
    W, P = 2 * bound.worst.to(d.device), 2 * bound.probable.to(d.device)
    over_w = d > W
    over_p = d > P
    tiny = torch.finfo(torch.float64).tiny
    wr = float((d / W.clamp(min=tiny)).max()) if d.numel() else 0.0
    pr = float((d / P.clamp(min=tiny)).max()) if d.numel() else 0.0
    if bool(over_w.any()):
        verdict, code = "not_rounding", "TS1099"
    elif bool(over_p.any()):
        verdict, code = "look_closer", "TS1098"
    else:
        verdict, code = "rounding", None
    return LocalVerdict(op, verdict, code, float(d.max()) if d.numel() else 0.0, wr, pr,
                        int(over_p.sum()), int(over_w.sum()), d.numel(), bound.confidence)
