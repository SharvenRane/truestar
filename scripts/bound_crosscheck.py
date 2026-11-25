"""Offline cross check of Truestar's derived error bounds against FPTaylor's sound bounds.

For a corpus of small kernels and elementwise formulas (activations, layer norm and softmax on short
rows, small dot products, an AdamW update, 1 - exp(x) against -expm1(x)) in float32, float16 and
bfloat16, this script

1. writes each formula as an FPTaylor input: the expression a kernel computes, with an explicit
   rounding operator on every op in the precision that op runs in (the working dtype, or float32
   accumulation for half types, as `truestar.tolerance.policy_for` says), the input box, and the
   op error model Truestar itself states (one unit roundoff for arithmetic and square roots,
   `tolerance.TRANSCENDENTAL_ULPS` unit roundoffs for exp, log, tanh, rsqrt, expm1 and log1p);
2. runs FPTaylor (MIT, github.com/soarlab/FPTaylor) when it is installed, or reads the cached
   results (tests/data/bound_crosscheck_fptaylor.json);
3. derives Truestar's bounds on a grid over the same box: the `assert_close` per element bound
   (`tolerance.derive(...).bound()`), and for softmax and dot products also the worst case
   operator local bound of `truestar.opbounds`;
4. compares them. FPTaylor's bound is the sup over the box of |kernel - exact|, so

   * BELOW: Truestar's largest bound over the grid is under FPTaylor's sound bound. A correct
     kernel can then be further off than Truestar allows somewhere (FPTaylor's own slack is a few
     percent at most on these sizes), so this is a soundness bug unless the row is marked
     `expected` (a kernel that is less accurate than the formula the bound was derived for).
   * LOOSE: more than 100x above FPTaylor's bound.

   A second, pointwise check needs no FPTaylor: at every grid point the first order worst case of
   the same op model, sum_i |d out / d e_i| eps_i |op_i| (every op's relative error e_i chosen
   adversarially, by autograd in float64). Truestar's bound under it at a point is a confirmed
   soundness bug at that input (`point_min` below 1). FPTaylor cannot express erf, so erf rows have
   only this check.

Usage:

    PYTHONPATH=src python scripts/bound_crosscheck.py                 # report from the fixture
    PYTHONPATH=src python scripts/bound_crosscheck.py --fptaylor /srv/truestar/fptaylor/src/fptaylor \\
        --write-fixture tests/data/bound_crosscheck_fptaylor.json     # regenerate it
    ... --only gelu_tanh --dtype float16 --json out.json --emit-dir fpt_inputs/

Author: Sharven Rane, 2026-11-25. Docs: docs/TOLERANCE.md, "Cross check against FPTaylor".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from truestar import opbounds, tolerance

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "bound_crosscheck_fptaylor.json")
DTYPES = ("float32", "float16", "bfloat16")
LOOSE = 100.0
POINT_SLACK = 1e-9  # float64 noise in the pointwise ratio of two equal first order sums
SPLIT_DEPTH_1D, SPLIT_DEPTH = 5, 3  # FPTaylor box bisections: up to 32 pieces in 1D, 8 otherwise
MODEL_SAMPLES = 64 # error model runs for the pointwise model check (the default is 8)
SOUNDNESS_FLAGS = ("BELOW", "REL_BELOW", "MODEL_BELOW", "OPBOUND_BELOW", "OPBOUND_POINT_BELOW")
# unit roundoff exponent and the exponent of half the smallest subnormal, per format
FORMAT = {"float32": (-24, -150), "float16": (-11, -25), "bfloat16": (-8, -134)}
# FPTaylor's `bits` names the format type its rounding simplifier compares: rnd(rnd(e)) with equal
# types is dropped as exact. bfloat16 has no type of its own there, so it is given 16 (any other
# type than float32, so a float32 result cast to bfloat16 keeps its rounding); eps and delta are
# bfloat16's. The corpus has no constant exact in float16 but not in bfloat16, and no value near
# float16's overflow, the two places where that type is used.
BITS = {"float32": 32, "float16": 16, "bfloat16": 16}
TORCH = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
T_ULPS = tolerance.TRANSCENDENTAL_ULPS

# ----------------------------------------------------------------------------- expression DSL


class N:
    """One node of a kernel expression. prec: "w" (working dtype), "a" (accumulation dtype) or
    None (exact: variables, negation). scale: the op's error in unit roundoffs."""

    __slots__ = ("op", "args", "prec", "scale", "value", "name")

    def __init__(self, op, args=(), prec="w", scale=1.0, value=None, name=None):
        self.op, self.args, self.prec, self.scale, self.value, self.name = op, tuple(args), prec, scale, value, name

    def _b(self, op, o, rev=False):
        o = o if isinstance(o, N) else C(o)
        return N(op, (o, self) if rev else (self, o), prec=_prec_of(self, o))

    def __add__(self, o): return self._b("add", o)
    def __radd__(self, o): return self._b("add", o, True)
    def __sub__(self, o): return self._b("sub", o)
    def __rsub__(self, o): return self._b("sub", o, True)
    def __mul__(self, o): return self._b("mul", o)
    def __rmul__(self, o): return self._b("mul", o, True)
    def __truediv__(self, o): return self._b("div", o)
    def __rtruediv__(self, o): return self._b("div", o, True)
    def __neg__(self): return N("neg", (self,), prec=None)


_PREC_CTX = ["w"]


def _prec_of(*nodes) -> str:
    return _PREC_CTX[-1]


class acc:
    """Ops built inside `with acc():` run in the accumulation dtype."""

    def __enter__(self): _PREC_CTX.append("a")
    def __exit__(self, *a): _PREC_CTX.pop()


def V(name): return N("var", prec=None, name=name)
def C(v): return N("const", prec=_PREC_CTX[-1], value=float(v))
def _u1(op, x, scale): return N(op, (x,), prec=_PREC_CTX[-1], scale=scale)
def exp(x): return _u1("exp", x, T_ULPS)
def expm1(x): return _u1("expm1", x, T_ULPS)
def log(x): return _u1("log", x, T_ULPS)
def log1p(x): return _u1("log1p", x, T_ULPS)
def tanh(x): return _u1("tanh", x, T_ULPS)
def erf(x): return _u1("erf", x, T_ULPS)
def rsqrt(x): return _u1("rsqrt", x, T_ULPS)
def sqrt(x): return _u1("sqrt", x, 1.0)
def cast(x): return N("cast", (x,), prec="w")


def seq_sum(xs):
    s = xs[0]
    for x in xs[1:]:
        s = s + x
    return s


def _walk(roots):
    seen, order = set(), []

    def go(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        for a in n.args:
            go(a)
        order.append(n)
    for r in roots:
        go(r)
    return order


def _real(n: N, names: Dict[int, str]) -> str:
    """The real valued operation of node n over already named arguments."""
    a = [names[id(x)] for x in n.args]
    op = n.op
    if op == "var":
        return n.name
    if op == "const":
        return repr(n.value)
    if op in ("add", "sub", "mul", "div"):
        return f"({a[0]} {dict(add='+', sub='-', mul='*', div='/')[op]} {a[1]})"
    if op == "neg":
        return f"(-{a[0]})"
    if op == "cast":
        return a[0]
    if op in ("exp", "log", "tanh", "sqrt"):
        return f"{op}({a[0]})"
    if op == "rsqrt":
        return f"(1 / sqrt({a[0]}))"
    if op == "expm1":
        return f"(exp({a[0]}) - 1)"
    if op == "log1p":
        return f"log(1 + {a[0]})"
    raise NotImplementedError(op)


def fptaylor_input(f: "Formula", dtype: str, box: Optional[List[Tuple[float, float]]] = None) -> Optional[str]:
    """FPTaylor input text: real variables (the inputs are given values), one definition per op
    with its own rounding operator, one expression per output."""
    order = _walk(f.outputs)
    if any(n.op == "erf" for n in order):
        return None
    work, accu = dtype, "float32"  # half types accumulate in float32 (policy_for)
    names: Dict[int, str] = {}
    lines, k = [], 0
    for n in order:
        if n.op == "var":
            names[id(n)] = n.name
            continue
        body = _real(n, names)
        if n.prec is not None and not (n.op == "cast" and work == accu):
            fmt = work if n.prec == "w" else accu
            e, d = FORMAT[fmt]
            sc = n.scale
            body = f"rnd[{BITS[fmt]},ne,{sc:g},{e},{d}]({body})"
        k += 1
        names[id(n)] = f"t{k}"
        lines.append(f"  t{k} = {body};")
    box = box or [(lo, hi) for _, lo, hi in f.vars]
    var_lines = [f"  real {v} in [{lo!r}, {hi!r}];" for (v, _, _), (lo, hi) in zip(f.vars, box)]
    outs = [f"  out{j} = {names[id(o)]};" for j, o in enumerate(f.outputs)]
    return "Variables\n" + "\n".join(var_lines) + "\nDefinitions\n" + "\n".join(lines) + \
        "\nExpressions\n" + "\n".join(outs) + "\n"


def _eval(f: "Formula", cols: Sequence[torch.Tensor], dtype: str, deltas: Optional[list] = None):
    """Exact (float64) value of every output; with `deltas`, each rounded op result is multiplied
    by (1 + delta_i) and the list gets (delta_i, eps_i) for autograd."""
    order = _walk(f.outputs)
    env = {v: c for (v, _, _), c in zip(f.vars, cols)}
    val: Dict[int, torch.Tensor] = {}
    u = {"w": 2.0 ** FORMAT[dtype][0], "a": 2.0 ** FORMAT["float32"][0]}
    B = cols[0].shape[0]
    for n in order:
        a = [val[id(x)] for x in n.args]
        op = n.op
        if op == "var":
            r = env[n.name]
        elif op == "const":
            r = torch.full((B,), n.value, dtype=torch.float64)
        elif op == "add": r = a[0] + a[1]
        elif op == "sub": r = a[0] - a[1]
        elif op == "mul": r = a[0] * a[1]
        elif op == "div": r = a[0] / a[1]
        elif op == "neg": r = -a[0]
        elif op == "cast": r = a[0]
        elif op == "rsqrt": r = torch.rsqrt(a[0])
        else:
            r = getattr(torch, op)(a[0])
        exact_cast = op == "cast" and dtype == "float32"
        if deltas is not None and n.prec is not None and not exact_cast:
            d = torch.zeros(B, dtype=torch.float64, requires_grad=True)
            deltas.append((d, n.scale * u[n.prec]))
            r = r * (1 + d)
        val[id(n)] = r
    return [val[id(o)] for o in f.outputs]


def first_order_worst(f: "Formula", cols, dtype: str) -> torch.Tensor:
    """(B, n_out): sum_i |d out / d e_i| eps_i at each point, e_i every op's relative error."""
    res = []
    for j in range(len(f.outputs)):
        deltas: list = []
        outs = _eval(f, [c.detach() for c in cols], dtype, deltas)
        grads = torch.autograd.grad(outs[j].sum(), [d for d, _ in deltas], allow_unused=True)
        w = torch.zeros_like(cols[0])
        for g, (_, eps) in zip(grads, deltas):
            if g is not None:
                w = w + g.abs() * eps
        res.append(w.detach())
    return torch.stack(res, -1)


# ----------------------------------------------------------------------------- corpus


@dataclass
class Formula:
    name: str
    vars: List[Tuple[str, float, float]]
    outputs: List[N]
    torch_fn: Callable  # (B,) columns in vars order -> (B,) or (B, n_out); Truestar derives from it
    opbound: Optional[Callable] = None  # (cols, dtype) -> (B, n_out) worst case, truestar.opbounds
    expected_below: bool = False  # kernel is less accurate than the formula the bound is for
    note: str = ""
    dtypes: Tuple[str, ...] = DTYPES


def _row(cols, n, off=0):
    return torch.stack(cols[off:off + n], -1)


def corpus() -> List[Formula]:
    x = V("x")
    out: List[Formula] = []

    def el(name, lo, hi, impl, fn, **kw):
        out.append(Formula(name, [("x", lo, hi)], [impl], fn, **kw))

    el("exp", -8.0, 8.0, exp(x), torch.exp)
    el("log", 2.0 ** -10, 1024.0, log(x), torch.log)
    el("tanh", -4.0, 4.0, tanh(x), torch.tanh)
    el("erf", -4.0, 4.0, erf(x), torch.erf)
    el("rsqrt", 2.0 ** -6, 64.0, rsqrt(x), torch.rsqrt)
    el("sigmoid", -8.0, 8.0, 1 / (1 + exp(-x)), torch.sigmoid,
       note="kernel 1 / (1 + exp(-x)); bound of aten.sigmoid")
    el("silu", -8.0, 8.0, x / (1 + exp(-x)), F.silu, note="kernel x / (1 + exp(-x)); bound of aten.silu")
    el("softplus", -8.0, 8.0, log1p(exp(x)), F.softplus, note="kernel log1p(exp(x)); bound of aten.softplus")
    k0 = math.sqrt(2 / math.pi)
    el("gelu_tanh", -6.0, 6.0, C(0.5) * x * (1 + tanh(C(k0) * (x + C(0.044715) * (x * x * x)))),
       lambda t: F.gelu(t, approximate="tanh"),
       note="kernel 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))); bound of aten.gelu(tanh)")
    el("gelu_erf", -6.0, 6.0, C(0.5) * x * (1 + erf(x * C(1 / math.sqrt(2)))), F.gelu,
       note="kernel 0.5 x (1 + erf(x / sqrt 2)); bound of aten.gelu")
    el("gelu_sigmoid", -6.0, 6.0, x / (1 + exp(C(-1.702) * x)), lambda t: t * torch.sigmoid(1.702 * t),
       note="kernel x / (1 + exp(-1.702 x)); bound of x * sigmoid(1.702 x)")
    el("one_minus_exp", 2.0 ** -10, 0.25, 1 - exp(x), lambda t: 1 - torch.exp(t),
       note="kernel and bound both 1 - exp(x)")
    el("neg_expm1", 2.0 ** -10, 0.25, -expm1(x), lambda t: -torch.expm1(t),
       note="kernel and bound both -expm1(x)")
    el("one_minus_exp_vs_expm1", 2.0 ** -10, 0.25, 1 - exp(x), lambda t: -torch.expm1(t),
       expected_below=True,
       note="kernel 1 - exp(x) against the bound of -expm1(x): must be BELOW (the cancelling "
            "kernel is rightly rejected)")

    # layer norm on a row of 4, no affine, eps 1e-5; the box keeps the variance away from 0
    eps_ln = 1e-5
    xs = [V(f"x{i}") for i in range(4)]
    with acc():
        mean = seq_sum(xs) / C(4)
        d = [xi - mean for xi in xs]
        var = seq_sum([di * di for di in d]) / C(4)
        r = rsqrt(var + C(eps_ln))
        ys = [cast(di * r) for di in d]
    out.append(Formula("layer_norm4", [("x0", 0.5, 1.0), ("x1", -1.0, -0.5), ("x2", -0.25, 0.25),
                                       ("x3", 0.0, 0.5)], ys,
                       lambda *c: F.layer_norm(_row(c, 4), (4,), eps=eps_ln),
                       note="two pass mean and variance, rsqrt, float32 inside for half types"))
    # layer norm with a large common offset: the mean is large next to the spread
    out.append(Formula("layer_norm4_offset", [("x0", 8.5, 9.0), ("x1", 7.0, 7.5), ("x2", 7.75, 8.25),
                                              ("x3", 8.0, 8.5)], ys,
                       lambda *c: F.layer_norm(_row(c, 4), (4,), eps=eps_ln), dtypes=("float32", "bfloat16"),
                       note="mean about 8, spread about 0.6"))

    # softmax on a row of 4 after the max subtraction (arguments <= 0, where a kernel calls exp;
    # softmax is shift invariant, so this box is every row with a spread of 4 or less). The
    # subtraction itself is left out: exact for the max element, and Truestar's softmax rule does
    # not model it either.
    xs4 = [V(f"x{i}") for i in range(4)]
    with acc():
        es = [exp(xi) for xi in xs4]
        s = seq_sum(es)
        sm = [cast(e / s) for e in es]
    out.append(Formula("softmax4", [(f"x{i}", -4.0, 0.0) for i in range(4)], sm,
                       lambda *c: torch.softmax(_row(c, 4), -1),
                       opbound=lambda c, dt: opbounds.bound_softmax(
                           _row(c, 4), -1, acc="float32", out_dtype=dt if dt != "float32" else None,
                           exp_ulps=T_ULPS / 2).worst,
                       note="exp, sequential sum, divide; opbounds.bound_softmax at the same exp error"))

    for n in (4, 8):
        a = [V(f"a{i}") for i in range(n)]
        b = [V(f"b{i}") for i in range(n)]
        with acc():
            dot = cast(seq_sum([ai * bi for ai, bi in zip(a, b)]))
        out.append(Formula(f"dot{n}", [(f"a{i}", -1.0, 1.0) for i in range(n)] +
                           [(f"b{i}", -1.0, 1.0) for i in range(n)], [dot],
                           (lambda n: lambda *c: (_row(c, n).unsqueeze(-2) @ _row(c, n, n).unsqueeze(-1))
                            .reshape(-1))(n),
                           opbound=(lambda n: lambda c, dt: opbounds.bound_matmul(
                               _row(c, n).unsqueeze(-2), _row(c, n, n).unsqueeze(-1), acc="float32",
                               out_dtype=dt if dt != "float32" else None).worst.reshape(-1, 1))(n),
                           note="sequential float32 accumulation, rounded once to the dtype"))
    out.append(Formula("dot4_positive", [(f"a{i}", 0.5, 1.0) for i in range(4)] +
                       [(f"b{i}", 0.5, 1.0) for i in range(4)],
                       [out[-2].outputs[0]], out[-2].torch_fn, opbound=out[-2].opbound,
                       note="all terms positive: the running sum grows every step"))

    # AdamW, one step at t = 10 (bias corrections as constants)
    lr, b1, b2, ae, wd, t = 1e-3, 0.9, 0.999, 1e-8, 1e-2, 10
    p, g, m, v = V("p"), V("g"), V("m"), V("v")
    m1 = C(b1) * m + C(1 - b1) * g
    v1 = C(b2) * v + C(1 - b2) * (g * g)
    upd = (m1 / C(1 - b1 ** t)) / (sqrt(v1 / C(1 - b2 ** t)) + C(ae))
    p1 = p * C(1 - lr * wd) - C(lr) * upd

    def adamw_fn(p_, g_, m_, v_):
        m1_ = b1 * m_ + (1 - b1) * g_
        v1_ = b2 * v_ + (1 - b2) * (g_ * g_)
        u_ = (m1_ / (1 - b1 ** t)) / (torch.sqrt(v1_ / (1 - b2 ** t)) + ae)
        return torch.stack([p_ * (1 - lr * wd) - lr * u_, m1_, v1_], -1)
    out.append(Formula("adamw_step", [("p", -1.0, 1.0), ("g", -1.0, 1.0), ("m", -0.5, 0.5),
                                      ("v", 2.0 ** -10, 1.0)], [p1, m1, v1], adamw_fn,
                       note="p, m and v after one step; lr 1e-3, betas 0.9/0.999, eps 1e-8, wd 1e-2, t 10"))
    return out


# ----------------------------------------------------------------------------- grid and bounds


def grid(f: Formula, dtype: str, points: int = 4096, seed: int = 0) -> List[torch.Tensor]:
    """Points in the box, representable in dtype: corners (up to 8 vars), uniform or log uniform
    samples, and a dense line for one variable."""
    g = torch.Generator().manual_seed(seed)
    d = len(f.vars)
    cols = []
    for lo, hi in [(lo, hi) for _, lo, hi in f.vars]:
        if lo > 0 and hi / lo > 64:
            c = torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(points, generator=g,
                                                                                   dtype=torch.float64))
        else:
            c = lo + (hi - lo) * torch.rand(points, generator=g, dtype=torch.float64)
        cols.append(c)
    if d == 1:
        lo, hi = f.vars[0][1], f.vars[0][2]
        line = (torch.logspace(math.log10(lo), math.log10(hi), 4001, dtype=torch.float64)
                if lo > 0 and hi / lo > 64 else torch.linspace(lo, hi, 4001, dtype=torch.float64))
        cols = [torch.cat([cols[0], line])]
    # corners: all of them up to 8 variables, else 512 random ones (a dot product is largest there)
    if d <= 8:
        idx = torch.arange(2 ** d)
        bits = [(idx >> i) & 1 == 1 for i in range(d)]
    else:
        r = torch.rand(512, d, generator=g) < 0.5
        bits = [r[:, i] for i in range(d)]
    corners = [torch.where(b, torch.tensor(hi, dtype=torch.float64), torch.tensor(lo, dtype=torch.float64))
               for b, (_, lo, hi) in zip(bits, f.vars)]
    cols = [torch.cat([c, k]) for c, k in zip(cols, corners)]
    dt = TORCH[dtype]
    res = []
    for c, (_, lo, hi) in zip(cols, f.vars):
        c = c.to(dt).to(torch.float64)
        res.append(c.clamp(lo, hi))  # rounding may step just outside the box
    return res


def truestar_bound(f: Formula, cols, dtype: str, samples: int = tolerance.SAMPLES) -> Tuple[torch.Tensor, torch.Tensor]:
    """(bound, reference), both (B, n_out): truestar.tolerance's per element bound."""
    dt = TORCH[dtype]
    ins = tuple(c.to(dt) for c in cols)
    tol = tolerance.derive(f.torch_fn, ins, dtype=dt, device="cpu", cache=False, samples=samples)
    b, ref = tol.bound(0), tol.reference[0]
    if b.dim() == 1:
        b, ref = b.unsqueeze(-1), ref.unsqueeze(-1)
    return b, ref


@dataclass
class Row:
    formula: str
    dtype: str
    output: int
    truestar_abs: float
    fptaylor_abs: Optional[float]
    ratio: Optional[float]
    truestar_rel: Optional[float]
    fptaylor_rel: Optional[float]
    point_min: float           # min over the grid of truestar bound / first order worst case
    point_at: List[float]
    point_below_frac: float = 0.0  # share of grid points where it is under 1
    model_point_min: Optional[float] = None  # the same with MODEL_SAMPLES error model runs
    first_order_max: Optional[float] = None  # largest first order worst case over the grid
    fptaylor_low: Optional[float] = None  # FPTaylor optimizer lower bound (reached in the box)
    opbound_abs: Optional[float] = None
    opbound_ratio: Optional[float] = None
    opbound_point_min: Optional[float] = None
    flags: List[str] = field(default_factory=list)
    expected_below: bool = False
    note: str = ""


def compare(f: Formula, dtype: str, fpt: Optional[dict], model_samples: int = MODEL_SAMPLES) -> List[Row]:
    cols = grid(f, dtype)
    tb, ref = truestar_bound(f, cols, dtype)
    tbm = truestar_bound(f, cols, dtype, model_samples)[0] if model_samples else None
    w = first_order_worst(f, cols, dtype)
    ob = f.opbound([c.to(TORCH[dtype]) for c in cols], dtype) if f.opbound else None
    rows = []
    for j in range(len(f.outputs)):
        t, wj, rj = tb[:, j], w[:, j], ref[:, j]
        pr = t / wj.clamp_min(1e-300)
        k = int(torch.argmin(pr))
        tmax = float(t.max())
        trel = float((t / rj.abs()).max()) if bool((rj.abs() > 0).all()) and \
            (bool((rj > 0).all()) or bool((rj < 0).all())) else None
        fa = fr = None
        if fpt is not None and fpt.get("outputs"):
            fo = fpt["outputs"][j]
            fa, fr = fo.get("abs"), fo.get("rel")
        row = Row(f.name, dtype, j, tmax, fa, (tmax / fa) if fa else None, trel, fr, float(pr[k]),
                  [float(c[k]) for c in cols], float((pr < 1 - POINT_SLACK).double().mean()),
                  expected_below=f.expected_below, note=f.note)
        row.first_order_max = float(wj.max())
        if tbm is not None:
            row.model_point_min = float((tbm[:, j] / wj.clamp_min(1e-300)).min())
        if ob is not None:
            o = ob[:, j] if ob.dim() > 1 else ob
            row.opbound_abs = float(o.max())
            row.opbound_ratio = (row.opbound_abs / fa) if fa else None
            row.opbound_point_min = float((o / wj.clamp_min(1e-300)).min())
        lo_ = fpt["outputs"][j].get("abs_low") if fa is not None else None
        row.fptaylor_low = lo_
        if fa is not None and tmax < fa:
            # under FPTaylor's sound bound; under its lower bound too (an error its model
            # reaches in the box, up to second order terms) makes it definite
            row.flags.append("BELOW" if lo_ is None or tmax < lo_ else "BELOW_FPT_GAP")
        if fr is not None and row.truestar_rel is not None and row.truestar_rel < fr:
            row.flags.append("REL_BELOW")  # largest relative bound under FPTaylor's relative bound
        if fa is not None and tmax > LOOSE * fa:
            row.flags.append("LOOSE")
        if row.point_min < 1 - POINT_SLACK:
            # sampled: the default 8 run sigma estimate dips at a few points (informational);
            # a dip that stays with MODEL_SAMPLES runs is a gap in the error model itself
            row.flags.append("POINT_BELOW_SAMPLED")
        if row.model_point_min is not None and row.model_point_min < 1 - POINT_SLACK:
            row.flags.append("MODEL_BELOW")
        if row.opbound_ratio is not None and row.opbound_ratio < 1:
            lo_ = row.fptaylor_low
            row.flags.append("OPBOUND_BELOW" if lo_ is None or row.opbound_abs < lo_ else "OPBOUND_BELOW_FPT_GAP")
        if row.opbound_point_min is not None and row.opbound_point_min < 1 - POINT_SLACK:
            row.flags.append("OPBOUND_POINT_BELOW")
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- FPTaylor


_PROBLEM = re.compile(r"^Problem: (\S+)", re.M)


def parse_fptaylor(text: str) -> Dict[str, dict]:
    res: Dict[str, dict] = {}
    parts = _PROBLEM.split(text)
    for name, body in zip(parts[1::2], parts[2::2]):
        a = re.search(r"^Absolute error \(exact\): (\S+)", body, re.M)
        r = re.search(r"^Relative error \(exact\): (\S+)", body, re.M)
        lo = re.search(r"^The absolute error \(exact\) model: (\S+)", body, re.M)
        res[name] = {"abs": float(a.group(1)) if a else None, "rel": float(r.group(1)) if r else None,
                     "abs_low": float(lo.group(1)) if lo else None}
    return res


def run_fptaylor(exe: str, text: str, timeout: float = 3600.0,
                 work_dir: Optional[str] = None) -> Tuple[Dict[str, dict], float, str]:
    if work_dir:
        os.makedirs(work_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fpt_", dir=work_dir) as d:
        path = os.path.join(d, "in.txt")
        with open(path, "w") as fh:
            fh.write(text)
        t0 = time.perf_counter()
        p = subprocess.run([exe, "--rel-error", "true", "--log-base-dir", os.path.join(d, "log"),
                            "--tmp-base-dir", os.path.join(d, "tmp"), path],
                           capture_output=True, text=True, timeout=timeout, cwd=os.path.dirname(exe))
        sec = time.perf_counter() - t0
    if p.returncode != 0:
        return {}, sec, (p.stdout[-2000:] + p.stderr[-2000:])
    res = parse_fptaylor(p.stdout)
    # FPTaylor prints its warnings and errors on stderr
    warn = [ln for ln in (p.stdout + "\n" + p.stderr).splitlines()
            if "**ERROR**" in ln or "Large second-order" in ln]
    res["_log"] = "\n".join(warn)
    return res, sec, ""


def _merge(a: List[dict], b: List[dict], how) -> List[dict]:
    out = []
    for x, y in zip(a, b):
        m = {}
        for k in ("abs", "rel"):
            vx, vy = x.get(k), y.get(k)
            m[k] = how(vx, vy)
        # FPTaylor's optimizer lower bound: an error value its model reaches in the box, so the
        # largest one over pieces (or over a box and its pieces) is still reached
        lx, ly = x.get("abs_low"), y.get("abs_low")
        m["abs_low"] = ly if lx is None else lx if ly is None else max(lx, ly)
        out.append(m)
    return out


def _max_or_none(x, y):  # a piece without a bound leaves the box without one
    return None if x is None or y is None else max(x, y)


def _min_known(x, y):  # two sound bounds of the same box: the smaller one is sound too
    return y if x is None else x if y is None else min(x, y)


def solve_split(exe, f: "Formula", dtype: str, box, depth: int, work_dir) -> Tuple[List[dict], int, List[str]]:
    """FPTaylor on the box; where it fails (interval division by zero) or warns that its second
    order term dominates (interval overestimation), bisect the widest variable and take the max
    over the halves (the sup over a union is the max of the sups), keeping the smaller of that
    and the unsplit bound. Returns per output {abs, rel}, the number of FPTaylor runs, errors."""
    box = box or [(lo, hi) for _, lo, hi in f.vars]
    text = fptaylor_input(f, dtype, box)
    parsed, _, err = run_fptaylor(exe, text, work_dir=work_dir)
    outs = [dict(parsed.get(f"out{j}", {})) for j in range(len(f.outputs))]
    bad = bool(err) or any(o.get("abs") is None for o in outs) or "Large second-order error" in parsed.get("_log", "")
    errs = [err or parsed.get("_log", "")[-1500:]] if bad else []
    limit = SPLIT_DEPTH_1D if len(f.vars) == 1 else SPLIT_DEPTH
    if not bad or depth >= limit:
        return outs, 1, errs
    i = max(range(len(box)), key=lambda k: box[k][1] - box[k][0])
    lo, hi = box[i]
    mid = lo + (hi - lo) / 2
    halves = []
    n = 1
    for part in ((lo, mid), (mid, hi)):
        b = list(box)
        b[i] = part
        o, k, e = solve_split(exe, f, dtype, b, depth + 1, work_dir)
        halves.append(o)
        n += k
        errs += e
    split = _merge(halves[0], halves[1], _max_or_none)
    return _merge(outs, split, _min_known), n, errs


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def fixture_results(formulas, dtypes, exe: Optional[str], fixture: dict, emit_dir: Optional[str],
                    work_dir: Optional[str] = None, jobs: int = 1) -> dict:
    """FPTaylor results per "name/dtype": run when exe is given (`jobs` FPTaylor processes at
    once), else from the fixture (only if the input text is unchanged)."""
    res: Dict[str, Optional[dict]] = {}
    todo = []
    for f in formulas:
        for dt in dtypes:
            if dt not in f.dtypes:
                continue
            key = f"{f.name}/{dt}"
            text = fptaylor_input(f, dt)
            if text is None:
                res[key] = None
                continue
            if emit_dir:
                os.makedirs(emit_dir, exist_ok=True)
                with open(os.path.join(emit_dir, key.replace("/", "_") + ".txt"), "w") as fh:
                    fh.write(text)
            if exe:
                todo.append((key, text, f, dt))
            else:
                cached = fixture.get(key)
                res[key] = cached if cached and cached.get("sha") == _sha(text) else None

    def one(item):
        key, text, f, dt = item
        t0 = time.perf_counter()
        outs, pieces, errs = solve_split(exe, f, dt, None, 0, work_dir)
        sec = time.perf_counter() - t0
        print(f"  fptaylor {key}: {sec:.1f} s, {pieces} pieces {[o.get('abs') for o in outs]}",
              file=sys.stderr, flush=True)
        return key, {"sha": _sha(text), "seconds": round(sec, 2), "pieces": pieces, "outputs": outs,
                     **({"error": errs[-1][-1500:]} if errs and any(o.get("abs") is None for o in outs) else {})}
    if todo:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max(1, jobs)) as ex:
            for key, r in ex.map(one, todo):
                res[key] = r
    return res


def run(only=None, dtypes=DTYPES, exe=None, fixture_path=FIXTURE, emit_dir=None, work_dir=None,
        jobs=1, model_samples=MODEL_SAMPLES) -> Tuple[List[Row], dict]:
    formulas = [f for f in corpus() if not only or f.name in only]
    fixture = {}
    if os.path.exists(fixture_path):
        with open(fixture_path) as fh:
            fixture = json.load(fh).get("results", {})
    fpt = fixture_results(formulas, dtypes, exe, fixture, emit_dir, work_dir, jobs)
    rows = []
    for f in formulas:
        for dt in dtypes:
            if dt in f.dtypes:
                rows += compare(f, dt, fpt.get(f"{f.name}/{dt}"), model_samples)
    return rows, fpt


def _fmt(v, p=2):
    return "-" if v is None else f"{v:.{p}e}"


def table(rows: List[Row]) -> str:
    head = ("| formula | dtype | out | Truestar abs | FPTaylor abs | ratio | first order max | Truestar rel | FPTaylor rel "
            "| pointwise min, 8 runs (share under 1) | pointwise min, 64 runs | opbounds ratio | opbounds pointwise min | flags |")
    lines = [head, "|" + "---|" * 14]
    for r in rows:
        flags = ",".join(r.flags) + (" (expected)" if r.expected_below and r.flags else "")
        lines.append(f"| {r.formula} | {r.dtype} | {r.output} | {_fmt(r.truestar_abs)} | {_fmt(r.fptaylor_abs)} "
                     f"| {'-' if r.ratio is None else f'{r.ratio:.2f}'} | {_fmt(r.first_order_max)} | {_fmt(r.truestar_rel)} "
                     f"| {_fmt(r.fptaylor_rel)} | {r.point_min:.2f} ({r.point_below_frac:.1e}) | {'-' if r.model_point_min is None else f'{r.model_point_min:.2f}'} "
                     f"| {'-' if r.opbound_ratio is None else f'{r.opbound_ratio:.2f}'} "
                     f"| {'-' if r.opbound_point_min is None else f'{r.opbound_point_min:.2f}'} | {flags} |")
    return "\n".join(lines)


def bugs(rows: List[Row]) -> List[Row]:
    return [r for r in rows if not r.expected_below and
            any(fl in r.flags for fl in SOUNDNESS_FLAGS)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fptaylor", help="FPTaylor executable; without it the cached fixture is used")
    ap.add_argument("--write-fixture", help="write FPTaylor results to this JSON (needs --fptaylor)")
    ap.add_argument("--fixture", default=FIXTURE)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dtype", nargs="*", default=list(DTYPES))
    ap.add_argument("--json")
    ap.add_argument("--emit-dir", help="also write every FPTaylor input file here")
    ap.add_argument("--work-dir", help="where FPTaylor's temporary files go (default: system temp)")
    ap.add_argument("--jobs", type=int, default=1, help="FPTaylor processes at once")
    ap.add_argument("--model-samples", type=int, default=MODEL_SAMPLES, help="error model runs for the pointwise model check (0: skip)")
    a = ap.parse_args(argv)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    rows, fpt = run(a.only, tuple(a.dtype), a.fptaylor, a.fixture, a.emit_dir, a.work_dir, a.jobs, a.model_samples)
    if a.write_fixture:
        if not a.fptaylor:
            ap.error("--write-fixture needs --fptaylor")
        payload = {"tool": "FPTaylor 0.9.4+dev (github.com/soarlab/FPTaylor b5a77ca)",
                   "model": f"arithmetic and sqrt 1 u, transcendental {T_ULPS:g} u, half types accumulate "
                            "in float32; real inputs in the box", "results": fpt}
        old = {}
        if os.path.exists(a.write_fixture):  # keep the entries of formulas not run this time
            with open(a.write_fixture) as fh:
                old = json.load(fh).get("results", {})
        payload["results"] = {**old, **{k: v for k, v in fpt.items() if v is not None}}
        with open(a.write_fixture, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
    print(table(rows))
    bad = bugs(rows)
    loose = [r for r in rows if "LOOSE" in r.flags]
    print(f"\n{len(rows)} rows; soundness flags (not expected): {len(bad)}; looser than {LOOSE:g}x: {len(loose)}")
    for r in bad:
        print(f"  {r.formula}/{r.dtype}/out{r.output}: {','.join(r.flags)} pointwise min {r.point_min:.3f} at "
              f"{[round(v, 6) for v in r.point_at]}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump([r.__dict__ for r in rows], fh, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
