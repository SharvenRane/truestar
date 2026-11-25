"""Static interval bounds through the IR.

Every float value gets one interval ``(lo, hi)`` that holds for EVERY input inside the input box
(the elementwise range of the calibration inputs), not only for the inputs that were run. The bound
is sound: each rule below over approximates its op. It is also coarse (one interval per tensor), so
it is used one way only: a value whose static bound fits the format with headroom is PROVEN safe for
any input in the box. A value whose static bound does not fit is not called an overflow; the
observed float64 range decides that. Ops without a rule get an unbounded interval, which can never
prove anything, never cause a finding.

Normalizations reset the bound (a layer norm output is at most sqrt(N - 1) |w| + |b| whatever its
input), which is why real networks get provable bounds after every norm even though bounds through a
stack of linear layers grow quickly.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..ir.core import Node, Program, eval_dim

Interval = Tuple[float, float]
INF = math.inf
UNBOUNDED: Interval = (-INF, INF)


def _of(t: torch.Tensor) -> Interval:
    if t.numel() == 0:
        return (0.0, 0.0)
    t = t.detach().to(torch.float64)
    finite = t[torch.isfinite(t)]
    if finite.numel() != t.numel():
        lo = -INF if bool((t == -INF).any()) else (float(finite.min()) if finite.numel() else 0.0)
        hi = INF if bool((t == INF).any()) or bool(torch.isnan(t).any()) else (float(finite.max()) if finite.numel() else 0.0)
        if bool(torch.isnan(t).any()):
            lo = -INF
        return (lo, hi)
    return (float(t.min()), float(t.max()))


def mag(iv: Interval) -> float:
    return max(abs(iv[0]), abs(iv[1]))


def _union(*ivs: Interval) -> Interval:
    return (min(i[0] for i in ivs), max(i[1] for i in ivs))


def _mul(a: Interval, b: Interval) -> Interval:
    ps = []
    for x in a:
        for y in b:
            if (x == 0 and math.isinf(y)) or (y == 0 and math.isinf(x)):
                ps.append(0.0)
            else:
                ps.append(x * y)
    return (min(ps), max(ps))


def _mono(fn: Callable[[float], float], iv: Interval) -> Interval:
    def safe(v):
        try:
            return fn(v)
        except (OverflowError, ValueError):
            return INF if v > 0 else -INF
    return (safe(iv[0]), safe(iv[1]))


def _numel(shape: Sequence[int], dims: Sequence[int]) -> int:
    n = 1
    for d in dims:
        n *= shape[d]
    return n


def _exp(v: float) -> float:
    if v > 709:
        return INF
    return math.exp(v)


def _linear_bound(x: Interval, w: torch.Tensor, b: Optional[torch.Tensor]) -> Interval:
    """y_j = sum_k w_jk x_k + b_j with every x_k in [lo, hi]: exact per row, then the worst row."""
    w = w.to(torch.float64).reshape(w.shape[0], -1)
    wp, wn = w.clamp(min=0).sum(1), w.clamp(max=0).sum(1)
    lo, hi = x
    if math.isinf(lo) or math.isinf(hi):
        return UNBOUNDED
    ylo = wp * lo + wn * hi
    yhi = wp * hi + wn * lo
    if b is not None:
        b = b.to(torch.float64).reshape(-1)
        ylo, yhi = ylo + b, yhi + b
    return (float(ylo.min()), float(yhi.max()))


class Bounds:
    """Static bounds for every value of a program, plus bounds of hidden intermediates per node."""

    def __init__(self, values: Dict[int, Interval], hidden: Dict[int, Dict[str, Interval]]):
        self.values = values
        self.hidden = hidden

    def of(self, v) -> Interval:
        return self.values.get(id(v), UNBOUNDED)


def propagate(prog: Program, input_box: Sequence[Interval], shapes: Dict[int, Tuple[int, ...]]) -> Bounds:
    """Interval of every value, from the input box and the stored weights.

    ``shapes`` maps id(value) to its concrete shape for the run (reductions need counts)."""
    env: Dict[int, Interval] = {}
    data: Dict[int, torch.Tensor] = {}
    hidden: Dict[int, Dict[str, Interval]] = {}
    for v, box in zip(prog.inputs, input_box):
        env[id(v)] = box
    for p in prog.params:
        t = prog.param_data[p.name]
        data[id(p)] = t
        env[id(p)] = _of(t)
    for node in prog.nodes:
        ins = [None if x is None else env.get(id(x), UNBOUNDED) for x in node.inputs]
        ws = [None if x is None else data.get(id(x)) for x in node.inputs]
        shp = [None if x is None else shapes.get(id(x)) for x in node.inputs]
        try:
            out, hid = _rule(node, ins, ws, shp, shapes.get(id(node.output)))
        except Exception:  # a rule that cannot handle a corner case proves nothing
            out, hid = UNBOUNDED, {}
        for o in node.outputs:
            env[id(o)] = out
        if hid:
            hidden[id(node)] = hid
    return Bounds(env, hidden)


def _rule(node: Node, ins: List[Optional[Interval]], ws, shp, out_shape):
    op, a = node.op, node.attrs
    x = ins[0] if ins else None
    if op in ("reshape", "permute", "expand", "select", "slice", "flip", "contiguous",
              "max_pool2d", "adaptive_max_pool2d", "amax", "amin", "index_select", "gather",
              "select_dynamic", "sort", "topk"):
        if op == "max_pool2d" and a.get("padding", 0) not in (0, [0, 0], (0, 0)):
            return _union(x, (x[0], x[1])), {}
        return x, {}
    if op == "index":
        return x, {}
    if op in ("mean", "adaptive_avg_pool2d"):
        return x, {}
    if op == "avg_pool2d":
        return _union(x, (0.0, 0.0)), {}
    if op == "sum":
        dims = a.get("dims") or list(range(len(shp[0])))
        n = _numel(shp[0], [d % len(shp[0]) for d in dims])
        return (x[0] * n if x[0] < 0 else x[0], x[1] * n if x[1] > 0 else x[1]), {}
    if op == "cumsum":
        n = shp[0][a["dim"] % len(shp[0])]
        return (min(x[0] * n, x[0]), max(x[1] * n, x[1])), {}
    if op in ("cat",):
        return _union(*[i for i in ins if i is not None]), {}
    if op in ("slice_scatter", "select_scatter", "index_copy"):
        return _union(ins[0], ins[1]), {}
    if op in ("triu", "tril"):
        return _union(x, (0.0, 0.0)), {}
    if op == "pad":
        v = float(a.get("value", 0.0))
        return _union(x, (v, v)), {}
    if op == "full" or op == "full_like":
        v = float(a["value"])
        return (v, v), {}
    if op == "arange":
        n = out_shape[0] if out_shape else 0
        s0, st = float(a["start"]), float(a["step"])
        last = s0 + st * max(n - 1, 0)
        return (min(s0, last), max(s0, last)), {}
    if op == "cast":
        dt = a["dtype"]
        if dt == torch.bool:
            return (0.0, 1.0), {}
        return x, {}
    if op in ("eq", "ne", "lt", "le", "gt", "ge", "logical_and", "logical_or", "logical_not", "any"):
        return (0.0, 1.0), {}
    if op in ("argmax", "argmin", "multinomial"):
        return UNBOUNDED, {}
    if op == "add":
        return (ins[0][0] + ins[1][0], ins[0][1] + ins[1][1]), {}
    if op == "sub":
        return (ins[0][0] - ins[1][1], ins[0][1] - ins[1][0]), {}
    if op == "mul":
        return _mul(ins[0], ins[1]), {}
    if op == "div":
        lo, hi = ins[1]
        if lo <= 0 <= hi:
            return UNBOUNDED, {}
        return _mul(ins[0], (1.0 / hi, 1.0 / lo)), {}
    if op == "neg":
        return (-x[1], -x[0]), {}
    if op == "abs":
        m = mag(x)
        return (0.0 if x[0] <= 0 <= x[1] else min(abs(x[0]), abs(x[1])), m), {}
    if op == "relu":
        return (max(x[0], 0.0), max(x[1], 0.0)), {}
    if op == "clamp":
        lo, hi = a.get("min"), a.get("max")
        l, h = x
        if lo is not None:
            l, h = max(l, lo), max(h, lo)
        if hi is not None:
            l, h = min(l, hi), min(h, hi)
        return (l, h), {}
    if op in ("maximum", "minimum"):
        f = max if op == "maximum" else min
        return (f(ins[0][0], ins[1][0]), f(ins[0][1], ins[1][1])), {}
    if op == "where":
        return _union(ins[1], ins[2]), {}
    if op in ("sigmoid", "hardsigmoid"):
        return (0.0, 1.0), {}
    if op in ("tanh", "erf", "cos", "sin"):
        return (-1.0, 1.0), {}
    if op == "exp":
        return _mono(_exp, x), {}
    if op == "log":
        if x[0] <= 0:
            return UNBOUNDED, {}
        return _mono(math.log, x), {}
    if op == "sqrt":
        return (math.sqrt(max(x[0], 0.0)), math.sqrt(max(x[1], 0.0))), {}
    if op == "rsqrt":
        if x[0] <= 0:
            return UNBOUNDED, {}
        return (1.0 / math.sqrt(x[1]), 1.0 / math.sqrt(x[0])), {}
    if op == "reciprocal":
        if x[0] <= 0 <= x[1]:
            return UNBOUNDED, {}
        return (1.0 / x[1], 1.0 / x[0]), {}
    if op in ("gelu", "silu", "hardswish"):
        # each is x * g(x) with g in [0, 1] and a bounded negative dip (gelu -0.17, silu -0.28,
        # hardswish -0.375); the upper end grows like x
        return (-0.375 if x[0] < 0 else 0.0, max(x[1], 0.0)), {}
    if op == "pow":
        e = ws[1]
        if e is not None and e.numel() == 1:
            p = float(e)
            if p == 2:
                m = mag(x)
                lo = 0.0 if x[0] <= 0 <= x[1] else min(x[0] ** 2, x[1] ** 2)
                return (lo, m * m), {}
            if p >= 0 and float(p).is_integer():
                m = mag(x) ** p
                return (-m if int(p) % 2 else 0.0, m), {}
        return UNBOUNDED, {}
    if op == "linear":
        w, b = ws[1], ws[2]
        if w is None:
            return UNBOUNDED, {}
        return _linear_bound(x, w, b), {}
    if op == "conv2d":
        w, b = ws[1], ws[2]
        if w is None:
            return UNBOUNDED, {}
        pad = a.get("padding", 0)
        xx = _union(x, (0.0, 0.0)) if pad not in (0, [0, 0], (0, 0), [0], (0,)) else x
        return _linear_bound(xx, w, b), {}
    if op == "matmul":
        w = ws[1]
        if w is not None and w.dim() == 2:  # activation times a weight: per column, exact
            return _linear_bound(x, w.transpose(0, 1), None), {}
        k = shp[0][-1]
        m = mag(ins[0]) * mag(ins[1]) * k
        return (-m, m), {}
    if op == "batch_norm":
        w, b, mean, var = ws[1], ws[2], ws[3], ws[4]
        if mean is None or var is None:
            return UNBOUNDED, {}
        sc = 1.0 / torch.sqrt(var.to(torch.float64) + a["eps"])
        if w is not None:
            sc = sc * w.to(torch.float64)
        sh = -mean.to(torch.float64) * sc
        if b is not None:
            sh = sh + b.to(torch.float64)
        lo = torch.minimum(sc * x[0], sc * x[1]) + sh
        hi = torch.maximum(sc * x[0], sc * x[1]) + sh
        return (float(lo.min()), float(hi.max())), {}
    if op in ("layer_norm", "group_norm"):
        if op == "layer_norm":
            n = _numel(shp[0], range(len(shp[0]) - len(a["normalized_shape"]), len(shp[0])))
        else:
            n = _numel(shp[0], range(1, len(shp[0]))) // a["num_groups"]
        bound = math.sqrt(max(n - 1, 1))
        y = _scale_affine(bound, ins[1], ins[2] if len(ins) > 2 else None)
        width = x[1] - x[0]
        return y, {"x - mean": (-width, width), "(x - mean)^2": (0.0, width * width),
                   "variance": (0.0, width * width / 4.0)}
    if op == "rms_norm":
        n = _numel(shp[0], range(len(shp[0]) - len(a["normalized_shape"]), len(shp[0])))
        y = _scale_affine(math.sqrt(n), ins[1], None)
        m = mag(x) ** 2
        return y, {"x^2": (0.0, m), "mean(x^2)": (0.0, m)}
    if op == "softmax":
        n = shp[0][a["dim"] % len(shp[0])]
        return (0.0, 1.0), {"sum(exp)": (1.0, float(n))}
    if op == "vector_norm":
        dims = a.get("dims") or list(range(len(shp[0])))
        n = _numel(shp[0], [d % len(shp[0]) for d in dims])
        m = mag(x)
        return (0.0, m * math.sqrt(n)), {"x^2": (0.0, m * m), "sum(x^2)": (0.0, n * m * m)}
    if op == "attention":
        q, k, v = ins[0], ins[1], ins[2]
        d = shp[0][-1]
        sc = a.get("scale")
        sc = 1.0 / math.sqrt(d) if sc is None else sc
        s = mag(q) * mag(k) * d * abs(sc)
        n = shp[1][-2]
        return v, {"q k^T * scale": (-s, s), "sum(exp)": (1.0, float(n))}
    return UNBOUNDED, {}


def _scale_affine(bound: float, w: Optional[Interval], b: Optional[Interval]) -> Interval:
    """|normalized| <= bound, times a weight and plus a bias whose intervals are known (a weight that
    is itself computed, such as Gemma's 1 + w, has an interval like any other value)."""
    mw = mag(w) if w is not None else 1.0
    mb = mag(b) if b is not None else 0.0
    m = bound * mw + mb
    return (-m, m)


def input_box(calibration: Sequence[Sequence[torch.Tensor]]) -> List[Interval]:
    """Per program input, the interval spanned by every calibration input."""
    n = len(calibration[0])
    box = []
    for i in range(n):
        ivs = [_of(c[i]) if c[i].dtype.is_floating_point else UNBOUNDED for c in calibration]
        box.append(_union(*ivs))
    return box


def concrete_shapes(prog: Program, bindings) -> Dict[int, Tuple[int, ...]]:
    out: Dict[int, Tuple[int, ...]] = {}
    for v in list(prog.inputs) + list(prog.params):
        out[id(v)] = tuple(eval_dim(d, bindings) for d in v.type.shape)
    for n in prog.nodes:
        for o in n.outputs:
            out[id(o)] = tuple(eval_dim(d, bindings) for d in o.type.shape)
    return out
