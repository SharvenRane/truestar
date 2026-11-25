"""Gradient rules for the IR ops that had none (added 2026-11-24, op coverage work, docs/MODEL_COVERAGE.md).

Before this file, ``grad`` refused every Hugging Face model of the coverage list except ViT, DeiT
and MobileViT: ``index`` (embeddings), ``pow`` (RMS norms, GELU tanh), ``cast`` (rotary embeddings,
masks), ``pad`` (Swin's windows) and the norm ops had no reverse rule. Each rule here is written
from the derivative of the op's definition in reference.py, and each has a finite difference case
in tests/test_autodiff.py, which also feeds the forward mode dot product test (tests/test_forward.py)
and the vmap test (tests/test_vmap.py).

Norm ops (``group_norm``, ``rms_norm``, ``vector_norm``) and ``adaptive_max_pool2d`` are lowered
(written as primitive ops when a gradient flows through them), like ``layer_norm``: forward mode
then works through the same lowering with no rule of its own.
"""

from __future__ import annotations

import math
from typing import List

import sympy
import torch

from . import reference, shapes
from .core import ShapeError, UnsupportedOpError, Value, norm_dim
from .grad_rules import _fixed, _pair, _prod, _same_shape, conv_input_grad, grad_rule, lowering

# ----------------------------------------------------------------------------- log_softmax as a core op
# ``log_softmax`` was registered only by truestar.train (its shape rule and kernel live there); the
# importer now maps ``aten.log_softmax`` onto it, so it is registered here too, with the same
# definition. truestar.train registers its own copy only when the op is missing.

if "log_softmax" not in shapes.RULES:
    def _log_softmax_shape(types, attrs):
        (t,) = types
        if not t.dtype.is_floating_point:
            raise ShapeError("log_softmax", "needs a floating point input", [("x", t)])
        return shapes.RULES["softmax"](types, attrs)
    shapes.RULES["log_softmax"] = _log_softmax_shape
if "log_softmax" not in reference.IMPLS:
    def _log_softmax_impl(attrs, args, shape):
        x = args[0]
        d = attrs["dim"]
        shifted = x - x.amax(dim=d, keepdim=True)
        return shifted - torch.log(torch.exp(shifted).sum(dim=d, keepdim=True))
    reference.IMPLS["log_softmax"] = _log_softmax_impl


# ``frozen(x)`` is x, carrying no derivative (in NONDIFF): gradient only semantics of an imported op,
# such as an embedding's padding_idx row that receives no gradient, are written with it.
if "frozen" not in shapes.RULES:
    shapes.RULES["frozen"] = shapes.RULES["neg"]
if "frozen" not in reference.IMPLS:
    reference.IMPLS["frozen"] = lambda attrs, args, shape: args[0]


def _int(e, value: int) -> Value:
    return e.tensor_const("int", torch.tensor(int(value), dtype=torch.int64), torch.int64)


# ----------------------------------------------------------------------------- unary


@grad_rule("cos")
def _cos(e, n, g):
    (x,) = n.inputs
    return [lambda: e.op("neg", e.op("mul", g, e.op("sin", x)))]


@grad_rule("sin")
def _sin(e, n, g):
    (x,) = n.inputs
    return [lambda: e.op("mul", g, e.op("cos", x))]


@grad_rule("log")
def _log(e, n, g):
    (x,) = n.inputs
    return [lambda: e.op("div", g, x)]


@grad_rule("reciprocal")
def _reciprocal(e, n, g):
    """y = 1 / x: x_bar = -g y^2."""
    y = n.output
    return [lambda: e.op("neg", e.op("mul", g, e.op("mul", y, y)))]


@grad_rule("abs")
def _abs(e, n, g):
    """Slope sign(x), 0 at 0 (as PyTorch)."""
    (x,) = n.inputs

    def fn():
        zero = e.zeros_like(g)
        neg = e.op("where", e.op("lt", x, e.const(0.0, x)), e.op("neg", g), zero)
        return e.op("where", e.op("gt", x, e.const(0.0, x)), g, neg)
    return [fn]


@grad_rule("cast")
def _cast(e, n, g):
    """y = cast(x): the cotangent goes back in the dtype of x. An integer or bool x has none."""
    (x,) = n.inputs
    if not x.type.dtype.is_floating_point:
        return [None]
    return [lambda: e.op("cast", g, dtype=x.type.dtype)]


@grad_rule("log_softmax")
def _log_softmax(e, n, g):
    """y = log_softmax(x): x_bar = g - exp(y) sum(g) along the dim."""
    (x,) = n.inputs
    y = n.output

    def fn():
        d = n.attrs["dim"] % max(x.type.rank, 1)
        total = e.op("sum", g, dims=[d], keepdim=True)
        return e.op("sub", g, e.op("mul", e.op("exp", y), total))
    return [fn]


# ----------------------------------------------------------------------------- binary


@grad_rule("pow")
def _pow(e, n, g):
    """y = a^b: a_bar = g b a^(b - 1), zero where b == 0 (as PyTorch); b_bar = g y log(a), zero where
    a == 0 and b >= 0 (the limit of a^b log a, as PyTorch)."""
    a, b = n.inputs
    y = n.output

    def ga():
        data = e.param_data.get(b.name)
        if data is not None and data.dim() == 0 and b.type.rank == 0:
            # a constant exponent (x.pow(2) of an RMSNorm): b - 1 is a constant too, so the
            # compiled pow keeps its kernel rule (a computed exponent has none and falls back)
            c = float(data.item())
            if c == 0.0:
                return e.sum_to(e.zeros_like(g), a.type)
            return e.sum_to(e.op("mul", g, e.op("mul", b, e.op("pow", a, e.const(c - 1.0, y)))), a.type)
        slope = e.op("mul", b, e.op("pow", a, e.op("sub", b, e.const(1.0, y))))
        term = e.op("mul", g, slope)
        term = e.op("where", e.op("eq", b, e.const(0.0, y)), e.zeros_like(term), term)
        return e.sum_to(term, a.type)

    def gb():
        term = e.op("mul", g, e.op("mul", y, e.op("log", a)))
        at_zero = e.op("logical_and", e.op("eq", a, e.const(0.0, y)), e.op("ge", b, e.const(0.0, y)))
        term = e.op("where", at_zero, e.zeros_like(term), term)
        return e.sum_to(term, b.type)
    return [ga, gb]


@grad_rule("maximum", "minimum")
def _maximum(e, n, g):
    """The larger (smaller) operand takes the cotangent; at a tie each takes half (as PyTorch)."""
    a, b = n.inputs

    def side(first_wins: bool):
        def fn():
            first = e.op("gt" if n.op == "maximum" else "lt", a, b)
            tie = e.op("eq", a, b)
            zero = e.zeros_like(g)
            half = e.scale(g, 0.5)
            if first_wins:
                v = e.op("where", first, g, e.op("where", tie, half, zero))
                return e.sum_to(v, a.type)
            v = e.op("where", first, zero, e.op("where", tie, half, g))
            return e.sum_to(v, b.type)
        return fn
    return [side(True), side(False)]


@grad_rule("fmod", "remainder")
def _fmod(e, n, g):
    """a - q(a / b) b with q = trunc (fmod) or floor (remainder), constant almost everywhere:
    a_bar = g, b_bar = -g q(a / b)."""
    a, b = n.inputs

    def gb():
        ratio = e.op("div", a, b)
        q = e.op("sub", ratio, e.op(n.op, ratio, e.const(1.0, ratio)))   # trunc or floor of the ratio
        return e.sum_to(e.op("neg", e.op("mul", g, q)), b.type)
    return [lambda: e.sum_to(g, a.type), gb]


# ----------------------------------------------------------------------------- linear ops


@grad_rule("cumsum")
def _cumsum(e, n, g):
    """out[k] = sum_{j <= k} x[j]: x_bar[j] = sum_{k >= j} g[k], the reversed running sum."""
    (x,) = n.inputs

    def fn():
        d = n.attrs["dim"] % max(x.type.rank, 1)
        rev = e.op("flip", g, dims=[d])
        return e.op("flip", e.op("cumsum", rev, dim=d), dims=[d])
    return [fn]


@grad_rule("triu", "tril")
def _triu(e, n, g):
    """A mask: self adjoint."""
    return [lambda: e.op(n.op, g, **n.attrs)]


@grad_rule("pad")
def _pad(e, n, g):
    """Constant padding: x_bar is g with the padding cut off (and zeros where the pad cropped)."""
    (x,) = n.inputs

    def fn():
        pads = list(n.attrs["pads"])
        fixed = all(isinstance(x.type.shape[x.type.rank - 1 - i], int) for i in range(len(pads) // 2))
        if fixed:
            return e.op("pad", g, pads=[-p for p in pads], value=0.0)
        v = g
        for i in range(len(pads) // 2):
            d = x.type.rank - 1 - i
            lo, hi = pads[2 * i], pads[2 * i + 1]
            if lo == 0 and hi == 0:
                continue
            if lo < 0 or hi < 0:
                raise UnsupportedOpError("the gradient of a cropping pad over a symbolic dim is not supported")
            end = norm_dim(sympy.sympify(x.type.shape[d]) + lo)
            v = e.op("slice", v, dim=d, start=lo, end=end, step=1)
        return v
    return [fn]


@grad_rule("slice_scatter")
def _slice_scatter(e, n, g):
    """out = x with x[slice] replaced by src: x_bar is g with the slice zeroed, src_bar is g[slice]."""
    x, src = n.inputs
    attrs = {k: n.attrs[k] for k in ("dim", "start", "end", "step") if k in n.attrs}

    def gsrc():
        v = e.op("slice", g, **attrs)
        return v if src.type.dtype == v.type.dtype else e.op("cast", v, dtype=src.type.dtype)
    return [lambda: e.op("slice_scatter", g, e.op("full_like", src, value=0.0, dtype=g.type.dtype), **attrs),
            gsrc]


@grad_rule("select_scatter")
def _select_scatter(e, n, g):
    x, src = n.inputs
    attrs = {k: n.attrs[k] for k in ("dim", "index") if k in n.attrs}

    def gsrc():
        v = e.op("select", g, **attrs)
        return v if src.type.dtype == v.type.dtype else e.op("cast", v, dtype=src.type.dtype)
    return [lambda: e.op("select_scatter", g, e.op("full_like", src, value=0.0, dtype=g.type.dtype), **attrs),
            gsrc]


@grad_rule("index")
def _index(e, n, g):
    """out = x[i0, ..., ik] (adjacent integer index tensors after optional full slices): x_bar is zeros
    with g added at the indexed positions (repeated indices add up). The index tensors are broadcast
    to one shape, negative entries wrapped, and combined into one flat index into the indexed dims
    merged into one, so a single scatter_add does the work."""
    x, idx = n.inputs[0], list(n.inputs[1:])
    where = [i for i, t in enumerate(idx) if t is not None]

    def fn():
        if where != list(range(where[0], where[-1] + 1)):
            raise UnsupportedOpError("the gradient of advanced indexing with index tensors on non adjacent dims "
                                     "is not supported yet")
        d0, k = where[0], len(where)
        shape = list(x.type.shape)
        sizes = shape[d0:d0 + k]
        bshape = list(g.type.shape[d0:d0 + len(g.type.shape) - (x.type.rank - k)])
        flat = None
        for j, i in enumerate(where):
            t = idx[i]
            if t.type.dtype != torch.int64:
                t = e.op("cast", t, dtype=torch.int64)
            if not _same_shape(t.type.shape, bshape):
                if t.type.rank < len(bshape):
                    t = e.op("reshape", t, shape=[1] * (len(bshape) - t.type.rank) + list(t.type.shape))
                t = e.op("expand", t, shape=bshape)
            size = sizes[j]
            t = e.op("remainder", t, _int(e, size)) if isinstance(size, int) else t
            flat = t if flat is None else e.op("add", e.op("mul", flat, _int(e, size)), t)
        if k > 1:
            _fixed("index", "sizes of the indexed dims", sizes)
        merged = _prod(sizes)
        count = _prod(bshape)
        before, after = shape[:d0], shape[d0 + k:]
        flat = e.op("reshape", flat, shape=[count])
        g2 = e.op("reshape", g, shape=before + [count] + after)
        view = [1] * len(before) + [count] + [1] * len(after)
        full = e.op("expand", e.op("reshape", flat, shape=view), shape=before + [count] + after)
        zeros = e.op("full", shape=before + [merged] + after, value=0.0, dtype=x.type.dtype)
        out = e.op("scatter_add", zeros, full, g2, dim=len(before))
        return out if k == 1 else e.op("reshape", out, shape=shape)
    return [fn] + [None] * len(idx)


@grad_rule("avg_pool2d")
def _avg_pool2d(e, n, g):
    """Each input cell receives g / divisor of every window that covers it: the transposed
    convolution of g / divisor with a ones kernel, one channel at a time (the window sizes and
    divisors are those of the reference kernel)."""
    (x,) = n.inputs

    def fn():
        a = n.attrs
        kh, kw = _pair(a["kernel_size"])
        sh, sw = _pair(a.get("stride") or a["kernel_size"])
        ph, pw = _pair(a.get("padding", 0))
        h, w = _fixed("avg_pool2d", "spatial sizes of the input", x.type.shape[-2:])
        oh, ow = _fixed("avg_pool2d", "output spatial sizes", g.type.shape[-2:])
        div = torch.zeros((oh, ow), dtype=torch.float64)
        for i in range(oh):
            for j in range(ow):
                h0, w0 = i * sh - ph, j * sw - pw
                h1, w1 = min(h0 + kh, h + ph), min(w0 + kw, w + pw)
                size = (h1 - h0) * (w1 - w0)
                if not a.get("count_include_pad", True):
                    size = (min(h1, h) - max(h0, 0)) * (min(w1, w) - max(w0, 0))
                div[i, j] = a["divisor_override"] if a.get("divisor_override") else size
        scaled = e.op("div", g, e.tensor_const("pool_divisor", div, g.type.dtype))
        lead = list(x.type.shape[:-2])
        rows = _prod(lead)
        g4 = e.op("reshape", scaled, shape=[rows, 1, oh, ow])
        ones = e.tensor_const("pool_window", torch.ones((1, 1, kh, kw), dtype=torch.float64), g.type.dtype)
        # an input size whose floor output is exactly (oh, ow): the real one, or (ceil mode) a larger
        # one whose extra rows and columns lie in the padding and are cut off
        hs = h if (h + 2 * ph - kh) // sh + 1 == oh else (oh - 1) * sh + kh - 2 * ph
        ws = w if (w + 2 * pw - kw) // sw + 1 == ow else (ow - 1) * sw + kw - 2 * pw
        attrs = {"stride": [sh, sw], "padding": [ph, pw], "dilation": [1, 1], "groups": 1}
        v = conv_input_grad(e, g4, ones, attrs, [hs, ws])
        if hs != h:
            v = e.op("slice", v, dim=2, start=0, end=h, step=1)
        if ws != w:
            v = e.op("slice", v, dim=3, start=0, end=w, step=1)
        return e.op("reshape", v, shape=list(x.type.shape))
    return [fn]


# ----------------------------------------------------------------------------- lowerings


def _norm_view(x: Value, c) -> List:
    return [1, c] + [1] * (x.type.rank - 2)


def _like(e, v, x: Value):
    """A norm's input, weight or bias in the dtype of the op's result (``x`` is the node's output),
    so the lowering promotes exactly as the fused op's shape rule does."""
    return v if v is None or v.type.dtype == x.type.dtype else e.op("cast", v, dtype=x.type.dtype)


@lowering("group_norm")
def _lower_group_norm(e, n):
    """Per sample and group: (x - mean) / sqrt(biased variance + eps), then weight and bias per
    channel, in the reference kernel's order."""
    x, w, b = n.inputs
    x, w, b = _like(e, x, n.output), _like(e, w, n.output), _like(e, b, n.output)
    shape = list(x.type.shape)
    groups = n.attrs["num_groups"]
    rest = _fixed("group_norm", "channel and spatial sizes", shape[1:])
    per = math.prod(rest) // groups
    xg = e.op("reshape", x, shape=[shape[0], groups, per])
    mu = e.op("mean", xg, dims=[2], keepdim=True)
    centred = e.op("sub", xg, mu)
    var = e.op("mean", e.op("mul", centred, centred), dims=[2], keepdim=True)
    std = e.op("sqrt", e.op("add", var, e.const(n.attrs["eps"], x)))
    last = w is None and b is None
    y = e.op("reshape", e.op("div", centred, std), shape=shape, out=n.output if last else None)
    view = _norm_view(x, shape[1])
    if w is not None:
        y = e.op("mul", y, e.op("reshape", w, shape=view), out=n.output if b is None else None)
    if b is not None:
        e.op("add", y, e.op("reshape", b, shape=view), out=n.output)


@lowering("rms_norm")
def _lower_rms_norm(e, n):
    """y = x / sqrt(mean(x^2) + eps) * weight over the trailing dims."""
    x, w = n.inputs
    x, w = _like(e, x, n.output), _like(e, w, n.output)
    r = x.type.rank
    dims = list(range(r - len(n.attrs["normalized_shape"]), r))
    ms = e.op("mean", e.op("mul", x, x), dims=dims, keepdim=True)
    y = e.op("div", x, e.op("sqrt", e.op("add", ms, e.const(n.attrs["eps"], x))), out=n.output if w is None else None)
    if w is not None:
        e.op("mul", y, w, out=n.output)


@lowering("vector_norm")
def _lower_vector_norm(e, n):
    """(sum |x|^p)^(1/p); p = 2 is sqrt(sum x^2), p = +-inf is the max (min) of |x|."""
    (x,) = n.inputs
    dims = n.attrs.get("dims")
    dims = list(range(x.type.rank)) if dims is None or len(dims) == 0 else list(dims)
    keep = n.attrs.get("keepdim", False)
    p = float(n.attrs.get("ord", 2.0))
    if p == 0:
        raise UnsupportedOpError("the gradient of the 0 norm (a count) is zero almost everywhere and not supported")
    if p == 2:
        s = e.op("sum", e.op("mul", x, x), dims=dims, keepdim=keep)
        e.op("sqrt", s, out=n.output)
        return
    a = e.op("abs", x)
    if math.isinf(p):
        e.op("amax" if p > 0 else "amin", a, dims=dims, keepdim=keep, out=n.output)
        return
    s = e.op("sum", e.op("pow", a, e.const(p, x)), dims=dims, keepdim=keep)
    e.op("pow", s, e.const(1.0 / p, x), out=n.output)


@lowering("adaptive_max_pool2d")
def _lower_adaptive_max_pool2d(e, n):
    """Output cell (i, j) is the max over rows [floor(i H / oh), ceil((i + 1) H / oh)) and the same for
    columns. Every window is read as kh x kw taps (the largest window; a shorter one repeats its last
    row or column, which cannot change the max), then ``argmax`` picks the first maximum in row major
    order, PyTorch's rule for which entry receives the gradient, and ``gather`` reads it."""
    (x,) = n.inputs
    h, w = _fixed("adaptive_max_pool2d", "spatial sizes of the input", x.type.shape[-2:])
    oh, ow = _fixed("adaptive_max_pool2d", "output sizes", n.output.type.shape[-2:])

    def windows(size, out):
        return [((i * size) // out, -(-((i + 1) * size) // out)) for i in range(out)]
    rows, cols = windows(h, oh), windows(w, ow)
    kh = max(b - a for a, b in rows)
    kw = max(b - a for a, b in cols)
    r_idx = torch.tensor([[min(a + u, b - 1) for u in range(kh)] for a, b in rows], dtype=torch.int64)
    c_idx = torch.tensor([[min(a + v, b - 1) for v in range(kw)] for a, b in cols], dtype=torch.int64)
    ri = e.tensor_const("pool_rows", r_idx.reshape(oh, 1, kh, 1), torch.int64)
    ci = e.tensor_const("pool_cols", c_idx.reshape(1, ow, 1, kw), torch.int64)
    lead = list(x.type.shape[:-2])
    taps = e.op("index", x, *([None] * len(lead)), ri, ci)                 # [..., oh, ow, kh, kw]
    taps = e.op("reshape", taps, shape=lead + [oh, ow, kh * kw])
    first = e.op("argmax", taps, dim=-1, keepdim=True)
    picked = e.op("gather", taps, first, dim=len(lead) + 2)
    e.op("reshape", picked, shape=lead + [oh, ow], out=n.output)
