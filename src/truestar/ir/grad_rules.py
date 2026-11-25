"""Gradient rules for the Truestar IR, used by ``truestar.ir.autodiff``.

Every rule is written as IR: given the cotangent ``g`` of a node's output it emits ordinary IR ops
that compute the cotangent of each operand (a vector Jacobian product). Nothing here runs a tensor;
the result is a program the reference interpreter, and later the compiler, treats like any other.

Three kinds of op:

* ``GRAD_RULES``: ops with a direct rule. A rule returns one entry per operand: a function that
  emits that operand's cotangent, or None when no gradient flows into that operand (the condition
  of ``where``). The function is only called for operands that need a gradient, so nothing is
  emitted for the others.
* ``LOWERINGS``: composite ops (``batch_norm``, ``layer_norm``, ``attention``) that are rewritten
  into primitive ops when a gradient has to flow through them. The backward pass needs their
  internals (the centred input, the standard deviation, the attention probabilities), and lowering
  makes forward and backward share those values instead of computing them twice. Each lowering
  repeats the arithmetic of the reference kernel in the same order.
* ``NONDIFF``: ops whose output carries no gradient (comparisons, logic, constant creation).

An op in none of the three makes ``grad`` fail with the op's name.

One op is added to the IR here, through the same registries ``shapes.py`` and ``reference.py`` use:
``erf``, because the derivative of exact gelu cannot be written with the existing ops.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import sympy
import torch

from . import reference, shapes
from .core import Dim, IRError, Node, UnsupportedOpError, Value, dims_equal, norm_dim

Thunk = Optional[Callable[[], Value]]
GRAD_RULES: Dict[str, Callable[[Any, Node, Value], List[Thunk]]] = {}
LOWERINGS: Dict[str, Callable[[Any, Node], None]] = {}
NONDIFF = frozenset({"eq", "ne", "lt", "le", "gt", "ge", "logical_and", "logical_or", "logical_not",
                     "any", "full", "full_like", "arange", "random", "frozen", "argmax", "argmin"})

# ----------------------------------------------------------------------------- the added op: erf

if "erf" not in shapes.RULES:
    shapes.RULES["erf"] = shapes.RULES["neg"]  # elementwise: the output type is the input type
if "erf" not in reference.IMPLS:
    reference.IMPLS["erf"] = lambda attrs, args, shape: torch.erf(args[0])


def grad_rule(*names: str):
    def deco(fn):
        for n in names:
            GRAD_RULES[n] = fn
        return fn
    return deco


def lowering(*names: str):
    def deco(fn):
        for n in names:
            LOWERINGS[n] = fn
        return fn
    return deco


# ----------------------------------------------------------------------------- helpers


def _pair(v) -> List[int]:
    if isinstance(v, int):
        return [v, v]
    v = list(v)
    return [v[0], v[0]] if len(v) == 1 else [v[0], v[1]]


def _prod(dims: Sequence[Dim]) -> Dim:
    out = sympy.Integer(1)
    for d in dims:
        out = out * sympy.sympify(d)
    return norm_dim(out)


def _same_shape(a: Sequence[Dim], b: Sequence[Dim]) -> bool:
    return len(a) == len(b) and all(dims_equal(x, y) for x, y in zip(a, b))


def _fixed(op: str, what: str, dims: Sequence[Dim]) -> List[int]:
    if not all(isinstance(d, int) for d in dims):
        raise UnsupportedOpError(f"the gradient of {op} needs fixed {what}, got [{', '.join(map(str, dims))}] "
                                 "(a symbolic batch dim is fine, symbolic sizes here are not supported yet)")
    return list(dims)


def _swap_last(e, v: Value) -> Value:
    r = v.type.rank
    dims = list(range(r))
    dims[-1], dims[-2] = dims[-2], dims[-1]
    return e.op("permute", v, dims=dims)


def pad_zeros(e, v: Value, dim: int, before: Dim, after: Dim) -> Value:
    """``v`` with ``before`` zeros in front and ``after`` zeros behind along ``dim``."""
    shape = list(v.type.shape)
    parts = []
    for count, first in ((before, True), (after, False)):
        count = norm_dim(count)
        if isinstance(count, int):
            if count < 0:
                raise IRError(f"internal error: negative zero padding {count} in a gradient rule")
            if count == 0:
                if first:
                    parts.append(v)
                continue
        z = e.zeros(shape[:dim] + [count] + shape[dim + 1:], v.type.dtype)
        parts.extend([z, v] if first else [z])
    return parts[0] if len(parts) == 1 else e.op("cat", *parts, dim=dim)


def scatter_strided(e, g: Value, dim: int, start: int, step: int, size: Dim, op: str) -> Value:
    """Adjoint of ``x[start::step]`` along ``dim``: zeros of length ``size`` with ``g[i]`` placed at
    ``start + i * step``. Built from reshape, cat with zeros and slice, so no scatter op is needed."""
    shape = list(g.type.shape)
    n = shape[dim]
    if step > 1:
        (n,) = _fixed(op, "size along a strided dim", [n])
        if n > 1:
            v = e.op("reshape", g, shape=shape[:dim] + [n, 1] + shape[dim + 1:])
            z = e.zeros(shape[:dim] + [n, step - 1] + shape[dim + 1:], g.type.dtype)
            v = e.op("cat", v, z, dim=dim + 1)
            v = e.op("reshape", v, shape=shape[:dim] + [n * step] + shape[dim + 1:])
            g = e.op("slice", v, dim=dim, start=0, end=(n - 1) * step + 1, step=1)
    length = g.type.shape[dim]
    return pad_zeros(e, g, dim, start, norm_dim(sympy.sympify(size) - start - sympy.sympify(length)))


# ----------------------------------------------------------------------------- elementwise


@grad_rule("add")
def _add(e, n, g):
    a, b = n.inputs
    return [lambda: e.sum_to(g, a.type), lambda: e.sum_to(g, b.type)]


@grad_rule("sub")
def _sub(e, n, g):
    a, b = n.inputs
    return [lambda: e.sum_to(g, a.type), lambda: e.sum_to(e.op("neg", g), b.type)]


@grad_rule("mul")
def _mul(e, n, g):
    a, b = n.inputs
    return [lambda: e.sum_to(e.op("mul", g, b), a.type), lambda: e.sum_to(e.op("mul", g, a), b.type)]


@grad_rule("div")
def _div(e, n, g):
    """y = a / b: dy/da = 1 / b, dy/db = -y / b."""
    a, b = n.inputs
    y = n.output
    return [lambda: e.sum_to(e.op("div", g, b), a.type),
            lambda: e.sum_to(e.op("neg", e.op("div", e.op("mul", g, y), b)), b.type)]


@grad_rule("neg")
def _neg(e, n, g):
    return [lambda: e.op("neg", g)]


@grad_rule("exp")
def _exp(e, n, g):
    return [lambda: e.op("mul", g, n.output)]


@grad_rule("expm1")
def _expm1(e, n, g):
    """y = exp(x) - 1: dy/dx = exp(x), read from the input (one exp in the fused backward kernel)
    rather than y + 1, which would round where y is tiny."""
    (x,) = n.inputs
    return [lambda: e.op("mul", g, e.op("exp", x))]


@grad_rule("sqrt")
def _sqrt(e, n, g):
    """y = sqrt(x): dy/dx = 1 / (2 y)."""
    return [lambda: e.scale(e.op("div", g, n.output), 0.5)]


@grad_rule("rsqrt")
def _rsqrt(e, n, g):
    """y = x^(-1/2): dy/dx = -y^3 / 2."""
    y = n.output
    return [lambda: e.op("mul", e.scale(g, -0.5), e.op("mul", y, e.op("mul", y, y)))]


@grad_rule("tanh")
def _tanh(e, n, g):
    """dy/dx = 1 - y^2."""
    y = n.output
    return [lambda: e.op("mul", g, e.op("sub", e.const(1.0, y), e.op("mul", y, y)))]


@grad_rule("sigmoid")
def _sigmoid(e, n, g):
    """dy/dx = y (1 - y)."""
    y = n.output
    return [lambda: e.op("mul", g, e.op("mul", y, e.op("sub", e.const(1.0, y), y)))]


@grad_rule("erf")
def _erf(e, n, g):
    """dy/dx = 2 / sqrt(pi) * exp(-x^2)."""
    (x,) = n.inputs
    return [lambda: e.op("mul", g, e.scale(e.op("exp", e.op("neg", e.op("mul", x, x))), 2.0 / math.sqrt(math.pi)))]


@grad_rule("relu")
def _relu(e, n, g):
    """1 for x > 0, else 0 (0 at the kink, as PyTorch). Read from the output: y > 0 exactly when
    x > 0 (NaN included: both comparisons are false), and the output is what the next layer's
    weight gradient keeps anyway, so the input is never stored for the backward pass and a
    compiler can leave it inside the producing kernel's epilogue. The zero is a scalar constant,
    not a zeros_like tensor: a compiled step then never writes and reads back a tensor of zeros of
    the activation's size (18 per resnet18 training step, 0.26 ms of writes alone on the RTX 5070 Ti)."""
    y = n.output
    return [lambda: e.op("where", e.op("gt", y, e.const(0.0, y)), g, e.const(0.0, g))]


@grad_rule("silu")
def _silu(e, n, g):
    """y = x s with s = sigmoid(x): dy/dx = s (1 + x (1 - s))."""
    (x,) = n.inputs

    def fn():
        s = e.op("sigmoid", x)
        one = e.const(1.0, x)
        return e.op("mul", g, e.op("mul", s, e.op("add", one, e.op("mul", x, e.op("sub", one, s)))))
    return [fn]


@grad_rule("hardsigmoid")
def _hardsigmoid(e, n, g):
    """y = clamp(x + 3, 0, 6) / 6: dy/dx = 1/6 for -3 < x < 3, else 0."""
    (x,) = n.inputs

    def fn():
        inside = e.op("logical_and", e.op("gt", x, e.const(-3.0, x)), e.op("lt", x, e.const(3.0, x)))
        return e.op("where", inside, e.scale(g, 1.0 / 6.0), e.zeros_like(g))
    return [fn]


@grad_rule("hardswish")
def _hardswish(e, n, g):
    """y = x clamp(x + 3, 0, 6) / 6: dy/dx = 0 for x < -3, x / 3 + 1 / 2 for -3 <= x <= 3, 1 above."""
    (x,) = n.inputs

    def fn():
        mid = e.op("mul", g, e.op("add", e.scale(x, 1.0 / 3.0), e.const(0.5, x)))
        upper = e.op("where", e.op("le", x, e.const(3.0, x)), mid, g)
        return e.op("where", e.op("lt", x, e.const(-3.0, x)), e.zeros_like(g), upper)
    return [fn]


@grad_rule("clamp")
def _clamp(e, n, g):
    """1 where min <= x <= max (bounds included, as torch.clamp before 2.14), else 0. With ``strict``
    (hardtanh, relu6 and torch 2.14 and later clamp, whose torch backward is 0 at the bounds) only
    min < x < max passes."""
    (x,) = n.inputs
    lo, hi = n.attrs.get("min"), n.attrs.get("max")
    strict = bool(n.attrs.get("strict", False))

    def fn():
        keep = None
        if lo is not None:
            keep = e.op("gt" if strict else "ge", x, e.const(float(lo), x))
        if hi is not None:
            below = e.op("lt" if strict else "le", x, e.const(float(hi), x))
            keep = below if keep is None else e.op("logical_and", keep, below)
        return g if keep is None else e.op("where", keep, g, e.zeros_like(g))
    return [fn]


@grad_rule("gelu")
def _gelu(e, n, g):
    """Exact: y = x Phi(x), dy/dx = Phi(x) + x phi(x), with Phi = (1 + erf(x / sqrt 2)) / 2 and
    phi = exp(-x^2 / 2) / sqrt(2 pi).

    Tanh form: y = x (1 + t) / 2 with t = tanh(u), u = c (x + k x^3), c = sqrt(2 / pi), k = 0.044715:
    dy/dx = (1 + t) / 2 + x (1 - t^2) c (1 + 3 k x^2) / 2.
    """
    (x,) = n.inputs

    def fn():
        one = e.const(1.0, x)
        xx = e.op("mul", x, x)
        if n.attrs.get("approximate", "none") == "tanh":
            c, k = math.sqrt(2.0 / math.pi), 0.044715
            u = e.scale(e.op("add", x, e.scale(e.op("mul", xx, x), k)), c)
            t = e.op("tanh", u)
            du = e.scale(e.op("add", one, e.scale(xx, 3.0 * k)), c)
            left = e.scale(e.op("add", one, t), 0.5)
            right = e.scale(e.op("mul", x, e.op("mul", e.op("sub", one, e.op("mul", t, t)), du)), 0.5)
            return e.op("mul", g, e.op("add", left, right))
        cdf = e.scale(e.op("add", one, e.op("erf", e.scale(x, 1.0 / math.sqrt(2.0)))), 0.5)
        pdf = e.scale(e.op("exp", e.scale(xx, -0.5)), 1.0 / math.sqrt(2.0 * math.pi))
        return e.op("mul", g, e.op("add", cdf, e.op("mul", x, pdf)))
    return [fn]


@grad_rule("where")
def _where(e, n, g):
    c, a, b = n.inputs
    return [None,
            lambda: e.sum_to(e.op("where", c, g, e.zeros_like(g)), a.type),
            lambda: e.sum_to(e.op("where", c, e.zeros_like(g), g), b.type)]


# ----------------------------------------------------------------------------- reductions


def _spread(e, n, g):
    """Cotangent of a reduction's output put back on the input shape (each reduced entry repeated)."""
    x = n.inputs[0]
    r = x.type.rank
    dims = n.attrs.get("dims")
    axes = list(range(r)) if dims is None or len(dims) == 0 else sorted({d % max(r, 1) for d in dims})
    shape = list(x.type.shape)
    if not n.attrs.get("keepdim", False) and r > 0:
        g = e.op("reshape", g, shape=[1 if i in axes else d for i, d in enumerate(shape)])
    if not _same_shape(g.type.shape, shape):
        g = e.op("expand", g, shape=shape)
    return g, axes


@grad_rule("sum")
def _sum(e, n, g):
    return [lambda: _spread(e, n, g)[0]]


@grad_rule("mean")
def _mean(e, n, g):
    """Each input entry gets g / count. A symbolic count (a mean over the batch dim) is computed in
    the program as the sum of a ones vector of that length, since the IR has no size op."""
    x = n.inputs[0]

    def fn():
        spread, axes = _spread(e, n, g)
        count = _prod([x.type.shape[i] for i in axes])
        if isinstance(count, int):
            return e.scale(spread, 1.0 / count)
        ones = e.op("full", shape=[count], value=1.0, dtype=x.type.dtype)
        return e.op("div", spread, e.op("sum", ones, dims=None, keepdim=False))
    return [fn]


@grad_rule("softmax")
def _softmax(e, n, g):
    """y = softmax(x): x_bar = y (g - sum(g y)) along the softmax dim."""
    (x,) = n.inputs
    y = n.output

    def fn():
        d = n.attrs["dim"] % max(x.type.rank, 1)
        inner = e.op("sum", e.op("mul", g, y), dims=[d], keepdim=True)
        return e.op("mul", y, e.op("sub", g, inner))
    return [fn]


# ----------------------------------------------------------------------------- linear algebra


@grad_rule("matmul")
def _matmul(e, n, g):
    """y = a b: a_bar = g b^T, b_bar = a^T g, each summed back over broadcast batch dims."""
    a, b = n.inputs
    return [lambda: e.sum_to(e.op("matmul", g, _swap_last(e, b)), a.type),
            lambda: e.sum_to(e.op("matmul", _swap_last(e, a), g), b.type)]


@grad_rule("linear")
def _linear(e, n, g):
    """y = x w^T + b: x_bar = g w, w_bar = sum over leading dims of g^T x, b_bar = sum of g."""
    x, w, b = n.inputs
    out_f, in_f = w.type.shape
    rank = g.type.rank

    def gx():
        if rank == 1:
            return e.op("reshape", e.op("matmul", e.op("reshape", g, shape=[1, out_f]), w), shape=[in_f])
        return e.op("matmul", g, w)

    def gw():
        rows = _prod(x.type.shape[:-1])
        g2 = g if rank == 2 else e.op("reshape", g, shape=[rows, out_f])
        x2 = x if rank == 2 else e.op("reshape", x, shape=[rows, in_f])
        return e.op("matmul", e.op("permute", g2, dims=[1, 0]), x2)

    def gb():
        return g if rank == 1 else e.op("sum", g, dims=list(range(rank - 1)), keepdim=False)

    return [gx, gw, gb]


def _conv_attrs(n: Node) -> Dict[str, Any]:
    return {k: n.attrs[k] for k in ("stride", "padding", "dilation", "groups") if k in n.attrs}


def conv_input_grad(e, g: Value, w: Value, attrs, input_size) -> Value:
    """The transposed convolution as one op. A compiled backward reads g at the strided indices
    (h + p - u d) / s, exact only; the zero inserted copy of g that the textbook form builds is never
    made (docs/COMPILER.md, convolution)."""
    return e.op("conv2d_input_grad", g, w, input_size=list(input_size), **attrs)


def conv_weight_grad(e, x: Value, g: Value, attrs, kernel_size) -> Value:
    return e.op("conv2d_weight_grad", x, g, kernel_size=list(kernel_size), **attrs)


@grad_rule("conv2d")
def _conv2d(e, n, g):
    x, w, _ = n.inputs
    attrs = _conv_attrs(n)

    def gx():
        hw = _fixed("conv2d", "spatial sizes of the input", x.type.shape[2:])
        return conv_input_grad(e, g, w, attrs, hw)

    def gw():
        _fixed("conv2d", "spatial sizes of the input", x.type.shape[2:])
        k = _fixed("conv2d", "kernel size", w.type.shape[2:])
        return conv_weight_grad(e, x, g, attrs, k)

    return [gx, gw, lambda: e.op("sum", g, dims=[0, 2, 3], keepdim=False)]


@grad_rule("conv2d_input_grad")
def _conv2d_input_grad(e, n, gg):
    """x_bar = A_w^T g is bilinear in (g, w): g_bar = A_w gg = conv2d(gg, w); w_bar is the weight
    gradient of <A_w gg, g>, conv2d_weight_grad(gg, g)."""
    g, w = n.inputs
    attrs = _conv_attrs(n)
    k = _fixed("conv2d", "kernel size", w.type.shape[2:])
    return [lambda: e.op("conv2d", gg, w, None, **attrs),
            lambda: conv_weight_grad(e, gg, g, attrs, k)]


@grad_rule("conv2d_weight_grad")
def _conv2d_weight_grad(e, n, ww):
    """w_bar = B(x, g) with <ww, B(x, g)> = <conv2d(x, ww), g>: x_bar = conv2d_input_grad(g, ww),
    g_bar = conv2d(x, ww)."""
    x, g = n.inputs
    attrs = _conv_attrs(n)
    hw = _fixed("conv2d", "spatial sizes of the input", x.type.shape[2:])
    return [lambda: conv_input_grad(e, g, ww, attrs, hw),
            lambda: e.op("conv2d", x, ww, None, **attrs)]


# ----------------------------------------------------------------------------- pooling


def _pool_attrs(n: Node) -> Dict[str, Any]:
    return {k: n.attrs[k] for k in ("kernel_size", "stride", "padding", "dilation", "ceil_mode") if k in n.attrs}


@grad_rule("max_pool2d")
def _max_pool2d(e, n, g):
    """Each output cell sends its cotangent to the input entry that was the first maximum of its
    window in row major order (PyTorch's tie rule), as one op: compiled, every input cell gathers
    from the windows that read it, with the same strided reads as the forward."""
    (x,) = n.inputs
    return [lambda: e.op("max_pool2d_grad", x, g, **_pool_attrs(n))]


@grad_rule("max_pool2d_grad")
def _max_pool2d_grad(e, n, xx):
    """max_pool2d_grad is linear in g: g_bar[window] = xx at the window's first maximum. x moves
    the selection only at ties, so x_bar is zero (as for relu at 0 and max_pool2d itself)."""
    x, g = n.inputs

    def fn():
        kh, kw = _pair(n.attrs["kernel_size"])
        sh, sw = _pair(n.attrs.get("stride") or n.attrs["kernel_size"])
        ph, pw = _pair(n.attrs.get("padding", 0))
        dh, dw = _pair(n.attrs.get("dilation", 1))
        h, w = _fixed("max_pool2d", "spatial sizes of the input", x.type.shape[2:])
        oh, ow = _fixed("max_pool2d", "output spatial sizes", g.type.shape[2:])
        y = e.op("max_pool2d", x, **_pool_attrs(n))
        big_h = max(h + 2 * ph, (oh - 1) * sh + dh * (kh - 1) + 1)
        big_w = max(w + 2 * pw, (ow - 1) * sw + dw * (kw - 1) + 1)

        def padded(v, value):
            shape = list(v.type.shape)
            for dim, before, after in ((2, ph, big_h - ph - h), (3, pw, big_w - pw - w)):
                parts = []
                if before:
                    parts.append(e.op("full", shape=shape[:dim] + [before] + shape[dim + 1:], value=value,
                                      dtype=v.type.dtype))
                parts.append(v)
                if after:
                    parts.append(e.op("full", shape=shape[:dim] + [after] + shape[dim + 1:], value=value,
                                      dtype=v.type.dtype))
                if len(parts) > 1:
                    v = e.op("cat", *parts, dim=dim)
                    shape = list(v.type.shape)
            return v

        xp, xxp = padded(x, -math.inf), padded(xx, 0.0)
        taken, total = None, None
        zeros = e.zeros_like(g)
        for u in range(kh):
            for v in range(kw):
                r0, c0 = u * dh, v * dw

                def tap(t):
                    t = e.op("slice", t, dim=2, start=r0, end=r0 + sh * (oh - 1) + 1, step=sh)
                    return e.op("slice", t, dim=3, start=c0, end=c0 + sw * (ow - 1) + 1, step=sw)

                hit = e.op("eq", tap(xp), y)
                first = hit if taken is None else e.op("logical_and", hit, e.op("logical_not", taken))
                taken = hit if taken is None else e.op("logical_or", taken, hit)
                part = e.op("where", first, tap(xxp), zeros)
                total = part if total is None else e.op("add", total, part)
        return total

    return [lambda: e.zeros_like(x), fn]


def _averaging_matrix(size: int, out: int) -> torch.Tensor:
    """A[i, r] = 1 / window length when input row r is inside output cell i's window, else 0."""
    a = torch.zeros((out, size), dtype=torch.float64)
    for i in range(out):
        r0, r1 = (i * size) // out, -(-((i + 1) * size) // out)
        a[i, r0:r1] = 1.0 / (r1 - r0)
    return a


@grad_rule("adaptive_avg_pool2d")
def _adaptive_avg_pool2d(e, n, g):
    """The pool is separable: y = A_h x A_w^T with constant averaging matrices, so x_bar = A_h^T g A_w."""
    (x,) = n.inputs

    def fn():
        h, w = _fixed("adaptive_avg_pool2d", "spatial sizes of the input", x.type.shape[-2:])
        oh, ow = _fixed("adaptive_avg_pool2d", "output sizes", g.type.shape[-2:])
        if oh == 1 and ow == 1:
            # global average pool: every input cell gets g / (h w), one correctly rounded division.
            # As two matmuls with 1 x h and 1 x w averaging matrices it was a contraction, and a
            # relu backward fused into it made a resnet18 training step's slowest kernel (4.1 ms).
            return e.op("expand", e.op("div", g, e.const(float(h * w), g)), shape=list(x.type.shape))
        left = e.tensor_const("pool_rows", _averaging_matrix(h, oh).transpose(0, 1).contiguous(), x.type.dtype)
        right = e.tensor_const("pool_cols", _averaging_matrix(w, ow), x.type.dtype)
        return e.op("matmul", e.op("matmul", left, g), right)
    return [fn]


# ----------------------------------------------------------------------------- shape ops


@grad_rule("reshape")
def _reshape(e, n, g):
    (x,) = n.inputs
    return [lambda: e.op("reshape", g, shape=list(x.type.shape))]


@grad_rule("permute")
def _permute(e, n, g):
    (x,) = n.inputs

    def fn():
        dims = [d % x.type.rank for d in n.attrs["dims"]]
        inverse = [0] * len(dims)
        for i, d in enumerate(dims):
            inverse[d] = i
        return e.op("permute", g, dims=inverse)
    return [fn]


@grad_rule("expand")
def _expand(e, n, g):
    (x,) = n.inputs
    return [lambda: e.sum_to(g, x.type)]


@grad_rule("select")
def _select(e, n, g):
    """x_bar is zero except at the selected index along dim, where it is g."""
    (x,) = n.inputs

    def fn():
        d = n.attrs["dim"] % x.type.rank
        size = x.type.shape[d]
        idx = n.attrs["index"]
        if isinstance(size, int):
            idx %= size
        shape = list(x.type.shape)
        v = e.op("reshape", g, shape=shape[:d] + [1] + shape[d + 1:])
        return pad_zeros(e, v, d, idx, norm_dim(sympy.sympify(size) - idx - 1))
    return [fn]


@grad_rule("slice")
def _slice(e, n, g):
    """x_bar is zero except at the sliced positions, which hold g."""
    (x,) = n.inputs

    def fn():
        d = n.attrs["dim"] % x.type.rank
        size = x.type.shape[d]
        start, end, step = n.attrs.get("start"), n.attrs.get("end"), n.attrs.get("step", 1)
        if isinstance(size, int):
            start, _, _ = slice(start, end, step).indices(size)
        else:
            start = start or 0
            if step != 1:
                raise UnsupportedOpError("the gradient of slice with a step over a symbolic dim is not supported yet")
        return scatter_strided(e, g, d, start, step, size, "slice")
    return [fn]


@grad_rule("cat")
def _cat(e, n, g):
    """Each operand gets its own block of g along the cat dim.

    With a fixed total size the block is a slice. With a symbolic total size the IR cannot slice to
    a fixed end, so a fixed size block is assembled from single index selects, and the last operand
    (whose size may be symbolic) is the slice from its start to the end. An operand that sits
    behind a symbolic sized one starts at a symbolic offset, which the IR cannot express yet.
    """
    d = n.attrs["dim"] % max(n.inputs[0].type.rank, 1)
    sizes = [t.type.shape[d] for t in n.inputs]

    def block(i):
        def fn():
            if len(sizes) == 1:
                return g
            last = i == len(sizes) - 1
            if not all(isinstance(s, int) for s in sizes[:i]) or not (isinstance(sizes[i], int) or last):
                raise UnsupportedOpError("the gradient of cat along a symbolic dim needs every operand to have a "
                                         "fixed size, except the last one (sizes along dim "
                                         f"{d}: [{', '.join(map(str, sizes))}])")
            start = sum(sizes[:i])
            if isinstance(g.type.shape[d], int):
                return e.op("slice", g, dim=d, start=start, end=start + sizes[i], step=1)
            if last:
                return e.op("slice", g, dim=d, start=start, end=None, step=1)
            shape = list(n.inputs[i].type.shape)
            one = shape[:d] + [1] + shape[d + 1:]
            rows = [e.op("reshape", e.op("select", g, dim=d, index=start + j), shape=one) for j in range(sizes[i])]
            if not rows:
                return e.zeros(shape, g.type.dtype)
            return rows[0] if len(rows) == 1 else e.op("cat", *rows, dim=d)
        return fn
    return [block(i) for i in range(len(sizes))]


# ----------------------------------------------------------------------------- lowerings


@lowering("batch_norm")
def _lower_batch_norm(e, n):
    """y = (x - running_mean) / sqrt(running_var + eps) * weight + bias, per channel (the reference formula)."""
    x, w, b, mean, var = n.inputs
    view = [1, x.type.shape[1]] + [1] * (x.type.rank - 2)

    def chan(v):
        return e.op("reshape", v, shape=view)

    centred = e.op("sub", x, chan(mean))
    std = e.op("sqrt", e.op("add", chan(var), e.const(n.attrs["eps"], var)))
    y = e.op("div", centred, std, out=n.output if w is None and b is None else None)
    if w is not None:
        y = e.op("mul", y, chan(w), out=n.output if b is None else None)
    if b is not None:
        e.op("add", y, chan(b), out=n.output)


@lowering("layer_norm")
def _lower_layer_norm(e, n):
    """y = (x - mean) / sqrt(biased variance + eps) * weight + bias over the trailing dims."""
    x, w, b = n.inputs
    r = x.type.rank
    dims = list(range(r - len(n.attrs["normalized_shape"]), r))
    if not dims:
        raise UnsupportedOpError("the gradient of layer_norm with an empty normalized_shape is not supported")
    mu = e.op("mean", x, dims=dims, keepdim=True)
    centred = e.op("sub", x, mu)
    var = e.op("mean", e.op("mul", centred, centred), dims=dims, keepdim=True)
    std = e.op("sqrt", e.op("add", var, e.const(n.attrs["eps"], x)))
    y = e.op("div", centred, std, out=n.output if w is None and b is None else None)
    if w is not None:
        y = e.op("mul", y, w, out=n.output if b is None else None)
    if b is not None:
        e.op("add", y, b, out=n.output)


@lowering("attention")
def _lower_attention(e, n):
    """softmax(q k^T * scale + mask) v, with scale = 1 / sqrt(head dim) unless given."""
    q, k, v, mask = n.inputs
    scale = n.attrs.get("scale")
    if scale is None:
        (head,) = _fixed("attention", "head dim", [q.type.shape[-1]])
        scale = 1.0 / math.sqrt(head)
    scores = e.scale(e.op("matmul", q, _swap_last(e, k)), scale)
    if n.attrs.get("is_causal", False):
        rows, cols = _fixed("attention", "sequence lengths for a causal mask", [q.type.shape[-2], k.type.shape[-2]])
        keep = e.tensor_const("causal_mask", torch.ones((rows, cols), dtype=torch.bool).tril(), torch.bool)
        scores = e.op("where", keep, scores, e.op("full_like", scores, value=-math.inf, dtype=None))
    if mask is not None:
        if mask.type.dtype == torch.bool:
            scores = e.op("where", mask, scores, e.op("full_like", scores, value=-math.inf, dtype=None))
        else:
            scores = e.op("add", scores, mask)
    probs = e.op("softmax", scores, dim=-1)
    e.op("matmul", probs, v, out=n.output)


# ----------------------------------------------------------------------------- decoder LLM ops (2026-11-24)
# Checked by finite differences in tests/test_autodiff.py like every other rule. sort and topk have
# rules below (scatter style); argmax, argmin and multinomial return integers (no gradient).


@grad_rule("flip")
def _flip_rule(e, n, g):
    """Flipping is its own inverse: x_bar = flip(g)."""
    return [lambda: e.op("flip", g, dims=list(n.attrs["dims"]))]


@grad_rule("amax", "amin")
def _amax_rule(e, n, g):
    """The cotangent goes to the entries equal to the result, split evenly among ties (PyTorch's
    convention for amax and amin)."""
    (x,) = n.inputs
    y = n.output

    def fn():
        dims = n.attrs.get("dims") or list(range(x.type.rank))
        dims = sorted(d % max(x.type.rank, 1) for d in dims)
        kept = [1 if i in dims else s for i, s in enumerate(x.type.shape)]
        yk = e.op("expand", e.op("reshape", y, shape=kept), shape=list(x.type.shape))
        gk = e.op("expand", e.op("reshape", g, shape=kept), shape=list(x.type.shape))
        hit = e.op("eq", x, yk)
        ones = e.op("where", hit, e.op("full_like", x, value=1.0, dtype=None), e.zeros_like(x))
        count = e.op("expand", e.op("sum", ones, dims=dims, keepdim=True), shape=list(x.type.shape))
        return e.op("where", hit, e.op("div", gk, count), e.zeros_like(x))
    return [fn]


@grad_rule("index_copy")
def _index_copy_rule(e, n, g):
    """out = x with rows ``index`` replaced by src: x_bar is g with those rows zeroed, src_bar is
    g's rows at ``index``. The index carries no gradient."""
    x, index, src = n.inputs
    d = n.attrs["dim"]
    return [lambda: e.op("index_copy", g, index, e.zeros_like(src), dim=d),
            None,
            lambda: e.op("index_select", g, index, dim=d)]


# ----------------------------------------------------------------------------- sort, topk and gathers
# The backward of an op that moves entries is a scatter of the cotangent back to where each entry
# came from: ``scatter_add`` (added with these rules). For sort and topk the positions are the
# indices output, which the stable sort fixes even with ties, so tied entries get the cotangent of
# their own position, as torch.sort(stable=True) does. A row never repeats an index there, so each
# entry receives at most one value and the result is exact.


def _scatter_back(e, x: Value, index: Value, g: Value, dim: int) -> Value:
    # a cotangent can arrive wider than x (float32 router scores times float64 expert outputs in a
    # float64 MoE model): the gradient of x is stored in x's dtype
    if g.type.dtype != x.type.dtype:
        g = e.op("cast", g, dtype=x.type.dtype)
    return e.op("scatter_add", e.zeros_like(x), index, g, dim=dim)


@grad_rule("sort", "topk")
def _sort_rule(e, n, g):
    """values = x[indices] along dim: x_bar = zeros with g added at indices."""
    (x,) = n.inputs
    indices = n.extra_outputs[0]
    if x.type.rank == 0:
        return [lambda: g]
    d = n.attrs.get("dim", -1) % x.type.rank
    return [lambda: _scatter_back(e, x, indices, g, d)]


@grad_rule("gather")
def _gather_rule(e, n, g):
    """out[p] = x[p with dim replaced by index[p]]: x_bar = zeros with g added at index (repeated
    indices add up). The index carries no gradient."""
    x, index = n.inputs
    d = n.attrs["dim"] % max(x.type.rank, 1)
    return [lambda: _scatter_back(e, x, index, g, d), None]


@grad_rule("index_select")
def _index_select_rule(e, n, g):
    """out = x rows ``index`` along dim (an embedding lookup is this with dim 0): x_bar = zeros with
    g's rows added at index; the index is broadcast to g's shape first."""
    x, index = n.inputs
    d = n.attrs["dim"] % max(x.type.rank, 1)

    def fn():
        view = [1] * x.type.rank
        view[d] = index.type.shape[0] if index.type.rank else 1
        full = e.op("expand", e.op("reshape", index, shape=view), shape=list(g.type.shape))
        return _scatter_back(e, x, full, g, d)
    return [fn, None]


@grad_rule("scatter_add")
def _scatter_add_rule(e, n, g):
    """out = x + scatter(src): x_bar = g, src_bar = g read back at index."""
    x, index, src = n.inputs
    d = n.attrs["dim"] % max(x.type.rank, 1)
    return [lambda: g, None, lambda: e.op("gather", g, index, dim=d)]


def rule_table() -> Dict[str, str]:
    """Every op the pass knows, with how it is handled: "rule", "lowered" or "no gradient"."""
    table = {op: "rule" for op in GRAD_RULES}
    table.update({op: "lowered" for op in LOWERINGS})
    table.update({op: "no gradient" for op in NONDIFF})
    return dict(sorted(table.items()))


from . import grad_rules_extra  # noqa: E402,F401  (rules added with the op coverage work)

from . import linattn_ops as _linattn_ops  # noqa: E402  (linear_attention, P60)
_linattn_ops.register_grad(grad_rule)

from . import attention_grad as _attention_grad  # noqa: E402  (attention_grad, the fused attention backward)
_attention_grad.register_grad(grad_rule)
_attention_grad.register_lowering(lowering)   # its definition, for forward mode and a second reverse pass
