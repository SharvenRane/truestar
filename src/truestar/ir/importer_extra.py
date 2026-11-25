"""ATen ops added with the op coverage work (2026-11-24, docs/MODEL_COVERAGE.md, scripts/op_coverage.py).

Every op here is lowered onto existing IR ops, so it gets the float64 reference, the reverse,
forward and batching rules and the compiler's kernels of those ops with no new rule. Each lowering
is written from the op's mathematical definition (PyTorch's documented semantics), not from its
source:

* 1D and 3D convolution and pooling: 1D is the 2D op over a [.., 1, L] view; 3D convolution is a
  sum over the depth taps of 2D convolutions with the depth folded into the batch; max and average
  3D pooling are separable (the max or mean over a box is that over depth of that over the plane).
* Transposed convolution is ``conv2d_input_grad``, the IR's adjoint of ``conv2d`` (the same
  arithmetic, with the output size from ``output_padding``); 3D is placed tap by tap along depth.
* Interpolation (nearest, linear, bilinear, bicubic) is separable: nearest is an ``index_select``
  per axis, the others ``A_h x A_w^T`` with constant interpolation matrices built from the source
  index formula (half pixel or align corners) and, for bicubic, the cubic convolution weights with
  a = -0.75 and clamped border taps.
* Reflect, replicate and circular padding are an ``index_select`` per axis with constant indices.
* Recurrent layers (RNN, LSTM, GRU and their cells) are unrolled over a fixed sequence length.
* Activations, losses and small layers are compositions; ``log1p`` is a composite accurate near
  zero (a series on a small interval, the direct formula outside), as ``ops.expm1``. ``aten.expm1``
  imports as the ``expm1`` IR op (importer.py: its own reference, gradient and kernel); the
  ``_expm1`` composite remains inside the elu family of activations.

An op or setting that cannot be lowered exactly raises with the reason; nothing is approximated.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import sympy
import torch

from .core import TensorType, UnsupportedOpError, Value, norm_dim
from .importer import HANDLERS, _describe, _Missing, aten

# ----------------------------------------------------------------------------- helpers


def _unsupported(c, why: str, code: str = "TS860"):
    """Refuse with a stable code: TS860 a setting this lowering cannot express exactly, TS861 a size
    that must be fixed (``truestar codes``)."""
    err = UnsupportedOpError(f"[{code}] {_describe(c.node)}: {why}")
    err.code = code
    raise err


def _t(c, base: str, data: torch.Tensor, dtype: torch.dtype) -> Value:
    """A constant tensor as a param of declared ``dtype``: computed in float64 and stored rounded once
    to ``dtype`` (a float scalar stays float64, exact; core.declared_data)."""
    from .core import declared_data
    name = c.fresh(base)
    v = Value(name, TensorType(dtype, tuple(data.shape)), "param")
    c.params.append(v)
    c.param_data[name] = declared_data(data, dtype)
    return v


def _f(c, value: float, like: Value) -> Value:
    """A float scalar with the dtype of ``like`` (a 0 dim operand never widens)."""
    return c.const(float(value), like)


def _op(c, op: str, *inputs, **attrs) -> Value:
    return c.emit(op, list(inputs), **attrs)


class _E:
    """The emitter interface of the gradient rules (``op``, ``zeros``) over the importer context, so
    the helpers of grad_rules.py (strided placement) can be reused."""

    def __init__(self, c):
        self.c = c

    def op(self, op, *inputs, **attrs):
        return self.c.emit(op, list(inputs), **attrs)

    def zeros(self, shape, dtype):
        return self.c.emit("full", [], shape=list(shape), value=0.0, dtype=dtype)


def _fixed(c, what: str, dims) -> List[int]:
    if not all(isinstance(d, int) for d in dims):
        _unsupported(c, f"{what} must be fixed, got [{', '.join(map(str, dims))}]", "TS861")
    return list(dims)


def _ints(v, n: int) -> List[int]:
    if isinstance(v, int):
        return [v] * n
    v = list(v)
    return v * n if len(v) == 1 else v


def _reshape(c, x: Value, shape) -> Value:
    return _op(c, "reshape", x, shape=list(shape))


def _expm1(c, x: Value) -> Value:
    """exp(x) - 1: the Taylor series on [-1/2, 1/2] (nested, 17 terms, below 2^-60 relative), exp(x) - 1
    outside, where the subtraction loses at most a bit. The series reads x clamped to the interval,
    so neither branch overflows and the unselected one gives a finite zero to a gradient."""
    cl = _op(c, "clamp", x, min=-0.5, max=0.5)
    acc = None
    for n in range(17, 1, -1):
        term = _op(c, "mul", cl, _f(c, 1.0 / n, x))
        if acc is not None:
            term = _op(c, "mul", term, acc)
        acc = _op(c, "add", term, _f(c, 1.0, x))
    series = _op(c, "mul", cl, acc)
    direct = _op(c, "sub", _op(c, "exp", x), _f(c, 1.0, x))
    inside = _op(c, "logical_and", _op(c, "ge", x, _f(c, -0.5, x)), _op(c, "le", x, _f(c, 0.5, x)))
    return _op(c, "where", inside, series, direct)


def _log1p(c, x: Value) -> Value:
    """log(1 + x) for x > -1: on [-1/2, 1/2] the series 2 atanh(z), z = x / (2 + x), |z| <= 1/3 (19 odd
    terms, below 1e-17 relative); log(1 + x) outside, where 1 + x is exact or far from 1."""
    cl = _op(c, "clamp", x, min=-0.5, max=0.5)
    z = _op(c, "div", cl, _op(c, "add", cl, _f(c, 2.0, x)))
    z2 = _op(c, "mul", z, z)
    acc = None
    for k in range(37, 1, -2):
        term = _f(c, 1.0 / k, x)
        if acc is not None:
            term = _op(c, "add", term, _op(c, "mul", z2, acc))
        acc = term
    series = _op(c, "mul", _op(c, "mul", z, _f(c, 2.0, x)), _op(c, "add", _f(c, 1.0, x), _op(c, "mul", z2, acc)))
    inside = _op(c, "logical_and", _op(c, "ge", x, _f(c, -0.5, x)), _op(c, "le", x, _f(c, 0.5, x)))
    # the direct branch reads 1 inside the series interval, so it is finite there and its gradient
    # (through the where) is zero; outside it is x itself (log(0) = -inf at x = -1, as torch)
    outer = _op(c, "where", inside, _op(c, "full_like", x, value=1.0, dtype=None), x)
    direct = _op(c, "log", _op(c, "add", outer, _f(c, 1.0, x)))
    return _op(c, "where", inside, series, direct)


def _softplus(c, z: Value) -> Value:
    """log(1 + exp(z)) = max(z, 0) + log1p(exp(-|z|)), finite for every z."""
    return _op(c, "add", _op(c, "relu", z), _log1p(c, _op(c, "exp", _op(c, "neg", _op(c, "abs", z)))))


def _reduce(c, x: Value, reduction: int) -> Value:
    """ATen loss reductions: 0 none, 1 mean, 2 sum."""
    if reduction == 0:
        return x
    op = "mean" if reduction == 1 else "sum"
    return _op(c, op, x, dims=None, keepdim=False)


# ----------------------------------------------------------------------------- convolution


def _conv_nd(c, x, w, b, stride, padding, dilation, groups, nd):
    if isinstance(padding, str):    # "valid" or "same" (coverage refresh, importer_current.py)
        from .importer_current import resolve_string_padding
        x, padding = resolve_string_padding(c, x, w, padding, stride, dilation, nd)
    if nd == 2:
        return _op(c, "conv2d", x, w, b, stride=_ints(stride, 2), padding=_ints(padding, 2),
                   dilation=_ints(dilation, 2), groups=groups)
    if nd == 1:
        return _conv1d(c, x, w, b, stride, padding, dilation, groups)
    if nd == 3:
        return _conv3d(c, x, w, b, stride, padding, dilation, groups)
    _unsupported(c, f"{nd}D convolution is not supported")


def _unbatched(c, x: Value, rank: int):
    """A missing batch dim (torch accepts [C, ...] for conv and pooling): add it, and say so."""
    if x.type.rank == rank - 1:
        return _reshape(c, x, [1] + list(x.type.shape)), True
    return x, False


def _conv1d(c, x, w, b, stride, padding, dilation, groups):
    x, squeeze = _unbatched(c, x, 3)
    n, ch, length = x.type.shape
    x4 = _reshape(c, x, [n, ch, 1, length])
    w4 = _reshape(c, w, [w.type.shape[0], w.type.shape[1], 1, w.type.shape[2]])
    y = _op(c, "conv2d", x4, w4, b, stride=[1, _ints(stride, 1)[0]], padding=[0, _ints(padding, 1)[0]],
            dilation=[1, _ints(dilation, 1)[0]], groups=groups)
    s = y.type.shape
    y = _reshape(c, y, [s[0], s[1], s[3]])
    return _reshape(c, y, list(y.type.shape[1:])) if squeeze else y


def _conv3d(c, x, w, b, stride, padding, dilation, groups):
    """Sum over the depth taps u of conv2d(x at depths od * s + u * d - p, w[:, :, u]), depth in the batch."""
    x, squeeze = _unbatched(c, x, 5)
    sd, sh, sw = _ints(stride, 3)
    pd, ph, pw = _ints(padding, 3)
    dd, dh, dw = _ints(dilation, 3)
    n, ch, depth, h, wd = x.type.shape
    (depth,) = _fixed(c, "the depth of a 3D convolution", [depth])
    o, cg, kd, kh, kw = w.type.shape
    if pd:
        x = _op(c, "pad", x, pads=[0, 0, 0, 0, pd, pd], value=0.0)
    od = (depth + 2 * pd - dd * (kd - 1) - 1) // sd + 1
    total = None
    for u in range(kd):
        xs = _op(c, "slice", x, dim=2, start=u * dd, end=u * dd + sd * (od - 1) + 1, step=sd)
        xs = _op(c, "permute", xs, dims=[0, 2, 1, 3, 4])
        xs = _reshape(c, xs, [norm_dim(sympy.sympify(n) * od), ch, h, wd])
        wu = _op(c, "select", w, dim=2, index=u)
        y = _op(c, "conv2d", xs, wu, None, stride=[sh, sw], padding=[ph, pw], dilation=[dh, dw], groups=groups)
        total = y if total is None else _op(c, "add", total, y)
    oh, ow = total.type.shape[2:]
    y = _op(c, "permute", _reshape(c, total, [n, od, o, oh, ow]), dims=[0, 2, 1, 3, 4])
    if b is not None:
        y = _op(c, "add", y, _reshape(c, b, [o, 1, 1, 1]))
    return _reshape(c, y, list(y.type.shape[1:])) if squeeze else y


def _out_len(i, k, s, p, d, op):
    return (i - 1) * s - 2 * p + d * (k - 1) + op + 1


def _conv_transpose2d(c, x, w, b, stride, padding, output_padding, groups, dilation):
    """The adjoint of conv2d (weight [C_in, C_out / groups, kh, kw]) is conv2d_input_grad with the
    output size (H - 1) s - 2 p + d (k - 1) + output_padding + 1."""
    x, squeeze = _unbatched(c, x, 4)
    s, p, op_, d = _ints(stride, 2), _ints(padding, 2), _ints(output_padding, 2), _ints(dilation, 2)
    h, wd = _fixed(c, "the spatial sizes of a transposed convolution", x.type.shape[2:])
    kh, kw = w.type.shape[2:]
    size = [_out_len(h, kh, s[0], p[0], d[0], op_[0]), _out_len(wd, kw, s[1], p[1], d[1], op_[1])]
    for i in range(2):
        if op_[i] >= s[i]:
            _unsupported(c, f"output_padding {op_[i]} not smaller than the stride {s[i]} is not supported")
    y = _op(c, "conv2d_input_grad", x, w, stride=s, padding=p, dilation=d, groups=groups, input_size=size)
    if b is not None:
        y = _op(c, "add", y, _reshape(c, b, [b.type.shape[0], 1, 1]))
    return _reshape(c, y, list(y.type.shape[1:])) if squeeze else y


def _conv_transpose1d(c, x, w, b, stride, padding, output_padding, groups, dilation):
    x, squeeze = _unbatched(c, x, 3)
    n, ch, length = x.type.shape
    x4 = _reshape(c, x, [n, ch, 1, length])
    w4 = _reshape(c, w, [w.type.shape[0], w.type.shape[1], 1, w.type.shape[2]])
    y = _conv_transpose2d(c, x4, w4, b, [1, _ints(stride, 1)[0]], [0, _ints(padding, 1)[0]],
                          [0, _ints(output_padding, 1)[0]], groups, [1, _ints(dilation, 1)[0]])
    sh = y.type.shape
    y = _reshape(c, y, [sh[0], sh[1], sh[3]])
    return _reshape(c, y, list(y.type.shape[1:])) if squeeze else y


def _conv_transpose3d(c, x, w, b, stride, padding, output_padding, groups, dilation):
    """Each depth tap u: the 2D transposed convolution of every input depth plane with w[:, :, u],
    placed at depth i * s + u * d of an uncropped output, which is then cropped by the padding."""
    from .grad_rules import scatter_strided
    x, squeeze = _unbatched(c, x, 5)
    sd, sh, sw = _ints(stride, 3)
    pd, ph, pw = _ints(padding, 3)
    od_, oh_, ow_ = _ints(output_padding, 3)
    dd, dh, dw = _ints(dilation, 3)
    n, ch, depth, h, wd = x.type.shape
    (depth,) = _fixed(c, "the depth of a 3D transposed convolution", [depth])
    cin, cg, kd, kh, kw = w.type.shape
    full = (depth - 1) * sd + dd * (kd - 1) + 1 + od_
    xs = _reshape(c, _op(c, "permute", x, dims=[0, 2, 1, 3, 4]), [norm_dim(sympy.sympify(n) * depth), ch, h, wd])
    e = _E(c)
    total = None
    for u in range(kd):
        wu = _op(c, "select", w, dim=2, index=u)
        y = _conv_transpose2d(c, xs, wu, None, [sh, sw], [ph, pw], [oh_, ow_], groups, [dh, dw])
        oc, oh, ow = y.type.shape[1:]
        y = _reshape(c, y, [n, depth, oc, oh, ow])
        y = scatter_strided(e, y, 1, u * dd, sd, full, "conv_transpose3d")
        total = y if total is None else _op(c, "add", total, y)
    out_d = full - 2 * pd
    y = _op(c, "slice", total, dim=1, start=pd, end=pd + out_d, step=1)
    y = _op(c, "permute", y, dims=[0, 2, 1, 3, 4])
    if b is not None:
        y = _op(c, "add", y, _reshape(c, b, [b.type.shape[0], 1, 1, 1]))
    return _reshape(c, y, list(y.type.shape[1:])) if squeeze else y


@aten("aten.conv1d.default", "aten.conv1d.padding")
def _aten_conv1d(c, a):
    return _conv_nd(c, c.tensor(a["input"]), c.tensor(a["weight"]), c.tensor(a["bias"]), a["stride"], a["padding"],
                    a["dilation"], a["groups"], 1)


@aten("aten.conv3d.default", "aten.conv3d.padding")
def _aten_conv3d(c, a):
    return _conv_nd(c, c.tensor(a["input"]), c.tensor(a["weight"]), c.tensor(a["bias"]), a["stride"], a["padding"],
                    a["dilation"], a["groups"], 3)


@aten("aten.conv_transpose1d.default")
def _aten_conv_transpose1d(c, a):
    return _conv_transpose1d(c, c.tensor(a["input"]), c.tensor(a["weight"]), c.tensor(a["bias"]), a["stride"],
                             a["padding"], a["output_padding"], a["groups"], a["dilation"])


@aten("aten.conv_transpose2d.input")
def _aten_conv_transpose2d(c, a):
    return _conv_transpose2d(c, c.tensor(a["input"]), c.tensor(a["weight"]), c.tensor(a["bias"]), a["stride"],
                             a["padding"], a["output_padding"], a["groups"], a["dilation"])


@aten("aten.conv_transpose3d.input")
def _aten_conv_transpose3d(c, a):
    return _conv_transpose3d(c, c.tensor(a["input"]), c.tensor(a["weight"]), c.tensor(a["bias"]), a["stride"],
                             a["padding"], a["output_padding"], a["groups"], a["dilation"])


def _convolution(c, a):
    """aten.convolution (the core ATen graph's form of every convolution)."""
    x, w, b = c.tensor(a["input"]), c.tensor(a["weight"]), c.tensor(a["bias"])
    nd = len(a["stride"])
    if a["transposed"]:
        fn = {1: _conv_transpose1d, 2: _conv_transpose2d, 3: _conv_transpose3d}.get(nd)
        if fn is None:
            _unsupported(c, f"{nd}D transposed convolution is not supported")
        return fn(c, x, w, b, a["stride"], a["padding"], a["output_padding"], a["groups"], a["dilation"])
    return _conv_nd(c, x, w, b, a["stride"], a["padding"], a["dilation"], a["groups"], nd)


HANDLERS["aten.convolution.default"] = _convolution


# ----------------------------------------------------------------------------- pooling


def _as_2d(c, x: Value):
    """[.., L] -> [.., 1, L] for a 1D pool through the 2D op."""
    return _reshape(c, x, list(x.type.shape[:-1]) + [1, x.type.shape[-1]])


def _from_2d(c, y: Value):
    return _reshape(c, y, list(y.type.shape[:-2]) + [y.type.shape[-1]])


@aten("aten.max_pool1d.default", "aten.max_pool1d_with_indices.default")
def _max_pool1d(c, a):
    x = c.tensor(a["self"])
    k = _ints(a["kernel_size"], 1)[0]
    s = _ints(a["stride"] or [k], 1)[0]
    y = _op(c, "max_pool2d", _as_2d(c, x), kernel_size=[1, k], stride=[1, s], padding=[0, _ints(a["padding"], 1)[0]],
            dilation=[1, _ints(a["dilation"], 1)[0]], ceil_mode=bool(a["ceil_mode"]))
    y = _from_2d(c, y)
    if "with_indices" in str(c.node.target):
        return (y, _Missing("the indices of max_pool1d"))
    return y


@aten("aten.avg_pool1d.default")
def _avg_pool1d(c, a):
    x = c.tensor(a["self"])
    k = _ints(a["kernel_size"], 1)[0]
    s = _ints(a["stride"] or [k], 1)[0]
    y = _op(c, "avg_pool2d", _as_2d(c, x), kernel_size=[1, k], stride=[1, s], padding=[0, _ints(a["padding"], 1)[0]],
            ceil_mode=bool(a["ceil_mode"]), count_include_pad=bool(a["count_include_pad"]), divisor_override=None)
    return _from_2d(c, y)


@aten("aten.adaptive_max_pool1d.default")
def _adaptive_max_pool1d(c, a):
    x = c.tensor(a["self"])
    y = _op(c, "adaptive_max_pool2d", _as_2d(c, x), output_size=[1, _ints(a["output_size"], 1)[0]])
    return (_from_2d(c, y), _Missing("the indices of adaptive_max_pool1d"))


def _depth_view(c, x: Value):
    """[N, C, D, H, W] -> [N, C * D, H, W] (or without N) so a 2D pool works on every depth plane."""
    s = list(x.type.shape)
    lead = s[:-4]
    return _reshape(c, x, lead + [norm_dim(sympy.sympify(s[-4]) * s[-3]), s[-2], s[-1]]), lead, s[-4], s[-3]


def _pool3d(c, x: Value, plane, along_depth):
    """A separable 3D pool: ``plane`` on every depth slice, then ``along_depth`` over [.., C, D, H' W']."""
    v, lead, ch, depth = _depth_view(c, x)
    y = plane(v)
    oh, ow = y.type.shape[-2:]
    y = _reshape(c, y, lead + [ch, depth, norm_dim(sympy.sympify(oh) * ow)])
    y = along_depth(y)
    od = y.type.shape[-2]
    return _reshape(c, y, lead + [ch, od, oh, ow])


@aten("aten.max_pool3d.default", "aten.max_pool3d_with_indices.default")
def _max_pool3d(c, a):
    x = c.tensor(a["self"])
    k, s = _ints(a["kernel_size"], 3), _ints(a["stride"] or a["kernel_size"], 3)
    p, d, ceil = _ints(a["padding"], 3), _ints(a["dilation"], 3), bool(a["ceil_mode"])
    y = _pool3d(c, x,
                lambda v: _op(c, "max_pool2d", v, kernel_size=k[1:], stride=s[1:], padding=p[1:], dilation=d[1:],
                              ceil_mode=ceil),
                lambda v: _op(c, "max_pool2d", v, kernel_size=[k[0], 1], stride=[s[0], 1], padding=[p[0], 0],
                              dilation=[d[0], 1], ceil_mode=ceil))
    if "with_indices" in str(c.node.target):
        return (y, _Missing("the indices of max_pool3d"))
    return y


@aten("aten.avg_pool3d.default")
def _avg_pool3d(c, a):
    """The window size of a box is the product of its sizes per axis, so the mean is separable."""
    if a.get("divisor_override"):
        _unsupported(c, "avg_pool3d with divisor_override is not supported yet")
    x = c.tensor(a["self"])
    k, s = _ints(a["kernel_size"], 3), _ints(a["stride"] or a["kernel_size"], 3)
    p, ceil, inc = _ints(a["padding"], 3), bool(a["ceil_mode"]), bool(a["count_include_pad"])
    return _pool3d(c, x,
                   lambda v: _op(c, "avg_pool2d", v, kernel_size=k[1:], stride=s[1:], padding=p[1:], ceil_mode=ceil,
                                 count_include_pad=inc, divisor_override=None),
                   lambda v: _op(c, "avg_pool2d", v, kernel_size=[k[0], 1], stride=[s[0], 1], padding=[p[0], 0],
                                 ceil_mode=ceil, count_include_pad=inc, divisor_override=None))


@aten("aten.adaptive_avg_pool3d.default", "aten._adaptive_avg_pool3d.default")
def _adaptive_avg_pool3d(c, a):
    x = c.tensor(a["self"])
    o = list(a["output_size"])
    o = [x.type.shape[-3 + i] if v is None else v for i, v in enumerate(_ints(o, 3))]
    return _pool3d(c, x,
                   lambda v: _op(c, "adaptive_avg_pool2d", v, output_size=o[1:]),
                   lambda v: _op(c, "adaptive_avg_pool2d", v, output_size=[o[0], v.type.shape[-1]]))


# ----------------------------------------------------------------------------- interpolation


def _scale(in_size: int, out_size: int, factor: Optional[float], align: bool) -> float:
    if align:
        return (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
    return 1.0 / factor if factor else in_size / out_size


def _source(dst: int, scale: float, align: bool) -> float:
    return dst * scale if align else (dst + 0.5) * scale - 0.5


def _nearest_index(in_size: int, out_size: int, factor: Optional[float]) -> torch.Tensor:
    scale = 1.0 / factor if factor else in_size / out_size
    return torch.tensor([min(int(math.floor(i * scale)), in_size - 1) for i in range(out_size)], dtype=torch.int64)


def _linear_matrix(in_size: int, out_size: int, factor: Optional[float], align: bool) -> torch.Tensor:
    m = torch.zeros((out_size, in_size), dtype=torch.float64)
    scale = _scale(in_size, out_size, factor, align)
    for i in range(out_size):
        src = max(_source(i, scale, align), 0.0)
        i0 = min(int(math.floor(src)), in_size - 1)
        i1 = min(i0 + 1, in_size - 1)
        lam = src - i0
        m[i, i0] += 1.0 - lam
        m[i, i1] += lam
    return m


def _cubic_matrix(in_size: int, out_size: int, factor: Optional[float], align: bool) -> torch.Tensor:
    """Cubic convolution (a = -0.75) with the four taps clamped to the border."""
    A = -0.75

    def near(t):   # |t| <= 1
        return ((A + 2) * t - (A + 3)) * t * t + 1

    def far(t):    # 1 < |t| < 2
        return ((A * t - 5 * A) * t + 8 * A) * t - 4 * A

    m = torch.zeros((out_size, in_size), dtype=torch.float64)
    scale = _scale(in_size, out_size, factor, align)
    for i in range(out_size):
        src = _source(i, scale, align)
        x0 = int(math.floor(src))
        t = src - x0
        ws = (far(t + 1), near(t), near(1 - t), far(2 - t))
        for j, wgt in enumerate(ws):
            m[i, min(max(x0 - 1 + j, 0), in_size - 1)] += wgt
    return m


def _interp(c, x: Value, out_sizes: Sequence[int], factors, mode: str, align: bool) -> Value:
    n = len(out_sizes)
    ins = _fixed(c, "the interpolated sizes", x.type.shape[-n:])
    factors = list(factors) if factors else [None] * n
    y = x
    r = x.type.rank
    for i in range(n):
        dim = r - n + i
        if ins[i] == out_sizes[i] and (mode == "nearest" or not factors[i] or factors[i] == 1.0):
            continue
        if mode == "nearest":
            idx = _t(c, "nearest_index", _nearest_index(ins[i], out_sizes[i], factors[i]), torch.int64)
            y = _op(c, "index_select", y, idx, dim=dim)
            continue
        if mode == "linear_aa":   # antialiased (importer_vision.py)
            from .importer_vision import aa_matrix
            mat = aa_matrix(ins[i], out_sizes[i], factors[i])
        else:
            build = _cubic_matrix if mode == "cubic" else _linear_matrix
            mat = build(ins[i], out_sizes[i], factors[i], align)
        if dim == r - 1:
            y = _op(c, "matmul", y, _t(c, "interp_cols", mat.transpose(0, 1).contiguous(), x.type.dtype))
        else:   # move the axis last, contract, move it back
            perm = list(range(r))
            perm[dim], perm[-1] = perm[-1], perm[dim]
            yt = _op(c, "permute", y, dims=perm)
            yt = _op(c, "matmul", yt, _t(c, "interp_rows", mat.transpose(0, 1).contiguous(), x.type.dtype))
            y = _op(c, "permute", yt, dims=perm)
    return y


def _out_sizes(c, x: Value, a, n: int) -> List[int]:
    fake = c.node.meta.get("val")
    if isinstance(fake, torch.Tensor):
        return _fixed(c, "the interpolated output sizes", [c.ir_dim(d) for d in fake.shape[-n:]])
    return list(a["output_size"])


def _interp_handler(mode: str, n: int, has_align: bool):
    def fn(c, a):
        x = c.tensor(a["input"] if "input" in a else a["self"])
        factors = a.get("scale_factors")
        if factors is None and any(k in a for k in ("scales", "scales_h", "scales_d")):
            keys = {1: ["scales"], 2: ["scales_h", "scales_w"], 3: ["scales_d", "scales_h", "scales_w"]}[n]
            factors = [a.get(k) for k in keys]
        align = bool(a.get("align_corners")) if has_align else False
        return _interp(c, x, _out_sizes(c, x, a, n), factors, mode, align)
    return fn


for _n, _dims in ((1, "1d"), (2, "2d"), (3, "3d")):
    for _ov in ("vec", "default"):
        HANDLERS[f"aten.upsample_nearest{_dims}.{_ov}"] = _interp_handler("nearest", _n, False)
for _ov in ("vec", "default"):
    HANDLERS[f"aten.upsample_linear1d.{_ov}"] = _interp_handler("linear", 1, True)
    HANDLERS[f"aten.upsample_bilinear2d.{_ov}"] = _interp_handler("linear", 2, True)
    HANDLERS[f"aten.upsample_trilinear3d.{_ov}"] = _interp_handler("linear", 3, True)
    HANDLERS[f"aten.upsample_bicubic2d.{_ov}"] = _interp_handler("cubic", 2, True)


# ----------------------------------------------------------------------------- rearrangement


@aten("aten.pixel_shuffle.default")
def _pixel_shuffle(c, a):
    """[.., C r^2, H, W] -> [.., C, H r, W r]: out[c, h r + i, w r + j] = x[c r^2 + i r + j, h, w]."""
    x = c.tensor(a["self"])
    r = a["upscale_factor"]
    s = list(x.type.shape)
    lead, ch, h, w = s[:-3], s[-3], s[-2], s[-1]
    (ch,) = _fixed(c, "the channels of pixel_shuffle", [ch])
    k = len(lead)
    v = _reshape(c, x, lead + [ch // (r * r), r, r, h, w])
    v = _op(c, "permute", v, dims=list(range(k)) + [k, k + 3, k + 1, k + 4, k + 2])
    return _reshape(c, v, lead + [ch // (r * r), norm_dim(sympy.sympify(h) * r), norm_dim(sympy.sympify(w) * r)])


@aten("aten.pixel_unshuffle.default")
def _pixel_unshuffle(c, a):
    x = c.tensor(a["self"])
    r = a["downscale_factor"]
    s = list(x.type.shape)
    lead, ch, h, w = s[:-3], s[-3], s[-2], s[-1]
    h, w = _fixed(c, "the spatial sizes of pixel_unshuffle", [h, w])
    k = len(lead)
    v = _reshape(c, x, lead + [ch, h // r, r, w // r, r])
    v = _op(c, "permute", v, dims=list(range(k)) + [k, k + 2, k + 4, k + 1, k + 3])
    return _reshape(c, v, lead + [norm_dim(sympy.sympify(ch) * r * r), h // r, w // r])


@aten("aten.channel_shuffle.default", "aten.native_channel_shuffle.default")
def _channel_shuffle(c, a):
    x = c.tensor(a["self"])
    g = a["groups"]
    s = list(x.type.shape)
    (ch,) = _fixed(c, "the channels of channel_shuffle", [s[1]])
    v = _reshape(c, x, [s[0], g, ch // g] + s[2:])
    v = _op(c, "permute", v, dims=[0, 2, 1] + list(range(3, len(s) + 1)))
    return _reshape(c, v, s)


# ----------------------------------------------------------------------------- padding


def _pad_index(n: int, lo: int, hi: int, mode: str) -> torch.Tensor:
    out = []
    for i in range(-lo, n + hi):
        if mode == "reflect":
            j = -i if i < 0 else 2 * (n - 1) - i if i >= n else i
        elif mode == "replicate":
            j = min(max(i, 0), n - 1)
        else:   # circular
            j = i % n
        out.append(j)
    return torch.tensor(out, dtype=torch.int64)


def _index_pad(c, x: Value, pads: Sequence[int], mode: str) -> Value:
    y = x
    for i in range(len(pads) // 2):
        d = x.type.rank - 1 - i
        lo, hi = int(pads[2 * i]), int(pads[2 * i + 1])
        if lo == 0 and hi == 0:
            continue
        (n,) = _fixed(c, f"the padded dims of {mode} padding", [x.type.shape[d]])
        if mode == "reflect" and (lo >= n or hi >= n):
            _unsupported(c, f"reflect padding {lo}, {hi} must be smaller than the dim size {n}")
        if mode == "circular" and (lo > n or hi > n or lo < 0 or hi < 0):
            _unsupported(c, f"circular padding {lo}, {hi} must be between 0 and the dim size {n}")
        idx = _t(c, f"{mode}_pad_index", _pad_index(n, lo, hi, mode), torch.int64)
        y = _op(c, "index_select", y, idx, dim=d)
    return y


def _pad_any(c, a):
    mode = a.get("mode", "constant")
    x = c.tensor(a["self"])
    if mode == "constant":
        v = a.get("value")
        return _op(c, "pad", x, pads=c.dims(a["pad"]), value=0.0 if v is None else v)
    if mode not in ("reflect", "replicate", "circular"):
        _unsupported(c, f"pad mode {mode!r} is not supported")
    return _index_pad(c, x, c.dims(a["pad"]), mode)


HANDLERS["aten.pad.default"] = _pad_any
for _mode in ("reflection", "replication"):
    for _d in ("1d", "2d", "3d"):
        HANDLERS[f"aten.{_mode}_pad{_d}.default"] = (
            lambda c, a, m=("reflect" if _mode == "reflection" else "replicate"):
            _index_pad(c, c.tensor(a["self"]), c.dims(a["padding"]), m))
HANDLERS["aten._pad_circular.default"] = lambda c, a: _index_pad(c, c.tensor(a["self"]), c.dims(a["pad"]), "circular")


# ----------------------------------------------------------------------------- activations


def _act(fn):
    def handler(c, a):
        return fn(c, c.tensor(a["self"]), a)
    return handler


def _elu_like(c, x, alpha, scale, input_scale):
    """scale * (x if x > 0 else alpha * expm1(x * input_scale))."""
    neg = _op(c, "mul", _expm1(c, _op(c, "mul", x, _f(c, input_scale, x))), _f(c, alpha, x))
    y = _op(c, "where", _op(c, "gt", x, _f(c, 0.0, x)), x, neg)
    return y if scale == 1.0 else _op(c, "mul", y, _f(c, scale, x))


HANDLERS["aten.elu.default"] = _act(lambda c, x, a: _elu_like(c, x, float(a["alpha"]), float(a["scale"]),
                                                               float(a["input_scale"])))
HANDLERS["aten.elu_.default"] = HANDLERS["aten.elu.default"]
_SELU_ALPHA = 1.6732632423543772848170429916717
_SELU_SCALE = 1.0507009873554804934193349852946
HANDLERS["aten.selu.default"] = _act(lambda c, x, a: _elu_like(c, x, _SELU_ALPHA, _SELU_SCALE, 1.0))
HANDLERS["aten.selu_.default"] = HANDLERS["aten.selu.default"]
HANDLERS["aten.celu.default"] = _act(lambda c, x, a: _elu_like(c, x, float(a["alpha"]), 1.0, 1.0 / float(a["alpha"])))
HANDLERS["aten.celu_.default"] = HANDLERS["aten.celu.default"]


def _leaky(c, x, a):
    return _op(c, "where", _op(c, "gt", x, _f(c, 0.0, x)), x, _op(c, "mul", x, _f(c, float(a["negative_slope"]), x)))


HANDLERS["aten.leaky_relu.default"] = _act(_leaky)
HANDLERS["aten.leaky_relu_.default"] = _act(_leaky)


def _softplus_h(c, x, a):
    """x where beta x > threshold, else log(1 + exp(beta x)) / beta."""
    beta, thr = float(a["beta"]), float(a["threshold"])
    z = x if beta == 1.0 else _op(c, "mul", x, _f(c, beta, x))
    sp = _softplus(c, z)
    sp = sp if beta == 1.0 else _op(c, "div", sp, _f(c, beta, x))
    return _op(c, "where", _op(c, "gt", z, _f(c, thr, x)), x, sp)


HANDLERS["aten.softplus.default"] = _act(_softplus_h)
HANDLERS["aten.mish.default"] = _act(lambda c, x, a: _op(c, "mul", x, _op(c, "tanh", _softplus(c, x))))
HANDLERS["aten.mish_.default"] = HANDLERS["aten.mish.default"]
HANDLERS["aten.log_sigmoid.default"] = _act(lambda c, x, a: _op(c, "neg", _softplus(c, _op(c, "neg", x))))


def _hardshrink(c, x, a):
    lam = float(a["lambd"])
    keep = _op(c, "gt", _op(c, "abs", x), _f(c, lam, x))
    return _op(c, "where", keep, x, _op(c, "full_like", x, value=0.0, dtype=None))


def _softshrink(c, x, a):
    lam = float(a["lambd"])
    up = _op(c, "sub", x, _f(c, lam, x))
    down = _op(c, "add", x, _f(c, lam, x))
    zero = _op(c, "full_like", x, value=0.0, dtype=None)
    return _op(c, "where", _op(c, "gt", x, _f(c, lam, x)), up, _op(c, "where", _op(c, "lt", x, _f(c, -lam, x)), down, zero))


def _threshold(c, x, a):
    return _op(c, "where", _op(c, "gt", x, _f(c, float(a["threshold"]), x)), x,
               _op(c, "full_like", x, value=a["value"], dtype=None))


HANDLERS["aten.hardshrink.default"] = _act(_hardshrink)
HANDLERS["aten.softshrink.default"] = _act(_softshrink)
HANDLERS["aten.threshold.default"] = _act(_threshold)
HANDLERS["aten.threshold_.default"] = _act(_threshold)


@aten("aten.glu.default")
def _glu(c, a):
    x = c.tensor(a["self"])
    d = a["dim"] % x.type.rank
    (n,) = _fixed(c, "the split dim of glu", [x.type.shape[d]])
    first = _op(c, "slice", x, dim=d, start=0, end=n // 2, step=1)
    second = _op(c, "slice", x, dim=d, start=n // 2, end=n, step=1)
    return _op(c, "mul", first, _op(c, "sigmoid", second))


@aten("aten.prelu.default", "aten._prelu_kernel.default")
def _prelu(c, a):
    x, w = c.tensor(a["self"]), c.tensor(a["weight"])
    if w.type.rank == 1 and x.type.rank >= 2 and w.type.shape[0] != 1:
        w = _reshape(c, w, [w.type.shape[0]] + [1] * (x.type.rank - 2))
    return _op(c, "where", _op(c, "gt", x, _f(c, 0.0, x)), x, _op(c, "mul", x, w))


@aten("aten.log_softmax.int", "aten._log_softmax.default")
def _log_softmax(c, a):
    x = c.tensor(a["self"])
    if a.get("dtype") is not None and a["dtype"] != x.type.dtype:
        x = _op(c, "cast", x, dtype=a["dtype"])
    if a.get("half_to_float"):
        _unsupported(c, "log_softmax with half_to_float is not supported")
    return _op(c, "log_softmax", x, dim=a["dim"])


@aten("aten.log1p.default")
def _aten_log1p(c, a):
    return _log1p(c, c.tensor(a["self"]))


# ----------------------------------------------------------------------------- small layers


@aten("aten.bilinear.default")
def _bilinear(c, a):
    """y[n, o] = sum_ij x1[n, i] W[o, i, j] x2[n, j] + b[o]."""
    x1, x2, w, b = (c.tensor(a[k]) for k in ("input1", "input2", "weight", "bias"))
    o, i, j = w.type.shape
    lead = list(x1.type.shape[:-1])
    rows = norm_dim(math.prod([sympy.sympify(d) for d in lead]) if lead else 1)
    a1 = _reshape(c, x1, [rows, i])
    a2 = _reshape(c, x2, [rows, 1, j])
    wt = _reshape(c, _op(c, "permute", w, dims=[1, 0, 2]), [i, norm_dim(o * j)])
    t = _reshape(c, _op(c, "matmul", a1, wt), [rows, o, j])
    y = _op(c, "sum", _op(c, "mul", t, a2), dims=[2], keepdim=False)
    if b is not None:
        y = _op(c, "add", y, b)
    return _reshape(c, y, lead + [o])


@aten("aten.im2col.default")
def _im2col(c, a):
    """[N, C, H, W] -> [N, C kh kw, L]: the padded input read at every kernel tap, taps after channels."""
    x = c.tensor(a["self"])
    x, squeeze = _unbatched(c, x, 4)
    kh, kw = _ints(a["kernel_size"], 2)
    dh, dw = _ints(a["dilation"], 2)
    ph, pw = _ints(a["padding"], 2)
    sh, sw = _ints(a["stride"], 2)
    n, ch, h, w = x.type.shape
    h, w = _fixed(c, "the spatial sizes of unfold", [h, w])
    if ph or pw:
        x = _op(c, "pad", x, pads=[pw, pw, ph, ph], value=0.0)
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    taps = []
    for u in range(kh):
        for v in range(kw):
            t = _op(c, "slice", x, dim=2, start=u * dh, end=u * dh + sh * (oh - 1) + 1, step=sh)
            t = _op(c, "slice", t, dim=3, start=v * dw, end=v * dw + sw * (ow - 1) + 1, step=sw)
            taps.append(_reshape(c, t, [n, ch, 1, oh * ow]))
    y = taps[0] if len(taps) == 1 else _op(c, "cat", *taps, dim=2)
    y = _reshape(c, y, [n, norm_dim(sympy.sympify(ch) * kh * kw), oh * ow])
    return _reshape(c, y, list(y.type.shape[1:])) if squeeze else y


@aten("aten.col2im.default")
def _col2im(c, a):
    """The adjoint of unfold: every column entry added at its tap's position. That is the transposed
    convolution of the columns (as channels C kh kw) with a one hot kernel per tap, C groups."""
    x = c.tensor(a["self"])
    x, squeeze = _unbatched(c, x, 3)
    H, W = _ints(a["output_size"], 2)
    kh, kw = _ints(a["kernel_size"], 2)
    dh, dw = _ints(a["dilation"], 2)
    ph, pw = _ints(a["padding"], 2)
    sh, sw = _ints(a["stride"], 2)
    n, ck, length = x.type.shape
    (ck,) = _fixed(c, "the channels of fold", [ck])
    ch = ck // (kh * kw)
    oh = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    g = _reshape(c, x, [n, ck, oh, ow])
    onehot = torch.zeros((ck, 1, kh, kw), dtype=torch.float64)
    for cc in range(ch):
        for u in range(kh):
            for v in range(kw):
                onehot[cc * kh * kw + u * kw + v, 0, u, v] = 1.0
    w = _t(c, "fold_taps", onehot, x.type.dtype)
    y = _op(c, "conv2d_input_grad", g, w, stride=[sh, sw], padding=[ph, pw], dilation=[dh, dw], groups=ch,
            input_size=[H, W])
    return _reshape(c, y, list(y.type.shape[1:])) if squeeze else y


def _broadcast_pair(c, x: Value, y: Value):
    from .shapes import broadcast
    shape = list(broadcast("broadcast", [x.type, y.type], ["x1", "x2"]))
    xs = x if list(x.type.shape) == shape else _op(c, "expand", x, shape=shape)
    ys = y if list(y.type.shape) == shape else _op(c, "expand", y, shape=shape)
    return xs, ys


@aten("aten.cosine_similarity.default")
def _cosine_similarity(c, a):
    """sum(x1 x2) / (max(|x1|, eps) max(|x2|, eps)) along dim."""
    x1, x2 = _broadcast_pair(c, c.tensor(a["x1"]), c.tensor(a["x2"]))
    d, eps = a["dim"], float(a["eps"])
    dot = _op(c, "sum", _op(c, "mul", x1, x2), dims=[d], keepdim=False)
    n1 = _op(c, "clamp", _op(c, "sqrt", _op(c, "sum", _op(c, "mul", x1, x1), dims=[d], keepdim=False)), min=eps, max=None)
    n2 = _op(c, "clamp", _op(c, "sqrt", _op(c, "sum", _op(c, "mul", x2, x2), dims=[d], keepdim=False)), min=eps, max=None)
    return _op(c, "div", dot, _op(c, "mul", n1, n2))


@aten("aten.pairwise_distance.default")
def _pairwise_distance(c, a):
    """||x1 - x2 + eps||_p over the last dim."""
    x1, x2 = c.tensor(a["x1"]), c.tensor(a["x2"])
    diff = _op(c, "add", _op(c, "sub", x1, x2), _f(c, float(a["eps"]), x1))
    return _op(c, "vector_norm", diff, ord=float(a["p"]), dims=[-1], keepdim=bool(a["keepdim"]))


@aten("aten.broadcast_tensors.default")
def _broadcast_tensors(c, a):
    from .shapes import broadcast
    xs = [c.tensor(t) for t in a["tensors"]]
    shape = list(broadcast("broadcast_tensors", [x.type for x in xs], [f"t{i}" for i in range(len(xs))]))
    return [x if list(x.type.shape) == shape else _op(c, "expand", x, shape=shape) for x in xs]


# ----------------------------------------------------------------------------- embedding bag


@aten("aten.embedding_bag.padding_idx", "aten._embedding_bag.default", "aten.embedding_bag.default")
def _embedding_bag(c, a):
    """Sum, mean or max of the rows of each bag. 2D indices are one bag per row; 1D indices with
    offsets give bag k the positions [offsets[k], offsets[k + 1]). padding_idx rows are left out."""
    w, idx = c.tensor(a["weight"]), c.tensor(a["indices"])
    offsets = c.tensor(a["offsets"]) if a.get("offsets") is not None else None
    mode = int(a.get("mode", 0))
    psw = c.tensor(a["per_sample_weights"]) if a.get("per_sample_weights") is not None else None
    pad = a.get("padding_idx")
    pad = None if pad is None or pad < 0 else pad
    if a.get("include_last_offset"):
        _unsupported(c, "embedding_bag with include_last_offset is not supported yet")
    missing = (_Missing("offset2bag of embedding_bag"), _Missing("bag_size of embedding_bag"),
               _Missing("max_indices of embedding_bag"))
    if idx.type.rank == 2 and offsets is not None:
        _unsupported(c, "2D indices with offsets")
    if idx.type.rank == 1:
        if offsets is None:
            _unsupported(c, "1D indices need offsets")
        return (_bags_by_offsets(c, w, idx, offsets, mode, psw, pad),) + missing
    rows = _op(c, "index", w, idx)                                    # [B, L, dim]
    keep = None
    if pad is not None:
        keep = _reshape(c, _op(c, "ne", idx, c.const(pad, idx)), list(idx.type.shape) + [1])
    if psw is not None:
        rows = _op(c, "mul", rows, _reshape(c, psw, list(psw.type.shape) + [1]))
    if mode == 2:
        if keep is not None:
            rows = _op(c, "where", keep, rows, _op(c, "full_like", rows, value=-math.inf, dtype=None))
        out = _op(c, "amax", rows, dims=[1], keepdim=False)
        if keep is not None:   # a bag of padding only gives zeros
            any_ = _op(c, "any", keep, dims=[1], keepdim=False)
            out = _op(c, "where", any_, out, _op(c, "full_like", out, value=0.0, dtype=None))
        return (out,) + missing
    if keep is not None:
        rows = _op(c, "where", keep, rows, _op(c, "full_like", rows, value=0.0, dtype=None))
    total = _op(c, "sum", rows, dims=[1], keepdim=False)
    if mode == 0:
        return (total,) + missing
    if keep is None:
        return (_op(c, "mean", rows, dims=[1], keepdim=False),) + missing
    count = _op(c, "sum", _op(c, "cast", keep, dtype=w.type.dtype), dims=[1], keepdim=False)
    count = _op(c, "clamp", count, min=1.0, max=None)
    return (_op(c, "div", total, count),) + missing


def _bags_by_offsets(c, w, idx, offsets, mode, psw, pad):
    """bag of position i = (number of offsets <= i) - 1; rows are scatter added per bag."""
    n = idx.type.shape[0]
    nb = offsets.type.shape[0]
    pos = _reshape(c, _op(c, "arange", start=0, end=n, step=1, dtype=torch.int64), [n, 1])
    offs = _reshape(c, offsets if offsets.type.dtype == torch.int64 else _op(c, "cast", offsets, dtype=torch.int64), [1, nb])
    le = _op(c, "cast", _op(c, "le", offs, pos), dtype=torch.int64)
    bag = _op(c, "sub", _op(c, "sum", le, dims=[1], keepdim=False), c.const(1, pos))        # [n]
    rows = _op(c, "index_select", w, idx, dim=0)                                             # [n, dim]
    ones = _op(c, "full", shape=[n, 1], value=1.0, dtype=w.type.dtype)
    if pad is not None:
        keep = _reshape(c, _op(c, "ne", idx, c.const(pad, idx)), [n, 1])
        rows = _op(c, "where", keep, rows, _op(c, "full_like", rows, value=0.0, dtype=None))
        ones = _op(c, "where", keep, ones, _op(c, "full_like", ones, value=0.0, dtype=None))
    if psw is not None:
        rows = _op(c, "mul", rows, _reshape(c, psw, [n, 1]))
    dim = w.type.shape[1]
    if mode == 2:
        # max: a [bags, n] membership mask, rows outside a bag (or padding) at -inf, max over n;
        # an empty bag gives 0, as torch
        member = _op(c, "eq", _reshape(c, bag, [1, n]),
                     _reshape(c, _op(c, "arange", start=0, end=nb, step=1, dtype=torch.int64), [nb, 1]))
        if pad is not None:
            member = _op(c, "logical_and", member, _reshape(c, _op(c, "ne", idx, c.const(pad, idx)), [1, n]))
        m3 = _reshape(c, member, [nb, n, 1])
        r3 = _op(c, "expand", _reshape(c, rows, [1, n, dim]), shape=[nb, n, dim])
        vals = _op(c, "where", m3, r3, _op(c, "full_like", r3, value=-math.inf, dtype=None))
        best = _op(c, "amax", vals, dims=[1], keepdim=False)
        nonempty = _op(c, "any", member, dims=[1], keepdim=True)
        return _op(c, "where", nonempty, best, _op(c, "full_like", best, value=0.0, dtype=None))
    index = _op(c, "expand", _reshape(c, bag, [n, 1]), shape=[n, dim])
    total = _op(c, "scatter_add", _op(c, "full", shape=[nb, dim], value=0.0, dtype=w.type.dtype), index, rows, dim=0)
    if mode == 0:
        return total
    count = _op(c, "scatter_add", _op(c, "full", shape=[nb, 1], value=0.0, dtype=w.type.dtype),
                _reshape(c, bag, [n, 1]), ones, dim=0)
    return _op(c, "div", total, _op(c, "clamp", count, min=1.0, max=None))


# ----------------------------------------------------------------------------- recurrent layers


def _cell(c, kind: str, x: Value, h: Value, cstate: Optional[Value], w_ih, w_hh, b_ih, b_hh, hidden: int):
    """One step. x [B, I], h [B, H]. Gate order as torch.nn: LSTM i, f, g, o; GRU r, z, n."""
    def lin(v, w, b):
        return _op(c, "linear", v, w, b)

    def part(v, k):
        return _op(c, "slice", v, dim=v.type.rank - 1, start=k * hidden, end=(k + 1) * hidden, step=1)

    gi = lin(x, w_ih, b_ih)
    gh = lin(h, w_hh, b_hh)
    if kind == "rnn_tanh":
        return _op(c, "tanh", _op(c, "add", gi, gh)), None
    if kind == "rnn_relu":
        return _op(c, "relu", _op(c, "add", gi, gh)), None
    if kind == "lstm":
        g = _op(c, "add", gi, gh)
        i = _op(c, "sigmoid", part(g, 0))
        f = _op(c, "sigmoid", part(g, 1))
        cand = _op(c, "tanh", part(g, 2))
        o = _op(c, "sigmoid", part(g, 3))
        c2 = _op(c, "add", _op(c, "mul", f, cstate), _op(c, "mul", i, cand))
        return _op(c, "mul", o, _op(c, "tanh", c2)), c2
    if kind == "gru":
        r = _op(c, "sigmoid", _op(c, "add", part(gi, 0), part(gh, 0)))
        z = _op(c, "sigmoid", _op(c, "add", part(gi, 1), part(gh, 1)))
        nn_ = _op(c, "tanh", _op(c, "add", part(gi, 2), _op(c, "mul", r, part(gh, 2))))
        # (1 - z) n + z h
        return _op(c, "add", nn_, _op(c, "mul", z, _op(c, "sub", h, nn_))), None
    raise UnsupportedOpError(kind)


def _rnn(kind: str):
    def handler(c, a):
        x = c.tensor(a["input"])
        if a.get("train") and float(a.get("dropout", 0.0)) > 0:
            _unsupported(c, "recurrent dropout in training mode is not supported")
        params = [c.tensor(p) for p in a["params"]]
        has_b, layers, bi, bf = bool(a["has_biases"]), int(a["num_layers"]), bool(a["bidirectional"]), bool(a["batch_first"])
        dirs = 2 if bi else 1
        per = 4 if has_b else 2
        if len(params) != layers * dirs * per:
            _unsupported(c, "recurrent layers with projections (proj_size) are not supported")
        hx = a["hx"]
        if kind == "lstm":
            h0, c0 = c.tensor(hx[0]), c.tensor(hx[1])
        else:
            h0, c0 = c.tensor(hx), None
        unbatched = x.type.rank == 2
        if unbatched:
            x = _reshape(c, x, [1] + list(x.type.shape)) if bf else _reshape(c, x, [x.type.shape[0], 1, x.type.shape[1]])
            h0 = _reshape(c, h0, [h0.type.shape[0], 1, h0.type.shape[1]])
            if c0 is not None:
                c0 = _reshape(c, c0, [c0.type.shape[0], 1, c0.type.shape[1]])
        seq = x if bf else _op(c, "permute", x, dims=[1, 0, 2])          # [B, T, F]
        (steps,) = _fixed(c, "the sequence length of a recurrent layer (it is unrolled)", [seq.type.shape[1]])
        hidden = h0.type.shape[2]
        b = seq.type.shape[0]
        hn, cn = [], []
        layer_in = seq
        for layer in range(layers):
            outs_dir = []
            for d in range(dirs):
                k = (layer * dirs + d) * per
                w_ih, w_hh = params[k], params[k + 1]
                b_ih, b_hh = (params[k + 2], params[k + 3]) if has_b else (None, None)
                slot = layer * dirs + d
                h = _op(c, "select", h0, dim=0, index=slot)
                cs = _op(c, "select", c0, dim=0, index=slot) if c0 is not None else None
                order = range(steps) if d == 0 else range(steps - 1, -1, -1)
                outs = [None] * steps
                for t in order:
                    xt = _op(c, "select", layer_in, dim=1, index=t)
                    h, cs2 = _cell(c, kind, xt, h, cs, w_ih, w_hh, b_ih, b_hh, hidden)
                    cs = cs2 if cs2 is not None else cs
                    outs[t] = _reshape(c, h, [b, 1, hidden])
                outs_dir.append(outs[0] if steps == 1 else _op(c, "cat", *outs, dim=1))
                hn.append(_reshape(c, h, [1, b, hidden]))
                if cs is not None:
                    cn.append(_reshape(c, cs, [1, b, hidden]))
            layer_in = outs_dir[0] if dirs == 1 else _op(c, "cat", *outs_dir, dim=2)
        out = layer_in if bf else _op(c, "permute", layer_in, dims=[1, 0, 2])
        h_n = hn[0] if len(hn) == 1 else _op(c, "cat", *hn, dim=0)
        c_n = (cn[0] if len(cn) == 1 else _op(c, "cat", *cn, dim=0)) if cn else None
        if unbatched:
            out = _reshape(c, out, [steps, out.type.shape[-1]])
            h_n = _reshape(c, h_n, [h_n.type.shape[0], hidden])
            c_n = _reshape(c, c_n, [c_n.type.shape[0], hidden]) if c_n is not None else None
        return (out, h_n, c_n) if kind == "lstm" else (out, h_n)
    return handler


HANDLERS["aten.lstm.input"] = _rnn("lstm")
HANDLERS["aten.gru.input"] = _rnn("gru")
HANDLERS["aten.rnn_tanh.input"] = _rnn("rnn_tanh")
HANDLERS["aten.rnn_relu.input"] = _rnn("rnn_relu")


def _cell_handler(kind: str):
    def handler(c, a):
        x = c.tensor(a["input"])
        if kind == "lstm":
            h, cs = (c.tensor(v) for v in a["hx"])
        else:
            h, cs = c.tensor(a["hx"]), None
        w_ih, w_hh, b_ih, b_hh = (c.tensor(a[k]) for k in ("w_ih", "w_hh", "b_ih", "b_hh"))
        hidden = h.type.shape[-1]
        h2, c2 = _cell(c, kind, x, h, cs, w_ih, w_hh, b_ih, b_hh, hidden)
        return (h2, c2) if kind == "lstm" else h2
    return handler


HANDLERS["aten.lstm_cell.default"] = _cell_handler("lstm")
HANDLERS["aten.gru_cell.default"] = _cell_handler("gru")
HANDLERS["aten.rnn_tanh_cell.default"] = _cell_handler("rnn_tanh")
HANDLERS["aten.rnn_relu_cell.default"] = _cell_handler("rnn_relu")


# ----------------------------------------------------------------------------- losses


def _pair(c, a):
    x, t = c.tensor(a["self"]), c.tensor(a["target"])
    return x, t


@aten("aten.mse_loss.default")
def _mse(c, a):
    x, t = _pair(c, a)
    d = _op(c, "sub", x, t)
    return _reduce(c, _op(c, "mul", d, d), a["reduction"])


@aten("aten.l1_loss.default")
def _l1(c, a):
    x, t = _pair(c, a)
    return _reduce(c, _op(c, "abs", _op(c, "sub", x, t)), a["reduction"])


@aten("aten.smooth_l1_loss.default")
def _smooth_l1(c, a):
    """|d| < beta: d^2 / (2 beta), else |d| - beta / 2 (beta = 0 is the L1 loss)."""
    x, t = _pair(c, a)
    beta = float(a["beta"])
    d = _op(c, "abs", _op(c, "sub", x, t))
    if beta == 0.0:
        return _reduce(c, d, a["reduction"])
    quad = _op(c, "mul", _op(c, "mul", d, d), _f(c, 0.5 / beta, x))
    lin = _op(c, "sub", d, _f(c, 0.5 * beta, x))
    return _reduce(c, _op(c, "where", _op(c, "lt", d, _f(c, beta, x)), quad, lin), a["reduction"])


@aten("aten.huber_loss.default")
def _huber(c, a):
    """|d| < delta: d^2 / 2, else delta (|d| - delta / 2)."""
    x, t = _pair(c, a)
    delta = float(a["delta"])
    d = _op(c, "abs", _op(c, "sub", x, t))
    quad = _op(c, "mul", _op(c, "mul", d, d), _f(c, 0.5, x))
    lin = _op(c, "mul", _op(c, "sub", d, _f(c, 0.5 * delta, x)), _f(c, delta, x))
    return _reduce(c, _op(c, "where", _op(c, "lt", d, _f(c, delta, x)), quad, lin), a["reduction"])


def _class_dim(x: Value) -> int:
    return 0 if x.type.rank == 1 else 1


def _nll(c, lp: Value, target: Value, weight: Optional[Value], reduction: int, ignore: int, smoothing: float):
    """-w[t] lp[t] per element (masked where t == ignore_index), with label smoothing
    (1 - eps) nll + eps * mean_c(-w[c] lp[c]); mean divides by the sum of w[t] over counted elements."""
    cd = _class_dim(lp)
    ncls = lp.type.shape[cd]
    keep = _op(c, "ne", target, c.const(ignore, target))
    safe = _op(c, "where", keep, target, _op(c, "full_like", target, value=0, dtype=None))
    if safe.type.dtype != torch.int64:
        safe = _op(c, "cast", safe, dtype=torch.int64)
    tshape = list(target.type.shape)
    idx = _reshape(c, safe, tshape[:cd] + [1] + tshape[cd:]) if lp.type.rank > 1 else _reshape(c, safe, [1])
    picked = _op(c, "gather", lp, idx, dim=cd)
    picked = _reshape(c, picked, tshape) if lp.type.rank > 1 else _reshape(c, picked, [])
    zero = _op(c, "full_like", picked, value=0.0, dtype=None)
    if weight is not None:
        wt = _op(c, "index_select", weight, _reshape(c, safe, [norm_dim(math.prod(
            [sympy.sympify(d) for d in tshape]) if tshape else 1)]), dim=0)
        wt = _reshape(c, wt, tshape)
    else:
        wt = _op(c, "full_like", picked, value=1.0, dtype=None)
    wt = _op(c, "where", keep, wt, zero)
    loss = _op(c, "neg", _op(c, "mul", picked, wt))
    if smoothing:
        wl = lp if weight is None else _op(c, "mul", lp, _reshape(c, weight, [ncls] + [1] * (lp.type.rank - cd - 1)))
        smooth = _op(c, "neg", _op(c, "sum", wl, dims=[cd], keepdim=False))
        smooth = _op(c, "where", keep, smooth, zero)
        loss = _op(c, "add", _op(c, "mul", loss, _f(c, 1.0 - smoothing, lp)),
                   _op(c, "mul", smooth, _f(c, smoothing / ncls, lp)))
    if reduction == 0:
        return loss
    total = _op(c, "sum", loss, dims=None, keepdim=False)
    if reduction == 2:
        return total
    return _op(c, "div", total, _op(c, "sum", wt, dims=None, keepdim=False))


@aten("aten.nll_loss_nd.default", "aten.nll_loss.default", "aten.nll_loss2d.default")
def _nll_loss(c, a):
    w = c.tensor(a["weight"]) if a.get("weight") is not None else None
    return _nll(c, c.tensor(a["self"]), c.tensor(a["target"]), w, a["reduction"], a["ignore_index"], 0.0)


@aten("aten.nll_loss_forward.default")
def _nll_loss_forward(c, a):
    w = c.tensor(a["weight"]) if a.get("weight") is not None else None
    out = _nll(c, c.tensor(a["self"]), c.tensor(a["target"]), w, a["reduction"], a["ignore_index"], 0.0)
    return (out, _Missing("the total weight of nll_loss"))


@aten("aten.cross_entropy_loss.default")
def _cross_entropy(c, a):
    x, t = c.tensor(a["self"]), c.tensor(a["target"])
    w = c.tensor(a["weight"]) if a.get("weight") is not None else None
    cd = _class_dim(x)
    lp = _op(c, "log_softmax", x, dim=cd)
    eps = float(a.get("label_smoothing", 0.0))
    if t.type.dtype.is_floating_point:   # class probabilities
        ncls = x.type.shape[cd]
        if eps:
            t = _op(c, "add", _op(c, "mul", t, _f(c, 1.0 - eps, x)), _f(c, eps / ncls, x))
        wl = lp if w is None else _op(c, "mul", lp, _reshape(c, w, [ncls] + [1] * (x.type.rank - cd - 1)))
        loss = _op(c, "neg", _op(c, "sum", _op(c, "mul", wl, t), dims=[cd], keepdim=False))
        return _reduce(c, loss, a["reduction"])     # mean: over the batch elements, as torch
    return _nll(c, lp, t, w, a["reduction"], a["ignore_index"], eps)


@aten("aten.binary_cross_entropy_with_logits.default")
def _bce_logits(c, a):
    """(1 - y) x + (1 + (p - 1) y) softplus(-x), times the weight."""
    x, y = _pair(c, a)
    sp = _softplus(c, _op(c, "neg", x))
    pw = c.tensor(a["pos_weight"]) if a.get("pos_weight") is not None else None
    coef = _f(c, 1.0, x) if pw is None else _op(c, "add", _f(c, 1.0, x), _op(c, "mul", _op(c, "sub", pw, _f(c, 1.0, x)), y))
    loss = _op(c, "add", _op(c, "mul", _op(c, "sub", _f(c, 1.0, x), y), x), _op(c, "mul", coef, sp))
    if a.get("weight") is not None:
        loss = _op(c, "mul", loss, c.tensor(a["weight"]))
    return _reduce(c, loss, a["reduction"])


@aten("aten.binary_cross_entropy.default")
def _bce(c, a):
    """-(y log p + (1 - y) log(1 - p)) with each log clamped at -100 (PyTorch's convention)."""
    p, y = _pair(c, a)
    lp = _op(c, "clamp", _op(c, "log", p), min=-100.0, max=None)
    lq = _op(c, "clamp", _log1p(c, _op(c, "neg", p)), min=-100.0, max=None)
    loss = _op(c, "neg", _op(c, "add", _op(c, "mul", y, lp), _op(c, "mul", _op(c, "sub", _f(c, 1.0, p), y), lq)))
    if a.get("weight") is not None:
        loss = _op(c, "mul", loss, c.tensor(a["weight"]))
    return _reduce(c, loss, a["reduction"])


@aten("aten.kl_div.default")
def _kl_div(c, a):
    """target (log target - input), 0 where target is 0; with log_target: exp(target) (target - input)."""
    x, t = _pair(c, a)
    if a.get("log_target"):
        loss = _op(c, "mul", _op(c, "exp", t), _op(c, "sub", t, x))
    else:
        pos = _op(c, "gt", t, _f(c, 0.0, t))
        safe = _op(c, "where", pos, t, _op(c, "full_like", t, value=1.0, dtype=None))
        xlogx = _op(c, "where", pos, _op(c, "mul", t, _op(c, "log", safe)), _op(c, "full_like", t, value=0.0, dtype=None))
        loss = _op(c, "sub", xlogx, _op(c, "mul", t, x))
    return _reduce(c, loss, a["reduction"])


from . import importer_current  # noqa: E402,F401  (ops of the 2025 to 2026 families)
from . import importer_vision  # noqa: E402,F401  (SigLIP2 and Pixtral vision)
from . import importer_gen  # noqa: E402,F401  (ops of the image generation, detection and segmentation models)
