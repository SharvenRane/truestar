"""Shape and dtype rules for every IR op.

``infer(op, input_types, attrs)`` returns the output type or raises ShapeError naming the op and
every operand type. Rules work on symbolic dims: two dims match only when they are provably equal,
so a program that type checks is correct for every value of its symbols.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import sympy
import torch

from .core import Dim, ShapeError, TensorType, UnsupportedOpError, dims_equal, norm_dim

Types = List[Optional[TensorType]]
RULES: Dict[str, Callable[[Types, Dict[str, Any]], TensorType]] = {}


def rule(*names: str):
    def deco(fn):
        for n in names:
            RULES[n] = fn
        return fn
    return deco


def infer_outputs(op: str, types: Types, attrs: Dict[str, Any]) -> List[TensorType]:
    """Types of every value the op defines. Control flow ops (``cond``, ``loop``) define several."""
    fn = RULES.get(op)
    if fn is None:
        raise UnsupportedOpError(f"IR op {op!r} does not exist. Known IR ops: {sorted(RULES)}")
    out = fn(types, attrs)
    return list(out) if isinstance(out, (tuple, list)) else [out]


def infer(op: str, types: Types, attrs: Dict[str, Any]) -> TensorType:
    """Type of the single value the op defines. Use ``infer_outputs`` for multi value ops."""
    outs = infer_outputs(op, types, attrs)
    if len(outs) != 1:
        raise ShapeError(op, f"defines {len(outs)} values; build it with Builder.op_multi")
    return outs[0]


# ----------------------------------------------------------------------------- helpers


def _labels(types: Types, names: Sequence[str]):
    return [(n, t) for n, t in zip(names, types) if t is not None]


def _need(op: str, types: Types, n_required: int, names: Sequence[str]):
    if len(types) != len(names):
        raise ShapeError(op, f"expects {len(names)} operands ({', '.join(names)}), got {len(types)}",
                         _labels(types, [f"operand {i}" for i in range(len(types))]))
    for i in range(n_required):
        if types[i] is None:
            raise ShapeError(op, f"operand {names[i]} is required", _labels(types, names))


def _axis(op: str, d: int, rank: int, t: TensorType) -> int:
    if not -max(rank, 1) <= d < max(rank, 1):
        raise ShapeError(op, f"dim {d} is out of range for rank {rank}", [("input", t)])
    return d % max(rank, 1)


def _is_float(dtype: torch.dtype) -> bool:
    return dtype.is_floating_point


def _numel(shape) -> sympy.Expr:
    out = sympy.Integer(1)
    for d in shape:
        out = out * sympy.sympify(d)
    return sympy.expand(out)


def broadcast(op: str, types: Sequence[TensorType], names: Sequence[str]) -> tuple:
    rank = max(t.rank for t in types)
    out = []
    for i in range(rank):
        dim: Dim = 1
        for t in types:
            j = i - (rank - t.rank)
            if j < 0:
                continue
            d = t.shape[j]
            if dims_equal(d, 1):
                continue
            if dims_equal(dim, 1):
                dim = d
            elif not dims_equal(dim, d):
                raise ShapeError(op, f"cannot broadcast: dim {i - rank} is {dim} in one operand and {d} "
                                     "in another (symbolic dims must be provably equal)",
                                 list(zip(names, types)))
        out.append(dim)
    return tuple(out)


def result_dtype(a: TensorType, b: TensorType) -> torch.dtype:
    """PyTorch's promotion rule, including the rule that 0 dim operands do not widen a dtype."""
    pa = torch.empty(() if a.rank == 0 else (1,), dtype=a.dtype)
    pb = torch.empty(() if b.rank == 0 else (1,), dtype=b.dtype)
    return torch.result_type(pa, pb)


def _ints(op: str, v: Any, n: int, name: str) -> List[int]:
    if isinstance(v, int):
        return [v] * n
    v = list(v)
    if len(v) == 1:
        v = v * n
    if len(v) != n or not all(isinstance(x, int) for x in v):
        raise ShapeError(op, f"attribute {name} must be {n} ints, got {v}")
    return v


def _floor_div(num: Dim, den: int) -> Dim:
    if isinstance(num, int):
        return num // den
    return norm_dim(sympy.floor(sympy.sympify(num) / den))


def _ceil_div(num: Dim, den: int) -> Dim:
    if isinstance(num, int):
        return -(-num // den)
    return norm_dim(sympy.ceiling(sympy.sympify(num) / den))


# ----------------------------------------------------------------------------- elementwise


@rule("neg", "exp", "sqrt", "rsqrt", "tanh", "relu", "gelu", "hardswish", "hardsigmoid", "silu",
      "sigmoid", "clamp")
def _unary(types, attrs):
    (t,) = types
    if "approximate" in attrs and attrs["approximate"] not in ("none", "tanh"):
        raise UnsupportedOpError(f"gelu approximate={attrs['approximate']!r} is not supported")
    return t


@rule("logical_not")
def _logical_not(types, attrs):
    (t,) = types
    return TensorType(torch.bool, t.shape)


def _binary(op):
    def fn(types, attrs):
        _need(op, types, 2, ["a", "b"])
        a, b = types
        shape = broadcast(op, [a, b], ["a", "b"])
        dtype = torch.bool if op in _BOOL_RESULT else result_dtype(a, b)
        if op == "div" and not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        return TensorType(dtype, shape)
    return fn


_BOOL_RESULT = ("eq", "ne", "lt", "le", "gt", "ge", "logical_and", "logical_or")

for _op in ("add", "sub", "mul", "div") + _BOOL_RESULT:
    RULES[_op] = _binary(_op)


@rule("where")
def _where(types, attrs):
    _need("where", types, 3, ["condition", "a", "b"])
    c, a, b = types
    if c.dtype != torch.bool:
        raise ShapeError("where", "condition must be bool", [("condition", c)])
    shape = broadcast("where", [c, a, b], ["condition", "a", "b"])
    return TensorType(result_dtype(a, b), shape)


@rule("full_like")
def _full_like(types, attrs):
    (t,) = types
    return TensorType(attrs.get("dtype") or t.dtype, t.shape)


@rule("full")
def _full(types, attrs):
    if types:
        raise ShapeError("full", "takes no tensor operands")
    return TensorType(attrs["dtype"], tuple(attrs["shape"]))


@rule("arange")
def _arange(types, attrs):
    if types:
        raise ShapeError("arange", "takes no tensor operands")
    start, end, step = attrs["start"], attrs["end"], attrs["step"]
    if step == 0:
        raise ShapeError("arange", "step must not be 0")
    if any(isinstance(v, sympy.Basic) for v in (start, end, step)):
        # arange(T) or arange(start, T, step) with a symbolic end: n = ceil((end - start) / step)
        if not (isinstance(start, int) and isinstance(step, int) and step > 0):
            raise UnsupportedOpError("arange with a symbolic end needs an int start and a positive int step")
        # floor((end - start + step - 1) / step): the form torch.export records, so the two compare equal
        n = _floor_div(norm_dim(sympy.sympify(end) - start + step - 1), step)
        return TensorType(attrs["dtype"], (n,))
    n = max(0, math.ceil((end - start) / step))
    return TensorType(attrs["dtype"], (n,))


# ----------------------------------------------------------------------------- reductions


def _reduce(op):
    def fn(types, attrs):
        (t,) = types
        dims = attrs.get("dims")
        dims = list(range(t.rank)) if dims is None or len(dims) == 0 else dims
        axes = sorted({_axis(op, d, t.rank, t) for d in dims})
        keep = attrs.get("keepdim", False)
        shape = []
        for i, d in enumerate(t.shape):
            if i in axes:
                if keep:
                    shape.append(1)
            else:
                shape.append(d)
        dtype = torch.bool if op == "any" else t.dtype
        if op == "mean" and not dtype.is_floating_point:
            raise ShapeError(op, "mean needs a floating point input", [("input", t)])
        return TensorType(dtype, tuple(shape))
    return fn


for _op in ("mean", "sum", "any"):
    RULES[_op] = _reduce(_op)


@rule("softmax")
def _softmax(types, attrs):
    (t,) = types
    _axis("softmax", attrs["dim"], t.rank, t)
    return t


# ----------------------------------------------------------------------------- linear algebra


@rule("linear")
def _linear(types, attrs):
    _need("linear", types, 2, ["x", "weight", "bias"])
    x, w, b = types
    if w.rank != 2:
        raise ShapeError("linear", "weight must be rank 2 [out, in]", _labels(types, ["x", "weight", "bias"]))
    if x.rank < 1 or not dims_equal(x.shape[-1], w.shape[1]):
        raise ShapeError("linear", f"input feature size {x.shape[-1] if x.rank else '?'} does not match "
                                   f"weight in features {w.shape[1]}", _labels(types, ["x", "weight", "bias"]))
    if b is not None and (b.rank != 1 or not dims_equal(b.shape[0], w.shape[0])):
        raise ShapeError("linear", "bias must be [out]", _labels(types, ["x", "weight", "bias"]))
    return TensorType(result_dtype(x, w), x.shape[:-1] + (w.shape[0],))


@rule("matmul")
def _matmul(types, attrs):
    _need("matmul", types, 2, ["a", "b"])
    a, b = types
    if a.rank < 2 or b.rank < 2:
        raise ShapeError("matmul", "both operands must have rank 2 or more", [("a", a), ("b", b)])
    if not dims_equal(a.shape[-1], b.shape[-2]):
        raise ShapeError("matmul", f"inner dims differ: {a.shape[-1]} vs {b.shape[-2]}", [("a", a), ("b", b)])
    batch = broadcast("matmul", [TensorType(a.dtype, a.shape[:-2]), TensorType(b.dtype, b.shape[:-2])],
                      ["a batch", "b batch"])
    return TensorType(result_dtype(a, b), batch + (a.shape[-2], b.shape[-1]))


@rule("conv2d")
def _conv2d(types, attrs):
    names = ["x", "weight", "bias"]
    _need("conv2d", types, 2, names)
    x, w, b = types
    if x.rank != 4 or w.rank != 4:
        raise ShapeError("conv2d", "input must be [N, C, H, W] and weight [O, C/groups, kH, kW]",
                         _labels(types, names))
    stride = _ints("conv2d", attrs.get("stride", 1), 2, "stride")
    pad = _ints("conv2d", attrs.get("padding", 0), 2, "padding")
    dil = _ints("conv2d", attrs.get("dilation", 1), 2, "dilation")
    g = attrs.get("groups", 1)
    if not dims_equal(x.shape[1], sympy.sympify(w.shape[1]) * g):
        raise ShapeError("conv2d", f"input channels {x.shape[1]} != weight channels {w.shape[1]} x groups {g}",
                         _labels(types, names))
    if isinstance(w.shape[0], int) and w.shape[0] % g:
        raise ShapeError("conv2d", f"out channels {w.shape[0]} not divisible by groups {g}", _labels(types, names))
    if b is not None and (b.rank != 1 or not dims_equal(b.shape[0], w.shape[0])):
        raise ShapeError("conv2d", "bias must be [out channels]", _labels(types, names))
    out_hw = []
    for i in range(2):
        span = dil[i] * (w.shape[2 + i] - 1) + 1
        num = norm_dim(sympy.sympify(x.shape[2 + i]) + 2 * pad[i] - span)
        if isinstance(num, int) and num < 0:
            raise ShapeError("conv2d", "kernel is larger than the padded input", _labels(types, names))
        out_hw.append(norm_dim(sympy.sympify(_floor_div(num, stride[i])) + 1))
    return TensorType(result_dtype(x, w), (x.shape[0], w.shape[0], *out_hw))


# ----------------------------------------------------------------------------- normalization


@rule("batch_norm")
def _batch_norm(types, attrs):
    names = ["x", "weight", "bias", "running_mean", "running_var"]
    _need("batch_norm", types, 1, names)
    x = types[0]
    if types[3] is None or types[4] is None:
        raise ShapeError("batch_norm", "inference batch norm needs running_mean and running_var",
                         _labels(types, names))
    if x.rank < 2:
        raise ShapeError("batch_norm", "input must be [N, C, ...]", _labels(types, names))
    for t, n in zip(types[1:], names[1:]):
        if t is not None and (t.rank != 1 or not dims_equal(t.shape[0], x.shape[1])):
            raise ShapeError("batch_norm", f"{n} must be [C] with C = {x.shape[1]}", _labels(types, names))
    return x


@rule("layer_norm")
def _layer_norm(types, attrs):
    names = ["x", "weight", "bias"]
    _need("layer_norm", types, 1, names)
    x = types[0]
    ns = tuple(attrs["normalized_shape"])
    if x.rank < len(ns) or not all(dims_equal(a, b) for a, b in zip(x.shape[x.rank - len(ns):], ns)):
        raise ShapeError("layer_norm", f"trailing dims of the input do not match normalized_shape {list(ns)}",
                         _labels(types, names))
    for t, n in zip(types[1:], names[1:]):
        if t is not None and not (t.rank == len(ns) and all(dims_equal(a, b) for a, b in zip(t.shape, ns))):
            raise ShapeError("layer_norm", f"{n} must have shape {list(ns)}", _labels(types, names))
    return x


@rule("attention")
def _attention(types, attrs):
    names = ["q", "k", "v", "mask"]
    _need("attention", types, 3, names)
    q, k, v, m = types
    if q.rank < 2 or k.rank != q.rank or v.rank != q.rank:
        raise ShapeError("attention", "q, k, v must have the same rank (at least 2)", _labels(types, names))
    if not dims_equal(q.shape[-1], k.shape[-1]):
        raise ShapeError("attention", "q and k embedding dims differ", _labels(types, names))
    if not dims_equal(k.shape[-2], v.shape[-2]):
        raise ShapeError("attention", "k and v sequence lengths differ", _labels(types, names))
    batch = broadcast("attention", [TensorType(q.dtype, q.shape[:-2]), TensorType(k.dtype, k.shape[:-2]),
                                    TensorType(v.dtype, v.shape[:-2])], ["q batch", "k batch", "v batch"])
    if m is not None:
        broadcast("attention", [m, TensorType(q.dtype, batch + (q.shape[-2], k.shape[-2]))],
                  ["mask", "scores"])
    return TensorType(q.dtype, batch + (q.shape[-2], v.shape[-1]))


# ----------------------------------------------------------------------------- pooling


def pool_out(op, size: Dim, k: int, s: int, p: int, d: int, ceil_mode: bool) -> Dim:
    num = norm_dim(sympy.sympify(size) + 2 * p - d * (k - 1) - 1)
    out = norm_dim(sympy.sympify(_ceil_div(num, s) if ceil_mode else _floor_div(num, s)) + 1)
    if ceil_mode and isinstance(out, int) and isinstance(size, int) and (out - 1) * s >= size + p:
        out -= 1
    if ceil_mode and not isinstance(out, int):
        raise UnsupportedOpError(f"{op} with ceil_mode=True on a symbolic spatial dim is not supported")
    return out


@rule("max_pool2d")
def _max_pool2d(types, attrs):
    (x,) = types
    if x.rank != 4:
        raise ShapeError("max_pool2d", "input must be [N, C, H, W]", [("x", x)])
    k = _ints("max_pool2d", attrs["kernel_size"], 2, "kernel_size")
    s = _ints("max_pool2d", attrs.get("stride") or k, 2, "stride")
    p = _ints("max_pool2d", attrs.get("padding", 0), 2, "padding")
    d = _ints("max_pool2d", attrs.get("dilation", 1), 2, "dilation")
    if any(p[i] > k[i] // 2 for i in range(2)):
        raise ShapeError("max_pool2d", f"padding {p} must be at most half the kernel {k}", [("x", x)])
    hw = [pool_out("max_pool2d", x.shape[2 + i], k[i], s[i], p[i], d[i], attrs.get("ceil_mode", False))
          for i in range(2)]
    return TensorType(x.dtype, (x.shape[0], x.shape[1], *hw))


@rule("adaptive_avg_pool2d")
def _adaptive_avg_pool2d(types, attrs):
    (x,) = types
    if x.rank not in (3, 4):
        raise ShapeError("adaptive_avg_pool2d", "input must be [N, C, H, W] or [C, H, W]", [("x", x)])
    out = _ints("adaptive_avg_pool2d", attrs["output_size"], 2, "output_size")
    for i in range(2):
        if not isinstance(x.shape[x.rank - 2 + i], int):
            raise UnsupportedOpError("adaptive_avg_pool2d on a symbolic spatial dim is not supported")
    return TensorType(x.dtype, x.shape[:-2] + tuple(out))


# ----------------------------------------------------------------------------- shape ops


@rule("reshape")
def _reshape(types, attrs):
    (x,) = types
    shape = [norm_dim(d) for d in attrs["shape"]]
    if sum(1 for d in shape if d == -1) > 1:
        raise ShapeError("reshape", f"at most one -1 allowed in {shape}", [("x", x)])
    total = _numel(x.shape)
    known = _numel([d for d in shape if d != -1])
    if -1 in shape:
        rest = sympy.simplify(total / known)
        if not (rest.is_integer or (rest.is_Integer)):
            raise ShapeError("reshape", f"cannot infer -1 in {shape}: {total} is not a multiple of {known}",
                             [("x", x)])
        shape = [norm_dim(rest) if d == -1 else d for d in shape]
    elif not dims_equal(total, known):
        raise ShapeError("reshape", f"element count differs: input has {total}, target shape "
                                    f"[{', '.join(map(str, shape))}] has {known}", [("x", x)])
    return TensorType(x.dtype, tuple(shape))


@rule("permute")
def _permute(types, attrs):
    (x,) = types
    dims = [_axis("permute", d, x.rank, x) for d in attrs["dims"]]
    if sorted(dims) != list(range(x.rank)):
        raise ShapeError("permute", f"dims {attrs['dims']} are not a permutation of rank {x.rank}", [("x", x)])
    return TensorType(x.dtype, tuple(x.shape[d] for d in dims))


@rule("expand")
def _expand(types, attrs):
    (x,) = types
    shape = list(attrs["shape"])
    if len(shape) < x.rank:
        raise ShapeError("expand", f"target rank {len(shape)} is below input rank {x.rank}", [("x", x)])
    out = []
    off = len(shape) - x.rank
    for i, d in enumerate(shape):
        src = x.shape[i - off] if i >= off else None
        if isinstance(d, int) and d == -1:
            if src is None:
                raise ShapeError("expand", "-1 is only allowed on existing dims", [("x", x)])
            out.append(src)
        elif src is None or dims_equal(src, d) or dims_equal(src, 1):
            out.append(d)
        else:
            raise ShapeError("expand", f"dim {i} of size {src} cannot expand to {d}", [("x", x)])
    return TensorType(x.dtype, tuple(out))


@rule("select")
def _select(types, attrs):
    (x,) = types
    d = _axis("select", attrs["dim"], x.rank, x)
    idx = attrs["index"]
    size = x.shape[d]
    if isinstance(size, int) and not -size <= idx < size:
        raise ShapeError("select", f"index {idx} out of range for dim {d} of size {size}", [("x", x)])
    if not isinstance(size, int) and idx < 0:
        raise UnsupportedOpError("select with a negative index on a symbolic dim is not supported")
    return TensorType(x.dtype, x.shape[:d] + x.shape[d + 1:])


@rule("slice")
def _slice(types, attrs):
    (x,) = types
    d = _axis("slice", attrs["dim"], x.rank, x)
    size = x.shape[d]
    start, end, step = attrs.get("start"), attrs.get("end"), attrs.get("step", 1)
    if step < 1:
        raise ShapeError("slice", f"step must be positive, got {step}", [("x", x)])
    if isinstance(size, int):
        start, end, _ = slice(start, end, step).indices(size)
        n = max(0, -(-(end - start) // step))
    else:
        start = start or 0
        if start < 0 or (end is not None and end < 0 and end != -1):
            raise UnsupportedOpError("slice with negative bounds on a symbolic dim is not supported")
        if end is None or end >= 2 ** 62:
            n = _ceil_div(norm_dim(sympy.sympify(size) - start), step)
        else:
            raise UnsupportedOpError("slice with a fixed end on a symbolic dim is not supported")
    return TensorType(x.dtype, x.shape[:d] + (n,) + x.shape[d + 1:])


@rule("cat")
def _cat(types, attrs):
    if not types:
        raise ShapeError("cat", "needs at least one tensor")
    t0 = types[0]
    d = _axis("cat", attrs["dim"], t0.rank, t0)
    total = sympy.Integer(0)
    for i, t in enumerate(types):
        if t.rank != t0.rank or any(not dims_equal(a, b) for j, (a, b) in enumerate(zip(t.shape, t0.shape)) if j != d):
            raise ShapeError("cat", f"tensor {i} does not match tensor 0 outside dim {d}",
                             [(f"tensor {j}", u) for j, u in enumerate(types)])
        total = total + sympy.sympify(t.shape[d])
    dtype = t0.dtype
    for t in types[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    return TensorType(dtype, t0.shape[:d] + (norm_dim(total),) + t0.shape[d + 1:])


# ----------------------------------------------------------------------------- dynamic index, dims


@rule("select_dynamic")
def _select_dynamic(types, attrs):
    """x.select(dim, index) where index is an int64 scalar tensor known only at run time."""
    _need("select_dynamic", types, 2, ["x", "index"])
    x, i = types
    d = _axis("select_dynamic", attrs["dim"], x.rank, x)
    if i.rank != 0 or i.dtype not in (torch.int64, torch.int32):
        raise ShapeError("select_dynamic", "index must be an integer scalar", [("x", x), ("index", i)])
    return TensorType(x.dtype, x.shape[:d] + x.shape[d + 1:])


@rule("dim_scalar")
def _dim_scalar(types, attrs):
    """A rank 0 tensor holding the run time value of a (possibly symbolic) dim expression."""
    if types:
        raise ShapeError("dim_scalar", "takes no tensor operands")
    norm_dim(attrs["expr"])
    return TensorType(attrs["dtype"], ())


# ----------------------------------------------------------------------------- control flow
#
# Regions are sub programs stored in attrs. They capture nothing implicitly: every outer value a
# region reads is an explicit operand of the node, so a region is a closed, pure function and the
# node stays a pure op.


def _check_region(op: str, label: str, region, want_inputs: Sequence[TensorType]) -> None:
    from .core import Program

    if not isinstance(region, Program):
        raise ShapeError(op, f"region {label} must be a Program, got {type(region).__name__}")
    region.verify()
    if len(region.inputs) != len(want_inputs):
        raise ShapeError(op, f"region {label} takes {len(region.inputs)} inputs, the node passes "
                             f"{len(want_inputs)}",
                         [(f"{label} %{v.name}", v.type) for v in region.inputs]
                         + [(f"operand {i}", t) for i, t in enumerate(want_inputs)])
    for v, t in zip(region.inputs, want_inputs):
        if not v.type.same_as(t):
            raise ShapeError(op, f"region {label} input %{v.name} is declared {v.type} but receives {t}",
                             [(f"{label} %{v.name}", v.type), ("operand", t)])


def _scalar_bool(op: str, label: str, t: Optional[TensorType]) -> None:
    if t is None or t.dtype != torch.bool or t.rank != 0:
        raise ShapeError(op, f"{label} must be a bool scalar, got {t}", [(label, t)] if t else [])


@rule("cond")
def _cond(types, attrs):
    """if pred: then_branch(*operands) else: else_branch(*operands). Both regions take the same
    operands and must return the same number of values with provably equal types."""
    if not types:
        raise ShapeError("cond", "needs a predicate")
    pred, operands = types[0], list(types[1:])
    _scalar_bool("cond", "predicate", pred)
    if any(t is None for t in operands):
        raise ShapeError("cond", "operands cannot be absent")
    then, other = attrs["then_branch"], attrs["else_branch"]
    _check_region("cond", "then_branch", then, operands)
    _check_region("cond", "else_branch", other, operands)
    if len(then.outputs) != len(other.outputs):
        raise ShapeError("cond", f"then_branch returns {len(then.outputs)} values, else_branch "
                                 f"{len(other.outputs)}")
    if not then.outputs:
        raise ShapeError("cond", "a cond must return at least one value")
    for i, (a, b) in enumerate(zip(then.outputs, other.outputs)):
        if not a.type.same_as(b.type):
            raise ShapeError("cond", f"result {i} has type {a.type} in then_branch but {b.type} in "
                                     "else_branch", [("then", a.type), ("else", b.type)])
    return tuple(v.type for v in then.outputs)


@rule("loop")
def _loop(types, attrs):
    """Carried state through a loop. Operands: the first ``n_carried`` are the initial carried
    values, the rest are loop invariant captures.

    * ``trip_count`` (a dim, may be symbolic): runs ``body`` that many times (zero when not positive).
    * ``cond`` (a region): runs ``body`` while ``cond(*carried, *captures)`` is true.

    ``body(index: i64[], *carried, *captures)`` returns the new carried values, whose types must be
    provably equal to the carried types: a loop never changes a shape.
    """
    n = attrs["n_carried"]
    if n < 1 or n > len(types):
        raise ShapeError("loop", f"n_carried={n} but the node has {len(types)} operands")
    if any(t is None for t in types):
        raise ShapeError("loop", "operands cannot be absent")
    carried, captures = list(types[:n]), list(types[n:])
    trip, cond = attrs.get("trip_count"), attrs.get("cond")
    if (trip is None) == (cond is None):
        raise ShapeError("loop", "exactly one of trip_count and cond must be given")
    if trip is not None:
        norm_dim(trip)
    else:
        _check_region("loop", "cond", cond, carried + captures)
        if len(cond.outputs) != 1:
            raise ShapeError("loop", f"cond must return one value, returns {len(cond.outputs)}")
        _scalar_bool("loop", "cond result", cond.outputs[0].type)
    body = attrs["body"]
    _check_region("loop", "body", body, [TensorType(torch.int64, ())] + carried + captures)
    if len(body.outputs) != n:
        raise ShapeError("loop", f"body returns {len(body.outputs)} values for {n} carried values")
    for i, (v, t) in enumerate(zip(body.outputs, carried)):
        if not v.type.same_as(t):
            raise ShapeError("loop", f"carried value {i} enters the loop as {t} but the body returns "
                                     f"{v.type}", [("carried", t), ("body result", v.type)])
    return tuple(carried)


# ----------------------------------------------------------------------------- ops added 2026-11-24
# (op coverage for every torchvision classifier up to 100M params and Hugging Face style decoders)


def _float_dtype(dtype: torch.dtype) -> torch.dtype:
    return dtype if dtype.is_floating_point else torch.get_default_dtype()


@rule("cos", "sin", "log", "reciprocal", "erf", "expm1")
def _float_unary(types, attrs):
    (t,) = types
    return TensorType(_float_dtype(t.dtype), t.shape)


@rule("abs")
def _abs(types, attrs):
    (t,) = types
    if t.dtype == torch.bool:
        raise ShapeError("abs", "abs of a bool tensor is not defined", [("x", t)])
    return t


for _op in ("pow", "fmod", "remainder", "maximum", "minimum"):
    RULES[_op] = _binary(_op)


@rule("cast")
def _cast(types, attrs):
    """Dtype conversion. Float to integer truncates toward zero, anything to bool is ``!= 0``."""
    (t,) = types
    if not isinstance(attrs.get("dtype"), torch.dtype):
        raise ShapeError("cast", f"dtype must be a torch dtype, got {attrs.get('dtype')!r}", [("x", t)])
    return TensorType(attrs["dtype"], t.shape)


@rule("cumsum")
def _cumsum(types, attrs):
    (t,) = types
    _axis("cumsum", attrs["dim"], t.rank, t)
    dtype = t.dtype if t.dtype.is_floating_point else torch.int64
    return TensorType(dtype, t.shape)


@rule("triu", "tril")
def _tri(types, attrs):
    (t,) = types
    if t.rank < 2:
        raise ShapeError("triu/tril", "input must have rank 2 or more", [("x", t)])
    if not isinstance(attrs.get("diagonal", 0), int):
        raise ShapeError("triu/tril", "diagonal must be an int", [("x", t)])
    return t


@rule("vector_norm")
def _vector_norm(types, attrs):
    (t,) = types
    if not t.dtype.is_floating_point:
        raise ShapeError("vector_norm", "needs a floating point input", [("x", t)])
    ord_ = attrs.get("ord", 2.0)
    if not isinstance(ord_, (int, float)) or isinstance(ord_, bool):
        raise ShapeError("vector_norm", f"ord must be a number, got {ord_!r}", [("x", t)])
    return _reduce("sum")([t], {"dims": attrs.get("dims"), "keepdim": attrs.get("keepdim", False)})


@rule("group_norm")
def _group_norm(types, attrs):
    names = ["x", "weight", "bias"]
    _need("group_norm", types, 1, names)
    x = types[0]
    g = attrs["num_groups"]
    if x.rank < 2:
        raise ShapeError("group_norm", "input must be [N, C, ...]", _labels(types, names))
    c = x.shape[1]
    if not isinstance(c, int):
        raise UnsupportedOpError("group_norm over a symbolic channel dim is not supported")
    if not isinstance(g, int) or g < 1 or c % g:
        raise ShapeError("group_norm", f"channels {c} are not divisible by num_groups {g}", _labels(types, names))
    for t, n in zip(types[1:], names[1:]):
        if t is not None and (t.rank != 1 or not dims_equal(t.shape[0], c)):
            raise ShapeError("group_norm", f"{n} must be [C] with C = {c}", _labels(types, names))
    return x


@rule("rms_norm")
def _rms_norm(types, attrs):
    names = ["x", "weight"]
    _need("rms_norm", types, 1, names)
    x, w = types
    ns = tuple(attrs["normalized_shape"])
    if x.rank < len(ns) or not all(dims_equal(a, b) for a, b in zip(x.shape[x.rank - len(ns):], ns)):
        raise ShapeError("rms_norm", f"trailing dims of the input do not match normalized_shape {list(ns)}",
                         _labels(types, names))
    if w is not None and not (w.rank == len(ns) and all(dims_equal(a, b) for a, b in zip(w.shape, ns))):
        raise ShapeError("rms_norm", f"weight must have shape {list(ns)}", _labels(types, names))
    if not isinstance(attrs.get("eps"), float):
        raise ShapeError("rms_norm", "eps must be a float (the importer resolves eps=None)", _labels(types, names))
    return x if w is None else TensorType(result_dtype(x, w), x.shape)


@rule("avg_pool2d")
def _avg_pool2d(types, attrs):
    (x,) = types
    if x.rank not in (3, 4):
        raise ShapeError("avg_pool2d", "input must be [N, C, H, W] or [C, H, W]", [("x", x)])
    k = _ints("avg_pool2d", attrs["kernel_size"], 2, "kernel_size")
    s = _ints("avg_pool2d", attrs.get("stride") or k, 2, "stride")
    p = _ints("avg_pool2d", attrs.get("padding", 0), 2, "padding")
    if any(p[i] > k[i] // 2 for i in range(2)):
        raise ShapeError("avg_pool2d", f"padding {p} must be at most half the kernel {k}", [("x", x)])
    div = attrs.get("divisor_override")
    if div is not None and (not isinstance(div, int) or div <= 0):
        raise ShapeError("avg_pool2d", f"divisor_override must be a positive int, got {div!r}", [("x", x)])
    hw = [pool_out("avg_pool2d", x.shape[x.rank - 2 + i], k[i], s[i], p[i], 1, attrs.get("ceil_mode", False))
          for i in range(2)]
    return TensorType(x.dtype, x.shape[:-2] + tuple(hw))


@rule("adaptive_max_pool2d")
def _adaptive_max_pool2d(types, attrs):
    (x,) = types
    if x.rank not in (3, 4):
        raise ShapeError("adaptive_max_pool2d", "input must be [N, C, H, W] or [C, H, W]", [("x", x)])
    out = _ints("adaptive_max_pool2d", attrs["output_size"], 2, "output_size")
    for i in range(2):
        if not isinstance(x.shape[x.rank - 2 + i], int):
            raise UnsupportedOpError("adaptive_max_pool2d on a symbolic spatial dim is not supported")
    return TensorType(x.dtype, x.shape[:-2] + tuple(out))


@rule("pad")
def _pad(types, attrs):
    """Constant padding. ``pads`` lists (before, after) pairs starting from the LAST dim, as F.pad."""
    (x,) = types
    pads = list(attrs["pads"])
    if len(pads) % 2 or len(pads) // 2 > x.rank or not all(isinstance(v, int) for v in pads):
        raise ShapeError("pad", f"pads {pads} must be (before, after) int pairs for at most {x.rank} dims",
                         [("x", x)])
    shape = list(x.shape)
    for i in range(len(pads) // 2):
        d = x.rank - 1 - i
        new = norm_dim(sympy.sympify(shape[d]) + pads[2 * i] + pads[2 * i + 1])
        if isinstance(new, int) and new < 0:
            raise ShapeError("pad", f"negative padding removes more than dim {d} holds", [("x", x)])
        if not isinstance(new, int) and (pads[2 * i] < 0 or pads[2 * i + 1] < 0):
            raise UnsupportedOpError("negative padding on a symbolic dim is not supported")
        shape[d] = new
    return TensorType(x.dtype, tuple(shape))


def _index_dtype(op: str, t: TensorType):
    if t.dtype not in (torch.int64, torch.int32):
        if t.dtype in (torch.bool, torch.uint8):
            raise UnsupportedOpError(f"{op} with a boolean mask has a data dependent shape and is not supported")
        raise ShapeError(op, f"index must be int64 or int32, got {t.dtype}", [("index", t)])


@rule("index")
def _index(types, attrs):
    """x[i0, i1, ...] with integer index tensors (an absent index keeps its dim). PyTorch advanced
    indexing: the index shapes broadcast; if the indexed dims are adjacent the broadcast shape
    replaces them in place, otherwise it goes first."""
    if not types or types[0] is None:
        raise ShapeError("index", "needs a tensor to index")
    x, idx = types[0], list(types[1:])
    if len(idx) > x.rank:
        raise ShapeError("index", f"{len(idx)} indices for a rank {x.rank} tensor", [("x", x)])
    where = [i for i, t in enumerate(idx) if t is not None]
    if not where:
        raise ShapeError("index", "needs at least one index tensor", [("x", x)])
    for i in where:
        _index_dtype("index", idx[i])
    b = broadcast("index", [idx[i] for i in where], [f"index {i}" for i in where])
    if where == list(range(where[0], where[-1] + 1)):
        before = tuple(x.shape[i] for i in range(where[0]))
        after = tuple(x.shape[i] for i in range(where[-1] + 1, x.rank))
        return TensorType(x.dtype, before + b + after)
    kept = tuple(x.shape[i] for i in range(x.rank) if i >= len(idx) or idx[i] is None)
    return TensorType(x.dtype, b + kept)


@rule("index_select")
def _index_select(types, attrs):
    _need("index_select", types, 2, ["x", "index"])
    x, i = types
    d = _axis("index_select", attrs["dim"], x.rank, x)
    _index_dtype("index_select", i)
    if i.rank > 1:
        raise ShapeError("index_select", "index must be rank 0 or 1", [("x", x), ("index", i)])
    n = i.shape[0] if i.rank == 1 else 1
    return TensorType(x.dtype, x.shape[:d] + (n,) + x.shape[d + 1:])


@rule("gather")
def _gather(types, attrs):
    _need("gather", types, 2, ["x", "index"])
    x, i = types
    d = _axis("gather", attrs["dim"], x.rank, x)
    _index_dtype("gather", i)
    if i.rank != x.rank:
        raise ShapeError("gather", "index must have the same rank as the input", [("x", x), ("index", i)])
    for j in range(x.rank):
        if j != d and isinstance(i.shape[j], int) and isinstance(x.shape[j], int) and i.shape[j] > x.shape[j]:
            raise ShapeError("gather", f"index dim {j} is larger than the input dim", [("x", x), ("index", i)])
    return TensorType(x.dtype, i.shape)


@rule("slice_scatter")
def _slice_scatter(types, attrs):
    """x with x[start:end:step] along dim replaced by src."""
    _need("slice_scatter", types, 2, ["x", "src"])
    x, src = types
    want = _slice([x], attrs)
    if not want.same_as(TensorType(want.dtype, src.shape)):
        raise ShapeError("slice_scatter", f"src must have the slice's shape {want}", [("x", x), ("src", src)])
    return x


@rule("select_scatter")
def _select_scatter(types, attrs):
    """x with x.select(dim, index) replaced by src."""
    _need("select_scatter", types, 2, ["x", "src"])
    x, src = types
    want = _select([x], attrs)
    if not want.same_as(TensorType(want.dtype, src.shape)):
        raise ShapeError("select_scatter", f"src must have the selected shape {want}", [("x", x), ("src", src)])
    return x


# ----------------------------------------------------------------------------- ops added for decoder LLMs
# (2026-11-24, see docs/LLM.md). Each is a new op name: no existing op changed meaning, so a consumer
# that does not know one of these ops fails on its name instead of computing something else.


@rule("flip")
def _flip(types, attrs):
    """x reversed along every dim in ``dims``."""
    (x,) = types
    dims = attrs.get("dims")
    if not isinstance(dims, (list, tuple)) or not all(isinstance(d, int) and not isinstance(d, bool) for d in dims):
        raise ShapeError("flip", f"dims must be a list of ints, got {dims!r}", [("x", x)])
    axes = [_axis("flip", d, x.rank, x) for d in dims]
    if len(set(axes)) != len(axes):
        raise ShapeError("flip", f"dims {list(dims)} name the same dim twice", [("x", x)])
    return x


def _arg_reduce(op):
    def fn(types, attrs):
        (x,) = types
        if x.dtype == torch.bool:
            raise ShapeError(op, f"{op} of a bool tensor is not defined; cast it first", [("x", x)])
        dim = attrs.get("dim")
        keep = bool(attrs.get("keepdim", False))
        if dim is None:
            return TensorType(torch.int64, (1,) * x.rank if keep else ())
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ShapeError(op, f"dim must be an int or None, got {dim!r}", [("x", x)])
        d = _axis(op, dim, x.rank, x)
        if x.rank and dims_equal(x.shape[d], 0):
            raise ShapeError(op, f"{op} over an empty dim is not defined", [("x", x)])
        shape = list(x.shape)
        if x.rank:
            if keep:
                shape[d] = 1
            else:
                del shape[d]
        return TensorType(torch.int64, tuple(shape))
    return fn


RULES["argmax"] = _arg_reduce("argmax")
RULES["argmin"] = _arg_reduce("argmin")


@rule("sort")
def _sort(types, attrs):
    """(values, indices) of x sorted along dim. ``stable`` keeps equal values in input order; NaN
    sorts as the largest value, as in PyTorch."""
    (x,) = types
    if x.dtype == torch.bool:
        raise ShapeError("sort", "sorting a bool tensor is not supported; cast it first", [("x", x)])
    _axis("sort", attrs.get("dim", -1), x.rank, x)
    return x, TensorType(torch.int64, x.shape)


@rule("topk")
def _topk(types, attrs):
    """(values, indices) of the k largest (or smallest) entries along dim, in sorted order."""
    (x,) = types
    if x.dtype == torch.bool:
        raise ShapeError("topk", "topk of a bool tensor is not supported; cast it first", [("x", x)])
    k = attrs.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ShapeError("topk", f"k must be a non negative int, got {k!r}", [("x", x)])
    d = _axis("topk", attrs.get("dim", -1), x.rank, x)
    size = x.shape[d] if x.rank else 1
    if not isinstance(size, int):
        raise UnsupportedOpError(f"topk along the symbolic dim {size} is not supported: k={k} cannot be "
                                 "proved to fit for every value of the dim")
    if k > size:
        raise ShapeError("topk", f"k={k} is larger than the dim size {size}", [("x", x)])
    shape = list(x.shape)
    if x.rank:
        shape[d] = k
    return TensorType(x.dtype, tuple(shape)), TensorType(torch.int64, tuple(shape))


@rule("index_copy")
def _index_copy(types, attrs):
    """x with x.select(dim, index[j]) replaced by src.select(dim, j) for every j. The index is data
    (for example a decode position), the shapes are not, so one program serves every position."""
    _need("index_copy", types, 3, ["x", "index", "src"])
    x, i, src = types
    names = [("x", x), ("index", i), ("src", src)]
    d = _axis("index_copy", attrs["dim"], x.rank, x)
    if i.dtype not in (torch.int64, torch.int32):
        raise ShapeError("index_copy", f"index must be int64 or int32, got {i.dtype}", names)
    if i.rank != 1:
        raise ShapeError("index_copy", "index must be rank 1", names)
    if src.dtype != x.dtype:
        raise ShapeError("index_copy", f"src dtype {src.dtype} differs from x dtype {x.dtype}", names)
    want = x.shape[:d] + (i.shape[0],) + x.shape[d + 1:]
    if src.rank != x.rank or not all(dims_equal(a, b) for a, b in zip(src.shape, want)):
        raise ShapeError("index_copy", f"src must have shape [{', '.join(map(str, want))}]", names)
    return x


@rule("multinomial")
def _multinomial(types, attrs):
    """Draw ``num_samples`` category indices per row of non negative weights, from a counter based
    generator keyed by the ``seed`` attribute and an optional run time ``offset`` (an int64 scalar,
    for example the decode step). The same seed and offset always give the same draws."""
    if not types or types[0] is None:
        raise ShapeError("multinomial", "needs a weights tensor")
    if len(types) > 2:
        raise ShapeError("multinomial", f"takes weights and an optional offset, got {len(types)} operands")
    p = types[0]
    off = types[1] if len(types) == 2 else None
    if not p.dtype.is_floating_point or p.rank not in (1, 2):
        raise ShapeError("multinomial", "weights must be a float tensor of rank 1 or 2", [("weights", p)])
    if off is not None and (off.rank != 0 or off.dtype not in (torch.int64, torch.int32)):
        raise ShapeError("multinomial", "offset must be an integer scalar", [("weights", p), ("offset", off)])
    seed, n = attrs.get("seed"), attrs.get("num_samples", 1)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 64:
        raise ShapeError("multinomial", f"seed must be an int in [0, 2**64), got {seed!r}", [("weights", p)])
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ShapeError("multinomial", f"num_samples must be a positive int, got {n!r}", [("weights", p)])
    if not attrs.get("replacement", True) and n > 1:
        raise UnsupportedOpError("multinomial without replacement and num_samples > 1 is not supported")
    return TensorType(torch.int64, p.shape[:-1] + (n,))


@rule("amax", "amin")
def _amax(types, attrs):
    """Max (or min) over ``dims`` (all dims when empty or None); NaN propagates, as in PyTorch."""
    (x,) = types
    dims = attrs.get("dims")
    for d in (range(x.rank) if not dims else dims):
        if x.rank and dims_equal(x.shape[_axis("amax", d, x.rank, x)], 0):
            raise ShapeError("amax/amin", "a max or min over an empty dim is not defined", [("x", x)])
    return _reduce("sum")([x], {"dims": dims, "keepdim": attrs.get("keepdim", False)})


@rule("random")
def _random(types, attrs):
    """A tensor drawn from the keyed counter based generator (truestar.rng, docs/RANDOM.md).

    Operands: the key (an int64 scalar) and any number of stream values (int64 scalars, for example
    the index of each enclosing loop). Attributes: ``dist`` (uniform, normal, bernoulli, randint),
    ``shape``, ``dtype``, ``site`` (a 64 bit int, the hash of ``site_name``), ``site_name`` and the
    distribution's parameters: ``p`` (bernoulli), ``low`` and ``high`` (randint). The value of every
    element is a pure function of (key, site, streams, flat index): the same bits on every backend."""
    from .. import rng

    names = [("key", t) for t in types[:1]] + [(f"stream {i}", t) for i, t in enumerate(types[1:])]
    if not types:
        raise ShapeError("random", "needs a key operand (an int64 scalar)")
    for label, t in names:
        if t is None or t.rank != 0 or t.dtype not in (torch.int64, torch.int32):
            raise ShapeError("random", f"{label} must be an int64 scalar", names)
    site = attrs.get("site")
    if not isinstance(site, int) or isinstance(site, bool) or not 0 <= site < 2 ** 64:
        raise ShapeError("random", f"site must be an int in [0, 2**64), got {site!r}", names)
    try:
        rng.check_params(attrs.get("dist"), attrs.get("dtype"), attrs.get("p"), attrs.get("low", 0),
                         attrs.get("high"))
    except ValueError as e:
        raise ShapeError("random", str(e), names) from None
    shape = tuple(attrs.get("shape", ()))
    for d in shape:
        if isinstance(d, int) and d < 0:
            raise ShapeError("random", f"shape {list(shape)} has a negative dim", names)
    return TensorType(attrs["dtype"], shape)


@rule("scatter_add")
def _scatter_add(types, attrs):
    """x with src[p] added at x[p with coordinate dim replaced by index[p]], for every position p of
    index (which has src's shape). The gradient of ``gather``, ``sort``, ``topk`` and
    ``index_select``. Repeated indices add up, in increasing order of their position along dim."""
    _need("scatter_add", types, 3, ["x", "index", "src"])
    x, i, src = types
    names = [("x", x), ("index", i), ("src", src)]
    d = _axis("scatter_add", attrs["dim"], x.rank, x)
    _index_dtype("scatter_add", i)
    if src.dtype != x.dtype:
        raise ShapeError("scatter_add", f"src dtype {src.dtype} differs from x dtype {x.dtype}", names)
    if i.rank != x.rank or src.rank != x.rank:
        raise ShapeError("scatter_add", "index and src must have the rank of x", names)
    if not all(dims_equal(a, b) for a, b in zip(i.shape, src.shape)):
        raise ShapeError("scatter_add", "index and src must have the same shape", names)
    for j in range(x.rank):
        if j == d:
            continue
        a, b = i.shape[j], x.shape[j]
        if isinstance(a, int) and isinstance(b, int):
            if a > b:
                raise ShapeError("scatter_add", f"index dim {j} is larger than the dim of x", names)
        elif not dims_equal(a, b):
            raise ShapeError("scatter_add", f"index dim {j} ({a}) cannot be proved to fit x's dim ({b})", names)
    return x


# ----------------------------------------------------------------------------- convolution and pooling
# gradients (2026-11-24). The gradient rules of conv2d and max_pool2d emit these, so a compiled
# backward reads its operands at strided indices instead of building zero inserted copies.


def _conv_grad_geometry(op: str, attrs):
    return (_ints(op, attrs.get("stride", 1), 2, "stride"), _ints(op, attrs.get("padding", 0), 2, "padding"),
            _ints(op, attrs.get("dilation", 1), 2, "dilation"), int(attrs.get("groups", 1)))


@rule("conv2d_input_grad")
def _conv2d_input_grad(types, attrs):
    """Cotangent of conv2d's input: (g [N, O, OH, OW], weight [O, C/groups, kH, kW]) to [N, C, H, W]."""
    names = ["g", "weight"]
    _need("conv2d_input_grad", types, 2, names)
    g, w = types
    if g.rank != 4 or w.rank != 4:
        raise ShapeError("conv2d_input_grad", "g must be [N, O, OH, OW] and weight [O, C/groups, kH, kW]",
                         _labels(types, names))
    s, p, d, groups = _conv_grad_geometry("conv2d_input_grad", attrs)
    hw = _ints("conv2d_input_grad", attrs["input_size"], 2, "input_size")
    if not dims_equal(g.shape[1], w.shape[0]):
        raise ShapeError("conv2d_input_grad", f"g channels {g.shape[1]} != weight out channels {w.shape[0]}",
                         _labels(types, names))
    for i in range(2):
        want = pool_out("conv2d_input_grad", hw[i], w.shape[2 + i], s[i], p[i], d[i], False) \
            if isinstance(w.shape[2 + i], int) else None
        if want is not None and isinstance(g.shape[2 + i], int) and want != g.shape[2 + i]:
            raise ShapeError("conv2d_input_grad", f"input size {hw} does not give the g spatial sizes",
                             _labels(types, names))
    c = norm_dim(sympy.sympify(w.shape[1]) * groups)
    return TensorType(result_dtype(g, w), (g.shape[0], c, hw[0], hw[1]))


@rule("conv2d_weight_grad")
def _conv2d_weight_grad(types, attrs):
    """Cotangent of conv2d's weight: (x [N, C, H, W], g [N, O, OH, OW]) to [O, C/groups, kH, kW]."""
    names = ["x", "g"]
    _need("conv2d_weight_grad", types, 2, names)
    x, g = types
    if x.rank != 4 or g.rank != 4:
        raise ShapeError("conv2d_weight_grad", "x must be [N, C, H, W] and g [N, O, OH, OW]",
                         _labels(types, names))
    s, p, d, groups = _conv_grad_geometry("conv2d_weight_grad", attrs)
    k = _ints("conv2d_weight_grad", attrs["kernel_size"], 2, "kernel_size")
    if not dims_equal(x.shape[0], g.shape[0]):
        raise ShapeError("conv2d_weight_grad", "x and g batch sizes differ", _labels(types, names))
    if isinstance(x.shape[1], int) and x.shape[1] % groups:
        raise ShapeError("conv2d_weight_grad", f"channels {x.shape[1]} not divisible by groups {groups}",
                         _labels(types, names))
    cg = norm_dim(sympy.sympify(x.shape[1]) / groups)
    return TensorType(result_dtype(x, g), (g.shape[1], cg, k[0], k[1]))


@rule("max_pool2d_grad")
def _max_pool2d_grad(types, attrs):
    """Cotangent of max_pool2d's input: (x, g with the pooled shape) to the shape of x."""
    names = ["x", "g"]
    _need("max_pool2d_grad", types, 2, names)
    x, g = types
    y = _max_pool2d([x], attrs)
    if g.rank != 4 or not all(dims_equal(a, b) for a, b in zip(g.shape, y.shape)):
        raise ShapeError("max_pool2d_grad", f"g must have the pooled shape {list(y.shape)}",
                         _labels(types, names))
    return TensorType(x.dtype, x.shape)

@rule("mla_attention")
def _mla_attention(types, attrs):
    """Fused MLA latent attention (truestar.mla.latent): q_nope [B, H, T, dn], q_rope [B, H, T, dr],
    caches [B, S, r] and [B, S, dr], new latents [B, T, r] and [B, T, dr], pos [T], W_uk [H, dn, r],
    W_uv [H, dv, r]; the output is [B, H, T, dv] in q_nope's dtype."""
    names = ["q_nope", "q_rope", "cc", "rc", "lat", "kr", "pos", "w_uk", "w_uv"]
    _need("mla_attention", types, 9, names)
    qn, qr, cc, rc, lat, kr, pos, wk, wv = types
    if (qn.rank, qr.rank, cc.rank, rc.rank, lat.rank, kr.rank, pos.rank, wk.rank, wv.rank) != (4, 4, 3, 3, 3, 3, 1, 3, 3):
        raise ShapeError("mla_attention", "ranks must be 4, 4, 3, 3, 3, 3, 1, 3, 3", _labels(types, names))
    if not dims_equal(wk.shape[2], cc.shape[2]) or not dims_equal(wv.shape[2], cc.shape[2]):
        raise ShapeError("mla_attention", "W_uk and W_uv must end in the latent rank", _labels(types, names))
    return TensorType(qn.dtype, (qn.shape[0], qn.shape[1], qn.shape[2], wv.shape[1]))


@rule("kvq_attention")
def _kvq_attention(types, attrs):
    """Decode attention over a quantized KV cache (truestar.kvquant): q [B, Hq, T, D]; k and v data
    [B, Hkv, N, D] (int8, or fp8 stored as its uint8 bits); scales; float tails [B, Hkv, C, D] and
    their live length [1]. The output has q's dtype and shape."""
    names = ["q", "k_data", "k_scale", "v_data", "v_scale", "k_tail", "v_tail", "tail_len", "prefix_len"]
    if len(types) == 11:                  # int4 / nvfp4: E4M3 block scales (uint8 bits)
        names = names + ["k_bscale", "v_bscale"]
    _need("kvq_attention", types, 8, names)
    q, kd, ks, vd, vs, tk, tv, tl = types[:8]
    if q.rank != 4 or kd.rank != 4 or vd.rank != 4 or tk.rank != 4 or tv.rank != 4:
        raise ShapeError("kvq_attention", "q, the caches and the tails must be [B, H, N, D]",
                         _labels(types, names))
    low = attrs.get("fmt") in ("int4", "nvfp4")
    if low and (len(types) < 11 or types[9] is None or types[10] is None):
        raise ShapeError("kvq_attention", "a 4 bit cache needs its E4M3 block scales k_bscale and v_bscale",
                         _labels(types, names))
    if (not low and not dims_equal(q.shape[-1], kd.shape[-1])) or not dims_equal(kd.shape[-1], vd.shape[-1]):
        raise ShapeError("kvq_attention", "q, k and v head dims differ", _labels(types, names))
    if not dims_equal(kd.shape[2], vd.shape[2]):
        raise ShapeError("kvq_attention", "k and v cache lengths differ", _labels(types, names))
    if kd.dtype not in (torch.int8, torch.uint8) or vd.dtype != kd.dtype:
        raise ShapeError("kvq_attention", "cache data must be int8 (or fp8 bits as uint8)",
                         _labels(types, names))
    return TensorType(q.dtype, tuple(q.shape))
