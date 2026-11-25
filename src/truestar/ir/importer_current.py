"""ATen ops of the 2025 to 2026 model families (coverage refresh, 2026-11-24, docs/MODEL_COVERAGE.md).

Measured first: the op coverage sweep on transformers 5.24 (``scripts/op_coverage.py``) listed the
ATen ops the new families and the new transformers masking code produce that the importer did not
take. Each one here is lowered onto existing IR ops, so it gets the float64 reference, the reverse,
forward and batching rules and the compiler's kernels of those ops with no new rule. Each lowering
is written from the op's documented meaning, not from PyTorch's source:

* ``diff``: the input with ``prepend`` and ``append`` joined along the dim, then ``x[1:] - x[:-1]``
  ``n`` times (``ne`` for booleans, as torch). transformers 5 calls it on the position ids of every
  decoder to find packed sequences, so without it no current decoder imports.
* ``square``: ``x * x``. ``floor``: truncation through int64, minus one where that is above ``x``,
  below 2^52 (above it every float is an integer; inf and nan pass through); ``frozen``, so its
  gradient is 0 as torch's. Ministral 3 floors the positions of its Llama 4 style query scale.
* ``bitwise_not``: ``logical_not`` on booleans, ``-x - 1`` on integers.
* ``meshgrid`` and ``tile``: reshapes and expands (no arithmetic).
* ``bucketize``: the count of boundaries below each value (``right=True``: at or below), an integer.
* ``erfinv`` of a value computed from constants only (Gemma 3n's Gaussian top k cut, ``Normal(0,
  1).icdf(0.95)`` evaluated in the forward): the constant subgraph is run by the float64 reference
  interpreter at import and ``torch.special.erfinv`` is applied to the result in float64. Of a value
  that depends on an input it is refused (TS860): no composite of the IR's ops keeps the float64
  bar in the tails, where ``erf`` is flat.
* ``std`` and ``var`` (``unbiased`` or ``correction``): ``sum((x - mean)^2) / (N - correction)``, the
  square root for ``std``; ``multiply`` is ``mul``; ``detach_`` is the identity (no gradient flows
  into a constant); ``_is_all_true`` (torch.distributions' argument validation) is ``all``.
* ``_assert_async`` (torch 2.15 exports that validation as a run time assertion): checked at import
  when its condition is computed from constants, refused (TS860) when it depends on an input.
* String convolution padding (``"valid"``, ``"same"``): resolved to explicit padding, with an
  explicit pad op first when ``"same"`` needs one more element on the right.

An op or setting that cannot be lowered exactly raises with the reason; nothing is approximated.
"""

from __future__ import annotations

import math

import torch

from .core import Value
from .importer import aten
from .importer_extra import _f, _op, _reshape, _t, _unsupported


def _join(c, x: Value, a, d: int) -> Value:
    parts = []
    for key in ("prepend", "append"):
        v = a.get(key)
        if v is not None:
            v = c.tensor(v)
            if v.type.rank != x.type.rank:
                _unsupported(c, f"diff {key} of rank {v.type.rank} for an input of rank {x.type.rank}")
            parts.append((key, v))
    pre = [v for k, v in parts if k == "prepend"]
    post = [v for k, v in parts if k == "append"]
    return x if not parts else _op(c, "cat", *(pre + [x] + post), dim=d)


@aten("aten.diff.default")
def _diff(c, a):
    x = c.tensor(a["self"])
    d = a.get("dim", -1) % x.type.rank
    x = _join(c, x, a, d)
    for _ in range(int(a.get("n", 1))):
        n = x.type.shape[d]
        hi = _op(c, "slice", x, dim=d, start=1, end=None, step=1)
        lo = _op(c, "slice", x, dim=d, start=0, end=n - 1, step=1)
        x = _op(c, "ne", hi, lo) if x.type.dtype == torch.bool else _op(c, "sub", hi, lo)
    return x


@aten("aten.square.default")
def _square(c, a):
    x = c.tensor(a["self"])
    return _op(c, "mul", x, x)


@aten("aten.floor.default")
def _floor(c, a):
    return floor_value(c, c.tensor(a["self"]))


def floor_value(c, x):
    """floor of an IR value (see the module docstring)."""
    if not x.type.dtype.is_floating_point:
        return x
    # truncate through int64 (exact below 2^52, where a float can have a fraction), one less where
    # that is above x; from 2^52 on every float is an integer (inf and nan too) and stays as it is
    t = _op(c, "cast", _op(c, "cast", x, dtype=torch.int64), dtype=x.type.dtype)
    down = _op(c, "sub", t, _op(c, "cast", _op(c, "gt", t, x), dtype=x.type.dtype))
    # x itself from 2^52 on, and at zero (the int64 path would turn -0.0 into +0.0)
    small = _op(c, "logical_and", _op(c, "lt", _op(c, "abs", x), _f(c, 2.0 ** 52, x)), _op(c, "ne", x, _f(c, 0.0, x)))
    return _op(c, "frozen", _op(c, "where", small, down, x))     # derivative 0, as torch's


@aten("aten.bitwise_not.default")
def _bitwise_not(c, a):
    x = c.tensor(a["self"])
    if x.type.dtype == torch.bool:
        return _op(c, "logical_not", x)
    if x.type.dtype.is_floating_point:
        _unsupported(c, "bitwise_not of a float tensor")
    return _op(c, "sub", _op(c, "neg", x), c.const(1, x))


@aten("aten.meshgrid.default", "aten.meshgrid.indexing")
def _meshgrid(c, a):
    ts = [c.tensor(t) for t in a["tensors"]]
    if any(t.type.rank > 1 for t in ts):
        _unsupported(c, "meshgrid of tensors of rank above 1")
    ts = [_reshape(c, t, [1]) if t.type.rank == 0 else t for t in ts]
    sizes = [t.type.shape[0] for t in ts]
    xy = a.get("indexing", "ij") == "xy" and len(ts) >= 2
    order = list(range(len(ts)))
    if xy:   # the first two inputs swap places in the output shape
        order[0], order[1] = 1, 0
    shape = [sizes[i] for i in order]
    outs = []
    for i, t in enumerate(ts):
        axis = order.index(i)
        view = [1] * len(ts)
        view[axis] = sizes[i]
        outs.append(_op(c, "expand", _reshape(c, t, view), shape=shape))
    return tuple(outs)


@aten("aten.tile.default")
def _tile(c, a):
    x = c.tensor(a["self"])
    reps = [int(r) for r in a["dims"]]
    rank = max(len(reps), x.type.rank)
    reps = [1] * (rank - len(reps)) + reps
    if x.type.rank < rank:
        x = _reshape(c, x, [1] * (rank - x.type.rank) + list(x.type.shape))
    for d, r in enumerate(reps):
        if r == 1:
            continue
        shape = list(x.type.shape)
        lead = shape[:d] + [1] + shape[d:]
        x = _op(c, "expand", _reshape(c, x, lead), shape=shape[:d] + [r] + shape[d:])
        x = _reshape(c, x, shape[:d] + [r * shape[d]] + shape[d + 1:])
    return x


@aten("aten.bucketize.Tensor")
def _bucketize(c, a):
    x, b = c.tensor(a["self"]), c.tensor(a["boundaries"])
    if b.type.rank != 1:
        _unsupported(c, "bucketize with boundaries of rank other than 1")
    out = torch.int32 if a.get("out_int32") else torch.int64
    xs = _reshape(c, x, list(x.type.shape) + [1])
    bs = _reshape(c, b, [1] * x.type.rank + [b.type.shape[0]])
    below = _op(c, "ge" if a.get("right") else "gt", xs, bs)      # boundary below (or at) the value
    n = _op(c, "sum", _op(c, "cast", below, dtype=torch.int64), dims=[x.type.rank], keepdim=False)
    return n if out == torch.int64 else _op(c, "cast", n, dtype=out)


def constant_value(c, x: Value):
    """The float64 value of ``x`` when it depends on params and constants only (no program input),
    computed by the reference interpreter over the nodes emitted so far; else None."""
    from .core import Program
    from .reference import run

    if x.kind == "param":
        return c.param_data[x.name]
    producer = {id(n.output): n for n in c.nodes}
    need, seen, stack = [], set(), [x]
    while stack:
        v = stack.pop()
        if v is None or id(v) in seen:
            continue
        seen.add(id(v))
        if v.kind == "param":
            continue
        n = producer.get(id(v))
        if n is None or v.kind == "input" or n.extra_outputs:
            return None
        need.append(n)
        stack.extend(n.inputs)
    if any(not isinstance(d, int) for d in x.type.shape):
        return None
    order = [n for n in c.nodes if any(n is m for m in need)]
    used = {i.name for n in order for i in n.inputs if i is not None and i.kind == "param"}
    params = [p for p in c.params if p.name in used]
    prog = Program("constant", [], params, {k: c.param_data[k] for k in used}, order, [x])
    out = run(prog)
    return out[0] if isinstance(out, (list, tuple)) else out


@aten("aten.erfinv.default", "aten.special_erfinv.default")
def _erfinv(c, a):
    x = c.tensor(a["self"])
    if not x.type.dtype.is_floating_point:
        _unsupported(c, "erfinv of an integer tensor")
    data = constant_value(c, x)
    if data is None:
        _unsupported(c, "erfinv of a value that depends on an input (only a value computed from constants is "
                        "supported: it is folded at import in float64)")
    return _t(c, "erfinv", torch.special.erfinv(data.to(torch.float64)), x.type.dtype)


def _var(c, a, root: bool):
    x = c.tensor(a["self"])
    dim = a.get("dim")
    dims = list(range(x.type.rank)) if dim is None else ([dim] if isinstance(dim, int) else list(dim))
    dims = [d % x.type.rank for d in dims]
    if "correction" in a and a["correction"] is not None:
        corr = float(a["correction"])
    elif "unbiased" in a:
        corr = 1.0 if a["unbiased"] in (None, True) else 0.0
    else:
        corr = 1.0
    sizes = [x.type.shape[d] for d in dims]
    if not all(isinstance(s_, int) for s_ in sizes):
        _unsupported(c, "std or var over a symbolic dim", "TS861")
    n = math.prod(sizes)
    keep = bool(a.get("keepdim", False))
    mu = _op(c, "mean", x, dims=dims, keepdim=True)
    dev = _op(c, "sub", x, mu)
    ss = _op(c, "sum", _op(c, "mul", dev, dev), dims=dims, keepdim=keep)
    v = _op(c, "div", ss, _f(c, max(n - corr, 0.0), x))   # N - correction <= 0 gives inf or nan, as torch
    return _op(c, "sqrt", v) if root else v


for _ov in ("dim", "correction", "default"):
    aten(f"aten.std.{_ov}")(lambda c, a: _var(c, a, True))
    aten(f"aten.var.{_ov}")(lambda c, a: _var(c, a, False))


@aten("aten.multiply.Tensor")
def _multiply(c, a):
    x = c.tensor(a["self"])
    return _op(c, "mul", x, c.operand(a["other"], x))


@aten("aten.detach_.default")
def _detach_(c, a):
    return c.tensor(a["self"])


@aten("aten._is_all_true.default")
def _is_all_true(c, a):
    x = c.tensor(a["self"])
    if x.type.dtype != torch.bool:
        x = _op(c, "ne", x, c.const(0, x))
    return _op(c, "logical_not", _op(c, "any", _op(c, "logical_not", x), dims=None, keepdim=False))


@aten("aten._assert_async.default", "aten._assert_async.msg")
def _assert_async(c, a):
    """A run time assertion that a one element tensor is nonzero. torch 2.15's export emits it for
    torch.distributions' argument validation (``Normal(0, 1).icdf(...)`` in the forward), where
    torch 2.11 left a dead comparison. Its condition is computed from constants there, so it is
    checked at import by the float64 reference interpreter and nothing is emitted; a false one is
    the error the model raises in PyTorch. On a value that depends on an input it is refused: the
    IR has no assertion op, and dropping the check would silently accept inputs torch rejects."""
    from .core import IRError
    from .importer import _describe

    cond = c.tensor(a["self"])
    data = constant_value(c, cond)
    if data is None:
        _unsupported(c, "a run time assertion on a value that depends on an input (only an assertion on "
                        "constants is supported: it is checked at import)")
    if data.numel() != 1:
        raise IRError(f"{_describe(c.node)}: asserts a tensor of {data.numel()} elements; it takes one")
    if not bool(data.reshape(()) != 0):
        msg = a.get("assert_msg") or "assertion failed"
        raise IRError(f"{_describe(c.node)}: the model's assertion fails on its constants: {msg}")
    return None


@aten("aten.conv2d.padding")
def _conv2d_padding(c, a):
    from .importer_extra import _conv_nd
    return _conv_nd(c, c.tensor(a["input"]), c.tensor(a["weight"]), c.tensor(a["bias"]), a["stride"], a["padding"],
                    a["dilation"], a["groups"], 2)


def resolve_string_padding(c, x: Value, w: Value, padding: str, stride, dilation, nd: int):
    """(x, padding ints) for torch's string padding of an ``nd`` D convolution: ``valid`` is no
    padding; ``same`` pads ``dilation * (k - 1)`` in total per dim, the extra element of an odd total
    on the right (as torch), which needs an explicit pad op since convolution padding is symmetric."""
    if padding == "valid":
        return x, [0] * nd
    if padding != "same":
        _unsupported(c, f"convolution padding={padding!r}")
    strides = [stride] * nd if isinstance(stride, int) else list(stride) * (nd if len(stride) == 1 else 1)
    dils = [dilation] * nd if isinstance(dilation, int) else list(dilation) * (nd if len(dilation) == 1 else 1)
    if any(s != 1 for s in strides):
        _unsupported(c, "padding='same' with a stride above 1 (torch refuses it too)")
    ks = list(w.type.shape[2:])
    totals = [d * (k - 1) for d, k in zip(dils, ks)]
    sym = [t // 2 for t in totals]
    if all(t % 2 == 0 for t in totals):
        return x, sym
    pads = []
    for t in reversed(totals):            # pad lists the last dim first
        pads += [0, t % 2]
    return _op(c, "pad", x, pads=pads, value=0.0), sym


from . import importer_text  # noqa: E402,F401  (text, retrieval, speech and vision language sweep)
