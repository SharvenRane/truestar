"""Kernel rules added with the op coverage work (2026-11-24, docs/MODEL_COVERAGE.md, docs/COMPILER.md).

Before these, every op below ran on the float64 CPU reference (a fallback) in the compiled programs
of the coverage list: the causal mask of Llama, Qwen2 and Whisper (``triu``), Swin's window padding
and shifted window mask (``pad``, ``slice_scatter``), group and RMS norms, the 3D and unfold padding.

* ``triu`` and ``tril``: ``where(col - row >= k, x, 0)`` with the kernel IR's index values.
* ``pad`` (constant): a strided read of the input at ``i - before`` along every padded dim, with the
  pad value where the read falls outside (the same load convolution uses for zero padding).
* ``flip``: a read at ``size - 1 - i`` (fixed sizes).
* ``slice_scatter`` (step 1) and ``select_scatter``: ``where(i in the slice, src at i - start, x)``.
* ``pow`` with any constant exponent: an integer exponent up to 16 is a product chain (binary
  powering); another exponent is ``exp(c log x)`` for x > 0, with the reference's values at 0 and
  below (0 or inf at 0, NaN below 0 for a non integer exponent).

``group_norm``, ``rms_norm`` and ``vector_norm`` are compiled through the lowerings autodiff uses
(ir/grad_rules_extra.py, each equal to the fused op to rounding): ``lower_composites`` rewrites them
into primitive ops, which all have kernel rules, before the program is lowered.
"""

from __future__ import annotations

import math
from typing import List

import torch

from .. import kernel as K
from ..ir.core import Program
from ..kernel import ir as kir
from .conv_rules import aff
from .lower import NotLowerable, _num, bidx, rule

# ----------------------------------------------------------------------------- masks


@rule("triu", "tril")
def _tri(L, node):
    x = L.operand(node.inputs[0])
    shape = L.shape_of(node.output)
    if len(shape) < 2:
        raise NotLowerable("triu and tril need at least two dims")
    k = int(node.attrs.get("diagonal", 0))
    upper = node.op == "triu"

    def fn(idx, lo):
        r, c = idx[-2], idx[-1]
        diff = K.index(c) - K.index(r)
        keep = (diff >= float(k)) if upper else (diff <= float(k))
        return K.where(keep, _num(lo(x, idx), x.kind), 0.0)

    return L.new(shape, "f", fn, (x,), node=node)


# ----------------------------------------------------------------------------- data movement


def _shifted(a, offset: int):
    """The index ``a + offset`` as a strided index even when offset is 0, so a load of a buffer whose
    dim differs from the axis (padding, a scatter source) checks its bounds and reads the pad value
    outside."""
    if isinstance(a, kir.Axis):
        return kir.Strided([kir.Term(a, 1)], offset, 1)
    return aff([(a, 1)], offset)


@rule("pad")
def _pad(L, node):
    x = L.strided_operand(node.inputs[0])
    pads = [int(p) for p in node.attrs["pads"]]
    value = float(node.attrs.get("value", 0.0) or 0.0)
    shape = L.shape_of(node.output)
    rank = len(shape)
    before = [0] * rank
    padded = [False] * rank
    for i in range(len(pads) // 2):
        before[rank - 1 - i] = pads[2 * i]
        padded[rank - 1 - i] = bool(pads[2 * i] or pads[2 * i + 1])

    def fn(idx, lo):
        read = tuple(_shifted(a, -b) if p else a for a, b, p in zip(idx, before, padded))
        return lo(x, read, value)

    return L.new(shape, "f", fn, (x,), node=node)


@rule("flip")
def _flip(L, node):
    x = L.strided_operand(node.inputs[0])
    shape = L.shape_of(node.output)
    dims = {d % len(shape) for d in node.attrs["dims"]}
    for d in dims:
        if not isinstance(shape[d], int):
            raise NotLowerable("flip of a symbolic dim has no kernel form yet")

    def fn(idx, lo):
        read = tuple(aff([(a, -1)], shape[d] - 1) if d in dims else a for d, a in enumerate(idx))
        return lo(x, read)

    return L.new(shape, "f", fn, (x,), node=node)


def _bounds(size, start, end):
    """Normalized [start, end) of a step 1 slice; symbolic sizes take a plain start and end None."""
    if isinstance(size, int):
        s, e, _ = slice(start, end, 1).indices(size)
        return s, e
    if (start or 0) < 0 or (end is not None and end < 0):
        raise NotLowerable("a negative slice bound over a symbolic dim")
    return start or 0, end


@rule("slice_scatter")
def _slice_scatter(L, node):
    xv, sv = node.inputs
    x = L.operand(xv)
    src = L.strided_operand(sv)
    a = node.attrs
    shape = L.shape_of(node.output)
    d = a["dim"] % len(shape)
    step = int(a.get("step", 1) or 1)
    if step != 1:
        # a strided write of one entry over a fixed size dim (Qwen3.5's rotary sections at small head
        # widths, P60) is a write at one index; more entries have no kernel form yet (an exact
        # division read gave wrong values in the first call check, 2026-11-24) and fall back
        size = xv.type.shape[d]
        if not isinstance(size, int):
            raise NotLowerable("slice_scatter with a step other than 1 over a symbolic dim")
        pos = list(range(*slice(a.get("start"), a.get("end"), step).indices(size)))
        if len(pos) != 1:
            raise NotLowerable(f"slice_scatter with a step other than 1 writing {len(pos)} entries (kernel form: 1)")

        def fn_step(idx, lo):
            read = tuple(_shifted(e, -pos[0]) if k == d else e for k, e in enumerate(idx))
            return K.where(K.index(idx[d]) == float(pos[0]), _num(lo(src, read, 0.0), src.kind), _num(lo(x, idx), x.kind))

        return L.new(shape, "f", fn_step, (x, src), node=node)
    start, end = _bounds(xv.type.shape[d], a.get("start"), a.get("end"))

    def fn(idx, lo):
        i = K.index(idx[d])
        inside = i >= float(start)
        if end is not None:
            inside = inside & (i < float(end))
        read = tuple(_shifted(e, -start) if k == d else e for k, e in enumerate(idx))
        return K.where(inside, _num(lo(src, read, 0.0), src.kind), _num(lo(x, idx), x.kind))

    return L.new(shape, "f", fn, (x, src), node=node)


@rule("select_scatter")
def _select_scatter(L, node):
    xv, sv = node.inputs
    x, src = L.operand(xv), L.operand(sv)
    shape = L.shape_of(node.output)
    d = node.attrs["dim"] % len(shape)
    index = int(node.attrs["index"])
    if index < 0:
        if not isinstance(shape[d], int):
            raise NotLowerable("a negative select_scatter index over a symbolic dim")
        index += shape[d]

    def fn(idx, lo):
        at = K.eq(K.index(idx[d]), float(index))
        read = tuple(e for k, e in enumerate(idx) if k != d)
        return K.where(at, _num(lo(src, read), src.kind), _num(lo(x, idx), x.kind))

    return L.new(shape, "f", fn, (x, src), node=node)


# ----------------------------------------------------------------------------- maximum and minimum


@rule("maximum", "minimum")
def _extreme(L, node):
    """The larger (smaller) operand; a NaN in either propagates, as the reference (torch.maximum)."""
    a, b = (L.operand(v) for v in node.inputs)
    shape = L.shape_of(node.output)
    larger = node.op == "maximum"

    def fn(idx, lo):
        va = _num(lo(a, bidx(a.shape, shape, idx)), a.kind)
        vb = _num(lo(b, bidx(b.shape, shape, idx)), b.kind)
        pick = (va > vb) if larger else (va < vb)
        return K.where(pick | K.ne(va, va), va, vb)

    return L.new(shape, "f", fn, (a, b), node=node)


# ----------------------------------------------------------------------------- pow


def _int_power(x, n: int):
    """x ** n for an integer 1 <= n <= 16 by binary powering (at most 7 products)."""
    result, base = None, x
    while n:
        if n & 1:
            result = base if result is None else result * base
        n >>= 1
        if n:
            base = base * base
    return result


@rule("pow")
def _pow(L, node):
    a, b = node.inputs
    xb = L.operand(b)
    xa = L.operand(a)
    if (xb.const is None and xa.const is not None and a.type.rank == 0 and float(xa.const) > 0.0
            and xb.kind == "f"):
        # c ** x for a positive constant base (the rotary inverse frequencies 10000 ** (2i / d)):
        # exp(x * ln c). It took the float64 CPU fallback on every call before (2026-11-25).
        lc = math.log(float(xa.const))
        shape = L.shape_of(node.output)
        return L.new(shape, "f", lambda idx, lo: K.exp(_num(lo(xb, bidx(xb.shape, shape, idx)), xb.kind) * lc),
                     (xb,), node=node)
    if xb.const is None or b.type.rank != 0:
        raise NotLowerable("pow with a tensor exponent has no kernel rule yet")
    c = float(xb.const)
    x = L.operand(a)
    shape = L.shape_of(node.output)

    def base(idx, lo):
        return _num(lo(x, bidx(x.shape, shape, idx)), x.kind)

    if c == 0.0:
        return L.new(shape, "f", lambda idx, lo: kir.Const(1.0) + base(idx, lo) * 0.0, (x,), node=node)
    if c == 0.5:
        return L.new(shape, "f", lambda idx, lo: K.sqrt(base(idx, lo)), (x,), node=node)
    if c == -0.5:
        return L.new(shape, "f", lambda idx, lo: K.rsqrt(base(idx, lo)), (x,), node=node)
    if c.is_integer() and abs(c) <= 16:
        n = int(abs(c))
        if c > 0:
            return L.new(shape, "f", lambda idx, lo: _int_power(base(idx, lo), n), (x,), node=node)
        return L.new(shape, "f", lambda idx, lo: 1.0 / _int_power(base(idx, lo), n), (x,), node=node)

    at_zero = 0.0 if c > 0 else math.inf

    def general(idx, lo):
        v = base(idx, lo)
        pos = v > 0.0
        safe = K.where(pos, v, 1.0)
        y = K.exp(K.log(safe) * c)
        below = math.nan if not c.is_integer() else None
        if below is None:   # an integer exponent beyond 16: |x|^c with the sign of x^c
            mag = K.exp(K.log(K.abs_(K.where(K.eq(v, 0.0), 1.0, v))) * c)
            neg = (-mag) if int(c) % 2 else mag
            return K.where(pos, y, K.where(K.eq(v, 0.0), at_zero, neg))
        return K.where(pos, y, K.where(K.eq(v, 0.0), at_zero, below))

    return L.new(shape, "f", general, (x,), node=node)


# ----------------------------------------------------------------------------- composites by lowering

COMPOSITE_OPS = ("group_norm", "rms_norm", "vector_norm", "adaptive_max_pool2d")


def device_arg_extreme(attrs, args, shape):
    """argmax and argmin on the device's own tensors: the reference's definition (first index of the
    extreme, a NaN counts as the extreme) with the position vector made on the same device."""
    x = args[0]
    largest = attrs.get("_largest", True)
    dim = attrs.get("dim")
    keep = bool(attrs.get("keepdim", False))
    rank = x.dim()
    y = x.reshape(-1) if dim is None else (x if rank == 0 else x.movedim(dim % rank, -1))
    if y.dim() == 0:
        y = y.reshape(1)
    n = y.shape[-1]
    ext = y.amax(dim=-1, keepdim=True) if largest else y.amin(dim=-1, keepdim=True)
    target = y == ext
    if y.dtype.is_floating_point:
        nan = y != y
        target = torch.where(nan.any(dim=-1, keepdim=True), nan, target)
    pos = torch.arange(n, dtype=torch.int64, device=y.device).expand(y.shape)
    out = torch.where(target, pos, torch.full_like(pos, n)).amin(dim=-1)
    if dim is None:
        return out.reshape((1,) * rank) if keep else out.reshape(())
    if keep and rank:
        out = out.unsqueeze(dim % rank)
    return out


DEVICE_INT_IMPLS = {"argmax": lambda attrs, args, shape: device_arg_extreme(dict(attrs, _largest=True), args, shape),
                    "argmin": lambda attrs, args, shape: device_arg_extreme(dict(attrs, _largest=False), args, shape)}


def lower_composites(prog: Program) -> Program:
    """``prog`` with every ``group_norm``, ``rms_norm`` and ``vector_norm`` written as the primitive
    ops of its verified lowering (the node's output value is kept, so readers are unchanged). A node
    whose lowering refuses it (a 0 norm) is kept and takes its usual path."""
    from ..ir.grad_rules import LOWERINGS
    from ..ir.transform_util import Emitter
    if not any(n.op in COMPOSITE_OPS for n in prog.nodes):
        return prog
    e = Emitter(Emitter.names_of(prog), "c_", list(prog.params), dict(prog.param_data))
    nodes: List = []
    for n in prog.nodes:
        if n.op in COMPOSITE_OPS:
            tmp = Emitter(set(e.names), "c_")
            tmp.context = n.source
            try:
                LOWERINGS[n.op](tmp, n)
            except Exception:   # noqa: BLE001  (an unsupported setting: compile the node as it is)
                nodes.append(n)
                continue
            e.names |= tmp.names
            e.params.extend(tmp.params)
            e.param_data.update(tmp.param_data)
            nodes.extend(tmp.nodes)
            continue
        nodes.append(n)
    out = Program(prog.name, prog.inputs, e.params, e.param_data, nodes, prog.outputs, prog.symbols,
                  prog.state, prog.state_init, prog.key)
    return out

