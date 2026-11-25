"""Lower a tensor IR program onto the kernel layer's graph, when every op fits.

The kernel layer (``truestar.kernel``) fuses a graph of tensor ops into Triton or numba kernels and
verifies every compiled kernel against its own float64 reference. This module maps tensor IR nodes
onto that graph. An op the kernel graph cannot express raises ``NotLowerable`` naming the op, and
the caller then runs the float64 reference interpreter instead and says so.

The graph gets four extra ops through a subclass, without touching ``truestar.kernel``:
a general ``permute`` (the graph's transpose already carries a full permutation), ``view1`` (a
reshape that only inserts or removes size 1 dims), ``expand`` and ``fill`` (a constant tensor).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import sympy
import torch

from ..ir.core import Program, Value
from ..kernel import ir as kir
from ..kernel.graph import Graph, Tensor as KTensor

_KDTYPE = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32",
           torch.float64: "float64"}


class NotLowerable(Exception):
    """The program uses something the kernel layer cannot express. Carries the reason."""


class _Graph(Graph):
    """The kernel graph plus the shape ops a tensor IR program needs."""

    def view1(self, x: KTensor, shape) -> KTensor:
        return KTensor(self, "view1", (x,), tuple(shape), x.dtype)

    def expand(self, x: KTensor, shape) -> KTensor:
        return KTensor(self, "expand", (x,), tuple(shape), x.dtype)

    def fill(self, shape, value: float, dtype: str) -> KTensor:
        return KTensor(self, "fill", (), tuple(shape), dtype, {"value": float(value)})

    def permute(self, x: KTensor, perm: Sequence[int]) -> KTensor:
        perm = [p % x.ndim for p in perm]
        return KTensor(self, "transpose", (x,), [x.shape[p] for p in perm], x.dtype, {"perm": perm})

    def lower(self, t, idx, mat, root):
        if t.op == "input" or (t.id in mat and t is not root):
            return super().lower(t, idx, mat, root)
        if t.op == "fill":
            return kir.Const(t.attrs["value"])
        if t.op == "view1":
            x = t.args[0]
            outer = [ix for ix, s in zip(idx, t.shape) if s != 1]
            inner, k = [], 0
            for s in x.shape:
                if s == 1:
                    inner.append(0)
                else:
                    inner.append(outer[k])
                    k += 1
            return self.lower(x, tuple(inner), mat, root)
        if t.op == "expand":
            x = t.args[0]
            off = len(t.shape) - len(x.shape)
            inner = [0 if (x.shape[k] == 1 and t.shape[off + k] != 1) else idx[off + k]
                     for k in range(len(x.shape))]
            return self.lower(x, tuple(inner), mat, root)
        return super().lower(t, idx, mat, root)


def _size(d, op: str):
    if isinstance(d, int):
        if d < 1:
            raise NotLowerable(f"{op}: a dim of size {d} (the kernel layer needs sizes of at least 1)")
        return d
    if isinstance(d, sympy.Symbol):
        return kir.Sym(str(d))
    raise NotLowerable(f"{op}: the symbolic dim {d} is an expression, the kernel layer binds plain symbols only")


def _shape(t, op: str):
    return tuple(_size(d, op) for d in t.shape)


def _kdtype(dtype: torch.dtype, what: str) -> str:
    if dtype not in _KDTYPE:
        raise NotLowerable(f"{what} has dtype {dtype}; the kernel layer handles float tensors only")
    return _KDTYPE[dtype]


def _ax(d: int, rank: int) -> int:
    return d % rank if rank else 0


_UNARY = {
    "neg": lambda g, x: -x,
    "exp": lambda g, x: g.exp(x),
    "sqrt": lambda g, x: g.sqrt(x),
    "rsqrt": lambda g, x: g.rsqrt(x),
    "tanh": lambda g, x: g.tanh(x),
    "sigmoid": lambda g, x: g.sigmoid(x),
    "relu": lambda g, x: g.relu(x),
    "abs": lambda g, x: g.abs(x),
    "silu": lambda g, x: g.pointwise(lambda a: a * kir.sigmoid(a), x, name="silu"),
}

_BINARY = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b, "maximum": kir.maximum, "minimum": kir.minimum,
    "lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b, "eq": kir.eq, "ne": kir.ne,
    "logical_and": lambda a, b: a & b, "logical_or": lambda a, b: a | b,
}


def _gelu_tanh(a):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * a * (1.0 + kir.tanh(c * (a + 0.044715 * a * a * a)))


class Lowered:
    """A program lowered onto a kernel graph: the graph, its outputs and the input order."""

    def __init__(self, graph: _Graph, outputs: Dict[str, KTensor], input_names: List[str],
                 param_inputs: Dict[str, torch.Tensor], passthrough: Dict[int, int], n_outputs: int):
        self.graph = graph
        self.outputs = outputs
        self.input_names = input_names
        self.param_inputs = param_inputs
        self.passthrough = passthrough  # output index -> program input index (output is an input)
        self.n_outputs = n_outputs


def lower(program: Program) -> Lowered:
    """Map every node of ``program`` onto the kernel graph, or raise NotLowerable naming the op."""
    g = _Graph()
    env: Dict[int, KTensor] = {}
    names: List[str] = []
    for i, v in enumerate(program.inputs):
        name = f"in{i}"
        env[id(v)] = g.input(name, _shape(v.type, f"input {v.name}"), _kdtype(v.type.dtype, f"input %{v.name}"))
        names.append(name)
    consts: Dict[int, float] = {}
    param_inputs: Dict[str, torch.Tensor] = {}
    for j, p in enumerate(program.params):
        data = program.param_data[p.name]
        if p.type.rank == 0:
            if not p.type.dtype.is_floating_point:
                raise NotLowerable(f"constant @{p.name} has dtype {p.type.dtype}")
            consts[id(p)] = float(data)
        else:
            name = f"p{j}"
            env[id(p)] = g.input(name, _shape(p.type, f"param {p.name}"), _kdtype(p.type.dtype, f"param @{p.name}"))
            param_inputs[name] = data

    def get(v: Optional[Value]):
        if v is None:
            return None
        if id(v) in consts:
            return consts[id(v)]
        return env[id(v)]

    for n in program.nodes:
        if not n.output.type.dtype.is_floating_point and n.op not in ("lt", "le", "gt", "ge", "eq", "ne",
                                                                    "logical_and", "logical_or", "logical_not"):
            raise NotLowerable(f"{n.op} produces {n.output.type}; the kernel layer handles float tensors only")
        if n.extra_outputs:
            raise NotLowerable(f"control flow op {n.op} has no kernel layer form yet")
        args = [get(x) for x in n.inputs]
        out_shape = _shape(n.output.type, n.op)
        op, a = n.op, n.attrs
        if op in _UNARY:
            r = _UNARY[op](g, args[0]) if not isinstance(args[0], float) else None
        elif op in _BINARY:
            r = g.pointwise(_BINARY[op], args[0], args[1], name=op)
        elif op == "logical_not":
            r = g.pointwise(lambda x: ~x, args[0], name="not")
        elif op == "where":
            r = g.pointwise(kir.where, args[0], args[1], args[2], name="where")
        elif op == "clamp":
            lo, hi = a.get("min"), a.get("max")

            def _clamp(x, lo=lo, hi=hi):
                if lo is not None:
                    x = kir.maximum(x, float(lo))
                if hi is not None:
                    x = kir.minimum(x, float(hi))
                return x
            r = g.pointwise(_clamp, args[0], name="clamp")
        elif op == "gelu":
            if a.get("approximate", "none") != "tanh":
                raise NotLowerable("gelu with approximate='none' needs erf, which the kernel IR does not have "
                                   "(the tanh form lowers)")
            r = g.pointwise(_gelu_tanh, args[0], name="gelu_tanh")
        elif op in ("sum", "mean"):
            x = args[0]
            rank = n.inputs[0].type.rank
            dims = a.get("dims")
            dims = sorted({_ax(d, rank) for d in (dims or range(rank))}, reverse=True)
            keep = bool(a.get("keepdim", False))
            if not dims:
                raise NotLowerable(f"{op} of a rank 0 tensor")
            r = x
            for d in dims:
                r = g.sum(r, d, keepdims=keep) if op == "sum" else g.mean(r, d, keepdims=keep)
        elif op == "softmax":
            r = g.softmax(args[0], _ax(a["dim"], n.inputs[0].type.rank))
        elif op == "matmul":
            if n.inputs[0].type.rank < 2 or n.inputs[1].type.rank < 2:
                raise NotLowerable("matmul with a vector operand")
            r = args[0] @ args[1]
        elif op == "linear":
            x, w, b = args
            if n.inputs[1].type.rank != 2 or n.inputs[0].type.rank < 2:
                raise NotLowerable("linear with a vector input")
            r = x @ w.T
            if b is not None:
                r = r + b
        elif op == "layer_norm":
            if len(a["normalized_shape"]) != 1:
                raise NotLowerable("layer_norm over more than one trailing dim")
            r = g.layer_norm(args[0], args[1], args[2], eps=a.get("eps", 1e-5))
        elif op == "attention":
            q, k, v, mask = args
            if mask is not None:
                raise NotLowerable("attention with an explicit mask operand")
            if n.inputs[0].type.rank < 2:
                raise NotLowerable("attention on vectors")
            scale = a.get("scale")
            if scale is None:
                scale = 1.0 / math.sqrt(n.inputs[0].type.shape[-1])
            s = (q @ k.T) * float(scale)
            if a.get("is_causal", False):
                s = g.pointwise(lambda sc, *ix: kir.where(ix[-1] <= ix[-2], sc, kir.NEG_INF), s,
                                with_index=True, name="causal")
            r = g.softmax(s, -1) @ v
        elif op == "permute":
            r = g.permute(args[0], a["dims"])
        elif op == "reshape":
            src = [d for d in n.inputs[0].type.shape if not (isinstance(d, int) and d == 1)]
            dst = [d for d in n.output.type.shape if not (isinstance(d, int) and d == 1)]
            if len(src) != len(dst) or any(sympy.simplify(x - y) != 0 for x, y in zip(src, dst)):
                raise NotLowerable("reshape that merges or splits dims (only size 1 dims may be added or removed)")
            r = g.view1(args[0], out_shape)
        elif op == "expand":
            r = g.expand(args[0], out_shape)
        elif op == "full_like":
            r = g.fill(out_shape, a["value"], _kdtype(n.output.type.dtype, op))
        elif op == "full":
            r = g.fill(out_shape, a["value"], _kdtype(n.output.type.dtype, op))
        else:
            raise NotLowerable(f"IR op {op!r} has no kernel layer lowering yet")
        if r is None:
            raise NotLowerable(f"{op} applied to a constant")
        if isinstance(r, float):
            raise NotLowerable(f"{op} folds to a constant")
        if tuple(r.shape) != out_shape:
            raise NotLowerable(f"internal: {op} lowered to shape {list(r.shape)}, the IR says {list(out_shape)}")
        env[id(n.output)] = r

    outputs: Dict[str, KTensor] = {}
    passthrough: Dict[int, int] = {}
    in_index = {id(v): i for i, v in enumerate(program.inputs)}
    for i, v in enumerate(program.outputs):
        if id(v) in in_index:
            passthrough[i] = in_index[id(v)]
            continue
        if id(v) in consts:
            raise NotLowerable("an output is a constant")
        t = env[id(v)]
        if t.op in ("input", "fill"):
            t = g.pointwise(lambda x: x, t, name="copy")
        if not t.shape:
            raise NotLowerable("a rank 0 output")
        if any(t is o for o in outputs.values()):
            t = g.pointwise(lambda x: x, t, name="copy")
        outputs[f"out{i}"] = t
    if not outputs:
        raise NotLowerable("every output is an input")
    return Lowered(g, outputs, names, param_inputs, passthrough, len(program.outputs))
