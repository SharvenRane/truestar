"""Lower the tensor IR to the kernel layer.

Every IR op that has a kernel rule is turned into one or more *lowered values*. A lowered value
is defined only by its index function: given one kernel IR index per dim, return the expression
for that element (the lowering contract of docs/KERNEL_IR.md, section 4). Fusion is then the
choice of which lowered values become buffers:

* values read by something that is not a kernel (a layout op, a library kernel, control flow, a
  fallback, the program outputs) and values with more than one reader are materialized;
* everything else is inlined into its reader, and the backend scheduler
  (``triton_backend.plan``) is asked whether the fused kernel fits a loop shape. If it does not,
  the nearest reduction producer is materialized and the question is asked again. This is the
  fusion loop of ``kernel.graph.Graph.fuse`` driven from the tensor IR;
* on the CPU backend a reduction is never inlined under another reduction, because the numba
  loop nest would redo the inner reduction for every element of the outer one.

Ops without a kernel rule run on a declared path that the compile report lists: layout ops (views
and copies, no arithmetic), library kernels (convolution and pooling through torch), host scalars
(integer bookkeeping) and the fallback (the float64 reference implementation of the op).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import sympy
import torch

from .. import kernel as K
from ..ir import reference as iref
from ..ir.core import IRError, Node, Program, UnsupportedOpError, Value
from ..kernel import ir as kir
from ..kernel.triton_backend import Unschedulable, plan as gpu_plan


class NotLowerable(Exception):
    """An op (or this use of it) has no kernel form. The reason goes into the compile report."""


# ----------------------------------------------------------------------------- sizes

_SIZE_NAMES: Dict[str, K.Sym] = {}
_SIZE_TAKEN: Dict[str, str] = {}


def size_of(d) -> kir.Size:
    """Kernel size of an IR dim: an int, or one symbol per distinct dim expression."""
    if isinstance(d, int):
        if d < 1:
            raise NotLowerable(f"a dim of size {d} has no kernel form")
        return d
    text = str(d)
    hit = _SIZE_NAMES.get(text)
    if hit is None:
        name = text if text.isidentifier() else "d" + re.sub(r"\W+", "_", text.replace("**", "e").replace(
            "*", "x").replace("+", "p").replace("-", "m").replace("/", "q"))
        base, n = name, 0
        while _SIZE_TAKEN.get(name, text) != text:
            n += 1
            name = f"{base}_{n}"
        _SIZE_TAKEN[name] = text
        hit = _SIZE_NAMES[text] = K.Sym(name)
    return hit


def kind_of(dtype: torch.dtype) -> str:
    """'f' float, 'b' bool, 'i' integer."""
    if dtype.is_floating_point:
        return "f"
    return "b" if dtype == torch.bool else "i"


class DimTable:
    """Every dim expression the runtime needs as a number, evaluated once per set of symbol values."""

    def __init__(self, symbols: Sequence[str]):
        self.symbols = sorted(symbols)
        self._syms = [sympy.Symbol(s, integer=True, positive=True) for s in self.symbols]
        self.exprs: List[Any] = []
        self._index: Dict[str, int] = {}
        self._cache: Dict[Tuple[int, ...], List[int]] = {}

    def add(self, d) -> int:
        """Index of dim ``d`` in the evaluated table (ints are registered too, for uniform access)."""
        key = str(d)
        i = self._index.get(key)
        if i is None:
            if not isinstance(d, int):
                free = {str(s) for s in sympy.sympify(d).free_symbols}
                if not free <= set(self.symbols):
                    raise NotLowerable(f"dim {d} uses symbols {sorted(free - set(self.symbols))} that the "
                                       f"program inputs do not bind")
            i = self._index[key] = len(self.exprs)
            self.exprs.append(d)
            self._cache.clear()
        return i

    def shape(self, dims: Sequence[Any]) -> Tuple[int, ...]:
        return tuple(self.add(d) for d in dims)

    def evaluate(self, values: Tuple[int, ...]) -> List[int]:
        hit = self._cache.get(values)
        if hit is None:
            sub = dict(zip(self._syms, values))
            hit = [d if isinstance(d, int) else int(sympy.sympify(d).subs(sub)) for d in self.exprs]
            self._cache[values] = hit
        return hit


# ----------------------------------------------------------------------------- lowered values


def nominal_float(prog, opts=None) -> Optional[torch.dtype]:
    """The program's own float dtype: the float dtype that holds the most elements among its params
    and inputs (a symbolic dim counts as 1). An import of ``model.double()`` is float64 throughout;
    float32 quantization scales next to float64 weights, or float32 constants next to float64
    inputs, are the minority and so do not decide it. Ties go to the first seen.

    With ``opts.mixed`` (a mixed precision program compiled in a 16 bit working precision, such as an
    autocast training step) the program's own dtype is the working precision: every value declared
    in it is computed in it, and every value declared float32 or float64 (master weights, optimizer
    state, the loss, the loss scale) is a float32 island, whatever holds the most elements."""
    if opts is not None and getattr(opts, "mixed", False):
        return opts.torch_dtype
    weight: Dict[torch.dtype, int] = {}
    for v in list(prog.params) + list(prog.inputs):
        dt = v.type.dtype
        if not dt.is_floating_point:
            continue
        n = 1
        for d in v.type.shape:
            n *= d if isinstance(d, int) else 1     # a symbolic dim adds nothing: concrete sizes decide
        weight[dt] = weight.get(dt, 0) + max(n, 1)
    if not weight:
        return None
    return max(weight, key=lambda d: weight[d])


def island_dtype(opts) -> Optional[str]:
    """The storage and math dtype of this program's islands: float32 in a 16 bit program, float64 in
    a float32 program, none in a float64 program."""
    return {"float16": "float32", "bfloat16": "float32", "float32": "float64"}.get(opts.dtype)


def is_island(opts, nominal: Optional[torch.dtype], dtype: torch.dtype) -> bool:
    """A mixed precision island: a float value the model declares in a wider dtype than the
    program's own, at least as wide as the island dtype. In a float16 or bfloat16 program: a value
    declared in 32 or 64 bits (an explicit ``.float()`` in a ``.double()`` import, or in a model
    imported in float16); it keeps float32 storage and math. In a float32 program imported in
    float32: a value declared float64 (an explicit ``.double()``, such as rotary angles at long
    positions); it keeps float64 storage and math. Every other float value is computed in the
    working precision, as before, so a program without such a declaration is unchanged."""
    isl = island_dtype(opts)
    if isl is None or nominal is None or not dtype.is_floating_point or dtype == nominal:
        return False
    if torch.finfo(nominal).bits > torch.finfo(opts.torch_dtype).bits and opts.dtype == "float32":
        return False     # a .double() import compiled in float32: float64 is every value's dtype
    return torch.finfo(dtype).bits >= torch.finfo(getattr(torch, isl)).bits


def storage_dtype(opts, nominal: Optional[torch.dtype], dtype: torch.dtype) -> torch.dtype:
    """The torch dtype a float value of declared ``dtype`` is stored in."""
    return getattr(torch, island_dtype(opts)) if is_island(opts, nominal, dtype) else opts.torch_dtype


def convention_values(prog, opts) -> set:
    """ids of the float64 values of a float32 program that are float64 by calling convention, not
    because the model declares them so: program inputs declared float64 (a train step's ``lr``, its
    optimizer ``step``, the loss scaler's growth tracker: Python numbers are float64), every value
    computed from one of them, and the scalar constants only such values read. They are computed and
    stored in the working precision, as before float64 islands; a float64 island starts only where
    the model declares float64 (a ``.double()`` cast, a float64 param). Empty for other programs."""
    if island_dtype(opts) != "float64":
        return set()
    f64 = torch.float64
    conv = {id(v) for v in prog.inputs if v.type.dtype == f64}
    if not conv:
        return conv
    readers: Dict[int, List[bool]] = {}
    for node in prog.nodes:
        tainted = any(id(v) in conv for v in node.inputs)
        if tainted:
            conv.update(id(o) for o in node.outputs if o.type.dtype == f64)
        for v in node.inputs:
            readers.setdefault(id(v), []).append(tainted)
    for p in prog.params:   # a float64 scalar constant read only by convention values
        if p.type.dtype == f64 and not p.type.shape and readers.get(id(p)) and all(readers[id(p)]):
            conv.add(id(p))
    return conv


def input_storage_dtypes(prog, opts) -> List[Optional[torch.dtype]]:
    """The storage dtype of each program input: its island dtype or the working dtype for a float
    input (see convention_values), None for others."""
    nom, conv = nominal_float(prog, opts), convention_values(prog, opts)
    return [(opts.torch_dtype if id(v) in conv else storage_dtype(opts, nom, v.type.dtype))
            if v.type.dtype.is_floating_point else None for v in prog.inputs]


class LV:
    """A lowered value: a tensor defined by an index function, or a leaf that already is a buffer."""

    __slots__ = ("id", "shape", "kind", "fn", "args", "heavy", "node", "value", "leaf", "const", "slot",
                 "kslot", "uses", "must_mat", "need_true", "force_mat", "pos", "name", "contract", "fdtype")

    def __init__(self, id_, shape, kind, fn=None, args=(), heavy=False, node=None, value=None, leaf=False,
                 const=None, pos=0.0, name=""):
        self.id = id_
        self.shape = None if shape is None else tuple(shape)   # None: a zero sized leaf (shape_of_leaf)
        self.kind = kind
        self.fn = fn
        self.args = tuple(args)
        self.heavy = heavy
        self.node = node
        self.value = value      # the IR value this is, or None for a step inside one IR op
        self.leaf = leaf
        self.const = const      # Python number for a scalar constant (inlined), else None
        self.slot = None        # runtime slot of the tensor in its declared dtype
        self.kslot = None       # runtime slot of the form kernels read (floats for bool and int)
        self.uses = 0
        self.must_mat = False
        self.need_true = False
        self.force_mat = False
        self.contract = False   # a matrix product (matmul, linear, attention scores)
        self.pos = pos
        self.name = name
        self.fdtype = None      # float storage dtype (str) of a float value: working, or float32 (island)


def bidx(a_shape, out_shape, idx):
    """Index of an operand of shape ``a_shape`` inside a broadcast result indexed by ``idx``."""
    off = len(out_shape) - len(a_shape)
    return tuple(0 if (a_shape[k] == 1 and out_shape[off + k] != 1) else idx[off + k]
                 for k in range(len(a_shape)))


def _num(e, kind):
    return K.where(e, 1.0, 0.0) if kind == "b" else e


def _bool(e, kind):
    return e if kind == "b" else K.ne(e, 0.0)


def _index_expr(a):
    return kir.Index(a) if isinstance(a, kir.Axis) else kir.Const(float(a))


# ----------------------------------------------------------------------------- elementwise tables

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _clamp(x, lo, hi):
    if lo is not None:
        x = K.maximum(x, float(lo))
    if hi is not None:
        x = K.minimum(x, float(hi))
    return x


def _gelu(x, attrs):
    if attrs.get("approximate", "none") == "tanh":
        return 0.5 * x * (1.0 + K.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))
    if "erf" not in kir.UNARY_OPS:
        raise NotLowerable("the kernel IR has no erf")
    return 0.5 * x * (1.0 + kir.Unary("erf", x * (1.0 / math.sqrt(2.0))))


UNARY: Dict[str, Callable] = {
    "neg": lambda x, a: -x,
    "exp": lambda x, a: K.exp(x),
    "sqrt": lambda x, a: K.sqrt(x),
    "rsqrt": lambda x, a: K.rsqrt(x),
    "tanh": lambda x, a: K.tanh(x),
    "relu": lambda x, a: K.relu(x),
    "sigmoid": lambda x, a: K.sigmoid(x),
    "silu": lambda x, a: x * K.sigmoid(x),
    "hardsigmoid": lambda x, a: _clamp(x + 3.0, 0.0, 6.0) / 6.0,
    "hardswish": lambda x, a: x * _clamp(x + 3.0, 0.0, 6.0) / 6.0,
    "clamp": lambda x, a: _clamp(x, a.get("min"), a.get("max")),
    "gelu": _gelu,
    # added 2026-11-24 for rotary positions and Hugging Face models
    "cos": lambda x, a: kir.Unary("cos", x),
    "sin": lambda x, a: kir.Unary("sin", x),
    "log": lambda x, a: kir.Unary("log", x),
    "abs": lambda x, a: kir.Unary("abs", x),
    "reciprocal": lambda x, a: 1.0 / x,
    # added 2026-11-24 for training programs: the exact gelu gradient reads erf, and AdamW's bias
    # correction 1 - b^t is -expm1(t ln b), which keeps its digits where 1 - exp cancels
    "erf": lambda x, a: kir.Unary("erf", x),
    "expm1": lambda x, a: kir.Unary("expm1", x),
    # added 2026-11-24 with the op coverage work: the value itself (no derivative, docs/AUTODIFF.md)
    "frozen": lambda x, a: x,
}

BINARY: Dict[str, Tuple[Callable, str, str]] = {   # op: (expression, operand form, result kind)
    "add": (lambda a, b: a + b, "n", "f"),
    "sub": (lambda a, b: a - b, "n", "f"),
    "mul": (lambda a, b: a * b, "n", "f"),
    "div": (lambda a, b: a / b, "n", "f"),
    "eq": (lambda a, b: K.eq(a, b), "n", "b"),
    "ne": (lambda a, b: K.ne(a, b), "n", "b"),
    "lt": (lambda a, b: a < b, "n", "b"),
    "le": (lambda a, b: a <= b, "n", "b"),
    "gt": (lambda a, b: a > b, "n", "b"),
    "ge": (lambda a, b: a >= b, "n", "b"),
    "logical_and": (lambda a, b: a & b, "b", "b"),
    "logical_or": (lambda a, b: a | b, "b", "b"),
}


# ----------------------------------------------------------------------------- rules
# A rule takes (lowering, node) and returns the lowered value of the node's output. It may create
# intermediate lowered values (a reduction over two dims is two single axis reductions).

RULES: Dict[str, Callable] = {}


def rule(*names):
    def deco(fn):
        for n in names:
            RULES[n] = fn
        return fn
    return deco


def _unary_rule(L: "Lowering", node: Node):
    f, attrs = UNARY[node.op], node.attrs
    x = L.operand(node.inputs[0])
    shape = L.shape_of(node.output)
    return L.new(shape, "f", lambda idx, lo: f(_num(lo(x, idx), x.kind), attrs), (x,), node=node)


for _op in UNARY:
    RULES[_op] = _unary_rule


@rule("cast")
def _cast_rule(L, node):
    """Every float value is computed in the executor's one precision, so a cast to float32, float64 or
    the executor's own dtype is a copy; a bool or integer source becomes its number (integers exact
    up to 2^24 in float32). A cast that rounds to another 16 bit dtype, or to an integer or bool
    result, is not a kernel (it takes the fallback)."""
    x = L.operand(node.inputs[0])
    to = node.output.type.dtype
    if not to.is_floating_point:
        raise NotLowerable("a cast to an integer or bool dtype has no kernel type")
    if torch.finfo(to).bits < 32 and to != L.opts.torch_dtype:
        # the reference rounds through 16 bit dtypes; a copy in the working precision would not
        raise NotLowerable(f"a cast to {to} rounds, and the executor computes in {L.opts.dtype}")
    isl = island_dtype(L.opts)
    island = isl is not None and L.value_fdtype(node.output) == isl
    if island and x.leaf and x.kind == "f" and L.is_param(x.value):
        x.fdtype = isl           # a param read through an island cast is stored in the island dtype (inv_freq)
    if (L.opts.mixed and x.kind == "f" and not x.leaf and island and x.fdtype != isl
            and torch.finfo(node.inputs[0].type.dtype).bits < 32):
        # mixed=True: a value declared in 16 bits read through a cast up to its float32 island is
        # stored first, so it rounds (and overflows) in 16 bits as declared; fused into the cast's
        # kernel it would stay a float32 register value (a float16 gradient that never overflows)
        x.force_mat = True
    out = L.new(L.shape_of(node.output), "f", lambda idx, lo: _num(lo(x, idx), x.kind), (x,), node=node)
    if x.kind == "f" and isl is not None and x.fdtype == isl and out.fdtype != isl and not out.force_mat:
        out.force_mat = True     # back from an island: stored, so it rounds where the model casts
        if isl == "float64" and not x.leaf:
            # a float64 island's value is stored too: the kernel that computes it then has a float64
            # output and so float64 math (kernel math follows the output dtype, triton_backend
            # Emitter.cdt); inlined into the cast's float32 kernel it would be computed in float32
            x.force_mat = True
    return out


_POW = {2.0: lambda x: x * x, 3.0: lambda x: x * x * x, 1.0: lambda x: x, 0.5: lambda x: K.sqrt(x),
        -0.5: lambda x: K.rsqrt(x), -1.0: lambda x: 1.0 / x}


@rule("pow")
def _pow_rule(L, node):
    """x ** c for a constant exponent the kernel IR writes exactly (2, 3, 1, 0.5, -0.5, -1: RMSNorm's
    ``x.pow(2)``). Other exponents take the fallback."""
    a, b = node.inputs
    xb = L.operand(b)
    if xb.const is None or float(xb.const) not in _POW or node.inputs[1].type.rank != 0:
        raise NotLowerable("pow with an exponent other than a constant 2, 3, 1, 0.5, -0.5 or -1")
    f = _POW[float(xb.const)]
    x = L.operand(a)
    shape = L.shape_of(node.output)
    return L.new(shape, "f", lambda idx, lo: f(_num(lo(x, bidx(x.shape, shape, idx)), x.kind)), (x,), node=node)


@rule("logical_not")
def _logical_not(L, node):
    x = L.operand(node.inputs[0])
    return L.new(L.shape_of(node.output), "b", lambda idx, lo: ~_bool(lo(x, idx), x.kind), (x,), node=node)


def _binary_rule(L: "Lowering", node: Node):
    f, form, kind = BINARY[node.op]
    a, b = L.operand(node.inputs[0]), L.operand(node.inputs[1])
    shape = L.shape_of(node.output)
    conv = _num if form == "n" else _bool

    def fn(idx, lo):
        return f(conv(lo(a, bidx(a.shape, shape, idx)), a.kind), conv(lo(b, bidx(b.shape, shape, idx)), b.kind))

    return L.new(shape, kind, fn, (a, b), node=node)


for _op in BINARY:
    RULES[_op] = _binary_rule


@rule("where")
def _where(L, node):
    c, a, b = (L.operand(v) for v in node.inputs)
    shape = L.shape_of(node.output)

    kind = kind_of(node.output.type.dtype)
    conv = _bool if kind == "b" else _num

    def fn(idx, lo):
        return K.where(_bool(lo(c, bidx(c.shape, shape, idx)), c.kind),
                       conv(lo(a, bidx(a.shape, shape, idx)), a.kind),
                       conv(lo(b, bidx(b.shape, shape, idx)), b.kind))

    return L.new(shape, kind, fn, (a, b, c), node=node)


def _reduce_chain(L: "Lowering", node: Node, x: LV, op: str, dims: List[int], keep: bool, first=None) -> LV:
    """Reduce ``x`` over ``dims`` one axis at a time (the GPU tile shape takes one reduced size)."""
    cur = x
    for n, d in enumerate(sorted(dims, reverse=True)):
        src, size = cur, cur.shape[d]
        shape = list(src.shape)
        if keep:
            shape[d] = 1
        else:
            del shape[d]
        lid = len(L.lvs)
        conv = first if (first is not None and n == 0) else (lambda e, k: _num(e, k))

        def fn(idx, lo, src=src, d=d, size=size, lid=lid, conv=conv):
            r = K.axis(f"r{lid}", size)
            ix = list(idx)
            if keep:
                ix[d] = r
            else:
                ix.insert(d, r)
            return kir.Reduce(op, r, conv(lo(src, tuple(ix)), src.kind))

        cur = L.new(shape, "f", fn, (src,), heavy=True, node=node)
    return cur


def _reduce_dims(node: Node, x: LV) -> List[int]:
    dims = node.attrs.get("dims")
    rank = len(x.shape)
    if dims is None or len(dims) == 0:
        return list(range(rank))
    return sorted({d % max(rank, 1) for d in dims})


@rule("sum")
def _sum(L, node):
    x = L.operand(node.inputs[0])
    dims = _reduce_dims(node, x)
    if not dims:
        return L.new(x.shape, "f", lambda idx, lo: _num(lo(x, idx), x.kind) + 0.0, (x,), node=node)
    return _reduce_chain(L, node, x, "sum", dims, node.attrs.get("keepdim", False))


@rule("mean")
def _mean(L, node):
    x = L.operand(node.inputs[0])
    dims = _reduce_dims(node, x)
    sizes = [x.shape[d] for d in dims]
    s = _reduce_chain(L, node, x, "sum", dims, node.attrs.get("keepdim", False)) if dims else x
    static = 1
    for n in sizes:
        if isinstance(n, int):
            static *= n

    def fn(idx, lo):
        e = _num(lo(s, idx), s.kind)
        for n in sizes:
            if not isinstance(n, int):
                e = e / K.dim(n)
        return e / float(static) if static != 1 else e + 0.0

    return L.new(s.shape, "f", fn, (s,), node=node)


@rule("any")
def _any(L, node):
    x = L.operand(node.inputs[0])
    dims = _reduce_dims(node, x)
    if not dims:
        return L.new(x.shape, "b", lambda idx, lo: _bool(lo(x, idx), x.kind) | False, (x,), node=node)
    m = _reduce_chain(L, node, x, "max", dims, node.attrs.get("keepdim", False),
                      first=lambda e, k: K.where(_bool(e, k), 1.0, 0.0))
    return L.new(m.shape, "b", lambda idx, lo: lo(m, idx) > 0.5, (m,), node=node)


def _softmax_lv(L: "Lowering", node: Node, x: LV, a: int) -> LV:
    lid = len(L.lvs)
    size = x.shape[a]

    def fn(idx, lo):
        def at(ax):
            ix = list(idx)
            ix[a] = ax
            return lo(x, tuple(ix))

        r1, r2 = K.axis(f"r{lid}m", size), K.axis(f"r{lid}s", size)
        m = kir.Reduce("max", r1, at(r1))
        total = kir.Reduce("sum", r2, K.exp(at(r2) - m))
        return K.exp(lo(x, idx) - m) / total

    return L.new(x.shape, "f", fn, (x,), heavy=True, node=node)


def _log_softmax_lv(L: "Lowering", node: Node, x: LV, a: int) -> LV:
    """x - m - log(sum(exp(x - m))) with m the max along the axis: the stable form, no exp of a
    large value and no log of an underflowed probability."""
    lid = len(L.lvs)
    size = x.shape[a]

    def fn(idx, lo):
        def at(ax):
            ix = list(idx)
            ix[a] = ax
            return lo(x, tuple(ix))

        r1, r2 = K.axis(f"r{lid}m", size), K.axis(f"r{lid}s", size)
        m = kir.Reduce("max", r1, at(r1))
        total = kir.Reduce("sum", r2, K.exp(at(r2) - m))
        return lo(x, idx) - m - kir.Unary("log", total)

    return L.new(x.shape, "f", fn, (x,), heavy=True, node=node)


@rule("log_softmax")
def _log_softmax(L, node):
    x = L.operand(node.inputs[0])
    if x.kind != "f" or not x.shape:
        raise NotLowerable("log_softmax needs a float tensor with at least one dim")
    return _log_softmax_lv(L, node, x, node.attrs["dim"] % len(x.shape))


@rule("amax", "amin")
def _amax(L, node):
    """A max (or min, as -max(-x)) reduction, one axis at a time."""
    x = L.operand(node.inputs[0])
    dims = _reduce_dims(node, x)
    keep = node.attrs.get("keepdim", False)
    if x.kind != "f":
        raise NotLowerable(f"{node.op} of a non float tensor has no kernel type")
    if not dims:
        return L.new(x.shape, "f", lambda idx, lo: _num(lo(x, idx), x.kind) + 0.0, (x,), node=node)
    if node.op == "amax":
        return _reduce_chain(L, node, x, "max", dims, keep)
    m = _reduce_chain(L, node, x, "max", dims, keep, first=lambda e, k: -_num(e, k))
    return L.new(m.shape, "f", lambda idx, lo: -lo(m, idx), (m,), node=node)


@rule("softmax")
def _softmax(L, node):
    x = L.operand(node.inputs[0])
    if x.kind != "f" or not x.shape:
        raise NotLowerable("softmax needs a float tensor with at least one dim")
    return _softmax_lv(L, node, x, node.attrs["dim"] % len(x.shape))


@rule("log_softmax")
def _log_softmax(L, node):
    """x - m - log(sum(exp(x - m))) with m the max along the dim: the reference's stable form, one
    kernel with two reductions (a training program's cross entropy reads it)."""
    x = L.operand(node.inputs[0])
    if x.kind != "f" or not x.shape:
        raise NotLowerable("log_softmax needs a float tensor with at least one dim")
    a = node.attrs["dim"] % len(x.shape)
    lid = len(L.lvs)
    size = x.shape[a]

    def fn(idx, lo):
        def at(ax):
            ix = list(idx)
            ix[a] = ax
            return lo(x, tuple(ix))

        r1, r2 = K.axis(f"r{lid}m", size), K.axis(f"r{lid}s", size)
        m = kir.Reduce("max", r1, at(r1))
        total = kir.Reduce("sum", r2, K.exp(at(r2) - m))
        return (lo(x, idx) - m) - kir.Unary("log", total)

    return L.new(x.shape, "f", fn, (x,), heavy=True, node=node)


def _matmul_lv(L: "Lowering", node: Node, a: LV, b: LV, shape) -> LV:
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise NotLowerable("matmul with a one dim operand has no kernel rule yet")
    lid = len(L.lvs)
    nb = len(shape) - 2
    inner = a.shape[-1]

    def fn(idx, lo):
        r = K.axis(f"r{lid}", inner)
        ia = bidx(a.shape[:-2], shape[:nb], idx[:nb]) + (idx[-2], r)
        ib = bidx(b.shape[:-2], shape[:nb], idx[:nb]) + (r, idx[-1])
        return kir.Reduce("sum", r, _num(lo(a, ia), a.kind) * _num(lo(b, ib), b.kind))

    lv = L.new(shape, "f", fn, (a, b), heavy=True, node=node)
    lv.contract = True
    return lv


@rule("matmul")
def _matmul(L, node):
    a, b = L.operand(node.inputs[0]), L.operand(node.inputs[1])
    return _matmul_lv(L, node, a, b, L.shape_of(node.output))


def kn_weights(prog: Program, opts: "Options") -> set:
    """Names of the constant weights stored [K, N] in memory (prepacked at compile time).

    A strict float32 contraction (CUDA, TF32 off) reads a linear weight W [N, K] stored K
    contiguous as the B tile of x @ W^T; Triton then stages that tile K major in shared memory
    and the FMA loop reads it with 8 way bank conflicts. The same values stored [K, N] (the logical
    shape and every index unchanged, only the strides) give the same fused multiply add chain per
    output, so the same bits, and 1.06x to 1.98x faster kernels (measured on 96 cases,
    truestar-research lossmm/REPORT.md section 3). Only a 2 dim float32 param whose every use in
    the program is the weight operand of a linear, and which is not itself an output, is
    relaid; a weight that is an input (computed or passed at run time) is read as it is. The copy
    replaces the param's device buffer, so no memory is added, and nothing is keyed by address."""
    if not (opts.device == "cuda" and not opts.platform and opts.dtype == "float32"
            and not opts.allow_tf32):
        return set()
    uses: Dict[str, List[bool]] = {}
    pids = {id(p): p.name for p in prog.params}
    for node in prog.nodes:
        for i, x in enumerate(node.inputs):
            if x is not None and id(x) in pids:
                uses.setdefault(pids[id(x)], []).append(node.op == "linear" and i == 1)
        for v in node.attrs.values():   # a region that reads an outer value keeps the plain layout
            if isinstance(v, Program):
                for x in list(v.inputs) + list(v.params):
                    if id(x) in pids:
                        uses.setdefault(pids[id(x)], []).append(False)
    outs = {id(v) for v in prog.outputs}
    out = set()
    for p in prog.params:
        data = prog.param_data.get(p.name)
        if (data is None or data.dim() != 2 or data.dtype != torch.float32 or id(p) in outs
                or not uses.get(p.name) or not all(uses[p.name]) or min(data.shape) < 2):
            continue
        out.add(p.name)
    return out


def store_kn(data: torch.Tensor) -> torch.Tensor:
    """``data`` [N, K] with the same values and shape, stored [K, N] (strides (1, N))."""
    return data.t().contiguous().t()


@rule("linear")
def _linear(L, node):
    x, w = L.operand(node.inputs[0]), L.operand(node.inputs[1])
    b = None if node.inputs[2] is None else L.operand(node.inputs[2])
    if len(x.shape) < 2:
        raise NotLowerable("linear on a one dim input has no kernel rule yet")
    lid = len(L.lvs)
    inner = w.shape[1]

    def fn(idx, lo):
        r = K.axis(f"r{lid}", inner)
        y = kir.Reduce("sum", r, lo(x, tuple(idx[:-1]) + (r,)) * lo(w, (idx[-1], r)))
        return y if b is None else y + lo(b, (idx[-1],))

    lv = L.new(L.shape_of(node.output), "f", fn, (x, w) + (() if b is None else (b,)), heavy=True, node=node)
    lv.contract = True
    return lv


@rule("batch_norm")
def _batch_norm(L, node):
    """Inference batch norm. With constant statistics it folds to x * scale[c] + shift[c]."""
    x = L.operand(node.inputs[0])
    w, b, mean, var = node.inputs[1:5]
    eps = node.attrs["eps"]
    if len(x.shape) < 2:
        raise NotLowerable("batch_norm needs a channel dim")
    data = L.prog.param_data
    if all(v is None or v.name in data for v in (w, b)) and mean.name in data and var.name in data \
            and all(v is None or L.is_param(v) for v in (w, b, mean, var)):
        m64, v64 = data[mean.name].double(), data[var.name].double()
        scale = 1.0 / torch.sqrt(v64 + eps)
        if w is not None:
            scale = scale * data[w.name].double()
        shift = -m64 * scale
        if b is not None:
            shift = shift + data[b.name].double()
        s_lv = L.derived(f"{node.output.name}.scale", scale)
        t_lv = L.derived(f"{node.output.name}.shift", shift)

        def fn(idx, lo):
            return lo(x, idx) * lo(s_lv, (idx[1],)) + lo(t_lv, (idx[1],))

        return L.new(x.shape, "f", fn, (x, s_lv, t_lv), node=node)
    ops = [None if v is None else L.operand(v) for v in (w, b, mean, var)]

    def fn(idx, lo):
        c = (idx[1],)
        y = (lo(x, idx) - lo(ops[2], c)) / K.sqrt(lo(ops[3], c) + eps)
        if ops[0] is not None:
            y = y * lo(ops[0], c)
        if ops[1] is not None:
            y = y + lo(ops[1], c)
        return y

    return L.new(x.shape, "f", fn, (x,) + tuple(o for o in ops if o is not None), node=node)


@rule("layer_norm")
def _layer_norm(L, node):
    x = L.operand(node.inputs[0])
    w = None if node.inputs[1] is None else L.operand(node.inputs[1])
    b = None if node.inputs[2] is None else L.operand(node.inputs[2])
    if len(node.attrs["normalized_shape"]) != 1:
        raise NotLowerable("layer_norm over more than the last dim has no kernel rule yet")
    eps = node.attrs["eps"]
    lid = len(L.lvs)
    n = x.shape[-1]

    def fn(idx, lo):
        def at(ax):
            return lo(x, tuple(idx[:-1]) + (ax,))

        r1, r2 = K.axis(f"r{lid}a", n), K.axis(f"r{lid}b", n)
        mean = kir.Reduce("sum", r1, at(r1)) / K.dim(n)
        d = at(r2) - mean
        var = kir.Reduce("sum", r2, d * d) / K.dim(n)
        y = (lo(x, idx) - mean) * K.rsqrt(var + eps)
        if w is not None:
            y = y * lo(w, (idx[-1],))
        if b is not None:
            y = y + lo(b, (idx[-1],))
        return y

    args = (x,) + tuple(o for o in (w, b) if o is not None)
    return L.new(x.shape, "f", fn, args, heavy=True, node=node)


ATTENTION_MAX_HEAD = 256  # limit of the GPU attention loop shape (docs/KERNEL_IR.md, section 8)


@rule("attention")
def _attention(L, node):
    """softmax(q k^T * scale + mask) v as scores, weights and a product; fusion joins them on the GPU."""
    q, k, v = (L.operand(x) for x in node.inputs[:3])
    mask = None if node.inputs[3] is None else L.operand(node.inputs[3])
    if min(len(q.shape), len(k.shape), len(v.shape)) < 2:
        raise NotLowerable("attention needs at least two dims")
    out_shape = L.shape_of(node.output)
    batch = out_shape[:-2]
    s_shape = batch + (q.shape[-2], k.shape[-2])
    e = q.shape[-1]
    scale = node.attrs.get("scale")
    causal = bool(node.attrs.get("is_causal", False))
    lid = len(L.lvs)
    nb = len(batch)

    def scores(idx, lo):
        c = K.axis(f"r{lid}", e)
        iq = bidx(q.shape[:-2], batch, idx[:nb]) + (idx[-2], c)
        ik = bidx(k.shape[:-2], batch, idx[:nb]) + (idx[-1], c)
        s = kir.Reduce("sum", c, lo(q, iq) * lo(k, ik))
        if scale is not None:
            s = s * float(scale)
        elif isinstance(e, int):
            s = s * (1.0 / math.sqrt(e))
        else:
            s = s * K.rsqrt(K.dim(e))
        if causal:
            s = K.where(_index_expr(idx[-1]) <= _index_expr(idx[-2]), s, K.NEG_INF)
        if mask is not None:
            m = lo(mask, bidx(mask.shape, s_shape, idx))
            s = K.where(m, s, K.NEG_INF) if mask.kind == "b" else s + m
        return s

    s_lv = L.new(s_shape, "f", scores, (q, k) + (() if mask is None else (mask,)), heavy=True, node=node)
    s_lv.contract = True
    w_lv = _softmax_lv(L, node, s_lv, len(s_shape) - 1)
    heads = [d for d in (e, v.shape[-1])]
    if any(not isinstance(d, int) or d > ATTENTION_MAX_HEAD for d in heads):
        s_lv.force_mat = w_lv.force_mat = True   # too wide for the fused attention shape: three kernels
    return _matmul_lv(L, node, w_lv, v, out_shape)


# ----------------------------------------------------------------------------- op classes

LAYOUT_OPS = ("reshape", "permute", "expand", "select", "slice", "select_dynamic", "cat")
CREATE_OPS = ("full", "full_like", "arange")
# Ops with a torch library kernel (cuDNN on the GPU). Since 2026-11-24 they all have kernel rules;
# the library runs them only when asked (``library=``) or when the rule cannot take this use of the
# op (a symbolic spatial size, for example), and the report says so.
LIBRARY_OPS = ("conv2d", "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d")
# Data movement and sorting through torch on the executor's device (runtime.IndexStep): exact copies
# of entries, checked against the reference like library kernels (2026-11-24).
INDEX_OPS = ("sort", "topk", "gather", "scatter_add", "index_select", "index_copy", "index",
             "kvq_attention",   # quantized KV decode attention (truestar.kvquant), opted into by the model
             "mla_attention",   # fused MLA latent attention (truestar.mla.latent), decode="fused"
             "linear_attention", "linear_attention_vjp", "linear_attention_jvp",   # truestar.linattn (P60)
             "attention_grad")   # the fused attention backward (compiler/attention_bwd.py)
# Integer tensor arithmetic (positions, slot and block indices) on the device through the reference
# implementation itself, which is plain torch integer arithmetic and therefore exact anywhere.
INT_DEVICE_OPS = ("add", "sub", "mul", "neg", "abs", "remainder", "fmod", "maximum", "minimum", "where",
                  "sum", "cumsum", "amax", "amin",   # these four added with the op coverage work
                  "clamp")                           # DeBERTa v2 and v3 relative positions (text sweep)
# Ops with an integer or bool result read from float operands (a cast to an integer dtype, argmax):
# run by the reference implementation on the device's own tensors, checked exactly (2026-11-24).
INT_RESULT_OPS = ("cast", "argmax", "argmin")
CONTROL_OPS = ("cond", "loop")
HOST_OPS = ("dim_scalar",)


@dataclass
class Options:
    device: str = "cuda"
    dtype: str = "float32"
    allow_tf32: bool = False   # resolved: None ("follow torch") is resolved by CompiledProgram
    deterministic: bool = False  # no split K; None ("follow torch") is resolved by CompiledProgram
    hidden: Dict[str, Any] = field(default_factory=dict)   # test hooks passed to the kernel compiler
    memory_reuse: bool = True   # free values after their last reader and reuse dead buffers (docs/MEMORY.md)
    checks: frozenset = frozenset()   # compiled in check classes (kernel/checks.py, compiler/checks.py)
    sentinel_read: str = "call"          # "call": read the flags before returning; "deferred": at a later call
    library: Tuple[str, ...] = ()   # ops the user asked to run through torch library kernels
    cublas: bool = False   # float32 contractions may run as a timed, verified cuBLAS call (kernel/library.py):
    #                        only when the program follows PyTorch's TF32 setting and is not deterministic
    platform: str = ""   # "rocm" when the "cuda" device is an AMD GPU (a ROCm build of torch, docs/ROCM.md)
    accept_ill_conditioned: bool = False   # end to end: accept float32 outputs within 4x of an independent float32 evaluation (TS782)
    portable_tensor_cores: bool = False   # half and TF32 dot products as exact tensor core blocks (kernel/tcexact.py)
    mixed: bool = False   # values keep their declared dtype: 16 bit ones in the working precision, wider ones in float32 islands (nominal_float)

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)

    @property
    def torch_device(self) -> torch.device:
        # Metal kernels run on the MPS device's tensors
        return torch.device("mps" if self.device == "metal" else self.device)


@dataclass
class KernelSpec:
    """One fused kernel: what the runtime needs to launch it and to check it."""

    root: LV
    kernel: Any                 # kernel.ir.Kernel
    inputs: List[LV]            # in the order of kernel.inputs
    nodes: List[Node]           # IR nodes whose work is in this kernel
    plan_kind: str = ""


class Lowering:
    """Lowers one program (or one region) and partitions it into steps. See ``build``."""

    def __init__(self, prog: Program, opts: Options, table: DimTable, path: str):
        self.prog = prog
        self.opts = opts
        self.table = table
        self.path = path
        self.lvs: List[LV] = []
        self.vals: Dict[int, LV] = {}
        self.node_index: Dict[int, int] = {id(n): i for i, n in enumerate(prog.nodes)}
        self.nslots = 0
        self._pos = 0.0
        self.events: List[Tuple[float, str, Any]] = []      # (position, kind, payload) for non kernel steps
        self.derived_params: List[Tuple[LV, torch.Tensor]] = []
        self.adapters: List[LV] = []
        self._param_ids = {id(p) for p in prog.params}
        self.nominal = nominal_float(prog, opts)
        self._conv = convention_values(prog, opts)   # float64 by calling convention: no island
        self._cur_fdtype = opts.dtype   # float dtype of the values the node being lowered makes
        self.kernels: List[KernelSpec] = []
        self.marks: List[Tuple[int, LV]] = []    # (rule start, operand) marked for strided reads

    # ---- bookkeeping
    def tick(self) -> float:
        self._pos += 1.0
        return self._pos

    def slot(self) -> int:
        self.nslots += 1
        return self.nslots - 1

    def is_param(self, v: Value) -> bool:
        return id(v) in self._param_ids

    def shape_of(self, v: Value):
        return tuple(size_of(d) for d in v.type.shape)

    def fdtype_of(self, dtype: torch.dtype) -> str:
        return island_dtype(self.opts) if is_island(self.opts, self.nominal, dtype) else self.opts.dtype

    def value_fdtype(self, v: Value) -> str:
        """fdtype_of the value's declared dtype, except a float64 value by calling convention."""
        return self.opts.dtype if id(v) in self._conv else self.fdtype_of(v.type.dtype)

    def new(self, shape, kind, fn, args, heavy=False, node=None) -> LV:
        lv = LV(len(self.lvs), shape, kind, fn, args, heavy, node, pos=self.tick())
        lv.fdtype = self._cur_fdtype
        self.lvs.append(lv)
        return lv

    def leaf(self, v: Value, const=None) -> LV:
        lv = LV(len(self.lvs), self.shape_of_leaf(v), kind_of(v.type.dtype), leaf=True, value=v, const=const,
                pos=self.tick(), name=v.name)
        lv.fdtype = self.value_fdtype(v)
        lv.slot = self.slot()
        if lv.kind == "f":
            lv.kslot = lv.slot
        self.lvs.append(lv)
        self.vals[id(v)] = lv
        return lv

    def shape_of_leaf(self, v: Value):
        try:
            return self.shape_of(v)
        except NotLowerable:
            return None   # a zero sized dim: usable by non kernel steps only

    def strided_operand(self, v: Value) -> LV:
        """Operand of a convolution or pooling rule. It is read at strided indices with a pad value,
        which only a buffer can serve, so it is materialized (its own producer then takes any
        elementwise chain as an epilogue)."""
        lv = self.operand(v)
        if lv.kind != "f":
            raise NotLowerable("convolution and pooling need a float input")
        if not lv.leaf and not lv.force_mat:
            lv.force_mat = True
            self.marks.append((len(self.lvs), lv))
        return lv

    def derived(self, name: str, data: torch.Tensor) -> LV:
        """A constant computed at compile time from params (float64 in, working dtype at run time)."""
        lv = LV(len(self.lvs), tuple(int(s) for s in data.shape), "f", leaf=True, pos=self.tick(), name=name)
        lv.fdtype = self.opts.dtype
        lv.slot = lv.kslot = self.slot()
        self.lvs.append(lv)
        self.derived_params.append((lv, data))
        return lv

    def operand(self, v: Value) -> LV:
        """Lowered value of an IR value, for use inside a kernel expression."""
        lv = self.vals[id(v)]
        if lv.shape is None:
            raise NotLowerable(f"{v.ref} has a zero sized dim")
        return lv

    def need(self, v: Value, true_typed: bool = True) -> int:
        """Slot of ``v`` as a real tensor, for a reader that is not a kernel. Forces materialization."""
        lv = self.vals[id(v)]
        if lv.leaf:
            return lv.slot
        lv.must_mat = True
        if lv.kind == "f" or true_typed:
            lv.need_true = lv.need_true or lv.kind != "f"
            if lv.slot is None:
                lv.slot = self.slot()
            return lv.slot
        if lv.kslot is None:
            lv.kslot = self.slot()
        return lv.kslot

    def kernel_slot(self, lv: LV) -> int:
        """Slot of the form of ``lv`` that kernels load (a float tensor)."""
        if lv.kslot is None:
            if lv.leaf:                      # bool or int leaf: one cast step, shared by every kernel
                lv.kslot = self.slot()
                self.adapters.append(lv)
            elif lv.kind == "f":
                if lv.slot is None:
                    lv.slot = self.slot()
                lv.kslot = lv.slot
            else:
                lv.kslot = self.slot()
        return lv.kslot

    # ---- kernel construction
    def buffer(self, lv: LV):
        dtype = "float32" if lv.kind == "i" else (lv.fdtype or self.opts.dtype) if lv.kind == "f" else self.opts.dtype
        return K.Buffer(f"b{lv.id}", lv.shape or (1,), dtype)   # a scalar is loaded as one element

    def expr(self, lv: LV, idx, mat, root: LV, pad=None):
        """Expression of ``lv`` at ``idx``. ``pad`` is for strided reads (convolution and pooling):
        the value read where a strided index falls outside the tensor. Only a buffer can be read
        that way, so the reader's rule marks its operand ``force_mat``."""
        if lv.const is not None:
            return kir.Const(bool(lv.const) if lv.kind == "b" else float(lv.const))
        if lv.leaf or (lv.id in mat and lv is not root):
            e = kir.Load(self.buffer(lv), tuple(idx) if lv.shape else (0,), pad)
            return K.ne(e, 0.0) if lv.kind == "b" else e
        if pad is not None or any(isinstance(a, kir.Strided) for a in idx):
            raise kir.IRError(f"internal: a strided read of {self.kernel_name(lv)}, which is not a buffer")
        return kir.as_expr(lv.fn(tuple(idx), lambda a, ix, pad=None: self.expr(a, ix, mat, root, pad)))

    def kernel_for(self, root: LV, mat) -> Any:
        axes = tuple(K.axis(f"o{root.id}_{k}", s) for k, s in enumerate(root.shape))
        e = self.expr(root, axes, mat, root)
        if root.kind == "b":
            e = K.where(e, 1.0, 0.0)
        if not axes:   # a scalar is computed as one element; the runtime drops the unit dim
            axes = (K.axis(f"o{root.id}_unit", 1),)
        dt = (root.fdtype or self.opts.dtype) if root.kind == "f" else self.opts.dtype
        return K.kernel(self.kernel_name(root), [K.output(f"o{root.id}", axes, e, dt)])

    def kernel_name(self, root: LV) -> str:
        out = root.node.output.name if root.node is not None else root.name
        return f"{self.path}:%{out}" + ("" if root.value is not None else f".step{root.id}")

    def inlined(self, root: LV, mat) -> List[LV]:
        """Lowered values inlined into root's kernel, nearest first."""
        out, seen, frontier = [], {root.id}, [root]
        while frontier:
            nxt = []
            for t in frontier:
                for a in t.args:
                    if a.id in seen or a.leaf or a.id in mat:
                        continue
                    seen.add(a.id)
                    out.append(a)
                    nxt.append(a)
            frontier = nxt
        return out

    def _nested_heavy(self, root: LV, mat) -> Optional[LV]:
        """CPU rule: the nearest inlined value that the loop nest would recompute, or None. That is a
        reduction under another reduction, or anything under a matrix product (the product's loop
        would redo it for every output column)."""
        frontier, seen = [(root, root.heavy, root.contract)], {root.id}
        while frontier:
            nxt = []
            for t, under, product in frontier:
                for a in t.args:
                    if a.id in seen or a.leaf or a.id in mat:
                        continue
                    seen.add(a.id)
                    if product or (a.heavy and under):
                        return a
                    nxt.append((a, under or a.heavy, product or a.contract))
            frontier = nxt
        return None

    def reject(self, root: LV, mat, pick: bool = True) -> Optional[Tuple[str, Optional[LV]]]:
        """None when root's fused kernel is accepted, else (reason, value to materialize or None)."""
        try:
            k = self.kernel_for(root, mat)
        except kir.IRError as e:
            prods = self.inlined(root, mat)
            return str(e), (prods[0] if prods else None)
        if self.opts.device == "cpu":
            bad = self._nested_heavy(root, mat)
            return None if bad is None else ("the CPU loop nest would recompute an inlined value", bad)
        try:
            gpu_plan(kir.canonicalize(k)[0])
            return None
        except Unschedulable as e:
            return str(e), (self._split_point(root, mat) if pick else None)

    def _unmaterialize(self, mat, required):
        """Undo split points the greedy pass above no longer needs. A split point is chosen while
        other values are still inlined; a value materialized later can make an earlier one
        unnecessary (a batch norm's gamma gradient sum read g * xhat through the forward
        convolution until xhat itself was stored, so g * xhat stayed a kernel of its own: a write
        and a read of the activation's size per batch norm). A value leaves the set when exactly
        one kernel would then inline it and that kernel still fits a loop shape; only elementwise
        values over stored inputs. float32 programs
        only: there an inlined value is the same float32 number as the stored one, so the result
        does not change by a bit."""
        if self.opts.dtype != "float32":
            return mat
        mat = set(mat)
        for lid in sorted(mat - required, reverse=True):
            lv = self.lvs[lid]
            if lv.fdtype not in (None, "float32") or lv.kind != "f" or lv.heavy:
                continue        # a reduction stays stored: inlined into a smaller reduction it runs in fewer programs
            trial = mat - {lid}
            if any(m.heavy for m in self.inlined(lv, trial)):
                continue        # it would pull a reduction or a product into its reader (measured: a
                                # matmul pulled into a relu backward epilogue ran 40x slower)
            readers = [r for r in trial if any(m.id == lid for m in self.inlined(self.lvs[r], trial))]
            if len(readers) == 1 and self.accepts(self.lvs[readers[0]], trial):
                mat = trial
        return mat

    def accepts(self, root: LV, mat) -> bool:
        return self.reject(root, mat, pick=False) is None

    def _split_point(self, root: LV, mat) -> Optional[LV]:
        """What to materialize when root's fused kernel fits no loop shape: the nearest reduction
        producer, or better the top of the elementwise chain that reads it, so that the chain becomes
        the producer's epilogue (computed once) and not the reader's prologue."""
        parent: Dict[int, LV] = {}
        order, seen, frontier = [], {root.id}, [root]
        while frontier:
            nxt = []
            for t in frontier:
                for a in t.args:
                    if a.id in seen or a.leaf or a.id in mat:
                        continue
                    seen.add(a.id)
                    parent[a.id] = t
                    order.append(a)
                    nxt.append(a)
            frontier = nxt
        heavy = [p for p in order if p.heavy]
        if not heavy:
            return order[0] if order else None
        for prod in heavy:
            path = [prod]
            while parent[path[-1].id] is not root:
                path.append(parent[path[-1].id])
            for cand in reversed(path):          # highest first: the longest epilogue that still fits
                if self.accepts(cand, mat | {cand.id}):
                    return cand
        return heavy[0]

    def standalone_ok(self, first: int) -> Optional[str]:
        """Check that every lowered value made since ``first`` is a valid kernel on its own."""
        everything = {lv.id for lv in self.lvs}
        for lv in self.lvs[first:]:
            if lv.leaf:
                continue
            try:
                bad = self.reject(lv, everything, pick=False)
            except (NotLowerable, kir.IRError) as e:
                return str(e)
            if bad is not None:
                return bad[0]
        return None

    # ---- the pass
    def lower_nodes(self, handlers) -> None:
        """Walk the nodes once. ``handlers`` builds the non kernel steps (see compiler.runtime)."""
        prog = self.prog
        for v in prog.inputs:
            self.leaf(v)
        for p in prog.params:
            data = prog.param_data[p.name]
            const = None
            if data.dim() == 0 and data.dtype != torch.bool:
                const = data.item()
                if data.dtype.is_floating_point:
                    from .runtime import saturate_scalar
                    const = saturate_scalar(const, self.opts.torch_dtype)
            elif data.dim() == 0:
                const = bool(data.item())
            self.leaf(p, const)
        for node in prog.nodes:
            self._lower_node(node, handlers)
        for v in prog.outputs:
            self.need(v)

    def islands(self) -> List[LV]:
        """Values kept in the island dtype (float32 inside a 16 bit program, float64 inside a float32
        program): the mixed precision islands."""
        isl = island_dtype(self.opts)
        return [lv for lv in self.lvs if lv.kind == "f" and isl is not None and lv.fdtype == isl
                and (lv.value is not None or lv.leaf)]

    def _lower_node(self, node: Node, handlers) -> None:
        op = node.op
        kinds = [kind_of(o.type.dtype) for o in node.outputs]
        out_f = [o for o in node.outputs if o.type.dtype.is_floating_point]
        self._cur_fdtype = self.value_fdtype(out_f[0]) if out_f else self.opts.dtype
        if op in CONTROL_OPS:
            try:
                handlers.control(self, node)
            except NotLowerable as e:
                handlers.fallback(self, node, f"control flow region not compiled: {e}")
            return
        if op in LAYOUT_OPS:
            in_kinds = {kind_of(x.type.dtype) for x in node.inputs if x is not None and x.type.rank > 0}
            if op != "cat" or len(in_kinds) <= 1:
                handlers.layout(self, node)
                return
            handlers.fallback(self, node, "cat of mixed dtypes")
            return
        if op in CREATE_OPS:
            handlers.create(self, node)
            return
        if op in HOST_OPS:
            handlers.host(self, node)
            return
        if op in INT_RESULT_OPS and hasattr(handlers, "int_device") and all(k in "ib" for k in kinds)                 and node.inputs[0] is not None and node.inputs[0].type.rank > 0:
            handlers.int_device(self, node)
            return
        if op in ("random", "random_batched") and hasattr(handlers, "random"):
            handlers.random(self, node)
            return
        if op in LIBRARY_OPS and op in handlers.library_ops and (op in self.opts.library or op not in RULES):
            handlers.library(self, node)
            return
        if op in INDEX_OPS and hasattr(handlers, "index") and node.inputs[0] is not None                 and node.inputs[0].type.rank > 0:
            handlers.index(self, node)
            return
        if op == "pad" and hasattr(handlers, "int_device") and kinds[0] in "ib" and node.inputs[0] is not None                 and kind_of(node.inputs[0].type.dtype) != "f":
            # a bool or integer pad (Gemma 3's image block ids and their mask): the pad rule reads a float
            # buffer, so it runs on the device by the reference implementation, checked exactly
            handlers.int_device(self, node)
            return
        if op in RULES and kinds[0] in "fb" and len(node.outputs) == 1:
            first, pos = len(self.lvs), self._pos
            try:
                lv = RULES[op](self, node)
                why = self.standalone_ok(first)
            except NotLowerable as e:
                why = str(e)
            if why is None:
                lv.value = node.output
                lv.name = node.output.name
                self.vals[id(node.output)] = lv
                return
            del self.lvs[first:]
            self.derived_params = [d for d in self.derived_params if d[0].id < first]
            self._pos = pos
            while self.marks and self.marks[-1][0] >= first:   # undo what the failed rule marked
                _, lv_ = self.marks.pop()
                lv_.force_mat = False
            if op in LIBRARY_OPS and op in handlers.library_ops:
                handlers.library(self, node, why)
                return
            handlers.fallback(self, node, why)
            return
        if all(k == "i" for k in kinds) and all(o.type.rank == 0 for o in node.outputs) \
                and all(x is None or x.type.rank == 0 for x in node.inputs) and op in iref.IMPLS:
            handlers.host(self, node)
            return
        if op in INT_DEVICE_OPS and hasattr(handlers, "int_device") and all(k == "i" for k in kinds)                 and any(x is not None and x.type.rank > 0 for x in node.inputs)                 and all(x is None or kind_of(x.type.dtype) != "f" for x in node.inputs):
            handlers.int_device(self, node)
            return
        if op not in iref.IMPLS:
            raise UnsupportedOpError(f"{op} has no kernel lowering and no reference implementation "
                                     f"({node.source or 'no source'}); the program cannot be compiled")
        reason = (f"{op} has no kernel lowering yet" if op not in RULES
                  else f"{op} with an integer result has no kernel type")
        handlers.fallback(self, node, reason)

    def fuse(self) -> List[KernelSpec]:
        """Choose the materialized values, then build one kernel per materialized value."""
        for lv in self.lvs:
            for a in lv.args:
                a.uses += 1
        # A value with several readers is materialized when it holds a reduction. An elementwise value
        # with several readers is inlined as long as all of them end up in one kernel (equal
        # subexpressions are shared there); if two kernels would each recompute it, it is materialized.
        mat = {lv.id for lv in self.lvs if not lv.leaf and (lv.must_mat or lv.force_mat
                                                            or (lv.uses > 1 and lv.heavy))}
        required = set(mat)
        changed = True
        while changed:
            changed = False
            for lid in sorted(mat):
                bad = self.reject(self.lvs[lid], mat)
                if bad is None:
                    continue
                reason, pick = bad
                if pick is None:
                    raise IRError(f"internal: {self.kernel_name(self.lvs[lid])} was accepted alone but is "
                                  f"rejected now: {reason}")
                mat.add(pick.id)
                changed = True
                break
            if changed:
                continue
            readers: Dict[int, int] = {}
            for lid in mat:
                for m in self.inlined(self.lvs[lid], mat):
                    readers[m.id] = readers.get(m.id, 0) + 1
            shared = [i for i, n in readers.items() if n > 1]
            if shared:
                mat.add(max(shared))
                changed = True
        mat = self._unmaterialize(mat, required)
        carried: Dict[int, List[Node]] = {}   # virtual root id -> IR nodes whose work it holds
        specs = []
        for lid in sorted(mat):
            root = self.lvs[lid]
            k = self.kernel_for(root, mat)
            members = [root] + self.inlined(root, mat)
            nodes = {id(m.node): m.node for m in members if m.node is not None}
            by_name = {}
            for m in members:
                for a in m.args:
                    if a.const is None and (a.leaf or a.id in mat):
                        by_name[f"b{a.id}"] = a
            inputs = [by_name[b.name] for b in k.inputs]
            for a in inputs:
                for n in carried.get(a.id, ()):
                    nodes[id(n)] = n
            ordered = sorted(nodes.values(), key=lambda n: self.node_index[id(n)])
            if root.value is None:
                carried[root.id] = ordered
            if any(n.op == "cast" and n.attrs.get("directed") for n in ordered):
                # a rounding the program means to lean (stochastic rounding finds the next bfloat16
                # number by rounding up on purpose): exempt from the lean test only (kernel/verify.py)
                k.directed_rounding = True
            specs.append(KernelSpec(root, k, inputs, ordered))
        self.kernels = specs
        return specs


from . import conv_rules  # noqa: E402,F401  (registers the convolution and pooling rules)
from . import coverage_rules  # noqa: E402,F401  (masks, padding, scatters, pow: op coverage work)
