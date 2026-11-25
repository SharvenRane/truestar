"""Loop level kernel IR.

A kernel is a set of outputs. Each output is an expression evaluated at every point of its named
axes. Expressions are elementwise math over loads of input buffers, index values and reductions
(sum, max, min) over a further named axis. Reductions may nest. Sizes are ints or symbols bound
from the input shapes at call time.

Nodes are immutable and compare structurally, so equal subexpressions are shared (common
subexpression elimination) and a kernel has a stable text form that is hashed for the compile cache.

See docs/KERNEL_IR.md for the interface.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

DTYPES = ("float16", "bfloat16", "float32", "float64")

# Roughly how precise each dtype is; used to pick the compute dtype of a kernel.
_PRECISION_RANK = {"bfloat16": 0, "float16": 1, "float32": 2, "float64": 3}


def lowest_precision(dtypes: Iterable[str]) -> str:
    ds = list(dtypes)
    return min(ds, key=lambda d: _PRECISION_RANK[d]) if ds else "float32"


class IRError(ValueError):
    """A kernel was built wrong (bad shapes, unknown op, repeated axis)."""


# --------------------------------------------------------------------------- sizes and axes


class Sym:
    """A symbolic size, bound from input shapes when the kernel is called."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        if not name.isidentifier():
            raise IRError(f"symbol name must be an identifier, got {name!r}")
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Sym) and other.name == self.name

    def __hash__(self):
        return hash(("Sym", self.name))

    def __repr__(self):
        return self.name


Size = Union[int, Sym]


def _check_size(s) -> Size:
    if isinstance(s, Sym):
        return s
    if isinstance(s, int) and not isinstance(s, bool) and s >= 1:
        return s
    raise IRError(f"a size must be a positive int or a Sym, got {s!r}")


class Axis:
    """A named iteration axis with a size."""

    __slots__ = ("name", "size", "_hash")

    def __init__(self, name: str, size: Size):
        self.name = name
        self.size = _check_size(size)
        self._hash = hash(("Axis", name, self.size))

    def __eq__(self, other):
        return isinstance(other, Axis) and other.name == self.name and other.size == self.size

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f"{self.name}:{self.size}"


class Term:
    """One integer term of a strided index: ``coeff * (((axis * mul) // div) % mod)``.

    ``mod`` 0 means no modulo. Axes are never negative, so floor division and modulo of a term
    behave the same in every backend."""

    __slots__ = ("axis", "coeff", "mul", "div", "mod", "_hash")

    def __init__(self, axis: Axis, coeff: int = 1, mul: int = 1, div: int = 1, mod: int = 0):
        if not isinstance(axis, Axis):
            raise IRError(f"a strided index term needs an Axis, got {axis!r}")
        for nm, v, lo in (("coeff", coeff, None), ("mul", mul, 1), ("div", div, 1), ("mod", mod, 0)):
            if not isinstance(v, int) or isinstance(v, bool) or (lo is not None and v < lo):
                raise IRError(f"strided index term {nm} must be an int{'' if lo is None else f' >= {lo}'}, "
                              f"got {v!r}")
        self.axis, self.coeff, self.mul, self.div, self.mod = axis, coeff, mul, div, mod
        self._hash = hash(("Term", axis, coeff, mul, div, mod))

    def key(self):
        return (self.axis, self.mul, self.div, self.mod)

    def __eq__(self, other):
        return (isinstance(other, Term) and other.axis == self.axis and other.coeff == self.coeff
                and other.mul == self.mul and other.div == self.div and other.mod == self.mod)

    def __hash__(self):
        return self._hash

    def value_at(self, i: int) -> int:
        v = (i * self.mul) // self.div
        if self.mod:
            v %= self.mod
        return self.coeff * v

    def bounds(self) -> Optional[Tuple[int, int]]:
        """(lowest, highest) value over the axis, or None when the axis size is symbolic."""
        if not isinstance(self.axis.size, int):
            return None
        hi_in = ((self.axis.size - 1) * self.mul) // self.div
        if self.mod:
            hi_in = min(hi_in, self.mod - 1)
        lo, hi = 0, self.coeff * hi_in
        return (min(lo, hi), max(lo, hi))

    def text(self, name: str) -> str:
        s = name if self.mul == 1 else f"{name}*{self.mul}"
        if self.div != 1:
            s = f"({s})//{self.div}"
        if self.mod:
            s = f"({s})%{self.mod}"
        return s if self.coeff == 1 else f"{self.coeff}*{s}"


class Strided:
    """A strided load index: ``(sum of terms + offset) / exact``.

    Convolution and pooling read ``x[i * stride + tap * dilation - pad]``; a transposed
    convolution reads ``g[(i + pad - tap * dilation) / stride]`` only where the division is exact.
    The index is valid when it is exactly divisible by ``exact`` and inside the buffer dim; a load
    with a strided index that may be invalid must give a ``pad`` value, which is what it reads
    there (zero padding, minus infinity for max pooling). Build with ``K.strided``."""

    __slots__ = ("terms", "offset", "exact", "_hash")

    def __init__(self, terms: Sequence[Term], offset: int = 0, exact: int = 1):
        merged: Dict[tuple, int] = {}
        for t in terms:
            if not isinstance(t, Term):
                raise IRError(f"strided index terms must be Term objects, got {t!r}")
            merged[t.key()] = merged.get(t.key(), 0) + t.coeff
        if not isinstance(offset, int) or isinstance(offset, bool):
            raise IRError(f"strided index offset must be an int, got {offset!r}")
        if not isinstance(exact, int) or isinstance(exact, bool) or exact < 1:
            raise IRError(f"strided index divisor must be a positive int, got {exact!r}")
        self.terms = tuple(Term(k[0], c, k[1], k[2], k[3]) for k, c in merged.items() if c != 0)
        self.offset = offset
        self.exact = exact
        self._hash = hash(("Strided", self.terms, offset, exact))

    def __eq__(self, other):
        return (isinstance(other, Strided) and other.terms == self.terms
                and other.offset == self.offset and other.exact == self.exact)

    def __hash__(self):
        return self._hash

    def axes(self) -> Tuple[Axis, ...]:
        out: List[Axis] = []
        for t in self.terms:
            if t.axis not in out:
                out.append(t.axis)
        return tuple(out)

    def numerator_bounds(self) -> Optional[Tuple[int, int]]:
        lo = hi = self.offset
        for t in self.terms:
            b = t.bounds()
            if b is None:
                return None
            lo += b[0]
            hi += b[1]
        return lo, hi

    def always_valid(self, size) -> bool:
        """True when the index is provably exact and inside [0, size) for every axis value."""
        if self.exact != 1 or not isinstance(size, int):
            return False
        b = self.numerator_bounds()
        return b is not None and b[0] >= 0 and b[1] < size

    def rename(self, f) -> "Strided":
        """The same index with every axis mapped through f (an Axis, or an int to fix it)."""
        terms, off = [], self.offset
        for t in self.terms:
            a = f(t.axis)
            if isinstance(a, Axis):
                terms.append(Term(a, t.coeff, t.mul, t.div, t.mod))
            else:
                off += t.value_at(int(a))
        return Strided(terms, off, self.exact)

    def text(self, nm) -> str:
        parts = [t.text(nm(t.axis)) for t in self.terms]
        if self.offset or not parts:
            parts.append(str(self.offset))
        s = " + ".join(parts).replace("+ -", "- ")
        return f"({s})" if self.exact == 1 else f"({s})/{self.exact}"

    def __repr__(self):
        return self.text(lambda a: a.name)


def strided(terms, offset: int = 0, exact: int = 1):
    """A strided index from ``[(axis, coeff), ...]`` or Term objects, plus an offset and an exact
    divisor. An index that is just one axis comes back as that Axis."""
    ts = [t if isinstance(t, Term) else Term(t[0], *t[1:]) for t in terms]
    s = Strided(ts, offset, exact)
    if (len(s.terms) == 1 and s.offset == 0 and s.exact == 1
            and (s.terms[0].coeff, s.terms[0].mul, s.terms[0].div, s.terms[0].mod) == (1, 1, 1, 0)):
        return s.terms[0].axis
    return s


# a load index is an axis, a constant (0 for broadcast dims) or a strided index
IndexAtom = Union[Axis, int, Strided]


# --------------------------------------------------------------------------- expressions


UNARY_OPS = ("neg", "exp", "log", "sqrt", "rsqrt", "sigmoid", "tanh", "abs", "not", "erf", "cos", "sin", "expm1")
BINARY_OPS = ("add", "sub", "mul", "div", "maximum", "minimum",
              "lt", "le", "gt", "ge", "eq", "ne", "and", "or")
COMPARE_OPS = ("lt", "le", "gt", "ge", "eq", "ne")
LOGIC_OPS = ("and", "or")
REDUCE_OPS = ("sum", "max", "min", "osum")
# osum: an order free float32 sum (kernel/ordersum.py, docs/ORDER_FREE_SUM.md): the same bits for
# any order, block size, split or SM count; same value as "sum" in the float64 reference


class Expr:
    """Base class. Subclasses set `_key` (a tuple of plain values and child nodes)."""

    __slots__ = ("_key", "_hash", "_free")

    def _init(self, key):
        self._key = key
        self._hash = hash((type(self).__name__,) + key)
        self._free = None

    def __eq__(self, other):
        if self is other:
            return True
        return (type(other) is type(self) and other._hash == self._hash
                and other._key == self._key)

    def __hash__(self):
        return self._hash

    def __bool__(self):
        raise IRError("an IR expression has no truth value; use where() instead of if/and/or")

    # arithmetic
    def __add__(self, o): return Binary("add", self, as_expr(o))
    def __radd__(self, o): return Binary("add", as_expr(o), self)
    def __sub__(self, o): return Binary("sub", self, as_expr(o))
    def __rsub__(self, o): return Binary("sub", as_expr(o), self)
    def __mul__(self, o): return Binary("mul", self, as_expr(o))
    def __rmul__(self, o): return Binary("mul", as_expr(o), self)
    def __truediv__(self, o): return Binary("div", self, as_expr(o))
    def __rtruediv__(self, o): return Binary("div", as_expr(o), self)
    def __neg__(self): return Unary("neg", self)

    # comparisons build boolean expressions (== stays structural equality; use eq())
    def __lt__(self, o): return Binary("lt", self, as_expr(o))
    def __le__(self, o): return Binary("le", self, as_expr(o))
    def __gt__(self, o): return Binary("gt", self, as_expr(o))
    def __ge__(self, o): return Binary("ge", self, as_expr(o))
    def __and__(self, o): return Binary("and", self, as_expr(o))
    def __rand__(self, o): return Binary("and", as_expr(o), self)
    def __or__(self, o): return Binary("or", self, as_expr(o))
    def __ror__(self, o): return Binary("or", as_expr(o), self)
    def __invert__(self): return Unary("not", self)

    def children(self) -> Tuple["Expr", ...]:
        return tuple(k for k in self._key if isinstance(k, Expr))

    def __repr__(self):
        return to_text(self)


class Const(Expr):
    __slots__ = ("value",)

    def __init__(self, value):
        if isinstance(value, bool):
            v = bool(value)
        else:
            v = float(value)
        self.value = v
        # -0.0 and 0.0 differ for codegen of 1/x; repr keeps them apart, nan compares by repr
        self._init((repr(v),))


class Dim(Expr):
    """The value of a size (an int or a symbol bound at call time)."""

    __slots__ = ("size",)

    def __init__(self, size: Size):
        self.size = _check_size(size)
        self._init((repr(self.size),))


class Index(Expr):
    """The current index along an axis, as a number."""

    __slots__ = ("axis",)

    def __init__(self, axis: Axis):
        if not isinstance(axis, Axis):
            raise IRError(f"Index needs an Axis, got {axis!r}")
        self.axis = axis
        self._init((axis,))


class Buffer:
    """A named input buffer with a shape and a dtype."""

    __slots__ = ("name", "shape", "dtype", "_hash")

    def __init__(self, name: str, shape: Sequence[Size], dtype: str = "float32"):
        if dtype not in DTYPES:
            raise IRError(f"unsupported dtype {dtype!r}; supported: {', '.join(DTYPES)}")
        self.name = name
        self.shape = tuple(_check_size(s) for s in shape)
        self.dtype = dtype
        self._hash = hash(("Buffer", name, self.shape, dtype))

    def __eq__(self, other):
        return (isinstance(other, Buffer) and other.name == self.name
                and other.shape == self.shape and other.dtype == self.dtype)

    def __hash__(self):
        return self._hash

    def __getitem__(self, idx) -> "Load":
        if not isinstance(idx, tuple):
            idx = (idx,)
        return Load(self, idx)

    def __repr__(self):
        return f"{self.name}{list(self.shape)}:{self.dtype}"


class Load(Expr):
    """Element of an input buffer. With a strided index that can fall outside the buffer (or is
    not exactly divisible), ``pad`` is the value read there; it is required in that case."""

    __slots__ = ("buffer", "index", "pad")

    def __init__(self, buffer: Buffer, index: Sequence[IndexAtom], pad: Optional[float] = None):
        index = tuple(index)
        needs_pad = False
        for k, (a, s) in enumerate(zip(index, buffer.shape)):
            if isinstance(a, Strided) and not a.always_valid(s):
                needs_pad = True
        if pad is not None:
            pad = float(pad)
        if needs_pad and pad is None:
            raise IRError(f"load of {buffer.name}: a strided index may fall outside the buffer or not "
                          f"divide exactly; give the value read there (pad=)")
        self.pad = pad if needs_pad else None
        if len(index) != len(buffer.shape):
            raise IRError(f"load of {buffer.name} has {len(index)} indices, "
                          f"buffer has {len(buffer.shape)} dims")
        seen = set()
        for k, (a, s) in enumerate(zip(index, buffer.shape)):
            if isinstance(a, Axis):
                if a.size != s:
                    raise IRError(f"load of {buffer.name} dim {k}: axis {a.name} has size "
                                  f"{a.size}, buffer dim is {s}")
                if a in seen:
                    raise IRError(f"load of {buffer.name} repeats axis {a.name}")
                seen.add(a)
            elif isinstance(a, int) and not isinstance(a, bool):
                if isinstance(s, int) and not 0 <= a < s:
                    raise IRError(f"load of {buffer.name} dim {k}: constant index {a} out of range")
            elif not isinstance(a, Strided):
                raise IRError(f"load index must be an Axis, an int or a strided index, got {a!r}")
        self.buffer = buffer
        self.index = index
        self._init((buffer, index, repr(self.pad)))

    @property
    def is_strided(self) -> bool:
        return any(isinstance(a, Strided) for a in self.index)

    def axes(self) -> Tuple[Axis, ...]:
        """Every axis the load depends on, in order of first appearance."""
        out: List[Axis] = []
        for a in self.index:
            for x in ((a,) if isinstance(a, Axis) else a.axes() if isinstance(a, Strided) else ()):
                if x not in out:
                    out.append(x)
        return tuple(out)

    def children(self):
        return ()


class Unary(Expr):
    __slots__ = ("op", "x")

    def __init__(self, op: str, x):
        if op not in UNARY_OPS:
            raise IRError(f"unknown unary op {op!r}")
        self.op = op
        self.x = as_expr(x)
        self._init((op, self.x))


class Binary(Expr):
    __slots__ = ("op", "a", "b")

    def __init__(self, op: str, a, b):
        if op not in BINARY_OPS:
            raise IRError(f"unknown binary op {op!r}")
        self.op = op
        self.a = as_expr(a)
        self.b = as_expr(b)
        self._init((op, self.a, self.b))


class Where(Expr):
    __slots__ = ("cond", "a", "b")

    def __init__(self, cond, a, b):
        self.cond = as_expr(cond)
        self.a = as_expr(a)
        self.b = as_expr(b)
        self._init((self.cond, self.a, self.b))


class Reduce(Expr):
    __slots__ = ("op", "axis", "body")

    def __init__(self, op: str, axis: Axis, body):
        if op not in REDUCE_OPS:
            raise IRError(f"unknown reduction {op!r}; supported: {', '.join(REDUCE_OPS)}")
        if not isinstance(axis, Axis):
            raise IRError(f"reduction axis must be an Axis, got {axis!r}")
        self.op = op
        self.axis = axis
        self.body = as_expr(body)
        self._init((op, axis, self.body))


class Placeholder(Expr):
    """Stands in for a value computed outside an expression (used by schedulers)."""

    __slots__ = ("tag",)

    def __init__(self, tag: str):
        self.tag = tag
        self._init((tag,))


def as_expr(x) -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, Axis):
        return Index(x)
    if isinstance(x, (int, float, bool)):
        return Const(x)
    raise IRError(f"cannot use {x!r} ({type(x).__name__}) in a kernel expression")


# --------------------------------------------------------------------------- builder helpers


def sym(name: str) -> Sym:
    return Sym(name)


def axis(name: str, size: Size) -> Axis:
    return Axis(name, size)


def buffer(name: str, shape: Sequence[Size], dtype: str = "float32") -> Buffer:
    return Buffer(name, shape, dtype)


def const(v) -> Const:
    return Const(v)


def index(a: Axis) -> Index:
    return Index(a)


def dim(s: Size) -> Expr:
    return Const(s) if isinstance(s, int) else Dim(s)


def exp(x): return Unary("exp", x)
def log(x): return Unary("log", x)
def sqrt(x): return Unary("sqrt", x)
def rsqrt(x): return Unary("rsqrt", x)
def sigmoid(x): return Unary("sigmoid", x)
def tanh(x): return Unary("tanh", x)
def abs_(x): return Unary("abs", x)
def erf(x): return Unary("erf", x)
def maximum(a, b): return Binary("maximum", a, b)
def minimum(a, b): return Binary("minimum", a, b)
def relu(x): return Binary("maximum", x, Const(0.0))
def eq(a, b): return Binary("eq", a, b)
def ne(a, b): return Binary("ne", a, b)
def where(c, a, b): return Where(c, a, b)


NEG_INF = float("-inf")


def reduce_sum(ax: Axis, body) -> Reduce:
    return Reduce("sum", ax, body)


def order_free_sum(ax: Axis, body) -> Reduce:
    """Sum whose float32 result has the same bits for every summation order (docs/ORDER_FREE_SUM.md)."""
    return Reduce("osum", ax, body)


def to_order_free(e: Expr) -> Expr:
    """e with every sum replaced by an order free sum."""
    return substitute_reduces(e, lambda r: Reduce("osum", r.axis, r.body) if r.op == "sum" else r)


def substitute_reduces(e: Expr, fn) -> Expr:
    """Rebuild e bottom up, passing every Reduce (with rebuilt body) through fn."""
    memo: Dict = {}

    def go(n):
        if n in memo:
            return memo[n]
        if isinstance(n, Reduce):
            out = fn(Reduce(n.op, n.axis, go(n.body)))
        elif isinstance(n, Unary):
            out = Unary(n.op, go(n.x))
        elif isinstance(n, Binary):
            out = Binary(n.op, go(n.a), go(n.b))
        elif isinstance(n, Where):
            out = Where(go(n.cond), go(n.a), go(n.b))
        else:
            out = n
        memo[n] = out
        return out

    return go(e)


def reduce_max(ax: Axis, body) -> Reduce:
    return Reduce("max", ax, body)


def reduce_min(ax: Axis, body) -> Reduce:
    return Reduce("min", ax, body)


def softmax_weight(score, over: Axis, tag: str = "") -> Expr:
    """exp(s - max_r s) / sum_r exp(s - max_r s), where `score` depends on axis `over`.

    This is the canonical softmax form the attention scheduler recognizes; the graph level
    softmax op produces the same form.
    """
    r1 = Axis(f"{over.name}_mx{tag}", over.size)
    r2 = Axis(f"{over.name}_sm{tag}", over.size)
    m = Reduce("max", r1, substitute(score, {over: r1}))
    l = Reduce("sum", r2, exp(substitute(score, {over: r2}) - m))
    return exp(score - m) / l


class Output:
    __slots__ = ("name", "axes", "expr", "dtype")

    def __init__(self, name: str, axes: Sequence[Axis], expr, dtype: Optional[str] = None):
        self.name = name
        self.axes = tuple(axes)
        self.expr = as_expr(expr)
        if len(set(self.axes)) != len(self.axes):
            raise IRError(f"output {name} repeats an axis")
        for a in self.axes:
            if not isinstance(a, Axis):
                raise IRError(f"output {name} axes must be Axis objects, got {a!r}")
        extra = free_axes(self.expr) - set(self.axes)
        if extra:
            raise IRError(f"output {name} uses axes {sorted(a.name for a in extra)} that are "
                          f"neither output axes nor reduced")
        if dtype is None:
            loads = [n for n in walk(self.expr) if isinstance(n, Load)]
            dtype = loads[0].buffer.dtype if loads else "float32"
        if dtype not in DTYPES:
            raise IRError(f"unsupported output dtype {dtype!r}")
        self.dtype = dtype

    @property
    def shape(self):
        return tuple(a.size for a in self.axes)

    def __repr__(self):
        return f"{self.name}[{', '.join(a.name for a in self.axes)}] = {to_text(self.expr)}"


class Kernel:
    """A fused kernel: outputs computed from input buffers."""

    def __init__(self, name: str, outputs: Sequence[Output]):
        if not outputs:
            raise IRError(f"kernel {name} has no outputs")
        self.name = name
        self.outputs = list(outputs)
        names = [o.name for o in self.outputs]
        if len(set(names)) != len(names):
            raise IRError(f"kernel {name} has duplicate output names")
        bufs: Dict[str, Buffer] = {}
        for o in self.outputs:
            for n in walk(o.expr):
                if isinstance(n, Load):
                    b = n.buffer
                    if b.name in bufs and bufs[b.name] != b:
                        raise IRError(f"kernel {name}: two different buffers named {b.name}")
                    bufs[b.name] = b
        clash = set(bufs) & set(names)
        if clash:
            raise IRError(f"kernel {name}: names used as both input and output: {sorted(clash)}")
        self.inputs: List[Buffer] = list(bufs.values())  # order of first appearance

    def symbols(self) -> List[Sym]:
        if getattr(self, "_symbols", None) is None:
            self._symbols = self._find_symbols()
        return list(self._symbols)

    def _find_symbols(self) -> List[Sym]:
        seen: Dict[str, Sym] = {}
        for b in self.inputs:
            for s in b.shape:
                if isinstance(s, Sym):
                    seen.setdefault(s.name, s)
        for o in self.outputs:
            for n in walk(o.expr):
                for s in _node_sizes(n):
                    if isinstance(s, Sym):
                        seen.setdefault(s.name, s)
            for a in o.axes:
                if isinstance(a.size, Sym):
                    seen.setdefault(a.size.name, a.size)
        return list(seen.values())

    def reduces(self) -> List[Reduce]:
        out, seen = [], set()
        for o in self.outputs:
            for n in walk(o.expr):
                if isinstance(n, Reduce) and n not in seen:
                    seen.add(n)
                    out.append(n)
        return out

    def to_text(self) -> str:
        lines = [f"kernel {self.name}("
                 + ", ".join(repr(b) for b in self.inputs) + ")"]
        for o in self.outputs:
            lines.append(f"  {o.name}:{o.dtype}[{', '.join(repr(a) for a in o.axes)}] = "
                         f"{to_text(o.expr)}")
        return "\n".join(lines)

    def __repr__(self):
        return self.to_text()


def kernel(name: str, outputs: Sequence[Output]) -> Kernel:
    return Kernel(name, outputs)


def output(name: str, axes: Sequence[Axis], expr, dtype: Optional[str] = None) -> Output:
    return Output(name, axes, expr, dtype)


# --------------------------------------------------------------------------- analysis


def _node_sizes(n: Expr):
    if isinstance(n, Dim):
        yield n.size
    elif isinstance(n, Index):
        yield n.axis.size
    elif isinstance(n, Reduce):
        yield n.axis.size
    elif isinstance(n, Load):
        yield from n.buffer.shape
        for a in n.axes():
            yield a.size


def walk(e: Expr) -> Iterator[Expr]:
    """Every distinct node under e (post order, children first), including inside reductions."""
    seen = set()
    stack = [(e, False)]
    while stack:
        n, done = stack.pop()
        if done:
            yield n
            continue
        if n in seen:
            continue
        seen.add(n)
        stack.append((n, True))
        for c in reversed(n.children()):
            if c not in seen:
                stack.append((c, False))


def free_axes(e: Expr) -> frozenset:
    """Axes an expression depends on (reduction axes are bound, not free)."""
    if e._free is not None:
        return e._free
    if isinstance(e, Index):
        f = frozenset([e.axis])
    elif isinstance(e, Load):
        f = frozenset(e.axes())
    elif isinstance(e, Reduce):
        f = free_axes(e.body) - {e.axis}
    else:
        f = frozenset()
        for c in e.children():
            f = f | free_axes(c)
    e._free = f
    return f


def contains_reduce(e: Expr) -> bool:
    return any(isinstance(n, Reduce) for n in walk(e))


def top_reduces(e: Expr) -> List[Reduce]:
    """Reductions in e that are not inside another reduction."""
    out, seen = [], set()

    def go(n):
        if n in seen:
            return
        seen.add(n)
        if isinstance(n, Reduce):
            out.append(n)
            return
        for c in n.children():
            go(c)

    go(e)
    return out


def substitute(e: Expr, mapping: Dict) -> Expr:
    """Replace axes (Axis to Axis) and nodes (Expr to Expr) throughout e."""
    memo: Dict[Expr, Expr] = {}

    def ax(a):
        return mapping.get(a, a) if isinstance(a, Axis) else a

    def go(n: Expr) -> Expr:
        if n in memo:
            return memo[n]
        if n in mapping and isinstance(n, Expr):
            r = mapping[n]
        elif isinstance(n, (Const, Dim, Placeholder)):
            r = n
        elif isinstance(n, Index):
            r = Index(ax(n.axis))
        elif isinstance(n, Load):
            r = Load(n.buffer, tuple(a.rename(ax) if isinstance(a, Strided) else ax(a) for a in n.index),
                     n.pad)
        elif isinstance(n, Unary):
            r = Unary(n.op, go(n.x))
        elif isinstance(n, Binary):
            r = Binary(n.op, go(n.a), go(n.b))
        elif isinstance(n, Where):
            r = Where(go(n.cond), go(n.a), go(n.b))
        elif isinstance(n, Reduce):
            r = Reduce(n.op, ax(n.axis), go(n.body))
        else:
            raise IRError(f"substitute: unknown node {type(n).__name__}")
        memo[n] = r
        return r

    return go(e)


def reduction_length(e: Expr) -> int:
    """Longest chain of nested reduction sizes, product (symbols count as 1024)."""
    def size(a):
        return a.size if isinstance(a.size, int) else 1024

    best = 1

    def go(n, acc):
        nonlocal best
        if isinstance(n, Reduce):
            acc = acc * size(n.axis)
            best = max(best, acc)
        for c in n.children():
            go(c, acc)

    go(e, 1)
    return best


# --------------------------------------------------------------------------- text form


_SYMBOL = {"add": "+", "sub": "-", "mul": "*", "div": "/", "lt": "<", "le": "<=", "gt": ">",
           "ge": ">=", "eq": "==", "ne": "!=", "and": "&", "or": "|"}


def to_text(e: Expr, names: Optional[Dict] = None) -> str:
    """Readable, deterministic text. `names` renames axes and buffers (used for canonical forms)."""
    names = names or {}

    def nm(x):
        return names.get(x, x.name)

    def go(n):
        if isinstance(n, Const):
            return n._key[0]
        if isinstance(n, Dim):
            return f"dim({names.get(n.size, n.size)!r})" if isinstance(n.size, Sym) else str(n.size)
        if isinstance(n, Index):
            return f"idx({nm(n.axis)})"
        if isinstance(n, Load):
            ix = ", ".join(nm(a) if isinstance(a, Axis) else a.text(nm) if isinstance(a, Strided)
                           else str(a) for a in n.index)
            pad = "" if n.pad is None else f" pad {n.pad!r}"
            return f"{nm(n.buffer)}[{ix}{pad}]"
        if isinstance(n, Unary):
            return f"{n.op}({go(n.x)})"
        if isinstance(n, Binary):
            if n.op in _SYMBOL:
                return f"({go(n.a)} {_SYMBOL[n.op]} {go(n.b)})"
            return f"{n.op}({go(n.a)}, {go(n.b)})"
        if isinstance(n, Where):
            return f"where({go(n.cond)}, {go(n.a)}, {go(n.b)})"
        if isinstance(n, Reduce):
            return f"{n.op}[{nm(n.axis)}]({go(n.body)})"
        if isinstance(n, Placeholder):
            return f"<{n.tag}>"
        raise IRError(f"unknown node {type(n).__name__}")

    return go(e)


def canonicalize(k: Kernel) -> Tuple[Kernel, Dict[str, str], Dict[str, str]]:
    """Rename axes, buffers, symbols and outputs by order of first appearance.

    Two kernels that differ only in names become identical, so they share one compiled binary and
    an unrelated edit elsewhere in a model never changes this kernel's cache key.
    Returns (renamed kernel, input name -> canonical name, output name -> canonical name).
    """
    syms: Dict[str, Sym] = {}
    axes: Dict[Axis, Axis] = {}
    bufs: Dict[str, Buffer] = {}

    def s_(s):
        if isinstance(s, int):
            return s
        if s.name not in syms:
            syms[s.name] = Sym(f"s{len(syms)}")
        return syms[s.name]

    def a_(a):
        if not isinstance(a, Axis):
            return a
        if a not in axes:
            axes[a] = Axis(f"a{len(axes)}", s_(a.size))
        return axes[a]

    def b_(b):
        if b.name not in bufs:
            bufs[b.name] = Buffer(f"b{len(bufs)}", [s_(x) for x in b.shape], b.dtype)
        return bufs[b.name]

    memo: Dict[Expr, Expr] = {}

    def go(n):
        if n in memo:
            return memo[n]
        if isinstance(n, (Const, Placeholder)):
            r = n
        elif isinstance(n, Dim):
            r = Dim(s_(n.size))
        elif isinstance(n, Index):
            r = Index(a_(n.axis))
        elif isinstance(n, Load):
            nb = b_(n.buffer)
            r = Load(nb, tuple(a.rename(a_) if isinstance(a, Strided) else a_(a) for a in n.index), n.pad)
        elif isinstance(n, Unary):
            r = Unary(n.op, go(n.x))
        elif isinstance(n, Binary):
            a = go(n.a)
            r = Binary(n.op, a, go(n.b))
        elif isinstance(n, Where):
            c = go(n.cond)
            a = go(n.a)
            r = Where(c, a, go(n.b))
        elif isinstance(n, Reduce):
            ax = a_(n.axis)
            r = Reduce(n.op, ax, go(n.body))
        else:
            raise IRError(f"canonicalize: unknown node {type(n).__name__}")
        memo[n] = r
        return r

    outs = []
    for i, o in enumerate(k.outputs):
        oa = [a_(a) for a in o.axes]
        outs.append(Output(f"o{i}", oa, go(o.expr), o.dtype))
    ck = Kernel("canonical", outs)
    in_map = {name: b.name for name, b in bufs.items()}
    out_map = {o.name: f"o{i}" for i, o in enumerate(k.outputs)}
    return ck, in_map, out_map


def is_neg_inf(e: Expr) -> bool:
    return isinstance(e, Const) and isinstance(e.value, float) and e.value == -math.inf
