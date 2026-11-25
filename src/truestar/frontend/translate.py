"""Source to IR translation for ``@truestar.function``.

The user's function is never executed. Its AST is walked once, statement by statement, by an
abstract interpreter whose values are either compile time Python values (ints, floats, dims,
tuples, modules) or tensors (IR values). Python control flow is handled by kind:

* ``if`` on a compile time value picks a branch now; the other branch is not translated.
* ``if`` on tensor data translates both branches into regions of an IR ``cond``.
* ``a if c else b`` on tensor data becomes an elementwise ``where`` (a select).
* ``for`` over a static range or a tuple is unrolled.
* ``for`` over a symbolic range (``range(x.shape[1])``) becomes an IR ``loop`` with a symbolic
  trip count, so one compile serves every length.
* ``while`` on tensor data becomes an IR ``loop`` with a ``cond`` region.

Every op is shape checked the moment it is emitted, so a mismatch is reported at compile time with
the user's line, the op, both operand types and, when obvious, a fix. Anything not handled raises
an error naming the line; nothing is skipped.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import math
import operator
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sympy
import torch
import torch.nn.functional as F

from ..ir.core import (Builder, IRError, ShapeError, TensorType, UnsupportedOpError, Value,
                       dims_equal, norm_dim)
from ..ir.importer import gqa_factor, repeat_heads
from . import ops as user_ops
from .errors import (FrontendError, FrontendNameError, FrontendShapeError, StaticValueError,
                     UnsupportedConstructError)
from .types import DTYPES, TensorSpec

MAX_UNROLL = 10000
MAX_INLINE_DEPTH = 32


# ----------------------------------------------------------------------------- values


class T:
    """A tensor during translation: an IR value owned by one builder (program or region)."""

    __slots__ = ("v", "b")

    def __init__(self, v: Value, b: Builder):
        self.v = v
        self.b = b

    @property
    def type(self) -> TensorType:
        return self.v.type

    def __repr__(self):
        return f"T({self.v.ref}: {self.v.type})"


class Undefined:
    def __init__(self, reason: str):
        self.reason = reason


class BoundMethod:
    def __init__(self, obj: T, name: str):
        self.obj = obj
        self.name = name


class _Return:
    def __init__(self, value: Any, node: ast.AST):
        self.value = value
        self.node = node


MISSING = object()


def is_tensor(v) -> bool:
    return isinstance(v, T)


def is_dim(v) -> bool:
    return isinstance(v, sympy.Basic)


def is_number(v) -> bool:
    return isinstance(v, (bool, int, float)) or is_dim(v)


# ----------------------------------------------------------------------------- scopes


class Scope:
    """Variables of one program or region. A region scope reads outer variables through ``parent``;
    any outer tensor it uses becomes an explicit capture (a region input), never an implicit read."""

    def __init__(self, b: Builder, parent: Optional["Scope"] = None, loop_depth: int = 0):
        self.b = b
        self.parent = parent
        self.vars: Dict[str, Any] = {}
        self.cap_order: List[T] = []  # outer tensors (owned by parent.b)
        self.cap_map: Dict[int, T] = {}  # id(outer value) -> inner tensor
        self.loop_depth = loop_depth

    def lookup(self, name: str):
        s = self
        while s is not None:
            if name in s.vars:
                return s.vars[name]
            s = s.parent
        return MISSING

    def adopt(self, t: T) -> T:
        """The tensor as seen from this scope: captured through every region boundary it crosses."""
        if t.b is self.b:
            return t
        if self.parent is None:
            raise IRError(f"internal: tensor {t} is not visible from this scope")
        outer = self.parent.adopt(t)
        key = id(outer.v)
        if key not in self.cap_map:
            inner = T(self.b.input(outer.v.name, outer.type.dtype, outer.type.shape), self.b)
            self.cap_map[key] = inner
            self.cap_order.append(outer)
        return self.cap_map[key]


def _region(scope: Scope, label: str, loop: bool = False) -> Scope:
    b = Builder(label)
    b.symbols = dict(scope.b.symbols)
    return Scope(b, scope, scope.loop_depth + (1 if loop else 0))


def _finish_regions(parts: List[Tuple[Scope, List[T], List[T]]]):
    """Give every region of one node the same capture list (union, in first use order) and build
    them. ``parts`` holds (scope, prefix inputs, outputs). Returns (programs, outer capture tensors)."""
    outs_adopted = [[sc.adopt(o) for o in outs] for sc, _, outs in parts]
    union: List[T] = []
    seen = set()
    for sc, _, _ in parts:
        for o in sc.cap_order:
            if id(o.v) not in seen:
                seen.add(id(o.v))
                union.append(o)
    progs = []
    for (sc, prefix, _), outs in zip(parts, outs_adopted):
        caps = []
        for o in union:
            inner = sc.cap_map.get(id(o.v))
            if inner is None:
                inner = T(sc.b.input(o.v.name, o.type.dtype, o.type.shape), sc.b)
            caps.append(inner.v)
        sc.b.inputs = [p.v for p in prefix] + caps
        progs.append(sc.b.build([o.v for o in outs]))
    return progs, union


def assigned_names(stmts: Sequence[ast.stmt]) -> List[str]:
    """Names stored anywhere in the statements, in first appearance order."""
    out: List[str] = []

    class V(ast.NodeVisitor):
        def visit_Name(self, n):
            if isinstance(n.ctx, ast.Store) and n.id not in out:
                out.append(n.id)

        def visit_FunctionDef(self, n):
            pass

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

    for s in stmts:
        V().visit(s)
    return out


def always_returns(stmts: Sequence[ast.stmt]) -> bool:
    if not stmts:
        return False
    last = stmts[-1]
    if isinstance(last, ast.Return):
        return True
    if isinstance(last, ast.If):
        return always_returns(last.body) and always_returns(last.orelse)
    return False


# ----------------------------------------------------------------------------- source


class FunctionSource:
    """The parsed source of a user function, with what is needed to report exact locations."""

    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        try:
            lines, start = inspect.getsourcelines(fn)
            self.file = inspect.getsourcefile(fn) or "<unknown>"
        except (OSError, TypeError) as e:
            raise FrontendError(f"cannot read the source of {self.name}: {e}. Native functions must be "
                                "defined in a file", code="source_unavailable", function=self.name) from None
        self.start = start
        self.lines = lines
        self.text = textwrap.dedent("".join(lines))
        self.indent = len(lines[0]) - len(lines[0].lstrip())
        tree = ast.parse(self.text)
        defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not defs:
            raise FrontendError(f"no function definition found for {self.name}", function=self.name,
                                code="source_unavailable")
        self.tree = defs[0]
        if isinstance(self.tree, ast.AsyncFunctionDef):
            raise self.error(self.tree, "async functions are not supported", UnsupportedConstructError)
        self.globals = fn.__globals__
        self.cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))

    def line_of(self, node: ast.AST) -> int:
        return self.start + getattr(node, "lineno", 1) - 1

    def source_line(self, node: ast.AST) -> str:
        i = getattr(node, "lineno", 1) - 1
        return self.lines[i].rstrip("\n") if 0 <= i < len(self.lines) else ""

    def segment(self, node: ast.AST) -> str:
        try:
            s = ast.get_source_segment(self.text, node)
        except Exception:
            s = None
        return s if s else ast.unparse(node)

    def where(self, node: ast.AST) -> str:
        return f"{self.file}:{self.line_of(node)} in {self.name}: {self.source_line(node).strip()}"

    def error(self, node: ast.AST, message: str, cls=FrontendError, **kw) -> FrontendError:
        return cls(message, function=self.name, file=self.file, line=self.line_of(node),
                   col=getattr(node, "col_offset", 0) + self.indent, source_line=self.source_line(node), **kw)

    def resolve_global(self, name: str):
        if name in self.cells:
            try:
                return self.cells[name].cell_contents
            except ValueError:
                return MISSING
        if name in self.globals:
            return self.globals[name]
        if hasattr(builtins, name):
            return getattr(builtins, name)
        return MISSING


# ----------------------------------------------------------------------------- op tables

_UNARY_OPS = {"relu", "sigmoid", "tanh", "silu", "exp", "sqrt", "rsqrt", "cos", "sin", "log"}
_LAYER_OPS = ("expm1", "log_softmax", "conv2d", "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d",
              "conv_transpose2d", "group_norm")

# torch callables accepted as aliases of truestar ops (same argument conventions)
TORCH_ALIASES = {
    torch.relu: "relu", torch.sigmoid: "sigmoid", torch.tanh: "tanh", torch.exp: "exp",
    torch.sqrt: "sqrt", torch.rsqrt: "rsqrt", torch.abs: "abs", torch.matmul: "matmul",
    torch.softmax: "softmax", torch.where: "where", torch.cat: "cat", torch.maximum: "maximum",
    torch.minimum: "minimum", torch.zeros_like: "zeros_like", torch.ones_like: "ones_like",
    torch.clamp: "clamp", torch.sum: "sum", torch.mean: "mean", torch.reshape: "reshape",
    torch.permute: "permute", torch.transpose: "transpose",
    F.relu: "relu", F.gelu: "gelu", F.silu: "silu", F.softmax: "softmax", F.layer_norm: "layer_norm",
    F.linear: "linear", F.scaled_dot_product_attention: "attention",
    # decoder LLM ops (2026-11-24); torch.multinomial is deliberately absent: its generator differs
    torch.cos: "cos", torch.sin: "sin", torch.log: "log", torch.cumsum: "cumsum", torch.flip: "flip",
    torch.argmax: "argmax", torch.argmin: "argmin", torch.amax: "amax", torch.amin: "amin", torch.sort: "sort", torch.topk: "topk",
    torch.gather: "gather", torch.index_select: "index_select", torch.index_copy: "index_copy",
    torch.chunk: "chunk", torch.split: "split", torch.repeat_interleave: "repeat_interleave",
    F.rms_norm: "rms_norm",
    F.dropout: "dropout",   # its training flag must be a compile time value; the mask is keyed
}

# PyTorch's random functions read a global generator whose bits differ per device; they are refused
# with the keyed op to use instead (docs/RANDOM.md).
_TORCH_RANDOM = {torch.rand: "rand", torch.randn: "randn", torch.bernoulli: "bernoulli", torch.randint: "randint",
                 torch.rand_like: "rand_like", torch.randn_like: "randn_like", torch.normal: "normal",
                 torch.multinomial: "multinomial"}
_RANDOM_OPS = ("rand", "randn", "normal", "bernoulli", "randint", "rand_like", "randn_like", "dropout")

_BIN = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.MatMult: "matmul"}
_PYBIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
          ast.Pow: operator.pow, ast.MatMult: operator.matmul, ast.LShift: operator.lshift,
          ast.RShift: operator.rshift, ast.BitAnd: operator.and_, ast.BitOr: operator.or_,
          ast.BitXor: operator.xor}
_CMP = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}
_PYCMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
          ast.Gt: operator.gt, ast.GtE: operator.ge}
_SYMCMP = {ast.Eq: sympy.Eq, ast.NotEq: sympy.Ne, ast.Lt: sympy.Lt, ast.LtE: sympy.Le,
           ast.Gt: sympy.Gt, ast.GtE: sympy.Ge}
_STATIC_BUILTINS = {range, len, min, max, abs, int, float, bool, tuple, list, dict, enumerate, zip,
                    isinstance, round, reversed, sorted, str}

_UNSUPPORTED_STMT = {
    ast.Try: "try blocks", ast.With: "with blocks", ast.FunctionDef: "nested function definitions",
    ast.AsyncFunctionDef: "nested function definitions", ast.ClassDef: "class definitions",
    ast.Global: "global", ast.Nonlocal: "nonlocal", ast.Delete: "del", ast.Raise: "raise",
    ast.Import: "import inside a native function", ast.ImportFrom: "import inside a native function",
    ast.AsyncFor: "async for", ast.AsyncWith: "async with",
}
if hasattr(ast, "Match"):
    _UNSUPPORTED_STMT[ast.Match] = "match statements"
if hasattr(ast, "TryStar"):
    _UNSUPPORTED_STMT[ast.TryStar] = "try blocks"

_UNSUPPORTED_EXPR = {
    ast.Lambda: "lambda", ast.ListComp: "list comprehensions", ast.SetComp: "set comprehensions",
    ast.DictComp: "dict comprehensions", ast.GeneratorExp: "generator expressions",
    ast.Yield: "yield", ast.YieldFrom: "yield", ast.Await: "await", ast.JoinedStr: "f strings",
    ast.NamedExpr: "the walrus operator", ast.Set: "set literals",
    ast.Starred: "starred expressions",
}


_CONTAINER_METHODS = {"items", "keys", "values", "get", "index", "count"}


def _is_structure_class(func) -> bool:
    import dataclasses
    return isinstance(func, type) and (dataclasses.is_dataclass(func)
                                       or (issubclass(func, tuple) and hasattr(func, "_fields")))


def _build_structure(cls, args, kwargs):
    """An instance of a dataclass or NamedTuple holding translation values. A dataclass is filled
    field by field without running its __init__ or __post_init__ (they may do tensor math the
    translator cannot see)."""
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return cls(*args, **kwargs)
    names = [f.name for f in dataclasses.fields(cls)]
    if len(args) > len(names):
        raise TypeError(f"{cls.__name__} has {len(names)} fields, got {len(args)} values")
    vals = dict(zip(names, args))
    for k, v in kwargs.items():
        if k not in names:
            raise TypeError(f"{cls.__name__} has no field {k!r} (fields: {', '.join(names)})")
        if k in vals:
            raise TypeError(f"field {k!r} given twice")
        vals[k] = v
    defaults = {f.name: f.default for f in dataclasses.fields(cls) if f.default is not dataclasses.MISSING}
    for n in names:
        if n not in vals:
            if n not in defaults:
                raise TypeError(f"missing field {n!r}")
            vals[n] = defaults[n]
    obj = object.__new__(cls)
    for n in names:
        object.__setattr__(obj, n, vals[n])
    return obj


def _fmt_type(t) -> str:
    return str(t.type) if is_tensor(t) else repr(t)


# ----------------------------------------------------------------------------- translator


class RandomSites:
    """Keys and site names of the random ops of one compiled program (docs/RANDOM.md).

    A site is ``<function path>/<op>#<k>`` (k counts that op in the function's source) or
    ``<function path>/<name>`` when the call gives ``name=``. Inlined calls add ``<callee>#<k>`` to
    the path, unrolled static loops add ``[i]``, and loops over a symbolic length or a tensor condition
    pass their iteration index as a stream value, so every iteration draws fresh numbers. Two different
    ops with one site are a compile error."""

    def __init__(self, top: "Scope", fn_name: str):
        self.top = top
        self.path: List[str] = [fn_name]
        self.key: Optional[T] = None
        self.used: Dict[str, Tuple[int, int]] = {}
        self.ordinals: Dict[Tuple[str, int], int] = {}
        self.counters: Dict[Tuple[str, str], int] = {}
        self.unroll: List[int] = []
        self.streams: List[T] = []

    def program_key(self) -> T:
        """The int64 program input every implicitly keyed op reads (added on first use)."""
        if self.key is None:
            self.key = T(self.top.b.input("rng_key", torch.int64, ()), self.top.b)
        return self.key

    def ordinal(self, node: ast.AST, kind: str) -> int:
        path = "/".join(self.path)
        k = self.ordinals.get((path, id(node)))
        if k is None:
            k = self.counters.get((path, kind), 0)
            self.counters[(path, kind)] = k + 1
            self.ordinals[(path, id(node))] = k
        return k

    def site(self, tr: "Translator", node: ast.AST, kind: str, name: Optional[str]) -> str:
        if name is not None and (not isinstance(name, str) or not name or "/" in name):
            raise tr.err(node, f"name= must be a non empty string without '/', got {name!r}", code="type_error")
        base = name if name is not None else f"{kind}#{self.ordinal(node, kind)}"
        site = "/".join(self.path) + "/" + base + "".join(f"[{i}]" for i in self.unroll)
        prev = self.used.get(site)
        if prev is not None and prev[0] != id(node):
            raise tr.err(node, f"two random ops share the site {site!r} (the other is at line {prev[1]}), so they "
                               "would draw the same numbers", code="duplicate_site",
                         suggestion="give each op a different name=, or draw once and reuse the tensor")
        self.used[site] = (id(node), tr.src.line_of(node))
        return site


class Translator:
    def __init__(self, src: FunctionSource, depth: int = 0, rng: Optional[RandomSites] = None):
        self.src = src
        self.depth = depth
        self.rng = rng

    # -------------------------------------------------------------- errors and emission

    def err(self, node, message, cls=FrontendError, **kw):
        return self.src.error(node, message, cls, **kw)

    def unsupported(self, node, what, suggestion=None):
        return self.err(node, f"{what} {'is' if not what.endswith('s') else 'are'} not supported in a "
                              "native function", UnsupportedConstructError, suggestion=suggestion)

    def emit(self, scope: Scope, op: str, inputs: Sequence[Optional[T]], node: ast.AST,
             labels: Sequence[str] = (), **attrs) -> T:
        ins = [None if x is None else scope.adopt(x) for x in inputs]
        try:
            v = scope.b.op(op, *[None if x is None else x.v for x in ins], source=self.src.where(node), **attrs)
        except ShapeError as e:
            raise self._shape_error(e, op, ins, labels, node) from None
        except UnsupportedOpError as e:
            raise self.err(node, str(e), UnsupportedConstructError, op=op) from None
        return T(v, scope.b)

    def emit_multi(self, scope: Scope, op: str, inputs: Sequence[Optional[T]], node: ast.AST,
                   labels: Sequence[str] = (), **attrs) -> Tuple[T, ...]:
        """``emit`` for an op that defines several values (sort, topk)."""
        ins = [None if x is None else scope.adopt(x) for x in inputs]
        try:
            vs = scope.b.op_multi(op, *[None if x is None else x.v for x in ins], source=self.src.where(node), **attrs)
        except ShapeError as e:
            raise self._shape_error(e, op, ins, labels, node) from None
        except UnsupportedOpError as e:
            raise self.err(node, str(e), UnsupportedConstructError, op=op) from None
        return tuple(T(v, scope.b) for v in vs)

    def _shape_error(self, e: ShapeError, op: str, ins, labels, node) -> FrontendShapeError:
        operands = []
        for i, x in enumerate(ins):
            if x is None:
                continue
            label = labels[i] if i < len(labels) and labels[i] else f"operand {i}"
            operands.append((label, x.type))
        if not operands:
            operands = list(e.operands)
        return self.err(node, e.reason, FrontendShapeError, op=op, operands=operands,
                        suggestion=self._suggest(op, ins, labels))

    def _suggest(self, op, ins, labels) -> Optional[str]:
        ts = [x.type for x in ins if x is not None]
        lab = [l or f"operand {i}" for i, l in enumerate(list(labels) + [""] * len(ins))]
        if op == "matmul" and len(ts) == 2 and ts[0].rank >= 2 and ts[1].rank >= 2:
            a, b = ts
            if dims_equal(a.shape[-1], b.shape[-1]):
                return f"{lab[1]} looks transposed: try {lab[0]} @ {lab[1]}.T"
            if dims_equal(a.shape[-2], b.shape[-2]):
                return f"{lab[0]} looks transposed: try {lab[0]}.T @ {lab[1]}"
        if op == "linear" and len(ts) >= 2 and ts[1].rank == 2 and ts[0].rank >= 1:
            x, w = ts[0], ts[1]
            if dims_equal(x.shape[-1], w.shape[0]):
                return (f"linear expects weight as [out, in] but {lab[1]} is [in, out]: "
                        f"pass {lab[1]}.T, or write {lab[0]} @ {lab[1]}")
        if op in ("add", "sub", "mul", "div", "where") and len(ts) >= 2:
            a, b = ts[-2], ts[-1]
            if a.rank == b.rank == 2 and dims_equal(a.shape[0], b.shape[1]) and dims_equal(a.shape[1], b.shape[0]):
                return f"the operands are transposes of each other: try {lab[len(ts) - 1]}.T"
            if a.rank == b.rank and sorted(map(str, a.shape)) == sorted(map(str, b.shape)):
                return "the operands hold the same dims in a different order: permute one of them"
        if op == "layer_norm" and ts:
            return f"normalized_shape must equal the trailing dims of the input, for example [{ts[0].shape[-1]}]"
        return None

    def const(self, scope: Scope, value, like: Optional[TensorType], node) -> T:
        """A compile time number as a rank 0 tensor. Its dtype follows Python scalar rules against
        ``like``: it never widens a tensor's dtype."""
        if is_tensor(value):
            return value
        if isinstance(value, bool):
            dtype = torch.bool
        elif isinstance(value, int) or is_dim(value):
            dtype = like.dtype if like is not None and like.dtype != torch.bool else torch.int64
        elif isinstance(value, float):
            dtype = like.dtype if like is not None and like.dtype.is_floating_point else torch.float32
        else:
            raise self.err(node, f"expected a tensor or a number, got {type(value).__name__} {value!r}",
                           code="type_error")
        if is_dim(value):
            return self.emit(scope, "dim_scalar", [], node, expr=value, dtype=dtype)
        cache = scope.b.__dict__.setdefault("_ts_consts", {})
        key = (type(value), value, dtype)
        if key not in cache:
            # Typed like a Python scalar, but the stored value keeps the exact Python number (as the
            # importer does), so the float64 reference never inherits a float32 rounding.
            b = scope.b
            v = Value(b._fresh("const"), TensorType(dtype, ()), "param")
            b.params.append(v)
            b.param_data[v.name] = torch.tensor(value, dtype=torch.float64 if dtype.is_floating_point else dtype)
            cache[key] = T(v, b)
        return cache[key]

    def tensor_pair(self, scope, a, b, node):
        like_a = a.type if is_tensor(a) else None
        like_b = b.type if is_tensor(b) else None
        return self.const(scope, a, like_b, node), self.const(scope, b, like_a, node)

    # -------------------------------------------------------------- statements

    def block(self, stmts: Sequence[ast.stmt], scope: Scope) -> Optional[_Return]:
        stmts = list(stmts)
        i = 0
        while i < len(stmts):
            s = stmts[i]
            rest = stmts[i + 1:]
            if isinstance(s, ast.If):
                r, consumed = self.stmt_if(s, scope, rest)
                if consumed:
                    return r
                if r is not None:
                    return r
            else:
                r = self.stmt(s, scope)
                if r is not None:
                    return r
            i += 1
        return None

    def stmt(self, s: ast.stmt, scope: Scope) -> Optional[_Return]:
        if type(s) in _UNSUPPORTED_STMT:
            raise self.unsupported(s, _UNSUPPORTED_STMT[type(s)])
        if isinstance(s, ast.Return):
            if scope.loop_depth:
                raise self.unsupported(s, "return inside a loop over a symbolic length or tensor condition",
                                       suggestion="carry the value in a variable and return after the loop")
            value = None if s.value is None else self.expr(s.value, scope)
            return _Return(value, s)
        if isinstance(s, ast.Assign):
            value = self.expr(s.value, scope)
            for tgt in s.targets:
                self.assign(tgt, value, scope, s)
            return None
        if isinstance(s, ast.AnnAssign):
            if s.value is None:
                return None
            value = self.expr(s.value, scope)
            spec = self.expr(s.annotation, scope)
            if isinstance(spec, TensorSpec):
                self.check_spec(value, spec, s, f"{self.src.segment(s.target)}")
            self.assign(s.target, value, scope, s)
            return None
        if isinstance(s, ast.AugAssign):
            if not isinstance(s.target, ast.Name):
                raise self.unsupported(s, "augmented assignment to an attribute or subscript",
                                       suggestion="tensors are immutable: assign a new name")
            cur = self.name(s.target.id, scope, s.target)
            value = self.expr(s.value, scope)
            out = self.binop(s.op, cur, value, s, self.src.segment(s.target), self.src.segment(s.value), scope)
            self.assign(s.target, out, scope, s)
            return None
        if isinstance(s, ast.Expr):
            if isinstance(s.value, ast.Constant):
                return None  # docstring
            self.expr(s.value, scope)
            return None
        if isinstance(s, ast.Pass):
            return None
        if isinstance(s, ast.Assert):
            v = self.expr(s.test, scope)
            if is_tensor(v):
                raise self.unsupported(s, "assert on tensor data",
                                       suggestion="asserts must be about compile time facts such as shapes")
            if not self.truth(v, s.test):
                msg = self.expr(s.msg, scope) if s.msg is not None else ""
                raise self.err(s, f"assertion failed at compile time: {self.src.segment(s.test)}"
                                  + (f" ({msg})" if msg else ""), code="assertion_failed")
            return None
        if isinstance(s, ast.For):
            return self.stmt_for(s, scope)
        if isinstance(s, ast.While):
            return self.stmt_while(s, scope)
        if isinstance(s, (ast.Break, ast.Continue)):
            raise self.unsupported(s, "break and continue",
                                   suggestion="use a while loop whose condition covers the exit")
        raise self.unsupported(s, f"{type(s).__name__} statements")

    def assign(self, tgt: ast.expr, value, scope: Scope, node):
        if isinstance(tgt, ast.Name):
            scope.vars[tgt.id] = value
            return
        if isinstance(tgt, (ast.Tuple, ast.List)):
            if is_tensor(value):
                raise self.unsupported(node, "unpacking a tensor",
                                       suggestion="index it explicitly, for example a = x[0]")
            try:
                items = list(value)
            except TypeError:
                raise self.err(node, f"cannot unpack {value!r}", code="type_error") from None
            if len(items) != len(tgt.elts):
                raise self.err(node, f"cannot unpack {len(items)} values into {len(tgt.elts)} names",
                               code="type_error")
            for t, v in zip(tgt.elts, items):
                self.assign(t, v, scope, node)
            return
        if isinstance(tgt, (ast.Subscript, ast.Attribute)):
            raise self.unsupported(node, "assignment to a subscript or attribute",
                                   suggestion="tensors are immutable in a native function: build a new tensor "
                                              "with ops.where or ops.cat")
        raise self.unsupported(node, f"assignment to {type(tgt).__name__}")

    def check_spec(self, value, spec: TensorSpec, node, label: str):
        if not is_tensor(value):
            raise self.err(node, f"{label} is annotated {spec} but is {value!r}", FrontendShapeError,
                           code="annotation_mismatch")
        want = spec.to_type()
        if not value.type.same_as(want):
            raise self.err(node, f"{label} is annotated {want} but has type {value.type}", FrontendShapeError,
                           code="annotation_mismatch", operands=[(f"{label} (actual)", value.type),
                                                                 (f"{label} (annotation)", want)])

    # -------------------------------------------------------------- if

    def stmt_if(self, s: ast.If, scope: Scope, rest: List[ast.stmt]):
        """Returns (return value or None, whether ``rest`` was consumed)."""
        test = self.expr(s.test, scope)
        if not is_tensor(test):
            branch = s.body if self.truth(test, s.test) else s.orelse
            return self.block(branch, scope), False
        pred = self.predicate(test, s.test, scope)
        body, orelse, consumed = list(s.body), list(s.orelse), False
        tr, er = always_returns(body), always_returns(orelse)
        if tr and not er:
            orelse, consumed = orelse + rest, True
        elif er and not tr:
            body, consumed = body + rest, True
        elif tr and er:
            consumed = True
        return self.if_tensor(s, pred, scope, body, orelse), consumed

    def predicate(self, t: T, node, scope: Scope) -> T:
        if t.type.dtype != torch.bool:
            t = self.emit(scope, "ne", [t, self.const(scope, 0, t.type, node)], node)
        if not all(dims_equal(d, 1) for d in t.type.shape):
            raise self.err(node, f"the condition is a tensor of type {t.type}, not a single value, so "
                                 "its truth is ambiguous", FrontendShapeError, code="ambiguous_condition",
                           operands=[(self.src.segment(node), t.type)],
                           suggestion="reduce it with .any() or .all(), or select elementwise with "
                                      "ops.where(cond, a, b) or `a if cond else b`")
        if t.type.rank:
            t = self.emit(scope, "reshape", [t], node, shape=[])
        return t

    def if_tensor(self, s: ast.If, pred: T, scope: Scope, body, orelse) -> Optional[_Return]:
        A, B = _region(scope, "then"), _region(scope, "else")
        ra, rb = self.block(body, A), self.block(orelse, B)
        line = self.src.line_of(s)
        if (ra is None) != (rb is None):
            raise self.unsupported(s, "a return in only one branch of a data dependent if",
                                   suggestion="return in both branches, or assign a variable in both and return it")
        if ra is not None:
            va, vb = self._flat_return(ra), self._flat_return(rb)
            if len(va) != len(vb):
                raise self.err(s, f"the branches return {len(va)} and {len(vb)} values", FrontendShapeError,
                               code="branch_type_mismatch")
            names = [f"result {i}" for i in range(len(va))]
            pairs = [self._branch_pair(A, B, n, a, b, s) for n, a, b in zip(names, va, vb)]
            outs = self._emit_cond(s, pred, scope, A, B, pairs, names)
            tup = isinstance(ra.value, tuple)
            return _Return(tuple(outs) if tup else outs[0], s)
        names, pairs = [], []
        for name in assigned_names(list(body) + list(orelse)):
            va = A.vars.get(name, MISSING)
            vb = B.vars.get(name, MISSING)
            if va is MISSING:
                va = scope.lookup(name)
            if vb is MISSING:
                vb = scope.lookup(name)
            bad_a = va is MISSING or isinstance(va, Undefined)
            bad_b = vb is MISSING or isinstance(vb, Undefined)
            if bad_a or bad_b:
                if not (bad_a and bad_b):
                    scope.vars[name] = Undefined(f"{name} is assigned in only one branch of the data "
                                                 f"dependent if at line {line}, so it may be undefined here")
                continue
            if not is_tensor(va) and not is_tensor(vb) and self._same_static(va, vb):
                scope.vars[name] = va
                continue
            names.append(name)
            pairs.append(self._branch_pair(A, B, name, va, vb, s))
        if names:
            outs = self._emit_cond(s, pred, scope, A, B, pairs, names)
            for n, o in zip(names, outs):
                scope.vars[n] = o
        return None

    @staticmethod
    def _same_static(a, b) -> bool:
        if a is b:
            return True
        try:
            return type(a) == type(b) and bool(a == b)
        except Exception:
            return False

    def _flat_return(self, r: _Return) -> list:
        v = r.value
        return list(v) if isinstance(v, tuple) else [v]

    def _branch_pair(self, A: Scope, B: Scope, name, va, vb, node):
        for v in (va, vb):
            if not is_tensor(v) and not is_number(v):
                raise self.unsupported(node, f"a data dependent if that gives {name} different Python values "
                                             f"({va!r} and {vb!r})")
        ta = va if is_tensor(va) else self.const(A, va, vb.type if is_tensor(vb) else None, node)
        tb = vb if is_tensor(vb) else self.const(B, vb, ta.type, node)
        if not ta.type.same_as(tb.type):
            raise self.err(node, f"{name} has type {ta.type} after the then branch but {tb.type} after the "
                                 "else branch; both branches must agree", FrontendShapeError,
                           code="branch_type_mismatch", op="cond",
                           operands=[(f"{name} (then)", ta.type), (f"{name} (else)", tb.type)])
        return ta, tb

    def _emit_cond(self, s, pred, scope, A, B, pairs, names) -> List[T]:
        (pa, pb), caps = _finish_regions([(A, [], [p[0] for p in pairs]), (B, [], [p[1] for p in pairs])])
        pred = scope.adopt(pred)
        try:
            vals = scope.b.op_multi("cond", pred.v, *[c.v for c in caps], names=names,
                                    source=self.src.where(s), then_branch=pa, else_branch=pb)
        except ShapeError as e:
            raise self.err(s, e.reason, FrontendShapeError, op="cond") from None
        return [T(v, scope.b) for v in vals]

    # -------------------------------------------------------------- loops

    def stmt_for(self, s: ast.For, scope: Scope) -> Optional[_Return]:
        if s.orelse:
            raise self.unsupported(s, "for else")
        it = s.iter
        callee = self.expr(it.func, scope) if isinstance(it, ast.Call) else None
        if callee is range:
            args = [self.expr(a, scope) for a in it.args]
            if it.keywords:
                raise self.unsupported(it, "keyword arguments to range")
            if any(is_tensor(a) for a in args):
                raise self.unsupported(it, "range over a tensor value",
                                       suggestion="loop over a dim (range(x.shape[1])) or use a while loop "
                                                  "on tensor data")
            if any(is_dim(a) for a in args):
                start, stop, step = (0, args[0], 1) if len(args) == 1 else (
                    (args[0], args[1], 1) if len(args) == 2 else tuple(args))
                if not isinstance(step, int) or step <= 0:
                    raise self.unsupported(it, "a symbolic range with a non positive or symbolic step")
                trip = sympy.ceiling((sympy.sympify(stop) - sympy.sympify(start)) / step) if step != 1 else \
                    sympy.sympify(stop) - sympy.sympify(start)
                return self.loop_symbolic(s, scope, trip=trip, start=start, step=step)
            seq = range(*[int(a) for a in args])
        else:
            seq = self.expr(it, scope)
        if is_tensor(seq):
            raise self.unsupported(it, "iterating over a tensor",
                                   suggestion=f"loop over its first dim: for i in range({self.src.segment(it)}"
                                              f".shape[0]) and index it")
        try:
            items = list(seq)
        except TypeError:
            raise self.err(it, f"cannot iterate over {seq!r}", code="type_error") from None
        if len(items) > MAX_UNROLL:
            raise self.err(it, f"a static loop of {len(items)} iterations exceeds the unroll limit "
                               f"{MAX_UNROLL}", UnsupportedConstructError,
                           suggestion="declare the length as a symbolic dim so it becomes one loop op")
        for i, item in enumerate(items):
            self.assign(s.target, item, scope, s)
            if self.rng is not None:
                self.rng.unroll.append(i)
            try:
                r = self.block(s.body, scope)
            finally:
                if self.rng is not None:
                    self.rng.unroll.pop()
            if r is not None:
                return r
        return None

    def _carried(self, s, scope: Scope, names: List[str], exclude=()) -> Tuple[List[str], List[T]]:
        carried, init = [], []
        for n in names:
            if n in exclude:
                continue
            v = scope.lookup(n)
            if v is MISSING or isinstance(v, Undefined):
                continue
            if is_number(v):
                v = self.const(scope, v, None, s)
            elif not is_tensor(v):
                raise self.unsupported(s, f"reassigning {n} (a Python {type(v).__name__}) inside a loop "
                                          "over a symbolic length or tensor condition")
            carried.append(n)
            init.append(scope.adopt(v))
        return carried, init

    def _loop_results(self, s, body: Scope, carried: List[str], inner: List[T], init: List[T]) -> List[T]:
        outs = []
        for n, ci, t0 in zip(carried, inner, init):
            v = body.vars.get(n, ci)
            if isinstance(v, Undefined):
                raise self.err(s, v.reason, FrontendNameError)
            if not is_tensor(v):
                if not is_number(v):
                    raise self.unsupported(s, f"assigning a Python {type(v).__name__} to carried value {n}")
                v = self.const(body, v, t0.type, s)
            if not v.type.same_as(t0.type):
                raise self.err(s, f"{n} enters the loop as {t0.type} but one iteration makes it {v.type}; a "
                                  "loop carried value must keep its type", FrontendShapeError,
                               code="loop_type_change", op="loop",
                               operands=[(f"{n} (before)", t0.type), (f"{n} (after one iteration)", v.type)])
            outs.append(v)
        return outs

    def _after_loop(self, s, scope: Scope, carried, vals, names, exclude=()):
        for n, v in zip(carried, vals):
            scope.vars[n] = v
        line = self.src.line_of(s)
        for n in names:
            if n not in carried:
                scope.vars[n] = Undefined(f"{n} is assigned only inside the loop at line {line}; define it "
                                          "before the loop to keep its last value")

    def loop_symbolic(self, s: ast.For, scope: Scope, trip, start, step) -> None:
        target = s.target.id if isinstance(s.target, ast.Name) else None
        if target is None:
            raise self.unsupported(s, "unpacking in a loop over a symbolic range")
        names = assigned_names(s.body)
        if target not in names:
            names.append(target)
        carried, init = self._carried(s, scope, names, exclude=(target,))
        body = _region(scope, "body", loop=True)
        idx = T(body.b.input(target, torch.int64, ()), body.b)
        inner = [T(body.b.input(n, t.type.dtype, t.type.shape), body.b) for n, t in zip(carried, init)]
        for n, ci in zip(carried, inner):
            body.vars[n] = ci
        i_val = idx
        if step != 1:
            i_val = self.emit(body, "mul", [idx, self.const(body, step, idx.type, s)], s)
        if not (isinstance(start, int) and start == 0):
            i_val = self.emit(body, "add", [i_val, self.const(body, start, idx.type, s)], s)
        body.vars[target] = i_val
        self._block_with_stream(s.body, body, idx)  # translated even when nothing escapes (shape errors)
        if not carried:
            self._after_loop(s, scope, [], [], names)
            return None
        outs = self._loop_results(s, body, carried, inner, init)
        (pbody,), caps = _finish_regions([(body, [idx] + inner, outs)])
        vals = self._emit_loop(s, scope, init, caps, carried, body=pbody, trip_count=trip)
        self._after_loop(s, scope, carried, vals, names)
        return None

    def _block_with_stream(self, stmts, body: Scope, idx: T):
        """Translate a loop body with its iteration index as a random stream value."""
        if self.rng is not None:
            self.rng.streams.append(idx)
        try:
            return self.block(stmts, body)
        finally:
            if self.rng is not None:
                self.rng.streams.pop()

    def _emit_loop(self, s, scope, init, caps, carried, **attrs) -> List[T]:
        try:
            vals = scope.b.op_multi("loop", *[t.v for t in init], *[c.v for c in caps], names=carried,
                                    source=self.src.where(s), n_carried=len(init), **attrs)
        except ShapeError as e:
            raise self.err(s, e.reason, FrontendShapeError, op="loop") from None
        return [T(v, scope.b) for v in vals]

    def stmt_while(self, s: ast.While, scope: Scope) -> Optional[_Return]:
        if s.orelse:
            raise self.unsupported(s, "while else")
        probe = _region(scope, "probe")
        if not is_tensor(self.expr(s.test, probe)):
            n = 0
            while True:
                test = self.expr(s.test, scope)
                if is_tensor(test):
                    raise self.err(s, "the while condition was a compile time value but became tensor "
                                      "data after an iteration", StaticValueError,
                                   suggestion="make the condition tensor data from the start")
                if not self.truth(test, s.test):
                    return None
                n += 1
                if n > MAX_UNROLL:
                    raise self.err(s, f"a static while loop ran more than {MAX_UNROLL} iterations at compile "
                                      "time", UnsupportedConstructError)
                if self.rng is not None:
                    self.rng.unroll.append(n - 1)
                try:
                    r = self.block(s.body, scope)
                finally:
                    if self.rng is not None:
                        self.rng.unroll.pop()
                if r is not None:
                    return r
        names = assigned_names(s.body)
        carried, init = self._carried(s, scope, names)
        if not carried:
            raise self.err(s, "a while loop on tensor data that assigns no variable defined before it can "
                              "never change its condition", UnsupportedConstructError,
                           suggestion="define the loop variables before the loop")
        cond = _region(scope, "cond", loop=True)
        cin = [T(cond.b.input(n, t.type.dtype, t.type.shape), cond.b) for n, t in zip(carried, init)]
        for n, ci in zip(carried, cin):
            cond.vars[n] = ci
        test = self.expr(s.test, cond)
        if not is_tensor(test):
            test = self.const(cond, bool(self.truth(test, s.test)), None, s.test)
        pred = self.predicate(test, s.test, cond)
        body = _region(scope, "body", loop=True)
        idx = T(body.b.input("iteration", torch.int64, ()), body.b)
        inner = [T(body.b.input(n, t.type.dtype, t.type.shape), body.b) for n, t in zip(carried, init)]
        for n, ci in zip(carried, inner):
            body.vars[n] = ci
        self._block_with_stream(s.body, body, idx)
        outs = self._loop_results(s, body, carried, inner, init)
        (pcond, pbody), caps = _finish_regions([(cond, cin, [pred]), (body, [idx] + inner, outs)])
        vals = self._emit_loop(s, scope, init, caps, carried, body=pbody, cond=pcond)
        self._after_loop(s, scope, carried, vals, names)
        return None

    # -------------------------------------------------------------- static helpers

    def truth(self, v, node) -> bool:
        if is_tensor(v):
            raise IRError("internal: truth of a tensor")
        if isinstance(v, sympy.logic.boolalg.Boolean) and v not in (sympy.true, sympy.false):
            raise self.err(node, f"{self.src.segment(node)} depends on a symbolic dim", StaticValueError)
        return bool(v)

    def static_compare(self, op, a, b, node):
        if is_dim(a) or is_dim(b):
            try:
                r = _SYMCMP[type(op)](sympy.sympify(a), sympy.sympify(b))
            except (TypeError, sympy.SympifyError):
                r = None
            if r is sympy.true or r is True:
                return True
            if r is sympy.false or r is False:
                return False
            syms = sorted({str(x) for x in (sympy.sympify(a).free_symbols | sympy.sympify(b).free_symbols)})
            raise self.err(node, f"`{self.src.segment(node)}` depends on the symbolic dim "
                                 f"{', '.join(syms)}, so it is not known at compile time. Truestar does not "
                                 "specialize on shapes (that would mean one compile per shape)",
                           StaticValueError, suggestion="give the dim a fixed size in the signature, or branch "
                                                        "on tensor data instead")
        try:
            return _PYCMP[type(op)](a, b)
        except Exception as e:
            raise self.err(node, f"cannot compare {a!r} and {b!r}: {e}", code="type_error") from None

    # -------------------------------------------------------------- expressions

    def name(self, ident: str, scope: Scope, node):
        v = scope.lookup(ident)
        if v is MISSING:
            v = self.src.resolve_global(ident)
            if v is MISSING:
                raise self.err(node, f"name {ident!r} is not defined", FrontendNameError)
        if isinstance(v, Undefined):
            raise self.err(node, v.reason, FrontendNameError)
        return v

    def expr(self, e: ast.expr, scope: Scope):
        if type(e) in _UNSUPPORTED_EXPR:
            raise self.unsupported(e, _UNSUPPORTED_EXPR[type(e)])
        if isinstance(e, ast.Constant):
            return e.value
        if isinstance(e, ast.Name):
            return self.name(e.id, scope, e)
        if isinstance(e, ast.Tuple):
            return tuple(self.expr(x, scope) for x in e.elts)
        if isinstance(e, ast.List):
            return [self.expr(x, scope) for x in e.elts]
        if isinstance(e, ast.Dict):  # a structure of values, resolved at compile time (pytrees)
            out = {}
            for k, v in zip(e.keys, e.values):
                if k is None:
                    raise self.unsupported(e, "** unpacking in a dict literal")
                key = self.expr(k, scope)
                if is_tensor(key):
                    raise self.unsupported(k, "a tensor as a dict key")
                out[key] = self.expr(v, scope)
            return out
        if isinstance(e, ast.Attribute):
            return self.attribute(e, scope)
        if isinstance(e, ast.Subscript):
            return self.subscript(e, scope)
        if isinstance(e, ast.Call):
            return self.call(e, scope)
        if isinstance(e, ast.BinOp):
            a, b = self.expr(e.left, scope), self.expr(e.right, scope)
            return self.binop(e.op, a, b, e, self.src.segment(e.left), self.src.segment(e.right), scope)
        if isinstance(e, ast.UnaryOp):
            return self.unary(e, scope)
        if isinstance(e, ast.Compare):
            return self.compare(e, scope)
        if isinstance(e, ast.BoolOp):
            return self.boolop(e, scope)
        if isinstance(e, ast.IfExp):
            c = self.expr(e.test, scope)
            if not is_tensor(c):
                return self.expr(e.body if self.truth(c, e.test) else e.orelse, scope)
            a, b = self.expr(e.body, scope), self.expr(e.orelse, scope)
            if c.type.dtype != torch.bool:
                c = self.emit(scope, "ne", [c, self.const(scope, 0, c.type, e)], e)
            ta, tb = self.tensor_pair(scope, a, b, e)
            return self.emit(scope, "where", [c, ta, tb], e,
                             [self.src.segment(e.test), self.src.segment(e.body), self.src.segment(e.orelse)])
        if isinstance(e, ast.Slice):
            return slice(*(None if x is None else self.expr(x, scope) for x in (e.lower, e.upper, e.step)))
        raise self.unsupported(e, f"{type(e).__name__} expressions")

    def binop(self, op, a, b, node, la, lb, scope):
        if not is_tensor(a) and not is_tensor(b):
            fn = _PYBIN.get(type(op))
            if fn is None:
                raise self.unsupported(node, f"operator {type(op).__name__}")
            try:
                return fn(a, b)
            except Exception as ex:
                raise self.err(node, f"cannot compute {self.src.segment(node)} at compile time: {ex}",
                               code="type_error") from None
        if isinstance(op, ast.Pow):
            return self.power(a, b, node, scope)
        name = _BIN.get(type(op))
        if name is None:
            raise self.unsupported(node, f"the {type(op).__name__} operator on tensors")
        if name == "matmul":
            return self.matmul(scope, a, b, node, la, lb)
        ta, tb = self.tensor_pair(scope, a, b, node)
        return self.emit(scope, name, [ta, tb], node, [la, lb])

    def matmul(self, scope, a, b, node, la, lb):
        if not (is_tensor(a) and is_tensor(b)):
            raise self.err(node, "@ needs two tensors", code="type_error")
        for t, l in ((a, la), (b, lb)):
            if t.type.rank < 2:
                raise self.err(node, f"{l} has rank {t.type.rank}; matmul needs rank 2 or more",
                               FrontendShapeError, op="matmul", operands=[(la, a.type), (lb, b.type)],
                               suggestion=f"reshape it, for example {l}.reshape(1, -1)")
        if a.type.dtype != b.type.dtype:
            raise self.err(node, f"matmul operands have different dtypes ({a.type} and {b.type})",
                           FrontendShapeError, code="dtype_mismatch", op="matmul",
                           operands=[(la, a.type), (lb, b.type)])
        return self.emit(scope, "matmul", [a, b], node, [la, lb])

    def power(self, a, b, node, scope):
        if is_tensor(b):
            raise self.unsupported(node, "a tensor exponent")
        if b == 0.5:
            return self.emit(scope, "sqrt", [a], node)
        if b == -0.5:
            return self.emit(scope, "rsqrt", [a], node)
        if isinstance(b, int) and 1 <= abs(b) <= 16:
            out = a
            for _ in range(abs(b) - 1):
                out = self.emit(scope, "mul", [out, a], node)
            if b < 0:
                out = self.emit(scope, "div", [self.const(scope, 1.0, a.type, node), out], node)
            return out
        raise self.unsupported(node, f"the exponent {b!r} (supported: integers up to 16, 0.5, -0.5)")

    def unary(self, e: ast.UnaryOp, scope):
        v = self.expr(e.operand, scope)
        if not is_tensor(v):
            fn = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_,
                  ast.Invert: operator.invert}[type(e.op)]
            if isinstance(e.op, ast.Not):
                return not self.truth(v, e.operand)
            return fn(v)
        if isinstance(e.op, ast.USub):
            return self.emit(scope, "neg", [v], e)
        if isinstance(e.op, ast.UAdd):
            return v
        return self.emit(scope, "logical_not", [self.as_bool(v, e, scope)], e)

    def as_bool(self, t: T, node, scope) -> T:
        if t.type.dtype == torch.bool:
            return t
        return self.emit(scope, "ne", [t, self.const(scope, 0, t.type, node)], node)

    def compare(self, e: ast.Compare, scope):
        left = self.expr(e.left, scope)
        lnode = e.left
        result = None
        for op, rnode in zip(e.ops, e.comparators):
            right = self.expr(rnode, scope)
            if isinstance(op, (ast.Is, ast.IsNot, ast.In, ast.NotIn)):
                if is_tensor(left) or is_tensor(right):
                    if isinstance(op, (ast.Is, ast.IsNot)) and (left is None or right is None):
                        r = isinstance(op, ast.IsNot)
                    else:
                        raise self.unsupported(e, f"`{type(op).__name__}` on tensors")
                else:
                    r = {ast.Is: operator.is_, ast.IsNot: operator.is_not,
                         ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}[type(op)](left, right)
            elif is_tensor(left) or is_tensor(right):
                ta, tb = self.tensor_pair(scope, left, right, e)
                r = self.emit(scope, _CMP[type(op)], [ta, tb], e, [self.src.segment(lnode), self.src.segment(rnode)])
            else:
                r = self.static_compare(op, left, right, e)
            result = r if result is None else self._and(result, r, e, scope)
            left, lnode = right, rnode
        return result

    def _and(self, a, b, node, scope):
        if not is_tensor(a) and not is_tensor(b):
            return bool(a) and bool(b)
        if not is_tensor(a):
            return b if self.truth(a, node) else False
        if not is_tensor(b):
            return a if self.truth(b, node) else False
        return self.emit(scope, "logical_and", [self.as_bool(a, node, scope), self.as_bool(b, node, scope)], node)

    def boolop(self, e: ast.BoolOp, scope):
        is_and = isinstance(e.op, ast.And)
        cur = self.expr(e.values[0], scope)
        for vnode in e.values[1:]:
            if not is_tensor(cur):
                t = self.truth(cur, vnode)
                if is_and and not t or (not is_and and t):
                    return cur
                cur = self.expr(vnode, scope)
                continue
            nxt = self.expr(vnode, scope)
            if not is_tensor(nxt):
                t = self.truth(nxt, vnode)
                if is_and:
                    cur = self.as_bool(cur, e, scope) if t else False
                else:
                    cur = True if t else self.as_bool(cur, e, scope)
                continue
            cur = self.emit(scope, "logical_and" if is_and else "logical_or",
                            [self.as_bool(cur, e, scope), self.as_bool(nxt, e, scope)], e)
        return cur

    # -------------------------------------------------------------- attributes, indexing

    def attribute(self, e: ast.Attribute, scope):
        base = self.expr(e.value, scope)
        if is_tensor(base):
            a = e.attr
            if a == "shape":
                return tuple(base.type.shape)
            if a == "ndim":
                return base.type.rank
            if a == "dtype":
                return base.type.dtype
            if a == "T":
                return self.emit(scope, "permute", [base], e, dims=list(range(base.type.rank))[::-1])
            if a == "mT":
                if base.type.rank < 2:
                    raise self.err(e, ".mT needs rank 2 or more", FrontendShapeError)
                dims = list(range(base.type.rank))
                dims[-1], dims[-2] = dims[-2], dims[-1]
                return self.emit(scope, "permute", [base], e, dims=dims)
            if a in _TENSOR_METHODS:
                return BoundMethod(base, a)
            raise self.unsupported(e, f"the tensor attribute .{a}")
        if isinstance(base, BoundMethod):
            raise self.unsupported(e, "attributes of a method")
        try:
            return getattr(base, e.attr)
        except AttributeError:
            raise self.err(e, f"{self.src.segment(e.value)} has no attribute {e.attr!r}",
                           FrontendNameError) from None

    def subscript(self, e: ast.Subscript, scope):
        base = self.expr(e.value, scope)
        idx = self.expr(e.slice, scope)
        if not is_tensor(base):
            if is_tensor(idx):
                raise self.unsupported(e, "indexing a Python value with a tensor")
            if is_dim(idx):
                raise self.err(e, "a symbolic dim cannot index a Python sequence", StaticValueError)
            try:
                return base[idx]
            except Exception as ex:
                raise self.err(e, f"cannot index {self.src.segment(e.value)}: {ex}", code="type_error") from None
        return self.index_tensor(base, idx if isinstance(idx, tuple) else (idx,), e, scope)

    def index_tensor(self, x: T, items, node, scope) -> T:
        rank = x.type.rank
        n_real = sum(1 for i in items if i is not None and i is not Ellipsis)
        if n_real > rank:
            raise self.err(node, f"too many indices ({n_real}) for a tensor of type {x.type}",
                           FrontendShapeError, op="index", operands=[(self.src.segment(node.value), x.type)])
        d = 0
        for k, it in enumerate(items):
            if it is Ellipsis:
                d = x.type.rank - sum(1 for j in items[k + 1:] if j is not None and j is not Ellipsis)
                continue
            if it is None:
                shape = list(x.type.shape)
                shape.insert(d, 1)
                x = self.emit(scope, "reshape", [x], node, shape=shape)
                d += 1
                continue
            if isinstance(it, slice):
                if it.start is None and it.stop is None and it.step is None:
                    d += 1
                    continue
                for part in (it.start, it.stop, it.step):
                    if is_tensor(part):
                        raise self.unsupported(node, "slicing with tensor bounds")
                x = self.emit(scope, "slice", [x], node, dim=d, start=it.start, end=it.stop,
                              step=1 if it.step is None else it.step)
                d += 1
                continue
            if isinstance(it, bool):
                raise self.unsupported(node, "boolean indexing")
            if isinstance(it, int):
                x = self.emit(scope, "select", [x], node, [self.src.segment(node.value)], dim=d, index=it)
                continue
            if is_dim(it) or is_tensor(it):
                i = it if is_tensor(it) else self.const(scope, it, None, node)
                if i.type.rank != 0 or i.type.dtype not in (torch.int64, torch.int32):
                    raise self.err(node, f"a tensor index must be an integer scalar, got {i.type}",
                                   FrontendShapeError, op="index",
                                   suggestion="use ops.where for masking; gather is not supported yet")
                x = self.emit(scope, "select_dynamic", [x, i], node, [self.src.segment(node.value), "index"], dim=d)
                continue
            raise self.unsupported(node, f"indexing with {type(it).__name__}")
        return x

    # -------------------------------------------------------------- calls

    def call(self, e: ast.Call, scope):
        func = self.expr(e.func, scope)
        args, labels = [], []
        for a in e.args:
            if isinstance(a, ast.Starred):
                raise self.unsupported(a, "starred arguments")
            args.append(self.expr(a, scope))
            labels.append(self.src.segment(a))
        kwargs = {}
        for k in e.keywords:
            if k.arg is None:
                raise self.unsupported(e, "**kwargs in calls")
            kwargs[k.arg] = self.expr(k.value, scope)
        if isinstance(func, BoundMethod):
            return self.method(func, args, kwargs, e, labels, scope)
        opname = getattr(func, "__truestar_op__", None)
        if opname is None:
            try:
                opname = TORCH_ALIASES.get(func)
                keyed = _TORCH_RANDOM.get(func)
            except TypeError:
                opname = keyed = None
            if keyed is not None:
                raise self.unsupported(e, f"{getattr(func, '__name__', func)} (PyTorch's global generator gives other "
                                          "bits on other devices)",
                                       suggestion=f"use ops.{keyed}: keyed, the same bits on every backend "
                                                  "(docs/RANDOM.md)")
            if opname == "attention":
                if kwargs.pop("dropout_p", 0.0) not in (0, 0.0):
                    raise self.unsupported(e, "attention dropout")
                if "attn_mask" in kwargs:
                    kwargs["mask"] = kwargs.pop("attn_mask")
        if opname is not None:
            return self.lower(opname, args, kwargs, e, labels, scope)
        from .function import NativeFunction
        if isinstance(func, NativeFunction):
            return self.inline(func, args, kwargs, e, scope)
        if getattr(func, "__truestar_structure__", False):
            # a compile time function over structures (dicts, lists) of values that only rearranges
            # them and emits no op (``truestar.lora.merge_trees``); run as Python on the traced values
            try:
                return func(*args, **kwargs)
            except Exception as ex:
                raise self.err(e, f"cannot compute {self.src.segment(e)} at compile time: {ex}",
                               code="type_error") from None
        hook = None if isinstance(func, type) else getattr(func, "__truestar_inline__", None)
        if hook is not None:  # a layer or another object whose call is a native function (docs/NN.md)
            fn, lead = hook()
            return self.inline(fn, list(lead) + args, kwargs, e, scope)
        if func in (len,) and args and is_tensor(args[0]):
            return args[0].type.shape[0] if args[0].type.rank else self._raise_rank0(e)
        if func is abs and args and is_tensor(args[0]):
            return self.lower("abs", args, kwargs, e, labels, scope)
        if _is_structure_class(func):  # a dataclass or NamedTuple of tensors: a compile time structure
            try:
                return _build_structure(func, args, kwargs)
            except Exception as ex:
                raise self.err(e, f"cannot build {func.__name__}: {ex}", code="type_error") from None
        owner = getattr(func, "__self__", None)
        if isinstance(owner, (dict, list, tuple)) and getattr(func, "__name__", "") in _CONTAINER_METHODS:
            try:
                out = func(*args, **kwargs)
            except Exception as ex:
                raise self.err(e, f"cannot compute {self.src.segment(e)} at compile time: {ex}",
                               code="type_error") from None
            return list(out) if func.__name__ in ("items", "keys", "values") else out
        static_ok = func in _STATIC_BUILTINS or getattr(func, "__module__", None) == "math" or \
            isinstance(func, type) and func in (TensorSpec,)
        if static_ok:
            if any(is_tensor(a) for a in list(args) + list(kwargs.values())):
                raise self.unsupported(e, f"calling {getattr(func, '__name__', func)} on a tensor")
            try:
                return func(*args, **kwargs)
            except Exception as ex:
                raise self.err(e, f"cannot compute {self.src.segment(e)} at compile time: {ex}",
                               code="type_error") from None
        name = getattr(func, "__qualname__", None) or getattr(func, "__name__", None) or repr(func)
        raise self.unsupported(e, f"calls to {name}",
                               suggestion="use the truestar ops namespace (from truestar import ops) or another "
                                          "@truestar.function")

    def _raise_rank0(self, e):
        raise self.err(e, "len() of a rank 0 tensor", FrontendShapeError)

    def inline(self, fn, args, kwargs, e, scope):
        if self.depth >= MAX_INLINE_DEPTH:
            raise self.unsupported(e, f"call nesting deeper than {MAX_INLINE_DEPTH} (recursion)")
        try:
            bound = fn.py_signature.bind(*args, **kwargs)
        except TypeError as ex:
            raise self.err(e, f"bad arguments for {fn.name}: {ex}", code="type_error") from None
        bound.apply_defaults()
        sub = Translator(fn.src, self.depth + 1, self.rng)
        root = Scope(scope.b, None, scope.loop_depth)
        for pname, v in bound.arguments.items():
            spec = fn.specs.get(pname)
            if is_tensor(v):
                v = scope.adopt(v)
                if spec is not None:
                    sub.check_spec(v, spec, e, f"argument {pname} of {fn.name}")
            elif spec is not None:
                raise self.err(e, f"argument {pname} of {fn.name} must be a tensor {spec}, got {v!r}",
                               code="type_error")
            root.vars[pname] = v
        if getattr(fn, "_custom_grad", None) is not None:
            from .custom_grad import emit_call
            return emit_call(self, fn, root.vars, e, scope)
        if self.rng is not None:   # random sites inside the callee live under <callee>#<k>
            self.rng.path.append(f"{fn.name}#{self.rng.ordinal(e, fn.name)}")
        try:
            r = sub.block(sub.src.tree.body, root)
        finally:
            if self.rng is not None:
                self.rng.path.pop()
        if r is None:
            raise sub.err(sub.src.tree, f"{fn.name} returns no value", code="no_return")
        return r.value

    def method(self, m: BoundMethod, args, kwargs, e, labels, scope):
        x, name = m.obj, m.name
        lbl = [self.src.segment(e.func.value)] + labels
        if name in ("reshape", "view"):
            shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)
            return self.lower("reshape", [x, shape], {}, e, lbl, scope)
        if name == "permute":
            dims = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)
            return self.lower("permute", [x, dims], {}, e, lbl, scope)
        if name in ("size",):
            return tuple(x.type.shape) if not args else x.type.shape[args[0]]
        if name == "dim":
            return x.type.rank
        if name in ("float", "double", "long"):
            return self.lower("cast", [x, {"float": "f32", "double": "f64", "long": "i64"}[name]], {}, e, lbl, scope)
        if name == "unsqueeze":
            d = args[0] % (x.type.rank + 1)
            shape = list(x.type.shape)
            shape.insert(d, 1)
            return self.emit(scope, "reshape", [x], e, shape=shape)
        if name == "squeeze":
            d = args[0] % x.type.rank
            if not dims_equal(x.type.shape[d], 1):
                raise self.err(e, f"cannot squeeze dim {d} of size {x.type.shape[d]}", FrontendShapeError,
                               op="reshape", operands=[(lbl[0], x.type)])
            shape = list(x.type.shape)
            del shape[d]
            return self.emit(scope, "reshape", [x], e, shape=shape)
        return self.lower(_TENSOR_METHODS[name], [x] + list(args), kwargs, e, lbl, scope)

    def lower(self, name, args, kwargs, e, labels, scope):
        fn = getattr(user_ops, name)
        try:
            bound = inspect.signature(fn).bind(*args, **kwargs)
        except TypeError as ex:
            raise self.err(e, f"bad arguments for ops.{name}: {ex}", code="type_error") from None
        bound.apply_defaults()
        a = bound.arguments
        L = list(labels) + [""] * 8

        def need_t(key, i=0, like=None):
            v = a[key]
            if is_tensor(v):
                return v
            if like is not None and is_number(v):
                return self.const(scope, v, like, e)
            raise self.err(e, f"ops.{name} argument {key} must be a tensor, got {v!r}", code="type_error")

        def static(key):
            v = a[key]
            if is_tensor(v):
                raise self.err(e, f"ops.{name} argument {key} must be a compile time value, got a tensor",
                               StaticValueError)
            return v

        if name in _UNARY_OPS:
            return self.emit(scope, name, [need_t("x")], e, L[:1])
        if name == "gelu":
            return self.emit(scope, "gelu", [need_t("x")], e, L[:1], approximate=static("approximate"))
        if name == "abs":
            x = need_t("x")
            neg = self.emit(scope, "neg", [x], e)
            lt = self.emit(scope, "lt", [x, self.const(scope, 0, x.type, e)], e)
            return self.emit(scope, "where", [lt, neg, x], e)
        if name == "clamp":
            return self.emit(scope, "clamp", [need_t("x")], e, L[:1], min=static("min"), max=static("max"))
        if name in ("maximum", "minimum"):
            x, y = self.tensor_pair(scope, a["a"], a["b"], e)
            c = self.emit(scope, "gt" if name == "maximum" else "lt", [x, y], e, L[:2])
            return self.emit(scope, "where", [c, x, y], e, ["condition"] + L[:2])
        if name == "where":
            c = need_t("condition")
            c = self.as_bool(c, e, scope)
            x, y = self.tensor_pair(scope, a["a"], a["b"], e)
            return self.emit(scope, "where", [c, x, y], e, L[:3])
        if name in ("sum", "mean", "any", "all"):
            x = need_t("x")
            dim = static("dim")
            dims = [] if dim is None else ([dim] if isinstance(dim, int) else list(dim))
            keep = static("keepdim")
            if name == "all":
                nb = self.emit(scope, "logical_not", [self.as_bool(x, e, scope)], e)
                r = self.emit(scope, "any", [nb], e, L[:1], dims=dims, keepdim=keep)
                return self.emit(scope, "logical_not", [r], e)
            if name == "any":
                x = self.as_bool(x, e, scope)
            return self.emit(scope, name, [x], e, L[:1], dims=dims, keepdim=keep)
        if name == "softmax":
            x = need_t("x")
            if a["temperature"] is not None:
                x = self.emit(scope, "div", [x, need_t("temperature", like=x.type)], e, [L[0], "temperature"])
            return self.emit(scope, "softmax", [x], e, L[:1], dim=static("dim"))
        if name == "matmul":
            return self.matmul(scope, need_t("a"), need_t("b"), e, L[0], L[1])
        if name == "linear":
            b = a["bias"]
            return self.emit(scope, "linear", [need_t("x"), need_t("weight"), None if b is None else need_t("bias")],
                             e, L[:3])
        if name == "layer_norm":
            ns = static("normalized_shape")
            ns = [ns] if isinstance(ns, int) or is_dim(ns) else list(ns)
            w, b = a["weight"], a["bias"]
            return self.emit(scope, "layer_norm", [need_t("x"), None if w is None else need_t("weight"),
                                                   None if b is None else need_t("bias")], e, L[:3],
                             normalized_shape=ns, eps=float(static("eps")))
        if name == "attention":
            m = a["mask"]
            q, k, v = need_t("q"), need_t("k"), need_t("v")
            m = None if m is None else need_t("mask")
            is_causal = bool(static("is_causal"))
            if static("enable_gqa"):
                try:
                    n = gqa_factor(q.type, k.type)
                except UnsupportedOpError as ex:
                    raise self.err(e, str(ex), FrontendShapeError, op="attention",
                                   operands=[(L[0], q.type), (L[1], k.type)]) from None
                k = repeat_heads(self._emitter(scope, e), k, n, k.type.rank - 3)
                v = repeat_heads(self._emitter(scope, e), v, n, v.type.rank - 3)
            window = static("window")
            if window is not None:
                if not is_causal or not isinstance(window, int) or window < 1:
                    raise self.err(e, "window must be a positive int and needs is_causal=True", code="type_error")
                m = self.window_mask(scope, q, k, m, window, e)
                is_causal = False
            return self.emit(scope, "attention", [q, k, v, m], e, L[:4], scale=static("scale"), is_causal=is_causal)
        if name == "reshape":
            shape = static("shape")
            shape = [shape] if isinstance(shape, int) or is_dim(shape) else list(shape)
            return self.emit(scope, "reshape", [need_t("x")], e, L[:1], shape=shape)
        if name == "permute":
            return self.emit(scope, "permute", [need_t("x")], e, L[:1], dims=list(static("dims")))
        if name == "transpose":
            x = need_t("x")
            dims = list(range(x.type.rank))
            if x.type.rank == 0:
                raise self.err(e, "transpose of a rank 0 tensor", FrontendShapeError)
            d0, d1 = static("dim0") % x.type.rank, static("dim1") % x.type.rank
            dims[d0], dims[d1] = dims[d1], dims[d0]
            return self.emit(scope, "permute", [x], e, L[:1], dims=dims)
        if name == "cat":
            ts = static("tensors") if not is_tensor(a["tensors"]) else None
            if ts is None or not all(is_tensor(t) for t in ts):
                raise self.err(e, "ops.cat needs a list or tuple of tensors", code="type_error")
            return self.emit(scope, "cat", list(ts), e, [f"tensor {i}" for i in range(len(ts))], dim=static("dim"))
        if name in ("full", "zeros", "ones"):
            shape = static("shape")
            shape = [shape] if isinstance(shape, int) or is_dim(shape) else list(shape)
            dt = static("dtype")
            dt = DTYPES[dt] if isinstance(dt, str) else dt
            value = {"zeros": 0, "ones": 1}.get(name)
            if value is None:
                value = static("value")
            if not dt.is_floating_point and isinstance(value, float):
                raise self.err(e, f"value {value} does not fit dtype {dt}", code="type_error")
            return self.emit(scope, "full", [], e, shape=shape, value=value, dtype=dt)
        if name in ("zeros_like", "ones_like", "full_like"):
            x = need_t("x")
            value = {"zeros_like": 0, "ones_like": 1}.get(name)
            if value is None:
                value = static("value")
            return self.emit(scope, "full_like", [x], e, L[:1], value=value, dtype=x.type.dtype)
        if name in _RANDOM_OPS:
            return self.lower_random(name, a, need_t, static, e, L, scope)
        if name == "batch_norm":
            return self.lower_batch_norm(a, need_t, static, e, L, scope)
        if name in _LAYER_OPS:
            return self.lower_layer_op(name, a, need_t, static, e, L, scope)
        out = self.lower_llm(name, a, need_t, static, e, L, scope)
        if out is not None:
            return out
        raise self.unsupported(e, f"ops.{name}")

    # -------------------------------------------------------------- random and training ops (docs/RANDOM.md)

    def lower_random(self, name, a, need_t, static, e, L, scope):
        from .. import rng
        from ..ir.train_ops import dropout as dropout_ops

        if self.rng is None:
            raise self.err(e, f"ops.{name} needs a compiled function context", code="type_error")
        if name == "dropout":
            p, training = static("p"), static("training")
            x = need_t("x")
            if not training or p == 0:
                return x
        key_arg = a["key"]
        if key_arg is None:
            key = self.rng.program_key()
        elif is_tensor(key_arg):
            key = key_arg
        elif isinstance(key_arg, int) and not isinstance(key_arg, bool):
            try:
                key = self.const(scope, rng.as_i64(key_arg), None, e)
            except (TypeError, ValueError) as ex:
                raise self.err(e, f"ops.{name}: {ex}", code="type_error") from None
        else:
            raise self.err(e, f"ops.{name} key= must be an int or an int64 scalar tensor, got {key_arg!r}",
                           code="type_error")
        site = self.rng.site(self, e, name, static("name"))
        streams = list(self.rng.streams)

        def dtype_of(dt):
            return DTYPES[dt] if isinstance(dt, str) else dt

        def draw(dist, shape, dt, **kw):
            shape = [shape] if isinstance(shape, int) or is_dim(shape) else list(shape)
            return self.emit(scope, "random", [key] + streams, e, ["key"] + [f"stream {i}" for i in range(len(streams))],
                             dist=dist, shape=shape, dtype=dt, site=rng.site_id(site), site_name=site, **kw)

        if name == "dropout":
            p = float(p)
            if not 0.0 <= p < 1.0:
                raise self.err(e, f"dropout probability must be in [0, 1), got {p}", code="type_error")
            y, _ = dropout_ops(self._emitter(scope, e), lambda v, like: self.const(scope, v, like.type, e),
                               scope.adopt(x), p, key, streams, site)
            return y
        if name in ("rand_like", "randn_like"):
            x = need_t("x")
            return draw("uniform" if name == "rand_like" else "normal", x.type.shape, x.type.dtype)
        if name == "rand":
            return draw("uniform", static("shape"), dtype_of(static("dtype")))
        if name in ("randn", "normal"):
            z = draw("normal", static("shape"), dtype_of(static("dtype")))
            if name == "normal":
                std, mean = static("std"), static("mean")
                if std != 1:
                    z = self.emit(scope, "mul", [z, self.const(scope, std, z.type, e)], e)
                if mean != 0:
                    z = self.emit(scope, "add", [z, self.const(scope, mean, z.type, e)], e)
            return z
        if name == "bernoulli":
            return draw("bernoulli", static("shape"), dtype_of(static("dtype")), p=float(static("p")))
        return draw("randint", static("shape"), dtype_of(static("dtype")), low=static("low"), high=static("high"))

    def lower_batch_norm(self, a, need_t, static, e, L, scope):
        """Eval: the IR batch_norm op. Training: primitive ops with the running statistics as state,
        returning (y, new_running_mean, new_running_var) (ir/train_ops.py)."""
        from ..ir.train_ops import batch_norm_train

        x = need_t("x")
        opt = {k: (None if a[k] is None else need_t(k)) for k in ("weight", "bias", "running_mean", "running_var")}
        if not static("training"):
            return self.emit(scope, "batch_norm", [x] + [opt[k] for k in ("weight", "bias", "running_mean",
                                                                            "running_var")],
                             e, L[:5], eps=float(static("eps")))
        ins = {k: (None if v is None else scope.adopt(v).v) for k, v in opt.items()}
        try:
            y, nm, nv = batch_norm_train(
                lambda op, vs, **kw: self.emit(scope, op, [None if v is None else T(v, scope.b) for v in vs], e,
                                               **kw).v,
                lambda v, like: self.const(scope, v, like.type, e).v,
                scope.adopt(x).v, ins["weight"], ins["bias"], ins["running_mean"], ins["running_var"],
                static("momentum"), float(static("eps")))
        except UnsupportedOpError as ex:
            raise self.err(e, str(ex), UnsupportedConstructError, op="batch_norm") from None
        if nm is None:
            return T(y, scope.b)
        return (T(y, scope.b), T(nm, scope.b), T(nv, scope.b))

    # -------------------------------------------------------------- layer library ops (docs/NN.md)

    def lower_layer_op(self, name, a, need_t, static, e, L, scope):
        def ints2(v, what):
            v = [v, v] if isinstance(v, int) else list(v)
            if len(v) != 2 or not all(isinstance(i, int) and not isinstance(i, bool) for i in v):
                raise self.err(e, f"ops.{name} {what} must be an int or a pair of ints, got {v!r}", code="type_error")
            return v

        if name == "expm1":
            return self.expm1(scope, need_t("x"), e)
        if name == "log_softmax":
            from ..train import ops as _train_ops  # noqa: F401  registers the log_softmax op
            x = need_t("x")
            d = static("dim")
            if not isinstance(d, int) or not -max(x.type.rank, 1) <= d < max(x.type.rank, 1):
                raise self.err(e, f"dim {d!r} is out of range for {x.type}", FrontendShapeError, op=name,
                               operands=[(L[0], x.type)])
            return self.emit(scope, "log_softmax", [x], e, L[:1], dim=d % max(x.type.rank, 1))
        if name == "conv2d":
            b = None if a["bias"] is None else need_t("bias")
            return self.emit(scope, "conv2d", [need_t("x"), need_t("weight"), b], e, L[:3],
                             stride=ints2(static("stride"), "stride"), padding=ints2(static("padding"), "padding"),
                             dilation=ints2(static("dilation"), "dilation"), groups=int(static("groups")))
        if name == "conv_transpose2d":
            x, w = need_t("x"), need_t("weight")
            s, p = ints2(static("stride"), "stride"), ints2(static("padding"), "padding")
            op_, d = ints2(static("output_padding"), "output_padding"), ints2(static("dilation"), "dilation")
            hw, k = x.type.shape[2:], w.type.shape[2:]
            if x.type.rank != 4 or w.type.rank != 4 or not all(isinstance(v, int) for v in tuple(hw) + tuple(k)):
                raise self.err(e, f"ops.conv_transpose2d needs [N, C, H, W] with fixed H, W and kernel, got {x.type}",
                               FrontendShapeError, op=name, operands=[(L[0], x.type), (L[1], w.type)])
            if any(o >= st for o, st in zip(op_, s)):
                raise self.err(e, f"ops.conv_transpose2d output_padding {op_} must be smaller than the stride {s}",
                               code="type_error")
            size = [(hw[i] - 1) * s[i] - 2 * p[i] + d[i] * (k[i] - 1) + op_[i] + 1 for i in range(2)]
            y = self.emit(scope, "conv2d_input_grad", [x, w], e, L[:2], stride=s, padding=p, dilation=d,
                          groups=int(static("groups")), input_size=size)
            if a["bias"] is None:
                return y
            bias = self.emit(scope, "reshape", [need_t("bias")], e, L[2:3], shape=[w.type.shape[1] * int(static("groups")), 1, 1])
            return self.emit(scope, "add", [y, bias], e, [])
        if name == "group_norm":
            w = None if a["weight"] is None else need_t("weight")
            b = None if a["bias"] is None else need_t("bias")
            g = static("num_groups")
            if not isinstance(g, int) or isinstance(g, bool) or g < 1:
                raise self.err(e, f"ops.group_norm num_groups must be a positive int, got {g!r}", code="type_error")
            return self.emit(scope, "group_norm", [need_t("x"), w, b], e, L[:1], num_groups=g, eps=float(static("eps")))
        if name == "max_pool2d":
            k = ints2(static("kernel_size"), "kernel_size")
            s = static("stride")
            return self.emit(scope, "max_pool2d", [need_t("x")], e, L[:1], kernel_size=k,
                             stride=k if s is None else ints2(s, "stride"), padding=ints2(static("padding"), "padding"),
                             dilation=[1, 1], ceil_mode=False)
        if name == "adaptive_avg_pool2d":
            return self.emit(scope, "adaptive_avg_pool2d", [need_t("x")], e, L[:1],
                             output_size=ints2(static("output_size"), "output_size"))
        # avg_pool2d: non overlapping windows are exactly an adaptive pool to H / k by W / k
        x = need_t("x")
        k = static("kernel_size")
        if not isinstance(k, int) or k < 1:
            raise self.err(e, f"ops.avg_pool2d kernel_size must be a positive int, got {k!r}", code="type_error")
        hw = x.type.shape[-2:] if x.type.rank == 4 else ()
        if len(hw) != 2 or not all(isinstance(d, int) and d % k == 0 for d in hw):
            raise self.err(e, f"ops.avg_pool2d needs [N, C, H, W] with fixed H and W that are multiples of {k}, "
                              f"got {x.type}", FrontendShapeError, op="avg_pool2d", operands=[(L[0], x.type)],
                           suggestion="pad or crop the input, or use ops.adaptive_avg_pool2d")
        return self.emit(scope, "adaptive_avg_pool2d", [x], e, L[:1], output_size=[hw[0] // k, hw[1] // k])

    def expm1(self, scope, x, e):
        """exp(x) - 1 without cancellation. On [-1/2, 1/2] the Taylor series in nested form
        x (1 + x/2 (1 + x/3 (... (1 + x/17)))); its truncation error there is below 2^-60 relative.
        Outside, exp(x) - 1 is used, where exp(x) <= 0.61 or >= 1.65 and the subtraction loses at most
        about one bit. The series reads x clamped to the interval, so neither branch can overflow and
        the unselected one contributes a finite zero to a gradient."""
        if not x.type.dtype.is_floating_point:
            raise self.err(e, f"ops.expm1 needs a float tensor, got {x.type}", code="type_error")
        c = self.emit(scope, "clamp", [x], e, min=-0.5, max=0.5)
        acc = None
        for n in range(17, 1, -1):
            term = self.emit(scope, "mul", [c, self.const(scope, 1.0 / n, c.type, e)], e)
            if acc is not None:
                term = self.emit(scope, "mul", [term, acc], e)
            acc = self.emit(scope, "add", [term, self.const(scope, 1.0, c.type, e)], e)
        series = self.emit(scope, "mul", [c, acc], e)
        direct = self.emit(scope, "sub", [self.emit(scope, "exp", [x], e), self.const(scope, 1.0, x.type, e)], e)
        inside = self.emit(scope, "logical_and", [self.emit(scope, "ge", [x, self.const(scope, -0.5, x.type, e)], e),
                                                  self.emit(scope, "le", [x, self.const(scope, 0.5, x.type, e)], e)], e)
        return self.emit(scope, "where", [inside, series, direct], e)

    # -------------------------------------------------------------- decoder LLM ops (2026-11-24)

    def _emitter(self, scope, e):
        return lambda op, ins, **kw: self.emit(scope, op, ins, e, **kw)

    def _arange(self, scope, n, e):
        return self.emit(scope, "arange", [], e, start=0, end=n, step=1, dtype=torch.int64)

    def window_mask(self, scope, q, k, mask, window, e):
        """keep[i, j] = j <= i and j > i - window, combined with an existing bool or additive mask."""
        rows = self.emit(scope, "reshape", [self._arange(scope, q.type.shape[-2], e)], e, shape=[q.type.shape[-2], 1])
        cols = self.emit(scope, "reshape", [self._arange(scope, k.type.shape[-2], e)], e, shape=[1, k.type.shape[-2]])
        causal = self.emit(scope, "le", [cols, rows], e)
        lo = self.emit(scope, "sub", [rows, self.const(scope, window, rows.type, e)], e)
        keep = self.emit(scope, "logical_and", [causal, self.emit(scope, "gt", [cols, lo], e)], e)
        if mask is None:
            return keep
        if mask.type.dtype == torch.bool:
            return self.emit(scope, "logical_and", [keep, mask], e, ["window", "mask"])
        ninf = self.emit(scope, "full_like", [mask], e, value=-math.inf, dtype=mask.type.dtype)
        return self.emit(scope, "where", [keep, mask, ninf], e, ["window", "mask", "-inf"])

    def pieces(self, scope, x, d, sizes, e, what):
        """Split x along dim d into pieces of the given sizes: slices on a fixed dim, a reshape and
        selects on a symbolic dim (equal pieces whose size is provably an integer only)."""
        size = x.type.shape[d]
        if isinstance(size, int) and all(isinstance(p, int) for p in sizes):
            if sum(sizes) != size:
                raise self.err(e, f"{what} sizes {sizes} do not add up to dim {d} of size {size}", FrontendShapeError,
                               op=what, operands=[("x", x.type)])
            out, start = [], 0
            for p in sizes:
                out.append(self.emit(scope, "slice", [x], e, dim=d, start=start, end=start + p, step=1))
                start += p
            return tuple(out)
        each = sizes[0] if sizes else None
        ok = each is not None and all(dims_equal(p, each) for p in sizes) and \
            dims_equal(sympy.sympify(each) * len(sizes), size) and sympy.sympify(each).is_integer is True
        if not ok:
            raise self.err(e, f"{what} along the symbolic dim {size} needs a fixed number of equal pieces whose size "
                              "is provably an integer (for example 2 pieces of a dim 2*T)", FrontendShapeError,
                           op=what, operands=[("x", x.type)])
        shape = list(x.type.shape)
        grid = self.emit(scope, "reshape", [x], e, shape=shape[:d] + [len(sizes), each] + shape[d + 1:])
        return tuple(self.emit(scope, "select", [grid], e, dim=d, index=i) for i in range(len(sizes)))

    def lower_llm(self, name, a, need_t, static, e, L, scope):
        def axis(x, d):
            r = max(x.type.rank, 1)
            if not isinstance(d, int) or not -r <= d < r:
                raise self.err(e, f"dim {d} is out of range for {x.type}", FrontendShapeError, op=name,
                               operands=[(L[0], x.type)])
            return d % r

        def dtype_of(dt):
            return DTYPES[dt] if isinstance(dt, str) else dt

        if name == "cast":
            return self.emit(scope, "cast", [need_t("x")], e, L[:1], dtype=dtype_of(static("dtype")))
        if name == "arange":
            start, end, step = static("start"), static("end"), static("step")
            if end is None:
                start, end = 0, start
            return self.emit(scope, "arange", [], e, start=start, end=end, step=step, dtype=dtype_of(static("dtype")))
        if name == "rms_norm":
            ns = static("normalized_shape")
            ns = [ns] if isinstance(ns, int) or is_dim(ns) else list(ns)
            w = a["weight"]
            return self.emit(scope, "rms_norm", [need_t("x"), None if w is None else need_t("weight")], e, L[:2],
                             normalized_shape=ns, eps=float(static("eps")))
        if name == "cumsum":
            x = need_t("x")
            return self.emit(scope, "cumsum", [x], e, L[:1], dim=axis(x, static("dim")))
        if name == "flip":
            x = need_t("x")
            dims = static("dims")
            dims = [dims] if isinstance(dims, int) else list(dims)
            return self.emit(scope, "flip", [x], e, L[:1], dims=[axis(x, d) for d in dims])
        if name in ("argmax", "argmin"):
            x = need_t("x")
            dim = static("dim")
            return self.emit(scope, name, [x], e, L[:1], dim=None if dim is None else axis(x, dim),
                             keepdim=bool(static("keepdim")))
        if name in ("amax", "amin"):
            x = need_t("x")
            dim = static("dim")
            dims = [] if dim is None else ([dim] if isinstance(dim, int) else list(dim))
            return self.emit(scope, name, [x], e, L[:1], dims=[axis(x, d) for d in dims] or None,
                             keepdim=bool(static("keepdim")))
        if name == "sort":
            x = need_t("x")
            return self.emit_multi(scope, "sort", [x], e, L[:1], dim=axis(x, static("dim")),
                                   descending=bool(static("descending")), stable=True)
        if name == "topk":
            x = need_t("x")
            return self.emit_multi(scope, "topk", [x], e, L[:1], k=static("k"), dim=axis(x, static("dim")),
                                   largest=bool(static("largest")), sorted=True)
        if name in ("gather", "index_select"):
            x = need_t("x")
            return self.emit(scope, name, [x, need_t("index")], e, [L[0], L[2]], dim=axis(x, static("dim")))
        if name == "embedding":
            return self.emit(scope, "index", [need_t("weight"), need_t("ids")], e, L[:2])
        if name == "index_copy":
            x = need_t("x")
            return self.emit(scope, "index_copy", [x, need_t("index"), need_t("src")], e, [L[0], L[2], L[3]],
                             dim=axis(x, static("dim")))
        if name == "masked_fill":
            x = need_t("x")
            m = self.as_bool(need_t("mask"), e, scope)
            return self.emit(scope, "where", [m, need_t("value", like=x.type), x], e, [L[1], L[2], L[0]])
        if name == "repeat_interleave":
            x = need_t("x")
            return repeat_heads(self._emitter(scope, e), x, static("repeats"), axis(x, static("dim")))
        if name == "chunk":
            x = need_t("x")
            d, n = axis(x, static("dim")), static("chunks")
            size = x.type.shape[d]
            if isinstance(size, int):
                each = -(-size // n)
                sizes = [min(each, size - i) for i in range(0, size, each)] if size else [0]
            else:
                sizes = [norm_dim(sympy.sympify(size) / n)] * n
            return self.pieces(scope, x, d, sizes, e, "chunk")
        if name == "split":
            x = need_t("x")
            d, step = axis(x, static("dim")), static("split_size")
            size = x.type.shape[d]
            if isinstance(step, (list, tuple)):
                sizes = list(step)
            elif isinstance(size, int) and isinstance(step, int):
                sizes = [min(step, size - i) for i in range(0, size, step)] if size else [0]
            else:
                count = sympy.simplify(sympy.sympify(size) / step)
                if not count.is_Integer:
                    raise self.err(e, f"split of the symbolic dim {size} into pieces of {step} has a shape dependent "
                                      "piece count", FrontendShapeError, op="split", operands=[(L[0], x.type)])
                sizes = [step] * int(count)
            return self.pieces(scope, x, d, sizes, e, "split")
        if name == "multinomial":
            off = a["offset"]
            return self.emit(scope, "multinomial", [need_t("weights")] + ([] if off is None else [need_t("offset")]),
                             e, L[:1], seed=static("seed"), num_samples=static("num_samples"), replacement=True)
        if name == "top_p_filter":
            return self.top_p(scope, need_t("logits"), static("p"), static("min_keep"), e)
        return None

    def top_p(self, scope, x, p, min_keep, e):
        """Sort ascending, softmax, cumsum, drop tokens whose cumulative probability stays at or below
        1 - p (the last ``min_keep`` sorted tokens are always kept), then put the mask back in the
        original order by gathering with the inverse permutation (a second sort of the indices)."""
        if x.type.rank < 1 or not isinstance(x.type.shape[-1], int):
            raise self.err(e, "top_p_filter needs a fixed size last dim", FrontendShapeError, op="top_p_filter",
                           operands=[("logits", x.type)])
        n = x.type.shape[-1]
        d = x.type.rank - 1
        vals, idx = self.emit_multi(scope, "sort", [x], e, dim=d, descending=False, stable=True)
        cum = self.emit(scope, "cumsum", [self.emit(scope, "softmax", [vals], e, dim=d)], e, dim=d)
        remove = self.emit(scope, "le", [cum, self.const(scope, 1.0 - float(p), cum.type, e)], e)
        pos = self._arange(scope, n, e)
        early = self.emit(scope, "lt", [pos, self.const(scope, n - int(min_keep), pos.type, e)], e)
        remove = self.emit(scope, "logical_and", [remove, early], e)
        _, inv = self.emit_multi(scope, "sort", [idx], e, dim=d, descending=False, stable=True)
        remove = self.emit(scope, "gather", [remove, inv], e, dim=d)
        ninf = self.emit(scope, "full_like", [x], e, value=-math.inf, dtype=x.type.dtype)
        return self.emit(scope, "where", [remove, ninf, x], e)


_TENSOR_METHODS = {
    "sum": "sum", "mean": "mean", "any": "any", "all": "all", "softmax": "softmax", "relu": "relu",
    "exp": "exp", "tanh": "tanh", "sigmoid": "sigmoid", "sqrt": "sqrt", "rsqrt": "rsqrt", "abs": "abs",
    "clamp": "clamp", "matmul": "matmul", "transpose": "transpose", "reshape": "reshape", "view": "reshape",
    "permute": "permute", "unsqueeze": "unsqueeze", "squeeze": "squeeze", "size": "size", "dim": "dim",
    # decoder LLM ops (2026-11-24): same argument order as the ops, the tensor first
    "cos": "cos", "sin": "sin", "log": "log", "cumsum": "cumsum", "flip": "flip", "argmax": "argmax",
    "argmin": "argmin", "sort": "sort", "topk": "topk", "gather": "gather", "index_select": "index_select",
    "index_copy": "index_copy", "masked_fill": "masked_fill", "repeat_interleave": "repeat_interleave",
    "chunk": "chunk", "split": "split", "amax": "amax", "amin": "amin", "float": "float", "double": "double", "long": "long",
}
