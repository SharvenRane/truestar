"""Core data structures of the Truestar tensor IR.

A program is a list of nodes in SSA form. Every value has a type: a dtype and a shape whose dims are
ints or symbolic expressions over named symbols (for example ``B`` or ``4*B``). Ops are pure: a
node reads its input values and defines exactly one new value. Nothing is ever mutated and no two
values share storage, so a pass can reorder or fuse nodes without alias analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import sympy
import torch

Dim = Union[int, sympy.Expr]
Shape = Tuple[Dim, ...]


# ----------------------------------------------------------------------------- dims


def declared_data(data: torch.Tensor, dtype: torch.dtype, exact_only: bool = False) -> torch.Tensor:
    """The stored data of a constant table of declared ``dtype``: a float tensor with at least one dim
    rounded once to that dtype (an exported artifact stores every tensor in its declared dtype,
    ir/serialize.py, so a float32 program with a float64 table could not be written); a float scalar
    keeps its float64 value (exact, the way scalar constants are typed like Python numbers); other
    data unchanged. ``exact_only``: round only when no entry changes, else keep float64 (autodiff
    tables, whose float64 evaluation must stay exact: 1/3 of an adaptive pool gradient)."""
    if data.dtype.is_floating_point and dtype.is_floating_point:
        if data.dim() == 0:
            return data.to(torch.float64)
        out = data.to(dtype)
        if exact_only and not torch.equal(out.to(torch.float64), data.to(torch.float64)):
            return data.to(torch.float64)
        return out
    return data


def symbol(name: str) -> sympy.Symbol:
    """A named symbolic dim. Symbolic dims are positive integers."""
    return sympy.Symbol(name, integer=True, positive=True)


def norm_dim(d: Any) -> Dim:
    """Canonical form of a dim: a plain int when it is a constant, else an expanded expression."""
    if isinstance(d, bool):
        raise TypeError(f"a dim cannot be a bool: {d!r}")
    if isinstance(d, int):
        return d
    if isinstance(d, sympy.Basic):
        d = sympy.expand(d)
        if d.is_Integer:
            return int(d)
        return d
    raise TypeError(f"a dim must be an int or a sympy expression, got {type(d).__name__}: {d!r}")


def dims_equal(a: Dim, b: Dim) -> bool:
    """True only when the two dims are provably equal for every value of their symbols."""
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    return sympy.simplify(sympy.sympify(a) - sympy.sympify(b)) == 0


def is_const(d: Dim) -> bool:
    return isinstance(d, int)


def eval_dim(d: Dim, bindings: Dict[sympy.Symbol, int]) -> int:
    """Value of a dim once every symbol has a concrete value."""
    if isinstance(d, int):
        return d
    v = sympy.sympify(d).subs(bindings)
    if not v.is_Integer:
        raise ValueError(f"dim {d} does not evaluate to an integer with {bindings} (got {v})")
    return int(v)


def dim_str(d: Dim) -> str:
    return str(d)


# ----------------------------------------------------------------------------- types

_DTYPE_NAMES = {
    torch.float64: "f64", torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16",
    torch.int64: "i64", torch.int32: "i32", torch.int16: "i16", torch.int8: "i8",
    torch.uint8: "u8", torch.bool: "bool",
}


@dataclass(frozen=True)
class TensorType:
    dtype: torch.dtype
    shape: Shape

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(norm_dim(d) for d in self.shape))

    @property
    def rank(self) -> int:
        return len(self.shape)

    def same_as(self, other: "TensorType") -> bool:
        return (self.dtype == other.dtype and self.rank == other.rank
                and all(dims_equal(a, b) for a, b in zip(self.shape, other.shape)))

    def __str__(self) -> str:
        name = _DTYPE_NAMES.get(self.dtype, str(self.dtype).replace("torch.", ""))
        return f"{name}[{', '.join(dim_str(d) for d in self.shape)}]"


@dataclass(eq=False)
class Value:
    """An SSA value. ``kind`` is "input", "param" (weights and constants) or "op"."""

    name: str
    type: TensorType
    kind: str = "op"

    def __repr__(self) -> str:
        sigil = "@" if self.kind == "param" else "%"
        return f"{sigil}{self.name}: {self.type}"

    @property
    def ref(self) -> str:
        return ("@" if self.kind == "param" else "%") + self.name


@dataclass(eq=False)
class Node:
    """One pure op. ``inputs`` may hold None for an absent optional operand (for example no bias)."""

    op: str
    inputs: Tuple[Optional[Value], ...]
    attrs: Dict[str, Any]
    output: Value
    source: str = ""  # where it came from: ATen target, node name, user source line
    extra_outputs: Tuple[Value, ...] = ()  # control flow ops (cond, loop) define more than one value

    @property
    def outputs(self) -> Tuple[Value, ...]:
        """Every value this node defines: ``output`` first, then ``extra_outputs``."""
        return (self.output,) + tuple(self.extra_outputs)


# ----------------------------------------------------------------------------- errors


class IRError(Exception):
    pass


class ShapeError(IRError):
    """A shape or dtype mismatch found before anything runs. Names the op and every operand type."""

    def __init__(self, op: str, reason: str, operands: Sequence[Tuple[str, Any]] = (), source: str = ""):
        self.op = op
        self.reason = reason
        self.operands = list(operands)
        self.source = source
        super().__init__(self._message())

    def _message(self) -> str:
        lines = [f"shape error in {self.op}: {self.reason}"]
        for label, t in self.operands:
            lines.append(f"  {label}: {t}")
        if self.source:
            lines.append(f"  at: {self.source}")
        return "\n".join(lines)

    def with_source(self, source: str) -> "ShapeError":
        return ShapeError(self.op, self.reason, self.operands, source)


class UnsupportedOpError(IRError):
    """An op (or an op setting) the IR cannot represent. Raised, never skipped."""


# ----------------------------------------------------------------------------- program


@dataclass
class Program:
    name: str
    inputs: List[Value]
    params: List[Value]
    param_data: Dict[str, torch.Tensor]
    nodes: List[Node]
    outputs: List[Value]
    symbols: Dict[str, Tuple[int, Optional[int]]] = field(default_factory=dict)  # name: (min, max)
    # Training programs (docs/RANDOM.md): state carried from one call to the next, as
    # (name, input index, output index), with starting values; and the name of the int64 key input
    # that every random op reads, when the program has one.
    state: List[Tuple[str, int, int]] = field(default_factory=list)
    state_init: Dict[str, torch.Tensor] = field(default_factory=dict)
    key: Optional[str] = None

    def verify(self) -> None:
        """Re check SSA form and every shape rule. Raises IRError or ShapeError."""
        from .shapes import infer_outputs

        defined = {id(v) for v in self.inputs} | {id(v) for v in self.params}
        names = set()
        for v in self.inputs + self.params:
            if v.name in names:
                raise IRError(f"value name defined twice: {v.name}")
            names.add(v.name)
        for v in self.params:
            data = self.param_data.get(v.name)
            if data is None:
                raise IRError(f"param @{v.name} has no data")
            if tuple(data.shape) != tuple(v.type.shape):
                raise ShapeError("param", "stored data does not match the declared type",
                                 [(f"@{v.name}", v.type), ("data", tuple(data.shape))])
        for node in self.nodes:
            for x in node.inputs:
                if x is not None and id(x) not in defined:
                    raise IRError(f"{node.op} uses {x.ref} before it is defined")
            ts = infer_outputs(node.op, [None if x is None else x.type for x in node.inputs], node.attrs)
            if len(ts) != len(node.outputs):
                raise IRError(f"{node.op} defines {len(node.outputs)} values but its rule gives {len(ts)}")
            for out, t in zip(node.outputs, ts):
                if not t.same_as(out.type):
                    raise ShapeError(node.op, "stored output type does not match the shape rule",
                                     [("stored", out.type), ("rule", t)], node.source)
                if id(out) in defined or out.name in names:
                    raise IRError(f"value defined twice: {out.ref}")
                defined.add(id(out))
                names.add(out.name)
        for v in self.outputs:
            if id(v) not in defined:
                raise IRError(f"output {v.ref} is never defined")

    def to_text(self, show_params: bool = True, _region: str = "") -> str:
        ins = ", ".join(f"%{v.name}: {v.type}" for v in self.inputs)
        outs = ", ".join(str(v.type) for v in self.outputs)
        if _region:
            lines = [f"{_region}({ins}) -> ({outs}) {{"]
            if show_params:
                for p in self.params:
                    lines.append(f"    @{p.name}: {p.type}")
        else:
            lines = [f"program {self.name}({ins}) -> ({outs})"]
            for s, (lo, hi) in self.symbols.items():
                lines.append(f"  symbol {s} in [{lo}, {'inf' if hi is None else hi}]")
            n_values = sum(int(self.param_data[p.name].numel()) for p in self.params)
            lines.append(f"  params: {len(self.params)} tensors, {n_values} values")
            if show_params:
                for p in self.params:
                    lines.append(f"    @{p.name}: {p.type}")
        for node in self.nodes:
            args = ", ".join("_" if x is None else x.ref for x in node.inputs)
            attrs = "".join(f" {k}={_fmt_attr(v)}" for k, v in node.attrs.items() if not isinstance(v, Program))
            outs_ = ", ".join(f"%{o.name}" for o in node.outputs)
            types = ", ".join(str(o.type) for o in node.outputs)
            lines.append(f"  {outs_} = {node.op}({args}){attrs} : {types}")
            for k, v in node.attrs.items():
                if isinstance(v, Program):
                    lines.extend("    " + ln for ln in v.to_text(show_params, _region=k).splitlines())
        lines.append(("  yield " if _region else "  return ") + ", ".join(v.ref for v in self.outputs))
        if _region:
            lines.append("}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()

    def op_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for n in self.nodes:
            counts[n.op] = counts.get(n.op, 0) + 1
        return counts

    def regions(self) -> List["Program"]:
        """Sub programs held by control flow nodes (cond branches, loop bodies), direct children only."""
        return [v for n in self.nodes for v in n.attrs.values() if isinstance(v, Program)]


def _fmt_attr(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_attr(x) for x in v) + "]"
    if isinstance(v, torch.dtype):
        return _DTYPE_NAMES.get(v, str(v))
    return str(v)


# ----------------------------------------------------------------------------- builder


class Builder:
    """Builds a program op by op. Every op is shape checked the moment it is added.

    >>> b = Builder("mlp")
    >>> x = b.input("x", torch.float32, ("B", 16))
    >>> w = b.param("w", torch.randn(8, 16))
    >>> y = b.op("linear", x, w, None)
    >>> prog = b.build([y])
    """

    def __init__(self, name: str = "program"):
        self.name = name
        self.inputs: List[Value] = []
        self.params: List[Value] = []
        self.param_data: Dict[str, torch.Tensor] = {}
        self.nodes: List[Node] = []
        self.symbols: Dict[str, Tuple[int, Optional[int]]] = {}
        self._names: set = set()
        self._counter = 0

    def _fresh(self, base: str) -> str:
        name = base
        while name in self._names:
            self._counter += 1
            name = f"{base}_{self._counter}"
        self._names.add(name)
        return name

    def symbol(self, name: str, lo: int = 1, hi: Optional[int] = None) -> sympy.Symbol:
        self.symbols.setdefault(name, (lo, hi))
        return symbol(name)

    def input(self, name: str, dtype: torch.dtype, shape: Sequence[Union[int, str, sympy.Expr]]) -> Value:
        dims = [self.symbol(d) if isinstance(d, str) else d for d in shape]
        v = Value(self._fresh(name), TensorType(dtype, tuple(dims)), "input")
        self.inputs.append(v)
        return v

    def param(self, name: str, data: torch.Tensor) -> Value:
        data = data.detach()
        v = Value(self._fresh(name), TensorType(data.dtype, tuple(data.shape)), "param")
        self.params.append(v)
        self.param_data[v.name] = data
        return v

    def op(self, op: str, *inputs: Optional[Value], name: Optional[str] = None,
           source: str = "", **attrs: Any) -> Value:
        from .shapes import infer

        try:
            t = infer(op, [None if x is None else x.type for x in inputs], attrs)
        except ShapeError as e:
            raise e.with_source(source) if source else e
        out = Value(self._fresh(name or op), t, "op")
        self.nodes.append(Node(op, tuple(inputs), dict(attrs), out, source))
        return out

    def op_multi(self, op: str, *inputs: Optional[Value], names: Optional[Sequence[str]] = None,
                 source: str = "", **attrs: Any) -> Tuple[Value, ...]:
        """Add an op that defines several values (``cond``, ``loop``). Returns them in order."""
        from .shapes import infer_outputs

        try:
            ts = infer_outputs(op, [None if x is None else x.type for x in inputs], attrs)
        except ShapeError as e:
            raise e.with_source(source) if source else e
        names = list(names or [])
        outs = tuple(Value(self._fresh(names[i] if i < len(names) and names[i] else op), t, "op")
                     for i, t in enumerate(ts))
        self.nodes.append(Node(op, tuple(inputs), dict(attrs), outs[0], source, outs[1:]))
        return outs

    def build(self, outputs: Sequence[Value]) -> Program:
        prog = Program(self.name, list(self.inputs), list(self.params), dict(self.param_data),
                       list(self.nodes), list(outputs), dict(self.symbols))
        prog.verify()
        return prog
