"""Shared plumbing for the program transformations ``vmap`` (``vmap.py``) and forward mode
(``forward.py``): an emitter that appends shape checked nodes, inlining of a region or program,
dead code removal, and resolving a program from a native function.

The emitter offers the same methods the lowerings in ``grad_rules.py`` call (``op`` with ``out=``,
``const``, ``scale``, ``tensor_const``, ``zeros``, ``zeros_like``), so those lowerings, which are
checked against the fused reference ops to 1e-15, are reused here unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

import sympy
import torch

from .core import Dim, IRError, Node, Program, TensorType, Value, norm_dim
from .shapes import infer, infer_outputs


def as_program(f: Any) -> Program:
    """A Program, or the program of a ``@truestar.function`` (its ``program`` attribute)."""
    if isinstance(f, Program):
        return f
    prog = getattr(f, "program", None)
    if isinstance(prog, Program):
        return prog
    raise IRError(f"expected an IR Program or a @truestar.function, got {type(f).__name__}")


class Emitter:
    """Appends shape checked nodes to a program under construction.

    ``prefix`` names new values (``v_`` for vmap, ``t_`` for forward mode) so a printed program
    shows which pass wrote each node."""

    def __init__(self, names: Set[str], prefix: str, params: Optional[List[Value]] = None,
                 param_data: Optional[Dict[str, torch.Tensor]] = None):
        self.names = set(names)
        self.prefix = prefix
        self.inputs: List[Value] = []
        self.params: List[Value] = list(params or [])
        self.param_data: Dict[str, torch.Tensor] = dict(param_data or {})
        self.nodes: List[Node] = []
        self._counters: Dict[str, int] = {}
        self._consts: Dict[Any, Value] = {}
        self.context = ""

    @staticmethod
    def names_of(prog: Program) -> Set[str]:
        names = {v.name for v in prog.inputs + prog.params}
        names |= {o.name for n in prog.nodes for o in n.outputs}
        return names

    def fresh(self, base: str) -> str:
        name = base
        while name in self.names:
            self._counters[base] = self._counters.get(base, 0) + 1
            name = f"{base}_{self._counters[base]}"
        self.names.add(name)
        return name

    def new_input(self, name: str, t: TensorType) -> Value:
        v = Value(self.fresh(name), t, "input")
        self.inputs.append(v)
        return v

    def op(self, op: str, *inputs: Optional[Value], out: Optional[Value] = None, **attrs: Any) -> Value:
        t = infer(op, [None if x is None else x.type for x in inputs], attrs)
        if out is None:
            out = Value(self.fresh(f"{self.prefix}{op}"), t, "op")
        elif not t.same_as(out.type):
            raise IRError(f"internal error: {op} gives {t} but {out.ref} is declared {out.type}")
        self.nodes.append(Node(op, tuple(inputs), dict(attrs), out, self.context))
        return out

    def op_multi(self, op: str, *inputs: Optional[Value], outs: Optional[Sequence[Value]] = None,
                 **attrs: Any) -> List[Value]:
        ts = infer_outputs(op, [None if x is None else x.type for x in inputs], attrs)
        if outs is None:
            outs = [Value(self.fresh(f"{self.prefix}{op}"), t, "op") for t in ts]
        else:
            outs = list(outs)
            if len(outs) != len(ts) or not all(o.type.same_as(t) for o, t in zip(outs, ts)):
                raise IRError(f"internal error: {op} gives {[str(t) for t in ts]} but the node declares "
                              f"{[str(o.type) for o in outs]}")
        self.nodes.append(Node(op, tuple(inputs), dict(attrs), outs[0], self.context, tuple(outs[1:])))
        return outs

    def append(self, node: Node) -> None:
        self.nodes.append(node)

    def const(self, value: float, like: Value) -> Value:
        """A float scalar constant (0 dim, so it never widens a dtype) with ``like``'s dtype."""
        dtype = like.type.dtype if like.type.dtype.is_floating_point else torch.get_default_dtype()
        key = ("f", float(value), dtype)
        if key not in self._consts:
            v = Value(self.fresh(f"{self.prefix}const"), TensorType(dtype, ()), "param")
            self.params.append(v)
            self.param_data[v.name] = torch.tensor(float(value), dtype=torch.float64)
            self._consts[key] = v
        return self._consts[key]

    def int_const(self, value: int) -> Value:
        key = ("i", int(value))
        if key not in self._consts:
            v = Value(self.fresh(f"{self.prefix}int"), TensorType(torch.int64, ()), "param")
            self.params.append(v)
            self.param_data[v.name] = torch.tensor(int(value), dtype=torch.int64)
            self._consts[key] = v
        return self._consts[key]

    def tensor_const(self, base: str, data: torch.Tensor, dtype: torch.dtype) -> Value:
        from .core import declared_data
        v = Value(self.fresh(f"{self.prefix}{base}"), TensorType(dtype, tuple(data.shape)), "param")
        self.params.append(v)
        self.param_data[v.name] = declared_data(data, dtype, exact_only=True)   # declared dtype when exact
        return v

    def scale(self, v: Value, factor: float) -> Value:
        return self.op("mul", v, self.const(factor, v))

    def zeros(self, shape: Sequence[Dim], dtype: torch.dtype) -> Value:
        return self.op("full", shape=list(shape), value=0.0, dtype=dtype)

    def zeros_like(self, v: Value) -> Value:
        return self.op("full_like", v, value=0.0, dtype=None)

    def inline(self, region: Program, args: Sequence[Value], label: str = "") -> List[Value]:
        """Copy ``region``'s nodes here with its inputs bound to ``args``. Returns its outputs."""
        m: Dict[int, Value] = {}
        if len(args) != len(region.inputs):
            raise IRError(f"internal error: {region.name} takes {len(region.inputs)} inputs, got {len(args)}")
        for v, a in zip(region.inputs, args):
            if not v.type.same_as(a.type):
                raise IRError(f"internal error: input %{v.name}: {v.type} of {region.name} bound to {a.ref}: {a.type}")
            m[id(v)] = a
        for p in region.params:
            nv = Value(self.fresh(p.name), p.type, "param")
            self.params.append(nv)
            self.param_data[nv.name] = region.param_data[p.name]
            m[id(p)] = nv
        for n in region.nodes:
            outs = [Value(self.fresh(o.name), o.type, "op") for o in n.outputs]
            for o, no in zip(n.outputs, outs):
                m[id(o)] = no
            src = n.source if not label else (f"{label}; {n.source}" if n.source else label)
            self.nodes.append(Node(n.op, tuple(None if x is None else m[id(x)] for x in n.inputs), dict(n.attrs),
                                   outs[0], src, tuple(outs[1:])))
        return [m[id(v)] for v in region.outputs]


def numel(shape: Sequence[Dim]) -> Dim:
    out = sympy.Integer(1)
    for d in shape:
        out = out * sympy.sympify(d)
    return norm_dim(out)


def prune(prog: Program) -> Program:
    """Remove nodes none of whose values reaches an output, and params nothing reads. Every IR op is
    pure (random ops are pure functions of their seed and operands), so this changes no result.
    Regions are pruned too. Returns a new verified program."""
    def walk(p: Program) -> Program:
        needed = {id(v) for v in p.outputs}
        keep: List[Node] = []
        for n in reversed(p.nodes):
            if any(id(o) in needed for o in n.outputs):
                keep.append(n)
                needed.update(id(x) for x in n.inputs if x is not None)
        keep.reverse()
        nodes = []
        for n in keep:
            if any(isinstance(a, Program) for a in n.attrs.values()):
                attrs = {k: (walk(a) if isinstance(a, Program) else a) for k, a in n.attrs.items()}
                n = Node(n.op, n.inputs, attrs, n.output, n.source, n.extra_outputs)
            nodes.append(n)
        params = [v for v in p.params if id(v) in needed]
        data = {v.name: p.param_data[v.name] for v in params}
        return Program(p.name, list(p.inputs), params, data, nodes, list(p.outputs), dict(p.symbols))

    out = walk(prog)
    out.verify()
    return out
