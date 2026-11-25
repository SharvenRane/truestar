"""Extend an existing IR program with new inputs, constants and nodes.

The training step is built by appending to programs that already exist (an imported model, its
gradient program), so this is a builder that starts from a program instead of from nothing. Every
op goes through the IR's shape rules the moment it is added, exactly as ``ir.Builder`` does, and
``build`` re verifies the whole program.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch

from ..ir.core import IRError, Node, Program, ShapeError, TensorType, Value, symbol
from ..ir.shapes import infer


class Editor:
    def __init__(self, prog: Program, name: Optional[str] = None):
        self.name = name or prog.name
        self.inputs: List[Value] = list(prog.inputs)
        self.params: List[Value] = list(prog.params)
        self.param_data: Dict[str, torch.Tensor] = dict(prog.param_data)
        self.nodes: List[Node] = list(prog.nodes)
        self.outputs: List[Value] = list(prog.outputs)
        self.symbols: Dict[str, Tuple[int, Optional[int]]] = dict(prog.symbols)
        self.names: Set[str] = {v.name for v in self.inputs + self.params}
        for n in self.nodes:
            self.names.update(o.name for o in n.outputs)
        self._counters: Dict[str, int] = {}
        self._consts: Dict[Any, Value] = {}
        self.source = "truestar.train"

    def fresh(self, base: str) -> str:
        name = base
        while name in self.names:
            self._counters[base] = self._counters.get(base, 0) + 1
            name = f"{base}_{self._counters[base]}"
        self.names.add(name)
        return name

    def input(self, name: str, dtype: torch.dtype, shape: Sequence[Any]) -> Value:
        """A new program input. The name is used as given; a clash is an error, not a rename,
        because callers address training step inputs by name."""
        if name in self.names:
            raise IRError(f"cannot add input %{name}: the program already defines a value of that name")
        dims = []
        for d in shape:
            if isinstance(d, str):
                self.symbols.setdefault(d, (1, None))
                d = symbol(d)
            dims.append(d)
        self.names.add(name)
        v = Value(name, TensorType(dtype, tuple(dims)), "input")
        self.inputs.append(v)
        return v

    def const(self, value: float, dtype: torch.dtype) -> Value:
        """A 0 dim constant. The stored data is the exact Python number in float64 (as the importer
        stores scalar constants); a 0 dim operand never widens the dtype of the other operand."""
        key = (float(value), dtype)
        if key not in self._consts:
            v = Value(self.fresh("t_const"), TensorType(dtype, ()), "param")
            self.params.append(v)
            data_dtype = torch.float64 if dtype.is_floating_point else dtype
            self.param_data[v.name] = torch.tensor(value, dtype=data_dtype)
            self._consts[key] = v
        return self._consts[key]

    def op(self, op: str, *inputs: Optional[Value], name: Optional[str] = None, source: str = "",
           **attrs: Any) -> Value:
        src = source or self.source
        try:
            t = infer(op, [None if x is None else x.type for x in inputs], attrs)
        except ShapeError as e:
            raise e.with_source(src)
        out = Value(self.fresh(name or op), t, "op")
        self.nodes.append(Node(op, tuple(inputs), dict(attrs), out, src))
        return out

    def build(self, outputs: Optional[Sequence[Value]] = None, name: Optional[str] = None) -> Program:
        prog = Program(name or self.name, list(self.inputs), list(self.params), dict(self.param_data),
                       list(self.nodes), list(self.outputs if outputs is None else outputs), dict(self.symbols))
        prog.verify()
        return prog


def params_to_inputs(prog: Program, names: Sequence[str]) -> Program:
    """A copy of ``prog`` where the params ``names`` are program inputs instead (appended after the
    existing inputs, in the given order). Training feeds weights in and gets new weights out, so the
    weights cannot be constants baked into the program."""
    by_name = {p.name: p for p in prog.params}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise IRError(f"{prog.name} has no param(s) {missing}. Params: {sorted(by_name)}")
    if len(set(names)) != len(names):
        raise IRError(f"a param is listed twice in {list(names)}")
    repl: Dict[int, Value] = {}
    new_inputs = list(prog.inputs)
    for n in names:
        p = by_name[n]
        if not p.type.dtype.is_floating_point:
            raise IRError(f"param @{n}: {p.type} is not floating point and cannot be trained")
        v = Value(p.name, p.type, "input")
        repl[id(p)] = v
        new_inputs.append(v)
    nodes = []
    for node in prog.nodes:
        if any(x is not None and id(x) in repl for x in node.inputs):
            node = Node(node.op, tuple(repl.get(id(x), x) if x is not None else None for x in node.inputs),
                        dict(node.attrs), node.output, node.source, node.extra_outputs)
        nodes.append(node)
    params = [p for p in prog.params if id(p) not in repl]
    data = {p.name: prog.param_data[p.name] for p in params}
    outputs = [repl.get(id(v), v) for v in prog.outputs]
    out = Program(prog.name, new_inputs, params, data, nodes, outputs, dict(prog.symbols))
    out.verify()
    return out


def inputs_to_group(prog: Program, names: Sequence[str], group: str,
                    multiple: int = 1) -> Tuple[Program, Dict[str, Tuple[int, tuple]]]:
    """A copy of ``prog`` where the inputs ``names`` (one float dtype, fixed shapes) are views of one
    new flat input ``group``: each is ``reshape(slice(group, offset, offset + numel))``, so a training
    step keeps every param in one buffer and its optimizer update is one kernel per value instead of
    one per param. The group input takes the place of the first of them in the input order. Its
    length is rounded up to a multiple of ``multiple``; the tail belongs to no param. Returns the
    program and the layout ``{name: (offset, shape)}``."""
    by_name = {v.name: v for v in prog.inputs}
    vals = [by_name[n] for n in names]
    dtypes = {v.type.dtype for v in vals}
    if len(dtypes) != 1:
        raise IRError(f"a param group needs one dtype, got {sorted(map(str, dtypes))} for {list(names)}")
    layout: Dict[str, Tuple[int, tuple]] = {}
    off = 0
    for v in vals:
        if not all(isinstance(d, int) for d in v.type.shape):
            raise IRError(f"param %{v.name}: {v.type} has a symbolic size and cannot join a param group")
        layout[v.name] = (off, tuple(v.type.shape))
        off += _numel(v.type.shape)
    off = -(-off // multiple) * multiple
    e = Editor(prog)
    e.nodes = []
    if group in e.names:
        raise IRError(f"cannot add the param group %{group}: the program already defines a value of that name")
    flat = Value(group, TensorType(next(iter(dtypes)), (off,)), "input")
    e.names.add(group)
    e.source = "truestar.train param group views"
    repl: Dict[int, Value] = {}
    for v in vals:
        start, shape = layout[v.name]
        s = e.op("slice", flat, dim=0, start=start, end=start + _numel(shape), step=1, name=f"{v.name}.flat")
        repl[id(v)] = e.op("reshape", s, shape=list(shape), name=f"{v.name}.view")
    nodes = list(e.nodes)
    for node in prog.nodes:
        if any(x is not None and id(x) in repl for x in node.inputs):
            node = Node(node.op, tuple(repl.get(id(x), x) if x is not None else None for x in node.inputs),
                        dict(node.attrs), node.output, node.source, node.extra_outputs)
        nodes.append(node)
    first = min(prog.inputs.index(v) for v in vals)
    inputs = [flat if i == first else x for i, x in enumerate(prog.inputs) if id(x) not in repl or i == first]
    outputs = [repl.get(id(v), v) for v in prog.outputs]
    out = Program(prog.name, inputs, list(prog.params), dict(prog.param_data), nodes, outputs, dict(prog.symbols))
    out.verify()
    return out, layout


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def retype(prog: Program, per_node, name: Optional[str] = None, extra_inputs=()) -> Program:
    """Rebuild ``prog`` node by node. ``per_node(editor, node, new_inputs)`` returns the value that
    replaces ``node.output`` (it may emit casts around the op). Values are retyped from the IR's own
    rules, so a dtype change propagates to every reader."""
    e = Editor(prog, name)
    e.nodes = []  # e.names keeps every original name, so fresh names never clash with them
    mapping: Dict[int, Value] = {}
    for node in prog.nodes:
        if node.extra_outputs:
            raise IRError(f"cannot transform {node.op} (%{node.output.name}): nodes with several outputs "
                          "(control flow) are not supported by this transformation yet")
        ins = [None if x is None else mapping.get(id(x), x) for x in node.inputs]
        mapping[id(node.output)] = per_node(e, node, ins)
    outs = [mapping.get(id(v), v) for v in prog.outputs]
    return e.build(outs)


def copy_node(e: Editor, node: Node, ins: Sequence[Optional[Value]]) -> Value:
    """Re emit ``node`` with new operands under its own name, retyped by the shape rules."""
    if all(a is b for a, b in zip(ins, node.inputs)):
        e.nodes.append(node)
        return node.output
    t = infer(node.op, [None if x is None else x.type for x in ins], node.attrs)
    if t.same_as(node.output.type):
        out = node.output
    else:
        out = Value(node.output.name, t, "op")
    e.nodes.append(Node(node.op, tuple(ins), dict(node.attrs), out, node.source))
    return out
