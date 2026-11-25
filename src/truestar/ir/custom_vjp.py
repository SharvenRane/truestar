"""The ``custom_vjp`` IR op: a function whose backward is written by the user.

    %y = custom_vjp(%x, %w) name=softplus : f32[N]
      fwd(%x: f32[N], %w: f32[N]) -> (f32[N]) { ... }               the primal
      bwd(%x, %w, %out, %g) -> (f32[N], f32[N]) { ... }               one gradient per float operand

Running the node runs ``fwd``. Differentiating it (``truestar.ir.grad``) does not look inside ``fwd``:
the rule inlines ``bwd`` with the node's operands, its output and the output cotangent, so the
emitted backward code is the user's own rule as ordinary IR nodes. Both regions are closed programs
(every value they read is an operand), like the regions of ``cond`` and ``loop``.

Compiled executors do not need to know the op: ``inline_custom_vjp(program)`` replaces every
``custom_vjp`` node with the nodes of its ``fwd`` region (the forward is the primal, exactly).

The rule is checked when the front end registers it (``truestar.frontend.custom_grad``), not here.
Registered on import of ``truestar.ir``. See docs/AUTODIFF.md, section "Custom gradients".
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from . import reference, shapes
from .core import IRError, Node, Program, ShapeError, TensorType, Value
from .grad_rules import grad_rule

OP = "custom_vjp"


def _float_operands(types: Sequence[TensorType]) -> List[int]:
    return [i for i, t in enumerate(types) if t is not None and t.dtype.is_floating_point]


@shapes.rule(OP)
def _shape(types, attrs):
    if any(t is None for t in types):
        raise ShapeError(OP, "operands cannot be absent")
    fwd, bwd = attrs.get("fwd"), attrs.get("bwd")
    shapes._check_region(OP, "fwd", fwd, types)
    if len(fwd.outputs) != 1:
        raise ShapeError(OP, f"fwd must return exactly one value, it returns {len(fwd.outputs)}")
    out = fwd.outputs[0].type
    if not out.dtype.is_floating_point:
        raise ShapeError(OP, f"the output must be floating point to have a gradient, got {out}")
    shapes._check_region(OP, "bwd", bwd, list(types) + [out, out])
    fl = _float_operands(types)
    if len(bwd.outputs) != len(fl):
        raise ShapeError(OP, f"bwd must return one gradient per floating point operand ({len(fl)}), "
                             f"it returns {len(bwd.outputs)}")
    for i, o in zip(fl, bwd.outputs):
        if not o.type.same_as(types[i]):
            raise ShapeError(OP, f"the gradient of operand {i} has type {o.type}, the operand is {types[i]}",
                             [(f"operand {i}", types[i]), ("gradient", o.type)])
    return out


@reference.impl(OP)
def _run(attrs, args, shape, bindings=None):
    return reference._run(attrs["fwd"], args, bindings)[0]


reference.BINDING_OPS.add(OP)


# ----------------------------------------------------------------------------- inlining


def _inline_region(region: Program, args: Sequence[Value], fresh, add_param, append, source: str) -> List[Value]:
    """Copy ``region``'s nodes behind the caller's nodes with its inputs bound to ``args``.

    ``fresh(base)`` gives an unused value name, ``add_param(value, data)`` registers a constant,
    ``append(node)`` emits a node. Returns the values the region's outputs map to."""
    m: Dict[int, Value] = {}
    for v, a in zip(region.inputs, args):
        if not v.type.same_as(a.type):
            raise IRError(f"internal error: region input %{v.name}: {v.type} bound to {a.ref}: {a.type}")
        m[id(v)] = a
    for p in region.params:
        nv = Value(fresh(p.name), p.type, "param")
        add_param(nv, region.param_data[p.name])
        m[id(p)] = nv
    for n in region.nodes:
        outs = [Value(fresh(o.name), o.type, "op") for o in n.outputs]
        for o, no in zip(n.outputs, outs):
            m[id(o)] = no
        src = n.source if not source else (f"{source}; {n.source}" if n.source else source)
        append(Node(n.op, tuple(None if x is None else m[id(x)] for x in n.inputs), dict(n.attrs), outs[0],
                    src, tuple(outs[1:])))
    return [m[id(v)] for v in region.outputs]


@grad_rule(OP)
def _grad(e, node: Node, g: Value):
    """Inline the user's ``bwd`` region once, the first time any operand asks for its cotangent."""
    done: List[List[Value]] = []

    def emit() -> List[Value]:
        if not done:
            def add_param(v, data):
                e.params.append(v)
                e.param_data[v.name] = data
            label = f"custom gradient of {node.attrs.get('name', 'function')}"
            done.append(_inline_region(node.attrs["bwd"], list(node.inputs) + [node.output, g],
                                       lambda base: e.fresh(f"d_{base}"), add_param, e.nodes.append, label))
        return done[0]

    thunks = []
    k = 0
    for x in node.inputs:
        if x is not None and x.type.dtype.is_floating_point:
            thunks.append((lambda k=k: emit()[k]))
            k += 1
        else:
            thunks.append(None)
    return thunks


def has_custom_vjp(program: Program) -> bool:
    return any(n.op == OP or any(isinstance(a, Program) and has_custom_vjp(a) for a in n.attrs.values())
               for n in program.nodes)


def inline_custom_vjp(program: Program) -> Program:
    """``program`` with every ``custom_vjp`` node replaced by its forward region, also inside the
    regions of control flow nodes. Returns ``program`` itself when there is nothing to replace."""
    if not has_custom_vjp(program):
        return program
    names = {v.name for v in program.inputs + program.params}
    names |= {o.name for n in program.nodes for o in n.outputs}
    params = list(program.params)
    data = dict(program.param_data)
    nodes: List[Node] = []
    counter = [0]

    def fresh(base: str) -> str:
        name = base
        while name in names:
            counter[0] += 1
            name = f"{base}_{counter[0]}"
        names.add(name)
        return name

    def add_param(v, d):
        params.append(v)
        data[v.name] = d

    remap: Dict[int, Value] = {}
    for n in program.nodes:
        ins = tuple(None if x is None else remap.get(id(x), x) for x in n.inputs)
        if n.op == OP:
            fwd = inline_custom_vjp(n.attrs["fwd"])
            (y,) = _inline_region(fwd, ins, fresh, add_param, nodes.append, "")
            remap[id(n.output)] = y
            continue
        attrs = {k: (inline_custom_vjp(v) if isinstance(v, Program) else v) for k, v in n.attrs.items()}
        nodes.append(Node(n.op, ins, attrs, n.output, n.source, n.extra_outputs))
    outputs = [remap.get(id(v), v) for v in program.outputs]
    out = Program(program.name, list(program.inputs), params, data, nodes, outputs, dict(program.symbols))
    out.verify()
    return out
