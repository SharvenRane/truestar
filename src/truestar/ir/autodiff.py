"""Reverse mode automatic differentiation as a compiler pass.

``grad(program, wrt=[...])`` returns a new IR program that computes the original outputs and the
gradient of those outputs with respect to chosen inputs or params. The gradient is a vector
Jacobian product: for an output ``y`` and a cotangent ``v`` of the same shape, the program returns
``J^T v`` for every ``wrt`` value. For a scalar output with ``v = 1`` that is the plain gradient.

It is a pure program transformation. The backward pass is emitted as ordinary IR nodes after the
forward nodes, so there is no tape and no interpreter state at run time, the result type checks
like any program (every emitted op goes through the IR's shape rules, symbolic dims included), and
a compiler can later fuse and schedule forward and backward together.

What the pass takes care of:

* **Fan out.** A value read by several nodes receives one cotangent from each; they are added.
* **Broadcasting.** A cotangent is summed back to the operand's shape. With symbolic dims the IR
  only lets a dim broadcast when it is provably 1, so the axes to sum are known at transform time.
* **Dead code.** A node gets backward code only when its output depends on a ``wrt`` value and
  reaches a differentiated output, and only for the operands that depend on a ``wrt`` value.
* **No silent gaps.** An op with no gradient rule on the gradient path is an error naming the op
  and where it came from. All such ops are listed at once and nothing is skipped.

See docs/AUTODIFF.md and ``grad_rules.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Union

import torch

from .core import (Dim, IRError, Node, Program, ShapeError, TensorType, UnsupportedOpError, Value,
                   dims_equal)
from .grad_rules import GRAD_RULES, LOWERINGS, NONDIFF, rule_table as _rule_table
from . import attention_grad as _attention_grad
from .shapes import infer
from .autodiff_cf import CF_FORWARD, CF_RULES, LOOP_MEMORY

__all__ = ["grad", "rule_table", "GRAD_RULES", "LOWERINGS", "NONDIFF", "LOOP_MEMORY"]


def rule_table() -> Dict[str, str]:
    """Every op the pass knows, with how it is handled: "rule", "lowered", "no gradient" or
    "control flow" (cond and loop, autodiff_cf.py)."""
    table = _rule_table()
    table.update({op: "control flow" for op in CF_RULES})
    return table


def _control_flow_backward(e, n: Node, total, add_cotangent, active: Set[int]) -> None:
    """Backward of a node that defines several values (cond, loop): collect a cotangent per result,
    let the control flow rule emit the backward node, hand each operand its cotangent."""
    gs = [total(o) for o in n.outputs]
    if all(g is None for g in gs):
        return
    try:
        grads = CF_RULES[n.op](e, n, gs, active)
    except UnsupportedOpError as err:
        raise UnsupportedOpError(f"grad cannot differentiate {n.op} ({_where(n)}): {err}") from None
    for x, g in zip(n.inputs, grads):
        if x is not None and g is not None and id(x) in active:
            add_cotangent(x, g, f"the gradient of {n.op}")


class _Emitter:
    """Appends shape checked nodes and constants to the program being built."""

    def __init__(self, prog: Program):
        self.inputs: List[Value] = list(prog.inputs)
        self.params: List[Value] = list(prog.params)
        self.param_data: Dict[str, torch.Tensor] = dict(prog.param_data)
        self.nodes: List[Node] = []
        self.names: Set[str] = {v.name for v in prog.inputs} | {v.name for v in prog.params}
        self.names |= {o.name for n in prog.nodes for o in n.outputs}
        self._counters: Dict[str, int] = {}
        self._consts: Dict[Any, Value] = {}
        self.context = ""  # what is being differentiated, kept in each emitted node's source
        self.loop_memory = "auto"  # memory policy for loops, see autodiff_cf.py
        self.cf_plans: Dict[int, Any] = {}  # control flow node -> what its backward needs

    def fresh(self, base: str) -> str:
        name = base
        while name in self.names:
            self._counters[base] = self._counters.get(base, 0) + 1
            name = f"{base}_{self._counters[base]}"
        self.names.add(name)
        return name

    def op(self, op: str, *inputs: Optional[Value], out: Optional[Value] = None, **attrs: Any) -> Value:
        """Emit one node. With ``out`` the node defines that existing value (used by lowerings, whose
        last op must define the value the composite op defined) after checking the type agrees."""
        t = infer(op, [None if x is None else x.type for x in inputs], attrs)
        if out is None:
            out = Value(self.fresh(f"d_{op}"), t, "op")
        elif not t.same_as(out.type):
            raise IRError(f"internal error: lowering to {op} gives {t} but {out.ref} is declared {out.type}")
        self.nodes.append(Node(op, tuple(inputs), dict(attrs), out, self.context))
        return out

    def const(self, value: float, like: Value) -> Value:
        """A scalar constant as a 0 dim param with the dtype of ``like``. The stored value keeps the
        exact Python number, the same way the importer stores scalar constants."""
        dtype = like.type.dtype if like.type.dtype.is_floating_point else torch.get_default_dtype()
        key = (float(value), dtype)
        if key not in self._consts:
            v = Value(self.fresh("d_const"), TensorType(dtype, ()), "param")
            self.params.append(v)
            self.param_data[v.name] = torch.tensor(float(value), dtype=torch.float64)
            self._consts[key] = v
        return self._consts[key]

    def tensor_const(self, base: str, data: torch.Tensor, dtype: torch.dtype) -> Value:
        from .core import declared_data
        v = Value(self.fresh(f"d_{base}"), TensorType(dtype, tuple(data.shape)), "param")
        self.params.append(v)
        self.param_data[v.name] = declared_data(data, dtype, exact_only=True)   # declared dtype when exact
        return v

    def scale(self, v: Value, factor: float) -> Value:
        return self.op("mul", v, self.const(factor, v))

    def zeros(self, shape: Sequence[Dim], dtype: torch.dtype) -> Value:
        return self.op("full", shape=list(shape), value=0.0, dtype=dtype)

    def zeros_like(self, v: Value) -> Value:
        return self.op("full_like", v, value=0.0, dtype=None)

    def sum_to(self, g: Value, t: TensorType) -> Value:
        """Undo broadcasting: sum ``g`` down to the shape of an operand of type ``t``.

        Leading dims the operand does not have are summed away. A dim where the operand is provably
        1 and ``g`` is not is summed and kept. The shape rules only allow a dim to broadcast when it
        is provably 1, so this is exact for every value of the symbols.
        """
        gs = g.type.shape
        lead = len(gs) - t.rank
        if lead < 0:
            raise IRError(f"internal error: cotangent {g.type} has lower rank than its operand {t}")
        axes = list(range(lead))
        for i, d in enumerate(t.shape):
            if dims_equal(d, 1) and not dims_equal(gs[lead + i], 1):
                axes.append(lead + i)
        if axes:
            g = self.op("sum", g, dims=axes, keepdim=True)
        if lead:
            g = self.op("reshape", g, shape=list(t.shape))
        return g


def _where(node: Node) -> str:
    return f"%{node.output.name}" + (f", from {node.source}" if node.source else "")


def _active(nodes: Sequence[Node], wrt: Sequence[Value]) -> Set[int]:
    """Values that depend on a ``wrt`` value through differentiable ops (every floating point value
    a node defines, so the results of ``cond`` and ``loop`` count one by one)."""
    active = {id(v) for v in wrt}
    for n in nodes:
        if n.op in NONDIFF:
            continue
        if any(x is not None and id(x) in active for x in n.inputs):
            active.update(id(o) for o in n.outputs if o.type.dtype.is_floating_point)
    return active


def _needed(nodes: Sequence[Node], targets: Sequence[Value]) -> Set[int]:
    """Values that reach a differentiated output."""
    needed = {id(v) for v in targets}
    for n in reversed(nodes):
        if any(id(o) in needed for o in n.outputs):
            needed.update(id(x) for x in n.inputs if x is not None)
    return needed


def _on_path(n: Node, active: Set[int], needed: Set[int]) -> bool:
    return any(id(o) in active and id(o) in needed for o in n.outputs)


def _resolve(program: Program, wrt: Sequence[Union[str, Value]]) -> List[Value]:
    by_name = {v.name: v for v in program.inputs + program.params}
    out: List[Value] = []
    for w in wrt:
        v = by_name.get(w) if isinstance(w, str) else w
        if v is None or by_name.get(v.name) is not v:
            raise IRError(f"grad: {w!r} is not an input or param of program {program.name}. "
                          f"Inputs: {[x.name for x in program.inputs]}. Params: {[x.name for x in program.params]}")
        if not v.type.dtype.is_floating_point:
            raise IRError(f"grad: cannot differentiate with respect to {v.ref}: {v.type}, it is not floating point")
        if any(v is u for u in out):
            raise IRError(f"grad: {v.ref} is listed twice in wrt")
        out.append(v)
    if not out:
        raise IRError("grad: wrt is empty, name at least one input or param")
    return out


def grad(program: Program, wrt: Sequence[Union[str, Value]], of: Optional[Sequence[int]] = None,
         seed: str = "auto", name: Optional[str] = None, loop_memory: str = "auto") -> Program:
    """Transform ``program`` into a program that also returns gradients.

    wrt: names (or values) of the inputs and params to differentiate with respect to.
    of: indices of the outputs to differentiate. Default: every output.
    seed: where the cotangent of each differentiated output comes from.
      "auto" (default): a scalar (rank 0) output is seeded with 1, so its entries are plain
        gradients; any other output gets a new program input holding its cotangent.
      "input": every differentiated output gets a cotangent input.
      "ones": every differentiated output is seeded with ones (the gradient of its sum).

    The returned program has
      inputs:  the original inputs, then one cotangent input per seeded output, named ``d_<output>``;
      outputs: the original outputs, then one gradient per ``wrt`` entry, in order, each with the
               shape of its ``wrt`` value;
      params:  the original params (same names, data shared) plus constants the backward pass needs.

    loop_memory: where the backward of a ``loop`` gets each iteration's carried values (see
      autodiff_cf.py and docs/AUTODIFF.md): "store" keeps them in buffers (memory grows with the
      trip count), "recompute" reruns the forward body (constant memory, quadratic compute),
      "auto" (default) stores when the iteration count has a bound and recomputes otherwise.

    A ``wrt`` value that no differentiated output depends on gets a gradient of zeros.
    Raises UnsupportedOpError naming every op on the gradient path that has no gradient rule.
    """
    if seed not in ("auto", "input", "ones"):
        raise IRError(f"grad: seed must be 'auto', 'input' or 'ones', got {seed!r}")
    if loop_memory not in LOOP_MEMORY:
        raise IRError(f"grad: loop_memory must be one of {list(LOOP_MEMORY)}, got {loop_memory!r}")
    wrt_values = _resolve(program, wrt)
    indices = list(range(len(program.outputs))) if of is None else list(of)
    for i in indices:
        if not isinstance(i, int) or not 0 <= i < len(program.outputs):
            raise IRError(f"grad: of={list(of)} names output {i!r}, program {program.name} has "
                          f"{len(program.outputs)} outputs")
    targets = [program.outputs[i] for i in indices]
    for t in targets:
        if not t.type.dtype.is_floating_point:
            raise IRError(f"grad: output {t.ref}: {t.type} is not floating point and cannot be differentiated")

    e = _Emitter(program)
    e.loop_memory = loop_memory

    # 1. forward: copy the nodes, lowering composite ops that a gradient has to flow through.
    active = _active(program.nodes, wrt_values)
    needed = _needed(program.nodes, targets)
    for n in program.nodes:
        if n.op in CF_FORWARD and _on_path(n, active, needed):
            e.context = f"forward of {n.op} {_where(n)}"
            CF_FORWARD[n.op](e, n, active)  # control flow, autodiff_cf.py
        elif (n.op in LOWERINGS and id(n.output) in active and id(n.output) in needed
              and not _attention_grad.eligible(n)):   # plain self attention: kept whole, attention_grad
            e.context = f"lowered {n.op} {_where(n)}"
            try:
                LOWERINGS[n.op](e, n)
            except UnsupportedOpError as err:
                raise UnsupportedOpError(f"grad cannot differentiate {n.op} ({_where(n)}): {err}") from None
        else:
            e.nodes.append(n)
    forward = list(e.nodes)
    active = _active(forward, wrt_values)
    needed = _needed(forward, targets)

    # 2. every op on the gradient path must have a rule. Report all that do not, before any work.
    missing = [n for n in forward
               if _on_path(n, active, needed) and n.op not in GRAD_RULES and n.op not in CF_RULES]
    if missing:
        lines = [f"  no gradient rule for IR op {n.op!r} ({_where(n)})" for n in missing]
        raise UnsupportedOpError(f"grad cannot differentiate program {program.name} ({len(missing)} op(s) on the "
                                 "gradient path have no rule; nothing was skipped):\n" + "\n".join(lines)
                                 + f"\nOps with gradients: {sorted(set(GRAD_RULES) | set(LOWERINGS))}")

    # 3. seeds
    pending: Dict[int, List[Value]] = {}

    def add_cotangent(v: Value, g: Value, why: str) -> None:
        if g.type.rank != v.type.rank or not all(dims_equal(a, b) for a, b in zip(g.type.shape, v.type.shape)):
            raise IRError(f"internal error: {why} produced a cotangent of type {g.type} for {v.ref}: {v.type}")
        pending.setdefault(id(v), []).append(g)

    def total(v: Value) -> Optional[Value]:
        parts = pending.pop(id(v), None)
        if not parts:
            return None
        acc = parts[0]
        for p in parts[1:]:
            acc = e.op("add", acc, p)
        return acc

    for t in targets:
        e.context = f"seed of {t.ref}"
        if seed == "ones" or (seed == "auto" and t.type.rank == 0):
            g = e.const(1.0, t) if t.type.rank == 0 else e.op("full_like", t, value=1.0, dtype=None)
        else:
            g = Value(e.fresh(f"d_{t.name}"), t.type, "input")
            e.inputs.append(g)
        add_cotangent(t, g, "the seed")

    # 4. backward: reverse order, so every reader of a value has added its cotangent before the
    #    value's own node is differentiated.
    for n in reversed(forward):
        if n.op in CF_RULES:  # control flow defines several values, autodiff_cf.py
            if _on_path(n, active, needed):
                _control_flow_backward(e, n, total, add_cotangent, active)
            continue
        if id(n.output) not in active:
            continue
        e.context = f"grad of {n.op} {_where(n)}"
        g = total(n.output)
        if g is None:
            continue  # nothing downstream of this value reaches a differentiated output
        try:
            thunks = GRAD_RULES[n.op](e, n, g)
            if len(thunks) != len(n.inputs):
                raise IRError(f"internal error: the gradient rule of {n.op} returned {len(thunks)} entries "
                              f"for {len(n.inputs)} operands")
            for x, thunk in zip(n.inputs, thunks):
                if x is None or thunk is None or id(x) not in active:
                    continue
                add_cotangent(x, thunk(), f"the gradient rule of {n.op}")
        except UnsupportedOpError as err:
            raise UnsupportedOpError(f"grad cannot differentiate {n.op} ({_where(n)}): {err}") from None
        except ShapeError as err:
            raise IRError(f"internal error in the gradient rule of {n.op} ({_where(n)}): {err}") from None

    # 5. results
    grads: List[Value] = []
    for w in wrt_values:
        e.context = f"grad wrt {w.ref}"
        g = total(w)
        grads.append(g if g is not None else e.zeros(w.type.shape, w.type.dtype))

    out = Program(name or f"grad_{program.name}", e.inputs, e.params, e.param_data, e.nodes,
                  list(program.outputs) + grads, dict(program.symbols))
    out.verify()
    return out
