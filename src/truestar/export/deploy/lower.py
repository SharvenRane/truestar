"""What every deploy writer starts from: a verified Truestar program with its input independent
parts folded to constants.

A node whose inputs are all constants (params, or values folded before it) and whose output shape
is static is evaluated once with the IR reference interpreter in float64 and stored in its declared
dtype. That removes the mask, position and index arithmetic a model builds from constants (a causal
mask from ``ones(t, t).tril()``, positions from ``arange``), so writers only see the math that
depends on the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import sympy
import torch

from ...ir import reference
from ...ir.core import Node, Program, Value, is_const


class LoweringError(Exception):
    """An op or attribute this target has no exact lowering for (TS1412). Nothing is written."""

    code = "TS1412"

    def __init__(self, target: str, node: Optional[Node], reason: str):
        self.target, self.node, self.reason = target, node, reason
        where = ""
        if node is not None:
            where = f" (node %{node.output.name}, op {node.op}"
            if node.source:
                where += f", {node.source}"
            where += ")"
        super().__init__(f"{target}: {reason}{where}; nothing was written")


@dataclass
class Lowered:
    """The program a writer emits: inputs, outputs, nodes, and every constant (params and folded
    values) by value name, in its declared dtype."""

    name: str
    inputs: List[Value]
    outputs: List[Value]
    nodes: List[Node]
    consts: Dict[str, torch.Tensor]
    symbols: Dict[str, Tuple[int, Optional[int]]] = field(default_factory=dict)
    folded: int = 0
    meta: Dict[str, object] = field(default_factory=dict)  # what a writer learned (ExecuTorch plan)

    def is_const(self, v: Optional[Value]) -> bool:
        return v is not None and v.name in self.consts

    def static(self) -> bool:
        return all(is_const(d) for v in self.inputs for d in v.type.shape)


def shape_of(v: Value) -> Tuple:
    return tuple(v.type.shape)


def static_shape(target: str, v: Value, node: Optional[Node] = None) -> Tuple[int, ...]:
    if not all(is_const(d) for d in v.type.shape):
        raise LoweringError(target, node, f"%{v.name} has a symbolic shape {list(v.type.shape)}; this target "
                                          "needs static shapes (export with --static-batch)")
    return tuple(int(d) for d in v.type.shape)


def symbolic_entries(shape: Sequence) -> List[int]:
    return [i for i, d in enumerate(shape) if isinstance(d, sympy.Basic) and not d.is_Integer]


_NEVER_FOLD = {"random", "random_batched", "multinomial", "cond", "loop", "custom_vjp", "frozen"}


def prepare(prog: Program) -> Lowered:
    """Fold every input independent node with a static shape into a constant."""
    if prog.state or prog.key:
        raise LoweringError("export", None, "training programs with state or a random key are not deployable")
    consts: Dict[str, torch.Tensor] = {p.name: prog.param_data[p.name].detach().to(p.type.dtype).clone()
                                       for p in prog.params}
    keep: List[Node] = []
    folded = 0
    for n in prog.nodes:
        if n.extra_outputs:
            for r in n.attrs.values():
                if isinstance(r, Program):
                    raise LoweringError("export", n, "control flow (cond, loop) has no deploy lowering yet")
        foldable = (n.op not in _NEVER_FOLD and not n.extra_outputs and n.op in reference.IMPLS
                    and all(i is None or i.name in consts for i in n.inputs)
                    and all(is_const(d) for d in n.output.type.shape)
                    and not any(isinstance(a, sympy.Basic) for a in n.attrs.values()))
        if foldable:
            args = [None if i is None else consts[i.name].to(reference._compute_dtype(consts[i.name].dtype))
                    for i in n.inputs]
            out = reference.IMPLS[n.op](n.attrs, args, tuple(int(d) for d in n.output.type.shape))
            consts[n.output.name] = out.to(n.output.type.dtype)
            folded += 1
        else:
            keep.append(n)
    # drop constants nothing reads any more
    used = {i.name for n in keep for i in n.inputs if i is not None} | {v.name for v in prog.outputs}
    consts = {k: v for k, v in consts.items() if k in used}
    return Lowered(prog.name, list(prog.inputs), list(prog.outputs), keep, consts, dict(prog.symbols), folded)


def pair(v) -> Tuple[int, int]:
    if isinstance(v, (list, tuple)):
        return (int(v[0]), int(v[-1])) if len(v) else (0, 0)
    return int(v), int(v)
