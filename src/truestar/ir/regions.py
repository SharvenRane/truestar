"""Small utilities over IR programs used as regions (cond branches, loop bodies).

``inline`` copies a program's nodes into another program being built, ``prune`` drops what no
output needs and ``used_inputs`` says which inputs a program reads. Used by the autodiff of control
flow (autodiff_cf.py) and by the importer of torch.export control flow (scan bodies).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

from .core import Builder, IRError, Node, Program, Value


def inline(b: Builder, prog: Program, args: Sequence[Optional[Value]], source: str = "") -> List[Value]:
    """Copy ``prog``'s nodes into the program ``b`` is building, with its inputs bound to ``args``.

    An argument may be None when the program never reads that input; reading it is an internal
    error. Params are copied with their declared types (scalar constants keep their exact value).
    Regions held by nodes are shared, not copied: they are closed and never mutated.
    """
    if len(args) != len(prog.inputs):
        raise IRError(f"internal error: inlining {prog.name} with {len(args)} arguments for {len(prog.inputs)} inputs")
    env: Dict[int, Optional[Value]] = {id(v): a for v, a in zip(prog.inputs, args)}
    for p in prog.params:
        v = Value(b._fresh(p.name), p.type, "param")
        b.params.append(v)
        b.param_data[v.name] = prog.param_data[p.name]
        env[id(p)] = v

    def get(x: Optional[Value]) -> Optional[Value]:
        if x is None:
            return None
        got = env[id(x)]
        if got is None:
            raise IRError(f"internal error: inlined {prog.name} reads %{x.name}, which was not passed")
        return got

    for n in prog.nodes:
        outs = tuple(Value(b._fresh(o.name), o.type, "op") for o in n.outputs)
        b.nodes.append(Node(n.op, tuple(get(x) for x in n.inputs), dict(n.attrs), outs[0],
                            n.source or source, outs[1:]))
        for o, new in zip(n.outputs, outs):
            env[id(o)] = new
    return [get(v) for v in prog.outputs]


def prune(prog: Program) -> Program:
    """Drop the nodes and params no output depends on. Inputs are kept, so the interface is unchanged."""
    need = {id(v) for v in prog.outputs}
    kept: List[Node] = []
    for n in reversed(prog.nodes):
        if any(id(o) in need for o in n.outputs):
            kept.append(n)
            need.update(id(x) for x in n.inputs if x is not None)
    kept.reverse()
    params = [p for p in prog.params if id(p) in need]
    return Program(prog.name, list(prog.inputs), params, {p.name: prog.param_data[p.name] for p in params},
                   kept, list(prog.outputs), dict(prog.symbols))


def used_inputs(prog: Program) -> Set[int]:
    """Positions of the inputs some node or output reads."""
    read = {id(v) for v in prog.outputs}
    for n in prog.nodes:
        read.update(id(x) for x in n.inputs if x is not None)
    return {i for i, v in enumerate(prog.inputs) if id(v) in read}
