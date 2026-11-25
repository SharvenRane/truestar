"""Remedies for range sites, each applied to the IR and verified by emulation.

Three remedies, tried in this order:

* ``rescale``: an exact power of two reparameterization. The values that overflow (for example the
  residual stream of a Gemma style decoder) are carried scaled by s = 2^-k. Weights and constants
  that feed them are scaled at the edge (folded into the weights for deployment), consumers that
  are scale invariant (layer, RMS and group norm) take eps * s^2 instead, and any other consumer
  gets the value back as x * 2^k. A power of two scale is exact in binary floating point, so the
  float64 output of the rewritten program equals the original (checked, not assumed), costs
  nothing at run time and needs no float32 op on the device.
* ``fp32``: keep the site and every consumer that still sees an out of range value in float32
  (for a hidden intermediate: run the composite op fused, with float32 internals).
* ``clamp``: clip the op's output to +-max * 2^-headroom (the Hub fix for Gemma 3 was 18 Clip nodes
  at +-32752). It changes values, so it is accepted only if verification passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from ..ir.core import Node, Program, TensorType, Value

# ops that are homogeneous of degree one in every float operand together: f(s a, s b) = s f(a, b)
_COSCALED = {"add", "sub", "maximum", "minimum", "cat", "slice_scatter", "select_scatter"}
# homogeneous of degree one in operand 0 only (the others are indices or shape attributes)
_UNARY_HOMOG = {"neg", "relu", "reshape", "permute", "expand", "select", "slice", "flip", "index",
                "index_select", "gather", "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d",
                "adaptive_max_pool2d", "mean", "sum", "cumsum", "amax", "amin", "abs", "triu", "tril"}
# degree one in exactly one multiplicative operand
_MULTIPLICATIVE = {"mul", "div", "matmul", "linear", "conv2d"}
# scale invariant in operand 0 once eps is multiplied by s^2
_INVARIANT = {"layer_norm", "rms_norm", "group_norm"}


@dataclass
class Region:
    """Values carried scaled by 2^-k, and how every edge into and out of them is treated."""

    k: int
    values: Set[int]                                  # id(Value) carried scaled
    scaled_operands: Dict[int, Set[int]]              # id(node) -> operand indices that carry s
    entries: List[Tuple[str, str]] = field(default_factory=list)   # (value, consumer) scaled at the edge
    exits: List[Tuple[str, str]] = field(default_factory=list)     # (value, consumer) given back x 2^k
    norm_exits: List[str] = field(default_factory=list)            # invariant norms with eps * s^2


class RegionError(Exception):
    pass


def _float(v: Optional[Value]) -> bool:
    return v is not None and v.type.dtype.is_floating_point


def find_region(prog: Program, seeds: Dict[int, float], peaks: Dict[int, float], limit: float) -> Region:
    """Grow the set of values to rescale from ``seeds`` (id(Value) -> required peak reduction factor).

    ``peaks`` holds the observed float64 peak of every value; ``limit`` is the largest magnitude
    allowed after rescaling. Raises RegionError when no exact rescale exists (for example a value
    that overflows is produced by a nonlinear op, or a region value is multiplied by itself)."""
    producer: Dict[int, Node] = {}
    consumers: Dict[int, List[Tuple[Node, int]]] = {}
    for n in prog.nodes:
        for o in n.outputs:
            producer[id(o)] = n
        for i, x in enumerate(n.inputs):
            if x is not None:
                consumers.setdefault(id(x), []).append((n, i))
    names = {id(v): v for v in list(prog.inputs) + list(prog.params)}
    for n in prog.nodes:
        for o in n.outputs:
            names[id(o)] = o
    outputs = {id(v) for v in prog.outputs}

    need = max(seeds.values()) if seeds else 1.0
    S: Set[int] = set()
    scaled_ops: Dict[int, Set[int]] = {}
    work = list(seeds)
    big = lambda vid: peaks.get(vid, math.inf) > limit  # noqa: E731

    def add(vid: int) -> None:
        if vid not in S:
            S.add(vid)
            work.append(vid)

    while work:
        vid = work.pop()
        if vid not in S:
            S.add(vid)
        n = producer.get(vid)
        if n is None:
            raise RegionError(f"{names[vid].ref} is a program input or weight that is already out of range")
        if id(n) not in scaled_ops:
            scaled_ops[id(n)] = _choose_operands(n, S, big)
            for i in scaled_ops[id(n)]:
                x = n.inputs[i]
                if x is not None and x.kind == "op" and big(id(x)):
                    add(id(x))
        if vid in outputs and peaks.get(vid, 0.0) > limit:
            raise RegionError(f"program output {names[vid].ref} itself is out of range")
        for c, i in consumers.get(vid, []):
            if c.op in _COSCALED or (c.op in _UNARY_HOMOG and i == 0):
                add(id(c.output))
                ops = scaled_ops.setdefault(id(c), set())
                ops.update(_coscaled_indices(c) if c.op in _COSCALED else {0})
                for j in ops:
                    x = c.inputs[j]
                    if x is not None and x.kind == "op" and big(id(x)):
                        add(id(x))
            elif c.op in _MULTIPLICATIVE and not (c.op == "div" and i == 1):
                prev = scaled_ops.get(id(c))
                if prev is not None and i not in prev and any(c.inputs[j] is not None and c.inputs[j].kind != "param"
                                                             for j in prev):
                    raise RegionError(f"{c.op} {c.output.ref} multiplies two rescaled values")
                scaled_ops[id(c)] = {i} | ({2} if c.op in ("linear", "conv2d") and c.inputs[2] is not None else set())
                add(id(c.output))
            elif c.op in _INVARIANT and i == 0:
                pass
            elif c.op == "where" and i in (1, 2):
                add(id(c.output))
                scaled_ops.setdefault(id(c), set()).update({1, 2})
                for j in (1, 2):
                    x = c.inputs[j]
                    if x is not None and x.kind == "op" and big(id(x)):
                        add(id(x))
            else:
                if peaks.get(vid, math.inf) > limit:
                    raise RegionError(f"{names[vid].ref} feeds {c.op} {c.output.ref}, which is not scale "
                                      f"equivariant, while it is out of range")
    # consistency: every scaled node's operands that are in S must be among its scaled operands
    for n in prog.nodes:
        if id(n.output) in S:
            ops = scaled_ops.get(id(n), set())
            for i, x in enumerate(n.inputs):
                if x is not None and id(x) in S and i not in ops:
                    raise RegionError(f"{n.op} {n.output.ref} mixes a rescaled and an unscaled operand")
            if n.op in ("mul", "matmul") and sum(1 for i in ops if n.inputs[i] is not None and i < 2) > 1:
                raise RegionError(f"{n.op} {n.output.ref} would scale both factors")
    need = max([need] + [peaks.get(v, 0.0) / limit for v in S])
    k = max(0, math.ceil(math.log2(need))) if need > 1 else 0
    if k == 0:
        raise RegionError("nothing to rescale")
    return Region(k, S, scaled_ops)


def _coscaled_indices(n: Node) -> Set[int]:
    return {i for i, x in enumerate(n.inputs) if _float(x)}


def _choose_operands(n: Node, S: Set[int], big) -> Set[int]:
    """Operands of a node whose output must carry the scale."""
    if n.op in _COSCALED:
        return _coscaled_indices(n)
    if n.op == "where":
        return {1, 2}
    if n.op in _UNARY_HOMOG:
        return {0}
    if n.op in _INVARIANT:
        # a norm's output is linear in its weight and bias: scale those (folded at deployment)
        ws = {i for i in (1, 2) if i < len(n.inputs) and n.inputs[i] is not None}
        if 1 not in ws:
            raise RegionError(f"{n.op} {n.output.ref} has no weight to carry the scale")
        return ws
    if n.op in _MULTIPLICATIVE:
        ins = n.inputs
        inside = [i for i, x in enumerate(ins[:2]) if x is not None and id(x) in S]
        if len(inside) > 1:
            raise RegionError(f"{n.op} {n.output.ref} multiplies two rescaled values")
        if inside:
            i = inside[0]
        elif n.op in ("linear", "conv2d"):
            i = 1 if ins[1] is not None and ins[1].kind == "param" else 0
        elif n.op == "div":
            i = 0
        else:
            params = [j for j, x in enumerate(ins[:2]) if x is not None and x.kind == "param"]
            i = params[0] if params else 0
        bias = {2} if n.op in ("linear", "conv2d") and ins[2] is not None else set()
        return {i} | bias
    raise RegionError(f"{n.op} {n.output.ref} is out of range and is not a scale equivariant op "
                      f"(its input cannot carry the scale)")


def apply_rescale(prog: Program, region: Region) -> Program:
    """A new program carrying the region's values scaled by 2^-k. Weights stay shared unless scaled."""
    s = 2.0 ** -region.k
    params = list(prog.params)
    param_data = dict(prog.param_data)
    names = {v.name for v in list(prog.inputs) + list(prog.params)}
    for n in prog.nodes:
        for o in n.outputs:
            names.add(o.name)
    new_nodes: List[Node] = []
    remap: Dict[int, Value] = {}
    const_cache: Dict[Tuple[float, torch.dtype], Value] = {}
    scaled_param_cache: Dict[int, Value] = {}

    def fresh(base: str) -> str:
        i = 0
        name = base
        while name in names:
            i += 1
            name = f"{base}_{i}"
        names.add(name)
        return name

    def const(f: float, dtype: torch.dtype) -> Value:
        key = (f, dtype)
        if key not in const_cache:
            v = Value(fresh(f"range_scale_{'down' if f < 1 else 'up'}"), TensorType(dtype, ()), "param")
            params.append(v)
            param_data[v.name] = torch.tensor(f, dtype=dtype)
            const_cache[key] = v
        return const_cache[key]

    def scaled(x: Value, consumer: Node) -> Value:
        """x times s, at the edge into a region node."""
        if x.kind == "param":
            if id(x) not in scaled_param_cache:
                v = Value(fresh(x.name + "_rescaled"), x.type, "param")
                params.append(v)
                param_data[v.name] = prog.param_data[x.name] * s
                scaled_param_cache[id(x)] = v
            region.entries.append((x.ref, consumer.output.ref))
            return scaled_param_cache[id(x)]
        cur = remap.get(id(x), x)
        out = Value(fresh(x.name + "_rescaled"), cur.type, "op")
        new_nodes.append(Node("mul", (cur, const(s, cur.type.dtype)), {}, out,
                              f"truestar range: x 2^-{region.k} into the rescaled region"))
        region.entries.append((x.ref, consumer.output.ref))
        return out

    def unscaled(x: Value, consumer: Node) -> Value:
        cur = remap.get(id(x), x)
        out = Value(fresh(x.name + "_unscaled"), cur.type, "op")
        new_nodes.append(Node("mul", (cur, const(1.0 / s, cur.type.dtype)), {}, out,
                              f"truestar range: x 2^{region.k} out of the rescaled region"))
        region.exits.append((x.ref, consumer.output.ref))
        return out

    for n in prog.nodes:
        in_region = id(n.output) in region.values
        ops = region.scaled_operands.get(id(n), set()) if in_region else set()
        new_inputs = []
        attrs = dict(n.attrs)
        for i, x in enumerate(n.inputs):
            if x is None:
                new_inputs.append(None)
                continue
            xs = id(x) in region.values
            if in_region and i in ops:
                new_inputs.append(remap.get(id(x), x) if xs else scaled(x, n))
            elif xs and n.op in _INVARIANT and i == 0:
                new_inputs.append(remap.get(id(x), x))
                eps = attrs.get("eps")
                if eps is None:
                    eps = torch.finfo(torch.float32).eps
                attrs["eps"] = eps * s * s
                region.norm_exits.append(n.output.ref)
            elif xs:
                new_inputs.append(unscaled(x, n))
            else:
                new_inputs.append(remap.get(id(x), x))
        node = Node(n.op, tuple(new_inputs), attrs, n.output, n.source, n.extra_outputs)
        new_nodes.append(node)
    outs = []
    for v in prog.outputs:
        if id(v) in region.values:
            o = Value(fresh(v.name + "_unscaled"), v.type, "op")
            new_nodes.append(Node("mul", (v, const(1.0 / s, v.type.dtype)), {}, o,
                                  "truestar range: program output out of the rescaled region"))
            outs.append(o)
        else:
            outs.append(v)
    out = Program(prog.name, list(prog.inputs), params, param_data, new_nodes, outs, dict(prog.symbols))
    out.verify()
    return out


def fp32_region(prog: Program, site_nodes: Sequence[Node], peaks: Dict[int, float], limit: float) -> Set[int]:
    """Node ids to keep in float32: each site plus, transitively, every consumer of a value that is
    still out of range (it would overflow again when cast back to the target format)."""
    consumers: Dict[int, List[Node]] = {}
    for n in prog.nodes:
        for x in n.inputs:
            if x is not None:
                consumers.setdefault(id(x), []).append(n)
    keep: Set[int] = set()
    work = list(site_nodes)
    while work:
        n = work.pop()
        if id(n) in keep:
            continue
        keep.add(id(n))
        for o in n.outputs:
            if peaks.get(id(o), 0.0) > limit:
                work.extend(consumers.get(id(o), []))
    return keep
