"""Forward mode autodiff and the Jacobian family as program transformations.

* ``jvp(program, wrt)``: a program that also takes one tangent per ``wrt`` value and returns the
  Jacobian vector product ``J t`` of each output. Emitted as ordinary IR nodes beside the primal
  nodes, so no tape, no dual tensor type, one program for every batch size.
* ``jacfwd(program, wrt)``: the full Jacobian, as ``vmap`` of ``jvp`` over the columns of an
  identity matrix built inside the program (``arange`` and ``eq``, so symbolic sizes work).
* ``jacrev(program, wrt)``: the full Jacobian as ``vmap`` of the reverse mode ``grad`` over the rows.
* ``hessian(program, wrt)``: forward over reverse, ``jacfwd(grad(program))``: one program.
* ``linearize(program, wrt)``: the ``jvp`` program split in two, a primal program that returns the
  outputs and the values the tangent code reads (residuals), and a linear program from residuals
  and tangents to output tangents, so repeated Jacobian vector products skip the primal work.

Every rule here is written independently of the reverse mode rules in ``grad_rules.py``, so the
two can check each other: tests/test_forward.py checks every rule against finite differences and,
for every op that has both, the dot product identity ``<J t, c> == <t, J^T c>`` against the
reverse mode rule. Loops carry their tangents alongside their values, so forward mode through a
``loop`` stores nothing per iteration.

See docs/TRANSFORMS.md.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch

from .core import IRError, Node, Program, TensorType, UnsupportedOpError, Value, dims_equal
from .grad_rules import LOWERINGS, NONDIFF
from .transform_util import Emitter, as_program, numel, prune

__all__ = ["jvp", "jacfwd", "jacrev", "hessian", "linearize", "JVP_RULES", "jvp_rule_table"]

Rule = Callable[[Emitter, Node, List[Optional[Value]]], List[Optional[Value]]]
JVP_RULES: Dict[str, Rule] = {}
_CONTROL_FLOW: Dict[str, Callable[..., None]] = {}


def jvp_rule(*names: str):
    def deco(fn):
        for n in names:
            JVP_RULES[n] = fn
        return fn
    return deco


def jvp_rule_table() -> Dict[str, str]:
    """Every op forward mode knows: "rule", "lowered", "control flow" or "no tangent"."""
    table = {op: "rule" for op in JVP_RULES}
    table.update({op: "lowered" for op in LOWERINGS})
    table.update({op: "control flow" for op in _CONTROL_FLOW})
    table.update({op: "no tangent" for op in NONDIFF})
    return dict(sorted(table.items()))


def _where_from(n: Node) -> str:
    return f"%{n.output.name}" + (f", from {n.source}" if n.source else "")


def _float(v: Value) -> bool:
    return v.type.dtype.is_floating_point


# ----------------------------------------------------------------------------- the sweep


def _forward(e: Emitter, nodes: Sequence[Node], tan: Dict[int, Value]) -> None:
    """Append ``nodes`` and, behind each node that reads a value with a tangent, the nodes that
    compute the tangents of its results. ``tan`` maps a value to its tangent (absent: zero)."""
    for n in nodes:
        ts = [None if x is None else tan.get(id(x)) for x in n.inputs]
        if all(t is None for t in ts) or n.op in NONDIFF or not any(_float(o) for o in n.outputs):
            e.append(n)
            continue
        e.context = f"jvp of {n.op} {_where_from(n)}"
        if n.op in _CONTROL_FLOW:
            _CONTROL_FLOW[n.op](e, n, ts, tan)
            continue
        if n.op in JVP_RULES:
            e.append(n)
            outs = JVP_RULES[n.op](e, n, ts)
        elif n.op in LOWERINGS:
            tmp = Emitter(set(e.names), e.prefix)
            tmp.context = f"lowered {n.op} {_where_from(n)}"
            LOWERINGS[n.op](tmp, n)
            e.names |= tmp.names
            e.params.extend(tmp.params)
            e.param_data.update(tmp.param_data)
            _forward(e, tmp.nodes, tan)
            continue
        else:
            raise UnsupportedOpError(f"jvp cannot differentiate {n.op} ({_where_from(n)}): no forward mode rule. "
                                     f"Ops with rules: {sorted(set(JVP_RULES) | set(LOWERINGS) | set(_CONTROL_FLOW))}")
        if len(outs) != len(n.outputs):
            raise IRError(f"internal error: the jvp rule of {n.op} returned {len(outs)} tangents for "
                          f"{len(n.outputs)} values")
        for o, t in zip(n.outputs, outs):
            if t is None or not _float(o):
                continue
            if not t.type.same_as(o.type):
                raise IRError(f"internal error: the jvp rule of {n.op} ({_where_from(n)}) gives a tangent of "
                              f"type {t.type} for {o.ref}: {o.type}")
            tan[id(o)] = t


def _missing(nodes: Sequence[Node], active: Set[int]) -> List[Node]:
    out = []
    for n in nodes:
        if not any(x is not None and id(x) in active for x in n.inputs):
            continue
        if n.op in NONDIFF or not any(_float(o) for o in n.outputs):
            continue
        if n.op not in JVP_RULES and n.op not in LOWERINGS and n.op not in _CONTROL_FLOW:
            out.append(n)
        active.update(id(o) for o in n.outputs if _float(o))
    return out


def _resolve(prog: Program, wrt: Sequence[Union[str, Value]], what: str) -> List[Value]:
    by_name = {v.name: v for v in prog.inputs + prog.params}
    if isinstance(wrt, (str, Value)):
        wrt = [wrt]
    out: List[Value] = []
    for w in wrt:
        v = by_name.get(w) if isinstance(w, str) else w
        if v is None or by_name.get(v.name) is not v:
            raise IRError(f"{what}: {w!r} is not an input or param of program {prog.name}. "
                          f"Inputs: {[x.name for x in prog.inputs]}. Params: {[x.name for x in prog.params]}")
        if not _float(v):
            raise IRError(f"{what}: cannot differentiate with respect to {v.ref}: {v.type}, it is not floating point")
        if any(v is u for u in out):
            raise IRError(f"{what}: {v.ref} is listed twice in wrt")
        out.append(v)
    if not out:
        raise IRError(f"{what}: wrt is empty, name at least one input or param")
    return out


def _outputs(prog: Program, of: Optional[Sequence[int]], what: str) -> List[int]:
    if of is None:
        return [i for i, v in enumerate(prog.outputs) if _float(v)]
    idx = [of] if isinstance(of, int) else list(of)
    for i in idx:
        if not isinstance(i, int) or not 0 <= i < len(prog.outputs):
            raise IRError(f"{what}: of={idx} names output {i!r}, program {prog.name} has {len(prog.outputs)} outputs")
        if not _float(prog.outputs[i]):
            raise IRError(f"{what}: output {prog.outputs[i].ref}: {prog.outputs[i].type} is not floating point")
    return idx


def jvp(program: Any, wrt: Sequence[Union[str, Value]], of: Optional[Sequence[int]] = None,
        name: Optional[str] = None) -> Program:
    """Forward mode: a program that also returns Jacobian vector products.

    wrt: names (or values) of the inputs and params that get a tangent.
    of: indices of the outputs whose tangents are returned. Default: every floating point output.

    The returned program has
      inputs:  the original inputs, then one tangent input per ``wrt`` entry, named ``t_<name>``,
               with the type of its value;
      outputs: the original outputs, then the tangent of each ``of`` output, in order;
      params:  the original params plus constants the tangent code needs.

    An output that does not depend on any ``wrt`` value gets a tangent of zeros. Raises
    UnsupportedOpError naming every op on the tangent path without a forward mode rule.
    """
    prog = as_program(program)
    wrt_values = _resolve(prog, wrt, "jvp")
    idx = _outputs(prog, of, "jvp")
    missing = _missing(prog.nodes, {id(v) for v in wrt_values})
    if missing:
        lines = [f"  no forward mode rule for IR op {n.op!r} ({_where_from(n)})" for n in missing]
        raise UnsupportedOpError(f"jvp cannot differentiate program {prog.name} ({len(missing)} op(s) on the tangent "
                                 "path have no rule; nothing was skipped):\n" + "\n".join(lines)
                                 + f"\nOps with forward mode rules: {sorted(jvp_rule_table())}")
    e = Emitter(Emitter.names_of(prog), "t_", prog.params, prog.param_data)
    e.inputs = list(prog.inputs)
    tan: Dict[int, Value] = {}
    for w in wrt_values:
        tan[id(w)] = e.new_input(f"t_{w.name}", w.type)
    _forward(e, prog.nodes, tan)
    e.context = "zero tangent"
    outs = []
    for i in idx:
        v = prog.outputs[i]
        t = tan.get(id(v))
        outs.append(t if t is not None else e.zeros(v.type.shape, v.type.dtype))
    out = Program(name or f"jvp_{prog.name}", e.inputs, e.params, e.param_data, e.nodes,
                  list(prog.outputs) + outs, dict(prog.symbols))
    return prune(out)


# ----------------------------------------------------------------------------- helpers for rules


def _fit(e: Emitter, t: Value, out: Value) -> Value:
    """A tangent broadcast and cast to the type of ``out``."""
    if t.type.dtype != out.type.dtype:
        t = e.op("cast", t, dtype=out.type.dtype)
    if t.type.rank != out.type.rank or not all(dims_equal(a, b) for a, b in zip(t.type.shape, out.type.shape)):
        t = e.op("expand", t, shape=list(out.type.shape))
    return t


def _total(e: Emitter, parts: Sequence[Optional[Value]], out: Value) -> Optional[Value]:
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    acc = _fit(e, parts[0], out)
    for p in parts[1:]:
        acc = e.op("add", acc, _fit(e, p, out))
    return acc


def _zero(e: Emitter, like: Value) -> Value:
    return e.const(0.0, like)


def _replay(e: Emitter, n: Node, ts: List[Optional[Value]], linear: Sequence[int], **override) -> Value:
    """For an op linear in the operands at ``linear`` (jointly): the op applied to the tangents,
    zeros for the linear operands without one, the primal for the others (indices, conditions)."""
    args = []
    for i, (x, t) in enumerate(zip(n.inputs, ts)):
        if x is None:
            args.append(None)
        elif i in linear:
            args.append(t if t is not None else e.zeros_like(x))
        else:
            if t is not None:
                raise IRError(f"internal error: {n.op} is not linear in operand {i}, which has a tangent")
            args.append(x)
    attrs = dict(n.attrs)
    attrs.update(override)
    return e.op(n.op, *args, **attrs)


# ----------------------------------------------------------------------------- elementwise


@jvp_rule("neg")
def _neg(e, n, ts):
    return [e.op("neg", ts[0])]


@jvp_rule("exp")
def _exp(e, n, ts):
    return [e.op("mul", ts[0], n.output)]


@jvp_rule("sqrt")
def _sqrt(e, n, ts):
    """d sqrt(x) = t / (2 sqrt(x))."""
    return [e.op("div", e.scale(ts[0], 0.5), n.output)]


@jvp_rule("rsqrt")
def _rsqrt(e, n, ts):
    """d x^(-1/2) = -t x^(-3/2) / 2 = -t y^3 / 2."""
    y = n.output
    return [e.op("mul", e.scale(ts[0], -0.5), e.op("mul", e.op("mul", y, y), y))]


@jvp_rule("tanh")
def _tanh(e, n, ts):
    y = n.output
    return [e.op("mul", ts[0], e.op("sub", e.const(1.0, y), e.op("mul", y, y)))]


@jvp_rule("sigmoid")
def _sigmoid(e, n, ts):
    y = n.output
    return [e.op("mul", ts[0], e.op("sub", y, e.op("mul", y, y)))]


@jvp_rule("erf")
def _erf(e, n, ts):
    (x,) = n.inputs
    return [e.op("mul", ts[0], e.scale(e.op("exp", e.op("neg", e.op("mul", x, x))), 2.0 / math.sqrt(math.pi)))]


@jvp_rule("expm1")
def _expm1(e, n, ts):
    (x,) = n.inputs
    return [e.op("mul", ts[0], e.op("exp", x))]


@jvp_rule("relu")
def _relu(e, n, ts):
    (x,) = n.inputs
    return [e.op("where", e.op("gt", x, _zero(e, x)), ts[0], _zero(e, ts[0]))]


@jvp_rule("silu")
def _silu(e, n, ts):
    """d (x s(x)) = (s + x s (1 - s)) t."""
    (x,) = n.inputs
    s = e.op("sigmoid", x)
    xs = e.op("mul", x, s)
    return [e.op("mul", ts[0], e.op("add", s, e.op("sub", xs, e.op("mul", xs, s))))]


@jvp_rule("gelu")
def _gelu(e, n, ts):
    (x,) = n.inputs
    t = ts[0]
    one = e.const(1.0, x)
    if n.attrs.get("approximate", "none") == "tanh":
        # y = x (1 + tanh u) / 2, u = c (x + k x^3): dy = [(1 + th) / 2 + x (1 - th^2) c (1 + 3 k x^2) / 2] t
        c, k = math.sqrt(2.0 / math.pi), 0.044715
        x2 = e.op("mul", x, x)
        th = e.op("tanh", e.scale(e.op("add", x, e.scale(e.op("mul", x2, x), k)), c))
        du = e.scale(e.op("add", one, e.scale(x2, 3.0 * k)), c)
        sech2 = e.op("sub", one, e.op("mul", th, th))
        d = e.scale(e.op("add", e.op("add", one, th), e.op("mul", e.op("mul", x, sech2), du)), 0.5)
        return [e.op("mul", t, d)]
    # y = x Phi(x): dy = (Phi(x) + x phi(x)) t
    phi_cdf = e.scale(e.op("add", one, e.op("erf", e.scale(x, 1.0 / math.sqrt(2.0)))), 0.5)
    pdf = e.scale(e.op("exp", e.scale(e.op("mul", x, x), -0.5)), 1.0 / math.sqrt(2.0 * math.pi))
    return [e.op("mul", t, e.op("add", phi_cdf, e.op("mul", x, pdf)))]


@jvp_rule("hardsigmoid")
def _hardsigmoid(e, n, ts):
    (x,) = n.inputs
    inside = e.op("logical_and", e.op("gt", x, e.const(-3.0, x)), e.op("lt", x, e.const(3.0, x)))
    return [e.op("where", inside, e.scale(ts[0], 1.0 / 6.0), _zero(e, ts[0]))]


@jvp_rule("hardswish")
def _hardswish(e, n, ts):
    """Slope 0 below -3, x / 3 + 1 / 2 on [-3, 3], 1 above (PyTorch's conventions at the kinks)."""
    (x,) = n.inputs
    t = ts[0]
    mid = e.op("mul", t, e.op("add", e.scale(x, 1.0 / 3.0), e.const(0.5, x)))
    upper = e.op("where", e.op("le", x, e.const(3.0, x)), mid, t)
    return [e.op("where", e.op("lt", x, e.const(-3.0, x)), _zero(e, t), upper)]


@jvp_rule("clamp")
def _clamp(e, n, ts):
    """Slope 1 where min <= x <= max (bounds included, as torch.clamp before 2.14), else 0; with
    ``strict`` (hardtanh, relu6, torch 2.14 and later clamp) only strictly inside."""
    (x,) = n.inputs
    lo, hi = n.attrs.get("min"), n.attrs.get("max")
    strict = bool(n.attrs.get("strict", False))
    keep = None
    if lo is not None:
        keep = e.op("gt" if strict else "ge", x, e.const(float(lo), x))
    if hi is not None:
        below = e.op("lt" if strict else "le", x, e.const(float(hi), x))
        keep = below if keep is None else e.op("logical_and", keep, below)
    return [ts[0] if keep is None else e.op("where", keep, ts[0], _zero(e, ts[0]))]


@jvp_rule("log")
def _log(e, n, ts):
    return [_fit(e, e.op("div", ts[0], n.inputs[0]), n.output)]


@jvp_rule("sin")
def _sin(e, n, ts):
    return [_fit(e, e.op("mul", ts[0], e.op("cos", n.inputs[0])), n.output)]


@jvp_rule("cos")
def _cos(e, n, ts):
    return [_fit(e, e.op("neg", e.op("mul", ts[0], e.op("sin", n.inputs[0]))), n.output)]


@jvp_rule("reciprocal")
def _reciprocal(e, n, ts):
    y = n.output
    return [e.op("neg", e.op("mul", ts[0], e.op("mul", y, y)))]


@jvp_rule("abs")
def _abs(e, n, ts):
    """Slope sign(x), 0 at 0 (as PyTorch)."""
    (x,) = n.inputs
    t = ts[0]
    neg = e.op("where", e.op("lt", x, _zero(e, x)), e.op("neg", t), _zero(e, t))
    return [e.op("where", e.op("gt", x, _zero(e, x)), t, neg)]


@jvp_rule("cast")
def _cast(e, n, ts):
    return [e.op("cast", ts[0], dtype=n.attrs["dtype"])]


@jvp_rule("softmax")
def _softmax(e, n, ts):
    """dy = y (t - sum(t y)) along the softmax dim."""
    y = n.output
    d = n.attrs["dim"] % max(y.type.rank, 1)
    inner = e.op("sum", e.op("mul", ts[0], y), dims=[d], keepdim=True)
    return [e.op("mul", y, e.op("sub", ts[0], inner))]


@jvp_rule("log_softmax")
def _log_softmax(e, n, ts):
    """y = x - logsumexp(x): dy = t - sum(softmax(x) t) along the dim (op of truestar.train)."""
    d = n.attrs["dim"] % max(n.output.type.rank, 1)
    p = e.op("exp", n.output)
    return [e.op("sub", ts[0], e.op("sum", e.op("mul", p, ts[0]), dims=[d], keepdim=True))]


# ----------------------------------------------------------------------------- binary


@jvp_rule("add")
def _add(e, n, ts):
    return [_total(e, ts, n.output)]


@jvp_rule("sub")
def _sub(e, n, ts):
    ta, tb = ts
    if ta is not None and tb is not None:
        return [_fit(e, e.op("sub", ta, tb), n.output)]
    return [_fit(e, ta if ta is not None else e.op("neg", tb), n.output)]


@jvp_rule("mul")
def _mul(e, n, ts):
    a, b = n.inputs
    ta, tb = ts
    return [_total(e, [None if ta is None else e.op("mul", ta, b), None if tb is None else e.op("mul", a, tb)],
                   n.output)]


@jvp_rule("div")
def _div(e, n, ts):
    """d (a / b) = ta / b - y tb / b."""
    a, b = n.inputs
    ta, tb = ts
    y = n.output
    parts = [None if ta is None else e.op("div", ta, b),
             None if tb is None else e.op("neg", e.op("div", e.op("mul", y, tb), b))]
    return [_total(e, parts, y)]


@jvp_rule("pow")
def _pow(e, n, ts):
    """d a^b = b a^(b-1) ta + a^b log(a) tb; the first term is 0 where b == 0 (as PyTorch)."""
    a, b = n.inputs
    ta, tb = ts
    y = n.output
    parts = []
    if ta is not None:
        slope = e.op("mul", b, e.op("pow", a, e.op("sub", b, e.const(1.0, y))))
        term = e.op("mul", ta, slope)
        parts.append(e.op("where", e.op("eq", b, e.const(0.0, y)), _zero(e, term), term))
    if tb is not None:
        parts.append(e.op("mul", tb, e.op("mul", y, e.op("log", a))))
    return [_total(e, parts, y)]


@jvp_rule("maximum", "minimum")
def _maximum(e, n, ts):
    """The tangent of the larger (smaller) operand; at a tie the mean of both (as PyTorch)."""
    a, b = n.inputs
    y = n.output
    ta = _fit(e, ts[0], y) if ts[0] is not None else _fit(e, _zero(e, y), y)
    tb = _fit(e, ts[1], y) if ts[1] is not None else _fit(e, _zero(e, y), y)
    first = e.op("gt" if n.op == "maximum" else "lt", a, b)
    tie = e.op("eq", a, b)
    half = e.scale(e.op("add", ta, tb), 0.5)
    return [e.op("where", first, ta, e.op("where", tie, half, tb))]


@jvp_rule("fmod", "remainder")
def _fmod(e, n, ts):
    """y = a - q(a / b) b with q = trunc (fmod) or floor (remainder), piecewise constant: dy = ta - q tb
    (added 2026-11-24 with the op coverage work)."""
    a, b = n.inputs
    ta, tb = ts
    y = n.output
    parts = [ta]
    if tb is not None:
        ratio = e.op("div", a, b)
        q = e.op("sub", ratio, e.op(n.op, ratio, e.const(1.0, ratio)))
        parts.append(e.op("neg", e.op("mul", q, tb)))
    return [_total(e, parts, y)]


@jvp_rule("where")
def _where(e, n, ts):
    c, a, b = n.inputs
    _, ta, tb = ts
    y = n.output
    return [_fit(e, e.op("where", c, _zero(e, y) if ta is None else ta, _zero(e, y) if tb is None else tb), y)]


# ----------------------------------------------------------------------------- linear algebra


@jvp_rule("matmul")
def _matmul(e, n, ts):
    a, b = n.inputs
    ta, tb = ts
    return [_total(e, [None if ta is None else e.op("matmul", ta, b), None if tb is None else e.op("matmul", a, tb)],
                   n.output)]


@jvp_rule("linear")
def _linear(e, n, ts):
    x, w, b = n.inputs
    tx, tw, tb = ts
    parts = [None if tx is None else e.op("linear", tx, w, None),
             None if tw is None else e.op("linear", x, tw, None), tb]
    return [_total(e, parts, n.output)]


@jvp_rule("conv2d")
def _conv2d(e, n, ts):
    x, w, b = n.inputs
    tx, tw, tb = ts
    parts = [None if tx is None else e.op("conv2d", tx, w, None, **n.attrs),
             None if tw is None else e.op("conv2d", x, tw, None, **n.attrs),
             None if tb is None else e.op("reshape", tb, shape=[1, tb.type.shape[0], 1, 1])]
    return [_total(e, parts, n.output)]


@jvp_rule("conv2d_input_grad", "conv2d_weight_grad")
def _conv2d_grads(e, n, ts):
    """Both conv gradient ops are bilinear in their two operands."""
    a, b = n.inputs
    ta, tb = ts
    parts = [None if ta is None else e.op(n.op, ta, b, **n.attrs),
             None if tb is None else e.op(n.op, a, tb, **n.attrs)]
    return [_total(e, parts, n.output)]


@jvp_rule("max_pool2d_grad")
def _max_pool2d_grad(e, n, ts):
    """Linear in g; x only chooses which tap each window sends to, so its tangent adds nothing."""
    return [_replay(e, n, ts, [1])]


# ----------------------------------------------------------------------------- linear structure ops


def _linear_rule(positions: Sequence[int], **override):
    def rule(e, n, ts):
        return [_replay(e, n, ts, positions, **override)]
    return rule


for _op in ("sum", "mean", "permute", "reshape", "expand", "select", "slice", "flip", "triu", "tril", "cumsum",
            "avg_pool2d", "adaptive_avg_pool2d"):
    JVP_RULES[_op] = _linear_rule([0])
for _op in ("select_dynamic", "index_select", "gather", "index"):
    JVP_RULES[_op] = _linear_rule([0])
JVP_RULES["pad"] = _linear_rule([0], value=0.0)
JVP_RULES["all_reduce"] = _linear_rule([0])  # a sum (or mean) across ranks: linear
for _op in ("slice_scatter", "select_scatter"):
    JVP_RULES[_op] = _linear_rule([0, 1])
JVP_RULES["scatter_add"] = _linear_rule([0, 2])
JVP_RULES["index_copy"] = _linear_rule([0, 2])


@jvp_rule("cat")
def _cat(e, n, ts):
    if all(t is None or not _float(x) for x, t in zip(n.inputs, ts)):
        return [None]
    return [_replay(e, n, ts, list(range(len(n.inputs))))]


@jvp_rule("amax", "amin")
def _amax(e, n, ts):
    """The mean of the tangents of the entries equal to the result (ties split evenly, as the
    reverse rule and PyTorch)."""
    (x,) = n.inputs
    y = n.output
    t = ts[0]
    dims = n.attrs.get("dims") or list(range(x.type.rank))
    dims = sorted(d % max(x.type.rank, 1) for d in dims)
    kept = [1 if i in dims else s for i, s in enumerate(x.type.shape)]
    hit = e.op("eq", x, e.op("expand", e.op("reshape", y, shape=kept), shape=list(x.type.shape)))
    keep = n.attrs.get("keepdim", False)
    num = e.op("sum", e.op("where", hit, t, _zero(e, t)), dims=dims, keepdim=keep)
    ones = e.op("where", hit, e.const(1.0, x), _zero(e, x))
    den = e.op("sum", ones, dims=dims, keepdim=keep)
    return [e.op("div", num, den)]


@jvp_rule("sort", "topk")
def _sort(e, n, ts):
    """values = x at indices along dim, so their tangent is the tangent read at the indices."""
    (x,) = n.inputs
    idx = n.extra_outputs[0]
    if x.type.rank == 0:
        return [ts[0], None]
    d = n.attrs.get("dim", -1) % x.type.rank
    return [e.op("gather", ts[0], idx, dim=d), None]


@jvp_rule("max_pool2d")
def _max_pool2d(e, n, ts):
    """The tangent of the first maximum of each window (row major order, as PyTorch): the forward
    kernel's taps over the -inf padded input, with the tangent padded by zeros alongside."""
    from .grad_rules import _fixed, _pair
    (x,) = n.inputs
    y = n.output
    t = ts[0]
    kh, kw = _pair(n.attrs["kernel_size"])
    sh, sw = _pair(n.attrs.get("stride") or n.attrs["kernel_size"])
    ph, pw = _pair(n.attrs.get("padding", 0))
    dh, dw = _pair(n.attrs.get("dilation", 1))
    h, w = _fixed("max_pool2d", "spatial sizes of the input", x.type.shape[2:])
    oh, ow = _fixed("max_pool2d", "output spatial sizes", y.type.shape[2:])
    big_h = max(h + 2 * ph, (oh - 1) * sh + dh * (kh - 1) + 1)
    big_w = max(w + 2 * pw, (ow - 1) * sw + dw * (kw - 1) + 1)

    def padded(v: Value, fill: float) -> Value:
        for dim, before, after in ((2, ph, big_h - ph - h), (3, pw, big_w - pw - w)):
            shape = list(v.type.shape)
            parts = []
            if before:
                parts.append(e.op("full", shape=shape[:dim] + [before] + shape[dim + 1:], value=fill, dtype=v.type.dtype))
            parts.append(v)
            if after:
                parts.append(e.op("full", shape=shape[:dim] + [after] + shape[dim + 1:], value=fill, dtype=v.type.dtype))
            if len(parts) > 1:
                v = e.op("cat", *parts, dim=dim)
        return v

    xp, tp = padded(x, -math.inf), padded(t, 0.0)
    taken = None
    total = None
    taps = [(u, v) for u in range(kh) for v in range(kw)]
    for i, (u, v) in enumerate(taps):
        r0, c0 = u * dh, v * dw

        def tap(src):
            s = e.op("slice", src, dim=2, start=r0, end=r0 + sh * (oh - 1) + 1, step=sh)
            return e.op("slice", s, dim=3, start=c0, end=c0 + sw * (ow - 1) + 1, step=sw)
        hit = e.op("eq", tap(xp), y)
        first = hit if taken is None else e.op("logical_and", hit, e.op("logical_not", taken))
        if i + 1 < len(taps):
            taken = hit if taken is None else e.op("logical_or", taken, hit)
        part = e.op("where", first, tap(tp), _zero(e, t))
        total = part if total is None else e.op("add", total, part)
    return [total]


# ----------------------------------------------------------------------------- control flow


def _region(region: Program, flags: Sequence[bool], force: Optional[Sequence[bool]] = None
            ) -> Tuple[Program, List[bool]]:
    """The jvp of a region: inputs are its inputs, then a tangent for each input flagged in
    ``flags``; outputs are its outputs, then a tangent for each floating point output that has one
    (or is forced by ``force``). Returns the region and which outputs got a tangent."""
    e = Emitter(Emitter.names_of(region), "t_", region.params, region.param_data)
    e.inputs = list(region.inputs)
    tan: Dict[int, Value] = {}
    for v, f in zip(region.inputs, flags):
        if f:
            tan[id(v)] = e.new_input(f"t_{v.name}", v.type)
    _forward(e, region.nodes, tan)
    tangents, oflags = [], []
    for i, v in enumerate(region.outputs):
        t = tan.get(id(v))
        if t is None and force is not None and force[i]:
            t = e.zeros(v.type.shape, v.type.dtype)
        oflags.append(t is not None)
        if t is not None:
            tangents.append(t)
    prog = Program(region.name, e.inputs, e.params, e.param_data, e.nodes, list(region.outputs) + tangents,
                   dict(region.symbols))
    return prune(prog), oflags


def _new_values(e: Emitter, vals: Sequence[Value], prefix: str = "t_") -> List[Value]:
    return [Value(e.fresh(f"{prefix}{v.name}"), v.type, "op") for v in vals]


def _cond(e: Emitter, n: Node, ts: List[Optional[Value]], tan: Dict[int, Value]) -> None:
    ops = n.inputs[1:]
    flags = [t is not None for t in ts[1:]]
    then, other = n.attrs["then_branch"], n.attrs["else_branch"]
    _, a = _region(then, flags)
    _, b = _region(other, flags)
    force = [(x or y) and _float(o) for x, y, o in zip(a, b, n.outputs)]
    then2, _ = _region(then, flags, force)
    other2, _ = _region(other, flags, force)
    touts = _new_values(e, [o for o, f in zip(n.outputs, force) if f])
    attrs = dict(n.attrs)
    attrs["then_branch"], attrs["else_branch"] = then2, other2
    e.op_multi("cond", n.inputs[0], *ops, *[t for t in ts[1:] if t is not None], outs=list(n.outputs) + touts, **attrs)
    it = iter(touts)
    for o, f in zip(n.outputs, force):
        if f:
            tan[id(o)] = next(it)


def _loop(e: Emitter, n: Node, ts: List[Optional[Value]], tan: Dict[int, Value]) -> None:
    """Tangents ride along as extra carried values: the body's jvp maps (values, tangents) to (new
    values, new tangents). Nothing is stored per iteration."""
    k = n.attrs["n_carried"]
    carried, caps = list(n.inputs[:k]), list(n.inputs[k:])
    cflags = [t is not None for t in ts[:k]]
    kflags = [t is not None for t in ts[k:]]
    body, cond = n.attrs["body"], n.attrs.get("cond")
    while True:  # a carried value gets a tangent when the body gives it one: a fixed point
        _, out = _region(body, [False] + cflags + kflags)
        new = [(a or b) and _float(v) for a, b, v in zip(cflags, out, carried)]
        if new == cflags:
            break
        cflags = new
    body2, _ = _region(body, [False] + cflags + kflags, cflags)
    nc, nk = sum(cflags), sum(kflags)
    # body2 inputs: [i, carried, captures, tangents of carried, tangents of captures]; the loop wants
    # [i, carried, tangents of carried, captures, tangents of captures]
    ins = body2.inputs
    order = [ins[0]] + ins[1:1 + k] + ins[1 + k + len(caps):1 + k + len(caps) + nc] + ins[1 + k:1 + k + len(caps)] \
        + ins[1 + k + len(caps) + nc:]
    body2 = Program(body2.name, order, body2.params, body2.param_data, body2.nodes, body2.outputs, body2.symbols)
    body2.verify()
    attrs = dict(n.attrs)
    attrs["body"] = body2
    attrs["n_carried"] = k + nc
    if cond is not None:
        tc = [Value(f"t_{v.name}", v.type, "input") for v, f in zip(cond.inputs[:k], cflags) if f]
        tk = [Value(f"t_{v.name}", v.type, "input") for v, f in zip(cond.inputs[k:], kflags) if f]
        c2 = Program(cond.name, cond.inputs[:k] + tc + cond.inputs[k:] + tk, cond.params, cond.param_data,
                     cond.nodes, cond.outputs, cond.symbols)
        c2.verify()
        attrs["cond"] = c2
    init_t = [ts[i] if ts[i] is not None else e.zeros_like(carried[i]) for i in range(k) if cflags[i]]
    cap_t = [t for t in ts[k:] if t is not None]
    touts = _new_values(e, [o for o, f in zip(n.outputs, cflags) if f])
    e.op_multi("loop", *carried, *init_t, *caps, *cap_t, outs=list(n.outputs) + touts, **attrs)
    it = iter(touts)
    for o, f in zip(n.outputs, cflags):
        if f:
            tan[id(o)] = next(it)


def _custom_vjp(e: Emitter, n: Node, ts: List[Optional[Value]], tan: Dict[int, Value]) -> None:
    """The primal stays the custom_vjp node; the tangent is the forward mode derivative of its
    ``fwd`` region (the primal function). A rule registered with ``unverified=`` (a straight through
    estimator) is a reverse mode rule only, and the dot product test exposes the difference."""
    e.append(n)
    flags = [t is not None for t in ts]
    fwd2, (has,) = _region(n.attrs["fwd"], flags)
    if not has:
        return
    outs = e.inline(fwd2, list(n.inputs) + [t for t in ts if t is not None],
                    label=f"jvp of custom_vjp {n.attrs.get('name', '')}")
    tan[id(n.output)] = outs[1]


_CONTROL_FLOW.update({"cond": _cond, "loop": _loop, "custom_vjp": _custom_vjp})


# ----------------------------------------------------------------------------- the Jacobian family


def _eye(e: Emitter, m, shape, dtype) -> Value:
    """The identity of size m reshaped to [m, *shape], built in the program (works for symbolic m)."""
    r = e.op("arange", start=0, end=m, step=1, dtype=torch.int64)
    hit = e.op("eq", e.op("reshape", r, shape=[m, 1]), e.op("reshape", r, shape=[1, m]))
    eye = e.op("cast", hit, dtype=dtype)
    return e.op("reshape", eye, shape=[m] + list(shape))


def _block(prog: Program, what: str):
    e = Emitter(Emitter.names_of(prog), "j_", prog.params, prog.param_data)
    e.inputs = list(prog.inputs)
    e.context = what
    return e


def jacfwd(program: Any, wrt: Sequence[Union[str, Value]], of: Optional[Sequence[int]] = None,
           name: Optional[str] = None) -> Program:
    """The Jacobian by forward mode: ``vmap`` of ``jvp`` over the columns of the identity.

    Outputs: the original outputs, then for each ``of`` output (default: every floating point
    output) and each ``wrt`` value in order, the Jacobian of shape ``out.shape + wrt.shape`` (the
    torch.func.jacfwd layout). One ``vmap`` per ``wrt`` value; sizes may be symbolic."""
    from .vmap import vmap

    prog = as_program(program)
    wrt_values = _resolve(prog, wrt, "jacfwd")
    idx = _outputs(prog, of, "jacfwd")
    e = _block(prog, "jacfwd")
    jac: Dict[Tuple[int, int], Value] = {}
    primal = None
    for j, w in enumerate(wrt_values):
        jp = jvp(prog, [w], of=idx)
        m = numel(w.type.shape)
        tname = jp.inputs[len(prog.inputs)].name
        no = len(prog.outputs)
        vp = vmap(jp, in_axes={tname: 0}, size=m, out_axes=[None] * no + [0] * len(idx))
        basis = _eye(e, m, w.type.shape, w.type.dtype)
        outs = e.inline(vp, list(prog.inputs) + [basis], label=f"jacfwd column block of {w.ref}")
        if primal is None:
            primal = outs[:no]
        for i, t in zip(idx, outs[no:]):
            o = prog.outputs[i]
            ro, rw = o.type.rank, w.type.rank
            t = e.op("reshape", t, shape=list(w.type.shape) + list(o.type.shape))
            if rw and ro:
                t = e.op("permute", t, dims=list(range(rw, rw + ro)) + list(range(rw)))
            jac[(i, j)] = t
    outputs = list(primal) + [jac[(i, j)] for i in idx for j in range(len(wrt_values))]
    out = Program(name or f"jacfwd_{prog.name}", e.inputs, e.params, e.param_data, e.nodes, outputs,
                  dict(prog.symbols))
    return prune(out)


def jacrev(program: Any, wrt: Sequence[Union[str, Value]], of: Optional[Sequence[int]] = None,
           name: Optional[str] = None) -> Program:
    """The Jacobian by reverse mode: ``vmap`` of ``grad`` over the rows of the identity. Same
    layout as ``jacfwd``."""
    from .autodiff import grad
    from .vmap import vmap

    prog = as_program(program)
    wrt_values = _resolve(prog, wrt, "jacrev")
    idx = _outputs(prog, of, "jacrev")
    e = _block(prog, "jacrev")
    outputs_by: Dict[Tuple[int, int], Value] = {}
    primal = None
    no = len(prog.outputs)
    for i in idx:
        o = prog.outputs[i]
        gp = grad(prog, wrt_values, of=[i], seed="input")
        cname = gp.inputs[len(prog.inputs)].name
        m = numel(o.type.shape)
        vp = vmap(gp, in_axes={cname: 0}, size=m, out_axes=[None] * no + [0] * len(wrt_values))
        basis = _eye(e, m, o.type.shape, o.type.dtype)
        outs = e.inline(vp, list(prog.inputs) + [basis], label=f"jacrev row block of {o.ref}")
        if primal is None:
            primal = outs[:no]
        for j, (w, g) in enumerate(zip(wrt_values, outs[no:])):
            outputs_by[(i, j)] = e.op("reshape", g, shape=list(o.type.shape) + list(w.type.shape))
    if primal is None:
        primal = list(prog.outputs)
    outputs = list(primal) + [outputs_by[(i, j)] for i in idx for j in range(len(wrt_values))]
    out = Program(name or f"jacrev_{prog.name}", e.inputs, e.params, e.param_data, e.nodes, outputs,
                  dict(prog.symbols))
    return prune(out)


def hessian(program: Any, wrt: Sequence[Union[str, Value]], of: int = 0, name: Optional[str] = None) -> Program:
    """Second derivatives of a scalar output, forward over reverse: ``jacfwd(grad(program))``.

    Outputs: the original outputs, then for each pair (i, j) of ``wrt`` values in order the block
    of shape ``wrt[i].shape + wrt[j].shape`` (the torch.func.hessian layout)."""
    from .autodiff import grad

    prog = as_program(program)
    wrt_values = _resolve(prog, wrt, "hessian")
    (i,) = _outputs(prog, [of], "hessian")
    if prog.outputs[i].type.rank != 0:
        raise IRError(f"hessian: output {prog.outputs[i].ref}: {prog.outputs[i].type} is not a scalar; take the "
                      "jacfwd of a jacrev program for the second derivatives of a tensor output")
    g = grad(prog, wrt_values, of=[i], seed="auto")
    no = len(prog.outputs)
    gnames = [v.name for v in wrt_values]
    jf = jacfwd(g, gnames, of=list(range(no, no + len(wrt_values))))
    # jf outputs: g's outputs (originals, then gradients), then the blocks; keep originals and blocks
    keep = jf.outputs[:no] + jf.outputs[no + len(wrt_values):]
    out = Program(name or f"hessian_{prog.name}", jf.inputs, jf.params, jf.param_data, jf.nodes, keep,
                  dict(jf.symbols))
    return prune(out)


def linearize(program: Any, wrt: Sequence[Union[str, Value]], of: Optional[Sequence[int]] = None,
              name: Optional[str] = None) -> Tuple[Program, Program]:
    """Split the jvp program into a primal part and a linear part.

    Returns ``(primal, linear)``:
      primal: the original inputs -> the original outputs, then the residuals (every value the
              tangent code reads);
      linear: the residuals, then one tangent per ``wrt`` entry -> the tangent of each ``of``
              output. It is linear in the tangents and contains no node that depends only on the
              primal, so each further Jacobian vector product costs only the tangent work.

    ``linear(*residuals, *tangents)`` equals the tangent outputs of ``jvp(program, wrt, of)``.
    Through a ``cond`` or ``loop`` whose operands have tangents the primal of that node is computed
    again inside the linear program, since the node computes both together."""
    prog = as_program(program)
    jp = jvp(prog, wrt, of=of)
    n_in = len(prog.inputs)
    tangents = jp.inputs[n_in:]
    dep = {id(v) for v in tangents}
    lin_nodes, pri_nodes = [], []
    for n in jp.nodes:
        if any(x is not None and id(x) in dep for x in n.inputs):
            lin_nodes.append(n)
            dep.update(id(o) for o in n.outputs)
        else:
            pri_nodes.append(n)
    t_outs = jp.outputs[len(prog.outputs):]
    params = {id(p) for p in jp.params}
    residuals: List[Value] = []
    seen: Set[int] = set()

    def need(v: Value) -> None:
        if id(v) not in dep and id(v) not in params and id(v) not in seen:
            seen.add(id(v))
            residuals.append(v)

    for n in lin_nodes:
        for x in n.inputs:
            if x is not None:
                need(x)
    for v in t_outs:
        need(v)
    primal = Program(f"{name or 'linearize_' + prog.name}_primal", list(prog.inputs), list(jp.params),
                     dict(jp.param_data), pri_nodes, list(prog.outputs) + residuals, dict(prog.symbols))
    remap = {id(v): Value(v.name, v.type, "input") for v in residuals}
    nodes = [Node(n.op, tuple(None if x is None else remap.get(id(x), x) for x in n.inputs), n.attrs, n.output,
                  n.source, n.extra_outputs) for n in lin_nodes]
    linear = Program(f"{name or 'linearize_' + prog.name}_linear", [remap[id(v)] for v in residuals] + list(tangents),
                     list(jp.params), dict(jp.param_data), nodes, [remap.get(id(v), v) for v in t_outs],
                     dict(prog.symbols))
    return prune(primal), prune(linear)


from . import linattn_ops as _linattn_ops  # noqa: E402  (linear_attention, P60)
_linattn_ops.register_jvp(jvp_rule)
