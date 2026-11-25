"""Reverse mode gradients through control flow: ``cond``, ``loop`` and ``select_dynamic``.

Used by ``truestar.ir.autodiff.grad``; see docs/AUTODIFF.md, section "Control flow". Everything here
is still a program transformation: the backward of a ``cond`` is a ``cond``, the backward of a
``loop`` is a ``loop`` that runs the iterations in reverse, and both are built from the gradient
program of their regions (``grad`` applied to the region itself, so nested control flow recurses).

**cond.** ``cond(pred, *operands)`` gets a backward ``cond(pred, *operands, *cotangents)`` whose two
branches are the vector Jacobian products of ``then_branch`` and ``else_branch``. Nothing is stored:
the taken branch's internals are recomputed inside the backward branch.

**loop.** Iteration ``k`` maps the carried values ``c_k`` to ``c_{k+1} = body(k, c_k, captures)``.
The backward loop walks ``k = n - 1, ..., 0`` and needs ``c_k`` at each step. Where ``c_k`` comes
from is the memory policy, one word for the whole program (``grad(..., loop_memory=...)``):

* ``"store"``: the forward loop also writes each ``c_k`` the backward reads into a buffer of
  ``n`` rows, so memory grows with ``n`` and each body runs once forward and once more inside the
  backward step. Only the carried values the body's gradient actually reads are stored: a value
  only written by ``index_copy`` (a scan's stacked outputs) or only fed to comparisons costs nothing.
* ``"recompute"``: nothing is stored. Step ``k`` of the backward reruns the forward body ``k`` times
  from the initial values, so memory is constant and compute is about ``n^2 / 2`` body runs.
* ``"auto"`` (default): ``"store"`` when the trip count is a dim or a ``while`` loop has
  ``max_iterations``, ``"recompute"`` for a ``while`` loop with no bound (a buffer needs a size).

A ``while`` loop (``cond`` region) also gets a counter carried through its forward run, so the
backward knows how many iterations to undo. Gradients of captured values are accumulated across the
iterations in carried accumulators of the backward loop.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import sympy
import torch

from .core import Builder, IRError, Node, Program, TensorType, UnsupportedOpError, Value, norm_dim
from .grad_rules import grad_rule
from .regions import inline, prune, used_inputs
from .shapes import infer_outputs

LOOP_MEMORY = ("auto", "store", "recompute", "sqrt")
I64 = torch.int64


# ----------------------------------------------------------------------------- region utilities


def vjp_region(region: Program, wrt: Sequence[int], of: Sequence[int], loop_memory: str) -> Program:
    """The vector Jacobian product of a region: inputs are the region's inputs, then one cotangent
    per output in ``of``; outputs are the gradients of the inputs in ``wrt``. Pruned, so its
    used inputs are exactly what a backward pass has to supply."""
    from .autodiff import grad

    g = grad(region, wrt=[region.inputs[i] for i in wrt], of=list(of), seed="input",
             name=f"vjp_{region.name}", loop_memory=loop_memory)
    out = Program(g.name, g.inputs, g.params, g.param_data, g.nodes, g.outputs[len(region.outputs):],
                  dict(g.symbols))
    return prune(out)


def emit_multi(e, op: str, inputs: Sequence[Value], **attrs: Any) -> Tuple[Value, ...]:
    """Append a node that defines several values to the program the emitter builds."""
    ts = infer_outputs(op, [x.type for x in inputs], attrs)
    outs = tuple(Value(e.fresh(f"d_{op}"), t, "op") for t in ts)
    e.nodes.append(Node(op, tuple(inputs), dict(attrs), outs[0], e.context, outs[1:]))
    return outs


def _region_builder(name: str, symbols: Dict[str, Any]) -> Builder:
    b = Builder(name)
    b.symbols = dict(symbols)
    return b


def _row(b: Builder, x: Value) -> Value:
    """x as a single row ``[1, *shape]``, to write into a buffer with ``index_copy``."""
    return b.op("reshape", x, shape=[1] + list(x.type.shape))


def _as_index(b: Builder, k: Value) -> Value:
    return b.op("reshape", k, shape=[1])


def _i64(b: Builder, value: int) -> Value:
    return b.op("full", shape=[], value=value, dtype=I64)


# ----------------------------------------------------------------------------- select_dynamic


@grad_rule("select_dynamic")
def _select_dynamic_rule(e, n, g):
    """y = x.select(dim, i) with i known at run time: x_bar is zeros with g written at row i
    (``index_copy``, which also wraps a negative i). The index carries no gradient."""
    x, i = n.inputs
    d = n.attrs["dim"] % x.type.rank

    def fn():
        row = e.op("reshape", g, shape=list(x.type.shape[:d]) + [1] + list(x.type.shape[d + 1:]))
        return e.op("index_copy", e.zeros_like(x), e.op("reshape", i, shape=[1]), row, dim=d)
    return [fn, None]


# ----------------------------------------------------------------------------- cond


def _cond_forward(e, n: Node, active: Set[int]) -> None:
    e.nodes.append(n)


def _cond_rule(e, n: Node, gs: Sequence[Optional[Value]], active: Set[int]) -> List[Optional[Value]]:
    """cond(pred, *x) with results y: x_bar = cond(pred, vjp(then), vjp(else))(*x, *y_bar)."""
    operands = list(n.inputs[1:])
    of = [i for i, g in enumerate(gs) if g is not None]
    wrt = [j for j, x in enumerate(operands) if id(x) in active]
    out: List[Optional[Value]] = [None] * len(n.inputs)
    if not of or not wrt:
        return out
    then = vjp_region(n.attrs["then_branch"], wrt, of, e.loop_memory)
    other = vjp_region(n.attrs["else_branch"], wrt, of, e.loop_memory)
    grads = emit_multi(e, "cond", [n.inputs[0]] + operands + [gs[i] for i in of],
                       then_branch=then, else_branch=other)
    for j, gv in zip(wrt, grads):
        out[1 + j] = gv
    return out


# ----------------------------------------------------------------------------- loop


class _LoopPlan:
    """What the backward of one loop needs, decided while the forward is copied."""

    def __init__(self, n_carried: int, policy: str, vjp: Program, float_carried: List[int],
                 active_caps: List[int], stored: List[int], buffers: List[Value], trips: Any,
                 body: Program, cond: Optional[Program], segment: Any = None, segments: Any = None):
        self.n = n_carried
        self.policy = policy
        self.vjp = vjp
        self.float_carried = float_carried  # carried positions that get cotangents
        self.active_caps = active_caps      # captured positions that get gradients
        self.stored = stored                # carried positions the gradient step reads
        self.buffers = buffers              # "store": one per stored position; "sqrt": one per carried
        self.trips = trips                  # a dim expression, or the counter value of a while loop
        self.body = body
        self.cond = cond
        self.segment = segment              # "sqrt": iterations per checkpoint (a dim expression)
        self.segments = segments            # "sqrt": number of checkpoints


def _policy(n: Node, requested: str) -> str:
    """The policy one loop gets. ``requested`` is the program wide choice; a loop that cannot follow
    it says so in the ``memory`` attribute of its backward loop:

    | loop                       | "store" | "sqrt"  | "recompute" | "auto"  |
    |----------------------------|---------|---------|-------------|---------|
    | trip count                 | store   | sqrt    | recompute   | store   |
    | while with max_iterations  | store   | store   | recompute   | store   |
    | while with no bound        | recompute (a buffer needs a size)                |
    """
    has_trip = n.attrs.get("trip_count") is not None
    bounded = has_trip or n.attrs.get("max_iterations") is not None
    if not bounded:
        return "recompute"
    if requested == "auto" or (requested == "sqrt" and not has_trip):
        return "store"
    return requested


def _nonneg(expr) -> Any:
    expr = norm_dim(expr)
    if isinstance(expr, int):
        return max(expr, 0)
    return expr if sympy.sympify(expr).is_nonnegative else norm_dim(sympy.Max(expr, 0))


def _ceil_div(a, b) -> Any:
    if isinstance(a, int) and isinstance(b, int):
        return -(-a // b)
    return norm_dim(sympy.ceiling(sympy.sympify(a) / sympy.sympify(b)))


def _sqrt_segment(trips) -> Any:
    """Iterations per checkpoint: ceil(sqrt(n)), at least 1."""
    if isinstance(trips, int):
        return max(1, math.isqrt(max(trips - 1, 0)) + 1) if trips > 0 else 1
    return norm_dim(sympy.Max(1, sympy.ceiling(sympy.sqrt(sympy.sympify(trips)))))


def _dim_i64(b: Builder, expr) -> Value:
    return b.op("dim_scalar", expr=norm_dim(expr), dtype=I64)


def _inputs(b: Builder, specs: Sequence[Tuple[str, TensorType]]) -> List[Value]:
    return [b.input(name, t.dtype, t.shape) for name, t in specs]


def _types(prefix: str, values: Sequence[Value]) -> List[Tuple[str, TensorType]]:
    return [(f"{prefix}{i}", v.type) for i, v in enumerate(values)]


def _buffer(b_or_e, rows, t: TensorType) -> Value:
    return b_or_e.op("full", shape=[rows] + list(t.shape), value=0, dtype=t.dtype)


def _loop_forward(e, n: Node, active: Set[int]) -> None:
    """Copy a loop into the forward program, augmented with what its backward needs."""
    attrs = n.attrs
    nc = attrs["n_carried"]
    carried, caps = list(n.inputs[:nc]), list(n.inputs[nc:])
    body, cond = attrs["body"], attrs.get("cond")
    policy = _policy(n, e.loop_memory)
    float_carried = [i for i, x in enumerate(carried) if x.type.dtype.is_floating_point]
    active_caps = [j for j, x in enumerate(caps) if id(x) in active]
    wrt = [1 + i for i in float_carried] + [1 + nc + j for j in active_caps]
    vjp = vjp_region(body, wrt, float_carried, e.loop_memory)
    used = used_inputs(vjp)
    stored = [i for i in range(nc) if 1 + i in used]
    is_while = cond is not None
    trips = None if is_while else _nonneg(attrs["trip_count"])
    plan = _LoopPlan(nc, policy, vjp, float_carried, active_caps, stored, [], trips, body, cond)

    # what the forward loop also carries: [count] for a while loop, then buffers
    extra_init: List[Value] = []
    if is_while:
        extra_init.append(e.op("full", shape=[], value=0, dtype=I64))
    if policy == "store":
        rows = trips if not is_while else attrs["max_iterations"]
        extra_init += [_buffer(e, rows, carried[i].type) for i in stored]
    elif policy == "sqrt" and stored:
        plan.segment = _sqrt_segment(trips)
        plan.segments = _ceil_div(trips, plan.segment)
        extra_init.append(e.op("full", shape=[], value=0, dtype=I64))  # checkpoints written so far
        extra_init += [_buffer(e, plan.segments, x.type) for x in carried]
    if not extra_init:
        e.nodes.append(n)
        e.cf_plans[id(n)] = plan
        return

    b = _region_builder(f"{body.name}_fwd", {})
    k = b.input("k", I64, ())
    c_in = _inputs(b, _types("c", carried))
    x_in = _inputs(b, _types("x", extra_init))
    cap_in = _inputs(b, _types("cap", caps))
    new_c = inline(b, body, [k] + c_in + cap_in)
    extra_out: List[Value] = []
    rest = list(x_in)
    if is_while:
        extra_out.append(b.op("add", rest.pop(0), _i64(b, 1)))
    if policy == "store":
        for i, buf in zip(stored, rest):
            extra_out.append(b.op("index_copy", buf, _as_index(b, k), _row(b, c_in[i]), dim=0))
    elif policy == "sqrt" and stored:
        slot, cps = rest[0], rest[1:]
        first = b.op("eq", b.op("remainder", k, _dim_i64(b, plan.segment)), _i64(b, 0))
        extra_out.append(b.op("add", slot, b.op("cast", first, dtype=I64)))
        # checkpoint c_k at row ``slot`` when k is the first iteration of a segment
        t = _region_builder("checkpoint", {})
        ts, tc, tb = t.input("slot", I64, ()), _inputs(t, _types("c", carried)), _inputs(t, _types("cp", cps))
        written = t.build([t.op("index_copy", cp, _as_index(t, ts), _row(t, c), dim=0) for c, cp in zip(tc, tb)])
        f = _region_builder("keep", {})
        f.input("slot", I64, ())
        _inputs(f, _types("c", carried))
        kept = f.build(_inputs(f, _types("cp", cps)))
        extra_out += list(b.op_multi("cond", first, slot, *c_in, *cps, then_branch=written, else_branch=kept))
    new_body = b.build(new_c + extra_out)

    new_attrs = dict(attrs)
    new_attrs["body"] = new_body
    new_attrs["n_carried"] = nc + len(extra_init)
    if is_while:
        bc = _region_builder(f"{cond.name}_fwd", {})
        cc = _inputs(bc, _types("c", carried))
        _inputs(bc, _types("x", extra_init))
        ccap = _inputs(bc, _types("cap", caps))
        new_attrs["cond"] = bc.build(inline(bc, cond, cc + ccap))
    ins = carried + extra_init + caps
    ts_ = infer_outputs("loop", [x.type for x in ins], new_attrs)
    extras = list(Value(e.fresh(f"{n.output.name}_saved"), t_, "op") for t_ in ts_[nc:])
    new = Node("loop", tuple(ins), new_attrs, n.output, n.source, tuple(n.extra_outputs) + tuple(extras))
    e.nodes.append(new)
    if is_while:
        plan.trips = extras.pop(0)
    if policy == "sqrt" and stored:
        extras.pop(0)  # the checkpoint count, not needed after the loop
    plan.buffers = extras
    e.cf_plans[id(new)] = plan


def _accumulate(b: Builder, acc: Value, g: Value) -> Value:
    """acc + g. When g is zeros with a few rows written (``index_copy`` into zeros, the gradient of
    ``x[:, t]`` in a loop), only those rows are added: ``acc`` with rows ``i`` replaced by
    ``acc[i] + rows``. The index of one iteration never repeats, so this is exact, and each step
    costs the rows it touches instead of the whole captured tensor."""
    src = next((n for n in reversed(b.nodes) if n.output is g), None)
    if src is not None and src.op == "index_copy":
        base = next((n for n in b.nodes if n.output is src.inputs[0]), None)
        if base is not None and base.op in ("full", "full_like") and base.attrs.get("value") == 0.0:
            _, index, rows = src.inputs
            d = src.attrs["dim"]
            old = b.op("index_select", acc, index, dim=d)
            return b.op("index_copy", acc, index, b.op("add", old, rows), dim=d)
    return b.op("add", acc, g)


def _step_back(b: Builder, plan: _LoopPlan, k: Value, c_k: Sequence[Optional[Value]], caps: Sequence[Value],
               g: Sequence[Value], acc: Sequence[Value]) -> List[Value]:
    """One reverse iteration: the body's vector Jacobian product at iteration k. Returns the
    cotangents of the carried values entering iteration k, then the updated capture gradients."""
    grads = inline(b, plan.vjp, [k] + list(c_k) + list(caps) + list(g))
    nf = len(plan.float_carried)
    return grads[:nf] + [_accumulate(b, a, gc) for a, gc in zip(acc, grads[nf:])]


def _while(b: Builder, name: str, carried: Sequence[Value], caps: Sequence[Value], limit_pos: int,
           step: Callable[[Builder, Value, List[Value], List[Value]], List[Value]], **attrs: Any) -> List[Value]:
    """A while loop ``i = 0; while i < caps[limit_pos]: carried = step(i, carried, caps); i += 1``
    built in ``b``. Returns the final carried values (without the counter)."""
    c = _region_builder(f"{name}_cond", {})
    ci = c.input("i", I64, ())
    _inputs(c, _types("c", carried))
    ccaps = _inputs(c, _types("cap", caps))
    cond = c.build([c.op("lt", ci, ccaps[limit_pos])])
    r = _region_builder(name, {})
    r.input("iteration", I64, ())
    ri = r.input("i", I64, ())
    rc = _inputs(r, _types("c", carried))
    rcaps = _inputs(r, _types("cap", caps))
    new = step(r, ri, rc, rcaps)
    body = prune(r.build([r.op("add", ri, _i64(r, 1))] + list(new)))
    outs = b.op_multi("loop", _i64(b, 0), *carried, *caps, n_carried=1 + len(carried), body=body, cond=cond,
                      **attrs)
    return list(outs[1:])


def _replay(b: Builder, plan: _LoopPlan, start: Value, count: Value, c0: Sequence[Value],
            caps: Sequence[Value], record: Sequence[int], rows) -> List[Value]:
    """Run the forward body ``count`` times from iteration ``start`` and carried values ``c0``. With
    ``record``, returns buffers of ``rows`` rows holding the carried values at those positions as
    each iteration began; without, returns the final carried values."""
    nc = len(c0)
    bufs = [_buffer(b, rows, c0[i].type) for i in record]

    def step(r, i, cur, rcaps):
        k = r.op("add", rcaps[0], i)
        c, bs = cur[:nc], cur[nc:]
        new_b = [r.op("index_copy", buf, _as_index(r, i), _row(r, c[pos]), dim=0) for pos, buf in zip(record, bs)]
        return inline(r, plan.body, [k] + c + rcaps[2:]) + new_b

    out = _while(b, "replay", list(c0) + bufs, [start, count] + list(caps), 1, step)
    return out[nc:] if record else out[:nc]


def _loop_rule(e, n: Node, gs: Sequence[Optional[Value]], active: Set[int]) -> List[Optional[Value]]:
    plan: _LoopPlan = e.cf_plans[id(n)]
    nc = plan.n
    carried, caps = list(n.inputs[:nc]), list(n.inputs[n.attrs["n_carried"]:])
    out: List[Optional[Value]] = [None] * len(n.inputs)
    gs = list(gs[:nc])
    if not plan.float_carried or all(gs[i] is None for i in plan.float_carried):
        return out
    is_while = plan.cond is not None
    nf = len(plan.float_carried)
    g_init = [gs[i] if gs[i] is not None else e.zeros(carried[i].type.shape, carried[i].type.dtype)
              for i in plan.float_carried]
    acc_init = [e.zeros(caps[j].type.shape, caps[j].type.dtype) for j in plan.active_caps]
    state = g_init + acc_init

    if plan.policy == "sqrt" and plan.stored:
        res = _sqrt_backward(e, plan, state, caps)
    else:
        count = [plan.trips] if is_while else []
        saved = list(plan.buffers) if plan.policy == "store" else []
        init = carried if plan.policy == "recompute" else []
        side = count + saved + init + caps  # captures of the backward loop

        def reverse_step(r, j, cur, rcaps):
            cnt, rest = (rcaps[:1], rcaps[1:]) if is_while else ([], rcaps)
            bufs, rest = rest[:len(saved)], rest[len(saved):]
            init_in, cap_in = rest[:len(init)], rest[len(init):]
            last = r.op("sub", cnt[0], _i64(r, 1)) if is_while else _dim_i64(r, sympy.sympify(plan.trips) - 1)
            k = r.op("sub", last, j)
            if plan.policy == "recompute":
                c_k: List[Optional[Value]] = _replay(r, plan, _i64(r, 0), k, init_in, cap_in, [], None)
            else:
                c_k = [None] * nc
                for i, buf in zip(plan.stored, bufs):
                    c_k[i] = r.op("select_dynamic", buf, k, dim=0)
            return _step_back(r, plan, k, c_k, cap_in, cur[:nf], cur[nf:])

        if is_while:
            res = _while(e_builder(e), "bwd", state, side, 0, reverse_step, memory=plan.policy)
        else:
            res = _counted(e, plan.trips, state, side, reverse_step, plan.policy)
    for pos, i in enumerate(plan.float_carried):
        if id(carried[i]) in active:
            out[i] = res[pos]
    for pos, j_ in enumerate(plan.active_caps):
        out[n.attrs["n_carried"] + j_] = res[nf + pos]
    return out


def _sqrt_backward(e, plan: _LoopPlan, state: List[Value], caps: Sequence[Value]) -> List[Value]:
    """Reverse over checkpoint segments, last first: restore the segment's first carried values,
    replay the segment storing what the gradient step reads (at most ``segment`` rows), then walk
    the segment backwards."""
    nc, nf = plan.n, len(plan.float_carried)
    S, n_seg = plan.segment, plan.segments

    def segment_step(r, jj, cur, rcaps):
        cps, cap_in = rcaps[:nc], rcaps[nc:]
        seg = r.op("sub", _dim_i64(r, sympy.sympify(n_seg) - 1), jj)
        seg_len = _dim_i64(r, S)
        start = r.op("mul", seg, seg_len)
        count = r.op("minimum", seg_len, r.op("sub", _dim_i64(r, plan.trips), start))
        c_start = [r.op("select_dynamic", cp, seg, dim=0) for cp in cps]
        local = _replay(r, plan, start, count, c_start, cap_in, plan.stored, S)

        def inner(q, i, st, qcaps):
            qstart, qcount, qbufs, qcap = qcaps[0], qcaps[1], qcaps[2:2 + len(local)], qcaps[2 + len(local):]
            pos = q.op("sub", q.op("sub", qcount, _i64(q, 1)), i)
            k = q.op("add", qstart, pos)
            c_k: List[Optional[Value]] = [None] * nc
            for idx, buf in zip(plan.stored, qbufs):
                c_k[idx] = q.op("select_dynamic", buf, pos, dim=0)
            return _step_back(q, plan, k, c_k, qcap, st[:nf], st[nf:])

        return _while(r, "segment_bwd", cur, [start, count] + local + list(cap_in), 1, inner)

    return _counted(e, n_seg, state, list(plan.buffers) + list(caps), segment_step, "sqrt")


class _EmitterBuilder:
    """Lets the region helpers above emit into the outer program through the autodiff emitter."""

    def __init__(self, e):
        self.e = e

    def op(self, op, *inputs, **attrs):
        return self.e.op(op, *inputs, **attrs)

    def op_multi(self, op, *inputs, **attrs):
        return emit_multi(self.e, op, list(inputs), **attrs)


def e_builder(e) -> _EmitterBuilder:
    return _EmitterBuilder(e)


def _counted(e, trips, state: List[Value], side: List[Value],
             step: Callable[[Builder, Value, List[Value], List[Value]], List[Value]], memory: str) -> List[Value]:
    """A loop with a trip count emitted into the outer program: ``state = step(j, state, side)`` for
    j = 0 .. trips - 1."""
    r = _region_builder("bwd", {})
    j = r.input("iteration", I64, ())
    cur = _inputs(r, _types("g", state))
    rcaps = _inputs(r, _types("cap", side))
    body = prune(r.build(list(step(r, j, cur, rcaps))))
    return list(emit_multi(e, "loop", state + side, n_carried=len(state), body=body, trip_count=trips,
                           memory=memory))


# ----------------------------------------------------------------------------- registry

CF_FORWARD: Dict[str, Callable] = {"cond": _cond_forward, "loop": _loop_forward}
CF_RULES: Dict[str, Callable] = {"cond": _cond_rule, "loop": _loop_rule}
