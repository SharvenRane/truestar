"""``vmap`` as a program transformation.

``vmap(program, in_axes=..., out_axes=...)`` returns a new IR program that runs ``program`` once
per element of a new leading batch axis, written as ordinary IR ops over the whole batch. There is
no batched tensor type and no run time dispatch layer: the result is a plain program that the
reference interpreter and the compiler run like any other, and one program serves every batch size
(the batch axis is a symbolic dim unless ``size=`` fixes it).

How it works. Every value of the new program is either *unbatched* (the same value for every
element, kept exactly as in the original program, so the pass never copies a weight) or *batched*
(its type is the original type with the batch dim in front). Each node is rewritten by a rule:

* elementwise and broadcasting ops: batched operands get size 1 dims after the batch dim so the
  original broadcasting still lines up; a 0 dim operand that the batch dim would let widen a dtype
  is cast first, so the result dtype is the original one (PyTorch's 0 dim promotion rule);
* ops with a ``dim`` attribute: the dim moves one to the right;
* batch like ops (conv2d, pooling, batch norm, group norm): the batch dim merges with the op's own
  batch dim and is split again after;
* indexing with a batched index: one advanced index with ``arange`` over the batch dim;
* ``cond`` with a batched predicate: each branch runs batched when some element takes it (a branch
  no element takes is skipped) and ``where`` picks per element;
  with an unbatched predicate the branches are batched and the node stays a ``cond``;
* ``loop``: carried values become batched when the body makes them batched (a fixed point); a
  ``while`` condition that differs per element runs until every element is done, and an element
  that is done keeps its carried values (``where`` with the per element condition);
* random ops (``multinomial``): every element gets its own stream, see RANDOMNESS below;
* every other op: an exact per element ``loop`` over the batch (``select_dynamic`` in,
  ``index_copy`` out). No op is left without a rule.

After every node the pass checks that each batched result has the original type with the batch
dim in front, so a wrong rule cannot produce a mistyped program.

RANDOMNESS. A random op inside vmap is always batched, even when its operands are not, and every
element gets its own stream that can be reproduced one element at a time:

* the keyed ``random`` op (truestar.rng): element ``k`` draws exactly what the op draws alone with
  ``k`` appended to its streams, as if the batch were a loop whose index is a stream. With an
  unbatched key this is one vectorized ``random_batched`` op (truestar/ir/random_batched.py);
* ``multinomial``: element ``k`` draws what the op draws alone with its run time offset increased
  by ``k * 2**40`` (offset 0 when it has none).

To share one draw across the batch, draw it outside vmap and pass it in unbatched.

See docs/TRANSFORMS.md.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import sympy
import torch

from .core import Dim, IRError, Node, Program, ShapeError, TensorType, UnsupportedOpError, Value, symbol
from .grad_rules import LOWERINGS
from .shapes import RULES, result_dtype
from .transform_util import Emitter, as_program, numel, prune

__all__ = ["vmap", "BATCH_RULES", "LOOP_FALLBACK", "RANDOM_OPS", "UNDEFINED", "batching_table"]

# Offset stride between the random streams of two batch elements (see RANDOMNESS above).
RANDOM_STRIDE = 1 << 40

Pair = Tuple[Value, bool]                       # (value in the new program, batched?)
Rule = Callable[["_Batcher", Node, List[Optional[Pair]]], List[Pair]]
BATCH_RULES: Dict[str, Rule] = {}
# Ops batched only by the exact per element loop. Every other op has a direct rule, and the rules use
# the loop themselves only for the cases they cannot vectorize (a per element weight of group_norm or
# rms_norm, a per element index of index_copy, conv2d with per element weights and a symbolic batch).
# kvq_attention (truestar.kvquant's decode attention) takes the loop: its cache layout, scale groups
# and live tail length have no batch dim to fold, and the op is pure, so the loop is exact.
LOOP_FALLBACK: set = {"kvq_attention", "mla_attention"}   # mla_attention: the same, pos and caches
RANDOM_OPS = {"multinomial", "random", "random_batched"}
# Ops whose per element meaning vmap cannot define. Empty for the ops on main; an op added later is
# refused until it is classified (test_every_ir_op_has_a_batching_rule), so a new random op can
# never silently share one stream across the batch.
UNDEFINED: Dict[str, str] = {}


def batch_rule(*names: str):
    def deco(fn):
        for n in names:
            BATCH_RULES[n] = fn
        return fn
    return deco


def batching_table() -> Dict[str, str]:
    """Every IR op with how vmap batches it: "rule", "loop" (exact per element loop), "random"
    (per element stream), "no operands" (never batched) or "undefined: reason"."""
    out = {}
    for op in RULES:
        if op in UNDEFINED:
            out[op] = f"undefined: {UNDEFINED[op]}"
        elif op in RANDOM_OPS:
            out[op] = "random"
        elif op in BATCH_RULES:
            out[op] = "rule"
        elif op in LOOP_FALLBACK:
            out[op] = "loop"
        elif op in _NO_OPERANDS:
            out[op] = "no operands"
        else:
            out[op] = "unclassified"
    return dict(sorted(out.items()))


_NO_OPERANDS = {"full", "arange", "dim_scalar"}


def _where(node: Node) -> str:
    return f"%{node.output.name}" + (f", from {node.source}" if node.source else "")


# ----------------------------------------------------------------------------- the batcher


class _Batcher:
    """Rewrites the nodes of one program (or region) for a batch of size ``n``."""

    def __init__(self, e: Emitter, n: Dim, symbols: Dict[str, Tuple[int, Optional[int]]]):
        self.e = e
        self.n = n
        self.symbols = symbols
        self.env: Dict[int, Pair] = {}

    # --- value helpers

    def bt(self, t: TensorType) -> TensorType:
        return TensorType(t.dtype, (self.n,) + tuple(t.shape))

    def full(self, p: Pair) -> Value:
        """The batched form of a value: an unbatched value is expanded along a new batch dim."""
        v, b = p
        if b:
            return v
        return self.e.op("expand", v, shape=[self.n] + list(v.type.shape))

    def lift(self, p: Pair, rank: int) -> Value:
        """A batched value of original rank r, reshaped to [n, 1, ..., 1, *shape] with ``rank``
        dims after n, so it broadcasts like the original against operands of rank ``rank``."""
        v, b = p
        if not b:
            return v
        r = v.type.rank - 1
        if r == rank:
            return v
        shape = list(v.type.shape)
        return self.e.op("reshape", v, shape=[shape[0]] + [1] * (rank - r) + shape[1:])

    def merge(self, v: Value) -> Value:
        """[n, b, *rest] -> [n * b, *rest]."""
        s = list(v.type.shape)
        return self.e.op("reshape", v, shape=[numel(s[:2])] + s[2:])

    def split(self, v: Value, b: Dim) -> Value:
        """[n * b, *rest] -> [n, b, *rest]."""
        s = list(v.type.shape)
        return self.e.op("reshape", v, shape=[self.n, b] + s[1:])

    def arange(self) -> Value:
        return self.e.op("arange", start=0, end=self.n, step=1, dtype=torch.int64)

    # --- driver

    def run(self, nodes: Sequence[Node]) -> None:
        for node in nodes:
            ins = [None if x is None else self.env[id(x)] for x in node.inputs]
            batched = any(p is not None and p[1] for p in ins)
            if node.op in UNDEFINED and (batched or node.op in RANDOM_OPS):
                raise UnsupportedOpError(f"vmap cannot batch {node.op} ({_where(node)}): {UNDEFINED[node.op]}")
            if not batched and node.op not in RANDOM_OPS:
                if all(p is None or p[0] is x for p, x in zip(ins, node.inputs)):
                    self.e.append(node)
                    for o in node.outputs:
                        self.env[id(o)] = (o, False)
                    continue
                # an operand was replaced by an equal unbatched value (inside a rewritten region)
                new = Node(node.op, tuple(None if p is None else p[0] for p in ins), dict(node.attrs),
                           node.output, node.source, node.extra_outputs)
                self.e.append(new)
                for o in node.outputs:
                    self.env[id(o)] = (o, False)
                continue
            self.e.context = node.source
            if node.op in RANDOM_OPS:
                outs = _random(self, node, ins)
            elif node.op in BATCH_RULES:
                outs = BATCH_RULES[node.op](self, node, ins)
            elif node.op in LOOP_FALLBACK:
                outs = self.loop_fallback(node, ins)
            else:
                raise UnsupportedOpError(
                    f"vmap has no batching rule for IR op {node.op!r} ({_where(node)}). Every IR op must be "
                    "classified in truestar/ir/vmap.py: give it a rule, or add it to LOOP_FALLBACK when it is a "
                    "pure deterministic op (an exact per element loop), or to RANDOM_OPS with a per element "
                    "key rule when it draws random numbers")
            if len(outs) != len(node.outputs):
                raise IRError(f"internal error: the batching rule of {node.op} returned {len(outs)} values "
                              f"for {len(node.outputs)}")
            for o, (v, b) in zip(node.outputs, outs):
                want = self.bt(o.type) if b else o.type
                if not v.type.same_as(want):
                    raise IRError(f"internal error: the batching rule of {node.op} ({_where(node)}) gives "
                                  f"{v.type} for {o.ref}: {o.type}, expected {want}")
                self.env[id(o)] = (v, b)

    # --- the exact per element loop

    def loop_fallback(self, node: Node, ins: List[Optional[Pair]], offset_operand: Optional[int] = None,
                      random_offset: bool = False, append_index: bool = False) -> List[Pair]:
        """Run the op once per batch element: a ``loop`` over the batch whose body selects element
        ``i`` of every batched operand, runs the op, and writes the results into row ``i`` of
        buffers (``index_copy``). Exact for any pure op, since it is the op itself per element.

        With ``random_offset`` the op's offset operand (position ``offset_operand``, added when
        absent) becomes ``offset + i * RANDOM_STRIDE``."""
        e = self.e
        bufs = [e.zeros([self.n] + list(o.type.shape), o.type.dtype) for o in node.outputs]
        caps = [p for p in ins if p is not None]
        rb = Emitter(set(), "v_")
        rb.context = node.source
        idx = rb.new_input("i", TensorType(torch.int64, ()))
        rbufs = [rb.new_input(f"buf{k}", b.type) for k, b in enumerate(bufs)]
        rcaps = [rb.new_input(f"x{k}", p[0].type) for k, p in enumerate(caps)]
        it = iter(zip(rcaps, caps))
        args: List[Optional[Value]] = []
        for p in ins:
            if p is None:
                args.append(None)
                continue
            rv, (v, b) = next(it)
            args.append(rb.op("select_dynamic", rv, idx, dim=0) if b else rv)
        if random_offset:
            step = rb.op("mul", idx, rb.int_const(RANDOM_STRIDE))
            if offset_operand is not None and offset_operand < len(args) and args[offset_operand] is not None:
                base = args[offset_operand]
                if base.type.dtype != torch.int64:
                    base = rb.op("cast", base, dtype=torch.int64)
                args[offset_operand] = rb.op("add", base, step)
            else:
                while len(args) < (offset_operand or 0):
                    args.append(None)
                args.append(step)
        if append_index:  # keyed random ops: the element index is one more stream operand
            args.append(idx)
        if len(node.outputs) == 1:
            res = [rb.op(node.op, *args, **node.attrs)]
        else:
            res = rb.op_multi(node.op, *args, **node.attrs)
        at = rb.op("reshape", idx, shape=[1])
        new = []
        for buf, r in zip(rbufs, res):
            row = rb.op("reshape", r, shape=[1] + list(r.type.shape))
            new.append(rb.op("index_copy", buf, at, row, dim=0))
        body = Program(f"{node.op}_per_element", rb.inputs, rb.params, rb.param_data, rb.nodes, new,
                       dict(self.symbols))
        outs = e.op_multi("loop", *bufs, *[p[0] for p in caps], n_carried=len(bufs), trip_count=self.n, body=body)
        return [(o, True) for o in outs]

    # --- regions

    def region(self, region: Program, flags: Sequence[bool], force: Optional[Sequence[bool]] = None
               ) -> Tuple[Program, List[bool]]:
        """Batch a region whose inputs are batched where ``flags`` says. Returns the new region and
        which outputs are batched; ``force`` makes those outputs batched."""
        e = Emitter(Emitter.names_of(region), "v_", region.params, region.param_data)
        sub = _Batcher(e, self.n, self.symbols)
        for v, f in zip(region.inputs, flags):
            nv = Value(v.name, self.bt(v.type) if f else v.type, "input")
            e.inputs.append(nv)
            sub.env[id(v)] = (nv, f)
        for p in region.params:
            sub.env[id(p)] = (p, False)
        sub.run(region.nodes)
        outs, oflags = [], []
        for i, v in enumerate(region.outputs):
            nv, b = sub.env[id(v)]
            if force is not None and force[i] and not b:
                nv, b = sub.full((nv, b)), True
            outs.append(nv)
            oflags.append(b)
        symbols = dict(region.symbols)
        symbols.update(self.symbols)
        prog = Program(region.name, e.inputs, e.params, e.param_data, e.nodes, outs, symbols)
        prog.verify()
        return prog, oflags


# ----------------------------------------------------------------------------- the entry point


def _axis_spec(spec, names: Sequence[str], what: str, extra: Sequence[str] = ()) -> Dict[str, Optional[int]]:
    if spec is None or isinstance(spec, int):
        return {n: spec for n in names}
    if isinstance(spec, dict):
        unknown = [k for k in spec if k not in names and k not in extra]
        if unknown:
            raise IRError(f"vmap: {what} names {unknown}, which are not inputs or params. Inputs: {list(names)}"
                          + (f". Params: {list(extra)}" if extra else ""))
        out = {n: None for n in names}
        out.update(spec)
        return out
    spec = list(spec)
    if len(spec) != len(names):
        raise IRError(f"vmap: {what} has {len(spec)} entries for {len(names)} values ({list(names)})")
    return dict(zip(names, spec))


def _fresh_symbol(prog: Program, name: Optional[str]) -> sympy.Symbol:
    used = set(prog.symbols)
    for v in prog.inputs + prog.params:
        used |= {str(s) for d in v.type.shape if isinstance(d, sympy.Basic) for s in d.free_symbols}
    if name is not None:
        if name in used:
            raise IRError(f"vmap: axis_name {name!r} is already a symbol of program {prog.name}")
        return symbol(name)
    base, k = "V", 0
    cand = base
    while cand in used:
        k += 1
        cand = f"{base}{k}"
    return symbol(cand)


def vmap(program: Any, in_axes: Union[int, None, Sequence, Dict[str, Optional[int]]] = 0,
         out_axes: Union[int, None, Sequence] = 0, size: Optional[Union[int, Dim]] = None,
         axis_name: Optional[str] = None, name: Optional[str] = None) -> Program:
    """Batch ``program`` over a new axis.

    in_axes: where each input carries the batch axis. An int (or None) applies to every input; a
      list gives one entry per input; a dict maps input or param names to an axis, and unnamed
      inputs are unbatched. A param named in the dict becomes a new program input (after the
      original inputs) holding one copy per element, for example a stack of ensemble weights.
    out_axes: where each output carries the batch axis (int, or a list with one per output). An
      output that does not depend on the batch is broadcast along it; None leaves an output
      unbatched and is an error for an output that depends on the batch.
    size: the batch size. Default: a new symbol (``V``, or ``axis_name``), so one program serves
      every batch size. Needed when no input is batched.

    The new program verifies, and every node is checked to give the original type with the batch
    dim in front. Raises UnsupportedOpError naming the op where batching is undefined.
    """
    prog = as_program(program)
    in_names = [v.name for v in prog.inputs]
    param_names = [v.name for v in prog.params]
    axes = _axis_spec(in_axes, in_names, "in_axes", param_names)
    batched_params = {k: a for k, a in axes.items() if k in param_names and a is not None}
    if not any(a is not None for a in axes.values()) and size is None:
        raise IRError("vmap: no input is batched and no size= is given, so the batch size is unknown; "
                      "name a batched input in in_axes or pass size=")
    n = _fresh_symbol(prog, axis_name) if size is None else (
        symbol(size) if isinstance(size, str) else size)
    symbols = dict(prog.symbols)
    if isinstance(n, sympy.Symbol) and str(n) not in symbols:
        symbols[str(n)] = (1, None)

    e = Emitter(Emitter.names_of(prog), "v_", [p for p in prog.params if p.name not in batched_params],
                {k: d for k, d in prog.param_data.items() if k not in batched_params})
    bt = _Batcher(e, n, symbols)
    moves = []
    for v in list(prog.inputs) + [p for p in prog.params if p.name in batched_params]:
        a = axes.get(v.name)
        if a is None:
            if v.kind == "input":
                e.inputs.append(v)
            bt.env[id(v)] = (v, False)
            continue
        r = v.type.rank
        if not -(r + 1) <= a <= r:
            raise IRError(f"vmap: in_axes {a} for {v.ref}: {v.type} is out of range for rank {r} plus the batch dim")
        a %= r + 1
        shape = list(v.type.shape)
        nv = Value(v.name, TensorType(v.type.dtype, tuple(shape[:a] + [n] + shape[a:])), "input")
        e.inputs.append(nv)
        moves.append((v, nv, a))
    for p in prog.params:
        if p.name not in batched_params:
            bt.env[id(p)] = (p, False)
    for v, nv, a in moves:
        if a:
            r = v.type.rank
            nv = e.op("permute", nv, dims=[a] + [i for i in range(r + 1) if i != a])
        bt.env[id(v)] = (nv, True)

    bt.run(prog.nodes)

    if out_axes is None or isinstance(out_axes, int):
        oaxes = [out_axes] * len(prog.outputs)
    else:
        oaxes = list(out_axes)
        if len(oaxes) != len(prog.outputs):
            raise IRError(f"vmap: out_axes has {len(oaxes)} entries for {len(prog.outputs)} outputs")
    outs = []
    for v, a in zip(prog.outputs, oaxes):
        nv, b = bt.env[id(v)]
        if a is None:
            if b:
                raise IRError(f"vmap: output {v.ref} depends on a batched input, so out_axes None cannot hold for it")
            outs.append(nv)
            continue
        nv = bt.full((nv, b))
        r = v.type.rank
        if not -(r + 1) <= a <= r:
            raise IRError(f"vmap: out_axes {a} for output {v.ref}: {v.type} is out of range")
        a %= r + 1
        if a:
            dims = list(range(1, r + 1))
            dims.insert(a, 0)
            nv = e.op("permute", nv, dims=dims)
        outs.append(nv)
    out = Program(name or f"vmap_{prog.name}", e.inputs, e.params, e.param_data, e.nodes, outs, symbols)
    return prune(out)


# ----------------------------------------------------------------------------- rules: elementwise


@batch_rule("neg", "exp", "sqrt", "rsqrt", "tanh", "relu", "gelu", "hardswish", "hardsigmoid", "silu",
            "sigmoid", "clamp", "logical_not", "cos", "sin", "log", "reciprocal", "erf", "expm1", "abs", "cast",
            "full_like", "pad", "triu", "tril", "all_reduce", "frozen")
def _unary(bt: _Batcher, n: Node, ins):
    """Ops that act on each entry, or only on trailing dims (pad, triu, tril): the batch dim in front
    changes nothing. all_reduce (truestar.train) reduces each entry across ranks, so one
    collective on the batched tensor equals one per element."""
    return [(bt.e.op(n.op, ins[0][0], **n.attrs), True)]


def _promote_operands(bt: _Batcher, n: Node, ins, value_positions: Sequence[int]) -> List[Optional[Value]]:
    """Lift batched operands for broadcasting and keep the original result dtype: a 0 dim operand
    does not widen a dtype, but once batched it has a dim, so it is cast to the dtype the original
    promotion gave."""
    rank = n.output.type.rank
    orig = [x.type for x in n.inputs]
    target = None
    vals = [orig[i] for i in value_positions]
    if vals:
        target = vals[0].dtype if len(vals) == 1 else result_dtype(vals[0], vals[1])
        for t in vals[2:]:
            target = torch.promote_types(target, t.dtype)
    args = []
    for i, p in enumerate(ins):
        v = bt.lift(p, rank)
        if i in value_positions and p[1] and orig[i].rank == 0 and v.type.dtype != target:
            v = bt.e.op("cast", v, dtype=target)
        args.append(v)
    return args


@batch_rule("add", "sub", "mul", "div", "eq", "ne", "lt", "le", "gt", "ge", "logical_and", "logical_or",
            "pow", "fmod", "remainder", "maximum", "minimum")
def _binary(bt: _Batcher, n: Node, ins):
    args = _promote_operands(bt, n, ins, [0, 1])
    return [(bt.e.op(n.op, *args, **n.attrs), True)]


@batch_rule("where")
def _where_rule(bt: _Batcher, n: Node, ins):
    args = _promote_operands(bt, n, ins, [1, 2])
    return [(bt.e.op("where", *args), True)]


# ----------------------------------------------------------------------------- rules: dims move right


def _d(d: int, rank: int) -> int:
    return d % max(rank, 1) + 1


def _scalar_case(bt: _Batcher, x: Value, fn: Callable[[Value], List[Value]], ranks: Sequence[int]) -> List[Pair]:
    """Run ``fn`` on a batched rank 0 value viewed as [n, 1] (dim 1), then drop the 1 again from
    results of original rank 0 (``ranks`` gives each result's original rank)."""
    x2 = bt.e.op("reshape", x, shape=[bt.n, 1])
    outs = fn(x2)
    res = []
    for o, r in zip(outs, ranks):
        if r == 0 and o.type.rank == 2:
            o = bt.e.op("reshape", o, shape=[bt.n])
        res.append((o, True))
    return res


@batch_rule("sum", "mean", "any", "amax", "amin")
def _reduce(bt: _Batcher, n: Node, ins):
    x = ins[0][0]
    r = n.inputs[0].type.rank
    keep = n.attrs.get("keepdim", False)
    attrs = {k: v for k, v in n.attrs.items() if k not in ("dims", "keepdim")}
    if r == 0:
        return _scalar_case(bt, x, lambda x2: [bt.e.op(n.op, x2, dims=[1], keepdim=False, **attrs)], [0])
    dims = n.attrs.get("dims")
    dims = list(range(r)) if not dims else list(dims)
    return [(bt.e.op(n.op, x, dims=[_d(d, r) for d in dims], keepdim=keep, **attrs), True)]


@batch_rule("vector_norm")
def _vector_norm(bt: _Batcher, n: Node, ins):
    return _reduce(bt, n, ins)




@batch_rule("softmax", "cumsum", "log_softmax")
def _along_dim(bt: _Batcher, n: Node, ins):
    x = ins[0][0]
    r = n.inputs[0].type.rank
    attrs = dict(n.attrs)
    if r == 0:
        attrs["dim"] = 1
        return _scalar_case(bt, x, lambda x2: [bt.e.op(n.op, x2, **attrs)], [0])
    attrs["dim"] = _d(n.attrs["dim"], r)
    return [(bt.e.op(n.op, x, **attrs), True)]


@batch_rule("sort", "topk")
def _sort(bt: _Batcher, n: Node, ins):
    x = ins[0][0]
    r = n.inputs[0].type.rank
    attrs = dict(n.attrs)
    if r == 0:
        attrs["dim"] = 1
        return _scalar_case(bt, x, lambda x2: bt.e.op_multi(n.op, x2, **attrs), [0, 0])
    attrs["dim"] = _d(n.attrs.get("dim", -1), r)
    return [(o, True) for o in bt.e.op_multi(n.op, x, **attrs)]


@batch_rule("argmax", "argmin")
def _argext(bt: _Batcher, n: Node, ins):
    x = ins[0][0]
    r = n.inputs[0].type.rank
    keep = bool(n.attrs.get("keepdim", False))
    dim = n.attrs.get("dim")
    if dim is None or r == 0:
        flat = bt.e.op("reshape", x, shape=[bt.n, numel(n.inputs[0].type.shape)])
        y = bt.e.op(n.op, flat, dim=1, keepdim=False)
        if keep and r:
            y = bt.e.op("reshape", y, shape=[bt.n] + [1] * r)
        return [(y, True)]
    return [(bt.e.op(n.op, x, dim=_d(dim, r), keepdim=keep), True)]


@batch_rule("flip")
def _flip(bt: _Batcher, n: Node, ins):
    x = ins[0][0]
    r = n.inputs[0].type.rank
    if r == 0:
        return [(bt.e.op("reshape", x, shape=[bt.n]), True)]
    return [(bt.e.op("flip", x, dims=[_d(d, r) for d in n.attrs["dims"]]), True)]


@batch_rule("permute")
def _permute(bt: _Batcher, n: Node, ins):
    r = n.inputs[0].type.rank
    return [(bt.e.op("permute", ins[0][0], dims=[0] + [_d(d, r) for d in n.attrs["dims"]]), True)]


@batch_rule("reshape")
def _reshape(bt: _Batcher, n: Node, ins):
    return [(bt.e.op("reshape", ins[0][0], shape=[bt.n] + list(n.attrs["shape"])), True)]


@batch_rule("expand")
def _expand(bt: _Batcher, n: Node, ins):
    shape = list(n.attrs["shape"])
    x = bt.lift(ins[0], len(shape))
    return [(bt.e.op("expand", x, shape=[bt.n] + shape), True)]


@batch_rule("select", "slice", "select_dynamic")
def _select(bt: _Batcher, n: Node, ins):
    r = n.inputs[0].type.rank
    attrs = dict(n.attrs)
    attrs["dim"] = _d(n.attrs["dim"], r)
    if n.op == "select_dynamic" and ins[1][1]:
        return _select_dynamic_batched_index(bt, n, ins)
    return [(bt.e.op(n.op, *[p[0] for p in ins], **attrs), True)]


def _select_dynamic_batched_index(bt: _Batcher, n: Node, ins):
    """x.select(dim, i[k]) per element k: one advanced index with arange over the batch dim."""
    x = bt.full(ins[0])
    r = n.inputs[0].type.rank
    d = n.attrs["dim"] % r
    idx: List[Optional[Value]] = [bt.arange()] + [None] * d + [ins[1][0]]
    return [(bt.e.op("index", x, *idx), True)]


@batch_rule("cat")
def _cat(bt: _Batcher, n: Node, ins):
    r = n.inputs[0].type.rank
    return [(bt.e.op("cat", *[bt.full(p) for p in ins], dim=_d(n.attrs["dim"], r)), True)]


@batch_rule("select_scatter", "slice_scatter")
def _scatter_view(bt: _Batcher, n: Node, ins):
    r = n.inputs[0].type.rank
    attrs = dict(n.attrs)
    attrs["dim"] = _d(n.attrs["dim"], r)
    return [(bt.e.op(n.op, bt.full(ins[0]), bt.full(ins[1]), **attrs), True)]


@batch_rule("gather")
def _gather(bt: _Batcher, n: Node, ins):
    r = n.inputs[0].type.rank
    return [(bt.e.op("gather", bt.full(ins[0]), bt.full(ins[1]), dim=_d(n.attrs["dim"], r)), True)]


@batch_rule("scatter_add")
def _scatter_add(bt: _Batcher, n: Node, ins):
    r = n.inputs[0].type.rank
    if r == 0:
        return [(bt.e.op("add", bt.full(ins[0]), bt.full(ins[2])), True)]
    return [(bt.e.op("scatter_add", bt.full(ins[0]), bt.full(ins[1]), bt.full(ins[2]),
                     dim=_d(n.attrs["dim"], r)), True)]


@batch_rule("index_copy")
def _index_copy(bt: _Batcher, n: Node, ins):
    if ins[1][1]:
        return bt.loop_fallback(n, ins)  # each element writes its own rows: an exact per element loop
    r = n.inputs[0].type.rank
    return [(bt.e.op("index_copy", bt.full(ins[0]), ins[1][0], bt.full(ins[2]), dim=_d(n.attrs["dim"], r)), True)]




@batch_rule("index_select")
def _index_select(bt: _Batcher, n: Node, ins):
    r = n.inputs[0].type.rank
    d = n.attrs["dim"] % r
    if not ins[1][1]:
        return [(bt.e.op("index_select", ins[0][0], ins[1][0], dim=d + 1), True)]
    x = bt.full(ins[0])
    i = ins[1][0]
    if n.inputs[1].type.rank == 0:
        i = bt.e.op("reshape", i, shape=[bt.n, 1])
    rows = bt.e.op("reshape", bt.arange(), shape=[bt.n, 1])
    y = bt.e.op("index", x, rows, *([None] * d), i)
    if d:  # the index dims were not adjacent, so [n, k] came first: move k back to position d + 1
        dims = [0] + list(range(2, d + 2)) + [1] + list(range(d + 2, r + 1))
        y = bt.e.op("permute", y, dims=dims)
    return [(y, True)]


@batch_rule("index")
def _index(bt: _Batcher, n: Node, ins):
    x_orig = n.inputs[0].type
    idx_orig = list(n.inputs[1:])
    where = [i for i, t in enumerate(idx_orig) if t is not None]
    adjacent = where == list(range(where[0], where[-1] + 1))
    b_rank = max(idx_orig[i].type.rank for i in where)
    idx_batched = any(p is not None and p[1] for p in ins[1:])
    if not idx_batched:
        y = bt.e.op("index", ins[0][0], None, *[None if p is None else p[0] for p in ins[1:]])
        if not adjacent:  # result is [*b, n, *kept]: move n to the front
            dims = [b_rank] + [i for i in range(y.type.rank) if i != b_rank]
            y = bt.e.op("permute", y, dims=dims)
        return [(y, True)]
    x = bt.full(ins[0])
    rows = bt.e.op("reshape", bt.arange(), shape=[bt.n] + [1] * b_rank)
    idx = [rows] + [None if p is None else bt.lift(p, b_rank) for p in ins[1:]]
    y = bt.e.op("index", x, *idx)
    if adjacent and where[0] > 0:
        # original result: before + b + after; batched (indices at 0 and later dims are not adjacent):
        # [n, *b, *before, *after]
        nb = b_rank
        before = where[0]
        dims = [0] + list(range(1 + nb, 1 + nb + before)) + list(range(1, 1 + nb)) + \
            list(range(1 + nb + before, y.type.rank))
        y = bt.e.op("permute", y, dims=dims)
    return [(y, True)]


# ----------------------------------------------------------------------------- rules: linear algebra


@batch_rule("matmul")
def _matmul(bt: _Batcher, n: Node, ins):
    rank = n.output.type.rank
    return [(bt.e.op("matmul", bt.lift(ins[0], rank), bt.lift(ins[1], rank)), True)]


@batch_rule("linear")
def _linear(bt: _Batcher, n: Node, ins):
    x, w, b = ins
    if not w[1] and (b is None or not b[1]):
        return [(bt.e.op("linear", x[0], w[0], None if b is None else b[0]), True)]
    # a batched weight or bias: x w^T + b with matmul, the reference formula of linear
    e = bt.e
    xr = n.inputs[0].type.rank
    xv, xb = x
    if xr == 1:
        xv = e.op("reshape", xv, shape=([bt.n] if xb else []) + [1, n.inputs[0].type.shape[0]])
    rank = max(xr, 2)
    wt = e.op("permute", w[0], dims=[0, 2, 1] if w[1] else [1, 0])
    y = e.op("matmul", bt.lift((xv, xb), rank), bt.lift((wt, w[1]), rank))
    if b is not None:
        y = e.op("add", y, bt.lift(b, rank))
    if xr == 1:
        y = e.op("reshape", y, shape=[bt.n, n.output.type.shape[0]])
    return [(y, True)]


@batch_rule("conv2d")
def _conv2d(bt: _Batcher, n: Node, ins):
    x, w, b = ins
    e = bt.e
    if not w[1] and (b is None or not b[1]):
        y = e.op("conv2d", bt.merge(x[0]), w[0], None if b is None else b[0], **n.attrs)
        return [(bt.split(y, n.inputs[0].type.shape[0]), True)]
    if not isinstance(bt.n, int):
        return bt.loop_fallback(n, ins)  # a per element weight with a symbolic batch: per element loop
    # per element weights with a fixed batch: one grouped conv, element k is groups k*g .. k*g+g-1
    g = n.attrs.get("groups", 1)
    batch, c = n.inputs[0].type.shape[:2]
    xb = bt.full(x)                                                  # [n, B, C, H, W]
    xs = list(xb.type.shape)
    xg = e.op("permute", xb, dims=[1, 0, 2, 3, 4])                   # [B, n, C, H, W]
    xg = e.op("reshape", xg, shape=[xs[1], bt.n * c] + xs[3:])
    wb = bt.full(w)                                                  # [n, O, C/g, kh, kw]
    ws = list(wb.type.shape)
    wg = e.op("reshape", wb, shape=[bt.n * ws[1]] + ws[2:])
    bias = None
    if b is not None:
        bias = e.op("reshape", bt.full(b), shape=[bt.n * ws[1]])
    attrs = dict(n.attrs)
    attrs["groups"] = g * bt.n
    y = e.op("conv2d", xg, wg, bias, **attrs)                        # [B, n*O, H', W']
    ys = list(y.type.shape)
    y = e.op("reshape", y, shape=[ys[0], bt.n, ws[1]] + ys[2:])
    return [(e.op("permute", y, dims=[1, 0, 2, 3, 4]), True)]


@batch_rule("attention")
def _attention(bt: _Batcher, n: Node, ins):
    q, k, v, m = ins
    rank = n.output.type.rank
    if not (q[1] or k[1] or v[1]):  # only the mask is batched: the output batch comes from q, k, v
        q = (bt.full(q), True)

    def lift_all(p):
        val, b = p
        return bt.lift(p, rank) if b else bt.e.op("reshape", val, shape=[1] + list(val.type.shape))
    args = [lift_all(q), lift_all(k), lift_all(v)]
    mask = None if m is None else bt.lift(m, rank)
    return [(bt.e.op("attention", *args, mask, **n.attrs), True)]


# ----------------------------------------------------------------------------- rules: batch like ops


def _lowered(bt: _Batcher, n: Node, ins):
    """Rewrite a composite op into primitives (the verified lowerings of grad_rules.py) and batch
    those. Used when a per element weight or statistic makes the fused op unusable."""
    tmp = Emitter(set(bt.e.names), "v_")
    LOWERINGS[n.op](tmp, n)
    bt.e.names |= tmp.names
    for p in tmp.params:
        bt.e.params.append(p)
        bt.e.param_data[p.name] = tmp.param_data[p.name]
        bt.env[id(p)] = (p, False)
    for x, p in zip(n.inputs, ins):
        if x is not None:
            bt.env[id(x)] = p
    bt.run(tmp.nodes)
    return [bt.env[id(n.output)]]


@batch_rule("batch_norm")
def _batch_norm(bt: _Batcher, n: Node, ins):
    if any(p is not None and p[1] for p in ins[1:]):
        return _lowered(bt, n, ins)
    y = bt.e.op("batch_norm", bt.merge(ins[0][0]), *[None if p is None else p[0] for p in ins[1:]], **n.attrs)
    return [(bt.split(y, n.inputs[0].type.shape[0]), True)]


@batch_rule("layer_norm")
def _layer_norm(bt: _Batcher, n: Node, ins):
    if any(p is not None and p[1] for p in ins[1:]):
        return _lowered(bt, n, ins)
    return [(bt.e.op("layer_norm", ins[0][0], *[None if p is None else p[0] for p in ins[1:]], **n.attrs), True)]


@batch_rule("rms_norm")
def _rms_norm(bt: _Batcher, n: Node, ins):
    if ins[1] is not None and ins[1][1]:
        return bt.loop_fallback(n, ins)
    return [(bt.e.op("rms_norm", ins[0][0], None if ins[1] is None else ins[1][0], **n.attrs), True)]


@batch_rule("group_norm")
def _group_norm(bt: _Batcher, n: Node, ins):
    if any(p is not None and p[1] for p in ins[1:]):
        return bt.loop_fallback(n, ins)
    y = bt.e.op("group_norm", bt.merge(ins[0][0]), *[None if p is None else p[0] for p in ins[1:]], **n.attrs)
    return [(bt.split(y, n.inputs[0].type.shape[0]), True)]




@batch_rule("max_pool2d", "avg_pool2d", "adaptive_avg_pool2d", "adaptive_max_pool2d")
def _pool(bt: _Batcher, n: Node, ins):
    x = ins[0][0]
    if n.inputs[0].type.rank == 3:  # [C, H, W]: the batch dim is the op's own batch dim
        return [(bt.e.op(n.op, x, **n.attrs), True)]
    y = bt.e.op(n.op, bt.merge(x), **n.attrs)
    return [(bt.split(y, n.inputs[0].type.shape[0]), True)]




@batch_rule("max_pool2d_grad")
def _max_pool2d_grad(bt: _Batcher, n: Node, ins):
    x, g = (bt.merge(bt.full(p)) for p in ins)
    y = bt.e.op("max_pool2d_grad", x, g, **n.attrs)
    return [(bt.split(y, n.inputs[0].type.shape[0]), True)]


def _channels_by_element(bt: _Batcher, v: Value) -> Value:
    """[n, N, C, H, W] -> [N, n * C, H, W]: element k owns channels k*C .. k*C+C-1."""
    s = list(v.type.shape)
    v = bt.e.op("permute", v, dims=[1, 0, 2, 3, 4])
    return bt.e.op("reshape", v, shape=[s[1], bt.n * s[2]] + s[3:])


def _channels_back(bt: _Batcher, v: Value) -> Value:
    """[N, n * C, H, W] -> [n, N, C, H, W]."""
    s = list(v.type.shape)
    v = bt.e.op("reshape", v, shape=[s[0], bt.n, s[1] // bt.n] + s[2:])
    return bt.e.op("permute", v, dims=[1, 0, 2, 3, 4])


@batch_rule("conv2d_input_grad")
def _conv2d_input_grad(bt: _Batcher, n: Node, ins):
    g, w = ins
    e = bt.e
    if not w[1]:   # a shared weight: the vmap batch joins the op's own batch dim
        y = e.op("conv2d_input_grad", bt.merge(g[0]), w[0], **n.attrs)
        return [(bt.split(y, n.inputs[0].type.shape[0]), True)]
    if not isinstance(bt.n, int):
        return bt.loop_fallback(n, ins)
    # per element weights: one grouped op, element k owns groups k*G .. k*G+G-1
    ws = list(bt.full(w).type.shape)
    wg = e.op("reshape", bt.full(w), shape=[bt.n * ws[1]] + ws[2:])
    attrs = dict(n.attrs, groups=n.attrs.get("groups", 1) * bt.n)
    y = e.op("conv2d_input_grad", _channels_by_element(bt, bt.full(g)), wg, **attrs)
    return [(_channels_back(bt, y), True)]


@batch_rule("conv2d_weight_grad")
def _conv2d_weight_grad(bt: _Batcher, n: Node, ins):
    """The op sums over its own batch dim, so the vmap batch cannot join it; it joins the channels
    instead, with the groups multiplied (exact: each element only meets its own channels)."""
    if not isinstance(bt.n, int):
        return bt.loop_fallback(n, ins)
    e = bt.e
    x, g = (bt.full(p) for p in ins)
    attrs = dict(n.attrs, groups=n.attrs.get("groups", 1) * bt.n)
    y = e.op("conv2d_weight_grad", _channels_by_element(bt, x), _channels_by_element(bt, g), **attrs)
    s = list(y.type.shape)                                        # [n * O, C/g, kh, kw]
    return [(e.op("reshape", y, shape=[bt.n, s[0] // bt.n] + s[1:]), True)]


# ----------------------------------------------------------------------------- random ops


def _random(bt: _Batcher, n: Node, ins):
    if n.op == "multinomial":
        return bt.loop_fallback(n, ins, offset_operand=1, random_offset=True)
    if n.op == "random" and not any(p[1] for p in ins):
        # element k draws random(key, *streams, k): one vectorized op over the batch
        attrs = dict(n.attrs)
        attrs["shape"] = [bt.n] + list(n.attrs.get("shape", ()))
        return [(bt.e.op("random_batched", *[p[0] for p in ins], **attrs), True)]
    if n.op in ("random", "random_batched"):
        # a per element key or stream, or vmap of vmap: the same definition, one element at a time
        return bt.loop_fallback(n, ins, append_index=True)
    raise UnsupportedOpError(f"vmap has no per element key rule for the random op {n.op} ({_where(n)})")


# ----------------------------------------------------------------------------- control flow


@batch_rule("cond")
def _cond(bt: _Batcher, n: Node, ins):
    pred, ops = ins[0], ins[1:]
    flags = [p[1] for p in ops]
    then, other = n.attrs["then_branch"], n.attrs["else_branch"]
    t_prog, t_out = bt.region(then, flags)
    o_prog, o_out = bt.region(other, flags)
    if not pred[1]:
        force = [a or b for a, b in zip(t_out, o_out)]
        if force != t_out:
            t_prog, t_out = bt.region(then, flags, force)
        if force != o_out:
            o_prog, o_out = bt.region(other, flags, force)
        attrs = dict(n.attrs)
        attrs["then_branch"], attrs["else_branch"] = t_prog, o_prog
        outs = bt.e.op_multi("cond", pred[0], *[p[0] for p in ops], **attrs)
        return list(zip(outs, force))
    # The predicate differs per element: run both branches batched and pick per element with where.
    # Each branch is guarded by "some element takes it" (a cond on any(pred)), so a branch no element
    # takes is not run at all: no wasted work when the batch agrees, and no error or endless loop from
    # a branch that only makes sense for the elements that take it (JAX runs both, jax#2947, #8409).
    args = [p[0] for p in ops]
    some_then = bt.e.op("any", pred[0], dims=None, keepdim=False)
    some_else = bt.e.op("any", bt.e.op("logical_not", pred[0]), dims=None, keepdim=False)
    a = bt.e.op_multi("cond", some_then, *args, then_branch=t_prog, else_branch=_zeros_region(bt, t_prog))
    b = bt.e.op_multi("cond", some_else, *args, then_branch=o_prog, else_branch=_zeros_region(bt, o_prog))
    res = []
    for o, ta, tb_, fa, fb in zip(n.outputs, a, b, t_out, o_out):
        c = bt.lift(pred, o.type.rank)
        res.append((bt.e.op("where", c, bt.lift((ta, fa), o.type.rank), bt.lift((tb_, fb), o.type.rank)), True))
    return res


def _zeros_region(bt: _Batcher, like: Program) -> Program:
    """A region with ``like``'s inputs that returns zeros of ``like``'s output types: the side of a
    guarding cond taken when no element needs the branch (its values are never selected)."""
    e = Emitter(set(), "v_")
    for v in like.inputs:
        e.new_input(v.name, v.type)
    outs = [e.zeros(list(o.type.shape), o.type.dtype) for o in like.outputs]
    return Program("skip_" + like.name, e.inputs, e.params, e.param_data, e.nodes, outs, dict(bt.symbols))


@batch_rule("loop")
def _loop(bt: _Batcher, n: Node, ins):
    k = n.attrs["n_carried"]
    carried, caps = ins[:k], ins[k:]
    cflags = [p[1] for p in carried]
    kflags = [p[1] for p in caps]
    body, cond = n.attrs["body"], n.attrs.get("cond")
    while True:  # carried values the body makes batched become batched: a fixed point
        _, out = bt.region(body, [False] + cflags + kflags)
        new = [a or b for a, b in zip(cflags, out)]
        if new == cflags:
            break
        cflags = new
    pred_batched = False
    if cond is not None:
        _, (pred_batched,) = bt.region(cond, cflags + kflags)
    if pred_batched:
        return _while_batched(bt, n, carried, caps)
    body_b, _ = bt.region(body, [False] + cflags + kflags, cflags)
    attrs = dict(n.attrs)
    attrs["body"] = body_b
    if cond is not None:
        attrs["cond"], _ = bt.region(cond, cflags + kflags)
    init = [bt.full(p) if f else p[0] for p, f in zip(carried, cflags)]
    outs = bt.e.op_multi("loop", *init, *[p[0] for p in caps], **attrs)
    return list(zip(outs, cflags))


def _while_batched(bt: _Batcher, n: Node, carried, caps):
    """A while loop whose condition differs per element: run while any element is still going; an
    element whose condition is false keeps its carried values (it is frozen, so its condition stays
    false and it never restarts). Element k's iteration j is the global iteration j, so the body
    sees the same index as it would alone."""
    k = len(carried)
    kflags = [p[1] for p in caps]
    body, cond = n.attrs["body"], n.attrs["cond"]
    all_b = [True] * k
    body_b, _ = bt.region(body, [False] + all_b + kflags, all_b)
    cond_b, _ = bt.region(cond, all_b + kflags, [True])

    ce = Emitter(set(), "v_")
    c_in = [ce.new_input(v.name, v.type) for v in cond_b.inputs]
    (act,) = ce.inline(cond_b, c_in)
    any_ = ce.op("any", act, dims=None, keepdim=False)
    new_cond = Program("any_" + cond.name, ce.inputs, ce.params, ce.param_data, ce.nodes, [any_], dict(bt.symbols))

    be = Emitter(set(), "v_")
    b_in = [be.new_input(v.name, v.type) for v in body_b.inputs]
    idx, c_vals, k_vals = b_in[0], b_in[1:1 + k], b_in[1 + k:]
    (act,) = be.inline(cond_b, c_vals + k_vals)
    stepped = be.inline(body_b, b_in)
    outs = []
    for old, new in zip(c_vals, stepped):
        rank = old.type.rank - 1
        mask = be.op("reshape", act, shape=[bt.n] + [1] * rank) if rank else act
        outs.append(be.op("where", mask, new, old))
    new_body = Program("masked_" + body.name, be.inputs, be.params, be.param_data, be.nodes, outs, dict(bt.symbols))

    attrs = dict(n.attrs)
    attrs["body"], attrs["cond"] = new_body, new_cond
    init = [bt.full(p) for p in carried]
    res = bt.e.op_multi("loop", *init, *[p[0] for p in caps], **attrs)
    return [(o, True) for o in res]


@batch_rule("custom_vjp")
def _custom_vjp(bt: _Batcher, n: Node, ins):
    """Both regions batched with every operand batched, so the batched backward returns per element
    gradients for every operand; autodiff sums them over the batch through the ``expand`` that
    batched an unbatched operand."""
    ops = [bt.full(p) for p in ins]
    flags = [True] * len(ops)
    fwd, _ = bt.region(n.attrs["fwd"], flags, [True])
    bwd, _ = bt.region(n.attrs["bwd"], flags + [True, True], [True] * len(n.attrs["bwd"].outputs))
    attrs = dict(n.attrs)
    attrs["fwd"], attrs["bwd"] = fwd, bwd
    return [(bt.e.op("custom_vjp", *ops, **attrs), True)]


from . import linattn_ops as _linattn_ops  # noqa: E402  (linear_attention, P60: the exact per element loop)
_linattn_ops.register_vmap(LOOP_FALLBACK)
from . import attention_grad as _attention_grad  # noqa: E402  (the fused attention backward)
_attention_grad.register_vmap(batch_rule)
