"""GPU backend: schedule a kernel into one of four loop shapes and generate Triton code.

The scheduler looks only at the structure of the IR, never at op names, so any expression that
fits a shape compiles. The shapes:

* tile: elementwise math plus reductions that all run over one axis (pointwise chains, softmax,
  layer norm, row sums). A program owns a block of rows; the reduced axis maps to the lanes.
* contraction: one sum of a product of two factors (matmul) with arbitrary elementwise math on
  the factors (prologue) and on the result (epilogue). Uses tl.dot.
* attention: sum over keys of (a linear combination of softmax weights) times values, where each
  softmax score is any elementwise function of one dot product, indices and loads. Runs as one
  online softmax pass (flash attention style). This covers score modifications that FlexAttention
  can express, and combinations it cannot (differential attention is two softmaxes in one kernel).
  Causal or windowed masks written as index comparisons are detected and fully masked key
  blocks are skipped.
* anything else raises Unschedulable, and the fusion pass splits the kernel.
"""

from __future__ import annotations

import contextlib
import contextvars
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import ir, triton_conv
from . import checks as _checks
from .ir import (Axis, Binary, Const, Dim, Expr, Index, Kernel, Load, Placeholder, Reduce,
                 Unary, Where, free_axes, top_reduces)


class Unschedulable(Exception):
    """The kernel does not fit any GPU loop shape. The message says why."""


class ConfigResourceError(ValueError):
    """TS700: a compiled launch config needs more local memory (registers spilled by the
    compiler) than this GPU allows, so it is refused before its first launch instead of failing
    inside the CUDA context. The tuner skips such a candidate; a default or forced config raises."""
    code = "TS700"


LOCAL_MAX_PER_THREAD = 512 * 1024  # bytes; CUDA's limit per thread (Programming Guide, appendix 5.1)
LOCAL_SHARE = 4  # local memory for every resident thread may take at most 1/4 of the GPU's memory


def check_resources(kernels, cfg: Optional[Dict] = None) -> Optional[str]:
    """Why the compiled Triton kernels of one launch cannot run on the current GPU, or None.

    Triton reports a kernel's local memory as ``n_spills`` (CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES / 4).
    The driver reserves that much for every thread that can be resident on every SM, so the check is
    the per thread size against LOCAL_MAX_PER_THREAD and the full reservation against the GPU's
    memory.

    On a ROCm build of torch (``torch.version.hip``) Triton's HIP driver reports the same attribute
    (HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, scratch per lane); the per lane limit is then the
    target's largest scratch per wave divided by its wave size (kernel/rocm.py), and the units are
    CUs."""
    import torch
    for k in kernels:
        local = int(getattr(k, "n_spills", 0) or 0) * 4
        if local == 0:
            continue
        what = f"launch config {cfg}" if cfg is not None else "kernel"
        limit, who, unit = LOCAL_MAX_PER_THREAD, "CUDA allows", "SMs"
        if getattr(torch.version, "hip", None):
            from . import rocm
            t = rocm.current_target()
            limit, who, unit = t.scratch_per_lane, f"{t.arch} allows (scratch per lane)", "CUs"
        if local > limit:
            return (f"{what} needs {local} bytes of local memory per thread (registers spilled), "
                    f"above the {limit} {who}")
        pr = torch.cuda.get_device_properties(torch.cuda.current_device())
        total = local * pr.max_threads_per_multi_processor * pr.multi_processor_count
        if total > pr.total_memory // LOCAL_SHARE:
            return (f"{what} needs {local} bytes of local memory per thread, {total / 2 ** 20:.0f} MiB "
                    f"for every resident thread of {pr.multi_processor_count} {unit}, above 1/{LOCAL_SHARE} "
                    f"of the GPU's {pr.total_memory / 2 ** 30:.1f} GiB")
    return None


def _pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


# =========================================================================== plans


@dataclass
class TilePlan:
    kind = "tile"
    axes: Tuple[Axis, ...]          # output axes
    lane: Optional[Axis]            # output axis mapped to lanes (None: outputs are per row)
    rows: Tuple[Axis, ...]          # output axes mapped to rows
    reduces: List[Reduce]           # in dependency order (inner first)
    rsize: Optional[ir.Size]        # size of every reduced axis (None for pointwise)


@dataclass
class ContractionPlan:
    kind = "contraction"
    red: Reduce
    a: Expr
    b: Expr
    m: Axis
    n: Axis
    k: Axis
    batch: Tuple[Axis, ...]
    tf32: bool = False  # float32 operands may use TF32 tensor cores (set by generate from opts)
    deterministic: bool = False  # no split K: results independent of M and the launch config
    # batch axes the right factor does not depend on (a linear layer on [B, T, C] activations),
    # folded into the rows: one program tile covers rows of several batch entries, so a decode
    # step (T = 1) does not run one 16 row tile per sequence, each reading the whole weight
    fold: Tuple[Axis, ...] = ()
    # deterministic strict float32 with a long K: the K range is summed in chunks of this many
    # elements, each from zero, and the chunk sums are added in the order 0, 1, 2, ... (det_k_chunk)
    kchunk: int = 0


@dataclass
class Term:
    coef: Expr
    score: Expr       # S with the dot product replaced by Placeholder("dot")
    q: Expr
    kk: Expr
    c: Axis           # contraction axis of the dot product
    full_score: Expr  # S as written


@dataclass
class AttentionPlan:
    kind = "attention"
    red: Reduce
    v: Expr
    i: Axis
    j: Axis
    d: Axis
    batch: Tuple[Axis, ...]
    terms: List[Term]
    lo_off: Optional[int] = None   # first key = start_m + lo_off (block aligned)
    hi_off: Optional[int] = None   # last key + 1 = start_m + BM + hi_off
    deterministic: bool = False    # split KV chunks of a fixed size (DET_KV_CHUNK), never chosen
                                   # from the SM count, batch or query length


def plan(k: Kernel):
    axes0 = k.outputs[0].axes
    for o in k.outputs[1:]:
        if o.axes != axes0:
            raise Unschedulable("outputs of one GPU kernel must share the same axes")
    reds = k.reduces()
    if not reds:
        lane = axes0[-1] if axes0 else None
        return TilePlan(axes0, lane, tuple(a for a in axes0 if a != lane), [], None)
    errors = []
    for fn in (_plan_attention, _plan_contraction, _plan_tile):
        try:
            return fn(k, reds)
        except Unschedulable as e:
            errors.append(f"{fn.__name__[6:]}: {e}")
    # convolution and pooling shapes (triton_conv.py), tried last so they never take a kernel
    # one of the shapes above accepts
    for fn in (triton_conv.plan_conv, triton_conv.plan_window):
        try:
            return fn(k)
        except Unschedulable as e:
            errors.append(f"{fn.__name__[5:]}: {e}")
    raise Unschedulable("; ".join(errors))


def _plan_tile(k: Kernel, reds: List[Reduce]) -> TilePlan:
    axes = k.outputs[0].axes
    sizes = {r.axis.size for r in reds}
    if len(sizes) != 1:
        raise Unschedulable(f"reductions run over axes of different sizes {sorted(map(str, sizes))}")
    rsize = next(iter(sizes))
    red_axes = {r.axis for r in reds}
    # every reduction result must be one value per row: it may not depend on a reduced axis
    for r in reds:
        if free_axes(r) & red_axes:
            raise Unschedulable("a reduction depends on another reduction's axis (nested contraction)")
    for r in reds:
        if _is_outer_product(r):
            raise Unschedulable("a reduction is a matrix product (each output would redo a "
                                "whole row of work with no reuse)")
    cands = [a for a in reversed(axes) if a.size == rsize]
    for lane in cands + [None]:
        if all(lane not in free_axes(r) for r in reds):
            rows = tuple(a for a in axes if a != lane)
            return TilePlan(axes, lane, rows, reds, rsize)
    raise Unschedulable("no axis assignment fits")  # unreachable: lane None always fits


def _is_outer_product(r: Reduce) -> bool:
    """True if two loads along the reduced axis each miss an output axis the other has."""
    F = free_axes(r)
    if len(F) < 2:
        return False
    sets = []
    for n in ir.walk(r.body):
        if isinstance(n, Load):
            ax = {a for a in n.index if isinstance(a, Axis)}
            if r.axis in ax:
                sets.append(ax & F)
    for s1 in sets:
        for s2 in sets:
            if (s1 - s2) and (s2 - s1):
                return True
    return False


def _split_product(body: Expr) -> Optional[Tuple[Expr, Expr]]:
    if isinstance(body, Binary) and body.op == "mul":
        return body.a, body.b
    return None


def _plan_contraction(k: Kernel, reds: List[Reduce]) -> ContractionPlan:
    if len(k.outputs) != 1:
        raise Unschedulable("contraction kernels have one output")
    o = k.outputs[0]
    tops = top_reduces(o.expr)
    if len(tops) != 1 or len(reds) != 1:
        raise Unschedulable("needs exactly one reduction")
    red = tops[0]
    if red.op != "sum":
        raise Unschedulable("the reduction is not a sum")
    ab = _split_product(red.body)
    if ab is None:
        raise Unschedulable("the summed body is not a product")
    a, b = ab
    P = set(o.axes)
    fa, fb = free_axes(a), free_axes(b)
    # the tile axes are the last output axes found only in one factor; the rest are batch axes
    ms = [x for x in o.axes if x in fa and x not in fb]
    ns = [x for x in o.axes if x in fb and x not in fa]
    if not ms or not ns:
        raise Unschedulable("needs an output axis only in the left factor and one only in the right")
    m, n = ms[-1], ns[-1]
    batch = tuple(x for x in o.axes if x not in (m, n))
    if not fa <= set(batch) | {m, red.axis} or not fb <= set(batch) | {n, red.axis}:
        raise Unschedulable("a factor depends on an axis it cannot")
    fold = tuple(x for x in batch if x not in fb)
    if len(fold) != len(batch):
        fold = ()       # only all or nothing: the program grid keeps one batch axis or none
    return ContractionPlan(red, a, b, m, n, red.axis, tuple(x for x in batch if x not in fold), fold=fold)


def _softmax_terms(w: Expr, j: Axis):
    """Decompose w into [(coef, S, M, L)] with w = sum coef * exp(S - M) / L, or None."""
    if isinstance(w, Binary) and w.op == "div":
        num, den = w.a, w.b
        if (isinstance(num, Unary) and num.op == "exp" and isinstance(num.x, Binary)
                and num.x.op == "sub" and isinstance(num.x.b, Reduce)
                and isinstance(den, Reduce)):
            s, mx, l = num.x.a, num.x.b, den
            if (mx.op == "max" and l.op == "sum"
                    and mx.body == ir.substitute(s, {j: mx.axis})
                    and l.body == ir.exp(ir.substitute(s, {j: l.axis}) - mx)
                    and j in free_axes(s)):
                return [(Const(1.0), s, mx, l)]
        if j not in free_axes(den) and not ir.contains_reduce(den):
            t = _softmax_terms(num, j)
            return None if t is None else [(c / den, s, m, l) for c, s, m, l in t]
        return None
    if isinstance(w, Binary) and w.op == "mul":
        for x, y in ((w.a, w.b), (w.b, w.a)):
            if j not in free_axes(y) and not ir.contains_reduce(y):
                t = _softmax_terms(x, j)
                if t is not None:
                    return [(c * y if not _is_one(c) else y, s, m, l) for c, s, m, l in t]
        return None
    if isinstance(w, Binary) and w.op in ("add", "sub"):
        ta, tb = _softmax_terms(w.a, j), _softmax_terms(w.b, j)
        if ta is None or tb is None:
            return None
        if w.op == "sub":
            tb = [(-c, s, m, l) for c, s, m, l in tb]
        return ta + tb
    if isinstance(w, Unary) and w.op == "neg":
        t = _softmax_terms(w.x, j)
        return None if t is None else [(-c, s, m, l) for c, s, m, l in t]
    return None


def _is_one(e):
    return isinstance(e, Const) and e.value == 1.0


def _plan_attention(k: Kernel, reds: List[Reduce]) -> AttentionPlan:
    if len(k.outputs) != 1:
        raise Unschedulable("attention kernels have one output")
    o = k.outputs[0]
    tops = top_reduces(o.expr)
    if len(tops) != 1 or tops[0].op != "sum":
        raise Unschedulable("needs one top level sum over keys")
    red = tops[0]
    ab = _split_product(red.body)
    if ab is None:
        raise Unschedulable("the summed body is not weights times values")
    j = red.axis
    w, v = (ab if ir.contains_reduce(ab[0]) else (ab[1], ab[0]))
    if ir.contains_reduce(v) or not ir.contains_reduce(w):
        raise Unschedulable("no softmax weight found")
    P = set(o.axes)
    fw, fv = free_axes(w), free_axes(v)
    i_s = (fw & P) - fv
    d_s = (fv & P) - fw
    if len(i_s) != 1 or len(d_s) != 1:
        raise Unschedulable("needs one query axis in the weights and one value axis in the values")
    i, d = next(iter(i_s)), next(iter(d_s))
    batch = tuple(x for x in o.axes if x not in (i, d))
    B = set(batch)
    if not fw <= B | {i, j} or not fv <= B | {j, d}:
        raise Unschedulable("weights or values depend on an axis they cannot")
    raw = _softmax_terms(w, j)
    if raw is None:
        raise Unschedulable("weights are not a linear combination of softmaxes over the key axis")
    merged: Dict[Expr, Expr] = {}
    order: List[Expr] = []
    for c, s, _, _ in raw:
        if s in merged:
            merged[s] = merged[s] + c
        else:
            merged[s] = c
            order.append(s)
    terms = []
    for s in order:
        coef = merged[s]
        if not free_axes(coef) <= B | {i}:
            raise Unschedulable("a softmax coefficient depends on the key or value axis")
        dots = top_reduces(s)
        if len(dots) != 1 or dots[0].op != "sum":
            raise Unschedulable("each score must contain exactly one dot product")
        dot = dots[0]
        qk = _split_product(dot.body)
        if qk is None:
            raise Unschedulable("the score's reduction is not a dot product")
        q, kk = qk if j not in free_axes(qk[0]) else (qk[1], qk[0])
        c = dot.axis
        if not free_axes(q) <= B | {i, c} or not free_axes(kk) <= B | {j, c}:
            raise Unschedulable("query or key factor depends on an axis it cannot")
        if ir.contains_reduce(q) or ir.contains_reduce(kk):
            raise Unschedulable("query or key factor contains a reduction")
        smod = ir.substitute(s, {dot: Placeholder("dot")})
        if ir.contains_reduce(smod):
            raise Unschedulable("the score contains a second reduction")
        terms.append(Term(coef, smod, q, kk, c, s))
    p = AttentionPlan(red, v, i, j, d, batch, terms)
    _mask_bounds(p)
    return p


def _affine(e: Expr, i: Axis, j: Axis):
    """e as {i: a, j: b, None: c} if it is affine in the two indices, else None."""
    if isinstance(e, Const) and not isinstance(e.value, bool):
        return {None: e.value}
    if isinstance(e, Index) and e.axis in (i, j):
        return {e.axis: 1.0}
    if isinstance(e, Binary) and e.op in ("add", "sub"):
        a, b = _affine(e.a, i, j), _affine(e.b, i, j)
        if a is None or b is None:
            return None
        sgn = 1.0 if e.op == "add" else -1.0
        out = dict(a)
        for key, val in b.items():
            out[key] = out.get(key, 0.0) + sgn * val
        return out
    if isinstance(e, Unary) and e.op == "neg":
        a = _affine(e.x, i, j)
        return None if a is None else {key: -val for key, val in a.items()}
    return None


def _conjuncts(c: Expr):
    if isinstance(c, Binary) and c.op == "and":
        return _conjuncts(c.a) + _conjuncts(c.b)
    return [c]


def _cond_bound(c: Expr, i: Axis, j: Axis):
    """(l, h) for one comparison: the key j is allowed iff i + l <= j (l) and j <= i + h (h).
    Either is None when the comparison gives no such bound; the result is None when the
    comparison is not an affine index comparison of the form j OP i + const."""
    if not (isinstance(c, Binary) and c.op in ("le", "lt", "ge", "gt")):
        return None
    a, b = _affine(c.a, i, j), _affine(c.b, i, j)
    if a is None or b is None:
        return None
    diff = dict(a)  # lhs - rhs
    for key, val in b.items():
        diff[key] = diff.get(key, 0.0) - val
    cj, ci, c0 = diff.get(j, 0.0), diff.get(i, 0.0), diff.get(None, 0.0)
    op = c.op
    if cj == -1.0:
        cj, ci, c0 = 1.0, -ci, -c0
        op = {"le": "ge", "lt": "gt", "ge": "le", "gt": "lt"}[op]
    if cj != 1.0 or ci != -1.0 or c0 != int(c0):
        return None
    c0 = int(c0)  # j - i + c0 OP 0, so j OP i - c0
    h = {"le": -c0, "lt": -c0 - 1}.get(op)
    l = {"ge": -c0, "gt": -c0 + 1}.get(op)
    return l, h


def _mask_chain(score: Expr):
    """Conjuncts of the where(cond, s, -inf) chain at the top of a score."""
    conds = []
    e = score
    while isinstance(e, Where) and ir.is_neg_inf(e.b):
        conds += _conjuncts(e.cond)
        e = e.a
    return conds


def _term_bounds(score: Expr, i: Axis, j: Axis):
    """Key range allowed by index comparisons that send the score to -inf when false:
    (lo, hi) with lo the offset of the first key from start_m and hi the offset of the last key
    plus one from start_m + BM (each None when unbounded)."""
    lo = hi = None
    for c in _mask_chain(score):
        b = _cond_bound(c, i, j)
        if b is None:
            continue
        l, h = b
        if h is not None:
            hi = h if hi is None else min(hi, h)
        if l is not None:
            lo = l if lo is None else max(lo, l)
    return lo, hi


def _strip_bounds(score: Expr, i: Axis, j: Axis) -> Expr:
    """The score with every analysed index comparison removed from its top mask chain. Equal to
    the score on key blocks where all those comparisons hold for every row (interior blocks)."""
    if isinstance(score, Where) and ir.is_neg_inf(score.b):
        inner = _strip_bounds(score.a, i, j)
        keep = [c for c in _conjuncts(score.cond) if _cond_bound(c, i, j) is None]
        if not keep:
            return inner
        cond = keep[0]
        for c in keep[1:]:
            cond = Binary("and", cond, c)
        return Where(cond, inner, score.b)
    return score


def _interior_bounds(p: "AttentionPlan"):
    """(l, h): key blocks within [start_m + BM - 1 + l, start_m + h] need no index mask for any
    term. None when no term bounds that side."""
    ls, hs = [], []
    for t in p.terms:
        for c in _mask_chain(t.score):
            b = _cond_bound(c, p.i, p.j)
            if b is None:
                continue
            if b[0] is not None:
                ls.append(b[0])
            if b[1] is not None:
                hs.append(b[1])
    return (max(ls) if ls else None), (min(hs) if hs else None)


def _mask_bounds(p: AttentionPlan):
    los, his = [], []
    for t in p.terms:
        lo, hi = _term_bounds(t.score, p.i, p.j)
        los.append(lo)
        his.append(hi)
    # the loop must cover every term, so take the loosest bound; one unbounded term disables it
    p.lo_off = None if any(x is None for x in los) else min(los)
    p.hi_off = None if any(x is None for x in his) else max(his)


LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


def _scale_expr(e: Expr, c: float) -> Expr:
    """c * e with the multiplication folded into constants where the structure allows (a score
    written as dot * scale becomes dot * (scale * c), masks keep their -inf)."""
    if isinstance(e, Const) and not isinstance(e.value, bool):
        return Const(e.value * c)
    if isinstance(e, Where):
        return Where(e.cond, _scale_expr(e.a, c), _scale_expr(e.b, c))
    if isinstance(e, Unary) and e.op == "neg":
        return Unary("neg", _scale_expr(e.x, c))
    if isinstance(e, Binary) and e.op == "mul":
        if isinstance(e.b, Const) and not isinstance(e.b.value, bool):
            return Binary("mul", e.a, Const(e.b.value * c))
        if isinstance(e.a, Const) and not isinstance(e.a.value, bool):
            return Binary("mul", Const(e.a.value * c), e.b)
    if isinstance(e, Binary) and e.op in ("add", "sub"):
        return Binary(e.op, _scale_expr(e.a, c), _scale_expr(e.b, c))
    return Binary("mul", e, Const(c))


# =========================================================================== code emission


_NEUTRAL = {"sum": "0.0", "max": "float('-inf')", "min": "float('inf')"}


class Ctx:
    """How each axis is indexed in the current block: axis -> (index code, mask code or None)."""

    def __init__(self, m: Dict[Axis, Tuple[str, Optional[str]]], env_syms: Dict[str, str]):
        self.m = dict(m)
        self.syms = env_syms

    def with_(self, **_):
        return self

    def extend(self, extra: Dict[Axis, Tuple[str, Optional[str]]]) -> "Ctx":
        m = dict(self.m)
        m.update(extra)
        return Ctx(m, self.syms)


class Emitter:
    def __init__(self, k: Kernel, opts: Dict):
        self.k = k
        self.opts = opts
        self.lines: List[str] = []
        self.ind = 1
        self.scopes: List[Dict[str, str]] = [{}]
        self.n = 0
        self.values: Dict[Expr, str] = {}   # reductions and placeholders already computed
        # math dtype: float32, or float64 for a kernel that writes a float64 output (a float64 island
        # in a float32 program: compiler/lower.py is_island). Every other kernel is unchanged.
        self.cdt = "tl.float64" if any(o.dtype == "float64" for o in k.outputs) else "tl.float32"
        # compiled in checks (kernel/checks.py): classes this kernel flags into p_chk
        self.checks = frozenset(c for c in str(opts.get("checks", "")).split(",") if c)
        # buffers checked where they are loaded: "b0:nan;b3:nan+inf" (canonical names)
        self.load_checks: Dict[str, List[str]] = {}
        for part in str(opts.get("check_loads", "")).split(";"):
            if part:
                name, cls = part.split(":")
                self.load_checks[name] = cls.split("+")

    # ---- lines and variables
    def line(self, s: str):
        self.lines.append("    " * self.ind + s)

    def var(self, code: str, prefix="t") -> str:
        for sc in reversed(self.scopes):
            if code in sc:
                return sc[code]
        name = f"{prefix}{self.n}"
        self.n += 1
        self.line(f"{name} = {code}")
        self.scopes[-1][code] = name
        return name

    def fresh(self, prefix: str) -> str:
        name = f"{prefix}{self.n}"
        self.n += 1
        return name

    def push(self):
        self.scopes.append({})
        self.ind += 1

    def pop(self):
        self.scopes.pop()
        self.ind -= 1

    # ---- sizes
    def size(self, s) -> str:
        return str(s) if isinstance(s, int) else f"S_{s.name}"

    # ---- expressions
    def load(self, n: Load, ctx: Ctx, raw: bool = False) -> str:
        if n.is_strided:
            return triton_conv.emit_strided_load(self, n, ctx, raw)
        b = n.buffer
        off, masks = [], []
        for d, a in enumerate(n.index):
            if isinstance(a, Axis):
                idx, mk = ctx.m[a]
                off.append(f"({idx}) * st_{b.name}_{d}")
                if mk and mk not in masks:
                    masks.append(mk)
            elif a:
                off.append(f"{a} * st_{b.name}_{d}")
        offs = " + ".join(off) if off else "0"
        mask = " & ".join(masks)
        code = f"tl.load(p_{b.name} + {offs}" + (f", mask={mask}, other=0.0)" if mask else ")")
        fresh = not any(code in sc for sc in self.scopes)
        v = self.var(code, "ld")
        if fresh and self.load_checks.get(b.name):
            # a buffer written by a torch step (convolution, fallback) is checked where a kernel
            # reads it: a predicated store, no extra pass over the tensor (kernel/checks.py)
            zero = self.var(f"tl.zeros_like({v}).to(tl.int32)", "cz")
            sel = f" & ({mask})" if mask else ""
            for c in self.load_checks[b.name]:
                bad = f"({v} != {v})" if c == "nan" else f"(tl.abs({v}.to(tl.float32)) == float('inf'))"
                self.line(f"tl.store(p_chk + {_checks.SLOT[c]} + {zero}, 1, mask={bad}{sel})")
        if raw:
            return v
        return self.var(f"{v}.to({self.cdt})")

    def ex(self, e: Expr, ctx: Ctx) -> str:
        if e in self.values:
            return self.values[e]
        if isinstance(e, Const):
            if isinstance(e.value, bool):
                return "True" if e.value else "False"
            v = e.value
            if math.isinf(v):
                return "float('inf')" if v > 0 else "float('-inf')"
            if math.isnan(v):
                return "float('nan')"
            return repr(v)
        if isinstance(e, Dim):
            return self.size(e.size)
        if isinstance(e, Index):
            return f"({ctx.m[e.axis][0]})"
        if isinstance(e, Load):
            return self.load(e, ctx)
        if isinstance(e, Unary):
            x = self.ex(e.x, ctx)
            code = {
                "neg": f"-({x})", "exp": f"tl.exp({x})", "log": f"tl.log({x})",
                "sqrt": f"tl.sqrt({x})", "rsqrt": f"tl.rsqrt({x})",
                "sigmoid": f"tl.sigmoid({x})", "tanh": f"libdevice.tanh({x})",
                "abs": f"tl.abs({x})", "not": f"~({x})",
                "erf": f"tl.erf({x})", "cos": f"libdevice.cos({x})", "sin": f"libdevice.sin({x})",
                "expm1": f"libdevice.expm1({x})",
            }[e.op]
            if self.opts.get("math") == "round_once":
                # docs/ROUND_ONCE.md: float64 evaluation rounded once; sqrt is IEEE in float32
                if e.op in _ROUND_ONCE:
                    code = f"_ro_{e.op}({x})"
                elif e.op == "sqrt":
                    code = f"tl.sqrt_rn({x})"
            if e.op in self.opts.get("_break", {}):
                code = self.opts["_break"][e.op](x)
            return self.var(code)
        if isinstance(e, Binary):
            a, b = self.ex(e.a, ctx), self.ex(e.b, ctx)
            sym = {"add": "+", "sub": "-", "mul": "*", "div": "/", "lt": "<", "le": "<=",
                   "gt": ">", "ge": ">=", "eq": "==", "ne": "!=", "and": "&", "or": "|"}
            if e.op in sym:
                code = f"({a}) {sym[e.op]} ({b})"
            elif e.op == "maximum":
                code = f"tl.maximum({a}, {b})"
            else:
                code = f"tl.minimum({a}, {b})"
            if e.op in self.opts.get("_break", {}):
                code = self.opts["_break"][e.op](a, b)
            if e.op == "div" and "div" in self.checks and _checks.may_be_zero(e.b):
                self.check_div(e.b, b, ctx)
            return self.var(code)
        if isinstance(e, Where):
            c, a, b = self.ex(e.cond, ctx), self.ex(e.a, ctx), self.ex(e.b, ctx)
            return self.var(f"tl.where({c}, {a}, {b})")
        if isinstance(e, Reduce):
            raise Unschedulable("internal: reduction used before it was computed")
        if isinstance(e, Placeholder):
            raise Unschedulable(f"internal: placeholder {e.tag} has no value")
        raise Unschedulable(f"cannot emit {type(e).__name__}")

    def check_div(self, den: Expr, b: str, ctx: Ctx):
        """Flag a zero denominator of a division (check class div) among the valid entries of
        the block: a store predicated on the zero entries, so nothing is written when none is."""
        key = f"chk_div::{b}"
        for sc in reversed(self.scopes):
            if key in sc:
                return
        self.scopes[-1][key] = ""
        try:
            axes = free_axes(den)
        except AttributeError:      # a placeholder inside (attention score): not a user division
            return
        masks = []
        for a in axes:
            mk = ctx.m.get(a, (None, None))[1]
            if mk and mk not in masks:
                masks.append(mk)
        bad = f"(({b}) == 0)" + "".join(f" & ({m})" for m in masks)
        slot = _checks.SLOT["div"]
        if not axes:
            self.line(f"tl.store(p_chk + {slot}, 1, mask={bad})")
        else:
            zero = self.var(f"tl.zeros_like({b}).to(tl.int32)", "cz")
            self.line(f"tl.store(p_chk + {slot} + {zero}, 1, mask={bad})")

    def fn(self, op: str, *xs: str) -> str:
        """Code for a math function used by a loop shape itself (respects the test break hook)."""
        if self.opts.get("math") == "round_once":
            from .roundonce import RoundOnceError
            raise RoundOnceError("TS1093", f"kernel {self.k.name}: this loop shape computes {op} "
                                           f"itself (online softmax) and has no round once form")
        brk = self.opts.get("_break", {})
        if op in brk:
            return brk[op](*xs)
        if op == "exp2" and "exp" in brk:  # a broken exp also breaks the base 2 form
            return brk["exp"](f"({xs[0]}) * {LN2!r}")
        return {"exp": f"tl.exp({xs[0]})", "exp2": f"tl.exp2({xs[0]})"}[op]

    def full(self, v: str, e: Expr, need, shape: str, dt: str, prefix: str = "t") -> str:
        """v broadcast to the block shape; free when e already varies along every block axis."""
        if set(need) <= free_axes(e) and _sched(self.opts, "lean"):
            return v
        return self.var(f"{v} + tl.zeros({shape}, {dt})", prefix)

    def operand(self, e: Expr, ctx: Ctx, dt: str, zero_mask: Optional[str]) -> str:
        """Value for a tl.dot operand in dtype dt, zero outside zero_mask."""
        if isinstance(e, Load) and e.buffer.dtype == dt:
            return self.load(e, ctx, raw=True)  # masked loads already give 0 outside
        v = self.ex(e, ctx)
        if zero_mask:
            v = self.var(f"tl.where({zero_mask}, {v}, 0.0)")
        return self.var(f"({v}).to({_TL[dt]})")


_TL = {"float16": "tl.float16", "bfloat16": "tl.bfloat16", "float32": "tl.float32",
       "float64": "tl.float64"}


def _dot_dtype(*exprs: Expr) -> str:
    dts = {n.buffer.dtype for e in exprs for n in ir.walk(e) if isinstance(n, Load)}
    if len(dts) == 1 and next(iter(dts)) in ("float16", "bfloat16"):
        return next(iter(dts))
    return "float32"


def _precision_arg(dt: str, opts) -> str:
    if dt != "float32":
        return ""
    return ", input_precision=\"tf32\"" if opts.get("allow_tf32") else ", input_precision=\"ieee\""


def _exact_tc(dt: str, opts) -> bool:
    """True when a dot product of operand dtype dt runs as an exact tensor core block
    (portable_tensor_cores, kernel/tcexact.py): half precision, or float32 with TF32 allowed."""
    return bool(opts.get("portable_tc")) and (dt in ("float16", "bfloat16")
                                              or (dt == "float32" and bool(opts.get("allow_tf32"))))


def _exact_dot(a: str, b: str, kb: str, dt: str) -> str:
    """Source of one exact tensor core block a (BM, kb) @ b (kb, BN), float32 result."""
    return f"_tsx_xdot({a}, {b}, {kb}, {dt == 'float32'})"


def _signature(k: Kernel, checks: bool = False) -> List[str]:
    args = []
    for b in sorted(k.inputs, key=lambda b: b.name):
        args.append(f"p_{b.name}")
    for o in k.outputs:
        args.append(f"p_{o.name}")
    if checks:
        args.append("p_chk")     # int32 flag words, one per check class (kernel/checks.py)
    for b in sorted(k.inputs, key=lambda b: b.name):
        args += [f"st_{b.name}_{d}" for d in range(len(b.shape))]
    for o in k.outputs:
        args += [f"st_{o.name}_{d}" for d in range(len(o.axes))]
    args += [f"S_{s.name}" for s in sorted(k.symbols(), key=lambda s: s.name)]
    return args


def _jit_line(sig: Sequence[str]) -> str:
    """The triton.jit decorator for a kernel with this signature.

    Triton by default compiles a new variant when an integer argument is 1, or when it changes
    between divisible by 16 and not. A symbolic size that moves from 1 to 7 to 64 would then pay
    three Triton compiles for one Truestar kernel (measured: 1.0 to 1.7 s each on an MLP). Sizes
    are therefore never specialized. Strides and pointers keep Triton's default: measured on the
    same MLP, dropping their alignment specialization made the matmul kernels 1.6x slower.
    """
    names = [x.split(":")[0].strip() for x in sig if "constexpr" not in x]
    sizes = [n for n in names if n.startswith("S_") or n in ("ROWS", "LANES")]
    return f"@triton.jit(do_not_specialize={sizes!r})"


def _decompose(em: Emitter, axes: Sequence[Axis], flat: str) -> Dict[Axis, str]:
    """Row major decomposition of a flat index into one index per axis."""
    out = {}
    rest = flat
    for a in reversed(axes):
        out[a] = em.var(f"{rest} % {em.size(a.size)}", "ix")
        rest = em.var(f"{rest} // {em.size(a.size)}", "ix")
    return out


def _store(em: Emitter, o: ir.Output, ctx: Ctx, val: str, shape: str, mask: str,
           full: bool = False):
    """Store val; full says val already has the block shape (no broadcast needed)."""
    off = " + ".join(f"({ctx.m[a][0]}) * st_{o.name}_{d}" for d, a in enumerate(o.axes)) or "0"
    v = val if full else em.var(f"{val} + tl.zeros({shape}, {em.cdt})")
    _store_checks(em, o, v, mask)
    em.line(f"tl.store(p_{o.name} + {off}, {v}.to({_TL[o.dtype]}), mask={mask})")


def _store_checks(em: Emitter, o: ir.Output, v: str, mask: str):
    """Compiled in checks on a stored block (kernel/checks.py): a store into the class's flag word,
    predicated on the bad entries themselves. No reduction and no change of the accumulator's
    layout, and when nothing is bad the predicate is false everywhere and nothing is written.
    (A block reduction to one bit per class was measured first: it made float16 contraction
    kernels several times slower.)"""
    S = _checks.SLOT
    zero = em.var(f"tl.zeros_like({v}).to(tl.int32)", "cz")
    if "nan" in em.checks:
        em.line(f"tl.store(p_chk + {S['nan']} + {zero}, 1, mask=({v} != {v}) & ({mask}))")
    if "inf" in em.checks:
        em.line(f"tl.store(p_chk + {S['inf']} + {zero}, 1, mask=(tl.abs({v}) == float('inf')) & ({mask}))")
    if "overflow" in em.checks and o.dtype in ("float16", "bfloat16"):
        vc = em.var(f"tl.abs({v}.to({_TL[o.dtype]}).to(tl.float32))", "ov")
        em.line(f"tl.store(p_chk + {S['overflow']} + {zero}, 1, mask=({vc} == float('inf')) & "
                f"(tl.abs({v}) < float('inf')) & ({mask}))")


# ---------------------------------------------------------------- tile kernels


def _tile_full(p: TilePlan, o: ir.Output) -> bool:
    """True when an output expression already spans the (rows, lanes) block."""
    F = free_axes(o.expr)
    return (p.lane is None or p.lane in F) and any(a in F for a in p.rows)


def _emit_osum(em: "Emitter", r: Reduce, ctx: "Ctx", looped: bool, R: str) -> str:
    """An order free sum per row (kernel/ordersum.py has the definition and proofs): the row's
    anchor exponent (a max of exponent fields), then every float32 term as an exact int64 on the
    anchor's grid, an int64 sum (associative: any block size or order gives the same total), and
    one rounding to float32. Looped rows evaluate the body twice (one pass per step)."""
    def terms():
        body = em.ex(r.body, ctx.extend({r.axis: ("lane", "lmask")}))
        x = em.var(f"tl.where(lmask, ({body}) + tl.zeros((BR, BC), tl.float32), 0.0)", "ox")
        bits = em.var(f"{x}.to(tl.int32, bitcast=True)", "ob")
        e = em.var(f"({bits} >> 23) & 255", "oe")
        return bits, e

    def loop_head():
        em.line(f"for off in range(0, {R}, BC):")
        em.push()
        em.line("lane = off + tl.arange(0, BC)[None, :]")
        em.line(f"lmask = lane < {R}")

    emax = em.fresh("oemax")
    if looped:
        em.line(f"{emax} = tl.full((BR, 1), 1, tl.int32)")
        loop_head()
        _, e = terms()
        em.line(f"{emax} = tl.maximum({emax}, tl.max(tl.where({e} == 255, 1, tl.maximum({e}, 1)), "
                f"axis=1, keep_dims=True))")
        em.pop()
    acc = em.fresh("oacc")
    em.line(f"{acc} = tl.zeros((BR, 1), tl.int64)")
    flags = em.fresh("oflags")   # bit 2: a NaN, bit 1: a +inf, bit 0: a -inf
    em.line(f"{flags} = tl.zeros((BR, 1), tl.int32)")
    if looped:
        loop_head()
    bits, e = terms()
    if not looped:
        em.line(f"{emax} = tl.max(tl.where({e} == 255, 1, tl.maximum({e}, 1)), axis=1, keep_dims=True)")
    mant = em.var(f"{bits} & 8388607", "om")
    # q = x / 2**L rounded half to even, as the integer steps of ordersum.quantize do it, but
    # with float steps that are exact: two power of two scales (no overflow, and an underflow
    # only where q is 0 anyway), truncation to int64, then the half even correction
    nL = em.var(f"{126 + _ordersum_bits()} - {emax}", "onl")
    sa = em.var(f"((({nL} >> 1) + 127) << 23).to(tl.float32, bitcast=True)", "osa")
    sb = em.var(f"((({nL} - ({nL} >> 1)) + 127) << 23).to(tl.float32, bitcast=True)", "osb")
    xf = em.var(f"tl.where({e} == 255, 0.0, {bits}.to(tl.float32, bitcast=True))", "oxf")
    y = em.var(f"({xf} * {sa}) * {sb}", "oy")
    t = em.var(f"{y}.to(tl.int64)", "ot")
    fr = em.var(f"{y} - {t}.to(tl.float32)", "ofr")
    odd = em.var(f"({t} & 1) == 1", "oodd")
    q = em.var(f"{t} + tl.where(({fr} > 0.5) | (({fr} == 0.5) & {odd}), 1, 0) - "
               f"tl.where(({fr} < -0.5) | (({fr} == -0.5) & {odd}), 1, 0)", "oq")
    em.line(f"{acc} = {acc} + tl.sum({q}, axis=1, keep_dims=True)")
    fl = em.var(f"tl.where({e} == 255, tl.where({mant} != 0, 4, tl.where({bits} < 0, 1, 2)), 0)", "ofl")
    em.line(f"{flags} = {flags} | tl.reduce({fl}, 1, _ts_or, keep_dims=True)")
    if looped:
        em.pop()
    # one rounding: int64 to float32 (round to nearest even), then an exact power of two scale
    L = em.var(f"{emax} - {126 + _ordersum_bits()}", "oL")
    low = em.var(f"{L} < -126", "olow")
    sc = em.var(f"((tl.where({low}, {L} + 64, {L}) + 127) << 23).to(tl.float32, bitcast=True)", "osc")
    f = em.var(f"{acc}.to(tl.float32) * {sc}", "of")
    f = em.var(f"tl.where({low}, {f} * {2.0 ** -64!r}, {f})", "of")
    f = em.var(f"tl.where({acc} == 0, 0.0, {f})", "of")
    f = em.var(f"tl.where(({flags} & 1) != 0, float('-inf'), {f})", "of")
    f = em.var(f"tl.where(({flags} & 2) != 0, float('inf'), {f})", "of")
    return em.var(f"tl.where((({flags} & 4) != 0) | (({flags} & 3) == 3), float('nan'), {f})", "osum")


def _ordersum_bits() -> int:
    from .ordersum import PRECISION_BITS
    return PRECISION_BITS


def _ordersum_shift() -> int:
    from .ordersum import PRECISION_BITS
    return PRECISION_BITS - 24


def gen_tile(k: Kernel, p: TilePlan, opts, looped: bool) -> str:
    em = Emitter(k, opts)
    sig = _signature(k, bool(em.checks)) + ["ROWS", "LANES", "BR: tl.constexpr", "BC: tl.constexpr"]
    name = "kern_loop" if looped else "kern"
    head = [_jit_line(sig), f"def {name}({', '.join(sig)}):"]
    em.line("pid_r = tl.program_id(0)")
    em.line("row = pid_r * BR + tl.arange(0, BR)[:, None]")
    em.line("rmask = row < ROWS")
    rows = _decompose(em, p.rows, "row")
    base = {a: (rows[a], "rmask") for a in p.rows}
    ctx = Ctx(base, {})
    shape_full = "(BR, BC)" if p.lane is not None else "(BR, 1)"
    if not p.reduces:
        em.line("lane = tl.program_id(1) * BC + tl.arange(0, BC)[None, :]")
        em.line("lmask = lane < LANES")
        c = ctx.extend({p.lane: ("lane", "lmask")}) if p.lane is not None else ctx
        for o in k.outputs:
            val = em.ex(o.expr, c)
            mask = "rmask & lmask" if p.lane is not None else "rmask"
            _store(em, o, c, val, shape_full, mask, full=_tile_full(p, o))
        return "\n".join(head + em.lines)
    R = em.size(p.rsize)
    if not looped:
        em.line("lane = tl.arange(0, BC)[None, :]")
        em.line(f"lmask = lane < {R}")
    for r in p.reduces:
        if r.op == "osum":
            em.values[r] = _emit_osum(em, r, ctx, looped, R)
            continue
        neutral = _NEUTRAL[r.op]
        fn = {"sum": "tl.sum", "max": "tl.max", "min": "tl.min"}[r.op]
        if looped:
            acc = em.fresh("acc")  # mutable: never shared through CSE
            em.line(f"{acc} = tl.full((BR, 1), {neutral}, tl.float32)")
            em.line(f"for off in range(0, {R}, BC):")
            em.push()
            em.line("lane = off + tl.arange(0, BC)[None, :]")
            em.line(f"lmask = lane < {R}")
            body = em.ex(r.body, ctx.extend({r.axis: ("lane", "lmask")}))
            part = em.var(f"{fn}(tl.where(lmask, {body} + tl.zeros((BR, BC), tl.float32), "
                          f"{neutral}), axis=1, keep_dims=True)")
            comb = {"sum": f"{acc} + {part}", "max": f"tl.maximum({acc}, {part})",
                    "min": f"tl.minimum({acc}, {part})"}[r.op]
            em.line(f"{acc} = {comb}")
            em.pop()
            em.values[r] = acc
        else:
            body = em.ex(r.body, ctx.extend({r.axis: ("lane", "lmask")}))
            v = em.var(f"{fn}(tl.where(lmask, {body} + tl.zeros((BR, BC), tl.float32), "
                       f"{neutral}), axis=1, keep_dims=True)", "red")
            em.values[r] = v
    if p.lane is None:
        for o in k.outputs:
            val = em.ex(o.expr, ctx)
            _store(em, o, ctx, val, "(BR, 1)", "rmask", full=_tile_full(p, o))
    elif looped:
        em.line(f"for off in range(0, {R}, BC):")
        em.push()
        em.line("lane = off + tl.arange(0, BC)[None, :]")
        em.line(f"lmask = lane < {R}")
        c = ctx.extend({p.lane: ("lane", "lmask")})
        for o in k.outputs:
            _store(em, o, c, em.ex(o.expr, c), "(BR, BC)", "rmask & lmask",
                   full=_tile_full(p, o))
        em.pop()
    else:
        c = ctx.extend({p.lane: ("lane", "lmask")})
        for o in k.outputs:
            _store(em, o, c, em.ex(o.expr, c), "(BR, BC)", "rmask & lmask",
                   full=_tile_full(p, o))
    return "\n".join(head + em.lines)


# ---------------------------------------------------------------- contraction


def _contraction_head(em: Emitter, p: ContractionPlan):
    """Program to tile mapping shared by the main and the split K epilogue kernel: grouped order
    so consecutive programs reuse the same B columns in L2. Returns the batch index context."""
    M, N = _rows_code(em, p), em.size(p.n.size)
    em.line("pid = tl.program_id(0)")
    em.line(f"num_m = tl.cdiv({M}, BM)")
    em.line(f"num_n = tl.cdiv({N}, BN)")
    em.line("group = GROUP * num_n")
    em.line("first_m = (pid // group) * GROUP")
    em.line("gsize = tl.minimum(num_m - first_m, GROUP)")
    em.line("pm = first_m + (pid % group) % gsize")
    em.line("pn = (pid % group) // gsize")
    em.line("pb = tl.program_id(1)")
    bidx = _decompose(em, p.batch, "pb")
    base = {a: (bidx[a], None) for a in p.batch}
    em.line("rm = pm * BM + tl.arange(0, BM)")
    em.line("rn = pn * BN + tl.arange(0, BN)")
    em.line(f"mm = rm[:, None] < {M}")
    em.line(f"mn = rn[None, :] < {N}")
    if p.fold:   # rows are (fold axes..., m) flattened row major
        Mm = em.size(p.m.size)
        em.line(f"rmm = rm % {Mm}")
        em.line(f"rmf = rm // {Mm}")
        fidx = _decompose(em, p.fold, "rmf")
        base.update({a: (f"{fidx[a]}[:, None]", "mm") for a in p.fold})
    return base


def _rows_code(em: Emitter, p) -> str:
    """Row count of a contraction launch: M, times the folded batch sizes."""
    return " * ".join(f"({em.size(a.size)})" for a in tuple(p.fold) + (p.m,))


def _mrow(p) -> str:
    return "rmm[:, None]" if p.fold else "rm[:, None]"


def _ws_offset(em: Emitter, p: ContractionPlan, split: str) -> str:
    """Offset into the split K workspace, laid out [SPLIT_K, batch, M, N] in float32."""
    M, N = _rows_code(em, p), em.size(p.n.size)
    nb = " * ".join(f"({em.size(a.size)})" for a in p.batch) or "1"
    return f"((({split}) * ({nb}) + pb) * {M} + rm[:, None]) * {N} + rn[None, :]"


def _epilogue(em: Emitter, k: Kernel, p: ContractionPlan, base):
    o = k.outputs[0]
    em.values[p.red] = "acc"
    ce = Ctx({**base, p.m: (_mrow(p), "mm"), p.n: ("rn[None, :]", "mn")}, {})
    val = em.ex(o.expr, ce)
    _store(em, o, ce, val, "(BM, BN)", "mm & mn", full={p.m, p.n} <= free_axes(o.expr))


def gen_contraction(k: Kernel, p: ContractionPlan, opts) -> str:
    """The contraction kernel `kern` and its split K epilogue `kern_splitk`.

    SPLIT_K (launch config) cuts the K range into SPLIT_K chunks of whole BK slabs, one program per
    chunk along grid axis 2. With SPLIT_K == 1 the program applies the epilogue and stores the
    output itself. Otherwise each program stores its float32 partial sum to the workspace `p_ws`
    and `kern_splitk` adds the partials in the fixed order 0, 1, ..., SPLIT_K - 1 and applies the
    epilogue. No atomics: the same inputs and config give the same bits on every run."""
    em = Emitter(k, opts)
    sig = _signature(k, bool(em.checks)) + ["p_ws", "BM: tl.constexpr", "BN: tl.constexpr", "BK: tl.constexpr",
                           "GROUP: tl.constexpr", "KTAIL: tl.constexpr", "SPLIT_K: tl.constexpr"]
    head = [_jit_line(sig), f"def kern({', '.join(sig)}):"]
    K = em.size(p.k.size)
    base = _contraction_head(em, p)
    dt = _dot_dtype(p.a, p.b)
    em.line("acc = tl.zeros((BM, BN), tl.float32)")

    def step(masked: bool):
        """One BK slab of the contraction. Without a K mask the loads stay vectorized along K:
        the mask's constancy along K is one element when K is a runtime size."""
        em.line("rk = k0 + tl.arange(0, BK)")
        ka = kb = None
        if masked:
            em.line("mka = rk[None, :] < k_hi")
            em.line("mkb = rk[:, None] < k_hi")
            ka, kb = "mka", "mkb"
        ca = Ctx({**base, p.m: (_mrow(p), "mm"), p.k: ("rk[None, :]", ka)}, {})
        cb = Ctx({**base, p.k: ("rk[:, None]", kb), p.n: ("rn[None, :]", "mn")}, {})
        a = em.operand(p.a, ca, dt, ka)
        a = em.full(a, p.a, (p.m, p.k), "(BM, BK)", _TL[dt])
        b = em.operand(p.b, cb, dt, kb)
        b = em.full(b, p.b, (p.k, p.n), "(BK, BN)", _TL[dt])
        if _exact_tc(dt, opts):   # fixed K order: block results added one by one
            em.line(f"acc = acc + {_exact_dot(a, b, 'BK', dt)}")
        else:
            em.line(f"acc = tl.dot({a}, {b}, acc{_precision_arg(dt, opts)})")

    def k_loop():
        # KTAIL (launch config): whole BK slabs without a K mask, then at most one masked tail
        # slab. Faster for most tiles; 128 x 128 half precision tiles measured faster with one
        # masked loop, so it is a switch the tuner tries.
        em.line("if KTAIL:")
        em.push()
        em.line("k_full = k_lo + ((k_hi - k_lo) // BK) * BK")
        em.line("for k0 in range(k_lo, k_full, BK):")
        em.push()
        step(False)
        em.pop()
        em.line("if k_full < k_hi:")
        em.push()
        em.line("k0 = k_full")
        step(True)
        em.pop()
        em.pop()
        em.line("else:")
        em.push()
        em.line("for k0 in range(k_lo, k_hi, BK):")
        em.push()
        step(True)
        em.pop()
        em.pop()

    if p.kchunk:
        # deterministic K split (det_k_chunk): chunk c sums [c * KCH, (c + 1) * KCH) from zero; a
        # program runs a contiguous range of chunks and either adds their sums in order itself
        # (SPLIT_K == 1) or stores each to the chunk's own workspace slot
        nch = -(-p.k.size // p.kchunk)
        em.line("tot = acc")
        em.line("if SPLIT_K == 1:")
        em.push()
        em.line("c_lo = 0")
        em.line(f"c_hi = {nch}")
        em.pop()
        em.line("else:")
        em.push()
        em.line(f"c_per = tl.cdiv({nch}, SPLIT_K)")
        em.line("c_lo = tl.program_id(2) * c_per")
        em.line(f"c_hi = tl.minimum({nch}, c_lo + c_per)")
        em.pop()
        em.line("for c in range(c_lo, c_hi):")
        em.push()
        em.line(f"k_lo = tl.multiple_of(c * {p.kchunk}, {p.kchunk})")
        em.line(f"k_hi = tl.minimum({K}, k_lo + {p.kchunk})")
        em.line("acc = tl.zeros((BM, BN), tl.float32)")
        k_loop()
        em.line("if SPLIT_K == 1:")
        em.push()
        em.line("tot = tot + acc")
        em.pop()
        em.line("else:")
        em.push()
        em.line(f"tl.store(p_ws + {_ws_offset(em, p, 'c')}, acc, mask=mm & mn)")
        em.pop()
        em.pop()
        em.line("acc = tot")
    else:
        # The K range of this program. SPLIT_K == 1 keeps every bound a plain size, so Triton
        # still sees the loop start as 0 and the slab offsets as multiples of BK.
        em.line("if SPLIT_K == 1:")
        em.push()
        em.line("k_lo = 0")
        em.line(f"k_hi = {K}")
        em.pop()
        em.line("else:")
        em.push()
        em.line(f"k_chunk = tl.cdiv(tl.cdiv({K}, SPLIT_K), BK) * BK")
        em.line("k_lo = tl.multiple_of(tl.program_id(2) * k_chunk, BK)")
        em.line(f"k_hi = tl.minimum({K}, k_lo + k_chunk)")
        em.pop()
        k_loop()
    em.line("if SPLIT_K == 1:")
    em.push()
    _epilogue(em, k, p, base)
    em.pop()
    if not p.kchunk:
        em.line("else:")
        em.push()
        em.line(f"tl.store(p_ws + {_ws_offset(em, p, 'tl.program_id(2)')}, acc, mask=mm & mn)")
        em.pop()
    main = "\n".join(head + em.lines)

    # second pass: fixed order sum of the partials, then the epilogue
    em = Emitter(k, opts)
    sig2 = _signature(k, bool(em.checks)) + ["p_ws", "BM: tl.constexpr", "BN: tl.constexpr", "GROUP: tl.constexpr",
                            "SPLIT_K: tl.constexpr"]
    head2 = [_jit_line(sig2), f"def kern_splitk({', '.join(sig2)}):"]
    base = _contraction_head(em, p)
    em.line("acc = tl.zeros((BM, BN), tl.float32)")
    slots = -(-p.k.size // p.kchunk) if p.kchunk else "SPLIT_K"
    em.line(f"for s in range(0, {slots}):")  # a loop, not unrolled: a short compile
    em.push()
    em.line(f"acc += tl.load(p_ws + {_ws_offset(em, p, 's')}, mask=mm & mn, other=0.0)")
    em.pop()
    _epilogue(em, k, p, base)
    return main + "\n\n\n" + "\n".join(head2 + em.lines)


# ---------------------------------------------------------------- attention


def _block_mask(em: Emitter, size, rng: str, name: str, col: bool) -> Optional[str]:
    """Mask for a padded block along a dim, or None when the size is a constant equal to the
    block width that prepare always picks for it (max(16, next power of two))."""
    if isinstance(size, int) and size == max(16, _pow2(size)) and _sched(em.opts, "lean"):
        return None
    sel = "[:, None]" if col else "[None, :]"
    em.line(f"{name} = {rng}{sel} < {em.size(size)}")
    return name


# Code generation choices that are not launch parameters. Each was measured on its own (see the
# 2026-11-24 kernel performance entry in docs/QUALITY_BAR.md); the hidden compile option `_sched`
# overrides them for such measurements and is part of the cache key.
SCHEDULE_DEFAULTS = {"exp2": True, "lean": True}


def _sched(opts, key):
    return opts.get("_sched", {}).get(key, SCHEDULE_DEFAULTS[key])


def _attention_head(em: Emitter, p: AttentionPlan):
    """Program ids (LPT order), batch indices, query rows and value columns of an attention
    program; shared by the main kernel and the split KV combine."""
    em.line("if LPT:")
    em.push()
    em.line("pb = tl.program_id(0)")
    em.line("pm = tl.num_programs(1) - 1 - tl.program_id(1)")
    em.pop()
    em.line("else:")
    em.push()
    em.line("pm = tl.program_id(0)")
    em.line("pb = tl.program_id(1)")
    em.pop()
    bidx = _decompose(em, p.batch, "pb")
    base = {a: (bidx[a], None) for a in p.batch}
    em.line("start_m = pm * BM")
    em.line("ri = start_m + tl.arange(0, BM)")
    em.line(f"mi = ri[:, None] < {em.size(p.i.size)}")
    em.line("rd = tl.arange(0, BD)")
    md = _block_mask(em, p.d.size, "rd", "md", col=False)
    return base, md


def _kv_offset(em: Emitter, p: AttentionPlan, chunk: str, t: int) -> str:
    """Offset of the split KV partial record of query rows `ri`, term t, chunk `chunk`. The
    workspace is float32 laid out [chunks, batch, NI, terms, BD + 2]: per row and term the running
    max, the sum of exponentials, then BD accumulator columns."""
    nb = " * ".join(f"({em.size(a.size)})" for a in p.batch) or "1"
    T = len(p.terms)
    return f"(((({chunk}) * ({nb}) + pb) * {em.size(p.i.size)} + ri) * {T} + {t}) * (BD + 2)"


def _attention_epilogue(em: Emitter, k: Kernel, p: AttentionPlan, base, md):
    o = k.outputs[0]
    ci = Ctx({**base, p.i: ("ri[:, None]", "mi"), p.d: ("rd[None, :]", md)}, {})
    parts = []
    for t, term in enumerate(p.terms):
        inv = em.var(f"1.0 / l{t}", "inv")
        if _is_one(term.coef):
            parts.append(f"acc{t} * {inv}[:, None]")
        else:
            parts.append(f"({em.ex(term.coef, ci)}) * (acc{t} * {inv}[:, None])")
    total = em.var(" + ".join(parts), "att")
    em.values[p.red] = total
    val = em.ex(o.expr, ci)
    _store(em, o, ci, val, "(BM, BD)", "mi & md" if md else "mi",
           full={p.i, p.d} <= free_axes(o.expr))


def gen_attention(k: Kernel, p: AttentionPlan, opts) -> str:
    """Online softmax attention, `kern`, and its split KV combine, `kern_splitkv`.

    Two constexpr switches are chosen at launch (by the autotuner or the defaults): LPT runs the
    query blocks with the most key blocks first (largest start_m first, batch fastest), SPLIT runs
    the key blocks that no mask can touch in their own loop with every index mask and bounds check
    removed, and only the edge blocks with masks.

    KV_CHUNK (launch config, a multiple of BN, or 0) splits the keys: with KV_CHUNK > 0 grid axis 2
    runs one program per chunk of KV_CHUNK keys, each storing its float32 running max, sum and
    accumulator to the workspace `p_ws`, and `kern_splitkv` merges the chunks in the fixed order
    0, 1, 2, ... before the epilogue. Decode (one query row) then uses many programs instead of
    one per batch and head. No atomics: the same inputs and config give the same bits every run."""
    em = Emitter(k, opts)
    T = len(p.terms)
    sig = _signature(k, bool(em.checks)) + ["p_ws", "BM: tl.constexpr", "BN: tl.constexpr",
                                            "BD: tl.constexpr"]
    sig += [f"BC{t}: tl.constexpr" for t in range(T)]
    sig += ["LPT: tl.constexpr", "SPLIT: tl.constexpr", "KV_CHUNK: tl.constexpr"]
    head = [_jit_line(sig), f"def kern({', '.join(sig)}):"]
    NI, NJ = em.size(p.i.size), em.size(p.j.size)
    use_exp2 = _sched(opts, "exp2")
    ex = "exp2" if use_exp2 else "exp"
    base, md = _attention_head(em, p)
    vdt = _dot_dtype(p.v)
    # scores as written (edge blocks) and with the analysed index masks removed (interior
    # blocks); with exp2 both are multiplied by log2(e), folded into the score's constants
    scores, inner = [], []
    for term in p.terms:
        sc = term.score
        st = _strip_bounds(sc, p.i, p.j)
        if use_exp2:
            sc, st = _scale_expr(sc, LOG2E), _scale_expr(st, LOG2E)
        scores.append(sc)
        inner.append(st)
    # queries: loaded once per program
    qs = []
    for t, term in enumerate(p.terms):
        em.line(f"rc{t} = tl.arange(0, BC{t})")
        mcq = _block_mask(em, term.c.size, f"rc{t}", f"mcq{t}", col=False)
        mck = _block_mask(em, term.c.size, f"rc{t}", f"mck{t}", col=True)
        cq = Ctx({**base, p.i: ("ri[:, None]", "mi"), term.c: (f"rc{t}[None, :]", mcq)}, {})
        qdt = _dot_dtype(term.q, term.kk)
        q = em.operand(term.q, cq, qdt, mcq)
        q = em.full(q, term.q, (p.i, term.c), f"(BM, BC{t})", _TL[qdt], "q")
        qs.append((q, qdt, mck))
        em.line(f"m{t} = tl.full((BM,), float('-inf'), tl.float32)")
        em.line(f"l{t} = tl.zeros((BM,), tl.float32)")
        em.line(f"acc{t} = tl.zeros((BM, BD), tl.float32)")

    def body(masked: bool):
        em.push()
        em.line("rj = n0 + tl.arange(0, BN)")
        if masked:
            em.line(f"mjr = rj[None, :] < {NJ}")
            em.line(f"mjc = rj[:, None] < {NJ}")
            jr, jc = "mjr", "mjc"
        else:
            jr = jc = None
        cv = Ctx({**base, p.j: ("rj[:, None]", jc), p.d: ("rd[None, :]", md)}, {})
        v = em.operand(p.v, cv, vdt, jc)
        v = em.full(v, p.v, (p.j, p.d), "(BN, BD)", _TL[vdt], "v")
        for t, term in enumerate(p.terms):
            q, qdt, mck = qs[t]
            ck = Ctx({**base, p.j: ("rj[None, :]", jr), term.c: (f"rc{t}[:, None]", mck)}, {})
            kt = em.operand(term.kk, ck, qdt, mck)
            kt = em.full(kt, term.kk, (p.j, term.c), f"(BC{t}, BN)", _TL[qdt], "kt")
            if _exact_tc(qdt, opts):
                dot = em.var(_exact_dot(q, kt, f"BC{t}", qdt), "qk")
            else:
                dot = em.var(f"tl.dot({q}, {kt}{_precision_arg(qdt, opts)})", "qk")
            cs = Ctx({**base, p.i: ("ri[:, None]", "mi"), p.j: ("rj[None, :]", jr)}, {})
            em.values[Placeholder("dot")] = dot
            s = em.ex(scores[t] if masked else inner[t], cs)
            del em.values[Placeholder("dot")]
            if masked:
                s = em.var(f"tl.where(mjr, {s}, float('-inf'))", "s")
            em.line(f"mnew{t} = tl.maximum(m{t}, tl.max({s}, axis=1))")
            em.line(f"msafe{t} = tl.where(mnew{t} == float('-inf'), 0.0, mnew{t})")
            em.line(f"alpha{t} = {em.fn(ex, f'm{t} - msafe{t}')}")
            em.line(f"pr{t} = {em.fn(ex, f'{s} - msafe{t}[:, None]')}")
            em.line(f"l{t} = l{t} * alpha{t} + tl.sum(pr{t}, axis=1)")
            if _exact_tc(vdt, opts):
                em.line(f"acc{t} = acc{t} * alpha{t}[:, None] + "
                        f"{_exact_dot(f'pr{t}.to({_TL[vdt]})', v, 'BN', vdt)}")
            else:
                em.line(f"acc{t} = acc{t} * alpha{t}[:, None] + "
                        f"tl.dot(pr{t}.to({_TL[vdt]}), {v}{_precision_arg(vdt, opts)})")
            em.line(f"m{t} = mnew{t}")
        em.pop()

    lo = "0"
    hi = NJ
    if p.lo_off is not None:
        em.line(f"lo = (tl.maximum(start_m + ({p.lo_off}), 0) // BN) * BN")
        lo = "lo"
    if p.hi_off is not None:
        em.line(f"hi = tl.minimum(start_m + BM + ({p.hi_off}), {NJ})")
        hi = "hi"
    il, ih = _interior_bounds(p)
    em.line("if KV_CHUNK > 0:")
    em.push()
    # one chunk of keys: [c0, c0 + KV_CHUNK) within [lo, hi); every block masked (chunks are
    # short, so the interior split would not pay); an empty chunk stores max -inf and sum 0
    em.line("c0 = tl.program_id(2) * KV_CHUNK")
    em.line(f"for n0 in range(tl.maximum({lo}, c0), tl.minimum({hi}, c0 + KV_CHUNK), BN):")
    body(True)
    em.line(f"mrow = ri < {NI}")
    for t in range(T):
        off = em.var(_kv_offset(em, p, "tl.program_id(2)", t), "wo")
        em.line(f"tl.store(p_ws + {off}, m{t}, mask=mrow)")
        em.line(f"tl.store(p_ws + {off} + 1, l{t}, mask=mrow)")
        em.line(f"tl.store(p_ws + {off}[:, None] + 2 + rd[None, :], acc{t}, mask=mrow[:, None])")
    em.pop()
    em.line("else:")     # no split: the whole key range, then the epilogue
    em.push()
    em.line("if SPLIT:")
    em.push()
    # key blocks [lo, a_in): masked (window start); [a_in, b_in): interior, no masks at all;
    # [b_in, hi): masked (causal diagonal and the ragged tail)
    if il is not None:
        em.line(f"a_in = {lo} + ((tl.maximum(start_m + BM - 1 + ({il}) - {lo}, 0) + BN - 1) "
                f"// BN) * BN")
        em.line(f"a_in = tl.minimum(a_in, {hi})")
        em.line(f"for n0 in range({lo}, a_in, BN):")
        body(True)
        a_in = "a_in"
    else:
        a_in = lo
    hin = f"tl.minimum(start_m + ({ih + 1}), {NJ})" if ih is not None else NJ
    em.line(f"b_in = {lo} + (tl.maximum({hin} - {lo}, 0) // BN) * BN")
    em.line(f"b_in = tl.maximum(b_in, {a_in})")
    em.line(f"for n0 in range({a_in}, b_in, BN):")
    body(False)
    em.line(f"for n0 in range(b_in, {hi}, BN):")
    body(True)
    em.pop()
    em.line("else:")
    em.push()
    em.line(f"for n0 in range({lo}, {hi}, BN):")
    body(True)
    em.pop()
    _attention_epilogue(em, k, p, base, md)
    em.pop()
    main = "\n".join(head + em.lines)

    # split KV combine: merge the chunk partials in the fixed order 0, 1, 2, ..., then the epilogue
    em = Emitter(k, opts)
    sig2 = _signature(k, bool(em.checks)) + ["p_ws", "BM: tl.constexpr", "BD: tl.constexpr",
                                             "LPT: tl.constexpr", "KV_CHUNK: tl.constexpr"]
    head2 = [_jit_line(sig2), f"def kern_splitkv({', '.join(sig2)}):"]
    base, md = _attention_head(em, p)
    em.line(f"mrow = ri < {NI}")
    em.line(f"nch = tl.cdiv({NJ}, KV_CHUNK)")
    for t in range(T):
        em.line(f"m{t} = tl.full((BM,), float('-inf'), tl.float32)")
        em.line(f"l{t} = tl.zeros((BM,), tl.float32)")
        em.line(f"acc{t} = tl.zeros((BM, BD), tl.float32)")
    em.line("for c in range(0, nch):")   # a loop, not unrolled: the chunk count is a runtime size
    em.push()
    for t in range(T):
        off = em.var(_kv_offset(em, p, "c", t), "wo")
        em.line(f"mc{t} = tl.load(p_ws + {off}, mask=mrow, other=float('-inf'))")
        em.line(f"lc{t} = tl.load(p_ws + {off} + 1, mask=mrow, other=0.0)")
        em.line(f"ac{t} = tl.load(p_ws + {off}[:, None] + 2 + rd[None, :], mask=mrow[:, None], "
                f"other=0.0)")
        em.line(f"mnew{t} = tl.maximum(m{t}, mc{t})")
        em.line(f"msafe{t} = tl.where(mnew{t} == float('-inf'), 0.0, mnew{t})")
        em.line(f"wa{t} = {em.fn(ex, f'm{t} - msafe{t}')}")
        em.line(f"wb{t} = {em.fn(ex, f'mc{t} - msafe{t}')}")
        em.line(f"l{t} = l{t} * wa{t} + lc{t} * wb{t}")
        em.line(f"acc{t} = acc{t} * wa{t}[:, None] + ac{t} * wb{t}[:, None]")
        em.line(f"m{t} = mnew{t}")
    em.pop()
    _attention_epilogue(em, k, p, base, md)
    return main + "\n\n\n" + "\n".join(head2 + em.lines)


# =========================================================================== module + launch


HEADER = ("import triton\nimport triton.language as tl\nfrom triton.language.extra import libdevice\n\n\n"
          "@triton.jit\ndef _ts_or(a, b):\n    return a | b\n\n\n")  # or reduction (order free sum flags)


_ROUND_ONCE = ("exp", "log", "tanh", "erf", "rsqrt", "sigmoid")


def generate(k: Kernel, opts: Dict) -> Tuple[str, object]:
    """Source of a Python module with the Triton kernel(s) for canonical kernel k, and its plan."""
    p = plan(k)
    head = HEADER
    if opts.get("math") == "round_once":
        from . import roundonce
        if p.kind == "attention":
            raise roundonce.RoundOnceError(
                "TS1093", f"kernel {k.name}: the attention loop shape computes its softmax exp "
                          f"itself and has no round once form; compile it without math='round_once'")
        # read by prepare(): these modules launch with enable_fp_fusion=False, so no a*b+c of the
        # float64 evaluation (or of the float32 kernel around it) is contracted into one rounding
        head += "TS_FP_FUSION = False\n\n\n" + roundonce.triton_source(roundonce.used(k)) + "\n\n"
    if opts.get("portable_tc") and p.kind in ("contraction", "conv", "attention"):
        from . import tcexact
        head += tcexact.triton_source() + "\n\n"
        p.portable = True
        p.deterministic = True
    if p.kind == "tile":
        src = gen_tile(k, p, opts, looped=False)
        if p.reduces:
            src += "\n\n\n" + gen_tile(k, p, opts, looped=True)
    elif p.kind == "contraction":
        p.tf32 = bool(opts.get("allow_tf32"))
        p.deterministic = bool(opts.get("deterministic"))
        p.kchunk = det_k_chunk(p)
        src = gen_contraction(k, p, opts)
    elif p.kind == "conv":
        p.tf32 = bool(opts.get("allow_tf32"))
        src = triton_conv.gen_conv(k, p, opts)
    elif p.kind == "window":
        src = triton_conv.gen_window(k, p, opts)
    else:
        p.deterministic = bool(opts.get("deterministic"))
        src = gen_attention(k, p, opts)
    return head + src + "\n", p


TILE_SINGLE_MAX = 4096

# A launch config is a flat dict of ints and bools. Its keys depend on the loop shape only:
#   tile pointwise   BR BC num_warps
#   tile reduction   BR num_warps            (single pass; BC is the padded row length)
#   tile looped      BC num_warps            (rows longer than TILE_SINGLE_MAX)
#   contraction      BM BN BK GROUP num_warps num_stages KTAIL SPLIT_K
#   attention        BM BN num_warps num_stages LPT SPLIT
# default_config gives the fixed heuristic used when tuning is off; candidates gives the small
# search space the autotuner times (default first); check_config validates a config read back
# from the disk cache.


def _sz(s, env):
    return s if isinstance(s, int) else env[s.name]


def _prod(axes, env) -> int:
    n = 1
    for a in axes:
        n *= _sz(a.size, env)
    return n


def _msize(p, env) -> int:
    """Rows of a contraction launch: M times the folded batch axes."""
    return _prod(tuple(getattr(p, "fold", ())) + (p.m,), env)


def _sym_rows(p) -> bool:
    """True when a tile kernel's row count is symbolic. BR is a compile time constant of the Triton
    kernel, so it must then not follow the row count, or every new batch size would compile a new
    Triton variant of the same Truestar kernel."""
    return any(not isinstance(a.size, int) for a in p.rows)


def batch_symbols(p) -> List[str]:
    """Names of the symbolic sizes that only repeat the kernel's work (tile rows, batch axes of a
    contraction or an attention). A launch config must not depend on them: the autotuner keys its
    result without them, so one tuned config, and one Triton variant, serves every batch size."""
    if p.kind in triton_conv.KEYS:
        return triton_conv.batch_symbols(p)
    axes = p.rows if p.kind == "tile" else p.batch
    return sorted({a.size.name for a in axes if not isinstance(a.size, int)})


def coarse_symbols(p) -> List[str]:
    """Symbolic sizes the tuning key buckets by powers of four instead of two: the row count of a
    contraction (batch times tokens in a linear layer), where tile choices change slowly."""
    if p.kind == "contraction":
        return sorted({a.size.name for a in tuple(p.fold) + (p.m,) if not isinstance(a.size, int)})
    return []


# A launch too small to tune may reuse a config its kernel already compiled when the model below
# puts it at most this much slower than the default: such a launch takes microseconds, while a new
# Triton variant takes 0.3 to 0.6 s to compile.
REUSE_SLACK = 1.5
# Only below this many multiply adds (tens of microseconds): a larger launch runs often enough to
# repay its own variant's compile (on the transformer layer at batch 64, reusing batch 1 and 7
# configs cost 8% of the step).
REUSE_MAX_WORK = 1 << 28


def estimate(p, env: Dict[str, int], cfg: Dict, sms: int) -> float:
    """Relative cost of a contraction launch config, for choosing among configs already compiled:
    waves of programs times the time of one program, which is its multiply adds times a load
    overhead that grows as the tile gets thinner (operand loads per output are 1/BM + 1/BN)."""
    M, N, K = _msize(p, env), _sz(p.n.size, env), _sz(p.k.size, env)
    split = cfg.get("SPLIT_K", 1)
    progs = _programs(M, N, cfg["BM"], cfg["BN"]) * _prod(p.batch, env) * split
    chunk = -(-(-(-K // split)) // cfg["BK"]) * cfg["BK"]
    per_program = cfg["BM"] * cfg["BN"] * chunk * (1 + 32 / cfg["BM"] + 32 / cfg["BN"])
    return -(-progs // max(1, sms)) * per_program


def config_kind(p, env: Dict[str, int]) -> str:
    if p.kind == "tile":
        if not p.reduces:
            return "tile_pointwise"
        return "tile_single" if _sz(p.rsize, env) <= TILE_SINGLE_MAX else "tile_looped"
    return p.kind


_KEYS = {
    "tile_pointwise": ("BR", "BC", "num_warps"),
    "tile_single": ("BR", "num_warps"),
    "tile_looped": ("BC", "num_warps"),
    "contraction": ("BM", "BN", "BK", "GROUP", "num_warps", "num_stages", "KTAIL", "SPLIT_K"),
    "attention": ("BM", "BN", "num_warps", "num_stages", "LPT", "SPLIT", "KV_CHUNK"),
}
_BOOL_KEYS = ("LPT", "SPLIT", "KTAIL")


def _att_dims(p, env):
    bd = max(16, _pow2(_sz(p.d.size, env)))
    bcs = [max(16, _pow2(_sz(t.c.size, env))) for t in p.terms]
    if bd > 256 or max(bcs) > 256:
        raise Unschedulable("attention head dims above 256 are not supported")
    return bd, bcs


def _cfg(bm, bn, bk, warps, stages, ktail=True, split=1):
    return dict(BM=bm, BN=bn, BK=bk, GROUP=8, num_warps=warps, num_stages=stages, KTAIL=ktail,
                SPLIT_K=split)


def _programs(M, N, bm, bn) -> int:
    return -(-M // bm) * -(-N // bn)


def _contraction_class(p) -> str:
    """'half' (float16 or bfloat16 tensor cores), 'tf32' (float32 on TF32 tensor cores, only when
    the kernel was compiled with allow_tf32) or 'ieee' (strict float32, FMA units)."""
    if _dot_dtype(p.a, p.b) != "float32":
        return "half"
    return "tf32" if getattr(p, "tf32", False) else "ieee"


_ESIZE = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}

# The launch defaults were measured on an RTX 5070 Ti: 70 SMs, 101376 bytes of shared memory per
# program (opt in), 16 GB. Thresholds that count programs are written for those 70 SMs and scaled
# by the SM count of the GPU the kernel runs on (see `_waves`), so a 70 SM GPU gets exactly the
# measured configs and a smaller or larger one gets the same fill of its SMs.
REF_SMS = 70
REF_SMEM = 101376
REF_MEMORY = 16 * 1024 ** 3
SMEM_RESERVE = 3 * 1024  # bytes kept free for the epilogue and Triton's own scratch
_LIMITS: Dict[int, Tuple[int, int, int]] = {}
# Limits of a GPU that is not attached (an AMD target compiled ahead of time, kernel/rocm.py):
# while set, device_limits returns them instead of asking torch. Per thread and per task.
_TARGET_LIMITS = contextvars.ContextVar("truestar_target_limits", default=None)


@contextlib.contextmanager
def target_limits(sms: int, smem: int, memory: int):
    """Plan launches (defaults, candidates, shared memory budget, split K workspace) for a GPU with
    ``sms`` SMs or CUs, ``smem`` bytes of shared memory (LDS) per program and ``memory`` bytes,
    instead of the attached one."""
    tok = _TARGET_LIMITS.set((int(sms), int(smem), int(memory)))
    try:
        yield
    finally:
        _TARGET_LIMITS.reset(tok)


def device_limits() -> Tuple[int, int, int]:
    """(SMs, shared memory per program in bytes, total memory in bytes) of the current CUDA device,
    read once per device from torch.cuda.get_device_properties. Without CUDA (the CPU side of the
    tests, plan inspection) the reference GPU's values, so defaults do not depend on the machine
    that only builds a plan. Inside ``target_limits`` the target's values. On a ROCm build of torch
    the device is an AMD GPU and the SM count is its CU count."""
    over = _TARGET_LIMITS.get()
    if over is not None:
        return over
    import torch
    try:     # CUDA_VISIBLE_DEVICES="" can leave is_available() True with no device to ask
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return REF_SMS, REF_SMEM, REF_MEMORY
        dev = torch.cuda.current_device()
    except (RuntimeError, AssertionError):
        return REF_SMS, REF_SMEM, REF_MEMORY
    lim = _LIMITS.get(dev)
    if lim is None:
        pr = torch.cuda.get_device_properties(dev)
        smem = getattr(pr, "shared_memory_per_block_optin", 0) or getattr(pr, "shared_memory_per_block", 0)
        lim = (int(pr.multi_processor_count), int(smem) or REF_SMEM, int(pr.total_memory))
        _LIMITS[dev] = lim
    return lim


def _waves(programs: int, sms: Optional[int] = None) -> int:
    """A program count measured on the 70 SM reference GPU, scaled to this GPU's SMs."""
    if sms is None:
        sms = device_limits()[0]
    if sms == REF_SMS:
        return programs
    return max(1, round(programs * sms / REF_SMS))


def smem_budget() -> int:
    """Shared memory a contraction's pipelined operand loads may use: what the device allows per
    program minus SMEM_RESERVE (96 KiB on sm_120, as measured; less on GPUs with less)."""
    return device_limits()[1] - SMEM_RESERVE


SMEM_BUDGET = REF_SMEM - SMEM_RESERVE  # the reference GPU's budget (smem_budget() is what is used)


def _fit_shared(p, cfg: Dict) -> Dict:
    """cfg with fewer stages (then a shorter BK) when the pipelined operand loads would not fit
    in shared memory. Each distinct load in a prologue is buffered per stage, so an operand built
    from several buffers (a softmax backward, for example) needs several times the plain size."""
    def loads(e):
        return {n for n in ir.walk(e) if isinstance(n, Load)}

    def need(c):
        a = sum(_ESIZE.get(n.buffer.dtype, 4) for n in loads(p.a)) * c["BM"] * c["BK"]
        b = sum(_ESIZE.get(n.buffer.dtype, 4) for n in loads(p.b)) * c["BK"] * c["BN"]
        return (a + b) * c["num_stages"]

    c = dict(cfg)
    budget = smem_budget()
    while need(c) > budget and c["num_stages"] > 2:
        c["num_stages"] -= 1
    while need(c) > budget and c["BK"] > 16:
        c["BK"] //= 2
    return c


def _contraction_default(p, env: Dict[str, int]) -> Dict:
    return _fit_shared(p, _contraction_heuristic(p, env))


def _contraction_heuristic(p, env: Dict[str, int]) -> Dict:
    """Heuristic contraction config by operand class and by how many programs a tile size gives.

    Measured on an RTX 5070 Ti (70 SMs) for linear layers with a bias and relu epilogue, M from 1
    to 4096 and N, K from 256 to 4096 (2026-11-24 matmul entry in docs/QUALITY_BAR.md). Small M
    wants 16 or 32 row tiles so that enough programs run, with a long BK so each program streams a
    wide slab of the weight per step; large problems want 128 x 128 tiles."""
    M, N = _msize(p, env), _sz(p.n.size, env)
    nb = _prod(p.batch, env)
    cls = _contraction_class(p)

    def progs(bm, bn):
        """Programs of a launch with bm x bn tiles, batch included: a linear layer imported
        from [B, T, C] activations is a batched contraction with M = T, so B x T rows count."""
        return _programs(M, N, bm, bn) * nb
    sms = device_limits()[0]
    if sms < FEW_SMS and cls in ("half", "tf32"):
        c = (_few_sm_half if cls == "half" else _few_sm_tf32)(p, env, M, progs, sms)
        if c is not None:
            return c
    if cls == "half":
        # 64 x 64 x 64 tiles matched or beat 128 x 128 ones from 512 x 1024 up to 4096 cubed
        if M > 32 and progs(64, 64) >= 128:
            return _cfg(64, 64, 64, 4, 3)
        if M > 16 and progs(32, 64) >= 64:
            return _cfg(32, 64, 128, 4, 3)
        return _cfg(16, 64 if progs(16, 64) >= 64 else 32, 128, 4, 3)
    if cls == "tf32":
        if M > 32 and progs(64, 64) >= 128:
            return _cfg(64, 64, 32, 4, 3)
        if M > 64 and progs(128, 64) >= 32:
            return _cfg(128, 64, 32, 4, 3)
        if M > 16 and progs(32, 64) >= 64:
            return _cfg(32, 64, 64, 4, 3)
        return _cfg(16, 32, 64, 4, 3)
    if sms < FEW_SMS:
        return _few_sm_ieee(p, env, M, progs, sms)
    c = _ref_sm_ieee(p, env, M, progs, sms)
    # Not for folded batch axes: on an L4 the serving step (folded decode and prefill linears) gave
    # other bits with these tiles than with the 70 SM ones, so a request's tokens changed with its
    # batch (TS1401, tests/test_serve.py); the tiles were measured bitwise equal on plain [M, K]
    # contractions only.
    if sms < REF_SMS and c["SPLIT_K"] == 1 and not p.fold:
        return _mid_sm_ieee(p, env, M, progs, sms, c)
    return c


def _ref_sm_ieee(p, env: Dict[str, int], M: int, progs, sms: int) -> Dict:
    """Strict float32 defaults from FEW_SMS SMs up, measured on the 70 SM reference GPU."""
    # strict float32 (FMA units): the widest tiles that still give a full wave, since each thread
    # then reuses every loaded operand for more output elements
    # 128 row tiles only when M fills them: with batch axes (attention projections) M can be 1
    if M > 64 and getattr(p, "kchunk", 0) and progs(128, 64) >= 2 * sms:
        # deterministic K chunks keep a second accumulator (the sum of the finished chunks):
        # 128 x 64 with 4 warps, measured 2026-11-25 on 70 SMs, 4096 cubed 5085 us against 5430
        # for 128 x 256 x 16 with 8 warps, 1024 x 1024 x 2048 157 us against 197 for 128 x 128
        return _cfg(128, 64, 16, 4, 3)
    if M > 64 and progs(128, 256) >= 64:
        return _cfg(128, 256, 16, 8, 3)
    if M > 64 and progs(128, 128) >= 64:
        # under one wave of the SMs: two K chunks, 1024 cubed from 131 to 126 us on 70 SMs
        if progs(128, 128) < sms:
            return _cfg(128, 128, 16, 4, 3, split=min(2, _split_k(p, env, 128, 128, 128)))
        return _cfg(128, 128, 16, 4, 3)
    if M > 64 and progs(128, 64) >= 32:
        return _cfg(128, 64, 16, 4, 3)
    # Too few output tiles for 70 SMs: split K (when allowed and K is long enough) runs wide tiles
    # on a chunk of K each, then a second pass adds the partials in a fixed order. Measured
    # 2026-11-24: 64 x 2048 x 1024 from 38 to 28 us, 64 x 1000 x 2048 from 56 to 30 us.
    if M > 16:
        split = _split_k(p, env, 64, 128, 128)
        if split > 1:
            return _cfg(64, 128, 16, 4, 3, split=split)
    else:
        # rows up to 16: the 2 warp 16 x 64 x 64 tile, split only until 64 programs run
        # (1 x 4096 x 4096 split 4 ways was 9% slower; 16 x 1024 x 16384 from 436 to 111 us)
        split = _split_k(p, env, 16, 64, 64)
        if split > 1:
            return _cfg(16, 64, 64, 2, 3, split=split)
    if M > 32 and progs(64, 128) >= 128:   # many batches of up to 64 rows
        return _cfg(64, 128, 16, 4, 3)
    # small tiles: 2 warps, so each thread computes more of the tile per loaded operand
    if M > 16 and progs(32, 64) >= 64:
        return _cfg(32, 64, 32, 2, 4)
    return _cfg(16, 64, 64, 2, 3)


FEW_SMS = 48  # below this SM count the strict float32 defaults follow _few_sm_ieee


def _mid_sm_ieee(p, env: Dict[str, int], M: int, progs, sms: int, ref: Dict) -> Dict:
    """Strict float32 tiles from FEW_SMS to REF_SMS SMs where the reference rules do not split K
    (deterministic mode, or K too short): only the tile changes, and without split K the tile does
    not change a single bit (each output is one fused multiply add chain over K in order).

    Measured on an NVIDIA L4 (58 SMs, sm_89, 2026-11-24, scripts/probe_ieee_mm.py sweep, 65 to 78
    configs per shape, 0 bit mismatches): with few output tiles a 64 x 16 tile with one warp (two
    from K 8192) keeps more programs in flight (64 x 2048 x 2048 from 183 to 129 us, 512 x 256 x
    1024 from 89 to 64 us, 128 x 256 x 8192 from 318 to 262 us); up to 16 rows a 16 x 32 x 64
    tile (16 x 4096 x 1024 from 80 to 63 us, 16 x 1024 x 16384 from 620 to 401 us; 16 x 128 x 64
    was 10% faster on N 4096 at 3 stages, but its operands need 2 at this shared memory); large
    problems under two waves of 128 x 256 tiles a 256 x 128 tile with 8 warps (1024 cubed from 383
    to 302 us; at 4096 cubed it was 2% slower than 128 x 256), or 64 x 128 for N up to 256 (8192 x
    256 x 1024 from 638 to 588 us). Only the L4 was measured in this band."""
    K = _sz(p.k.size, env)
    N = _sz(p.n.size, env)
    if M <= 16:
        return _cfg(16, 32, 64, 2, 3)
    if progs(128, 64) < sms:
        return _cfg(64, 16, 32, 1 if K <= 4096 else 2, 3)
    if M > 64 and progs(128, 128) >= sms:
        if N <= 256:
            return _cfg(64, 128, 16, 4, 3)
        if progs(128, 256) < 2 * sms:
            return _cfg(256, 128, 16, 8, 3)
        if getattr(p, "kchunk", 0):
            # deterministic K chunks on the L4, weight [N, K]: 4096 cubed 18977 us with 128 x 256
            # x 16 and 8 warps, 20628 with the 70 SM chunk tile 128 x 64 (2026-11-25)
            return _cfg(128, 256, 16, 8, 3)
    return ref



def _few_sm_ieee(p, env: Dict[str, int], M: int, progs, sms: int) -> Dict:
    """Strict float32 defaults for GPUs with fewer than FEW_SMS SMs.

    Measured on an RTX 5060 Laptop GPU (26 SMs, 2026-11-24, scripts/sweep_contraction.py over 18
    shapes; results/laptop-sm-aware/): with few SMs a contraction wants several waves of programs
    (about 4, against about 1 on 70 SMs), so small row counts take a wider 16 x 128 tile and more
    K chunks, and wide 128 row tiles need more programs before they pay. Rules are in waves of this
    GPU's SMs. Only 26 and 70 SMs were measured; FEW_SMS is between them."""
    wave = lambda n: n * sms  # noqa: E731
    if M > 64 and getattr(p, "kchunk", 0) and progs(128, 64) >= wave(2):
        # deterministic K chunks: the 70 SM tile for their second accumulator (not measured on
        # 26 SMs; 128 x 256 x 16 with 8 warps was 7% slower than 128 x 64 on 70 SMs)
        return _cfg(128, 64, 16, 4, 3)
    if M > 64 and progs(128, 256) >= wave(5):
        return _cfg(128, 256, 16, 8, 3)
    if M > 64 and progs(128, 128) >= wave(2):
        return _cfg(128, 128, 16, 4, 3)
    if M > 64 and progs(64, 128) >= wave(2):
        return _cfg(64, 128, 16, 4, 3)
    target = wave(4)
    if M > 32:   # 64 row tiles, split K to about 4 waves (64 x 2048 x 1024: 64 to 58 us)
        split = _split_k(p, env, 64, 128, target, FEW_SMS_MIN_CHUNK)
        return _cfg(64, 128, 16, 4, 3, split=split)
    if M > 16:   # 32 x 4096 x 4096: 537 to 442 us
        split = _split_k(p, env, 32, 64, target, FEW_SMS_MIN_CHUNK)
        return _cfg(32, 64, 32, 2, 4, split=split)
    # up to 16 rows: 1 x 4096 x 4096 from 437 to 282 us, 16 x 4096 x 1024 from 110 to 65 us
    split = _split_k(p, env, 16, 128, target, FEW_SMS_MIN_CHUNK)
    if split == 1 and progs(16, 128) < wave(0.5):
        return _cfg(16, 64, 64, 2, 3)    # without split K too few programs: 16 x 1024 x 16384
    return _cfg(16, 128, 32, 2, 4, split=split)


def _few_sm_half(p, env: Dict[str, int], M: int, progs, sms: int) -> Optional[Dict]:
    """float16 and bfloat16 defaults below FEW_SMS SMs, or None for the 70 SM rules.

    Measured on 26 SMs (2026-11-24, scripts/sweep_contraction.py, results/laptop-sm-half/): the
    70 SM rules were already the best tile on 12 of 18 shapes. Two changes: 64 x 128 x 64 tiles
    from 1024 rows, a K of 1024 and 5 waves of programs (2048 x 4096 x 1024 from 697 to 655 us in
    scripts/bench_matmul.py; at 1024 cubed and at K 256 the sweep and the benchmark disagreed, so
    those keep 64 x 64), and split K to
    4 waves with 32 x 64 x 128 tiles when fewer than 2 waves of 32 x 64 tiles run over a K of at
    least 2048 (64 x 512 x 8192 from 33.4 to 28.2 us). Without split K (deterministic mode, or K
    too short) the tall K rule does not apply and the 70 SM rules decide."""
    wave = lambda n: n * sms  # noqa: E731
    K = _sz(p.k.size, env)
    if M >= 1024 and K >= 1024 and progs(64, 128) >= wave(5):
        return _cfg(64, 128, 64, 4, 3)
    if M > 16 and progs(32, 64) < wave(2) and K >= 2048:
        split = _split_k(p, env, 32, 64, wave(4), FEW_SMS_MIN_CHUNK)
        if split > 1:
            return _cfg(32, 64, 128, 4, 3, split=split)
    return None


def _few_sm_tf32(p, env: Dict[str, int], M: int, progs, sms: int) -> Optional[Dict]:
    """TF32 defaults below FEW_SMS SMs, or None for the 70 SM rules.

    Measured on 26 SMs (as _few_sm_half): 64 x 128 x 32 tiles for large products (1024 cubed from
    158 to 147 us; not when N or K is below 1024, where 64 x 64 stays ahead), and split K wherever
    few programs run: 64 x 64 x 32 tiles to 4 waves above 32 rows (64 x 2048 x 2048 from 55.4 to
    44.3 us, 128 x 256 x 8192 from 70.2 to 49.3 us), and 32 x 64 or 16 x 32 row tiles to 8 waves for
    fewer rows (32 x 4096 x 4096 from 225 to 202 us). Without split K the 70 SM rules decide."""
    wave = lambda n: n * sms  # noqa: E731
    N, K = _sz(p.n.size, env), _sz(p.k.size, env)
    if M >= 256 and N >= 1024 and K >= 1024 and progs(64, 128) >= wave(4):
        return _cfg(64, 128, 32, 4, 3)
    if M > 32:
        if progs(64, 64) < wave(2):
            split = _split_k(p, env, 64, 64, wave(4), FEW_SMS_MIN_CHUNK)
            if split > 1:
                return _cfg(64, 64, 32, 4, 3, split=split)
        return None
    bm, bn, bk = (32, 64, 64) if M > 16 else (16, 32, 64)
    split = _split_k(p, env, bm, bn, wave(8), FEW_SMS_MIN_CHUNK)
    return _cfg(bm, bn, bk, 4, 3, split=split) if split > 1 else None


DET_K_CHUNK = 512   # deterministic strict float32: K is summed in chunks of this many elements
DET_K_MIN = 1024    # ... when K is above this (K up to 1024 keeps one fused multiply add chain)


def det_k_chunk(p) -> int:
    """Chunk length of a deterministic K split, or 0 (one chain over K, as before 2026-11-25).

    In deterministic mode a strict float32 contraction with a constant K above DET_K_MIN sums K
    in chunks of DET_K_CHUNK elements: chunk c covers [c * DET_K_CHUNK, min(K, (c + 1) *
    DET_K_CHUNK)), its sum is one fused multiply add chain in K order starting from zero, and the
    chunk sums are added left to right, ((0 + s0) + s1) + s2 ..., before the epilogue. The chunks
    and the order follow K alone: not M (the batch), not the tile, not the SM count, not the GPU.
    Whether the chunks run in one program (the sum kept in registers) or in SPLIT_K programs (each
    chunk sum stored to its own workspace slot, then added by kern_splitk in the same order) is a
    launch choice and gives the same bits. A symbolic K, TF32 and half precision keep one chain."""
    if not getattr(p, "deterministic", False) or _contraction_class(p) != "ieee":
        return 0
    K = p.k.size
    if not isinstance(K, int) or K <= DET_K_MIN:
        return 0
    return DET_K_CHUNK


def k_slots(p, env: Dict[str, int], cfg: Dict) -> int:
    """Partial sums a split K launch stores: one per chunk of a deterministic K split, else one
    per program along K (SPLIT_K); 0 without a second pass."""
    if cfg.get("SPLIT_K", 1) == 1:
        return 0
    if getattr(p, "kchunk", 0):
        return -(-_sz(p.k.size, env) // p.kchunk)
    return cfg["SPLIT_K"]


SPLIT_K_MIN_CHUNK = 256  # elements of K each split program reduces, at least
FEW_SMS_MIN_CHUNK = 128  # the same below FEW_SMS SMs: 64 x 1000 x 2048 from 69 to 64 us


def _split_k(p, env: Dict[str, int], bm: int, bn: int, target: int,
             min_chunk: int = SPLIT_K_MIN_CHUNK) -> int:
    """Number of K chunks so that bm x bn tiles give about `target` programs, or 1.

    Always 1 for a plan compiled deterministic: split K adds the chunks' partial sums in a fixed
    order, so every run gives the same bits, but the chunk count follows M (the batch), so the
    rounding of a row would then depend on the batch size it was computed in.

    The exception is a deterministic K split (det_k_chunk): its chunks are fixed by K, so the
    programs along K (at most one per chunk) are a launch choice that changes no bit."""
    K = _sz(p.k.size, env)
    nch = 0
    if getattr(p, "deterministic", False):
        if not getattr(p, "kchunk", 0):
            return 1
        nch = -(-K // p.kchunk)
    tiles = _programs(_msize(p, env), _sz(p.n.size, env), bm, bn) * _prod(p.batch, env)
    split = 1
    while (tiles * split < target and split < MAX_SPLIT_K
           and (split * 2 <= nch if nch else K // (2 * split) >= min_chunk)):
        split *= 2
    slots = lambda s: (nch if nch else s) * workspace_elems(p, env)  # noqa: E731
    if split > 1 and slots(split) >= 1 << 31:
        return 1
    # the float32 workspace is allocated on every call: keep it to a small share of the GPU
    while split > 1 and slots(split) * 4 > device_limits()[2] // WORKSPACE_SHARE:
        split = 1 if nch else split // 2
    return split


WORKSPACE_SHARE = 16  # a split K workspace may take at most 1/16 of the GPU's memory


def portable_pin(p, cfg: Dict) -> Dict:
    """cfg with the K block of a portable tensor core contraction or convolution fixed at
    tcexact.PORTABLE_BK (and no split K): the exact blocks' rounding follows the K block, so it is
    a constant of the mode, never a tuned value. Other configs come back unchanged."""
    if not getattr(p, "portable", False) or p.kind not in ("contraction", "conv"):
        return cfg
    from .tcexact import PORTABLE_BK
    if cfg.get("BK") == PORTABLE_BK and cfg.get("SPLIT_K", 1) == 1:
        return cfg
    c = dict(cfg, BK=PORTABLE_BK)
    if "SPLIT_K" in c:
        c["SPLIT_K"] = 1
    # the longer K block must still fit shared memory: fewer stages first, then smaller tiles
    budget = smem_budget()

    def need(c):
        return (c["BM"] + c["BN"]) * c["BK"] * 4 * c["num_stages"]

    while need(c) > budget and c["num_stages"] > 2:
        c["num_stages"] -= 1
    while need(c) > budget and max(c["BM"], c["BN"]) > 16:
        key = "BN" if c["BN"] >= c["BM"] else "BM"
        c[key] //= 2
    return c


def default_config(p, env: Dict[str, int]) -> Dict:
    """The fixed heuristic config (used when tuning is off or not worth it)."""
    return portable_pin(p, _default_config(p, env))


def _default_config(p, env: Dict[str, int]) -> Dict:
    if p.kind in triton_conv.KEYS:
        return triton_conv.default_config(p, env)
    kind = config_kind(p, env)
    if kind == "tile_pointwise":
        rows = _prod(p.rows, env)
        lanes = _sz(p.lane.size, env) if p.lane is not None else 1
        # 4096 elements per program with 8 warps reaches DRAM bandwidth on an RTX 5070 Ti
        bc = min(_pow2(lanes), 4096)
        br = max(1, 4096 // bc if _sym_rows(p) else min(_pow2(rows), 4096 // bc))
        return dict(BR=br, BC=bc, num_warps=8 if br * bc >= 4096 else 4)
    if kind == "tile_single":
        # BR does not follow the row count, static or symbolic: BR and the warps fix the layout
        # tl.sum reduces a row in, so a row count of 3 with BR 4 summed rows in another order
        # than BR 16 (measured on an L4: a static compile at batch 3 differed from the symbolic
        # one in 64 of 192 softmax outputs). A row's bits now depend on its row length only, not
        # on the batch, and a static compile equals the symbolic one (compiler/dynamic.py).
        bc = max(16, _pow2(_sz(p.rsize, env)))
        br = max(1, min(4096 // bc, 16))
        return dict(BR=br, num_warps=8 if bc >= 2048 else 4)
    if kind == "tile_looped":
        return dict(BC=1024, num_warps=4)
    if kind == "contraction":
        return _contraction_default(p, env)
    return _attention_default(p, env)


DECODE_ROWS = 16   # query rows up to which an attention launch is a decode (one 16 row tile)
DET_KV_CHUNK = 256  # keys per split KV chunk of a deterministic decode, on every GPU
KV_TARGET_WAVES = 4  # a split KV decode aims at about this many waves of programs
# A split adds a second launch (about 20 us of host time on the 26 SM laptop GPU); it pays only
# when one program would walk this many keys in a row (waves of programs times the context).
# Measured 2026-11-24: batch 1 at 1024 keys and batch 8 at 512 keys were faster unsplit.
KV_MIN_SERIAL_KEYS = 2048


def _sm_count() -> int:
    """SMs of the current CUDA device (70, the reference GPU, without one)."""
    import torch
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            dev = torch.cuda.current_device()
            return int(torch.cuda.get_device_properties(dev).multi_processor_count)
    except (RuntimeError, AssertionError):
        pass
    return 70


def _attention_blocks(p, env: Dict[str, int], split: bool = False) -> Dict:
    NI = _sz(p.i.size, env)
    bd, bcs = _att_dims(p, env)
    half = _dot_dtype(p.v) != "float32"
    T = len(p.terms)
    wide = max([bd] + bcs)
    if NI <= DECODE_ROWS and split:
        # split KV decode: one 16 row tile (BM 64 left 63 of 64 rows idle), keys in blocks of 64
        return dict(BM=16, BN=64 if wide <= 128 else 32, num_warps=4, num_stages=3 if half else 2,
                    LPT=True, SPLIT=True)
    # measured on an RTX 5070 Ti (fp16, causal, S 2048 to 8192): head dim 64 is best at 64 x 64
    # blocks with 4 warps and 3 stages, head dim 128 at 128 x 64 with 8 warps and 2 stages
    if not half:
        return dict(BM=64, BN=32, num_warps=4, num_stages=2, LPT=True, SPLIT=True)
    if wide <= 64:
        return dict(BM=64, BN=64, num_warps=4, num_stages=3, LPT=True, SPLIT=True)
    if wide <= 128 and T == 1 and NI >= 128:
        return dict(BM=128, BN=64, num_warps=8, num_stages=2, LPT=True, SPLIT=True)
    return dict(BM=64, BN=32, num_warps=4, num_stages=2, LPT=True, SPLIT=True)


def kv_chunk(p, env: Dict[str, int], cfg: Dict, sms: Optional[int] = None) -> int:
    """Keys per split KV chunk for a launch with blocks `cfg`, or 0 for no split.

    Only decodes (up to DECODE_ROWS query rows) split: a prefill has a program per query block
    already. Deterministic: DET_KV_CHUNK whenever the keys exceed it, so the chunking, and so the
    bits, depend on the context length only, never on the SM count or the batch. Otherwise the
    chunk is the smallest power of two (at least BN) that keeps the programs near KV_TARGET_WAVES
    waves of this GPU's SMs."""
    NI, NJ = _sz(p.i.size, env), _sz(p.j.size, env)
    if NI > DECODE_ROWS:
        return 0
    bn = cfg["BN"]
    if getattr(p, "deterministic", False):     # the context length decides, nothing else
        return DET_KV_CHUNK if NJ >= KV_MIN_SERIAL_KEYS and DET_KV_CHUNK % bn == 0 else 0
    sms = _sm_count() if sms is None else sms
    progs = _prod(p.batch, env) * -(-NI // cfg["BM"])
    want = KV_TARGET_WAVES * sms
    if progs >= want or NJ <= 2 * bn or -(-progs // sms) * NJ < KV_MIN_SERIAL_KEYS:
        return 0
    chunks = -(-want // progs)
    chunk = max(bn, _pow2(-(-NJ // chunks)))
    return 0 if chunk >= NJ else chunk


def _attention_default(p, env: Dict[str, int]) -> Dict:
    """Prefill, and a decode that does not split: the blocks measured before split KV, unchanged.
    A decode that splits: one 16 row tile and kv_chunk's chunk."""
    if _sz(p.i.size, env) <= DECODE_ROWS:
        cfg = _attention_blocks(p, env, split=True)
        ch = kv_chunk(p, env, cfg)
        if ch:
            return dict(cfg, KV_CHUNK=ch)
        cfg = _attention_blocks(p, env)
        if cfg == dict(BM=64, BN=32, num_warps=4, num_stages=2, LPT=True, SPLIT=True) and \
                _dot_dtype(p.v) != "float32":
            # An unsplit half precision decode: one 16 row tile instead of 64 rows of which at most
            # 16 are live, 3 stages; same BN and warps, so the same bits (L4, head dim 128, batch 1
            # to 8, 1 to 16 rows, 512 to 2000 keys, 14 cases bitwise equal: 1024 keys from 48 to
            # 25 us, scripts/probe_decode_tiles.py).
            cfg = dict(cfg, BM=16, num_stages=3)
        return dict(cfg, KV_CHUNK=0)
    return dict(_attention_blocks(p, env), KV_CHUNK=0)


def candidates(p, env: Dict[str, int]) -> List[Dict]:
    """The small search space the autotuner times, default first, without duplicates."""
    out, seen = [], set()
    for c in _candidates(p, env):
        c = portable_pin(p, c)
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _candidates(p, env: Dict[str, int]) -> List[Dict]:
    if p.kind in triton_conv.KEYS:
        return triton_conv.candidates(p, env)
    kind = config_kind(p, env)
    d = default_config(p, env)
    out = [d]
    if kind == "tile_pointwise":
        lanes = _sz(p.lane.size, env) if p.lane is not None else 1
        rows = _prod(p.rows, env)
        for elems, warps in ((1024, 4), (2048, 4), (4096, 4), (8192, 8), (16384, 16)):
            bc = min(_pow2(lanes), 4096, elems)
            br = max(1, min(elems // bc, 8192) if _sym_rows(p) else min(_pow2(rows), elems // bc))
            out.append(dict(BR=br, BC=bc, num_warps=warps))
    elif kind == "tile_single":
        rows = _prod(p.rows, env)
        bc = max(16, _pow2(_sz(p.rsize, env)))
        for elems in (2048, 4096, 8192):
            for warps in (4, 8):
                br = max(1, min(elems // bc, 32) if _sym_rows(p) else min(_pow2(rows), elems // bc, 32))
                out.append(dict(BR=br, num_warps=warps))
    elif kind == "tile_looped":
        for bc, warps in ((1024, 8), (2048, 8), (4096, 8), (2048, 4)):
            out.append(dict(BC=bc, num_warps=warps))
    elif kind == "contraction":
        cls = _contraction_class(p)
        out += _prune_contraction(p, env, [_cfg(*g) for g in CONTRACTION_GRID[cls]])
        if out[0]["SPLIT_K"] > 1:  # neighbours of the default's chunk count
            top = (-(-_sz(p.k.size, env) // p.kchunk) if getattr(p, "kchunk", 0)
                   and getattr(p, "deterministic", False) else MAX_SPLIT_K)
            for sk in (out[0]["SPLIT_K"] // 2, out[0]["SPLIT_K"] * 2):
                if 1 <= sk <= min(MAX_SPLIT_K, top):
                    out.append(dict(out[0], SPLIT_K=sk))
        elif not getattr(p, "deterministic", False) or getattr(p, "kchunk", 0):
            for bm, bn, bk, w, target in ((64, 128, 16, 4, 128), (16, 64, 64, 2, 64)):
                sk = _split_k(p, env, bm, bn, target)
                if sk > 1:
                    out.append(_cfg(bm, bn, bk, w, 3, split=sk))
        if cls == "half":
            out.append(_cfg(128, 128, 64, 4, 3, ktail=False))
        out = [out[0]] + [_fit_shared(p, c) for c in out[1:]]
    else:
        bd, bcs = _att_dims(p, env)
        half = _dot_dtype(p.v) != "float32"
        big = bd > 128 or max(bcs) > 128
        if not half:
            grid = ((64, 32, 4, 2), (64, 64, 4, 2), (32, 32, 4, 2), (64, 32, 4, 3))
        elif big:
            grid = ((64, 32, 4, 2), (64, 32, 8, 2), (128, 32, 8, 2), (64, 64, 4, 2))
        elif bd > 64 or max(bcs) > 64:
            grid = ((128, 64, 8, 2), (128, 32, 4, 3), (64, 64, 4, 3), (128, 64, 8, 3),
                    (64, 32, 4, 2))
        else:
            grid = ((64, 64, 4, 2), (64, 32, 4, 3), (128, 64, 4, 2), (128, 32, 4, 3),
                    (128, 64, 8, 3), (128, 128, 8, 2))
        if _sz(p.i.size, env) <= DECODE_ROWS:   # decode: 16 row tiles, chunks per kv_chunk
            grid = ((16, 64, 4, 3), (16, 32, 4, 2), (16, 64, 4, 2), (16, 128, 8, 2))
        for bm, bn, w, st in grid:
            c = dict(BM=bm, BN=bn, num_warps=w, num_stages=st, LPT=True, SPLIT=True)
            c["KV_CHUNK"] = kv_chunk(p, env, c)
            out.append(c)
        out.append(dict(d, SPLIT=False))
        out.append(dict(d, LPT=False))
        if d["KV_CHUNK"] and not getattr(p, "deterministic", False):
            for ch in (d["KV_CHUNK"] // 2, d["KV_CHUNK"] * 2):
                if d["BN"] <= ch < _sz(p.j.size, env):
                    out.append(dict(d, KV_CHUNK=ch))
    seen, uniq = set(), []
    for c in out:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


# Tile configs (BM, BN, BK, warps, stages) the tuner may time, per operand class, in priority
# order: each won on some shape of the 2026-11-24 sweeps. candidates() prunes them by shape.
CONTRACTION_GRID = {
    "half": ((64, 64, 64, 4, 3), (64, 64, 64, 4, 4), (64, 64, 32, 4, 5), (64, 128, 64, 4, 3),
             (128, 128, 64, 4, 3), (32, 64, 128, 4, 3), (16, 64, 128, 4, 3), (16, 32, 128, 4, 3)),
    "tf32": ((64, 64, 32, 4, 3), (64, 64, 16, 4, 5), (128, 64, 32, 4, 3), (64, 128, 32, 4, 3),
             (128, 128, 32, 8, 3), (32, 64, 64, 4, 3), (16, 64, 64, 4, 3), (16, 32, 64, 4, 3)),
    "ieee": ((128, 256, 16, 8, 3), (128, 128, 32, 8, 3), (128, 128, 16, 4, 3), (128, 64, 16, 4, 3),
             (64, 128, 16, 4, 3), (32, 64, 32, 2, 4), (16, 64, 64, 2, 3), (16, 128, 32, 2, 4)),
}
CANDIDATES_MAX = 5  # contraction candidates besides the default, after pruning by shape


def _prune_contraction(p, env: Dict[str, int], grid: List[Dict]) -> List[Dict]:
    """The tile configs of `grid` worth timing for this shape, at most CANDIDATES_MAX.

    Every candidate costs one Triton compile (most of a search). A tile is dropped when it gives
    fewer than half a wave of programs while another tile fills one (it would starve the SMs), or
    when it wastes more than half its rows on a small M. Grid order is the priority order (tiles
    that won somewhere first)."""
    M, N = _msize(p, env), _sz(p.n.size, env)
    nb = _prod(p.batch, env)
    progs = [_programs(M, N, c["BM"], c["BN"]) * nb for c in grid]
    fills = any(n >= _waves(64) for n in progs)        # about a wave, then half a wave, of the SMs
    keep = []
    for c, n in zip(grid, progs):
        if fills and n < _waves(32):
            continue
        if c["BM"] >= 2 * _pow2(max(M, 16)):
            continue
        keep.append(c)
    return (keep or grid)[:CANDIDATES_MAX]


def _is_pow2(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x > 0 and x & (x - 1) == 0


def check_config(p, env: Dict[str, int], cfg) -> Optional[str]:
    """Why cfg cannot be a launch config for this plan and sizes, or None when it can."""
    if not isinstance(cfg, dict):
        return "config is not a mapping"
    if getattr(p, "portable", False) and p.kind in ("contraction", "conv") and portable_pin(p, cfg) != cfg:
        return "a portable tensor core kernel has a fixed K block and no split K"
    if p.kind in triton_conv.KEYS:
        return triton_conv.check_config(p, env, cfg)
    kind = config_kind(p, env)
    keys = _KEYS[kind]
    if set(cfg) != set(keys):
        return f"config keys {sorted(cfg)} do not match {sorted(keys)} for a {kind} kernel"
    for k_, v in cfg.items():
        if k_ in _BOOL_KEYS:
            if not isinstance(v, bool):
                return f"{k_} must be a bool, got {v!r}"
            continue
        if k_ == "num_stages":
            if not (isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 8):
                return f"num_stages must be an int from 1 to 8, got {v!r}"
            continue
        if k_ == "KV_CHUNK" and v == 0:     # no split KV (checked with the attention rules below)
            continue
        if not _is_pow2(v):
            return f"{k_} must be a positive power of two, got {v!r}"
    if cfg.get("num_warps", 4) > 16:
        return "num_warps above 16"
    for k_ in ("BR", "BC", "BM", "BN", "BK"):
        if k_ in cfg and cfg[k_] > 8192:
            return f"{k_} above 8192"
    if kind == "tile_pointwise" and cfg["BR"] * cfg["BC"] > 16384:
        return "tile block above 16384 elements"
    if kind == "tile_single":
        bc = max(16, _pow2(_sz(p.rsize, env)))
        if cfg["BR"] * bc > 32768:
            return "tile block above 32768 elements"
    if kind == "attention" and (cfg["BM"] < 16 or cfg["BN"] < 16 or cfg["BM"] > 256
                                or cfg["BN"] > 256):
        return "attention blocks must be between 16 and 256"
    if kind == "attention" and cfg["KV_CHUNK"]:
        ch = cfg["KV_CHUNK"]
        if not _is_pow2(ch) or ch < cfg["BN"]:
            return "KV_CHUNK must be 0 or a power of two of at least BN"
        if getattr(p, "deterministic", False) and ch != DET_KV_CHUNK:
            return f"a deterministic attention splits keys only in chunks of {DET_KV_CHUNK}"
        if kv_workspace_elems(p, env, cfg) >= 1 << 31:
            return "split KV workspace above 2^31 elements (offsets are 32 bit)"
    if kind == "contraction":
        if min(cfg["BM"], cfg["BN"], cfg["BK"]) < 16:
            return "tl.dot blocks must be at least 16"
        if cfg["SPLIT_K"] > MAX_SPLIT_K:
            return f"SPLIT_K above {MAX_SPLIT_K}"
        if cfg["SPLIT_K"] > 1 and getattr(p, "deterministic", False):
            if not getattr(p, "kchunk", 0):
                return "split K is off for a kernel compiled deterministic"
            if cfg["SPLIT_K"] > -(-_sz(p.k.size, env) // p.kchunk):
                return "a deterministic K split runs at most one program per K chunk"
        if cfg["SPLIT_K"] > 1 and k_slots(p, env, cfg) * workspace_elems(p, env) >= 1 << 31:
            return "split K workspace above 2^31 elements (offsets are 32 bit)"
    return None


MAX_SPLIT_K = 16


def workspace_elems(p, env: Dict[str, int]) -> int:
    """float32 elements of one split K partial: batch times M times N."""
    return _prod(p.batch, env) * _msize(p, env) * _sz(p.n.size, env)


def kv_workspace_elems(p, env: Dict[str, int], cfg: Dict) -> int:
    """float32 elements of a split KV workspace: chunks x batch x query rows x terms x (BD + 2)."""
    if not cfg.get("KV_CHUNK"):
        return 0
    bd, _ = _att_dims(p, env)
    nch = -(-_sz(p.j.size, env) // cfg["KV_CHUNK"])
    return nch * _prod(p.batch, env) * _sz(p.i.size, env) * len(p.terms) * (bd + 2)


def launch_workspace_elems(p, env: Dict[str, int], cfg: Dict) -> int:
    """float32 workspace a launch needs: split K partials, split KV partials, or 0."""
    if p.kind == "contraction":
        return k_slots(p, env, cfg) * workspace_elems(p, env)
    if p.kind == "attention":
        return kv_workspace_elems(p, env, cfg)
    return 0


def _no_fusion(module, meta: Dict) -> Dict:
    """meta with enable_fp_fusion=False for a module generated in round once mode."""
    if getattr(module, "TS_FP_FUSION", True) is False:
        meta = dict(meta, enable_fp_fusion=False)
    return meta


def split_k_epilogue(module, p, env: Dict[str, int], cfg: Dict):
    """(function, grid, meta) of the second pass of a split K or split KV launch, or None."""
    got = _split_k_epilogue(module, p, env, cfg)
    if got is None:
        return None
    return got[0], got[1], _no_fusion(module, got[2])


def _split_k_epilogue(module, p, env: Dict[str, int], cfg: Dict):
    """(function, grid, meta) of the second pass of a split K or split KV launch, or None."""
    if p.kind == "attention":
        if not cfg.get("KV_CHUNK"):
            return None
        import triton
        nm = triton.cdiv(_sz(p.i.size, env), cfg["BM"])
        nb = _prod(p.batch, env)
        bd, _ = _att_dims(p, env)
        meta = dict(BM=cfg["BM"], BD=bd, LPT=cfg["LPT"], KV_CHUNK=cfg["KV_CHUNK"], num_warps=4)
        return module.kern_splitkv, ((nb, nm) if cfg["LPT"] else (nm, nb)), meta
    if p.kind != "contraction" or cfg.get("SPLIT_K", 1) == 1:
        return None
    import triton
    tiles = triton.cdiv(_msize(p, env), cfg["BM"]) * triton.cdiv(_sz(p.n.size, env), cfg["BN"])
    meta = dict(BM=cfg["BM"], BN=cfg["BN"], GROUP=cfg["GROUP"], SPLIT_K=cfg["SPLIT_K"],
                num_warps=4)
    return module.kern_splitk, (tiles, _prod(p.batch, env)), meta


def work(p, env: Dict[str, int]) -> int:
    """Rough element operations of one launch, used to decide whether tuning is worth it."""
    if p.kind in triton_conv.KEYS:
        return triton_conv.work(p, env)
    kind = config_kind(p, env)
    if kind == "tile_pointwise":
        return _prod(p.rows, env) * (_sz(p.lane.size, env) if p.lane is not None else 1)
    if kind in ("tile_single", "tile_looped"):
        return _prod(p.rows, env) * _sz(p.rsize, env) * max(1, len(p.reduces))
    if kind == "contraction":
        return (_prod(p.batch, env) * _msize(p, env) * _sz(p.n.size, env)
                * _sz(p.k.size, env))
    c = sum(_sz(t.c.size, env) for t in p.terms) + _sz(p.d.size, env) * len(p.terms)
    return _prod(p.batch, env) * _sz(p.i.size, env) * _sz(p.j.size, env) * c


# Below these, a launch takes microseconds and tuning cannot pay for itself.
WORTH_TUNING = {"tile_pointwise": 1 << 24, "tile_single": 1 << 24, "tile_looped": 1 << 24,
                "contraction": 1 << 32, "attention": 1 << 28}


# Kernel kinds whose result bits do not depend on the launch config (docs/KERNEL_IR.md 6b): a
# pointwise tile computes every element on its own, and a contraction without split K accumulates
# each output over K in the same order in every tile shape (measured over 72 configs). Reductions
# (tile_single, tile_looped) and attention change their summation or softmax rescaling order with
# the block sizes and warps, so a config tuned by timing (which differs per GPU and per run) would
# change their bits; a kernel compiled deterministic keeps their default config.
CONFIG_FREE_KINDS = ("tile_pointwise", "contraction")


def bits_follow_config(p, env: Dict[str, int]) -> bool:
    """True when this launch's result bits can change with its launch config. A row reduction
    whose every sum is an order free sum (osum) has the same bits for every block size and warp
    count, so it is tuned even in deterministic mode (docs/ORDER_FREE_SUM.md). A max or min is
    exact and keeps that property; the one value it can leave order dependent is the sign of a
    zero maximum among +0 and -0 terms (documented there)."""
    if (p.kind == "tile" and p.reduces and any(r.op == "osum" for r in p.reduces)
            and all(r.op in ("osum", "max", "min") for r in p.reduces)):
        return False
    return config_kind(p, env) not in CONFIG_FREE_KINDS


# Launch parameters that keep a loop shape's summation order. LPT only maps tiles to programs in
# another order, num_stages only changes how far ahead loads are issued, and BR only puts more
# independent rows in one program: none of them changes which values a reduction adds or the order
# it adds them in. Block sizes, warps, SPLIT and KV_CHUNK do (warps change tl.sum's tree). Measured
# in scripts/order_classes.py: 18 cases (attention causal, full and decode, softmax and layer norm,
# float16 and float32), every config of a grid, the parameters below never changed a bit. A kernel
# compiled deterministic tunes only these, from its default config, and its choice is checked
# bitwise against the default on the tuning inputs (kernel/compiler.py).
ORDER_FREE = {"attention": ("LPT", "num_stages"), "tile_single": ("BR",)}
_ORDER_VALUES = {"LPT": (True, False), "num_stages": (1, 2, 3, 4), "BR": (1, 2, 4, 8, 16, 32)}


def order_key(p, env: Dict[str, int], cfg: Dict) -> tuple:
    """The part of a launch config that decides the summation order: two configs with the same key
    give the same bits."""
    free = ORDER_FREE.get(config_kind(p, env), ())
    return tuple(sorted((k, repr(v)) for k, v in cfg.items() if k not in free))


def order_candidates(p, env: Dict[str, int]) -> List[Dict]:
    """The default config and every valid config that differs from it only in ORDER_FREE parameters
    (the search space of a deterministic kernel), default first."""
    import itertools
    d = default_config(p, env)
    free = ORDER_FREE.get(config_kind(p, env), ())
    out = [d]
    for combo in itertools.product(*(_ORDER_VALUES[k] for k in free)):
        c = dict(d, **dict(zip(free, combo)))
        if c not in out and check_config(p, env, c) is None:
            out.append(c)
    return out


def worth_tuning(p, env: Dict[str, int]) -> bool:
    if p.kind in triton_conv.KEYS:
        return work(p, env) >= triton_conv.WORTH_TUNING[p.kind]
    return work(p, env) >= WORTH_TUNING[config_kind(p, env)]


def prepare(module, k: Kernel, p, env: Dict[str, int], config: Optional[Dict] = None):
    """As _prepare, with enable_fp_fusion=False added for round once modules."""
    fn, grid, extra, meta = _prepare(module, k, p, env, config)
    return fn, grid, extra, _no_fusion(module, meta)


def _prepare(module, k: Kernel, p, env: Dict[str, int], config: Optional[Dict] = None):
    """Everything about a launch that depends only on sizes: (function, grid, extra int args,
    meta kwargs). The caller passes pointers, then strides, symbol values and the extra args.
    config is a launch config (see candidates); None uses default_config."""
    import triton  # imported here so planning and codegen work without a GPU stack

    cfg = default_config(p, env) if config is None else portable_pin(p, config)
    why = check_config(p, env, cfg)
    if why is not None:
        raise ValueError(f"invalid launch config {cfg!r}: {why}")
    if p.kind in triton_conv.KEYS:
        return triton_conv.prepare(module, k, p, env, cfg)
    kind = config_kind(p, env)
    sz = lambda s: _sz(s, env)  # noqa: E731
    if p.kind == "tile":
        rows = _prod(p.rows, env)
        lanes = sz(p.lane.size) if p.lane is not None else 1
        if kind == "tile_pointwise":
            br, bc = cfg["BR"], cfg["BC"]
            grid = (triton.cdiv(rows, br), triton.cdiv(lanes, bc))
            return module.kern, grid, [rows, lanes], dict(BR=br, BC=bc,
                                                          num_warps=cfg["num_warps"])
        if kind == "tile_single":
            bc = max(16, _pow2(sz(p.rsize)))
            br = cfg["BR"]
            return (module.kern, (triton.cdiv(rows, br), 1), [rows, lanes],
                    dict(BR=br, BC=bc, num_warps=cfg["num_warps"]))
        return (module.kern_loop, (rows, 1), [rows, lanes],
                dict(BR=1, BC=cfg["BC"], num_warps=cfg["num_warps"]))
    if p.kind == "contraction":
        M, N = _msize(p, env), sz(p.n.size)
        nb = _prod(p.batch, env)
        tiles = triton.cdiv(M, cfg["BM"]) * triton.cdiv(N, cfg["BN"])
        split = cfg["SPLIT_K"]
        return module.kern, (tiles, nb, split), [], dict(cfg)
    NI = sz(p.i.size)
    nb = _prod(p.batch, env)
    bd, bcs = _att_dims(p, env)
    nm = triton.cdiv(NI, cfg["BM"])
    nch = triton.cdiv(sz(p.j.size), cfg["KV_CHUNK"]) if cfg["KV_CHUNK"] else 1
    grid = (nb, nm, nch) if cfg["LPT"] else (nm, nb, nch)
    meta = dict(cfg, BD=bd, **{f"BC{t}": b for t, b in enumerate(bcs)})
    return module.kern, grid, [], meta
