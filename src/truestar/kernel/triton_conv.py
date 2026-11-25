"""GPU loop shapes for convolution and pooling: strided loads, implicit GEMM and window loops.

Two shapes, tried after tile, contraction and attention (so they never take a kernel one of those
accepts):

* conv (implicit GEMM): a chain of two or more nested sums over a product ``a * b``, where the
  output axes split into M axes (only in ``a``), N axes (only in ``b``) and batch axes (in both).
  M axes are flattened into the GEMM rows (batch times output pixels for a convolution), N axes
  into the columns (output channels), the summed axes into K (input channels times kernel taps).
  The ``a`` operand is read through strided loads (``x[n, c, i*s + u*d - p, ...]``) and zero
  padding is a load mask, so no unfolded (im2col) tensor is ever written. Any elementwise
  epilogue on the sum (bias, a folded batch norm, relu, a residual add) runs on the accumulator
  before the one store.
* window: no tl.dot at all. Each program owns a block of output elements; every reduction axis has
  a static size and is a loop inside the program, nested as written. Pooling (max, average,
  adaptive average), depthwise and grouped convolution and the gather form of their gradients
  have this shape.

Strided loads themselves (``emit_strided_load``) work in every shape: an index
``(sum of coeff * ((axis * mul // div) % mod) + offset) / exact`` is computed per element, and an
element whose index is out of range or not exactly divisible is masked and reads the load's pad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from . import ir
from .ir import Axis, Binary, Kernel, Load, Reduce, Strided, free_axes, top_reduces


class _Unschedulable(Exception):
    pass


def _unsched():
    from .triton_backend import Unschedulable
    return Unschedulable


# =========================================================================== strided loads


def _pad_code(pad: Optional[float]) -> str:
    if pad is None:
        return "0.0"
    if math.isinf(pad):
        return "float('inf')" if pad > 0 else "float('-inf')"
    if math.isnan(pad):
        return "float('nan')"
    return repr(pad)


def check_accumulator(em, acc: str, mask: str):
    """Sentinels on a reduction's accumulator, before any epilogue: a NaN or an infinity made
    inside a convolution's sum (an infinite input times a zero weight) is flagged even when the
    epilogue hides it (relu's maximum returns 0 for NaN on the GPU). A predicated store, as for
    stored outputs (kernel/checks.py); nothing is written when the sum is clean."""
    from . import checks as _checks
    S = _checks.SLOT
    if not ({"nan", "inf"} & set(em.checks)):
        return
    zero = em.var(f"tl.zeros_like({acc}).to(tl.int32)", "cz")
    if "nan" in em.checks:
        em.line(f"tl.store(p_chk + {S['nan']} + {zero}, 1, mask=({acc} != {acc}) & ({mask}))")
    if "inf" in em.checks:
        em.line(f"tl.store(p_chk + {S['inf']} + {zero}, 1, mask=(tl.abs({acc}) == float('inf')) & ({mask}))")


def emit_strided_load(em, n: Load, ctx, raw: bool = False) -> str:
    """Triton code for a load with strided index entries (Emitter.load calls this)."""
    b = n.buffer
    off, masks = [], []

    def mask(mk):
        if mk and mk not in masks:
            masks.append(mk)

    for d, a in enumerate(n.index):
        if isinstance(a, Axis):
            idx, mk = ctx.m[a]
            off.append(f"({idx}) * st_{b.name}_{d}")
            mask(mk)
            continue
        if isinstance(a, int):
            if a:
                off.append(f"{a} * st_{b.name}_{d}")
            continue
        parts = []
        for t in a.terms:
            idx, mk = ctx.m[t.axis]
            mask(mk)
            v = f"({idx})" if t.mul == 1 else f"({idx}) * {t.mul}"
            if t.div != 1:
                v = f"(({v}) // {t.div})"
            if t.mod:
                v = f"(({v}) % {t.mod})"
            parts.append(v if t.coeff == 1 else f"{t.coeff} * {v}")
        if a.offset or not parts:
            parts.append(str(a.offset))
        num = em.var(" + ".join(parts), "si")
        size = em.size(b.shape[d])
        if a.exact != 1:
            # nonnegative first, so truncating and floor division agree
            mask(em.var(f"({num} >= 0) & ({num} % {a.exact} == 0) & ({num} < {size} * {a.exact})", "sv"))
            num = em.var(f"{num} // {a.exact}", "si")
        elif not a.always_valid(b.shape[d]):
            mask(em.var(f"({num} >= 0) & ({num} < {size})", "sv"))
        off.append(f"({num}) * st_{b.name}_{d}")
    offs = " + ".join(off) if off else "0"
    other = _pad_code(n.pad)
    mask = " & ".join(masks)
    code = f"tl.load(p_{b.name} + {offs}" + (f", mask={mask}, other={other})" if masks else ")")
    fresh = not any(code in sc for sc in em.scopes)
    v = em.var(code, "ld")
    checks = getattr(em, "load_checks", {}).get(b.name)
    if fresh and checks:
        # a buffer written by a torch step is checked where a kernel reads it (kernel/checks.py);
        # only elements really read count, never the pad of an index outside the buffer
        from . import checks as _checks
        zero = em.var(f"tl.zeros_like({v}).to(tl.int32)", "cz")
        sel = f" & ({mask})" if mask else ""
        for c in checks:
            bad = f"({v} != {v})" if c == "nan" else f"(tl.abs({v}.to(tl.float32)) == float('inf'))"
            em.line(f"tl.store(p_chk + {_checks.SLOT[c]} + {zero}, 1, mask={bad}{sel})")
    return v if raw else em.var(f"{v}.to(tl.float32)")


# =========================================================================== plans


@dataclass
class ConvPlan:
    kind = "conv"
    red: Reduce                   # outermost sum of the chain
    chain: Tuple[Axis, ...]       # summed axes, outermost first (flattened into K, last fastest)
    a: ir.Expr
    b: ir.Expr
    ms: Tuple[Axis, ...]          # output axes flattened into GEMM rows
    ns: Tuple[Axis, ...]          # output axes flattened into GEMM columns
    batch: Tuple[Axis, ...]
    tf32: bool = False


@dataclass
class WindowPlan:
    kind = "window"
    axes: Tuple[Axis, ...]
    reduces: List[Reduce]


WINDOW_MAX_TAPS = 4096   # product of the reduction sizes one output element loops over


def _sum_chain(red: Reduce):
    chain, inner = [red.axis], red.body
    while isinstance(inner, Reduce) and inner.op == "sum":
        chain.append(inner.axis)
        inner = inner.body
    return chain, inner


def plan_conv(k: Kernel) -> ConvPlan:
    U = _unsched()
    if len(k.outputs) != 1:
        raise U("conv kernels have one output")
    o = k.outputs[0]
    tops = top_reduces(o.expr)
    if len(tops) != 1 or tops[0].op != "sum":
        raise U("needs exactly one top level sum")
    red = tops[0]
    chain, inner = _sum_chain(red)
    if len(chain) < 2:
        raise U("needs a chain of two or more nested sums")
    if not _has_strided(k):
        raise U("no strided load (the conv shape is for convolution)")
    if len(k.reduces()) != len(chain):
        raise U("a reduction outside the summed chain")
    if not (isinstance(inner, Binary) and inner.op == "mul"):
        raise U("the summed body is not a product")
    a, b = inner.a, inner.b
    fa, fb = free_axes(a), free_axes(b)
    O, C = set(o.axes), set(chain)
    if not (fa <= O | C and fb <= O | C):
        raise U("a factor depends on an axis it cannot")
    ms = [x for x in o.axes if x in fa and x not in fb]
    ns = [x for x in o.axes if x in fb and x not in fa]
    if not ns and ms:
        a, b, fa, fb = b, a, fb, fa
        ms, ns = ns, ms
    if not ms or not ns:
        raise U("needs an output axis only in one factor and one only in the other (a depthwise or "
                "grouped conv takes the window shape)")
    batch = [x for x in o.axes if x in fa and x in fb]
    ms = [x for x in o.axes if x not in ns and x not in batch]   # axes in neither factor join M
    for e in (a, b):
        for n in ir.walk(e):
            if isinstance(n, Load) and n.pad not in (None, 0.0):
                raise U("a conv operand pads with a value other than zero")
    return ConvPlan(red, tuple(chain), a, b, tuple(ms), tuple(ns), tuple(batch))


def _gemm_shaped(red: Reduce, out_axes) -> bool:
    """True for a chain of sums over a product whose output axes split into some only in one
    factor and some only in the other: a convolution with a GEMM inside, which the conv shape
    runs on tensor cores and the window shape would run element by element."""
    if red.op != "sum":
        return False
    chain, inner = _sum_chain(red)
    if not (isinstance(inner, Binary) and inner.op == "mul"):
        return False
    fa, fb = free_axes(inner.a), free_axes(inner.b)
    return (any(x in fa and x not in fb for x in out_axes)
            and any(x in fb and x not in fa for x in out_axes))


def _has_strided(k: Kernel) -> bool:
    return any(isinstance(n, Load) and n.is_strided for o in k.outputs for n in ir.walk(o.expr))


def plan_window(k: Kernel) -> WindowPlan:
    U = _unsched()
    axes0 = k.outputs[0].axes
    if not _has_strided(k):
        raise U("no strided load (the window shape is for convolution and pooling)")
    reds = k.reduces()
    if any(r.op == "osum" for r in reds):
        raise U("order free sums run in the row reduction shapes only")
    for r in top_reduces(k.outputs[0].expr):
        if _gemm_shaped(r, axes0):
            raise U("a GEMM shaped sum (it takes the conv shape in a kernel of its own)")
    taps = 1
    for r in reds:
        if not isinstance(r.axis.size, int):
            raise U(f"window reductions need static sizes, {r.axis.name} is {r.axis.size}")
    for o in k.outputs:
        taps = max(taps, _taps(o.expr))
    if taps > WINDOW_MAX_TAPS:
        raise U(f"{taps} reduction steps per output element is too many for the window shape")
    return WindowPlan(axes0, reds)


def _taps(e) -> int:
    best = 1

    def go(n, acc):
        nonlocal best
        if isinstance(n, Reduce):
            acc = acc * n.axis.size
            best = max(best, acc)
        for c in n.children():
            go(c, acc)

    go(e, 1)
    return best


# =========================================================================== code generation


def _helpers():
    from . import triton_backend as tb
    return tb


def _dot_dtype(*exprs) -> str:
    dts = {n.buffer.dtype for e in exprs for n in ir.walk(e) if isinstance(n, Load)}
    if len(dts) == 1 and next(iter(dts)) in ("float16", "bfloat16"):
        return next(iter(dts))
    return "float32"


def _flat(em, axes, code: str) -> Dict[Axis, str]:
    """Row major decomposition of a flat 1D index (no decomposition for one axis)."""
    if len(axes) == 1:
        return {axes[0]: code}
    out, rest = {}, code
    for a in reversed(axes):
        out[a] = em.var(f"{rest} % {em.size(a.size)}", "ix")
        rest = em.var(f"{rest} // {em.size(a.size)}", "ix")
    return out


def _prod_code(em, axes) -> str:
    return " * ".join(em.size(a.size) for a in axes) or "1"


def gen_conv(k: Kernel, p: ConvPlan, opts) -> str:
    tb = _helpers()
    em = tb.Emitter(k, opts)
    o = k.outputs[0]
    sig = tb._signature(k, bool(em.checks)) + ["BM: tl.constexpr", "BN: tl.constexpr", "BK: tl.constexpr",
                              "GROUP: tl.constexpr"]
    head = [tb._jit_line(sig), f"def kern({', '.join(sig)}):"]
    M = em.var(_prod_code(em, p.ms), "M")
    N = em.var(_prod_code(em, p.ns), "N")
    Kc = em.var(_prod_code(em, p.chain), "K")
    em.line("pid = tl.program_id(0)")
    em.line(f"num_m = tl.cdiv({M}, BM)")
    em.line(f"num_n = tl.cdiv({N}, BN)")
    em.line("group = GROUP * num_n")
    em.line("first_m = (pid // group) * GROUP")
    em.line("gsize = tl.minimum(num_m - first_m, GROUP)")
    em.line("pm = first_m + (pid % group) % gsize")
    em.line("pn = (pid % group) // gsize")
    em.line("pb = tl.program_id(1)")
    bidx = tb._decompose(em, p.batch, "pb") if p.batch else {}
    base = {a: (bidx[a], None) for a in p.batch}
    em.line("rm = pm * BM + tl.arange(0, BM)")
    em.line("rn = pn * BN + tl.arange(0, BN)")
    em.line(f"mm = rm[:, None] < {M}")
    em.line(f"mn = rn[None, :] < {N}")
    mi = _flat(em, p.ms, "rm")
    ni = _flat(em, p.ns, "rn")
    dt = _dot_dtype(p.a, p.b)
    prec = "" if dt != "float32" else (", input_precision=\"tf32\"" if opts.get("allow_tf32")
                                       else ", input_precision=\"ieee\"")
    em.line("acc = tl.zeros((BM, BN), tl.float32)")
    if tap_mode(p):
        # one loop over (kernel tap, channel block): the tap's spatial offsets and padding masks are
        # per row of the tile, only scalars are divided, and the channel slab is contiguous K
        ch, taps = p.chain[0], p.chain[1:]
        C = em.size(ch.size)
        T = _prod_code(em, taps)
        em.line(f"nblk = tl.cdiv({C}, BK)")
        em.line(f"for kk in range(0, {T} * nblk):")
        em.push()
        em.line("tap = kk // nblk")
        em.line("rk = (kk % nblk) * BK + tl.arange(0, BK)")
        em.line(f"mka = rk[None, :] < {C}")
        em.line(f"mkb = rk[:, None] < {C}")
        ti = _flat(em, taps, "tap")
        ka = {ch: ("rk[None, :]", "mka"), **{a: (ti[a], None) for a in taps}}
        kb = {ch: ("rk[:, None]", "mkb"), **{a: (ti[a], None) for a in taps}}
    else:
        em.line(f"for k0 in range(0, {Kc}, BK):")
        em.push()
        em.line("rk = k0 + tl.arange(0, BK)")
        em.line(f"mka = rk[None, :] < {Kc}")
        em.line(f"mkb = rk[:, None] < {Kc}")
        ki = _flat(em, p.chain, "rk")
        ka = {a: (f"{ki[a]}[None, :]", "mka") for a in p.chain}
        kb = {a: (f"{ki[a]}[:, None]", "mkb") for a in p.chain}
    ca = tb.Ctx({**base, **{a: (f"{mi[a]}[:, None]", "mm") for a in p.ms}, **ka}, {})
    cb = tb.Ctx({**base, **kb, **{a: (f"{ni[a]}[None, :]", "mn") for a in p.ns}}, {})
    av = em.operand(p.a, ca, dt, "mka")
    av = em.full(av, p.a, tuple(p.ms) + tuple(p.chain), "(BM, BK)", tb._TL[dt])
    bv = em.operand(p.b, cb, dt, "mkb")
    bv = em.full(bv, p.b, tuple(p.chain) + tuple(p.ns), "(BK, BN)", tb._TL[dt])
    if tb._exact_tc(dt, opts):   # portable tensor cores: exact blocks added in K order
        em.line(f"acc = acc + {tb._exact_dot(av, bv, 'BK', dt)}")
    else:
        em.line(f"acc = tl.dot({av}, {bv}, acc{prec})")
    em.pop()
    em.values[p.red] = "acc"
    check_accumulator(em, "acc", "mm & mn")
    ce = tb.Ctx({**base, **{a: (f"{mi[a]}[:, None]", "mm") for a in p.ms},
                 **{a: (f"{ni[a]}[None, :]", "mn") for a in p.ns}}, {})
    val = em.ex(o.expr, ce)
    full = any(a in free_axes(o.expr) for a in p.ms) and any(a in free_axes(o.expr) for a in p.ns)
    tb._store(em, o, ce, val, "(BM, BN)", "mm & mn", full=full)
    return "\n".join(head + em.lines)


TAP_MAX = 64      # kernel taps (product of the inner summed axes) unrolled as one loop


def tap_mode(p: ConvPlan) -> bool:
    """True when the outermost summed axis is wide (channels) and the others are small fixed
    kernel taps: the loop then runs over (tap, channel block) instead of a flat K."""
    ch, taps = p.chain[0], p.chain[1:]
    if not all(isinstance(a.size, int) for a in taps):
        return False
    t = 1
    for a in taps:
        t *= a.size
    return t <= TAP_MAX and (not isinstance(ch.size, int) or ch.size >= 16)


_NEUTRAL = {"sum": "0.0", "max": "float('-inf')", "min": "float('inf')"}


def _pending(em, roots, bound: Set[Axis]) -> List[Reduce]:
    """Reductions needed by roots that can be computed with the bound axes, inner ones first."""
    out, seen = [], set()

    def go(n):
        if n in seen or n in em.values:
            return
        seen.add(n)
        if isinstance(n, Reduce):
            if free_axes(n) <= bound:
                go(n.body)
                out.append(n)
                return
            go(n.body)
            return
        for c in n.children():
            go(c)

    for r in roots:
        go(r)
    return out


STATIC_UNROLL = 8   # loops of at most this many steps are unrolled (tl.static_range)


def _emit_window_reduce(em, r: Reduce, ctx, bound: Set[Axis]):
    acc = em.fresh("acc")
    em.line(f"{acc} = tl.full((BLOCK,), {_NEUTRAL[r.op]}, tl.float32)")
    lv = em.fresh("w")
    rng = "tl.static_range" if r.axis.size <= STATIC_UNROLL else "range"
    em.line(f"for {lv} in {rng}({r.axis.size}):")
    em.push()
    saved = dict(em.values)
    inner = bound | {r.axis}
    c = ctx.extend({r.axis: (lv, None)})
    for sub in _pending(em, [r.body], inner):
        _emit_window_reduce(em, sub, c, inner)
    v = em.ex(r.body, c)
    comb = {"sum": f"{acc} + {v}", "max": f"tl.maximum({acc}, {v})",
            "min": f"tl.minimum({acc}, {v})"}[r.op]
    em.line(f"{acc} = {comb}")
    em.values = saved
    em.pop()
    em.values[r] = acc
    if r.op == "sum":     # a max or min of pooled values is checked when stored
        check_accumulator(em, acc, "omask")


def gen_window(k: Kernel, p: WindowPlan, opts) -> str:
    tb = _helpers()
    em = tb.Emitter(k, opts)
    sig = tb._signature(k, bool(em.checks)) + ["ROWS", "BLOCK: tl.constexpr"]
    head = [tb._jit_line(sig), f"def kern({', '.join(sig)}):"]
    em.line("offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)")
    em.line("omask = offs < ROWS")
    idx = _flat(em, p.axes, "offs") if p.axes else {}
    ctx = tb.Ctx({a: (idx[a], "omask") for a in p.axes}, {})
    bound = set(p.axes)
    for o in k.outputs:
        for r in _pending(em, [o.expr], bound):
            _emit_window_reduce(em, r, ctx, bound)
        val = em.ex(o.expr, ctx)
        tb._store(em, o, ctx, val, "(BLOCK,)", "omask", full=False)
    return "\n".join(head + em.lines)


# =========================================================================== launch configs


def _sz(s, env):
    return s if isinstance(s, int) else env[s.name]


def _prod(axes, env) -> int:
    n = 1
    for a in axes:
        n *= _sz(a.size, env)
    return n


def conv_class(p: ConvPlan) -> str:
    if _dot_dtype(p.a, p.b) != "float32":
        return "half"
    return "tf32" if p.tf32 else "ieee"


def _cc(bm, bn, bk, w, st):
    return dict(BM=bm, BN=bn, BK=bk, GROUP=8, num_warps=w, num_stages=st)


def default_config(p, env) -> Dict:
    if p.kind == "window":
        return dict(BLOCK=512, num_warps=4)
    M, N = _prod(p.ms, env), _prod(p.ns, env)
    cls = conv_class(p)
    bn = 32 if N <= 32 else 64
    progs = lambda bm, bn_: -(-M // bm) * -(-N // bn_)  # noqa: E731
    if cls == "half":
        if N >= 128 and progs(128, 128) >= 140:
            return _cc(128, 128, 32, 8, 3)
        if progs(128, bn) >= 140:
            return _cc(128, bn, 32, 4, 3)
        return _cc(64, bn, 32, 4, 3)
    if cls == "tf32":
        if progs(128, bn) >= 140:
            return _cc(128, bn, 32, 4, 3)
        return _cc(64, bn, 32, 4, 3)
    if N >= 128 and progs(128, 128) >= 70:
        return _cc(128, 128, 16, 8, 2)
    return _cc(64, bn, 16, 4, 2)


def candidates(p, env) -> List[Dict]:
    d = default_config(p, env)
    out = [d]
    if p.kind == "window":
        for blk, w in ((128, 4), (256, 4), (1024, 4), (1024, 8), (2048, 8)):
            out.append(dict(BLOCK=blk, num_warps=w))
    else:
        cls = conv_class(p)
        if cls == "ieee":
            grid = ((128, 128, 16, 8, 2), (128, 64, 16, 4, 2), (64, 64, 16, 4, 2), (64, 128, 16, 4, 2),
                    (128, 64, 32, 4, 2), (64, 32, 32, 4, 2), (32, 64, 32, 2, 3))
        else:
            grid = ((128, 128, 32, 8, 3), (128, 64, 32, 4, 3), (64, 64, 32, 4, 3), (64, 128, 32, 4, 3),
                    (128, 64, 64, 4, 3), (64, 64, 64, 4, 3), (64, 32, 64, 4, 3), (32, 64, 64, 4, 3),
                    (128, 32, 32, 4, 4))
        for g in grid:
            out.append(_cc(*g))
    seen, uniq = set(), []
    for c in out:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


KEYS = {"conv": ("BM", "BN", "BK", "GROUP", "num_warps", "num_stages"),
        "window": ("BLOCK", "num_warps")}


def check_config(p, env, cfg) -> Optional[str]:
    keys = KEYS[p.kind]
    if set(cfg) != set(keys):
        return f"config keys {sorted(cfg)} do not match {sorted(keys)} for a {p.kind} kernel"
    for k_, v in cfg.items():
        if not (isinstance(v, int) and not isinstance(v, bool) and v > 0):
            return f"{k_} must be a positive int, got {v!r}"
        if k_ not in ("num_stages", "GROUP") and v & (v - 1):
            return f"{k_} must be a power of two, got {v!r}"
    if cfg.get("num_warps", 4) > 16:
        return "num_warps above 16"
    if p.kind == "conv":
        if min(cfg["BM"], cfg["BN"], cfg["BK"]) < 16:
            return "tl.dot blocks must be at least 16"
        if max(cfg["BM"], cfg["BN"], cfg["BK"]) > 256 or cfg["num_stages"] > 8:
            return "conv blocks above 256 or more than 8 stages"
    elif cfg["BLOCK"] > 8192:
        return "BLOCK above 8192"
    return None


def work(p, env) -> int:
    if p.kind == "window":
        return _prod(p.axes, env) * max(1, max([_taps(r) for r in p.reduces] or [1]))
    return _prod(p.batch, env) * _prod(p.ms, env) * _prod(p.ns, env) * _prod(p.chain, env)


WORTH_TUNING = {"conv": 1 << 26, "window": 1 << 22}


def batch_symbols(p) -> List[str]:
    """Symbolic sizes that only repeat the work: the window's output axes and the conv's batch
    axes. A conv's M axes (batch times pixels) stay in the tuning key, as a contraction's M does."""
    axes = p.axes if p.kind == "window" else tuple(p.batch)   # a conv's M (batch x pixels) is tuned per bucket
    return sorted({a.size.name for a in axes if not isinstance(a.size, int)})


def prepare(module, k: Kernel, p, env, cfg):
    import triton
    if p.kind == "window":
        rows = _prod(p.axes, env)
        return (module.kern, (triton.cdiv(rows, cfg["BLOCK"]),), [rows],
                dict(BLOCK=cfg["BLOCK"], num_warps=cfg["num_warps"]))
    M, N = _prod(p.ms, env), _prod(p.ns, env)
    grid = (triton.cdiv(M, cfg["BM"]) * triton.cdiv(N, cfg["BN"]), _prod(p.batch, env))
    return module.kern, grid, [], dict(cfg)
