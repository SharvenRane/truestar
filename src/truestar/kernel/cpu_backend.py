"""CPU backend: generate a numba loop nest from any kernel.

Every kernel is schedulable on CPU. Each output gets a loop nest over its axes; the outermost loop
runs in parallel (prange). A reduction is computed once at the shallowest loop depth where all the
axes it depends on are bound, so softmax statistics are computed once per row, not once per
element. Elementwise subexpressions are inlined and left to LLVM to simplify.

Arithmetic runs in float64 inside the loops (numba promotes float literals), and the result is
rounded to the output dtype on store. float16 and bfloat16 inputs are widened to float32 before the
call, since numba has no half precision arithmetic.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set

from . import checks as _checks
from . import ir
from .ir import Axis, Binary, Const, Dim, Expr, Index, Kernel, Load, Reduce, Unary, Where, free_axes

HEADER = '''import math
import numba
import numpy as np
OS_BITS = 39   # kernel/ordersum.PRECISION_BITS (tests/test_ordersum.py checks they match)
OS_SHIFT = OS_BITS - 24


@numba.njit(inline="always")
def _max(a, b):
    return a if (a >= b or a != a) else b


@numba.njit(inline="always")
def _min(a, b):
    return a if (a <= b or a != a) else b


@numba.njit(inline="always")
def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


@numba.njit(inline="always")
def _rsqrt(x):
    return 1.0 / math.sqrt(x) if x > 0 else (math.inf if x == 0 else math.nan)


@numba.njit(inline="always")
def _log(x):
    return math.log(x) if x > 0 else (-math.inf if x == 0 else math.nan)


@numba.njit(inline="always")
def _sqrt(x):
    return math.sqrt(x) if x >= 0 else math.nan


@numba.njit(inline="always")
def _div(a, b):
    if b != 0:
        return a / b
    if a != a or a == 0:
        return math.nan
    return math.inf if (a > 0) == (math.copysign(1.0, b) > 0) else -math.inf


@numba.njit(inline="always")
def _divc(a, b, chk):
    if b == 0:
        chk[3] = 1
    return _div(a, b)


# order free sum (kernel/ordersum.py): exponent field of a float32 value held in a float64
@numba.njit(inline="always")
def _os_exp(v):
    if v != v or v == 0.0 or abs(v) == math.inf:
        return 1
    fe = math.frexp(abs(v))[1]
    return max(fe + 126, 1)


@numba.njit(inline="always")
def _os_q(v, emax):
    if v != v or v == 0.0 or abs(v) == math.inf:
        return 0
    a = abs(v)
    ee = max(math.frexp(a)[1] + 126, 1)
    m = np.int64(math.ldexp(a, 150 - ee))
    sh = ee - emax + OS_SHIFT
    if sh >= 0:
        q = m << sh
    else:
        s = min(-sh, 62)
        hi = m >> s
        rem = m & ((np.int64(1) << s) - 1)
        half = np.int64(1) << (s - 1)
        q = hi + (1 if (rem > half or (rem == half and (hi & 1) == 1)) else 0)
    return -q if v < 0 else q


@numba.njit(inline="always")
def _os_finish(tot, emax, nan, pinf, ninf):
    if nan > 0 or (pinf > 0 and ninf > 0):
        return math.nan
    if pinf > 0:
        return math.inf
    if ninf > 0:
        return -math.inf
    if tot == 0:
        return 0.0
    f = np.float32(tot)
    L = emax - 126 - OS_BITS
    if L < -126:
        r = f * np.float32(math.ldexp(1.0, L + 64))
        r = r * np.float32(2.0 ** -64)
    else:
        r = f * np.float32(math.ldexp(1.0, L))
    return float(r)


'''


class _Gen:
    def __init__(self, k: Kernel, opts):
        self.k = k
        self.opts = opts
        self.lines: List[str] = []
        self.ind = 1
        self.n = 0
        self.loopvar: Dict[Axis, str] = {}
        self.values: Dict[Expr, str] = {}
        self.checks = frozenset(c for c in str(opts.get("checks", "")).split(",") if c)

    def line(self, s):
        self.lines.append("    " * self.ind + s)

    def size(self, s):
        return str(s) if isinstance(s, int) else f"S_{s.name}"

    def fresh(self, p):
        self.n += 1
        return f"{p}{self.n}"

    def ex(self, e: Expr) -> str:
        if e in self.values:
            return self.values[e]
        if isinstance(e, Const):
            if isinstance(e.value, bool):
                return "True" if e.value else "False"
            v = e.value
            if math.isinf(v):
                return "math.inf" if v > 0 else "(-math.inf)"
            if math.isnan(v):
                return "math.nan"
            return f"({v!r})"
        if isinstance(e, Dim):
            return f"float({self.size(e.size)})"
        if isinstance(e, Index):
            return self.loopvar[e.axis]
        if isinstance(e, Load):
            if e.is_strided:
                return self.strided_load(e)
            ix = ", ".join(self.loopvar[a] if isinstance(a, Axis) else str(a) for a in e.index)
            return f"{e.buffer.name}[{ix}]"
        if isinstance(e, Unary):
            x = self.ex(e.x)
            code = {"neg": f"(-{x})", "exp": f"math.exp({x})", "log": f"_log({x})",
                    "sqrt": f"_sqrt({x})", "rsqrt": f"_rsqrt({x})", "sigmoid": f"_sigmoid({x})",
                    "tanh": f"math.tanh({x})", "abs": f"abs({x})", "not": f"(not {x})",
                    "erf": f"math.erf({x})", "cos": f"math.cos({x})", "sin": f"math.sin({x})",
                    "expm1": f"math.expm1({x})"}[e.op]
            if self.opts.get("math") == "round_once":
                # docs/ROUND_ONCE.md: the same float64 sequence as the GPU, rounded once to float32;
                # sqrt of a float32 value in float64 rounded to float32 is the IEEE float32 sqrt
                if e.op in _ROUND_ONCE:
                    code = f"_ro_{e.op}({x})"
                elif e.op == "sqrt":
                    code = f"np.float32(math.sqrt(np.float64(np.float32({x}))))"
            if e.op in self.opts.get("_break", {}):
                code = self.opts["_break"][e.op](x)
            return code
        if isinstance(e, Binary):
            a, b = self.ex(e.a), self.ex(e.b)
            sym = {"add": "+", "sub": "-", "mul": "*", "lt": "<", "le": "<=", "gt": ">",
                   "ge": ">=", "eq": "==", "ne": "!=", "and": "and", "or": "or"}
            if e.op in sym:
                code = f"({a} {sym[e.op]} {b})"
            elif e.op == "div" and "div" in self.checks and _checks.may_be_zero(e.b):
                code = f"_divc({a}, {b}, chk)"
            elif e.op == "div":
                code = f"_div({a}, {b})"
            elif e.op == "maximum":
                code = f"_max({a}, {b})"
            else:
                code = f"_min({a}, {b})"
            if e.op in self.opts.get("_break", {}):
                code = self.opts["_break"][e.op](a, b)
            return code
        if isinstance(e, Where):
            return f"({self.ex(e.a)} if {self.ex(e.cond)} else {self.ex(e.b)})"
        if isinstance(e, Reduce):
            raise ir.IRError("internal: reduction used before it was computed")
        raise ir.IRError(f"cpu backend cannot emit {type(e).__name__}")

    def strided_load(self, e: Load) -> str:
        """buf[...] where every strided index is exact and in range, else the load's pad value.
        Python integer semantics (floor division, nonnegative modulo), as in the reference."""
        ix, conds = [], []
        for d, a in enumerate(e.index):
            if isinstance(a, Axis):
                ix.append(self.loopvar[a])
                continue
            if isinstance(a, int):
                ix.append(str(a))
                continue
            parts = []
            for t in a.terms:
                v = self.loopvar[t.axis] if t.mul == 1 else f"({self.loopvar[t.axis]} * {t.mul})"
                if t.div != 1:
                    v = f"({v} // {t.div})"
                if t.mod:
                    v = f"({v} % {t.mod})"
                parts.append(v if t.coeff == 1 else f"({t.coeff} * {v})")
            num = " + ".join(parts + [str(a.offset)])
            if a.exact != 1:
                conds.append(f"({num}) % {a.exact} == 0")
                num = f"({num}) // {a.exact}"
            v = f"({num})"
            size = self.size(e.buffer.shape[d])
            if not a.always_valid(e.buffer.shape[d]):
                conds.append(f"0 <= {v} < {size}")
            ix.append(v)
        load = f"{e.buffer.name}[{', '.join(ix)}]"
        if not conds or e.pad is None:
            return load
        pad = e.pad
        pv = "math.inf" if pad == math.inf else "(-math.inf)" if pad == -math.inf else \
            "math.nan" if pad != pad else f"({pad!r})"
        return f"({load} if ({' and '.join(conds)}) else {pv})"

    def pending(self, roots: Sequence[Expr], bound: Set[Axis]) -> List[Reduce]:
        """Reductions needed by roots, computable with the bound axes, not yet computed."""
        out: List[Reduce] = []
        seen = set()

        def go(n):
            if n in seen or n in self.values:
                return
            seen.add(n)
            if isinstance(n, Reduce):
                if free_axes(n) <= bound:
                    go(n.body)     # nested reductions that do not need its axis come first
                    out.append(n)  # the rest are computed inside its loop
                    return
                go(n.body)  # depends on a deeper axis; look for hoistable parts inside
                return
            for c in n.children():
                go(c)

        for r in roots:
            go(r)
        return out

    def emit_reduce(self, r: Reduce, bound: Set[Axis]):
        if r.op == "osum":
            return self.emit_osum(r, bound)
        acc = self.fresh("acc")
        init = {"sum": "0.0", "max": "-math.inf", "min": "math.inf"}[r.op]
        self.line(f"{acc} = {init}")
        lv = self.fresh("r")
        self.loopvar[r.axis] = lv
        self.line(f"for {lv} in range({self.size(r.axis.size)}):")
        self.ind += 1
        inner = bound | {r.axis}
        saved = dict(self.values)
        for sub in self.pending([r.body], inner):
            self.emit_reduce(sub, inner)
        v = self.ex(r.body)
        if r.op == "sum":
            self.line(f"{acc} += {v}")
        elif r.op == "max":
            self.line(f"{acc} = _max({acc}, {v})")
        else:
            self.line(f"{acc} = _min({acc}, {v})")
        self.values = saved  # values computed inside the loop are not visible outside
        self.ind -= 1
        self.values[r] = acc

    def emit_osum(self, r: Reduce, bound: Set[Axis]):
        """Order free sum (kernel/ordersum.py): the body is rounded to float32, the anchor
        exponent is found in a first pass, the exact int64 total in a second."""
        emax, tot, nan, pinf, ninf = (self.fresh(p) for p in ("oe", "ot", "on", "op", "om"))
        self.line(f"{emax} = 1")
        for v in (tot, nan, pinf, ninf):
            self.line(f"{v} = np.int64(0)")
        inner = bound | {r.axis}
        for step in (0, 1):
            lv = self.fresh("r")
            self.loopvar[r.axis] = lv
            self.line(f"for {lv} in range({self.size(r.axis.size)}):")
            self.ind += 1
            saved = dict(self.values)
            for sub in self.pending([r.body], inner):
                self.emit_reduce(sub, inner)
            v = self.fresh("ov")
            self.line(f"{v} = float(np.float32({self.ex(r.body)}))")
            if step == 0:
                self.line(f"{emax} = max({emax}, _os_exp({v}))")
            else:
                self.line(f"{tot} += _os_q({v}, {emax})")
                self.line(f"if {v} != {v}:")
                self.line(f"    {nan} += 1")
                self.line(f"elif {v} == math.inf:")
                self.line(f"    {pinf} += 1")
                self.line(f"elif {v} == -math.inf:")
                self.line(f"    {ninf} += 1")
            self.values = saved
            self.ind -= 1
        acc = self.fresh("acc")
        self.line(f"{acc} = _os_finish({tot}, {emax}, {nan}, {pinf}, {ninf})")
        self.values[r] = acc

    def emit_output(self, o: ir.Output):
        saved = dict(self.values)
        bound: Set[Axis] = set()
        for r in self.pending([o.expr], bound):
            self.emit_reduce(r, bound)
        depth = 0
        for d, a in enumerate(o.axes):
            lv = self.fresh("i")
            self.loopvar[a] = lv
            rng = "numba.prange" if d == 0 else "range"
            self.line(f"for {lv} in {rng}({self.size(a.size)}):")
            self.ind += 1
            depth += 1
            bound = bound | {a}
            for r in self.pending([o.expr], bound):
                self.emit_reduce(r, bound)
        ix = ", ".join(self.loopvar[a] for a in o.axes) if o.axes else "0"
        self.line(f"{o.name}[{ix}] = {self.ex(o.expr)}")
        if "nan" in self.checks:
            self.line(f"if {o.name}[{ix}] != {o.name}[{ix}]:")
            self.line(f"    chk[{_checks.SLOT['nan']}] = 1")
        if "inf" in self.checks:
            self.line(f"if abs({o.name}[{ix}]) == math.inf:")
            self.line(f"    chk[{_checks.SLOT['inf']}] = 1")
        self.ind -= depth
        self.values = saved


_ROUND_ONCE = ("exp", "log", "tanh", "erf", "rsqrt", "sigmoid")


def generate(k: Kernel, opts) -> str:
    g = _Gen(k, opts)
    head = HEADER
    if opts.get("math") == "round_once":
        from . import roundonce
        head += "\n\n" + roundonce.numba_source(roundonce.used(k)) + "\n\n"
    args = [b.name for b in sorted(k.inputs, key=lambda b: b.name)]
    args += [o.name for o in k.outputs]
    args += [f"S_{s.name}" for s in sorted(k.symbols(), key=lambda s: s.name)]
    if g.checks:
        args.append("chk")    # int32 flag words (kernel/checks.py)
    for o in k.outputs:
        g.emit_output(o)
    body = "\n".join(g.lines)
    return (head + "@numba.njit(parallel=True, cache=True, fastmath=False)\n"
            + f"def kern({', '.join(args)}):\n{body}\n")
