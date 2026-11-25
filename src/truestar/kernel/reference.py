"""Float64 reference interpreter for the kernel IR.

Slow and obviously correct: every expression is evaluated as a NumPy float64 array whose dims are
labelled by axes, with broadcasting by label. Sums of products use einsum so contractions stay
cheap in memory. This is the ground truth every compiled kernel is checked against.
"""

from __future__ import annotations

import itertools
import math
import string
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import ir
from .ir import Axis, Binary, Const, Dim, Index, Kernel, Load, Reduce, Unary, Where

Named = Tuple[np.ndarray, Tuple[Axis, ...]]


class ShapeError(ValueError):
    """Input shapes do not fit the kernel's buffers."""


def bind_sizes(k: Kernel, shapes: Dict[str, Sequence[int]]) -> Dict[str, int]:
    """Bind every symbol from the input shapes, checking fixed sizes. Returns {symbol: value}."""
    env: Dict[str, int] = {}
    where_bound: Dict[str, str] = {}
    for b in k.inputs:
        if b.name not in shapes:
            raise ShapeError(f"kernel {k.name}: missing input {b.name}")
        shp = tuple(int(s) for s in shapes[b.name])
        if len(shp) != len(b.shape):
            raise ShapeError(f"kernel {k.name}: input {b.name} has {len(shp)} dims, "
                             f"expected {len(b.shape)} ({list(b.shape)})")
        for d, (want, got) in enumerate(zip(b.shape, shp)):
            if isinstance(want, int):
                if want != got:
                    raise ShapeError(f"kernel {k.name}: input {b.name} dim {d} must be {want}, "
                                     f"got {got}")
            else:
                if want.name in env and env[want.name] != got:
                    raise ShapeError(
                        f"kernel {k.name}: input {b.name} dim {d} is {want.name}={got}, but "
                        f"{where_bound[want.name]} already set {want.name}={env[want.name]}")
                env.setdefault(want.name, got)
                where_bound.setdefault(want.name, f"{b.name} dim {d}")
    for s in k.symbols():
        if s.name not in env:
            raise ShapeError(f"kernel {k.name}: symbol {s.name} is not the size of any input dim, "
                             f"so it cannot be bound")
    return env


def size_of(s, env: Dict[str, int]) -> int:
    return s if isinstance(s, int) else env[s.name]


def _align(vals: List[Named]) -> Tuple[List[np.ndarray], Tuple[Axis, ...]]:
    axes: List[Axis] = []
    for _, ax in vals:
        for a in ax:
            if a not in axes:
                axes.append(a)
    out = []
    for arr, ax in vals:
        if ax == tuple(axes):
            out.append(arr)
            continue
        order = [ax.index(a) for a in axes if a in ax]
        t = np.transpose(arr, order) if arr.ndim else arr
        shape = [arr.shape[ax.index(a)] if a in ax else 1 for a in axes]
        out.append(np.reshape(t, shape))
    return out, tuple(axes)


_UNARY = {
    "neg": np.negative, "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "rsqrt": lambda x: 1.0 / np.sqrt(x), "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "tanh": np.tanh, "abs": np.abs, "not": np.logical_not,
    "erf": np.vectorize(math.erf, otypes=[np.float64]), "cos": np.cos, "sin": np.sin, "expm1": np.expm1,
}
_BINARY = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide,
    "maximum": np.maximum, "minimum": np.minimum, "lt": np.less, "le": np.less_equal,
    "gt": np.greater, "ge": np.greater_equal, "eq": np.equal, "ne": np.not_equal,
    "and": np.logical_and, "or": np.logical_or,
}


class _NP:
    """The array functions load_strided needs, for NumPy."""
    arange = staticmethod(lambda lo, hi: np.arange(lo, hi, dtype=np.int64))
    clip = staticmethod(np.clip)
    where = staticmethod(np.where)
    broadcast_to = staticmethod(np.broadcast_to)
    asarray = staticmethod(lambda v: np.asarray(v, dtype=np.int64))


def _torch_ns(torch, device):
    class _T:
        arange = staticmethod(lambda lo, hi: torch.arange(lo, hi, dtype=torch.int64, device=device))
        clip = staticmethod(torch.clamp)
        broadcast_to = staticmethod(torch.broadcast_to)
        asarray = staticmethod(lambda v: torch.as_tensor(v, dtype=torch.int64, device=device))

        @staticmethod
        def where(c, a, b):
            return torch.where(c, a, torch.as_tensor(b, dtype=a.dtype, device=device))
    return _T


def load_strided(e: Load, arr, axis_range, xp):
    """Gather a load with strided index entries. Every axis the load depends on becomes one dim of
    the result (in order of first appearance); an index that is not exactly divisible or falls
    outside its buffer dim reads ``e.pad``. Floor division and a nonnegative modulo, as in Python."""
    axes = e.axes()
    nd = len(axes)
    ranges = {a: axis_range(a) for a in axes}

    def axval(a):
        lo, hi = ranges[a]
        shape = [1] * nd
        shape[axes.index(a)] = hi - lo
        return xp.arange(lo, hi).reshape(shape)

    idxs, valid = [], None
    for d, a in enumerate(e.index):
        if isinstance(a, Axis):
            idxs.append(axval(a))
            continue
        if isinstance(a, int):
            idxs.append(a)
            continue
        size = arr.shape[d]
        num = xp.asarray(a.offset)
        for t in a.terms:
            v = (axval(t.axis) * t.mul) // t.div
            if t.mod:
                v = v % t.mod
            num = num + t.coeff * v
        ok = None
        if a.exact != 1:
            ok = (num % a.exact) == 0
            num = num // a.exact
        inside = (num >= 0) & (num < size)
        ok = inside if ok is None else ok & inside
        valid = ok if valid is None else valid & ok
        idxs.append(xp.clip(num, 0, size - 1))
    full = tuple(hi - lo for lo, hi in (ranges[a] for a in axes))
    v = xp.broadcast_to(arr[tuple(idxs)], full)
    if e.pad is not None and valid is not None:
        v = xp.where(xp.broadcast_to(valid, full), v, e.pad)
    return v, axes


class Evaluator:
    def __init__(self, inputs: Dict[str, np.ndarray], env: Dict[str, int]):
        self.inputs = {k: np.asarray(v, dtype=np.float64) for k, v in inputs.items()}
        self.env = env
        self.memo: Dict[ir.Expr, Named] = {}

    def size(self, s) -> int:
        return size_of(s, self.env)

    def ev(self, e: ir.Expr) -> Named:
        if e in self.memo:
            return self.memo[e]
        r = self._ev(e)
        self.memo[e] = r
        return r

    def _ev(self, e) -> Named:
        if isinstance(e, Const):
            return np.array(e.value, dtype=bool if isinstance(e.value, bool) else np.float64), ()
        if isinstance(e, Dim):
            return np.array(float(self.size(e.size))), ()
        if isinstance(e, Index):
            return np.arange(self.size(e.axis.size), dtype=np.float64), (e.axis,)
        if isinstance(e, Load):
            arr = self.inputs[e.buffer.name]
            if e.is_strided:
                return load_strided(e, arr, lambda a: (0, self.size(a.size)), _NP)
            sl = tuple(a if isinstance(a, int) else slice(None) for a in e.index)
            v = arr[sl]
            return v, tuple(a for a in e.index if isinstance(a, Axis))
        if isinstance(e, Unary):
            x, ax = self.ev(e.x)
            return _UNARY[e.op](x), ax
        if isinstance(e, Binary):
            (a, b), ax = _align([self.ev(e.a), self.ev(e.b)])
            return _BINARY[e.op](a, b), ax
        if isinstance(e, Where):
            (c, a, b), ax = _align([self.ev(e.cond), self.ev(e.a), self.ev(e.b)])
            return np.where(c, a, b), ax
        if isinstance(e, Reduce):
            return self._reduce(e if e.op != "osum" else Reduce("sum", e.axis, e.body))  # same value in float64
        raise ir.IRError(f"reference: cannot evaluate {type(e).__name__}")

    def _chain(self, e: Reduce, einsum, widen):
        """A chain of nested sums over a product (a convolution sums over channels and both kernel
        taps) as one einsum, so the intermediate never holds a reduced axis. None if e is not one."""
        chain, inner = [e.axis], e.body
        while isinstance(inner, Reduce) and inner.op == "sum" and inner not in self.memo:
            chain.append(inner.axis)
            inner = inner.body
        if len(chain) < 2 or not (isinstance(inner, Binary) and inner.op == "mul"):
            return None
        a, aa = self.ev(inner.a)
        b, ba = self.ev(inner.b)
        letters = {}
        for x in aa + ba:
            letters.setdefault(x, string.ascii_letters[len(letters)])
        out_axes = tuple(x for x in letters if x not in chain)
        so = "".join(letters[x] for x in out_axes)
        r = einsum(f"{''.join(letters[x] for x in aa)},{''.join(letters[x] for x in ba)}->{so}",
                   widen(a), widen(b))
        for x in chain:
            if x not in letters:
                r = r * self.size(x.size)
        return r, out_axes

    def _reduce(self, e: Reduce) -> Named:
        n = self.size(e.axis.size)
        body = e.body
        if e.op == "sum":
            r = self._chain(e, lambda s, a, b: np.einsum(s, a, b, optimize=True), lambda x: x)
            if r is not None:
                return r
        if e.op == "sum" and isinstance(body, Binary) and body.op == "mul":
            a, aa = self.ev(body.a)
            b, ba = self.ev(body.b)
            if e.axis in aa or e.axis in ba:
                letters = {}
                for x in aa + ba:
                    if x not in letters:
                        letters[x] = string.ascii_letters[len(letters)]
                if e.axis not in aa and e.axis not in ba:
                    pass
                out_axes = tuple(x for x in letters if x != e.axis)
                sa = "".join(letters[x] for x in aa)
                sb = "".join(letters[x] for x in ba)
                so = "".join(letters[x] for x in out_axes)
                r = np.einsum(f"{sa},{sb}->{so}", a, b, optimize=True)
                if e.axis not in aa and e.axis not in ba:
                    r = r * n
                return r, out_axes
        v, ax = self.ev(body)
        if e.axis not in ax:  # body does not vary along the axis
            if e.op == "sum":
                return v * n, ax
            return v, ax
        k = ax.index(e.axis)
        if e.op == "sum":
            r = np.sum(v, axis=k)
        elif e.op == "max":
            r = np.max(v, axis=k)
        else:
            r = np.min(v, axis=k)
        return r, ax[:k] + ax[k + 1:]

    def output(self, o: ir.Output) -> np.ndarray:
        v, ax = self.ev(o.expr)
        target = list(o.axes)
        (arr,), full = _align([(v, ax)])
        # place into output axis order, broadcasting missing axes
        order = [full.index(a) for a in target if a in full]
        arr = np.transpose(arr, order) if arr.ndim else arr
        shape = [arr.shape[order.index(full.index(a))] if a in full else 1 for a in target]
        arr = np.reshape(arr, shape)
        final = tuple(self.size(a.size) for a in target)
        return np.broadcast_to(arr, final).astype(np.float64)


CHUNK_ELEMENTS = 1 << 23  # rough cap on the size of one intermediate array, in elements


def fix_axes(e: ir.Expr, values: Dict[Axis, int]) -> ir.Expr:
    """e with the given axes fixed to constant indices."""
    memo: Dict[ir.Expr, ir.Expr] = {}

    def go(n):
        if n in memo:
            return memo[n]
        if isinstance(n, Index) and n.axis in values:
            r = Const(float(values[n.axis]))
        elif isinstance(n, Load):
            r = Load(n.buffer, tuple(values.get(a, a) if isinstance(a, Axis)
                                     else a.rename(lambda x: values.get(x, x))
                                     if isinstance(a, ir.Strided) else a
                                     for a in n.index), n.pad)
        elif isinstance(n, Unary):
            r = Unary(n.op, go(n.x))
        elif isinstance(n, Binary):
            r = Binary(n.op, go(n.a), go(n.b))
        elif isinstance(n, Where):
            r = Where(go(n.cond), go(n.a), go(n.b))
        elif isinstance(n, Reduce):
            r = Reduce(n.op, n.axis, go(n.body))
        else:
            r = n
        memo[n] = r
        return r

    return go(e)


def _peak_elements(o: ir.Output, env, fixed: int) -> int:
    """Largest intermediate array if the first `fixed` output axes are fixed (rough upper bound)."""
    rest = [a for a in o.axes[fixed:]]
    total = 1
    for a in rest:
        total *= size_of(a.size, env)
    worst = 1
    for n in ir.walk(o.expr):
        if isinstance(n, Reduce):
            # a sum of a product goes through einsum, which never builds the full product
            if n.op == "sum" and isinstance(n.body, Binary) and n.body.op == "mul":
                parts = [n.body.a, n.body.b]
            else:
                parts = [n.body]
            for part in parts:
                sz = 1
                for a in ir.free_axes(part):
                    if a not in o.axes[:fixed]:
                        sz *= size_of(a.size, env)
                worst = max(worst, sz)
    return max(total, worst)


class RoundedEvaluator(Evaluator):
    """The float64 evaluation with every arithmetic node's result rounded to float32: what a correct
    kernel computing in float32 and rounding once per op gives (the rounded run of docs/TOLERANCE.md).
    Where a formula cancels (1 - exp(x) for a tiny x, a difference of two close sums) every correct
    float32 kernel is far from float64 and close to this; a wrong kernel is far from both."""

    def _ev(self, e) -> Named:
        v, ax = super()._ev(e)
        if isinstance(e, (Unary, Binary, Reduce)) and getattr(v, "dtype", None) == np.float64:
            v = np.asarray(v, dtype=np.float32).astype(np.float64)
        return v, ax


U32 = 2.0 ** -24                                 # float32 unit roundoff
CORRECTLY_ROUNDED = ("add", "sub", "mul", "div")   # IEEE basic operations (and sqrt)
TRANSCENDENTAL = ("exp", "log", "tanh", "sigmoid", "erf", "cos", "sin", "expm1", "rsqrt")


class ErrorModelEvaluator(Evaluator):
    """The float64 evaluation with every arithmetic node's result given a random error of the size a
    correct float32 kernel can make there (docs/TOLERANCE.md, Method): one unit roundoff of the result
    for arithmetic and sqrt, four for transcendental functions (fast GPU math), and one for a
    reduction's result (the sums' own error is judged against their terms' magnitude). The error
    propagates through the rest of the expression, so a cancellation downstream amplifies it as it
    does in a real kernel. ``seed`` picks the draw."""

    def __init__(self, inputs, env, seed: int = 0):
        super().__init__(inputs, env)
        self.rng = np.random.default_rng(seed)

    def _ev(self, e) -> Named:
        v, ax = super()._ev(e)
        if isinstance(e, (Unary, Binary, Reduce)) and getattr(v, "dtype", None) == np.float64:
            c = 4.0 if isinstance(e, Unary) and e.op in TRANSCENDENTAL else 1.0
            noise = c * U32 * self.rng.uniform(-1.0, 1.0, np.shape(v))
            if (isinstance(e, Binary) and e.op in CORRECTLY_ROUNDED) or (isinstance(e, Unary) and e.op == "sqrt"):
                # correctly rounded: a result representable in float32 is returned exactly
                with np.errstate(all="ignore", over="ignore"):
                    noise = np.where(np.asarray(v, dtype=np.float32).astype(np.float64) == v, 0.0, noise)
            v = v * (1.0 + noise)
        return v, ax


def run_reference(k: Kernel, inputs: Dict[str, np.ndarray], evaluator=None) -> Dict[str, np.ndarray]:
    """Evaluate every output of k in float64. `inputs` maps buffer name to an array. ``evaluator``:
    the Evaluator class (RoundedEvaluator for the rounded run).

    Large outputs are evaluated in chunks over their leading axes so intermediate arrays (an
    attention score matrix, say) stay within CHUNK_ELEMENTS.
    """
    env = bind_sizes(k, {n: np.shape(v) for n, v in inputs.items()})
    res = {}
    with np.errstate(all="ignore"):
        for o in k.outputs:
            fixed = 0
            while fixed < len(o.axes) and _peak_elements(o, env, fixed) > CHUNK_ELEMENTS:
                fixed += 1
            ev_cls = evaluator or Evaluator
            if fixed == 0:
                res[o.name] = ev_cls(inputs, env).output(o)
                continue
            lead = o.axes[:fixed]
            out = np.empty(tuple(size_of(a.size, env) for a in o.axes), dtype=np.float64)
            ev_inputs = {n: np.asarray(v, dtype=np.float64) for n, v in inputs.items()}
            for vals in itertools.product(*[range(size_of(a.size, env)) for a in lead]):
                fx = dict(zip(lead, vals))
                sub = ir.Output(o.name, o.axes[fixed:], fix_axes(o.expr, fx), o.dtype)
                out[vals] = ev_cls(ev_inputs, env).output(sub)
            res[o.name] = out
    return res


# ---------------------------------------------------------------------------------------------
# The same interpreter on a CUDA device, in float64 (torch). Identical math: every node maps to
# the torch float64 op that computes the same IEEE function as the NumPy op above (einsum for sums
# of products, NaN propagating maximum and minimum, NaN propagating max and min reductions). The
# results differ from the NumPy run only by float64 rounding (summation order, libm ULPs), about
# 1e-15 relative, at least six orders of magnitude below every kernel tolerance (float64 kernels
# 1e-9). An op torch cannot run on the device in float64 is evaluated on the CPU with the NumPy
# function for that node only, and its name is added to `cpu_fallbacks`.

GPU_CHUNK_ELEMENTS = 1 << 25  # 256 MB of float64 per intermediate array on the device

# Below this much work the NumPy reference takes a few milliseconds, less than the GPU's own fixed
# costs (its first float64 use in a process costs 0.1 to 0.3 s), so small kernels keep it.
GPU_MIN_WORK = 1 << 20


def work_estimate(k: Kernel, env: Dict[str, int]) -> int:
    """Rough element count of the reference's biggest step: the largest output, or the largest
    iteration space of a reduction (its body's free axes and its own axis)."""
    w = 1
    for o in k.outputs:
        n = 1
        for a in o.axes:
            n *= size_of(a.size, env)
        w = max(w, n)
        for node in ir.walk(o.expr):
            if isinstance(node, Reduce):
                axes = set(ir.free_axes(node.body)) | {node.axis}
                n = 1
                for a in axes:
                    n *= size_of(a.size, env)
                w = max(w, n)
    return w

cpu_fallbacks: set = set()  # ops that had to leave the device, by name


def _t_unary():
    import torch
    return {
        "neg": torch.neg, "exp": torch.exp, "log": torch.log, "sqrt": torch.sqrt,
        "rsqrt": lambda x: 1.0 / torch.sqrt(x), "sigmoid": lambda x: 1.0 / (1.0 + torch.exp(-x)),
        "tanh": torch.tanh, "abs": torch.abs, "not": torch.logical_not, "erf": torch.erf,
        "cos": torch.cos, "sin": torch.sin, "expm1": torch.expm1,
    }


def _t_binary():
    import torch
    return {
        "add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div,
        "maximum": torch.maximum, "minimum": torch.minimum, "lt": torch.lt, "le": torch.le,
        "gt": torch.gt, "ge": torch.ge, "eq": torch.eq, "ne": torch.ne,
        "and": torch.logical_and, "or": torch.logical_or,
    }


def _f64(torch, x):
    return x if x.dtype == torch.float64 else x.to(torch.float64)


class TorchEvaluator(Evaluator):
    """Evaluator whose arrays are float64 (or bool) torch tensors on one device."""

    def __init__(self, inputs, env: Dict[str, int], device, window=None):
        import torch
        self.torch = torch
        self.device = device
        self.window = window or {}  # output axis -> (start, stop): evaluate only that range
        self.inputs = inputs  # float64 tensors on `device` (run_reference_torch converts them)
        self.env = env
        self.memo = {}
        self.un = _t_unary()
        self.bin = _t_binary()

    def _scalar(self, v, dtype):
        return self.torch.tensor(v, dtype=dtype, device=self.device)

    def _align(self, vals):
        axes: List[Axis] = []
        for _, ax in vals:
            for a in ax:
                if a not in axes:
                    axes.append(a)
        out = []
        for arr, ax in vals:
            if ax == tuple(axes):
                out.append(arr)
                continue
            order = [ax.index(a) for a in axes if a in ax]
            t = arr.permute(order) if arr.dim() else arr
            shape = [arr.shape[ax.index(a)] if a in ax else 1 for a in axes]
            out.append(t.reshape(shape))
        return out, tuple(axes)

    def _op(self, table, numpy_table, op, *args):
        torch = self.torch
        # NumPy promotes bool operands of arithmetic to float64; do the same before the op
        if op not in ("not", "and", "or") and any(a.dtype == torch.bool for a in args) and \
                not all(a.dtype == torch.bool for a in args):
            args = tuple(_f64(torch, a) for a in args)
        if op in ("div", "exp", "log", "sqrt", "rsqrt", "sigmoid", "tanh", "erf") and                 any(a.dtype == torch.bool for a in args):
            args = tuple(_f64(torch, a) for a in args)
        try:
            return table[op](*args)
        except (RuntimeError, NotImplementedError):  # not available on the device in float64
            cpu_fallbacks.add(op)
            r = numpy_table[op](*[a.cpu().numpy() for a in args])
            return torch.as_tensor(np.asarray(r)).to(self.device)

    def _ev(self, e):
        torch = self.torch
        if isinstance(e, Const):
            return self._scalar(e.value, torch.bool if isinstance(e.value, bool)
                                else torch.float64), ()
        if isinstance(e, Dim):
            return self._scalar(float(self.size(e.size)), torch.float64), ()
        if isinstance(e, Index):
            lo, hi = self.window.get(e.axis, (0, self.size(e.axis.size)))
            return torch.arange(lo, hi, dtype=torch.float64, device=self.device), (e.axis,)
        if isinstance(e, Load):
            arr = self.inputs[e.buffer.name]
            if e.is_strided:
                return load_strided(e, arr, lambda a: self.window.get(a, (0, self.size(a.size))),
                                    _torch_ns(torch, self.device))
            sl = tuple(a if isinstance(a, int) else
                       slice(*self.window[a]) if a in self.window else slice(None)
                       for a in e.index)
            return arr[sl], tuple(a for a in e.index if isinstance(a, Axis))
        if isinstance(e, Unary):
            x, ax = self.ev(e.x)
            return self._op(self.un, _UNARY, e.op, x), ax
        if isinstance(e, Binary):
            (a, b), ax = self._align([self.ev(e.a), self.ev(e.b)])
            return self._op(self.bin, _BINARY, e.op, a, b), ax
        if isinstance(e, Where):
            (c, a, b), ax = self._align([self.ev(e.cond), self.ev(e.a), self.ev(e.b)])
            if c.dtype != torch.bool:  # np.where takes any nonzero value as true
                c = c != 0
            if a.dtype != b.dtype:  # np.where promotes bool with float64 to float64
                a, b = _f64(torch, a), _f64(torch, b)
            return torch.where(c, a, b), ax
        if isinstance(e, Reduce):
            return self._reduce(e if e.op != "osum" else Reduce("sum", e.axis, e.body))  # same value in float64
        raise ir.IRError(f"reference: cannot evaluate {type(e).__name__}")

    def _reduce(self, e: Reduce):
        torch = self.torch
        n = self.size(e.axis.size)
        body = e.body
        if e.op == "sum":
            r = self._chain(e, einsum64, lambda x: _f64(torch, x))
            if r is not None:
                return r
        if e.op == "sum" and isinstance(body, Binary) and body.op == "mul":
            a, aa = self.ev(body.a)
            b, ba = self.ev(body.b)
            if e.axis in aa or e.axis in ba:
                letters = {}
                for x in aa + ba:
                    if x not in letters:
                        letters[x] = string.ascii_letters[len(letters)]
                out_axes = tuple(x for x in letters if x != e.axis)
                sa = "".join(letters[x] for x in aa)
                sb = "".join(letters[x] for x in ba)
                so = "".join(letters[x] for x in out_axes)
                r = einsum64(f"{sa},{sb}->{so}", _f64(torch, a), _f64(torch, b))
                return r, out_axes
        v, ax = self.ev(body)
        if e.axis not in ax:  # body does not vary along the axis
            if e.op == "sum":
                return _f64(torch, v) * n, ax
            return v, ax
        k = ax.index(e.axis)
        if e.op == "sum":
            r = torch.sum(_f64(torch, v), dim=k)
        elif e.op == "max":
            r = torch.amax(v, dim=k)
        else:
            r = torch.amin(v, dim=k)
        return r, ax[:k] + ax[k + 1:]

    def output(self, o: ir.Output):
        torch = self.torch
        v, ax = self.ev(o.expr)
        target = list(o.axes)
        (arr,), full = self._align([(v, ax)])
        order = [full.index(a) for a in target if a in full]
        arr = arr.permute(order) if arr.dim() else arr
        shape = [arr.shape[order.index(full.index(a))] if a in full else 1 for a in target]
        arr = arr.reshape(shape)
        final = tuple(self.window[a][1] - self.window[a][0] if a in self.window
                      else self.size(a.size) for a in target)
        return torch.broadcast_to(_f64(torch, arr), final)


SLICED_MIN_WORK = 1 << 30      # multiply adds below which the native float64 einsum is kept
sliced_stats = {"used": 0, "native": 0}


def einsum64(spec, a, b):
    """A float64 einsum of two operands for the device reference. With TRUESTAR_VERIFY_F64=sliced
    (opt in) a large contraction runs as the sliced float64 matmul on int8 tensor cores
    (kernel/f64emu.py, docs/SLICED_F64.md), whose every entry is proven at least as accurate as
    native float64; anything it cannot take keeps torch.einsum."""
    import os
    import torch

    if (os.environ.get("TRUESTAR_VERIFY_F64", "native").strip().lower() == "sliced"
            and a.is_cuda and b.is_cuda):
        letters = {}
        ins = spec.split("->")[0].split(",")
        for t, sub in zip((a, b), ins):
            for c, n in zip(sub, t.shape):
                letters[c] = n
        work = 1
        for n in letters.values():
            work *= n
        if work >= SLICED_MIN_WORK:
            from . import f64emu
            try:
                r = f64emu.sliced_einsum(spec, a, b)
            except f64emu.SlicedF64Error:
                r = None
            if r is not None:
                sliced_stats["used"] += 1
                return r
        sliced_stats["native"] += 1
    return torch.einsum(spec, a, b)


def run_reference_torch(k: Kernel, inputs: Dict, device) -> Dict:
    """run_reference on a torch device (normally CUDA): the same float64 math. Large outputs are
    evaluated in blocks of their leading axes, each block within GPU_CHUNK_ELEMENTS (the last
    chunked axis takes as many indices per block as fit, not one). Returns float64 torch tensors
    on that device."""
    import torch
    env = bind_sizes(k, {n: tuple(np.shape(v)) for n, v in inputs.items()})
    ev_inputs = {n: (v.detach() if isinstance(v, torch.Tensor)
                     else torch.as_tensor(np.asarray(v, dtype=np.float64)))
                 .to(device=device, dtype=torch.float64) for n, v in inputs.items()}
    res = {}
    for o in k.outputs:
        fixed = 0
        while fixed < len(o.axes) and _peak_elements(o, env, fixed) > GPU_CHUNK_ELEMENTS:
            fixed += 1
        if fixed == 0:
            res[o.name] = TorchEvaluator(ev_inputs, env, device).output(o)
            continue
        lead = o.axes[:fixed]
        out = torch.empty(tuple(size_of(a.size, env) for a in o.axes), dtype=torch.float64,
                          device=device)
        last = size_of(lead[-1].size, env)
        block = max(1, min(last, GPU_CHUNK_ELEMENTS // max(1, _peak_elements(o, env, fixed))))
        for vals in itertools.product(*[range(size_of(a.size, env)) for a in lead[:-1]]):
            for lo in range(0, last, block):
                hi = min(last, lo + block)
                win = {a: (v, v + 1) for a, v in zip(lead[:-1], vals)}
                win[lead[-1]] = (lo, hi)
                sl = tuple(slice(v, v + 1) for v in vals) + (slice(lo, hi),)
                out[sl] = TorchEvaluator(ev_inputs, env, device, win).output(o)
        res[o.name] = out
    return res
