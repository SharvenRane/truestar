"""A small graph of tensor ops and the template free fusion pass that turns it into kernels.

Each op is defined only by how one output element is computed from its inputs (an index
function). Fusion inlines producers into consumers, which yields one IR expression per kernel. The
backend scheduler then accepts the fused kernel or explains why not, and the pass materializes the
nearest reduction producer and tries again. Nothing is keyed on op names: an attention variant is
fused because its IR has the shape of an online softmax, whatever ops wrote it.

    g = Graph()
    q, k, v = g.input("q", (B, H, N, D), "float16"), ...
    s = (q @ k.transpose(-1, -2)) * 0.125
    s = g.pointwise(lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF), s, with_index=True)
    o = g.softmax(s) @ v
    fn = g.compile({"o": o}, backend="cuda")     # one fused kernel
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import ir
from .ir import Axis, Buffer, Const, Expr, IRError, Load, Reduce, Size
from .reference import ShapeError

HEAVY = ("reduce", "softmax", "layer_norm", "matmul")


def _same(a: Size, b: Size) -> bool:
    return a == b


def _broadcast(*shapes: Sequence[Size]) -> Tuple[Size, ...]:
    n = max(len(s) for s in shapes)
    out = []
    for k in range(n):
        dims = [s[len(s) - n + k] for s in shapes if len(s) - n + k >= 0]
        non1 = [d for d in dims if d != 1]
        if any(not _same(d, non1[0]) for d in non1):
            raise ShapeError(f"shapes {', '.join(str(list(s)) for s in shapes)} do not broadcast "
                             f"(dim {k - n})")
        out.append(non1[0] if non1 else 1)
    return tuple(out)


class Tensor:
    def __init__(self, graph: "Graph", op: str, args: Tuple["Tensor", ...], shape, dtype: str,
                 attrs: Optional[Dict] = None, name: Optional[str] = None):
        self.graph = graph
        self.op = op
        self.args = args
        self.shape = tuple(shape)
        self.dtype = dtype
        self.attrs = attrs or {}
        self.name = name
        self.id = len(graph.nodes)
        graph.nodes.append(self)

    @property
    def ndim(self):
        return len(self.shape)

    def __repr__(self):
        return f"Tensor({self.op}#{self.id}, {list(self.shape)}, {self.dtype})"

    def _bin(self, other, fn, name):
        return self.graph._ew(fn, name, self, other)

    def __add__(self, o): return self._bin(o, lambda a, b: a + b, "add")
    def __radd__(self, o): return self.graph._ew(lambda a, b: a + b, "add", o, self)
    def __sub__(self, o): return self._bin(o, lambda a, b: a - b, "sub")
    def __rsub__(self, o): return self.graph._ew(lambda a, b: a - b, "sub", o, self)
    def __mul__(self, o): return self._bin(o, lambda a, b: a * b, "mul")
    def __rmul__(self, o): return self.graph._ew(lambda a, b: a * b, "mul", o, self)
    def __truediv__(self, o): return self._bin(o, lambda a, b: a / b, "div")
    def __rtruediv__(self, o): return self.graph._ew(lambda a, b: a / b, "div", o, self)
    def __neg__(self): return self.graph._ew(lambda a: -a, "neg", self)
    def __matmul__(self, o): return self.graph.matmul(self, o)

    def transpose(self, d0: int, d1: int) -> "Tensor":
        return self.graph.transpose(self, d0, d1)

    @property
    def T(self):
        return self.transpose(-1, -2)


class Graph:
    def __init__(self):
        self.nodes: List[Tensor] = []

    # ------------------------------------------------------------------ building
    def input(self, name: str, shape: Sequence[Size], dtype: str = "float32") -> Tensor:
        if dtype not in ir.DTYPES:
            raise IRError(f"unsupported dtype {dtype!r}")
        if any(t.op == "input" and t.name == name for t in self.nodes):
            raise IRError(f"duplicate input name {name!r}")
        return Tensor(self, "input", (), tuple(shape), dtype, name=name)

    def _wrap(self, x) -> Tensor:
        if isinstance(x, Tensor):
            if x.graph is not self:
                raise IRError("tensor belongs to a different graph")
            return x
        if isinstance(x, (int, float)):
            return Tensor(self, "const", (), (), "float32", {"value": float(x)})
        raise IRError(f"cannot use {x!r} in a graph op")

    def _ew(self, fn: Callable, name: str, *xs, with_index: bool = False) -> Tensor:
        ts = [self._wrap(x) for x in xs]
        shape = _broadcast(*[t.shape for t in ts])
        real = [t for t in ts if t.op != "const"]
        dtype = real[0].dtype if real else "float32"
        return Tensor(self, "ew", tuple(ts), shape, dtype,
                      {"fn": fn, "name": name, "with_index": with_index})

    def pointwise(self, fn: Callable, *xs, with_index: bool = False, name: str = "map") -> Tensor:
        """Elementwise op written as a plain Python function of IR expressions.

        With with_index=True, fn also receives the index of every output dim (as expressions)
        after the values, so masks and position terms can be written directly:
        `lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF)`.
        """
        return self._ew(fn, name, *xs, with_index=with_index)

    def exp(self, x): return self._ew(ir.exp, "exp", x)
    def log(self, x): return self._ew(ir.log, "log", x)
    def sqrt(self, x): return self._ew(ir.sqrt, "sqrt", x)
    def rsqrt(self, x): return self._ew(ir.rsqrt, "rsqrt", x)
    def sigmoid(self, x): return self._ew(ir.sigmoid, "sigmoid", x)
    def tanh(self, x): return self._ew(ir.tanh, "tanh", x)
    def relu(self, x): return self._ew(ir.relu, "relu", x)
    def abs(self, x): return self._ew(ir.abs_, "abs", x)
    def maximum(self, a, b): return self._ew(ir.maximum, "maximum", a, b)
    def minimum(self, a, b): return self._ew(ir.minimum, "minimum", a, b)
    def scale(self, x, c: float): return self._ew(lambda a: a * float(c), "scale", x)
    def where(self, c, a, b): return self._ew(ir.where, "where", c, a, b)

    def mask(self, x, keep, fill: float = ir.NEG_INF):
        """x where keep is true, else fill (keep is a boolean tensor, or computed from indices)."""
        return self._ew(lambda v, k: ir.where(k, v, float(fill)), "mask", x, keep)

    def _axis(self, x: Tensor, axis: int) -> int:
        a = axis + x.ndim if axis < 0 else axis
        if not 0 <= a < x.ndim:
            raise ShapeError(f"axis {axis} out of range for shape {list(x.shape)}")
        return a

    def sum(self, x, axis: int = -1, keepdims: bool = False) -> Tensor:
        return self._reduce("sum", x, axis, keepdims)

    def max(self, x, axis: int = -1, keepdims: bool = False) -> Tensor:
        return self._reduce("max", x, axis, keepdims)

    def min(self, x, axis: int = -1, keepdims: bool = False) -> Tensor:
        return self._reduce("min", x, axis, keepdims)

    def mean(self, x, axis: int = -1, keepdims: bool = False) -> Tensor:
        x = self._wrap(x)
        a = self._axis(x, axis)
        n = x.shape[a]
        s = self.sum(x, a, keepdims)
        return self._ew(lambda v: v / ir.dim(n), "mean", s)

    def _reduce(self, op, x, axis, keepdims):
        x = self._wrap(x)
        a = self._axis(x, axis)
        shape = list(x.shape)
        if keepdims:
            shape[a] = 1
        else:
            del shape[a]
        return Tensor(self, "reduce", (x,), shape, x.dtype, {"op": op, "axis": a, "keep": keepdims})

    def softmax(self, x, axis: int = -1) -> Tensor:
        x = self._wrap(x)
        return Tensor(self, "softmax", (x,), x.shape, x.dtype, {"axis": self._axis(x, axis)})

    def layer_norm(self, x, weight=None, bias=None, eps: float = 1e-5) -> Tensor:
        x = self._wrap(x)
        n = x.shape[-1]
        args = [x]
        for p, nm in ((weight, "weight"), (bias, "bias")):
            if p is not None:
                p = self._wrap(p)
                if p.shape != (n,):
                    raise ShapeError(f"layer_norm {nm} must have shape [{n}], got {list(p.shape)}")
                args.append(p)
        return Tensor(self, "layer_norm", tuple(args), x.shape, x.dtype,
                      {"eps": float(eps), "w": weight is not None, "b": bias is not None})

    def matmul(self, a, b) -> Tensor:
        a, b = self._wrap(a), self._wrap(b)
        if a.ndim < 2 or b.ndim < 2:
            raise ShapeError(f"matmul needs at least 2 dims, got {list(a.shape)} @ {list(b.shape)}")
        if not _same(a.shape[-1], b.shape[-2]):
            raise ShapeError(f"matmul inner dims differ: {list(a.shape)} @ {list(b.shape)} "
                             f"({a.shape[-1]} vs {b.shape[-2]})")
        batch = _broadcast(a.shape[:-2], b.shape[:-2])
        shape = batch + (a.shape[-2], b.shape[-1])
        return Tensor(self, "matmul", (a, b), shape, a.dtype)

    def transpose(self, x, d0: int, d1: int) -> Tensor:
        x = self._wrap(x)
        d0, d1 = self._axis(x, d0), self._axis(x, d1)
        perm = list(range(x.ndim))
        perm[d0], perm[d1] = perm[d1], perm[d0]
        return Tensor(self, "transpose", (x,), [x.shape[p] for p in perm], x.dtype, {"perm": perm})

    # ------------------------------------------------------------------ lowering
    def _buffer(self, t: Tensor) -> Buffer:
        return Buffer(t.name if t.op == "input" else f"t{t.id}", t.shape, t.dtype)

    def lower(self, t: Tensor, idx: Tuple, mat: set, root: Tensor) -> Expr:
        """IR expression for element idx of t, inlining every producer not in mat."""
        if t.op == "input" or (t.id in mat and t is not root):
            return Load(self._buffer(t), tuple(idx))
        if t.op == "const":
            return Const(t.attrs["value"])
        lo = lambda x, ix: self.lower(x, tuple(ix), mat, root)  # noqa: E731

        if t.op == "ew":
            vals = []
            for a in t.args:
                off = len(t.shape) - len(a.shape)
                ix = [idx[off + k] if not (a.shape[k] == 1 and t.shape[off + k] != 1) else 0
                      for k in range(len(a.shape))]
                vals.append(lo(a, ix))
            if t.attrs["with_index"]:
                vals += [ir.Index(x) if isinstance(x, Axis) else Const(float(x)) for x in idx]
            return ir.as_expr(t.attrs["fn"](*vals))
        if t.op == "transpose":
            perm = t.attrs["perm"]
            ix = [None] * len(perm)
            for out_d, in_d in enumerate(perm):
                ix[in_d] = idx[out_d]
            return lo(t.args[0], ix)
        if t.op == "reduce":
            x = t.args[0]
            a = t.attrs["axis"]
            r = Axis(f"r{t.id}", x.shape[a])
            ix = list(idx)
            if t.attrs["keep"]:
                ix[a] = r
            else:
                ix.insert(a, r)
            return Reduce(t.attrs["op"], r, lo(x, ix))
        if t.op == "softmax":
            x = t.args[0]
            a = t.attrs["axis"]

            def at(ax):
                ix = list(idx)
                ix[a] = ax
                return lo(x, ix)

            r1 = Axis(f"r{t.id}m", x.shape[a])
            r2 = Axis(f"r{t.id}s", x.shape[a])
            m = Reduce("max", r1, at(r1))
            l = Reduce("sum", r2, ir.exp(at(r2) - m))
            return ir.exp(lo(x, idx) - m) / l
        if t.op == "layer_norm":
            x = t.args[0]
            n = x.shape[-1]

            def at(ax):
                return lo(x, list(idx[:-1]) + [ax])

            r1 = Axis(f"r{t.id}a", n)
            r2 = Axis(f"r{t.id}b", n)
            mean = Reduce("sum", r1, at(r1)) / ir.dim(n)
            d = at(r2) - mean
            var = Reduce("sum", r2, d * d) / ir.dim(n)
            y = (lo(x, idx) - mean) * ir.rsqrt(var + t.attrs["eps"])
            k = 1
            if t.attrs["w"]:
                y = y * lo(t.args[k], [idx[-1]])
                k += 1
            if t.attrs["b"]:
                y = y + lo(t.args[k], [idx[-1]])
            return y
        if t.op == "matmul":
            a, b = t.args
            r = Axis(f"r{t.id}", a.shape[-1])
            nb = len(t.shape) - 2

            def bidx(x):
                off = nb - (x.ndim - 2)
                return [idx[off + k] if not (x.shape[k] == 1 and t.shape[off + k] != 1) else 0
                        for k in range(x.ndim - 2)]

            ia = bidx(a) + [idx[-2], r]
            ib = bidx(b) + [r, idx[-1]]
            return Reduce("sum", r, lo(a, ia) * lo(b, ib))
        raise IRError(f"unknown graph op {t.op}")

    def kernel_for(self, root: Tensor, mat: set, name: Optional[str] = None) -> ir.Kernel:
        bname = self._buffer(root).name if root.op != "input" else root.name
        axes = tuple(Axis(f"{bname}_{k}", s) for k, s in enumerate(root.shape))
        e = self.lower(root, axes, mat, root)
        out = ir.Output(name or bname, axes, e, root.dtype)
        kname = f"{root.op}_{root.id}" if root.op != "ew" else f"{root.attrs['name']}_{root.id}"
        return ir.Kernel(kname, [out])

    def _inlined_producers(self, root: Tensor, mat: set) -> List[Tensor]:
        """Producers inlined into root's kernel, nearest first (breadth first)."""
        out, seen, frontier = [], {root.id}, [root]
        while frontier:
            nxt = []
            for t in frontier:
                for a in t.args:
                    if a.id in seen or a.op in ("input", "const") or a.id in mat:
                        continue
                    seen.add(a.id)
                    out.append(a)
                    nxt.append(a)
            frontier = nxt
        return out

    # ------------------------------------------------------------------ fusion + compile
    def fuse(self, outputs: Dict[str, Tensor], backend: str = "cuda"):
        """Split the graph into kernels. Returns [(root tensor, kernel)] in execution order."""
        from .compiler import _BACKENDS
        be = _BACKENDS[backend]
        for nm, t in outputs.items():
            if t.op in ("input", "const"):
                raise IRError(f"output {nm} is an input or a constant; nothing to compute")
        mat = {t.id for t in outputs.values()}
        names = {t.id: nm for nm, t in outputs.items()}
        while True:
            changed = False
            for tid in sorted(mat):
                root = self.nodes[tid]
                k = self.kernel_for(root, mat, names.get(tid))
                if be not in ("triton", "metal"):   # Metal fuses with the GPU plan's shapes
                    continue
                from .triton_backend import Unschedulable, plan
                try:
                    plan(ir.canonicalize(k)[0])
                except Unschedulable as e:
                    prods = self._inlined_producers(root, mat)
                    heavy = [p for p in prods if p.op in HEAVY]
                    pick = (heavy or prods or [None])[0]
                    if pick is None:
                        raise Unschedulable(f"kernel {k.name} cannot run on the GPU and cannot be "
                                            f"split further: {e}") from None
                    mat.add(pick.id)
                    changed = True
                    break
            if not changed:
                break
        return [(self.nodes[tid], self.kernel_for(self.nodes[tid], mat, names.get(tid)))
                for tid in sorted(mat)]

    def compile(self, outputs: Union[Dict[str, Tensor], Tensor], backend: str = "cuda",
                allow_tf32: bool = False, tune=None, math: str = "default") -> "CompiledGraph":
        """Compile every fused kernel; math="round_once" as in K.compile (docs/ROUND_ONCE.md)."""
        if isinstance(outputs, Tensor):
            outputs = {"out": outputs}
        from .compiler import compile as kcompile
        parts = self.fuse(outputs, backend)
        compiled = [(root, kcompile(k, backend, allow_tf32=allow_tf32, tune=tune, math=math))
                    for root, k in parts]
        return CompiledGraph(self, outputs, compiled)

    def reference(self, outputs: Dict[str, Tensor], inputs: Dict) -> Dict:
        """Float64 result of the whole graph with nothing fused away (one expression per output)."""
        from .compiler import _to_f64
        from .reference import run_reference
        res = {}
        for nm, t in outputs.items():
            k = self.kernel_for(t, {t.id}, nm)
            res[nm] = run_reference(k, {b.name: _to_f64(inputs[b.name]) for b in k.inputs})[nm]
        return res


class CompiledGraph:
    def __init__(self, graph: Graph, outputs: Dict[str, Tensor], kernels):
        self.graph = graph
        self.outputs = outputs
        self.kernels = kernels  # [(root tensor, CompiledKernel)]

    @property
    def num_kernels(self) -> int:
        return len(self.kernels)

    def describe(self) -> str:
        return "\n".join(f"{ck.name}: {ck.plan_kind}" for _, ck in self.kernels)

    def __call__(self, **inputs):
        bufs = dict(inputs)
        for root, ck in self.kernels:
            args = {b.name: bufs[b.name] for b in ck.kernel.inputs}
            res = ck(**args)
            oname = ck.kernel.outputs[0].name
            bufs[oname] = res
            bufs[f"t{root.id}"] = res
        return {nm: bufs[nm] for nm in self.outputs}
