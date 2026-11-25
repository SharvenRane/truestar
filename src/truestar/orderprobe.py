"""Order probe: recover the summation order a backend uses for a reduction (docs/ORDERPROBE.md).

No library documents in which order it adds the terms of a sum, a matmul's K dimension or a
softmax denominator, yet that order is why two correct backends, or one library on two GPUs, give
float results that differ in their last bits. This module recovers the order from outputs alone,
with crafted inputs whose float32 results depend on it, and says it in plain words.

    truestar orderprobe --op linear --n 512 --rows 128 --cols 128 --backends torch:cuda,truestar:cuda

How the order is read
---------------------
Every order we consider is a tree: each internal node adds its children's values and rounds once
to the accumulator's precision (p significand bits, 24 for float32). Additions commute in IEEE
arithmetic, so the unordered tree is all that decides the result.

* Sums and matmuls (signed terms). Put +M at term a, -M at term b and a unit u at every other term
  of a set S (zeros elsewhere), with M = u * 2^(p + 1 + ceil(log2 n)). A unit that meets +M or -M
  before the two cancel is absorbed (it is below half an ulp of M, and so is any partial sum of
  units); units that only meet the result after the cancellation survive. The result is u times
  the number of terms of S outside the subtree where a and b first meet, so one call per pair
  (a, b) gives the size of that subtree. For a fixed pivot a, the other terms fall into levels by
  that size: the terms that join a's partial sum first, then the next ones, and so on up to the
  root. Each level is then probed on its own (its terms alone nonzero) the same way. Matmuls use a
  weight of ones, so every product is exact and the K reduction is the sum of the probe row.
* Softmax denominators (positive terms only: exp(x) > 0). Put logit 0 at term a (exp gives 1
  exactly after the max is subtracted), logit log(d) at terms b and c, minus infinity elsewhere,
  with d = 0.75 * 2^-p. One d meeting 1 is absorbed; two d added together first give 1.5 * 2^-p,
  which survives. So output[a] < 1 exactly when b and c meet before either meets a: a triplet test.
  Levels along a's path come from grouping terms whose pairs meet first, ordered by the same test
  with a level's term as the big one.

The probe runs every batch twice; a different answer means the order is not fixed (TS963).

Verification (the reason to trust the words): the recovered tree is replayed in exact arithmetic
with one round to nearest per node on random inputs, and must reproduce the backend's float32
results bit for bit (TS962); otherwise TS964 says no recovered order explains them (a wider
accumulator, fused multiply adds with other rounding, or a method that is not a tree). For a
matmul the replay also tries random weights under two product models (each product rounded to
float32 first, or fused multiply adds that round once per added product) and reports which one
matches. ``explain`` replays two backends' trees on the same input and shows that the difference
between their outputs is exactly the difference the trees predict.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

Tree = Union[int, tuple]
OPS = ("sum", "linear", "matmul", "softmax")


class OrderProbeError(RuntimeError):
    """The probe cannot run (unknown op or backend, backend unavailable)."""


# ----------------------------------------------------------------------------- trees


def leaves(t: Tree) -> List[int]:
    if isinstance(t, int):
        return [t]
    out: List[int] = []
    for c in t:
        out += leaves(c)
    return out


def canonical(t: Tree) -> Tree:
    """Children sorted by their smallest term: two trees are the same order iff canonical equal."""
    if isinstance(t, int):
        return t
    kids = [canonical(c) for c in t]
    kids.sort(key=lambda c: min(leaves(c)))
    return tuple(kids)


def depth(t: Tree) -> int:
    return 0 if isinstance(t, int) else 1 + max([depth(c) for c in t])


def to_text(t: Tree, limit: int = 64) -> str:
    """The tree as a nested sum, for small trees: ((0+1)+(2+3))."""
    if isinstance(t, int):
        return str(t)
    if len(leaves(t)) > limit:
        return f"<tree over {len(leaves(t))} terms>"
    return "(" + "+".join(to_text(c, limit) for c in t) + ")"


# ----------------------------------------------------------------------------- plain words


def _contig(ix: Sequence[int]) -> bool:
    s = sorted(ix)
    return s == list(range(s[0], s[0] + len(s)))


def _left_chain(t: Tree) -> Optional[List[Tree]]:
    """The items of a chain (((i0 + i1) + i2) + i3) ... in the order they join, when every node
    adds exactly two values and one of them is the chain so far; None otherwise. Items are terms,
    or pieces summed on their own first."""
    top: List[Tree] = []
    cur = t
    while True:
        if isinstance(cur, int) or len(cur) != 2:
            return None
        a, b = cur
        if isinstance(a, int) and isinstance(b, int):
            x, y = sorted((a, b))
            nxt = top[-1] if top else None
            bottom = [y, x] if isinstance(nxt, int) and nxt < x else [x, y]
            return bottom + top[::-1]
        if isinstance(a, int) != isinstance(b, int):
            chain, item = (b, a) if isinstance(a, int) else (a, b)
        else:
            chain, item = (a, b) if len(leaves(a)) >= len(leaves(b)) else (b, a)
        top.append(item)
        cur = chain


def _pairwise_block(t: Tree) -> Optional[int]:
    """The block size B when t is a balanced binary tree over contiguous halves down to blocks of
    B terms (B = 1: fully pairwise); None when the top split is not two equal contiguous halves."""
    if isinstance(t, int):
        return 1
    if len(t) != 2:
        return None
    a, b = t
    la, lb = leaves(a), leaves(b)
    if len(la) != len(lb) or not (_contig(la) and _contig(lb)):
        return None
    ba, bb = _pairwise_block(a), _pairwise_block(b)
    if ba is not None and ba == bb:
        return ba
    return len(la)


def _halving(t: Tree) -> bool:
    """A balanced tree over interleaved halves: every node adds the even ranked and the odd ranked
    terms of its own terms (strides n/2, n/4, ... 1 at the leaves: a shuffle style tree)."""
    if isinstance(t, int):
        return True
    if len(t) != 2:
        return False
    L = sorted(leaves(t))
    a, b = sorted(t, key=lambda c: min(leaves(c)))
    return (sorted(leaves(a)) == L[0::2] and sorted(leaves(b)) == L[1::2]
            and _halving(a) and _halving(b))


def _parts_at(t: Tree, s: int) -> List[Tree]:
    """Maximal subtrees with at most s terms."""
    if isinstance(t, int) or len(leaves(t)) <= s:
        return [t]
    out: List[Tree] = []
    for c in t:
        out += _parts_at(c, s)
    return out


def _quotient(t: Tree, parts: List[Tree]) -> Tree:
    """t with every part replaced by its index in ``parts`` (ordered by smallest term)."""
    key = {tuple(sorted(leaves(p))): i for i, p in enumerate(parts)}

    def go(n):
        lk = tuple(sorted(leaves(n)))
        if lk in key:
            return key[lk]
        return tuple([go(c) for c in n])
    return go(t)


def describe(t: Tree, _top: bool = True) -> str:
    """The order in plain words: sequential, pairwise, chunks, interleaved accumulators, fused."""
    n = len(leaves(t))
    if isinstance(t, int):
        return "a single term"
    if n == 2:
        return "one addition"
    if len(t) == n:
        return f"all {n} terms added with a single rounding (fused, or a wider accumulator)"
    chain = _left_chain(t)
    if chain is not None and all(isinstance(c, int) for c in chain):
        if chain == sorted(chain):
            return f"sequential, one term at a time in index order ({n} terms)"
        if chain == sorted(chain, reverse=True):
            return f"sequential, one term at a time from the last index down ({n} terms)"
        return f"sequential, one term at a time in the order {chain[:8]}{'...' if n > 8 else ''}"
    if n & (n - 1) == 0 and _halving(t):
        return (f"a balanced tree over interleaved halves: term i meets term i + {n // 2}, then + {n // 4}, "
                f"and so on ({n} terms, the shape of a shuffle reduction)")
    pb = _pairwise_block(t)
    if pb is not None and pb * 2 <= n and n % pb == 0 and (n // pb) & (n // pb - 1) == 0:
        if pb == 1:
            return f"pairwise: a balanced binary tree over contiguous halves ({n} terms)"
        blocks = _parts_at(t, pb)
        inner = {describe(b, False) for b in blocks}
        if len(inner) == 1:
            return f"pairwise over {n // pb} contiguous blocks of {pb}; inside each block: {inner.pop()}"
    # a uniform cut: parts of equal size that are contiguous chunks or interleaved residues
    sizes = sorted({len(leaves(x)) for x in _all_nodes(t)} - {1, n})
    for s in sizes:
        parts = _parts_at(t, s)
        if len(parts) < 2:
            continue
        ps = sorted(parts, key=lambda p: min(leaves(p)))
        lens = [len(leaves(p)) for p in ps]
        contig = all(_contig(leaves(p)) for p in ps)
        if contig and all(L == s for L in lens[:-1]) and lens[-1] <= s:
            k = len(ps)
            q = _quotient(t, ps)
            inner = {describe(p, False) for p in ps if len(leaves(p)) == s}
            tail = f" (the last one {lens[-1]})" if lens[-1] != s else ""
            ins = inner.pop() if len(inner) == 1 else "different orders per chunk"
            return (f"split into {k} contiguous chunks of {s}{tail}; inside each chunk: {ins}; the {k} chunk "
                    f"results combined: {describe(q, False) if not isinstance(q, int) else 'as is'}")
        if all(L == s for L in lens) and len(ps) >= 2:
            k = len(ps)
            res = [sorted(leaves(p)) for p in ps]
            if all(r == list(range(r[0], n, k)) for r in res) and sorted(r[0] for r in res) == list(range(k)):
                q = _quotient(t, ps)
                inner = {describe(_relabel(p), False) for p in ps}
                ins = inner.pop() if len(inner) == 1 else "different orders per accumulator"
                return (f"{k} interleaved accumulators (term i goes to accumulator i mod {k}); each: {ins}; "
                        f"the {k} accumulators combined: {describe(q, False)}")
    if chain is not None:
        sz = [len(leaves(c)) for c in chain]
        return (f"sequential over {len(chain)} pieces of sizes {sz[:8]}{'...' if len(sz) > 8 else ''}, "
                f"pieces summed first")
    fan = max(len(x) for x in _all_nodes(t) if not isinstance(x, int))
    return f"an irregular tree: depth {depth(t)}, up to {fan} values added per rounding"


def _relabel(p: Tree) -> Tree:
    """Terms of p renumbered 0.. by rank, so an interleaved accumulator reads like a plain sum."""
    rank = {v: i for i, v in enumerate(sorted(leaves(p)))}

    def go(n):
        return rank[n] if isinstance(n, int) else tuple([go(c) for c in n])
    return go(p)


def _all_nodes(t: Tree):
    yield t
    if not isinstance(t, int):
        for c in t:
            yield from _all_nodes(c)


# ----------------------------------------------------------------------------- exact replay


def _round_sum(vals: List[np.ndarray], acc: np.dtype) -> np.ndarray:
    """Round to nearest of the exact sum of float values (one rounding), vectorized over rows."""
    if acc == np.float64:
        if len(vals) == 2:
            return vals[0] + vals[1]
        return np.array([math.fsum(r) for r in zip(*vals)], dtype=np.float64)
    if len(vals) == 2:
        a, b = vals[0].astype(np.float64), vals[1].astype(np.float64)
        s = a + b
        bb = s - a
        e = (a - (s - bb)) + (b - bb)                  # s + e == a + b exactly (two sum)
        return _to32_exact(s, e)
    exact = [math.fsum(map(float, r)) for r in zip(*vals)]
    err = [math.fsum([float(x) for x in r] + [-v]) for r, v in zip(zip(*vals), exact)]
    return _to32_exact(np.array(exact), np.array(err))


def _to32_exact(s: np.ndarray, e: np.ndarray) -> np.ndarray:
    """float32 round to nearest of s + e (s float64, e its tiny exact remainder)."""
    r = s.astype(np.float32)
    up = np.nextafter(r, np.float32(np.inf))
    dn = np.nextafter(r, np.float32(-np.inf))
    for other in (up, dn):
        mid = (r.astype(np.float64) + other.astype(np.float64)) / 2
        hit = (s == mid) & (e != 0) & np.isfinite(mid)
        go = hit & (np.sign(e) == np.sign(other.astype(np.float64) - r.astype(np.float64)))
        r = np.where(go, other, r)
    return r


def replay(t: Tree, x: np.ndarray, acc=np.float32, products: Optional[np.ndarray] = None,
           fused: bool = False) -> np.ndarray:
    """The tree's result for every row of x [R, n] (float32 values), one rounding per node.

    ``products``: [R, n] float64 exact products a_k * b_k for a matmul replay; with ``fused`` a
    node that adds a single product to a partial sum rounds once (fused multiply add), otherwise
    every product is rounded to float32 first."""
    acc = np.dtype(acc)
    if products is not None and not fused:
        x = products.astype(np.float32)
        products = None
    if products is None:
        cols = [x[:, k].astype(acc) for k in range(x.shape[1])]

        def go(n):
            if isinstance(n, int):
                return cols[n]
            return _round_sum([go(c) for c in n], acc)
        return go(t).astype(np.float64)

    prod = products

    def gof(n):
        if isinstance(n, int):
            return prod[:, n].astype(np.float32), True       # (value, still an unrounded product?)
        kids = list(n)
        if len(kids) == 2:
            a, b = kids
            if isinstance(b, int) and not isinstance(a, int):
                a, b = b, a
            if isinstance(a, int) and not isinstance(b, int):  # partial sum + one product: one rounding
                pv, _ = gof(b)
                p = prod[:, a]
                s = pv.astype(np.float64) + p
                bb = s - pv.astype(np.float64)
                e = (pv.astype(np.float64) - (s - bb)) + (p - bb)
                return _to32_exact(s, e), False
            if isinstance(a, int) and isinstance(b, int):      # an accumulator's first two products:
                lo, hi = min(a, b), max(a, b)                   # fma(hi, round(lo))
                pv = prod[:, lo].astype(np.float32).astype(np.float64)
                p = prod[:, hi]
                s_ = pv + p
                bb = s_ - pv
                e = (pv - (s_ - bb)) + (p - bb)
                return _to32_exact(s_, e), False
        vals = [gof(c)[0] for c in kids]
        return _round_sum(vals, np.dtype(np.float32)), False
    return gof(t)[0].astype(np.float64)


# ----------------------------------------------------------------------------- backends


@dataclass
class Target:
    """What is probed: an op on a backend at a shape. ``run`` maps probe rows [rows, n] (float64
    values exactly representable in ``dtype``) to the reduced values [rows] (sum, linear, matmul:
    output column ``col``) or to the full softmax output [rows, n]."""

    op: str
    backend: str
    n: int
    rows: int
    cols: int = 64
    dtype: str = "float32"
    col: int = 0
    run: Optional[Callable[[np.ndarray], np.ndarray]] = None
    notes: List[str] = field(default_factory=list)
    row: int = 0                    # the row probed when rows do not share one order
    uniform: Optional[bool] = None  # every row sums in the same order (rows_uniform)


def rows_uniform(tg: Target, trials: int = 4, seed: int = 11) -> bool:
    """True when the same data in every row gives the same result in every row, on ``trials``
    random rows: then the rows share one order and probe rows can be batched. A kernel whose edge
    rows take another code path (CPU GEMM micro kernels) fails it, and each probe then runs alone
    in row ``tg.row``."""
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        r = _quantize(rng.normal(size=(1, tg.n)), tg.dtype)
        out = np.asarray(tg.run(np.repeat(r, tg.rows, axis=0)))
        flat = out.reshape(tg.rows, -1)
        if not np.all(flat == flat[:1]):
            return False
    return True


def _torch_dtype(name):
    import torch
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def make_target(op: str, backend: str, n: int, rows: int = 64, cols: int = 64, dtype: str = "float32",
                col: int = 0) -> Target:
    """A Target for a built in backend: ``torch:cpu``, ``torch:cuda``, ``torch:mps``,
    ``compile:cpu``, ``compile:cuda``, ``compile:mps`` (torch.compile), ``truestar:cpu``,
    ``truestar:cuda``, ``truestar:mps`` (a Truestar kernel; Metal on the Mac's GPU), ``ort:cpu``,
    ``ort:cuda`` (ONNX Runtime). ``linear`` is torch's nn.Linear call (x W^T + b with
    a zero bias, the cuBLAS path of a model), ``matmul`` is x @ W."""
    if op not in OPS:
        raise OrderProbeError(f"unknown op {op!r}; one of {', '.join(OPS)}")
    kind, _, dev = backend.partition(":")
    dev = dev or "cpu"
    tg = Target(op, backend, n, rows, cols, dtype, col)
    if kind in ("torch", "compile"):
        tg.run = _torch_runner(op, dev, n, rows, cols, dtype, col, compiled=kind == "compile")
    elif kind == "truestar":
        tg.run = _truestar_runner(op, dev, n, rows, cols, dtype, col)
    elif kind == "ort":
        tg.run = _ort_runner(op, dev, n, rows, cols, dtype, col)
    else:
        raise OrderProbeError(f"unknown backend {backend!r}; use torch:, compile:, truestar: or ort: with cpu, "
                              f"cuda or mps")
    return tg


def _need(dev):
    import torch
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise OrderProbeError("no CUDA device")
    if dev.startswith("mps") and not torch.backends.mps.is_available():
        raise OrderProbeError("no MPS device")


def _sync(dev):
    import torch
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    elif dev.startswith("mps"):
        torch.mps.synchronize()


def _to_numpy(out):
    # to the CPU first, then float64: the MPS device has no float64, and MPS to("cpu", float64) in
    # one call returns zeros on torch 2.14 (docs/METAL.md)
    return out.detach().cpu().double().numpy()


def _torch_runner(op, dev, n, rows, cols, dtype, col, compiled=False):
    import torch
    _need(dev)
    dt = _torch_dtype(dtype)
    w = torch.ones(cols, n, dtype=dt, device=dev)
    bias = torch.zeros(cols, dtype=dt, device=dev)

    def f(x):
        if op == "sum":
            return x.sum(dim=-1)
        if op == "linear":
            return torch.nn.functional.linear(x, w, bias)[:, col]
        if op == "matmul":
            return (x @ w.t())[:, col]
        return torch.softmax(x, dim=-1)
    g = f
    if compiled:
        g = torch.compile(f, dynamic=False)

    def run(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(x).to(dt).to(dev)
            out = g(t)
            _sync(dev)
        return _to_numpy(out)
    return run


def _truestar_runner(op, dev, n, rows, cols, dtype, col):
    import torch
    from .kernel import compile as kcompile, ir
    _need(dev)
    dt = _torch_dtype(dtype)
    i, r = ir.axis("i", rows), ir.axis("r", n)
    x = ir.buffer("x", (rows, n), dtype)
    if op == "sum":
        k = ir.kernel("probe_sum", [ir.output("y", [i], ir.reduce_sum(r, x[i, r]), "float32")])
    elif op in ("linear", "matmul"):
        w = ir.buffer("w", (n, cols), dtype)
        j = ir.axis("j", cols)
        k = ir.kernel("probe_mm", [ir.output("y", [i, j], ir.reduce_sum(r, x[i, r] * w[r, j]), "float32")])
    else:
        j = ir.axis("j", n)
        k = ir.kernel("probe_softmax", [ir.output("y", [i, j], ir.softmax_weight(x[i, j], j), "float32")])
    fn = kcompile(k, "cuda" if dev.startswith("cuda") else "metal" if dev.startswith("mps") else "cpu")
    wt = torch.ones(n, cols, dtype=dt, device=dev)

    def run(xn: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(xn).to(dt).to(dev)
        out = fn(x=t, w=wt) if op in ("linear", "matmul") else fn(x=t)
        _sync(dev)
        out = _to_numpy(out)
        return out[:, col] if op in ("linear", "matmul") else out
    return run


def _ort_runner(op, dev, n, rows, cols, dtype, col):
    try:
        import onnx
        from onnx import TensorProto, helper
        import onnxruntime as ort
    except Exception as exc:
        raise OrderProbeError(f"ONNX Runtime backend needs onnx and onnxruntime ({exc})")
    if dtype != "float32":
        raise OrderProbeError("the ONNX Runtime probe runs float32 only")
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, n])
    nodes, inits = [], []
    if op == "sum":
        inits.append(helper.make_tensor("ax", TensorProto.INT64, [1], [1]))
        nodes.append(helper.make_node("ReduceSum", ["x", "ax"], ["y"], keepdims=0))
        Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [rows])
    elif op in ("linear", "matmul"):
        inits.append(helper.make_tensor("w", TensorProto.FLOAT, [n, cols], [1.0] * (n * cols)))
        nodes.append(helper.make_node("MatMul", ["x", "w"], ["y"]))
        Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [rows, cols])
    else:
        nodes.append(helper.make_node("Softmax", ["x"], ["y"], axis=-1))
        Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [rows, n])
    g = helper.make_graph(nodes, "probe", [X], [Y], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = min(m.ir_version, 9)
    prov = ["CUDAExecutionProvider"] if dev.startswith("cuda") else ["CPUExecutionProvider"]
    s = ort.InferenceSession(m.SerializeToString(), providers=prov)
    if prov[0] not in s.get_providers():
        raise OrderProbeError(f"ONNX Runtime has no {prov[0]} here")

    def run(x):
        out = s.run(None, {"x": x.astype(np.float32)})[0].astype(np.float64)
        return out[:, col] if op in ("linear", "matmul") else out
    return run


# ----------------------------------------------------------------------------- the oracle


_MAXEXP = {"float32": 127, "float16": 15, "bfloat16": 127}


class _Oracle:
    """Batches probe rows through a Target, twice each, and answers subtree size or triplet
    questions. ``p``: accumulator significand bits assumed."""

    def __init__(self, tg: Target, p: int):
        self.tg, self.p = tg, p
        self.calls = 0
        self.repeat_mismatch = 0
        n = tg.n
        if tg.op == "softmax":
            self.d = 0.75 * 2.0 ** -p
        else:
            lg = max(1, math.ceil(math.log2(n)))
            e_big = min(_MAXEXP[tg.dtype], p + 1 + lg + 4)   # M as large as the input type allows
            self.M = 2.0 ** e_big
            self.u = 2.0 ** (e_big - p - 1 - lg)

    def _run(self, rows: List[np.ndarray]) -> np.ndarray:
        tg = self.tg
        out = []
        if not tg.uniform:                       # one probe per call, in the probed row
            for r in rows:
                X = np.zeros((tg.rows, tg.n))
                X[tg.row] = r
                a, b = tg.run(X), tg.run(X)
                self.calls += 2
                if not ((a[tg.row] == b[tg.row]) | (np.isnan(a[tg.row]) & np.isnan(b[tg.row]))).all():
                    self.repeat_mismatch += 1
                out.append(a[tg.row:tg.row + 1])
            return np.concatenate(out) if out else np.zeros((0,))
        for s in range(0, len(rows), tg.rows):
            chunk = rows[s:s + tg.rows]
            X = np.zeros((tg.rows, tg.n))          # unused rows: zeros (a softmax of zeros is defined)
            for k, r in enumerate(chunk):
                X[k] = r
            a = tg.run(X)
            b = tg.run(X)
            self.calls += 2
            same = (a == b) | (np.isnan(a) & np.isnan(b))
            if not np.all(same):
                self.repeat_mismatch += int((~same).reshape(len(X), -1).any(axis=1).sum())
            out.append(a[:len(chunk)])
        return np.concatenate(out) if out else np.zeros((0,))

    # sum and matmul: the size of the subtree where a and b meet, inside the set S
    def sizes(self, jobs: List[Tuple[int, int, Sequence[int]]]) -> List[Optional[int]]:
        rows = []
        for a, b, S in jobs:
            r = np.zeros(self.tg.n)
            r[list(S)] = self.u
            r[a], r[b] = self.M, -self.M
            rows.append(r)
        res = self._run(rows)
        out: List[Optional[int]] = []
        for (a, b, S), v in zip(jobs, res):
            c = v / self.u
            ok = np.isfinite(c) and c == round(c) and 0 <= c <= len(S) - 2
            out.append(len(S) - int(round(c)) if ok else None)
        return out

    # softmax: do b and c meet before either meets a?
    def triplets(self, jobs: List[Tuple[int, int, int]]) -> List[bool]:
        rows = []
        ld = math.log(self.d)
        for a, b, c in jobs:
            r = np.full(self.tg.n, -np.inf)
            r[a], r[b], r[c] = 0.0, ld, ld
            rows.append(r)
        res = self._run(rows)
        return [bool(res[k, a] < 1.0) for k, (a, b, c) in enumerate(jobs)]


def _levels_sizes(o: _Oracle, problems):
    """For each (pivot, S): the other terms grouped into levels along the pivot's path."""
    jobs, owner = [], []
    for pi, (a, S) in enumerate(problems):
        for b in S:
            if b != a:
                jobs.append((a, b, S))
                owner.append(pi)
    got = o.sizes(jobs)
    per: List[Dict[int, List[int]]] = [dict() for _ in problems]
    bad = [False] * len(problems)
    for (a, b, S), pi, s in zip(jobs, owner, got):
        if s is None:
            bad[pi] = True
            continue
        per[pi].setdefault(s, []).append(b)
    out = []
    for pi, (a, S) in enumerate(problems):
        if bad[pi]:
            out.append(None)
            continue
        lv = [per[pi][s] for s in sorted(per[pi])]
        tot, ok = 1, True
        for s in sorted(per[pi]):         # a level's size is everything up to it
            tot += len(per[pi][s])
            ok &= s == tot
        out.append(lv if ok else None)
    return out


def _levels_triplets(o: _Oracle, problems):
    out = []
    for a, S in problems:
        others = [b for b in S if b != a]
        if len(others) == 1:
            out.append([others])
            continue
        pairs = [(a, b, c) for x, b in enumerate(others) for c in others[x + 1:]]
        res = o.triplets(pairs)
        par = {b: b for b in others}

        def find(z):
            while par[z] != z:
                par[z] = par[par[z]]
                z = par[z]
            return z
        for (a_, b, c), together in zip(pairs, res):
            if together:
                par[find(b)] = find(c)
        groups: Dict[int, List[int]] = {}
        for b in others:
            groups.setdefault(find(b), []).append(b)
        gl = list(groups.values())
        if len(gl) == 1:
            out.append(gl)
            continue
        # G2 is nearer to a than G1 when a and G2's term meet before either meets G1's term
        cmp_jobs = [(g1[0], a, g2[0]) for x, g1 in enumerate(gl) for y, g2 in enumerate(gl) if x != y]
        near = o.triplets(cmp_jobs)
        score = [0] * len(gl)
        k = 0
        for x in range(len(gl)):
            for y in range(len(gl)):
                if x != y:
                    if near[k]:
                        score[y] += 1
                    k += 1
        order = sorted(range(len(gl)), key=lambda z: -score[z])
        lv, cur, last = [], [], None
        for z in order:
            if last is not None and score[z] != last:
                lv.append(cur)
                cur = []
            cur += gl[z]
            last = score[z]
        lv.append(cur)
        out.append(lv)
    return out


def _reconstruct(o: _Oracle) -> Optional[Tree]:
    """Breadth first over subproblems, so every round is one batch of probe rows."""
    n = o.tg.n
    root = {"S": list(range(n))}
    pending = [root]
    levels = _levels_triplets if o.tg.op == "softmax" else _levels_sizes
    while pending:
        nxt = []
        work = [p for p in pending if len(p["S"]) > 2]
        for p in pending:
            if len(p["S"]) == 1:
                p["tree"] = p["S"][0]
            elif len(p["S"]) == 2:
                p["tree"] = tuple(p["S"])
        if work:
            res = levels(o, [(p["S"][0], p["S"]) for p in work])
            for p, lv in zip(work, res):
                if lv is None:
                    return None
                p["kids"] = [{"S": sorted(g)} for g in lv]
                nxt += p["kids"]
        pending = nxt
    def assemble(p):
        if "tree" in p:
            return p["tree"]
        t: Tree = p["S"][0]
        for k in p["kids"]:
            t = (t, assemble(k))
        return t
    return canonical(assemble(root))


# ----------------------------------------------------------------------------- the probe


@dataclass
class OrderResult:
    op: str
    backend: str
    n: int
    rows: int
    cols: int
    dtype: str
    tree: Optional[Tree] = None
    accumulator_bits: Optional[int] = None
    description: str = ""
    verified: Optional[bool] = None       # replay bit for bit on random inputs
    product_model: str = ""               # matmul: which product rounding matched
    repeatable: bool = True
    calls: int = 0
    codes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    label: str = ""

    def as_dict(self):
        return {"label": self.label, "op": self.op, "backend": self.backend, "n": self.n, "rows": self.rows, "cols": self.cols,
                "dtype": self.dtype, "accumulator_bits": self.accumulator_bits,
                "description": self.description, "verified": self.verified,
                "product_model": self.product_model, "repeatable": self.repeatable, "calls": self.calls,
                "depth": None if self.tree is None else depth(self.tree),
                "tree": None if self.tree is None else _jsonable(self.tree),
                "codes": self.codes, "notes": self.notes}

    def to_text(self) -> str:
        head = (f"[{self.label}] " if self.label else "") + f"{self.backend} {self.op} n={self.n} rows={self.rows}" + (
            f" cols={self.cols}" if self.op in ("linear", "matmul") else "") + f" {self.dtype}"
        lines = [head]
        if not self.repeatable:
            lines.append("  TS963 not repeatable: the same probe input gave different results across runs")
        if self.tree is None:
            lines.append("  TS964 no summation order explains the results" + (f" ({'; '.join(self.notes)})" if self.notes else ""))
            return "\n".join(lines)
        lines.append(f"  order: {self.description}")
        lines.append(f"  accumulator: {self.accumulator_bits} significand bits; tree depth {depth(self.tree)}")
        if self.n <= 32:
            lines.append(f"  tree: {to_text(self.tree)}")
        v = {True: "TS962 replayed bit for bit on random inputs", False: "TS964 recovered, but the replay "
             "does not reproduce random inputs bit for bit", None: "not replayed"}[self.verified]
        lines.append(f"  {v}" + (f"; products: {self.product_model}" if self.product_model else ""))
        for x in self.notes:
            lines.append(f"  note: {x}")
        return "\n".join(lines)


def _jsonable(t):
    return t if isinstance(t, int) else [_jsonable(c) for c in t]


def _from_json(t):
    return t if isinstance(t, int) else tuple([_from_json(c) for c in t])


def _deep_enough(n: int) -> None:
    """The tree helpers recurse once per tree level, and a sequential order of n terms is n levels
    deep (a 4096 term sequential sum overflowed Python's default limit of 1000)."""
    need = 8 * int(n) + 1000
    if sys.getrecursionlimit() < need:
        sys.setrecursionlimit(need)


def _on_big_stack(fn, *args, **kw):
    """fn(*args, **kw) in a thread with a 64 MB stack: deep recursion on a raised recursion limit
    must not overflow the main thread's C stack (1 MB on Windows)."""
    import threading
    box = {}

    def body():
        try:
            box["value"] = fn(*args, **kw)
        except BaseException as exc:  # re-raised in the caller's thread
            box["error"] = exc
    old = threading.stack_size()
    threading.stack_size(64 * 1024 * 1024)
    try:
        t = threading.Thread(target=body, name="truestar-orderprobe")
        t.start()
    finally:
        threading.stack_size(old)
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


DEEP = 400  # reductions longer than this are probed on a big stack


def probe(target: Union[Target, str], backend: Optional[str] = None, n: int = 64, rows: int = 64,
          cols: int = 64, dtype: str = "float32", seed: int = 0, verify_rows: int = 64) -> OrderResult:
    """Recover and verify the summation order of ``target`` (a Target, or an op name with
    ``backend``). Returns an OrderResult; see the module docstring for the method."""
    nn = target.n if isinstance(target, Target) else n
    if nn > DEEP:
        import threading
        if threading.current_thread().name != "truestar-orderprobe":
            return _on_big_stack(probe, target, backend, n, rows, cols, dtype, seed, verify_rows)
    tg = target if isinstance(target, Target) else make_target(target, backend or "torch:cpu", n, rows, cols, dtype)
    _deep_enough(tg.n)
    res = OrderResult(tg.op, tg.backend, tg.n, tg.rows, tg.cols, tg.dtype, notes=list(tg.notes))
    if tg.uniform is None:
        tg.uniform = rows_uniform(tg)
    if not tg.uniform:
        res.notes.append(f"rows do not all sum in the same order (identical rows gave different results); "
                         f"the order of row {tg.row} is the one probed and replayed")
    tree, used = None, None
    for p in ((24,) if tg.op == "softmax" else (24, 53)):
        o = _Oracle(tg, p)
        tree = _reconstruct(o)
        res.calls += o.calls
        if o.repeat_mismatch:
            res.repeatable = False
            res.codes += ["TS963", "TS964"]
            res.notes.append(f"{o.repeat_mismatch} probe rows changed between two identical runs: the order is "
                             f"not fixed (atomics or a run dependent split); no single tree describes it")
            return res
        if tree is not None:
            used = p
            break
    if tree is not None and tg.op == "softmax" and not isinstance(tree, int) and len(tree) == tg.n:
        res.notes.append("every pair of terms survived together: the denominator is accumulated wider than "
                         "float32 or in one fused step, so the probe sees a single rounding")
    if tree is None:
        res.codes.append("TS964")
        res.notes.append("the probe answers fit no tree (neither a float32 nor a float64 accumulator)")
        return res
    res.tree = tree
    res.accumulator_bits = used
    res.description = describe(tree)
    _verify(tg, res, seed, verify_rows)
    res.codes.append("TS962" if res.verified else "TS964")
    return res


def _verify(tg: Target, res: OrderResult, seed: int, nrows: int) -> None:
    """Replay the tree on random inputs; must match the backend bit for bit."""
    if tg.op == "softmax":
        res.verified = None
        res.notes.append("softmax: the tree is checked by the triplet answers only (the backend's exp is "
                         "not replayed)")
        return
    rng = np.random.default_rng(seed)
    if tg.uniform:
        nrows = min(nrows, tg.rows)
        X = np.zeros((tg.rows, tg.n))
        X[:nrows] = rng.normal(size=(nrows, tg.n))
        X = _quantize(X, tg.dtype)
        got = tg.run(X)[:nrows]
        Xs = X[:nrows]
    else:
        nrows = min(nrows, 16)
        Xs = _quantize(rng.normal(size=(nrows, tg.n)), tg.dtype)
        got = []
        for r in Xs:
            X = np.zeros((tg.rows, tg.n))
            X[tg.row] = r
            got.append(tg.run(X)[tg.row])
        got = np.array(got)
    acc = np.float64 if res.accumulator_bits == 53 else np.float32
    out_dt = "float32" if tg.backend.startswith("truestar") else tg.dtype
    want = _quantize(replay(res.tree, Xs.astype(np.float32), acc), out_dt)
    res.verified = bool(np.array_equal(got, want))
    if not res.verified:
        k = int((got != want).sum())
        res.notes.append(f"{k} of {nrows} random rows differ from the replay of the recovered tree")
        return
    if tg.op in ("linear", "matmul") and tg.backend.split(":")[0] in ("torch", "compile"):
        res.product_model = _product_model(tg, res, rng, nrows)


def _quantize(X, dtype):
    import torch
    return torch.from_numpy(X).to(_torch_dtype(dtype)).double().numpy()


def _product_model(tg: Target, res: OrderResult, rng, nrows) -> str:
    """With random weights, which product rounding reproduces the backend: fused multiply adds
    (one rounding per product added to a partial sum) or products rounded to float32 first."""
    import torch
    dev = tg.backend.split(":")[1] if ":" in tg.backend else "cpu"
    dt = _torch_dtype(tg.dtype)
    X = _quantize(rng.normal(size=(tg.rows, tg.n)), tg.dtype)
    w = _quantize(rng.normal(size=(tg.n,)), tg.dtype)
    W = torch.ones(tg.cols, tg.n, dtype=dt, device=dev)
    W[tg.col] = torch.from_numpy(w).to(dt).to(dev)
    with torch.no_grad():
        xt = torch.from_numpy(X).to(dt).to(dev)
        if tg.op == "linear":
            out = torch.nn.functional.linear(xt, W, torch.zeros(tg.cols, dtype=dt, device=dev))[:, tg.col]
        else:
            out = (xt @ W.t())[:, tg.col]
    sel = slice(0, nrows) if tg.uniform else slice(tg.row, tg.row + 1)
    got = _to_numpy(out)[sel]
    prods = X[sel] * w[None, :]
    found = []
    for fused in (True, False):
        want = replay(res.tree, X[sel].astype(np.float32), np.float32, prods, fused)
        if np.array_equal(got, want):
            found.append("fused multiply add per added product" if fused else "each product rounded to float32 first")
    return " or ".join(found) if found else "neither product model reproduces random weights bit for bit"


# ----------------------------------------------------------------------------- comparing backends


def explain(a: OrderResult, b: OrderResult, x: Optional[np.ndarray] = None, seed: int = 1) -> Dict:
    """Why two backends differ on the same reduction: their trees, and on one random input row the
    two results and the two trees' replays, which must agree pairwise (TS965 when trees differ)."""
    out = {"same_order": None, "codes": []}
    if a.tree is None or b.tree is None:
        out["note"] = "an order was not recovered"
        return out
    same = canonical(a.tree) == canonical(b.tree)
    out["same_order"] = same
    if not same:
        out["codes"].append("TS965")
    if x is None:
        x = np.random.default_rng(seed).normal(size=(4, a.n)).astype(np.float32)
    ra = _quantize(replay(a.tree, x, np.float64 if a.accumulator_bits == 53 else np.float32), "float32")
    rb = _quantize(replay(b.tree, x, np.float64 if b.accumulator_bits == 53 else np.float32), "float32")
    out["replay_a"], out["replay_b"] = ra.tolist(), rb.tolist()
    out["predicted_bits_differ"] = int((ra != rb).sum())
    out["rows"] = int(len(x))
    return out


def report_text(results: List[OrderResult]) -> str:
    lines = [r.to_text() for r in results]
    ok = [r for r in results if r.tree is not None]
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            A, B = ok[i], ok[j]
            if (A.label, A.op, A.n, A.rows, A.cols) == (B.label, B.op, B.n, B.rows, B.cols) and A.backend != B.backend:
                e = explain(A, B)
                what = f"[{A.label}] " if A.label else f"{A.op} n={A.n}: "
                if e["same_order"]:
                    lines.append(f"{what}{A.backend} and {B.backend}: the same order")
                else:
                    lines.append(f"TS965 {what}{A.backend} and {B.backend} sum in different orders; on {e['rows']} "
                                 f"random rows the two trees give different float32 results in "
                                 f"{e['predicted_bits_differ']}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- cases and comparing files


def case_reductions(case: str, batch: int = 4) -> List[Tuple[str, str, int, int, int]]:
    """(label, op, n, rows, cols) of the reductions of a fingerprint case (scripts/laptop_fingerprint.py):
    ``transformer`` is nn.TransformerEncoderLayer(128, 4, 512) on [batch, 32, 128] inputs. The
    attention score and value products (batched matmuls over heads, K = 32) and the layer norms
    (not plain sums) are not probed."""
    if case != "transformer":
        raise OrderProbeError(f"unknown case {case!r}; known: transformer")
    t = batch * 32
    return [("self_attn.in_proj", "linear", 128, t, 384), ("self_attn.out_proj", "linear", 128, t, 128),
            ("linear1", "linear", 128, t, 512), ("linear2", "linear", 512, t, 128),
            ("attention softmax", "softmax", 32, batch * 4 * 32, 1)]


def compare_files(pa: str, pb: str) -> Tuple[List[str], List[str]]:
    """Text lines and codes comparing two ``--json`` files (for example one per GPU): every reduction
    present in both, with the same op and shape, is the same order or not (TS965)."""
    _deep_enough(1 << 15)
    def load(p):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d.get("env", {}), {(r.get("label", ""), r["op"], r["n"], r["rows"], r["cols"], r["dtype"]): r
                                  for r in d["results"]}
    ea, A = load(pa)
    eb, B = load(pb)
    lines = [f"A: {ea.get('device', pa)}", f"B: {eb.get('device', pb)}"]
    codes = []
    for k in A:
        if k not in B:
            continue
        ra, rb = A[k], B[k]
        name = k[0] or f"{k[1]} n={k[2]} rows={k[3]}"
        if ra["tree"] is None or rb["tree"] is None:
            lines.append(f"{name}: an order was not recovered on one side")
            continue
        ta, tb = canonical(_from_json(ra["tree"])), canonical(_from_json(rb["tree"]))
        if ta == tb:
            lines.append(f"{name}: same order ({ra['description']})")
        else:
            codes.append("TS965")
            x = np.random.default_rng(7).normal(size=(64, k[2])).astype(np.float32)
            d = int((_quantize(replay(ta, x), "float32") != _quantize(replay(tb, x), "float32")).sum())
            lines.append(f"TS965 {name}: different orders; on 64 random rows the two trees give different "
                         f"float32 results in {d}\n  A: {ra['description']}\n  B: {rb['description']}")
    return lines, sorted(set(codes))


def _env() -> Dict:
    import platform
    import torch
    e = {"torch": torch.__version__, "platform": platform.platform()}
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        e.update(device=f"{pr.name}, {pr.multi_processor_count} SMs", cuda=torch.version.cuda)
    elif torch.backends.mps.is_available():
        import subprocess
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:  # informational only
            chip = ""
        e["device"] = f"{chip or 'Apple silicon'} GPU (MPS)"
    else:
        e["device"] = platform.processor() or "cpu"
    return e


# ----------------------------------------------------------------------------- CLI


def add_parser(sub) -> None:
    p = sub.add_parser("orderprobe", help="recover the summation order (tree) a backend uses for a sum, a "
                                          "matmul's K reduction or a softmax denominator")
    p.add_argument("action", nargs="?", default="run", choices=("run", "compare"))
    p.add_argument("files", nargs="*", help="compare: two --json files (for example one per GPU)")
    p.add_argument("--op", choices=OPS, default="sum")
    p.add_argument("--case", help="probe every reduction of a fingerprint case instead (transformer)")
    p.add_argument("--batch", type=int, default=4, help="--case batch size")
    p.add_argument("--n", type=int, default=64, help="reduction length (sum length, K, softmax row length)")
    p.add_argument("--rows", type=int, default=64, help="rows of the probe input (the M of a matmul)")
    p.add_argument("--cols", type=int, default=64, help="matmul output columns (N)")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    p.add_argument("--backends", default="torch:cpu",
                   help="comma list: torch:cpu, torch:cuda, torch:mps, compile:cpu, compile:cuda, compile:mps, "
                        "truestar:cpu, truestar:cuda, truestar:mps (Metal), ort:cpu, ort:cuda")
    p.add_argument("--json", dest="json_path", help="write every result (with the trees) here")
    p.add_argument("--format", choices=("text", "agent"), default="text")


def run(a) -> int:
    """The CLI: on a big stack, since deep trees are walked after the probes too (printing,
    comparing, writing JSON)."""
    import threading
    if threading.current_thread().name == "truestar-orderprobe":
        return _run(a)
    return _on_big_stack(_run, a)


def _run(a) -> int:
    if a.action == "compare":
        if len(a.files) != 2:
            print("truestar orderprobe compare needs two --json files", file=sys.stderr)
            return 2
        lines, codes = compare_files(*a.files)
        print("\n".join(lines))
        return 1 if codes else 0
    results = []
    issues = []
    try:
        jobs = (case_reductions(a.case, a.batch) if a.case else [("", a.op, a.n, a.rows, a.cols)])
    except OrderProbeError as exc:
        print(f"truestar orderprobe: {exc}", file=sys.stderr)
        return 2
    for b in [x.strip() for x in a.backends.split(",") if x.strip()]:
        for label, op, n, rows, cols in jobs:
            try:
                r = probe(op, b, n, rows, cols, a.dtype)
            except OrderProbeError as exc:
                issues.append({"code": "TS001", "message": f"{b}: {exc}"})
                if a.format == "text":
                    print(f"{b}: unavailable ({exc})")
                break
            r.label = label
            results.append(r)
    body = {"env": _env(), "results": [r.as_dict() for r in results]}
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            json.dump(body, f, indent=1)
    bad = any(not r.repeatable or r.tree is None or r.verified is False for r in results)
    if a.format == "agent":
        from truestar import __version__
        codes = sorted({c for r in results for c in r.codes} | {i["code"] for i in issues})
        print(json.dumps({"schema_version": 1, "tool": "truestar", "command": "orderprobe",
                          "truestar_version": __version__, "status": "finding" if bad else "ok",
                          "exit_code": 1 if bad else 0, "codes": codes, "issues": issues, **body}))
    else:
        print(report_text(results))
    return 1 if bad else 0
