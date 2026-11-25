"""The fused attention backward: ``attention_grad`` (2026-11-25).

Before, the gradient pass lowered ``attention`` to its parts (a score matmul, the causal ``where``,
``softmax``, a matmul with v) and differentiated those: the forward stored the full ``[.., T, T]``
probabilities, and the backward was two score sized matmuls, a softmax backward and several full
passes over score sized intermediates. Now, when the attention is plain self attention (q, k, v
and the output of one shape ``[.., T, D]``, no explicit mask, causal or not, a fixed head dim of
at most 128), the forward ``attention`` node is kept whole (it compiles to the online softmax
kernel and stores nothing score sized) and its gradient rule emits one op:

    attention_grad(q, k, v, o, do) scale= is_causal=  ->  [3, .., T, D]  (dq, dk, dv stacked)

* Reference (``ir.reference``): the definition in float64, ``s = q k^T scale`` (masked above the
  diagonal when causal), ``p = softmax(s)``, ``dv = p^T do``, ``dp = do v^T``,
  ``ds = p (dp - rowsum(dp p))``, ``dq = ds k scale``, ``dk = ds^T q scale``. It does not read
  ``o``: the reference is the gradient of the attention itself, so a wrong forward output cannot
  vouch for its own gradient.
* Compiled (``compiler/attention_bwd.py``): on CUDA three Truestar Triton kernels (row statistics,
  then dk and dv per key block, then dq per query block), each output element written by exactly
  one program that sums in a fixed order fixed by the shapes alone, so the bits are the same on
  every call; on the CPU the same three passes in float32 torch blocks. Region checked against the
  reference on its first run like every library step.

``TRUESTAR_FUSED_ATTENTION_GRAD=0`` keeps the old lowering (for measurement).
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch

from . import reference, shapes
from .core import ShapeError, TensorType, dims_equal

OP = "attention_grad"
MAX_HEAD = 128


def _same(a, b) -> bool:
    return len(a) == len(b) and all(dims_equal(x, y) for x, y in zip(a, b))


@shapes.rule(OP)
def _shape(types, attrs):
    if len(types) != 5 or any(t is None for t in types):
        raise ShapeError(OP, f"takes 5 operands (q, k, v, o, do), got {len(types)}")
    q = types[0]
    labels = shapes._labels(types, ["q", "k", "v", "o", "do"])
    if q.rank < 3:
        raise ShapeError(OP, "q must be [.., T, D] with at least one batch dim", labels)
    for t in types[1:]:
        if not _same(t.shape, q.shape) or t.dtype != q.dtype:
            raise ShapeError(OP, "q, k, v, o and do must have one shape and dtype (self attention)", labels)
    if not q.dtype.is_floating_point:
        raise ShapeError(OP, "operands must be floating point", labels)
    if not isinstance(attrs.get("scale"), (int, float)) or isinstance(attrs.get("scale"), bool):
        raise ShapeError(OP, f"scale must be a number, got {attrs.get('scale')!r}", labels)
    return TensorType(q.dtype, (3,) + tuple(q.shape))


def reference_grad(q, k, v, do, scale: float, causal: bool):
    """dq, dk, dv of softmax(q k^T scale) v in the dtype of the operands (float64 for the reference)."""
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        t = s.shape[-1]
        keep = torch.ones((s.shape[-2], t), dtype=torch.bool, device=s.device).tril()
        s = s.masked_fill(~keep, -math.inf)
    p = torch.softmax(s, dim=-1)
    dv = torch.matmul(p.transpose(-1, -2), do)
    dp = torch.matmul(do, v.transpose(-1, -2))
    ds = p * (dp - (dp * p).sum(-1, keepdim=True))
    dq = torch.matmul(ds, k) * scale
    dk = torch.matmul(ds.transpose(-1, -2), q) * scale
    return dq, dk, dv


@reference.impl(OP)
def _ref(attrs, args, shape):
    # the interpreter's compute dtype: float64 for the reference, float32 for the independent float32
    # evaluation the end to end check compares against (accept_ill_conditioned, docs/COMPILER.md)
    q, k, v, _o, do = (a.to(reference.COMPUTE_DTYPE) for a in args)
    return torch.stack(reference_grad(q, k, v, do, float(attrs["scale"]), bool(attrs["is_causal"])))


def enabled() -> bool:
    return os.environ.get("TRUESTAR_FUSED_ATTENTION_GRAD", "1") != "0"


def eligible(n) -> bool:
    """True when the gradient pass keeps this ``attention`` node whole and differentiates it with
    ``attention_grad`` instead of lowering it to its parts."""
    if n.op != "attention" or not enabled():
        return False
    q, k, v, mask = n.inputs
    if mask is not None:
        return False
    out = n.output.type
    qt = q.type
    if qt.rank < 3 or not qt.dtype.is_floating_point:
        return False
    if not all(x.type.dtype == qt.dtype and _same(x.type.shape, qt.shape) for x in (k, v)):
        return False
    if not _same(out.shape, qt.shape) or out.dtype != qt.dtype:
        return False
    d = qt.shape[-1]
    return isinstance(d, int) and 1 <= d <= MAX_HEAD


def scale_of(n) -> float:
    s = n.attrs.get("scale")
    return float(s) if s is not None else 1.0 / math.sqrt(n.inputs[0].type.shape[-1])


def register_grad(grad_rule) -> None:
    @grad_rule("attention")
    def _grad(e, n, g):
        """One ``attention_grad`` for all three cotangents (emitted once, on the first request)."""
        q, k, v, _mask = n.inputs
        memo = {}

        def packed():
            if "p" not in memo:
                memo["p"] = e.op(OP, q, k, v, n.output, g, scale=scale_of(n),
                                 is_causal=bool(n.attrs.get("is_causal", False)))
            return memo["p"]

        def one(i):
            return lambda: e.op("select", packed(), dim=0, index=i)
        return [one(0), one(1), one(2), None]


def register_lowering(lowering) -> None:
    @lowering(OP)
    def _lower(e, n):
        """attention_grad as its definition in IR ops (the reference's formulas), for transforms that
        look inside it: forward mode (a Hessian of a model with attention is forward over reverse)
        and a second reverse pass. It reads q, k, v and do, not o, like the reference."""
        q, k, v, _o, do = n.inputs
        scale = float(n.attrs["scale"])
        swap = list(range(q.type.rank))
        swap[-1], swap[-2] = swap[-2], swap[-1]
        s = e.scale(e.op("matmul", q, e.op("permute", k, dims=swap)), scale)
        if n.attrs.get("is_causal", False):
            rows, cols = q.type.shape[-2], k.type.shape[-2]
            if not (isinstance(rows, int) and isinstance(cols, int)):
                from .core import UnsupportedOpError
                raise UnsupportedOpError(f"{OP} with is_causal needs a fixed sequence length to be lowered, "
                                         f"got {rows} x {cols}")
            keep = e.tensor_const("causal_mask", torch.ones((rows, cols), dtype=torch.bool).tril(), torch.bool)
            s = e.op("where", keep, s, e.op("full_like", s, value=-math.inf, dtype=None))
        p = e.op("softmax", s, dim=-1)
        dv = e.op("matmul", e.op("permute", p, dims=swap), do)
        dp = e.op("matmul", do, e.op("permute", v, dims=swap))
        row = e.op("sum", e.op("mul", dp, p), dims=[q.type.rank - 1], keepdim=True)
        ds = e.op("mul", p, e.op("sub", dp, row))
        dq = e.scale(e.op("matmul", ds, k), scale)
        dk = e.scale(e.op("matmul", e.op("permute", ds, dims=swap), q), scale)
        parts = [e.op("reshape", x, shape=[1] + list(q.type.shape)) for x in (dq, dk, dv)]
        e.op("cat", *parts, dim=0, out=n.output)


def register_vmap(batch_rule) -> None:
    @batch_rule(OP)
    def _batch(bt, n, ins):
        """Every operand has the same shape: unbatched ones are expanded along the batch dim, the op
        runs on [n, .., T, D] operands and its stacked [3, n, ..] result is permuted to [n, 3, ..]."""
        args = [bt.full(p) for p in ins]
        y = bt.e.op(OP, *args, **n.attrs)
        dims = list(range(y.type.rank))
        dims[0], dims[1] = 1, 0
        return [(bt.e.op("permute", y, dims=dims), True)]
