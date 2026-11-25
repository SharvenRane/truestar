"""The IR ops of chunked linear attention and state space scans (pain row P60).

``linear_attention(q, k, v, g, beta) rule= scale=`` is the recurrence of docs/LINEAR_ATTENTION.md
section 1 as one IR op: per batch, head and step, ``S_t = D_t S_{t-1} + k_t u_t^T`` with
``u_t = v_t`` (rule ``add``, the Mamba 2 SSD scan) or ``u_t = beta_t (v_t - (D_t S_{t-1})^T k_t)``
(rule ``delta``, Gated DeltaNet), ``D_t = diag(exp(g_t))`` and ``o_t = scale * S_t^T q_t``. q and
k are ``[B, T, H, dk]``, v ``[B, T, H, dv]``, g ``[B, T, H]``, ``[B, T, H, dk]`` or absent, beta
``[B, T, H]`` or absent (the delta rule needs it); the output has v's type.

* Reference: the float64 sequential recurrence (``truestar.linattn.recurrent``), one step at a
  time, no chunks: the definition, not an algorithm.
* Reverse mode: ``linear_attention_vjp(q, k, v, g, beta, do) wrt=i``, the cotangent of operand i,
  whose reference is torch.autograd through the float64 recurrence.
* Forward mode: ``linear_attention_jvp(q, k, v, g, beta, tq, tk, tv, tg, tbeta)``, the tangent of
  o (an absent tangent is zero), reference torch.func.jvp through the float64 recurrence.
* Compiled (compiler/runtime.py, library steps on the executor's device): ``linear_attention`` runs
  ``truestar.linattn.linear_attention`` (the Triton kernels on CUDA, the chunked PyTorch form on
  the CPU), the two derivative ops run autograd and forward AD through the chunked form at chunk
  16. Every step is checked against these references on its first run, like a library kernel.
* vmap: the exact per element loop (the ops are pure).

The importer maps ``torch.ops.truestar.linear_attention`` (truestar/linattn/torch_op.py) to
``linear_attention``; the Hugging Face lowerings in ``hf_linattn.py`` produce that op.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from . import reference, shapes
from .core import ShapeError, TensorType, dims_equal

OPS = ("linear_attention", "linear_attention_vjp", "linear_attention_jvp")
NAMES = ["q", "k", "v", "g", "beta"]


def _same(a: Sequence, b: Sequence) -> bool:
    return len(a) == len(b) and all(dims_equal(x, y) for x, y in zip(a, b))


def _check(op: str, types: Sequence[Optional[TensorType]], attrs) -> TensorType:
    q, k, v, g, beta = types[:5]
    labels = shapes._labels(types[:5], NAMES)
    rule = attrs.get("rule")
    if rule not in ("add", "delta"):
        raise ShapeError(op, f"rule must be 'add' or 'delta', got {rule!r}", labels)
    if not isinstance(attrs.get("scale"), (int, float)) or isinstance(attrs.get("scale"), bool):
        raise ShapeError(op, f"scale must be a number, got {attrs.get('scale')!r}", labels)
    if q is None or k is None or v is None:
        raise ShapeError(op, "q, k and v are required", labels)
    if q.rank != 4 or not _same(q.shape, k.shape):
        raise ShapeError(op, "q and k must both be [B, T, H, dk]", labels)
    if v.rank != 4 or not _same(v.shape[:3], q.shape[:3]):
        raise ShapeError(op, "v must be [B, T, H, dv] with q's B, T and H", labels)
    if g is not None and not (_same(g.shape, q.shape[:3]) or _same(g.shape, q.shape)):
        raise ShapeError(op, "g must be [B, T, H] (one decay per head) or [B, T, H, dk] (per key channel)", labels)
    if rule == "delta" and beta is None:
        raise ShapeError(op, "the delta rule needs beta [B, T, H]", labels)
    if rule == "add" and beta is not None:
        raise ShapeError(op, "beta is only read by the delta rule", labels)
    if beta is not None and not _same(beta.shape, q.shape[:3]):
        raise ShapeError(op, "beta must be [B, T, H]", labels)
    for t in (q, k, v, g, beta):
        if t is not None and (not t.dtype.is_floating_point or t.dtype != v.dtype):
            raise ShapeError(op, f"every operand must be floating point with v's dtype {v.dtype}", labels)
    return v


@shapes.rule("linear_attention")
def _shape(types, attrs):
    if len(types) != 5:
        raise ShapeError("linear_attention", f"takes 5 operands (q, k, v, g, beta), got {len(types)}")
    return _check("linear_attention", types, attrs)


@shapes.rule("linear_attention_vjp")
def _shape_vjp(types, attrs):
    op = "linear_attention_vjp"
    if len(types) != 6:
        raise ShapeError(op, f"takes 6 operands (q, k, v, g, beta, do), got {len(types)}")
    out = _check(op, types, attrs)
    do = types[5]
    if do is None or not do.dtype.is_floating_point or not _same(do.shape, out.shape):
        raise ShapeError(op, f"do must be a float cotangent of o's shape {out}", [("do", do)] if do else [])
    i = attrs.get("wrt")
    if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < 5 or types[i] is None:
        raise ShapeError(op, f"wrt must name a present operand (0 to 4), got {i!r}")
    return types[i]


@shapes.rule("linear_attention_jvp")
def _shape_jvp(types, attrs):
    op = "linear_attention_jvp"
    if len(types) != 10:
        raise ShapeError(op, f"takes 10 operands (q, k, v, g, beta and one tangent each), got {len(types)}")
    out = _check(op, types, attrs)
    for name, x, t in zip(NAMES, types[:5], types[5:]):
        if t is not None and (x is None or not t.same_as(x)):
            raise ShapeError(op, f"the tangent of {name} must have its operand's type", [(name, x), ("tangent", t)])
    return out


# ----------------------------------------------------------------------------- float64 references


def _f64(args):
    return [None if a is None else a.to(torch.float64) for a in args]


def _recurrent(attrs):
    from ..linattn.reference import recurrent

    def f(q, k, v, g, beta):
        return recurrent(q, k, v, g, beta, rule=attrs["rule"], scale=float(attrs["scale"]))[0]
    return f


@reference.impl("linear_attention")
def _ref(attrs, args, shape):
    return _recurrent(attrs)(*_f64(args[:5]))


@reference.impl("linear_attention_vjp")
def _ref_vjp(attrs, args, shape):
    ins, do = _f64(args[:5]), args[5].to(torch.float64)
    i = attrs["wrt"]
    with torch.enable_grad():
        leaves = [None if x is None else x.detach().requires_grad_(j == i) for j, x in enumerate(ins)]
        o = _recurrent(attrs)(*leaves)
        (gx,) = torch.autograd.grad(o, [leaves[i]], do, allow_unused=True)
    return torch.zeros_like(ins[i]) if gx is None else gx


@reference.impl("linear_attention_jvp")
def _ref_jvp(attrs, args, shape):
    ins, ts = _f64(args[:5]), _f64(args[5:])
    present = [j for j, x in enumerate(ins) if x is not None]
    f = _recurrent(attrs)

    def g(*xs):
        full = list(ins)
        for j, x in zip(present, xs):
            full[j] = x
        return f(*full)
    tans = tuple(torch.zeros_like(ins[j]) if ts[j] is None else ts[j] for j in present)
    return torch.func.jvp(g, tuple(ins[j] for j in present), tans)[1]


# ----------------------------------------------------------------------------- derivative rules
# Called at the end of grad_rules.py, forward.py, vmap.py and importer.py, so each registry gets the
# rule when it is loaded and this module never imports them at its own import (no cycle).


def register_grad(grad_rule) -> None:
    @grad_rule("linear_attention")
    def _grad(e, n, g):
        """One ``linear_attention_vjp`` per operand that needs a cotangent."""
        ins = list(n.inputs)
        attrs = dict(rule=n.attrs["rule"], scale=n.attrs["scale"])

        def one(i):
            return lambda: e.op("linear_attention_vjp", *ins, g, wrt=i, **attrs)
        return [None if ins[i] is None else one(i) for i in range(5)]


def register_jvp(jvp_rule) -> None:
    @jvp_rule("linear_attention")
    def _jvp(e, n, ts):
        ts = list(ts) + [None] * (5 - len(ts))
        if all(t is None for t in ts):
            return [None]
        return [e.op("linear_attention_jvp", *n.inputs, *ts, rule=n.attrs["rule"], scale=n.attrs["scale"])]


def register_vmap(loop_fallback: set) -> None:
    loop_fallback.update(OPS)


def register_import(aten) -> None:
    @aten("truestar.linear_attention.default")
    def _import(c, a):
        """torch.ops.truestar.linear_attention (truestar/linattn/torch_op.py), produced by the
        Hugging Face lowerings of hf_linattn.py."""
        ins = [None if a.get(n) is None else c.tensor(a[n]) for n in NAMES]
        return c.emit("linear_attention", ins, rule=str(a["rule"]), scale=float(a["scale"]))
