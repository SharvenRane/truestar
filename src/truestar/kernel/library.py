"""cuBLAS (through torch) as one more implementation of a float32 contraction kernel.

Triton's IEEE float32 ``tl.dot`` runs on FMA units at about 20 TFLOPS on an RTX 5070 Ti where
cuBLAS's SIMT kernels reach about 27 (docs/QUALITY_BAR.md, contraction entries), and in strict
float32 the contractions are most of a transformer layer's time. So for a contraction kernel whose
epilogue cuBLAS can apply itself, the launch config ``{"LIB": 1}`` runs it as one torch call:

=====================================  =============================================
kernel IR output expression            torch call (cuBLAS, cuBLASLt epilogue)
=====================================  =============================================
``sum[k](A * B)``                      ``mm``
``sum[k](A * B) + bias[n]``            ``addmm(bias, A, B)``
``maximum(sum + bias[n], 0)``          ``_addmm_activation(bias, A, B)`` (relu)
``R + sum``                            ``addmm(R, A, B)``
``R + (sum + bias[n])``                ``addmm(R, A, B)`` then ``add_(bias)``
=====================================  =============================================

A and B must be plain loads (B viewable as a ``[K, N]`` matrix; A is copied into row order when its
strides do not allow a view, and the duel then times the copy too), and the output's axes must be the
rows then N. Anything else keeps the Triton kernel.


Whether it runs is measured, not assumed: the first launch of each shape bucket times the Triton
config against this one and keeps the faster (``CompiledKernel._choose_config``); the choice is
cached with the tuned configs. Like every launch config it is a different binary, so it is checked
against the float64 reference on its first call at every new shape signature.

Precision: only float32 kernels, and only when PyTorch's float32 matmul precision at the call is
the one the kernel was compiled for (strict IEEE or TF32), so cuBLAS computes in exactly the
precision the program asked for; otherwise the Triton launch runs. Deterministic kernels never use
it: cuBLAS picks its algorithm by shape, so a row's bits could depend on the batch size it ran in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .ir import Axis, Binary, Const, Load, Reduce


@dataclass(frozen=True)
class LibForm:
    a: str                      # canonical buffer name of the left factor
    a_axes: Tuple[Axis, ...]    # its index axes, in buffer dim order
    b: str
    b_axes: Tuple[Axis, ...]
    rows: Tuple[Axis, ...]      # output row axes (folded batch axes, then m)
    n: Axis
    k: Axis
    bias: Optional[str] = None  # [N] vector added to every row
    res: Optional[str] = None   # [rows..., N] matrix added (same axes as the output)
    relu: bool = False


def _plain(e, want: set) -> Optional[Tuple[str, Tuple[Axis, ...]]]:
    if not isinstance(e, Load):
        return None
    idx = tuple(e.index)
    if not all(isinstance(a, Axis) for a in idx) or len(set(idx)) != len(idx) or set(idx) != want:
        return None
    return e.buffer.name, idx


def _is_zero(e) -> bool:
    return isinstance(e, Const) and float(e.value) == 0.0


def library_form(k, p) -> Optional[LibForm]:
    """The cuBLAS form of contraction kernel ``k`` (canonical) with plan ``p``, or None."""
    if p.kind != "contraction" or p.batch or getattr(p, "deterministic", False):
        return None
    if len(k.outputs) != 1:
        return None
    o = k.outputs[0]
    rows = tuple(p.fold) + (p.m,)
    if tuple(o.axes) != rows + (p.n,) or o.dtype != "float32":
        return None
    a = _plain(p.a, set(rows) | {p.k})
    b = _plain(p.b, {p.k, p.n})
    if a is None or b is None or a[0] == b[0]:
        return None
    for buf in (a[0], b[0]):
        if next(x for x in k.inputs if x.name == buf).dtype != "float32":
            return None
    red = p.red
    e = o.expr
    relu = False
    if isinstance(e, Binary) and e.op == "maximum" and _is_zero(e.b):
        relu, e = True, e.a
    res = bias = None

    def split_add(x):
        if isinstance(x, Binary) and x.op == "add":
            return [x.a, x.b]
        return [x]
    terms = []
    for t in split_add(e):
        terms += split_add(t) if not isinstance(t, Reduce) else [t]
    from .ir import to_text
    rt = to_text(red)
    is_red = lambda t: isinstance(t, Reduce) and (t is red or to_text(t) == rt)  # noqa: E731
    if sum(1 for t in terms if is_red(t)) != 1:
        return None
    for t in terms:
        if is_red(t):
            continue
        v = _plain(t, {p.n})
        if v is not None and len(v[1]) == 1 and bias is None:
            bias = v[0]
            continue
        r = _plain(t, set(o.axes))
        if r is not None and r[1] == tuple(o.axes) and res is None:
            res = r[0]
            continue
        return None
    if relu and res is not None:
        return None
    if len(terms) > 3:
        return None
    return LibForm(a[0], a[1], b[0], b[1], rows, p.n, p.k, bias, res, relu)


def _matrix(t, axes: Tuple[Axis, ...], order: Tuple[Axis, ...], last: Axis):
    """t (indexed by ``axes``) as a 2D view [prod(order) , size(last)] without a copy, or None."""
    perm = [axes.index(a) for a in order] + [axes.index(last)]
    v = t.permute(perm)
    rows = 1
    for d in v.shape[:-1]:
        rows *= int(d)
    shape, strides = list(v.shape[:-1]), list(v.stride()[:-1])
    # merge the leading dims: each must step by the next one's extent times its stride
    st = None
    for d in range(len(shape) - 1, -1, -1):
        if shape[d] == 1:
            continue
        if st is None:
            st, span = strides[d], shape[d] * strides[d]
        elif strides[d] != span:
            return None
        else:
            span = shape[d] * strides[d]
    st = 1 if st is None else st
    return t.as_strided((rows, int(v.shape[-1])), (st if rows > 1 else 1, v.stride()[-1]), t.storage_offset())


def views(form: LibForm, by_canon: Dict[str, object]):
    """(A [rows, K], B [K, N]) of the input tensors: views where the strides allow, else a copy
    of the left factor in row order (an attention projection on [T, B, C] activations whose rows
    are not one stride apart). The copy is part of what the duel times."""
    t = by_canon[form.a]
    A = _matrix(t, form.a_axes, form.rows, form.k)
    if A is None:
        perm = [form.a_axes.index(a) for a in form.rows] + [form.a_axes.index(form.k)]
        v = t.permute(perm)
        A = v.reshape(-1, v.shape[-1])
    B = _matrix(by_canon[form.b], form.b_axes, (form.k,), form.n)
    if B is None:
        return None
    return A, B


def run(form: LibForm, by_canon: Dict[str, object], out, mats=None):
    """Compute the kernel's output into ``out`` (contiguous, the output's shape) with torch."""
    import torch
    A, B = views(form, by_canon) if mats is None else mats
    o2 = out.view(A.shape[0], B.shape[1])
    bias = by_canon[form.bias] if form.bias else None
    if form.res is not None:
        R = by_canon[form.res]
        if R.is_contiguous():
            R2 = R.view(A.shape[0], B.shape[1])
        else:
            R2 = R.reshape(A.shape[0], B.shape[1])
        torch.addmm(R2, A, B, out=o2)
        if bias is not None:
            o2.add_(bias)
    elif form.relu:
        if bias is not None:
            torch.ops.aten._addmm_activation.out(bias, A, B, beta=1, alpha=1, use_gelu=False, out=o2)
        else:
            torch.mm(A, B, out=o2)
            o2.relu_()
    elif bias is not None:
        torch.addmm(bias, A, B, out=o2)
    else:
        torch.mm(A, B, out=o2)
    return out


def torch_precision_is(tf32: bool) -> bool:
    """True when PyTorch's float32 matmul precision right now is TF32 exactly when ``tf32``."""
    import torch
    getter = getattr(torch._C, "_get_fp32_precision_getter", None)
    if getter is not None:
        try:
            return (getter("cuda", "matmul") == "tf32") == bool(tf32)
        except RuntimeError:
            pass
    try:
        return bool(torch.backends.cuda.matmul.allow_tf32) == bool(tf32)
    except RuntimeError:
        return False
