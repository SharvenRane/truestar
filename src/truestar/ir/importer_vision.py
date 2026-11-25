"""ATen ops of the per image size vision encoders (SigLIP2 and Pixtral vision, docs/MODEL_COVERAGE.md).

Both encoders read each image's patch grid from a small integer tensor (``spatial_shapes``,
``image_sizes``) with ``.tolist()`` or ``.item()`` and use the numbers as sizes: SigLIP2 resizes its
position embedding grid to each image with antialiased bilinear interpolation, Pixtral slices each
image's patch grid. When the model builds that tensor from its own constants (a fixed image size),
``import_module`` exports with torch's constant tracking for small tensors, so every such number is
a known integer in the graph; these handlers then take:

* ``item`` of a known value: the number itself (no op). Of a value that depends on an input it is
  refused (TS861): the size it decides is data dependent.
* ``_assert_scalar`` of a known value: nothing when true (the assertion torch.export kept), refused
  when false. Python comparisons of known numbers (``ge``, ``gt`` and so on) are evaluated.
* ``floor_divide`` and ``div`` with a rounding mode are in importer_gen.py (constants folded with
  torch.div itself, integers through float64); ``empty`` is in importer.py (NaN until written,
  SigLIP2 writes every element before reading).
* ``_upsample_bilinear2d_aa``: ``A_h x A_w^T`` with constant matrices holding the triangle filter
  stretched by the scale when downsampling (support ``max(scale, 1)``), each row normalized to sum
  1, as torch's antialiased resize defines it; with no downsampling it is plain bilinear except at
  the borders, where the taps outside the image are dropped and the rest renormalized.

Written from the ops' documented meaning; each is checked against eager PyTorch in float64 in
tests/test_hf_models.py and tests/test_ir.py.
"""

from __future__ import annotations

import numbers
from typing import Optional

import torch

from .importer import HANDLERS, aten
from .importer_extra import _interp, _op, _out_sizes, _unsupported


def _known(v) -> bool:
    return isinstance(v, numbers.Number) and not isinstance(v, torch.SymInt)


_CHECK_TARGETS = ("aten._assert_scalar.default", "aten.sym_constrain_range_for_size.default",
                  "aten.sym_constrain_range.default")


def _feeds_only_checks(node, seen=None) -> bool:
    """True when the value reaches nothing but assertions (directly or through Python arithmetic and
    comparisons): torch.export keeps such reads after it used the known value for the sizes."""
    import operator

    from .importer import _target_name

    seen = set() if seen is None else seen
    for u in node.users:
        if u in seen:
            continue
        seen.add(u)
        if u.op == "call_function" and _target_name(u) in _CHECK_TARGETS:
            continue
        if u.op == "call_function" and getattr(u.target, "__module__", "") == "_operator"                 and u.target is not operator.getitem and _feeds_only_checks(u, seen):
            continue
        return False
    return True


@aten("aten.item.default", "aten._local_scalar_dense.default")
def _item(c, a):
    v = c.node.meta.get("val")
    if not _known(v) and _feeds_only_checks(c.node):
        return None      # read again after export used the known value; only assertions see it
    if isinstance(v, (torch.SymInt, torch.SymFloat, torch.SymBool)) or not _known(v):
        _unsupported(c, ".item() or .tolist() of a value that depends on an input decides a size or a branch "
                        "here (data dependent); build that tensor from constants (a fixed image size) or "
                        "pass the sizes as Python ints", "TS861")
    return bool(v) if isinstance(v, bool) else v


@aten("aten._assert_scalar.default")
def _assert_scalar(c, a):
    v = c.val(a["self"])
    if v is None:        # an assertion on a read nothing else uses (see _item)
        return None
    if isinstance(v, torch.fx.Node) or not _known(v):
        _unsupported(c, f"an assertion on a data dependent value ({a.get('assert_msg')})", "TS861")
    if not v:
        _unsupported(c, f"the model's own assertion fails here: {a.get('assert_msg')}")
    return None


def aa_matrix(in_size: int, out_size: int, factor: Optional[float] = None) -> torch.Tensor:
    """[out, in] float64 weights of torch's antialiased bilinear resize (align_corners=False)."""
    scale = 1.0 / factor if factor else in_size / out_size
    support = max(scale, 1.0)          # the triangle filter's half width (1) stretched when downsampling
    inv = 1.0 / scale if scale >= 1.0 else 1.0
    m = torch.zeros((out_size, in_size), dtype=torch.float64)
    for i in range(out_size):
        center = scale * (i + 0.5)
        lo = max(int(center - support + 0.5), 0)
        hi = min(int(center + support + 0.5), in_size)
        w = [max(0.0, 1.0 - abs((j - center + 0.5) * inv)) for j in range(lo, hi)]
        total = sum(w)
        for j, wj in zip(range(lo, hi), w):
            m[i, j] = wj / total if total != 0.0 else wj
    return m


def _aa_handler(c, a):
    x = c.tensor(a["input"] if "input" in a else a["self"])
    if a.get("align_corners"):
        _unsupported(c, "antialiased interpolation with align_corners=True")
    factors = a.get("scale_factors")
    if factors is None and ("scales_h" in a or "scales_w" in a):
        factors = [a.get("scales_h"), a.get("scales_w")]
    return _interp(c, x, _out_sizes(c, x, a, 2), factors, "linear_aa", False)


for _ov in ("vec", "default"):
    HANDLERS[f"aten._upsample_bilinear2d_aa.{_ov}"] = _aa_handler


@aten("aten.baddbmm.default")
def _baddbmm(c, a):
    """beta * input + alpha * (batch1 @ batch2) (nn.MultiheadAttention with a mask, SigLIP2's head)."""
    inp, b1, b2 = c.tensor(a["self"]), c.tensor(a["batch1"]), c.tensor(a["batch2"])
    beta, alpha = a.get("beta", 1), a.get("alpha", 1)
    prod = _op(c, "matmul", b1, b2)
    if alpha != 1:
        prod = _op(c, "mul", prod, c.const(alpha, prod))
    if beta == 0:    # torch ignores input then, nan and inf included
        return prod
    if beta != 1:
        inp = _op(c, "mul", inp, c.const(beta, inp))
    return _op(c, "add", inp, prod)


@aten("aten.numpy_T.default")
def _numpy_t(c, a):
    """``x.T``: every dim reversed."""
    x = c.tensor(a["self"])
    r = x.type.rank
    return x if r < 2 else _op(c, "permute", x, dims=list(range(r - 1, -1, -1)))


# transformers 5.24's SigLIP2 resizes the position grid without a loop: the antialiased weights are
# computed from spatial_shapes with these ops (its checks use all and prod)

@aten("aten.all.default", "aten.all.dim", "aten.all.dims")
def _all(c, a):
    """``not any(not x)``, as ``_is_all_true`` in importer_current."""
    x = c.tensor(a["self"])
    if x.type.dtype != torch.bool:
        x = _op(c, "ne", x, c.const(0, x))
    dim = a.get("dim")
    dims = None if dim is None else ([dim] if isinstance(dim, int) else list(dim))
    return _op(c, "logical_not", _op(c, "any", _op(c, "logical_not", x), dims=dims, keepdim=bool(a.get("keepdim", False))))


@aten("aten.prod.dim_int")
def _prod_dim(c, a):
    """The product along a dim of fixed size: its slices multiplied in order (as a sequential product)."""
    x = c.tensor(a["self"])
    if a.get("dtype") is not None and a["dtype"] != x.type.dtype:
        x = _op(c, "cast", x, dtype=a["dtype"])
    elif not x.type.dtype.is_floating_point and x.type.dtype != torch.int64:
        x = _op(c, "cast", x, dtype=torch.int64)      # torch multiplies bool and small ints as int64
    d = a["dim"] % x.type.rank
    n = x.type.shape[d]
    if not isinstance(n, int):
        _unsupported(c, f"prod along a symbolic dim ({n})", "TS861")
    if n == 0:
        shape = [1 if k == d else s for k, s in enumerate(x.type.shape)] if a.get("keepdim") else             [s for k, s in enumerate(x.type.shape) if k != d]
        return _op(c, "full", shape=shape, value=1, dtype=x.type.dtype)
    y = _op(c, "slice", x, dim=d, start=0, end=1, step=1)
    for i in range(1, n):
        y = _op(c, "mul", y, _op(c, "slice", x, dim=d, start=i, end=i + 1, step=1))
    if a.get("keepdim"):
        return y
    return _op(c, "reshape", y, shape=[s for k, s in enumerate(x.type.shape) if k != d])
