"""Kernel rules for convolution and pooling (and the gradients of both).

Each rule writes the textbook formula with strided load indices (docs/KERNEL_IR.md, strided
loads), so the kernels are verified against the float64 reference like every other kernel:

* ``conv2d``: ``y[n, o, i, j] = b[o] + sum over c, u, v of
  x[n, g(o) * Cg + c, i*s + u*d - p, j*s + v*d - p] * w[o, c, u, v]`` with zero padding as the
  load's pad value. On the GPU a chain of sums like this is an implicit GEMM, or a window loop for
  a depthwise or grouped convolution; whatever elementwise chain reads the result (bias, a folded
  batch norm, relu, a residual add) is its epilogue.
* ``conv2d_input_grad``: the transposed convolution in gather form,
  ``x_bar[n, c, h, w] = sum over o, u, v of g[n, o, (h + p - u*d) / s, ...] * w[o, c, u, v]``,
  where the division must be exact. No zero inserted copy of the gradient is ever made.
* ``conv2d_weight_grad``: ``w_bar[o, c, u, v] = sum over n, i, j of
  x[n, g(o) * Cg + c, i*s + u*d - p, ...] * g[n, o, i, j]``.
* pooling: ``max_pool2d`` (pad minus infinity), ``avg_pool2d`` (a divisor per output cell computed
  once at compile time), ``adaptive_avg_pool2d`` (window start ``(i * H) // OH``, per cell weights
  computed once) and ``max_pool2d_grad`` (each input cell gathers the cotangent of every window
  whose first maximum it is, the tie rule of the reference).

The operand read at strided indices is always a buffer (``Lowering.strided_operand``), because a
pad value belongs to a stored tensor, not to an expression.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch

from .. import kernel as K
from ..kernel import ir as kir
from .lower import NotLowerable, rule


def _pair(v):
    if isinstance(v, int):
        return v, v
    v = list(v)
    return (v[0], v[0]) if len(v) == 1 else (int(v[0]), int(v[1]))


def _ints(what: str, dims) -> List[int]:
    if not all(isinstance(d, int) for d in dims):
        raise NotLowerable(f"{what} must be fixed sizes for a kernel, got {list(map(str, dims))}")
    return list(dims)


def aff(parts: Sequence[Tuple[object, int]], offset: int = 0, exact: int = 1):
    """An index ``sum of coeff * entry + offset`` (divided exactly by ``exact``), where each entry is
    an index the caller received: an Axis, an int, or a strided index without division."""
    terms, off = [], offset
    for e, c in parts:
        if c == 0:
            continue
        if isinstance(e, kir.Axis):
            terms.append(kir.Term(e, c))
        elif isinstance(e, int):
            off += c * e
        elif isinstance(e, kir.Strided) and e.exact == 1:
            terms += [kir.Term(t.axis, t.coeff * c, t.mul, t.div, t.mod) for t in e.terms]
            off += c * e.offset
        else:
            raise NotLowerable("a strided index of a strided index with exact division")
    if exact == 1 and not terms:
        return off
    return K.strided(terms, off, exact)


def group_of(e, size_per_group: int, scale: int):
    """``(e // size_per_group) * scale`` for an index entry e (an Axis or an int)."""
    if size_per_group == 1:
        return aff([(e, scale)])
    if isinstance(e, int):
        return (e // size_per_group) * scale
    if isinstance(e, kir.Axis):
        return K.strided([kir.Term(e, scale, 1, size_per_group)])
    raise NotLowerable("a grouped index of a strided index")


def _plus(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    parts = []
    for e in (a, b):
        parts.append((e, 1))
    return aff(parts)


def _member_of_group(e, groups_size: int):
    """``e % groups_size`` for an index entry."""
    if groups_size == 1:
        return 0
    if isinstance(e, int):
        return e % groups_size
    if isinstance(e, kir.Axis):
        return K.strided([kir.Term(e, 1, 1, 1, groups_size)])
    raise NotLowerable("a grouped index of a strided index")


def _geometry(node):
    a = node.attrs
    return (_pair(a.get("stride", 1)), _pair(a.get("padding", 0)), _pair(a.get("dilation", 1)),
            int(a.get("groups", 1)))


def _chain(L, lid: int, sizes, body_fn):
    """sum over axes of the given sizes (outermost first) of body_fn(*axes)."""
    axes = [K.axis(f"r{lid}_{k}", s) for k, s in enumerate(sizes)]
    e = body_fn(*axes)
    for a in reversed(axes):
        e = kir.Reduce("sum", a, e)
    return e


def _conv_lv(L, node, x, w, b, xshape, wshape, out_shape, geo, widx=None):
    (sh, sw), (ph, pw), (dh, dw), groups = geo
    O, Cg, KH, KW = _ints("the conv2d weight shape", wshape)
    widx = widx or (lambda o, c, u, v: (o, c, u, v))
    OCg = O // groups
    lid = len(L.lvs)
    xcl = L.opts.device == "cuda" and groups == 1 and channels_last_input(xshape, (sh, sw), KH * KW)
    if xcl:
        x = _relayout(L, x, (0, 2, 3, 1), node)
    xat = (lambda n, c, h, w_: (n, h, w_, c)) if xcl else (lambda n, c, h, w_: (n, c, h, w_))

    def fn(idx, lo):
        n, o, i, j = idx

        def body(c, u, v):
            ch = c if groups == 1 else _plus(group_of(o, OCg, Cg), c)
            hi = aff([(i, sh), (u, dh)], -ph)
            wi = aff([(j, sw), (v, dw)], -pw)
            return lo(x, xat(n, ch, hi, wi), 0.0) * lo(w, widx(o, c, u, v))

        y = _chain(L, lid, (Cg, KH, KW), body)
        return y if b is None else y + lo(b, (o,))

    lv = L.new(out_shape, "f", fn, (x, w) + (() if b is None else (b,)), heavy=True, node=node)
    lv.contract = True
    return lv


SMALL_IMAGE = 16 * 16     # pixels per channel at or below which a GEMM operand image is read channels last


def channels_last_input(shape, stride, taps: int) -> bool:
    """True when a conv reads its input from a channels last copy: small images (small_image),
    and strided convs with a spatial kernel on 128 or more channels, whose tile rows are strided
    reads (measured, batch 32: resnet18's 128 channel 3 x 3 stride 2 conv 0.22 to 0.17 ms; on 64
    channels no gain, and on the 3 channel stem a loss, 0.41 to 0.45 ms)."""
    if small_image(shape):
        return True
    return (len(shape) == 4 and all(isinstance(d, int) for d in shape) and max(stride) > 1 and taps > 1
            and shape[1] >= 128)


def small_image(shape) -> bool:
    """True for an [N, C, H, W] operand with H x W <= SMALL_IMAGE and fixed sizes. The implicit
    GEMM's A tile is [pixels, channels]; read from NCHW, a tile row run is one image row, so for
    7 x 7 images a 64 row tile spans nine short runs and its loads coalesce badly. A copy as
    [N, H, W, C] (cheap at that size: 3.2 MB for resnet18's last layer at batch 32) makes the tile
    contiguous along channels. Measured on the RTX 5070 Ti, batch 32: resnet18's 512 channel 7 x 7
    forward conv 0.46 ms read in place."""
    return (len(shape) == 4 and all(isinstance(d, int) for d in shape)
            and shape[2] * shape[3] <= SMALL_IMAGE)


def _gemm_weight(L, wv, node, n_axis: int = 0):
    """The weight of a conv as the implicit GEMM's B operand, with the GEMM's N dim contiguous.
    n_axis is the weight dim that is the GEMM's N (0, output channels, for the forward conv;
    1, input channels, for the input gradient). Returns (lowered value, index map from
    (o, c, u, v) to its index).

    A constant weight is stored once, at compile time, in that order (the same values, the
    logical shape [O, C, kH, kW] unchanged). A weight computed at run time (training: the weight
    is a step input) gets a relayout copy [kH, kW, K, N] made by its own small kernel once per
    call, so the GEMM reads its [K, N] tile along contiguous memory instead of a stride of
    C x kH x kW (measured on resnet18 layers at batch 32: 2 to 3.9x slower forward convolutions
    with the weight read in place)."""
    data = L.prog.param_data.get(wv.name)
    w = L.operand(wv)
    same = lambda o, c, u, v: (o, c, u, v)  # noqa: E731
    if len(w.shape) != 4 or not all(isinstance(s, int) for s in w.shape):
        return w, same
    k_axis = 1 - n_axis
    if data is not None and L.is_param(wv) and data.dim() == 4:
        w64 = data.double()
        perm = (2, 3, k_axis, n_axis)
        inv = [perm.index(d) for d in range(4)]
        stored = w64.permute(*perm).contiguous().permute(*inv)
        return L.derived(f"{node.output.name}.weight{n_axis}", stored), same
    if L.opts.device != "cuda" or int(node.attrs.get("groups", 1)) != 1:
        return w, same           # a grouped conv takes the window shape, which reads w per element
    O, C, KH, KW = w.shape

    def fn(idx, lo):
        u, v, k, n = idx
        oc = (n, k) if n_axis == 0 else (k, n)
        return lo(w, oc + (u, v))

    shape = (KH, KW) + ((C, O) if n_axis == 0 else (O, C))
    t = L.new(shape, "f", fn, (w,), node=node)
    t.force_mat = True       # stored once, read by every program of the GEMM
    if n_axis == 0:
        return t, lambda o, c, u, v: (u, v, c, o)
    return t, lambda o, c, u, v: (u, v, o, c)


@rule("conv2d")
def _conv2d(L, node):
    xv, wv, bv = node.inputs
    x = L.strided_operand(xv)
    w, widx = _gemm_weight(L, wv, node)
    b = None if bv is None else L.operand(bv)
    if len(x.shape) != 4:
        raise NotLowerable("conv2d needs an [N, C, H, W] input")
    _ints("conv2d input channels and spatial sizes", x.shape[1:])
    out = L.shape_of(node.output)
    _ints("conv2d output spatial sizes", out[2:])
    return _conv_lv(L, node, x, w, b, x.shape, L.shape_of(wv), out, _geometry(node), widx)


@rule("conv2d_input_grad")
def _conv2d_input_grad(L, node):
    """x_bar[n, c, h, w] = sum over oo, u, v of g[n, grp(c) * OCg + oo, (h + p - u d) / s, ...]
    * w[grp(c) * OCg + oo, c % Cg, u, v], the division exact (else that term is zero)."""
    gv, wv = node.inputs
    g = L.strided_operand(gv)
    w, widx = _gemm_weight(L, wv, node, n_axis=1)
    (sh, sw), (ph, pw), (dh, dw), groups = _geometry(node)
    O, Cg, KH, KW = _ints("the conv2d weight shape", L.shape_of(wv))
    OCg = O // groups
    _ints("conv2d gradient spatial sizes", g.shape[2:])
    out = L.shape_of(node.output)
    _ints("conv2d input sizes", out[1:])
    if (L.opts.device == "cuda" and groups == 1 and (sh, sw) == (2, 2) and (dh, dw) == (1, 1)
            and out[2] % 2 == 0 and out[3] % 2 == 0 and min(KH, KW) >= 2):
        g, gat = _cl_operand(L, g, node)
        return _input_grad_by_parity(L, node, g, gat, w, widx, out, O, KH, KW, ph, pw)
    if groups == 1:
        g, gat = _cl_operand(L, g, node)
    else:
        gat = lambda n, o, i, j: (n, o, i, j)  # noqa: E731
    lid = len(L.lvs)

    def fn(idx, lo):
        n, c, h, ww = idx

        def body(oo, u, v):
            o = oo if groups == 1 else _plus(group_of(c, Cg, OCg), oo)
            hi = aff([(h, 1), (u, -dh)], ph, sh)
            wi = aff([(ww, 1), (v, -dw)], pw, sw)
            cw = c if groups == 1 else _member_of_group(c, Cg)
            return lo(g, gat(n, o, hi, wi), 0.0) * lo(w, widx(o, cw, u, v))

        return _chain(L, lid, (OCg, KH, KW), body)

    lv = L.new(out, "f", fn, (g, w), heavy=True, node=node)
    lv.contract = True
    return lv


def _input_grad_by_parity(L, node, g, gat, w, widx, out, O, KH, KW, ph, pw):
    """The stride 2 input gradient split by output parity (sub pixel form). Output row
    h = 2 h2 + r reads only the kernel rows u with u = q (mod 2), q = (r + p) mod 2, and so on for
    columns. With the tap residues (qh, qw) as axes of their own, u = q + 2 t and the cotangent row
    (h + p - u) / 2 is an exact affine index, so each (qh, qw) is a batch of one implicit GEMM over
    (o, th, tw) with ceil(k / 2) taps per dim: 4 taps instead of 9 for a 3 x 3 kernel, and none of
    the masked out taps the gather form loads (three of four, at stride 2). The result is stored as
    [N, C, H / 2, 2, W / 2, 2] by residue and read back at (h // 2, residue of h % 2): a view for
    every consumer, which fuses into it. Same sum, same terms: only the zero terms are dropped, and
    the order over (o, t) is fixed by the shapes."""
    N, C, H, W = out
    TH, TW = -(-KH // 2), -(-KW // 2)
    lid = len(L.lvs)

    def part(idx, lo):
        n, c, h2, qh, w2, qw = idx

        def row(h2_, q, t, p):
            # (2 h2 + r + p - q - 2 t) / 2 with r = (q - p) mod 2, an integer
            if p % 2 == 0:
                return aff([(h2_, 1), (t, -1)], p // 2)
            return aff([(h2_, 1), (q, -1), (t, -1)], (p + 1) // 2)

        def body(o, th, tw):
            u = aff([(qh, 1), (th, 2)])
            v = aff([(qw, 1), (tw, 2)])
            return lo(g, gat(n, o, row(h2, qh, th, ph), row(w2, qw, tw, pw)), 0.0) * lo(w, widx(o, c, u, v), 0.0)

        return _chain(L, lid, (O, TH, TW), body)

    pt = L.new((N, C, H // 2, 2, W // 2, 2), "f", part, (g, w), heavy=True, node=node)
    pt.contract = True
    pt.force_mat = True

    def res(e, p):
        # residue q of output index e: q = (e % 2 + p) % 2, so e % 2 when p is even, else 1 - e % 2
        if p % 2 == 0:
            return K.strided([kir.Term(e, 1, 1, 1, 2)])
        return K.strided([kir.Term(e, -1, 1, 1, 2)], 1)

    def view(idx, lo):
        n, c, h, ww = idx
        return lo(pt, (n, c, K.strided([kir.Term(h, 1, 1, 2)]), res(h, ph),
                       K.strided([kir.Term(ww, 1, 1, 2)]), res(ww, pw)))

    return L.new(out, "f", view, (pt,), node=node)


@rule("conv2d_weight_grad")
def _conv2d_weight_grad(L, node):
    """w_bar[o, c, u, v] = sum over n, i, j of x[n, grp(o) * Cg + c, i s + u d - p, ...] * g[n, o, i, j]."""
    xv, gv = node.inputs
    (sh, sw), (ph, pw), (dh, dw), groups = _geometry(node)
    out = L.shape_of(node.output)
    O, Cg, KH, KW = _ints("the conv2d weight shape", out)
    if (L.opts.device == "cuda" and groups == 1
            and all(isinstance(d, int) for d in L.shape_of(xv) + L.shape_of(gv))):
        return _weight_grad_channels_last(L, node, xv, gv, out)
    x = L.strided_operand(xv)
    g = L.operand(gv)
    OCg = O // groups
    N = g.shape[0]
    OH, OW = _ints("conv2d gradient spatial sizes", g.shape[2:])
    lid = len(L.lvs)
    S = weight_grad_splits(O * Cg * KH * KW, OH, OW) if L.opts.device == "cuda" else 1
    if S > 1:
        return _weight_grad_split(L, node, x, gv, S, (sh, sw), (ph, pw), (dh, dw), groups, out, OCg, Cg, N, OH, OW)

    def fn(idx, lo):
        o, c, u, v = idx

        def body(n, i, j):
            ch = c if groups == 1 else _plus(group_of(o, OCg, Cg), c)
            hi = aff([(i, sh), (u, dh)], -ph)
            wi = aff([(j, sw), (v, dw)], -pw)
            return lo(x, (n, ch, hi, wi), 0.0) * lo(g, (n, o, i, j))

        return _chain(L, lid, (N, OH, OW), body)

    lv = L.new(out, "f", fn, (x, g), heavy=True, node=node)
    lv.contract = True
    return lv


WG_TILE = 64 * 64        # output entries one contraction program covers (the largest tile)
WG_PROGRAMS = 560        # programs wanted: eight per streaming multiprocessor of the RTX 5070 Ti (measured: 140 left 128 channel layers 1.4x slower)
WG_MIN_ROWS_ELEMS = 64   # do not split below this many output pixels per chunk (per image)
WG_FEW = 140             # fewer programs than this (two per SM) after the row split: split images too
WG_FEW_TARGET = 280      # up to this many programs
WG_MAX_PARTS = 32        # and at most this many partial sums: each output adds them one after the other, and a
                         # 1 x 1 layer cut into 112 parts spent 0.10 ms in that sum, more than the GEMM (0.03 ms)


def weight_grad_splits(outputs: int, OH: int, OW: int) -> int:
    """How many row chunks the weight gradient's sum over (batch, rows, columns) is cut into.

    The weight gradient has only O x C x KH x KW outputs (36864 for a 64 channel 3 x 3 conv: 9
    programs of 64 x 64) and sums over every image and pixel, so one kernel leaves most of the GPU
    idle. Cutting the output rows into S chunks gives S partial sums, each its own program set, and
    a second kernel adds them in the fixed order 0, 1, ..., S - 1. S depends only on the shapes, so
    the summation order is the same on every call and with every launch config (no atomics): the
    result is deterministic and a recompute is bitwise equal. S is the smallest divisor of OH that
    reaches WG_PROGRAMS programs, keeping at least WG_MIN_ROWS_ELEMS pixels per chunk."""
    tiles = max(1, -(-outputs // WG_TILE))
    if tiles >= WG_PROGRAMS:
        return 1
    best = 1
    for d in range(2, OH + 1):
        if OH % d or (OH // d) * OW < WG_MIN_ROWS_ELEMS:
            continue
        best = d
        if tiles * d >= WG_PROGRAMS:
            break
    return best


def _cl_operand(L, g, node):
    """The input gradient's A operand (the cotangent, [pixels, out channels] tiles): a channels
    last copy for small images (small_image), else as it is. Returns (value, index map from
    (n, o, i, j))."""
    if L.opts.device == "cuda" and small_image(g.shape):
        return _relayout(L, g, (0, 2, 3, 1), node), (lambda n, o, i, j: (n, i, j, o))
    return g, (lambda n, o, i, j: (n, o, i, j))


def _relayout(L, v, perm, node):
    """A copy of v with its dims in the order perm (stored, by its own kernel). One copy
    per (value, order): the input and weight gradients of a conv share their cotangent's."""
    memo = L.__dict__.setdefault("_conv_relayouts", {})
    key = (v.id, tuple(perm))
    t = memo.get(key)
    if t is not None and t.id < len(L.lvs) and L.lvs[t.id] is t:   # not undone with a failed rule
        return t
    shape = tuple(v.shape[d] for d in perm)
    inv = [perm.index(d) for d in range(len(perm))]

    def fn(idx, lo):
        return lo(v, tuple(idx[inv[d]] for d in range(len(perm))))

    t = L.new(shape, "f", fn, (v,), node=node)
    t.force_mat = True
    memo[key] = t
    return t


def channels_last_splits(outputs: int, N: int, OH: int, OW: int) -> Tuple[int, int]:
    """(image chunks, row chunks) of the channels last weight gradient's sum over (n, rows, cols):
    rows first, as weight_grad_splits; then, when that leaves fewer than WG_FEW programs of
    WG_TILE outputs, images too, up to WG_FEW_TARGET programs and WG_MAX_PARTS partial sums.
    Shapes only, so the order of the sum is fixed (see weight_grad_splits)."""
    tiles = max(1, -(-outputs // WG_TILE))
    sr = weight_grad_splits(outputs, OH, OW)
    sn = 1
    if tiles * sr >= WG_FEW:
        return sn, sr
    for d in range(2, N + 1):
        if tiles * sr * sn >= WG_FEW_TARGET or sr * d > WG_MAX_PARTS:
            break
        if N % d == 0:
            sn = d
    return sn, sr


def _weight_grad_channels_last(L, node, xv, gv, out):
    """The weight gradient as an implicit GEMM whose B operand (N = o) is a copy of the cotangent
    as [N, OH, OW, O], contiguous along N, which a float32 tl.dot needs: on the RTX 5070 Ti a
    4608 x 512 x 1568 GEMM with the B tile contiguous along K took 0.52 ms, 0.29 ms contiguous
    along N. M is (c, u, v), read from x in place; for spatial kernels on 128 or more channels, M is
    (u, v, c) read from a copy of x as [N, H, W, C] instead, contiguous along M (0.27 ms on that
    GEMM): measured on resnet18's layers at batch 32, the copy pays for itself there (layer4 3 x 3:
    0.45 to 0.36 ms) and not on wide, shallow inputs (the 7 x 7 stem: 0.82 to 2.06 ms).

    The sum over (n, i, j) is cut into (image chunks x row chunks) partial sums
    (channels_last_splits), stored and added in the fixed order of the chunk index, so the
    result does not depend on launch configs and a recompute is bitwise equal."""
    (sh, sw), (ph, pw), (dh, dw), _ = _geometry(node)
    O, C, KH, KW = out
    x_cl = KH * KW > 1 and C >= 128
    x = _relayout(L, L.operand(xv), (0, 2, 3, 1), node) if x_cl else L.strided_operand(xv)
    gt = _relayout(L, L.operand(gv), (0, 2, 3, 1), node)
    N, OH, OW = gt.shape[0], gt.shape[1], gt.shape[2]
    Sn, Sr = channels_last_splits(O * C * KH * KW, N, OH, OW)
    NB, R = N // Sn, OH // Sr
    lid = len(L.lvs)

    def part(idx, lo):
        if x_cl:
            a, b, u, v, c, o = idx
        else:
            a, b, o, c, u, v = idx

        def body(nn, r, j):
            n = aff([(a, NB), (nn, 1)])
            i = aff([(b, R), (r, 1)])
            hi = aff([(i, sh), (u, dh)], -ph)
            wi = aff([(j, sw), (v, dw)], -pw)
            xs = lo(x, (n, hi, wi, c), 0.0) if x_cl else lo(x, (n, c, hi, wi), 0.0)
            return xs * lo(gt, (n, i, j, o), 0.0)

        return _chain(L, lid, (NB, R, OW), body)

    pshape = (Sn, Sr) + ((KH, KW, C, O) if x_cl else (O, C, KH, KW))
    p = L.new(pshape, "f", part, (x, gt), heavy=True, node=node)
    p.contract = True
    p.force_mat = True        # read by the ordered sum below: stored, never recomputed per reader

    def total(idx, lo):
        o, c, u, v = idx
        acc = None
        for a in range(Sn):
            for b in range(Sr):
                t = lo(p, (a, b) + ((u, v, c, o) if x_cl else (o, c, u, v)))
                acc = t if acc is None else acc + t
        return acc

    return L.new(tuple(out), "f", total, (p,), node=node)


def _weight_grad_split(L, node, x, gv, S, stride, pad, dil, groups, out, OCg, Cg, N, OH, OW):
    """The split form of ``conv2d_weight_grad`` (see ``weight_grad_splits``): partial sums over row
    chunks of R = OH / S rows, materialized as [S, O, C, KH, KW], then added in order."""
    (sh, sw), (ph, pw), (dh, dw) = stride, pad, dil
    g = L.strided_operand(gv)
    R = OH // S
    lid = len(L.lvs)

    def part(idx, lo):
        s, o, c, u, v = idx

        def body(n, r, j):
            ch = c if groups == 1 else _plus(group_of(o, OCg, Cg), c)
            hi = aff([(s, R * sh), (r, sh), (u, dh)], -ph)
            wi = aff([(j, sw), (v, dw)], -pw)
            return lo(x, (n, ch, hi, wi), 0.0) * lo(g, (n, o, aff([(s, R), (r, 1)]), j), 0.0)

        return _chain(L, lid, (N, R, OW), body)

    p = L.new((S,) + tuple(out), "f", part, (x, g), heavy=True, node=node)
    p.contract = True
    p.force_mat = True        # read S times by the sum below: stored, never recomputed per reader

    def total(idx, lo):
        acc = lo(p, (0,) + tuple(idx))
        for k in range(1, S):
            acc = acc + lo(p, (k,) + tuple(idx))
        return acc

    return L.new(out, "f", total, (p,), node=node)


def _pool_attrs(node):
    a = node.attrs
    k = _pair(a["kernel_size"])
    s = _pair(a.get("stride") or a["kernel_size"])
    p = _pair(a.get("padding", 0))
    d = _pair(a.get("dilation", 1))
    return k, s, p, d


@rule("max_pool2d")
def _max_pool2d(L, node):
    x = L.strided_operand(node.inputs[0])
    (kh, kw), (sh, sw), (ph, pw), (dh, dw) = _pool_attrs(node)
    out = L.shape_of(node.output)
    _ints("max_pool2d spatial sizes", tuple(x.shape[-2:]) + tuple(out[-2:]))
    lid = len(L.lvs)

    def fn(idx, lo):
        *lead, i, j = idx
        u, v = K.axis(f"r{lid}u", kh), K.axis(f"r{lid}v", kw)
        hi = aff([(i, sh), (u, dh)], -ph)
        wi = aff([(j, sw), (v, dw)], -pw)
        return kir.Reduce("max", u, kir.Reduce("max", v, lo(x, tuple(lead) + (hi, wi), -math.inf)))

    return L.new(out, "f", fn, (x,), heavy=True, node=node)


@rule("avg_pool2d")
def _avg_pool2d(L, node):
    """Sum over the window of the zero padded input, divided by a per cell divisor computed once
    at compile time with the reference's formula."""
    from ..ir import reference as iref
    x = L.strided_operand(node.inputs[0])
    a = node.attrs
    kh, kw = _pair(a["kernel_size"])
    sh, sw = _pair(a.get("stride") or a["kernel_size"])
    ph, pw = _pair(a.get("padding", 0))
    out = L.shape_of(node.output)
    H, W, OH, OW = _ints("avg_pool2d spatial sizes", tuple(x.shape[-2:]) + tuple(out[-2:]))
    # the divisor of each cell: the reference pool of a tensor of ones with the sum replaced by a count
    ones = torch.ones((1, 1, H, W), dtype=torch.float64)
    mean_of_ones = iref.IMPLS["avg_pool2d"](dict(a), [ones], (1, 1, OH, OW))[0, 0]
    total = iref.IMPLS["avg_pool2d"](dict(a, divisor_override=1), [ones], (1, 1, OH, OW))[0, 0]
    div = torch.where(mean_of_ones != 0, total / mean_of_ones, torch.ones_like(total))
    if a.get("divisor_override"):
        div = torch.full_like(div, float(a["divisor_override"]))
    d_lv = L.derived(f"{node.output.name}.divisor", div)
    lid = len(L.lvs)

    def fn(idx, lo):
        *lead, i, j = idx
        u, v = K.axis(f"r{lid}u", kh), K.axis(f"r{lid}v", kw)
        hi = aff([(i, sh), (u, 1)], -ph)
        wi = aff([(j, sw), (v, 1)], -pw)
        s = kir.Reduce("sum", u, kir.Reduce("sum", v, lo(x, tuple(lead) + (hi, wi), 0.0)))
        return s / lo(d_lv, (i, j))

    return L.new(out, "f", fn, (x, d_lv), heavy=True, node=node)


def _adaptive_weights(size: int, out: int):
    """(window length bound, weights[out, length]): weight 1/len inside cell i's window, 0 beyond."""
    starts = [(i * size) // out for i in range(out)]
    ends = [-(-((i + 1) * size) // out) for i in range(out)]
    length = max(e - s for s, e in zip(starts, ends))
    wgt = torch.zeros((out, length), dtype=torch.float64)
    for i, (s, e) in enumerate(zip(starts, ends)):
        wgt[i, : e - s] = 1.0 / (e - s)
    return length, wgt


@rule("adaptive_avg_pool2d")
def _adaptive_avg_pool2d(L, node):
    """Output cell i averages input rows (i H) // OH to ceil((i + 1) H / OH): a window that starts at
    the strided index (i * H) // OH, with per row weights 1 / length inside it and 0 beyond."""
    x = L.strided_operand(node.inputs[0])
    out = L.shape_of(node.output)
    H, W, OH, OW = _ints("adaptive_avg_pool2d spatial sizes", tuple(x.shape[-2:]) + tuple(out[-2:]))
    lh, rw = _adaptive_weights(H, OH)
    lw, cw = _adaptive_weights(W, OW)
    r_lv = L.derived(f"{node.output.name}.rows", rw)
    c_lv = L.derived(f"{node.output.name}.cols", cw)
    lid = len(L.lvs)

    def start(e, size, n):
        if isinstance(e, int):
            return (e * size) // n
        if isinstance(e, kir.Axis):
            return kir.Term(e, 1, size, n)
        raise NotLowerable("adaptive pooling of a strided index")

    def fn(idx, lo):
        *lead, i, j = idx
        u, v = K.axis(f"r{lid}u", lh), K.axis(f"r{lid}v", lw)
        si, sj = start(i, H, OH), start(j, W, OW)
        hi = K.strided(([si] if isinstance(si, kir.Term) else []) + [kir.Term(u)],
                       0 if isinstance(si, kir.Term) else si)
        wi = K.strided(([sj] if isinstance(sj, kir.Term) else []) + [kir.Term(v)],
                       0 if isinstance(sj, kir.Term) else sj)
        # weight 0 past a cell's window; the where keeps an infinity there out of the sum
        cwt, rwt = lo(c_lv, (j, v)), lo(r_lv, (i, u))
        xv = lo(x, tuple(lead) + (hi, wi), 0.0)
        inner = kir.Reduce("sum", v, K.where(cwt > 0.0, xv * cwt, 0.0))
        return kir.Reduce("sum", u, K.where(rwt > 0.0, inner * rwt, 0.0))

    return L.new(out, "f", fn, (x, r_lv, c_lv), heavy=True, node=node)


@rule("max_pool2d_grad")
def _max_pool2d_grad(L, node):
    """x_bar[n, c, h, w] = sum over taps (u, v) whose window (i, j) = ((h + p - u d) / s, ...) is
    exact and in range, of g[n, c, i, j] where x[n, c, h, w] is that window's first maximum in row
    major order (the window's own taps are scanned with the same strided reads as the forward)."""
    xv, gv = node.inputs
    x = L.strided_operand(xv)
    g = L.strided_operand(gv)
    (kh, kw), (sh, sw), (ph, pw), (dh, dw) = _pool_attrs(node)
    H, W = _ints("max_pool2d input sizes", x.shape[-2:])
    OH, OW = _ints("max_pool2d output sizes", g.shape[-2:])
    out = L.shape_of(node.output)
    lid = len(L.lvs)
    BIG = float(kh * kw)

    def fn(idx, lo):
        *lead, h, w = idx
        lead = tuple(lead)
        u, v = K.axis(f"r{lid}u", kh), K.axis(f"r{lid}v", kw)
        a, b = K.axis(f"r{lid}a", kh), K.axis(f"r{lid}b", kw)
        # window (i, j) that reads (h, w) at tap (u, v); i = (h + p - u d) / s exactly
        i = aff([(h, 1), (u, -dh)], ph, sh)
        j = aff([(w, 1), (v, -dw)], pw, sw)
        if not isinstance(i, kir.Strided) or not isinstance(j, kir.Strided):
            raise NotLowerable("max_pool2d_grad needs strided window indices")
        # the window's taps (a, b): x[(i s + a d - p), ...] = x[h + (a - u) d, w + (b - v) d]
        tap_h = aff([(h, 1), (a, dh), (u, -dh)])
        tap_w = aff([(w, 1), (b, dw), (v, -dw)])
        # a tap index is valid only when the window itself is (else the first max is meaningless,
        # and the cotangent read below is the pad 0 anyway)
        xt = lo(x, lead + (tap_h, tap_w), -math.inf)
        # the window's max, then the first tap (row major) that equals it
        mx = kir.Reduce("max", a, kir.Reduce("max", b, xt))
        a2, b2 = K.axis(f"r{lid}c", kh), K.axis(f"r{lid}d", kw)
        xt2 = lo(x, lead + (aff([(h, 1), (a2, dh), (u, -dh)]), aff([(w, 1), (b2, dw), (v, -dw)])),
                 -math.inf)
        first = kir.Reduce("min", a2, kir.Reduce("min", b2, K.where(
            K.eq(xt2, mx), K.index(a2) * float(kw) + K.index(b2), BIG)))
        mine = K.eq(first, K.index(u) * float(kw) + K.index(v))
        gij = lo(g, lead + (i, j), 0.0)
        return kir.Reduce("sum", u, kir.Reduce("sum", v, K.where(mine, gij, 0.0)))

    return L.new(out, "f", fn, (x, g), heavy=True, node=node)
