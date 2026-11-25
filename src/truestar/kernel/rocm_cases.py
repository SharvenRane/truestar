"""Representative kernels of every GPU loop shape, for AMD builds and the hardware validation.

``cases()`` gives (name, kernel, input shapes, build options). Each GPU loop shape (tile pointwise,
tile single pass and looped reductions, contraction in every operand class with and without split
K, attention, conv and window) appears at least once, in the dtypes the defaults treat apart. Used
by tests/test_rocm.py, scripts/rocm_compile_matrix.py and scripts/rocm_validate.py.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from . import ir
from . import ir as _ir  # noqa: F401  (K style names below)
from .graph import Graph
from .ir import Load, axis, buffer, kernel, output, reduce_max, reduce_sum, relu, strided, sym


def _pointwise(dt):
    g = Graph()
    x = g.input("x", (sym("M"), 4096), dt)
    (_, k), = g.fuse({"y": x * g.sigmoid(x)})
    return k, {"x": (64, 4096)}


def _softmax(dt):
    g = Graph()
    x = g.input("x", (sym("M"), 1000), dt)
    (_, k), = g.fuse({"y": g.softmax(x)})
    return k, {"x": (128, 1000)}


def _row_sum_long(dt):
    g = Graph()
    x = g.input("x", (sym("M"), 16384), dt)
    (_, k), = g.fuse({"y": g.sum(x, 1)})
    return k, {"x": (32, 16384)}


def _linear(dt, M, N, Kd):
    g = Graph()
    x = g.input("x", (sym("M"), Kd), dt)
    w = g.input("w", (N, Kd), dt)
    b = g.input("b", (N,), dt)
    (_, k), = g.fuse({"y": g.relu(x @ g.transpose(w, 0, 1) + b)})
    return k, {"x": (M, Kd), "w": (N, Kd), "b": (N,)}


def _attention(dt, D, S=1024, causal=True):
    g = Graph()
    B, H = 2, 8
    q, kk, v = (g.input(n, (B, H, sym("S"), D), dt) for n in "qkv")
    s = (q @ kk.T) * D ** -0.5
    if causal:
        s = g.pointwise(lambda s, b, h, i, j: ir.where(j <= i, s, ir.NEG_INF), s, with_index=True)
    (_, k), = g.fuse({"o": g.softmax(s) @ v})
    return k, {n: (B, H, S, D) for n in "qkv"}


def _conv(dt, C=64, O=64, H=28, W=28, stride=1, pad=1):
    N = sym("N")
    KH = KW = 3
    OH = (H + 2 * pad - (KH - 1) - 1) // stride + 1
    OW = (W + 2 * pad - (KW - 1) - 1) // stride + 1
    x, w = buffer("x", (N, C, H, W), dt), buffer("w", (O, C, KH, KW), dt)
    n, o, i, j = axis("n", N), axis("o", O), axis("i", OH), axis("j", OW)
    c, u, v = axis("c", C), axis("u", KH), axis("v", KW)
    xl = Load(x, (n, c, strided([(i, stride), (u, 1)], -pad), strided([(j, stride), (v, 1)], -pad)), pad=0.0)
    e = reduce_sum(c, reduce_sum(u, reduce_sum(v, xl * w[o, c, u, v])))
    return kernel("conv3x3", [output("y", (n, o, i, j), relu(e), dt)]), {"x": (8, C, H, W), "w": (O, C, KH, KW)}


def _maxpool(dt):
    N = sym("N")
    x = buffer("x", (N, 64, 56, 56), dt)
    n, c, i, j, u, v = (axis(a, s) for a, s in (("n", N), ("c", 64), ("i", 28), ("j", 28), ("u", 3), ("v", 3)))
    ld = Load(x, (n, c, strided([(i, 2), (u, 1)], -1), strided([(j, 2), (v, 1)], -1)), pad=-math.inf)
    return kernel("maxpool", [output("y", (n, c, i, j), reduce_max(u, reduce_max(v, ld)), dt)]), {"x": (8, 64, 56, 56)}


def cases() -> List[Tuple[str, object, Dict[str, tuple], Dict]]:
    out = []

    def add(name, built, **kw):
        k, shapes = built
        out.append((name, k, shapes, kw))

    for dt in ("float32", "float16"):
        add(f"pointwise_{dt}", _pointwise(dt))
        add(f"softmax_{dt}", _softmax(dt))
    add("row_sum_16384_float32", _row_sum_long("float32"))
    add("linear_1024_float32", _linear("float32", 1024, 1024, 1024))
    add("linear_1024_tf32", _linear("float32", 1024, 1024, 1024), allow_tf32=True)
    add("linear_1024_float16", _linear("float16", 1024, 1024, 1024))
    add("linear_1024_bfloat16", _linear("bfloat16", 1024, 1024, 1024))
    add("linear_4096_float16", _linear("float16", 4096, 4096, 4096))
    add("linear_m16_k8192_float32", _linear("float32", 16, 1024, 8192))            # split K by default
    add("linear_m16_k8192_float32_det", _linear("float32", 16, 1024, 8192), deterministic=True)
    add("attention_causal_d64_float16", _attention("float16", 64))
    add("attention_causal_d128_float16", _attention("float16", 128))
    add("attention_d64_float32", _attention("float32", 64, causal=False))
    add("conv3x3_float16", _conv("float16"))
    add("conv3x3_float32", _conv("float32"))
    add("conv3x3_s2_float16", _conv("float16", stride=2))
    add("maxpool3x3_s2_float32", _maxpool("float32"))
    return out
