"""Small GPU workloads built from kernel layer calls, with matching PyTorch versions.

Used by scripts/bench_overhead.py and tests/test_runtime.py so both measure the same programs.
Each builder returns (truestar_fn, torch_fn, inputs) where both functions take the same tensors
by name and return one tensor (or a dict for the Truestar graphs).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

import torch

from .. import kernel as K
from ..kernel import Graph

Workload = Tuple[Callable, Callable, Dict[str, torch.Tensor]]


def _single(g: Graph, out) -> Callable:
    cg = g.compile({"y": out}, "cuda")
    assert cg.num_kernels == 1, cg.describe()
    ck = cg.kernels[0][1]
    return ck


def gelu_residual(M: int, N: int, seed: int = 0) -> Workload:
    """Elementwise chain: tanh GELU of x plus a residual (eight ops, one fused kernel), float32."""
    g = Graph()
    x = g.input("x", (K.sym("M"), N))
    r = g.input("r", (K.sym("M"), N))
    y = 0.5 * x * (1.0 + g.tanh(0.7978845608 * (x + 0.044715 * x * x * x))) + r
    ck = _single(g, y)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ins = {"x": torch.randn(M, N, device="cuda", generator=gen),
           "r": torch.randn(M, N, device="cuda", generator=gen)}

    def ref(x, r):
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x))) + r
    return ck, ref, ins


def softmax(M: int, N: int, seed: int = 0) -> Workload:
    g = Graph()
    x = g.input("x", (K.sym("M"), N))
    ck = _single(g, g.softmax(x))
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ins = {"x": torch.randn(M, N, device="cuda", generator=gen)}
    return ck, (lambda x: torch.softmax(x, -1)), ins


def layer_norm(M: int, N: int, seed: int = 0) -> Workload:
    g = Graph()
    x = g.input("x", (K.sym("M"), N))
    w = g.input("w", (N,))
    b = g.input("b", (N,))
    ck = _single(g, g.layer_norm(x, w, b))
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ins = {"x": torch.randn(M, N, device="cuda", generator=gen),
           "w": torch.randn(N, device="cuda", generator=gen),
           "b": torch.randn(N, device="cuda", generator=gen)}
    return ck, (lambda x, w, b: torch.nn.functional.layer_norm(x, (N,), w, b)), ins


def attention(B: int, H: int, S: int, D: int, seed: int = 0) -> Workload:
    """Causal attention, float16, one fused kernel."""
    g = Graph()
    Sy = K.sym("S")
    q, k, v = (g.input(n, (B, H, Sy, D), "float16") for n in "qkv")
    s = (q @ k.T) * D ** -0.5
    s = g.pointwise(lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF), s, with_index=True)
    ck = _single(g, g.softmax(s) @ v)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ins = {n: torch.randn(B, H, S, D, device="cuda", generator=gen).half() for n in "qkv"}

    def ref(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    return ck, ref, ins


# ----------------------------------------------------------------------------- 50 kernel MLP


class MLPStack:
    """Pre norm MLP blocks chained by hand from kernel layer calls, five kernels per block:

        h = layer_norm(x) ; u = h @ W1 ; a = gelu_tanh(u + b1) ; o = a @ W2 ; x = x + o + b2

    Ten blocks give 50 launches per step. The five kernels are compiled once and shared by every
    block (same IR, same binary); each block has its own weights. `torch_step` is the same math
    in PyTorch eager (cuBLAS matmuls, unfused elementwise ops).
    """

    def __init__(self, rows: int, hidden: int, blocks: int = 10, seed: int = 0):
        self.rows, self.hidden, self.blocks = rows, hidden, blocks
        H, F = hidden, 4 * hidden
        M = K.sym("M")

        g = Graph()
        x, w, b = g.input("x", (M, H)), g.input("w", (H,)), g.input("b", (H,))
        self.k_ln = _single(g, g.layer_norm(x, w, b))

        g = Graph()
        a, w1 = g.input("a", (M, H)), g.input("w1", (H, F))
        self.k_up = _single(g, a @ w1)

        g = Graph()
        u, b1 = g.input("u", (M, F)), g.input("b1", (F,))
        z = u + b1
        self.k_act = _single(g, 0.5 * z * (1.0 + g.tanh(0.7978845608 * (z + 0.044715 * z * z * z))))

        g = Graph()
        a, w2 = g.input("a", (M, F)), g.input("w2", (F, H))
        self.k_down = _single(g, a @ w2)

        g = Graph()
        x, o, b2 = g.input("x", (M, H)), g.input("o", (M, H)), g.input("b2", (H,))
        self.k_res = _single(g, x + o + b2)

        gen = torch.Generator(device="cuda").manual_seed(seed)

        def r(*s, scale=1.0):
            return torch.randn(*s, device="cuda", generator=gen) * scale
        self.params = [dict(w=1.0 + 0.1 * r(H), b=0.1 * r(H), w1=r(H, F, scale=1 / math.sqrt(H)),
                            b1=0.1 * r(F), w2=r(F, H, scale=1 / math.sqrt(F)), b2=0.1 * r(H))
                       for _ in range(blocks)]
        self.x = r(rows, H)

    @property
    def kernels_per_step(self) -> int:
        return 5 * self.blocks

    def step(self, x):
        for p in self.params:
            h = self.k_ln(x=x, w=p["w"], b=p["b"])
            u = self.k_up(a=h, w1=p["w1"])
            a = self.k_act(u=u, b1=p["b1"])
            o = self.k_down(a=a, w2=p["w2"])
            x = self.k_res(x=x, o=o, b2=p["b2"])
        return x

    def torch_step(self, x):
        F_ = torch.nn.functional
        for p in self.params:
            h = F_.layer_norm(x, (self.hidden,), p["w"], p["b"])
            z = h @ p["w1"] + p["b1"]
            a = 0.5 * z * (1.0 + torch.tanh(0.7978845608 * (z + 0.044715 * z * z * z)))
            x = x + a @ p["w2"] + p["b2"]
        return x
