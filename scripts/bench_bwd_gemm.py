"""The backward matmuls of a linear layer, Truestar's contraction kernel against cuBLAS (torch.mm).

    PYTHONPATH=src python scripts/bench_bwd_gemm.py

For y = x W^T with x [N, K] and W [M, K] (torch.nn.Linear), the backward computes
dX = dY W ([N, M] x [M, K], W read untransposed) and dW = dY^T x ([M, N] x [N, K], both read
transposed: the sum runs over the token dim). Each is compiled as its own one op program and timed
with CUDA events (median of 30) next to torch.mm on the same layouts; float32, TF32 off. The shapes
are the 94M decoder of scripts/bench_train_llama.py at batch 8, sequence 512 (N = 4096 tokens).
"""

from __future__ import annotations

import json
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from truestar import ir  # noqa: E402
from truestar.compiler import compile_program  # noqa: E402


def timed(fn, reps=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


class DX(nn.Module):
    def forward(self, g, w):
        return g @ w


class DW(nn.Module):
    def forward(self, g, x):
        return g.t() @ x


class FWD(nn.Module):
    def forward(self, x, w):
        return x @ w.t()


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    N = 4096
    rows = []
    for M, K, what in ((1024, 1024, "attn proj"), (2816, 1024, "gate/up"), (1024, 2816, "down"),
                       (8192, 1024, "head")):
        for name, mod, a_shape, b_shape in (
                ("fwd x W^T", FWD(), (N, K), (M, K)),
                ("dX = dY W", DX(), (N, M), (M, K)),
                ("dW = dY^T x", DW(), (N, M), (N, K))):
            a = torch.randn(a_shape, device="cuda")
            b = torch.randn(b_shape, device="cuda")
            prog = ir.import_module(mod, (a.cpu()[:2] if False else a.cpu(), b.cpu()))
            cp = compile_program(prog, "cuda", "float32")
            out = cp([a, b])[0]
            ref = mod(a, b)
            ts = timed(lambda: cp([a, b]))
            tt = timed(lambda: mod(a, b))
            fl = 2 * out.numel() * (a_shape[1] if name != "dW = dY^T x" else N)
            rows.append({"layer": what, "gemm": name, "M_N_K": [M, N, K], "truestar_ms": round(ts, 4),
                         "cublas_ms": round(tt, 4), "ratio": round(ts / tt, 2),
                         "truestar_tflops": round(fl / ts / 1e9, 2), "cublas_tflops": round(fl / tt / 1e9, 2),
                         "rel_err": float((out - ref).norm() / ref.norm())})
            print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
