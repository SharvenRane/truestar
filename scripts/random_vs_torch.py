"""Evidence for docs/RANDOM.md: keyed random numbers against PyTorch on this machine.

1. Bitwise agreement: N draws of each distribution from the Triton kernel (CUDA), the numba kernel
   (CPU) and the torch implementation the reference interpreter uses (CPU), compared in chunks; and,
   for contrast, PyTorch with the same manual_seed on CPU and on CUDA.
2. Throughput on CUDA for 1e8 float32 draws: Truestar's Triton kernel against torch.rand, torch.randn,
   bernoulli_ and F.dropout.

Usage: PYTHONPATH=src python scripts/random_vs_torch.py [N] [reference N]
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

from truestar import rng

CHUNK = 10_000_000


def bits(t: torch.Tensor) -> torch.Tensor:
    t = t.cpu()
    if t.dtype.is_floating_point:
        return t.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])
    return t


def agreement(n: int, n_ref: int) -> None:
    cases = [("uniform", torch.float32, {}), ("normal", torch.float32, {}), ("normal", torch.float64, {}),
             ("bernoulli", torch.bool, {"p": 0.9}), ("randint", torch.int64, {"low": 0, "high": 50257})]
    print(f"bitwise agreement, key 2026, site 'evidence' ({n:.0e} draws; reference path {n_ref:.0e})")
    for dist, dtype, kw in cases:
        t0 = time.perf_counter()
        diff_cpu = diff_ref = 0
        for start in range(0, n, CHUNK):
            m = min(CHUNK, n - start)
            g = rng.generate(dist, (m,), dtype, 2026, "evidence", device="cuda", backend="triton", start=start, **kw)
            c = rng.generate(dist, (m,), dtype, 2026, "evidence", device="cpu", backend="numba", start=start, **kw)
            diff_cpu += int((bits(g) != bits(c)).sum())
            if start < n_ref:
                r = rng.generate(dist, (min(m, n_ref - start),), dtype, 2026, "evidence", device="cpu",
                                 backend="torch", start=start, **kw)
                diff_ref += int((bits(g)[:r.numel()] != bits(r)).sum())
        name = f"{dist} {str(dtype).replace('torch.', '')}"
        print(f"  {name:18s} Triton vs numba: {diff_cpu} differing of {n}; Triton vs reference: {diff_ref} "
              f"differing of {min(n, n_ref)} ({time.perf_counter() - t0:.0f} s)")
    m = 1_000_000
    print(f"PyTorch, torch.manual_seed(0) then the same call on CPU and on CUDA ({m:.0e} draws):")
    for name, fn in [("rand", lambda d: torch.rand(m, device=d)), ("randn", lambda d: torch.randn(m, device=d)),
                     ("dropout mask p=0.1", lambda d: F.dropout(torch.ones(m, device=d), 0.1) != 0),
                     ("randint", lambda d: torch.randint(0, 50257, (m,), device=d))]:
        torch.manual_seed(0)
        a = fn("cpu")
        torch.manual_seed(0)
        b = fn("cuda").cpu()
        print(f"  {name:18s} {int((bits(a) != bits(b)).sum())} differing of {m}")


def bench(fn, reps: int = 10) -> float:
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def throughput() -> None:
    n = 100_000_000
    x = torch.randn(n // 4, 4, device="cuda")
    rows = [
        ("uniform f32", lambda: rng.generate("uniform", (n,), torch.float32, 1, "s", device="cuda", backend="triton"),
         lambda: torch.rand(n, device="cuda")),
        ("normal f32", lambda: rng.generate("normal", (n,), torch.float32, 1, "s", device="cuda", backend="triton"),
         lambda: torch.randn(n, device="cuda")),
        ("normal f64", lambda: rng.generate("normal", (n,), torch.float64, 1, "s", device="cuda", backend="triton"),
         lambda: torch.randn(n, device="cuda", dtype=torch.float64)),
        ("bernoulli mask", lambda: rng.generate("bernoulli", (n,), torch.bool, 1, "s", p=0.9, device="cuda",
                                                backend="triton"),
         lambda: torch.empty(n, device="cuda").bernoulli_(0.9)),
        ("dropout p=0.1", lambda: rng.dropout(x, 0.1, 1, "d", backend="triton"), lambda: F.dropout(x, 0.1)),
    ]
    print(f"CUDA throughput, {n:.0e} elements, ms per call (best effort, shared GPU):")
    for name, ours, theirs in rows:
        a, b = bench(ours), bench(theirs)
        print(f"  {name:15s} Truestar {a:7.2f}  torch {b:7.2f}  ratio {a / b:5.2f}")


if __name__ == "__main__":
    n = int(float(sys.argv[1])) if len(sys.argv) > 1 else 100_000_000
    n_ref = int(float(sys.argv[2])) if len(sys.argv) > 2 else 10_000_000
    agreement(n, n_ref)
    if torch.cuda.is_available():
        throughput()
