"""Exact tensor core blocks (src/truestar/kernel/tcexact.py) against plain tensor cores and float32
FMA, as standalone Triton matmuls: bits, accuracy against float64 and speed.

    PYTHONPATH=src python scripts/tcexact_bench.py --json out/tcexact_bench.json [--quick]

Bits: inputs come from NumPy with fixed seeds (the same values on every machine); the hash of each
portable output is printed, and a small case is compared with the NumPy emulation on this host.
Accuracy: error over |A| |B| (elementwise) against a float64 matmul of the same input values.
Speed: best of a few tile configs per kernel, triton.testing.do_bench median, TFLOPS of 2 M N K.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from truestar.kernel import tcexact  # noqa: E402

KERNELS = '''
import triton
import triton.language as tl
from triton.language.extra import libdevice

''' + tcexact.TRITON_SOURCE + '''

@triton.jit
def mm_plain(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             PREC: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        if PREC == 1:
            acc = tl.dot(a, b, acc, input_precision="tf32")
        elif PREC == 2:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


@triton.jit
def mm_exact(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             TF32: tl.constexpr, INT8: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        acc = acc + _tsx_xdot_e(a, b, TF32, INT8)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)
'''

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "tf32": torch.float32,
      "float32": torch.float32}


def load_kernels():
    d = tempfile.mkdtemp(prefix="tsx_")
    path = os.path.join(d, "tsx_bench_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(KERNELS)
    spec = importlib.util.spec_from_file_location("tsx_bench_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def inputs(M, N, K, dtype, seed):
    rs = np.random.RandomState(seed)
    a = torch.from_numpy(rs.standard_normal((M, K)).astype(np.float32)).to(DT[dtype])
    b = torch.from_numpy(rs.standard_normal((K, N)).astype(np.float32)).to(DT[dtype])
    return a, b


def run(mod, kind, a, b, dtype, cfg, int8=True):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    bm, bn, bk, w, st = cfg
    grid = (M // bm, N // bn)
    if kind == "exact":
        mod.mm_exact[grid](a, b, c, M, N, K, BM=bm, BN=bn, BK=bk, TF32=dtype == "tf32", INT8=int8,
                           num_warps=w, num_stages=st)
    else:
        prec = {"tf32": 1, "float32": 2}.get(dtype, 0)
        mod.mm_plain[grid](a, b, c, M, N, K, BM=bm, BN=bn, BK=bk, PREC=prec, num_warps=w,
                           num_stages=st)
    return c


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


PLAIN_CFGS = [(128, 128, 32, 4, 3), (128, 128, 64, 8, 3), (128, 256, 32, 8, 3), (64, 128, 32, 4, 4),
              (64, 64, 64, 4, 3)]
FMA_CFGS = [(64, 64, 16, 4, 3), (128, 64, 16, 4, 3), (128, 128, 16, 8, 3), (64, 64, 32, 4, 2)]
EXACT_CFGS = [(64, 64, 64, 4, 2), (64, 64, 64, 4, 3), (128, 64, 64, 4, 2), (128, 64, 64, 8, 2),
              (64, 128, 64, 4, 2), (128, 128, 64, 8, 2), (128, 128, 64, 4, 2), (128, 128, 64, 8, 1)]
EXACT_BK128 = [(64, 128, 128, 4, 2), (64, 64, 128, 4, 2), (128, 64, 128, 8, 2)]


def bench(mod, kind, a, b, dtype, cfgs, int8=True):
    best = None
    for cfg in cfgs:
        try:
            run(mod, kind, a, b, dtype, cfg, int8)
            torch.cuda.synchronize()
            ms = triton.testing.do_bench(lambda: run(mod, kind, a, b, dtype, cfg, int8), warmup=10, rep=60)
        except Exception as e:  # a config that does not fit this GPU (shared memory, registers)
            print(f"  {kind} {dtype} {cfg}: {type(e).__name__} {str(e)[:120]}", flush=True)
            continue
        if best is None or ms < best[0]:
            best = (ms, cfg)
    return best


def accuracy(c, a, b):
    a64, b64 = a.double(), b.double()
    if c is None:
        return None
    ref = a64 @ b64
    den = a64.abs() @ b64.abs()
    e = ((c.double() - ref).abs() / den.clamp_min(1e-300))
    return {"max": float(e.max()), "mean": float(e.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    p = torch.cuda.get_device_properties(0)
    tcexact.check_device()
    mod = load_kernels()
    doc = {"gpu": p.name, "capability": f"{p.major}.{p.minor}", "torch": torch.__version__,
           "triton": triton.__version__, "bits": [], "accuracy": [], "speed": []}
    print(doc["gpu"], doc["capability"], flush=True)
    dtypes = ("float16", "bfloat16", "tf32")
    # bits: a small case against the NumPy emulation, then fixed size hashes
    for dtype in dtypes:
        for (M, N, K, seed) in ((128, 128, 320, 1), (256, 128, 1024, 2), (512, 512, 512, 3)):
            a, b = inputs(M, N, K, dtype, seed)
            c = run(mod, "exact", a.cuda(), b.cuda(), dtype, (64, 64, 64, 4, 2))
            c2 = run(mod, "exact", a.cuda(), b.cuda(), dtype, (128, 64, 64, 8, 1))
            c3 = run(mod, "exact", a.cuda(), b.cuda(), dtype, (64, 64, 64, 4, 2), int8=False)
            emu = tcexact.emulate_matmul(a.float().numpy(), b.float().numpy(), 64, dtype == "tf32")
            row = {"dtype": dtype, "shape": [M, N, K], "seed": seed, "sha": sha(c),
                   "config_free": bool(torch.equal(c, c2)), "engines_equal": bool(torch.equal(c, c3)),
                   "emulation_equal": bool(np.array_equal(c.cpu().numpy().view(np.int32),
                                                          emu.view(np.int32)))}
            doc["bits"].append(row)
            print("bits", row, flush=True)
    # accuracy against float64 at 1024 and 4096 deep
    for dtype in dtypes:
        for (M, N, K) in ((256, 256, 1024), (256, 256, 4096)):
            a, b = inputs(M, N, K, dtype, 7)
            ag, bg = a.cuda(), b.cuda()
            av = ag.float()   # tf32 rows: against the float32 values the program holds
            bv = bg.float()
            plain = run(mod, "plain", ag, bg, dtype, (64, 64, 32, 4, 3))
            exact = run(mod, "exact", ag, bg, dtype, (64, 64, 64, 4, 2))
            fma = run(mod, "plain", av.contiguous(), bv.contiguous(), "float32", (64, 64, 16, 4, 3))
            row = {"dtype": dtype, "shape": [M, N, K], "plain_tc": accuracy(plain, av, bv),
                   "exact_tc": accuracy(exact, av, bv), "fp32_fma": accuracy(fma, av, bv)}
            doc["accuracy"].append(row)
            print("accuracy", row, flush=True)
    shapes = [(4096, 4096, 4096)] if args.quick else [(2048, 2048, 2048), (4096, 4096, 4096),
                                                      (512, 4096, 1024), (2048, 1024, 4096)]
    for (M, N, K) in shapes:
        a32, b32 = inputs(M, N, K, "float32", 11)
        ag32, bg32 = a32.cuda(), b32.cuda()
        fma = bench(mod, "plain", ag32, bg32, "float32", FMA_CFGS)
        for dtype in dtypes:
            ag, bg = ag32.to(DT[dtype]), bg32.to(DT[dtype])
            plain = bench(mod, "plain", ag, bg, dtype, PLAIN_CFGS)
            exact = bench(mod, "exact", ag, bg, dtype, EXACT_CFGS)
            ex16 = bench(mod, "exact", ag, bg, dtype, EXACT_CFGS[:4], int8=False)
            ex128 = bench(mod, "exact", ag, bg, dtype, EXACT_BK128)
            fl = 2.0 * M * N * K
            row = {"dtype": dtype, "shape": [M, N, K],
                   "fp32_fma_ms": fma[0], "fp32_fma_cfg": fma[1],
                   "plain_tc_ms": plain[0], "plain_tc_cfg": plain[1],
                   "exact_tc_ms": exact[0] if exact else None, "exact_tc_cfg": exact[1] if exact else None}
            if ex128:
                row["exact_bk128_ms"] = ex128[0]
                row["exact_bk128_cfg"] = ex128[1]
            if ex16:
                row["exact_fp16_engine_ms"] = ex16[0]
                row["exact_fp16_engine_cfg"] = ex16[1]
            if exact:
                row["exact_over_plain"] = exact[0] / plain[0]
                row["fma_over_exact"] = fma[0] / exact[0]
                row["exact_tflops"] = fl / exact[0] / 1e9
            row["fma_tflops"] = fl / fma[0] / 1e9
            row["plain_tflops"] = fl / plain[0] / 1e9
            doc["speed"].append(row)
            print("speed", json.dumps(row), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    print("wrote", args.json)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"seconds {time.time() - t0:.0f}")
