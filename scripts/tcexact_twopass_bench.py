"""Exact tensor core blocks in two passes: digitize each operand once (int8 digit planes and one
exponent per row or column per K block), then an int8 tensor core matmul over the planes. Same
digits, same sums, same rounding order as the one pass kernel (``_tsx_xdot``), so the bits must be
equal; the question is speed, against float32 FMA and plain tensor cores.

    PYTHONPATH=src python scripts/tcexact_twopass_bench.py --json out/twopass.json
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np
import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from truestar.kernel import tcexact  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tcexact_bench as one  # noqa: E402

SRC = '''
import triton
import triton.language as tl
''' + tcexact.TRITON_SOURCE + '''

@triton.jit
def digitize_rows(x_ptr, d_ptr, e_ptr, R, K, BR: tl.constexpr, KB: tl.constexpr, TF32: tl.constexpr):
    # x [R, K]: digits d [3, R, K] int8, exponents e [R, K // KB] (one per row per K block)
    rr = tl.program_id(0) * BR + tl.arange(0, BR)
    pk = tl.program_id(1)
    rk = pk * KB + tl.arange(0, KB)
    x = tl.load(x_ptr + rr[:, None] * K + rk[None, :]).to(tl.float32)
    if TF32:
        x = _tsx_tf32(x)
    e = _tsx_exp(tl.max(tl.abs(x), axis=1) * 1.0078125)
    d0, d1, d2 = _tsx_digits(x * _tsx_p2(7 - e)[:, None])
    off = rr[:, None] * K + rk[None, :]
    tl.store(d_ptr + off, d0.to(tl.int8))
    tl.store(d_ptr + R * K + off, d1.to(tl.int8))
    tl.store(d_ptr + 2 * R * K + off, d2.to(tl.int8))
    tl.store(e_ptr + rr * (K // KB) + pk, e)


@triton.jit
def digitize_cols(x_ptr, d_ptr, e_ptr, K, N, BC: tl.constexpr, KB: tl.constexpr, TF32: tl.constexpr):
    # x [K, N]: digits d [3, K, N] int8, exponents e [K // KB, N] (one per column per K block)
    rc = tl.program_id(0) * BC + tl.arange(0, BC)
    pk = tl.program_id(1)
    rk = pk * KB + tl.arange(0, KB)
    x = tl.load(x_ptr + rk[:, None] * N + rc[None, :]).to(tl.float32)
    if TF32:
        x = _tsx_tf32(x)
    e = _tsx_exp(tl.max(tl.abs(x), axis=0) * 1.0078125)
    d0, d1, d2 = _tsx_digits(x * _tsx_p2(7 - e)[None, :])
    off = rk[:, None] * N + rc[None, :]
    tl.store(d_ptr + off, d0.to(tl.int8))
    tl.store(d_ptr + K * N + off, d1.to(tl.int8))
    tl.store(d_ptr + 2 * K * N + off, d2.to(tl.int8))
    tl.store(e_ptr + pk * N + rc, e)


@triton.jit
def mm_digits(da, ea, db, eb, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    nkb = K // KB
    acc = tl.zeros((BM, BN), tl.float32)
    for kb in range(0, nkb):
        rk = kb * KB + tl.arange(0, KB)
        oa = rm[:, None] * K + rk[None, :]
        ob = rk[:, None] * N + rn[None, :]
        a0 = tl.load(da + oa)
        a1 = tl.load(da + M * K + oa)
        a2 = tl.load(da + 2 * M * K + oa)
        b0 = tl.load(db + ob)
        b1 = tl.load(db + K * N + ob)
        b2 = tl.load(db + 2 * K * N + ob)
        xa = tl.load(ea + rm * nkb + kb)
        xb = tl.load(eb + kb * N + rn)
        i0 = tl.dot(a0, b0, out_dtype=tl.int32)
        i1 = tl.dot(a0, b1, out_dtype=tl.int32)
        i1 = tl.dot(a1, b0, i1, out_dtype=tl.int32)
        i2 = tl.dot(a0, b2, out_dtype=tl.int32)
        i2 = tl.dot(a1, b1, i2, out_dtype=tl.int32)
        i2 = tl.dot(a2, b0, i2, out_dtype=tl.int32)
        s = _tsx_f32(i2) * 0.0078125 + _tsx_f32(i1)
        s = s * 0.0078125 + _tsx_f32(i0)
        acc = acc + s * _tsx_p2(xa - 7)[:, None] * _tsx_p2(xb - 7)[None, :]
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)
'''

KB = tcexact.PORTABLE_BK
MM_CFGS = [(128, 128, 4, 3), (128, 128, 8, 3), (128, 256, 8, 3), (64, 128, 4, 3), (128, 64, 4, 4),
           (64, 64, 4, 4)]


def load():
    d = tempfile.mkdtemp(prefix="tsx2_")
    path = os.path.join(d, "tsx_twopass_kernels.py")
    open(path, "w", encoding="utf-8").write(SRC)
    spec = importlib.util.spec_from_file_location("tsx_twopass_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Planes:
    def __init__(self, M, N, K):
        dev = "cuda"
        self.da = torch.empty(3, M, K, device=dev, dtype=torch.int8)
        self.ea = torch.empty(M, K // KB, device=dev, dtype=torch.int32)
        self.db = torch.empty(3, K, N, device=dev, dtype=torch.int8)
        self.eb = torch.empty(K // KB, N, device=dev, dtype=torch.int32)
        self.c = torch.empty(M, N, device=dev, dtype=torch.float32)


def digitize(mod, a, b, P, tf32):
    M, K = a.shape
    N = b.shape[1]
    mod.digitize_rows[(M // 32, K // KB)](a, P.da, P.ea, M, K, BR=32, KB=KB, TF32=tf32, num_warps=4)
    mod.digitize_cols[(N // 32, K // KB)](b, P.db, P.eb, K, N, BC=32, KB=KB, TF32=tf32, num_warps=4)


def matmul(mod, P, M, N, K, cfg):
    bm, bn, w, st = cfg
    mod.mm_digits[(M // bm, N // bn)](P.da, P.ea, P.db, P.eb, P.c, M, N, K, BM=bm, BN=bn, KB=KB,
                                      num_warps=w, num_stages=st)


def best(fn, cfgs):
    out = None
    for cfg in cfgs:
        try:
            fn(cfg)
            torch.cuda.synchronize()
            ms = triton.testing.do_bench(lambda: fn(cfg), warmup=10, rep=60)
        except Exception as e:
            print("  skip", cfg, type(e).__name__, str(e)[:100], flush=True)
            continue
        if out is None or ms < out[0]:
            out = (ms, cfg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    mod = load()
    one_mod = one.load_kernels()
    p = torch.cuda.get_device_properties(0)
    doc = {"gpu": p.name, "capability": f"{p.major}.{p.minor}", "bits": [], "speed": []}
    print(doc["gpu"], flush=True)
    for dtype in ("float16", "bfloat16", "tf32"):
        M, N, K = 512, 256, 1024
        a, b = one.inputs(M, N, K, dtype, 2)
        ag, bg = a.cuda(), b.cuda()
        P = Planes(M, N, K)
        digitize(mod, ag, bg, P, dtype == "tf32")
        matmul(mod, P, M, N, K, (64, 64, 4, 2))
        ref = one.run(one_mod, "exact", ag, bg, dtype, (64, 64, 64, 4, 2))
        emu = tcexact.emulate_matmul(a.float().numpy(), b.float().numpy(), KB, dtype == "tf32")
        row = {"dtype": dtype, "shape": [M, N, K], "sha": one.sha(P.c),
               "equals_one_pass": bool(torch.equal(P.c, ref)),
               "equals_emulation": bool(np.array_equal(P.c.cpu().numpy().view(np.int32), emu.view(np.int32)))}
        doc["bits"].append(row)
        print("bits", row, flush=True)
    for (M, N, K) in ((2048, 2048, 2048), (4096, 4096, 4096), (512, 4096, 1024), (2048, 1024, 4096)):
        a32, b32 = one.inputs(M, N, K, "float32", 11)
        ag32, bg32 = a32.cuda(), b32.cuda()
        fma = one.bench(one_mod, "plain", ag32, bg32, "float32", one.FMA_CFGS)
        for dtype in ("float16", "tf32"):
            ag, bg = ag32.to(one.DT[dtype]), bg32.to(one.DT[dtype])
            plain = one.bench(one_mod, "plain", ag, bg, dtype, one.PLAIN_CFGS)
            P = Planes(M, N, K)
            tf = dtype == "tf32"
            dig = triton.testing.do_bench(lambda: digitize(mod, ag, bg, P, tf), warmup=10, rep=60)
            mm = best(lambda cfg: matmul(mod, P, M, N, K, cfg), MM_CFGS)
            fl = 2.0 * M * N * K
            row = {"dtype": dtype, "shape": [M, N, K], "fp32_fma_ms": fma[0], "plain_tc_ms": plain[0],
                   "digitize_ms": dig, "int8_matmul_ms": mm[0], "int8_matmul_cfg": mm[1],
                   "two_pass_ms": dig + mm[0],
                   "fma_tflops": fl / fma[0] / 1e9, "plain_tflops": fl / plain[0] / 1e9,
                   "two_pass_tflops": fl / (dig + mm[0]) / 1e9, "matmul_only_tflops": fl / mm[0] / 1e9,
                   "fma_over_two_pass": fma[0] / (dig + mm[0])}
            doc["speed"].append(row)
            print("speed", json.dumps(row), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    json.dump(doc, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
