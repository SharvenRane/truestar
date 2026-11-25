"""Do the two exact block engines (int8 and fp16 digit dot products) give the exact integer sums on
this GPU? One 64 x 64 x KB block: the three weight sums of each engine against NumPy int64.

    PYTHONPATH=src python scripts/tcexact_engine_probe.py --json out/engine_probe.json
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
from truestar.kernel import tcexact as X  # noqa: E402

SRC = '''
import triton
import triton.language as tl
''' + X.TRITON_SOURCE + '''

@triton.jit
def sums(a_ptr, b_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, KB: tl.constexpr, INT8: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, KB)
    a = tl.load(a_ptr + rm[:, None] * KB + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    ea = _tsx_exp(tl.max(tl.abs(a), axis=1) * 1.0078125)
    eb = _tsx_exp(tl.max(tl.abs(b), axis=0) * 1.0078125)
    a0, a1, a2 = _tsx_digits(a * _tsx_p2(7 - ea)[:, None])
    b0, b1, b2 = _tsx_digits(b * _tsx_p2(7 - eb)[None, :])
    if INT8:
        f0 = _tsx_f32(tl.dot(a0.to(tl.int8), b0.to(tl.int8), out_dtype=tl.int32))
        i1 = tl.dot(a0.to(tl.int8), b1.to(tl.int8), out_dtype=tl.int32)
        f1 = _tsx_f32(tl.dot(a1.to(tl.int8), b0.to(tl.int8), i1, out_dtype=tl.int32))
        i2 = tl.dot(a0.to(tl.int8), b2.to(tl.int8), out_dtype=tl.int32)
        i2 = tl.dot(a1.to(tl.int8), b1.to(tl.int8), i2, out_dtype=tl.int32)
        f2 = _tsx_f32(tl.dot(a2.to(tl.int8), b0.to(tl.int8), i2, out_dtype=tl.int32))
    else:
        f0 = tl.dot(a0.to(tl.float16), b0.to(tl.float16))
        f1 = tl.dot(a0.to(tl.float16), b1.to(tl.float16))
        f1 = tl.dot(a1.to(tl.float16), b0.to(tl.float16), f1)
        f2 = tl.dot(a0.to(tl.float16), b2.to(tl.float16))
        f2 = tl.dot(a1.to(tl.float16), b1.to(tl.float16), f2)
        f2 = tl.dot(a2.to(tl.float16), b0.to(tl.float16), f2)
    o = o_ptr + rm[:, None] * N + rn[None, :]
    tl.store(o, f0)
    tl.store(o + M * N, f1)
    tl.store(o + 2 * M * N, f2)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    d = tempfile.mkdtemp()
    path = os.path.join(d, "tsx_probe_kernels.py")
    open(path, "w").write(SRC)
    spec = importlib.util.spec_from_file_location("tsx_probe_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = torch.cuda.get_device_properties(0)
    out = {"gpu": p.name, "capability": f"{p.major}.{p.minor}", "cases": []}
    rs = np.random.RandomState(5)
    M = N = 64
    for kb in (32, 64):
        a = torch.from_numpy(rs.standard_normal((M, kb)).astype(np.float32)).half().float()
        b = torch.from_numpy(rs.standard_normal((kb, N)).astype(np.float32)).half().float()
        da, _ = X.block_digits(a.numpy(), 1)
        db, _ = X.block_digits(b.numpy(), 0)
        want = [da[0] @ db[0], da[0] @ db[1] + da[1] @ db[0], da[0] @ db[2] + da[1] @ db[1] + da[2] @ db[0]]
        for int8 in (True, False):
            o = torch.empty(4, M, N, device="cuda")
            h = mod.sums[(1,)](a.cuda(), b.cuda(), o, M, N, kb, int8, num_warps=4)
            got = o.cpu().numpy().astype(np.int64)
            row = {"kb": kb, "int8": int8,
                   "equal": [bool(np.array_equal(got[w], want[w])) for w in range(3)],
                   "max_abs_diff": [int(np.abs(got[w] - want[w]).max()) for w in range(3)],
                   "count_diff": [int((got[w] != want[w]).sum()) for w in range(3)],
                   "instructions": sorted(set(__import__("re").findall(
                       r"\b(wgmma\.mma_async[\w.]*|mma\.sync\.aligned\.m\d+n\d+k\d+[\w.]*)",
                       h.asm.get("ptx", ""))))}
            out["cases"].append(row)
            print(row, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
