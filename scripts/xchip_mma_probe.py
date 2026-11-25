"""Which tensor core instruction Triton 3.7 emits on this GPU for tl.dot, and how its accumulator
rounds: one large product plus 15 small ones of size 2**-k. An exact float32 sum keeps some of the
small ones; a tensor core that aligns products to the largest exponent and drops the bits below its
window returns 1.0 once 2**-k falls outside it.

    PYTHONPATH=src python scripts/xchip_mma_probe.py --json out/mma_probe.json
"""
import argparse
import json
import re

import torch
import triton
import triton.language as tl


@triton.jit
def _dot(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, PREC: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    if PREC == "tf32":
        c = tl.dot(a, b, input_precision="tf32")
    else:
        c = tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


def instr(ptx: str):
    found = sorted(set(re.findall(r"\b(wgmma\.mma_async[\w.]*|mma\.sync\.aligned\.m\d+n\d+k\d+[\w.]*)", ptx)))
    return found[:6]


def probe(dtype: str, M: int, N: int, K: int, warps: int):
    td = {"float16": torch.float16, "bfloat16": torch.bfloat16, "tf32": torch.float32}[dtype]
    res = {}
    ptx = None
    for k in range(18, 34):
        a = torch.zeros(M, K, dtype=torch.float32)
        b = torch.zeros(K, N, dtype=torch.float32)
        # row 0 of a times column 0 of b: product 1.0 plus 15 products 2**-k (each factor 2**-(k/2))
        a[0, 0], b[0, 0] = 1.0, 1.0
        h1, h2 = k // 2, k - k // 2
        a[0, 1:16] = 2.0 ** -h1
        b[1:16, 0] = 2.0 ** -h2
        # row 1: the large product last instead of first
        a[1, 15], b[15, 1] = 1.0, 1.0
        a[1, 0:15] = 2.0 ** -h1
        b[0:15, 1] = 2.0 ** -h2
        a, b = a.to("cuda", td), b.to("cuda", td)
        c = torch.empty(M, N, device="cuda", dtype=torch.float32)
        h = _dot[(1,)](a, b, c, M, N, K, dtype, num_warps=warps)
        if ptx is None:
            ptx = h.asm.get("ptx", "")
        v0, v1 = float(c[0, 0]), float(c[1, 1])
        res[k] = {"first": (v0 - 1.0) * 2.0 ** 23, "last": (v1 - 1.0) * 2.0 ** 23}
    return {"dtype": dtype, "M": M, "N": N, "K": K, "num_warps": warps, "instructions": instr(ptx or ""),
            "excess_over_1_in_float32_ulps_by_k": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    p = torch.cuda.get_device_properties(0)
    doc = {"gpu": p.name, "capability": f"{p.major}.{p.minor}", "torch": torch.__version__,
           "triton": triton.__version__, "probes": []}
    for dtype in ("float16", "bfloat16", "tf32"):
        for (M, N, K, w) in ((16, 16, 16, 1), (16, 64, 16, 2), (128, 128, 16, 4)):
            r = probe(dtype, M, N, K, w)
            doc["probes"].append(r)
            exact = {k: v for k, v in r["excess_over_1_in_float32_ulps_by_k"].items()}
            print(dtype, M, N, K, w, r["instructions"],
                  " ".join(f"k{k}:{v['first']:g}/{v['last']:g}" for k, v in exact.items()), flush=True)
    import os

    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)


if __name__ == "__main__":
    main()
