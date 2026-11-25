"""What low precision dot products does Triton run on this GPU, and are they native?

For each mode: compile a one tile Triton kernel, run it, compare with an exact float64 product of the
same decoded operands, and search the generated PTX for the MMA instruction it used. A mode whose
PTX holds no low precision MMA is reported as emulated (upcast), never as native.

    PYTHONPATH=src python scripts/probe_quant_triton.py
"""

from __future__ import annotations

import json
import re
import sys

import torch

import triton
import triton.language as tl


@triton.jit
def _fp8_dot(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.dot(a, b, out_dtype=tl.float32)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


@triton.jit
def _scaled_dot(a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr,
                K: tl.constexpr, G: tl.constexpr, FMT: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rkp = tl.arange(0, K // 2)
    rs = tl.arange(0, K // G)
    a = tl.load(a_ptr + rm[:, None] * (K // 2) + rkp[None, :])          # [M, K/2] packed
    b = tl.load(b_ptr + rkp[:, None] * N + rn[None, :])                  # [K/2, N] packed along K
    sa = tl.load(as_ptr + rm[:, None] * (K // G) + rs[None, :])
    sb = tl.load(bs_ptr + rn[:, None] * (K // G) + rs[None, :])
    c = tl.dot_scaled(a, sa, FMT, b, sb, FMT)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    t = torch.tensor(E2M1, dtype=torch.float64)
    mag = t[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -mag, mag)


def _mma_lines(ptx: str):
    return sorted({m.group(0) for m in re.finditer(r"\b(mma|wgmma|tcgen05)\.[\w.:]+", ptx)})


def probe_fp8(kind: str) -> dict:
    dt = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[kind]
    M = N = 32
    K = 64
    g = torch.Generator().manual_seed(0)
    a = (torch.randn(M, K, generator=g) * 2).to(dt).cuda()
    b = (torch.randn(K, N, generator=g) * 2).to(dt).cuda()
    c = torch.empty(M, N, device="cuda")
    out = {"mode": f"tl.dot {kind} x {kind}"}
    try:
        h = _fp8_dot[(1,)](a, b, c, M, N, K)
        ref = a.double() @ b.double()
        err = ((c.double() - ref).abs().max() / ref.abs().max()).item()
        mma = _mma_lines(h.asm["ptx"])
        out.update(ran=True, rel_err=err, mma=mma,
                   native=any(kind in m for m in mma))
    except Exception as e:  # report, never hide
        out.update(ran=False, error=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
    return out


def probe_fp4(scale_kind: str) -> dict:
    """MXFP4 (e8m0 scales, block 32) and NVFP4 (e4m3 scales, block 16) through tl.dot_scaled."""
    M = N = 32
    K = 128
    G = 32 if scale_kind == "e8m0" else 16
    g = torch.Generator().manual_seed(1)
    ca = torch.randint(0, 16, (M, K), generator=g, dtype=torch.uint8)
    cb = torch.randint(0, 16, (K, N), generator=g, dtype=torch.uint8)
    pa = ca[:, 0::2] | (ca[:, 1::2] << 4)          # first element in the low bits
    pb = cb[0::2, :] | (cb[1::2, :] << 4)
    out = {"mode": f"tl.dot_scaled e2m1, {scale_kind} scales, block {G}"}
    if scale_kind == "e8m0":
        sa = torch.randint(124, 130, (M, K // G), generator=g, dtype=torch.uint8)
        sb = torch.randint(124, 130, (N, K // G), generator=g, dtype=torch.uint8)
        fa = torch.pow(2.0, sa.double() - 127)
        fb = torch.pow(2.0, sb.double() - 127)
        sa_d, sb_d = sa.cuda(), sb.cuda()
    else:
        va = (torch.rand(M, K // G, generator=g) + 0.5).to(torch.float8_e4m3fn)
        vb = (torch.rand(N, K // G, generator=g) + 0.5).to(torch.float8_e4m3fn)
        fa, fb = va.double(), vb.double()
        sa_d, sb_d = va.cuda(), vb.cuda()
    c = torch.empty(M, N, device="cuda")
    try:
        h = _scaled_dot[(1,)](pa.cuda(), sa_d, pb.cuda(), sb_d, c, M, N, K, G, "e2m1")
        A = _decode_e2m1(ca) * fa.repeat_interleave(G, dim=1)
        B = _decode_e2m1(cb) * fb.repeat_interleave(G, dim=1).T
        ref = A @ B
        err = ((c.double().cpu() - ref).abs().max() / ref.abs().max()).item()
        mma = _mma_lines(h.asm["ptx"])
        out.update(ran=True, rel_err=err, mma=mma,
                   native=any(("e2m1" in m or "mxf4" in m or "block_scale" in m) for m in mma))
    except Exception as e:
        out.update(ran=False, error=f"{type(e).__name__}: {str(e).splitlines()[0][:300]}")
    return out


def main() -> int:
    res = {"torch": torch.__version__, "triton": triton.__version__,
           "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
           "probes": [probe_fp8("e4m3"), probe_fp8("e5m2"), probe_fp4("e8m0"), probe_fp4("e4m3")]}
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
