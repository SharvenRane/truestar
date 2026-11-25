"""Quantized matmul kernels (Triton, CUDA): y[M, N] = x[M, K] @ W[N, K]^T.

* `w8`: weight only INT8, per channel float32 scale. INT8 widens exactly to the activation dtype,
  `tl.dot` accumulates in float32, the scale multiplies the accumulator once per output.
* `w4`: weight only INT4, per group float16 scale. One loop step per group: the low nibbles are
  the group's first half and the high nibbles its second (see `formats.QuantWeight`), so two
  `tl.dot` calls need no interleave; the group's exact integer dot product (float32) is scaled
  once and added. The weight is never rounded to float16 times scale, which is what kernels that
  dequantize before the dot do (an extra rounding per weight that the format does not have).
* `fp8`: W8A8 with `tl.dot` on float8 operands (native `mma ... e4m3` / `e5m2` on sm_89 and up,
  checked in scripts/probe_quant_triton.py), float32 accumulate, per token activation scale times
  per channel or per tensor weight scale in the epilogue.
* `fp4`: W4A4 NVFP4 (E4M3 block 16 plus tensor scale) and MXFP4 (E8M0 block 32) with
  `tl.dot_scaled` (native `mma ... kind::mxf4nvf4.block_scale` on sm_120).
* `quant_act_fp8`, `quant_act_fp4`: the run time activation quantizers, one program per row (FP8)
  or per row and 128 inputs (FP4).

Tile shapes are fixed per M bucket (no autotuning, no split K). `inject=` compiles a deliberately
wrong variant of a kernel (a `BUG` constexpr) so tests can show that verification names the
kernel bug and keeps it apart from the format's own loss. It is never set by Truestar itself.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except Exception:  # pragma: no cover
    triton = None
    tl = None

from .formats import FormatError, QuantAct, QuantWeight, parse

# Bugs a test can inject, per kernel. Keys are names used in the docs and tests.
BUGS = {
    "w8": {"channel_scale_shift": 1},
    "w4": {"group_scale_shift": 1, "group_boundary_half": 2, "nibble_swap": 3,
           "last_group_scale": 4},
    "fp8": {"tensor_scale_for_channel": 1, "fp16_accumulate": 4},
    "fp4": {"block_scale_shift": 1, "block32_for_nvfp4": 2, "nibble_swap": 3},
}


class QuantKernelError(RuntimeError):
    """A quantized kernel cannot run here (no CUDA, no Triton, a shape it does not support).
    Raised, never answered by a quiet fallback to another path."""


def available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _need_cuda(t: torch.Tensor, what: str) -> None:
    if triton is None:
        raise QuantKernelError("quantized kernels need Triton, which is not installed")
    if t.device.type != "cuda":
        raise QuantKernelError(f"{what}: quantized kernels run on CUDA only, got a {t.device.type} "
                               "tensor (no CPU fallback)")


def _bug(kind: str, inject: Optional[str]) -> int:
    if inject is None:
        return 0
    if inject not in BUGS[kind]:
        raise ValueError(f"unknown injected bug {inject!r} for {kind}; known: {sorted(BUGS[kind])}")
    return BUGS[kind][inject]


def _tiles(M: int, kind: str):
    """Fixed tile shapes per M bucket: (BM, BN, BK, warps, stages). Chosen from a sweep on an RTX
    5070 Ti over the Llama 7B class shapes of scripts/bench_quant.py (docs/QUANT.md)."""
    if kind == "fp4":
        if M <= 16:
            return 16, 64, 128, 4, 3
        if M <= 64:
            return 64, 64, 128, 4, 3
        return 128, 128, 128, 8, 3
    if kind == "w4":                      # BK is the group
        if M <= 16:
            return 16, 32, 0, 4, 4
        if M <= 64:
            return 32, 64, 0, 4, 3
        return 64, 128, 0, 4, 3
    if kind == "w8":
        if M <= 16:
            return 16, 32, 128, 4, 4
        if M <= 64:
            return 32, 64, 128, 4, 3
        return 64, 64, 64, 4, 3
    if M <= 16:                           # fp8
        return 16, 32, 128, 4, 4
    if M <= 64:
        return 64, 64, 128, 4, 3
    return 128, 128, 128, 8, 3


if triton is not None:

    @triton.jit
    def _w8_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, sxm, sxk, swn, swk, sym, syn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BUG: tl.constexpr,
                   BF16: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            a = tl.load(x_ptr + rm[:, None] * sxm + kk[None, :] * sxk,
                        mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
            b = tl.load(w_ptr + rn[None, :] * swn + kk[:, None] * swk,
                        mask=(rn[None, :] < N) & (kk[:, None] < K), other=0)
            if BF16:
                bw = b.to(a.dtype)
            else:
                # exact INT8 to float16: q + 128 in the mantissa of 1024, minus 1152
                u = (b.to(tl.int16, bitcast=False) + 128) | 0x6400
                bw = u.to(tl.float16, bitcast=True) - 1152.0
            acc = tl.dot(a, bw, acc)
        sn = rn
        if BUG == 1:
            sn = tl.minimum(rn + 1, N - 1)
        s = tl.load(s_ptr + sn, mask=rn < N, other=0.0)
        out = acc * s[None, :]
        tl.store(y_ptr + rm[:, None] * sym + rn[None, :] * syn, out.to(y_ptr.dtype.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    @triton.jit
    def _w4_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, NG, sxm, sxk, swn, sym, syn,
                   BM: tl.constexpr, BN: tl.constexpr, G: tl.constexpr, BUG: tl.constexpr,
                   BF16: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rj = tl.arange(0, G // 2)
        mm = rm[:, None] < M
        nm = rn[None, :] < N
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for g in range(0, NG):
            k0 = g * G
            xl = tl.load(x_ptr + rm[:, None] * sxm + (k0 + rj[None, :]) * sxk, mask=mm, other=0.0)
            xh = tl.load(x_ptr + rm[:, None] * sxm + (k0 + G // 2 + rj[None, :]) * sxk, mask=mm,
                         other=0.0)
            b = tl.load(w_ptr + rn[None, :] * swn + (g * (G // 2) + rj[:, None]), mask=nm,
                        other=0).to(tl.int16)
            lo = (b & 15) ^ 8          # offset binary: q + 8 in 0..15
            hi = ((b >> 4) & 15) ^ 8
            if BUG == 3:
                lo, hi = hi, lo
            # exact INT4 to float without an integer to float conversion: put q + 8 in the low
            # mantissa bits of 1024 (float16) or 128 (bfloat16), then subtract 1032 or 136
            if BF16:
                lo = (lo | 0x4300).to(tl.bfloat16, bitcast=True) - 136.0
                hi = (hi | 0x4300).to(tl.bfloat16, bitcast=True) - 136.0
            else:
                lo = (lo | 0x6400).to(tl.float16, bitcast=True) - 1032.0
                hi = (hi | 0x6400).to(tl.float16, bitcast=True) - 1032.0
            if BUG == 2:
                # the kernel believes the groups hold G / 2 inputs: two scales per real group,
                # read at the wrong flat positions
                il = rn * NG + 2 * g
                sl = tl.load(s_ptr + il, mask=(rn < N) & (il < N * NG), other=0.0).to(tl.float32)
                sh = tl.load(s_ptr + il + 1, mask=(rn < N) & (il + 1 < N * NG),
                             other=0.0).to(tl.float32)
                acc += tl.dot(xl, lo) * sl[None, :] + tl.dot(xh, hi) * sh[None, :]
            else:
                gi = g
                if BUG == 1:
                    gi = (g + 1) % NG
                if BUG == 4:
                    # an off by one at the last group boundary: it reuses the previous scale
                    gi = tl.where(g == NG - 1, g - 1, g)
                s = tl.load(s_ptr + rn * NG + gi, mask=rn < N, other=0.0).to(tl.float32)
                part = tl.dot(xh, hi, tl.dot(xl, lo))
                acc += part * s[None, :]
        tl.store(y_ptr + rm[:, None] * sym + rn[None, :] * syn, acc.to(y_ptr.dtype.element_ty),
                 mask=mm & nm)

    @triton.jit
    def _fp8_kernel(x_ptr, sx_ptr, w_ptr, sw_ptr, y_ptr, M, N, K, sxm, sxk, swn, swk, sym, syn,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    SW_CHANNEL: tl.constexpr, BUG: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        if BUG == 4:
            acc16 = tl.zeros((BM, BN), dtype=tl.float16)
        else:
            acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            a = tl.load(x_ptr + rm[:, None] * sxm + kk[None, :] * sxk,
                        mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
            b = tl.load(w_ptr + rn[None, :] * swn + kk[:, None] * swk,
                        mask=(rn[None, :] < N) & (kk[:, None] < K), other=0.0)
            if BUG == 4:
                acc16 = tl.dot(a, b, acc16, out_dtype=tl.float16)
            else:
                acc = tl.dot(a, b, acc)
        if BUG == 4:
            acc = acc16.to(tl.float32)
        sx = tl.load(sx_ptr + rm, mask=rm < M, other=0.0)
        if SW_CHANNEL:
            if BUG == 1:
                sw = tl.load(sw_ptr + rn * 0, mask=rn < N, other=0.0)
            else:
                sw = tl.load(sw_ptr + rn, mask=rn < N, other=0.0)
        else:
            sw = tl.load(sw_ptr + rn * 0, mask=rn < N, other=0.0)
        out = acc * sx[:, None] * sw[None, :]
        tl.store(y_ptr + rm[:, None] * sym + rn[None, :] * syn, out.to(y_ptr.dtype.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    @triton.jit
    def _fp4_kernel(x_ptr, sx_ptr, w_ptr, sw_ptr, tsx_ptr, tsw_ptr, y_ptr, M, N, K, KS, sym, syn,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
                    FMT_NV: tl.constexpr, BUG: tl.constexpr):
        # x: uint8 [M, K/2]; sx: [M, K/G]; w: uint8 [N, K/2]; sw: [N, K/G]; KS = K / G
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rkp = tl.arange(0, BK // 2)
        rs = tl.arange(0, BK // G)
        mm = rm[:, None] < M
        nm = rn[:, None] < N
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(x_ptr + rm[:, None] * (K // 2) + (k0 // 2 + rkp[None, :]), mask=mm, other=0)
            b = tl.load(w_ptr + rn[None, :] * (K // 2) + (k0 // 2 + rkp[:, None]),
                        mask=rn[None, :] < N, other=0)
            if BUG == 3:
                b = ((b >> 4) | (b << 4)).to(tl.uint8)
            sa = tl.load(sx_ptr + rm[:, None] * KS + (k0 // G + rs[None, :]), mask=mm)
            si = k0 // G + rs[None, :]
            if BUG == 1:
                si = tl.minimum(si + 1, KS - 1)
            if BUG == 2:
                # an NVFP4 tensor read as if its scales covered 32 inputs each
                flat = rn[:, None] * (K // 32) + (k0 + rs[None, :] * G) // 32
                sb = tl.load(sw_ptr + flat, mask=nm)
            else:
                sb = tl.load(sw_ptr + rn[:, None] * KS + si, mask=nm)
            acc = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc)
        if FMT_NV:
            acc = acc * (tl.load(tsx_ptr) * tl.load(tsw_ptr))
        tl.store(y_ptr + rm[:, None] * sym + rn[None, :] * syn, acc.to(y_ptr.dtype.element_ty),
                 mask=mm & (rn[None, :] < N))

    @triton.jit
    def _div_odd(x, d):
        # x / d rounded to odd: truncate, then set the last bit when the division was inexact.
        # A second rounding to a format with at least two fewer bits then equals one correct
        # rounding of the exact quotient (no double rounding at the format's midpoints).
        q = libdevice.div_rz(x, d)
        r = libdevice.fma(-q, d, x)
        bits = q.to(tl.int32, bitcast=True) | (r != 0).to(tl.int32)
        return bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _quant_fp8_rows(x_ptr, q_ptr, s_ptr, K, sxm, sxk, FMAX: tl.constexpr,
                        BLOCK: tl.constexpr):
        row = tl.program_id(0)
        r = tl.arange(0, BLOCK)
        am = tl.zeros((BLOCK,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK):
            v = tl.load(x_ptr + row * sxm + (k0 + r) * sxk, mask=k0 + r < K, other=0.0)
            am = tl.maximum(am, tl.abs(v.to(tl.float32)))
        amax = tl.max(am, axis=0)
        s = tl.math.div_rn(amax, FMAX)
        tl.store(s_ptr + row, s)
        d = tl.where(s > 0, s, 1.0)
        for k0 in range(0, K, BLOCK):
            v = tl.load(x_ptr + row * sxm + (k0 + r) * sxk, mask=k0 + r < K, other=0.0)
            q = _div_odd(v.to(tl.float32), d)
            q = tl.minimum(tl.maximum(q, -FMAX), FMAX)
            tl.store(q_ptr + row * K + k0 + r, q.to(q_ptr.dtype.element_ty), mask=k0 + r < K)

    @triton.jit
    def _e2m1_code(v, u):
        # E2M1 code of v / u, round to nearest even, saturating at 6, by comparing |v| with the
        # rounding midpoints times u. u has at most 4 significant bits (an E4M3 or E8M0 scale)
        # and every midpoint at most 3, so each product is exact; v is the exact value or its
        # round to odd, which compares with an exact short threshold as the exact value does.
        a = tl.abs(v)
        c = ((a > 0.25 * u).to(tl.int32) + (a >= 0.75 * u).to(tl.int32)
             + (a > 1.25 * u).to(tl.int32) + (a >= 1.75 * u).to(tl.int32)
             + (a > 2.5 * u).to(tl.int32) + (a >= 3.5 * u).to(tl.int32)
             + (a > 5.0 * u).to(tl.int32))
        c = tl.where(u > 0, c, 0)
        return tl.where((v < 0) & (c != 0), c + 8, c)

    @triton.jit
    def _quant_fp4_rows(x_ptr, q_ptr, s_ptr, am_ptr, ts_ptr, K, sxm, sxk,
                        G: tl.constexpr, NB: tl.constexpr, FMT_NV: tl.constexpr):
        # one program per (row, NB blocks of G inputs)
        row = tl.program_id(0)
        cb = tl.program_id(1)
        rb = tl.arange(0, NB)
        rg = tl.arange(0, G)
        k = cb * NB * G + rb[:, None] * G + rg[None, :]
        km = k < K
        v = tl.load(x_ptr + row * sxm + k * sxk, mask=km, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(v), axis=1)                                   # [NB]
        bi = cb * NB + rb
        bm = bi * G < K
        if FMT_NV:
            # tensor scale amax / (448 * 6), correctly rounded (each program computes the same)
            ts = tl.math.div_rn(tl.load(am_ptr), 2688.0)
            if (row == 0) & (cb == 0):
                tl.store(ts_ptr, ts)
            bs = tl.math.div_rn(tl.math.div_rn(amax, 6.0), tl.where(ts > 0, ts, 1.0))
            bs8 = bs.to(tl.float8e4nv)
            tl.store(s_ptr + row * (K // G) + bi, bs8, mask=bm)
            y = _div_odd(v, tl.where(ts > 0, ts, 1.0))
            u = tl.where(ts > 0, bs8.to(tl.float32), 0.0)
            c = _e2m1_code(y, u[:, None])                                  # [NB, G]
        else:
            bits = amax.to(tl.int32, bitcast=True)
            e = ((bits >> 23) & 255) - 127
            e = tl.where(amax > 0, e, -127) - 2
            e = tl.minimum(tl.maximum(e, -127), 127)
            tl.store(s_ptr + row * (K // G) + bi, (e + 127).to(tl.uint8), mask=bm)
            inv = ((127 - tl.minimum(tl.maximum(e, -126), 127)) << 23).to(tl.float32, bitcast=True)
            c = _e2m1_code(v * inv[:, None], 1.0)                         # [NB, G]
        pairs = tl.reshape(c, (NB * G // 2, 2))
        lo, hi = tl.split(pairs)
        byte = (lo | (hi << 4)).to(tl.uint8)
        o = cb * (NB * G // 2) + tl.arange(0, NB * G // 2)
        tl.store(q_ptr + row * (K // 2) + o, byte, mask=o < K // 2)


# ----------------------------------------------------------------------------- launchers


def _check_x(x: torch.Tensor, K: int, what: str):
    _need_cuda(x, what)
    if x.dim() != 2 or x.shape[1] != K:
        raise QuantKernelError(f"{what}: x must be [M, {K}], got {tuple(x.shape)}")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise QuantKernelError(f"{what}: activations must be float16 or bfloat16, got {x.dtype}")


def matmul_w8(x: torch.Tensor, qw: QuantWeight, inject: Optional[str] = None) -> torch.Tensor:
    N, K = qw.shape
    _check_x(x, K, "int8 matmul")
    M = x.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BM, BN, BK, nw, ns = _tiles(M, "w8")
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _w8_kernel[grid](x, qw.data, qw.scale, y, M, N, K, x.stride(0), x.stride(1),
                     qw.data.stride(0), qw.data.stride(1), y.stride(0), y.stride(1),
                     BM=BM, BN=BN, BK=BK, BUG=_bug("w8", inject), BF16=x.dtype == torch.bfloat16,
                     num_warps=nw, num_stages=ns)
    return y


def matmul_w4(x: torch.Tensor, qw: QuantWeight, inject: Optional[str] = None) -> torch.Tensor:
    N, K = qw.shape
    _check_x(x, K, "int4 matmul")
    M = x.shape[0]
    G = qw.fmt.group
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BM, BN, _, nw, ns = _tiles(M, "w4")
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _w4_kernel[grid](x, qw.data, qw.scale, y, M, N, K, K // G, x.stride(0), x.stride(1),
                     qw.data.stride(0), y.stride(0), y.stride(1),
                     BM=BM, BN=BN, G=G, BUG=_bug("w4", inject), BF16=x.dtype == torch.bfloat16,
                     num_warps=nw, num_stages=ns)
    return y


def quant_act_fp8(x: torch.Tensor, elem: str) -> QuantAct:
    """Per token (row) FP8 quantization: scale = amax / max (correctly rounded float32 division),
    codes = round to nearest even of x / scale, saturating."""
    from .formats import FP8
    _need_cuda(x, "fp8 activation quantizer")
    M, K = x.shape
    dt = torch.float8_e4m3fn if elem == "e4m3" else torch.float8_e5m2
    q = torch.empty((M, K), device=x.device, dtype=dt)
    s = torch.empty((M,), device=x.device, dtype=torch.float32)
    _quant_fp8_rows[(M,)](x, q, s, K, x.stride(0), x.stride(1), FMAX=FP8[elem].max_finite,
                          BLOCK=1024, num_warps=4)
    return QuantAct(parse(f"fp8_{elem}:channel"), q, s)


def quant_act_fp4(x: torch.Tensor, kind: str) -> QuantAct:
    """FP4 activations: NVFP4 (block 16, E4M3 block scale over a per call tensor scale amax /
    (448 * 6)) or MXFP4 (block 32, E8M0 scale 2^(floor(log2 amax) - 2))."""
    _need_cuda(x, "fp4 activation quantizer")
    fmt = parse(kind)
    M, K = x.shape
    G = fmt.group
    if K % 128:
        raise QuantKernelError(f"fp4 activations need K a multiple of 128, got {K}")
    q = torch.empty((M, K // 2), device=x.device, dtype=torch.uint8)
    ts = None
    am = s = None
    if fmt.kind == "nvfp4":
        am = x.abs().amax().float().reshape(1)          # exact: a float16 or bfloat16 value
        ts = torch.empty((1,), device=x.device, dtype=torch.float32)
        s = torch.empty((M, K // G), device=x.device, dtype=torch.float8_e4m3fn)
    else:
        s = torch.empty((M, K // G), device=x.device, dtype=torch.uint8)
    NB = 128 // G
    _quant_fp4_rows[(M, K // 128)](x, q, s, am if am is not None else s,
                                   ts if ts is not None else s, K, x.stride(0),
                                   x.stride(1), G=G, NB=NB, FMT_NV=fmt.kind == "nvfp4",
                                   num_warps=4)
    return QuantAct(fmt, q, s, ts)


def matmul_fp8(xq: QuantAct, qw: QuantWeight, out_dtype=torch.float16,
               inject: Optional[str] = None) -> torch.Tensor:
    N, K = qw.shape
    x = xq.data
    _need_cuda(x, "fp8 matmul")
    if x.dtype != qw.data.dtype:
        raise QuantKernelError(f"fp8 matmul: activation {x.dtype} and weight {qw.data.dtype} differ")
    M = x.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=out_dtype)
    BM, BN, BK, nw, ns = _tiles(M, "fp8")
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _fp8_kernel[grid](x, xq.scale, qw.data, qw.scale, y, M, N, K, x.stride(0), x.stride(1),
                      qw.data.stride(0), qw.data.stride(1), y.stride(0), y.stride(1),
                      BM=BM, BN=BN, BK=BK, SW_CHANNEL=qw.fmt.scale == "channel",
                      BUG=_bug("fp8", inject), num_warps=nw, num_stages=ns)
    return y


def matmul_fp4(xq: QuantAct, qw: QuantWeight, out_dtype=torch.float16,
               inject: Optional[str] = None) -> torch.Tensor:
    N, K = qw.shape
    if xq.fmt.kind != qw.fmt.kind:
        raise QuantKernelError(f"fp4 matmul: activations {xq.fmt.kind}, weights {qw.fmt.kind}")
    x = xq.data
    _need_cuda(x, "fp4 matmul")
    M = x.shape[0]
    G = qw.fmt.group
    nv = qw.fmt.kind == "nvfp4"
    y = torch.empty((M, N), device=x.device, dtype=out_dtype)
    BM, BN, BK, nw, ns = _tiles(M, "fp4")
    if K % BK:
        raise QuantKernelError(f"fp4 matmul needs K a multiple of {BK}, got {K}")
    tsx = xq.tensor_scale if nv else qw.scale
    tsw = qw.tensor_scale if nv else qw.scale
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _fp4_kernel[grid](x, xq.scale, qw.data, qw.scale, tsx, tsw, y, M, N, K, K // G,
                      y.stride(0), y.stride(1), BM=BM, BN=BN, BK=BK, G=G, FMT_NV=nv,
                      BUG=_bug("fp4", inject), num_warps=nw, num_stages=ns)
    return y


def kernel_kind(spec) -> str:
    fmt = parse(spec) if isinstance(spec, str) else spec
    return {"int8": "w8", "int4": "w4", "fp8": "fp8", "nvfp4": "fp4", "mxfp4": "fp4"}[fmt.kind]


def quantized_matmul(x: torch.Tensor, qw: QuantWeight, inject: Optional[str] = None,
                     return_act: bool = False):
    """y = x @ dequant(W)^T on the kernel for qw's format; activations are quantized first for
    FP8 and FP4. Returns y (and the QuantAct the matmul read, with return_act)."""
    k = kernel_kind(qw.fmt)
    act = None
    if k == "w8":
        y = matmul_w8(x, qw, inject)
    elif k == "w4":
        y = matmul_w4(x, qw, inject)
    elif k == "fp8":
        _check_x(x, qw.shape[1], "fp8 matmul")
        act = quant_act_fp8(x, qw.fmt.elem)
        y = matmul_fp8(act, qw, x.dtype, inject)
    elif k == "fp4":
        _check_x(x, qw.shape[1], "fp4 matmul")
        act = quant_act_fp4(x, qw.fmt.spec)
        y = matmul_fp4(act, qw, x.dtype, inject)
    else:  # pragma: no cover
        raise FormatError(f"no kernel for {qw.fmt.spec}")
    return (y, act) if return_act else y
