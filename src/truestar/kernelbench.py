"""A small benchmark of correct and subtly wrong kernels, written in plain PyTorch.

Each case is a reference function (what the kernel means), a candidate (an implementation), a
flag saying whether the candidate is correct, and an input maker. Correct candidates compute the
same function in another valid order or precision policy (blocked accumulation, float32
accumulation for half inputs, online softmax, Welford). Wrong candidates carry the bugs people hit
in hand and AI written kernels: a tail the last block misses, a mask edge off by one, float16
accumulation, a lost update in a racy reduction, a wrong epsilon, a one pass variance, a contiguity
assumption, a shape that only works when it divides the block.

Used by scripts/bench_tolerance.py (R6) and scripts/bench_kernel_check.py (R1), and by the tests.
Candidates are written so that the bug fires only where it would in a real kernel, which is often
not on the one input a single allclose check uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

BLOCK = 64  # the "tile" size the hand kernels below use


@dataclass
class Case:
    name: str
    reference: Callable
    candidate: Callable
    correct: bool
    make_inputs: Callable  # (shape dict, dtype, generator) -> tuple of inputs
    dims: Dict[str, int]  # the shape a single input test uses
    dtype: torch.dtype = torch.float32
    bug: str = ""
    families: Tuple[str, ...] = ("normal",)  # value families the bug needs (for documentation)
    nondeterministic: bool = False
    note: str = ""
    # What the bug needs to show: "strided" inputs, "large" values, another "shape". Empty: it shows
    # on the single example input already.
    needs: Tuple[str, ...] = ()


def _randn(shape, dtype, g, scale=1.0, offset=0.0):
    return (torch.randn(shape, generator=g, dtype=torch.float64) * scale + offset).to(dtype)


# ----------------------------------------------------------------------------- matmul


def mm_ref(a, b):
    return a @ b


def mm_blocked(a, b):
    """Correct: K in blocks of 64, accumulated in float32 (a tiled GEMM's order)."""
    acc = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float32)
    for k0 in range(0, a.shape[1], BLOCK):
        acc += a[:, k0:k0 + BLOCK].float() @ b[k0:k0 + BLOCK].float()
    return acc.to(a.dtype)


def mm_fp16_acc(a, b):
    """Wrong: accumulates the K blocks in the input dtype (float16 accumulation)."""
    acc = torch.zeros(a.shape[0], b.shape[1], dtype=a.dtype)
    for k0 in range(0, a.shape[1], 16):
        acc = acc + (a[:, k0:k0 + 16].float() @ b[k0:k0 + 16].float()).to(a.dtype)
    return acc


def mm_tail(a, b):
    """Wrong: grid is N // BLOCK, so the last partial column tile is never written (stays 0)."""
    n = b.shape[1]
    out = torch.zeros(a.shape[0], n, dtype=torch.float32)
    full = (n // BLOCK) * BLOCK
    out[:, :full] = a.float() @ b[:, :full].float()
    return out.to(a.dtype)


def mm_k_tail(a, b):
    """Wrong: the K loop stops at K // BLOCK * BLOCK, so a ragged K drops its last terms."""
    k = (a.shape[1] // BLOCK) * BLOCK
    return (a[:, :k].float() @ b[:k].float()).to(a.dtype)


def mm_contig(a, b):
    """Wrong: reads b through a contiguous layout assumption (as_strided with row major strides)."""
    bb = torch.as_strided(b, b.shape, (b.shape[1], 1)) if not b.is_contiguous() else b
    return (a.float() @ bb.float()).to(a.dtype)


def mm_tf32(a, b):
    """Wrong under the IEEE float32 policy: rounds the operands to TF32 first."""
    from .gradcheck import _tf32

    return (_tf32(a.double()) @ _tf32(b.double())).to(a.dtype)


def mm_scale_1e5(a, b):
    """Wrong: a constant off by 1e-5 relative (a folded scale written with too few digits)."""
    return ((a.float() @ b.float()) * (1 + 1e-5)).to(a.dtype)


def mm_scale_1e3(a, b):
    """Wrong: a constant off by 1e-3 relative, about two float16 ulps."""
    return ((a.float() @ b.float()) * (1 + 1e-3)).to(a.dtype)


def mm_split_k(a, b):
    """Correct: split K in 4 parts, partial sums in float32, then summed (split K GEMM)."""
    parts = torch.tensor_split(torch.arange(a.shape[1]), 4)
    acc = sum((a[:, p].float() @ b[p].float()) for p in parts if len(p))
    return acc.to(a.dtype)


TILE_ROWS = 1024  # the row tile of the small fault cases below (adversarial review T3)


def _row_tiles(a, b, last_tile):
    """A GEMM by row tiles of 1024, K in steps of 16 (one tensor core MMA step) accumulated in
    float32; `last_tile(a_rows, b)` computes the last tile instead when given."""
    m = a.shape[0]
    out = torch.empty(m, b.shape[1], dtype=a.dtype)
    for r0 in range(0, m, TILE_ROWS):
        rows = a[r0:r0 + TILE_ROWS]
        if last_tile is not None and r0 + TILE_ROWS >= m:
            out[r0:] = last_tile(rows, b)
            continue
        acc = torch.zeros(rows.shape[0], b.shape[1], dtype=torch.float32)
        for k0 in range(0, a.shape[1], 16):
            acc += rows[:, k0:k0 + 16].float() @ b[k0:k0 + 16].float()
        out[r0:r0 + TILE_ROWS] = acc.to(a.dtype)
    return out


def mm_row_tiles(a, b):
    """Correct: row tiles of 1024, float32 accumulation in K steps of 16."""
    return _row_tiles(a, b, None)


def mm_row_tile_half_acc(a, b):
    """Wrong: the last tile of 1024 rows (a remainder path) accumulates in the input dtype."""
    def tail(rows, bb):
        acc = torch.zeros(rows.shape[0], bb.shape[1], dtype=rows.dtype)
        for k0 in range(0, rows.shape[1], 16):
            acc = acc + (rows[:, k0:k0 + 16].float() @ bb[k0:k0 + 16].float()).to(rows.dtype)
        return acc
    return _row_tiles(a, b, tail)


def mm_row_tile_stale_scale(a, b):
    """Wrong: the last tile of 1024 rows applies a stale epilogue scale, 1 + 2^-7 (0.8%)."""
    return _row_tiles(a, b, lambda rows, bb: ((rows.float() @ bb.float()) * (1 + 2 ** -7))
                      .to(rows.dtype))


def ln_last_channel_no_bias(x, w, b):
    """Wrong: the last channel (the vector tail of the channel loop) is stored without its bias."""
    xf = x.float()
    mu = xf.mean(-1, keepdim=True)
    var = ((xf - mu) ** 2).mean(-1, keepdim=True)
    y = (xf - mu) * torch.rsqrt(var + 1e-5) * w.float()
    bias = b.float().clone()
    bias[-1] = 0
    return (y + bias).to(x.dtype)


def _mm_inputs(d, dtype, g, scale=1.0):
    return (_randn((d["M"], d["K"]), dtype, g, scale), _randn((d["K"], d["N"]), dtype, g, scale))


# ----------------------------------------------------------------------------- softmax / attention


def softmax_ref(x):
    return torch.softmax(x, dim=-1)


def softmax_online(x):
    """Correct: online softmax over blocks of 64 in float32 (running max and sum)."""
    xf = x.float()
    m = torch.full(xf.shape[:-1] + (1,), -math.inf)
    s = torch.zeros(xf.shape[:-1] + (1,))
    for j in range(0, xf.shape[-1], BLOCK):
        blk = xf[..., j:j + BLOCK]
        mn = torch.maximum(m, blk.amax(-1, keepdim=True))
        s = s * torch.exp(m - mn) + torch.exp(blk - mn).sum(-1, keepdim=True)
        m = mn
    return (torch.exp(xf - m) / s).to(x.dtype)


def softmax_no_max(x):
    """Wrong: skips the max subtraction; overflows for large logits."""
    e = torch.exp(x.float())
    return (e / e.sum(-1, keepdim=True)).to(x.dtype)


def softmax_tail(x):
    """Wrong: the reduction loop covers whole blocks only; the last partial block is not summed."""
    xf = x.float()
    n = xf.shape[-1]
    full = max(BLOCK, (n // BLOCK) * BLOCK)
    m = xf.amax(-1, keepdim=True)
    e = torch.exp(xf - m)
    return (e / e[..., :full].sum(-1, keepdim=True)).to(x.dtype)


def causal_attn_ref(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def _attn(q, k, v, mask_fn, scale=None):
    qf, kf, vf = q.float(), k.float(), v.float()
    s = (qf @ kf.transpose(-1, -2)) * (scale if scale is not None else qf.shape[-1] ** -0.5)
    i = torch.arange(s.shape[-2]).unsqueeze(-1)
    j = torch.arange(s.shape[-1]).unsqueeze(0)
    s = s.masked_fill(~mask_fn(i, j), -math.inf)
    return (torch.softmax(s, -1) @ vf).to(q.dtype)


def attn_correct(q, k, v):
    """Correct: explicit causal attention in float32 (j <= i)."""
    return _attn(q, k, v, lambda i, j: j <= i)


def attn_mask_edge(q, k, v):
    """Wrong: mask edge off by one on the last query block (j <= i + 1 where i >= S - BLOCK).
    Every row in the last block sees one future key."""
    s = q.shape[-2]
    return _attn(q, k, v, lambda i, j: (j <= i) | ((i >= s - BLOCK) & (j == i + 1)))


def attn_scale(q, k, v):
    """Wrong: scale 1/sqrt(d + 1) instead of 1/sqrt(d)."""
    return _attn(q, k, v, lambda i, j: j <= i, scale=(q.shape[-1] + 1) ** -0.5)


def _attn_inputs(d, dtype, g, scale=1.0):
    shp = (d["B"], d["H"], d["S"], d["D"])
    return tuple(_randn(shp, dtype, g, scale) for _ in range(3))


# ----------------------------------------------------------------------------- norms


def ln_ref(x, w, b):
    return F.layer_norm(x, (x.shape[-1],), w, b, 1e-5)


def ln_two_pass(x, w, b):
    xf = x.float()
    mu = xf.mean(-1, keepdim=True)
    var = ((xf - mu) ** 2).mean(-1, keepdim=True)
    return ((xf - mu) * torch.rsqrt(var + 1e-5) * w.float() + b.float()).to(x.dtype)


def ln_welford(x, w, b):
    """Correct: Welford mean and variance over blocks of 64, merged pairwise (Chan et al.)."""
    xf = x.float()
    n, mean, m2 = 0, None, None
    for j in range(0, xf.shape[-1], BLOCK):
        blk = xf[..., j:j + BLOCK]
        nb = blk.shape[-1]
        mb = blk.mean(-1, keepdim=True)
        m2b = ((blk - mb) ** 2).sum(-1, keepdim=True)
        if mean is None:
            n, mean, m2 = nb, mb, m2b
            continue
        delta = mb - mean
        tot = n + nb
        mean = mean + delta * nb / tot
        m2 = m2 + m2b + delta * delta * n * nb / tot
        n = tot
    var = m2 / n
    return ((xf - mean) * torch.rsqrt(var + 1e-5) * w.float() + b.float()).to(x.dtype)


def ln_one_pass(x, w, b):
    """Wrong: variance as E[x^2] - E[x]^2, which cancels when the mean is large."""
    xf = x.float()
    mu = xf.mean(-1, keepdim=True)
    var = (xf * xf).mean(-1, keepdim=True) - mu * mu
    return ((xf - mu) * torch.rsqrt(var.clamp_min(0) + 1e-5) * w.float() + b.float()).to(x.dtype)


def ln_eps(x, w, b):
    """Wrong: epsilon 1e-3 instead of 1e-5 (visible when the row variance is small)."""
    xf = x.float()
    mu = xf.mean(-1, keepdim=True)
    var = ((xf - mu) ** 2).mean(-1, keepdim=True)
    return ((xf - mu) * torch.rsqrt(var + 1e-3) * w.float() + b.float()).to(x.dtype)


def _ln_inputs(d, dtype, g, scale=1.0, offset=0.0):
    x = _randn((d["M"], d["N"]), dtype, g, scale, offset)
    w = _randn((d["N"],), dtype, g, 0.1, 1.0)
    b = _randn((d["N"],), dtype, g, 0.1)
    return (x, w, b)


# ----------------------------------------------------------------------------- reductions


def rowsum_ref(x):
    return x.sum(-1)


def rowsum_sequential(x):
    """Correct: one float32 accumulator per row, one element at a time (a naive loop)."""
    return torch.cumsum(x.float(), -1)[..., -1].to(x.dtype)


def rowsum_blocks(x):
    """Correct: block partial sums in float32, then a tree over blocks."""
    xf = x.float()
    n = xf.shape[-1]
    pad = (-n) % BLOCK
    xp = F.pad(xf, (0, pad))
    return xp.reshape(*xf.shape[:-1], -1, BLOCK).sum(-1).sum(-1).to(x.dtype)


class RacyRowSum:
    """Wrong: block partials are combined with a non atomic read, add, write, so now and then one
    block's partial is lost (here: with probability p per row, seeded per call). Nondeterministic."""

    def __init__(self, p: float = 0.02):
        self.p = p
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        g = torch.Generator().manual_seed(1234 + self.calls)
        xf = x.float()
        n = xf.shape[-1]
        pad = (-n) % BLOCK
        parts = F.pad(xf, (0, pad)).reshape(*xf.shape[:-1], -1, BLOCK).sum(-1)
        lost = torch.rand(parts.shape[:-1], generator=g) < self.p
        drop = torch.zeros_like(parts)
        drop[..., -1] = parts[..., -1]
        parts = torch.where(lost.unsqueeze(-1), parts - drop, parts)
        return parts.sum(-1).to(x.dtype)


def rowsum_tail(x):
    """Wrong: the last partial block is not summed."""
    n = x.shape[-1]
    full = max(BLOCK, (n // BLOCK) * BLOCK)
    return x[..., :full].float().sum(-1).to(x.dtype)


# ----------------------------------------------------------------------------- elementwise


def gelu_ref(x):
    return F.gelu(x)


def gelu_fp32(x):
    return F.gelu(x.float()).to(x.dtype)


def gelu_tanh(x):
    """Wrong: the tanh approximation where the exact (erf) GELU is the contract."""
    return F.gelu(x.float(), approximate="tanh").to(x.dtype)


def gelu_last(x):
    """Wrong: the last element of a flat launch is never written (grid one short when the size
    is not a multiple of the block): it keeps whatever the buffer held."""
    y = F.gelu(x.float()).to(x.dtype).reshape(-1).clone()
    n = y.numel()
    if n % BLOCK:
        y[-1] = 0
    return y.reshape(x.shape)


def silu_ref(x):
    return F.silu(x)


def silu_steps(x):
    """Correct: x * sigmoid(x) with both steps rounded to the input dtype (unfused)."""
    return x * torch.sigmoid(x)


# ----------------------------------------------------------------------------- the set


def _dims(**kw):
    return dict(kw)


def cases() -> List[Case]:
    """The benchmark: (name, reference, candidate, correct, inputs, dims, dtype, bug)."""
    mm_d = _dims(M=96, K=1000, N=200)  # N not a multiple of 64, K ragged too
    mm_d_even = _dims(M=64, K=1024, N=256)
    attn_d = _dims(B=1, H=2, S=192, D=64)
    ln_d = _dims(M=64, N=768)
    red_d = _dims(M=128, N=4000)
    ew_d = _dims(M=33, N=127)
    C = Case
    mmi = _mm_inputs
    out = [
        # matmul
        C("mm_blocked_fp32", mm_ref, mm_blocked, True, mmi, mm_d, torch.float32),
        C("mm_blocked_fp16", mm_ref, mm_blocked, True, mmi, mm_d, torch.float16),
        C("mm_blocked_bf16", mm_ref, mm_blocked, True, mmi, mm_d, torch.bfloat16),
        C("mm_split_k_fp32", mm_ref, mm_split_k, True, mmi, mm_d, torch.float32),
        C("mm_split_k_fp16", mm_ref, mm_split_k, True, mmi, mm_d, torch.float16),
        C("mm_fp16_accumulation", mm_ref, mm_fp16_acc, False, mmi, mm_d, torch.float16,
          bug="float16 accumulation"),
        C("mm_bf16_accumulation", mm_ref, mm_fp16_acc, False, mmi, mm_d, torch.bfloat16,
          bug="bfloat16 accumulation"),
        C("mm_column_tail_fp16", mm_ref, mm_tail, False, mmi, mm_d, torch.float16,
          bug="last column tile not written"),
        C("mm_column_tail_even_shape", mm_ref, mm_tail, False, mmi, mm_d_even, torch.float16,
          bug="last column tile not written (only at shapes that do not divide 64)",
          needs=("shape",)),
        C("mm_k_tail_fp32", mm_ref, mm_k_tail, False, mmi, mm_d, torch.float32,
          bug="ragged K tail dropped"),
        C("mm_scale_1e-5_fp32", mm_ref, mm_scale_1e5, False, mmi, mm_d, torch.float32,
          bug="constant off by 1e-5 relative"),
        C("mm_scale_1e-3_fp16", mm_ref, mm_scale_1e3, False, mmi, mm_d, torch.float16,
          bug="constant off by 1e-3 relative (two float16 ulps)"),
        C("mm_tf32_fp32", mm_ref, mm_tf32, False, mmi, mm_d, torch.float32,
          bug="TF32 operands under an IEEE float32 policy"),
        C("mm_contiguity_fp32", mm_ref, mm_contig, False, mmi, mm_d, torch.float32,
          bug="assumes contiguous operands (only wrong on strided inputs)", needs=("strided",)),
        # softmax and attention
        C("softmax_online_fp16", softmax_ref, softmax_online, True,
          lambda d, dt, g: (_randn((d["M"], d["N"]), dt, g, 3.0),), _dims(M=64, N=1000),
          torch.float16),
        C("softmax_online_fp32", softmax_ref, softmax_online, True,
          lambda d, dt, g: (_randn((d["M"], d["N"]), dt, g, 3.0),), _dims(M=64, N=1000),
          torch.float32),
        C("softmax_no_max_fp32", softmax_ref, softmax_no_max, False,
          lambda d, dt, g: (_randn((d["M"], d["N"]), dt, g, 3.0),), _dims(M=64, N=1000),
          torch.float32, bug="no max subtraction (overflows only on large logits)",
          needs=("large",)),
        C("softmax_tail_fp16", softmax_ref, softmax_tail, False,
          lambda d, dt, g: (_randn((d["M"], d["N"]), dt, g, 3.0),), _dims(M=64, N=1000),
          torch.float16, bug="last partial block left out of the sum"),
        C("attn_causal_fp16", causal_attn_ref, attn_correct, True, _attn_inputs, attn_d,
          torch.float16),
        C("attn_causal_bf16", causal_attn_ref, attn_correct, True, _attn_inputs, attn_d,
          torch.bfloat16),
        C("attn_mask_edge_fp16", causal_attn_ref, attn_mask_edge, False, _attn_inputs, attn_d,
          torch.float16, bug="causal mask edge off by one in the last query block"),
        C("attn_scale_fp16", causal_attn_ref, attn_scale, False, _attn_inputs, attn_d,
          torch.float16, bug="scale 1/sqrt(d+1)"),
        # norms
        C("ln_two_pass_fp32", ln_ref, ln_two_pass, True, _ln_inputs, ln_d, torch.float32),
        C("ln_welford_fp16", ln_ref, ln_welford, True, _ln_inputs, ln_d, torch.float16),
        C("ln_welford_offset_fp32", ln_ref, ln_welford, True,
          lambda d, dt, g: _ln_inputs(d, dt, g, 1.0, 300.0), ln_d, torch.float32),
        C("ln_one_pass_offset_fp32", ln_ref, ln_one_pass, False,
          lambda d, dt, g: _ln_inputs(d, dt, g, 1.0, 300.0), ln_d, torch.float32,
          bug="one pass variance (cancellation with a large mean)"),
        C("ln_eps_fp32", ln_ref, ln_eps, False, _ln_inputs, ln_d, torch.float32,
          bug="epsilon 1e-3 instead of 1e-5"),
        C("ln_eps_small_var_fp16", ln_ref, ln_eps, False,
          lambda d, dt, g: _ln_inputs(d, dt, g, 0.05), ln_d, torch.float16,
          bug="epsilon 1e-3 instead of 1e-5 (rows with small variance)"),
        # reductions
        C("rowsum_sequential_fp32", rowsum_ref, rowsum_sequential, True,
          lambda d, dt, g: (_randn((d["M"], d["N"]), dt, g),), red_d, torch.float32),
        C("rowsum_blocks_fp16", rowsum_ref, rowsum_blocks, True,
          lambda d, dt, g: (_randn((d["M"], d["N"]), dt, g),), red_d, torch.float16),
        C("rowsum_racy_fp32", rowsum_ref, RacyRowSum(0.02), False,
          lambda d, dt, g: (_randn((d["M"], d["N"]), dt, g),), red_d, torch.float32,
          bug="racy reduction loses a block partial now and then", nondeterministic=True),
        C("rowsum_tail_fp32", rowsum_ref, rowsum_tail, False,
          lambda d, dt, g: (_randn((d["M"], d["N"]), dt, g),), red_d, torch.float32,
          bug="last partial block not summed"),
        # elementwise
        C("gelu_fp32_internal_fp16", gelu_ref, gelu_fp32, True,
          lambda d, dt, g: (_randn((d["M"], d["N"]), dt, g, 2.0),), ew_d, torch.float16),
        C("gelu_torch_fp32", gelu_ref, gelu_fp32, True,
          lambda d, dt, g: (_randn((d["M"], d["N"]), dt, g, 4.0),), ew_d, torch.float32),
        C("silu_torch_fp32", silu_ref, silu_steps, True,
          lambda d, dt, g: (_randn((d["M"], d["N"]), dt, g, 8.0),), ew_d, torch.float32),
        C("gelu_tanh_fp32", gelu_ref, gelu_tanh, False,
          lambda d, dt, g: (_randn((d["M"], d["N"]), dt, g, 2.0),), ew_d, torch.float32,
          bug="tanh approximation instead of erf"),
        C("gelu_last_element_fp16", gelu_ref, gelu_last, False,
          lambda d, dt, g: (_randn((d["M"], d["N"]), dt, g, 2.0),), ew_d, torch.float16,
          bug="last element of a ragged launch never written"),
        C("silu_unfused_fp16", silu_ref, silu_steps, True,
          lambda d, dt, g: (_randn((d["M"], d["N"]), dt, g, 2.0),), ew_d, torch.float16),
        C("silu_unfused_bf16", silu_ref, silu_steps, True,
          lambda d, dt, g: (_randn((d["M"], d["N"]), dt, g, 2.0),), ew_d, torch.bfloat16),
    ]
    # Small faults (adversarial review T3): one tail tile of 1024 rows, one channel; float16 and
    # bfloat16; each with a correct kernel of the same shape and order.
    tile_d = _dims(M=4 * TILE_ROWS, K=512, N=64)
    ln_big = _dims(M=1024, N=768)
    for dt, tag in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        out += [
            C(f"mm_row_tiles_{tag}", mm_ref, mm_row_tiles, True, mmi, tile_d, dt),
            C(f"mm_last_row_tile_half_acc_{tag}", mm_ref, mm_row_tile_half_acc, False, mmi,
              tile_d, dt, bug=f"last tile of 1024 rows accumulates in {tag}"),
            C(f"mm_last_row_tile_scale_{tag}", mm_ref, mm_row_tile_stale_scale, False, mmi,
              tile_d, dt, bug="last tile of 1024 rows scaled by 1 + 2^-7"),
            C(f"ln_two_pass_{tag}", ln_ref, ln_two_pass, True, _ln_inputs, ln_big, dt),
            C(f"ln_last_channel_no_bias_{tag}", ln_ref, ln_last_channel_no_bias, False,
              _ln_inputs, ln_big, dt, bug="one channel (the last) stored without its bias"),
        ]
    return out
