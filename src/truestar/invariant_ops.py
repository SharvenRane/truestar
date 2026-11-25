"""Batch invariant reference ops (proof of concept for pain P14).

Each op here gives a row of the batch the same bits whatever else is in the batch, how big the
batch is and where the row sits in it. They are slow on purpose: they show what "fixed reduction
order per row" means and what it costs, they are not wired into the kernel layer.

Two ways to fix the order:

* fixed tiling (`matmul`, `linear`, `bmm`): rows are processed in tiles of exactly `TILE` rows,
  the last tile padded with zero rows, and every tile is one call to the vendor GEMM at the same
  shape (TILE, K) x (K, N). A row then sees the same kernel, the same split of K and the same
  reduction tree whatever the batch is. This relies on the vendor kernel being a function of the
  shape only (no dependence on the values of other rows or on the row's position inside the tile),
  which the tests check on every backend they run on.
* explicit trees (`tree_sum`, `tree_max`, `softmax`, `layer_norm`, `rms_norm`, `matmul(mode="tree")`):
  reductions are written as a fixed pairwise tree of elementwise adds whose shape depends only on
  the reduced length. Elementwise IEEE adds and multiplies are exact per element on every backend,
  so these are invariant by construction.

Elementwise ops with a vectorized body and a scalar tail (on CPU, torch 2.11: sigmoid, silu and
tanh GELU give different bits in the tail) are run through `elementwise`, which pads the flat
tensor so that no element ever lands in a tail.

`measure_overhead()` times each op against the default PyTorch op on a device.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

TILE = 64  # rows per GEMM call; fixed, never derived from the batch
CHUNK = 64  # K chunk of the explicit tree matmul
_VEC = 64  # elementwise ops see a multiple of this many elements (no scalar tail)
_SPAN = 16384  # and at most this many per call (half the PyTorch CPU grain size: one thread)


def _acc_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


# ----------------------------------------------------------------------------- elementwise


def elementwise(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """fn(x) computed so that every element takes the same code path whatever the tensor size.

    The flat tensor is padded to a multiple of 64 elements and run in pieces of 16384, so a CPU
    kernel never splits it across threads and never reaches its scalar tail loop.
    """
    flat = x.reshape(-1)
    n = flat.numel()
    if n == 0:
        return fn(x)
    pad = (-n) % _VEC
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    outs = [fn(flat[i:i + _SPAN]) for i in range(0, flat.numel(), _SPAN)]
    return torch.cat(outs)[:n].reshape(x.shape)


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    return elementwise(lambda t: F.gelu(t, approximate=approximate), x)


def silu(x: torch.Tensor) -> torch.Tensor:
    return elementwise(F.silu, x)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return elementwise(torch.sigmoid, x)


# ----------------------------------------------------------------------------- reductions


def _tree(x: torch.Tensor, dim: int, op, identity: float, keepdim: bool,
          width: int = CHUNK) -> torch.Tensor:
    """Reduce `dim` in chunks of exactly `width` elements (the last one padded with the
    identity), each chunk by a pairwise tree, chunks accumulated left to right.

    The order depends on nothing but the reduced length, and padding with identity elements (masked
    attention positions, padded tokens) leaves every partial result unchanged, so the result is
    also the same for a sequence padded with zeros."""
    dim = dim % x.ndim
    n = x.shape[dim]
    if n == 0:
        raise ValueError("tree reduction over an empty dimension")
    pad = (-n) % width
    if pad:
        shape = list(x.shape)
        shape[dim] = pad
        x = torch.cat([x, x.new_full(shape, identity)], dim=dim)
    acc = None
    for c0 in range(0, x.shape[dim], width):
        part = x.narrow(dim, c0, width)
        while part.shape[dim] > 1:
            h = part.shape[dim] // 2
            part = op(part.narrow(dim, 0, h), part.narrow(dim, h, h))
        acc = part if acc is None else op(acc, part)
    return acc if keepdim else acc.squeeze(dim)


def tree_sum(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Sum over `dim` by a fixed pairwise tree (accumulates half types in float32)."""
    acc = _acc_dtype(x.dtype)
    return _tree(x.to(acc), dim, torch.add, 0.0, keepdim).to(x.dtype)


def tree_max(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return _tree(x, dim, torch.maximum, float("-inf"), keepdim)


def mean(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    acc = _acc_dtype(x.dtype)
    return (_tree(x.to(acc), dim, torch.add, 0.0, keepdim) / x.shape[dim]).to(x.dtype)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    acc = _acc_dtype(x.dtype)
    xa = x.to(acc)
    m = tree_max(xa, dim, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))  # rows that are all -inf
    e = elementwise(torch.exp, xa - m)
    return (e / _tree(e, dim, torch.add, 0.0, True)).to(x.dtype)


def layer_norm(x: torch.Tensor, normalized_shape: Sequence[int], weight=None, bias=None,
               eps: float = 1e-5) -> torch.Tensor:
    k = len(tuple(normalized_shape))
    acc = _acc_dtype(x.dtype)
    xa = x.to(acc).reshape(*x.shape[: x.ndim - k], -1)
    n = xa.shape[-1]
    mu = _tree(xa, -1, torch.add, 0.0, True) / n
    d = xa - mu
    var = _tree(d * d, -1, torch.add, 0.0, True) / n
    y = (d * torch.rsqrt(var + eps)).reshape(x.shape)
    if weight is not None:
        y = y * weight.to(acc)
    if bias is not None:
        y = y + bias.to(acc)
    return y.to(x.dtype)


def rms_norm(x: torch.Tensor, weight=None, eps: float = 1e-6) -> torch.Tensor:
    acc = _acc_dtype(x.dtype)
    xa = x.to(acc)
    ms = _tree(xa * xa, -1, torch.add, 0.0, True) / x.shape[-1]
    y = xa * torch.rsqrt(ms + eps)
    if weight is not None:
        y = y * weight.to(acc)
    return y.to(x.dtype)


# ----------------------------------------------------------------------------- matmul


def _mm_tiled(a: torch.Tensor, b: torch.Tensor, tile: int) -> torch.Tensor:
    m = a.shape[0]
    pad = (-m) % tile
    if pad:
        a = torch.cat([a, a.new_zeros(pad, a.shape[1])])
    out = torch.cat([a[i:i + tile] @ b for i in range(0, a.shape[0], tile)]) if a.shape[0] else a @ b
    return out[:m]


def _mm_tree(a: torch.Tensor, b: torch.Tensor, chunk: int) -> torch.Tensor:
    """a[..., M, K] @ b[..., K, N] from elementwise products and fixed trees only, so leading
    batch dims need no loop: no op here rounds differently for a different batch."""
    acc_t = _acc_dtype(a.dtype)
    a, b = a.to(acc_t), b.to(acc_t)
    pad = (-a.shape[-1]) % chunk
    if pad:  # zero columns of a and zero rows of b add exact zeros
        a = torch.cat([a, a.new_zeros(*a.shape[:-1], pad)], dim=-1)
        b = torch.cat([b, b.new_zeros(*b.shape[:-2], pad, b.shape[-1])], dim=-2)
    acc = None
    for k0 in range(0, a.shape[-1], chunk):
        part = a[..., :, k0:k0 + chunk, None] * b[..., None, k0:k0 + chunk, :]
        t = _tree(part, -2, torch.add, 0.0, False, width=chunk)
        acc = t if acc is None else acc + t
    return acc


def matmul(x: torch.Tensor, w: torch.Tensor, mode: str = "tile", tile: int = TILE,
           chunk: int = CHUNK) -> torch.Tensor:
    """x[..., K] @ w[K, N] with each row's reduction order fixed.

    mode "tile": vendor GEMM on fixed (tile, K) blocks. mode "tree": explicit chunked tree, slow
    but invariant by construction.
    """
    lead = x.shape[:-1]
    # One memory layout whatever the batch: a reshape can be a view for one batch size and a copy
    # for another, and the GEMM kernel (and its rounding) depends on the operand strides.
    a = x.reshape(-1, x.shape[-1]).contiguous()
    if mode == "tile":
        # A row major [K, N] right operand: with a transposed one (nn.Linear's weight.t()), CPU
        # float32 GEMM rounds half the positions of a tile differently (torch 2.11, MKL).
        out = _mm_tiled(a, w.contiguous(), tile)
    elif mode == "tree":
        out = _mm_tree(a, w, chunk).to(x.dtype)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return out.reshape(*lead, w.shape[1])


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
           mode: str = "tile") -> torch.Tensor:
    y = matmul(x, weight.t(), mode=mode)
    return y + bias if bias is not None else y


def bmm(a: torch.Tensor, b: torch.Tensor, mode: str = "tile") -> torch.Tensor:
    """a[..., M, K] @ b[..., K, N]. In tile mode each matrix of the batch runs on its own (no
    batched GEMM, whose kernel choice depends on the batch count)."""
    if mode == "tree":
        return _mm_tree(a, b, CHUNK).to(a.dtype)
    lead = a.shape[:-2]
    a3 = a.reshape(-1, *a.shape[-2:])
    b3 = b.expand(*lead, *b.shape[-2:]).reshape(-1, *b.shape[-2:])
    outs = [matmul(a3[i], b3[i].contiguous(), mode=mode) for i in range(a3.shape[0])]
    return torch.stack(outs).reshape(*lead, a.shape[-2], b.shape[-1])


def scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal: bool = False,
                                 scale: Optional[float] = None, mode: str = "tree"):
    """Math attention from the invariant bmm and softmax. q, k, v are [..., S, D].

    The default "tree" mode is invariant to the batch and also to right padding of the sequence
    (the vendor GEMM in "tile" mode changes its kernel with the key length)."""
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    s = bmm(q, k.transpose(-2, -1), mode=mode) * scale
    if is_causal:
        L, S = q.shape[-2], k.shape[-2]
        keep = torch.ones(L, S, dtype=torch.bool, device=q.device).tril(S - L)
        s = s.masked_fill(~keep, float("-inf"))
    if attn_mask is not None:
        s = s.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else s + attn_mask
    return bmm(softmax(s, -1), v, mode=mode)


# ----------------------------------------------------------------------------- modules


class Linear(nn.Linear):
    """nn.Linear with the invariant matmul (same parameters, drop in)."""

    def forward(self, x):
        return linear(x, self.weight, self.bias)


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class GELU(nn.GELU):
    def forward(self, x):
        return gelu(x, self.approximate)


# ----------------------------------------------------------------------------- overhead


def _time(fn, device: str, reps: int) -> float:
    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    for _ in range(2):
        fn()
    sync()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        sync()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


def measure_overhead(device: str = "cpu", dtype: torch.dtype = torch.float32,
                     batch: int = 64, seq: int = 16, dim: int = 256,
                     reps: int = 10) -> List[Dict[str, object]]:
    """Time each invariant op against the default op on `device` (seconds per call).

    Shapes model one decoder layer at `batch` x `seq` tokens of width `dim`.
    """
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g).to(device=device, dtype=dtype)
    x, w, b = r(batch * seq, dim), r(4 * dim, dim), r(4 * dim)
    heads, hd = 4, dim // 4
    q, k, v = r(batch, heads, seq, hd), r(batch, heads, seq, hd), r(batch, heads, seq, hd)
    lw, lb = r(dim), r(dim)
    cases = [
        ("linear (tile)", lambda: F.linear(x, w, b), lambda: linear(x, w, b)),
        ("linear (tree)", lambda: F.linear(x, w, b), lambda: linear(x, w, b, mode="tree")),
        ("sdpa causal", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
         lambda: scaled_dot_product_attention(q, k, v, is_causal=True)),
        ("softmax", lambda: torch.softmax(x, -1), lambda: softmax(x, -1)),
        ("layer_norm", lambda: F.layer_norm(x, (dim,), lw, lb), lambda: layer_norm(x, (dim,), lw, lb)),
        ("gelu tanh", lambda: F.gelu(x, approximate="tanh"), lambda: gelu(x, "tanh")),
    ]
    out = []
    with torch.no_grad():
        for name, base, inv in cases:
            reps_i = max(1, reps // 5) if "tree" in name else reps
            tb, ti = _time(base, device, reps_i), _time(inv, device, reps_i)
            out.append({"op": name, "device": device, "dtype": str(dtype).replace("torch.", ""),
                        "default_s": tb, "invariant_s": ti, "ratio": ti / tb if tb > 0 else None})
    return out
