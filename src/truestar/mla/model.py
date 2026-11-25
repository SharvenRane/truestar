"""A DeepSeek V3 style decoder with multi head latent attention and a latent KV cache.

Notation per layer (row vectors, ``nn.Linear`` computes x W^T), H heads, r = ``kv_rank``,
dn = ``nope``, dr = ``rope``, dv = ``v_dim``:

* queries: ``q = q_b(norm(q_a(x)))`` (or ``q_proj(x)`` without a query rank), split per head into
  q_nope [dn] and q_rope [dr]; RoPE acts on q_rope only (the decoupled RoPE).
* latent: ``kv_a(x)`` splits into c [r] (normed) and k_rope [dr] (RoPE, one per token, shared by
  every head). The cache holds c and k_rope: ``r + dr`` numbers per token and layer.
* expanded attention (Hugging Face's computation): ``kv_b(c)`` gives per head k_nope [dn] and v
  [dv]; keys are ``[k_nope, k_rope]``, ``softmax(q k^T * scale) v``.
* absorbed attention: with ``W_kv_b`` viewed as ``[H, dn + dv, r]``, ``W_uk = W[:, :dn]`` and
  ``W_uv = W[:, dn:]``, ``q_nope . (W_uk c) = (q_nope W_uk) . c``, so each head's query moves into
  the latent space (``q_lat = q_nope W_uk``, [r]) and the attention runs on the cache itself as one
  shared KV head: keys ``[c, k_rope]`` [r + dr], values c [r]. The head output is
  ``W_uv (sum_s p_s c_s)``. Equal in real arithmetic; ``prove.prove_absorption`` checks it in
  float64 before a compiled absorbed step is built.

Interleaved RoPE (``rope_interleave``, DeepSeek's checkpoint layout) is folded into the weights:
the rope rows of q_b and kv_a are permuted to even entries first, then odd ones, after which the
half split rotation computes the same products (``a - b`` and ``a + (-b)`` round the same).

The step functions take a fixed capacity cache (``cc [L, B, S, r]``, ``rc [L, B, S, dr]``), int64
positions ``pos [T]`` (the same for every sequence) and write the new entries with ``index_copy``;
keys past a query's position are masked, so unwritten slots never count. One step serves the
prompt (T tokens) and every decode step (T = 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

MODES = ("expand", "absorb", "fused")


@dataclass(frozen=True)
class MLAConfig:
    vocab: int = 96
    dim: int = 32
    layers: int = 2
    heads: int = 4
    q_rank: Optional[int] = 24     # None: one q_proj, no query compression
    kv_rank: int = 16
    nope: int = 8
    rope: int = 4
    v_dim: int = 6
    ffn: int = 64
    eps: float = 1e-6              # the layer norms (input, post attention, final)
    lat_eps: float = 1e-6          # the q_a and kv_a norms (Hugging Face builds them with the default)
    theta: float = 10000.0
    scaling: Optional[float] = None   # softmax scale; None is (nope + rope) ** -0.5
    rope_scale: float = 1.0        # cos and sin are multiplied by it (YaRN's attention scaling)
    bias: bool = False             # attention_bias: q_a, kv_a and o carry a bias

    @property
    def scale(self) -> float:
        return self.scaling if self.scaling is not None else (self.nope + self.rope) ** -0.5

    def cache_bytes(self, batch: int, tokens: int, itemsize: int = 4) -> Tuple[int, int]:
        """(latent cache, full K/V cache) bytes for ``tokens`` positions of ``batch`` sequences:
        the latent cache holds ``r + dr`` per token and layer, a full cache ``H (dn + dr + dv)``."""
        per = batch * tokens * self.layers * itemsize
        return per * (self.kv_rank + self.rope), per * self.heads * (self.nope + self.rope + self.v_dim)


def rms_norm(x, w, eps):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class MLALayer(nn.Module):
    def __init__(self, c: MLAConfig):
        super().__init__()
        h = c.heads
        self.norm1 = nn.Parameter(torch.ones(c.dim))
        self.norm2 = nn.Parameter(torch.ones(c.dim))
        if c.q_rank is None:
            self.q = nn.Linear(c.dim, h * (c.nope + c.rope), bias=False)
        else:
            self.q_a = nn.Linear(c.dim, c.q_rank, bias=c.bias)
            self.q_norm = nn.Parameter(torch.ones(c.q_rank))
            self.q_b = nn.Linear(c.q_rank, h * (c.nope + c.rope), bias=False)
        self.kv_a = nn.Linear(c.dim, c.kv_rank + c.rope, bias=c.bias)
        self.kv_norm = nn.Parameter(torch.ones(c.kv_rank))
        self.kv_b = nn.Linear(c.kv_rank, h * (c.nope + c.v_dim), bias=False)
        self.o = nn.Linear(h * c.v_dim, c.dim, bias=c.bias)
        self.gate = nn.Linear(c.dim, c.ffn, bias=False)
        self.up = nn.Linear(c.dim, c.ffn, bias=False)
        self.down = nn.Linear(c.ffn, c.dim, bias=False)


class MLAModel(nn.Module):
    """Embedding, ``layers`` x (RMSNorm, MLA, RMSNorm, SwiGLU MLP), RMSNorm, lm head."""

    def __init__(self, c: MLAConfig):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab, c.dim)
        self.blocks = nn.ModuleList([MLALayer(c) for _ in range(c.layers)])
        self.norm = nn.Parameter(torch.ones(c.dim))
        self.lm_head = nn.Linear(c.dim, c.vocab, bias=False)
        inv = 1.0 / (c.theta ** (torch.arange(0, c.rope, 2, dtype=torch.int64).float() / c.rope))
        self.register_buffer("inv_freq", inv)

    # ------------------------------------------------------------------ pieces
    def rope_tables(self, pos):
        f = pos.to(self.inv_freq.dtype)[:, None] * self.inv_freq
        e = torch.cat((f, f), dim=-1)
        return e.cos() * self.c.rope_scale, e.sin() * self.c.rope_scale    # [T, dr]

    def queries(self, blk, h, cos, sin):
        c = self.c
        b, t, _ = h.shape
        q = blk.q(h) if c.q_rank is None else blk.q_b(rms_norm(blk.q_a(h), blk.q_norm, c.lat_eps))
        q = q.view(b, t, c.heads, c.nope + c.rope).transpose(1, 2)
        q_nope, q_rope = q.split([c.nope, c.rope], dim=-1)
        return q_nope, q_rope * cos + rotate_half(q_rope) * sin

    def latent(self, blk, h, cos, sin):
        c = self.c
        lat, kr = blk.kv_a(h).split([c.kv_rank, c.rope], dim=-1)
        return rms_norm(lat, blk.kv_norm, c.lat_eps), kr * cos + rotate_half(kr) * sin   # [B, T, r], [B, T, dr]

    def absorbed(self, blk) -> Tuple[torch.Tensor, torch.Tensor]:
        """(W_uk [H, dn, r], W_uv [H, dv, r]): kv_b's weight split per head into its key and value parts."""
        c = self.c
        w = blk.kv_b.weight.view(c.heads, c.nope + c.v_dim, c.kv_rank)
        return w[:, :c.nope], w[:, c.nope:]

    def attend(self, blk, q_nope, q_rope, cc, rc, keep, mode: str):
        """Attention of the queries [B, H, T, .] over the latent cache cc [B, S, r], rc [B, S, dr]."""
        c = self.c
        b, s = cc.shape[0], cc.shape[1]
        if mode == "expand":
            kv = blk.kv_b(cc).view(b, s, c.heads, c.nope + c.v_dim).transpose(1, 2)
            k_nope, v = kv.split([c.nope, c.v_dim], dim=-1)
            k = torch.cat((k_nope, rc[:, None].expand(b, c.heads, s, c.rope)), dim=-1)
            q = torch.cat((q_nope, q_rope), dim=-1)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=keep, scale=c.scale)
        if mode == "absorb":
            w_uk, w_uv = self.absorbed(blk)
            q = torch.cat((q_nope @ w_uk, q_rope), dim=-1)                  # [B, H, T, r + dr]
            k = torch.cat((cc, rc), dim=-1)[:, None]                         # [B, 1, S, r + dr]
            o = F.scaled_dot_product_attention(q, k, cc[:, None], attn_mask=keep, scale=c.scale,
                                               enable_gqa=True)              # [B, H, T, r]
            return o @ w_uv.transpose(1, 2)                                  # [B, H, T, dv]
        raise ValueError(f"[TS1200] unknown MLA mode {mode!r}; the modes are {MODES}")

    def finish(self, blk, x, a):
        c = self.c
        b, t, _ = x.shape
        x = x + blk.o(a.transpose(1, 2).reshape(b, t, c.heads * c.v_dim))
        y = rms_norm(x, blk.norm2, c.eps)
        return x + blk.down(F.silu(blk.gate(y)) * blk.up(y))

    # ------------------------------------------------------------------ steps
    def step(self, ids, pos, cc, rc, mode: str = "expand", record: Optional[List] = None):
        """One step: ids [B, T], pos int64 [T], cc [L, B, S, r], rc [L, B, S, dr]. Returns
        (logits [B, T, V], cc, rc) with the new entries written at ``pos``. ``record`` collects
        each layer's attention output (``prove_absorption``)."""
        if mode == "fused":
            return self._step_fused(ids, pos, cc, rc, record)
        cos, sin = self.rope_tables(pos)
        x = self.embed(ids)
        s = cc.shape[2]
        keep = (torch.arange(s, device=pos.device)[None, :] <= pos[:, None]).view(1, 1, -1, s)
        ccs, rcs = [], []
        for i, blk in enumerate(self.blocks):
            h = rms_norm(x, blk.norm1, self.c.eps)
            q_nope, q_rope = self.queries(blk, h, cos, sin)
            lat, kr = self.latent(blk, h, cos, sin)
            cci = cc[i].index_copy(1, pos, lat)
            rci = rc[i].index_copy(1, pos, kr)
            a = self.attend(blk, q_nope, q_rope, cci, rci, keep, mode)
            if record is not None:
                record.append(a)
            x = self.finish(blk, x, a)
            ccs.append(cci)
            rcs.append(rci)
        return self.lm_head(rms_norm(x, self.norm, self.c.eps)), torch.stack(ccs), torch.stack(rcs)

    def _step_fused(self, ids, pos, cc, rc, record=None):
        """Mode ``"fused"``: each layer attends with ``torch.ops.truestar.mla_attention`` (mla/latent.py)
        over its cache as it was and this step's new latents; the new latents of every layer are
        written into the caches once, at the end (two ``index_copy`` instead of two per layer plus a
        stack). The same attention as ``"absorb"`` for consecutive positions ``pos``."""
        from . import latent  # noqa: F401  (registers torch.ops.truestar.mla_attention)
        c = self.c
        cos, sin = self.rope_tables(pos)
        x = self.embed(ids)
        lats, krs = [], []
        for i, blk in enumerate(self.blocks):
            h = rms_norm(x, blk.norm1, c.eps)
            q_nope, q_rope = self.queries(blk, h, cos, sin)
            lat, kr = self.latent(blk, h, cos, sin)
            w_uk, w_uv = self.absorbed(blk)
            a = torch.ops.truestar.mla_attention(q_nope, q_rope, cc[i], rc[i], lat, kr, pos, w_uk, w_uv, c.scale)
            if record is not None:
                record.append(a)
            x = self.finish(blk, x, a)
            lats.append(lat)
            krs.append(kr)
        cc = cc.index_copy(2, pos, torch.stack(lats))
        rc = rc.index_copy(2, pos, torch.stack(krs))
        return self.lm_head(rms_norm(x, self.norm, c.eps)), cc, rc

    def empty_cache(self, batch: int, capacity: int, dtype=None, device=None):
        c = self.c
        dtype = dtype or self.norm.dtype
        cc = torch.zeros(c.layers, batch, capacity, c.kv_rank, dtype=dtype, device=device)
        return cc, torch.zeros(c.layers, batch, capacity, c.rope, dtype=dtype, device=device)


class Step(nn.Module):
    """``model.step`` in one mode, as a module torch.export can take."""

    def __init__(self, model: MLAModel, mode: str):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"[TS1200] unknown MLA mode {mode!r}; the modes are {MODES}")
        self.model, self.mode = model, mode

    def forward(self, ids, pos, cc, rc):
        return self.model.step(ids, pos, cc, rc, self.mode)


def make_model(c: MLAConfig = MLAConfig(), seed: int = 0, gain: float = 3.0) -> MLAModel:
    """Random weights scaled so attention and the MLP matter (a default init decodes one token
    over and over and would not exercise the cache)."""
    torch.manual_seed(seed)
    m = MLAModel(c).eval()
    with torch.no_grad():
        m.embed.weight.normal_(0.0, 0.3)
        for blk in m.blocks:
            for p in (blk.norm1, blk.norm2, blk.kv_norm) + ((blk.q_norm,) if c.q_rank is not None else ()):
                p.normal_(1.0, 0.2)
            for mod in blk.modules():
                if isinstance(mod, nn.Linear):
                    mod.weight.mul_(gain)
    return m


def generate(step, model_cfg: MLAConfig, prompt: torch.Tensor, steps: int, capacity: int,
             decode=None, device="cpu", dtype=torch.float64):
    """Greedy decoding: ``step(ids, pos, cc, rc)`` runs the prompt, ``decode`` (default ``step``)
    every later token. Returns (tokens [B, steps], last position logits [steps, B, V])."""
    c = model_cfg
    decode = decode or step
    b, t0 = prompt.shape
    cc = torch.zeros(c.layers, b, capacity, c.kv_rank, dtype=dtype, device=device)
    rc = torch.zeros(c.layers, b, capacity, c.rope, dtype=dtype, device=device)
    ids, pos = prompt.to(device), torch.arange(t0, device=device)
    toks, lg = [], []
    fn = step
    for s in range(steps):
        logits, cc, rc = fn(ids, pos, cc, rc)
        last = logits[:, -1]
        nxt = last.argmax(-1)
        toks.append(nxt)
        lg.append(last)
        ids, pos, fn = nxt[:, None], torch.tensor([t0 + s], device=device), decode
    return torch.stack(toks, 1), torch.stack(lg)


def rope_permutation(dr: int) -> torch.Tensor:
    """Even entries first, then odd ones: interleaved RoPE pairs to the half split layout."""
    return torch.cat((torch.arange(0, dr, 2), torch.arange(1, dr, 2)))


__all__ = ["MLAConfig", "MLAModel", "MLALayer", "Step", "MODES", "make_model", "generate", "rope_permutation",
           "rms_norm", "rotate_half"]
