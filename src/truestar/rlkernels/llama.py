"""A small Llama style decoder built in code, with the three paths an RL stack runs.

RMSNorm, rotary positions, grouped query attention through `F.scaled_dot_product_attention`
(4 query heads on 2 KV heads by default), a SwiGLU MLP and an untied lm head; random weights, no
download. Plain `torch.nn`: under `rl_mode()` the same code runs on the RL kernel set.

* `train_logprobs`: the trainer's teacher forced forward over whole sequences, right padded,
  causal attention without a mask.
* `prefill_logprobs`: the sampler's prefill of whole sequences into a KV cache, left padded,
  with an explicit key padding mask (Hugging Face style position ids from the mask).
* `decode_logprobs`: the sampler's rollout: prefill of the prompts (optionally in chunks), then
  one token per step against the cache, rows in lockstep, finished rows fed a masked pad.
  `cache="dynamic"` appends to the cache, `cache="static"` writes into a preallocated buffer of
  fixed capacity whose unwritten slots are masked.
* `tp_train_logits`, `tp_train_logprobs`: the trainer forward with the weights split over `tp` simulated tensor
  parallel ranks on one device (see `tp.py`). It is a simulation: one process, one GPU, no
  communication library.

Every function returns, per row, the logprobs of tokens 1..T-1 of that row (float32 [T - 1]).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from . import kernels as K
from . import tp as TP


@dataclass
class Config:
    vocab: int = 512
    dim: int = 128
    layers: int = 2
    heads: int = 4
    kv_heads: int = 2
    hidden: int = 256
    max_len: int = 256
    theta: float = 10000.0
    eps: float = 1e-5


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Rotary(nn.Module):
    def __init__(self, hd: int, theta: float):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        self.register_buffer("inv", inv, persistent=False)

    def forward(self, pos: torch.Tensor):
        f = pos[..., None].float() * self.inv          # elementwise: same bits for a position anywhere
        e = torch.cat((f, f), dim=-1)
        return e.cos(), e.sin()


class KVCache:
    """Per layer K and V. "dynamic" concatenates; "static" writes into a zeroed buffer of fixed
    capacity, so the attention reads `capacity` keys with the unwritten ones masked."""

    def __init__(self, layers: int, layout: str = "dynamic", capacity: int = 0):
        if layout not in ("dynamic", "static"):
            raise ValueError("layout is 'dynamic' or 'static'")
        self.layout, self.capacity = layout, capacity
        self.k: List[Optional[torch.Tensor]] = [None] * layers
        self.v: List[Optional[torch.Tensor]] = [None] * layers
        self.seen = 0  # columns written so far (same for every layer once a step finishes)

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor, start: int):
        L = k.shape[2]
        if self.layout == "dynamic":
            if self.k[layer] is None:
                self.k[layer], self.v[layer] = k, v
            else:
                self.k[layer] = torch.cat([self.k[layer], k], dim=2)
                self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        else:
            if self.k[layer] is None:
                shape = (k.shape[0], k.shape[1], self.capacity, k.shape[3])
                self.k[layer] = k.new_zeros(shape)
                self.v[layer] = v.new_zeros(shape)
            if start + L > self.capacity:
                raise ValueError(f"static cache capacity {self.capacity} exceeded")
            self.k[layer][:, :, start:start + L] = k
            self.v[layer][:, :, start:start + L] = v
        return self.k[layer], self.v[layer]


class Attention(nn.Module):
    def __init__(self, c: Config, layer: int):
        super().__init__()
        self.layer, self.h, self.hk, self.hd = layer, c.heads, c.kv_heads, c.dim // c.heads
        self.q_proj = nn.Linear(c.dim, c.heads * self.hd, bias=False)
        self.k_proj = nn.Linear(c.dim, c.kv_heads * self.hd, bias=False)
        self.v_proj = nn.Linear(c.dim, c.kv_heads * self.hd, bias=False)
        self.o_proj = nn.Linear(c.heads * self.hd, c.dim, bias=False)
        self.decode_impl = "sdpa"  # "manual": a hand written attention in one token decode steps

    def project(self, x, cos, sin):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, -1, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, L, -1, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, L, -1, self.hd).transpose(1, 2)
        cos, sin = cos[:, None].to(q.dtype), sin[:, None].to(q.dtype)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin, v

    def forward(self, x, cos, sin, mask=None, cache: Optional[KVCache] = None):
        B, L, _ = x.shape
        q, k, v = self.project(x, cos, sin)
        if cache is not None:
            k, v = cache.update(self.layer, k, v, cache.seen)
        if mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        elif self.decode_impl == "manual" and L == 1:
            # a hand written decode attention, as engines ship: same math, other reduction order
            rep = self.h // self.hk
            kk, vv = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
            sc = (q @ kk.transpose(-1, -2)) * (self.hd ** -0.5)
            y = torch.softmax(sc.masked_fill(~mask, float("-inf")), dim=-1) @ vv
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        return self.o_proj(y.transpose(1, 2).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.gate_proj = nn.Linear(c.dim, c.hidden, bias=False)
        self.up_proj = nn.Linear(c.dim, c.hidden, bias=False)
        self.down_proj = nn.Linear(c.hidden, c.dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, c: Config, layer: int):
        super().__init__()
        self.input_norm = nn.RMSNorm(c.dim, eps=c.eps)
        self.attn = Attention(c, layer)
        self.post_norm = nn.RMSNorm(c.dim, eps=c.eps)
        self.mlp = MLP(c)

    def forward(self, x, cos, sin, mask=None, cache=None):
        x = x + self.attn(self.input_norm(x), cos, sin, mask, cache)
        return x + self.mlp(self.post_norm(x))


class Llama(nn.Module):
    def __init__(self, c: Optional[Config] = None):
        super().__init__()
        c = c or Config()
        self.c = c
        self.embed = nn.Embedding(c.vocab, c.dim)
        self.rotary = Rotary(c.dim // c.heads, c.theta)
        self.blocks = nn.ModuleList(Block(c, i) for i in range(c.layers))
        self.norm = nn.RMSNorm(c.dim, eps=c.eps)
        self.lm_head = nn.Linear(c.dim, c.vocab, bias=False)

    def forward(self, ids: torch.Tensor, pos: Optional[torch.Tensor] = None,
                key_valid: Optional[torch.Tensor] = None, cache: Optional[KVCache] = None):
        """ids [B, L]. Without `key_valid` and cache: the causal training forward (positions
        0..L-1). With them: `key_valid` [B, columns so far + L] (True = a real token), `pos` the
        rotary positions of the L new tokens, and the cache gets the new K and V."""
        B, L = ids.shape
        if pos is None:
            pos = torch.arange(L, device=ids.device).expand(B, L)
        cos, sin = self.rotary(pos)
        mask = None
        if key_valid is not None:
            start = cache.seen if cache is not None else 0
            S = start + L
            cols = S if cache is None or cache.layout == "dynamic" else cache.capacity
            kv = key_valid
            if kv.shape[1] < cols:
                kv = torch.cat([kv, kv.new_zeros(B, cols - kv.shape[1])], dim=1)
            j = torch.arange(cols, device=ids.device)
            qcol = start + torch.arange(L, device=ids.device)
            mask = (kv[:, None, None, :] & (j[None, None, None, :] <= qcol[None, None, :, None]))
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x, cos, sin, mask, cache)
        if cache is not None:
            cache.seen += L
        return self.lm_head(self.norm(x))


def make_model(seed: int = 0, device: str = "cuda", dtype: torch.dtype = torch.float32,
               config: Optional[Config] = None) -> Llama:
    torch.manual_seed(seed)
    m = Llama(config)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.normal_(0.0, 0.06)
        m.embed.weight.normal_(0.0, 1.0)
    return m.to(device=device, dtype=dtype).eval()


def make_rows(n: int, seed: int, vocab: int, min_len: int = 6, max_len: int = 24) -> List[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        L = int(torch.randint(min_len, max_len + 1, (1,), generator=g))
        out.append(torch.randint(0, vocab, (L,), generator=g))
    return out


def _lp(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log_softmax over the vocab in float32, then the targets' entries (as `truestar rlpair`
    computes them from logits). Runs routed under rl_mode."""
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, targets[..., None]).squeeze(-1)


PAD = 0


@torch.no_grad()
def train_logprobs(model: Llama, rows: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Teacher forced forward over the right padded batch (causal, no mask)."""
    dev = next(model.parameters()).device
    T = max(len(r) for r in rows)
    ids = torch.full((len(rows), T), PAD, dtype=torch.long)
    for b, r in enumerate(rows):
        ids[b, :len(r)] = r
    ids = ids.to(dev)
    logits = model(ids)
    lp = _lp(logits[:, :-1], ids[:, 1:])
    return [lp[b, :len(r) - 1] for b, r in enumerate(rows)]


def _left_pad(rows: Sequence[torch.Tensor], dev):
    T = max(len(r) for r in rows)
    ids = torch.full((len(rows), T), PAD, dtype=torch.long)
    valid = torch.zeros((len(rows), T), dtype=torch.bool)
    for b, r in enumerate(rows):
        ids[b, T - len(r):] = r
        valid[b, T - len(r):] = True
    return ids.to(dev), valid.to(dev)


def _positions(valid: torch.Tensor) -> torch.Tensor:
    # Hugging Face style: the running count of real tokens, pads at 0
    return (valid.long().cumsum(-1) - 1).clamp(min=0)


@torch.no_grad()
def prefill_logprobs(model: Llama, rows: Sequence[torch.Tensor], cache: str = "dynamic") -> List[torch.Tensor]:
    """One prefill of the whole left padded sequences into a KV cache, with a key padding mask."""
    dev = next(model.parameters()).device
    ids, valid = _left_pad(rows, dev)
    T = ids.shape[1]
    kc = KVCache(model.c.layers, cache, capacity=T + 8)
    logits = model(ids, _positions(valid), valid, kc)
    lp = _lp(logits[:, :-1], ids[:, 1:])
    return [lp[b, T - len(r):] for b, r in enumerate(rows)]


@torch.no_grad()
def decode_logprobs(model: Llama, rows: Sequence[torch.Tensor], prompt_lens: Sequence[int],
                    cache: str = "dynamic", chunk: Optional[int] = None) -> List[torch.Tensor]:
    """Prefill the left padded prompts (in chunks of `chunk` tokens if given), then decode one
    token per step against the cache, feeding each row's own next token. Returns, per row, the
    logprobs of tokens 1..T-1 (the prompt's from the prefill, the completion's from decoding)."""
    dev = next(model.parameters()).device
    prompts = [r[:p] for r, p in zip(rows, prompt_lens)]
    pids, pvalid = _left_pad(prompts, dev)
    B, P = pids.shape
    steps = max(len(r) - p for r, p in zip(rows, prompt_lens))
    kc = KVCache(model.c.layers, cache, capacity=P + steps + 3)
    valid_all = pvalid
    pos_all = _positions(pvalid)
    chunks = [(s, min(P, s + chunk)) for s in range(0, P, chunk)] if chunk else [(0, P)]
    outs = []
    for s, e in chunks:
        logits = model(pids[:, s:e], pos_all[:, s:e], valid_all[:, :e], kc)
        outs.append(logits)
    plog = torch.cat(outs, dim=1)
    prompt_lp = _lp(plog[:, :-1], pids[:, 1:])
    per_row = [[prompt_lp[b, P - p:]] for b, p in enumerate(prompt_lens)]
    last = plog[:, -1]
    next_pos = pos_all[:, -1] + 1
    for t in range(steps):
        tok = torch.full((B,), PAD, dtype=torch.long)
        live = torch.zeros(B, dtype=torch.bool)
        for b, (r, p) in enumerate(zip(rows, prompt_lens)):
            if p + t < len(r):
                tok[b], live[b] = r[p + t], True
        tok, live = tok.to(dev), live.to(dev)
        lp_t = _lp(last, tok)
        for b in range(B):
            if bool(live[b]):
                per_row[b].append(lp_t[b:b + 1])
        valid_all = torch.cat([valid_all, live[:, None]], dim=1)
        logits = model(tok[:, None], next_pos[:, None], valid_all, kc)
        last = logits[:, -1]
        next_pos = next_pos + live.long()
    return [torch.cat(x) for x in per_row]


# ----------------------------------------------------------------------------- tensor parallel


@torch.no_grad()
def tp_train_logprobs(model: Llama, rows: Sequence[torch.Tensor], tp: int) -> List[torch.Tensor]:
    """`train_logprobs` with the weights split over `tp` simulated ranks (`tp_train_logits`)."""
    dev = next(model.parameters()).device
    T = max(len(r) for r in rows)
    ids = torch.full((len(rows), T), PAD, dtype=torch.long)
    for b, r in enumerate(rows):
        ids[b, :len(r)] = r
    ids = ids.to(dev)
    logits = tp_train_logits(model, ids, tp)
    lp = _lp(logits[:, :-1], ids[:, 1:])
    return [lp[b, :len(r) - 1] for b, r in enumerate(rows)]


@torch.no_grad()
def tp_train_logits(model: Llama, ids: torch.Tensor, tp: int) -> torch.Tensor:
    """The trainer forward with every weight split Megatron style over `tp` simulated ranks.

    Column parallel (no reduction changes): q, k, v, gate and up projections, the lm head (vocab
    split, gathered by concatenation). Row parallel (each rank holds a contiguous 1/tp of K):
    o_proj and down_proj; each rank returns float32 partials over its K segments and
    `tp.all_reduce` adds them in a fixed binary tree. Attention runs per rank on its heads.
    Must run under `rl_mode()` (norms, attention and log_softmax route to the RL kernels).
    SIMULATION: all ranks run in this process on one device."""
    c = model.c
    if c.heads % tp or c.kv_heads % tp or c.hidden % tp or c.vocab % tp:
        raise ValueError(f"tp {tp} does not divide heads, kv heads, hidden size and vocab")
    if K.SEGMENTS % tp:
        raise ValueError(f"tp {tp} must divide the kernel's K segment count {K.SEGMENTS}")
    dev = next(model.parameters()).device
    ids = ids.to(dev)
    B, T = ids.shape
    hd = c.dim // c.heads
    cos, sin = model.rotary(torch.arange(T, device=dev).expand(B, T))
    x = model.embed(ids)
    for blk in model.blocks:
        a = blk.attn
        h = blk.input_norm(x)
        ys = []
        for r in range(tp):
            qw = TP.shard_rows(a.q_proj.weight, tp, r)
            kw = TP.shard_rows(a.k_proj.weight, tp, r)
            vw = TP.shard_rows(a.v_proj.weight, tp, r)
            q = F.linear(h, qw).view(B, T, -1, hd).transpose(1, 2)
            k = F.linear(h, kw).view(B, T, -1, hd).transpose(1, 2)
            v = F.linear(h, vw).view(B, T, -1, hd).transpose(1, 2)
            cs, sn = cos[:, None].to(q.dtype), sin[:, None].to(q.dtype)
            q, k = q * cs + _rotate_half(q) * sn, k * cs + _rotate_half(k) * sn
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            ys.append(y.transpose(1, 2).reshape(B, T, -1))
        x = x + TP.row_parallel_linear(ys, a.o_proj.weight, tp)
        h = blk.post_norm(x)
        ms = []
        for r in range(tp):
            g = F.linear(h, TP.shard_rows(blk.mlp.gate_proj.weight, tp, r))
            u = F.linear(h, TP.shard_rows(blk.mlp.up_proj.weight, tp, r))
            ms.append(F.silu(g) * u)
        x = x + TP.row_parallel_linear(ms, blk.mlp.down_proj.weight, tp)
    h = model.norm(x)
    return TP.all_gather([F.linear(h, TP.shard_rows(model.lm_head.weight, tp, r)) for r in range(tp)])
