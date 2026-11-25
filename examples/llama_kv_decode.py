"""A Llama style decoder written natively with Truestar, decoded greedily with a KV cache.

Run: ``python examples/llama_kv_decode.py``

One ``@truestar.function`` (``forward``) serves the prompt (T tokens) and every decode step (T = 1):
the batch ``B``, the token count ``T`` and the cache capacity ``S`` are symbolic, and the positions
are data (an int64 tensor), so the cache write ``ops.index_copy(cache, 2, pos, k)`` lands at a run
time position. It compiles once. The script checks it token by token against a plain ``torch.nn``
model of the same architecture (RMSNorm, rotary positions, grouped query attention, SwiGLU, lm head
tied to the embedding; random weights) that recomputes the whole sequence at every step in float64.

Everything here is float64: the reference interpreter computes in float64, and keeping the cache in
float64 between steps means the comparison is exact up to float64 rounding.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import truestar
from truestar import Tensor, ops
from truestar.frontend import compile_count

V, D, HQ, HKV, HD, L, FF = 64, 32, 4, 2, 8, 2, 96  # vocab, width, query heads, KV heads, head dim, layers, MLP
EPS = 1e-6
THETA = 10000.0


# ----------------------------------------------------------------------------- plain torch.nn reference


class RMSNorm(nn.Module):
    def __init__(self, d, eps=EPS):
        super().__init__()
        self.weight = nn.Parameter(1 + 0.2 * torch.randn(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(D), RMSNorm(D)
        self.q = nn.Linear(D, HQ * HD, bias=False)
        self.k = nn.Linear(D, HKV * HD, bias=False)
        self.v = nn.Linear(D, HKV * HD, bias=False)
        self.o = nn.Linear(HQ * HD, D, bias=False)
        self.gate = nn.Linear(D, FF, bias=False)
        self.up = nn.Linear(D, FF, bias=False)
        self.down = nn.Linear(FF, D, bias=False)

    def qkv(self, x, cos, sin):
        b, t, _ = x.shape
        h = self.norm1(x)
        q = self.q(h).view(b, t, HQ, HD).transpose(1, 2)
        k = self.k(h).view(b, t, HKV, HD).transpose(1, 2)
        v = self.v(h).view(b, t, HKV, HD).transpose(1, 2)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin, v

    def finish(self, x, a):
        b, t, _ = x.shape
        x = x + self.o(a.transpose(1, 2).reshape(b, t, HQ * HD))
        y = self.norm2(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))  # SwiGLU


class Llama(nn.Module):
    """Embedding, rotary positions, grouped query attention (HQ query heads share HKV KV heads),
    RMSNorm, SwiGLU, and an lm head tied to the embedding."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.norm = RMSNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.embed.weight
        self.register_buffer("inv_freq", 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float64) / HD)).float())

    def rope(self, pos):
        f = pos.to(self.inv_freq.dtype)[:, None] * self.inv_freq[None, :]
        e = torch.cat((f, f), dim=-1)
        return e.cos(), e.sin()

    def forward(self, ids):
        """Full sequence, no cache: logits [B, T, V] and the K and V of every layer."""
        cos, sin = self.rope(torch.arange(ids.shape[1]))
        x = self.embed(ids)
        kv = []
        for blk in self.blocks:
            q, k, v = blk.qkv(x, cos, sin)
            x = blk.finish(x, F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True))
            kv += [k, v]
        return (self.lm_head(self.norm(x)),) + tuple(kv)

    def step(self, ids, pos, *cache):
        """One decode step with a fixed capacity cache per layer ([B, HKV, S, HD]): the new K and V
        are written at ``pos`` (int64 [T]) and the queries see the cache up to their own position."""
        cos, sin = self.rope(pos)
        x = self.embed(ids)
        keep = (torch.arange(cache[0].shape[2])[None, :] <= pos[:, None]).view(1, 1, -1, cache[0].shape[2])
        out = []
        for i, blk in enumerate(self.blocks):
            q, k, v = blk.qkv(x, cos, sin)
            kc = cache[2 * i].index_copy(2, pos, k)
            vc = cache[2 * i + 1].index_copy(2, pos, v)
            x = blk.finish(x, F.scaled_dot_product_attention(q, kc, vc, attn_mask=keep, enable_gqa=True))
            out += [kc, vc]
        return (self.lm_head(self.norm(x)),) + tuple(out)


def make_model(seed: int = 0) -> Llama:
    """Random weights. The block weights are scaled up and the embedding down so attention and the
    MLP outweigh the tied embedding: with the default init greedy decoding repeats one token forever,
    and a broken cache could then go unnoticed."""
    torch.manual_seed(seed)
    model = Llama().eval()
    with torch.no_grad():
        model.embed.weight.normal_(0.0, 0.3, generator=torch.Generator().manual_seed(seed + 3))
        for blk in model.blocks:
            for lin in (blk.q, blk.k, blk.v, blk.o, blk.gate, blk.up, blk.down):
                lin.weight.mul_(3.0)
    return model


# ----------------------------------------------------------------------------- the native version


@truestar.function
def rope(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + ops.cat([-x[..., half:], x[..., :half]], dim=-1) * sin


@truestar.function
def block(x, cos, sin, kc, vc, pos, keep, n1, wq, wk, wv, wo, n2, wg, wu, wd):
    b = x.shape[0]
    t = x.shape[1]
    h = ops.rms_norm(x, [D], n1, eps=EPS)
    q = ops.linear(h, wq).reshape((b, t, HQ, HD)).transpose(1, 2)
    k = ops.linear(h, wk).reshape((b, t, HKV, HD)).transpose(1, 2)
    v = ops.linear(h, wv).reshape((b, t, HKV, HD)).transpose(1, 2)
    q = rope(q, cos, sin)
    k = rope(k, cos, sin)
    kc = ops.index_copy(kc, 2, pos, k)  # the cache write at run time positions
    vc = ops.index_copy(vc, 2, pos, v)
    a = ops.attention(q, kc, vc, mask=keep, enable_gqa=True)
    x = x + ops.linear(a.transpose(1, 2).reshape((b, t, HQ * HD)), wo)
    y = ops.rms_norm(x, [D], n2, eps=EPS)
    return x + ops.linear(ops.silu(ops.linear(y, wg)) * ops.linear(y, wu), wd), kc, vc


@truestar.function
def forward(ids: Tensor["B", "T", "i64"], pos: Tensor["T", "i64"],
            k_cache: Tensor[L, "B", HKV, "S", HD, "f64"], v_cache: Tensor[L, "B", HKV, "S", HD, "f64"],
            embed: Tensor[V, D, "f64"], inv_freq: Tensor[HD // 2, "f64"], norm: Tensor[D, "f64"],
            n1: Tensor[L, D, "f64"], wq: Tensor[L, HQ * HD, D, "f64"], wk: Tensor[L, HKV * HD, D, "f64"],
            wv: Tensor[L, HKV * HD, D, "f64"], wo: Tensor[L, D, HQ * HD, "f64"], n2: Tensor[L, D, "f64"],
            wg: Tensor[L, FF, D, "f64"], wu: Tensor[L, FF, D, "f64"], wd: Tensor[L, D, FF, "f64"]
            ) -> (Tensor["B", "T", V, "f64"], Tensor["B", "i64"],
                  Tensor[L, "B", HKV, "S", HD, "f64"], Tensor[L, "B", HKV, "S", HD, "f64"]):
    """Logits for T new tokens at positions ``pos``, the greedy next token, and the updated caches."""
    f = pos.double()[:, None] * inv_freq[None, :]
    e = ops.cat([f, f], dim=-1)
    cos = ops.cos(e)
    sin = ops.sin(e)
    keep = ops.arange(k_cache.shape[3])[None, :] <= pos[:, None]  # query i sees keys up to pos[i]
    x = ops.embedding(embed, ids)
    for i in range(L):
        x, kc, vc = block(x, cos, sin, k_cache[i], v_cache[i], pos, keep, n1[i], wq[i], wk[i], wv[i], wo[i],
                          n2[i], wg[i], wu[i], wd[i])
        if i == 0:
            ks = kc[None]
            vs = vc[None]
        else:
            ks = ops.cat([ks, kc[None]], dim=0)
            vs = ops.cat([vs, vc[None]], dim=0)
    logits = ops.linear(ops.rms_norm(x, [D], norm, eps=EPS), embed)  # lm head tied to the embedding
    nxt = ops.argmax(logits[:, ids.shape[1] - 1], dim=-1)  # greedy sampling as an IR op
    return logits, nxt, ks, vs


def native_weights(model: Llama) -> Dict[str, torch.Tensor]:
    """The torch model's weights in the native function's layout (per layer tensors stacked), float64."""
    def stack(get):
        return torch.stack([get(b).detach() for b in model.blocks]).double()

    return dict(
        embed=model.embed.weight.detach().double(), inv_freq=model.inv_freq.double(),
        norm=model.norm.weight.detach().double(),
        n1=stack(lambda b: b.norm1.weight), wq=stack(lambda b: b.q.weight), wk=stack(lambda b: b.k.weight),
        wv=stack(lambda b: b.v.weight), wo=stack(lambda b: b.o.weight), n2=stack(lambda b: b.norm2.weight),
        wg=stack(lambda b: b.gate.weight), wu=stack(lambda b: b.up.weight), wd=stack(lambda b: b.down.weight))


def native_generate(model: Llama, prompt: torch.Tensor, steps: int, capacity: int):
    """Prompt then ``steps`` greedy tokens through the native function. Returns the generated ids
    [B, steps] and the logits of each new token [steps, B, V]."""
    w = native_weights(model)
    b, t0 = prompt.shape
    kc = torch.zeros(L, b, HKV, capacity, HD, dtype=torch.float64)
    vc = torch.zeros_like(kc)
    logits, nxt, kc, vc = forward(prompt, torch.arange(t0), kc, vc, **w)
    out, step_logits = [nxt], [logits[:, -1]]
    for s in range(steps - 1):
        logits, nxt, kc, vc = forward(nxt[:, None], torch.tensor([t0 + s]), kc, vc, **w)
        out.append(nxt)
        step_logits.append(logits[:, -1])
    return torch.stack(out, dim=1), torch.stack(step_logits)


def torch_generate(model: Llama, prompt: torch.Tensor, steps: int):
    """Greedy decoding with no cache: the float64 model recomputes the whole sequence every step."""
    m = copy.deepcopy(model).double()
    ids = prompt
    toks, step_logits = [], []
    with torch.no_grad():
        for _ in range(steps):
            last = m(ids)[0][:, -1]
            nxt = last.argmax(-1)
            toks.append(nxt)
            step_logits.append(last)
            ids = torch.cat([ids, nxt[:, None]], dim=1)
    return torch.stack(toks, dim=1), torch.stack(step_logits)


def main():
    model = make_model()
    prompt = torch.randint(0, V, (2, 5), generator=torch.Generator().manual_seed(1))
    before = compile_count()  # forward was compiled once, at decoration
    got, got_logits = native_generate(model, prompt, steps=16, capacity=24)
    want, want_logits = torch_generate(model, prompt, steps=16)
    err = ((got_logits - want_logits).abs().max() / want_logits.abs().max()).item()
    print(forward.signature)
    print("native tokens:", got.tolist())
    print("torch tokens: ", want.tolist())
    print(f"tokens equal: {torch.equal(got, want)}, logits max relative error {err:.2e}, "
          f"compiles during the 16 calls: {compile_count() - before}")


if __name__ == "__main__":
    main()
