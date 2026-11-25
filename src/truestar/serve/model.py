"""The model `truestar serve` runs: a small Llama style decoder built from a config in code.

Weights come from NumPy's PCG64 generator (frozen across NumPy versions), never from a download and
never from torch's generator, whose draws changed between torch releases. The one step program the
server runs is a paged KV cache step in which every token is its own row:

    step(ids [N, 1], pos [N, 1], slots [N, 1], table [N, M], k_pool, v_pool) -> logits [N, V], pools

* each row writes its token's K and V into the shared pool at ``slots`` (all rows of a layer are
  written before any row reads),
* each row gathers its own sequence's cache through its block table row, always ``M`` blocks wide,
  and attends with the mask ``arange(M * block) <= pos``.

A prefill chunk of a sequence is several rows of that sequence; a decode is one row. So one program
serves any mix of prefills, chunks and decodes, and a row's arithmetic never depends on which other
rows share the call: matrix products are per row, the attention of a row runs over the same
``M * block`` slots in the same order, and slots it may not see contribute exact zeros (softmax of a
masked score is 0, and 0 times a finite value adds nothing). That is the property `serve` promises
and checks bit for bit.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ServeConfig:
    vocab: int = 320          # bytes 0..255 plus 64 more ids
    dim: int = 64
    heads: int = 4
    kv_heads: int = 2
    head_dim: int = 16
    layers: int = 2
    ffn: int = 128
    eps: float = 1e-6
    theta: float = 10000.0
    block: int = 8            # tokens per KV block
    max_len: int = 128        # positions per sequence (prompt plus generated); table width = max_len / block
    n_blocks: int = 96        # KV blocks in the pool, shared by all sequences and the prefix cache
    seed: int = 0             # weight seed

    @property
    def width(self) -> int:
        return -(-self.max_len // self.block)


PRESETS: Dict[str, ServeConfig] = {
    "tiny": ServeConfig(),
    # the decode benchmark's shape (docs/LLM.md): vocab 8192, width 512, 4 layers, 8 query and 2 KV heads
    "bench": ServeConfig(vocab=8192, dim=512, heads=8, kv_heads=2, head_dim=64, layers=4, ffn=1408,
                         block=16, max_len=128, n_blocks=40 * 8 + 8),
}


def config(name_or_cfg, **over) -> ServeConfig:
    c = PRESETS[name_or_cfg] if isinstance(name_or_cfg, str) else name_or_cfg
    return replace(c, **over) if over else c


def make_weights(c: ServeConfig) -> Dict[str, np.ndarray]:
    """float64 weights from PCG64(seed). Linear weights are scaled up (std 1.7 / sqrt(fan in)) so
    attention and the MLP outweigh the embedding and greedy text does not repeat one token."""
    g = np.random.Generator(np.random.PCG64(c.seed))

    def lin(o, i):
        return g.standard_normal((o, i)) * (1.7 / np.sqrt(i))

    w = {"embed": g.standard_normal((c.vocab, c.dim)) * 0.3}
    for l in range(c.layers):
        p = f"l{l}."
        w[p + "n1"] = 1.0 + 0.2 * g.standard_normal(c.dim)
        w[p + "n2"] = 1.0 + 0.2 * g.standard_normal(c.dim)
        w[p + "q"] = lin(c.heads * c.head_dim, c.dim)
        w[p + "k"] = lin(c.kv_heads * c.head_dim, c.dim)
        w[p + "v"] = lin(c.kv_heads * c.head_dim, c.dim)
        w[p + "o"] = lin(c.dim, c.heads * c.head_dim)
        w[p + "gate"] = lin(c.ffn, c.dim)
        w[p + "up"] = lin(c.ffn, c.dim)
        w[p + "down"] = lin(c.dim, c.ffn)
    w["norm"] = 1.0 + 0.2 * g.standard_normal(c.dim)
    w["head"] = lin(c.vocab, c.dim)
    return w


def _rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _half_turn(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class TinyLM(nn.Module):
    """RMSNorm, rotary positions, grouped query attention, SwiGLU, untied head."""

    NAMES = ("n1", "n2", "q", "k", "v", "o", "gate", "up", "down")

    def __init__(self, c: ServeConfig, weights: Optional[Dict[str, np.ndarray]] = None):
        super().__init__()
        self.c = c
        w = make_weights(c) if weights is None else weights
        t = {k: torch.tensor(v, dtype=torch.float64) for k, v in w.items()}
        self.embed = nn.Parameter(t["embed"])
        self.norm = nn.Parameter(t["norm"])
        self.head = nn.Parameter(t["head"])
        self.layer_w = nn.ParameterList()
        for l in range(c.layers):
            for n in self.NAMES:
                self.layer_w.append(nn.Parameter(t[f"l{l}.{n}"]))
        inv = 1.0 / (c.theta ** (torch.arange(0, c.head_dim, 2, dtype=torch.float64) / c.head_dim))
        self.register_buffer("inv_freq", inv)

    def lw(self, l):
        k = len(self.NAMES)
        return dict(zip(self.NAMES, self.layer_w[l * k:(l + 1) * k]))

    def _rope(self, pos):
        f = pos.to(self.inv_freq.dtype)[..., None] * self.inv_freq
        e = torch.cat((f, f), dim=-1)
        return e.cos(), e.sin()

    def _layer(self, w, x, cos, sin, attend):
        """x [B, T, D]; cos, sin broadcast against [B, H, T, HD]; ``attend(q, k, v)`` -> [B, H, T, HD]."""
        c = self.c
        b, t, _ = x.shape
        h = _rms(x, w["n1"], c.eps)
        q = F.linear(h, w["q"]).view(b, t, c.heads, c.head_dim).transpose(1, 2)
        k = F.linear(h, w["k"]).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        v = F.linear(h, w["v"]).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        q = q * cos + _half_turn(q) * sin
        k = k * cos + _half_turn(k) * sin
        a = attend(q, k, v)
        x = x + F.linear(a.transpose(1, 2).reshape(b, t, c.heads * c.head_dim), w["o"])
        y = _rms(x, w["n2"], c.eps)
        return x + F.linear(F.silu(F.linear(y, w["gate"])) * F.linear(y, w["up"]), w["down"])

    def token_step(self, ids, pos, slots, table, kp, vp):
        """The server's step; see the module docstring. ids, pos, slots int64 [N, 1]; table [N, M];
        pools [L, P, HKV, HD]."""
        c = self.c
        n = ids.shape[0]
        m = table.shape[1]
        s = m * c.block
        cos, sin = self._rope(pos)                       # [N, 1, HD]
        cos, sin = cos[:, None], sin[:, None]            # [N, 1, 1, HD]
        x = F.embedding(ids, self.embed)                 # [N, 1, D]
        read = (table[:, :, None] * c.block + torch.arange(c.block, device=table.device)[None, None, :]).reshape(n * s)
        keep = (torch.arange(s, device=pos.device)[None, None, :] <= pos[:, :, None])[:, None]   # [N, 1, 1, S]
        flat = slots.reshape(n)
        ks, vs = [], []
        for l in range(c.layers):
            w = self.lw(l)
            state = {}

            def attend(q, k, v, l=l, state=state):
                kpl = kp[l].index_copy(0, flat, k.transpose(1, 2).reshape(n, c.kv_heads, c.head_dim))
                vpl = vp[l].index_copy(0, flat, v.transpose(1, 2).reshape(n, c.kv_heads, c.head_dim))
                state["k"], state["v"] = kpl, vpl
                kc = kpl.index_select(0, read).view(n, s, c.kv_heads, c.head_dim).transpose(1, 2)
                vc = vpl.index_select(0, read).view(n, s, c.kv_heads, c.head_dim).transpose(1, 2)
                return F.scaled_dot_product_attention(q, kc, vc, attn_mask=keep, enable_gqa=True)

            x = self._layer(w, x, cos, sin, attend)
            ks.append(state["k"])
            vs.append(state["v"])
        logits = F.linear(_rms(x, self.norm, c.eps), self.head)
        return logits.reshape(n, c.vocab), torch.stack(ks), torch.stack(vs)

    def full_logits(self, ids):
        """Plain causal forward over ids [B, T] (no cache): the float64 reference."""
        t = ids.shape[1]
        cos, sin = self._rope(torch.arange(t, device=ids.device))
        x = F.embedding(ids, self.embed)
        for l in range(self.c.layers):
            x = self._layer(self.lw(l), x, cos, sin,
                            lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True))
        return F.linear(_rms(x, self.norm, self.c.eps), self.head)

    def dense_step(self, ids, start: int, kc, vc):
        """Eager baseline step with a dense cache [L, B, HKV, Cap, HD], the way a static batch
        ``generate`` loop runs: ids [B, T] at positions start .. start + T - 1 for every row. The caches
        are written in place; returns logits [B, T, V]."""
        b, t = ids.shape
        pos = torch.arange(start, start + t, device=ids.device)
        cos, sin = self._rope(pos)
        x = F.embedding(ids, self.embed)
        cap = kc.shape[3]
        keep = (torch.arange(cap, device=ids.device)[None, :] <= pos[:, None]).view(1, 1, t, cap)
        for l in range(self.c.layers):
            def attend(q, k, v, l=l):
                kc[l, :, :, start:start + t] = k
                vc[l, :, :, start:start + t] = v
                return F.scaled_dot_product_attention(q, kc[l], vc[l], attn_mask=keep, enable_gqa=True)
            x = self._layer(self.lw(l), x, cos, sin, attend)
        return F.linear(_rms(x, self.norm, self.c.eps), self.head)


class _TokenStep(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids, pos, slots, table, kp, vp):
        return self.m.token_step(ids, pos, slots, table, kp, vp)


def import_token_step(model: TinyLM, max_rows: int = 4096):
    """The token step exported with symbolic row count N, table width M and pool size P (float64)."""
    from torch.export import Dim

    from truestar import ir

    c = model.c
    N, M, P = Dim("N", min=1, max=max_rows), Dim("M", min=1, max=4096), Dim("P", min=2, max=1 << 22)
    n, m, nb = 3, 2, 6
    ids = torch.zeros(n, 1, dtype=torch.int64)
    pos = torch.arange(n).reshape(n, 1)
    table = torch.arange(n * m).reshape(n, m) % nb
    kp = torch.zeros(c.layers, nb * c.block, c.kv_heads, c.head_dim, dtype=torch.float64)
    mod = _TokenStep(copy.deepcopy(model).double())
    return ir.import_module(mod, (ids, pos, pos.clone() + 1, table, kp, kp.clone()),
                            dynamic_shapes=({0: N}, {0: N}, {0: N}, {0: N, 1: M}, {1: P}, {1: P}),
                            name="serve_token_step")


def to_hf(model: TinyLM):
    """The same weights as a Hugging Face ``LlamaForCausalLM`` (needs transformers; benchmark only)."""
    import transformers

    c = model.c
    cfg = transformers.LlamaConfig(vocab_size=c.vocab, hidden_size=c.dim, num_hidden_layers=c.layers,
                                   num_attention_heads=c.heads, num_key_value_heads=c.kv_heads,
                                   intermediate_size=c.ffn, rms_norm_eps=c.eps, rope_theta=c.theta,
                                   head_dim=c.head_dim, max_position_embeddings=max(4096, c.max_len),
                                   tie_word_embeddings=False, attn_implementation="sdpa")
    hf = transformers.LlamaForCausalLM(cfg).eval()
    sd = {"model.embed_tokens.weight": model.embed, "model.norm.weight": model.norm, "lm_head.weight": model.head}
    names = {"n1": "input_layernorm", "n2": "post_attention_layernorm", "q": "self_attn.q_proj",
             "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj", "gate": "mlp.gate_proj",
             "up": "mlp.up_proj", "down": "mlp.down_proj"}
    for l in range(c.layers):
        for mine, theirs in names.items():
            sd[f"model.layers.{l}.{theirs}.weight"] = model.lw(l)[mine]
    with torch.no_grad():
        own = hf.state_dict()
        for k, v in sd.items():
            own[k].copy_(v.detach().to(own[k].dtype))
    return hf
