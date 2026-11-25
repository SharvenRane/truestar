"""The native decoder: one ``truestar.nn`` layer family that covers Llama 3, Mistral, Qwen2, Qwen3,
Phi 3, Gemma 2 and Gemma 3 (text), configured from ``DecoderConfig``.

Every ``forward`` is in the native subset (docs/NN.md), so the model runs eagerly through PyTorch
and compiles as a native function from this source: no torch.export, no PyTorch exporter.

Precision follows transformers: the RMSNorm, the rotary angles and a capped attention's softmax
are computed in ``upcast`` (float32 for float32 and narrower models, float64 for a float64 model)
and cast back, as the Hugging Face modules do with ``.float()``.
"""

from __future__ import annotations

from typing import Tuple

from ..frontend import ops
from ..nn.core import Layer, Param, layer, ones, zeros
from ..nn.layers import Embedding, Linear
from ..nn.layers_more import GatedMLP


@layer
class DecoderNorm(Layer):
    """RMSNorm as transformers computes it. ``kind`` "llama": ``w * cast(norm(x))``; "gemma":
    ``cast(norm(x) * (1 + w))`` with the product in ``upcast``."""

    dim: int
    eps: float = 1e-6
    kind: str = "llama"
    upcast: str = "f32"
    dtype: str = "f32"

    def params(self):
        return {"w": Param((self.dim,), zeros if self.kind == "gemma" else ones)}

    def forward(self, p, x):
        h = ops.cast(x, self.upcast)
        h = h * ops.rsqrt(ops.mean(h * h, dim=-1, keepdim=True) + self.eps)
        if self.kind == "gemma":
            return ops.cast(h * (ops.cast(p["w"], self.upcast) + 1.0), x.dtype)
        return p["w"] * ops.cast(h, x.dtype)


@layer
class RopeTable(Layer):
    """Rotary frequencies held as a float32 tree entry ``inv`` (the values transformers computes),
    and the cos and sin tables for positions: ``cos, sin = table(p, pos, dtype)``, each [T, rot]."""

    half: int
    upcast: str = "f32"
    dtype: str = "f32"

    def params(self):
        return {"inv": Param((self.half,), zeros)}

    def forward(self, p, pos, like):
        inv = ops.cast(p["inv"], self.upcast)
        ang = ops.reshape(ops.cast(pos, self.upcast), [pos.shape[0], 1]) * ops.reshape(inv, [1, self.half])
        ang = ops.cat([ang, ang], dim=-1)
        return ops.cast(ops.cos(ang), like.dtype), ops.cast(ops.sin(ang), like.dtype)


@layer
class RotateHalf(Layer):
    """x cos + rotate_half(x) sin over the first ``rot`` features of x [..., T, D] (the rest pass)."""

    rot: int
    head_dim: int

    def forward(self, x, cos, sin):
        h = self.rot // 2
        if self.rot == self.head_dim:
            r = ops.cat([-x[..., h:], x[..., :h]], dim=-1)
            return x * cos + r * sin
        xr = x[..., :self.rot]
        r = ops.cat([-xr[..., h:], xr[..., :h]], dim=-1)
        return ops.cat([xr * cos + r * sin, x[..., self.rot:]], dim=-1)


@layer
class DecoderAttention(Layer):
    """Grouped query self attention with rotary positions, optional per head q and k norms, a
    sliding window (``window`` > 0) and score capping (``softcap`` > 0)."""

    dim: int
    heads: int
    kv_heads: int
    head_dim: int
    rot: int
    scale: float
    qkv_bias: bool = False
    o_bias: bool = False
    qk_norm: bool = False
    norm_kind: str = "llama"
    eps: float = 1e-6
    softcap: float = 0.0
    upcast: str = "f32"
    dtype: str = "f32"

    def children(self):
        q, kv = self.heads * self.head_dim, self.kv_heads * self.head_dim
        out = {"q": Linear(self.dim, q, self.qkv_bias, self.dtype), "k": Linear(self.dim, kv, self.qkv_bias, self.dtype),
               "v": Linear(self.dim, kv, self.qkv_bias, self.dtype), "o": Linear(q, self.dim, self.o_bias, self.dtype),
               "rotate": RotateHalf(self.rot, self.head_dim)}
        if self.qk_norm:
            out["q_norm"] = DecoderNorm(self.head_dim, self.eps, self.norm_kind, self.upcast, self.dtype)
            out["k_norm"] = DecoderNorm(self.head_dim, self.eps, self.norm_kind, self.upcast, self.dtype)
        return out

    def forward(self, p, x, cos, sin, keep, cache=None, pos=None):
        B = x.shape[0]
        T = x.shape[1]
        hd = self.head_dim
        q = ops.reshape(self.q(p["q"], x), [B, T, self.heads, hd])
        k = ops.reshape(self.k(p["k"], x), [B, T, self.kv_heads, hd])
        v = ops.reshape(self.v(p["v"], x), [B, T, self.kv_heads, hd])
        if self.qk_norm:
            q = self.q_norm(p["q_norm"], q)
            k = self.k_norm(p["k_norm"], k)
        q = self.rotate(ops.permute(q, [0, 2, 1, 3]), cos, sin)
        k = self.rotate(ops.permute(k, [0, 2, 1, 3]), cos, sin)
        v = ops.permute(v, [0, 2, 1, 3])
        if cache is not None:
            k = ops.index_copy(cache["k"], 2, pos, k)
            v = ops.index_copy(cache["v"], 2, pos, v)
        if self.softcap > 0.0:
            g = self.heads // self.kv_heads
            kk = k
            vv = v
            if g > 1:
                kk = ops.repeat_interleave(k, g, 1)
                vv = ops.repeat_interleave(v, g, 1)
            s = ops.matmul(q, ops.transpose(kk, 2, 3)) * self.scale
            s = ops.tanh(s / self.softcap) * self.softcap
            s = ops.where(keep, s, ops.full_like(s, float("-inf")))
            w = ops.cast(ops.softmax(ops.cast(s, self.upcast), dim=-1), x.dtype)
            a = ops.matmul(w, vv)
        else:
            a = ops.attention(q, k, v, mask=keep, scale=self.scale, enable_gqa=self.kv_heads != self.heads)
        y = self.o(p["o"], ops.reshape(ops.permute(a, [0, 2, 1, 3]), [B, T, self.heads * hd]))
        if cache is None:
            return y
        return y, {"k": k, "v": v}


@layer
class DecoderBlock(Layer):
    """One decoder layer: pre norm (Llama family) or sandwich norms (Gemma)."""

    dim: int
    heads: int
    kv_heads: int
    head_dim: int
    hidden: int
    rot: int
    scale: float
    act: str = "silu"
    eps: float = 1e-6
    norm_kind: str = "llama"
    qkv_bias: bool = False
    o_bias: bool = False
    mlp_bias: bool = False
    qk_norm: bool = False
    sandwich: bool = False
    softcap: float = 0.0
    upcast: str = "f32"
    dtype: str = "f32"

    def children(self):
        def norm():
            return DecoderNorm(self.dim, self.eps, self.norm_kind, self.upcast, self.dtype)
        out = {"in_norm": norm(),
               "attn": DecoderAttention(self.dim, self.heads, self.kv_heads, self.head_dim, self.rot, self.scale,
                                        self.qkv_bias, self.o_bias, self.qk_norm, self.norm_kind, self.eps,
                                        self.softcap, self.upcast, self.dtype),
               "post_attn_norm": norm(),
               "mlp": GatedMLP(self.dim, self.hidden, self.act, self.mlp_bias, self.dtype)}
        if self.sandwich:
            out["pre_ff_norm"] = norm()
            out["post_ff_norm"] = norm()
        return out

    def forward(self, p, x, cos, sin, keep, cache=None, pos=None):
        h = self.in_norm(p["in_norm"], x)
        if cache is None:
            a = self.attn(p["attn"], h, cos, sin, keep)
        else:
            a, cache = self.attn(p["attn"], h, cos, sin, keep, cache=cache, pos=pos)
        if self.sandwich:
            x = x + self.post_attn_norm(p["post_attn_norm"], a)
            m = self.mlp(p["mlp"], self.pre_ff_norm(p["pre_ff_norm"], x))
            x = x + self.post_ff_norm(p["post_ff_norm"], m)
        else:
            x = x + a
            x = x + self.mlp(p["mlp"], self.post_attn_norm(p["post_attn_norm"], x))
        if cache is None:
            return x
        return x, cache


@layer
class DecoderLM(Layer):
    """A causal language model: ``logits = lm(p, ids, pos)`` for ids int64 [B, T] at positions pos
    int64 [T]; with ``cache`` (``lm.init_cache``) ``logits, cache = lm(p, ids, pos, cache)``, the
    keys and values written at ``pos`` and every query attending to cache slots <= its position
    (and inside its window)."""

    vocab: int
    dim: int
    depth: int
    heads: int
    kv_heads: int
    head_dim: int
    hidden: int
    rot: int
    scale: float
    windows: Tuple[int, ...]
    local_rope: Tuple[bool, ...]
    act: str = "silu"
    eps: float = 1e-6
    norm_kind: str = "llama"
    qkv_bias: bool = False
    o_bias: bool = False
    mlp_bias: bool = False
    qk_norm: bool = False
    sandwich: bool = False
    attn_softcap: float = 0.0
    final_softcap: float = 0.0
    embed_scale: float = 0.0      # 0: none; else the factor, already rounded as transformers rounds it
    tie: bool = False
    upcast: str = "f32"
    dtype: str = "f32"

    @property
    def has_local(self) -> bool:
        return any(self.local_rope)

    def children(self):
        out = {"embed": Embedding(self.vocab, self.dim, 1.0, self.dtype),
               "rope": RopeTable(self.rot // 2, self.upcast, "f32")}
        if self.has_local:
            out["rope_local"] = RopeTable(self.rot // 2, self.upcast, "f32")
        for i in range(self.depth):
            out[f"l{i}"] = DecoderBlock(self.dim, self.heads, self.kv_heads, self.head_dim, self.hidden, self.rot,
                                        self.scale, self.act, self.eps, self.norm_kind, self.qkv_bias, self.o_bias,
                                        self.mlp_bias, self.qk_norm, self.sandwich, self.attn_softcap, self.upcast,
                                        self.dtype)
        out["norm"] = DecoderNorm(self.dim, self.eps, self.norm_kind, self.upcast, self.dtype)
        if not self.tie:
            out["head"] = Linear(self.dim, self.vocab, False, self.dtype)
        return out

    def forward(self, p, ids, pos, cache=None):
        x = self.embed(p["embed"], ids)
        if self.embed_scale != 0.0:
            x = x * self.embed_scale
        cos, sin = self.rope(p["rope"], pos, x)
        cos_l = cos
        sin_l = sin
        if self.has_local:
            cos_l, sin_l = self.rope_local(p["rope_local"], pos, x)
        T = ids.shape[1]
        if cache is None:
            S = T
        else:
            S = cache["l0"]["k"].shape[2]
        rows = ops.reshape(pos, [T, 1])
        cols = ops.reshape(ops.arange(S), [1, S])
        causal = cols <= rows
        new = ()
        for i in range(self.depth):
            key = "l" + str(i)
            w = self.windows[i]
            keep = causal
            if w > 0:
                keep = ops.where(causal, cols > rows - w, causal)
            c = cos
            s = sin
            if self.local_rope[i]:
                c = cos_l
                s = sin_l
            if cache is None:
                x = self.parts[key](p[key], x, c, s, keep)
            else:
                x, kv = self.parts[key](p[key], x, c, s, keep, cache=cache[key], pos=pos)
                new = new + (kv,)
        h = self.norm(p["norm"], x)
        if self.tie:
            logits = ops.linear(h, p["embed"]["w"])
        else:
            logits = self.head(p["head"], h)
        if self.final_softcap > 0.0:
            logits = ops.tanh(logits / self.final_softcap) * self.final_softcap
        if cache is None:
            return logits
        return logits, dict(zip(cache.keys(), new))

    def init_cache(self, batch: int, capacity: int):
        import torch

        from ..nn.core import _dtype
        shape = (batch, self.kv_heads, capacity, self.head_dim)
        return {f"l{i}": {"k": torch.zeros(shape, dtype=_dtype(self.dtype)),
                          "v": torch.zeros(shape, dtype=_dtype(self.dtype))} for i in range(self.depth)}

    def cache_spec(self, batch="B", capacity="S"):
        from ..frontend.types import Tensor
        t = Tensor[batch, self.kv_heads, capacity, self.head_dim, self.dtype]
        return {f"l{i}": {"k": t, "v": t} for i in range(self.depth)}
