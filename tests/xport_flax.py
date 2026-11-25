"""The tiny Llama of tests/engine_models.py ported to Flax NNX, loading the PyTorch weights from a
safetensors file the way a conversion script does. Fault switches inject the port bugs people hit:
a hardcoded eps, another rope theta, a kernel loaded without its transpose, rotary pairs taken
adjacent instead of as halves."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

DIM, LAYERS, HEADS, KV_HEADS, FFN = 256, 2, 4, 2, 512
HD = DIM // HEADS
F64 = jnp.float64


class RMSNorm(nnx.Module):
    def __init__(self, d, eps):
        self.scale = nnx.Param(jnp.ones((d,), F64))
        self.eps = eps

    def __call__(self, x):
        return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps) * self.scale.value


def _lin(i, o, rngs):
    return nnx.Linear(i, o, use_bias=False, dtype=F64, param_dtype=F64, rngs=rngs)


class Attention(nnx.Module):
    def __init__(self, theta, pairs, rngs):
        self.wq, self.wk = _lin(DIM, HEADS * HD, rngs), _lin(DIM, KV_HEADS * HD, rngs)
        self.wv, self.wo = _lin(DIM, KV_HEADS * HD, rngs), _lin(HEADS * HD, DIM, rngs)
        self.theta = theta
        self.pairs = pairs

    def _rope(self, x):  # x [B, T, H, D]
        T = x.shape[1]
        inv = 1.0 / (self.theta ** (jnp.arange(0, HD, 2, dtype=F64) / HD))
        ang = jnp.arange(T, dtype=F64)[:, None] * inv[None]
        c, s = jnp.cos(ang)[None, :, None], jnp.sin(ang)[None, :, None]
        if self.pairs:  # bug: rotates adjacent pairs while the weights are in half order
            a, b = x[..., 0::2], x[..., 1::2]
            return jnp.stack([a * c - b * s, a * s + b * c], axis=-1).reshape(x.shape)
        a, b = x[..., : HD // 2], x[..., HD // 2:]
        return jnp.concatenate([a * c - b * s, b * c + a * s], axis=-1)

    def __call__(self, x):
        B, T, _ = x.shape
        q = self._rope(self.wq(x).reshape(B, T, HEADS, HD))
        k = self._rope(self.wk(x).reshape(B, T, KV_HEADS, HD))
        v = self.wv(x).reshape(B, T, KV_HEADS, HD)
        k = jnp.repeat(k, HEADS // KV_HEADS, axis=2)
        v = jnp.repeat(v, HEADS // KV_HEADS, axis=2)
        s = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(HD)
        s = jnp.where(jnp.tril(jnp.ones((T, T), bool)), s, -jnp.inf)
        o = jnp.einsum("bhts,bshd->bthd", jax.nn.softmax(s, axis=-1), v)
        return self.wo(o.reshape(B, T, HEADS * HD))


class MLP(nnx.Module):
    def __init__(self, rngs):
        self.w1, self.w3, self.w2 = _lin(DIM, FFN, rngs), _lin(DIM, FFN, rngs), _lin(FFN, DIM, rngs)

    def __call__(self, x):
        return self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))


class Block(nnx.Module):
    def __init__(self, eps, theta, pairs, rngs):
        self.attn_norm = RMSNorm(DIM, eps)
        self.attn = Attention(theta, pairs, rngs)
        self.mlp_norm = RMSNorm(DIM, eps)
        self.mlp = MLP(rngs)

    def __call__(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class Model(nnx.Module):
    def __init__(self, vocab, eps=1e-5, theta=10000.0, pairs=False, rngs=None):
        rngs = rngs or nnx.Rngs(0)
        self.tok = nnx.Embed(vocab, DIM, dtype=F64, param_dtype=F64, rngs=rngs)
        self.blocks = [Block(eps, theta, pairs, rngs) for _ in range(LAYERS)]
        self.norm = RMSNorm(DIM, eps)
        self.head = _lin(DIM, vocab, rngs)

    def __call__(self, ids):
        x = self.tok(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def build(weights_path, eps=1e-5, theta=10000.0, pairs=False, untransposed=()):
    """Convert the PyTorch state dict in `weights_path` (safetensors) into the Flax model."""
    from truestar.interop import load_file

    sd = {k: v.double().numpy() for k, v in load_file(weights_path).items()}
    m = Model(sd["model.embed_tokens.weight"].shape[0], eps, theta, pairs)

    def put(lin, name):
        w = sd[name]
        lin.kernel.value = jnp.asarray(w if name in untransposed else w.T)

    m.tok.embedding.value = jnp.asarray(sd["model.embed_tokens.weight"])
    m.norm.scale.value = jnp.asarray(sd["model.norm.weight"])
    put(m.head, "lm_head.weight")
    for i, b in enumerate(m.blocks):
        p = f"model.layers.{i}."
        b.attn_norm.scale.value = jnp.asarray(sd[p + "input_layernorm.weight"])
        b.mlp_norm.scale.value = jnp.asarray(sd[p + "post_attention_layernorm.weight"])
        for lin, n in [(b.attn.wq, "self_attn.q_proj"), (b.attn.wk, "self_attn.k_proj"), (b.attn.wv, "self_attn.v_proj"),
                       (b.attn.wo, "self_attn.o_proj"), (b.mlp.w1, "mlp.gate_proj"), (b.mlp.w3, "mlp.up_proj"),
                       (b.mlp.w2, "mlp.down_proj")]:
            put(lin, p + n + ".weight")
    return m


def get_model(weights_path):
    return build(weights_path)
