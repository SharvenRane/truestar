"""The tiny Llama of tests/engine_models.py ported to Keras 3 (any backend), loading the PyTorch
weights from safetensors. Fault switches as in tests/xport_flax.py, plus `backend_bug`, a code path
taken only on one backend (the kind of difference a "same numerics on every backend" claim hides)."""

from __future__ import annotations

import math

import keras
from keras import ops

DIM, LAYERS, HEADS, KV_HEADS, FFN = 256, 2, 4, 2, 512
HD = DIM // HEADS


class RMSNorm(keras.layers.Layer):
    def __init__(self, eps, **kw):
        super().__init__(**kw)
        self.eps = eps

    def build(self, shape):
        self.scale = self.add_weight(shape=(shape[-1],), initializer="ones", name="scale")

    def call(self, x):
        return x * ops.rsqrt(ops.mean(x * x, axis=-1, keepdims=True) + self.eps) * self.scale


def _dense(n):
    return keras.layers.Dense(n, use_bias=False)


class Attention(keras.layers.Layer):
    def __init__(self, theta, backend_bug=None, **kw):
        super().__init__(**kw)
        self.q, self.k = _dense(HEADS * HD), _dense(KV_HEADS * HD)
        self.v, self.o = _dense(KV_HEADS * HD), _dense(DIM)
        self.theta = theta
        self.backend_bug = backend_bug

    def _rope(self, x):
        T = ops.shape(x)[1]
        inv = 1.0 / (self.theta ** (ops.cast(ops.arange(0, HD, 2), "float64") / HD))
        ang = ops.cast(ops.arange(T), "float64")[:, None] * inv[None]
        c, s = ops.cos(ang)[None, :, None], ops.sin(ang)[None, :, None]
        a, b = x[..., : HD // 2], x[..., HD // 2:]
        return ops.concatenate([a * c - b * s, b * c + a * s], axis=-1)

    def call(self, x):
        B, T = ops.shape(x)[0], ops.shape(x)[1]
        q = self._rope(ops.reshape(self.q(x), (B, T, HEADS, HD)))
        k = self._rope(ops.reshape(self.k(x), (B, T, KV_HEADS, HD)))
        v = ops.reshape(self.v(x), (B, T, KV_HEADS, HD))
        k = ops.repeat(k, HEADS // KV_HEADS, axis=2)
        v = ops.repeat(v, HEADS // KV_HEADS, axis=2)
        s = ops.einsum("bthd,bshd->bhts", q, k) / math.sqrt(HD)
        mask = ops.tril(ops.ones((T, T), "bool"))
        s = ops.where(mask, s, float("-inf"))
        if keras.backend.backend() == self.backend_bug:  # bug: a scale applied twice on one backend
            s = s * 1.01
        p = ops.softmax(s, axis=-1)
        o = ops.einsum("bhts,bshd->bthd", p, ops.cast(v, p.dtype))
        return self.o(ops.reshape(o, (B, T, HEADS * HD)))


class MLP(keras.layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.gate, self.up, self.down = _dense(FFN), _dense(FFN), _dense(DIM)

    def call(self, x):
        return self.down(ops.silu(self.gate(x)) * self.up(x))


class Block(keras.layers.Layer):
    def __init__(self, eps, theta, backend_bug=None, **kw):
        super().__init__(**kw)
        self.n1, self.attn = RMSNorm(eps), Attention(theta, backend_bug)
        self.n2, self.mlp = RMSNorm(eps), MLP()

    def call(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class Model(keras.Model):
    def __init__(self, vocab, eps=1e-5, theta=10000.0, backend_bug=None, **kw):
        super().__init__(**kw)
        self.emb = keras.layers.Embedding(vocab, DIM)
        self.blocks = [Block(eps, theta, backend_bug) for _ in range(LAYERS)]
        self.norm = RMSNorm(eps)
        self.head = _dense(vocab)

    def call(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def build(weights_path, eps=1e-5, theta=10000.0, untransposed=(), backend_bug=None):
    from truestar.interop import load_file

    keras.config.set_floatx("float64")
    sd = {k: v.double().numpy() for k, v in load_file(weights_path).items()}
    vocab = sd["model.embed_tokens.weight"].shape[0]
    m = Model(vocab, eps, theta, backend_bug)
    m(ops.zeros((1, 3), "int32"))  # build every layer

    def put(dense, name):
        w = sd[name]
        dense.kernel.assign(w if name in untransposed else w.T)

    m.emb.embeddings.assign(sd["model.embed_tokens.weight"])
    m.norm.scale.assign(sd["model.norm.weight"])
    put(m.head, "lm_head.weight")
    for i, b in enumerate(m.blocks):
        p = f"model.layers.{i}."
        b.n1.scale.assign(sd[p + "input_layernorm.weight"])
        b.n2.scale.assign(sd[p + "post_attention_layernorm.weight"])
        for d, n in [(b.attn.q, "self_attn.q_proj"), (b.attn.k, "self_attn.k_proj"), (b.attn.v, "self_attn.v_proj"),
                     (b.attn.o, "self_attn.o_proj"), (b.mlp.gate, "mlp.gate_proj"), (b.mlp.up, "mlp.up_proj"),
                     (b.mlp.down, "mlp.down_proj")]:
            put(d, p + n + ".weight")
    return m


def get_model(weights_path):
    return build(weights_path)
