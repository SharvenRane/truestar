"""The layer library. Every ``forward`` is written in the native subset: it runs eagerly through
PyTorch outside a native function and is translated from this source inside one (docs/NN.md).

Conventions, the same for every layer:

* ``layer(p, x)``: a layer with parameters takes its parameter tree first.
* ``layer(x)``: a layer without parameters (activations, pools, dropout) takes none.
* ``layer(p, s, x, training=...)`` returns ``(y, new_s)``: a layer with state (batch norm) takes its
  state tree after the parameters and returns the new state. Nothing is updated in place.
* ``training=`` is a compile time argument: the evaluation program has no dropout and no running
  statistic update in it at all, instead of a flag read at run time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from ..frontend import ops
from .core import Layer, Param, fan_in_uniform, inline, layer, normal, ones, zeros

_ACTS = ("relu", "gelu", "gelu_tanh", "silu", "tanh", "sigmoid", "none")


def _check_act(act):
    if act not in _ACTS:
        raise ValueError(f"unknown activation {act!r}; one of {', '.join(_ACTS)}")


# ----------------------------------------------------------------------------- dense


@layer
class Linear(Layer):
    """y = x W^T + b with W [d_out, d_in]. Both drawn uniform on +-1/sqrt(d_in)."""

    d_in: int
    d_out: int
    bias: bool = True
    dtype: str = "f32"

    def params(self):
        out = {"w": Param((self.d_out, self.d_in), fan_in_uniform(self.d_in))}
        if self.bias:
            out["b"] = Param((self.d_out,), fan_in_uniform(self.d_in))
        return out

    def forward(self, p, x):
        if self.bias:
            return ops.linear(x, p["w"], p["b"])
        return ops.linear(x, p["w"])


@layer
class Embedding(Layer):
    """Row ``i`` of a [count, dim] table for each id ``i``; ids of any shape. Drawn N(0, 1)."""

    count: int
    dim: int
    std: float = 1.0
    dtype: str = "f32"

    def params(self):
        return {"w": Param((self.count, self.dim), normal(self.std))}

    def forward(self, p, ids):
        n = 1
        for d in ids.shape:
            n = n * d
        rows = ops.index_select(p["w"], 0, ops.reshape(ids, [n]))
        return ops.reshape(rows, list(ids.shape) + [self.dim])

    def attend(self, p, x):
        """Logits against the table (a tied output head): x [..., dim] -> [..., count]."""
        return ops.linear(x, p["w"])


# ----------------------------------------------------------------------------- convolution and pooling


@layer
class Conv2d(Layer):
    """2D convolution over [N, C, H, W]; weight [c_out, c_in / groups, k, k]."""

    c_in: int
    c_out: int
    kernel: int
    stride: int = 1
    padding: int = 0
    groups: int = 1
    bias: bool = True
    dtype: str = "f32"

    def __post_init__(self):
        if self.c_in % self.groups or self.c_out % self.groups:
            raise ValueError(f"c_in {self.c_in} and c_out {self.c_out} must both be multiples of groups {self.groups}")

    def params(self):
        fan = self.c_in // self.groups * self.kernel * self.kernel
        out = {"w": Param((self.c_out, self.c_in // self.groups, self.kernel, self.kernel), fan_in_uniform(fan))}
        if self.bias:
            out["b"] = Param((self.c_out,), fan_in_uniform(fan))
        return out

    def forward(self, p, x):
        b = p["b"] if self.bias else None
        return ops.conv2d(x, p["w"], b, stride=self.stride, padding=self.padding, groups=self.groups)


@layer
class MaxPool2d(Layer):
    kernel: int
    stride: Optional[int] = None
    padding: int = 0

    def forward(self, x):
        return ops.max_pool2d(x, self.kernel, self.kernel if self.stride is None else self.stride, self.padding)


@layer
class AvgPool2d(Layer):
    """Mean over non overlapping ``kernel`` windows; H and W must be multiples of ``kernel``."""

    kernel: int

    def forward(self, x):
        return ops.avg_pool2d(x, self.kernel)


@layer
class GlobalAvgPool(Layer):
    """[N, C, H, W] -> [N, C]: the mean over each channel's plane."""

    def forward(self, x):
        return ops.mean(x, dim=(2, 3))


@layer
class Flatten(Layer):
    """Merge every dim from ``start`` on: [N, C, H, W] -> [N, C * H * W]."""

    start: int = 1

    def forward(self, x):
        n = 1
        for d in x.shape[self.start:]:
            n = n * d
        return ops.reshape(x, list(x.shape[:self.start]) + [n])


# ----------------------------------------------------------------------------- normalization


@layer
class LayerNorm(Layer):
    """(x - mean) / sqrt(var + eps) * w + b over the last dim (biased variance, as torch)."""

    dim: int
    eps: float = 1e-5
    dtype: str = "f32"

    def params(self):
        return {"w": Param((self.dim,), ones), "b": Param((self.dim,), zeros)}

    def forward(self, p, x):
        return ops.layer_norm(x, [self.dim], p["w"], p["b"], self.eps)


@layer
class RMSNorm(Layer):
    """x / sqrt(mean(x^2) + eps) * w over the last dim."""

    dim: int
    eps: float = 1e-6
    dtype: str = "f32"

    def params(self):
        return {"w": Param((self.dim,), ones)}

    def forward(self, p, x):
        r = ops.rsqrt(ops.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * r * p["w"]


@layer
class BatchNorm(Layer):
    """Batch norm over dim 1 ([N, C] or [N, C, H, W]). Running statistics are explicit state:
    ``y, s = bn(p, s, x, training=True)`` normalizes with the batch statistics and returns the
    updated running mean and (unbiased) variance; ``training=False`` uses ``s`` and returns it."""

    channels: int
    momentum: float = 0.1
    eps: float = 1e-5
    dtype: str = "f32"
    uses_training = True

    def params(self):
        return {"w": Param((self.channels,), ones), "b": Param((self.channels,), zeros)}

    def state(self):
        return {"mean": Param((self.channels,), zeros), "var": Param((self.channels,), ones)}

    def forward(self, p, s, x, training=False):
        if training:
            y, m, v = ops.batch_norm(x, p["w"], p["b"], s["mean"], s["var"], training=True,
                                     momentum=self.momentum, eps=self.eps)
            return y, {"mean": m, "var": v}
        y = ops.batch_norm(x, p["w"], p["b"], s["mean"], s["var"], training=False, eps=self.eps)
        return y, s


# ----------------------------------------------------------------------------- activations and dropout


@inline
def _act(kind, x):
    if kind == "relu":
        return ops.relu(x)
    if kind == "gelu":
        return ops.gelu(x)
    if kind == "gelu_tanh":
        return ops.gelu(x, approximate="tanh")
    if kind == "silu":
        return ops.silu(x)
    if kind == "tanh":
        return ops.tanh(x)
    if kind == "sigmoid":
        return ops.sigmoid(x)
    return x


@layer
class Activation(Layer):
    """An elementwise activation by name: relu, gelu, gelu_tanh, silu, tanh, sigmoid, none."""

    kind: str = "relu"

    def __post_init__(self):
        _check_act(self.kind)

    def forward(self, x):
        return _act(self.kind, x)


def ReLU():
    return Activation("relu")


def GELU(approximate: str = "none"):
    return Activation("gelu" if approximate == "none" else "gelu_tanh")


def SiLU():
    return Activation("silu")


def Tanh():
    return Activation("tanh")


@layer
class Dropout(Layer):
    """Inverted dropout with a keyed mask. Inside a native function the key comes from the
    function's seed, its call number and this call site, so a step replays bit for bit on every
    backend; eagerly pass ``key=``. ``training=False`` (the default) is the identity and emits
    nothing."""

    p: float = 0.1
    uses_training = True

    def __post_init__(self):
        if not 0.0 <= self.p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {self.p}")

    def forward(self, x, training=False, key=None):
        if not training or self.p == 0.0:
            return x
        return ops.dropout(x, self.p, True, key=key)


# ----------------------------------------------------------------------------- containers


@dataclass(frozen=True, init=False)
class Sequential(Layer):
    """Layers applied in order. The parameter tree has one entry per layer with parameters, keyed
    by position (``"0"``, ``"2"``); with a stateful layer inside, ``seq(p, x, training=..., state=s)``
    returns ``(y, new_s)``."""

    layers: Tuple[Layer, ...]

    def __init__(self, *layers: Layer):
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = tuple(layers[0])
        for i, l in enumerate(layers):
            if not isinstance(l, Layer):
                raise TypeError(f"Sequential item {i} is {type(l).__name__}, not a truestar.nn layer")
        object.__setattr__(self, "layers", tuple(layers))

    @property
    def dtype(self):
        return self.layers[0].dtype if self.layers else "f32"

    @property
    def uses_training(self):
        return any(l.uses_training for l in self.layers)

    @property
    def state_keys(self):
        return tuple(str(i) for i, l in enumerate(self.layers) if l.has_state)

    def children(self):
        return {str(i): l for i, l in enumerate(self.layers)}

    def forward(self, p, x, training=False, state=None):
        new = ()
        for i, item in enumerate(self.layers):
            k = str(i)
            if item.has_state:
                x, s = item(p[k], state[k], x, training=training)
                new = new + (s,)
            elif item.has_params:
                if item.uses_training:
                    x = item(p[k], x, training=training)
                else:
                    x = item(p[k], x)
            elif item.uses_training:
                x = item(x, training=training)
            else:
                x = item(x)
        if self.has_state:
            return x, dict(zip(self.state_keys, new))
        return x


@layer
class MLP(Layer):
    """d -> hidden -> d_out (default d) with an activation between and optional dropout after it."""

    dim: int
    hidden: int
    d_out: Optional[int] = None
    act: str = "gelu"
    dropout: float = 0.0
    bias: bool = True
    dtype: str = "f32"
    uses_training = True

    def __post_init__(self):
        _check_act(self.act)

    def children(self):
        return {"fc1": Linear(self.dim, self.hidden, self.bias, self.dtype),
                "fc2": Linear(self.hidden, self.dim if self.d_out is None else self.d_out, self.bias, self.dtype)}

    def forward(self, p, x, training=False):
        h = _act(self.act, self.fc1(p["fc1"], x))
        if training and self.dropout > 0.0:
            h = ops.dropout(h, self.dropout, True)
        return self.fc2(p["fc2"], h)


# ----------------------------------------------------------------------------- attention


@layer
class MultiHeadAttention(Layer):
    """Self attention over x [B, T, dim] with ``heads`` query heads and ``kv_heads`` key and value
    heads (fewer is grouped query attention; 1 is multi query).

    ``mask``: bool [B, S], True where a key may be attended (padding mask). ``causal``: position i
    attends to positions <= i. ``cache``: ``{"k": [B, kv_heads, S, head_dim], "v": ...}`` with
    ``pos`` int64 [T], the absolute positions of the T new tokens: the new keys and values are
    written at ``pos`` and every query attends to cache slots <= its position, so unwritten slots
    never take part. With a cache the call returns ``(y, new_cache)``. One compiled program serves
    every batch size, length, position and cache capacity."""

    dim: int
    heads: int
    kv_heads: Optional[int] = None
    causal: bool = False
    bias: bool = True
    dtype: str = "f32"

    def __post_init__(self):
        if self.dim % self.heads:
            raise ValueError(f"dim {self.dim} is not a multiple of heads {self.heads}")
        if self.heads % self.n_kv:
            raise ValueError(f"heads {self.heads} is not a multiple of kv_heads {self.n_kv}")

    @property
    def n_kv(self) -> int:
        return self.heads if self.kv_heads is None else self.kv_heads

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    def children(self):
        kv = self.n_kv * self.head_dim
        return {"q": Linear(self.dim, self.dim, self.bias, self.dtype), "k": Linear(self.dim, kv, self.bias, self.dtype),
                "v": Linear(self.dim, kv, self.bias, self.dtype), "o": Linear(self.dim, self.dim, self.bias, self.dtype)}

    def init_cache(self, batch: int, capacity: int):
        """An empty cache for ``batch`` sequences of up to ``capacity`` tokens."""
        import torch

        from .core import _dtype
        shape = (batch, self.n_kv, capacity, self.head_dim)
        return {"k": torch.zeros(shape, dtype=_dtype(self.dtype)), "v": torch.zeros(shape, dtype=_dtype(self.dtype))}

    def cache_spec(self, batch="B", capacity="S"):
        from ..frontend.types import Tensor
        t = Tensor[batch, self.n_kv, capacity, self.head_dim, self.dtype]
        return {"k": t, "v": t}

    def forward(self, p, x, mask=None, cache=None, pos=None):
        B = x.shape[0]
        T = x.shape[1]
        hd = self.head_dim
        q = ops.permute(ops.reshape(self.q(p["q"], x), [B, T, self.heads, hd]), [0, 2, 1, 3])
        k = ops.permute(ops.reshape(self.k(p["k"], x), [B, T, self.n_kv, hd]), [0, 2, 1, 3])
        v = ops.permute(ops.reshape(self.v(p["v"], x), [B, T, self.n_kv, hd]), [0, 2, 1, 3])
        keep = None
        if cache is None:
            S = T
            if self.causal:   # an explicit mask: differentiable for every (symbolic) length T
                rows = ops.reshape(ops.arange(T), [T, 1])
                cols = ops.reshape(ops.arange(T), [1, T])
                keep = cols <= rows
        else:
            k = ops.index_copy(cache["k"], 2, pos, k)
            v = ops.index_copy(cache["v"], 2, pos, v)
            S = cache["k"].shape[2]
            rows = ops.reshape(pos, [T, 1])
            cols = ops.reshape(ops.arange(S), [1, S])
            keep = cols <= rows
        if mask is not None:
            m = ops.reshape(mask, [B, 1, 1, S])
            if keep is None:
                keep = m
            else:
                keep = ops.where(keep, m, keep)
        a = ops.attention(q, k, v, mask=keep, enable_gqa=self.n_kv != self.heads)
        y = self.o(p["o"], ops.reshape(ops.permute(a, [0, 2, 1, 3]), [B, T, self.dim]))
        if cache is None:
            return y
        return y, {"k": k, "v": v}


@layer
class TransformerBlock(Layer):
    """Pre norm block: x + attn(norm1(x)), then x + mlp(norm2(x)), dropout on both residual
    branches when training. ``norm`` is "layer" or "rms"."""

    dim: int
    heads: int
    hidden: Optional[int] = None
    kv_heads: Optional[int] = None
    causal: bool = True
    norm: str = "layer"
    act: str = "gelu"
    dropout: float = 0.0
    bias: bool = True
    dtype: str = "f32"
    uses_training = True

    def __post_init__(self):
        if self.norm not in ("layer", "rms"):
            raise ValueError(f"norm must be 'layer' or 'rms', got {self.norm!r}")
        _check_act(self.act)

    def children(self):
        mk = (lambda: LayerNorm(self.dim, dtype=self.dtype)) if self.norm == "layer" else \
            (lambda: RMSNorm(self.dim, dtype=self.dtype))
        return {"n1": mk(),
                "attn": MultiHeadAttention(self.dim, self.heads, self.kv_heads, self.causal, self.bias, self.dtype),
                "n2": mk(),
                "mlp": MLP(self.dim, 4 * self.dim if self.hidden is None else self.hidden, None, self.act, 0.0,
                           self.bias, self.dtype)}

    def forward(self, p, x, mask=None, training=False, cache=None, pos=None):
        h = self.n1(p["n1"], x)
        if cache is None:
            a = self.attn(p["attn"], h, mask=mask)
        else:
            a, cache = self.attn(p["attn"], h, mask=mask, cache=cache, pos=pos)
        if training and self.dropout > 0.0:
            a = ops.dropout(a, self.dropout, True)
        x = x + a
        m = self.mlp(p["mlp"], self.n2(p["n2"], x))
        if training and self.dropout > 0.0:
            m = ops.dropout(m, self.dropout, True)
        x = x + m
        if cache is None:
            return x
        return x, cache


__all__ = ["Linear", "Embedding", "Conv2d", "MaxPool2d", "AvgPool2d", "GlobalAvgPool", "Flatten", "LayerNorm",
           "RMSNorm", "BatchNorm", "Activation", "ReLU", "GELU", "SiLU", "Tanh", "Dropout", "Sequential", "MLP",
           "MultiHeadAttention", "TransformerBlock"]
