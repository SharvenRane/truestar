"""More layers (added 2026-11-24 with the op coverage work, docs/NN.md): the layers the models of
docs/MODEL_COVERAGE.md need beyond the first library. Same conventions as layers.py; every forward is
the native subset, eager through PyTorch outside a native function and translated inside one.

* GroupNorm, InstanceNorm: the IR ``group_norm`` op (instance norm is one group per channel).
* Conv1d (a 2D convolution over [N, C, 1, L]) and ConvTranspose2d (IR ``conv2d_input_grad``, the
  adjoint of conv2d). Depthwise and grouped 2D convolution are ``Conv2d(groups=...)``.
* Upsample (nearest, bilinear with or without align_corners) from source indices computed with
  ``arange`` and two ``index_select`` reads per axis; PixelShuffle as reshape and permute.
* LSTM and GRU over a symbolic sequence length with an IR loop (one compile for every length).
* EmbeddingBag: sum, mean or max of the rows of each bag ([B, L] ids, optional per sample weights).
* Rotary positions (RoPE, the rotate half layout of Llama) and ALiBi biases.
* SwiGLU and GEGLU feed forward blocks.
"""

from __future__ import annotations

import math

from ..frontend import ops
from .core import Layer, Param, fan_in_uniform, inline, layer, normal, ones, zeros

# ----------------------------------------------------------------------------- normalization


@layer
class GroupNorm(Layer):
    """(x - mean) / sqrt(biased variance + eps) per sample and group of channels / groups channels,
    then a weight and bias per channel. x is [N, C, ...]."""

    groups: int
    channels: int
    eps: float = 1e-5
    affine: bool = True
    dtype: str = "f32"

    def __post_init__(self):
        if self.channels % self.groups:
            raise ValueError(f"channels {self.channels} must be a multiple of groups {self.groups}")

    def params(self):
        if not self.affine:
            return {}
        return {"w": Param((self.channels,), ones), "b": Param((self.channels,), zeros)}

    def forward(self, p, x):
        if self.affine:
            return ops.group_norm(x, self.groups, p["w"], p["b"], self.eps)
        return ops.group_norm(x, self.groups, None, None, self.eps)


@layer
class InstanceNorm(Layer):
    """Normalization over the spatial dims of each sample and channel (torch's InstanceNorm1d, 2d,
    3d with track_running_stats=False): group norm with one group per channel."""

    channels: int
    eps: float = 1e-5
    affine: bool = False
    dtype: str = "f32"

    def params(self):
        if not self.affine:
            return {}
        return {"w": Param((self.channels,), ones), "b": Param((self.channels,), zeros)}

    def forward(self, p, x):
        if self.affine:
            return ops.group_norm(x, self.channels, p["w"], p["b"], self.eps)
        return ops.group_norm(x, self.channels, None, None, self.eps)


# ----------------------------------------------------------------------------- convolution


@layer
class Conv1d(Layer):
    """1D convolution over [N, C, L]; weight [c_out, c_in / groups, k]."""

    c_in: int
    c_out: int
    kernel: int
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    dtype: str = "f32"

    def __post_init__(self):
        if self.c_in % self.groups or self.c_out % self.groups:
            raise ValueError(f"c_in {self.c_in} and c_out {self.c_out} must both be multiples of groups {self.groups}")

    def params(self):
        fan = self.c_in // self.groups * self.kernel
        out = {"w": Param((self.c_out, self.c_in // self.groups, self.kernel), fan_in_uniform(fan))}
        if self.bias:
            out["b"] = Param((self.c_out,), fan_in_uniform(fan))
        return out

    def forward(self, p, x):
        n, c, length = x.shape
        x4 = ops.reshape(x, [n, c, 1, length])
        w4 = ops.reshape(p["w"], [self.c_out, self.c_in // self.groups, 1, self.kernel])
        b = p["b"] if self.bias else None
        y = ops.conv2d(x4, w4, b, stride=[1, self.stride], padding=[0, self.padding], dilation=[1, self.dilation],
                       groups=self.groups)
        return ops.reshape(y, [y.shape[0], y.shape[1], y.shape[3]])


@layer
class ConvTranspose2d(Layer):
    """The transposed 2D convolution (the adjoint of Conv2d) over [N, C, H, W]; weight
    [c_in, c_out / groups, k, k] as torch.nn.ConvTranspose2d."""

    c_in: int
    c_out: int
    kernel: int
    stride: int = 1
    padding: int = 0
    output_padding: int = 0
    groups: int = 1
    dilation: int = 1
    bias: bool = True
    dtype: str = "f32"

    def __post_init__(self):
        if self.c_in % self.groups or self.c_out % self.groups:
            raise ValueError(f"c_in {self.c_in} and c_out {self.c_out} must both be multiples of groups {self.groups}")
        if self.output_padding >= self.stride:
            raise ValueError(f"output_padding {self.output_padding} must be smaller than stride {self.stride}")

    def params(self):
        fan = self.c_out // self.groups * self.kernel * self.kernel   # torch's fan in for this layout
        out = {"w": Param((self.c_in, self.c_out // self.groups, self.kernel, self.kernel), fan_in_uniform(fan))}
        if self.bias:
            out["b"] = Param((self.c_out,), fan_in_uniform(fan))
        return out

    def forward(self, p, x):
        b = p["b"] if self.bias else None
        return ops.conv_transpose2d(x, p["w"], b, stride=self.stride, padding=self.padding,
                                    output_padding=self.output_padding, groups=self.groups, dilation=self.dilation)


# ----------------------------------------------------------------------------- resampling


@layer
class Upsample(Layer):
    """Resize the last two dims of [N, C, H, W] by an integer ``scale``: ``nearest`` reads source row
    floor(i / scale); ``bilinear`` interpolates between the two nearest source rows and columns at
    (i + 1/2) / scale - 1/2 (clamped at 0), or at i (H - 1) / (H scale - 1) with align_corners, as
    torch.nn.Upsample."""

    scale: int = 2
    mode: str = "nearest"
    align_corners: bool = False

    def __post_init__(self):
        if self.mode not in ("nearest", "bilinear"):
            raise ValueError(f"mode must be nearest or bilinear, got {self.mode!r}")

    def forward(self, x):
        y = _resize(x, 2, self.scale, self.mode, self.align_corners)
        return _resize(y, 3, self.scale, self.mode, self.align_corners)


@inline
def _resize(x, d, scale, mode, align_corners):
    size = x.shape[d]
    out = size * scale
    i = ops.cast(ops.arange(out), x.dtype)
    if mode == "nearest":
        src = ops.cast(i * (1.0 / scale), "i64")
        return ops.index_select(x, d, src)
    if align_corners:
        pos = i * ((size - 1) / (out - 1))     # out >= 2: scale >= 2 and size >= 1
    else:
        pos = ops.clamp((i + 0.5) * (1.0 / scale) - 0.5, min=0.0)
    i0 = ops.cast(pos, "i64")
    i1 = ops.minimum(i0 + 1, ops.full_like(i0, size - 1))
    lam = pos - ops.cast(i0, x.dtype)
    a = ops.index_select(x, d, i0)
    b = ops.index_select(x, d, i1)
    if d == 2:
        lam = ops.reshape(lam, [out, 1])
    return a * (1.0 - lam) + b * lam


@layer
class PixelShuffle(Layer):
    """[N, C r^2, H, W] -> [N, C, H r, W r]: out[c, h r + i, w r + j] = x[c r^2 + i r + j, h, w]."""

    factor: int

    def forward(self, x):
        n, c, h, w = x.shape
        r = self.factor
        v = ops.reshape(x, [n, c // (r * r), r, r, h, w])
        v = ops.permute(v, [0, 1, 4, 2, 5, 3])
        return ops.reshape(v, [n, c // (r * r), h * r, w * r])


# ----------------------------------------------------------------------------- recurrent layers


@layer
class LSTM(Layer):
    """One layer LSTM over [B, T, d_in] (batch first) with zero initial state. Gates i, f, g, o in
    torch.nn.LSTM's order and parameter layout (w_ih [4 H, d_in], w_hh [4 H, H], b_ih, b_hh). The
    time steps are an IR loop, so one compile serves every sequence length; the outputs are
    collected with a one hot sum (O(T^2) work, exact). Returns (out [B, T, H], h, c)."""

    d_in: int
    hidden: int
    bias: bool = True
    dtype: str = "f32"

    def params(self):
        h = self.hidden
        out = {"w_ih": Param((4 * h, self.d_in), fan_in_uniform(h)), "w_hh": Param((4 * h, h), fan_in_uniform(h))}
        if self.bias:
            out["b_ih"] = Param((4 * h,), fan_in_uniform(h))
            out["b_hh"] = Param((4 * h,), fan_in_uniform(h))
        return out

    def forward(self, p, x):
        b, steps, _ = x.shape
        hd = self.hidden
        h = ops.zeros((b, hd), dtype=self.dtype)
        c = ops.zeros((b, hd), dtype=self.dtype)
        out = ops.zeros((b, steps, hd), dtype=self.dtype)
        slots = ops.arange(steps)
        for t in range(steps):
            if self.bias:
                g = ops.linear(x[:, t], p["w_ih"], p["b_ih"]) + ops.linear(h, p["w_hh"], p["b_hh"])
            else:
                g = ops.linear(x[:, t], p["w_ih"]) + ops.linear(h, p["w_hh"])
            i = ops.sigmoid(g[:, 0:hd])
            f = ops.sigmoid(g[:, hd:2 * hd])
            cand = ops.tanh(g[:, 2 * hd:3 * hd])
            o = ops.sigmoid(g[:, 3 * hd:4 * hd])
            c = f * c + i * cand
            h = o * ops.tanh(c)
            here = ops.cast(slots == t, self.dtype)
            out = out + ops.reshape(h, [b, 1, hd]) * ops.reshape(here, [1, steps, 1])
        return out, h, c


@layer
class GRU(Layer):
    """One layer GRU over [B, T, d_in] (batch first), zero initial state, torch.nn.GRU's gates (r, z,
    n) and layout: n = tanh(W_in x + b_in + r (W_hn h + b_hn)), h = (1 - z) n + z h. An IR loop over
    time like LSTM. Returns (out [B, T, H], h)."""

    d_in: int
    hidden: int
    bias: bool = True
    dtype: str = "f32"

    def params(self):
        h = self.hidden
        out = {"w_ih": Param((3 * h, self.d_in), fan_in_uniform(h)), "w_hh": Param((3 * h, h), fan_in_uniform(h))}
        if self.bias:
            out["b_ih"] = Param((3 * h,), fan_in_uniform(h))
            out["b_hh"] = Param((3 * h,), fan_in_uniform(h))
        return out

    def forward(self, p, x):
        b, steps, _ = x.shape
        hd = self.hidden
        h = ops.zeros((b, hd), dtype=self.dtype)
        out = ops.zeros((b, steps, hd), dtype=self.dtype)
        slots = ops.arange(steps)
        for t in range(steps):
            if self.bias:
                gi = ops.linear(x[:, t], p["w_ih"], p["b_ih"])
                gh = ops.linear(h, p["w_hh"], p["b_hh"])
            else:
                gi = ops.linear(x[:, t], p["w_ih"])
                gh = ops.linear(h, p["w_hh"])
            r = ops.sigmoid(gi[:, 0:hd] + gh[:, 0:hd])
            z = ops.sigmoid(gi[:, hd:2 * hd] + gh[:, hd:2 * hd])
            n = ops.tanh(gi[:, 2 * hd:3 * hd] + r * gh[:, 2 * hd:3 * hd])
            h = n + z * (h - n)
            here = ops.cast(slots == t, self.dtype)
            out = out + ops.reshape(h, [b, 1, hd]) * ops.reshape(here, [1, steps, 1])
        return out, h


# ----------------------------------------------------------------------------- embeddings


@layer
class EmbeddingBag(Layer):
    """The sum, mean or max of the rows of each bag: ids [B, L] give B bags of L rows each (torch's
    EmbeddingBag with 2D input); ``per_sample_weights`` [B, L] scale the rows (sum mode)."""

    count: int
    dim: int
    mode: str = "mean"
    std: float = 1.0
    dtype: str = "f32"

    def __post_init__(self):
        if self.mode not in ("sum", "mean", "max"):
            raise ValueError(f"mode must be sum, mean or max, got {self.mode!r}")

    def params(self):
        return {"w": Param((self.count, self.dim), normal(self.std))}

    def forward(self, p, ids, per_sample_weights=None):
        b, length = ids.shape
        rows = ops.reshape(ops.index_select(p["w"], 0, ops.reshape(ids, [b * length])), [b, length, self.dim])
        if per_sample_weights is not None:
            rows = rows * ops.reshape(per_sample_weights, [b, length, 1])
        if self.mode == "sum":
            return ops.sum(rows, dim=1)
        if self.mode == "mean":
            return ops.mean(rows, dim=1)
        return ops.amax(rows, dim=1)


# ----------------------------------------------------------------------------- positions


@layer
class Rotary(Layer):
    """Rotary position embedding (RoPE) in the rotate half layout of Llama: for head dim D,
    frequency f_i = base^(-2 i / D) for i < D / 2, angle a = pos f, and
    y = x cos(a) + rotate_half(x) sin(a) with rotate_half(x) = cat(-x[D/2:], x[:D/2]) and both angle
    halves repeated. ``rot(x, pos)``: x [..., T, D], pos int [T]."""

    head_dim: int
    base: float = 10000.0

    def __post_init__(self):
        if self.head_dim % 2:
            raise ValueError(f"head_dim must be even, got {self.head_dim}")

    def forward(self, x, pos):
        half = self.head_dim // 2
        k = ops.cast(ops.arange(half), "f64")
        freq = ops.exp(k * (-2.0 * math.log(self.base) / self.head_dim))
        ang = ops.reshape(ops.cast(pos, "f64"), [pos.shape[0], 1]) * ops.reshape(freq, [1, half])
        cos = ops.cast(ops.cos(ang), x.dtype)
        sin = ops.cast(ops.sin(ang), x.dtype)
        cos = ops.cat([cos, cos], dim=-1)
        sin = ops.cat([sin, sin], dim=-1)
        rotated = ops.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rotated * sin


@layer
class ALiBi(Layer):
    """Attention with linear biases (Press et al. 2022): head h of H gets slope m_h = 2^(-8 h / H)
    for h = 1..H when H is a power of two; otherwise the slopes of the next lower power of two and
    every other slope of the next higher one. ``alibi(q_len, k_len)`` returns the additive bias
    [H, q_len, k_len] = m_h (j - i), to add to the attention scores (with a causal mask, j <= i)."""

    heads: int

    def slopes(self):
        def powers(n):
            start = 2.0 ** (-8.0 / n)
            return [start ** (i + 1) for i in range(n)]
        n = self.heads
        low = 1
        while low * 2 <= n:
            low *= 2
        out = powers(low)
        if low < n:
            out = out + powers(2 * low)[0::2][: n - low]
        return out

    def forward(self, q_len, k_len, dtype="f32"):
        i = ops.reshape(ops.cast(ops.arange(q_len), dtype), [1, q_len, 1])
        j = ops.reshape(ops.cast(ops.arange(k_len), dtype), [1, 1, k_len])
        m = ops.zeros((self.heads, 1, 1), dtype=dtype)
        for h, s in enumerate(self.slopes()):
            m = m + ops.cast(ops.reshape(ops.arange(self.heads), [self.heads, 1, 1]) == h, dtype) * s
        return m * (j - i)


# ----------------------------------------------------------------------------- gated feed forward


@layer
class GatedMLP(Layer):
    """w2(act(w1 x) * w3 x): SwiGLU with act = silu (Llama's MLP), GEGLU with act = gelu."""

    dim: int
    hidden: int
    act: str = "silu"
    bias: bool = False
    dtype: str = "f32"

    def __post_init__(self):
        if self.act not in ("silu", "gelu", "gelu_tanh"):
            raise ValueError(f"act must be silu, gelu or gelu_tanh, got {self.act!r}")

    def params(self):
        out = {"w1": Param((self.hidden, self.dim), fan_in_uniform(self.dim)),
               "w3": Param((self.hidden, self.dim), fan_in_uniform(self.dim)),
               "w2": Param((self.dim, self.hidden), fan_in_uniform(self.hidden))}
        if self.bias:
            out["b1"] = Param((self.hidden,), fan_in_uniform(self.dim))
            out["b3"] = Param((self.hidden,), fan_in_uniform(self.dim))
            out["b2"] = Param((self.dim,), fan_in_uniform(self.hidden))
        return out

    def forward(self, p, x):
        if self.bias:
            a = ops.linear(x, p["w1"], p["b1"])
            u = ops.linear(x, p["w3"], p["b3"])
        else:
            a = ops.linear(x, p["w1"])
            u = ops.linear(x, p["w3"])
        if self.act == "silu":
            g = ops.silu(a)
        elif self.act == "gelu":
            g = ops.gelu(a)
        else:
            g = ops.gelu(a, approximate="tanh")
        if self.bias:
            return ops.linear(g * u, p["w2"], p["b2"])
        return ops.linear(g * u, p["w2"])


def SwiGLU(dim: int, hidden: int, bias: bool = False, dtype: str = "f32"):
    return GatedMLP(dim, hidden, "silu", bias, dtype)


def GEGLU(dim: int, hidden: int, bias: bool = False, dtype: str = "f32"):
    return GatedMLP(dim, hidden, "gelu", bias, dtype)
