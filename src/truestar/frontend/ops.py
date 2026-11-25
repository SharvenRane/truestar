"""The truestar ops namespace for native functions.

Inside an ``@truestar.function`` these calls are translated to IR ops at compile time. Outside a
compiled function each one runs eagerly through PyTorch, so the same source can be executed as plain
Python (``f.python(...)``) for comparison.

    from truestar import ops
    y = ops.softmax(ops.relu(x @ w), dim=-1)
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .types import DTYPES


def _op(name: str):
    def deco(fn):
        fn.__truestar_op__ = name
        return fn
    return deco


def _dt(dtype):
    if dtype is None:
        return torch.float32
    return DTYPES[dtype] if isinstance(dtype, str) else dtype


# ----------------------------------------------------------------------------- elementwise


@_op("relu")
def relu(x):
    return torch.relu(x)


@_op("gelu")
def gelu(x, approximate: str = "none"):
    return F.gelu(x, approximate=approximate)


@_op("sigmoid")
def sigmoid(x):
    return torch.sigmoid(x)


@_op("tanh")
def tanh(x):
    return torch.tanh(x)


@_op("silu")
def silu(x):
    return F.silu(x)


@_op("exp")
def exp(x):
    return torch.exp(x)


@_op("sqrt")
def sqrt(x):
    return torch.sqrt(x)


@_op("rsqrt")
def rsqrt(x):
    return torch.rsqrt(x)


@_op("abs")
def abs(x):  # noqa: A001 (mirrors torch.abs)
    return torch.abs(x)


@_op("clamp")
def clamp(x, min: Optional[float] = None, max: Optional[float] = None):  # noqa: A002
    return torch.clamp(x, min=min, max=max)


@_op("maximum")
def maximum(a, b):
    a, b = _pair(a, b)
    return torch.where(a > b, a, b)


@_op("minimum")
def minimum(a, b):
    a, b = _pair(a, b)
    return torch.where(a < b, a, b)


def _pair(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a, dtype=b.dtype)
    if not isinstance(b, torch.Tensor):
        b = torch.as_tensor(b, dtype=a.dtype)
    return a, b


@_op("where")
def where(condition, a, b):
    return torch.where(condition, a, b)


# ----------------------------------------------------------------------------- reductions


@_op("sum")
def sum(x, dim=None, keepdim: bool = False):  # noqa: A001
    return x.sum() if dim is None and not keepdim else x.sum(dim=_dims(x, dim), keepdim=keepdim)


@_op("mean")
def mean(x, dim=None, keepdim: bool = False):
    return x.mean() if dim is None and not keepdim else x.mean(dim=_dims(x, dim), keepdim=keepdim)


@_op("any")
def any(x, dim=None, keepdim: bool = False):  # noqa: A001
    if dim is None and not keepdim:
        return x.any()
    out = x
    for d in sorted([d % x.dim() for d in _dims(x, dim)], reverse=True):
        out = out.any(dim=d, keepdim=keepdim)
    return out


@_op("all")
def all(x, dim=None, keepdim: bool = False):  # noqa: A001
    return ~any(~x.bool(), dim=dim, keepdim=keepdim)


def _dims(x, dim):
    if dim is None:
        return list(range(x.dim()))
    return [dim] if isinstance(dim, int) else list(dim)


@_op("softmax")
def softmax(x, dim: int = -1, temperature=None):
    """softmax(x / temperature) along dim; ``temperature`` is a number or a scalar tensor."""
    return torch.softmax(x if temperature is None else x / temperature, dim=dim)


# ----------------------------------------------------------------------------- linear algebra, norms


@_op("matmul")
def matmul(a, b):
    return torch.matmul(a, b)


@_op("linear")
def linear(x, weight, bias=None):
    return F.linear(x, weight, bias)


@_op("layer_norm")
def layer_norm(x, normalized_shape: Sequence[int], weight=None, bias=None, eps: float = 1e-5):
    return F.layer_norm(x, list(normalized_shape), weight, bias, eps)


@_op("attention")
def attention(q, k, v, mask=None, is_causal: bool = False, scale: Optional[float] = None,
              enable_gqa: bool = False, window: Optional[int] = None):
    """Scaled dot product attention. With ``enable_gqa`` k and v may have fewer heads (dim -3) than q
    (grouped query attention; one KV head is multi query attention). ``window`` (with ``is_causal``)
    keeps, for query i, only the keys i - window + 1 to i: a sliding window, aligned top left like
    ``is_causal``."""
    mask, is_causal = _window_mask(q, k, mask, is_causal, window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal, scale=scale,
                                          enable_gqa=enable_gqa)


# ----------------------------------------------------------------------------- shape ops


@_op("reshape")
def reshape(x, shape):
    return x.reshape(*shape)


@_op("permute")
def permute(x, dims):
    return x.permute(*dims)


@_op("transpose")
def transpose(x, dim0: int, dim1: int):
    return x.transpose(dim0, dim1)


@_op("cat")
def cat(tensors, dim: int = 0):
    return torch.cat(list(tensors), dim=dim)


# ----------------------------------------------------------------------------- creation


@_op("full")
def full(shape, value, dtype="f32"):
    return torch.full(tuple(int(d) for d in shape), value, dtype=_dt(dtype))


@_op("zeros")
def zeros(shape, dtype="f32"):
    return torch.zeros(tuple(int(d) for d in shape), dtype=_dt(dtype))


@_op("ones")
def ones(shape, dtype="f32"):
    return torch.ones(tuple(int(d) for d in shape), dtype=_dt(dtype))


@_op("zeros_like")
def zeros_like(x):
    return torch.zeros_like(x)


@_op("ones_like")
def ones_like(x):
    return torch.ones_like(x)


@_op("full_like")
def full_like(x, value):
    return torch.full_like(x, value)



# ----------------------------------------------------------------------------- decoder LLM ops (2026-11-24)
# See docs/LLM.md. Eager versions follow PyTorch; inside a compiled function each becomes IR ops.


@_op("cos")
def cos(x):
    return torch.cos(x)


@_op("sin")
def sin(x):
    return torch.sin(x)


@_op("log")
def log(x):
    return torch.log(x)


@_op("cast")
def cast(x, dtype):
    return x.to(_dt(dtype))


# ----------------------------------------------------------------------------- layer library ops (2026-11-24)
# Used by truestar.nn (docs/NN.md); each is an existing IR op or a short composition of them.


@_op("expm1")
def expm1(x):
    """exp(x) - 1, accurate near 0 (no cancellation): a degree 9 series on [-1/2, 1/2], exp(x) - 1
    outside, where the subtraction loses at most about one bit."""
    return torch.expm1(x)


@_op("log_softmax")
def log_softmax(x, dim: int = -1):
    """x - max - log(sum(exp(x - max))) along dim, as one op with its own gradient rule."""
    return torch.log_softmax(x, dim=dim)


@_op("conv2d")
def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups: int = 1):
    """2D cross correlation over [N, C, H, W] with weight [O, C / groups, kH, kW]."""
    return F.conv2d(x, weight, bias, stride, padding, dilation, groups)


@_op("conv_transpose2d")
def conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups: int = 1, dilation=1):
    """The adjoint of conv2d over [N, C_in, H, W] with weight [C_in, C_out / groups, kH, kW]; the
    output size is (H - 1) stride - 2 padding + dilation (k - 1) + output_padding + 1 (added
    2026-11-24 with the op coverage work; IR ``conv2d_input_grad``)."""
    return F.conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups, dilation)


@_op("group_norm")
def group_norm(x, num_groups: int, weight=None, bias=None, eps: float = 1e-5):
    """Per sample and group of C / num_groups channels: (x - mean) / sqrt(biased variance + eps), then
    weight and bias per channel ([N, C, ...]; added 2026-11-24, IR ``group_norm``)."""
    return F.group_norm(x, num_groups, weight, bias, eps)


@_op("max_pool2d")
def max_pool2d(x, kernel_size, stride=None, padding=0):
    return F.max_pool2d(x, kernel_size, stride, padding)


@_op("avg_pool2d")
def avg_pool2d(x, kernel_size: int):
    """Mean over non overlapping kernel_size windows (stride = kernel_size); H and W must be
    multiples of kernel_size, so every window is full and none is padded."""
    return F.avg_pool2d(x, kernel_size)


@_op("adaptive_avg_pool2d")
def adaptive_avg_pool2d(x, output_size):
    return F.adaptive_avg_pool2d(x, output_size)


@_op("arange")
def arange(start, end=None, step: int = 1, dtype="i64"):
    if end is None:
        start, end = 0, start
    return torch.arange(int(start), int(end), int(step), dtype=_dt(dtype))


@_op("rms_norm")
def rms_norm(x, normalized_shape, weight=None, eps: float = 1e-6):
    ns = [normalized_shape] if isinstance(normalized_shape, int) else list(normalized_shape)
    dims = list(range(x.dim() - len(ns), x.dim()))
    y = x * torch.rsqrt((x * x).mean(dim=dims, keepdim=True) + eps)
    return y if weight is None else y * weight


@_op("cumsum")
def cumsum(x, dim: int):
    return torch.cumsum(x, dim=dim)


@_op("amax")
def amax(x, dim=None, keepdim: bool = False):
    return torch.amax(x, dim=() if dim is None else dim, keepdim=keepdim)


@_op("amin")
def amin(x, dim=None, keepdim: bool = False):
    return torch.amin(x, dim=() if dim is None else dim, keepdim=keepdim)


@_op("flip")
def flip(x, dims):
    return torch.flip(x, dims=[dims] if isinstance(dims, int) else list(dims))


@_op("argmax")
def argmax(x, dim=None, keepdim: bool = False):
    return torch.argmax(x, dim=dim, keepdim=keepdim)


@_op("argmin")
def argmin(x, dim=None, keepdim: bool = False):
    return torch.argmin(x, dim=dim, keepdim=keepdim)


@_op("sort")
def sort(x, dim: int = -1, descending: bool = False):
    """(values, indices); stable: equal values keep their input order."""
    out = torch.sort(x, dim=dim, descending=descending, stable=True)
    return out.values, out.indices


@_op("topk")
def topk(x, k: int, dim: int = -1, largest: bool = True):
    out = torch.topk(x, k, dim=dim, largest=largest, sorted=True)
    return out.values, out.indices


@_op("gather")
def gather(x, dim: int, index):
    return torch.gather(x, dim, index)


@_op("index_select")
def index_select(x, dim: int, index):
    return torch.index_select(x, dim, index)


@_op("embedding")
def embedding(weight, ids):
    """weight[ids]: one row of ``weight`` per id."""
    return F.embedding(ids, weight)


@_op("index_copy")
def index_copy(x, dim: int, index, src):
    """x with x.select(dim, index[j]) replaced by src.select(dim, j): a KV cache write at run time
    positions."""
    return torch.index_copy(x, dim, index, src)


@_op("masked_fill")
def masked_fill(x, mask, value):
    return x.masked_fill(mask, value)


@_op("repeat_interleave")
def repeat_interleave(x, repeats: int, dim: int):
    return torch.repeat_interleave(x, repeats, dim=dim)


@_op("chunk")
def chunk(x, chunks: int, dim: int = 0):
    return tuple(torch.chunk(x, chunks, dim=dim))


@_op("split")
def split(x, split_size, dim: int = 0):
    return tuple(torch.split(x, split_size if isinstance(split_size, int) else list(split_size), dim=dim))


def _window_mask(q, k, mask, is_causal, window):
    if window is None:
        return mask, is_causal
    if not is_causal:
        raise ValueError("window needs is_causal=True")
    i = torch.arange(q.shape[-2]).reshape(-1, 1)
    j = torch.arange(k.shape[-2]).reshape(1, -1)
    keep = (j <= i) & (j > i - window)
    if mask is None:
        return keep, False
    if mask.dtype == torch.bool:
        return keep & mask, False
    return mask.masked_fill(~keep, float("-inf")), False


@_op("multinomial")
def multinomial(weights, num_samples: int = 1, seed: int = 0, offset=None):
    """Category indices drawn by inverse CDF from the IR's counter based generator (docs/LLM.md):
    deterministic for a given seed and offset, identical inside and outside a compiled function."""
    from ..ir.reference import uniform01

    off = 0 if offset is None else int(offset)
    rows = weights.reshape(1, -1) if weights.dim() == 1 else weights
    out = torch.zeros((rows.shape[0], num_samples), dtype=torch.int64)
    for r in range(rows.shape[0]):
        w = rows[r].double()
        cum = torch.cumsum(w, 0)
        last = int(torch.nonzero(w > 0).max())
        for s in range(num_samples):
            target = uniform01(seed, off, r * num_samples + s) * float(cum[-1])
            out[r, s] = min(int((cum <= target).sum()), last)
    return out.reshape(*weights.shape[:-1], num_samples)


@_op("top_p_filter")
def top_p_filter(logits, p: float, min_keep: int = 1):
    """Nucleus filtering along the last dim: logits outside the smallest set of top tokens whose
    probability reaches ``p`` become -inf (the cumsum recipe of Hugging Face's TopPLogitsWarper)."""
    sorted_logits, idx = torch.sort(logits, dim=-1, descending=False, stable=True)
    cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cum <= (1 - p)
    remove[..., -min_keep:] = False
    remove = remove.scatter(-1, idx, remove)
    return logits.masked_fill(remove, float("-inf"))

# ----------------------------------------------------------------------------- random and training (docs/RANDOM.md)
#
# Every draw is a pure function of (key, name, element index), the same bits on every backend. Inside
# an @truestar.function the key defaults to one derived from the function's seed and call count and
# the name to the op's site in the source (``fn/dropout#0``), so neither needs writing. Eagerly there
# is no function to derive them from: ``key`` is required and ``name`` defaults to the op's name.


def _eager_key(op: str, key) -> int:
    if key is None:
        raise TypeError(f"ops.{op} outside an @truestar.function needs key= (an int): eager code has no "
                        "function seed and call count to derive one from. Inside a compiled function the key "
                        "is derived for you (docs/RANDOM.md)")
    if isinstance(key, torch.Tensor):
        key = int(key.item())
    return int(key)


def _draw(op, dist, shape, dtype, key, name, device="cpu", **kw):
    from .. import rng

    shape = (int(shape),) if isinstance(shape, int) else tuple(int(d) for d in shape)
    return rng.generate(dist, shape, _dt(dtype) if dtype != "bool" else torch.bool, _eager_key(op, key),
                        op if name is None else name, device=device, **kw)


@_op("rand")
def rand(shape, dtype="f32", *, key=None, name=None):
    """Uniform in [0, 1)."""
    return _draw("rand", "uniform", shape, dtype, key, name)


@_op("randn")
def randn(shape, dtype="f32", *, key=None, name=None):
    """Standard normal."""
    return _draw("randn", "normal", shape, dtype, key, name)


@_op("normal")
def normal(shape, mean=0.0, std=1.0, dtype="f32", *, key=None, name=None):
    """mean + std * a standard normal draw."""
    return _draw("normal", "normal", shape, dtype, key, name) * std + mean


@_op("bernoulli")
def bernoulli(shape, p=0.5, dtype="bool", *, key=None, name=None):
    """True (or 1) with probability p."""
    return _draw("bernoulli", "bernoulli", shape, dtype, key, name, p=float(p))


@_op("randint")
def randint(low, high, shape, dtype="i64", *, key=None, name=None):
    """Integers in [low, high)."""
    return _draw("randint", "randint", shape, dtype, key, name, low=int(low), high=int(high))


@_op("rand_like")
def rand_like(x, *, key=None, name=None):
    return _draw("rand_like", "uniform", x.shape, x.dtype, key, name, device=x.device)


@_op("randn_like")
def randn_like(x, *, key=None, name=None):
    return _draw("randn_like", "normal", x.shape, x.dtype, key, name, device=x.device)


@_op("dropout")
def dropout(x, p=0.5, training=True, *, key=None, name=None):
    """Inverted dropout with a keyed mask; identity when ``training`` is False or p is 0."""
    from .. import rng

    if not training or p == 0:
        return x
    return rng.dropout(x, float(p), _eager_key("dropout", key), "dropout" if name is None else name)


@_op("batch_norm")
def batch_norm(x, weight=None, bias=None, running_mean=None, running_var=None, training=False, momentum=0.1,
               eps=1e-5):
    """Batch norm over dim 1. ``training=False``: normalize with the running statistics, return y.
    ``training=True``: normalize with the batch statistics and return ``(y, new_running_mean,
    new_running_var)``; the inputs are never changed (state in, state out)."""
    if not training:
        return F.batch_norm(x, running_mean, running_var, weight, bias, False, 0.0, eps)
    if running_mean is None:
        return F.batch_norm(x, None, None, weight, bias, True, 0.0, eps)
    rm, rv = running_mean.clone(), running_var.clone()
    y = F.batch_norm(x, rm, rv, weight, bias, True, momentum, eps)
    return y, rm, rv


__all__ = [n for n, v in dict(globals()).items() if hasattr(v, "__truestar_op__")]
