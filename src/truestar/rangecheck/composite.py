"""Composite ops written once, with a rounding hook and a recorder for hidden intermediates.

Fused kernels hide intermediates that the source never names: x minus mean squared inside a layer
norm, the mean of squares inside an RMS norm, the cube inside a tanh gelu, the sum of exponentials
inside a softmax, the attention scores. A runtime that runs these ops decomposed stores every one of
them in its low precision format, and that is where real overflows happened (onnxruntime#31626: the
``Pow`` of a decomposed layer norm reached about 630,000 against float16's 65,504).

Each function takes ``r`` (rounding applied after every step: identity for the float64 observation,
the format's rounding for a decomposed target) and ``rec(name, tensor)``, called with every hidden
intermediate. The math follows the ONNX decomposition (ReduceMean, Sub, Pow, ReduceMean, Add, Sqrt,
Div, Mul, Add for layer norm), which is the form decomposing runtimes execute.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import torch

from ..ir.reference import IMPLS, _dims

R = Callable[[torch.Tensor], torch.Tensor]
Rec = Callable[[str, torch.Tensor], None]


def _trailing(x: torch.Tensor, n: int) -> List[int]:
    return list(range(x.dim() - n, x.dim()))


def _mean(x: torch.Tensor, dims: List[int], keepdim: bool = True) -> torch.Tensor:
    count = 1
    for d in dims:
        count *= x.shape[d]
    return x.sum(dim=dims, keepdim=keepdim) / count


def _const(r: R, v: float, like: torch.Tensor) -> torch.Tensor:
    return r(torch.tensor(float(v), dtype=like.dtype))


def layer_norm(attrs, args, r: R, rec: Rec) -> torch.Tensor:
    x, w, b = args
    dims = _trailing(x, len(attrs["normalized_shape"]))
    mu = r(_mean(x, dims))
    d = r(x - mu)
    rec("x - mean", d)
    sq = r(d * d)
    rec("(x - mean)^2", sq)
    var = r(_mean(sq, dims))
    rec("variance", var)
    sd = r(torch.sqrt(r(var + _const(r, attrs["eps"], x))))
    y = r(d / sd)
    if w is not None:
        y = r(y * w)
    if b is not None:
        y = r(y + b)
    return y


def group_norm(attrs, args, r: R, rec: Rec) -> torch.Tensor:
    x, w, b = args
    n, c = x.shape[0], x.shape[1]
    xg = x.reshape(n, attrs["num_groups"], -1)
    mu = r(_mean(xg, [2]))
    d = r(xg - mu)
    rec("x - mean", d)
    sq = r(d * d)
    rec("(x - mean)^2", sq)
    var = r(_mean(sq, [2]))
    rec("variance", var)
    y = r(d / r(torch.sqrt(r(var + _const(r, attrs["eps"], x))))).reshape(x.shape)
    view = (1, c) + (1,) * (x.dim() - 2)
    if w is not None:
        y = r(y * w.reshape(view))
    if b is not None:
        y = r(y + b.reshape(view))
    return y


def rms_norm(attrs, args, r: R, rec: Rec) -> torch.Tensor:
    x, w = args
    dims = _trailing(x, len(attrs["normalized_shape"]))
    sq = r(x * x)
    rec("x^2", sq)
    ms = r(_mean(sq, dims))
    rec("mean(x^2)", ms)
    eps = attrs.get("eps")
    if eps is None:
        eps = torch.finfo(torch.float32).eps
    y = r(x * r(1.0 / r(torch.sqrt(r(ms + _const(r, eps, x))))))
    return y if w is None else r(y * w)


def softmax_(x: torch.Tensor, dim: int, r: R, rec: Rec) -> torch.Tensor:
    m = x.amax(dim=dim, keepdim=True)
    m = torch.where(torch.isinf(m) & (m < 0), torch.zeros_like(m), m)
    e = r(torch.exp(r(x - m)))
    s = r(e.sum(dim=dim, keepdim=True))
    rec("sum(exp)", s)
    return r(e / s)


def softmax(attrs, args, r: R, rec: Rec) -> torch.Tensor:
    return softmax_(args[0], attrs["dim"], r, rec)


def gelu(attrs, args, r: R, rec: Rec) -> torch.Tensor:
    x = args[0]
    if attrs.get("approximate", "none") != "tanh":
        return r(IMPLS["gelu"](attrs, args, tuple(x.shape)))
    x3 = r(r(x * x) * x)
    rec("x^3", x3)
    inner = r(x + r(_const(r, 0.044715, x) * x3))
    t = torch.tanh(r(_const(r, math.sqrt(2.0 / math.pi), x) * inner))
    return r(r(_const(r, 0.5, x) * x) * r(1.0 + r(t)))


def vector_norm(attrs, args, r: R, rec: Rec) -> torch.Tensor:
    x = args[0]
    p = float(attrs.get("ord", 2.0))
    if p != 2:
        return r(IMPLS["vector_norm"](attrs, args, ()))
    dims = _dims(attrs, x)
    sq = r(x * x)
    rec("x^2", sq)
    s = r(sq.sum(dim=dims, keepdim=attrs.get("keepdim", False)))
    rec("sum(x^2)", s)
    return r(torch.sqrt(s))


def attention(attrs, args, r: R, rec: Rec) -> torch.Tensor:
    q, k, v, mask = args
    scale = attrs.get("scale")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    scores = r(torch.matmul(q, k.transpose(-2, -1)) * scale)
    rec("q k^T * scale", scores)
    L, S = q.shape[-2], k.shape[-2]
    if attrs.get("is_causal", False):
        keep = torch.ones((L, S), dtype=torch.bool).tril()
        scores = torch.where(keep, scores, torch.full_like(scores, -math.inf))
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = torch.where(mask, scores, torch.full_like(scores, -math.inf))
        else:
            scores = r(scores + mask)
    p = softmax_(scores, -1, r, rec)
    return r(torch.matmul(p, v))


COMPOSITES: Dict[str, Callable] = {
    "layer_norm": layer_norm, "group_norm": group_norm, "rms_norm": rms_norm, "softmax": softmax,
    "gelu": gelu, "vector_norm": vector_norm, "attention": attention,
}

# Hidden intermediates whose overflow corrupts the op's result. The exponentials inside sigmoid and
# the masked scores are left out on purpose: exp overflowing inside 1 / (1 + exp(-x)) still gives the
# right answer (0), so flagging it would be a false alarm. The same holds for the cube inside a tanh
# gelu: tanh of an infinite argument is +-1 and the result is x or 0, as it should be.
HARMFUL_HIDDEN = {
    "layer_norm": ("(x - mean)^2", "variance", "x - mean"),
    "group_norm": ("(x - mean)^2", "variance", "x - mean"),
    "rms_norm": ("x^2", "mean(x^2)"),
    "softmax": ("sum(exp)",),
    "vector_norm": ("x^2", "sum(x^2)"),
    "attention": ("q k^T * scale", "sum(exp)"),
}


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def no_record(name: str, x: torch.Tensor) -> None:
    return None
