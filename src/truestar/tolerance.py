"""Derived tolerances: `truestar.assert_close(fn, inputs, out)` with no rtol or atol to guess.

The question a tolerance answers is "how far from exact can a correct implementation of fn be, on
these inputs, in this precision?". Truestar answers it per element, from fn itself:

1. Reference: fn in float64 on the inputs widened exactly (on the device of `out`).
2. Rounded run: fn in float64 with every op result rounded to the working dtype (and the inputs of
   matmuls and convolutions rounded to TF32 when the device policy allows TF32). This is what a
   correct kernel that rounds once per op computes; the backward check uses the same run.
3. Error model runs: fn in float64 where every op result gets a random error of the size a correct
   kernel can make: about one unit roundoff of the result for arithmetic, four for transcendental
   functions (fast GPU math), and for an op that reduces (matmul, convolution, sum, mean, softmax,
   norms) sqrt(K) unit roundoffs of the accumulation precision times the sum of the absolute terms
   (computed by the op on |inputs|). Errors propagate through the rest of fn as they would in a real
   kernel. The spread over these runs gives a standard deviation per output element.

An element passes when |out - ref| <= LAMBDA * sigma + |rounded - ref| + 1 ulp (+ a tiny absolute
floor), and the whole output passes when its RMS error is at most LAMBDA_RMS times the RMS of sigma.
LAMBDA is a floor: sigma is estimated from SAMPLES runs, so an element's error over it is Student t
with SAMPLES degrees of freedom, and the multiple is that distribution's quantile at FAMILY_ALPHA /
elements (element_lambda: 8 for small outputs, larger for outputs of many elements). So the rounded run itself always passes, a near tie that the rounded run flips is allowed, and a
broad accuracy loss (float16 accumulation, TF32 where IEEE float32 was required) fails even when
each element alone is small. The model is conservative (unit roundoff as the standard deviation of
a rounding error whose true deviation is u / sqrt(3)), so LAMBDA = 8 sits far out in the tail.

Failures say which elements, by how much (also in ulps), whether the error looks like reordering
noise, where in the tensor the bad elements sit (a tail, one index, a diagonal), and which policy
would explain them (float16 accumulation, TF32).
"""

from __future__ import annotations

import contextlib
import copy
import math
import time
import types
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .capture import flatten_tensors, map_tensors

LAMBDA = 8.0  # per element: allowed multiples of the error model's standard deviation (at least)
FAMILY_ALPHA = 1e-3  # per output: chance that a correct output has any element over its bound
LAMBDA_RMS = 2.0  # whole output: allowed RMS error over the RMS of the standard deviation
SAMPLES = 8  # error model runs per derived tolerance
ABS_FLOOR = 1e-3  # times unit roundoff times the RMS of the reference: floor for exact zeros

UNIT_ROUNDOFF = {torch.float64: 2.0 ** -53, torch.float32: 2.0 ** -24,
                 torch.float16: 2.0 ** -11, torch.bfloat16: 2.0 ** -8}
MANTISSA = {torch.float64: 53, torch.float32: 24, torch.float16: 11, torch.bfloat16: 8}
MIN_EXP = {torch.float64: -1021, torch.float32: -125, torch.float16: -13, torch.bfloat16: -125}
TF32_U = 2.0 ** -11
RPR_FACTOR = 2.0  # split K partial sums rounded to the working dtype (see _matmul_sigma)

def _t_two_sided_tail(x: float, nu: int) -> float:
    """P(|T| > x) for Student's t with nu degrees of freedom, by the closed forms in theta =
    atan(x / sqrt(nu)) (Abramowitz and Stegun 26.7.3 and 26.7.4)."""
    th = math.atan(x / math.sqrt(nu))
    s, c = math.sin(th), math.cos(th)
    if nu % 2 == 0:
        term, acc = 1.0, 1.0
        for k in range(1, nu // 2):
            term *= (2 * k - 1) / (2 * k) * c * c
            acc += term
        inside = s * acc
    elif nu == 1:
        inside = 2.0 * th / math.pi
    else:
        term, acc = c, c
        for k in range(1, (nu - 1) // 2):
            term *= (2 * k) / (2 * k + 1) * c * c
            acc += term
        inside = 2.0 / math.pi * (th + s * acc)
    return max(0.0, 1.0 - inside)


def element_lambda(samples: int, elements: int, alpha: float = FAMILY_ALPHA) -> float:
    """Multiples of the estimated sigma an element may be off.

    sigma is estimated from ``samples`` zero mean error model runs, so sigma_hat^2 / sigma^2 is
    chi^2(samples) / samples, and a correct element's error over sigma_hat is Student t with ``samples``
    degrees of freedom, not a normal deviate. With 8 samples P(|t| > 8) is 4.4e-5: an output of a
    million correct elements would have about 40 over 8 sigma_hat (the fair benchmark's encoder
    layer: 2 of 786432 at 10 sigma_hat, with an independent float32 evaluation at the same places).
    The bound is the two sided t quantile at alpha / elements (Bonferroni: a correct output fails
    with probability at most alpha), and never below LAMBDA."""
    p = alpha / max(1, int(elements))
    if _t_two_sided_tail(LAMBDA, samples) <= p:
        return LAMBDA
    lo, hi = LAMBDA, LAMBDA
    while _t_two_sided_tail(hi, samples) > p:
        lo, hi = hi, hi * 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _t_two_sided_tail(mid, samples) > p:
            lo = mid
        else:
            hi = mid
    return hi


_PRECISION_ORDER = [torch.float64, torch.float32, torch.bfloat16, torch.float16]


class ToleranceError(AssertionError):
    """`truestar.assert_close` failed: the output is further off than a correct implementation of
    fn can be on these inputs. `report` holds the full CloseReport."""

    def __init__(self, message: str, report: "CloseReport"):
        super().__init__(message)
        self.report = report


# ----------------------------------------------------------------------------- policy


@dataclass(frozen=True)
class Policy:
    """How a correct implementation computes: working dtype, accumulation dtype inside one op's
    reduction, TF32 for matmuls and convolutions, and cuBLAS reduced precision split K."""

    dtype: torch.dtype
    accumulate: torch.dtype
    tf32_mm: bool = False
    tf32_conv: bool = False
    reduced_precision_reduction: bool = False

    @property
    def u(self) -> float:
        return UNIT_ROUNDOFF[self.dtype]

    @property
    def u_acc(self) -> float:
        return UNIT_ROUNDOFF[self.accumulate]

    def describe(self) -> str:
        d = _name(self.dtype)
        parts = [f"{d} results", f"{_name(self.accumulate)} accumulation"]
        if self.tf32_mm or self.tf32_conv:
            which = " and ".join(w for w, on in (("matmuls", self.tf32_mm),
                                                 ("convolutions", self.tf32_conv)) if on)
            parts.append(f"TF32 {which}")
        if self.reduced_precision_reduction:
            parts.append("reduced precision split K")
        return ", ".join(parts)

    def as_dict(self) -> Dict[str, Any]:
        return {"dtype": _name(self.dtype), "accumulate": _name(self.accumulate),
                "tf32_matmul": self.tf32_mm, "tf32_conv": self.tf32_conv,
                "reduced_precision_reduction": self.reduced_precision_reduction}


def _name(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def _dtype(x) -> Optional[torch.dtype]:
    if x is None or isinstance(x, torch.dtype):
        return x
    return getattr(torch, str(x).replace("torch.", ""))


def policy_for(dtype: torch.dtype, device: Union[str, torch.device] = "cpu",
               tf32: Optional[bool] = None, accumulate=None) -> Policy:
    """The device policy for `dtype` on `device`: half types accumulate in float32; TF32 follows
    torch's flags on CUDA (`torch.backends.cuda.matmul.allow_tf32`, the float32 matmul precision,
    `torch.backends.cudnn.allow_tf32`) unless `tf32` says otherwise; never on the CPU."""
    dtype = _dtype(dtype)
    dev = torch.device(device).type
    acc = _dtype(accumulate) or (torch.float64 if dtype == torch.float64 else torch.float32)
    cuda = dev == "cuda"
    if dtype != torch.float32:
        tf_mm = tf_conv = False
    elif tf32 is not None:
        tf_mm = tf_conv = bool(tf32)
    else:
        tf_mm = cuda and (torch.backends.cuda.matmul.allow_tf32
                          or torch.get_float32_matmul_precision() != "highest")
        tf_conv = cuda and bool(torch.backends.cudnn.allow_tf32)
    rpr = False
    if cuda and dtype == torch.float16:
        rpr = bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    elif cuda and dtype == torch.bfloat16:
        rpr = bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    return Policy(dtype, acc, tf_mm, tf_conv, rpr)


def lowest_precision(dtypes) -> torch.dtype:
    fl = [d for d in dtypes if d in UNIT_ROUNDOFF]
    if not fl:
        return torch.float32
    return max(fl, key=lambda d: UNIT_ROUNDOFF[d])


# ----------------------------------------------------------------------------- error model

_EXACT = {
    # data movement, selection and sign changes: a correct kernel computes these exactly
    "aten.view", "aten._unsafe_view", "aten.reshape", "aten.permute", "aten.transpose", "aten.t",
    "aten.expand", "aten.squeeze", "aten.unsqueeze", "aten.slice", "aten.select", "aten.as_strided",
    "aten.clone", "aten.contiguous", "aten._to_copy", "aten.to", "aten.copy", "aten.copy_",
    "aten.cat", "aten.stack", "aten.split", "aten.split_with_sizes", "aten.chunk", "aten.unbind",
    "aten.index", "aten.index_select", "aten.gather", "aten.take", "aten.where", "aten.masked_fill",
    "aten.masked_fill_", "aten.neg", "aten.abs", "aten.relu", "aten.relu_", "aten.threshold",
    "aten.threshold_", "aten.maximum", "aten.minimum", "aten.max", "aten.min", "aten.amax",
    "aten.amin", "aten.clamp", "aten.clamp_", "aten.clamp_min", "aten.clamp_max", "aten.sort",
    "aten.topk", "aten.flip", "aten.roll", "aten.repeat", "aten.constant_pad_nd", "aten.zeros",
    "aten.zeros_like", "aten.ones", "aten.ones_like", "aten.full", "aten.full_like",
    "aten.empty", "aten.empty_like", "aten.fill", "aten.fill_", "aten.zero_", "aten.new_zeros",
    "aten.new_empty", "aten.new_full", "aten.new_ones", "aten.scalar_tensor", "aten.lift_fresh",
    "aten.detach", "aten.alias", "aten.unfold", "aten.max_pool2d_with_indices", "aten.tril",
    "aten.triu", "aten.sign", "aten.floor", "aten.ceil", "aten.round", "aten.trunc",
    "aten.hardtanh", "aten.hardtanh_", "aten.index_put", "aten.index_put_", "aten.scatter",
    "aten.diagonal", "aten.narrow", "aten.movedim", "aten.flatten", "aten.unflatten",
    "aten.view_as_real", "aten.arange", "aten.eye", "aten.cummax", "aten.cummin",
}
# Correctly rounded arithmetic: one rounding of the result.
_ARITH = {"aten.add", "aten.add_", "aten.sub", "aten.sub_", "aten.rsub", "aten.mul", "aten.mul_",
          "aten.div", "aten.div_", "aten.reciprocal", "aten.square", "aten.lerp", "aten.addcmul",
          "aten.addcdiv", "aten.sqrt", "aten.leaky_relu", "aten.hardsigmoid", "aten.hardswish",
          "aten.true_divide", "aten.fmod", "aten.remainder", "aten._native_batch_norm_legit_no_training",
          "aten.native_batch_norm", "aten.batch_norm", "aten.dropout", "aten.native_dropout"}
_CORRECTLY_ROUNDED = {"aten.add", "aten.add_", "aten.sub", "aten.sub_", "aten.rsub", "aten.mul",
                      "aten.mul_", "aten.div", "aten.div_", "aten.true_divide", "aten.sqrt",
                      "aten.square"}
TRANSCENDENTAL_ULPS = 4.0  # fast GPU exp, log, tanh, erf and rsqrt are a few ulps off

_MATMUL = {"aten.mm", "aten.bmm", "aten.addmm", "aten.baddbmm", "aten.addbmm", "aten.matmul",
           "aten.dot", "aten.mv", "aten.addmv", "aten.linear"}
_CONV = {"aten.convolution", "aten._convolution", "aten.conv2d", "aten.conv1d", "aten.conv3d"}
_SUM = {"aten.sum", "aten.mean", "aten.nansum", "aten.cumsum", "aten.logsumexp"}
_SOFTMAX = {"aten._softmax", "aten._log_softmax", "aten.softmax", "aten.log_softmax",
            "aten._safe_softmax"}
_NORM = {"aten.native_layer_norm", "aten.native_group_norm", "aten._fused_rms_norm"}
# Batch norm (also instance norm, which runs it on an (1, N C, ...) view): batch statistics in
# training mode, an affine map of the running statistics otherwise.
_BATCH_NORM = {"aten.native_batch_norm", "aten._native_batch_norm_legit",
               "aten._native_batch_norm_legit_functional", "aten.cudnn_batch_norm",
               "aten.miopen_batch_norm", "aten._native_batch_norm_legit_no_training"}
_VAR = {"aten.var", "aten.std", "aten.var_mean", "aten.std_mean", "aten.norm",
        "aten.linalg_vector_norm"}


def _reduced_count(x: torch.Tensor, dims, keepdim_unused=None) -> int:
    if dims is None or (isinstance(dims, (list, tuple)) and len(dims) == 0):
        return max(1, x.numel())
    if isinstance(dims, int):
        dims = [dims]
    n = 1
    for d in dims:
        n *= x.shape[d] if x.dim() else 1
    return max(1, n)


def _tf32_sq(t: torch.Tensor) -> torch.Tensor:
    return t * t


CHUNK = 32  # partial sums are checkpointed every CHUNK terms of a reduction


def _independent(n_nz: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor,
                 smax: torch.Tensor, u_acc: float) -> torch.Tensor:
    """The cheap test of the probabilistic tier's assumption, per output element: the rounding
    errors of a running sum are independent with mean zero only when the added terms vary below
    one ulp of the running sum, so that each step's discarded bits are spread over the ulp. Here:
    at least two nonzero terms whose standard deviation (from the count n_nz, the sum s1 and the
    sum of squares s2 of the nonzero terms) is at least one ulp, 2 u_acc, of the largest running
    sum smax. Repeated values (a constant, a masked constant, a one hot row) fail it."""
    n = n_nz.clamp_min(1.0)
    mean = s1 / n
    var = (s2 / n - mean * mean).clamp_min(0.0)
    return (n_nz >= 2) & (torch.sqrt(var) >= 2.0 * u_acc * smax)


def _partial_sum_sigma(x: torch.Tensor, y: torch.Tensor, u_acc: float) -> torch.Tensor:
    """Accumulation error of sum_k x[..., k] y[..., k, :], in two tiers.

    Rigorous (running error analysis, Wilkinson; Higham, Accuracy and Stability, sec. 3.3):
    rounding the running sum s_k at each of the K steps adds at most u_acc |s_k|, so the error is
    at most u_acc sum_k |s_k|, whatever the order of a sequential or tiled sum and with no
    assumption on the data. The running sum is evaluated every CHUNK terms (s_b), and inside a
    chunk |s_k| is at most |s_b| plus that chunk's absolute sum (a_b), so
    R = u_acc sum_b CHUNK (|s_b| + a_b) bounds it.

    Probabilistic (Higham and Mary 2019): when the step errors are independent with mean zero,
    the error has standard deviation about u_acc sqrt(sum_k s_k^2), bounded the same way by
    u_acc sqrt(CHUNK sum_b (|s_b| + a_b)^2): about sqrt(K) times tighter than R on random data.

    The probabilistic tier is used where `_independent` says its assumption holds; elsewhere R
    (a constant tensor, where every step rounds the same way, needs it: held out sums, means and
    cumsums of a constant failed the probabilistic bound by 1.2x to 6x)."""
    k = x.shape[-1]
    s = None
    acc = lin = smax = None
    for k0 in range(0, k, CHUNK):
        xb, yb = x[..., k0:k0 + CHUNK], y[..., k0:k0 + CHUNK, :]
        p = xb @ yb
        a = xb.abs() @ yb.abs()
        s = p if s is None else s + p
        m = s.abs() + a
        acc = m.pow(2) * xb.shape[-1] if acc is None else acc + m.pow(2) * xb.shape[-1]
        lin = m * xb.shape[-1] if lin is None else lin + m * xb.shape[-1]
        smax = s.abs() if smax is None else torch.maximum(smax, s.abs())
    nz = (x != 0).to(x.dtype) @ (y != 0).to(y.dtype)
    ok = _independent(nz, s, (x * x) @ (y * y), smax, u_acc)
    return torch.where(ok, u_acc * torch.sqrt(acc), u_acc * lin)


def _matmul_sigma(func, packet: str, args, kwargs, out: torch.Tensor, pol: Policy,
                  tf32: bool) -> torch.Tensor:
    """Accumulation error (see _partial_sum_sigma), TF32 operand rounding when the policy allows
    it (u_tf32 sqrt(sum_k (x_k y_k)^2)) and the rounding of an added bias term."""
    a = list(args)
    alpha, extra = 1.0, None
    if packet in ("aten.mm", "aten.bmm", "aten.matmul", "aten.dot", "aten.mv"):
        x, y = a[0], a[1]
    else:  # addmm(c, a, b, beta, alpha) and friends
        c, x, y = a[0], a[1], a[2]
        beta = kwargs.get("beta", a[3] if len(a) > 3 else 1)
        alpha = kwargs.get("alpha", a[4] if len(a) > 4 else 1)
        extra = abs(beta) * c.abs()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    yy = y.unsqueeze(-1) if y.dim() == 1 else y
    sig = _partial_sum_sigma(x, yy, pol.u_acc)
    if packet == "aten.addbmm":
        sig = torch.sqrt(sig.pow(2).sum(0))
    if tf32 or pol.reduced_precision_reduction:
        sq = torch.sqrt((x * x) @ (yy * yy))
        if packet == "aten.addbmm":
            sq = torch.sqrt(sq.pow(2).sum(0))
        if tf32:
            sig = sig + TF32_U * sq
        if pol.reduced_precision_reduction:
            # cuBLAS split K (allowed by allow_fp16/bf16_reduced_precision_reduction) rounds each
            # partial sum to the working dtype; a partial over K/s terms is about
            # sqrt(sum_k (x_k y_k)^2 / s), and s of them are rounded and added: about
            # u sqrt(sum_k (x_k y_k)^2) in all, large next to an output that cancels to near 0.
            sig = sig + RPR_FACTOR * pol.u * sq
    sig = abs(alpha) * sig
    if sig.numel() == out.numel() and tuple(sig.shape) != tuple(out.shape):
        sig = sig.reshape(out.shape)
    if extra is not None:
        sig = sig + pol.u * extra
    return sig


def _conv_sigma(args, kwargs, out: torch.Tensor, pol: Policy, tf32: bool) -> torch.Tensor:
    x, w = args[0], args[1]
    b = args[2] if len(args) > 2 else None
    rest = list(args[3:])
    absout = torch.ops.aten.convolution(x.abs(), w.abs(), None if b is None else b.abs(), *rest)
    k = w[0].numel()
    sig = math.sqrt(k) * pol.u_acc * absout
    if tf32:
        sig = sig + TF32_U * torch.sqrt(torch.ops.aten.convolution(x * x, w * w, None, *rest))
    return sig


def _sum_sigma(func, packet: str, args, kwargs, out: torch.Tensor, pol: Policy) -> torch.Tensor:
    x = args[0]
    if packet == "aten.cumsum":
        dim = args[1] if len(args) > 1 else kwargs["dim"]
        sig = math.sqrt(x.shape[dim]) * pol.u_acc * torch.cumsum(x.abs(), dim)
        if x.dim() > 0 and x.shape[dim] > 1:
            # The two tiers of _partial_sum_sigma along the scan: the running bound at position
            # i is u_acc sum_{j <= i} |s_j|; the assumption test is per scanned row.
            run = torch.cumsum(x, dim)
            nz = (x != 0).to(x.dtype).sum(dim, keepdim=True)
            ok = _independent(nz, x.sum(dim, keepdim=True), (x * x).sum(dim, keepdim=True),
                              run.abs().amax(dim, keepdim=True), pol.u_acc)
            sig = torch.where(ok, sig, pol.u_acc * torch.cumsum(run.abs(), dim))
        return sig
    dims = args[1] if len(args) > 1 else kwargs.get("dim")
    if packet == "aten.logsumexp":
        k = _reduced_count(x, dims)
        return (TRANSCENDENTAL_ULPS * pol.u + math.sqrt(k) * pol.u_acc) * (out.abs() + 1.0)
    if packet in ("aten.sum", "aten.mean") and x.dim() > 0:
        # Running sums over the reduced dims, as for a matmul (_partial_sum_sigma).
        if dims is None or (isinstance(dims, (list, tuple)) and len(dims) == 0):
            rd = list(range(x.dim()))
        else:
            rd = sorted({d % x.dim() for d in ([dims] if isinstance(dims, int) else dims)})
        keep = [d for d in range(x.dim()) if d not in rd]
        xt = x.permute(keep + rd).reshape([x.shape[d] for d in keep] + [1, -1])
        ones = torch.ones(xt.shape[-1], 1, dtype=x.dtype, device=x.device)
        sig = _partial_sum_sigma(xt, ones, pol.u_acc)
        if packet == "aten.mean":
            sig = sig / max(1, xt.shape[-1])
        return sig.reshape(out.shape) if sig.numel() == out.numel() else sig.reshape(-1)[0]
    try:
        absout = func(x.abs(), *args[1:], **kwargs)
    except Exception:
        absout = out.abs()
    k = _reduced_count(x, dims)
    return math.sqrt(k) * pol.u_acc * absout


def _norm_sigma(packet: str, args, out: torch.Tensor, pol: Policy) -> torch.Tensor:
    """Layer and group norm: the mean and variance carry sqrt(K) u_acc relative error each, and the
    mean's error is |mean| / std times larger on the output (two pass, or Welford)."""
    x = args[0]
    if packet == "aten.native_layer_norm":
        nshape = list(args[1])
        dims = list(range(x.dim() - len(nshape), x.dim()))
        w = args[2] if len(args) > 2 else None
    elif packet == "aten.native_group_norm":
        n, c = x.shape[0], x.shape[1]
        g = args[6] if len(args) > 6 else 1
        xs = x.reshape(n, g, -1)
        mu = xs.mean(-1, keepdim=True).abs()
        sd = xs.std(-1, keepdim=True, correction=0).clamp_min(1e-300)
        k = xs.shape[-1]
        rel = ((mu / sd) + 1.0).expand_as(xs).reshape(x.shape)
        w = args[1]
        sig = math.sqrt(k) * pol.u_acc * rel
        if isinstance(w, torch.Tensor):
            sig = sig * w.abs().reshape([1, -1] + [1] * (x.dim() - 2))
        return sig
    else:  # _fused_rms_norm(input, normalized_shape, weight, eps)
        nshape = list(args[1])
        dims = list(range(x.dim() - len(nshape), x.dim()))
        w = args[2] if len(args) > 2 else None
        k = 1
        for d in dims:
            k *= x.shape[d]
        sig = math.sqrt(k) * pol.u_acc * out.abs()
        return sig
    k = 1
    for d in dims:
        k *= x.shape[d]
    mu = x.mean(dims, keepdim=True).abs()
    sd = x.std(dims, keepdim=True, correction=0).clamp_min(1e-300)
    sig = math.sqrt(k) * pol.u_acc * ((mu / sd) + 1.0).expand_as(x)
    if isinstance(w, torch.Tensor):
        sig = sig * w.abs()
    return sig


def _batch_norm_sigma(packet: str, args, kwargs, out: torch.Tensor, pol: Policy) -> torch.Tensor:
    """Training mode: the batch mean and variance over every dim but 1 carry the layer norm error
    (_norm_sigma). Otherwise an affine map of the running statistics: the subtraction of the
    running mean is the step that loses precision, relative to |x| + |running mean|."""
    a = list(args)
    x, w = a[0], (a[1] if len(a) > 1 else None)
    if packet == "aten._native_batch_norm_legit_no_training":
        training = False
    elif packet == "aten._native_batch_norm_legit" and len(a) == 6:
        training = bool(a[3])  # (input, weight, bias, training, momentum, eps): no running stats
    else:
        training = bool(a[5]) if len(a) > 5 else bool(kwargs.get("training", True))
    shape = [1, -1] + [1] * (x.dim() - 2)
    wa = w.abs().reshape(shape) if isinstance(w, torch.Tensor) else 1.0
    if training or len(a) < 5 or not isinstance(a[3], torch.Tensor):
        dims = [d for d in range(x.dim()) if d != 1]
        k = 1
        for d in dims:
            k *= x.shape[d]
        mu = x.mean(dims, keepdim=True).abs()
        sd = x.std(dims, keepdim=True, correction=0).clamp_min(1e-300)
        return math.sqrt(k) * pol.u_acc * ((mu / sd) + 1.0).expand_as(x) * wa
    rm, rv = a[3].reshape(shape), a[4].reshape(shape)
    eps = a[-1] if isinstance(a[-1], float) else 1e-5
    return pol.u * (x.abs() + rm.abs()) / torch.sqrt(rv + eps) * wa


def _op_sigma(func, args, kwargs, out: torch.Tensor, pol: Policy,
              unknown: Dict[str, int]) -> Optional[torch.Tensor]:
    """Standard deviation of a correct kernel's error on this op's first output, or None when the
    op is exact. `out` is the exact float64 result given these (already perturbed) inputs."""
    packet = str(func.overloadpacket)
    if packet in _EXACT:
        return None
    local = pol.u * out.abs()
    if packet in _MATMUL:
        s = _matmul_sigma(func, packet, args, kwargs, out, pol, pol.tf32_mm)
        return s + local
    if packet in _CONV:
        return _conv_sigma(args, kwargs, out, pol, pol.tf32_conv) + local
    if packet in _SUM:
        return _sum_sigma(func, packet, args, kwargs, out, pol) + local
    if packet in _SOFTMAX:
        x = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", -1)
        k = x.shape[dim] if x.dim() else 1
        probs = packet in ("aten._softmax", "aten.softmax", "aten._safe_softmax")
        base = out.abs() if probs else (out.abs() + 1.0)
        return (TRANSCENDENTAL_ULPS * pol.u + math.sqrt(k) * pol.u_acc) * base
    if packet in _NORM:
        return _norm_sigma(packet, args, out, pol) + TRANSCENDENTAL_ULPS * local
    if packet in _BATCH_NORM:
        return _batch_norm_sigma(packet, args, kwargs, out, pol) + TRANSCENDENTAL_ULPS * local
    if packet in _VAR:
        x = args[0]
        k = max(1, x.numel() // max(1, out.numel()))
        return (math.sqrt(k) * pol.u_acc + TRANSCENDENTAL_ULPS * pol.u) * out.abs()
    if packet in _ARITH:
        if packet in _CORRECTLY_ROUNDED and pol.dtype != torch.float64:
            # IEEE add, sub, mul, div and sqrt are correctly rounded: where the exact result is
            # representable in the working dtype a correct kernel returns it exactly. Constant
            # arithmetic (an exponent 2 - 1 in a pow gradient) stays exact; perturbing it gave
            # pow(negative x, 1 + noise) = NaN and wiped out the whole error model run.
            exact = out == out.to(pol.dtype).to(out.dtype)
            return torch.where(exact, torch.zeros_like(local), local)
        return local
    if packet in _GATED:
        # x * g(x) activations: common formulas (0.5 x (1 + erf(x / sqrt 2)), x / (1 + exp(-x)))
        # cancel where the result is tiny next to x, so a correct kernel is a few u |x| off there
        # (PyTorch float32 GELU on the CPU: 3.7 u |x| measured at x = -3.3).
        x = args[0]
        return TRANSCENDENTAL_ULPS * (local + pol.u * x.abs())
    if packet in _TRANSCENDENTAL:
        return TRANSCENDENTAL_ULPS * local
    unknown[packet] = unknown.get(packet, 0) + 1
    if out.numel() < max((t.numel() for t in _floats(args)), default=0):
        k = max((t.numel() for t in _floats(args)), default=1) // max(1, out.numel())
        return (TRANSCENDENTAL_ULPS * pol.u + math.sqrt(k) * pol.u_acc) * out.abs()
    return TRANSCENDENTAL_ULPS * local


_GATED = {"aten.gelu", "aten.silu", "aten.mish", "aten.softplus", "aten.elu", "aten.selu",
          "aten.celu", "aten.hardswish"}
_TRANSCENDENTAL = {
    "aten.exp", "aten.exp2", "aten.expm1", "aten.log", "aten.log2", "aten.log10", "aten.log1p",
    "aten.tanh", "aten.sigmoid", "aten.erf", "aten.erfc", "aten.rsqrt", "aten.pow", "aten.sin",
    "aten.cos", "aten.tan", "aten.atan", "aten.atan2", "aten.gelu", "aten.silu", "aten.mish",
    "aten.softplus", "aten.elu", "aten.selu", "aten.celu", "aten.logit", "aten.asinh",
    "aten.acosh", "aten.atanh", "aten.sinh", "aten.cosh", "aten.xlogy", "aten.special_erf",
}


def _floats(obj) -> List[torch.Tensor]:
    return [t for t in flatten_tensors(obj) if t.is_floating_point()]


def _leaves(out) -> List[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        return [t for x in out for t in _leaves(x)]
    return []


def _error_mode(pol: Policy, gen: Dict[str, torch.Generator], seed: int,
                unknown: Dict[str, int], memo: Optional[list] = None):
    """Float64 math where each op result gets a random N(0, sigma) error, sigma from _op_sigma.

    `memo` holds the sigma of each op call from the first run, in call order; later runs reuse it
    (their inputs differ only by rounding size noise), so the error model costs one sigma pass."""
    calls = [0]

    def sigma(func, args, kwargs, t, first: bool):
        i = calls[0]
        calls[0] += 1
        if memo is not None and i < len(memo):
            name, shape, val = memo[i]
            if name == str(func) and shape == tuple(t.shape):
                return val
        if first:
            val = _op_sigma(func, args, kwargs, t, pol, unknown)
        else:
            val = TRANSCENDENTAL_ULPS * pol.u * t.abs()
        if memo is not None and i == len(memo):
            memo.append((str(func), tuple(t.shape), val))
        return val
    from torch.utils._python_dispatch import TorchDispatchMode

    def noise_like(t: torch.Tensor) -> torch.Tensor:
        key = str(t.device)
        if key not in gen:
            gen[key] = torch.Generator(device=t.device).manual_seed(seed)
        return torch.randn(t.shape, generator=gen[key], dtype=t.dtype, device=t.device)

    def perturb(t: torch.Tensor, sig) -> torch.Tensor:
        if sig is None:
            return t
        sig = torch.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        return t + sig.expand_as(t) * noise_like(t) if sig.shape != t.shape else t + sig * noise_like(t)

    class ErrorModelMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            schema = func._schema
            leaves = _leaves(out)
            if not any(t.dtype == torch.float64 for t in leaves):
                return out
            if any(r.alias_info is not None for r in schema.returns):
                if schema.is_mutable:  # in place: perturb the written tensor in place
                    for t in leaves:
                        if t.dtype == torch.float64:
                            sig = sigma(func, args, kwargs, t, True)
                            if sig is not None:
                                with torch.no_grad():
                                    t.copy_(perturb(t, sig))
                return out
            if isinstance(out, torch.Tensor):
                return perturb(out, sigma(func, args, kwargs, out, True))
            if isinstance(out, (tuple, list)):
                res = []
                for i, t in enumerate(out):
                    if isinstance(t, torch.Tensor) and t.dtype == torch.float64:
                        sig = None if str(func.overloadpacket) in _EXACT else \
                            sigma(func, args, kwargs, t, i == 0)
                        t = perturb(t, sig)
                    res.append(t)
                return type(out)(res)
            return out

    return ErrorModelMode()


def _math_sdpa():
    """Run scaled_dot_product_attention as its math decomposition, so the error model sees the
    matmuls and the softmax inside it."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.MATH])
    except Exception:
        return contextlib.nullcontext()


# ----------------------------------------------------------------------------- derive


@dataclass
class Tolerance:
    """A derived tolerance for fn on one set of inputs; check any number of candidates with it."""

    policy: Policy
    device: torch.device
    reference: List[torch.Tensor]  # float64
    rounded: List[torch.Tensor]  # float64, the correctly rounded run
    sigma: List[torch.Tensor]  # float64 standard deviation of a correct kernel's error
    samples: int
    seconds: float
    unknown_ops: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def bound(self, i: int = 0) -> torch.Tensor:
        """Per element allowed |out - reference| for output i (float64, on the device)."""
        ref, rnd, sig = self.reference[i], self.rounded[i], self.sigma[i]
        out_dt = self.policy.dtype
        lam = element_lambda(self.samples, int(torch.isfinite(ref).sum()) if ref.numel() else 0)
        return (lam * sig + _nan0((rnd - ref).abs()) + _ulp(ref, out_dt)
                + _abs_floor(ref, self.policy))

    def check(self, out, explain: bool = True, fn=None, inputs=None, kwargs=None) -> "CloseReport":
        return _check(self, out, explain, fn, inputs, kwargs)


def _nan0(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def _ulp(ref: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """One unit in the last place of `dt` at |ref| (subnormal spacing at the bottom)."""
    p, emin = MANTISSA.get(dt, 24), MIN_EXP.get(dt, -125)
    a = _nan0(ref.abs())
    _, e = torch.frexp(a)
    e = torch.clamp(e.to(torch.float64), min=float(emin))
    return torch.pow(torch.tensor(2.0, dtype=torch.float64, device=ref.device), e - p)


def _abs_floor(ref: torch.Tensor, pol: Policy) -> torch.Tensor:
    fin = ref[torch.isfinite(ref)]
    rms = float(fin.pow(2).mean().sqrt()) if fin.numel() else 0.0
    return torch.tensor(ABS_FLOOR * pol.u * rms, dtype=torch.float64, device=ref.device)


# key -> (token, weak ref to fn, weak refs to the input tensors, tolerance). The cache never keeps
# a caller's function, module or tensors alive: an entry can only be hit with those very objects, so
# once one of them is gone the entry is dropped at once (weakref callback), with its float64
# reference tensors. Holding them strongly pinned up to 16 inputs and their references on the
# device for the life of the process, which code that lists live CUDA tensors then saw.
_CACHE: "OrderedDict[Tuple, Tuple[object, Callable[[], Any], Tuple[weakref.ref, ...], Tolerance]]" = OrderedDict()
_CACHE_SIZE = 16


def _weak(obj, callback) -> Callable[[], Any]:
    """A weak reference to obj (a bound method by its object and function), or a strong stand in
    when obj cannot be weakly referenced."""
    if isinstance(obj, types.MethodType):
        return weakref.WeakMethod(obj, callback)
    try:
        return weakref.ref(obj, callback)
    except TypeError:
        return lambda: obj


def _evicter(key, token) -> Callable[[Any], None]:
    # holds the key and a plain token only: no path back to the entry, so no reference cycle
    def evict(_ref) -> None:
        entry = _CACHE.get(key)
        if entry is not None and entry[0] is token:
            del _CACHE[key]
    return evict


def _cache_put(key, fn, objs, tol: Tolerance) -> None:
    token = object()
    evict = _evicter(key, token)
    refs = tuple(weakref.ref(t, evict) for t in flatten_tensors(objs))
    _CACHE[key] = (token, _weak(fn, evict), refs, tol)
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)


def _cache_get(key, fn, objs) -> Optional[Tolerance]:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    _, fn_ref, refs, tol = entry
    ts = flatten_tensors(objs)
    if fn_ref() is not fn or len(ts) != len(refs) or any(r() is not t for r, t in zip(refs, ts)):
        return None
    _CACHE.move_to_end(key)
    return tol


def _sig(x) -> Tuple:
    if isinstance(x, torch.Tensor):
        return ("T", x.data_ptr(), x._version, tuple(x.shape), tuple(x.stride()), str(x.dtype),
                str(x.device))
    if isinstance(x, (list, tuple)):
        return tuple(_sig(v) for v in x)
    if isinstance(x, dict):
        return tuple((k, _sig(v)) for k, v in sorted(x.items()))
    return ("V", repr(x))


def _as_tuple(inputs) -> tuple:
    if isinstance(inputs, torch.Tensor):
        return (inputs,)
    return tuple(inputs)


def _widen(obj, device):
    def f(t: torch.Tensor):
        if t.is_floating_point():
            return t.detach().to(device=device, dtype=torch.float64)
        return t.detach().to(device)
    return map_tensors(obj, f)


def _fn64(fn, device):
    if isinstance(fn, nn.Module):
        m = copy.deepcopy(fn).to(device)
        for t in list(m.parameters()) + list(m.buffers()):
            if t.is_floating_point():
                t.data = t.data.to(torch.float64)
        return m.eval() if not fn.training else m
    return fn


def _run(fn, args, kwargs) -> List[torch.Tensor]:
    with torch.no_grad(), _math_sdpa():
        return [t for t in flatten_tensors(fn(*args, **kwargs))]


def derive(fn: Callable, inputs, *, like=None, dtype=None, device=None, tf32: Optional[bool] = None,
           accumulate=None, samples: int = SAMPLES, seed: int = 0, kwargs=None,
           cache: bool = True, memo: bool = True) -> Tolerance:
    """Derive the per element tolerance of fn on `inputs`.

    Args:
        fn: a function of torch tensors (or an nn.Module) that computes the intended result. It
            is run in float64; it must not depend on the inputs' dtype for its meaning.
        inputs: a tensor or a sequence of positional arguments, in the candidate's precision.
        like: the candidate output (a tensor or a sequence); fixes dtype and device when given.
        dtype: working precision (default: the lowest floating precision among inputs and
            `like`). device: where the float64 runs happen (default: the device of `like`).
        tf32: force TF32 on or off for float32 matmuls and convolutions (default: torch's flags on
            CUDA, off on CPU). accumulate: accumulation dtype inside one reduction (default:
            float32 for half types). memo: keep every op's sigma from the first error model run
            for the others (faster; holds one sigma tensor per op call, so a large training step
            may not fit on its device without it).
    """
    t0 = time.perf_counter()
    args = _as_tuple(inputs)
    kwargs = dict(kwargs or {})
    likes = [t for t in flatten_tensors(like)] if like is not None else []
    if device is None:
        device = likes[0].device if likes else next(
            (t.device for t in flatten_tensors(args) if isinstance(t, torch.Tensor)), "cpu")
    device = torch.device(device)
    if dtype is None:
        dtype = lowest_precision([t.dtype for t in flatten_tensors((args, kwargs)) + likes
                                  if t.is_floating_point()])
    pol = policy_for(dtype, device, tf32, accumulate)
    key = None
    if cache:
        key = (id(fn), _sig(args), _sig(kwargs), pol, str(device), samples, seed)
        # a hit must be the very fn and input tensors the entry was derived for: a new tensor at a
        # freed one's address has the same signature (the entry was dropped when the old one died)
        hit = _cache_get(key, fn, (args, kwargs))
        if hit is not None:
            return hit
    a64, k64 = _widen((args, kwargs), device)
    f64 = _fn64(fn, device)
    ref = _run(f64, a64, k64)
    from .gradcheck import _rounding_mode

    if pol.dtype == torch.float64:
        rounded = [r.clone() for r in ref]
    else:
        ar, kr = map_tensors((a64, k64), lambda t: t.to(pol.dtype).to(torch.float64)
                             if t.is_floating_point() else t)
        with _rounding_mode(pol.dtype, pol.tf32_conv, pol.tf32_mm):
            rounded = _run(f64, ar, kr)
    unknown: Dict[str, int] = {}
    # sigma of a bool or integer output is 0: it has no rounding (an accumulator of its own dtype could
    # not hold the float squares: a bool output, such as a non finite flag, raised here)
    acc = [torch.zeros_like(r, dtype=torch.float64) if not r.is_floating_point() else torch.zeros_like(r)
           for r in ref]
    gen: Dict[str, torch.Generator] = {}
    memo_list: Optional[list] = [] if memo else None
    for s in range(samples):
        with _error_mode(pol, gen, seed * 1000003 + s, unknown, memo_list):
            outs = _run(f64, a64, k64)
        for i, (o, r) in enumerate(zip(outs, ref)):
            acc[i] += _nan0(o - r).pow(2) if o.is_floating_point() else 0
    sigma = [torch.sqrt(a / max(1, samples)) for a in acc]
    notes = []
    if unknown:
        notes.append("ops without an error rule (taken as 4 ulp elementwise or a plain "
                     "reduction): " + ", ".join(f"{k} x{v}" for k, v in sorted(unknown.items())))
    tol = Tolerance(pol, device, ref, rounded, sigma, samples, time.perf_counter() - t0,
                    {k: v for k, v in unknown.items()}, notes)
    if key is not None:
        _cache_put(key, fn, (args, kwargs), tol)
    return tol


def clear_cache() -> None:
    _CACHE.clear()


# ----------------------------------------------------------------------------- check


@dataclass
class OutputReport:
    index: int
    shape: Tuple[int, ...]
    ok: bool
    max_err: float = 0.0
    worst_index: Optional[Tuple[int, ...]] = None
    worst_value: Optional[float] = None
    worst_reference: Optional[float] = None
    worst_ratio: float = 0.0  # |out - ref| / bound at the worst element
    max_ulps: float = 0.0  # in ulps of the working dtype at |ref|
    rms_ratio: float = 0.0  # RMS error over RMS sigma
    bad: int = 0
    total: int = 0
    nonfinite_mismatch: int = 0
    shape_mismatch: bool = False
    unresolved: int = 0  # elements the working dtype cannot represent (rounded run overflows)
    location: str = ""
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        def num(x):
            return x if isinstance(x, (int, float)) and math.isfinite(x) else None
        return {"index": self.index, "shape": list(self.shape), "ok": self.ok,
                "max_err": num(self.max_err), "worst_index": list(self.worst_index)
                if self.worst_index is not None else None,
                "worst_value": num(self.worst_value), "worst_reference": num(self.worst_reference),
                "worst_ratio": num(self.worst_ratio), "max_ulps": num(self.max_ulps),
                "rms_ratio": num(self.rms_ratio), "bad": self.bad, "total": self.total,
                "nonfinite_mismatch": self.nonfinite_mismatch,
                "shape_mismatch": self.shape_mismatch, "unresolved": self.unresolved,
                "location": self.location, "detail": self.detail}


@dataclass
class CloseReport:
    ok: bool
    policy: Policy
    outputs: List[OutputReport]
    hint: str = ""
    consistent_with_noise: bool = True
    derive_seconds: float = 0.0
    check_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def worst(self) -> Optional[OutputReport]:
        bad = [o for o in self.outputs if not o.ok]
        pool = bad or self.outputs
        return max(pool, key=lambda o: (o.shape_mismatch, o.nonfinite_mismatch, o.worst_ratio)) \
            if pool else None

    def message(self) -> str:
        if self.ok:
            w = self.worst
            extra = "" if w is None else (f" (worst element {w.worst_ratio:.2f} of its bound, "
                                          f"{w.max_ulps:.1f} ulp, RMS {w.rms_ratio:.2f} of a "
                                          f"correct kernel's)")
            return f"close to float64 within the derived tolerance{extra}"
        parts = []
        for o in self.outputs:
            if o.ok:
                continue
            tag = f"output {o.index}" if len(self.outputs) > 1 else "output"
            if o.shape_mismatch:
                parts.append(f"{tag}: {o.detail}")
                continue
            if o.nonfinite_mismatch:
                parts.append(f"{tag}: NaN or infinity at {o.nonfinite_mismatch} places where the "
                             f"float64 reference has none, or the other way round")
            if o.bad and o.worst_value is None:  # integer or boolean output
                parts.append(f"{tag}: {o.detail}")
            elif o.bad:
                parts.append(
                    f"{tag}: {o.bad} of {o.total} elements beyond the derived bound; worst at "
                    f"{list(o.worst_index) if o.worst_index is not None else '?'}: "
                    f"{o.worst_value:.6g} vs float64 {o.worst_reference:.6g} "
                    f"(error {o.max_err:.3g}, {o.worst_ratio:.1f}x its bound, "
                    f"{o.max_ulps:.0f} ulp of {_name(self.policy.dtype)})")
            if o.rms_ratio > LAMBDA_RMS:
                parts.append(f"{tag}: overall error is {o.rms_ratio:.1f}x what a correct "
                             f"{_name(self.policy.dtype)} implementation shows (limit "
                             f"{LAMBDA_RMS:g}x)")
            if o.location:
                parts.append(f"{tag}: bad elements {o.location}")
        msg = "; ".join(parts)
        msg += (". Not reordering noise." if not self.consistent_with_noise else "")
        if self.hint:
            msg += f" {self.hint}"
        msg += f" Policy: {self.policy.describe()}."
        return msg

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "policy": self.policy.as_dict(),
                "outputs": [o.as_dict() for o in self.outputs], "hint": self.hint,
                "consistent_with_noise": self.consistent_with_noise,
                "derive_seconds": self.derive_seconds, "check_seconds": self.check_seconds,
                "notes": list(self.notes), "message": self.message()}


def _check(tol: Tolerance, out, explain: bool, fn, inputs, kwargs) -> CloseReport:
    t0 = time.perf_counter()
    outs = [t for t in flatten_tensors(out)]
    reports: List[OutputReport] = []
    if len(outs) != len(tol.reference):
        r = OutputReport(0, (), False, shape_mismatch=True,
                         detail=f"{len(outs)} output tensors vs {len(tol.reference)} from fn")
        return CloseReport(False, tol.policy, [r], derive_seconds=tol.seconds,
                           check_seconds=time.perf_counter() - t0, notes=list(tol.notes),
                           consistent_with_noise=False)
    for i, (o, ref) in enumerate(zip(outs, tol.reference)):
        reports.append(_check_one(i, o, ref, tol))
    ok = all(r.ok for r in reports)
    noise = all(r.rms_ratio <= 1.0 and not r.nonfinite_mismatch and not r.shape_mismatch
                for r in reports)
    rep = CloseReport(ok, tol.policy, reports, consistent_with_noise=noise,
                      derive_seconds=tol.seconds, notes=list(tol.notes))
    if not ok and explain and fn is not None and inputs is not None:
        rep.hint = _explain(fn, inputs, out, tol, kwargs)
    rep.check_seconds = time.perf_counter() - t0
    return rep


def _check_one(i: int, o: torch.Tensor, ref: torch.Tensor, tol: Tolerance) -> OutputReport:
    shape = tuple(ref.shape)
    if tuple(o.shape) != shape:
        return OutputReport(i, shape, False, shape_mismatch=True,
                            detail=f"shape {tuple(o.shape)} vs float64 reference {shape}")
    if not ref.is_floating_point():
        same = bool(torch.equal(o.detach().to(ref.device), ref.to(o.dtype).to(ref.device)))
        return OutputReport(i, shape, same, bad=0 if same else 1, total=ref.numel(),
                            detail="" if same else "integer or boolean values differ")
    x = o.detach().to(device=ref.device, dtype=torch.float64)
    rnd = tol.rounded[i]
    n = ref.numel()
    fin_ref, fin_x = torch.isfinite(ref), torch.isfinite(x)
    # Where the rounded run is not finite but float64 is, the working dtype itself overflows:
    # a correct kernel may or may not (it can keep a wider intermediate); accept both.
    unres = fin_ref & ~torch.isfinite(rnd)
    # Non finite pattern: where float64 is NaN or infinite, the output must be the same value.
    nf_bad = (~fin_ref) & ~(torch.isnan(ref) & torch.isnan(x)) & ~(torch.isinf(ref) & (x == ref))
    nf_bad |= fin_ref & ~fin_x & ~unres
    nonfinite = int(nf_bad.sum())
    use = fin_ref & fin_x & ~unres
    rep = OutputReport(i, shape, True, total=n, unresolved=int(unres.sum()),
                       nonfinite_mismatch=nonfinite)
    if bool(use.any()):
        err = torch.where(use, (x - ref).abs(), torch.zeros_like(ref))
        bnd = tol.bound(i)
        ratio = torch.where(use, err / bnd.clamp_min(1e-300), torch.zeros_like(ref))
        bad = ratio > 1.0
        rep.bad = int(bad.sum())
        flat = int(torch.argmax(ratio))
        idx = tuple(int(v) for v in torch.unravel_index(torch.tensor(flat), shape)) if shape else ()
        rep.worst_index = idx
        rep.worst_ratio = float(ratio.reshape(-1)[flat])
        rep.worst_value = float(x.reshape(-1)[flat])
        rep.worst_reference = float(ref.reshape(-1)[flat])
        rep.max_err = float(err.max())
        ulps = err / _ulp(ref, tol.policy.dtype)
        rep.max_ulps = float(torch.where(use, ulps, torch.zeros_like(ulps)).max())
        sig = tol.sigma[i]
        floor = (sig + _ulp(ref, tol.policy.dtype) / 2 + _nan0((tol.rounded[i] - ref).abs()))
        num = float(err[use].pow(2).mean().sqrt())
        den = float(floor[use].pow(2).mean().sqrt())
        rep.rms_ratio = num / den if den > 0 else (0.0 if num == 0 else math.inf)
        if rep.bad:
            rep.location = _locate(bad, shape)
    rep.ok = rep.bad == 0 and rep.nonfinite_mismatch == 0 and rep.rms_ratio <= LAMBDA_RMS
    if nonfinite and not rep.location:
        rep.location = _locate(nf_bad, shape)
    return rep


def _locate(bad: torch.Tensor, shape: Tuple[int, ...]) -> str:
    """Where the bad elements sit: a tail or head of one dim, a few indices, a diagonal."""
    if not shape or not bool(bad.any()):
        return ""
    idx = torch.nonzero(bad).cpu()
    n = idx.shape[0]
    total = bad.numel()
    found = []
    for d, size in enumerate(shape):
        if size <= 1:
            continue
        vals = torch.unique(idx[:, d])
        lo, hi = int(vals.min()), int(vals.max())
        if len(vals) == size:
            continue
        if hi == size - 1 and lo > 0 and len(vals) > 3:
            found.append(f"in the tail of dim {d} (indices {lo} to {size - 1} of {size})")
        elif lo == 0 and hi < size - 1 and len(vals) > 3:
            found.append(f"in the head of dim {d} (indices 0 to {hi} of {size})")
        elif len(vals) > 3 and hi - lo + 1 <= size // 2:
            found.append(f"only in indices {lo} to {hi} of dim {d} (size {size})")
        elif len(vals) <= 3:
            found.append(f"only at index {', '.join(str(int(v)) for v in vals)} of dim {d} "
                         f"(size {size})")
    if len(shape) >= 2 and shape[-1] > 1 and shape[-2] > 1:
        off = idx[:, -1] - idx[:, -2]
        offs = torch.unique(off)
        if len(offs) <= 2:
            o = ", ".join(str(int(v)) for v in offs)
            found.append(f"on diagonal offset {o} of the last two dims (a mask edge?)")
    frac = n / max(1, total)
    head = f"({n} of {total}, {frac:.1%})"
    if not found:
        return f"spread over the tensor {head}"
    return "; ".join(found) + f" {head}"


def _explain(fn, inputs, out, tol: Tolerance, kwargs) -> str:
    """Name a policy under which the output would pass: float16 accumulation, TF32, a wider
    working dtype. Only runs on failure."""
    pol = tol.policy
    tries = []
    if pol.dtype in (torch.float16, torch.bfloat16) and pol.accumulate != pol.dtype:
        tries.append((f"{_name(pol.dtype)} accumulation (the policy accumulates in "
                      f"{_name(pol.accumulate)})", dict(accumulate=pol.dtype)))
    if pol.dtype == torch.float32 and not (pol.tf32_mm or pol.tf32_conv):
        tries.append(("TF32 matmuls or convolutions (the policy says IEEE float32)",
                      dict(tf32=True)))
    for what, kw in tries:
        try:
            alt = derive(fn, inputs, like=out, dtype=pol.dtype, device=tol.device,
                         samples=tol.samples, kwargs=kwargs,
                         **{"tf32": kw.get("tf32"), "accumulate": kw.get("accumulate")})
            r = alt.check(out, explain=False)
            if r.ok:
                return f"The error is consistent with {what}."
        except Exception:
            continue
    return ""


# ----------------------------------------------------------------------------- public API


def close_report(fn: Callable, inputs, out, *, dtype=None, tf32: Optional[bool] = None,
                 accumulate=None, samples: int = SAMPLES, seed: int = 0, kwargs=None,
                 explain: bool = True, memo: bool = True) -> CloseReport:
    """Compare `out` with fn(*inputs) under a derived tolerance; never raises on a mismatch."""
    tol = derive(fn, inputs, like=out, dtype=dtype, tf32=tf32, accumulate=accumulate,
                 samples=samples, seed=seed, kwargs=kwargs, memo=memo)
    return tol.check(out, explain=explain, fn=fn, inputs=inputs, kwargs=kwargs)


def assert_close(fn: Callable, inputs, out, *, dtype=None, tf32: Optional[bool] = None,
                 accumulate=None, samples: int = SAMPLES, seed: int = 0, kwargs=None,
                 msg: Optional[str] = None) -> CloseReport:
    """Assert that `out` is what a correct implementation of `fn` returns on `inputs`.

    No rtol or atol: the tolerance is derived per element from a float64 run of fn, a correctly
    rounded run and an error model of the working precision (see the module docstring). Raises
    ToleranceError (an AssertionError) naming the bad elements, their error in ulps, where they sit,
    and a policy that would explain them. Returns the CloseReport when it passes.

        truestar.assert_close(torch.nn.functional.softmax, (x, -1), my_softmax(x))
    """
    rep = close_report(fn, inputs, out, dtype=dtype, tf32=tf32, accumulate=accumulate,
                       samples=samples, seed=seed, kwargs=kwargs)
    if not rep.ok:
        text = rep.message()
        raise ToleranceError(f"{msg}: {text}" if msg else text, rep)
    return rep
