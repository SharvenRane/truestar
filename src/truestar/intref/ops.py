"""Integer reference ops: conv, fully connected, matmul, add, mul, pooling, softmax, requantize,
quantize. Every op takes integer tensors, zero points and float scales, and a `Convention`, and
returns the integer output tensor the convention defines.

Layouts follow PyTorch and ONNX (NCHW activations, OIHW weights). TFLite stores NHWC; the
arithmetic per output element is the same, so transpose a TFLite tensor before passing it.

Accumulators are exact: products are summed in float64, which is exact while every partial sum stays
below 2^53 (int8 times int8 times 2^38 terms, or int16 times int8 times 2^30 terms), and the bound
is asserted. That is about 10x faster than an int64 einsum and gives the same bits.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .conventions import CONVENTIONS, POOL_ROUNDING, QUANTIZERS, RULES, Convention
from . import fixedpoint as fp

Params = Dict[str, Any]
EXACT_LIMIT = float(1 << 53)


def _conv(c: Union[str, Convention]) -> Convention:
    return CONVENTIONS[c] if isinstance(c, str) else c


def qrange(bits: int, signed: bool = True) -> Tuple[int, int]:
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def _clamp(v: np.ndarray, p: Params, default_bits: int = 8) -> np.ndarray:
    lo, hi = qrange(int(p.get("out_bits", default_bits)), bool(p.get("out_signed", True)))
    lo = max(lo, int(p.get("act_min", lo)))
    hi = min(hi, int(p.get("act_max", hi)))
    return np.clip(v, lo, hi)


def _exact_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Integer a @ b, exact. Rejects inputs whose partial sums could pass 2^53."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    bound = float(np.abs(a).max(initial=0)) * float(np.abs(b).max(initial=0)) * a.shape[-1]
    if bound >= EXACT_LIMIT:
        raise ValueError(f"accumulator bound {bound:.3g} is not exact in float64 (2^53); "
                         "inputs are wider than int16 x int8 at this reduction length")
    return np.rint(a.astype(np.float64) @ b.astype(np.float64)).astype(np.int64)


def _pair(v, n=2) -> Tuple[int, ...]:
    if isinstance(v, (list, tuple)):
        return tuple(int(t) for t in v)
    return (int(v),) * n


def _per_channel(v, n: int) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    return np.full(n, float(a)) if a.ndim == 0 else a.reshape(n)


def conv_accumulator(x: np.ndarray, w: np.ndarray, bias: Optional[np.ndarray], x_zp, w_zp,
                     stride=1, padding=0, dilation=1, groups: int = 1) -> np.ndarray:
    """int64 accumulator of a 2D conv: sum (x - x_zp) * (w - w_zp) + bias, NCHW and OIHW.
    Padding holds the zero point, so padded taps add nothing (as in TFLite and ONNX)."""
    x = np.asarray(x, dtype=np.int64) - np.asarray(x_zp, dtype=np.int64)
    w = np.asarray(w, dtype=np.int64)
    O = w.shape[0]
    wz = np.asarray(w_zp, dtype=np.int64)
    w = w - (wz.reshape(O, 1, 1, 1) if wz.ndim else wz)
    sh, sw_ = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    N, C, H, W = x.shape
    _, Cg, KH, KW = w.shape
    if C != Cg * groups or O % groups:
        raise ValueError(f"conv shapes do not fit: input channels {C}, weight {w.shape}, groups {groups}")
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    OH = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    OW = (W + 2 * pw - dw * (KW - 1) - 1) // sw_ + 1
    out = np.empty((N, O, OH, OW), dtype=np.int64)
    og = O // groups
    for g in range(groups):
        xs = xp[:, g * Cg:(g + 1) * Cg]
        cols = np.empty((N, OH, OW, Cg, KH, KW), dtype=np.int64)
        for i in range(KH):
            for j in range(KW):
                cols[:, :, :, :, i, j] = xs[:, :, i * dh:i * dh + sh * (OH - 1) + 1:sh,
                                            j * dw:j * dw + sw_ * (OW - 1) + 1:sw_].transpose(0, 2, 3, 1)
        wg = w[g * og:(g + 1) * og].reshape(og, -1)
        acc = _exact_matmul(cols.reshape(N * OH * OW, -1), wg.T)
        out[:, g * og:(g + 1) * og] = acc.reshape(N, OH, OW, og).transpose(0, 3, 1, 2)
    if bias is not None:
        out = out + np.asarray(bias, dtype=np.int64).reshape(1, O, 1, 1)
    return out


def _is16(x) -> bool:
    return np.asarray(x).dtype in (np.int16, np.uint16)


def conv2d(x, w, bias, p: Params, convention) -> np.ndarray:
    """Quantized conv. p: x_zp, w_zp, sx, sw (scalar or per output channel), sy, y_zp, stride,
    padding, dilation, groups, out_bits, act_min, act_max."""
    c = _conv(convention)
    acc = conv_accumulator(x, w, bias, p.get("x_zp", 0), p.get("w_zp", 0), p.get("stride", 1),
                           p.get("padding", 0), p.get("dilation", 1), int(p.get("groups", 1)))
    O = acc.shape[1]
    real = (float(p["sx"]) * _per_channel(p["sw"], O) / float(p["sy"])).reshape(1, O, 1, 1)
    int16 = bool(p.get("int16", _is16(x)))
    rule = c.rule_for("conv", int16)
    y = rule.requant(acc, real) + int(p.get("y_zp", 0))
    return _clamp(y, p, 16 if int16 else 8)


def fully_connected(x, w, bias, p: Params, convention, op: str = "fully_connected") -> np.ndarray:
    """Quantized fully connected: x [N, K], w [O, K]; same params as conv2d."""
    c = _conv(convention)
    xz = np.asarray(x, dtype=np.int64) - int(p.get("x_zp", 0))
    w = np.asarray(w, dtype=np.int64)
    wz = np.asarray(p.get("w_zp", 0), dtype=np.int64)
    w = w - (wz.reshape(-1, 1) if wz.ndim else wz)
    acc = _exact_matmul(xz, w.T)
    if bias is not None:
        acc = acc + np.asarray(bias, dtype=np.int64).reshape(1, -1)
    O = w.shape[0]
    real = (float(p["sx"]) * _per_channel(p["sw"], O) / float(p["sy"])).reshape(1, O)
    int16 = bool(p.get("int16", _is16(x)))
    y = c.rule_for(op, int16).requant(acc, real) + int(p.get("y_zp", 0))
    return _clamp(y, p, 16 if int16 else 8)


def matmul(a, b, p: Params, convention) -> np.ndarray:
    """Quantized a @ b (ONNX QLinearMatMul, TFLite BATCH_MATMUL): p: a_zp, b_zp, sa, sb, sy, y_zp."""
    c = _conv(convention)
    az = np.asarray(a, dtype=np.int64) - int(p.get("a_zp", 0))
    bz = np.asarray(b, dtype=np.int64) - int(p.get("b_zp", 0))
    acc = _exact_matmul(az, bz)
    real = float(p["sa"]) * float(p["sb"]) / float(p["sy"])
    y = c.rule_for("matmul", _is16(a)).requant(acc, real) + int(p.get("y_zp", 0))
    return _clamp(y, p)


def add(a, b, p: Params, convention) -> np.ndarray:
    """Quantized elementwise add. p: a_zp, b_zp, sa, sb, sy, y_zp.

    TFLite structure (TFLite, CMSIS-NN): both inputs are shifted left by 20 (15 for int16),
    rescaled to twice the larger input scale, summed and rescaled to the output, each step through
    the convention's rule. Float structure (ONNX): float32 (sa/sy)(a - za) + (sb/sy)(b - zb).
    """
    c = _conv(convention)
    ai = np.asarray(a, dtype=np.int64) - int(p.get("a_zp", 0))
    bi = np.asarray(b, dtype=np.int64) - int(p.get("b_zp", 0))
    sa, sb, sy = float(p["sa"]), float(p["sb"]), float(p["sy"])
    rule = c.rule_for("add")
    if c.add_structure == "float":
        f = np.float32
        v = f(f(sa) / f(sy)) * ai.astype(f) + f(f(sb) / f(sy)) * bi.astype(f)
        r = fp.round_half_away if rule.name.endswith("away") else fp.round_half_even
        y = r(v.astype(np.float64)).astype(np.int64)
    else:
        left = 15 if _is16(a) else 20
        twice = 2.0 * max(sa, sb)
        ra = rule.requant(ai * (1 << left), sa / twice)
        rb = rule.requant(bi * (1 << left), sb / twice)
        y = rule.requant(ra + rb, twice / ((1 << left) * sy))
    return _clamp(y + int(p.get("y_zp", 0)), p, 16 if _is16(a) else 8)


def mul(a, b, p: Params, convention) -> np.ndarray:
    """Quantized elementwise mul: (a - za)(b - zb) rescaled by sa sb / sy."""
    c = _conv(convention)
    acc = (np.asarray(a, dtype=np.int64) - int(p.get("a_zp", 0))) * \
          (np.asarray(b, dtype=np.int64) - int(p.get("b_zp", 0)))
    y = c.rule_for("mul").requant(acc, float(p["sa"]) * float(p["sb"]) / float(p["sy"]))
    return _clamp(y + int(p.get("y_zp", 0)), p, 16 if _is16(a) else 8)


def _windows(x: np.ndarray, p: Params, pad_value: int):
    kh, kw = _pair(p["kernel"])
    sh, sw_ = _pair(p.get("stride", p["kernel"]))
    ph, pw = _pair(p.get("padding", 0))
    x = np.asarray(x, dtype=np.int64)
    N, C, H, W = x.shape
    valid = np.pad(np.ones((H, W), dtype=np.int64), ((ph, ph), (pw, pw)))
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=pad_value)
    OH = (H + 2 * ph - kh) // sh + 1
    OW = (W + 2 * pw - kw) // sw_ + 1
    vals, cnt = [], np.zeros((OH, OW), dtype=np.int64)
    for i in range(kh):
        for j in range(kw):
            vals.append(xp[:, :, i:i + sh * (OH - 1) + 1:sh, j:j + sw_ * (OW - 1) + 1:sw_])
            cnt += valid[i:i + sh * (OH - 1) + 1:sh, j:j + sw_ * (OW - 1) + 1:sw_]
    return vals, cnt


def avg_pool(x, p: Params, convention) -> np.ndarray:
    """Average pool with the same input and output quantization; padding is not counted
    (TFLite and CMSIS-NN; ONNX count_include_pad=0)."""
    c = _conv(convention)
    vals, cnt = _windows(x, p, 0)
    acc = np.sum(vals, axis=0)
    y = POOL_ROUNDING[c.pool](acc, cnt[None, None])
    return _clamp(y, p, 16 if _is16(x) else 8)


def max_pool(x, p: Params, convention=None) -> np.ndarray:
    """Max pool: exact in every convention (a control: a device that differs here is not rounding)."""
    lo = qrange(16 if _is16(x) else 8)[0]
    vals, _ = _windows(x, p, lo - 1)
    return np.max(vals, axis=0)


def requantize(x, p: Params, convention) -> np.ndarray:
    """Rescale integers to new quantization: (x - x_zp) * sx / sy + y_zp (TFLite REQUANTIZE,
    int8 to int16 and back)."""
    c = _conv(convention)
    acc = np.asarray(x, dtype=np.int64) - int(p.get("x_zp", 0))
    y = c.rule_for("requantize").requant(acc, float(p["sx"]) / float(p["sy"])) + int(p.get("y_zp", 0))
    return _clamp(y, p, int(p.get("out_bits", 8)))


def quantize(x, p: Params, convention) -> np.ndarray:
    """Float to integer with the convention's quantizer: round(x / scale) + zp, saturated."""
    c = _conv(convention)
    q = QUANTIZERS[c.quantizer](x, float(p["scale"])).astype(np.int64) + int(p.get("zp", 0))
    return _clamp(q, p, int(p.get("out_bits", 8)))


def softmax(x, p: Params, convention) -> np.ndarray:
    """Softmax over the last axis: dequantize, exp in float64, quantize with the convention's
    quantizer (default output scale 1/256, zero point -128, the TFLite int8 spec).

    TFLite's and CMSIS-NN's own int8 softmax use fixed point exp tables; those are not modelled
    here, so softmax distinguishes conventions only by their final rounding (docs/INTREF.md)."""
    c = _conv(convention)
    v = (np.asarray(x, dtype=np.int64) - int(p.get("x_zp", 0))).astype(np.float64) * float(p["sx"])
    v = v - v.max(axis=-1, keepdims=True)
    e = np.exp(v)
    s = e / e.sum(axis=-1, keepdims=True)
    sy = float(p.get("sy", 1.0 / 256))
    q = QUANTIZERS[c.quantizer](s.astype(np.float32), sy).astype(np.int64) + int(p.get("y_zp", -128))
    return _clamp(q, p)


OPS = {
    "conv": lambda i, p, c: conv2d(i["x"], i["w"], i.get("b"), p, c),
    "fully_connected": lambda i, p, c: fully_connected(i["x"], i["w"], i.get("b"), p, c),
    "matmul": lambda i, p, c: matmul(i["a"], i["b"], p, c),
    "add": lambda i, p, c: add(i["a"], i["b"], p, c),
    "mul": lambda i, p, c: mul(i["a"], i["b"], p, c),
    "avg_pool": lambda i, p, c: avg_pool(i["x"], p, c),
    "max_pool": lambda i, p, c: max_pool(i["x"], p, c),
    "softmax": lambda i, p, c: softmax(i["x"], p, c),
    "requantize": lambda i, p, c: requantize(i["x"], p, c),
    "quantize": lambda i, p, c: quantize(i["x"], p, c),
}


def run_op(op: str, inputs: Dict[str, np.ndarray], params: Params, convention) -> np.ndarray:
    """Run one op under one convention."""
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; known: {', '.join(OPS)}")
    return OPS[op](inputs, params, convention)
