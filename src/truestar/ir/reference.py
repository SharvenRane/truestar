"""Reference interpreter: slow, obviously correct, float64.

Every floating point value is computed in float64 on CPU. Torch is used only for basic arithmetic
(elementwise math, sums, matmul, indexing). Composite ops are written from their textbook formulas
instead of calling the fused PyTorch op, so this reference does not share a bug with PyTorch's
fused kernels:

* conv2d is a sum over kernel taps of strided slices times weights, not ``F.conv2d``;
* batch norm, layer norm, softmax, gelu, hardswish, silu, attention and pooling are written out.

After every node the interpreter checks the produced tensor's shape and dtype against the type the
IR declared, so a wrong shape rule or a wrong kernel can never pass silently.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import sympy
import torch

from .core import IRError, Node, Program, ShapeError, TensorType, eval_dim, symbol

Impl = Callable[[Dict[str, Any], List[Optional[torch.Tensor]], Tuple[int, ...]], torch.Tensor]
IMPLS: Dict[str, Impl] = {}

COMPUTE_DTYPE = torch.float64


def impl(*names: str):
    def deco(fn):
        for n in names:
            IMPLS[n] = fn
        return fn
    return deco


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return COMPUTE_DTYPE if dtype.is_floating_point else dtype


# ----------------------------------------------------------------------------- elementwise


def _sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def _clamp(x, lo, hi):
    if lo is not None:
        x = torch.where(x < lo, torch.full_like(x, lo), x)
    if hi is not None:
        x = torch.where(x > hi, torch.full_like(x, hi), x)
    return x


_UNARY = {
    "neg": lambda x, a: -x,
    "exp": lambda x, a: torch.exp(x),
    "sqrt": lambda x, a: torch.sqrt(x),
    "rsqrt": lambda x, a: 1.0 / torch.sqrt(x),
    "tanh": lambda x, a: torch.tanh(x),
    "relu": lambda x, a: torch.where(x < 0, torch.zeros_like(x), x),
    "sigmoid": lambda x, a: _sigmoid(x),
    "silu": lambda x, a: x * _sigmoid(x),
    "hardsigmoid": lambda x, a: _clamp(x + 3.0, 0.0, 6.0) / 6.0,
    "hardswish": lambda x, a: x * _clamp(x + 3.0, 0.0, 6.0) / 6.0,
    "clamp": lambda x, a: _clamp(x, a.get("min"), a.get("max")),
    "logical_not": lambda x, a: ~x,
}


def _gelu(x, a):
    if a.get("approximate", "none") == "tanh":
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


_UNARY["gelu"] = _gelu

for _name, _fn in _UNARY.items():
    IMPLS[_name] = (lambda fn: lambda attrs, args, shape: fn(args[0], attrs))(_fn)


def _binary(fn):
    def run(attrs, args, shape):
        a, b = args
        if a.dtype != b.dtype and (a.dtype.is_floating_point or b.dtype.is_floating_point):
            a, b = a.to(COMPUTE_DTYPE), b.to(COMPUTE_DTYPE)
        return fn(a, b)
    return run


IMPLS["add"] = _binary(lambda a, b: a + b)
IMPLS["sub"] = _binary(lambda a, b: a - b)
IMPLS["mul"] = _binary(lambda a, b: a * b)
IMPLS["div"] = _binary(lambda a, b: a.to(COMPUTE_DTYPE) / b.to(COMPUTE_DTYPE))
IMPLS["eq"] = _binary(lambda a, b: a == b)
IMPLS["ne"] = _binary(lambda a, b: a != b)
IMPLS["lt"] = _binary(lambda a, b: a < b)
IMPLS["le"] = _binary(lambda a, b: a <= b)
IMPLS["gt"] = _binary(lambda a, b: a > b)
IMPLS["ge"] = _binary(lambda a, b: a >= b)
IMPLS["logical_and"] = _binary(lambda a, b: (a != 0) & (b != 0))
IMPLS["logical_or"] = _binary(lambda a, b: (a != 0) | (b != 0))


@impl("full")
def _full(attrs, args, shape):
    return torch.full(shape, attrs["value"], dtype=_compute_dtype(attrs["dtype"]))


@impl("arange")
def _arange(attrs, args, shape):
    if any(isinstance(attrs[k], sympy.Basic) for k in ("start", "end", "step")):
        # symbolic end: the length is already solved into ``shape``; value i is start + i * step
        return attrs["start"] + attrs["step"] * _iota(shape[0], _compute_dtype(attrs["dtype"]))
    return torch.arange(attrs["start"], attrs["end"], attrs["step"], dtype=_compute_dtype(attrs["dtype"]))


@impl("where")
def _where(attrs, args, shape):
    c, a, b = args
    if a.dtype != b.dtype:
        a, b = a.to(COMPUTE_DTYPE), b.to(COMPUTE_DTYPE)
    return torch.where(c, a, b)


@impl("full_like")
def _full_like(attrs, args, shape):
    dtype = _compute_dtype(attrs.get("dtype") or args[0].dtype)
    return torch.full(shape, attrs["value"], dtype=dtype)


# ----------------------------------------------------------------------------- reductions


def _dims(attrs, x):
    dims = attrs.get("dims")
    return list(range(x.dim())) if dims is None or len(dims) == 0 else [d % max(x.dim(), 1) for d in dims]


@impl("sum")
def _sum(attrs, args, shape):
    x = args[0]
    return x.sum(dim=_dims(attrs, x), keepdim=attrs.get("keepdim", False))


@impl("mean")
def _mean(attrs, args, shape):
    x = args[0]
    dims = _dims(attrs, x)
    count = 1
    for d in dims:
        count *= x.shape[d]
    return x.sum(dim=dims, keepdim=attrs.get("keepdim", False)) / count


@impl("any")
def _any(attrs, args, shape):
    x = args[0]
    out = x
    for d in sorted(_dims(attrs, x), reverse=True):
        out = out.any(dim=d, keepdim=attrs.get("keepdim", False))
    return out


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = x - x.amax(dim=dim, keepdim=True)
    e = torch.exp(shifted)
    return e / e.sum(dim=dim, keepdim=True)


@impl("softmax")
def _softmax(attrs, args, shape):
    return softmax(args[0], attrs["dim"])


# ----------------------------------------------------------------------------- linear algebra


@impl("matmul")
def _matmul(attrs, args, shape):
    return torch.matmul(args[0], args[1])


@impl("linear")
def _linear(attrs, args, shape):
    x, w, b = args
    y = torch.matmul(x, w.transpose(0, 1))
    return y if b is None else y + b


@impl("conv2d")
def _conv2d(attrs, args, shape):
    """out[n, o, i, j] = b[o] + sum over c, u, v of x[n, c, i*s + u*d, j*s + v*d] * w[o, c, u, v]."""
    x, w, b = args
    sh, sw = _pair(attrs.get("stride", 1))
    ph, pw = _pair(attrs.get("padding", 0))
    dh, dw = _pair(attrs.get("dilation", 1))
    g = attrs.get("groups", 1)
    n, c, h, wd = x.shape
    o, cg, kh, kw = w.shape
    _, _, oh, ow = shape
    xp = torch.zeros((n, c, h + 2 * ph, wd + 2 * pw), dtype=x.dtype)
    xp[:, :, ph:ph + h, pw:pw + wd] = x
    xg = xp.reshape(n, g, cg, h + 2 * ph, wd + 2 * pw)
    wg = w.reshape(g, o // g, cg, kh, kw)
    out = torch.zeros((n, g, o // g, oh, ow), dtype=x.dtype)
    for u in range(kh):
        for v in range(kw):
            r0, c0 = u * dh, v * dw
            patch = xg[:, :, :, r0:r0 + sh * (oh - 1) + 1:sh, c0:c0 + sw * (ow - 1) + 1:sw]
            out = out + torch.einsum("ngchw,goc->ngohw", patch, wg[:, :, :, u, v])
    out = out.reshape(n, o, oh, ow)
    if b is not None:
        out = out + b.reshape(1, o, 1, 1)
    return out


def _pair(v):
    if isinstance(v, int):
        return v, v
    v = list(v)
    return (v[0], v[0]) if len(v) == 1 else (v[0], v[1])


# ----------------------------------------------------------------------------- normalization


@impl("batch_norm")
def _batch_norm(attrs, args, shape):
    """y = (x - running_mean) / sqrt(running_var + eps) * weight + bias, per channel."""
    x, w, b, mean, var = args
    view = (1, -1) + (1,) * (x.dim() - 2)
    y = (x - mean.reshape(view)) / torch.sqrt(var.reshape(view) + attrs["eps"])
    if w is not None:
        y = y * w.reshape(view)
    if b is not None:
        y = y + b.reshape(view)
    return y


@impl("layer_norm")
def _layer_norm(attrs, args, shape):
    """y = (x - mean) / sqrt(biased variance + eps) * weight + bias over the trailing dims."""
    x, w, b = args
    dims = list(range(x.dim() - len(attrs["normalized_shape"]), x.dim()))
    count = 1
    for d in dims:
        count *= x.shape[d]
    mu = x.sum(dim=dims, keepdim=True) / count
    var = ((x - mu) * (x - mu)).sum(dim=dims, keepdim=True) / count
    y = (x - mu) / torch.sqrt(var + attrs["eps"])
    if w is not None:
        y = y * w
    if b is not None:
        y = y + b
    return y


@impl("attention")
def _attention(attrs, args, shape):
    """softmax(q k^T * scale + mask) v, with scale = 1 / sqrt(head dim) unless given."""
    q, k, v, mask = args
    scale = attrs.get("scale")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    L, S = q.shape[-2], k.shape[-2]
    if attrs.get("is_causal", False):
        keep = torch.ones((L, S), dtype=torch.bool).tril()
        scores = torch.where(keep, scores, torch.full_like(scores, -math.inf))
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = torch.where(mask, scores, torch.full_like(scores, -math.inf))
        else:
            scores = scores + mask
    return torch.matmul(softmax(scores, -1), v)


# ----------------------------------------------------------------------------- pooling


@impl("max_pool2d")
def _max_pool2d(attrs, args, shape):
    x = args[0]
    kh, kw = _pair(attrs["kernel_size"])
    sh, sw = _pair(attrs.get("stride") or attrs["kernel_size"])
    ph, pw = _pair(attrs.get("padding", 0))
    dh, dw = _pair(attrs.get("dilation", 1))
    n, c, h, w = x.shape
    _, _, oh, ow = shape
    # pad with -inf on every side, enough for the last window (ceil mode may need extra on the right)
    H = max(h + 2 * ph, (oh - 1) * sh + dh * (kh - 1) + 1)
    W = max(w + 2 * pw, (ow - 1) * sw + dw * (kw - 1) + 1)
    xp = torch.full((n, c, H, W), -math.inf, dtype=x.dtype)
    xp[:, :, ph:ph + h, pw:pw + w] = x
    out = torch.full((n, c, oh, ow), -math.inf, dtype=x.dtype)
    for u in range(kh):
        for v in range(kw):
            r0, c0 = u * dh, v * dw
            tap = xp[:, :, r0:r0 + sh * (oh - 1) + 1:sh, c0:c0 + sw * (ow - 1) + 1:sw]
            out = torch.maximum(out, tap)
    return out


@impl("adaptive_avg_pool2d")
def _adaptive_avg_pool2d(attrs, args, shape):
    """Output cell i averages input rows floor(i*H/out) to ceil((i+1)*H/out), same for columns."""
    x = args[0]
    h, w = x.shape[-2:]
    oh, ow = shape[-2:]
    rows = []
    for i in range(oh):
        r0, r1 = (i * h) // oh, -(-((i + 1) * h) // oh)
        cols = []
        for j in range(ow):
            c0, c1 = (j * w) // ow, -(-((j + 1) * w) // ow)
            cell = x[..., r0:r1, c0:c1]
            cols.append(cell.sum(dim=(-2, -1)) / ((r1 - r0) * (c1 - c0)))
        rows.append(torch.stack(cols, dim=-1))
    return torch.stack(rows, dim=-2)


# ----------------------------------------------------------------------------- shape ops


@impl("reshape")
def _reshape(attrs, args, shape):
    return args[0].reshape(shape)


@impl("permute")
def _permute(attrs, args, shape):
    return args[0].permute(*attrs["dims"])


@impl("expand")
def _expand(attrs, args, shape):
    return args[0].expand(*shape)


@impl("select")
def _select(attrs, args, shape):
    return args[0].select(attrs["dim"], attrs["index"])


@impl("slice")
def _slice(attrs, args, shape):
    x = args[0]
    d = attrs["dim"] % x.dim()
    idx = [slice(None)] * x.dim()
    idx[d] = slice(attrs.get("start"), attrs.get("end"), attrs.get("step", 1))
    return x[tuple(idx)]


@impl("cat")
def _cat(attrs, args, shape):
    dtype = args[0].dtype
    for t in args[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    return torch.cat([t.to(dtype) for t in args], dim=attrs["dim"])


# ----------------------------------------------------------------------------- interpreter


# ----------------------------------------------------------------------------- dynamic index, dims,
# control flow. These impls also receive the symbol bindings of the running program.

BINDING_OPS = {"dim_scalar", "cond", "loop"}


@impl("select_dynamic")
def _select_dynamic(attrs, args, shape):
    x, i = args
    d = attrs["dim"] % x.dim()
    idx = int(i.item())
    if not -x.shape[d] <= idx < x.shape[d]:
        raise IRError(f"select_dynamic index {idx} is out of range for dim {d} of size {x.shape[d]}")
    return x.select(d, idx)


@impl("dim_scalar")
def _dim_scalar(attrs, args, shape, bindings=None):
    return torch.tensor(eval_dim(attrs["expr"], bindings or {}), dtype=_compute_dtype(attrs["dtype"]))


# Called as COND_TRACE(attrs, taken, operands, bindings) at every cond the interpreter runs, when set.
# The exporter uses it to report which branches its checks exercised and to verify the other one.
COND_TRACE = None


@impl("cond")
def _cond(attrs, args, shapes, bindings=None):
    pred, operands = args[0], args[1:]
    taken = bool(pred.item())
    if COND_TRACE is not None:
        COND_TRACE(attrs, taken, operands, bindings)
    region = attrs["then_branch"] if taken else attrs["else_branch"]
    return _run(region, operands, bindings, device=_active_device())


@impl("loop")
def _loop(attrs, args, shapes, bindings=None):
    n = attrs["n_carried"]
    carried, captures = list(args[:n]), list(args[n:])
    body, cond = attrs["body"], attrs.get("cond")
    limit = attrs.get("max_iterations")
    if cond is None:
        trips = max(0, eval_dim(attrs["trip_count"], bindings or {}))
        for k in range(trips):
            carried = _run(body, [torch.tensor(k, dtype=torch.int64)] + carried + captures, bindings,
                           device=_active_device())
        return carried
    k = 0
    while bool(_run(cond, carried + captures, bindings, device=_active_device())[0].item()):
        if limit is not None and k >= limit:
            raise IRError(f"loop ran more than max_iterations={limit} times")
        carried = _run(body, [torch.tensor(k, dtype=torch.int64)] + carried + captures, bindings,
                       device=_active_device())
        k += 1
    return carried


# ----------------------------------------------------------------------------- interpreter


def bind_symbols(prog: Program, inputs: Sequence[torch.Tensor],
                 known: Optional[Dict[sympy.Symbol, int]] = None) -> Dict[sympy.Symbol, int]:
    """Match concrete input shapes against the declared input types and solve for every symbol.

    ``known`` seeds the bindings (a region inherits the symbols of the program that runs it)."""
    if len(inputs) != len(prog.inputs):
        raise IRError(f"program {prog.name} takes {len(prog.inputs)} inputs, got {len(inputs)}")
    bindings: Dict[sympy.Symbol, int] = dict(known or {})
    for v, t in zip(prog.inputs, inputs):
        if not isinstance(t, torch.Tensor):
            raise IRError(f"input %{v.name} must be a tensor, got {type(t).__name__}")
        if t.dim() != v.type.rank:
            raise ShapeError("input", f"%{v.name} has rank {t.dim()}, declared rank {v.type.rank}",
                             [(f"%{v.name} declared", v.type), ("given", list(t.shape))])
        for i, (d, n) in enumerate(zip(v.type.shape, t.shape)):
            if isinstance(d, sympy.Symbol):
                if d in bindings and bindings[d] != n:
                    raise ShapeError("input", f"symbol {d} is {bindings[d]} elsewhere but {n} in %{v.name} dim {i}",
                                     [(f"%{v.name} declared", v.type), ("given", list(t.shape))])
                lo, hi = prog.symbols.get(str(d), (1, None))
                if n < lo or (hi is not None and n > hi):
                    raise ShapeError("input", f"symbol {d} = {n} is outside its range [{lo}, {hi}]",
                                     [(f"%{v.name} declared", v.type), ("given", list(t.shape))])
                bindings[d] = int(n)
    # Derived dims (2*T, T + 1): solve each for its one still unknown symbol (added 2026-11-24).
    for v, t in zip(prog.inputs, inputs):
        for i, (d, n) in enumerate(zip(v.type.shape, t.shape)):
            if isinstance(d, sympy.Basic) and not isinstance(d, sympy.Symbol):
                free = [s for s in d.free_symbols if s not in bindings]
                if len(free) != 1:
                    continue
                sols = [r for r in sympy.solve(sympy.Eq(d.subs(bindings), int(n)), free[0])
                        if r.is_Integer and r > 0]
                if len(sols) != 1:
                    raise ShapeError("input", f"%{v.name} dim {i} is {n}, which {d} cannot equal for a positive "
                                              f"integer {free[0]}",
                                     [(f"%{v.name} declared", v.type), ("given", list(t.shape))])
                lo, hi = prog.symbols.get(str(free[0]), (1, None))
                val = int(sols[0])
                if val < lo or (hi is not None and val > hi):
                    raise ShapeError("input", f"symbol {free[0]} = {val} is outside its range [{lo}, {hi}]",
                                     [(f"%{v.name} declared", v.type), ("given", list(t.shape))])
                bindings[free[0]] = val
    for v, t in zip(prog.inputs, inputs):
        want = [eval_dim(d, bindings) for d in v.type.shape]
        if want != list(t.shape):
            raise ShapeError("input", f"%{v.name} shape does not match its declared type",
                             [(f"%{v.name} declared", v.type), ("given", list(t.shape))])
    return bindings


def _run(prog: Program, inputs: Sequence[torch.Tensor], known=None, return_all: bool = False, device=None,
         overrides: Optional[Dict[int, torch.Tensor]] = None):
    dev = torch.device("cpu") if device is None else torch.device(device)
    if overrides and dev.type != "cpu":
        raise ValueError("overrides are supported by the CPU interpreter only")
    if dev.type != "cpu":
        return _run_on_device(prog, inputs, known, return_all, dev)
    bindings = bind_symbols(prog, inputs, known)
    env: Dict[int, torch.Tensor] = {}
    for v, t in zip(prog.inputs, inputs):
        t = t.detach()
        if t.device.type == "mps":   # copy first, widen on the CPU (kernel.verify.to_host_f64)
            t = t.cpu()
        env[id(v)] = t.to("cpu", _compute_dtype(t.dtype))
    for p in prog.params:
        data = prog.param_data[p.name]
        env[id(p)] = data.detach().to("cpu", _compute_dtype(data.dtype))
    named: Dict[str, torch.Tensor] = {}
    for node in prog.nodes:
        fn = IMPLS.get(node.op)
        if fn is None:
            raise IRError(f"the reference interpreter has no implementation of {node.op}")
        args = [None if x is None else env[id(x)] for x in node.inputs]
        outs_decl = node.outputs
        shapes = [tuple(eval_dim(d, bindings) for d in o.type.shape) for o in outs_decl]
        if node.op in BINDING_OPS:
            result = fn(node.attrs, args, shapes[0] if len(shapes) == 1 else shapes, bindings=bindings)
        else:
            result = fn(node.attrs, args, shapes[0] if len(shapes) == 1 else shapes)
        results = list(result) if isinstance(result, (list, tuple)) else [result]
        if len(results) != len(outs_decl):
            raise IRError(f"reference {node.op} produced {len(results)} values, the node defines {len(outs_decl)}")
        for out, o, shape in zip(results, outs_decl, shapes):
            want_dtype = _compute_dtype(o.type.dtype)
            if tuple(out.shape) != shape or out.dtype != want_dtype:
                raise IRError(f"reference {node.op} produced {out.dtype} {list(out.shape)} but %{o.name} "
                              f"is declared {o.type} = {want_dtype} {list(shape)} ({node.source})")
            if overrides is not None and id(o) in overrides:
                out = overrides[id(o)].to("cpu", out.dtype)
            env[id(o)] = out
            if return_all:
                named[o.name] = out
    outs = [env[id(v)] for v in prog.outputs]
    return (outs, named) if return_all else outs


def run(prog: Program, *inputs: torch.Tensor, return_all: bool = False, device=None,
        overrides: Optional[Dict[int, torch.Tensor]] = None):
    """Execute ``prog`` in float64. Returns a list of outputs (or every value with return_all).

    ``device`` (default: the CPU) is where the float64 math runs; see ``_run_on_device``.
    ``overrides`` (CPU only): ``id(value)`` of top level values to the tensor to use instead of
    what their node computes, after the node's shape and dtype check. The compiler replays a
    program with the branch decisions a compiled run took (runtime ``_branch_flips``)."""
    return _run(prog, inputs, None, return_all, device=device, overrides=overrides)


# ----------------------------------------------------------------------------- the same math on a GPU
# Added 2026-11-24 for the compiler's region and end to end checks (docs/COMPILER.md). The node
# implementations above are plain torch float64 code, so they run unchanged on a CUDA device: the
# interpreter only puts its tensors there and runs each node under ``torch.device(dev)``, so the
# tensors an implementation creates (zeros, full, arange, loop counters) land there too. Results
# differ from the CPU run only by float64 rounding (summation order, libm last place).
#
# Three things differ from the CPU loop, none of them in the math:
# * a value is freed after its last reader (unless return_all), so a whole program fits;
# * attention whose score matrix exceeds DEVICE_CHUNK_ELEMENTS runs in blocks of query rows, each
#   row with exactly the formula of ``_attention`` (rows never interact);
# * an op torch cannot run on the device in float64 runs on the CPU for that node only and is named
#   in ``device_cpu_fallbacks``. On torch 2.11 with CUDA 12.8 no op needed it.

DEVICE_CHUNK_ELEMENTS = 1 << 24          # 128 MB of float64 per score block on the device
device_cpu_fallbacks: set = set()

_state = threading.local()


def _active_device():
    return getattr(_state, "device", None)


def _attention_rows(attrs, args, shape):
    """``_attention`` evaluated in blocks of query rows when the score matrix is large."""
    q, k, v, mask = args
    L, S = q.shape[-2], k.shape[-2]
    lead = 1
    for d in torch.broadcast_shapes(q.shape[:-2], k.shape[:-2]):
        lead *= d
    rows = max(1, DEVICE_CHUNK_ELEMENTS // max(1, lead * S))
    if rows >= L:
        return _attention(attrs, args, shape)
    scale = attrs.get("scale")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    causal = attrs.get("is_causal", False)
    mask_rows = mask is not None and mask.dim() >= 2 and mask.shape[-2] == L
    out = torch.empty(shape, dtype=torch.promote_types(q.dtype, v.dtype), device=q.device)
    cols = torch.arange(S, device=q.device)
    for lo in range(0, L, rows):
        hi = min(L, lo + rows)
        scores = torch.matmul(q[..., lo:hi, :], k.transpose(-2, -1)) * scale
        if causal:            # tril of ones(L, S): keep column j of row i when j <= i
            keep = cols[None, :] <= torch.arange(lo, hi, device=q.device)[:, None]
            scores = torch.where(keep, scores, torch.full_like(scores, -math.inf))
        if mask is not None:
            m = mask[..., lo:hi, :] if mask_rows else mask
            if m.dtype == torch.bool:
                scores = torch.where(m, scores, torch.full_like(scores, -math.inf))
            else:
                scores = scores + m
        out[..., lo:hi, :] = torch.matmul(softmax(scores, -1), v)
        del scores
    return out


DEVICE_IMPLS: Dict[str, Impl] = {"attention": _attention_rows}


def _device_unsupported(e: BaseException) -> bool:
    if isinstance(e, NotImplementedError):
        return True
    return isinstance(e, RuntimeError) and "not implemented for" in str(e)


def call_on_device(op: str, attrs, args, shape, dev: torch.device, bindings=None):
    """One node's reference implementation on ``dev`` (float64 tensors there), with the CPU as the
    fallback for an op torch lacks on the device in float64."""
    fn = DEVICE_IMPLS.get(op) or IMPLS[op]
    prev = _active_device()
    _state.device = dev
    try:
        with torch.device(dev):
            return fn(attrs, args, shape, bindings=bindings) if op in BINDING_OPS else fn(attrs, args, shape)
    except Exception as e:  # noqa: BLE001
        if not _device_unsupported(e):
            raise
    finally:
        _state.device = prev
    device_cpu_fallbacks.add(op)
    cpu_args = [None if a is None else a.cpu() for a in args]
    fn = IMPLS[op]
    res = fn(attrs, cpu_args, shape, bindings=bindings) if op in BINDING_OPS else fn(attrs, cpu_args, shape)
    if isinstance(res, (list, tuple)):
        return [r.to(dev) for r in res]
    return res.to(dev)


def _last_use(prog: Program) -> Dict[int, int]:
    """Index of the last node reading each value (program outputs are never freed)."""
    last: Dict[int, int] = {}
    for i, node in enumerate(prog.nodes):
        for x in node.inputs:
            if x is not None:
                last[id(x)] = i
    for v in prog.outputs:
        last[id(v)] = len(prog.nodes)
    return last


def _run_on_device(prog: Program, inputs, known, return_all: bool, dev: torch.device):
    bindings = bind_symbols(prog, inputs, known)
    env: Dict[int, torch.Tensor] = {}
    for v, t in zip(prog.inputs, inputs):
        env[id(v)] = t.detach().to(dev, _compute_dtype(t.dtype))
    for p in prog.params:
        data = prog.param_data[p.name]
        env[id(p)] = data.detach().to(dev, _compute_dtype(data.dtype))
    last = {} if return_all else _last_use(prog)
    named: Dict[str, torch.Tensor] = {}
    for i, node in enumerate(prog.nodes):
        if node.op not in IMPLS:
            raise IRError(f"the reference interpreter has no implementation of {node.op}")
        args = [None if x is None else env[id(x)] for x in node.inputs]
        outs_decl = node.outputs
        shapes = [tuple(eval_dim(d, bindings) for d in o.type.shape) for o in outs_decl]
        result = call_on_device(node.op, node.attrs, args, shapes[0] if len(shapes) == 1 else shapes, dev,
                                bindings=bindings)
        del args
        results = list(result) if isinstance(result, (list, tuple)) else [result]
        if len(results) != len(outs_decl):
            raise IRError(f"reference {node.op} produced {len(results)} values, the node defines "
                          f"{len(outs_decl)}")
        for out, o, shape in zip(results, outs_decl, shapes):
            want_dtype = _compute_dtype(o.type.dtype)
            if tuple(out.shape) != shape or out.dtype != want_dtype:
                raise IRError(f"reference {node.op} produced {out.dtype} {list(out.shape)} but %{o.name} "
                              f"is declared {o.type} = {want_dtype} {list(shape)} ({node.source})")
            env[id(o)] = out
            if return_all:
                named[o.name] = out
        if not return_all:
            for x in list(node.inputs) + list(outs_decl):
                if x is not None and last.get(id(x), -1) <= i:
                    env.pop(id(x), None)
    outs = [env[id(v)] for v in prog.outputs]
    return (outs, named) if return_all else outs


# ----------------------------------------------------------------------------- ops added 2026-11-24
# Same policy as above: written from the definition with elementwise math, sums, comparisons and
# plain indexing; no fused PyTorch op (no F.avg_pool2d, F.group_norm, F.rms_norm, torch.cumsum,
# torch.gather, torch.linalg.vector_norm, F.pad, torch.triu).


def _iota(n: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(n, dtype=dtype)


IMPLS["cos"] = lambda attrs, args, shape: torch.cos(args[0].to(COMPUTE_DTYPE))
IMPLS["sin"] = lambda attrs, args, shape: torch.sin(args[0].to(COMPUTE_DTYPE))
IMPLS["log"] = lambda attrs, args, shape: torch.log(args[0].to(COMPUTE_DTYPE))
IMPLS["erf"] = lambda attrs, args, shape: torch.erf(args[0].to(COMPUTE_DTYPE))
IMPLS["expm1"] = lambda attrs, args, shape: torch.expm1(args[0].to(COMPUTE_DTYPE))
IMPLS["reciprocal"] = lambda attrs, args, shape: 1.0 / args[0].to(COMPUTE_DTYPE)
IMPLS["abs"] = lambda attrs, args, shape: torch.where(args[0] < 0, -args[0], args[0])


def _pow(a, b):
    return a ** b


def _fmod(a, b):
    """a - trunc(a / b) * b: the result has the sign of a (C fmod)."""
    if a.dtype.is_floating_point:
        return a - torch.trunc(a / b) * b
    return a - torch.div(a, b, rounding_mode="trunc") * b


def _remainder(a, b):
    """a - floor(a / b) * b: the result has the sign of b (Python %)."""
    if a.dtype.is_floating_point:
        return a - torch.floor(a / b) * b
    return a - torch.div(a, b, rounding_mode="floor") * b


def _same(a, b):
    if a.dtype != b.dtype:
        dtype = torch.promote_types(a.dtype, b.dtype)
        a, b = a.to(dtype), b.to(dtype)
    return a, b


IMPLS["pow"] = _binary(lambda a, b: _pow(*_same(a, b)))
IMPLS["fmod"] = _binary(lambda a, b: _fmod(*_same(a, b)))
IMPLS["remainder"] = _binary(lambda a, b: _remainder(*_same(a, b)))
# NaN propagates, as in torch.maximum / torch.minimum
IMPLS["maximum"] = _binary(lambda a, b: (lambda a, b: torch.where((a > b) | (a != a), a, b))(*_same(a, b)))
IMPLS["minimum"] = _binary(lambda a, b: (lambda a, b: torch.where((a < b) | (a != a), a, b))(*_same(a, b)))


_CREATION = {"tensor", "as_tensor", "full", "zeros", "ones", "empty", "arange", "linspace", "full_like",
             "zeros_like", "ones_like", "empty_like", "rand", "randn", "scalar_tensor"}


class exact_float32_casts(torch.overrides.TorchFunctionMode):
    """Run a float64 PyTorch model with its casts to float32 kept in float64.

    The reference interpreter treats ``cast`` to float32 as exact (see ``_cast``), so a model that
    computes part of its math in float32 on purpose (Hugging Face rotary embeddings and T5's layer
    norm call ``.float()``) differs from its own float64 copy by float32 rounding, about 1e-7, while
    the imported program is right. Inside this mode ``.float()``, ``.to(torch.float32)`` and
    ``.type(torch.float32)`` convert to float64 instead, so both sides compute the same exact math and
    the 1e-10 bar still applies. 16 bit casts are untouched (the reference rounds those). ``count``
    is the number of casts changed; callers report it. A ``dtype=torch.float32`` argument of any
    other function given a float64 tensor (``softmax(x, dim, dtype=torch.float32)`` in Gemma 2's eager
    attention) is kept float64 the same way. A tensor created with ``dtype=torch.float32`` (Gemma 3n's
    sparsity target, ``torch.tensor(0.95, dtype=torch.float32)``) is created in float32, so its value
    is rounded as in the model and as in the exported constant, and then converted to float64.
    """

    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        name = getattr(func, "__name__", "")
        if name in ("float", "to", "type") and args and isinstance(args[0], torch.Tensor):
            if name == "float":
                self.count += 1
                return args[0].to(torch.float64)
            if any(a is torch.float32 for a in args[1:]) or kwargs.get("dtype") is torch.float32:
                self.count += 1
                args = tuple(torch.float64 if a is torch.float32 else a for a in args)
                if kwargs.get("dtype") is torch.float32:
                    kwargs["dtype"] = torch.float64
        elif kwargs.get("dtype") is torch.float32 and name in _CREATION:
            # created as the model creates it (a Python 0.95 rounds to float32), then carried in
            # float64, so the values are the export's constants and the math after them is exact
            self.count += 1
            return func(*args, **kwargs).to(torch.float64)
        elif kwargs.get("dtype") is torch.float32 and any(
                isinstance(x, torch.Tensor) and x.dtype == torch.float64 for x in args):
            self.count += 1
            kwargs["dtype"] = torch.float64
        return func(*args, **kwargs)


@impl("cast")
def _cast(attrs, args, shape):
    x, dtype = args[0], attrs["dtype"]
    if attrs.get("profile") and dtype.is_floating_point:
        # A rounding inserted by emulate_precision(profile=...): round to the simulated device's
        # storage dtype under its rules (truestar.devsim), the value keeps the declared dtype.
        from ..devsim import get_profile
        from ..devsim.kernels import rnd

        p = get_profile(attrs["profile"])
        store = attrs.get("storage") or p.storage or dtype
        if torch.finfo(store).bits > torch.finfo(dtype).bits:
            store = dtype
        return rnd(x.to(COMPUTE_DTYPE), store, p)
    if attrs.get("toward_zero") and dtype.is_floating_point and torch.finfo(dtype).bits < 32:
        # A rounding inserted by emulate_precision(rounding="toward_zero"): the 16 bit neighbour on
        # the zero side (at most one unit in the last place from round to nearest). Comparing it with
        # the round to nearest run shows how far one ulp of every 16 bit value moves each output.
        r = x.to(dtype)
        fin = torch.isfinite(r)
        back = torch.nextafter(r, torch.zeros_like(r))
        r = torch.where(fin & (r.to(x.dtype).abs() > x.abs()), back, r)
        return r.to(COMPUTE_DTYPE)
    if attrs.get("range_only") and dtype.is_floating_point:
        # A range check inserted by emulate_precision(range_only=True): the value stays exact but
        # overflows where the declared dtype overflows. Round to nearest sends |x| >= max + ulp(max) / 2
        # to infinity; NaN passes through.
        fi = torch.finfo(dtype)
        thr = fi.max + fi.max / (2.0 - fi.eps) * fi.eps / 2.0
        x = x.to(COMPUTE_DTYPE)
        return torch.where(x.abs() >= thr, torch.where(x > 0, math.inf, -math.inf).to(x.dtype), x)
    if dtype == torch.bool:
        return x != 0
    if dtype.is_floating_point:
        # Every float is computed in float64 here, but a cast to a 16 bit or 8 bit float really loses
        # precision and can overflow, so it rounds through the target dtype (mixed precision training
        # and its loss scaling depend on that). float32 and wider stay exact, matching how models are
        # compared against their float64 PyTorch copies.
        if torch.finfo(dtype).bits < 32:
            return x.to(dtype).to(COMPUTE_DTYPE)
        return x.to(COMPUTE_DTYPE)
    if x.dtype.is_floating_point:
        x = torch.trunc(x)
    return x.to(dtype)


@impl("cumsum")
def _cumsum(attrs, args, shape):
    """out[k] = out[k - 1] + x[k] along dim."""
    x = args[0]
    if not x.dtype.is_floating_point:
        x = x.to(torch.int64)
    d = attrs["dim"] % x.dim()
    if x.shape[d] == 0:
        return x.clone()
    parts = []
    running = torch.zeros_like(x.select(d, 0))
    for k in range(x.shape[d]):
        running = running + x.select(d, k)
        parts.append(running)
    return torch.stack(parts, dim=d)


def _tri_keep(x, diagonal, upper):
    rows = _iota(x.shape[-2], torch.int64).reshape(-1, 1)
    cols = _iota(x.shape[-1], torch.int64).reshape(1, -1)
    return (cols - rows >= diagonal) if upper else (cols - rows <= diagonal)


@impl("triu")
def _triu(attrs, args, shape):
    """Keep x[..., i, j] where j - i >= diagonal, else 0."""
    x = args[0]
    return torch.where(_tri_keep(x, attrs.get("diagonal", 0), True), x, torch.zeros_like(x))


@impl("tril")
def _tril(attrs, args, shape):
    """Keep x[..., i, j] where j - i <= diagonal, else 0."""
    x = args[0]
    return torch.where(_tri_keep(x, attrs.get("diagonal", 0), False), x, torch.zeros_like(x))


@impl("vector_norm")
def _vector_norm(attrs, args, shape):
    """(sum |x|^p)^(1/p); p = inf is max |x|, p = -inf is min |x|, p = 0 counts nonzeros."""
    x = args[0]
    dims = _dims(attrs, x)
    keep = attrs.get("keepdim", False)
    p = float(attrs.get("ord", 2.0))
    a = torch.where(x < 0, -x, x)
    if p == math.inf:
        return a.amax(dim=dims, keepdim=keep)
    if p == -math.inf:
        return a.amin(dim=dims, keepdim=keep)
    if p == 0:
        return (a != 0).to(x.dtype).sum(dim=dims, keepdim=keep)
    if p == 2:
        return torch.sqrt((x * x).sum(dim=dims, keepdim=keep))
    return (a ** p).sum(dim=dims, keepdim=keep) ** (1.0 / p)


@impl("group_norm")
def _group_norm(attrs, args, shape):
    """Per sample and group of C / G channels: (x - mean) / sqrt(biased variance + eps), then
    weight and bias per channel."""
    x, w, b = args
    n, c = x.shape[0], x.shape[1]
    g = attrs["num_groups"]
    xg = x.reshape(n, g, -1)
    count = xg.shape[-1]
    mu = xg.sum(dim=-1, keepdim=True) / count
    var = ((xg - mu) * (xg - mu)).sum(dim=-1, keepdim=True) / count
    y = ((xg - mu) / torch.sqrt(var + attrs["eps"])).reshape(x.shape)
    view = (1, c) + (1,) * (x.dim() - 2)
    if w is not None:
        y = y * w.reshape(view)
    if b is not None:
        y = y + b.reshape(view)
    return y


@impl("rms_norm")
def _rms_norm(attrs, args, shape):
    """y = x / sqrt(mean(x^2) + eps) * weight over the trailing dims."""
    x, w = args
    dims = list(range(x.dim() - len(attrs["normalized_shape"]), x.dim()))
    count = 1
    for d in dims:
        count *= x.shape[d]
    ms = (x * x).sum(dim=dims, keepdim=True) / count
    y = x / torch.sqrt(ms + attrs["eps"])
    return y if w is None else y * w


@impl("avg_pool2d")
def _avg_pool2d(attrs, args, shape):
    """Sum over the kernel taps of a zero padded input, divided per output cell by the window size:
    the window clipped to the padded input when count_include_pad, else clipped to the real input."""
    x = args[0]
    kh, kw = _pair(attrs["kernel_size"])
    sh, sw = _pair(attrs.get("stride") or attrs["kernel_size"])
    ph, pw = _pair(attrs.get("padding", 0))
    h, w = x.shape[-2:]
    oh, ow = shape[-2:]
    lead = tuple(x.shape[:-2])
    H = max(h + 2 * ph, (oh - 1) * sh + kh)
    W = max(w + 2 * pw, (ow - 1) * sw + kw)
    xp = torch.zeros(lead + (H, W), dtype=x.dtype)
    xp[..., ph:ph + h, pw:pw + w] = x
    total = torch.zeros(lead + (oh, ow), dtype=x.dtype)
    for u in range(kh):
        for v in range(kw):
            total = total + xp[..., u:u + sh * (oh - 1) + 1:sh, v:v + sw * (ow - 1) + 1:sw]
    div = torch.zeros((oh, ow), dtype=x.dtype)
    for i in range(oh):
        for j in range(ow):
            h0, w0 = i * sh - ph, j * sw - pw
            h1, w1 = min(h0 + kh, h + ph), min(w0 + kw, w + pw)
            size = (h1 - h0) * (w1 - w0)
            if not attrs.get("count_include_pad", True):
                size = (min(h1, h) - max(h0, 0)) * (min(w1, w) - max(w0, 0))
            div[i, j] = attrs["divisor_override"] if attrs.get("divisor_override") else size
    return total / div


@impl("adaptive_max_pool2d")
def _adaptive_max_pool2d(attrs, args, shape):
    """Output cell i takes the max of input rows floor(i*H/out) to ceil((i+1)*H/out), same for columns."""
    x = args[0]
    h, w = x.shape[-2:]
    oh, ow = shape[-2:]
    rows = []
    for i in range(oh):
        r0, r1 = (i * h) // oh, -(-((i + 1) * h) // oh)
        cols = []
        for j in range(ow):
            c0, c1 = (j * w) // ow, -(-((j + 1) * w) // ow)
            cols.append(x[..., r0:r1, c0:c1].amax(dim=(-2, -1)))
        rows.append(torch.stack(cols, dim=-1))
    return torch.stack(rows, dim=-2)


@impl("pad")
def _pad(attrs, args, shape):
    """A tensor of ``value`` with x copied in; negative pads crop."""
    x = args[0]
    pads = list(attrs["pads"])
    out = torch.full(shape, attrs.get("value", 0.0), dtype=x.dtype)
    src = [slice(None)] * x.dim()
    dst = [slice(None)] * x.dim()
    for i in range(len(pads) // 2):
        d = x.dim() - 1 - i
        lo, hi = pads[2 * i], pads[2 * i + 1]
        s0, s1 = max(0, -lo), x.shape[d] - max(0, -hi)
        src[d] = slice(s0, s1)
        dst[d] = slice(max(0, lo), max(0, lo) + max(0, s1 - s0))
    out[tuple(dst)] = x[tuple(src)]
    return out


def _check_bounds(op, index, size):
    if index.numel() and (int(index.min()) < -size or int(index.max()) >= size):
        raise IRError(f"{op}: index out of range for a dim of size {size} "
                      f"(min {int(index.min())}, max {int(index.max())})")


@impl("index")
def _index(attrs, args, shape):
    x, idx = args[0], list(args[1:])
    for d, i in enumerate(idx):
        if i is not None:
            _check_bounds("index", i, x.shape[d])
    return x[tuple(slice(None) if i is None else i.to(torch.int64) for i in idx)]


@impl("index_select")
def _index_select(attrs, args, shape):
    x, i = args
    d = attrs["dim"] % x.dim()
    _check_bounds("index_select", i, x.shape[d])
    return x[(slice(None),) * d + (i.reshape(-1).to(torch.int64),)]


@impl("gather")
def _gather(attrs, args, shape):
    """out[p] = x[p with coordinate dim replaced by index[p]]."""
    x, i = args
    d = attrs["dim"] % x.dim()
    _check_bounds("gather", i, x.shape[d])
    coords = []
    for j in range(x.dim()):
        if j == d:
            coords.append(i.to(torch.int64))
        else:
            view = [1] * x.dim()
            view[j] = i.shape[j]
            coords.append(_iota(i.shape[j], torch.int64).reshape(view).expand(i.shape))
    return x[tuple(coords)]


@impl("slice_scatter")
def _slice_scatter(attrs, args, shape):
    x, src = args
    out = x.clone()
    d = attrs["dim"] % x.dim()
    idx = [slice(None)] * x.dim()
    idx[d] = slice(attrs.get("start"), attrs.get("end"), attrs.get("step", 1))
    out[tuple(idx)] = src.to(out.dtype)
    return out


@impl("select_scatter")
def _select_scatter(attrs, args, shape):
    x, src = args
    out = x.clone()
    d = attrs["dim"] % x.dim()
    idx = [slice(None)] * x.dim()
    idx[d] = attrs["index"]
    out[tuple(idx)] = src.to(out.dtype)
    return out


# ----------------------------------------------------------------------------- ops added for decoder LLMs
# (2026-11-24, see docs/LLM.md). Same policy: written from the definition with comparisons, sums and
# plain indexing. No torch.flip, torch.sort, torch.topk, torch.argmax, torch.index_copy or
# torch.multinomial.


def _reverse_index(n: int) -> torch.Tensor:
    return n - 1 - _iota(n, torch.int64)


@impl("flip")
def _flip(attrs, args, shape):
    """out[..., i, ...] = x[..., n - 1 - i, ...] along every flipped dim."""
    x = args[0]
    if x.dim() == 0:
        return x.clone()
    for d in attrs["dims"]:
        d %= x.dim()
        idx = [slice(None)] * x.dim()
        idx[d] = _reverse_index(x.shape[d])
        x = x[tuple(idx)]
    return x


def _isnan(x):
    return x != x


def _less(u, v):
    """u < v with NaN as the largest value (PyTorch's sort order)."""
    if not u.dtype.is_floating_point:
        return u < v
    return (u < v) | (~_isnan(u) & _isnan(v))


def _equal(u, v):
    if not u.dtype.is_floating_point:
        return u == v
    return (u == v) | (_isnan(u) & _isnan(v))


def _sort_last(y: torch.Tensor, descending: bool):
    """Stable sort of the last dim by rank counting: the rank of entry i is the number of entries
    that go before it (strictly smaller, or larger when descending) plus the number of equal entries
    at an earlier position. Returns (values, indices)."""
    n = y.shape[-1]
    pos = _iota(n, torch.int64)
    ranks = []
    block = 512
    for s in range(0, n, block):
        yi = y[..., s:s + block].unsqueeze(-1)       # [..., b, 1]  entry i
        yj = y.unsqueeze(-2)                         # [..., 1, n]  entry j
        before = _less(yi, yj) if descending else _less(yj, yi)
        tie = _equal(yj, yi) & (pos.reshape(1, n) < pos[s:s + block].reshape(-1, 1))
        ranks.append((before | tie).to(torch.int64).sum(dim=-1))
    rank = torch.cat(ranks, dim=-1) if ranks else torch.zeros(y.shape, dtype=torch.int64)
    # indices[r] = the i whose rank is r
    hit = rank.unsqueeze(-1) == pos.reshape(*([1] * rank.dim()), n)   # [..., i, r]
    indices = (hit.to(torch.int64) * pos.reshape(-1, 1)).sum(dim=-2)
    values = IMPLS["gather"]({"dim": -1}, [y, indices], tuple(indices.shape))
    return values, indices


@impl("sort")
def _sort(attrs, args, shapes):
    x = args[0]
    if x.dim() == 0:
        return [x.clone(), torch.zeros((), dtype=torch.int64)]
    d = attrs.get("dim", -1) % x.dim()
    v, i = _sort_last(x.movedim(d, -1), bool(attrs.get("descending", False)))
    return [v.movedim(-1, d).contiguous(), i.movedim(-1, d).contiguous()]


@impl("topk")
def _topk(attrs, args, shapes):
    """The first k entries of a stable sort, descending for the largest, ascending for the smallest."""
    x = args[0]
    k = attrs["k"]
    if x.dim() == 0:
        return [x.clone(), torch.zeros((), dtype=torch.int64)]
    d = attrs.get("dim", -1) % x.dim()
    v, i = _sort_last(x.movedim(d, -1), bool(attrs.get("largest", True)))
    return [v[..., :k].movedim(-1, d).contiguous(), i[..., :k].movedim(-1, d).contiguous()]


def _arg_extreme(attrs, x, largest: bool):
    """Index of the first maximum (or minimum) along dim; a NaN counts as the extreme, as in PyTorch."""
    dim = attrs.get("dim")
    keep = bool(attrs.get("keepdim", False))
    rank = x.dim()
    y = x.reshape(-1) if dim is None else (x if rank == 0 else x.movedim(dim % rank, -1))
    if y.dim() == 0:
        y = y.reshape(1)
    n = y.shape[-1]
    if n == 0:
        raise IRError("argmax/argmin over an empty dim")
    ext = y.amax(dim=-1, keepdim=True) if largest else y.amin(dim=-1, keepdim=True)
    target = y == ext
    if y.dtype.is_floating_point:
        nan = _isnan(y)
        target = torch.where(nan.any(dim=-1, keepdim=True), nan, target)
    pos = _iota(n, torch.int64).expand(y.shape)
    out = torch.where(target, pos, torch.full_like(pos, n)).amin(dim=-1)
    if dim is None:
        return out.reshape((1,) * rank) if keep else out.reshape(())
    if rank == 0:
        return out.reshape(())
    return out.unsqueeze(dim % rank) if keep else out


IMPLS["argmax"] = lambda attrs, args, shape: _arg_extreme(attrs, args[0], True)
IMPLS["argmin"] = lambda attrs, args, shape: _arg_extreme(attrs, args[0], False)


@impl("index_copy")
def _index_copy(attrs, args, shape):
    """out = x; out.select(dim, index[j]) = src.select(dim, j). Repeated indices would make the result
    depend on write order, so they are an error."""
    x, i, src = args
    d = attrs["dim"] % x.dim()
    _check_bounds("index_copy", i, x.shape[d])
    idx = i.to(torch.int64) % max(x.shape[d], 1)
    if len(set(idx.tolist())) != idx.numel():
        raise IRError(f"index_copy: repeated indices {idx.tolist()}; the result would depend on write order")
    out = x.clone()
    out[(slice(None),) * d + (idx,)] = src.to(out.dtype)
    return out


_MASK64 = (1 << 64) - 1


def splitmix64(z: int) -> int:
    """One SplitMix64 step (Steele, Lea and Flood 2014) on a 64 bit unsigned integer."""
    z = (z + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def uniform01(seed: int, offset: int, counter: int) -> float:
    """The uniform draw in [0, 1) of the IR's sampling ops: 53 bits of
    splitmix64(splitmix64(splitmix64(seed) ^ offset) ^ counter). Pure integer arithmetic, so every
    backend can reproduce it exactly."""
    h = splitmix64(splitmix64(splitmix64(seed & _MASK64) ^ (offset & _MASK64)) ^ (counter & _MASK64))
    return (h >> 11) * (2.0 ** -53)


@impl("multinomial")
def _multinomial(attrs, args, shape):
    """Inverse CDF sampling: with u the uniform draw for (seed, offset, row * num_samples + s), the
    sample is the first category whose running weight sum exceeds u times the total weight."""
    p = args[0]
    offset = int(args[1].item()) if len(args) > 1 and args[1] is not None else 0
    rows = p.reshape(1, -1) if p.dim() == 1 else p
    n = attrs.get("num_samples", 1)
    out = torch.zeros((rows.shape[0], n), dtype=torch.int64)
    for r in range(rows.shape[0]):
        w = rows[r].tolist()
        if any(not math.isfinite(v) or v < 0 for v in w):
            raise IRError(f"multinomial: row {r} has a negative, infinite or NaN weight")
        cum, run = [], 0.0
        for v in w:
            run += v
            cum.append(run)
        total = run
        if total <= 0:
            raise IRError(f"multinomial: row {r} has no positive weight")
        last = max(k for k, v in enumerate(w) if v > 0)
        for s in range(n):
            target = uniform01(attrs["seed"], offset, r * n + s) * total
            k = sum(1 for c in cum if c <= target)
            out[r, s] = min(k, last)
    return out.reshape(shape)


def _extreme_over(attrs, x, largest: bool):
    """The running max (or min) over each reduced dim in turn; a NaN anywhere in a window wins."""
    keep = attrs.get("keepdim", False)
    out = x
    for d in sorted(_dims(attrs, x), reverse=True):
        if out.dim() == 0:
            break
        acc = out.select(d, 0)
        for k in range(1, out.shape[d]):
            v = out.select(d, k)
            better = (v > acc) if largest else (v < acc)
            if v.dtype.is_floating_point:
                better = better | (v != v)
                better = better & (acc == acc)
            acc = torch.where(better, v, acc)
        out = acc.unsqueeze(d) if keep else acc
    return out


IMPLS["amax"] = lambda attrs, args, shape: _extreme_over(attrs, args[0], True)
IMPLS["amin"] = lambda attrs, args, shape: _extreme_over(attrs, args[0], False)


@impl("random")
def _random(attrs, args, shape):
    """The keyed generator through its torch implementation (truestar.rng), on the device the
    interpreter runs on (its torch path gives the same bits on the CPU and on CUDA). Floating draws
    are exact in their declared dtype, so holding them in float64 loses nothing."""
    from .. import rng

    key = int(args[0].item())
    streams = [int(a.item()) for a in args[1:]]
    out = rng.generate(attrs["dist"], shape, attrs["dtype"], key, attrs["site"], streams,
                       p=attrs.get("p"), low=attrs.get("low", 0), high=attrs.get("high"),
                       device=args[0].device, backend="torch")
    return out.to(COMPUTE_DTYPE) if out.dtype.is_floating_point else out


@impl("scatter_add")
def _scatter_add(attrs, args, shape):
    """out = x; out[p with coordinate dim replaced by index[p]] += src[p] for every p.

    Written with indexed assignment, never torch.scatter_add. When no row repeats an index along dim
    (the gradients of sort and topk) every entry receives at most one addition and one assignment
    does it. Otherwise the positions along dim are added one at a time, in increasing order, so
    repeated indices sum in a fixed order."""
    x, i, src = args
    d = attrs["dim"] % max(x.dim(), 1)
    out = x.to(COMPUTE_DTYPE).clone() if x.dtype.is_floating_point else x.clone()
    if x.dim() == 0:
        return out + src
    _check_bounds("scatter_add", i, x.shape[d])
    i = i.to(torch.int64) % x.shape[d]
    src = src.to(out.dtype)
    grids = []
    for j in range(x.dim()):
        view = [1] * x.dim()
        view[j] = i.shape[j]
        grids.append(_iota(i.shape[j], torch.int64).reshape(view).expand(i.shape))
    srt = i.sort(dim=d).values if i.shape[d] > 1 else i
    repeats = i.shape[d] > 1 and bool((srt.narrow(d, 1, i.shape[d] - 1) == srt.narrow(d, 0, i.shape[d] - 1)).any())
    if not repeats:
        coords = tuple(i if j == d else grids[j] for j in range(x.dim()))
        out[coords] = out[coords] + src
        return out
    for k in range(i.shape[d]):
        coords = tuple((i if j == d else grids[j]).select(d, k) for j in range(x.dim()))
        out[coords] = out[coords] + src.select(d, k)
    return out


# ----------------------------------------------------------------------------- convolution and pooling
# gradients, written from the forward formulas (each kernel tap is a strided slice), not F.conv2d.


def _padded_extent(size: int, out: int, s: int, d: int, k: int, p: int) -> int:
    return max(size + 2 * p, (out - 1) * s + d * (k - 1) + 1)


@impl("conv2d_input_grad")
def _conv2d_input_grad(attrs, args, shape):
    """x_bar = the adjoint of conv2d in x: each tap (u, v) sends g[n, o, i, j] * w[o, c, u, v] to
    padded position (i*s + u*d, j*s + v*d); the padding is dropped at the end."""
    g, w = args
    sh, sw = _pair(attrs.get("stride", 1))
    ph, pw = _pair(attrs.get("padding", 0))
    dh, dw = _pair(attrs.get("dilation", 1))
    groups = attrs.get("groups", 1)
    n, c, h, wd = shape
    o, cg, kh, kw = w.shape
    oh, ow = g.shape[2:]
    H, W = _padded_extent(h, oh, sh, dh, kh, ph), _padded_extent(wd, ow, sw, dw, kw, pw)
    xp = torch.zeros((n, groups, cg, H, W), dtype=g.dtype)
    gg = g.reshape(n, groups, o // groups, oh, ow)
    wg = w.reshape(groups, o // groups, cg, kh, kw)
    for u in range(kh):
        for v in range(kw):
            r0, c0 = u * dh, v * dw
            xp[:, :, :, r0:r0 + sh * (oh - 1) + 1:sh, c0:c0 + sw * (ow - 1) + 1:sw] += \
                torch.einsum("ngohw,goc->ngchw", gg, wg[:, :, :, u, v])
    return xp.reshape(n, c, H, W)[:, :, ph:ph + h, pw:pw + wd].contiguous()


@impl("conv2d_weight_grad")
def _conv2d_weight_grad(attrs, args, shape):
    """w_bar[o, c, u, v] = sum over n, i, j of xpad[n, g(o) Cg + c, i*s + u*d, j*s + v*d] g[n, o, i, j]."""
    x, g = args
    sh, sw = _pair(attrs.get("stride", 1))
    ph, pw = _pair(attrs.get("padding", 0))
    dh, dw = _pair(attrs.get("dilation", 1))
    groups = attrs.get("groups", 1)
    o, cg, kh, kw = shape
    n, c, h, wd = x.shape
    oh, ow = g.shape[2:]
    H, W = _padded_extent(h, oh, sh, dh, kh, ph), _padded_extent(wd, ow, sw, dw, kw, pw)
    xp = torch.zeros((n, c, H, W), dtype=x.dtype)
    xp[:, :, ph:ph + h, pw:pw + wd] = x
    xg = xp.reshape(n, groups, cg, H, W)
    gg = g.reshape(n, groups, o // groups, oh, ow)
    out = torch.zeros((groups, o // groups, cg, kh, kw), dtype=x.dtype)
    for u in range(kh):
        for v in range(kw):
            r0, c0 = u * dh, v * dw
            patch = xg[:, :, :, r0:r0 + sh * (oh - 1) + 1:sh, c0:c0 + sw * (ow - 1) + 1:sw]
            out[:, :, :, u, v] = torch.einsum("ngchw,ngohw->goc", patch, gg)
    return out.reshape(o, cg, kh, kw)


@impl("max_pool2d_grad")
def _max_pool2d_grad(attrs, args, shape):
    """Each window sends its cotangent to its first maximal tap in row major order (PyTorch's tie
    rule), the taps walked as strided slices of the input padded with minus infinity."""
    x, g = args
    kh, kw = _pair(attrs["kernel_size"])
    sh, sw = _pair(attrs.get("stride") or attrs["kernel_size"])
    ph, pw = _pair(attrs.get("padding", 0))
    dh, dw = _pair(attrs.get("dilation", 1))
    n, c, h, w = x.shape
    oh, ow = g.shape[2:]
    y = IMPLS["max_pool2d"](attrs, [x], (n, c, oh, ow))
    H, W = _padded_extent(h, oh, sh, dh, kh, ph), _padded_extent(w, ow, sw, dw, kw, pw)
    xp = torch.full((n, c, H, W), -math.inf, dtype=x.dtype)
    xp[:, :, ph:ph + h, pw:pw + w] = x
    out = torch.zeros((n, c, H, W), dtype=g.dtype)
    taken = torch.zeros((n, c, oh, ow), dtype=torch.bool)
    for u in range(kh):
        for v in range(kw):
            r0, c0 = u * dh, v * dw
            rows, cols = slice(r0, r0 + sh * (oh - 1) + 1, sh), slice(c0, c0 + sw * (ow - 1) + 1, sw)
            first = (xp[:, :, rows, cols] == y) & ~taken
            taken = taken | first
            out[:, :, rows, cols] += torch.where(first, g, torch.zeros_like(g))
    return out[:, :, ph:ph + h, pw:pw + w].contiguous()

@impl("mla_attention")
def _mla_attention(attrs, args, shape):
    """float64 absorbed MLA attention over the cache before pos[0] and the step's new latents
    (truestar.mla.latent.reference)."""
    from ..mla.latent import reference as mla_ref
    return mla_ref(*args, attrs["scale"]).to(args[0].dtype)


@impl("kvq_attention")
def _kvq_attention(attrs, args, shape):
    """float64 attention over the dequantized cache and the live tail after it, grouped query heads
    mapped h -> h // (Hq / Hkv): exactly what truestar.kvquant's kernel computes, unrounded."""
    q, kd, ks, vd, vs, tk, tv, tl = args[:8]
    pl = args[8] if len(args) > 8 else None      # live keys of a fixed capacity prefix, or None
    kb, vb = (args[9], args[10]) if len(args) > 10 and args[9] is not None else (None, None)
    fmt, group, vgroup, channel = attrs["fmt"], attrs["group"], attrs["vgroup"], attrs["channel"]

    def deq(d, s, g, per_channel, bs=None):
        if bs is not None:                       # int4 / nvfp4 (truestar.kvquant.low), back to x
            from ..kvquant import low
            x = low.dequantize(d.to(torch.uint8), bs.to(torch.uint8).view(torch.float8_e4m3fn), s, fmt, g)
            return low.unrotate(x) if attrs.get("rot") else x
        if fmt == "fp8":
            d = d.to(torch.uint8).view(torch.float8_e4m3fn)
        n = d.shape[2]
        s = s.to(torch.float64).repeat_interleave(g, dim=2)[:, :, :n]
        return d.to(torch.float64) * (s if per_channel else s[..., None])

    k = deq(kd, ks, group, channel, kb)
    v = deq(vd, vs, vgroup, False, vb)
    if pl is not None:
        p = int(pl.reshape(-1)[0])
        k, v = k[:, :, :p], v[:, :, :p]
    n = int(tl.reshape(-1)[0])
    k = torch.cat([k, tk[:, :, :n].to(torch.float64)], dim=2)
    v = torch.cat([v, tv[:, :, :n].to(torch.float64)], dim=2)
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    scale = attrs.get("scale") or 1.0 / math.sqrt(q.shape[-1])
    w = softmax(torch.matmul(q.to(torch.float64), k.transpose(-2, -1)) * scale, -1)
    return torch.matmul(w, v).to(q.dtype)
