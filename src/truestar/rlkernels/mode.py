"""`rl_mode()`: run a plain PyTorch model on the RL kernel set.

Inside the context every call of the functions below goes to the batch and position invariant
kernels of `kernels.py`, in the training forward, the prefill and every decode step alike:

| PyTorch call | RL kernel |
|---|---|
| `F.linear`, `nn.Linear` | `linear`, weight contraction in `SEGMENTS` K segments added as a tree |
| `torch.matmul`, `@`, `mm`, `bmm` | `matmul` (a 2 dim right operand is split like a weight) |
| `F.scaled_dot_product_attention` (bool mask, causal, GQA) | `attention` |
| `F.rms_norm`, `nn.RMSNorm`, `F.layer_norm`, `nn.LayerNorm` | `rms_norm`, `layer_norm` |
| `softmax`, `log_softmax` (any spelling) | anchored row kernels |
| `sum`, `mean` over one dim or all dims | fixed chunk row sum |

Elementwise ops, indexing, `cat`, embedding lookups and order free reductions (`max`, `min`,
`argmax`, `any`, integer sums) run unchanged: their result per element does not depend on the
batch. Floating point reductions that have no invariant kernel (`cumsum`, `var`, `std`, `norm`,
`logsumexp`, `einsum`, `addmm`, convolutions, `cross_entropy`, ...) raise `RLModeError` naming
the call: there is no quiet fallback to a batch dependent kernel. `strict=False` lets them run
eagerly and counts them in `mode.eager_calls` instead.

Gradients: each routed call is an autograd function whose backward recomputes the eager PyTorch
op on the saved inputs, so the trainer can train on the RL kernel forward. Gradients are not
bitwise invariant and are not claimed to be.

`overrides={"attention": "eager"}` (op names as in `KERNEL_OPS`) runs one op family on the
eager PyTorch kernel instead, which is how the tests inject a swapped kernel.
`trace=True` records, per call, the op family, the kernel that ran and the innermost module
(when a `KernelTrace` is active), which `truestar rlpair --kernels rl` uses to name the op whose
kernel differs between the trainer and the sampler.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from . import kernels as K

KERNEL_OPS = ("linear", "matmul", "attention", "rms_norm", "layer_norm", "softmax", "log_softmax",
              "sum", "mean")


class RLModeError(RuntimeError):
    """A call inside `rl_mode()` that the RL kernel set cannot run invariantly."""


def _f(name: str):
    return getattr(F, name, None)


def _t(name: str):
    return getattr(torch, name, None)


def _m(name: str):
    return getattr(torch.Tensor, name, None)


_FUNCS: Dict[Any, str] = {}


def _reg(op: str, *fns) -> None:
    for fn in fns:
        if fn is not None:
            _FUNCS[fn] = op


_reg("linear", F.linear, getattr(torch._C._nn, "linear", None))
_reg("matmul", torch.matmul, _m("matmul"), _m("__matmul__"), torch.mm, _m("mm"), torch.bmm, _m("bmm"))
_reg("attention", F.scaled_dot_product_attention)
_reg("rms_norm", _f("rms_norm"), _t("rms_norm"))
_reg("layer_norm", F.layer_norm, _t("layer_norm"))
_reg("softmax", F.softmax, torch.softmax, _m("softmax"), getattr(torch.special, "softmax", None))
_reg("log_softmax", F.log_softmax, torch.log_softmax, _m("log_softmax"),
     getattr(torch.special, "log_softmax", None))
_reg("sum", torch.sum, _m("sum"))
_reg("mean", torch.mean, _m("mean"))

# Floating point reductions or contractions with no invariant kernel here.
_REFUSED: Dict[Any, str] = {}
for _name in ("cumsum", "cumprod", "var", "std", "var_mean", "std_mean", "logsumexp", "norm", "prod",
              "nansum", "nanmean", "einsum", "addmm", "addbmm", "baddbmm", "addmv", "mv", "dot",
              "vdot", "inner", "tensordot", "trace", "logcumsumexp", "cdist", "conv1d", "conv2d",
              "conv3d", "conv_transpose1d", "conv_transpose2d", "group_norm", "batch_norm",
              "instance_norm", "native_layer_norm", "_scaled_dot_product_flash_attention",
              "_scaled_dot_product_efficient_attention"):
    for fn in (_t(_name), _m(_name)):
        if fn is not None:
            _REFUSED[fn] = _name
for _name in ("cross_entropy", "nll_loss", "kl_div", "mse_loss", "l1_loss", "normalize", "group_norm",
              "batch_norm", "instance_norm", "conv1d", "conv2d", "conv3d", "bilinear", "embedding_bag",
              "cosine_similarity", "pairwise_distance", "multi_head_attention_forward"):
    fn = _f(_name)
    if fn is not None:
        _REFUSED[fn] = _name
for _name in ("vector_norm", "matrix_norm", "norm", "matmul", "vecdot"):
    fn = getattr(torch.linalg, _name, None)
    if fn is not None:
        _REFUSED[fn] = f"linalg.{_name}"


# ----------------------------------------------------------------------------- autograd


class _Routed(torch.autograd.Function):
    """Forward: the RL kernel. Backward: the eager op recomputed on the saved inputs."""

    @staticmethod
    def forward(ctx, fast: Callable, ref: Callable, n: int, *tensors):
        ctx.ref = ref
        ctx.n = n
        ctx.save_for_backward(*tensors)
        return fast(*tensors)

    @staticmethod
    def backward(ctx, grad):
        saved = ctx.saved_tensors
        needs = ctx.needs_input_grad[3:]
        with torch._C.DisableTorchFunction(), torch.enable_grad():
            ins = [t.detach().requires_grad_(bool(n)) for t, n in zip(saved, needs)]
            out = ctx.ref(*ins)
            need = [t for t in ins if t.requires_grad]
            got = torch.autograd.grad(out, need, grad, allow_unused=True) if need else ()
        it = iter(got)
        grads = [next(it) if t.requires_grad else None for t in ins]
        return (None, None, None, *grads)


def _run(fast: Callable, ref: Callable, *tensors):
    """Run `fast(*tensors)`; with autograd recording, attach the eager backward."""
    if torch.is_grad_enabled() and any(isinstance(t, torch.Tensor) and t.requires_grad for t in tensors):
        return _Routed.apply(fast, ref, len(tensors), *tensors)
    return fast(*tensors)


# ----------------------------------------------------------------------------- module stack


_tls = threading.local()


def _stack() -> List[str]:
    if not hasattr(_tls, "stack"):
        _tls.stack = []
    return _tls.stack


class KernelTrace:
    """Tag the calls a tracing `rl_mode` records with the innermost running module and its call
    index (`name#k`, the keys `truestar rlpair` uses)."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handles = []
        self.counts: Counter = Counter()

    def __enter__(self):
        for name, mod in self.model.named_modules():
            label = name or "<model>"

            def pre(m, a, label=label):
                k = self.counts[label]
                self.counts[label] += 1
                _stack().append(f"{label}#{k}")

            def post(m, a, o):
                if _stack():
                    _stack().pop()

            self.handles.append(mod.register_forward_pre_hook(pre))
            self.handles.append(mod.register_forward_hook(post))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        _stack().clear()


# ----------------------------------------------------------------------------- the mode


def _dim_arg(args, kwargs, pos: int):
    if "dim" in kwargs:
        return kwargs["dim"]
    return args[pos] if len(args) > pos else None


class rl_mode(TorchFunctionMode):
    """Route a model's reductions and contractions to the RL kernel set (see module docstring)."""

    def __init__(self, strict: bool = True, overrides: Optional[Dict[str, str]] = None,
                 trace: bool = False, segments: Optional[int] = None):
        super().__init__()
        bad = set(overrides or {}) - set(KERNEL_OPS)
        if bad:
            raise ValueError(f"unknown op names in overrides: {sorted(bad)}; known {KERNEL_OPS}")
        for v in (overrides or {}).values():
            if v not in ("eager", "rl"):
                raise ValueError(f"override values are 'eager' or 'rl', got {v!r}")
        self.strict = strict
        self.overrides = dict(overrides or {})
        self.segments = segments
        self.trace_on = trace
        self.trace: List[Tuple[Optional[str], str, str]] = []  # (module key, op, kernel)
        self.eager_calls: Counter = Counter()
        self.routed_calls: Counter = Counter()

    # -- helpers
    def _log(self, op: str, kernel: str) -> None:
        if kernel == "rl":
            self.routed_calls[op] += 1
        if self.trace_on:
            st = _stack()
            self.trace.append((st[-1] if st else None, op, kernel))

    def _refuse(self, what: str, why: str):
        raise RLModeError(f"rl_mode: {what} cannot run on the RL kernel set ({why}). Rewrite it "
                          "with the supported ops, or pass strict=False to run it eagerly "
                          "(the result then may depend on the batch).")

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        op = _FUNCS.get(func)
        if op is None:
            name = _REFUSED.get(func)
            if name is not None and _any_float(args, kwargs):
                if self.strict:
                    self._refuse(name, "a floating point reduction with no invariant kernel")
                self.eager_calls[name] += 1
                self._log(name, "eager")
            return func(*args, **kwargs)
        if self.overrides.get(op) == "eager":
            self._log(op, "eager")
            return func(*args, **kwargs)
        try:
            handler = getattr(self, f"_h_{op}")
            out = handler(func, args, kwargs)
        except _PassThrough as p:
            if p.refuse and self.strict:
                self._refuse(f"{op} ({p.why})", p.why)
            if p.refuse:
                self.eager_calls[op] += 1
                self._log(op, "eager")
            return func(*args, **kwargs)
        self._log(op, "rl")
        return out

    # -- handlers. Each raises _PassThrough for inputs the plain call handles exactly.
    def _h_linear(self, func, args, kwargs):
        x, w = args[0], args[1] if len(args) > 1 else kwargs["weight"]
        b = args[2] if len(args) > 2 else kwargs.get("bias")
        _check_float(x, w, "linear")
        seg = self.segments
        return _run(lambda x_, w_, *b_: K.linear(x_, w_, b_[0] if b_ else None, segments=seg),
                    lambda x_, w_, *b_: F.linear(x_, w_, b_[0] if b_ else None),
                    x, w, *([b] if b is not None else []))

    def _h_matmul(self, func, args, kwargs):
        a, b = args[0], args[1] if len(args) > 1 else kwargs.get("other", kwargs.get("mat2"))
        if not isinstance(b, torch.Tensor):
            raise _PassThrough("not a tensor operand", refuse=False)
        _check_float(a, b, "matmul")
        if a.dim() < 2 or b.dim() < 2:
            raise _PassThrough("a one dim operand (a vector product)", refuse=True)
        seg = self.segments if b.dim() == 2 else 1
        return _run(lambda a_, b_: K.matmul(a_, b_, segments=seg), torch.matmul, a, b)

    def _h_attention(self, func, args, kwargs):
        names = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")
        p = dict(zip(names, args))
        p.update(kwargs)
        q, k, v = p["query"], p["key"], p["value"]
        _check_float(q, k, "scaled_dot_product_attention")
        if p.get("dropout_p", 0.0):
            raise _PassThrough("dropout", refuse=True)
        mask = p.get("attn_mask")
        if mask is not None and mask.dtype != torch.bool:
            raise _PassThrough("an additive float mask; pass a bool mask", refuse=True)
        if q.dim() != 4:
            raise _PassThrough("inputs that are not [batch, heads, seq, dim]", refuse=True)
        causal, scale, gqa = bool(p.get("is_causal", False)), p.get("scale"), bool(p.get("enable_gqa", False))
        if v.shape[-1] != q.shape[-1]:
            raise _PassThrough("a value head dim other than the query's", refuse=True)
        tensors = [q, k, v] + ([mask] if mask is not None else [])

        def fast(q_, k_, v_, *m_):
            return K.attention(q_, k_, v_, m_[0] if m_ else None, causal, scale, gqa)

        def ref(q_, k_, v_, *m_):
            return F.scaled_dot_product_attention(q_, k_, v_, attn_mask=m_[0] if m_ else None,
                                                  is_causal=causal, scale=scale, enable_gqa=gqa)

        return _run(fast, ref, *tensors)

    def _h_rms_norm(self, func, args, kwargs):
        names = ("input", "normalized_shape", "weight", "eps")
        p = dict(zip(names, args))
        p.update(kwargs)
        x, shape, w, eps = p["input"], p["normalized_shape"], p.get("weight"), p.get("eps")
        _check_float(x, None, "rms_norm")
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        ts = [x] + ([w] if w is not None else [])
        return _run(lambda x_, *w_: K.rms_norm(x_, shape, w_[0] if w_ else None, eps),
                    lambda x_, *w_: torch.rms_norm(x_, shape, w_[0] if w_ else None, eps), *ts)

    def _h_layer_norm(self, func, args, kwargs):
        names = ("input", "normalized_shape", "weight", "bias", "eps")
        p = dict(zip(names, args))
        p.update(kwargs)
        x, shape = p["input"], p["normalized_shape"]
        w, b, eps = p.get("weight"), p.get("bias"), p.get("eps", 1e-5)
        _check_float(x, None, "layer_norm")
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        hw, hb = w is not None, b is not None
        ts = [x] + ([w] if hw else []) + ([b] if hb else [])

        def split(rest):
            rest = list(rest)
            return (rest.pop(0) if hw else None), (rest.pop(0) if hb else None)

        return _run(lambda x_, *r: K.layer_norm(x_, shape, *split(r), eps),
                    lambda x_, *r: F.layer_norm(x_, shape, *split(r), eps), *ts)

    def _softmaxish(self, args, kwargs, log: bool):
        x = args[0] if args else kwargs["input"]
        dim = _dim_arg(args, kwargs, 1)
        dtype = kwargs.get("dtype", args[2] if len(args) > 2 and isinstance(args[2], torch.dtype) else None)
        if dim is None:
            raise _PassThrough("softmax without dim", refuse=True)
        _check_float(x, None, "softmax")
        fn = torch.log_softmax if log else torch.softmax
        kfn = K.log_softmax if log else K.softmax
        return _run(lambda x_: kfn(x_, dim, dtype), lambda x_: fn(x_, dim, dtype=dtype), x)

    def _h_softmax(self, func, args, kwargs):
        return self._softmaxish(args, kwargs, False)

    def _h_log_softmax(self, func, args, kwargs):
        return self._softmaxish(args, kwargs, True)

    def _reduce(self, args, kwargs, mean: bool):
        x = args[0] if args else kwargs["input"]
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise _PassThrough("integer or bool sum", refuse=False)
        dim = _dim_arg(args, kwargs, 1)
        keepdim = kwargs.get("keepdim", args[2] if len(args) > 2 and isinstance(args[2], bool) else False)
        dtype = kwargs.get("dtype")
        if isinstance(dim, torch.dtype):  # sum(x, dtype)
            dtype, dim = dim, None
        _check_float(x, None, "mean" if mean else "sum")
        if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
            if keepdim:
                raise _PassThrough("a full reduction with keepdim", refuse=True)

            def fast(x_):
                return K.sum_last(x_.reshape(1, -1), mean=mean, dtype=dtype).reshape(())
        else:
            dims = dim if isinstance(dim, (list, tuple)) else (dim,)
            if len(dims) != 1:
                raise _PassThrough("a reduction over several dims", refuse=True)
            d = dims[0] % x.dim()

            def fast(x_):
                moved = x_.movedim(d, -1) if d != x_.dim() - 1 else x_
                out = K.sum_last(moved, keepdim=False, mean=mean, dtype=dtype)
                return out.unsqueeze(d) if keepdim else out

        ref_fn = torch.mean if mean else torch.sum

        def ref(x_):
            if dim is None:
                return ref_fn(x_, dtype=dtype) if dtype else ref_fn(x_)
            return ref_fn(x_, dim, keepdim=keepdim, dtype=dtype)

        return _run(fast, ref, x)

    def _h_sum(self, func, args, kwargs):
        return self._reduce(args, kwargs, False)

    def _h_mean(self, func, args, kwargs):
        return self._reduce(args, kwargs, True)


class _PassThrough(Exception):
    def __init__(self, why: str, refuse: bool):
        super().__init__(why)
        self.why = why
        self.refuse = refuse


def _any_float(args, kwargs) -> bool:
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, torch.Tensor) and a.is_floating_point():
            return True
        if isinstance(a, (list, tuple)) and any(isinstance(t, torch.Tensor) and t.is_floating_point() for t in a):
            return True
    return False


def _check_float(a, b, what: str) -> None:
    for t in (a, b):
        if t is None:
            continue
        if not t.is_floating_point():
            raise _PassThrough("integer operands are exact", refuse=False)
        if t.device.type != "cuda":
            raise RLModeError(f"rl_mode: {what} got a {t.device.type} tensor; the RL kernel set "
                              "runs on CUDA only (no CPU path, no fallback)")
        if t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise _PassThrough(f"dtype {t.dtype}", refuse=True)
