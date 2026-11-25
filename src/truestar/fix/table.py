"""Built in remedy table: equivalent decompositions for ATen ops that a backend gets wrong.

Each entry rewrites one ATen op as other ops that compute the same math. A remedy only replaces the
op on the device where it was blamed, and it is only reported as working after Truestar's own check
is rerun with it applied (see truestar.fix.engine). The table is ordered: for a blamed op the
candidates are tried first to last and the first that makes the module and the model clean wins.

Signatures follow the ATen schema of the op they replace (positional names and keyword only
arguments), because a TorchDispatchMode or a torch.library kernel receives the call exactly as the
dispatcher saw it.

`calls_self` marks a decomposition that calls the op it replaces (with other arguments). It is
fine under a dispatch mode, which steps aside while it runs, but would recurse forever as a
torch.library kernel, so the library mechanism skips it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

# Elementwise multiply and sum decompositions materialize a product tensor; above this many
# elements they are skipped rather than risking an out of memory on a shared GPU.
MAX_PRODUCT_ELEMENTS = 1 << 26

_HALF = (torch.float16, torch.bfloat16)


@dataclass(frozen=True)
class Decomposition:
    op: str  # full overload name, e.g. "aten.addmm.default"
    name: str  # short id, unique per op, e.g. "mm_add"
    description: str
    fn: Callable = field(repr=False, compare=False)
    # Device types this entry is offered on; None means every device.
    devices: Optional[Tuple[str, ...]] = None
    calls_self: bool = False
    # Optional guard: (args, kwargs) -> reason string when it cannot handle this call, else None.
    guard: Optional[Callable] = field(default=None, repr=False, compare=False)
    note: str = ""

    def unsupported(self, args, kwargs) -> Optional[str]:
        if self.guard is None:
            return None
        try:
            return self.guard(*args, **kwargs)
        except TypeError as exc:
            return f"unexpected arguments: {exc}"

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


# ----------------------------------------------------------------------------- helpers


def _scale_add(out: torch.Tensor, input: torch.Tensor, beta, alpha) -> torch.Tensor:
    """beta * input + alpha * out, with addmm's rule that beta == 0 ignores input entirely."""
    if alpha != 1:
        out = out * alpha
    if beta == 0:
        return out
    return out + (input if beta == 1 else input * beta)


def _big(*shape_numel: int) -> Optional[str]:
    n = 1
    for s in shape_numel:
        n *= int(s)
    if n > MAX_PRODUCT_ELEMENTS:
        return f"product tensor of {n} elements is above the {MAX_PRODUCT_ELEMENTS} limit"
    return None


# ----------------------------------------------------------------------------- addmm


def _addmm_mm_add(input, mat1, mat2, *, beta=1, alpha=1):
    return _scale_add(torch.mm(mat1, mat2), input, beta, alpha)


def _addmm_bmm_add(input, mat1, mat2, *, beta=1, alpha=1):
    return _scale_add(torch.bmm(mat1.unsqueeze(0), mat2.unsqueeze(0)).squeeze(0), input, beta, alpha)


def _addmm_mulsum(input, mat1, mat2, *, beta=1, alpha=1):
    return _scale_add((mat1.unsqueeze(2) * mat2.unsqueeze(0)).sum(1), input, beta, alpha)


def _addmm_mulsum_guard(input, mat1, mat2, *, beta=1, alpha=1):
    return _big(mat1.shape[0], mat1.shape[1], mat2.shape[1])


# ----------------------------------------------------------------------------- linear


def _linear_matmul_add(input, weight, bias=None):
    y = torch.matmul(input, weight.t())
    return y if bias is None else y + bias


def _linear_addmm(input, weight, bias=None):
    x2 = input.reshape(-1, input.shape[-1])
    y = torch.mm(x2, weight.t()) if bias is None else torch.addmm(bias, x2, weight.t())
    return y.reshape(*input.shape[:-1], weight.shape[0])


def _linear_nobias_add(input, weight, bias=None):
    y = torch.ops.aten.linear.default(input, weight)
    return y if bias is None else y + bias


def _linear_mulsum(input, weight, bias=None):
    y = (input.unsqueeze(-2) * weight).sum(-1)
    return y if bias is None else y + bias


def _linear_mulsum_guard(input, weight, bias=None):
    return _big(input.numel() // max(input.shape[-1], 1), weight.shape[0], weight.shape[1])


# ----------------------------------------------------------------------------- mm, bmm


def _mm_bmm(self, mat2):
    return torch.bmm(self.unsqueeze(0), mat2.unsqueeze(0)).squeeze(0)


def _mm_mulsum(self, mat2):
    return (self.unsqueeze(2) * mat2.unsqueeze(0)).sum(1)


def _mm_mulsum_guard(self, mat2):
    return _big(self.shape[0], self.shape[1], mat2.shape[1])


def _bmm_loop(self, mat2):
    return torch.stack([torch.mm(self[i], mat2[i]) for i in range(self.shape[0])])


def _bmm_loop_guard(self, mat2):
    return None if self.shape[0] <= 4096 else f"batch {self.shape[0]} is above 4096 for a loop"


def _bmm_mulsum(self, mat2):
    return (self.unsqueeze(3) * mat2.unsqueeze(1)).sum(2)


def _bmm_mulsum_guard(self, mat2):
    return _big(self.shape[0], self.shape[1], self.shape[2], mat2.shape[2])


# ----------------------------------------------------------------------------- softmax family


def _softmax_explicit(self, dim, half_to_float):
    x = self.float() if half_to_float else self
    e = (x - x.amax(dim, keepdim=True)).exp()
    return e / e.sum(dim, keepdim=True)


def _softmax_fp32(self, dim, half_to_float):
    x = self.float()
    e = (x - x.amax(dim, keepdim=True)).exp()
    out = e / e.sum(dim, keepdim=True)
    return out if half_to_float else out.to(self.dtype)


def _only_half(self, *args, **kwargs):
    return None if self.dtype in _HALF else "input is not float16 or bfloat16"


def _log_softmax_explicit(self, dim, half_to_float):
    x = self.float() if half_to_float else self
    s = x - x.amax(dim, keepdim=True)
    return s - s.exp().sum(dim, keepdim=True).log()


def _log_softmax_fp32(self, dim, half_to_float):
    x = self.float()
    s = x - x.amax(dim, keepdim=True)
    out = s - s.exp().sum(dim, keepdim=True).log()
    return out if half_to_float else out.to(self.dtype)


# ----------------------------------------------------------------------------- layer norm, gelu


def _layer_norm_explicit(input, normalized_shape, weight, bias, eps):
    k = len(normalized_shape)
    dims = tuple(range(input.dim() - k, input.dim()))
    x = input.float() if input.dtype in _HALF else input
    mean = x.mean(dims, keepdim=True)
    var = ((x - mean) ** 2).mean(dims, keepdim=True)
    rstd = (var + eps).rsqrt()
    out = (x - mean) * rstd
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out.to(input.dtype), mean, rstd


def _gelu_explicit(self, *, approximate="none"):
    x = self.float() if self.dtype in _HALF else self
    if approximate == "tanh":
        y = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
    else:
        y = 0.5 * x * (1.0 + torch.erf(x * (1.0 / math.sqrt(2.0))))
    return y.to(self.dtype)


# ----------------------------------------------------------------------------- convolution


def _pair(v) -> Tuple[int, int]:
    v = list(v) if isinstance(v, (list, tuple)) else [v]
    return (int(v[0]), int(v[-1]))


def _conv_im2col(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    n, c, h, w = input.shape
    o, cg, kh, kw = weight.shape
    s, p, d = _pair(stride), _pair(padding), _pair(dilation)
    cols = F.unfold(input, (kh, kw), dilation=d, padding=p, stride=s)  # [N, C*kh*kw, L]
    ho = (h + 2 * p[0] - d[0] * (kh - 1) - 1) // s[0] + 1
    wo = (w + 2 * p[1] - d[1] * (kw - 1) - 1) // s[1] + 1
    g = int(groups)
    cols = cols.reshape(n, g, cg * kh * kw, ho * wo)
    wm = weight.reshape(g, o // g, cg * kh * kw)
    out = torch.matmul(wm.unsqueeze(0), cols).reshape(n, o, ho, wo)
    return out if bias is None else out + bias.reshape(1, -1, 1, 1)


def _conv_im2col_guard(input, weight, bias, stride, padding, dilation, transposed, output_padding,
                       groups):
    if transposed:
        return "transposed convolution is not covered"
    if input.dim() != 4:
        return "only 2D convolution (4D input) is covered"
    return None


# ----------------------------------------------------------------------------- the table

_ENTRIES: List[Decomposition] = [
    Decomposition("aten.addmm.default", "mm_add", "mm(mat1, mat2) then beta * input added",
                  _addmm_mm_add),
    Decomposition("aten.addmm.default", "bmm_add", "batched matmul of one batch, then the add",
                  _addmm_bmm_add),
    Decomposition("aten.addmm.default", "mul_sum", "broadcast multiply and sum (no matmul kernel)",
                  _addmm_mulsum, guard=_addmm_mulsum_guard),
    Decomposition("aten.linear.default", "matmul_add", "matmul(x, w.T) plus bias",
                  _linear_matmul_add,
                  note="MPS on macOS 26: x @ w.T + b was correct in 144 of 144 probe cases while "
                       "F.linear with bias was wrong (docs/upstream/mps_linear_bias_DRAFT.md)"),
    Decomposition("aten.linear.default", "addmm", "addmm(bias, x flattened to 2D, w.T)",
                  _linear_addmm),
    Decomposition("aten.linear.default", "linear_nobias_add", "linear without bias, then add bias",
                  _linear_nobias_add, calls_self=True),
    Decomposition("aten.linear.default", "mul_sum", "broadcast multiply and sum (no matmul kernel)",
                  _linear_mulsum, guard=_linear_mulsum_guard),
    Decomposition("aten.mm.default", "bmm", "batched matmul of one batch", _mm_bmm),
    Decomposition("aten.mm.default", "mul_sum", "broadcast multiply and sum", _mm_mulsum,
                  guard=_mm_mulsum_guard),
    Decomposition("aten.bmm.default", "mm_loop", "one mm per batch entry", _bmm_loop,
                  guard=_bmm_loop_guard),
    Decomposition("aten.bmm.default", "mul_sum", "broadcast multiply and sum", _bmm_mulsum,
                  guard=_bmm_mulsum_guard),
    Decomposition("aten._softmax.default", "explicit", "exp(x - max) / sum, the stable form",
                  _softmax_explicit),
    Decomposition("aten._softmax.default", "explicit_fp32", "stable form computed in float32",
                  _softmax_fp32, guard=_only_half),
    Decomposition("aten._log_softmax.default", "explicit", "x - max - log(sum(exp(x - max)))",
                  _log_softmax_explicit),
    Decomposition("aten._log_softmax.default", "explicit_fp32", "stable form computed in float32",
                  _log_softmax_fp32, guard=_only_half),
    Decomposition("aten.native_layer_norm.default", "explicit",
                  "mean, variance and rsqrt written out (float32 for half inputs)",
                  _layer_norm_explicit),
    Decomposition("aten.gelu.default", "explicit", "erf (or tanh) formula written out",
                  _gelu_explicit),
    Decomposition("aten.convolution.default", "im2col", "unfold then matmul (2D, not transposed)",
                  _conv_im2col, guard=_conv_im2col_guard),
]


def _check_unique(entries: Sequence[Decomposition]) -> None:
    seen = set()
    for e in entries:
        key = (e.op, e.name)
        if key in seen:
            raise ValueError(f"duplicate remedy {key}")
        seen.add(key)


_check_unique(_ENTRIES)

# Public, read only view: op -> ordered candidates.
TABLE: Dict[str, Tuple[Decomposition, ...]] = {}
for _e in _ENTRIES:
    TABLE[_e.op] = TABLE.get(_e.op, ()) + (_e,)


def candidates(op: str, device_type: str, mechanism: str = "mode",
               table: Optional[Dict[str, Sequence[Decomposition]]] = None) -> List[Decomposition]:
    """Ordered decompositions for `op` on `device_type` usable with `mechanism`."""
    entries = (TABLE if table is None else table).get(op, ())
    out = []
    for e in entries:
        if e.devices is not None and device_type not in e.devices:
            continue
        if mechanism == "library" and e.calls_self:
            continue
        out.append(e)
    return out


def lookup(op: str, name: str, table: Optional[Dict[str, Sequence[Decomposition]]] = None
           ) -> Decomposition:
    for e in (TABLE if table is None else table).get(op, ()):
        if e.name == name:
            return e
    raise KeyError(f"no remedy {name!r} for {op}")


def table_rows() -> List[Dict[str, str]]:
    """The table as plain rows, for docs and `truestar fix --list`."""
    return [{"op": e.op, "name": e.name, "description": e.description,
             "devices": "all" if e.devices is None else ",".join(e.devices),
             "library_ok": "no" if e.calls_self else "yes", "note": e.note} for e in _ENTRIES]
