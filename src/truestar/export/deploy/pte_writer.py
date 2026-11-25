"""Truestar IR to ExecuTorch (.pte).

ExecuTorch programs are written by ExecuTorch's own serializer from a torch.export graph, so the
verified IR program is first turned back into a module (``IRModule``: one torch op per IR node,
weights as buffers, float32 as declared), exported with torch.export, lowered to the Edge dialect
and serialized for the portable kernels. (An XNNPACK delegated program was tried and not shipped:
the XNNPACK partitioner needs torchao, and no torchao build matching torch 2.11 imported cleanly
here, so a delegated file could not be verified.)

The executorch package is an optional extra; without it the target is refused with TS1419.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...ir import reference
from ...ir.core import Node, eval_dim
from .lower import Lowered, LoweringError, pair

T = "executorch"


def _missing(exc: Exception):
    from ...deps import MissingDependencyError

    return MissingDependencyError("executorch", "Writing a .pte file (truestar export --to executorch) "
                                  f"[import failed: {type(exc).__name__}: {exc}]", extra="executorch")


def _attention(q, k, v, mask, a):
    scale = a.get("scale")
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale)
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    if a.get("is_causal"):
        keep = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).tril()
        s = s.masked_fill(~keep, float("-inf"))
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else s + mask
    return torch.matmul(torch.softmax(s, dim=-1), v)


def _op(n: Node, args: List[Optional[torch.Tensor]], dims=None) -> torch.Tensor:
    a, op = n.attrs, n.op
    x = args[0] if args else None
    simple = {"relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh, "exp": torch.exp,
              "neg": torch.neg, "abs": torch.abs, "sqrt": torch.sqrt, "rsqrt": torch.rsqrt, "sin": torch.sin,
              "cos": torch.cos, "log": torch.log, "erf": torch.erf, "logical_not": torch.logical_not,
              "silu": F.silu, "reciprocal": torch.reciprocal}
    binary = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div, "maximum": torch.maximum,
              "minimum": torch.minimum, "eq": torch.eq, "lt": torch.lt, "gt": torch.gt, "le": torch.le,
              "ge": torch.ge}
    if op in simple:
        return simple[op](x)
    if op in binary:
        return binary[op](args[0], args[1])
    if op == "gelu":
        return F.gelu(x, approximate=a.get("approximate", "none"))
    if op == "linear":
        return F.linear(x, args[1], args[2])
    if op == "matmul":
        return torch.matmul(x, args[1])
    if op == "conv2d":
        return F.conv2d(x, args[1], args[2], stride=pair(a.get("stride", 1)), padding=pair(a.get("padding", 0)),
                        dilation=pair(a.get("dilation", 1)), groups=int(a.get("groups", 1)))
    if op == "batch_norm":
        return F.batch_norm(x, args[3], args[4], args[1], args[2], False, 0.0, float(a["eps"]))
    if op == "layer_norm":
        return F.layer_norm(x, [int(d) for d in a["normalized_shape"]], args[1], args[2], float(a["eps"]))
    if op == "max_pool2d":
        return F.max_pool2d(x, pair(a["kernel_size"]), pair(a.get("stride") or a["kernel_size"]),
                            pair(a.get("padding", 0)), pair(a.get("dilation", 1)), bool(a.get("ceil_mode", False)))
    if op == "adaptive_avg_pool2d":
        return F.adaptive_avg_pool2d(x, tuple(int(d) for d in n.output.type.shape[-2:]))
    if op == "softmax":
        return torch.softmax(x, dim=int(a["dim"]))
    if op == "reshape":
        return x.reshape([eval_dim(d, dims or {}) for d in n.output.type.shape])
    if op == "permute":
        return x.permute(*[int(d) for d in a["dims"]])
    if op == "select":
        return x.select(int(a["dim"]), int(a["index"]))
    if op == "slice":
        idx = [slice(None)] * x.dim()
        idx[int(a["dim"]) % x.dim()] = slice(a.get("start"), a.get("end"), a.get("step", 1))
        return x[tuple(idx)]
    if op == "where":
        return torch.where(args[0], args[1], args[2])
    if op == "index" and len(args) == 2:
        return F.embedding(args[1], x)
    if op == "mean":
        return x.mean(dim=a.get("dim"), keepdim=bool(a.get("keepdim", False)))
    if op == "attention":
        return _attention(args[0], args[1], args[2], args[3], a)
    if op == "cat":
        return torch.cat(args, dim=int(a["dim"]))
    raise LoweringError(T, n, f"op {op} has no ExecuTorch lowering")


class IRModule(nn.Module):
    """A torch module that computes the lowered program node by node. It is what ExecuTorch
    serializes, and what the per element tolerance is derived on: every node is a plain op with
    an error rule, where the user's module may call a fused kernel that has none."""

    def __init__(self, low: Lowered):
        super().__init__()
        self.low = low
        self.cnames: Dict[str, str] = {}
        for i, (k, t) in enumerate(low.consts.items()):
            bn = f"c{i}"
            self.register_buffer(bn, t.clone())
            self.cnames[k] = bn

    def forward(self, *xs):
        env = {v.name: x for v, x in zip(self.low.inputs, xs)}
        dims = {} if self.low.static() else reference.bind_symbols(self.low, list(xs))
        for k, bn in self.cnames.items():
            env[k] = getattr(self, bn)
        for n in self.low.nodes:
            env[n.output.name] = _op(n, [None if i is None else env[i.name] for i in n.inputs], dims)
        return tuple(env[v.name] for v in self.low.outputs)


def write(low: Lowered, example: List[torch.Tensor]) -> bytes:
    if not low.static():
        raise LoweringError(T, None, "ExecuTorch files are written with static shapes (export with --static-batch)")
    try:
        from executorch.exir import to_edge
    except Exception as exc:
        raise _missing(exc) from None
    mod = IRModule(low).eval()
    for n in low.nodes:  # refuse unknown ops before tracing, with the node named
        if n.op not in _KNOWN:
            raise LoweringError(T, n, f"op {n.op} has no ExecuTorch lowering")
    with torch.no_grad():
        ep = torch.export.export(mod, tuple(example))
    prog = to_edge(ep).to_executorch()
    low.meta["instructions"] = _instructions(prog.executorch_program.execution_plan[0])
    return bytes(prog.buffer)


def _instructions(plan) -> List[Dict[str, str]]:
    """Kernel and delegate calls of the serialized plan, read from the program object (reading them
    back from the bytes needs the flatc binary, which the Windows wheel does not ship)."""
    out = []
    for c in plan.chains:
        for ins in c.instructions:
            a = ins.instr_args
            kind = type(a).__name__
            if kind == "KernelCall":
                op = plan.operators[a.op_index]
                out.append({"kind": "kernel", "op": op.name + (f".{op.overload}" if getattr(op, "overload", "") else "")})
            elif kind == "DelegateCall":
                out.append({"kind": "delegate", "backend": plan.delegates[a.delegate_index].id})
    return out


_KNOWN = {"relu", "sigmoid", "tanh", "exp", "neg", "abs", "sqrt", "rsqrt", "sin", "cos", "log", "erf",
          "logical_not", "silu", "reciprocal", "add", "sub", "mul", "div", "maximum", "minimum", "eq", "lt",
          "gt", "le", "ge", "gelu", "linear", "matmul", "conv2d", "batch_norm", "layer_norm", "max_pool2d",
          "adaptive_avg_pool2d", "softmax", "reshape", "permute", "select", "slice", "where", "index", "mean",
          "attention", "cat"}
