"""``torch.ops.truestar.linear_attention``: the recurrence as one PyTorch op (pain row P60).

The importer's lowerings of Hugging Face Mamba 2 SSD and Gated DeltaNet layers
(``truestar.ir.hf_linattn``) call this op in place of the layer's own chunked PyTorch code, so
torch.export records the recurrence as one node. The importer turns that node into the IR op
``linear_attention`` (float64 reference: the sequential recurrence; compiled: the kernels of this
package), see ``truestar.ir.linattn_ops``.

Eager calls run ``linear_attention`` (Triton on CUDA, the chunked form elsewhere, checked against
the float64 recurrence on the first call per signature). The gradient is the autograd gradient of
the chunked form, as for ``linear_attention`` itself. Shapes as in reference.py: q, k [B, T, H, dk],
v [B, T, H, dv], g [B, T, H] or [B, T, H, dk] or None, beta [B, T, H] or None (the delta rule needs
it). ``scale`` multiplies every output.
"""

from __future__ import annotations

from typing import Optional

import torch

from .chunked import chunked
from .op import linear_attention as _linear_attention
from .triton_kernels import DET_CHUNK


@torch.library.custom_op("truestar::linear_attention", mutates_args=())
def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: Optional[torch.Tensor],
                     beta: Optional[torch.Tensor], rule: str, scale: float) -> torch.Tensor:
    """o [B, T, H, dv] of the add or delta rule recurrence, in v's dtype."""
    return _linear_attention(q, k, v, g, beta, rule=rule, scale=scale).clone()


@linear_attention.register_fake
def _(q, k, v, g, beta, rule, scale):
    return torch.empty_like(v)


def _setup(ctx, inputs, output):
    q, k, v, g, beta, rule, scale = inputs
    ctx.save_for_backward(q, k, v, g, beta)
    ctx.rule, ctx.scale = rule, scale


def _backward(ctx, do):
    ins = list(ctx.saved_tensors)
    need = [x is not None and ctx.needs_input_grad[i] for i, x in enumerate(ins)]
    with torch.enable_grad():
        leaves = [None if x is None else x.detach().requires_grad_(n) for x, n in zip(ins, need)]
        o, _ = chunked(*leaves, rule=ctx.rule, chunk=DET_CHUNK, scale=ctx.scale)
        wanted = [x for x, n in zip(leaves, need) if n]
        got = iter(torch.autograd.grad(o, wanted, do.to(o.dtype), allow_unused=True) if wanted else [])
    res = []
    for x, n in zip(ins, need):
        gx = next(got) if n else None
        res.append(None if not n else (torch.zeros_like(x) if gx is None else gx.to(x.dtype)))
    return tuple(res) + (None, None)


linear_attention.register_autograd(_backward, setup_context=_setup)
