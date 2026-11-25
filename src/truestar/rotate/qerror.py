"""Quantization error of a model with and without rotations, from float64 references only.

``format_model(model64, spec)`` replaces every linear layer (except the lm head) by the format's
exact float64 promise (truestar.quant: weights dequantized from their codes, activations rounded
by the format's rule for the W4A4 and W8A8 formats). No kernel runs, so this is verdict (b) of
docs/QUANT.md: what the format loses on this model. Run it on the original model and on a rotated
model that ``fold.prove`` accepted, and the difference is what the rotation buys.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Iterable, List, Sequence

import torch
from torch import nn

from ..quant import formats as QF
from ..quant.model import RefQuantLinear

F64 = torch.float64


def linear_names(model: nn.Module, skip: Iterable[str] = ("lm_head",)) -> List[str]:
    skip = tuple(skip)
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)
            and not any(n == s or n.startswith(s + ".") for s in skip)]


def _set(model: nn.Module, name: str, mod: nn.Module):
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, leaf, mod)


@torch.no_grad()
def format_model(model64: nn.Module, spec: str, skip: Sequence[str] = ("lm_head",)) -> nn.Module:
    """A copy of a float64 model whose linear layers compute the format's exact answer."""
    q = copy.deepcopy(model64)
    swap_in(q, spec, skip)
    return q


# The settings QuaRot and SpinQuant report: symmetric round to nearest integers, one scale per
# weight output channel and one per activation token (no groups). Not truestar.quant formats (no
# kernel exists for them here); reference only, to compare with the published setting.
RTN = {"w4a4_int_rtn": (4, 4), "w8a8_int_rtn": (8, 8), "w4a16_int_rtn": (4, 0)}

# Where a rotation measurably helps (docs/ROTATION.md, SmolLM2-360M): per token integer
# activations, W4A4 perplexity ratio 1041 to 12.3 and W8A8 KL 8x lower. Every block and float
# format (int4:g64, nvfp4, mxfp4, fp8_e4m3:channel) lost 1.9x to 5.9x more KL with every rotation
# and weight only integers 1.6x to 2.4x, so a rotated model is refused there (TS1024).
ROTATION_FORMATS = ("w4a4_int_rtn", "w8a8_int_rtn")


def check_rotation_allowed(model64: nn.Module, spec: str) -> None:
    """Raise TS1024 when model64 came from ``fold.apply`` and spec is a format the rotation makes
    worse; an unrotated model takes every format."""
    from .hadamard import RotationRefusedError
    if getattr(model64, "_truestar_rotated", False) and spec not in ROTATION_FORMATS:
        raise RotationRefusedError(
            f"[TS1024] a rotated model is refused for {spec!r}: on SmolLM2-360M every rotation made "
            f"the block and float formats (int4:g64, nvfp4, mxfp4, fp8_e4m3:channel) lose 1.9x to 5.9x "
            f"more KL than no rotation, weight only integers 1.6x to 2.4x (docs/ROTATION.md). Rotations are "
            f"offered for per token integer activation formats only ({', '.join(ROTATION_FORMATS)}); "
            f"quantize the unrotated model for {spec!r}.")


def _rtn(x: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(-1, keepdim=True) / qmax
    s = torch.where(s > 0, s, torch.ones_like(s))
    return torch.clamp(torch.round(x / s), -qmax, qmax) * s


class RefRTNLinear(nn.Module):
    def __init__(self, lin: nn.Linear, wbits: int, abits: int):
        super().__init__()
        self.register_buffer("w", _rtn(lin.weight.detach().to(F64), wbits))
        self.register_buffer("b", None if lin.bias is None else lin.bias.detach().to(F64).clone())
        self.abits = abits

    def forward(self, x):
        x = x.to(F64)
        if self.abits:
            x = _rtn(x, self.abits)
        y = x @ self.w.T
        return y if self.b is None else y + self.b


@torch.no_grad()
def swap_in(model64: nn.Module, spec: str, skip: Sequence[str] = ("lm_head",)) -> list:
    """In place: replace the linear layers of model64 by the format's float64 reference; returns
    what ``swap_out`` needs to put the float layers back (no copy of the model is made). A rotated
    model takes only ``ROTATION_FORMATS`` (TS1024)."""
    check_rotation_allowed(model64, spec)
    undo = []
    for n in linear_names(model64, skip):
        lin = model64.get_submodule(n)
        if spec in RTN:
            ref = RefRTNLinear(lin, *RTN[spec])
        else:
            qw = QF.quantize_weight(lin.weight.detach().to(F64), spec)
            ref = RefQuantLinear(qw, lin.bias)
            qw.data = qw.scale = qw.tensor_scale = qw._deq_cache = None    # keep only the float64 weight
        _set(model64, n, ref)
        undo.append((n, lin))
    return undo


def swap_out(model64: nn.Module, undo: list) -> None:
    for n, lin in undo:
        _set(model64, n, lin)


@torch.no_grad()
def logits(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids, use_cache=False).logits.to(F64)


def ce(lg: torch.Tensor, ids: torch.Tensor) -> float:
    lp = torch.log_softmax(lg[:, :-1], -1)
    return float(-lp.gather(-1, ids[:, 1:, None]).mean())


@torch.no_grad()
def layer_inputs(model: nn.Module, ids: torch.Tensor, names: Sequence[str]) -> Dict[str, tuple]:
    caps = {}
    hooks = [model.get_submodule(n).register_forward_hook(
        lambda mod, inp, out, n=n: caps.__setitem__(n, (inp[0], out))) for n in names]
    try:
        model(input_ids=ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return caps


@torch.no_grad()
def evaluate(model64: nn.Module, ids: torch.Tensor, specs: Sequence[str],
             skip: Sequence[str] = ("lm_head",), ref_logits: torch.Tensor = None,
             layer_ids: torch.Tensor = None) -> Dict:
    """Per format: cross entropy of the format model, its change against the float64 model
    ``ref_logits`` (the unrotated original's, so every row compares to the same model), KL, top 1
    agreement, and the per layer output change on each layer's own float64 inputs (on
    ``layer_ids``, default ``ids``: the captured inputs are large); plus the activation outlier
    ratio (max |x| / rms) of every quantized layer's input. A rotated model takes only
    ``ROTATION_FORMATS`` (TS1024, checked for every spec before anything runs)."""
    for spec in specs:
        check_rotation_allowed(model64, spec)
    names = linear_names(model64, skip)
    L = logits(model64, ids) if ref_logits is None else ref_logits
    p = torch.log_softmax(L[:, :-1], -1)
    caps = layer_inputs(model64, ids if layer_ids is None else layer_ids, names)
    out = {"ce_float": ce(L, ids), "positions": int(ids[:, 1:].numel()),
           "act_max_over_rms": float(sum(float(x.abs().max() / x.pow(2).mean().sqrt())
                                         for x, _ in caps.values()) / len(caps)),
           "formats": {}}
    for spec in specs:
        undo = swap_in(model64, spec, skip)
        qm = model64
        Lq = logits(qm, ids)
        pq = torch.log_softmax(Lq[:, :-1], -1)
        lay = {}
        for n, (x, y) in caps.items():
            yq = qm.get_submodule(n)(x)
            lay[n] = float((yq - y).norm() / y.norm().clamp_min(1e-300))
        vals = sorted(lay.values())
        out["formats"][spec] = {
            "ce": ce(Lq, ids), "delta_ce": ce(Lq, ids) - out["ce_float"],
            "ppl_ratio": math.exp(ce(Lq, ids) - out["ce_float"]),
            "kl": float((p.exp() * (p - pq)).sum(-1).mean()),
            "top1_agree": float((pq.argmax(-1) == p.argmax(-1)).double().mean()),
            "layer_change_median": vals[len(vals) // 2], "layer_change_max": vals[-1],
            "layers": lay}
        swap_out(model64, undo)
        del qm, undo
    return out
