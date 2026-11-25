"""Losses as IR builders.

A loss declares the target inputs it needs (``targets``) and emits its nodes onto an
:class:`~truestar.train.edit.Editor` (``emit``), returning a 0 dim loss value. Every loss is built
from IR ops, with ``log_softmax`` (one op, the stable ``x - max - log sum exp(x - max)`` form) for
the classification losses. Definitions follow PyTorch exactly, including how label smoothing and
``ignore_index`` interact with the mean:

* ``MSE``: ``mean((pred - target)^2)`` (or the sum), as ``F.mse_loss``.
* ``CrossEntropy``: logits ``[N, C]``, int64 class targets ``[N]``. Per row, ``nll = -log p[y]`` and
  ``smooth = -sum_c log p[c]``; rows whose target equals ``ignore_index`` count for nothing. With
  reduction "mean" the loss is ``(1 - eps) * sum(nll) / n + eps / C * sum(smooth) / n`` with ``n``
  the number of counted rows, which is what ``F.cross_entropy(..., label_smoothing=eps)`` computes.
* ``CausalLM``: logits ``[B, T, V]`` and int64 tokens ``[B, T]``; position ``t`` predicts token
  ``t + 1``: ``CrossEntropy`` of ``logits[:, :-1]`` against ``tokens[:, 1:]``, flattened.

Class targets are compared with an ``arange`` to build the one hot mask, so no gather op is needed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Sequence, Tuple

import sympy
import torch

from ..ir.core import IRError, Value, dims_equal

TargetSpec = Tuple[str, torch.dtype, Tuple]


class Loss:
    def targets(self, outputs: Sequence[Value]) -> List[TargetSpec]:
        raise NotImplementedError

    def emit(self, e, outputs: Sequence[Value], targets: Sequence[Value]) -> Value:
        raise NotImplementedError

    def config(self) -> dict:
        return {"loss": type(self).__name__, **asdict(self)}


def _check_reduction(r):
    if r not in ("mean", "sum"):
        raise IRError(f"reduction must be 'mean' or 'sum', got {r!r}")


@dataclass
class MSE(Loss):
    reduction: str = "mean"
    output: int = 0

    def __post_init__(self):
        _check_reduction(self.reduction)

    def targets(self, outputs):
        t = outputs[self.output].type
        return [("target", t.dtype, t.shape)]

    def emit(self, e, outputs, targets):
        pred, (y,) = outputs[self.output], targets
        d = e.op("sub", pred, y, name="mse_diff")
        sq = e.op("mul", d, d, name="mse_sq")
        return e.op(self.reduction, sq, dims=None, keepdim=False, name="mse_loss")


def cross_entropy(e, logits: Value, target: Value, label_smoothing: float = 0.0, reduction: str = "mean",
                  ignore_index: int = -100) -> Value:
    """Emit cross entropy of ``logits: [N, C]`` against int64 ``target: [N]``. Returns a 0 dim value."""
    _check_reduction(reduction)
    if logits.type.rank != 2:
        raise IRError(f"cross_entropy takes logits [N, C], got {logits.type}; reshape first")
    if target.type.rank != 1 or target.type.dtype not in (torch.int64, torch.int32):
        raise IRError(f"cross_entropy takes integer class targets [N], got {target.type}")
    n, c = logits.type.shape
    if not isinstance(c, int):
        raise IRError(f"cross_entropy needs a fixed class count, got {logits.type}")
    if not dims_equal(n, target.type.shape[0]):
        raise IRError(f"cross_entropy: logits {logits.type} and target {target.type} disagree on N")
    if not 0.0 <= label_smoothing <= 1.0:
        raise IRError(f"label_smoothing must be in [0, 1], got {label_smoothing}")
    dt = logits.type.dtype
    lsm = e.op("log_softmax", logits, dim=1, name="ce_log_softmax")
    classes = e.op("arange", start=0, end=c, step=1, dtype=target.type.dtype, name="ce_classes")
    col = e.op("reshape", target, shape=[n, 1], name="ce_target_col")
    onehot = e.op("eq", classes, col, name="ce_onehot")  # bool [N, C]
    zero, one = e.const(0.0, dt), e.const(1.0, dt)
    picked = e.op("where", onehot, lsm, zero, name="ce_picked")
    nll = e.op("neg", e.op("sum", picked, dims=[1], keepdim=False), name="ce_nll")  # [N]; 0 on ignored rows
    valid = e.op("ne", target, e.const(ignore_index, target.type.dtype), name="ce_valid")
    terms = [(1.0 - label_smoothing, nll)]
    if label_smoothing > 0.0:
        smooth = e.op("neg", e.op("sum", lsm, dims=[1], keepdim=False), name="ce_smooth")
        smooth = e.op("where", valid, smooth, zero, name="ce_smooth_valid")
        terms.append((label_smoothing / c, smooth))
    count = None
    if reduction == "mean":
        count = e.op("sum", e.op("where", valid, one, zero), dims=None, keepdim=False, name="ce_count")
    total = None
    for w, t in terms:
        s = e.op("sum", t, dims=None, keepdim=False)
        if count is not None:
            s = e.op("div", s, count)
        if w != 1.0:
            s = e.op("mul", s, e.const(w, dt))
        total = s if total is None else e.op("add", total, s)
    return total


@dataclass
class CrossEntropy(Loss):
    label_smoothing: float = 0.0
    reduction: str = "mean"
    ignore_index: int = -100
    output: int = 0

    def __post_init__(self):
        _check_reduction(self.reduction)

    def targets(self, outputs):
        t = outputs[self.output].type
        if t.rank != 2:
            raise IRError(f"CrossEntropy takes model output [N, C], got {t}")
        return [("target", torch.int64, (t.shape[0],))]

    def emit(self, e, outputs, targets):
        return cross_entropy(e, outputs[self.output], targets[0], self.label_smoothing, self.reduction,
                             self.ignore_index)


@dataclass
class CausalLM(Loss):
    """Next token prediction: logits ``[B, T, V]``, tokens ``[B, T]``."""

    label_smoothing: float = 0.0
    ignore_index: int = -100
    output: int = 0

    def targets(self, outputs):
        t = outputs[self.output].type
        if t.rank != 3:
            raise IRError(f"CausalLM takes logits [B, T, V], got {t}")
        return [("tokens", torch.int64, t.shape[:2])]

    def emit(self, e, outputs, targets):
        logits, tokens = outputs[self.output], targets[0]
        b, t, v = logits.type.shape
        if not isinstance(t, int) or t < 2:
            raise IRError(f"CausalLM needs a fixed sequence length of at least 2, got {logits.type}")
        pred = e.op("slice", logits, dim=1, start=0, end=t - 1, step=1, name="lm_pred")
        nxt = e.op("slice", tokens, dim=1, start=1, end=t, step=1, name="lm_next")
        rows = sympy.sympify(b) * (t - 1)
        pred = e.op("reshape", pred, shape=[rows, v], name="lm_pred_rows")
        nxt = e.op("reshape", nxt, shape=[rows], name="lm_next_rows")
        return cross_entropy(e, pred, nxt, self.label_smoothing, "mean", self.ignore_index)


def attach_loss(prog, loss: Loss, cast_outputs_to=None, name=None):
    """``prog`` extended with the loss's target inputs and a scalar loss as its only output.

    ``cast_outputs_to``: cast float model outputs to this dtype before the loss (mixed precision runs
    the loss in float32, as torch's autocast does for ``mse_loss`` and ``cross_entropy``).
    Returns the program and the names of the target inputs."""
    from .edit import Editor

    e = Editor(prog, name or f"loss_{prog.name}")
    e.source = f"truestar.train loss {type(loss).__name__}"
    outs = list(prog.outputs)
    if cast_outputs_to is not None:
        outs = [e.op("cast", o, dtype=cast_outputs_to, name=f"{o.name}_loss_dtype")
                if o.type.dtype.is_floating_point and o.type.dtype != cast_outputs_to else o for o in outs]
    tvals = [e.input(n, dt, shape) for n, dt, shape in loss.targets(outs)]
    value = loss.emit(e, outs, tvals)
    if value.type.rank != 0:
        raise IRError(f"internal error: {type(loss).__name__} produced {value.type}, not a scalar")
    return e.build([value]), [v.name for v in tvals]
