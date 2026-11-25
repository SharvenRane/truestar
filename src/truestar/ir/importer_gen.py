"""ATen ops of the image generation, detection and segmentation models (docs/MODEL_COVERAGE.md,
section "Image generation, detection and segmentation"; models in ``truestar.testing_vision``).

Measured first: ``scripts/op_coverage.py --sources gen`` listed the ops these models produce that the
importer did not take.

* ``concat`` and ``concatenate``: other names of ``cat`` (SDXL's added time embedding joins the
  pooled text embedding and the time ids with ``torch.concat``).
* ``div`` with ``rounding_mode`` (``"floor"`` or ``"trunc"``) and ``floor_divide``: DETR's sine
  position embedding builds its frequency table with ``torch.div(arange, 2, rounding_mode="floor")``.
  A value computed from constants only is computed at import with ``torch.div`` itself on the
  folded float64 (or integer) data, so the program holds exactly torch's quotient. Of a value that
  depends on an input it is refused (TS860): torch's float floor division is Python's ``//``, which
  differs from ``floor(a / b)`` when the quotient rounds up to an integer (``1 // 0.1`` is 9, while
  ``floor(1 / 0.1)`` is 10), and no composite of the IR's ops gives the exact remainder it needs.
  ``rounding_mode=None`` is the plain division.
"""

from __future__ import annotations

import torch

from .importer import HANDLERS, aten
from .importer_current import constant_value
from .importer_extra import _t, _unsupported


@aten("aten.concat.default", "aten.concatenate.default")
def _concat(c, a):
    return HANDLERS["aten.cat.default"](c, a)


def _declared(c, base: str, data: torch.Tensor, dtype: torch.dtype):
    """A folded constant stored in its declared dtype (``data`` rounded once to it). ``_t`` keeps float
    data in float64, which an exported artifact refuses for a value of another dtype unless it is a
    scalar (ir/serialize.py), so a float32 model with a folded table could not be exported."""
    v = _t(c, base, data, dtype)
    c.param_data[v.name] = data.to(dtype)
    return v


def _operand_value(c, v):
    """The folded data of a tensor operand, the Python number itself for a scalar, None when the
    value depends on an input."""
    if isinstance(v, (int, float, bool)):
        return v
    return constant_value(c, c.tensor(v))


def _div_mode(c, a):
    mode = a.get("rounding_mode")
    if mode is None:
        return HANDLERS["aten.div.Tensor"](c, a)
    if mode not in ("floor", "trunc"):
        _unsupported(c, f"div with rounding_mode {mode!r}")
    fake = c.node.meta.get("val")
    if fake is None:
        _unsupported(c, f"div with rounding_mode {mode!r}: the exported node has no dtype")
    x, y = _operand_value(c, a["self"]), _operand_value(c, a["other"])
    if (x is None or y is None) and not fake.dtype.is_floating_point:
        return _int_div_mode(c, a, mode, fake.dtype)
    if x is None or y is None:
        _unsupported(c, f"div with rounding_mode {mode!r} of a value that depends on an input (only a value "
                        "computed from constants is supported: it is computed at import with torch.div; torch's "
                        "floor division is Python's //, which floor(a / b) does not give exactly)")
    out = torch.div(x, y, rounding_mode=mode)
    out = out if torch.is_tensor(out) else torch.tensor(out)
    return _declared(c, f"div_{mode}", out, fake.dtype)


def _int_div_mode(c, a, mode, dtype):
    """Integer ``torch.div(x, y, rounding_mode=...)`` of a value that depends on an input (per image
    sizes in Pixtral and SigLIP2): through float64, where the quotient of integers below 2^52 rounds
    to the right side of every integer, so floor and truncation are exact; cast back."""
    from .importer_current import floor_value
    from .importer_extra import _op

    f64 = torch.float64
    x = c.tensor(a["self"])
    y = c.operand(a["other"], x)
    q = _op(c, "div", _op(c, "cast", x, dtype=f64), _op(c, "cast", y, dtype=f64))
    if mode == "floor":
        r = floor_value(c, q)
    else:   # toward zero: -floor(-q) below zero, floor(q) from zero on
        neg = _op(c, "neg", floor_value(c, _op(c, "neg", q)))
        r = _op(c, "where", _op(c, "lt", q, c.const(0.0, q)), neg, floor_value(c, q))
    return _op(c, "cast", r, dtype=dtype)


@aten("aten.div.Tensor_mode", "aten.div.Scalar_mode", "aten.div_.Tensor_mode", "aten.div_.Scalar_mode")
def _div_rounding(c, a):
    return _div_mode(c, a)


@aten("aten.floor_divide.default", "aten.floor_divide.Scalar")
def _floor_divide(c, a):
    return _div_mode(c, {**a, "rounding_mode": "floor"})


# ----------------------------------------------------------------------------- constants folded at import
#
# Flux builds its RoPE frequencies in the forward (``theta ** (arange(0, dim, 2) / dim)``, a scalar
# base with a tensor exponent) and SAM its image wide position grid (``ones(...).cumsum(0)``). Both
# depend on neither an input nor a weight, and the compiler has no kernel for either form, so each
# call ran them on the float64 CPU reference (a fallback). Such a value is computed once at import by
# the float64 reference interpreter, the same computation the reference does on every call, and held
# as a constant. A value that reaches an input or a model parameter is emitted as before (its
# gradient and its per call value are kept).


def _depends_on_nothing(c, x) -> bool:
    """True when ``x`` is computed from buffers and constants only: no program input, no model
    parameter, no node with extra outputs."""
    producer = {id(n.output): n for n in c.nodes}
    seen, stack = set(), [x]
    while stack:
        v = stack.pop()
        if v is None or id(v) in seen:
            continue
        seen.add(id(v))
        if v.kind == "param":
            if v.name in c.weights:
                return False
            continue
        n = producer.get(id(v))
        if n is None or v.kind == "input" or n.extra_outputs:
            return False
        stack.extend(n.inputs)
    return True


def _folding(handler):
    def fn(c, a):
        out = handler(c, a)
        if (c.nodes and c.nodes[-1].output is out and all(isinstance(d, int) for d in out.type.shape)
                and _depends_on_nothing(c, out)):
            data = constant_value(c, out)
            if data is not None:
                op = c.nodes.pop().op
                return _declared(c, f"{op}_folded", data, out.type.dtype)
        return out
    return fn


for _name in ("aten.pow.Scalar", "aten.pow.Tensor_Tensor", "aten.cumsum.default"):
    HANDLERS[_name] = _folding(HANDLERS[_name])
