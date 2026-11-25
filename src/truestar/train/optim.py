"""Optimizers as pure IR update programs.

An optimizer turns ``(params, grads, state, lr)`` into ``(new params, new state)`` with IR ops only,
so the update can be fused with the backward pass by a compiler later. State is explicit: every
moment and every step counter is a program input and a program output, nothing is hidden.
Hyperparameters other than the learning rate are constants baked into the program; the learning
rate is a 0 dim input so a schedule needs no recompile.

The formulas follow ``torch.optim`` (2.11, single tensor path) operation by operation:

* ``SGD``: ``g += wd * p``; with momentum ``buf = g`` on the first step, else
  ``buf = momentum * buf + (1 - dampening) * g``; Nesterov uses ``g + momentum * buf``, otherwise
  ``buf``; ``p -= lr * g``.
* ``Adam`` / ``AdamW``: ``t += 1``; AdamW decays ``p *= 1 - lr * wd`` (Adam adds ``wd * p`` to the
  gradient); ``m = lerp(m, g, 1 - b1)``; ``v = b2 v + (1 - b2) g^2``; bias corrections
  ``c1 = 1 - b1^t``, ``c2 = 1 - b2^t``; ``p -= (lr / c1) * m / (sqrt(v) / sqrt(c2) + eps)``; with
  amsgrad the running max of ``v`` replaces ``v`` in the denominator.
* ``Lion`` (Chen et al. 2023; not in torch.optim): ``c = b1 m + (1 - b1) g``;
  ``p *= 1 - lr * wd``; ``p -= lr * sign(c)``; ``m = b2 m + (1 - b2) g``.

``b^t`` is computed as ``exp(t * ln b)`` because the IR has no power op; in float64 that differs from
Python's ``b ** t`` by a few units in the last place. The bias corrections ``1 - b^t`` are
``-expm1(t * ln b)``, which does not cancel when ``b^t`` is close to 1 (float32 steps).

Gradient clipping by global norm follows ``torch.nn.utils.clip_grad_norm_``: ``total = sqrt(sum of
squares)``, ``coef = min(max_norm / (total + 1e-6), 1)``, every gradient times ``coef``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..ir.core import IRError, Program, TensorType, Value
from .edit import Editor

StateSpec = Tuple[str, TensorType, float]  # name, type, initial fill value


def _scalar_type(dtype=torch.float64) -> TensorType:
    return TensorType(dtype, ())


class Optimizer:
    """Base class. Subclasses define ``state_spec`` and ``emit``."""

    def __init__(self, lr: float):
        if not lr >= 0.0:
            raise IRError(f"lr must be >= 0, got {lr}")
        self.lr = lr

    # -- state
    def state_spec(self, params: Sequence[Value]) -> List[StateSpec]:
        raise NotImplementedError

    def init_state(self, params: Sequence[Value]) -> Dict[str, torch.Tensor]:
        out = {}
        for name, t, fill in self.state_spec(params):
            out[name] = torch.full(tuple(int(d) for d in t.shape), fill, dtype=t.dtype)
        return out

    # -- update
    def emit(self, e: Editor, params: Sequence[Value], grads: Sequence[Value], state: Dict[str, Value],
             lr: Value) -> Tuple[List[Value], Dict[str, Value]]:
        raise NotImplementedError

    def config(self) -> dict:
        return {"optimizer": type(self).__name__, **{k: v for k, v in vars(self).items()}}

    # helpers
    @staticmethod
    def _c(e: Editor, value: float, like: Value) -> Value:
        return e.const(value, like.type.dtype)

    @staticmethod
    def _pow(e: Editor, base: float, t: Value) -> Value:
        """base ** t for a 0 dim step count t, as exp(t * ln base)."""
        if base <= 0.0:
            return e.op("mul", t, e.const(0.0, t.type.dtype))  # only base 0 reaches here (betas are in [0, 1))
        return e.op("exp", e.op("mul", t, e.const(math.log(base), t.type.dtype)))

    @staticmethod
    def _one_minus_pow(e: Editor, base: float, t: Value, name: str) -> Value:
        """1 - base ** t as -expm1(t * ln base). For a base near 1 (b2 = 0.999) and a small t,
        1 - exp(...) cancels: in float32 it keeps about 3 of its 7 digits at t = 1, and a compiled
        float32 step then fails its kernel check against float64. expm1 keeps them."""
        if base <= 0.0:
            return e.op("sub", e.const(1.0, t.type.dtype), e.op("mul", t, e.const(0.0, t.type.dtype)), name=name)
        return e.op("neg", e.op("expm1", e.op("mul", t, e.const(math.log(base), t.type.dtype))), name=name)


def _check_beta(name, b):
    if not 0.0 <= b < 1.0:
        raise IRError(f"{name} must be in [0, 1), got {b}")


class SGD(Optimizer):
    def __init__(self, lr: float, momentum: float = 0.0, dampening: float = 0.0, weight_decay: float = 0.0,
                 nesterov: bool = False):
        super().__init__(lr)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise IRError("Nesterov momentum requires a momentum and zero dampening (as torch.optim.SGD)")
        self.momentum, self.dampening, self.weight_decay, self.nesterov = momentum, dampening, weight_decay, nesterov

    def state_spec(self, params):
        if self.momentum == 0:
            return []
        spec = [("step", _scalar_type(), 0.0)]
        spec += [(f"{p.name}.momentum_buffer", p.type, 0.0) for p in params]
        return spec

    def emit(self, e, params, grads, state, lr):
        new_p, new_s = [], {}
        first = None
        if self.momentum != 0:
            step = state["step"]
            first = e.op("eq", step, e.const(0.0, step.type.dtype), name="sgd_first_step")
            new_s["step"] = e.op("add", step, e.const(1.0, step.type.dtype), name="sgd_step")
        for p, g in zip(params, grads):
            e.source = f"truestar.train SGD update of {p.name}"
            if self.weight_decay != 0:
                g = e.op("add", g, e.op("mul", p, self._c(e, self.weight_decay, p)))
            if self.momentum != 0:
                buf = state[f"{p.name}.momentum_buffer"]
                later = e.op("add", e.op("mul", buf, self._c(e, self.momentum, buf)),
                             e.op("mul", g, self._c(e, 1.0 - self.dampening, g)) if self.dampening else g)
                buf = e.op("where", first, g, later, name=f"{p.name}_momentum")
                new_s[f"{p.name}.momentum_buffer"] = buf
                g = e.op("add", g, e.op("mul", buf, self._c(e, self.momentum, buf))) if self.nesterov else buf
            new_p.append(e.op("sub", p, e.op("mul", g, lr), name=f"{p.name}_new"))
        return new_p, new_s


class Adam(Optimizer):
    decoupled = False

    def __init__(self, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, amsgrad: bool = False):
        super().__init__(lr)
        _check_beta("beta1", betas[0])
        _check_beta("beta2", betas[1])
        self.betas, self.eps, self.weight_decay, self.amsgrad = tuple(betas), eps, weight_decay, amsgrad

    def state_spec(self, params):
        spec = [("step", _scalar_type(), 0.0)]
        for p in params:
            spec += [(f"{p.name}.exp_avg", p.type, 0.0), (f"{p.name}.exp_avg_sq", p.type, 0.0)]
            if self.amsgrad:
                spec.append((f"{p.name}.max_exp_avg_sq", p.type, 0.0))
        return spec

    def emit(self, e, params, grads, state, lr):
        b1, b2 = self.betas
        e.source = f"truestar.train {type(self).__name__} step count and bias corrections"
        step = e.op("add", state["step"], e.const(1.0, state["step"].type.dtype), name="adam_step")
        bc1 = self._one_minus_pow(e, b1, step, name="adam_bias_correction1")
        bc2 = self._one_minus_pow(e, b2, step, name="adam_bias_correction2")
        step_size = e.op("div", lr, bc1, name="adam_step_size")
        bc2_sqrt = e.op("sqrt", bc2, name="adam_bias_correction2_sqrt")
        decay = None
        if self.weight_decay != 0 and self.decoupled:
            decay = e.op("sub", e.const(1.0, lr.type.dtype),
                         e.op("mul", lr, e.const(self.weight_decay, lr.type.dtype)), name="adamw_decay")
        new_p, new_s = [], {"step": step}
        for p, g in zip(params, grads):
            base = p.name
            e.source = f"truestar.train {type(self).__name__} update of {base}"
            if self.weight_decay != 0:
                if self.decoupled:
                    p = e.op("mul", p, decay, name=f"{base}_decayed")
                else:
                    g = e.op("add", g, e.op("mul", p, self._c(e, self.weight_decay, p)))
            m = state[f"{base}.exp_avg"]
            v = state[f"{base}.exp_avg_sq"]
            m = e.op("add", m, e.op("mul", e.op("sub", g, m), self._c(e, 1.0 - b1, m)), name=f"{base}_exp_avg")
            v = e.op("add", e.op("mul", v, self._c(e, b2, v)),
                     e.op("mul", e.op("mul", g, g), self._c(e, 1.0 - b2, v)), name=f"{base}_exp_avg_sq")
            new_s[f"{base}.exp_avg"], new_s[f"{base}.exp_avg_sq"] = m, v
            v_hat = v
            if self.amsgrad:
                vmax = state[f"{base}.max_exp_avg_sq"]
                vmax = e.op("where", e.op("gt", vmax, v), vmax, v, name=f"{base}_max_exp_avg_sq")
                new_s[f"{base}.max_exp_avg_sq"] = vmax
                v_hat = vmax
            denom = e.op("add", e.op("div", e.op("sqrt", v_hat), bc2_sqrt), self._c(e, self.eps, v_hat),
                         name=f"{base}_denom")
            upd = e.op("mul", e.op("div", m, denom), step_size)
            new_p.append(e.op("sub", p, upd, name=f"{base}_new"))
        return new_p, new_s


class AdamW(Adam):
    decoupled = True

    def __init__(self, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2, amsgrad: bool = False):
        super().__init__(lr, betas, eps, weight_decay, amsgrad)


class Lion(Optimizer):
    def __init__(self, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0):
        super().__init__(lr)
        _check_beta("beta1", betas[0])
        _check_beta("beta2", betas[1])
        self.betas, self.weight_decay = tuple(betas), weight_decay

    def state_spec(self, params):
        return [(f"{p.name}.exp_avg", p.type, 0.0) for p in params]

    def emit(self, e, params, grads, state, lr):
        b1, b2 = self.betas
        decay = None
        if self.weight_decay != 0:
            e.source = "truestar.train Lion decay"
            decay = e.op("sub", e.const(1.0, lr.type.dtype),
                         e.op("mul", lr, e.const(self.weight_decay, lr.type.dtype)), name="lion_decay")
        new_p, new_s = [], {}
        for p, g in zip(params, grads):
            e.source = f"truestar.train Lion update of {p.name}"
            m = state[f"{p.name}.exp_avg"]
            c = e.op("add", e.op("mul", m, self._c(e, b1, m)), e.op("mul", g, self._c(e, 1.0 - b1, g)),
                     name=f"{p.name}_lion_c")
            zero = self._c(e, 0.0, c)
            sign = e.op("where", e.op("gt", c, zero), self._c(e, 1.0, c),
                        e.op("where", e.op("lt", c, zero), self._c(e, -1.0, c), zero), name=f"{p.name}_sign")
            q = e.op("mul", p, decay) if decay is not None else p
            new_p.append(e.op("sub", q, e.op("mul", sign, lr), name=f"{p.name}_new"))
            new_s[f"{p.name}.exp_avg"] = e.op("add", e.op("mul", m, self._c(e, b2, m)),
                                              e.op("mul", g, self._c(e, 1.0 - b2, g)), name=f"{p.name}_exp_avg")
        return new_p, new_s


# ----------------------------------------------------------------------------- clipping, finiteness


ROW = 4096   # a flat param group's length is a multiple of this (see _rows)


def _rows(e: Editor, v: Value) -> Optional[Value]:
    """A long flat value as [n / ROW, ROW], so a reduction over all of it runs as one program per row
    and then one short sum over the rows (two kernels, a fixed order) instead of one program over
    millions of entries (2.9 ms for 7.3M entries on the RTX 5070 Ti, against tens of microseconds).
    None when ``v`` is not such a value."""
    shape = v.type.shape
    if len(shape) != 1 or not isinstance(shape[0], int) or shape[0] < 2 * ROW or shape[0] % ROW:
        return None
    return e.op("reshape", v, shape=[shape[0] // ROW, ROW], name=f"{v.name}_rows")


def _reduce_all(e: Editor, op: str, v: Value) -> Value:
    r = _rows(e, v)
    if r is None:
        return e.op(op, v, dims=None, keepdim=False)
    return e.op(op, e.op(op, r, dims=[1], keepdim=False), dims=[0], keepdim=False)


def global_norm(e: Editor, grads: Sequence[Value]) -> Value:
    """sqrt of the sum of squares of every gradient entry, as a 0 dim value."""
    e.source = "truestar.train global gradient norm"
    total = None
    for g in grads:
        s = _reduce_all(e, "sum", e.op("mul", g, g))
        total = s if total is None else e.op("add", total, s)
    return e.op("sqrt", total, name="grad_norm")


def clip_by_global_norm(e: Editor, grads: Sequence[Value], max_norm: float,
                        norm: Optional[Value] = None) -> Tuple[List[Value], Value]:
    """torch.nn.utils.clip_grad_norm_ (norm type 2). Returns the clipped gradients and the norm."""
    if not max_norm > 0:
        raise IRError(f"max_norm must be > 0, got {max_norm}")
    norm = norm if norm is not None else global_norm(e, grads)
    e.source = "truestar.train gradient clipping"
    coef = e.op("div", e.const(max_norm, norm.type.dtype), e.op("add", norm, e.const(1e-6, norm.type.dtype)))
    coef = e.op("clamp", coef, max=1.0, name="clip_coef")
    return [e.op("mul", g, coef, name=f"{g.name}_clipped") for g in grads], norm


def any_nonfinite(e: Editor, values: Sequence[Value]) -> Value:
    """A 0 dim bool: some entry of some value is NaN or infinite. ``x <= max and x >= -max`` is false
    exactly for NaN and both infinities, so only comparisons are needed."""
    e.source = "truestar.train non finite check"
    flag = None
    for v in values:
        big = torch.finfo(v.type.dtype).max
        ok = e.op("logical_and", e.op("le", v, e.const(big, v.type.dtype)),
                  e.op("ge", v, e.const(-big, v.type.dtype)))
        bad = _reduce_all(e, "any", e.op("logical_not", ok))
        flag = bad if flag is None else e.op("logical_or", flag, bad)
    return flag


# ----------------------------------------------------------------------------- standalone program


def update_program(optimizer: Optimizer, param_types: Dict[str, TensorType], name: str = "update") -> Program:
    """The optimizer alone as an IR program.

    inputs: ``lr``, then one input per param (its name), one gradient per param (``<name>.grad``),
    then every state value in ``state_spec`` order. outputs: the new params, then the new state in
    the same order."""
    from ..ir.core import Builder

    b = Builder(name)
    lr = b.input("lr", torch.float64, ())
    params = [b.input(n, t.dtype, t.shape) for n, t in param_types.items()]
    grads = [b.input(f"{n}.grad", t.dtype, t.shape) for n, t in param_types.items()]
    spec = optimizer.state_spec(params)
    state = {sname: b.input(sname, t.dtype, t.shape) for sname, t, _ in spec}
    e = Editor(b.build([lr]), name)
    new_p, new_s = optimizer.emit(e, params, grads, state, lr)
    return e.build(list(new_p) + [new_s[s] for s, _, _ in spec])
