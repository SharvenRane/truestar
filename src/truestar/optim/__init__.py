"""truestar.optim: optimizers, learning rate schedules and clipping as pure functions over parameter
trees (docs/OPTIM.md).

    opt = optim.AdamW(lr=optim.warmup_cosine(3e-4, warmup=100, total=2000), weight_decay=0.1)
    state = opt.init(params)                                 # {"m": tree, "v": tree}
    params, state = opt.update(params, grads, state, step=0)  # eager, pure: nothing is mutated

The same update is compiled into one training step program by ``truestar.train_step``: the
per step scalars (``scalars``) and the per tensor rule (``leaf``) are written in the native subset
and translated from their source, so the eager update and the compiled one are one definition.

Bias corrections are ``-expm1(t ln b)``, not ``1 - b ** t``: for small t, ``b ** t`` is close to 1
and the subtraction cancels (in float32 at t = 1 with b = 0.999 the relative error of ``1 - b ** t``
is about 6e-5; ``-expm1`` keeps it at a few units in the last place at every t).
"""

from __future__ import annotations

import bisect
import dataclasses
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .. import pytree
from ..frontend import ops
from ..nn.core import inline

# ----------------------------------------------------------------------------- schedules


@dataclasses.dataclass(frozen=True)
class Constant:
    lr: float

    def __call__(self, step: int) -> float:
        return float(self.lr)


@dataclasses.dataclass(frozen=True)
class WarmupCosine:
    """Linear warmup from ``start * peak`` to ``peak`` over ``warmup`` steps, then a half cosine from
    ``peak`` to ``final`` at step ``total``; ``final`` after that. ``step`` is the number of updates
    already applied (the first update uses step 0)."""

    peak: float
    warmup: int
    total: int
    final: float = 0.0
    start: float = 0.0

    def __call__(self, step: int) -> float:
        s = int(step)
        if self.warmup > 0 and s < self.warmup:
            f = self.start + (1.0 - self.start) * s / self.warmup
            return self.peak * f
        if s >= self.total:
            return float(self.final)
        span = max(self.total - self.warmup, 1)
        c = 0.5 * (1.0 + math.cos(math.pi * (s - self.warmup) / span))
        return self.final + (self.peak - self.final) * c


@dataclasses.dataclass(frozen=True)
class WarmupLinear:
    """Linear warmup to ``peak`` over ``warmup`` steps, then linear decay to ``final`` at ``total``."""

    peak: float
    warmup: int
    total: int
    final: float = 0.0

    def __call__(self, step: int) -> float:
        s = int(step)
        if self.warmup > 0 and s < self.warmup:
            return self.peak * s / self.warmup
        if s >= self.total:
            return float(self.final)
        f = (s - self.warmup) / max(self.total - self.warmup, 1)
        return self.peak + (self.final - self.peak) * f


def constant(lr: float) -> Constant:
    return Constant(float(lr))


def warmup_cosine(peak: float, warmup: int, total: int, final: float = 0.0, start: float = 0.0) -> WarmupCosine:
    return WarmupCosine(float(peak), int(warmup), int(total), float(final), float(start))


def warmup_linear(peak: float, warmup: int, total: int, final: float = 0.0) -> WarmupLinear:
    return WarmupLinear(float(peak), int(warmup), int(total), float(final))


@dataclasses.dataclass(frozen=True)
class StepDecay:
    """``lr * gamma ** (step // every)``: torch's StepLR."""

    lr: float
    every: int
    gamma: float = 0.1

    def __call__(self, step: int) -> float:
        return self.lr * self.gamma ** (int(step) // self.every)


@dataclasses.dataclass(frozen=True)
class MultiStep:
    """``lr * gamma ** (number of milestones <= step)``: torch's MultiStepLR."""

    lr: float
    milestones: Tuple[int, ...]
    gamma: float = 0.1

    def __call__(self, step: int) -> float:
        return self.lr * self.gamma ** bisect.bisect_right(self.milestones, int(step))


@dataclasses.dataclass(frozen=True)
class Exponential:
    """``lr * gamma ** step``: torch's ExponentialLR."""

    lr: float
    gamma: float

    def __call__(self, step: int) -> float:
        return self.lr * self.gamma ** int(step)


@dataclasses.dataclass(frozen=True)
class Polynomial:
    """``final + (lr - final) * (1 - min(step, total) / total) ** power``; with final 0, torch's
    PolynomialLR."""

    lr: float
    total: int
    power: float = 1.0
    final: float = 0.0

    def __call__(self, step: int) -> float:
        f = (1.0 - min(int(step), self.total) / self.total) ** self.power
        return self.final + (self.lr - self.final) * f


@dataclasses.dataclass(frozen=True)
class CosineRestarts:
    """SGDR (Loshchilov and Hutter 2017): a half cosine from ``lr`` to ``final`` over ``period``
    steps, restarted, each period ``mult`` times longer: torch's CosineAnnealingWarmRestarts."""

    lr: float
    period: int
    mult: int = 1
    final: float = 0.0

    def __call__(self, step: int) -> float:
        s, t = int(step), self.period
        if self.mult == 1:
            s = s % t
        else:
            # the cycle holding s: the first n with period (mult^n - 1) / (mult - 1) > s
            n = int(math.log(s / t * (self.mult - 1) + 1, self.mult)) if s >= t else 0
            start = t * (self.mult ** n - 1) // (self.mult - 1)
            while start > s:                                   # guard the float log at a boundary
                n -= 1
                start = t * (self.mult ** n - 1) // (self.mult - 1)
            while s >= start + t * self.mult ** n:
                start += t * self.mult ** n
                n += 1
            s, t = s - start, t * self.mult ** n
        return self.final + (self.lr - self.final) * (1.0 + math.cos(math.pi * s / t)) / 2.0


@dataclasses.dataclass(frozen=True)
class OneCycle:
    """The one cycle policy (Smith and Topin 2018) as torch's OneCycleLR defines it: from
    ``peak / div`` up to ``peak`` over the first ``pct_start`` of ``total`` steps, then down to
    ``peak / div / final_div`` (with ``three_phase``: down to ``peak / div``, then to the minimum).
    ``anneal`` "cos" or "linear"."""

    peak: float
    total: int
    pct_start: float = 0.3
    div: float = 25.0
    final_div: float = 1e4
    anneal: str = "cos"
    three_phase: bool = False

    def __post_init__(self):
        if self.anneal not in ("cos", "linear"):
            raise ValueError(f"anneal must be 'cos' or 'linear', got {self.anneal!r}")
        if not 0.0 < self.pct_start < 1.0 or self.total < 2:
            raise ValueError("need 0 < pct_start < 1 and total >= 2")

    def __call__(self, step: int) -> float:
        lo = self.peak / self.div
        end = lo / self.final_div
        a = float(self.pct_start * self.total) - 1
        if self.three_phase:
            phases = [(a, lo, self.peak), (float(2 * self.pct_start * self.total) - 2, self.peak, lo),
                      (self.total - 1, lo, end)]
        else:
            phases = [(a, lo, self.peak), (self.total - 1, self.peak, end)]
        s, begin = int(step), 0.0
        for i, (stop, x, y) in enumerate(phases):
            if s <= stop or i == len(phases) - 1:
                pct = (s - begin) / (stop - begin)
                if self.anneal == "cos":
                    return y + (x - y) / 2.0 * (math.cos(math.pi * pct) + 1)
                return (y - x) * pct + x
            begin = stop
        raise AssertionError("unreachable")


@dataclasses.dataclass(frozen=True)
class Warmup:
    """Linear warmup over ``steps`` updates from ``start * then(0)`` to ``then(0)``, then the
    schedule ``then`` from its own step 0: torch's ``SequentialLR([LinearLR(start, 1, steps), then],
    [steps])``."""

    then: Any
    steps: int
    start: float = 0.0

    def __call__(self, step: int) -> float:
        s = int(step)
        if s < self.steps:
            return self.then(0) * (self.start + (1.0 - self.start) * s / self.steps)
        return self.then(s - self.steps)


def step_decay(lr: float, every: int, gamma: float = 0.1) -> StepDecay:
    if int(every) < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    return StepDecay(float(lr), int(every), float(gamma))


def multistep(lr: float, milestones, gamma: float = 0.1) -> MultiStep:
    ms = tuple(sorted(int(m) for m in milestones))
    return MultiStep(float(lr), ms, float(gamma))


def exponential(lr: float, gamma: float) -> Exponential:
    return Exponential(float(lr), float(gamma))


def polynomial(lr: float, total: int, power: float = 1.0, final: float = 0.0) -> Polynomial:
    if int(total) < 1:
        raise ValueError(f"total must be >= 1, got {total}")
    return Polynomial(float(lr), int(total), float(power), float(final))


def cosine_restarts(lr: float, period: int, mult: int = 1, final: float = 0.0) -> CosineRestarts:
    if int(period) < 1 or int(mult) < 1:
        raise ValueError(f"need period >= 1 and an integer mult >= 1, got {period}, {mult}")
    return CosineRestarts(float(lr), int(period), int(mult), float(final))


def one_cycle(peak: float, total: int, pct_start: float = 0.3, div: float = 25.0, final_div: float = 1e4,
              anneal: str = "cos", three_phase: bool = False) -> OneCycle:
    return OneCycle(float(peak), int(total), float(pct_start), float(div), float(final_div), anneal, bool(three_phase))


def warmup(then, steps: int, start: float = 0.0) -> Warmup:
    """Put a linear warmup of ``steps`` updates in front of any schedule (or a constant lr)."""
    if int(steps) < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    return Warmup(_schedule(then), int(steps), float(start))


Schedule = Union[Constant, WarmupCosine, WarmupLinear, StepDecay, MultiStep, Exponential, Polynomial,
                 CosineRestarts, OneCycle, Warmup]


def _schedule(lr) -> Schedule:
    if isinstance(lr, (int, float)):
        if not lr >= 0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        return Constant(float(lr))
    if callable(lr):
        return lr
    raise TypeError(f"lr must be a number or a schedule (step -> lr), got {type(lr).__name__}")


# ----------------------------------------------------------------------------- clipping


@inline
def global_norm(grads, norm_type=2.0):
    """The norm of all entries of every tensor in the list ``grads`` taken as one vector:
    ``norm_type`` 2 (sqrt of the sum of squares), 1 (sum of absolute values) or ``inf`` (largest
    absolute value), as torch.nn.utils.clip_grad_norm_ computes it."""
    if norm_type == math.inf:
        total = ops.amax(ops.abs(grads[0]))
        for g in grads[1:]:
            total = ops.maximum(total, ops.amax(ops.abs(g)))
        return total
    total = 0.0
    if norm_type == 1.0:
        for g in grads:
            total = total + ops.sum(ops.abs(g))
        return total
    for g in grads:
        total = total + ops.sum(g * g)
    return ops.sqrt(total)


@inline
def clip_by_global_norm(grads, max_norm, norm=None, norm_type=2.0):
    """Scale every gradient by min(1, max_norm / (norm + 1e-6)), the factor of
    torch.nn.utils.clip_grad_norm_ (``norm_type`` 2, 1 or inf). Returns (clipped list, norm before
    clipping)."""
    if norm is None:
        norm = global_norm(grads, norm_type)
    coef = ops.clamp(max_norm / (norm + 1e-6), max=1.0)
    out = []
    for g in grads:
        out = out + [g * ops.cast(coef, g.dtype)]
    return out, norm


@inline
def clip_by_value(grads, limit):
    """Clamp every gradient entry to [-limit, limit]: torch.nn.utils.clip_grad_value_."""
    out = []
    for g in grads:
        out = out + [ops.clamp(g, min=-limit, max=limit)]
    return out


# ----------------------------------------------------------------------------- the optimizer protocol


def _leaves(tree):
    return pytree.flatten(tree, lambda x: isinstance(x, torch.Tensor))


class Optimizer:
    """An optimizer is an immutable config with two native subset methods:

    * ``scalars(self, t, lr)``: the per step 0 dim values every tensor's rule needs (``t`` is the
      1 based update number, ``lr`` the learning rate of this update, both float64 0 dim);
    * ``leaf(self, p, g, s, k)``: one tensor's update: param ``p``, gradient ``g``, its slots ``s``
      (a dict, ``slots`` names them) and the scalars ``k``; returns ``(new p, new slots)``.

    ``init`` and ``update`` apply them to whole trees eagerly; ``truestar.train_step`` translates
    the same two methods into the step program."""

    slots: Tuple[str, ...] = ()

    @property
    def schedule(self) -> Schedule:
        return _schedule(self.lr)

    def lr_at(self, step: int) -> float:
        return float(self.schedule(step))

    def init(self, params) -> Dict[str, Any]:
        """Zeros for every slot of every tensor, as ``{slot: tree shaped like params}``."""
        return {s: _map(params, lambda t: torch.zeros_like(t)) for s in self.slots}

    def update(self, params, grads, state, step: int):
        """Eager update of update number ``step`` (0 based): returns ``(new params, new state)``."""
        ps, spec = _leaves(params)
        gs, gspec = _leaves(grads)
        if pytree.describe(spec) != pytree.describe(gspec):
            raise ValueError(f"grads have structure {pytree.describe(gspec)}, params {pytree.describe(spec)}")
        slot_leaves = {s: _leaves(state[s])[0] for s in self.slots}
        t = torch.tensor(float(step + 1), dtype=torch.float64)
        lr = torch.tensor(self.lr_at(step), dtype=torch.float64)
        k = type(self).scalars(self, t, lr)
        new_p, new_s = [], {s: [] for s in self.slots}
        for i, (p, g) in enumerate(zip(ps, gs)):
            q, sl = type(self).leaf(self, p, g, {s: slot_leaves[s][i] for s in self.slots}, k)
            new_p.append(q)
            for s in self.slots:
                new_s[s].append(sl[s])
        return pytree.unflatten(spec, new_p), {s: pytree.unflatten(spec, new_s[s]) for s in self.slots}

    def config(self) -> Dict[str, Any]:
        out = {"optimizer": type(self).__name__}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            out[f.name] = dataclasses.asdict(v) | {"schedule": type(v).__name__} if dataclasses.is_dataclass(v) else v
        return out


def _map(tree, fn):
    leaves, spec = _leaves(tree)
    return pytree.unflatten(spec, [fn(t) for t in leaves])


def _check_beta(name, b):
    if not 0.0 <= b < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {b}")


@dataclasses.dataclass(frozen=True)
class SGD(Optimizer):
    """p -= lr * d with d = g (plain), d = buf = momentum * buf + g (heavy ball), or
    d = g + momentum * buf (Nesterov). ``weight_decay`` adds wd * p to g first (L2). Equal to
    torch.optim.SGD with dampening 0."""

    lr: Any = 1e-2
    momentum: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0

    def __post_init__(self):
        _schedule(self.lr)
        if self.nesterov and self.momentum <= 0.0:
            raise ValueError("Nesterov needs a momentum > 0")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}")

    @property
    def slots(self):
        return ("buf",) if self.momentum != 0.0 else ()

    def scalars(self, t, lr):
        return {"lr": lr}

    def leaf(self, p, g, s, k):
        lr = ops.cast(k["lr"], p.dtype)
        if self.weight_decay != 0.0:
            g = g + p * self.weight_decay
        if self.momentum == 0.0:
            return p - g * lr, {}
        buf = s["buf"] * self.momentum + g
        if self.nesterov:
            d = g + buf * self.momentum
        else:
            d = buf
        return p - d * lr, {"buf": buf}


@dataclasses.dataclass(frozen=True)
class Adam(Optimizer):
    """m = m + (g - m)(1 - b1); v = b2 v + (1 - b2) g^2;
    p -= (lr / c1) * m / (sqrt(v) / sqrt(c2) + eps) with c_i = -expm1(t ln b_i) = 1 - b_i^t.
    ``weight_decay`` is L2 (added to g); AdamW decouples it."""

    lr: Any = 1e-3
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    slots = ("m", "v")
    decoupled = False

    def __post_init__(self):
        _schedule(self.lr)
        _check_beta("b1", self.b1)
        _check_beta("b2", self.b2)

    def scalars(self, t, lr):
        c1 = -ops.expm1(t * math.log(self.b1)) if self.b1 > 0.0 else ops.ones_like(t)
        c2 = -ops.expm1(t * math.log(self.b2)) if self.b2 > 0.0 else ops.ones_like(t)
        return {"step": lr / c1, "c2s": ops.sqrt(c2), "decay": 1.0 - lr * self.weight_decay}

    def leaf(self, p, g, s, k):
        dt = p.dtype
        if self.weight_decay != 0.0:
            if self.decoupled:
                p = p * ops.cast(k["decay"], dt)
            else:
                g = g + p * self.weight_decay
        m = s["m"] + (g - s["m"]) * (1.0 - self.b1)
        v = s["v"] * self.b2 + g * g * (1.0 - self.b2)
        denom = ops.sqrt(v) / ops.cast(k["c2s"], dt) + self.eps
        return p - (m / denom) * ops.cast(k["step"], dt), {"m": m, "v": v}


@dataclasses.dataclass(frozen=True)
class AdamW(Adam):
    """Adam with decoupled weight decay: p *= 1 - lr * wd before the Adam step."""

    weight_decay: float = 1e-2
    decoupled = True


@dataclasses.dataclass(frozen=True)
class Lion(Optimizer):
    """Lion (Chen et al. 2023, "Symbolic Discovery of Optimization Algorithms", Algorithm 2):
    c = b1 m + (1 - b1) g; p *= 1 - lr * wd; p -= lr * sign(c); m = b2 m + (1 - b2) g.
    One slot (half of Adam's memory). The paper's defaults: lr 3 to 10 times smaller than AdamW's,
    weight decay 3 to 10 times larger. sign(0) is 0."""

    lr: Any = 1e-4
    b1: float = 0.9
    b2: float = 0.99
    weight_decay: float = 0.0
    slots = ("m",)

    def __post_init__(self):
        _schedule(self.lr)
        _check_beta("b1", self.b1)
        _check_beta("b2", self.b2)

    def scalars(self, t, lr):
        return {"lr": lr, "decay": 1.0 - lr * self.weight_decay}

    def leaf(self, p, g, s, k):
        dt = p.dtype
        m = s["m"]
        c = m * self.b1 + g * (1.0 - self.b1)
        sign = ops.cast(c > 0.0, dt) - ops.cast(c < 0.0, dt)
        if self.weight_decay != 0.0:
            p = p * ops.cast(k["decay"], dt)
        return p - sign * ops.cast(k["lr"], dt), {"m": m * self.b2 + g * (1.0 - self.b2)}


@dataclasses.dataclass(frozen=True)
class RMSprop(Optimizer):
    """v = alpha v + (1 - alpha) g^2; with ``centered`` also a = a + (g - a)(1 - alpha) and the
    denominator sqrt(v - a^2) + eps, else sqrt(v) + eps; with ``momentum`` buf = momentum buf +
    g / denom and p -= lr buf, else p -= lr g / denom. ``weight_decay`` is L2. Equal to
    torch.optim.RMSprop."""

    lr: Any = 1e-2
    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0.0
    momentum: float = 0.0
    centered: bool = False

    def __post_init__(self):
        _schedule(self.lr)
        _check_beta("alpha", self.alpha)
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}")

    @property
    def slots(self):
        return ("v",) + (("a",) if self.centered else ()) + (("buf",) if self.momentum != 0.0 else ())

    def scalars(self, t, lr):
        return {"lr": lr}

    def leaf(self, p, g, s, k):
        lr = ops.cast(k["lr"], p.dtype)
        if self.weight_decay != 0.0:
            g = g + p * self.weight_decay
        v = s["v"] * self.alpha + g * g * (1.0 - self.alpha)
        if self.centered:
            a = s["a"] + (g - s["a"]) * (1.0 - self.alpha)
            denom = ops.sqrt(v - a * a) + self.eps
        else:
            denom = ops.sqrt(v) + self.eps
        if self.momentum != 0.0:
            buf = s["buf"] * self.momentum + g / denom
            if self.centered:
                return p - buf * lr, {"v": v, "a": a, "buf": buf}
            return p - buf * lr, {"v": v, "buf": buf}
        if self.centered:
            return p - (g / denom) * lr, {"v": v, "a": a}
        return p - (g / denom) * lr, {"v": v}


@dataclasses.dataclass(frozen=True)
class Adagrad(Optimizer):
    """sum += g^2; p -= lr_t g / (sqrt(sum) + eps) with lr_t = lr / (1 + (t - 1) lr_decay) (t the 1
    based update number). ``initial_accumulator_value`` starts ``sum``; ``weight_decay`` is L2.
    Equal to torch.optim.Adagrad."""

    lr: Any = 1e-2
    lr_decay: float = 0.0
    weight_decay: float = 0.0
    initial_accumulator_value: float = 0.0
    eps: float = 1e-10
    slots = ("sum",)

    def __post_init__(self):
        _schedule(self.lr)
        if self.lr_decay < 0.0 or self.initial_accumulator_value < 0.0:
            raise ValueError("lr_decay and initial_accumulator_value must be >= 0")

    def init(self, params) -> Dict[str, Any]:
        return {"sum": _map(params, lambda t: torch.full_like(t, float(self.initial_accumulator_value)))}

    def scalars(self, t, lr):
        return {"lr": lr / (1.0 + (t - 1.0) * self.lr_decay)}

    def leaf(self, p, g, s, k):
        lr = ops.cast(k["lr"], p.dtype)
        if self.weight_decay != 0.0:
            g = g + p * self.weight_decay
        acc = s["sum"] + g * g
        return p - (g / (ops.sqrt(acc) + self.eps)) * lr, {"sum": acc}


# ----------------------------------------------------------------------------- EMA of the weights


class _Bound:
    """A method of a static config object callable from a native method: eagerly it runs the Python
    method, inside a native function it inlines the translated method (as a layer call does)."""

    def __init__(self, obj, name: str):
        self.obj, self.name = obj, name

    def __truestar_inline__(self):
        from ..nn.core import native_method

        return native_method(type(self.obj), self.name), (self.obj,)

    def __call__(self, *args):
        return getattr(type(self.obj), self.name)(self.obj, *args)


@dataclasses.dataclass(frozen=True)
class EMA(Optimizer):
    """``inner`` plus an exponential moving average of the weights, kept as one more slot ``"ema"``:
    after each update ``ema = ema + (p - ema)(1 - d)`` with ``p`` the updated weights (torch's
    ``AveragedModel`` with ``get_ema_multi_avg_fn(d)``, timm's ModelEmaV2). ``ema`` starts as a copy
    of the initial weights. ``warmup``: d at update t is ``min(decay, (1 + t) / (10 + t))``, so early
    averages are not dominated by the random start.

    Because it is an optimizer, ``train_step`` compiles the average into the step program and
    ``save``/``load`` carry it: ``ts.opt["ema"]`` is the averaged tree."""

    inner: Any
    decay: float = 0.999
    warmup: bool = False

    def __post_init__(self):
        if not isinstance(self.inner, Optimizer) or isinstance(self.inner, EMA):
            raise ValueError("EMA wraps one truestar.optim optimizer")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {self.decay}")
        if "ema" in self.inner.slots:
            raise ValueError("the inner optimizer already has a slot named 'ema'")

    @property
    def lr(self):
        return self.inner.lr

    @property
    def slots(self):
        return tuple(self.inner.slots) + ("ema",)

    @property
    def inner_scalars(self):
        return _Bound(self.inner, "scalars")

    @property
    def inner_leaf(self):
        return _Bound(self.inner, "leaf")

    def init(self, params) -> Dict[str, Any]:
        return dict(self.inner.init(params), ema=_map(params, lambda t: t.detach().clone()))

    def scalars(self, t, lr):
        k = self.inner_scalars(t, lr)
        if self.warmup:
            d = ops.clamp((t + 1.0) / (t + 10.0), max=self.decay)
        else:
            d = ops.ones_like(t) * self.decay
        return dict(list(k.items()) + [("ema_w", 1.0 - d)])   # no ** or tensor kwargs in the native subset

    def leaf(self, p, g, s, k):
        q, sl = self.inner_leaf(p, g, s, k)
        e = s["ema"]
        return q, dict(list(sl.items()) + [("ema", e + (q - e) * ops.cast(k["ema_w"], e.dtype))])

    def config(self) -> Dict[str, Any]:
        return {"optimizer": "EMA", "decay": self.decay, "warmup": self.warmup, "inner": self.inner.config()}


__all__ = ["Optimizer", "SGD", "Adam", "AdamW", "Lion", "RMSprop", "Adagrad", "EMA", "Constant", "WarmupCosine",
           "StepDecay", "MultiStep", "Exponential", "Polynomial", "CosineRestarts", "OneCycle", "Warmup",
           "step_decay", "multistep", "exponential", "polynomial", "cosine_restarts", "one_cycle", "warmup",
           "WarmupLinear", "constant", "warmup_cosine", "warmup_linear", "global_norm", "clip_by_global_norm",
           "clip_by_value"]

from .muon import Muon, MuonOrthogonalityError, OrthoReport, newton_schulz, orthogonality_report  # noqa: E402

__all__ += ["Muon", "MuonOrthogonalityError", "OrthoReport", "newton_schulz", "orthogonality_report"]
