"""Training spike watch: a float64 shadow of sampled steps, and a run directory any step can be replayed from.

The usual response to a loss spike is to rewind about 100 steps, skip a few hundred batches and
hope; that hides the cause. The watch keeps what a replay needs instead: a checkpoint every
`ckpt_every` steps (weights and optimizer state before that step), every batch since, and the
learning rate each step used. `truestar replay RUN --step N` (truestar.spikes.replay) then reruns
the window exactly and says whether the spike is numerics or data.

The shadow check is the cheap online part. On every `shadow_every`-th step, on a step whose loss
jumps, or after `request_shadow()`, the same batch and the same weights are run once more in
float64 (forward and backward) and compared with the low precision step: loss error, gradient
cosine and norm ratio, and the own error of chosen modules (each recomputed in float64 on the
live low precision inputs). Steps in between cost nothing extra but one batch save.

The step itself is fixed so a replay can reproduce it: forward in the model's dtype, the spec's
loss on float32 outputs, backward, optional clipping by the global norm, optimizer step.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import statistics
import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32, "float64": torch.float64}


def load_spec(path: str):
    """The spike spec file: get_model(), get_loss() (a mean loss callable), get_optimizer(params)."""
    import importlib.util
    import sys

    path = os.path.abspath(path)
    name = "truestar_spike_spec_" + str(abs(hash(path)))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    for fn in ("get_model", "get_loss", "get_optimizer"):
        if not hasattr(mod, fn):
            raise ValueError(f"{path} must define {fn}()")
    return mod


def _cast_batch(x, dtype):
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        return x.to(dtype)
    return x


def loss_on(loss_fn, out: torch.Tensor, y, dtype: torch.dtype):
    """The spec's loss on float32 outputs (float64 in the float64 arm), targets cast to match."""
    out = out if dtype == torch.float64 else out.float()
    return loss_fn(out, _cast_batch(y, out.dtype))


def forward_backward(model: nn.Module, loss_fn, x, y, dtype: torch.dtype) -> float:
    loss = loss_on(loss_fn, model(_cast_batch(x, dtype)), y, dtype)
    loss.backward()
    return float(loss.detach())


def apply_update(model: nn.Module, opt, lr: float, clip: float) -> float:
    for g in opt.param_groups:
        g["lr"] = lr
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    norm = float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads))) if grads else 0.0
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return norm


def train_step(model: nn.Module, opt, loss_fn, x, y, dtype: torch.dtype, lr: float, clip: float):
    """One step, the same code for the run, its replay and the float64 arm. Returns (loss, grad norm)."""
    loss = forward_backward(model, loss_fn, x, y, dtype)
    return loss, apply_update(model, opt, lr, clip)


def own_errors(model: nn.Module, x, dtype: torch.dtype, names: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Each chosen module (default every leaf) recomputed in float64 on the inputs it saw in a low
    precision forward: max abs difference over the float64 output's max abs. A non finite low
    precision output where float64 is finite is infinite error."""
    mods = dict(model.named_modules())
    if names is None:
        names = [n for n, m in mods.items() if n and not list(m.children())]
    seen: Dict[str, tuple] = {}
    handles = [mods[n].register_forward_hook(lambda m, a, o, n=n: (seen.setdefault(n, (a, o)), None)[1]) for n in names]
    try:
        with torch.no_grad():
            model(_cast_batch(x, dtype))
    finally:
        for h in handles:
            h.remove()
    errs: Dict[str, float] = {}
    with torch.no_grad():
        for n in names:
            if n not in seen or not isinstance(seen[n][1], torch.Tensor):
                continue
            a, o = seen[n]
            ref = copy.deepcopy(mods[n]).double()(*[_cast_batch(t, torch.float64) for t in a])
            errs[n] = rel_error(o, ref)
    return errs


def rel_error(lowp: torch.Tensor, ref: torch.Tensor) -> float:
    lowp, ref = lowp.double(), ref.double()
    if not bool(torch.isfinite(ref).all()):
        return 0.0 if torch.equal(torch.isfinite(lowp), torch.isfinite(ref)) else math.inf
    if not bool(torch.isfinite(lowp).all()):
        return math.inf
    s = float(ref.abs().max())
    d = float((lowp - ref).abs().max())
    return d / s if s > 0 else d


@dataclasses.dataclass
class Shadow:
    loss_lowp: float
    loss_f64: float
    loss_error: float
    grad_cosine: float
    grad_norm_ratio: float
    module_error: Dict[str, float]
    flagged: bool
    seconds: float


def shadow_check(model: nn.Module, loss_fn, x, y, dtype: torch.dtype, modules: Optional[Sequence[str]] = None,
                 history: Sequence[float] = ()) -> Shadow:
    """The step's loss and gradients in float64 from the same weights and batch, against the low
    precision ones. Leaves `model` untouched."""
    t0 = time.perf_counter()
    lp = copy.deepcopy(model)
    hp = copy.deepcopy(model).double()
    for m in (lp, hp):
        for p in m.parameters():
            p.grad = None
    loss_l = loss_on(loss_fn, lp(_cast_batch(x, dtype)), y, dtype)
    loss_l.backward()
    loss_h = loss_on(loss_fn, hp(_cast_batch(x, torch.float64)), y, torch.float64)
    loss_h.backward()
    gl = torch.cat([p.grad.double().reshape(-1) for p in lp.parameters() if p.grad is not None])
    gh = torch.cat([p.grad.reshape(-1) for p in hp.parameters() if p.grad is not None])
    ll, lh = float(loss_l.detach()), float(loss_h.detach())
    if math.isfinite(ll) and bool(torch.isfinite(gl).all()):
        cos = float((gl @ gh) / (gl.norm() * gh.norm() + 1e-300))
        ratio = float(gl.norm() / (gh.norm() + 1e-300))
        lerr = abs(ll - lh) / max(abs(lh), 1e-300)
    else:
        cos, ratio, lerr = math.nan, math.nan, math.inf
    merr = own_errors(model, x, dtype, modules) if modules is not None else {}
    prev = [h for h in history if math.isfinite(h)]
    bar = max(0.05, 20 * statistics.median(prev)) if prev else 0.05
    flagged = (not math.isfinite(lerr)) or lerr > bar or (math.isfinite(cos) and cos < 0.9)
    return Shadow(ll, lh, lerr, cos, ratio, merr, flagged, time.perf_counter() - t0)


class SpikeWatch:
    """Run a training loop through `watch.step(x, y)`; everything a replay needs lands in `run_dir`."""

    def __init__(self, spec_path: str, run_dir: str, dtype: str = "float16", lr: Optional[float] = None,
                 clip: float = 0.0, shadow_every: int = 50, ckpt_every: int = 20, spike_factor: float = 3.0,
                 modules: Optional[Sequence[str]] = None, seed: int = 0, save_batches: bool = True) -> None:
        if dtype not in DTYPES:
            raise ValueError(f"dtype {dtype!r}: one of {', '.join(DTYPES)}")
        self.spec_path = os.path.abspath(spec_path)
        self.spec = load_spec(spec_path)
        self.run_dir = run_dir
        os.makedirs(os.path.join(run_dir, "batches"), exist_ok=True)
        self.dtype = DTYPES[dtype]
        torch.manual_seed(seed)
        self.model = self.spec.get_model().to(self.dtype)
        self.loss_fn = self.spec.get_loss()
        self.opt = self.spec.get_optimizer(self.model.parameters())
        self.base_lr = self.opt.param_groups[0]["lr"] if lr is None else lr
        self.clip, self.shadow_every, self.ckpt_every = clip, shadow_every, ckpt_every
        self.spike_factor, self.modules, self.save_batches = spike_factor, modules, save_batches
        self.n = 0
        self.losses: List[float] = []
        self.shadow_errors: List[float] = []
        self._want_shadow = False
        self.shadow_seconds = 0.0
        self.step_seconds = 0.0
        with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"spec": self.spec_path, "dtype": dtype, "clip": clip, "ckpt_every": ckpt_every,
                       "spike_factor": spike_factor, "format": 1}, f, indent=2)
        self._log = open(os.path.join(run_dir, "watch.jsonl"), "a", encoding="utf-8")

    def request_shadow(self) -> None:
        """Shadow the next step whatever its number."""
        self._want_shadow = True

    def is_spike(self, loss: float) -> bool:
        prev = [v for v in self.losses[-20:] if math.isfinite(v)]
        if not math.isfinite(loss):
            return True
        return len(prev) >= 5 and loss > self.spike_factor * statistics.median(prev)

    def step(self, x, y, lr: Optional[float] = None) -> dict:
        n = self.n
        lr = self.base_lr if lr is None else lr
        if n % self.ckpt_every == 0:
            torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "step": n},
                       os.path.join(self.run_dir, f"ckpt_{n}.pt"))
        if self.save_batches:
            torch.save((x, y), os.path.join(self.run_dir, "batches", f"{n}.pt"))
        t0 = time.perf_counter()
        loss = forward_backward(self.model, self.loss_fn, x, y, self.dtype)
        self.step_seconds += time.perf_counter() - t0
        spike = self.is_spike(loss)
        sh = None
        if self._want_shadow or spike or (self.shadow_every > 0 and n % self.shadow_every == 0):
            sh = shadow_check(self.model, self.loss_fn, x, y, self.dtype, self.modules, self.shadow_errors)
        t0 = time.perf_counter()
        gnorm = apply_update(self.model, self.opt, lr, self.clip)
        self.step_seconds += time.perf_counter() - t0
        rec = {"step": n, "loss": loss if math.isfinite(loss) else None, "loss_text": repr(loss),
               "grad_norm": gnorm if math.isfinite(gnorm) else None,
               "lr": lr, "spike": spike}
        if sh is not None:
            self.shadow_seconds += sh.seconds
            if not sh.flagged and math.isfinite(sh.loss_error):
                self.shadow_errors.append(sh.loss_error)
            rec["shadow"] = {k: (v if not isinstance(v, float) or math.isfinite(v) else None)
                             for k, v in dataclasses.asdict(sh).items() if k != "module_error"}
            rec["shadow"]["module_error"] = {k: (v if math.isfinite(v) else None) for k, v in sh.module_error.items()}
            if sh.flagged:
                rec["code"] = "TS691"
        self._log.write(json.dumps(rec, allow_nan=False) + "\n")
        self._log.flush()
        self._want_shadow = False
        self.losses.append(loss)
        self.n += 1
        return rec

    def close(self) -> None:
        self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
