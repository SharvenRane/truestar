"""Run a training step program: learning rate schedule, logging, NaN and Inf attribution, checkpoints.

The trainer holds the carried values (params and state) as tensors in their declared dtypes, and per
step makes exactly one executor call with the batch, the learning rate and the carried values. The
executor is anything with ``prepare(program) -> callable(inputs) -> outputs`` (the seam of
``truestar.frontend.Executor``) or a plain ``fn(program, inputs) -> outputs``; the default is the
float64 reference interpreter.

**Non finite values.** The step program reports whether the loss or any gradient holds a NaN or an
Inf. Without loss scaling that is an error: the trainer does not apply the step, reruns the same
inputs through the reference interpreter keeping every intermediate value, and raises
:class:`NonFiniteError` naming the first op whose result is not finite, with its source (the ATen
op and the user's line for imported nodes). With loss scaling an overflow is expected and the
program itself skips the step, so the trainer only records it.

**Keys.** With ``seed=`` the trainer fills the step's random key input (dropout and other keyed
draws, docs/RANDOM.md) with ``rng.call_key(seed, step)``, so a step's randomness depends on the seed
and the step number only.

**Checkpoints** (``truestar.train.checkpoint``) reuse the ``.tsx`` file format: the carried tensors
(params and optimizer state) plus the step count, history, seed and, when given, the data loader
position. Same state, same bytes. Resuming restores all of it, so the continuation is bit identical
to an uninterrupted run.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch

from .. import rng
from ..data.errors import DataError
from ..ir.core import IRError, Program, TensorType
from ..ir import serialize
from . import checkpoint as ckpt
from .checkpoint import CHECKPOINT_KEY, CHECKPOINT_VERSION
from .step import N_HEAD, TrainStep


# ----------------------------------------------------------------------------- schedules


@dataclass(frozen=True)
class WarmupCosine:
    """Linear warmup, then cosine decay. The learning rate of step ``k`` (0 based) is
    ``base * (s + (1 - s) * k / W)`` for ``k < W``, then
    ``min_lr + (base - min_lr) * (1 + cos(pi * (k - W) / (T - W))) / 2`` up to ``T``, then ``min_lr``.
    Equal to torch's ``SequentialLR([LinearLR(s, 1, W), CosineAnnealingLR(T - W, min_lr)], [W])``."""

    base_lr: float
    warmup_steps: int
    total_steps: int
    min_lr: float = 0.0
    warmup_start_factor: float = 0.0

    def __post_init__(self):
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError(f"need 0 <= warmup_steps < total_steps, got {self.warmup_steps}, {self.total_steps}")

    def __call__(self, k: int) -> float:
        w, t, s = self.warmup_steps, self.total_steps, self.warmup_start_factor
        if k < w:
            return self.base_lr * (s + (1.0 - s) * k / w)
        if k >= t:
            return self.min_lr
        return self.min_lr + (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * (k - w) / (t - w))) / 2.0

    def config(self) -> dict:
        return {"schedule": "WarmupCosine", **vars(self)}


@dataclass(frozen=True)
class Constant:
    lr: float

    def __call__(self, k: int) -> float:
        return self.lr

    def config(self) -> dict:
        return {"schedule": "Constant", "lr": self.lr}


# ----------------------------------------------------------------------------- non finite attribution


class NonFiniteError(IRError):
    """A NaN or Inf in the loss or the gradients, with the first op that produced it."""

    def __init__(self, message: str, report: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.report = report or {}


def _count(t: torch.Tensor) -> Dict[str, int]:
    return {"nan": int(torch.isnan(t).sum()), "inf": int(torch.isinf(t).sum()), "numel": int(t.numel())}


def first_nonfinite(program: Program, inputs: Sequence[torch.Tensor]) -> Dict[str, Any]:
    """Run ``program`` on the reference interpreter keeping every value, and return where the first
    NaN or Inf appears: a program input, or the first node (in program order, so its operands are all
    finite) whose result is not finite."""
    from ..ir import reference

    for v, t in zip(program.inputs, inputs):
        if t.dtype.is_floating_point and not torch.isfinite(t).all():
            return {"kind": "input", "value": v.name, "type": str(v.type), **_count(t),
                    "message": f"program input %{v.name}: {v.type} already holds non finite values"}
    for p in program.params:
        t = program.param_data[p.name]
        if t.dtype.is_floating_point and not torch.isfinite(t).all():
            return {"kind": "param", "value": p.name, "type": str(p.type), **_count(t),
                    "message": f"param @{p.name}: {p.type} holds non finite values"}
    _, values = reference.run(program, *inputs, return_all=True)
    for n in program.nodes:
        t = values.get(n.output.name)
        if t is None or not t.dtype.is_floating_point or torch.isfinite(t).all():
            continue
        args = ", ".join("_" if x is None else x.ref for x in n.inputs)
        c = _count(t)
        return {"kind": "node", "op": n.op, "value": n.output.name, "type": str(n.output.type),
                "source": n.source, **c,
                "message": f"%{n.output.name} = {n.op}({args}) : {n.output.type} is the first non finite value "
                           f"({c['nan']} NaN, {c['inf']} Inf of {c['numel']}); its operands are finite"
                           + (f"\n  at: {n.source}" if n.source else "")}
    return {"kind": "none", "message": "the reference interpreter found no non finite value (the executor "
                                       "disagrees with the reference: check the executor)"}


# ----------------------------------------------------------------------------- executor seam


def _prepare(executor, program: Program) -> Callable[[Sequence[torch.Tensor]], List[torch.Tensor]]:
    if executor is None:
        from ..frontend.function import ReferenceExecutor

        executor = ReferenceExecutor()
    if hasattr(executor, "prepare"):
        return executor.prepare(program)
    if callable(executor):
        return lambda inputs: executor(program, inputs)
    raise TypeError(f"executor must have prepare(program) or be a callable fn(program, inputs), got {executor!r}")


# ----------------------------------------------------------------------------- trainer


@dataclass
class StepResult:
    step: int
    lr: float
    loss: float
    grad_norm: float
    skipped: bool
    scale: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(vars(self))


class Trainer:
    """Runs a :class:`TrainStep`. ``lr`` is a float or a schedule ``k -> lr``; default: the
    optimizer's own ``lr``."""

    def __init__(self, step: TrainStep, lr: Union[float, Callable[[int], float], None] = None,
                 executor: Any = None, log_every: int = 0, stream=None, seed: Optional[int] = None):
        self.step_def = step
        self.seed = None if seed is None else int(seed)
        self.key_input = getattr(step, "key_input", None)
        if self.key_input is not None and self.key_input not in step.batch_inputs:
            self.key_input = None
        if lr is None:
            lr = step.optimizer.lr
        self.schedule = Constant(float(lr)) if isinstance(lr, (int, float)) else lr
        self.executor = executor
        self._run = _prepare(executor, step.program)
        self.log_every = log_every
        self.stream = stream if stream is not None else sys.stdout
        prog = step.program
        self._carried_types: Dict[str, TensorType] = {v.name: v.type for v in prog.inputs}
        self.state: Dict[str, torch.Tensor] = {n: self._as_declared(n, t) for n, t in step.initial.items()}
        self.step_count = 0
        self.history: List[Dict[str, Any]] = []

    # -- helpers
    def _as_declared(self, name: str, t: torch.Tensor) -> torch.Tensor:
        return t.detach().to(self._carried_types[name].dtype).clone()

    def params(self) -> Dict[str, torch.Tensor]:
        return {n: self.state[n] for n in self.step_def.param_names}

    def lr_at(self, k: int) -> float:
        return float(self.schedule(k))

    def key_at(self, k: int) -> int:
        """The random key of step ``k`` (signed int64 form), with ``seed=`` set."""
        return rng.as_i64(rng.call_key(self.seed, k))

    def _inputs(self, batch: Sequence[torch.Tensor], lr: float) -> List[torch.Tensor]:
        s = self.step_def
        if self.seed is not None and self.key_input is not None and len(batch) == len(s.batch_inputs) - 1:
            at = s.batch_inputs.index(self.key_input)
            batch = list(batch[:at]) + [torch.tensor(self.key_at(self.step_count), dtype=torch.int64)] + list(batch[at:])
        if len(batch) != len(s.batch_inputs):
            raise IRError(f"the training step takes {len(s.batch_inputs)} batch tensors ({s.batch_inputs}), "
                          f"got {len(batch)}")
        return list(batch) + [torch.tensor(lr, dtype=torch.float64)] + [self.state[n] for n in s.carried]

    # -- one step
    def train_step(self, *batch: torch.Tensor) -> StepResult:
        s = self.step_def
        lr = self.lr_at(self.step_count)
        inputs = self._inputs(batch, lr)
        outs = self._run(inputs)
        loss, norm, flag = outs[0], outs[1], bool(outs[2].item())
        if flag and s.scaler is None:
            self._raise_nonfinite(inputs, float(loss.item()), float(norm.item()))
        for n, t in zip(s.carried, outs[N_HEAD:]):
            self.state[n] = self._as_declared(n, t)
        scale = float(self.state["amp.scale"].item()) if s.scaler is not None else None
        r = StepResult(self.step_count, lr, float(loss.item()), float(norm.item()), flag, scale)
        self.history.append(r.to_dict())
        self.step_count += 1
        if self.log_every and self.step_count % self.log_every == 0:
            extra = f" scale {scale:g}" if scale is not None else ""
            extra += " SKIPPED (overflow)" if flag else ""
            print(f"step {r.step} lr {lr:.4g} loss {r.loss:.6g} grad_norm {r.grad_norm:.4g}{extra}",
                  file=self.stream)
        return r

    def fit(self, batches: Iterable[Sequence[torch.Tensor]], steps: Optional[int] = None) -> List[StepResult]:
        out = []
        for i, b in enumerate(batches):
            if steps is not None and i >= steps:
                break
            out.append(self.train_step(*b))
        return out

    def _raise_nonfinite(self, inputs, loss: float, norm: float) -> None:
        prog = self.step_def.program
        if hasattr(self.executor, "transform"):
            prog = self.executor.transform(prog)
        rep = first_nonfinite(prog, inputs)
        what = "the loss is" if not math.isfinite(loss) else "the gradients are"
        msg = (f"training step {self.step_count}: {what} not finite (loss {loss}, grad norm {norm}); "
               f"the step was not applied.\nFirst non finite value: {rep['message']}")
        rep["step"] = self.step_count
        raise NonFiniteError(msg, rep)

    # -- checkpoints
    def checkpoint_bytes(self, loader: Any = None) -> bytes:
        """The checkpoint of this trainer, with ``loader``'s position when given."""
        s = self.step_def
        sched = self.schedule.config() if hasattr(self.schedule, "config") else {"schedule": repr(self.schedule)}
        state = {"step": self.step_count, "step_program_hash": serialize.program_hash(s.program),
                 "params": s.param_names, "state": s.state_names, "config": _jsonable(s.config()),
                 "schedule": _jsonable(sched), "history": self.history,
                 "rng": {"seed": self.seed, "key_input": self.key_input},
                 "loader": loader.state_dict() if loader is not None else None}
        return ckpt.dumps({n: self.state[n] for n in s.carried}, state)

    def save(self, path: str, loader: Any = None) -> str:
        """Write the checkpoint atomically (temporary file, flush, rename)."""
        return serialize.write_bytes(self.checkpoint_bytes(loader), path)

    def load(self, path_or_bytes: Union[str, bytes], loader: Any = None) -> None:
        """Restore params, optimizer state, step count, history and seed from a checkpoint written by
        ``save``; with ``loader``, also its position, so the next batch is the one the uninterrupted run
        would have taken."""
        tensors, meta = ckpt.loads(path_or_bytes)
        s = self.step_def
        want = serialize.program_hash(s.program)
        if meta["step_program_hash"] != want:
            raise DataError("TS1388", "the checkpoint was written by a different training step program (model, "
                            f"loss, optimizer settings or policy differ): checkpoint {meta['step_program_hash'][:16]}, "
                            f"this step {want[:16]}")
        if meta["params"] != s.param_names or meta["state"] != s.state_names:
            raise DataError("TS1388", "checkpoint params or state names differ from this training step")
        for n in s.carried:
            t = tensors.get(n)
            if (t is None or t.dtype != self._carried_types[n].dtype
                    or tuple(t.shape) != tuple(self._carried_types[n].shape)):
                got = "missing" if t is None else f"{t.dtype} {tuple(t.shape)}"
                raise DataError("TS1388", f"checkpoint {n} is {got}, the step declares {self._carried_types[n]}")
        saved_seed = (meta.get("rng") or {}).get("seed")
        if self.key_input is not None and self.seed is not None and saved_seed != self.seed:
            raise DataError("TS1389", f"the checkpoint holds seed {saved_seed}, this trainer {self.seed}: the "
                                        "resumed steps would draw other random numbers")
        loader_state = meta.get("loader")
        if loader is not None and loader_state is None:
            raise DataError("TS1389", "a loader was passed but the checkpoint has no loader position (save it "
                                        "with trainer.save(path, loader=loader))")
        if loader is not None:
            loader.load_state_dict(loader_state)
        for n in s.carried:
            self.state[n] = tensors[n].clone()
        if self.seed is None and saved_seed is not None:
            self.seed = int(saved_seed)
        self.step_count = int(meta["step"])
        self.history = list(meta["history"])


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        items = sorted(x) if isinstance(x, (set, frozenset)) else x
        return [_jsonable(v) for v in items]
    if isinstance(x, torch.dtype):
        return str(x).replace("torch.", "")
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return repr(x)
