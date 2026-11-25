"""`truestar replay RUN --step N`: rerun a spiked step from the last checkpoint and say what caused it.

Three arms start from the checkpoint at or before N, with the batches and learning rates the run used:

* the run's precision, replayed: it must reproduce the logged losses (else TS695, no verdict is
  trusted);
* float64 from the same checkpoint (weights and optimizer state upcast exactly): the same steps in
  exact arithmetic;
* batch N alone in float64 on the checkpoint weights, row by row.

Verdict at step N (a spike is a non finite loss or one above `spike_factor` times the median of the
20 logged losses before it):

* numerics (TS692): the low precision step spikes, float64 does not. Named: the first module
  whose own error (recomputed in float64 on its live inputs) is above 50 low precision epsilons,
  and the step where the two trajectories first part by more than 5%;
* data (TS693): float64 spikes too and batch N is already high on the checkpoint weights. Named:
  the rows that carry the loss;
* optimizer (TS694): float64 spikes too but batch N is normal on the checkpoint weights, so the
  updates in between drove it (learning rate, optimizer state), not the batch and not rounding;
* no spike (TS696) at step N in the replay.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import statistics
import time
from typing import Dict, List, Optional

import torch

from .watch import DTYPES, _cast_batch, forward_backward, load_spec, loss_on, own_errors, train_step

EPS = {torch.float16: 2.0 ** -10, torch.bfloat16: 2.0 ** -7, torch.float32: 2.0 ** -23, torch.float64: 2.0 ** -52}


class ReplayError(ValueError):
    """The run directory cannot replay this step (TS698)."""


@dataclasses.dataclass
class ReplayReport:
    step: int
    checkpoint: int
    dtype: str
    verdict: str  # numerics, data, optimizer, none, not_reproduced
    code: str
    message: str
    baseline: float
    logged: List[Optional[float]]
    lowp: List[float]
    f64: List[float]
    batch_on_checkpoint: float
    first_module: Optional[str] = None
    module_errors: Dict[str, float] = dataclasses.field(default_factory=dict)
    step_errors: List[float] = dataclasses.field(default_factory=list)
    first_gap_step: Optional[int] = None
    rows: List[tuple] = dataclasses.field(default_factory=list)
    seconds: float = 0.0

    @property
    def status(self) -> str:
        return {"none": "ok", "not_reproduced": "error"}.get(self.verdict, "spike")

    def to_text(self) -> str:
        f = lambda v: "nan" if v is None else f"{v:.6g}"  # noqa: E731
        lines = [f"truestar replay: step {self.step} from checkpoint {self.checkpoint} ({self.dtype}), "
                 f"{self.step - self.checkpoint + 1} steps replayed, {self.seconds:.2f} s",
                 f"  baseline loss (median of 20 logged steps before): {self.baseline:.6g}",
                 f"  step {self.step}: logged {f(self.logged[-1])}, replayed {f(self.lowp[-1])}, float64 "
                 f"{f(self.f64[-1])}, batch {self.step} on checkpoint weights in float64 {f(self.batch_on_checkpoint)}",
                 f"verdict: {self.verdict.upper()} [{self.code}] {self.message}"]
        if self.module_errors:
            top = sorted(self.module_errors.items(), key=lambda kv: -kv[1] if math.isfinite(kv[1]) else -math.inf)[:3]
            lines.append("  largest own errors: " + ", ".join(f"{k} {v:.1e}" for k, v in top))
        return "\n".join(lines)

    def to_agent(self, run_dir: Optional[str] = None) -> dict:
        from ..agent import _envelope
        from ..codes import Issue

        clean = lambda v: v if v is None or not isinstance(v, float) or math.isfinite(v) else None  # noqa: E731
        doc = _envelope(self.status, {"ok": 0, "spike": 1, "error": 1}[self.status])
        doc.update({"command": "replay", "run_dir": os.path.abspath(run_dir) if run_dir else None,
                    "step": self.step, "checkpoint": self.checkpoint, "dtype": self.dtype, "verdict": self.verdict,
                    "codes": [self.code], "issues": [Issue(self.code, self.message).as_dict()],
                    "baseline_loss": clean(self.baseline),
                    "losses": {"logged": [clean(v) for v in self.logged], "replayed": [clean(v) for v in self.lowp],
                               "float64": [clean(v) for v in self.f64]},
                    "batch_on_checkpoint_float64": clean(self.batch_on_checkpoint),
                    "first_module": self.first_module,
                    "module_errors": {k: clean(v) for k, v in self.module_errors.items()},
                    "step_errors": [clean(v) for v in self.step_errors],
                    "first_gap_step": self.first_gap_step,
                    "rows": [{"row": r, "loss": clean(v), "share": clean(s)} for r, v, s in self.rows],
                    "seconds": self.seconds})
        acts = []
        if self.verdict == "numerics" and self.first_module:
            acts.append({"kind": "raise_precision", "command": None,
                         "description": f"Run {self.first_module} in float32 (or rescale its input) and replay again."})
        elif self.verdict == "data":
            acts.append({"kind": "inspect_rows", "command": None,
                         "description": f"Inspect rows {', '.join(str(r) for r, _, _ in self.rows)} of batch {self.step}."})
        elif self.verdict == "optimizer":
            acts.append({"kind": "inspect_updates", "command": None,
                         "description": "Check the learning rate and optimizer state of the steps since the checkpoint."})
        doc["next_actions"] = acts
        return doc


def _log(run_dir: str) -> Dict[int, dict]:
    path = os.path.join(run_dir, "watch.jsonl")
    if not os.path.exists(path):
        raise ReplayError(f"{path} not found: not a spike watch run directory")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["step"]] = r
    return out


def _loss_of(rec: dict) -> float:
    return float(rec["loss_text"]) if rec.get("loss") is None else rec["loss"]


def _spiked(v: float, base: float, factor: float) -> bool:
    return (not math.isfinite(v)) or v > factor * base


def replay(run_dir: str, step: int, spec_path: Optional[str] = None, rows_top: int = 3) -> ReplayReport:
    t0 = time.perf_counter()
    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    spec = load_spec(spec_path or meta["spec"])
    dtype = DTYPES[meta["dtype"]]
    log = _log(run_dir)
    if step not in log:
        raise ReplayError(f"step {step} is not in the log (logged steps {min(log)} to {max(log)})")
    ckpts = sorted(int(f[5:-3]) for f in os.listdir(run_dir) if f.startswith("ckpt_") and f.endswith(".pt"))
    usable = [c for c in ckpts if c <= step]
    if not usable:
        raise ReplayError(f"no checkpoint at or before step {step}")
    c = usable[-1]
    if c == step and len(usable) > 1:
        c = usable[-2]  # keep at least one step before N, so a drift has room to show
    batches = []
    for n in range(c, step + 1):
        f = os.path.join(run_dir, "batches", f"{n}.pt")
        if not os.path.exists(f):
            raise ReplayError(f"batch {n} was not saved; replay needs every batch from checkpoint {c} to {step}")
        batches.append(torch.load(f, weights_only=False))
    ck = torch.load(os.path.join(run_dir, f"ckpt_{c}.pt"), weights_only=False)
    loss_fn = spec.get_loss()
    clip = meta.get("clip", 0.0)
    factor = meta.get("spike_factor", 3.0)
    prev = [_loss_of(log[n]) for n in range(max(0, step - 20), step) if n in log]
    prev = [v for v in prev if math.isfinite(v)]
    if not prev:
        raise ReplayError(f"no finite logged loss before step {step} to compare with")
    base = statistics.median(prev)
    lrs = [log[n]["lr"] for n in range(c, step + 1)]
    logged = [log[n].get("loss") for n in range(c, step + 1)]

    def same_weights_f64(m, x, y):
        with torch.no_grad():
            return float(loss_on(loss_fn, copy.deepcopy(m).double()(_cast_batch(x, torch.float64)), y, torch.float64))

    def arm(dt, shadow=False):
        m = spec.get_model().to(dtype)
        m.load_state_dict(ck["model"])
        m = m.to(dt)
        opt = spec.get_optimizer(m.parameters())
        opt.load_state_dict(ck["opt"])  # floating state is cast to the parameters' dtype
        losses, same = [], []
        for (x, y), lr in zip(batches[:-1], lrs[:-1]):
            if shadow:
                same.append(same_weights_f64(m, x, y))
            losses.append(train_step(m, opt, loss_fn, x, y, dt, lr, clip)[0])
        state_before_n = copy.deepcopy(m)
        if shadow:
            same.append(same_weights_f64(m, *batches[-1]))
        losses.append(forward_backward(m, loss_fn, *batches[-1], dt))
        return losses, state_before_n, same

    lowp, lowp_model, same = arm(dtype, shadow=True)
    f64, _, _ = arm(torch.float64)
    # the rounding error of each replayed step: the low precision loss against the same weights and
    # batch in float64 (the trajectories themselves drift apart in any low precision run)
    step_err = [abs(a - b) / max(abs(b), 1e-30) if math.isfinite(a) else math.inf for a, b in zip(lowp, same)]
    m0 = spec.get_model().to(dtype)
    m0.load_state_dict(ck["model"])
    m0 = m0.double()
    xN, yN = batches[-1]
    with torch.no_grad():
        on_ckpt = float(loss_on(loss_fn, m0(_cast_batch(xN, torch.float64)), yN, torch.float64))
    faithful = all((a is None and not math.isfinite(b)) or (a is not None and abs(a - b) <= 1e-6 * max(abs(a), 1e-30))
                   for a, b in zip(logged, lowp))
    gap = next((c + i for i, e in enumerate(step_err) if e > 0.05), None)
    kw = dict(step=step, checkpoint=c, dtype=meta["dtype"], baseline=base, logged=logged, lowp=lowp, f64=f64,
              batch_on_checkpoint=on_ckpt, first_gap_step=gap, step_errors=step_err)
    if not faithful:
        bad = next(c + i for i, (a, b) in enumerate(zip(logged, lowp))
                   if not ((a is None and not math.isfinite(b)) or (a is not None and abs(a - b) <= 1e-6 * max(abs(a), 1e-30))))
        rep = ReplayReport(verdict="not_reproduced", code="TS695",
                           message=f"the replay does not reproduce the logged loss at step {bad} (logged "
                                   f"{logged[bad - c]}, replayed {lowp[bad - c]:.6g}): the run is not deterministic "
                                   "or its state was not all saved, so no verdict is given", **kw)
    elif not _spiked(lowp[-1], base, factor):
        rep = ReplayReport(verdict="none", code="TS696",
                           message=f"step {step} loss {lowp[-1]:.6g} is within {factor:g}x the baseline {base:.6g}",
                           **kw)
    elif not _spiked(f64[-1], base, factor):
        errs = own_errors(lowp_model, xN, dtype)
        bar = 50 * EPS[dtype]
        first = next((n for n, e in errs.items() if e > bar), None)
        msg = (f"the spike vanishes in float64: loss {lowp[-1]:.6g} in {meta['dtype']}, {f64[-1]:.6g} in float64 from "
               f"the same checkpoint and batches")
        if first:
            e = errs[first]
            msg += f"; first split {first}, own error {'inf (non finite output)' if not math.isfinite(e) else f'{e:.1e}'}"
        if gap is not None:
            before = [e for e in step_err[:gap - c]]
            msg += (f"; the step's own rounding error (same weights in float64) is {step_err[gap - c]:.1e} from step "
                    f"{gap}" + (f", at most {max(before):.1e} on the {len(before)} steps before" if before else ""))
        rep = ReplayReport(verdict="numerics", code="TS692", message=msg, first_module=first, module_errors=errs, **kw)
    elif _spiked(on_ckpt, base, factor):
        n_rows = xN.shape[0] if isinstance(xN, torch.Tensor) and xN.dim() else 0
        per = []
        with torch.no_grad():
            for i in range(min(n_rows, 256)):
                per.append(float(loss_on(loss_fn, m0(_cast_batch(xN[i:i + 1], torch.float64)), yN[i:i + 1],
                                         torch.float64)))
        tot = sum(v for v in per if math.isfinite(v)) or 1.0
        med = statistics.median(per) if per else 0.0
        rows = sorted(range(len(per)), key=lambda i: -per[i])
        top = [i for i in rows if per[i] > 5 * med][:max(rows_top, 1)] or rows[:1]
        rep = ReplayReport(verdict="data", code="TS693",
                           message=f"the spike survives float64 ({f64[-1]:.6g}) and batch {step} is already high on the "
                                   f"checkpoint weights ({on_ckpt:.6g} against a baseline of {base:.6g}): rows "
                                   f"{', '.join(str(i) for i in top)} carry {100 * sum(per[i] for i in top) / tot:.0f}% "
                                   "of its loss",
                           rows=[(i, per[i], per[i] / tot) for i in top], **kw)
    else:
        jumps = [c + i for i, lr in enumerate(lrs[:-1]) if lr > 2 * statistics.median(lrs)]
        msg = (f"the spike survives float64 ({f64[-1]:.6g}) but batch {step} is normal on the checkpoint weights "
               f"({on_ckpt:.6g}): the updates since step {c} drove it, not the batch and not rounding")
        if jumps:
            msg += f"; the learning rate jumped at step{'s' if len(jumps) > 1 else ''} {', '.join(map(str, jumps))}"
        rep = ReplayReport(verdict="optimizer", code="TS694", message=msg, **kw)
    rep.seconds = time.perf_counter() - t0
    return rep
