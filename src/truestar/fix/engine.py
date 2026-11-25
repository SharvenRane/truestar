"""`truestar fix`: find, apply and verify a remedy for each op Truestar blames.

Method, per backend that is not clean:

1. Take the op that `truestar check` named (BackendResult.first_divergent_op). If no op is named
   (the module's own code, a device branch, an op only the reference runs) an op remedy cannot
   apply and the backend is reported as not applicable (TS093).
2. For each candidate decomposition in table order (truestar.fix.table):
   a. op level: the single op is run on the backend with the saved reference inputs, with the
      remedy applied, and compared with the float64 answer at the op's own threshold. A candidate
      that fails here is not tried further (cheap filter).
   b. model level: `truestar.check` is rerun on that backend with every remedy so far plus this
      one applied. The candidate wins when the module that was blamed is now OK.
3. If the backend is still not clean and a different op is now blamed, repeat for that op
   (at most `max_rounds` ops).
4. Outcome: resolved (the final check is clean, TS090), improved (an applied remedy measurably
   lowers the error or the flips but the check is not clean, TS091), unresolved (no candidate
   cleared the blamed module, TS092) or not applicable (TS093).
5. Cost: the model's forward on the backend is timed with and without the remedies (median of
   `time_reps` runs after warm up) and the overhead is reported.

Nothing is reported as working without the rerun check in 2b; the attempts and their numbers are
all kept in the report.
"""

from __future__ import annotations

import copy
import json
import os
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ..check import BackendResult, CheckReport, _cast_inputs, _model_dtype, _sync, check
from ..codes import Issue
from ..compare import DIVERGENT, OK, SUSPECT, diff_lists
from .mode import MECHANISMS, Remedy, patched
from .table import candidates

RESOLVED, IMPROVED, UNRESOLVED, NOT_APPLICABLE, CLEAN = (
    "resolved", "improved", "unresolved", "not_applicable", "clean")
_OUTCOME_RANK = {CLEAN: 0, RESOLVED: 1, IMPROVED: 2, NOT_APPLICABLE: 3, UNRESOLVED: 3}
MAX_ROUNDS = 4
# "Measurably better": the final output error falls at least this many times, or real flips drop.
IMPROVEMENT_FACTOR = 2.0

UPSTREAM_DRAFTS = {
    ("aten.linear.default", "mps"): "docs/upstream/mps_linear_bias_DRAFT.md",
}


# ----------------------------------------------------------------------------- records


@dataclass
class Snapshot:
    """What one check said about one backend, reduced to the numbers a remedy changes."""

    status: str
    output_rel_err: Optional[float]
    flips: int
    tie_flips: int
    decisions: int
    module: Optional[str]
    module_rel_err: Optional[float]
    module_threshold: Optional[float]
    op: Optional[str]
    op_rel_err: Optional[float]
    op_threshold: Optional[float]
    codes: List[str]

    @classmethod
    def of(cls, r: BackendResult) -> "Snapshot":
        fd, op = r.first_divergent, r.first_divergent_op
        return cls(
            status=r.status,
            output_rel_err=None if r.output is None else r.output.rel_err,
            flips=r.flips, tie_flips=r.tie_flips, decisions=r.decisions,
            module=None if fd is None else fd.key,
            module_rel_err=None if fd is None or fd.isolated is None else fd.isolated.rel_err,
            module_threshold=None if fd is None else fd.threshold,
            op=None if op is None else op.op,
            op_rel_err=None if op is None else op.isolated.rel_err,
            op_threshold=None if op is None else op.threshold,
            codes=list(r.codes),
        )

    @property
    def real_flips(self) -> int:
        return self.flips - self.tie_flips

    def as_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["real_flips"] = self.real_flips
        return d


@dataclass
class Attempt:
    remedy: Remedy
    op_rel_err: Optional[float] = None  # the op alone, with the remedy, vs float64
    op_threshold: Optional[float] = None
    op_ok: bool = False
    skipped: str = ""  # why the candidate was not run (guard, error)
    module_status: Optional[str] = None  # the blamed module in the rerun check
    module_rel_err: Optional[float] = None
    model_status: Optional[str] = None  # the backend in the rerun check
    output_rel_err: Optional[float] = None
    hits: int = 0  # how many calls the remedy replaced during the rerun check
    won: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {"remedy": self.remedy.as_dict(), "op_rel_err": self.op_rel_err,
                "op_threshold": self.op_threshold, "op_ok": self.op_ok,
                "skipped": self.skipped or None, "module_status": self.module_status,
                "module_rel_err": self.module_rel_err, "model_status": self.model_status,
                "output_rel_err": self.output_rel_err, "hits": self.hits, "won": self.won}


@dataclass
class Cost:
    baseline_ms: float
    remedied_ms: float
    reps: int

    @property
    def overhead(self) -> float:
        return self.remedied_ms / self.baseline_ms - 1.0 if self.baseline_ms > 0 else float("nan")

    def as_dict(self) -> Dict[str, Any]:
        return {"baseline_ms": self.baseline_ms, "remedied_ms": self.remedied_ms,
                "overhead_pct": 100.0 * self.overhead, "reps": self.reps}


@dataclass
class BackendFix:
    backend: str
    device: str
    device_type: str
    outcome: str
    before: Snapshot
    after: Snapshot
    applied: List[Remedy] = field(default_factory=list)
    attempts: List[Attempt] = field(default_factory=list)
    cost: Optional[Cost] = None
    reason: str = ""
    upstream: Dict[str, Optional[str]] = field(default_factory=dict)
    seconds: float = 0.0  # time spent searching and verifying
    issues: List[Issue] = field(default_factory=list)

    def add(self, code: str, message: str) -> None:
        self.issues.append(Issue(code, message, self.backend))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend, "device": self.device, "device_type": self.device_type,
            "outcome": self.outcome, "reason": self.reason or None,
            "before": self.before.as_dict(), "after": self.after.as_dict(),
            "remedies": [r.as_dict() for r in self.applied],
            "attempts": [a.as_dict() for a in self.attempts],
            "cost": None if self.cost is None else self.cost.as_dict(),
            "upstream": dict(self.upstream), "search_seconds": self.seconds,
            "codes": [i.code for i in self.issues], "issues": [i.as_dict() for i in self.issues],
        }


@dataclass
class FixReport:
    model_name: str
    mechanism: str
    before: CheckReport
    backends: List[BackendFix]
    notes: List[str] = field(default_factory=list)

    def remedies(self) -> List[Remedy]:
        """Every remedy that passed verification (resolved or measurably improved backends)."""
        out: List[Remedy] = []
        for b in self.backends:
            if b.outcome in (RESOLVED, IMPROVED):
                out.extend(r for r in b.applied if r not in out)
        return out

    @property
    def status(self) -> str:
        if not self.backends:
            return CLEAN
        return max((b.outcome for b in self.backends), key=lambda o: _OUTCOME_RANK[o])

    @property
    def issues(self) -> List[Issue]:
        return [i for b in self.backends for i in b.issues]

    @property
    def codes(self) -> List[str]:
        out: List[str] = []
        for i in self.issues:
            if i.code not in out:
                out.append(i.code)
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {"model": self.model_name, "mechanism": self.mechanism, "status": self.status,
                "codes": self.codes, "notes": list(self.notes),
                "backends": [b.as_dict() for b in self.backends],
                "check_before": self.before.as_dict()}

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, default=str)

    def to_text(self) -> str:
        from .output import fix_text

        return fix_text(self)

    def to_agent(self, model_file: Optional[str] = None) -> Dict[str, Any]:
        from .output import fix_agent

        return fix_agent(self, model_file)


# ----------------------------------------------------------------------------- verification steps


def verify_op(remedy: Remedy, op_result, device: str, mechanism: str = "mode"
              ) -> Tuple[Optional[float], bool, str]:
    """Run the blamed op alone on `device` with its saved inputs and the remedy applied.

    Returns (relative error vs float64, ok at the op's own threshold, skip reason)."""
    from ..opdrill import _call  # private helper, same isolation the drilldown uses

    e = remedy.entry
    args, kwargs = op_result.args, op_result.kwargs
    why = e.unsupported(args, kwargs)
    if why:
        return None, False, why
    func = resolve_op(remedy.op)
    try:
        with patched([remedy], mechanism):
            got = _call(func, args, kwargs, device)
        _sync(device)
    except Exception as exc:
        return None, False, f"{type(exc).__name__}: {exc}"
    d = diff_lists(got, op_result.reference)
    return d.rel_err, d.status(op_result.threshold) == OK, ""


def resolve_op(name: str):
    ns, packet, overload = name.split(".")
    return getattr(getattr(getattr(torch.ops, ns), packet), overload)


def _layer(r: BackendResult, key: Optional[str]):
    if key is None:
        return None
    for l in r.layers:
        if l.key == key:
            return l
    return None


def time_forward(model: nn.Module, args: Tuple, kwargs: Dict, device: str, dtype: torch.dtype,
                 remedies: Sequence[Remedy] = (), mechanism: str = "mode", reps: int = 20,
                 warmup: int = 3) -> float:
    """Median forward time in milliseconds on `device`, with `remedies` applied if any."""
    m = copy.deepcopy(model).to(device).eval()
    a, k = _cast_inputs(args, device, dtype), _cast_inputs(kwargs, device, dtype)
    times = []
    with torch.no_grad(), patched(list(remedies), mechanism):
        for i in range(warmup + reps):
            _sync(device)
            t0 = time.perf_counter()
            m(*a, **k)
            _sync(device)
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000.0)
    del m
    return statistics.median(times)


# ----------------------------------------------------------------------------- main entry


def fix(model: nn.Module, inputs: Union[torch.Tensor, Sequence[Any]],
        kwargs: Optional[Dict[str, Any]] = None, backends: Optional[List[str]] = None, *,
        report: Optional[CheckReport] = None, mechanism: str = "mode",
        repro_dir: Optional[str] = None, time_reps: int = 20, max_rounds: int = MAX_ROUNDS,
        model_name: Optional[str] = None, table=None) -> FixReport:
    """Find and verify remedies for every backend where `truestar check` blames an ATen op.

    Args:
        model, inputs, kwargs, backends: as for truestar.check.
        report: an existing CheckReport for the same model and inputs (else check runs first).
        mechanism: "mode" (dispatch mode, default) or "library" (process wide kernel override).
        repro_dir: where check writes repros (the upstream report paths); default a temp folder.
        time_reps: forward runs timed with and without the remedy (0 skips timing).
        max_rounds: at most this many different ops are remedied per backend.
        table: op -> ordered decompositions, replacing the built in table (truestar.fix.table).
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS}")
    sims = [b for b in (backends or []) if b.strip().lower().startswith("sim:")] + [
        r.backend.name for r in (report.results if report is not None else []) if r.backend.sim]
    if sims:
        from ..backends import BackendUnavailable

        # Remedies are verified and timed on the backend's device; for a simulated backend that
        # would be the plain CPU, so refuse rather than verify on the wrong arithmetic.
        raise BackendUnavailable(f"truestar fix does not support simulated backends ({', '.join(sims)}); "
                                 "check them with truestar check --backends", sims)
    args = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    kwargs = dict(kwargs or {})
    name = model_name or type(model).__name__
    repro_dir = repro_dir or tempfile.mkdtemp(prefix="truestar_fix_")
    before = report or check(model, args, kwargs=kwargs, backends=backends,
                             repro_dir=os.path.join(repro_dir, "before"), model_name=name)
    dtype = _model_dtype(model)
    out = FixReport(name, mechanism, before, [])
    for r in before.results:
        if r.status == OK:
            continue
        out.backends.append(_fix_backend(model, args, kwargs, r, dtype, mechanism, repro_dir,
                                         time_reps, max_rounds, name, table))
    return out


def _fix_backend(model, args, kwargs, r0: BackendResult, dtype, mechanism, repro_dir, time_reps,
                 max_rounds, name, table=None) -> BackendFix:
    t_start = time.perf_counter()
    b = r0.backend
    dt = torch.device(b.device).type
    snap0 = Snapshot.of(r0)
    bf = BackendFix(b.name, b.device, dt, UNRESOLVED, snap0, snap0)
    current = r0
    applied: List[Remedy] = []
    seen_ops: List[str] = []

    def upstream(r: BackendResult) -> None:
        bf.upstream = {"op_repro": r.op_repro_path, "repro": r.repro_path,
                       "draft": UPSTREAM_DRAFTS.get((r.first_divergent_op.op, dt))
                       if r.first_divergent_op is not None else None}

    upstream(r0)
    if r0.error:
        bf.outcome, bf.reason = NOT_APPLICABLE, f"the model fails to run on {b.name}: {r0.error}"
    for rnd in range(max_rounds):
        if current.status == OK or r0.error:
            break
        op = current.first_divergent_op
        if op is None:
            what = (f"module {current.first_divergent.key}" if current.first_divergent is not None
                    else "no module (TS012)")
            bf.reason = (f"no single ATen op is blamed ({what}): the cause is the module's own "
                         "code or work outside any module, which an op remedy cannot replace")
            bf.outcome = NOT_APPLICABLE if not applied else bf.outcome
            break
        if op.op in seen_ops:
            bf.reason = f"{op.op} is still blamed with its remedy applied"
            break
        seen_ops.append(op.op)
        cands = candidates(op.op, dt, mechanism, table)
        if not cands:
            bf.reason = f"no remedy for {op.op} on {dt} in the table"
            upstream(current)
            break
        blamed_key = current.first_divergent.key if current.first_divergent is not None else None
        winner = None
        for cand in cands:
            rem = Remedy(op.op, dt, cand.name, cand if table is not None else None)
            att = Attempt(rem, op_threshold=op.threshold)
            bf.attempts.append(att)
            att.op_rel_err, att.op_ok, att.skipped = verify_op(rem, op, b.device, mechanism)
            if not att.op_ok:
                continue
            with patched(applied + [rem], mechanism) as p:
                rep = check(model, args, kwargs=kwargs, backends=[b.name],
                            repro_dir=os.path.join(repro_dir, f"round{rnd + 1}_{cand.name}"),
                            model_name=name)
            att.hits = int(p.hits.get(op.op, 0))
            rr = rep.results[0]
            lay = _layer(rr, blamed_key)
            att.module_status = None if lay is None else lay.status
            att.module_rel_err = (None if lay is None or lay.isolated is None
                                  else lay.isolated.rel_err)
            att.model_status = rr.status
            att.output_rel_err = None if rr.output is None else rr.output.rel_err
            # Wins only if it fired, the blamed module is clean, and the model is clean or what
            # is still wrong is a different op (a later round). The same op blamed elsewhere
            # means the remedy is not right for every call.
            still_same = (rr.first_divergent_op is not None and rr.first_divergent_op.op == op.op)
            if (att.hits > 0 and lay is not None and lay.status == OK
                    and (rr.status == OK or not still_same)):
                att.won = True
                winner = (rem, rr)
                break
        if winner is None:
            bf.reason = (f"none of {len(cands)} candidate(s) for {op.op} made module "
                         f"{blamed_key} clean")
            upstream(current)
            break
        applied.append(winner[0])
        current = winner[1]
        if current.status != OK:
            upstream(current)

    bf.applied = applied
    bf.after = Snapshot.of(current)
    if not r0.error and applied:
        if current.status == OK:
            bf.outcome, bf.reason = RESOLVED, ""
        elif _better(snap0, bf.after):
            bf.outcome = IMPROVED
        elif bf.outcome != NOT_APPLICABLE:
            bf.outcome = UNRESOLVED
    if applied and time_reps > 0:
        try:
            base = time_forward(model, args, kwargs, b.device, dtype, (), mechanism, time_reps)
            rem_ms = time_forward(model, args, kwargs, b.device, dtype, applied, mechanism,
                                  time_reps)
            bf.cost = Cost(base, rem_ms, time_reps)
        except Exception as exc:  # timing must never hide the verified result
            bf.reason = (bf.reason + "; " if bf.reason else "") + f"timing failed: {exc}"
    bf.seconds = time.perf_counter() - t_start
    _codes(bf)
    return bf


def _better(before: Snapshot, after: Snapshot) -> bool:
    if after.real_flips < before.real_flips:
        return True
    a, b = after.output_rel_err, before.output_rel_err
    return a is not None and b is not None and a * IMPROVEMENT_FACTOR <= b


def _e(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.2e}"


def _codes(bf: BackendFix) -> None:
    rems = "; ".join(str(r) for r in bf.applied)
    cost = ""
    if bf.cost is not None:
        cost = (f"; forward {bf.cost.baseline_ms:.3f} ms -> {bf.cost.remedied_ms:.3f} ms "
                f"({100 * bf.cost.overhead:+.1f}%)")
    if bf.outcome == RESOLVED:
        bf.add("TS090", f"verified: with {rems} the rerun check on {bf.backend} is clean (output "
                        f"error {_e(bf.before.output_rel_err)} -> {_e(bf.after.output_rel_err)})"
                        + cost)
    elif bf.outcome == IMPROVED:
        bf.add("TS091", f"with {rems} the output error goes {_e(bf.before.output_rel_err)} -> "
                        f"{_e(bf.after.output_rel_err)} and real flips {bf.before.real_flips} -> "
                        f"{bf.after.real_flips}, but the check is still {bf.after.status}: "
                        f"{bf.reason}" + cost)
    elif bf.outcome == UNRESOLVED:
        bf.add("TS092", f"unresolved on {bf.backend}: {bf.reason}. Report upstream with "
                        f"{bf.upstream.get('op_repro') or bf.upstream.get('repro') or 'the repro'}")
    elif bf.outcome == NOT_APPLICABLE:
        bf.add("TS093", f"{bf.reason}")
