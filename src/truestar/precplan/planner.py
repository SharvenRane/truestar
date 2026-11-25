"""Per layer precision planner: as fast as possible on this GPU, decision flips under a budget.

Pipeline (docs/PRECISION.md):

1. Float64 CPU reference on the calibration and held out inputs; a decision is the argmax of the
   last output dim per row (a class, or a token per position).
2. Units and tie groups (units.py): what can get its own precision, found by running the model.
3. Range (truestar range, float64 IR): a unit whose values leave float16 or bfloat16 range (or
   lose their digits there) never gets that activation dtype. Proven by analysis, not by luck on
   the calibration set.
4. Candidates per unit, each measured on the calibration inputs with only that unit (or tie
   group) changed: output error against float64, decision flips, time per call on this GPU.
   Quantized candidates carry the two verdicts of truestar.quant: (a) the kernel against the
   exact dequantized float64 product of its own operands (a failure removes the candidate as a
   kernel bug, it is never counted as format loss) and (b) the format's own loss from float64
   references only.
5. Search. One knob: a per unit error threshold tau. For a tau every unit may use any candidate
   whose own error is at most tau; a dynamic program over the units in execution order then
   picks, per unit, the candidate that minimizes measured time plus the casts between activation
   dtypes. Plans get faster as tau grows, so the largest tau whose plan keeps calibration flips
   within budget is found by bisection: O(log n) verification runs, not a sweep. A short greedy
   pass then tries the largest remaining time savings one at a time, each verified.
6. Verify. The candidate plans are timed end to end and the fastest one that keeps the budget on
   the calibration inputs is run on the held out inputs. If the held out budget fails, the
   calibration budget is tightened and the search runs again. The held out inputs never steer the
   search; they only pass or fail a finished plan.
"""

from __future__ import annotations

import copy
import hashlib
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ..codes import Issue
from . import units as U
from .units import ACCURACY_RANK, PLAIN, QUANT, STORAGE, PlanError, Wrapped

NOISE_REL = 0.05     # timing differences below this share (or NOISE_ABS) count as equal speed
NOISE_ABS_US = 2.0


# ----------------------------------------------------------------------------- inputs and outputs


def _as_list(inputs) -> List[Tuple]:
    if inputs is None:
        return []
    if isinstance(inputs, torch.Tensor):
        return [(inputs,)]
    if isinstance(inputs, tuple) and all(isinstance(t, torch.Tensor) for t in inputs):
        return [inputs]
    return [(x,) if isinstance(x, torch.Tensor) else tuple(x) for x in inputs]


def _split(batches: List[Tuple]) -> Tuple[List[Tuple], List[Tuple]]:
    """Half for calibration, half held out: by batch when there are several, else by rows."""
    if len(batches) >= 2:
        h = len(batches) // 2
        return batches[:h], batches[h:]
    (b,) = batches
    n = next(t.shape[0] for t in b if isinstance(t, torch.Tensor))
    if n < 2:
        raise PlanError("held out inputs are needed: pass holdout=..., or calibration inputs with "
                        "at least 2 rows to split")
    h = n // 2
    return ([tuple(t[:h] if isinstance(t, torch.Tensor) else t for t in b)],
            [tuple(t[h:] if isinstance(t, torch.Tensor) else t for t in b)])


def _to(batch: Tuple, device, dtype=None) -> Tuple:
    out = []
    for t in batch:
        if isinstance(t, torch.Tensor):
            t = t.to(device)
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype)
        out.append(t)
    return tuple(out)


def _slice(batch: Tuple, n: int) -> Tuple:
    return tuple(t[:n] if isinstance(t, torch.Tensor) and t.dim() > 0 else t for t in batch)


def _logits(out) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        t = out
    elif hasattr(out, "logits"):
        t = out.logits
    elif isinstance(out, (tuple, list)):
        t = next(o for o in out if isinstance(o, torch.Tensor))
    elif isinstance(out, dict):
        t = next(o for o in out.values() if isinstance(o, torch.Tensor))
    else:
        raise PlanError(f"cannot read decisions from a {type(out).__name__} output")
    return t.reshape(-1, t.shape[-1]) if t.dim() > 1 else t.reshape(-1, 1)


@dataclass
class Score:
    flips: int
    decisions: int
    rel_err: float
    nonfinite_rows: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {"flips": self.flips, "decisions": self.decisions,
                "rel_err": self.rel_err if math.isfinite(self.rel_err) else None,
                "nonfinite_rows": self.nonfinite_rows}


def score(outs: Sequence[torch.Tensor], refs: Sequence[torch.Tensor]) -> Score:
    flips = rows = bad = 0
    num = den = 0.0
    for o, r in zip(outs, refs):
        o = o.to(r.device, torch.float64)
        fin = torch.isfinite(o).all(1)
        bad += int((~fin).sum())
        if o.shape[1] > 1:
            flips += int(((o.argmax(1) != r.argmax(1)) | ~fin).sum())
        else:
            flips += int((~fin).sum())
        rows += o.shape[0]
        num += float(((o - r) ** 2).sum())
        den += float((r ** 2).sum())
    rel = math.sqrt(num / den) if den > 0 else (0.0 if num == 0 else math.inf)
    if not math.isfinite(rel) or bad:
        rel = math.inf
    return Score(flips, rows, rel, bad)


# ----------------------------------------------------------------------------- budget


@dataclass
class Budget:
    """`max_flips` as a count on the calibration decisions, or as a share ("0.5%")."""

    value: float
    share: bool

    @classmethod
    def parse(cls, v: Union[int, float, str]) -> "Budget":
        if isinstance(v, str):
            s = v.strip()
            if s.endswith("%"):
                x = float(s[:-1]) / 100.0
                if not 0 <= x <= 1:
                    raise ValueError(f"--max-flips {v}: a share must be between 0% and 100%")
                return cls(x, True)
            v = float(s)
        if v < 0 or (float(v) != int(v)):
            raise ValueError(f"--max-flips {v}: a count must be a whole number >= 0 (or a share like 0.5%)")
        return cls(float(v), False)

    def allowed(self, decisions: int, cal_decisions: int) -> int:
        rate = self.value if self.share else self.value / max(cal_decisions, 1)
        return int(math.floor(rate * decisions + 1e-9))

    def text(self) -> str:
        return f"{100 * self.value:g}%" if self.share else f"{int(self.value)}"


# ----------------------------------------------------------------------------- report rows


@dataclass
class Cand:
    precision: str
    status: str = "ok"        # ok, unavailable, kernel_bug, range, fp32_only_group, not_requested
    reason: str = ""
    time_us: Optional[float] = None
    err: Optional[float] = None
    flips: Optional[int] = None
    kernel: Optional[Dict[str, Any]] = None        # verdict (a)
    format_loss: Optional[Dict[str, Any]] = None   # verdict (b)

    def as_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        if d["err"] is not None and not math.isfinite(d["err"]):
            d["err"] = None
        return d


@dataclass
class UnitRow:
    name: str
    module_type: str
    kind: str
    group: int
    order: int
    calls: int
    cands: Dict[str, Cand]
    chosen: str = "fp32"
    why: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "module_type": self.module_type, "kind": self.kind,
                "group": self.group, "order": self.order, "calls": self.calls,
                "chosen": self.chosen, "why": self.why,
                "candidates": {p: c.as_dict() for p, c in self.cands.items()}}


@dataclass
class Measured:
    name: str
    assign: Optional[Dict[str, str]]
    cal: Score
    hold: Optional[Score] = None
    latency_us: Optional[float] = None
    latency_range: Tuple[float, float] = (0.0, 0.0)
    predicted_us: Optional[float] = None
    eager_us: Optional[float] = None       # wall time per call without a CUDA graph

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "calibration": self.cal.as_dict(),
                "held_out": self.hold.as_dict() if self.hold else None,
                "latency_us": self.latency_us, "latency_range_us": list(self.latency_range),
                "eager_us": self.eager_us, "predicted_us": self.predicted_us}


@dataclass
class PrecisionPlan:
    model_name: str
    device: str
    gpu: str
    torch_version: str
    budget: Budget
    allowed_cal: int
    allowed_hold: int
    assign: Dict[str, str]
    units: List[UnitRow]
    groups: List[List[str]]
    result: Measured
    baselines: List[Measured]
    search: List[Dict[str, Any]]
    verifications: int
    seconds: Dict[str, float]
    fingerprint: Dict[str, Any]
    issues: List[Issue]
    notes: List[str] = field(default_factory=list)
    met: bool = True
    tightened: int = 0

    @property
    def codes(self) -> List[str]:
        out: List[str] = []
        for i in self.issues:
            if i.code not in out:
                out.append(i.code)
        return out

    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for p in self.assign.values():
            c[p] = c.get(p, 0) + 1
        return c

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": "truestar.precision_plan/1",
            "model": self.model_name, "device": self.device, "gpu": self.gpu,
            "torch_version": self.torch_version,
            "max_flips": self.budget.text(), "allowed_flips_calibration": self.allowed_cal,
            "allowed_flips_held_out": self.allowed_hold, "met": self.met,
            "tightened_rounds": self.tightened,
            "plan": dict(self.assign), "counts": self.counts(),
            "result": self.result.as_dict(),
            "baselines": [b.as_dict() for b in self.baselines],
            "units": [u.as_dict() for u in self.units],
            "groups": [g for g in self.groups if len(g) > 1],
            "search": self.search, "verification_runs": self.verifications,
            "seconds": self.seconds, "fingerprint": self.fingerprint,
            "issues": [i.as_dict() for i in self.issues], "codes": self.codes,
            "notes": self.notes,
        }

    def save(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=2, allow_nan=False)

    def to_text(self, top: int = 60) -> str:
        from .output import plan_text

        return plan_text(self, top)

    def to_agent(self, model_file: Optional[str] = None) -> Dict[str, Any]:
        from .output import plan_agent

        return plan_agent(self, model_file)


# ----------------------------------------------------------------------------- fingerprint


def fingerprint(model: nn.Module) -> Dict[str, Any]:
    """Names, shapes and a checksum of the weights: a plan applies only to the model it was made
    for."""
    h = hashlib.sha256()
    shapes = {}
    for n, p in model.state_dict().items():
        shapes[n] = list(p.shape)
        h.update(n.encode())
        h.update(str(tuple(p.shape)).encode())
        if p.is_floating_point() and p.numel():
            t = p.detach().cpu().double()
            h.update(f"{float(t.sum()):.12e},{float(t.abs().sum()):.12e}".encode())
    return {"sha256": h.hexdigest(), "tensors": len(shapes)}


# ----------------------------------------------------------------------------- timing


def time_us(fn: Callable[[], Any], reps: int = 20, rounds: int = 3, warmup: int = 3) -> Tuple[float, List[float]]:
    """GPU time per call in microseconds: CUDA events around each call, median per round, the
    smallest round median returned (other jobs on a shared GPU only ever add time)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    meds = []
    for _ in range(rounds):
        ts = []
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e) * 1000.0)
        meds.append(statistics.median(ts))
    return min(meds), meds


def _graph_time(fn: Callable[[], Any], k: int, repeats: int, spin_s: float = 0.01) -> float:
    """k calls recorded into one CUDA graph, replayed `repeats` times; median GPU time per call.
    Captured with capture_begin and capture_end directly: the torch.cuda.graph context also
    empties the allocator cache (and may collect garbage) per capture, which made timing every
    candidate take minutes on the larger models."""
    cur = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(cur)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        fn()
        fn()
        side.synchronize()
        g.capture_begin()
        try:
            for _ in range(k):
                fn()
        finally:
            g.capture_end()
    cur.wait_stream(side)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < spin_s:  # clocks up before timing; synchronized, so the
        g.replay()                             # spin is spin_s of GPU work, not a queued backlog
        cur.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    vals = []
    for _ in range(repeats):
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        vals.append(start.elapsed_time(end) * 1e3 / k)
    del g
    return statistics.median(vals)


def gpu_us(fn: Callable[[], Any], k: int = 10, repeats: int = 7) -> Tuple[float, str]:
    """GPU time per call from k calls replayed in one CUDA graph, so Python launch cost is not
    counted. Falls back to event timing, and says so, when fn cannot be captured."""
    try:
        return _graph_time(fn, k, repeats), "graph"
    except Exception:
        torch.cuda.synchronize()
        t, _ = time_us(fn, reps=max(10, repeats * 3), rounds=3)
        return t, "events"


# ----------------------------------------------------------------------------- the planner


class _Planner:
    def __init__(self, model, cal, hold, budget, precisions, device, reps, rounds, refine,
                 use_range, format_loss, range_rows, log):
        self.model = model
        self.cal_cpu, self.hold_cpu = cal, hold
        self.budget = budget
        self.precisions = tuple(precisions)
        self.device = device
        self.reps, self.rounds = reps, rounds
        self.refine = refine
        self.use_range = use_range
        self.do_format_loss = format_loss
        self.range_rows = range_rows
        self.log = log
        self.issues: List[Issue] = []
        self.notes: List[str] = []
        self.secs: Dict[str, float] = {}
        self.verifications = 0
        self.search_log: List[Dict[str, Any]] = []
        self._cache: Dict[Tuple, Score] = {}
        self.event_timed: List[str] = []
        self.unstable = 0

    # -- runs

    def _run(self, m: nn.Module, batches: Sequence[Tuple]) -> List[torch.Tensor]:
        outs = []
        with torch.no_grad(), U.strict_fp32():
            for b in batches:
                outs.append(_logits(m(*b)).detach().double())
        return outs

    def run_assign(self, assign: Dict[str, str], which: str = "cal") -> Score:
        key = (which, tuple(sorted(assign.items())))
        if key in self._cache:
            return self._cache[key]
        self.w.set(assign)
        batches = self.cal if which == "cal" else self.hold
        refs = self.ref_cal if which == "cal" else self.ref_hold
        s = score(self._run(self.w.model, batches), refs)
        self._cache[key] = s
        return s

    # -- steps

    def reference(self):
        t = time.perf_counter()
        ref = copy.deepcopy(self.model).cpu().double().eval()
        with torch.no_grad():
            self.ref_cal_cpu = [_logits(ref(*_to(b, "cpu", torch.float64))).double() for b in self.cal_cpu]
            self.ref_hold_cpu = [_logits(ref(*_to(b, "cpu", torch.float64))).double() for b in self.hold_cpu]
        del ref
        self.ref_cal = [r.to(self.device) for r in self.ref_cal_cpu]
        self.ref_hold = [r.to(self.device) for r in self.ref_hold_cpu]
        self.n_cal = sum(r.shape[0] for r in self.ref_cal)
        self.n_hold = sum(r.shape[0] for r in self.ref_hold)
        self.allow_cal = self.budget.allowed(self.n_cal, self.n_cal)
        self.allow_hold = self.budget.allowed(self.n_hold, self.n_cal)
        self.cal = [_to(b, self.device) for b in self.cal_cpu]
        self.hold = [_to(b, self.device) for b in self.hold_cpu]
        self.secs["reference"] = time.perf_counter() - t

    def units(self):
        t = time.perf_counter()
        full = self.cal[0]
        probe = _slice(full, 2)
        self.w: Wrapped = U.wrap(self.model, self.device, lambda m: m(*full))
        U.find_groups(self.w, lambda m: m(*probe))
        for n in self.w.merged:
            self.notes.append(f"{n} reads its children's weights directly, so it is planned as one unit")
        tied = [(i, g) for i, g in enumerate(self.w.groups) if len(g) > 1]
        for i, g in tied:
            self.issues.append(Issue("TS667", f"{len(g)} units share one activation dtype because "
                                              f"code between them does not promote mixed dtypes "
                                              f"({self.w.tie_reasons.get(i, '')}): "
                                              + ", ".join(g[:6]) + (" ..." if len(g) > 6 else "")))
        for i, why in self.w.fp32_only_groups.items():
            self.issues.append(Issue("TS667", "float32 only: " + why))
        self.secs["units"] = time.perf_counter() - t

    def range(self):
        """Activation dtypes each unit may not use, from truestar range on float64."""
        t = time.perf_counter()
        self.forbid: Dict[str, Dict[torch.dtype, str]] = {n: {} for n in self.w.units}
        stor = {STORAGE[p] for p in self.precisions} - {torch.float32}
        if not self.use_range or not stor:
            self.secs["range"] = time.perf_counter() - t
            return
        from ..rangecheck.analysis import range_check
        from .. import ir

        m = copy.deepcopy(self.model).cpu().float().eval()
        calib = [_slice(_to(self.cal_cpu[0], "cpu"), self.range_rows)]  # one batch: cost, not coverage, is the limit
        try:
            with torch.no_grad():
                prog = ir.import_module(m, calib[0])
        except Exception as exc:
            self.issues.append(Issue("TS668", f"range check could not import the model "
                                              f"({type(exc).__name__}: {str(exc).splitlines()[0][:200]}); "
                                              "range safety rests on measurement and verification only"))
            self.secs["range"] = time.perf_counter() - t
            return
        names = list(self.w.units)
        hits: Dict[str, List[str]] = {}
        for dt, target in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
            if dt not in stor:
                continue
            try:
                rep = range_check(m, calib[0], target=target, calibration=calib[1:] or None,
                                  remedies=False, program=prog)
            except Exception as exc:
                self.issues.append(Issue("TS668", f"range check for {target} failed "
                                                  f"({type(exc).__name__}: {str(exc)[:200]})"))
                continue
            for s in list(rep.sites) + list(rep.info):
                if s.kind == "hidden_in_fused":
                    continue  # fused kernels keep these internals in float32
                if s.kind == "underflow" and s.effect != "harmful":
                    continue
                mod = s.where.module
                if not mod and s.node is None:  # a saturated parameter: map by its name
                    mod = next((n for n in names if s.value.startswith(n + ".")), None)
                    if mod is None:
                        continue
                hit = [n for n in names if n == mod or mod.startswith(n + ".")] \
                    or [n for n in names if mod == "" or n.startswith(mod + ".")]
                what = (f"{target}: {s.kind} at {s.where.short()} (peak {s.peak:.3g}, "
                        f"{s.headroom_bits:+.2f} bits"
                        + (f", {s.precision_bits:.1f} correct bits" if s.precision_bits is not None else "")
                        + ")")
                for n in hit:
                    if dt not in self.forbid[n]:
                        self.forbid[n][dt] = what
                        hits.setdefault(target, []).append(n)
        for target, ns in hits.items():
            self.issues.append(Issue("TS663", f"range check keeps {len(ns)} units out of {target}: "
                                              + ", ".join(ns[:8]) + (" ..." if len(ns) > 8 else "")))
        self.secs["range"] = time.perf_counter() - t

    def candidates(self):
        """Error, flips, kernel verdict and format loss of every candidate, one change at a time."""
        t = time.perf_counter()
        w = self.w
        self.rows: Dict[str, UnitRow] = {}
        for n, u in w.units.items():
            cands = {}
            for p in U.candidates_for(u.kind):
                c = Cand(p)
                if p not in self.precisions:
                    c.status, c.reason = "not_requested", "not in the requested precisions"
                elif w.group_of[n] in w.fp32_only_groups and p != "fp32":
                    c.status, c.reason = "fp32_only_group", w.fp32_only_groups[w.group_of[n]]
                elif STORAGE[p] in self.forbid[n]:
                    c.status, c.reason = "range", self.forbid[n][STORAGE[p]]
                cands[p] = c
            self.rows[n] = UnitRow(n, type(u.base).__name__, u.kind, w.group_of[n], u.order or 0,
                                   0, cands)
        base = self.run_assign({})
        self.base_err = base.rel_err
        self.verifications += 1
        # group level: every member at the plain precision of a 16 bit dtype
        self.group_err: Dict[Tuple[int, torch.dtype], Score] = {}
        for gi, g in enumerate(w.groups):
            for dt in (torch.bfloat16, torch.float16):
                p = PLAIN[dt]
                if any(self.rows[n].cands.get(p) is None or self.rows[n].cands[p].status != "ok"
                       for n in g):
                    # every member must allow the dtype; a member that cannot blocks the group
                    blocker = next((self.rows[n].cands.get(p) for n in g
                                    if self.rows[n].cands.get(p) is not None
                                    and self.rows[n].cands[p].status != "ok"), None)
                    if blocker is not None and len(g) > 1:
                        for n in g:
                            c = self.rows[n].cands.get(p)
                            if c is not None and c.status == "ok":
                                c.status, c.reason = blocker.status, f"tied group member: {blocker.reason}"
                    continue
                try:
                    s = self.run_assign({n: p for n in g})
                except Exception as exc:
                    for n in g:
                        self.rows[n].cands[p].status = "unavailable"
                        self.rows[n].cands[p].reason = f"{type(exc).__name__}: {str(exc)[:160]}"
                    continue
                self.verifications += 1
                self.group_err[(gi, dt)] = s
                for n in g:
                    c = self.rows[n].cands[p]
                    c.err, c.flips = s.rel_err, s.flips
        # unit level: tf32 and quantized candidates, with the group at that dtype's plain precision
        for n, row in self.rows.items():
            gi = row.group
            g = w.groups[gi]
            for p, c in row.cands.items():
                if p in ("fp32", "bf16", "fp16") or c.status != "ok":
                    continue
                dt = STORAGE[p]
                if dt != torch.float32 and (gi, dt) not in self.group_err:
                    c.status, c.reason = "range", f"its activation dtype is unavailable to the group"
                    continue
                assign = {} if dt == torch.float32 else {m: PLAIN[dt] for m in g}
                assign[n] = p
                try:
                    s = self.run_assign(assign)
                except Exception as exc:
                    c.status, c.reason = "unavailable", f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
                    self.issues.append(Issue("TS665", f"{n}: {p} cannot run here: {c.reason}"))
                    continue
                self.verifications += 1
                c.err, c.flips = s.rel_err, s.flips
                if p in QUANT:
                    self._kernel_verdict(n, p, c)
        fp32_err = self.base_err
        for row in self.rows.values():
            row.cands["fp32"].err, row.cands["fp32"].flips = fp32_err, base.flips
        self.secs["candidates"] = time.perf_counter() - t
        if self.do_format_loss:
            t = time.perf_counter()
            self._format_loss()
            self.secs["format_loss"] = time.perf_counter() - t

    def _kernel_verdict(self, n: str, p: str, c: Cand):
        checks = self.w.units[n].kernel_checks(p)
        if not checks:
            return
        worst = max(checks, key=lambda k: (not k.ok, k.ratio))
        c.kernel = {"ok": all(k.ok for k in checks), "ratio_to_bound": worst.ratio,
                    "code": worst.code, "message": worst.message(), "shapes": [list(k.shape) for k in checks]}
        if not c.kernel["ok"]:
            c.status = "kernel_bug"
            c.reason = worst.message()
            self.issues.append(Issue("TS664", f"{n}: {p} kernel failed its exact check on this layer's "
                                              f"inputs and is not used ({worst.message()}); this is a "
                                              "kernel bug, not format loss"))

    def _format_loss(self):
        """Verdict (b): the format alone, float64 everywhere else, no kernel involved."""
        from ..quant.model import RefQuantLinear

        need = [(n, p) for n, row in self.rows.items() for p, c in row.cands.items()
                if p in QUANT and c.status in ("ok", "kernel_bug")]
        if not need:
            return
        dev = self.device
        try:
            f64 = copy.deepcopy(self.model).to(dev).double().eval()
        except Exception:  # pragma: no cover (memory)
            dev = "cpu"
            f64 = copy.deepcopy(self.model).cpu().double().eval()
        batches = [_slice(_to(self.cal_cpu[0], dev, torch.float64), 128)]
        with torch.no_grad():
            ref = [_logits(f64(*b)).double() for b in batches]
        for n, p in need:
            unit = self.w.units[n]
            qw = getattr(unit.variants.get(p), "qw", None)
            if qw is None:
                continue
            orig = f64.get_submodule(n)
            U._set(f64, n, RefQuantLinear(qw, unit.base.bias).to(dev))
            try:
                with torch.no_grad():
                    s = score([_logits(f64(*b)) for b in batches], ref)
            finally:
                U._set(f64, n, orig)
            self.rows[n].cands[p].format_loss = {"flips": s.flips, "rel_err": s.rel_err,
                                                 "spec": qw.fmt.spec}
        del f64

    def timing(self):
        t = time.perf_counter()
        gen = torch.Generator(device=self.device)
        gen.manual_seed(0)
        self.cast_us: Dict[int, float] = {}
        with torch.no_grad(), U.strict_fp32():
            for n, row in self.rows.items():
                u = self.w.units[n]
                spec = u.spec
                for p, c in row.cands.items():
                    if c.status not in ("ok",):
                        continue
                    dt = STORAGE[p]
                    a = U.materialize(spec.args, dt, self.device, gen)
                    k = U.materialize(spec.kwargs, dt, self.device, gen)
                    mod = u.variant(p)

                    def call(mod=mod, a=a, k=k, p=p):
                        if p == "tf32":
                            torch.backends.cuda.matmul.allow_tf32 = True
                            torch.backends.cudnn.allow_tf32 = True
                        try:
                            return mod(*a, **k)
                        finally:
                            if p == "tf32":
                                torch.backends.cuda.matmul.allow_tf32 = False
                                torch.backends.cudnn.allow_tf32 = False
                    try:
                        c.time_us, how = gpu_us(call, 5, self.reps)
                        if how != "graph":
                            self.event_timed.append(f"{n}:{p}")
                    except Exception as exc:
                        c.status, c.reason = "unavailable", f"{type(exc).__name__}: {str(exc)[:160]}"
                        self.issues.append(Issue("TS665", f"{n}: {p} failed while timing: {c.reason}"))
            for gi, g in enumerate(self.w.groups):
                u = self.w.units[g[0]]
                numel = u.spec.float_numel
                if numel == 0:
                    self.cast_us[gi] = 0.0
                    continue
                x = torch.randn(numel, device=self.device)
                self.cast_us[gi], _ = gpu_us(lambda: x.to(torch.float16), 5, self.reps)
        self.calls = self._count_calls()
        for n, row in self.rows.items():
            row.calls = self.calls.get(n, 1)
        self.secs["timing"] = time.perf_counter() - t

    def _count_calls(self) -> Dict[str, int]:
        for u in self.w.units.values():
            u.calls = 0
        self.w.set({})
        with torch.no_grad(), U.strict_fp32():
            self.w.model(*self.cal[0])
        return {n: max(1, u.calls) for n, u in self.w.units.items()}

    # -- search

    def _t(self, n: str, p: str) -> float:
        return self.rows[n].cands[p].time_us * self.calls.get(n, 1)

    def _member_choice(self, n: str, dt: torch.dtype, tau: float) -> Optional[str]:
        row = self.rows[n]
        opts = []
        for p, c in row.cands.items():
            if STORAGE[p] != dt or c.status != "ok" or c.time_us is None:
                continue
            if p != "fp32" and not (c.err is not None and c.err <= tau):
                continue
            opts.append(p)
        if not opts:
            return None
        tmin = min(self._t(n, p) for p in opts)
        near = [p for p in opts if self._t(n, p) <= tmin * (1 + NOISE_REL) + NOISE_ABS_US]
        return min(near, key=lambda p: ACCURACY_RANK[p])

    def _group_options(self, gi: int, tau: float) -> Dict[torch.dtype, Dict[str, str]]:
        g = self.w.groups[gi]
        out = {}
        for dt in (torch.float32, torch.bfloat16, torch.float16):
            if dt != torch.float32:
                s = self.group_err.get((gi, dt))
                if s is None or not (s.rel_err <= tau):
                    continue
            ch = {}
            for n in g:
                p = self._member_choice(n, dt, tau)
                if p is None:
                    ch = None
                    break
                ch[n] = p
            if ch is not None:
                out[dt] = ch
        return out

    def predict(self, assign: Dict[str, str]) -> Optional[float]:
        """Sum of measured unit times plus a cast wherever the activation dtype changes between
        groups in execution order. None when a unit's precision was never timed (excluded)."""
        total = 0.0
        prev = torch.float32
        for gi, g in enumerate(self.w.groups):
            dt = STORAGE[assign.get(g[0], "fp32")]
            for n in g:
                if self.rows[n].cands[assign.get(n, "fp32")].time_us is None:
                    return None
            total += sum(self._t(n, assign.get(n, "fp32")) for n in g)
            if dt != prev:
                total += self.cast_us.get(gi, 0.0)
            prev = dt
        return total

    def choose(self, tau: float) -> Dict[str, str]:
        """Dynamic program over the tie groups in execution order, state = activation dtype."""
        best: Dict[torch.dtype, Tuple[float, List[Dict[str, str]]]] = {torch.float32: (0.0, [])}
        for gi, g in enumerate(self.w.groups):
            opts = self._group_options(gi, tau)
            new: Dict[torch.dtype, Tuple[float, List[Dict[str, str]]]] = {}
            for dt, ch in opts.items():
                c = sum(self._t(n, p) for n, p in ch.items())
                for pdt, (pc, pa) in best.items():
                    tot = pc + c + (self.cast_us.get(gi, 0.0) if pdt != dt else 0.0)
                    if dt not in new or tot < new[dt][0]:
                        new[dt] = (tot, pa + [ch])
            best = new
        _, chs = min(best.values(), key=lambda v: v[0])
        out: Dict[str, str] = {}
        for ch in chs:
            out.update(ch)
        return out

    def ok(self, assign: Dict[str, str], budget: int, why: str) -> Tuple[bool, Score]:
        fresh = ("cal", tuple(sorted(assign.items()))) not in self._cache
        s = self.run_assign(assign)
        if fresh:
            self.verifications += 1
        self.search_log.append({"step": why, "flips": s.flips, "rel_err": s.rel_err if math.isfinite(s.rel_err) else None,
                                "predicted_us": _round(self.predict(assign)),
                                "counts": _counts(assign)})
        return s.flips <= budget, s

    def search(self, budget: int) -> List[Dict[str, str]]:
        """Plans that keep `budget` calibration flips, fastest predicted first."""
        errs = sorted({c.err for r in self.rows.values() for c in r.cands.values()
                       if c.err is not None and math.isfinite(c.err) and c.status == "ok"}
                      | {s.rel_err for s in self.group_err.values() if math.isfinite(s.rel_err)})
        taus = [t for t in [0.0] + errs + [math.inf] if t == 0.0 or t < self.cap]
        plans: List[Tuple[float, Dict[str, str]]] = []
        for tau in taus:
            a = self.choose(tau)
            if not plans or a != plans[-1][1]:
                plans.append((tau, a))
        found: List[Dict[str, str]] = []
        ok_hi, s_hi = self.ok(plans[-1][1], budget, f"fastest (tau {plans[-1][0]:.3g})")
        self.tau_fail = None
        if ok_hi:
            best_i = len(plans) - 1
        else:
            fail_flips = s_hi.flips
            ok_lo, _ = self.ok(plans[0][1], budget, "most accurate (tau 0)")
            if not ok_lo:
                return []
            lo, hi = 0, len(plans) - 1      # plans[lo] ok, plans[hi] not
            while hi - lo > 1:
                mid = (lo + hi) // 2
                good, s_mid = self.ok(plans[mid][1], budget, f"bisect tau {plans[mid][0]:.3g}")
                if good:
                    lo = mid
                else:
                    hi, fail_flips = mid, s_mid.flips
            best_i = lo
            self.tau_fail, self.fail_flips, self.fail_budget = plans[hi][0], fail_flips, budget
        self.tau_ok = plans[best_i][0]
        cur = dict(plans[best_i][1])
        found.append(dict(cur))
        # greedy refinement: largest predicted savings first, each one verified
        tried = 0
        while tried < self.refine:
            moves = self._moves(cur)
            moves = [(sv, a) for sv, a in moves if sv > NOISE_ABS_US and _key(a) not in self._tried]
            if not moves:
                break
            sv, a = moves[0]
            self._tried.add(_key(a))
            tried += 1
            good, _ = self.ok(a, budget, f"refine (saves {sv:.0f} us predicted)")
            if good:
                cur = a
                found.append(dict(cur))
        return found

    def plan_tau(self, assign: Dict[str, str]) -> float:
        """The largest error alone among the plan's choices (its unit or its group's dtype)."""
        worst = 0.0
        for n, p in assign.items():
            if p == "fp32":
                continue
            row = self.rows[n]
            e = row.cands[p].err
            gs = self.group_err.get((row.group, STORAGE[p]))
            worst = max(worst, e if e is not None else math.inf,
                        gs.rel_err if gs is not None and STORAGE[p] != torch.float32 else 0.0)
        return worst

    def _moves(self, cur: Dict[str, str]) -> List[Tuple[float, Dict[str, str]]]:
        base = self.predict(cur)
        out = []
        moves = self._moves_all(cur, base)
        return [(sv, a) for sv, a in moves if self.plan_tau(a) < self.cap]

    def _moves_all(self, cur: Dict[str, str], base: float) -> List[Tuple[float, Dict[str, str]]]:
        out = []
        for gi, g in enumerate(self.w.groups):
            dt = STORAGE[cur.get(g[0], "fp32")]
            for n in g:
                for p, c in self.rows[n].cands.items():
                    if c.status != "ok" or c.time_us is None or STORAGE[p] != dt or p == cur.get(n, "fp32"):
                        continue
                    a = dict(cur)
                    a[n] = p
                    out.append((base - self.predict(a), a))
            for odt in (torch.bfloat16, torch.float16):
                if odt == dt or (gi, odt) not in self.group_err:
                    continue
                ch = {}
                for n in g:
                    p = self._member_choice(n, odt, math.inf)
                    if p is None:
                        ch = None
                        break
                    ch[n] = p
                if ch:
                    a = dict(cur)
                    a.update(ch)
                    out.append((base - self.predict(a), a))
        out.sort(key=lambda v: -v[0])
        return out

    # -- end to end

    def latency(self, fn: Callable[[], Any]) -> Tuple[float, Tuple[float, float], Optional[float]]:
        """GPU time per forward from CUDA graph replay (median of `rounds` medians, with their
        range), and the eager wall time per call for reference."""
        from ..runtime.timing import host_and_wall

        meds = []
        for _ in range(max(self.rounds, 3)):
            t, how = gpu_us(fn, 2, 2 * self.reps)
            meds.append(t)
        # a shared GPU can stall one round by orders of magnitude; measure again (at most twice)
        # and say so when the rounds still disagree by more than 1.5x
        for _ in range(2):
            if max(meds) <= 1.5 * min(meds):
                break
            meds = sorted(meds)[: len(meds) // 2 + 1] + [gpu_us(fn, 2, 2 * self.reps)[0]
                                                          for _ in range(max(self.rounds, 3))]
        if max(meds) > 1.5 * min(meds):
            self.unstable += 1
        eager = host_and_wall(fn, n=10, repeats=5, spin=0.02)["wall_us"]
        return statistics.median(meds), (min(meds), max(meds)), eager

    def measure_assign(self, name: str, assign: Dict[str, str]) -> Measured:
        cal = self.run_assign(assign)
        self.w.set(assign)
        b = self.cal[0]
        with torch.no_grad(), U.strict_fp32():
            lat, rng, eager = self.latency(lambda: self.w.model(*b))
        return Measured(name, dict(assign), cal, latency_us=lat, latency_range=rng,
                        predicted_us=self.predict(assign), eager_us=eager)

    def plan(self) -> Tuple[Dict[str, str], Measured, int]:
        t = time.perf_counter()
        self._tried = set()
        budget = self.allow_cal
        self.cap = math.inf
        rounds = 0
        while True:
            found = self.search(budget)
            if not found:
                break
            cands = {}
            for a in found:
                cands[_key(a)] = a
            measured = [self.measure_assign(f"candidate {i}", a) for i, a in enumerate(cands.values())]
            measured.sort(key=lambda m: m.latency_us)
            failing = []
            for m in measured:
                if m.cal.flips > self.allow_cal:
                    continue
                m.hold = self.run_assign(m.assign, "hold")
                self.verifications += 1
                self.search_log.append({"step": f"held out gate ({m.name})", "flips": m.hold.flips,
                                        "allowed": self.allow_hold, "latency_us": round(m.latency_us, 1)})
                if m.hold.flips <= self.allow_hold:
                    self.secs["search"] = time.perf_counter() - t
                    return m.assign, m, rounds
                failing.append(m)
            rounds += 1
            if rounds > 6 or not failing:
                break
            # tighter on both axes: fewer calibration flips than any failing plan when that is
            # possible, and a per unit error threshold below the failing plans' own
            fewer = min(m.cal.flips for m in failing) - 1
            if fewer >= 0:
                budget = min(budget, fewer)
            self.cap = min(self.cap, min(self.plan_tau(m.assign) for m in failing))
            if self.cap <= 0.0:
                break
            self.search_log.append({"step": f"tighten: calibration budget {budget}, error alone "
                                            f"below {self.cap:.3g}"})
        # last resort: the most accurate plan (float32, TF32 only where its error is zero)
        a0 = self.choose(0.0)
        m = self.measure_assign("most accurate", a0)
        if m.cal.flips <= self.allow_cal:
            m.hold = self.run_assign(a0, "hold")
            self.verifications += 1
            self.search_log.append({"step": "held out gate (most accurate plan)", "flips": m.hold.flips,
                                    "allowed": self.allow_hold})
            if m.hold.flips <= self.allow_hold:
                self.secs["search"] = time.perf_counter() - t
                return a0, m, max(rounds, 1)
        self.secs["search"] = time.perf_counter() - t
        return {}, None, rounds

    # -- baselines (what people do today)

    def baselines(self) -> List[Measured]:
        out = []
        b0 = self.cal[0]

        def literal(name, dtype, flags=False):
            m = copy.deepcopy(self.model).to(self.device).to(dtype).eval()

            def f(batch):
                return m(*_to(batch, self.device, dtype))
            return self._measure_fn(name, f, b0, tf32=flags)

        out.append(literal("all fp32", torch.float32))
        if "tf32" in self.precisions:
            out.append(literal("all tf32", torch.float32, flags=True))
        if "fp16" in self.precisions:
            out.append(literal("all fp16", torch.float16))
        if "bf16" in self.precisions:
            out.append(literal("all bf16", torch.bfloat16))
        if "fp16" in self.precisions:
            assign = {n: ("fp32" if isinstance(u.base, U.NORM_TYPES) else "fp16")
                      for n, u in self.w.units.items()}
            try:
                m = self.measure_assign("fp16 except norms", assign)
                m.assign = None
                m.hold = self.run_assign(assign, "hold")
                out.append(m)
            except Exception as exc:
                self.notes.append(f"fp16 except norms could not run: {type(exc).__name__}: {exc}")
            mf = copy.deepcopy(self.model).to(self.device).float().eval()

            def ac(batch):
                with torch.autocast("cuda", dtype=torch.float16):
                    return mf(*batch)
            out.append(self._measure_fn("torch.autocast fp16", ac, b0))
        return out

    def _measure_fn(self, name, f, b0, tf32=False) -> Measured:
        t_start = time.perf_counter()
        def scores(batches, refs):
            outs = []
            with torch.no_grad():
                for b in batches:
                    outs.append(_logits(f(b)).detach().double())
            return score(outs, refs)

        m, c = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        try:
            try:
                cal = scores(self.cal, self.ref_cal)
                hold = scores(self.hold, self.ref_hold)
            except Exception as exc:
                self.notes.append(f"baseline {name} failed: {type(exc).__name__}: {str(exc)[:160]}")
                return Measured(name, None, Score(-1, 0, math.inf))
            with torch.no_grad():
                lat, rng, eager = self.latency(lambda: f(b0))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = m
            torch.backends.cudnn.allow_tf32 = c
        self.secs[f"baseline {name}"] = time.perf_counter() - t_start
        return Measured(name, None, cal, hold, lat, rng, eager_us=eager)

    # -- explanation

    def explain(self, assign: Dict[str, str]):
        for n, row in self.rows.items():
            p = assign.get(n, "fp32")
            row.chosen = p
            c = row.cands[p]
            rejected = []
            faster = [q for q, cq in row.cands.items()
                      if q != p and cq.time_us is not None and c.time_us is not None
                      and cq.time_us < c.time_us * (1 - NOISE_REL) - NOISE_ABS_US]
            for q, cq in row.cands.items():
                if q == p or cq.status == "not_requested":
                    continue
                if cq.status != "ok":
                    rejected.append(f"{q}: {cq.status} ({cq.reason})" if cq.reason else f"{q}: {cq.status}")
                elif q in faster:
                    fl = cq.format_loss
                    extra = f", format alone {fl['flips']} flips" if fl else ""
                    gs = self.group_err.get((row.group, STORAGE[q]))
                    e = max(cq.err if cq.err is not None else math.inf,
                            gs.rel_err if gs is not None and STORAGE[q] != torch.float32 else 0.0)
                    tau_ok = getattr(self, "tau_ok", math.inf)
                    if e > tau_ok and getattr(self, "tau_fail", None) is not None:
                        rejected.append(f"{q} {cq.time_us:.1f} us: error alone {_fmt(e)}{extra}, above the "
                                        f"accepted {_fmt(tau_ok)}; the plan admitting up to "
                                        f"{_fmt(self.tau_fail)} flipped {self.fail_flips} (allowed "
                                        f"{self.fail_budget})")
                    else:
                        rejected.append(f"{q} {cq.time_us:.1f} us: error alone {_fmt(e)}{extra}; not "
                                        "faster once the casts between activation dtypes are counted, "
                                        "or not reached by the refinement")
            head = f"{p}"
            if c.time_us is not None:
                head += f" {c.time_us:.1f} us"
            if c.err is not None and p != "fp32":
                head += f", error alone {_fmt(c.err)}"
            if c.kernel:
                head += f", kernel {c.kernel['ratio_to_bound']:.2f} of its bound"
            if c.format_loss:
                head += f", format alone {c.format_loss['flips']} flips"
            if not faster and p != "fp32":
                head += "; fastest allowed"
            elif not faster:
                head += "; no faster allowed candidate (timing noise counts as equal)"
            row.why = head + ("; " + "; ".join(rejected) if rejected else "")


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if not math.isfinite(x):
        return "inf"
    return f"{x:.2e}"


def _round(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(x, 1)


def _key(a: Dict[str, str]) -> Tuple:
    return tuple(sorted(a.items()))


def _counts(a: Dict[str, str]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for p in a.values():
        c[p] = c.get(p, 0) + 1
    return c


def plan(model: nn.Module, calibration, holdout=None, max_flips: Union[int, float, str] = 0,
         precisions: Optional[Sequence[str]] = None, device: str = "cuda", reps: int = 7,
         rounds: int = 3, refine: int = 6, use_range: bool = True, format_loss: bool = True,
         baselines: bool = True, range_rows: int = 16, model_name: Optional[str] = None,
         log: Optional[Callable[[str], None]] = None) -> PrecisionPlan:
    """Choose a precision per unit so the model is as fast as measured possible on this GPU while
    decision flips against float64 stay within `max_flips` on the calibration inputs and (scaled to
    their size) on the held out inputs.

    `calibration`, `holdout`: forward argument tuples (a tensor, a tuple, or a list of either).
    Without `holdout` the calibration inputs are split in two. `max_flips`: a count on the
    calibration decisions or a share such as "0.5%".
    """
    t0 = time.perf_counter()
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise PlanError("TS669: the precision planner times real kernels and needs a CUDA GPU; "
                        f"device {device!r} is not one (nothing is planned on the CPU)")
    precisions = tuple(precisions or U.PRECISIONS)
    bad = [p for p in precisions if p not in U.PRECISIONS]
    if bad or "fp32" not in precisions:
        raise PlanError(f"precisions must include fp32 and come from {', '.join(U.PRECISIONS)}; got "
                        + ", ".join(precisions))
    budget = Budget.parse(max_flips)
    cal = _as_list(calibration)
    hold = _as_list(holdout)
    notes = []
    if not hold:
        cal, hold = _split(cal)
        notes.append("no held out inputs given: the calibration inputs were split in two halves")
    model = model.eval()
    P = _Planner(model, cal, hold, budget, precisions, device, reps, rounds, refine, use_range,
                 format_loss, range_rows, log)
    P.notes.extend(notes)
    P.reference()
    P.units()
    P.range()
    P.candidates()
    P.timing()
    if P.event_timed:
        P.notes.append("timed with CUDA events, not a CUDA graph (capture failed), so launch cost "
                       "is included: " + ", ".join(P.event_timed[:8]))
    assign, result, tightened = P.plan()
    met = result is not None
    if not met:
        assign = {}
        result = P.measure_assign("all fp32 (budget not reachable)", {})
        result.hold = P.run_assign({}, "hold")
        P.issues.append(Issue("TS662", f"no plan keeps the budget of {P.allow_cal} flips on the "
                                       f"calibration inputs and {P.allow_hold} on the held out "
                                       f"inputs, not even float32 everywhere ({result.cal.flips} and "
                                       f"{result.hold.flips} flips); the plan is float32 everywhere"))
    else:
        if tightened:
            P.issues.append(Issue("TS661", f"a plan within budget on the calibration inputs flipped "
                                           f"too many held out decisions; the calibration budget was "
                                           f"tightened {tightened} time(s) and the final plan keeps "
                                           "both"))
        P.issues.insert(0, Issue("TS660", f"plan verified: {result.cal.flips} of {result.cal.decisions} "
                                          f"calibration and {result.hold.flips} of "
                                          f"{result.hold.decisions} held out decisions flip against "
                                          f"float64 (allowed {P.allow_cal} and {P.allow_hold})"))
    result.name = "plan"
    P.explain(assign)
    P.secs["planning"] = time.perf_counter() - t0
    tb = time.perf_counter()
    base = P.baselines() if baselines else []
    P.secs["baselines"] = time.perf_counter() - tb
    if P.unstable:
        P.notes.append(f"{P.unstable} end to end timings varied by more than 1.5x between rounds even "
                       "after measuring again (a GPU shared with other work); compare the ranges")
    if base:
        within = [b for b in base if b.cal.flips >= 0 and b.cal.flips <= P.allow_cal and b.hold is not None
                  and b.hold.flips <= P.allow_hold and b.latency_us is not None]
        faster = [b for b in within if b.latency_us < result.latency_range[0]
                  and b.latency_range[1] < result.latency_us]
        if met and faster:
            f = min(faster, key=lambda b: b.latency_us)
            P.issues.append(Issue("TS670", f"{f.name} keeps the budget and measured faster "
                                           f"({f.latency_us:.0f} us against {result.latency_us:.0f} us): "
                                           "the per unit casts and dispatch of the plan cost more than "
                                           "they save on this model"))
    P.secs["total"] = time.perf_counter() - t0
    return PrecisionPlan(
        model_name=model_name or type(model).__name__, device=device,
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, budget=budget,
        allowed_cal=P.allow_cal, allowed_hold=P.allow_hold,
        assign={n: assign.get(n, "fp32") for n in P.w.units}, units=list(P.rows.values()),
        groups=P.w.groups, result=result, baselines=base, search=P.search_log,
        verifications=P.verifications, seconds={k: round(v, 3) for k, v in P.secs.items()},
        fingerprint=fingerprint(model), issues=P.issues, notes=P.notes, met=met, tightened=tightened)


# ----------------------------------------------------------------------------- applying a plan


def apply(model: nn.Module, plan_or_doc: Union[PrecisionPlan, Dict[str, Any], str],
          device: str = "cuda", check_fingerprint: bool = True) -> nn.Module:
    """A copy of model running the plan (units wrapped, each in its planned precision). The plan
    must have been made for this model: names, shapes and a weight checksum are compared."""
    import json

    if isinstance(plan_or_doc, str):
        with open(plan_or_doc, encoding="utf-8") as f:
            doc = json.load(f)
    elif isinstance(plan_or_doc, PrecisionPlan):
        doc = plan_or_doc.as_dict()
    else:
        doc = plan_or_doc
    if check_fingerprint:
        fp = fingerprint(model)
        if fp != doc.get("fingerprint"):
            raise PlanError(f"TS666: this plan was made for another model or other weights "
                            f"(fingerprint {str(doc.get('fingerprint', {}).get('sha256'))[:12]}, this "
                            f"model {fp['sha256'][:12]}); make a new plan")
    return PlannedModel(model, doc["plan"], device)


class PlannedModel(nn.Module):
    """A model running a precision plan. Units are discovered on the first call (as the planner
    did) and compared with the plan's names; a mismatch raises TS666."""

    def __init__(self, model: nn.Module, assign: Dict[str, str], device: str = "cuda"):
        super().__init__()
        self._src = [model]
        self.assign = dict(assign)
        self.device = device
        self._w: Optional[Wrapped] = None

    def forward(self, *args, **kwargs):
        if self._w is None:
            args0 = _to(tuple(args), self.device)
            w = U.wrap(self._src[0], self.device, lambda m: m(*args0, **kwargs))
            if set(w.units) != set(self.assign):
                missing = sorted(set(self.assign) ^ set(w.units))
                raise PlanError("TS666: the plan's units differ from this model's: " + ", ".join(missing[:8]))
            w.set(self.assign)
            self._w = w
        with U.strict_fp32():
            return self._w.model(*_to(tuple(args), self.device), **kwargs)
