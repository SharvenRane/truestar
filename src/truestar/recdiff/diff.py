"""`truestar diff`: name the first module, then the op, whose own result changed between two
environments, and say what kind of change it is.

Two ways to compare:

* `diff_recordings(a, b)`: two recordings made in two environments. A module whose inputs hash
  the same in both runs is isolated exactly: everything before it was bitwise identical, so any
  change in its output is its own. Modules downstream of the first change see different inputs
  and are reported as inherited (their own behaviour is unknown offline; see replay).
* `diff_replay(a, model)`: replay recording `a` in the current environment. Every module is fed
  its recorded inputs and run alone, so every module's own change is measured, not only the
  first one. Op drilldown then runs inside the blamed module: with a recorded op trace each op is
  fed its recorded inputs (exact); without one, opdrill compares the module's ops against float64
  CPU math in the current environment.

Each module's change gets one class:

* identical: bitwise equal outputs;
* noise: within the dtype tolerance (calibrated against float64 in replay, as `truestar check`);
* precision: beyond the dtype tolerance but within the tolerance of a precision mode that the
  fingerprints show was switched (for example TF32 on CUDA);
* divergent: anything else. This is the class an upgrade regression lands in.

A parent module is blamed only when its own class is worse than every class owned by a module
inside it, so a block is never blamed for its child's change.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from ..capture import ROOT, flatten_tensors
from ..codes import Issue
from ..compare import DIVERGENT_FACTOR, DTYPE_FLOOR, OK, SUSPECT, Diff, diff_lists, threshold_for
from . import store as S
from .fingerprint import PRECISION, VERSION, differences, fingerprint
from .record import Recording, record, state_hashes

IDENTICAL, NOISE, PRECISION_CHANGE, DIVERGED = "identical", "noise", "precision", "divergent"
UNCHECKED, WEIGHTS = "unchecked", "weights_differ"
CLASSES = (IDENTICAL, NOISE, PRECISION_CHANGE, DIVERGED)
RANK = {UNCHECKED: -1, WEIGHTS: -1, IDENTICAL: 0, NOISE: 1, PRECISION_CHANGE: 2, DIVERGED: 3}


class ReplayError(Exception):
    """Replay cannot start (missing backend, wrong model)."""


def classify(d: Optional[Diff], identical: bool, t_noise: float, t_prec: Optional[float],
             leaf: bool) -> str:
    """One change class for an output difference `d` (current vs recorded)."""
    if identical:
        return IDENTICAL
    if d is None or d.shape_mismatch or d.nonfinite_mismatch:
        return DIVERGED
    # Parents compound their children's precision; as in `truestar check`, only a clear error
    # in a parent moves it up a class.
    allowed = (OK,) if leaf else (OK, SUSPECT)
    if d.status(t_noise) in allowed:
        return NOISE
    if t_prec is not None and d.status(t_prec) in allowed:
        return PRECISION_CHANGE
    return DIVERGED


@dataclass
class ModuleChange:
    key: str
    name: str
    type_name: str
    leaf: bool
    start: int
    cls: str
    isolated: bool  # this module's own behaviour was measured (replay, or equal input hashes)
    method: str  # "replay", "hash" or "none"
    diff: Optional[Diff] = None
    noise_band: float = 0.0
    precision_band: Optional[float] = None
    exact_recorded: Optional[float] = None  # rel error vs float64 of the recorded output
    exact_current: Optional[float] = None  # rel error vs float64 of the current output
    error: str = ""
    own: bool = False  # set by _assign_ownership

    def as_dict(self) -> Dict[str, Any]:
        d = self.diff
        return {
            "key": self.key, "name": self.name, "type": self.type_name, "leaf": self.leaf,
            "class": self.cls, "own": self.own, "isolated": self.isolated, "method": self.method,
            "rel_err": None if d is None else _num(d.rel_err),
            "max_abs": None if d is None else _num(d.max_abs),
            "nonfinite_mismatch": bool(d.nonfinite_mismatch) if d else False,
            "shape_mismatch": bool(d.shape_mismatch) if d else False,
            "noise_band": self.noise_band, "precision_band": self.precision_band,
            "exact_recorded_rel_err": _num(self.exact_recorded),
            "exact_current_rel_err": _num(self.exact_current),
            "error": self.error or None,
        }


@dataclass
class OpChange:
    op: str
    module: str
    index: int
    total: int
    cls: str
    diff: Diff
    band: float
    source: str  # "recorded_trace" (old environment's op inputs) or "float64_reference"
    isolated: bool
    input_shapes: List[List[int]] = field(default_factory=list)
    input_dtypes: List[str] = field(default_factory=list)
    args: Any = field(default=None, repr=False)
    kwargs: Dict[str, Any] = field(default_factory=dict, repr=False)
    reference: List[torch.Tensor] = field(default_factory=list, repr=False)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op, "module": self.module, "index": self.index, "total": self.total,
            "class": self.cls, "rel_err": _num(self.diff.rel_err),
            "max_abs": _num(self.diff.max_abs), "band": self.band, "source": self.source,
            "isolated": self.isolated,
            "inputs": [{"dtype": d, "shape": s} for d, s in zip(self.input_dtypes,
                                                                 self.input_shapes)],
        }


@dataclass
class DiffReport:
    mode: str  # "recordings", "replay" or "replay_outputs_only"
    recorded: str  # path or label of recording a
    current: str  # path of recording b, or "current environment"
    model: str
    dtype: torch.dtype
    device: str
    env_recorded: Dict[str, Any]
    env_current: Dict[str, Any]
    env_changes: List[Tuple[str, Any, Any, str]]
    precision_explained: bool
    modules: List[ModuleChange]
    blamed: Optional[ModuleChange] = None
    blamed_op: Optional[OpChange] = None
    issues: List[Issue] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    baseline: Dict[str, Any] = field(default_factory=dict)
    final: Optional[Diff] = None
    op_repro_path: Optional[str] = None
    blame_source: Optional[Dict[str, Any]] = None

    # -- summary
    def dtype_name(self) -> str:
        return S.dtype_name(self.dtype)

    @property
    def status(self) -> str:
        if any(m.cls == WEIGHTS for m in self.modules) or any(i.code == "TS069"
                                                               for i in self.issues):
            return "not_comparable"
        # Own changes, plus changes nobody could isolate (offline, downstream of the first
        # change): those may hide a worse change, so they count rather than pass silently.
        counted = [m.cls for m in self.modules
                   if m.own or (not m.isolated and m.cls in CLASSES)]
        return max(counted, key=RANK.get) if counted else IDENTICAL

    @property
    def codes(self) -> List[str]:
        out: List[str] = []
        for i in self.issues:
            if i.code not in out:
                out.append(i.code)
        return out

    def counts(self) -> Dict[str, int]:
        c = {k: 0 for k in CLASSES + (UNCHECKED, WEIGHTS)}
        inherited = 0
        for m in self.modules:
            if m.own or m.cls in (IDENTICAL, UNCHECKED, WEIGHTS):
                c[m.cls] += 1
            else:
                inherited += 1
        c["inherited"] = inherited
        return c

    def exit_code(self, strict: bool = False) -> int:
        s = self.status
        if s in (DIVERGED, "not_comparable"):
            return 1
        if strict and s == PRECISION_CHANGE:
            return 1
        return 0

    def add(self, code: str, message: str) -> None:
        self.issues.append(Issue(code, message))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode, "recorded": self.recorded, "current": self.current,
            "model": self.model, "dtype": S.dtype_name(self.dtype), "device": self.device,
            "status": self.status, "codes": self.codes,
            "issues": [i.as_dict() for i in self.issues],
            "environment_changes": [{"key": k, "recorded": a, "current": b, "kind": kd}
                                    for k, a, b, kd in self.env_changes],
            "precision_explained": self.precision_explained,
            "counts": self.counts(),
            "blamed_module": None if self.blamed is None else self.blamed.as_dict(),
            "blamed_op": None if self.blamed_op is None else self.blamed_op.as_dict(),
            "final_output": None if self.final is None else {
                "rel_err": _num(self.final.rel_err), "max_abs": _num(self.final.max_abs)},
            "baseline": self.baseline,
            "op_repro_path": self.op_repro_path,
            "blame_source": self.blame_source,
            "notes": list(self.notes),
            "env_recorded": self.env_recorded,
            "env_current": self.env_current,
            "modules": [m.as_dict() for m in self.modules],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, allow_nan=False, default=str)

    def to_text(self, top: int = 8) -> str:
        from .report import render_text

        return render_text(self, top)

    def to_agent(self, recorded_path: Optional[str] = None, model_file: Optional[str] = None,
                 strict: bool = False) -> Dict[str, Any]:
        from .report import to_agent

        return to_agent(self, recorded_path, model_file, strict)


def _num(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if x == x and abs(x) != float("inf") else None


# ----------------------------------------------------------------------------- shared steps


def _is_cuda(device: str) -> bool:
    return torch.device(device).type == "cuda"


def _precision_explained(changes, *devices: str) -> bool:
    return any(k == PRECISION for _, _, _, k in changes) and any(_is_cuda(d) for d in devices)


def _runs_alone(call: Dict[str, Any]) -> bool:
    """No submodule ran inside this call, so its output is all its own code and gets the leaf
    band. nn.MultiheadAttention has an out_proj child it only uses functionally: a leaf here."""
    return not call["children"]


def _descendants(calls: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    by_key = {c["key"]: c for c in calls}
    memo: Dict[str, List[str]] = {}

    def walk(k: str) -> List[str]:
        if k not in memo:
            out: List[str] = []
            for ch in by_key[k]["children"] if k in by_key else []:
                out.append(ch)
                out.extend(walk(ch))
            memo[k] = out
        return memo[k]

    for c in calls:
        walk(c["key"])
    return memo


def _assign_ownership(changes: Dict[str, ModuleChange], calls: List[Dict[str, Any]]) -> None:
    """A measured module owns its change unless a module inside it owns an equal or worse one."""
    desc = _descendants(calls)
    for c in calls:  # completion order: children first
        m = changes.get(c["key"])
        if m is None or not m.isolated or m.cls in (UNCHECKED, WEIGHTS):
            continue
        inner = [changes[k] for k in desc.get(c["key"], []) if k in changes and changes[k].own]
        worst_inner = max((RANK[x.cls] for x in inner), default=-1)
        m.own = RANK[m.cls] > worst_inner


def _blame(modules: List[ModuleChange]) -> Optional[ModuleChange]:
    own = [m for m in modules if m.own and RANK[m.cls] >= RANK[NOISE]]
    if not own:
        return None
    top = max(RANK[m.cls] for m in own)
    return min((m for m in own if RANK[m.cls] == top), key=lambda m: m.start)


def _baseline(cur: List[torch.Tensor], rec: List[torch.Tensor]) -> Dict[str, Any]:
    """What people do today: torch.allclose on the final outputs, default tolerances."""
    ok = len(cur) == len(rec) and all(
        a.shape == b.shape and bool(torch.allclose(a.double(), b.double()))
        if a.is_floating_point() else torch.equal(a, b) for a, b in zip(cur, rec))
    return {"method": "torch.allclose(final outputs, rtol=1e-05, atol=1e-08)", "allclose": ok,
            "names_module": False, "names_op": False}


def _finish(rep: DiffReport, calls: List[Dict[str, Any]], changes: Dict[str, ModuleChange]):
    _assign_ownership(changes, calls)
    start = {c["key"]: c["start"] for c in calls}
    rep.modules = sorted(changes.values(), key=lambda m: start[m.key])
    rep.blamed = _blame(rep.modules)
    if rep.env_changes:
        shown = ", ".join(f"{k}: {a} -> {b}" for k, a, b, _ in rep.env_changes[:6])
        more = len(rep.env_changes) - 6
        rep.add("TS063", f"{len(rep.env_changes)} environment setting(s) differ: {shown}"
                + (f" and {more} more" if more > 0 else ""))
    weights = [m.key for m in rep.modules if m.cls == WEIGHTS]
    if weights:
        rep.add("TS064", f"{len(weights)} module(s) have different parameters or buffers than "
                         f"when recorded, so this compares a different model: {weights[:5]}")
    b = rep.blamed
    if b is not None:
        rel = b.diff.rel_err if b.diff is not None else float("nan")
        base = (f"module {b.key} ({b.type_name}) own output changed by {rel:.2e} "
                f"(noise band {b.noise_band:.1e}")
        if b.cls == DIVERGED:
            hint = ""
            if (b.diff is not None and not rep.precision_explained and b.diff.rel_err
                    <= threshold_for(rep.dtype, True, None) and rep.dtype == torch.float32):
                hint = ("; the size matches a TF32 class precision change although no precision "
                        "flag differs, so the new library may pick reduced precision kernels")
            rep.add("TS060", base + ")" + hint)
        elif b.cls == PRECISION_CHANGE:
            keys = [k for k, _, _, kd in rep.env_changes if kd == PRECISION]
            rep.add("TS061", base + f", precision band {b.precision_band:.1e}); explained by "
                    + ", ".join(keys))
        elif b.cls == NOISE:
            rep.add("TS062", base + "): numerical noise within the dtype tolerance")
    unisolated = [m for m in rep.modules if not m.isolated and RANK[m.cls] > RANK[NOISE]]
    if unisolated:
        rep.add("TS068", f"{len(unisolated)} changed module(s) saw different inputs than when "
                         "recorded, so their own behaviour is not measured here; "
                         "`truestar diff a.tsrec --replay` in the new environment isolates "
                         "every module")
    root = next((m for m in rep.modules if m.name == ROOT), None)
    if root is not None:
        rep.final = root.diff


def _blame_source(model: Optional[nn.Module], rep: DiffReport) -> None:
    if model is None or rep.blamed is None:
        return
    try:
        from ..agent import module_source

        rep.blame_source = module_source(model, rep.blamed.name)
    except Exception:
        rep.blame_source = None


# ----------------------------------------------------------------------------- two recordings


def _hashes_of(rec: Recording, idx: List[int]) -> List[str]:
    return [rec.entries[i]["hash"] for i in idx]


def diff_recordings(a: Recording, b: Recording, mode: str = "recordings",
                    current_label: Optional[str] = None) -> DiffReport:
    """Compare recording b (new environment) against recording a (old environment)."""
    dtype = a.dtype
    changes = differences(a.env, b.env)
    explained = _precision_explained(changes, a.device, b.device)
    rep = DiffReport(mode=mode, recorded=a.path or "<recording a>",
                     current=current_label or b.path or "<recording b>",
                     model=a.header["model"]["name"], dtype=dtype, device=b.device,
                     env_recorded=a.env, env_current=b.env, env_changes=changes,
                     precision_explained=explained, modules=[])
    if b.dtype != dtype:
        rep.notes.append(f"recorded dtypes differ: {S.dtype_name(dtype)} vs "
                         f"{S.dtype_name(b.dtype)}")
    ka, kb = [c["key"] for c in a.calls], [c["key"] for c in b.calls]
    if ka != kb:
        missing = [k for k in ka if k not in b.by_key]
        extra = [k for k in kb if k not in a.by_key]
        rep.add("TS066", f"different modules ran (data dependent control flow?). Missing: "
                         f"{missing[:5]}; extra: {extra[:5]}")
    ra, rb = a.by_key.get(f"{ROOT}#0"), b.by_key.get(f"{ROOT}#0")
    if ra is not None and rb is not None and ra["in_hash"] != rb["in_hash"]:
        rep.add("TS069", "the two runs fed the model different inputs, so their outputs are "
                         "not comparable")
    t_n = threshold_for(dtype, False, None)
    t_p = threshold_for(dtype, True, None) if explained else None
    out: Dict[str, ModuleChange] = {}
    for ca in a.calls:
        cb = b.by_key.get(ca["key"])
        m = ModuleChange(ca["key"], ca["name"], ca["type"], ca["leaf"], ca["start"], UNCHECKED,
                         False, "none", noise_band=t_n, precision_band=t_p)
        out[ca["key"]] = m
        if cb is None:
            m.error = "did not run in the new environment"
            continue
        if ca["state"] != cb["state"]:
            m.cls = WEIGHTS
            continue
        same = _hashes_of(a, ca["out"]) == _hashes_of(b, cb["out"])
        m.diff = Diff(0.0, 0.0, False, False) if same else diff_lists(b.outputs(cb),
                                                                       a.outputs(ca))
        m.isolated = ca["in_hash"] == cb["in_hash"]
        m.method = "hash" if m.isolated else "none"
        m.cls = classify(m.diff, same, t_n, t_p, _runs_alone(ca))
    _finish(rep, a.calls, out)
    if rep.blamed is not None and rep.blamed.cls in (PRECISION_CHANGE, DIVERGED):
        rep.blamed_op = _op_offline(a, b, rep.blamed.key, t_n, t_p)
        if rep.blamed_op is None and not a.ops_under(rep.blamed.key):
            rep.notes.append(f"no op trace for {rep.blamed.key} in the recordings: record with "
                             f"--ops {rep.blamed.name} (or --ops all) to name the op offline")
    rep.baseline = _baseline([b.tensor(i) for i in b.header["plain_out"]],
                             [a.tensor(i) for i in a.header["plain_out"]])
    _path_note(rep, a, b)
    return rep


def _path_note(rep: DiffReport, *recs: Recording, live_differs: bool = False) -> None:
    bad = [r for r in recs if not r.header.get("instrumented_matches_plain", True)]
    if bad or live_differs:
        rep.add("TS071", "recording hooks changed this model's code path (forward hooks switch "
                         "off fused fast paths such as TransformerEncoderLayer's), so module "
                         "results describe the instrumented path; the baseline line compares "
                         "the plain path")


def _op_offline(a: Recording, b: Recording, key: str, t_n: float,
                t_p: Optional[float]) -> Optional[OpChange]:
    from ..opdrill import _align

    oa, ob = a.ops_under(key), b.ops_under(key)
    if not oa or not ob:
        return None
    pairs = _align([o["name"] for o in oa], [o["name"] for o in ob])
    first_any: Optional[OpChange] = None
    first_own: Optional[OpChange] = None
    for j, o in enumerate(ob):
        if j not in pairs:
            continue
        r = oa[pairs[j]]
        if _hashes_of(a, r["out"]) == _hashes_of(b, o["out"]):
            continue
        d = diff_lists(b.outputs(o), a.outputs(r))
        cls = classify(d, False, t_n, t_p, True)
        isolated = r["in_hash"] == o["in_hash"]
        ch = OpChange(o["name"], key, j + 1, len(ob), cls, d, t_n, "recorded_trace", isolated,
                      reference=a.outputs(r))
        if not isolated:
            first_any = first_any or ch
            continue
        if RANK[cls] <= RANK[NOISE]:
            continue
        if cls == DIVERGED:
            return ch
        first_own = first_own or ch
    return first_own or first_any


# ----------------------------------------------------------------------------- replay


def _cast(obj: Any, device: str, dtype: torch.dtype) -> Any:
    """check._cast_inputs, but a tensor passed twice stays one object."""

    def fn(t: torch.Tensor) -> torch.Tensor:
        if t.is_floating_point():
            return t.detach().to(device=device, dtype=dtype)
        return t.detach().to(device=device)

    return S.alias_map(obj, fn)


def _run_module(mod: nn.Module, args, kwargs, device: str, dtype: torch.dtype):
    from ..check import _sync

    a, k = _cast((args, kwargs), device, dtype)
    with torch.no_grad():
        out = mod(*a, **k)
    _sync(device)
    return [t.detach().cpu() for t in flatten_tensors(out)]


def _module(model: nn.Module, name: str) -> nn.Module:
    return model if name == ROOT else model.get_submodule(name)


def diff_replay(a: Recording, model: nn.Module, device: Optional[str] = None,
                exact: bool = True, repro_dir: Optional[str] = None) -> DiffReport:
    """Replay recording `a` in the current environment, every module alone with its inputs."""
    from ..backends import available_backends
    from ..check import _model_dtype

    device = device or a.device
    dev_type = torch.device(device).type
    if dev_type != "cpu" and not any(torch.device(b.device).type == dev_type
                                     for b in available_backends()):
        raise ReplayError(f"the recording was made on {device}, which is not available here; "
                          f"pass --backend to replay on another device")
    dtype = a.dtype
    base = copy.deepcopy(model).cpu().eval()
    if _model_dtype(base) != dtype:
        raise ReplayError(f"the model is {S.dtype_name(_model_dtype(base))} but the recording "
                          f"is {S.dtype_name(dtype)}")
    if a.outputs_only:
        return _replay_outputs_only(a, base, device, repro_dir)
    env = fingerprint()
    changes = differences(a.env, env)
    explained = _precision_explained(changes, device)
    rep = DiffReport(mode="replay", recorded=a.path or "<recording>", current="current environment",
                     model=a.header["model"]["name"], dtype=dtype, device=device,
                     env_recorded=a.env, env_current=env, env_changes=changes,
                     precision_explained=explained, modules=[])
    if str(device) != a.device:
        rep.notes.append(f"recorded on {a.device}, replayed on {device}")
    states = state_hashes(base)
    plain_live = copy.deepcopy(base).to(device)
    live = _hooked(copy.deepcopy(base).to(device))
    f64 = cpu_model = None
    if exact:
        try:
            f64 = _hooked(copy.deepcopy(base).to(torch.float64))
        except Exception:
            f64 = None
        cpu_model = _hooked(copy.deepcopy(base)) if dev_type != "cpu" else None
    floor = DTYPE_FLOOR.get(dtype, 1e-4)
    out: Dict[str, ModuleChange] = {}
    rec_inputs: Dict[str, Any] = {}
    failed: List[str] = []
    for c in a.calls:
        m = ModuleChange(c["key"], c["name"], c["type"], c["leaf"], c["start"], UNCHECKED,
                         False, "replay")
        out[c["key"]] = m
        if states.get(c["name"]) != c["state"]:
            m.cls = WEIGHTS
            continue
        try:
            args, kwargs = a.inputs(c)
        except S.Unreplayable as exc:
            m.error = str(exc)
            failed.append(c["key"])
            continue
        rec_inputs[c["key"]] = (args, kwargs)
        try:
            cur = _run_module(_module(live, c["name"]), *_clone((args, kwargs)), device,
                              dtype)
        except Exception as exc:
            m.error = f"{type(exc).__name__}: {exc}"
            failed.append(c["key"])
            continue
        rec_out = a.outputs(c)
        same = S.hashes(cur) == _hashes_of(a, c["out"])
        noise = None
        if not same and f64 is not None:
            try:
                r64 = _run_module(_module(f64, c["name"]), *_clone((args, kwargs)), "cpu",
                                  torch.float64)
                cpu_out = cur if cpu_model is None else _run_module(
                    _module(cpu_model, c["name"]), *_clone((args, kwargs)), "cpu", dtype)
                noise = diff_lists(cpu_out, r64).rel_err
                if noise != noise or noise > floor * DIVERGENT_FACTOR:
                    noise = None  # the CPU itself is off: never let that widen the band
                m.exact_recorded = diff_lists(rec_out, r64).rel_err
                m.exact_current = diff_lists(cur, r64).rel_err
            except Exception:
                noise = None
        m.noise_band = threshold_for(dtype, False, noise)
        m.precision_band = threshold_for(dtype, True, noise) if explained else None
        m.diff = Diff(0.0, 0.0, False, False) if same else diff_lists(cur, rec_out)
        m.isolated = True
        m.cls = classify(m.diff, same, m.noise_band, m.precision_band, _runs_alone(c))
    if failed:
        rep.add("TS067", f"{len(failed)} module call(s) could not be replayed: {failed[:5]} "
                         f"({out[failed[0]].error})")
    _finish(rep, a.calls, out)
    root = a.by_key.get(f"{ROOT}#0")
    if root is not None and f"{ROOT}#0" in rec_inputs and out[root["key"]].cls != WEIGHTS:
        args, kwargs = rec_inputs[root["key"]]
        cur = _run_module(plain_live, *_clone((args, kwargs)), device, dtype)
        rep.baseline = _baseline(cur, [a.tensor(i) for i in a.header["plain_out"]])
        hooked = _run_module(live, *_clone((args, kwargs)), device, dtype)
        live_differs = S.hashes(hooked) != S.hashes(cur)
        _path_note(rep, a, live_differs=live_differs)
    else:
        _path_note(rep, a)
    b = rep.blamed
    if b is not None and b.cls in (PRECISION_CHANGE, DIVERGED) and b.key in rec_inputs:
        args, kwargs = rec_inputs[b.key]
        mod = _module(live, b.name)
        if a.ops_under(b.key):
            rep.blamed_op = _op_recorded(a, b.key, mod, args, kwargs, device, dtype,
                                         b.noise_band, b.precision_band)
        else:
            rep.blamed_op = _op_float64(base, b, mod, args, kwargs, device, dtype, rep)
        if rep.blamed_op is None:
            rep.notes.append(f"op drilldown on {b.key}: no single ATen op was confirmed changed "
                             "for its own inputs")
    _blame_source(base, rep)
    if repro_dir and rep.blamed_op is not None and rep.blamed_op.args is not None:
        from .repro import write_op_repro

        try:
            rep.op_repro_path = write_op_repro(repro_dir, rep, device)
        except Exception as exc:
            rep.notes.append(f"no op repro written ({type(exc).__name__}: {exc})")
    return rep


def _hooked(model: nn.Module) -> nn.Module:
    """Attach do nothing forward hooks to every module, as the recording did, so a replayed
    module takes the same code path (hooks switch off some fused fast paths)."""
    for mod in model.modules():
        mod.register_forward_pre_hook(lambda *a, **k: None, with_kwargs=True)
        mod.register_forward_hook(lambda *a, **k: None, with_kwargs=True)
    return model


def _clone(obj: Any) -> Any:
    return S.alias_map(obj, torch.clone)


def _replay_outputs_only(a: Recording, base: nn.Module, device: str,
                         repro_dir: Optional[str]) -> DiffReport:
    root = a.by_key[f"{ROOT}#0"]
    args, kwargs = a.inputs(root)
    ops = a.header.get("ops", "none")
    b = record(base, args, kwargs, device=device, outputs_only=True,
               ops=None if ops == "none" else ops, max_bytes=1 << 62,
               model_name=a.header["model"]["name"])
    rep = diff_recordings(a, b, mode="replay_outputs_only", current_label="current environment")
    rep.notes.append("outputs only recording: modules are isolated by input hash, so only the "
                     "first changed module is measured on its own")
    _blame_source(base, rep)
    return rep


def _op_recorded(a: Recording, key: str, mod: nn.Module, args, kwargs, device: str,
                 dtype: torch.dtype, t_n: float, t_p: Optional[float]) -> Optional[OpChange]:
    """Feed every op inside `mod` the inputs it had in the old environment."""
    from ..opdrill import _align, _call, _comparable, _leaves, trace_ops

    rec_ops = a.ops_under(key)
    dev_args, dev_kwargs = _cast(_clone((args, kwargs)), device, dtype)
    live_ops, _ = trace_ops(mod, dev_args, dev_kwargs)
    live_ops = [o for o in live_ops if _comparable(o)]
    pairs = _align([o["name"] for o in rec_ops], [o.name for o in live_ops])
    first: Optional[OpChange] = None
    for i, lo in enumerate(live_ops):
        if i not in pairs:
            continue
        ro = rec_ops[pairs[i]]
        want = _hashes_of(a, ro["out"])
        if S.hashes(lo.outputs) == want:
            continue
        try:
            r_args, r_kwargs = S.decode_call(ro["args"], ro["kwargs"], a.tensor)
            got = _call(lo.func, r_args, r_kwargs, device)
        except Exception:
            continue
        ref = a.outputs(ro)
        iso = diff_lists(got, ref)
        cls = classify(iso, S.hashes(got) == want, t_n, t_p, True)
        if RANK[cls] <= RANK[NOISE]:
            continue  # inherited from an earlier op in this module
        leaves = _leaves((r_args, r_kwargs))
        ch = OpChange(lo.name, key, i + 1, len(live_ops), cls, iso, t_n, "recorded_trace", True,
                      [list(t.shape) for t in leaves],
                      [S.dtype_name(t.dtype) for t in leaves], r_args, r_kwargs, ref)
        if cls == DIVERGED:
            return ch
        first = first or ch
    return first


def _op_float64(base: nn.Module, b: ModuleChange, mod: nn.Module, args, kwargs, device: str,
                dtype: torch.dtype, rep: DiffReport) -> Optional[OpChange]:
    """No recorded op trace: let opdrill find the op that is off from float64 math here."""
    from ..opdrill import drill

    try:
        ref_mod = copy.deepcopy(_module(base, b.name)).to(torch.float64)
        a64, k64 = _cast((args, kwargs), "cpu", torch.float64)  # float32 to float64 is exact
        op, notes = drill(ref_mod, mod, b.key, a64, k64, device, dtype, b.noise_band,
                          torch.device(device).type == "cpu")
    except Exception as exc:
        rep.notes.append(f"op drilldown failed on {b.key}: {type(exc).__name__}: {exc}")
        return None
    rep.notes.extend(notes)
    rep.notes.append(f"no op trace for {b.key} in the recording, so its ops were compared "
                     f"with float64 CPU math in this environment; record with --ops {b.name} to "
                     "compare against the recorded environment op by op")
    if op is None:
        return None
    cls = classify(op.isolated, False, b.noise_band, b.precision_band, True)
    return OpChange(op.op, b.key, op.index, op.total, cls, op.isolated, op.threshold,
                    "float64_reference", True, op.input_shapes, op.input_dtypes, op.args,
                    op.kwargs, op.reference)
