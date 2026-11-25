"""`truestar conform`: does the deployed file compute the model, in the runtime that will run it?

For each runtime:

1. **Verdict from the unmodified engine.** The artifact runs as users run it on every sample;
   each output is compared with the PyTorch model in float64 (relative L2 against a threshold
   calibrated on a float32 PyTorch run, as `truestar check` does), elements beyond the derived
   bound (docs/TOLERANCE.md) are counted, and decisions are compared per row with near ties
   separated.
2. **Two verdicts.** The file's own math is evaluated in float64 (onnx.reference, the TFLite rules in
   tflite_graph, the IR reference for .tsx). Conversion verdict: file in float64 against the model
   in float64. Runtime verdict: the runtime against the file in float64. A converter fault and a
   runtime fault are told apart instead of both showing up as "the outputs differ".
3. **Precision.** A runtime whose error fits a correct float16 implementation (the float16
   rounded run of the model) is reported as computing in float16 (TS627), not as a bug.
4. **Residency.** Where every node ran (provider, delegate, accelerator), named; nodes off the
   requested device are TS628, a runtime that cannot say is TS629.
5. **Localization only after a mismatch, on cuts.** Each module mapped into the file is cut out as
   a model of its own, run in the same runtime on its float64 reference inputs, so its error is its
   own; descent over the module tree names the deepest module at fault and says whether the file
   (conversion) or the runtime is wrong there.
6. **Observer guard.** Every instrumented run (a profiled session, preserved tensors, extra
   outputs) is compared with two plain runs; when it differs by more than the plain runs differ
   from each other, its per node data is marked untrusted (TS630).
7. **Golden bundle** with inputs, float64 expected outputs, thresholds, derived bounds, each
   runtime's outputs and fingerprint, and per module cut outputs (docs/CONFORM.md).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ..capture import flatten_tensors
from ..compare import DIVERGENT, DTYPE_FLOOR, OK, SUSPECT, TF32_FLOOR, diff_tensors, threshold_for, worst
from . import artifact as A
from . import runtimes as R

UNAVAILABLE, FAILED = "unavailable", "failed"
CONVERSION_BAR = 1e-7  # file in float64 against the model in float64 (measured agreement ~1e-12)
MAX_UNIT_CUTS = 48


# ----------------------------------------------------------------------------- results

@dataclass
class OutputVerdict:
    index: int
    name: str
    shape: List[int]
    status: str
    rel_err: float
    max_abs: float
    threshold: float
    beyond_bound: int = 0
    total: int = 0
    note: str = ""

    def as_dict(self):
        return {k: (_num(v) if isinstance(v, float) else v) for k, v in vars(self).items()}


@dataclass
class Blame:
    module: str
    own_error: float
    threshold: float
    status: str
    cause: str  # "conversion" (the file computes something else) or "runtime"
    conversion_error: Optional[float]
    ops: List[str]
    nodes: List[str]
    user_source: Optional[Dict[str, Any]] = None
    path: List[Dict[str, Any]] = field(default_factory=list)  # descent steps

    def as_dict(self):
        d = dict(vars(self))
        d["own_error"] = _num(self.own_error)
        d["conversion_error"] = _num(self.conversion_error) if self.conversion_error is not None else None
        return d


@dataclass
class RuntimeResult:
    runtime: str
    status: str
    issues: List[Tuple[str, str]] = field(default_factory=list)
    outputs: List[OutputVerdict] = field(default_factory=list)
    decisions: int = 0
    flips: int = 0
    tie_flips: int = 0
    precision: Dict[str, Any] = field(default_factory=dict)
    verdicts: Dict[str, Any] = field(default_factory=dict)
    residency: Optional[Dict[str, Any]] = None
    observer: List[Dict[str, Any]] = field(default_factory=list)
    blamed: Optional[Blame] = None
    units_checked: List[Dict[str, Any]] = field(default_factory=list)
    fingerprint: Dict[str, Any] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    raw: List[List[np.ndarray]] = field(default_factory=list)  # per sample outputs (not in JSON)
    unit_raw: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    def add(self, code: str, msg: str):
        self.issues.append((code, msg))

    def as_dict(self):
        from ..codes import Issue

        return {"runtime": self.runtime, "status": self.status,
                "codes": sorted({c for c, _ in self.issues}),
                "issues": [Issue(c, m, self.runtime).as_dict() for c, m in self.issues],
                "outputs": [o.as_dict() for o in self.outputs],
                "decisions": self.decisions, "flips": self.flips, "tie_flips": self.tie_flips,
                "precision": self.precision, "verdicts": self.verdicts, "residency": self.residency,
                "observer": self.observer, "blamed": self.blamed.as_dict() if self.blamed else None,
                "units_checked": self.units_checked, "fingerprint": self.fingerprint,
                "info": _jsonable(self.info), "seconds": round(self.seconds, 3)}


@dataclass
class ConformReport:
    artifact: Dict[str, Any]
    samples: int
    results: List[RuntimeResult]
    golden: Optional[str] = None
    issues: List[Tuple[str, str]] = field(default_factory=list)
    reference_seconds: float = 0.0
    units: int = 0

    @property
    def status(self) -> str:
        run = [r.status for r in self.results if r.status not in (UNAVAILABLE,)]
        if any(c for c, _ in self.issues if c in ("TS620", "TS637")):
            return FAILED
        st = [DIVERGENT if s == FAILED else s for s in run]
        return worst(*st) if st else FAILED

    def to_text(self) -> str:
        a = self.artifact
        lines = [f"truestar conform: {a['path']} ({a['format']}, {a['nodes']} nodes, sha256 "
                 f"{a['sha256'][:12]}), {self.samples} samples, {self.units} modules named in the file"]
        for c, m in self.issues:
            lines.append(f"  [{c}] {m}")
        for r in self.results:
            if r.status == UNAVAILABLE:
                lines.append(f"  {r.runtime:<15} UNAVAILABLE  {r.issues[0][1] if r.issues else ''}")
                continue
            w = max(r.outputs, key=lambda o: o.rel_err) if r.outputs else None
            flips = (f"{r.flips - r.tie_flips}/{r.decisions} flipped" + (f" ({r.tie_flips} near ties)" if r.tie_flips else "")
                     if r.decisions else "no decisions")
            err = f"max rel {w.rel_err:.2g}" if w else ""
            lines.append(f"  {r.runtime:<15} {r.status.upper():<10} {flips}  {err}")
            if r.precision.get("observed") and r.precision.get("observed") != r.precision.get("declared"):
                lines.append(f"      precision: declared {r.precision['declared']}, observed {r.precision['observed']}")
            res = r.residency
            if res:
                if res["off_target"]:
                    names = ", ".join(f"{p['op']} ({p['node']}{' in ' + p['module'] if p['module'] else ''}) on {p['where']}"
                                      for p in res["off_target"][:6])
                    lines.append(f"      residency: {len(res['off_target'])} of {res['nodes']} nodes off "
                                 f"{res['requested']}: {names}")
                elif res["known"]:
                    lines.append(f"      residency: all {res['nodes']} nodes on {res['requested']} ({res['method']})")
                else:
                    lines.append(f"      residency: unknown ({res['method']})")
            for ob in r.observer:
                lines.append(f"      observer check ({ob['run']}): " + (
                    f"identical to the plain runs (difference {ob['effect']:.2g}, run to run {ob['noise']:.2g}): data trusted"
                    if ob["trusted"] else
                    f"CHANGED the outputs (difference {ob['effect']:.2g}, run to run {ob['noise']:.2g}, "
                    f"{ob['decisions_changed']} decisions differ): its per node data is untrusted"))
            v = r.verdicts
            if v.get("conversion") is not None and r.status != OK:
                lines.append(f"      file in float64 vs model: rel {v['conversion']:.2g}; runtime vs file: "
                             f"rel {v.get('runtime', float('nan')):.2g}")
            if r.blamed:
                b = r.blamed
                src = ""
                if b.user_source:
                    src = f" ({b.user_source.get('class', '')}, {b.user_source.get('file', '')}:{b.user_source.get('line', '')})"
                lines.append(f"      first module at fault: {b.module}{src}, own error {b.own_error:.2g} "
                             f"(limit {b.threshold:.2g}), cause: {b.cause}; nodes {', '.join(b.ops[:8])}")
            for c, m in r.issues:
                lines.append(f"      [{c}] {m}")
        if self.golden:
            lines.append(f"  wrote {self.golden} ({self.samples} samples, float64 expected, thresholds, "
                         f"derived bounds, runtime outputs and fingerprints)")
        lines.append(f"status: {self.status}")
        return "\n".join(lines)


def _num(x):
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float):
        return _num(o)
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


# ----------------------------------------------------------------------------- helpers

def _as_tuple(x) -> tuple:
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def make_samples(example, n: int, seed: int = 0) -> List[tuple]:
    """Sample 0 is the example; the others draw each float tensor from a normal with the example's
    mean and std (per tensor), integer tensors are kept. Shapes stay those of the example, since
    deployed files usually have static shapes."""
    ex = _as_tuple(example)
    g = torch.Generator().manual_seed(seed)
    out = [ex]
    for _ in range(max(0, n - 1)):
        s = []
        for t in ex:
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                std = float(t.double().std()) if t.numel() > 1 else 1.0
                s.append((torch.randn(t.shape, generator=g, dtype=torch.float64) * (std or 1.0)
                          + float(t.double().mean())).to(t.dtype))
            else:
                s.append(t)
        out.append(tuple(s))
    return out


def _decisions(out: np.ndarray, ref: np.ndarray, threshold: float) -> Tuple[int, int, int]:
    from ..check import _decisions as dec

    return dec(torch.from_numpy(np.asarray(out)), torch.from_numpy(np.asarray(ref)), threshold)


def _rel(a: np.ndarray, b: np.ndarray):
    return diff_tensors(torch.from_numpy(np.asarray(a)), torch.from_numpy(np.asarray(b)))


def observer_check(run_name: str, plain_a: List[np.ndarray], plain_b: List[np.ndarray],
                   instrumented: List[np.ndarray], threshold: float) -> Dict[str, Any]:
    """An instrumented run must agree with the plain run within the plain run's own run to run
    noise; otherwise what it shows per node is not what the plain engine computes."""
    def mx(xs, ys):
        m = 0.0
        for x, y in zip(xs, ys):
            x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
            if x.shape != y.shape:
                return float("inf")
            d = np.abs(np.nan_to_num(x - y, nan=np.inf))
            m = max(m, float(d.max()) if d.size else 0.0)
        return m

    noise = mx(plain_a, plain_b)
    effect = mx(instrumented, plain_a) if len(instrumented) == len(plain_a) else float("inf")
    changed = 0
    for x, y in zip(instrumented, plain_a):
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim == 2 and x.shape == y.shape and x.shape[1] > 1:
            changed += int((x.argmax(1) != y.argmax(1)).sum())
    return {"run": run_name, "noise": noise, "effect": effect, "trusted": effect <= noise,
            "decisions_changed": changed}


# ----------------------------------------------------------------------------- reference

class _Reference:
    """Float64 outputs per sample, float32 calibration, derived bounds and per module records."""

    def __init__(self, model: nn.Module, samples: List[tuple], derive_bounds: bool = True):
        t0 = time.perf_counter()
        self.model = model.eval()
        self.m64 = copy.deepcopy(model).double().eval()
        self.m32 = copy.deepcopy(model).float().eval()
        self.samples = samples
        self.out64: List[List[np.ndarray]] = []
        self.noise: List[float] = []
        self.bounds: List[List[np.ndarray]] = []
        noise = None
        for s in samples:
            with torch.no_grad():
                o64 = [t.detach().double().numpy() for t in flatten_tensors(self.m64(*[_d(x) for x in s]))]
                o32 = [t.detach().numpy() for t in flatten_tensors(self.m32(*[_f(x) for x in s]))]
            self.out64.append(o64)
            e = [_rel(a, b).rel_err for a, b in zip(o32, o64)]
            noise = e if noise is None else [max(x, y) for x, y in zip(noise, e)]
        self.noise = noise or []
        if derive_bounds:
            from ..tolerance import derive

            for s in samples:
                try:
                    tol = derive(self.m32, tuple(_f(x) for x in s), dtype=torch.float32, device="cpu",
                                 tf32=False, cache=False)
                    self.bounds.append([tol.bound(i).cpu().numpy() for i in range(len(tol.reference))])
                except Exception:
                    self.bounds.append([np.full(o.shape, np.inf) for o in self.out64[len(self.bounds)]])
        self._rounded: Dict[str, List[float]] = {}
        self.records: Dict[str, Dict[str, Any]] = {}
        self.module_noise: Dict[str, float] = {}
        self.seconds = time.perf_counter() - t0

    def threshold(self, j: int, dtype=torch.float32, tf32: bool = False) -> float:
        return threshold_for(dtype, tf32, self.noise[j] if j < len(self.noise) else None)

    def rounded_error(self, kind: str) -> List[float]:
        """Relative error per output of a correct lower precision implementation, worst over
        samples: every op rounded to float16 ("float16"), or the inputs of every matmul and
        convolution rounded to TF32 ("tf32"), in a float64 run (docs/TOLERANCE.md)."""
        if kind not in self._rounded:
            from ..tolerance import derive

            errs = None
            for s, o64 in zip(self.samples, self.out64):
                try:
                    if kind == "float16":
                        args = tuple(_f(x).half() if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in s)
                        tol = derive(self.m32, args, dtype=torch.float16, device="cpu", tf32=False, samples=1, cache=False)
                    else:
                        tol = derive(self.m32, tuple(_f(x) for x in s), dtype=torch.float32, device="cpu", tf32=True,
                                     samples=1, cache=False)
                    e = [_rel(r.cpu().numpy(), ref).rel_err for r, ref in zip(tol.rounded, o64)]
                except Exception:
                    e = [float("nan")] * len(o64)
                errs = e if errs is None else [max(x, y) for x, y in zip(errs, e)]
            self._rounded[kind] = errs or []
        return self._rounded[kind]

    def record_modules(self) -> None:
        """Inputs and outputs (float64) of every module call on sample 0, and the float32 noise of
        each module run alone on them."""
        if self.records:
            return
        calls: Dict[str, Dict[str, Any]] = {}
        hooks = []
        for name, mod in self.m64.named_modules():
            if not name:
                continue

            def hook(m, args, kwargs, out, name=name):
                c = calls.get(name)
                if c is None:
                    calls[name] = {"inputs": [t.detach().clone() for t in flatten_tensors((args, kwargs))],
                                   "outputs": [t.detach().clone() for t in flatten_tensors(out)],
                                   "args": (args, kwargs), "calls": 1}
                else:
                    c["calls"] += 1

            hooks.append(mod.register_forward_hook(hook, with_kwargs=True))
        try:
            with torch.no_grad():
                self.m64(*[_d(x) for x in self.samples[0]])
        finally:
            for h in hooks:
                h.remove()
        mods32 = dict(self.m32.named_modules())
        for name, c in calls.items():
            try:
                args, kwargs = c["args"]
                from ..capture import map_tensors

                a32, k32 = map_tensors((args, kwargs), lambda t: t.float() if t.is_floating_point() else t)
                with torch.no_grad():
                    o32 = flatten_tensors(mods32[name](*a32, **k32))
                self.module_noise[name] = max((_rel(x.numpy(), y.numpy()).rel_err
                                               for x, y in zip(o32, c["outputs"])), default=0.0)
            except Exception:
                self.module_noise[name] = float("nan")
            c.pop("args", None)
        self.records = calls


def _d(x):
    return x.double() if isinstance(x, torch.Tensor) and x.is_floating_point() else x


def _f(x):
    return x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x


def _feeds(sample: tuple, specs) -> List[np.ndarray]:
    ts = [t for t in flatten_tensors(sample)]
    out = []
    for i, t in enumerate(ts):
        dt = specs[i][2] if i < len(specs) else np.float32
        a = t.detach().cpu().numpy()
        out.append(a.astype(dt) if a.dtype.kind == "f" and np.dtype(dt).kind == "f" else a)
    return out


# ----------------------------------------------------------------------------- module map

def _match(arrays: List[np.ndarray], tensors: List[torch.Tensor]) -> Optional[List[int]]:
    """For each artifact array, the index of the module tensor it is: the only one of that shape,
    or among several of that shape the closest in value. Values only break ties, since a faulty
    file's values differ from the model's by definition. None when any array has no counterpart."""
    used = set()
    out = []
    for a in arrays:
        cands = [k for k, t in enumerate(tensors) if k not in used and tuple(t.shape) == tuple(np.shape(a))]
        if not cands:
            return None
        if len(cands) == 1:
            best = cands[0]
        else:
            def err(k):
                t = tensors[k]
                if np.asarray(a).dtype.kind == "f" and t.is_floating_point():
                    return _rel(np.asarray(a, np.float64), t.double().numpy()).rel_err
                return 0.0 if np.array_equal(np.asarray(a), t.numpy()) else float("inf")
            best = min(cands, key=err)
        used.add(best)
        out.append(best)
    return out


def _tree(units: List[A.Unit]) -> Tuple[List[A.Unit], Dict[str, List[A.Unit]]]:
    paths = {u.module for u in units}
    kids: Dict[str, List[A.Unit]] = {}
    roots = []
    for u in units:
        parts = u.module.split(".")
        parent = None
        for k in range(len(parts) - 1, 0, -1):
            p = ".".join(parts[:k])
            if p in paths:
                parent = p
                break
        if parent is None:
            roots.append(u)
        else:
            kids.setdefault(parent, []).append(u)
    return roots, kids


def _unit_source(model: nn.Module, module: str) -> Optional[Dict[str, Any]]:
    try:
        from ..agent import module_source

        src = module_source(model, module)
        u = src.get("user") or src.get("module")
        if u is None:
            return None
        return {"class": u.get("qualname", "").rsplit(".", 1)[-1], "file": u.get("file"),
                "line": u.get("line"), "module": u.get("module", module)}
    except Exception:
        return None


class _Localizer:
    """Own error of each mapped module, from cuts run in the same runtime."""

    def __init__(self, art: A.Artifact, ref: _Reference):
        self.art = art
        self.ref = ref
        self.units = art.units() if art.can_cut else []
        self.boundary: Dict[str, np.ndarray] = {}
        self.matches: Dict[str, Optional[Tuple[List[int], List[int]]]] = {}
        self.skipped: List[str] = []
        self.cut_outputs: Dict[str, Dict[str, List[np.ndarray]]] = {}

    def prepare(self) -> None:
        if not self.units:
            return
        self.ref.record_modules()
        # the file's own values at every unit boundary, float64, to match tensors to module tensors
        wanted = sorted({t for u in self.units for t in u.inputs + u.outputs})
        specs = self.art.input_specs()
        feeds = {n: a for (n, _, _), a in zip(specs, _feeds(self.ref.samples[0], specs))}
        vals = self.art.meaning64(feeds, wanted) if self.art.can_mean else None
        if vals is not None:
            self.boundary = dict(zip(wanted, vals))
        for u in self.units:
            rec = self.ref.records.get(u.module)
            if rec is None or rec["calls"] != 1:
                self.matches[u.module] = None
                continue
            if self.boundary:
                ins = [self.boundary[t] for t in u.inputs]
                outs = [self.boundary[t] for t in u.outputs]
            else:
                ins = [np.zeros(self._shape(t)) for t in u.inputs]
                outs = [np.zeros(self._shape(t)) for t in u.outputs]
            mi = _match(ins, rec["inputs"]) if self.boundary else _shape_match(ins, rec["inputs"])
            mo = _match(outs, rec["outputs"]) if self.boundary else _shape_match(outs, rec["outputs"])
            self.matches[u.module] = (mi, mo) if mi is not None and mo is not None else None

    def _shape(self, tid: str):
        if self.art.format == A.TFLITE:
            return self.art.graph.shape(self.art.index(tid))
        return ()

    def mapped(self) -> List[A.Unit]:
        return [u for u in self.units if self.matches.get(u.module)]

    def own(self, rt: R.Runtime, u: A.Unit) -> Dict[str, Any]:
        """Run the unit's cut on its float64 reference inputs (cast to the file's dtypes)."""
        mi, mo = self.matches[u.module]
        rec = self.ref.records[u.module]
        feeds64 = [rec["inputs"][k].numpy() for k in mi]
        exp = [rec["outputs"][k].numpy() for k in mo]
        dts = [self._dtype(t) for t in u.inputs]
        feeds = [self.art.encode(t, f, dt) for t, f, dt in zip(u.inputs, feeds64, dts)]
        data = self.art.cut(u)
        s = rt.open(data, self.art)
        try:
            got = [self.art.decode(t, g) for t, g in zip(u.outputs, s.run(feeds))]
        finally:
            s.close()
        self.cut_outputs.setdefault(u.module, {})[rt.name] = got
        err = max((_rel(g, e).rel_err for g, e in zip(got, exp)), default=0.0)
        noise = self.ref.module_noise.get(u.module)
        tf32 = bool(rt.options.get("_tf32")) and any(op in _MATMUL_OPS for op in u.ops)
        # a runtime shown to compute in float16 is judged per module as float16, so descent looks
        # for what float16 alone does not explain (an overflow, a wrong kernel)
        dt = torch.float16 if rt.options.get("_precision") == "float16" else torch.float32
        thr = threshold_for(dt, tf32, noise)
        d = [_rel(g, e) for g, e in zip(got, exp)]
        st = worst(*[x.status(thr) for x in d]) if d else OK
        conv = None
        if self.art.can_mean:
            m = self.art.meaning64({t: f for t, f in zip(u.inputs, feeds64)}, u.outputs, u)
            if m is not None:
                conv = max((_rel(a, e).rel_err for a, e in zip(m, exp)), default=0.0)
        bar = max(CONVERSION_BAR, 2.0 * noise) if noise is not None and noise == noise else CONVERSION_BAR
        return {"module": u.module, "own_error": err, "threshold": thr, "status": st,
                "conversion_error": conv, "conversion_bar": bar}

    def _dtype(self, tid: str):
        if self.art.format == A.TFLITE:
            return self.art.graph.dtype(self.art.index(tid))
        if self.art.format == A.ONNX:
            for vi in list(self.art.model.graph.value_info) + list(self.art.model.graph.input):
                if vi.name == tid:
                    return A._ONNX_NP.get(vi.type.tensor_type.elem_type, np.float32)
        return np.float32

    def descend(self, rt: R.Runtime, result: RuntimeResult) -> Optional[Blame]:
        mapped = self.mapped()
        if not mapped:
            return None
        roots, kids = _tree(mapped)
        level = roots
        path = []
        blamed = None
        budget = MAX_UNIT_CUTS
        while level and budget > 0:
            bad = None
            for u in level:
                budget -= 1
                try:
                    o = self.own(rt, u)
                except Exception as exc:
                    result.units_checked.append({"module": u.module, "error": f"{type(exc).__name__}: {exc}"[:200]})
                    continue
                result.units_checked.append({k: (_num(v) if isinstance(v, float) else v) for k, v in o.items()})
                conv_bad = o["conversion_error"] is not None and o["conversion_error"] > o["conversion_bar"]
                if o["status"] != OK or conv_bad:
                    bad = (u, o)
                    break
                if budget <= 0:
                    break
            if bad is None:
                break
            blamed = bad
            path.append({"module": bad[0].module, "own_error": _num(bad[1]["own_error"])})
            level = kids.get(bad[0].module, [])
        if blamed is None:
            return None
        u, o = blamed
        conv = o["conversion_error"]
        conv_bad = conv is not None and conv > o["conversion_bar"]
        # the file is at fault when its float64 math is off and the runtime's own error is within
        # what its precision allows, or the file's error is a large part of it
        cause = "conversion" if conv_bad and (o["status"] == OK or conv >= 0.1 * o["own_error"]) else "runtime"
        nodes = [self.art.node_info(i)["node"] for i in u.nodes]
        return Blame(u.module, o["own_error"], o["threshold"], o["status"], cause, conv, list(u.ops),
                     nodes, _unit_source(self.ref.model, u.module), path)


_MATMUL_OPS = {"Gemm", "MatMul", "Conv", "ConvTranspose", "FULLY_CONNECTED", "CONV_2D",
               "BATCH_MATMUL", "DEPTHWISE_CONV_2D"}


def _shape_match(arrays, tensors):
    used, out = set(), []
    for a in arrays:
        k = next((k for k, t in enumerate(tensors) if k not in used and tuple(t.shape) == tuple(np.shape(a))), None)
        if k is None:
            return None
        used.add(k)
        out.append(k)
    return out


# ----------------------------------------------------------------------------- main

def conform(model: nn.Module, example_inputs, artifact_path: str, *, runtimes: Optional[Sequence[str]] = None,
            samples: int = 8, inputs: Optional[List[tuple]] = None, golden: Optional[str] = None,
            options: Optional[Dict[str, Dict[str, Any]]] = None, localize: bool = True,
            model_file: Optional[str] = None) -> ConformReport:
    """Check a deployed artifact against the PyTorch model it came from.

    `runtimes`: names from runtimes.RUNTIMES (default: the format's defaults that can run here).
    `inputs`: explicit samples (each a tensor or tuple); otherwise `samples` are drawn around the
    example. `golden`: path of the bundle to write. `options[runtime]` passes runtime options and
    test hooks (`fault_instrumented`, `fault_outputs`: functions applied to outputs).
    """
    art = A.load(artifact_path)
    options = options or {}
    smp = [_as_tuple(s) for s in inputs] if inputs else make_samples(example_inputs, samples)
    report = ConformReport(art.describe(), len(smp), [])
    names = list(runtimes) if runtimes else None
    explicit = names is not None
    if names is None:
        names = R.DEFAULTS.get(art.format, [])
    for n in names:
        if n not in R.RUNTIMES:
            raise ValueError(f"unknown runtime {n!r}; known: {', '.join(R.RUNTIMES)}")
        if R.RUNTIMES[n][0] != art.format:
            raise ValueError(f"runtime {n} runs {R.RUNTIMES[n][0]} files, not {art.format}")
    specs = art.input_specs()
    nin = len(flatten_tensors(smp[0]))
    if specs and len(specs) != nin:
        report.issues.append(("TS637", f"the artifact takes {len(specs)} inputs "
                                       f"({', '.join(s[0] for s in specs)}) but get_inputs() gives {nin} tensors"))
        return report
    ref = _Reference(model, smp)
    report.reference_seconds = ref.seconds
    loc = _Localizer(art, ref) if localize else None
    if art.can_cut:
        report.units = len(loc.units if loc is not None else art.units())
    prepared = False
    meaning = None
    if art.can_mean:
        meaning = []
        for s in smp:
            feeds = {n: a for (n, _, _), a in zip(specs, _feeds(s, specs))}
            m = art.meaning64(feeds, art.output_names())
            if m is None:
                meaning = None
                break
            meaning.append(m)
    if art.can_mean and meaning is None:
        report.issues.append(("TS638", f"the {art.format} file's float64 meaning could not be evaluated (an op "
                                       "without a float64 rule); conversion and runtime faults are not told apart"))
    for name in names:
        rt = R.make(name, options.get(name))
        res = RuntimeResult(name, OK)
        why = rt.unavailable()
        if why:
            res.status = UNAVAILABLE
            res.add("TS621", why)
            if explicit:
                report.results.append(res)
            else:
                res.info["skipped"] = True
                report.results.append(res)
            continue
        t0 = time.perf_counter()
        try:
            _run_runtime(rt, art, ref, smp, specs, res, meaning)
        except (R.RuntimeFailed, R.RuntimeUnavailable) as exc:
            res.status = FAILED
            res.add("TS636", str(exc))
        if res.status in (SUSPECT, DIVERGENT) and loc is not None and art.can_cut:
            if not prepared:
                loc.prepare()
                prepared = True
            try:
                res.blamed = loc.descend(rt, res)
            except Exception as exc:
                res.info["localization_error"] = f"{type(exc).__name__}: {exc}"[:300]
            if res.blamed is None and not loc.mapped():
                res.add("TS631", "no artifact node could be mapped to a module of the model (the file keeps "
                                 "no module names, or they do not match), so the fault is not localized")
            elif res.blamed is None and any(c == "TS627" for c, _ in res.issues):
                res.info["localization"] = (f"no module is beyond what {res.precision.get('observed')} compute "
                                            "explains when cut out and run alone")
            elif res.blamed is None:
                res.add("TS631", "every mapped module matches when cut out and run alone: the fault needs "
                                 "several modules together, or lives in nodes outside any module")
            else:
                b = res.blamed
                code = "TS625" if b.cause == "conversion" else "TS626"
                what = ("the file itself computes something else here (its float64 evaluation differs "
                        f"from the model by {b.conversion_error:.2g}): a conversion fault"
                        if b.cause == "conversion" else
                        "the file is right here in float64; the runtime computes it wrongly"
                        if b.conversion_error is not None else
                        "the file's float64 meaning could not be evaluated here (TS638), so the file "
                        "and the runtime are not told apart")
                res.add(code, f"module {b.module}: own error {b.own_error:.2g} (limit {b.threshold:.2g}) "
                              f"when cut out and run alone in {name}; {what}. Nodes: {', '.join(b.ops[:10])}")
        elif res.status in (SUSPECT, DIVERGENT) and art.format == A.PTE:
            res.add("TS631", "a .pte program keeps no module names, so the fault is not localized below "
                             "the outputs; residency names the kernels outside the delegate")
        res.seconds = time.perf_counter() - t0
        report.results.append(res)
    if prepared or (golden and loc is not None and art.can_cut):
        if not prepared and golden:
            try:
                loc.prepare()
            except Exception:
                pass
    if golden:
        from .bundle import write_bundle

        write_bundle(golden, art, ref, smp, specs, report, loc, model_file, options)
        report.golden = golden
    return report


def _apply(hook, outs):
    return [np.asarray(o) for o in hook(outs)] if hook else outs


def _run_runtime(rt: R.Runtime, art: A.Artifact, ref: _Reference, smp, specs, res: RuntimeResult,
                 meaning) -> None:
    res.fingerprint = rt.fingerprint()
    s = rt.open(art.data, art)
    fault = rt.options.get("fault_outputs")
    try:
        res.info.update(s.info)
        tf32 = bool(s.info.get("tf32"))
        rt.options["_tf32"] = tf32
        outs_all = []
        for k, sample in enumerate(smp):
            outs_all.append(_apply(fault, art.decode_outputs(s.run(art.encode_inputs(_feeds(sample, specs))))))
        again = _apply(fault, art.decode_outputs(s.run(art.encode_inputs(_feeds(smp[0], specs)))))
        residency = None
        if rt.format != A.ONNX:
            residency = rt.residency(art, s)
    except Exception:
        s.close()
        raise
    res.raw = outs_all
    onames = art.output_names()
    declared = "float32"
    nouts = len(ref.out64[0])
    if len(outs_all[0]) != nouts:
        res.status = DIVERGENT
        res.add("TS622", f"the artifact gives {len(outs_all[0])} outputs, the model {nouts}")
        s.close()
        return
    # per output verdict over all samples
    for j in range(nouts):
        thr = ref.threshold(j, tf32=tf32)
        worst_d, beyond, total = None, 0, 0
        st = OK
        for k in range(len(smp)):
            d = _rel(outs_all[k][j], ref.out64[k][j])
            st = worst(st, d.status(thr))
            if worst_d is None or d.rel_err > worst_d.rel_err or d.shape_mismatch:
                worst_d = d
            if k < len(ref.bounds) and j < len(ref.bounds[k]) and not d.shape_mismatch:
                e = np.abs(np.asarray(outs_all[k][j], np.float64) - ref.out64[k][j])
                beyond += int((np.nan_to_num(e, nan=np.inf) > ref.bounds[k][j]).sum())
                total += e.size
            n, f, t = _decisions(outs_all[k][j], ref.out64[k][j], thr)
            res.decisions += n
            res.flips += f
            res.tie_flips += t
        res.outputs.append(OutputVerdict(j, onames[j] if j < len(onames) else str(j),
                                         list(np.shape(outs_all[0][j])), st, worst_d.rel_err,
                                         worst_d.max_abs, thr, beyond, total, worst_d.note))
    res.status = worst(*[o.status for o in res.outputs])
    # two verdicts: the file's math in float64 against the model, the runtime against the file
    if meaning is not None:
        conv = max(_rel(meaning[k][j], ref.out64[k][j]).rel_err for k in range(len(smp)) for j in range(nouts))
        run_e = max(_rel(outs_all[k][j], meaning[k][j]).rel_err for k in range(len(smp)) for j in range(nouts))
        # the bar: what float32 rounding of the model itself moves it (constants written in
        # float32, a scalar the model holds in float64), never below CONVERSION_BAR
        bar = max([CONVERSION_BAR] + [2.0 * n for n in ref.noise])
        res.verdicts = {"conversion": conv, "runtime": run_e, "conversion_bar": bar,
                        "conversion_ok": conv <= bar}
        if conv > bar:
            # the file's own math is off: real even when the float32 error stays under its limit
            res.status = worst(res.status, SUSPECT)
            res.add("TS625", f"the file in float64 differs from the model in float64 by {conv:.2g} (relative): "
                             "the conversion changed the math, before any runtime ran")
    res.precision = {"declared": declared, "observed": "tf32" if tf32 else declared}
    if tf32:
        res.precision["note"] = ("TF32 matmuls are on in this runtime (session option use_tf32, ONNX "
                                 "Runtime's CUDA default): judged with the TF32 tolerance")
    if res.status != OK:
        # against the file's own float64 math when there is one, so a conversion fault does not
        # hide that the runtime also computes in lower precision
        worst_rel = res.verdicts.get("runtime", max(o.rel_err for o in res.outputs))
        kind = "tf32" if tf32 else "float16"
        e = ref.rounded_error(kind)
        if e and all(not math.isnan(x) for x in e):
            floor = TF32_FLOOR if kind == "tf32" else DTYPE_FLOOR[torch.float16]
            lim = max(10 * max(e), floor)
            res.precision[f"{kind}_reference_error"] = max(e)
            if worst_rel <= lim and worst_rel > 0.2 * max(e):
                res.precision["observed"] = kind
                rt.options["_precision"] = kind
                what = ("computes this float32 file in float16" if kind == "float16" else
                        "rounds matmul inputs to TF32, and the rest of the model amplifies it (a "
                        "quantization step, a near tie)")
                res.add("TS627", f"the error ({worst_rel:.2g}) is what a correct {kind} implementation shows "
                                 f"({max(e):.2g}): {rt.name} {what}")
    real_flips = res.flips - res.tie_flips
    if real_flips > 0:
        res.add("TS624", f"{real_flips} of {res.decisions} decisions changed beyond near ties "
                         f"({res.tie_flips} near ties)")
        res.status = DIVERGENT
    if res.status == DIVERGENT and not any(c in ("TS627",) for c, _ in res.issues):
        w = max(res.outputs, key=lambda o: o.rel_err)
        res.add("TS622", f"output {w.name}: relative error {w.rel_err:.2g} against float64 (limit {w.threshold:.2g})")
    elif res.status == SUSPECT and not any(c == "TS627" for c, _ in res.issues):
        w = max(res.outputs, key=lambda o: o.rel_err)
        res.add("TS623", f"output {w.name}: relative error {w.rel_err:.2g} is above the limit {w.threshold:.2g} "
                         "but below the divergent cutoff")
    # residency (and the observer guard for instrumented runs)
    thr0 = ref.threshold(0)
    if rt.format == A.ONNX and rt.target != "cpu":
        sp = rt.open(art.data, art, profile=True)
        try:
            prof_out = _apply(rt.options.get("fault_instrumented"), _apply(fault, art.decode_outputs(sp.run(art.encode_inputs(_feeds(smp[0], specs))))))
            residency = rt.residency(art, sp)
        finally:
            sp.close()
        res.observer.append(observer_check("profiled session", outs_all[0], again, prof_out, thr0))
    elif rt.format == A.ONNX:
        residency = rt.residency(art, s)
    if rt.format == A.TFLITE and rt.target in ("interp", "ref") and res.status != OK:
        # preserved tensors (the interpreter's debug mode) give the tapped per module profile
        sp = rt.open(art.data, art, preserve=True)
        try:
            tap_out = _apply(rt.options.get("fault_instrumented"), _apply(fault, art.decode_outputs(sp.run(art.encode_inputs(_feeds(smp[0], specs))))))
            res.info["tapped_delegates"] = sum(1 for d in sp.interpreter._get_ops_details() if d.get("op_name") == "DELEGATE")
        finally:
            sp.close()
        res.observer.append(observer_check("preserve_all_tensors", outs_all[0], again, tap_out, thr0))
    s.close()
    if residency is not None and rt.format == A.PTE and residency.off_target:
        _name_kernels(ref.model, smp[0], residency.off_target)
    if residency is not None:
        res.residency = residency.as_dict()
        for code, msg in residency.findings:  # runtime specific facts (LiteRT NPU: TS1124 to TS1126)
            res.add(code, msg)
        if residency.off_target:
            names = ", ".join(f"{p.op} ({p.node}{' in module ' + p.module if p.module else ''})"
                              for p in residency.off_target[:8])
            res.add("TS628", f"{len(residency.off_target)} of {residency.nodes} nodes did not run on "
                             f"{residency.requested} ({residency.off_target[0].where}): {names}. "
                             f"The result is not {rt.name}'s alone")
        elif not residency.known:
            res.add("TS629", f"{rt.name} does not say where nodes ran ({residency.method}); fallback not ruled out")
    for ob in res.observer:
        if not ob["trusted"]:
            res.add("TS630", f"the {ob['run']} changed the outputs by {ob['effect']:.2g} while plain runs differ "
                             f"by {ob['noise']:.2g} ({ob['decisions_changed']} decisions differ): what it shows "
                             "per node is not what the plain engine computes")
            if res.residency is not None and ob["run"] == "profiled session":
                res.residency["trusted"] = False


def _op_stem(name: str) -> str:
    n = name.split("::")[-1].split(".")[0]
    for p in ("native_", "_"):
        if n.startswith(p):
            n = n[len(p):]
    return n


def _name_kernels(model: nn.Module, sample: tuple, placements) -> None:
    """A .pte keeps no module names; the model's own torch.export graph does. Kernel call number k
    of an op is matched to call number k of the same op in the exported graph."""
    try:
        ep = torch.export.export(copy.deepcopy(model).eval(), tuple(_f(x) for x in sample))
    except Exception:
        return
    calls: Dict[str, List[str]] = {}
    for n in ep.graph.nodes:
        if n.op == "call_function":
            stack = n.meta.get("nn_module_stack") or {}
            path = list(stack.values())[-1][0] if stack else ""
            calls.setdefault(_op_stem(str(n.target).split(".")[-2] if "." in str(n.target) else str(n.target)), []).append(path)
    seen: Dict[str, int] = {}
    for p in placements:
        stem = _op_stem(p.op)
        k = seen.get(stem, 0)
        seen[stem] = k + 1
        paths = calls.get(stem, [])
        if k < len(paths):
            p.module = paths[k]
