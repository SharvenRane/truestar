"""Write golden bundles from a conform run, and replay them after an upgrade (the baseline).

`replay(bundle, artifact)` needs no model code: it reruns the artifact in each recorded runtime on
the bundle's inputs, checks the outputs against the float64 expected outputs and thresholds,
compares them with what the same runtime produced when the bundle was written, diffs the runtime
fingerprints (recdiff's classification: version, precision, hardware, algorithm) and, when the
outputs moved, reruns each recorded module cut on its stored inputs to name the first module whose
own output changed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import artifact as A
from . import runtimes as R
from .golden import Bundle, GoldenError, Writer

MAX_UNITS = 32
MAX_UNIT_BYTES = 64 << 20


def write_bundle(path, art, ref, smp, specs, report, loc, model_file, options) -> None:
    from .core import UNAVAILABLE, FAILED, _feeds

    w = Writer()
    inputs = []
    for k, s in enumerate(smp):
        files = []
        for i, a in enumerate(_feeds(s, specs)):
            files.append(w.array(f"inputs/s{k}_{i}.npy", a))
        inputs.append(files)
    outs = []
    for j in range(len(ref.out64[0])):
        for k in range(len(smp)):
            w.array(f"expected/s{k}_{j}.npy", ref.out64[k][j])
            w.array(f"bound/s{k}_{j}.npy", ref.bounds[k][j] if k < len(ref.bounds) else np.full(ref.out64[k][j].shape, np.inf))
        names = art.output_names()
        outs.append({"index": j, "name": names[j] if j < len(names) else str(j),
                     "shape": list(ref.out64[0][j].shape), "threshold": ref.threshold(j),
                     "dtype": str(np.asarray(report.results[0].raw[0][j]).dtype) if report.results and report.results[0].raw else "float32"})
    runtimes: Dict[str, Any] = {}
    ran = [r for r in report.results if r.status not in (UNAVAILABLE, FAILED) and r.raw]
    for r in ran:
        files = []
        for k, o in enumerate(r.raw):
            files.append([w.array(f"runtime/{r.runtime}/s{k}_{j}.npy", a) for j, a in enumerate(o)])
        runtimes[r.runtime] = {"fingerprint": r.fingerprint, "status": r.status, "outputs": files,
                               "residency": r.residency, "options": sorted((options.get(r.runtime) or {}).keys())}
    units = []
    if loc is not None and loc.units:
        size = 0
        for u in loc.mapped()[:MAX_UNITS]:
            mi, mo = loc.matches[u.module]
            rec = ref.records[u.module]
            ins64 = [rec["inputs"][k].numpy() for k in mi]
            exp = [rec["outputs"][k].numpy() for k in mo]
            size += sum(a.nbytes for a in ins64 + exp)
            if size > MAX_UNIT_BYTES:
                break
            tag = f"u{len(units)}"
            entry = {"module": u.module, "inputs": u.inputs, "outputs": u.outputs, "ops": u.ops,
                     "input_files": [w.array(f"units/{tag}/in_{i}.npy", a) for i, a in enumerate(ins64)],
                     "expected_files": [w.array(f"units/{tag}/exp_{i}.npy", a) for i, a in enumerate(exp)],
                     "runtime_files": {}}
            for r in ran:
                rt = R.make(r.runtime, options.get(r.runtime))
                got = loc.cut_outputs.get(u.module, {}).get(r.runtime)
                if got is None:
                    try:
                        got = _run_cut(rt, art, u, ins64, loc)
                    except Exception:
                        continue
                entry["runtime_files"][r.runtime] = [w.array(f"units/{tag}/{r.runtime}_{i}.npy", a)
                                                     for i, a in enumerate(got)]
            units.append(entry)
    from .. import __version__

    manifest = {"truestar_version": __version__, "samples": len(smp), "inputs": inputs,
                "input_specs": [{"name": n, "shape": list(sh), "dtype": np.dtype(dt).name} for n, sh, dt in specs],
                "outputs": outs, "artifact": {"name": os.path.basename(art.path), "format": art.format,
                                               "sha256": art.sha256},
                "model_file": os.path.basename(model_file) if model_file else None,
                "runtimes": runtimes, "units": units}
    w.write(path, manifest)


def _run_cut(rt, art, u, ins64, loc) -> List[np.ndarray]:
    dts = [loc._dtype(t) for t in u.inputs]
    feeds = [art.encode(t, f, dt) for t, f, dt in zip(u.inputs, ins64, dts)]
    s = rt.open(art.cut(u), art)
    try:
        return [art.decode(t, g) for t, g in zip(u.outputs, s.run(feeds))]
    finally:
        s.close()


# ----------------------------------------------------------------------------- replay

@dataclass
class ReplayResult:
    runtime: str
    verdict: str  # identical, changed, regressed, unavailable, failed
    conformant: bool = True
    max_change: float = 0.0
    rel_change: float = 0.0
    decisions_moved: int = 0
    worst_rel_err: float = 0.0
    fingerprint_changes: List[Tuple[str, Any, Any, str]] = field(default_factory=list)
    first_changed_module: Optional[Dict[str, Any]] = None
    modules: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[Tuple[str, str]] = field(default_factory=list)

    def as_dict(self):
        from ..codes import Issue
        from .core import _num

        return {"runtime": self.runtime, "verdict": self.verdict, "conformant": self.conformant,
                "max_change": _num(self.max_change), "rel_change": _num(self.rel_change),
                "decisions_moved": self.decisions_moved, "worst_rel_err": _num(self.worst_rel_err),
                "fingerprint_changes": [{"key": k, "recorded": a, "current": b, "kind": kd}
                                        for k, a, b, kd in self.fingerprint_changes],
                "first_changed_module": self.first_changed_module, "modules": self.modules,
                "codes": sorted({c for c, _ in self.issues}),
                "issues": [Issue(c, m, self.runtime).as_dict() for c, m in self.issues]}


def replay(bundle_path: str, artifact_path: str, runtimes: Optional[Sequence[str]] = None,
           options: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[List[ReplayResult], List[Tuple[str, str]]]:
    from ..recdiff.fingerprint import differences
    from .core import _rel

    options = options or {}
    b = Bundle(bundle_path)
    m = b.manifest
    art = A.load(artifact_path)
    top: List[Tuple[str, str]] = []
    if art.sha256 != m["artifact"]["sha256"]:
        top.append(("TS635", f"{artifact_path} (sha256 {art.sha256[:12]}) is not the artifact the bundle was "
                             f"written for ({m['artifact']['name']}, sha256 {m['artifact']['sha256'][:12]}); "
                             "a replay compares runtimes on the same file"))
        return [], top
    names = list(runtimes) if runtimes else sorted(m.get("runtimes", {}))
    results = []
    for name in names:
        rec = m.get("runtimes", {}).get(name)
        res = ReplayResult(name, "identical")
        results.append(res)
        if rec is None:
            res.verdict = "failed"
            res.issues.append(("TS632", f"the bundle has no recording for {name} (recorded: "
                                        f"{', '.join(sorted(m.get('runtimes', {}))) or 'none'})"))
            continue
        rt = R.make(name, options.get(name))
        why = rt.unavailable()
        if why:
            res.verdict = "unavailable"
            res.issues.append(("TS621", why))
            continue
        now_fp = rt.fingerprint()
        res.fingerprint_changes = differences(rec["fingerprint"], now_fp)
        try:
            s = rt.open(art.data, art)
        except Exception as exc:
            res.verdict = "failed"
            res.issues.append(("TS636", str(exc)))
            continue
        fault = rt.options.get("fault_outputs")
        changed = False
        try:
            for k in range(m["samples"]):
                got = art.decode_outputs(s.run(art.encode_inputs(b.inputs(k))))
                if fault:
                    got = [np.asarray(x) for x in fault(got)]
                old = b.runtime_outputs(name, k)
                exp = b.expected(k)
                for j, o in enumerate(m["outputs"]):
                    d = _rel(got[j], exp[j])
                    res.worst_rel_err = max(res.worst_rel_err, d.rel_err)
                    if d.shape_mismatch or d.nonfinite_mismatch or d.rel_err > o["threshold"]:
                        res.conformant = False
                    if old is not None:
                        a, c = np.asarray(got[j], np.float64), np.asarray(old[j], np.float64)
                        if a.shape != c.shape or not np.array_equal(np.asarray(got[j]), np.asarray(old[j]), equal_nan=True):
                            changed = True
                            if a.shape == c.shape:
                                res.max_change = max(res.max_change, float(np.nanmax(np.abs(a - c))))
                                res.rel_change = max(res.rel_change, _rel(a, c).rel_err)
                                if a.ndim == 2 and a.shape[1] > 1:
                                    res.decisions_moved += int((a.argmax(1) != c.argmax(1)).sum())
                            else:
                                res.max_change = float("inf")
        finally:
            s.close()
        what = "; ".join(f"{k} {a} -> {c}" for k, a, c, _ in res.fingerprint_changes[:6]) or "no fingerprint change"
        if changed:
            res.verdict = "changed" if res.conformant else "regressed"
            for u in m.get("units", []):
                files = u["runtime_files"].get(name)
                if not files:
                    continue
                unit = A.Unit(u["module"], [], u["inputs"], u["outputs"], u["ops"])
                try:
                    ins = [b.array(f) for f in u["input_files"]]
                    dts = [_unit_dtype(art, t) for t in u["inputs"]]
                    feeds = [art.encode(t, x, dt) for t, x, dt in zip(u["inputs"], ins, dts)]
                    unit.nodes = _unit_nodes(art, u)
                    sc = rt.open(art.cut(unit), art)
                    try:
                        got = [art.decode(t, g) for t, g in zip(u["outputs"], sc.run(feeds))]
                    finally:
                        sc.close()
                    if fault:
                        got = [np.asarray(x) for x in fault(got)]
                except Exception as exc:
                    res.modules.append({"module": u["module"], "error": f"{type(exc).__name__}: {exc}"[:200]})
                    continue
                old = [b.array(f) for f in files]
                exp = [b.array(f) for f in u["expected_files"]]
                same = all(np.array_equal(x, y, equal_nan=True) for x, y in zip(got, old))
                entry = {"module": u["module"], "identical": same,
                         "change": max((_rel(x, y).rel_err for x, y in zip(got, old)), default=0.0),
                         "own_error_before": max((_rel(y, e).rel_err for y, e in zip(old, exp)), default=0.0),
                         "own_error_now": max((_rel(x, e).rel_err for x, e in zip(got, exp)), default=0.0)}
                entry["_order"] = (min(unit.nodes) if unit.nodes else 0, -u["module"].count("."))
                res.modules.append(entry)
            changed_units = [e for e in res.modules if e.get("identical") is False]
            # a parent changes when a child does: keep the changed modules with no changed module
            # inside them, then the earliest in the graph
            leaves = [e for e in changed_units
                      if not any(o["module"].startswith(e["module"] + ".") for o in changed_units)]
            if leaves:
                res.first_changed_module = min(leaves, key=lambda e: e["_order"])
            for e in res.modules:
                e.pop("_order", None)
            where = ""
            if res.first_changed_module:
                f = res.first_changed_module
                where = (f"; first module whose own output changed: {f['module']} (own error "
                         f"{f['own_error_before']:.2g} -> {f['own_error_now']:.2g})")
            code = "TS634" if res.conformant else "TS633"
            res.issues.append((code, f"outputs changed since the bundle was written (max change {res.max_change:.2g}, "
                                     f"relative {res.rel_change:.2g}, {res.decisions_moved} decisions moved)"
                                     f"{where}; {'still within' if res.conformant else 'now beyond'} the float64 "
                                     f"tolerance; changed: {what}"))
        else:
            res.issues.append(("TS639", f"outputs bit for bit identical to the recording; {what}"))
            if not res.conformant:
                res.issues.append(("TS622", f"but beyond the float64 tolerance (relative {res.worst_rel_err:.2g}), "
                                            "as when it was recorded"))
    return results, top


def _unit_dtype(art, tid):
    if art.format == A.TFLITE:
        return art.graph.dtype(art.index(tid))
    if art.format == A.ONNX:
        for vi in list(art.model.graph.value_info) + list(art.model.graph.input):
            if vi.name == tid:
                return A._ONNX_NP.get(vi.type.tensor_type.elem_type, np.float32)
    return np.float32


def _unit_nodes(art, u) -> List[int]:
    for x in art.units():
        if x.module == u["module"]:
            return x.nodes
    raise GoldenError(f"module {u['module']} is no longer mapped in the artifact")


def replay_text(results: List[ReplayResult], top) -> str:
    lines = []
    for c, msg in top:
        lines.append(f"[{c}] {msg}")
    for r in results:
        lines.append(f"{r.runtime:<15} {r.verdict.upper():<10} "
                     f"{'conformant' if r.conformant else 'NOT conformant'} (worst rel {r.worst_rel_err:.2g})")
        for k, a, c, kd in r.fingerprint_changes[:10]:
            lines.append(f"    {kd:<9} {k}: {a} -> {c}")
        for c, msg in r.issues:
            lines.append(f"    [{c}] {msg}")
    return "\n".join(lines)
