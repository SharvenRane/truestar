"""truestar export --to onnx|litert|coreml|executorch: the deployable file, written from Truestar's
verified IR and run in its own runtime before it is kept.

1. The model is imported and verified against PyTorch in float64 exactly as for .tsx (exporter.py).
2. Input independent nodes are folded (lower.prepare) and the target writer emits the file.
3. The file runs in its runtime (conform.runtimes: ONNX Runtime CPU, the LiteRT interpreter, the
   ExecuTorch runtime) on the example, more samples around it and, for a symbolic batch, a second
   batch size. Each output is judged against Truestar's float64 reference with the derived
   tolerance: the float32 threshold `truestar check` uses (relative L2, calibrated on a float32
   PyTorch run) and the per element bound of docs/TOLERANCE.md.
4. The file's own math is evaluated in float64 too (ONNX reference evaluator, Truestar's TFLite
   rules, the decoded Core ML program), so a wrong lowering is told apart from a runtime error.
5. What the runtime actually ran is reported: the kernels and their element types (ONNX Runtime
   profile), delegate partitions and the nodes left on builtin or portable kernels.
6. A file whose runtime result fails is not written (TS1414, TS1418) unless force=True (TS1415).
   Core ML runs only on Apple's framework, so its runtime check is pending (TS1413).

Codes TS1410 to TS1419, see docs/EXPORT.md ("Deploy targets").
"""

from __future__ import annotations

import collections
import datetime as _dt
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..exporter import DEFAULT_TOLERANCE, ExportError, _second_batch, export
from .lower import Lowered, LoweringError, prepare

TARGETS = {
    "onnx": {"ext": ".onnx", "runtime": "ort:cpu", "runtimes": ("ort:cpu", "ort:cuda", "ort:webgpu")},
    "litert": {"ext": ".tflite", "runtime": "tflite:interp",
               "runtimes": ("tflite:interp", "tflite:ref", "tflite:cpu", "tflite:gpu", "tflite:gpu-f32")},
    "coreml": {"ext": ".mlpackage", "runtime": None, "runtimes": ()},
    "executorch": {"ext": ".pte", "runtime": "pte:cpu", "runtimes": ("pte:cpu",)},
}
ALIASES = {"tflite": "litert", "pte": "executorch", "mlpackage": "coreml"}
EVIDENCE_SUFFIX = ".truestar.json"


class DeployError(ExportError):
    """Export to a deploy target refused; nothing was written. ``code`` is the TS code."""

    def __init__(self, code: str, message: str, report: Optional["DeployResult"] = None):
        super().__init__(f"{code}: {message}")
        self.code, self.report = code, report


@dataclass
class DeployResult:
    target: str
    path: Optional[str]
    runtime: Optional[str]
    status: str  # verified, pending, failed (written only with force)
    written: bool
    checks: List[Dict[str, Any]] = field(default_factory=list)
    conversion: Optional[Dict[str, Any]] = None
    used: Dict[str, Any] = field(default_factory=dict)
    issues: List[Tuple[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def codes(self) -> List[str]:
        return [c for c, _ in self.issues]

    def as_dict(self) -> Dict[str, Any]:
        return {"target": self.target, "path": self.path, "runtime": self.runtime, "status": self.status,
                "written": self.written, "checks": self.checks, "conversion": self.conversion, "used": self.used,
                "issues": [{"code": c, "message": m} for c, m in self.issues], "notes": self.notes,
                "seconds": round(self.seconds, 3)}

    def to_text(self) -> str:
        head = {"verified": "verified in its runtime", "pending": "runtime verification pending",
                "failed": "FAILED its runtime check"}[self.status]
        lines = [f"{self.target}: {self.path or '(not written)'}: {head}"
                 + (f" ({self.runtime})" if self.runtime else "")]
        for c in self.checks:
            lines.append(f"  check {c['name']} {c['shapes']}: " + ", ".join(
                f"out{j} rel {o['rel_err']:.2e} (bar {o['threshold']:.0e}), {o['beyond_bound']} of {o['elements']} "
                f"beyond the derived bound" for j, o in enumerate(c["outputs"])) + (" PASS" if c["passed"] else " FAIL"))
        if self.conversion:
            cv = self.conversion
            if cv.get("error") is None:
                lines.append(f"  file in float64 vs Truestar float64: {cv.get('reason', 'not evaluated')}")
            else:
                lines.append(f"  file in float64 vs Truestar float64: rel {cv['error']:.2e} (bar {cv['bar']:.0e}) "
                             + ("PASS" if cv["passed"] else "FAIL"))
        u = self.used
        if u.get("kernels"):
            lines.append("  runtime ran: " + ", ".join(f"{k} x{v}" for k, v in sorted(u["kernels"].items())))
        if u.get("element_types"):
            lines.append("  element types seen by kernels: " + ", ".join(u["element_types"]))
        if u.get("residency"):
            r = u["residency"]
            lines.append(f"  residency ({r['method']}): {r['nodes']} nodes, "
                         + ("all on the requested device" if r["resident"] else
                            f"{len(r['off_target'])} off it: " + ", ".join(
                                f"{p['op']} ({p['where']})" for p in r["off_target"][:8])
                            + (" ..." if len(r["off_target"]) > 8 else "")))
        for c, m in self.issues:
            lines.append(f"  {c}: {m}")
        for nline in self.notes:
            lines.append(f"  note: {nline}")
        return "\n".join(lines)


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _rel(a, b):
    from ...compare import diff_tensors

    return diff_tensors(torch.from_numpy(np.asarray(a)), torch.from_numpy(np.asarray(b)))


def _f64(x: torch.Tensor) -> torch.Tensor:
    return x.double() if x.is_floating_point() else x


def _feeds(sample: Sequence[torch.Tensor], target: str, low: Lowered) -> List[np.ndarray]:
    out = []
    for v, x in zip(low.inputs, sample):
        a = _np(x)
        if target == "coreml" and a.dtype == np.int64:
            a = a.astype(np.int32)
        out.append(np.ascontiguousarray(a))
    return out


def _write(target: str, low: Lowered, example):
    if target == "onnx":
        from .onnx_writer import write
        return write(low)
    if target == "litert":
        from .tflite_writer import write
        return write(low)
    if target == "coreml":
        from .coreml_writer import write
        return write(low)
    from .pte_writer import write
    return write(low, example)


class _PtePlan:
    """What conform's ExecuTorch residency reads, from the plan the writer kept."""

    def __init__(self, instructions):
        self._ins = list(instructions)

    def instructions(self):
        return self._ins

    def node_count(self):
        return len(self._ins)


def _artifact(target: str, data, low: Optional[Lowered] = None):
    from ...conform import artifact as A

    if target == "executorch" and low is not None and "instructions" in low.meta:
        return _PtePlan(low.meta["instructions"])

    cls = {"onnx": A.OnnxArtifact, "litert": A.TfliteArtifact, "executorch": A.PteArtifact}[target]
    return cls(f"<{target} export>", data)


def _ort_used(data: bytes, rt_name: str, feeds: List[np.ndarray], plain: List[np.ndarray]) -> Dict[str, Any]:
    """One profiled ONNX Runtime session: the kernels that ran (after ORT's graph fusions), their
    providers and element types, and whether profiling changed the outputs (observer guard)."""
    import onnxruntime as ort

    d = tempfile.mkdtemp(prefix="truestar_export_")
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.enable_profiling = True
        so.profile_file_prefix = os.path.join(d, "p")
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if rt_name == "ort:cuda" else ["CPUExecutionProvider"]
        sess = ort.InferenceSession(data, so, providers=prov)
        names = [i.name for i in sess.get_inputs()]
        outs = sess.run(None, dict(zip(names, feeds)))
        path = sess.end_profiling()
        with open(path, encoding="utf-8") as f:
            events = json.load(f)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    kernels: collections.Counter = collections.Counter()
    types = set()
    for e in events:
        if e.get("cat") != "Node" or not str(e.get("name", "")).endswith("_kernel_time"):
            continue
        a = e.get("args", {})
        kernels[f"{a.get('op_name', '?')}@{a.get('provider', '?').replace('ExecutionProvider', '')}"] += 1
        for io in a.get("input_type_shape", []) + a.get("output_type_shape", []):
            types.update(io.keys())
    same = all(np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True) for x, y in zip(outs, plain))
    return {"kernels": dict(kernels), "element_types": sorted(types), "observer_changed_outputs": not same}


def _tflite_used(session, resid) -> Dict[str, Any]:
    it = getattr(session, "interpreter", None)
    if it is None:
        return {}
    off = collections.Counter(p.op for p in resid.off_target)
    kernels: collections.Counter = collections.Counter()
    for d in it._get_ops_details():
        op = d.get("op_name", "?")
        if op == "DELEGATE":
            kernels["XNNPACK partition"] += 1
        elif off.get(op, 0) > 0:
            off[op] -= 1
            kernels[f"{op}@builtin"] += 1
        elif resid.method.startswith("XNNPACK"):
            kernels[f"{op}@XNNPACK"] += 1
        else:
            kernels[op] += 1
    types = sorted({str(np.dtype(d["dtype"])) for d in it.get_tensor_details() if "dtype" in d})
    return {"kernels": dict(kernels), "element_types": types}


def _pte_used(art) -> Dict[str, Any]:
    ins = art.instructions()
    k = collections.Counter(i["op"] if i["kind"] == "kernel" else f"delegate:{i['backend']}" for i in ins)
    return {"kernels": dict(k)}


def _coreml_run(data: Dict[str, bytes], feeds_all) -> Any:
    """Run the written .mlpackage in Core ML (macOS with coremltools): outputs per feed with the
    CPU_ONLY compute unit (float32, the verified run), plus each unit's planned devices per op and
    its error against the CPU run. A reason string when Core ML cannot run here."""
    import sys
    import tempfile
    if sys.platform != "darwin":
        from ...conform.artifact import COREML_REASON
        return COREML_REASON.split(";")[0]
    try:
        import coremltools as ct
    except Exception as exc:  # pragma: no cover - macOS without coremltools
        return f"coremltools is not installed ({type(exc).__name__})"
    d = tempfile.mkdtemp(prefix="truestar_coreml_")
    pkg = os.path.join(d, "model.mlpackage")
    for rel, body in data.items():
        f = os.path.join(pkg, rel)
        os.makedirs(os.path.dirname(f), exist_ok=True)
        with open(f, "wb") as fh:
            fh.write(body)
    try:
        compiled = str(ct.utils.compile_model(pkg, os.path.join(d, "model.mlmodelc")))
        spec = ct.models.MLModel(pkg, skip_model_load=True).get_spec()
        ins = [i.name for i in spec.description.input]
        onames = [o.name for o in spec.description.output]
        used: Dict[str, Any] = {}
        base = None
        for unit in ("CPU_ONLY", "CPU_AND_GPU", "ALL"):
            m = ct.models.CompiledMLModel(compiled, compute_units=getattr(ct.ComputeUnit, unit))
            outs = []
            for f in feeds_all:
                r = m.predict({n: np.asarray(x) for n, x in zip(ins, f)})
                outs.append([np.asarray(r[n]) for n in onames])
            rec: Dict[str, Any] = {}
            if base is None:
                base = outs
            else:
                rec["max_abs_vs_cpu"] = max(float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))
                                           for oa, ob in zip(outs, base) for a, b in zip(oa, ob))
            try:
                rec["planned"] = _coreml_planned(compiled, getattr(ct.ComputeUnit, unit))
            except Exception as exc:  # the plan is informational
                rec["planned_error"] = f"{type(exc).__name__}: {exc}"[:200]
            used[unit] = rec
        return base, used
    except Exception as exc:
        return f"Core ML could not load or run the file ({type(exc).__name__}: {str(exc)[:300]})"


def _coreml_planned(compiled: str, unit) -> Dict[str, int]:
    from coremltools.models import compute_device as cd
    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(compiled, unit)
    counts: Dict[str, int] = {}

    def walk(block):
        for op in block.operations:
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if u is not None and op.operator_name != "const":
                dv = u.preferred_compute_device
                k = ("NE" if isinstance(dv, cd.MLNeuralEngineComputeDevice) else
                     "GPU" if isinstance(dv, cd.MLGPUComputeDevice) else "CPU")
                counts[k] = counts.get(k, 0) + 1
            for b in getattr(op, "blocks", None) or []:
                walk(b)
    walk(plan.model_structure.program.functions["main"].block)
    return counts


def export_to(model: torch.nn.Module, args: Any, output: Optional[str] = None, *, to: str,
              runtime: Optional[str] = None, force: bool = False, static_batch: Optional[bool] = None,
              samples: int = 4, second_batch: int = 7,
              tolerance: float = DEFAULT_TOLERANCE, decompose: bool = False,
              name: Optional[str] = None) -> DeployResult:
    """Write ``model`` as a deployable file for ``to`` and verify it in its runtime.

    args: the example inputs (a tensor or a tuple of tensors). runtime: a conform runtime name of
    the target's format (default: ONNX Runtime CPU, the LiteRT interpreter, the ExecuTorch runtime).
    static_batch: freeze the batch (always for litert, coreml and executorch; ONNX keeps a
    symbolic batch by default).
    force: write the file even when its runtime result fails the check (TS1415).

    Raises DeployError (with the TS code) and writes nothing when the model, the lowering or the
    runtime check fails.
    """
    t0 = time.perf_counter()
    to = ALIASES.get(to, to)
    if to not in TARGETS:
        raise DeployError("TS1419", f"unknown target {to!r}; targets: {', '.join(TARGETS)}")
    spec = TARGETS[to]
    rt_name = runtime or spec["runtime"]
    if runtime is not None and runtime not in spec["runtimes"]:
        raise DeployError("TS1419", f"runtime {runtime} does not run {to} files; choose from "
                                    f"{', '.join(spec['runtimes']) or 'none (Core ML runs only on Apple devices)'}")
    if output is not None and not output.lower().endswith(spec["ext"]):
        raise DeployError("TS1419", f"{to} files end in {spec['ext']}, got {output}")
    args = tuple(args) if isinstance(args, (tuple, list)) else (args,)
    if not args or not all(isinstance(a, torch.Tensor) for a in args):
        raise DeployError("TS1419", "deploy targets take positional tensor inputs (nested or keyword "
                                    "inputs export to .tsx only)")
    name = name or type(model).__name__
    model = model.eval()
    notes: List[str] = []
    dynamic = to == "onnx" and not static_batch and all(a.dim() and a.shape[0] == args[0].shape[0] for a in args) \
        and args[0].shape[0] >= 2
    try:
        res = export(model, args, None, dynamic_batch=dynamic, tolerance=tolerance, decompose=decompose, name=name)
    except ExportError as e:
        if not dynamic or static_batch is False:
            raise DeployError("TS1411", f"the model's Truestar program could not be verified against PyTorch, "
                                        f"so no {to} file was written:\n{e}") from None
        res = None
        first = str(e).splitlines()[0]
    if res is None:
        try:
            res = export(model, args, None, dynamic_batch=False, tolerance=tolerance, decompose=decompose, name=name)
        except ExportError as e:
            raise DeployError("TS1411", f"the model's Truestar program could not be verified against PyTorch, "
                                        f"so no {to} file was written:\n{e}") from None
        dynamic = False
        notes.append(f"the batch is frozen at {args[0].shape[0]}: a symbolic batch was refused ({first})")
    try:
        low = prepare(res.program)
        data = _write(to, low, list(args))
    except LoweringError as e:
        raise DeployError("TS1412", str(e)) from None
    if low.folded:
        notes.append(f"{low.folded} input independent node(s) folded to constants in float64")

    from ...conform.core import make_samples

    smp = [tuple(s) for s in make_samples(args, max(1, samples))]
    gen = torch.Generator().manual_seed(1)
    for k in range(1, len(smp)):  # integer inputs (token ids) get other values in the range the example uses
        smp[k] = tuple(x if x.is_floating_point() or x.dtype == torch.bool or x.numel() == 0 else
                       torch.randint(int(x.min()), int(x.max()) + 1, x.shape, generator=gen, dtype=x.dtype)
                       for x in smp[k])
    labels = ["example"] + [f"sample{k}" for k in range(1, len(smp))]
    if dynamic:
        smp.append(tuple(_second_batch(list(args), second_batch)))
        labels.append(f"batch{second_batch}")

    from ... import ir
    from ...compare import DTYPE_FLOOR, threshold_for
    from ...tolerance import derive

    from .pte_writer import IRModule

    ref64, noise, bounds = [], None, []
    m32 = model.float()
    irm = IRModule(low).eval()
    unsound = 0
    for s in smp:
        r = [_np(t) for t in ir.run(res.program, *[_f64(x) for x in s])]
        ref64.append(r)
        with torch.no_grad():
            o32 = m32(*s)
        o32 = [_np(t) for t in (o32 if isinstance(o32, (tuple, list)) else (o32,))]
        e = [_rel(a, b).rel_err for a, b in zip(o32, r)]
        noise = e if noise is None else [max(x, y) for x, y in zip(noise, e)]
        try:
            # derived on the IR as a module of plain ops (each with an error rule), then checked
            # against that module's own float32 run: a bound its own run breaks is not used
            tol = derive(irm, s, dtype=torch.float32, device="cpu", tf32=False, cache=False)
            bd = [_np(tol.bound(j)) for j in range(len(r))]
            with torch.no_grad():
                own = irm(*s)
            for j, (o, b) in enumerate(zip(own, bd)):
                if np.asarray(r[j]).dtype.kind == "f":
                    bad = int((np.abs(_np(o).astype(np.float64) - r[j]) > b).sum())
                    if bad:
                        unsound += bad
                        bd[j] = np.full(np.shape(b), np.inf)
            bounds.append(bd)
        except Exception as exc:  # derivation needs every op to have a rule
            bounds.append([np.full(np.shape(x), np.inf) for x in r])
            if len(bounds) == 1:
                notes.append(f"per element bounds not derived ({type(exc).__name__}: {exc}); relative L2 only")
    thr = [threshold_for(torch.float32, False, n) for n in noise]
    if unsound:
        notes.append(f"the derived per element bound failed on {unsound} element(s) of Truestar's own float32 "
                     "run of the program, so those outputs are judged by relative L2 only")

    result = DeployResult(to, None, rt_name, "pending", False, notes=notes)
    feeds_all = [_feeds(s, to, low) for s in smp]

    # the file's own math in float64 (conversion verdict)
    from ...conform.core import CONVERSION_BAR

    bar = max([CONVERSION_BAR] + [2.0 * n for n in noise])
    meaning = None
    reason = ""
    try:
        if to == "coreml":
            from .coreml_writer import MODEL_PATH, decode, evaluate64

            dec = decode(data[MODEL_PATH])
            meaning = [evaluate64(dec, f) for f in feeds_all]
        elif to == "executorch":
            reason = "ExecuTorch programs have no float64 evaluator; judged by the runtime check only"
        else:
            art = _artifact(to, data)
            names = [n for n, _, _ in art.input_specs()] if to == "onnx" else [v.name for v in low.inputs]
            meaning = [art.meaning64(dict(zip(names, f)), art.output_names()) for f in feeds_all]
            if any(m is None for m in meaning):
                meaning, reason = None, "an op of the file has no float64 rule"
    except Exception as exc:
        meaning, reason = None, f"float64 evaluation failed ({type(exc).__name__}: {exc})"
    if meaning is not None:
        err = max(_rel(meaning[k][j], ref64[k][j]).rel_err for k in range(len(smp)) for j in range(len(ref64[0])))
        result.conversion = {"error": err, "bar": bar, "passed": err <= bar}
        if err > bar:
            result.issues.append(("TS1418", f"the written file computes something else than the verified program: "
                                            f"float64 evaluation of the file differs by {err:.2e} (bar {bar:.0e})"))
    else:
        result.conversion = {"error": None, "bar": bar, "passed": None, "reason": reason}

    # the runtime
    runtime_ok = None
    def _judge(outs):
        nonlocal runtime_ok
        runtime_ok = True
        low_prec = False
        for k, s in enumerate(smp):
            c = {"name": labels[k], "shapes": [list(x.shape) for x in s], "outputs": [], "passed": True}
            for j, ref in enumerate(ref64[k]):
                got = outs[k][j] if j < len(outs[k]) else None
                if got is None:
                    c["outputs"].append({"rel_err": float("inf"), "threshold": thr[j], "beyond_bound": -1,
                                         "elements": int(np.size(ref)), "note": "output missing"})
                    c["passed"] = False
                    continue
                d = _rel(got, ref)
                beyond = -1
                if not d.shape_mismatch and np.asarray(ref).dtype.kind == "f":
                    e = np.abs(np.asarray(got, np.float64) - ref)
                    beyond = int((np.nan_to_num(e, nan=np.inf) > bounds[k][j]).sum())
                ok = (not d.shape_mismatch and not d.nonfinite_mismatch and d.rel_err <= thr[j]
                      and beyond <= 0)
                if (not d.shape_mismatch and d.rel_err > thr[j]
                        and d.rel_err <= max(DTYPE_FLOOR[torch.float16], 10 * thr[j])):
                    low_prec = True
                c["outputs"].append({"rel_err": d.rel_err, "max_abs": d.max_abs, "threshold": thr[j],
                                     "beyond_bound": beyond, "elements": int(np.size(ref)), "note": d.note})
                c["passed"] = c["passed"] and ok
            result.checks.append(c)
            runtime_ok = runtime_ok and c["passed"]
        if not runtime_ok:
            result.issues.append(("TS1414", f"the file's outputs in {rt_name} are outside the derived float32 "
                                            "tolerance of Truestar's float64 reference"))
            if low_prec:
                result.issues.append(("TS1417", "the error is what a lower precision (float16) run shows: the "
                                                "runtime does not compute in float32"))
        kinds = result.used.get("kernels", {})
        fused = sorted(k for k in kinds if k.split("@")[0] in ("FusedConv", "FusedGemm", "FusedMatMul", "Gelu",
                                                                 "BiasGelu", "SkipLayerNormalization",
                                                                 "FastGelu", "QuickGelu"))
        if fused:
            result.issues.append(("TS1417", "the runtime replaced written ops with its own fused kernels: "
                                            + ", ".join(fused)))

    if to == "coreml":
        from ...conform.artifact import COREML_REASON

        cm = _coreml_run(data, feeds_all)
        if isinstance(cm, str):
            result.runtime = None
            result.issues.append(("TS1413", "Core ML runtime verification pending: " + cm
                                  + ". The file was checked by decoding it and evaluating its program in float64."))
        else:
            rt_name = "coreml:cpu"
            outs, result.used["coreml"] = cm
            result.runtime = rt_name
            _judge(outs)
    else:
        from ...conform import runtimes as R

        rt = R.make(rt_name)
        why = rt.unavailable()
        if why:
            result.issues.append(("TS1413", f"{rt_name} cannot run here ({why}); the file is written, its runtime "
                                            "verification is pending"))
        else:
            art = _artifact(to, data, low)
            sess = rt.open(data, art)
            try:
                outs = [sess.run(f) for f in feeds_all]
                resid = rt.residency(art, sess)
                if to == "onnx":
                    result.used.update(_ort_used(data, rt_name, feeds_all[0], outs[0]))
                    if result.used.get("observer_changed_outputs"):
                        result.notes.append("the profiled run gave different outputs than the plain run; "
                                            "the kernel list may not describe the plain run")
                elif to == "litert":
                    result.used.update(_tflite_used(sess, resid))
                else:
                    result.used.update(_pte_used(art))
            finally:
                sess.close()
            result.used["residency"] = resid.as_dict()
            if resid.off_target:
                result.issues.append(("TS1416", f"{len(resid.off_target)} node(s) did not run on the requested "
                                                f"{resid.requested} path ({resid.method}): " + ", ".join(
                                                    sorted({p.op for p in resid.off_target}))))
            _judge(outs)

    failed = (runtime_ok is False) or (result.conversion.get("passed") is False)
    result.status = "failed" if failed else ("verified" if runtime_ok else "pending")
    result.seconds = time.perf_counter() - t0
    if failed and not force:
        code = "TS1418" if result.conversion.get("passed") is False else "TS1414"
        raise DeployError(code, f"{to} file refused, nothing was written (pass force=True / --force to write it "
                                f"anyway):\n{result.to_text()}", result)
    if failed:
        result.issues.append(("TS1415", "written with --force although it failed its check; the evidence says so"))
    if output is None:
        result.data = data
        return result

    path = os.path.abspath(output)
    evidence = {"verified": result.status == "verified", "status": result.status, "target": to,
                "program_hash": res.program_hash, "model": name, "truestar_evidence": res.evidence,
                "deploy": result.as_dict(),
                "created_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()}
    _put(path, data)
    _put_bytes(path + EVIDENCE_SUFFIX, json.dumps(evidence, indent=2, default=str, sort_keys=True).encode("utf-8"))
    result.path, result.written = path, True
    if result.status == "verified":
        result.issues.insert(0, ("TS1410", f"{to} file written and verified in {rt_name}"))
    result.seconds = time.perf_counter() - t0
    return result


def _put_bytes(path: str, blob: bytes) -> None:
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".ts-", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    with open(path, "rb") as f:
        if f.read() != blob:
            raise ExportError(f"internal error: {path} does not hold the bytes that were written")


def _put(path: str, data) -> None:
    if isinstance(data, (bytes, bytearray)):
        _put_bytes(path, bytes(data))
        return
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".ts-", dir=parent)
    try:
        for rel, blob in data.items():
            p = os.path.join(tmp, *rel.split("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(blob)
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.replace(tmp, path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    for rel, blob in data.items():
        with open(os.path.join(path, *rel.split("/")), "rb") as f:
            if f.read() != blob:
                raise ExportError(f"internal error: {path}/{rel} does not hold the bytes that were written")


__all__ = ["export_to", "DeployError", "DeployResult", "TARGETS", "LoweringError"]
