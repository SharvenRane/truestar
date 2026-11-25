"""`truestar check --backends webgpu`: the model exported to ONNX, run by ONNX Runtime's WebGPU EP.

What one backend run does (docs/WEBGPU.md has the reasons):

1. Export the whole model to ONNX in the model's dtype (legacy TorchScript exporter, opset 17).
2. Plain run on WebGPU, twice (run to run determinism, TS565), and the same graph on ONNX Runtime's
   CPU EP as a control: when the CPU EP is off too, the export changed the result (TS563), not
   WebGPU.
3. Residency: a profiled run lists the EP of every node. A session that fell back to the CPU EP is
   TS560 and is never reported as WebGPU; compute nodes left on the CPU are TS561, named.
4. Observer guard: the profiled run and a tapped run (every module's output added as a graph output)
   must give the same final output as the plain run. Otherwise their per module numbers are marked
   untrusted (TS564) and every module is replayed in isolation instead.
5. Blame: modules whose tapped output is over threshold (all modules when taps are untrusted) are
   exported alone with the float64 reference inputs and run on WebGPU, children before parents,
   exactly like the torch backends. The first module whose own error crosses its threshold is
   blamed.
6. Op drilldown: inside the blamed module's graph, each ONNX node is run alone on WebGPU with the
   CPU EP's values as inputs; the first node over threshold is named.
7. Repros: the blamed module's ONNX file with inputs and float64 expected outputs, and the single
   node graph, each with a script that needs only numpy and onnxruntime with the WebGPU plugin.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ..capture import ROOT, flatten_tensors, map_tensors
from ..compare import DIVERGENT, DIVERGENT_FACTOR, DTYPE_FLOOR, OK, SUSPECT, Diff, diff_lists, \
    threshold_for, worst

OPSET = 17
MAX_DRILL_NODES = 300


@dataclass(frozen=True)
class EpTarget:
    """Which accelerated runtime a worker drives. The WebGPU worker, the Windows ML worker
    (truestar.winml) and the WebNN page (truestar.webnn) all answer the same `run` call; only the
    provider key, the EP name in the profile, the label and the codes differ."""

    provider: str  # value of payload["provider"] for the accelerated run ("cpu" is the control)
    ep_name: str  # provider name the profile reports for nodes that ran on the target
    label: str  # in messages
    fallback_code: str = "TS560"
    cpu_nodes_code: str = "TS561"
    nondeterministic_code: str = "TS565"
    repro: Optional[Callable] = None  # write_repro(folder, stem, backend, data, feeds, expected, ...)


WEBGPU = EpTarget("webgpu", "WebGpuExecutionProvider", "WebGPU")


class ExportFailed(RuntimeError):
    """torch.onnx.export refused the model or module (TS562)."""


# ----------------------------------------------------------------------------- export


def _leaves(obj: Any) -> List[torch.Tensor]:
    """Tensors in map_tensors order (dicts in insertion order), so they can be put back."""
    out: List[torch.Tensor] = []

    def take(t: torch.Tensor) -> torch.Tensor:
        out.append(t)
        return t

    map_tensors(obj, take)
    return out


class _Flat(nn.Module):
    """forward(*tensors) -> tuple of output tensors, plus every module's output when tapping.

    Non tensor arguments are kept as constants. Tap keys follow capture.Recorder ("name#index"),
    so they line up with the reference calls."""

    def __init__(self, mod: nn.Module, template: Any, tap: bool):
        super().__init__()
        self.mod = mod
        self.template = template
        self.tap = tap
        self.layout: List[Tuple[str, int]] = []

    def forward(self, *flat):
        it = iter(flat)
        args, kwargs = map_tensors(self.template, lambda t: next(it))
        taps: List[torch.Tensor] = []
        layout: List[Tuple[str, int]] = []
        handles = []
        if self.tap:
            counts: Dict[str, int] = {}

            def make(label: str):
                def hook(module, a, o):
                    idx = counts.get(label, 0)
                    counts[label] = idx + 1
                    ts = [t.clone() for t in flatten_tensors(o)]
                    layout.append((f"{label}#{idx}", len(ts)))
                    taps.extend(ts)
                return hook

            for name, m in self.mod.named_modules():
                if name:
                    handles.append(m.register_forward_hook(make(name)))
        try:
            out = self.mod(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()
        self.layout = layout
        return tuple(flatten_tensors(out)) + tuple(taps)


def _first_line(text: str, limit: int = 300) -> str:
    """Exporter errors can embed the whole TorchScript graph; keep the first line."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line[:limit] + ("..." if len(line) > limit else "")


def to_numpy(t: torch.Tensor, dtype: torch.dtype) -> np.ndarray:
    t = t.detach().cpu()
    if t.is_floating_point():
        t = t.to(dtype)
    return t.contiguous().numpy()


def export(module: nn.Module, args: Sequence[Any], kwargs: Dict[str, Any], dtype: torch.dtype,
           tap: bool = False):
    """Export module(*args, **kwargs) in dtype. Returns (onnx bytes, feeds, number of real outputs,
    tap layout [(call key, tensor count)])."""
    def cast(t: torch.Tensor) -> torch.Tensor:
        t = t.detach().cpu()
        return t.to(dtype) if t.is_floating_point() else t

    template = map_tensors((tuple(args), dict(kwargs)), cast)
    flat = _leaves(template)
    # Inputs as fed to the graph, copied before anything runs: an in place module (ReLU or SiLU
    # with inplace=True) would otherwise overwrite them during the counting run and the trace.
    # Found on EfficientNet B0: SiLU replayed alone looked 67% wrong because its feed was
    # already silu(x).
    feed_arrays = [to_numpy(t, dtype) for t in flat]

    def fresh():
        return [t.clone() for t in flat]
    wrapper = _Flat(module, template, tap)
    # The exporter switches the model to eval and back to the WRAPPER's mode afterwards, which
    # recurses into the user's module: a new wrapper is in training mode, so batch norm would come
    # back in training mode (found on ResNet18: every batch norm then used batch statistics).
    # Snapshot every submodule's mode and restore it exactly.
    modes = [(m, m.training) for m in module.modules()]
    wrapper.train(module.training)
    with torch.no_grad():
        it = iter(fresh())
        a0, k0 = map_tensors(template, lambda t: next(it))
        n_out = len(flatten_tensors(module(*a0, **k0)))
    in_names = [f"in{i}" for i in range(len(flat))]
    errors = []
    data = None
    try:
        # The TorchScript exporter first: about 10x faster here and it keeps module scopes in node
        # names. When it refuses an op (GPT2: aten::diff), the torch.export based exporter.
        for exporter in ("torchscript", "dynamo"):
            try:
                with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    warnings.simplefilter("ignore")
                    if exporter == "torchscript":
                        f = io.BytesIO()
                        torch.onnx.export(wrapper, tuple(fresh()), f, dynamo=False,
                                          opset_version=OPSET, input_names=in_names,
                                          do_constant_folding=True)
                        data = f.getvalue()
                    else:
                        prog = torch.onnx.export(wrapper, tuple(fresh()), dynamo=True,
                                                 input_names=in_names, verbose=False)
                        data = prog.model_proto.SerializeToString()
                break
            except Exception as exc:
                errors.append(f"{exporter} exporter: {type(exc).__name__}: "
                              + _first_line(str(exc)))
                for m, mode in modes:
                    m.training = mode
    finally:
        for m, mode in modes:
            m.training = mode
    if data is None:
        raise ExportFailed("; ".join(errors))
    import onnx

    proto = onnx.load_from_string(data)
    graph_inputs = {i.name for i in proto.graph.input}
    feeds = {n: a for n, a in zip(in_names, feed_arrays) if n in graph_inputs}
    if len(proto.graph.output) != n_out + sum(n for _, n in wrapper.layout):
        raise ExportFailed(f"the exported graph has {len(proto.graph.output)} outputs, expected "
                           f"{n_out} plus {sum(n for _, n in wrapper.layout)} taps")
    return data, feeds, n_out, list(wrapper.layout)


# ----------------------------------------------------------------------------- residency


_INT_TYPES = {6, 7, 9, 2, 3, 4, 5, 12, 13}  # int32, int64, bool, uint8, int8, uint16, int16, ...


def residency(nodes: List[Tuple[str, str, str]], model: bytes,
              ep_name: str = "WebGpuExecutionProvider") -> Dict[str, Any]:
    """Classify where the profiled nodes ran.

    `shape_on_cpu`: nodes on the CPU EP whose outputs are all integer or bool (shape arithmetic,
    which ONNX Runtime keeps on the host on purpose). `compute_on_cpu`: every other CPU node,
    including fused nodes the original graph does not name. Memcpy nodes are transfers. `device`
    counts nodes on `ep_name` (also under the key `webgpu` for the WebGPU EP, as before)."""
    import onnx

    proto = onnx.load_from_string(model)
    try:
        proto = onnx.shape_inference.infer_shapes(proto)
    except Exception:
        pass
    etype: Dict[str, int] = {}
    for vi in list(proto.graph.value_info) + list(proto.graph.output) + list(proto.graph.input):
        etype[vi.name] = vi.type.tensor_type.elem_type
    for init in proto.graph.initializer:
        etype[init.name] = init.data_type
    by_name = {n.name: n for n in proto.graph.node}
    out = {"nodes": len(nodes), "device": 0, "compute_on_cpu": [], "shape_on_cpu": [],
           "transfers": 0, "other": []}
    for name, op, prov in nodes:
        if prov == ep_name:
            out["device"] += 1
        elif op.startswith("Memcpy"):
            out["transfers"] += 1
        elif prov == "CPUExecutionProvider":
            n = by_name.get(name)
            if n is not None and n.output and all(etype.get(o) in _INT_TYPES for o in n.output):
                out["shape_on_cpu"].append((name, op))
            else:
                out["compute_on_cpu"].append((name, op))
        else:
            out["other"].append((name, op, prov))
    if ep_name == "WebGpuExecutionProvider":
        out["webgpu"] = out["device"]
    return out


# ----------------------------------------------------------------------------- helpers


def _tensors(arrays: Sequence[np.ndarray]) -> List[torch.Tensor]:
    return [torch.from_numpy(np.ascontiguousarray(a)) for a in arrays]


def _same(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.shape != y.shape:
            return False
        if x.dtype.kind in "fc":
            if not np.array_equal(x, y, equal_nan=True):
                return False
        elif not np.array_equal(x, y):
            return False
    return True


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


# ----------------------------------------------------------------------------- op drilldown


def _all_values(worker, model: bytes, feeds: Dict[str, np.ndarray]):
    """Every node output of the graph, computed by the CPU EP. Returns (proto, values)."""
    import onnx
    from onnx import numpy_helper

    proto = onnx.load_from_string(model)
    inferred = onnx.shape_inference.infer_shapes(proto)
    known = {vi.name: vi for vi in inferred.graph.value_info}
    have = {o.name for o in proto.graph.output}
    extra = []
    for n in proto.graph.node:
        for o in n.output:
            if o and o not in have and o in known and known[o].type.tensor_type.elem_type:
                extra.append(onnx.helper.make_tensor_value_info(
                    o, known[o].type.tensor_type.elem_type, None))
                have.add(o)
    tapped = onnx.ModelProto()
    tapped.CopyFrom(proto)
    tapped.graph.output.extend(extra)
    res = worker.call("run", {"model": tapped.SerializeToString(), "feeds": feeds,
                              "provider": "cpu"})
    values = dict(zip(res["output_names"], res["outputs"]))
    values.update(feeds)
    for init in proto.graph.initializer:
        values[init.name] = numpy_helper.to_array(init)
    return proto, values


def _node_model(proto, node, values, as_float64: bool = False
                ) -> Tuple[bytes, Dict[str, np.ndarray]]:
    """The node alone as a graph fed with `values`. as_float64: every float input and output
    becomes double, for a per node float64 reference on the CPU EP."""
    from onnx import helper

    def cast(v: np.ndarray) -> np.ndarray:
        return v.astype(np.float64) if as_float64 and v.dtype.kind == "f" else v

    ins, feeds = [], {}
    for name in node.input:
        if not name or name in feeds:
            continue
        v = cast(values[name])
        ins.append(helper.make_tensor_value_info(
            name, helper.np_dtype_to_tensor_dtype(v.dtype), list(v.shape)))
        feeds[name] = v
    outs = []
    for name in node.output:
        if not name:
            continue
        v = values.get(name)
        if v is not None:
            v = cast(v)
        elem = helper.np_dtype_to_tensor_dtype(v.dtype) if v is not None else 0
        outs.append(helper.make_tensor_value_info(name, elem, None))
    g = helper.make_graph([node], "node", ins, outs)
    m = helper.make_model(g, opset_imports=list(proto.opset_import))
    m.ir_version = proto.ir_version
    return m.SerializeToString(), feeds


def drill(worker, model: bytes, feeds: Dict[str, np.ndarray], threshold: float, module_key: str,
          target: EpTarget = WEBGPU):
    """First ONNX node of the module's graph whose own output on the target is off, given the CPU EP's
    values for its inputs. Returns (OpResult or None, notes, (node bytes, node feeds, expected))."""
    from ..opdrill import OpResult

    notes: List[str] = []
    proto, values = _all_values(worker, model, feeds)
    nodes = [n for n in proto.graph.node if n.op_type != "Constant"]
    total = len(nodes)
    if total > MAX_DRILL_NODES:
        notes.append(f"op drilldown on {module_key}: {total} nodes, only the first "
                     f"{MAX_DRILL_NODES} were run alone")
    on_cpu = []
    skipped = 0
    for i, node in enumerate(nodes[:MAX_DRILL_NODES]):
        if any(a.type in (5, 10) for a in node.attribute):  # GRAPH, GRAPHS: control flow
            skipped += 1
            continue
        if any(name and name not in values for name in node.input) or not all(
                o in values for o in node.output if o):
            skipped += 1
            continue
        data, nfeeds = _node_model(proto, node, values)
        try:
            r = worker.call("run", {"model": data, "feeds": nfeeds, "provider": target.provider,
                                    "profile": True})
        except Exception as exc:
            notes.append(f"node {node.name} ({node.op_type}) failed alone on {target.label}: {exc}")
            skipped += 1
            continue
        if r["fallback"] or any(p != target.ep_name for _, op, p in (r["nodes"] or [])
                                if not op.startswith("Memcpy")):
            on_cpu.append(f"{node.op_type} {node.name}")
            continue
        # Reference for this node: the same node in float64 on the CPU EP, fed the CPU EP's
        # values; when the CPU EP has no float64 kernel for it, the CPU EP's own output.
        expected = [values[o] for o in node.output if o]
        ref_kind = "the CPU EP in the model dtype"
        if any(v.dtype.kind == "f" for v in nfeeds.values()) and node.op_type != "Cast":
            try:
                d64, f64 = _node_model(proto, node, values, as_float64=True)
                r64 = worker.call("run", {"model": d64, "feeds": f64, "provider": "cpu"})
                if all(o.shape == e.shape for o, e in zip(r64["outputs"], expected)):
                    expected = r64["outputs"]
                    ref_kind = "the node in float64 on the CPU EP"
            except Exception:
                pass
        d = diff_lists(_tensors(r["outputs"]), _tensors(expected))
        status = d.status(threshold)
        if status != OK:
            op = OpResult(
                op=f"onnx::{node.op_type} ({node.name})", module=module_key, index=i + 1,
                total=total, input_shapes=[list(v.shape) for v in nfeeds.values()],
                input_dtypes=[str(v.dtype) for v in nfeeds.values()], isolated=d,
                accumulated=None, threshold=threshold, status=status, matched=True)
            notes.append(f"op drilldown reference for {node.op_type} {node.name}: {ref_kind}")
            if on_cpu:
                notes.append("op drilldown: nodes that ran on the CPU EP even alone (not "
                             "checked): " + ", ".join(on_cpu[:8]))
            return op, notes, (data, nfeeds, expected)
    if on_cpu:
        notes.append("op drilldown: nodes that ran on the CPU EP even alone (not checked): "
                     + ", ".join(on_cpu[:8]))
    notes.append(f"op drilldown on {module_key}: none of {total} ONNX nodes is off alone "
                 f"({skipped} not run alone); the error comes from how they combine on {target.label} "
                 "(fusion inside the EP) or from accumulated rounding")
    return None, notes, None


# ----------------------------------------------------------------------------- repro

_SCRIPT = '''"""Repro generated by truestar check for the WebGPU backend.

{what}
Backend:  {backend} ({description})
Expected: relative error below {threshold:.1e} versus {expected_from}.

Run:  python {script}          (ONNX Runtime WebGPU EP, Dawn {dawn}, adapter {adapter})
      python {script} cpu      (control run on ONNX Runtime's CPU EP)
Needs numpy, onnxruntime and onnxruntime-ep-webgpu. Exit code 1 means the divergence reproduced.
"""
import os
import sys

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = {threshold!r}
EP = "WebGpuExecutionProvider"


def main():
    so = ort.SessionOptions()
    use_cpu = len(sys.argv) > 1 and sys.argv[1] == "cpu"
    providers = ["CPUExecutionProvider"]
    if not use_cpu:
        import onnxruntime_ep_webgpu as ep

        ort.register_execution_provider_library(EP, ep.get_library_path())
        devs = [d for d in ort.get_ep_devices() if d.ep_name == EP]
        # One device only: the EP lists it twice on some machines, and passing both makes
        # ONNX Runtime fall back to the CPU EP silently.
        so.add_provider_for_devices(devs[:1], {options!r})
        providers = None
    sess = ort.InferenceSession(os.path.join(HERE, "{onnx}"), sess_options=so, providers=providers)
    if not use_cpu and EP not in sess.get_providers():
        print("the WebGPU EP was not created; ONNX Runtime fell back to", sess.get_providers())
        return 2
    feeds = dict(np.load(os.path.join(HERE, "{inputs}")))
    expected = np.load(os.path.join(HERE, "{expected}"))
    outs = sess.run(None, feeds)
    worst = 0.0
    for i, got in enumerate(outs):
        ref = expected["out%d" % i].astype(np.float64)
        got = got.astype(np.float64)
        if got.shape != ref.shape:
            print("output", i, "shape", got.shape, "expected", ref.shape)
            return 1
        fin = np.isfinite(ref)
        if not np.array_equal(fin, np.isfinite(got)):
            print("output", i, ": NaN or Inf where the reference has none (or the reverse)")
            return 1
        num = np.linalg.norm((got - ref)[fin])
        den = np.linalg.norm(ref[fin])
        rel = 0.0 if num == 0 else (num / den if den else float("inf"))
        worst = max(worst, rel)
        print("output", i, "relative error %.3e" % rel)
    print("on", "CPU EP" if use_cpu else "WebGPU", ": worst relative error %.3e, threshold %.1e"
          % (worst, THRESHOLD))
    return 1 if worst > THRESHOLD else 0


if __name__ == "__main__":
    sys.exit(main())
'''


def write_repro(folder: str, stem: str, backend, data: bytes, feeds: Dict[str, np.ndarray],
                expected: Sequence[np.ndarray], threshold: float, what: str,
                expected_from: str) -> str:
    os.makedirs(folder, exist_ok=True)
    onnx_name, in_name, exp_name, script = (f"{stem}.onnx", f"{stem}_inputs.npz",
                                           f"{stem}_expected.npz", f"{stem}.py")
    with open(os.path.join(folder, onnx_name), "wb") as f:
        f.write(data)
    np.savez(os.path.join(folder, in_name), **feeds)
    np.savez(os.path.join(folder, exp_name),
             **{f"out{i}": np.asarray(e, dtype=np.float64) for i, e in enumerate(expected)})
    from . import parse_spec

    dawn, adapter = parse_spec(backend.name)
    options = {"dawnBackendType": dawn, "adapterIndex": str(adapter)}
    options.update(backend.options or {})
    path = os.path.join(folder, script)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_SCRIPT.format(what=what, backend=backend.name, description=backend.description,
                               threshold=threshold, expected_from=expected_from, script=script,
                               dawn=dawn, adapter=adapter, options=options, onnx=onnx_name,
                               inputs=in_name, expected=exp_name))
    return path


# ----------------------------------------------------------------------------- main


def check_backend(res, base: nn.Module, args, kwargs, dtype, ref_calls, ref_flat, noise,
                  op_counting, repro_dir, model_name, drill_ops: bool = True,
                  target: EpTarget = WEBGPU, worker=None) -> None:
    """Fill a check.BackendResult for an ONNX backend (same fields as the torch backends): the
    WebGPU EP by default, or another target with its own worker (truestar.winml, truestar.webnn)."""
    from ..check import LayerResult, _decisions, _module_by_name, _needs_replay
    from . import get_worker

    b = res.backend
    if worker is None:
        worker = get_worker(b.name, b.options or None)
    T = target
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    ref_by_key = {c.key: c for c in ref_calls}

    # 1. Export.
    try:
        model_bytes, feeds, n_out, _ = export(base, args, kwargs, dtype)
    except ExportFailed as exc:
        res.status = SUSPECT
        res.add("TS562", f"the model could not be exported to ONNX, so nothing ran on {b.name}: "
                         f"{exc}")
        return
    timings["export"] = time.perf_counter() - t0

    # 2. Plain run (twice) and the CPU EP control.
    t = time.perf_counter()
    plain = worker.call("run", {"model": model_bytes, "feeds": feeds, "provider": T.provider,
                                "repeat": 1})
    timings[f"{T.provider} run"] = plain["seconds"]
    if plain["fallback"]:
        res.status = SUSPECT
        res.add(T.fallback_code, f"the {T.label} EP could not be created for this graph and ONNX Runtime "
                         f"fell back to its CPU EP; nothing is reported as a {T.label} result. EP error: "
                + (plain["ep_error"] or worker.log_tail(600) or "none printed"))
        return
    if plain["repeat_equal"] is False:
        res.add(T.nondeterministic_code, f"two identical runs on {T.label} gave different outputs")
    out_plain = plain["outputs"][:n_out]
    out_thr = threshold_for(dtype, False, None)
    # The CPU EP control runs only when WebGPU is off: it can only change the attribution, and
    # ONNX Runtime's float16 CPU kernels are slow (20 s of a 22 s check on a float16 encoder).
    if diff_lists(_tensors(out_plain), ref_flat).status(out_thr) != OK:
        control = worker.call("run", {"model": model_bytes, "feeds": feeds, "provider": "cpu"})
        ctrl_diff = diff_lists(_tensors(control["outputs"][:n_out]), ref_flat)
        if ctrl_diff.status(out_thr) != OK:
            res.add("TS563", f"the exported graph on ONNX Runtime's CPU EP is already off by "
                             f"{ctrl_diff.rel_err:.2e} against float64: the export or ONNX "
                             f"Runtime's own kernels change the result, not only {T.label}")
    timings["runs"] = time.perf_counter() - t

    # 3. Residency, from a profiled run; the profiled run must match the plain one.
    t = time.perf_counter()
    prof = worker.call("run", {"model": model_bytes, "feeds": feeds, "provider": T.provider,
                               "profile": True})
    where = residency(prof["nodes"] or [], model_bytes, T.ep_name)
    res.extra["residency"] = {k: v for k, v in where.items()}
    timings["residency"] = time.perf_counter() - t
    profile_trusted = _same(prof["outputs"][:n_out], out_plain)
    if not profile_trusted:
        res.add("TS564", "the profiled run (used for residency) gave a different final output "
                         "than the plain run; node placement is still reported")
    comp = where["compute_on_cpu"]
    res.notes.append(
        f"residency: {where['device']} of {where['nodes']} nodes ran on {T.label}, "
        f"{len(where['shape_on_cpu'])} integer shape nodes on the CPU EP, {len(comp)} compute "
        f"nodes on the CPU EP, {where['transfers']} transfers")
    if comp:
        names = ", ".join(f"{op} {n}" for n, op in comp[:10])
        res.add(T.cpu_nodes_code, f"{len(comp)} compute node(s) ran on the CPU EP, not on {T.label}: {names}"
                + (" ..." if len(comp) > 10 else ""))
        res.status = worst(res.status, SUSPECT)

    # 4. Taps (observer guard).
    t = time.perf_counter()
    tapped_ok = False
    tapped: Dict[str, List[torch.Tensor]] = {}
    try:
        tap_bytes, tap_feeds, tap_n, layout = export(base, args, kwargs, dtype, tap=True)
        tr = worker.call("run", {"model": tap_bytes, "feeds": tap_feeds, "provider": T.provider})
        if tr["fallback"]:
            res.notes.append("tapped graph fell back to the CPU EP; per module taps not used")
        elif _same(tr["outputs"][:tap_n], out_plain):
            tapped_ok = True
        else:
            d = diff_lists(_tensors(tr["outputs"][:tap_n]), _tensors(out_plain))
            res.add("TS564", f"adding module outputs as graph outputs changed the final output on "
                             f"{T.label} (relative difference {d.rel_err:.2e} from the plain run): "
                             "per module accumulated errors are untrusted; every module is "
                             "replayed in isolation instead")
        pos = tap_n
        for key, n in layout:
            tapped[key] = _tensors(tr["outputs"][pos:pos + n])
            pos += n
    except ExportFailed as exc:
        res.notes.append(f"tapped export failed ({exc}); every module is replayed in isolation")
    timings["taps"] = time.perf_counter() - t

    # 5. Isolated replay, children before parents.
    t = time.perf_counter()
    base_floor = DTYPE_FLOOR.get(dtype, 1e-4)
    bad_inside: Dict[str, bool] = {}
    results: Dict[str, LayerResult] = {}
    exports_run = 0
    export_failures: List[str] = []
    export_off: List[str] = []
    isolated_bytes: Dict[str, Tuple[bytes, Dict[str, np.ndarray]]] = {}
    for c in ref_calls:
        n = noise.get(c.key)
        if n is not None and n > base_floor * DIVERGENT_FACTOR:
            n = None
        thr = threshold_for(dtype, False, n)
        prop = None
        if c.key in tapped:
            prop = diff_lists(tapped[c.key], c.outputs)
        elif c.name == ROOT:
            prop = diff_lists(_tensors(out_plain), c.outputs)
        inner_bad = any(results[k].status != OK or bad_inside.get(k, False) for k in c.children
                        if k in results)
        bad_inside[c.key] = inner_bad
        prop_status = prop.status(thr) if prop is not None else None
        unexplained = prop_status == DIVERGENT
        needs = _needs_replay(c, op_counting) or unexplained
        # With trusted taps, a module whose output already matches in the full run is not
        # replayed: its own error cannot be over threshold unless later layers cancel it.
        matched = tapped_ok and prop_status == OK
        if inner_bad or not needs or matched:
            results[c.key] = LayerResult(c.key, c.name, c.type_name, c.is_leaf, None, prop, thr,
                                         OK, replayed=False)
            continue
        iso, err, status = None, "", DIVERGENT
        try:
            mod = _module_by_name(base, c.name)
            data, mfeeds, m_out, _ = export(mod, c.args, c.kwargs, dtype)
            exports_run += 1
            r = worker.call("run", {"model": data, "feeds": mfeeds, "provider": T.provider})
            if r["fallback"]:
                raise RuntimeError(f"the isolated graph ran on the CPU EP, not on {T.label}")
            iso = diff_lists(_tensors(r["outputs"][:m_out]), c.outputs)
            status = iso.status(thr)
            if not c.is_leaf and status == SUSPECT:
                status = OK
            if status != OK:
                isolated_bytes[c.key] = (data, mfeeds)
                cr = worker.call("run", {"model": data, "feeds": mfeeds, "provider": "cpu"})
                cd = diff_lists(_tensors(cr["outputs"][:m_out]), c.outputs)
                if cd.status(thr) != OK:
                    export_off.append(f"{c.key} ({c.type_name}) CPU EP {cd.rel_err:.2e}, "
                                      f"{T.label} {iso.rel_err:.2e}")
        except ExportFailed as exc:
            export_failures.append(f"{c.key} ({c.type_name}): {exc}")
            results[c.key] = LayerResult(c.key, c.name, c.type_name, c.is_leaf, None, prop, thr,
                                         OK, replayed=False, error=f"export failed: {exc}")
            continue
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        results[c.key] = LayerResult(c.key, c.name, c.type_name, c.is_leaf, iso, prop, thr, status,
                                     True, err)
        res.status = worst(res.status, status)
    timings["isolated replay"] = time.perf_counter() - t
    if export_off:
        res.add("TS563", f"{len(export_off)} module(s) exported alone are already off on ONNX "
                         "Runtime's CPU EP, so the export or ONNX Runtime's own kernels change the "
                         f"result, not only {T.label}: " + "; ".join(export_off[:3])
                + (" ..." if len(export_off) > 3 else ""))
    if export_failures:
        res.add("TS562", f"{len(export_failures)} module(s) could not be exported alone and were "
                         f"not replayed: {'; '.join(export_failures[:3])}")
    res.notes.append(
        f"per module data: {'taps trusted (tapped final output identical to the plain run)' if tapped_ok else 'taps untrusted or missing'}; "
        f"{exports_run} module(s) exported and replayed alone")

    res.layers = sorted(results.values(), key=lambda l: ref_by_key[l.key].start)
    bad = [l for l in res.layers if l.status != OK]
    divergent = [l for l in bad if l.status == DIVERGENT]
    res.first_divergent = divergent[0] if divergent else (bad[0] if bad else None)
    fd = res.first_divergent
    if fd is not None:
        own = fd.isolated.rel_err if fd.isolated is not None else float("nan")
        others = len(bad) - 1
        res.add("TS010" if fd.status == DIVERGENT else "TS011",
                f"module {fd.key} ({fd.type_name}) own error {own:.2e} vs threshold "
                f"{fd.threshold:.1e} on {b.name}"
                + (f"; {others} more module(s) over threshold" if others else ""))
        if fd.error:
            res.add("TS014", f"module {fd.key} failed when replayed on {b.name}: {fd.error}")
        try:
            from ..agent import module_source

            res.blame_source = module_source(base, fd.name)
        except Exception:
            res.blame_source = None

    # Final output and decisions.
    out_t = _tensors(out_plain)
    res.output = diff_lists(out_t, ref_flat)
    if res.output.status(out_thr) == DIVERGENT:
        res.status = worst(res.status, DIVERGENT)
        if res.first_divergent is None:
            res.add("TS012", f"The final output diverges on {T.label} but no module is to blame on its "
                             "own: the cause is code outside any module, or how the EP combines "
                             "nodes across module boundaries.")
    if out_t and ref_flat:
        nd, flips, ties = _decisions(out_t[0], ref_flat[0], out_thr)
        res.decisions, res.flips, res.tie_flips = nd, flips, ties
        if flips - ties > 0:
            res.status = worst(res.status, SUSPECT)
            res.add("TS013", f"{flips - ties} of {nd} predictions changed beyond near ties "
                             f"({ties} near ties)")

    # 6. Op drilldown and 7. repros.
    t = time.perf_counter()
    op_case = None
    if fd is not None and fd.key in isolated_bytes:
        data, mfeeds = isolated_bytes[fd.key]
        if drill_ops:
            try:
                op, op_notes, op_case = drill(worker, data, mfeeds, fd.threshold, fd.key, T)
                res.first_divergent_op = op
                res.notes.extend(op_notes)
            except Exception as exc:
                res.notes.append(f"op drilldown failed on {fd.key}: {type(exc).__name__}: {exc}")
        if repro_dir:
            c = ref_by_key[fd.key]
            folder = os.path.join(repro_dir, f"{_safe(model_name)}_{_safe(b.name)}_{_safe(fd.key)}")
            try:
                res.repro_path = (T.repro or write_repro)(
                    folder, "repro", b, data, mfeeds, [o.numpy() for o in c.outputs],
                    fd.threshold, f"Module:  {fd.key} ({fd.type_name}) from model {model_name}, "
                    "exported alone with the float64 reference inputs",
                    "the float64 PyTorch CPU reference")
                if op_case is not None and res.first_divergent_op is not None:
                    ndata, nfeeds, nexp = op_case
                    res.op_repro_path = (T.repro or write_repro)(
                        folder, "op_repro", b, ndata, nfeeds, nexp, fd.threshold,
                        f"Node:    {res.first_divergent_op.op} in module {fd.key}, alone, with "
                        "the CPU EP's values as inputs", "ONNX Runtime's CPU EP")
            except Exception as exc:
                res.notes.append(f"No repro written ({type(exc).__name__}: {exc}).")
    timings["drilldown and repro"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t0
    res.notes.append(f"{T.provider} timings (s): " + ", ".join(f"{k} {v:.2f}" for k, v in timings.items()))
