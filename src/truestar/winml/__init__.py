"""Windows ML execution providers as `truestar check` backends (docs/BACKENDS.md, research item 6.1).

Backend names: ``winml:<ep>[:<device index>][+option=value...]`` with ``<ep>`` one of

    nvtrtx     NVIDIA TensorRT for RTX (NvTensorRTRTXExecutionProvider)
    webgpu     WebGPU (WebGpuExecutionProvider, experimental in Windows ML 2.x)
    dml        DirectML (DmlExecutionProvider, built into the Windows ML ONNX Runtime)
    openvino   Intel OpenVINO          qnn       Qualcomm QNN
    vitisai    AMD Vitis AI            amdgpu    AMD GPU          migraphx   AMD MIGraphX

Every run is the shared ONNX path of :mod:`truestar.webgpu.onnx_check`: the model exported to ONNX,
a plain run twice, residency from a profiled run (a session that fell back to the CPU EP is TS1120,
CPU compute nodes TS1121), per module taps and isolated replay against float64 (the per layer error
of every module), op drilldown and a repro. The EP runs in a worker process (`TRUESTAR_WINML_PYTHON`
picks its Python; it needs onnxruntime 1.23 or later with plugin EP support, onnx and numpy).
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

from ..backends import Backend, BackendUnavailable

PREFIX = "winml"
SHORT = ("nvtrtx", "webgpu", "dml", "openvino", "qnn", "vitisai", "amdgpu", "migraphx")

_WORKERS: Dict[str, object] = {}
_PROBED: Dict[str, Tuple[Optional[Backend], str]] = {}


def is_winml(name: str) -> bool:
    return name.strip().lower().startswith(PREFIX + ":")


def parse_spec(name: str) -> Tuple[str, int, Dict[str, str]]:
    """'winml:nvtrtx:0+opt=v' -> ('nvtrtx', 0, {'opt': 'v'})."""
    parts = name.strip().split("+")
    opts: Dict[str, str] = {}
    for p in parts[1:]:
        if "=" not in p:
            raise ValueError(f"EP option {p!r} in {name!r} must be key=value")
        k, v = p.split("=", 1)
        opts[k.strip()] = v.strip()
    spec = parts[0].strip().lower().split(":")
    if spec[0] != PREFIX or len(spec) < 2 or len(spec) > 3:
        raise ValueError(f"not a Windows ML backend name: {name!r} (use winml:<ep>[:<index>], "
                         f"<ep> one of {', '.join(SHORT)})")
    if spec[1] not in SHORT:
        raise ValueError(f"unknown Windows ML EP {spec[1]!r} in {name!r}; use one of {', '.join(SHORT)}")
    index = 0
    if len(spec) == 3:
        if not spec[2].isdigit():
            raise ValueError(f"device index must be a number in {name!r}")
        index = int(spec[2])
    return spec[1], index, opts


def worker_python() -> str:
    return os.environ.get("TRUESTAR_WINML_PYTHON") or sys.executable


def WinmlWorkerClient(short: str, index: int, ep_options: Dict[str, str]):
    """The WebGPU worker client protocol, hosting the Windows ML worker module."""
    from ..webgpu.client import WorkerClient

    return WorkerClient(short, index, ep_options, timeout=900.0, module="truestar.winml.worker",
                        python=worker_python(), label="Windows ML")


def get_worker(name: str):
    short, index, opts = parse_spec(name)
    key = name.strip()
    w = _WORKERS.get(key)
    if w is not None and w.proc is not None and w.proc.poll() is None:
        return w
    w = WinmlWorkerClient(short, index, opts)
    _WORKERS[key] = w
    w.info = w.call("hello")
    w.ep_name = w.info.get("ep", "")
    return w


def close_workers() -> None:
    for w in list(_WORKERS.values()):
        try:
            w.close()
        except Exception:
            pass
    _WORKERS.clear()


def probe(name: str) -> Tuple[Optional[Backend], str]:
    """(Backend, "") when the named Windows ML EP runs a one node graph here, else (None, reason)."""
    key = name.strip()
    if key in _PROBED:
        return _PROBED[key]
    try:
        short, index, opts = parse_spec(name)
    except ValueError as exc:
        return None, str(exc)
    if sys.platform != "win32":
        result = (None, "Windows ML runs on Windows only")
        _PROBED[key] = result
        return result
    try:
        import onnx  # noqa: F401  (needed in this process to export)
    except Exception as exc:
        result = (None, f"the onnx package is not importable here ({type(exc).__name__})")
        _PROBED[key] = result
        return result
    try:
        w = get_worker(name)
        info = w.info
    except Exception as exc:
        result = (None, f"the Windows ML worker did not start: {exc}")
        _PROBED[key] = result
        return result
    if info.get("unavailable"):
        result = (None, info["unavailable"])
        _PROBED[key] = result
        return result
    dev = info.get("device") or {}
    where = dev.get("description") or ("DirectML device %d" % index if short == "dml" else f"device {index}")
    desc = (f"Windows ML EP {info['ep']} ({info.get('source')}"
            + (f" {info['ep_version']}" if info.get("ep_version") else "")
            + f") on {where.strip()}, ONNX Runtime {info['onnxruntime']} ({info.get('onnxruntime_package')})")
    if opts:
        desc += ", EP options " + ", ".join(f"{k}={v}" for k, v in sorted(opts.items()))
    notes = []
    if info.get("onnxruntime_package") != "onnxruntime-windowsml" and info.get("source", "").startswith(
            ("Windows ML", "bring")):
        notes.append(f"TS1123: {info['ep']} ran inside {info.get('onnxruntime_package')} "
                     f"{info['onnxruntime']}, not the ONNX Runtime Windows ML ships; the EP's kernels are "
                     "its own, graph partitioning and the CPU fallback kernels are this runtime's")
    b = Backend(name=key, device="winml", description=desc, notes=notes, kind="winml",
                info=dict(info), options=dict(opts))
    result = (b, "")
    _PROBED[key] = result
    return result


def resolve(name: str) -> Backend:
    b, reason = probe(name)
    if b is None:
        raise BackendUnavailable(f"Requested backend not available on this machine: {name}. "
                                 f"Reason: {reason}", [name])
    return b


def catalog(python: Optional[str] = None) -> Dict:
    """What the Windows ML EP catalog offers on this PC, without downloading anything:
    {"providers": [(name, state, library path)], "note": str}."""
    w = WinmlWorkerClient("webgpu", 0, {})
    try:
        info = w.call("hello")
        return {"providers": info.get("catalog"), "note": info.get("catalog_note", "")}
    finally:
        w.close()


def ensure_ready(short: str) -> Dict:
    """Make one catalog EP ready: downloads and installs it through Windows Update when it is
    NotPresent (system wide, can take minutes). Returns the catalog after the attempt and the
    result (status 0 success; diagnostic text and HRESULT otherwise)."""
    if short not in SHORT:
        raise ValueError(f"unknown Windows ML EP {short!r}")
    w = WinmlWorkerClient(short, 0, {})
    try:
        w.call("hello")
        return w.call("ensure")
    finally:
        w.close()


# ----------------------------------------------------------------------------- check


_SCRIPT = '''"""Repro generated by truestar check for a Windows ML execution provider.

{what}
Backend:  {backend} ({description})
Expected: relative error below {threshold:.1e} versus {expected_from}.

Run:  python {script}          ({ep})
      python {script} cpu      (control run on ONNX Runtime's CPU EP)
Needs numpy and onnxruntime 1.23 or later; the EP library recorded below (or DirectML built into
onnxruntime-windowsml or onnxruntime-directml). Exit code 1 means the divergence reproduced.
"""
import os
import sys

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = {threshold!r}
EP = {ep!r}
LIBRARY = {library!r}  # where the EP library was on the machine that ran the check
OPTIONS = {options!r}


def main():
    so = ort.SessionOptions()
    use_cpu = len(sys.argv) > 1 and sys.argv[1] == "cpu"
    providers = ["CPUExecutionProvider"]
    if not use_cpu:
        if EP == "DmlExecutionProvider":
            providers = [(EP, OPTIONS), "CPUExecutionProvider"]
        else:
            ort.register_execution_provider_library(EP, os.environ.get("TRUESTAR_EP_LIBRARY", LIBRARY))
            devs = [d for d in ort.get_ep_devices() if d.ep_name == EP]
            so.add_provider_for_devices(devs[{index}:{index} + 1], OPTIONS)
            providers = None
    sess = ort.InferenceSession(os.path.join(HERE, "{onnx}"), sess_options=so, providers=providers)
    if not use_cpu and EP not in sess.get_providers():
        print(EP, "was not created; ONNX Runtime fell back to", sess.get_providers())
        return 2
    feeds = dict(np.load(os.path.join(HERE, "{inputs}")))
    expected = np.load(os.path.join(HERE, "{expected}"))
    worst = 0.0
    for i, got in enumerate(sess.run(None, feeds)):
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
    print("worst relative error %.3e, threshold %.1e" % (worst, THRESHOLD))
    return 1 if worst > THRESHOLD else 0


if __name__ == "__main__":
    sys.exit(main())
'''


def write_repro(folder, stem, backend, data, feeds, expected, threshold, what, expected_from) -> str:
    import numpy as np

    os.makedirs(folder, exist_ok=True)
    onnx_name, in_name, exp_name, script = (f"{stem}.onnx", f"{stem}_inputs.npz",
                                           f"{stem}_expected.npz", f"{stem}.py")
    with open(os.path.join(folder, onnx_name), "wb") as f:
        f.write(data)
    np.savez(os.path.join(folder, in_name), **feeds)
    np.savez(os.path.join(folder, exp_name),
             **{f"out{i}": np.asarray(e, dtype=np.float64) for i, e in enumerate(expected)})
    _, index, _ = parse_spec(backend.name)
    path = os.path.join(folder, script)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_SCRIPT.format(what=what, backend=backend.name, description=backend.description,
                               threshold=threshold, expected_from=expected_from, script=script,
                               ep=backend.info.get("ep", ""), library=backend.info.get("library_path", ""),
                               options=dict(backend.options or {}), index=index, onnx=onnx_name,
                               inputs=in_name, expected=exp_name))
    return path


def check_backend(res, base, args, kwargs, dtype, ref_calls, ref_flat, noise, op_counting,
                  repro_dir, model_name, drill_ops: bool = True) -> None:
    """Fill a check.BackendResult for a Windows ML backend (the shared ONNX path)."""
    from ..webgpu.onnx_check import EpTarget, check_backend as onnx_check_backend

    b = res.backend
    worker = get_worker(b.name)
    target = EpTarget("winml", b.info.get("ep", ""), f"Windows ML {b.info.get('ep', '')}",
                      fallback_code="TS1120", cpu_nodes_code="TS1121",
                      nondeterministic_code="TS1122", repro=write_repro)
    for n in b.notes:
        if n.startswith("TS1123: "):
            res.add("TS1123", n[len("TS1123: "):])
    onnx_check_backend(res, base, args, kwargs, dtype, ref_calls, ref_flat, noise, op_counting,
                       repro_dir, model_name, drill_ops, target=target, worker=worker)
