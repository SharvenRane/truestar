"""WebGPU as a local `truestar check` backend (docs/WEBGPU.md).

Backend names:

* ``webgpu``: ONNX Runtime's WebGPU EP (Dawn, the WebGPU implementation inside Chrome) on the
  platform's default Dawn backend (D3D12 on Windows, Metal on macOS, Vulkan on Linux), adapter 0;
* ``webgpu:d3d12``, ``webgpu:vulkan``, ``webgpu:metal``: a chosen Dawn backend;
* ``webgpu:d3d12:1``: a chosen adapter index in Dawn's list for that backend (this machine: 0 is the
  discrete GPU, 1 the integrated GPU, 2 Microsoft's software rasterizer).

The model is exported to ONNX in the parent process (torch and onnx needed) and run in a worker
process that needs only numpy and onnxruntime with the ``onnxruntime-ep-webgpu`` plugin.
"""

from __future__ import annotations

import sys
from typing import Dict, Optional, Tuple

from ..backends import Backend, BackendUnavailable

PREFIX = "webgpu"
_DAWN = {"d3d12": "D3D12", "vulkan": "Vulkan", "metal": "Metal"}

# Live workers, one per backend name; closed by close_workers() at the end of a check.
_WORKERS: Dict[str, object] = {}
_PROBED: Dict[str, Tuple[Optional[Backend], str]] = {}


def is_webgpu(name: str) -> bool:
    n = name.strip().lower()
    return n == PREFIX or n.startswith((PREFIX + ":", PREFIX + "+"))


def default_dawn_backend() -> str:
    if sys.platform == "win32":
        return "D3D12"
    if sys.platform == "darwin":
        return "Metal"
    return "Vulkan"


def split_options(name: str) -> Tuple[str, Dict[str, str]]:
    """'webgpu:d3d12+enableMatmulFp32Accumulation=1' -> ('webgpu:d3d12', {option: value}).

    Options after each "+" go to the WebGPU EP unchanged (their case matters)."""
    parts = name.strip().split("+")
    opts: Dict[str, str] = {}
    for p in parts[1:]:
        if "=" not in p:
            raise ValueError(f"EP option {p!r} in {name!r} must be key=value")
        k, v = p.split("=", 1)
        opts[k.strip()] = v.strip()
    return parts[0].strip().lower(), opts


def parse_spec(name: str) -> Tuple[str, int]:
    """'webgpu[:backend[:adapter]][+option=value...]' -> (Dawn backend type, adapter index)."""
    spec, _ = split_options(name)
    parts = spec.split(":")
    if parts[0] != PREFIX or len(parts) > 3:
        raise ValueError(f"not a WebGPU backend name: {name!r} (use webgpu, webgpu:d3d12, "
                         "webgpu:vulkan, webgpu:metal, optionally :<adapter index>)")
    backend = default_dawn_backend()
    if len(parts) >= 2 and parts[1]:
        if parts[1] not in _DAWN:
            raise ValueError(f"unknown Dawn backend {parts[1]!r} in {name!r}; use one of "
                             + ", ".join(sorted(_DAWN)))
        backend = _DAWN[parts[1]]
    index = 0
    if len(parts) == 3:
        if not parts[2].isdigit():
            raise ValueError(f"adapter index must be a number in {name!r}")
        index = int(parts[2])
    return backend, index


def get_worker(name: str, ep_options: Optional[Dict[str, str]] = None):
    """The live worker for a backend name, started on first use."""
    from .client import WorkerClient

    key = name.strip() + ("|" + repr(sorted(ep_options.items())) if ep_options else "")
    w = _WORKERS.get(key)
    if w is not None and w.proc is not None and w.proc.poll() is None:
        return w
    backend, index = parse_spec(name)
    w = WorkerClient(backend, index, ep_options)
    _WORKERS[key] = w
    w.info = w.call("hello")
    return w


def close_workers() -> None:
    for w in list(_WORKERS.values()):
        try:
            w.close()
        except Exception:
            pass
    _WORKERS.clear()


def probe(name: str, ep_options: Optional[Dict[str, str]] = None) -> Tuple[Optional[Backend], str]:
    """(Backend, "") when the named WebGPU backend runs here, else (None, reason).

    Starts the worker and runs a one node graph that must execute on the WebGPU EP (ONNX Runtime
    otherwise falls back to its CPU EP without raising)."""
    key = name.strip() + (repr(sorted(ep_options.items())) if ep_options else "")
    if key in _PROBED:
        return _PROBED[key]
    try:
        backend_type, index = parse_spec(name)
        ep_options = dict(split_options(name)[1], **(ep_options or {}))
    except ValueError as exc:
        return None, str(exc)
    try:
        import onnx  # noqa: F401  (needed in this process to export)
    except Exception as exc:
        result = (None, f"the onnx package is not importable here ({type(exc).__name__}); it is "
                        "needed to export the model for WebGPU")
        _PROBED[key] = result
        return result
    try:
        w = get_worker(name, ep_options)
        info = w.info
    except Exception as exc:
        result = (None, f"the WebGPU worker did not start: {exc}")
        _PROBED[key] = result
        return result
    if info.get("unavailable"):
        result = (None, info["unavailable"])
        _PROBED[key] = result
        return result
    adapter = info.get("adapter")
    n = len(info["adapters"]) if info.get("adapters") is not None else None
    if adapter is not None:
        where = f"{adapter['device']} ({adapter['adapter_type']}), adapter {index}" + (
            f" of {n}" if n is not None else "")
    elif n is not None:
        where = f"adapter {index} of {n} (index out of range per wgpudawn)"
    else:
        where = f"adapter {index} (not named: {info.get('adapter_note', '')})"
    desc = (f"WebGPU via ONNX Runtime {info.get('onnxruntime')} WebGPU EP "
            f"{info.get('ep_version') or '?'}, Dawn {backend_type}, {where}")
    notes = []
    if adapter is not None and str(adapter.get("adapter_type", "")).upper() == "CPU":
        notes.append("This adapter is a software rasterizer on the CPU (for example Microsoft "
                     "Basic Render Driver): results come from WebGPU, but not from a GPU.")
    if ep_options:
        desc += ", EP options " + ", ".join(f"{k}={v}" for k, v in sorted(ep_options.items()))
    b = Backend(name=name.strip(), device="webgpu", description=desc, notes=notes,
                kind="webgpu", info=dict(info), options=dict(ep_options or {}))
    result = (b, "")
    _PROBED[key] = result
    return result


def resolve(name: str) -> Backend:
    b, reason = probe(name)
    if b is None:
        raise BackendUnavailable(f"Requested backend not available on this machine: {name}. "
                                 f"Reason: {reason}", [name])
    return b
