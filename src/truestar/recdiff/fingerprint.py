"""Environment fingerprint: everything outside the model that can change its numerics.

A flat dict of stable values (no time, no host name, no process id), so identical environments
give identical fingerprints and identical recordings. Each key has a kind:

* version:   library and driver versions (torch, CUDA, cuDNN, Triton, driver, ...)
* precision: settings that knowingly trade accuracy for speed (TF32, reduced precision reductions)
* algorithm: settings that change which kernel runs without changing its precision (cuDNN
             benchmark, deterministic algorithms, SDPA backends, thread count)
* hardware:  the device and platform
"""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any, Callable, Dict, List, Tuple

import torch

VERSION, PRECISION, ALGORITHM, HARDWARE = "version", "precision", "algorithm", "hardware"

# Keys whose change explains a precision class difference on CUDA.
_PRECISION_KEYS = (
    "flags.cuda.matmul.allow_tf32",
    "flags.cudnn.allow_tf32",
    "flags.float32_matmul_precision",
    "flags.cuda.matmul.allow_fp16_reduced_precision_reduction",
    "flags.cuda.matmul.allow_bf16_reduced_precision_reduction",
    "flags.cuda.matmul.fp32_precision",
    "flags.cudnn.conv.fp32_precision",
    "flags.cudnn.rnn.fp32_precision",
    "flags.fp32_precision",
    "env.NVIDIA_TF32_OVERRIDE",
    "env.TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
)
_VERSION_KEYS = ("python", "torch", "torch.git", "cuda.runtime", "hip", "cudnn", "triton",
                 "numpy", "driver")
_HARDWARE_KEYS = ("platform", "machine", "gpu.name", "gpu.capability", "gpu.count")


def kind(key: str) -> str:
    if key.startswith("runtime."):  # deployed artifact runtimes (docs/CONFORM.md)
        if key.endswith(".precision"):
            return PRECISION
        if key.endswith("adapter"):
            return HARDWARE
        return VERSION
    if key in _PRECISION_KEYS:
        return PRECISION
    if key in _VERSION_KEYS:
        return VERSION
    if key in _HARDWARE_KEYS:
        return HARDWARE
    return ALGORITHM


def _safe(fn: Callable[[], Any]) -> Any:
    try:
        v = fn()
    except Exception as exc:  # a missing or deprecated API must never break a recording
        return f"unavailable ({type(exc).__name__})"
    if isinstance(v, (bool, int, str)) or v is None:
        return v
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _dist_version(*names: str) -> Any:
    from importlib.metadata import PackageNotFoundError, version

    for n in names:
        try:
            return f"{n} {version(n)}"
        except PackageNotFoundError:
            continue
    return None


_DRIVER_CACHE: Dict[str, Any] = {}


def _driver() -> Any:
    if "v" not in _DRIVER_CACHE:
        v = None
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                  "--format=csv,noheader"], capture_output=True, text=True,
                                 timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                v = out.stdout.strip().splitlines()[0].strip()
        except Exception:
            v = None
        _DRIVER_CACHE["v"] = v
    return _DRIVER_CACHE["v"]


def fingerprint() -> Dict[str, Any]:
    cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    m = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    fp: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.system() + " " + platform.release(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch.git": _safe(lambda: torch.version.git_version),
        "cuda.runtime": _safe(lambda: torch.version.cuda),
        "hip": _safe(lambda: getattr(torch.version, "hip", None)),
        "cudnn": _safe(lambda: cudnn.version() if cudnn.is_available() else None),
        "triton": _dist_version("triton", "pytorch-triton", "triton-windows",
                                "pytorch-triton-rocm"),
        "numpy": _dist_version("numpy"),
        "driver": _driver() if cuda and not getattr(torch.version, "hip", None) else None,
        "gpu.name": _safe(lambda: torch.cuda.get_device_name(0)) if cuda else None,
        "gpu.capability": _safe(lambda: ".".join(map(str, torch.cuda.get_device_capability(0))))
        if cuda else None,
        "gpu.count": _safe(torch.cuda.device_count) if cuda else 0,
        "flags.cuda.matmul.allow_tf32": _safe(lambda: m.allow_tf32),
        "flags.cudnn.allow_tf32": _safe(lambda: cudnn.allow_tf32),
        "flags.float32_matmul_precision": _safe(torch.get_float32_matmul_precision),
        "flags.cuda.matmul.allow_fp16_reduced_precision_reduction":
            _safe(lambda: m.allow_fp16_reduced_precision_reduction),
        "flags.cuda.matmul.allow_bf16_reduced_precision_reduction":
            _safe(lambda: m.allow_bf16_reduced_precision_reduction),
        "flags.cuda.matmul.fp32_precision": _safe(lambda: m.fp32_precision),
        "flags.cudnn.conv.fp32_precision": _safe(lambda: cudnn.conv.fp32_precision),
        "flags.cudnn.rnn.fp32_precision": _safe(lambda: cudnn.rnn.fp32_precision),
        "flags.fp32_precision": _safe(lambda: torch.backends.fp32_precision),
        "flags.cudnn.enabled": _safe(lambda: cudnn.enabled),
        "flags.cudnn.benchmark": _safe(lambda: cudnn.benchmark),
        "flags.cudnn.deterministic": _safe(lambda: cudnn.deterministic),
        "flags.deterministic_algorithms": _safe(torch.are_deterministic_algorithms_enabled),
        "flags.mkldnn.enabled": _safe(lambda: torch.backends.mkldnn.enabled),
        "flags.num_threads": _safe(torch.get_num_threads),
        "flags.sdp.flash": _safe(torch.backends.cuda.flash_sdp_enabled),
        "flags.sdp.mem_efficient": _safe(torch.backends.cuda.mem_efficient_sdp_enabled),
        "flags.sdp.math": _safe(torch.backends.cuda.math_sdp_enabled),
        "flags.sdp.cudnn": _safe(lambda: torch.backends.cuda.cudnn_sdp_enabled()),
    }
    for var in ("CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUDA_LAUNCH_BLOCKING", "OMP_NUM_THREADS"):
        fp[f"env.{var}"] = os.environ.get(var)
    return fp


def differences(a: Dict[str, Any], b: Dict[str, Any]) -> List[Tuple[str, Any, Any, str]]:
    """(key, recorded, current, kind) for every key whose value differs, sorted by kind then key."""
    order = {PRECISION: 0, VERSION: 1, HARDWARE: 2, ALGORITHM: 3}
    out = [(k, a.get(k), b.get(k), kind(k)) for k in sorted(set(a) | set(b))
           if a.get(k) != b.get(k)]
    return sorted(out, key=lambda x: (order[x[3]], x[0]))


def runtime_fingerprint(runtime: str, versions: Dict[str, Any]) -> Dict[str, Any]:
    """Fingerprint of a deployed artifact runtime (`truestar conform`, docs/CONFORM.md): the host
    keys that can change its numerics (Python, OS, numpy, GPU and driver) plus the runtime's own
    versions, SDK files and precision settings as `runtime.<key>`. Stable values only, like
    fingerprint(), so two identical environments compare equal."""
    cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    fp: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.system() + " " + platform.release(),
        "machine": platform.machine(),
        "numpy": _dist_version("numpy"),
        "driver": _driver() if cuda and not getattr(torch.version, "hip", None) else None,
        "gpu.name": _safe(lambda: torch.cuda.get_device_name(0)) if cuda else None,
        "runtime.name": runtime,
    }
    for k, v in sorted(versions.items()):
        fp[f"runtime.{k}"] = v if isinstance(v, (bool, int, str)) or v is None else str(v)
    return fp
