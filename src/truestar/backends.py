"""Discover compute backends and describe the precision settings that affect their results."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

KNOWN = ("cpu", "cuda", "mps", "xpu")


class BackendUnavailable(RuntimeError):
    """A requested backend is not available on this machine (code TS001)."""

    def __init__(self, message: str, missing: List[str]):
        super().__init__(message)
        self.missing = missing


@dataclass
class Backend:
    name: str  # "cpu", "cuda", "rocm", "mps", "xpu"
    device: str  # torch device string
    description: str
    tf32: bool = False  # TF32 math enabled for conv or matmul
    notes: List[str] = field(default_factory=list)
    # "torch": a torch device runs the model. "webgpu": the model is exported to ONNX and run by
    # ONNX Runtime's WebGPU EP in a worker process (truestar.webgpu, docs/WEBGPU.md).
    kind: str = "torch"
    info: Dict = field(default_factory=dict)  # backend specific facts (adapter, versions)
    options: Dict = field(default_factory=dict)  # backend specific settings (EP options)
    # A simulated device (docs/DEVICES.md): the name of a truestar.devsim profile. The model runs on
    # the CPU under that profile's numerics; nothing runs on the device itself.
    sim: Optional[str] = None

    def as_dict(self) -> Dict:
        d = {
            "name": self.name,
            "device": self.device,
            "description": self.description,
            "tf32": self.tf32,
            "notes": list(self.notes),
        }
        if self.kind != "torch":
            d["kind"] = self.kind
            d["info"] = {k: v for k, v in self.info.items() if k != "python"}
        if self.sim:
            d["simulated"] = True
        return d

    def context(self, notes=None):
        """Context manager to run this backend's ops in: the profile's numerics for a simulated
        backend, nothing for a real one."""
        if not self.sim:
            import contextlib

            return contextlib.nullcontext()
        from .devsim import simulate

        return simulate(self.sim, notes)

    def calibration_context(self):
        """Correct arithmetic in this backend's precision (the noise floor), or None for a real
        backend, which is calibrated on the CPU in the model dtype."""
        if not self.sim:
            return None
        from .devsim import get_profile, simulate

        return simulate(get_profile(self.sim).ideal())

    def floor_dtype(self, dtype):
        """The dtype whose tolerance floor applies on this backend."""
        if not self.sim:
            return dtype
        from .devsim import get_profile

        fd = get_profile(self.sim).floor_dtype
        if fd is None or not dtype.is_floating_point or torch.finfo(fd).bits >= torch.finfo(dtype).bits:
            return dtype
        return fd


def sim_backend(name: str) -> Backend:
    """The Backend for a simulated device such as ``sim:ane`` (KeyError when unknown)."""
    from .devsim import get_profile

    p = get_profile(name)
    guesses = len(p.guesses)
    notes = [f"simulated on the CPU, no device ran: {p.description}. {len(p.rules)} rules, "
             f"{guesses} of them guesses (truestar backends --sims lists each with its source)."]
    return Backend(name=p.name, device="cpu", description=f"simulated {p.name[4:]}", notes=notes,
                   sim=p.name)


def _cuda_backend() -> Optional[Backend]:
    # is_available() can be True with zero usable devices (for example CUDA_VISIBLE_DEVICES=""),
    # so require a device that actually answers.
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return None
        dev_name = torch.cuda.get_device_name(0)
    except Exception:
        return None
    hip = getattr(torch.version, "hip", None)
    name = "rocm" if hip else "cuda"
    version = f"HIP {hip}" if hip else f"CUDA {torch.version.cuda}"
    tf32_conv = bool(torch.backends.cudnn.allow_tf32)
    tf32_mm = bool(torch.backends.cuda.matmul.allow_tf32)
    notes = []
    if tf32_conv or tf32_mm:
        parts = []
        if tf32_conv:
            parts.append("convolutions")
        if tf32_mm:
            parts.append("matmuls")
        notes.append(
            "TF32 is enabled for " + " and ".join(parts) + ": these run with about 10 bits of "
            "mantissa, so differences around 1e-3 are expected, not bugs."
        )
    return Backend(
        name=name,
        device="cuda:0",
        description=f"{dev_name}, {version}",
        tf32=tf32_conv or tf32_mm,
        notes=notes,
    )


def _mps_backend() -> Optional[Backend]:
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return None
    notes = []
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        notes.append(
            "PYTORCH_ENABLE_MPS_FALLBACK=1 is set: unsupported ops silently run on the CPU, so "
            "some results below may not come from the GPU."
        )
    return Backend(name="mps", device="mps", description="Apple Metal (MPS)", notes=notes)


def _xpu_backend() -> Optional[Backend]:
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not xpu.is_available():
        return None
    return Backend(name="xpu", device="xpu:0", description=f"Intel XPU {xpu.get_device_name(0)}")


def available_backends() -> List[Backend]:
    """Every backend this machine can run, CPU first."""
    found = [Backend(name="cpu", device="cpu", description="CPU")]
    for probe in (_cuda_backend, _mps_backend, _xpu_backend):
        b = probe()
        if b is not None:
            found.append(b)
    return found


def resolve_backends(requested: Optional[List[str]], allow_sim: bool = False,
                     allow_stacks: bool = False
                     ) -> Tuple[List[Backend], List[Tuple[str, str]]]:
    """Return (backends to run, skipped) where skipped lists (name, reason).

    Explicitly requested backends that are missing raise, so nothing is skipped silently.
    With no request, every available backend runs and unavailable known ones are listed as skipped.
    Simulated backends (``sim:*``) are only ever run when requested, and only by commands that
    pass ``allow_sim`` (today ``truestar check`` and ``truestar probe``); any other command refuses
    them rather than run the model on the plain CPU under a simulated name. Compiler and runtime
    stacks (``cuda+inductor``, ``ort:*``, ``trt:*``, truestar.stacks) likewise need
    ``allow_stacks``, which only ``truestar check`` passes.
    """
    avail = {b.name: b for b in available_backends()}
    # On a ROCm build the device is still torch "cuda"; accept either spelling.
    if "rocm" in avail:
        avail.setdefault("cuda", avail["rocm"])
    if requested:
        chosen, missing = [], []
        reasons: List[str] = []
        for name in requested:
            key = name.strip().lower()
            if key == "webgpu" or key.startswith(("webgpu:", "webgpu+")):
                # Opt in only, never in the default set: it exports the model to ONNX.
                from .webgpu import probe

                b, reason = probe(name.strip())  # EP options after "+" keep their case
                if b is None:
                    missing.append(key)
                    reasons.append(f"{key}: {reason}")
                elif all(b is not c for c in chosen):
                    chosen.append(b)
                continue
            if key.startswith(("winml:", "webnn")) and (key == "webnn" or ":" in key):
                # Windows ML EPs and WebNN: opt in only, both export the model to ONNX
                # (truestar.winml, truestar.webnn); never in the default set.
                if key.startswith("winml:"):
                    from .winml import probe as other_probe
                else:
                    from .webnn import probe as other_probe
                b, reason = other_probe(name.strip())
                if b is None:
                    missing.append(key)
                    reasons.append(f"{key}: {reason}")
                elif all(b is not c for c in chosen):
                    chosen.append(b)
                continue
            if key.startswith("sim:"):
                if not allow_sim:
                    raise BackendUnavailable(
                        f"simulated backend {key} is supported by truestar check and truestar "
                        "probe only; this command would run it as the plain CPU", [key])
                try:
                    sb = sim_backend(key)
                except KeyError as exc:
                    raise BackendUnavailable(str(exc.args[0]), [key]) from None
                if all(sb.name != c.name for c in chosen):
                    chosen.append(sb)
                continue
            if "+" in key or ":" in key or key == "metal":  # a compiler or runtime stack (truestar.stacks)
                if not allow_stacks:
                    raise BackendUnavailable(
                        f"compiler and runtime backend {key} is supported by truestar check only; "
                        "this command would run the model in eager PyTorch under its name", [key])
                from .stacks import probe

                sb, why = probe(key)
                if sb is None:
                    missing.append(key)
                    reasons.append(f"{key}: {why}")
                elif all(c.name != sb.name for c in chosen):
                    chosen.append(sb)
                continue
            if key in avail:
                if avail[key] not in chosen:
                    chosen.append(avail[key])
            else:
                missing.append(key)
        if missing:
            raise BackendUnavailable(
                "Requested backend(s) not available on this machine: "
                + ", ".join(missing)
                + (" (" + "; ".join(reasons) + ")" if reasons else "")
                + ". Available: "
                + ", ".join(sorted(set(b.name for b in avail.values())))
                + ("" if not reasons else ". " + " ".join(reasons)),
                missing,
            )
        return chosen, []
    chosen = []
    for b in avail.values():
        if all(b is not c for c in chosen):
            chosen.append(b)
    present = {b.name for b in chosen}
    skipped = []
    for name in ("cuda", "mps", "xpu"):
        if name == "cuda" and "rocm" in present:
            continue
        if name not in present:
            skipped.append((name, "not available on this machine"))
    return chosen, skipped
