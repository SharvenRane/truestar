"""Per backend precision manifest for `truestar check` (pain research R8).

The same float32 code runs with TF32 on NVIDIA Ampere and newer, with bf16 passes on TPUs and with
float16 accumulation on some paths, and flags do not say what a kernel really did (cuDNN picks its
own algorithm). Instead of reading flags, each backend runs three tiny crafted ops per op class
(linear, batched matmul, conv2d) in the model's dtype and the result says what happened:

* input significand bits: x = 1 + 2^-m against ones. A kernel that rounds its inputs to q bits loses
  the 2^-m term for m >= q, so the first m whose term is lost is the number of bits really used
  (24 for float32, 11 for TF32, 8 for bfloat16 passes).
* accumulator (16 bit dtypes only): 2^p followed by 2^(p-1) ones. Summed in the dtype each 1 is
  lost; summed wider the result is 1.5 * 2^p. A blocked reduction in the dtype can also give 1.5 *
  2^p, so "wider" means no narrow accumulation was seen on this probe, not a proof.
* subnormals: half the smallest normal value times one; 0 means the kernel flushes subnormals.

The flags that are supposed to control this are reported next to the measurement.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F

_K = 16
_BITS = {torch.float32: 24, torch.float16: 11, torch.bfloat16: 8, torch.float64: 53}


def _ops() -> Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    return {
        "linear": lambda x, w: F.linear(x.reshape(1, -1), w.reshape(1, -1)).reshape(()),
        "bmm": lambda x, w: torch.bmm(x.reshape(1, 1, -1), w.reshape(1, -1, 1)).reshape(()),
        "conv2d": lambda x, w: F.conv2d(x.reshape(1, -1, 1, 1), w.reshape(1, -1, 1, 1)).reshape(()),
    }


def _input_bits(fn, device: str, dtype: torch.dtype, p: int) -> int:
    w = torch.ones(_K, dtype=dtype, device=device)
    for m in range(1, p):
        x = torch.full((_K,), 1.0 + 2.0 ** -m, dtype=dtype, device=device)
        y = float(fn(x, w).cpu().double())
        if y <= _K:  # the 2^-m term was lost: inputs were rounded to m significand bits
            return m
    return p


def _accumulator(fn, device: str, dtype: torch.dtype, p: int) -> str:
    n = 2 ** (p - 1)
    x = torch.ones(n + 1, dtype=dtype, device=device)
    x[0] = 2.0 ** p
    w = torch.ones(n + 1, dtype=dtype, device=device)
    y = float(fn(x, w).cpu().double())
    if y == 1.5 * 2 ** p:
        return "wider than the dtype"
    if y == 2.0 ** p:
        return "dtype (ones lost)"
    return f"other ({y:g})"


def _subnormals(fn, device: str, dtype: torch.dtype) -> str:
    tiny = torch.finfo(dtype).tiny
    x = torch.zeros(_K, dtype=dtype, device=device)
    x[0] = tiny / 2
    w = torch.ones(_K, dtype=dtype, device=device)
    y = float(fn(x, w).cpu().double())
    return "kept" if y > 0 else "flushed"


def _label(bits: int, dtype: torch.dtype) -> str:
    p = _BITS.get(dtype)
    if bits == p:
        return str(dtype).replace("torch.", "")
    if dtype == torch.float32 and bits == 11:
        return "tf32"
    if bits == 8:
        return "bf16 inputs"
    return f"{bits} bit inputs"


def flags(device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"float32_matmul_precision": torch.get_float32_matmul_precision()}
    if device.startswith("cuda"):
        out.update({
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "fp16_reduced_precision_reduction": bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
            "bf16_reduced_precision_reduction": bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        })
    return out


def manifest(device: str, dtype: torch.dtype) -> Dict[str, Any]:
    """What each op class really computes on ``device`` in ``dtype``. Never raises: an op class that
    cannot run there is reported with its error."""
    p = _BITS.get(dtype)
    ops: Dict[str, Any] = {}
    if p is None:
        return {"dtype": str(dtype), "ops": {}, "flags": flags(device), "note": "not a float dtype"}
    for name, fn in _ops().items():
        try:
            with torch.no_grad():
                bits = _input_bits(fn, device, dtype, p)
                entry = {"input_bits": bits, "dtype_bits": p, "effective": _label(bits, dtype),
                         "subnormals": _subnormals(fn, device, dtype)}
                if p <= 11:
                    entry["accumulator"] = _accumulator(fn, device, dtype, p)
        except Exception as exc:  # e.g. an op without a kernel for this dtype on this device
            entry = {"error": f"{type(exc).__name__}: {exc}"}
        ops[name] = entry
    return {"dtype": str(dtype).replace("torch.", ""), "ops": ops, "flags": flags(device)}


def manifest_line(m: Dict[str, Any]) -> str:
    """One line for the text report."""
    parts = []
    for name, e in m.get("ops", {}).items():
        if "error" in e:
            parts.append(f"{name} failed")
            continue
        acc = f", accumulate {e['accumulator']}" if "accumulator" in e else ""
        parts.append(f"{name} {e['effective']} ({e['input_bits']} bits{acc}, subnormals {e['subnormals']})")
    return "precision: " + "; ".join(parts)
