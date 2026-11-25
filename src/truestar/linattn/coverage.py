"""The hook that lets the importer's refusal by name (ir/refusals.py, TS941 and TS942) be lifted.

The coverage refresh refuses models with chunked linear attention (TS941) and chunked SSM scans
(TS942) by module class name. Two things must hold before a refusal may be lifted for a layer:

1. a verified kernel computes its recurrence: ``LAYERS`` says which rule and decay kind each
   refused class needs, and ``kernel_ready`` is True when ``linear_attention`` runs that
   combination (every one listed here runs, in Triton or in the chunked PyTorch form);
2. the importer lowers the layer's module to ``linear_attention``: a function registered with
   ``register_lowering(class_name, fn)`` that maps the module's projections, convolution, gates
   and norms onto the op. None is registered yet; that is the importer's half of the work.

``refusal_lifted(kind, class_name)`` is True only when both hold, so the refusal code can call it
unconditionally today and stays a refusal until a lowering lands:

    from truestar.linattn import refusal_lifted
    kind = classify(cls)
    if kind in ("linear", "ssm") and refusal_lifted(kind, cls):
        kind = None   # lowered to the verified kernels, not refused
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

# class name: (refusal kind, rule, decay kind, what the layer computes)
LAYERS: Dict[str, Tuple[str, str, str, str]] = {
    "Mamba2Mixer": ("ssm", "add", "scalar", "Mamba 2 SSD: q = C, k = B, v = dt x, g = dt A per head"),
    "FalconH1Mixer": ("ssm", "add", "scalar", "Mamba 2 SSD"),
    "NemotronHMamba2Mixer": ("ssm", "add", "scalar", "Mamba 2 SSD"),
    "BambaMixer": ("ssm", "add", "scalar", "Mamba 2 SSD"),
    "Zamba2MambaMixer": ("ssm", "add", "scalar", "Mamba 2 SSD"),
    "GraniteMoeHybridMambaLayer": ("ssm", "add", "scalar", "Mamba 2 SSD"),
    "Qwen3NextGatedDeltaNet": ("linear", "delta", "scalar", "Gated DeltaNet: scalar decay, delta rule"),
    "Qwen3_5GatedDeltaNet": ("linear", "delta", "scalar", "Gated DeltaNet"),
    "Qwen3_5MoeGatedDeltaNet": ("linear", "delta", "scalar", "Gated DeltaNet"),
    "KimiDeltaAttention": ("linear", "delta", "channel", "Kimi Delta Attention: per channel decay, delta rule"),
    "MiniMaxLightningAttention": ("linear", "add", "scalar", "lightning attention: fixed per head decay"),
}

_LOWERINGS: Dict[str, Callable] = {}


def register_lowering(class_name: str, fn: Callable) -> None:
    """Register the importer's lowering of a module class to ``linear_attention``."""
    if class_name not in LAYERS:
        raise KeyError(f"{class_name} is not a layer this package knows (truestar.linattn.coverage.LAYERS)")
    _LOWERINGS[class_name] = fn


def lowering(class_name: str) -> Optional[Callable]:
    return _LOWERINGS.get(class_name)


def kernel_ready(class_name: str) -> bool:
    """True when ``linear_attention`` computes this layer's recurrence (verified, any device)."""
    from .reference import RULES
    spec = LAYERS.get(class_name)
    return spec is not None and spec[1] in RULES and spec[2] in ("none", "scalar", "channel")


def refusal_lifted(kind: str, class_name: Optional[str] = None) -> bool:
    """True when a refused layer of this kind (``"linear"`` or ``"ssm"``) and class may be imported:
    its kernel is ready and the importer registered a lowering for the class (truestar.ir.hf_linattn
    registers Mamba 2 SSD and Gated DeltaNet when imported). False otherwise."""
    if kind not in ("linear", "ssm") or class_name is None:
        return False
    spec = LAYERS.get(class_name)
    return spec is not None and spec[0] == kind and kernel_ready(class_name) and class_name in _LOWERINGS
