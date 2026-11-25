"""The FP8 variant trap: the same 8 bits mean different numbers on different GPUs.

There are two families of FP8 in use. The OCP formats (OFP8: E4M3 "fn" with max 448 and a negative
zero, E5M2 with infinities) are what NVIDIA GPUs and AMD MI350 (gfx950) compute with. The "fnuz"
formats (finite, no negative zero: E4M3FNUZ with exponent bias 8 and max 240, E5M2FNUZ with bias
16) are what AMD MI300 class GPUs (gfx942) compute with. A normal E4M3 code decodes to exactly
half its OCP value under FNUZ, and 0x80 is negative zero in one family and the only NaN in the
other. An FP8 checkpoint made for one family and run on the other is off by 2x with no error.

This module names the variant of an FP8 artifact from its metadata (dtype names of torch, ONNX,
MLIR, ml_dtypes, safetensors, Triton and HIP), infers it from the bit patterns when the name does
not say, and reports a mismatch with the target GPU's native variant:

* TS1000 `fp8_variant_mismatch`: the artifact's variant is not the target GPU's.
* TS1001 `fp8_variant_ambiguous_name`: the name does not say which variant ("e4m3", "fp8").
* TS1002 `fp8_variant_inferred`: no usable name; the variant was inferred from the bits.
* TS1003 `fp8_variant_bits_conflict`: the name says one variant, the bits look like the other.
* TS1004 `fp8_variant_undetermined`: neither the name nor the bits decide.
* TS1005 `fp8_target_unknown`: the target's native variant is not known here; nothing compared.

Bit evidence (E4M3): 0x7F and 0xFF are NaN in OCP E4M3 but +-240, the largest values, in FNUZ, so
a scaled FNUZ tensor (amax maps to the max code) has them and an OCP one almost never does; 0x80
is a negative zero in OCP (any tiny negative value that rounds to zero) and NaN in FNUZ. For E5M2
the codes with all exponent bits set are Inf or NaN in OCP and finite in FNUZ.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch

from ..codes import Issue
from .formats import E4M3, E4M3B11FNUZ, E4M3FNUZ, E4M3IEEE, E5M2, E5M2FNUZ, Codec

OCP, FNUZ = "ocp", "fnuz"

# variant: (family, element kind, codec)
VARIANTS: Dict[str, Tuple[str, str, Codec]] = {
    "e4m3fn": (OCP, "e4m3", E4M3),
    "e5m2": (OCP, "e5m2", E5M2),
    "e4m3fnuz": (FNUZ, "e4m3", E4M3FNUZ),
    "e5m2fnuz": (FNUZ, "e5m2", E5M2FNUZ),
    "e4m3b11fnuz": ("other", "e4m3", E4M3B11FNUZ),
    "e4m3ieee": ("other", "e4m3", E4M3IEEE),
}

# exact dtype names (lower case, spaces and the "torch." / "tl." prefixes removed) -> variant
NAMES: Dict[str, str] = {
    # torch
    "float8_e4m3fn": "e4m3fn", "float8_e5m2": "e5m2",
    "float8_e4m3fnuz": "e4m3fnuz", "float8_e5m2fnuz": "e5m2fnuz",
    # ONNX TensorProto
    "float8e4m3fn": "e4m3fn", "float8e5m2": "e5m2",
    "float8e4m3fnuz": "e4m3fnuz", "float8e5m2fnuz": "e5m2fnuz",
    # MLIR builtin types
    "f8e4m3fn": "e4m3fn", "f8e5m2": "e5m2", "f8e4m3fnuz": "e4m3fnuz", "f8e5m2fnuz": "e5m2fnuz",
    "f8e4m3b11fnuz": "e4m3b11fnuz", "f8e4m3": "e4m3ieee",
    # ml_dtypes (JAX): float8_e4m3 is the IEEE style one with infinities, max 240
    "float8_e4m3": "e4m3ieee", "float8_e4m3b11fnuz": "e4m3b11fnuz",
    # safetensors
    "f8_e4m3": "e4m3fn", "f8_e5m2": "e5m2", "f8_e4m3fnuz": "e4m3fnuz", "f8_e5m2fnuz": "e5m2fnuz",
    # Triton
    "fp8e4nv": "e4m3fn", "fp8e5": "e5m2", "fp8e4b8": "e4m3fnuz", "fp8e5b16": "e5m2fnuz",
    # HIP
    "__hip_fp8_e4m3": "e4m3fn", "__hip_fp8_e5m2": "e5m2",
    "__hip_fp8_e4m3_fnuz": "e4m3fnuz", "__hip_fp8_e5m2_fnuz": "e5m2fnuz",
}

# names that are FP8 but do not say which variant
AMBIGUOUS = re.compile(r"^(fp8|float8|f8)?[_ ]?(e4m3|e5m2)?$|^(fp8|float8)(_e4m3|_e5m2|e4m3|e5m2)$")

# target GPU -> native FP8 family. Sources: AMD ROCm MI300/MI350 workload optimization page
# ("MI300X uses the FNUZ FP8 variant, while MI350X uses the OCP FP8 variant"); NVIDIA FP8 on
# sm_89 and later is OCP E4M3/E5M2.
TARGETS: Dict[str, str] = {
    "gfx940": FNUZ, "gfx941": FNUZ, "gfx942": FNUZ, "mi300": FNUZ, "mi300x": FNUZ,
    "mi300a": FNUZ, "mi325x": FNUZ,
    "gfx950": OCP, "mi350": OCP, "mi350x": OCP, "mi355x": OCP,
}


def _norm(name) -> str:
    s = str(name).strip()
    for p in ("torch.", "tl.", "triton.language.", "ml_dtypes.", "np.", "jnp."):
        if s.lower().startswith(p):
            s = s[len(p):]
    return s.lower().replace(" ", "")


def variant_of_name(name) -> Optional[str]:
    """The variant an exact dtype name means, or None when the name does not say."""
    return NAMES.get(_norm(name))


def is_ambiguous_name(name) -> bool:
    s = _norm(name)
    return s not in NAMES and bool(AMBIGUOUS.match(s)) and s != ""


def target_family(target: str) -> Optional[str]:
    """The native FP8 family of a GPU ("gfx942", "MI300X", "sm_90", "cuda", ...), or None."""
    t = _norm(target).replace("amd", "").replace("instinct", "").strip("_-")
    if t in TARGETS:
        return TARGETS[t]
    m = re.match(r"^(sm_?|compute_?)(\d+)", t)
    if m:
        return OCP if int(m.group(2)) >= 89 else None
    if t in ("cuda", "nvidia"):
        return OCP
    return None


@dataclass
class BitEvidence:
    """Counts of the codes that tell the families apart."""

    kind: str                  # e4m3 or e5m2
    n: int
    ocp_signs: int             # codes valid only as OCP (0x80: negative zero)
    fnuz_signs: int            # codes valid only as FNUZ (E4M3 0x7F / 0xFF; E5M2 exponent 11111)

    @property
    def guess(self) -> Optional[str]:
        if self.fnuz_signs and not self.ocp_signs:
            return FNUZ
        if self.ocp_signs and not self.fnuz_signs:
            return OCP
        return None


def bit_evidence(codes: torch.Tensor, kind: str) -> BitEvidence:
    c = codes.reshape(-1)
    c = c.view(torch.uint8) if c.dtype.is_floating_point or c.dtype == torch.int8 else c
    c = c.long() & 0xFF
    ocp = int((c == 0x80).sum())
    if kind == "e4m3":
        fnuz = int(((c & 0x7F) == 0x7F).sum())
    elif kind == "e5m2":
        fnuz = int(((c & 0x7C) == 0x7C).sum())
    else:
        raise ValueError(f"bit evidence for e4m3 or e5m2, got {kind!r}")
    return BitEvidence(kind, c.numel(), ocp, fnuz)


def _kind_of(name) -> Optional[str]:
    s = _norm(name)
    if "e4m3" in s or "e4nv" in s or "e4b" in s:
        return "e4m3"
    if "e5m2" in s or "fp8e5" in s:
        return "e5m2"
    return None


def _ratio_note(kind: str, have: str, want: str) -> str:
    if kind == "e4m3" and {have, want} == {OCP, FNUZ}:
        return ("a normal E4M3 code decodes to half its OCP value under FNUZ (bias 8 versus 7), "
                "and 0x80 is negative zero in OCP but NaN in FNUZ")
    if kind == "e5m2" and {have, want} == {OCP, FNUZ}:
        return ("a normal E5M2 code decodes to half its OCP value under FNUZ (bias 16 versus "
                "15); OCP Inf and NaN codes are finite FNUZ values")
    return "the two formats give the same bits different values"


@dataclass
class VariantReport:
    tensor: str
    name: str
    variant: Optional[str]          # from the name
    family: Optional[str]           # resolved family (name, else bits)
    evidence: Optional[BitEvidence]
    issues: List[Issue] = field(default_factory=list)


def check_tensor(name, codes: Optional[torch.Tensor] = None, target: Optional[str] = None,
                 tensor: str = "") -> VariantReport:
    """Resolve one FP8 tensor's variant from its dtype name and (optionally) its codes, and
    compare it with the target GPU's native variant."""
    label = tensor or "tensor"
    var = variant_of_name(name)
    kind = VARIANTS[var][1] if var else _kind_of(name)
    ev = bit_evidence(codes, kind) if (codes is not None and kind in ("e4m3", "e5m2")) else None
    issues: List[Issue] = []
    fam = VARIANTS[var][0] if var else None
    if var is None:
        if is_ambiguous_name(name):
            issues.append(Issue("TS1001", f"{label}: the dtype name {name!r} does not say whether "
                                          "it is OCP FP8 or FNUZ FP8"))
        if ev is not None and ev.guess:
            fam = ev.guess
            issues.append(Issue("TS1002", f"{label}: inferred {fam.upper()} from the bits "
                                          f"({ev.fnuz_signs} codes valid only as FNUZ, "
                                          f"{ev.ocp_signs} only as OCP, of {ev.n})"))
        else:
            detail = "" if ev is None else (f" (bits: {ev.fnuz_signs} codes valid only as FNUZ, "
                                            f"{ev.ocp_signs} only as OCP, of {ev.n})")
            issues.append(Issue("TS1004", f"{label}: the FP8 variant of {name!r} could not be "
                                          f"determined{detail}"))
    elif ev is not None and fam in (OCP, FNUZ) and ev.guess and ev.guess != fam:
        issues.append(Issue("TS1003", f"{label}: named {name!r} ({fam.upper()}) but the bits look "
                                      f"{ev.guess.upper()} ({ev.fnuz_signs} codes valid only as "
                                      f"FNUZ, {ev.ocp_signs} only as OCP, of {ev.n})"))
    if target is not None:
        tf = target_family(target)
        if tf is None:
            issues.append(Issue("TS1005", f"{label}: the native FP8 variant of target {target!r} "
                                          "is not known; no comparison made"))
        elif fam is not None and fam != tf:
            issues.append(Issue("TS1000", f"{label}: {fam.upper()} FP8 data for {target} whose FP8 "
                                          f"is {tf.upper()}: " + _ratio_note(kind or "e4m3", fam, tf)))
    return VariantReport(label, str(name), var, fam, ev, issues)


def check_tensors(tensors: Mapping[str, torch.Tensor], target: Optional[str] = None) -> List[VariantReport]:
    """Every FP8 tensor of a state dict (torch float8 dtypes), checked against the target."""
    out = []
    for k, t in tensors.items():
        if isinstance(t, torch.Tensor) and t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2,
                                                       torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
            out.append(check_tensor(str(t.dtype), t, target, k))
    return out


def check_safetensors(path: str, target: Optional[str] = None,
                      sample_bytes: int = 1 << 20) -> List[VariantReport]:
    """Every 8 bit float tensor of a .safetensors file, from its header dtype names and the
    first `sample_bytes` of its data (read directly; no safetensors package needed)."""
    out = []
    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
        base = 8 + hlen
        for k, v in header.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            dt = str(v.get("dtype", ""))
            if not (dt.upper().startswith("F8") or variant_of_name(dt) or is_ambiguous_name(dt)):
                continue
            b0, b1 = v["data_offsets"]
            fh.seek(base + b0)
            raw = fh.read(min(b1 - b0, sample_bytes))
            codes = torch.frombuffer(bytearray(raw), dtype=torch.uint8) if raw else None
            out.append(check_tensor(dt, codes, target, k))
    return out


def check_onnx(path: str, target: Optional[str] = None) -> List[VariantReport]:
    """Every FP8 initializer of an ONNX model (needs the onnx package)."""
    import onnx

    m = onnx.load(path)
    names = {getattr(onnx.TensorProto, n): n for n in ("FLOAT8E4M3FN", "FLOAT8E4M3FNUZ",
                                                        "FLOAT8E5M2", "FLOAT8E5M2FNUZ")
             if hasattr(onnx.TensorProto, n)}
    out = []
    for init in m.graph.initializer:
        if init.data_type in names:
            raw = init.raw_data
            codes = torch.frombuffer(bytearray(raw), dtype=torch.uint8) if raw else None
            if codes is None and len(init.int32_data):
                codes = torch.tensor(list(init.int32_data), dtype=torch.int64)
            out.append(check_tensor(names[init.data_type], codes, target, init.name))
    return out


def issues(reports: Iterable[VariantReport]) -> List[Issue]:
    return [i for r in reports for i in r.issues]
