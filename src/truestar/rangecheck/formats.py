"""Low precision number formats and the deployment targets `truestar range` emulates.

A format is described by its largest finite value, its smallest normal and subnormal values and its
significand bits. Rounding uses PyTorch's own dtypes (round to nearest even, overflow to infinity or,
for float8 e4m3fn which has no infinity, to NaN), so the emulation is not a second hand rounding
routine. Values go float64 to float32 to the format inside PyTorch's cast; that double rounding can
differ from a single rounding in the last bit for a value exactly between two float32 neighbours,
which does not change any range question.

``round_mantissa`` is the "range unlimited twin": the same significand bits with the exponent range of
float64. Comparing a target run with its twin separates range failures (overflow, underflow) from the
ordinary rounding of the format, which a deployment cannot avoid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class Format:
    name: str
    dtype: torch.dtype
    max: float           # largest finite value
    min_normal: float    # smallest positive normal value
    min_subnormal: float  # smallest positive value
    bits: int            # significand bits, implicit bit included (float16: 11)
    has_inf: bool

    def round(self, x: torch.Tensor) -> torch.Tensor:
        """Round a float64 tensor to this format; the result is float64 again."""
        return x.to(self.dtype).to(torch.float64)

    def round_mantissa(self, x: torch.Tensor) -> torch.Tensor:
        """Round to ``bits`` significand bits with no exponent limits (the twin run)."""
        m, e = torch.frexp(x)
        scale = float(2 ** self.bits)
        return torch.ldexp(torch.round(m * scale) / scale, e)

    def headroom_bits(self, peak: float) -> float:
        """log2(max / peak): bits left before overflow; negative when the peak overflows."""
        if peak <= 0 or not math.isfinite(peak):
            return math.inf if peak == 0 else -math.inf
        return math.log2(self.max / peak)


def _fmt(name: str, dtype: torch.dtype, has_inf: bool) -> Format:
    fi = torch.finfo(dtype)
    bits = {torch.float16: 11, torch.bfloat16: 8, torch.float8_e4m3fn: 4, torch.float8_e5m2: 3}[dtype]
    return Format(name, dtype, float(fi.max), float(fi.tiny), float(fi.smallest_normal) * 2.0 ** (1 - bits),
                  bits, has_inf)


FORMATS: Dict[str, Format] = {
    "fp16": _fmt("fp16", torch.float16, True),
    "bf16": _fmt("bf16", torch.bfloat16, True),
    "fp8-e4m3": _fmt("fp8-e4m3", torch.float8_e4m3fn, False),
    "fp8-e5m2": _fmt("fp8-e5m2", torch.float8_e5m2, True),
}


@dataclass(frozen=True)
class Target:
    """How a low precision runtime stores and computes.

    * ``fmt``: every float activation, weight and constant is stored in this format.
    * ``decomposed``: composite ops (layer norm, RMS norm, group norm, softmax, tanh gelu, vector
      norm) run as separate elementwise and reduction ops with every intermediate stored in ``fmt``,
      as in ONNX graphs exported below opset 17 and run unfused (onnxruntime#31626). When False they
      are fused kernels that keep their internal sums in float32, as the fused WebGPU, TensorRT and
      Core ML kernels do; their hidden intermediates are still reported, as information.
    * ``nan_to_inf``: NaN becomes +infinity (the Apple Neural Engine never produces NaN,
      arXiv 2606.22283).
    Accumulation inside matmul, conv and reductions is float32 on every preset (each op is computed
    exactly and its output rounded once); a float16 accumulator is not modelled.
    """

    name: str
    fmt: Format
    decomposed: bool = False
    nan_to_inf: bool = False
    description: str = ""


TARGETS: Dict[str, Target] = {
    "fp16": Target("fp16", FORMATS["fp16"], False, False,
                   "float16 storage, fused kernels with float32 accumulation (TensorRT fp16, Core ML GPU)"),
    "bf16": Target("bf16", FORMATS["bf16"], False, False, "bfloat16 storage, fused kernels"),
    "fp8-e4m3": Target("fp8-e4m3", FORMATS["fp8-e4m3"], False, False,
                       "unscaled float8 e4m3 storage (no per tensor scale; scaled FP8 is not modelled)"),
    "fp8-e5m2": Target("fp8-e5m2", FORMATS["fp8-e5m2"], False, False, "unscaled float8 e5m2 storage"),
    "ane": Target("ane", FORMATS["fp16"], False, True,
                  "Apple Neural Engine: float16 storage, NaN coerced to +infinity"),
    "webgpu-f16": Target("webgpu-f16", FORMATS["fp16"], True, False,
                         "WebGPU f16 running an unfused graph: decomposed norms, every edge float16"),
}


def get_target(name: str) -> Target:
    if name not in TARGETS:
        raise ValueError(f"unknown target {name!r}; choose one of {', '.join(TARGETS)}")
    return TARGETS[name]
