"""Requantization rules and the library conventions built from them.

A *rule* turns an integer accumulator and a real multiplier (input scale times weight scale over
output scale) into an integer before the output zero point is added. A *convention* says which
rule each op kind uses in one library build, plus how the library rounds a float to an integer
(Quantize) and an average pool sum to an integer.

Rules are data, not subclasses: each has a `kind` (q31 fixed point, q15 fixed point, float) and a
rounding function, so adding the next library is a table row. Sources for every row are listed in
docs/INTREF.md with the date they were read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from . import fixedpoint as fp

OP_KINDS = ("conv", "fully_connected", "matmul", "add", "mul", "avg_pool", "max_pool", "softmax",
            "requantize", "quantize")


@dataclass(frozen=True)
class Rule:
    """One way to compute round(acc * multiplier)."""

    name: str
    kind: str  # "q31", "q15" or "float"
    description: str
    fn: Callable  # (acc int64, m0 int64, shift int64) for q31/q15; (acc, real float64) for float
    documented: bool = True

    def requant(self, acc, real) -> np.ndarray:
        """round(acc * real) under this rule; `real` broadcasts against `acc` (per channel)."""
        acc = np.asarray(acc, dtype=np.int64)
        real = np.asarray(real, dtype=np.float64)
        if self.kind in ("q31", "q15"):
            m0, sh = fp.quantize_multiplier(real)
            return np.asarray(self.fn(acc, m0, sh), dtype=np.int64)
        return np.asarray(self.fn(acc, real), dtype=np.int64)


def _f32(rnd):
    def f(acc, real):
        prod = acc.astype(np.float32) * np.asarray(real, dtype=np.float32)
        return rnd(prod.astype(np.float64)).astype(np.int64)
    return f


def _f64(rnd):
    def f(acc, real):
        return rnd(acc.astype(np.float64) * real).astype(np.int64)
    return f


def _f16_multiplier(acc, real):
    m = np.asarray(real, dtype=np.float16).astype(np.float32)
    return fp.round_half_even((acc.astype(np.float32) * m).astype(np.float64)).astype(np.int64)


RULES: Dict[str, Rule] = {r.name: r for r in [
    Rule("q31_double", "q31",
         "gemmlowp SaturatingRoundingDoublingHighMul (ties up at bit 31) then RoundingDivideByPOT "
         "(ties away from zero): TFLite default build",
         fp.mbqm_double),
    Rule("q31_double_cmsis", "q31",
         "CMSIS-NN arm_nn_doubling_high_mult_no_sat (+2^30 then >> 31: ties up at bit 31) then "
         "arm_nn_divide_by_power_of_two (ties away): CMSIS-NN default build; equal to q31_double "
         "on every input with a non negative multiplier",
         fp.mbqm_double_cmsis),
    Rule("q31_single_up", "q31",
         "(x * M0 + 2^(t-1)) >> t, one rounding, ties toward plus infinity: TFLITE_SINGLE_ROUNDING "
         "and CMSIS_NN_USE_SINGLE_ROUNDING",
         fp.mbqm_single),
    Rule("q31_single_even", "q31", "x * M0 / 2^t rounded once, ties to even (exact)",
         lambda x, m, s: fp.mbqm_exact(x, m, s, "even")),
    Rule("q31_single_away", "q31", "x * M0 / 2^t rounded once, ties away from zero (exact)",
         lambda x, m, s: fp.mbqm_exact(x, m, s, "away")),
    Rule("q15_single_up", "q15",
         "multiplier reduced to 16 bits, one rounding half up: TFLite int64 MultiplyByQuantizedMultiplier "
         "and CMSIS-NN arm_nn_requantize_s64 (int16 activations)",
         fp.mbqm_q15),
    Rule("f32_even", "float",
         "float32(acc) * float32(multiplier), round half to even: ONNX QLinear spec, ONNX Runtime "
         "and XNNPACK fp32 requantization",
         _f32(fp.round_half_even)),
    Rule("f32_away", "float", "float32(acc) * float32(multiplier), round half away from zero",
         _f32(fp.round_half_away)),
    Rule("f64_even", "float", "float64 product, round half to even", _f64(fp.round_half_even)),
    Rule("f16_multiplier", "float",
         "multiplier rounded to float16, float32 product, round half to even (hypothesis for a "
         "half precision requantizer; not a documented library)",
         _f16_multiplier, documented=False),
]}


# ----------------------------------------------------------------------------- float to integer

def _q_div_away(x, scale):
    """TFLite reference Quantize: TfLiteRound(x / scale) (std::round, half away), float32."""
    v = (np.asarray(x, dtype=np.float32) / np.float32(scale)).astype(np.float64)
    return fp.round_half_away(v)


def _q_div_even(x, scale):
    """ONNX QuantizeLinear: x / scale rounded half to even, float32."""
    v = (np.asarray(x, dtype=np.float32) / np.float32(scale)).astype(np.float64)
    return fp.round_half_even(v)


def _q_fp16_recip(x, scale):
    """Hypothesis: input converted to float16 and multiplied by a float16 reciprocal of the scale
    in float16 arithmetic, then rounded half to even."""
    h = np.float16
    v = (np.asarray(x, dtype=np.float32).astype(h) * h(1.0 / float(scale))).astype(h)
    return fp.round_half_even(v.astype(np.float64))


QUANTIZERS: Dict[str, Callable] = {
    "div_away_f32": _q_div_away,
    "div_even_f32": _q_div_even,
    "fp16_reciprocal": _q_fp16_recip,
}


# ----------------------------------------------------------------------------- average pool sum

def pool_div_away(acc, count):
    """TFLite and CMSIS-NN avg pool: acc > 0 ? (acc + n/2) / n : (acc - n/2) / n, C division."""
    acc = np.asarray(acc, dtype=np.int64)
    n = np.asarray(count, dtype=np.int64)
    num = np.where(acc > 0, acc + n // 2, acc - n // 2)
    q = np.abs(num) // n
    return np.where(num >= 0, q, -q)


def pool_float_even(acc, count):
    """Average in float32, round half to even (ONNX Runtime QLinearAveragePool style)."""
    return fp.round_half_even((np.asarray(acc, np.float32) / np.asarray(count, np.float32))
                              .astype(np.float64)).astype(np.int64)


POOL_ROUNDING: Dict[str, Callable] = {"div_away": pool_div_away, "float_even": pool_float_even}


# ----------------------------------------------------------------------------- conventions

@dataclass(frozen=True)
class Convention:
    """Which rule a library build uses per op kind."""

    name: str
    description: str
    default_rule: str
    rules: Dict[str, str] = field(default_factory=dict)  # op kind -> rule, overriding the default
    int16_rule: Optional[str] = None  # rule for conv / fully connected with int16 activations
    add_structure: str = "tflite"  # "tflite" (left shift 20, three multipliers) or "float"
    quantizer: str = "div_away_f32"
    pool: str = "div_away"
    documented: bool = True
    source: str = ""

    def rule_for(self, op: str, int16: bool = False) -> Rule:
        if int16 and self.int16_rule and op in ("conv", "fully_connected", "matmul"):
            return RULES[self.int16_rule]
        return RULES[self.rules.get(op, self.default_rule)]


CONVENTIONS: Dict[str, Convention] = {c.name: c for c in [
    Convention("tflite_ref", "TFLite reference kernels, default build (double rounding)",
               "q31_double", int16_rule="q15_single_up",
               source="tensorflow/lite/kernels/internal/common.cc, reference/integer_ops"),
    Convention("tflite_single", "TFLite built with TFLITE_SINGLE_ROUNDING", "q31_single_up",
               int16_rule="q15_single_up", source="tensorflow/lite/kernels/internal/common.cc"),
    Convention("tflite_2_21_py", "TFLite 2.21 Python interpreter as reported in CMSIS-NN#214: "
               "single rounding for FULLY_CONNECTED, double rounding for MUL",
               "q31_double", rules={"fully_connected": "q31_single_up"},
               int16_rule="q15_single_up",
               source="github.com/ARM-software/CMSIS-NN/issues/214 (reported, not run here)"),
    Convention("cmsis_nn", "CMSIS-NN default build (double rounding)", "q31_double_cmsis",
               int16_rule="q15_single_up", source="CMSIS-NN Include/arm_nnsupportfunctions.h"),
    Convention("cmsis_nn_single", "CMSIS-NN built with CMSIS_NN_USE_SINGLE_ROUNDING",
               "q31_single_up", int16_rule="q15_single_up",
               source="CMSIS-NN Include/arm_nnsupportfunctions.h"),
    Convention("onnx_qlinear", "ONNX QLinear ops and QuantizeLinear: float32 scale, half to even",
               "f32_even", add_structure="float", quantizer="div_even_f32", pool="float_even",
               source="onnx.ai operator spec; measured against onnxruntime (docs/INTREF.md)"),
    Convention("exact_half_even", "x * M0 / 2^t rounded once, ties to even", "q31_single_even",
               add_structure="tflite", source="definition"),
    Convention("exact_half_away", "x * M0 / 2^t rounded once, ties away from zero",
               "q31_single_away", add_structure="tflite", source="definition"),
    Convention("npu_fp16_input", "hypothesis: input quantized in float16 (x times a float16 "
               "reciprocal scale), integer ops as ONNX QLinear", "f32_even", add_structure="float",
               quantizer="fp16_reciprocal", pool="float_even", documented=False,
               source="inferred from docs/NPU.md data, see docs/INTREF.md"),
]}

# Qualcomm Hexagon / QNN HTP requantization is not publicly documented (checked 2026-11-24); no
# convention is claimed for it. `npu_fp16_input` is a hypothesis tested against the saved S24 data.


def documented_conventions() -> List[str]:
    return [n for n, c in CONVENTIONS.items() if c.documented]


def describe() -> List[dict]:
    """JSON description of every rule and convention (`truestar intref conventions`)."""
    return [{"name": c.name, "description": c.description, "default_rule": c.default_rule,
             "rules": dict(c.rules), "int16_rule": c.int16_rule, "quantizer": c.quantizer,
             "pool": c.pool, "documented": c.documented, "source": c.source,
             "rule_descriptions": {r: RULES[r].description
                                   for r in {c.default_rule, *c.rules.values(),
                                             *( [c.int16_rule] if c.int16_rule else [])}}}
            for c in CONVENTIONS.values()]
