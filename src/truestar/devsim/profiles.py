"""Device numerics profiles: what a chip does to float arithmetic, one cited rule at a time.

A profile is data. Each field changes how :mod:`truestar.devsim.mode` evaluates an ATen op, and each
field is backed by a :class:`Rule` that says where the behaviour comes from and how sure we are:

* ``documented``: the vendor's own documentation, source code or a specification says so;
* ``measured``: a public issue or paper measured it on the device (numbers quoted);
* ``hypothesis``: a reporter or paper proposed it as the cause, nobody confirmed it;
* ``guess``: no public source; we chose a behaviour and say so.

Fields, in the order the simulation applies them to one op:

``storage``     every float op reads and writes this dtype (None: the model's own dtype)
``low_ops``     ops whose float inputs are rounded to ``low_dtype`` (autocast style, for example
                Neuron's matmul to bf16); the output keeps ``storage`` or the model dtype
``acc``         ``"wide"``: sums inside matmul, convolution and reductions are exact (float64
                here, which stands for a float32 class accumulator) and round once at the end;
                ``"storage"``: the running sum is rounded to the storage dtype after every
                ``acc_block`` elements, in order (a float16 accumulator)
``tile``        with a wide accumulator: groups of ``tile`` products are summed and rounded to the
                storage dtype before they reach the accumulator (0: no such stage)
``ftz``         subnormal results flush to zero (sign kept)
``saturate``    overflow clamps to the largest finite value instead of becoming infinity
``nan_to_inf``  every NaN becomes +inf (inputs and results)
``decompose``   layer norm and variance run step by step, each step rounded to storage (as an
                ONNX graph without a fused LayerNormalization does); otherwise they are one fused op
                that rounds once
``overrides``   op formulations that differ from the textbook one (implemented in :mod:`.kernels`)
``mma``         a tensor core generation of :mod:`.tensorcore` (``"h100"``, ``"sm120"``, ...): mm,
                addmm, bmm, baddbmm, mv and dot run through that matrix unit's bit accurate inner
                product model, inputs rounded to ``mma_fmt`` first (docs/TENSOR_CORE_MODELS.md);
                ``"exact"`` is the ideal: the same input rounding, the exact sum rounded once
``mma_fmt``     the unit's input format: fp16, bf16, tf32, e4m3 or e5m2
``validated``   the unit model was compared bit for bit with the device by Truestar itself

The ``ideal`` of a profile keeps its storage and per op precision and drops everything else: wide
accumulation, round to nearest even, IEEE special values, textbook formulations. It is what a
correct implementation in the same precision computes, and ``truestar check`` uses it as the noise
floor for that profile, so a simulated backend is blamed only for what the device does beyond
correct low precision arithmetic.

Not modelled anywhere (said once here, repeated in docs/DEVICES.md): non tensor scalar arguments
(an ``eps``, a scale) are passed in float64 and not rounded to the storage dtype.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, List, Optional, Tuple

import torch

DOCUMENTED, MEASURED, HYPOTHESIS, GUESS = "documented", "measured", "hypothesis", "guess"

# The dtype whose tolerance floor applies to a tensor core input format. tf32 has float16's ten
# fraction bits (and float32's range), so float16 stands in for its precision.
_MMA_FLOOR = {"fp16": torch.float16, "bf16": torch.bfloat16, "tf32": torch.float16,
              "e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}

# ATen op names (overload packet, no namespace) that run on a matmul or convolution engine.
MATMUL_OPS = frozenset({"mm", "addmm", "bmm", "baddbmm", "mv", "dot", "convolution",
                        "_scaled_dot_product_flash_attention_for_cpu"})


@dataclass(frozen=True)
class Rule:
    what: str  # the behaviour, one sentence
    source: str  # url or reference
    status: str  # documented, measured, hypothesis, guess
    quote: str = ""  # the numbers or words the rule rests on

    def as_dict(self) -> Dict[str, str]:
        return {"what": self.what, "source": self.source, "status": self.status, "quote": self.quote}


@dataclass(frozen=True)
class Profile:
    name: str  # "sim:ane"
    description: str
    storage: Optional[torch.dtype] = None
    low_ops: FrozenSet[str] = frozenset()
    low_dtype: Optional[torch.dtype] = None
    acc: str = "wide"
    acc_block: int = 1
    tile: int = 0
    ftz: bool = False
    saturate: bool = False
    nan_to_inf: bool = False
    decompose: bool = False
    overrides: FrozenSet[str] = frozenset()
    rules: Tuple[Rule, ...] = ()
    # IR op names for emulate_precision (train/amp.py) that match low_ops
    ir_low_ops: FrozenSet[str] = frozenset()
    mma: Optional[str] = None
    mma_fmt: Optional[str] = None
    validated: bool = False

    def __post_init__(self):
        if self.acc not in ("wide", "storage"):
            raise ValueError(f"acc must be 'wide' or 'storage', got {self.acc!r}")
        if self.acc == "storage" and self.storage is None:
            raise ValueError("a storage accumulator needs a storage dtype")
        if self.low_ops and self.low_dtype is None:
            raise ValueError("low_ops needs low_dtype")
        if self.acc_block < 1 or self.tile < 0:
            raise ValueError("acc_block must be >= 1 and tile >= 0")
        if self.mma and self.mma_fmt not in _MMA_FLOOR:
            raise ValueError(f"mma needs mma_fmt in {sorted(_MMA_FLOOR)}, got {self.mma_fmt!r}")

    @property
    def floor_dtype(self) -> Optional[torch.dtype]:
        """The narrowest float dtype this profile computes in, for the tolerance floor."""
        cands = [d for d in (self.storage, self.low_dtype) if d is not None]
        if self.mma:
            cands.append(_MMA_FLOOR[self.mma_fmt])
        return min(cands, key=lambda d: torch.finfo(d).bits) if cands else None

    def ideal(self) -> "Profile":
        """Correct arithmetic in the same precision: the noise floor for this profile."""
        return replace(self, name=self.name + "/ideal", description="correct arithmetic in the "
                       "same precision as " + self.name, acc="wide", acc_block=1, tile=0,
                       ftz=False, saturate=False, nan_to_inf=False, decompose=False,
                       overrides=frozenset(), rules=(),
                       mma="exact" if self.mma else None)

    @property
    def guesses(self) -> List[Rule]:
        return [r for r in self.rules if r.status == GUESS]

    def as_dict(self) -> Dict:
        dt = lambda d: None if d is None else str(d).replace("torch.", "")
        return {"name": self.name, "description": self.description, "storage": dt(self.storage),
                "low_ops": sorted(self.low_ops), "low_dtype": dt(self.low_dtype), "acc": self.acc,
                "acc_block": self.acc_block, "tile": self.tile, "ftz": self.ftz,
                "saturate": self.saturate, "nan_to_inf": self.nan_to_inf,
                "decompose": self.decompose, "overrides": sorted(self.overrides),
                "mma": self.mma, "mma_fmt": self.mma_fmt, "validated": self.validated,
                "rules": [r.as_dict() for r in self.rules]}


F16, BF16 = torch.float16, torch.bfloat16

_COREML_TYPED = "https://apple.github.io/coremltools/docs-guides/source/typed-execution.html"
_ANE_PAPER = "https://arxiv.org/html/2606.22283v2"
_WGSL_CTS = "https://github.com/gpuweb/cts/blob/main/docs/fp_primer.md"
_NEURON = ("https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/appnotes/neuronx-cc/"
           "neuronx-cc-training-mixed-precision.html")

PROFILES: Dict[str, Profile] = {}


def _add(p: Profile) -> Profile:
    PROFILES[p.name] = p
    return p


_add(Profile(
    name="sim:ieee-f16",
    description="textbook float16: float16 storage, float32 class accumulation, round to nearest "
                "even, IEEE special values (what a correct float16 kernel computes)",
    storage=F16,
    rules=(Rule("float16 storage, wide accumulation, IEEE 754 binary16 rounding", "IEEE 754-2019",
                DOCUMENTED),),
))

_add(Profile(
    name="sim:ieee-bf16",
    description="textbook bfloat16: bfloat16 storage, float32 class accumulation, round to "
                "nearest even, IEEE special values",
    storage=BF16,
    rules=(Rule("bfloat16 storage, wide accumulation, round to nearest even", "IEEE 754-2019 "
                "(bfloat16 is binary32 with 16 fraction bits dropped)", DOCUMENTED),),
))

_add(Profile(
    name="sim:ane",
    description="Apple Neural Engine through Core ML: float16 datapath, float32 class "
                "accumulator fed by float16 rounded tiles of four, NaN becomes +inf, the measured "
                "softplus and logsumexp cliffs",
    storage=F16, acc="wide", tile=4, nan_to_inf=True,
    overrides=frozenset({"softplus", "mish", "logsumexp"}),
    rules=(
        Rule("ML programs are typed in float16 by default, and the Neural Engine is barred when "
             "a program is typed float32, so ANE runs are float16", _COREML_TYPED, DOCUMENTED,
             "By default, an ML program produced by the converter is typed in float 16 precision"),
        Rule("input and output ports round to float16; the accumulator is float32 class; the "
             "first reduction stage groups lanes into tiles of four, float16 rounded",
             _ANE_PAPER + " section 3.2", MEASURED,
             "The accumulator is wide, of fp32 class (M1/H13 measured)"),
        Rule("bias is added in the float16 datapath, after the accumulator is rounded",
             _ANE_PAPER + " section 3.1", MEASURED,
             "bias may be held as fp32, but it is added in the fp16 datapath"),
        Rule("NaN is coerced to +inf at the input boundary and the engine never produces NaN",
             _ANE_PAPER + " section 3.6", MEASURED, "The engine coerces a NaN to positive infinity"),
        Rule("overflow goes to infinity, not to the largest finite value",
             _ANE_PAPER + " chapter 3 summary", MEASURED,
             "A fill above 4094 therefore overflows silently to infinity"),
        Rule("softplus returns 0 once float16(x) > ln(2^15) = 10.3972 (cliff between x = 10.394 "
             "and 10.395); the mechanism is unknown", "https://github.com/apple/coremltools/issues/2687",
             MEASURED, "x=10.4077 ne_out=0.0000 fp32_ref=10.4077 (M3 Max, CPU_AND_NE)"),
        Rule("mish shows the same cliff (it contains softplus)",
             "https://github.com/apple/coremltools/issues/2687 (M5 comment)", MEASURED,
             "analogous cliff at x = 10.4077"),
        Rule("reduce_log_sum_exp sums exp(x) without the max shift; when the float16 sum "
             "overflows the output is 0", "https://github.com/apple/coremltools/issues/2690",
             MEASURED, "C = 32: cliff at x = 7.6313, reference 11.0971, NE output 0.0000 (the "
                       "missing max shift is the reporter's hypothesis)"),
        Rule("subnormal float16 values are kept (no flush to zero)", "no public source", GUESS),
    ),
))

_add(Profile(
    name="sim:webgpu-f16",
    description="WebGPU with `enable f16` (ONNX Runtime WebGPU, Transformers.js): float16 storage "
                "and float16 accumulation in vec4 steps, subnormals flushed",
    storage=F16, acc="storage", acc_block=4, ftz=True, decompose=True,
    rules=(
        Rule("f16 is IEEE binary16; WGSL does not prescribe a rounding mode (either neighbour is "
             "allowed); round to nearest even is simulated", _WGSL_CTS, DOCUMENTED,
             "WGSL does not prescribe a specific rounding mode when quantizing"),
        Rule("an implementation may replace a subnormal input or output with zero; flushing is "
             "simulated", _WGSL_CTS, DOCUMENTED,
             "an implementation may choose to replace it with zero"),
        Rule("near overflow may round to the edge value or to infinity; infinity is simulated",
             _WGSL_CTS, DOCUMENTED,
             "may choose to round it to the edge value or the appropriate infinity"),
        Rule("reductions and matmuls accumulate in f16 when the kernel is instantiated for f16",
             "https://github.com/microsoft/onnxruntime/issues/29130 (as quoted in the gap research; "
             "the issue returned 404 on 2026-11-24)", HYPOTHESIS,
             "enable f16 lets the Metal backend accumulate reductions in fp16"),
        Rule("accumulation advances four elements at a time (vec4 dot products)",
             "no public source", GUESS),
        Rule("layer norm runs as separate ops (mean, subtract, square, mean), each rounded to f16",
             "https://github.com/onnxsim/onnxsim/pull/1742 (the decomposed ONNX LayerNorm)", GUESS),
    ),
))

_add(Profile(
    name="sim:mali-fp16",
    description="Arm Mali GPU through ncnn Vulkan defaults: float16 storage and float16 "
                "arithmetic, the accumulator advancing one element at a time",
    storage=F16, acc="storage", acc_block=1, decompose=True,
    rules=(
        Rule("ncnn defaults: use_fp16_packed, use_fp16_storage and use_fp16_arithmetic are true",
             "https://github.com/Tencent/ncnn/blob/master/src/option.cpp", DOCUMENTED,
             "use_fp16_arithmetic = true;"),
        Rule("ncnn forces fp16 arithmetic on for Mali drivers that accept fp16 SPIR-V",
             "https://github.com/Tencent/ncnn/blob/master/src/gpu.cpp", DOCUMENTED,
             "force capability on as long as the driver accept spirv with fp16 arithmetic"),
        Rule("the inner product shader accumulates `afp sum` one element per loop iteration",
             "https://github.com/Tencent/ncnn/blob/master/src/layer/vulkan/shader/innerproduct.comp",
             DOCUMENTED,
             "sum += buffer_ld1(weight_data, w_offset + i) * buffer_ld1(bottom_blob_data, i);"),
        Rule("`afp` is float16 when fp16 arithmetic is on", "ncnn shader naming (sfp storage, afp "
             "arithmetic); the macro definition was not read", GUESS),
        Rule("layer norm runs step by step in float16", "follows from fp16 arithmetic; not read "
             "in the layernorm shader", GUESS),
        Rule("subnormal float16 values are kept", "no public source (Vulkan float controls for "
             "Mali-G720 not found)", GUESS),
    ),
))

_add(Profile(
    name="sim:neuron-bf16",
    description="AWS Neuron compiler default before SDK 2.27: float32 matmuls and convolutions "
                "cast to bfloat16, float32 accumulation, everything else float32",
    storage=None, low_ops=MATMUL_OPS, low_dtype=BF16, acc="wide",
    ir_low_ops=frozenset({"linear", "matmul", "conv2d", "attention"}),
    rules=(
        Rule("by default the compiler casts FP32 matrix multiplications to BF16 "
             "(--auto-cast matmult --auto-cast-type bf16)", _NEURON, DOCUMENTED,
             "By default, the Neuron Compiler will automatically cast FP32 matrix multiplication "
             "operations to BF16"),
        Rule("matmul accumulates in FP32", _NEURON, DOCUMENTED, "accumulations using FP32"),
        Rule("the default became --auto-cast=none at neuronx-cc 2.22 / Neuron SDK 2.27; this "
             "profile simulates the earlier default", "gap research section 3.6 (latest Neuron "
             "mixed precision page)", DOCUMENTED),
        Rule("convolutions count as matrix multiplications for the cast", "no public source read",
             GUESS),
        Rule("rounding to bfloat16 is round to nearest even (stochastic rounding off)",
             "no public source read", GUESS),
    ),
))

_ROCM_MI350 = ("https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/"
               "workload-optimization.html")
_AMD_TF32_EMU = "https://www.amd.com/en/blogs/2023/application-transparent-emulation-of-tensorfloat32.html"

_add(Profile(
    name="sim:mi350-tf32",
    description="AMD MI350 / MI355X (gfx950) asked for TF32: there is no TF32 hardware, the "
                "matmul runs as a BF16 emulation; modelled as BF16 inputs with float32 "
                "accumulation, so TF32 tolerances (u = 2^-11) do not apply (BF16 u = 2^-8)",
    storage=None, low_ops=MATMUL_OPS, low_dtype=BF16, acc="wide",
    ir_low_ops=frozenset({"linear", "matmul", "conv2d", "attention"}),
    rules=(
        Rule("TF32 moved from hardware to software emulation via BF16 on MI350X; the data type "
             "table lists TF32 as N/A (SW via BF16) for MI350X and MI355X", _ROCM_MI350,
             DOCUMENTED, "TF32 moved from hardware to software emulation via BF16 on MI350X"),
        Rule("the emulation rounds the matmul inputs to BF16 once and accumulates in FP32 (one "
             "BF16 product, as AMD's 2023 TF32 emulation on MI250X did); a split of each input "
             "into several BF16 terms would be more accurate and is not modelled",
             _AMD_TF32_EMU, HYPOTHESIS,
             "casts weights and activations to BF16 and accumulates the output in FP32 (MI250X)"),
        Rule("convolutions follow the matmul path", "no public source read", GUESS),
    ),
))

_add(Profile(
    name="sim:openvino-gpu-f16",
    description="OpenVINO GPU plugin with its default f16 inference precision: f32 inputs "
                "converted to f16, gemm computed in f16",
    storage=F16, acc="storage", acc_block=1,
    rules=(
        Rule("default GPU inference precision is f16 since 2023.0; f32 inputs are converted and "
             "the gemm computes in f16", "https://github.com/openvinotoolkit/openvino/issues/22613",
             DOCUMENTED, "[1,1,2049]x[1,2049,1] of ones: CPU 2049.0, GPU 2048.0 (maintainer: "
                         "default precision changed to fp16)"),
        Rule("the f16 accumulator advances one element at a time", "no public source", GUESS),
    ),
))

_add(Profile(
    name="sim:openvino-npu-f16",
    description="Intel NPU through OpenVINO (NPU compiler 8.3 and older): f16 compute, a wide "
                "accumulator whose f16 result saturates at 65504",
    storage=F16, acc="wide", saturate=True,
    rules=(
        Rule("ReduceL2 squares are summed in f16 and saturate at 65504, so the output clamps at "
             "255.9 = sqrt(65504)", "https://github.com/openvinotoolkit/openvino/issues/39703",
             MEASURED, "[1,384,1,28,28] over axis 1, true norm 400.0: NPU 255.9, max rel err 0.36 "
                       "(NPU3720 and NPU5010, compiler 8.3; fixed in compiler 8.5)"),
        Rule("saturation applies to every f16 result, not only to reductions",
             "generalised from openvino#39703", GUESS),
        Rule("the sum itself is wide: below the cliff the measured error (5.9e-4 at norm <= 256) "
             "matches correct f16 (4.6e-4 simulated), not an f16 accumulator advancing one "
             "element at a time (4.6e-3 simulated)",
             "https://github.com/openvinotoolkit/openvino/issues/39703 (inferred from its numbers)",
             MEASURED, "Max rel err, norm <= 256: 0.000587"),
    ),
))

_QAIHUB = ("Truestar measurement through Qualcomm AI Hub, battery v1 (docs/QAIHUB.md): Galaxy S26 "
           "(Hexagon v81, TFLite and QNN DLC) and Snapdragon X Elite (v73, QNN DLC and ONNX "
           "Runtime QNN), docs/DEVICES.md section 7")

_add(Profile(
    name="sim:hexagon",
    description="Qualcomm Hexagon NPU (HTP v73, v81) running a float model: float16 datapath, "
                "wide accumulator, GELU fused into the linear layer before it and computed with "
                "the device's own approximation, fitted to real Snapdragon outputs",
    storage=F16, acc="wide",
    overrides=frozenset({"gelu_hexagon", "fuse_linear_gelu"}),
    rules=(
        Rule("float models run in float16 on the NPU, with a float32 class accumulator: "
             "convolutions equal correct float16 (sim:ieee-f16) bit for bit", _QAIHUB, MEASURED,
             "conv taps 94% to 100% bitwise equal to sim:ieee-f16 on S26 and X Elite"),
        Rule("GELU reads the linear layer's unrounded accumulator (the two are fused)", _QAIHUB,
             MEASURED, "same curve: 89.4% of held out outputs bit equal from the accumulator, "
                       "72.5% from the float16 rounded input"),
        Rule("GELU is x * phi(x) rounded once, phi a piecewise fit of the device's normal CDF "
             "approximation (line below |x| = 0.6, degree 5 above, 0 below -3.02, 1 above 3.00)",
             _QAIHUB + "; fit: scripts/hexagon_sim_fit.py", MEASURED,
             "held out: rel L2 2.73e-4 to the device against 2.71e-3 for exact GELU; bit equal "
             "89.4% against 6.7%"),
        Rule("GELU whose input is not a linear layer's output (a convolution, a matmul of two "
             "activations) reads its float16 input, unfused", "not measured (the battery has "
             "GELU after linear layers only)", GUESS),
        Rule("GELU with approximate='tanh' is correct float16", "not measured", GUESS),
        Rule("softmax and attention are 1.7x to 2.0x further from float64 on the device than "
             "correct float16; not modelled here (they run as correct float16)", _QAIHUB,
             MEASURED, "enc.softmax and enc.attn_out, S26 and X Elite"),
        Rule("special values (NaN, infinity, subnormals, overflow) follow IEEE float16",
             "not measured", GUESS),
    ),
))

# ----------------------------------------------------------------------------- tensor cores
# Bit accurate matrix unit models (devsim/tensorcore.py, docs/TENSOR_CORE_MODELS.md). Only the
# sm_120 profiles were compared with a device by Truestar; every other generation is exactly as
# good as the published description it was written from.

_KM = "Khattak and Mikaitis, Accurate Models of NVIDIA Tensor Cores, arXiv 2512.07004 v4"
_FHMP = ("Fasi, Higham, Mikaitis and Pranesh, Numerical behavior of NVIDIA tensor cores, "
         "PeerJ Computer Science 7:e330, 2021")
_TS = ("Truestar measurement on an RTX 5070 Ti (sm_120): scripts/tc_probe_sm120.py, "
       "docs/data/tensor_core_sm120.jsonl")
_MM_ONLY = frozenset({"mm", "addmm", "bmm", "baddbmm", "mv", "dot"})

_TC_COMMON = (
    Rule("products are exact and enter the sum unnormalized, so a product in [2, 4) keeps its "
         "exponent ea + eb and loses no bit", _KM + " section 4.1.1", MEASURED,
         "1.5 x 1.5 gives d = 2.25 + 2^-22 where 1 x 2.25 gives 2.25"),
    Rule("terms are aligned to the largest exponent among the products and c; bits below the "
         "alignment width are truncated (sign magnitude), the aligned sum is exact", _KM +
         " section 4.1 and Table 3", MEASURED, "Align with truncation"),
    Rule("c is added together with the products of one group, not after them", _KM +
         " section 4.4.1", MEASURED, "c is summed together with the products"),
    Rule("a dot product longer than one group chains groups in K order, each group's rounded "
         "result becoming the next c, starting from zero; no split K, no fp32 promotion step",
         "the order of one warp's MMA instructions; a library may schedule differently (TS1097)",
         GUESS),
    Rule("a zero result is +0 whatever the signs of the addends", _TS, MEASURED,
         "c = -0 with products -0 gives +0; a negative sum truncated to zero gives +0"),
    Rule("addmm and baddbmm add the bias after the product, alpha * acc + beta * bias rounded "
         "to nearest even in fp32 (an epilogue, not the MMA accumulator)", "no public source",
         GUESS),
    Rule("inf and NaN inputs follow IEEE float64 arithmetic for the outputs they reach",
         "no public source for the units' special value rules", GUESS),
)


def _tc(gen: str, fmt: str, label: str, description: str, rules, validated: bool = False):
    low = {"fp16": F16, "bf16": BF16}.get(fmt)
    extra = ()
    if fmt == "tf32":
        extra = (Rule("float32 inputs are rounded to tf32 (10 fraction bits) to nearest even before "
                      "the unit", "no public source for cuBLAS's rounding of tf32 inputs", GUESS),)
    if fmt in ("e4m3", "e5m2"):
        extra = (Rule("each matmul operand is scaled per tensor by a power of two so its largest "
                      "magnitude fits the fp8 range, rounded to nearest even, and the output is "
                      "scaled back exactly (a common per tensor recipe; a real model brings its own "
                      "scales)", "Truestar choice, no device behaviour", GUESS),)
    return _add(Profile(
        name=f"sim:{label}", description=description, storage=None,
        low_ops=MATMUL_OPS - {"_scaled_dot_product_flash_attention_for_cpu"} if low else frozenset(),
        low_dtype=low, acc="wide", mma=gen, mma_fmt=fmt, validated=validated,
        ir_low_ops=frozenset({"linear", "matmul"}) if low else frozenset(),
        rules=tuple(rules) + _TC_COMMON + extra,
    ))


def _km(what: str, where: str, quote: str = "") -> Rule:
    return Rule(what, f"{_KM} {where}", MEASURED, quote)


_tc("v100", "fp16", "v100-fp16", "NVIDIA V100 tensor core, fp16 inputs, fp32 accumulator: groups "
    "of 4, alignment with 23 fraction bits, truncation (from the published model, unvalidated)", (
        _km("groups of 4 products and c are summed at once", "section 4.1.1", "N_FMA = 4"),
        _km("alignment keeps 2 integer and 23 fraction bits (no extra bit)", "Table 3", "(2,23)"),
        _km("the normalized sum is truncated to fp32", "Table 3", "Trunc"),
        Rule("V100 has no exponent floor in the alignment (bf16 does not exist there)", _KM +
             " Table 3", DOCUMENTED, "-"),
    ))
for _fmt, _g in (("fp16", 8), ("bf16", 8), ("tf32", 4)):
    _tc("a100", _fmt, f"a100-{_fmt}", f"NVIDIA A100 (also A2, A30) tensor core, {_fmt} inputs, fp32 "
        f"accumulator: groups of {_g}, 24 fraction bits, truncation, exponent floor -132 "
        "(published model, unvalidated)", (
            _km(f"groups of {_g} products and c are summed at once", "section 4.1.2 and Table 3",
                f"N_FMA = {_g}"),
            _km("alignment keeps 2 integer and 24 fraction bits (one extra bit)", "Table 3",
                "(2,24)"),
            _km("the normalized sum is truncated to fp32", "section 4.1.2", "truncation"),
            _km("the alignment exponent is at least -132 when every product is tinier",
                "section 4.1.2", "e_min^align = -132"),
        ))
_L4 = ("Truestar measurement on an NVIDIA L4 (Ada, sm_89, Modal cloud): scripts/tc_probe_sm120.py "
       "fitall, docs/data/tensor_core_l4.jsonl")
for _fmt, _g, _n in (("fp16", 8, "65,536"), ("bf16", 8, "81,920"), ("tf32", 4, "81,920")):
    _tc("ada", _fmt, f"ada-{_fmt}", f"NVIDIA Ada (L4, L40S, RTX 40 series) tensor core, {_fmt} "
        f"inputs, fp32 accumulator: groups of {_g}, 24 fraction bits, truncation, exponent floor "
        f"-132 (A100's unit; fitted bit for bit on an L4)", (
            Rule(f"groups of {_g} products and c, 2 integer and 24 fraction bits, sign magnitude "
                 "truncation, sum truncated to fp32" + (", alignment exponent at least -132"
                                                        if _fmt != "fp16" else ""), _L4, MEASURED,
                 f"the parameter fit's only exact set: {_n} of {_n} dot products through Triton "
                 "tl.dot"),
            _km("Ada's 16 bit and tf32 units are identical to A100's", "sections 4.1.4, 4.1.5",
                "identical to that of the A100"),
        ), validated=True)
_tc("ada", "e4m3", "ada-fp8", "NVIDIA Ada (L40S, RTX 1000) fp8 tensor core through mma.sync "
    "(QMMA): groups of 16, only 13 fraction bits, truncation (published model; NOT reproduced on "
    "an L4 through Triton, see docs/TENSOR_CORE_MODELS.md)", (
        Rule("the paper's Ada fp8 model does not reproduce Triton tl.dot on an L4: its best fit is "
             "this model at 52,880 of 65,536 E4M3 and 55,801 of 65,536 E5M2 dot products; an fp16 "
             "unit path (groups of 8) fits worse. Whether Triton's code path or the model is "
             "responsible is open", _L4, MEASURED, "52880/65536 E4M3, 55801/65536 E5M2"),
        _km("groups of 16 fp8 products and c are summed at once", "Table 3", "N_FMA 16"),
        _km("alignment keeps 13 fraction bits: an fp32 output with 13 useful bits per group",
            "section 4.1.4", "the accumulation takes place with 13 fractional bits"),
        _km("the normalized sum is truncated", "Table 3", "Trunc"),
    ))
for _fmt, _g in (("fp16", 16), ("bf16", 16), ("tf32", 8)):
    _tc("h100", _fmt, f"h100-{_fmt}", f"NVIDIA H100/H200 tensor core, {_fmt} inputs, fp32 "
        f"accumulator: groups of {_g}, 25 fraction bits, truncation, exponent floor -133 "
        "(published model, unvalidated)", (
            _km(f"groups of {_g} products and c are summed at once", "section 4.1.6",
                f"N_FMA = {_g}"),
            _km("alignment keeps 2 integer and 25 fraction bits (two extra bits)",
                "section 4.1.6", "exactly n_eab = 2 extra alignment bits"),
            _km("the normalized sum is truncated to fp32", "section 4.1.6",
                "results are truncated rather than rounded"),
            _km("the alignment exponent is at least -133", "section 4.1.6",
                "e_min^align = -133"),
        ))
_tc("h100", "e4m3", "h100-fp8", "NVIDIA H100/H200 fp8 tensor core through wgmma (QGMMA), the "
    "path cuBLAS and CUTLASS fp8 GEMMs use: groups of 32 with only 13 fraction bits, truncation "
    "(published model, unvalidated)", (
        _km("groups of 32 fp8 products and c are summed at once", "section 4.1.6",
            "N_FMA = 32"),
        _km("alignment keeps 13 fraction bits (the narrow Hopper fp8 accumulator)",
            "section 4.1.6", "n_eab = -10, i.e., 13 fractional bits"),
        _km("the normalized sum is truncated", "Table 3", "Trunc"),
        Rule("libraries that promote the fp8 accumulator to fp32 every few K steps (cuBLAS "
             "without fast accumulation) are not modelled; this is the raw instruction chain",
             "no public source for each library's promotion interval", GUESS),
    ))
_tc("h100-mma", "e4m3", "h100-fp8-mma", "NVIDIA H100/H200 (and B200) fp8 through mma.sync: "
    "converted to fp16, two interleaved halves on the fp16 unit, c added late with round to "
    "nearest even (published model, unvalidated)", (
        _km("fp8 inputs are converted to fp16 and run on the fp16 unit (HMMA)",
            "section 4.1.6", "fp8 inputs are first converted to fp16"),
        _km("the 32 products are split into two halves by alternating pairs (1, 2, 5, 6, ... and "
            "3, 4, 7, 8, ...)", "section 4.1.6", "distributed across two tensor core "
            "invocations by alternating pairs of elements"),
        _km("c is added to the normalized sum of products with round to nearest even",
            "section 4.1.6", "This late addition of c is only present in this case"),
        Rule("the half holding products 1 and 2 runs first, from zero, and the second half "
             "continues from its truncated result", _KM + " Fig. 5c (order not stated in the "
             "text)", HYPOTHESIS),
    ))
for _fmt, _g in (("fp16", 16), ("bf16", 16), ("tf32", 8)):
    _tc("b200", _fmt, f"b200-{_fmt}", f"NVIDIA B200 tensor core, {_fmt} inputs, fp32 accumulator: "
        "the same unit as H100 (published model, unvalidated)", (
            _km("B200's 16 and 19 bit paths behave as H100's", "section 4.1.7",
                "identical numerical behaviour to the H100/H200 models"),
            _km(f"groups of {_g}, 2 extra alignment bits (25 fraction bits)", "section 4.1.7",
                "the accumulator provides 2 extra bits"),
            _km("the normalized sum is truncated; exponent floor -133", "Table 3", "-133, Trunc"),
        ))
_tc("b200", "e4m3", "b200-fp8", "NVIDIA B200 fp8 tensor core through tcgen05 (UTCQMMA): groups "
    "of 32 with 25 fraction bits, truncation (published model, unvalidated)", (
        _km("groups of 32 fp8 products and c are summed at once", "section 4.1.7",
            "N_FMA = 32 with 25 fractional bits"),
        _km("the normalized sum is truncated", "Table 3", "Trunc"),
    ))
_SM120 = ("NVIDIA Blackwell sm_120 (RTX 50 series, RTX PRO 6000) measured on an RTX 5070 Ti")
for _fmt, _g, _what in (("fp16", 16, "fp16"), ("bf16", 16, "bf16"), ("tf32", 8, "tf32"),
                        ("e4m3", 32, "fp8")):
    _floor = ("alignment exponent at least -133" if _fmt in ("bf16", "tf32") else
              "no exponent floor reachable with these inputs")
    _tc("sm120", _fmt, f"sm120-{_what}", f"{_SM120}, {_what} inputs, fp32 accumulator: groups of "
        f"{_g}, 25 fraction bits, truncation (bit for bit against the device, "
        "docs/TENSOR_CORE_MODELS.md)", (
            Rule(f"groups of {_g} products and c are summed at once", _TS, MEASURED,
                 "the parameter fit's only exact set; position probes for fp16"),
            Rule("alignment keeps 2 integer and 25 fraction bits, truncation toward zero (sign "
                 "magnitude), sum truncated to fp32", _TS, MEASURED,
                 "fraction bits 24 and 26 and two's complement cuts all mismatch"),
            Rule(_floor, _TS, MEASURED if _fmt in ("bf16", "tf32") else DOCUMENTED,
                 "-133 fits; -132, -134 and none mismatch" if _fmt in ("bf16", "tf32")
                 else "fp16 and fp8 products never fall below 2^-133"),
            Rule("the same parameters as the paper's RTX PRO 6000 and B200 models", _KM +
                 " Table 3", MEASURED, "H100/H200/B200/RTX PRO (2,25), Trunc"),
        ), validated=True)


def tensor_core_profiles() -> List[Profile]:
    """Every sim: profile backed by a matrix unit model."""
    return [p for p in PROFILES.values() if p.mma]


def get_profile(name: str) -> Profile:
    key = name.strip().lower()
    if not key.startswith("sim:"):
        key = "sim:" + key
    if key not in PROFILES:
        raise KeyError(f"unknown simulated backend {name!r}; known: {', '.join(sorted(PROFILES))}")
    return PROFILES[key]
