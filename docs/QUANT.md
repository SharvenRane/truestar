# Quantized kernels with two verdicts

Author: Sharven Rane. First written 2026-11-24. Code: `src/truestar/quant/` (`formats.py`,
`kernels.py`, `verify.py`, `model.py`); tests in `tests/test_quant.py`; study in
`scripts/quant_study.py`; speed in `scripts/bench_quant.py`; what Triton runs natively in
`scripts/probe_quant_triton.py`.

Pain row: P25 (low precision kernel bugs hidden in format loss; research items GAP_COMPETITORS 5
and PAIN_RESEARCH_3 R4). Never introduce items 1 (no silent fallback) and 3 (no silent wrong
results of our own).

## The problem

Every quantization stack reports one number per model: an accuracy or perplexity table against
the float model. When an NVFP4 or FP8 model scores badly on one kernel path, nobody can tell
whether the format lost the accuracy (expected) or the kernel is wrong (a bug): the comparison
against the float model mixes both, and no exact reference of the format exists in the stack.

## The redesign: two verdicts that never mix

* **(a) Kernel verdict.** The kernel's output is compared with the exact float64 product of the
  operands it was given, decoded exactly from their format: the dequantized weight, and for W8A8
  and W4A4 the very codes and scales the run time activation quantizer wrote. Only the rounding
  the kernel is allowed to do may remain. Anything more is a kernel bug: `TS460` for the matmul,
  `TS461` for the activation quantizer. A failing kernel is then diagnosed from the outside (see
  below) down to the wrong scale index, group size or element order.
* **(b) Format verdict.** Computed from float64 references only: the float model against the same
  model with every quantized weight dequantized and every quantized activation rounded by the
  format's own rule. No kernel runs in it, so a kernel bug can neither inflate nor hide it.
  Reported as loss delta, perplexity ratio, KL, per layer output change and decision flips beyond
  near ties (`TS462` when flips remain beyond near ties, `TS463` otherwise).

The usual baseline (the quantized model against the float16 model) is reported next to both, to
show what it can and cannot see.

## Use

```python
from truestar.quant import formats, kernels, verify, model

qw = formats.quantize_weight(w, "int4:g128")          # w: [N, K] float, any device
y = kernels.quantized_matmul(x, qw)                   # x: [M, K] float16 or bfloat16 on CUDA
chk = verify.verify_kernel(x, qw)                     # verdict (a), diagnosis on failure
print(chk.ok, chk.ratio, chk.message())

qm = model.quantize_model(model16, "nvfp4")           # every nn.Linear except lm_head
res = model.compare(model16, ids, ["int8", "int4:g128", "fp8_e4m3:channel", "nvfp4", "mxfp4"])
```

`QuantLinear` checks its kernel on the first call of every new input shape and dtype (the same
schedule as every other Truestar kernel, docs/KERNEL_IR.md section 7) and raises
`QuantKernelMismatchError` with the code, the layer, the ratio to the bound and the diagnosis.

## Formats (`formats.py`)

A format is its value table. Decoding is a lookup; encoding is round to nearest, ties to the even
code, saturating at the largest finite value (the `cvt.rn.satfinite` rule and the OCP MX rule).
Nothing in the reference calls a PyTorch or Triton cast; the tests compare the tables with torch's
float8 casts instead (all 256 codes of E4M3 and E5M2, and every tie).

| spec | elements | scales | activations |
|---|---|---|---|
| `int8` | INT8, [-127, 127] | float32 per output channel, amax / 127 | float16 or bfloat16 |
| `int4:g32`, `int4:g64`, `int4:g128` | INT4, [-8, 7] | float16 per group of G inputs, amax / 7 | float16 or bfloat16 |
| `fp8_e4m3:channel`, `fp8_e4m3:tensor`, `fp8_e5m2:...` | FP8 | float32 per channel or per tensor, amax / max | FP8 per token, amax / max |
| `nvfp4` | E2M1 | E4M3 per 16 inputs, times a float32 tensor scale amax / (448 * 6) | the same, per call |
| `mxfp4` | E2M1 | E8M0 per 32 inputs, 2^(floor(log2 amax) - 2) | the same |

INT4 layout: within each group, byte j holds input j in its low nibble and input j + G/2 in its
high nibble, so the kernel's low nibbles are the group's first half and its high nibbles the
second, and two `tl.dot` calls need no interleave. FP4 keeps the `tl.dot_scaled` layout (input 2i
in the low nibble).

## Kernels (`kernels.py`)

What Triton 3.7.1 runs on this RTX 5070 Ti (sm_120), from `scripts/probe_quant_triton.py`, each
checked against an exact float64 product and by the MMA instruction in the PTX:

| mode | PTX instruction | native |
|---|---|---|
| `tl.dot` E4M3 x E4M3 | `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` | yes |
| `tl.dot` E5M2 x E5M2 | `mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32` | yes |
| `tl.dot_scaled` MXFP4 (E8M0, block 32) | `mma.sync.aligned.m16n8k64...kind::mxf4nvf4.block_scale.scale_vec::2X...ue8m0` | yes |
| `tl.dot_scaled` NVFP4 (E4M3 scales, block 16) | `mma.sync.aligned.m16n8k64...kind::mxf4nvf4.block_scale.scale_vec::4X...ue4m3` | yes |

So NVFP4 and MXFP4 both run on native block scaled FP4 tensor cores here; nothing is emulated.

* `w8`: weight only INT8. INT8 becomes float16 exactly by a magic number (q + 128 in the mantissa
  of 1024, minus 1152), `tl.dot` accumulates in float32, the channel scale multiplies once.
* `w4`: weight only INT4. Per group: two `tl.dot` calls on the exact integer weights (offset binary
  nibble in the mantissa of 1024, minus 1032; 128 and 136 for bfloat16), then one multiply by the
  group scale and one add. The weight is never rounded to float16 times scale, which is what
  kernels that dequantize before the dot do: an extra rounding per weight that the format does
  not have, and that verdict (a) would count against them. The magic number unpack made the
  kernel 1.85x faster at M = 2048 than integer conversions (1439 to 778 us, 4096 x 4096).
* `fp8`: W8A8 on the native FP8 MMA, float32 accumulate, per token activation scale times
  channel or tensor weight scale in the epilogue.
* `fp4`: W4A4 NVFP4 and MXFP4 through `tl.dot_scaled`; the NVFP4 tensor scales multiply in the
  epilogue.
* Activation quantizers: one program per row (FP8) or per row and 128 inputs (FP4). Division
  results are rounded to odd (truncate, then set the last bit when inexact) before the format's
  rounding, so each code is one correct rounding of the exact quotient; NVFP4 elements are
  encoded by comparing with the format's midpoints times the block scale (exact products), and
  MXFP4 scales come from the exponent bits. Result: every activation code equals the format's
  rule exactly. With a plain correctly rounded division instead, 35 of 262144 E4M3 codes and 72
  of 131072 NVFP4 codes differed from the rule (all at double rounding ties), and 7 tests fail.
* Tile shapes are fixed per M bucket (M up to 16, up to 64, above), chosen by a sweep over the
  shapes of the benchmark; no autotuning and no split K.
* No fallback: a CPU tensor, a float32 activation, an unknown format or a K that the FP4 tiles
  cannot cover raises `QuantKernelError` or `FormatError`.

## Verdict (a): the bound

Per output element, for a kernel that promises float32 accumulation of exact products (every
product of two codes here is exact in float32):

    |y - ref| <= u_out |ref| + 16 sqrt(K) 2^-24 S + floor_out,     S = sum_k |x_k| |w_k|

`u_out` is the output dtype's unit roundoff (2^-11 float16, 2^-8 bfloat16), `floor_out` half its
smallest subnormal. The sqrt(K) term is the probabilistic accumulation bound (Higham and Mary) with
lambda 16; the rigorous worst case `K 2^-23 S` is reported as `worst_case_ratio`. NaN or infinity
where the reference is finite always fails. Measured on correct kernels: the accumulation part
of the error stays under 0.036 of its term over every format and K of 512 and 4096; the whole
ratio peaks at 0.92 in the model study, where the output rounding itself dominates.

Why per element and not one relative tolerance: Truestar's kernel layer uses `max |out - ref| /
max |ref|` under 2e-3 times a log factor for float16. That rule would also catch the injected
bugs below against the dequantized reference; the per element bound is what makes the
statement exact (no element may deviate beyond its own rounding), and it is what the
diagnosis builds on.

## Diagnosis: naming the bug from outside the kernel

`verify.diagnose(qw, run)` reads the weight the kernel actually uses: input row k is the unit
vector e_k, so output row k is column k of the effective weight (one exact product, one scale,
one rounding; unit vectors quantize exactly in every format). Comparing it with the dequantized
weight gives the wrong channels and groups, and per wrong channel whether whole groups are off
by one factor (a wrong scale: which group's scale it got, or the granularity at which the factor
changes) or the right values sit at the wrong inputs (pairs swapped, or the two nibbles of each
byte swapped). No access to the kernel's code is needed.

Injected bugs (a `BUG` constexpr in each kernel, `kernels.BUGS`; used by tests only), on a
64 x 1024 x 128 problem:

| kernel | injected bug | ratio to bound | diagnosis |
|---|---|---|---|
| int8 | scale of the neighbouring channel | 587 | every input of the channel off by one factor: a wrong scale |
| int4:g128 | scale of the next group | 1480 | whole groups off; group 0 got the scale of group 1 |
| int4:g128 | last group reuses the previous scale (off by one at the boundary) | 765 | group 7 got the scale of group 6 |
| int4:g128 | groups of 64 assumed (wrong group boundary) | 2180 | the factor is constant over 64 inputs but the format's groups hold 128 |
| int4:g32 | low and high nibbles swapped | 6190 | the first and second half of each group swapped: nibbles in the wrong order |
| fp8_e4m3:channel | tensor scale used where channel scales were promised | 459 | every input of the channel off by one factor |
| fp8_e4m3:channel | float16 accumulator | inf | effective weights not finite: accumulator too narrow (FP8 codes times FP8 codes overflow float16) |
| nvfp4 | scale of the next block | 2230 | block 0 got the scale of block 1 |
| nvfp4 | scales read as if each covered 32 inputs | 2040 | whole blocks off: a wrong scale index |
| nvfp4 | the two codes of each byte swapped | 5830 | neighbouring inputs swapped in pairs |
| mxfp4 | scale of the next block | 3300 | a wrong scale index |

Format loss of the same layer (verdict b) is identical with and without the bug, by construction
and asserted in the tests.

## Results: a trained small Llama (scripts/quant_study.py)

Model: `truestar.rlkernels.llama` (RMSNorm, rotary, grouped query attention 8 on 4 heads, SwiGLU),
dim 512, 4 layers, hidden 1536, vocab 512, trained 800 steps (17 s) on a synthetic language built
in code (a Markov chain with a copy rule, entropy floor about 0.83 nats). Every linear layer of the
blocks is quantized (28 layers); the lm head stays float16. Held out: 16 sequences, 2048 scored
positions. float64 cross entropy 0.8128, float16 0.8128. Near ties: the float64 model's top 1
margin within 4 times the float16 model's own logit error at that position.

| format | weights | (b) delta CE | (b) ppl ratio | (b) KL | (b) flips (near ties) | (b) layer output change | (a) worst ratio | baseline vs fp16: rel, flips |
|---|---|---|---|---|---|---|---|---|
| int8 | 12.7 MB | +0.0001 | 1.0001 | 0.0000 | 0 (0) | 0.35% to 0.53% | 0.90 | 0.005, 0 |
| int4:g128 | 6.5 MB | -0.0019 | 0.9981 | 0.0018 | 13 (0) | 5.5% to 8.3% | 0.91 | 0.086, 13 |
| int4:g32 | 7.1 MB | +0.0012 | 1.0012 | 0.0014 | 14 (0) | 4.6% to 7.1% | 0.91 | 0.106, 14 |
| fp8_e4m3:channel | 12.7 MB | +0.0014 | 1.0014 | 0.0003 | 3 (0) | 1.8% to 2.8% | 0.92 | 0.038, 3 |
| fp8_e4m3:tensor | 12.6 MB | -0.0013 | 0.9987 | 0.0003 | 5 (0) | 1.8% to 2.8% | 0.92 | 0.034, 4 |
| fp8_e5m2:channel | 12.7 MB | +0.0004 | 1.0004 | 0.0011 | 8 (0) | 3.5% to 5.5% | 0.91 | 0.066, 12 |
| nvfp4 | 7.1 MB | -0.0039 | 0.9961 | 0.0042 | 20 (0) | 6.0% to 9.9% | 0.91 | 0.137, 17 |
| mxfp4 | 6.7 MB | -0.0064 | 0.9937 | 0.0073 | 25 (1) | 9.2% to 13.5% | 0.92 | 0.306, 27 |

The float16 model's linear weights are 25.2 MB. Negative loss deltas are within the noise of
2048 positions on a model near its entropy floor; KL and flips grow as expected from INT8 to
MXFP4. Every kernel passed verdict (a) on every layer; the kernel model against the float64
format reference flips 0 to 12 decisions, of which those beyond near ties come from float16
compute elsewhere in the model (attention, norms) feeding the activation quantizers.

**Injected bug in one layer, the whole model run again:**

| format, layer, bug | (a) | (b) | baseline vs fp16 without bug | with bug |
|---|---|---|---|---|
| int4:g128, blocks.2.mlp.down_proj, last group scale | TS460 at that layer only, ratio 2810, "group 11 got the scale of group 10" | unchanged | rel 0.0859, 13 flips | rel 0.0861, 14 flips |
| int4:g128, blocks.1.attn.o_proj, next group's scale | TS460, 6260, "group 0 got the scale of group 1" | unchanged | 0.0859, 13 | 0.0854, 10 |
| nvfp4, blocks.0.mlp.up_proj, next block's scale | TS460, 5150, "block 1 got the scale of block 2" | unchanged | 0.137, 17 | 0.161, 17 |
| fp8_e4m3:channel, blocks.3.attn.q_proj, tensor scale for channel | TS460, 1980, "off by the factor 1.065: a wrong scale" | unchanged | 0.038, 3 | 0.074, 9 |

The baseline moves by less than the spread between formats (and for the first two, by less
than its own run to run meaning: fewer flips with a bug than without), so it cannot tell these
bugs from format loss. Verdict (a) names the layer and the bug in every case.

## Speed (scripts/bench_quant.py)

RTX 5070 Ti shared with other jobs, torch 2.11, Triton 3.7.1, Llama 7B class shapes, GPU time per
call in microseconds: each call captured 20 times in a CUDA graph and replayed by
`triton.testing.do_bench`, median of 5 rounds. The cuBLAS baseline is the faster of graph and
plain launches. Quantized rows include the activation quantizer.

M = 1 (decode at batch 1):

| shape | cuBLAS fp16 | Truestar fp16 | int8 | int4:g128 | fp8 e4m3 | nvfp4 | mxfp4 | torch._scaled_mm fp8 (tensorwise, with quant) |
|---|---|---|---|---|---|---|---|---|
| 4096 x 4096 | 43 | 19 | 12 | 11 | 12 | 13 | 11 | 24 |
| 11008 x 4096 | 110 | 111 | 38 | 23 | 33 | 23 | 18 | 40 |
| 4096 x 11008 | 116 | 111 | 36 | 28 | 39 | 27 | 25 | 76 |

Linear layers of one 7B class block (4 x 4096 x 4096, 2 x 11008 x 4096, 1 x 4096 x 11008) at
batch 1: cuBLAS 508 us, Truestar fp16 409 us, int8 160 us, int4:g128 118 us, fp8 153 us, nvfp4
125 us, mxfp4 105 us, `torch._scaled_mm` FP8 252 us. INT4 weight only is 3.5x Truestar's float16
kernels and 4.3x cuBLAS on the linear layers of a decode step (the research target was 1.8x over
float16 at batch 1). This counts linear layers only, not attention or norms.

M = 16: 1.5x to 5.2x over cuBLAS for every format except NVFP4 on 4096 x 4096 (1.0x, its
quantizer launch dominates), 3.7x on the larger shapes. M = 256 and 2048 (compute bound; large M rows
moved by up to 1.8x between two runs because other jobs shared the GPU, ranges over both runs):

| format | M = 256 vs cuBLAS | M = 2048 vs cuBLAS |
|---|---|---|
| int8 | 0.76x to 0.96x | 0.56x to 1.07x |
| int4:g128 | 0.47x to 0.97x | 0.46x to 1.05x |
| fp8 e4m3 (with quant) | 1.54x to 1.62x | 1.12x to 2.06x |
| nvfp4 (with quant) | 1.72x to 3.15x | 2.28x to 4.05x |
| mxfp4 (with quant) | 1.77x to 3.07x | 2.34x to 3.84x |

Weight only INT8 and INT4 do not speed up large prefill (they dequantize on CUDA cores; that is
expected, they are a decode format). FP8 and FP4 use their own tensor cores and win at every M.

`torch._scaled_mm` on torch 2.11 with this GPU refuses row wise scales ("Rowwise scaling is not
currently supported on your device"), so the comparison uses tensorwise scales. Matmul only, our
FP8 kernel against it: 8 against 14 us (4096 x 4096, M = 1), 33 against 31 (11008 x 4096), 29
against 65 (4096 x 11008); at M = 2048 1.5x to 2.1x of cuBLAS for ours against 1.6x to 2.2x for
`_scaled_mm`, which is slightly faster at large M.

## Found while building

* **PyTorch on CUDA divides by a Python scalar as a multiply by the rounded reciprocal.** 37 of
  64 rows of `x.abs().amax(1).float() / 448` were one float32 ulp off the correctly rounded
  quotient (CPU torch and Triton's `div_rn` agree with exact arithmetic). The first quantizer
  check called a correct FP8 kernel wrong because of it. `formats.div_f32_rn` rounds correctly.
* **`torch.pow(2.0, e)` and `torch.ldexp` on CUDA are not exact for 35 of the 254 E8M0
  exponents** (2^-19 came back as 2^-19 (1 - 2^-53)). That made the reference call a correct MXFP4
  quantizer wrong on 505 elements. `formats.pow2` builds powers of two from exponent bits.
* Both were false alarms of the reference, found by its own checks before anything shipped.

## Not done or not verified

* No CLI flag yet (`truestar check --quant`); the API and the study script are the interface.
* Activations of FP8 are per token only (no static per tensor activation scale); INT4 is
  symmetric (no zero point); no GPTQ or AWQ style calibration, round to nearest only.
* Compared against torchao: not installed. Against `torch._scaled_mm`: tensorwise only (rowwise
  unsupported here).
* No published NVFP4 checkpoint decoded byte for byte (no downloads); the format tables are
  checked against torch's float8 casts and the GPU's own FP4 MMA, not against a vendor file.
* One GPU (sm_120). Other GPUs, ROCm and the Hopper 14 bit FP8 accumulator are not tested; on a
  GPU whose FP8 MMA accumulates in fewer bits the bound would flag it (by design), which is not
  yet seen on hardware.
* End to end decode tokens per second through a whole model with CUDA graphs is not measured;
  the decode numbers above are the linear layers of one block.
* Real pretrained models: none (no downloads); the study model is trained in code for 17 s.

## Low precision formats: MX, the FP8 variant trap, keyed stochastic rounding

Added 2026-11-24 (ledger row P50; research items 2.4, 2.3 and 2.1 of
<research>/LIVE_RESEARCH_2026_11.md). Code: `quant/mx.py`, `quant/mx_kernels.py`,
`quant/fp8variant.py`, `quant/sround.py`, the `sim:mi350-tf32` profile in `devsim/profiles.py`. Tests: `tests/test_lowp.py`. Numbers:
`scripts/lowp_study.py`, `results/lowp/study.json` (RTX 5070 Ti, torch 2.11, Triton 3.7.1).

### MX formats (`mx.py`)

MXFP8 (E4M3, E5M2) and MXFP4 (E2M1): one UE8M0 scale per 32 elements, every value its table
entry times 2^(code - 127) in float64. All 16 FP4 codes and the FP8 codes equal ml_dtypes' tables; encoding at every tie
equals ml_dtypes' round to nearest even. Three scale rules, because they decide which blocks clip:

| rule | X | source |
|---|---|---|
| `floor` | 2^(floor(log2 amax) - emax) | OCP MX v1.0, and v1.1 with expAdj = 0 |
| `floor_adj` | one power of two higher when the block max would round out of range | OCP MX v1.1 (Oct 2026) equation 4, expAdj |
| `ceil` | 2^ceil(log2(amax / max)) | arXiv 2506.08027 appendix A.1 (the MXFP8 training recipe) |

Measured on the linears of 4 of the 32 blocks of SmolLM2-360M (39.3M weights): relative MSE
against float64, and elements the clamp changed:

| format | floor | floor_adj | ceil |
|---|---|---|---|
| mxfp8_e4m3 | 9.23e-4, 196373 clipped | 7.06e-4, 0 | 7.06e-4, 0 |
| mxfp8_e5m2 | 2.94e-3, 115669 | 2.78e-3, 0 | 2.78e-3, 0 |
| mxfp4 | 1.38e-2, 293914 | **1.31e-2**, 0 | 1.44e-2, 0 |

The v1.0 rule clips 0.1 % to 0.75 % of elements; `floor_adj` never clips (asserted in the
tests) and has the lowest error in every row; `ceil` equals it on FP8 but is worse than
`floor` on E2M1, where rounding the scale up wastes part of a 2 bit exponent range.
`mx.check_quantizer` checks any MX quantizer's output against a rule (TS1008 for a scale, naming
the rules the scales do follow; TS1009 for an element).

Kernel (`mx_kernels.py`, one `tl.dot_scaled` per K tile, float32 accumulate): MXFP8 E4M3 x E4M3,
E5M2 x E5M2 and mixed pass verdict (a) (worst ratio 0.0014 at 4096 x 4096 x 4096); a scale moved
by one block fails it. At 4096 cubed: 1155 us (E4M3) against 1393 us for the float16 cuBLAS
matmul, with untuned tiles.

Not shipped: MXFP6 (E2M3, E3M2). No GPU kernel runs it (Triton 3.7.1's `tl.dot_scaled` raises
"NYI: lhs_format e2m3"), so the format, its FP6 tables and its stochastic rounding targets were
removed rather than offered as a reference with nothing to check.

### The FP8 variant trap (`fp8variant.py`)

OCP FP8 (E4M3 "fn", max 448, negative zero at 0x80; E5M2 with Inf) is what NVIDIA GPUs and AMD
MI350 (gfx950) compute; FNUZ FP8 (E4M3FNUZ bias 8, max 240, 0x80 the only NaN, no Inf; E5M2FNUZ)
is what MI300 class GPUs (gfx942) compute (AMD ROCm workload optimization page: "MI300X uses the
FNUZ FP8 variant, while MI350X uses the OCP FP8 variant"). Every normal E4M3 code is worth exactly
half under FNUZ (asserted for each), so a checkpoint for one family read by the other is off by 2x
with no error. `check_tensor`, `check_tensors`, `check_safetensors` (reads the header itself) and
`check_onnx` resolve each FP8 tensor's variant from its dtype name (torch, ONNX, MLIR, ml_dtypes,
safetensors, Triton `fp8e4nv` / `fp8e4b8`, HIP) or, when the name does not say, from the bits
(0x7F and 0xFF are the FNUZ maximum but NaN in OCP E4M3; 0x80 is a common OCP negative zero but
the FNUZ NaN), then compare with the target: TS1000 mismatch, TS1001 ambiguous name (`e4m3`,
`fp8`), TS1002 inferred, TS1003 the bits contradict the name, TS1004 undetermined, TS1005 target
unknown. ml_dtypes' `float8_e4m3` is a third E4M3 (IEEE style with Inf, max 240) and is named as
such, never as OCP.

`sim:mi350-tf32` (devsim): MI350 has no TF32 hardware; TF32 is software emulation through BF16
(documented). It is modelled as BF16 inputs with float32 accumulation, a hypothesis since the
algorithm is not published (a split into several BF16 terms would be more accurate). Its
tolerance floor is BF16 (u = 2^-8), not TF32's 2^-11.

Tried and not shipped: adaptive per block 4/6 scale lowered weight error but did not improve
model loss (WikiText, SmolLM2-360M). Measured: weight MSE lower on 224 of 224 linears (median
ratio 0.84), W4A4 cross entropy delta +0.272 against +0.238 for plain NVFP4, KL 0.281 against
0.277. Plain `nvfp4` is unchanged.

### Keyed stochastic rounding (`sround.py`)

Element i of a call at (key, site, step) rounds away from zero exactly when r < f * 2^32, with r
the top 32 bits of Truestar's keyed hash `mix(K ^ i)` (docs/RANDOM.md) and f = (|x| - lo) /
(hi - lo). It is integer arithmetic on the float32 bits of x, with an optional per element power
of two scale (an MX or NVFP4 block exponent) folded into the shifts, so nothing rounds before the
decision. Targets: E4M3, E5M2, E2M1, bfloat16, float16. Four implementations give the
same codes: `round_python` (exact rationals, one element at a time), `round_torch` on the CPU and
on CUDA, and `round_triton`. They are bitwise equal on 1M elements for E2M1, E4M3 and BF16 and, in
the tests, for all five targets with and without scales, including zeros, subnormals and
saturating values. `verify_codes` checks any SR kernel bit for bit (TS1006 names the first
differing element); `bias_test` checks unbiasedness statistically (TS1007).

Bias bound: E[round(x)] - x lies in [0, 2^-32 quantum). Measured over 400 draws of 4000 elements
per target: mean error between -1.8e-4 and 5.2e-4 quanta, pooled |z| at most 1.61, variance ratio
to f (1 - f) 0.9985 to 1.0006; a rounder skewed on 5 % of elements fails the test. Saturation
(|x| at or above the max) clamps and is excluded from the test, as it is biased by design.

Gain shown: adding 1e-3 ten thousand times to a bfloat16 accumulator at 1.0 stays at 1.0 with
round to nearest; keyed SR ends at 11.008 (mean of 1024, std 0.59; float64 11.0).

Cost: 16M float32 elements, Triton SR to BF16 codes 781 us and to E4M3 845 us against 139 and 92
us for torch's round to nearest casts (5.6x and 9.2x; the SR kernel writes int64 codes, 8 bytes
per element); the torch ops path takes 34.5 ms. Hardware SR (Blackwell `cvt.rs`, CDNA4 `sr`
converts) is not wrapped, so there is no explicitly nondeterministic fast mode yet.

### Not done

* SR writes int64 codes; no SR fused into a matmul epilogue or an optimizer step; no hardware SR.
* The FP8 variant check is an API; it is not wired into `truestar conform` or a CLI flag.
* No AMD GPU was used; the gfx942 and gfx950 facts come from AMD's documentation.
