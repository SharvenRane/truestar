# Devices without the device: simulated numerics, size sweeps, special values

Three tools for the question "will this model give the same answer on that chip", usable before
anyone owns the chip. Pain ledger row P28 (docs/QUALITY_BAR.md). Code: `src/truestar/devsim/`.

| question | command |
|---|---|
| does the model break on a device's arithmetic | `truestar check model.py --backends cpu,sim:ane,sim:webgpu-f16` |
| at which size does it start to break | `truestar check model.py --backends ... --sweep n:512..8192` |
| how does each backend treat NaN, inf, subnormals, minus zero, casts | `truestar probe --specials` |
| which profiles exist, with every rule and its source | `truestar backends --sims` |

## 1. Simulated backends (`sim:*`)

A simulated backend is a profile of how a device does float arithmetic, applied on the CPU to
every ATen op the model runs. It plugs into `truestar check` like a real backend: per module
isolated replay, op drilldown, decision flips, repros and agent JSON all work unchanged. Nothing
runs on the device; every result carries TS520 so nobody mistakes it for a device run.

What a profile can say, and how it is evaluated (`devsim/mode.py`, `devsim/kernels.py`):

* **storage**: every float op reads and writes this dtype (float16 on the ANE, WebGPU f16, Mali,
  OpenVINO). Values are rounded on the way in and out; the op itself is computed in float64.
* **per op precision**: some ops read a lower dtype than the rest (AWS Neuron casts matmuls and
  convolutions to bfloat16 and keeps everything else in float32).
* **accumulation**: *wide* (a float32 class accumulator: the sum is exact here and rounds once),
  *tiled* (ANE: tiles of four products rounded to float16 before the wide accumulator) or
  *storage* (a float16 accumulator rounded after every block of `acc_block` products, in order).
  Matmuls, convolutions (by im2col for 1d and 2d), sums, means, norms, softmax, logsumexp, layer
  norm, variance and attention follow it.
* **special values**: flush subnormals to zero, saturate at the largest finite value instead of
  overflowing to infinity, NaN becomes +inf.
* **formulations**: an op the device computes differently from the textbook (the ANE's softplus and
  logsumexp), and composite ops that run step by step with every step rounded (an ONNX LayerNorm
  decomposed into mean, subtract, square, mean).

Every rule carries its source and one of four labels: **documented** (vendor docs, source or a
specification), **measured** (a public issue or paper with numbers), **hypothesis** (proposed by a
reporter, not confirmed), **guess** (no public source; we picked a behaviour). `truestar backends
--sims` prints them all as JSON.

| backend | what it simulates | evidence (label) |
|---|---|---|
| `sim:ieee-f16`, `sim:ieee-bf16` | textbook float16 or bfloat16: storage rounding, wide accumulator, IEEE specials | IEEE 754 (documented) |
| `sim:ane` | Apple Neural Engine via Core ML: float16 datapath, float32 class accumulator fed by float16 rounded tiles of four, bias added in float16, NaN becomes +inf, overflow to inf, softplus is 0 above ln(2^15), reduce_log_sum_exp without the max shift returns 0 when its float16 sum overflows | [coremltools typed execution](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html) (documented); [arXiv 2606.22283](https://arxiv.org/html/2606.22283v2) sections 3.1, 3.2, 3.6 (measured); [coremltools#2687](https://github.com/apple/coremltools/issues/2687), [#2690](https://github.com/apple/coremltools/issues/2690) (measured); subnormals kept (guess) |
| `sim:webgpu-f16` | WGSL with `enable f16`: float16 storage, subnormals flushed, a float16 accumulator in vec4 steps, layer norm step by step | [WebGPU CTS floating point primer](https://github.com/gpuweb/cts/blob/main/docs/fp_primer.md) (documented: flushing, rounding and overflow are the implementation's choice, we simulate one); float16 accumulation from onnxruntime#29130 as quoted in the gap research (hypothesis; the issue returned 404 on 2026-11-24); vec4 steps and decomposed layer norm (guess) |
| `sim:mali-fp16` | Arm Mali through ncnn's Vulkan defaults: float16 storage and arithmetic, the accumulator advancing one element per step | [ncnn option.cpp](https://github.com/Tencent/ncnn/blob/master/src/option.cpp) `use_fp16_arithmetic = true`, [gpu.cpp](https://github.com/Tencent/ncnn/blob/master/src/gpu.cpp) forcing fp16 arithmetic on Mali, [innerproduct.comp](https://github.com/Tencent/ncnn/blob/master/src/layer/vulkan/shader/innerproduct.comp) `afp sum` loop (documented); `afp` being float16 and subnormals kept (guess) |
| `sim:neuron-bf16` | AWS Neuron compiler default before SDK 2.27: float32 matmuls cast to bfloat16, float32 accumulation | [Neuron mixed precision 2.26](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/appnotes/neuronx-cc/neuronx-cc-training-mixed-precision.html) (documented); convolutions included and round to nearest even (guess) |
| `sim:mi350-tf32` | AMD MI350 / MI355X asked for TF32: no TF32 hardware, matmuls as BF16 inputs with float32 accumulation | [ROCm MI350 workload optimization](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html) (documented: TF32 is software emulation via BF16); one BF16 product per input pair (hypothesis, from AMD's 2023 MI250X TF32 emulation blog); convolutions included (guess) |
| `sim:openvino-gpu-f16` | OpenVINO GPU plugin default f16 inference precision, gemm in f16 | [openvino#22613](https://github.com/openvinotoolkit/openvino/issues/22613) maintainer answer (documented); accumulator one element per step (guess) |
| `sim:openvino-npu-f16` | Intel NPU, compiler 8.3 and older: f16 results that saturate at 65504, a wide accumulator | [openvino#39703](https://github.com/openvinotoolkit/openvino/issues/39703) (measured; the wide accumulator is inferred from its error below the cliff); saturation of every result (guess) |
| `sim:hexagon` | Qualcomm Hexagon NPU (v73, v81) running a float model: float16 storage, wide accumulator, GELU fused into the linear layer before it and computed with the device's own approximation | Truestar's own Snapdragon runs through Qualcomm AI Hub (measured, section 7); GELU after other ops, tanh GELU and special values (guess) |

**GPU tensor cores.** `sim:v100-fp16`, `sim:a100-bf16`, `sim:h100-fp8`, `sim:b200-fp16`,
`sim:sm120-fp16` and the rest run mm, addmm, bmm, baddbmm, mv and dot through a bit accurate
model of that generation's matrix unit (group size, alignment bits, truncation, the narrow Hopper
fp8 accumulator). The sm_120 profiles match an RTX 5070 Ti bit for bit on 32.7 million dot
products and the Ada fp16, bf16 and tf32 profiles were fitted bit for bit on an L4; every other
profile follows a published description and carries TS1094. See
[TENSOR_CORE_MODELS.md](TENSOR_CORE_MODELS.md).

**Calibration.** A real backend is calibrated against the CPU in the model's dtype. A simulated
backend is calibrated against its profile's *ideal*: the same storage and per op precision with
correct arithmetic (wide accumulation, round to nearest even, IEEE specials, textbook
formulations). Thresholds use the floor of the narrowest dtype the profile computes in (float16:
5e-3, bfloat16: 4e-2) and never go below 10 times the ideal's own error on that module. So a
simulated backend is blamed for what the device does beyond correct low precision math. When a
module fails under the ideal too, the report says the model does not survive that precision on any
device, so the finding is not specific to the device (random init MobileNetV3 below is the case).

**Repros.** The generated `repro.py` and `op_repro.py` take a `sim:...` device argument and replay
the simulation on the CPU (this needs truestar installed); `cpu` is the control. A simulated
finding should be confirmed on the device before it is reported upstream.

**The IR path.** `truestar.train.emulate_precision(program, profile="sim:ane")` (and
`EmulatedReferenceExecutor(profile=...)`) rounds a Truestar IR program the way the device stores
values: every float result to the storage dtype with flush to zero, saturation and NaN to +inf,
and the inputs of the profile's low precision ops (`linear`, `matmul`, `conv2d`, `attention` for
Neuron) to their dtype. Accumulation width, tiles and op formulations are modelled only in the eager
simulation.

**Not modelled.** Non tensor scalars (an `eps`, a scale) stay float64. Ops outside the list above
compute exactly and round once; when the profile's accumulator would matter for one (cumsum, pools
other than global average, group norm, fused transformer fast paths, transposed and 3d
convolutions on a narrow accumulator), the report names it (TS525). Autograd is not simulated
(forward only). Rounding goes through float32 on the way to float16, a double rounding that can
differ from a direct float64 to float16 rounding in rare ties. The simulation is as good as its
rules: a guess is a guess.

## 2. Validation against cited evidence

`tests/test_devsim.py` reproduces each case below from the issue's own setup.

| evidence | issue's numbers | simulated |
|---|---|---|
| coremltools#2687, ANE softplus | sweep of 2000 points over [-15, 15]: first point with reference > 5 and output < 1 is x = 10.4077 (output 0, reference 10.4077); fine sweep 10.394 gives 10.39, 10.395 gives 0 | `sim:ane`: x = 10.4077, output 0.0, reference 10.40773; 10.394 gives 10.3906, 10.395 gives 0.0. `sim:ieee-f16`: no cliff |
| coremltools#2690, ANE logsumexp over 32 channels | cliff at x = 7.6313, reference 11.0971, output 0.0000; output about 11.10 just before | `sim:ane`: cliff at 7.6313, reference 11.09705, output 0.0; 11.086 at the point before. `sim:ieee-f16`: no cliff |
| openvino#39703, NPU ReduceL2 over 384 channels, norms 32 to 400 | max output 255.9; max rel err 0.36 above 256 and 0.000587 below; `ReduceL2(x/2)*2` gives 400.0 | `sim:openvino-npu-f16`: max 255.875 (255.9 at one decimal), 0.3603 above 256, 0.00068 below, workaround 400.0. `sim:ieee-f16`: 400.0 |
| openvino#22613, GPU gemm of n ones | CPU 2049.0, GPU 2048.0 at n = 2049; 2048 matches | `sim:openvino-gpu-f16`: 2048.0 at 2048 and 2049, and 2048.0 at every larger n (the accumulator stops). `sim:ieee-f16`: 2048.0 at 2049 (2049 is not a float16 value), 3072.0 at 3072 |
| onnxsim#1742, decomposed LayerNorm in float16, ViT activations up to 1418 | Pow(x, 2) overflows float16; ViT B/16 top 1 falls to 19% | `sim:webgpu-f16` (step by step layer norm): the row holding 1418 is destroyed; `sim:ieee-f16` (fused): within 1e-2. The 19% accuracy needs pretrained weights and was not reproduced |
| arXiv 2606.22283 section 3.6, ANE NaN | NaN coerced to +inf; relu gives inf, sigmoid and tanh give 1, exp gives inf, softmax puts all mass on the NaN lane | `truestar probe --specials --backends sim:ane`: the same five results |

Not reproduced: the onnxruntime WebGPU f16 index and size cases (#28976, #29130) cited by the gap
research return 404 from this account, so their numbers could not be read. `sim:webgpu-f16` does
not model index math in float16.

## 3. Size sweep (`--sweep NAME:LO..HI`)

The model file takes the size as a keyword in `get_model(NAME=...)`, `get_inputs(NAME=...)` or
both. Method (`devsim/sweep.py`):

1. A ladder: LO, HI, and every `2^k` and `2^k + 1` between them (both sides of every binary
   cliff). Each size gets a quick check on every backend (no op drilldown, no repro).
2. Per backend, the last OK ladder size and the first failing one bound the cliff; bisection
   between them finds the first failing size, checking that backend only. One cliff between two
   ladder points is assumed, never inferred: every reported size was checked.
3. The full check runs once at the first failing size, naming the module and the op and writing
   the repro (TS521; TS522 when the smallest size already fails).

Example, `examples/size_cliff.py` (the openvino#22613 reproducer):

```
truestar check examples/size_cliff.py --backends cpu,sim:ieee-f16,sim:openvino-gpu-f16 --sweep n:512..8192
[OK] cpu: clean at every size checked (512, 513, 1024, 1025, 2048, 2049, 4096, 4097, 8192)
[OK] sim:ieee-f16: clean at every size checked (512, 513, 1024, 1025, 2048, 2049, 4096, 4097, 8192)
[SUSPECT] sim:openvino-gpu-f16: clean up to n=2058; SUSPECT from n=2059  TS521 TS011 TS520
  module dot#0 (Dot), own error 5.3e-03 vs threshold 5.0e-03
  op aten.bmm.default
```

Why 2059 and not the issue's 2049: at n = 2049 the result 2048 is 4.9e-4 away from the truth,
inside the float16 floor, and correct float16 gives 2048 there too (2049 is not a float16 value).
The maintainer called that expected. What is not expected of correct float16 is an accumulator
that stops at 2048; it leaves the float16 tolerance at n = 2059, and the error grows to 0.5 at
n = 4096. Truestar reports the size where the device leaves correct float16, not the first size
where any bit differs.

## 4. Special value probe (`truestar probe --specials`)

28 single op cases (`devsim/specials.py`): NaN through add, relu, sigmoid, tanh, exp, sums, amax,
softmax, layer norm, batch norm and a copy through reshape; infinities through the same; log and
sqrt edges; division by minus zero; minus zero through multiply and relu; subnormals through
multiply and matmul; float16 cast overflow; matmul overflow; float to int casts. Each element is
compared with float64 and with IEEE arithmetic in the backend's own precision (simulated, so the
CPU's kernels are not trusted as the reference). Departures from the latter are TS523; differences
that only the precision explains are TS524. What IEEE leaves to the implementation (float to int of
NaN, inf and out of range values; an overflowing dot product, which is 0, NaN or inf depending on
when products round; a mean of huge values; relu(-0), which is -0 under IEEE 754-2008 maxNum and +0
under 754-2019 maximum) is listed, never judged, and marked when backends disagree.

Measured on this machine (RTX 5070 Ti, torch 2.11, float32), 10 backends, 2 to 3 s:

* `cpu` and `cuda`: 0 departures from IEEE. Real disagreements between them: NaN to int32 is
  -2147483648 on the CPU and 0 on CUDA; +inf and 3e9 to int32 are -2147483648 on the CPU and
  2147483647 on CUDA; relu(-0) is -0 on the CPU and +0 on CUDA. A model whose output depends on
  any of these is not portable between the two.
* `sim:ane`: 24 departures, all from NaN becoming +inf (sigmoid of NaN is 1, softmax puts all mass
  on the NaN lane, layer norm and sums return inf).
* `sim:webgpu-f16`: 5 departures, all subnormals flushed to zero.
* `sim:openvino-npu-f16`: overflow saturates at 65504 (the guess generalising openvino#39703).
* The IEEE profiles, Mali, Neuron and OpenVINO GPU: 0 departures; their float16 overflows to inf
  and subnormal underflows are listed as precision (TS524).

## 5. Measurements

Measured 2026-11-24 on the CPU of this machine (other jobs were running, so times are upper
bounds), torch 2.11, random init models, `truestar check` with op drilldown.

**Cases reproduced: 6 of 6 readable cited cases** (table in section 2), plus a real CPU against
CUDA disagreement found by the probe (section 4). The two onnxruntime WebGPU issues could not be
read and are not counted.

**False positives on clean models.** A false positive is a simulated backend blaming a module that
correct arithmetic in the same precision gets right.

| model (input) | backends | flagged | of which device specific |
|---|---|---|---|
| MLP (32 x 32) | all 8 | 0 | 0 |
| 2 layer transformer encoder (4 x 16 x 64) | all 8 | 0 | 0 |
| ResNet18 (2 x 3 x 64 x 64) | all 8 | 0 | 0 |
| ViT B/16 (2 x 3 x 64 x 64) | ieee-f16, ane, neuron-bf16, webgpu-f16, openvino-npu-f16 | 0 | 0 |
| MobileNetV3 small (2 x 3 x 64 x 64) | all 8 | 6 (every float16 profile, `sim:ieee-f16` included) | 0 |
| tests: MLP, CNN with batch norm and grouped conv, attention block | all 8 | 0 | 0 |

MobileNetV3 with random weights carries activations of 1e-7 to 1e-10 into its last blocks, below
the smallest float16 subnormal (6e-8), the same cause as the NPU false alarm in docs/NPU.md. Every
float16 profile flags it, correct float16 included, and the report says so in a note ("over its
threshold under correct float16 arithmetic too ... not specific to sim:ane"). The two bfloat16
profiles are clean. That is a real finding about the model in float16, not about a device.

**Runtime** (seconds for one `truestar check` on one backend, including its calibration; the CPU
row is the plain check):

| model | cpu | ieee-f16 | ane | neuron-bf16 | openvino-npu-f16 | webgpu-f16 | mali-fp16 | openvino-gpu-f16 |
|---|---|---|---|---|---|---|---|---|
| MLP | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| transformer | 0.03 | 0.07 | 0.09 | 0.10 | 0.09 | 0.14 | 0.19 | 0.20 |
| ResNet18 | 0.19 | 1.2 | 2.2 | 0.9 | 1.1 | 6.9 | 8.5 | 6.0 |
| MobileNetV3 small | 0.24 | 1.8 | 2.0 | 1.4 | 3.5 | 25.8 | 5.2 | 5.5 |
| ViT B/16 | 18.8 | 58.9 | 27.1 | 6.4 | 10.6 | 125 | not run | not run |

Profiles with a wide accumulator cost 5 to 10 times the plain CPU check. Profiles with a float16
accumulator loop over the reduced dimension (that order is the point of the simulation) and cost
25 to 100 times; webgpu-f16 on MobileNetV3 pays the most because its grouped convolutions run one
group at a time. Against the alternative (export, device job, compare per guess) every row is one
local run with zero device jobs. The ViT CPU number was taken while other jobs held the machine.

Sweep of `examples/size_cliff.py` over 512 to 8192 on three backends: 33 checks, 5.6 s (the ladder
is 9 sizes; bisection added 11 per failing backend; a full grid would be 7681 sizes). Special
value probe: 28 cases on 10 backends in 2 to 3 s.

## 6. Redesign against what people use today

* **Core ML users** convert, then compare on a Mac with `CPU_ONLY` as the reference, and that
  reference is itself float16 and sometimes further from float32 than the ANE (coremltools#2625).
  Here: one CPU run, the float64 reference, the module and op named, no Apple hardware.
* **Hailo's emulator** runs only Hailo's chip. Here: one profile per device, the same report for
  all of them, and every rule is a cited line a user can read and dispute.
* **Size bugs** are found today when a user uploads a 12 MP photo. Here: a ladder over every binary
  cliff plus bisection, about `2 log2(HI/LO) + log2(gap)` checks instead of a full grid (33 checks,
  5.6 s for 512 to 8192 on three backends).
* **Special values** are found one issue at a time (burn#5284: "I couldn't find a statement in the
  documentation about whether backends are expected to agree"). Here: one table per backend, with
  IEEE in the backend's own precision as the judge and implementation defined cases kept apart.

## 7. `sim:hexagon`: fitted to real Snapdragon outputs (P59)

Every other profile in this file follows a public description. `sim:hexagon` follows Truestar's
own device runs: the Qualcomm AI Hub battery (docs/QAIHUB.md, battery v1, seed 20261124) on a
Galaxy S26 (Hexagon v81, TFLite NPU and QNN DLC) and a Snapdragon X Elite (v73, QNN DLC and ONNX
Runtime QNN). The four float runs computed in float16 and agree with each other on 99.98% of the
GELU outputs. Fit and measurement: `scripts/hexagon_sim_fit.py` (fit, measure, fixture); the small
fixture the tests read is `tests/data/hexagon_npu_taps.npz` (66 KB).

What the profile does, each rule labelled in `truestar backends --sims`:

* float16 storage and a wide accumulator, which is `sim:ieee-f16`: the convolutions measured equal
  to it bit for bit on 94% to 100% of elements, so they are left alone;
* GELU is fused into the linear layer before it: it reads the unrounded accumulator (kept through
  `view`, dropped by any in place write) and returns `x * phi(x)` rounded once to float16;
* `phi` is the device's normal CDF approximation, fitted per sign: a line for |x| < 0.6 (slope
  0.3871 where the true slope is 0.3989), a degree 5 polynomial up to the clamp, 0 for x at or below
  -3.02 (every device output there is exactly 0) and 1 at or above 3.00. The device is not symmetric
  (`phi(-x)` is not `1 - phi(x)`), and the tanh and sigmoid forms of GELU do not reproduce it.

TS1140 names each GELU that used the fit (and says when its input was not a linear layer, so the
fusion was not applied); TS1141 names ops whose device behaviour is known to differ but is not
modelled (softmax, attention) or was never measured (`approximate='tanh'`).

**Measured, GELU alone** (the first MLP layer; fit on every other input sorted by value, scored on
the other 4096):

| GELU on held out inputs | rel L2 to the S26 | bit equal |
|---|---|---|
| exact GELU of the float16 input (what `sim:ieee-f16` computes) | 2.71e-3 | 6.7% |
| fitted curve, float16 rounded input (no fusion) | 4.14e-4 | 72.5% |
| fitted curve, fused accumulator input (`sim:ieee-f16` to `sim:hexagon`) | 2.73e-4 | 89.4% |

**Measured, the whole battery** (`scripts/hexagon_sim_fit.py measure`, rel L2 of each simulation to
the device, S26 TFLite NPU; the S26 DLC run gives the same numbers and both X Elite runs the
same ratios, 10.9x, 7.8x, 6.1x and 1.8x):

| tap | sim:ieee-f16 | sim:hexagon | closer | bit equal before, after |
|---|---|---|---|---|
| mlp.fc1_gelu | 2.71e-3 | 2.50e-4 | 10.9x | 6.7%, 89.6% |
| mlp.fc2_relu | 2.64e-3 | 3.41e-4 | 7.8x | 54.2%, 77.1% |
| mlp.out | 2.19e-3 | 3.56e-4 | 6.1x | 10.0%, 61.3% |
| enc.out (a GELU the fit never saw, 3d input) | 1.42e-3 | 7.74e-4 | 1.8x | 15.9%, 30.1% |
| enc.softmax, enc.attn_out, enc.norm1 | 1.26e-3, 1.16e-3, 5.38e-4 | unchanged | 1.0x | unchanged |
| conv.stem, block1, down, block2, pool, out | 5.1e-6 to 1.3e-4 | identical bits | 1.0x | 100%, 99.7%, 99.1%, 94.3%, 98.4%, 80% (unchanged) |

`sim:hexagon` and `sim:ieee-f16` give bitwise identical convolution outputs on the battery (a test).
The device GELU's errors lean upward (TS961's own test on the real S26 tap: mean +6.53e-4 per
element, p 1.6e-95); `sim:hexagon` predicts +6.59e-4 and `sim:ieee-f16` shows no lean. So a model
with GELU gets TS961 (suspect) on `sim:hexagon`: that is the device's behaviour, not a false alarm,
and models without GELU are clean (tests/test_hexagon_sim.py, tests/test_devsim.py).
What is left on the encoder is softmax and attention, which the device computes 1.7x to 2.0x further
from float64 than correct float16; nothing here models that yet (TS1141).

**Tried and not shipped: an INT8 requantization rule.** On the NPU, 1% to 4.6% of the INT8 MLP's
elements per layer are one step from exact requantization (never more). Replayed per layer from the
device's own input codes, every mismatch sits within 0.03 of a half step on v73 and v81 (both
directions) and within 0.12 on v68 (almost always rounding down), with the same rate at every output
magnitude, which is what an additive error of about 0.01 step looks like. None of these reproduced
the device better than exact requantization on every run: a float16 product, a float16 or bfloat16
copy of each scale (27 combinations), a float16 or float32 dequantize, matmul and requantize
pipeline, double rounding through a grid of 1/2 to 1/256 of a step with five inner and three outer
rounding rules, a constant offset per output channel (consistent on 97% of channels, not all), a
16 bit internal activation, and twelve input quantizers for the first layer (float32 division stays
best: 4.6% off on the S26 DLC run, 0% on ONNX Runtime CPU). With 16 rows per layer and no device
input codes for the first layer, the data cannot separate these further; the INT8 comparison keeps
using `truestar conform qaihub` and its intref conventions (TS926, TS928).

Limits: one input sample, untrained weights, GELU inputs within [-3.4, 3.4] (the clamp beyond is
measured on 4 inputs per side); v68 boards do not run float16 on the NPU, so they are not in the
float fit.
