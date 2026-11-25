# Integer reference and rounding conventions (`truestar intref`, pain P29)

An INT8 model has one set of scales and zero points, yet libraries compute it differently: TFLite
rounds twice (a high multiply, then a rounding shift) unless built with `TFLITE_SINGLE_ROUNDING`,
CMSIS-NN has one global flag for the same choice, ONNX Runtime rescales in float32 and rounds half
to even, int16 activations use a 16 bit multiplier. People find out when a board disagrees with the
desktop by 1 LSB and nobody can say whose rounding is right: "This makes matching TFL in CMSIS-NN
impossible" and "for some reference kernels not even TFL and TFLM match"
([CMSIS-NN#214](https://github.com/ARM-software/CMSIS-NN/issues/214), 2026-04-08). Phone NPUs add a
third party: docs/NPU.md measured 4.44% of the S24 NPU's first layer elements one step away from
the CPU, with the cause left as an inference.

`truestar.intref` is an exact integer reference for quantized ops under each convention, and three
uses of it: say which convention a device output follows, per op; explain a device difference when
the layer's parameters are unknown; write C golden vectors that any board's harness can compare
against.

## Use

```
truestar intref conventions                   # every rule and convention, JSON
truestar intref match case.npz                # which convention a device output follows
truestar intref cases                         # CMSIS-NN#214, ONNX Runtime oracle, pair table
truestar intref npu                           # explain docs/NPU.md from the saved jobs (no jobs)
truestar intref npu --summary                 # same from the recorded numbers, no job cache needed
truestar intref golden -o golden.h --convention cmsis_nn --tolerance portable
```

Each takes `--format agent` (one JSON document, envelope of docs/AGENT_INTERFACE.md with
`"command": "intref"`). Exit codes: 0 matched or explained, 1 a finding (no convention matches,
or the difference is not a rounding convention), 2 usage error or invalid case.

A case file for `match` is an `.npz` with `op` (conv, fully_connected, matmul, add, mul, avg_pool,
max_pool, softmax, requantize, quantize), the integer inputs by name (`x`, `w`, `b` or `a`, `b`),
`device` (the device's integer output) and `params` (a JSON string with zero points, scales,
stride and padding; keys as in `truestar/intref/ops.py`). Layout is NCHW with OIHW weights;
transpose TFLite's NHWC tensors first.

```python
from truestar.intref.ops import run_op
from truestar.intref.match import match_op, distinguishing_case
y = run_op("mul", {"a": a, "b": b}, params, "tflite_ref")          # exact int8 output
v = match_op("mul", {"a": a, "b": b}, params, device_output)      # v.matched, v.groups, v.code
case = distinguishing_case("fully_connected", params)             # inputs that separate conventions
```

## Rules and conventions

A rule computes round(acc times multiplier). A convention says which rule each op uses in one
library build, how it rounds a float to an integer, and how it rounds an average pool sum.

| rule | what it does | used by |
|---|---|---|
| `q31_double` | gemmlowp SaturatingRoundingDoublingHighMul then RoundingDivideByPOT | TFLite default build |
| `q31_double_cmsis` | `arm_nn_doubling_high_mult_no_sat` then `arm_nn_divide_by_power_of_two` | CMSIS-NN default build |
| `q31_single_up` | (x M0 + 2^(t-1)) >> t, one rounding, ties toward plus infinity | `TFLITE_SINGLE_ROUNDING`, `CMSIS_NN_USE_SINGLE_ROUNDING` |
| `q31_single_even`, `q31_single_away` | x M0 / 2^t rounded once, exact, ties to even or away | definitions, to separate tie rules |
| `q15_single_up` | multiplier reduced to 16 bits, one rounding | TFLite and CMSIS-NN with int16 activations (int64 accumulators) |
| `f32_even` | float32(acc) times float32(multiplier), half to even | ONNX QLinear spec, ONNX Runtime, XNNPACK |
| `f32_away`, `f64_even` | float variants | for comparison |
| `f16_multiplier` | multiplier in float16 | hypothesis only, not a library |

| convention | rules | quantize | avg pool |
|---|---|---|---|
| `tflite_ref` | `q31_double`; int16 conv and FC `q15_single_up`; ADD as TFLite (left shift 20, three multipliers) | x / s, half away | (acc ± n/2) / n |
| `tflite_single` | `q31_single_up` | as above | as above |
| `tflite_2_21_py` | FC `q31_single_up`, everything else `q31_double` (as reported in CMSIS-NN#214; TensorFlow is not installed here, not reproduced) | as above | as above |
| `cmsis_nn`, `cmsis_nn_single` | `q31_double_cmsis` or `q31_single_up` | as above | as above |
| `onnx_qlinear` | `f32_even`, ADD in float32 | x / s, half to even | float, half to even |
| `exact_half_even`, `exact_half_away` | the exact rules | as TFLite | as TFLite |
| `npu_fp16_input` | as `onnx_qlinear`, but the input is quantized as float16(x) times a float16 reciprocal scale | | hypothesis, see below |

No public description of Qualcomm Hexagon or QNN HTP requantization was found (the web search
tool was out of credits this session, so the search was not thorough), so no convention is claimed
for it.

Measured, not assumed (tests/test_intref.py): gemmlowp's high multiply and CMSIS-NN's give the same
result on every input with a non negative multiplier, including exact ties, because gemmlowp's
negative nudge `1 - 2^30` sends negative ties toward zero, so both round ties toward plus infinity
(gemmlowp's comment calls it rounding to nearest; it is not half away from zero). So the default
TFLite and CMSIS-NN builds agree with each other everywhere; the CMSIS-NN#214 conflict is single
against double rounding, not TFLite against CMSIS-NN. The two spellings of single rounding (TFLite
one shift, CMSIS-NN two shifts) are also identical. Accumulators are summed in float64 matmuls,
exact while partial sums stay under 2^53 (asserted), about ten times faster than an int64 einsum.

Sources, all read 2026-11-24 from the main branches: tensorflow/lite/kernels/internal/common.cc,
quantization_util.cc and reference/integer_ops (add.h, mul.h, conv.h, fully_connected.h,
pooling.h); google/gemmlowp fixedpoint/fixedpoint.h; ARM-software/CMSIS-NN
Include/arm_nnsupportfunctions.h.

## Measured

### Distinguishing conventions on cited cases (`truestar intref cases`, 0.6 s)

CMSIS-NN#214. The device follows the reported TFLite 2.21 behaviour. On random inputs and on
distinguishing inputs alike, the per op verdict is: FULLY_CONNECTED matches the single rounding
class (tflite_single, cmsis_nn_single, onnx_qlinear, exact tie rules) and not tflite_ref or
cmsis_nn; MUL matches tflite_ref and cmsis_nn and not the single rounding class. No single
CMSIS-NN build matches both ops, which is the issue's complaint, now stated in one line.

Pairs of the 8 documented conventions separated (28 pairs per op):

| op | random inputs | distinguishing inputs | still identical |
|---|---|---|---|
| conv | 0 | 15 | double rounding class; single rounding and float class |
| fully_connected | 12 | 12 | tflite_ref and cmsis_nn; the single rounding class (with tflite_2_21_py) |
| mul (every int8 pair) | 15 | 15 | double class; single and float class |
| add (every int8 pair) | 19 | 19 | double class; single class |
| requantize | 15 | 15 | double class; single and float class |
| mul, scales with an exact power of two ratio | 23 | 23 | double class; single class; half even (onnx_qlinear, exact_half_even) |

With ordinary scales an exact tie almost never happens, so tie rules (half up, half even, half
away) cannot be separated by any input; Truestar says so (TS542) instead of guessing. When the
scale ratio is a power of two, ties occur and the four classes separate: double rounding, single
half up, half even, half away. Pairs that are the same rule by construction stay identical; that is
the measured 23 of 28.

Against float fake quantization (what a QAT simulation computes): on every int8 pair of a MUL with
the cited scales it equals tflite_ref on 99.16% of 65,536 elements; with power of two scales on
97.22%. It cannot express double rounding at all, so it can never say which convention a device
used.

ONNX Runtime 1.20.1 CPU kernels as a real device (graph optimizations on):

| ORT op | verdict |
|---|---|
| QLinearConv (uint8, per channel int8 weights) | matches onnx_qlinear and the single rounding class bit for bit; differs from tflite_ref and cmsis_nn (TS540) |
| QuantizeLinear with exact ties | matches onnx_qlinear only (half to even) (TS540) |
| QLinearMul (com.microsoft), every int8 pair | matches the single rounding class, not onnx_qlinear: ORT computes this op in fixed point (TS540) |
| QLinearMatMul | every convention agrees on these inputs (TS542) |
| QLinearAdd (com.microsoft), every int8 pair | matches no modelled convention; closest onnx_qlinear on 99.81%, worst 1 LSB (TS541). Its source could not be fetched this session; reported, not hidden |

### The NPU 1 LSB differences (`truestar intref npu`, 10.7 s, 0 device jobs)

Data: the cached INT8 tap runs of docs/NPU.md (S24 NPU jp0810ee5, S24 CPU j5w0d6nmp), relu tap
after conv1 and bn1, 16 inputs, 4,194,304 elements, output step 0.0297666 (from the data). The
quantized model was never saved on this machine, so conv1's integer weights and input scale are
unknown.

* 186,314 elements differ (4.44%, as docs/NPU.md says): 104,655 up, 81,659 down, worst 3 LSB. The
  rate grows with the output level: 7.6% at levels 1 to 15, 8.3%, 10.0%, 12.9%, 17.6% at 128 to 254.
* Rounding only test: all 64 channels differ in both directions; the minority direction is 3.39% of
  elements (median per channel). Two documented rules applied to the same accumulator differ in
  the minority direction on at most 0.041% of elements (worst pair over 64 multipliers, measured
  in tests), so the bound is 0.5%. Verdict: not a requantization rounding difference; the
  accumulators differ, so the cause is before the requantizer (TS543).
* Control: the NPU's max pool equals the exact max pool of its own relu output on 100% of elements.
* Hypotheses, simulated with the integer reference on conv1 (BN folded, per channel symmetric int8
  weights, the measured step, input scale swept over 0.030 to 0.045 because it is unknown), scored
  by the mean gap to the observed rate per level bin:

| hypothesis | simulated rate | gap |
|---|---|---|
| input quantized in float16 (`npu_fp16_input`) | 3.86% to 4.34% | 0.009 |
| tflite_ref, tflite_2_21_py, cmsis_nn (double rounding) | 0.08% to 0.13% | 0.111 |
| tflite_single, cmsis_nn_single, exact tie rules | 0.00% | 0.113 |

What this says: the 1 LSB pattern is not a rounding convention. It is what an NPU produces when it
quantizes the float32 input by converting it to float16 and multiplying by a float16 reciprocal of
the input scale: some inputs land one step away, conv1 sums 147 of them, and the output moves by
1 to 3 LSB in both directions, more often for larger outputs. The simulated signature matches the
observed one within one percentage point in every bin, for every input scale tried. This is an
inference from saved data; Qualcomm does not document it, and a per element confirmation needs
the quantized model (one download, no new device job) or a run with integer input. The practical
consequence: feeding the NPU an already quantized uint8 input, instead of float32, should remove
this difference.

The rounding only test and the float16 hypothesis also hold in a unit test on a synthetic layer:
every documented convention against `onnx_qlinear` passes the test, float16 input quantization
fails it.

### C golden vectors (`truestar intref golden`, 0.26 s)

`golden.h` is one C99 header with only `<stdint.h>`: per layer the integer input(s), the expected
output under the chosen convention, a tolerance in LSB, an FNV-1a checksum of the expected bytes,
zero points and output scale, plus `ts_golden_check(layer, got, &worst)` and `ts_golden_intact()`.
Each layer's input is the previous layer's expected output, so layers can be checked one at a time.
The built in network covers conv (with relu), max pool, avg pool, fully connected, add, mul and
softmax (10.7 KB). `--tolerance exact` gives 0 LSB; `--tolerance portable` gives each layer the
largest difference between the chosen convention and any other documented one on the same input
(cmsis_nn: conv 1 LSB, avg pool 1 LSB, all others 0).

No C compiler exists on this PC (searched PATH, the empty Visual Studio folders, and the disk to
depth 5; none installed, as instructed). The header is validated by parsing it back in Python: every
array length, value range, checksum and the layer count, and a round trip against the arrays
written. A single changed value is caught by the checksum; a 2 LSB error on one layer is counted
by the Python version of `ts_golden_check`. Compiling it on a board is not verified.

## QAIRT range modes (`truestar.intref.qairt`, research item 6.5, ledger P56)

The conventions above say how a library rounds an accumulator. Before that, a quantizer picks the
encoding (scale, offset) from calibration data, and Qualcomm's QAIRT has four ways to pick it. A
device whose kernels are correct still gives other codes when its encoding came from another range
mode, and codes one step apart when its tie rule differs. `qairt` encodes each mode as data so a
difference on Snapdragon can be named as a range mode choice before anyone calls it a kernel bug.

Source: the QAIRT quantization page (https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/quantization.html,
read 2026-11-24): "QNN supports four quantization modes: tf, symmetric, enhanced, and adjusted.
The primary difference is between how they select the quantization range to be used"; "Minimum
range of 0.0001 is enforced"; "Floating point zero is exactly representable"; "quantized value =
round(255 * (floating point value - encoding.min) / (encoding.max - encoding.min))"; "Per-channel
quantization must use symmetric representation with offset == -2^(bitwidth-1)".

| mode | how the range is chosen | status |
|---|---|---|
| `tf` | true min and max, zero included, then shifted by up to half a step so zero is a code | documented; the doc's two worked examples are tests ([-1.8, 0.5] gives codes 0, 89, 200, 255; [-5.1, 5.1] gives -5.12 to 5.08) |
| `symmetric` | (-a, a) with a = max(abs(min), abs(max)); offset -2^(bw-1) | documented |
| `enhanced` | Truestar's own search: the clipped range with the least mean squared error over a 48 by 48 grid | hypothesis: the doc only says it may cut long tails |
| `adjusted` | the smallest grid with an integer offset that covers the whole true range (widen, never shift inward) | hypothesis: Truestar's reading of "the max will be expanded or the min will be decreased if necessary" |

Tie rule: the doc writes "round" without a tie rule, so `half_away` (C++ std::round) and `half_even`
are both modelled and both scored. Bit widths 2 to 32 (8 and 16 tested), signed codes are unsigned
codes minus 2^(bw-1).

Use:

```python
from truestar.intref import qairt
enc = qairt.encoding(calibration, bw=8, mode="tf")        # Encoding(min, max, scale, offset, ...)
a = qairt.attribute(reference, device_dequantized)          # TS1130 .. TS1133, every mode scored
qairt.compare_to_device(ref_taps, device_taps)              # the same per tap (hook for device runs)
sim = qairt.simulate(model, calib_inputs, inputs, qairt.SimConfig(act_mode="tf", act_bw=16,
                     per_channel=True, float16_modules=("head",)))   # float64 fake quant prediction
```

`attribute` verdicts: TS1130 one mode and tie rule reproduces every element; TS1131 the device uses
one mode's grid and differs by exactly one step on some elements (tie or input rounding); TS1132 no
mode gives the device's grid, or the grid matches but values are more than a step off (the cause is
before the quantizer); TS1133 the device values are on no grid. On constructed device tensors (each
mode applied to 4002 values with a long tail) every mode is named; a one step change on every 97th
element is TS1131, a five step change on every 50th is TS1132 (tests/test_qairt.py).

`simulate` fake quantizes the model input, every leaf module output (activation mode and width) and
every conv or linear weight (parameter mode, or symmetric per output channel), in float64; bias
stays float64 (QNN keeps it in 32 bits). Modules named in `float16_modules` run with float16 inputs,
weights and outputs instead (the float fallback and FP16 override modes). Inputs outside the
calibration range are clamped, as QNN clamps. On an 8, 16, 4 MLP the int16 prediction is within
1e-3 of float64 and int8 within 5e-2.

Not verified on a device here: whether QNN HTP's tie rule is half away or half even, and whether
`enhanced` and `adjusted` match Qualcomm's code. A real QNN job's per layer outputs go into
`compare_to_device`; that is the check that settles both.

## Limits

* Softmax: TFLite's and CMSIS-NN's int8 softmax use fixed point exp tables, which are not modelled;
  softmax here is float64 then the convention's quantizer, so it separates conventions only by
  that final rounding.
* `tflite_2_21_py` encodes a user's report; TensorFlow is not installed here, so it is not
  reproduced against the real interpreter. TFLM, the Corstone FVP path of gap item 13 and real
  boards are not run.
* Depthwise and transpose convolutions, LSTM, int4 weights and asymmetric weights per channel are
  not covered; conv with `groups` covers depthwise arithmetic but not TFLite's depthwise multiplier
  layout.
* The NPU explanation is a signature match, not a per element match: without the quantized model
  the per element target of gap item 2 (99.9% under the best convention) cannot be computed.
* ONNX Runtime's QLinearAdd matches no modelled convention.
