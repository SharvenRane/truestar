# Real phone, laptop and IoT NPUs through Qualcomm AI Hub (`truestar conform qaihub`)

Pain ledger row P45 (docs/QUALITY_BAR.md). Code: `src/truestar/conform/qaihub_battery.py` (the
battery: models, inputs, references, ONNX files), `src/truestar/conform/qaihub.py` (the file based
check), tests in `tests/test_qaihub.py`. Measured 2026-11-24 with qai-hub 0.59.

Until now Truestar checked phones and NPUs through simulated devices (`sim:*`, docs/DEVICES.md)
and through deployed files run on local runtimes (`truestar conform`). This is the first check
against hosted Snapdragon hardware with a fixed battery whose every intermediate tensor is an
output, so a run can be compared with float64, float32 and the `sim:*` prediction tap by tap.

## The battery

Two ONNX files, each one compile and one inference job per target:

* `float`: three float32 networks side by side, 13 taps. An MLP (8 x 512 input, 512 to 1024 GELU,
  1024 to 1024 ReLU, 1024 to 10), a residual conv net (1 x 3 x 64 x 64, 3x3 convs with 32 and 64
  channels, two residual blocks, one stride 2 conv, global mean pool, linear head) and one post
  norm transformer encoder layer written op by op (64 tokens, width 128, 4 heads, FFN 512, GELU).
  Exported with the torch.onnx dynamo exporter at opset 18 (GELU becomes Erf, layer norm becomes
  LayerNormalization).
* `int8`: a three layer MLP (16 x 256 input, 256, 256, 10) as a QDQ graph: uint8 activations with
  per tensor scales from a separate calibration draw, int8 weights per output channel from
  `truestar.quant.quantize_weight(w, "int8")`, int32 biases, ReLU before each requantize. Every
  requantized layer output is a tap. Truestar has no QDQ exporter, so the graph is written with
  `onnx.helper` in `qaihub_battery.int8_onnx_bytes`.

Weights and inputs come from numpy PCG64 `random()` scaled to unit variance; the input and weight
hashes are pinned in the tests and are identical on Windows and Linux.

## What the check computes (no network)

`truestar conform qaihub DIR` reads `summary.json` and one `.npz` per run, regenerates the inputs
and checks their hashes, and recomputes every reference itself.

* Float taps: relative L2 error against float64, set against float32 eager PyTorch (a correct
  float32 device) and `sim:ieee-f16` (correct float16: float16 storage, wide accumulation). Within
  8x the float32 error (or 1e-5): float32. Within 4x the `sim:ieee-f16` error: float16 as simulated
  (TS923 on a float32 model). Above: beyond correct float16 (TS924), with the ratio to
  `sim:mali-fp16` (a float16 accumulator) as the next simulated explanation. An NPU or GPU run that
  is float32 accurate everywhere gets TS925 with the per layer placement AI Hub's profile reports.
* INT8 taps: each layer is replayed with `truestar.intref` from the device's own previous tap (the
  battery input for the first layer), so a layer is judged on its own and not on inherited drift.
  Elements one step from the intended arithmetic (ONNX QLinear: exact integer accumulation, float32
  requantization, ties to even) are TS926, any element further off TS927, and the intref
  convention that matches best is named (TS928). The chained comparison against the intended model
  end to end is reported too.
* Output names: AI Hub's TFLite and QNN targets rename outputs `output_0`, `output_1`, ... in ONNX
  order; they are matched by position when every size fits, and NHWC conv taps are transposed back.

## First run, 2026-11-24: four devices, 25 jobs

Samsung Galaxy S26 (Snapdragon 8 Elite Gen 5, Hexagon v81), Samsung Galaxy S21 (Snapdragon 888,
v68), Snapdragon X Elite CRD (v73, Windows), Dragonwing RB3 Gen 2 (QCS6490, v68, Linux). 7 compiles
and 18 inference jobs, each with AI Hub's per layer profile pass. Host controls: ONNX Runtime CPU
matches float32 eager on every float tap (rel L2 under 4.2e-7) and intref ONNX QLinear bitwise on
every INT8 tap.

Float battery (worst tap; ratio = device error over the `sim:ieee-f16` prediction):

| device, runtime / unit | placement | ran in | worst rel L2 | verdict |
|---|---|---|---|---|
| S26 TFLite npu, S26 QNN DLC npu (bitwise identical) | NPU | float16 | 2.71e-3, mlp.fc1_gelu, 6.2x | TS924 |
| X Elite ORT QNN EP npu, X Elite DLC npu (bitwise identical) | NPU | float16 | 2.71e-3, 6.2x | TS924 |
| S26 TFLite gpu, S21 TFLite gpu | CPU, all 61 layers | float32 | 7.6e-7 | TS925 |
| S21 TFLite npu, RB3 Gen 2 TFLite npu | 13 transposes on NPU, all arithmetic on CPU | float32 | 7.3e-7 | TS925 |
| S26 TFLite cpu, X Elite ORT cpu | CPU | float32 | 7.8e-7 | ok |
| S21 DLC npu, RB3 Gen 2 DLC npu | compile on device failed (v68 has no float16 HTP) | | | TS922 |

* Convolutions are `sim:ieee-f16` exactly: error ratio 1.00x and 94% to 100% of elements bitwise
  equal after float16 rounding. `sim:mali-fp16` (a float16 accumulator) is ruled out.
* Layer norm 1.02x to 1.04x; softmax and attention 1.7x to 2.0x, different between v81 and v73.
* GELU (exported as Erf) is 6.2x correct float16 on both NPU generations, with the same bits. The
  error is float16 level for |x| < 0.5 and about 2.7e-3 rms for 0.5 < |x| < 1: an approximation of
  Erf, not the matmul (tiled and float16 accumulator variants do not reproduce it, nor do tanh or
  sigmoid GELU). No `sim:*` profile models it; this is where the simulation underpredicts the NPU.
* An NPU or GPU request that returned SUCCESS ran in float32 on the CPU in 4 of the 8 float accelerator runs that completed;
  only the error level and the profile show it (TS925).
* X Elite outputs are float16 values moved about 1e-7 relative off the float16 grid (the float16 to
  float32 output conversion is inexact there); round to float16 before comparing devices bitwise.

INT8 battery (per layer replay from the device's own input):

| device, runtime | l1 / l2 / out elements one step off | verdict |
|---|---|---|
| S26 DLC npu | 4.59% / 0.93% / 3.75% | TS926 |
| S21 DLC npu and RB3 Gen 2 DLC npu (bitwise identical, both v68) | 4.32% / 2.32% / 3.12% | TS926 |
| X Elite DLC npu | 3.88% / 1.20% / 4.38% | TS926 |
| X Elite ORT QNN EP npu (input Quantize on the CPU) | 0.98% / 1.00% / 2.50% | TS926 |
| X Elite ORT cpu | 0 / 0 / 0 | ok |
| TFLite, every device | compile failed | TS922 |

* Never more than one step, but 23% to 31% of the final outputs differ from the intended model.
* No intref requantization rule, input quantizer or float16 pipeline explains the one step
  differences (all within 0.05 points of each other), as docs/NPU.md found for ResNet18. The first
  layer drops from about 4% to 1% when the float input is quantized on the CPU instead of the NPU.
* AI Hub's ONNX to TFLite converter rejects per channel QDQ weights in both the MatMul and the
  Gemm form ("'tfl.transpose' op has mismatched quantized axes of input and output"); the same
  graphs compile to QNN DLC and ONNX. Since 2026-11-25 the battery also writes the INT8 MLP directly as a full integer
  .tflite (`qaihub_battery.int8_tflite_bytes`: int8 activations with the uint8 zero points shifted
  by 128, per channel int8 weights in FULLY_CONNECTED's own layout, so no transpose), which AI Hub
  runs as it is with no conversion. On the host, LiteRT's CPU runtime gives every tap of it exactly
  as intref's ONNX QLinear convention (`tests/test_qaihub.py`). It has not been run on a device yet.

## Limits

One sample per input, one seed, seeded untrained weights. Taps force tensors to exist, which may
change fusion. Float taps carry inherited error (no per op replay on the device). No GPU number:
the TFLite GPU delegate took no layer on either phone and QNN GPU was not tried. No `sim:*` profile
for the Hexagon NPU yet; the measurements above are what one would be fitted to.
