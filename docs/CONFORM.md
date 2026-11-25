# Conformance of a deployed artifact (`truestar conform`, pain P31)

Author: Sharven Rane. First written 2026-11-24.

The file that ships is not the model that was tested. A converter rewrites ops (gelu as its tanh
approximation, layer norm as a sequence of means and square roots, a quantize step), and the
runtime that executes the file picks its own kernels, delegates and precision. The usual check is
the converter's quickstart: run one input through the file and through PyTorch and call
`numpy.allclose(..., atol=1e-5, rtol=1e-5)`. It answers pass or fail on one input on the host CPU.
It cannot say where the file is wrong, whether the converter or the runtime is at fault, whether
the GPU result is a float16 result, or whether half the graph quietly ran on the CPU
([litert-torch#718](https://github.com/google-ai-edge/litert-torch/issues/718): LayerNorm zeros on
the LiteRT GPU; [WindowsAppSDK#6610](https://github.com/microsoft/WindowsAppSDK/issues/6610): 100%
CPU and no per node detail; [LiteRT#10342](https://github.com/google-ai-edge/LiteRT/issues/10342):
turning on GPU profiling changes 128 of 512 outputs; [MNN#4653](https://github.com/alibaba/MNN/issues/4653):
"3.5.x correct, 3.6.0 wrong").

`truestar conform` runs the deployed file in its own runtime and judges it against the PyTorch
model in float64.

Code: `src/truestar/conform/` (artifact.py formats and cuts, runtimes.py, core.py, bundle.py,
golden.py, tflite_graph.py, tflite_build.py, cli.py). Tests: `tests/test_conform.py`, models and
injected faults in `tests/conform_models.py`. Measurements: `scripts/conform_study.py`.

## Use

```
truestar conform model.py model.onnx                       # the format's runtimes that can run here
truestar conform model.py model.tflite --runtimes tflite:interp,tflite:gpu,tflite:gpu-f32
truestar conform model.py model.pte --inputs calib.pt --format agent
truestar conform replay model.onnx.tsgold model.onnx       # after a runtime, SDK or driver change
truestar conform runtimes                                  # every runtime and why it cannot run here
```

`model.py` is the file `truestar check` takes (`get_model()`, `get_inputs()`). `--inputs` is a
`.pt` file holding a list of samples; without it, 8 samples are drawn around `get_inputs()` (the
example itself, then normals with each input's mean and std; shapes stay the example's, since
deployed files usually have static shapes). The golden bundle goes to `<artifact>.tsgold` unless
`--golden PATH` or `--no-golden`. Exit codes: 0 every runtime conforms (replay: identical or changed
within tolerance), 1 a finding, 2 usage error.

```python
from truestar.conform import conform, replay
rep = conform(model, example, "model.tflite", runtimes=["tflite:interp", "tflite:gpu"], golden="m.tsgold")
print(rep.to_text())
rep.results[0].blamed        # module, own error, cause ("conversion" or "runtime"), nodes, user source
rep.results[0].residency     # where every node ran, the ones off the requested device named
results, issues = replay("m.tsgold", "model.tflite")
```

## Runtimes

| runtime | what runs | residency read from |
|---|---|---|
| `ort:cpu` | ONNX Runtime, CPU provider | CPU only by construction |
| `ort:cuda` | ONNX Runtime, providers `[CUDA, CPU]` as users write them | one profiled run, provider per node |
| `ort:webgpu` | ONNX Runtime with the WebGPU plugin provider (`onnxruntime-ep-webgpu`) | one profiled run |
| `tflite:interp` | LiteRT `Interpreter(model)`: the XNNPACK delegate applies by default | delegate partitions in the execution plan |
| `tflite:ref` | LiteRT `Interpreter` without default delegates (builtin kernels) | builtin only |
| `tflite:cpu` | LiteRT `CompiledModel`, CPU accelerator | `is_fully_accelerated`, then one compile per operator |
| `tflite:gpu` | LiteRT `CompiledModel`, GPU accelerator (WebGPU on Direct3D 12 here), its default precision | as tflite:cpu |
| `tflite:gpu-f32` | as tflite:gpu with `enforce_f32` | as tflite:cpu |
| `pte:cpu` | ExecuTorch runtime (`executorch.runtime`) | kernel calls and delegate calls in the program |
| `tsx:reference`, `tsx:cpu`, `tsx:cuda` | Truestar's IR reference interpreter or compiled executor | one executor |

Runtime names use the file format (`tflite`, `pte`, `tsx`), and `ort` as the existing stack
backends do. A runtime that cannot run here is refused with the reason (TS621) and nothing else
runs in its place. A file whose reader is not installed (`onnx` for .onnx, `ai-edge-litert` for
.tflite, `executorch` for .pte) is refused with TS022 and the pip command, exit 2. Core ML
(`.mlpackage`, `.mlmodel`) is refused with TS620: only Apple's Core ML
framework runs it, on macOS and iOS; coremltools 9.0 has no Windows wheel (pip builds it from
source) and its `predict()` needs macOS. LiteRT's GPU accelerator on Windows needs `dxcompiler.dll`
(without it the device fails with Windows error 87): Truestar loads the one from `TRUESTAR_DXC_DIR`,
next to the ONNX Runtime WebGPU plugin, or on PATH, and refuses `tflite:gpu` when there is none.

## How a runtime is judged

1. **The verdict comes from the unmodified engine.** The file runs as users run it, on every
   sample. Each output is compared with the model in float64: relative L2 error against a
   threshold calibrated on a float32 PyTorch run (the threshold `truestar check` uses), elements
   beyond the derived per element bound (docs/TOLERANCE.md) are counted, and per row decisions are
   compared with near ties separated.
2. **Two verdicts, so the converter and the runtime are told apart.** The file's own math is
   evaluated in float64: the ONNX reference evaluator on a float64 copy of the graph, Truestar's
   float64 rules for the TFLite operators it lists (`tflite_graph._EVAL`), the IR reference
   interpreter for `.tsx`. *Conversion verdict*: the file in float64 against the model in float64.
   *Runtime verdict*: the runtime against the file in float64. The conversion bar is what float32
   rounding of the model itself moves it (constants written in float32, a scalar the model holds in
   float64), never below 1e-7. A file whose float64 math is off is a finding (TS625) even when the
   float32 output stays under its limit: the gelu tanh rows below are 1.2e-4 against a 1e-4 limit
   at a glance, and 4,427 of 5,120 elements are beyond the derived bound. When an op has no float64
   rule, the report says so (TS638) instead of guessing.
   A quantized TFLite file (full integer: QUANTIZE, int8 FULLY_CONNECTED with per channel weights,
   DEQUANTIZE) is read in real numbers: each quantized tensor is its codes times its scale (per
   channel along `quantizedDimension`), each quantized op's exact result is put on its output grid
   (round half away from zero, clamped), and values cross into and out of the runtime and its cuts
   as real numbers. The quantization is then the file's own math (TS625, cause conversion), and the
   runtime verdict says whether the runtime computes that integer arithmetic: on a tiny generated
   model LiteRT matches it to 4e-8 and a cut equals it exactly
   (`tests/test_conform.py::test_int8_per_channel_tflite_conforms_as_quantized_not_as_a_runtime_fault`).
   The float32 limit still applies to the outputs, so an INT8 file is reported divergent from the
   float model by its quantization error; a quantization aware limit is not built.
3. **Precision.** A runtime whose error against the file is what a correct lower precision
   implementation shows (every op rounded to float16, or matmul inputs rounded to TF32, in a
   float64 run of the model) is reported as computing in that precision (TS627), not as a bug.
   Decision flips are still counted. ONNX Runtime CUDA's `use_tf32` is read from the session and
   judged with the TF32 tolerance, as the stack backends do.
4. **Residency.** Where every node ran, from what each runtime exposes (table above). Nodes off the
   requested device are named with their op and module (TS628); a runtime that cannot say is TS629.
   LiteRT `CompiledModel` only says "not fully accelerated"; Truestar then compiles each operator
   alone and names the ones that do not accelerate. For `.pte` files, which keep no module names,
   kernel call number k of an op is matched to call number k of that op in the model's own
   `torch.export` graph.
5. **Localization only after a mismatch, on cuts.** Each module whose name the file keeps (ONNX
   `pkg.torch.onnx.name_scopes` metadata from the dynamo exporter or `/a/b/Op` node names from the
   TorchScript exporter; TFLite tensor names `<module path>/<name>`) is cut out of the file as a
   model of its own (onnx.utils.Extractor; Truestar's TFLite cut), run in the same runtime with the
   same options on that module's float64 reference inputs, and compared with its float64 reference
   outputs. Its error is its own, nothing inherited, and the run that is judged is never
   instrumented. Descent over the module tree (children in order, first one at fault, repeat) names
   the deepest module at fault and whether the file or the runtime is wrong there. A runtime shown
   to compute in float16 is judged per module as float16, so the descent looks for what float16
   does not explain. Tensors are matched to the module's inputs and outputs by shape, values only
   breaking ties (a faulty file's values differ by definition).
6. **Observer guard.** Every instrumented run (an ONNX Runtime profiled session, LiteRT
   `experimental_preserve_all_tensors`) is compared with two plain runs. If it differs from the
   plain run by more than the two plain runs differ from each other, TS630 marks what it shows per
   node as untrusted, and residency read from it says so.

## Golden bundle (`.tsgold`)

A zip of `.npy` arrays and a `manifest.json`, readable with numpy alone: inputs per sample in the
file's input dtype, float64 expected outputs, the threshold per output and the derived per element
bound, every runtime's outputs with its fingerprint, and for each mapped module (up to 32, 64 MiB)
its float64 reference inputs and outputs and each runtime's output of the module cut alone. It
holds `check_outputs.py`, a numpy only checker for a team that runs the file on its own device:

```
python check_outputs.py model.tflite.tsgold outputs.npz    # s{k}_{j}: output j of sample k
```

Entries are written in a fixed order with a fixed timestamp, so the same content gives the same
bytes. Every file's sha256 is in the manifest; a damaged or foreign bundle is refused (TS632).

## Upgrade baseline (`truestar conform replay`)

`replay` needs no model code. It reruns the artifact in each recorded runtime on the bundle's
inputs, checks against the float64 expected outputs, compares bit for bit with what the same
runtime produced when the bundle was written, and diffs the runtime fingerprints with recdiff's
classification (`truestar.recdiff.fingerprint.runtime_fingerprint`: Python, OS, numpy, GPU, driver,
plus `runtime.*` keys: the runtime's package versions, ONNX Runtime provider package and plugin
versions, the WebGPU adapter, the LiteRT GPU precision setting and a hash of the DXC compiler it
loaded). When outputs moved, each recorded module cut is rerun on its stored inputs, and the first
changed module with no changed module inside it is named. Verdicts: identical (TS639), changed but
still conformant (TS634), regressed (TS633). A different artifact is refused (TS635): replay
compares runtimes on the same file.

## Measured (2026-11-24, RTX 5070 Ti shared with other jobs, Ryzen 9 9900X)

Model: two residual blocks (Linear, gelu, LayerNorm) and a 10 way head, random weights, batch 64,
8 samples (512 decisions). INT8 model: Linear, fake INT8 quantize (round half to even, clamp),
Linear, quantize, head. Faults are edits of the file itself (tests/conform_models.py). Baseline:
one input, `numpy.allclose(atol=1e-5, rtol=1e-5)`. cv venv: torch 2.11, onnxruntime-gpu 1.29.0,
onnxruntime-ep-webgpu 0.6.0, ai-edge-litert 2.4.0, executorch 1.2.0. Output of
`scripts/conform_study.py`:

| artifact | runtime | allclose flags | truestar | codes | blamed (cause) | off target nodes | s |
|---|---|---|---|---|---|---|---|
| onnx clean | ort:cpu | no | ok | - | - | - | 0.06 |
| onnx clean | ort:cuda | yes | ok tf32 | - | - | - | 0.14 |
| onnx clean | ort:webgpu | no | ok | - | - | - | 0.51 |
| onnx gelu as tanh (exporter decomposition) | ort:cpu | yes | suspect | TS623,TS625 | blocks.0 (conversion) | - | 0.17 |
| onnx gelu as tanh (exporter decomposition) | ort:cuda | yes | suspect tf32 | TS625,TS627 | blocks.0 (conversion) | - | 0.21 |
| onnx gelu as tanh (exporter decomposition) | ort:webgpu | yes | suspect | TS623,TS625 | blocks.0 (conversion) | - | 1.29 |
| onnx Det node (no GPU kernel) | ort:cpu | no | ok | - | - | - | 0.01 |
| onnx Det node (no GPU kernel) | ort:cuda | yes | ok tf32 | TS628 | - | Det@head | 0.07 |
| onnx Det node (no GPU kernel) | ort:webgpu | no | ok | TS628 | - | Det@head | 0.35 |
| onnx INT8 fake quant clean | ort:cpu | no | ok | - | - | - | 0.01 |
| onnx INT8 fake quant clean | ort:cuda | yes | suspect tf32 | TS627 | - | - | 0.34 |
| onnx INT8 fake quant clean | ort:webgpu | no | ok | - | - | - | 0.19 |
| onnx INT8 round to floor in q2 | ort:cpu | yes | divergent | TS622,TS624,TS625 | q2 (conversion) | - | 0.12 |
| onnx INT8 round to floor in q2 | ort:cuda | yes | divergent tf32 | TS624,TS625,TS627 | q2 (conversion) | - | 0.22 |
| onnx INT8 round to floor in q2 | ort:webgpu | yes | divergent | TS622,TS624,TS625 | q2 (conversion) | - | 0.91 |
| tflite clean | tflite:interp | no | ok | - | - | - | 0.21 |
| tflite clean | tflite:ref | no | ok | - | - | - | 0.20 |
| tflite clean | tflite:cpu | no | ok | - | - | - | 0.23 |
| tflite clean | tflite:gpu | yes | divergent float16 | TS624,TS627 | - | - | 1.03 |
| tflite clean | tflite:gpu-f32 | no | ok | - | - | - | 0.34 |
| tflite cumsum (outside XNNPACK) | tflite:interp | no | ok | TS628 | - | CUMSUM@top | 0.18 |
| tflite cumsum (outside XNNPACK) | tflite:ref | no | ok | - | - | - | 0.20 |
| tflite cumsum (outside XNNPACK) | tflite:cpu | no | ok | TS628 | - | CUMSUM@top | 5.76 |
| tflite cumsum (outside XNNPACK) | tflite:gpu | yes | suspect float16 | TS627 | - | - | 0.93 |
| tflite cumsum (outside XNNPACK) | tflite:gpu-f32 | no | ok | - | - | - | 0.28 |
| tflite layer norm eps outside sqrt | tflite:interp | yes | suspect | TS623,TS625 | blocks.0.ln (conversion) | - | 1.34 |
| tflite layer norm eps outside sqrt | tflite:ref | yes | suspect | TS623,TS625 | blocks.0.ln (conversion) | - | 1.54 |
| tflite layer norm eps outside sqrt | tflite:cpu | yes | suspect | TS623,TS625 | blocks.0.ln (conversion) | - | 1.22 |
| tflite layer norm eps outside sqrt | tflite:gpu | yes | suspect float16 | TS625,TS627 | blocks.0.ln (conversion) | - | 1.33 |
| tflite layer norm eps outside sqrt | tflite:gpu-f32 | yes | suspect | TS623,TS625 | blocks.0.ln (conversion) | - | 1.42 |
| tflite gelu as tanh | tflite:interp | yes | suspect | TS623,TS625 | blocks.0 (conversion) | - | 1.76 |
| tflite gelu as tanh | tflite:ref | yes | suspect | TS623,TS625 | blocks.0 (conversion) | - | 1.34 |
| tflite gelu as tanh | tflite:cpu | yes | suspect | TS623,TS625 | blocks.0 (conversion) | - | 1.15 |
| tflite gelu as tanh | tflite:gpu | yes | suspect float16 | TS625,TS627 | blocks.0 (conversion) | - | 1.20 |
| tflite gelu as tanh | tflite:gpu-f32 | yes | suspect | TS623,TS625 | blocks.0 (conversion) | - | 1.27 |
| pte XNNPACK, cumsum model | pte:cpu | no | ok | TS628 | - | aten::native_layer_norm.out@blocks.0.ln, aten::native_layer_norm.out@blocks.1.ln, aten::cumsum.out@top | 0.04 |

What the rows show:

* **Faults in the file are named at the module, with the cause.** gelu decomposed as tanh: module
  `blocks.0` (the Block whose forward calls gelu), conversion, on all 8 runtimes that ran it.
  Layer norm with eps outside the square root: `blocks.0.ln`, conversion. INT8 quantize step
  rounding toward minus infinity: `q2`, conversion, decisions flipped. The baseline flags these
  files too and names nothing.
* **The baseline's alarms on clean files are explained, not repeated.** ONNX Runtime CUDA on the
  clean file: allclose fails (TF32 is ORT's CUDA default), conform reports ok under the TF32
  tolerance. LiteRT GPU on the clean file: allclose fails; conform says the GPU computes the float32
  file in float16 (TS627, error 1.6e-3 against 5.7e-4 for a correct float16 run), counts the flipped
  decision, and `tflite:gpu-f32` is ok, so the remedy is one option. ONNX Runtime CUDA on the clean
  INT8 file is suspect with TS627: its default TF32 moves quantization levels (output error 1.6e-2),
  which is what a correct TF32 implementation of this model shows (1.9e-2); every module alone is
  within the TF32 limit.
* **Fallback the baseline cannot see.** `Det` on the CPU in both ONNX Runtime GPU sessions, named
  with its module `head`; `CUMSUM` outside the XNNPACK delegate in LiteRT, from the execution plan
  (`tflite:interp`) and from per operator compiles (`tflite:cpu`); in the ExecuTorch file, both
  layer norms (`blocks.0.ln`, `blocks.1.ln`) and the cumsum run on portable kernels outside
  XNNPACK. In every one of these the outputs pass allclose.
* **Observer guard.** The ONNX Runtime profiled sessions and LiteRT's preserved tensors gave
  outputs identical to the plain runs on every row (trusted). An injected change of 1e-3 in the
  instrumented run is flagged TS630 (tests/test_conform.py).
* **Upgrade, real.** A bundle written with onnxruntime 1.20.1 (ml venv, Python 3.10) replayed with
  onnxruntime-gpu 1.29.0 (cv venv, Python 3.13) on the same file: outputs changed (max 7.2e-7,
  relative 1.6e-7, 0 decisions moved), still conformant (TS634); the first changed module is
  `blocks.0.ln`, the LayerNormalization kernel, while every Gemm module is bit for bit identical;
  the fingerprint diff lists `runtime.onnxruntime 1.20.1 -> 1.29.0`, the provider package, numpy
  and Python. Replayed in the venv that wrote it: identical (TS639). Injected: a 1% change with a
  changed version is TS633 naming `runtime.onnxruntime` and the first module.
* **No false alarms**: clean files of three models (16 samples each) on every CPU runtime of their
  format are ok with zero flips (tests/test_conform.py::test_no_false_alarms_on_clean_files), and
  every clean row above is ok except the float16 and TF32 explanations.

Cost: conform on these files takes 0.3 s (ExecuTorch) to 3 s (ONNX, three runtimes) and 10 to
14 s for five LiteRT runtimes, most of it compiling GPU engines and, for `tflite:cpu` when it is not
fully accelerated, one compile per operator (5 s for 29 operators).

## LiteRT NPU (`tflite:npu`, research item 6.2, ledger P56)

`--runtimes tflite:npu` opens the file with LiteRT's one NPU interface for every vendor
(`CompiledModel`, `HardwareAccelerator.NPU | CPU`, the way LiteRT's NPU docs write it) instead of a
vendor SDK. LiteRT falls back to the CPU or GPU by itself, and LiteRT's Intel NPU page warns that
`is_fully_accelerated()` alone is not proof: "if the dispatch library never loaded, ops were fully
offloaded to XNNPACK/CPU, not the NPU". So residency comes from LiteRT's own log, captured while the
environment and the model are created:

* no compiler plugin partition and no dispatch: TS1125, every operator listed off the NPU (TS628),
  with LiteRT's reason (NPU accelerator not registered, plugin library that did not load);
* `Partitioned subgraph<i>, selected N ops, from a total of M ops` with N < M: TS1124, and the
  operators left out are named by compiling each operator alone;
* an ahead of time compiled file (operators replaced by the `DISPATCH_OP` custom op): TS1126 lists
  the dispatch partitions and the operators outside them from the file alone, no NPU needed; if
  the dispatch library does not load the whole file is TS1125.

Measured on this PC (ai-edge-litert 2.4.0, Windows, Ryzen 9 9900X, no NPU), Linear, ReLU, Linear:
the wheel ships only the Intel OpenVINO compiler and dispatch plugins; the NPU accelerator did not
register (`kLiteRtStatusErrorNotFound`) and the plugin DLL did not load (it links openvino.dll,
which is not installed). LiteRT then ran all 3 operators on XNNPACK, max relative error 1.4e-7,
while `is_fully_accelerated()` returned True. Conform reports TS1125 and TS628 for `tflite:npu` and
nothing for `tflite:cpu` on the same file. LiteRT's Intel page also says "Only the NPU device is
supported through the OpenVINO dispatch path", so installing OpenVINO would not give an NPU here.

Not run: a real NPU (Qualcomm, MediaTek, Google Tensor, Samsung, Intel Lunar or Panther Lake) and a
real AOT file. AOT compiling needs the vendor SDK wheels (`ai-edge-litert-sdk-intel-nightly` pins a
nightly OpenVINO and comes with `ai-edge-litert-nightly`, which would replace the stable
ai-edge-litert other checks use), so the AOT reader is tested on a file with operators swapped for
`DISPATCH_OP` (tests/test_litert_npu.py). An AOT file made elsewhere (AI Hub, a Linux host) is read
the same way.
## Limits

* The TFLite files in the tests come from Truestar's own small writer (`tflite_build.py`, from the
  published schema), since no PyTorch to TFLite converter runs here without another torch. Files
  from other converters run the same way, but module mapping needs module paths in tensor names
  and the float64 meaning covers only the operators in `tflite_graph._EVAL` (else TS638).
* `.pte`: no cuts and no float64 meaning (the program keeps neither module names nor a portable
  float64 form), so a wrong `.pte` output is not localized below the outputs (TS631); residency is
  read from the program. ETDump intermediate outputs were not used.
* `.tsx`: the reference interpreter is the meaning; modules are not mapped.
* Not run: the published litert-torch#718 and transformers.js#1728 files (no model downloads in
  this task), DirectML, Core ML (macOS only), LiteRT NPU accelerators, remote targets (AI Hub), ORT
  versions spanning a known fixed bug (the real upgrade above is two installed versions with no
  known bug between them). LiteRT writes its own log lines to stderr; stdout stays one document.
* Inputs are positional tensors; `get_kwargs()` is refused.
