# Compilers and runtimes as check backends

`truestar check` compares a model against a float64 CPU reference on every device it can find.
The same check runs a model through a compiler or a runtime: torch.compile with Inductor, ONNX
Runtime and TensorRT become backends in the same report, with the same module blame, op drilldown
and repro.

```
truestar check model.py --backends cuda,cuda+inductor,ort:cpu,ort:cuda,trt:fp32,trt:fp16
```

Code: `src/truestar/stacks.py` (discovery, runners, ONNX export, ORT placement),
`src/truestar/stackcheck.py` (verdict, localization, op drilldown), `src/truestar/stackrepro.py`
(repros). Tests: `tests/test_stacks.py`, models and injected faults in `tests/stack_models.py`.
Measurements: `scripts/stack_study.py`.

## Names

| backend | what runs |
|---|---|
| `cpu+inductor`, `cuda+inductor` | `torch.compile(model)` with Inductor, default mode, on that device; TF32 follows torch's flags |
| `rocm+inductor` | the same on an AMD GPU; needs a ROCm build of torch, refused with the reason otherwise (docs/ROCM.md) |
| `ort:cpu` | `torch.onnx.export` (dynamo exporter, the default), ONNX Runtime with the CPU provider |
| `ort:cuda` | same export, providers `[CUDA, CPU]` as users write them; ORT's `use_tf32` is read from the session |
| `trt:fp32` | same export, TensorRT engine with TF32 off (strict float32) |
| `trt:tf32` | TensorRT's defaults: float32 with TF32 on |
| `trt:fp16` | FP16 enabled, TF32 off; the tolerance is float16's |
| `metal` | Truestar's own compiler: `torch.export` into the Truestar IR, then the compiled executor with Metal kernels on the MPS device (docs/METAL.md); macOS only; ops without a kernel lowering are reported as TS400 |

Stacks mix freely with devices in one `--backends` list. A stack that cannot run here is refused
with the reason (`trt:fp32: tensorrt is not installed`, `cpu+inductor: ... needs a C++ compiler
(cl from Visual Studio Build Tools) on PATH`), never skipped silently.

## How a stack is checked

**The verdict comes from the real engine.** The whole model is compiled the way a user compiles
it: default flags, the graph's own outputs, nothing marked for observation. Its output is compared
with the float64 reference. When it matches, the check ends there, so a clean model costs one
compile and one run on top of the reference.

Polygraphy, the usual TensorRT accuracy tool, checks per layer by marking every tensor as an
engine output (`--trt-outputs mark all`). A marked tensor must be written to memory, so TensorRT
cannot fuse through it, and the engine that gets checked is not the engine that ships. Truestar
never changes the engine it judges.

**Localization happens only after a mismatch, in separate engines, by descent over the module
tree.** Each child of the current module, in execution order, is compiled alone by the same stack
and fed its float64 reference inputs (cast to the model dtype), so its error is its own and not
inherited. The first child over threshold becomes the current module and the descent repeats. It
stops at a module whose children all match alone: that module is blamed. Cost: the children of
each module on the path to the fault, up to the first bad one, instead of one compile per module
(the Inductor fault below needed 9 module compiles in a model with 13 modules, the ORT fault 5).
Same module and input shapes share one engine, and TensorRT shares one builder and one timing
cache for the whole check (a small engine build drops from about 15 s to under 1 s here, and the
per module engines pick the tactics the whole engine timed).

If the whole model diverges but every top level module compiled alone matches, the fault needs
the modules compiled together, fusion across a module boundary, or lives in code between modules:
code TS402.

**Op drilldown, again in separate engines.** Each ATen op of the blamed module (traced in the
float64 reference) is compiled alone by the stack with its reference inputs; the first over
threshold is named. An op with several outputs that the stack cannot return (TensorRT's
LayerNormalization has no mean and rstd outputs) is retried returning its first output. The same
op is then run eagerly on the device, in the precision the stack computes in, and the report says
which side is at fault:

* eager matches: the fault is in the stack's code for this op (a miscompile);
* eager in float16 is wrong too, for `trt:fp16` on a float32 model: float16 cannot hold the op's
  values, keep it in float32;
* eager in the model dtype is wrong too: the device kernel is at fault.

**Inductor: from fused kernel to op.** Inductor writes the ATen ops each generated kernel came
from (`Original ATen: [...]`). Truestar reads that from the blamed module's compiled code and
names the kernel that computes the blamed op, for example `aten.sigmoid.default` in
`triton_poi_fused_add_addmm_mul_sigmoid_1 (aten.addmm, aten.sigmoid, aten.mul, aten.add)`. Dynamo's
recompile limit is raised and set to raise when hit, so no module quietly runs eager code while
counted as compiled; real graph breaks are reported (TS403) since the user's run has them too.

**ONNX Runtime: silent CPU fallback.** With `[CUDA, CPU]` providers, ORT places nodes the CUDA
provider lacks on the CPU and says nothing beyond a memcpy warning. The whole model session is
profiled once (one run) and every node's provider is read from the profile. Nodes on the CPU that
produce floating point data are reported with their ONNX node and the PyTorch module they came
from (TS400); integer shape arithmetic, which ORT keeps on the CPU on purpose, is not. When the
CUDA provider does not load at all (missing CUDA or cuDNN libraries), `get_providers()` lacks it
and the whole model ran on the CPU (TS401).

**Tolerances** are the eager check's: per dtype floors, never below 10x the error a correct CPU
run in the same precision shows on the same module or op. The precision is the one the stack
computes in (float16 for `trt:fp16`). For a stack below the model's precision, a CPU calibration
run that is itself off by more than the precision's floor is not trusted: it hit the same range
limit (float16 overflow), and using it would excuse the fault. TF32 tolerance applies only when the
stack runs TF32 and only to modules and ops with a matmul or convolution.

## Faults caught (scripts/stack_study.py faults)

Each row injects one real fault into one stack, or uses a model that hits a real limit of it. The
check names the module and op in one run.

| backend | fault | result | blamed module | blamed op | check s | compiles |
|---|---|---|---|---|---|---|
| cuda+inductor | Inductor's sigmoid lowering returns 1.05 x sigmoid | suspect TS011 | blocks.0#0 | aten.sigmoid.default in triton_poi_fused_add_addmm_mul_sigmoid_1 | 7.22 | 15 |
| ort:cpu | exporter translates gelu as the tanh approximation | suspect TS011 | act1#0 | aten.gelu.default | 4.76 | 7 |
| ort:cuda | Det has no CUDA kernel: silent CPU fallback | suspect TS400 | (node Det in module det) | - | 1.00 | 1 |
| trt:fp32 | LayerNorm layers forced to FP16 in an FP32 engine | divergent TS010 TS013 | norm#0 | aten.native_layer_norm.default | 16.44 | 5 |
| trt:fp16 | hand written RMSNorm, `x.pow(2)` overflows float16 | divergent TS010 TS013 | norm#0 | aten.pow.Tensor_Scalar | 11.48 | 4 |

Compiles include the whole model engine and the single op engines. The Inductor and ORT faults
are small (5% and 2e-4 relative) and land as suspect; the TensorRT ones change predictions.

## Overhead and false alarms (scripts/stack_study.py clean)

Random weights, torchvision resnet18 at batch 4 x 3 x 224 x 224 and a `TransformerEncoderLayer`
(width 256, 8 heads, batch first) at 4 x 64 x 256, float32, RTX 5070 Ti shared with other jobs.
`check` is the whole `truestar check` run; `stack alone` is export or `torch.compile`, engine
build and one run with the same code; both warm (each run twice, the second kept).

| model | backend | status | output rel err | check s | stack alone s | overhead s |
|---|---|---|---|---|---|---|
| resnet18 | cuda+inductor | ok | 3.5e-04 | 1.88 | 0.47 | +1.40 |
| resnet18 | ort:cpu | ok | 3.2e-07 | 3.85 | 1.43 | +2.42 |
| resnet18 | ort:cuda | ok | 4.5e-04 | 4.45 | 2.40 | +2.04 |
| resnet18 | trt:fp32 | ok | 4.6e-07 | 11.96 | 10.14 | +1.82 |
| resnet18 | trt:fp16 | ok | 1.2e-03 | 23.41 | 22.59 | +0.82 |
| TransformerEncoderLayer | cuda+inductor | ok | 1.2e-07 | 0.44 | 0.56 | -0.11 |
| TransformerEncoderLayer | ort:cpu | ok | 1.5e-07 | 1.49 | 1.26 | +0.23 |
| TransformerEncoderLayer | ort:cuda | ok | 9.7e-05 | 1.31 | 1.33 | -0.02 |
| TransformerEncoderLayer | trt:fp32 | ok | 1.2e-07 | 3.35 | 3.02 | +0.33 |
| TransformerEncoderLayer | trt:fp16 | ok | 4.8e-04 | 13.79 | 4.49 | +9.30 |

Zero false alarms in 10 runs. On a clean model the overhead is the float64 reference run and the
comparison: 0.8 s to 2.4 s on resnet18, under 0.4 s on the encoder layer. The +9.30 s row is
TensorRT build time varying on the shared GPU: rerun alone it measured 5.60 s against 5.29 s
(+0.30 s). Negative values are timing noise. The Inductor and ORT CUDA errors near 4e-4 on
resnet18 are TF32 convolutions (cuDNN TF32 is on by default in torch, `use_tf32` is on by default
in ORT CUDA) and are within the TF32 tolerance.

## Repros

A blamed module gets `repro.py` and a named op gets `op_repro.py`, each running the saved inputs
through the same stack (`python repro.py`) or eager PyTorch on CPU as the control (`python repro.py
eager`). They need PyTorch plus that stack. Test hooks set through `stack_options` (the injected
faults above) are not carried into the script; the report says so when they were set.

## Windows ML execution providers (`winml:<ep>`, research item 6.1, ledger P56)

`--backends winml:nvtrtx`, `winml:webgpu`, `winml:dml`, `winml:openvino`, `winml:qnn`,
`winml:vitisai`, `winml:amdgpu`, `winml:migraphx` (optionally `:<device index>` and `+key=value` EP
options). Each runs the shared ONNX path of the WebGPU backend (docs/WEBGPU.md): export, a plain
run twice, residency from a profiled run, per module taps (the in model error of every module
against float64), isolated replay of the modules over threshold (their own error), op drilldown and
a repro script that needs only numpy and onnxruntime. A session that fell back to the CPU EP is
TS1120 and is never reported as the EP's result; CPU compute nodes are TS1121.

Where the EP comes from, in order: the Windows ML EP catalog when the EP is installed and ready;
otherwise the bring your own package Windows ML documents (`onnxruntime-ep-nv-tensorrt-rtx-cu13`,
`onnxruntime-ep-webgpu`); DirectML only as the built in EP of `onnxruntime-windowsml` or
`onnxruntime-directml`. The worker never downloads: `truestar.winml.ensure_ready("nvtrtx")` asks
the catalog to (it goes through Windows Update and installs system wide). When the EP runs inside an
ONNX Runtime other than the one Windows ML ships, the report says so (TS1123, info).
`TRUESTAR_WINML_PYTHON` picks the worker's Python.

Measured on this PC (Windows 11 build 26200, RTX 5070 Ti, driver 610.47, Ryzen 9 9900X, 2026-11-24),
`ts` env, onnxruntime-gpu 1.29.0, model Linear 64 to 128, GELU, LayerNorm, Linear 128 to 10,
batch 8, float32:

| backend | EP source | status | final output vs float64 | per layer own error |
|---|---|---|---|---|
| `winml:nvtrtx` | onnxruntime-ep-nv-tensorrt-rtx-cu13 0.5.0 | suspect (TS011) | 4.8e-4 | Linear 2.79e-4 and 2.91e-4, GELU 5.2e-8, LayerNorm 5.5e-8 |
| `winml:webgpu` | onnxruntime-ep-webgpu 0.6.0 | ok | 2.0e-7 | every layer under 1.7e-7 in the model, none replayed |

The NvTensorRtRtx Linear error is TF32: torch on the same GPU and data gives 2.7906e-4 with TF32
on and 9.0e-8 with it off, and the EP gave 2.7906e-4. So NvTensorRtRtx computes float32 GEMMs with
TF32 by default, which Truestar reports as suspect against strict float32 (the module is named).
Residency: one fused NvTensorRtRtx node holds the whole graph, no CPU node.

Not run here, with the reason the probe gives: `winml:dml` (DirectML is built only into
onnxruntime-windowsml and onnxruntime-directml; either replaces the `onnxruntime` package of the env,
which would remove the CUDA, TensorRT and WebGPU providers that `ts` uses, so neither was
installed); `openvino`, `qnn`, `vitisai`, `amdgpu`, `migraphx` (the catalog does not offer them for
this hardware: no Intel, Qualcomm or AMD NPU, and the AMD iGPU gets no catalog EP). The catalog
offers NvTensorRtRtx and WebGPU here, both NotPresent; `ensure_ready` failed twice after 244 s
with HRESULT 0x800705B4 (timeout) and no progress callback, with Windows Update running, not
paused and no reboot pending, so the bring your own packages are what ran.

## Limits

* Stacks run in `truestar check` only. Other commands (`--backward`, `invariance`, `record`)
  refuse a stack name rather than run eager PyTorch under it, and `check(..., train=True)` refuses
  stacks since they compile the model in eval mode. Simulated backends (`sim:*`) and stacks mix in
  one `check`.
* Inductor on CPU needs a C++ compiler; this Windows machine has none, so `cpu+inductor` is
  refused here with that reason and was not run.
* `torch.compile` modes other than the default, ORT's TensorRT provider, ExecuTorch and other
  runtimes are not wired in.
* Localization needs each module to compile alone. A module the stack cannot export or compile
  alone is skipped and listed; the descent goes on with its siblings.
* The ORT placement check runs on the whole model session only (one profiled run).
* Polygraphy and the Inductor accuracy minifier were not run on the injected faults: the faults
  are injected in process (a patched lowering, a custom translation table, a builder hook) and both
  tools rebuild in their own process from files.
