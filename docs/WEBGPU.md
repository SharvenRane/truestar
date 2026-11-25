# WebGPU backend (`truestar check --backends webgpu`)

Author: Sharven Rane. First written 2026-11-24.

Browser and native WebGPU runs give wrong answers that users find by hand, in production, by
comparing with WASM (onnxruntime#28976, #29651, transformers.js#1728; gap research item 4).
Truestar runs the model on ONNX Runtime's WebGPU execution provider (Dawn, the WebGPU
implementation inside Chrome) on this machine, without a browser, and reports it like any other
backend: final output and decisions against the float64 CPU reference, the first module whose own
error crosses its threshold, the ONNX node inside it, and repros.

Code: `src/truestar/webgpu/` (`onnx_check.py` the check, `worker.py` and `client.py` the worker
process, `dawn.py` single op probes). Tests: `tests/test_webgpu.py`. Script:
`scripts/webgpu_f16_matmul_repro.py`.

## Use

```
pip install onnx onnxruntime onnxruntime-ep-webgpu      # plus wgpudawn to name the adapter
truestar check model.py --backends cpu,webgpu
truestar check model.py --backends webgpu,webgpu:vulkan,webgpu:d3d12:1
truestar check model.py --backends "webgpu+enableMatmulFp32Accumulation=1"
```

| backend name | runs on |
|---|---|
| `webgpu` | the platform's default Dawn backend (D3D12 on Windows, Metal on macOS, Vulkan on Linux), adapter 0 |
| `webgpu:d3d12`, `webgpu:vulkan`, `webgpu:metal` | that Dawn backend |
| `webgpu:d3d12:1` | adapter 1 in Dawn's list for that backend (here: 0 RTX 5070 Ti, 1 Radeon integrated GPU, 2 Microsoft Basic Render Driver, a CPU rasterizer) |
| `...+key=value` | a WebGPU EP option, passed unchanged (case kept), several allowed |

WebGPU is opt in: it never joins the default backend set, because it exports the model. The
report names the adapter, the Dawn backend, the ONNX Runtime and EP versions and the EP options.
A software adapter (Basic Render Driver) is labelled as not a GPU.

The worker process needs only numpy, onnx and onnxruntime with the plugin; `TRUESTAR_WEBGPU_PYTHON`
points it at another Python, so the torch environment does not need WebGPU packages.

## What one backend run does, and why

| step | what | why (the redesign) |
|---|---|---|
| export | whole model to ONNX in the model dtype (TorchScript exporter, opset 17; the torch.export exporter when it refuses an op, as for GPT2's `aten::diff`) | the graph that ships to the browser is the graph that is checked |
| plain run | twice on WebGPU (run to run difference is TS565) | the final output and decisions come only from the uninstrumented run |
| control | the same graph on ONNX Runtime's CPU EP, only when WebGPU is off | separates the export and ONNX Runtime's own kernels (TS563) from WebGPU; skipped when clean because the float16 CPU kernels cost 20 s of a 22 s check |
| residency | a profiled run gives the EP of every node | ONNX Runtime falls back to its CPU EP without raising; TS560 when the whole session did (never reported as WebGPU), TS561 naming compute nodes left on the CPU; integer shape nodes on the host are counted separately |
| observer guard | the profiled run and a tapped run (every module output added as a graph output) must give bit identical final outputs to the plain run | taps stop ONNX Runtime's fusions; when they change the output, per module numbers from the tapped run are marked untrusted (TS564) and every module is replayed in isolation instead |
| blame | with trusted taps, only modules whose tapped output is over threshold are exported alone with the float64 reference inputs and run on WebGPU, children before parents | a clean model with trusted taps needs no extra exports; the blamed module's error is its own |
| op drilldown | each ONNX node of the blamed module alone on WebGPU, fed the CPU EP's values, compared with the same node in float64 on the CPU EP | names the node, not only the module |
| repros | `repro.py` (module alone, float64 expected) and `op_repro.py` (node alone), each with its `.onnx`, inputs and expected `.npz`; `python repro.py cpu` is the control on the CPU EP | needs numpy and onnxruntime only, no torch, no model code |

Each backend runs in its own worker process: ONNX Runtime's WebGPU EP owns one Dawn device per
process, and a device loss or driver crash then ends the worker, not the check. Workers close at
the end of each check.

## Problems found while building it (each guarded by a test)

* **Silent CPU fallback in ONNX Runtime.** Any failure to create the WebGPU EP (an invalid option,
  no adapter) prints "EP Error ... Falling back to ['CPUExecutionProvider']" and builds the session
  anyway. Passing both devices that `ort.get_ep_devices()` lists for WebGPU (it lists it twice on
  this machine) does the same. Truestar passes one device and checks the providers of every
  session (`test_real_bad_ep_option_is_unavailable_not_cpu`, `test_session_fallback_is_never_reported_as_webgpu`).
* **Vulkan needs the loader preloaded.** With onnxruntime-ep-webgpu 0.6.0 on Windows 11,
  `dawnBackendType=Vulkan` fails with "Failed to get a WebGPU adapter: No supported adapters"
  (then falls back to CPU) unless `vulkan-1.dll` is already loaded in the process. With it loaded
  both Vulkan adapters appear. The worker loads it first.
* **The exporter flips batch norm into training mode.** `torch.onnx.export` restores the mode of
  the module it was given; a fresh wrapper module is in training mode, so every batch norm of the
  user's model came back in training mode (ResNet18: 20 modules looked up to 33% wrong). The
  wrapper now copies the model's mode and every submodule's mode is restored.
* **In place modules overwrote the feeds.** Replaying `SiLU(inplace=True)` alone fed the graph
  `silu(x)` instead of `x` (EfficientNet B0 looked 67% wrong, MobileNetV3 Hardswish 75%). Feeds
  are copied before anything runs.

## Measurements (2026-11-24, this machine)

Windows 11, RTX 5070 Ti and AMD Radeon integrated GPU, onnxruntime 1.29.0, onnxruntime-ep-webgpu
0.6.0, wgpudawn 0.33.0.0, torch 2.11. Random init models built in code, no downloads. Error is
relative L2 against float64; check time includes the worker start (about 1 to 3 s).

| model | dtype | backend | final output error | result | check time (s) |
|---|---|---|---|---|---|
| MLP 16-32-4 | f32 | webgpu (D3D12, RTX 5070 Ti) | 1.1e-7 | OK | 3.4 |
| ResNet18, batch 2 | f32 | webgpu | 3.3e-7 | OK, TS564, 69 modules replayed | 14.4 |
| ResNet18 | f32 | webgpu:d3d12:1 (Radeon) | 5.0e-7 | OK, TS564 | 8.3 (WebGPU part) |
| ResNet18 | f32 | webgpu:d3d12:2 (Basic Render Driver) | 3.2e-7 | OK, TS564, labelled not a GPU | 6.5 (WebGPU part) |
| ResNet18 | f32 | webgpu:vulkan (RTX 5070 Ti) | 3.3e-7 | OK, TS564 | 5.5 (WebGPU part) |
| ResNet18, batch 4 | f16 | webgpu, webgpu:vulkan | 1.9e-3 | OK, taps trusted | 26 each |
| ResNet18, batch 4 | f16 | webgpu:d3d12:1 | 4.3e-3 | OK | 22 |
| MobileNetV3 small | f32 | webgpu, webgpu:d3d12:1 | 4.9e-7, 7.8e-7 | OK, TS564, 157 modules replayed | 37 for both |
| EfficientNet B0 | f32 | webgpu | 1.1e-6 | OK, TS564, 249 modules replayed | 20.7 |
| TransformerEncoder 4 layers, width 256 | f32 | webgpu, webgpu:d3d12:1 | 2.3e-7 | OK, taps trusted | 27 for both |
| TransformerEncoder | f16 | webgpu | 1.9e-3 | OK | 6.6 (was 49.8 before the control became lazy) |
| BERT 4 layers, width 256 | f32 | webgpu, webgpu:d3d12:1 | 2.6e-7, 3.4e-7 | OK | 21.8 for both |
| BERT | f16 | webgpu, webgpu:vulkan, webgpu:d3d12:1 | 2.0e-3, 1.9e-3, 2.0e-3 | OK | 32.8 for all three |
| GPT2 4 layers, width 256 | f32 | webgpu | 8.5e-7 | OK, TS564, torch.export exporter | 20.7 |
| ViT B/16 | f32 | webgpu | 0 | OK; the random init head is all zeros, so the output check is trivial here | 35.3 |

Residency: every node of every model above ran on WebGPU except one integer shape node in GPT2.
Taps changed the final output (TS564) on every CNN and on GPT2, by 1.1e-7 to 2.1e-6 relative, never
on BERT or the encoder: per layer dumps of a CNN on WebGPU describe a different computation from
the one that ships, which is why blame never uses them unchecked.

## Findings (real divergences on this GPU; nothing was sent anywhere)

### 1. Float16 MatMul accumulates in float16 for some shapes (ONNX Runtime WebGPU EP default)

`scripts/webgpu_f16_matmul_repro.py`: one MatMul node, positive float16 inputs.

| M | K | N | D3D12 RTX 5070 Ti | Vulkan RTX 5070 Ti | with `enableMatmulFp32Accumulation=1` |
|---|---|---|---|---|---|
| 1 | 4096 | 64 | 1.7e-4 | 1.7e-4 | 1.7e-4 |
| 1 | 8192 | 64 | 1.7e-4 | 1.7e-4 | |
| 1 | 8193 | 64 | **2.2e-1** | **2.2e-1** | |
| 1 | 12288 | 64 | **3.3e-1** | **3.3e-1** | |
| 1 | 16384 | 64 | **5.0e-1** | **5.0e-1** | 2.1e-4 |
| 8 | 8193 | 64 | **2.2e-1** | **2.2e-1** | |
| 64 | 4096 | 64 | **7.3e-2** | 1.7e-3 | 2.3e-4 |
| 64 | 16384 | 64 | **5.0e-1** | 3.7e-3 | 2.1e-4 |

The Radeon (D3D12) shows the same pattern. Above K 8192 for few rows, and for 64 rows on D3D12, the
kernel keeps a float16 running sum, which stops growing once the sum reaches the range where
float16 spacing exceeds the addends (2048 for values below 1): half of the sum is lost at K 16384.
Normal (signed) inputs give 1e-2 to 1.7e-2 instead. The shape decides the kernel and the Dawn
backend decides the shape cutoff, so a model that is fine at one sequence length or batch is wrong at
another. CUDA float16 (float32 accumulation) gives 2.4e-4 on the same layer.

Truestar finds it from the model alone: a float16 feed forward block (LayerNorm, Linear 1024 to
4096, GELU, Linear 4096 to 1024, 64 tokens) is SUSPECT on `webgpu` with `down#0` (Linear) blamed at
own error 9.3e-3 against a threshold of 5e-3, node `onnx::MatMul` named (float64 reference), and
the op repro exits 1 on WebGPU and 0 on the CPU EP; CUDA is OK at 4.8e-4. With
`webgpu+enableMatmulFp32Accumulation=1` the same check is OK at 3.2e-4, and the agent JSON offers
exactly that rerun (`try_ep_option`).

ONNX Runtime's own CPU EP is off on that layer too (6.6e-3, a fused float16 MatMul and Add), so a
user comparing WebGPU with WASM by hand sees two wrong answers that roughly agree. Truestar compares
both with float64 and reports the CPU EP part separately (TS563).

Reductions are not affected: float16 ReduceSum, Softmax and LayerNormalization stay at 1.7e-4 to
1.2e-3 up to length 16384 (a float16 running sum would give 7.5e-1 for the sum).

### 2. The same WGSL shader rounds differently on D3D12 and Vulkan (Dawn direct probes)

`python -m truestar.webgpu.dawn --backend Vulkan --adapter 0 --dtype f32` runs textbook shaders
and compares each with float64 and with a step by step emulation in the same precision:

| probe | D3D12, RTX 5070 Ti | Vulkan, RTX 5070 Ti | Vulkan, Radeon |
|---|---|---|---|
| f32 matmul `s = s + a*b`, K 1024 | 100% equal to separate rounding | 13.3% separate, **100% fused multiply add** | 100% fused multiply add |
| f16 matmul, f16 accumulator | no `shader-f16` (see below) | 100% fused multiply add | 100% fused multiply add |
| f16 storage, f32 accumulator, `f16(s)` at the end | | 100% round to nearest even | **100% round toward zero** |
| f32 and f16 sums (sequential, 256 lane tree) | 100% equal | 100% equal | 100% equal |
| softmax (exp) | within 8e-8 of numpy (exp ULPs) | same | same |

So one shader gives different bits per backend on the same GPU (contraction to a fused multiply add
on Vulkan only), and the Radeon converts float32 to float16 by truncation where the RTX rounds to
nearest. WGSL permits both, so the probe reports them as explained (`explained_by`), not as TS566.
The truncation shows up through ONNX Runtime too: the Radeon's float16 MatMul output matches
float32 accumulation then round toward zero on 99.8% of elements, and its error is twice the RTX's
(4.6e-4 against 2.4e-4 on the dot product check).

Float16 availability: Dawn reports no `shader-f16` for the RTX on D3D12 through wgpudawn (no DXC
next to it; Dawn forces `use_dxc` off) and none on Vulkan unless the toggle
`vulkan_enable_f16_on_nvidia` is set, which the probe sets. ONNX Runtime's EP ships its own
`dxcompiler.dll` and runs float16 graphs on D3D12 with every node on WebGPU; whether each
program computes in float16 internally is not visible from outside, which is what finding 1 measures
instead.

## Interface with `sim:webgpu-f16` (devsim, another branch)

`sim:webgpu-f16` simulates a WebGPU float16 device on the CPU; `webgpu` runs the real one. Both are
names for `--backends` and both are compared with the same float64 reference, so once both
branches are merged one check can list them side by side (not runnable on this branch). The simulation
assumes float16 accumulation in reductions and matmuls (marked a hypothesis in its profile) with
flush to zero. Measured here on ONNX Runtime 1.29 and EP 0.6.0: MatMul accumulates in float16 only
for the shapes in finding 1, reductions, softmax and layer norm do not, and the f16 conversion
rounding direction depends on the GPU (finding 2). The real backend is the ground truth for
calibrating that profile; this branch does not edit it.

## WebNN (`webnn`, `webnn:cpu|gpu|npu`, research item 6.3, ledger P56)

WebNN (W3C Candidate Recommendation Draft, 2026-10-21) is a browser API, so the backend drives a
real browser, headless: Playwright starts installed Chrome (`TRUESTAR_WEBNN_CHANNEL=msedge` for Edge)
with `headless=True` and `--enable-features=WebMachineLearningNeuralNetwork`; no window is shown.
The page is served from a routed https origin inside the browser (WebNN needs a secure context),
with no network traffic.

The model goes through the same ONNX path as WebGPU (export, plain run twice, residency, taps,
isolated replay against float64, op drilldown, repro). Truestar's own translator
(`truestar.webnn.translate`) turns each ONNX node into one `MLGraphBuilder` call, so every node is
a WebNN op and nothing is left to another runtime: a node with no rule is TS1127 and nothing runs.
Translated: Add, Sub, Mul, Div, Pow, Max, Min, MatMul, Gemm, Relu, Sigmoid, Tanh, Erf, Sqrt, Exp,
Log, Abs, Neg, Reciprocal, HardSwish, HardSigmoid, LeakyRelu, Elu, Softmax, LayerNormalization,
ReduceMean, Transpose, Reshape, Flatten, Squeeze, Unsqueeze, Conv, MaxPool, AveragePool,
GlobalAveragePool, BatchNormalization, Concat, Clip, Cast and a few more; float32 graphs only (WebNN
has no float64). The CPU control runs the ONNX graph on ONNX Runtime's CPU EP in the parent.

Measured (Chrome 158.0.8127.38 headless, Windows 11 build 26200, RTX 5070 Ti, 2026-11-24):

| model | webnn:cpu | webnn:gpu | webnn:npu |
|---|---|---|---|
| Linear, GELU, LayerNorm, Linear, Softmax (batch 8) | ok, 1.7e-7 | ok, 1.3e-7 | ok, 1.3e-7 |
| Conv, BatchNorm, ReLU, MaxPool, grouped Conv, Hardswish, mean, Linear | ok, 3.9e-8 | ok, 3.2e-8 | ok, 4.2e-8 |

Every module's own error was under 1e-7 on all three device types. Findings on the way:

* Naming one operand as two graph outputs (a tapped graph where a module's output is also the
  model's output) made Chrome lose the WebNN context at `build` ("Context is lost"); the page now
  gives the second name an identity, and a lost context is made again and the graph rerun once
  (reported in the notes). Regression test: tests/test_webnn.py.
* On the CNN, the tapped graph changed the final output by 1e-8 to 6e-8 on every device type
  (TS564), so taps were untrusted and every module was replayed alone. On gpu and npu the tapped
  BatchNorm output was off by a relative 1.0 while the same BatchNorm run alone was at 4e-8: the
  extra output, not the op, is what the browser computed wrongly. Cause inside Chrome not
  investigated.
* ReLU of a negative value returns -0.0 on gpu and npu and +0.0 on cpu (seen in the first probe):
  equal as numbers, different bits.

`webnn:npu` ran here, but this PC has no NPU. WebNN's device selection notes say an NPU request
falls back to the GPU selection when there is no NPU, so these npu rows are almost certainly the
GPU path. The page cannot tell which native stack (Windows ML EP, LiteRT, Core ML) the browser
picked; the report names the device type and the browser version.

## Limits and what was not verified

* Only this Windows machine: an RTX 5070 Ti and a Radeon integrated GPU, D3D12 and Vulkan. No
  Metal, no Intel GPU, no Linux, no browser. Issues specific to Apple M series (onnxruntime#29130)
  or Iris Xe (#29807) cannot reproduce here.
* onnxruntime#28976 (float16 index math above 2048) and #29651 (Conv with more than 3 input
  channels) were not rediscovered: float16 ResNet18 at 224 px with 3 and 64 channel convolutions was
  clean; inputs above 2048 px were not run.
* A module whose output matches in a trusted tapped run is not replayed alone; an own error
  cancelled by later layers would be missed there (the torch backends replay every module).
* The op drilldown runs at most 300 nodes alone and skips nodes with subgraphs (If, Loop).
* Residency comes from ONNX Runtime's profile; nodes fused away by ONNX Runtime are counted under
  the fused node's name.
* Export: models ONNX cannot express are reported as TS562 and not checked. Float16 export traces
  on the CPU, which is slow for convolutions (7.7 s for ResNet18).
* The observer guard demands bit identical final outputs, so on CNNs it always falls back to full
  isolated replay (10 to 11 s of a 13 to 18 s check).
