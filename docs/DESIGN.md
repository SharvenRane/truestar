# Truestar design

Author: Sharven Rane
First written: 2026-11-23

## One line

PyTorch is fast on NVIDIA and risky everywhere else. Truestar compiles models into fast code for
any chip and proves every generated kernel matches a correct reference.

## The problem (evidence gathered 2026-11-23)

1. **Silent wrong answers off NVIDIA.** MPS matmul off by 10 to 30% depending on allocator state in
   torch 2.7 through 2.13 (pytorch#193487). The default fp32 SDPA backend is wrong on RX 9070 XT, and
   a model drops from 86% to 0% accuracy with no warning (pytorch#201476). ROCm on Windows returns
   zeros for basic ops (ROCm/TheRock#8379). Nobody owns this problem.
2. **torch.compile is unpredictable.** Graph breaks, guard driven recompiles, warmups that went from
   30 minutes to hours after an upgrade (pytorch#174704), compiles long enough to trip NCCL timeouts
   (pytorch#108971), and debugging that users call painful.
3. **Training to deployment is broken.** ONNX export bakes in Python branches and freezes batch size.
   Stock torchvision CNNs failed to export (pytorch#166163). TorchScript is deprecated and users say
   ExecuTorch is not ready to replace it.
4. **New layers are slow until someone hand writes a kernel.** FlexAttention calls this a software
   lottery. Its templates do not cover differential or gated attention (Flashlight, MLSys 2026).
5. **JAX has its own limits.** Tracing rejects Python control flow, recompiles on every new shape,
   compile time can blow up, and the library ecosystem is split.
6. **Failures have no attribution.** NCCL hangs and CUDA OOMs do not say which layer or rank caused them.
7. **Shape errors show up at runtime,** often deep into a long run.

Root cause for 2, 3, 5 and 7: today's frameworks recover the program by watching Python run (tracing
or guards). A whole program, ahead of time compiler with typed shapes removes these by construction.

## Who already works on what

| Gap | Existing effort | State |
|---|---|---|
| Attention variant kernels | FlexAttention, Helion, Flashlight | Templates, research |
| Hardware portability | Mojo / MAX | Kernel language, slow adoption |
| NCCL debugging | PyTorch Flight Recorder | After the fact |
| Static shapes | Pyrefly, jaxtyping | Bolted onto Python |
| Correctness across backends | none found yet | **Open** |
| One artifact, train to deploy | none complete | **Open** |
| Fast, predictable compiles | none | **Open** |

To confirm before building: NVIDIA Polygraphy (compares ONNX Runtime vs TensorRT), PyTorch
`opcheck` / OpInfo, ONNX checkers, and overlap with exportdiff.

Update 2026-11-24: Inductor, ONNX Runtime and TensorRT are now backends of `truestar check`
(docs/BACKENDS.md). Unlike Polygraphy, the engine judged is the one users build (no tensor marked
as an extra output); localization runs only after a mismatch, in separate per module engines, by
descent over the module tree.

## Device spectrum findings (2026-11-23, second pass)

* **Phones.** NNAPI deprecated in Android 15. LiteRT unifies Qualcomm, MediaTek, Google Tensor and
  Samsung NPUs. coremltools `torch.export` path is beta with about 70% op coverage. Qualcomm's own
  AIPC paper (Oct 2026) lists all zero outputs, NaN/Inf and drift after quantization as a core
  deployment problem that needs automated checks against reference outputs.
* **INT8 is not portable.** Same model, same scales: 958 to 965 of 1000 predictions agree across
  CPUs with different integer kernels, 939 of 1000 on a vendor NPU, while aggregate accuracy looks
  unchanged (arXiv 2609.16085). Only per input comparison exposes it.
* **Browser.** WebGPU shows up to 19.4x performance variation across devices; FP16 WebGPU flipped
  33 of 3079 labels versus a q8 reference; f16 accumulation produced incoherent LLM output on Apple
  M series.
* **LLMs across GPUs.** Greedy decoding is not reproducible across GPU types; reasoning benchmark
  accuracy varies up to 9% across runtime configurations.
* **PC and games.** DirectML is in sustained engineering. Microsoft is building DirectX Compute Graph
  Compiler (MLIR based, Windows ML 2.9.42 experimental, Nov 2026) with AMD, Intel, NVIDIA, Qualcomm.
  Shader level ML (Cooperative Vector, DX Linear Algebra, AMD MiniDXNN) targets tiny MLPs inside
  rendering. Console SDKs are under NDA; out of scope until partners exist.

## Existing per vendor checkers (competition for Stage 1)

* NVIDIA Polygraphy: ONNX Runtime vs TensorRT, per layer.
* Qualcomm QAIRT accuracy debugger: framework vs NPU/GPU/CPU, layerwise snooping, ONNX only for some modes.
* Apple Core AI Debugger: Core AI model vs PyTorch reference, per op similarity.
* ExecuTorch Inspector: AOT graph vs runtime intermediate outputs.
* SAY-5/export-validator: PyTorch vs ONNX Runtime per layer.

Each covers one vendor or one runtime. Gap: one vendor neutral tool across CUDA, ROCm, MPS, NPUs,
WebGPU and CPU, that measures per input decision flips (not only cosine or PSNR), bisects to the
first diverging op automatically, and emits a minimal repro. Long term the moat is the verified
compiler, not the checker.

## Architecture

1. **Front end (two paths).**
   * Import existing models from `torch.export` graphs.
   * A small native front end from day one: a Python decorator for writing one new layer or kernel
     with typed shapes (`Tensor[B, 512, f16]`, symbolic dims) and real `if` / loops.
2. **IR.** Shape typed tensor IR with value semantics and no aliasing. Autodiff as a compiler pass.
3. **Reference interpreter.** A slow, obviously correct kernel per op. Ground truth for everything.
4. **Optimizer.** Automatic fusion without templates. Halide style separation of algorithm and
   schedule, with an autotuner and a learned cost model.
5. **Backends.** CPU via LLVM, GPU via Triton first, later MLIR GPU dialects, WebGPU.
6. **Verification layer.** Every compiled kernel is checked against the reference automatically in
   test, optionally at runtime. Cross backend differential runs with first divergence reporting.
7. **Runtime.** Ahead of time compiled single binary. The trained artifact is the deployed artifact.

Hard requirement from the start: fast incremental compiles (XLA compile time grows superlinearly
with program size; do not inherit that).

## Build order

| Stage | Build | Proof |
|---|---|---|
| 1 (weeks 1 to 4) | `truestar check`: run a model on cuda, rocm, mps, cpu; compare layer by layer; report the first diverging op; emit a minimal repro | Rediscovers pytorch#193487 and pytorch#201476 automatically |
| 2 (months 2 to 3) | Reference interpreter and shape typed IR, `torch.export` importer | Matches PyTorch CPU on a model zoo |
| 3 (months 3 to 6) | Compiler: CPU via llvmlite, GPU via Triton, fusion, autotune, verified kernels | nanoGPT trains; a FlexAttention inexpressible variant runs fused |
| 4 (month 6+) | Native language growth, single binary deploy to laptop, phone, browser | Same trained model runs on all three |

Each stage is useful on its own.

## Rules for this project

* Built only on personal hardware, personal accounts, personal time.
* No code, data or models from any employer.
* Framed as general purpose ML infrastructure.
* Private until the Stage 1 demo works. Then open source under Apache 2.0.
