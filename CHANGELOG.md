# Changelog

All notable changes are listed here. Versions follow semantic versioning
(docs/QUALITY_BAR.md, never introduce #9). Numbers come from the dated status log in
[docs/QUALITY_BAR.md](docs/QUALITY_BAR.md).

## Unreleased

## 0.1.1 (2026-11-25)

- `truestar conform serving` now says which package it needs: it requires `safetensors`, available as
  the new `serving` extra (`pip install "truestar[serving]"`). 0.1.0 failed with a bare import error.
- Tests that need optional packages (transformers, safetensors, ai-edge-litert, flatbuffers) skip when
  those are absent, so the suite passes on a plain install.
- README: new header and badges.


* Lean 4 proofs under proofs/ of the exact tensor core window, fixed order determinism and the
  digit split, with a generator for the constants they use and a test that checks the generated
  constants match the code (skipped without lake).
* Training features: Lion, RMSprop and Adagrad as native optimizers for train_step; step,
  multistep, exponential, polynomial, cosine restarts, one cycle and warmup schedules; BCE (logits
  and probabilities), L1, smooth L1, Huber, NLL, KL divergence and cosine similarity losses with
  log1p and softplus; truestar.optim.EMA inside the compiled step; gradient clipping by value and
  by 1 or inf norm; Xavier, Kaiming and truncated normal initializers with per parameter rules;
  truestar.nn summary and parameter counts; truestar.value_and_grad with has_aux.
* Compiled mixed precision training: compile(..., mixed=True) and train_step under autocast keep
  float32 islands for master weights, optimizer state and the loss, judged end to end by bars
  derived for the declared precision (TS1480, near ties TS1481, ill conditioned outputs TS1482), and keyed
  stochastic rounding for bfloat16 weights; L4 measurements in docs/TRAINING.md.
* Multi GPU: the fixed order scatter sum by all to all is the default (rank order fold then all
  gather, ring all reduce bytes, bitwise equal to the gather fold), gradient sync uses persistent
  padded bucket buffers donated to the reduction with a multi tensor write back, the exact tree
  leaf runs as a Triton kernel on CUDA (bitwise equal to the torch leaf, 29x faster at 1024 cubed
  on an A10G), and scripts/dp_scaling.py compares against torch DDP, FSDP2 and tensor parallel at
  1, 2 and 4 GPUs (docs/PARALLEL.md).
* Decode speed: fused MLA latent attention is the default compiled MLA decode (a shape it does not
  take is TS1460), and 4 bit KV caches run inside the compiled decode.
* Training step speed: a fused attention backward op (attention_grad), and CUDA graph replay of
  training steps that index with embeddings, the indices checked on every replay; the profiled 94M
  decoder step in docs/TRAINING.md.
* Convolution training speed: conv training kernels read contiguous GEMM operands (run time
  weights and small images relaid by shared copy kernels, stride 2 input gradients split by output
  parity), the relu gradient uses a scalar zero and a global pool's gradient is an expand instead
  of two matmuls, and the fusion pass undoes split points it no longer needs; resnet18 batch 32
  strict float32 step 41.2 to 24.4 ms on an RTX 5070 Ti with the same bits on every call
  (docs/COMPILER.md); test_remat budgets recalibrated.
* Matmul and decode speed with the same bits: constant float32 linear weights are prepacked as [K,
  N] at compile time, strict float32 tiles on 48 to 69 SM GPUs, a 16 row unsplit decode tile,
  deterministic K chunks of 512 fixed by K alone for K above 1024 (kernel CODEGEN_VERSION 7, so
  cached kernels rebuild), grouped query heads in one tile for quantized decode, shape only chunks
  that split short contexts; tests/data/contraction_defaults_70sm.json regenerated.
* opbounds.bound_softmax soundness fix: both exp errors count at 2 u per ulp and the sum at gamma
  over n - 1 roundings. A bound cross check against FPTaylor (pointwise first order worst case on
  a small kernel corpus, optimizer lower bounds that separate a definite BELOW from its gap) with
  a cached fixture, so CI checks it without FPTaylor installed; results in docs/TOLERANCE.md.
* truestar kernel held out evaluation (docs/KERNEL_CHECK.md): false pass and false fail rates on
  KernelBench and METR kernels with the protocol written before any run; NaN handling (TS202, now
  a warning when every finite element is right) and nondeterminism are their own verdicts, the
  lean verdict is on by default (TS1300) with reduced precision the source asks for named apart
  (TS1301); the derived tolerance uses a running error bound as the sound tier and the
  probabilistic bound only where its assumption holds; a wrong integer output is reported, and out
  of memory in a check raises instead of skipping a variant.
* Training step verification: the first call's end to end check judges training steps by the error
  bound derived from the program (compile(..., end_to_end='auto'|'bar'|'derived')), graphed in
  place training steps included; a training verification benchmark with fair configs and a Llama
  step.
* Packaging and compiler fixes: sympy is a declared dependency, truestar mcp serves both the mcp
  1.x and 2.x SDK server APIs (the <2 pin is gone), the stray file 0 is removed; float64 islands
  start only where the model declares float64 (train step lr and step inputs stay float32);
  constant tables are stored in their declared dtype, so float32 programs holding one export.
* torch 2.15 graphs import: torch.clamp's gradient at its bounds follows the installed torch
  (measured from its autograd), aten._assert_async checked at import (TS860 when it depends on an
  input); pyproject declares the tested range torch>=2.11,<2.16.
* Real Apple M2 results (docs/METAL.md section 5): Metal contraction and attention kernels, order
  free sums, round once math, linear attention and orderprobe on Metal; no combined float64 copy
  off a device (MPS gave silent zeros); truestar conform coreml judges placement and numerics of
  saved Core ML runs (TS1160 to TS1168, P61, docs/COREML.md).
* Portable tensor cores: compile(..., portable_tensor_cores=True) gives float16, bfloat16 and TF32
  dot products the same bits on Ampere, Ada, Hopper and Blackwell (A10G, L4 and H100 measured
  equal), 2.1x to 3.4x faster than float32 without tensor cores (TS1330 to TS1333,
  docs/PORTABLE_TC.md).
* truestar export --to onnx, litert, coreml and executorch: the written file is run and verified
  in its own runtime, Core ML included on macOS (TS1410 to TS1419, docs/EXPORT.md).
* truestar serve: an OpenAI compatible LLM server whose answers do not change with batching, the
  prefix cache or chunked prefill, with truestar serve --check (TS1400 to TS1409, docs/SERVE.md).
* LoRA and QLoRA fine tuning (truestar.lora, docs/LORA.md): adapters on a Llama decoder, adapter
  files, merges and quantized base weights checked by code (TS1370 to TS1377).
* truestar.data: deterministic, resumable loaders (array, folder and file sources, sharding,
  workers) and checkpoint format v2 with an exact resume of the loader position and RNG (TS1380 to
  TS1389, docs/DATA.md).
* Native from_pretrained: truestar.models loads local Hugging Face decoder checkpoints into
  Truestar modules without transformers (TS1390 to TS1397, docs/MODELS.md).
* Dynamic shapes: one compile per declared dim range, checks per bucket, optional proof of bitwise
  equality with the static compile (TS1420 to TS1427, docs/DYNAMIC_SHAPES.md).
* Persistent verified compile cache: a warm new process reuses the verified program record and is
  accepted only on the recorded kernel binaries and output bits (TS1350 to TS1354,
  docs/COMPILER.md).
* Mixture of experts: Mixtral, Qwen3 MoE, OLMoE and DeepSeek V3 import, compile and train
  (`truestar.moe`, docs/MOE.md); near tie router report (TS1180 to TS1184).
* Multi head latent attention (`truestar.mla`, docs/MLA.md): DeepSeek V3 imports (torch.empty as
  NaN), latent KV cache prefill and decode with an absorbed decode proven in float64, TS1200 to
  TS1203.
* `truestar conform llamacpp` judges an official llama.cpp build against float64 per format and
  setting (TS1310 to TS1319).
* `truestar conform vllm` and `sglang`: GPU serving engines against float64 and against themselves
  (TS1320 to TS1329).
* Ordinary compiled CUDA calls replay from a verified CUDA graph from the second call at a
  signature (opt out with `graphs="off"` or `TRUESTAR_GRAPHS=off`).
* Speculative decoding bitwise equal to greedy decoding (`truestar.specdec`, TS705, TS706).
* Float64 islands for rotary angles in float32 programs (`Cfg(rope_f64=True)`).
* `sim:hexagon`, fitted to real Snapdragon NPU outputs (TS1140, TS1141).
* Mamba 2 SSD and Gated DeltaNet families lowered at import onto `linear_attention` (TS1150).
* Text, retrieval, speech and vision language coverage; image generation, detection and
  segmentation coverage; torchvision detectors refused by name (TS1260).
* Tree tensor parallel for any tp = W * 2^a; unequal data parallel batches and uneven reduce
  scatters refused (TS1210, TS1211).
* `scripts/audit_codes.py`: audit table of every TS code (raised, documented, tested, message grade).
* Clearer refusals: TS020 (model file functions), TS021 (fingerprint compare, conform get_kwargs),
  TS1022 (rotation steps), TS680 (ranks mode), TS821 (process group backend), TS1080
  (order_free_sum_cuda on a non CUDA tensor).
* Docs: Switching to Truestar (docs/SWITCHING.md) and a cookbook of ten CPU recipes (docs/cookbook/,
  examples/cookbook/), each run by tests/test_cookbook.py.
* Fixed: `memory_plan()` on programs with a gather (cross entropy) raised AttributeError.
* Fixed: spurious NumPy RuntimeWarning from the kernel check when both sides hold the same infinity.
* Fixed: FutureWarning from `TreeSpec.children_specs` on every export with torch 2.11.
* `pip install "truestar[mcp]"` now installs mcp 1.x; with mcp 2.x present `truestar mcp` serves the
  builtin transport instead of crashing at start.
* Coverage refresh to transformers 5.24: 28 current families (Gemma 2, 3, 3n, 4, Qwen3, Phi-4 mini,
  SmolLM3, OLMo 2 and 3, Granite, Cohere 2, GLM 4 and others, plus current vision encoders) import,
  compile and train equal to torch; the ops they need (`diff`, `floor`, `bucketize`, `std`, string
  conv padding and more); mixture of experts, chunked linear attention and chunked SSM models are
  refused by name with TS940, TS941 and TS942 (docs/MODEL_COVERAGE.md).
* `truestar.linattn.linear_attention`: chunked linear attention and state space kernels (Mamba 2
  SSD, Gated DeltaNet, lightning attention; Kimi Delta Attention and gated linear attention in the
  chunked PyTorch form), checked against a float64 sequential recurrence, deterministic across SM
  counts, with `chunk_invariance` and `chain_error` reports; codes TS1060 to TS1063; `truestar codes`
  sorts four digit codes by number (docs/LINEAR_ATTENTION.md).
* `truestar.assert_close(fn, inputs, out)` and `truestar.close_report`: test assertions with a
  tolerance derived per element from float64, a correctly rounded run and an error model of the
  dtype and device policy; no rtol or atol. pytest fixtures in `truestar.pytest_plugin`
  (docs/TOLERANCE.md).
* `truestar kernel ref.py cand.py` and the MCP tool `verify_kernel`: a kernel against its reference
  over shape, value, stride, dtype and repeat variants; codes TS200 to TS206 (docs/KERNEL_CHECK.md).
* `@truestar.verified_autotune`: Triton autotuning that checks every config against float64 and
  removes wrong ones (TS207).
* `truestar check --backward`: a parent whose float16 gradients are off only through ReLU masks
  flipping at near ties is SUSPECT with code TS185 and next action `accept_near_tie`, instead of a
  false TS180.

## 0.1.0 (2026-11-25)

First public release, on PyPI 2026-11-25. Built 2026-11-23 to 2026-11-24.

### Commands

* `truestar check model.py`: runs a PyTorch model on every backend on the machine (CPU, CUDA, ROCm,
  MPS, XPU), compares each with a float64 CPU reference, replays every module alone with reference
  inputs, names the first divergent module and ATen op, and writes a repro that needs only PyTorch.
* `truestar check --backward`: the same per module replay for gradients, plus gradient norm drift
  over training steps.
* `truestar record` and `truestar diff --replay`: store every module call with an environment
  fingerprint, replay it after an upgrade and classify each change as divergence, precision change
  (for example TF32) or noise.
* `truestar invariance`: does a row's result depend on the rest of its batch, with the op and flag
  involved.
* `truestar port`: does a second implementation compute the same model, with a weight inferred
  module map.
* `truestar export` and `truestar run`: `.tsx` artifacts written only when they match PyTorch at two
  batch sizes; `truestar run --executor` picks the reference interpreter or the compiled executor.
* `truestar fix`: verified remedies for a blamed op, and `--int8` precision plans in simulation.
* `truestar rlpair`: RL trainer versus sampler logprob mismatch, blamed to a module and op.
* `truestar backends`, `truestar codes`, `truestar mcp` (an MCP server with a tool per command).

### For AI agents

* `--format agent`: one versioned JSON document (`schema_version` 1) with stable `TS` codes, the
  blamed file and line, and `next_actions` with shell commands. Codes are listed in
  docs/AGENT_INTERFACE.md and by `truestar codes`.

### Libraries

* `@truestar.function`: typed shapes, loops and branches compiled from Python source.
* `truestar.torch_op`: native functions registered as PyTorch custom ops with no C++ toolchain.
* A shape typed IR with a `torch.export` importer, a float64 reference interpreter and reverse mode
  autodiff; a training stack on that IR.
* A loop level kernel IR with fusion, Triton (GPU) and numba (CPU) code generation, launch config
  autotuning and per kernel verification against float64 (on the kernel's own GPU for large kernels).
* `truestar.compiler`: the tensor IR lowered onto those kernels, with a compile report that lists
  every fallback and library kernel.
* `truestar.runtime`: CUDA graph capture with bitwise replay checks. `truestar.diag`: the layer
  behind a CUDA OOM and the rank behind a distributed hang.

### Added in the release candidate polish

* Code TS022 (`optional_dependency_missing`): the CPU kernel paths name the `cpu-kernels` extra
  when numba is missing instead of failing with a raw `ModuleNotFoundError` from a generated file.
  Custom ops keep their documented fallback to the reference interpreter, with TS022 in the
  warning.
* numpy is a declared dependency (the kernel layer imports it; a clean install with only torch
  failed to import `truestar.compiler`).
* Package metadata for an index release (classifiers, project URLs, readme content type, version
  read from `truestar.__version__`), a docs site config (`mkdocs.yml`, `.readthedocs.yaml`, a
  manual Pages workflow). Nothing was published.

### Known limits

* Not on PyPI. Requires Python 3.10 or newer (3.10 and 3.12 run in CI) and PyTorch 2.1 or newer.
* Tested on Windows with CPU and an RTX 5070 Ti, Linux CPU in CI, a Tesla T4, and hosted macOS
  runners for MPS. ROCm, Intel XPU and physical Apple hardware are not tested.
* The checkers replay modules in eval mode; training mode batch norm and dropout are not covered.
* The compiled executor is slower than eager on the MLP and the transformer layer in the benchmark
  (strict float32 contractions against cuBLAS).
* Each doc ends with a Limits section.
