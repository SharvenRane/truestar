# Truestar docs

Start with [GETTING_STARTED.md](GETTING_STARTED.md): one runnable example per command, on CPU.

Coming from PyTorch: [SWITCHING.md](SWITCHING.md) writes a training loop and an inference script in
Truestar's own API and brings an existing model in once through the importer. Short recipes, each
a script that runs on a CPU in under a minute: [the cookbook](cookbook/index.md).

## Commands

| doc | command | question it answers |
|---|---|---|
| [GETTING_STARTED.md](GETTING_STARTED.md) | `truestar check` | which module and op make a backend disagree with the float64 reference |
| [GRAD_CHECK.md](GRAD_CHECK.md) | `truestar check --backward` | which module's own backward is wrong, and does the gradient norm drift over steps |
| [RECORD_DIFF.md](RECORD_DIFF.md) | `truestar record`, `truestar diff` | what an upgrade of torch, CUDA, cuDNN, Triton or the driver changed, and whether a precision flag explains it |
| [INVARIANCE.md](INVARIANCE.md) | `truestar invariance` | does a row's result depend on the rest of its batch, and which op and flag are involved |
| [PORT_CHECK.md](PORT_CHECK.md) | `truestar port` | does a second implementation compute the same model, and where it does not |
| [CONFORM.md](CONFORM.md) | `truestar conform`, `truestar conform replay` | does the deployed .onnx, .tflite, .pte or .tsx file compute the model in its own runtime, where not, converter or runtime, which nodes left the device, and what an upgrade changed |
| [CONFORM_LLAMACPP.md](CONFORM_LLAMACPP.md) | `truestar conform llamacpp` | does the llama.cpp build you run compute the GGUF (or the checkpoint it came from) in every format and setting, engine error apart from quantization, near ties apart from bugs |
| [TPU.md](TPU.md) | `truestar conform tpu` | how a TPU (or any remote device) computes a fixed seeded battery against float64, judged from saved files |
| [PORT_ENGINES.md](PORT_ENGINES.md) | `truestar port ref.py gguf:m.gguf`, `jax:`, `keras:` | does llama.cpp, a JAX/Flax or a Keras port compute the same model; engine, quantization or settings |
| [EXPORT.md](EXPORT.md) | `truestar export`, `truestar run` | the verified `.tsx` artifact, its file format and the runner |

## For AI agents

* [../AGENTS.md](../AGENTS.md): how an agent runs Truestar and reads the result.
* [AGENT_INTERFACE.md](AGENT_INTERFACE.md): the versioned JSON schema, every `TS` code, the MCP tools.

## Libraries

These are Python APIs, used from code rather than from the command line.

* [FRONTEND.md](FRONTEND.md): `@truestar.function`, typed Python with loops and branches compiled from source.
* [CUSTOM_OPS.md](CUSTOM_OPS.md): `truestar.torch_op`, native functions registered as PyTorch custom ops with no C++ toolchain.
* [IR.md](IR.md): the shape typed IR, the torch.export importer and the float64 reference interpreter.
* [AUTODIFF.md](AUTODIFF.md): reverse mode autodiff as an IR to IR pass.
* [TRANSFORMS.md](TRANSFORMS.md): vmap, forward mode, jacfwd, jacrev, hessian and linearize as IR to IR passes.
* [LLM.md](LLM.md): decoder LLMs in the IR: sampling ops, dense and paged KV cache decoding.
* [MLA.md](MLA.md): `truestar.mla`, DeepSeek V3 multi head latent attention with a latent KV cache, the weight absorbed decode proven in float64, and the token sparse selection refusal (TS1202).
* [MODEL_COVERAGE.md](MODEL_COVERAGE.md): Hugging Face model families built from configs, through check, import, compile and export.
* [TRAINING.md](TRAINING.md): losses, optimizers, mixed precision and data parallel steps as IR programs.
* [TRAIN_NUMERICS.md](TRAIN_NUMERICS.md): Muon with a float64 orthogonality verdict, verified FP8 and NVFP4 training GEMMs, invariant sentinels, attention logit bounds.
* [KERNEL_IR.md](KERNEL_IR.md): the loop level kernel IR, fusion, Triton and numba code generation, per kernel verification.
* [COMPILER.md](COMPILER.md): the compiled executor, lowering the tensor IR onto Truestar kernels, with benchmarks; sentinels, compiled in checks (NaN, Inf, overflow, division by zero, bounds).
* [PROFILE.md](PROFILE.md): `truestar profile`, GPU time per compiled step mapped to IR ops and source lines.
* [DIAGNOSTICS.md](DIAGNOSTICS.md): `truestar.diag`, naming the layer behind a CUDA OOM and the rank behind a distributed hang.
* [NPU.md](NPU.md): INT8 agreement and per layer blame on phone NPUs through Qualcomm AI Hub (needs an AI Hub account).
* [FIX.md](FIX.md): `truestar fix`, verified remedies for a blamed op or layer, with before and after numbers and cost.
* [RANGE.md](RANGE.md): `truestar range`, where a model overflows or underflows in float16, bfloat16 or float8 before deployment, with a remedy verified by emulation.
* [PRECISION.md](PRECISION.md): `truestar compile --precision auto`, a precision per layer chosen by measured GPU time under a decision flip budget against float64, verified on held out inputs.
* [RL_PAIR.md](RL_PAIR.md): `truestar rlpair`, trainer versus sampler logprob mismatch in RL, blamed to a module and op.
* [LINEAR_ATTENTION.md](LINEAR_ATTENTION.md): `truestar.linattn`, chunked linear attention and Mamba 2 style SSD kernels checked against a float64 recurrence, with the chunk size invariance proof and the measured chain error.
* [RL_KERNELS.md](RL_KERNELS.md): `truestar.rlkernels`, one batch and position invariant kernel set for the RL trainer, prefill and decode, with bitwise equal logprobs.
* [RUNTIME.md](RUNTIME.md): host overhead measurements and CUDA graph capture.
* [QUANT.md](QUANT.md): `truestar.quant`, INT8, INT4, FP8, NVFP4 and MXFP4 matmul kernels with two verdicts: kernel bug against the exact dequantized float64 reference, and format loss from float64 references only.
* [ROTATION.md](ROTATION.md): `truestar.rotate`, Hadamard and learned rotations folded into Llama and Qwen2 weights and proven equal to the model in float64 before quantizing (a wrong placement is named, TS1020); measured with and without rotation on SmolLM2-360M.

## Background

* [DESIGN.md](DESIGN.md): the plan, the evidence behind it and the architecture.
* [QUALITY_BAR.md](QUALITY_BAR.md): the pain ledger, the rules every feature must pass, and the dated status log.
* [PAIN_RESEARCH_2.md](PAIN_RESEARCH_2.md): the second round of research into problems people report, with sources.
* [upstream/](upstream/): drafts of upstream bug reports, filed only after confirmation on real hardware.

Paths that start with `<research>/` (and `<repo>`, `<home>` in raw output) point to the author's unpublished research notes and raw files; they are kept so each claim stays traceable.
