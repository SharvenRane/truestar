# Quality bar

Owner rule (2026-11-23): every part of Truestar must solve a problem people actually hit, measurably
better than what they use today, and must not bring in a new problem. This file is how that rule is
enforced. It is checked before any feature is called done.

## 1. Pain ledger

Every feature maps to a row here. A feature with no row does not get built. A row is closed only when
its acceptance test passes against the named baseline, and the test stays in the suite forever.

| # | Pain people hit today | Evidence | Truestar answer | Acceptance test (vs baseline) | Status (from the status log) |
|---|---|---|---|---|---|
| P1 | Silent wrong results off NVIDIA | pytorch#193487, pytorch#201476, ROCm/TheRock#8379 | `truestar check` cross backend comparison | Rediscovers each listed bug from the user's model alone, names the op, emits a repro that reproduces on stock PyTorch | **Partly solved**: `truestar fix` swaps a blamed ATen op for a verified decomposition; module level device branches are only detected (TS093); the listed upstream bugs are not yet rediscovered on physical hardware |
| P2 | INT8 / FP16 prediction flips hidden by unchanged accuracy | arXiv 2609.16085, WebGPU label flip report | Per input decision flip counts, not only cosine or PSNR | Reports flips on a quantized model where aggregate accuracy is unchanged | **Partly solved**: solved in simulation by `truestar fix --int8`; on the phone NPU it is detected only |
| P3 | Divergence found only at the final output | export-validator README, Qualcomm accuracy debugger docs | Automatic bisection to the first diverging op | First bad op found in O(log n) runs on a model with an injected fault at a known layer | **Detected only**: names the first diverging module and op; the remedy comes from P1 |
| P4 | Each vendor has its own checker | Polygraphy, QAIRT, Core AI Debugger, ExecuTorch Inspector | One tool, one report format, every backend present on the machine | Same command and same report on CUDA, ROCm, MPS, CPU (and NPU / WebGPU when added) | **Partly solved**: one report on CUDA, CPU, hosted MPS, one hosted phone NPU and local WebGPU (ONNX Runtime WebGPU EP on D3D12 and Vulkan, two GPUs, docs/WEBGPU.md); ROCm, Metal WebGPU and browsers untested |
| P5 | Graph breaks and guard recompiles | pytorch#174704, forum threads | Whole program AOT compile, native control flow | Zero graph breaks by construction on models that break `torch.compile`; shape changes within declared symbolic dims never recompile | **Partly solved**: 0 compiles across batch 1, 7, 64 and loop lengths with compiled execution; Triton block size buckets for a symbolic reduced axis open |
| P6 | Compile time of minutes to hours | pytorch#108971, jax#33364 | Incremental compile, cache keyed on IR | Recompile after a one layer edit finishes in seconds; cold compile time tracked in CI and never regresses | **Partly solved**: recompile after an edit 0.17 to 0.84 s; slower than torch.compile on the transformer; not tracked in CI |
| P7 | Export bakes in branches, freezes batch, fails on stock CNNs | ONNX export writeup, pytorch#166163 | Export is compile: the trained artifact is the deployed artifact | Branchy model and dynamic batch behave identically after export; every torchvision classifier exports | **Partly solved**: dynamic batch survives export (unbounded since 2026-11-24); 62 of 63 torchvision classifiers; `torch.cond`, `while_loop` and `scan` export with both branch sides verified |
| P8 | New layers are slow without hand written kernels | FlexAttention, Flashlight papers | Template free fusion with verified kernels | A variant FlexAttention cannot express runs fused and within 20% of a hand written kernel | **Partly solved**: an attention variant FlexAttention cannot express fuses into one kernel; speed against a hand written kernel measured at the kernel layer only |
| P9 | Shape errors discovered deep into a run | jaxtyping, Pyrefly threads | Shapes are types, checked before running | Shape mismatch reported at compile time with the line and both shapes | **Solved**: front end and importer report shape errors at compile time with the op and both shapes |
| P10 | NCCL hangs and OOMs with no attribution | r/MachineLearning debugging thread, Flight Recorder post | Errors name the layer, rank and tensor | Injected hang and OOM are reported with layer name and rank | **Detected only**: CUDA OOM and gloo hangs attributed to layer and rank; NCCL untested; no remedy |
| P11 | CUDA version and install hell | r/CUDA threads | Pure pip install for `truestar check`, no toolkit needed | Fresh venv: `pip install truestar` then `truestar check` works on Windows, Linux, macOS | **Partly solved**: clean install proven in CI on Linux and Windows; on macOS 26.6 (Mac mini M2, 2026-11-25) a fresh venv with `pip install` of the source tree (torch 2.15.0 pulled in) ran `truestar check --backends cpu,mps,metal` in 4.6 s; not yet from PyPI |
| P12 | AI coding agents now do much ML debugging, and prose reports make them guess | Owner observation (2026-11-23); no external survey yet | `--format agent` versioned JSON, stable TS codes, `next_actions`, `truestar mcp` | An agent gets status, blamed module with file and line, repro command and next step from JSON alone; schema and MCP round trip tests in tests/test_agent.py | **Solved**: agent JSON, stable codes, `next_actions` and `truestar mcp` pass their round trip tests; not tried with a real agent client |
| P13 | Upgrades silently change numerics (N1) | docs/PAIN_RESEARCH_2.md | `truestar record` and `truestar diff` per module across environments | Injected op change and TF32 drift are named and classified; identical venvs report clean; baseline final logit `allclose` cannot name the op | **Detected only**: passing (tests/test_recdiff.py), injected op change and TF32 drift named and classified; two real torch versions not yet compared; no remedy |
| P14 | LLM outputs depend on batch, TP and GPU (N2) | docs/PAIN_RESEARCH_2.md | Invariance check plus batch invariant kernels as a verified property | Non invariant op named; Truestar kernels bitwise identical across batch 1, 2, 7, 64 at under 20% overhead | **Detected only**: checker met (`truestar invariance`); invariant reference ops are bitwise identical but 2.8x to 146x slower per op (tree linear up to 478x), so they do not count as a fix; compiled invariant kernels not built |
| P15 | Trainer and sampler disagree in RL (N3) | docs/PAIN_RESEARCH_2.md | `truestar rlpair`: pair check with per position module and op blame (docs/RL_PAIR.md), then one kernel set for prefill, decode and training forward | Bitwise equal logprobs at TP 1 and 2; swapped kernel named | **Partly solved**: `truestar.rlkernels` (docs/RL_KERNELS.md) gives bitwise equal per token logprobs for the training forward, prefill and token by token decode at batch 1, 3 and 8 with padding (float32, bfloat16), TP 1 equal to a one device TP 2 simulation, `truestar rlpair --kernels rl` verifies it and names a swapped kernel (TS185); costs 1.04x to 5.5x per op (training forward 2.0x in float32, 0.60x in bfloat16 against an eager math attention); real multi GPU TP, real engines and pretrained models not run |
| P16 | Ports and engines compute a different model (N4) | docs/PAIN_RESEARCH_2.md | `truestar port`: two implementation check with weight inferred module map (docs/PORT_CHECK.md) | Three injected port bugs named at the right module; correct port clean | **Detected only**: three injected port bugs named at the right module, correct port clean (tests/test_portcheck.py); no remedy |
| P17 | Custom ops need C++ builds tied to versions (N5) | docs/PAIN_RESEARCH_2.md | `@truestar.function` ops registered as torch ops, no toolchain | Under 2 s cold, opcheck passes, zero graph breaks, same source on two torch versions, Linux and Windows | **Partly solved**: registration under 2 s, opcheck, zero graph breaks and gradients on CPU and CUDA verified; first call not under 2 s; second torch version and Linux not run |
| P18 | Wrong gradients from trusted low precision kernels (N6) | docs/PAIN_RESEARCH_2.md | `truestar check --backward`: per module backward replay against float64 (docs/GRAD_CHECK.md) | Backward only fault blamed while forward check is OK; clean BF16 model OK | **Detected only**: passing on CPU and CUDA (tests/test_gradcheck.py); FP8 not tested; no remedy for kernels. Hand written backwards in the native front end are checked and refused at registration (`@fn.grad`, docs/AUTODIFF.md) |
| P19 | AI and hand written kernels pass a one input allclose and are still wrong; tolerances are guessed; autotune keeps a fast wrong config (R1, R6, R7) | <research>/PAIN_RESEARCH_3.md (YAKK, arXiv 2608.12700, executorch#13347, triton wrong result issues) | `truestar kernel` and MCP `verify_kernel` (docs/KERNEL_CHECK.md) on `truestar.assert_close` (derived tolerances, docs/TOLERANCE.md); `@truestar.verified_autotune` | No correct kernel rejected and more bugs caught than a one input allclose and `torch.testing.assert_close` defaults, on Truestar's correct and broken kernel set; wrong autotune config removed, under 20% tuning overhead | **Partly solved**: 0 of 69 correct runs rejected, 76 of 81 wrong runs caught (one input allclose at 1e-2: 47 of 81); a wrong autotune config removed at +5% overhead (tests/test_tolerance.py, test_kernelcheck.py, test_autotune.py); the YAKK and METR kernel sets not yet run |
| P20 | Same seed, other random numbers on another device; training mode forward never checked | PyTorch randomness notes ("not reproducible between CPU and GPU executions, even when using identical seeds"), pytorch#62467, #79496, #158398; checkers replayed in eval mode only | Keyed counter based random ops bitwise identical on every backend (truestar.rng, IR `random`, front end ops); training batch norm with state in and out; `--train` in check, grad_check and invariance (docs/RANDOM.md) | 1e8 draws per distribution bitwise equal on Triton and numba, 1e7 against the reference path, while PyTorch differs in every draw for one seed; KS, moments and independence tests; training only batch norm faults caught in training mode and invisible in eval; resnet18 training steps match torch float64 | **Solved** for keyed draws on this CPU and GPU (0 of 1e8 differ) and for training mode checks (5 injected faults caught, clean models OK); normal draws 1.9x slower than `torch.randn` on CUDA; MPS and ROCm not run |
| P21 | fp16 only runtimes (Neural Engine, WebGPU f16, TensorRT fp16) overflow quietly where the trained model never did (research round 3, R3) | onnxruntime#31626, PR #32294, onnxruntime#26732, executorch#15833, coremltools#2359 | `truestar range --target fp16`: float64 peaks of every value and hidden intermediate, static interval proofs, target emulation judged against a range unlimited twin, remedies verified by emulation (docs/RANGE.md) | bge small and Gemma 3 style overflows found on CPU at the right op against PyTorch's own float16 kernels, remedy brings cosine to at least 0.9999 under fp16 emulation, clean models clean; baseline: run in fp16 and compare final cosine, which cannot name the op | **Solved in emulation**: 6 of 6 causes found, 0 false positives on 9 model and target pairs; exact rescale verified (cosine 0.999996 and 1.000000); no real Neural Engine, WebGPU or TensorRT run |
| P22 | Compilers and runtimes silently change results, and their checkers perturb the engine or give up (R2, gap item 1) | PyTorch `module: correctness (silent)` label: 503 of 1,062 issues since 2025 also carry `oncall: pt2`; pytorch#179520, #173793, #188541; the torch.compile troubleshooting page says the minifier often does not work; Polygraphy marks every tensor as an output; ORT provider fallback is silent | Inductor, ONNX Runtime and TensorRT as `truestar check` backends (docs/BACKENDS.md): verdict from the unmodified engine, localization by module descent in separate engines only after a mismatch, op drilldown, Inductor kernel origins, ORT node placement | One injected fault per stack named at the right module and op in one run; zero false alarms on resnet18 and a TransformerEncoderLayer on every stack; overhead over the stack alone measured | **Detected only**: 5 of 5 faults named (Inductor miscompile, exporter fault on ORT CPU, ORT CUDA fallback, TensorRT FP32 layer precision, TensorRT FP16 overflow); 0 false alarms in 10 clean runs; pytorch issue repros in pinned torch versions, `cpu+inductor` (no C++ compiler here) and the minifier and Polygraphy on the same faults not run; no remedy |
| P29 | INT8 outputs differ by 1 LSB across libraries and NPUs and nobody can say which rounding each op used | CMSIS-NN#214, LiteRT#11747, docs/NPU.md (4.44% of S24 NPU conv1 elements) | `truestar intref`: exact integer reference per library convention, per op verdict, C golden vectors (docs/INTREF.md) | Names the convention per op on CMSIS-NN#214 style inputs where float fake quantization cannot; explains the NPU.md differences with zero new device jobs | **Partly solved**: per op verdicts on the cited cases and on ONNX Runtime kernels; NPU differences shown not to be rounding (float16 input quantization fits); TFLite, TFLM and boards not run, NPU per element match needs the quantized model |
| P28 | A model breaks on a device nobody on the team owns: float16 datapaths, vendor op formulations, size cliffs, NaN and inf rules | coremltools#2687, #2690, openvino#39703, #22613, onnxsim#1742, arXiv 2606.22283, arXiv 2511.11601 | `--backends sim:*` device numerics profiles with cited rules, `--sweep`, `truestar probe --specials` (docs/DEVICES.md) | Each readable cited case reproduced with the issue's own numbers; clean models clean on every profile; first failing size found by bisection | **Detected only**: 6 of 6 readable cases reproduced in simulation (tests/test_devsim.py), 0 device specific false positives on 6 clean models, no device confirmation, no remedy |
| P25 | Low precision kernel bugs hidden in format loss: FP8 and FP4 models score badly on one kernel path and nobody can tell the format's loss from a kernel bug | vllm#36094, vllm#64106, vllm#37030, triton#11633, AMD ROCm NVFP4 to MXFP4 blog (<research> PAIN_RESEARCH_3 R4, GAP_COMPETITORS 5) | `truestar.quant`: formats as exact float64 tables, quantized kernels, verdict (a) kernel against the exact dequantized reference with a diagnosis, verdict (b) format loss from float64 references only (docs/QUANT.md) | Injected scale index, group boundary, nibble order and accumulator bugs named at the right layer as TS460 while format loss is unchanged; correct kernels of every format pass; baseline: the quantized model against the float16 model | **Partly solved**: kernels for INT8, INT4, FP8 and NVFP4 / MXFP4 (native on sm_120) pass verdict (a) on every layer of a trained model; 11 injected bugs named; INT4 decode linears 3.5x Truestar float16; no CLI flag, one GPU, no published checkpoint decoded |
| P24 | vmap refuses data dependent control flow and random ops, and lacks batching rules; forward mode lacks per op rules | torch.func UX limitations page, pytorch#200456, `NotImplementedError` for forward AD of CPU flash attention (measured 2026-11-24), jax#2947, jax#8409 | `truestar.ir.vmap`, `jvp`, `jacfwd`, `jacrev`, `hessian`, `linearize` as IR passes (docs/TRANSFORMS.md) | vmap equals the per element loop on every op, including cond, while and random ops that torch.func refuses; every forward rule passes finite differences and the dot product test against the reverse rule; Hessian of a transformer layer matches torch.func | **Partly solved**: 301 vmap cases equal the per element loop (worst 3.4e-16), 102 forward programs pass finite differences, 89 pass the dot product test (worst 1.3e-14); a few forward rules missing (fmod, remainder, vector_norm, group_norm, rms_norm, adaptive_max_pool2d) |
| P23 | Peak memory unknown until the OOM; fragmentation; activation checkpointing that silently corrupts gradients (N8, pytorch#201158) | docs/PAIN_RESEARCH_2.md N8; GAP_PYTORCH_JAX R5, R6; GAP_COMPETITORS item 7 | Static memory planner with liveness, in place reuse and donation; `memory_budget` refused before the first launch (TS420); recompute under a budget, verified bitwise (TS421) (docs/MEMORY.md) | Planned peak equals measured `requested_bytes` on MLP, transformer, attention and training programs; an over budget program refused before launch naming the line; recompute gradients bit identical with measured peak under the budget, against torch.utils.checkpoint | **Partly solved**: exact on programs whose allocations Truestar owns (tests/test_memory.py); resnet18 off by cuDNN workspace (42 to 99 MiB); recompute meets 35% of the peak but stops above what torch.utils.checkpoint reaches; fragmentation not addressed (no static arena; the caching allocator still places blocks) |
| P27 | Compiled code hides where a NaN, an Inf or a zero division came from, and where GPU time goes (N11, N13) | `detect_anomaly` 1.4x to 2x per step (nantrace) and blind to forward NaN with a finite gradient; pytorch#192021 (profiler drops kernel events silently) | Compiled in checks (`CompiledExecutor(sentinels=...)`, docs/COMPILER.md, Sentinels) and `truestar profile` (docs/PROFILE.md) | Each class caught on an injected fault with op, line and first element; 0 false alarms on MLP, transformer, resnet18 with every class; overhead per class against `detect_anomaly` and no checks; per step times sum to the call within 5% and every launch accounted for (tests/test_sentinels.py, tests/test_profile.py) | **Partly solved**: every class named on injected faults, no false alarm on the three models; nan and div cost 0.8% to 15% of wall time with the deferred read against 7x to 12x for `detect_anomaly`, under the 3% target only on the MLP, resnet18 and its training step; profile within 2.5% of the call on three models, every launch accounted for; CPU overhead, ROCm and Nsight not measured |
| P34 | Parallel training (tensor, data, sequence parallel, gradient accumulation) silently computes another step than one device: wrong scaling, a missing sync, accumulation divided twice, a wrong split (research round 3, R5) | Megatron-LM#7452, #5974, #1754, pytorch#149768, #168092, #162692, verl#4710, transformers#34694, #35204, #37197, #35856, #40564 | `truestar parallel --layout tp=N,dp=N,ga=N,sp=0/1`: one step single device and under the layout on simulated ranks, both float64, pieces reassembled and compared in causal order, blame at a module or a collective (docs/PARALLEL.md) | Each injected bug class named at the right module or collective in one step; 0 false alarms on correct layouts; baseline: the step 1 loss, which names nothing | **Detected only, in simulation**: 10 of 10 injected faults named in one step (0.1 to 0.4 s on threads), the step 1 loss sees 4; 0 of 88 correct runs flagged (worst 3.7e-11 against 1e-9); since P39 the same check also runs on real ranks (2 T4s with NCCL: 14 of 14 clean, 11 of 11 faults named; 2 machines with gloo); the layout arm is Truestar's own, a third party step (Megatron, FSDP) is not run; pipeline, FSDP and ZeRO sharding not built |
| P35 | Loss spikes cannot be attributed: rewinding and skipping batches hides whether data, the optimizer or numerics caused them (research round 3, R9) | arXiv 2610.38277 (time bomb spikes, FP32 reference logits by hand), PaLM rewind and skip practice, 2026 reproducibility runbooks | `SpikeWatch`: checkpoints, batches and learning rates kept for replay, float64 shadow of sampled and spiking steps; `truestar replay RUN --step N`: numerics, data or optimizer verdict from three replayed arms (docs/SPIKES.md) | Constructed numerics, data and optimizer spikes each get their verdict with the module or rows named; clean runs flag nothing; a replay that does not reproduce the log gives no verdict; baseline: loss and grad norm curves, which show the spike but not its class | **Partly solved, in a small CPU construction**: 15 of 15 constructed spikes classified (5 seeds each), the numerics one at the norm; 0 of 60 shadow flags and 0 of 55 replay verdicts on clean runs; shadow every 50 steps measured +0% on a 146 ms step, every 10 +6%; the 28M time bomb setup on the GPU not reproduced, custom training loops not replayable |
| P31 | The deployed file is never checked: converters rewrite ops, runtimes pick their own kernels, delegates and precision, and fall back to the CPU silently; profiling changes the outputs it reports; a runtime or driver upgrade changes results with no record of what changed (gap analysis items 3, 6, 8, 10) | litert-torch#718, WindowsAppSDK#6610, LiteRT#10342, MNN#4653, TensorRT#4813; the LiteRT PyTorch quickstart's one input `numpy.allclose(atol=1e-5, rtol=1e-5)` | `truestar conform model.py artifact` (docs/CONFORM.md): the file in its own runtime against float64, two verdicts (file in float64 vs model, runtime vs file), float16 and TF32 explanations, residency per node, module localization on cuts, observer guard, golden bundle, `conform replay` with runtime fingerprints | Injected converter faults named at the right module as conversion faults on every runtime; fallback nodes named with their module on every runtime that exposes placement; an instrumented run that changes outputs flagged; a real runtime upgrade names the versions and the first changed module; zero false alarms on clean files; baseline allclose on the same files | **Partly solved**: 3 converter faults (gelu tanh, layer norm eps, INT8 floor) named at the right module as conversion on all 8, 5 and 3 runtimes that ran them; Det, CUMSUM and the ExecuTorch layer norms named off their delegate; ORT 1.20.1 to 1.29.0 replay names blocks.0.ln; LiteRT GPU's float16 default found and fixed by one option (`tflite:gpu-f32`); 0 false alarms on clean files; published bug files, DirectML, Core ML and remote targets not run; `.pte` not localized below outputs |
| P32 | Local LLM engines and cross framework ports silently compute another model, and nobody can tell engine bug from quantization loss from a wrong setting (gap items 11, 12) | llama.cpp issues: 236 mention gibberish, 338 garbage output; #27579, #21888, #26754 (a "GPU bug" that was sampler settings); test-backend-ops is per op against CPU only; Keras 3 FAQ promise of identical numerics, keras#24170, #24248 | `truestar port ref.py gguf:model.gguf` (three verdicts: engine on its own inputs with modeled rounding, conversion by setting and tensor content, quantization from float64 only) and `jax:`, `keras:` candidates with isolated per module replay (docs/PORT_ENGINES.md) | Injected rope theta, norm eps, transposed weight named with the right blame on F32 and quantized files; an engine node fault named at its node under Q4_K_M loss; clean F32, F16, Q8_0, Q4_K_M clean; clean Flax and Keras ports clean; baseline: final logits and a naive per layer dump | **Detected only**: 0 of 40 clean engine runs and 0 of 8 clean framework runs alarm; 140 of 140 file faults with the right blame; 60 of 60 injected engine node faults at the right node (final logits 19, naive per layer 33); 18 of 18 framework faults at the right module; CPU llama.cpp only, injected engine faults only, tiny models only; no remedy |
| P33 | Per layer precision is chosen away from the tool that knows the hardware (TensorRT 11 moved it to a separate ModelOpt pass, reported slower), or by uniform dtypes and class rules judged by aggregate accuracy, so layers that 16 bit floats ruin ship and decisions flip | NVIDIA/TensorRT#4807, <research>/GAP_COMPETITORS.md item 4 | `truestar compile --precision auto --max-flips N` and `truestar.precplan` (docs/PRECISION.md): per unit candidates fp32, tf32, bf16, fp16, int8w, int4w, fp8 measured against float64, range exclusions from `truestar range`, quant verdicts (a) and (b), GPU times from CUDA graphs, bisection over one error threshold with a cast aware dynamic program, held out gate | Plan holds the flip budget on held out inputs on an MLP, a Llama style decoder, resnet18 and a ViT; equal or faster than all fp32, all fp16, all bf16, fp16 except norms and torch.autocast whenever they also hold the budget; a planted one pass variance layer stays fp32; planning under 30 s | **Partly solved**: budget held on held out inputs in every run; where uniform dtypes break the budget the plan is the fastest config that holds it (MLP 1.37x over autocast at 0.5%, 2.1x over fp32 at 0; decoder 1.27x over all tf32 at 0); planted norm kept fp32 by range and by measurement; ViT 3% slower than literal `model.half()` (TS670); planning 4 to 32 s on a quiet GPU, 130 s for resnet18 under contention; eager PyTorch modules, not the compiled executor; ModelOpt not compared |
| P36 | Writing a model and its training loop means hidden mutable state: torch.nn keeps parameters, buffers, gradients and the train/eval flag inside objects (a forgotten `model.eval()`, batch norm buffers mutated in place, `zero_grad`), JAX libraries add variable collections, key threading and module filtering; the optimizer step is Python between kernels; AdamW's `1 - b^t` cancels in float32 (GAP_PYTORCH_JAX items value_and_grad, dropout, training mode; docs/MEMORY.md Limits) | GAP_PYTORCH_JAX.md (training mode row, dropout row), pytorch `torch.func.replace_all_batch_norm_modules_`, docs/MEMORY.md AdamW kernel check failure | `truestar.nn` (immutable layer configs, explicit parameter and state trees, keyed initializers, training as a compile time argument, layer forwards inlined from native source), `truestar.optim` (optimizers as native subset rules over trees, `-expm1` bias corrections), `truestar.train_step` and `fit` (one IR program per step, safetensors checkpoints, autocast and loss scaling) (docs/NN.md, docs/OPTIM.md) | An MLP, a CNN with batch norm and a small GPT train end to end in native Truestar and match the same torch.nn models with torch.optim step for step in float64 (bar 1e-9); every layer and its gradient matches torch in float64; a resumed run equals a straight one bit for bit; the evaluation program contains no dropout or statistic update; speed against eager PyTorch measured; baseline: the torch.nn training loop | **Partly solved**: the three examples match torch to 6.4e-16, 7.1e-15 and 3.0e-15 over 200, 100 and 200 steps (float32 compiled to 2e-7 to 6e-7, the CNN on CUDA 3e-4); 16 layer tests at 1e-12; resume and dropout replay bit identical; the compiled float32 AdamW step passes its check; but the compiled step is 3 to 10 times slower than eager PyTorch on these small models (preliminary), no Lion, gradient accumulation or data parallel on the native step, mixed precision only for straight line losses |
| P38 | A verified framework that is slower than eager PyTorch is not used for inference: the compiled transformer layer ran 0.89 ms against eager 0.67 in strict float32, and decode steps were host bound and copied the KV cache every step | owner goal (2026-11-24); docs/COMPILER.md Limits; ADVERSARIAL_REVIEW.md 1.5 and 1.6 | Contraction rows folded over batch axes, cuBLAS as a timed and verified implementation of float32 contraction kernels, KV caches updated in place (`update=`), grouped query attention without KV copies, CUDA graph replay after the verified first call (`cp.graphed()`) | `scripts/bench_xfmr.py`: TransformerEncoderLayer and a Llama style block (prefill 512, decode at batch 1, 8, 32) in fp32, TF32, bf16, fp16 against eager with every SDPA backend and torch.compile default and max-autotune: faster than the best of them, every output still checked against float64 | **Partly solved, provisional**: wins on the encoder layer in TF32, bf16, fp16 (1.10x to 1.37x) and on prefill in bf16, fp16 (1.04x over max-autotune); losses on the encoder layer and prefill in fp32 and TF32 and on every decode case against max-autotune (0.48x to 0.78x); one shared GPU, no idle rerun |
| P41 | "Same answer on every chip" cannot be checked: no script, no recorded conditions, tiny shapes that never reach tuning or split K, one architecture (adversarial review section 2, fix items 1, 7 and 12) | <research>/ADVERSARIAL_REVIEW.md section 2; NVIDIA cuDNN determinism notes (no equality across architectures or versions); PyTorch reproducibility notes | `truestar fingerprint` and `truestar fingerprint compare` (docs/CROSS_CHIP.md); deterministic mode pins config dependent launch configs | Deterministic rows bitwise equal on two architectures at batch 1, 7, 64 and 256 with tuning reached; compare names every differing setting; baseline eager PyTorch and torch.compile on the same rows | **Partly solved**: float32 deterministic mlp, TransformerEncoderLayer and Llama decode step bitwise equal on sm_120 and sm_75 with Triton 3.7.1 (12 of 12 rows; eager and torch.compile 0 of 12); float16, bfloat16, TF32, resnet18 (cuDNN) and another Triton version differ; Ampere, Ada, Hopper, AMD, Apple not run |
| P39 | Real multi GPU training differs from one GPU or between runs with no signal: all reduce order and algorithm change the bits (NVLS on H20 gives other sums for the same data), parameters that only data parallel touches drift apart across tensor parallel ranks, "deterministic mode" is not reproducible, sharded state silently skips updates; Truestar itself only simulated ranks (docs/PARALLEL.md) | NVIDIA/nccl#1497, pytorch#160169, Megatron-LM#2369, pytorch#149205, pytorch#176076, pytorch#36871 | Real ranks, one process per GPU (NCCL, gloo): fixed order sums (all gather, rank order add), gradient buckets started from post accumulate hooks in bucket order, accumulation, sharded optimizer state and parameters, tensor parallel on real devices, `truestar parallel --ranks devices` on one or several machines, native `train_step(data_parallel=, accumulate=, check_replicas=)` (docs/PARALLEL.md, docs/TRAINING.md Multi GPU) | On 2 real GPUs: the data parallel step bitwise equal to one GPU on the same reduction tree, within float32 rounding of the whole batch step; sharded equal to replicated; tensor parallel within float32 rounding of one GPU; every injected fault named on real ranks; faster than one GPU; baseline: comparing loss curves of a multi GPU run with a single GPU run | **Partly solved, 2 GPUs only**: on Kaggle T4 x2 (NCCL) the 2 rank step equals one T4 on the same tree bitwise in 8 of 8 settings and run to run, the whole batch step within 3.6e-7 (first step gradients); sharded AdamW state bitwise equal at half the state memory; tensor parallel MLP within 4.4e-6; native step bitwise equal to `accumulate=2`; 14 of 14 clean float64 layouts and 11 of 11 faults on real ranks; 1.76x on 2 T4s for a 13.1M transformer, 1.97x for a tensor parallel MLP block, native step 0.59x (slower) on a small MLP; desktop plus laptop over the wireless network correct but 75x slower; no more than 2 GPUs, no NVLink or multi node NCCL, no comparison with PyTorch's data parallel wrapper |
| P40 | Apple GPUs have no verified compiled path: Metal code from compilers (MLX compile, tinygrad, PyTorch's MPS kernels) is trusted, not checked, and a cross backend check can only run the vendor's eager kernels on a Mac | <research>/GAP_COMPETITORS.md item 3 (ROCm, then Metal) and 2.15 (MLX shapeless compile bakes shapes silently), GAP_DEVICES_FRAMEWORKS.md (Metal vs CPU, llama.cpp#23241) | Metal backend of the kernel IR (docs/METAL.md): Metal Shading Language generated from the IR, compiled at run time by `torch.mps.compile_shader`, launched on the MPS tensors, every kernel checked against float64 on its first call; a `metal` device in the compiled executor; `truestar check --backends metal` | Every loop shape compiles and matches float64 on a Mac in float32, float16 and bfloat16; an MLP and a small transformer compiled end to end within the error of MPS eager; a broken kernel refused on its first call; bitwise equal run to run; baseline: MPS eager accuracy and speed | **Partly solved** (2026-11-25, a real Mac mini M2, docs/METAL.md section 5): 109 Metal tests pass on the GPU (4 failed on main before a contraction fix); accuracy at MPS eager's level for every model; attention now a SIMD group matrix shape (loops 322.8 ms, now 2.63 ms on a 512 token causal head batch; 3.9x MPS SDPA), the 128 token encoder 6.58 to 1.27 ms; order free sums and round once math (soft float64 on the bits) run on Metal with the CPU's bits; the decode step 1.27x faster than MPS eager, the MLP and encoder 1.2x to 2.1x slower, 1.8x to 3.8x slower than mx.compile; no implicit GEMM convolution, no autotuning, no sentinels |
| P52 | Low precision and new optimizer training fails late and nobody can tell format loss from a bug: an FP8 or FP4 backward kernel is wrong, a stale softmax delta breaks FP8 attention, Muon's bf16 iteration is trusted unmeasured, attention logits overflow fp16 or fp8 only after training has started | PAIN_RESEARCH_2 N6; arXiv 2609.37852 (Delta-Matching); NVFP4 pretraining and Quartet II (arXiv 2601.22813); ICML 2026 rank aware logit bounds (PMLR 306); torch.optim.Muon (PyTorch 2.10) | `truestar.optim.Muon` with an orthogonality verdict against float64; `truestar.lpgemm` (three verified GEMMs per linear, W / W^T invariant, gradient error planner); `truestar.invariants` (softmax, softmax gradient and gate rows); `truestar.logitbound` (per head bound with a reachable input) (docs/TRAIN_NUMERICS.md) | Injected Muon faults (wrong steps, sign, f32 coefficient typo) are TS1040 while torch's and Truestar's bf16 iterations pass; injected FP8 and NVFP4 kernel bugs are TS1042 with format loss unchanged; per channel W scales are TS1043; a stale delta in an attention backward is TS1048 while the clean program and torch's half precision softmax backward pass; a weight set whose constructed input overflows fp16 is TS1053 and the overflow happens | **Partly solved**: all four acceptance tests pass (tests/test_muon.py, test_lpgemm.py, test_invariants.py, test_logitbound.py); open: bf16 Muon verdict only catches faults above about 14x the rounding, lpgemm is a verification harness (float64 quantizers, 28 ms per verified fp8 step against 1.5 ms f32), no MXFP8 on main, invariant checks are extra outputs not in kernel flags, RoPE bound uses the product form |
| P43 | Compiled kernels and compilers are written and tuned for NVIDIA; on AMD GPUs they do not compile, spill registers on wave32 parts, or need a separate port, and nobody can check them before renting the GPU | Owner goal (hardware breadth, 2026-11-24); ROCm/TheRock#8379 (row P1); Triton's AMD backend allows TF32 on gfx942 only (Triton 3.7.1 source) | One kernel IR and one Triton emitter for NVIDIA and AMD: `K.compile(..., "rocm")`, `CompiledExecutor("rocm")`, `rocm+inductor`, ahead of time AMD builds without the GPU (`truestar.kernel.rocm.build`, docs/ROCM.md) | Every loop shape compiles for gfx942, gfx1100 and gfx1103 with no spill and LDS within the target (compile only, tests/test_rocm.py); on an AMD GPU every kernel test passes against float64, deterministic results are bitwise stable and the benchmark is recorded (scripts/rocm_validate.py) | **Partly solved**: 57 of 57 default launches compile for the three targets on Windows and Linux with no ROCm installed, none spills after relief; never run on an AMD GPU, so nothing about values or speed is verified |
| P42 | A framework that covers fewer ops and layers than PyTorch fails on the first model that uses one: an unsupported op at import, a missing gradient rule at the first training step, a silent CPU fallback in the compiled program, or a gradient that differs from torch's at a convention (padding rows, clamp bounds) | The op coverage sweep at 7aca8a9 (scripts/op_coverage.py, results/op_coverage/before.json): 67 of 170 models of the torchvision zoo, 15 Hugging Face configs and 92 torch.nn layers and losses refused at import, 53 of 161 ATen ops unknown, 88 of 161 with a reverse rule for every IR op they produce, 10 ops on the float64 CPU fallback in 17 of the compiled models; BERT's embedding gradient 0.36 off and mobilenet_v2's 0.35 off against torch | Every ATen op of the list lowered onto IR ops that have a float64 reference, reverse, forward and batching rules and compiled steps (ir/importer_extra.py, ir/grad_rules_extra.py, compiler/coverage_rules.py); the gradient only conventions written with `frozen`; the layers in truestar.nn (nn/layers_more.py) (docs/MODEL_COVERAGE.md, docs/NN.md) | Every model of the list imports and matches PyTorch float64 (1e-10), compiles on CUDA with no CPU fallback, and one training step's gradient equals torch.autograd in float64 (1e-8); every new rule passes finite differences, the forward dot product test and the vmap per element test; every new truestar.nn layer matches torch.nn in float64 (1e-12) eagerly, compiled and in its gradient; baseline: the same sweep at 7aca8a9 | **Solved on the list**: 170 of 170 import (was 103), 161 of 161 ATen ops with every rule (import was 108, reverse 88), 0 CPU fallbacks (was 10 ops in 17 models), 169 of 170 training steps equal to torch (the one left is torch's float32 1/6 in its float64 hardsigmoid backward); 10 ops still run part of their work as torch steps on the device (gathers, argmax); no pretrained weights, no models over 100M params, the torchvision zoo compiled on CUDA only |
| P37 | Compiled training is slower than eager in Truestar and needs more memory: a whole step program 2.3x slower than eager, conv backward 3.5x, the autodiff pass stores more than PyTorch (256 against 181 MiB on a six layer transformer); compiled AdamW did not run at all | docs/MEMORY.md and the conv status entry (2026-11-24); owner goal: the best general ML framework | `CompiledTrainStep` (donation every step, verified CUDA graph replay, TS780, TS781), training ops as kernels, `expm1` bias correction, relu gradient from its output, deterministic split conv weight gradient, `param_group` flat optimizer (docs/TRAINING.md, Compiled) | MLP, six layer transformer and resnet18 training steps at two batch sizes faster than eager and level with torch.compile default and max-autotune, peak memory at or below eager, every kernel still float64 checked on its first call | **Partly solved**: faster than eager and torch.compile default on the MLP (batch 64) and the transformer (batch 8) with the graph, 1.4x to 6.6x faster than before; resnet18 2x slower than eager; max-autotune 1.7x faster on the transformer; memory below eager on MLP and resnet18, above on the transformer in one run; batch 16 transformer refused by the end to end near tie check (eager float32 is as far off) |
| P48 | A serving engine's prefix cache or chunked prefill silently changes the answer: the cached prefix run gives other bits than one full prefill of the same sequence, and nobody can tell rounding from a wrong path (research LIVE_RESEARCH_2026_11, ranked 2) | <research>/LIVE_RESEARCH_2026_11.md (prefix caching, its sources) | `truestar prefixcheck` and `truestar.prefixcheck.compare`: bitwise or the first token and layer that differ, each run's error against float64, TS702 to TS704 (docs/PREFIX_CHECK.md); the same check on Truestar's dense, paged, decode and quantized caches (scripts/prefix_check_llama.py) | Dense, paged and token by token decode caches bitwise equal to the full prefill in deterministic float32 at every split; where they cannot be (split KV, quantized prefix) the step and the size of the difference are named and a less accurate path is told apart from rounding; files from any engine checked | **Partly solved**: dense, paged and decode caches bitwise at every split (lengths 96 to 2200, float32); the fused attention kernel's split KV decode differs from the prefill row from 2048 keys (3e-5, named, no accuracy loss); int8 and fp8 prefixes named (fp8 flagged TS703); float16 waits for the island fix (laptop-fp16decode); no external engine's files run yet |
| P47 | "Verified against float64" checks only the size of errors: a kernel whose errors are small but all lean one way (truncation, round toward zero accumulation, a biased exp or softmax) passes, yet that bias is what made low precision flash attention training explode; and nobody can say in which order a backend adds, so bit differences between correct backends or GPUs (the README's eager transformer row) stay unexplained | Qiu and Yao, ICLR 2026, arXiv 2510.04212; arXiv 2506.20752; Xie et al., USENIX ATC 2025 (FPRev); <research>/LIVE_RESEARCH_2026_11.md items 1.1 and 1.2 (ranked 6 and 7); README_DRAFT.md "Same model, two GPUs" | Lean test in kernel verification (TS960, TF32 reported as TS966) and in `truestar check` (TS961), docs/BIAS.md; `truestar orderprobe` recovers, describes and replays the summation tree (TS962 to TS965), docs/ORDERPROBE.md | Injected biased kernels that pass the size check are refused (truncating store, round toward zero accumulation, biased exp, a truncating module in check); zero firings on correct kernels and modules (CPU battery 806 outputs, GPU kernel tests 199, zoo 796 modules on CPU and on CUDA); recovered trees replay the backend's float32 bits | **Done on one PC** (2026-11-24 status entry): the cause of the two GPU eager difference is inferred (cuBLAS split K of 2 and 4 on 70 SMs), the second GPU not probed |
| P46 | A coverage claim measured on a 2024 model list: the op coverage sweep ran on transformers 4.48, which predates every 2025 to 2026 family (Gemma 3, 3n and 4, Qwen3, SmolLM3, OLMo 3, gpt-oss, Qwen3 Next, Kimi Linear), and on a current transformers every decoder of the list, old or new, is refused at import | <research>/LIVE_RESEARCH_2026_11.md item 5.1 (ranked 9, before launch); the sweep at e96cd2c on transformers 5.24.0: 10 of the 15 first list models and 2 of 28 current families import (`aten.diff` from the transformers 5 packed sequence check, then `square`, `floor`, string conv padding, `meshgrid`, `tile`, `bucketize`, `std`, `erfinv`); MoE, linear attention and SSD models fail with a list of unknown ops, not a reason | 28 current families in the sweep (`testing_hf.MODELS_2026`), the ops they need lowered onto existing IR ops (`ir/importer_current.py`), and MoE routing, chunked linear attention and chunked SSM scans refused by name before import with TS940, TS941, TS942 (`ir/refusals.py`, docs/MODEL_COVERAGE.md) | On the latest stable transformers, every current dense and vision family imports and matches PyTorch float64 (1e-10), compiles with no CPU fallback and trains one step equal to torch.autograd in float64 (1e-8); every MoE, linear attention and chunked SSM family is refused with its code and layer, never a fallback; the tests pass on transformers 4.48 and 5.24; baseline: the same sweep at e96cd2c on 5.24 | **Solved on the list, CPU**: on 5.24, 15 of 15 first list models (was 10) and 26 of 28 current families (was 2) import, compile on the CPU with no fallback and train equal to torch; 11 of 11 MoE, linear attention and SSM families refused by name; SigLIP2 and Pixtral vision not supported (data dependent per image sizes); CUDA not run for the new families (GPU busy); no pretrained weights; MoE and linear attention themselves are planned, not built |
| P50 | Low precision formats bite silently: an FP8 checkpoint made for one GPU family decodes 2x off on the other (OCP versus FNUZ), MX block scales clip their own maxima under the v1.0 rule, NVFP4 wastes accuracy with one fixed block scale target, and hardware stochastic rounding is neither reproducible nor checkable | AMD ROCm MI300/MI350 workload optimization page ("MI300X uses the FNUZ FP8 variant, while MI350X uses the OCP FP8 variant"; TF32 "software emulation via BF16 on MI350X"); OCP MX v1.1 (Oct 2026) expAdj; arXiv 2506.08027 (MXFP8 recipe, round up scales); arXiv 2512.02010 (adaptive 4 or 6 block scale); arXiv 2509.25149 (NVFP4 pretraining, SR on gradients); <research>/LIVE_RESEARCH_2026_11.md items 2.1, 2.3, 2.4 | `truestar.quant.mx` (MXFP8, MXFP4 tables, three scale rules, quantizer check), `mx_kernels` (MXFP8 on `tl.dot_scaled`), `fp8variant` (variant from names or bits, TS1000 to TS1005), `sround` (keyed stochastic rounding, TS1006, TS1007), `sim:mi350-tf32` (docs/QUANT.md) | Tables equal ml_dtypes and torch on every code; FNUZ bits under an OCP target named; SR bitwise equal on CPU, CUDA torch and Triton and unbiased over many draws; baselines: round to nearest, the OCP v1.0 scale, plain NVFP4 | **Partly solved**: tables, variant check and SR bitwise and unbiased (5 targets, |z| <= 1.61); `floor_adj` lowest MX error, 0 clipped; tried and not shipped: adaptive NVFP4 (weight MSE 0.84x on 224 of 224 SmolLM2-360M layers, model cross entropy not better) and MXFP6 (no GPU kernel, Triton NYI); SR 5.6x slower than a round to nearest cast; no AMD GPU run |
| P45 | Phone and NPU numerics are only simulated (`sim:*`): nobody knows whether a profile's arithmetic is what real Hexagon hardware computes, which precision a runtime really used, or whether an "npu" run ran on the NPU at all | Owner goal (real devices, 2026-11-24); docs/NPU.md (INT8 flips on S24/S25); Qualcomm AI Hub docs (TFLite `allow_fp32_as_fp16` and HTP float16 defaults, CPU fallback always on) | A fixed battery with every intermediate tensor as an output (`truestar.conform.qaihub_battery`), run on hosted devices, and `truestar conform qaihub DIR`: file based judgement of saved outputs against float64, float32 eager and `sim:ieee-f16` per tap, INT8 layers replayed with intref from the device's own input (docs/QAIHUB.md) | On 3 or more Snapdragon generations and runtimes: every run's real precision named, every accelerator run that fell back named, each tap set against the simulated prediction, INT8 judged per layer; baseline: the simulated profiles alone | **Partly solved**: 4 devices (v81 phone, v73 laptop, two v68 boards), 25 jobs; convs bitwise equal to `sim:ieee-f16` on 94% to 100% of elements; GELU (Erf) 6.2x correct float16 and not modelled by any profile; 4 of the 8 float accelerator runs that completed were silently on the CPU in float32; INT8 on the NPU one step off on 1% to 4.6% of elements per layer, never more; TFLite INT8 from per channel QDQ fails to compile; no GPU result, one sample; the float Hexagon profile is fitted in P59 (`sim:hexagon`) |
| P53 | 2025 to 2026 models (Qwen3 Next, Qwen3.5, Kimi Linear, Mamba 2 hybrids) run linear attention and SSM recurrences in chunks; libraries pick chunk 64 or 16 and get other bits, nobody reports how rounding grows along the chunk chain, and Truestar refused these layers (TS941, TS942) and ran Mamba only as the sequential scan | <research>/LIVE_RESEARCH_2026_11.md 5.2 (Qwen FlashQLA, the flash linear attention GDN layer, arXiv 2503.14376, Megatron falling back to a torch GDN path for determinism) | `truestar.linattn.linear_attention`: chunked Triton kernels with the chunk size as a pinned reduction order parameter, float64 sequential reference, `chunk_invariance` and `chain_error` reports (docs/LINEAR_ATTENTION.md) | Float64 chunked equals the recurrence for every chunk size and its gradients equal the recurrence's; float32 within tolerance of float64; deterministic outputs bitwise equal on 26 and 70 SM GPUs; faster than the sequential scan and the eager chunked form (tests/test_linattn.py, scripts/bench_linattn.py) | **Partly solved**: kernels for the add and delta rules with none or scalar decay are verified, bitwise across a 26 and a 70 SM GPU, and faster than both baselines (docs/LINEAR_ATTENTION.md section 7); per channel decay runs only in the chunked PyTorch form; no fused backward; no importer lowering, so the models are still refused |
| P44 | Accelerators nobody has locally (Google TPU) compute float32 differently: JAX's DEFAULT matmul precision on TPU is one bfloat16 pass, and no float64 reference sits next to the device | JAX `jax.lax.Precision` docs (DEFAULT, HIGH bf16_3x, HIGHEST); Kaggle TPU v5e-8 free quota | `truestar conform tpu DIR` (docs/TPU.md): a fixed seeded battery of 18 cases run on the device as one file, outputs judged here against float64 by effective bits, rel L2, ulp and bitwise stability (TS900 to TS906) | TPU v5e default precision cases named as reduced precision and highest cases clean; local CPU and GPU runs clean except known TF32 and XLA tanh; planted faults named; baseline: allclose at 1e-5 | **Detected only, TPU not yet run**: locally TF32 flagged (14.3 bits vs 16), XLA CPU tanh 4.5 ulp flagged, all float32 paths 23 to 24 bits, every case bitwise stable; 5 planted fault kinds named in tests; the Kaggle TPU kernel is queued (batch TPU sessions wait 10 to 24 h), so no TPU number yet |
| P60 | Truestar had verified chunked linear attention and SSD kernels (P53) but still refused every Mamba 2 SSD and Gated DeltaNet family at import (TS941, TS942): Mamba 2, Falcon H1, Qwen3.5, Bamba and Zamba 2 could not be checked, compiled or trained at all | the owner's task (2026-11-24); the P46 sweep on transformers 5.24 (0 of these import) | importer lowerings of the layers' chunk functions onto `torch.ops.truestar.linear_attention` (`ir/hf_linattn.py`), the IR op with float64 reference, reverse, forward and vmap rules and a compiled library step on the linattn kernels (`ir/linattn_ops.py`); refusal kept, with the reason, for every uncovered layer or call (TS941, TS942, TS1150) | the five families import, match PyTorch float64 (1e-10), train one step equal to torch.autograd (1e-10), compile with no CPU fallback; refused variants stay refused; baseline: main 9a9ad96 refuses all five | **Solved on the list, CPU**: 5 of 5 import (was 0), fp64 error at most 6.4e-16, gradients at most 5.4e-16, CPU compile with no fallback; CUDA compiles 4 of 5 at the test shape (Falcon H1 refused loudly by the kernel check, TS1060); not faster than eager at these test widths (0.25x to 2.5x CPU, 0.45x to 1.18x CUDA, docs/LINEAR_ATTENTION.md section 10) |
| P61 | Core ML decides per operation where a model runs and how precisely: a Neural Engine request is silently served by the CPU, small models ignore the GPU request, and float16 arithmetic differs by unit, with no API that reports any of it | Mac mini M2 study 2026-11-25 (<research>/mac_real/coreml/); Apple MLComputePlan documentation; the Snapdragon study of P45 (<research>/qaihub/REPORT.md) for the method | `truestar conform coreml DIR` on runs saved by `python -m truestar.conform.coreml collect DIR` (docs/COREML.md): per run error against float64, float32, sim:ieee-f16 and sim:mali-fp16, and the compute plan's placement against the requested unit | A FLOAT32 program asked for the Neural Engine is named as running on the CPU (TS1161); a unit beyond correct float16 is named with its ratio to a float16 accumulator (TS1163); a planned fallback the unit supports is TS1160; baseline: a successful prediction, which says nothing | **Detected, one Mac** (2026-11-25, M2, macOS 26.6.2): GPU float16 0.74x to 1.09x sim:ieee-f16 on every model; CPU and Neural Engine 8.7x to 11.2x on the 2048 term MLP (TS1163); FLOAT32 on CPU_AND_NE all CPU on every model (TS1161); no remedy beyond choosing the unit |
| P56 | Device breadth without the hardware owner: Windows ML picks a vendor EP per PC and falls back to the CPU EP silently, LiteRT's NPU path falls back to XNNPACK while `is_fully_accelerated()` says True, WebNN maps one graph to a different native stack per OS, and a 1 LSB difference on Snapdragon cannot be told from a kernel bug because QAIRT's range modes are not modelled | <research>/LIVE_RESEARCH_2026_11.md items 6.1, 6.2, 6.3, 6.5; Windows ML supported EPs page; LiteRT NPU and Intel pages ("is_fully_accelerated() alone is not sufficient"); W3C WebNN CRD 2026-10-21; QAIRT quantization page | `winml:<ep>` and `webnn[:device]` check backends on the shared ONNX path (per module error against float64, residency, op drilldown, repro); `tflite:npu` conform runtime reading LiteRT's own log and AOT dispatch partitions; `truestar.intref.qairt` range modes, `attribute` and `simulate` (docs/BACKENDS.md, docs/CONFORM.md, docs/WEBGPU.md, docs/INTREF.md) | Every EP present on the machine runs a clean model with per layer errors and names a CPU fallback (TS1120, TS1121); an NPU request that ran on the CPU is TS1125 on a real run; WebNN localizes a wrong op translation to its module; each QAIRT mode applied to device like data is named (TS1130) and a one step difference is not called a kernel bug (TS1131); baseline: final output allclose, which names nothing | **Partly solved, one PC**: NvTensorRtRtx (TF32 GEMMs found, 2.79e-4, equal to torch TF32) and WebGPU EPs run with per layer errors; DirectML, QNN, OpenVINO, VitisAI and AMD EPs not available here; LiteRT NPU request ran on XNNPACK with is_fully_accelerated() True (TS1125), no real NPU or AOT file; WebNN on cpu, gpu, npu device types in Chrome 158 headless, all under 1.7e-7, native stack not identifiable; QAIRT tf and symmetric follow the doc examples, enhanced and adjusted are hypotheses, no QNN job compared yet |
| P55 | A tensor parallel step trains another model than one device does in its last bits (partial sums added in another order), so training and rollout or a TP 2 and a TP 4 run disagree; and activation memory is planned by a greedy checkpoint choice with no offload | arXiv 2511.17826 (TBIK, ICML 2026), NCCL issue 1497, PyTorch selective activation checkpointing and torch_remat devlog 2026-11-19, Poolside C2C offload 2026-03-18 (research items 4.4 and 4.5) | `truestar.parallel.tree`: one canonical reduction tree for every linear contraction, the ranks walk subtrees and finish it; `recompute="mincut"`, `"offload"`, `"mincut+offload"` in the compiler (docs/PARALLEL.md, docs/MEMORY.md) | tp 2 and 4 float32 training step bitwise equal to one device on CPU, CUDA and across two machines (plain step: every run differs); min cut meets every budget the greedy pass meets with bitwise equal gradients and planned equal to measured peak, with less recompute work | **Solved for tp and the exact leaf** (2026-11-24): bitwise in every measured run, 21x to 174x slower per linear layer (tiled leaf 5x to 8x, bitwise on CPU only); min cut and offload measured in docs/MEMORY.md |
| P54 | Same answer on every chip and fast are at odds: reduction bits follow the block size, split and SM count (so deterministic mode pins configs), exp, log, tanh, erf and rsqrt differ in their last bits across vendors, the float64 reference is slow on consumer GPUs (float64 at 1/64 rate), tensor core accumulation of GPUs we do not own cannot be tested, and two nondeterministic stacks cannot be told apart from a fault | <research>/LIVE_RESEARCH_2026_11.md items 1.3, 1.4, 1.6, 1.7, 1.8 (NVIDIA CCCL determinism levels, arXiv 2510.09180, 2512.07004, 2510.16028, 2504.08009, CUDA 13.5 notes) | Order free sums `osum` and `compile(order_free=True)` (docs/ORDER_FREE_SUM.md); round once math `compile(math="round_once")` (docs/ROUND_ONCE.md); sliced float64 matmul on int8 tensor cores for the device reference, `TRUESTAR_VERIFY_F64=sliced` (docs/SLICED_F64.md); bit accurate tensor core models as `sim:` devices (docs/TENSOR_CORE_MODELS.md); operator local worst case and probable bounds, `truestar.opbounds` (docs/TOLERANCE.md) | Order free rows bitwise equal over every launch config, permutation and split, on CPU and CUDA, with a proven error bound; round once results correctly rounded and bitwise equal across devices; sliced float64 every entry within the native float64 guarantee and faster than native on large matmuls; tensor core profile bitwise equal to the local GPU on 10^6 random dot products per format; baseline: plain float32 sums, native torch math, native float64, the vendor GPU | **Partly solved** (2026-11-24): osum equal bits over 3 configs per kernel where plain sums differ, 0 mismatches against two independent definitions (RTX 5070 Ti and L4), 1.2x to 2x slower per sum, up to 5x for a layer norm; round once 0 mismatches against mpmath over all float16 and bfloat16 inputs for 6 functions, 0 over 2^32 float32 inputs under a measured (not proven) error bound, 0 bit differences CPU vs CUDA vs L4 where native torch differs on 40 to 700 million float32 inputs per function, 6x to 77x slower; sliced float64 more accurate than native in every sample, 1.4x to 2.3x faster at n >= 2048 on an L4 and 2.05x for the float64 check of a 4096 cubed matmul, slower below 2048; sm_120 tensor core profiles bitwise equal on 32.7 million dot products, Ada fp16/bf16/tf32 fitted on an L4, Ada fp8 not matched, V100 to B200 from the paper only; opbounds tells reordering from a 1% fault on CPU tests. Not run on AMD or Apple; contractions and attention keep ordered sums |
| P51 | Rotations before 4 bit quantization (QuaRot, SpinQuant, FlatQuant) are folded into weights by hand and never checked: a norm scale left in place or a rotation across RoPE or SiLU silently changes the model, and the quantized accuracy table cannot tell that from format loss; random rotations vary (SpinQuant: up to 13 points); 4 bit KV caches (TurboQuant, NVFP4 KV) ship with no bound on what they change | SpinQuant (ICLR 2025), FlatQuant (ICML 2025), DartQuant (NeurIPS 2025), AMD Quark rotation docs, arXiv 2605.09825, vLLM TurboQuant study 2026-05-11, NVIDIA NVFP4 KV cache blog 2025-12 (research items 2.2, 2.9) | `truestar.rotate`: deterministic Hadamard (Sylvester, Paley) and learned rotations folded in float64 and proven equal to the original at 10 points per layer before anything is quantized, wrong placements named (TS1020); int4 and NVFP4 KV caches, optionally rotated, with the proven error bound (docs/ROTATION.md, docs/KV_QUANT.md) | The proof accepts correct folds on a real pretrained model and names each of three wrong placements; format loss with and without rotation measured on real layers; 4 bit KV error never above the bound, paged equals dense bitwise | **Solved for Llama and Qwen2 layouts**: SmolLM2-360M proven at 321 points (worst 4.6e-14, bound 1e-9), the three wrong placements named; rotation measured as a large win for per token integer activations (W4A4 perplexity ratio 1041 to 9.3) and a loss on block and FP formats (NVFP4 KL 0.25 to 0.90 or more), so a rotated model is refused for block and float formats (TS1024) and offered for per token integer activations only; rotated 4 bit KV: perplexity +1.7% (int4) against +4.9% unrotated, 1.7x to 3.0x slower than int8 |
| P59 | The phone NPU simulation is wrong exactly where the chip differs: real Snapdragon NPUs compute GELU about 6x further from float64 than correct float16 and INT8 one step off on 1% to 4.6% of elements, while every `sim:*` profile follows a public description and none was fitted to a device Truestar measured | <research>/qaihub/REPORT.md (P45 runs: S26 v81, X Elite v73, S21 and RB3 v68), surprises 1, 2 and 6 | `sim:hexagon` (`truestar.devsim.hexagon`): float16 with a wide accumulator, GELU fused into the linear layer before it and computed with the device's approximation fitted per sign; TS1140 names each fitted op, TS1141 each op known to differ but not modelled; fit and measurement in `scripts/hexagon_sim_fit.py` (docs/DEVICES.md section 7) | On held out device outputs the profile is materially closer than `sim:ieee-f16` for the GELU and every tap after it, the convolutions stay bitwise equal to `sim:ieee-f16`, and clean models stay clean; baseline: `sim:ieee-f16` (the best profile before) | **Partly solved** (2026-11-25): held out GELU rel L2 to the S26 2.71e-3 to 2.73e-4 (9.9x), bit equal 6.7% to 89.4%; battery taps 10.9x (GELU), 7.8x, 6.1x (MLP), 1.8x (encoder output, a GELU the fit never saw); convs identical bits; tried and not shipped: an INT8 requantization rule (no fp16, scale storage, double rounding, per channel offset or input quantizer variant beat exact requantization; the error looks additive, about 0.01 step); softmax and attention (1.7x to 2.0x) not modelled; one sample, untrained weights |
| P57 | Speculative decoding is sold as lossless but changes the output: the verify call scores k+1 query rows and kernels round differently from the one row decode, so greedy tokens flip at near ties (research LIVE_RESEARCH_2026_11 4.2) | arXiv 2510.22876, 2607.17283, 2609.15504, 2601.17768 | `truestar.specdec`: `check_rows` proves the target row invariant (TS706, or TS705), then `speculative` returns plain greedy's tokens bitwise (docs/SPEC_DECODE.md) | Faster decode with bitwise the tokens of greedy decoding, the invariance checked before use | **Partly solved**: 1.43x graphed (702 to 1007 tokens/s) and 1.31x ungraphed on the laptop with a trained draft pair, 8 of 8 prompts bitwise equal at every k; greedy and batch 1 only, no keyed sampling, no MTP head |
| P49 | Deterministic mode is slow because it cannot autotune: any config search that changes a reduction's order changes the bits, so attention and reductions run their default config (research LIVE_RESEARCH_2026_11) | <research>/LIVE_RESEARCH_2026_11.md | Order keeping tuning: a deterministic kernel is tuned only over the parameters that cannot change its summation order (`triton_backend.ORDER_FREE`: LPT and num_stages for attention, BR for single pass reductions), cached picks with another order refused, the pick checked bitwise against the default (docs/KERNEL_IR.md 6b) | Faster deterministic kernels with the same bits on every GPU; fingerprint unchanged; 26 and 70 SM tests pass | **Partly solved**: 18 case sweep (scripts/order_classes.py) never saw those parameters change a bit; fingerprint 67 of 67 deterministic rows bitwise equal to main with fresh caches; float32 attention prefill 1.23x on the laptop, other cases within noise; block sizes and warps (the large levers) stay pinned since they change the order; looped reductions have no free parameter |
| P63 | A float32 decode at a long position fails Truestar's own end to end check: float32 rotary angles (position times inverse frequency) lose up to half an ulp of the product, so the new key row is 4.3e-5 off float64 at position 2100 and 4.9e-4 at 32000 (found by the prefix cache check, P48) | docs/PREFIX_CHECK.md | Float64 islands in float32 programs: a value the model declares float64 keeps float64 storage and math and rounds to float32 once where the model casts (docs/COMPILER.md) | Long position decode passes the check; default programs bit identical | **Solved for a model that declares its angles float64** (`rope_f64`, a float32 import): key row 2.4e-7 to 4.7e-7 from 100 to 32000, the position 2100 check passes (it failed at 8.6e-4); 9 of 9 compiled hashes and 67 of 67 deterministic fingerprint rows unchanged. A model that does not declare float64 angles keeps the float32 result |
| P64 | MoE models were refused (TS940), and where they run, a token can reach different experts per chip with no warning | transformers 5.24 Mixtral, Qwen3 MoE, OLMoE and DeepSeek V3 MoE blocks; routers' top k decided at a near tie | routed top k MoE (Mixtral, Qwen3 MoE, OLMoE, DeepSeek V3 incl. group limited) import, compile and train (`truestar.moe`, docs/MOE.md); one tie rule on every device (larger score first, equal scores to the lower expert); grouped experts checked against float64 on the first call (TS1181); the near tie router report names the token and layer whose expert choice the working precision cannot decide (TS1182, TS1183); what stays refused names why (TS940) | 5 of 5 configs import and train equal to float64 (forward <= 4.4e-16, gradients <= 6.5e-16; before: 5 of 5 refused); grouped experts bitwise across runs on CPU and an L4; a router set 1e-9 apart flags 8 of 8 tokens; baselines: transformers' eager experts and grouped_mm | **Partly solved**: correct on tiny random configs; compiled path runs every expert on every token (E/k times the work); grouped kernel 1.6x slower than grouped_mm at the small L4 size; gpt-oss, Llama 4, per expert modules and capacity dropping stay refused; the report is a Python API only |
| P65 | DeepSeek V2 and V3 style models use multi head latent attention (one compressed latent and one shared RoPE key per token instead of per head K and V); Truestar could not import a Hugging Face DeepSeek V3 (expand_kv's `new_empty` was unsupported), had no latent cache decode, and the weight absorbed decode serving engines use was unchecked; DeepSeek V3.2's token sparse selection was not refused by name | DeepSeek V2 and V3 technical reports (MLA, decoupled RoPE, absorption); transformers 5.24.1 `DeepseekV3Attention` and the V3.2 and V4 indexers | `truestar.mla`: latent cache prefill and decode steps (expanded and absorbed), `from_hf`, `prove_absorption` (TS1201), `compile_steps`; `torch.empty` imports as NaN; TS1202 refuses token sparse selection, TS1200 other MLA variants (docs/MLA.md) | HF DeepSeek V3 imports and matches float64 (2.1e-16); five HF variants match in both modes within 1e-13 and decode the same greedy tokens; a wrong absorption (heads rolled in W_uk or W_uv) is TS1201 at layer 0; compiled float32 tokens equal float64; latent cache 4.44x smaller than per head K and V; two compiles bitwise equal; baselines: HF eager, the eager step, the eager step in a CUDA graph | **Partly solved**: correct and verified on tiny and small random configs (tests/test_mla.py 20 of 20 on an L4, transformers 5.24.1); dense layers only (MoE still TS940); compiled decode is 2x slower than eager PyTorch without graphs and 1.2x to 1.4x slower than eager in a CUDA graph on one L4 config (host bound, GPU time per step equal); default compile mode not bitwise across processes in 1 of 4 containers, `deterministic=True` was in 2 of 2; no paged latent cache, no pretrained checkpoint |
| P66 | Multi GPU beyond 2 ranks: the canonical tree refused any tp that is not a power of two, and a data parallel step over unequal replica batches trained silently on the wrong mean | a batch of 16 over 3 ranks (6, 6, 4); tp=3 refused with TS1102; NCCL needs equal reduce scatter pieces | tree tp for any tp = W * 2^a (W the odd factor); unequal replica batches refused on every replica (TS1210), uneven reduce scatters on every backend (TS1211); each gloo rank on its own GPU (`cuda:r`) | tree tp 3, 4, 6, 8 bitwise equal to the one device step on CPU; dp, shard and tp layouts at 3 and 4 real gloo ranks within 1e-9 in float64; native dp bitwise at 1, 3, 4 ranks; the two refusals fail on the previous code | **Partly solved, CPU only**: the 4 x L4 run is pending (Modal app ap-cTmcf1LooMe9DzCIef1S6y, result not collected); IR level `data_parallel=True` and torch.nn pieces do not check equal shards; CPU silu, gelu and sigmoid stride dependence breaks the bitwise claim for silu gates |

**Solves versus detects (2026-11-24).** Most rows above detect a problem and hand over a repro;
the owner asked for problems to be solved. Status per row:

| row | today | how it is solved, when it is |
|---|---|---|
| P1 | solves, for a blamed ATen op | `truestar fix` swaps the op for a verified decomposition on that device only; the rerun check must be clean (docs/FIX.md). Module level device branches are still only detected (TS093) |
| P2 | solves in simulation | `truestar fix --int8` keeps the fewest layers in float so simulated INT8 flips meet a target, verified by simulation; on the phone NPU it still only detects |
| P3 | detects | names the op or layer; the fix comes from P1 |
| P10, P13, P14, P16, P18, P22, P34 | detect | no remedy yet (P34 names the collective or module to fix; nothing is patched) |
| P37 | partly solves | a compiled training step faster than eager on two of three models, with every kernel checked; resnet18 and max-autotune comparisons are still losses |
| P35 | classifies | the verdict says which fix applies (raise the named module's precision, drop or fix the named rows, lower the learning rate); no fix is applied or verified yet |
| P29 | explains, partly solves | `truestar intref` names the convention per op and writes golden vectors with per layer tolerances a board can check against; for the NPU it rules out rounding and points at float16 input quantization (remedy to try: feed quantized input), not yet confirmed on the device |
| P31 | detects, and solves the precision case | `truestar conform` names the module and says converter or runtime; for a runtime that computes in float16 the remedy is verified by rerunning with float32 forced (`tflite:gpu-f32` ok on the same file); converter and kernel faults are detected only |
| P25 | solves for its own kernels | `truestar.quant` kernels are verified against the exact format on every new shape and pass; a third party kernel can be checked (`verify.check_matmul`) and its bug named, but not fixed |
| P15 | solves on the RL kernel set | `truestar.rlkernels`: one invariant kernel set for the training forward, prefill and decode, bitwise equal logprobs, checked by `truestar rlpair --kernels rl`; TP only as a one device simulation; other kernels are still only detected |
| P19 | solves for autotuning, detects for kernels | `verified_autotune` removes a wrong config and uses the fastest right one; `truestar kernel` names the broken property and leaves the fix to the author |
| P39 | partly solves, 2 GPUs | data parallel, tensor parallel and sharded state run on real GPUs and are verified against one GPU (bitwise on the same reduction tree); more GPUs and interconnects are not run |
| P36 | partly solves | correctness and simplicity are solved and measured against torch.nn and torch.optim; speed is not (the compiled step is slower than eager PyTorch today) |
| P42 | solves on the measured list | every model of the coverage list imports, compiles with no CPU fallback and trains one step equal to torch; ops outside the list are still refused by name (TS860, TS861 or the unknown op list) |
| P46 | solves on the measured list, refuses the rest by name | current dense and vision families import, compile and train equal to torch on transformers 5.24; MoE, chunked linear attention and chunked SSM layers are refused with TS940 to TS942 naming the layer |
| P50 | detects the variant trap, solves rounding | FP8 variant mismatches are named (TS1000) and not converted; keyed SR and the `floor_adj` MX scale are the remedy for irreproducible SR and clipped maxima, verified exactly |
| P53 | solves for its own kernels | the chunked recurrence is verified against float64 on every new signature and pinned to one reduction order; the models need an importer lowering before the refusal lifts (`truestar.linattn.refusal_lifted`) |
| P60 | solves on the measured list, refuses the rest with the reason | Mamba 2, Falcon H1, Qwen3.5, Bamba and Zamba 2 import, compile and train equal to torch on transformers 5.24; other linear attention layers, cache calls and transformers 4.x stay refused (TS941, TS942, TS1150) |
| P55 | solves for tp layouts, slowly | the tp step is bitwise the one device step by construction (exact leaf); the speed cost is the open part (no fused tree kernel); dp, ga and sp are not covered (batch invariance of every op) |
| P20 | solves | Truestar's own random ops give the same bits everywhere (the remedy is to draw with them); training mode faults are detected, with no remedy |
| P21 | solves in emulation | `truestar range` proposes an exact power of two rescale, float32 ops or clamps and keeps the first that passes emulation of the target; not yet run on a real low precision device |
| P23 | solves where Truestar owns the allocations | the plan is exact before any launch, an over budget program never launches, recomputed values are bit for bit equal and checked; cuDNN convolution workspace is outside the plan |
| P17 | partly solves | custom ops need no toolchain today; the first call is not under 2 s and the second torch version and Linux are not run |
| P33 | solves | the plan is verified end to end against float64 on held out inputs before it is returned; a layer 16 bit floats ruin stays float32, a buggy quantized kernel is removed (TS664) |
| P38 | solves for some shapes and precisions | faster than eager and torch.compile where the P38 status entry says win; each loss is listed there with its cause |
| P56 | detects, and explains the QAIRT case | names the EP or NPU fallback and the module whose error is over threshold; for a Snapdragon 1 LSB difference `qairt.attribute` says which range mode or tie rule explains it (not a fix; the remedy is to choose that mode); nothing is patched |
| P51 | solves for Llama and Qwen2 layouts | a rotation is proven equal to the model in float64 before quantizing, and a wrong one is refused with the point and steps named (TS1020); whether it helps is measured by the format loss, not assumed (it hurt every block and FP format on the model measured) |
| P59 | solves for float GELU, not INT8 | the simulated Hexagon GELU reproduces 89% of held out device outputs bit for bit where correct float16 reproduced 7%, so a GELU finding on `sim:hexagon` is the device's; INT8 one step differences are still only detected (`truestar conform qaihub`), no rule reproduces them |
| P4 to P9, P11, P12, P24, P40 | not a detect or solve question | these rows are about the tool itself (one report, compiler, install, agent output); their Status column says how much of the acceptance test is met |

## 2. Never introduce (the "no new problems" list)

Each item is a failure mode we saw in other tools. Each has a guard test in CI.

1. **No silent fallback.** If an op cannot run on the requested backend, say so loudly. Never quietly run it on CPU and report it as the backend's result.
2. **No false alarms.** Tolerances are per dtype and per op, calibrated on known good runs. A clean model must report clean. Noise from float reordering is labelled as noise, not as a bug.
3. **No silent wrong results of our own.** Every Truestar kernel is checked against the float64 reference interpreter on its first call per input shape signature in each process, per element against the element's own magnitude (docs/KERNEL_IR.md, Verification, lists what a first call check misses). Truestar must never be the source of the bug it hunts.
4. **No heavy overhead.** `truestar check` time is measured and published per model. Current budget: a ResNet50 class model checks in under 10 s on a desktop CPU plus GPU (measured 3.5 s on 2026-11-23 after the comparison fast path; was 27.5 s including CUDA start up). The compiler's runtime overhead is measured against PyTorch eager and `torch.compile` on every release.
5. **No compile time cliffs.** Compile time is a tracked metric with a budget per model size. A change that blows the budget does not merge.
6. **No cryptic errors.** Every error says what went wrong, where in the user's code, and what to try. Errors never speak IR dialect to the user.
7. **No vendor lock in.** No backend gets features the others are denied by design; gaps are documented.
8. **No install pain.** No system CUDA toolkit, compiler or driver beyond what the user's framework already needs.
9. **No breaking changes without notice.** Public APIs follow semantic versioning; deprecations warn for at least one minor release.
10. **No data leaves the machine.** No telemetry by default. Reports contain only what the user chooses to share.
11. **No closed core.** Everything users depend on ships under Apache 2.0 at launch.

## 3. Done means

A feature is done only when all of these hold:

* Its pain ledger row's acceptance test passes against the named baseline, and the numbers are written in the PR.
* The never introduce guards all pass.
* It was run on real models, not only toy ones, on every backend available locally.
* Failures that could not be tested (missing hardware) are listed by name, never implied as passing.

## 4. Status log

### 2026-11-24: training numerics (P52, docs/TRAIN_NUMERICS.md)

Research items 2.5 to 2.8. Muon (`optim.Muon`, eager and compiled from one definition) splits each
update's distance from orthogonal into the coefficients' intended part (0.21 per unit) and the
rounding (bf16: torch 0.010 to 0.011, Truestar eager 0.013, f32 below 1e-5), checked against a
dtype derived estimate; it ties AdamW on the nn_mlp example (held out loss 0.558 against 0.562).
`lpgemm` runs forward, input and weight gradient GEMMs in FP8 (Truestar Triton, cuBLASLt hybrid)
and NVFP4 forward with kernel verdicts below 0.02 of the float32 bound for FP8, weight gradient
error against float64 0.063 to 0.073 (fp8), 0.069 to 0.108 (E5M2 gradients), 0.21 to 0.23 (NVFP4
forward), 0.006 (bf16). Invariant sentinels name a stale delta at 200 to 26,000 times tolerance.
The logit bound is within 1% of a reachable value on pre norm encoders. Tests: 45 in the four new
files. Not done: sharded Muon check, MXFP8, NVFP4 backward GEMMs, spikes integration.

### 2026-11-23: Stage 1 first build (`truestar check` 0.0.1)

Verified on this machine (CPU + RTX 5070 Ti, torch 2.11, CUDA 12.8), 24 tests:

* **No false alarms (never introduce #2):** clean MLP, two head model, tiny transformer, attention
  block, in place activation net and ResNet18 report OK on CPU and CUDA. Real architectures also
  report OK: MobileNetV3, EfficientNet B0, ViT B/16, ResNet50, BERT (random init).
  Two false alarm causes were found on real models and fixed: in place activations (inputs are now
  copied before forward) and noise from TF32 (precision aware thresholds).
* **P1 / P3, exact blame:** injected faults (scale, zeroed channels, NaN, input dependent rows) are
  blamed on the exact module, downstream modules are not blamed; a fault deep inside ResNet18 is the
  only module flagged. Instead of bisection, each module is replayed alone with reference inputs,
  which names the module directly.
* **Functional code:** a bug in a parent's own code (attention math) or in a device only branch is
  blamed on that parent; all 12 attention modules in ViT are checked. A divergent final output can
  never be reported OK.
* **P2:** per input decision flips with near tie separation.
* **Repro:** the generated script reproduces on CUDA (exit 1) and passes on CPU (exit 0), needs only PyTorch.
* **No silent fallback (never introduce #1):** a requested missing backend is an error; unavailable
  backends are listed as skipped; the MPS CPU fallback flag is reported.
* **Tests can fail:** two deliberately broken checkers (never flags, isolation broken) fail 8 and 17 tests.

Not yet verified (no hardware or not built yet):

* MPS, ROCm, Intel XPU, phone NPUs, WebGPU: code paths exist for MPS, ROCm, XPU but are untested.
* Rediscovering pytorch#193487 and pytorch#201476 needs a Mac and an RX 9070 class GPU.
* P11 clean `pip install` on Windows, Linux and macOS not yet tested.
* float16 and bfloat16 models not yet tested.
* Speed: see never introduce #4.

### 2026-11-23: first runs on other hardware

* **Tesla T4 (Kaggle, CUDA 13.0, no TF32):** 24/24 tests pass, model zoo clean.
* **Apple MPS, GitHub macos-15 runner (macOS 15.8.2, torch 2.14.1):** 24/24 tests pass, zoo clean,
  184/184 linear probe cases correct.
* **Apple MPS, GitHub macos-26 runner (macOS 26.7.2, torch 2.14.1): first real divergence found.**
  `truestar check` flagged ResNet18 `fc` (own error 3.2e-2) and every Linear in the transformer
  (8.8e-2 to 1.6e-1), blaming the exact layers, while macOS 15 with the same torch is clean.
  The probe (scripts/mps_linear_probe.py) narrows it to `F.linear` **with bias**: 40/40 cases wrong
  (max relative error 0.87) across every shape tested, while `F.linear` without bias, `x @ w.T` and
  `torch.addmm` are all correct. Caveat: GitHub's macOS runners expose a paravirtual GPU, so this is
  not yet confirmed on physical Apple hardware. Next: confirm on a real Mac, check whether the bias is
  dropped or misapplied, then report upstream with the generated repro.

### 2026-11-23: speed and half precision

* Comparison fast path for finite tensors: ResNet50 check 9.6 s to 3.5 s with CUDA warm,
  EfficientNet B0 25 s to 4.1 s. Results unchanged (29 tests, 5 real models clean).
* float16 and bfloat16 models: clean on CPU and CUDA (ResNet18, transformer), injected faults blamed.
  Two weaknesses found by the new bf16 test and fixed: a 50% error was only SUSPECT (divergent
  cutoff now capped at 10% relative error in every precision), and near tie flips were 10x too
  generous. TF32 note no longer shown for non float32 models.
* Broken checker tests still fail (never flags: 8+, isolation broken: 17+).

### 2026-11-23: phone NPU backend (Qualcomm AI Hub, real hosted Galaxy S24)

* `truestar.remote.qai_hub_backend.check_qai_hub`: compiles the model for a hosted phone, runs it on
  the NPU and, as a control, on the phone CPU, and compares both with the float64 CPU reference.
* ResNet18 with realistic logits on Galaxy S24 (Snapdragon 8 Gen 3, TFLite): CPU 4.1e-7, NPU 3.1e-4
  (float16 level), no prediction flips. OK.
* False alarm caught and fixed before it shipped: a random init MobileNetV3 has logits near 1e-5,
  below float16 resolution, so the NPU returned zeros and the run looked 100% wrong. Such runs are now
  reported as inconclusive with the reason, never as divergent.
* Not yet: per layer blame on the NPU (needs intermediate outputs exported), quantized models, CLI flag.

### 2026-11-23: op level drilldown

Motivated by the macOS 26 MPS finding, where Truestar blamed the Linear module but the probe showed
only `F.linear` with bias is wrong. Built `truestar.opdrill`: the blamed module (only that one) is
traced under a TorchDispatchMode on CPU float64 and on the backend, the op sequences are aligned by
name, and each op that looks wrong is run alone on the backend with reference inputs. The first op
whose own error crosses the threshold becomes `BackendResult.first_divergent_op` (text report and
JSON), and `op_repro.py` calls that single op with saved tensors. Per op thresholds never go below
10x the op's own CPU error in the model dtype, the same calibration modules use.

Verified on this machine (RTX 5070 Ti, CUDA 12.8, torch 2.11), 43 tests (29 before, 14 new):

* **Names the op:** `truestar.testing.OpFault` corrupts one ATen op on one device. Faults in
  `aten.addmm.default` (MLP, also float16 and bfloat16) and `aten._softmax.default` (op 17 of 27 in
  an attention parent) are named exactly. An op only the backend runs (a device branch calling
  addmm where the reference runs mm plus add, like a backend with its own linear kernel) is named
  and marked as not run by the reference.
* **Real models:** injected faults are named in ResNet18 (`aten.convolution.default` in conv1, the
  residual `aten.add_.Tensor` in layer1.0) and ViT B/16 (`aten.addmm.default` in the first
  self_attention), 0.5 s to 3.9 s per check including the drilldown.
* **No false blame:** a module level device branch (DeviceFault, a correct `aten.mul` that only runs
  on the device) names no op and says the module's own code is the cause. Clean modules drilled
  directly name no op: Linear, an attention block, TransformerEncoderLayer (fast path), conv plus
  batch norm (cuDNN ops that have no CPU kernel are skipped, not blamed), LSTM, ResNet18 layer1.
* **Cost:** a clean model is never traced; a faulty one traces exactly one module twice.
* **Op repro:** reproduces under the fault (exit 1), passes with the stock kernel and on CPU (exit 0).
* **Tests can fail:** confirming by isolation removed fails 1 test (the device branch is wrongly
  blamed on aten.mul); blaming the first traced op fails 13 of 14.

Not verified:

* On MPS itself. Whether MPS reports the bug as `aten.addmm.default`, as `aten.linear.default` or as
  another fused kernel depends on how MPS dispatches linear; both the matched and the backend only
  paths are tested, but only on CUDA.
* Only the first divergent module is drilled, not every blamed module.
* Ops that cannot be rerun from copied inputs (stride or storage dependent views) are skipped, so a
  bug inside one of them is reported at module level only.
* Ops that only the reference runs cannot be blamed; they show up in the note instead.


### 2026-11-23: Stage 2 foundation (shape typed IR, importer, reference interpreter)

Built `src/truestar/ir/` (see docs/IR.md): 47 pure IR ops with symbolic shape rules, an importer
for 115 ATen overloads from `torch.export` (export graph and core ATen graph), and a float64
reference interpreter written from textbook formulas. 34 new tests; full suite 51 passed, 12 GPU
tests skipped (run with the GPU hidden, CPU only).

* **Reference matches PyTorch float64** on resnet18, mobilenet_v3_small, mobilenet_v2, two
  TransformerEncoderLayer variants and a two layer TransformerEncoder, in both graph forms:
  worst relative error 1.5e-15 (bar 1e-10). Batch norm stats were randomized so the check is not
  hiding behind a near identity normalization.
* **P9:** shape mismatches are reported at import or build time with the op, both shapes, the
  user's source line and module path, before anything runs. Symbolic dims must be provably equal.
* **Symbolic batch (P5, P7 groundwork):** one import runs at batch 1, 2 and 5 with no re import;
  size arithmetic stays symbolic (`4*B`); range violations are reported.
* **No silent fallback:** unknown ops are all listed in one error with their source lines;
  training mode, attention dropout and unsafe in place ops are rejected.
* **Tests can fail:** removing eps from the reference batch norm failed 7 tests; a permanent test
  swaps in an unbiased variance layer norm and asserts the comparison fails.

Not yet: control flow ops, autodiff, conv other than 2D, embedding and gather ops, data dependent
shapes, serialization, the native front end. The Stage 1 backend probe fails to import when
`CUDA_VISIBLE_DEVICES` is an empty string (torch reports CUDA available with 0 devices); left
unchanged since Stage 1 behaviour is frozen, noted for a fix.

### 2026-11-24: op drilldown on Apple GPUs (GitHub macos runners, torch 2.14.1)

* macOS 26 runner: op drilldown names **`aten.linear.default`** for both divergences (ResNet18 `fc`,
  transformer `linear1`), own error 3.2e-2, not matched by the reference because CPU decomposes
  linear into addmm while MPS runs a fused linear kernel. This explains why `x @ w.T`, `addmm` and
  linear without bias were correct in the probe. A 21 line `op_repro.py` calling
  `torch.ops.aten.linear.default` was generated. Still pending: confirmation on a physical Mac.
* Tests that silently assumed CUDA were caught there: 6 op drilldown tests injected faults into
  `aten.addmm`, which MPS never calls, so the fault never fired on macOS 15. The tests now detect
  which op `nn.Linear` dispatches to on the device under test.
### 2026-11-24: P10 diagnostics for CUDA OOM and distributed hangs (`truestar.diag`)

Verified on this machine (RTX 5070 Ti, torch 2.11, Windows, gloo over CPU processes), 9 new tests,
38 in the suite:

* **OOM names the instance, not the type:** four identical `MLP` blocks, only `blocks.2.mlp` blows up
  and only at step 4. The report names `blocks.2.mlp`, the forward phase, its source line, the exact
  failing line, the requested size, live holders and a steady trend over steps 1 to 3. The original
  `torch.OutOfMemoryError` is re raised unchanged with the report attached. These are real allocator
  OOMs: each asks for 1 TiB so it fails at once without filling the shared GPU.
* **Backward OOM:** a custom autograd function that needs a huge buffer in backward at step 3 is
  blamed on its module instance (`mid.1`) in the backward phase, in a model with in place ReLUs.
* **Outside modules and leaks:** an OOM after `backward()` is reported as outside modules (not
  blamed on the last module whose backward ran), and 2 MiB kept per step is classified as a slow leak.
* **Hangs, real multi process gloo runs:** rank 2 of 4 calling collectives in a different order,
  rank 1 of 3 skipping an `all_reduce` inside an `if`, and rank 2 of 3 raising an exception. Each
  diagnosis names the right rank, sequence number and source line, on every rank that wrote one and
  when rerun after the fact. A clean run reports nothing. Each scenario ends in about 6 s (2 s
  watchdog), and the harness kills any worker still alive.
* **Tests can fail:** four deliberate breaks were each caught: call site taken one frame too high
  (3 tests fail), skip detection removed (1), forward blame on the outermost module (2), backward
  blame on the outermost module (1).

Not verified:

* NCCL itself: one GPU here, so no NCCL run at all. The recorder only wraps the Python API, but NCCL
  specific behaviour (its own watchdog aborting first, CUDA stream semantics) is untested.
* Multi node: shared directory over a network file system untested.
* Collectives issued from C++ (DDP gradient buckets, FSDP internals) are not recorded.
* `async_op=True` collectives are recorded but not timed.
* Gradual OOMs from really filling the card were not provoked, to keep the shared GPU free. On
  Windows the driver's system memory fallback let a 20 GiB request succeed on the 16 GB card.
### 2026-11-24: agent interface (P12)

* `truestar check --format agent`: strict, versioned JSON (`schema_version` 1) on stdout, also on
  exit 2. Blamed module with class source file and line, plus the nearest user class; error vs
  threshold; repro and control commands; ordered `next_actions`. Text output stays the default.
* Stable codes TS001 to TS021 in agent JSON, full `--json` report and the text report.
* `truestar mcp`: tools `list_backends`, `check_model`, `run_repro`, served by the official `mcp`
  package when installed and by a builtin stdio server otherwise. Both pass the same round trip tests.
* Problems found and fixed while building it: on Windows a child process that inherits the server's
  stdin pipe hangs under the SDK (children now get no stdin); several torch processes at once hit
  "paging file too small" (the server and `truestar codes` no longer import torch); a repro that
  cannot be pickled used to turn into "model failed on backend" (now a note, the finding stays);
  `truestar check missing.py` exited 1, not the documented 2; classes in the model file had no
  findable source and could not be unpickled by repros (the file is now registered under its name and
  repros add its folder to `sys.path`).
* Tests can fail: off by one source line fails the user source test; TS010 reported as TS011 fails
  the blame test; removing the repro marker check fails both MCP rejection tests.
* 56 tests pass on CPU + RTX 5070 Ti. Not tested: MCP with a real agent client, MPS / ROCm.
### 2026-11-24: native front end (`@truestar.function`)

Built `src/truestar/frontend/` (see docs/FRONTEND.md): typed Python functions translated from
their AST, never traced, into the IR. Added IR ops `cond`, `loop` (regions with explicit captures,
symbolic trip count or a cond region), `select_dynamic` and `dim_scalar`, with shape rules and
reference interpreter support. 42 new tests; full suite 120 passed on CPU (78 before, unchanged).

Verified for P5:

* **Zero graph breaks by construction.** A data dependent `if`, a loop over a symbolic length and
  a data dependent `while` each become one IR op, so a program translates whole or fails with the
  line. On the same programs torch.compile 2.11 (CPU, counting backend) breaks once on the branch
  (`fullgraph=True` raises) and compiles 7 graphs over 6 calls; Truestar compiles once.
* **No recompile within declared dims.** Tests count compiles: one compile serves batch 1, 2, 3, 5,
  7 and 64 and an RNN over lengths 1, 3, 4, 5, 9 and 17. torch.compile recompiles the RNN for each new length
  (even with `dynamic=True`) and stops compiling after its recompile limit of 8.
* **Honest control:** on branch free code torch.compile has no breaks and needs 2 or 3 compiles,
  and a manual `torch.cond` rewrite of the branch removes the break. The win is on control flow.

Verified for P9:

* Shape mismatches raise at decoration, before any data, with the user's file and line, the op,
  both operand types and a fix where obvious (transposed matmul operand, linear weight as
  `[in, out]`, transposed elementwise operands, wrong layer norm shape). Covered: matmul, linear,
  broadcasting, symbolic dims that are not provably equal, a branch never taken at run time, a
  loop body, a carried value changing type, branches disagreeing, return and assignment
  annotations, a whole tensor used as a condition. Line numbers are asserted in tests.
* Errors are machine readable (`to_dict()`, schema `truestar.frontend.error/1`) with a stable code.
* A branch on a symbolic dim is rejected instead of silently specialized.

Never introduce:

* **#1 no silent fallback:** unsupported Python (try, with, lambda, comprehensions, break,
  subscript assignment, unknown calls, f strings, iterating a tensor, return in a symbolic loop)
  raises with the line; 9 constructs are tested.
* **#3 no wrong results of our own:** every construct matches the same source run by PyTorch in
  float64 to 1e-10, with both branches and zero trip loops exercised.
* **Tests can fail:** swapping the cond branches in the translator failed 7 tests; ignoring a
  loop's start offset failed 1. Permanent guards monkeypatch the reference `cond` and `loop` and
  assert the comparisons fail.

Not verified or not built:

* Speed: execution is the float64 reference interpreter only; the code generator plugs in behind
  `Executor` later. Compile time was not measured beyond a decoration taking 2 to 160 ms here.
* Autodiff, dtype casts, gather and scatter, slice updates in loops, `break`, closures, max and
  min reductions, data dependent shapes, loops whose carried value changes shape.
* Importing `torch.cond` and `while_loop` from `torch.export` graphs into the new IR ops.
* Real models written in the native front end beyond the test layers (MLP, attention, RNN,
  Newton iteration).
* The comparison runs Dynamo with a counting backend, not Inductor, so it measures breaks and
  recompiles, not kernel speed. With the GPU hidden by an empty `CUDA_VISIBLE_DEVICES`, Dynamo
  crashes while probing Triton; the script forces `torch.cuda.is_available()` to False.
### 2026-11-24: kernel layer (loop IR, fusion, Triton and numba backends, verified kernels)

Built `src/truestar/kernel/` (interface in docs/KERNEL_IR.md): a loop level IR with named axes,
symbolic sizes and nested reductions; template free fusion by inlining, where the GPU scheduler
picks a loop shape (tile, contraction, online softmax attention) from the IR structure; Triton and
numba code generation; a cache keyed on the canonical IR; float64 verification of every compiled
kernel on first use per shape signature.

Measured on this machine (RTX 5070 Ti, torch 2.11, triton 3.7.1), `scripts/bench_kernels.py`, cold
caches for every tool, run times are medians after a 0.3 s spin:

| case | Truestar | eager | torch.compile | hand written / template |
|---|---|---|---|---|
| (a) tanh GELU + residual, 8192x4096 fp32 | 0.534 ms, compile 0.54 s | 4.38 ms | 0.522 ms, compile 2.12 s | none |
| (b) causal attention B2 H16 S2048 D64 fp16 | 0.299 ms, compile 0.56 s | 6.65 ms (math) | 1.54 ms (math), compile 0.63 s | SDPA flash 0.456 ms; FlexAttention 0.254 ms, compile 1.59 s |
| (c) causal differential attention, same sizes, D 32x2 | 0.348 ms, one kernel, compile 1.07 s | 15.2 ms (math) | 2.19 ms (math), compile 1.57 s | 2x SDPA + combine 0.740 ms; 2x FlexAttention + combine 0.374 ms, compile 2.36 s |

* **P8:** the FlexAttention inexpressible variant (c) runs as one fused kernel, 7% faster than two
  FlexAttention calls plus a combine, 2.1x faster than two hand written SDPA calls, 6.3x faster
  than torch.compile of the same math. Loss: plain causal attention (b) is 18% slower than
  FlexAttention (0.299 vs 0.254 ms) though 1.5x faster than SDPA; elementwise (a) ties
  torch.compile (2% slower).
* **P6:** cold compile plus first launch 0.5 to 1.1 s per case versus 0.6 to 2.4 s for torch.compile.
  Recompile after a one constant edit 0.21 s (a), 0.53 s (b), 0.80 s (c) versus 0.65 s, 0.55 s,
  1.55 s. A new process with warm caches compiles the attention block in 0.41 s versus 4.31 s for
  torch.compile. In process cache hit, including graph build and fusion: 1.2 ms. An edit
  recompiles only the kernels whose canonical IR changed (tested).
* **Never introduce #3:** every compiled kernel is checked against the float64 interpreter on its
  first call per shape signature; a mismatch raises `KernelMismatchError` naming the kernel. Max
  relative error vs float64 in the benchmark: 7.6e-8 (a), 3.5e-4 (b, identical to SDPA and
  FlexAttention), 3.7e-4 (c, lower than every PyTorch path at 5.8e-4 to 8.2e-4). The verifier
  found two real codegen bugs while this was built (CSE merging two reduction accumulators in the
  looped tile kernel; a tile shape that accepted a matmul with no data reuse), both fixed before
  commit. Cost: the float64 check takes 1.7 s (a), 4.5 s (b), 9.1 s (c) on first use at these sizes.
* **Never introduce #1:** a kernel that fits no GPU loop shape raises `Unschedulable` with the reason;
  there is no CPU fallback. float32 dots stay IEEE unless TF32 is requested.
* **Tests:** 32 kernel test functions, 142 cases with backend and dtype parametrization (141 pass,
  1 skipped), every op and fusion compared with an independent float64 PyTorch computation. Full
  suite: 170 pass, 1 skipped. Tests can fail: skipping the diagonal key block in causal attention fails 18 tests through `KernelMismatchError`;
  an off by one in the causal bound fails 12 plan tests; a CPU max reduction that starts at 0
  instead of minus infinity passed every test at first (all max inputs were positive and softmax is
  shift invariant), so a one signed rows test was added and now fails 3 tests.

Not yet verified or not built:

* No autotuning: fixed block heuristics. No backward pass.
* CPU backend is correct but not benchmarked against PyTorch CPU; its attention is a plain loop nest.
* Not run on ROCm, MPS or other GPUs; Triton codegen is only tested on CUDA (sm_120).
* Not yet wired to the tensor IR (being built in parallel); the graph layer stands in for it.
* torch.compile's first compile in a fresh process paid up to 18 s of one time setup in one run
  (2.1 s in the run above); the table reports the run above.

### 2026-11-24: P11 clean install on Linux and Windows (GitHub Actions, `ci.yml`)

`.github/workflows/ci.yml` builds the sdist and wheel, checks the wheel (every subpackage of
`src/truestar` present, no tests, examples, scripts or caches, readme metadata, console script; a
wheel rebuilt from the sdist has the same 28 files), then on each OS and Python makes a fresh venv,
installs torch 2.14.1 from the PyTorch CPU index, installs the wheel, and from a folder outside the
checkout runs `truestar --help`, `truestar backends`, `truestar codes` and
`truestar check examples/cpu_check.py --format agent` (asserts `schema_version` 1, status ok, exit 0).
Then torchvision 0.29.1 (CPU) and pytest are added and the suite runs against the installed wheel.

Green on run 45341273728 (commit a15896e), all four jobs:

| runner | Python | CPU torch install | truestar wheel install | `truestar check` | tests |
|---|---|---|---|---|---|
| `ubuntu-latest` | 3.10 | 21.0 s | 0.5 s | 2.7 s | 73 passed, 35 skipped |
| `ubuntu-latest` | 3.12 | 21.5 s | 0.4 s | 2.5 s | 73 passed, 35 skipped |
| `windows-latest` | 3.10 | 29.3 s | 1.3 s | 3.8 s | 73 passed, 35 skipped |
| `windows-latest` | 3.12 | 39.2 s | 1.5 s | 3.8 s | 73 passed, 35 skipped |

* Wheel 82,963 bytes (29 files with metadata), sdist 92,358 bytes. Install times are with a warm pip
  cache; the first run with a cold cache took 20.9 s to 39.6 s for torch, so the cache saves little
  and the time is mostly unpacking torch. Whole jobs take 1.5 to 4 minutes.
* All 35 skips are tests that need a GPU and say so; no skip is for a missing package.
* OS specific failure fixed: on Linux two hang tests failed (skip and crash scenarios) while
  Windows passed. The diagnosis was right on both; only the exit code differed. When a peer process
  exits, gloo on Linux raises "Connection closed by peer" at once, so a waiting rank ends with that
  error (exit 1) after the recorder writes its diagnosis, before the 2 s watchdog would exit with
  code 86. On Windows gloo keeps blocking until the watchdog. The tests now accept either ending for
  a waiting rank, as long as it fails and its diagnosis was triggered by the collective's error. No
  change to `src/truestar` was needed; no packaging fix was needed either.
* The workflow had to run on push to `wt/ci` for this proof, because GitHub only dispatches a
  workflow that already exists on the default branch. That temporary trigger was removed afterwards.

Not verified:

* macOS: the `macos-15` job exists behind the `macos` dispatch input and was not run, to save macOS
  minutes. P11 stays open for macOS until it runs.
* A real `pip install truestar` from an index: the package is not published, so CI installs the
  built wheel file. The dependency resolution is the same, the download source is not.
* Linux arm64 and Windows arm64, Python 3.9, 3.11 and 3.13, and torch versions other than 2.14.1.
* No LICENSE file and no license metadata in the wheel yet, while never introduce #11 says Apache
  2.0 at launch. Left for the owner to add before any public release.

### 2026-11-24: export path (P7, `truestar export`, `.tsx` artifacts)

Built `truestar.ir.serialize` (versioned single file format), `truestar.export` (exporter, runner)
and `truestar export` / `truestar run` (see docs/EXPORT.md). 41 new tests in tests/test_export.py;
full suite 162 passed, 36 skipped (GPU tests, run with the GPU hidden, CPU only).

* **The verified artifact is the shipped artifact:** the program parsed back from the artifact's
  own bytes is checked against the PyTorch module in float64 at the example batch and at a second
  batch size; on any disagreement nothing is written. A reference relu off by 0.1% and a linear
  wrong only above batch 2 are both refused by permanent tests. Evidence (inputs sha256, max
  errors, tolerance, program hash, torch version, UTC date) is stored in the file.
* **P7 acceptance:** dynamic batch survives (exported at batch 2, run at batch 7 and 1, float64
  relative error under 1e-12); 46 of 63 swept torchvision classifiers up to 100M params export and
  round trip (worst float64 error 6.4e-15), 16 are refused with the missing ops named (avg_pool2d,
  chunk, swin ops). Python branches on tensor values or on the batch size are refused with the
  branch's line; `torch.cond` is refused with its line until the importer maps it.
* **Baseline, torch.onnx.export (torch 2.11, onnxruntime 1.20.1):** the TorchScript exporter
  silently bakes in both branch kinds (wrong by 3 and 2 on the other side). The default dynamo
  exporter refuses the value branch but, for the batch branch, applies torch.export's suggested
  fix behind the scenes and writes an ONNX file declaring `x: [B, 3]` whose graph is one `Add`:
  wrong at batch 7 with no warning.
* **Format:** deterministic bytes, program hash independent of metadata, loud errors for unknown
  versions, bad magic, truncation, a flipped bit (naming the tensor) and edits with a refreshed
  checksum. Ops serialize generically (an op the IR does not define round trips); unencodable
  attributes and unknown structural fields raise with the op, node and attribute named. Native
  front end programs (`cond` regions with two outputs, a `loop` over `T - 1`) round trip bit for
  bit and run in a fresh process.
* **Standalone:** a fresh process with the model source deleted, the model module and
  torchvision import blocked, and torch.export and TorchScript loaders replaced by functions that
  raise, runs the artifact and matches eager PyTorch.
* **No silent wrong results (#3):** a vacuous check (every output zero, as torchvision's zero
  initialized ViT head gives, or outputs independent of the input) is flagged in the evidence.
* **Tests can fail:** skipping the checksum failed 2 tests, skipping the version check 1, dropping
  the second batch check 3, dropping extra outputs 1. Each was restored.

Not verified or not built:

* Any compiled executor; the runner uses the float64 reference interpreter (seam: `register_executor`).
* Importing torch.export `cond` and `while_loop` into the IR control flow ops.
* Keyword inputs, non tensor inputs, symbolic dims other than the batch, output pytree structure.
* The batch symbol is bounded (default 1024) because the importer fails on an unbounded `Dim`.
* Models above 100M params were not swept (float64 reference time).

### 2026-11-24: autodiff as a compiler pass (`truestar.ir.autodiff`)

Built `grad(program, wrt=[...])` (see docs/AUTODIFF.md): reverse mode differentiation as a pure IR
to IR transformation. The backward pass is ordinary IR nodes behind the forward nodes, with no
tape at run time, shape checked by the IR's own rules as it is emitted. 32 direct rules, 3 ops
lowered to primitives when differentiated (`batch_norm`, `layer_norm`, `attention`), 13 ops with no
gradient, and one new op (`erf`, registered from the autodiff module). Conv gradients are written
with `conv2d` itself, so no new kernels. 108 new tests, CPU, float64, torch 2.11.

* **Every rule against finite differences:** 74 cases over all 35 differentiable ops, fourth order
  central differences of the reference interpreter. Worst relative error 1.6e-10, median 9e-13,
  bar 1e-8.
* **Whole models against `torch.autograd` in float64,** input and every parameter, export graph and
  core ATen graph: MLP 1.9e-16, resnet18 with inference batch norm 4.2e-15, two
  TransformerEncoderLayer variants 7.3e-16, mobilenet_v3_small 8.6e-15, a feature zoo 1.9e-15.
  Target 1e-8, asserted bar 1e-10.
* **Symbolic batch (P5, P7):** each model result is one transformed program run at batch 1 and 3.
* **Training:** an MLP with its loss, imported from `torch.export`, trained 10 SGD steps with the
  reference interpreter. Loss falls, and loss and parameters match `torch.optim.SGD` to 1e-13 at
  every step.
* **No silent gaps (never introduce #1):** an op with no rule on the gradient path fails naming the
  op and the user's source line, all listed at once. Unsupported symbolic cases fail by name.
* **Dead code:** tests assert the exact backward ops for dead branches, unrequested operands and
  values behind comparisons (none are emitted).
* **Found in PyTorch:** the float64 `hardsigmoid` backward multiplies by 1/6 rounded to float32,
  a 3.0e-8 relative error in every gradient through it (torch 2.11). Our rule is exact; finite
  differences agree with ours at 8e-12. Not yet reported upstream.
* **Tests can fail:** three manual breaks (softmax rule without its sum term, conv input gradient
  without the kernel flip, mean without the division) failed 14, 14 and 16 tests and were
  restored. Permanent tests break the tanh rule and the layer norm eps and assert the checks fail.

Not verified or not built:

* Gradients on any backend other than the float64 reference interpreter (none exists yet), so
  nothing here says how float32 or half precision gradients behave.
* Conv and pooling gradients over a symbolic height or width, `cat` behind a symbolic sized
  operand, `slice` with a step over a symbolic dim: these raise.
* Training mode batch norm, dropout, optimizer state in the IR, control flow.
* Second order beyond one Hessian vector product test.
* The backward code is not simplified (resnet18: 69 nodes to 1165).

### 2026-11-24: INT8 agreement and per layer blame on phone NPUs (P2, P3)

Built `truestar.remote.qai_hub_quant` (AI Hub quantize job to w8a8, TFLite compile, all inputs in
one inference job per device and unit, job cache keyed by model hash), `truestar.remote.agreement`
(per input agreement with near and exact ties, INT8 aware per layer blame) and a deterministic
ResNet18 task in `truestar.remote.synthetic`. Real numbers and job ids are in docs/NPU.md.

* **P2 acceptance met on real phones:** the same INT8 TFLite file with the same scales agrees with
  itself on 242 of 256 inputs between NPU and CPU of a Galaxy S24, and the same on a Galaxy S25
  (945 per 1000, paper: 939), while accuracy moves at most 0.4 points (79.3% to 80.5% across all
  INT8 runs, float64 80.1%). Phone CPUs agree with each other 256/256; float on the NPU agrees with
  float64 256/256.
* **Per layer:** 11 block taps plus logits as extra outputs (cap `MAX_TAPS = 12`). Float NPU stays
  under the float16 floor at every tap. INT8 NPU already differs from the host simulation of the same
  QDQ model on 4.4% of elements (1 LSB) at the first conv, where the phone CPU differs on 0.00%;
  the drift first exceeds one LSB rms at `layer2.0`. Reported as accumulated rounding, not a bad
  layer.
* **Cost:** 18 AI Hub jobs and 3 dataset uploads for the whole study; an unchanged rerun submits 0
  jobs and takes 25 s.
* **Not an artifact of ours:** the INT8 NPU vs CPU gap is confirmed against an independent host
  simulation of the same QDQ model, which the phone CPU matches almost exactly, so it is not a tap
  ordering or layout mistake (outputs are sorted naturally, NHWC outputs are transposed back).
* **Bug found in the old backend while building this:** `check_qai_hub` sorted output names as
  strings, which scrambles models with 11 or more outputs (`output_10` before `output_2`). It only
  reads one output so no result changed; it now sorts naturally and a test covers the sort.
* **Tests can fail:** 17 offline tests with a fake AI Hub client (no qai_hub import). Breaking the
  agreement counting four ways (near tie rule, tie tolerance, agree count, dropped disagreements)
  failed 1 to 3 tests each; making the INT8 layer threshold 3 LSB failed the INT8 blame test.

Not verified:

* One model, one seed, 256 inputs: per pair counts sit in a wide noise band; only the ordering of pairs is solid.
* TFLite only; QNN context binaries and ONNX Runtime QNN are untested.
* Taps change the compiled graph; whether they change the NPU's rounding was not measured.
* The NPU rounding rule is inferred from the 1 LSB pattern; AI Hub does not expose kernels.
### 2026-11-24: IR op coverage (torchvision zoo and decoder transformers)

Added 26 IR ops (77 now: `avg_pool2d`, `adaptive_max_pool2d`, `group_norm`, `rms_norm`, `index`,
`index_select`, `gather`, `slice_scatter`, `select_scatter`, `pad`, `cast`, `cumsum`, `triu`,
`tril`, `vector_norm`, `pow`, `cos`, `sin`, `log`, `abs`, `reciprocal`, `erf`, `fmod`,
`remainder`, `maximum`, `minimum`), each with a shape rule on symbolic dims and a float64 reference
written from its formula, plus symbolic `arange(T)`. The importer handles 211 ATen overloads (was
115); chunk, split, roll, einsum, stack, repeat, embedding and instance norm lower onto IR ops. See
docs/IR.md. 44 new tests in tests/test_ir_ops.py; full suite 383 passed, 1 skipped (CPU only).

* **P7 acceptance, torchvision sweep (scripts/export_zoo.py):** before, 46 PASS, 1 WEAK, 16
  REFUSED; after, 62 PASS, 1 WEAK (googlenet, unchanged), 0 REFUSED, 17 not swept (over 100M
  params). Worst float64 error 6.0e-15. The 16 formerly refused models also match from the core
  ATen graph.
* **Decoder only transformer** (embedding, rotary, RMSNorm, causal attention, SwiGLU, tied head):
  one import with `[B, T]` symbolic matches PyTorch float64 at three (B, T) pairs, worst 1.9e-16,
  in both graph forms.
* **No silent wrong results (#3):** in place changes through slices (Swin's mask) are rewritten as
  `slice_scatter` only when exact; a view of the same storage read after the change, a slice of
  a program input, or a base that is itself a view is refused. The plain in place check now also
  refuses a change another view reads later (before, that case was not detected).
* **No silent fallback (#1):** reflect padding, einsum diagonals, strided views that are not a
  reshape, boolean mask indexing and rolling or splitting a symbolic dim raise with the ATen op
  and the user line; each has a test.
* **Tests can fail:** three broken reference ops fail the bar in a permanent test; removing `eps`
  from the real `rms_norm` reference failed 1 test, then it was restored.

Not verified or not built:

* Reflect and replicate padding, `flip`, sort, top k, scatter with a tensor index, 1D and 3D
  pools, grouped query attention, data dependent shapes.
* Hugging Face models themselves (not installed, no downloads); the decoder is written with plain
  torch.nn in their style.
* `cast` to float16 or bfloat16 does not round in the float64 reference.
### 2026-11-24: port check (`truestar port`, P16)

Built `truestar.portcheck` and `truestar port reference.py candidate.py` (see docs/PORT_CHECK.md):
two `nn.Module` implementations of the same model with different names and layouts are compared
through a module map inferred from the weights. Weights are matched by content (equal bytes,
transpose, row permutation, row block of a fused tensor in either direction, dtype cast, closest
same shape tensor, merged adapter `c * B @ A`), identical tensors are paired by the order the
forward first reads them, modules map when their matched weights cover each other. Both models run
in float64 on CPU, each mapped candidate module is replayed alone with the reference module's
inputs, thresholds are 10x the reference module's own float32 error (never below 1e-9), and the
blamed pair is drilled at the ATen op level. 30 new tests in tests/test_portcheck.py on a Llama style
decoder written three times (tests/port_models.py); full suite 477 passed, 1 skipped, CPU only.

* **Correct ports clean:** renamed modules; fused `wqkv` and `w13` with RMSNorm written inline;
  interleaved rotary with permuted q and k rows (matched as a row permutation); a fused reference
  against a separate candidate; trained (random) norm weights; LoRA merged correctly into separate
  and fused weights. Final output error 4e-16 or less in float64 (3.9e-8 for merged LoRA, float32
  rounding of the merged weight), no module over threshold.
* **P16 acceptance, each bug named at its module with the reason:** eps 1e-6 versus 1e-5 at
  `blocks.0.norm1` (class) and at `layers.0` with the inline norms listed (`aten.add.Tensor`,
  constant 1e-06 instead of 1e-05); rotary pairs interleaved without the weight permutation at the
  attention module (`aten.slice.Tensor` takes every 2nd element where the reference takes a half);
  rotary base 500000 (`aten.pow.Scalar`, 500000 instead of 10000); attention scale dividing by the
  head dim (`aten.div.Tensor`, 8 instead of 2.82843, flagged as a missing square root); a
  bfloat16 round trip in a norm (casts only the candidate runs). Each in both port layouts where it
  applies. A bug in one layer only names only that layer.
* **Weights:** a dropped LoRA (TS162 lists all 8 adapter tensors, blamed at `q_proj` against `wq`,
  or columns 0:32 of the fused `wqkv`); LoRA loaded but never applied (TS163); LoRA merged without
  its scaling (blamed, the factor 1 is reported); one weight rounded to bfloat16 (TS165); an extra
  candidate weight is loud even when outputs agree.
* **Baseline:** comparing final logits flags every injected bug only as "outputs differ" (relative
  error 7e-3 to 1.3e-1) and names nothing.
* **Agent JSON:** `--format agent` with `"command": "port"`, codes TS160 to TS169, blamed module with
  file and line, op finding, next actions; strict JSON; exit codes 0, 1, 2 tested.
* **Tests can fail:** permanent tests swap one module pair and disable fused slice matching and
  assert the check is not clean. Manual breaks, each restored: one sided module covering failed 1
  test, reversed pairing of identical tensors failed 7, replaying parents whose children are bad
  failed 17.

Not verified or not built:

* Real ports (Hugging Face against vLLM, MLX or JAX conversions); only the three test
  implementations were checked.
* Fused tensors along a dim other than 0, buffers, modules without weights, inputs whose layout
  differs between the two sides (kept as the candidate's own values and named), MCP tool.
### 2026-11-24: training on the IR (`truestar.train`)

Built losses, optimizers, mixed precision and data parallel gradient averaging as IR, and a
training step builder that composes forward, loss, backward, all reduce, clipping, the update and
loss scaling into one program per step (see docs/TRAINING.md). A small trainer runs it through an
executor (default the reference interpreter) with a warmup cosine schedule, logging, NaN and Inf
attribution and `.tsx` checkpoints. Three ops added through the IR registries from
`train/ops.py`: `cast`, `log_softmax`, `all_reduce`. 46 new tests, CPU, torch 2.11; full suite
493 passed, 1 skipped.

* **Optimizers against torch.optim in float64,** 20 steps, loss and every parameter after every
  step, on an MLP, a CNN and a TransformerEncoderLayer with a vocabulary head, all imported with
  `torch.export`: SGD with momentum and dampening, SGD Nesterov, AdamW and Lion, worst 5.4e-15
  (bar 1e-10). Adam with L2, amsgrad and clipping 4.2e-16. Lion is checked against the paper's
  algorithm in plain torch. The attention key bias, whose exact gradient is zero, is excluded
  with a test that shows its gradient is rounding noise.
* **Losses:** cross entropy with label smoothing and ignore index, MSE and the causal LM loss
  match torch to 1e-14 in value and gradient; logits of 1e4 stay finite.
* **Mixed precision against torch.autocast and GradScaler:** loss to 2.4e-7, 99.6% (float16) and
  100% (bfloat16) of the master weight gradient entries bit identical, updates from the same
  gradients to 1.2e-7, and skips, scale and growth counter identical to torch on every step of
  20 (5 or 6 overflows each run).
* **Data parallel (P10):** 2 gloo ranks match 1 rank on the concatenated batch to 6.2e-16; a rank
  that runs out of data is diagnosed by the collective recorder with the user's line.
* **Checkpoints (P7):** same state gives the same bytes; resume is bit identical to a run that
  never stopped.
* **No cryptic errors (#6):** a NaN names the first op, its ATen op and user line, and the step is
  not applied.
* **Tests can fail:** manual breaks (Adam without bias correction, Lion without decay, a loss
  scaler that never skips) failed 8, 3 and 3 tests and were restored; two permanent tests break
  Adam and Nesterov and assert the comparison fails.

Not verified or not built:

* Any executor other than the float64 reference interpreter; mixed precision is verified through
  a precision emulation pass, not on half precision hardware.
* The amp policy on conv, attention and the norms (lists follow measured CPU autocast, no amp
  training test); CUDA's autocast lists.
* Training mode batch norm and dropout; NCCL and multi node data parallel; overlap of the all
  reduce with the backward pass.
### 2026-11-24: batch invariance check (`truestar invariance`, P14 part a)

Built `truestar.invariance` and the CLI command (see docs/INVARIANCE.md): each row runs alone and
again repeated, inside batches of 2, 7 and 64 at the first, middle and last position, in a zero
padded batch and, for token models, with the sequence right padded. Rows are compared bitwise and
numerically, argmax and greedy token flips are counted, then the first batch dependent module is
found by replaying each module with the row's own inputs substituted, and the op inside it the same
way. Flag sets are verified by rerunning. Codes TS050 to TS054. 30 new tests on CPU and the RTX 5070
Ti (CUDA 12.8, torch 2.11).

* **Names the op on real backends:** CUDA float32 `aten.addmm` breaks invariance in an MLP, a
  TransformerEncoder (in self_attn, linear1, linear2) and a GPT style decoder at batch 2 already;
  CPU float32 `aten.addmm` (MKL, position dependent) in the MLP; CPU flash attention
  (`aten._scaled_dot_product_flash_attention_for_cpu`) under sequence padding in the GPT, float32
  and bfloat16. The CPU TransformerEncoder and every bfloat16 model on CUDA were invariant.
* **No tested flag fixes any of these:** deterministic algorithms, no reduced precision reductions,
  math attention, one CPU thread and oneDNN off were each rerun and left the pairs unchanged.
  `one_thread` was verified as the fix only for a thread split case.
* **Clean model reports clean:** the same GPT built from `truestar.invariant_ops` is bitwise
  identical in all 22 (row, context) pairs on CPU and CUDA, including sequence padding.
* **Injected faults named:** an op whose output scales with the batch size (`aten.mul.Tensor` in
  `scale#0`) and a real batch mix (`aten.mean.dim` in a user module, TS052, user file and line,
  argmax and greedy token flips).
* **Reference ops (proof of concept):** fixed 64 row GEMM tiles and fixed 64 element reduction trees.
  Overhead 2.8x to 15x for linear, 52x to 94x for attention, 4x to 146x for softmax and layer
  norm; a whole two block decoder 13x on CPU and 9.2x on CUDA. Far above the 20% bar, which is for
  compiled kernels.
* **Found while building:** CPU `sigmoid`, `silu` and tanh GELU give other bits in the scalar tail
  than in the vectorized body; a CPU float32 GEMM with a transposed right operand rounds half the
  positions of a 64 row tile differently; a permuted operand that is a view at batch 1 and a copy
  at batch 2 changes the CUDA GEMM result.
* **Tests can fail:** making op drilldown return nothing failed 11 tests, marking every batch
  dependent module as its own cause failed 10; both restored. A permanent test removes the row
  substitution and asserts the report changes.

Not verified or not built:

* Tensor parallel degree, other GPUs, ROCm, MPS.
* Truestar compiled batch invariant kernels and the float64 verifier's bitwise invariance check
  (P14 part b): untouched, the kernel layer was not edited.
* Sequence padding with attention masks, left padding, KV cache decoding.
* No prediction or token flips occurred naturally in the random weight models.
### 2026-11-24: record and diff across environments (P13, `truestar record`, `truestar diff`)

Built `truestar record model.py -o run.tsrec` and `truestar diff` (see docs/RECORD_DIFF.md). A
recording holds per module call the inputs (copied before forward, tensor identity kept), outputs,
input hash and parameter hash, optional ATen op traces, the plain unhooked model output and an
environment fingerprint (versions, TF32 and fp32_precision settings, determinism flags, driver,
GPU). `diff --replay` feeds every module its recorded inputs alone in the current environment;
`diff a b` isolates modules by equal input hashes. Changes are classified identical, noise,
precision or divergent, then op drilldown names the op. Text, full JSON and agent JSON with
`next_actions` (`pin_version`, `report_upstream` with an op repro, `restore_precision_flag`).
Codes TS060 to TS071. 27 new tests, CPU and CUDA (RTX 5070 Ti, torch 2.11.0+cu128, cuDNN 91900).

* **Injected op change:** `OpFault("aten.addmm.default")` is named at the first faulty module and
  op, class divergent, offline and in replay, on resnet18 (`fc#0`), a TransformerEncoder
  (`layers.0.linear1#0`) and a small GPT style decoder (`blocks.0.mlp.0#0`); the same under TF32,
  where the TF32 convolutions are still labelled precision rather than hidden.
* **TF32 toggle:** named and classified precision (TS061, exit 0, exit 1 with `--strict`),
  explained by the fingerprint keys: resnet18 `layer1.0.conv1#0` with `aten.convolution.default`
  (own change 2.98e-04 against a 1e-4 noise band and 5e-3 TF32 band), both transformers at the
  attention module with `aten._native_multi_head_attention.default`.
* **Identical environments report clean:** two recordings are byte identical on CPU and CUDA,
  and both diff modes report identical with no blame on all three models.
* **Baseline:** `torch.allclose` on the final logits fails for the TF32 change, which we classify
  as an explained precision change, and returns one bool that names no module and no op in
  every case.
* **No false alarms (never introduce #2):** a 1e-6 relative op change is noise, exit 0. TF32
  flags never explain a CPU recording.
* **No silent wrong results (never introduce #1, #3):** forward hooks switch off
  TransformerEncoderLayer's fused path, so the instrumented run can differ from production; this
  is detected and reported (TS071) and the baseline uses the plain path. Replay attaches the same
  hooks so modules take the recorded path. Lost tensor identity in replay made
  nn.MultiheadAttention take another kernel (noise in the same environment); fixed by keeping
  identity, now bitwise identical.
* **Cost (never introduce #4):** resnet18 at 2x3x64x64 on CUDA records in 0.64 s and replays in
  0.31 s without op traces; 5.3 MiB, 52 MiB with `--ops all`, 7.5 MiB with `--ops all
  --outputs-only`.
* **Tests can fail:** four manual breaks (no precision class, noise band ignored, parents own
  their children's changes, precision explained on CPU) failed 5, 2, 8 and 1 tests and were
  restored.

Not verified or not built:

* Two real torch, cuDNN or Triton versions were not compared (one venv on this machine, no
  downloads). The pytorch#181169 and #157274 cases are reproduced only in kind.
* Precision explanation is per run, not per op family.
* Backward passes, training mode, random ops in eval, opaque or sparse inputs.
* MPS, ROCm and XPU replay were not run (no such hardware here).
### 2026-11-24: P18 backward check (`truestar check --backward`)

Built `truestar.grad_check` and `truestar check --backward [--steps N]` (docs/GRAD_CHECK.md). Each
module call's backward is replayed alone with float64 reference inputs and reference upstream
gradients (captured by tensor hooks in a float64 CPU run), on every backend, and every input and
parameter gradient is compared with float64. New codes TS180 to TS184, agent JSON with
`"mode": "backward"`, a backward repro script, and `truestar.gradfault.BackwardOpFault`, an injector
that fires only inside backward. 32 new tests; CPU plus RTX 5070 Ti, torch 2.11, CUDA 12.8.

* **Acceptance, backward only fault:** `aten.mm` corrupted only in backward is blamed on `fc2#0`
  (the first linear to run in backward), op `aten.mm.default` in `AddmmBackward0`, TS180, while
  `truestar check` on the same model reports OK. Passes on CPU in float32 and bfloat16 and on CUDA
  in bfloat16. Example: `TS180 module fc2#0 (Linear) gradient of param weight: own error 3.01e-01 vs
  threshold 4.0e-02, norm ratio 1.258, cosine 0.9962; its forward matches (error 2.17e-03)`.
* **Acceptance, the N6 failure mode:** a softmax backward that leaks its row sum term in a
  transformer block is blamed on the attention module's own code, op
  `aten._softmax_backward_data.default` in `SoftmaxBackward0`, with its q, k, v children clean, on
  CPU and CUDA in float32 and bfloat16. The total gradient norm of that module moves by 0.2% (cosine
  0.9993), so a grad norm monitor would not see it. Baseline `detect_anomaly` finds nothing: no NaN.
* **Acceptance, clean models (never introduce #2):** MLP, the transformer block,
  `nn.TransformerEncoderLayer` and resnet18 (32x32) report OK on CPU and CUDA in float32 and
  bfloat16, resnet18 also in float16. Off the suite: resnet50 (64x64) on CUDA in float32 with TF32
  convolutions, float16 and bfloat16 reports OK.
* **False alarms found and fixed on the way:** bfloat16 and float16 on the CPU backend flagged max
  pool and ReLU heavy blocks (argmax routes and masks flip when inputs round), and resnet50 float32
  with TF32 convolutions flagged the whole model replay at 0.11. The fix is a second noise floor:
  the same replay in float64 with weights, inputs, upstream gradients and every op result rounded to
  the model dtype (convolution inputs to TF32 when enabled). It matches the CPU bfloat16 trajectory
  to 0.2% over 20 SGD steps. Gradients that are zero in exact math (attention's key bias) are
  measured against the call's largest gradient.
* **Slow accumulation (`--steps`):** a backward error growing 0.002 per pass is not over threshold in
  the bfloat16 single step replay but is flagged as TS183 after 20 steps (gap to a correct bfloat16
  run 0.0075 at step 1, 0.049 at step 20). Clean 12 step loops raise nothing on CPU or CUDA.
* **Attention on the GPU, reported honestly:** `F.scaled_dot_product_attention` backward in bfloat16
  at (2, 8, 1024, 64), causal and not, is OK. PyTorch picks the memory efficient kernel here; this
  Windows build has no flash attention kernel, so FlashAttention itself was not tested. Relative
  error of dq, dk, dv: efficient 4.4e-3, 3.8e-3, 1.8e-3; cuDNN 4.6e-3, 4.5e-3, 3.8e-3; math 4.2e-3,
  4.2e-3, 3.4e-3; the emulated correct bfloat16 floor is about 1e-2; norm ratio within 1e-4.
* **Agent JSON (P12):** blamed module with gradient name, norm ratio, cosine, forward status and user
  source; blamed op with autograd node; per call `grad_norm_ratio`; trend; repro commands;
  `injected_faults`; `next_actions` (`confirm_repro`, `mitigate_fp32`, `inspect_source`, `pin_torch`,
  `watch_grad_norm`, `check_forward`, `injected_fault`). The backward repro of a user
  `autograd.Function` with a wrong backward exits 1 on its own with only PyTorch.
* **Tests can fail:** three manual breaks of the backward replay failed tests and were restored: the
  backend replay returning the reference gradients failed 10, the backend replay ignoring the
  reference upstream gradients failed 23, disabling the emulated floor failed 8 (false alarms on
  clean bfloat16 and float16 models).

Not verified or not built:

* FP8 (no FP8 rounding mode, no FP8 kernels here), FlashAttention 2 or 3 kernels (not in this
  Windows build), ROCm, MPS.
* Training mode (batch norm batch statistics, dropout), optimizers other than SGD, gradient scaling.
* A wrong constant inside a custom `autograd.Function` backward is blamed at the module with its
  source, not at an op, because op isolation replaces wrapped Python scalars too.
* Cost budget: resnet50 (64x64) takes about 14 s in float32 and 26 s in bfloat16 on CUDA.
### 2026-11-24: P17 native functions as PyTorch custom ops (`truestar.torch_op`)

Built `src/truestar/customop/` (see docs/CUSTOM_OPS.md): `truestar.torch_op(fn)` registers an
`@truestar.function` through `torch.library.custom_op` with a schema from the typed signature, a
fake implementation derived from the IR shape rules (symbols bound to SymInts, checks through
`torch._check`), and an autograd formula from `truestar.ir.grad` registered as a second custom op
so AOTAutograd can trace it. Execution goes through `customop.execute`: registered executors
first (the seam for the compiled executor), then the kernel layer through a new tensor IR to kernel
graph lowering, then the float64 reference interpreter with a `SlowPathWarning` naming the reason.
29 new tests (7 need CUDA), RTX 5070 Ti, torch 2.11, Windows; full suite 476 passed, 1 skipped.

* **Under 2 s cold:** registration takes 0.28 s to 1.21 s per op in a fresh process with no `cl`,
  `nvcc`, `gcc` or `clang` on PATH (asserted under 2 s). The first call is not under 2 s: 0.6 s to
  14 s with cold caches, mostly numba compile on CPU and float64 verification on CUDA (0.4 s to
  3.5 s on CUDA with verification off).
* **opcheck passes** (schema, autograd registration, fake tensor, AOT dispatch dynamic) for a fused
  tanh GELU MLP, attention with a tanh softcap score modification and RMSNorm over a residual sum,
  on CPU and CUDA, plus a reference path op and a two output op.
* **Zero graph breaks:** `torch._dynamo.explain` gives 0 breaks and 1 graph per op; one graph over
  batch sizes 3, 5 and 7 with `fullgraph=True, dynamic=True`; `aot_eager` matches eager values and
  gradients; Inductor works on CUDA with backward. Inductor on CPU fails here for lack of `cl`,
  independent of Truestar.
* **Gradients match torch** float64 within 1e-4 (values within 1e-5) for all three ops at three
  batch sizes on CPU and CUDA, forward and backward both on the kernel path.
* **Warm speed is mixed:** CUDA attention 1.65 ms vs eager 4.62 ms; MLP 5.11 vs 2.44 ms; RMSNorm
  18.0 vs 1.79 ms. Reported, not hidden.
* **No silent wrong results (#3):** every fast path result is checked against the float64
  reference on its first call per signature, which the kernel layer's own check cannot do for a
  lowering bug. A plugged executor returning a 1% wrong backward is refused.
* **No silent fallback (#1):** the reference path warns with the reason; ops without autodiff rules
  register and refuse backward by name.
* **Found while building it:** an f64 tensor passed to an f32 signature ran silently on the numba
  path; the real implementation now runs the same dtype and shape checks as the fake one.
* **Baseline:** `torch.utils.cpp_extension` was not built; this machine has no MSVC and no CUDA
  toolkit, so it could not be. Documented numbers only (90 s `load_inline` toy kernels per
  pytorch#152032, hours for tens of ops per the PyTorch forums).
* **Tests can fail:** permanent test with a wrong fake shape fails opcheck. Manual breaks, restored:
  reversed fake output shape failed 12 tests; `sub` lowered as `b - a` failed 5 tests through
  `ProgramMismatchError`.

Not verified:

* Same source on a second torch version, and Linux (only torch 2.11 on Windows here).
* `torch.export` with the op inside, MPS, ROCm, double backward, control flow gradients.
* Ops outside the lowering table (exact gelu, select, slice, merging reshapes) run on the slow path.

### 2026-11-24: verified remedies (`truestar fix`, P1 and P2 move from detect to solve)

Built `src/truestar/fix/` (docs/FIX.md). For each backend where `truestar check` names an ATen op,
candidate decompositions from a built in table are tried in order: first the op alone with its
saved inputs, then a full rerun of `truestar check` on that backend with the remedy applied. A
candidate wins only if it fired, the blamed module is now OK, and the same op is not still blamed
elsewhere. The forward pass is timed with and without the remedy. Remedies apply through a
TorchDispatchMode scoped to the device (`truestar.fix.patched`, or `truestar.fix.apply` on one
model) or as a process wide `torch.library` kernel on the device's dispatch key. Text, full JSON and
agent JSON (`command: fix`), new codes TS090 to TS095.

Verified on this machine (RTX 5070 Ti shared with other jobs, CUDA 12.8, torch 2.11), 22 new tests:

* **Resolved, with numbers:** OpFault on `aten.addmm.default` (examples/fix_demo.py) output error
  1.5e-1 and 3 of 32 flips to 1.6e-7 and 0 flips with `mm_add`; OpFault on
  `aten.convolution.default` in ResNet18 5.2e-1 to 6.7e-7 with `im2col`; two faulted ops in one
  block (addmm and softmax) take two rounds and end clean.
* **Cost:** on a clean ResNet18 the addmm remedy adds 19% as a mode and 4% as a library kernel; the
  convolution remedy adds 67% and 51% (im2col is slower than cuDNN). The GPU was shared, so single
  digit percentages are noise.
* **Honest failures:** an MLP with addmm, mm, bmm and mul all faulted is unresolved (TS092): all 3
  candidates fail the op filter, nothing is applied and the op repro path is given for upstream. A
  module level device branch is not applicable (TS093). The library mechanism cannot clear a
  dispatch mode fault and says so.
* **MPS prepared, not run:** the macOS 26 `aten.linear.default` finding has a built in remedy
  (`matmul_add` first, matching the probe). Its code path is tested on CUDA by registering a fused
  CUDA linear kernel, so linear is blamed as an op the reference never runs, as on MPS: faulted by
  OpFault it is resolved with the mode; made wrong inside the kernel it is resolved with the
  library mechanism, and removing the remedy brings the broken kernel back.
* **INT8 precision plan (offline):** synthetic ResNet18 of docs/NPU.md, CPU w8a8 fake quant vs
  float64: 15 flips of 256 with every layer quantized (the S24 CPU had 16). Target 4: 6 of 21 layers
  in float, 26.3% of MACs, 4 flips, 52 s. Target 0: 18 of 21 layers, 80.9% of MACs.
* **No silent wrong results of our own (never introduce #3):** every table entry is checked against
  the real op in float64 on CPU at 1e-10.
* **Tests can fail:** making the op filter pretend success fails 2 tests (the genuinely broken
  model is no longer reported unresolved; a wrong candidate wins); making the rerun model check
  pretend success fails 1 test (a remedy right only for the blamed call is caught only there).

Not verified:

* MPS, ROCm and XPU: no hardware here.
* AI Hub execution of a precision plan: `PrecisionPlan.qai_hub_hint()` is the hook; whether the
  quantize job accepts per layer exclusions was not checked.
* The INT8 simulation is the intended arithmetic; the NPU's own rounding differences (docs/NPU.md)
  are not modelled.
* Only the op named by the check is remedied, up to 4 rounds per backend.
### 2026-11-24: kernel performance (autotuning, attention schedule)

Added launch config autotuning (`src/truestar/kernel/tune.py`, interface in docs/KERNEL_IR.md
section 6a) and a new attention schedule. Public kernel API unchanged; `compile` and
`Graph.compile` gained an optional `tune` argument.

Schedule rules, causal attention B2 H16 S2048 D64 fp16, each measured alone (fixed configs, no
tuning, medians of `do_bench`):

| change | with the old 128x64, 8 warp, 2 stage config | removed from the final 64x64, 4 warp, 3 stage config | kept |
|---|---|---|---|
| longest first block order (LPT) | 0.297 to 0.260 ms (12% faster) | 0.196 to 0.225 ms (15% slower) | yes |
| interior key blocks without masks (SPLIT) | 0.297 to 0.284 ms (4%) | 0.196 to 0.202 ms (3%) | yes |
| exp2 with log2(e) folded into the scale | 0.297 to 0.282 ms (5%) | 0.196 to 0.202 ms (3%) | yes |
| plain loads straight into `tl.dot`, no constant pad masks | within noise | 0.196 to 0.203 ms (3%) | yes |
| accumulate into `tl.dot(p, v, acc)` | 0.2967 vs 0.2966 ms (no effect) | 0.1965 vs 0.1963 ms | dropped |
| block config 64x64, 4 warps, 3 stages (all rules on) | 0.245 to 0.196 ms (20%) | | default for head dim 64 |

Skipping fully masked key blocks was already present; verified again (the plan tests assert the
bounds, and the interior and edge loops are now tested separately). The accumulator, running max
and sum were already float32 and stay so. A flat 1D elementwise kernel and streaming cache hints
were tried for case (a) and gave nothing measurable (every variant 0.516 to 0.520 ms, at DRAM
bandwidth), so they were not added; a larger default block (4096 elements, 8 warps) reaches the
same speed.

Benchmark, `scripts/bench_kernels.py` (RTX 5070 Ti, torch 2.11, triton 3.7.1, cold caches for
every tool). Run time is the median of 5 interleaved rounds; spread is (max minus min) over
median of those rounds. Truestar compile is code generation plus Triton compile plus first launch,
tuned compile is the extra tuning time on top of that. The GPU is shared with another agent, so
some noise from it is possible; the spread column bounds it for this run.

| case | Truestar tuned | Truestar tuning off | FlexAttention (torch.compile) | SDPA flash | torch.compile math | speedup vs best other |
|---|---|---|---|---|---|---|
| (a) GELU + residual 8192x4096 fp32 | 0.508 ms (0.0%) | 0.508 ms, compile 0.38 s | | | 0.508 ms (0.1%), compile 1.46 s | 1.00x (tie) |
| (b) causal B2 H16 S2048 D64 | 0.196 ms (0.1%) | 0.197 ms, compile 0.75 s | 0.251 ms (0.7%), compile 1.26 s | 0.451 ms | 1.512 ms | 1.28x |
| (b) causal B2 H16 S2048 D128 | 0.415 ms (0.0%) | 0.415 ms, compile 1.01 s | 0.489 ms (0.0%) | 1.008 ms | | 1.18x |
| (b) causal B1 H8 S4096 D64 | 0.194 ms (1.0%) | 0.194 ms, compile 0.92 s | 0.294 ms (1.0%) | 0.548 ms | | 1.52x |
| (b) causal B1 H8 S4096 D128 | 0.394 ms (0.3%) | 0.395 ms, compile 0.84 s | 0.524 ms (0.0%) | 1.036 ms | | 1.33x |
| (b) causal B1 H2 S8192 D64 | 0.191 ms (1.0%) | 0.190 ms, compile 0.85 s | 0.409 ms (0.0%) | 0.653 ms | | 2.14x |
| (b) causal B1 H2 S8192 D128 | 0.399 ms (0.0%) | 0.407 ms, compile 1.16 s | 0.630 ms (0.0%) | 1.129 ms | | 1.58x |
| (c) differential B2 H16 S2048 D32x2 | 0.286 ms (0.3%) | 0.286 ms, compile 1.36 s | 2x Flex + combine 0.368 ms (0.6%) | 2x SDPA 0.733 ms | 2.137 ms | 1.29x |

* **Wins:** causal attention beats FlexAttention at every size, 1.18x to 2.14x (was 16% slower at
  S2048 D64: 0.311 vs 0.260 ms before this change). Differential attention beats two
  FlexAttention calls by 1.29x (was 1.07x). The largest gains are at long S with few heads, where
  LPT keeps the SMs busy as the work per query block grows.
* **Loss / tie:** the elementwise case ties torch.compile (0.508 vs 0.508 ms, was 0.561 vs
  0.532). Both run at about 790 GB/s, the DRAM limit of this card for this access pattern, so it
  cannot be beaten by scheduling; the goal of beating torch.compile here is not met.
* **Tuning gains are small** because the measured defaults are already near the best candidate:
  tuned vs default at most 2% (S8192 D128 0.399 vs 0.407 ms), and the winner equals the default
  in 3 of 8 cases. Tuning is still the guard for other GPUs and shapes the defaults were not
  measured on.
* **Tuning cost:** 1.05 s (a), 1.8 to 2.1 s per attention bucket, 2.5 s (c), once per kernel
  structure, shape bucket, dtype and device. Candidates compile in a thread pool (8 serial
  compiles 6.7 s, in parallel 1.7 s). Bounded by `TRUESTAR_TUNE_BUDGET` (30 s). A new process
  pays nothing: the warm start of the attention block took 0.15 s with 0 searches and 1 config
  read from disk (torch.compile warm start 3.05 s). A constant edit reuses the tuned config: edit
  recompile 0.17 s (a), 0.63 s (b), 1.07 s (c), versus torch.compile 0.51 s, 0.50 s, 1.28 s.
* **Compile time regression (never introduce #5):** a cold attention compile without tuning is
  0.75 to 1.36 s, up from 0.56 s (b) and 1.07 s (c), because SPLIT emits two or three copies of
  the loop body. Still below FlexAttention's 1.26 to 2.28 s at the same sizes; with tuning the
  first call costs 1.8 to 2.5 s, comparable to FlexAttention.
* **Never introduce #3:** max relative error vs float64 is unchanged in kind: 2.7e-4 to 3.3e-4
  for attention (identical to SDPA and FlexAttention), 3.0e-4 for (c) (PyTorch paths 4.6e-4 to
  6.3e-4), 7.6e-8 for (a). Every config is verified on its first call: the verified set is keyed
  on the launch config, so a tuned config is checked even after the default was. Verification
  time, excluded from compile and run times: 1.4 s (a), 3.6 s (b), 7.3 s (c), 15.6 to 38.1 s at
  S4096 and 20.8 to 26.0 s at S8192 (the CPU float64 reference is quadratic in S, which is why
  batch times heads shrinks as S grows in this benchmark).
* **Robustness:** across 5 interleaved rounds the spread of every Truestar row is at most 1.0%,
  every FlexAttention row at most 1.0%; the noisiest row is eager math (11.5%).
* **Tests:** tests/test_kernel_tune.py, 29 cases: a second process reads the tuned config and
  runs no search; same bucket shares a search; a constant edit reuses the config but recompiles
  and verifies the binary; six kinds of corrupted cache file (garbage, truncated, edited config
  failing the digest, self consistent file with an invalid config, wrong key, old version) each
  warn, retune and heal the file; auto, off and zero budget modes; and every candidate config of
  every loop shape is run and compared with float64 (attention fp16, bf16, fp32, causal, window,
  unmasked, head dims 32 to 128, differential; tile, reduction, contraction). Full suite: 368
  passed, 1 skipped (339 before the 29 new cases). Tests can fail: rounding the
  window interior start down instead of up fails 2 tests; an off by one in the LPT block order
  fails 32 tests. Both were restored.

Not verified:

* Tuned configs and defaults only on this GPU (sm_120); other GPUs fall back to the same
  candidate lists, untested.
* Non causal and windowed attention, bf16 and fp32 attention were tested for correctness under
  every config but not benchmarked.
* Contraction and reduction candidates are tested for correctness, not benchmarked against cuBLAS
  or torch.compile.
### 2026-11-24: Runtime host overhead and CUDA graph capture (`truestar.runtime`)

Built `src/truestar/runtime/` (see docs/RUNTIME.md): `capture(fn, example_args)` records any
executor's launches into a `torch.cuda.CUDAGraph` with static input and output buffers, one graph
per shape bucket, inputs bound by address with `static=`, and `scripts/bench_overhead.py`. Serves
never introduce #4 (overhead measured against PyTorch eager and `torch.compile`). 27 new tests,
all need CUDA; RTX 5070 Ti, torch 2.11, Triton 3.7.1, Windows. Timings are medians of three
processes with two other agents using the GPU.

* **Where overhead dominates:** a kernel layer launch costs 14 to 20 us of host time (about 7 us
  driver floor, 5 to 7 us Triton binder, 4 to 6 us Truestar wrapper; cache lookup 0.1 to 0.4 us).
  Small kernels run 2 to 6 us on the GPU (host 3x to 10x GPU); medium kernels 55 to 260 us (host
  hidden).
* **Gain:** a 50 kernel MLP step at 32x256 goes from 1319 us to 418 us (3.2x), equal to its GPU
  time; host time after capture 13 to 26 us. PyTorch eager captured by the same layer: 1301 to
  246 us. `torch.compile` `reduce-overhead` wins this step at 148 us because its kernels take 239 us
  of GPU time against Truestar's 413 us (kernel quality, not host). Medium step (1024x1024): all
  GPU bound, no gain. Single small kernels: `torch.compile` `reduce-overhead` is slowest (42 to
  60 us); one kernel graph helps only with static buffers (8 to 10 us vs 17 to 20 us eager).
* **Rust or C++ verdict:** not now. Remaining host time is 3% to 6% of GPU time and overlapped;
  replay alone (6 to 7 us) is a floor in any language.
* **No silent wrong results (#3):** every capture compares its replay bitwise with the eager run;
  padded buckets compare the first call at each size with an unpadded eager run. Found: Truestar's
  tile kernels pick their row block from the row count below 16 rows, so padding 1 or 2 rows into
  a bucket of 16 changes the layer norm's reduction order; the runtime refuses those sizes.
* **No silent fallback (#1):** host syncs, input writes, CPU inputs, shapes outside buckets,
  dtype changes and swapped bound inputs all raise named errors.
* **Tests can fail:** manual breaks, restored: skipping the static input copy failed 15 of 27 tests
  (replay correctness, chain, bucket, bound input and stale output tests); returning static buffers in copy
  mode failed the stale output test.

Not verified:

* Linux, ROCm, multi stream use, backward capture, the compiled executor (not landed in this
  worktree; the interface is a plain callable).
### 2026-11-24: RL trainer and sampler pair check (`truestar rlpair`, P15 part a)

Built `truestar.rlpair` and `truestar rlpair pair.py` (docs/RL_PAIR.md). The trainer's teacher
forced forward and the sampler's incremental decoding (KV cache, optional batching with left
padding, optional other dtype) are both recorded per module call; each sampler call is tagged with
its decode step and mapped to the trainer positions dim by dim (chunk, cache prefix, cache before
the step, batch row with its pad offset). Every module is replayed on the full sequence with the
sampler's inputs written in at the mapped positions: the trainer's module, the sampler's module in
its own dtype and, across dtypes, the trainer's module in the sampler dtype as the precision band.
That splits each module's own error into decode path, implementation and precision. The first
divergent module with clean submodules is blamed with the first differing position, then the op
inside it (dispatch tracing inside that one sampler call, ops rerun alone with mapped full path
inputs). Codes TS190 to TS199, agent JSON with `"command": "rlpair"` and `next_actions`
(`inspect_source`, `fix_position_offset`, `inspect_op`, `check_position_ids`, `match_temperature`,
`match_dtype`, `sync_weights`, `check_batch_invariance`, `check_mapping`). 25 new tests on a two
block GPT style decoder with a KV cache path written in plain `torch.nn` (tests/rl_models.py,
random weights); CPU plus one short CUDA test on the RTX 5070 Ti (torch 2.11, CUDA 12.8).

* **Correct sampler reports a match to rounding:** float32 max absolute logprob difference 4.8e-07
  on CPU and 2.4e-07 on CUDA, importance ratio 1 to within 1e-6, nothing blamed. With localization
  forced, all 29 module calls map to sampler positions over all 10 steps and none differs; the same
  with the rows left padded by 3 inside a batch of 4.
* **(a) Temperature:** a sampler whose `Head` divides by 0.8 is blamed at `head#0` from position 0,
  op `aten.div.Tensor` with constant 0.8 that only the sampling path runs (max diff 0.347, 3 of 20
  ratios outside the 0.2 clip). Applied outside every module, it is TS193 with the temperature
  fitted as 0.7000 (residual 6e-07).
* **(b) KV cache rotary offset off by one:** blamed at `blocks.0.attn#0` from position 6, decode
  step 1 (the prefill is right), op `aten.slice.Tensor` on the rotary table starting at 7 for the
  step at position 6, reported as offset +1. Max diff only 0.026 and every ratio inside the clip,
  so a ratio monitor would not flag it. Same blame with left padding and with a bfloat16 sampler.
  The Hugging Face style variant (position ids built from the cache offset) is TS197 at the first
  block (sampler 7, trainer 6) and blamed at the model's own code, op `aten.arange.start`, offset +1.
* **(c) bfloat16 sampler, float32 trainer:** classified precision (TS191), not divergent: max diff
  9.8e-03, mean 3.1e-03, 17 of 20 tokens over 1e-3, ratio 0.9946 to 1.0098, k3 KL 9e-06; every
  module within 3x its emulated bfloat16 band (largest own error 6.3e-03 against a 6.0e-03 band).
  Same class on CUDA (max diff 1.05e-02).
* **Batch composition, reported as measured:** the correct float32 sampler's logprobs for a row are
  not bitwise identical inside other batches: on CPU at batch 2 (4.8e-07), on CUDA at batches 2
  and 7 but not 64 (4.8e-07). The bfloat16 sampler differs on CPU at batch 7 (4.0e-03) and was
  bitwise identical at 2, 7 and 64 on CUDA for this small model. TS195 warns; the pair status stays
  a match, `--strict` makes both exit 1.
* **Stale weights:** a sampler copy with one weight scaled by 1.01 is TS194 naming the parameter and
  blamed at `blocks.1.mlp.up#0` as an implementation difference; a bfloat16 copy is not stale.
* **Baseline:** logprob statistics alone (the vllm#21475 style comparison) show each injected bug
  only as a gap of 2e-3 to 0.6 and name no module, position or op.
* **Tests can fail:** a permanent test shifts the chunk mapping by one position and asserts the
  correct pair is no longer clean. Manual breaks, each restored: the chunk mapped one position late
  failed 11 tests; left padding ignored in the mapping failed 3 tests.

Not verified or not built:

* P15 part b: one kernel set for prefill, decode and the training forward, bitwise equal
  logprobs, tensor parallel 1 and 2. Untouched; the kernel layer was not edited.
* Real engines (vLLM, SGLang) and Hugging Face models; only the test decoder was checked.
  Engines in another process give statistics only, and different module names need `truestar port`.
* Speculative decoding, chunked prefill, flattened batch times sequence tensors at op level, MPS,
  ROCm.
### 2026-11-24: launch docs, examples and MCP tools for every command

* README rewritten for launch; docs/index.md lists every doc; docs/GETTING_STARTED.md walks
  through one example per command. Every command on that page was run on 2026-11-24 and gave the
  exit code the page states; tests/test_examples.py runs them all (13 tests, about 53 s, two need
  CUDA). Each example command finishes in under 10 s on CPU.
* MCP tools added: `grad_check`, `port_check`, `invariance_check`, `record`, `diff`,
  `export_model`, with round trip tests over real stdio on the builtin and SDK transports.
* **No new problems:** `diff` reads only files with the recording magic (a replay imports the
  model a recording names); `record` and `export_model` refuse to overwrite a file truestar did
  not write. Removing those checks failed 4 tests (restored). The server still never imports torch.
* Full suite: 727 passed, 1 skipped.

### 2026-11-24: compiled executor (tensor IR lowered to verified kernels, `truestar.compiler`)

Built `src/truestar/compiler/` (see docs/COMPILER.md): the tensor IR is lowered to the kernel layer,
fused, and run through Triton (GPU) or numba (CPU), behind both execution seams
(`@truestar.function(executor=CompiledExecutor("cuda"))`, `export.load(path, executor="cuda")`),
in float32, float16 or bfloat16. 98 new test cases; full suite 433 passed, 3 skipped on CPU plus
RTX 5070 Ti (torch 2.11, triton 3.7.1).

Coverage:

* **Truestar kernels:** every elementwise op, comparison and logic op of the IR, `where`, `sum`,
  `mean`, `any`, `softmax`, `matmul`, `linear`, `batch_norm`, `layer_norm`, `attention`. The MLP,
  TransformerEncoderLayer (relu post norm and gelu pre norm), the native RNN, a Newton `while` loop
  and five attention variants compile with zero fallbacks. Attention is one fused kernel on the GPU,
  from the IR op and from plain ops written in the native front end.
* **Library kernels (torch, declared in the report, checked like kernels):** `conv2d`, `max_pool2d`,
  `adaptive_avg_pool2d`. resnet18 is 18 Truestar kernels (20 batch norms, 17 relus, 8 adds and the
  final linear fused into them) plus 22 library kernels.
* **Fallback (float64 reference implementation of the one op, listed with the reason):** layer norm
  over more than the last dim on the GPU, matmul or linear with a one dim operand, integer tensor
  arithmetic, any op without a rule. Layout ops are torch views, integer scalars stay on the host.

P5, with real code generation behind the front end for the first time:

* Programs, regions and kernels compiled after the first call across batch 1, 7 and 64: 0 in all
  four benchmark cases and in the tests (which also assert the kernel cache is not even asked).
  torch.compile compiled 1 more graph for the MLP, 1 for resnet18 and 3 for the native function.
* Loop lengths 8 and 48 through the RNN compiled at 32: Truestar 1.93 ms and 3.40 ms with 0 compiles; torch.compile 0.80 s and 2.34 s with 2 new graphs, and 1 graph break on the branch.
* Found and fixed below Truestar: Triton compiled a new variant of the same kernel whenever a
  symbolic size was 1 or changed divisibility by 16, and whenever the tile block row count followed
  the batch. That cost 1.0 to 1.7 s at each of batch 1, 7 and 64 on the MLP, with Truestar's own
  compile count at 0. Symbolic sizes are now excluded from Triton's specialization and the block
  row count is fixed for a symbolic row count: about 1 ms. Excluding strides too was measured and
  rejected (matmul kernels 1.6x slower).
* Still open: a symbolic axis that is reduced, or mapped to lanes, compiles one Triton variant per
  power of two bucket (seen once: 147 ms at batch 64 in the native case).

P6 (cold caches, fresh process per measurement, fastest of three runs on a shared GPU):

| case | tool | first result at batch 1 (s) | first call at batch 7, then 64 (ms) | compiles for the new batch sizes | compile again after an edit (s) |
|---|---|---|---|---|---|
| MLP | Truestar | 2.97 (import 1.10, Truestar compile 0.09, Triton and first run 1.73) plus checks 0.04 | 12.7 (of which checks 11.5), 30.2 (of which checks 28.7) | 0 | 0.46, kernels generated: 1 |
| MLP | torch.compile | 2.81 | 806.0, 1.23 | 1 graph | 0.42 |
| TransformerEncoderLayer | Truestar | 4.82 (import 1.56, Truestar compile 0.12, Triton and first run 3.14) plus checks 0.05 | 125.1 (of which checks 121.3), 1375.9 (of which checks 1371.3) | 0 | 0.84, kernels generated: 2 |
| TransformerEncoderLayer | torch.compile | 1.90 | 69.1, 1.77 | 0 graphs | 0.17 |
| resnet18 | Truestar | 4.36 (import 0.90, Truestar compile 0.12, Triton and first run 3.33) plus checks 0.23 | 311.4 (of which checks 281.0), 2730.2 (of which checks 2705.2) | 0 | 0.33, kernels generated: 0 |
| resnet18 | torch.compile | 7.89 | 5849.9, 23.2 | 1 graph | 2.47 |
| native RNN | Truestar | 1.68 (Truestar compile 0.31, Triton and first run 1.37) plus checks 0.02 | 12.3 (of which checks 8.87), 147.1 (of which checks 3.52) | 0 | 0.17, kernels generated: 1 |
| native RNN | torch.compile | 3.43 | 1921.6, 267.4 | 3 graphs | 0.28 |

* Truestar's own compile (lowering, fusion, code generation) is 0.09 to 0.31 s per program; the
  rest of the first result is torch.export and Triton.
* Loss: the transformer's first result is 2.5x slower than torch.compile and its recompile after an
  edit 5x slower, because each new kernel is one Triton compile of about 0.3 s.

Run time, never introduce #4 (measured against eager and torch.compile, float32, verification on):

| case | tool | run, batch 64 (ms) | run, batch 1 (ms) | vs eager at 64 |
|---|---|---|---|---|
| MLP, 1024 to 2048 x3 to 1000 | Truestar | 0.51 | 0.50 | 2.16x slower |
| MLP, 1024 to 2048 x3 to 1000 | PyTorch eager | 0.23 | 0.09 | baseline |
| MLP, 1024 to 2048 x3 to 1000 | torch.compile | 0.16 | 0.12 | 1.46x faster |
| TransformerEncoderLayer, d 256, 4 heads, seq 128 | Truestar | 1.61 | 0.48 | 2.41x slower |
| TransformerEncoderLayer, d 256, 4 heads, seq 128 | PyTorch eager | 0.67 | 0.15 | baseline |
| TransformerEncoderLayer, d 256, 4 heads, seq 128 | torch.compile | 0.67 | 0.18 | same |
| resnet18, 224 px | Truestar | 11.2 | 1.65 | 1.05x faster |
| resnet18, 224 px | Truestar, TF32 allowed | 10.4 | 0.98 | 1.14x faster |
| resnet18, 224 px | PyTorch eager | 11.8 | 1.19 | baseline |
| resnet18, 224 px | torch.compile | 10.1 | 0.93 | 1.17x faster |
| native RNN, hidden 512, 32 steps, branch on data | Truestar | 1.41 | 1.62 | 1.12x faster |
| native RNN, hidden 512, 32 steps, branch on data | PyTorch eager | 1.58 | 1.22 | baseline |
| native RNN, hidden 512, 32 steps, branch on data | torch.compile | 1.23 | 0.89 | 1.29x faster |

* Losses: 2.2x behind eager on the MLP and 2.4x on the transformer layer (the float32 contraction
  kernel against cuBLAS). resnet18 and the native RNN are within 5% to 15% of eager and
  torch.compile on either side.

Never introduce #3:

* Three checks, on by default: the kernel layer's check of each kernel; a region check of every
  kernel group and library step against the float64 reference implementation of the IR ops it
  covers, on the inputs it really received; an end to end check against the reference interpreter
  on the first inputs. A mismatch raises `ProgramMismatchError` naming the region and the op, found
  by compiling each op of the region alone. Branches are checked the first time they run.
* Largest end to end error against float64 in the benchmark: MLP 1.1e-6, transformer 2.3e-7,
  resnet18 8.3e-7 (8.0e-4 with TF32 allowed), native RNN 8.1e-7.
* Cost: 0.02 to 0.23 s on the first call at batch 1; at batch 64 the kernel layer's repeated check
  costs 1.4 s for the transformer and 2.7 s for resnet18.
* While building it, no wrong result was found in the kernel layer. Two rank 0 gaps were found as
  crashes (a Triton tile kernel storing a reduced scalar, a numba load of a rank 0 buffer) and are
  worked around by computing scalars as one element tensors; a `while` loop that cannot converge in
  float16 used to spin forever and now raises with the reason.

Never introduce #1: the compile report lists every fallback with its reason and every library
kernel; `allow_fallback=False` refuses to compile; a missing CUDA device raises.

Tests can fail:

* A relu rule that leaks 1% of negative values (`max(x, 0.01 x)`) failed all 18 selected tests whose
  program contains a relu (MLP, transformer, resnet18, elementwise group, native loop and branch),
  each with `ProgramMismatchError` naming `relu`; the 8 selected tests without a relu passed. With verification turned off, the same break failed all 3
  tests run, through their own comparison with float64. The rule was restored.
* Permanent tests break a lowering rule, a rule inside a fused region, a rule inside a loop body, a
  rule in a branch not taken at first, the kernel code generator, a library kernel and a layout
  step, and assert each is caught with the right region and op; a control test shows the wrong
  result ships when verification is off.

Not verified or not built:

* Convolution and pooling as Truestar kernels (the kernel IR has no strided index). Done 2026-11-24, see the native convolution entry.
* Any GPU other than the RTX 5070 Ti; ROCm, MPS. The CPU backend is correct and slow and was not
  benchmarked.
* Compile time and run time are not tracked in CI yet.
* Models larger than resnet18 and a TransformerEncoderLayer; a whole transformer stack.
* Comparisons within rounding of a tie can take the other branch than float64 and are then reported
  as a mismatch.
* Benchmarks ran on a GPU shared with other jobs. Times are the fastest of three runs; single runs
  varied by up to 3x when another job was active.

Scorecard after this entry (rows this work changed are marked):

| Row | State | Changed here |
|---|---|---|
| P1 silent wrong results off NVIDIA | Injected faults blamed exactly on CUDA; one real MPS divergence found on a hosted runner; the listed upstream bugs not yet rediscovered on physical hardware | no |
| P2 flips hidden by unchanged accuracy | Per input flip counts exist; not yet shown on a quantized model | no |
| P3 first diverging op | Module replay plus op drilldown name the op on injected faults | no |
| P4 one tool for every backend | CUDA, CPU, hosted MPS, one hosted phone NPU; ROCm untested | no |
| P5 graph breaks and recompiles | Holds with compiled execution: 0 compiles across batch 1, 7, 64 and across loop lengths, against 1 to 3 graphs and a graph break for torch.compile. Open: Triton block size buckets for a symbolic reduced axis | yes |
| P6 compile time | Recompile after an edit 0.17 to 0.84 s, first result 1.7 to 4.8 s, only changed kernels generated. Slower than torch.compile on the transformer. Not tracked in CI | yes |
| P7 export | The verified artifact now also runs compiled on GPU and CPU through the runner. Model coverage unchanged (46 of 63 classifiers) | yes |
| P8 new layers without hand written kernels | Differential attention written with plain ops in the native front end fuses into one attention kernel through the tensor IR. Speed against a hand written kernel was measured at the kernel layer only | yes |
| P9 shape errors before running | Front end and importer report at compile time | no |
| P10 hang and OOM attribution | CUDA OOM and gloo hangs attributed; NCCL untested | no |
| P11 install | Linux and Windows proven in CI; macOS not run | no |

### 2026-11-24: compiled executor merged with kernel autotuning

Merged main (autotuning, LPT and SPLIT attention schedule) into the compiled executor branch. Both
behaviours kept: `_jit_line` and `erf` in the Triton backend next to `default_config`, `candidates`
and the new attention code; for a symbolic row count the block row count is fixed in the defaults
and in the tuning candidates; the autotuner's key leaves out tile row symbols and the batch symbols
of contraction and attention kernels, so one tuned config serves every batch size. Full suite
848 passed, 3 skipped (CPU plus RTX 5070 Ti).

* `scripts/bench_kernels.py --quick`: tuned causal attention B1 H8 S1024 D64 fp16 0.026 ms against
  FlexAttention 0.038 ms (1.47x), as on main; differential attention 0.034 ms against 0.051 ms;
  S2048 D128 0.110 ms against 0.180 ms.
* `scripts/bench_compiler.py`, one run: Truestar compiles across batch 1, 7 and 64 still 0 in all
  four cases; loop lengths 8 and 48 still 0 compiles (2.7 ms and 4.5 ms). Run time at batch 64 with
  tuning: MLP 0.41 ms (was 0.51), transformer 1.00 ms (was 1.61), resnet18 11.3 ms, native RNN 2.45 ms.
* Loss: the tuning search is paid once per kernel at its first launch that is big enough to tune:
  2.2 s (MLP), 10.2 s (transformer), 8.1 s (resnet18) on the first call at batch 64, and 0.9 s for
  resnet18 at batch 7. A config tuned at one batch size is used at every batch size, so it is not
  the per bucket optimum for the others.
* Found by the merged suite: with a symbolic row count and one lane, a tuning candidate asked for
  16384 block rows, above the 8192 limit; the fixed row count is now capped.
* New test: default configs, candidates and tuning keys are equal across batch 1, 7, 64 and 4096
  for symbolic rows, while static rows and symbolic lanes keep their own configs.
### 2026-11-24: decoder LLMs in the IR, the importer and the native front end

Built per docs/LLM.md (rows P5, P7, P15). Nine new IR ops: `flip`, `sort`, `topk`, `argmax`,
`argmin`, `amax`, `amin`, `index_copy`, `multinomial` (86 ops, 238 ATen overloads). Grouped and
multi query attention, `repeat_interleave`, sliding windows, temperature and cumsum top p filtering
lower onto existing ops. torch 2.11, CPU, float64.

* **KV cache decoding, native:** one `@truestar.function` serves the prompt and every decode step
  (symbolic batch, token count and cache capacity, positions as data written with `index_copy`).
  Greedy decoding for 16 tokens matches a plain `torch.nn` Llama style model (RMSNorm, rotary, GQA,
  SwiGLU, tied head; random weights) that recomputes the whole sequence: tokens equal, logits within
  1.9e-15 relative; zero compiles during the 17 calls, at two batch sizes and two cache sizes.
* **Import:** prefill and decode step graphs of the same model through `torch.export`, export graph
  and core ATen graph, within 2.7e-15 of PyTorch float64 over three shapes each; greedy decoding with
  the imported programs gives the same 16 tokens.
* **Ops:** 19 import cases (both graph forms, symbolic batch, a second shape) and 8 native functions
  match PyTorch float64 under 1e-10, integer results exactly. Gradients for `flip`, `index_copy`,
  `amax` and `amin` pass finite differences; a GQA layer's gradient matches `torch.autograd`.
  `sort` and `topk` have no gradient yet and fail by name.
* **Max and min reductions for torch 2.14 (from CI):** `aten.amax`, `aten.amin`, `aten.max.default`,
  `aten.min.default`, `aten.max.dim`, `aten.min.dim`, `aten.max.other` and `aten.min.other` map to IR
  ops. Not run against torch 2.14 here.
* **No silent wrong results (#3):** found that `torch.export` keeps only one of two example inputs
  sharing storage, so the imported decode program ignored three of four caches; `import_module` now
  refuses such inputs. Repeated `index_copy` positions and out of range positions are errors.
* **No silent fallback (#1):** boolean mask reads, PyTorch random ops, scatter with several index
  tensors, uneven symbolic chunks and `topk` along a symbolic dim are refused with the reason.
* **Tests can fail:** six permanent broken reference cases, a stale cache test and a mask ignoring
  attention test. Manual breaks, restored: inverted sort tie rule failed 1 test; `index_copy` writing
  its source flipped failed 5.
* Full suite after the change: see the commit message.

Not verified: torch 2.14, a pretrained checkpoint (no downloads), compiled kernels, paged caches,
`sort` and `topk` at real vocabulary sizes (the reference compares every pair).

### 2026-11-24: kernel verification on the GPU

The float64 reference interpreter of the kernel IR now also runs on the kernel's own CUDA device
(`reference.run_reference_torch`, `verify.compare_torch`; docs/KERNEL_IR.md section 7). Same
float64 math node by node, same tolerances, same schedule (every new shape signature and launch
config is checked). RTX 5070 Ti shared with another job, torch 2.11, CUDA 12.8, Windows.

* **Kernel benchmark** (`scripts/bench_kernels.py --quick`, verification seconds, before with
  `TRUESTAR_VERIFY_DEVICE=cpu`, after with the default): (a) GELU 2048x2048 0.13 to 0.07 s (this
  includes the GPU's one time float64 start up), (b) causal attention S1024 1.38 to 0.03 s,
  (c) differential attention S1024 1.85 and 1.62 to 0.02 s, (b) S2048 D128 0.76 and 0.94 to 0.13
  and 0.07 s. An S8192 B1 H2 D128 attention check: 1.1 s (the CPU reference took 20.8 to 26.0 s
  per S8192 row). Max error against float64 is unchanged in every row.
* **Compiler benchmark** (`scripts/bench_compiler.py --tools truestar`, verification at the first
  call of each new batch size, before, then two runs after): batch 64 resnet18 2.6 s to 0.15 to
  0.55 s, transformer 1.26 s to 57 to 134 ms, MLP 27 ms to 10 to 11 ms, native RNN 10 ms to 4 to
  40 ms. Batch 7: MLP 13 to 8 ms, transformer 120 to 23 to 39 ms, resnet18 0.29 to 0.21 to 0.44 s
  (no gain, noise of the shared GPU). Cost: the GPU's first float64 use in a process, 0.1 to 0.2 s
  once; the MLP's batch 1 check went from 0.04 to 0.17 to 0.20 s and the native RNN's batch 7
  check from 14 ms to 0.13 to 0.20 s. Checks below 2^20 elements of work keep the NumPy reference
  for that reason. The rest of the first call at batch 64 (1.9 s MLP, 8 s transformer, 5.7 s
  resnet18 on this shared GPU) is not verification and was not looked at here.
* **Same reference (never introduce #3):** with every check forced to the GPU
  (`TRUESTAR_VERIFY_DEVICE=gpu`), a test hook ran the NumPy reference next to every GPU reference
  in tests/test_kernel.py, test_kernel_tune.py, test_compiler.py: 801 outputs of 158 kernels, worst
  relative difference 9.6e-16, no op fell back to the CPU; all those tests passed in that mode,
  including every injected broken kernel test.
* **Reduced schedule rejected:** checking once at a representative size and once at the first
  distinct size is not as strong (a fault in a second row block or a partial last tile appears
  first at a later size). New test: a kernel right for 64 rows and wrong past them, checked at
  batch 7 and 64, must fail at 100 and 4096 with the GPU reference, the size rule and NumPy. Made
  the compiler stop after two checked sizes by hand: that test failed, restored.
* **New tests** (tests/test_kernel_gpu_verify.py, 14): GPU equals NumPy reference on pointwise,
  reduction, contraction and attention kernels; chunked equals whole (blocks of one, blocks of two
  with a partial last block); a simulated missing op falls back to the CPU for that op only;
  `compare_torch` agrees with `compare_numpy` on NaN, infinity, shape and scale cases; CUDA kernels
  never call the NumPy reference in `gpu` mode and small ones do in `auto` mode; broken kernels
  (scaled sigmoid, matmul epilogue, NaN) caught with each device setting.
* Full suite after the change: 993 passed, 3 skipped in 540 s (shared GPU).

Not done: the region and end to end checks of the compiler (docs/COMPILER.md) still evaluate the
tensor IR reference on the CPU; only the kernel layer moved.
### 2026-11-24: contraction kernels closer to cuBLAS

Work in `src/truestar/kernel/triton_backend.py` (contraction code generation, default configs,
candidates) and `tune.py` (search space version in the tune key). RTX 5070 Ti, torch 2.11, Triton
3.7.1, GPU shared with another job throughout, so single numbers move by up to 2x between runs;
every comparison below was interleaved in one process.

Method: linear with bias and relu, `x @ W.T + b`, weights `[N, K]`, default launch config (tuning
off), CUDA graphs of 20 calls, 5 rounds interleaved with `F.linear` plus relu and `torch.matmul`,
median. Every row was checked against float64 (max relative error 2.2e-6 strict float32, 4e-4
float16, 3e-3 bfloat16, 6.6e-4 TF32). `F.linear` plus relu is two kernels; `torch.matmul` is the
bare cuBLAS product without bias.

| operands | M x N x K | before (us) | after (us) | F.linear + relu (us) | torch.matmul (us) | after / F.linear |
|---|---|---|---|---|---|---|
| float32 | 1 x 4096 x 4096 | 289.2 | 117.1 | 80.7 | 81.0 | 1.45 |
| float32 | 16 x 4096 x 1024 | 74.1 | 30.1 | 38.3 | 37.2 | 0.79 |
| float32 | 64 x 2048 x 1024 | 77.0 | 40.5 | 37.3 | 23.6 | 1.09 |
| float32 | 64 x 2048 x 2048 | 150.1 | 79.2 | 67.7 | 30.2 | 1.17 |
| float32 | 64 x 1000 x 2048 | 151.9 | 58.4 | 39.0 | 21.2 | 1.50 |
| float32 | 512 x 1024 x 256 | 35.0 | 21.9 | 18.3 | 20.9 | 1.20 |
| float32 | 512 x 256 x 1024 | 77.1 | 40.6 | 24.9 | 17.3 | 1.63 |
| float32 | 1024 x 1024 x 1024 | 262.3 | 124.0 | 90.3 | 93.0 | 1.37 |
| float32 | 2048 x 4096 x 1024 | 1963.0 | 892.9 | 815.1 | 778.0 | 1.10 |
| float32 | 4096 x 4096 x 4096 | 15376.8 | 7093.4 | 5318.9 | 5055.5 | 1.33 |
| float16 | 1 x 4096 x 4096 | 131.3 | 11.8 | 22.7 | 21.4 | 0.52 |
| float16 | 16 x 4096 x 1024 | 34.5 | 5.1 | 6.4 | 5.3 | 0.80 |
| float16 | 64 x 2048 x 1024 | 39.9 | 6.5 | 14.3 | 6.8 | 0.45 |
| float16 | 64 x 2048 x 2048 | 75.2 | 10.2 | 21.6 | 11.2 | 0.47 |
| float16 | 64 x 1000 x 2048 | 72.0 | 7.6 | 9.3 | 6.2 | 0.82 |
| float16 | 512 x 1024 x 256 | 19.9 | 8.4 | 6.8 | 5.6 | 1.24 |
| float16 | 512 x 256 x 1024 | 54.6 | 6.5 | 20.9 | 6.9 | 0.31 |
| float16 | 1024 x 1024 x 1024 | 59.7 | 30.9 | 33.3 | 25.6 | 0.93 |
| float16 | 2048 x 4096 x 1024 | 302.4 | 198.6 | 208.7 | 179.9 | 0.95 |
| float16 | 4096 x 4096 x 4096 | 1955.4 | 1496.5 | 1624.6 | 1415.2 | 0.92 |
| bfloat16 | 1 x 4096 x 4096 | 131.5 | 11.8 | 22.4 | 21.3 | 0.53 |
| bfloat16 | 16 x 4096 x 1024 | 34.8 | 5.2 | 7.5 | 6.5 | 0.69 |
| bfloat16 | 64 x 2048 x 1024 | 39.9 | 6.4 | 8.2 | 7.0 | 0.78 |
| bfloat16 | 64 x 2048 x 2048 | 75.1 | 9.9 | 13.3 | 12.3 | 0.74 |
| bfloat16 | 64 x 1000 x 2048 | 144.6 | 7.6 | 10.6 | 9.4 | 0.72 |
| bfloat16 | 512 x 1024 x 256 | 19.9 | 8.4 | 7.0 | 5.3 | 1.20 |
| bfloat16 | 512 x 256 x 1024 | 54.8 | 6.5 | 8.5 | 7.4 | 0.76 |
| bfloat16 | 1024 x 1024 x 1024 | 59.8 | 31.2 | 28.5 | 26.4 | 1.09 |
| bfloat16 | 2048 x 4096 x 1024 | 299.9 | 198.7 | 208.8 | 196.7 | 0.95 |
| bfloat16 | 4096 x 4096 x 4096 | 1951.3 | 1444.5 | 1543.5 | 1489.2 | 0.94 |
| float32, TF32 allowed | 1 x 4096 x 4096 | 203.8 | 80.8 | 80.7 | 80.9 | 1.00 |
| float32, TF32 allowed | 16 x 4096 x 1024 | 20.2 | 9.5 | 26.1 | 12.9 | 0.36 |
| float32, TF32 allowed | 64 x 2048 x 1024 | 20.8 | 10.4 | 14.4 | 13.2 | 0.72 |
| float32, TF32 allowed | 64 x 2048 x 2048 | 39.7 | 18.7 | 25.8 | 24.7 | 0.72 |
| float32, TF32 allowed | 64 x 1000 x 2048 | 113.5 | 15.5 | 16.2 | 14.5 | 0.96 |
| float32, TF32 allowed | 512 x 1024 x 256 | 21.8 | 9.2 | 10.8 | 8.5 | 0.85 |
| float32, TF32 allowed | 512 x 256 x 1024 | 20.8 | 10.1 | 14.4 | 13.2 | 0.70 |
| float32, TF32 allowed | 1024 x 1024 x 1024 | 79.1 | 52.1 | 52.6 | 49.5 | 0.99 |
| float32, TF32 allowed | 2048 x 4096 x 1024 | 598.2 | 363.8 | 439.5 | 388.6 | 0.83 |
| float32, TF32 allowed | 4096 x 4096 x 4096 | 6320.3 | 2863.6 | 2943.0 | 2946.6 | 0.97 |

Kept, each measured on its own against the previous step:

* **Configs by operand class and program count.** Half, TF32 (only when compiled with
  `allow_tf32`, recorded on the plan) and strict float32 get their own defaults and candidates. The
  old defaults ran 4 programs for float16 at M 64 N 512; small M now uses 16 or 32 row tiles with BK
  128 (half) or 2 warp tiles (strict float32). Float16 64 x 2048 x 2048 went from 75 to 21 us.
* **Unmasked K loop with one masked tail (`KTAIL`).** With K a runtime size the K mask kept the
  operand loads from vectorizing. Float16 64 x 2048 x 2048 from 20.9 to 13.7 us, strict float32
  2048 cubed from 930 to 866 us. A launch config switch, since 128 x 128 half tiles measured 8% to
  15% faster with the masked loop.
* **Wide strict float32 tiles.** 128 x 256 x 16 with 8 warps when it fills a wave, 128 x 128 x 16
  with 4 warps below that: an FMA dot gains from more outputs per thread. 8192 x 256 x 1024 from 644
  to 528 us, 1024 cubed from 154 to 124 us.
* **64 x 64 tiles for large half and TF32 problems.** They matched or beat 128 x 128 tiles from 512
  x 1024 up to 4096 cubed (float16 8192 x 1024 x 256 from 127 to 77 us).
* `tune.SEARCH_SPACE` in the tune key: configs tuned over the old space are searched again (a plain
  miss, no `TuneCacheWarning`). Candidates were trimmed to the ones that won somewhere (8 to 11 per
  class).

Dropped or not done: GROUP 4 and 16 against 8 (no difference); 3 and 4 stages against each other
(within noise); 1 warp strict float32 tiles (no consistent gain). Split K, stream K and persistent
kernels are not implemented: split K needs a workspace and a second pass in the launch path and
changes the summation order with the config, so results would depend on the batch size bucket.

End to end, `scripts/bench_compiler.py`, batch 64, float32 (run ms, interleaved runs of the old
and new source on the same GPU):

| case | Truestar before | Truestar after | eager | torch.compile |
|---|---|---|---|---|
| MLP | 0.406 | 0.253 | 0.238 | 0.158 |
| TransformerEncoderLayer | 1.00 | 0.894 | 0.67 | 0.68 |

* GPU time per MLP step (torch.profiler): 397 us before, 243 us after (strict float32); 62 us with
  `allow_tf32=True`, where the step is bound by host overhead and still runs in 0.25 ms. MLP at
  batch 1: 0.47 to 0.20 ms.
* Transformer GPU time per step: 974 us before, 866 us after; 380 us with TF32 allowed (attention
  also uses TF32 then). The four contractions take 685 us of the 866 in strict float32.
* Cost: more candidates and M dependent defaults. First call at batch 64 (which tunes) went from 2.0
  to 4.4 s on the MLP and 9.6 to about 13 s on the transformer; first call at batch 7 on the
  transformer from 0.12 to 0.5 s, since M 896 picks another tile and compiles a new Triton variant.

* Found by the full suite: a contraction whose operand reads four buffers (softcap attention
  backward) needed 123 KB of shared memory at the new 16 x 64 x 64 default; defaults now count the
  bytes each pipeline stage buffers and drop stages, then BK, to stay within 96 KB.
* Full suite after the change: 986 passed, 3 skipped (7 min 26 s).

Not verified: other GPUs (the program count thresholds assume about 70 SMs), Triton versions other
than 3.7.1, an idle GPU.

### 2026-11-24: launch prep for 0.1.0 (local only, nothing published)

Branch `wt/polish`. CPU only: every run below had the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`) or
used CPU torch.

* **Missing numba (never introduce #6):** the CPU kernel paths raised a raw `ModuleNotFoundError`
  from inside a generated kernel file. New code TS022 (`optional_dependency_missing`) and
  `truestar.deps.MissingDependencyError` (a `ModuleNotFoundError`, so `except ImportError` still
  works) with `to_dict()` in the agent style (`next_actions`: `install_extra` with
  `pip install "truestar[cpu-kernels]"`, and `use_reference` where that is the way out). The kernel
  layer's `cpu` backend and `CompiledExecutor("cpu")` raise it before generating code; a generated
  file that still fails on `import numba` is translated; other import errors are not relabelled.
  Custom ops keep their documented fallback to the reference interpreter, and the
  `SlowPathWarning` names TS022. `truestar run` now forwards `--executor` (docs/COMPILER.md
  already showed it; the top level parser dropped it). Tests: tests/test_optional_deps.py, 9 cases,
  numba hidden through `sys.modules`. Tests can fail: removing the translation failed 1 test,
  removing the executor check failed 1 test (both restored). Proven for real in a clean venv
  without numba: `CompiledExecutor("cpu")` raised TS022 with the install command, a custom op fell
  back with TS022 in its warning and matched torch.
* **Packaging bug found by the clean install:** the kernel layer imports numpy, which was not a
  declared dependency and torch does not pull in; with only torch and the wheel installed,
  `import truestar.compiler` failed with `No module named 'numpy'`. `numpy>=1.22` is now a core
  dependency. CI never saw it because its test step installs numba, which brings numpy.
* **Metadata:** version 0.1.0 (read from `truestar.__version__`), SPDX license `Apache-2.0` with
  `license-files` (setuptools 77), `requires-python >=3.10` (the oldest CI runs; 3.9 was claimed
  but never run), classifiers, keywords, project URLs at github.com/SharvenRane/truestar.
* **Build** (throwaway venv in a temp folder: build 1.6.1, twine 7.0.0, setuptools 79.0.1 for the
  isolated build; deleted afterwards) from a `git archive` of the commit: sdist 568,037 bytes,
  wheel 465,174 bytes (98 files, LICENSE under `licenses/`, no tests, examples, scripts or docs),
  0 build warnings, `twine check --strict` PASSED for both.
* **Clean install** (fresh venv, Python 3.10.20, Windows): CPU torch 2.14.1 in 50 s, the wheel in
  1 s. From a folder outside the checkout: `truestar --help` exit 0; `truestar codes` lists 61
  codes including TS022; `truestar check examples/cpu_check.py --format agent` exit 0,
  `schema_version` 1, status ok, `truestar_version` 0.1.0, 3.6 s. Then with the `cpu-kernels`
  extra, pytest and torchvision 0.29.1: the test suite against the installed wheel (no `src`
  folder next to the tests) 720 passed, 149 skipped in 7 min 37 s; every skip needs CUDA, Triton
  or a GPU except one marked slow on the scalar CPU path.
* **Docs site, not deployed:** `mkdocs.yml` (mkdocs 1.6.1, builtin readthedocs theme;
  internal notes and upstream/ excluded) builds with `mkdocs build --strict`;
  `.readthedocs.yaml`; `.github/workflows/docs.yml` runs on manual dispatch only. CHANGELOG.md and
  a launch checklist list what is ready and the steps left (PyPI name, trademark,
  public repo, Pages or Read the Docs, tag and upload). None of them was done.
* **README:** the status section said no compiled executor exists (it does, behind
  `truestar run --executor`); a Measured section quotes only numbers from this log, losses
  included (attention against FlexAttention 1.18x to 2.14x, warm start 0.15 s against 3.05 s,
  CUDA graph step 3.2x, contraction ratios, compiled executor run times).
* Full suite in the worktree with the GPU hidden: 732 passed, 150 skipped in 8 min 29 s. GPU tests
  were not run in this entry. One earlier run, by mistake with the GPU visible (an empty
  `CUDA_VISIBLE_DEVICES` does not hide it on Windows), failed
  `test_customop.py::test_forward_and_gradients_match_torch[gelu_mlp-cuda]` on a GPU shared with
  benchmarks; the change here does not touch the CUDA path, but that test was not rerun.

Not verified: macOS install, Python 3.11 to 3.13, a real install from an index, the Pages and Read
the Docs builds on their hosts.

### 2026-11-24: float16 resnet backward false alarm on the scalar CPU draw

tests/test_gradcheck.py::test_clean_float16_resnet_reports_clean failed on some GitHub ubuntu
runners (TS180 at `<model>#0`, layer4.1.bn2.weight 1.13e-01) and locally with
`ATEN_CPU_CAPABILITY=default`. The cause is not a PyTorch kernel. It is the random draw plus a
Truestar false alarm.

* **No float16 kernel differs per capability.** scripts/probe_cpu_scalar_fp16.py (PyTorch only)
  runs the model on one fixed draw under DEFAULT, AVX2 and AVX512: the float16 gradients are bitwise
  identical (one hash for all three), and an op by op replay of all 275 forward and backward ATen
  ops of the DEFAULT run under AVX2 gave identical outputs.
* **The draw differs.** float32 `normal_` has an AVX2 Box-Muller (taken on AVX2 and on AVX512, which
  falls back to the AVX2 kernel) and a scalar one (DEFAULT, in
  aten/src/ATen/native/cpu/DistributionTemplates.h). They differ in the last bit, so
  `torch.manual_seed(0)` gives different resnet18 weights and inputs. The scalar draw holds a near
  tie: one layer4.1 ReLU input is -5.3e-4 in float64 and +1.2e-4 in float16. On that draw float16 is
  1.98e-1 off float64 at layer4.1.bn2.weight, a correctly rounded float16 run (float64 math, each op
  rounded) is also 1.98e-1 off, and the float16 kernels are 1.2e-3 from that rounded run. The AVX2
  draw has no flip at layer4 (1.3e-3 off). PyTorch is right; no upstream issue is drafted.
* **Truestar was wrong.** The emulated floor of that gradient is over the cap (20x the precision
  floor), so it was dropped and correct kernels were called a backward bug (never introduce #2).
  Fix: where the emulated floor is dropped, the backend gradient is also compared with the rounded
  run itself at the same threshold; landing there is OK and noted in the report. On the scalar draw
  the error there is 9.5e-4 vs threshold 5.0e-3.
* New tests, each shown failing on the old checker: the scalar draw in a subprocess with
  `ATEN_CPU_CAPABILITY=default` reports OK with the rounded run note (old: DIVERGENT), and a unit test
  of the rule (a gradient 3% off the rounded run stays DIVERGENT; old: TypeError; an excuse
  everything variant fails it). A guard test injects a 30% `aten.threshold_backward` fault on the
  same draw and requires DIVERGENT at `layer4.1.relu#1` with that op.
* Full suite after the change: 1011 passed, 3 skipped (11 min 6 s).

Not proven or not fixed:

* Which capability the failing ubuntu runners report. The CI error (1.13e-01) equals the local
  DEFAULT draw's exactly, which points to the scalar `normal_`, but x86 runners with AVX2 should take
  the vectorized path. Dispatch probe.yml with `probe_cpu_scalar_fp16.py`: it prints the CPU model,
  the process capability and the draw hash (scalar draw input hash 737795cbf0, AVX2 dedbf8fe6c).
  The new subprocess tests skip, with a reason, where the scalar draw has other bits.
* Parent replays still false alarm on some float16 draws where the backend and the rounded run flip
  a different near tie: 5 of 80 portable resnet18 draws on the old checker, 4 on the new one
  (docs/GRAD_CHECK.md, Limits). Needs an owner decision on how parents are judged in low precision.
### 2026-11-24: region and end to end checks on the GPU

The compiler's region checks (each kernel group and library step against the float64 reference
implementation of the IR ops it covers, `GroupCheck` in `compiler/runtime.py`) and its end to end
check (`ir.run` on the whole program, `CompiledProgram._verified_run`) now run on the program's
own GPU, with the device rule of the kernel layer (`TRUESTAR_VERIFY_DEVICE`, auto by default: the
GPU from 2^20 elements of work). Same reference implementations (`ir.reference`, now runnable
under `torch.device`), same tolerances, same schedule (first run of every program and region,
every loop body on its first iteration, a branch the first time it runs), same comparison rules
(`compare_torch`). RTX 5070 Ti shared with other jobs (another job at 75% utilization during the
after runs), torch 2.11, CUDA 12.8, Windows.

First call after compile, `scripts/bench_compiler.py --first-call --repeat 3` (new mode: fresh
process per run, cold caches, median of 3, float32), before (region and end to end checks on the
CPU, kernel checks already on the GPU) and after:

| case | first call (s) | region checks (s) | end to end check (s) | kernel checks (s) | the rest: Triton, tuning, run (s) |
|---|---|---|---|---|---|
| MLP, batch 64 | 5.40 to 5.94 | 0.026 to 0.022 | 0.011 to 0.019 | 0.29 to 0.17 | 5.07 to 5.72 |
| TransformerEncoderLayer, batch 64 | 13.9 to 16.3 | 0.363 to 0.066 | 0.114 to 0.030 | 0.18 to 0.23 | 13.2 to 16.0 |
| resnet18, batch 64 | **24.6 to 12.3** | **9.90 to 1.46** | **6.20 to 0.86** | 0.27 to 0.15 | 8.2 to 10.2 |
| causal attention B1 H8 S1024 D64 | 4.78 to 6.50 | 0.060 to 0.036 | 0.040 to 0.010 | 0.16 to 0.18 | 4.5 to 6.3 |
| causal attention B1 H2 S8192 D128 | 71.6 to 83.4 | 1.49 to 0.17 | 1.20 to 0.15 | 1.5 to 1.8 | 66.8 to 81.4 |

* The program level checks were the bottleneck only on resnet18: 16.1 of 24.6 s, now 2.3 s; its
  first call halves. On the others they were 0.04 to 2.7 s and are now 0.04 to 0.32 s. What is
  left of every first call is not verification: Triton compiles and the launch config search
  (66 to 81 s at attention S8192, 13 to 16 s on the transformer). It moved up between the runs
  because another job held the GPU at 75%; it was not looked at here.
* Of resnet18's 40 region checks, 39 ran on the GPU; the remaining 1.46 s is mostly the first
  check (conv1, 0.48 s, which pays the GPU's first float64 use in the process).
* Max error against float64 is identical before and after in every case (end to end and
  largest region error to three digits). Peak GPU memory: unchanged for the four model cases
  (314 MB to 3.86 GB, the program and the tuner); attention S8192 288 to 624 MB, from the blocked
  float64 reference (128 MB per score block). `torch.cuda.empty_cache()` after a check on the GPU
  gives that memory back to other jobs.

Never introduce #3, equivalence (tests/test_region_gpu_verify.py, 44 cases):

* **Same reference:** a hook runs the CPU reference next to every GPU region reference on MLP,
  TransformerEncoderLayer, resnet18 and three attention variants, float32 and float16: worst
  relative difference under 1e-12 (asserted) in every region value; outputs, max end to end error
  and max region error agree to 1e-12, reports equal. The interpreter on the GPU matches the CPU
  with a `while` loop, a branch, a symbolic loop and `return_all`.
* **Same blame:** broken lowering rules (relu in an MLP, in resnet18 after a convolution, in the
  transformer; tanh inside a fused region), a matmul broken inside a loop body, a neg broken in a
  branch that is checked when it first runs, a library convolution off by 0.1%, wrong wiring of a
  layout step (end to end stage) and attention wrong only in query rows 30 and up: each is caught
  on the GPU path with the same op, region, program, stage, fused op list, value, tolerance and
  the same error message text as on the CPU path.
* **Row blocks:** blocked attention equals the whole matrix (causal, bool, float and broadcast
  masks, a given scale; blocks of 1, 2, 5 and 36 rows) on CPU and GPU within 1e-14, and the
  late row fault above is caught with blocks of four rows.
* **Fallbacks:** a check over `TRUESTAR_VERIFY_GPU_MB` stays on the CPU and still catches; an op
  torch lacks on the device in float64 (simulated) falls back for that node only and is listed;
  a real error is not hidden as a fallback; CPU programs never use the GPU reference.
* **Tests can fail:** skipping the comparison when the reference ran on the GPU failed 11 of the
  new tests; scaling every GPU reference value by 1 + 1e-9 failed 22.
* One measured limit, not a regression: a 5% tanh error behind a sigmoid moves the float16
  output by 1.5e-3, inside the float16 tolerance on either device, so that case runs in float32.

Left on the CPU, on purpose:

* Programs compiled for the CPU, and checks below 2^20 elements of work in auto mode (the GPU's
  first float64 use costs 0.1 to 0.5 s once per process; the MLP's region checks are 22 ms).
* A check whose float64 working set estimate exceeds 2 GB, or that hits an out of memory error on
  the shared GPU.
* Naming the op after a mismatch (each op compiled alone): runs only on a failure.
* Host steps and fallback ops: these compute results, not checks.

Not verified: other GPUs, an idle GPU (the times above share the GPU with another job), ROCm.
* Full suite after the change: 1052 passed, 4 skipped (the 3 earlier skips plus the float16 tanh case above) in 9 min 43 s on the shared GPU.
### 2026-11-24: one kernel set for the RL trainer and sampler (P15 part b)

Built `truestar.rlkernels` (docs/RL_KERNELS.md): Triton kernels for matmul, attention (training
forward, prefill and decode against a KV cache in one kernel), RMSNorm and LayerNorm, softmax and
log_softmax and sums, with constant tiles, fixed order chunked reductions, chunks anchored at the
first unmasked key, and a 4 segment binary tree for weight contractions. `rl_mode()` routes a
plain PyTorch model to them and refuses reductions it has no invariant kernel for (no fallback).
`truestar rlpair --kernels rl` and `--tp N` verify it; new codes TS185 (kernel swapped, op named),
TS186 (TP mismatch), TS187 (not bitwise with the same kernels). RTX 5070 Ti, torch 2.11, CUDA
12.8, Triton 3.7.1, Windows; a Llama style model built in code (GQA through SDPA, RMSNorm,
rotary, SwiGLU; random weights).

* **Bitwise equal (acceptance):** per token logprobs of the training forward, the prefill (cache
  appended or a fixed capacity buffer), token by token decode (both cache layouts, and with the
  prompt prefilled in chunks of 3) at batch sizes 1, 3 and 8, the row first, middle or last,
  left padded in the sampler and right padded in the trainer: 84 of 84 comparisons bitwise in
  float32 and in bfloat16. Baseline, the same paths on PyTorch's kernels: 78 of 84 differ in
  float32 (max 9.5e-07), 10 of 84 in bfloat16 (max 3.8e-03).
* **TP 1 vs TP 2, a SIMULATION on one device** (ranks run in turn in one process, the all reduce
  adds float32 partials in a fixed tree): bitwise equal in float32 and bfloat16. A control with
  rank partials reduced in one pass is not bitwise. No real multi GPU collective was run.
* **Swapped kernel named:** a hand written decode attention in the sampler is TS185 at
  `blocks.0.attn#0` (trainer `attention:rl`, sampler `matmul:rl`, first decode step) on a gap of
  4.8e-07 that statistics call rounding; an eager attention or RMSNorm swapped in is named the
  same way. `truestar rlpair examples/rl_kernels_pair.py --device cuda --kernels rl --strict
  --tp 2` exits 0 (identical, batch invariant at 2 and 7, TP bitwise); without `--kernels rl` it
  exits 1.
* **Speed** (`scripts/bench_rlkernels.py`, fastest of 5 rounds, idle GPU), RL kernels over eager
  PyTorch: bfloat16 linear 1.04x to 1.18x at 512 and 4096 rows, 2.9x at 8 rows; float32 linear
  2.5x to 5.5x (FMA path against cuBLAS SIMT); attention prefill 1.4x (float32) and 2.4x
  (bfloat16, 5.5x over Truestar's tuned attention); decode attention 1.2x and 1.7x; norms 1.1x
  to 1.2x; softmax 2.0x in float32 and 0.83x to 0.88x in bfloat16. Model (dim 1024, 4 layers,
  vocab 32000): training forward B4 T512 2.0x in float32 and 0.60x in bfloat16 (eager SDPA with
  `enable_gqa` ran the math attention here), decode step 2.1x to 2.7x and 1.4x. The P14 target
  of under 20% is met by the large bfloat16 contractions, norms and softmaxes only.
* **Found while building (never introduce #3):** as `constexpr` flags, the masked decode attention
  compiled to another binary than the causal training forward and gave other bits from position
  2 (now runtime flags); Triton's alignment specialization gave an unaligned view its own binary
  with another `tl.sum` order (now off for reducing kernels, tested). Every kernel is compared
  with float64 PyTorch (float32 within 2e-5, bfloat16 within 2e-2 relative).
* **Training:** routed calls carry the eager backward; gradients of the model under `rl_mode`
  within 1e-3 relative of eager. Gradients are not claimed bitwise invariant.
* **Tests can fail:** permanent broken cases (unanchored attention, sequential TP partials, a K
  split chosen by row count, a swapped kernel, a TP mismatch). Manual breaks, restored: Triton's
  default specialization failed `test_unaligned_views_give_the_same_bits`; fp fusion turned back
  on failed nothing (kept as a guard).
* Full suite after the change: 1033 passed, 3 skipped (10 min 26 s); the 25 new tests need CUDA and skip without it.

Not verified: real tensor parallel on two GPUs, other GPUs and Triton versions, float16 end to
end, sliding window masks (not invariant with the per tile anchor), pretrained checkpoints, real
engines (vLLM, SGLang), CPU (no CPU path by design).
### 2026-11-24: sort and topk gradients, paged KV cache, Hugging Face coverage

Branch wt/opsgap. torch 2.11, transformers 4.48.0, Triton 3.7.1, RTX 5070 Ti shared with other jobs.
Rows served: P5, P7, P15 (one decode program), never introduce #1 and #3.

* **Sort and topk gradients (docs/AUTODIFF.md).** New IR op `scatter_add` (src added at indexed
  positions, repeated indices in increasing position order, reference written with indexed
  assignment). Rules for `sort`, `topk`, `gather`, `index_select` (so embeddings train) and
  `scatter_add`. Ties follow the stable sort: each tied entry gets the cotangent of its own sorted
  position, as `torch.sort(stable=True)`. Finite differences: worst 3.1e-13 over 9 new cases (bar
  1e-8). Against `torch.autograd`, an imported sampler (sort, softmax of sorted values, topk both
  ways, gather by sort indices), one program at batch 2 and 5: 2.4e-16, with ties 5.5e-16. Compiled
  on CPU and CUDA in float32: no fallback, within 1e-5 of the reference, bitwise equal run to run;
  a sort backward on CUDA equals the reference exactly.
* **Compiler.** `sort topk gather scatter_add index_select index_copy index` run through torch on
  the device (stable sort, clamped indices whose range and uniqueness are checked on the device and
  read once per call, `scatter_add` under deterministic algorithms); integer index arithmetic runs on
  the device exactly; kernel rules for `cos sin` (libdevice), `log abs reciprocal`, `cast`, `pow`
  with exponents the kernel IR writes exactly; `arange` with a symbolic end; float64 constants beyond
  the working range saturate (Hugging Face `finfo(float64).min` masks times 0 stayed NaN otherwise).
  Found by the first call check: a cast to float16 written as a copy (the reference rounds); it keeps
  the fallback.
* **Paged KV cache (docs/LLM.md).** Block table, pool shared by all sequences, `index_copy` writes
  at host computed slots, `index_select` gather by block table, per sequence positions. Logits
  bitwise equal to the dense cache in the float64 reference and compiled on CUDA in float32
  (scattered blocks, garbage in unused slots); ragged batches match each sequence alone (1e-13);
  tokens equal Hugging Face `generate` with copied weights. Decode benchmark (vocab 8192, width 512,
  4 layers, 64 new tokens), ms per token over two runs: batch 1 paged 4.65 and 5.71, dense 3.11 and
  3.68, eager dense step 3.39 and 2.98, HF generate 3.42 and 3.46; batch 8 paged 4.99 and 5.41,
  dense 3.70 and 4.73, eager 3.76 and 6.72, generate 3.30 and 3.63. Paged costs 1 to 2 ms per token
  more than dense; it is not faster than `generate`. Per op index syncs were the first bottleneck
  (8.2 ms per token), removed.
* **Hugging Face coverage (docs/MODEL_COVERAGE.md).** Llama, Mistral, Qwen2, GPT2, BERT, ViT and the
  T5 encoder from config classes, no downloads. Before: check 7 of 7, import 3 of 7 within 1e-10
  (Llama and Qwen2 refused `wrap_with_set_grad_enabled`, Mistral `bitwise_or_`, T5 off by 1.1e-7),
  compile 3 of 7 (GPT2 overflowed on its mask value), export 4 of 7. After: 7 of 7 at every stage,
  import worst 7.2e-16; compiled fallbacks on CUDA down from 5 (BERT) and 28 (T5) to 1 and 2.
  Fixes: higher order grad mode and disabled autocast wrappers are inlined; in place bitwise masks;
  `ir.reference.exact_float32_casts` keeps a model's own float32 casts in float64 when comparing
  against its float64 copy (the reference treats those casts as exact; the 1e-9 export bar is
  unchanged and a test shows the exporter refuses T5 without the mode); constants saturate.
* **No silent wrong results (#3) and no silent fallback (#1):** an enabled autocast region is
  refused; bad or repeated indices raise `IRError` in the reference and the compiled path without a
  CUDA device assert; every new device step is checked against the reference on the first call;
  remaining fallbacks are named in the report (`triu`, casts to integers, `pow` with other exponents).
* **Tests can fail:** a sort rule that gathers instead of scattering (off by over 1e-2), a device
  sort ignoring `descending` (stopped by the first call check), a shifted block table (tokens
  change), the exporter without the exact cast mode (refuses T5). Manual breaks, restored: a paged
  mask one slot too far failed 4 of 7 paged tests; `.float()` left in float32 inside the exact cast
  mode failed 6 Hugging Face tests.
* Full suite after the change: see the commit message.

Not verified: pretrained checkpoints (no downloads), the T5 decoder, torch versions other than 2.11,
an idle GPU for the benchmark, in place pool updates and a fused paged attention kernel (not built).

### 2026-11-24: custom gradients verified at registration, DLPack, safetensors

Branch `wt/interop`. Research items R9 and R15 (custom gradients, DLPack and safetensors interop).
Docs: docs/AUTODIFF.md (custom gradients section), docs/INTEROP.md (new). Everything below was run
on CPU; the DLPack CUDA tests (zero copy, stream handles, a consumer stream waiting on pending
producer work) and the CUDA load rows ran on the RTX 5070 Ti.

* **Custom gradients (P18 for hand written backwards, N6).** `@fn.grad` on a native function
  takes the rule by the primal's argument names plus `g` and `out`. Registration type checks every
  gradient against its parameter, then compares the rule with reverse mode autodiff of the primal
  (entrywise, bar 1e-8, when every op has a rule) and with fourth order finite differences along
  three random directions per input (bar 1e-6, kink directions redrawn), at two sizes of each
  symbolic dim. A wrong rule raises `CustomGradientError` naming every wrong input and both errors;
  `unverified="reason"` is the explicit opt out. The program becomes one new IR op `custom_vjp`
  (regions `fwd` and `bwd`); `grad` inlines `bwd`, `torch_op` registers it as the backward, the
  compiled executor and the kernel path run the `fwd` region (both forward and backward ran on the
  kernel path in a test).
* **Against torch.autograd.Function** (`scripts/custom_grad_compare.py`): sign error, missing term
  and mean instead of sum are accepted by PyTorch and run without error; a hand run `gradcheck`
  catches them. Truestar refuses each at registration in 0.09 s to 0.11 s, naming x, w or b. An
  unreduced bias gradient is summed back silently by the PyTorch engine (gradcheck passes);
  Truestar refuses its type and suggests `.sum(0)`. A correct rule verifies in 0.06 s (bar 1 s).
  JAX is not installed; its documentation describes no check of `bwd`.
* **DLPack.** `truestar.interop.from_dlpack` reads the capsule (DLPack 0.x and 1.0) before PyTorch
  does. Found on the way: `torch.from_dlpack` of a reversed NumPy view kills the process on torch 2.11
  (exit 0xC0000409; known as pytorch#188023, already a catchable RuntimeError on 2.14.1 and 2.15.0, measured on Linux and Windows CI), and a read only NumPy array comes back as a
  writable tensor. Truestar refuses both with a code and a fix, checks dtype, rank, fixed dims and
  device against a declared type, never copies across devices without `copy=True`, passes the
  consumer stream (legacy default as 1, never 0), and asserts zero copy by pointer. Native functions
  take NumPy arrays directly. Cost per import: 4.6 us against 1.1 us (NumPy), 23.6 us against 18.0
  us (foreign CUDA producer); a torch tensor passes through in 0.3 us against 18.2 us.
* **safetensors without the package.** Reader validates the whole header and lists every problem
  (duplicates, which Python's json hides, holes, overlaps, byte lengths, trailing bytes); tensors are
  copy on write memory map views. Writer output is byte identical to `safetensors.torch.save_file`.
  `load_weights(path, into=function | program | module)` lists every missing, unexpected, mistyped
  and misshapen tensor before assigning anything, binds symbolic dims across weights, fills tied
  module weights. `save_weights(..., check=model)` stores reference outputs on inputs generated
  from a formula; `load_weights` reruns them, which catches swapped same shape tensors and
  transposed square weights that `load_state_dict(strict=True)` accepts.
* **Load time** (Qwen2.5 0.5B, 988 MB bfloat16, warm cache, median of 5, every byte read): Truestar
  0.82 s, `safetensors` 0.79 s (4% slower, target 10%), `torch.load` 0.80 s, `torch.load(mmap=True)`
  0.98 s; lazy open 0.0025 s; to CUDA 0.40 s against 0.43 s. GPT2 (548 MB) into `GPT2LMHeadModel`
  with the record rerun: 0.20 s, logits identical to `from_pretrained`; a transposed 768 x 768
  `c_proj` weight refused (error 0.51) while `load_state_dict` accepts it.
* Tests: tests/test_custom_grad.py 16, tests/test_dlpack.py 18, tests/test_safetensors.py 15, plus a
  `custom_vjp` finite difference case in tests/test_autodiff.py. Tests can fail: with the gradient
  bars at 1e9, 4 tests fail; with safetensors duplicate detection and the record rerun disabled, 3
  fail; without the negative stride check the DLPack test would take the test process down.
* Full suite in the worktree with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`): 782 passed, 153 skipped in 8 min 28 s (skip reasons not listed in this run); the CUDA DLPack and safetensors tests ran separately with the GPU visible (all pass).

Not verified or not done: JAX and CuPy producers (not installed; imitated by a wrapper), ROCm
streams, cold cache load times, custom gradients with several outputs or a forward mode rule, the
Hugging Face GPT2 file into the native decoder of docs/LLM.md (needs a name map), big endian hosts.

### 2026-11-24: low precision range check (`truestar range`, P21, research R3) and precision manifest (R8)

Branch `wt/range`, CPU only (torch 2.11, transformers 4.48, 2 threads on a shared PC). Docs:
docs/RANGE.md. Code: `src/truestar/rangecheck/`, `src/truestar/precision_probe.py`.

Redesign, not a copy of today's practice (export, run in fp16, add intermediates as graph outputs,
count infinities, force nodes to CPU):

* **More correct.** One float64 pass records every value's peak and the hidden intermediates of
  composite ops ((x - mean)^2, mean of squares, attention scores), static interval bounds prove
  values safe for any input in the calibration box, and the emulated target is judged against a
  twin with the same significand bits and unlimited exponent range, so rounding is never called a
  range problem. Ground truth for the measurements is PyTorch's own float16 CPU kernels, every op
  fed correct inputs (`rangecheck/groundtruth.py`), never the emulator.
* **Verified remedies.** An exact power of two rescale of the out of range region (weights and
  edges scaled in, norms take eps * 2^-2k), float32 regions, clamps; each rerun in emulation, the
  first that passes is chosen. The rescale's float64 output must equal the original's.
* **Easier and faster.** One command on the CPU before export: 1.7 to 6.9 s per model here (import
  included), no device and no export loop.

Measured (docs/RANGE.md, `scripts/range_study.py`):

| model | target | true causes | found | false pos. | emulated cosine (PyTorch chained) | remedy, cosine min after |
|---|---|---|---|---|---|---|
| bge small style (outliers at layers 0, 3, 4) | webgpu-f16 | 3 | 3 | 0 | 0.1552 (0.1552) | rescale 2^-1, 0.999996 |
| bge small style | fp16 | 0 | 0 | 0 | 1.0000 | clean |
| Gemma 3 style (residual peaks 1.08e5) | fp16 | 1 | 1 | 0 | NaN (NaN) | rescale 2^-2, 1.000000 |
| Gemma 3 style | ane | 1 | 1 | 0 | infinite (NaN) | rescale 2^-2, 1.000000 |
| Gemma 3 style | bf16 | 0 | 0 | 0 | 1.0000 | clean |
| BERT base shape | fp16 and webgpu-f16 | 0 | 0 | 0 | 1.0000 | clean |
| ResNet50, calibrated batch norm | fp16 | 0 | 0 | 0 | 0.9983 (0.9984) | clean |
| examples/range_model.py | fp16 | 1 | 1 | 0 | NaN (NaN) | rescale 2^-2, 1.000000 |

* The Gemma Hub workaround (clip at +-32752) fails verification on the Gemma style model: cosine
  0.984, all 32 argmax positions change. The exact rescale passes with float64 rewrite error 1.4e-16.
* Static bounds prove 273 of 316 values safe in bge small, 883 of 944 in the Gemma style decoder,
  12 of 175 in ResNet50.
* Also found: underflowing logits near 1e-5 (TS304, fixed by one float32 op) and the one pass
  variance of examples/check_model.py as a precision loss at check_model.py:20 (TS305).
* False alarms found and fixed while building: the float32 mask constant -3.4e38 became -inf in
  float16 and 0 * -inf gave NaN on clean BERT (constants now saturate, as float16 converters do);
  an unsound static bound for norms whose weight is computed (Gemma's 1 + w) "proved" a 6.75e4 value
  safe at 258 (now uses the weight's interval; a soundness test over four models guards it); the
  tanh gelu cube and the exponential in sigmoid overflow harmlessly and are not reported.
* **R8 precision manifest** in `truestar check`: per backend, the input significand bits, the
  accumulator (16 bit dtypes) and subnormal handling of linear, bmm and conv2d, measured with crafted
  inputs, next to the flags. On this CPU: float32 24 bits everywhere; bfloat16 conv2d flushes
  subnormals while linear and bmm keep them.
* New codes TS300 to TS309 (docs/AGENT_INTERFACE.md). Tests: tests/test_rangecheck.py (17),
  tests/test_precision_probe.py (4), one example test. Tests can fail: dropping constant saturation
  failed 4, a loose layer norm bound failed the soundness test, running webgpu-f16 fused failed 2,
  making the twin round like the target failed 4, dropping the eps rescale failed the exact rescale
  test (all restored).
* Full suite with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`): 757 passed, 150 skipped (the GPU tests), 0 failed in 10 min 19 s. GPU tests were not run in this entry.

Not verified: any real low precision device (Neural Engine, WebGPU, TensorRT); the real bge small
and Gemma 3 270M checkpoints (synthetic weights scaled to the published peaks); float16
accumulation; scaled FP8; the precision manifest on a GPU (its TF32 and fp16 accumulation labels are
tested with simulated kernels only).

### 2026-11-24: gradients through control flow, torch.export control flow, pytrees

Branch `wt/ctrlflow`. CPU only, float64 reference unless stated. Pain rows P5 (one program for every
loop length, now also for training), P7 (branchy and looping models export, nothing baked in) and
P9 (structure mismatches named by leaf). Owner rule: each feature redesigned, not copied.

* **Gradients through `cond` and `loop`** (`ir/autodiff_cf.py`, docs/AUTODIFF.md "Control flow").
  Before: `grad` refused any program with control flow on the gradient path. Now the backward of
  a `cond` is a `cond` of the branches' gradients and the backward of a `loop` is a reverse `loop`
  of the body's gradient, both built by `grad` on the regions, so nesting recurses. A memory policy
  per program: `store` (only the carried values the body gradient reads), `sqrt` (checkpoints),
  `recompute`; each backward loop names the policy it got. Measured on a native RNN (batch 4,
  hidden 32): kept copies of the carried state at T = 8, 32, 48: store 8, 32, 48; sqrt 6, 12, 14
  (at most `2 ceil(sqrt T)`); recompute 0; reference run time at T = 48: 48.7, 100.5 and 540.1 ms.
  Errors: finite differences worst 2.4e-10 (bar 1e-8) over 8 programs times 3 policies; against
  torch.autograd on the unrolled programs worst 7.8e-17 (bar 1e-12) over 14 cases times 4
  policies, including zero iteration loops, both sides of every branch and a `while` inside an `if`
  inside a `for`. One compile serves every T. Checked once by hand: `CompiledExecutor("cpu")`
  runs the three gradient programs in float32 within 2.7e-7 of the reference.
* **`truestar.grad(fn, wrt="params")`** on native functions: value and gradients with the
  structure of the parameter, against torch.autograd below 1e-12 at T = 1, 4, 9.
* **Custom ops** made from native functions with control flow now have a backward (it was refused
  by name); gradients match PyTorch on both sides of the branch.
* **Importer** (`torch.cond`, `while_loop`, `scan`): each subgraph becomes an IR region converted by
  the same code, `scan` becomes a `loop` with `select_dynamic` and `index_copy`. Both graph forms,
  both sides of every branch, data dependent trip counts, nested `cond`, `cond` inside `scan`,
  reverse scan, symbolic T and B: all match PyTorch in float64 at 0 relative error on the tested
  inputs (bar 1e-12); gradients of the imported `cond` and `scan` match torch.autograd. An unbounded
  `torch.export.Dim` crashed the importer (torch's `int_oo`); fixed.
* **Export**: the batch symbol is unbounded by default (was `[1, 1024]`); `dynamic_dims` and
  `--dynamic x:1=T`, each symbol checked at a second size; `torch.cond` exports with the side no
  check takes run on the operands that reached it and compared with torch's own captured branch
  (a permanent test breaks only that side and the export is refused); coverage in the evidence.
  Nested inputs and outputs and keyword arguments round trip through the artifact, also in a fresh
  process; a mismatch names the leaf (`batch.x: is torch.float64 but the artifact declares
  f32[B, 4]`).
* **Front end pytrees**: dict, list, tuple, dataclass and NamedTuple parameters and results with
  no registration; the IR stays flat (one input per leaf, named by path).
* **Tests can fail**: dropping capture gradients and zeroing stored carries each fail by more than
  1e-2 (permanent); a broken never taken branch side refuses export (permanent).
* Changed expectations, both on purpose: `test_export_dynamic_batch_preserved` now expects the
  batch range `[1, None]`; the custom op control flow test now checks gradients instead of the
  refusal. `test_captured_branch_is_exported_correctly_or_refused` lost its refusal path.
* Full suite with the GPU hidden: 881 passed, 1 failed, 150 skipped in 8 min 53 s. The failure was tests/test_examples.py::test_export_then_run expecting the old batch range `B in [1, 1024]`; it now expects `[1, inf]` and that file was rerun green (11 passed, 4 skipped). GPU tests were not run.

Not verified or not built:

* CUDA: nothing in this entry ran on a GPU.
* The `store` policy keeps carried values, not residuals, so each backward step reruns the body
  forward once. An unbounded `while` loop always recomputes (quadratic).
* `torch.map`, `associative_scan`; a structure reassigned inside a symbolic loop in the front end;
  a symbolic dim inside a nested export argument.
* The never taken branch side is compared with torch's captured subgraph on the operands the other
  side saw first, not on inputs that would take it.

### 2026-11-24: derived tolerances, kernel check, verified autotune, near tie parents (P19, P18)

Branch `wt/tolerance`. CPU plus RTX 5070 Ti (shared with other agents), torch 2.11, CUDA 12.8,
Triton 3.7.1, Windows. Owner's rule applied: each part is a redesign measured against the tool
people use today.

* **R6, `truestar.assert_close(fn, inputs, out)`** (docs/TOLERANCE.md): no rtol or atol. Per element
  bound from float64, the rounded run the backward check already uses, and 8 error model runs (each
  op result perturbed by what a correct kernel can be off: one or four ulps, running sum
  accumulation error for reductions, TF32 operands and cuBLAS reduced precision split K per the
  device policy). `scripts/bench_tolerance.py --draws 5 --cuda` on 38 kernels (105 correct runs, 85
  wrong): false fail 0.0% and false pass 5.9% (all 5 are a constant off by one to two float16 ulps),
  against `torch.testing.assert_close` defaults 32.4% false fail and allclose 1e-2 28.2% false pass.
  Correct kernels peak at 0.46 of their bound. Median 0.017 s per check. pytest fixtures in
  `truestar.pytest_plugin`.
* **Bugs found in my own first version by the benchmark, each fixed with a test shown failing on the
  old code:** the reference cache keyed on `data_ptr` returned another tensor's reference when a new
  input landed at a freed address (entries now hold their inputs); float32 GELU cancels to 3.7 u |x|
  where the output is tiny (rule for `x * g(x)` activations); cuBLAS float16 GEMMs at K = 2048 and
  8192 are thousands of ulps off float64 at outputs that cancel, which PyTorch's default reduced
  precision reduction allows (split K term); `aten._safe_softmax` had no rule.
* **R1, `truestar kernel ref.py cand.py` and MCP `verify_kernel`** (docs/KERNEL_CHECK.md): shape,
  value, stride, dtype and repeat variants, NaN poisoned CUDA allocator, input mutation check,
  reference after the candidate, ATen op blame, codes TS200 to TS206. `scripts/bench_kernel_check.py
  --draws 3 --cuda` on 50 kernels (39 PyTorch on CPU, 11 Triton on GPU; 69 correct runs, 81 wrong):
  false fail 0.0%, false pass 6.2%; one input allclose 1e-2 0.0% and 42.0%; allclose 1e-4 36.2% and
  22.2%; torch defaults 23.2% and 16.0%. Median 0.29 s per kernel, worst 3.6 s. The 5 false passes:
  the two ulp constant (2 of 3 draws) and a 16 step float16 accumulator that PyTorch's default policy
  lets cuBLAS match (caught at over 100x its bound with that flag off, and at K = 4096 under it).
  Example `examples/kernel_reference.py` with `kernel_candidate.py`: right at 1024 columns, named
  wrong at 1025 in 1.8 s.
* **R7, `@truestar.verified_autotune`** (TS207): every config checked against float64 with NaN
  prefilled outputs before timing; wrong ones removed with a reason; none right raises. Matmul with no
  K mask at K = 1040: `triton.autotune` picked BK = 64 and returned a wrong result; verified removed
  the 4 BK = 32 and 64 configs and picked BK = 16, right. Overhead when nothing is removed: +5.4%,
  +4.7%, +8.3% (budget 20%). triton#12525 did not reproduce on Triton 3.7.1; #11153 not tried.
* **Owner call, float16 parent false alarms** (docs/GRAD_CHECK.md, `src/truestar/gradties.py`, new
  code TS188, next action `accept_near_tie`): a parent over threshold with every leaf clean is
  SUSPECT, not DIVERGENT, only when every ReLU mask that differs from float64 is a near tie and the
  rounded run with the backend's masks forced matches the backend gradient. 80 portable resnet18
  float16 draws (`scripts/probe_fp16_parent_ties.py`, CPU): before 76 OK and 4 DIVERGENT (draws 8,
  27, 36, 70), after 76 OK and 4 SUSPECT (10 to 22 flipped elements each, all within 0.18% of the
  input RMS of zero), 0 DIVERGENT. Injected faults DIVERGENT on 80 of 80 draws before and after: a
  30% `aten.threshold_backward` fault, a 30% `aten.convolution_backward` fault (both caught at
  leaves) and a 15% wrong backward in the model's own code, seen only by the whole model replay,
  which is the case the new rule could have excused. Smaller faults, measured on draws 0 and 1 with
  the old checker only: 1% and 3% are SUSPECT at a leaf (TS181), 0.4% is not caught; the new rule
  never changes a leaf, so it cannot change those.
* **Tests can fail:** tolerance with LAMBDA at 1e6 failed 20 tests, without the accumulation term 5;
  kernel check without shape and stride variants failed 14; verified autotune accepting every config
  failed 2; the two near tie tests fail on the old checker.
* Full suite with the GPU visible (shared with other jobs): 1101 passed, 3 skipped in 12 min 20 s
  (was 1011 passed, 3 skipped).

Not done or not proven:

* The YAKK lemon families and the 208 METR kernels were not run; the acceptance numbers above are on
  Truestar's own kernel set.
* Process isolation of the candidate, a roofline bound on impossible speed, and a check that the
  target DSL ran are not built.
* Convolutions keep the classic `sqrt(K) u sum|x w|` accumulation bound, less sensitive than the
  running sum rule for matmuls.
* Only ReLU masks are forced in the near tie rule; max pool route flips are not explained that way.
### 2026-11-24: contraction speed, split K, PyTorch precision flags, first call cost

Follow up of the "contraction kernels closer to cuBLAS" entry. RTX 5070 Ti shared with up to three
other agents, torch 2.11, Triton 3.7.1, Windows. Times are medians of 9 interleaved rounds of CUDA
graphs of 20 calls (`scripts/bench_matmul.py`, new), the minimum is given where noise showed;
before is main at ae8e73f run by the same script in the same session.

**1. Strict float32 (TF32 off), `relu(x @ W.T + b)`, default launch config (us):**

| M x N x K | before | after (median / min) | F.linear + relu | bare `x @ W.T` | torch.compile | after / F.linear | after / best cuBLAS path | split K |
|---|---|---|---|---|---|---|---|---|
| 1 x 4096 x 4096 | 112.2 | 112.1 / 111.7 | 82.3 | 82.4 | 82.3 | 1.36 | 1.36 | 1 |
| 16 x 4096 x 1024 | 29.2 | 29.2 / 29.2 | 43.1 | 37.8 | 38.9 | 0.68 | 0.77 | 1 |
| 64 x 2048 x 1024 | 38.2 | 26.8 / 26.7 | 39.3 | 24.0 | 25.3 | 0.68 | 1.12 | 4 |
| 64 x 2048 x 2048 | 74.5 | 45.6 / 45.4 | 68.1 | 30.9 | 32.2 | 0.67 | 1.47 | 8 |
| 64 x 1000 x 2048 | 56.4 | 28.7 / 28.6 | 40.0 | 21.6 | 22.7 | 0.72 | 1.33 | 8 |
| 512 x 1024 x 256 | 22.5 | 22.5 / 22.4 | 18.4 | 21.1 | 24.6 | 1.22 | 1.07 | 1 |
| 512 x 256 x 1024 | 38.2 | 27.0 / 26.8 | 26.7 | 17.5 | 18.9 | 1.01 | 1.54 | 4 |
| 1024 x 1024 x 1024 | 132.0 | 126.7 / 126.4 | 90.6 | 94.7 | 98.7 | 1.40 | 1.34 | 2 |
| 2048 x 4096 x 1024 | 931.3 | 865.2 / 856.5 | 839.0 | 796.8 | 832.0 | 1.03 | 1.09 | 1 |
| 4096 x 4096 x 4096 | 7626 | 6880 / 6809 | 5380 | 5098 | 5211 | 1.28 | 1.35 | 1 |
| 64 x 512 x 8192 (tall K) | 226.7 | 62.2 / 53.5 | 76.7 | 30.2 | 39.0 | 0.81 | 2.06 | 16 |
| 16 x 1024 x 16384 (tall K) | 437.6 | 124.4 / 112.4 | 188.1 | 176.0 | 179.4 | 0.66 | 0.71 | 4 |
| 128 x 256 x 8192 (tall K) | 219.4 | 62.2 / 53.6 | 103.1 | 28.5 | 29.6 | 0.60 | 2.18 | 16 |
| 8192 x 768 x 256 | 184.6 | 176.3 / 174.0 | 176.7 | 178.3 | 195.9 | 1.00 | 0.99 | 1 |
| 8192 x 1024 x 256 | 243.7 | 269.5 / 229.6 | 250.8 | 238.0 | 273.2 | 1.07 | 1.13 | 1 |
| 8192 x 256 x 1024 | 234.5 | 249.3 / 216.8 | 193.2 | 187.2 | 196.2 | 1.29 | 1.33 | 1 |

Max relative error against float64: 1.7e-7 to 2.2e-6, equal to or below cuBLAS on every split K
shape (split K shortens each sequential sum, 2.1e-7 to 3.5e-7 against cuBLAS 5.9e-7 to 8.6e-7),
equal on the large shapes, and above cuBLAS (up to 1.3e-6 against 1.7e-7) on 1 x 4096 x 4096 and
16 x 1024 x 16384, all far below the 2e-5 float32 tolerance. The last three rows are noisy
(another agent ran during the after run; their minimums are at or below before).

* **Split K** (new, launch config `SPLIT_K`): the tall K and small M shapes went from 2.2x to 3.5x
  behind F.linear to 0.6x to 0.8x of it (64 x 512 x 8192 from 227 to 62 us, 16 x 1024 x 16384 from
  438 to 124 us, faster than cuBLAS there). Second pass in a fixed order, no atomics.
* **Tile configs:** a 270 config sweep per shape (every BM, BN, BK, warps, stages) found the old
  defaults within 3% of the best tile for strict float32, except 128 x 256 x 16 with 3 stages
  instead of 2 (2048 x 4096 x 1024 from 931 to 865 us) and two K chunks for 128 x 128 tiles under
  one wave (1024 cubed 131 to 126 us).
* **3xTF32 tried, dropped.** Triton's `tf32x3` (and `bf16x6`, `bf16x3`) were measured through the
  same kernel: `tf32x3` is more accurate than the FMA dot (2e-7 against 1e-6) and wins only where
  split K now wins too (64 x 2048 x 2048: 41 us), but is 1.5x to 2x slower on large shapes (1024
  cubed 231 us, 2048 x 4096 x 1024 1535 us: on this GeForce TF32 tensor cores run at the FMA rate,
  so three products cost three times). It also overflows to infinity for inputs above about
  3.40e38 (the TF32 rounding of the high part) and loses accuracy for whole tensors below about
  1e-34 (the low parts flush to zero): new silent problems, so it is not used. `bf16x3` misses the
  accuracy bar (3e-6 to 4e-6 against cuBLAS 5e-7 to 1e-6).
* **Still loses:** large strict float32 problems are 1.03x to 1.4x behind cuBLAS (1024 cubed 1.34x,
  4096 cubed 1.35x, 8192 x 256 x 1024 1.33x): Triton's IEEE FMA dot reaches about 20 TFLOPS where
  cuBLAS reaches 27, and no tile config closes it. Small M with mid K is 1.1x to 1.5x behind the
  bare cuBLAS product (64 x 2048 x 2048 1.47x, 512 x 256 x 1024 1.54x) though 0.67x to 1.0x of
  F.linear with its separate relu; 1 x 4096 x 4096 is 1.36x behind (no split K setting helped).
  Tall K with N 256 or 512 is 2x behind the bare product (the FMA loop's load rate, not
  parallelism: split 32 and 64 were slower).

**2. Split K and determinism.** Without split K a contraction's result is bitwise independent of
the launch config (72 configs of BM, BN, BK and KTAIL, IEEE and TF32 dots), so of the batch size.
With split K the chunk count follows M, so a row computed in a batch of 7 and of 300 can differ in
the last bits (test shows it). Runs stay deterministic in both modes. `deterministic=True` (kernel
layer and compiled executor), or the default `None` in the compiled executor, which follows
`torch.use_deterministic_algorithms`, never splits K: rows are then bitwise equal across batch 7,
64 and 300 (tests/test_contraction_split.py, tests/test_compiler_tf32.py). Cost in deterministic
mode: the before column of the small M rows.

**3. PyTorch precision flags.** The compiled executor (`CompiledExecutor`, `compile_program`, the
export runner) now defaults to `allow_tf32=None`: every call reads
`torch.backends.cuda.matmul.fp32_precision`, which all three APIs set (`allow_tf32`,
`fp32_precision`, `set_float32_matmul_precision`; "high" and "medium" are both TF32 on CUDA, as
verified against what cuBLAS runs). Found: once the legacy and new APIs are mixed, reading
`allow_tf32` or `get_float32_matmul_precision()` raises in torch 2.11, so they are only a fallback
for older torch. The first call under a new setting compiles and verifies that variant (one
program, like a torch.compile guard recompile); switching back is free. Tests both ways for each
API, pinned True and False, the native front end, float16 ignoring the flag.

**4. First call cost** (`scripts/bench_compiler.py --warm --repeat 2`, cold caches, fastest of two):

| case | first call batch 7, before / after (ms) | first call batch 64, before / after (s) | of which search, before / after (s) | warm cache batch 64 (ms) | run batch 64 (ms), before / after | eager / torch.compile (ms) |
|---|---|---|---|---|---|---|
| MLP | 9 / 14 | 4.05 / 1.90 | 3.6 (2 searches) / 0 | 44 / 48 | 0.252 / 0.170 | 0.239 / 0.163 |
| TransformerEncoderLayer | 467 / 31 | 11.3 / 4.8 | 11.3 (7 searches) / 1.7 (1, attention) | 91 / 105 | 0.897 / 0.901 | 0.668 / 0.667 |

* Contraction defaults now count the programs of batch axes (a projection imported from
  `[B, T, C]` is a batched contraction with M = T, or M = B with the tokens as batch) and use 128
  row tiles only above 64 rows. Before, the batch 64 transformer defaults were up to 3.4x slower
  than tuned; now every contraction default there is within 2.5% of its tuned config, measured by
  the tuner itself. So automatic tuning starts at 2^32 multiply adds for contractions and 2^24
  elements for tile kernels (tile searches gained at most 4% for 0.8 s); attention keeps 2^28. An
  in process A/B of the transformer step at batch 64: 0.894 ms with the old thresholds (5 searches),
  0.901 ms with the new (1 search).
* A search builds every candidate, the default included, in one thread pool, and contraction
  candidates are pruned by shape (at most 5 tiles plus split neighbours): a transformer contraction
  search fell from 1.5 to 2.2 s to 1.0 to 1.6 s.
* Tuned contraction configs are also stored under a shape key (operands and sizes, no epilogue):
  in the MLP, `linear + relu` read the config tuned for `linear + gelu` (source `shape cache`).
  The row count of a contraction enters tune keys in powers of four.
* Batch 7 first call: a launch below 2^28 multiply adds reuses a config its kernel already compiled
  when the cost model rates it at most 1.5x the default (`REUSE_SLACK`), so no new Triton variant:
  transformer 467 to 31 ms. Only for small launches: reusing batch 1 and 7 configs at batch 64 cost
  8% of the transformer step, so larger launches compile their own.
* Split K launches compile their two kernels in two threads and the partial sum is a loop, not
  unrolled: the MLP's first call at batch 64 went from 8.1 s (first version of split K) to 1.9 s.
* Warm caches (a second process): 48 and 105 ms at batch 64, no search, as before.
* Loss: the MLP's batch 7 first call is 9 to 14 ms (noise level); transformer run at batch 64
  0.897 to 0.901 ms, within noise, still 1.35x behind eager (the strict float32 FMA gap above);
  the MLP now beats eager (0.170 against 0.239 ms) and is level with torch.compile (0.163).

**Tests.** New: tests/test_contraction_split.py (19: split K 1 to 16 with ragged and empty chunks,
both K loops, float16 and bfloat16, batched with a residual epilogue, run to run equality, config
independence without split K, batch dependence with it and its absence in deterministic mode,
deterministic rejects a split config), tests/test_compiler_tf32.py (11), and in
tests/test_kernel_tune.py candidate pruning, shape cache, power of four buckets, variant reuse and
its size limit. Full suite: 1044 passed, 3 skipped (was 993 at ae8e73f). Found by the full suite: the
TF32 tests left PyTorch in its "mixed precision APIs" state, where reading `allow_tf32` raises, and
74 later tests failed; the fixture now restores every API. The contraction correctness test now runs the whole tile grid of each class plus
split K configs, not only the pruned candidates. Tests can fail (each restored): summing only
SPLIT_K - 1 partials failed 22 tests; a precision reader that always says IEEE failed 7; reuse that
ignores the cost model failed 1.

Not verified: other GPUs (thresholds assume about 70 SMs), Linux, Triton other than 3.7.1, an idle
GPU (every number above was taken next to other agents' jobs), float16 and bfloat16 speed after
the change (correctness only), `torch.backends.cudnn.allow_tf32` (convolutions follow the same
resolved setting as matmuls, not that flag).
### 2026-11-24: DLPack consumer stream fix (torch tensor pass through skipped the wait)

Found after the interop merge: `tests/test_dlpack.py::test_cuda_consumer_stream_waits_for_the_producer`
failed in the full suite while other GPU work ran and passed alone. Root cause in
`src/truestar/interop/dlpack.py`: a torch tensor was returned as is, so the `stream=` argument was
ignored and no capsule was made, which meant nothing ordered the consumer stream after the producer
stream. The old test passed alone only because the GPU finished the producer's kernels before the
consumer's launched. Under load (a stress repro with a hog thread) 17 of 40 exchanges read the value
before the producer finished.

* Fix: for a CUDA torch tensor the consumer stream now waits on the stream current at the call (what
  a producer's `__dlpack__(stream=...)` must do per the array API standard), skipped when they are
  the same stream or the consumer passes -1; `record_stream(consumer)` is set on the tensor so the
  caching allocator does not reuse its block while consumer kernels are pending. Tensors imported
  from a capsule get the same `record_stream` (ignored for memory a foreign allocator owns). The
  stream argument of a CPU torch tensor and the ambiguous stream 0 are now refused on this path too.
* New permanent tests: 20 exchanges per producer kind (torch tensor, foreign producer) while a hog
  stream queues 8 matmuls of 4096 x 4096 before each, 0 failures required; a stream handling test
  for the pass through path. On the old code: torch tensor 15 of 20 failed, the stream handling test
  failed, the foreign producer passed (its `__dlpack__` waited). On the fix: 0 of 20 in 5 repeated
  runs, and the stress repro 0 of 40 (was 17 of 40).
* Full suite on main 28d4fe9 plus this fix, GPU hidden: 1000 passed, 235 skipped in 13 min 33 s; tests/test_dlpack.py with the GPU visible: 20 passed. The full suite was not run with the GPU visible.

### 2026-11-24: keyed random ops and training mode (P20, docs/RANDOM.md)

Branch `wt/rng`. Built per GAP_PYTORCH_JAX items 1 and 2 (R1, R2). torch 2.11, RTX 5070 Ti, CUDA 12.8,
Windows.

* **Generator (`truestar.rng`):** every value is a pure function of (key, site, element index) via
  the SplitMix64 mixer `multinomial` already used (its `uniform01` is a special case, test). uniform,
  normal, bernoulli and randint are defined with integer ops and IEEE exact float ops in a fixed
  order (Acklam in float64, PPND7 in float32 arithmetic, a fixed atanh series for log). Four
  implementations (Python definition, torch ops, numba, Triton) are bitwise equal: 0 of 1e8 draws
  differ between Triton and numba for each of uniform f32, normal f32 and f64, bernoulli and randint,
  0 of 1e7 against the reference path. PyTorch after `manual_seed(0)`, CPU against CUDA: 1,000,000 of
  1e6 rand and randn draws differ, 179,563 dropout mask entries at p = 0.1.
* **Found on the way, each pinned by a test:** torch's CPU float64 `sqrt` is 1 ulp off on some
  inputs (the torch path uses numpy's); Acklam's central rational is up to 3000 ulps off in float32
  (PPND7 replaces it there); Triton compiles `-z` as `0 - z`, which turned 2 of 1e8 normals from -0.0
  into +0.0 (fixed, the test fails on the old kernel); Triton passes Python floats as float32 and
  its float32 `/` and `sqrt` are approximate (constants come from a table, `div_rn`, `sqrt_rn`, no
  fused multiply add).
* **Statistics (1e6 draws):** mean, variance, skewness, kurtosis, P(|z| > 4) within 5 sigma; KS under
  2.1/sqrt(n); randint chi square under the p = 1e-4 bound; correlations across keys, sites, streams
  and adjacent counters under 5/sqrt(n); every output bit balanced. Accuracy: float64 normal within
  1.13e-9 relative of `torch.special.ndtri`, float32 within 5.6 ulps.
* **Speed, CUDA, 1e8 elements:** uniform 0.89x torch's time, bernoulli 0.27x, normal float32 1.90x,
  normal float64 5.1x, eager dropout 1.73x (three passes; compiled programs fuse the mask). Normal is
  the open gap: both inverse CDF branches run for every element.
* **IR, front end, compiler:** IR op `random` (no gradient); dropout and training batch norm are
  compositions (`ir/train_ops.py`). Front end: `ops.rand`, `randn`, `normal`, `bernoulli`, `randint`,
  `*_like`, `dropout`, `batch_norm`; key from the function seed and call count, site from the source,
  inlined calls, unrolled loops and loop iteration streams kept distinct, duplicate sites and
  `torch.rand` refused with a fix. Usage: 0 extra lines and 0 key variables against JAX's split and
  thread (2 lines, 3 keys in the doc's example). Compiled executor: `random` runs through Triton or
  numba, checked bit for bit on the first run, mask fused into the next kernel.
* **Training import:** `import_module(..., training=True)`: mutated buffers become state inputs and
  outputs (`Program.state`) in both graph forms, dropout a keyed draw. resnet18, 20 SGD steps on the
  float64 reference with IR autodiff against PyTorch float64 autograd: 1.0e-12 worst per step error
  (each step started from PyTorch's state), both graph forms. Free running, the IR and a control
  (PyTorch against PyTorch with weights moved by 1e-15) both drift to about 3 by step 20: training at
  lr 0.05 is chaotic, so that is not a bar any implementation meets.
* **Checkers (`--train`):** keyed dropout and attention dropout masks identical on every backend and
  in replays, per call state restored before an isolated replay and compared after it, other random
  ops refused. Injected 10% faults that only the training path runs: batch norm training kernel
  output on CPU (op named) and CUDA, running statistic update only on CPU and CUDA, batch norm
  training backward on CPU (op named) and CUDA. Each is DIVERGENT or TS180 at the batch norm in
  training mode and OK with 0 hits in eval mode. Clean in training mode on CPU and CUDA: conv batch
  norm dropout model, TransformerEncoder with dropout, MultiheadAttention with SDPA dropout, resnet18
  at batch 4 and 16 (check and grad_check, TF32 on).
* **No false alarms (#2):** resnet18 grad_check in training mode first flagged parents (TS180, 1.9e-2
  vs 1e-4 at batch 16, on CPU too). Plain PyTorch float32 against float64 shows the same 1.1e-2 in
  training mode (6.8e-7 in eval): batch statistics couple every element. Training mode parents now keep
  their measured floors uncapped; leaves keep the cap and still catch the injected faults.
* **Tests can fail:** reverting the CPU sqrt fix failed 4 rng tests; the old Triton negation fails
  the signed zero test; permanent tests make keys depend on the dtype (the clean dropout model then
  reports DIVERGENT at the dropout) and skip the state restore (the clean batch norm model then
  reports DIVERGENT).
* Full suite after the change, GPU visible: 1083 passed, 3 skipped in 11 min 5 s (1011 passed before).

Not verified: MPS, ROCm and other GPU models (the same tests run there; not run here), a pretrained
checkpoint, randint above 2**31, channel dropout keying (refused instead), an eager fused dropout kernel.
### 2026-11-24: integer INT8 reference with rounding conventions (`truestar intref`, P29)

Branch `wt/int8ref`. Research gap items 2 and 13 (docs/INTREF.md). CPU only, no device jobs, no
cloud accounts; the NPU data is the AI Hub job cache from the docs/NPU.md study.

* **Exact integer reference.** `truestar.intref` computes conv, fully connected, matmul, add, mul,
  avg and max pool, softmax, requantize and quantize with int64 exact accumulators (float64 matmul,
  2^53 bound asserted), int8 and int16 activations, per tensor and per channel scales, under eight
  documented conventions built from rules transcribed from the C sources (TFLite double and single
  rounding and its 16 bit int64 path, gemmlowp, CMSIS-NN both builds, ONNX QLinear float32 half to
  even, exact tie rules) plus one labelled hypothesis. Found while transcribing, and tested:
  gemmlowp's SaturatingRoundingDoublingHighMul rounds negative ties toward zero, so it equals
  CMSIS-NN's no saturation version on every input; the default TFLite and CMSIS-NN builds agree,
  and CMSIS-NN#214 is single against double rounding.
* **Distinguished on cited cases** (`truestar intref cases`, 0.6 s). CMSIS-NN#214: the per op verdict
  recovers FULLY_CONNECTED as single rounding and MUL as double rounding and says no single
  CMSIS-NN build matches both. Convention pairs separated of 28: conv 0 on random inputs and 15 on
  distinguishing inputs; mul and requantize 15; add 19; with power of two scale ratios 23, the
  maximum (four classes: double, single half up, half even, half away). Tie rules cannot be
  separated with ordinary scales and the tool says so (TS542). Float fake quantization equals
  tflite_ref on 99.16% of every int8 MUL pair (97.22% with power of two scales) and cannot name a
  convention. ONNX Runtime 1.20.1 as a real device: QLinearConv and QuantizeLinear match
  onnx_qlinear bit for bit, QLinearMul matches the fixed point single rounding class, QLinearAdd
  matches nothing modelled (99.81% best, TS541, reported).
* **NPU.md 1 LSB differences explained** (`truestar intref npu`, 10.7 s, 0 jobs; `--summary` 0.25 s).
  At the relu tap 186,314 of 4,194,304 elements differ (4.44%), 104,655 up and 81,659 down, rate
  rising from 7.6% to 17.6% with the output level. All 64 channels differ both ways (minority 3.39%
  of elements); documented rule pairs stay under 0.041% (measured, bound 0.5%), so it is not a
  rounding convention (TS543). The NPU's max pool is exact on its own relu output (100%). Simulated
  on conv1, float16 input quantization gives 3.86% to 4.34% over the unknown input scale range
  (mean bin gap 0.009); every documented convention gives 0% to 0.13% (gap 0.111 or more). This is
  an inference: the quantized model is not on this machine, so no per element match.
* **MCU check path** (`truestar intref golden`, 0.26 s). C99 header, `<stdint.h>` only: per layer
  inputs, expected outputs, tolerance in LSB (exact, or portable: the worst difference to any other
  documented convention, 1 LSB on conv and avg pool for cmsis_nn), FNV-1a checksums,
  `ts_golden_check`. No C compiler exists on this PC (none installed), so it is validated by parsing
  it back: lengths, ranges, checksums, round trip; a one value edit is caught.
* **No false alarms (#2):** every documented convention against `onnx_qlinear` on a synthetic conv
  passes the rounding only test; max pool, where conventions cannot differ, reports TS542, never a
  convention. **No silent wrong results (#3):** primitives checked against scalar transcriptions on
  random and edge inputs, conv accumulators against float64 torch.
* **Tests can fail:** RoundingDivideByPOT without its negative threshold fails 2 tests; a rounding
  only test that always says yes fails 3; skipping the header checksum fails 1; QuantizeLinear
  rounding half away fails 1 (all restored).
* Tests: tests/test_intref.py, 15. Full suite: 846 passed, 233 skipped in 14 min 55 s with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`, skip reasons not listed).

Not verified or not done: TensorFlow, TFLM, the Corstone FVP and real boards (not installed or not
available); the header compiled by a C compiler; the NPU hypothesis per element (needs the quantized
model, or an NPU run with integer input, which would also test the remedy); fixed point int8
softmax tables; depthwise multiplier layout, int4 weights; Hexagon requantization (no public
description found; web search was limited this session).
### 2026-11-24: simulated device numerics, size sweep, special value probe (P28)

Branch `wt/devsim`. Gap research items 1, 5 and 9 (devices and frameworks). Docs: docs/DEVICES.md
(new). Codes TS520 to TS525. Everything below ran on the CPU; the probe also ran on the RTX 5070 Ti.

* **Simulated backends (`--backends sim:*`).** Eight profiles: `sim:ane`, `sim:webgpu-f16`,
  `sim:mali-fp16`, `sim:neuron-bf16`, `sim:openvino-gpu-f16`, `sim:openvino-npu-f16`,
  `sim:ieee-f16`, `sim:ieee-bf16`. A profile is data: storage dtype, per op precision, wide,
  tiled or float16 accumulation, flush to zero, saturation, NaN to +inf, step by step composites
  and op formulations; each of its 33 rules is labelled documented, measured, hypothesis or guess
  with its source (`truestar backends --sims`). A dispatch mode evaluates every ATen op in float64
  under the profile. Redesign against copying a vendor emulator: one profile per device in one
  report, and each simulated backend is calibrated on its own *ideal* (same precision, correct
  arithmetic), so it is blamed only for what the device does beyond correct low precision math.
  The same profiles round Truestar IR programs (`emulate_precision(profile=...)`).
* **Cases reproduced, 6 of 6 readable:** coremltools#2687 (ANE softplus: x = 10.4077 gives 0,
  10.394 gives 10.39, 10.395 gives 0, as in the issue), coremltools#2690 (logsumexp over 32: cliff
  at 7.6313, reference 11.0971, output 0), openvino#39703 (NPU ReduceL2: 255.875, rel err 0.360
  above 256, 6.8e-4 below against the issue's 5.9e-4, workaround 400), openvino#22613 (GPU f16
  gemm: 2048 at n = 2049), onnxsim#1742 (decomposed LayerNorm overflow, mechanism only; the 19%
  accuracy needs pretrained weights), arXiv 2606.22283 section 3.6 (ANE NaN to +inf, sigmoid 1,
  softmax mass on the NaN lane). The NPU numbers changed the profile: a float16 accumulator
  advancing one element at a time gave 4.6e-3 below the cliff, ten times the measured 5.9e-4, so
  the NPU profile uses a wide accumulator that saturates on its float16 result.
  onnxruntime#28976 and #29130 return 404 from this account and are not counted.
* **False positives:** 0 device specific findings on MLP, a 2 layer transformer encoder, ResNet18,
  ViT B/16 and the three test models, on every profile. Random init MobileNetV3 small is flagged
  by all six float16 profiles, `sim:ieee-f16` included: its activations reach 1e-7 to 1e-10, below
  the float16 subnormal range; the report now says when a module fails under correct arithmetic
  in the same precision ("not specific to the device"). bfloat16 profiles are clean on it.
* **Runtime:** wide accumulator profiles cost 5 to 10 times the plain CPU check (ResNet18 0.9 to
  2.2 s against 0.19 s); float16 accumulator profiles 25 to 100 times (ResNet18 6 to 8.5 s,
  MobileNetV3 on webgpu-f16 25.8 s, ViT B/16 on webgpu-f16 125 s with other jobs on the machine).
  Opt in only; the default check is unchanged.
* **`--sweep NAME:LO..HI`:** ladder of LO, HI and every `2^k`, `2^k + 1` between, bisection
  between the last OK and first failing ladder size, then one full check there naming module, op
  and repro. On the openvino#22613 reproducer (examples/size_cliff.py), 512 to 8192: cpu and
  `sim:ieee-f16` clean, `sim:openvino-gpu-f16` and `sim:mali-fp16` clean up to 2058 and SUSPECT
  from 2059 (aten.bmm.default), 33 checks in 5.6 s. Not 2049: at 2049 the error (4.9e-4) is inside
  the float16 floor and correct float16 gives 2048 too, which the maintainer called expected; 2059
  is where the stuck accumulator leaves correct float16.
* **`truestar probe --specials`:** 28 cases, each element compared with float64 and with IEEE in
  the backend's own precision; implementation defined cases listed, not judged. On this machine
  cpu and cuda have 0 IEEE departures but disagree with each other: NaN to int32 is -2147483648
  on the CPU and 0 on CUDA, +inf and 3e9 are -2147483648 and 2147483647, relu(-0) is -0 and +0.
  `sim:ane` 24 departures (NaN to +inf), `sim:webgpu-f16` 5 (flush to zero), 10 backends in 2 to 3 s.
* **No silent fallback (#1):** `sim:*` is never added by default; `truestar invariance`, `check
  --backward`, `record` and `fix` refuse a simulated backend (TS001) instead of running the CPU
  under its name; every simulated result carries TS520, and ops whose accumulator is not modelled
  are named (TS525). **No new wrong results (#3):** an earlier version left the first float16
  accumulation step unrounded (an overflowing dot product gave 0 instead of inf); the probe found
  it and a test now covers it.
* **Tests can fail:** with the NaN rule and the float16 accumulator rounding switched off, 5 of 25
  tests in tests/test_devsim.py fail; the ideal of `sim:ane` (no softplus rule) does not blame the
  softplus (asserted in a test).
* Full suite in the worktree with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`): 860 passed, 233 skipped in 13 min 29 s (skip reasons not listed in this run). The probe and example CUDA rows above ran with the GPU visible.

Not verified: any real device (no Mac, no WebGPU runtime, no Mali phone, no Neuron instance, no
Intel NPU); the guesses listed per profile; index math in float16 (onnxruntime#28976 unreadable);
autograd under simulation; scalar arguments rounded to storage; transposed and 3d convolutions,
pools other than global average and fused transformer fast paths on a float16 accumulator.
### 2026-11-24: quantized kernels with two verdicts (`truestar.quant`, P25)

Branch `wt/quant`. Research items GAP_COMPETITORS 5 and PAIN_RESEARCH_3 R4. torch 2.11, Triton
3.7.1, RTX 5070 Ti (sm_120) shared with other jobs. Docs: docs/QUANT.md. Codes TS460 to TS464.

* **Formats as data.** INT8, INT4 (groups 32, 64, 128), FP8 E4M3 and E5M2 (channel or tensor
  scales), NVFP4 (E2M1, E4M3 block 16, float32 tensor scale) and MXFP4 (E8M0 block 32) are value
  tables in float64 with round to nearest even and saturation; all 256 FP8 codes and every tie
  equal torch's float8 casts.
* **Native on this GPU** (`scripts/probe_quant_triton.py`, PTX inspected): `tl.dot` on E4M3 and
  E5M2 (`mma ... e4m3` / `e5m2`), and `tl.dot_scaled` for both MXFP4 and NVFP4
  (`mma ... kind::mxf4nvf4.block_scale`, ue8m0 and ue4m3 scales). Nothing emulated.
* **Verdict (a), kernel bug:** each kernel against the exact float64 product of the operands it
  was given (dequantized weights, the activation codes and scales its quantizer wrote), per
  element bound `u_out |ref| + 16 sqrt(K) 2^-24 S`; correct kernels at most 0.92 of it (the output
  rounding), the accumulation part under 0.036. Activation quantizers are exact to the format's
  rule (round to odd before the format's rounding; with a plain division 35 of 262144 E4M3 codes
  differed). A failing kernel is diagnosed from its effective weight (unit vector inputs): which
  group's scale it read, the granularity at which the scale changes, or the element order.
* **Injected bugs** (11, `kernels.BUGS`): neighbouring channel, group and block scales, the last
  group reusing the previous scale, groups of 64 assumed for 128, swapped nibbles in INT4 and FP4,
  tensor scale for channel scales, float16 accumulation, NVFP4 scales read as block 32. All are
  TS460 at 459x to 6190x the bound (the float16 accumulator overflows to infinity) and named;
  format loss is identical with and without them.
* **Verdict (b), format loss**, trained Llama style model built in code (28 quantized layers,
  2048 positions): delta CE from +0.0001 (INT8) to -0.0064 (MXFP4, noise of 2048 positions), KL
  0.0000 to 0.0073, flips 0 (INT8), 3 to 8 (FP8), 13 to 14 (INT4), 20 (NVFP4), 25 (MXFP4).
* **Baseline cannot separate (acceptance):** with a last group scale bug in one INT4 layer, the
  quantized model against the float16 model moves from 0.0859 to 0.0861 relative and 13 to 14
  flips, inside INT4's own loss; a next group scale bug moves it to 0.0854 and 10 flips (fewer
  than without the bug). Verdict (a) names the layer in all four injected model runs.
* **Speed** (CUDA graph replay, median of 5): linear layers of a 7B class block at batch 1,
  cuBLAS 508 us, Truestar float16 409 us, INT8 160 us, INT4 118 us (3.5x Truestar float16, target
  1.8x), FP8 153 us, NVFP4 125 us, MXFP4 105 us, `torch._scaled_mm` FP8 252 us. At M 2048:
  FP8 1.1x to 2.1x, NVFP4 2.3x to 4.1x, MXFP4 2.3x to 3.8x over cuBLAS; weight only INT8 and INT4
  0.5x to 1.1x (they are decode formats). The magic number INT4 unpack made INT4 1.85x faster at
  M 2048 than integer conversions.
* **Found while building (never introduce #2 and #3):** two false alarms of the reference itself,
  fixed before anything shipped: PyTorch on CUDA divides by a scalar through the rounded
  reciprocal (37 of 64 FP8 row scales one ulp off), and `torch.pow(2.0, e)` and `torch.ldexp` on
  CUDA are inexact for 35 of 254 E8M0 exponents. The reference now uses correctly rounded
  division and powers of two from exponent bits.
* **Tests can fail:** tests/test_quant.py has 65 tests. Manual breaks, restored: the bound's
  lambda at 1e9 fails 13; a plain division in the FP8 quantizer fails 7; ties rounded down in the
  codec fails 18.
* Full suite after the change: 1266 passed, 4 skipped, 1 failed in 17 min 34 s with the GPU shared.
  The failure, `test_dlpack.py::test_cuda_consumer_stream_waits_for_the_producer`, is in code this
  branch did not touch (a stream ordering test sensitive to other GPU work); tests/test_dlpack.py
  alone passed 18 of 18 three times right after.

Not verified or not done: a CLI flag (`truestar check --quant`), torchao (not installed),
row wise `torch._scaled_mm` (refused on this GPU by torch 2.11), a published NVFP4 checkpoint
decoded byte for byte (no downloads), other GPUs and the Hopper FP8 accumulator, end to end decode
with CUDA graphs through a whole model, pretrained models, static activation scales, INT4 zero
points and GPTQ or AWQ calibration.
### 2026-11-24: vmap and forward mode as IR passes (P24)

Built `truestar.ir.vmap` and `truestar.ir.forward` (`jvp`, `jacfwd`, `jacrev`, `hessian`,
`linearize`), docs/TRANSFORMS.md. Program to program: no batched tensor type, no dual numbers, one
program for every batch size. Measured on this machine (CPU, torch 2.11, float64 unless stated):

* **vmap equals the per element loop.** 301 cases: the 89 reverse mode corpus programs with all
  inputs batched, only the first batched and the batch on the last axis, plus 34 cases for the rest
  (batched indices, per element weights, cond with a per element predicate, while loops with per
  element trip counts, custom_vjp, sampling). Worst 3.4e-16, bar 1e-12. Every node's batched type is
  checked as it is emitted; that check found a wrong attention rule during development.
* **Cases torch.func.vmap refuses** (`scripts/transforms_compare.py`): `if` and `while` on data,
  random ops without a flag, `torch._foreach_add` (pytorch#200456). Truestar runs the first two
  exactly (random: element k's draws are the op's draws at offset plus k times 2**40, reproducible one
  at a time) and has no foreach op. A batched `cond` skips a branch no element takes, so the jax#2947
  program (a loop that only ends for its own branch) terminates when the batch agrees; JAX always
  runs both branches.
* **Speed, best of 21 on a shared machine:** per sample gradients of an MLP at batch 64: Truestar
  vmap 0.52 to 0.70 ms against a Truestar loop of 6.3 to 8.5 ms; torch.func.vmap 0.59 to 1.59 ms
  against a torch loop of 6.4 to 6.7 ms. Branchy function at batch 256: 1.7 to 1.9 ms against a 33 to
  35 ms loop; torch.func refuses it. The vmapped program compiles unchanged
  (`CompiledExecutor("cpu", "float32")`, 1.2 ms, 1.2e-7 from the reference).
* **Forward mode.** Rules written independently of the reverse rules: finite differences on 102
  programs (corpus worst 6.6e-11; overall worst 4.5e-9, an RNN loop where the stencil itself loses
  accuracy, and that loop equals `torch.func.jvp` exactly); dot product test `<J t, c> = <t, J^T c>`
  on all 89 corpus programs, every one of the 45 ops with both rules, worst 1.3e-14, bar 1e-12. Loops
  carry tangents as extra carried values: nothing stored per iteration.
* **Jacobian family.** `jacfwd`, `jacrev` and `hessian` match torch.func to 1e-12; `jacfwd` with a
  symbolic batch serves two sizes from one program. Hessian of a TransformerEncoderLayer: 1.0e-15
  against `torch.func.hessian`, which itself raises `NotImplementedError` on that stock layer on the
  CPU (no forward AD for the CPU flash attention kernel; `jacrev(jacrev)` raises too) until the math
  attention backend is selected by hand. MLP 16, 32, 16 at input [8, 16]: `jacfwd` 0.17 to 0.33 ms
  against 0.88 to 1.44 ms, `hessian` 0.57 to 0.78 ms against 2.5 to 3.3 ms (call overhead dominates
  at this size; larger models not timed). `linearize` gives a linear program with no primal node,
  equal to `jvp` to 1e-14.
* **Tests can fail:** a softmax batching rule that does not move its dim, a conv rule that mixes
  elements and a mistyped sum rule are caught; a forward tanh rule with `1 - y` and a softmax rule
  missing the `sum(t y)` term fail both finite differences and the dot product test.
* Tests: tests/test_vmap.py 318, tests/test_forward.py 115. The first full suite run found that
  importing `truestar.train` registers `log_softmax` and `all_reduce`, which the classification test
  refused as unclassified; both now have vmap and forward rules.

Not verified or not done: forward rules for `fmod`, `remainder`, `vector_norm`, `group_norm`,
`rms_norm`, `adaptive_max_pool2d`; sharing the primal between the vmaps of `jacfwd` (no common
subexpression pass); timing on larger models and on the GPU; JAX not installed (its behavior is
quoted from its docs and issues); the keyed `random` op of the rng branch is not on main, so its
vmap rule is not written (it would be refused by name, never batched with one shared stream). Found
on the way, not fixed: the reference `cumsum` of a rank 0 tensor raises `ZeroDivisionError`.

### 2026-11-24: compilers and runtimes as check backends (P22)

Branch wt/backends. Research items: gap analysis item 1 and pain research round 3 R2, both with
their redesign sections. Docs: docs/BACKENDS.md (new). torch 2.11; Inductor with Triton 3.7.1 (ml
venv) and 3.6.0 (cv venv); ONNX Runtime 1.20.1 CPU (ml) and 1.29.0 GPU (cv); TensorRT 10.16.1; RTX
5070 Ti shared with other jobs. Rows served: P22 (new), P4, never introduce #1 and #2.

* **Backends.** `cuda+inductor`, `cpu+inductor`, `ort:cpu`, `ort:cuda`, `trt:fp32`, `trt:tf32`,
  `trt:fp16` in the existing `--backends` list, next to cpu and cuda. A stack that cannot run is
  refused with the reason (TS001), never skipped.
* **Design choice (do not change the engine being checked).** The verdict comes from the whole
  model compiled as users compile it, graph outputs only. Localization runs only after that engine
  mismatches, in separate per module engines fed float64 reference inputs, by descent over the
  module tree (children in order, first one over threshold, repeat). Polygraphy instead marks
  every tensor as an output, which stops TensorRT fusing through them. Op drilldown compiles each
  ATen op of the blamed module alone; the same op run eagerly in the stack's precision says
  whether the stack, the precision or the device kernel is at fault. Inductor: the named op is
  traced to the fused kernel computing it from Inductor's ATen origins. ORT CUDA: one profiled run
  reads each node's provider; floating point nodes on the CPU are TS400 with node and module, a
  CUDA provider that did not load is TS401. TS402 when only the modules compiled together diverge,
  TS403 for graph breaks.
* **Faults caught, one run each** (scripts/stack_study.py faults): Inductor sigmoid lowering x1.05
  at blocks.0#0, `aten.sigmoid.default` in `triton_poi_fused_add_addmm_mul_sigmoid_1`, 7.2 s, 15
  compiles; exporter gelu translated as the tanh approximation on ort:cpu at act1#0,
  `aten.gelu.default`, 4.8 s, 7 compiles; Det on ort:cuda reported as CPU fallback in module det,
  1.0 s, 1 compile; LayerNorm forced to FP16 in a trt:fp32 engine at norm#0,
  `aten.native_layer_norm.default`, 16.4 s, 5 compiles; hand written RMSNorm on trt:fp16 at norm#0,
  `aten.pow.Tensor_Scalar` with Inf, blamed on float16 range, 11.5 s, 4 compiles. 5 of 5 at the
  right module and op. The Inductor and ORT faults are small and land as suspect (TS011).
* **False alarms and overhead** (scripts/stack_study.py clean): resnet18 and a
  TransformerEncoderLayer, random weights, on all five stacks: 10 of 10 ok. Overhead over the
  stack alone, warm: resnet18 +0.8 s to +2.4 s, encoder layer under +0.4 s, one encoder trt:fp16
  run +9.3 s from TensorRT build variance (rerun +0.3 s).
* **Found on the way and fixed before it shipped.** A first version calibrated trt:fp16 with a CPU
  float16 run of the whole model; CPU float16 overflowed the same way, so a model with output error
  1.0 and 59 of 64 flipped predictions reported OK. A CPU calibration above the lower precision's
  own floor is now not trusted. Also: TensorRT cannot return LayerNormalization's mean and rstd
  (ops are retried returning the first output); TensorRT ran the planned overflow model
  (Linear weights x2000 into LayerNorm) without overflow, so it is not used as a fault; TensorRT
  without a shared builder and timing cache took about 15 s per small engine, under 1 s with them.
* **Tests can fail:** tests/test_stacks.py, 13 tests. With the descent disabled and the ORT
  placement check disabled, 5 of 12 that run in the cv venv fail; restored.
* Tests in the cv venv (all stacks): tests/test_stacks.py 12 passed, 1 skipped (the TensorRT
  missing case). Full suite in the ml venv: 860 passed, 233 skipped (GPU visible; skips include
  the TensorRT and ORT CUDA stack tests, which ran in the cv venv) in 13 min 29 s.

Not verified: `cpu+inductor` (no C++ compiler on this machine), the ten pt2 issue repros in pinned
torch versions from the R2 acceptance test, the Inductor accuracy minifier and Polygraphy on the
same faults (the faults are injected in process and both tools rebuild from files in their own
process), Inductor modes other than the default, ORT's TensorRT provider, ExecuTorch, ROCm, an
idle GPU for timings.
quoted from its docs and issues). (The keyed `random` op was not on main when this entry was
written; its rule is the next entry.) Found on the way, not fixed: the reference `cumsum` of a rank 0 tensor raises `ZeroDivisionError`.

### 2026-11-24: vmap of the keyed random op (P24, after the rng merge)

* Merging main brought the keyed `random` op (docs/RANDOM.md), and `test_every_ir_op_has_a_batching_rule`
  refused it as unclassified, as designed. Rule: element k draws exactly the op with k appended to its
  streams (`fold(K, k)`, the scheme's own rule for a loop index), so vmap of a random function equals a
  loop over the batch. Vectorized: a new op `random_batched` (row k is `random(key, *streams, k)`) with
  `truestar.rng.generate_batch`, one `mix` over `arange` for the row keys, no Python loop; a new op name
  so that a consumer that does not know it fails instead of drawing one shared stream. The compiler's
  random step runs it on the CPU and CUDA. A batched key or stream and vmap of vmap use the exact per
  element loop with the same rule.
* Verified for uniform, normal (float32 and float64), bernoulli and randint: each row bitwise equal to the
  op alone with the extra stream and to the python definition; rows independent (correlation below 0.02
  over 50000 normals, expected 0.0045); `generate_batch` equals every CPU backend row by row; CPU and CUDA
  reference runs, the Triton kernel row by row and the compiled executor on CPU and CUDA give the same
  bits (RTX 5070 Ti). A native function with `ops.dropout` and `ops.normal` vmaps to two
  `random_batched` nodes with the key unbatched.
* Tests: tests/test_vmap.py 337 (13 new, 5 of them on CUDA), tests/test_forward.py 117. The merge also
  brought 2 new reverse mode corpus programs: vmap now checks 307 cases against the loop (worst 3.4e-16)
  and the dot product test 91 programs (worst 1.3e-14).
### 2026-11-24: WebGPU as a local check backend (`truestar check --backends webgpu`)

Branch `wt/webgpu`. Gap research item 4 with items 6 (residency) and 8 (observer guard). Docs:
docs/WEBGPU.md. Codes TS560 to TS566. Machine: Windows 11, RTX 5070 Ti and AMD Radeon integrated
GPU, onnxruntime 1.29.0, onnxruntime-ep-webgpu 0.6.0, wgpudawn 0.33.0.0, torch 2.11 (cv venv).

* **Backend (P4).** `webgpu`, `webgpu:d3d12`, `webgpu:vulkan`, `webgpu:<backend>:<adapter>`, EP
  options after `+`. The model is exported to ONNX and run in a worker process per backend (the EP
  owns one Dawn device per process; a driver crash ends the worker, not the check). Same report,
  agent JSON and repro contract as the torch backends; repros need only numpy and onnxruntime.
* **No silent fallback (#1).** ONNX Runtime falls back to its CPU EP without raising when the
  WebGPU EP cannot be created (bad option, no adapter, or both listed devices passed). Truestar
  checks the providers of every session (TS560, backend refused at resolve time) and the EP of
  every node from a profiled run (TS561 names compute nodes on the CPU). Found on the way: the EP's
  Dawn sees no Vulkan adapter unless `vulkan-1.dll` is preloaded; the worker preloads it.
* **Observer guard (item 8).** The profiled and the tapped runs must equal the plain run bit for
  bit, else per module data is untrusted (TS564) and every module is replayed alone. Taps changed
  the output by 1.1e-7 to 2.1e-6 on ResNet18, MobileNetV3, EfficientNet B0 and GPT2 (fusions), never
  on BERT or a TransformerEncoder.
* **No false alarms (#2).** Clean on 22 model and adapter combinations (docs/WEBGPU.md table):
  ResNet18, MobileNetV3, EfficientNet B0, ViT B/16, TransformerEncoder, BERT and GPT2, float32 and
  float16, on the RTX (D3D12 and Vulkan), the Radeon and the CPU rasterizer; float32 errors 1e-7 to
  1.1e-6, float16 1.9e-3 to 4.3e-3. Two false alarms of our own were found and fixed first: the
  exporter put batch norm back in training mode (ResNet18), and in place activations overwrote the
  feeds of isolated replays (EfficientNet SiLU 67%, MobileNetV3 Hardswish 75%).
* **Real finding 1 (P1, P2):** the EP's default float16 MatMul keeps a float16 running sum for
  some shapes (K above 8192 with 1 to 8 rows; 64 rows from K 1024 on D3D12): up to 50% relative
  error on positive inputs, both GPUs, both Dawn backends with different cutoffs.
  `enableMatmulFp32Accumulation=1` brings every shape to about 2e-4. Truestar blames the Linear and
  names `onnx::MatMul` from the model alone (9.3e-3 against 5e-3 in a float16 feed forward block;
  CUDA 4.8e-4), the op repro exits 1 on WebGPU and 0 on the CPU EP, and the agent JSON offers the
  EP option rerun, which is clean (3.2e-4). Minimal repro: scripts/webgpu_f16_matmul_repro.py.
* **Real finding 2 (Dawn direct probes, `python -m truestar.webgpu.dawn`):** the same WGSL matmul
  is contracted to a fused multiply add on Vulkan (100% of elements match an FMA emulation) and not
  on D3D12 (100% match separate rounding) on the same RTX; the Radeon converts float32 to float16
  toward zero (100% match), the RTX to nearest. Both are permitted by WGSL and reported as explained,
  not blamed; the truncation doubles the Radeon's float16 MatMul error through ONNX Runtime.
* **Efficiency.** The CPU EP control runs only when WebGPU is off (float16 encoder check 49.8 s to
  6.6 s); with trusted taps a clean model needs no extra exports. Check times 3.4 s (MLP) to 37 s
  (MobileNetV3 on two adapters), worker start included.
* **Tests:** tests/test_webgpu.py, 15 tests. 10 run in any environment with onnxruntime (a fake
  worker plays a WebGPU device on the CPU EP with an injected faulty kernel, CPU placed nodes, a
  fallen back session, taps that shift the output); 5 need a WebGPU adapter and skip with the
  reason elsewhere (they skip in the ml venv: onnxruntime_ep_webgpu not importable). Tests can fail:
  disabling the observer guard and the session fallback check failed 2 tests; dropping the mode
  copy and the feed copy in the exporter failed 1.
* Full suite in the ml venv (no WebGPU, GPU visible, 15 agents sharing it): 1211 passed, 9 skipped
  (the 5 real WebGPU tests among them), 1 failed in 15 min 8 s: `test_cuda_consumer_stream_waits_for_the_producer`
  (tests/test_dlpack.py, untouched here) read a sum before the producer stream finished; it passed
  3 of 3 times when rerun alone. tests/test_webgpu.py in the cv venv: 15 passed in 81 s.

Not verified: Metal, Intel GPUs, Linux, a browser; onnxruntime#28976 and #29651 were not
rediscovered (float16 ResNet18 at 224 px is clean; inputs above 2048 px not run); the
`sim:webgpu-f16` profile (devsim branch) is compared only on paper: real reductions, softmax and
layer norm do not accumulate in float16 here, MatMul does for the shapes above.
### 2026-11-24: static memory planner and verified recomputation (P23)

Branch `wt/memplan`. Research items R5 and R6 (GAP_PYTORCH_JAX) and item 7 (GAP_COMPETITORS).
Docs: docs/MEMORY.md (new). RTX 5070 Ti, torch 2.11, CUDA 12.8, float32, GPU shared with other jobs.

* **Planner.** `cp.memory_plan({"B": 64})` plans one call from a meta tensor dry run and the IR
  types, nothing launched, in 0.5 ms to 5 ms. The planned bytes equal torch's measured
  `requested_bytes` exactly on the MLP, TransformerEncoderLayer and causal attention at batch 1 to
  256, on a 3 layer language model's training program and on an SGD training step;
  `max_memory_allocated` is within 0.19 MiB above (unsplit cached blocks). Not exact on resnet18:
  cuDNN's convolution workspace is outside the plan (42 MiB at batch 1, 99 MiB at batch 64), and the
  plan says so in its notes.
* **Reuse.** Every value is dropped after its last reader; elementwise kernels write over a dead
  input behind a reference count guard; donated inputs are taken. Peak against the executor before
  this work: 33% less (MLP), 63% (transformer layer), 33% (attention), 73% measured on resnet18 at
  batch 64, 76% on the 6 layer training program (1056 to 256 MiB). Against PyTorch eager: level on
  MLP and transformer layer, 20% less on attention, 12% less on resnet18 at batch 64. Outputs are
  bitwise equal to the executor without reuse.
* **Donation.** An SGD with momentum step (params and momentum 64.04 MiB): 64.04 MiB transient
  without donation, 32.75 MiB with `donate=True` (planned equal to measured), bitwise equal; no
  second copy of params and state. A tensor the caller still holds is never written over (tested).
* **Budget (TS420).** `memory_budget=` refuses a program before any launch, naming the step, op and
  source line at the peak, the values live there and the largest batch that fits (binary search
  over the planner). Unbounded symbols are checked at each new input shape before its first launch.
* **Recompute (TS421).** A pass over the compiled steps recomputes values (with producer chains through
  views, like checkpointed segments) until the plan fits. 6 layer LM, batch 16, seq 128: budgets of
  75%, 50% and 35% of the 256 MiB peak all met, planned equal to measured (185.6, 127.5, 89.5 MiB),
  gradients bit identical to the run without recompute, every recomputed value compared bitwise on the
  first call, 1% to 15% more step time. torch.utils.checkpoint per layer: 181 to 63.6 MiB for 81%
  more step time.
* **Losses.** Truestar's compiled training step is 2.3x slower than eager here and its peak without
  recompute is above eager's (256 against 181 MiB: the autodiff pass stores both sides of each relu, the
  attention probabilities and the layer norm's centred input). Recompute stops at 89.5 MiB, above
  checkpointing's 63.6. Recompute compile time 5.4 s to 8.8 s. No static arena, so fragmentation is not
  addressed.
* **Tests.** tests/test_memory.py, 24 tests. Full suite with the GPU visible, after merging main (7deab07): 2129 passed, 2 failed, 4 skipped (2 CPU loop nest cases with no GPU loop shape, 1 slow scalar CPU path, 1 float16 case below tolerance on the CPU reference too) in 20 min. The 2 failures also fail on main itself: `test_ir_ops.py::test_op_matches_pytorch_float64[einsum]` (torch.export specializes B to 2) and `test_vmap.py::test_every_ir_op_has_a_batching_rule` (the `random` op has no batching rule). Tests can fail: guard always yes fails 4, planner that
  forgets frees fails 12, recompute bitwise check skipped fails 1. Building them found a real
  disagreement: the static schedule planned a write over an expand view (which does not cover its
  storage); the guard refused it, the plan counted it. The walk now mirrors the guard.

Not verified or not done: cuDNN workspace in the plan; CPU in place reuse (frees only); a static
arena; recompute or donation wired into `truestar.train.Trainer`; offload to the CPU; other GPUs and
Linux. Found and not fixed: a compiled AdamW step fails its kernel check on the base commit too
(`1 - exp(t ln b2)` cancels in float32, relative error 2.3e-5 against 2e-5); the importer cannot
take a `torch.export.Dim` without an upper bound.

### 2026-11-24: sentinels (compiled in runtime checks) and `truestar profile` (P27)

Branch `wt/sentinels`. Research items R13 and R14 (GAP_PYTORCH_JAX.md) and competitor item 14
(GAP_COMPETITORS.md). New row P27; never introduce #2 (no false alarms), #4 (overhead measured)
and #6 (no cryptic errors). Docs: docs/COMPILER.md (Sentinels), docs/PROFILE.md (new),
docs/AGENT_INTERFACE.md (codes TS500 to TS504, TS510 to TS512). RTX 5070 Ti shared with other
jobs, torch 2.11, CUDA 12.8, Triton 3.7.1, Windows.

* **Sentinels (`CompiledExecutor(..., sentinels=True | "all" | "nan,inf,overflow,div")`, off by
  default).** Kernels write int32 flag words into pinned host memory with stores predicated on the
  bad entries, so a clean run writes nothing and the call adds no launch and no copy. Torch run
  steps are checked where kernels load their output, else by one multi tensor norm per call. On a
  hit the call runs again step by step to find the first producer in time (loop iterations
  included), and its fused ops are evaluated in float64 on the inputs it received to name the op,
  its line and the first offending element. `sentinel_read="deferred"` reads the flags at a later
  call. Index bounds are always checked and now name the line and the bad index (TS504).
* **Each class caught on an injected fault** (tests/test_sentinels.py, `scripts/bench_sentinels.py
  --faults`): log of a negative entry (TS500, `log`, index (7, 11)), exp past float32 (TS501),
  a float16 product past 65504 (TS502, finite in float32), a row summing to zero (TS503, the div
  is named rather than the NaN after it), a NaN arising at loop iteration 4 (`iterations` [4]), a
  convolution overflowing to Inf that a following clamp hides (named at `conv2d`), an index 12
  into a table of 10 (TS504 with file, line and position). `torch.autograd.detect_anomaly` on the
  same log fault with a backward pass raised nothing (the gradient is finite), and nothing is
  checked under `torch.no_grad`.
* **No false alarms** with every class armed on the MLP, a `TransformerEncoderLayer` with a float
  `-inf` causal mask in the test, and resnet18. Deliberate infinities are followed through the
  kernel IR (`may_be_inf`): a masked softmax stays armed, and only a kernel whose stored value can
  hold the infinity loses the `inf` class.
* **Overhead** (docs/COMPILER.md has the full table; fastest of 7 interleaved rounds), wall time over
  no checks, nan and div: MLP +18.8% read per call, +2.0% deferred; transformer layer +14.9% and
  +15.1%; resnet18 +4.3% and +2.7%; training steps (forward and every parameter gradient): MLP
  +15.1% read per call, transformer layer +9.8% deferred, resnet18 +9.4% read per call and +0.8%
  deferred. `detect_anomaly`: forward +710% to +1091%, training step +322% to +812%.
  The 3% target is met with the deferred read on the MLP, resnet18 and its training step, not on
  the transformer layer or on small float16 programs. Measured on the way and changed: a block
  reduction per class (float16 contraction kernels several times slower), flags on the device
  (0.055 ms of zeroing and copying per call), a reduction per convolution output (12% of resnet18
  GPU time), a host read per call (0.107 ms on this machine).
* **Profile (`truestar profile model.py [--format agent] [--crosscheck]`).** Each timed call
  carries one CUDA event pair around one step, cycling through the steps, with a sleep kernel
  queued first so the GPU never waits on the host. Step runs timed against expected: 120/120 (MLP),
  840/840 (transformer layer), 1230/1230 (resnet18); sum of steps against the busy call: +0.4%,
  +1.6%, -2.5% at worst over three runs; every step maps to a source line. An event between every
  step was tried first and overstated the transformer layer by 12%. **torch.profiler dropped one
  Truestar kernel event** in one of three resnet18 runs (539 of 540) and once on the transformer
  layer (209 of 210), silently; `--crosscheck` reports it as TS511.
* **Found, not fixed:** the kernel layer's first call check rejects the float32 layer norm
  backward of the transformer layer's gradient program (a sum of terms that cancel: relative error
  0.56 against a near zero reference, tolerance 4e-5), so that training step ran with verification
  off in the benchmark; with checks off, a float32 overflow on a first call is still reported as a
  code generator mismatch (with checks on, the check is named instead).
* Tests: tests/test_sentinels.py 22, tests/test_profile.py 5. Tests can fail: with the flag stores
  emitting nothing, a NaN goes unflagged (test in the file); with a lost end event the profile
  reports TS510 and exits 1.
* Full suite (once, GPU visible, 16 min 52 s): 1225 passed, 4 skipped, 4 failed. Three failures
  were this branch's: a test wraps `CompiledKernel._run_triton` with the old two argument form,
  so the run methods are called that way again when no flag buffer is passed; tests/test_kernel_gpu_verify.py
  then passes (with tests/test_sentinels.py and tests/test_dlpack.py in one run: 54 passed). The
  fourth, `test_dlpack.py::test_cuda_consumer_stream_waits_for_the_producer`, failed under the
  full suite's load and passed 6 of 6 runs alone, with and without this branch; it is a timing
  sensitive two stream test from the interop branch and was not changed.
* After merging main (41adc00) into `wt/sentinels`: the sentinels, profile, codes, region check,
  runtime, compiler and kernel test files 394 passed, 4 skipped; full suite with the GPU visible
  (22 min 25 s) 2158 passed, 8 skipped, 1 failed. The failure,
  `test_ir_ops.py::test_op_matches_pytorch_float64[einsum]` (torch.export reports a violated
  constraint on B), fails the same way on main's own code at 41adc00 and at 22b2504, run alone.
* After merging main at 2d72266 (memory planner, recompute, WebGPU): sentinels and the memory
  plan both live per compiled variant. The flag words are pinned host memory and never enter the
  device plan; a watched torch step output is now reduced at once (one norm) when the planner
  reuses buffers, so the watch keeps no value alive past its last reader. Measured on resnet18 at
  batch 2 with every class: planned peak identical with and without sentinels, in place steps
  identical, measured device peak within 64 KiB (tests/test_sentinels.py). The step loop with
  sentinels or the profiler follows the planner's schedule (frees, kernels writing over a dead
  input, recompute checks); the locate run keeps every value and runs nothing in place. With
  `donate=True` and sentinels on, the call keeps references to the inputs for a locate run, so
  donated inputs are not overwritten in place. Targeted files (sentinels, memory, profile, codes,
  runtime, compiler, kernel, kernel verify and tune, region verify): 422 passed, 4 skipped.
  Full suite with the GPU visible (36 min 58 s): 2197 passed, 13 skipped, 0 failed.
* Fix (order dependent failure found on main): the sentinels held their compiled program
  strongly while the program held them, a reference cycle. A program compiled with sentinels was
  then freed only by the garbage collector, so after tests/test_sentinels.py a resnet18's weights
  were still live CUDA tensors, and `test_diag_memory.py::test_forward_oom_names_the_instance_and_reraises`
  in the same process listed them as the largest tensors instead of its model's params. The
  sentinels now hold the program by weak reference. New test: with the garbage collector off, a
  program compiled with sentinels (read per call and deferred, a watched torch step included) is
  freed by reference counting alone and none of its device buffers stay live; it fails without the
  fix. tests/test_sentinels.py, tests/test_diag_memory.py, tests/test_memory.py and
  tests/test_codes_unique.py in one process: 60 passed.

Not verified: CPU backend overhead (correct in the tests), ROCm, other GPUs, Linux (where a host
read may cost less), Nsight Systems as a third timing reference.

### 2026-11-24: parallel layouts against the single device step (`truestar parallel`, P34, research R5)

Branch wt/parallel. Research item R5 with its redesign part. Docs: docs/PARALLEL.md (new). Codes
TS680 to TS690. torch 2.11 CPU; every rank simulated (threads in one process, or gloo CPU
processes); the one GPU on this machine is not used.

* **Redesign against the usual check (two loss curves over days).** One step, both arms in float64,
  so a correct layout differs only by sum order (about 1e-16) and one fixed threshold (1e-9) needs no
  calibration. The pieces every rank holds (module outputs, the gradients arriving at them,
  parameter gradients after every sync, updated weights) are reassembled and compared in causal
  order (shards, forward, loss, output gradients, parameter gradients, update, drift), so the first
  difference names its cause: a module, a collective, or the loss normalization the layout
  actually computed. Everything is written from the definitions; no code from Megatron, DeepSpeed
  or the PyTorch distributed packages.
* **Layouts:** tensor parallel column (with fused parts) and row splits, data parallel with one
  gradient bucket all reduce, gradient accumulation with every micro loss divided by the all reduced
  global count (right for uneven token counts), sequence parallel norms between a row and the next
  column layer, clipping by the global norm. pp above 1 is refused (TS680).
* **Detection** (scripts/parallel_study.py): 10 injected faults, 10 named in one step at the right
  module or collective (table in docs/PARALLEL.md); the step 1 loss differs for 4 of them. Factors
  that coincide in a layout (dp equal to ga) are reported as ambiguous with both causes.
* **False alarms:** 0 of 88 correct runs (9 layouts, 4 seeds, SGD and AdamW with clipping, a padded
  cross entropy classifier, the automatic plan); worst error 3.7e-11.
* **Found on the way and fixed before it shipped:** the thread communicator first handed out
  references to the callers' tensors, and a rank that wrote its synced gradient back before a peer
  had summed changed the peer's sum (random errors up to 0.5 on sequence parallel norms); slots now
  hold copies. Attention's key bias gradient is zero in exact math and is compared against 1e-5 of
  its module's largest gradient. A column output fused from two parts is reassembled part by part.
* **Cost:** 0.17 s per check on threads (tiny transformer, tp=2,dp=2,ga=2,sp=1, 64 collectives per
  rank), 5.1 s on 4 gloo processes (mostly start up), 0.63 s for a 4 block width 256 model.
* **Tests can fail:** leaving out the sequence parallel sum over tp in the correct runtime fails 3
  tests; a threshold of 1e9 fails 10 of 25; restored.
* Tests: tests/test_parallel.py, 28 (2 start gloo processes or the command line). Example:
  examples/parallel_model.py in tests/test_examples.py.

Not verified: NCCL, real GPUs, more than one machine; a third party framework's own step; the
findings of Megatron-LM#7452 from their repros (other projects' code is not used here); FSDP, ZeRO,
pipeline, vocabulary and expert parallel; bfloat16 effects of a layout.

### 2026-11-24: training spike shadow and replay (`truestar replay`, P35, research R9)

Branch wt/parallel. Research item R9 with its redesign part. Docs: docs/SPIKES.md (new). Codes
TS691 to TS697. torch 2.11, float16 and bfloat16 training on the CPU.

* **Redesign against rewind and skip.** The watch keeps a checkpoint every few steps, every batch,
  every learning rate and loss, so any step in the window can be rerun exactly; the replay must
  reproduce the logged losses before it gives a verdict (TS695 otherwise). Three arms from the
  checkpoint: the run's precision, float64, and batch N alone on the checkpoint weights. Numerics
  when only low precision spikes (first module over 50 epsilons named), data when float64 spikes
  and the batch is high on the old weights (rows named), optimizer when float64 spikes but the batch
  is normal (learning rate jumps named). A shadow step reruns one batch in float64 from the same
  weights, sampled or triggered by a spike.
* **Detection** (scripts/spike_study.py, 5 data seeds per case): numerics 5 of 5 at `norm`, data 5
  of 5 with rows 3 and 9, optimizer 5 of 5 with the learning rate jump at step 44, clean 5 of 5 none;
  the watch shadowed and flagged every numerics spike (TS691) on its own. The same batch under
  bfloat16 does not spike.
* **False alarms:** 0 of 60 shadow flags and 0 of 55 replay verdicts on clean runs.
* **Found on the way and fixed:** the first replay used the gap between the low precision and the
  float64 trajectories as evidence; it is 14% after 5 steps in a clean float16 run, so it now uses
  each step's own error (the same weights in float64). A forward hook returning a tuple replaced
  module outputs in the first module error pass.
* **Overhead:** shadow every 50 steps +0%, every 10 +6%, every step +101% on a 146 ms float16 CPU
  step; the fixed cost of saving each batch and log line is about 0.7 ms per step (shared desktop,
  timings vary by about 30%).
* **Tests can fail:** a numerics module bar of 1e9 fails the norm blame test; restored.
* Tests: tests/test_spikes.py, 9. Example: examples/spike_demo.py in tests/test_examples.py.

Not verified: the 28M time bomb setup of arXiv 2610.38277 on the GPU, a rising module error before a
spike, custom training loops (accumulation, autocast, schedulers with state), GPU determinism for
replay.

### 2026-11-24: conformance of a deployed artifact (`truestar conform`, P31)

Branch wt/conform. Research items: gap analysis (GAP_DEVICES_FRAMEWORKS) items 3, 6, 8 and 10 with
their redesign sections. Docs: docs/CONFORM.md (new), docs/AGENT_INTERFACE.md (codes TS620 to
TS639). Installed: ai-edge-litert 2.4.0 (plus backports.strenum) into the ml and cv venvs, torch
and numpy unchanged. cv venv: torch 2.11, onnxruntime-gpu 1.29.0, onnxruntime-ep-webgpu 0.6.0,
executorch 1.2.0; ml venv: onnxruntime 1.20.1. RTX 5070 Ti shared with other jobs.

* **Command.** `truestar conform model.py ARTIFACT` for .onnx (`ort:cpu`, `ort:cuda`,
  `ort:webgpu`), .tflite (`tflite:interp`, `tflite:ref`, `tflite:cpu`, `tflite:gpu`,
  `tflite:gpu-f32`), .pte (`pte:cpu`) and .tsx (`tsx:reference`, `tsx:cpu`, `tsx:cuda`). Core ML
  is refused with the reason (TS620): no Windows wheel of coremltools 9.0 and `predict()` needs
  macOS. `truestar conform replay BUNDLE ARTIFACT` is the upgrade baseline.
* **Redesign against the quickstart's one input allclose.** The verdict comes from the unmodified
  engine on 8 samples (flips with near ties); the file's own math is evaluated in float64, so a
  conversion fault and a runtime fault are separate verdicts; a runtime whose error is a correct
  float16 or TF32 run's error is named as such; localization runs only after a mismatch, on module
  cuts of the file run alone in the same runtime with float64 reference inputs (no instrumented
  run is judged); residency is read per node from what each runtime exposes, and where LiteRT only
  says "not fully accelerated" each operator is compiled alone to name the ones that are not; every
  instrumented run is checked against two plain runs.
* **Faults, all named at the right module** (scripts/conform_study.py): gelu decomposed as tanh
  (ONNX and TFLite): `blocks.0`, conversion, 8 of 8 runtimes; TFLite layer norm with eps outside the
  square root: `blocks.0.ln`, conversion, 5 of 5; ONNX INT8 quantize step rounding to floor: `q2`,
  conversion, decisions flipped, 3 of 3. The baseline allclose flags these files and names nothing.
* **Residency, real:** a Det node runs on the CPU in both ONNX Runtime GPU sessions, named with
  module `head`; CUMSUM outside XNNPACK in LiteRT (`tflite:interp` from the plan, `tflite:cpu` from
  29 single operator compiles, 5.8 s); the ExecuTorch XNNPACK program leaves both layer norms
  (`blocks.0.ln`, `blocks.1.ln`) and the cumsum on portable kernels. allclose passes all of them.
* **Precision, real:** LiteRT's GPU accelerator computes float32 files in float16 by default
  (error 1.6e-3, a correct float16 run 5.7e-4, 1 of 512 decisions flipped on the clean file);
  `tflite:gpu-f32` on the same file is ok. ONNX Runtime CUDA's default TF32 is judged as TF32 (the
  baseline allclose fails the clean file there); on the INT8 fake quant model TF32 moves quantization
  levels (1.6e-2, a correct TF32 run 1.9e-2), reported as TS627, every module alone within limits.
* **Observer guard:** profiled ORT sessions and LiteRT preserved tensors were identical to plain
  runs on every file (no real observer effect found here); an injected 1e-3 change is TS630.
* **Upgrade baseline, real:** a bundle written with onnxruntime 1.20.1 and replayed with
  onnxruntime-gpu 1.29.0 on the same file: changed (max 7.2e-7, 0 decisions moved), still
  conformant (TS634), first changed module `blocks.0.ln` (LayerNormalization) while every Gemm
  module is bit for bit identical, fingerprint diff names the ORT version, package, numpy and
  Python. recdiff/fingerprint.py gained `runtime_fingerprint` and `runtime.*` key kinds.
* **Found on the way and fixed:** `stacks._module_path` joined the cumulative ONNX name scopes
  (`blocks.blocks.0.blocks.0.fc`); it now names the module once (P22 TS400 messages for nested
  modules). onnx.reference's LayerNormalization refuses a float64 stash type, so the float64
  meaning keeps it; the clean ONNX file's float64 meaning then differs from the model by 1.1e-8 on
  onnx 1.22 (5.7e-12 on onnx 1.17), under the 3.3e-7 conversion bar.
* **Tests can fail:** tests/test_conform.py, 21 tests (all pass in the cv venv; in the ml venv the
  GPU, ExecuTorch and ORT CUDA ones skip by name). With residency blinded 3 fail, with the float64
  meaning disabled 9 fail, with the observer guard always trusting 2 fail; restored.

Not verified: the published litert-torch#718 and transformers.js#1728 files (no downloads), files
from a real PyTorch to TFLite converter (the tests use Truestar's schema writer), DirectML, Core ML,
LiteRT NPU accelerators, remote targets, two ORT versions spanning a known fixed bug, `.pte`
localization below the outputs.

### 2026-11-24: engines and other frameworks as port candidates (P32, gap items 11 and 12)

Branch wt/port2. Research items: GAP_DEVICES_FRAMEWORKS.md items 11 and 12 with their redesign
sections. Docs: docs/PORT_ENGINES.md (new). Installed into the ml venv (torch 2.11.0+cu128 and
numpy 2.2.6 pinned, nothing else changed): llama-cpp-python 0.3.38 (CPU wheel), gguf 0.19.0 (tests
only, to cross check the decoders), jax 0.6.2, jaxlib 0.6.2, flax 0.10.7, keras 3.12.4 and their
dependencies. Codes TS640 to TS658. Rows served: P32 (new), P16, never introduce #1 and #2.

* **llama.cpp (`truestar port ref.py gguf:model.gguf`).** GGUF read, written and decoded by
  Truestar from the published format (F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K; bitwise equal to
  the gguf package on every Q8_0, Q4_K, Q6_K and F16 tensor tested). The engine runs through its C
  API; every graph node is read with the scheduler eval callback, so no graph truncation is needed.
  Three verdicts that never mix: (a) each step replayed in float64 on the engine's own input with
  the GGUF's weights and an explicit list of allowed rounding paths (Q8_0 and Q8_K activation
  blocks, f32 or f16 activations for F16 weights, an f16 KV cache), so its error is the engine's own;
  (b) settings by name and every tensor matched by content under transpose and the rotary row
  permutation, then located by a float64 replay; (c) quantization loss from float64 replays only.
  Settings the engine actually used are read from its graph (RMS_NORM eps, ROPE base and scale,
  SOFT_MAX scale), so "engine ignored the file" is told apart from "file holds a wrong value".
* **Measured** (scripts/engine_port_study.py, 5 seeds): 0 of 40 clean runs (F32, F16, Q8_0,
  Q4_K_M, KV f32 and f16) alarm; 140 of 140 injected file faults (rope theta, eps, rope scaling,
  transposed square weight, unpermuted q or k, scaled weight, no BOS) caught with the right blame,
  none blamed on the engine (final logits baseline: 116 flagged, none named); 60 of 60 injected
  engine node faults (1e-4 to 1e-2) at the right node, against 19 for the final logits and 33 for a
  naive per layer dump (3 of 15 on Q4_K_M). Without the rounding model clean Q8_0 and Q4_K_M files
  have 15 steps over threshold each. Median 0.09 s per check, engine pass 4 ms.
* **Cross framework (`jax:cand.py`, `keras:cand.py --keras-backends jax,torch`).** Each framework
  and Keras backend in its own float64 CPU process; weights cross by safetensors, activations by
  DLPack; modules map by weight content; each mapped module replayed alone on the reference's
  inputs. 0 of 8 clean runs alarm; 18 of 18 faults (eps, theta, adjacent rotary pairs, missing
  transpose, a bug on the torch backend only) named at the right module; TS656 for the one backend
  bug. Median 6.1 s (Flax), 14.6 s (Keras, two backends), mostly process start.
* **Found on the way.** (1) Keras 3.12.4 maps every 64 bit result type to 32 bit outside TensorFlow
  (keras/src/backend/common/dtypes.py), so with floatx float64 a Dense layer returns float32 on the
  torch backend and float64 dtypes with float32 level values on jax; reported as TS658 with float32
  thresholds, not as a wrong port. (2) transformers' LlamaForCausalLM keeps rotary cos and sin and
  RMSNorm in float32 after `.double()` (1.8e-7 from an exact replay, verified by patching each);
  accepted at float32 level with a note. (3) Keras on torch places variables on the GPU; the
  framework process hides the GPU. (4) First versions false alarmed and were fixed before merging:
  F16 weights and an f16 cache use other rounding paths at this batch than the first model assumed
  (now an explicit list, the fitting path reported), and an end to end check (TS657) fired on 7 of
  40 clean quantized runs because activation rounding is discontinuous (now bounded by the
  rounding's own effect on the logits).
* **Tests:** tests/test_engine_port.py 39, tests/test_xport.py 11. Tests that fail on purpose: the
  replay without rounding paths false alarms on a clean Q8_0 file; the matcher without transposes
  no longer names a transposed weight; the framework replay on the candidate's own inputs flags at
  least 10 downstream modules for one bad kernel.

Not verified: CUDA, Vulkan or Metal builds of llama.cpp (only the CPU wheel was installed), so the
per backend comparison of the research item is not shown; real llama.cpp bugs from a before and
after commit pair; flash attention engine runs; architectures other than llama; published GGUF
files or checkpoints; TensorFlow and MLX backends; the Keras pooling caveat and keras#24170 and
#24248; op level drilldown inside JAX or Keras modules.

Found after the full suite (same branch): installing jax brings `opt_einsum`, and with it present
`torch.einsum` picks a contraction path from the example shapes while `torch.export` traces, which
specialized a symbolic batch dim: `tests/test_ir_ops.py::test_op_matches_pytorch_float64[einsum]`
failed. The importer and the exporter now trace with opt_einsum switched off (`plain_einsum`), so
the imported graph no longer depends on what else is installed; tests/test_plain_einsum.py (2) fails
without the fix.

### 2026-11-24: SM aware launch defaults, measured on a 26 SM laptop GPU

Second machine: RTX 5060 Laptop GPU (sm_120, 26 SMs, 8 GB, 101376 bytes of shared memory per
program), Ryzen 9 270, Windows 11, torch 2.11.0+cu128, Triton 3.7.1, GPU idle. The contraction
defaults had been tuned for 70 SMs (laptop run of 2026-11-24, raw logs not published: at small M the default was 24% to
43% behind the tuned config there).

* `triton_backend.device_limits()` reads SMs, shared memory per program and total memory of the
  inputs' device. Thresholds written for 70 SMs scale by `sms / 70` (`_waves`). At 70 SMs every
  default and every tuning candidate is unchanged: 126 cases (strict float32, TF32, float16, each
  also deterministic) equal main's output, frozen in tests/data/contraction_defaults_70sm.json.
* `scripts/sweep_contraction.py` (new) timed every strict float32 grid tile at split K 1 to 16 on
  18 shapes. With 26 SMs the winners want about 4 waves of programs, not about 1, so scaling the
  70 SM thresholds down would have picked the wrong tiles (128 x 256 at 1024 cubed: 480 against 423
  us). Below `FEW_SMS` (48) strict float32 follows `_few_sm_ieee`: up to 16 rows 16 x 128 x 32 with 2
  warps, 4 stages and split K to 4 waves in chunks of at least 128; 17 to 32 rows 32 x 64 x 32;
  128 x 256 from 5 waves, 128 x 128 from 2. Every one of the 18 shapes is within 9% of its best
  config (17 within 5%), and the 18 together went from 8461 to 7833 us of default time.
* `scripts/bench_matmul.py`, default config, before / after (us, median of 7), on the laptop:

| M x N x K | before | after | after / F.linear | after / best cuBLAS path |
|---|---|---|---|---|
| 1 x 4096 x 4096 | 440.9 | 281.0 | 1.49 | 1.50 |
| 16 x 4096 x 1024 | 112.9 | 70.8 | 0.63 | 0.63 |
| 512 x 1024 x 256 | 86.9 | 70.6 | 1.36 | 1.64 |
| 16 x 1024 x 16384 | 450.0 | 318.7 | 0.52 | 0.52 |
| 1024 x 1024 x 1024 | 458.3 | 440.2 | 1.10 | 1.14 |
| 4096 x 4096 x 4096 | 24287 | 24311 | 0.84 | 1.26 |

  The other 10 shapes moved by at most 7% (64 x 2048 x 1024 from 68.2 to 63.4 us; the worst, 64 x 512
  x 8192, 141.0 to 146.5 us at the same config, which is noise). Max error against float64 is 1.8e-7
  to 2.0e-6, as before.
* Shared memory budget: the device's per program limit minus 3 KiB (96 KiB on sm_120, unchanged).
  Split K workspace: at most 1/16 of the GPU's memory (it was bounded only by 2^31 elements, which
  is 8 GB of float32, the whole laptop GPU). Candidate pruning counts programs in waves of the GPU.
* Determinism: deterministic mode keeps `SPLIT_K` 1 on any SM count (tested at 8, 26, 47, 70 and
  132), and the 26 SM and 70 SM deterministic defaults of a 64 x 4096 x 1024 linear are different
  tiles with the same bits. `scripts/laptop_fingerprint.py` hashes are unchanged after the change on
  the MLP, TransformerEncoderLayer and resnet18, which are bitwise equal to the RTX 5070 Ti's.
* Local memory (TS700, `ConfigResourceError`): every launch is now compiled before its first run
  and its local memory (Triton's `n_spills`) checked against 512 KiB per thread (the CUDA limit)
  and a quarter of the GPU's memory for every resident thread. The tuner records the reason and skips
  the candidate; a default or forced config raises. Found: on 26 SMs 128 x 256 x 16 (8 warps) spills
  48 bytes per thread and 128 x 128 x 16 (4 warps) 72 to 80, about 3 MiB in all, far below the limit.
* Not explained: the first sweep run crashed the CUDA context ("unspecified launch failure",
  nvlddmkm event 13, SKEDCHECK05_LOCAL_MEMORY_TOTAL_SIZE) while about 40 CUDA graphs were alive. It did
  not reproduce in three attempts: the full sweep with one graph at a time, every spilling config
  launched and replayed with the GPU filled to 0 MB free (scripts/probe_local_memory.py, 12 of 12
  clean), and the same 40 graphs of that shape alive at once. The TS700 check does not claim to
  cover it.
* Tests: tests/test_sm_aware.py (23: the frozen 70 SM table, the 26 SM choices, deterministic mode on
  five SM counts, pruning, shared memory, workspace, a GPU less process, deterministic bits across
  SM classes, a spilling config refused with TS700 and skipped by the tuner);
  test_contraction_default_config_follows_operands_and_tf32 is pinned to 70 SMs.

Not verified: SM counts other than 26 and 70 (FEW_SMS 48 is between them, not measured), half and
TF32 defaults on few SMs (not SM aware yet), conv defaults (the conv kernel is not on main yet),
first call cost (unchanged: attention S8192 is 155 s here against 72 to 83 s on 70 SMs, almost all
Triton compile and search).
### 2026-11-24: localized faults in `truestar check` and per element kernel verification

Branch wt/checkrows. Adversarial review (truestar-research ADVERSARIAL_REVIEW.md) findings 4.1 and
4.2, ranked fixes 2 and 9. Never introduce #2 (no false alarms) and #3 (no silent wrong results of
our own). RTX 5070 Ti shared with other jobs, torch 2.11, ml venv.

* **The bug, `truestar check`.** Each tensor was judged by one whole tensor relative error against
  fixed floors. One wrong row in 1024 is 1/sqrt(1024) = 0.031: OK under the bfloat16 floor 0.04,
  only suspect in float16. Now each output also gets a local measure (`compare._local`): the worst
  element against max(|b_i|, rms(b)) and the worst slice (every index of every dim) against its
  values floored at rms(b) * max(1/32, 1/sqrt(n)); local limit 8x the threshold, divergent by the
  aggregate rule on that limit with a 0.5 cap. TS010 and the text report name the slice
  (`at index 517 of dim 0 (1024 elements)`), the agent JSON adds `local_err`, `local_limit`,
  `local_at`, `localized` on the blamed module and `local_err` on outputs and modules over
  threshold. New `DeviceFault` kinds `row`, `element`, `last_col`, `region`, and
  `device_type="lowp"` for CPU only fault runs.
* **Measured, check** (`scripts/bench_localized.py` and `--old`; docs/TOLERANCE.md has the table).
  Clean: 11 models x 3 dtypes x CPU and CUDA = 66 runs. Faults: 4 localized kinds on a 1024 x 1024
  layer plus the existing scale, zero, NaN and rows faults, every dtype, CPU and CUDA = 50 runs.
  False pass: 24 of 50 before (every localized run), 0 of 50 now. False fail: 6 of 66 before and
  after, the same runs: mobilenet_v3_small, efficientnet_b0 and convnext_tiny with random weights
  in float16, whose activations fall to about 1e-7 (below float16's smallest normal 6.1e-5) and
  lose precision on every device. On the other 60 the worst clean local error is 0.14 of its limit
  (vit_b_16 float16 CPU). Overhead on resnet18 (batch 8, CPU and CUDA): median 3.0 s against 2.7 s
  without the local measure, 3 alternating rounds on a shared machine.
* **Gradients keep the aggregate only.** With the local measure on, 6 tests of
  tests/test_gradcheck.py failed (clean resnet18 backward checks among them: input and weight
  gradients are sparse by construction). `diff_tensors(..., local=False)` and the backward check
  uses it; written in docs/GRAD_CHECK.md Limits.
* **The bug, kernel verification.** `kernel/verify.py` divided every element's error by the
  output's largest magnitude, so small elements could be wrong by 100% and pass. Now each element
  is divided by max(|ref_i|, floor), floor = rms(ref) times 1/8 (no reduction), 2 (a reduction) or
  4 (executor region and end to end checks, which do not see the kernel), capped at the largest
  magnitude so it is never looser than before. The message names the element.
* **Measured, kernel verification.** Every check of the kernel, tuning, compiler, executor,
  frontend, LLM op and training tests was recorded (5583 checks): the 5455 on correct kernels
  needed at most 0.011 x rms without a reduction, 0.52 with one, 1.29 in region checks (floors
  give 11x, 3.8x and 3.1x margin); the other 128 are the tests' deliberately broken kernels and fail
  under both measures. New faults (tests/test_kernel_verify_elements.py): `exp` wrong only on small
  results in float32, float16 and bfloat16 and a row sum 0.5% wrong on all but 4 dominant rows,
  CPU and CUDA: 8 compiled runs, false pass 8 of 8 before, 0 of 8 now; the same correct kernels
  pass. False fail on the 30 kernel and compiler test files: 0 (1190 passed, 3 skipped, plus
  test_kernel.py and test_kernel_gpu_verify.py 155 passed, 1 skipped). Compare cost is unchanged
  (13 ms per 1M element output on the CPU).
* **Honest wording.** Verification runs on the first call per shape signature in each process
  (docs/KERNEL_IR.md, Verification, "When verification runs"): data dependent faults on later
  calls with the same shapes, races the first call does not show, and errors under the floor are
  not seen; `TRUESTAR_VERIFY=all` or `truestar kernel` for those. README, COMPILER.md and never
  introduce #3 say so; README no longer calls the check thresholds "calibrated per module".

Not verified: localized faults in gradients; a wrong tile of small values in a large bfloat16
tensor (seen only by the element measure on the tensor's scale); regions below rms/32; reduction
kernels with an even output gain only about largest / (2 rms) over the old kernel measure; the
held out kernel sets of review item 1.7.

### 2026-11-24: native convolution and pooling kernels (strided indices, conv and window shapes)

Branch `wt/conv`. Research item 6 of the competitor gap list (native convolution and pooling) with
its redesign: implicit GEMM on a kernel family of its own, fused epilogues, every kernel verified.
Docs: docs/KERNEL_IR.md (strided indices, conv and window shapes), docs/COMPILER.md (convolution
rules), docs/AUTODIFF.md (gradient ops). RTX 5070 Ti shared with another agent's matmul benchmarks,
torch 2.11, Triton 3.7.1, Windows.

* **Kernel IR.** A load index entry may be a strided index, `(sum of coeff * ((axis * mul // div) %
  mod) + offset) / exact`, valid only when exactly divisible and inside the buffer; a load that may
  be invalid must name its `pad` value (zero padding, minus infinity for max pooling). One node
  serves convolution (stride, padding, dilation), grouped and depthwise channels (`div`), adaptive
  pooling windows (`mul`, `div`) and the transposed convolution of the input gradient (`exact`).
  The NumPy and torch float64 references gather with the same rules; numba guards the load; Triton
  masks it with `other=pad`.
* **Two GPU loop shapes** (`kernel/triton_conv.py`, tried after the existing shapes and only for
  kernels with a strided load, so no kernel that compiled before changes): conv, an implicit GEMM
  over a chain of nested sums (M = batch times pixels, N = output channels, K = channels times taps,
  run as one loop over (tap, channel block)), and window, static reduction loops per output element
  (pooling, depthwise and grouped convolution, the max pooling gradient). The window shape refuses
  GEMM shaped sums, so two convolutions never share an element by element kernel (found on
  resnet18: the downsample conv and its block's second conv fused into one window kernel).
* **Compiler.** `conv2d`, `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d` have kernel rules; the
  input read at strided indices is materialized, so its producer takes batch norm, relu, hardswish
  and residual adds as its epilogue (conv + batch norm + add + relu is one kernel). Constant weights
  are stored once with output channels contiguous. The torch library kernels run only when asked
  (`library=("conv2d",)`) or when a rule cannot take an op (a symbolic spatial size), with the
  reason in the report.
* **Gradients.** Three new IR ops replace the composite the conv and max pool gradient rules
  emitted (zero insertion, padding, flips, per tap slices): `conv2d_input_grad` (gather form of the
  transposed convolution, exact division, no zero inserted copy), `conv2d_weight_grad`,
  `max_pool2d_grad`. Each has a shape rule, a float64 reference, a finite difference case and its
  own gradient rule (second order through conv works).

Library kernels left (compile report): resnet18 0 (was 22: 20 conv2d, max_pool2d,
adaptive_avg_pool2d; now 23 Truestar kernels: 20 conv, 2 window, 1 contraction), mobilenet_v3_small
0 (73 Truestar kernels), mobilenet_v2 0 (54), the resnet18 backward program 0 (139 Truestar
kernels). No fallbacks in any of them.

Run time (`scripts/bench_conv.py`, 224 x 224, random weights, one process per model and precision,
tools interleaved, median of 7 rounds of 20 synchronized calls, ms; fp32 means IEEE float32 on
every side, cuDNN's TF32 turned off; tf32 allows TF32 on every side; graph is the same Truestar
program replayed from one CUDA graph, `truestar.runtime.capture`):

| model | precision | batch | eager (cuDNN) | torch.compile | Truestar | Truestar, graph | Truestar / torch.compile |
|---|---|---|---|---|---|---|---|
| resnet18 | fp32 | 1 | 2.49 | 1.77 | 1.43 | 1.36 | 0.81 |
| resnet18 | fp32 | 8 | 4.18 | 3.55 | 2.09 | 1.99 | 0.59 |
| resnet18 | fp32 | 64 | 13.27 | 11.77 | 11.04 | 11.38 | 0.94 |
| resnet18 | tf32 | 1 | 1.69 | 1.14 | 0.73 | 0.51 | 0.64 |
| resnet18 | tf32 | 8 | 2.10 | 1.84 | 1.37 | 1.32 | 0.74 |
| resnet18 | tf32 | 64 | 19.78 | 14.47 | 8.68 | 11.58 | 0.60 |
| resnet18 | fp16 | 1 | 3.11 | 1.38 | 1.04 | 0.46 | 0.76 |
| resnet18 | fp16 | 8 | 3.13 | 1.45 | 0.97 | 0.86 | 0.67 |
| resnet18 | fp16 | 64 | 7.06 | 5.96 | 4.83 | 4.56 | 0.81 |
| mobilenet_v3_small | fp32 | 1 | 10.46 | 3.37 | 2.81 | 1.65 | 0.83 |
| mobilenet_v3_small | fp32 | 8 | 3.66 | 1.80 | 2.32 | 1.68 | 1.29 |
| mobilenet_v3_small | fp32 | 64 | 5.34 | 3.11 | 3.66 | 3.61 | 1.18 |
| mobilenet_v3_small | tf32 | 1 | 3.99 | 2.27 | 2.57 | 0.76 | 1.13 |
| mobilenet_v3_small | tf32 | 8 | 3.69 | 1.90 | 3.08 | 1.46 | 1.62 |
| mobilenet_v3_small | tf32 | 64 | 3.98 | 2.39 | 3.15 | 3.36 | 1.32 |
| mobilenet_v3_small | fp16 | 1 | 5.89 | 2.88 | 3.02 | 0.60 | 1.05 |
| mobilenet_v3_small | fp16 | 8 | 4.50 | 2.10 | 2.35 | 1.42 | 1.12 |
| mobilenet_v3_small | fp16 | 64 | 4.20 | 2.12 | 2.77 | 2.57 | 1.31 |
| mobilenet_v2 | fp32 | 1 | 3.49 | 2.06 | 1.94 | 0.49 | 0.94 |
| mobilenet_v2 | fp32 | 8 | 4.52 | 3.81 | 1.99 | 1.64 | 0.52 |
| mobilenet_v2 | fp32 | 64 | 21.73 | 16.78 | 9.07 | 9.07 | 0.54 |
| mobilenet_v2 | tf32 | 1 | 4.03 | 2.31 | 2.23 | 0.30 | 0.97 |
| mobilenet_v2 | tf32 | 8 | 2.99 | 1.75 | 1.73 | 0.69 | 0.99 |
| mobilenet_v2 | tf32 | 64 | 13.79 | 9.67 | 5.15 | 5.15 | 0.53 |
| mobilenet_v2 | fp16 | 1 | 4.46 | 1.98 | 1.71 | 0.31 | 0.86 |
| mobilenet_v2 | fp16 | 8 | 3.33 | 1.61 | 1.32 | 0.63 | 0.82 |
| mobilenet_v2 | fp16 | 64 | 7.48 | 5.19 | 3.89 | 3.84 | 0.75 |

* Wins: resnet18 and mobilenet_v2 (the depthwise heavy net: 17 depthwise 3 x 3 convolutions, each
  a window kernel with its batch norm and relu6) are at or ahead of torch.compile in all 18 cells
  (1.01x to 1.9x faster) and ahead of eager everywhere. The gap list's measure for this item,
  resnet18 at batch 64 at or below torch.compile with zero library kernels, holds in all three
  precisions in this run (fp32 11.0 ms against 11.8; the 10.1 ms in the compiler entry was another
  day of the shared GPU, when eager took 11.8 ms against 13.3 here).
* Losses: mobilenet_v3_small is behind torch.compile in 8 of 9 cells (1.05x to 1.62x) and ahead only
  at fp32 batch 1. Its squeeze and excite blocks pool each channel through element by element
  window loops and add many small kernels; with a CUDA graph it is ahead at batch 1 and 8 and still
  1.2x to 1.4x behind at batch 64. Noise: tf32 eager at batch 64 is slower than fp32 eager in this
  run (19.8 against 13.3 ms), a sign of the shared GPU; ratios inside one row are interleaved.
* Per layer, against cuDNN conv plus relu at batch 64 (each tuned): float16 0.27x to 1.47x of
  cuDNN's time, float32 0.42x to 2.18x. The worst is the 512 channel 7 x 7 stage: a long K and few
  programs, and no split K (it would make the summation order depend on the launch config). Before
  the tap loop and the weight layout, the flat K kernel took resnet18 at batch 64 float16 to 21.0 ms
  against eager 6.2 ms.
* Correctness: largest end to end error against a float64 run of the PyTorch model: fp32 6.7e-7 to
  1.5e-6, tf32 1.8e-3 to 8.0e-3, fp16 7.5e-4 to 1.1e-3. Every kernel and region was checked on its
  first inputs.
* First call cost (a loss): the first call at a new batch size compiles every Triton kernel, tunes
  the launch configs worth tuning and checks everything: 3.7 s to 55 s per model and batch size
  (resnet18 fp32 46.7 s at batch 1), where torch.compile took 0.8 s to 28.6 s.

Backward (resnet18 input and conv weight gradients, eval mode, fp32): 139 Truestar kernels, 0
library. Batch 8 runs in 31.7 ms against 9.1 ms for eager autograd (3.5x slower: the weight
gradient has only C x 9 x O outputs and sums over batch and pixels, so it runs few programs with a
long flat K). At batch 32 the end to end check refused the result: the input gradient was 9.0% off
float64 while all 139 region checks passed. Float32 eager PyTorch is itself 5.0% off float64 on the
same gradient (0.5% at batch 8): relu masks of values within rounding of 0 decide differently in
float32. This is the near tie limit of docs/COMPILER.md, now reachable through backward programs;
nothing is returned in that case, and it is listed here as found, not fixed.

Tests: tests/test_conv_kernels.py, 119 cases (strided index rules, conv and window kernels against
float64 on CPU and GPU, 10 conv and 6 pooling variants in float32, float16 and bfloat16,
channels_last input with a symbolic batch, 5 gradient programs, a resnet block backward against
autograd, the library opt in, a rule that gives up after marking its input). Tests can fail: a conv
rule without its padding offset is caught as `ProgramMismatchError` naming `conv2d` on both
devices; a strided load that drops its exact division is caught by the kernel layer's check.
Existing tests updated: the resnet18 compile test now asserts 0 library kernels and 23 kernels (20
on the CPU, where the numba loop nest also takes each downsample conv), the broken library kernel
tests opt in with `library=("conv2d",)`, five finite difference cases and a zero gradient test for
the new ops.
Every torchvision classifier (`scripts/conv_zoo.py`, random weights, symbolic batch, 64 pixels, 224
for vision transformers and swin, 299 for inception_v3, float32 on the GPU, one checked run at batch
2): of 80, the 17 above 100M parameters were skipped (the float64 references take minutes); the
other 63 all import, compile and pass their checks (largest error against float64 PyTorch 2.7e-6).
56 have zero library kernels and zero fallbacks (the gap list's measure asked for the 46 exported
ones). The other 7 (swin_t, s, b, swin_v2_t, s, b, maxvit_t) keep only ops that are not convolution
or pooling: `index` as an index library step, and `pad`, `slice_scatter`, `vector_norm` as
fallbacks. No convolution or pooling op is a library kernel or a fallback in any of the 63.
Full suite in the worktree (CPU plus the shared RTX 5070 Ti): 1326 passed, 1 failed, 4 skipped in
13 min 47 s. The failure is tests/test_dlpack.py::test_cuda_consumer_stream_waits_for_the_producer
(a stream race check in the interop code, which this branch does not touch): it read a result
before the producer stream finished while the GPU was loaded, and passed 3 of 3 runs alone right
after. Recorded as intermittent under load, not fixed here.

Pain ledger and scorecard: no row changes status. Never introduce #3 now covers convolution and
pooling (verified kernels instead of declared library steps), #1 holds (a library kernel or
fallback is listed with its reason), and the gap list's measures for this item are met (resnet18 at
batch 64 at or below torch.compile with zero library kernels; every compiled torchvision
classifier free of convolution and pooling library kernels and fallbacks). No new error codes.

Not verified or not done: GPUs other than the RTX 5070 Ti, ROCm, an idle GPU (numbers move by up to
about 10% between runs here); conv1d and conv3d, `adaptive_max_pool2d` and the `avg_pool2d`
gradient (no kernel rule or no gradient rule); split K for deep layers and the weight gradient;
channels_last activations end to end (kernels read any strides, but the compiler writes NCHW);
batch norm in training mode inside a conv epilogue.

After the merge with main (matmul2 split K, TF32 following torch, forward mode and vmap passes),
one rerun of `scripts/bench_conv.py --models resnet18 --batches 64` (same method, ms):

| precision | eager | torch.compile | Truestar | Truestar, graph | before the merge (Truestar, torch.compile) |
|---|---|---|---|---|---|
| fp32 | 12.82 | 14.91 | 10.97 | 11.26 | 11.04, 11.77 |
| tf32 | 12.21 | 8.89 | 8.73 | 8.17 | 8.68, 14.47 |
| fp16 | 6.86 | 3.94 | 4.79 | 4.74 | 4.83, 5.96 |

Truestar held within 1% in every precision, still 23 kernels and 0 library kernels. torch.compile
moved by factors of 1.27 to 1.63 between the two runs on the shared GPU, and in this run it is ahead in fp16
(Truestar 1.21x slower), so the claim above that resnet18 at batch 64 is at or below torch.compile
holds for fp32 and tf32 in both runs and for fp16 in the first run only. The merge also needed
forward mode and batching rules for the three gradient ops (bilinear tangents; the vmap batch joins
the op's batch dim, or the channels with the groups multiplied); tests/test_forward.py and
tests/test_vmap.py run them on the same corpus cases (474 passed).

After the second merge with main (memory planner and recompute, WebGPU backend, opt_einsum off
while tracing): the targeted runs (conv, compiler, kernel, runtime, region, forward, vmap, memory,
IR op and code uniqueness tests) gave 1201 passed, 4 skipped; the full suite with the GPU visible
gave 2311 passed, 1 failed, 13 skipped in 37 min 39 s. The failure is the 60 s budget of
tests/test_examples.py::test_fault_demo_blames_the_cuda_only_layer, which took 61.1 s under the full
suite on the shared GPU and 7.8 s and 6.3 s in two runs alone; the budget was left as it is.

Third merge with main (sentinels, parallel, conform, port2): the conv and window kernels take the
sentinel flag words like every kernel (stored values, divisions, loads of torch step outputs), and
check the convolution's sum before its epilogue. Found while adding the test: on the GPU a NaN made
inside a conv (an infinite input times a zero weight) vanished in the fused relu (observed: Triton's
maximum of NaN and 0 gave 0), and the call raised nothing; the CPU loop nest, whose max keeps NaN,
raised. With the accumulator check both name `conv2d`, its line and the first bad element (implicit
GEMM and depthwise window kernels, tests/test_conv_kernels.py); with every check armed, max
pooling's minus infinity padding raises nothing. main's test of a convolution checked where a kernel
loads it now asks for the library kernel with `library=("conv2d",)`. CODEGEN_VERSION 5.

Tests after this merge: tests/test_conv_kernels.py, test_sentinels.py, test_memory.py,
test_codes_unique.py and the compiler, kernel, runtime and region test files gave 581 passed, 1
failed, 4 skipped; the failure was tests/test_kernel_tune.py::test_tuning_picks_a_config_and_a_second_process_reuses_it,
whose subprocess got "CUDA device busy" on the shared GPU, and it passed in two reruns. The full
suite (already running when the coordinator asked for targeted runs only) then gave 2455 passed, 0
failed, 17 skipped in 34 min 19 s with the GPU visible.

### 2026-11-24: precision planner (`truestar compile --precision auto`, P33)

Branch `wt/precplan`. Research item GAP_COMPETITORS 4 with its Redesign section. Docs:
docs/PRECISION.md (new). Codes TS660 to TS670. RTX 5070 Ti shared with other suites the whole time
(up to 15.8 of 16 GB used by other jobs), torch 2.11, Triton 3.7.1.

* **What it does.** Per unit (leaf module, or a parent that reads its children's weights, found by
  running the model) one of fp32, tf32, bf16, fp16, int8w, int4w, fp8, chosen so GPU time per
  forward is lowest while decision flips against float64 stay within `--max-flips` on the
  calibration inputs and, scaled, on held out inputs that never steer the search. Tie groups where
  code between units does not promote dtypes (each attention module of the decoder, TS667).
* **Redesign against TensorRT 11 and ModelOpt.** The tool that times the kernels picks the
  precision; truestar range excludes dtypes by analysis (TS663); quantized candidates keep the two
  quant verdicts (a kernel bug is removed, TS664, never counted as format loss); one error
  threshold, a dynamic program over activation dtypes that counts casts, bisection (O(log n)
  verification runs), six verified greedy refinements, then a held out gate that tightens (TS661).
* **Measured** (scripts/precision_study.py, docs/PRECISION.md tables): MLP with a planted one pass
  variance norm: every uniform 16 bit baseline and fp16 except norms flip 359 of 2048 decisions;
  the plan keeps the norm in float32 and runs 548 us at 0.5% (autocast 750 us, fp32 1728 us) and
  809 us at 0 flips (only fp32 also holds it). Llama style decoder at 0 flips: plan 1760 us, 0 and 0
  flips, against all tf32 2241 us (fastest baseline within budget), all fp16 1682 us with 1 and 1
  flips, fp32 3520 us; the held out gate tightened once. resnet18 at 0 flips: plan 2334 us against
  all bf16 2806 and all fp16 2931, fp32 6520. ViT: plan all fp16 6923 us against literal all fp16
  6730 us (TS670, parent owned class token and position embedding stay float32).
* **Found on the way and fixed:** the CUDA graph context per capture made timing take minutes; the
  clock warm up queued about 0.4 s of replays per candidate (Truestar's own
  `runtime.timing._spin` queues replays the same way; not changed here); a shared graph memory
  pool tripped an allocator assert in torch 2.11; the held out tightening could run out and call a
  budget unreachable that float32 met (the most accurate plan is now gated last).
* **Tests can fail:** tests/test_precplan.py 16 tests; range skipped, kernel verdict ignored, held
  out gate disabled, tie groups or merging removed each fail their test (restored).
  tests/test_examples.py gains the precision example. Not run: the full suite (coordinator runs it
  after merging).

Not verified or not done: per node precision in the compiled executor (plans run as PyTorch
modules plus Truestar quantized kernels), TensorRT ModelOpt autocast (not installed), NVFP4 and
MXFP4 candidates, decode shaped inputs where INT4 wins, an idle GPU for the resnet18 and ViT
planning times (130 s and 437 s were measured under contention, the ViT one before the timing
fixes), an MCP tool, other GPUs.
### 2026-11-24: a native layer library, optimizers and one program training steps (P36)

Branch wt/nnlib. Docs: docs/NN.md and docs/OPTIM.md (new), docs/FRONTEND.md, docs/COMPILER.md,
docs/MEMORY.md, docs/AGENT_INTERFACE.md (codes TS760 to TS766). torch 2.11, numba 0.68.1, ml venv,
RTX 5070 Ti shared with other jobs. Nothing installed.

* **`truestar.nn`.** Linear, Embedding, Conv2d, MaxPool2d, AvgPool2d, GlobalAvgPool, Flatten,
  LayerNorm, RMSNorm, BatchNorm (state in, state out), Dropout (keyed), activations, Sequential,
  MLP, MultiHeadAttention (causal, padding mask, grouped and multi query heads, KV cache), a pre
  norm TransformerBlock; losses mse, cross_entropy (ignore_index, label smoothing), accuracy. A
  layer is a frozen config; parameters and state are trees the caller holds; `training=` is a
  compile time argument; each forward is native source, run eagerly outside a native function and
  inlined inside one (the translator inlines any object with `__truestar_inline__`).
  Initialization draws each parameter at key `fold(seed, path)`: same bits everywhere, independent
  of construction order. 16 tests against torch in float64 at 1e-12, values and gradients.
* **`truestar.optim`.** SGD (momentum, Nesterov, weight decay), Adam, AdamW as two native subset
  methods (`scalars`, `leaf`) used both eagerly over trees and compiled into the step; schedules
  (constant, warmup cosine, warmup linear) on the host; clipping by global norm. 30 steps against
  torch.optim in float64: worst 1e-12 bar met for all five configurations. Bias corrections are
  `-expm1(t ln b)` with the new `ops.expm1`: within 4 float32 ulps for t up to 1e6 on the compiled
  CPU executor, where `1 - exp(t ln b)` is off by over 5e-6; the compiled float32 AdamW step that
  failed its kernel check (docs/MEMORY.md) now passes.
* **`truestar.train_step` and `fit`.** Forward, autodiff backward, non finite check, global norm and
  clipping, the optimizer and the loss scale update in one IR program; model state and aux results
  carried; random key `call_key(seed, step)`; safetensors checkpoints that refuse another program;
  bf16 and f16 autocast with GradScaler rules (checked step by step). Tests: a 25 step AdamW run with
  schedule and clipping and an 8 step batch norm CNN against torch at 1e-12; dropout replay and
  resume bit identical; one `prepare` for three batch sizes.
* **Examples, real training** (docs/OPTIM.md table): MLP, CNN and GPT loss curves match torch.nn in
  float64 to 6.4e-16, 7.1e-15 and 3.0e-15; float32 compiled to 2e-7 to 6e-7 (CNN on CUDA 3.0e-4,
  likely TF32 convolutions on the torch side). The GPT then generates through the KV cache, one
  compile for the prompt and 15 steps, equal to full recomputation. Speed, preliminary: compiled CPU
  3.0, 8.7 and 30.8 ms per step against eager PyTorch 0.8, 2.9 and 4.7 ms; CUDA 56, 101 and 52 ms
  against 22, 25 and 5 ms. Truestar is slower today; this is the open part of P36.
* **Found and fixed on the way:** kernel and group checks refused correct float32 sums whose terms
  cancel (a GPT bias gradient on CUDA 2.95e-5 against 2.5e-5; a conv bias before batch norm, exact
  gradient zero, 1.5e-3); a failing sum like value is now measured once more against the
  magnitude of its terms, and real errors are still refused (tests/test_kernel_magnitude.py). The
  conv tolerance crashed on a symbolic size (the weight gradient's conv). The attention gradient
  needs fixed lengths for `is_causal`, so layers build the causal mask explicitly. New kernels:
  `erf`, `log_softmax`, `amax`, `amin` (the GPT step had three reference fallbacks).
* **Found, not fixed:** `train.emulate_precision` does not round float32 (the reference `cast` keeps
  32 bit floats exact); 16 bit emulation is unaffected.
* **Tests can fail:** bias correction dropped from Adam failed 3 tests; a causal mask `cols < rows`
  failed 3; a batch norm that keeps the old variance failed 2; restored.

Not verified or not built: speed parity with eager PyTorch (the compiled step is 3 to 10 times
slower on these models), torch.compile comparison, CUDA graph capture of the step, larger models,
Lion and other optimizers on the native path, gradient accumulation and data parallel steps, mixed
precision for losses with loops or branches, attention dropout, rotary layers, conv transpose,
macOS and Linux.

### 2026-11-24: SM aware half and TF32 contraction defaults (26 SM laptop GPU)

Follow up of the SM aware entry above, same machine (RTX 5060 Laptop GPU, 26 SMs, idle). Long jobs
ran as Windows scheduled tasks. `scripts/sweep_contraction.py` gained `--dtype`, one child process
per shape (a shape whose child dies is rerun one config per child) and resume; it swept every
grid tile at split K 1 to 16 on the 18 shapes in float16, bfloat16 and TF32 (results/laptop-sm-half/,
no config crashed).

* The 70 SM half rules were already the best tile on 12 of 18 shapes; TF32 lost 9% to 43% wherever
  few programs ran over a long K. Below `FEW_SMS` (48), `_few_sm_half` and `_few_sm_tf32` change
  only these cases and otherwise give way to the 70 SM rules: 64 x 128 tiles for large products
  (half from 1024 rows, K of 1024 and 5 waves; TF32 from 256 rows with N and K of at least 1024),
  and split K to 4 waves where few programs run over a long K (to 8 waves for TF32 up to 32 rows).
  The first half rule used 4 waves and every K; at 1024 cubed and at K 256 the sweep (up to 7% faster)
  and `bench_matmul.py` (up to 19% slower by the minimum) disagreed, so those shapes keep 64 x 64.
* `scripts/bench_matmul.py`, default config, sum over the 16 shapes of the script, before / after:
  float16 7544 / 7136 us (0.95), bfloat16 7256 / 6765 (0.93), TF32 14793 / 13896 (0.94). Per shape
  with a new config: tall K 0.72 to 0.83 (TF32 128 x 256 x 8192 from 73.9 to 53.3 us, float16 64 x 512
  x 8192 from 36.2 to 30.1 us), small M TF32 0.74 to 0.87 (16 x 4096 x 1024 from 30.4 to 23.6 us), large
  products 0.93 to 0.94, the rest 0.96 to 1.01. Slower: none of the shapes with a new config beyond 1%;
  four shapes with the same config as before ran 6% to 10% slower (float16 1024 cubed, 512 x 1024 x 256 in
  float16 and TF32, bfloat16 64 x 2048 x 2048), which is the run to run noise of this GPU.
* Against cuBLAS (F.linear with relu) after the change: float16 0.54 to 1.06, bfloat16 0.79 to 1.20,
  TF32 0.39 to 1.20; the losses are the tall K shapes in bfloat16 and TF32 (1.15 to 1.20).
* Determinism: without split K the new tall K rules do not apply, so deterministic defaults change
  only by tile size. Measured: the 26 SM and 70 SM deterministic defaults of a 2048 x 4096 x 1024
  float16 and bfloat16 linear (64 x 128 and 64 x 64 tiles) give the same bits, the first measurement of
  config independence for half precision tensor core dots. `scripts/laptop_fingerprint.py` gained
  `--dtypes`; compiled hashes of the MLP, TransformerEncoderLayer and resnet18 in float32, float16 and
  bfloat16 (9 cases) are unchanged by the change.
* At 70 SMs nothing changed: the frozen table now has 168 cases (bfloat16 added).
* Tests: tests/test_sm_aware.py 39 (14 new: half and TF32 choices on 26 SMs, deterministic fall back
  to the 70 SM rules, half deterministic bits across SM classes). Full suite: 2149 passed, 15 skipped.

Not verified: SM counts other than 26 and 70; the attention kernels (their defaults do not read the
SM count; decode runs one program per batch and head, which a split over keys would change).

### 2026-11-24: transformer inference against eager PyTorch and torch.compile (P38)

Branch `wt/xfmrperf`. New: `kernel/library.py`, `compiler/graphs.py`, `scripts/bench_xfmr.py`,
tests/test_xfmr_perf.py (21), codes TS800 to TS802. Docs: docs/COMPILER.md (Transformer inference
speed), docs/KERNEL_IR.md, docs/LLM.md, docs/RUNTIME.md. RTX 5070 Ti, torch 2.11.0+cu128, Triton
3.7.1, Windows. **Every number here is provisional** until the dedicated fair rerun on an idle GPU:
the matrix below ran while the GPU was mostly idle (utilization by others 0% to 60% between cases,
SM clock logged per case), but other agents shared the machine during the day, and Linux and torch
2.14 were not run.

**1. Where the time went (before, `truestar profile` and torch.profiler traces).**

* TransformerEncoderLayer, batch 64, seq 128, d 256, strict float32: 0.90 ms per call against eager
  0.67 to 0.73. GPU per call: the four contractions 0.71 ms (Triton IEEE `tl.dot` on FMA units,
  16.7 to 19.4 TFLOPS; eager's cuBLAS ran the same products in 0.53 ms), attention 0.126 ms against
  eager's 0.082 (cuBLAS bmm, softmax, bmm), head split copies 0.033 ms; fused add plus layer norm
  0.025 ms against eager's 0.050. Host 0.37 ms, hidden behind the GPU.
* Llama style decode step (batch 8, bf16): 0.396 ms of GPU steps and about 1.3 ms of host issue for
  39 steps (host bound): the KV cache copied whole by `index_copy` (0.123 ms), the KV heads repeated
  for grouped query attention (2 copies, 0.026 ms each in bf16, 0.053 and 0.033 in fp32), the
  projections run as one 16 row tile per sequence (75 to 80 GB/s in fp32), attention 0.26 ms in fp32.

**2. Changes, each measured on its own.**

* **Fold batch axes into contraction rows** (`ContractionPlan.fold`): decode batch 8 fp32 MLP
  projections 0.155 to 0.035 ms and 0.143 to 0.027 ms (75 GB/s to 330 to 440 GB/s).
* **In place KV cache** (`index_copy` writes into a dead or declared input; `cp(..., update={i: j})`):
  the cache write is the new positions only; outputs bitwise equal to the out of place call.
* **Grouped query attention by broadcast**: the two cache copies per layer are gone. Decode batch 8
  bf16 GPU steps 0.396 to 0.152 ms with the three changes above.
* **cuBLAS as a contraction implementation** (`kernel/library.py`), chosen per shape bucket by timing
  against the Triton launch, verified against float64 like any launch config, only in the float32
  precision PyTorch asks for, never in deterministic mode. Encoder layer strict float32: 0.90 to
  0.755 ms.
* **CUDA graph replay** (`cp.graphed()`): one launch per call; decode batch 8 bf16 1.02 to 0.208 ms,
  decode batch 1 fp32 1.56 to 0.300 ms. No gain on the encoder layer (GPU bound, 0.75 ms either way).

**3. Results.** Method (`scripts/bench_xfmr.py`, ADVERSARIAL_REVIEW.md 1.9): every tool in one
process, interleaved over 5 rounds of 20 calls, one CUDA event pair per call, L2 flushed (128 MB
write) before every timed call, warm up excluded (Truestar's first call with its float64 checks is in
the warm up). TF32 pinned identically (`matmul.allow_tf32` and `cudnn.allow_tf32` both False for
fp32, both True for tf32); bf16 and fp16 cast model and inputs; PyTorch's reduced precision reduction
flags left at their default (True) and printed. Eager runs with every SDPA backend that accepts the
case (no flash on Windows; efficient and cudnn refuse some masks and grouped query attention) and,
for decode, also with the cache sliced to the live positions; the fastest is the eager column and its
backend is named. torch.compile in `default` and `max-autotune` (which captures CUDA graphs). Truestar
runs with its defaults (verification on, first call excluded), the KV cache updated in place, plain
and replayed from CUDA graphs. Times are median / min in ms over 100 calls; the verdict compares the
best Truestar median with the best other median. Raw JSON: docs/bench/xfmr_2026-11-24_provisional.json.

| case | prec | Truestar | Truestar graphs | best eager (backend) | compile default | compile max-autotune | Truestar best / best other | SM MHz |
|---|---|---|---|---|---|---|---|---|
| tel | fp32 | 0.779 / 0.773 | 0.793 / 0.789 | 0.683 / 0.666 (efficient) | 0.669 / 0.664 | 0.683 / 0.678 | 0.86x LOSS | 2932 to 2940 |
| tel | tf32 | 0.399 / 0.396 | 0.418 / 0.416 | 0.484 / 0.480 (efficient) | 0.482 / 0.479 | 0.498 / 0.494 | 1.21x win | 2962 to 2962 |
| tel | bf16 | 0.227 / 0.222 | 0.231 / 0.229 | 0.313 / 0.308 (efficient) | 0.312 / 0.309 | 0.312 / 0.310 | 1.37x win | 2947 to 2947 |
| tel | fp16 | 0.365 / 0.223 | 0.233 / 0.229 | 0.255 / 0.249 (efficient) | 0.255 / 0.251 | 0.257 / 0.255 | 1.10x win | 2947 to 2947 |
| prefill | fp32 | 1.068 / 0.987 | 0.861 / 0.857 | 0.924 / 0.853 (math) | 0.670 / 0.664 | 0.638 / 0.633 | 0.74x LOSS | 2572 to 2572 |
| prefill | tf32 | 0.603 / 0.411 | 0.402 / 0.398 | 0.732 / 0.585 (math) | 0.414 / 0.410 | 0.377 / 0.373 | 0.94x LOSS | 2962 to 2962 |
| prefill | bf16 | 0.701 / 0.394 | 0.232 / 0.230 | 0.559 / 0.378 (cudnn) | 0.314 / 0.308 | 0.242 / 0.240 | 1.04x win | 2677 to 2677 |
| prefill | fp16 | 0.545 / 0.246 | 0.234 / 0.231 | 0.460 / 0.301 (cudnn) | 0.285 / 0.278 | 0.243 / 0.240 | 1.04x win | 2947 to 2947 |
| decode1 | fp32 | 1.556 / 1.189 | 0.300 / 0.221 | 0.913 / 0.403 (math) | 0.385 / 0.143 | 0.167 / 0.100 | 0.56x LOSS | 2332 to 2880 |
| decode1 | tf32 | 1.256 / 0.940 | 0.242 / 0.195 | 0.807 / 0.541 (math) | 0.218 / 0.150 | 0.137 / 0.104 | 0.57x LOSS | 1672 to 1972 |
| decode1 | bf16 | 1.061 / 0.656 | 0.187 / 0.148 | 0.545 / 0.325 (cudnn) | 0.184 / 0.120 | 0.106 / 0.082 | 0.57x LOSS | 540 to 2070 |
| decode1 | fp16 | 0.971 / 0.589 | 0.282 / 0.243 | 0.570 / 0.426 (cudnn) | 0.272 / 0.120 | 0.221 / 0.075 | 0.78x LOSS | 2962 to 2962 |
| decode8 | fp32 | 1.438 / 1.258 | 0.339 / 0.318 | 0.835 / 0.707 (math:sliced) | 0.376 / 0.367 | 0.225 / 0.192 | 0.66x LOSS | 2512 to 2857 |
| decode8 | tf32 | 0.907 / 0.591 | 0.238 / 0.203 | 0.659 / 0.468 (math:sliced) | 0.367 / 0.342 | 0.157 / 0.137 | 0.66x LOSS | 2947 to 2962 |
| decode8 | bf16 | 1.019 / 0.947 | 0.208 / 0.200 | 0.597 / 0.403 (cudnn:sliced) | 0.275 / 0.272 | 0.122 / 0.093 | 0.59x LOSS | 2640 to 2745 |
| decode8 | fp16 | 1.138 / 0.658 | 0.228 / 0.175 | 0.482 / 0.386 (cudnn:sliced) | 0.279 / 0.265 | 0.136 / 0.095 | 0.60x LOSS | 1440 to 1800 |
| decode32 | fp32 | 1.233 / 0.969 | 0.558 / 0.530 | 1.029 / 0.915 (math:sliced) | 1.261 / 1.225 | 0.279 / 0.264 | 0.50x LOSS | 2587 to 2587 |
| decode32 | tf32 | 2.143 / 0.420 | 0.456 / 0.322 | 1.675 / 0.922 (math:sliced) | 1.429 / 1.196 | 0.231 / 0.223 | 0.51x LOSS | 2947 to 2955 |
| decode32 | bf16 | 1.082 / 0.679 | 0.277 / 0.238 | 0.593 / 0.448 (cudnn:sliced) | 1.090 / 1.057 | 0.133 / 0.129 | 0.48x LOSS | 2640 to 2827 |
| decode32 | fp16 | 1.379 / 0.770 | 0.284 / 0.249 | 0.813 / 0.467 (cudnn:sliced) | 1.145 / 1.045 | 0.143 / 0.131 | 0.51x LOSS | 2587 to 2947 |

Rerun of the encoder layer in fp32 after the left factor copy for cuBLAS (another process, fresh
cache): Truestar 0.755 / 0.748, graphs 0.752 / 0.750, torch.compile default 0.669 / 0.662, eager
(efficient) 0.672 / 0.666, max-autotune 0.681 / 0.676: still a **loss**, 1.13x.

* **Wins:** the encoder layer in TF32 (1.21x), bf16 (1.37x), fp16 (1.10x with graphs; the plain fp16
  median was host noise, 0.365 against a min of 0.223); prefill in bf16 and fp16 (1.04x over
  max-autotune, 1.6x to 2.4x over eager). Every decode case beats eager by 1.5x to 3.0x with graphs.
* **Losses:** encoder layer strict float32 1.13x (fp32 attention on FMA units, head split copies);
  prefill fp32 0.74x and TF32 0.94x against max-autotune; **every decode case against
  max-autotune, 0.48x to 0.78x** (decode 32 bf16 0.277 against 0.133 ms). The decode step still runs
  attention as three kernels (scores, softmax, values) that read the cache twice, about 40 small
  kernels (rotary as neg, cat, multiply add; index checks; the mask), and an output copy per replay.
* **Roofline:** encoder layer bf16 0.227 ms is about 62 TFLOPS of 88 (70%); prefill bf16 0.232 ms
  about 51 TFLOPS (58%); encoder layer fp32 0.755 ms about 18.5 TFLOPS of a 53 TFLOPS FMA peak at the
  logged 2.95 GHz (35%); decode batch 1 bf16 0.187 ms moves 23 MB, 125 GB/s of 896 (max-autotune
  220 GB/s); decode batch 32 bf16 39 MB in 0.277 ms, 141 GB/s. Nothing above a hardware limit.
* **Errors against float64** (relative L2 of the first output): Truestar 1.4e-7 to 3.1e-7 in fp32,
  3.3e-4 to 9.1e-4 in TF32 (PyTorch's TF32 rows 9e-5 to 2e-4: Truestar's attention also uses TF32
  when TF32 is allowed), 3.4e-3 to 3.6e-3 in bf16, 4.2e-4 to 4.6e-4 in fp16 (at or below eager's).

**Never introduce:** #3, every compiled kernel and every cuBLAS choice is checked against float64 on
its first call at each shape signature, and a graph is used only after a bitwise comparison with the
verified call (TS801). #1, a graph that cannot be recorded raises (TS800) and never runs eagerly in
its place; a cuBLAS choice whose precision PyTorch has changed runs the Triton launch, never TF32 in a
strict program; a program with `allow_tf32` pinned never uses cuBLAS (found by the existing
tests/test_compiler_tf32.py: the first version gave other bits for a pinned program when PyTorch's
flag changed, since cuBLAS was timed in at one setting and not at the other). Determinism: deterministic mode never uses cuBLAS; in the default mode a
contraction's bits may depend on which implementation the timing picked, as with split K before. A
choice timed while another job loads the GPU can be the slower one later; both are correct.

**Tests can fail** (tests/test_xfmr_perf.py, each restored): folded rows that drop the batch index
failed 5 tests (the kernel layer's float64 check names the kernel); a graph replay that skips copying
its inputs failed 2; a cuBLAS call 0.1% off is caught as a kernel mismatch (permanent test).
tests/test_llm_ops.py now expects grouped query attention as reshapes and attention, not an expand.

Not verified or not done: an idle GPU rerun, Linux and torch 2.14; a split KV decode attention and
rotary fused into the projections (the kernel IR has no offset index), so decode stays behind
max-autotune; the head split copies of `nn.MultiheadAttention`; RMSNorm fused into the following
projection and a fused SwiGLU pair of contractions (the norm is about 3% of prefill GPU time and the
SwiGLU intermediate under 3 MB, so they were not built); cuBLAS for bf16 and fp16 kernels (only
float32 kernels have the cuBLAS form); paged KV attention reading the block table directly.
### 2026-11-24: cross chip fingerprint on three GPUs (`truestar fingerprint`, P41)

Branch wt/fpchip. Research items: ADVERSARIAL_REVIEW.md section 2 and fix items 1, 7 and 12;
a launch plan item. Docs: docs/CROSS_CHIP.md (new, the claim wording and the table),
docs/KERNEL_IR.md 6b, docs/AGENT_INTERFACE.md (TS740 to TS750). GPUs: RTX 5070 Ti (shared with other
jobs for the first runs), Kaggle Tesla T4 through the private kernel sharvenrane/truestar-fingerprint-t4,
RTX 5060 Laptop through the first script's files only.

* **Command.** `truestar fingerprint` (CLI and `truestar.fingerprint.run`): mlp, TransformerEncoderLayer,
  resnet18, a two layer Llama style decode step with a KV cache, at batch 1, 7, 64 and 256, in float32,
  TF32, float16 and bfloat16, for four arms (compiled deterministic, compiled default, eager,
  torch.compile), plus the first script's three cases at batch 4. Records commit, versions, driver, GPU,
  capability, SM count, settings and every launch config with its source. `truestar fingerprint compare`
  names every differing environment key and setting and exits 1 only on a deterministic row (TS743).
* **Deterministic mode fixed (#3, no silent wrong results of our own, in the sense of the claim).** It
  tuned attention by timing: at batch 256 the transformer's attention went from BN 32 to BN 64, its
  bits changed and row 0 differed between batch 64 and 256. Attention and reductions now keep their
  shape only default config when compiled deterministic (source `pinned`); contractions without split K
  and pointwise tiles stay tuned (float32, TF32, float16 and bfloat16 contractions were bitwise equal
  over their whole tuning grid here).
* **Portable inputs.** PyTorch's seeded CPU draws gave other bits on the Kaggle Linux host than on the
  Windows desktop for every case, so the first T4 run compared nothing (TS744 on 7 of 7 cases). The cases
  now draw from a splitmix64 counter hash with exact float arithmetic, pinned by `models.GOLDEN`; both
  GPUs matched it (TS750 otherwise).
* **Result, RTX 5070 Ti (sm_120) against T4 (sm_75), both torch 2.11.0 and Triton 3.7.1:** deterministic
  float32 mlp, transformer and llama_step bitwise equal at all four batch sizes (12 of 12 rows); eager and
  torch.compile 0 of 12. With the T4 image's Triton 3.6.0 the mlp still matches but the transformer (0 of
  4) and the decode step (1 of 4) do not: the Triton version is part of the condition. resnet18 differs
  in every dtype (cuDNN convolutions inside the program), TF32 differs (the T4 has none and computes IEEE
  float32, bits equal to its float32 rows), float16 differs everywhere and bfloat16 except batch 1 on mlp
  and llama_step (inferred: other tensor core instructions on Turing). Run to run equal on both GPUs in
  every arm, and across two processes on the T4. Errors against float64 in float32: 3.1e-7 to 1.4e-6.
* **Legacy rows:** the current 5070 Ti gives the first run's hashes for all three legacy cases, so the
  5060 Laptop still agrees with it there (eager transformer differs, as before); the new cases were not
  run on the laptop.
* **Found, not fixed:** asking for TF32 on the T4 gives IEEE float32 while the compile report says
  `fp32_precision: tf32`.
* **Tests can fail:** tests/test_fingerprint.py (13 tests, one skips unless `TRUESTAR_FP_OTHER` names a
  fingerprint from another GPU; the 11 that existed at commit a73d121 also passed on the T4). Counting
  attention, softmax rows and loops as config free makes the pin test fail; restored. Touched test
  files: tests/test_fingerprint.py and tests/test_codes_unique.py, 15 passed, 1 skipped; a full suite
  run on commit 335fc92 gave 2263 passed, 18 skipped (the coordinator runs the full split suite after
  merging).

Not verified: Ampere, Ada, Hopper and datacenter Blackwell, AMD, Apple, Intel, CPU rows, other drivers on
one GPU, torch other than 2.11.0, the new cases on the 5060 Laptop, the per kernel cause of the float16
and bfloat16 differences.
### 2026-11-24: real multi GPU ranks: data parallel, tensor parallel, sharded state (P39)

Branch wt/multigpu. Docs: docs/PARALLEL.md (Real ranks), docs/TRAINING.md (Multi GPU),
docs/AGENT_INTERFACE.md (codes TS820 to TS825). Hardware: Kaggle GPU T4 x2 (private kernel
sharvenrane/truestar-multigpu-t4x2, two Tesla T4, NCCL 2.28.9, torch 2.11.0+cu130, versions 1 and
2), and the desktop RTX 5070 Ti with the laptop RTX 5060 over the wireless network (gloo, torch 2.11.0+cu128). All
code written from the definitions; nothing taken from PyTorch's distributed wrappers or other
frameworks, and no other project's names used for the API.

* **Real ranks.** `DeviceComm` (devices.py) gives the layout runtime real collectives between
  processes; `--order fixed` sums as an all gather and a rank order add (bitwise the same on every
  rank and every run), `--order native` uses the backend's all reduce. `truestar parallel --ranks
  devices` spawns one process per GPU or joins ranks on several machines (`--dist-master`,
  `--dist-world`, `--dist-rank`); rank 0 runs the float64 single device arm on its GPU and writes the
  report (TS820 names every rank's host, GPU and backend). Fewer GPUs than ranks is TS821 (two ranks
  never share a GPU), a rendezvous that does not complete is TS822, mixed GPU models or native order
  over 3 or more ranks is TS825.
* **Data parallel:** fixed buckets started from post accumulate gradient hooks during backward, in
  bucket order on every rank; accumulation disarms the hooks for all but the last micro batch.
  **Sharded state:** layout key `shard=1` (optimizer state) and `shard=2` (parameters too, gathered
  for the step), reduce scatter, slice norm, optimizer on the slice, all gather; new fault
  `shard_stale_params` named as TS687 at the parameter all gather. **Native step:**
  `train_step(data_parallel=comm, accumulate=k, check_replicas=n)` builds a gradient program and an
  update program with the summed loss and gradients in between; model state, random ops and float16
  loss scaling are refused (TS823); replica fingerprints (TS824).
* **Results on 2 T4s:** 2 rank step bitwise equal to one T4 on the same reduction tree in 8 of 8
  settings (ga 1 and 2, both orders, overlap on and off), run to run and across ranks; against the
  whole batch on one T4, first step gradients 3.6e-7 and parameters after 4 AdamW steps 4.8e-6.
  Sharded AdamW state bitwise equal to replicated with and without clipping at 52.5 MB against 105.0
  MB per rank. Tensor parallel MLP block within 4.4e-6 of one T4. Native step bitwise equal to one T4
  with `accumulate=2`. Layout check on real ranks: 14 of 14 clean (worst 3.5e-11), 11 of 11 injected
  faults named with the same code, module and collective as on threads. tests/test_multigpu.py 28
  passed there, including the two GPU tests.
* **Speed (provisional):** 13.1M transformer, 4096 tokens: 115.7 ms on one T4, 65.7 ms on two (1.76x;
  overlap saves 3 to 6 ms; fixed order costs about 2 ms). Tensor parallel MLP block 1.97x. Native step
  on a small MLP 0.59x (slower: two executor calls and host side sums). Desktop plus laptop over
  wireless network: correct (dp=2 and tp=2 sp=1 layout checks clean, ranks bitwise equal to each other) but 75x
  slower than one GPU; not bitwise equal to the desktop's same tree run (1.3e-5), as expected for two
  GPU models.
* **Found and fixed on the way:** the desktop's host name resolves to its WSL adapter (172.30.32.1)
  first, so gloo bound an address the laptop cannot reach and the first two machine run hung to its
  timeout; `join` now pins `GLOO_SOCKET_IFNAME` to the interface that routes to rank 0 (with
  psutil). The update program of the split native step kept the batch symbol of the gradient
  program and failed to bind on the compiled executor (Kaggle run 1); the pruned program now keeps
  only the symbols it uses.
* **Tests can fail:** dropping the last gradient bucket fails 3 tests, skipping the reduce scatter of
  the sharded update fails 7, dividing the native sums by the micro batch count only fails 1;
  restored.
* Tests run here: tests/test_multigpu.py (21 passed, 4 skipped without two GPUs), tests/test_parallel.py,
  tests/test_codes_unique.py.

Not verified: more than 2 GPUs, NVLink, InfiniBand, multi node NCCL, NCCL hangs (P10), PyTorch's own
data parallel wrapper on the same machine as a speed baseline, per layer parameter gathering,
pipeline parallel, data parallel with batch norm statistics or dropout on the native step, float16
loss scaling with data parallel, the native step on a large model.
### 2026-11-24: Metal backend for the kernel IR (P40)

Branch wt/metal. A third code generator next to Triton and numba: `kernel/metal_backend.py` writes
Metal Shading Language from the kernel IR (no code from MLX, PyTorch's MPS kernels or tinygrad), and
`torch.mps.compile_shader` compiles it at run time and launches it on the MPS tensors in torch's
command stream. Chosen over PyObjC (`pyobjc-framework-Metal` installs cleanly, but needs the
MTLBuffer behind each MPS tensor through an undocumented pointer and its own synchronized queue) and
over a compiled helper (a toolchain at install time): it needs nothing beyond the torch wheel.

* Loop shapes: fusion uses the GPU plan's structure analysis, so "metal" fuses the same kernels as
  "cuda". Tile kernels run as `pointwise` (a thread per element) or `rows` (a threadgroup per row,
  reductions folded in a fixed order: strided per thread, then a fixed tree in threadgroup memory);
  contractions as 32 x 32 tiles with SIMD group matrices (8 x 8 multiply accumulate) or, where a probe
  says the device cannot, plain multiply adds; attention, conv and window kernels as `loops` (a
  thread per element, serial reductions). Float32 arithmetic, `#pragma METAL fp math_mode(safe)`,
  `precise::` functions, explicit NaN and infinity rules, an own `erf` (5.7e-7 relative in float32
  emulation). No atomics: bitwise equal run to run.
* `K.compile(k, "metal")`, `compile_program(prog, "metal", dtype)` and `CompiledExecutor("metal")`;
  `truestar check --backends metal` runs Truestar's own compiler as a stack (torch.export, IR,
  compiled executor) and reports ops that took the CPU reference path as TS400. New codes TS840 to
  TS845 (Metal unavailable, float64, sentinels, compiler rejection, SIMD group matrices unusable,
  32 bit limits).
* CI (manual, macos-15, torch 2.14.1, about one minute each; dispatched through the branch's
  `backends.yml` because `metal.yml` is not on the default branch yet, restored afterwards):
  45495094844 (19 failed, 54 passed: every failure was the torch copy below), 45495800195 (the
  probe script crashed and skipped the tests; the end to end script passed), 45495989149 (2 failed,
  72 passed: an `exp(100)` test input that overflows float32 in every backend, and `rsqrt(-0)` that
  must be minus infinity), 45496431034 (1 failed, 75 passed: a test expected the float32
  cancellation example to be localized, but the compiled executor refuses that kernel on every
  device), 45496817336 (76 passed, 1 skipped by design).
* End to end against float64 (docs/METAL.md, section 4): MLP float32 1.4e-7 (MPS eager 2.2e-7),
  float16 2.9e-4 (3.1e-4); 2 layer transformer d 64 seq 32 float32 3.2e-7 (2.3e-7), float16 6.6e-4
  (6.1e-4); d 128 seq 128 float32 3.9e-7 (2.3e-7), float16 6.4e-4 (6.3e-4). Time per call, Metal /
  MPS eager: 0.69 / 0.84 ms, 0.79 / 0.83, 1.75 / 2.99, 1.85 / 2.88, 8.49 / 2.47, 8.53 / 2.24.
  Provisional: a virtual machine, one run each.
* Found on the way: on torch 2.14.1 on that runner `t.to("cpu", torch.float64)` of an MPS float32
  tensor returns zeros or values no float32 can hold, at all 15490 elements tried
  (`scripts/mps_copy_probe.py`); `t.cpu().to(torch.float64)` is right, and casting copies to MPS
  were right. Truestar's references and comparisons now copy first and widen on the CPU whenever
  MPS is involved. Not yet reported upstream.
* Never introduce: no silent fallback (a machine without Metal raises TS840; float64 is refused
  with TS841 before any launch; `metal` in `truestar check` was first routed to eager MPS under its
  name because `check.py` lists stack kinds, fixed and guarded by a test); no new tolerance (the
  kernel layer's per element check unchanged); no install step beyond torch.
* Tests run on the PC (Windows, no Metal): tests/test_metal.py (28 pass, the Mac only tests skip
  with the reason) and tests/test_codes_unique.py. The full suite was not run here.

Not verified: a physical Mac (M1 to M5), macOS 14 or 26, torch other than 2.14.1, Metal kernels
for training steps (the native training tests, sentinels), resnet18 and other convolution models on
Metal (the `loops` shape will be slow), anything above 32 bit sizes.
### 2026-11-24: AMD GPU target (ROCm, HIP) for the kernel IR (P43, docs/ROCM.md)

Branch wt/rocm. No AMD GPU was used: the laptop's Radeon 780M was busy and not cleared for use, and
no cloud account was touched. Everything below is compile only.

* Triton 3.7.1 carries its AMD backend in the Windows wheel of the ml venv and in the Linux wheel,
  and lowers to an AMDGPU code object with no ROCm installed:
  `triton.compile(ASTSource(fn, signature, constexprs, attrs), target=GPUTarget("hip", arch, wave))`.
  On the CPU server a venv `/srv/truestar/venv-rocm` was made for it (Triton 3.7.1 only; torch and
  numpy come from `/srv/truestar/venv` through a .pth file). No ROCm build of torch was installed.
* `truestar.kernel.rocm`: a target table (gfx942 MI300X: wave 64, 304 CUs, MFMA, xf32; gfx1100 RX
  7900 XTX: wave 32, 96 CUs, WMMA; gfx1103 Radeon 780M: wave 32, 12 CUs, WMMA; 64 KiB of LDS per
  workgroup; scratch per lane from LLVM's largest scratch per wave), `current_target()` from the
  device on a ROCm build, and `build(kernel, arch, shapes)` that plans the launch with the target's
  CUs and LDS (`triton_backend.target_limits`, new), compiles every Triton function of it and reads
  LDS, VGPRs, spills and scratch from the code object. Codes TS880 to TS886.
* Compile matrix (`scripts/rocm_compile_matrix.py`, 19 cases of `rocm_cases.py` covering tile
  pointwise, single pass and looped reductions, contraction in strict float32, TF32, float16 and
  bfloat16 with and without split K and determinism, attention with head dims 64 and 128 in float16
  and 64 in float32, conv at stride 1 and 2 in float16 and float32, and window pooling), times
  gfx942, gfx1100, gfx1103: 57 of 57 default launches compile, every LDS within 64 KiB, none spills.
  49 s on the ml venv (Windows), 52 s on the CPU server (Linux, `nice -n 19`). With `--candidates`,
  347 tuning candidates: all compile, none over the LDS limit; 158 spill, all on RDNA3 (linear 68,
  conv 46, attention 44), none on gfx942.
* Wave size: the defaults derived from the NVIDIA measurements spill on wave32 RDNA3 in 11 of the
  13 dot cases per target (every linear and attention case, the float32 conv; 188 to 3076 bytes of
  scratch per lane) and in none on wave64 gfx942. A 4 wave program has 128 lanes on RDNA3 and 256 on
  CDNA for the same tile. `rocm.relief_ladder` (more waves up to 1024 lanes, then fewer stages, a
  shorter BK, narrower then shorter tiles, SPLIT_K never changed) gives a spill free config for
  all 22; a forced 64 x 64 x 64 float16 config with 4 waves spills on gfx1100 and the relieved one
  (8 waves) does not (tests/test_rocm.py). On hardware the same ladder runs from Triton's HIP
  `LOCAL_SIZE_BYTES` in `CompiledKernel._prepare` for default configs only (TS886).
* MFMA: dot kernels on gfx942 compile with `matrix_instr_nonkdim=16`, as ROCm's Triton guide advises
  for MI300X GEMMs; not timed. TF32: allowed float32 dots compile as xf32 on gfx942 and stay strict
  IEEE on RDNA3 (TS884), since Triton allows `tf32` on gfx942 only.
* CU aware defaults: the SM aware thresholds read the CU count (304 CUs split the small M strict
  float32 linear in 4, 12 CUs follow the few SM rules and split it in 8); deterministic mode keeps
  SPLIT_K 1 at both. LDS: the shared memory budget is the target's 64 KiB minus 3 KiB, so
  `_fit_shared` trims stages; a forced 128 x 128 x 64 strict float32 config with 3 stages is refused
  with TS882 before any launch.
* Device plumbing: `K.compile(..., "rocm")`, `CompiledExecutor("rocm")` and `compile_program(...,
  "rocm")` run on torch `"cuda"` tensors of a ROCm build and refuse with TS883 without one;
  `rocm+inductor` joins the check stacks; `truestar check --backends rocm` existed already. The
  TS700 local memory check uses the AMD scratch limit on a ROCm build. Float64 verification and the
  deterministic contraction rules are the same code as on NVIDIA.
* Hardware validation, written and not run: `scripts/rocm_validate.py` (environment, compile matrix
  for the attached arch, kernel tests on the GPU, `scripts/laptop_fingerprint.py` in three dtypes,
  `bench_matmul.py` in float32 and float16) and the manual workflow `.github/workflows/rocm.yml`
  for a self hosted runner labelled rocm. docs/ROCM.md, section 5, has the AMD Developer Cloud steps.
* Tests: tests/test_rocm.py, 103 (81 pass and 22 skip here: the hardware group skips with "needs an
  AMD GPU and a ROCm build of torch"; one test runs the launcher's relief path on the RTX 5070 Ti with
  the AMD target faked and the default reported as spilling, and checks the relieved config is
  launched and verified); on the CPU server (no GPU) 80 pass and 23 skip. The hardware group was
  also run once on the RTX 5070 Ti with "rocm" replaced by "cuda", to check the tests themselves.
  Kernel test files with the codes, agent and stacks tests: 458 passed, 5 skipped (595 s).

Not verified (no AMD GPU): every value, the float64 checks on an AMD GPU, determinism and speed on
it, the relief ladder on hardware, the CU aware rules and the MFMA option as speed, `rocm+inductor`,
CUDA graphs, sentinels and the memory planner on ROCm, archs other than gfx942, gfx1100 and gfx1103.
### 2026-11-24: split KV decode attention (`KV_CHUNK`)

A decode step ran one attention program per batch and head, each walking the whole KV cache alone,
so a small batch left most SMs idle. Now a launch of up to 16 query rows can split its keys into
chunks (`KV_CHUNK`), one program each storing a float32 running max, sum and accumulator, and a
second generated kernel (`kern_splitkv`) merges them in the fixed order 0, 1, 2, ... (docs/KERNEL_IR.md).
Measured on the RTX 5060 Laptop GPU (26 SMs), Windows, torch 2.11, provisional (a laptop GPU:
clocks move with temperature).

* `scripts/bench_decode_attention.py` (new): q [B, 8, 1, 128] against k, v [B, 8, N, 128], CUDA
  events around each call, 10 warm up calls, 5 interleaved rounds of 25 timed calls, median and
  minimum, every output against float64. Before is main at c41fd02 by the same script.
  Medians in us (minimum in parentheses where it tells a different story):

| dtype | B | N | before | after | eager SDPA | torch.compile max-autotune | after / SDPA | after / compile |
|---|---|---|---|---|---|---|---|---|
| float16 | 1 | 2048 | 94.6 | 66.4 (19.9) | 127.1 | 139.8 | 0.52 | 0.48 |
| float16 | 1 | 4096 | 181.7 | 79.4 (30.1) | 246.1 | 272.2 | 0.32 | 0.29 |
| float16 | 1 | 8192 | 357.4 | 71.1 (48.6) | 486.7 | 687.6 | 0.15 | 0.10 |
| float16 | 8 | 1024 | 111.5 | 69.8 (46.2) | 192.9 | 430.7 | 0.36 | 0.16 |
| float16 | 8 | 4096 | 538.9 | 379.3 | 835.8 | 1602.0 | 0.45 | 0.24 |
| float16 | 8 | 8192 | 1196.7 | 961.8 | 1546.5 | 3601.6 | 0.62 | 0.27 |
| bfloat16 | 1 | 8192 | 358.6 | 91.6 (48.6) | 487.1 | 699.6 | 0.19 | 0.13 |
| bfloat16 | 8 | 8192 | 1197.8 | 793.3 | 1622.9 | 3306.0 | 0.49 | 0.24 |
| float16 | 32 | 4096 | 1924.2 | 1615.1 | 3199.7 | 6227.7 | 0.50 | 0.26 |

  Full table (60 rows) in results/laptop-splitkv/. Split cases: 0.20 to 0.83 of before by the median
  (0.14 to 0.83 by the minimum). Batch 32 does not split (256 programs fill the GPU) and keeps
  exactly its old config; its rows moved by 0.83 to 1.22 with the same config, which is this GPU's
  noise. Against eager SDPA Truestar is 0.15 to 0.72 of its time everywhere except batch 1 at 512
  and 1024 keys (1.42 to 0.85: a call there is about 50 us of launch latency, where SDPA's single
  launch wins); against torch.compile max-autotune 0.10 to 0.62 everywhere. Inductor reported "Not
  enough SMs to use max_autotune_gemm mode" on this GPU, so its max-autotune is weaker here than on
  70 SMs; the desktop numbers (0.48 to 0.78 behind max-autotune before) need a rerun there.
* Found by the first measurement and fixed: splitting every small decode made batch 1 at 512 and
  1024 keys and batch 8 at 512 keys slower (60.7 to 68.1, 47.6 to 71.1, 58.5 to 72.0 us), since the
  second launch costs about 20 us of host time while those calls are launch bound. A decode now
  splits only when one program would walk at least 2048 keys in a row (`KV_MIN_SERIAL_KEYS`), and a
  decode that does not split keeps its old blocks: the 16 row tile alone had made batch 32 bfloat16
  at 8192 keys 13% slower.
* Correctness: tests/test_splitkv.py (29): every chunk size 0 to 1024 in float16, bfloat16 and float32
  against float64 over 1000 keys (a ragged last chunk), 1, 4 and 16 query rows with LPT on and off,
  an index mask and a data padding mask, chunks whose keys are all masked (finite, no NaN),
  differential attention (two softmaxes split together), run to run equality in both modes, the
  verifier catching a broken merge, and the config rules. Split also shortens each sum: float32
  error at 4096 keys and batch 8 was 2.1e-6 unsplit, 1.7e-6 with the default 2048 key chunk and
  9e-7 with 256 key chunks (deterministic mode).
* Determinism: deterministic mode chunks by `DET_KV_CHUNK` (256 keys) from 2048 keys of context, on
  any GPU: tests show the same config and bits with the SM count pinned to 26 and to 70, and a row
  decoded alone equal to the same row decoded in a batch of 4. `scripts/laptop_fingerprint.py`
  (float32, float16, bfloat16) is unchanged in 9 of 9 compiled hashes (no fingerprint model decodes).
* Paged KV and grouped query heads: both reach the kernel already gathered and expanded (unchanged),
  so they split the same way; tests/test_paged_kv.py still shows paged and dense decode bitwise equal.
* Full suite: 2249 passed, 30 skipped; 2 failed and 8 errors, all in tests/test_conform.py with
  `No module named 'onnxscript'`, identical on main without this change (the package is not in this
  machine's environment).

Not verified: 70 SMs (the desktop), head dims other than 128 in the benchmark, a split over keys
that reads the paged block table directly (the gather is still a separate step), CUDA graph capture
of a full decode step, and the tuner's split candidates (defaults only were measured).

### 2026-11-24: quantized KV cache for decode (`truestar.kvquant`, int8 and fp8)

K and V stored as int8 or fp8 e4m3 with a float32 scale per head and per group of tokens (16 to 128),
dequantized in registers inside a split KV decode kernel (float32 accumulation, fixed order merge,
grouped query heads read their KV head directly). Docs: docs/KV_QUANT.md. RTX 5060 Laptop GPU (26
SMs), provisional.

* KV memory per token, 8 KV heads of dim 128, groups of 32: float16 4096 bytes, int8 and fp8 2050
  (50.05%, scales included).
* `scripts/bench_kvquant.py` (new), q [B, 8, 1, 128] float16, CUDA events, 5 x 25 calls, median
  (minimum) us:

| B | N | int8 | fp8 | Truestar split KV, float16 cache | eager SDPA, float16 cache |
|---|---|---|---|---|---|
| 1 | 2048 | 72.1 (17.8) | 69.4 (15.8) | 65.4 (19.9) | 127.0 (122.3) |
| 1 | 8192 | 94.7 (50.6) | 93.0 (63.9) | 90.7 (48.5) | 484.9 (480.7) |
| 8 | 2048 | 82.7 (74.1) | 78.8 (73.1) | 196.4 (190.7) | 414.1 (389.2) |
| 8 | 8192 | 490.8 (386.5) | 488.8 (380.8) | 956.8 (739.6) | 1601.3 (1541.6) |

  At batch 8 the halved cache reads make int8 and fp8 2.0x to 2.4x faster than the float16 split KV
  and 3.3x to 5.0x faster than SDPA. At batch 1 they tie with the float16 split KV: those calls are
  launch bound (about 70 to 95 us median against 16 to 64 us minimum).
* Error against float64 attention over the unquantized cache, largest absolute: int8 8.2e-4 to
  2.4e-3, fp8 2.2e-3 to 7.3e-3, float16 cache 2.4e-5 to 5.6e-5 (about 36 and 98 times the float16
  cache at 4096 keys). The documented bound (`error_bound`, proven from the cache's own rounding
  errors, docs/KV_QUANT.md) holds on every element in tests, including peaked scores; it is 40 to
  160 times the measured error. Against float64 over the dequantized cache (the kernel's own
  rounding): 2.2e-4 to 3.4e-4 relative; this check runs on every first call.
* Storage bounds tested per value: int8 within half a step plus float32 rounding of x / s (found
  while testing: a value at a half step exceeded s / 2 by 2.6e-6 relative), fp8 within 2^-4 in the
  normal range and 2^-10 of the scale below it.
* Determinism: run to run equal; deterministic chunks (256 keys from 2048) give the same bits with
  the SM count pinned to 26 and 70 and for a sequence alone or in a batch; paged and dense caches
  give the same bits. New code TS701 (`kv_quant_unsupported`). Fingerprint (9 compiled hashes)
  unchanged; the kernel layer is not touched.
* Tests: tests/test_kvquant.py (26), with tests/test_codes_unique.py and tests/test_agent.py.

Not verified: the compiled executor does not choose the quantized cache yet (a library call); per
channel scales; GPUs other than this one; head dims other than 128 in the benchmark.
### 2026-11-24: op and layer coverage against PyTorch (P42)

Branch wt/coverage. Docs: docs/MODEL_COVERAGE.md (the measurement and the per op and per model
tables), docs/IR.md, docs/AUTODIFF.md, docs/TRANSFORMS.md, docs/COMPILER.md, docs/NN.md,
docs/AGENT_INTERFACE.md (codes TS860, TS861). torch 2.11, torchvision 0.26, transformers 4.48, ml
venv, RTX 5070 Ti. Nothing installed, nothing downloaded.

* **Measured honestly first.** `scripts/op_coverage.py` exports every model of a list (the 63
  torchvision classifiers under 100M params, 15 Hugging Face config models: the 7 before plus
  Whisper, CLIP text and vision, DeiT, Swin, ConvNeXt, MobileViT, Mamba; 92 torch.nn layers and
  losses, `truestar.testing_ops`), imports it, checks it against PyTorch float64, compiles it on CUDA
  and checks one training step's gradient against torch.autograd, and reports for each of the 161
  ATen ops: import, the IR ops produced, float64 reference, reverse, forward and batching rules and
  the compiled path. The before run is the same script on a copy of 7aca8a9.
* **Before / after** (results/op_coverage/before.json, after.json): models that import 103 / 170;
  ATen ops imported 108 / 161, with every reverse rule 88 / 161, forward 103 / 161, batching 108 /
  161, compiled natively only 96 / 151, able to take the CPU fallback 10 / 0; models compiled with no
  fallback 86 / 170; training steps equal to torch 70 / 169 (torchvision 50 / 63, Hugging Face 3 /
  15, torch.nn 17 / 91 of 92).
* **Import** (`ir/importer_extra.py`, 53 ATen ops, all lowered onto existing IR ops): 1D, 3D and
  transposed convolutions, 1D and 3D pools, interpolation (nearest, linear, bilinear, trilinear,
  bicubic from constant interpolation matrices), reflect, replicate and circular padding, shuffles,
  unfold and fold, RNN, LSTM and GRU layers and cells (unrolled), 13 activations (expm1 and log1p
  composites accurate near 0), embedding bags (offsets, padding, max), bilinear, distances and 9
  losses. Creation ops without a dtype take the dtype torch.export recorded.
* **Rules** (`ir/grad_rules_extra.py`): reverse rules for `index`, `pow`, `cast`, `log_softmax`,
  `pad`, `slice_scatter`, `select_scatter`, `avg_pool2d`, `cumsum`, `triu`, `tril`, `maximum`,
  `minimum`, `fmod`, `remainder`, `cos`, `sin`, `log`, `abs`, `reciprocal`; lowerings for
  `group_norm`, `rms_norm`, `vector_norm`, `adaptive_max_pool2d`; a forward rule for `fmod` and
  `remainder`; a new op `frozen` (no derivative). 31 new finite difference cases, each also in the
  forward mode dot product test and the vmap test.
* **Compiler** (`compiler/coverage_rules.py`): kernels for `triu`, `tril`, constant `pad`, `flip`,
  `slice_scatter`, `select_scatter`, `pow` with any constant exponent, float `maximum` and `minimum`
  (NaN propagates); norms compiled through their lowerings; integer casts, `argmax`, `argmin` and
  integer reductions as exact device steps.
* **Layers** (`nn/layers_more.py`): GroupNorm, InstanceNorm, Conv1d, ConvTranspose2d, Upsample,
  PixelShuffle, LSTM and GRU over an IR loop (one compile for every length), EmbeddingBag, Rotary,
  ALiBi, SwiGLU, GEGLU; new native ops `ops.conv_transpose2d`, `ops.group_norm`. 25 tests at 1e-12
  against torch.nn (eager, compiled, gradient), rotary against transformers' Llama code, ALiBi
  against the paper's slopes.
* **Wrong gradients found and fixed:** the importer dropped `padding_idx` and `scale_grad_by_freq`
  of embeddings (BERT's word embedding gradient 0.36 off in the padding row), and imported
  hardtanh and relu6 as clamp, whose gradient passes the bounds while torch's hardtanh sends none
  (mobilenet_v2 at its default init feeds 260 thousand exact zeros into ReLU6: 0.35 off). Both now
  equal torch.autograd. Found on the torch side: the float64 CPU backward of `hardsigmoid`
  multiplies by float32 1/6 (3e-8 relative).
* **Tests run** (the rule of this branch: touched files plus the shared list): test_codes_unique,
  test_ir_ops, test_hf_models (67, with a training step test per model and no fallback asserted),
  test_vmap, test_forward, test_autodiff, test_op_coverage (new, 381: every layer's value, gradient,
  vmap and forward mode), test_compiler_coverage (new, 25), test_compiler_elementwise_more,
  test_nn_more (new, 25): 1307 passed, 7 skipped. Three existing tests were changed because they asserted
  a refusal that is now support (reflect padding, pow with exponent 1.7, `fmod` without a forward
  rule); each now checks the support and keeps a refusal of what is still unsupported. The autodiff classification test lists the four new lowerings.
* **Tests can fail:** a bicubic matrix scaled by 0.98 and a GRU cell replaced by a tanh RNN cell
  fail the value test; a `triu` kernel that keeps one more diagonal is refused by the first call
  check; naive `exp(x) - 1` and `log(1 + x)` fail the composite accuracy test.

Not verified or not built: the torchvision zoo compiled on the CPU (the Hugging Face and torch.nn
models were: 107 of 107 with no fallback), pretrained weights, models over 100M params, the core
ATen graph of the new models, symbolic sequence lengths in imported recurrent layers (TS861; the
native LSTM and GRU layers loop over any length), bidirectional or multi layer native recurrent
layers, rotary and ALiBi as options of MultiHeadAttention, data dependent loads in the loop IR (the
10 ops that keep a torch device step), macOS and Linux.

### 2026-11-24: quantized KV cache in a compiled decode, per channel key scales

Follow up of the quantized KV entry above (RTX 5060 Laptop GPU, provisional). Docs: docs/KV_QUANT.md.

* Compiled executor, opt in only: `examples/llama_kvquant_decode.py` `make_decode_step(model,
  kv_cache=None | "int8" | "fp8")`. None is the unchanged dense step (fingerprint 9 of 9 compiled
  hashes unchanged). A format keeps per layer a quantized prefix of whole groups plus a float tail;
  the step writes the new token into the tail and calls `torch.ops.truestar.kvq_attention`, imported
  as the new IR op `kvq_attention` (shape rule, float64 reference over the dequantized cache and live
  tail) and run as a library step through the split KV kernel, which reads the tail as one more chunk
  merged last. The host quantizes each full tail once, so no value is quantized twice. The first call
  check against float64 caught a wrong softmax scale while building it (relative error 0.44, nothing
  returned). `layer_error_bounds` reports the proven bound per layer.
* End to end decode (`scripts/bench_kvquant_decode.py`, the scripts/bench_decode.py model, 32 tokens,
  float32 programs), tokens per second: batch 1 at 2048 / 8192 keys, float32 cache 198 / 177, int8
  173 / 208, fp8 201 / 198 (all host bound at about 5 ms a step); batch 8, float32 1017 / 422, int8
  1407 / 1452, fp8 1479 / 1490 (1.4x to 3.5x, part of it the dense step copying its whole cache each
  step). KV memory per token 4096 bytes float32, 2048 for a float16 cache, 1026 int8 and fp8. Found:
  the float16 dense program failed Truestar's own end to end check on these random weights (0.085
  against 0.066 after one step) and returned nothing, so every program ran in float32.
* Per channel key scales (`quantize(..., channel=True)`, one per head, channel and block of tokens),
  compared at equal memory (blocks of 4096 against groups of 32, `scripts/kvquant_channel_study.py`):
  int8 per channel wins 4 of 4 cases with 4 outlier channels (3.4x to 4.4x lower error), 1 of 4 on
  normal keys, 3 of 4 with drifting magnitudes; fp8 per channel loses every outlier and normal case.
  It does not win every case, so per token groups stay the default and per channel keys are an option.
* The split KV kernel gained a float tail (one more program per row, merged after the quantized
  chunks) and separate K and V groups.
* Tests: tests/test_kvquant.py (30: 4 new on per channel keys), tests/test_kvquant_decode.py (5),
  tests/test_examples.py (the new example), tests/test_ir.py, tests/test_paged_kv.py,
  tests/test_codes_unique.py; no new code (TS701 covers the refusals).

Not verified: a trained model's keys (all data random), prefill through the quantized path (prefill
runs the float dense step), sequences of different lengths in one batch, other GPUs.
### 2026-11-24: compiled training speed and memory (P37, `CompiledTrainStep`)

Branch `wt/trainperf` (on `wt/conv`). Docs: docs/TRAINING.md (Compiled, with every table),
docs/COMPILER.md (Training programs; ill conditioned outputs under Verification), docs/MEMORY.md
(donated input first), docs/AUTODIFF.md (store what the backward reads), docs/KERNEL_IR.md
(`expm1`). Codes TS780, TS781, TS782. RTX 5070 Ti shared with other jobs, torch 2.11, Triton 3.7.1,
Windows; every number provisional until the idle GPU rerun.

Profile first (`scripts/bench_train.py --profile`): the MLP step (AdamW) spent 72% of its GPU time
in two fallbacks (`log_softmax` from cross entropy 0.94 ms, `erf` from the gelu gradient 0.87 ms,
each a float64 host round trip) and was host bound (70 steps, 3.7 ms of issue against 0.9 ms of
GPU work). The transformer step has 1170 steps (783 kernels, 427 layout views), 48 ms of host issue
per call; 370 of its kernels are the optimizer's five per param (non finite check, norm, two moments,
update) over 74 params. A compiled AdamW step failed its kernel check (bias correction cancels in
float32), so it could not run.

Changes:

* `log_softmax`, `erf`, `expm1` lowering rules; `expm1` added to the kernel IR (all four backends)
  and to the IR (shape, reference, gradient, forward rule, vmap, importer, finite difference case);
  Adam's `1 - b^t` is `-expm1(t ln b)`.
* The relu gradient reads `y > 0` from its output, so the input is not stored (TinyLM gradient
  program peak 4.12 to 3.47 MiB, the six layer LM step 128 to 104 MiB).
* End to end check: the default stays strict. A float32 output over the bar is refused as before,
  and the error now also gives the distance from float64 of an independent float32 evaluation of the
  same program (the reference interpreter in float32, never the compiled kernels). Opt in with
  `accept_ill_conditioned=True`: outputs within 4x of that distance are accepted, listed in
  `cp.conditioning` and named in a TS782 warning. Needed for AdamW's first step: eager float32
  PyTorch is 6.8e-4 and 2.6e-3 from float64 on two MLP weights, over the 6.25e-4 bar. With the
  option on, a 1% corruption of one output after the region checks is still refused (test).
  float16 and bfloat16 keep the strict bar.
* Memory schedule: a program output written over a donated input takes it before a dead
  intermediate, so carried values keep their addresses (`Schedule.donated_first`); same peak.
* `CompiledTrainStep`: params and state on the device, donated every step, `graph=True` captures
  one CUDA graph after the checked call and compares its first replay bit for bit with a plain call
  (TS780); carried values not written in place are copied back inside the graph; a batch of another
  shape raises TS781.
* `conv2d_weight_grad` split into row chunks summed in a fixed order (deterministic, recompute stays
  bitwise): 1.24 to 0.26 ms on a 64 channel 56 x 56 layer at batch 8, 6.81 to 0.93 at batch 32.
* `param_group=True`: one flat buffer per carried value, the update one kernel per value, two
  stage reductions for the norm and the non finite check (a single program over 7.3M entries took
  2.9 ms).

Measured (AdamW, ms median, float32 on every side, Truestar with `accept_ill_conditioned=True`;
full tables in docs/TRAINING.md):

| model | batch | eager | torch.compile | max-autotune | Truestar | Truestar graph | peak eager / Truestar (MiB) |
|---|---|---|---|---|---|---|---|
| MLP | 64 | 1.10 | 1.18 | 0.92 | 2.01 | 0.64 | 49.9 / 32.3 |
| MLP | 256 | 0.97 | 0.99 | 0.94 | 1.61 | 1.15 | 52.1 / 33.8 |
| LM | 8 | 8.19 | 5.52 | 2.84 | 26.36 | 5.26 (group 4.92, its eager 6.66) | 113.8 / 104.0 |
| resnet18 | 8 | 6.94 | 9.85 | 9.06 | 20.84 | 13.84 | 240.5 / 190.1 |

SGD, before (`a1423a9`) and after, Truestar plain: MLP 64 3.63 to 1.56 ms (graph 0.55), MLP 256
4.06 to 2.13 (1.07), LM 8 29.78 to 20.90 (5.15), resnet18 8 31.62 to 22.44 (14.43), resnet18 16
59.34 to 26.09 (23.30); peaks LM 128 to 104 MiB, resnet18 252 to 191 and 469 to 338 MiB (eager 223
and 403).

Losses: max-autotune 1.7x faster on the LM; resnet18 2x slower than eager and 1.4x slower than
torch.compile (conv backward kernels); the LM peak above eager in one run (104 against 91.7 MiB);
`param_group` loses on the MLP (0.4 ms and 40 MiB more, the gradient concatenation copies), so it is
opt in; compile time 60 s (LM) and 281 s (resnet18) without the disk cache against 15.5 s and 28 s
for torch.compile; the LM at batch 16 (SGD and AdamW, before and after this work) and resnet18 at
batch 16 with AdamW are refused by the end to end near tie check (the LM's `layers.3.linear1.weight`
gradient 1.3% off float64 against a 0.21% bar, eager float32 itself 1.5% off on `layers.4`).

Not done: attention probabilities are still stored for the backward pass (no recompute from the
log sum exp); the conv input gradient was not retiled; the backward kernels do not write gradients
straight into the group buffer (hence the copy); elementwise backward chains are fused as before
(epilogues of the producing kernels), no new prologue fusion; multi output kernels (a moment and
its param update in one kernel) not built; `CompiledTrainStep` is not wired into `Trainer`; no
idle GPU rerun, no Linux, no TF32 or half precision rows.

Tests: tests/test_train_compiled.py (new, 10: kernels for the training ops with no fallback, relu
gradient source, compiled SGD and AdamW against torch.optim in float64, a wiring fault still
refused with the option on, the ill conditioned benchmark MLP AdamW step refused by default with the float32
reference distance in the error and accepted, listed and warned (TS782) with the option, graph replay bitwise and TS781, the split weight gradient deterministic, param group equal
to the per param step), tests/test_autodiff.py (expm1 finite difference case), tests/test_memory.py
(recompute budget at 55% of the now lower peak, a tighter absolute budget than before),
tests/test_train.py (op set and the broken bias correction break follow `expm1`). Runs, one file at
a time: test_train_compiled 10 passed; test_conv_kernels 124; test_memory 25; test_autodiff 132;
test_autodiff_cf 93; test_codes_unique 3; test_compiler 95 (2 skipped); test_compiler_elementwise_more
8; test_compiler_tf32 11; test_kernel 141 (1 skipped); test_kernel_gpu_verify 14; test_kernel_tune 43;
test_kernelcheck 22; test_forward 123; test_vmap 355; test_train 46; test_train_ir 9.

Tests can fail (manual breaks, restored, measured before the allowance became opt in): without the
allowance the ill conditioned AdamW test fails (ProgramMismatchError); with the copy back inside the
graph skipped, capture raises TS780 naming `step` and `4.bias.momentum_buffer` and the graph test
fails.

### 2026-11-24: why the float16 Llama decode fails its end to end check (root cause, no fix yet)

The float16 dense decode program of the quantized KV entry above failed Truestar's end to end check
(0.085 against 0.066). `scripts/probe_fp16_rope.py` compares one step with the float64 model, checks
off, at several positions (scripts/bench_decode.py model):

| position | logits rel error | new K rel error | float16 rotary angle error (rad) |
|---|---|---|---|
| 10 | 8.0e-4 | 2.1e-3 | 0.002 |
| 300 | 7.7e-4 | 3.3e-2 | 0.048 |
| 2047 | 6.2e-4 | 0.16 | 0.32 |
| 4000 | 5.7e-4 | 0.50 | 0.64 |

The new K's error grows with the position exactly as the rotary angle `pos * inv_freq` does when
computed in float16 (float16 spacing is 2 at 2048): neither a kernel losing precision nor a tolerance
that under budgets rounding. The program is wrong and the check is right to refuse it. The cause is a
design limit: the compiled executor has one float precision, so in a float16 program `inv_freq` is
stored in float16 and a model's explicit `.float()` (as Hugging Face computes RoPE) is a float16 copy
(lower.py, `_cast_rule`). Storing `inv_freq` in float16 alone moves angles at position 4000 by up to
about 2 rad, so computing the angle in float32 inside a kernel would not be enough. The fix is mixed
precision: values a model declares float32 keep float32 storage and math in a half precision program.
Not done here; proposed as its own change.

Host time of a batch 1 decode step (`scripts/profile_decode_host.py`, float32, 4 layers): about 176
plan steps a step, of them 57 compiled kernel calls making 91 Triton launches and about 100 layout,
index and range check steps; Python and launch overhead, not GPU work, so about 5 ms a step at batch 1.
CUDA graph capture of a whole step would remove it but conflicts with the per call host read of the
bounds sentinels and with value semantics (the dense step returns a new cache). Not done here.

### 2026-11-24: mixed precision islands in 16 bit programs (the float16 decode fix)

Fix for the entry above. Values a model declares wider than the program's own float dtype (at least
32 bits) keep float32 storage and math in a float16 or bfloat16 program, params read through such a
cast are stored in float32, and the cast back rounds where the model casts (docs/COMPILER.md). The
Llama example gained `rope_f32` (an explicit `.float()` for the rotary angles, as Hugging Face does;
off by default, so the example's other outputs are what they were). `scripts/probe_fp16_rope.py`,
scripts/bench_decode.py model, one step against float64, new K relative error:

| position | float16, no islands | float16 with rope_f32: 8 islands |
|---|---|---|
| 10 | 1.8e-3 | 7.6e-4 |
| 300 | 2.4e-2 | 5.9e-4 |
| 2047 | 0.32 | 9.7e-4 |
| 4000 | 0.41 | 4.1e-4 |

Without islands Truestar's checks refuse the float16 program; with them it passes, with no tolerance
changed (each kernel is judged at the lowest precision among its buffers, as before). Logits stay at
6e-4 to 8e-4 either way. Islands of that model: `model.inv_freq` (float32 storage) and the angle
chain. Both import paths form islands (a float64 import and a float16 import with `.float()`).

* Bit identity: the fingerprint's 9 compiled hashes are unchanged (no fingerprint model declares a
  wider value); tests/test_compiler*.py, test_ir, test_kernel, test_sm_aware (deterministic, 26 and 70
  SMs pinned), test_codes_unique, paged KV, split KV, quantized KV and the decode examples: 412
  passed, 3 skipped.
* Tests: tests/test_islands.py (6): refused without islands, islands and checks passing from a
  float64 and a float16 import, the new K against float64 at positions 300 and 4000, a pure 16 bit
  program reporting no islands, a float32 program untouched.

Not verified: islands inside control flow regions (a region takes its own dtype from its inputs),
library and random steps producing island values (they still produce the working precision).
* End to end decode in float16, now that it passes the checks (`scripts/bench_kvquant_decode.py
  --dtype float16 --rope-f32`, 32 tokens), tokens per second, float16 cache / int8 / fp8: batch 1 at
  2048 keys 207 / 174 / 213, at 8192 keys 142 / 222 / 213; batch 8 at 2048 keys 1630 / 1716 / 1659, at
  8192 keys 1225 / 1751 / 1739. KV memory 2048 bytes per token float16, 1026 int8 and fp8 (128.5 and
  64.9 MiB at batch 8, 8192 keys). Every row is near the 4.5 ms a step host floor, so the formats differ
  by at most 1.6x; the next change (CUDA graph capture of a whole step) targets that floor.

### 2026-11-24: whole decode step as a CUDA graph (fix in `cp.graphed()` for updates the plan cannot alias)

`cp.graphed(update=...)` refused the Llama dense decode step on main with TS801 ("updated input 2
differently from the verified call"), in float32 and float16, with and without islands. Root cause:
that step's cache update cannot be aliased in place by the plan, so the ordinary call ends with a copy
of the result into the caller's tensor (`CompiledProgram.__call__`), and the graph did not record that
copy, so a replay left the bound cache unwritten; the bitwise comparison with the verified call caught
it. Fix: `GraphedProgram._run` makes the same copy inside the recording. Regression test
tests/test_islands.py `test_graphed_llama_step_with_an_update_the_plan_cannot_alias` (5 replays equal the
ordinary call's bits); tests/test_xfmr_perf.py graph tests unchanged.

`scripts/bench_kvquant_decode.py --mode` (new: plain, inplace, graphed), float16 programs with float32
RoPE, 32 tokens, tokens per second, plain -> graphed:

| B | N | float16 cache | int8 | fp8 |
|---|---|---|---|---|
| 1 | 2048 | 205 -> 1287 | 198 -> 1584 | 214 -> 1717 |
| 1 | 8192 | 201 -> 1037 | 217 -> 1496 | 222 -> 1497 |
| 8 | 2048 | 1602 -> 6256 | 1662 -> 10488 | 1741 -> 10237 |
| 8 | 8192 | 1163 -> 1744 | 1754 -> 4591 | 1649 -> 4606 |

The 4.5 ms a step host floor falls to 0.58 to 1.28 ms at batch 1 and batch 8 with 2048 keys (6.3x to
8.0x). `update=` alone (inplace) changes nothing measurable: the copy back remains. The float16 cache at
batch 8 and 8192 keys gains only 1.5x because every step still copies the whole cache (128.5 MiB) on the
GPU, since the plan cannot alias it in place; the quantized steps copy only their float tails, so int8
and fp8 are 2.6x faster there. A quantized step's prefix grows every 32 tokens, a new signature, so its
graph is recorded again then (one capture in each run of 32 tokens).

Bit identity: fingerprint 9 of 9 compiled hashes unchanged; test_islands, test_xfmr_perf,
test_kvquant_decode, test_compiler*.py, test_sm_aware (deterministic, 26 and 70 SMs pinned),
test_codes_unique, test_paged_kv: 197 passed, 2 skipped.

Not done: aliasing the dense step's cache update in place (the remaining full cache copy per step), a
fixed capacity quantized prefix (one graph for the whole decode instead of one per group).

### 2026-11-24: the decode cache written in place (no whole cache copy per step)

The dense step's cache update could not be written in place: each layer's `index_copy` writes a view
(`kc[i]`) and `torch.stack` builds a new tensor, so every step copied the whole cache. Two changes:

* Runtime: a chain of in place writes over an `update=` input was cut after the first write. The
  planner folds each written value into its input's group, so the second `index_copy` of a chain was
  a write over a program input too, but the runtime accepted that only when the step's own input slot
  was the program input. The schedule now records, for each such write, the program input its buffer
  derives from (`Schedule.donated_root`), and the runtime accepts it when that input is in `update=`
  (the planner proved the intermediate dead after the step, as before; contiguity and offset are
  checked as before).
* Example: `FlatStep` (examples/llama_paged_decode.py) keeps the cache as one buffer
  `[L * S, B, HKV, HD]`, written by one `index_copy` per layer at rows `pos + i * S` given by the host
  (`flat_rows`, as the paged step's slot mapping). Its logits are the dense step's bit for bit, and
  with `update=` every call is in place (`last_update_inplace`). Two forms that computed the rows in
  the program are not importable yet: a slice with symbolic bounds, and a symbolic size as a tensor
  operand.

Graphed decode with `FlatStep` (`--mode graphed --flat`), float16 with float32 RoPE, tokens per
second, float16 cache: batch 8 at 8192 keys 1744 -> 4776 (1675 us a step, was 4587; the 128.5 MiB copy
is gone), batch 8 at 2048 keys 6256 -> 9303, batch 1 at 2048 keys 1287 -> 1594, batch 1 at 8192 keys
1037 -> 1084. The float16 cache now matches the quantized ones at batch 8 and 8192 keys (int8 4608,
fp8 5081).

Gates: fingerprint 9 of 9 unchanged; test_islands (the new test: in place every call, bitwise the dense
step, a graph replaying those bits), test_xfmr_perf, test_kvquant_decode, test_compiler*.py, test_sm_aware
(deterministic, 26 and 70 SMs pinned), test_codes_unique, test_paged_kv, test_memory*.py and the
examples: 225 passed, 2 skipped.

### 2026-11-24: fixed capacity quantized prefix (one CUDA graph for a whole quantized decode)

The quantized step's prefix grew by one group every 32 tokens, a new input signature, so a graphed
decode recorded a new graph each time. Now `kvq_attention` and `decode_attention` take an optional
`prefix_len` (an int32 device tensor): the prefix is allocated to its capacity and only its first
`prefix_len` keys are read (`QuantCache.from_dense(..., capacity=)`, `import_quant_step(fixed=True)`;
the host writes each full group in place at the live end). Chunks are laid over the capacity; a chunk
past the live keys stores max -inf and sum 0 and the merge adds exactly nothing, so at the same chunk
size the result is bitwise that of a cache of exactly the live keys (tested). In deterministic mode a
fixed capacity cache chunks by DET_CHUNK (256) always, since the live length is on the device: its bits
equal an exact cache's from 2048 live keys on (below that the exact cache does not split).

* One graph per batch and context for the whole decode (30 and 32 steps across group fills, tested
  bitwise against the growing step); quantized graphed decode, float16 with float32 RoPE, tokens per
  second: batch 1 at 2048 keys int8 1942, fp8 1823; batch 8 at 2048 keys 9134 and 9727; at 8192 keys
  4615 and 4657; the float16 flat step 1591, 8602, 4794.
* Found: a graph read a bound float input in the caller's dtype while the verified call converts it,
  which gave other bits; fixed as TS802 (its own commit). And the island rule's "program dtype" came
  from the first float input, which made float32 scales the program's dtype; it is now the float dtype
  holding the most elements (symbolic dims count as 1), so the scales are a float32 island, which also
  keeps them exact in a float16 program.
* Gates: fingerprint 9 of 9 unchanged; tests/test_kvquant.py, test_kvquant_decode, test_islands,
  test_xfmr_perf, test_compiler*.py, test_sm_aware (26 and 70 SMs pinned), test_codes_unique,
  test_paged_kv, test_memory*.py, test_ir and the decode examples: 295 passed, 2 skipped.

### 2026-11-24: prefix cache bitwise check (P48, `truestar prefixcheck`)

New `truestar.prefixcheck` (compare, check_files) and the command `truestar prefixcheck FULL CACHED
[--reference REF]` for files from any engine; codes TS702 (divergent: first token and layer named),
TS703 (less accurate than the full prefill against float64), TS704 (bitwise). docs/PREFIX_CHECK.md.

Truestar's own caches (`scripts/prefix_check_llama.py`, deterministic float32, RTX 5060 Laptop GPU):
dense, paged and token by token decode caches give the full prefill's bits at every split (1, 37, 64,
120 of 160 tokens and 2100 of 2200), K and V of every layer included. int8 and fp8 prefixes are named
from the first new token (TS702; fp8 also TS703, its error 1.2e-3 to 2.2e-3 against float64 where the
full prefill's is 3.5e-7 to 3.8e-7). The fused attention kernel's split KV decode is not the prefill
row's bits from 2048 keys (3.05e-5 at 2200 keys in float16, TS702 only: equally accurate) and is
bitwise below. The Llama decode is bitwise past 2048 keys because its attention runs unfused (a data
mask and grouped query heads), not through the split KV kernel.

Found on the way: a float32 decode at position 2100 with a nearly empty cache failed the end to end
check (8.6e-4 against 6.3e-4) from float32 rotary angles; the island change on laptop-fp16decode
(rope_f32 and float32 islands) does not cover float32 programs, so this stays a limit for now.

Tests: tests/test_prefixcheck.py (8): codes, bitwise, first token and layer, TS703 against a
reference, .pt and .npz files and the command's exit codes, Truestar's dense, paged, decode and int8
caches, the split KV decode named; with test_codes_unique and test_agent, 38 passed.

### 2026-11-24: lean test on rounding errors and the summation order probe (P47)

Branch wt/numprobe. Research items: LIVE_RESEARCH_2026_11.md 1.2 (rank 7) and 1.1 (rank 6). Docs:
docs/BIAS.md and docs/ORDERPROBE.md (new), docs/COMPILER.md Verification, docs/AGENT_INTERFACE.md
(TS960 to TS966; TS960 to TS979 reserved, none used on origin/main or any wt/* branch before).

* **Lean test.** Mean signed error in the size check's units, two directions (toward an infinity,
  toward zero), cluster robust t over every grouping a shared rounding may follow (from the kernel IR
  for kernels, every set of dims with at least 16 groups for modules), largest p kept; fires only
  when p is below 1e-6 and the mean exceeds 1/64 of the tolerance (5.2 u in float32, so constants
  rounded once never fire). A leaning output is measured again against the magnitude of its sums'
  terms before it is refused. Kernel verification raises `KernelBiasError` (TS960);
  `truestar check` adds TS961 per module whose own errors lean and none of whose children's do.
* **Catches** (tests/test_bias.py, each passes the size check): bfloat16 truncating store, round
  toward zero accumulation over 1024 terms, exp with a 2^-20 short log2(e), compiled CPU kernels with a
  truncating store in float32, float16 and bfloat16, and a check module that drops 8 mantissa bits
  (TS961 at p 3e-20 on that module only). Two broken kernels of the element rule tests are refused by
  the lean test alone too.
* **Zero false positives**: CPU kernel battery on a rented Linux server (50 kernel, compiler, training and other
  test files plus the check tests: 2214 passed, 806 kernel outputs tested, 0 fired, closest margin
  0.61), GPU kernel tests on the RTX 5070 Ti (168 passed, 199 outputs, 0 fired, closest 0.62), the
  zoo under `truestar check` on CPU and on CUDA (796 module outputs each, 0 fired, closest 0.095 and
  0.44).
* **Found:** TF32 contractions lean toward zero on the RTX 5070 Ti with Triton 3.7.1: -6.2e-4 on a
  positive 1024 long matmul, what truncating both operands to 10 bits predicts; strict float32 does
  not lean. A TF32 kernel's lean is reported (TS966, one warning per kernel) and the kernel is used.
  Half precision tensor core accumulation leans significantly but 15x under its floor.
* **Order probe** (`truestar orderprobe`, `truestar.orderprobe.probe`): +M, -M and unit probes give
  subtree sizes (sum, matmul K); a triplet test does softmax denominators; 53 bit accumulators,
  rows with different orders and run to run changes (TS963) are detected; every tree is replayed bit
  for bit in exact arithmetic (TS962) and matmuls name the product model (fused multiply add matched
  cuBLAS). Measured: eager PyTorch's Linears in the fingerprint transformer layer use cuBLAS split K
  (2 chunks at K 128, 4 at K 512) on the 70 SM GPU; the softmax is a shuffle tree; the numba CPU
  kernel accumulates in float64. `compare` diffs two GPUs' files.
* **Tests:** tests/test_bias.py (18), tests/test_orderprobe.py (19), tests/test_kernel_verify_elements.py
  (old rule now also switches the lean test off), tests/test_codes_unique.py, tests/test_agent.py.

Not verified: the second GPU (the cause of the README's eager transformer row is inferred from the
split K seen here, not measured on the 26 SM laptop or the T4), AMD and Apple GPUs, the compiled
executor's region and end to end checks (size only), `compile:` and `ort:` probes, attention batched
matmuls and layer norms in the probe case.

### 2026-11-24: coverage refresh to current transformers and the 2025 to 2026 families (P46)

Branch wt/cov26 from e96cd2c. Docs: docs/MODEL_COVERAGE.md ("Current models" and the regenerated
tables), docs/AGENT_INTERFACE.md (TS940 to TS942). Research item: LIVE_RESEARCH_2026_11.md 5.1
(ranked 9). Run on the Linux server in a new venv `/srv/truestar/venv-cov26` (transformers 5.24.0,
the latest stable release, torch 2.11.0 CPU, torchvision 0.26.0, numba 0.68.1, Python 3.12; CPU
only, nice 19). The Windows envs were checked and not changed (`ml`, `ts`: transformers 4.48.0;
`cv`: 5.17.0). Nothing downloaded; every model is a config built in code with random weights.

* **Measured first.** The sweep of e96cd2c on transformers 5.24 with the new list: 10 of the 15
  first list Hugging Face models import (Llama, Mistral, Qwen2, GPT-2 and Whisper are refused at
  `aten.diff`: transformers 5 runs `torch.diff` on the position ids of every decoder), 2 of 28
  current families, 2 of 20 current op patterns. MoE, linear attention and SSD models failed with a
  list of unknown ops (`histc`, `grouped_mm_fallback`, `eye`, `bitwise_not`), never with a reason.
* **Families added** (`testing_hf.MODELS_2026`): Gemma 2, 3, 3n and 4 (dense), Qwen3, Phi-4 mini,
  SmolLM3, OLMo 2 and 3, Granite, Cohere 2, GLM 4, ERNIE 4.5, EXAONE 4, Seed OSS, Apertus, Arcee,
  Ministral 3, Hunyuan dense, LFM2, BitNet, and the vision encoders SigLIP, SigLIP2, DINOv3, AIMv2,
  SmolVLM, Pixtral, ConvNeXt V2. Each config makes the family's new parts live at a small size.
* **After** (same list, same transformers, CPU compile): first list 15 / 15 / 15 (import, compile
  with no fallback, training step equal to torch in float64), current families 26 / 26 / 26 of 28,
  current ops and patterns 20 / 20 / 20, torch.nn 92 / 92 / 91 (unchanged, Hardsigmoid is torch's
  float32 1/6), torchvision 63 / 63 / 63 (its first CPU compile). Worst float64 import error 3.1e-15.
  197 distinct ATen ops: 190 import with every rule, 0 can take the CPU fallback; the 7 left are
  SigLIP2's.
* **Refused by name** (`ir/refusals.py`, TS940 MoE routing, TS941 chunked linear attention, TS942
  chunked SSM scan), before any op is imported, on both import paths, naming the layer class and
  path: gpt-oss, Qwen3 MoE, Qwen3 Next, Qwen3.5, Llama 4, DeepSeek V3, Kimi Linear, OLMoE, Mamba 2,
  Falcon H1, Nemotron H (11 of 11). Exact class names first, then name parts, so the Mamba 1
  sequential scan and look alike names (a detector class `SSD`) are not refused.
* **Ops** (`ir/importer_current.py`, all lowered onto existing IR ops; no new IR op, no new kernel
  rule): `diff`, `square`, `floor`, `bitwise_not`, `meshgrid`, `tile`, `bucketize`, `erfinv` of a
  constant subgraph (folded at import in float64; input dependent refused TS860), `std`, `var`,
  `multiply`, `detach_`, `_is_all_true`, `conv2d.padding` and string padding for every convolution;
  `sum` and `mean` with a dtype and bool sums as int64, `softmax` with a dtype. Attention sinks, soft
  capping, sliding windows, QK norm, partial and proportional RoPE, per layer embeddings, KV sharing
  and xIELU needed no new op.
* **Importer and comparison fixes found by the sweep:** an in place op on a slice of a tensor
  reached through identity aliases (Gemma 3n's AltUp correction) is rewritten as for a fresh
  tensor; `exact_float32_casts` keeps `dtype=torch.float32` arguments of float64 calls in float64
  (Gemma 2's softmax) and creates float32 constants in float32 before widening them (Gemma 3n's
  sparsity target, which first gave 6.6e-8).
* **Found on the torch side:** torch.autograd's float64 gradient of Gemma 3n has NaN entries at the
  test inputs; Truestar's has NaN at exactly the same entries (the test checks the masks), cause not
  traced. On transformers 4.x, Gemma 2 builds its embedding scale in the model's dtype, so the
  float32 export holds sqrt(32) rounded to float32 and the exporter's 1e-9 bar refuses it (2.1e-9);
  the test asserts that refusal on 4.x.
* **Tests run** (the rule of this branch: touched files plus the shared list): test_hf_models,
  test_op_coverage, test_codes_unique, test_agent on the server, 637 passed, 53 skipped (CUDA) on
  transformers 5.24 and 569 passed, 35 skipped on 4.48, at 8e35cfb. On the Windows PC (4.48) a first
  run with CUDA visible had 25 CUDA test failures while another agent was using the GPU; two of
  them (BERT compiled, the float64 mask test) pass when rerun alone; the rest were not rerun and are
  not classified. Tests can fail: a floor lowered as truncation, a bucketize with the other boundary
  rule and a sink softmax that keeps the sink column each fail the value test.

Not verified: CUDA for the new families (pending, the GPU was busy), pretrained weights, models over
100M params, the full multimodal models (only their vision encoders), SigLIP2 and Pixtral vision
(data dependent per image sizes), MoE and linear attention themselves (refused, planned: the near
tie routing report, 4.3, and chunk size invariance, 5.2).

### 2026-11-24: MX formats, FP8 variant trap, keyed stochastic rounding (P50)

Branch wt/lpfmt. Details and every number in docs/QUANT.md (last section); data in
results/lowp/study.json from scripts/lowp_study.py.

* MXFP8 / MXFP4 tables equal ml_dtypes on every code; three scale rules (OCP v1.0 floor,
  OCP v1.1 expAdj, the MXFP8 recipe's round up). On SmolLM2-360M weights the v1.0 rule clips
  0.1 % to 0.75 % of elements; the v1.1 rule clips none and has the lowest error for all five
  formats; round up is worse than floor on E2M3 and E2M1.
* MXFP8 matmul on `tl.dot_scaled` passes verdict (a) (ratio 0.0014), 1155 us against 1393 for
  cuBLAS float16 at 4096 cubed. MXFP6 was measured (no GPU kernel, Triton 3.7.1 NYI) and is not
  shipped.
* FP8 variants: OCP versus FNUZ from names of seven ecosystems or from bits, TS1000 to TS1005;
  `sim:mi350-tf32` profile (BF16 emulation, hypothesis on the algorithm).
* Adaptive NVFP4 (per block 4/6 scale) was tried and not shipped: lower weight MSE on 224 of 224
  layers (median 0.84x) but W4A4 CE delta +0.272 against +0.238 plain on 1020 positions.
* Keyed SR: integer rule on float32 bits, python / torch CPU / torch CUDA / Triton bitwise equal;
  bias within the 2^-32 quantum bound, pooled |z| at most 1.61 over the targets; bfloat16
  accumulation of 1e-3 x 10000 ends at 11.008 against 1.0 for round to nearest; Triton SR 781 us
  per 16M elements against 139 us for a round to nearest cast.
* Codes TS1000 to TS1009 (4 digit codes needed no change to codes.py or its uniqueness test).

### 2026-11-24: first real NPU numerics through Qualcomm AI Hub (P45, docs/QAIHUB.md)

Branch wt/qaihub. A fixed battery (float MLP, residual conv net and transformer encoder layer with 13
taps in one ONNX file; a QDQ INT8 MLP with 3 taps) ran on a Galaxy S26, a Galaxy S21, a Snapdragon X
Elite laptop and a Dragonwing RB3 Gen 2 board through TFLite, QNN DLC and ONNX Runtime with the QNN
EP. 25 AI Hub jobs (7 compiles, 18 inferences). `truestar conform qaihub DIR` judges a saved results
directory with no network (TS920 to TS929).

* Convs on the HTP compute what `sim:ieee-f16` computes: ratio 1.00x, 94% to 100% bitwise equal.
* GELU (Erf) on the HTP is 6.2x correct float16 on v81 and v73 with the same bits; no profile models
  it. Softmax and attention 1.7x to 2.0x.
* 4 of the 8 completed float NPU or GPU runs returned SUCCESS while computing in float32 on the CPU (TS925).
* INT8 on the NPU: 1% to 4.6% of elements one step off per layer, never more, unexplained by every
  intref convention; ORT CPU on the laptop is bitwise exact.
* Tests: tests/test_qaihub.py (8, no network: clean, float16, float16 accumulator, NHWC, planted
  faults, INT8 one step and three steps, intref convention naming, failed and missing runs, hash
  mismatch, CLI exit codes and agent JSON, host ONNX Runtime control), with tests/test_codes_unique.py.
  Breaking the float16 band or the per layer replay fails them.

### 2026-11-24: chunked linear attention and SSM kernels (P53, `truestar.linattn`)

Research item 5.2 (<research>/LIVE_RESEARCH_2026_11.md). docs/LINEAR_ATTENTION.md has
the math, the proof and every number below.

* **Float64 reference:** the recurrence one step at a time (`reference.recurrent`), add and delta
  rules, no, scalar or per channel decay, initial state. The chunked PyTorch form equals it within
  1e-13 for chunk sizes 1, 2, 3, 7, 16, 64, 100 and 128 (measured 2.6e-16 to 3.5e-15), and its float64
  gradients for q, k, v, g, beta and the initial state equal the recurrence's within 1e-11.
* **Chunk size in float32** (RTX 5070 Ti, T 2048, dk = dv = 128): chunk 16, 32 and 64 are 1.2e-6 to
  2.3e-6 from float64 and 1.4e-6 to 2.3e-6 from each other, with 62% to 84% of the elements
  differing in bits. Chunk 16 was closest to float64 in every case.
* **Chain error** (T 16384, 1024 chunks): flat (last over first quarter 1.01 to 1.02) with any decay
  or the delta rule; plain linear attention with no decay grows 7.0x, worst 1.0e-5, reported as
  TS1062.
* **Determinism:** the config depends on dims only; outputs and final states bitwise equal on the
  RTX 5070 Ti (70 SMs) and an RTX 5060 Laptop GPU (26 SMs) in 4 of 4 cases
  (scripts/linattn_fingerprint.py). The value block and forward warps were shown not to change the
  bits; the prep kernel's warps do, so they are fixed.
* **Verification:** first call per signature against float64 on the device; a broken element of
  1e-3 and a 0.1% error on the last step are both caught (TS1060).
* **Speed** (RTX 5070 Ti, float32, forward, minimum of three runs on a shared GPU): a Mamba 2 2.7B
  SSD layer at T 4096 6.13 ms against 437 ms sequential and 9.97 ms eager chunked; a Qwen3 Next
  Gated DeltaNet layer 5.38 ms against 632 and 12.1; batch 4 dk 64 layers 0.47 to 0.76 ms, 580x to
  629x the sequential scan and 9x to 16x eager. TF32 dots (opt in) 3.2 to 3.5 ms on the large layers.
* Codes TS1060 to TS1063; `truestar codes` now sorts by number so four digit codes follow TS886.

Not done: a Triton kernel for per channel decay (Kimi Delta Attention, gated linear attention run
in the chunked PyTorch form), a fused backward (gradients come from autograd through the chunked
form), and an importer lowering of the Hugging Face modules, so TS941 and TS942 still refuse the
models; `truestar.linattn.refusal_lifted` is the hook for lifting them.

### 2026-11-24: device breadth, Windows ML EPs, LiteRT NPU, WebNN, QAIRT range modes (P56)

Research items 6.1, 6.2, 6.3 and 6.5 of <research>/LIVE_RESEARCH_2026_11.md, on this
PC only (Windows 11 build 26200, RTX 5070 Ti, Ryzen 9 9900X, no NPU).

* `winml:<ep>` (docs/BACKENDS.md): NvTensorRtRtx and WebGPU EPs ran through their bring your own
  packages in onnxruntime-gpu 1.29.0, since the Windows ML catalog download timed out twice
  (0x800705B4 after 244 s, no progress). NvTensorRtRtx computes float32 GEMMs in TF32: Linear own
  error 2.79e-4, torch TF32 on the same data 2.7906e-4, strict float32 9.0e-8. WebGPU EP 2.0e-7.
  DirectML not run: its builds replace the env's onnxruntime package.
* `tflite:npu` (docs/CONFORM.md): the LiteRT NPU request ran all operators on XNNPACK while
  `is_fully_accelerated()` returned True; reported TS1125 with LiteRT's reasons (NPU accelerator not
  registered, Intel plugin missing openvino.dll). AOT partition reading tested on constructed files.
* `webnn:*` (docs/WEBGPU.md): headless Chrome 158, Truestar's own ONNX to WebNN translation; MLP and
  CNN clean on cpu, gpu, npu device types (max 1.7e-7). Found: one operand under two output names
  lost the context at build (fixed in the page), the tapped BatchNorm output wrong on gpu and npu
  while the op alone is right (caught as TS564), ReLU gives -0.0 on gpu and npu.
* `truestar.intref.qairt` (docs/INTREF.md): tf and symmetric match the QAIRT doc's worked examples;
  enhanced and adjusted are labelled hypotheses; `attribute` and `compare_to_device` are the hook
  for real QNN jobs (none compared here).
* Tests: tests/test_qairt.py, test_winml.py, test_litert_npu.py, test_webnn.py.

### 2026-11-25: float64 islands for rotary angles in float32 programs (P63)

Islands extended: in a float32 program imported in float32, a value declared float64 keeps float64
storage and math (its kernel computes in float64) and is rounded to float32 once where the model
casts. Before and after on the 3 layer Llama of the prefix cache check (`scripts/rope_long_positions.py`):
the new key row's error against float64 was 1.0e-6, 4.3e-5, 6.7e-5, 1.2e-4, 2.3e-4, 4.9e-4 at
positions 100, 2100, 4000, 8000, 16000, 32000 and is 3.2e-7, 4.7e-7, 4.3e-7, 3.0e-7, 2.6e-7, 2.4e-7
with `rope_f64`; the decode at position 2100 into capacity 2200 failed the end to end check (8.62e-4
against 6.3e-4) and passes. Logits were 2.6e-7 to 3.8e-7 off either way. Default programs unchanged:
9 of 9 compiled hashes (MLP, Transformer, resnet18 in float32, float16, bfloat16) and 67 of 67
deterministic fingerprint rows bitwise equal to laptop-fp16decode. A bug on the way: the first
version marked the island but computed it in float32, inlined into the cast's float32 kernel; a
float64 island's value is now stored. Tests: test_islands 11 (3 new); islands, compiler, kernel,
fingerprint, KV quant decode, graphs and examples 315 passed, 4 skipped.


### 2026-11-24: speculative decoding equal to greedy decoding (P57)

`truestar.specdec` (check_rows, greedy, speculative) with TS705 and TS706. Row invariance of the
compiled deterministic target checked at 3 to 7 rows (zero difference); eager PyTorch is not row
invariant even in float64. Measured with a pair trained on standard library source: graphed greedy
702 tokens/s, graphed speculative 926 to 1007 (1.32x to 1.43x); ungraphed 122 to 151 to 160 (1.24x
to 1.31x); tokens bitwise equal to greedy on 8 of 8 prompts at k = 2, 3, 4, 6. The graphed numbers
need laptop-fp16decode (main refuses the dense step graph, TS801). tests/test_specdec.py (10).


### 2026-11-24: deterministic autotuning that keeps the summation order (P49)

Deterministic kernels whose bits follow their config (attention, `tile_single`) are now tuned over
the parameters that cannot change the order of their sums (`triton_backend.ORDER_FREE`): `LPT` and
`num_stages` for attention, `BR` for single pass reductions; `tile_looped` stays pinned. Every
candidate shares the default's `order_key`, a cached config with another key is refused, and the
pick is run against the default on the tuning inputs and kept only when bitwise equal. Source
`order tuned` in `launch_configs()`.

Evidence (RTX 5060 Laptop, 26 SMs): `scripts/order_classes.py` over 18 cases (attention causal, full
and decode, softmax and layer norm, float16 and float32, up to 400 configs each) found those parameters
never changed a bit while BM, BN, warps, SPLIT and KV_CHUNK did. `truestar fingerprint --arms
truestar_det --fresh-cache` on main and on the branch: 67 of 67 rows bitwise equal, 4 launches order
tuned on the branch. The SM aware, split KV, KV quant, fingerprint and tuning suites (26 and 70 SM
pins included) pass: 161 passed, 1 skipped; tests/test_detautotune.py adds 5 (one forces a bit
changing pick and sees it refused).

Speed (`scripts/bench_detautotune.py`, before and after interleaved, best of 7): float32 attention
prefill 90.4 to 73.5 us causal 300x300 and 248.0 to 200.9 us full 200x700 (1.23x, num_stages 2 to
1); every other case within the noise of same config pairs (0.95x to 1.06x). Bitwise equal in all 18.
The large speed levers (block sizes and warps) are not tuned: they change the order.

### 2026-11-25: decode kernel overlap (P58), tried and not shipped

Programmatic dependent launch between the compiled decode step's Triton kernels (each kernel waits
with `gdc_wait` before its first read and is launched with `launch_pdl=True`, replayed from a CUDA
graph; RTX 5060 Laptop, deterministic float32; prototype only, not in the code). Baseline: the
graphed dense step (laptop-fp16decode). Bits unchanged in every run (sha256 of 32 greedy steps'
logits equal). Speed, microseconds per step, off and on:

| case | baseline | wait and early signal | wait only |
|---|---|---|---|
| 8 layer dim 384, batch 1 (355 plan steps) | 1562 to 1579 | 1798 to 1813 (15 percent slower) | 1551 |
| 4 layer dim 512, batch 1 (187 plan steps) | 1110 to 1127 | 1345 to 1362 (21 percent slower) | 1111 |
| 4 layer dim 512, batch 8 | 1171 to 1193 | 1388 to 1406 (18 percent slower) | 1160 |

Signalling dependents at the start lets the next kernel's blocks occupy SMs while they wait; wait
only is within the run to run noise (about 3 percent). In a chain of 100 tiny kernels in a graph
PDL saved 0.2 to 0.4 us per kernel, against 4 to 6 us per decode step kernel: the step is bound by
the kernels themselves, not the gaps between them, so a persistent megakernel's ceiling here is
small too (at most about 0.9 us per boundary, the whole plain cost of a tiny kernel). Not built:
no measured gain, and the launch deadline. Revisit on a GPU where one decode kernel takes well under
a microsecond (a larger SM count at batch 1), where the gaps are a larger share.
### 2026-11-25: sim:hexagon fitted to real Snapdragon outputs (P59, docs/DEVICES.md section 7)

Branch wt/hexsim. The P45 battery outputs (S26 v81 TFLite and DLC NPU, X Elite v73 DLC and ORT QNN
NPU) fitted into a `sim:hexagon` profile and measured against them.

* GELU on the HTP is fused into the linear layer before it (fed the accumulator, the fitted curve
  matches 89.4% of held out outputs bit for bit; fed the float16 input, 72.5%) and is `x * phi(x)`
  with a piecewise normal CDF approximation: a line below |x| = 0.6, degree 5 above, clamped at
  about -3.02 and 3.00, not symmetric between signs.
* Held out GELU rel L2 2.71e-3 (`sim:ieee-f16`) to 2.73e-4; on the battery 10.9x, 7.8x and 6.1x
  closer on the MLP taps and 1.8x on the encoder output, whose GELU was not in the fit. Convolution
  taps bitwise identical to `sim:ieee-f16` (94% to 100% equal to the device, unchanged).
* INT8: tried and not shipped. Mismatches sit within 0.03 of a half step (v73, v81; 0.12 on v68)
  at a rate flat in magnitude; no float16 product, scale storage, double rounding grid, per channel
  offset, 16 bit internal activation or input quantizer beat exact requantization.
* New codes TS1140 (fitted op), TS1141 (op known to differ, not modelled). Tests:
  tests/test_hexagon_sim.py against a 66 KB fixture of device taps.
### 2026-11-24: Mamba 2 SSD and Gated DeltaNet families lowered at import (P60)

* `ir/hf_linattn.py` replaces transformers' `mamba2_chunk_scan` and `torch_chunk_gated_delta_rule`
  during export by the same computation around `torch.ops.truestar.linear_attention`; the IR op
  `linear_attention` (with `_vjp` and `_jvp`) runs the linattn kernels when compiled.
* Families on transformers 5.24 (CPU): 5 of 5 import, compile and train (was 0 of 5, refused TS941
  and TS942): Mamba 2, Falcon H1, Qwen3.5, Bamba, Zamba 2; fp64 at most 6.4e-16, gradients at most
  5.4e-16. On transformers 4.48 they stay refused, with the reason.
* Importer: two writes through one view are imported when no other write came between (Qwen3.5),
  `aten.concatenate` (Zamba 2); compiler: a one entry strided `slice_scatter` is a kernel.
* Found: Falcon H1 on CUDA refused by the Triton kernel's check (TS1060, state width 256, strong
  decay); MobileViT's CUDA compile on transformers 5.24 fails its check on main too (not caused
  here); the double view write compiles wrong in a small module and is refused by the first call
  check. New code TS1150. Not a speed win at the test widths (section 10 of LINEAR_ATTENTION.md).

### 2026-11-25: batch 4 merge, rows the branches reported (P7, P11, P20, P24, P32, P42, P45, P46, P64 to P66)

Branches that could not edit this file reported these updates; numbers are theirs, measured on the
branch (reports under truestar-research/<branch>/REPORT.md).

* P7 (wt/lossimp): 63 of 63 torchvision classifiers export with a real check; googlenet's random init
  gave the same output for every input, so `scripts/export_zoo.py` now measures batch norm statistics
  on random inputs for such a model (float64 error 4.9e-14, runner error 1.6e-5).
* P11 (wt/ci): `.github/workflows/install-check.yml`: fresh venv, CPU torch, wheel, quickstart and CPU
  smoke on windows-latest, ubuntu-latest, macos-latest (macOS 26 arm64), Python 3.12. Not run on
  GitHub yet. The `mcp` extra is pinned below 2 (mcp 2.3.0 removed the API `truestar mcp` uses).
* P20 (wt/lossrng): keyed normal draws 1.15x to 1.9x (float32) and 1.32x (float64) the time of
  torch.randn on A10G, L4, T4 at 1e8 draws; bits unchanged (0 of 1e8 differ). Float32 parity needs a
  cheaper, versioned bit definition (owner decision).
* P24 (wt/lossrng): the six forward rules are on main (e37a50b), not missing; the remaining gap is the
  reverse rule for `slice` with a step over a symbolic dim.
* P32 (wt/servecpp): `truestar conform llamacpp`: official build v1.0.0, CPU; 0 of 9,216 clean
  positions over tolerance in 128 runs; 4 of 4 engine setting faults and 16 of 16 conversion faults
  caught with the right blame; threads never change bits, flash attention and Q4_K_M batch do;
  detected only (TS1310 to TS1319). (wt/servegpu): serving engines change answers under batching,
  prefix caching and chunked prefill, and nothing showed by how much or whether a token flip is a
  tie: `truestar conform vllm` and `sglang` on an L4: both engines change bits under all three, every
  flip is a near tie, a planted rope_theta is caught (TS1320 to TS1329). TensorRT-LLM refused by name
  (TS1321).
* P42 (wt/lossnative): 0 of 161 ops with a torch step on the device (was 10); native gathers 1.3x to
  3.6x the torch step on an L4. On the CPU the cross entropy class pick (0.73x) and the adaptive max
  pool tap read (0.68x) are slower than the torch step they replace (owner decision: keep native or
  send back); `scatter_add` with an index over 64 times its column count stays on the torch step.
* P45 (wt/lossimp): per channel INT8 TFLite written directly (`qaihub_battery.int8_tflite_bytes`) and
  run locally (LiteRT CPU equals the intended integer arithmetic on 8,352 of 8,352 values); conform
  reads quantized tensors as real numbers; the device run on AI Hub is pending. (wt/hexsim): P59.
* P46 (wt/lossimp): 28 of 28 current families on transformers 5.24; SigLIP2 and Pixtral vision at a
  fixed batch.
* P64 (wt/moe), P65 (wt/mla), P66 (wt/tp4): ledger rows above.

Other branch findings:

* wt/errmsg: a refusal named what is wrong but not what to type next, and four codes exist that
  nothing emits (TS464, TS566, TS1044, TS1045, now marked reserved in docs/AGENT_INTERFACE.md).
  `scripts/audit_codes.py` lists every code's raise sites, docs row, tests and message grade; eight
  refusals reworded (tests/test_error_messages.py, 8 tests fail on the old text and pass on the new).
* wt/docsrun: docs showed code that does not run: `scripts/check_docs_snippets.py` ran 99 items on
  the CPU; 6 of 71 blocks failed on a missing import at 9a9ad96, 0 after.
* wt/switchdocs: 10 cookbook recipes and 3 switching examples run on the CPU under 60 s each (16
  tests). Speculative decoding on the native KV cache is bitwise equal to greedy on the compiled CPU
  executor (24 of 24 logit rows), not on the float64 reference executor (1 of 24); CUDA not
  measured. Three fixes found by the recipes (memory plan with gather, two spurious warnings).
* wt/covgen: the Euler scheduler step stays a documented gap, not a Truestar refusal: torch.export
  itself stops on `EulerDiscreteScheduler.step`'s Python branch on a tensor value before anything
  reaches Truestar (tests/test_vision_models.py keeps it failing as documented).
