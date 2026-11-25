# Truestar pain research, round 2

Written 2026-11-24. Read only web research (Reddit, GitHub issues, vendor docs, papers, blogs). Every
source below was read in search results or fetched pages during this pass; dates are the publish or
issue creation dates those pages showed. Quotes are short excerpts, numbers are as reported by the source.

Scope: pains that are NOT already rows P1 to P12 in docs/QUALITY_BAR.md. Where a finding overlaps an
existing row, it is marked as an extension of that row rather than counted as new.

Scoring: severity (S), frequency (F) and Truestar architecture fit (A), each 1 to 5. Rank score is
S x F x A. Fit means how directly the existing pieces (float64 reference, per module replay, op
drilldown, shape typed IR, verified kernels with a canonical IR cache, `.tsx` artifacts, agent JSON)
carry the solution, with little new machinery.

## Ranked top 10

| Rank | ID | Pain (one line) | S | F | A | Score |
|---|---|---|---|---|---|---|
| 1 | N1 | Upgrading torch, CUDA, cuDNN or Triton silently changes model outputs and accuracy | 4 | 5 | 5 | 100 |
| 2 | N2 | LLM outputs change with batch size, concurrency, TP size and GPU type, even at temperature 0 | 4 | 5 | 5 | 100 |
| 3 | N3 | RL trainer and rollout engine disagree on logprobs for the same weights, and RL degrades or collapses | 5 | 3 | 5 | 75 |
| 4 | N4 | A second implementation (vLLM, llama.cpp, MLX port, JAX port) silently computes a different model than the reference | 4 | 4 | 4 | 64 |
| 5 | N5 | Custom CUDA ops take minutes to hours to build and break on every torch or CUDA version change | 3 | 4 | 5 | 60 |
| 6 | N6 | BF16 and FP8 training diverges late because a trusted kernel returns wrong gradients, with no NaN | 5 | 3 | 4 | 60 |
| 7 | N7 | Quantized LLM files ship broken or degraded, and perplexity or averages hide it (extends P2) | 4 | 4 | 3 | 48 |
| 8 | N8 | OOM from allocator fragmentation with plenty of free memory, and no reliable peak memory prediction before a run | 4 | 4 | 3 | 48 |
| 9 | N9 | Silent training bugs where the loss looks fine (optimizer no op on one backend, ranks drifting apart) | 4 | 3 | 3 | 36 |
| 10 | N10 | Resumed runs are not the same run: loss spikes, missing rank shards, lost data cursor | 4 | 3 | 3 | 36 |

Ties at 100 are broken by build cost: N1 reuses `truestar check` almost unchanged.

Below the cut (real pains, lower score or a good tool already exists): N11 NaN origin, N12 Windows GPU
compile path, N13 profiler trust and kernel to source mapping, N14 ROCm consumer performance.

Do NOT take on: N15 fleet scale silent data corruption, N16 dataloader throughput, N17 consumer
multi GPU P2P and NCCL topology, N18 CPU inference speed race. Reasons are given per item.

Already covered by the ledger, noted for evidence only: edge converter op gaps (P7), TensorRT engine
portability (P7), long LLM compiles (P6).

---

## N1. Library upgrades silently change numerics

**Statement.** Upgrading torch, cuDNN, Triton or the inference engine changes outputs or accuracy with no
error, and finding which op changed is manual bisection over versions.

**Evidence.**
* pytorch/pytorch#181169, 2026-04-22: torch 2.12 made vLLM gsm8k accuracy drop "0.54 -> 0.48", bit
  identical over three runs; root cause a single commit, reverted. Labelled `module: correctness (silent)`.
  https://github.com/pytorch/pytorch/issues/181169
* pytorch/pytorch#157274, 2025-06-30: torch 2.8 RC gave "10000 times larger" batched vs single row
  differences in transformers tests (7.4e-9 to 4.9e-5), traced to a cuDNN update that picked TF32
  kernels; "527 new failed tests". https://github.com/pytorch/pytorch/issues/157274
* PyTorch forums, 2025-03-03: after an OS and torch stack upgrade "the label and confidences completely
  change" on some inputs; cause was a changed loader. https://discuss.pytorch.org/t/model-output-drastically-different-after-upgrade/217399

**How common, how severe.** Every team upgrades several times a year. Severity is high because the
change is silent and deterministic, so it passes reruns. Maintainers state there is no bitwise
guarantee across versions, which is fair, but that leaves users unable to tell noise from a regression.

**Existing tools.** None found that compares two software environments per module. People bisect
versions by hand or compare final metrics. Polygraphy compares runtimes, not torch versions.

**Truestar proposal.** `truestar record model.py -o ref.tsr` stores per module inputs and outputs (the
float64 reference plus the backend result) with the environment fingerprint; `truestar diff ref.tsr`
in the new environment replays each module alone with the recorded inputs and blames the first module
and op whose own result moved beyond the calibrated noise band, labelled as noise or as a change. It
reuses the replay and op drilldown already built; only serialization of recorded activations is new.

**Acceptance test.** Two venvs that differ only in one injected op change (an `OpFault` registered in
venv B, plus one real case: TF32 cuDNN on vs off, which reproduces the #157274 pattern) are diffed;
the report names the op and module, classifies TF32 drift as precision change rather than bug, and
reports a clean result for two identical venvs. Baseline: comparing final logits with `allclose`,
which flags the TF32 case as failure and cannot name the op.

## N2. LLM inference is not batch, TP or hardware invariant

**Statement.** The same prompt at temperature 0 gives different completions depending on what else is
in the batch, the tensor parallel size and the GPU type, which breaks evals, caching, tests and audits.

**Evidence.**
* Thinking Machines Lab, 2025-09-10: 1000 greedy completions of one prompt gave "80 unique completions";
  batch invariant kernels made all 1000 identical. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference
* NeurIPS 2025, Yuan et al.: bf16 greedy decoding shows "up to 9% variation in accuracy" across GPU
  count, type and batch size. https://proceedings.neurips.cc/paper_files/paper/2025/hash/f80094a824ba5912d4a2de169c404a40-Abstract-Conference.html
* LMSYS SGLang, 2025-09-22: deterministic mode costs an average slowdown of 34.35% (TML reported 61.5%).
  https://www.lmsys.org/blog/2025-09-22-sglang-deterministic
* Eval4NLP 2025, Atil et al., 2025-12-04: hosted APIs at temperature 0 show "accuracy variations of up
  to 15% across runs". https://aclanthology.org/2025.eval4nlp-1.12.pdf

**How common, how severe.** Universal for anyone serving or evaluating LLMs. DESIGN.md already lists
this as a device spectrum finding but no ledger row or test covers it.

**Existing tools.** Batch invariant kernel libraries and deterministic modes in SGLang and vLLM, with a
large speed cost and limited backend coverage (SGLang lists three attention backends). TP invariance
needs a further kernel family (TBIK, below). Nothing tells a user which op in their own model breaks
invariance.

**Truestar proposal.** Two parts. (a) Checker: `truestar check --invariance batch,tp` runs each
module with a row alone and inside batches of several sizes and reports the first op whose per row
output changes, which is exactly the existing isolation plus op drilldown with a different reference.
(b) Compiler: batch invariance becomes a declared kernel property. The kernel IR already fixes the
reduction structure in the canonical IR, so the scheduler can be constrained to a reduction tree that
does not depend on the batch axis, and the float64 verifier gains a bitwise invariance check.

**Acceptance test.** On a two layer transformer, the checker names the non invariant op (for example a
split K matmul or a reduction whose tile depends on batch) on CUDA and reports invariant for a
module built from Truestar kernels. Truestar compiled attention and matmul produce bitwise identical
rows for batch sizes 1, 2, 7, 64 with overhead under 20% versus the non invariant Truestar kernel,
measured against SGLang's published 34% as the bar to beat.

## N3. Train and inference engines disagree in RL

**Statement.** RL for LLMs samples with one engine and trains with another; tiny numeric differences
make on policy training off policy and can degrade or collapse training.

**Evidence.**
* vLLM blog, 2025-11-10: running sampler and trainer with different kernels "shows a reduced reward";
  bitwise matching gave KL exactly 0 and higher reward. https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference
* vllm-project/vllm#21475, 2025-07-23: vLLM vs HF logprob max relative error "ranging 24%~170%" across
  four models. https://github.com/vllm-project/vllm/issues/21475
* ICML 2026 TBIK, Zhang et al.: TP size mismatch between FSDP and rollout "can degrade or even collapse
  training"; fixed with tree based invariant kernels. https://proceedings.mlr.press/v306/zhang26ag.html
* Google Developers Blog, 2026-11-16: Zero TIM on TPU makes sampler and trainer logprobs bitwise
  identical. https://developers.googleblog.com/en/unlocking-stable-on-policy-rl-on-tpu-with-bitwise-zero-traininference-mismatch
* arXiv 2610.41073, 2026-10-29: bf16 export rounding discards most RL updates; on Qwen3 1.7B only 0.4
  to 0.5% of served weights change per step. https://arxiv.org/abs/2610.41073

**How common, how severe.** Every lab doing RL post training hits it; the fix today is a hand audit of
"every single invocation of every kernel" (vLLM blog). Severity is high because it wastes whole runs.

**Existing tools.** Per stack solutions (vLLM plus TorchTitan, SkyRL IsoExec, Tunix on TPU), each tied
to one pair of frameworks and one model family. No general tool tells you which op causes the gap.

**Truestar proposal.** This is the strongest structural fit: Truestar's thesis is one verified
artifact for training and deployment. Short term, `truestar check --pair trainer.py rollout.py` runs
both forward paths on the same tokens and blames the first module and op where logprobs diverge
(N4 machinery). Long term, the same compiled kernels (with N2 invariance) serve prefill, decode and the
training forward, so the mismatch is zero by construction, and the export path checks the served
weights, not the fp32 master, which also covers the 2610.41073 rounding gap.

**Acceptance test.** On a small decoder, a trainer forward and a decode loop built from Truestar
kernels give bitwise identical logprobs for every sampled token at TP 1 and TP 2 (TP via the gloo CPU
harness until multi GPU is available); with one kernel swapped for a different reduction order the
pair check names that op. Baseline: HF eager vs vLLM logprob gap as reported in vllm#21475.

## N4. Second implementations silently compute a different model

**Statement.** Ports and serving engines reimplement a model and get it subtly wrong (eps, dtype casts,
RoPE convention, dropped adapters, cache state), with outputs that look plausible.

**Evidence.**
* AI21 Labs, 2026-01-29: vLLM produced confident gibberish about "once in a thousand prompts" for a
  Mamba model; found only by comparing logprobs against transformers. https://www.ai21.com/blog/vllm-debugging-mamba-bug/
* Blaizzy/mlx-vlm#2053, 2026-08-27: a converter dtype cast took top 1 accuracy "from 18/23 to 1/23";
  a hardcoded eps also diverged. https://github.com/Blaizzy/mlx-vlm/issues/2053
* Physical-Intelligence/openpi#958, 2026-06-01: the official JAX to PyTorch converter silently drops
  LoRA weights via `strict=False`. https://github.com/Physical-Intelligence/openpi/issues/958
* MoonMath, 2026-07-08, porting a 22B model to JAX: "The last 5% failed silently". https://moonmath.ai/posts/pytorch-to-jax-field-guide-22b-video-model/
* zai-org/GLM-4.5#53, 2025-08-16: degraded reasoning on vLLM and llama.cpp even with unquantized
  weights. https://github.com/zai-org/GLM-4.5/issues/53

**How common, how severe.** Every new architecture is ported to several engines within weeks (Hugging
Face now automates MLX ports with an agent skill that compares every layer against transformers, 2026-04-16,
https://huggingface.co/blog/transformers-to-mlx). Errors reach users before anyone notices.

**Existing tools.** Ad hoc per layer comparison scripts written per port; the HF MLX skill does this for
one direction. No general tool aligns two implementations layer by layer.

**Truestar proposal.** Extend `truestar check` from one module run on several backends to two
implementations of the same model: the user (or an agent, via `--format agent`) gives a module name
map, or Truestar infers it from weight tensors (each weight is used by exactly one module in both
graphs). Each mapped module is replayed with reference inputs on both sides, so the blame is the
module's own error, exactly as P1 does today. A checklist of known convention traps (eps from config,
fp32 parameters kept fp32, RoPE pairing, unused checkpoint keys) runs on the weight map.

**Acceptance test.** Rediscovers three injected port bugs on a small Llama style model, each named
at the right module: eps hardcoded to 1e-6 when config says 1e-5, an fp32 scale cast to bf16, and a
LoRA key silently dropped (reported as an unused checkpoint tensor). A correct port reports clean.
Baseline: final logit comparison, which flags all three only as "outputs differ".

## N5. Custom ops are slow to build and fragile across versions

**Statement.** Writing a custom CUDA op means C++ builds that take minutes to hours, break on torch or
CUDA version changes and often fail on Windows.

**Evidence.**
* PyTorch forums, 2025-09-04: "over 6 hours to recompile tens custom operators for just a small
  modification". https://discuss.pytorch.org/t/long-build-time-for-custom-c-cuda-extension/222857
* pytorch/pytorch#152032, 2025-04-23 (RFC by maintainers): `load_inline` compile times "on the order
  of 90s for toy kernels". https://github.com/pytorch/pytorch/issues/152032
* pytorch/pytorch#187320, 2026-06-15: xFormers says the stable ABI contract broke when the default
  CUDA moved 12.x to 13.x. https://github.com/pytorch/pytorch/issues/187320
* pytorch/pytorch#166123, 2025-10-23: an extension fails to build on Windows with torch newer than 2.8;
  users downgrade. https://github.com/pytorch/pytorch/issues/166123
* pytorch/pytorch#163510, 2025-09-22: an extension built with CUDA 13 against torch built with 12.8
  reported 1 SM instead of 148, silently. https://github.com/pytorch/pytorch/issues/163510

**How common, how severe.** Hits library authors (vLLM, xFormers, FlashAttention) and every researcher
with a novel layer. Overlaps P8 (slow new layers) but the pain here is build and ABI, not kernel speed.

**Existing tools.** Triton and `torch.compile` avoid C++ for many cases; nvrtc based
`torch.cuda._compile_kernel` (PR in progress) cuts compile to 0.01 s for raw CUDA. Stable ABI work is
ongoing but incomplete.

**Truestar proposal.** A fused op written with `@truestar.function` is pure Python, typed, compiled by
the kernel layer with the IR keyed cache, verified against float64, and registered as a
`torch.library` op with a fake (meta) function generated from the shape rules, so it works inside
`torch.compile` and `torch.export` with no C++ toolchain and no ABI coupling to a torch build.

**Acceptance test.** A custom fused op (for example gated RMSNorm plus residual) defined in
`@truestar.function` compiles cold in under 2 s and warm in under 50 ms, needs no nvcc, MSVC or gcc
on PATH, registers as a torch op, passes `torch.library.opcheck`, runs under `torch.compile` with zero
graph breaks, and the same source works unchanged on two torch minor versions in CI on Linux and
Windows. Baseline: the same op via `load_inline` (90 s class per #152032).

## N6. Low precision training fails late because a kernel's backward is wrong

**Statement.** BF16 and FP8 runs look healthy for billions of tokens, then gradients blow up with no NaN;
the cause is often numerical error inside a trusted fused kernel, and finding it takes a research paper.

**Evidence.**
* arXiv 2609.34272, 2026-09-28: FlashAttention 3 in BF16 was healthy for 25B tokens, then "the gradient
  norm grew a thousandfold" without a single NaN; recomputing two layers' attention backward in FP32
  removed it. https://arxiv.org/html/2609.34272v1
* arXiv 2510.04212 (v3 2026-04-18): mechanistic cause of a BF16 flash attention loss explosion, and
  "automated tools to detect and mitigate such numerical instabilities" named as future work.
  https://arxiv.org/html/2510.04212v3
* ICLR 2025, Fishman et al.: Llama2 7B FP8 diverges after about 200B tokens. https://proceedings.iclr.cc/paper_files/paper/2025/file/f48b5133e89854a9e97cc22a6db83f25-Paper-Conference.pdf
* Lablup report, 2026-06-15 (504 B200 GPUs): attributing divergence to cuDNN bugs, numerics or hardware
  "remains an open challenge". https://arxiv.org/html/2605.09370
* arXiv 2506.20752: cites about 10% of FP8 NanoGPT runs failing to converge versus none in BF16.
  https://arxiv.org/html/2506.20752v1

**How common, how severe.** Frequency is moderate (large pretraining runs, FP8 adopters), severity is
the highest in this list: weeks of compute.

**Existing tools.** Loss and grad norm monitoring, which by definition fires late. Manual FP32
recomputation of suspect layers, as the paper did.

**Truestar proposal.** Extend per module replay to backward: for a saved batch and checkpoint, each
module's backward runs alone in the model dtype on the backend and in float64 on CPU with identical
upstream gradients, and the report blames modules whose own gradient error exceeds the calibrated
threshold (gradient direction error and norm error, not only max abs). Needs the autodiff pass, which
DESIGN.md already plans; the Stage 1 checker can do it first through torch autograd on replayed
modules. Also add a guard to the kernel verifier: backward kernels are verified like forward ones.

**Acceptance test.** On a small transformer with an injected backward only fault (correct forward, a
softmax backward that leaks a row sum like 2609.34272 describes), `truestar check --grad` blames the
attention module's backward and names the op, while a forward check reports OK. A clean model in BF16
reports OK. Baseline: `detect_anomaly`, which finds nothing because no NaN appears.

## N7. Quantized LLM artifacts ship broken; averages hide it (extends P2)

**Statement.** Users download quantized files that are nonfunctional or degraded on specific tasks,
and the metrics people use to choose quants (perplexity, average scores) can mislead.

**Evidence.**
* arXiv 2609.05881 (v3 2026-10-29): tested all 305 eligible code capable GGUF tags in the Ollama
  library; "None of the registries we examined documented functional testing". https://arxiv.org/html/2609.05881
* EMNLP 2025: 4 bit methods lose up to 59% on long context tasks while 8 bit loses about 0.8%.
  https://aclanthology.org/2025.emnlp-main.479
* vllm-project/llm-compressor#2031, 2025-11-13: request for a KL divergence tool because "I suspect
  some of the quant I create are suboptimal". https://github.com/vllm-project/llm-compressor/issues/2031
* ggml-org/llama.cpp#28879, 2026-09-14: perplexity ranks Q4_K_M above F16 on a hybrid architecture.
  https://github.com/ggml-org/llama.cpp/issues/28879

**How common, how severe.** Very common in local LLM use. P2 already covers decision flips for
classifiers; the new part is token level KL against the reference and per layer sensitivity.

**Existing tools.** llama.cpp KLD mode, the quantcheck screen from 2609.05881, Qualcomm and NVIDIA
per vendor debuggers. Good tools exist for parts of this, so Truestar should not build a quant toolkit.

**Truestar proposal.** Keep it inside P2: add per token KL and top 1 agreement as decision metrics for
LM heads, and per module sensitivity (replay each quantized module alone in an otherwise float model)
so the report says which layers to keep in higher precision. Do not write quantizers.

**Acceptance test.** On a small LM quantized with one deliberately bad layer, the report names that
layer as the dominant KL contributor and shows top 1 agreement drop, while perplexity on the same text
changes by less than 1%.

## N8. Fragmentation OOMs and unknowable peak memory

**Statement.** Jobs OOM with large free memory because the caching allocator cannot find a contiguous
block, the suggested fix is incompatible with some stacks, and nobody can predict peak memory before
running.

**Evidence.**
* pytorch/pytorch#182286, 2026-05-04: OOM "Tried to allocate 3.48 GiB" with "50.40 GiB is free";
  `expandable_segments` breaks the driver on that setup. https://github.com/pytorch/pytorch/issues/182286
* pytorch/torchtitan#3865, 2026-07-06: about 32.7 GiB reserved but unallocated, and DeepEP requires
  `expandable_segments:False`. https://github.com/pytorch/torchtitan/issues/3865
* PyTorch devlog, 2026-06-01: "Expandable segments don't eliminate all fragmentation". https://docs.pytorch.org/devlogs/eager/2026-06-01-cuda-caching-allocator
* arXiv 2602.17817 (v3): memory estimators under or overestimate; "FakeTensor typically underestimates".
  https://arxiv.org/html/2602.17817v3
* openxla/xla#27322, 2025-06-02: one fusion pass made a JAX program need about 95 GiB instead of 13 GiB.
  https://github.com/openxla/xla/issues/27322

**How common, how severe.** Common at long context and in MoE serving. P10 attributes an OOM after the
fact; it does not prevent fragmentation or predict the peak.

**Existing tools.** `expandable_segments`, memory snapshot viewers, formula calculators (vramcalc, the
VRAM calculator sites) that are accurate only for the stacks they were fitted on.

**Truestar proposal.** For programs Truestar compiles, value semantics and shape types make liveness
static, so the compiler can plan one arena per program and report the exact peak at compile time as a
function of the symbolic dims. For PyTorch programs, `truestar diag` gains a fragmentation verdict
(free bytes versus largest free block at the OOM). Add a never introduce rule from the XLA case: a
fusion that raises planned peak memory above the unfused plan is rejected.

**Acceptance test.** For five models compiled by Truestar, predicted peak is within 5% of measured and
never below it, and repeated runs at a peak under 95% of capacity never OOM. A fusion pass that would
raise peak memory is refused in a test modelled on xla#27322.

## N9. Silent training bugs: loss looks fine, the run is wrong

**Statement.** Training bugs that raise no error (optimizer step that does nothing on one backend,
ranks whose weights drift apart, wrong train or eval mode) waste long runs because loss still falls.

**Evidence.**
* TrainCheck, OSDI 2025: 18 of 20 real silent errors caught within one iteration, baselines caught 2;
  6 new bugs found in DeepSpeed and Transformers. https://news.engin.umich.edu/2025/07/improving-ai-models-automated-tool-detects-silent-errors-in-deep-learning-training/
* SREcon 2026 slides (Jiang): BLOOM 176B weights diverged across TP ranks "Loss was healthy for 39,999
  steps"; and an MPS bug where Adam silently skipped updates on non contiguous tensors.
  https://www.usenix.org/system/files/sre26amer_slides_jiang.pdf

**How common, how severe.** Moderate frequency, high severity. The MPS optimizer case is a backend
correctness bug that P1 cannot see today because `truestar check` replays forward only.

**Existing tools.** TrainCheck (open source, invariant inference) covers the generic invariants well.

**Truestar proposal.** Do not rebuild TrainCheck. Extend P1 to the training step: replay
`optimizer.step()` on each backend with identical parameters, gradients and state, and compare updated
parameters and optimizer state per parameter against the float64 CPU result. That catches the MPS
class directly and fits the existing report and codes.

**Acceptance test.** An injected backend only optimizer fault (no update for non contiguous parameters
on CUDA, via an op fault on `addcmul_`) is blamed on the parameter and the op; clean AdamW and SGD
steps report OK on CPU and CUDA.

## N10. Resumed runs are not the same run

**Statement.** Resuming from a checkpoint gives a loss spike or a different trajectory, or fails,
because some state (optimizer, dtype, rank shards, data position, RNG) was not saved or restored right.

**Evidence.**
* Hugging Face forums, 2025-06-28: "huge loss spike" on resume from FSDP sharded state dict; full state
  dict resumes smoothly but is too slow. https://discuss.huggingface.co/t/loss-spike-when-resuming-from-fsdp-sharded-state-dict-checkpoint-possible-optimizer-state-mismatch/160974
* verl-project/verl#4024, 2025-11-05: multi node checkpoint saved only 2 of 4 ranks; resume fails.
  https://github.com/verl-project/verl/issues/4024
* Databricks docs: a dataloader that restarts the epoch is "silently biasing the data distribution with
  no error". https://docs.databricks.com/aws/en/machine-learning/ai-runtime/guides/performance-and-resiliency
* ByteCheckpoint, NSDI 2025: shows bitwise correct dataloader resume is achievable. https://www.usenix.org/system/files/nsdi25-wan-borui.pdf

**How common, how severe.** Every long run resumes; failures cost a rollback or a silently different
model.

**Existing tools.** PyTorch DCP, ByteCheckpoint, framework checkpoint managers. They save state; none
proves a resume is faithful.

**Truestar proposal.** `truestar resume-check`: run step N+1 from live state and from the reloaded
checkpoint with the same batch, compare loss, gradients and updated parameters per module (bitwise
when deterministic kernels are used, calibrated otherwise), and list state that differs by name
(missing optimizer moments, dtype changes, RNG, data cursor). The `.tsx` checksums and evidence block
already exist for artifacts and extend to checkpoints.

**Acceptance test.** Four injected faults are each named: optimizer state not restored, bf16 shard
reloaded where fp32 was saved, RNG not restored, one rank shard missing. A correct save and resume
reports identical step N+1 on CPU.

---

## Below the cut

### N11. Where did the first NaN come from

* nantrace on PyPI, 2026-11-12: finds the op where NaN first appears; cites `detect_anomaly` at "1.4x to
  2x every step". https://pypi.org/project/nantrace/
* Towards Data Science, 2026-04-28: `set_detect_anomaly` "points you at where the NaN propagated to".
  https://towardsdatascience.com/pytorch-nans-are-silent-killers-i-built-a-3ms-hook-to-catch-them-at-the-exact-layer
* LightlyTrain docs: capture and replay of the failing step. https://docs.lightly.ai/train/stable/debugging/model_instability.html

Common and painful, but nantrace (with an MCP server) appeared two weeks ago and solves the forward and
backward origin at op level. Truestar should only make sure its own report emits a non finite origin
code (it already isolates per module) and not build a competing NaN tool. Score 3 x 4 x 3 = 36, lowered
because a good tool exists.

### N12. Windows GPU compile path

* pytorch/pytorch#167062, 2025-11-05, still asked 2026-09-18 "Over a year and so, had anyone resolved
  this". https://github.com/pytorch/pytorch/issues/167062
* triton-lang/triton#1640: Windows support declined for lack of bandwidth; a community fork exists.
  https://github.com/triton-lang/triton/issues/1640 and https://github.com/triton-lang/triton-windows/blob/readme/README.md

Truestar already runs its Triton kernels on this Windows machine and has Windows CI for the checker.
Proposal: make Windows GPU kernel tests part of CI and P11, so the claim is proven. Score 3 x 3 x 3 = 27.

### N13. Profilers that drop data and cannot map kernels to source

* pytorch/pytorch#192021, 2026-08-04: profiler silently drops GPU kernel events; only an INFO log shows
  it. https://github.com/pytorch/pytorch/issues/192021
* NVIDIA forums, 2025-07-30: nsys can hang while profiling distributed PyTorch. https://forums.developer.nvidia.com/t/nsys-profile-can-hang-for-a-long-time-when-profiling-pytorch-distributed-training-runs/340759
* r/MachineLearning, 2025-12-27: users want per layer, per rank timing and memory trends.
  https://www.reddit.com/r/MachineLearning/comments/1px7sqm/d_what_debugging_info_do_you_wish_you_had_when

Truestar compiled kernels know their source lines, so a kernel to line map is cheap later. Not a priority
now. Score 3 x 4 x 2 = 24.

### N14. ROCm consumer performance

* r/ROCm, 2025-07-27: "I get 4 times better performance on a L4 GPU" than a 9070 XT. https://www.reddit.com/r/ROCm/comments/1max639/the_disappointing_state_of_rocm_on_rdna4/
* Comfy-Org/ComfyUI#7332, 2025-03-21: an RTX 2060 is "4x faster" than a 9070 XT. https://github.com/Comfy-Org/ComfyUI/issues/7332

Real and large, and the Triton backend could in principle target ROCm, but Truestar has no AMD
hardware, so no claim can be verified. Correctness on ROCm stays in P1; speed waits for hardware.
Score 4 x 3 x 2 = 24.

## Do NOT take on

### N15. Fleet scale silent data corruption

* OSDI 2026 SDCHunter: benchmarks "miss over 60% of defective devices"; 23 defective GPUs studied.
  https://www.usenix.org/system/files/osdi26-zheng.pdf
* OSDI 2026 AEGIS: 18 SDC incidents and 13 faulty GPUs in 35 million GPU hours, 0.86% overhead.
  https://www.usenix.org/conference/osdi26/presentation/lei
* arXiv 2502.12340: SDCs can perturb training with no loss spike. https://arxiv.org/abs/2502.12340

Severe, but it is a fleet operations problem solved in house at ByteDance and cloud providers with
cluster access Truestar will never have. A small side benefit exists for free: N1's record and diff,
run on a suspect card against a healthy one, already acts as a workload replay test. No dedicated work.

### N16. Dataloader throughput

* pytorch/pytorch#154318, 2025-05-25, reproduced 2026-03-20: DataLoader up to 124x slower than manual
  batching. https://github.com/pytorch/pytorch/issues/154318
* MinatoLoader, arXiv 2509.10712: default loaders "can cause up to 76% GPU idleness". https://arxiv.org/html/2509.10712
* PyTorch forums, 2025-09-08: more workers made epochs 3.5x slower; cause was numpy import order.
  https://discuss.pytorch.org/t/dataloader-higher-number-of-workers-and-prefetch-factor-is-worse-than-num-workers-0-and-prefetch-factor-0/222943

Very common, but it is host side I/O, decode and scheduling, nowhere near a compiler or a numerics
checker, and TraceML, DALI and MinatoLoader already work on it. At most `truestar diag` could print an
input bound verdict; no more.

### N17. Consumer multi GPU P2P and NCCL topology

* PyTorch forums: RTX 4090 pairs hang until `NCCL_P2P_DISABLE=1`, after which transfer is about 2GB/s.
  https://discuss.pytorch.org/t/ddp-training-on-rtx-4090-ada-cu118/168366 (2022 to 2023 thread)
* Workaround post, 2025-05-21: P2P is not enabled on some GeForce boards; modified kernel modules needed.
  https://allanchan339.github.io/bug-fixes/2025/05/21/4090-P2P.html
* NCCL docs: IOMMU on Linux bare metal can cause "silent device memory corruption" with P2P.
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html

Driver, BIOS and vendor policy; nothing a compiler can change. P10 could add one diagnosis line
("P2P disabled, collectives go through host"), nothing more.

### N18. CPU inference speed race

Evidence found was mixed and partly old (a 2023 benchmark with torch 2.0.1 showing ONNX Runtime about
14x faster than eager PyTorch on CPU at batch 1, https://github.com/zhangchaosd/ModelInferBench; a
2026 sentence transformers benchmark where OpenVINO INT8 or llama.cpp lead depending on the CPU,
https://github.com/huggingface/sentence-transformers/blob/main/docs/sentence_transformer/usage/efficiency.rst).
ONNX Runtime, OpenVINO and llama.cpp are mature and vendor tuned. Racing them is the cuBLAS trap
the design already rules out. Only fused novel ops on CPU (P8) are worth measuring.

## Covered by existing rows (evidence only)

* Edge converter gaps (P7): coremltools fails on `__ior__` for gemma 3 1B, open since 2025-08-10,
  https://github.com/apple/coremltools/issues/2584 ; an ExecuTorch Core ML model works with
  `torch.jit.trace` but not `torch.export`, 2025-03-21, https://github.com/pytorch/executorch/issues/9506
* Engine portability (P7): TensorRT engines are by default tied to one TensorRT version and one compute
  capability, https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html
* XLA memory surprise informs never introduce #5 and the N8 rule; JAX compile also exhausted swap in
  jax-ml/jax discussion #29678, 2025-06-24, https://github.com/jax-ml/jax/discussions/29678

## Suggested ledger rows (proposed)

| # | Pain | Truestar answer | Acceptance test (vs baseline) |
|---|---|---|---|
| P13 | Upgrades silently change numerics (N1) | `truestar record` and `truestar diff` per module across environments | Injected op change and TF32 drift are named and classified; identical venvs report clean; baseline final logit `allclose` cannot name the op |
| P14 | LLM outputs depend on batch, TP and GPU (N2) | Invariance check plus batch invariant kernels as a verified property | Non invariant op named; Truestar kernels bitwise identical across batch 1, 2, 7, 64 at under 20% overhead |
| P15 | Trainer and sampler disagree in RL (N3) | Pair check, then one kernel set for prefill, decode and training forward | Bitwise equal logprobs at TP 1 and 2; swapped kernel named |
| P16 | Ports and engines compute a different model (N4) | Two implementation check with weight inferred module map | Three injected port bugs named at the right module; correct port clean |
| P17 | Custom ops need C++ builds tied to versions (N5) | `@truestar.function` ops registered as torch ops, no toolchain | Under 2 s cold, opcheck passes, zero graph breaks, same source on two torch versions, Linux and Windows |
| P18 | Wrong gradients from trusted low precision kernels (N6) | Per module backward replay against float64 | Backward only fault blamed while forward check is OK; clean BF16 model OK |

N7 to N10 are better as extensions of P2, P10 and P1 than as new rows.

## Notes on method and limits

* Search tools: parallel search (hit its free tier rate limit once) and exa. Firecrawl was not used.
* Reddit coverage is thinner than GitHub and papers in these results; frequency scores lean on issue
  counts, maintainer statements and paper scope rather than on survey data. No survey of practitioners
  was found for any pain here, so frequency is a judgement, labelled as such.
* Several sources are very recent (October and November 2026 arXiv preprints); they are not peer
  reviewed yet.
* Nothing here was checked against hardware; all acceptance tests are proposals.
