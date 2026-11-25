<p align="center">
  <img alt="Truestar: the same answer on every chip. Verified against float64. Fast." src="https://raw.githubusercontent.com/SharvenRane/truestar/main/docs/assets/banner.png" width="100%">
</p>

<p align="center">
  <a href="https://pypi.org/project/truestar/"><img alt="PyPI" src="https://img.shields.io/pypi/v/truestar?color=8A8F98&label=pypi&style=flat-square"></a>
  <a href="https://pypi.org/project/truestar/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/truestar?color=2B2F36&style=flat-square"></a>
  <a href="https://github.com/SharvenRane/truestar/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-5A5F68?style=flat-square"></a>
  <a href="https://github.com/SharvenRane/truestar/actions/workflows/install-check.yml"><img alt="Install check" src="https://github.com/SharvenRane/truestar/actions/workflows/install-check.yml/badge.svg"></a>
</p>

<p align="center">
  <a href="#install">Install</a> &middot;
  <a href="#what-it-finds">What it finds</a> &middot;
  <a href="docs/index.md">Docs</a> &middot;
  <a href="CHANGELOG.md">Changelog</a>
</p>

| | |
|---|---|
| **Cross chip checks** | Runs your unmodified model on every backend on the machine and compares each with a float64 reference; names the module and op where answers diverge |
| **Bitwise reproducible** | Float32 results identical across NVIDIA generations; half precision identical too with portable tensor cores |
| **Proved, not just tested** | Core numerics theorems machine checked in Lean 4; error bounds cross checked against FPTaylor |
| **Catches what others miss** | Held out kernel sets: 3.6% false alarms with fewer misses than `assert_close` or `allclose` |


Truestar tells you when PyTorch gives a different answer on another backend, names the module and
the ATen op where it starts, and writes a repro script that runs on stock PyTorch. It runs your
unmodified model on every backend on the machine (CPU, CUDA, MPS) and compares each one with a
float64 CPU reference.

Version 0.1.0, the first public release. Every number on this page was measured in 2026-11 and
links to the doc or result file it comes from. Losses are listed next to the wins.

## Install

```
pip install truestar
```

Requirements: Python 3.10 or newer (3.10 and 3.12 run in CI on Linux and Windows) and PyTorch
`>=2.11,<2.16` (2.11, 2.14.1 and 2.15.0 tested). Install PyTorch the way pytorch.org says for your
machine first. `truestar check` and the other checks are pure Python: no CUDA toolkit, no compiler.

Optional extras:

```
pip install "truestar[mcp]"          # official MCP SDK for `truestar mcp` (a builtin server is used without it)
pip install "truestar[cpu-kernels]"  # numba, for the experimental compiler's CPU kernels
pip install "truestar[onnx]"         # `truestar export --to onnx` and ONNX Runtime backends
pip install "truestar[litert]"       # `truestar export --to litert`
pip install "truestar[executorch]"   # `truestar export --to executorch` (must match your torch)
```

The experimental compiler's GPU kernels are written in Triton. PyTorch's CUDA wheels for Linux
bring Triton with them; its CUDA wheels for Windows do not, so on Windows install the community
`triton-windows` package in the version its README pairs with your PyTorch. Without Triton,
`truestar check` still works; the compiled executor on `cuda` does not.

The examples are not part of the package. They come with a checkout:

```
git clone https://github.com/SharvenRane/truestar
cd truestar
pip install .
truestar check examples/cpu_check.py
python examples/nn_mlp.py
```

## The five minute demo

[examples/notebooks/five_minute_check.ipynb](examples/notebooks/five_minute_check.ipynb) runs on a
CPU with stock PyTorch. It plants a bug in the CPU kernel of `aten::addmm` (float32 only: the bias
is added twice), leaves the model code clean, and runs one command:

```
$ truestar check planted_model.py --backends cpu
truestar check: planted_model (float32), reference CPU float64

[DIVERGENT] cpu: CPU  TS010 TS013
  final output: relative error 7.1e-01
  TS013 decisions: 3/8 predictions changed (0 near ties, 3 real)
  TS010 first divergent module: blocks.0.up#0 (Linear), own error 1.2e-01 vs threshold 1.0e-04
  source: planted_model.py:36 (planted_model.Block)
  first divergent op: aten.addmm.default (op 3 of 4 in blocks.0.up#0), own error 1.2e-01 vs threshold 1.0e-04
    inputs: float32[256], float32[96, 64], float32[64, 256]
  repro: python truestar_repro/cpu_blocks.0.up_0/repro.py
  op repro: python truestar_repro/cpu_blocks.0.up_0/op_repro.py
  ...
overall: DIVERGENT
```

Real output from the saved notebook run on 2026-11-24 (Linux, CPU only, Python 3.12, torch 2.11;
one precision line and the module list cut). The same model with the planted kernel switched off
reports `OK` with exit code 0. The generated `repro.py` exits 1 with the planted kernel loaded and
0 on stock PyTorch, and `op_repro.py` isolates the single `aten.addmm` call.

## Run it on your model

A model file defines three functions:

```python
def get_model():      # an nn.Module
def get_inputs():     # a tensor, or a tuple of forward arguments
def get_kwargs():     # optional
```

Then:

```
truestar check my_model.py                      # every backend found on this machine
truestar check my_model.py --backends cpu,cuda  # or cpu,mps
truestar check my_model.py --format agent       # one JSON document for scripts and coding agents
```

Exit codes: 0 every backend matches the reference, 1 a finding, 2 a usage error. From Python:

```python
import truestar
report = truestar.check(model, inputs)
print(report.to_text())
```

How it decides: the model runs once in float64 on the CPU as the reference. Every module is then
replayed alone on each backend with the reference inputs, so the error a module shows is its own
and never inherited from upstream. Each module's error is the relative L2 error of its output.
Thresholds start from a fixed floor per dtype (float32 1e-4, float16 5e-3, bfloat16 4e-2, TF32 5e-3)
and are raised to 10 times the error a correct CPU float32 run shows on the same module when that
is larger. Inside the first module over its threshold, ATen ops are traced on both sides and the
first op over its own threshold is named.

## Same compiled answer on two GPU architectures

`truestar fingerprint` hashes the raw output bytes of small float32 models for Truestar's compiled
executor, eager PyTorch and `torch.compile`. The claim, scoped to what was measured
([docs/CROSS_CHIP.md](docs/CROSS_CHIP.md), files in [docs/cross_chip/](docs/cross_chip/), measured
2026-11-24):

> On an NVIDIA RTX 5070 Ti (Blackwell, 70 SMs, Windows, CUDA 12.8) and a Tesla T4 (Turing, 40 SMs,
> Linux, CUDA 13.0), both with PyTorch 2.11.0 and Triton 3.7.1, Truestar's compiled float32 outputs
> with `deterministic=True` and TF32 off were bitwise identical across the two GPUs for an MLP, a
> TransformerEncoderLayer and a two layer Llama style decode step, at batch sizes 1, 7, 64 and 256
> (12 of 12 rows), and identical run to run on each GPU. Eager PyTorch and `torch.compile` differed
> between the two GPUs in every one of those 12 rows.

It does not hold, or has not been run, for: float16, bfloat16 and TF32; resnet18 (its convolutions
are cuDNN calls); a different Triton version (Triton 3.6.0 on the same T4 changed the transformer
and decode step bits); AMD, Apple and Intel GPUs; Ampere, Ada and Hopper. Truestar does not claim
bitwise equality with PyTorch or with float64; its compiled results are checked against float64
within a stated bound. An earlier run on an RTX 5060 Laptop GPU (26 SMs) matched the 5070 Ti on the
legacy cases ([docs/data/cross_chip/](docs/data/cross_chip/)).

## What it found

* Injected faults (a scaled output, zeroed channels, a NaN, a fault in some rows only) are blamed on
  the exact module in small models and inside resnet18 and ViT, and downstream modules are not
  blamed. The generated repro exits 1 under the fault and 0 without it.
* Clean real architectures report clean on CPU and an RTX 5070 Ti: MobileNetV3, EfficientNet B0,
  ViT B/16, ResNet50 and BERT, all with random weights.
* On a real Apple M2 (macOS 26.6.2), the combined copy `t.to("cpu", torch.float64)` from MPS
  returned all zeros with torch 2.14.1 and raised with torch 2.15.0; the two step copy is right on
  both ([docs/METAL.md](docs/METAL.md) section 5, measured 2026-11-25).
* On GitHub hosted macOS 26 runners, `truestar check` flagged every `nn.Linear` on MPS. The same
  probe is clean on the real M2, so that fault belongs to the runners' paravirtual GPU, not to
  PyTorch ([docs/METAL.md](docs/METAL.md)).
* `truestar kernel` on 403 held out kernels written outside this project (KernelBench samples and
  METR's KernelBenchFiltered best agent solutions, Modal L4, measured 2026-11-25): false fails
  3.6% (KB 6/168, METR 3/84) and false passes 6.4% and 7.1% (KB 7/109, METR 3/42), against
  `torch.testing.assert_close` at 35.1% and 38.1% false fails. Not every kernel finished: 37 ran out
  of GPU memory and 17 did not build or finish in time. Protocol, bounds written before the run, and
  every table: [docs/KERNEL_CHECK.md](docs/KERNEL_CHECK.md); per kernel results in
  [results/heldout_runs/](results/heldout_runs/).

## Speed, measured (2026-11)

These numbers are for the experimental compiled executor; `truestar check` does not use it.

* Training, Llama style decoder (94M params, batch 8 x 512, float32, AdamW, Modal L4): 305.4 ms per
  step against torch.compile 263.0 ms (**1.16x slower**) and eager 277.8 ms (1.10x slower)
  ([docs/TRAINING.md](docs/TRAINING.md)).
* Mixed precision training, MLP at batch 512 (Modal L4): **0.82x the speed** of the best of eager
  AMP and torch.compile AMP in both bfloat16 and float16, with loss curves within 7.4e-6 relative of
  `torch.autocast` over 200 steps ([docs/TRAINING.md](docs/TRAINING.md)).
* Training, small models (RTX 5070 Ti, provisional: the GPU was shared): with a CUDA graph, 1.7x faster than eager on an MLP at batch
  64 and 1.6x on a six layer LM at batch 8; resnet18 is 2x slower than eager and 1.4x slower than
  torch.compile, and max-autotune is 1.7x faster on the LM ([docs/TRAINING.md](docs/TRAINING.md)).
* Causal attention kernel against FlexAttention: 1.18x to 2.14x faster across six sizes (float16,
  head dims 64 and 128, sequence 2048 to 8192, RTX 5070 Ti) ([docs/QUALITY_BAR.md](docs/QUALITY_BAR.md)).
* Inference (RTX 5070 Ti, provisional until a rerun on an idle GPU,
  [docs/bench/xfmr_2026-11-24_provisional.json](docs/bench/xfmr_2026-11-24_provisional.json)):
  TransformerEncoderLayer 1.21x faster than the best of eager and torch.compile in TF32 and 1.37x in
  bfloat16, but 1.13x slower in strict float32; Llama style decode steps 1.5x to 3.0x faster than
  eager with CUDA graphs, yet **0.48x to 0.78x the speed of torch.compile max-autotune** in every
  decode case ([docs/QUALITY_BAR.md](docs/QUALITY_BAR.md), 2026-11-24 entry).

## What it checks today

Status words come from the pain ledger in [docs/QUALITY_BAR.md](docs/QUALITY_BAR.md). "Detects"
means Truestar names the problem and writes a repro; it does not fix it.

| question | command | status |
|---|---|---|
| does a backend give a different answer, and where | `truestar check` ([GETTING_STARTED](docs/GETTING_STARTED.md)) | detects; `truestar fix` swaps a blamed ATen op for a verified decomposition ([FIX](docs/FIX.md)) |
| are the gradients right | `truestar check --backward` ([GRAD_CHECK](docs/GRAD_CHECK.md)) | detects |
| does Inductor, ONNX Runtime or TensorRT change the result | `truestar check --backends cuda+inductor,ort:cpu,trt:fp16` ([BACKENDS](docs/BACKENDS.md)) | detects: 5 of 5 injected faults named, 0 false alarms in 10 clean runs |
| what did an upgrade change | `truestar record`, then `truestar diff` ([RECORD_DIFF](docs/RECORD_DIFF.md)) | detects injected changes; two real torch versions not compared yet |
| does a row depend on the rest of the batch | `truestar invariance` ([INVARIANCE](docs/INVARIANCE.md)) | detects |
| does a port compute the same model | `truestar port` ([PORT_CHECK](docs/PORT_CHECK.md), [PORT_ENGINES](docs/PORT_ENGINES.md)) | detects: injected port bugs named at the right module, correct ports clean |
| does the deployed file compute the model in its own runtime | `truestar conform` ([CONFORM](docs/CONFORM.md)) | detects; solves the float16 runtime case |
| where does it overflow in float16 | `truestar range --target fp16` ([RANGE](docs/RANGE.md)) | in emulation only: 6 of 6 cited causes found, 0 false positives on 9 model and target pairs |
| is a hand or AI written kernel right | `truestar kernel ref.py cand.py` ([KERNEL_CHECK](docs/KERNEL_CHECK.md)) | detects; held out rates above |

## Limits

* Hardware actually run: Windows with CPU and an RTX 5070 Ti; an RTX 5060 Laptop GPU; a Tesla T4 and
  Modal L4 GPUs on Linux; Linux and Windows CPU in CI; a real Apple M2 Mac mini (Metal tests 109
  passed, 1 skipped; the full suite on the Mac still had 13 failures and 3 errors,
  [docs/METAL.md](docs/METAL.md) section 5). **AMD is unverified on hardware**: the ROCm code paths
  compile for gfx942, gfx1100 and gfx1103 but have not run on an AMD GPU
  ([docs/ROCM.md](docs/ROCM.md)). Intel XPU has not been run.
* The compiled executor is slower than PyTorch in several measured cases: transformer training 1.16x
  behind torch.compile, mixed precision training at 0.82x of PyTorch AMP, resnet18 training 2x
  behind eager, decode steps at 0.48x to 0.78x of torch.compile max-autotune (Speed, above).
* The verdict per module is a relative L2 error over the whole tensor. One wrong row in a large
  bfloat16 tensor can stay under the floor: 1 bad row in 1024 gives about 0.031 against a 0.04
  floor.
* The float64 reference is only float64 where the model lets it be. Models that cast explicitly
  (`.float()`, `.to(torch.float32)`, `softmax(dtype=torch.float32)`) round the reference at those
  points, and Truestar does not flag such casts yet. PyTorch CPU float64 kernel bugs are inherited.
* Op blame names the first op whose output error crosses its threshold. When a formula loses its
  digits in float32 (catastrophic cancellation), that op can be a correctly rounded one;
  `examples/check_model.py` shows this with `aten.sub`.
* Modules are replayed in eval mode by default; `--train` covers training mode batch norm and
  dropout.
* Compiled kernels are verified on the first call per shape signature in each process: data
  dependent faults on later calls with the same shapes are not seen unless `TRUESTAR_VERIFY=all`.
* macOS install is not in CI. Python 3.11 and 3.13 are not tested.
* Every doc ends with its own Limits section.

## For coding agents

`check`, `record`, `diff`, `invariance` and `port` take `--format agent` and print one versioned
JSON document with stable `TS` codes, the blamed file and line, and a `next_actions` list with shell
commands. `truestar mcp` exposes the same checks as MCP tools. Start with [AGENTS.md](AGENTS.md);
the schema is in [docs/AGENT_INTERFACE.md](docs/AGENT_INTERFACE.md).

## Also in the box (experimental)

These are libraries and commands outside the main claims. They are tested, each doc ends with its
own Limits section, and none of them is covered by the claims above. Index: [docs/index.md](docs/index.md).

* `truestar.assert_close(fn, inputs, out)`: a test assertion with a float64 derived tolerance
  ([TOLERANCE](docs/TOLERANCE.md)).
* `@truestar.verified_autotune(...)`: Triton autotuning that drops configs whose answer is wrong
  ([KERNEL_CHECK](docs/KERNEL_CHECK.md)).
* `truestar export model.py -o model.tsx` and `truestar run model.tsx inputs.pt`: an artifact
  written only if it matches PyTorch at two batch sizes ([EXPORT](docs/EXPORT.md)).
* A native front end, `@truestar.function` (typed shapes, loops and branches compiled from source,
  [FRONTEND](docs/FRONTEND.md)).
* A shape typed IR with autodiff and transforms (`truestar.ir.vmap`, `jvp`, `jacfwd`, `jacrev`,
  [TRANSFORMS](docs/TRANSFORMS.md)).
* A compiled executor that lowers the IR onto Triton and numba kernels and checks each kernel
  against float64 on its first call per shape signature ([COMPILER](docs/COMPILER.md)), with a
  Metal backend for Apple GPUs ([METAL](docs/METAL.md)).
* Portable tensor cores: an opt in mode where float16, bfloat16 and TF32 dot products run on tensor
  cores and give the same bits on every NVIDIA generation from sm_80 on
  ([PORTABLE_TC](docs/PORTABLE_TC.md)), with Lean proofs of the core lemmas in [proofs/](proofs/)
  ([PROOFS](docs/PROOFS.md)).
* Keyed random ops that give the same bits on this CPU and this GPU: 0 of 1e8 draws differ per
  distribution ([RANDOM](docs/RANDOM.md)).
* An RL kernel set with bitwise equal per token logprobs across the training forward, prefill and
  decode on a small Llama style model ([RL_KERNELS](docs/RL_KERNELS.md)).

## Why not torch.compile, deterministic algorithms or JAX

They solve other problems. `torch.compile` and XLA change how a model runs; Truestar checks how it
runs, and treats Inductor, ONNX Runtime and TensorRT as backends to check. PyTorch's own notes limit
reproducibility to one platform and one release, and say results are not reproducible between CPU
and GPU even with identical seeds. Deterministic flags make one GPU repeat itself; they do not say
which of two GPUs is right. Details and sources: [docs/FAQ.md](docs/FAQ.md).

## Prior art and credits

Truestar builds on, borrows ideas from, or is checked against:

* Batch invariant kernels: Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" and
  [batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops); SGLang's
  [deterministic inference](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic); vLLM's
  [batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/) and its bitwise
  consistent trainer and sampler work. They ship batch invariance in production engines; Truestar's
  `invariance` and `rlpair` commands check it.
* Keyed, counter based random numbers: JAX's PRNG keys (Threefry) and the Philox and Threefry
  generators of Salmon et al., "Parallel random numbers: as easy as 1, 2, 3" (SC 2011). SplitMix64
  (Steele, Lea and Flood, 2014). Normal draws use Acklam's and Wichura's (AS241) published
  approximations.
* [Triton](https://github.com/triton-lang/triton), which generates every Truestar GPU kernel; the
  Triton fused attention tutorial and FlashAttention (Dao et al.) for the attention schedule.
* [ONNX Runtime](https://onnxruntime.ai/), TensorRT and Polygraphy, torch.compile and Inductor,
  FlexAttention, LiteRT, ExecuTorch and llama.cpp, all run or checked as backends and engines.
* Integer rounding conventions from gemmlowp, CMSIS-NN and TFLite, reproduced bit for bit by
  `truestar intref`.
* [DLPack](https://github.com/dmlc/dlpack) and [safetensors](https://github.com/huggingface/safetensors)
  for interop; numba for CPU kernels.
* Held out test data: [KernelBench](https://github.com/ScalingIntelligence/KernelBench) (MIT) and
  its kernelbench-samples dataset (Apache 2.0), stored with attribution in
  [tests/data/heldout/kernelbench/](tests/data/heldout/kernelbench/README.md); METR's
  KernelBenchFiltered is downloaded on demand and not redistributed.
* The evidence for the pain ledger: PyTorch's reproducibility notes, NVIDIA's cuDNN determinism
  notes, "Mind the Gap" (arXiv 2511.11601), KernelBench (Stanford) and HUD's YAKK, and the issue
  threads cited row by row in [docs/QUALITY_BAR.md](docs/QUALITY_BAR.md).

## License

Apache 2.0, see [LICENSE](LICENSE). Author: Sharven Rane. No telemetry: nothing leaves the machine
unless you share a report. Security issues: see [SECURITY.md](SECURITY.md).
