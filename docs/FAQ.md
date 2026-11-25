# FAQ

Answers for a skeptical first reader. External sources were checked on 2026-11-24. Numbers come from
the dated status log in [QUALITY_BAR.md](QUALITY_BAR.md) or from the files in
[data/cross_chip/](data/cross_chip/); anything not measured yet says so.

## What is Truestar, in one paragraph?

A checker for PyTorch models. `truestar check model.py` runs the model once in float64 on the CPU
as a reference, then replays every module alone on each backend (CPU float32, CUDA, MPS, or a
compiler or runtime such as Inductor, ONNX Runtime or TensorRT) with the reference inputs. It
reports the first module whose own error is over its threshold, the ATen op inside it, the line in
your code, and writes a repro script that needs only PyTorch. The rest of the repository (an IR, a
compiler, kernels) is experimental and outside the launch scope.

## Why not just use torch.compile?

`torch.compile` makes a model faster. It does not tell you whether the answer is right, and it can
itself change the answer: PyTorch's issue tracker carries a `module: correctness (silent)` label,
and of the 1,062 issues with that label opened since 2025, 503 also carry `oncall: pt2` (counted in
the pain ledger, row P22). Truestar treats Inductor as one more backend to check:
`truestar check model.py --backends cuda+inductor`. See [BACKENDS.md](BACKENDS.md).

## Why not `torch.use_deterministic_algorithms(True)`?

It makes one GPU give the same bits run after run. It says nothing about whether those bits agree
with another GPU, another backend or the math. PyTorch's own reproducibility note says
"Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or
different platforms", and that results need not be reproducible "between CPU and GPU executions,
even when using identical seeds"
(https://docs.pytorch.org/docs/2.12/notes/randomness.html). NVIDIA's cuDNN documentation likewise
scopes its determinism to one GPU architecture and software version
(https://docs.nvidia.com/cudnn/operations/determinism.md). Determinism and correctness are
different questions; Truestar answers the second.

## Why not JAX?

JAX is a different framework, and switching frameworks is not a debugging step. It also does not
make results equal across chips: XLA's deterministic flags leave out nondeterministic ops, and its
autotuning picks kernels per GPU architecture (https://openxla.org/xla/determinism). Where JAX is
ahead, Truestar says so: JAX's keyed random numbers already give the same uniform bits on CPU, GPU
and TPU, and Truestar's keyed random ops follow the same idea ([RANDOM.md](RANDOM.md)). If you have
a Flax port of a PyTorch model, `truestar port reference.py jax:candidate.py` checks that the two
compute the same model ([PORT_ENGINES.md](PORT_ENGINES.md)).

## How is float64 a fair reference?

float64 carries 53 bits of mantissa against 24 for float32, 11 for float16 and 8 for bfloat16. The
rounding error a float64 run adds is about nine orders of magnitude below a float32 run's, so for
the question "is this float32 or float16 result right" the float64 result is the closest thing to
the exact answer that runs on any machine. It is also vendor neutral: every vendor checker we know
of uses its own CPU path as ground truth and covers one runtime; the float64 CPU path belongs to no
backend under test.

Where it is not fair, and what Truestar does about it:

* **Your model can downcast inside the reference.** `.float()`, `.to(torch.float32)` or
  `softmax(..., dtype=torch.float32)` round the float64 reference at that point, so it can share
  rounding with the float32 backend and hide its error. Truestar does not flag such casts yet.
* **The reference runs PyTorch's CPU float64 kernels**, so their bugs are inherited. The project
  found one itself: CPU float64 `sqrt` is not correctly rounded ([RANDOM.md](RANDOM.md)).
* **Some formulas lose their digits in float32 whatever the kernel does.** Catastrophic
  cancellation is real float32 behavior, and Truestar reports it as a divergence because the float32
  answer is wrong. The op it names can be a correctly rounded one (`aten.sub` in
  `examples/check_model.py`). Separating an ill conditioned formula from a wrong kernel is planned.
* **Thresholds are not zero.** A module passes when the relative L2 error of its output is under a
  floor per dtype (float32 1e-4, float16 5e-3, bfloat16 4e-2, TF32 5e-3), raised to 10 times the
  error a correct CPU float32 run shows on that module when that is larger. Because the measure is
  over the whole tensor, one wrong row in a large bfloat16 tensor can stay under the floor (1 bad
  row in 1024 gives about 0.031 against 0.04). A per element check is planned.

On clean real architectures the result is clean: MobileNetV3, EfficientNet B0, ViT B/16, ResNet50
and BERT (random weights) report OK on CPU and an RTX 5070 Ti.

## Does Truestar give "the same answer on every chip"?

No, and the docs no longer say so. Two different things are easy to mix up:

1. **`truestar check`** does not need bitwise equality. It asks whether each backend is within a
   stated bound of float64, and names where it is not. That works on any backend PyTorch runs on.
2. **Truestar's experimental compiled executor** gave bitwise equal outputs on two GPUs in one
   measurement. On an RTX 5070 Ti (70 SMs) and an RTX 5060 Laptop GPU (26 SMs), both sm_120, with
   the same driver, CUDA 12.8, PyTorch 2.11, Triton 3.7.1 and cuDNN 9.19, the compiled float32
   outputs of a small MLP, a TransformerEncoderLayer and resnet18 had identical SHA256 hashes, run
   to run and across the two GPUs. Eager PyTorch matched across the two GPUs on the MLP and resnet18
   and differed on the transformer layer. Batch 4, TF32 off, deterministic mode (no autotuning, no
   split K).

What it does not mean: it is not a result across GPU architectures (both GPUs are consumer
Blackwell), across vendors, across Triton or driver versions, in half precision, with TF32, or with
tuned configs, and we expect differences there. The compiled output is not bit equal to PyTorch's
(it differs on every case) or to float64; it is checked against float64 within a bound. resnet18's
convolutions are cuDNN calls inside the executor, so their agreement is cuDNN's, not Truestar's. The
script is `scripts/laptop_fingerprint.py`; the data is in [data/cross_chip/](data/cross_chip/). A
third architecture is pending.

## Does it find real bugs, or only planted ones?

So far the documented catches are injected faults and one unconfirmed finding. Injected faults
(scaled outputs, zeroed channels, NaN, faults in some rows only, a broken op below the module level)
are blamed on the right module and op, in small models and inside resnet18 and ViT. On GitHub hosted
macOS 26 runners, `truestar check` flagged every `nn.Linear` on MPS and a probe narrowed it to
`F.linear` with bias; macOS 15 with the same torch was clean. Those runners expose a paravirtual
GPU, so this is not confirmed on Apple hardware and has not been reported upstream. Rediscovering a
listed upstream bug on physical hardware is still open (pain ledger row P1).

## What does it cost?

Money: nothing. Apache 2.0, no account, no telemetry; nothing leaves the machine unless you share a
report.

Time and memory: one float64 forward pass on the CPU, one forward pass per backend, and one replay
per module per backend, plus memory to hold each module's reference inputs. Op tracing runs only
inside the first divergent module. Measured run times are pending a fair benchmark rerun; the
five minute notebook ([five_minute_check.ipynb](../examples/notebooks/five_minute_check.ipynb))
runs end to end on a CPU.

## Which platforms does it support?

* Python 3.10 and 3.12 in CI, on Linux and Windows, CPU only. PyTorch 2.11 to 2.15
  (pyproject.toml allows `>=2.11,<2.16`).
* GPUs it has run on: an RTX 5070 Ti (Windows), an RTX 5060 Laptop GPU (the fingerprint run), and a
  Tesla T4 on Kaggle, where only the 24 test first build suite ran, before the compiler existed.
* MPS: GitHub hosted macOS 15 and 26 runners only (paravirtual GPU).
* Not tested: ROCm, Intel XPU, physical Apple hardware, Python 3.11 and 3.13. A clean macOS install
  is not in CI yet.
* `truestar check` itself needs no CUDA toolkit and no C++ compiler. Compiler and runtime backends
  (Inductor, ONNX Runtime, TensorRT) need their own packages and are never picked by default.

## Is it on PyPI?

Not yet. Until it is, install from a checkout with `pip install .` after installing PyTorch.

## What is the license?

Apache 2.0, including its patent grant. Author: Sharven Rane. Everything users depend on ships
under that license; there is no closed core.

## Names in the docs that differ from the code

Some public names are being renamed so they do not collide with other projects. New docs use the
new names; until the rename lands, the code still uses the old ones:

| new name | in code today |
|---|---|
| `.tstar` artifact files | `.tsx` |
| `@truestar.native` | `@truestar.function` |
| `truestar.ir.batched` | `truestar.ir.vmap` |
| `truestar.ir.pushforward` | `truestar.ir.jvp` |
| `truestar.ir.jacobian(mode="forward" or "reverse")` | `truestar.ir.jacfwd`, `truestar.ir.jacrev` |
| `truestar.assert_derived_close` | `truestar.assert_close` |
