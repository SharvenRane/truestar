# Keyed random numbers and training mode

Author: Sharven Rane. First written 2026-11-24.

Pain row P20 (docs/QUALITY_BAR.md). Code: `src/truestar/rng.py` (the generator),
`src/truestar/ir/train_ops.py` (training batch norm and dropout as IR ops), `src/truestar/trainmode.py`
(training mode in the checkers). Tests: `tests/test_rng.py`, `tests/test_frontend_random.py`,
`tests/test_train_ir.py`, `tests/test_trainmode.py`. Evidence scripts: `scripts/random_vs_torch.py`,
`scripts/train_mode_resnet18.py`.

## The problem

PyTorch says it plainly: "results may not be reproducible between CPU and GPU executions, even when
using identical seeds" ([randomness notes](https://docs.pytorch.org/docs/main/notes/randomness.html)).
Users keep hitting it: the same seed gives other numbers on GPUs with another SM count
([pytorch#62467](https://github.com/pytorch/pytorch/issues/62467)), on CPU and CUDA
([pytorch#79496](https://github.com/pytorch/pytorch/issues/79496)), and across machines, closed with
"There is no way to guarantee reproducibility across machines"
([pytorch#158398](https://github.com/pytorch/pytorch/issues/158398)). Measured here
(`scripts/random_vs_torch.py`, torch 2.11, RTX 5070 Ti): after `torch.manual_seed(0)`, 1e6 draws on
CPU and on CUDA differ in 1,000,000 of 1,000,000 places for `rand` and `randn`, 999,977 for
`randint`, and 179,563 dropout mask entries differ at p = 0.1.

The second half of the problem is training mode. Every checker in Truestar replayed modules in eval
mode, because a training forward cannot be compared across backends: dropout masks differ per
device, and batch norm mutates its running statistics in place, so a replayed module starts from
statistics another call already changed. The training forward is the one every training step runs,
so a kernel that is wrong only in training (a batch norm training kernel, its running statistic
update, its backward) was invisible to every check.

## Usage, before and after

PyTorch: a global generator, order dependent, other bits on another device.

```python
torch.manual_seed(42)
noise = torch.randn(1, 64, 2048, device="cuda")    # differs from the CPU draw and across GPUs
x = F.dropout(x, p=0.1, training=True)             # consumes global state: reordering code changes masks
```

JAX: explicit keys that the user threads and must never reuse.

```python
key = jax.random.key(42)
key, k1, k2 = jax.random.split(key, 3)             # every call site needs its own split key
x = x * jax.random.bernoulli(k1, 0.9, x.shape) / 0.9
noise = jax.random.normal(k2, (1, 64, 2048))
```

Truestar: nothing to thread. Inside a compiled function the key comes from the function's seed and
its call count, the name from the op's place in the source.

```python
import truestar
from truestar import Tensor, ops

@truestar.function(seed=42)
def step(x: Tensor["B", 512]):
    h = ops.dropout(x, 0.1)                         # site step/dropout#0
    return h + 0.01 * ops.randn(h.shape)            # site step/randn#0
```

Escape hatch: `key=` (an int or an int64 scalar tensor) and `name=` (a site name) on any random op.
Eagerly (outside a compiled function) `key=` is required, since there is no function seed to derive
from; the error says so. `step.key_for(n)` gives the key of call n and `step.sites` the site names,
so any draw can be reproduced eagerly: `ops.randn(shape, key=step.key_for(0), name="step/randn#0")`
returns the same bits as call 0.

Counted on the two snippets above: PyTorch has one global state to reason about and no portable
bits; JAX needs 2 extra lines and 3 key variables, plus the rule that a key is never reused
(breaking it is silent: correlated noise); Truestar needs 0 extra lines and 0 key variables, and
reusing a site is a compile error (`duplicate_site`) rather than silent correlation.

Training batch norm, before and after:

```python
model.train(); y = bn(x)                            # PyTorch: running_mean changes in place, hidden
y, mean, var = ops.batch_norm(x, w, b, mean, var, training=True, momentum=0.1)   # Truestar: state in, state out
```

Checking a model in training mode: `truestar check model.py --train`, `truestar check model.py
--backward --train`, `truestar invariance model.py --train` (or `train=True` in Python).

## How the bits are made the same everywhere

Every value is a pure function of (key, site, element index). The stream key of an op is
`fold(fold(key, site), stream...)` with `fold(a, b) = mix(mix(a) ^ b)` and `mix` = SplitMix64, the
mixer `multinomial` already used: `uniform01(seed, offset, counter)` is exactly the float64 uniform
at `stream_key(seed, offset)`, so the existing sampling op is a special case. Element i takes the 64
bits `h = mix(K ^ i)`; a draw can be made in pieces (`start=`) and is the same.

Each distribution is defined with integer operations and with float operations IEEE 754 rounds
exactly (add, subtract, multiply, divide, square root), in one fixed order, with no fused multiply
add:

| distribution | definition |
|---|---|
| uniform | `(h >> (64 - p)) * 2**-p`, p = 24 (f32), 53 (f64), 11 (f16), 8 (bf16): exact in the dtype, in [0, 1) |
| bernoulli(p) | `(h >> 11) < ceil(p * 2**53)`, which is exactly `uniform53 < p`; no float work |
| randint(low, high) | `low + (((h >> 32) * (high - low)) >> 32)`, high - low <= 2**31 (bias under (high - low) / 2**32) |
| normal, float64 | sign from bit 63, `qq = (2m + 1) 2**-54` from 52 bits, Acklam's rational inverse CDF (relative error 1.13e-9 measured against `torch.special.ndtri`) |
| normal, float32 and half | the same qq rounded once to float32, Wichura's AS241 PPND7 in float32 arithmetic (within 5.6 ulps of the exact quantile of that qq), half types rounded from float32 |

The logarithm is an atanh series on the frexp mantissa, never a library `log`. Three platform traps
were found and fixed while building it, each pinned by a test:

* `torch.sqrt` on CPU float64 is not correctly rounded (1 ulp off for 1 of 3000 inputs); the torch
  path uses numpy's hardware sqrt on the CPU.
* Triton's float32 `/` and `sqrt` are approximate, and Triton fuses multiply add by default; the
  kernel uses `div_rn`, `sqrt_rn` and `enable_fp_fusion=False`. Triton passes Python floats as
  float32, so the float64 kernel reads its constants from a table; the float32 kernel writes them as
  literals that are float32 values already (a test checks every literal against the table).
* Triton compiles `-z` as `0 - z`, which turns -0.0 into +0.0. The 1e8 draw comparison caught 2 such
  normals; the kernel multiplies by -1.0.

Acklam's central rational has alternating coefficients near 200 and was measured up to 3000 ulps off
when evaluated in float32, which is why float32 uses PPND7 (all coefficients positive).

Four implementations exist: `python` (the definition, one element at a time), `torch` (vectorized
torch ops on any device; the reference interpreter uses it), `numba` (the compiled CPU kernel) and
`triton` (the compiled CUDA kernel). `rng.generate(..., backend="auto")` picks the compiled one.

## In the IR, the front end and the compiler

* IR op `random(key, *streams)` with attributes `dist`, `shape`, `dtype`, `site` (the 64 bit id),
  `site_name`, `p`, `low`, `high`. It has a reference implementation and no gradient (NONDIFF).
* Dropout and training batch norm are not new IR ops: they are compositions in
  `ir/train_ops.py` (`random`, `where`, `mul`; `mean`, `sub`, `rsqrt`, ...), so autodiff, the
  compiler and the serializer cover them unchanged. Batch norm's running statistics are inputs and
  outputs: `(y, new_mean, new_var)`, with torch's momentum rule and unbiased variance
  `var * n / (n - 1)`; with a symbolic batch, n is a run time `dim_scalar`.
* Front end: `ops.rand`, `randn`, `normal`, `bernoulli`, `randint`, `rand_like`, `randn_like`,
  `dropout`, `batch_norm`; `F.dropout` is accepted (its training flag must be static);
  `torch.rand` and friends are refused with the keyed op to use. Sites: `<function path>/<op>#<k>`
  or `<function path>/<name>`; inlined calls add `<callee>#<k>`, unrolled static loops add `[i]`,
  and loops over a symbolic length or a tensor condition pass their iteration index as a stream, so
  every iteration draws fresh numbers. The program gets an int64 input `rng_key` (`Program.key`)
  that the function fills with `rng.call_key(seed, call)`.
* Importer: `ir.import_module(model, args, training=True)` turns the buffers a training graph
  mutates (running statistics, `num_batches_tracked`) into state inputs after the user inputs and
  state outputs after the user outputs (`Program.state`, starting values in `Program.state_init`,
  `ir.initial_state(prog)`), in both the export graph (`aten.batch_norm` with training=True, in place
  `add_`) and the core ATen graph (`_native_batch_norm_legit_functional`, BUFFER_MUTATION outputs).
  Dropout becomes a keyed draw at the site `<module path>/dropout#k`. Without `training=True` a
  training graph is refused with that hint, as before.
* Compiler: `random` runs as its own step through the Triton or numba generator, and the first run
  checks it bit for bit against the reference interpreter; the mask then fuses into the following
  kernel (`linear + mul + where + relu` in one kernel on both backends).

## Training mode in the checkers (`truestar/trainmode.py`)

`check`, `grad_check` and `invariance` take `train=True` (CLI `--train`). Inside the check:

* `torch.nn.functional.dropout` (used by `nn.Dropout`, `nn.MultiheadAttention`, the Transformer
  layers) and `scaled_dot_product_attention` with `dropout_p > 0` are replaced by keyed versions. A
  mask is a function of (seed, module name, the module's call index, the dropout's ordinal inside
  that call): the float64 reference, every backend and every isolated replay of a module draw the
  same mask. The mask is drawn outside the dispatch modes, so it is not an op of the model.
* The buffers of every module call's subtree are recorded at the start of the call in the
  reference run and loaded before that call is replayed alone; the buffers after the call are
  compared like extra outputs. A module is judged on its training forward and on its running
  statistic update.
* Any other random ATen op run by the model (`bernoulli_`, `native_dropout`, `randn`, fused attention
  dropout) raises `TrainModeError`: comparing backends that drew different numbers would be a false
  alarm, and switching randomness off would silently check another model. The checker's own random
  targets, drawn outside any module, are allowed.
* `grad_check`: the backward goes through the same keyed masks. In training mode, batch statistics
  couple every element, and a correct float32 run of a parent module is far off float64 (plain
  PyTorch float32 against float64 on resnet18 at batch 16: up to 1.1e-2, eval mode 6.8e-7), so
  parents keep their measured noise floors uncapped; leaves, where a wrong kernel shows, keep the cap.
* `invariance`: masks are drawn once per row shape and broadcast along dim 0, so a row's mask does
  not depend on its batch position. Training batch norm is reported batch dependent (TS052), which it
  is by definition; `BatchNorm1d` cannot even run one row alone, and the report says so.
* Standalone repro scripts are not written in training mode (they would draw PyTorch's own masks);
  the report says so.

## Measured (2026-11-24, torch 2.11, RTX 5070 Ti, CUDA 12.8, Windows)

Bitwise agreement (`scripts/random_vs_torch.py`), 1e8 draws per distribution, key 2026:

| distribution | Triton vs numba | Triton vs reference path (1e7) |
|---|---|---|
| uniform float32 | 0 of 1e8 differ | 0 of 1e7 |
| normal float32 | 0 of 1e8 (2 before the -0.0 fix) | 0 of 1e7 |
| normal float64 | 0 of 1e8 | 0 of 1e7 |
| bernoulli p = 0.9 | 0 of 1e8 | 0 of 1e7 |
| randint [0, 50257) | 0 of 1e8 | 0 of 1e7 |

PyTorch with one seed, CPU against CUDA, 1e6 draws: rand 1,000,000 differ, randn 1,000,000, randint
999,977, dropout mask (p = 0.1) 179,563.

Statistics (tests/test_rng.py, 1e6 draws, fixed keys): mean, variance, skewness, kurtosis and
P(|z| > 4) within 5 sigma; Kolmogorov Smirnov distance under 2.1/sqrt(n) for uniform and normal in
float32 and float64; chi square of randint under the p = 1e-4 bound; correlation between keys, sites,
streams, swapped streams and adjacent counters under 5/sqrt(n); each of the 64 output bits set half the
time within 5 sigma.

Throughput on CUDA, 1e8 elements, ms per call, Truestar Triton kernel against torch (shared GPU, so
treat as approximate):

| op | Truestar | torch | ratio |
|---|---|---|---|
| uniform float32 | 0.49 | 0.55 | 0.89 |
| bernoulli mask | 0.21 | 0.80 | 0.27 |
| normal float32 | 0.96 | 0.50 | 1.90 |
| normal float64 | 59.5 | 11.6 | 5.1 |
| dropout p = 0.1 (eager, 3 passes) | 4.62 | 2.67 | 1.73 |

Uniform and bernoulli are at or above torch. The normal rows above are the first kernel, which
computed every branch of the inverse CDF for every element. The kernel now does less work for the
same bits (`scripts/random_vs_torch.py` draws are unchanged; 0 of 1e8 draws differ from the first
kernel on a T4, an L4 and an A10G, in float32 and float64):

* float32: each element selects its branch's coefficients and argument, then runs one rational and
  one division instead of three; a lower degree denominator gets a leading zero coefficient, which
  is exact. The r > 5 coefficients are selected only in a block that holds such a draw.
* float64: the central rational runs for every element; the 4.85% of draws in the tail are listed
  per block (a prefix sum gives each one its slot) and the float64 logarithm and square root run only
  on that dense list. The same split for float32 measured slower than the per element select.

Median of round robin timing, 1e8 draws, against `torch.randn` on the same GPU (2026-11-24, Modal,
torch 2.11, `NORMAL_LAUNCH` defaults):

| GPU | float32 first kernel | float32 now | float64 first kernel | float64 now |
|---|---|---|---|---|
| A10G | 1.40x | 1.15x | 3.13x | 1.32x |
| L4 | 1.44x | 1.40x | 3.13x | 1.32x |
| T4 | 2.45x | 1.92x | 3.11x | 1.32x |

float32 is still compute bound on these GPUs: the definition's arithmetic (no fused multiply add,
IEEE division and square root, and the tail's logarithm and square root on every float32 draw,
since a warp almost always holds a tail draw) costs more than `torch.randn`'s Philox and Box Muller.
The RTX 5070 Ti was not retaken. Eager dropout is three passes (mask, multiply, where); in compiled programs the mask fuses
into the next kernel, and a fused eager dropout kernel is not built yet.

Training mode in the IR (`scripts/train_mode_resnet18.py`): torchvision resnet18 in training mode,
imported with explicit state (60 state tensors), trained 20 SGD steps on the float64 reference with IR
autodiff next to PyTorch float64 autograd. Starting each step from PyTorch's weights and statistics,
the worst relative error over the 20 steps (loss, every gradient, updated weights, every running
statistic) is 1.0e-12 in both graph forms. Free running, both the IR run and a control (PyTorch against
PyTorch with weights moved by 1e-15) grow about 100x per step and reach 3 at step 20: training at lr
0.05 is chaotic, so long horizon agreement is not a meaningful bar for any implementation.

Training mode in the checkers (`tests/test_trainmode.py`):

| injected fault (10%) | eval mode check | training mode check |
|---|---|---|
| batch norm training kernel output, CPU (`native_batch_norm`, training=True) | OK, 0 hits | DIVERGENT at `1#0`, op `aten.native_batch_norm.default` |
| same, CUDA (`cudnn_batch_norm`) | OK | DIVERGENT at `1#0` |
| batch norm running statistic update only, CPU and CUDA (forward output right) | OK | DIVERGENT at `1#0` (state compared) |
| batch norm training backward, CPU (`native_batch_norm_backward`, train=True) | grad_check OK | grad_check TS180 at the batch norm, op named |
| same, CUDA (`cudnn_batch_norm_backward`) | grad_check OK | grad_check TS180 at the batch norm |

Clean models report OK in training mode on CPU and CUDA: a conv, batch norm, dropout model, a two
layer `nn.TransformerEncoder` with dropout, `nn.MultiheadAttention` with attention dropout through
`scaled_dot_product_attention`, and resnet18 at batch 4 and 16 (check and grad_check, TF32 on).
Two permanent tests show the checks can fail: keys that depend on the dtype make the clean dropout
model DIVERGENT at the dropout, and skipping the state restore makes the clean batch norm model
DIVERGENT.

## Keyed stochastic rounding (`truestar/quant/sround.py`)

Stochastic rounding to FP8, FP4, bfloat16 and float16 draws its bits from the same keyed
stream: element i of a call at (key, site, step) uses the top 32 bits of `mix(K ^ i)` with
`K = stream_key(key, site, (step,))`, so a rounded tensor is identical on the CPU, on CUDA through
torch ops, in its Triton kernel and in the one element at a time definition. The rounding rule is
integer arithmetic on the input's float32 bits. docs/QUANT.md (Keyed stochastic rounding) has the
rule, the bias bound and the measurements.

## Limits

* Throughput: normal draws take 1.15x to 1.9x (float32) and 1.32x (float64) the time of
  `torch.randn` on an A10G, an L4 and a T4 (table above; the first kernel measured 1.9x and 5.1x on
  the RTX 5070 Ti, not retaken); eager dropout is 1.73x slower than `F.dropout`. Not yet measured on
  MPS or ROCm; the bitwise tests pass on this CPU, the RTX 5070 Ti, an L4 and a T4.
* randint ranges above 2**31 and bernoulli with a tensor of probabilities are not supported (use
  `ops.rand(...) < p`).
* The op drilldown cannot name an op when the reference and the backend run different batch norm ops
  (CPU `native_batch_norm`, CUDA `cudnn_batch_norm`), as in eval mode; a fault only in the running
  statistic update is blamed at the module, not at an op.
* Training mode checks key `F.dropout` and attention dropout only; channel dropout (`dropout2d`),
  `alpha_dropout` and custom random ops are refused, not keyed. `momentum=None` batch norm is checked
  (state restore covers `num_batches_tracked`) but cannot be imported (torch.export fails on it).
* Invariance in training mode shares one mask across rows; a per row mask that is independent of the
  batch position for every layout (sequence first included) is not built.
* `vmap` over random ops does not exist yet (R7): when it does, the stream mechanism is how batched
  draws stay independent.
* The serializer stores `Program.state` and `Program.key` as optional fields; artifacts from before
  have none, and their hashes are unchanged.
