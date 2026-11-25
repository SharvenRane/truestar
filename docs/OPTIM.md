# truestar.optim and the training step

Author: Sharven Rane. First written 2026-11-24. Pain row: P36 (docs/QUALITY_BAR.md).
Code: `src/truestar/optim/__init__.py`, `src/truestar/fitting.py`; tests `tests/test_optim.py`,
`tests/test_fit.py`; layers in docs/NN.md. The older IR level trainer for imported PyTorch programs
is docs/TRAINING.md; this is the native path.

## Use

```python
import truestar
from truestar import Tensor, nn, optim

net = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 10))

@truestar.native
def loss(p: net.spec, x: Tensor["B", 32], y: Tensor["B", "i64"]) -> Tensor[()]:
    return nn.cross_entropy(net(p, x), y)

opt = optim.AdamW(lr=optim.warmup_cosine(3e-3, warmup=100, total=5000), weight_decay=0.01)
step = truestar.train_step(loss, opt, clip=1.0)           # executor="cpu" or "cuda" to compile
ts = step.init(net.init(seed=0))                          # TrainState(params, opt, step, model_state, amp)
ts, history = truestar.fit(step, ts, batches, log_every=100, checkpoint="run.safetensors",
                           checkpoint_every=1000)
ts = step.load("run.safetensors")                         # resume: bit identical continuation
```

Or by hand: `ts, r = step(ts, x, y)` with `r.loss`, `r.grad_norm`, `r.lr`, `r.skipped`, `r.aux`.

## Optimizers are pure functions over trees

An optimizer is an immutable config with two methods written in the native subset:

* `scalars(self, t, lr)`: the per step 0 dim values (t is the 1 based update number, lr the
  learning rate of this update);
* `leaf(self, p, g, s, k)`: one tensor's update, returning `(new p, new slots)`.

`opt.init(params)` gives `{slot: tree}` of zeros; `opt.update(params, grads, state, step)` applies the
two methods eagerly over any trees and returns new trees (nothing is mutated; tested). The training
step translates the same two methods from their source into its program, so the eager update and
the compiled one are one definition (tested equal to 1e-15 over 6 AdamW steps).

| Optimizer | Update | Equals |
|---|---|---|
| `SGD(lr, momentum=0, nesterov=False, weight_decay=0)` | `buf = m buf + g`; `d = g + m buf` (Nesterov) or `buf` | `torch.optim.SGD` with dampening 0 |
| `Adam(lr, b1, b2, eps, weight_decay)` | L2 decay added to g, then the Adam step | `torch.optim.Adam` |
| `AdamW(lr, b1, b2, eps, weight_decay=0.01)` | `p *= 1 - lr wd`, then the Adam step | `torch.optim.AdamW` |
| `Muon(lr, weight_decay, momentum, nesterov, ns_coefficients, ns_steps, ns_dtype)` | 2D leaves: momentum, Newton Schulz orthogonalization, shape ratio; other leaves AdamW | `torch.optim.Muon` (up to bf16 rounding); orthogonality checked against float64, docs/TRAIN_NUMERICS.md |
| `Lion(lr=1e-4, b1=0.9, b2=0.99, weight_decay=0)` | `c = b1 m + (1 - b1) g`; `p *= 1 - lr wd`; `p -= lr sign(c)`; `m = b2 m + (1 - b2) g` (one slot) | Algorithm 2 of Chen et al. 2023 (torch has no Lion) |
| `RMSprop(lr, alpha, eps, weight_decay, momentum, centered)` | `v = alpha v + (1 - alpha) g^2`, centered `sqrt(v - a^2)`, optional momentum buffer | `torch.optim.RMSprop` |
| `Adagrad(lr, lr_decay, weight_decay, initial_accumulator_value, eps=1e-10)` | `sum += g^2`; `p -= lr / (1 + (t - 1) lr_decay) g / (sqrt(sum) + eps)` | `torch.optim.Adagrad` |

**Bias correction without cancellation.** `c = 1 - b^t` is computed as `-expm1(t ln b)`. For small
t, `b^t` is close to 1 and `1 - b^t` cancels: in float32 at b = 0.999 the naive form loses about
1e-5 relative at t = 1, which is what made the compiled AdamW step fail its kernel check
(docs/MEMORY.md Limits, 2.3e-5 against 2e-5). `ops.expm1` is accurate everywhere (a degree 17 series
on [-1/2, 1/2], `exp(x) - 1` outside, where the subtraction loses at most about one bit).
Measured on the compiled CPU executor in real float32 for t from 1 to 1e6: within 4 float32 ulps,
while `1 - exp(t ln b)` on the same machine is off by more than 5e-6 (`tests/test_optim.py`). A
compiled float32 AdamW step now passes its first call check (tested).

**Schedules** are frozen dataclasses mapping the 0 based update count to a learning rate, evaluated
on the host in float64 and fed in as a 0 dim input, so a schedule never recompiles:
`optim.constant`, `optim.warmup_cosine(peak, warmup, total, final, start)`,
`optim.warmup_linear(peak, warmup, total, final)`, or any callable. Tested against
`torch.optim.lr_scheduler.LambdaLR` step for step.

Also (2026-11-25), each equal to its torch scheduler within 1e-12 relative for 60 to 100 updates
(`tests/test_optim_more.py`):

| Schedule | torch equivalent |
|---|---|
| `optim.step_decay(lr, every, gamma=0.1)` | `StepLR` |
| `optim.multistep(lr, milestones, gamma=0.1)` | `MultiStepLR` |
| `optim.exponential(lr, gamma)` | `ExponentialLR` |
| `optim.polynomial(lr, total, power=1, final=0)` | `PolynomialLR` (final 0) |
| `optim.cosine_restarts(lr, period, mult=1, final=0)` | `CosineAnnealingWarmRestarts` |
| `optim.one_cycle(peak, total, pct_start, div, final_div, anneal, three_phase)` | `OneCycleLR` |
| `optim.warmup(schedule_or_lr, steps, start=0)` | `SequentialLR([LinearLR(start, 1, steps), schedule], [steps])` |

Every schedule is a pure function of the update count, so resuming from a checkpoint needs only
the count (torch's chainable schedulers carry their own state and must be saved too).

**EMA of the weights.** `optim.EMA(inner, decay=0.999, warmup=False)` wraps any optimizer and
adds one slot, `"ema"`, updated after each step as `ema + (p - ema)(1 - d)` from the new weights
(`warmup`: `d = min(decay, (1 + t) / (10 + t))`). Being an optimizer, it is compiled into the step
program and carried by `save` / `load`; `ts.opt["ema"]` is the averaged tree. 20 compiled float64
AdamW steps match `torch.optim.swa_utils.AveragedModel(multi_avg_fn=get_ema_multi_avg_fn(d))`
(seeded with the initial weights) within 1e-12; the eager update equals the compiled one to 1e-13.

**Clipping.** `optim.clip_by_global_norm(grads, max_norm)` and `optim.global_norm(grads)`, inline
native functions over a list, with the factor of `torch.nn.utils.clip_grad_norm_`
(`min(1, max / (norm + 1e-6))`), tested against it. Since 2026-11-25 `norm_type` may be 2, 1 or `inf` (tested against
`clip_grad_norm_(norm_type=...)` within 1e-14), and `optim.clip_by_value(grads, limit)` is
`clip_grad_value_`. `train_step(clip=..., clip_norm_type=...)` or `train_step(clip_value=...)`
compiles either into the step (15 SGD momentum steps match torch with the same clipping within
1e-12; passing both is refused; the reported `grad_norm` is taken before clipping).

## The step program

`train_step(loss_fn, optimizer, params="p", state=None, clip=None, precision=None, executor=None,
static=None)` builds one IR program:

forward (the loss function's program), backward (the IR autodiff pass), the non finite check, the
global norm and clipping, the optimizer's scalars and per tensor rules, and with float16 the loss
scale update. Inputs: the parameter leaves, the model state leaves, the batch, the random key, `lr`
and `t` (float64 0 dim), every optimizer slot, and the loss scale. Outputs: loss, gradient norm, the
non finite flag, new parameters, new slots, new model state, the loss function's extra results
(`aux`), the new loss scale. Each step is one call of it; nothing in Python runs between its ops.

* **Model state.** With `state="s"` the loss takes the state tree and returns `(loss, new_s)`; the
  step carries it (batch norm, `examples/nn_cnn.py`).
* **Aux results.** Anything the loss returns after the loss (and the state) comes back in `r.aux`
  (`{"acc": nn.accuracy(...)}`).
* **Random keys.** Step n draws with `rng.call_key(loss_fn.seed, n)`: dropout masks depend on the step
  number only, so a run replays bit for bit and a resumed run equals a straight one (tested).
* **Static arguments** (`static={"training": True}`) pick the program; training and evaluation
  are different programs.
* **Executors.** The float64 reference interpreter by default; `"cpu"` (numba kernels) or `"cuda"`
  (Triton) compile it, and the compiled program is checked against the reference on its first call.
  One program serves every batch size (tested: one `prepare` for batches of 3, 7 and 12).
* **Mixed precision.** `precision="bf16"` or `"f16"` applies the autocast pass (docs/TRAINING.md) to
  the loss program: linear, matmul, conv2d and attention read half precision casts, softmax,
  log_softmax and layer norm run in float32, master weights stay float32. `"f16"` adds dynamic loss
  scaling with GradScaler's rules: a step whose gradients overflow is not applied and the scale
  halves; after `growth_interval` finite steps it doubles (tested step by step against the rule).
* **Non finite steps** without loss scaling raise `NonFiniteStep` (TS763) naming the first op whose
  result is not finite, with the user's line; the state is untouched.
* **Checkpoints** are safetensors files: every carried tensor by tree path (`params['0']['w']`,
  `opt.m...`, `model_state...`, `amp.scale`) plus metadata: the step count, a hash of the step
  program and its configuration. `load` refuses another program (TS764) and lists every missing,
  extra, misshapen or mistyped tensor (TS765).

## Measured (2026-11-24, torch 2.11, Windows, RTX 5070 Ti shared with other jobs)

**Loss curves against PyTorch.** Each example trains the same model written with torch.nn from the
same initial values (Truestar's keyed init copied in) on the same batches with the matching torch
optimizer and scheduler. Worst relative difference of the loss over the run:

| Example | Steps | float64, reference executor vs torch float64 (bar 1e-9) | float32 compiled CPU vs torch float32 CPU | float32 compiled CUDA vs torch float32 CUDA |
|---|---|---|---|---|
| `nn_mlp.py`: 3 layer MLP, 22k params, AdamW, warmup cosine, clip 1.0 | 200 | 6.4e-16 | 1.9e-7 | 6.0e-7 |
| `nn_cnn.py`: 2 conv + 2 batch norm CNN, SGD Nesterov, weight decay | 100 (f64), 40 (f32) | 7.1e-15 (running variance 2.2e-14) | 2.3e-7 | 3.0e-4 (torch's eager CUDA convolutions likely use TF32, which cuDNN allows by default; not isolated) |
| `nn_gpt.py`: 2 block GPT, 61k params, character corpus, AdamW, clip | 200 (f64), 100 (f32) | 3.0e-15 | 2.3e-7 | 2.4e-7 |

The documented tolerance is therefore 1e-9 in float64 (the example tests enforce it) and 1e-5 for
float32 runs of a few hundred steps on the same device type (measured 2e-7 to 6e-7; the CNN on CUDA
excepted as noted).

**Speed, preliminary.** Median time per step from step 4 on, including the host side of the step
(input assembly, the loss read back as a float), on a noisy shared machine:

| Example | compiled CPU (numba) | eager PyTorch CPU | compiled CUDA (Triton) | eager PyTorch CUDA |
|---|---|---|---|---|
| MLP, batch 64 | 2.97 ms | 0.79 ms | 56.2 ms | 21.9 ms |
| CNN, batch 32 | 8.70 ms | 2.85 ms | 100.9 ms | 25.0 ms |
| GPT, batch 16 x 32 tokens | 30.8 ms | 4.67 ms | 52.4 ms | 5.12 ms |

Truestar's compiled step is 3 to 10 times slower than eager PyTorch on these small models today.
For the MLP on the CPU, 2.7 ms of the 3.0 ms is the compiled program itself (54 kernel launches,
about 50 us each through the numba path; the step's own Python is 0.05 ms). First call cost (numba
or Triton compile plus the float64 check of every kernel): 2 to 25 s, 52 s for the GPT on CUDA,
less with a warm kernel cache. The float64 reference executor is the default because it is exact,
not fast. Not measured: torch.compile, larger models, CUDA graph capture of the step (index steps
read their range checks on the host, which a capture refuses).

## Found and fixed on the way

* **Kernel check false rejects on sums whose terms cancel.** A GPT step on CUDA was refused: a bias
  gradient summed over the batch had relative error 2.95e-5 against 2.5e-5 because the result is
  small next to its terms; a conv bias in front of batch norm has an exact zero gradient and was
  refused at 1.5e-3 by the group check. A failing sum, mean, matmul, linear or conv value is now
  measured once more against the same op on its operands' absolute values (the sum of its terms'
  magnitudes, which bounds the rounding of any summation order). `tests/test_kernel_magnitude.py`:
  cancelling float32 sums pass, an error of 1e-3 of the terms' magnitude and a dropped term are
  still refused.
* **The conv tolerance** read the weight shape with `int()`, which failed for the weight gradient's
  conv (the batch is a symbolic channel count there); a symbolic size now counts as 1024, as in the
  kernel tolerance.
* **Kernels** for `erf` (the exact GELU's gradient), `log_softmax`, `amax` and `amin`: the GPT step
  had three fallbacks to the reference on the CPU, now none.

Found, not fixed: `train.emulate_precision` does not round float32 values, because the reference
interpreter's `cast` keeps float32 and wider exact (only 16 bit casts round). Float16 and bfloat16
emulation is unaffected; float32 emulation is a no op today.

## Limits (2026-11-24)

* Speed: slower than eager PyTorch, above. The step is one program, but the compiler's kernel
  count and launch overhead dominate small models.
* No Adafactor or per parameter group settings in `truestar.optim` yet (Lion, RMSprop and Adagrad
  added 2026-11-25: 30 compiled float64 steps within 1e-12 of torch.optim, or of the paper for Lion;
  20 compiled float32 steps within 1e-4 of float64, `tests/test_optim_more.py`); no gradient accumulation or data parallel option on
  `train_step` (both exist on the imported program path).
* Mixed precision cannot rewrite a loss with loops or branches (TS766).
* Checkpoints hold the training state, not the data loader position.
