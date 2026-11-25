# Training on the IR

Author: Sharven Rane. First written 2026-11-24.

Training is built from the same IR programs as inference. Losses, optimizers, gradient clipping,
mixed precision and data parallel gradient averaging are all IR, and one training step is one IR
program: forward, loss, backward, all reduce, clipping, the optimizer update and loss scaling, with
no Python between ops. Code lives in `src/truestar/train/`, tests in `tests/test_train.py`.

## Use

```python
import copy
from torch.export import Dim
from truestar import ir, train as T

prog = ir.import_module(copy.deepcopy(model), (x,), dynamic_shapes=({0: Dim("B", max=64)},))
step = T.build_train_step(prog, T.CrossEntropy(label_smoothing=0.1), T.AdamW(3e-4, weight_decay=0.1),
                          params=model, clip_grad_norm=1.0)
trainer = T.Trainer(step, lr=T.WarmupCosine(3e-4, warmup_steps=100, total_steps=10_000), log_every=50)
for x, y in loader:
    trainer.train_step(x, y)
trainer.save("run.tsx")            # later: trainer.load("run.tsx")
```

`params` names the params to train, or is the module the program came from (its
`named_parameters`). The rest stay constants of the program.

## Pieces

| Module | What it holds |
|---|---|
| `losses.py` | `MSE`, `CrossEntropy(label_smoothing, reduction, ignore_index)`, `CausalLM` (logits `[B, T, V]`, position t predicts token t + 1), `attach_loss(program, loss)` |
| `optim.py` | `SGD(momentum, dampening, weight_decay, nesterov)`, `Adam`, `AdamW` (both with `amsgrad`), `Lion`, `clip_by_global_norm`, `any_nonfinite`, `update_program` (the optimizer alone as a program) |
| `step.py` | `build_train_step` (the one program per step) and `loss_and_grad` (forward, loss and backward alone) |
| `trainer.py` | `Trainer`, `WarmupCosine`, `Constant`, `first_nonfinite`, checkpoints |
| `amp.py` | `AutocastPolicy`, `autocast` (the policy as a program transformation), `LossScaler`, `emulate_precision`, `EmulatedReferenceExecutor` |
| `distributed.py` | the gradient bucket and its one `all_reduce` |
| `ops.py` | the three ops the IR lacked (`cast`, `log_softmax`, `all_reduce`) and their gradient rules |
| `edit.py` | `Editor` (append to an existing program, shape checked), `params_to_inputs`, `retype` |

### Losses

Every loss is IR. Cross entropy builds its one hot mask by comparing the targets with an `arange`,
so no gather op is needed, and uses `log_softmax`, one op computing `x - max - log(sum(exp(x - max)))`.
The forms follow PyTorch exactly, including label smoothing and `ignore_index` together: with
reduction "mean" the loss is `(1 - eps) * sum(nll) / n + eps / C * sum(smooth) / n` over the `n`
counted rows. The count is a sum over the batch inside the program, so one program serves every
batch size.

### Optimizers

An optimizer maps `(params, grads, state, lr)` to `(new params, new state)` with IR ops only. State
is explicit and carried as program inputs and outputs: `step`, `<param>.momentum_buffer`,
`<param>.exp_avg`, `<param>.exp_avg_sq`, `<param>.max_exp_avg_sq`. The learning rate is a 0 dim
input so a schedule never changes the program; every other hyperparameter is a constant inside it.
The update follows the torch.optim single tensor code line by line:

* SGD: weight decay added to the gradient; momentum buffer equal to the gradient on the first step
  (selected with `where` on the step count, so dampening behaves as in torch), then
  `momentum * buf + (1 - dampening) * g`; Nesterov uses `g + momentum * buf`.
* Adam and AdamW: `m` by lerp, `v = b2 v + (1 - b2) g^2`, bias corrections `1 - b^t`, update
  `(lr / c1) * m / (sqrt(v) / sqrt(c2) + eps)`. AdamW decays the param by `1 - lr * wd` first; Adam
  adds `wd * p` to the gradient. `b^t` is `exp(t ln b)` because the IR has no power op.
* Lion (Chen et al. 2023): `p *= 1 - lr * wd`, `p -= lr * sign(b1 m + (1 - b1) g)`,
  `m = b2 m + (1 - b2) g`. The sign is two `where` nodes.
* Clipping: `clip_grad_norm_` with norm type 2, `min(max_norm / (norm + 1e-6), 1)`.

### The step program

Layout: inputs are the model inputs, the loss targets, `lr`, every trainable param, every state
value. Outputs are `loss`, `grad_norm`, `nonfinite`, the new params and the new state in the same
order. Order of the parts inside: forward, loss, backward (the autodiff pass), then the data
parallel average, unscaling, the non finite check, clipping, the update, and with loss scaling the
skip and the scale update. The trainer prepares the program once and makes exactly one executor
call per step (a test counts them: one `prepare` and three calls for three batch sizes).

### Trainer

* **Executor seam.** Anything with `prepare(program)` (the `truestar.frontend.Executor` seam) or a
  plain `fn(program, inputs)`. Default: the float64 reference interpreter. Nothing in the package
  reaches into an executor, so the compiled executor plugs in unchanged.
* **Schedule.** `WarmupCosine(base_lr, warmup_steps, total_steps, min_lr, warmup_start_factor)`
  equals torch's `SequentialLR` of `LinearLR` and `CosineAnnealingLR` (tested to 1e-12 over 30 steps).
* **Logging.** `history` holds step, lr, loss, grad norm, skipped and scale for every step;
  `log_every=n` prints a line every n steps.
* **NaN and Inf.** The step program reports a NaN or Inf in the loss or any gradient. Without loss
  scaling the trainer then refuses the step (params and state stay as they were), reruns the same
  inputs on the reference interpreter keeping every value, and raises `NonFiniteError` naming the
  first node in program order whose result is not finite, so its operands are finite. Imported
  nodes carry the ATen op and the user's source line:

  ```
  training step 1: the loss is not finite (loss nan, grad norm nan); the step was not applied.
  First non finite value: %sqrt = sqrt(%linear) : f64[B, 4] is the first non finite value (1 NaN, 0 Inf of 12); its operands are finite
    at: aten.sqrt.default (node %sqrt), test_train.py:492 in forward: return torch.sqrt(h)
  ```

  A NaN already present in the batch is reported as that input instead.
* **Checkpoints.** A checkpoint is a `.tsx` file (docs/EXPORT.md format version 1): a program with
  no nodes whose params are the carried tensors in their declared dtypes, with the step count, the
  history, the configuration and the hash of the step program in the metadata. The format writes
  the same bytes for the same state. `load` refuses a checkpoint written by another step program
  (different model, loss, optimizer settings or policy) and says so. Resuming restores the exact
  tensors, so in float64 the continuation is bit identical to a run that never stopped (tested:
  3 plus 3 steps against 6 straight steps of the transformer with AdamW, clipping and a schedule;
  every tensor `torch.equal`, the history equal and the final checkpoint bytes equal).
  Checkpoint version 2 (2026-11-24) adds the RNG seed and the data loader position:
  `Trainer(..., seed=s)` fills the step's random key input with `rng.call_key(s, step)`, and
  `trainer.save(path, loader=loader)` / `trainer.load(path, loader=loader)` carry the loader's
  position, so a resume with shuffling and dropout is bit identical too (docs/DATA.md).

## Mixed precision

Two program transformations.

**The policy (`autocast`).** Ops in `low_ops` read float16 or bfloat16 casts of their float
operands; ops in `fp32_ops` read float32 casts; every other op runs in the dtype its operands
promote to. The default matches torch's CPU autocast as measured on torch 2.11: `linear`, `matmul`,
`conv2d` and `attention` run in half precision; `softmax`, `log_softmax`, `layer_norm`, `gelu`,
`tanh` and `relu` follow their input; the loss runs in float32 (model outputs are cast before it,
as autocast does for `mse_loss` and `cross_entropy`). Master weights stay float32 program inputs;
the cast of each weight is emitted once, like autocast's weight cache. The gradient rule of `cast`
casts the cotangent back to the operand's dtype, so master weight gradients arrive in float32.

**Dynamic loss scaling (`LossScaler`)** has `torch.amp.GradScaler`'s defaults and semantics, built
in IR: the loss is multiplied by `amp.scale` before the backward pass, gradients are multiplied by
`1 / scale` after it, a NaN or Inf in any gradient skips the step (every param and state value is
selected from its old value with `where`), the scale is halved and the growth counter reset; after
`growth_interval` finite steps the scale doubles. Default: on for float16, off for bfloat16.

**Running it on the reference interpreter (`emulate_precision`).** The interpreter computes every
float in float64 whatever the declared dtype. This pass inserts a `cast` to the declared dtype
after each op whose result is float32, float16 or bfloat16, so the run rounds and overflows where a
backend computing in those dtypes would. One refinement makes it match eager PyTorch: the nodes the
autodiff pass emits for one gradient rule (for tanh, `g * (1 - y * y)`) stand for one fused backward
kernel, which computes in float32 internally and rounds once, so only values that leave such a
group are rounded. Without that refinement about half of the float16 gradient entries differed
from torch by an ulp or more; with it, 0.4%. `EmulatedReferenceExecutor` applies the pass. A
compiled backend computes in the declared dtypes and needs neither.

### Mixed precision, compiled (2026-11-25, TS1480 to TS1482)

```python
step = T.build_train_step(prog, T.CrossEntropy(), T.SGD(1e-3, momentum=0.9), params=model,
                          amp=T.AutocastPolicy.cuda(torch.bfloat16))   # float16: a LossScaler by default
cs = CompiledTrainStep(step, "cuda", graph=True)   # compiles in bfloat16 with mixed=True
```

`compile_program(..., dtype="bfloat16" or "float16", mixed=True)` keeps every value in its
declared dtype: the policy's casts, contractions (Triton dots with float32 accumulation),
convolutions, attention and activations run in the 16 bit working precision; master weights,
gradients after their cast back, optimizer state, the loss, the loss scale and the growth tracker
are float32 islands. Without `mixed` the program's dtype is the one holding the most elements and
an autocast step either runs every value in 16 bits (master weights included) or takes a fallback
per cast. `CompiledTrainStep` turns `mixed` on for a step built with `amp`.
`AutocastPolicy.cuda(dtype)` is torch.autocast("cuda")'s lists for the IR's ops (softmax,
log_softmax, the norms including batch norm, exp, log, pow, rsqrt and sum in float32).

**Verification for the declared precision.** Kernel and region checks are unchanged (each kernel's
tolerance follows its own dtypes). A value declared in 16 bits may overflow where that dtype
overflows: the region check compares such a kernel with its reference sent through the same range.
End to end, the float64 reference is the program with a range check after every 16 bit value
(`emulate_precision(range_only=True)`: exact arithmetic that overflows where float16 overflows, so a
loss scaler skips the steps torch and the compiled step skip). Each float32 output gets its own
bar: `MIXED_FACTOR` (4) times the larger of (a) how far an independent float32 evaluation rounded to
the declared dtypes (`emulate_precision`, reference interpreter) is from float64, and (b) how far
the same evaluation moves when every 16 bit value rounds toward zero instead (one ulp of every
16 bit value: an output fed by few 16 bit values, a bias gradient, can land an ulp away in any
correct run); at least 4x the float32 kernel bar. An output still over its bar is accepted only if
the float64 reference taking this run's branch decisions matches every output within its bar and
every flipped decision is a near tie (a relu input within rounding of 0; TS1481,
`mixed_bounds[i]["near_tie"]`). An output whose bar exceeds 0.1 is ill conditioned in the declared
precision (a cancelling bias gradient behind batch norm) and listed (TS1482). Anything else is
TS1480. Before, the end to end bar of a 16 bit program was the working precision's for every
output: 0.1 for bfloat16, 0.06 for float16.

Caught (tests/test_train_amp_compiled.py, CPU kernels and CUDA): a master weight that comes back
rounded to bfloat16 or float16 after every region check passed (under the old 0.1 bar it passed),
and a missing unscale (gradients 2x with a scale of 2). Not caught by design: optimizer state
rounded to 16 bits on the first step (its error is the gradient's own 16 bit error).

Found and fixed on the way: the loss scaler's growth test compared with a float32 max constant
that a 16 bit working precision saturates to 65504, so a compiled float16 scaler never grew (now
`grown - grown == 0`); a keyed `random` draw declared float32 was made in the 16 bit working
precision; the pow gradient's computed exponent took a fallback (constant exponents now emit a
constant); a pure 16 bit cast kernel tripped the lean test (TS960) on structured inputs although
every element was exactly round to nearest even (now exempt when that holds bit for bit).

**bfloat16 weights with keyed stochastic rounding** (`weights="bfloat16_sr"`, `sr_seed`): params
are stored in bfloat16 with no float32 copy, gradients arrive in bfloat16, the optimizer state
stays float32 and the update is computed in float32; the new weight is
`amp.stochastic_round(v, key, site)`: rounded up with probability (|v| - lo) / spacing, the draw
from the keyed generator (docs/RANDOM.md) at a site per param under `sr.key =
rng.call_key(sr_seed, step)`. It is built from exact IEEE operations only (round to nearest casts,
products with at most 21 significant bits, Sterbenz differences, compares), so it is bitwise equal
between the reference interpreter and the compiled kernels and between runs, and unbiased. The
kernel that finds the next bfloat16 number rounds up on purpose (`cast(..., directed=True)`) and is
exempt from the lean test. Limits: no `param_group`, `data_parallel` or clipping yet; the random
step reads its key on the host (a sync per step, so no CUDA graph with it yet).

**Measured (2026-11-25, Modal L4, torch 2.11.0+cu130, `scripts/bench_train_amp.py`, SGD momentum
0.9, lr 1e-3, TF32 off, 4 rounds of 5 steps, median ms).** MLP 1024-4096-4096-1024 at batch 512,
cross entropy:

| dtype | Truestar | Truestar graph | eager AMP | torch.compile AMP | Truestar graph / best |
|---|---|---|---|---|---|
| bfloat16 | 6.23 | 5.72 | 4.80 | 4.71 | 0.82x (slower) |
| float16 (loss scaling) | 8.09 | 7.29 | 6.07 | 6.01 | 0.82x (slower) |

Loss curve parity over 200 steps against eager torch.autocast (+ GradScaler for float16) from the
same weights on a fixed cycle of 8 batches: largest relative loss difference 7.4e-6 (bfloat16) and
8.3e-7 (float16), final params 1.7e-4 and 7.7e-6 relative L2 apart; the loss drops by the same
amount on both sides (bfloat16 0.06326 against 0.06324, float16 0.06322 on both). Verified first call: 66 and 77 kernels, no fallback, no
near tie, every float32 output at most 0.07 and 0.11 of its bar; 5.8 s of checks. Bitwise run to
run: tests/test_train_amp_compiled.py, 20 verified steps, every carried value bit for bit (CPU
kernels and the L4, both dtypes, a float16 skipped step included).

Not measured: the encoder layer, the Llama style decoder and resnet18 at the fair benchmark sizes
did not finish on Modal within the time limits (one build plus its verified first call of the
encoder layer took over 12 minutes at batch 32 x 256 and over 9 at batch 8; not profiled, so where
the time goes (Triton compile and tuning, or the end to end check, which runs four float64 and
float32 reference passes of the whole step on the host CPU) is open), and H100 rows need the owner's yes
(the Modal runner allows T4, L4, A10G and L40S only). On CPU kernels at small sizes all four models
compile with no fallback and pass verification in bfloat16 and float16.

Not shipped as a speed feature: Truestar's mixed precision step is 18% slower than PyTorch AMP on
the one model measured. What ships is correctness: compiled mixed precision training that runs at
all (before, an autocast step either rounded the master weights to 16 bits or took a host
fallback per cast), checked against float64 with bars for the declared precision.

## What is verified (2026-11-24, torch 2.11, CPU)

`tests/test_train.py`, 46 tests, about 70 s.

**Optimizers against torch.optim, float64, 20 steps, loss and every parameter after every step.**
Worst relative error (to the largest entry of each tensor) over all steps:

| Model (loss) | SGD momentum + dampening + decay | SGD Nesterov + decay | AdamW | Lion |
|---|---|---|---|---|
| MLP, tanh and gelu (MSE) | 6.6e-16 | 1.0e-15 | 2.3e-16 | 1.6e-16 |
| CNN, conv, max pool, gelu (cross entropy, label smoothing 0.1) | 9.7e-16 | 6.9e-16 | 4.5e-16 | 4.2e-16 |
| TransformerEncoderLayer plus a vocabulary head (causal LM loss) | 8.7e-16 | 9.3e-16 | 5.4e-15 | 3.6e-16 |

Also Adam with L2 decay, amsgrad and clipping at 0.5 on the CNN (4.2e-16), plain SGD with clipping
at 0.1 on the MLP (3.0e-16), and the standalone AdamW update program against `torch.optim.AdamW`
for 5 steps (below 1e-14). Every model is imported with `torch.export` and trained with one
program across batch sizes. The bar is 1e-10. Lion is not in torch.optim; the reference is the
paper's algorithm written in plain torch in the test.

One justified exclusion: the key part of the attention input projection bias (`in_proj_bias[16:32]`)
has a gradient that is exactly zero in exact arithmetic, because adding `q . b_k` to every score of
a row does not change a softmax. Both sides compute rounding noise there (1e-18 against 1e-2 for
the query bias, asserted in `test_key_bias_gradient_is_zero`), and Lion, which uses only the sign,
moves those 16 entries by plus or minus lr depending on the noise. They are left out of the
transformer comparison for every optimizer; with them, Lion differs by 2.0 and AdamW by 2.6e-10 (the
bias starts at zero, so its scale is tiny).

**Losses against torch.nn.functional, float64.** Cross entropy with label smoothing 0, 0.1 and 0.5,
reductions mean and sum, an ignored row, two batch sizes from one program; MSE; the causal LM loss
with label smoothing and a padding token; values and input gradients all below 1e-14. Logits of
1e4 give finite results equal to torch (the stable form). The gradient rule of `log_softmax` passes
the fourth order finite difference check of docs/AUTODIFF.md at the same 1e-8 bar.

**Mixed precision against `torch.autocast("cpu")` with `torch.amp.GradScaler`.** A float32 MLP
(linear, relu, linear, tanh, linear), batch 32, MSE, run with `EmulatedReferenceExecutor`.

| Check | float16 | bfloat16 |
|---|---|---|
| loss, same weights and batch, 10 points along a training run | 2.4e-7 | 9.8e-8 |
| (plain float32 against autocast, for scale) | 9.2e-6 | 3.8e-5 |
| scaled master weight gradient entries bit identical to torch | 99.6% | 100% |
| worst gradient error, relative to the tensor's largest entry | 4.2e-4 | 2.2e-4 |
| one step from torch's exact state, same gradients: new params (SGD, AdamW) | 1.0e-7, 1.2e-7 | 1.1e-7, 1.2e-7 |
| skip decision, scale and growth counter, 20 steps from torch's state | identical | (no scaler) |
| free run of 20 steps: skip pattern and scale sequence | identical | (no scaler) |
| free run of 20 steps with AdamW: worst param error | 6.4e-5 | 1.7e-5 |

Plain float32 gradients share no entry with autocast's, so the bit identity check separates a right
policy from a wrong one, which a tolerance at float16 precision cannot. The gradient entries that
differ do so where torch's half precision matmul accumulates in float32 in its own order and the
reference accumulates exactly: a sum near a rounding boundary rounds the other way (1 of 2048
entries of a 256 long dot product in a direct measurement) and that ulp spreads through the
backward pass; the bar is one low precision epsilon of the largest entry. The loss scaler test uses
`growth_interval=3` so the scale grows into overflow again and again (5 skips in 20 steps with AdamW, 6 with SGD), and
checks that the step is skipped exactly when torch's own gradients overflow, that skipped steps
leave every param untouched, and that the scale and counter equal torch's after every step. SGD
at lr 0.05 and momentum 0.9 amplifies the rare one ulp differences over a free run (0.3% float16,
25% bfloat16 after 20 steps), so its free run checks the skip pattern and scales only.

**Data parallel, 2 gloo ranks on CPU, float64.** Every rank runs the same step program on half of
each batch; the program averages loss and gradients with one `all_reduce` of a 436 entry bucket.
Against one rank on the whole batch (AdamW, clipping at 0.5, 6 steps): losses, grad norms, params
and Adam state agree to 6.2e-16, and the two ranks stay bit identical to each other. Run inside
the collective recorder, a rank that runs out of data after 2 steps is diagnosed:

```
verdict: rank 1 finished after 2 collectives. The other ranks wait in seq 2 all_reduce (436,) float64 at train_workers.py:25.
```

The recorded site is the user's `trainer.train_step(...)` line: `collectives.skip_frames_from` (a
small hook added to `diag/collectives.py`) lets the recorder skip truestar's own frames the way it
skips torch's.

**Structure.** The step program serializes as an ordinary artifact; the policy inserts exactly 7
casts into the MLP (input, three weights, three biases) and none on the activations; the global
gradient rule table is unchanged after the training package is imported.

## Multi GPU (real ranks, ledger row P39)

Added 2026-11-24. One process per GPU, joined by torch.distributed: NCCL between GPUs on Linux, gloo
on Windows and across machines. Code: `src/truestar/parallel/devices.py` (`join`, `DeviceComm`),
`gradsync.py`, `sharded.py`, `native.py`; the layout check on real ranks is docs/PARALLEL.md, Real
ranks. Measurements: `scripts/multigpu_verify.py`; the Kaggle run is the private kernel
`sharvenrane/truestar-multigpu-t4x2` built by `scripts/kaggle_multigpu.py`.

### Native training step

```python
from truestar.parallel.devices import DeviceComm, join, make_groups
info = join(rank, world, "tcp://192.0.2.10:29511", device="cuda")      # this process's GPU
comm = DeviceComm(rank, world, info.device, info.backend, {tuple(range(world)): torch.distributed.new_group(list(range(world)))})
step = truestar.train_step(loss, optim.AdamW(lr=1e-3), clip=1.0, executor="cuda",
                           data_parallel=comm, accumulate=2, check_replicas=100)
ts, r = step(ts, x_of_this_rank, y_of_this_rank)
```

With `data_parallel` or `accumulate` the step is two programs: forward and backward, run once per
micro batch, and the update (clipping, the optimizer), run once. Between them the micro batch
losses and gradients are summed in order, summed over replicas in fixed buckets (each started as
soon as it is flattened), and divided once by the number of micro batches over all replicas. With
equal micro batches of a loss that is a mean over rows this is the whole batch mean, and every
replica runs the same update on the same inputs. `check_replicas=n` all gathers two float64
numbers per parameter every n steps and raises TS824 naming the first tensor where a replica
differs. Refused with TS823, because they would not compute the single device step: model state
(batch norm statistics per replica), random ops (each replica would draw its own masks), float16
loss scaling (the skip is not synchronized yet), and a batch that does not split into equal micro
batches. The backward pass is one executor call, so its compute does not overlap the transfers on
this path.

### torch.nn modules

`GradientSync(model.named_parameters(), comm, group, bucket_bytes=25 << 20, overlap=True)` sums
gradients over replicas in fixed buckets started from `register_post_accumulate_grad_hook` while the
backward pass runs, in bucket order on every rank; `with sync.no_sync():` for accumulation micro
batches; `sync.finish()` before clipping. `FlatShard` keeps one slice of the parameters and of the
optimizer state per replica (reduce scatter, the optimizer on the slice, all gather). Both are
documented in docs/PARALLEL.md, Real ranks; none of this code comes from another framework.

### Verified on Kaggle GPU T4 x2 (2026-11-24, NCCL 2.28.9, torch 2.11.0+cu130)

Model: the pre norm transformer of examples/parallel_model.py at width 512, 4 blocks, 13.1M
parameters, float32, global batch 32 by 128 tokens, AdamW with weight decay, clipping at 1.0,
`torch.use_deterministic_algorithms(True)`, 4 steps. "Same tree" is one T4 computing each replica's
gradient (with its micro batches) and adding them in rank order, the order a 2 rank fixed sum uses.

| check | result |
|---|---|
| 2 T4 step against one T4, same tree: every parameter after every step, ga 1 and 2, fixed and native order, overlap on and off (8 runs) | **bitwise equal** in all 8 |
| 2 T4 step against one T4 on the whole batch | first step gradients 3.6e-7 (ga 1), 2.9e-7 (ga 2) relative to each tensor's largest entry; parameters after 4 steps 4.8e-6 and 3.7e-6; loss 1.0e-7 |
| the same 2 T4 run twice | bitwise equal |
| rank 0 against rank 1 after the run | bitwise equal |
| sharded optimizer state against replicated data parallel, 4 steps, without and with clipping | bitwise equal both; AdamW state per rank 52.5 MB against 105.0 MB |
| tensor parallel MLP block (4096 tokens, width 2048, hidden 8192, column then row, gelu) on 2 T4 against one T4 | output 4.4e-6, input gradient 2.2e-6, weight gradients 1.6e-6 and 9.9e-7 |
| native step (`train_step(executor="cuda", data_parallel=comm)`, 3 layer MLP width 1024, batch 512, AdamW, clip, 6 steps) against one T4 with `accumulate=2` | **bitwise equal**; replica check passed every step |
| native step against one T4 on the whole batch | parameters 1.2e-4 after 6 steps, loss 1.0e-7 |

When bitwise equality holds and when it does not: the 2 rank sum equals one device only if one
device adds the same per replica pieces in the same order (the same tree); the whole batch on one
device sums inside one matmul in cuBLAS's order, so it differs by float32 rounding (above). Native
order equals fixed order for 2 ranks (one addition per element) but not for 3 or more (TS825).
Different GPU models compute the per replica pieces with different kernels: the desktop RTX 5070 Ti
plus laptop RTX 5060 run was not bitwise equal to the same tree on the desktop (parameters 1.3e-5
after 3 steps), while its two ranks stayed bitwise equal to each other and the run was bitwise
equal to itself. Whole batch errors after several AdamW steps are larger than after one step
because AdamW divides by the root of the second moment, which magnifies differences in tiny
gradients.

### Speed (provisional: two runs on a shared free Kaggle machine, one run over the wireless network)

Median time per training step from step 3 on, both ranks synchronized, float32:

| setup | time per step | against one GPU |
|---|---|---|
| T4 x2: one T4, whole batch (13.1M transformer, 4096 tokens) | 115.7 ms (110.4 ms in the first run) | 1.00 |
| T4 x2: 2 ranks, native order, buckets of 1 MB (25 buckets), overlap on | 65.7 ms | **1.76x** faster |
| the same, overlap off | 69.2 ms | 1.67x |
| fixed order, 1 MB buckets, overlap on / off | 67.8 / 71.9 ms | 1.71x / 1.61x |
| 25 MB buckets (2 buckets), native, overlap on / off | 67.2 / 68.7 ms | 1.72x / 1.68x |
| T4 x2: tensor parallel MLP block above, one T4 / 2 T4 native / 2 T4 fixed | 206.3 / 104.7 / 110.8 ms | 1.97x / 1.86x |
| T4 x2: native step MLP above, one T4 whole batch / 2 T4 | 6.8 / 11.5 ms | 0.59x (slower) |
| desktop + laptop over the wireless network, gloo: 1.6M transformer, one RTX 5070 Ti / 2 ranks (best of 8 settings) | 12.0 ms / 905 ms | 75x slower |

Overlapping the bucket sums with the backward pass saved 3 to 6 ms of a 70 ms step; fixed order
cost about 2 ms over native for 2 ranks. The native step is slower than one GPU on this small MLP:
two executor calls, the host side sums and an NCCL call per bucket cost more than half a 6.8 ms
step saves; it was not measured on a larger model. Over the home wireless network the gloo transfers (about
6.6 MB of gradients per step, staged through host memory) dominate completely; that setup is a
correctness run, not a way to train faster. NVLink and InfiniBand were not measured; 4 GPUs and
torch DDP are in the next section.

### Scaling against torch DDP, FSDP2 and tensor parallel (4 x A10G, 2026-11-25)

`scripts/dp_scaling.py` on Modal, 4 x NVIDIA A10G in one machine (PCIe, NCCL 2.28.9, torch
2.11.0+cu130), 16 CPU cores for the container, float32 with TF32 off and deterministic algorithms
on, AdamW (foreach) and clipping in every arm. Strong scaling: the global batch stays the same, W
GPUs get 1 / W of it each. Median of 8 timed steps after 3 warm ones, slowest rank. Results:
`results/multigpu_scaling/full1/` (stdout.txt, dp_scaling.json).

Data parallel, the example transformer (`examples/parallel_model.py`). "small": width 512, 4
blocks, 13.1M parameters, 4096 tokens a step. "big": width 1024, 8 blocks, 103.3M parameters, 8192
tokens a step. Buckets of 25 MB; "4/1" is 4 MB buckets with a first bucket of 1 MB.

| arm | small, 1 / 2 / 4 GPUs (ms) | 4 GPU speedup | big, 1 / 2 / 4 GPUs (ms) | 4 GPU speedup |
|---|---|---|---|---|
| one GPU, whole batch | 45.3 | 1.00 | 545.0 | 1.00 |
| Truestar native order | 28.3 / 20.2 | 2.25x | 288.2 / 167.2 | 3.26x |
| Truestar fixed order (all to all, fold, all gather) | 29.4 / 21.2 | 2.14x | 295.0 / 171.3 | 3.18x |
| Truestar fixed order, previous gather fold | 30.4 / 27.7 | 1.64x | 300.6 / 223.2 | 2.44x |
| torch DDP (defaults, gradient_as_bucket_view) | 28.4 / 20.3 | 2.23x | 289.1 / 165.1 | 3.30x |
| torch FSDP2 (fully_shard per block and root) | 29.2 / 27.8 | 1.63x | 292.4 / 164.6 | 3.31x |
| Truestar native, 4/1 MB buckets | 27.1 / 19.2 | 2.36x | 290.1 / 167.0 | 3.26x |
| Truestar fixed, 4/1 MB buckets | 28.3 / 21.2 | 2.14x | 293.7 / 179.0 | 3.05x |

Truestar's native order is at DDP's speed (within 1.3% on the big model, faster with small buckets on
the small one). The fixed order sum costs 2.5% over native on the big model at 4 GPUs (4.1 ms) and
5% on the small one (1 ms); the gather fold it replaces cost 33% and 37%. In every fixed order run the
4 replicas ended bitwise equal and a second run was bitwise equal to the first, and
`multigpu_verify.py --phases dp` at 4 GPUs found the fixed order step bitwise equal to one GPU adding
the same per replica gradients in rank order (ga 1 and 2, overlap on and off).

Where the time goes (big model, 4 GPUs): the step without any gradient sum takes 159.5 ms (40.6 ms of
it host time to launch); the bucket sums alone take 47.4 ms native (13.1 GB/s ring bus bandwidth over
PCIe), 52.7 ms fixed and 111.8 ms with the gather fold. Overlap with backward hides 84% of the native
communication and 77% of the fixed. On the small model the step is host bound (14 ms of launches
for 16.7 ms of compute at 4 GPUs), so no arm passes 2.4x.

The 2026-11-25 tp4 run, where 4 GPUs were slower than one (dp4 native 59 ms against 41.5 ms), ran
with Modal's default CPU share: 4 Python ranks launching kernels and NCCL's proxy threads shared a
fraction of a core. The same code with that default share (`results/multigpu_scaling/smoke_a10gx4.log`,
2 timed steps) gave 41 ms native and 35.5 ms DDP at 4 GPUs; with 16 cores, 20.2 and 20.3 ms. A
multi GPU container needs at least a core per rank.

Tensor parallel, column then row MLP with gelu, forward and backward (Truestar `ToColumn` and
`FromRow`; torch `ColwiseParallel` and `RowwiseParallel` on the same shards):

| shape | one GPU | 2 GPUs: native / fixed / torch TP | 4 GPUs: native / fixed / gather fold / torch TP |
|---|---|---|---|
| 4096 tokens, 2048, 8192 | 62.1 ms | 36.0 / 36.2 / 36.9 ms | 22.0 / 22.1 / 32.0 / 22.1 ms |
| 8192 tokens, 4096, 16384 | 478.8 ms | 259.1 / 260.2 / 260.6 ms | 146.3 / 145.9 / 184.0 / 147.5 ms |

The fixed order tensor parallel sum is now at native speed and at torch tensor parallel's; with the
gather fold it took 1.45x and 1.26x native at 4 GPUs.

## Tests that prove the tests work

Permanent tests break a rule and assert the comparison fails: Adam without bias correction
(`b^t` replaced by 0) and SGD that ignores the Nesterov flag both differ from torch by more than
1e-3 within 3 steps.

Manual breaks on 2026-11-24, each restored afterwards:

| Break | Tests that failed |
|---|---|
| Adam step size `lr` instead of `lr / (1 - b1^t)` | 8: AdamW on all three models, Adam amsgrad with clipping, the standalone update program, the schedule test, both AdamW mixed precision tests |
| Lion without the decoupled weight decay | 3: Lion on all three models |
| the loss scaler never skips a step | 3: both float16 mixed precision training tests and the forced overflow test |

## Compiled (2026-11-24, ledger row P37)

`truestar.train.compiled.CompiledTrainStep` runs a step program through `truestar.compiler` with
the params and optimizer state kept on the device:

```python
from truestar.train.compiled import CompiledTrainStep

step = T.build_train_step(prog, T.CrossEntropy(), T.AdamW(3e-4, weight_decay=0.01), params=model,
                          param_group=True)
cs = CompiledTrainStep(step, "cuda", "float32", graph=True,
                       accept_ill_conditioned=True)   # AdamW's first step, see below
for x, y in loader:
    loss, grad_norm, nonfinite = cs(x, y, lr=3e-4)      # device tensors, no host sync
weights = cs.params()                                    # per param views, by model name
```

What makes it fast, and what each part costs in verification (nothing is skipped: the first call
of the program checks every kernel and region against float64 and the whole step end to end):

* **No fallback in a training step.** `log_softmax`, `erf` and `expm1` are kernels
  (docs/COMPILER.md, Training programs). Before, cross entropy and the exact gelu gradient each ran
  as a float64 host round trip.
* **Donation every step.** The carried params and state are handed to the program, which writes
  the new values over the old buffers. A program output prefers the donated input's buffer over an
  intermediate's (docs/MEMORY.md), so every carried value keeps its address.
* **CUDA graph replay (`graph=True`).** After `warmup` plain calls (the first is the checked call),
  the step is captured once and replayed: one launch per step instead of one per kernel. The first
  replay is compared bit for bit with a plain call on a copy of the same state (TS780 on any
  difference). Values the step cannot write in place (the 0 dim step count, a moment updated inside
  a reduction kernel) are copied back inside the graph (`cs.copied_back`). A batch of another shape
  raises TS781; build another step for it.
* **One parameter group (`param_group=True`).** Every param is a view of one flat buffer
  `params.flat`, the gradients are concatenated once, and the optimizer runs on flat buffers: the
  update is one kernel per carried value (params, each moment) for the whole model, plus one norm
  and one non finite reduction, instead of five kernels per param (74 params in the six layer
  transformer: 370 optimizer kernels). The float64 result equals the per param step to 1e-12
  (tests/test_train_compiled.py). State is named `params.flat.exp_avg` and so on;
  `TrainStep.unflatten` and `cs.params()` give the per param views.
* **Accurate bias correction.** `1 - b^t` is `-expm1(t ln b)` (it cancelled in float32 and failed
  the kernel check, so a compiled AdamW step did not run at all before).
* **AdamW's first step is accepted by its derived bound (2026-11-25).** The end to end check now
  judges a training step's outputs by the bound derived from the step itself (docs/COMPILER.md,
  Verification; docs/TOLERANCE.md), which follows the cancellation into the division by
  `sqrt(v) + eps`: correct AdamW and SGD steps pass with default verification and no option, and an
  optimizer bug (wrong decay sign, a missing bias correction, a gradient over n - 1, a wrong cast) is
  refused (tests/test_trainverify.py). The rest of this item describes the older opt in, which still
  works.
* **AdamW's first step and `accept_ill_conditioned`.** On the first step AdamW divides each
  gradient entry by its own magnitude, so an entry that cancels in its float32 sum moves its weight
  by a random plus or minus lr. Every correct float32 execution is then far from float64: eager
  float32 PyTorch is 6.8e-4 and 2.6e-3 off on two weights of the benchmark MLP, over the end to end
  bar of 6.25e-4. By default the compiled step is refused, and the error names the output, its
  error, the bar and how far an independent float32 evaluation of the same program is from float64
  (docs/COMPILER.md, Verification). Pass `accept_ill_conditioned=True` to `CompiledTrainStep` (or
  `compile_program`) when you train with Adam, AdamW or Lion in float32 and that error shows the
  float32 evaluation is itself over the bar: outputs within 4x of it are accepted, listed in
  `cs.cp.conditioning` and reported by a TS782 warning. Do not use it to silence an error where the
  float32 evaluation is within the bar: that is a real mismatch. `scripts/bench_train.py` passes it
  for its AdamW rows; SGD rows run strict.
* **Store less for the backward.** The relu gradient reads the relu output (docs/AUTODIFF.md).
* **Conv weight gradient in row chunks** with a fixed order sum (docs/COMPILER.md).

### Measured (2026-11-24, provisional: RTX 5070 Ti shared with other jobs, Windows, torch 2.11, Triton 3.7.1)

`scripts/bench_train.py`, one process per row, sides interleaved inside a round, every step timed
with its own CUDA events, 3 warmup steps, 20 to 30 timed steps (4 to 6 rounds of 5), median (min) in
ms. IEEE float32 on every side (`matmul.allow_tf32` and `cudnn.allow_tf32` off), `cudnn.benchmark`
on for eager, eval form models (batch norm with running statistics, dropout 0), one training step =
forward, loss, backward, AdamW (weight decay 0.01). Eager uses `torch.optim.AdamW(fused=True)`;
torch.compile compiles the forward plus loss and the optimizer step (the documented way; backward
through AOT autograd). "graph" is `CompiledTrainStep(graph=True)`; compare it with max-autotune,
which also replays CUDA graphs. Peak is `max_memory_allocated` during one step above params, state
and batch; graph replays (Truestar graph, max-autotune) allocate from a pool reserved at capture,
so no peak is given for them (the Truestar graph replays the plain Truestar schedule). SM clock
2.6 to 2.95 GHz in every row but the MLP at batch 256 (1.85 GHz in that run).

MLP 1024, 2048, 2048, 1000 (relu, gelu), cross entropy; six layer transformer LM (d 256, 4 heads,
feed forward 1024, sequence 128, vocabulary 512), causal LM loss; resnet18 at 224 px, cross entropy.

| model | batch | eager | torch.compile | max-autotune | Truestar | Truestar graph | peak eager / compile / Truestar (MiB) |
|---|---|---|---|---|---|---|---|
| MLP | 64 | 1.10 (0.71) | 1.18 (0.89) | 0.92 (0.58) | 2.01 (1.53) | **0.64 (0.60)** | 49.9 / 32.4 / 32.3 |
| MLP | 256 | 0.97 (0.95) | 0.99 (0.96) | **0.94 (0.93)** | 1.61 (1.44) | 1.15 (1.12) | 52.1 / 33.9 / 33.8 |
| LM | 8 | 8.19 (7.04) | 5.52 (4.24) | **2.84 (2.70)** | 26.36 (22.09) | 5.26 (5.20) | 113.8 / 91.0 / 104.0 |
| LM | 16 | | | | refused (below) | refused | |
| resnet18 | 8 | **6.94 (6.87)** | 9.85 (9.58) | 9.06 (8.91) | 20.84 (18.54) | 13.84 (13.77) | 240.5 / 216.7 / 190.1 |
| resnet18 | 16 | | | | refused (below) | refused | |

With `param_group=True` (separate processes, eager rerun as the anchor):

| model | batch | eager | Truestar group | Truestar group, graph | peak eager / Truestar group |
|---|---|---|---|---|---|
| MLP | 64 | 0.98 (0.73) | 1.73 (1.46) | 1.03 (1.00) | 32.8 / 72.1 |
| LM | 8 | 6.66 (5.90) | 15.21 (10.73) | 4.92 (4.87) | 91.7 / 104.1 |
| resnet18 | 8 | 6.92 (6.89) | 14.58 (14.38) | 13.59 (13.46) | 224.3 / 191.7 |
| resnet18 | 16 | 9.88 (9.82) | 23.21 (22.96) | 22.20 (22.06) | 401.5 / 336.8 |

(the resnet18 batch 16 group row ran while another benchmark process used the GPU.)

Before (main plus the native conv kernels, `a1423a9`) and after, SGD with momentum 0.9, Truestar
plain and graph, eager rerun in each process:

| model | batch | eager | Truestar before | Truestar after | Truestar graph after | peak before / after (MiB) |
|---|---|---|---|---|---|---|
| MLP | 64 | 0.98 / 0.71 | 3.63 | 1.56 | 0.55 | 32.5 / 32.5 |
| MLP | 256 | 0.93 / 1.19 | 4.06 | 2.13 | 1.07 | 34.0 / 34.0 |
| LM | 8 | 9.49 / 6.85 | 29.78 | 20.90 | 5.15 | 128.0 / 104.0 |
| resnet18 | 8 | 6.92 / 7.52 | 31.62 | 22.44 | 14.43 | 252.4 / 190.6 |
| resnet18 | 16 | 10.56 / 10.48 | 59.34 | 26.09 | 23.30 | 469.1 / 338.2 |

Before, a compiled AdamW step did not run at all (its bias correction failed the kernel check).

Wins:

* **Faster than eager** with the graph on the MLP at batch 64 (1.7x) and the LM at batch 8 (1.6x,
  group 1.4x against its own eager run), and ahead of torch.compile default on both (1.8x, 1.05x).
  Ahead of max-autotune on the MLP at batch 64 (1.4x).
* **Against the tree before:** 2.3x (MLP 64), 1.9x (MLP 256), 1.4x (LM 8), 1.4x (resnet18 8) and
  2.3x (resnet18 16) without a graph; 6.6x, 3.8x, 5.8x, 2.2x and 2.5x with one.
* **Memory at or below eager** on the MLP (32 against 50 MiB with AdamW) and resnet18 (190 against
  240 MiB at batch 8, 338 against 403 at batch 16), and below the tree before: LM 128 to 104 MiB,
  resnet18 252 to 191 and 469 to 338 MiB.

Losses, stated plainly:

* **max-autotune is 1.7x faster on the LM** (2.84 against 4.92 ms) and level on the MLP at batch 256;
  the LM's remaining time is GPU time in float32 contractions and layout copies (permutes of the
  attention backward), not launches.
* **resnet18 is 2x slower than eager and 1.4x slower than torch.compile.** The conv backward is
  still the cost: the split weight gradient is 1.7x to 5x behind cuDNN per layer, and the input
  gradient was not retiled in this work.
* **The LM's peak is above eager's** in one run (104 against 91.7 MiB; eager measured 113.8 in
  another process). The attention probabilities are still stored for the backward pass.
* **Without a graph Truestar stays host bound** on the LM (1170 steps per call, 48 ms of host issue
  in the profile); `param_group` halves it (26 to 15 ms) but adds a copy of every gradient (the
  concatenation), which costs the MLP 0.4 ms and 40 MiB: there it is a loss, so it stays opt in.
* **Refused at batch 16 (the LM with SGD and AdamW, before and after this work; resnet18 with AdamW and no group, not diagnosed separately):** the end to end check against
  float64 rejects the step: `layers.3.linear1.weight` momentum (the gradient) 1.3% off float64
  against a 0.21% bar, every one of 614 region checks passing. Eager float32 PyTorch is itself 1.5%
  off float64 on `layers.4.linear1.weight` at the same batch: relu masks of values within rounding
  of 0 decide differently. This is the near tie limit (docs/COMPILER.md), found and not fixed: the
  float32 conditioning allowance does not cover it, because a tie flips in a different layer in
  each float32 run. Nothing is returned in that case.
* **Compile time:** the LM step takes 60 s to compile and resnet18 281 s when its kernels are not
  in the disk cache (torch.compile 15.5 s and 28 s, max-autotune 89 s and 171 s); with the cache, 4
  s. The first call (Triton compile, tuning, float64 checks) adds 7.5 s and 74 s.

## Transformer training step, profiled (2026-11-25)

`scripts/bench_train_llama.py`: a Llama style decoder written with torch.nn (token embedding, 6
pre norm blocks of RMSNorm, causal multi head attention with rotary embeddings through
`F.scaled_dot_product_attention`, RMSNorm and a SwiGLU feed forward; d 1024, 16 heads of 64, feed
forward 2816, vocabulary 8192, untied head: 93.9M params), batch 8 x 512 tokens, causal LM loss,
AdamW (weight decay 0.01), IEEE float32 on every side (TF32 off). Modal L4 (58 SMs, 24 GB),
torch 2.11.0+cu130 with its bundled Triton. One process; sides interleaved, 20 timed steps each
(4 rounds of 5), median ms. Eager is `torch.optim.AdamW(fused=True)` with SDPA (the memory
efficient kernel in float32); torch.compile compiles the forward plus loss and the optimizer step.
`--breakdown` gives GPU time by category: Truestar from `profile_program` (each step's IR ops, the
phase taken from the step program's node order), torch from `torch.profiler` kernel names.

| side | step ms (median) | tokens/s |
|---|---|---|
| Truestar before (attention lowered to its parts), CUDA graph | 334.6 | 12241 |
| **Truestar after (`attention_grad`), CUDA graph** | **305.4** | **13412** |
| Truestar after, plain calls | 306.1 | 13380 |
| eager PyTorch | 277.8 | 14744 |
| torch.compile | 263.0 | 15572 |

Step time against torch.compile: 1.27x before, **1.16x after**; against eager 1.20x before, 1.10x
after. The CUDA graph no longer matters at this size: the GPU is busy for the whole step.

GPU time per step by category (ms; "before" is the profile of the tree before this work, a
separate run on the same GPU type):

| category | Truestar before | Truestar after | torch.compile | eager |
|---|---|---|---|---|
| matmuls, forward and backward (Truestar contraction kernels, cuBLAS) | 235.3 | 177 to 185 (the score matmuls are gone) | 197.0 | 190.2 |
| attention forward | 10.4 (scores, softmax, product) | 7.3 to 7.5 (one online softmax kernel) | 5.3 | 4.7 |
| attention backward | spread over matmul, softmax backward and elementwise kernels | 33 to 46 (`attention_grad`, 6 calls) | 15.1 | 15.0 |
| elementwise and layout (RoPE, SwiGLU, residuals, copies) | 51.9 | about 35 (without `attention_grad`) | 14.5 | 39.6 |
| norms (RMSNorm forward and backward) | in elementwise and reductions | same | 10.7 | 2.0 |
| reductions of the backward (softmax backward, sums) | 13.1 | 3.3 to 3.4 | | |
| optimizer (AdamW, grad norm, non finite check) | 16.2 (376 launches) | 15.9 to 16.0 | 11.4 | 22.6 (11.3 kernels) |
| embedding gradient (`scatter_add`, fixed order) | 3.0 | 2.8 to 2.9 | 0.04 | 0.2 |
| GPU busy per step | 330.2 | 288.7 | 255.7 (kernel sum) | 277.3 (kernel sum) |

The profiler did not attribute torch's matmul kernels to forward or backward, so only the matmul
total is compared. Host: without a graph Truestar issues one call (about 1000 steps) in 349 ms
before and 293 ms after, equal to its GPU time, which is why the graph buys nothing here.

### Attention backward: `attention_grad`

The gradient pass lowered `attention` to a score matmul, the causal `where`, `softmax` and a
matmul, so the forward stored the `[B, H, T, T]` probabilities (33.5M floats,
134 MB per layer here) and the backward was two batched score sized matmuls, a softmax backward
and several full passes over score sized values. Now plain self attention (q, k, v and the output
of one shape, no explicit mask, causal or not, head dim at most 128) is kept whole: its forward is
the online softmax kernel and stores nothing score sized, and its gradient is one op,
`attention_grad(q, k, v, o, do)` giving dq, dk and dv stacked (`src/truestar/ir/attention_grad.py`,
`src/truestar/compiler/attention_bwd.py`). On CUDA it is three Truestar Triton kernels: row
statistics (each query row's log sum of exponentials and `rowsum(do * o)`), then dk and dv per key
block looping over the query blocks in increasing order, then dq per query block looping over the
key blocks. Every output element is written by one program and every sum runs in an order fixed
by the block table, which depends on the head dim and dtype only (no atomics, no autotuning, no
split), so the result has the same bits on every call. On the CPU the same passes run in float32
torch blocks. The reference is the definition in float64 (it does not read `o`, so a wrong forward
cannot vouch for its own gradient), and the op is region checked against it on its first run like
every library step. Masked attention, cross attention and grouped query attention keep the
lowering; `TRUESTAR_FUSED_ATTENTION_GRAD=0` restores it everywhere.

Transforms that look inside the op see its definition: `attention_grad` has a lowering to the
reference's formulas in IR ops, so forward mode (a Hessian of a model with attention is forward
over reverse) and a second reverse pass differentiate through it, and vmap batches it with one
rule (operands expanded along the batch dim, the stacked result permuted). The softmax and
gate invariant checks (`truestar.invariants`, docs/TRAIN_NUMERICS.md) instrument the lowered
attention backward; the fused op has no delta of its own to go stale and is region checked
against its float64 definition instead.

One layer (`scripts/bench_attention_grad.py`, L4, `[8, 16, 512, 64]`, causal, float32): 5.4 ms
(row statistics 0.85, dk and dv 2.1, dq 1.7) against 2.0 ms for torch's memory efficient SDPA
backward; relative error against float64 4.5e-7 to 6.3e-7 (torch 6.7e-7 to 9.7e-7). Inside the
step it costs more (33 to 46 ms for 6 layers against 32 from the micro benchmark), most likely
because the micro benchmark's 80 MB of operands sit partly in the L4's 48 MB L2 (not isolated).
The first block table spilled registers (64 x 64 float32 tiles for dk and dv): 48 ms per layer.
The table now uses 32 x 32 tiles and one stage for dk, dv and dq, sized for 64 KiB of shared
memory so that no GPU needs another table.

Tests (`tests/test_attention_grad.py`): the reference equals torch.autograd in float64 (1e-12);
the kept node gives the same float64 gradients as the lowering (1e-12); CPU blocks and CUDA kernels
are within 2e-6 of float64 for odd lengths (77, 130, 200) and head dims 40, 64, 80 and 128; the
CUDA result is bitwise repeatable; strided `[B, T, H, D]` views are read in place; bfloat16
operands are within 2e-2; a compiled decoder step with the op passes every float64 check on the
CPU and CUDA and matches torch float64 within 1e-5 over 3 SGD steps.

### CUDA graph replay of a step with an embedding

`CompiledTrainStep(graph=True)` could not capture any model with an embedding: the token id range
check read its flag on the host after the call, which a capture refuses
(`cudaErrorStreamCaptureUnsupported`). During a capture the flags are now copied to pinned host
memory inside the graph and read after each replay (one host sync per step, only when the program
has index checks), so an out of range id in a replayed step still raises with the op and the bad
index. Tested: replays equal plain calls bit for bit, and an id equal to the vocabulary size raises.

### Backward matmuls against cuBLAS

`scripts/bench_bwd_gemm.py`, the decoder's linear layers (N = 4096 tokens), each GEMM as a one op
program, float32, ms Truestar / cuBLAS:

| layer (M x K) | forward x W^T | dX = dY W | dW = dY^T x |
|---|---|---|---|
| attention projection 1024 x 1024 | 0.97 / 0.69 (1.41x) | 0.95 / 0.72 (1.31x) | 0.83 / 0.54 (1.52x) |
| gate, up 2816 x 1024 | 2.80 / 2.49 (1.12x) | 2.34 / 1.96 (1.20x) | 2.27 / 1.92 (1.18x) |
| down 1024 x 2816 | 2.95 / 2.63 (1.12x) | 2.62 / 2.39 (1.10x) | 2.26 / 1.94 (1.16x) |
| head 8192 x 1024 | 8.68 / 7.55 (1.15x) | 7.14 / 5.85 (1.22x) | 6.79 / 5.57 (1.22x) |

The transposed backward layouts are not worse than the forward one: Truestar reaches 9.0 to 10.4
TFLOP/s on dX and dW against 7.9 to 8.8 on the forward; cuBLAS 9.9 to 15.8. The gap is in the tile
kernels (largest on the 1024 x 1024 projection: 4096 x 1024 outputs are few tiles for 58 SMs),
which belong to the matmul tile work, so no backward specific layout handling was added.

### Tried, measured, not shipped or not done

* `param_group=True` on this model: 209.5 against 199.7 ms without it (2 layer run): the gradient
  concatenation and the group's update kernels cost more than the 140 small launches saved.
* A fused AdamW update (one kernel reading p, g, m, v and writing p, m, v): not built. The update is
  three kernels per param (m, v, p), 16 ms against 11.4 for torch's fused multi tensor kernel. One
  kernel needs either multi output kernels in the compiler's fusion or a library step whose outputs
  are written over the donated buffers.
* Fused norm and softmax backward: after `attention_grad` the softmax backward is gone from the step
  and the RMSNorm backward is a few small reductions (3.4 ms of backward reductions in all). The
  measured remaining cost is the elementwise and layout kernels (RoPE and SwiGLU products and their
  backward, `cat` and `reshape` copies), about 35 ms against 14.5 for torch.compile, decided by the
  compiler's fusion (one output per kernel), not by a norm rule.
* H100: not run. `tools/modal_gpu.py` allows T4, L4, A10G and L40S and asks for the owner's yes
  before an A100 or H100.
* bfloat16: the kernels take bfloat16 operands (tested), but the bfloat16 training step belongs to
  the mixed precision work; only float32 steps were measured.

## Limits

* `attention_grad` has no gradient rule of its own: a second derivative through a kept attention
  (a Hessian vector product) is refused by name; set `TRUESTAR_FUSED_ATTENTION_GRAD=0` to
  differentiate the lowering twice.
* The reference interpreter path (`Trainer`) is unchanged; `CompiledTrainStep` is the compiled
  path and is not yet wired into `Trainer` (checkpoints, the NaN report and schedules are
  `Trainer` features).
* The three new ops (`cast`, `log_softmax`, `all_reduce`) are registered from `train/ops.py`
  through the IR registries, and their gradient rules are installed only during `truestar.train`
  gradient calls. `truestar.ir.grad` alone fails loudly on them. Folding them into `shapes.py`,
  `reference.py` and `grad_rules.py`, and giving them compiled kernels, is merge work; a compiler
  coverage test that enumerates `shapes.RULES` will see three more ops.
* The autocast policy is checked against torch on linear, relu and tanh. Its lists for conv,
  attention and the norms follow the measured CPU autocast behaviour but have no amp training test.
  CUDA's lists differ (softmax and the norms run in float32 there) and are a policy argument.
* `emulate_precision` treats every imported forward node as its own kernel and every gradient rule
  as one kernel. Fused forward kernels that PyTorch splits differently would round differently.
* Training mode batch norm, dropout and running statistics are not expressible (the importer
  takes inference graphs), so models are trained in their eval form. Dropout set to 0 is fine.
* Lion has no torch.optim reference; it is checked against the paper's algorithm in plain torch.
* The IR level `data_parallel=True` of `build_train_step` (above) still averages one flat bucket
  inside the program; the bucketed, overlapped and real GPU paths are the native step and the
  torch.nn pieces of Multi GPU. Every data parallel path needs the batch to split evenly across
  ranks and micro batches for the mean of means to equal the global mean. The native step checks
  it every step (TS1210, one small all gather of the row counts before any compute, so every
  replica raises the same error); the IR level step and the torch.nn pieces do not.
* Multi GPU: up to 4 GPUs in one machine (4 x A10G over PCIe) and 2 machines with one GPU each were run;
  no NVLink, no multi node NCCL; the native data parallel step refuses model state, random ops and
  float16 loss scaling (TS823); sharded state gathers parameters for the whole step, not per layer.
* Checkpoints hold carried tensors and trainer counters, not data loader position or random state.

## Pain ledger mapping

| Row | How this serves it | Still open |
|---|---|---|
| P5 graph breaks and recompiles | A whole training step is one program; batch size changes never rebuild it (tested at batch 4, 1 and 7) | Control flow inside a step |
| P7 the trained artifact is the deployed artifact | The step program and the checkpoint are `.tsx` artifacts like the inference program | A compiled backend |
| P10 hangs with no attribution | The gradient all reduce runs under the collective recorder; a rank that runs out of data is named with the user's line | NCCL hangs, multi node |
| P39 real multi GPU training | Data parallel, tensor parallel and sharded state on 2 real T4s (NCCL) and across 2 machines (gloo), bitwise equal to one GPU on the same reduction tree, 1.76x on 2 T4s | More than 2 GPUs, NVLink, per layer parameter gathering |
| Never introduce #1, no silent fallback | Unsupported settings, ops without gradient rules and a mixed precision request on non float32 weights raise by name | |
| Never introduce #3, no silent wrong results of our own | Every optimizer, loss and the amp policy is compared with PyTorch, and breaking a rule fails the tests | Compiled kernels |
| Never introduce #6, no cryptic errors | A NaN names the first op, its source line and the step, and the step is not applied | |
