# Training spikes: float64 shadow and `truestar replay`

Author: Sharven Rane. First written 2026-11-24. Ledger row P35 (docs/QUALITY_BAR.md), research
item R9 in <research>/PAIN_RESEARCH_3.md.

When a run spikes, teams spend days deciding between bad data, the optimizer and numerics. The
standard response (PaLM rewound about 100 steps and skipped 200 to 500 batches, about 20 times) hides
the cause; 2026 runbooks prescribe the same by hand. arXiv 2610.38277 shows "time bomb" spikes
preceded by growing low precision error in the LM head, which its authors saw only by recomputing
FP32 reference logits every logging step.

Two parts:

* **`SpikeWatch`** runs the training step and keeps what a replay needs: a checkpoint every
  `ckpt_every` steps (weights and optimizer state before that step), every batch, every learning
  rate, every loss. On every `shadow_every`-th step, on any step whose loss jumps, or after
  `request_shadow()`, it reruns the same batch from the same weights in float64 and logs the gap
  (TS691 when it is far).
* **`truestar replay RUN --step N`** reruns the window from the checkpoint and gives a verdict:
  numerics, data or optimizer.

```python
from truestar.spikes import SpikeWatch
with SpikeWatch("spec.py", "run_dir", dtype="float16", shadow_every=50, ckpt_every=100) as w:
    for x, y in loader:
        w.step(x, y)            # or w.step(x, y, lr=schedule(step))
```

```
truestar replay run_dir --step 45
truestar replay run_dir --step 45 --format agent
```

The spec file defines `get_model()`, `get_loss()` (a mean loss callable such as `F.mse_loss`) and
`get_optimizer(params)`. examples/spike_demo.py is a spec file and a run in one.

## The step, fixed so it can be replayed

Forward in the model's dtype (`float16`, `bfloat16`, `float32`), the spec's loss on float32
outputs, backward, the gradient norm, optional clipping by the global norm, the optimizer step at
the step's learning rate. The run, the replay and the float64 arm run this one function.

## The shadow check

On a shadowed step the batch runs twice more from the same weights: once in the run's precision
on a copy (so the run itself is untouched) and once in float64 (weights upcast exactly). Logged
per shadowed step in `watch.jsonl`: both losses and their relative error, the cosine and norm
ratio of the two gradients, and, for modules named in `modules=[...]`, each module's own error
(recomputed in float64 on the inputs it saw in low precision, the paper's LM head method made
generic). TS691 when the loss error passes max(0.05, 20 times the median of earlier unflagged
shadow errors), the cosine drops below 0.9, or the low precision step is not finite where float64
is.

## Replay and the verdict

From the checkpoint at or before N (one step earlier when N itself is a checkpoint, so a drift
has room to show), three arms run with the batches and learning rates the run logged:

* **the run's precision**: it must reproduce every logged loss of the window (relative 1e-6),
  else TS695 and no verdict: a verdict on a replay that does not reproduce the run would be a guess;
* **float64 from the same checkpoint** (weights and optimizer state upcast exactly);
* **batch N alone in float64 on the checkpoint weights**, and row by row.

Each replayed step's own rounding error (its loss against the same weights and batch in float64)
is recorded too. The low precision and float64 trajectories themselves part by several percent in
a few steps in any float16 run (14% after 5 steps in the clean run here), so the trajectory gap is
not used as evidence; the per step error is.

A spike is a non finite loss or one above `spike_factor` (default 3) times the median of the 20
logged losses before N.

| replay at step N | verdict | named |
|---|---|---|
| low precision spikes, float64 does not | numerics (TS692) | first module whose own error is above 50 epsilons of the dtype; the step the per step error first passes 5% |
| both spike, batch N is high on the checkpoint weights | data (TS693) | the rows carrying the loss (over 5 times the median row) and their share |
| both spike, batch N is normal on the checkpoint weights | optimizer (TS694) | the updates since the checkpoint; any step whose learning rate jumped |
| no spike | none (TS696) | |

## Constructed spikes (tests/spike_models.py)

A net that normalizes its input first (a hand written RMS norm, `x * rsqrt(mean(x * x))`), two
Linear layers, SGD with momentum, float16 on the CPU, 60 steps, a regression target from a fixed
teacher. Each case changes one thing at step 45:

* **numerics**: batch 45 is a valid batch at 7.5 times the usual input scale. In exact arithmetic
  the net's first op removes the scale, so the batch is as easy as any other; in float16 `x * x`
  overflows past 256, the norm returns zeros and the loss jumps from about 0.6 to 10.8.
* **data**: the targets of rows 3 and 9 of batch 45 are set to plus and minus 40.
* **optimizer**: the learning rate of step 44 is 40 times its value.

Measured with `scripts/spike_study.py` (5 data seeds per case):

| case | verdict | watch flagged step 45 | named | replay time |
|---|---|---|---|---|
| numerics | 5 of 5 numerics | 5 of 5 (TS691, loss error 1.6e1 against at most 1.6e-3 on clean steps) | first split `norm`, own error 1.0 (up, act and down at most 3.5e-4); per step error 1.6e1 at step 45, at most 1.3e-3 on the 5 steps before | 52 ms |
| data | 5 of 5 data | 0 of 5 (correct: float64 spikes as much) | rows 3 and 9, 100% of the loss | 47 ms |
| optimizer | 5 of 5 optimizer | 0 of 5 (correct) | "the learning rate jumped at step 44" | 85 ms |
| clean | 5 of 5 none | 0 of 5 | | 23 ms |

The same overflow batch under bfloat16 (float32's range) does not spike and replays as none.

Baseline: the loss and grad norm curves show all three spikes and cannot tell them apart; the
numerics case has a normal batch, so batch skipping would have dropped a good batch and kept the
overflow.

## False alarms and overhead

* Clean runs, 5 seeds, 60 steps, shadow every 5 steps: 0 of 60 shadowed steps flagged (worst loss
  error 1.6e-3, bar 0.05); replays of every 5th step: 0 of 55 verdicts other than none. A tampered
  log (one loss times 1.5) is refused with TS695.
* Overhead, CPU, float16, median of 3 runs on a shared desktop (timings vary by about 30% here):
  the spec net (0.41 ms per step) costs 1.1 ms per step under the watch with no shadow, which is
  the fixed cost of saving each batch and log line; for a wider net (256, 1024, 256 at batch 256,
  146 ms per step) shadow every 50 steps measured +0%, every 10 steps +6%, every step +101%. One
  shadow costs about one step here because float64 CPU matmuls are faster than float16 ones on this
  CPU; on a GPU, where float64 is slow, a shadow costs more and should be sampled less often.

## Limits

* The 28M parameter time bomb setup of arXiv 2610.38277 (BF16 LM head, peak LR 0.1 on the GPU) was
  not reproduced; the constructed spikes are small and on the CPU. The watch's module errors are
  logged but no rising error before a spike was demonstrated.
* The watch owns the step (forward, loss, backward, clip, optimizer). A custom loop with gradient
  accumulation, autocast, schedulers that read state, or several optimizers is not replayable yet.
* Faithful replay needs a deterministic step. CPU runs here reproduce bitwise; on a GPU,
  nondeterministic kernels can make TS695 refuse a verdict.
* Every batch is saved to its own file; for large batches, keep `ckpt_every` small and the window
  short, or turn batch saving off (`save_batches=False`, which disables replay).
* Data loader state and random number generator state are not saved; models with dropout need
  keyed random ops (docs/RANDOM.md) to replay.
