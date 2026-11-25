# Memory planner and recomputation

Author: Sharven Rane. First written 2026-11-24.

The compiler knows every value's size from its shape type and every value's readers from the
program, and IR values never alias or change. So the memory a compiled call needs can be planned
exactly before anything runs, freed as soon as nothing reads it, written over when it is dead, and
recomputed instead of stored when a budget asks for it. This page is that work.

Pain served: N8 in docs/PAIN_RESEARCH_2.md ("OOM from allocator fragmentation with plenty of free
memory, and no reliable peak memory prediction before a run"), pain ledger row P23 in
docs/QUALITY_BAR.md. Research items: GAP_PYTORCH_JAX R5 (recompute as a verified pass) and R6 (static
planner with inferred donation), GAP_COMPETITORS item 7 (peak known before the first launch).

Code: `src/truestar/compiler/memory.py` (schedule, plan, budget), `src/truestar/compiler/recompute.py`
(the recompute pass), the step loop in `src/truestar/compiler/runtime.py`, `CompiledKernel.call_into` in
`src/truestar/kernel/compiler.py`. Tests: `tests/test_memory.py`. Benchmark:
`scripts/bench_memory.py`.

## Use

```python
from truestar.compiler import compile_program

cp = compile_program(program, "cuda", "float32")
plan = cp.memory_plan({"B": 64})        # no data, nothing launched; or cp.memory_plan([x])
print(plan)                             # peak, the step and source line at the peak, values live there
plan.total_bytes, plan.peak_bytes, plan.hold_peak_bytes, plan.to_dict()

# refuse a program that would not fit, before the first launch (TS420)
cp = compile_program(program, "cuda", "float32", memory_budget="6 GiB")              # plans at the declared max of B
cp = compile_program(program, "cuda", "float32", memory_budget="6 GiB", memory_dims={"B": 64})

# a training program (forward, loss, backward): recompute instead of store until it fits (TS421 verified)
cp = compile_program(train_program, "cuda", "float32", memory_budget="2 GiB", memory_dims={"B": 64}, recompute=True)
cp.recompute_report                         # what is recomputed, its bytes, its source line

# hand the inputs over: a dead input's buffer may take a result (params and optimizer state)
outs = cp(inputs, donate=True)          # takes the tensors out of the list `inputs`
```

`CompiledExecutor(...)` takes the same `memory_budget`, `memory_dims` and `recompute` arguments. A
program that follows PyTorch's TF32 or deterministic setting builds one variant per setting
(docs/COMPILER.md); each variant gets its own schedule, plan, budget check and recompute pass when it is
built, and `cp.memory_plan` plans the variant that is current.
`memory_reuse=False` gives the executor as it was before this work (every value held until the call
returns), kept for measurement.

Example (the transformer layer with a budget it does not fit):

```
[TS420] compiled program 'TransformerEncoderLayer' on cuda (float32) needs 7.13 MiB at B=256 (params and
inputs 1.13 MiB, peak during the call 6.00 MiB), over the memory budget of 2.86 MiB.
  peak at layout %view_2: reshape
    at activation.py:1492 in forward: attn_output, attn_output_weights = F.multi_head_attention_forward( (module self_attn: MultiheadAttention)
  largest values live at the peak:
       3.00 MiB  %select_2
       ...
  fits the budget: B up to 100
  try: a smaller batch, recompute=True for a training program (docs/MEMORY.md), or a larger budget.
Nothing was launched.
```

## How it works

**Liveness (the schedule).** For each compiled step the runtime knows the slots it reads (including
the slots its first call check compares against) and writes. One pass gives every slot its live
ranges; after each step the runtime drops the slots whose range ends there. Program outputs are
never dropped. The schedule does not depend on the symbol values, so it is built once per program.

**Reuse.** Dropping a slot gives its storage back to PyTorch's caching allocator, which hands it to
the next allocation of a fitting size. This alone is where most of the saving comes from.

**In place.** A kernel may write its result into the buffer of one of its inputs when all of this
holds, checked at compile time: the kernel has one output and no reduction; the input has the
output's shape and dtype; every load of that input is at the output's own index (so each element is
read only by the program instance that writes it, before it writes it); the input is not a param, a
program output or a program input (unless donated); and no slot that may share its storage (views
from layout ops, region passthrough) is read later, and no slot in that group is a param or a program output. At run time a guard checks, before every such
launch, that the runtime's slot holds the only reference to the tensor and to its storage, that the
tensor starts at offset 0, is contiguous and covers its storage. The reference counts the guard
expects are measured at import, not assumed. If the guard says no, the step allocates a new buffer.
Whether a view covers its storage depends on the shapes, so the plan applies the same conditions
with its meta tensors: the plan and the runtime agree on every in place launch (a test counts them).

**Donation, inferred safely.** `cp(inputs, donate=True)` takes the tensors out of the caller's list.
A donated input that dies is freed, and a kernel elementwise in it may write over it: a training
step's new params and momentum take the old buffers. A tensor the caller kept a reference to (in
another list, a variable, a view) fails the reference count guard and is never written over; a test
holds every input elsewhere and checks they are unchanged.

**Donated input first for program outputs (2026-11-24).** A step whose output is a program output
and that may write over both a dead intermediate and a donated input now takes the donated input
when the call donates. Before, a training step's new momentum took the gradient's buffer (an
intermediate wins), so the momentum moved every step and the old momentum buffer was freed: same
peak, but a carried value with a new address each call, which a CUDA graph replay cannot follow.
Now the carried value stays at its address and the gradient's buffer is freed instead (same peak;
the plan mirrors the rule, `Schedule.donated_first`). Calls without donation are unchanged.

**The plan, before any launch.** `memory_plan` dry runs the steps on meta tensors: kernel outputs and
host results get their sizes from the IR types at the given symbol values; layout ops and library
ops run on meta tensors, which tells exactly which outputs are views of which inputs and the shapes
of convolution and pooling results; control flow regions are planned recursively (the larger branch
of a `cond`; a `loop` with the previous iteration's carried values alive). Then a walk through the
schedule counts live bytes, rounded to 512 bytes per block the way the CUDA caching allocator rounds.
Planning the four benchmark programs takes 0.5 ms to 5 ms per call of `memory_plan`.

**Budget.** `memory_budget` compares params, inputs and the peak of the call with the budget at
`memory_dims`, or at the declared maximum of every symbol, at compile time. A symbol without a
maximum is planned at each new input shape, still before that shape's first launch. The error names
the step, ops and source line at the peak, the values live there, and the largest value of each
symbol that fits (a binary search over the planner).

**Recomputation.** The pass works on the compiled steps, not on the IR. A value chosen for
recompute is dropped after its last reader before the peak and produced again, by the same compiled
step, right before its next reader. Choosing is greedy over the values live at the planned peak:
each candidate is a value plus, when that helps, the producers it would otherwise keep alive (up to
five levels, through views, which is what a checkpointed segment is). Candidates are ranked by bytes
freed at the peak per unit of recompute work (output elements times reduction length), the best are
planned exactly, the best exact gain per work is taken, and this repeats until the plan fits.

Why the recomputed value has the same bits, by construction:

* It is the same step object: the same kernel binary and the same launch config (chosen per input
  shape signature, which does not change between the two runs). Truestar kernels have no atomics.
* Its inputs are either kept (no step mutates a value; in place reuse only writes over buffers no
  later step reads, and the recompute counts as a reader) or recomputed the same way.
* Only deterministic steps are recomputed: Truestar kernels, layout copies and views, dtype adapters.
  cuDNN convolutions, index steps, host and fallback steps are never recomputed (cuDNN may choose
  another algorithm when free memory differs).

Recomputing at the IR level would not give this: a copy of the IR nodes could be fused into a
different kernel with a different summation order.

And verified: on every check run (the first call, or every call with `verify="all"`) the runtime
copies a value aside before the step that drops it and compares the recomputed value with it bit for
bit (NaN payloads and signed zeros included). A difference raises `RecomputeMismatchError` (TS421)
naming the value, ops and source line, and nothing is returned. This is the failure class of
pytorch#201158 (compiled storage reuse corrupting selective checkpointing's cached tensors, 0.74
relative gradient error) turned into a loud error: a test corrupts one input of a recompute and gets
TS421.

## Measured (2026-11-24, RTX 5070 Ti, torch 2.11, CUDA 12.8, float32, shared GPU)

`PYTHONPATH=src python scripts/bench_memory.py`. Peaks are bytes allocated during one call above
what was allocated before it, after two warm calls; "requested" is torch's `requested_bytes`
statistic, "allocated" is `torch.cuda.max_memory_allocated`.

### Planned against measured peak, and reuse against the executor before it and eager

| program | batch | planned (MiB) | measured allocated | measured requested | before the planner, planned / measured | PyTorch eager, no_grad |
|---|---|---|---|---|---|---|
| MLP 1024, 2048 x2, 1000 | 1 | 0.02 | 0.02 | equal | 0.02 / 0.02 | 0.02 |
| | 64 | 1.00 | 1.00 | equal | 1.49 / 1.49 | 1.00 |
| | 256 | 4.00 | 4.19 | equal | 5.95 / 6.14 | 4.19 |
| TransformerEncoderLayer d256, seq 128 | 1 | 0.75 | 0.75 | equal | 1.62 / 1.62 | 0.75 |
| | 8 | 6.00 | 6.19 | equal | 16.00 / 16.00 | 6.19 |
| | 64 | 48.00 | 48.00 | equal | 128.00 / 128.00 | 48.00 |
| causal attention d256, 8 heads, seq 512 | 1 | 2.00 | 2.00 | equal | 3.00 / 3.00 | 2.50 |
| | 8 | 16.00 | 16.00 | equal | 24.00 / 24.75 | 20.00 |
| | 32 | 64.00 | 64.00 | equal | 96.00 / 96.00 | 80.00 |
| resnet18, 224 px | 1 | 3.83 | 46.38 | 45.38 | 19.05 / 64.95 | 15.42 |
| | 16 | 61.25 | 86.75 | 85.75 | 304.81 / 352.25 | 108.22 |
| | 64 | 245.00 | 344.00 | 343.00 | 1219.24 / 1288.50 | 392.00 |

* **Exact where Truestar owns the allocations.** MLP, transformer layer and attention: the planned
  requested bytes equal the measured requested bytes at every batch size, with and without reuse
  (also asserted in tests/test_memory.py at batch 1, 7 and 33, plus the training programs below).
  `max_memory_allocated` exceeds the plan by at most 0.19 MiB (4.8% at MLP batch 256): the caching
  allocator hands out a cached block without splitting it when less than 1 MiB would be left over.
  That is allocator waste, not a planning error.
* **resnet18 is not exact: cuDNN's convolution workspace is not planned.** It is allocated and freed
  inside each library call, its size depends on the algorithm cuDNN picks, and PyTorch's
  `CUDNN_CONV_WSCAP_DBG` cap had no effect on this build. The gap is 42 MiB at batch 1 and 99 MiB at
  batch 64. The plan says so in its notes ("20 convolutions: cuDNN's workspace ... is not planned"),
  and a budget check on a convolutional program is a lower bound until convolution has a Truestar
  kernel (docs/COMPILER.md, Limits).
* **Reuse against the executor before it:** 33% less (MLP), 63% (transformer layer), 33%
  (attention), 80% planned and 73% measured (resnet18 at batch 64, with 17 in place batch norm,
  add and relu kernels).
* **Against PyTorch eager:** level on the MLP and the transformer layer; 20% less on attention; resnet18 at batch 64 12% less measured (344 MiB
  against 392), both including cuDNN workspace.
* **Same results:** outputs with reuse are `torch.equal` to the executor without it, and in place
  kernels run exactly as often as planned (a spy counts them).

### Donation: a training step (forward, MSE, backward, SGD with momentum)

MLP 1024, 4096, 1024 at batch 64; params and momentum are 64.04 MiB.

| | planned peak of the call | measured | in place steps | result |
|---|---|---|---|---|
| inputs kept by the caller | 64.04 MiB | 64.04 MiB | 5 | |
| `donate=True` | 32.75 MiB | 32.75 MiB | 9 | bitwise equal |

Without donation the step holds a second copy of every param and momentum buffer; with it the
update writes over the old buffers, and what remains is the gradients and activations. Optimizer
step memory is params plus state, with no second copy.

### Recompute on a training program, against torch.utils.checkpoint

Six TransformerEncoderLayers (d256, 4 heads, feed forward 1024) and a 512 word head, causal LM loss,
batch 16, sequence 128: forward, loss and backward in one program (`truestar.train.loss_and_grad`).
Params and inputs: 20.6 MiB. Times are the fastest of 9 interleaved rounds of 5 calls on a GPU
other jobs were also using; an earlier run on a busier GPU gave the same memory to 0.01 MiB and
times up to twice as long, so read the times as rough.

| variant | planned peak (MiB) | measured (MiB) | recomputed values | step (ms) | gradients |
|---|---|---|---|---|---|
| Truestar, every value held (before the planner) | 1055.91 | 1056.07 | | 18.5 | |
| Truestar, liveness and reuse | 255.98 | 256.04 | | 19.2 | reference |
| Truestar recompute, budget 75% of that peak | 185.59 | 185.59 | 21 | 19.3 (+1%) | bitwise equal |
| Truestar recompute, budget 50% | 127.53 | 127.53 | 69 | 21.4 (+12%) | bitwise equal |
| Truestar recompute, budget 35% | 89.51 | 89.51 | 108 | 22.0 (+15%) | bitwise equal |
| PyTorch eager | | 180.87 | | 8.3 | |
| PyTorch eager, `torch.utils.checkpoint` per layer | | 63.62 | | 15.0 (+81%) | |

* **Every budget met, planned equal to measured, gradients bit identical to the run without recompute**
  (and every recomputed value compared bitwise on the first call). The research target was measured
  peak within 10% of the budget: all three are within 0.5% under it.
* **Memory saved against time added:** the 35% budget saves 65% of the peak (256 to 89.5 MiB) for
  15% more step time; the 50% budget saves 50% for 12%. `torch.utils.checkpoint` per layer saves
  65% of eager's peak (181 to 63.6 MiB) for 81% more step time. The pass recomputes only what the
  budget needs, where checkpointing reruns every layer's whole forward.
* **Losses, stated plainly.** Truestar's compiled training step is 2.3x slower than eager PyTorch in
  absolute terms here (19.2 ms against 8.3 ms; the float32 contraction kernels, docs/COMPILER.md). Its
  peak without recompute is above eager's (256 against 181 MiB) because the autodiff pass stores more
  than PyTorch does: both the input and the output of each feed forward relu, the attention
  probabilities and the layer norm's centred input and reciprocal deviation as full tensors. Recompute
  takes it under eager at the 50% budget, but checkpointing reaches 63.6 MiB and the pass stops at
  89.5 MiB: below that, the peak sits on values no deterministic step can recompute cheaply (the
  loss part and the last layer's backward). Recompute compile time is 5.4 s to 8.8 s.

## Min cut recompute and host offload (ledger row P55)

Added 2026-11-24. Code: `src/truestar/compiler/remat.py`, the four offload steps in
`src/truestar/compiler/runtime.py`; tests `tests/test_remat.py`; benchmark `scripts/bench_remat.py`,
results `results/remat/remat.json`. Research item 4.5 (sources: PyTorch selective activation
checkpointing and memory budget API, 2025-03-05; torch_remat in TorchTitan, devlog 2026-11-19; Poolside
C2C activation offload, 2026-03-18).

```python
cp = compile_program(train_program, "cuda", "float32", memory_budget=budget, memory_dims={"B": 16},
                     recompute="mincut")            # or "offload", "mincut+offload"; True stays the greedy pass
cp.recompute_report      # each value: TS1110 (recomputed, with its work) or TS1111 (offloaded)
cp.remat_summary         # cuts tried, planned peak, total recompute work
```

**Min cut.** b is the step at the planned peak. Every value written before b is two nodes joined by an
edge of its stored bytes (a view counts its base's bytes); the source reaches each value through an
edge of lambda times the recompute work of its step (infinite for library, index, host and random
steps, which are never rerun); a reader edge says a recompute needs its inputs, a sink edge says a step
after b reads the value. A cut is a store or recompute choice for every value backward needs, so only
what backward reads is replayed. The cut is solved (Dinic) for 50 values of lambda; each distinct cut
is planned exactly by the planner's walk, the one that fits with the least work is kept, and then each
recomputed step, most work first, is stored again if the plan still fits (the sweep jumps between cuts,
this removes what the budget does not need). Recomputed values are inserted right before their first
need, by the same compiled step, so the bitwise argument and the TS421 check above are unchanged.

**Offload.** For a value stored across b: an offload step copies it to pinned host memory on a side
stream right after its last reader before b, a fence step one step later makes the main stream wait for
that copy and the schedule drops the device buffer, a reload step starts the copy back 4 steps before
the first reader after b, and a wait step right before that reader makes the main stream wait. The
positions are fixed at compile time, so the planner counts the buffer until the fence and from the
reload, and the overlap cannot change any bit (a copy). Values are taken largest first until the plan
fits. Check runs compare every reloaded value with a copy of the stored one (TS1112, tested with a
corrupted reload). CUDA only (TS1113).

### Measured (2026-11-24, RTX 5070 Ti, torch 2.11, float32, shared GPU)

Training programs: the six layer transformer LM above (d256, batch 16, sequence 128) and resnet18 at
64 px (batch 32, 10 classes, batch norm in eval mode, Truestar convolution kernels). Budgets are a share
of the peak with reuse. "work" is the recompute work the pass plans (output elements times reduction
length, summed over recomputed steps): exact, unlike the times. Times are the fastest of 9 rounds of 3
calls on a GPU other jobs were using; two variants with the same plan (min cut, and min cut plus
offload where offload added nothing) measured 25.6 and 28.9 ms, and 28.7 and 34.3 ms, so differences
under about 20% are noise.

| program | budget | variant | planned peak (MiB) | measured | recomputed / offloaded | work | step (ms) |
|---|---|---|---|---|---|---|---|
| LM | none | reuse only | 255.98 | 256.01 | | | 25.0 |
| | 75% (192.0) | greedy | 185.59 | 185.59 | 21 | 1.89e7 | 28.7 |
| | | min cut | 191.98 | 192.01 | 20 | 1.68e7 (-11%) | 29.0 |
| | | offload | 191.98 | 192.01 | 8 offloaded | 0 | 27.8 |
| | 50% (128.0) | greedy | 127.54 | 127.54 | 68 | 2.59e9 | 29.1 |
| | | min cut | 127.98 | 128.10 | 52 | 1.97e9 (-24%) | 28.7 to 34.3 |
| | | offload | 127.59 | 127.65 | 25 offloaded | 0 | 42.5 |
| | 35% (89.6) | greedy | 89.51 | 89.51 | 108 | 8.78e9 | 36.7 |
| | | min cut | 89.59 | 89.59 | 163 | 4.94e9 (-44%) | 29.9 to 30.1 |
| | | offload | 89.59 | 89.59 | 44 offloaded | 0 | 36.4 |
| resnet18 | none | reuse only | 105.43 | 106.18 | | | 15.0 |
| | 75% (79.1) | greedy | 78.05 | 78.92 | 22 | 9.31e6 | 16.4 |
| | | min cut | 78.93 | 80.03 | 9 | 6.95e6 (-25%) | 15.1 to 16.4 |
| | | offload | 77.43 | 79.05 | 5 offloaded | 0 | 16.2 |
| | 50%, 35% | all four | over budget (TS420): best total 112.8 MiB (offload) against 96.2 MiB at 50% | | | | |

* **Every budget the greedy pass meets, the min cut meets, with 11% to 44% less recompute work** and
  gradients bitwise equal to the program without recompute (all 16 fitting variants). The LM at 35% is the
  one case where the time may differ beyond the noise: 30.0 ms against 36.7 ms for greedy in this run,
  26.6 against 28.4 in an earlier run with other CPU work going on.
* **Planned equals measured** on the LM within 0.12 MiB; on resnet18 the measured peak is 0.6 to 1.1 MiB
  over the plan (allocated bytes, which include the caching allocator's rounding; requested bytes equal
  the plan exactly in tests/test_remat.py on the small LM, not measured here for resnet18).
* **Offload alone** meets the 75% and 50% LM budgets and the 75% resnet budget with nothing recomputed,
  but here it is slower than recompute at 50% (42.5 ms): 25 copies of up to 4 MiB each way over PCIe,
  with 4 steps of lookahead, do not hide behind kernels this short. It is the option for values no step
  can recompute (resnet's convolution outputs when convolutions run through cuDNN, `library=`), and the
  closest to fitting the 50% resnet budget (112.8 MiB against 118.8 for greedy).
* **What does not work.** resnet18 at 50% and 35% fits with none of the four: what stays live at
  the planned peak is not something these passes can drop, and which values those are was not broken
  down (next step: the plan's live values at the peak, as in the TS420 message).
  "mincut+offload" adds nothing when the min cut fits already. The LM programs run with `verify="off"`
  in the benchmark because on origin/main e96cd2c this LM's training program fails its end to end
  float64 check before any recompute (`d_matmul_59`, relative error 0.00754 against 0.00153); the per
  value recompute and reload checks run in tests/test_remat.py on the small LM, which passes it.

## Tests that prove the tests work

Breaks made on 2026-11-24, each restored afterwards (tests/test_memory.py, 24 tests):

| break | tests that failed |
|---|---|
| the ownership guard always says yes | 4: the donated tensor the caller still holds is written over; both recompute tests and the in place count test (a planned write into an expand view, which does not cover its storage, is no longer refused: `call_into` raises or the result changes) |
| the planner forgets to free | 12: every planned against measured comparison |
| the recompute bitwise check skipped | 1: the corrupted recompute is no longer caught |

Found by these tests while building: the static schedule proposed writing over an expand view of a
scalar. The guard refused it at run time, so results were right, but the plan counted the write and
was off. The walk now applies the guard's conditions too (sole holder of a storage the tensor
covers, read from the dry run's meta tensors), and `test_every_planned_in_place_write_happens`
checks that the plan and the runtime agree on every in place launch.

## Automatic graphs and the plan (2026-11-25)

The plan describes an ordinary call. From its second call at a signature a CUDA program replays from
a CUDA graph (docs/COMPILER.md, Automatic graphs), whose private pool holds the call's working set
for as long as the program lives, next to what the caching allocator keeps. A program compiled with
``memory_budget`` or ``recompute`` is never graphed automatically, so its plan stays its footprint;
``graphs="off"`` does the same for any program.

## Limits (2026-11-24)

* cuDNN workspace is not planned (above). Index steps' range check temporaries are not planned
  either; both are listed in the plan's notes.
* No static arena: values still get their blocks from PyTorch's caching allocator, so the
  fragmentation half of N8 is not addressed; the planner predicts the bytes, not the block layout.
* A `cond` is planned with its larger branch; a `while` loop with at least two iterations.
* In place reuse and donation run only on CUDA (the Triton path); the CPU backend frees dead values
  but never writes over one.
* Inputs are planned as already on the executor's device in its precision; an input that needs a
  cast allocates a copy the plan does not include.
* Recompute recomputes only deterministic steps. The min cut is optimal for its objective (stored bytes
  plus lambda times work), not for the peak itself: the peak is planned exactly for each cut of the sweep
  and refined greedily, so the result is a good feasible plan, not a proven minimum of recompute work.
  Offload works only on CUDA and only for values with one live range and no view live across the
  boundary. Neither is wired into `truestar.train.Trainer` yet (pass `memory_budget` and `recompute` to
  the executor), and the Trainer does not donate yet.
* Fixed 2026-11-24: a compiled AdamW step failed its kernel check because the bias correction
  `1 - exp(t ln b2)` cancels in float32 (relative error 2.3e-5 against the 2e-5 bar). Both optimizers
  now write it as `-expm1(t ln b)`: `truestar.train` (`train/optim.py`, docs/TRAINING.md, Compiled)
  with an `expm1` IR op and kernel op, and `truestar.optim` (docs/OPTIM.md) with `ops.expm1`; a
  compiled float32 AdamW step passes its first call check. The donation tests still use SGD with
  momentum.
* Random steps: the keyed generator's own temporaries are not planned (listed in the notes).
* Measured only on the RTX 5070 Ti under Windows, with the GPU shared.
