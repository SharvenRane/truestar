# Parallel layouts against the single device step (`truestar parallel`)

Author: Sharven Rane. First written 2026-11-24. Ledger row P34 (docs/QUALITY_BAR.md), research
item R5 in <research>/PAIN_RESEARCH_3.md.

Tensor parallel, data parallel, gradient accumulation and sequence parallel layouts can train a
different model than one device does, with no error: the run trains, just worse. Megatron-LM#7452
(found by fuzzing, repros on CPU with gloo) lists silent wrong losses, gradients and activations
from parallelism composition; Megatron-LM#5974 reports a 1.3x to 2x smaller step under FSDP than
DDP for the same gradient; Megatron-LM#1754 a TP 2 loss off by 47 after 2000 steps; pytorch#149768
a DTensor gradient clipping bug; transformers#34694, #35204, #37197, #35856 and #40564 the gradient
accumulation normalization bug; verl#4710 a 3x grad norm gap between two backends. The usual check
is two loss curves over hundreds or thousands of steps, which names nothing.

`truestar parallel` runs one training step twice, as the single device program and under the
layout on simulated ranks, and names the module or the collective that differs.

```
truestar parallel examples/parallel_model.py --layout tp=2,dp=2,ga=2,sp=1
truestar parallel model.py --layout tp=2,dp=2 --optimizer adamw --clip 1.0 --format agent
truestar parallel model.py --layout tp=2 --ranks gloo        # one CPU process per rank, torch.distributed
```

```python
from truestar.parallel import parallel_check, CrossEntropy
r = parallel_check(model, x, "tp=2,dp=2,ga=2,sp=1", plan=plan, targets=y, loss=CrossEntropy())
print(r.to_text()); doc = r.to_agent()
```

**Simulated by default, real with `--ranks devices`.** With `--ranks threads` (default) or `gloo`
the ranks are threads of one process or gloo processes on the CPU: the arithmetic of the layout
(every shard, every partial sum, every gather) is real, an interconnect is not, and the report
carries TS690. With `--ranks devices` every rank is its own process on its own device, joined by
torch.distributed (NCCL between GPUs on Linux, gloo elsewhere), and the report carries TS820 with
each rank's host, device, GPU model and backend (section Real ranks below).

## The model file

`get_model()` and `get_inputs()` as for `truestar check` (the batch is dim 0), plus optional:

* `get_targets()`: default a fixed random target of the output's shape with mean squared error;
* `get_loss()`: an object with `sum(out, target)` and `count(target)`; `truestar.parallel.MSE()`
  and `CrossEntropy(ignore_index=-100)` are built in;
* `get_parallel_plan(layout)`: module name to style. Without it, only unambiguous MLP pairs are
  split (below).

## Layouts and plans

`--layout tp=N,dp=N,ga=N,sp=0|1,pp=1`. Ranks are numbered tp first (rank = dp rank * tp + tp rank).
`pp` above 1 is refused (TS680): pipeline parallel is not built, and nothing is approximated by
another layout. Plan styles:

| style | what each tp rank holds and does |
|---|---|
| `column` | a slice of the Linear's output features (weight rows and bias); input gradient summed over tp in backward |
| `column:K` | the same for a weight fused from K equal parts (gate and up, q k v): each part is sliced separately |
| `row` | a slice of the input features (weight columns); partial outputs summed over tp, the bias added once after the sum |
| `scatter` | with sp: the module's output is split along the sequence (the residual stream enters the sequence parallel region) |
| `gather` | with sp: the output is gathered along the sequence (the stream leaves it) |
| unnamed | replicated: every tp rank holds the whole module |

With `sp=1` a column layer all gathers its input along the sequence (backward: reduce scatter) and
a row layer reduce scatters its output (backward: all gather). Weights of modules that run on a
sequence piece (norms, row biases) get partial gradients, which the step sums over tp.

The automatic plan only pairs two nn.Linear leaves of the same parent, consecutive among the leaves
that own parameters, where the first widens (in differs from out) and the second takes exactly that
width back: an MLP up and down projection. Square projections are never guessed, because q followed
by k has the same shapes as a real pair, and a wrong plan made by Truestar would be a false alarm.

## Method

Both arms run in float64. A correct layout then differs from the single device run only by the
order of its sums, about 1e-16 relative, so one fixed threshold of 1e-9 separates a layout bug from
noise with seven orders of margin and needs no calibration run (measured worst over 88 correct
runs: 3.7e-11). A gradient that is zero in exact math (attention's key bias) is measured against
1e-5 of the largest gradient of the same module instead of against itself.

The single device arm is plain torch: the whole batch, one forward, the loss sum divided by the
count, one backward, `clip_grad_norm_`, one optimizer step. The layout arm is written from the
definitions: shard the named weights, wrap their forward with the collective their split needs, run
`ga` micro batches with every micro loss divided by the global count (all reduced over dp, so
uneven token counts stay right), sum the gradients of sequence parallel weights over tp, sum every
gradient over dp in one bucket, clip by the global norm (shards counted once each, replicated
weights once), and step a torch optimizer on the local pieces (SGD and AdamW act per element, so a
shard steps exactly as the same entries of the whole tensor). Every rank records each module's
output, the gradient arriving at it, every parameter gradient after every sync, and the updated
weights.

The pieces are reassembled (tp pieces joined by the split the rank recorded, fused parts
interleaved back, micro batches and replicas joined along the batch) and compared in an order that
makes the first difference its own cause:

1. **shards**: every rank's initial piece against the plan's split of the reference weight; a
   wrong piece is named with the split it actually is (TS681);
2. **forward**: the first module, in the order forwards finish, whose output differs (TS681); a
   row layer whose difference is exactly tp minus 1 copies of its bias is named as that;
3. **loss**: outputs match but the loss differs (TS682); the usual wrong normalizations (mean of
   micro batch means, sum or mean of replica means) are computed on the same outputs and the one
   the layout matched is named;
4. **backward**: the last leaf whose output gradient differs while every later one matches; the
   leaf after it passed the wrong gradient back (TS685), and for a column layer that is its
   backward sum over tp;
5. **parameter gradients**: a replicated weight on which ranks disagree (TS684, with whether the
   ranks' sum is the reference, that is a missing all reduce); one factor over every gradient
   (TS683) matched to dp, ga, tp, the world size or an inverse, and reported as ambiguous when two
   of those are equal in the layout; else the module (TS686);
6. **update**: right gradients and wrong weights after the step (TS687), with the clipping norm
   each rank used against the single device norm;
7. **drift**: replicated weights that differ between ranks after the step (TS688).

A rank that raises (a shape error from a wrong split, model code that reads a sharded size) is
TS689 with the module it was in and the user's line; the other ranks are released at once (thread
barriers abort; gloo processes are stopped 3 s after a peer fails).

## Injected faults: one step, one finding each

`--inject NAME` (or `faults=[...]`) switches one known bug into the layout arm, so the report can
be seen on a known cause. Measured with `scripts/parallel_study.py` (tiny transformer, layout
tp=2,dp=4,ga=3,sp=1, batch 24 by 4 tokens, clip 0.5, so dp 4, ga 3, tp 2 and world 8 all differ;
the mean of means case on a padded cross entropy classifier at tp=2,dp=2,ga=2), threads:

| fault | bug class (source) | first finding | named at | loss at step 1 differs |
|---|---|---|---|---|
| dp_sum_of_means | per replica mean, all reduce sums (Megatron-LM#5974, verl#4710) | TS682, sum of per replica means, gradients 4x | loss normalization | yes |
| dp_average_twice | averaged over dp on top of the global count | TS683, 0.25 times (1/dp) | gradient bucket all reduce over dp | no |
| ga_divided_twice | accumulated gradients divided by ga again (transformers#34694 class) | TS683, 0.333 times (1/ga) | gradient accumulation | no |
| ga_mean_of_means | per micro batch mean with padding (transformers#34694) | TS682, mean of per micro batch means with the counts | loss normalization | yes |
| missing_tp_sync | sequence parallel norm gradients not summed over tp (Megatron-LM#7452 class) | TS684, ranks' sum equals the reference | blocks.0.norm1, its all reduce over tp | no |
| column_input_grad_not_reduced | column input gradient not summed over tp | TS685 | blocks.1.mlp.gate_up input gradient reduce scatter (backward) | no |
| row_bias_every_rank | bias added on every rank before the sum | TS681, exactly 1 extra copy of the bias | blocks.0.attn.o output all reduce | yes |
| fused_split_as_block | fused gate and up split as one block (wrong split) | TS681, "a split along dim 0 as one block", plan column:2 | blocks.0.mlp.gate_up | yes |
| row_split_dim0 | row weight split on the wrong dim | TS689 shape error | blocks.0.attn.o | crash |
| clip_local_norm | clipping by the local shard norm (pytorch#149768 class) | TS687, both norms | gradient norm all reduce over tp | no |

10 of 10 named in one step, 0.1 s to 0.4 s each on threads. The baseline, comparing the step 1
loss, sees 4 of 10 (plus the crash) and names none; the other 5 change only gradients or the
update, which a loss curve shows only after enough steps for the trajectories to part. With equal
dp and ga (dp=2,ga=2) a ga fault is reported as "1/dp or 1/ga" with both causes, never as one of
them chosen at random.

The fault model is Truestar's own runtime with one rule broken, not the code of the frameworks in
those issues; the findings of Megatron-LM#7452 were not rerun (no code from other projects is used
here).

## False alarms and cost

* 0 of 88 correct runs flagged: the tiny transformer at tp=2, tp=4, tp=2,dp=2, tp=2,dp=2,ga=2,
  tp=2,ga=2,sp=1, tp=2,dp=2,ga=2,sp=1, tp=4,dp=2,sp=1, dp=4,ga=2 and dp=8, 4 seeds, SGD and AdamW
  with clipping; the padded classifier at three layouts with cross entropy; the MLP with the
  automatic plan. Worst error over every stage 3.7e-11 against the 1e-9 threshold.
* The wrong normalization `ga_mean_of_means` on micro batches with equal counts is the right
  normalization and is not flagged.
* Cost (tiny transformer, this desktop CPU): threads, tp=2,dp=2,ga=2,sp=1, 0.17 s in all (64
  collectives per rank); gloo processes, the same layout, 5.1 s (process start up is most of it).
  A 4 block, width 256 transformer at batch 16 by 32 tokens: 0.63 s on threads.

## Real ranks (`--ranks devices`, ledger row P39)

Added 2026-11-24. Code: `src/truestar/parallel/devices.py` (joining and the communicator),
`gradsync.py` (gradient buckets), `sharded.py` (sharded state), `realranks.py` (the rank processes),
tests `tests/test_multigpu.py`, measurements `scripts/multigpu_verify.py`, Kaggle kernel builder
`scripts/kaggle_multigpu.py`.

```
truestar parallel examples/parallel_model.py --layout tp=2,ga=2,sp=1 --ranks devices        # one GPU per rank
truestar parallel examples/parallel_model.py --layout dp=2,shard=2 --ranks devices --order native
truestar parallel examples/parallel_model.py --layout dp=2 --ranks devices --device cpu    # CPU processes
# two machines, one GPU each: the same command on both, rank 0 listens on the port
truestar parallel examples/parallel_model.py --layout dp=2 --ranks devices --dist-master 192.0.2.10:29515 --dist-world 2 --dist-rank 0
truestar parallel examples/parallel_model.py --layout dp=2 --ranks devices --dist-master 192.0.2.10:29515 --dist-world 2 --dist-rank 1
```

The rank program is the one the simulated ranks run, so every finding, code and fault of the
sections above holds unchanged; both arms stay float64 (the single device arm runs on the first
rank's device) and the threshold stays 1e-9. What changes:

* **Collectives are real.** `DeviceComm` calls torch.distributed between processes. Two orders for
  every sum over ranks: `--order fixed` (default) adds every element's pieces left to right in rank
  order, so the result is bitwise the same on every rank, every run, and equal to one device adding
  the same pieces in the same order; `--order native` is the backend's all reduce (for 2 ranks each
  element is one `a + b`, so it equals the fixed order bitwise; for 3 or more NCCL's ring or tree
  association is not the rank order, reported as TS825). The fixed sum cuts the flat tensor into
  world equal pieces: an all to all hands rank r piece r of every rank, rank r folds them in rank
  order, an all gather returns the folded pieces (the fixed reduce scatter stops after the fold).
  That moves the bytes of a ring all reduce, 2 (world - 1) / world of the tensor; the earlier all
  gather of every whole tensor moved world - 1 times it and is kept as `fixed_algo="gather"`, bitwise
  equal (tests/test_fixed_scatter.py). On CUDA the bucket version runs on a side stream, so the fold
  does not stall the backward pass. Measured against torch DDP on 4 GPUs: docs/TRAINING.md, Scaling.
  With gloo and CUDA tensors every collective goes through host memory.
* **Gradient buckets that start during backward.** The dp gradient sum is cut into fixed buckets
  (parameters in reverse registration order, `--bucket-mb`, default 25). A hook fires when a
  parameter's gradient is final (`register_post_accumulate_grad_hook`); a bucket starts when its
  last gradient arrives, and only after every earlier bucket started, so all ranks issue the same
  collectives in the same order whatever order their gradients finish in. With gradient accumulation
  only the backward of the last micro batch starts buckets. `--no-overlap` reduces them after the
  backward pass instead.
* **Sharded state (`shard=1`, `shard=2` in the layout).** Every trained parameter of a rank is laid
  out in one flat vector padded to a multiple of dp; replica i owns one contiguous slice. The flat
  gradient is reduce scattered, the squared norm of the slices is summed over dp (and tp) for
  clipping, the optimizer steps the slice with its own state (SGD and AdamW act per element), and the
  updated slices are all gathered back (`shard=1`). With `shard=2` a replica holds only its slice of
  the parameters between steps and the step all gathers them for compute. A new injected fault,
  `shard_stale_params` (the all gather after the update left out), is named as TS687 at "parameter
  all gather over dp (sharded state)". Gathering per layer during forward and backward, instead of
  for the whole step, is not built.
* **Starting ranks.** Locally one process per GPU (`cuda:0` to `cuda:N-1`; more ranks than GPUs is
  TS821, two ranks never share a GPU), meeting over a file store. Across machines every machine runs
  the command with its rank; rank 0 sends the float64 weights and the batch, gathers every rank's
  record and writes the report. A rendezvous that does not complete is TS822 with the address. With
  gloo, the interface that routes to rank 0 is pinned (`GLOO_SOCKET_IFNAME`, needs psutil): on the
  desktop the host name resolved to the WSL adapter (172.30.32.1) first, and the first two machine run
  hung until its timeout because of it.

### Measured on real GPUs (2026-11-24)

Kaggle GPU T4 x2 (private kernel `sharvenrane/truestar-multigpu-t4x2`, two Tesla T4 in one machine,
NCCL 2.28.9, torch 2.11.0+cu130), and across the desktop (RTX 5070 Ti) and the laptop (RTX 5060
Laptop GPU) over the home wireless network with gloo, torch 2.11.0+cu128.

| check | T4 x2, NCCL | desktop + laptop, gloo over the wireless network |
|---|---|---|
| clean layouts, float64, `parallel_check(ranks="devices")`: dp=2, dp=2 ga=2, tp=2, tp=2 sp=1, tp=2 ga=2 sp=1, dp=2 shard=1, dp=2 ga=2 shard=2, fixed and native order | 14 of 14 OK, worst 3.5e-11 against 1e-9, about 4.8 s each (process start and CUDA init) | dp=2 and tp=2 sp=1 OK, worst 3.5e-11, about 10 s each |
| injected faults on real ranks (the 10 of the table above plus `shard_stale_params`) | 11 of 11 named, the same code, module and collective as on threads | not run |
| `truestar parallel ... --ranks devices` command line, 4 layouts and one fault | exit 0 four times, exit 1 with TS684 at blocks.0.norm1 | exit 0 twice |
| `tests/test_multigpu.py` | 28 passed (including the two GPU tests) | not run |

The dp=2 layouts on threads compare against the simulated step; these compare against the same
step computed by two processes on two GPUs, so a wrong rank to device mapping, a collective issued
in another order on one rank, or a backend reduction bug would show here and not there.
Speed of the layout check itself is not the point (it records every activation in float64); the
float32 training speed of the same pieces is in docs/TRAINING.md, Multi GPU.

## Bitwise tensor parallel by one canonical tree (ledger row P55)

Added 2026-11-24. Code: `src/truestar/parallel/tree.py`; tests `tests/test_tree_tp.py`; measurements
`scripts/tree_tp_study.py`, results in `results/tree_tp/`. Research item 4.4 (sources: Zhang et al.,
"Deterministic Inference across Tensor Parallel Sizes", arXiv 2511.17826; NCCL issue 1497).

The check above compares in float64 within 1e-9, because a tp step adds its partial sums in another
order than one device does. This makes the order the same, so a float32 tp step is **bitwise** the
one device step. Every contraction of every nn.Linear (forward, input gradient, weight and bias
gradient) walks one canonical binary tree fixed by the full shapes: the reduction axis (per part, for
a fused `column:K` weight) is cut into G equal groups, G the largest power of two up to `cap`
(default 8) dividing its length; the group sums are added pairwise, (2i, 2i+1), an odd last one
carried up; the parts after that. A rank holding G/tp aligned groups of a split axis walks exactly a
subtree, and the cross rank step all gathers the tp subtree roots and finishes the same tree (row
forward, column input gradient). The clipping norm uses the same grouping on each weight's shard
axis (squares in float64). So both arms perform the same additions in the same order, for every
power of two tp that divides G.

tp with an odd factor (added 2026-11-24, ledger row P66): for tp = W * 2^a with W odd (3 for tp=3
and tp=6), each split axis is first cut into W equal blocks and each block into its own G groups,
so the tree has W * G groups; its first log2(G) levels pair inside a block, so a rank's G / 2^a
groups are again a subtree. `tree_check` gives the one device step the W of the tp it is compared
with (`ways`, default 1, which is the tree above bit for bit: 192 of 192 tensors equal to the
earlier code on three widths and both leaves). A split axis whose length W does not divide, or a
block whose G the 2^a part does not divide, is refused with TS1102 naming the layer and the counts.
Measured on CPU threads, AdamW and clipping: tp=3 and tp=6 (width 24, ff 48) and tp=4 and tp=8
(width 32, ff 64) bitwise equal to the one device step; tp=3 also as 3 gloo processes.

One measured exception outside the linear layers: torch 2.11's CPU `silu`, `gelu` and `sigmoid`
give other bits on a strided view than on a contiguous tensor for some widths (the gate half of a
fused `gate_up` output is a strided view). The tiny transformer with a silu gate is therefore not
bitwise at tp=2 for ff=48 (or tp=4 for ff=32 and 64), with TS1101 naming the outputs; with a tanh
gate, which showed no such difference, every case above is bitwise. In the same probe a contiguous
copy of the gate gave equal bits at every width tried (6 to 64), so a model can avoid it with
`.contiguous()` before the activation; the tree does not change user models.

```python
from truestar.parallel.tree import tree_check
r = tree_check(model, x, tp=4, plan=plan, leaf="exact", optimizer="adamw", clip=1.0)          # CPU threads
r = tree_check(model, x, tp=2, plan=plan, device="cuda", ranks="gloo", model_file="model.py")  # one process per rank
print(r.to_text())   # TS1100 bitwise, TS1101 differs (named), TS1102 cannot split, TS1103 the plain step's gap
```

How a group sum is formed (`leaf`):

* `exact`: each product is one IEEE multiply, the group is a pairwise tree down to single products.
  Equal by construction on any device whose float multiply and add round to nearest. This is the
  proven mode.
* `tiled`: one batched matmul whose entries are canonical output tiles (edges from the full sizes),
  each contiguous, so each entry has the same shape and layout in both arms; only the batch count
  differs. Equal only when the library's per entry result does not depend on the batch count:
  measured true on the CPU (MKL), false for cuBLAS on the small model (below).
* `block`: one matmul per group, shapes as they come. Equal only when the library kernel is shape
  stable; kept as the ablation that shows why the tree alone is not enough.

### Measured (2026-11-24)

One training step (forward, MSE, backward, clip 1.0, AdamW or SGD with momentum), float32. Models:
the example transformer (`examples/parallel_model.py`, d16) and the same at d256, ff 512, 8 heads,
batch 8 by 16 tokens. Values compared: loss, output, every gradient and every updated weight on
every rank (128 at tp=2, 256 at tp=4). 46 runs, `results/tree_tp/check.json`.

| leaf | CPU, threads and gloo processes (desktop) | CUDA, gloo processes on the RTX 5070 Ti | desktop + laptop, gloo over the network |
|---|---|---|---|
| exact | 10 of 10 bitwise (tp 2 and 4, both models, both optimizers) | 4 of 4 bitwise | bitwise (128 of 128) |
| tiled | 12 of 12 bitwise | 2 of 4: the d16 model differs in 32 and 59 values (cuBLAS picks by batch count) | bitwise |
| block | 6 of 12: the d16 model differs in 84 to 218 values | 3 of 4 | 107 of 128 differ |
| plain step (F.linear, rank order sums) | 122 to 256 of 128 or 256 values differ in every run, worst relative difference on loss, outputs and updated weights 1.3e-3 to 2.6e-2 (AdamW turns a last bit gradient change into a full step) | same | |

Across machines: rank 0 and the one device step on the desktop (AMD family 26, Zen 5), rank 1 on the
laptop (AMD family 25, Zen 4), torch 2.11.0 on both, tp=2, CPU, gloo over the home network
(`scripts/tree_tp_study.py machines`, `results/tree_tp/machines_*.json`). The exact and tiled steps
are bitwise equal to the desktop's one device step; block is not.

**Speed cost.** One nn.Linear forward and backward (`scripts/tree_tp_study.py speed`,
`results/tree_tp/speed.json`, best of 5, GPU shared):

| device | M x K x N | F.linear | tiled | exact |
|---|---|---|---|---|
| CPU | 256 x 512 x 512 | 0.66 ms | 3.4 ms (5.2x) | 75 ms (114x) |
| CPU | 1024 x 1024 x 1024 | 8.1 ms | 53 ms (6.5x) | 1363 ms (169x) |
| CUDA | 256 x 512 x 512 | 0.31 ms | 2.0 ms (6.5x) | 6.6 ms (21x) |
| CUDA | 1024 x 1024 x 1024 | 0.33 ms | 2.5 ms (7.6x) | 58 ms (174x) |

In that table the tree is torch ops: the exact leaf materializes every product, and both leaves pay
Python and launch overhead for the tree levels. The communication cost is the same bytes as an all
reduce times tp (an all gather of the tp roots, then the finish on every rank).

**Triton chunk kernel for the exact leaf (2026-11-25).** On CUDA float32 the exact leaf now runs
`parallel/tree_kernel.py`: one program forms a 16 x 16 tile of the products of an aligned chunk of C
consecutive reduction entries (C the largest power of two up to 64 that divides the group size) in
registers and adds them pairwise over log2(C) levels, with PTX `mul.rn.f32` and `add.rn.f32` (one
rounding each, no fused multiply add, subnormals kept). Those are the first levels of the same
pairwise tree, so `ptree` over the chunk sums finishes it with the same additions in the same order.
Bitwise equal to the torch leaf (tests/test_tree_kernel.py: chunk widths 2 to 64, carried odd chunks,
fused parts, strided operands, subnormal products, a tree_linear forward and backward) and the tp 2
step on two A10G processes stayed bitwise equal to one device (tests/test_tree_tp.py, A10G x2).
`TRUESTAR_TREE_KERNEL=0` runs the torch leaf. One nn.Linear forward and backward on an A10G
(`scripts/tree_tp_study.py speed --cuda`, `results/tree_kernel/speed/speed.json`, best of 5):

| M x K x N | F.linear | tiled | exact, torch leaf | exact, chunk kernel |
|---|---|---|---|---|
| 256 x 512 x 512 | 0.20 ms | 1.9 ms (9.4x) | 7.0 ms (34x) | 1.7 ms (8.5x) |
| 1024 x 1024 x 1024 | 0.53 ms | 4.3 ms (8.1x) | 109 ms (208x) | 3.7 ms (7.1x) |

The exact leaf, the one that is bitwise on every device, is now faster than tiled (which is not
bitwise on CUDA) and 29x faster than before at 1024. What is left is the group and chunk levels in
torch ops and the per call launches; a kernel that also walks those levels is not built.
(Since the `ways` field was added for odd tp, the study script passed the leaf positionally into it,
so every leaf timed the exact torch leaf: the first A10G run here gave 109 ms for all three. It now
passes keywords; the table above is from the fixed script.)

What it does not cover: dp, ga and sp layouts split the batch or the sequence, so their equality is
a batch invariance property of every op of the model (norms, attention, losses), not only of the
linear layers; `tree_check` runs tp layouts only. Ops other than nn.Linear run as torch ops in both
arms; they get the same inputs and shapes except attention, whose per head matmuls see fewer heads
per rank and were bitwise equal in every run above, which is measured, not proven. Thread ranks are
CPU only: CUDA backward passes of all threads share one autograd engine thread per device, so a rank's
backward collective waits forever for a peer queued behind it (found while building); CUDA ranks run
as processes.

## Limits

* Real ranks: two GPUs in one machine (T4 x2, NCCL) and two machines with one GPU each (gloo over
  wireless network) were run; more than two GPUs, NVLink, InfiniBand and multi node NCCL were not. Gloo reduce
  scatter is an all reduce and a slice.
* The layout arm is Truestar's own: a user's Megatron, FSDP or DeepSpeed step is not run. The
  check tells whether the model and its plan compose into the single device program, and shows on
  known faults what each class of bug looks like; checking a third party step needs that step's
  per rank pieces, which is not built.
* Sharded state gathers the parameters for the whole step, not layer by layer; pipeline parallel,
  vocabulary parallel embeddings and cross entropy, context parallel attention and expert parallel
  are not built.
* The model must return one tensor that the loss reads, its last module must be replicated or row
  (a column output is not gathered for the loss), and every input must carry the batch as dim 0.
* Float64 only: precision effects of a layout in bfloat16 (reduction order inside a sum over tp)
  are not measured here; that is P14's question.
* Parameters used outside module forwards are matched by name only; a weight shared by two modules
  is checked as one parameter.
