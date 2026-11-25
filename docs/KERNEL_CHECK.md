# Kernel check (`truestar kernel`) and verified autotuning

Pain R1 (research round 3): kernels written by coding agents, RL trained kernel models and people
pass a one input `allclose` with a hand picked tolerance and are still wrong: at another shape, on
strided inputs, on large or offset values, on NaN, now and then (races), or under one autotune config
(R7). The same check fails correct kernels that round differently. `truestar kernel` is the check to
point at a kernel inside your own project, with a tolerance derived instead of guessed
(docs/TOLERANCE.md) and one stable code per broken property.

```
truestar kernel ref.py cand.py                       # text report, exit 0 ok, 1 finding, 2 usage
truestar kernel ref.py cand.py --dtypes float16,bfloat16 --format agent
```

```python
from truestar.kernelcheck import check_kernel
report = check_kernel(reference, my_kernel, (x, w))
print(report.to_text())
```

`ref.py` defines `reference(*inputs)` (or a KernelBench `Model` with `get_init_inputs()`) and
`get_inputs()`, optionally `DTYPES = ["float16", ...]`. `cand.py` defines `kernel(*inputs)` or
`candidate(*inputs)` (or a KernelBench `ModelNew`, which gets the reference's weights). The MCP tool
`verify_kernel` runs the same check (docs/AGENT_INTERFACE.md).

## Method

The candidate runs on input variants built from the example inputs; every output is judged with the
derived tolerance of the reference on exactly those inputs:

| variant | what it catches |
|---|---|
| example inputs | the ordinary case |
| shapes: every distinct size above 1 is a symbol (equal sizes tied), moved by one up and down; all sizes odd; all sizes at most 7 | a last tile, ragged block or mask that only the example's size avoids (TS200), or a raise (TS201) |
| large values (max 100), a large offset (x + 64 rms), small values (x / 128), a constant tensor, NaN in one element, on the first floating input | no max subtraction, one pass variance, a wrong epsilon, ties, dropped or spread NaN (TS202) |
| transposed and sliced (step 2) storage for every input with two or more dims | strides assumed instead of read (TS204) |
| extra dtypes | a precision path the example does not use |
| the same inputs twice, compared bit for bit | races and timing dependent order (TS203) |

Around every candidate call: on CUDA the caching allocator's free blocks (small and large pool) are
filled with NaN first, so output elements a kernel never writes show as NaN (TS205); the inputs are
compared before and after (TS206); the reference runs after the candidate, in float64, so there is
nothing to read back from it. A variant the reference itself rejects is skipped and listed.

**Blame.** Every failure names the variant, the elements (count, worst index, value against float64,
ulps) and where they sit (`in the tail of dim 1 (indices 192 to 199 of 200)`, `only in indices 128 to
190 of dim 2`, a diagonal offset). On the first failing variant, a candidate built from ATen ops is
traced and each op is rerun alone in float64 on its own recorded inputs with its own derived
tolerance; the first op that is wrong on its own is `blamed_op` (for example a backend `aten.rsqrt`
that is 1% off). Otherwise `op_note` says that every op is right alone, so the bug is in how they are
combined or in a custom kernel launch, which ATen does not see.

## Measured (2026-11-24, `scripts/bench_kernel_check.py --draws 3 --cuda`, RTX 5070 Ti)

50 kernels (`truestar.kernelbench`, 39 in PyTorch on the CPU, and `truestar.kernelbench_triton`, 11
Triton kernels on the GPU), 3 input draws each: 69 correct runs, 81 wrong runs. Baselines check only
the example input, against the reference in the same dtype.

| checker | false fail | false pass |
|---|---|---|
| `truestar kernel` | 0.0% (0) | 6.2% (5) |
| `allclose(rtol=1e-2, atol=1e-2)` (KernelBench) | 0.0% (0) | 42.0% (34) |
| `allclose(rtol=1e-4, atol=1e-4)` (the YAKK baseline) | 36.2% (25) | 22.2% (18) |
| `torch.testing.assert_close` defaults | 23.2% (16) | 16.0% (13) |

Median 0.29 s per kernel, worst 3.6 s. Bugs a single allclose at 1e-2 passes and `truestar kernel`
names: a column tail right at 256 columns and wrong at 257 (TS200 on `d2=257`, TS205 for the Triton
grid that floors N), a contiguity assumption (TS204, PyTorch and Triton), softmax without the max
subtraction (TS200 on large offset values), a racy and an atomic row sum (TS203), a wrong epsilon, a
tanh GELU, a constant off by 1e-5, the attention scale 1/sqrt(d+1).

The 5 false passes: `mm_scale_1e-3_fp16` on 2 of 3 draws (a constant off by one to two float16
ulps, below the dtype's resolution; one draw is caught on a value variant) and `triton_mm_f16acc` on
3 of 3 (a float16 accumulator over 16 K steps at K = 512, which is the error PyTorch's default
`allow_fp16_reduced_precision_reduction` lets cuBLAS make; with that flag off it fails at over 100x
its bound, and the same kernel at K = 4096 fails under the default policy too).

Not measured here: process isolation, a roofline bound on impossible speed, and whether the target
DSL actually ran. The set above was written in this project, by the same author as the checker; the
held out sets below were not. The bounds in `tests/test_kernelcheck.py` (`fp <= 1`) and
`tests/test_tolerance.py` (`BELOW_PRECISION`) were set after measuring this set, not before.

## Held out sets: kernels this project did not write

### Protocol and bounds, written 2026-11-24 before any held out run

Sets (`scripts/fetch_heldout.py`; sources and labels only, no tool code):

* **KB**: KernelBench samples (huggingface.co/datasets/ScalingIntelligence/kernelbench-samples,
  Apache 2.0), `baseline_eval`, sample 0 of deepseek-r1 and openai-o1 on levels 1 and 2, every
  kernel that compiled (356 of 388), against the KernelBench problems (MIT) at commit b866d6ab57.
  Published label: KernelBench's eval on an L40S (5 random inputs, allclose atol = rtol = 1e-2
  against the float32 reference): 240 correct, 116 not (mismatches and runtime errors together).
  Stored in `tests/data/heldout/kernelbench/`.
* **METR**: METR's KernelBenchFiltered best agent solutions, levels 1 and 2 (171 kernels), commit
  9507be4 (the one the YAKK post links). Every one is published as passing METR's check
  (atol 0.01). The YAKK post (hud.ai, 2026-11-19) names 5 of them as numerically wrong (level 1
  tasks 14, 33, 100; level 2 tasks 24, 42); its other 9 and its per kernel verdicts are not public.
  The repository has no license file, so these files are fetched to `results/heldout_metr/`, which
  is never committed.
* **YAKK**: the 21 lemon families and YAKK's grader are not public (no repository, no data; the
  post promises a technical report). Not used.

Run: every kernel once on a Modal L4 (sm_89, the architecture of the L40S that labelled KB), each
in its own process, through `scripts/bench_heldout.py`: `check_kernel` with its default variants (no
extra dtypes, since the tasks are float32), and on the very same outputs of every variant three
baselines: `torch.testing.assert_close` (defaults, `equal_nan=True`) against the float64 reference
cast to the output dtype, `allclose(rtol=atol=1e-2)` and `allclose(rtol=atol=1e-4)` against the
float64 reference. Each method is scored on three variant scopes: example input only; task scope
(example input plus the value families, at the task's shapes and contiguous layout, plus
truestar's repeat run); all variants (adds the shape and stride variants). A variant the reference
rejects is skipped for every method; a candidate that raises fails every method.

Ground truth: the published label, except where adjudicated. Every kernel where the all variants
verdict of any method (truestar, assert_close, allclose 1e-2) differs from the published label is
read by hand and given one label with a written reason:

* wrong: wrong beyond float32 rounding on an input the task allows (the task's shapes, any finite
  values), including an unstable algorithm whose error at the value families exceeds a stable
  float32 algorithm's by orders of magnitude;
* precision shedding: computes in less than float32 (fp16, bf16, TF32 operands) without the task
  asking for it; counted wrong in the main table, correct in a sensitivity row;
* shape limited: right at the task's shapes and contiguous layout, wrong or raising at other
  shapes or strides; correct at task scope, wrong at all variants scope;
* correct.

Kernels that fail to load on the L4 or exceed 300 s are excluded and counted.

Bounds, set now (task scope, adjudicated labels, per set):

1. `truestar kernel` false fail at most 5% of correct kernels.
2. `truestar kernel` false pass at most 15% of wrong kernels.
3. On METR, `truestar kernel` rejects each of the 5 kernels the YAKK post names that run here.
4. `truestar kernel` is called better than a baseline on a set only when its false fail plus false
   pass is at least 10 points lower than the baseline's on the same variants and its false fail is
   at most 2 points higher; otherwise the doc says there is no measured advantage.
5. Self authored small faults (a tail tile of 1024 rows, one channel; float16 and bfloat16; 3 draws
   each): no false pass.

A bound that fails is reported as failed, not moved.

### First run: what ran (2026-11-25, Modal L4, torch 2.11.0, CUDA 12.8, TF32 off)

`scripts/modal_heldout.py` (per kernel results in `results/heldout_runs/out/`, tables in
`results/heldout_runs/summary.txt` from `scripts/heldout_summary.py`). Of 527 kernels (KB 356,
METR 171), 384 finished: KB 247, METR 137. Excluded and counted: 77 ran out of L4 memory with four
kernels sharing the card (KB 48, METR 29, among them YAKK named METR level 1 task 33), 9 failed to
build or load (KB 4, METR 5), 9 KB exceeded 300 s, and 48 KB never got a slot before the Modal
budget for this task was spent. The excluded kernels lean to the largest inputs, so the rates
below describe the smaller ones.

**Deviation from the protocol.** 265 finished kernels had a method disagree with the published
label, too many to read one by one within budget. Instead `heldout_summary.py --adjudicate` applies
written rules to every kernel (`tests/data/heldout/adjudication.json` holds each label and the
variant that set it):

* wrong: labelled correct, and on the example input or a finite value family (the NaN family is
  not a finite input) the candidate raised, or its error is at least 100x the derived float32 bound;
* shape limited: the same only on a shape or stride variant;
* correct: labelled wrong by KernelBench, yet every method passes every variant (4 kernels; their
  label comes from KernelBench's harness, which runs train mode BatchNorm and does not share
  weights, while this harness runs eval mode with the reference's weights for every method).

The 100x rule uses truestar's own bound, which favours truestar. The strict rule also asks for an
error above 1e-2 (rtol and atol) against float64, which favours allclose 1e-2; both are reported.
Precision shedding was not separated out (no rule finds it without reading the source).

### First run: rates (false fail = correct kernels rejected, false pass = wrong kernels accepted)

Task scope is the scope of the bounds; finite drops the NaN family and the repeat run from it.

| set | scope | method | false fail | false pass |
|---|---|---|---|---|
| KB | example | truestar | 5/164 (3.0%) | 7/83 (8.4%) |
| KB | example | assert_close | 30/164 (18.3%) | 11/83 (13.3%) |
| KB | example | allclose 1e-2 | 1/164 (0.6%) | 17/83 (20.5%) |
| KB | example | allclose 1e-4 | 21/164 (12.8%) | 12/83 (14.5%) |
| KB | finite | truestar | 15/164 (9.1%) | 5/83 (6.0%) |
| KB | finite | assert_close | 55/164 (33.5%) | 10/83 (12.0%) |
| KB | finite | allclose 1e-2 | 7/164 (4.3%) | 15/83 (18.1%) |
| KB | finite | allclose 1e-4 | 34/164 (20.7%) | 11/83 (13.3%) |
| KB | task | truestar | 58/164 (35.4%) | 5/83 (6.0%) |
| KB | task | assert_close | 87/164 (53.0%) | 10/83 (12.0%) |
| KB | task | allclose 1e-2 | 47/164 (28.7%) | 13/83 (15.7%) |
| KB | task | allclose 1e-4 | 70/164 (42.7%) | 11/83 (13.3%) |
| METR | example | truestar | 5/104 (4.8%) | 7/33 (21.2%) |
| METR | example | assert_close | 24/104 (23.1%) | 9/33 (27.3%) |
| METR | example | allclose 1e-2 | 2/104 (1.9%) | 21/33 (63.6%) |
| METR | example | allclose 1e-4 | 15/104 (14.4%) | 9/33 (27.3%) |
| METR | finite | truestar | 9/104 (8.7%) | 0/33 (0.0%) |
| METR | finite | assert_close | 39/104 (37.5%) | 3/33 (9.1%) |
| METR | finite | allclose 1e-2 | 8/104 (7.7%) | 6/33 (18.2%) |
| METR | finite | allclose 1e-4 | 26/104 (25.0%) | 3/33 (9.1%) |
| METR | task | truestar | 36/104 (34.6%) | 0/33 (0.0%) |
| METR | task | assert_close | 56/104 (53.8%) | 3/33 (9.1%) |
| METR | task | allclose 1e-2 | 31/104 (29.8%) | 5/33 (15.2%) |
| METR | task | allclose 1e-4 | 46/104 (44.2%) | 3/33 (9.1%) |

Strict rule, task scope: KB truestar 68/174 (39.1%) and 5/73 (6.8%), assert_close 90/174 (51.7%)
and 3/73 (4.1%), allclose 1e-2 49/174 (28.2%) and 5/73 (6.8%), allclose 1e-4 73/174 (42.0%) and
4/73 (5.5%); METR truestar 43/111 (38.7%) and 0/26, assert_close 60/111 (54.1%) and 0/26,
allclose 1e-2 33/111 (29.7%) and 0/26, allclose 1e-4 50/111 (45.0%) and 0/26. Against the
published labels alone, and over all variants, see `summary.txt`. Median check time per kernel
(all methods together) 4.6 s, max 290 s.

Where truestar's 94 task scope false fails come from: 54 only fail the NaN family (every baseline
fails those too: the kernel drops or moves the NaN, e.g. a fused `fmaxf`), 13 only the repeat run
(TS203, run to run differences from atomics), 3 both, and 24 a finite input. Of those 24, 5 are an
infinity where float64 is finite (cumprod, softplus, mish at large values) or an output of the
wrong shape, which allclose 1e-2 rejects too; 11 are at 1x to 13x the bound (cumsum, sum and mean
of a constant tensor, instance norm, one element of 8.9 million at 1.0x): real false fails of the
derived bound; 8 are at 17x to 90x, under the 100x line.

### First run: bounds

1. False fail at most 5% (task scope): **failed** on both sets (KB 35.4%, METR 34.6%). On the
   example input only it holds (3.0%, 4.8%); at finite scope it does not (9.1%, 8.7%).
2. False pass at most 15% (task scope): **held** (KB 6.0%, METR 0.0%); strict rule 6.8% and 0.0%.
3. YAKK named METR kernels: **held for the 4 that ran** (level 1 tasks 14 and 100, level 2 tasks
   24 and 42, all rejected; every baseline rejects them too at task scope); task 33 ran out of
   memory and is not measured.
4. Better than a baseline (task scope, FF + FP at least 10 points lower, FF at most 2 higher):
   better than assert_close on both sets (KB 41.4 against 65.0, METR 34.6 against 62.9) and
   better than allclose 1e-4 on both (56.0, 53.3); **no measured advantage over allclose 1e-2**
   (KB 41.4 against 44.4; METR 34.6 against 45.0 but its false fail is 4.8 points higher). Under
   the strict rule only the METR win over assert_close stays (38.7 against 54.1); on KB it is 9.9
   points, short of 10.
5. Self authored small faults, no false pass: **failed**. `scripts/bench_kernel_check.py --draws 3`
   on CPU (`results/heldout_runs/self_set.txt`): the tail tile accumulating in the input dtype and
   the last channel without its bias are rejected 3 of 3 in float16 and bfloat16, and the float16
   tail tile scaled by 1 + 2^-7 too; the bfloat16 tail tile scaled by 1 + 2^-7 passes 3 of 3. That
   scale is about one bfloat16 ulp on a quarter of the output, below the per element bound; only
   a statistic over the tile (a mean signed error) would see it.

### Rerun policy, written 2026-11-25 before the rerun

Changes to the check, made from the first run's false fails:

* NaN handling is its own verdict. On the NaN family, when every finite element is within the
  derived bound and only NaN or infinity sits elsewhere than in float64 (a NaN dropped or spread),
  the variant is TS202 with status suspect, a warning, not a wrong number. A NaN family run whose
  finite elements are also wrong stays TS200.
* Nondeterminism stays TS203, suspect, as before: Truestar promises the same answer every run, so
  it is reported, but it is not a numerical error.
* `report.numerics` is the verdict on the numbers alone (any divergent variant); `report.status`
  adds the two verdicts above.
* Derived bound, two root causes (`scripts/repro_heldout_misses.py` reproduces both on CPU): a
  sum, mean or cumsum whose terms are all the same value now uses the worst case accumulation
  bound u sum_k |s_k| instead of the independent error model, since every step's rounding error
  then repeats; and batch norm (which instance norm runs on) had no model and got an elementwise
  bound; it now gets the layer norm model over its batch statistics.
* `lean=True` (`truestar kernel --lean`, TS1300) runs the TS960 lean test of `truestar.bias` on the
  example input, effect floor 1/64 of the compiled kernels' size tolerance for the output dtype.
  Not on the value families: they squeeze intermediates into a few rounding intervals, where
  round to nearest itself leans (a correct unfused bfloat16 SiLU leans toward zero at x / 128,
  measured on the self set).

Reported for every method, both ways: **numerics** (truestar's TS202 and TS203 do not reject; the
baselines judge the NaN family on the elements where float64 is finite, and have no repeat run)
and **full** (TS202 and TS203 reject; the baselines compare NaN positions). `truestar+lean` is the
numerics verdict plus TS1300. The lean verdict ships on by default only when its held out false
fail rate (task scope, numerics, adjudicated labels) is at most 1 point above truestar's without
it, on both sets; otherwise it stays opt in.

Added during the rerun (message only, verdicts unchanged, so the rerun numbers below do not
depend on it): a TS202 verdict whose output matches the reference with the NaN read as minus
infinity (or plus infinity) says the candidate drops NaN as IEEE 754-2019 maximumNumber (CUDA
`fmaxf`) or minimumNumber does, while the float64 reference propagates it as maximum and minimum
do. The held out run did not record this message, so how many of the NaN only cases it names is
not measured here.

Next step, not done on this branch: the tie rule above is a per op fix for one broken assumption
of the error model (independent, zero mean rounding errors), which constant inputs and truncating
tensor cores both violate. The general fix is a running error bound of the Wilkinson and Higham
kind (data dependent, always sound) for every reduction, with the probabilistic bound used only
where a cheap test says its assumptions hold (Higham and Mary 2019, 2020).

Adjudication, extended before the rerun: a source scan adds two labels the first run could not
see: the tanh approximation of GELU where the task's GELU is erf (wrong), and float16, bfloat16 or
TF32 compute (precision shedding: wrong in the main table, correct in a sensitivity table). The
pre set bounds 1 to 5 are unchanged and are judged on the numerics policy, adjudicated labels,
task scope.

### Rerun: what ran (2026-11-25, Modal L4, same image)

Results in `results/heldout_runs/out2/`, tables in `results/heldout_runs/summary_rerun.txt`. The
Modal budget (US$1.20 for the rerun) ran out before every kernel ran again. Finished: 316 of 527
(KB 265, METR 51). Not measured: METR 110 (budget: the METR shard ran one round), 52 out of GPU
memory even at two kernels per card (KB 44, METR 8), 18 that do not build or load or exceeded
300 s in the first run were skipped, and 2 more timed out. The METR rows below rest on 51 kernels
and are the weaker half of this report.

### Rerun: rates, numerics policy, adjudicated labels, task scope

Truestar here is the check as shipped now: the lean verdict is on by default (see below). One
adjudication rule was corrected after the rerun, because it missed a case the protocol already
calls wrong: infinity or NaN where float64 is finite, or an output of the wrong shape, on a finite
input with allclose 1e-2 agreeing (softplus and cumprod overflowing at large values, one wrong
output shape; 5 kernels).

| set | method | false fail | false pass |
|---|---|---|---|
| KB | truestar | 9/165 (5.5%) | 7/100 (7.0%) |
| KB | truestar without the lean verdict | 9/165 (5.5%) | 8/100 (8.0%) |
| KB | assert_close | 54/165 (32.7%) | 12/100 (12.0%) |
| KB | allclose 1e-2 | 6/165 (3.6%) | 19/100 (19.0%) |
| KB | allclose 1e-4 | 34/165 (20.6%) | 15/100 (15.0%) |
| METR | truestar | 1/38 (2.6%) | 1/13 (7.7%) |
| METR | truestar without the lean verdict | 1/38 (2.6%) | 3/13 (23.1%) |
| METR | assert_close | 15/38 (39.5%) | 4/13 (30.8%) |
| METR | allclose 1e-2 | 2/38 (5.3%) | 4/13 (30.8%) |
| METR | allclose 1e-4 | 10/38 (26.3%) | 4/13 (30.8%) |

Full policy (TS202 and TS203 reject, the baselines compare NaN positions), same labels and scope:
KB truestar 54/165 (32.7%) and 7/100 (7.0%), assert_close 85/165 (51.5%) and 11/100 (11.0%),
allclose 1e-2 48/165 (29.1%) and 16/100 (16.0%); METR truestar 8/38 (21.1%) and 1/13 (7.7%),
assert_close 20/38 (52.6%) and 4/13 (30.8%), allclose 1e-2 7/38 (18.4%) and 4/13 (30.8%). So 45 of
truestar's 54 KB false fails under the full policy are the NaN handling and nondeterminism
verdicts, not wrong numbers. Example input only (numerics): KB truestar 2/165 (1.2%) and 13/100
(13.0%), allclose 1e-2 0/165 and 25/100 (25.0%); METR truestar 0/38 and 3/13, allclose 1e-2 0/38
and 8/13. Strict rule, task scope: KB truestar 17/173 (9.8%) and 7/92 (7.6%), allclose 1e-2 6/173
(3.5%) and 11/92 (12.0%); METR truestar 4/41 (9.8%) and 1/10, allclose 1e-2 2/41 (4.9%) and 1/10.
Published labels and every scope: `summary_rerun.txt`.

The 10 remaining false fails: 3 argmax or argmin kernels that skip NaN (KB 2, METR 1; an integer
output, so the run gave TS200; the check now tests the maximumNumber match on integer outputs too
and gives TS202, a change made after the run and not measured on it), the masked cumsum of a
constant (3.1x, two kernels: the mask breaks the identical terms test, the case the running bound
below would cover), Conv3d softmax mean on the offset family (11.8x), instance norm at x / 128
(2.3x), a sum of a constant after a sigmoid (1.1x) and one element of 8.9 million at 1.0x (two
kernels).

The lean verdict fired on 3 kernels: KB level 2 o1 task 41 (a tanh GELU, wrong) and METR level 1
tasks 2 and 18 (TF32 matmuls, precision shedding). Its false fail rise on the main table is 0
points on both sets, inside the 1 point rule, so it ships on by default (`--no-lean` turns it off).
Counting precision shedding as correct, it adds 2 METR false fails (2.4% to 7.3%), both TF32: the
lean test sees TF32 operand rounding. On the self authored set (`self_set_rerun.txt`, CPU, 3
draws) it removes all 5 false passes (the float16 scale of 1e-3 and the bfloat16 tail tile scaled
by 1 + 2^-7) and adds no false fail; the Triton GPU cases were not rerun with it.

### Rerun: bounds (numerics, adjudicated labels, task scope, lean on)

1. False fail at most 5%: **failed on KB** (5.5%, 9 of 165; was 35.4%), **held on METR** (2.6%,
   1 of 38). Without the 2 KB argmax kernels that the post run change reclassifies it would be
   4.2%, but that is not measured.
2. False pass at most 15%: **held** (KB 7.0%, METR 7.7%). Without the lean verdict METR is 23.1%
   (3 of 13) and would fail.
3. YAKK named METR kernels: **held for the 2 that ran again** (level 1 tasks 14 and 100, rejected);
   tasks 33 (out of memory), 24 and 42 (budget) were not rerun. All 4 that ran in the first run
   were rejected there.
4. Better than a baseline: **better than all three on both sets** (false fail plus false pass:
   KB 12.5 against assert_close 44.7, allclose 1e-2 22.6 with a false fail 1.9 points higher,
   allclose 1e-4 35.6; METR 10.3 against 70.3, 36.1, 57.1). Without the lean verdict, KB against
   allclose 1e-2 is 13.5 against 22.6, 9.1 points: no measured advantage there.
5. Self authored small faults, no false pass: **held** with the lean verdict (0 of 78 wrong runs
   pass; without it the bfloat16 tail tile scale passes 3 of 3).

### Third round: completion, the precision verdict, the running bound (2026-11-25)

Three steps, each table at task scope, numerics policy, adjudicated labels (`summary_step1.txt`,
`summary_step2.txt`, `summary_step3.txt` in `results/heldout_runs/`; every other scope, policy and
label set is in those files).

A harness bug found on the way, fixed in all three tables: when the float64 reference ran out of
GPU memory on a variant, the check read it as "the reference rejects this input" and skipped the
variant for every method, so the kernel could pass (KB level 2 r1 task 13, a wrong kernel, passed
that way). The check now raises on out of memory (reference or candidate), and the tables drop
the 19 earlier results that had such a skip (KB 15, METR 4). That is why the step 1 numbers differ
from the rerun section above.

**Step 1: the rest of METR and the out of memory kernels** (results arrived after the last rerun
round; check as in the rerun, lean verdict as TS1300). Done: KB 251, METR 126.

| set | method | false fail | false pass |
|---|---|---|---|
| KB | truestar | 9/154 (5.8%) | 5/97 (5.2%) |
| KB | assert_close | 53/154 (34.4%) | 10/97 (10.3%) |
| KB | allclose 1e-2 | 5/154 (3.2%) | 17/97 (17.5%) |
| METR | truestar | 5/84 (6.0%) | 1/42 (2.4%) |
| METR | assert_close | 32/84 (38.1%) | 3/42 (7.1%) |
| METR | allclose 1e-2 | 6/84 (7.1%) | 9/42 (21.4%) |

**Step 2: TS1301, reduced precision.** When the lean verdict fires on a float32 output and the
candidate's source asks for TF32 matmuls or float16 or bfloat16 compute, the verdict is TS1301
(suspect, a precision verdict), not TS1300. On the held out runs the same source scan is applied
to the recorded TS1300 results. TS1301: 2 kernels (METR level 1 tasks 2 and 18, TF32, precision
shedding); TS1300: 1 (KB level 2 o1 task 41, a tanh GELU, wrong). KB is unchanged; METR truestar
false fail stays 5/84 and false pass goes from 1/42 to 3/42 (7.1%), since precision shedding counts
as wrong in this table. Counting it correct, METR false fail with the lean verdict goes from 8/88
(9.1%) to 6/88 (6.8%), the same as without it: the lean verdict no longer costs a false fail.

**Step 3: the running error bound.** Every sum, mean, matmul partial sum and cumsum now gets
two tiers. Rigorous: the running error bound u sum_k |s_k| (Wilkinson; Higham sec. 3.3), sound with
no assumption on the data. Probabilistic (Higham and Mary 2019): about u sqrt(sum_k s_k^2), used
only where a cheap test says its assumption can hold, namely at least two nonzero terms whose spread
is at least one ulp of the largest running sum, so each step's discarded bits are spread over the
ulp. This replaces the identical terms rule; on CPU it also passes the masked cumsum of a constant
the old rule failed (2.9x, now 0.13x). The lean test skips an example input holding a constant
tensor. The kernels that could change were rerun on Modal L4: every one with a failing example or
value family variant, the 15 never run, and the out of memory kernels one per card (21 of 82 tried
within the budget: 13 done, 8 out of memory again). A kernel that passed every task variant before cannot fail now: the bound only grows, and
the lean test only runs less. Done: KB 277, METR 126.

| set | method | false fail | false pass |
|---|---|---|---|
| KB | truestar | 6/168 (3.6%) | 7/109 (6.4%) |
| KB | truestar without the lean verdict | 6/168 (3.6%) | 8/109 (7.3%) |
| KB | assert_close | 59/168 (35.1%) | 10/109 (9.2%) |
| KB | allclose 1e-2 | 7/168 (4.2%) | 21/109 (19.3%) |
| KB | allclose 1e-4 | 36/168 (21.4%) | 14/109 (12.8%) |
| METR | truestar | 3/84 (3.6%) | 3/42 (7.1%) |
| METR | assert_close | 32/84 (38.1%) | 3/42 (7.1%) |
| METR | allclose 1e-2 | 6/84 (7.1%) | 9/42 (21.4%) |
| METR | allclose 1e-4 | 19/84 (22.6%) | 3/42 (7.1%) |

On the kernels measured in both step 2 and step 3, 9 correct kernels moved from rejected to passed
(the argmax and argmin that skip NaN, KB 2 and METR 2; both masked cumsums; a Frobenius norm; the
Conv3d softmax mean) and no wrong kernel moved, so the false pass count is unchanged on the same
kernels. The KB false pass rate moves from 5.2% to 6.4% only because step 3 adds kernels (the out
of memory ones that now finished). Full policy: KB truestar 53/168 (31.5%) and 5/109, METR 29/84
(34.5%) and 1/42. Strict rule: KB 14/176 (8.0%) and 7/101, METR 7/88 (8.0%) and 3/38. Self
authored set (`self_set_step3.txt`, CPU, 3 draws): 0 false fails, 0 false passes with the lean
verdict.

**Bounds after step 3** (numerics, adjudicated labels, task scope, lean on):

1. False fail at most 5%: **held** (KB 3.6%, METR 3.6%).
2. False pass at most 15%: **held** (KB 6.4%, METR 7.1%).
3. YAKK named METR kernels: **held for the 4 that ran** (level 1 tasks 14 and 100, level 2 tasks 24
   and 42, all rejected); task 33 never finished (out of memory even alone on the L4).
4. Better than every baseline on both sets: **held** (false fail plus false pass: KB 10.0 against
   assert_close 44.3, allclose 1e-2 23.5, allclose 1e-4 34.2; METR 10.7 against 45.2, 28.5, 29.7).
5. Self authored small faults, no false pass: **held**.

Still not measured: 37 kernels out of memory with four or two per card and not finished one per
card (KB 31, METR 6; 8 of them out of memory even alone), 19 whose reference ran out of memory in
an earlier run, and 17 that do not build, load or finish in 300 s. The 9 remaining false fails, all
TS200: KB group norm (5.6x) and layer norm (45.1x) on the example input (both first measured in this
step), level 2 task 16 (two kernels, one element of 8.9 million at 1.0x), instance norm at x / 128
(2.3x), a sum after a sigmoid on the constant tensor (2.6x); METR the reverse cumsum (3.2x), a fast
math mish (3.1x) and a fast math tanh at large values (17.8x).

## `@truestar.verified_autotune` (R7)

`@triton.autotune` times each config and keeps the fastest, so a config that is fast and wrong is
picked and nothing says so. `truestar.verified_autotune` takes the same `configs` and `key` plus the
reference:

```python
@truestar.verified_autotune(configs=[...], key=["M", "N", "K"], reference=lambda a, b: a @ b,
                            inputs=["a_ptr", "b_ptr"], outputs=["c_ptr"])
@triton.jit
def mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ..., BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr): ...

mm_kernel[grid](a, b, c, M, N, K, ...)
```

On the first call per key: the tolerance is derived once (float64 on the GPU); each config runs on
the real arguments with the outputs prefilled with NaN (`reset_to_zero` arguments zeroed) and is
checked; a wrong config is removed with a `VerifiedAutotuneWarning` and a TS207 reason (elements,
location, ulps); configs that fail to compile are skipped as Triton does; only the right configs are
timed with `triton.testing.do_bench` and the fastest of them is cached. If none is right the call
raises `VerifiedAutotuneError` with every reason, so a wrong answer is never returned. `.reports`
holds a `TuneReport` per key (accepted configs with times, removed ones with reasons, and whether a
right config is nondeterministic).

Measured (`scripts/bench_autotune.py`, RTX 5070 Ti, Triton 3.7.1, compile caches warm, median of 3):

| kernel | triton.autotune | verified_autotune |
|---|---|---|
| matmul without a K mask, K = 1040 (only BK = 16 divides) | picked BK = 64, output wrong, 0.66 s | removed the 4 BK = 32 and 64 configs (NaN read past the rows, every element), picked BK = 16, output right, 0.31 s |
| same kernel, K = 1024 (every config right) | 0.62 s | 0.65 s, +5.4% |
| matmul with a K mask, K = 1040 | 0.62 s | 0.65 s, +4.7% |
| row softmax, N = 3000, 5 num_warps configs | 0.52 s | 0.56 s, +8.3% |

The overhead budget was 20%. Derivation takes 2 to 6 ms, the checks 25 to 80 ms; the rest is the
timing both tools do.

Not reproduced: triton#12525 (`loop_unroll_factor` running iterations of an empty range) does not
reproduce on Triton 3.7.1 here with unroll factors 1 to 8 on empty, reversed and short ranges;
triton#11153 was not tried. The Triton interpreter is not used as a second oracle; float64 is the
oracle.

## Limits

* Shape symbols come from equal sizes. Sizes equal by coincidence (a head dim equal to a sequence
  length) move together, which can make a variant the candidate cannot run (reported as TS201); pick
  example inputs with distinct sizes. Python int arguments are not remapped.
* Infinity is not in the default value families: PyTorch's own float32 kernels disagree with float64
  there (CPU float32 GELU(+inf) is NaN, float64 gives inf), so a candidate that copies PyTorch would
  be flagged.
* NaN poisoning works on CUDA only; on the CPU an unwritten element shows only when its stale value
  is wrong.
* Op blame sees ATen ops only. Triton and CUDA launches are not ATen ops; for them the element
  location is the blame.
* Nondeterminism is reported (TS203, suspect) but not separated from a wrong value when both happen.
* `verified_autotune` checks each config on the arguments of the first call per key; a config right
  there and wrong at another size within the same key is not caught. Key on every size that a block
  size must divide.
