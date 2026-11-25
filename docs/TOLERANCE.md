# Derived tolerances (`truestar.assert_close`)

Pain R6 (research round 3): every test suite picks `atol` and `rtol` by hand. Too tight and it fails
correct code that sums in another order or uses TF32; too loose and it passes broken kernels.
`torch.testing.assert_close` has per dtype defaults that know nothing about the op (a 4096 long
reduction and an elementwise add get the same tolerance) or about the device policy.

```python
import truestar

truestar.assert_close(lambda x: torch.softmax(x, -1), (x,), my_softmax(x))   # no rtol, no atol
rep = truestar.close_report(my_ref, (a, b), out)    # the same check, returning the report
print(rep.ok, rep.message())
```

With pytest, `pytest_plugins = ["truestar.pytest_plugin"]` gives the fixtures
`assert_derived_close` and `derived_close_report`.

## Method

The question a tolerance answers is how far from exact a correct implementation of `fn` can be on
these inputs in this precision. Truestar answers it per element, from `fn` itself
(`src/truestar/tolerance.py`):

1. **Reference.** `fn` in float64 on the inputs widened exactly, on the device of the output (a CUDA
   output gets its float64 reference on the GPU).
2. **Rounded run.** `fn` in float64 with every op result rounded to the working dtype, and the
   operands of matmuls and convolutions rounded to TF32 where the policy allows TF32. This is the
   run the backward check already uses (`gradcheck._rounding_mode`): what a correct kernel that
   rounds once per op computes.
3. **Error model runs** (8 by default). `fn` in float64 where each op result gets a random error of
   the size a correct kernel can make, propagated through the rest of `fn`:
   * arithmetic: one unit roundoff of the result, and none where the result of add, sub, mul, div,
     sqrt or square is representable in the working dtype (IEEE correctly rounds these, so a
     correct kernel returns such a result exactly: constant arithmetic such as the exponent
     `2 - 1` of a pow gradient stays exact; before 2026-11-25 it got noise, `pow(x < 0, 1 + noise)`
     was NaN and every error model run of a Llama step's gradients was NaN, so their sigma came out
     0 and correct steps were refused); transcendental functions: four (fast GPU math);
     `x * g(x)` activations (GELU, SiLU and others) also four unit roundoffs of `|x|`, since their
     usual formulas cancel where the output is tiny (PyTorch float32 GELU on the CPU is 3.7 u |x| off
     at x = -3.3);
   * reductions (matmul, bmm, addmm, sum, mean): u_acc times the running sums of a sequential order,
     checkpointed every 32 terms plus that chunk's absolute sum. That follows the data: about
     sqrt(K) times tighter than the classic `sqrt(K) u sum|x y|` on random signs, and as large as it
     when the running sum grows and then cancels. A tiled kernel holds the same running sums at its
     tile boundaries;
   * TF32 operands: `u_tf32 sqrt(sum (x y)^2)`; cuBLAS reduced precision split K (allowed by
     `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction`, on by default):
     `2 u sqrt(sum (x y)^2)`, measured on cuBLAS float16 GEMMs at K = 2048 and 8192, where outputs
     that cancel to near zero are thousands of ulps off float64;
   * convolutions, softmax, layer and group norm, variance: their own rules (the norms carry the
     mean's error times |mean| / std, so a two pass or Welford kernel passes and a one pass variance
     with a large mean does not);
   * data movement, selection, max and ReLU: exact.
   Each op's error size is computed once and reused by the other runs. The spread over the runs is a
   standard deviation `sigma` per element.

**Verdict.** An element passes when

    |out - ref| <= lambda sigma + |rounded - ref| + 1 ulp + 1e-3 u rms(ref)

where lambda is derived from the sample count and the output's size (`element_lambda`, 2026-11-25).
`sigma` is estimated from S = 8 zero mean runs, so `sigma_hat^2 / sigma^2` is `chi^2(S) / S` and a
correct element's error over `sigma_hat` is Student t with S degrees of freedom, whose tail is far
heavier than a normal one: `P(|t_8| > 8) = 4.4e-5`, so a million correct elements have about 40
over 8 sigma_hat. lambda is the two sided t quantile at `1e-3 / elements` (a correct output fails
with probability at most 1e-3, Bonferroni over its elements), never below 8: 8 up to about 40
elements, 13.3 at 1000, 19.0 at 16384, 31.0 at 786432, 35.1 at 2097152 (S = 8). Measured need: the
fair benchmark's encoder layer step had 2 of 786432 elements of `self_attn.in_proj_weight`'s
moments at 10 sigma_hat under the fixed 8, and the independent float32 evaluation (the reference
interpreter run by torch's float32 kernels) was over 8 at the same two elements by the same amount,
while every op's own float32 error was within 1.2 sigma of its rule (each op run once in float32
from rounded float64 inputs). The kernel check's verdict (`kernel/verify.py`) uses the same lambda.

and the output passes when its RMS error is at most 2 times the RMS of `sigma + ulp/2 + |rounded -
ref|`. So the rounded run itself always passes, a near tie the rounded run flips is allowed, and a
broad accuracy loss (float16 accumulation, TF32 where IEEE float32 was required) fails even when each
element alone is small. NaN and infinity must sit exactly where float64 has them, except where the
working dtype itself overflows in the rounded run (then a finite and an infinite answer both pass).

**Policy** (`truestar.tolerance.policy_for`): float16 and bfloat16 accumulate in float32; TF32
follows torch's flags on CUDA and is off on the CPU; `tf32=` and `accumulate=` override.

**Failure messages** name the elements over their bound, the worst one with its index, value and
float64 value, the error in ulps of the working dtype, whether it looks like reordering noise, where
the bad elements sit (a tail or head of a dim, a narrow index range, one index, a diagonal offset),
and a policy that would explain it when one does (float16 accumulation, TF32). Example:

```
output: 743 of 12288 elements beyond the derived bound; worst at [105, 34]: 0.347656 vs float64
0.432397 (error 0.301, 10.6x its bound, 16253 ulp of float16); ... Not reordering noise. The error is
consistent with float16 accumulation (the policy accumulates in float32).
```

**Efficiency.** The float64 runs are cached per input signature (16 entries). An entry holds the
very `fn` and input tensors it was derived for, so their memory cannot be reused by other tensors
while cached; an in place write bumps the version and misses. `truestar kernel` and
`verified_autotune` derive once and check many outputs against it.

## Measured (2026-11-24, `scripts/bench_tolerance.py --draws 5 --cuda`)

`truestar.kernelbench`: 36 CPU kernels whose bug shows on the example input (matmul, attention,
softmax, layer norm, row sums, elementwise; correct ones in other orders or policies, wrong ones with
tails, mask edges, float16 accumulation, racy lost updates, wrong epsilon, one pass variance, wrong
constants) plus 2 correct CUDA GEMMs (cuBLAS float16, float32 under TF32 flags), 5 input draws each:
105 correct runs, 85 wrong runs.

| checker | false fail | false pass |
|---|---|---|
| `truestar.assert_close` | 0.0% (0) | 5.9% (5) |
| `torch.testing.assert_close`, reference in the same dtype | 32.4% (34) | 0.0% (0) |
| `torch.testing.assert_close`, float64 reference cast to the dtype | 23.8% (25) | 0.0% (0) |
| `allclose(rtol=1e-2, atol=1e-2)` (KernelBench) | 0.0% (0) | 28.2% (24) |

Correct candidates come closest to their bound at 0.46 of it per element and 0.40 of the RMS limit
(the TF32 GEMM). Median time per check 0.017 s, worst 1.1 s (attention at S=192 on CPU).

The 5 false passes are all `mm_scale_1e-3_fp16`, a constant off by 1e-3 in float16: one to two
float16 ulps, which is within what a correct float16 kernel that rounds twice shows. It is below the
resolution of the dtype by design. The torch default catches it only because it compares with a
reference computed by the same kernel order; the same setting false fails 32% of correct kernels
(blocked and split K float32 GEMMs, attention, an offset layer norm).

## `truestar check` scoring (whole tensor, element and slice)

`truestar check` does not run the derived tolerance above on every module: it would cost one float64
error model derivation per module and backend. It scores each module output against the float64
reference with two measures (`src/truestar/compare.py`):

* **Aggregate:** `||a - b|| / ||b||` over the tensor, against the module's threshold. The threshold
  is a fixed floor per precision (float32 1e-4, TF32 5e-3 for modules that run a matmul or a
  convolution, float16 5e-3, bfloat16 4e-2), raised to 10x the error a correct CPU run shows on the
  same module replayed alone. It is not derived per op. Divergent above
  `min(20 x threshold, max(0.1, 2 x threshold))`.
* **Local:** the worst element and the worst slice. Element i is measured as
  `|a_i - b_i| / max(|b_i|, rms(b))`. A slice is every index of every dim (a row, a column, a
  channel, a batch item); a slice of n elements is measured as its error norm over the norm of its
  reference values, each floored at `rms(b) * max(1/32, 1/sqrt(n))`. So a slice of small values is
  judged on its own scale once it holds enough elements for rounding noise to average out, and an
  element that cancelled to near zero in exact math is judged on the tensor's scale. The local limit
  is 8 x the threshold; divergent above `min(20 x limit, max(0.5, 2 x limit))`. The report names
  the slice: `own error 1.00e+00 at index 517 of dim 0 (1024 elements) vs local limit 3.2e-01
  (whole tensor 3.13e-02)`; the agent JSON has `local_err`, `local_limit`, `local_at` and
  `localized` on the blamed module.

Why both: one wrong row in 1024 is 1/sqrt(1024) = 0.031 aggregate, under the bfloat16 floor of 0.04,
and was reported OK before 2026-11-24 (float16: only suspect). Its local error is 1.

**Measured (2026-11-24, `scripts/bench_localized.py`, then `--old` for the scoring before).** Clean
set: 11 models (resnet18, mobilenet_v3_small, efficientnet_b0, convnext_tiny, vit_b_16, two small
transformers, the MLP, the two head model, a conv net with in place activations, a 1024 x 1024
layer with 32 small value channels) in float32, float16 and bfloat16 on CPU and CUDA (RTX 5070 Ti,
torch 2.11): 66 runs. Fault set: one row in 1024, one element, the last column and the small value
region of the 1024 x 1024 layer (completely wrong in float16 and bfloat16, 5% wrong in float32),
plus the existing scale, zero, NaN and rows faults on the MLP, each on CPU and CUDA: 50 runs.

| scoring | false fail (clean) | false pass (faults) |
|---|---|---|
| aggregate only (before) | 6 of 66 | 24 of 50 (every localized fault: 14 suspect at the right module, 4 suspect with no module blamed, 6 OK) |
| aggregate and local (now) | 6 of 66 | 0 of 50 |

The 6 clean failures are the same runs under both scorings: mobilenet_v3_small, efficientnet_b0
and convnext_tiny with random weights in float16, on both backends. Their activations shrink to
about 1e-7 at depth, below float16's smallest normal (6.1e-5), so they lose precision in float16 on
every device; the check reports that correctly and the local measure changes nothing there. On the
other 60 clean runs the largest local error is 0.14 of its local limit (vit_b_16, float16, CPU) and
the largest aggregate error 0.28 of its threshold.

Limits of the local measure: a slice is a full index of one dim, so a wrong tile (a block of rows
and columns) of small values in a large low precision tensor is seen only by the element measure,
on the tensor's scale; values below `rms / 32` are judged on that floor, so a bfloat16 region at
1/1000 of the tensor's scale can still be wrong unseen. Gradients (`truestar check --backward`) keep
the aggregate measure only (docs/GRAD_CHECK.md).

## Cross check against FPTaylor (2026-11-25)

`scripts/bound_crosscheck.py` checks the derived bounds above against FPTaylor (MIT,
github.com/soarlab/FPTaylor), a tool that computes a sound bound of |kernel - exact| over a box
of inputs. The corpus: exp, log, tanh, erf, rsqrt, sigmoid, SiLU, softplus, three GELU forms,
1 - exp(x) and -expm1(x), layer norm on rows of 4 (centered and with a large common offset),
softmax on rows of 4, dot products of 4 and 8 terms, and one AdamW step, each in float32, float16
and bfloat16. Each formula is written once (a small expression language in the script) and turned
into both the FPTaylor input, with one rounding per op in the precision that op runs in, and the
float64 evaluation the pointwise check uses. The op model is the one Truestar states: one unit
roundoff for arithmetic and square root, `TRANSCENDENTAL_ULPS` (4) unit roundoffs for exp, log,
tanh, rsqrt, expm1 and log1p, float32 inside reductions for half types.

Per formula and output it reports Truestar's largest `assert_close` bound over a grid of the box
(about 4000 to 8000 points, corners included), FPTaylor's bound and their ratio, and two pointwise
checks against the first order worst case of the same op model (every op's error chosen against
the result, by autograd): the default 8 run bound, and a 64 run bound that removes the sampling
noise of the error model. Flags: `BELOW` (Truestar's largest bound under FPTaylor's: unsound),
`MODEL_BELOW` (the 64 run bound under the worst case at a point: a gap in the model), `OPBOUND_*`
(the same two for `truestar.opbounds` on softmax and dot products), `LOOSE` (over 100x FPTaylor),
`POINT_BELOW_SAMPLED` (informational, see below). Where FPTaylor fails or warns that its second
order term dominates (interval overestimation through exp and division), the box is bisected and
the max over the pieces kept. FPTaylor has no erf, so erf and erf GELU get the pointwise checks only.

`tests/test_bound_crosscheck.py` runs the comparison against cached FPTaylor results
(`tests/data/bound_crosscheck_fptaylor.json`, keyed by the hash of each FPTaylor input, so a change
to the corpus or the op model shows as stale) and needs no FPTaylor. Regenerate with
`--fptaylor <path> --write-fixture tests/data/bound_crosscheck_fptaylor.json`.

FPTaylor's bound is its optimizer's upper bound; it also reports a lower bound (an error its model
reaches in the box). A Truestar bound between the two is `*_FPT_GAP` (inconclusive, the gap is
FPTaylor's own slack), under both is a definite `BELOW`.

**Measured (2026-11-25, FPTaylor 0.9.4+dev at b5a77ca, OCaml 4.14.2, CPU server; about 50 min,
softmax in float32 alone 44 min over 13 box pieces).** 92 rows (formula, dtype, output):

| result | rows |
|---|---|
| Truestar `assert_close` bound under FPTaylor's sound bound (`BELOW`, `REL_BELOW`) | 0 (the planted 1 - exp(x) against -expm1(x) rows are, as they must be) |
| ratio Truestar / FPTaylor | 3.8 to 104; elementwise 16 to 76, layer norm and softmax 9 to 18 in float32 and 48 to 104 in half types, dot products 11 to 24, AdamW 8 to 12 |
| over 100x (`LOOSE`) | 4: softmax outputs 0 and 3 in float16 and bfloat16 (101x to 104x) |
| 64 run bound under the pointwise first order worst case (`MODEL_BELOW`) | 0; lowest ratio 1.79 (AdamW parameter, float16) |
| 8 run bound under it at some grid points (`POINT_BELOW_SAMPLED`) | 7 rows, at most 4.9e-4 of the points, lowest ratio 0.81 |
| `truestar.opbounds` under FPTaylor or the pointwise worst case | 0 after the fix below; softmax float32 sits between FPTaylor's lower and upper bound (0.77x to 0.95x of the upper) and equals the first order worst case pointwise |
| FPTaylor could not bound | AdamW parameter update in float16 and bfloat16 (its Taylor form overflows, `num_of_float: inf`, on every piece) |

Findings:

* **Fixed, `opbounds.bound_softmax`:** it counted one exp error at `exp_ulps u` where the numerator
  and the denominator each carry `2 exp_ulps u` (one ulp is up to 2 u, as `bound_elementwise`
  has it). The first order worst case on a row of 4 was 1.4x over the old bound. Now
  `y (2 c' (1 - y) + gamma_{n-1} + u)` with `c' = 2 exp_ulps u`, equal to the first order worst
  case at every grid point (test: `tests/test_opbounds.py::test_softmax_bound_covers_both_exp_errors`).
* **Sampling, not a model gap:** the per element sigma is an RMS of 8 random error model runs, so
  at about 1 in 4000 to 1 in 2000 elements of a composed formula (sigmoid GELU, layer norm with an
  offset, the AdamW first moment) it falls low enough that the bound is under the worst case its
  own op model allows (0.81x at the lowest). With 64 runs every point clears it by 1.8x or more.
  Left as is: removing it costs 8x the derivation time, and a correct kernel would have to sit at
  its worst case on exactly those elements.
* **Loose in half types:** softmax and layer norm in float16 and bfloat16 are 48x to 104x over
  FPTaylor. The rules charge `TRANSCENDENTAL_ULPS` working dtype unit roundoffs per element, while
  the policy (and PyTorch) computes them in float32 and rounds once to the half type. Charging the
  transcendental part at `u_acc` plus one `u` of output rounding would follow the policy; not
  changed here because it shifts the measured false fail rates above and needs that benchmark
  rerun.
* The interval comparison alone cannot see a cancelling kernel judged by the bound of a stable one
  (1 - exp(x) against -expm1(x) is only 3.8x on absolute error); the relative and pointwise
  checks flag it.

## Operator local bounds for comparing two stacks (2026-11-24)

`truestar.opbounds` answers a narrower question than `assert_close`: two stacks that are not bitwise
reproducible (two engines, two GPUs, two batch sizes) ran the same operator on the same inputs and
got different bits; is that rounding? Each operator is bounded on its own inputs, with no
propagation from earlier ones (research item 1.6; per operator bounds as in arXiv 2510.16028):

* worst case: `gamma_n u` times the sum of absolute terms, valid for any summation order;
* probable: Higham and Mary's probabilistic bound (SIAM J. Sci. Comput. 2019), which grows with
  `lam sqrt(n) u` instead of `n u`, with `lam` chosen from a stated confidence (default 0.999). It
  assumes independent, mean zero rounding errors, so it is labelled probable, never proven.

`bound_matmul` (with bias), `bound_sum` (and mean), `bound_softmax` (with the exp ulp bound) and
`bound_elementwise` (a stated ulp bound) return both per element; `compare(op, bound, a, b)` judges
`|a - b|` against twice each (both stacks within it of exact): `rounding`, `look_closer` (TS1098,
above probable, within worst case) or `not_rounding` (TS1099, above worst case). The accumulation
precision (`acc`) is declared by the caller because it is a property of the stack.

Measured (tests/test_opbounds.py, CPU): a float32 matmul (64 x 2048 x 48) summed in two orders is
`rounding` at 0.0032x of the pairwise probable bound and 0.0004x of the worst case (the probable
bound is 8x tighter there, and both are far above what two correct orders actually differ by). A 1% of the absolute sum fault on one element is `not_rounding`. A
stack with a float16 running accumulator compared to exact, under a declared float32 accumulation,
is flagged. 2000 reordered float32 sums of 4096 terms: none above the probable bound at 0.999.

Comparison only: nothing here changes a model's run, and the worst case bound stays the verdict that
counts. Not covered: layer norm, attention and convolution (they need their own rules), and inputs
that differ between the stacks (then the bounds do not apply).

## The compiled executor's first call checks (2026-11-25, P64)

The first call of a compiled program (`truestar.compiler`) is checked three ways against float64:
each kernel against the kernel IR's float64 reference, each region (the kernels covering a group of
IR ops) against those ops in float64, and the whole program end to end. Each check first applies a
fixed bar: `BASE_TOL` of the dtype for a kernel and a region (float32: 2e-5, relative to the
output's largest value), and `BASE_TOL x 4 x sqrt(kernels)` end to end. The bar knows nothing about
the computation, so it was both too tight and too loose: it refused correct float32 training steps
whose outputs are ill conditioned (11 of the fair benchmark's 12 training configs), and it passed
errors far larger than a correct step's (a wrong weight decay sign: 2.0e-5 off where the correct step
is 6.5e-8 off, both within the end to end bar of 5.3e-4).

**Over the bar, the derived bound decides.** An output over its fixed bar is judged again by the
bound this page derives (Method above), from the computation the check covers:

* **End to end** (`CompiledProgram._derived_verdict`): `fn` is the reference interpreter running
  the whole program in float64; the rounded run rounds every op result to the working dtype; the
  error model runs give every op result the error a correct kernel can make there, propagated
  through the rest of the program (through the backward and the optimizer of a training step).
  The output passes within `lambda sigma + |rounded - ref| + 1 ulp + 1e-3 u rms(ref)` per element and
  its RMS limit. Accepted outputs are listed in `cp.derived_accepts`.
* **Region** (`GroupCheck._within_derived`): the same, with `fn` the region's own IR ops from the
  region's inputs.
* **Kernel** (`kernel/verify.py check_outputs`, `kernel/compiler.py _derived_runs`): the kernel IR
  expression in float64 with every node rounded to float32 (`reference.RoundedEvaluator`), and 8
  error model runs with every node's result given a uniform relative error of one unit roundoff,
  four for transcendental functions (`reference.ErrorModelEvaluator`); the same verdict. Only for
  kernels that compute in float32 (float32, float16 and bfloat16 kernels: their math is float32); a
  float64 kernel keeps its fixed tolerance.

* **Cut at verified outputs** (`CompiledProgram._cut_verdict`). An output over its derived bound
  that is computed from other outputs which passed theirs is judged again from them: the part of
  the program between the program inputs, those outputs and this one runs in float64 with the
  compiled values of those outputs in place of their float64 ones, and the output must be within
  the bound derived for that part. The case it exists for: AdamW's first step moves each weight by
  about `-lr sign(g)`, so where a gradient entry is within its own error of 0 a correct execution
  may flip the update (2 lr). The error model runs draw normal errors around float64 and see such a
  flip only as often as it happens (5% per run for an entry 1.6 sigma from 0), so 8 runs leave its
  bound far too small: on the fair benchmark's resnet18 AdamW step the independent float32
  evaluation itself flipped one weight of `layer3.0.conv2` (10x its bound) and the compiled step
  flipped six, every one with a gradient (`exp_avg`, an output) within its derived bound. Given
  the compiled `exp_avg` and `exp_avg_sq`, the update is determined and the flip is checked
  exactly. Sound by composition: the moments are right by their bounds, the weight is what the
  program computes from them by this one. A wrong optimizer formula is in the part judged here
  (wrong decay sign, missing bias correction: still refused) and a wrong gradient fails at the
  moments. Accepted outputs are listed in `cp.derived_accepts` with `via_outputs`.
* **Where it runs.** On the outputs' CUDA device; when that runs out of memory, once more without
  keeping every op's sigma between runs (`derive(memo=False)`), then on the CPU. When no device can
  hold it, `cp.derived_unavailable` is set and the fixed bar is the verdict. Cost, measured as the
  whole first call (compile, launch tuning and every check, the derivation included) on an RTX 5070
  Ti, whose float64 runs at 1/64 rate: MLP 65 to 70 s, encoder layer 24 to 27 s, Llama style step
  50 to 54 s, resnet18 batch 32 1326 to 1339 s (its conv weight gradients sum 100352 terms each and
  the running sum rule walks them in chunks of 32; the derivation alone took 805 s in a separate
  run). Once, at the first call. The resnet18 derivation does not fit in 16 GB with the memo and
  falls back to running without it; its CPU fallback held about 20 GB of host memory.

Why this is a derivation and not a looser bar: a correct execution in the working precision rounds
every op result once (the rounded run) and its kernels may differ from correct rounding by the
error model's sizes; the bound is how far such executions are from float64 on these inputs. Where a
formula cancels (`1 - exp(x)` for a tiny x, AdamW's division by `sqrt(v) + eps` on its first step,
a momentum buffer of a gradient whose terms cancel) every correct float32 execution is far from
float64, the error model runs are too, and the bound is wide; where nothing cancels the bound is a
few ulps, tighter than the fixed bar. An output within the fixed bar is accepted as before (the
derivation runs only for outputs over it, so passing programs pay nothing).

**Strict for training loops.** `compile_program(..., end_to_end=...)`: `"bar"` is the rule above;
`"derived"` judges every float output by its derived bound alone, also one within the fixed bar; the
default `"auto"` is `"derived"` for a call that writes the caller's state in place (`update=` or
`donate=True`, as a training loop calls its step, the fair benchmark and `CompiledTrainStep`
included) and `"bar"` otherwise. Such a call's error is carried into every later step, and the
fixed bar is loose enough to pass a wrong optimizer (below), so a training step pays for the
derivation (about ten float64 runs of the program, once, at its first call) to be judged by its own
bound. The bar is kept for other programs only as a cheap first pass.

**Still caught** (`tests/test_trainverify.py`, a small MLP step, the wrong step's results put in
place of the compiled step's results at the first call `cp(inputs, update=upd)`, as a code
generator bug would make them; every one is refused by the default and passes when the results are
right): AdamW with its decoupled weight decay of the wrong sign (16333 of 16384 elements over their bound, worst 33x, RMS 106x its limit, in
`fc1.weight_new`; 2.0e-5 off, within the fixed bar of 5.3e-4, so `"bar"` alone passes
it), AdamW without `1 - b1^t`, without `sqrt(1 - b2^t)`, every gradient scaled by n / (n - 1) under
AdamW and under SGD, and every gradient rounded through float16 and back (a wrong cast) under AdamW
and under SGD (within the fixed bar too). A kernel computing `sqrt(exp(x) - 1)` where
`sqrt(1 - exp(-x))` was compiled is refused by the kernel check; the hand written `sqrt(1 - exp(x))`
itself passes and is not called a code generator bug.

Measured on the fair benchmark's 12 training configs and a Llama style full fine tune step: results/pc-trainverify/README.md.

## Limits

* `fn` is run in float64 and must mean the same function there. A function whose meaning depends on
  the input dtype (an explicit float16 cast inside it) is checked against what it computes in
  float64.
* Ops without an error rule are taken as four ulps elementwise, or as a plain reduction when they
  reduce; the report lists them (`notes`). Convolutions keep the classic `sqrt(K) u sum|x w|` bound
  (no running sum rule yet), so they are less sensitive than matmuls.
* The model is per op. A fused kernel that keeps wider intermediates is more accurate and passes; a
  kernel that is less accurate than one rounding per op (for example a float16 accumulator over
  more steps than split K uses) fails, which is the intended policy. A float16 accumulator over few
  steps (16 at K = 512) is within what PyTorch's default reduced precision reduction policy permits
  cuBLAS; set that flag to False to check the strict policy.
* Random sampling of the error model: 8 runs per derivation. The per element tolerance includes the
  rounded run and a ulp, and no correct kernel in the benchmark came within half of its bound.
