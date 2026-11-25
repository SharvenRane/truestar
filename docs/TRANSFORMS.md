# vmap and forward mode as program transformations

Author: Sharven Rane. First written 2026-11-24.

Pain row: P24 (docs/QUALITY_BAR.md). Code: `src/truestar/ir/vmap.py`, `src/truestar/ir/forward.py`,
`src/truestar/ir/transform_util.py`. Tests: `tests/test_vmap.py` (318), `tests/test_forward.py` (115).
Measurements: `scripts/transforms_compare.py`.

Every transform takes an IR program (or a `@truestar.function`) and returns a new IR program. Nothing
is traced, no batched tensor type or dual number exists at run time, and the result runs on the
reference interpreter or the compiler like any other program, for every batch size it declares.

## Use

Before (torch.func):

```python
from torch.func import vmap, grad, jacfwd, hessian, functional_call
per_sample = vmap(grad(loss_fn), in_dims=(None, 0, 0))(params, x, y)
vmap(f)(xs)                    # f has `if x.sum() > 0:` -> RuntimeError: data dependent control flow
vmap(dropout_fn)(xs)           # RuntimeError: random operation in randomness error mode
hessian(layer_loss)(x)         # TransformerEncoderLayer on the CPU -> NotImplementedError, no forward AD
```

After (Truestar):

```python
from truestar import ir

prog = ir.import_module(loss_module, (x[0], y[0]))                 # or a @truestar.function
per_sample = ir.vmap(ir.grad(prog, wrt=param_names), in_axes={"x": 0, "y": 0})
loss, *grads = ir.run(per_sample, x, y)                            # each grad has a leading batch dim

ir.vmap(branchy)                       # Python if and while on tensor data: batched, exact
ir.vmap(sampler, in_axes=None, size=8) # random ops: every element its own stream, no flag
ir.jvp(prog, wrt=["x"])                # inputs + t_x -> outputs + J t
ir.jacfwd(prog, wrt=["x"])             # outputs + Jacobian, shape out.shape + x.shape
ir.jacrev(prog, wrt=["x"])             # same layout, by reverse mode
ir.hessian(prog, wrt=["x"])            # forward over reverse, one program
primal, linear = ir.linearize(prog, wrt=["x"])
```

### vmap

`vmap(program, in_axes=0, out_axes=0, size=None, axis_name=None)`:

* `in_axes`: an int or None for every input, a list with one entry per input, or a dict from input
  or param names to an axis. A param named in the dict becomes a new input holding one copy per
  element (an ensemble).
* `out_axes`: an int or a list per output. An output that does not depend on the batch is broadcast;
  None for an output that does depend on it is an error.
* `size`: the batch size. Default: a new symbol `V`, so one program serves every batch size.

### Forward mode

* `jvp(program, wrt, of=None)`: inputs are the original inputs then `t_<name>` per `wrt`; outputs are
  the original outputs then the tangent of each `of` output (default: every floating point output).
* `jacfwd(program, wrt, of=None)`, `jacrev(program, wrt, of=None)`: the original outputs, then one
  Jacobian per (output, wrt) pair of shape `out.shape + wrt.shape`, the torch.func layout. Sizes may
  be symbolic: the identity matrix is built inside the program from `arange` and `eq`.
* `hessian(program, wrt, of=0)`: the original outputs, then one block per pair (i, j) of `wrt`
  values, shape `wrt[i].shape + wrt[j].shape`. The output must be a scalar.
* `linearize(program, wrt, of=None)`: `(primal, linear)`. `primal` maps the inputs to the outputs and
  the residuals (every value the tangent code reads); `linear` maps residuals and tangents to output
  tangents and contains no primal work.

## Design

### vmap: one rule per op, checked after every node

A value of the new program is unbatched (kept exactly as in the original, so weights are never
copied) or batched (the original type with the batch dim in front). Each node is rewritten:

| op family | how it is batched |
|---|---|
| elementwise, comparisons, `where` | batched operands get size 1 dims after the batch dim so the original broadcasting lines up. A 0 dim operand does not widen a dtype, but once batched it would, so it is cast to the dtype the original promotion gave (`f32[3, 4] * f64[]` stays f32) |
| reductions, `softmax`, `cumsum`, `sort`, `topk`, `cat`, `select`, `slice`, `gather`, `scatter_add`, `permute`, `flip` | the dim attribute moves one to the right; a rank 0 operand is viewed as `[V, 1]` |
| `matmul`, `attention` | batch dims lifted like broadcasting |
| `linear` | fused when only x is batched; a per element weight or bias becomes `matmul` plus `add` |
| `conv2d`, pooling, `batch_norm`, `group_norm` | batch dim merged into the op's own batch dim and split after. Per element conv weights with a fixed size: one grouped conv with `groups * V` groups |
| `layer_norm`, `batch_norm` with per element weights or statistics | the verified lowerings of grad_rules.py, then batched op by op |
| `index`, `index_select`, `select_dynamic` with a batched index | one advanced index with `arange(V)` on the batch dim, then a permute where PyTorch's advanced indexing puts the index dims first |
| `cond`, unbatched predicate | both regions batched, the node stays a `cond` |
| `cond`, batched predicate | each branch batched and guarded by a `cond` on "some element takes it", then `where` per element |
| `loop` with a trip count | carried values become batched when the body makes them batched (a fixed point) |
| `loop` with a batched `while` condition | runs while any element is still going; an element that is done keeps its values (`where`), so element k's iteration j is the global iteration j |
| `custom_vjp` | both regions batched with every operand batched; autodiff sums an unbatched operand's per element gradients through the `expand` that batched it |
| `random` (keyed) | one vectorized `random_batched` op: row k is the op with k as one more stream, see Randomness |
| `multinomial` (random) | always batched, see Randomness |
| anything a rule cannot vectorize (per element weight of `group_norm` or `rms_norm`, per element index of `index_copy`, per element conv weight with a symbolic size) | an exact per element `loop` over the batch: `select_dynamic` in, `index_copy` out |

After each node the pass checks that every batched result has the original type with the batch dim
in front and every unbatched result the original type. A wrong rule cannot produce a mistyped
program; the attention rule was fixed this way during development (with only the mask batched, the
output batch comes from q, k and v alone).

Every IR op is classified (`batching_table()`); a test fails when a new op is added without a rule.
An unclassified op is refused by name. `UNDEFINED` is the place for an op whose per element meaning
does not exist; it is empty, because every IR op has static types, so the per element loop is always
a correct fallback. The cases torch.func calls undefined (data dependent shapes such as `nonzero`)
cannot be written in the IR in the first place.

**Randomness.** A random op inside vmap is always batched, even when its operands are not, and
every element gets its own stream, reproducible one element at a time. There is no `randomness=`
flag to choose. To share a draw across the batch, draw it outside vmap and pass it in.

* The keyed `random` op (truestar.rng, docs/RANDOM.md): element k draws exactly what the op draws
  alone with **k appended to its streams**, the key `fold(K, k) = mix(mix(K) ^ k)` of the counter
  scheme. That is the rule the scheme already uses for the index of an enclosing loop, so vmap of a
  random function equals a loop over the batch by definition. With an unbatched key (the usual case:
  the program's `rng_key` input) it becomes one new op, `random_batched`, whose row k is
  `random(key, *streams, k)`; `truestar.rng.generate_batch` computes all per row keys with one `mix`
  over `arange` and every element in one vectorized pass, on the reference interpreter and in the
  compiled step, on the CPU and CUDA. It is a new op name, not a new attribute of `random`, so a
  consumer that does not know it fails on its name instead of drawing one shared stream. A per
  element key or stream, and vmap of vmap, use the exact per element loop with the same rule (the
  loop index appended as the stream; vmap of vmap appends the outer index, then the inner).
* `multinomial` (its own counter based generator): element k draws what the op draws alone with its
  run time offset increased by `k * 2**40` (offset 0 when it has none), so elements never share
  draws for offsets below 2**40.

**Branches no element takes are not run.** JAX turns a batched `cond` into a select that always runs
both branches ([jax.lax.cond docs](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html),
read 2026-11-24), which costs the work of both and can even fail to terminate when a branch contains
a loop that only ends for its own inputs ([jax#2947](https://github.com/jax-ml/jax/issues/2947),
[jax#8409](https://github.com/jax-ml/jax/issues/8409)). Truestar guards each side with "some element
takes it". A batch that agrees runs one branch; `test_a_branch_no_element_takes_is_not_run` is the
jax#2947 program. A batch that disagrees still runs each branch on all elements, like JAX.

### Forward mode: independent rules, two checks

Each forward rule is written from the derivative, not derived from the reverse rule, so the two can
check each other. Linear ops (sums, reshapes, slices, gathers, scatters, `pad` with value 0,
`cumsum`, `triu`, average pooling) replay the op on the tangent. Composite ops reuse the verified
lowerings. `cond` and `loop` get new nodes whose regions are the forward mode of the original
regions; a loop carries each tangent as one more carried value (a fixed point decides which carried
values need one), so forward mode through a loop stores nothing per iteration and serves every trip
count. A `custom_vjp` node keeps the user's backward for reverse mode; its tangent is the forward
mode of its primal. A rule registered with `unverified=` (a straight through estimator) therefore
disagrees with the forward tangent on purpose, and the dot product test reports it.

`jacfwd` is `vmap` of `jvp` over the columns of an identity built in the program, `jacrev` is `vmap`
of `grad` over the rows, `hessian` is `jacfwd(grad(f))`, and `linearize` splits the `jvp` program at
the nodes that read a tangent.

## Op coverage additions (2026-11-24)

New forward rule: `fmod` and `remainder` (`ta - q(a / b) tb`). The ops that got a reverse rule or a
lowering with the op coverage work (docs/AUTODIFF.md) are in the rule corpus of
tests/test_autodiff.py, so each is in the forward mode finite difference and dot product test and in
the vmap per element test (`frozen` has its own vmap case). Every op of the 144 ATen ops measured
on the coverage list maps to IR ops with a batching rule and, for float results, a forward rule
(docs/MODEL_COVERAGE.md). tests/test_op_coverage.py also runs vmap (against the per element loop) and
forward mode (finite differences and the dot product test) on every imported torch.nn layer and
loss of the list.

## What is verified (2026-11-24, torch 2.11, CPU, float64)

**vmap against the per element loop.** The definition of vmap is the test: the batched program must
equal the original program run once per element. 301 cases: the 89 programs of the reverse mode rule
corpus (every differentiable op) with every input batched, with only the first input batched and with
the batch on the last axis, plus 34 cases for everything that corpus does not reach (comparisons,
integer ops, indexing with batched indices in every adjacency layout, per element weights for linear,
conv, batch norm, layer norm, group norm and RMS norm, attention with only the mask batched, `cond`
with batched and unbatched predicates, a loop whose carried value becomes batched through a capture, a
`while` loop with a different trip count per element, `custom_vjp`). Worst relative error **3.4e-16**
(bar 1e-12). Integer outputs are compared exactly.

**Forward rules against finite differences.** 102 programs (the 89 corpus programs and 13 more for
ops without a reverse rule: `log`, `sin`, `cos`, `reciprocal`, `abs`, `pow`, `maximum`, `minimum`,
`cumsum`, `triu`, `tril`, `pad`, `slice_scatter`, `select_scatter`, `cast`, `avg_pool2d`, `index`,
`select_dynamic`, `cond` on both sides, an RNN `loop`, a `while` loop, `custom_vjp`), fourth order
stencil along a random direction, h = 1e-3, bar 1e-8 (the reasoning is in tests/test_autodiff.py).
Corpus worst 6.6e-11 (`chain_two_outputs`), median 1.5e-13; overall worst 4.5e-9 (the RNN loop at
length 5, where the stencil's own truncation grows with the length; against `torch.func.jvp` the same
loop agrees exactly, error 0.0, at lengths 1, 3, 8 and 17).

**Forward rules against the transposed reverse rules.** For all 89 corpus programs, every one of the
45 ops that have both rules (42 rules and 3 lowerings): `<J t, c>` from `jvp` against `<t, J^T c>` from `grad`, random t and c.
Worst **1.3e-14** (bar 1e-12).

**Jacobian family against torch.func.** `jacfwd` and `jacrev` of an MLP with respect to the input
and a weight match `torch.func.jacfwd` and `jacrev` to 1e-12; `jacfwd` with a symbolic batch, one
program at batch 1 and 3, also 1e-12; `hessian` with two `wrt` values matches `torch.func.hessian`
block by block to 1e-12. **Hessian of a TransformerEncoderLayer** (d_model 8, 2 heads, gelu): 1.0e-15
against `torch.func.hessian`, and symmetric. On this stock layer on the CPU, `torch.func.hessian`
itself raises `NotImplementedError` (no forward AD for `_scaled_dot_product_flash_attention_for_cpu`)
and `jacrev(jacrev(...))` raises too (no double backward for it); the reference had to select the math
attention backend by hand. The Truestar program goes from 37 to 398 nodes.

**linearize.** The linear program equals the `jvp` tangents to 1e-14, is linear
(`L(2 t1 - 3 t2) = 2 L(t1) - 3 L(t2)` to 1e-13) and contains no node of the primal program.

**Keyed random ops under vmap** (added after the merge with the rng work). For uniform float32,
normal float64 and float32, bernoulli and randint: the vmapped program is one `random_batched` node,
and every row is bitwise equal to the op run alone with the batch index as an extra stream, and to
`rng.generate` with the python backend (the definition). Rows are independent: four rows of 50000
normals have off diagonal correlation below 0.02 (expected scale 0.0045), mean and standard deviation
within 0.01. A batched key and vmap of vmap keep the definition. `generate_batch` equals
`rng.generate` row by row for every CPU backend (torch, numba, python); on CUDA the reference run, the
CPU run, the Triton kernel row by row and the compiled executor on the CPU and on CUDA give the same
bits (RTX 5070 Ti, torch 2.11).

**Per sample gradients.** `vmap(grad(loss))` of an imported MLP loss matches an explicit
`torch.autograd.grad` loop and `torch.func.vmap(grad)` to 1e-12.

## Measured against torch.func (2026-11-24, torch 2.11, CPU)

`scripts/transforms_compare.py`. The machine is shared with other jobs, so times are the best of 21
runs (medians varied by up to 10x between runs). The Truestar times are the float64 reference
interpreter unless marked compiled.

**Cases torch.func.vmap rejects:**

| case | torch.func.vmap | Truestar vmap |
|---|---|---|
| `if` and `while` on tensor data | `RuntimeError`: data dependent control flow not supported | runs; equal to the loop (error 0.0) |
| a random op with default flags | `RuntimeError`: random operation in randomness error mode | runs; 4 elements give 4 distinct, individually reproducible draws |
| `torch._foreach_add` ([pytorch#200456](https://github.com/pytorch/pytorch/issues/200456)) | `RuntimeError`: batching rule not implemented, no fallback | no foreach op in the IR; the elementwise ops have rules |
| `grad` of CPU attention under vmap | runs, with the warning `There is a performance drop because we have not yet implemented the batching rule for aten::_scaled_dot_product_flash_attention_for_cpu_backward` (seen in the tests) | attention has a direct rule |
| `nonzero` (data dependent shape) | `RuntimeError`: dynamic shape | not expressible in the IR (shapes are types) |

**Speed, vmap against a Python loop:**

| workload | Truestar loop | Truestar vmap | gain | torch loop | torch.func.vmap | gain |
|---|---|---|---|---|---|---|
| per sample gradients, MLP 32, 64, 10, batch 64 | 6.3 to 8.5 ms | 0.52 to 0.70 ms | 9x to 17x | 6.4 to 6.7 ms | 0.59 to 1.59 ms | 4x to 11x |
| `if` and `while` on data, batch 256 | 33 to 35 ms | 1.7 to 1.9 ms | 18x to 19x | 7.8 to 11.8 ms | refused | none |

The vmapped per sample gradient program has 26 nodes, as many as the unbatched one; the transform
takes 65 to 140 ms. Compiled with `CompiledExecutor("cpu", "float32")` it runs in 1.2 to 1.3 ms
against 0.6 to 1.0 ms for `torch.func.vmap` in float32, and agrees with the float64 reference to
1.2e-7: the vmapped program compiles with no change to the compiler.

**Speed, Jacobians and Hessians** (MLP 16, 32, 16, input [8, 16], so 128 x 128 blocks):

| quantity | Truestar transform | Truestar run | torch.func | agreement | program nodes |
|---|---|---|---|---|---|
| `jacfwd` | 1 ms | 0.17 to 0.33 ms | 0.88 to 1.44 ms | 2.0e-16 | 3 to 16 |
| `hessian` | 3 to 5 ms | 0.57 to 0.78 ms | 2.5 to 3.3 ms | 2.5e-16 | 5 to 38 |

The gain comes from running one flat program instead of stacking torch.func's dispatch layers per
call; at these sizes the work is small and the overhead dominates. Larger models were not timed.

## Tests that prove the tests work

* vmap: a softmax rule that does not move its dim fails the loop comparison (error above 1e-2); a
  conv rule that merges `[V, B]` in the wrong order type checks when V equals B but mixes elements,
  and fails; a sum rule with the wrong type is stopped by the per node type check; deleting a rule or
  marking an op undefined is refused by name.
* Forward: a tanh rule with `1 - y` instead of `1 - y^2` fails finite differences and the dot
  product test (both above 1e-2); a softmax rule without the `sum(t y)` term (wrong only off the
  diagonal) fails both.

## Limits

* `jacfwd` and `jacrev` run one `vmap` per `wrt` value (or per output), and each recomputes the
  primal; there is no common subexpression pass to share it.
* Forward mode through a `cond` or `loop` with tangents recomputes that node's primal inside the
  `linearize` linear program, since one node computes both.
* `slice` with a step over a symbolic dim has a forward rule but no reverse rule, so `grad`,
  `jacrev` and `hessian` refuse it by name; `jvp` and `jacfwd` run it. (`fmod`, `remainder`,
  `vector_norm`, `group_norm`, `rms_norm` and `adaptive_max_pool2d`, once listed here, have forward
  mode since the op coverage work above.)
* A batch that disagrees in a `cond` still runs each branch on every element; an error or a loop
  that only the other elements' branch would avoid can still surface (a `max_iterations` loop
  raises, never hangs).
* Under vmap, a `while` loop runs as long as its slowest element.
* The reference interpreter computes in float64; the compiled float32 run was checked only on the
  per sample gradient program.
* JAX is not installed here; JAX behavior is quoted from its documentation and issues, not run.
* `log_softmax` and `all_reduce` (registered by `truestar.train`) have vmap and forward rules;
  `log_softmax` is checked against the loop and finite differences, but its reverse rule lives in
  `truestar.train.grad`, so the dot product test does not cover it; `all_reduce` needs
  torch.distributed and was not run under vmap.
* Found on the way, not fixed (outside this work): the reference interpreter's `cumsum` of a rank 0
  tensor raises `ZeroDivisionError` (`reference.py`, `attrs["dim"] % x.dim()`).

## Pain ledger mapping

| Row | How this serves it | Still open |
|---|---|---|
| P24 vmap and forward mode gaps | data dependent control flow, random ops and every IR op batch; forward rules for every op with a reverse rule, verified two ways | the forward rules listed under Limits |
| P5 graph breaks and recompiles | one vmapped program serves every batch size (symbolic `V`); a `while` per element needs no recompile | |
| Never introduce #1, no silent fallback | the per element loop is exact, not an approximation; unclassified ops are refused by name | |
| Never introduce #3, no silent wrong results of our own | every batched node is type checked; every rule checked against the loop, finite differences and the transposed reverse rule | compiled kernels checked on one program only |
