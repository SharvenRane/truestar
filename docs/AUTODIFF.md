# Autodiff as a compiler pass

Author: Sharven Rane. First written 2026-11-24.

DESIGN.md asks for "autodiff as a compiler pass". This is that pass. It turns an IR program into a
new IR program that also returns gradients. Code lives in `src/truestar/ir/autodiff.py` (the pass)
and `src/truestar/ir/grad_rules.py` (one rule per op), tests in `tests/test_autodiff.py`.

## Use

```python
from truestar import ir
from truestar.ir.autodiff import grad

prog = ir.import_module(model, (x, y), dynamic_shapes=...)       # forward returns a scalar loss
step = grad(prog, wrt=["body.0.weight", "body.0.bias"])          # a new program
loss, g_weight, g_bias = ir.run(step, x, y)
```

`grad(program, wrt, of=None, seed="auto", name=None)`:

* `wrt`: names of inputs or params. Params are the weights the importer stored, under the names
  they have in the module's `state_dict`.
* `of`: indices of the outputs to differentiate. Default is every output.
* The result computes a vector Jacobian product. For an output `y` and a cotangent `v` of the same
  shape it returns `J^T v` for each `wrt` value. With a scalar output and `v = 1` that is the
  plain gradient.
* `seed` says where `v` comes from. `"auto"`: a rank 0 output is seeded with 1, any other output
  gets a new program input named `d_<output>`. `"input"`: every output gets a cotangent input.
  `"ones"`: every output is seeded with ones, which gives the gradient of its sum.

Layout of the returned program:

| Part | Content |
|---|---|
| inputs | the original inputs, then the cotangent inputs |
| outputs | the original outputs, then one gradient per `wrt` entry in order, each with the shape of its `wrt` value |
| params | the original params under the same names, plus the constants the backward code needs (`@d_const`, pooling matrices, a causal mask) |

The original program is not changed. A `wrt` value that no differentiated output depends on gets
zeros. `grad` is not exported from `truestar.ir` yet; import it from `truestar.ir.autodiff`.

## Design

**A program transformation, no tape.** The backward pass is emitted as ordinary IR nodes behind
the forward nodes. A two layer MLP differentiated with respect to one weight and one bias:

```
program grad_Sequential(%input: f32[B, 4], %d_linear_1: f32[B, 2]) -> (f32[B, 2], f32[3, 4], f32[2])
  symbol B in [1, 64]
  %linear = linear(%input, @0.weight, @0.bias) : f32[B, 3]
  %tanh = tanh(%linear) : f32[B, 3]
  %linear_1 = linear(%tanh, @2.weight, @2.bias) : f32[B, 2]
  %d_matmul = matmul(%d_linear_1, @2.weight) : f32[B, 3]
  %d_sum = sum(%d_linear_1) dims=[0] keepdim=False : f32[2]
  %d_mul = mul(%tanh, %tanh) : f32[B, 3]
  %d_sub = sub(@d_const, %d_mul) : f32[B, 3]
  %d_mul_1 = mul(%d_matmul, %d_sub) : f32[B, 3]
  %d_permute = permute(%d_mul_1) dims=[1, 0] : f32[3, B]
  %d_matmul_1 = matmul(%d_permute, %input) : f32[3, 4]
  return %linear_1, %d_matmul_1, %d_sum
```

Consequences:

* Nothing is recorded at run time. The interpreter and any later backend run the result like any
  other program.
* Every emitted op goes through the IR's own shape rules at the moment it is emitted, and the
  finished program is verified again. A gradient with the wrong shape cannot be built. The pass
  also checks every cotangent against the shape of the value it belongs to.
* The backward code reads forward values directly (`%tanh` above), so a compiler can fuse and
  schedule forward and backward as one program.
* One transformed program serves every batch size in the declared range.

**The sweep.** The pass first marks the values that depend on a `wrt` value through differentiable
ops, and the values that reach a differentiated output. It then walks the nodes in reverse. A node
is differentiated only when its output is marked and has received a cotangent, and a rule is asked
only for the operands that are marked. That is the dead code rule: no backward code for a branch
that does not reach the requested outputs, for an operand that does not depend on `wrt`, or for a
value that only feeds a comparison.

**Fan out.** A value read by several nodes receives one cotangent from each reader. They are added
when the value's own node is reached.

**Broadcasting.** A cotangent has the broadcast shape and is summed back to the operand's shape:
leading dims the operand lacks are summed away, and a dim where the operand is 1 is summed and
kept. The shape rules let a dim broadcast only when it is provably 1, so the axes to sum are known
at transform time for every value of the symbols. `f64[B, 1, 4] + f64[3, 1]` gives gradients of
type `f64[B, 1, 4]` and `f64[3, 1]`.

**Symbolic sizes.** A mean over the batch dim divides by a count that is only known at run time.
The IR has no size op, so the count is computed in the program as the sum of a ones vector of that
length. A fixed count becomes a constant.

**Lowering.** `batch_norm`, `layer_norm` and `attention` are rewritten into primitive ops when a
gradient has to flow through them, and are left fused otherwise. Their backward needs their
internals (the centred input, the standard deviation, the attention probabilities). Lowering lets
forward and backward share those values. Each lowering repeats the arithmetic of the reference
kernel in the same order, and a test asserts the lowered forward equals the fused op to 1e-15.

**Conv and pooling gradients as three IR ops (2026-11-24, replacing a composite).** The rules of
`conv2d` and `max_pool2d` used to write their gradients with `conv2d` itself plus zero insertion,
padding, flips and per tap slices (dozens of layout nodes per conv, and a zero inserted copy of the
cotangent `s^2` times its size). They now emit one op per gradient:

* `conv2d_input_grad(g, w)` (attrs: the conv's stride, padding, dilation, groups and `input_size`):
  the transposed convolution, `x_bar[n, c, h, w] = sum over o, u, v of g[n, o, (h + p - u d) / s, ...]
  * w[o, c, u, v]` where the division is exact;
* `conv2d_weight_grad(x, g)` (attrs: the conv's and `kernel_size`):
  `w_bar[o, c, u, v] = sum over n, i, j of x[n, c, i s + u d - p, ...] * g[n, o, i, j]`;
* the bias gradient stays a `sum`;
* `max_pool2d_grad(x, g)` (the pool's attrs): each window gives its cotangent to its first tap equal
  to the pooled value, row major, so ties resolve the way PyTorch resolves them.

Each op has a shape rule, a float64 reference written from the forward formulas (taps as strided
slices, no `F.conv2d`), a finite difference case in tests/test_autodiff.py, and a kernel rule in the
compiler (docs/COMPILER.md). Second order gradients work through their own rules: the input gradient
is bilinear in (g, w), so its rule is `conv2d(gg, w)` and `conv2d_weight_grad(gg, g)`; the weight
gradient's rule is `conv2d_input_grad(g, ww)` and `conv2d(x, ww)`; `max_pool2d_grad` is linear in g
(its g gradient picks each window's first maximum from the incoming cotangent) and its x gradient is
zero, since x only moves the selection. `adaptive_avg_pool2d` is separable, `y = A_h x A_w^T` with
constant averaging matrices, so its gradient is two `matmul` nodes.

**Second new IR op: `scatter_add` (2026-11-24).** `scatter_add(x, index, src) dim` adds `src[p]` at
`x[p with dim replaced by index[p]]` for every `p`. The backward of every op that moves entries
(`sort`, `topk`, `gather`, `index_select`) is a scatter of the cotangent back to where each entry
came from. The reference writes it with indexed assignment: one assignment when no row repeats an
index (sort and topk, so each entry receives at most one value and the result is exact), else one
assignment per position along dim, in increasing order, so repeated indices sum in a fixed order.
Multi value nodes needed nothing new in the pass: only the values output of `sort` and `topk` is
floating point, so only it is active and receives a cotangent, and the rule reads the node's
indices output.

**Store what the backward reads, not more (2026-11-24).** A rule should read the value the backward needs anyway. The relu rule used to read the input `x` (`x > 0`), while the next layer's weight gradient keeps the
output `y`; both were stored. It now reads `y > 0`, which is the same mask for every x, so the input
has one reader, the relu, and a compiler keeps it inside the producing kernel's epilogue. Measured
on the recompute test's three layer transformer (tests/test_memory.py): planned peak of the
gradient program 4.12 MiB before, 3.47 MiB after (16% less); the six layer transformer is in
docs/TRAINING.md (Compiled).

**One new IR op: `erf`.** The derivative of exact gelu is `Phi(x) + x phi(x)`, and `Phi` cannot be
written with the existing ops. `erf` is registered from `grad_rules.py` through the same
registries `shapes.py` and `reference.py` use, with its own gradient rule. `core.py` is not touched.

**Loud failures.** An op on the gradient path with no rule raises `UnsupportedOpError` naming the
op, the value and where it came from (the ATen op and the user's source line when imported). All
such ops are listed in one error, before any backward code is emitted. An op with no rule that is
off the gradient path is not an error. Control flow has rules since 2026-11-24 (section below).

## Control flow (added 2026-11-24)

Code: `src/truestar/ir/autodiff_cf.py`, tests `tests/test_autodiff_cf.py`, measurements
`scripts/bench_controlflow_grad.py`.

Before, `grad` refused any program with a `cond` or a `loop` on the gradient path ("no gradient
rule for IR op 'loop'"), so a model written with the native front end could run but not train.
Now:

```python
import truestar
from truestar import Tensor, ops

@truestar.function
def rnn_loss(params: {"wx": Tensor[4, 5], "wh": Tensor[5, 5]}, x: Tensor["B", "T", 4],
             y: Tensor["B", 5]) -> Tensor[()]:
    h = ops.zeros((x.shape[0], 5))
    for t in range(x.shape[1]):                 # a loop over a symbolic length
        h = ops.tanh(x[:, t] @ params["wx"] + h @ params["wh"])
    if h.mean() > 0:                            # a branch on data
        h = h * 2.0
    return ((h - y) * (h - y)).mean()

loss, g = truestar.grad(rnn_loss, wrt="params")(params, x, y)   # g["wx"], g["wh"]
```

On the IR, `grad(program, wrt, loop_memory="auto")` is the same call as before with one new
argument.

**Design.** The backward of a control flow op is the same op, built from `grad` applied to its
regions, so nesting (a `while` inside an `if` inside a `for`) recurses with no special case:

* `cond(pred, *x)` gets `cond(pred, *x, *y_bar)` whose branches are the vector Jacobian products
  of the two branches. Nothing is stored; the taken branch's internals are recomputed inside the
  backward branch.
* `loop` gets a loop that walks the iterations in reverse and applies the body's vector Jacobian
  product. Gradients of captured values (weights read inside the body) are accumulated in carried
  values of the backward loop. The gradient of `x[:, t]` is added one row per step
  (`index_copy` of `acc[t] + g`) instead of a zeros tensor of the whole `x` plus one row.
* A `while` loop also carries a counter in its forward run, so the backward knows how many
  iterations to undo.
* `select_dynamic` (reading `x[:, t]` at a run time index) has a rule: zeros with the row written
  back with `index_copy`.
* `scan` imported from torch.export is a `loop` that reads with `select_dynamic` and writes with
  `index_copy`, so it is differentiated with no rule of its own.

**Memory policy.** The backward of iteration `k` needs the carried values that entered iteration
`k`. One word for the whole program says where they come from:

| `loop_memory` | carried values kept | extra compute | notes |
|---|---|---|---|
| `"store"` | one copy per iteration, only of the values the body's gradient reads | one more body run per iteration | a value only written by `index_copy` (a scan's stacked outputs) costs nothing |
| `"sqrt"` | a checkpoint every `ceil(sqrt n)` iterations plus one segment, at most `2 ceil(sqrt n)` copies | about one more forward | needs a trip count |
| `"recompute"` | none | about `n^2 / 2` body runs | constant memory |
| `"auto"` (default) | `"store"` where the iteration count has a bound | | |

A loop that cannot follow the request says so: a `while` loop with no `max_iterations` has no size
for a buffer, so it always recomputes, and `"sqrt"` on a bounded `while` loop stores. Every backward
loop carries a `memory` attribute naming the policy it got, visible in the program text.

**Measured** (2026-11-24, torch 2.11, CPU, float64; `scripts/bench_controlflow_grad.py`, native RNN
loss, batch 4, hidden 32, input 16, reference interpreter):

| T | policy | carried copies kept | run ms | max rel error vs torch.autograd |
|---|---|---|---|---|
| 8 | store | 8 | 8.4 | 0 |
| 8 | recompute | 0 | 18.2 | 0 |
| 8 | sqrt | 6 | 18.8 | 0 |
| 32 | store | 32 | 33.5 | 0 |
| 32 | recompute | 0 | 271.2 | 0 |
| 32 | sqrt | 12 | 74.8 | 0 |
| 48 | store | 48 | 48.7 | 0 |
| 48 | recompute | 0 | 540.1 | 0 |
| 48 | sqrt | 14 | 100.5 | 0 |

* One compile of the function serves every `T` above (and the gradient program is one program for
  every `T` and `B`).
* Finite differences (fourth order, the bar of the other rules, 1e-8): worst 2.4e-10 over 8
  programs (RNN, a two output branch on both sides, a data dependent `while`, a `while` inside an
  `if` inside a `for`, two carried values plus an int counter, a scan style stack, a loop with zero
  iterations) times 3 policies.
* `torch.autograd` on the same computation unrolled in plain PyTorch: worst 7.8e-17 over 14 cases
  times 4 policies (bar 1e-12), including `T = 1`, zero iteration loops, both sides of every branch
  and a nested case checked to take both sides and to run its inner `while`.
* The `sqrt` checkpoints at `T = 48` are 7 plus a segment of 7 (14 copies, `2 ceil(sqrt 48) = 14`).

**Against JAX and PyTorch.** JAX differentiates `lax.scan` and `fori_loop` with a static trip count
but not `while_loop` in reverse mode, and users choose between three loop primitives; here a plain
`for` over a symbolic length and a plain `while` on data both differentiate, from one function, and
the memory trade off is one named argument instead of a choice of primitive. PyTorch differentiates
an eager loop by recording it, once per length; here the gradient program is built once for every
length.

**Tests that prove the tests work.** Permanent: dropping the capture gradients (`_accumulate`
returning the accumulator unchanged) makes the RNN weight gradients wrong by more than 1e-2;
zeroing the stored carried rows makes the `store` policy wrong by more than 1e-2 while `recompute`,
which stores nothing, stays exact.

## Rule table

48 ops are classified: the 47 in IR.md plus `erf`. `autodiff.rule_table()` returns this table.

| Op | Handling | Gradient |
|---|---|---|
| `add`, `sub`, `mul`, `div` | rule | elementwise, then summed back over broadcast dims |
| `neg`, `exp`, `sqrt`, `rsqrt`, `tanh`, `sigmoid`, `erf` | rule | closed form in terms of the output where possible (`tanh`: `1 - y^2`) |
| `relu`, `clamp`, `hardsigmoid`, `hardswish` | rule | piecewise, selected with `where`; kink conventions follow PyTorch (relu is 0 at 0, clamp passes the bounds); since 2026-11-24 the relu mask is `y > 0` on the output, not `x > 0` on the input (equal for every x, NaN included), so the input is not kept for the backward pass; since 2026-11-25 its zero is a scalar constant, not a `zeros_like` tensor |
| `expm1` (2026-11-24) | rule | `g exp(x)`; the IR op exists for Adam's bias correction `1 - b^t = -expm1(t ln b)`, which does not cancel in float32 |
| `silu` | rule | `s (1 + x (1 - s))` with `s = sigmoid(x)` |
| `gelu` | rule | exact form with `erf`; tanh form with its own derivative |
| `where` | rule | cotangent routed by the condition; none into the condition |
| `sum`, `mean` | rule | cotangent expanded back over the reduced dims, `mean` divided by the count |
| `softmax` | rule | `y (g - sum(g y))` along the softmax dim |
| `matmul` | rule | `g b^T` and `a^T g`, summed back over broadcast batch dims |
| `linear` | rule | input `g w`, weight `g^T x` over flattened leading dims, bias a sum |
| `conv2d` | rule | input and weight through `conv2d_input_grad` and `conv2d_weight_grad`, bias a `sum` (stride, padding, dilation, groups) |
| `max_pool2d` | rule | `max_pool2d_grad`: first max per window (padding, dilation, ceil mode) |
| `conv2d_input_grad`, `conv2d_weight_grad` | rule | each other and `conv2d` (bilinear ops) |
| `max_pool2d_grad` | rule | g: the window's first maximum of the cotangent; x: zero |
| `adaptive_avg_pool2d` | rule | two `matmul` nodes with constant averaging matrices; a global pool (1 x 1 output) is `expand(g / (H W))` (2026-11-25) |
| `reshape`, `permute`, `expand` | rule | reshape back, inverse permutation, sum back |
| `select`, `slice`, `cat` | rule | zeros around the cotangent (any step), blocks of the cotangent |
| `batch_norm` (inference form) | lowered | input, weight, bias, and also the running mean and variance |
| `layer_norm` | lowered | input, weight, bias |
| `attention` | lowered | q, k, v and an additive mask; bool mask, causal flag and custom scale handled |
| `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `logical_and`, `logical_or`, `logical_not`, `any` | no gradient | bool results |
| `full`, `full_like`, `arange` | no gradient | constants |
| `flip` (added 2026-11-24, docs/LLM.md) | rule | flip the cotangent back |
| `index_copy` (added 2026-11-24) | rule | x: the cotangent with the written rows zeroed; src: its rows at the index |
| `amax`, `amin` (added 2026-11-24) | rule | the cotangent goes to the entries equal to the result, split evenly among ties |
| `sort`, `topk` (added 2026-11-24) | rule | `scatter_add` of the cotangent into zeros at the indices output: each entry gets the cotangent of the sorted position it went to. The sort is stable, so tied entries keep their own positions, as `torch.sort(stable=True)` does; no splitting between ties (unlike `amax`) |
| `gather` (added 2026-11-24) | rule | `scatter_add` of the cotangent at the index; repeated indices add up in increasing position order |
| `index_select` (added 2026-11-24) | rule | the index broadcast to the cotangent's shape, then `scatter_add`; an embedding lookup is this with dim 0 |
| `scatter_add` (added 2026-11-24) | rule | x: the cotangent; src: a `gather` of the cotangent at the index |
| `cond`, `loop` (added 2026-11-24) | control flow | a backward `cond` of the branches' gradients; a reverse `loop` of the body's gradient, with a memory policy (section "Control flow") |
| `select_dynamic` (added 2026-11-24) | rule | zeros with the cotangent written at the run time index |

Added with the op coverage work (2026-11-24, `ir/grad_rules_extra.py`), each with a finite difference
case in tests/test_autodiff.py (which also feeds the forward mode dot product test and the vmap
test):

| Op | Handling | Gradient |
|---|---|---|
| `cos`, `sin`, `log`, `reciprocal`, `abs` | rule | closed forms; `abs` has slope 0 at 0 (as PyTorch) |
| `cast` | rule | float to float: cast back to the source dtype; an integer or bool source has none. Global now (before, only `truestar.train` installed it during its own `grad` calls) |
| `log_softmax` | rule | `g - exp(y) sum(g)`; global now, like `cast` |
| `pow` | rule | base: `g b a^(b - 1)`, 0 where b == 0; exponent: `g y log a`, 0 where a == 0 and b >= 0 (PyTorch's conventions) |
| `maximum`, `minimum` | rule | the winning operand takes the cotangent, a tie splits it in half (as PyTorch) |
| `fmod`, `remainder` | rule | a: g; b: `-g q(a / b)` with q = trunc or floor |
| `cumsum` | rule | the reversed running sum of g (`flip`, `cumsum`, `flip`) |
| `triu`, `tril` | rule | the same mask applied to g |
| `pad` | rule | g with the padding cut off (`pad` with negated pads; slices over a symbolic dim) |
| `slice_scatter`, `select_scatter` | rule | x: g with the written slice zeroed; src: g's slice |
| `index` | rule | `scatter_add` into zeros at the indexed positions: adjacent index tensors are broadcast, wrapped (negative entries) and combined into one flat index over the merged indexed dims |
| `avg_pool2d` | rule | `conv2d_input_grad` of g / divisor with a ones kernel per channel (the reference's divisors; ceil mode through a larger input size that is then cropped) |
| `group_norm`, `rms_norm`, `vector_norm` | lowered | primitive ops in the reference's order; promotion as the fused op's shape rule |
| `adaptive_max_pool2d` | lowered | every window read as kh x kw taps (shorter windows repeat their last row or column), `argmax` for the first maximum (PyTorch's rule), `gather` |
| `frozen` | no gradient | the value of x without a derivative |

With these, `grad` covers every float IR op except `multinomial` and the random ops (whose outputs
carry no gradient by definition). Measured on the coverage list (docs/MODEL_COVERAGE.md): every
model whose gradient was checked matches torch.autograd in float64 below 1e-10 relative, except
torch's own float64 `hardsigmoid` backward, which multiplies by the float32 constant 1/6
(0.1666666716337204, a 3e-8 relative error on the torch side; Truestar's slope is exactly 1/6).

**Embedding gradients now follow torch.** The importer used to drop `padding_idx` and
`scale_grad_by_freq` ("they only change gradients"), so the gradient of an imported BERT differed
from torch.autograd in the padding row of the word embeddings (found by the coverage sweep, 0.36
relative). Both are now written with `frozen` and are exact.

Everything the importer emits is covered: `addmm` arrives as `matmul` plus `add`, `view`,
`flatten`, `squeeze` and `unsqueeze` as `reshape`, `transpose` and `t` as `permute`, `hardtanh` and
`relu6` as `clamp`.

## What is verified (2026-11-24, torch 2.11, torchvision 0.26, CPU, float64)

`tests/test_autodiff.py`, 108 tests.

**Every rule against finite differences.** 74 small programs, at least one for each of the 35
differentiable ops, with variants for broadcasting on either side, a 0 dim operand, negative dims,
keepdim, strides that leave unread rows, dilation, groups, depthwise conv, padding larger than the
kernel, ceil mode, bool and additive attention masks, the causal flag, slices with a step. The
check compares the program's vector Jacobian product with a fourth order central difference of the
reference interpreter, entry by entry, for every float input.

* Worst relative error over the 74 cases: **1.6e-10** (`layer_norm` with a large eps). Median 9e-13.
* Bar: 1e-8. With step 1e-3 the truncation term is `h^4 / 30 * phi^(5)`, about 3e-14 times the
  fifth derivative, and the rounding term is about 3e-13 times the function value. A wrong rule is
  off by a factor, a sign or a missing term, an error of 1e-2 or more.
* Inputs of piecewise ops are kept 0.05 away from every kink, and max pool inputs are all distinct
  by 0.05, because a difference that straddles a kink measures nothing.

**Whole models against `torch.autograd`.** The gradient of the input and of every parameter, for a
random output cotangent, from one transformed program run at batch sizes 1 and 3, compared with
`torch.autograd.grad` on the same model in float64. Worst relative error per model, in the export
graph and in the core ATen graph:

| Model | Export graph | Core ATen graph |
|---|---|---|
| MLP (gelu, tanh, relu) | 1.9e-16 | 1.7e-16 |
| resnet18, inference batch norm, random statistics | 4.2e-15 | 4.2e-15 |
| TransformerEncoderLayer (relu, post norm) | 5.9e-16 | 5.9e-16 |
| TransformerEncoderLayer (gelu, pre norm) | 7.3e-16 | 6.6e-16 |
| mobilenet_v3_small | 3.9e-15 | 8.6e-15 |
| Zoo (grouped, dilated and depthwise conv, ceil mode pool, adaptive pool to a non divisor size, causal attention, silu, hardswish, relu6, tanh gelu, cat, strided slice, division, mean over the batch) | 1.9e-15 | 1.5e-15 |

The target was 1e-8. The tests assert 1e-10, the bar the forward tests use, and nothing needed a
looser one. The transformer layer is also checked without `torch.autograd`, by finite differences
along random directions through all its parameters.

**A PyTorch rounding found on the way.** With the stock module, mobilenet_v3_small differed from
our gradients by 3.2e-8. The cause is PyTorch's `hardsigmoid` backward kernel, which multiplies by
1/6 rounded to float32 (0.1666666716337204) even for float64 tensors, a relative error of 3.0e-8.
Our rule uses the float64 constant and passes finite differences at 8e-12. The model comparison
therefore swaps `nn.Hardsigmoid` for the same function written with `clamp` in the torch reference
only. `test_hardsigmoid_gradient_is_exact_where_torch_rounds` records it and still passes if
PyTorch fixes the constant.

**Symbolic batch.** Every model result above is one program at two batch sizes. A hand built
program also covers the symbolic cases rule by rule: broadcasting against `B`, a mean with a
symbolic count, `select`, `slice`, `expand` and `cat` along the symbolic dim, checked by finite
differences at batch 2 and 5.

**Training.** An MLP with its mean squared error loss is imported from `torch.export`,
differentiated, and trained by SGD with the reference interpreter for 10 steps (8 on a fixed batch
of 16, then batch sizes 1 and 5 with the same program). The loss falls on every fixed batch step,
and the loss and all six parameter tensors match `torch.optim.SGD` in float64 to 1e-13 at every step.

**Structure.** Tests assert the exact backward ops emitted for a dead branch (none), for a `wrt`
value that does not reach the output (one `full` of zeros), for a single requested operand of
`linear` (one `matmul`), and for a value hidden behind a comparison (none).

### Sort, topk and the gathers (2026-11-24, torch 2.11, CPU float64 and RTX 5070 Ti)

* Finite differences (`tests/test_autodiff.py`, 9 new cases, entries distinct by 0.05 so no step
  reorders them): sort ascending 1.1e-13, sort descending along dim 0 4.1e-14, topk largest 7.2e-15,
  topk smallest along a middle dim 1.4e-13, sort indices feeding a gather 1.3e-13, gather with
  repeated indices 2.8e-13, index_select with repeats 3.1e-13, an embedding lookup 1.8e-13,
  scatter_add with repeats 2.5e-13. Bar 1e-8.
* Against `torch.autograd` (`tests/test_sort_grad.py`): a sampler shaped module (sort descending of
  `x * w`, softmax of the sorted values, topk largest and smallest, a gather with the sort indices)
  imported with a symbolic batch, one gradient program at batch 2 and 5: worst 2.4e-16 with distinct
  values, 5.5e-16 with ties (values from {-1, -0.5, 0, 0.5, 1}: ties inside the sort, at the topk
  boundary and in the gather). With ties the torch side uses a stable sort for topk, because PyTorch
  does not promise which tied entry `torch.topk` returns. A topk tie case is also asserted exactly:
  `[1, 3, 3, 3, 0]` top 2 sends the two cotangents whole to positions 1 and 2.
* Compiled (`truestar.compiler`, float32): the gradient program has no fallback on CPU or CUDA
  (sort, topk, gather and scatter_add run as index library steps), matches the reference within
  1e-5 at batch 3 and 7 with ties, and gives bitwise equal results run to run. The backward of a sort
  is pure data movement, and on CUDA it equals the reference exactly.
* Tests that fail on purpose: a sort rule that gathers instead of scattering is off by more than 1e-2
  against `torch.autograd`; a device sort that ignores `descending` is stopped by the first call check.

## Tests that prove the tests work

Permanent tests:

* a tanh rule returning `1 - y` fails the finite difference check (error above 1e-2) and the MLP
  comparison with `torch.autograd`;
* a layer norm lowering with eps dropped, a shift of about 1e-5, fails the transformer comparison;
* a broadcast rule that forgets to sum back is rejected by the pass's own cotangent shape check.

Manual breaks on 2026-11-24, each restored afterwards:

| Break in `grad_rules.py` | Tests that failed |
|---|---|
| softmax rule without the `sum(g y)` term | 14: both softmax cases, all four attention cases, the chain case, both transformer models and the zoo in both graph forms, the transformer finite difference test |
| conv input gradient without the kernel flip | 14: eight conv cases (the 1x1 conv has nothing to flip), resnet18, mobilenet_v3_small and the zoo in both graph forms |
| mean rule without the division by the count | 16: the mean and layer norm cases, the chain case, both transformer models, the zoo, the core ATen forms of resnet18 and mobilenet_v3_small, the transformer finite difference test |

## Limits

* The gradients of `conv2d`, `max_pool2d` and `adaptive_avg_pool2d` need fixed spatial sizes. A
  symbolic batch is fine. A symbolic height or width raises.
* `slice` with a step over a symbolic dim raises. `cat` along a symbolic dim works when every
  operand but the last has a fixed size.
* Batch norm is the inference form only, like the importer. Training mode batch norm, dropout and
  the update of running statistics are not expressible yet.
* Higher order: since 2026-11-24 `hessian` (forward over reverse), `jacfwd`, `jacrev` and `vmap` exist,
  see docs/TRANSFORMS.md. Before that: the output is a plain program and `grad` applies to it again; one test
  checks a Hessian vector product against `torch.autograd` (3e-16). No model scale second order test.
* No optimizer state in the IR. The training test applies the SGD update outside the program.
* The backward code is not simplified. The max pool gradient emits about 24 nodes per kernel tap
  (214 for a 3x3 pool), a conv input gradient 18 to 28 nodes and a conv weight gradient 4 to 6.
  resnet18 goes from 69 nodes to 1165. Transform time is 1.8 s for resnet18 and 0.6 s for the
  transformer layer. Folding and common subexpression elimination belong to the optimizer.
* Kinks follow PyTorch's conventions. Ties in a max pool window go to the first max.
* The reference interpreter computes in float64 whatever the declared dtype, so the verification
  is of the mathematics. Gradients in float32 or half precision on a real backend are a job for
  the verification layer once backends exist.
* `pow` has no rule (from `x ** 2`, checked 2026-11-24): write `x * x`, or the pass refuses and
  lists the ops that do have rules. Control flow has rules since 2026-11-24.
* `sort` and `topk` gradients at a tie follow the stable order; the derivative itself is not
  defined there.
* Loop gradients: the `store` policy keeps each iteration's carried values, not the residuals
  inside the body, so each backward step runs the body forward once more. A `while` loop with no
  `max_iterations` always recomputes (quadratic compute). `"sqrt"` needs a trip count.
  Gradients through loops are verified on the float64 reference interpreter. Checked once by hand
  (2026-11-24): `CompiledExecutor("cpu")` runs the RNN gradient programs of all three policies in
  float32 within 2.7e-7 of the float64 reference (store at T = 32: 12.5 ms warm). Not run on CUDA,
  and not a permanent test.
* Params of an imported program share storage with the module they came from. Import a copy of
  the module before training it with PyTorch alongside.

## Custom gradients, verified at registration (added 2026-11-24)

Code: `src/truestar/frontend/custom_grad.py` (registration and the check), `src/truestar/ir/custom_vjp.py`
(the IR op and its rule). Tests: `tests/test_custom_grad.py`, 16 tests, and the `custom_vjp` case in
`tests/test_autodiff.py`. Comparison: `scripts/custom_grad_compare.py`.

```python
@truestar.function
def softplus(x: Tensor["N"]) -> Tensor["N"]:
    return ops.where(x > 20.0, x, ops.log(1.0 + ops.exp(x)))

@softplus.grad
def _(x, g):                      # the primal's argument names, plus g (and out, the primal output)
    return g * ops.sigmoid(x)
```

The rule is a native function typed from the primal: an argument named like a primal tensor
parameter gets that type, `g` and `out` get the output type. It returns one gradient per floating
point tensor parameter, in the primal's order (a bare tensor when there is one); the number 0 is a
zero gradient. No `ctx`, no `save_for_backward`, no `nondiff_argnums` reordering.

**What registration checks.** First the types: a gradient whose type is not its parameter's is
refused at once, naming the parameter, with the fix (`the gradient returned for b has type f32[B, 8]
but b is f32[8]`, fix `.sum(0)`). Then the numbers, on generated inputs (uniform in (-2, 2),
`domain=` changes it) at two sizes of every symbolic dim (3 and 2, `sizes=` changes them):

* against reverse mode autodiff of the primal, entry by entry, bar 1e-8, when every op of the primal
  has a rule;
* against finite differences of the primal, always: per floating point input, three random
  directions, fourth order stencil at h = 1e-3, bar 1e-6. A direction where steps h and h/2 disagree
  crossed a kink and is drawn again.

A wrong rule raises `CustomGradientError` (code `wrong_gradient`) naming every wrong input and both
errors, and nothing is attached. `unverified="reason"` registers a rule that differs on purpose (a
straight through estimator); the reason is kept in `fn.grad_report()`. A primal that is not finite
on the generated inputs, or a gradient that can be checked neither way, is refused with the fix.

**How it is used.** The function's program becomes one `custom_vjp` node holding two regions, the
primal (`fwd`) and the rule (`bwd`). Running it runs `fwd`. `grad` inlines `bwd` with the node's
operands, its output and the cotangent, so the backward code is the user's own rule as ordinary IR.
A call from another native function becomes the same node. `truestar.torch_op` therefore registers
the rule as its backward, and the compiled executor and the kernel path replace the node by its
`fwd` region (`inline_custom_vjp`), so both run on kernels.

**Measured (2026-11-24, torch 2.11, CPU).** `scripts/custom_grad_compare.py`, one function
`tanh(x * w) + b` with four injected bugs:

| bug in the backward | torch.autograd.Function | Truestar `@fn.grad` |
|---|---|---|
| sign of the x gradient | accepted; backward runs; caught only by a hand run `gradcheck` | refused at registration: x, error 2.0 against autodiff and finite differences, 0.09 s |
| missing `1 - t^2` factor in the w gradient | same | refused: w only, 0.92 against autodiff, 1.1 against finite differences, 0.11 s |
| bias gradient a mean instead of a sum | same | refused: b only, 0.67 against both, 0.11 s |
| bias gradient not reduced at all (`[B, 4]` for `[4]`) | accepted, and the engine sums it back silently, so `gradcheck` passes | refused before any run: the type check names b and suggests `.sum(0)` |
| correct rule | accepted | accepted, verified in 0.06 s; worst error 0 against autodiff, 5e-12 against finite differences |

Verification of a correct rule costs 0.06 s to 0.15 s for these sizes; a test asserts under 1 s for
a 16 x 16 matmul layer. The softplus above shows why a rule is worth having: torch autograd of the
same source gives NaN at x = 1000 (the `where` gradient multiplies the overflowed branch by 0), the
rule gives 1. When this was measured its primal's `log` had no autodiff rule, so finite differences
alone checked it (error 5.8e-13, 0.01 s); since the op coverage work `log` has one, and the rule is
checked against autodiff of the forward as well.

JAX: `jax.custom_vjp` with `defvjp(fwd, bwd)`; its documentation describes no check of `bwd`
against the primal, only that `bwd` returns one value per primal argument, and `nondiff_argnums` are
passed to `bwd` first whatever their position
([custom derivative rules](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html),
read 2026-11-24). JAX is not installed here, so this row was not run.

Tests that prove the tests work: with both bars set to 1e9, 4 tests fail (the sign, missing term,
finite difference only and straight through cases).

Limits: one tensor output per function with a custom gradient (multi value nodes need the autodiff
pass to carry several cotangents into one node); no custom forward mode rule; generated inputs are
uniform per input, so a rule that is wrong only far outside `domain` passes; the check runs on the
float64 reference interpreter, so it verifies the mathematics, not a low precision kernel.

## Pain ledger mapping

| Row | How this serves it | Still open |
|---|---|---|
| P5 graph breaks and recompiles | Training needs no tracing or tape; one transformed program covers every batch size, tested at two sizes per model; since 2026-11-24 loops and branches too, one program for every loop length | Compiled loops |
| P7 the trained artifact is the deployed artifact | The training step is an IR program of the same kind as the inference program | Optimizers and state in the IR, a compiled backend |
| P9 shape errors found deep into a run | Gradient shapes are checked at transform time by the same rules as forward shapes | |
| Never introduce #1, no silent fallback | Ops without a rule and unsupported symbolic cases raise by name; nothing is skipped or approximated | |
| P18 wrong gradients (N6) | A hand written backward is checked against autodiff and finite differences when it is registered and refused by name | Low precision kernels of the rule |
| Never introduce #3, no silent wrong results of our own | Every rule is checked against finite differences and whole models against `torch.autograd`; broken rules are shown to fail | Compiled kernels do not exist yet |
| Never introduce #6, no cryptic errors | Errors name the op, the value, the ATen op and the user's source line | |
