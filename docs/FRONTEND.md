# Truestar native front end

Author: Sharven Rane. First written 2026-11-24.

Pain rows served: P5 (graph breaks and guard recompiles) and P9 (shape errors found late).
Code in `src/truestar/frontend/`, tests in `tests/test_frontend.py`, comparison script in
`scripts/frontend_vs_torch_compile.py`.

## What it is

A user writes a layer or a model as an ordinary Python function with typed shapes:

```python
import torch
import truestar
from truestar import Tensor, ops

@truestar.function
def rnn(x: Tensor["B", "T", 16], wx: Tensor[16, 32], wh: Tensor[32, 32]) -> Tensor["B", 32]:
    h = ops.zeros((x.shape[0], 32))
    for t in range(x.shape[1]):          # a loop over a symbolic length
        h = ops.tanh(x[:, t] @ wx + h @ wh)
    if h.sum() > 0:                      # a branch on tensor data
        h = ops.relu(h)
    return h

rnn(torch.randn(4, 9, 16), wx, wh)       # any B, any T: same compiled program
print(rnn.signature)                     # rnn(x: f32[B, T, 16], wx: f32[16, 32], wh: f32[32, 32]) -> f32[B, 32]
print(rnn.ir_text())                     # the IR, with the loop and cond regions
```

The decorator reads the function's source and translates its AST into Truestar IR. The function
body is never executed, so nothing is traced and there is nothing to break: a program either
translates whole or fails with an error naming the line. The translation happens at decoration
time, before any data exists, so a shape mismatch is raised when the module is imported.

`Tensor["B", 128, "f32"]` declares a symbolic dim `B`, a fixed dim of 128 and a dtype (default
f32). Dims can be small expressions (`"2*B"`). Symbol ranges are optional:
`@truestar.function(dims={"B": (1, 4096)})`.

## How Python constructs are translated

| Construct | Translation |
|---|---|
| `+ - * /`, `@`, unary `-`, `**` with a constant exponent, comparisons, `and` `or` `not` on tensors | IR elementwise ops, `matmul`, logical ops |
| Python numbers mixed with tensors | rank 0 constants typed like Python scalars (they never widen a dtype), stored exactly |
| `x.shape`, `x.ndim`, `x.dtype`, `len(x)` | compile time values (ints or symbolic dims) |
| `ops.relu`, `gelu`, `sigmoid`, `tanh`, `silu`, `exp`, `sqrt`, `rsqrt`, `abs`, `clamp`, `maximum`, `minimum`, `where`, `sum`, `mean`, `any`, `all`, `softmax`, `matmul`, `linear`, `layer_norm`, `attention`, `reshape`, `permute`, `transpose`, `cat`, `zeros`, `ones`, `full`, `*_like` | the matching IR ops; the same calls run eagerly through PyTorch outside a compiled function |
| common `torch.*` and `F.*` spellings of the ops above | accepted as aliases |
| methods `x.sum()`, `.mean()`, `.reshape()`, `.view()`, `.permute()`, `.transpose()`, `.unsqueeze()`, `.squeeze()`, `.T`, `.mT` and friends | the matching IR ops |
| indexing with ints, slices, `None`, `...` | `select`, `slice`, `reshape` |
| indexing with a loop index or a symbolic dim | new IR op `select_dynamic` |
| `if` on a compile time value (shape facts, static arguments) | resolved now; only the taken branch is translated |
| `if` on tensor data | new IR op `cond`; both branches are translated and type checked; an early `return` in one branch takes the rest of the function as the other branch |
| `a if c else b` on tensor data | elementwise `where` (a select) |
| `for` over a static range, tuple or list | unrolled (limit 10000 iterations) |
| `for t in range(n)` where `n` involves a symbolic dim | new IR op `loop` with a symbolic `trip_count`; one compile serves every length |
| `while` on tensor data | IR `loop` with a `cond` region |
| `while` on a compile time value | unrolled at compile time |
| `assert` on compile time facts | checked at compile time |
| calls to another `@truestar.function` | inlined |
| `ops.rand`, `randn`, `normal`, `bernoulli`, `randint`, `rand_like`, `randn_like`, `dropout` (and `F.dropout`) | IR `random` ops keyed by the function's seed and call count and by their site; `key=` and `name=` override (docs/RANDOM.md) |
| `ops.batch_norm(..., training=True)` | primitive ops returning `(y, new_running_mean, new_running_var)`; the inputs are never mutated |
| `h: Tensor["B", 64] = ...` and the return annotation | checked against the inferred type |
| parameters that are not `Tensor[...]` | static: part of the compile key; compiled at decoration when they all have defaults |

Loop carried values are found from the source (names assigned in the body that exist before the
loop). A Python number carried through a symbolic loop becomes a scalar tensor. A carried value
must keep its type across an iteration; a name assigned only inside a symbolic loop, or only in one
branch of a data dependent `if`, is undefined afterwards and using it is an error naming both lines.

Everything else fails loudly with the line and the construct: `try`, `with`, `lambda`,
comprehensions, `break` and `continue`, `global`, nested `def`, f strings, assignment to a
subscript or attribute (tensors are immutable), iterating over a tensor, a `return` inside a
symbolic loop, calls to anything outside the ops namespace, `math`, a few builtins and other native
functions. A branch on a symbolic dim (`if x.shape[0] == 1`) is an error too: deciding it would
mean one compile per shape, which is exactly what the front end exists to avoid.

## Structured arguments and results (pytrees, 2026-11-24)

A parameter can be a structure of tensors, declared in its annotation; the result can be any
structure. Nothing is registered: a dataclass or NamedTuple is read from its field annotations.

```python
from dataclasses import dataclass
from typing import NamedTuple

import truestar
from truestar import Tensor, ops

@dataclass
class Params:
    w: Tensor[4, 3]
    b: Tensor[3]

class Out(NamedTuple):
    y: Tensor["B", 3]
    norm: Tensor[()]

@truestar.function
def layer(p: Params, x: Tensor["B", 4], extra: {"scale": Tensor[()], "shift": [Tensor[3], Tensor[3]]}) -> Out:
    h = (x @ p.w + p.b) * extra["scale"] + extra["shift"][0] - extra["shift"][1]
    for k, v in {"a": h, "b": h * 2.0}.items():   # dict literals and items() resolve at compile time
        h = h + v
    return Out(y=ops.tanh(h), norm=(h * h).sum())

print(layer.signature)
# layer(p.w: f32[4, 3], p.b: f32[3], x: f32[B, 4], extra['scale']: f32[], extra['shift'][0]: f32[3],
#       extra['shift'][1]: f32[3]) -> Out(y=T, norm=T) with .y: f32[B, 3], .norm: f32[]
```

* Annotations: `{"key": Tensor[...]}`, `[Tensor[...], ...]`, `(Tensor[...], ...)`, a dataclass or a
  NamedTuple class with `Tensor[...]` fields, nested freely. Every leaf must be a tensor
  annotation; mixing static values into a structure is refused at decoration.
* In the body: `p.w`, `p["w"]`, `p[0]`, `len`, iteration, `.items()`, `.keys()`, `.values()`,
  `.get()`, dict literals, and constructing a dataclass or NamedTuple (a dataclass is filled field
  by field; its `__init__` and `__post_init__` are not run).
* The IR program stays flat: one input per leaf, named by its path (`p_w`, `extra_shift_1`), so
  every pass, the serializer and the compiler see nothing new.
* A call that does not match names the exact leaf: `extra['shift']: missing key 'shift'; unexpected
  key 'shfit' (expected keys: 'scale', 'shift')`, `p.w must have dtype f32 ...`, `p.b shape does not
  match its declared type`. A return annotation that names other keys fails at decoration with both
  structures.
* Limit: a structure cannot be reassigned inside a symbolic loop or a data dependent `if` (carry
  its tensors as separate names); it can be read there freely.

Tests: tests/test_frontend_pytree.py.

## Gradients (`truestar.grad`, 2026-11-24)

`truestar.grad(fn, wrt="params")` calls `fn` and returns `(value, gradients)`. The first tensor `fn`
returns must be a float scalar (the loss). Each gradient has the structure of its parameter (a dict
of weights gives a dict of gradients; `wrt=["p", "x"]` gives a tuple). Loops and branches are
differentiated as written, one gradient program for every loop length; `loop_memory` chooses how
loops keep what their backward needs (docs/AUTODIFF.md, section "Control flow").

```python
loss, g = truestar.grad(rnn_loss, wrt="params")(params, x, y)   # g["wx"], g["wh"]
```

Measured against `torch.autograd` on the same loss written in PyTorch (float64): worst difference
below 1e-12 at T = 1, 4 and 9 for the policies auto, recompute and sqrt.

`truestar.value_and_grad(fn, wrt, has_aux=False)` (2026-11-25) is JAX's convention on top of it:
without `has_aux` `fn` must return only the loss and a call gives `(loss, grads)`; with `has_aux`
`fn` returns `(loss, aux)`, aux any structure of tensors that is returned but not differentiated,
and a call gives `((loss, aux), grads)`. The wrong flag is refused with the fix in the message
(`tests/test_value_and_grad.py`, gradients equal to torch.autograd within 1e-12).

## IR additions

Four ops were added to the IR (`core.py`, `shapes.py`, `reference.py`):

* `cond(pred, *operands)`: `then_branch` and `else_branch` are regions (sub programs) that take
  the same operands and return values of provably equal types.
* `loop(*carried, *captures)`: `body(index, *carried, *captures)` returns the new carried values,
  run `trip_count` times (a dim, may be symbolic) or while a `cond` region returns true.
* `select_dynamic(x, index)`: `select` with an int64 scalar tensor index known only at run time.
* `dim_scalar`: a rank 0 tensor holding the run time value of a dim expression (for
  `x / x.shape[1]` with a symbolic dim).

Regions capture nothing implicitly: every outer value a region reads is an explicit operand, so a
node stays a pure op and a later pass can reason about it locally. `Node` gained `extra_outputs`
(control flow nodes define several values) and `Builder.op_multi` builds them. The reference
interpreter runs regions recursively with the parent's symbol bindings and checks the shape and
dtype of every value a node defines, including each carried value of a loop.

## Layers, optimizers and training steps (added 2026-11-24)

`@truestar.native` is the Truestar name of the decorator; `@truestar.function` stays as an alias.
On top of the front end: `truestar.nn` (layers whose `forward` is native source, inlined when called
inside a native function, docs/NN.md), `truestar.optim` (optimizers as native subset rules over
trees) and `truestar.train_step` / `truestar.fit` (one IR program per training step, docs/OPTIM.md).
Front end changes they needed:

* new ops `conv2d`, `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`, `log_softmax`, `expm1`;
* a call to an object with `__truestar_inline__` is inlined like a native function (layers and
  `nn.inline` functions);
* `dict(...)` is a static builtin;
* `truestar.grad` includes the training gradient rules (`cast`, `log_softmax`);
* `emit_call(builder, fn, *args)` translates a native function into an existing builder or editor.

## Custom gradients and foreign arrays (added 2026-11-24)

* `@fn.grad` attaches a hand written backward, checked at registration against autodiff and finite
  differences of `fn` and refused by name when wrong (docs/AUTODIFF.md, custom gradients).
* Arguments may be any DLPack array (NumPy, CuPy, JAX) instead of a torch tensor: imported without a
  copy after a dtype, shape, stride and device check (docs/INTEROP.md). Weights load from
  safetensors with `truestar.interop.load_weights(path, into=fn)`.

## Errors

Every error names the user's file, line and column, the source line, the op and every operand
type, plus a fix when the fix is obvious:

```
shape mismatch in matmul: inner dims differ: 32 vs 64
  at <repo>\scripts/frontend_vs_torch_compile.py:148 in buggy
  148 |             h = h @ w
  h: f32[B, 32]
  w: f32[64, 32]
  fix: w looks transposed: try h @ w.T
```

`e.to_dict()` returns the same facts as JSON compatible data (schema
`truestar.frontend.error/1`): `code` (`shape_mismatch`, `branch_type_mismatch`,
`loop_type_change`, `annotation_mismatch`, `ambiguous_condition`, `not_static`,
`unsupported_construct`, `undefined_name`, `input_mismatch` and a few more), `file`, `line`, `col`,
`source_line`, `function`, `op`, `operands` (name, type, dtype, shape) and `suggestion`. An agent
can apply the suggestion and recompile without reading prose.

Suggestions exist today for a transposed matmul operand, a linear weight given as `[in, out]`,
elementwise operands that are transposes or permutations of each other, a wrong `layer_norm`
normalized shape and a condition that is a whole tensor instead of a single value.

## Execution

Calling a compiled function checks the arguments against the typed signature (dtype, rank, every
symbol consistent and in range) and runs the program through an executor. The default executor is
the float64 reference interpreter, so outputs come back as float64. The seam for a code generator
is `truestar.frontend.Executor`: `prepare(program)` is called once per compiled program and returns
a callable that runs it. Pass one per function (`@truestar.function(executor=...)`) or globally
(`set_default_executor`). A test plugs in a counting executor and sees one `prepare` for three
batch sizes.

## Side by side with torch.compile

Measured 2026-11-24 with torch 2.11.0 on CPU by `scripts/frontend_vs_torch_compile.py`. Graph
breaks come from `torch._dynamo.explain`. Compiles are counted with a backend that counts every
graph Dynamo hands it and then runs the graph eagerly, so the numbers describe Dynamo and are not
inflated by Inductor. Batch sizes used: 2, 3, 4, 8, 16, 1.

### 1. A data dependent branch

```python
h = x @ w
if h.sum() > 0: h = relu(h)
else:           h = -h * 0.5
```

| | graph breaks | compiles over 6 calls | notes |
|---|---|---|---|
| torch.compile default | 1 (`Data-dependent branching`) | 7 graphs (2 or 1 per frame) | `fullgraph=True` raises `Unsupported: Data-dependent branching` |
| torch.compile `dynamic=True` | 1 | 5 | |
| torch.compile after a manual `torch.cond` rewrite | 0, `fullgraph=True` works | 3 (first shape static, then dynamic, then batch 1 specialized) | PyTorch does fine once the user rewrites the branch |
| Truestar | 0 by construction | 1, at decoration, none during the 6 calls | the plain `if` becomes an IR `cond` |

### 2. A loop over a symbolic length (an RNN over T = 1 to 12)

```python
h = zeros(B, 32)
for t in range(x.shape[1]): h = tanh(x[:, t] @ wx + h @ wh)
```

| | graph breaks | compiles over 12 lengths | graph size |
|---|---|---|---|
| torch.compile default | 0 per length | 8, then `recompile limit exceeded`: lengths 9 to 12 run uncompiled | unrolled, grows from 6 to 41 FX ops |
| torch.compile `dynamic=True` | 0 per length | 8, same limit hit | same: the trip count is specialized even with dynamic shapes |
| Truestar | 0 | 1 | constant: one `loop` op with `trip_count=T` and a 5 op body |

Here torch.compile captures each length without breaks but specializes on the length, recompiles
for every new one and silently stops compiling after `recompile_limit` (8).

### 3. A shape bug in a rarely taken branch

```python
h = relu(x @ w)            # w: [64, 32]
if h.mean() > 10.0:        # rare on normal data
    h = h @ w              # bug: [B, 32] @ [64, 32]
```

| | when the bug is reported |
|---|---|
| torch.compile | never during 30 calls on normal data (0 errors); at the first call whose data takes the branch: `TorchRuntimeError ... a and b must have same reduction dim, but got [2, 32] X [64, 32]` |
| Truestar | at decoration, before any data: line 148, op `matmul`, operands `h: f32[B, 32]` and `w: f32[64, 32]`, fix `w looks transposed: try h @ w.T` |

Plain eager PyTorch behaves the same as torch.compile here: the bug waits for the data.

### Control: a branch free MLP, where torch.compile does fine

| | graph breaks | compiles over 6 calls |
|---|---|---|
| torch.compile default | 0 | 3 (static first shape, then dynamic, then batch 1) |
| torch.compile `dynamic=True` | 0 | 2 (batch 1 is still specialized) |
| Truestar | 0 | 1 |

On straight line code torch.compile has no graph breaks and needs only two or three compiles; the
difference is small. The gap opens with Python control flow on data or on lengths.

## Measured correctness

Every construct in the tests is compared against the same source run eagerly by PyTorch in float64
(`f.python(...)`), and the main ones against a hand written PyTorch reference too. Relative error
bar: 1e-10. Both branches of every data dependent `if` are
exercised, loops run with lengths from 1 to 17 and zero trip loops, and `while` loops run with
data that exits after 0 to several iterations.

## Limits (2026-11-24)

* The default executor is the float64 reference interpreter: correct and slow. Since 2026-11-24
  `truestar.compiler.CompiledExecutor("cuda")` (or `"cpu"`) plugs into the executor seam and runs
  the same program through verified kernels, see docs/COMPILER.md.
* Autodiff of a native function goes through its IR program (`truestar.ir.grad`, `truestar.torch_op`); hand written rules through `@fn.grad`.
* Gradients: `truestar.grad` (since 2026-11-24), on the float64 reference interpreter by default.
* Random ops: `@truestar.function(seed=...)` sets the program seed; call n draws with
  `f.key_for(n)`; `f.calls` is the call counter (settable, for replay) and `f.sites` the site names.
  `torch.rand` and the other global generator calls are refused with the keyed op to use.
* No `.to()`, no scatter other than `ops.index_copy`, no in place updates of a slice inside a loop
  (accumulate into a carried value instead), no `break` or `continue`, no closures or comprehensions,
  no data dependent shapes. Casts, gather, max and min reductions, sort, top k and sampling exist
  since 2026-11-24 (docs/LLM.md).
* A loop carried value cannot change shape (no growing a sequence with `cat` in a loop).
* `for` over a range bounded by tensor data is not supported; write a `while` loop.
* Shape facts must be provable: a branch on a symbolic dim is rejected instead of guessed.
* Calls into arbitrary Python helpers are rejected; only the ops namespace, `math`, a few builtins
  and other native functions can be called.
* The IR now has region attrs and multi value nodes; a serializer must handle both.
