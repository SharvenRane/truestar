# Truestar IR (Stage 2 foundation)

Author: Sharven Rane. First written 2026-11-23.

The IR is the layer every later stage builds on: the compiler lowers it, the verification layer
checks compiled kernels against its reference interpreter, and the native front end will emit it.
Code lives in `src/truestar/ir/`, tests in `tests/test_ir.py`.

## What exists

### Values and types (`core.py`)

* A **type** is a dtype plus a shape. Each dim is an int or a symbolic expression over named
  symbols, for example `B`, `4*B` or `7*B`. Symbols are positive integers with an optional range
  taken from `torch.export` (`symbol B in [1, 64]`).
* A **value** is defined once (SSA). Kinds: `input`, `param` (weights, buffers and scalar
  constants, printed with `@`) and `op` (printed with `%`).
* A **node** is one pure op: it reads values and defines exactly one new value. Nothing is
  mutated and no two values share storage, so later passes can reorder or fuse without alias analysis.
* A **program** holds inputs, params with their data, nodes and outputs. `Program.verify()`
  re checks SSA order and reruns every shape rule. `str(program)` prints the whole program with the
  type of every value:

```
program TransformerEncoderLayer(%src: f32[B, 5, 32]) -> (f32[B, 5, 32])
  symbol B in [1, 64]
  ...
  %view = reshape(%select) shape=[5, 4*B, 8] : f32[5, 4*B, 8]
  %scaled_dot_product_attention = attention(%view_3, %view_4, %view_5, _) scale=None is_causal=False : f32[B, 4, 5, 8]
  %layer_norm = layer_norm(%add, @norm1.weight, @norm1.bias) normalized_shape=[32] eps=1e-05 : f32[B, 5, 32]
```

* `Builder` constructs programs by hand. Each op is shape checked the moment it is added.

### Ops (89 with `erf`, `cast`, `log_softmax`, `all_reduce` and `scatter_add`) and shape rules (`shapes.py`)

Ops marked *new* were added on 2026-11-24 (op coverage work, see the status log in
docs/QUALITY_BAR.md). Ops marked *LLM* were added later the same day for decoder LLMs (docs/LLM.md).

| Group | IR ops |
|---|---|
| Linear algebra | `conv2d` (stride, padding, dilation, groups), `linear`, `matmul` (batched, broadcasting) |
| Normalization | `batch_norm` (inference form, running stats), `layer_norm`, *new* `group_norm`, *new* `rms_norm` |
| Attention | `attention` (scaled dot product, optional bool or additive mask, causal flag, custom scale). Grouped and multi query attention (`enable_gqa`, `repeat_interleave` of KV heads) and sliding windows lower onto it, see docs/LLM.md |
| Activations | `relu`, `gelu` (exact and tanh), `hardswish`, `hardsigmoid`, `silu`, `sigmoid`, `tanh`, `clamp` |
| Elementwise | `add`, `sub`, `mul`, `div`, `neg`, `exp`, `sqrt`, `rsqrt`, `where`; *new* `pow`, `cos`, `sin`, `log`, `abs`, `reciprocal`, `erf`, `fmod`, `remainder`, `maximum`, `minimum` |
| Dtype | *new* `cast` (float to int truncates toward zero, anything to bool is `!= 0`) |
| Comparison and logic | `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `logical_and`, `logical_or`, `logical_not` |
| Reductions | `mean`, `sum`, `any`, `softmax`, *new* `vector_norm` (any order, inf, 0), *new* `cumsum`, *LLM* `amax`, `amin` (any dims, NaN propagates), *LLM* `argmax`, `argmin` (first index of ties) |
| Ordering | *LLM* `sort` (stable, NaN last; defines values and indices), *LLM* `topk` (values and indices, along a fixed size dim), *LLM* `flip` |
| Sampling | *LLM* `multinomial` (inverse CDF from a counter based generator, `seed` attribute, optional run time offset); `random` (keyed uniform, normal, bernoulli, randint: a key operand, stream operands, a `site` attribute; the same bits on every backend, docs/RANDOM.md) |
| Masks | *new* `triu`, *new* `tril` (with a diagonal offset) |
| Pooling | `max_pool2d` (padding, dilation, ceil mode), `adaptive_avg_pool2d`, *new* `avg_pool2d` (padding, ceil mode, `count_include_pad`, `divisor_override`), *new* `adaptive_max_pool2d` |
| Indexing | *new* `index` (integer advanced indexing, adjacent and split index dims), *new* `index_select`, *new* `gather`, *new* `slice_scatter`, *new* `select_scatter`, *LLM* `index_copy` (writes at run time positions: the KV cache update), `scatter_add` (added 2026-11-24 with the gradients of sort, topk, gather and index_select: src added at the indexed positions, repeated indices in increasing position order; docs/AUTODIFF.md) |
| Shape | `reshape`, `permute`, `expand`, `select`, `slice`, `cat`, *new* `pad` (constant, negative pads crop) |
| Creation | `full`, `full_like`, `arange` (since 2026-11-24 also with a symbolic end, `arange(T)`) |
| Control flow (added 2026-11-24 with the native front end) | `cond`, `loop` (regions; see docs/FRONTEND.md), `select_dynamic`, `dim_scalar` |
| Gradient only (added 2026-11-24, op coverage work) | `frozen(x)`: the value of x with no derivative (in `NONDIFF`). The importer writes gradient only semantics with it: an embedding's `padding_idx` rows and `scale_grad_by_freq` |
| Core since the op coverage work | `log_softmax` (before, only `truestar.train` registered it; `aten.log_softmax` now imports onto it) |

Every rule works on symbolic dims. Two dims match only when they are **provably** equal for every
value of their symbols: `B` and `C` do not broadcast against each other, `B` and `1` do. Dtypes
follow PyTorch promotion, including the rule that a 0 dim operand does not widen a dtype.

A failing rule raises `ShapeError` with the op, the reason, every operand type and, when imported,
the ATen op, the graph node, the user's source line and the module path:

```
shape error in conv2d: input channels 4 != weight channels 3 x groups 1
  x: f32[B, 4, 64, 64]
  weight: f32[64, 3, 7, 7]
  at: aten.conv2d.default (node %conv2d), resnet.py:285 in forward: return self._forward_impl(x) (module conv1: Conv2d)
```

### Importer (`importer.py`)

`ir.import_module(model, args, dynamic_shapes=..., decompose=False)` exports with
`torch.export.export` and converts the graph; `ir.import_exported(ep)` takes an existing
`ExportedProgram`. Both the default export graph and the core ATen graph
(`decompose=True`, which is `run_decompositions()`) are accepted. 211 ATen overloads are handled
(`ir.supported_aten_ops()`, 115 before 2026-11-24; 238 after the decoder LLM work, docs/LLM.md), lowered onto the IR ops: `view`, `flatten`,
`unflatten`, `squeeze` and `unsqueeze` become `reshape`; `transpose`, `t`, `swapaxes` and `movedim`
become `permute`; `addmm` becomes `matmul` plus `add`; `hardtanh` and `relu6` become `clamp`;
`contiguous`, `clone` and `dropout` in eval mode map to their input. Added 2026-11-24:

* `chunk`, `split`, `split_with_sizes` and `unbind` become slices or selects (along a fixed size dim);
  `roll` becomes two slices and a `cat`; `stack` becomes reshapes and a `cat`; `repeat` becomes
  reshape, expand, reshape (repeats may be symbolic).
* `einsum` becomes `sum`, `permute`, `reshape` and `matmul`, contracting operands left to right
  (implicit outputs and any number of operands; an ellipsis or a repeated index is refused).
* `embedding` becomes `index`; `outer` becomes two reshapes and a `mul`; `expand_as`, `view_as`
  and `type_as` become `expand`, `reshape` and `cast`; `_to_copy` and `to` become `cast`.
* `new_zeros`, `zeros`, `ones`, `new_full`, `zeros_like`, `fill` and similar become `full` or
  `full_like`; `masked_fill` becomes `where`.
* Instance norm becomes `group_norm` with one group per channel (export graph) or, in the core ATen
  graph, batch norm with input statistics written out as `mean`, `sub`, `mul`, `rsqrt`.
* `as_strided` is accepted only when it is a plain reshape of a contiguous tensor (Swin's core
  ATen graph has one after its average pool); any other strided view is refused.
* torch.export's integer division and modulo (`FloorDiv`, `PythonMod`) in recorded sizes are read as
  plain sympy `floor` and `Mod`, the forms the IR's rules produce and the serializer encodes.
* **In place changes through views** (Swin's `mask[a:b, c:d] = k`, Swin V2's
  `bias.clone()[n:2*n].zero_()`, `y[:, 0] = ...`, `y[:, 1:3] += 1`): the in place op on the last
  slice becomes a pure op, then `slice_scatter` or `select_scatter` writes it back up the chain of
  slices and selects, and later readers of the base tensor see the new value. It is refused when it
  cannot be rewritten exactly: a view of the same storage taken before the change and read after
  it, a base that is a program input or weight, a base that is itself a view. Since 2026-11-24 the
  plain in place check also refuses a change that another view of the same storage reads later.

Added with the op coverage work (2026-11-24, `ir/importer_extra.py`, docs/MODEL_COVERAGE.md), all
lowered onto existing IR ops, so every rule of those ops (reference, reverse, forward, batching,
compiler) applies with nothing new:

* 1D convolution as 2D over `[N, C, 1, L]`; 3D convolution as a sum over depth taps of 2D
  convolutions with depth folded into the batch; transposed 1D, 2D and 3D convolution as
  `conv2d_input_grad` (the adjoint of `conv2d`; 3D placed tap by tap along depth), and
  `aten.convolution` in all these forms (the core ATen graph).
* 1D pools through the 2D ops; 3D max and average pools and adaptive average pool as separable
  2D pools (plane, then depth); `upsample_nearest`, `linear`, `bilinear`, `trilinear`, `bicubic`
  as an `index_select` (nearest) or constant interpolation matrices per axis (half pixel or align
  corners source index; cubic convolution weights with a = -0.75 and clamped taps).
* Reflect, replicate and circular padding as constant index reads per axis; `pixel_shuffle`,
  `pixel_unshuffle`, `channel_shuffle` as reshape and permute; `im2col` (unfold) as strided slices,
  `col2im` (fold) as a transposed convolution with a one hot kernel.
* `lstm`, `gru`, `rnn_tanh`, `rnn_relu` (layers, directions, biases, batch first or not, unbatched)
  and their cells, unrolled over a fixed sequence length (TS861 for a symbolic one).
* `elu`, `selu`, `celu`, `leaky_relu`, `softplus`, `mish`, `log_sigmoid`, `hardshrink`,
  `softshrink`, `threshold`, `glu`, `prelu`, `log_softmax`, `expm1`, `log1p`: `expm1` and `log1p`
  are composites accurate near 0 (a series on [-1/2, 1/2], the direct formula outside; below
  4e-16 relative in float64 from -0.9 to 30 and at 1e-300), softplus is `max(z, 0) +
  log1p(exp(-|z|))`, finite for every z.
* `bilinear`, `cosine_similarity`, `pairwise_distance`, `broadcast_tensors`, `embedding_bag` (sum,
  mean, max; 2D bags or 1D indices with run time offsets; `padding_idx`, per sample weights).
* Losses: `mse_loss`, `l1_loss`, `smooth_l1_loss`, `huber_loss`, `cross_entropy_loss` (class
  indices with weights, `ignore_index` and label smoothing, or class probabilities),
  `nll_loss` forms, `binary_cross_entropy` (logs clamped at -100 as torch),
  `binary_cross_entropy_with_logits` (pos_weight), `kl_div` (log target or not).
* A creation op without `dtype=` (`full`, `zeros`, `ones`, `arange`, `scalar_tensor`) takes the
  dtype torch.export recorded, not the default dtype at import time (CLIP's causal mask was typed
  float32 in a float64 program).

A setting a lowering cannot express exactly is refused with TS860 (for example `avg_pool3d` with
`divisor_override`, `embedding_bag` with `include_last_offset`, recurrent projections), a size that
must be fixed with TS861; both name the op and the user's line.

Guarantees:

* **Unsupported ops fail loudly.** Every unknown op in the graph is listed in one error, each with
  its node and source line, before any conversion work. Nothing is ever skipped.
* **Unsupported settings fail loudly.** Batch norm or dropout in training mode (unless
  `training=True`, below), buffer mutation, attention dropout, transposed conv, string padding, dtype changing
  reductions and softmax all raise with the reason.
* **In place ops are checked.** `relu_`, `add_` and friends become pure ops only if nobody reads
  the pre mutation value later and the target is not a program input or weight; otherwise the
  import fails.
* **Shapes are derived, then cross checked.** Every output type comes from the IR's own rules and
  is compared with the shape torch.export recorded. A disagreement is an internal error, so a wrong
  shape rule cannot hide.
* **Symbolic batch.** Dynamic dims from `torch.export.Dim` keep their names (`B`). Size arithmetic
  in the graph (`sym_size`, `B*4`) is folded into dim expressions at import, so the program has no
  scalar shape ops at run time.
* **Control flow (2026-11-24).** `torch.cond`, `while_loop` and `scan` from `torch.export` become
  IR `cond` and `loop`. Each branch or body subgraph is converted by the same code as the program
  (every op inside is known, shape typed and listed with the rest when unknown), and the outer
  symbols carry into it. `scan` becomes a `loop` over the scanned dim that reads `xs[k]` with
  `select_dynamic` and writes the stacked outputs with `index_copy` (reverse scans arrive already
  flipped by torch). A Python number carried through control flow is refused with the fix (carry
  a tensor). Tests: tests/test_import_controlflow.py, both graph forms, both sides of every branch,
  data dependent trip counts, nested `cond`, a `cond` inside a `scan`, gradients of the imported
  `cond` and `scan` against torch.autograd (worst 0 relative error in float64 on the tested inputs,
  bar 1e-12).
* **Unbounded dims (2026-11-24).** A `torch.export.Dim` with no maximum imports as a symbol with no
  upper bound (`symbol B in [1, inf]`); torch marks it with its own `int_oo`, which crashed the
  range read before.
* **Shape checks against new shapes.** `import_exported(ep, input_shapes={"x": ["B", 4, 64, 64]})`
  re types the program for different input shapes; every mismatch is reported at import.

### Training graphs (added 2026-11-24, docs/RANDOM.md)

`ir.import_module(model, args, training=True)` (the model in training mode) imports the training
graph. The buffers it mutates (batch norm `running_mean`, `running_var`, `num_batches_tracked`)
become explicit state: program inputs after the user inputs and program outputs after the user
outputs, listed in `Program.state` as (name, input index, output index), starting values in
`Program.state_init` (`ir.initial_state(prog)` returns them in input order). Training batch norm and
dropout are compositions of existing ops (`ir/train_ops.py`), so autodiff, the compiler and the
serializer cover them with no new rule. Dropout reads the program input `Program.key` through a
`random` op at the site `<module path>/dropout#k`. Both graph forms are handled: the export graph's
`aten.batch_norm` with training=True and in place `num_batches_tracked.add_(1)`, and the core ATen
graph's `_native_batch_norm_legit_functional` with BUFFER_MUTATION outputs. resnet18 in training mode
matches PyTorch float64 to 1.0e-12 per step over 20 SGD steps (loss, gradients, weights, running
statistics; `scripts/train_mode_resnet18.py`).

### Reference interpreter (`reference.py`)

`ir.run(program, *inputs)` executes in float64 on CPU. Torch supplies only basic arithmetic
(elementwise math, sums, matmul, indexing). Composite ops are written from their textbook formulas,
not by calling the fused PyTorch op:

* `conv2d`: a sum over kernel taps of strided slices of the zero padded input times the weight tap
  (`einsum`), with groups as an explicit reshape. Never calls `F.conv2d`.
* `batch_norm`: `(x - mean) / sqrt(var + eps) * weight + bias`.
* `layer_norm`: mean and biased variance over the trailing dims.
* `softmax`: `exp(x - max) / sum`. `attention`: `softmax(q k^T * scale + mask) v`.
* `gelu` from `erf` (or the tanh formula), `hardswish`, `silu`, `sigmoid` from `exp` and `clamp`.
* `max_pool2d` as a running max over taps of a `-inf` padded input, `adaptive_avg_pool2d` with the
  floor and ceil window bounds.
* Added 2026-11-24: `avg_pool2d` as a sum over taps of a zero padded input divided per output cell
  by its window size (clipped to the padded input, or to the real input without
  `count_include_pad`); `adaptive_max_pool2d` as the max over the same windows as the adaptive
  average; `group_norm` and `rms_norm` from their formulas; `cumsum` as a running sum; `triu` and
  `tril` from index comparisons; `vector_norm` as `(sum |x|^p)^(1/p)`; `gather` and `index` with
  explicit coordinate grids and bounds checks; `pad` as a filled tensor with the input copied in.
  None of them calls the fused PyTorch op.

Before running, input shapes are matched against the declared types and every symbol is solved and
range checked. After every node the produced tensor's shape and dtype are compared with the
declared type, so a kernel that returns the wrong shape is an error. Scalar constants keep their
exact Python value, so the float64 run never inherits a float32 rounding.

## Results (2026-11-23, torch 2.11, torchvision 0.26, CPU)

Maximum relative error of the interpreter against the same model run by PyTorch in float64, with
random batch norm statistics and norm affine parameters so normalization is not near identity:

| Model | Export graph | Core ATen graph |
|---|---|---|
| resnet18 | 1.5e-15 | 1.5e-15 |
| mobilenet_v3_small | 9.1e-16 | 9.1e-16 |
| mobilenet_v2 | 5.8e-16 | 5.8e-16 |
| TransformerEncoderLayer (relu, post norm) | 2.7e-16 | 2.7e-16 |
| TransformerEncoderLayer (gelu, pre norm) | 1.6e-16 | 2.0e-16 |
| TransformerEncoder, two layers (gelu) | 4.9e-16 | 3.1e-16 |

Added 2026-11-24 (`tests/test_ir_ops.py`): a decoder only transformer written with plain
torch.nn (embedding, rotary positions from `arange`, `outer`, `cos` and `sin`, RMSNorm as
`x * rsqrt(mean(x^2) + eps) * w`, causal `scaled_dot_product_attention`, SwiGLU MLP, lm head tied
to the embedding) imports with `i64[B, T]` input, stores the tied weight once, and matches PyTorch
in float64 at (B, T) = (2, 8), (3, 5) and (1, 13) from one import: 1.8e-16, 1.9e-16 and 1.6e-16 in
both graph forms (114 nodes in the export graph, 220 in the core ATen graph). Every torchvision
classifier up to 100M params now exports (docs/EXPORT.md).

The tests assert below 1e-10. One import serves batch sizes 1, 2 and 5. A test model covering
dilated grouped conv, depthwise conv, ceil mode pooling, adaptive pooling to a non divisor size,
tanh gelu, relu6, hardtanh, slicing with a step, concatenation, reductions, scalar on the left
arithmetic, masked and causal attention also matches in both graph forms.

## What does not exist yet

* Higher order wrappers other than grad mode and disabled autocast regions (those are inlined since
  2026-11-24, see docs/MODEL_COVERAGE.md).
* Optimizers as IR are in `truestar.train` (docs/TRAINING.md). Training graphs import since
  2026-11-24 with `training=True` (below). Reverse mode autodiff exists as a program transformation:
  `truestar.ir.grad`, see docs/AUTODIFF.md.
* Importing `torch.map` (`map_impl`) and `associative_scan`. `torch.cond`, `while_loop` and `scan`
  import since 2026-11-24.
* Scatter with a tensor index other than `index_copy` and `scatter_add` (no importer mapping for
  `aten.scatter` or `aten.scatter_add` yet), string padding, quantized ops, custom ops. (1D, 3D
  and transposed convolution, 1D and 3D pooling, reflect, replicate and circular padding import
  since the op coverage work.) Reading with a boolean mask (`x[mask]`, a data dependent
  shape) is refused with a suggestion; the assignment `x[mask] = value` is supported (docs/LLM.md).
* Data dependent shapes (`nonzero`, boolean masking). Symbolic slices with symbolic bounds, ceil
  mode or adaptive pooling over a symbolic spatial dim.
* Reductions with `dtype=` (except `cumsum` to a float dtype or int64). `cast` exists since
  2026-11-24, but the interpreter still computes every float in float64; the declared dtype is kept
  for the compiler, not used by the reference (a cast to float16 does not round).
* Splitting or chunking along a symbolic dim into unequal pieces, or into a piece count that depends on
  the dim (equal pieces of a provable integer size, such as two halves of `2*T`, are supported since
  the decoder LLM work). Rolling along a symbolic dim.
* Symbol relations beyond what each rule proves locally. There is no global constraint solver; a
  rule that cannot prove two dims equal reports an error instead of guessing.
* A serialized format. Programs live in memory only.
* Any compiler or backend. (The native front end exists since 2026-11-24, see docs/FRONTEND.md.)

## Pain ledger mapping

| Row | How the IR serves it now | Still open |
|---|---|---|
| P9 shape errors found deep into a run | Shapes are types. A mismatch is reported at import or build time with the op, both shapes, the user's source line and module path. Tests cover bad input shapes, bad weights in a builder and unprovable symbolic equality | The native front end must report the line in the user's own typed code |
| P5 graph breaks and recompiles | One import covers every batch size in the declared range, checked by tests at batch 1, 2 and 5 | Control flow ops; whole program compile |
| P7 export bakes in batch size | Symbolic batch survives import with its name and range, and size arithmetic stays symbolic (`4*B`, `floor((T + 1)/2)`); every torchvision classifier up to 100M params imports (2026-11-24 sweep); `torch.cond`, `while_loop` and `scan` import (2026-11-24) | `torch.map`, `associative_scan` |
| Never introduce #3, no silent wrong results of our own | The reference interpreter is independent of fused PyTorch kernels and is the ground truth future compiled kernels are checked against. It verifies its own output types after every node | Compiled kernels do not exist yet |
| Never introduce #1, no silent fallback | Unknown ops and unsupported settings raise, all listed at once, nothing skipped | |
| Never introduce #6, no cryptic errors | Errors name the ATen op, the user line and module, not only IR names | Errors still mention IR op names next to the ATen name |

## Tests that prove the tests work

* `test_broken_reference_op_is_caught` replaces layer norm with an unbiased variance version and
  asserts the transformer comparison then fails the 1e-10 bar.
* `test_reference_checks_its_own_output_types` makes a pooling kernel return the wrong shape and
  asserts the interpreter raises.
* `test_broken_new_reference_op_is_caught` (2026-11-24) breaks three new reference ops (group norm
  with an unbiased variance, average pooling that always counts the padding, an index off by one)
  and asserts each case then fails the 1e-10 bar.
* Manual check on 2026-11-24: removing `eps` from the real reference `rms_norm` made
  `test_op_matches_pytorch_float64[rms_norm_module]` fail; the code was restored and all 44 tests
  in tests/test_ir_ops.py passed.
* Manual check on 2026-11-23: removing `eps` from the reference batch norm made 7 tests fail
  (resnet18, mobilenet_v3_small and mobilenet_v2 in both graph forms, plus the resnet18 symbolic
  batch test). The code was then restored and all tests passed.
