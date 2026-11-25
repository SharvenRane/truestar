# Truestar compiler

Author: Sharven Rane. First written 2026-11-24.

The compiler connects the two halves of Truestar that existed side by side: the shape typed tensor
IR (docs/IR.md, the importer, the native front end, the export format) and the kernel layer
(docs/KERNEL_IR.md: loop level IR, fusion, Triton and numba backends, verified kernels). It lowers
an IR program to fused kernels, runs control flow regions as compiled programs, and checks every
compiled program against the float64 reference interpreter the first time it runs.

Pain rows served: P5 (graph breaks and recompiles), P6 (compile time), P7 (the exported artifact
now runs compiled) and P8 (new layers fuse without templates). Never introduce items 1 (no silent
fallback), 3 (no silent wrong results of our own) and 5 (no compile time cliffs).

Code: `src/truestar/compiler/` (`lower.py`, `conv_rules.py`, `runtime.py`, `report.py`,
`executor.py`), tests in `tests/test_compiler.py` and `tests/test_conv_kernels.py`, benchmarks in
`scripts/bench_compiler.py` and `scripts/bench_conv.py`.

## Use

```python
import truestar
from truestar import Tensor, ops
from truestar.compiler import CompiledExecutor, compile_program

# 1. the executor seam of native functions
ex = CompiledExecutor("cuda", dtype="float32")           # or "cpu"; dtype float32, float16, bfloat16

@truestar.function(executor=ex)
def rnn(x: Tensor["B", "T", 16], wx: Tensor[16, 32], wh: Tensor[32, 32]) -> Tensor["B", 32]:
    h = ops.zeros((x.shape[0], 32))
    for t in range(x.shape[1]):          # a loop over a symbolic length
        h = ops.tanh(x[:, t] @ wx + h @ wh)
    if h.sum() > 0:                      # a branch on tensor data
        h = ops.relu(h)
    return h

rnn(x, wx, wh)                           # any B, any T: compiled once
print(ex.last.report)                    # what became a kernel and what did not

# 2. the export runner
from truestar import export
outs = export.load("model.tsx", executor="cuda")(x)      # "cuda", "cpu", "cuda:float16", "cpu:float16"

# 3. an IR program directly
cp = compile_program(program, "cuda", "float16")
outs = cp([x])
```

`truestar run model.tsx inputs.pt --executor cuda` does the same from the command line.
`truestar.frontend.set_default_executor(CompiledExecutor("cuda"))` makes it the default for every
native function.

Outputs come back on the executor's device in its float precision (the export runner then casts to
the declared dtype). Inputs may live on any device. Asking for `"cuda"` on a machine without CUDA
raises; there is no quiet move to the CPU. The `"cpu"` executor runs numba kernels: without numba
it raises `truestar.deps.MissingDependencyError` (code TS022) naming the fix, `pip install
"truestar[cpu-kernels]"`, and `truestar run --executor cpu` exits 2 with the same message. The
default `reference` executor needs no numba.

## How an IR op runs

Every op takes exactly one of these paths, and the compile report lists which.

| Path | IR ops | What runs |
|---|---|---|
| Truestar kernel (2026-11-24, for docs/NN.md) | `erf`, `log_softmax` (max, then the log of the shifted sum, one kernel), `amax`, `amin` | Kernel IR expression, as above |
| Truestar kernel | `neg exp sqrt rsqrt tanh relu sigmoid silu hardsigmoid hardswish clamp gelu` (exact and tanh), `cos sin log abs reciprocal` (2026-11-24), `cast` to a float the executor computes exactly, `pow` with a constant exponent of 2, 3, 1, 0.5, -0.5 or -1, `add sub mul div`, `eq ne lt le gt ge`, `logical_and logical_or logical_not`, `where`, `sum mean any`, `softmax`, `matmul`, `linear`, `batch_norm`, `layer_norm`, `attention`; since 2026-11-24 also `conv2d` (stride, padding, dilation, groups, depthwise), `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d` and the gradient ops `conv2d_input_grad`, `conv2d_weight_grad`, `max_pool2d_grad` | Kernel IR expression, fused, compiled by Triton (GPU) or numba (CPU) |
| Layout | `reshape permute expand select slice select_dynamic cat` | A strided view or a copy made by torch. No arithmetic. Kernels read any strides, so a view costs nothing |
| Constant | `full full_like arange` | The tensor is created by torch (`arange` with a symbolic end takes its length from the dim table). A float64 constant beyond the working dtype's range becomes that dtype's largest finite number of the same sign (2026-11-24: Hugging Face masks hold `torch.finfo(float64).min` and Mistral multiplies them by 0, where an infinity would give NaN) |
| Library kernel | `conv2d max_pool2d avg_pool2d adaptive_avg_pool2d`, only when asked with `library=(...)` or when the kernel rule cannot take this use of the op (a symbolic spatial size); the report gives the reason | `torch.nn.functional` (cuDNN on the GPU). Declared, listed in the report, checked like a kernel |
| Index library (2026-11-24) | `sort topk gather scatter_add index_select index_copy index` | torch on the executor's device (`IndexStep`): a stable sort (topk is a stable sort and a narrow, so ties come out in the reference's order), copies of entries, `scatter_add` under deterministic algorithms. Indices are clamped before the torch op and their range (and, for `index_copy`, uniqueness) is recorded on the device and read once at the end of the call: a bad index raises `IRError` and the results are not returned. Checked like a library kernel |
| Truestar gather kernels (2026-11-24) | `index index_select gather scatter_add argmax argmin` | Truestar's own kernels (`compiler/gather_native.py`): Triton on CUDA, numba on the CPU. One plan covers the three reads (any number of index tensors up to 3, adjacent or not, broadcast, negative entries, strided inputs, output rank up to 6); entries are moved as integers of their width, so the result is the input's bits. `scatter_add` copies the input, then one lane per column walks its index in order (the sums of repeated indices in index order, the same on every run and device); an index longer than 64 times the column count keeps the torch step (measured slower). `argmax` and `argmin` scan each row in order (first extreme, a NaN wins). The range check is done in the kernel and read once at the end of the call, like the torch step's. The step's report entry says `truestar <op> (triton)` or `(numba)`; the torch step remains on Metal, for integer argmax, for half precision on the CPU, for index tensors that are masks, and the entry counts any call that took it. Speed and the case list: `scripts/bench_gather_native.py` |
| Integer device (2026-11-24) | `add sub mul neg abs remainder fmod maximum minimum where` with integer results | The reference implementation itself on the device's integer tensors (plain torch integer arithmetic, exact). Checked with zero tolerance |
| Host scalar | `dim_scalar`, integer scalar arithmetic (a loop counter) | Exact integer work on the CPU |
| Control flow | `cond loop` | The predicate or trip count is read on the host; the regions are compiled programs |
| Fallback | any op, or any use of an op, with no kernel rule | The float64 reference implementation of that one op on the CPU, result cast back. Correct and slow. Listed with the reason |

An op with no kernel rule and no reference implementation stops the compile with
`UnsupportedOpError`. `CompiledExecutor(..., allow_fallback=False)` turns a needed fallback into a
compile error (`FallbackNotAllowed`) that lists the ops.

Fallbacks that exist today:

* `layer_norm` over more than the last dim on the GPU (two nested reductions fit no GPU loop shape).
* `matmul` or `linear` with a one dim operand.
* Integer results of ops outside the integer device list (the kernel IR has float buffers only):
  casts to an integer or bool dtype (BERT's and T5's masks and buckets), `triu`, `tril`, `cumsum`
  and `argmax` of integers, and so on.
* A cast that rounds to a 16 bit dtype other than the executor's (the reference rounds; a copy
  would not). Found by the first call check when the cast rule was first written as a copy.
* `pow` with any other exponent.
* `cat` of mixed dtypes, and a `cond` or `loop` whose region uses a dim symbol the program inputs
  do not bind (the whole node then runs in the reference interpreter).
* Ops added to the IR after this was written run through the fallback until they get a rule, and
  show up in the report the day they are imported.

## Lowering rules

A rule turns one IR op into an index function: given one kernel IR index per output dim, return the
expression for that element. This is the lowering contract of docs/KERNEL_IR.md section 4. Details
that matter:

* **Broadcasting** follows the IR types. A dim that is `1` in an operand and larger in the result is
  read at constant index 0.
* **Symbolic dims.** Each distinct dim expression (`B`, `4*B`, `T`) becomes one kernel symbol, bound
  from the buffer shapes at call time. Nothing in a kernel knows a batch size.
* **Scalar constants** are inlined with their exact Python value, so `x * 0.1` does not load a buffer.
* **Reductions over several dims** become one single axis reduction per dim, last dim first, because
  the GPU tile shape reduces over axes of one size. `mean` divides by the dim values afterwards.
* **Batch norm** with constant statistics is folded at compile time (in float64) into
  `x * scale[c] + shift[c]`. With computed statistics the textbook formula is emitted.
* **Convolution and pooling** (`conv_rules.py`, 2026-11-24) are their textbook formulas with
  strided load indices (docs/KERNEL_IR.md, strided indices): `conv2d` is
  `b[o] + sum over c, u, v of x[n, g(o) Cg + c, i s + u d - p, j s + v d - p] * w[o, c, u, v]` with
  zero padding as the load's pad value. The input read at strided indices is always a buffer (a pad
  value belongs to a stored tensor), so it is materialized and its producer keeps any elementwise
  chain as an epilogue. Whatever reads the conv (bias, a folded inference batch norm, relu,
  hardswish, a residual add) becomes the conv kernel's epilogue: conv plus batch norm plus add plus
  relu is one kernel, one store. A constant weight is stored once at compile time with the output
  channels contiguous (same logical shape), which the implicit GEMM reads along contiguous memory.
  Pooling: `max_pool2d` pads with minus infinity, `avg_pool2d` divides by a per cell divisor
  computed once at compile time from the reference's formula, `adaptive_avg_pool2d` starts each
  window at the strided index `(i H) // OH` with per row and column weights computed once.
* **Gradients of convolution and pooling** are three IR ops the autodiff rules emit
  (docs/AUTODIFF.md): `conv2d_input_grad` is the transposed convolution in gather form,
  `x_bar[n, c, h, w] = sum over o, u, v of g[n, o, (h + p - u d) / s, ...] * w[o, c, u, v]` with the
  division exact (no zero inserted copy of g is ever made); `conv2d_weight_grad` sums
  `x[n, c, i s + u d - p, ...] * g[n, o, i, j]` over the batch and the output pixels;
  `max_pool2d_grad` has every input cell gather the cotangent of each window whose first maximum
  (row major, PyTorch's tie rule) it is.
* **Attention** is three lowered values: scores (a dot product, scaled, with the causal or user mask
  applied), softmax weights, and the product with the values. On the GPU fusion joins them into the
  online softmax attention shape; on the CPU they stay three kernels.
* **Bool values** are expressions inside a kernel. One that must become a tensor is computed as 0 or 1
  in the float dtype and cast. Bool and integer tensors that a kernel reads are cast once by an
  adapter step (integers to float32, exact up to 2^24).
* **Scalars** are computed and loaded as one element tensors, which works around two rank 0 gaps in
  the kernel backends.

## Fusion

Fusion decides which lowered values become buffers. Everything else is inlined into its reader.

1. Materialized from the start: program and region outputs, anything read by a step that is not a
   kernel, and reductions with more than one reader.
2. For each materialized value the fused kernel is built and the backend scheduler
   (`triton_backend.plan`) is asked whether it fits a loop shape (tile, contraction, attention).
   If not, a producer is materialized and the question is asked again. The producer chosen is the
   top of the elementwise chain above the nearest reduction for which that chain still fits, so
   the chain becomes the epilogue of the reduction (computed once) and not the prologue of its
   reader. `linear -> gelu -> linear` therefore becomes `[linear + gelu]` and `[linear]`.
3. An elementwise value with several readers is inlined when all readers end up in one kernel
   (equal subexpressions are shared there) and materialized when two kernels would each recompute it.
4. The CPU backend takes any kernel, so there the rule is about cost: a reduction is never inlined
   under another reduction and nothing is inlined under a matrix product, because the numba loop
   nest would redo that work for every element of the outer loop.

Examples from the test suite and the benchmark (GPU):

| Program | Kernels |
|---|---|
| MLP (3 linear, gelu, relu, softmax) | `[linear + gelu]`, `[linear + relu]`, `[linear]` contraction; `[softmax]` tile |
| TransformerEncoderLayer | `[linear]`, `[attention]` (one kernel), `[linear]`, `[add + layer_norm]`, `[linear + relu]`, `[linear + add]`, `[layer_norm]` |
| resnet18 (since 2026-11-24) | 20 conv kernels of `conv2d + batch_norm`, `conv2d + batch_norm + relu` or `conv2d + batch_norm + add + relu`, a window kernel each for `max_pool2d` and `adaptive_avg_pool2d`, one `linear`; 0 library kernels (it was 18 Truestar kernels plus 22 library kernels) |
| mobilenet_v3_small | 41 conv, 21 window (depthwise convolutions with their batch norm and hardswish, pooling), 9 tile, 2 contraction; 0 library kernels |
| resnet18 backward (gradient program of the input and every conv weight) | 139 kernels: 60 conv (forward, input and weight gradients), 3 window, 71 tile, 5 contraction; 0 library kernels |
| RNN body | `[matmul]`, `[matmul + add + tanh]` |
| Newton loop condition | `[mul + sub + neg + lt + where + gt + any]` |
| Differential attention written with plain ops in the native front end, `(softmax(q1 @ k1.T * c) - 0.6 * softmax(q2 @ k2.T * c)) @ v` | one attention kernel holding 3 `matmul`, 2 `softmax`, 3 `mul`, 1 `sub` |

## Control flow and symbolic dims

`cond` and `loop` regions are lowered like programs, once, when the parent is compiled. At run time
a `cond` reads its predicate on the host and runs one compiled branch. A `loop` runs its compiled
body `trip_count` times (the count is a dim, read from the evaluated dim table) or while its
compiled condition region returns true.

One compile serves every value of the symbolic dims:

* Dim expressions the runtime needs as numbers (reshape targets, trip counts, `dim_scalar`) are
  evaluated with sympy once per distinct set of input shapes and then looked up.
* Kernels bind their symbols from the buffers they receive.
* `tests/test_compiler.py` counts programs, regions and kernels compiled, and kernel cache traffic,
  across batch 1, 7 and 64 and across loop lengths: the counts do not move after the first call.
  Shapes outside the declared symbol ranges raise `ShapeError`; they never recompile.

Two changes in the Triton backend were needed for this to hold below Truestar as well. Triton by
default compiles a new variant when an integer argument is 1 or changes divisibility by 16, and the
tile kernel's block row count was a compile time constant that followed the row count. Symbolic
sizes are now excluded from Triton's specialization and the block row count is fixed when the row
count is symbolic. Before: 1.0 to 1.7 s of Triton compile at each of batch 1, 7 and 64 on the
benchmark MLP. After: about 1 ms.

A `while` loop whose condition needs more precision than the working dtype can never turn false
(Newton's method to a 1e-4 residual in float16). From iteration 256 on, the loop checks whether the
carried values repeat exactly and raises an error that says so instead of spinning.

## Dtypes

The executor has one float precision: `float32`, `float16` or `bfloat16`. Every float value of the
program is computed in it (weights are cast once at compile time, float inputs at call time). Bool
and integer tensors keep their dtype. The reference interpreter stays float64, which is what the
checks compare against. On the CPU the numba kernels compute in float64 and round on store, and
torch has no half precision convolution there, so library kernels compute in float32 and cast.

**Mixed precision islands** (2026-11-24). In a `float16` or `bfloat16` program, a float value the
model declares in a wider dtype than the program's own (at least 32 bits: an explicit `.float()` in
an import of `model.double()`, or in a model imported in float16) keeps float32 storage and math;
it is rounded to the working precision only where the model casts back (that cast is stored, so it
rounds there). A param read through such a cast is stored in float32 (a rotary `inv_freq`: in
float16 its rounding alone moves angles at position 4000 by up to 2 rad). The program's own dtype
is the float dtype holding the most elements among its params and inputs (a symbolic dim counts as
1), so float32 scales next to float64 weights are islands, not the program's dtype. `report.islands` (and the text
report of a 16 bit program) lists every island (value, dtype, shape). A program without wider
declared values has no islands and exactly the kernels it had; a float32 program is unchanged. The
checks judge each kernel at the lowest precision among its buffers, as before, so no tolerance
changed. Limits: a control flow region takes its own dtype from its inputs; library and random
steps still produce the working precision.

**Float64 islands in float32 programs** (2026-11-25, P63). In a `float32` program whose model is
imported in float32, a value the model declares float64 (an explicit `.double()`) keeps float64
storage and math the same way (`lower.island_dtype`); its kernel has a float64 output and so float64
math (`triton_backend.Emitter.cdt`), and it is stored where the model casts back, so it rounds to
float32 once. The use is rotary angles at long positions: in float32, position times inverse
frequency is off by up to half an ulp of the product (1.2e-4 rad at position 2100, 1e-3 rad at
32000), which moved the new key row 4.3e-5 to 4.9e-4 off the float64 reference and failed the end to
end check of a decode at position 2100 (8.6e-4 against the tolerance 6.3e-4; found by the prefix
cache check). With `Cfg(rope_f64=True)` and `import_steps(..., trace_dtype=torch.float32)` in
examples/llama_paged_decode.py the key row stays at 2.4e-7 to 4.7e-7 from position 100 to 32000 and
the check passes (`scripts/rope_long_positions.py`). An import of `model.double()` compiled in float32
has float64 as every value's dtype and so no islands: unchanged, as is every program without a
float64 declaration (the 9 compiled hashes of scripts/laptop_fingerprint.py and the 67 deterministic
fingerprint rows bitwise equal before and after).

**Float32 matmul precision follows PyTorch** (since 2026-11-24). With `allow_tf32=None`, the
default of `CompiledExecutor`, `compile_program` and the export runner, every call reads
PyTorch's float32 matmul precision (`torch.backends.cuda.matmul.fp32_precision`, which
`torch.backends.cuda.matmul.allow_tf32 = ...`, `fp32_precision = ...` and
`torch.set_float32_matmul_precision(...)` all set; "high" and "medium" both mean TF32 on CUDA, as in
PyTorch) and runs the variant of the program compiled for it. The first call under a new setting
compiles and verifies that variant (`STATS["programs"]` counts it, like a guard driven recompile
in torch.compile); switching back costs nothing. `allow_tf32=True` or `False` pins the choice.
TF32 covers float32 dot products and, as before, cuDNN convolutions (Truestar does not read
`torch.backends.cudnn.allow_tf32`); the tolerance of the affected checks widens to the TF32 level.
PyTorch's own default is strict float32 matmuls and TF32 convolutions, which is why the resnet18
benchmark has two Truestar rows. The compile report states the precision and whether it was
followed or pinned. Float16 and bfloat16 programs ignore the setting.

**Deterministic mode.** `deterministic=None` (default) follows
`torch.are_deterministic_algorithms_enabled()` the same way; `True` or `False` pins it. Every run is
deterministic in both modes (the same inputs give the same bits). The deterministic variant never
splits K, so a contraction's result is also independent of the batch size it runs in
(docs/KERNEL_IR.md section 6b).

## Verification

Three layers, all on by default (`TRUESTAR_VERIFY=first`):

1. **Kernel level** (from the kernel layer): every compiled kernel is compared with the float64
   NumPy interpreter of its own kernel IR on its first call per input shape signature.
2. **Region level**: on the first run of every program and every region, each kernel group and each
   library step is compared with the float64 reference implementation of the IR ops it covers,
   computed from the inputs that step really received. Inputs are identical on both sides, so each
   region's own error is measured, not the error it inherited. This is the check that covers the
   lowering rules: a kernel that faithfully computes the wrong expression passes layer 1 and fails
   here. Tolerance is the kernel layer's (float32 2e-5, float16 2e-3, bfloat16 1.6e-2, times the
   reduction growth factor); bool results must match exactly.
3. **End to end**: the first run of the whole program is compared with `ir.run` on the same inputs.
   This covers what layer 2 does not look at: layout steps and the wiring between regions.
   Tolerance: the base tolerance times `4 * sqrt(number of checked steps)`, capped at 10%.
   **Ill conditioned outputs (2026-11-24).** Some outputs are far from float64 in every correct
   float32 execution: AdamW's first step divides each gradient entry by its own magnitude, so an
   entry that cancels in its float32 sum flips the update (measured: eager PyTorch float32 is
   6.8e-4 and 2.6e-3 from float64 on two MLP weights, over this bar). **The default stays strict:**
   such an output is refused as before, and when a float32 output fails the bar the program is
   evaluated once more by the reference interpreter computing in float32 (the IR's own reference
   implementations run by torch's float32 kernels, accumulation included; never the compiled
   kernels, so a kernel bug cannot vouch for itself), and the error says how far that independent
   float32 evaluation is from float64:

   ```
   output %0.weight_new: relative error 0.000813 > tolerance 0.000625
   the reference interpreter in float32 (an independent float32 evaluation of the same program) is
   itself 0.000715 off float64 on this output, so the output is ill conditioned in float32;
   accept_ill_conditioned=True accepts outputs within 4x of that distance (TS782, docs/COMPILER.md).
   ```

   **Opt in:** `compile_program(..., accept_ill_conditioned=True)` (also on `CompiledExecutor` and
   `CompiledTrainStep`) accepts an output whose error is at most 4 times that distance. Every
   accepted output is listed in `cp.conditioning` (output, error, bar, float32 reference error) and
   named with its numbers in a `TS782` warning (`IllConditionedWarning`). A wiring fault is orders of
   magnitude beyond that: `tests/test_train_compiled.py` corrupts one output of an AdamW step by 1%
   after the region checks and it is refused with the option on. float16 and bfloat16 programs keep
   the strict bar whatever the option says.

   **Judged by the derived bound (2026-11-25).** The fixed bar knows nothing about the program: it
   refused 11 of the fair benchmark's 12 correct float32 training steps and passed a wrong one (AdamW
   with the wrong weight decay sign is 2.0e-5 off, within the bar of 5.3e-4). An output over the bar
   is now judged by the bound derived from the program itself (truestar.tolerance, error model runs
   propagated op by op through the forward, the backward and the optimizer; docs/TOLERANCE.md, The
   compiled executor's first call checks), and a call that writes the caller's state in place
   (`update=` or `donate=True`: a training loop's step) judges every float output by that bound
   alone (`end_to_end="auto"`, the default; `"bar"` and `"derived"` choose one rule for every call).
   Accepted outputs over the bar are listed in `cp.derived_accepts`; a refusal says how many elements
   are beyond their derived bound and by how much. `accept_ill_conditioned` is no longer needed for
   correct AdamW steps; it still works as described above.

Layer 1 also tests the sign of the errors (docs/BIAS.md): an output whose errors are within the
tolerance but lean one way beyond what unbiased round to nearest allows (cluster robust t over the
groupings the kernel IR implies, p below 1e-6, mean above 1/64 of the tolerance) is refused with TS960;
a TF32 kernel's lean is reported as TS966 and used. Layers 2 and 3 test size only.

Layers 2 and 3 measure each element against its own magnitude, floored at 4x the output rms
(layer 1 uses 1/8 or 2x, by whether the kernel reduces): docs/KERNEL_IR.md, Verification. All
three run on first use in each process (a kernel per shape signature, a program and each region on
their first run), not on every call; what that misses is listed there.

**Where the float64 reference runs (since 2026-11-24).** For a CUDA program, layers 2 and 3 run on
the program's own GPU when the check is big enough, with the rule of the kernel layer
(`TRUESTAR_VERIFY_DEVICE`): `auto` (default) moves a region check to the GPU when its work (largest
value, or a contraction's output times its summed length) reaches 2^20 elements, `gpu` moves every
check of a CUDA program, `cpu` keeps them all on the CPU. The end to end check runs on the GPU when
a region check of the same call did (or in `gpu` mode). Nothing else changes: the same reference
implementations (`ir.reference`, run under `torch.device`), the same tolerances, the same
schedule, comparison by `kernel.verify.compare_torch` with the rules of `compare_numpy`. A check whose
estimated float64 working set exceeds `TRUESTAR_VERIFY_GPU_MB` (default 2048), one that hits a
CUDA out of memory error, every check of a CPU program and the op isolation after a mismatch stay
on the CPU. On the GPU, values are freed after their last reader and attention with a large score
matrix runs in blocks of query rows (same formula per row); an op torch lacks on the device in
float64 would run on the CPU for that node only (`ir.reference.device_cpu_fallbacks`; none on
torch 2.11 with CUDA 12.8). A mismatch found on the GPU recomputes every value of the region there
and names the op exactly as the CPU path does (tests/test_region_gpu_verify.py).

A region inside a branch that the first inputs did not take is checked the first time it runs,
even on a later call. Every loop body is checked on its first iteration.

**An end to end mismatch at a near tie (2026-11-24).** A float comparison whose operands are within
rounding of each other (a relu mask whose input is 1e-8 of its rms) can go either way in a
correct float32 run, and everything downstream of it moves. The region checks record the decision
every top level comparison took in this run (`gt`, `ge`, `lt`, `le`, `eq`, `ne` on float operands,
from the values the step really received). When the end to end check fails, the comparisons decided
the other way than float64 are listed, and the float64 reference is run again taking this run's
decisions (`ir.reference.run(..., overrides=)`). The error says the output is **ill conditioned at a
near tie** only when that replay matches every output within the end to end tolerance and every element
decided the other way has operands within the tolerance times their rms of each other; it names the
comparison, the source line, the element and its float64 operands, and sets `error.ill_conditioned`
and `error.near_ties`. Otherwise it says the decisions do not explain the error and keeps the wiring
fault wording. The program is still refused: the result is not returned. Found on
`scripts/bench_memory.py`'s language model (6 layers, d 256, batch 16, seq 128, CUDA, float32):
layers.1 linear1 at [11, 23, 895] is -1.28e-08 in float64 (the sum of its terms' magnitudes is 6.5,
so one float32 rounding of them is 3.9e-7), +5.96e-08 in eager PyTorch float32 on the CPU and positive
in the compiled program, -2.0e-07 in eager PyTorch on CUDA. The compiled program and eager PyTorch CPU
are both 0.00754 off float64 on `layers.1.linear1.weight`'s gradient (bar 0.00153); eager CUDA is
7.3e-7 and the reference interpreter in float32 5.3e-7 (neither flips). With the compiled run's one
decision the float64 reference matches every output to 1.72e-6. Tests:
`tests/test_near_tie_diagnosis.py`.

A mismatch raises `ProgramMismatchError` and the result is not returned. The error names the region
(`rnn/loop %h/body`), the kernel and every op fused in it with its source line. To name the single
op, each op of the region is then compiled alone, fed the reference inputs and compared: the first
one that is wrong by itself is the op in the message (`error.op`). If every op matches alone, the
message says the fault is in how they were fused. A program that failed is checked again on the
next call; it is never marked verified.

```
truestar.compiler.runtime.ProgramMismatchError: compiled program 'guarded' on cuda (float32) does not match
the float64 reference interpreter.
  region guarded, kernel 'guarded:%relu' covering linear (%linear), relu (%relu)
  value %relu: relative error 0.12 > tolerance 2e-05
  op: relu. Compiled alone on the reference inputs, this op is wrong by itself (relative error 0.12 > tolerance 2e-05).
The result was not returned. This is a bug in the Truestar compiler, not in your program. Set TRUESTAR_VERIFY=0 only to debug.
```

`verify="all"` (or `TRUESTAR_VERIFY=all`) checks every call. `TRUESTAR_VERIFY=0` turns the checks
off and is for debugging only.

## Sentinels (compiled in runtime checks, 2026-11-24)

Research item R13 (GAP_PYTORCH_JAX.md). Rows served: never introduce #6 (no cryptic errors) and
P10's spirit (name the line); a NaN in a compiled program used to come out as a NaN output, or, on
the first call, as a `ProgramMismatchError` that called a float32 overflow a code generator bug.

```python
ex = CompiledExecutor("cuda", sentinels=True)            # nan and div; "all"; or "nan,inf,overflow,div"
cp = compile_program(program, "cuda", "float16", sentinels="all")
fast = CompiledExecutor("cuda", sentinels=True, sentinel_read="deferred")   # read the flags at a later call
# fast.last.read_sentinels() waits for pending calls and raises if one fired
# or TRUESTAR_SENTINELS=nan,div for every compiled program without a checks argument
```

Off by default. A fired check raises `truestar.compiler.SentinelError` (a subclass of `IRError`). With
`sentinel_read="call"` (the default) the results of the call are not returned; with
`sentinel_read="deferred"` the error comes from a later call (or `read_sentinels()`), names the call
whose flags fired, and that call's results were already returned:

```
truestar TS500 nan sentinel: sqrt (%sqrt) produced NaN.
  at: model_nan.py:11 in f: a = ops.sqrt(x - 0.5)
  first offending element: index (2, 17) of %sqrt (value nan); 1 of 160 entries
  step: f:%add, fused ops: sub + sqrt + softmax + mul + add; region f
  program 'f' on cuda (float32). The results of this call were not returned.
```

`SentinelError.to_dict()` gives the same fields for agents (docs/AGENT_INTERFACE.md).

| class | code | what fires it | where it is checked |
|---|---|---|---|
| `nan` | TS500 | a stored value is NaN; an input holding NaN (named as the origin when its first reader fires); a param holding NaN (once) | in the kernel at the store; for torch run steps, where a kernel loads their output, else one multi tensor norm per call |
| `inf` | TS501 | a stored value is infinite and no infinite constant explains it | in the kernel at the store (float32 registers) |
| `overflow` | TS502 | a float16 or bfloat16 value is finite in float32 and Inf once stored | in the kernel, before and after the narrowing |
| `div` | TS503 | a denominator is exactly zero, anywhere in a fused expression; an integer `remainder` or `fmod` by zero | in the kernel at the division; one reduction for integer steps |
| `bounds` | TS504 | an index out of range, or repeated where it must be unique | always on, every compiled program (`gather`, `index_select`, `index_copy`, `scatter_add`, `index`, `select_dynamic`) |

### How it works

* **Detection is a flag, not a value.** Each kernel compiled with checks takes four int32 flag
  words and stores 1 into a word with a store predicated on the bad entries themselves, so a clean
  run writes nothing. The words live in pinned host memory that the kernels write through the
  unified address space, so zeroing them is a memset and reading them needs no launch and no copy.
  The output of a step that runs through torch (a convolution, the fallback) is checked by the
  kernels that load it, the same predicated way; outputs that something else reads are reduced
  together by one `torch._foreach_norm` per call (the 2 norm is NaN exactly when an entry is NaN).
  JAX's `checkify` instead threads an error value out of the function, which the caller must
  unpack and `throw()`.
* **Convolution and pooling kernels** (conv and window shapes) flag the same way at the store, and
  also on the convolution's sum itself before the epilogue (and on each summed window): on the GPU a
  fused relu's maximum returns 0 for NaN, so an infinite input times a zero weight would otherwise
  leave no trace. Strided reads check only elements really read, never a pad value (max pooling's
  minus infinity outside the input cannot fire `inf`). Tested in tests/test_conv_kernels.py.
* **One wait per call, or none.** `sentinel_read="call"` waits for the call's event before returning
  and reads every word; `sentinel_read="deferred"` only records the event, and a later call (or
  `read_sentinels()`) reads the words of the calls that finished. A ring of buffers lets calls overlap.
* **Attribution only on failure.** When a flag fired, the call runs again with a read after every
  step (the kernel layer's own check off), so the first step that produced a bad value in time is
  found, inside loops too (`iterations`). Its fused IR ops are then evaluated in float64 on the
  inputs the step really received (the region check's reference), and the first op whose own
  result is bad is named, with its source line and the first offending element of its result. A
  NaN that only exists in float32 (an infinity minus an infinity) is traced to the op whose float64
  value left the float32 range. Classes are reported in the order div, overflow, inf, nan, since a
  zero denominator explains the NaN after it.
* **No false alarms from masks.** `kernel.checks.may_be_inf` follows deliberate infinities (an
  infinite constant, a `full(-inf)`, a param holding infinities) through the kernel IR with their
  sign. `exp(-inf)` is 0 and a softmax shift `s - max(s)` is at most 0, so a masked softmax or
  attention kernel stays armed; a kernel whose stored value can hold such an infinity has only the
  `inf` class removed (listed in the report as `checks_disarmed`). Denominators that cannot be zero
  (constants, sizes, a sum of exponentials) are not checked.
* **Precision events are named before mismatches.** With checks on, a first call whose region check
  fails because of a NaN or an overflow of the working precision reports the check, not "a bug in
  the Truestar code generator".
* `torch.autograd.detect_anomaly` only checks what backward functions return, so a NaN produced in
  the forward whose gradient is finite (log of a negative number: the gradient 1/x is finite) is
  never reported, and under `torch.no_grad` nothing is checked; it records a Python traceback for
  every op of the forward.

### Measured (2026-11-24, `scripts/bench_sentinels.py`)

RTX 5070 Ti shared with other jobs, torch 2.11 with CUDA 12.8, Windows. MLP 1024 to 2048 x3 to
1000 at batch 64, `TransformerEncoderLayer` d 256, 4 heads, seq 128 at batch 64, resnet18 at
224 px batch 16. Every variant is built first and the variants are timed in turn in each of 7
rounds; each number is the fastest round. "Wall" is N calls then one synchronize; "GPU" is one call
with a sleep kernel queued first, so it is the call's GPU busy time without host time (for the
`call` read mode the read itself waits for the GPU, so its GPU column also holds that wait).
Verification after the first call is off on the Truestar rows, as the PyTorch rows run unchecked.

| model | variant | wall ms | vs no checks | GPU ms | vs no checks |
|---|---|---|---|---|---|
| MLP | torch eager forward | 0.239 | | 0.234 | |
| MLP | torch eager forward, `detect_anomaly` | 2.842 | +1091% | 0.231 | -1% |
| MLP | torch eager forward, isfinite hook per module | 1.228 | +415% | 1.261 | +439% |
| MLP | torch eager train step, `detect_anomaly` | 4.765 | +812% | 2.016 | +348% |
| MLP | Truestar forward, no checks | 0.258 | | 0.254 | |
| MLP | Truestar, nan and div, read per call | 0.306 | +18.8% | 0.270 | +6.1% |
| MLP | Truestar, no checks plus one host read per call | 0.365 | +41.5% | | |
| MLP | Truestar, nan and div, deferred read | 0.263 | +2.0% | 0.256 | +0.7% |
| MLP | Truestar, all four classes, deferred read | 0.271 | +5.0% | 0.266 | +4.5% |
| MLP | Truestar float16, all, deferred read | 0.118 | +27.4% | 0.043 | +10.6% |
| MLP | Truestar train step, nan and div, read per call | 1.355 | +15.1% | 1.137 | +6.7% |
| transformer | torch eager forward | 0.840 | | 0.815 | |
| transformer | torch eager forward, `detect_anomaly` | 6.808 | +710% | 0.815 | 0% |
| transformer | torch eager train step, `detect_anomaly` | 14.057 | +436% | 7.342 | +188% |
| transformer | Truestar forward, no checks | 0.894 | | 0.889 | |
| transformer | Truestar, nan and div, read per call | 1.028 | +14.9% | 0.963 | +8.4% |
| transformer | Truestar, div only, read per call | 0.968 | +8.2% | 0.914 | +2.8% |
| transformer | Truestar, all, deferred read | 0.977 | +9.2% | 0.937 | +5.4% |
| transformer | Truestar float16, all, deferred read | 0.305 | +9.8% | 0.254 | +8.8% |
| transformer | Truestar train step, nan and div, deferred read | 3.541 | +9.8% | 3.427 | +8.1% |
| resnet18 | torch eager forward | 2.833 | | 2.823 | |
| resnet18 | torch eager forward, `detect_anomaly` | 24.479 | +764% | 2.809 | -1% |
| resnet18 | torch eager forward, isfinite hook per module | 13.517 | +377% | 12.643 | +348% |
| resnet18 | torch eager train step, `detect_anomaly` | 40.223 | +322% | 20.681 | +119% |
| resnet18 | Truestar forward, no checks | 3.153 | | 3.029 | |
| resnet18 | Truestar, nan and div, read per call | 3.289 | +4.3% | 3.236 | +6.8% |
| resnet18 | Truestar, nan and div, deferred read | 3.240 | +2.7% | 3.124 | +3.1% |
| resnet18 | Truestar, all, deferred read | 3.307 | +4.9% | 3.168 | +4.6% |
| resnet18 | Truestar float16, overflow, read per call | 1.882 | +4.0% | 1.813 | +1.6% |
| resnet18 | Truestar train step, nan and div, deferred read | 48.672 | +0.8% | 47.278 | +0.5% |

Per class, read per call, wall time over no checks: nan +25.3%, inf +30.3%, div +23.4% (MLP);
nan +16.2%, inf +13.0%, div +8.2% (transformer); nan +4.0%, inf +5.1%, div +2.5% (resnet18);
overflow in float16 +52.9% (MLP), +52.5% (transformer), +4.0% (resnet18). Full rows, with
every class in both read modes, come out of the script.

* **Against `detect_anomaly`:** it costs 7x to 12x the forward wall time and 3x to 8x a training
  step, and its GPU time in the forward is unchanged: the cost is the Python traceback it records
  for every op. Truestar with nan and div checks costs 0.8% to 15% of wall time with the deferred
  read and 4% to 19% with the read per call.
* **Where the remaining cost is.** On the MLP the in kernel flags cost under 1% of GPU time
  (per kernel GPU time 0.0599 ms without and 0.0590 ms with nan checks, profiled). The cost of
  the read per call is the host waiting for the GPU before it can return: 0.107 ms per call on
  this machine (Windows), 41% of a 0.26 ms MLP call and 1.6% of a resnet18 call. The deferred read
  removes it; small half precision programs (0.04 ms of GPU work) still pay about 10% for the
  extra compares in their epilogues.
* **The 3% target of the redesign is met** for nan and div with the deferred read on the MLP
  (+2.0%), resnet18 (+2.7%) and its training step (+0.8%); not on the transformer layer (+9% to
  +15%, of which 3% to 8% is GPU time), on float16 MLP and transformer programs (+10% to +27%),
  or with the read per call on programs under 1 ms.
* **What it took** (each measured on the way): a block reduction per class before the flag store
  made float16 contraction kernels several times slower (the accumulator changed layout), so the
  store is predicated on the bad entries instead; flags on the device needed a zeroing and a copy
  per call (0.055 ms of GPU time on the MLP), so they live in pinned host memory that the kernels
  write directly; one reduction per convolution output cost resnet18 12% of GPU time, so torch
  step outputs that only kernels read are checked where those kernels load them, and the rest are
  reduced together by one multi tensor norm per call.

Injected faults (`scripts/bench_sentinels.py --faults`, CUDA, `sentinels="all"`): each class is caught
and named with its line and first element. log of a negative entry: TS500 at `log`, index (7, 11);
exp overflowing float32: TS501 at `exp`, index (3, 5); a float16 product too large: TS502 at
`mul`, index (9, 0); a row that sums to zero: TS503 at `div`, index (12, 0). The same log fault
under `torch.autograd.detect_anomaly` with a backward pass raised nothing (the gradient of log at
a negative number is finite), and under `torch.no_grad` nothing is checked at all. An index out of
range is TS504 with the line and the bad index in every compiled program (tests/test_sentinels.py).

Not verified: the CPU backend's overhead (its checks are correct in the tests and it is the
correctness backend); ROCm and other GPUs; Linux, where a host read per call may cost less than on
this Windows machine.

## Transformer inference speed (2026-11-24, P38)

Four changes, each kept only where it measured faster, none of them weakening a check. Numbers and
the method are in docs/QUALITY_BAR.md (status log, P38); every number there is provisional until
the fair rerun on an idle GPU.

**Batch axes folded into contraction rows.** A linear layer on `[B, T, C]` activations is a batched
contraction with `M = T` (or `M = B` with the tokens as batch). When the right factor (the weight)
does not depend on the batch axes, the plan folds them into the rows (`ContractionPlan.fold`): one
program grid over `B x T` rows instead of one grid per batch entry. A decode step (`T = 1`) used to
run one 16 row tile per sequence, 15 rows of it padding, each rereading the whole weight; folded,
the weight is read once per row tile. Measured at decode batch 8 in float32: the MLP projections
went from 75 to 80 GB/s to 330 to 440 GB/s. Same arithmetic per output element (the K loop is
unchanged), so results are within the same float64 checks; the Triton source changed, so
`CODEGEN_VERSION` and the tuning search space version were bumped.

**cuBLAS as one more implementation of a float32 contraction (`kernel/library.py`).** Triton's IEEE
float32 `tl.dot` runs on FMA units at about 19 TFLOPS here where cuBLAS reaches about 27, and in
strict float32 (PyTorch's default) the contractions are most of a transformer layer. A contraction
kernel whose epilogue cuBLASLt applies itself (none, a bias, relu of a bias, a residual, a residual
and a bias) can run as one torch call (`mm`, `addmm`, `_addmm_activation`). The compiled executor
times both at the first launch of each shape bucket and keeps the faster (the choice is cached with
the tuned configs); the cuBLAS call is a launch config like any other, so it is compared with the
float64 reference at its first call at every new shape signature. It runs only in the float32
precision the kernel was compiled for, and only in programs that follow PyTorch's TF32 setting
(`allow_tf32=None`, the default): a pinned program keeps its own kernels, so its bits do not depend
on what PyTorch says (found by tests/test_compiler_tf32.py). If PyTorch's matmul precision still
differs at a call, that call runs the Triton launch, so cuBLAS never rounds to TF32 behind the
program's back. Never in deterministic mode (cuBLAS picks its algorithm
by shape, so a row's bits could depend on the batch size), never for kernels compiled directly with
`truestar.kernel.compile` (opt in with `library=True`), never with compiled in sentinels.
`TRUESTAR_LIBRARY=0` turns it off. A choice measured while another job loads the GPU can be the
slower one on an idle GPU (Triton contraction kernels that need a whole SM suffered most from
sharing in our runs); both are correct, and a new cache directory measures again.

**KV caches updated in place.** `index_copy` writes into its input when the memory planner proves
the input dead after it (or the caller donates it), and `cp(inputs, update={4: 1, 5: 2})` declares
that input 4 is updated in place to hold output 1: a decode step then writes only the new
positions, and output 1 comes back as the input tensor itself. When the producer of the output
cannot write in place, the output is copied into the input at the end (`cp.last_update_inplace`
says which). Bitwise equal to the out of place call (tests/test_xfmr_perf.py).

**Grouped query attention without copies.** The importer splits the query heads into
`(kv heads, group)` and lets attention broadcast each KV head over its group (docs/LLM.md), instead
of repeating the KV heads, which copied the cache group times per layer.

**CUDA graph replay (`cp.graphed(static=(), update=None, outputs="copy")`).** One graph per input
signature (and per PyTorch precision variant), recorded only after the ordinary verified first call
at that signature, then replayed once and compared bitwise with that call before it is used
(TS801 refuses it otherwise). Index checks are recorded inside the graph and read after every
replay, so a bad index raises as in the ordinary call and nothing is returned. A step that syncs
with the host and `TRUESTAR_VERIFY=all` refuse the recording (TS800). Compiled in sentinels stay
armed: each graph owns a pinned flag buffer its kernels write (torch step triggers are folded into it
inside the graph), every replay reads it, and a fired flag runs the ordinary locate run, which raises
`SentinelError` naming the op. An
input bound by address that is not the recorded tensor raises TS802. Other inputs are copied into
the graph's buffers in one launch per call.

```python
cp = compile_program(prog, "cuda", "bfloat16")
step = cp.graphed(update={4: 1, 5: 2})        # inputs 4 and 5 are the K and V caches
logits, kc, vc = step([x, cos, sin, pos, kc, vc])
```

## Automatic graphs (2026-11-25, `graphs="auto"`)

An ordinary call of a CUDA program launches every kernel from Python: 20 to 126 kernels for the
coverage refresh families, about 90 us of host time each (Triton's launcher, the index checks, the
step bookkeeping), so at batch 3 the call took 0.54x to 1.06x the time of eager PyTorch while the GPU
work was a tenth of it. From the second ordinary call at the same input shapes, dtypes and precision
settings, the call is now replayed from a CUDA graph (compiler/graphs.py ``GraphedProgram``): the
first call at a signature is the ordinary verified call; the second runs ordinarily, is recorded and
replayed once, and the replay must be bitwise equal to that call or the graph is refused (TS801);
later calls copy the inputs into the graph's buffers, replay, read the index check flags and return
fresh output tensors. The bits are the ordinary call's.

With ``update=`` (a training step updating its params and optimizer state in place, 2026-11-25) the
updated inputs are bound by address, as ``graphed(update=)`` binds them, and the recording is
checked bitwise against the ordinary call, updated inputs included; a later call passing other
tensors for them gets TS802 before anything is written and runs as an ordinary call from then on.
A call stays ordinary (``cp.auto_graph_refusals`` says why for a refused signature) when: ``donate``
is passed; an input is not a CUDA tensor; ``TRUESTAR_VERIFY=all``; the program was
compiled with ``memory_budget``, ``recompute`` or ``sentinel_read="deferred"``; a step syncs with the
host (a random op reads its key on the host, so draws keyed by the call count stay fresh); the
recording was refused; or the program already holds ``AUTO_GRAPHS_MAX`` (8) graphs. ``truestar
profile`` measures the ordinary call. Opt out with ``compile_program(..., graphs="off")`` or
``TRUESTAR_GRAPHS=off``. Memory: each graph keeps a pool for its call's working set, in addition to
the caching allocator's (docs/MEMORY.md); a memory budget therefore turns automatic graphs off.

Measured on the RTX 5060 Laptop, the 26 families of the coverage refresh at batch 3
(`scripts/bench_autograph.py`, transformers 5.20.2): Before: 0.54x to 1.06x of eager, median 0.75x. After: 4.78x to 15.86x, median 8.87x. Outputs bitwise equal to before for 26 of 26 families (batch 3 and batch 2, first replay and later ones).

## Compile report

`compiled.report` (a `CompileReport`; `to_text()`, `to_dict()` with schema
`truestar.compiler.report/1`, `fallbacks`, `library`, `kernels`, `host`):

```
compile report: wide_layer_norm on cuda, float32 (compiled in 0.04 s)
  Truestar kernels: 2 covering 2 IR ops (2 tile)
  library kernels (torch): 0
  fallbacks (float64 reference implementation on the CPU): 1
  host scalars (integer bookkeeping on the CPU): 0
  layout ops (views and copies, no arithmetic): 0; constants created: 0; dtype adapters: 0; cond: 0; loop: 0
  FALLBACK layer_norm %layer_norm: layer_norm over more than the last dim has no kernel rule yet
  steps:
    kernel   wide_layer_norm:%tanh [tile] tanh
    fallback %layer_norm layer_norm
    kernel   wide_layer_norm:%mul [tile] mul
```

## Memory

Every compiled program frees each value after its last reader, writes elementwise kernels over dead
inputs, plans its exact peak before the first launch (`cp.memory_plan({"B": 64})`), refuses a
program over `memory_budget` (TS420) and can recompute values under a budget (`recompute=True`, checked
bit for bit, TS421). See docs/MEMORY.md.

## Changes outside `truestar.compiler`

All additive:

* `kernel/ir.py`, `triton_backend.py`, `cpu_backend.py`, `kernel/reference.py`: a unary `erf`, so
  exact gelu runs in a kernel.
* `kernel/triton_backend.py`: symbolic sizes are excluded from Triton's argument specialization and
  the tile block row count no longer follows a symbolic row count (see above).
* `export/runner.py`: the names `cuda`, `cpu`, `cuda:float16`, `cpu:float16` import the compiler
  on first use and are accepted by `truestar run --executor`.
* `frontend/function.py`: a runner that says `binds_symbols` checks input shapes itself once per
  distinct shape, so a call no longer pays about 0.35 ms of sympy per call for the shape check.

## Measured results

Measured 2026-11-24 with `scripts/bench_compiler.py --repeat 3` on an RTX 5070 Ti (torch 2.11.0+cu128,
triton 3.7.1, Windows), float32. Every (case, tool) pair runs in its own fresh process with cold
Truestar, Triton and Inductor caches, one process at a time. The GPU is shared with other jobs, so
each time is the fastest of three runs of the whole matrix (and, inside a run, the fastest of seven
batches of calls); compile counts are the largest seen. Truestar runs with its default settings,
verification on.

### Run time

| case | tool | run, batch 64 (ms) | run, batch 1 (ms) | vs eager at 64 |
|---|---|---|---|---|
| MLP, 1024 to 2048 x3 to 1000 | Truestar | 0.51 | 0.50 | 2.16x slower |
| MLP, 1024 to 2048 x3 to 1000 | PyTorch eager | 0.23 | 0.09 | baseline |
| MLP, 1024 to 2048 x3 to 1000 | torch.compile | 0.16 | 0.12 | 1.46x faster |
| TransformerEncoderLayer, d 256, 4 heads, seq 128 | Truestar | 1.61 | 0.48 | 2.41x slower |
| TransformerEncoderLayer, d 256, 4 heads, seq 128 | PyTorch eager | 0.67 | 0.15 | baseline |
| TransformerEncoderLayer, d 256, 4 heads, seq 128 | torch.compile | 0.67 | 0.18 | same |
| resnet18, 224 px | Truestar | 11.2 | 1.65 | 1.05x faster |
| resnet18, 224 px | Truestar, TF32 allowed | 10.4 | 0.98 | 1.14x faster |
| resnet18, 224 px | PyTorch eager | 11.8 | 1.19 | baseline |
| resnet18, 224 px | torch.compile | 10.1 | 0.93 | 1.17x faster |
| native RNN, hidden 512, 32 steps, branch on data | Truestar | 1.41 | 1.62 | 1.12x faster |
| native RNN, hidden 512, 32 steps, branch on data | PyTorch eager | 1.58 | 1.22 | baseline |
| native RNN, hidden 512, 32 steps, branch on data | torch.compile | 1.23 | 0.89 | 1.29x faster |

Where the time goes:

* **MLP and transformer: a loss of 2.2x and 2.4x against eager.** Both are dominated by float32
  matrix products. Truestar's contraction kernel (Triton `tl.dot`, IEEE float32, fixed 64 by 64
  tiles) is slower than cuBLAS: in the transformer the four contraction kernels take 1.66 ms of the
  2.46 ms per step profile, the fused attention kernel 0.23 ms and the two layer norm kernels 0.19 ms.
  At batch 1 the fixed tile wastes most of its rows, which is why the MLP does not get faster.
* **resnet18: level with eager (5% faster), 11% behind torch.compile.** Convolution is cuDNN on
  both sides (10.3 ms of Truestar's 13.9 ms per step profile). The 17 fused kernels of `batch_norm`,
  `add` and `relu` take one pass over the data where eager takes two or three. Truestar's default keeps convolutions in IEEE
  float32 while PyTorch allows TF32; with `allow_tf32=True` Truestar measures 10.4 ms.
* **Native RNN: 12% faster than eager, 15% behind torch.compile**, with 68 kernel launches per call
  (so a launch costs at most about 0.02 ms).

### Compile behaviour

| case | tool | first result at batch 1 (s) | first call at batch 7, then 64 (ms) | compiles for the new batch sizes | compile again after an edit (s) |
|---|---|---|---|---|---|
| MLP | Truestar | 2.97 (import 1.10, Truestar compile 0.09, Triton and first run 1.73) plus checks 0.04 | 12.7 (of which checks 11.5), 30.2 (of which checks 28.7) | 0 | 0.46, kernels generated: 1 |
| MLP | torch.compile | 2.81 | 806.0, 1.23 | 1 graph | 0.42 |
| TransformerEncoderLayer | Truestar | 4.82 (import 1.56, Truestar compile 0.12, Triton and first run 3.14) plus checks 0.05 | 125.1 (of which checks 121.3), 1375.9 (of which checks 1371.3) | 0 | 0.84, kernels generated: 2 |
| TransformerEncoderLayer | torch.compile | 1.90 | 69.1, 1.77 | 0 graphs | 0.17 |
| resnet18 | Truestar | 4.36 (import 0.90, Truestar compile 0.12, Triton and first run 3.33) plus checks 0.23 | 311.4 (of which checks 281.0), 2730.2 (of which checks 2705.2) | 0 | 0.33, kernels generated: 0 |
| resnet18 | torch.compile | 7.89 | 5849.9, 23.2 | 1 graph | 2.47 |
| native RNN | Truestar | 1.68 (Truestar compile 0.31, Triton and first run 1.37) plus checks 0.02 | 12.3 (of which checks 8.87), 147.1 (of which checks 3.52) | 0 | 0.17, kernels generated: 1 |
| native RNN | torch.compile | 3.43 | 1921.6, 267.4 | 3 graphs | 0.28 |

"First result" is the time from the model in memory to the first output at batch 1 with a cold
cache; Truestar's own share (lowering, fusion, code generation) is 0.09 to 0.31 s, the rest is
torch.export and Triton compiling each kernel on first launch. For the native case both sides of the
branch are run before anything else is timed.

* **Batch sizes 1, 7, 64: Truestar compiled nothing after the first call in any case** (programs,
  regions and kernels counted, kernel cache untouched). torch.compile compiled 1 more graph for the
  MLP, 1 for resnet18 and 3 for the native function; none for the transformer.
* **Time to the first result at a new batch size is mostly verification for Truestar.** MLP 12.7 ms
  and 30.2 ms against 806 ms for torch.compile's recompile. resnet18 at batch 7: 0.31 s against
  5.85 s. But at batch 64 the kernel layer's float64 check of every kernel on the new shape costs
  1.4 s (transformer) and 2.7 s (resnet18), where torch.compile, already dynamic, takes 2 ms and
  23 ms. Without checks those calls take 4.6 ms and 25 ms.
* **One Triton variant at batch 64 in the native case** (147 ms, 3.5 ms of it checks): the mean
  over the batch reduces over a symbolic axis, and its block size bucket changed (see Limits).
* **Loop lengths 8 and 48 through the program compiled for 32: Truestar 1.93 ms and 3.40 ms with 0 compiles; torch.compile 0.80 s and 2.34 s with 2 new graphs, and 1 graph break on the branch.**
* **First result:** faster than torch.compile on resnet18 (4.36 s against 7.89 s) and the native
  function (1.68 s against 3.43 s), level on the MLP (2.97 s against 2.81 s), slower on the
  transformer (4.82 s against 1.90 s; seven kernels each compiled by Triton on first launch).
* **After an edit** (one constant changed, compiled again in the same process): resnet18 0.33 s
  against 2.47 s and the native function 0.17 s against 0.28 s; the MLP is level (0.46 s against
  0.42 s); the transformer is slower (0.84 s against 0.17 s: the edit changes both layer norm
  kernels, and each new kernel costs one Triton compile of about 0.3 s). Only kernels whose IR
  changed are generated: 1, 2, 0 and 1.

### Correctness

Largest end to end error against the float64 reference interpreter on the first inputs: MLP 1.1e-6,
transformer 2.3e-7, resnet18 8.3e-7 (8.0e-4 with TF32 allowed), native RNN 8.1e-7. No fallbacks in
any benchmark case; resnet18 has 22 library kernels (20 `conv2d`, `max_pool2d`,
`adaptive_avg_pool2d`).

Tests (`tests/test_compiler.py`, 98 cases from 33 functions): every model above plus one program
per lowering rule group, five attention variants, native loops, branches and a `while` loop, on CPU
and GPU in float32 and float16, each compared with float64.

### After the merge with kernel autotuning (2026-11-24)

Main gained launch config autotuning for kernels (docs/KERNEL_IR.md) after the numbers above were
taken. Merged so that both hold: the tile block row count does not follow a symbolic row count in
the defaults or in the tuning candidates, and the autotuner's key leaves out batch like symbols,
so one tuned config and one Triton variant serve every batch size. One run of
`scripts/bench_compiler.py` after the merge (single run, shared GPU):

| case | run, batch 64 (ms), before and after | Truestar compiles for batch 7, 64 | first call at batch 7, then 64 (ms) |
|---|---|---|---|
| MLP | 0.51, now 0.41 | 0 | 15.7, 2210 |
| TransformerEncoderLayer | 1.61, now 1.00 | 0 | 129, 10200 |
| resnet18 | 11.2, now 11.3 | 0 | 947, 8120 |
| native RNN | 1.41, now 2.45 (single run; 2.6 in single runs before) | 0 | 15.1, 183 |

* Tuning cut the transformer layer from 2.4x to 1.5x behind eager (0.67 ms) and the MLP from 2.2x
  to 1.7x (0.24 ms).
* Loss: the first launch of a kernel that is big enough to tune now pays the search, once per
  kernel: 2.2 s (MLP), 10.2 s (transformer) and 8.1 s (resnet18) on the first call at batch 64.
  `TRUESTAR_TUNE=off` gives the earlier behaviour. Tuned configs are stored on disk, so a later
  process reads them.
* `scripts/bench_kernels.py --quick` after the merge: tuned causal attention B1 H8 S1024 D64 fp16
  0.026 ms against FlexAttention 0.038 ms (1.47x); differential attention 0.034 ms against 0.051 ms;
  S2048 D128 0.110 ms against 0.180 ms; gelu plus residual 0.069 ms against torch.compile 0.067 ms.

### Convolution and pooling as Truestar kernels (2026-11-24)

`scripts/bench_conv.py` (full table in the status log of docs/QUALITY_BAR.md): resnet18,
mobilenet_v3_small and mobilenet_v2 at 224 pixels, batch 1, 8 and 64, fp32, TF32 and fp16, with 0
library kernels. resnet18 and mobilenet_v2 run at or ahead of torch.compile in all 18 cells (1.01x
to 1.9x faster; resnet18 batch 64 fp32 11.0 ms against 11.8 ms, eager 13.3 ms); mobilenet_v3_small
is 1.05x to 1.62x behind torch.compile in 8 of 9 cells (squeeze and excite pooling through window
loops). `scripts/conv_zoo.py`: every torchvision classifier under 100M parameters (63) compiles and
passes its checks; no convolution or pooling op is left as a library kernel or fallback in any.

### Op coverage rules (2026-11-24, `compiler/coverage_rules.py`)

Before, the compiled programs of the coverage list (docs/MODEL_COVERAGE.md) ran these ops on the
float64 CPU reference: `triu` (the causal masks of Llama, Qwen2 and Whisper), casts to an integer
dtype (BERT's attention mask, T5's buckets, CLIP), `argmax` (CLIP's pooling), `pad` and
`slice_scatter` (Swin's windows and shifted window mask), `group_norm` and `rms_norm` (the norm
layers), `pow` with an exponent other than 2, 3, 1, 0.5, -0.5, -1 (local response norm), integer
sums (embedding bags). Now:

| Op | Path |
|---|---|
| `triu`, `tril` | kernel: `where(col - row >= k, x, 0)` with the kernel IR's index values |
| `pad` (constant) | kernel: a strided read at `i - before` on every padded dim, the pad value outside (the zero padding load of convolution) |
| `flip` | kernel: a read at `size - 1 - i` (fixed sizes) |
| `slice_scatter` (step 1), `select_scatter` | kernel: `where(i in the slice, src at i - start, x)` |
| `pow`, any constant exponent | kernel: integer exponents up to 16 by binary powering; others `exp(c log x)` for x > 0 and the reference's values at 0 and below |
| `group_norm`, `rms_norm`, `vector_norm`, `adaptive_max_pool2d` | rewritten before lowering into the primitive ops of their verified lowerings (`lower_composites`, the same code autodiff uses), whose kernels and steps already exist |
| casts to an integer or bool dtype, `argmax`, `argmin`; integer `sum`, `cumsum`, `amax`, `amin` | the reference definition run on the device's own tensors (`IntStep`, a positions vector made on the device for argmax), checked exactly against the CPU reference on the first call |
| `frozen` | kernel: the value itself |

Measured (scripts/op_coverage.py, `--compile cuda`): every model of the list (torchvision zoo,
Hugging Face configs, torch.nn layers and losses) compiles with no float64 CPU fallback; the per
model column is in docs/MODEL_COVERAGE.md. tests/test_compiler_coverage.py
compiles each new path on the CPU and CUDA at two batch sizes against the reference and asserts no
fallback; a `triu` rule broken to keep one more diagonal is refused by the first call check.

## Training programs (2026-11-24)

A training step program (docs/TRAINING.md) runs with no fallback step: `log_softmax` is one tile
kernel with two reductions (max, then the sum of exponentials), `erf` (the exact gelu gradient) and
`expm1` (Adam's bias correction) are unary kernel ops, and `cast` to the working dtype was already a
copy. The MLP step went from 4.6 ms with two fallbacks (log_softmax 0.94 ms, erf 0.87 ms, each a
float64 host round trip) to 1.9 ms.

**Conv weight gradient, split in row chunks with a fixed order.** `conv2d_weight_grad` has only
O x C x KH x KW outputs and sums over every image and pixel: a 64 channel 3 x 3 conv gives 9 programs
of 64 x 64, which leaves most of the GPU idle. On CUDA the rule cuts the output rows into S chunks
(`conv_rules.weight_grad_splits`: the smallest divisor of the row count that reaches 560 programs,
at least 64 pixels per chunk), writes S partial sums as one `[S, O, C, KH, KW]` kernel and adds them
in a second kernel in the order 0, 1, ..., S - 1, written out as a chain of adds (no reduction whose
order a launch config could change, no atomics). S depends on the shapes only, so the result is
the same bits on every call, after a recompile and under `verify="all"`
(tests/test_train_compiled.py), and recompute stays bitwise. One layer, float32, batch 8 unless
stated (ms, median of 30):

| layer | splits | before | after | cuDNN `conv2d_weight` |
|---|---|---|---|---|
| 64 ch, 56 x 56 | 28 | 1.24 | 0.26 | 0.15 |
| 64 ch, 56 x 56, batch 32 | 28 | 6.81 | 0.93 | 0.42 |
| 128 ch, 28 x 28 | 7 | 0.57 | 0.26 | 0.06 |
| 256 ch, 14 x 14 | 2 | 0.36 | 0.33 | 0.07 to 0.12 |
| 512 ch, 7 x 7 | 1 | 0.35 | 0.35 | 0.12 |

Still 1.7x to 5x behind cuDNN per layer: the implicit GEMM over (image, row, column) with a
strided gather of the input is the cost now, not the parallelism.

**Convolution training layouts (2026-11-25).** A float32 `tl.dot` on the RTX 5070 Ti runs at half
speed when its B tile is contiguous along K instead of N (a 4608 x 512 x 1568 GEMM: 0.52 ms against
0.29 ms; the A tile's order barely matters, 0.27 ms contiguous along M). A training step reads every
conv weight and activation at run time, so the inference trick (a constant weight stored once in
the GEMM's order) did not apply and every conv of a training step read its B tile across memory.
The rules now make the GEMM operands contiguous where it pays, each copy by a small kernel of its
own, shared between the ops that read it (`conv_rules._relayout`):

* `conv2d`: a run time weight is copied to `[kH, kW, C, O]` (`_gemm_weight`); the input is read
  from a `[N, H, W, C]` copy for images of 16 x 16 or fewer pixels and for strided spatial
  kernels on 128 or more channels (`channels_last_input`).
* `conv2d_input_grad`: the weight as `[kH, kW, O, C]`; the cotangent channels last for small
  images; at stride 2 the sum is split by output parity (`_input_grad_by_parity`): with the tap
  residues as axes of their own each output reads only the taps that divide exactly, 4 of 9 for a
  3 x 3 kernel instead of loading all 9 with three of four masked out.
* `conv2d_weight_grad`: the cotangent as `[N, OH, OW, O]` (the B operand), and for 3 x 3 kernels on
  128 or more channels the input as `[N, H, W, C]` with the result summed in `[kH, kW, C, O]`
  order; the sum over (image, row) chunks is split as before plus image chunks when fewer than 140
  programs remain, at most 32 partial sums, added in the fixed chunk order
  (`channels_last_splits`).

Every change keeps the same terms, summed in an order fixed by the shapes: two calls give the same
bits (tests/test_conv_kernels.py). Two gradient rules changed with it: the relu gradient's zero is a
scalar, not a `zeros_like` tensor (18 zero tensors of activation size per resnet18 step, written
and read back, and the relu backward now fuses into the input gradient's epilogue), and a global
average pool's gradient is `expand(g / (H W))` instead of two matmuls with averaging matrices (a
relu backward fused into those made a 4.1 ms kernel). The fusion pass also undoes split points the
greedy pass no longer needs (`Lowering._unmaterialize`, float32 elementwise values over stored
inputs only), which keeps a batch norm's `g * xhat` inside its gamma gradient sum.

resnet18, batch 32, 224 x 224, SGD with momentum, strict float32, RTX 5070 Ti (ms per step, fair
benchmark conventions: 5 rounds of 50 samples, median; GPU time from `truestar profile`):

| | GPU time per step | step, plain calls | step, CUDA graph |
|---|---|---|---|
| before (main 9a9ad96) | 40.1 | 41.2 | 38.5 |
| after | 23.4 | 24.4 | 21.5 |

The fair benchmark's own script (FAIR_BENCH_2026_11 fb_torch.py, train_sgd_fp32, fresh caches, idle
gate) on the same afternoon: Truestar graphed 21.38 ms (verification off, diagnostic), torch.compile
max-autotune 21.25 ms, PyTorch eager 25.33 ms; gradient error against float64 8.6e-4 (eager
8.4e-4). As shipped, the first call check still refuses the step on a momentum buffer whose own
float32 reference is as far off float64 (TS782, the false alarm in the benchmark report).

GPU time of the step's convolutions by kind (ms, from the profile, copies included; cuDNN is the
sum of `F.conv2d` and `torch.nn.grad` calls on the same layers): forward 10.5 to 5.3 (cuDNN 4.3),
input gradient 8.8 to 5.8 (cuDNN 4.0), weight gradient 12.2 to 6.6 (cuDNN 5.1).

## Compile cache (2026-11-24, P6)

One directory, `TRUESTAR_CACHE_DIR` (default `~/.cache/truestar/kernels`), holds everything a new
process needs to reach its first result without generating, compiling, tuning or verifying again
(`truestar.compile_cache`):

| entry | where | key |
|---|---|---|
| generated kernel source | `triton/k_<key>.py`, `numba/`, `metal/` | canonical kernel IR, backend, options, `CODEGEN_VERSION` (existed) |
| Triton binaries | `triton/bin/` (Triton's own cache: `TRITON_CACHE_DIR` is pointed here unless the user set it) | Triton's key: source, Triton version, GPU, options |
| numba binaries | `numba/__pycache__` (`cache=True`) | numba's key |
| launch config and cuBLAS choice | `triton/tune/<key>.json` | kernel structure, size bucket, dtypes, GPU name, Triton version, search space (existed) |
| first call verification result | `programs/<key>.json` | below |

**The program record and when verification is skipped.** A verified first call writes a record:
the sha256 of its output bits, the list of kernel launches it ran (kernel key, shapes, dtypes and
launch config, that is the exact binaries), the end to end and region errors and any TS782
conditioning notes. Its key is the sha256 of everything that check depends on: the IR program
with its weights (`ir.serialize.program_hash`), the compile options of the variant (dtype, TF32,
deterministic, library ops, cuBLAS), the input values, shapes and dtypes of the call, the
`update` spec, the verify mode and where the float64 reference runs (`TRUESTAR_VERIFY_DEVICE`,
`TRUESTAR_VERIFY_GPU_MB`), the cuDNN and matmul precision flags, `TRUESTAR_LIBRARY` and
`TRUESTAR_TUNE`, a content hash of the Truestar compiler, kernel and IR sources (an edit to any
of them invalidates every record; the version number alone would not), `CODEGEN_VERSION`,
`TUNE_VERSION`, torch, CUDA and Triton versions and the GPU name and capability.

A later call with the same key runs once without the checks and is accepted (TS1350,
`cp.compile_cache["verify"] == "hit"`) only if it launched exactly the recorded binaries AND its
outputs are bitwise equal to the recorded ones. Same program, same inputs, same binaries, same
output bits: the float64 references and every comparison would see what the recorded check saw.
If either differs (another tuned config, a kernel giving other bits), the call is verified in full
from the inputs as they were (TS1354) and the record is rewritten. **So the cache skips
verification only for a first call on inputs a verified call has already seen**, which is what a
restarted process replaying a fixed warm up batch does. A first call on new inputs is verified as
before: in production, where every request differs, the cache saves compile, tuning and binary
load time but not the check. Not cached at all: `TRUESTAR_VERIFY=all`, programs with sentinels,
donated inputs, a memory budget, or test hooks. In a hit, regions inside control flow are not
marked checked (the top level program and every kernel that ran are), so a loop or branch region
is checked again on its next run.

**Safe by construction.** Every file is written to a temporary name and renamed over the target.
A record carries a format version and a sha256 of its body; one that does not parse, has another
version or digest, or is truncated is ignored with a TS1351 warning, deleted and rebuilt by the
same call (tests/test_compile_cache.py truncates, edits and re-versions records). A record that
is consistent but does not reproduce is a full verification, never an accepted result (the test
also corrupts a kernel's output on replay: refused with ProgramMismatchError). A failed write is
TS1353 and the call still returns. The directory is capped at `TRUESTAR_CACHE_MAX_MB` (default
4096) by deleting the least recently used entries, checked once per process after the first
record is written (TS1352); kernel sources, tuned configs and records are touched on every hit,
Triton's binary directories are ordered by when they were written (Triton does not report its hits).
`TRUESTAR_COMPILE_CACHE=0` turns cross process reuse off: no records, Triton back in its default
directory, tuned configs measured again. `truestar cache info` prints the directory, sizes per
kind and the cap as JSON; `truestar cache clear` empties it; `truestar cache trim` applies the cap.

**Measured** (`scripts/bench_compile_cache.py`, RTX 5070 Ti shared with other jobs, torch 2.11.0,
Triton 3.7.1, float32, one run each: small versions of the fair benchmark's models: MLP
1024-2048-2048-1000 batch 64, encoder layer d256 seq 128 batch 8, resnet18 batch 2, one Llama
style decode step d512 with an index_copy KV cache of 256). "First result" is compile plus first
call from the IR program in memory; every run starts with an empty cache directory, Triton's
binaries included. Before = 9a9ad96.

| workload | cold | warm in process, before / after | warm new process, before / after | of which verification before | outputs |
|---|---|---|---|---|---|
| MLP, cuda | 2.2 s | 0.036 / 0.024 s | 0.44 / 0.26 s | 0.21 s | bitwise equal |
| encoder layer, cuda | 3.9 to 4.6 s | 0.068 / 0.056 s | 0.65 / 0.40 s | 0.23 s | bitwise equal |
| resnet18, cuda | 27 to 30 s (tuning) | 0.35 / 0.21 s | 0.99 / 0.51 s | 0.50 s | bitwise equal |
| decode step, cuda | 5.6 to 6.3 s | 0.071 / 0.048 s | 0.64 / 0.46 to 0.61 s | 0.21 s | bitwise equal |
| MLP, cpu | 1.7 to 1.9 s | 0.047 / 0.033 s | 0.82 / 0.78 s | 0.05 s | bitwise equal |
| encoder layer, cpu | 5.9 to 6.1 s | 0.16 / 0.072 s | 1.38 / 1.01 s | 0.31 s | bitwise equal |
| resnet18, cpu | 39 to 40 s | 2.38 / 2.07 s | 28.5 / 3.5 s | 25.4 s | bitwise equal |
| decode step, cpu | 5.8 to 6.0 s | 0.023 / 0.029 s | 1.17 / 1.06 s | 0.03 s | bitwise equal |

Bitwise equal means the cold call, the warm in process call, the warm new process call and a new
process that reads the program from a `.tsx` give the same output bytes (sha256). On CUDA every
warm new process first result is now under 1 s (0.26 to 0.61 s); on the CPU three of four are
about 1 s or less, and resnet18 is its run time (2 s of numba compute at batch 2). What is left in a
warm CUDA process: Triton's first launch of each cached binary (about 10 ms per kernel, 0.22 s for
resnet18's 22 kernels), loading the generated modules (0.11 s), lowering and fusion (0.1 s) and
the record key (0.065 s, mostly hashing the weights; a `.tsx` load supplies the hash it already
checked). On the CPU, numba's first load of cached functions costs about 0.45 s per process.

**Not covered: the front end.** `torch.export` costs 0.6 to 1.4 s per process here, and importing
torch 1.2 to 1.7 s and Truestar 0.35 to 0.45 s, so a restarted script that exports again still
spends 2 to 3 s before its first result. Exporting once and loading the `.tsx` (`truestar
export`, `ir.serialize.load`, 0.01 to 0.08 s) removes the export: first result from the loaded
program 0.30 to 0.48 s on CUDA. A cache of torch.export keyed on Python model code cannot be made
safe (the forward can call any function), so there is none.

## Limits (2026-11-24)

* **Transformer inference against PyTorch (P38 entry in docs/QUALITY_BAR.md, provisional).** Wins: the
  TransformerEncoderLayer in TF32, bfloat16 and float16 (1.10x to 1.37x over the best of eager and
  torch.compile) and a Llama style prefill in bfloat16 and float16 (1.04x over max-autotune). Losses:
  the TransformerEncoderLayer in strict float32 (0.755 ms against 0.669 ms, torch.compile default),
  prefill in float32 and TF32 (0.74x, 0.94x) and every decode case against torch.compile max-autotune
  (0.48x to 0.78x; Truestar beats eager there by 1.5x to 3x with `graphed()`). What is left: float32
  attention on FMA units, the head split copies of `nn.MultiheadAttention`, a decode attention that
  splits the KV length, rotary embeddings fused into the projections (the kernel IR has no offset
  index for the half rotation), and the small per step kernels of a decode step.
* **Run time on matrix product heavy programs is behind PyTorch**, 2.2x on the MLP and 2.4x on the
  transformer layer above. The cause is the float32 contraction kernel against cuBLAS, not launch
  overhead (at most about 0.02 ms per kernel). Update 2026-11-24 (matmul entry in
  docs/QUALITY_BAR.md): with the new contraction launch configs the MLP at batch 64 runs in 0.25 ms
  against eager 0.24 (its kernels take 243 us of GPU time against 397 before) and the transformer
  layer in 0.89 ms against eager 0.67. Strict float32 is the remaining gap: Triton's IEEE float32
  `tl.dot` runs on FMA units and stays 1.1x to 1.6x behind cuBLAS. With `allow_tf32=True` the MLP
  kernels take 62 us and the MLP is bound by host overhead. Update 2026-11-24 (split K entry in
  docs/QUALITY_BAR.md): split K brought small M and tall K products to 0.6x to 0.8x of F.linear;
  the MLP at batch 64 runs in 0.170 ms (eager 0.239, torch.compile 0.163); the transformer layer
  0.90 ms against eager 0.67, and large strict float32 products remain 1.03x to 1.4x behind cuBLAS.
* **Convolution and pooling** are Truestar kernels since 2026-11-24 (strided indices). Left: a
  symbolic spatial size, `adaptive_max_pool2d`, `avg_pool2d` gradients and convolutions other than
  2D go to the library or the fallback, named in the report. Deep layers with few output pixels and a
  long K trail cuDNN (the conv shape has no split K, see docs/KERNEL_IR.md limits); the weight gradient sums over batch
  and pixels in one flat K and is the slowest of the three conv kernels.
* **Verification time.** The first call pays one float64 reference run of the program, one float64
  evaluation of every region, and the kernel layer's check of every kernel; the kernel layer
  repeats its check at every new input shape signature. Measured on resnet18 at 224 pixels: 0.23 s
  at batch 1, 0.28 s at batch 7, 2.7 s at batch 64. It is reported apart from compile time.
  Since 2026-11-24 the kernel layer's float64 reference runs on the GPU for every check above about
  1M elements of work (docs/KERNEL_IR.md section 7), with the same math, tolerances and schedule.
  Verification time of the first call at batch 64, `scripts/bench_compiler.py --tools truestar`,
  two runs on a shared GPU: resnet18 0.15 to 0.55 s (was 2.6 s), transformer 57 to 134 ms (was 1.26 s),
  MLP 10 to 11 ms (was 27 ms). The price is the GPU's first float64 use in a process, 0.1 to 0.2 s
  once (CUDA loads each float64 kernel and cuBLAS on first use): the MLP's first call verifies in
  0.17 to 0.20 s instead of 0.04 s, and the native RNN pays it at batch 7 (0.13 to 0.20 s instead of
  14 ms). Region and end to end checks moved to the GPU on 2026-11-24 too (see Verification and
  the status log in docs/QUALITY_BAR.md).
* **Triton can still compile a variant for a new shape** in three cases: a symbolic size that is
  the reduced or lane axis of a tile kernel crosses a power of two bucket; a stride that depends on
  a symbolic dim becomes 1 or changes divisibility by 16; an attention kernel crosses 128 queries
  in half precision. Truestar's own compile count does not change in any of them.
* **Comparisons near a tie.** A branch decided by `x > 0` where `x` is within rounding of 0 can go
  the other way than in float64. The end to end check then reports a mismatch, which names the tie
  when the run's own decisions explain it (Verification); the program is refused either way.
  Bool results of a region are compared exactly. Not covered by the tie
  diagnosis: comparisons inside loop or branch bodies, and a comparison fused into the kernel that
  computes its operand (the region check then decides on float64 values).
* **A program output that drifts** across many regions in half precision can exceed the end to end
  tolerance while every region passes; the error says so and names both possibilities.
* **Integers above 2^24** read by a kernel lose exactness (they are loaded as float32).
* **No autodiff** through compiled programs, no int8, no multi GPU. Not run on ROCm or any NVIDIA GPU
  other than the RTX 5070 Ti (and the laptop GPU of the SM aware entries); the "metal" device runs
  Metal kernels on a Mac (docs/METAL.md), measured only on GitHub's hosted macOS 15 runners; the CPU
  backend is the numba correctness backend, not a fast one.
* **The first call of each new branch** pays that branch's Triton compile and its check, since
  kernels are compiled by Triton on first launch.

## Sums whose terms cancel (2026-11-24)

A training step sums gradients over the batch, and some of those sums are tiny next to their terms
(a bias gradient; a conv bias in front of batch norm, whose gradient is zero in exact math). The
plain check divides the error by the result's own size and refused correct float32 kernels (GPT
step on CUDA: 2.95e-5 against 2.5e-5; CNN step on the CPU: 1.5e-3 at the group check). A value that
fails is measured once more: kernels by `kernel.verify.magnitude_kernel` (every sum's body under an
absolute value), group checks by the op on its operands' absolute values for `sum`, `mean`,
`matmul`, `linear` and `conv2d`. That is the sum of the terms' magnitudes, which bounds float
rounding in any order. Since the merge with the element measure (kernel/verify.py) it is per element:
each element's denominator is raised to its own terms' magnitude only, so a cancelling sum passes and
an element with small terms that is wrong is still refused and named (`tests/test_kernel_magnitude.py`,
`tests/test_kernel_verify_elements.py`).
