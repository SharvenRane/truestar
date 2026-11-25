# Metal backend

Author: Sharven Rane. First written 2026-11-24. Code: `src/truestar/kernel/metal_backend.py`.
Pain ledger row: P40 (docs/QUALITY_BAR.md).

Truestar's kernel IR compiles to Metal Shading Language (MSL) for Apple GPUs, next to Triton
(NVIDIA) and numba (CPU). The generator is written from the IR; no Metal code of another project
was used. Every Metal kernel is checked against the float64 reference on its first call per shape,
exactly like a CUDA kernel.

## 1. Use

```python
from truestar import kernel as K
fn = K.compile(kernel, "metal")              # kernel layer
y = fn(x=x_mps)                             # MPS tensors in, MPS tensors out

from truestar.compiler import compile_program, CompiledExecutor
cp = compile_program(program, "metal", "float16")   # compiled executor on the Mac's GPU
```

`truestar check model.py --backends cpu,metal` runs the model through Truestar's own compiler
(torch.export, the Truestar IR, the compiled executor with Metal kernels) and compares it with the
float64 reference like any other backend (docs/BACKENDS.md). `metal` is opt in, never in the
default set. Ops with no kernel lowering run through the float64 reference on the CPU; the check
reports them as TS400 instead of counting them as Metal results.

On a machine without Metal every entry point raises TS840 (`MetalUnavailable`); nothing falls back
to the CPU or another GPU.

## 2. How a kernel runs

**Compile and launch.** The generated source is compiled at run time with
`torch.mps.compile_shader` (torch 2.6 and later) and the kernel is launched with the MPS tensors as
its buffers, in torch's MPS command stream. So it shares memory with torch MPS tensors (no copy),
orders correctly with MPS ops, and needs nothing beyond the torch wheel. Two other routes were
considered: PyObjC's Metal bindings (`pyobjc-framework-Metal`, installs cleanly with pip on macOS)
would need the MTLBuffer behind each MPS tensor, which torch exposes only through an undocumented
pointer, and its own command queue synchronized with torch's; a small compiled helper needs a
toolchain at install time (never introduce #8). Neither gains anything here.

Inputs are passed contiguous (a strided or offset view is copied first); sizes and strides go in
one int32 parameter buffer, so a kernel compiled for symbolic sizes is reused at every size.

**Loop shapes.** Fusion uses the GPU plan's structure analysis (`triton_backend.plan`, which reads
only the IR), so "metal" fuses the same kernels as "cuda". Each structural shape maps to a Metal
loop shape:

| structure | Metal shape | code |
|---|---|---|
| tile without reductions | `pointwise` | one thread per output element |
| tile with reductions (softmax, layer norm, row sums) | `rows` | one threadgroup per row; each reduction folded by the group (thread t takes elements t, t + G, ... in order, then a fixed tree in threadgroup memory); the row's lanes shared out over the threads |
| tile with one value per element over a static reduction of at most 64 (pooling) | `loops` | one thread per output element, serial reduction |
| contraction (matmul with prologue and epilogue) | `contraction` | 32 x 32 output tile per threadgroup of 128 threads, factors staged through threadgroup memory 16 deep in K; SIMD group matrices (8 x 8 multiply accumulate, four per SIMD group) where the device runs them correctly, else plain multiply adds; epilogue per element |
| attention, one softmax term, head dims static multiples of 8 | `attention_simd` | 8 query rows per SIMD group, 8 to 32 rows per threadgroup; Q K^T and P V as 8 x 8 SIMD group matrix products over key chunks staged in threadgroup memory; two passes (row max and sum, then the weighted values), so the output is never rescaled; chunks the mask's static bounds exclude are skipped |
| other attention (up to 4 softmax terms, static head dim up to 256), or a handful of static query rows | `attention` | one query row per threadgroup: one key per thread per chunk, flash style running max, sum and output, fixed tree reductions |
| conv, window, attention beyond those | `loops` | one thread per output element, every reduction a serial loop at the shallowest depth where its axes are bound |

`TRUESTAR_METAL_ATTENTION=auto|simd|rows|loops` picks the attention shape (default auto: simd where
it applies and SIMD group matrices work, else rows); the choice is part of the kernel cache key.
The `loops` shape is correct for every kernel the plan accepts and slow for large convolutions; an
implicit GEMM convolution is not written yet.

Whether SIMD group matrices are used is decided once per process by compiling and running one 8 x 8
product against float64 (`metal_backend.simdgroup_probe`); TS844 says when they are not usable.
`TRUESTAR_METAL_SIMDGROUP=0` or `1` forces the choice (the tests run both).

**Numerics.** Metal has no float64. Kernels compute in float32: loads widen float16 and bfloat16,
every reduction accumulates in float32, stores round to the output dtype. A float64 kernel or
program is refused before any launch (TS841). The source starts with
`#pragma METAL fp math_mode(safe)` and calls the `precise::` math functions; NaN tests read the
bits, and division by zero, `log`, `sqrt`, `rsqrt`, `max` and `min` follow the reference's NaN and
infinity rules explicitly, so results do not depend on the compiler's default fast math. Metal has
no `erf`: `ts_erf` is x P(x^2) for |x| <= 1 and 1 - exp(-x^2) Q(1/x) up to 4 (Q fits exp(x^2)
erfc(x); both fitted for Truestar by least squares on Chebyshev nodes, degrees 9 and 16), 1 beyond.
On the M2, over every normal float32 in [-4, 4] (2,147,483,650 inputs) against float64 erf: worst
relative error 1.79e-7, every result within 3 ulps (99.98% within 2), 9.5 billion per second.
It replaced (2026-11-25) a 28 term Maclaurin series and a continued fraction, measured on the same
inputs at 9.5e-7 worst and 0.505 billion per second; the MLP with exact GELU went from 1.46 to
1.35 ms and its error from 1.1e-6 to 9.5e-7, MPS eager's. Subnormal inputs give 0 on the GPU,
which flushes them in arithmetic.

**Order free sums.** `osum` (docs/ORDER_FREE_SUM.md) and `compile(order_free=True)` run in the
rows and loops shapes: each thread's anchor and int64 total, an integer tree in threadgroup memory,
and the grid, the int64 to float32 rounding and the scale done on the bits, so the result does not
depend on the GPU's float conversions or subnormal handling. On the M2 it equals the definition bit
for bit (tests/test_ordersum.py).

**Determinism.** No atomics. Every reduction runs in a fixed order for a given kernel, shape and
device (the threadgroup size of the `rows` shape follows the reduced length, which is fixed per
shape), so a kernel gives the same bits run to run; the tests run every kernel twice and compare
bits. Results can differ between the SIMD group and the plain contraction (different summation
order), and between devices.

## 3. Verification

As on CUDA (docs/KERNEL_IR.md, section 7): the first call of each input shape, dtype and launch
config signature runs the float64 NumPy reference on the CPU (the MPS device has no float64) and
compares per element; a mismatch raises `KernelMismatchError` naming the kernel, the backend
"metal", the output and the element. The compiled executor adds its region and end to end checks.
`tests/test_metal.py` breaks the generated `exp` by 1% and asserts the first call refuses it.

## 4. Measured (2026-11-24, GitHub macos-15 runner, torch 2.14.1)

The runner reports `macOS-15.8.2-arm64`, MPS available; the SIMD group probe passes there. Final
run 45496817336: `tests/test_metal.py` 76 passed, 1 skipped (the refusal test for machines
without Metal). That covers every loop shape in float32 with SIMD group matrices forced on and
off, float16 and bfloat16 softmax, layer norm and matmul, the twelve unary ops on special values
(NaN and infinity at the same places as float64), causal attention, a 1% error put into `exp`
refused on the first call, two runs bitwise equal for every kernel, sentinels refused (TS842), the
compiled executor end to end, and `truestar check --backends cpu,metal` from the CLI.

End to end (`scripts/metal_e2e.py`), worst relative error against float64 over the output, and
the median time per call after warmup (20 calls; the runner is a virtual machine, so times are
provisional):

| model | dtype | Metal compiled | MPS eager | Metal ms | MPS eager ms |
|---|---|---|---|---|---|
| MLP 64, 256, 256, 10 wide, batch 32 | float32 | 1.4e-7 | 2.2e-7 | 0.69 | 0.84 |
| same | float16 | 2.9e-4 | 3.1e-4 | 0.79 | 0.83 |
| 2 layer pre norm transformer, d 64, seq 32, batch 4 | float32 | 3.2e-7 | 2.3e-7 | 1.75 | 2.99 |
| same | float16 | 6.6e-4 | 6.1e-4 | 1.85 | 2.88 |
| same, d 128, seq 128 | float32 | 3.9e-7 | 2.3e-7 | 8.49 | 2.47 |
| same | float16 | 6.4e-4 | 6.3e-4 | 8.53 | 2.24 |

Accuracy is at the level of MPS eager in every row, 0.6x to 1.8x its error (no fallbacks, no
library ops). Speed is close to or ahead
of eager for the MLP and the short sequence, and 3.4x to 3.8x slower at sequence 128, where the
`loops` attention recomputes every score for each of the 32 head dim outputs. The first call
(compile of every kernel plus the float64 checks) took 0.18 to 0.68 s; the checks 0.01 to 0.13 s
of it. In `tests/test_metal.py` (MLP and TransformerEncoderLayer 32 wide, 4 heads, batches 1 to 9)
the compiled program's error was 0.55x to 1.77x of MPS eager's, at float32 or float16 rounding.

**A torch MPS fault found on the way.** On torch 2.14.1 on this runner, `t.to("cpu",
torch.float64)` of an MPS float32 tensor returns wrong values: zeros, or numbers no float32 can
hold (2.8e-288), at every element of every shape tried (15490 of 15490 elements in
`scripts/mps_copy_probe.py`), for tensors from torch ops and from Metal kernels alike. The two step
`t.cpu().to(torch.float64)` is right, and CPU to MPS copies with a cast (`x.to("mps",
torch.float16)`) were right. Before this was known, the first CI run showed 19 kernel and model
failures that were all this copy. Truestar now never does a combined device and dtype copy with MPS
on either side (`kernel.verify.to_host_f64`, `to_device`, and the IR reference's input copy).

## 5. Measured on a real Apple M2 (2026-11-25)

Mac mini M2 (8 CPU cores, 10 core GPU, 16 GB), macOS 26.6.2, torch 2.14.1 and 2.15.0, MLX 0.32.4,
rented bare metal (Scaleway). Raw files: <research>/mac_real/.

**Tests.** On main 9a9ad96, `tests/test_metal.py` had 4 failures on the real GPU (69 passed): every
`TransformerEncoderLayer` end to end test failed to generate a contraction, because main's CUDA plan
had started folding batch axes the weight does not depend on into the rows (`ContractionPlan.fold`)
and the Metal contraction ignored them. Fixed; a generate test that fails without the fix runs on
any machine. Now 109 passed, 1 skipped (the refusal test for machines without Metal), with both
torch versions.

**The whole suite on the Mac** (`pytest -n 6 tests`, torch 2.14.1, CPU and MPS): on main 3125
passed, 24 failed, 3 errors. On this branch 3136 passed, 13 failed, 3 errors after fixing what was
Truestar's: `truestar check --backward` on MPS moved float64 targets to the device; the backward op
drilldown could not check MPS's `aten.linear_backward` (now computed on the CPU from mm and sum);
torch 2.14 changed `torch.clamp`'s gradient at the bounds (1 up to 2.13, 0 from 2.14) and the
import now follows the installed torch; tests assumed CUDA op names. What is left: the tree tensor
parallel step is not bitwise the one device step on the M2's CPU (9 tests; torch's CPU silu gives
other bits in the scalar tail than in the vector body, so a sharded tensor changes them: tp 2 is
bitwise with silu written as g / (1 + exp(-g)), tp 4 still is not; on the PC's x86 CPU torch's silu
is position dependent too), torch's arm64 CPU SDPA with float64 q, k, v and a float32 mask returns
wrong values (1.8 off; right on x86; repro <research>/mac_real/repros/cpu_sdpa_mixed_mask.py),
an AdamW update at 1.19e-10 against a test bound of 1e-10, and 5 tests that need `transformers`.

**The two torch MPS faults, on real hardware.** `F.linear` with a bias is **right** on the real M2
with macOS 26.6.2 (0 of 184 cases above 1e-3 in `scripts/mps_linear_probe.py`, 0 of 5 in
`scripts/mac_check.py`, torch 2.14.1 and 2.15.0): the wrong bias seen on hosted macos-26
runners belongs to their paravirtual GPU, so docs/upstream/mps_linear_bias_DRAFT.md is not to be
filed as a torch bug. The combined copy `t.to("cpu", torch.float64)` is **wrong** on the real M2
with torch 2.14.1 (all zeros, every element of every shape) and raises `RuntimeError: Undefined
type Double` on torch 2.15.0; the two step copy is right on both. Plain PyTorch repro:
<research>/mac_real/repros/mps_to_cpu_float64.py.

**Subnormals.** The M2 GPU flushes float32 subnormals to zero in arithmetic (multiply, add), in
Metal kernels compiled with `math_mode(safe)` and in torch MPS ops alike; loads, stores and bit
casts keep them (<research>/mac_real/ftz_probe.txt). Results with subnormal
intermediates therefore differ from CPU and CUDA runs.

**Speed and accuracy** (`scripts/apple_speed_table.py` in the research folder, idle machine, 20
untimed calls, then the median, p10 and p90 of 200 timed calls, each synchronized; error = max
relative error against float64 on the CPU). Truestar after the attention work of this session:

| model, float32 | Truestar Metal | MPS eager | torch.compile (MPS) | MLX | mx.compile |
|---|---|---|---|---|---|
| MLP 512, 2048, 2048, 10, batch 64 | 1.00 ms, 9.5e-7 | 0.66, 9.5e-7 | 0.65, 9.5e-7 | 0.75, 9.0e-7 | 0.70 |
| TransformerEncoderLayer(128, 4), 4 x 32 tokens | 0.45, 1.7e-7 | 0.50, 1.7e-7 | 0.56, 2.1e-7 | 0.36, 1.2e-7 | 0.34 |
| same, 4 x 128 tokens | 0.80, 1.6e-7 | 0.62, 1.8e-7 | 1.25, 1.8e-7 | 0.59, 1.8e-7 | 0.58 |
| Llama style decode step (2 layers, dim 256, GQA, batch 4) | 1.82, 1.0e-6 | 2.67, 2.6e-7 | 0.90, 5.2e-7 | 0.67, 8.3e-7 | 0.56 |

float16 rows are in <research>/mac_real/speed_table_after.md (errors 6e-4 to 2e-3 for every framework). Truestar's
accuracy is at the level of the others in every row. Times after the second round of the same day
(vector staged and directly loaded matmul operands, a batch axis folded into the rows, vector staged
attention chunks; every step bitwise equal to the one before it): the 32 token encoder is 1.1x
faster than MPS eager, the decode step 1.44x faster, the 128 token encoder 1.29x and the MLP 1.55x
slower; mx.compile is 1.3x to 3.3x faster than Truestar. Before that round: the decode step was 1.27x faster than
MPS eager (it took 6.1 ms before this session: four rotary `pow` ops ran in the float64 CPU
fallback on every call, now a kernel), the others 1.2x to 2.1x slower than MPS eager; against
mx.compile Truestar is 1.8x to 3.8x slower (host time per step across 94 steps, the 32 x 32
contraction tile on narrow matmuls). Before this session's attention shapes the encoder at 128
tokens took 6.58 ms and at 32 tokens 1.09 ms. torch 2.15.0 gives the same times within 2%. A 64 x
64 contraction tile was 3x to 3.8x slower end to end and is not included.

**Attention kernel alone** (`att_kernel_bench.py`, float32, B 4, H 4, D 32 unless noted, median ms):

| shape | loops (before) | rows | simd | MPS SDPA | MPS softmax(QK^T)V |
|---|---|---|---|---|---|
| 128 x 128 | 6.33 | 0.64 | 0.35 | 0.27 | 0.39 |
| 128 x 128 causal | 5.67 | 0.63 | 0.33 | 0.27 | 0.41 |
| 512 x 512 causal, B 2, H 8, D 64 | 322.8 | 6.58 | 2.63, now 1.82 | 0.68 | 2.22 |
| 1 query, 64 keys (decode), H 8 | 0.42 | 0.20 | n/a (rows chosen) | 0.28 | 0.29 |

Errors 1.8e-7 to 5.7e-7 for every Truestar row. The SIMD group shape is 2.2x to 2.5x faster than
rows on long sequences and still 3.9x slower than MPS SDPA at 512 tokens. A query tile variant with
plain multiply adds was measured no faster than rows and is not included; nor are, from the second
round, direct device loads of key and value blocks (3.06 ms at 512 tokens) and a one pass form with
per element rescaling of the output (2.30 ms), both slower than the staged two pass kernel.

**A Metal ordering hazard.** `simdgroup_barrier(mem_flags::mem_threadgroup)` did not make a lane's
threadgroup store visible to the next `simdgroup_load` on the M2: rows 7 and 15 of a tile lost one
key (reproduced with structured inputs, <research>/mac_real/scripts/simd_att_debug2.py).
The attention kernels use threadgroup barriers instead.

**Cross chip bits.** The portable fingerprint (same weights and inputs as `truestar fingerprint`,
golden hashes matched on the Mac) run through the compiled program on Metal: run to run bitwise
equal and batch invariant (row 0 the same bits at batch 1, 7, 64, 256) for all 4 models in float32,
float16 and bfloat16. MPS eager is run to run equal but not batch invariant for the MLP and resnet18 and the decode
step in float32 (and resnet18 and the decode step in float16). Against the CUDA hashes (RTX 5070 Ti, T4): 0 of 32 float32 rows
equal; only a few float16 and bfloat16 rows coincide (Metal and CUDA kernels sum in different
orders, as section 2 says).

## 6. Limits (2026-11-25)

* One physical Mac measured (M2, macOS 26.6.2); M3, M4 and other macOS versions not run.
* No autotuning: one fixed config per shape. No implicit GEMM convolution (`loops` shape). The
  contraction uses one 32 x 32 tile for every shape. Compiled in checks (sentinels) are refused
  (TS842).
* Offsets, sizes and the grid are 32 bit (TS845 above that).
* The memory planner's in place reuse is CUDA only, so Metal programs allocate every intermediate.
* bfloat16 needs Metal 3.1 (macOS 14 or later).
* A float32 kernel that cancels (a one pass variance on values near 3000, examples/check_model.py)
  cannot pass the float64 check, on Metal as on CUDA and the CPU: the compiled executor refuses it
  (ProgramMismatchError), and `truestar check --backends metal` reports TS002 with that message.
* `.github/workflows/metal.yml` can be dispatched only once it is on the default branch (GitHub's
  rule for `workflow_dispatch`); the runs above went through a copy of its job in the branch's
  `backends.yml`, restored afterwards.
