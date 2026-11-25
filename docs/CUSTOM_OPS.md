# Native functions as PyTorch custom ops

Author: Sharven Rane. First written 2026-11-24.

Pain row served: P17 (custom ops need C++ and CUDA builds that are slow and break across torch and
CUDA versions, research item N5 in docs/PAIN_RESEARCH_2.md). Code in `src/truestar/customop/`,
tests in `tests/test_customop.py`, measurements by `scripts/customop_bench.py`.

## Use

```python
import torch
import truestar
from truestar import Tensor, ops

@truestar.torch_op
def rms_norm_residual(x: Tensor["B", 128], r: Tensor["B", 128], w: Tensor[128]) -> Tensor["B", 128]:
    h = x + r
    return h * ops.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6) * w

y = rms_norm_residual(x, r, w)                  # calls torch.ops.truestar.rms_norm_residual
y.sum().backward()                              # gradients from truestar.ir.grad
torch.compile(model, fullgraph=True)            # the op is one node, no graph break
torch.library.opcheck(rms_norm_residual.overload, (x, r, w))
```

`truestar.torch_op` takes a plain typed function or an existing `@truestar.function`. Options:
`name=` (default: the function name; a taken name gets `_2`, `_3`), `static=` (values for non tensor
parameters, baked in), `dims=` (symbol ranges), `kernels=False` (force the reference interpreter),
`autograd=False`. The result is a `TorchOp` with `.overload` (the `OpOverload`), `.op` (the
`CustomOpDef`), `.program`, `.grad_program`, `.plan(device, backward=False)` and `.report()`.

Nothing here calls a C, C++ or CUDA compiler. The kernel layer emits Triton (GPU) or numba (CPU)
code from Python, and registration uses `torch.library.custom_op` with a schema string.

## What registration builds

| Piece | Where it comes from |
|---|---|
| Schema | the typed signature: one `Tensor` per `Tensor[...]` parameter, one `Tensor` per returned value, `mutates_args=()` |
| Fake (meta) implementation | the IR shape rules. Each plain symbol of the signature (`B`) is bound to the incoming size, which is a `torch.SymInt` under torch.compile. Fixed dims, repeated symbols and symbol ranges are stated with `torch._check`, and every output dim is the IR's own dim expression (`4*B`, `B + 1`) evaluated on those sizes, so the batch stays symbolic |
| Autograd formula | `truestar.ir.grad(program, wrt=float inputs, of=float outputs, seed="input")`, pruned to the gradient outputs and registered as a second custom op `<name>_backward` with its own derived fake implementation. `register_autograd` saves the inputs and calls that op, so AOTAutograd traces the backward like any other op |
| Real implementation | `customop.execute(program, inputs)`, see below. The same argument checks as the fake run first, so a wrong dtype or an inconsistent symbol raises in eager too, naming the argument and dim |

A function the autodiff pass cannot differentiate (today: any `cond` or `loop`) still registers.
Its forward works everywhere, and a backward through it raises `RuntimeError: truestar::<name> has
no backward: <the autodiff error naming the op>`. Nothing is silently zero.

## Execution paths

`customop.execute` picks the fastest path per program and device type, once, and caches the plan:

1. **Registered executors** (`customop.register_executor(name, factory)`). This is the seam for the
   compiled executor being built elsewhere: `factory(program, device_type)` returns a runner or None.
2. **The kernel layer** when every op lowers (`customop/lower.py`). Triton on CUDA, numba on CPU,
   with the kernel layer's fusion. Lowered today: elementwise arithmetic and math, comparisons,
   `where`, `clamp`, tanh `gelu`, `silu`, `sum` and `mean` over any dims, `softmax`, `matmul`,
   `linear`, `layer_norm` over the last dim, `attention` without a mask operand, `permute`,
   reshapes that only add or remove size 1 dims, `expand`, `full` and `full_like`.
3. **The float64 reference interpreter**, cast back to the declared dtype and moved back to the
   input device. Choosing it emits `customop.SlowPathWarning` once per program and device, with the
   reason (for example: exact `gelu` needs `erf`, which the kernel IR does not have).
   `op.plan(device).reason` carries the same text. On a CPU without numba the kernel layer is
   refused with code TS022 and the reason names the fix, `pip install "truestar[cpu-kernels]"`.

Every fast path result is checked against the float64 reference interpreter on the first call per
input signature (`TRUESTAR_VERIFY=all` checks every call, `0` turns it off), on top of the kernel
layer's own per kernel check. A mismatch raises `customop.ProgramMismatchError`. This catches a
lowering bug that the kernel layer cannot see, because the kernel layer only compares a kernel
with its own IR.

## Measured (2026-11-24, Windows 11, torch 2.11.0+cu128, triton 3.7.1, numba 0.68.1, RTX 5070 Ti)

`scripts/customop_bench.py` runs each device in a fresh process with empty Truestar, numba and
Triton cache directories. No `cl`, `nvcc`, `gcc`, `g++` or `clang` is on PATH on this machine, and
`torch.utils.cpp_extension.CUDA_HOME` is None.

The three ops: a fused tanh GELU MLP with residual (`x [B, D]`, hidden `HID`), attention with a
tanh softcap score modification (head dim 64), and RMSNorm over a residual sum. CUDA sizes: D 512,
HID 2048, B 4096; attention B 2, H 16, S 1024; RMSNorm 8192 x 4096. CPU sizes are smaller (D 128,
HID 512, B 64; attention 1 x 2 x 64; RMSNorm 64 x 1024) because the numba backend is a correctness
backend with plain loop nests.

**Cold registration** (decoration, IR translation, autodiff, both `torch.library` registrations;
after `import torch`): 0.28 s to 1.21 s per op, worst is the attention op (its autodiff pass is
about 0.5 s). `import truestar` adds 0.6 s to 0.9 s. A test runs this in a fresh process with PATH
stripped to the Python and OS folders and asserts under 2 s.

**First call** (code generation, Triton or numba compile, verification), cold caches:

| op | CPU fwd / bwd | CUDA fwd / bwd | CUDA fwd / bwd with `TRUESTAR_VERIFY=0` |
|---|---|---|---|
| GELU MLP | 4.1 s / 4.7 s | 5.0 s / 5.9 s | 3.0 s / 3.0 s |
| softcap attention | 1.5 s / 14.1 s | 5.1 s / 13.6 s | 1.6 s / 3.5 s |
| RMSNorm residual | 0.6 s / 2.6 s | 3.4 s / 11.6 s | 0.4 s / 1.5 s |

On CUDA most of the first call is the float64 verification at these sizes; on CPU it is the numba
compile (with verification off the CPU numbers barely move). The N5 target of a cold compile under
2 s is met for registration, not for the first call of a large op with verification on.

**Warm calls** (median of 20, forward only, no grad):

| op | CUDA Truestar | CUDA torch eager | CPU Truestar | CPU torch eager |
|---|---|---|---|---|
| GELU MLP | 5.11 ms (3 kernels) | 2.44 ms | 33.9 ms | 2.24 ms |
| softcap attention | 1.65 ms (1 kernel) | 4.62 ms | 21.9 ms | 1.12 ms |
| RMSNorm residual | 18.0 ms (1 kernel) | 1.79 ms | 2.87 ms | 6.03 ms |

The fused attention variant is 2.8x faster than eager PyTorch. The MLP is 2.1x slower than eager
(cuBLAS) and the RMSNorm forward is 10x slower: the kernel layer's tile schedule for a 4096 wide row
reduction is far from bandwidth bound. Speed of the generated kernels is the kernel layer's job and
is not claimed here.

**Accuracy** against torch float64, relative to the largest reference value: CUDA 1.1e-7 to 2.3e-6
(torch float32 on the same inputs: 1.3e-7 to 1.7e-6), CPU 3.4e-8 to 6.9e-8 (torch float32: 8.3e-8
to 4.9e-7).

**torch.compile**: `torch._dynamo.explain` reports 0 graph breaks and 1 graph for each op inside a
small model. With `fullgraph=True, dynamic=True` and a counting backend, batch sizes 3, 5 and 7 give
one graph. `backend="aot_eager"` with `dynamic=True` traces forward and backward through both
fake implementations and matches eager values and gradients. `backend="inductor"` works on CUDA
with backward. Inductor on CPU fails here with `Compiler: cl is not found`, which is Inductor's
own C++ code generation for CPU and happens with or without a Truestar op in the graph.

## Against `torch.utils.cpp_extension`

Not built here, and it could not be: this machine has no MSVC (`cl`) and no CUDA toolkit (`nvcc`,
`CUDA_HOME` is None), while the custom ops above run on both CPU and the GPU. Documented numbers
for the C++ route, from the sources in docs/PAIN_RESEARCH_2.md: `load_inline` compile times "on the
order of 90s for toy kernels" (pytorch/pytorch#152032, an RFC by the maintainers), "over 6 hours to
recompile tens custom operators" (PyTorch forums, 2025-09-04), an extension that stops building on
Windows above torch 2.8 (pytorch/pytorch#166123), and ABI breaks when CUDA moved from 12.x to 13.x
(pytorch/pytorch#187320). These are other people's measurements on other machines, not a side by
side run.

## Tests (`tests/test_customop.py`, 29 tests, 7 of them need CUDA)

* Forward and gradients of the three ops against torch float64 at B 1, 3 and 5, on CPU and CUDA:
  outputs within 1e-5, every input gradient within 1e-4, both directions on the kernel path.
* `torch.library.opcheck` (schema, autograd registration, fake tensor, AOT dispatch dynamic) for
  the three ops on CPU and CUDA, with and without `requires_grad`, and for the reference path op
  and the two output op.
* Zero graph breaks, one graph over three batch sizes, eager equals compiled with backward,
  Inductor on CUDA with backward, the fake implementation keeps `B` a SymInt under `make_fx`.
* The slow path is warned with its reason and is correct; `kernels=False` agrees with the kernel
  path; a control flow op runs forward and, since 2026-11-24, backward too (gradients through its
  `cond` match PyTorch on both sides of the branch, tests/test_customop.py), and compiles without a
  break.
* The executor seam takes over and can be removed. An executor returning a 1% wrong backward is
  refused with `ProgramMismatchError`.
* Cold registration in a fresh process with no compiler on PATH, under 2 s.

Tests that prove the tests work:

* Permanent: a fake implementation with one column too many fails opcheck.
* Manual, 2026-11-24, each restored: reversing the output shape in `_fake_outputs` failed 12 tests
  (opcheck for every op on CPU and CUDA, the eager versus compiled tests, the SymInt test, the
  reference path and two output tests); lowering `sub` as `b - a` failed 5 tests, every one through
  `ProgramMismatchError` on the GELU MLP backward (the only program using `sub`) on CPU and CUDA.

## Limits (2026-11-24)

* **One torch version only.** The acceptance row asks for the same source on two torch minor
  versions on Linux and Windows. Only torch 2.11 on Windows was run (no new environments may be
  made here). Not verified.
* First calls are slow: up to 14 s for the attention backward on CPU (numba) and up to 13.6 s on
  CUDA with verification on. Warm caches on disk cut code generation, not verification.
* Warm speed is mixed (see the table). The reference path is much slower again and says so.
* Not lowered to kernels: exact `gelu`, `select` and `slice`, reshapes that merge or split dims
  (so the backward of `linear` on rank 3 inputs), masks given as operands, control flow,
  non float tensors, symbolic dims written as expressions in the signature. These run on the
  reference path with a warning.
* No autograd through `cond` or `loop`; no double backward (the backward op has no autograd).
* Static parameters are baked in at registration (`static=`); one registration per value set.
* Only `Tensor` arguments and results in the schema; no scalars, lists or optional tensors.
* No output ever aliases an input (always a new tensor); non contiguous inputs were not tested.
* `torch.export` with the op inside was not tested, nor MPS or ROCm.
