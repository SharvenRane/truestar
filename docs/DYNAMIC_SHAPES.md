# Dynamic shapes

Author: Sharven Rane. 2026-11-24. Code: `src/truestar/compiler/dynamic.py`, tests
`tests/test_dynamic_shapes.py`, benchmark `scripts/bench_dynshape.py`. Codes TS1420 to TS1427.

One compile serves every batch size and sequence length inside declared ranges, and the result at
every shape is bitwise the result of compiling that exact shape statically.

```python
from truestar.compiler.dynamic import compile_dynamic

dp = compile_dynamic(layer, (x,), axes=[{1: "T"}], dims={"T": (2, 512)}, device="cuda")
# or: compile_dynamic(layer, (x,), dynamic_shapes=({1: torch.export.Dim("T", min=2, max=512)},))
# or: compile_dynamic(program)            # an IR program's own symbols and ranges
y = dp([x_any_length])[0]                 # no recompile anywhere in [2, 512]
dp.prove([(x37,), (x200,)])               # static compile per shape, bits compared (TS1426)
```

## How it works

The compiler already keeps symbolic dims symbolic (docs/COMPILER.md, Control flow and symbolic
dims): one Truestar compile, kernels bind sizes at call time, tails are masked in the kernel
(`lane < N`), Triton does not specialize sizes. Block sizes over a symbolic reduced axis follow the
next power of two, so those are the buckets in which Triton compiles a variant. This layer adds:

* **Declaration.** `axes` and `dims`, or a torch.export spec whose ranges are read back from the
  export. A range torch.export changed is refused (TS1424), so the declared range is the one that
  holds. A call outside it is TS1420, never a recompile.
* **Buckets.** `"pow2"` (default) or a list of upper bounds that must cover the range (TS1422).
  `verify="bucket"` checks each kernel binary once per bucket instead of once per shape
  (`kernel/compiler.py` `verify_group`). The binary is everything Triton compiles a variant for:
  function, constexpr meta, launch config, 1 and divisible by 16 specialization of strides, 16 byte
  alignment of pointers. A shape in the bucket that runs another binary is checked on its own.
  Default stays `"shape"` (see Measured).
* **Bitwise promise** (`bitwise_static=True`, default). A result's bits are set by the launch
  choices that fix summation order. In a deterministic program every one of them follows the exact
  sizes, which a static compile sees too, except where code read *whether* a size is symbolic:
  * `tile_single` row block BR followed a static row count and was fixed for a symbolic one.
    BR and the warps set the layout `tl.sum` reduces a row in, so on an L4 a static compile at
    batch 3 differed from the symbolic one in 64 of 192 softmax outputs (encoder and prefill at
    T = 1: 71 and 70 of 128). **Fixed**: BR no longer follows the row count anywhere
    (`triton_backend.default_config`), so a row's bits also no longer depend on the batch.
    Tuning search space bumped to 12.
  * a convolution's tap loop is taken for a symbolic channel count and for a static one only from
    16: refused with TS1425 when the declared minimum is below 16.
  * non deterministic launches (split K by row count, timed reduction blocks): TS1421.
  * library ops (cuDNN picks algorithms by shape): TS1423.
  Padding does not enter the sums: masked lanes load the reduction's neutral element inside the
  same block, and the block is the same in both compiles, so the order is the same, not just
  close. `prove` is the test that checks it.

## Measured (L4, Modal, float32, deterministic, empty caches per arm, 20 shapes each)

MLP with widths 256, 1024, 256 and a softmax, batch 1 to 128; TransformerEncoderLayer d 256, 8 heads, batch 4,
T 3 to 384; decoder prefill block (pre norm, causal SDPA, GELU MLP) d 256, batch 4, T 3 to 384.

| Model | Arm | Truestar compiles | Triton variants | Time to all 20 first results | Bit differences vs static |
|---|---|---|---|---|---|
| MLP | static per shape | 20 | 60 | 44.8 s | |
| MLP | dynamic | 1 | 3 | 7.0 s (bucket checks 8.2 s) | 0 of 20 shapes |
| Encoder | static per shape | 20 | 140 | 117.8 s | |
| Encoder | dynamic | 1 | 13 | 20.9 s (bucket checks 20.9 s) | 0 of 20 shapes |
| Prefill | static per shape | 20 | 120 | 122.3 s | |
| Prefill | dynamic | 1 | 12 | 21.6 s (bucket checks 20.4 s) | 0 of 20 shapes |

* Time to all first results is 5.6x to 6.4x lower. A new shape inside a seen bucket costs 5 to 60 ms
  (checks) where a static compile costs 2 to 8.6 s. The dynamic variants come at the first call
  (6 or 7), at T = 96 and 128 (a stride starts dividing by 16) and at 256 and 384.
* Step time, dynamic over static at the same shape: MLP median 1.06 (0.55 to 1.24, timing noise at
  0.3 ms steps), encoder and prefill about 1.0 to 1.08 (read off the per shape table). Sizes are
  runtime arguments instead of literals.
* Per bucket verification skipped 78 (encoder) and 86 (prefill) kernel checks but saved no
  measurable time: each check costs 5 to 50 ms at these sizes, inside run to run noise. It is
  opt in until a workload where checks dominate (batch 64 resnet18, docs/COMPILER.md) shows a gain.
* Before the BR fix the same sweep found the differences named above; after it,
  `tests/test_dynamic_shapes.py` (prove at 6 shapes per model) passes on CPU and on the L4.

## Limits

* Not run on any other GPU than the L4 here, not in float16 or bfloat16.
* `verify="bucket"` trusts a kernel binary verified at one shape for the others of its bucket; tail
  paths at other sizes are covered by the kernel test suite, not per call.
* `prove` needs the torch module (TS1427 for an IR program).
* A symbolic spatial size of a convolution is not lowered at all (docs/COMPILER.md, Limits).
