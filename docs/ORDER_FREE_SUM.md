# Order free sums

Author: Sharven Rane. First written 2026-11-24. Code: `src/truestar/kernel/ordersum.py`, the `osum`
reduction of the kernel IR (`kernel/ir.py`, `kernel/triton_backend.py`, `kernel/cpu_backend.py`).
Research item 1.4 of <research>/LIVE_RESEARCH_2026_11.md.

## The problem

A float32 sum depends on the order of its additions. Truestar got bitwise equal results across GPUs
by pinning that order: no split K, fixed chunk sizes, reduction launch configs never tuned in
deterministic mode (docs/KERNEL_IR.md, determinism). So speed and determinism fought: the fastest
block size or warp count for a reduction could not be used, because it changed the bits.

## What an order free sum is

`K.order_free_sum(axis, body)` (IR op `osum`) is a float32 sum whose result has the same bits for
every order of the terms, every block size, split and SM count:

1. Anchor. `emax` is the largest biased exponent field among the row's finite terms (1 for a row of
   zeros or subnormals). Every finite term is below `2**(emax - 126)`. A maximum is exact, so the
   anchor does not depend on the order either.
2. Grid. The grid step is `2**L`, `L = emax - 126 - 39`: each term keeps 39 bits below the top of
   the row's range. A term becomes the integer `q = round_half_even(x / 2**L)`, `|q| < 2**39`.
3. Exact total. The q are added in int64. Integer addition is associative, and at most `2**24 - 1`
   terms of magnitude below `2**39` stay below `2**63`, so the total is exact in any order.
4. One rounding. The total is converted to float32 (round to nearest even) and scaled by `2**L`
   (exact, except a second rounding when the result is subnormal).

Special values are counted, so they are order free too: any NaN, or both infinities, gives NaN; any
+inf gives +inf; any -inf gives -inf. A zero total is +0.0, also when every term is -0.0 (IEEE would
give -0.0).

Error bound, proven: `|osum - exact| <= n * 2**(L - 1) + half an ulp of the result`, and `2**L <=
2 * max|x| * 2**-39`, so the grid error is at most `n * max|x| * 2**-39`. A float32 pairwise sum is
only guaranteed `log2(n) * 2**-24 * sum|x|`. `ordersum.error_bound` returns the first part per row.

The name. This is one 64 bit fixed point accumulator anchored at the row's top exponent (a single
exponent bin), not the multi bin scheme of the reproducible summation literature (Demmel and
Nguyen; NVIDIA CCCL's `gpu_to_gpu` level), which the research item cites. One bin is enough for
float32 terms because 39 bits of headroom below the top already beat a float32 sum by orders of
magnitude, and it costs one int64 add per term.

## Where it runs

| path | what |
|---|---|
| `ordersum.order_free_sum(x, dim)` | torch integer ops on any device; `chunks=` and `perm=` exist to show order freedom |
| `ordersum.order_free_sum_numpy` | an independent NumPy implementation of the same definition |
| `ordersum.order_free_sum_cuda(x, block, programs)` | a whole tensor sum: two Triton passes (anchor per program, int64 partial per program) and a finish kernel |
| kernel IR `osum` on `cuda` and `rocm` | the tile reduction shapes (`tile_single`, `tile_looped`) of the Triton backend; the term is turned into its grid integer with exact float steps (two power of two scales, truncation, half even correction), the special values with one or reduction |
| kernel IR `osum` on `cpu` | the numba backend rounds the body to float32, then the same integer steps |
| kernel IR `osum` on `metal` | the rows and loops shapes: anchor and int64 total per thread, combined by an integer tree in threadgroup memory; the term grid, the int64 to float32 rounding and the final scaling done on the bits (no float conversion or subnormal rule of the GPU involved) |
| float64 reference | an `osum` is a plain sum in float64 (same value), so verification is unchanged |

`K.compile(k, backend, order_free=True)` rewrites every sum of a kernel that plans as a row
reduction (softmax denominators, norms, row sums, loss sums) into an `osum`. Contractions and
attention keep their sums: they run on tensor cores, where an integer accumulator per product is the
wrong trade (research item 1.4 says the same). The compiled kernel's `order_free` dict says which
way it went and why.

An order free tile kernel has the same bits for every launch config, so `bits_follow_config` is
False for it and it is tuned even with `deterministic=True` (the plain reductions keep their pinned
default config). A max or min next to the osum keeps that property: both are exact; the one value
that can still depend on the order is the sign of a zero maximum among +0 and -0 terms.

What "same bits on every chip" now means for these reductions: the sum itself is order free on any
device, since it uses only integer adds, comparisons, exact power of two scaling and one IEEE
conversion. The body's own bits must match too. Loads, products and additions of float32 values are
IEEE correctly rounded everywhere, but a body with exp or tanh inherits the vendor's math library
(research item 1.7), and the Triton compiler may contract `a * b + c` into a fused multiply add.

## Measured (2026-11-24)

Bitwise (tests/test_ordersum.py, all pass on the RTX 5070 Ti):

* the torch and NumPy definitions agree bit for bit on rows with Gaussian, wide exponent (2**-60 to
  2**60), subnormal, overflowing, cancelling, all -0.0, exact half and tie on the grid terms, and on
  NaN and infinity rows;
* the same bits after permuting the terms and cutting them into 1, 2, 3, 7, 64 or 599 partial sums;
* the Triton kernel equals the definition for every launch config tried (BR 1, 2, 4 with 2 to 8
  warps for rows up to 4096; BC 128, 1024, 4096 for looped rows of 9000), tuned and deterministic;
  the numba kernel equals it too; a dev sweep of 30 random edge case batches (rows of 7 to 30,000
  terms) gave 0 mismatches on CPU and CUDA;
* compiled softmax with `order_free=True`: equal bits over 3 launch configs; the plain softmax over
  the same configs differs;
* `order_free_sum_cuda` gives the same bits for 5 block and program counts (1 to 1000 programs) and
  after a random permutation of 16,777,215 terms, where `torch.sum` changed its bits.

Accuracy (256 rows of 4096 terms, error over the sum of absolute terms, worst row,
scripts/bench_order_free_sum.py): Gaussian 2.2e-9 (torch.sum on CUDA 8.1e-9), wide exponents 8.1e-9
(2.5e-8), cancelling halves 5.0e-15 (4.3e-9); every row within the proven bound.

Speed, RTX 5070 Ti, median of 100 calls, kernel IR, in us: default launch config / tuned with
`deterministic=True` (results/order_free_sum/bench.json, the last of four runs; another process
used the GPU during parts of the earlier three, and single figures moved by up to 2x between runs,
so the last column gives the range of the default config slowdown over all four):

| kernel, shape | plain sum | order free | slowdown (this run) | slowdown range, 4 runs |
|---|---|---|---|---|
| row sum 4096 x 1024 | 44.5 / 41.4 | 60.0 / 60.5 | 1.35x / 1.46x | 1.35x to 1.9x |
| row sum 1024 x 4096 | 37.7 / 39.8 | 61.2 / 61.8 | 1.62x / 1.56x | 1.3x to 2.4x |
| row sum 64 x 65536 (looped) | 58.6 / 59.2 | 116.6 / 75.3 | 1.99x / 1.27x | 1.7x to 2.0x |
| softmax 4096 x 1024 | 46.9 / 49.2 | 81.2 / 69.7 | 1.73x / 1.42x | 1.5x to 1.9x |
| softmax 1024 x 4096 | 65.9 / 51.1 | 78.5 / 78.4 | 1.19x / 1.53x | 1.2x to 1.7x |
| softmax 64 x 65536 (looped) | 108.8 / 119.3 | 188.4 / 96.5 | 1.73x / 0.81x | 1.4x to 1.7x |
| layer norm 4096 x 1024 | 59.1 / 70.3 | 118.3 / 63.1 | 2.00x / 0.90x | 1.5x to 2.9x |
| layer norm 1024 x 4096 | 54.0 / 42.1 | 116.5 / 121.0 | 2.16x / 2.87x | 2.2x to 5.0x |
| layer norm 64 x 65536 (looped) | 96.1 / 87.2 | 242.5 / 137.2 | 2.52x / 1.57x | 2.2x to 2.9x |

So one order free sum per row costs about 1.2x to 2x; a layer norm (two sums, the second over a
computed body, evaluated twice in looped rows) 1.5x to 5x. Because an order free kernel may be
tuned in deterministic mode, its tuned figure sometimes beats the pinned plain one (0.81x and 0.90x
above). Every plain row differed in its bits across the three launch configs tried; every order
free row was equal. The standalone whole tensor sum (`order_free_sum_cuda`, three launches) takes
73 us for 2**20 terms against 34 us for `torch.sum`, and 190 us against 89 us for 2**24 - 1 terms.
The torch integer op version takes 1.3 to 29 ms (many full size int64 temporaries): a reference,
not a fast path.

## Limits

* Rows of at most `2**24 - 1` terms (TS1080), float32 terms only (half precision terms widen to
  float32 exactly first).
* Not for contractions and attention: their sums stay ordered and pinned (docs/KERNEL_IR.md).
* On Metal (Mac mini M2, 2026-11-25) the osum equals the definition bit for bit on every hard row
  of the tests, subnormal terms and totals included, through a threadgroup per row and one thread
  per row. A body with arithmetic inherits the GPU's own float results, and the M2 flushes
  subnormal float32 arithmetic to zero (loads and stores keep subnormals; measured in
  <research>/mac_real/ftz_probe.txt), so rows of subnormal products differ from a CPU
  or CUDA run there; on rows of normal products the bits are the same.
* The body's own bits are not made portable here (exp, tanh: research item 1.7; FMA contraction).
* +0.0 for a total of -0.0 terms, a documented difference from IEEE.
* The two pass looped shape evaluates the body twice.
