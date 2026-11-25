# Sliced float64 matmul for verification

Author: Sharven Rane. First written 2026-11-24. Code: `src/truestar/kernel/f64emu.py`, hooked into
the device reference by `reference.einsum64`. Research item 1.8 of
<research>/LIVE_RESEARCH_2026_11.md (FP64 emulation on low precision tensor cores,
Ozaki schemes).

## Why

Every compiled kernel is checked against a float64 reference on its first call (docs/KERNEL_IR.md,
verification). On a consumer GPU float64 runs at 1/64 of float32: measured 0.63 TFLOPS on the RTX
5070 Ti, against about 190 TOPS for an int8 matmul written in Triton. So the float64 reference of a
large contraction is the slow part of checking it.

## Method

`sliced_matmul(a, b)` computes `a @ b` for float64 `a` (M x K) and `b` (K x N) from exact integer
products. It is written from the math of error free splitting (Ozaki's scheme family), not from any
implementation:

1. Scale each row of `a` and each column of `b` by a power of two so its largest magnitude lies in
   [1/2, 1) (exact).
2. Cut every scaled entry into `s` signed 7 bit digits, `a = sum_p a_p 2**(-7p) + da`, `|a_p| <=
   127`, every digit with the entry's sign, `|da| < 2**(-7s)` (exact float64 steps; one fused Triton
   kernel on CUDA, the same steps in torch otherwise).
3. For each digit weight `d = p + q <= s + 1`, add the products `a_p @ b_q` exactly as int8 tensor
   core matmuls with int32 accumulation (K is cut into chunks so the int32 sum cannot overflow),
   then add `S_d * 2**(-7d)` into a float64 result, smallest weight first. On CUDA this is one fused
   Triton kernel per weight; `engine="int_mm"` uses `torch._int_mm` per digit pair with int64 sums.
4. Unscale (exact).

Proven bound per entry, in scaled units (`P = |A'| |B'|`, `rA` and `cB` the row and column sums of
the scaled operands, `DA_i` the sum of the absolute digits of row i, `u = 2**-53`):

    2**(-7s) (cB_j + rA_i) + 2**(-7(s+1)) DA_i + gamma_t P_ij,    t = float64 additions into C

A native float64 matmul, in any summation order, is guaranteed only `gamma_K P_ij`. `P` is bounded
from below and above by one float32 matmul of `|A'|` and `|B'|` with its own rounding accounted for.
`s` is the smallest from 6 to 10 that meets the native guarantee on all but 0.1% of entries; the
entries that still miss it are recomputed as native float64 dot products. So every entry returned
carries a proven bound no larger than native float64's. `SlicedInfo.worst_bound_ratio` is the
largest (our bound) / (native guarantee) over entries.

It is not trusted blindly. The first call for each shape and digit count is compared with a CPU
float64 matmul on 64 sampled rows; a difference beyond the sum of both proven bounds raises
`SlicedF64Mismatch` (TS1085). The CUDA 13.5 release notes fixed a silent wrong result in the vendor's
own float64 emulation path, which is why the check exists. Non finite operands, CPU tensors or
shapes it cannot take raise `SlicedF64Error` (TS1086) and verification keeps the native einsum.

## Using it

`TRUESTAR_VERIFY_F64=sliced` (default `native`) makes the device reference (`run_reference_torch`)
send every two operand float64 contraction with at least `SLICED_MIN_WORK` = 2**30 multiply adds
and one contracted letter through `sliced_einsum` (batch letters loop). Smaller contractions, chains
of several sums (convolutions) and anything it refuses stay on `torch.einsum`. It is opt in: the
default reference is unchanged. It is never a user facing matmul.

## Measured (2026-11-24, scripts/bench_sliced_f64.py, square matrices, ms per matmul)

Accuracy against exact rational dot products (30 sampled entries, error over `|A||B|`): Gaussian
operands 4.5e-17 to 8.0e-17 sliced against 1.3e-16 to 2.2e-16 native GPU float64; operands with
exponents spread over 2**-20 to 2**20: 8.2e-17 to 2.0e-16 sliced against 9.8e-16 to 1.3e-15 native.
The sliced result was the more accurate one in every case, as the bound predicts (it combines s
terms, not K). Integer matrices are exact both ways (tests/test_f64emu.py).

NVIDIA L4 on Modal (a quiet GPU, results/sliced_f64/l4_modal.jsonl):

| n | operands | digits | sliced | native GPU float64 | CPU float64 | speedup |
|---|---|---|---|---|---|---|
| 1024 | Gaussian | 8 | 4.6 | 5.3 | 41.5 | 1.16x |
| 2048 | Gaussian | 8 | 26.6 | 43.2 | 283 | 1.62x |
| 2048 | wide exponents | 9 | 30.0 | 43.0 | 273 | 1.43x |
| 4096 | Gaussian | 8 | 180 | 334 | | 1.85x |
| 4096 | wide exponents | 9 | 207 | 334 | | 1.61x |
| 8192 | Gaussian | 7 | 1177 | 2656 | | 2.26x |
| 8192 | wide exponents | 9 | 1674 | 2656 | | 1.59x |

Verification of a 4096 x 4096 x 4096 float32 matmul kernel (call with the float64 check,
`TRUESTAR_VERIFY=all`): 373 ms native reference, 182 ms sliced (2.05x); the kernel itself 9 to 11 ms.

RTX 5070 Ti (results/sliced_f64/rtx5070ti.json): another process held the GPU at 99% during that
run, so its figures are only indicative: 4096 Gaussian 157 ms sliced against 424 ms native (2.7x),
8192 1.7x to 2.0x, and the 4096 kernel check 253 ms native against 201 ms sliced. Earlier on an idle
GPU (before the digit kernel was fused) 4096 took 92 ms sliced against 228 ms native (2.5x). Below
2048 it is slower than native (fixed costs: digits, bounds, one float32 matmul for P).

No entry needed a native recomputation in these runs; the tests force one ("tiny halves": half of
every row 1e-200 smaller) and check every returned entry against the native guarantee with exact
arithmetic.

## Limits

* Speedup is 1.4x to 2.9x measured, far from the vendor claims for FP64 emulation on data center
  parts (up to 83x on B300 in arXiv 2609.37693): consumer int8 throughput, 7 bit digits (int8) and a
  float32 matmul for the bound all cost; no FP8 or FP4 digits yet.
* Two operands with one contracted letter only; batches loop in Python.
* Overflow or a subnormal result after unscaling is not covered by the bound (native float64 has the
  same exposure).
* The bound check needs `K * 2**-24` small (true for K up to millions).
