# Portable tensor cores: exact tensor core blocks

Author: Sharven Rane
Date: 2026-11-24

Portable tensor cores are an opt in mode in which float16, bfloat16 and TF32 dot products still run
on the tensor cores but give the same bits on every NVIDIA generation from sm_80 on. Code:
`src/truestar/kernel/tcexact.py` (method, proof, Triton helpers, NumPy twin), wired into the
contraction, convolution and attention kernels of `kernel/triton_backend.py` and
`kernel/triton_conv.py`. Tests: `tests/test_tcexact.py` (CPU), `tests/test_tcexact_gpu.py` (GPU).

```python
from truestar import kernel as K
from truestar.compiler import compile_program

fn = K.compile(kernel, "cuda", portable_tensor_cores=True)                 # one kernel
cp = compile_program(program, "cuda", "float16", portable_tensor_cores=True)
cp = compile_program(program, "cuda", "float32", allow_tf32=True, portable_tensor_cores=True)
```
```
truestar fingerprint --arms truestar_det,truestar_portable --json fp.json
```

## Why

The cross chip run of 2026-11-24 (<research>/xchip, commit 97fd571) found every
float32 fingerprint row bitwise equal on A10G, L4, L40S, H100 and an RTX 5060 laptop, and every
half precision and TF32 row split into two groups: Ampere and Ada (sm_86, sm_89) against Hopper and
Blackwell (sm_90, sm_120). The probe (`scripts/xchip_mma_probe.py`) located it in one instruction:
Triton 3.7.1 emits `mma.sync m16n8k16` (`m16n8k8` for TF32) on all of them, and inside it the
products are aligned to the largest; sm_86 and sm_89 keep a product 2^-24 below the largest and drop
one 2^-25 below, sm_90 keeps it.

## Method

Make every tensor core instruction's sum exact, then round in software in a fixed order. Per K block
of 64 columns (a constant of the mode, never a tuned value), each operand row (of A) or column (of
B) is scaled by a power of two read from its block maximum and cut into three integer digits
(`|d0| <= 127`, `|d1|, |d2| <= 64`, weights 1, 2^-7, 2^-14, nearest integer steps, all exact float32
arithmetic). The six digit products of weight 0, 1 and 2 run as int8 tensor core dot products with
int32 accumulation: integer sums, exact on any GPU. Each weight's integer sum (below 2^22) converts
to float32 exactly, the three are combined by Horner in float32, scaled back by two exact powers of
two, and the K loop adds the block results in increasing K order. Every multiplication is by a power
of two, so whether the compiler fuses it into an FMA changes nothing. The result is a function of the
input values only: tile sizes, warps, pipeline depth, instruction shape and GPU generation do not
enter. The full step list is in the module docstring of `tcexact.py`.

TF32 inputs are rounded to TF32 to nearest even on their bits by the kernel (plain TF32 truncates
in hardware), so the TF32 value itself does not depend on the GPU.

K blocks of 16 (attention with head dimension 16) use the same digits as float16 values on the fp16
tensor core path; the two engines give the same bits (measured).

### The window constant and the proof

`TC_FRAC_BITS_MIN = 24`: the narrowest alignment window of any accepted generation, in fraction bits
below the exponent E of the largest term. sm_80, sm_86 and sm_89 keep 2^(E-24) (the xchip probe on
A10G and L4: a product 2^-24 below 1.0 is kept, 2^-25 dropped; the Ada model of
docs/TENSOR_CORE_MODELS.md, fitted bit for bit on an L4, has 24); sm_90 and sm_120 keep 2^(E-25).

Only the fp16 engine depends on it. There every product is an integer below 2^14, the float32
accumulator starts at C = 1.5 * 2^23 and every partial sum is C plus an integer of magnitude at most
2,605,056 < 2^22, so it lies in [2^23, 2^24). Every term is an integer, E = 23, and the grid
2^(E - 24) = 2^-1 is finer than 1: no bit of any term is ever dropped, and the exact sum, an integer
in [2^23, 2^24), is representable in float32, so any rounding of it returns it unchanged. The
argument holds for any group size, instruction shape and order, because it bounds every partial sum.
`check_bounds()` asserts each inequality at import; `tests/test_tcexact.py` runs the digits, worst
cases included, through the bit accurate models of V100, A100, Ada, H100, B200 and sm_120 and gets
the exact integers on each; a unit with a 10 bit window loses bits on the same kind of input.

The int8 engine needs no window: `mma.sync ... s32.s8.s8.s32` and its wgmma form are integer
arithmetic. Its sums stay below 2^22, far from int32 overflow (`satfinite` never triggers).

A pitfall found on the way: Triton folds `x + tl.dot(a, b)` (zero accumulator) into
`tl.dot(a, b, x)`. In a first draft that moved a fractional float32 value into the tensor core
accumulator, and the fp16 engine stopped being exact on L4 (the int8 engine was not affected). The
accumulator now starts at C and every value read from it is converted to an integer first.

### Refusals

| code | when |
|---|---|
| TS1330 | a backend other than CUDA Triton (cpu, metal) |
| TS1331 | ROCm, or an NVIDIA GPU below sm_80 |
| TS1332 | a dot product K block outside 16 to 128 (head dimensions above 128 in attention) |
| TS1333 | `library=` ops with `compile_program` (they run vendor kernels) |

The mode implies `deterministic=True` (no split K) and pins the K block of contraction and
convolution kernels to 64; the tuner still chooses BM, BN, warps and stages, which do not change the
bits.

## Results (Modal, torch 2.11.0+cu130, Triton 3.7.1, 2026-11-24)

### Bits

`truestar fingerprint` (the harness of the xchip run: mlp, transformer, resnet18, llama_step at
batch 1, 7, 64, 256), arms `truestar_det` (plain tensor cores) and `truestar_portable`:

| arm | half and TF32 rows equal on A10G (sm_86), L4 (sm_89), H100 (sm_90) | float32 rows |
|---|---|---|
| truestar_det (plain tensor cores) | 3 of 48 | 16 of 16 |
| truestar_portable | **48 of 48** | 16 of 16 (unchanged) |

Every row is also equal run to run and row 0 is batch invariant. The standalone kernels of
`scripts/tcexact_bench.py` (9 cases) and `scripts/tcexact_twopass_bench.py` (3 cases) give the same
hashes on all three GPUs and equal the NumPy emulation bit for bit; `tests/test_tcexact_gpu.py`
checks compiled contraction kernels against the emulation for four launch configs each (35 passed
on L4). The transformer layer of the speed section has the same output hash on all three GPUs in
each portable arm, and different hashes on H100 in each plain arm.

sm_120 (the laptop group) was not run on a GPU here: the int8 engine is integer arithmetic, and the
fp16 engine's digits are exact through the sm_120 tensor core model (validated bit for bit on the
RTX 5070 Ti, docs/TENSOR_CORE_MODELS.md).

### Accuracy against float64

Matmul (`scripts/tcexact_bench.py`, standard normal operands, error over |A||B| elementwise, max over
the 256 x 256 output; the same on all three GPUs for portable):

| dtype, K | plain tensor cores A10G/L4 | plain H100 | portable | float32 FMA |
|---|---|---|---|---|
| float16, 1024 | 4.9e-7 | 3.6e-7 | 1.0e-7 | 3.1e-7 |
| float16, 4096 | 8.3e-7 | 5.6e-7 | 5.4e-8 | 2.6e-7 |
| bfloat16, 1024 | 3.2e-7 | 2.6e-7 | 4.5e-8 | 1.3e-7 |
| bfloat16, 4096 | 6.5e-7 | 5.0e-7 | 3.4e-8 | 1.9e-7 |
| tf32, 1024 (against the float32 inputs) | 1.7e-4 | 1.7e-4 | 6.1e-5 | 2.7e-7 |
| tf32, 4096 | 8.0e-5 | 8.0e-5 | 2.9e-5 | 2.8e-7 |

Models (fingerprint, largest output error over the output maximum, portable over plain, 16 rows per
dtype): TF32 geometric mean 0.29 (never worse), float16 0.97 (never worse), bfloat16 1.01 to 1.04
(worse in 3 to 5 of 16 rows, at most 1.41x): there the bfloat16 storage of every activation (8 bits)
dominates and which way each stored value rounds moves the maximum both ways.

### Speed

Compiled linear layers `relu(x @ W.T + b)` through `K.compile` (the shapes of scripts/bench_matmul.py, `scripts/tcexact_compiled_matmul.py`), microseconds per call (back to back calls; the first rows are launch bound for every arm). `fma/x` is float32 time over portable time, `x/plain` portable time over plain tensor core time, float16:

| M x N x K | A10G fp32 / fp16 plain / fp16 portable (us) | A10G fma/x | A10G x/plain | L4 fp32 / fp16 plain / fp16 portable (us) | L4 fma/x | L4 x/plain | H100 fp32 / fp16 plain / fp16 portable (us) | H100 fma/x | H100 x/plain |
|---|---|---|---|---|---|---|---|---|---|
| 1 x 4096 x 4096 | 213 / 72 / 88 | 2.41 | 1.23 | 336 / 36 / 100 | 3.36 | 2.75 | 153 / 24 / 60 | 2.55 | 2.46 |
| 16 x 4096 x 1024 | 55 / 21 / 24 | 2.25 | 1.14 | 81 / 21 / 25 | 3.24 | 1.17 | 41 / 22 / 22 | 1.83 | 1.01 |
| 64 x 2048 x 1024 | 65 / 19 / 26 | 2.47 | 1.38 | 87 / 22 / 30 | 2.89 | 1.35 | 49 / 22 / 23 | 2.11 | 1.08 |
| 64 x 2048 x 2048 | 133 / 23 / 55 | 2.40 | 2.36 | 173 / 21 / 61 | 2.86 | 2.87 | 96 / 22 / 37 | 2.58 | 1.67 |
| 64 x 1000 x 2048 | 103 / 19 / 39 | 2.63 | 2.01 | 158 / 21 / 45 | 3.49 | 2.16 | 80 / 22 / 33 | 2.43 | 1.47 |
| 512 x 1024 x 256 | 37 / 19 / 21 | 1.73 | 1.13 | 53 / 19 / 25 | 2.10 | 1.32 | 30 / 22 / 23 | 1.28 | 1.05 |
| 512 x 256 x 1024 | 63 / 19 / 26 | 2.45 | 1.36 | 87 / 20 / 30 | 2.93 | 1.45 | 50 / 25 / 24 | 2.03 | 0.98 |
| 1024 x 1024 x 1024 | 276 / 52 / 160 | 1.73 | 3.09 | 402 / 49 / 187 | 2.15 | 3.82 | 218 / 25 / 40 | 5.39 | 1.60 |
| 2048 x 4096 x 1024 | 1788 / 343 / 1141 | 1.57 | 3.33 | 2098 / 424 / 1270 | 1.65 | 3.00 | 613 / 37 / 280 | 2.19 | 7.65 |
| 4096 x 4096 x 4096 | 14507 / 2466 / 9005 | 1.61 | 3.65 | 16631 / 2942 / 9799 | 1.70 | 3.33 | 4923 / 263 / 2061 | 2.39 | 7.83 |
| 64 x 512 x 8192 | 386 / 36 / 117 | 3.29 | 3.27 | 321 / 35 / 130 | 2.48 | 3.75 | 300 / 23 / 86 | 3.48 | 3.67 |
| 16 x 1024 x 16384 | 763 / 78 / 216 | 3.53 | 2.77 | 635 / 54 / 181 | 3.51 | 3.33 | 600 / 37 / 170 | 3.53 | 4.61 |
| 128 x 256 x 8192 | 383 / 33 / 112 | 3.41 | 3.42 | 320 / 35 / 126 | 2.54 | 3.65 | 300 / 24 / 86 | 3.47 | 3.59 |
| 8192 x 768 x 256 | 382 / 76 / 260 | 1.47 | 3.41 | 434 / 84 / 301 | 1.44 | 3.56 | 171 / 22 / 64 | 2.68 | 2.93 |
| 8192 x 1024 x 256 | 511 / 101 / 342 | 1.50 | 3.39 | 558 / 123 / 394 | 1.41 | 3.21 | 183 / 21 / 86 | 2.12 | 4.04 |
| 8192 x 256 x 1024 | 470 / 95 / 337 | 1.40 | 3.55 | 695 / 92 / 362 | 1.92 | 3.95 | 348 / 22 / 80 | 4.37 | 3.65 |

Geometric means over the shapes whose float32 call takes over 200 us:

| GPU | shapes | float32 / float16 portable | float32 / TF32 portable | float16 portable / plain |
|---|---|---|---|---|
| A10G | 10 | 2.05x | 1.45x | 3.00x |
| L4 | 10 | 2.11x | 1.37x | 3.41x |
| H100 | 7 | 3.40x | 2.47x | 4.15x |

Compiled matmul output hashes equal on the three GPUs: portable 48 of 48, plain 1 of 48.

One transformer encoder layer (d_model 1024, 16 heads, ffn 4096, 8 x 512 tokens, `scripts/tcexact_layer_bench.py`), whole compiled program, milliseconds per call:

| arm | A10G | L4 | H100 | output hash equal on the three |
|---|---|---|---|---|
| float32_fma | 13.14 | 15.57 | 4.67 | yes |
| tf32_plain | 4.21 | 5.01 | 0.74 | no |
| tf32_portable | 10.89 | 11.84 | 2.19 | yes |
| float16_plain | 2.07 | 2.29 | 0.76 | no |
| float16_portable | 8.13 | 8.98 | 1.83 | yes |
| bfloat16_plain | 2.07 | 2.19 | 0.77 | no |
| bfloat16_portable | 8.08 | 8.88 | 1.90 | yes |

| GPU | float32 / float16 portable | float32 / bfloat16 portable | float32 / TF32 portable | float16 portable / plain | TF32 portable / plain |
|---|---|---|---|---|---|
| A10G | 1.62x | 1.63x | 1.21x | 3.92x | 2.59x |
| L4 | 1.73x | 1.75x | 1.32x | 3.91x | 2.36x |
| H100 | 2.56x | 2.46x | 2.13x | 2.41x | 2.94x |

Standalone kernels (`scripts/tcexact_bench.py`, one Triton matmul per method, best of a few tile
configs each, no Truestar tuner) are less flattering, because the hand written float32 FMA kernel
there is faster than the compiled float32 contraction and the exact kernel's small config grid is
slower than what the tuner finds: portable over FMA 0.67x to 1.37x on L4, 0.71x to 1.00x on A10G,
0.69x to 0.75x on H100 (2048 to 4096 square and two linear shapes; 3x to 18x slower than plain tensor
cores). A two pass prototype (`scripts/tcexact_twopass_bench.py`: digitize each operand once, then a
pure int8 matmul; same bits, checked) reached 1.11x to 1.35x of that FMA kernel on H100 and 0.63x to
1.10x on L4: there the six int8 products per product, not the digit cutting, set the floor.

## Limits

* Faster than float32, slower than plain tensor cores: the six int8 products per product cost
  about three plain half precision tensor core passes plus the digit and combination work on the
  CUDA cores. See the multipliers above.
* NVIDIA sm_80 and newer only; not measured on sm_80 itself (A100), sm_120 checked through its model.
* K blocks of at most 128 (attention head dimensions up to 128).
* Two approximations, both fixed functions of the inputs: values far below their K block's
  maximum lose low bits (float16 and TF32 values within 2^10 of it and bfloat16 values within 2^13
  lose nothing), and the digit products of weight 3 and 4 (pairs (1,2), (2,1), (2,2), each below
  2^-21 of the block's largest product) are not formed. The accuracy tables show both stay below
  plain tensor core rounding on these inputs; data whose blocks mix very different magnitudes
  would see more of the first.
* Subnormal float32 results of the final power of two scaling round once, deterministically.
* A CUDA graph capture of compiled kernels gave an illegal address on replay in
  `scripts/tcexact_compiled_matmul.py` for plain float32 as well (not investigated; the script times
  back to back calls instead).
