# Lean test: errors that are small but all on one side

Truestar's checks bound how large an error is against float64. A kernel whose errors are all small
but all lean the same way passes a size bound: a store that truncates instead of rounding, a sum
whose partial sums are rounded toward zero, an exp approximation that is always a little high, a low
precision softmax whose rounding leans. That is the failure behind loss explosions in low precision
flash attention training (Qiu and Yao, "Why Low-Precision Transformer Training Fails: An Analysis on
Flash Attention", ICLR 2026, arXiv 2510.04212: biased BF16 rounding in the backward adds up into low
rank weight updates) and the systematic gradient bias of microscaling formats (arXiv 2506.20752). A
training run adds the same small push thousands of times.

The lean test measures the sign of the errors. It runs in two places:

* **Kernel verification** (every compiled Truestar kernel, first call per shape, layer 1 of
  docs/COMPILER.md Verification): an output that leans raises `truestar.kernel.KernelBiasError`
  (**TS960**, a subclass of `KernelMismatchError`) and the result is not used.
* **`truestar check`**: every module replayed alone on a backend is tested; a module whose own errors
  lean gets **TS961** (warning, backend status suspect), listed with its direction, mean, t and p.
  `LayerResult.bias` holds the numbers for every replayed module.

`TRUESTAR_VERIFY_BIAS=warn` turns TS960 into a `KernelBiasWarning`; `off` skips the test.
`TRUESTAR_BIAS_LOG=file.jsonl` appends one line per tested kernel output (used for the survey below,
summarized by `scripts/bias_survey.py`). Code: `truestar/bias.py`, `kernel/verify.py`
(`check_lean`, `error_groupings`), `check.py` (`_lean`, `_report_lean`).

## The statistic

Each finite element gets a normalized error `w_i = (out_i - ref_i) / den_i`, with the size check's
denominator `den_i = max(|ref_i|, f * rms(ref))` (f is 1/8 or 2 for kernels, by whether they reduce,
and 1 for modules), so `w` is in the units the tolerance is written in. Two directions:

* `up`: `w_i` (results lean toward plus or minus infinity);
* `outward`: `w_i * sign(ref_i)` (results lean away from zero, or toward it: truncation).

The bias is the mean `b` of the directed errors over the N finite elements.

## Threshold, derived from unbiased round to nearest

Round to nearest is unbiased: an exact value inside a rounding interval goes up or down with equal
weight and ties go to even, so each element's error has mean zero. The mean of N errors is zero up to
sampling noise, but that noise is not independent across elements: the elements of a softmax row
share one rounded denominator, a normalization shares one rounded scale per row or channel.
Elements that share a rounded intermediate have correlated errors, and treating them as independent
makes a shared rounding look like bias (tests/test_bias.py, `test_a_shared_row_rounding_needs_its_grouping`:
p below 1e-6 as independent elements, above 1e-6 grouped by row).

So the test groups elements by the output dims a shared rounding depends on (a *grouping*: the dims
held fixed), treats the G groups as independent and the elements inside a group as arbitrarily
correlated, and uses the cluster robust standard error of the ratio mean:

    V = G / (G - 1) * sum_g (S_g - b * n_g)^2 / N^2,     t = b / sqrt(V)

with `S_g` the sum of directed errors in group g and `n_g` its count. Under the null t follows
Student's t with G - 1 degrees of freedom whenever the groups are independent, whatever the
correlation inside a group. Every grouping that may carry shared rounding is tested and the largest
p value is kept: the test stays valid as long as one of them matches the real dependence, and an
extra grouping only costs power. p is exact (incomplete beta, checked against SciPy to 1.7e-9
relative over df 1 to 4e6).

* **Kernels** know their dependence: every rounding node of the kernel IR (arithmetic, reductions,
  float64 loads into a lower precision kernel) depends on some output axes, and elements that agree
  on those axes read the same rounded value (`error_groupings`). A softmax gets `[(rows), (elements)]`,
  an elementwise kernel only `[(elements)]`. A non constant value that depends on no output axis (a
  global sum broadcast everywhere) gives one group: that output is never called biased.
* **Modules** are black boxes: every set of dims with at least 16 groups is a grouping. A rounding
  shared by fewer than 16 groups (one scale per sample at batch 4) is not modelled; the effect floor
  bounds what it can do.

Two conditions must both hold, in one direction:

1. **Significance:** largest p over the groupings below `ALPHA = 1e-6` (about 4.9 standard errors
   for many groups). Two directions per output: at most 2e-6 false alarms per output from noise.
2. **Effect:** `|b|` above `BIAS_FRACTION = 1/64` of the size tolerance (kernels: the kernel
   tolerance; modules: the module's threshold). A correct kernel can lean by a fixed amount no
   grouping can separate from bias: a constant such as 1/sqrt(d) or a GELU coefficient is rounded once
   to float32, and that relative error, at most one unit roundoff u (2^-24 in float32), is shared by
   every element. The smallest float32 tolerance over 64 is 3.1e-7 = 5.2 u: room for five aligned
   shared roundings (`test_a_constant_rounded_once_is_inside_the_effect_floor`: x * fl(1/3) is
   significant at p below 1e-6 and stays under the floor). Kernels compute in float32 and round once
   at a float16 or bfloat16 store, where the floor is 0.064 of the output's unit roundoff, which no
   float32 constant reaches. Truncation to the output precision leans by about 0.72 u on average (the
   discarded fraction is uniform, divided by a mantissa in [1, 2)): 11 times the floor in float16 and
   bfloat16.

What this cannot see: a strict float32 store that truncates (0.72 u, under the float32 floor), a lean
shared by every element through one rounding, and in `truestar check` a lean confined to fewer than
16 groups of a black box module.

**Cancelling sums.** An output that leans is measured once more with each element's denominator
raised to the magnitude of its sums' terms (`magnitude_kernel`, the size check's second measure):
rounding in a sum scales with what it added. Crafted cancelling inputs (the order probe's +M, -M and
units, whose absorbed units are always lost the same way) lean in plain units and not in these, and
are not refused; a round toward zero accumulation of positive terms still is (terms magnitude equals
the result).

## Catches (tests/test_bias.py)

Each injected kernel passes the size check (`TRUESTAR_VERIFY_BIAS=off`) and is refused with TS960:

| injected fault | size check | lean |
|---|---|---|
| bfloat16 store by truncation (drop 16 bits) of x * sigmoid(1.702 x) | passes | refused, toward zero |
| row sum of exp(x) over 1024 terms, every partial sum rounded toward zero (float32) | passes | refused, toward zero |
| exp(x) as 2^(x c) with c = log2(e) (1 - 2^-20), x <= 0 (softmax arguments) | passes, error under half the tolerance | refused, toward plus infinity |
| compiled CPU kernel x * 1.7 whose store truncates (float32: 5 bits dropped; float16, bfloat16: truncating cast) | passes | refused, toward zero, all three dtypes |
| `truestar check`: a module whose float32 Linear output loses its low 8 mantissa bits (relative L2 1e-5 against threshold 1e-4) | ok | TS961 on that module only, toward zero, p 3e-20 |

Two broken kernels of tests/test_kernel_verify_elements.py (wrong by 3x on small float16 exp values;
row sums 0.5 percent wrong on ordinary rows) that the size measure before 2026-11-24 passed are also
refused by the lean test alone.

## False positive survey (2026-11-24)

| run | outputs tested | fired | closest margin (1 = fires) |
|---|---|---|---|
| CPU kernel battery on a rented Linux server (50 kernel, compiler, training, nn, autodiff and quant test files + check, devsim, stacks, opdrill, fix, agent, examples tests: 2214 passed, 438 skipped for GPU) | 806 kernel outputs, 333 kernels | 0 | 0.61 (an 8 element cancelling row sum) |
| GPU kernel tests on the RTX 5070 Ti (`-k "cuda or gpu or tf32"` of test_kernel, test_kernel_gpu_verify, test_compiler_tf32, test_kernel_verify_elements, test_conv_kernels: 168 passed) | 199 | 0 | 0.62 (a TF32 linear, after the magnitude recheck) |
| `truestar check` zoo, CPU (resnet18, mobilenet_v3_small, efficientnet_b0, convnext_tiny, vit_b_16, tiny_transformer; `scripts/bias_survey.py --zoo cpu`) | 796 module outputs | 0 | 0.095 |
| same zoo on CUDA (cuDNN TF32 convolutions on, torch default) | 796 | 0 | 0.44 (a TF32 1x1 convolution, 32 groups) |

Margin is the smaller of |b| / floor and log p / log ALPHA, the larger over the two directions.

## Found while measuring: TF32 leans toward zero

On the RTX 5070 Ti with Triton 3.7.1, a Truestar matmul (256 x 1024 x 256) leans as follows
(`up`/`outward` mean in tolerance units):

| precision | random signs, outward | positive inputs | p |
|---|---|---|---|
| strict float32 | +3.7e-10 | -6.8e-10 | 0.1 to 0.5 (no lean) |
| TF32 (`allow_tf32`) | -2.8e-4 | **-6.2e-4** | 0 |
| float16 tensor cores, float32 accumulate | -4.1e-7 | -5.3e-6 | 0 |
| bfloat16 tensor cores | -3.2e-7 | -1.9e-6 | 0 |

The TF32 lean on positive data, -6.2e-4, is what dropping the low 13 bits of both operands predicts
(2 * 0.72 * 2^-11 = 7.0e-4): TF32 operands are truncated, not rounded, on this GPU (inferred from the
measurement, not from a vendor document). It is 8x the effect floor for TF32, so a TF32 kernel would be
refused. The lean comes from the precision the caller chose, not from the code generator, so a kernel
compiled with `allow_tf32` that leans gets **TS966**: one `KernelBiasWarning` per kernel, and the
kernel is used. Half precision tensor core accumulation leans too (significant, p 0) but stays 15x
under its floor. cuDNN's TF32 convolutions in `truestar check` lean toward zero above the effect
floor in some 1x1 convolutions (mean -1.6e-4 against 7.8e-5) without reaching significance per channel
(p 0.002 over 32 groups). Use strict float32 where a lean matters, such as training.

## PyTorch MPS on a Mac mini M2 (2026-11-25)

`mps_lean.py` (<research>/mac_real/scripts/) runs the module lean test on 15 PyTorch
ops on the M2's GPU in float32, float16 and bfloat16 (matmul K 1024 and 4096, linear, row sums and
means of 4096, softmax, layer norm, an RMS norm, gelu, silu, exp, tanh, causal SDPA, a 3 x 3
convolution, cumsum), 4096 to 1.5 million elements each, against float64 of the same rounded inputs,
effect floor the kernel tolerance / 64. **None of the 45 fires.** The elementwise functions do lean
significantly (p = 0 for exp, gelu, silu and tanh in float32; exp and gelu up, silu and tanh
outward), but under the floor: the largest significant effect is 0.26 of it (gelu in bfloat16, mean
+6.6e-5 tolerance units). The largest effect of any is 0.46 of the floor (the RMS norm in bfloat16,
outward, p 3e-5, not significant at ALPHA 1e-6). Matmuls, sums, convolution and cumsum show no
significant lean; float32 softmax, layer norm, the RMS norm and SDPA lean significantly (p 1e-15 to
1e-100) by 0.007 to 0.03 of the floor. Raw: <research>/mac_real/mps_lean.txt.

## Not verified

Gradient kernels on the GPU beyond those in the files above, AMD GPUs, Apple GPUs beyond the MPS op
survey above, Triton versions
other than 3.7.1, GPUs other than the RTX 5070 Ti, and the region and end to end checks of the
compiled executor (library steps such as cuBLAS calls are checked for size only).
