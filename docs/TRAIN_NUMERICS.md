# Training numerics: Muon, low precision GEMMs, invariant sentinels, logit bounds

Author: Sharven Rane. First written 2026-11-24. Pain row: P52 (docs/QUALITY_BAR.md). Research items
2.5 to 2.8 of the November 2026 research round. Codes TS1040 to TS1053 (`truestar codes`).

| item | code | tests |
|---|---|---|
| Muon with orthogonality checked against float64 | `src/truestar/optim/muon.py` | `tests/test_muon.py` |
| Low precision training GEMMs with float64 verdicts | `src/truestar/lpgemm/__init__.py` | `tests/test_lpgemm.py` |
| Invariant sentinels | `src/truestar/invariants/__init__.py` | `tests/test_invariants.py` |
| Attention logit bound from weights | `src/truestar/logitbound/__init__.py` | `tests/test_logitbound.py` |

Measurements: `scripts/bench_trainnum.py`, output in `results/trainnum_2026_11_24.json` (torch 2.11,
Windows, RTX 5070 Ti shared with other jobs, CPU runs on the same PC).

## 1. Muon (TS1040, TS1041)

`optim.Muon(lr, weight_decay, momentum=0.95, nesterov=True, ns_coefficients, ns_steps=5, ns_dtype="bf16",
adjust_lr="original", adam_*)`. 2D leaves get Muon; every other leaf gets AdamW (betas 0.9, 0.95),
so one optimizer covers a whole model. It follows the Optimizer protocol of docs/OPTIM.md: one
definition runs eagerly and inside `truestar.train_step` (reference, CPU and CUDA executors).

The step is torch.optim.Muon's (its momentum buffer is an average, `lerp`, not the sum its
docstring shows; the two agree after the normalization). Differences on purpose: the Frobenius norm
is taken in the parameter dtype, not bfloat16, and each GEMM epilogue (`b A + c A A`, `a X + B X`)
is formed in float32 and stored once.

**The verdict.** In exact arithmetic the iteration maps each singular value s of the normalized
input through q(x) = a x + b x^3 + c x^5, ns_steps times, and keeps the singular vectors. One float64
SVD therefore gives the exact iteration result `U q^k(S/|u|) V^T`, and the polar factor `U V^T`.
`orthogonality_report(u, o)` splits the distance of an update `o` from orthogonal into:

* `intended`: exact iteration against the polar factor. That is the coefficients' choice: 3.4445,
  -4.7750, 2.0315 stop at singular values of about 0.68 to 1.13;
* `rounding`: the update against the exact iteration. That is what the dtype and the kernel add.
  It is checked against `tolerance = min(4 x estimate, worst case)`. The estimate carries each
  storage rounding (u/sqrt(3) times its norm) through the RMS of the per component first order
  gains, using the divided differences of q at the actual singular values. The worst case uses
  the largest gain instead.

`opt.check(params, grads, state, step)` reports every 2D leaf (TS1041). `raise_on_fail` raises
`MuonOrthogonalityError` (TS1040). For a compiled step, `opt.recover_update(p_old, p_new, step)` reads
the applied update back from the parameters, and `recover_resolution` adds the parameter's own
rounding to the tolerance.

**Measured** (rounding per unit of an orthogonal factor, `||o - exact|| / sqrt(rank)`):

| shape | torch.optim.Muon bf16 (CPU, CUDA) | Truestar eager bf16 | Truestar eager f16 | Truestar eager f32 | estimate bf16 | intended |
|---|---|---|---|---|---|---|
| 128 x 512 | 0.0113, 0.0113 | 0.0131 | 0.0017 | below 1e-5 | 0.049 | 0.220 |
| 512 x 128 | 0.0102, 0.0101 | 0.0131 | 0.0018 | below 1e-5 | 0.049 | 0.221 |
| 256 x 256 | 0.0102, 0.0102 | 0.0131 | 0.0017 | below 1e-5 | 0.048 | 0.207 |

* In bfloat16 the rounding is about 1/20 of the intended distance. The estimate sits 2 to 4 times
  above the measurement over every shape, dtype and spectrum in the tests (worst measured/estimate:
  0.53 bf16, 0.55 f16, 0.26 f32, 1.24 f64).
* Compiled steps (reference, numba CPU, Triton CUDA) of a small MLP: first step updates were read
  back from the parameters and checked. They pass, with rounding 0.004 to 0.08 per leaf.
* Faults named TS1040 in the tests: 4 or 6 steps run for 5, a wrong sign on b (diverges to inf),
  and in float32 a coefficient typo (2.0315 to 2.0135) and a single entry off by 1e-3.

**Muon against AdamW** (examples/nn_mlp.py model, 3 layers, 22k params, compiled CPU step, float32,
300 steps, warmup cosine, clip 1.0; one seed, so not a tuned comparison):

| optimizer | mean loss last 20 steps | held out loss | held out accuracy | median step |
|---|---|---|---|---|
| AdamW lr 3e-3 | 0.568 | 0.562 | 0.793 | 3.5 ms |
| Muon lr 0.02, bf16 iteration (AdamW 3e-3 on biases) | 0.572 | 0.558 | 0.789 | 13.6 ms |
| Muon lr 0.02, f32 iteration | 0.572 | 0.558 | 0.789 | 8.2 ms |

On this small model the two optimizers tie. Muon's step costs 2.4 to 3.9 times AdamW's on the CPU,
because each matrix runs 15 extra GEMMs. The bf16 iteration is slower than f32 on the CPU, which
emulates bfloat16.

**Limits.** In bf16 the verdict only catches faults above about 0.23 per unit. That is 14 times
the typical rounding, and roughly the size of a missing or extra step. Smaller bf16 faults (a
coefficient typo of 1%, one wrong entry) are hidden in the rounding. The f32 iteration catches
them. Each 2D leaf keeps an unused `v` slot, one parameter sized buffer per matrix. No sharded or
blockwise Muon, and no comparison of a sharded update to the single device one (research 2.8, second
half). No lr transfer study or larger models.

## 2. Low precision GEMM training (TS1042 to TS1045)

`lpgemm.convert(model, recipe)` swaps every `nn.Linear` for an `LPLinear`. Each layer runs its
three GEMMs in the recipe: forward `x W^T`, input gradient `dY W`, weight gradient `dY^T X`. With
`verify=True`, every call records two verdicts:

* **format loss**: the float64 product of the dequantized operands (the exact codes and scales the
  kernel read) against the float64 product of the unquantized operands;
* **kernel error**: the output against that dequantized product, divided by the float32
  accumulation bound of truestar.quant.verify (`u_out |ref| + 16 sqrt(K) 2^-24 sum|a||b|`). Above 1
  is TS1042, a kernel bug that is never counted as format loss. `strict=True` raises at that GEMM.

It also checks one invariant per layer: **the input gradient multiplies by the weight the forward
used**, compared bitwise after dequantization. Per tensor FP8 scales, and NVFP4 with 16 x 16 tiles,
quantize W and W^T to the same values. Per channel FP8 and 1 x 16 NVFP4 blocks do not
(`transpose_consistency`: 130,911 of 131,072 entries differ for a 256 x 512 FP8 per channel weight,
101,066 for NVFP4 1 x 16). Each such layer is reported as TS1043. The `nvfp4_fwd` recipe passes
the forward's block values (exact in bf16) and its tensor scale to the backward, so the bitwise
check holds.

`grad_report(model_f64, model_lp, loss, x, y)` runs one step of each model and gives per layer
weight and input gradient errors against the float64 model. `plan(make_model, recipe, ..., budget)`
keeps layers in bf16 until every layer's weight gradient error is within the budget, measuring
again after each move. This replaces the "last 15% of layers" rule with a measurement (TS1044 names
the layers kept).

Recipes: `fp8` (E4M3 everywhere, activation and gradient rows scaled, weights per tensor, Truestar
Triton FP8 kernel), `fp8_channel` (weights per channel), `fp8_hybrid` (E4M3 forward, E5M2
gradients, per tensor, cuBLASLt through `torch._scaled_mm`: rowwise scales are refused on sm_120
in torch 2.11, and E5M2 x E5M2 is not supported), `nvfp4_fwd` (NVFP4 forward on the Truestar FP4
kernel with native FP4 MMA on sm_120; bf16 backward with the forward's weight), and `bf16`.

**Measured, step 0** (MLP 256-512-512-256-64, batch 256, weight gradient error against float64, same
for every layer within 15%):

| recipe | dW error | dX error | format loss fwd / dgrad / wgrad | worst kernel / bound | W fwd = W bwd |
|---|---|---|---|---|---|
| bf16 | 0.0053 to 0.0058 | 0.002 to 0.005 | 0.0022 / 0.0022 / 0.0023 | 0.99 | yes |
| fp8 | 0.063 to 0.073 | 0.026 to 0.061 | 0.036 / 0.036 / 0.036 | 0.013 | yes |
| fp8_channel | 0.064 to 0.073 | 0.025 to 0.061 | 0.036 / 0.036 / 0.036 | 0.014 | no (TS1043) |
| fp8_hybrid (E5M2 grads) | 0.069 to 0.108 | 0.026 to 0.090 | 0.036 / 0.059 / 0.058 | 0.010 | yes |
| nvfp4_fwd | 0.209 to 0.227 | 0.10 to 0.19 | 0.136 / 0.101 / 0.002 | 0.99 | yes |

bf16 and NVFP4 sit near 1 because their outputs are stored in bf16 and rounding takes up the bound.
FP8 outputs are float32. **Training**, 200 AdamW steps on a teacher labelled task (mean loss over
the last 20 steps, held out loss): f32 2.133 / 2.337, bf16 2.133 / 2.338, fp8 2.141 / 2.343,
fp8_channel 2.138 / 2.329, fp8_hybrid 2.146 / 2.368, nvfp4_fwd 2.218 / 2.422. Injected kernel
bugs (float16 accumulation, a tensor scale read for a channel scale, a shifted NVFP4 block scale)
are TS1042 in the tests, and the forward's format verdict stays unchanged.

**Limits.** This is a verification harness, not a fast trainer. The quantizers are the exact
float64 encoders of truestar.quant, and every verdict costs a float64 GEMM on the CPU. A verified
fp8 step takes 28 ms against 1.5 ms in f32, and turning verification off does not remove the float64
quantizer. No stochastic rounding, no Hadamard transform on weight gradient inputs, no NVFP4
backward GEMMs. MXFP8 is not offered: truestar.quant on main has no E8M0 32 block FP8 format
(TS1045). Only nn.Linear layers are converted (attention matmuls stay as they are). The training
comparison is one seed on one synthetic task.

## 3. Invariant sentinels (TS1047 to TS1049)

`invariants.instrument(program)` finds three kinds of site in an IR program, forward or gradient:
softmax outputs, softmax input gradients `p (dp - delta)` where delta is `sum(dp p')` with any p',
and `x / sum(x)` normalizations. For each site it adds float64 row reductions as extra outputs.
`invariants.Sentinels(inst, runner)` runs the program on the reference interpreter or any compiled
program and checks every row:

| identity | tolerance per row (u_c arithmetic unit, float32 for 16 and 8 bit storage; u_s storage unit; n row length) |
|---|---|
| softmax rows sum to 1 | `2 ((n + 3) u_c + 2 u_s)` |
| softmax gradient rows sum to `delta (1 - sum p)` | `2 (((n + 1) u_c + u_s) sum abs(dp p) (abs(1 - sum p) + sum p) + (2 u_c + 2 u_s) sum abs(ds))` |
| gate rows sum to 1 | `2 ((n + 3) u_c + 2 u_s) sum abs(x) / abs(sum x)` |

A compiled program that sums the checks in float32 adds `n u_out` terms. The same checks run on
tensors through `prob_rows`, `softmax_grad_rows(ds, p, dp, delta=None)` and `normalized_rows`, for
example on a fused kernel's outputs.

**Measured.** A causal attention backward has its delta read from probabilities rounded to bf16,
f16 or E4M3 while the gradient multiplies by the unrounded ones (the stale delta of the FP8 attention
paper). That is TS1048 at 200 to 26,000 times the tolerance, on 56 of 64 rows (the other 8 have a
single unmasked entry). The clean program passes on the reference interpreter (residuals 1e-10 of
tolerance) and on the compiled CPU executor (0.02 of tolerance). torch's CUDA softmax backward in
float16 and bfloat16, on 2048 long rows, passes. FP8 probabilities used consistently pass; a delta
taken from the float32 ones fails. The instrumented program returns its original outputs bitwise.

**Limits.** These checks are added outputs, not the in kernel flag words of the TS500 sentinels.
A compiled instrumented program needs `verify="off"`, because the end to end relative check cannot
compare cancelling sums. Region checks still run. Truestar's own attention lowers to softmax, so its
backward is consistent by construction. A fused third party attention backward (FlashAttention)
does not expose `ds`, and is checked only when the caller passes its tensors. The pattern matcher
expects the IR autodiff's form of the softmax gradient. No overhead measurement on large programs.

## 4. Attention logit bound from the weights (TS1051 to TS1053)

`logitbound.check_module(model, target)` reads torch `TransformerEncoderLayer` and
`TransformerDecoderLayer` (with the norm in front when `norm_first`), and Hugging Face style decoder
layers (`self_attn.q_proj`, `k_proj`, `input_layernorm`, with `q_norm`/`k_norm` and logit softcap
when present). Per head:

* **upper**: `scale (sigma_max(W_q^T W_k) R^2 + (|W_k^T b_q| + |W_q^T b_k|) R + |b_q . b_k|)`, with
  sigma_max from a float64 SVD, and R the largest input 2 norm: `max|gamma| sqrt(d) + |beta|` after
  a LayerNorm, `max|gamma| sqrt(d)` after an RMSNorm, given by the caller otherwise. With RoPE it
  uses the product form `scale (sigma(W_q) R + |b_q|)(sigma(W_k) R + |b_k|)`, valid at every offset.
  With QK norm it is `scale max|gamma_q| max|gamma_k| d_h`, and a softcap is a cap;
* **reachable**: a query and key pair is built from the top singular pair and refined by exact best
  responses under the norm's constraint (mean 0, norm sqrt(d)), then run through the real modules.
  So reachable <= true max <= upper.

Verdicts per target (fp16, bf16, fp8_e4m3, fp8_e5m2): `safe` (upper inside the range, TS1051),
`may_overflow` (upper outside, reachable inside, TS1052 warning), `overflows` (reachable outside,
TS1053). Entries of q and k are checked too, for formats that store them unscaled. `guard` refuses
TS1053 and warns TS1052. `steps_to_limit(history)` projects the growth of a bound to the format's
limit.

**Measured.** 4 layer encoder, d 256, 8 heads: 32 heads in 0.72 s, gap upper/reachable 1.00001 to
1.009 (the true maximum is pinned within 1%). Tiny Llama (RoPE, GQA): gap 1.21 to 1.37 (reachable at
offset 0 only). Measured logits of random inputs never exceed the bound. With weights scaled until
TS1053 fires, the constructed pair's logit is above 65504 and becomes inf in float16.

**Limits.** Without a norm in front (post norm first layer) R must be given, and there is no
reachable value. RoPE bounds use the product form and do not search offsets. MLA, ALiBi and
sliding window specifics are not read. No integration into `truestar spikes` yet; `steps_to_limit`
is the building block.
