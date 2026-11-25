# Rotations proven before quantizing

Author: Sharven Rane. First written 2026-11-24. Code: `src/truestar/rotate/` (`hadamard.py`,
`fold.py`, `walsh_kernel.py`, `learned.py`, `qerror.py`); tests in `tests/test_rotate.py` and
`tests/test_kvquant_low.py`; study in `scripts/rotation_study.py` (weights and activations) and
`scripts/kvquant_low_study.py` (KV cache, docs/KV_QUANT.md). Pain row P51 (docs/QUALITY_BAR.md);
research items 2.2 and 2.9 of the November 2026 research round. Codes TS1020 to TS1024.

## The problem

Rotating a model before quantizing it (QuaRot, SpinQuant, FlatQuant, DartQuant) spreads outlier
channels over all channels, so 4 bit weights and activations lose less. In exact arithmetic the
rotated model is the same model. In practice the rotation is folded into weights by hand, and two
mistakes are easy and silent: a per channel norm scale left in place (RMSNorm(x Q) is RMSNorm(x) Q
only when the norm has no per channel weight), and a rotation that crosses a nonlinearity or RoPE.
Either one changes the model, and the quantized accuracy table that follows cannot tell a folding
mistake from quantization loss. Random rotations also vary: SpinQuant reports up to 13 points
between random draws.

## The redesign

1. **Deterministic matrices.** `hadamard(n)` builds a +1 / -1 Hadamard matrix for n = 2^k
   (Sylvester) and n = (q + 1) 2^k for a prime q = 3 (mod 4) (Paley's first construction, then a
   Kronecker product); H H^T = n I is checked exactly on construction. That covers every size of
   the models here (960 = 60 x 16, 2560 = 20 x 128, 576 = 72 x 8, 896 = 224 x 4). Other sizes
   are refused (TS1022); nothing falls back to a random rotation. `walsh(x)` applies it in
   O(n log n) on the power of two factor; on float16 rows of a power of two length it is exact
   in float64.
2. **The fold** (`fold.apply`, Hugging Face Llama and Qwen2 layouts): norm weights into the
   linears that read them; R1 on the residual stream (embedding, q, k, v, gate, up, o, down,
   lm head; a tied lm head is untied); R2 per attention head (value rows, o_proj input columns);
   R4 online at the down_proj input (`OnlineRotate`, folded H into down). Folding happens in
   float64. Learned and random matrices take the same path (TS1021 when not orthogonal to 1e-12).
3. **The proof** (`fold.prove`): the original and the rotated model run in float64 on the same
   tokens, and 10 points per layer plus the logits are compared once the rotation is undone (the
   layer input, q, k, v, the attention output, o, gate, up, the down input and output). Every
   point must be within 1e-9 of the original relative to its largest magnitude. A correct fold
   measures 1e-15 to 1e-13 (float64 rounding), a wrong one 1e-2 to 1 (a different function), so
   the gap is 7 orders of magnitude either side of the bound. A failure raises
   `RotationProofError` (TS1020) naming the first point that differs and the steps between it and
   the last point that held. Only a proven model goes to the quantizer.
4. **Then the existing two verdict quant flow** (docs/QUANT.md): the format loss (verdict b) of
   each format on the original and on each rotated model, all compared with the original float64
   logits, measures what the rotation buys. The kernels' own verdict (a) is unchanged.
5. **Online rotations get a verified kernel** (`walsh_kernel.fwht`): a Triton fast Walsh
   transform (log2 D float32 butterflies per row, rows independent, so batch invariant), checked
   on its first call per shape and dtype against the float64 transform (TS1023). It rotates the
   decode query and output of a rotated KV cache (docs/KV_QUANT.md).

## Use

```python
from truestar.rotate import fold, learned, qerror

rot = fold.apply(model, fold.Plan())            # norm fold + Hadamard R1, R2, online R4, float64
rep = fold.prove(model, rot, ids)               # raises TS1020 with the first wrong point
q = learned.learn_r1(model, calib, steps=100)   # Cayley parametrized R1 started at the Hadamard
rot2 = fold.apply(model, fold.Plan(r1=q))       # proven the same way
res = qerror.evaluate(fold.to_float64(rot.model), ids, ["w4a4_int_rtn", "w8a8_int_rtn"], ref_logits=ref)
qerror.evaluate(fold.to_float64(rot.model), ids, ["nvfp4"])   # TS1024: refused, see below
```

What a wrong placement reports (SmolLM2-360M, 128 WikiText tokens, `scripts/rotation_study.py`):

| mistake | first point that differs | relative difference | named steps |
|---|---|---|---|
| norm weight not folded before R1 | `model.layers.0.self_attn.q_proj` | 0.97 | `model.layers.0.input_layernorm (norm fold into q, k, v)`, R1 into q_proj |
| head rotation folded into W_q and W_k (crosses RoPE) | `model.layers.0.self_attn` (attention output) | 1.52 | RoPE, scores, softmax; head rotation of q and k folded before RoPE |
| rotation of gate and up outputs folded into down (crosses SiLU) | `model.layers.0.mlp.down_proj input` | 1.00 | `act_fn(gate) * up` |

In the RoPE case q and k themselves still agree once the head rotation is undone; the proof
blames the attention core, where RoPE is, not the fold.

## Results: SmolLM2-360M (scripts/rotation_study.py, results/rotation/)

Model: SmolLM2-360M from the local cache (Llama layout, hidden 960, 32 layers, 15 query and 5 KV
heads of 64, SwiGLU 2560, tied embeddings; loaded in its bfloat16 checkpoint dtype, held exactly
in float64). Evaluation: 2 x 384 WikiText-2 test tokens (766 scored positions), all float64 on the
CPU; every row is compared with the original model's float64 logits. Format loss only (verdict b,
no kernel). The learned R1 trained 60 steps (327 s, CPU, 8 threads) on 256 WikiText-2 train
tokens, its objective falling from 14.5 to 9.66.

**Proof.** The Hadamard plan (norm fold, R1 960 = Paley 60 x 16, R2 64, online R4 2560 = Paley
20 x 128) equals the original at all 321 points with a worst relative difference of 4.6e-14
(bound 1e-9) in 7 s; orthogonality of the matrices 2.3e-14 (R1), 0 (R2), 4.0e-14 (R4). Every
other row below was proven the same way before it was quantized (worst 4.5e-14 to 6.6e-14).

KL divergence from the float64 model (lower is better; perplexity ratio in brackets). The first
three columns are the QuaRot and SpinQuant setting (integers, one scale per weight row and per
activation token); the last four are truestar.quant formats (docs/QUANT.md).

| R1 (all rows with Hadamard R2 and R4 except the first two) | act max / rms | W4A4 int | W8A8 int | W4A16 int | int4:g64 (W4A16) | nvfp4 (W4A4) | mxfp4 (W4A4) | fp8_e4m3:channel (W8A8) |
|---|---|---|---|---|---|---|---|---|
| none (the original) | 38.6 | 6.97 (1041) | 0.0436 (1.042) | **0.62** (1.82) | **0.251** (1.27) | **0.254** (1.26) | **0.822** (2.08) | **0.0174** (1.022) |
| Hadamard R1 + R2, no R4 | 17.7 | 4.40 (77.2) | 0.0250 (1.038) | 1.40 (3.78) | 0.792 (2.20) | 1.51 (4.27) | 1.96 (7.12) | 0.0578 (1.064) |
| Hadamard | 5.6 | 2.57 (12.3) | 0.0054 (1.011) | 1.48 (4.06) | 0.669 (1.99) | 1.20 (3.26) | 2.16 (9.07) | 0.0572 (1.065) |
| random, seed 0 | 6.0 | 3.33 (29.9) | 0.0050 (1.009) | 1.34 (3.90) | 0.614 (1.90) | 1.12 (2.84) | 1.55 (4.68) | 0.0620 (1.060) |
| random, seed 1 | 5.9 | 3.05 (20.7) | 0.0055 (1.004) | 1.38 (4.09) | 0.636 (1.84) | 0.90 (2.31) | 1.88 (6.32) | 0.0605 (1.077) |
| learned (from the Hadamard) | 5.4 | **2.28** (9.26) | **0.0037** (0.998) | 0.98 (2.48) | 0.612 (1.67) | 1.12 (2.99) | 1.57 (4.47) | 0.0688 (1.055) |

What this shows, on this one model:

* **Where rotation is the published win, it is reproduced.** W4A4 with per token integer
  activations: perplexity 1041 times the float model without rotation, 12.3 with the Hadamard,
  9.26 with the learned R1. W8A8: KL 8x (Hadamard) and 12x (learned) lower. The online R4 matters
  (W4A4 77.2 without it, 12.3 with it). Two random rotations land at 20.7 and 29.9, worse than the
  deterministic Hadamard and 2.2x to 3.2x worse than the learned one: the spread SpinQuant reports.
* **On fine grained block formats and FP8 it is a loss here.** int4 with groups of 64, NVFP4
  (blocks of 16), MXFP4 (blocks of 32) and FP8 with per channel weights all lose 1.9x to 5.9x more
  KL with any rotation, though activation outliers fall from 38.6 to 5.6 times the rms. Measured
  per layer (FP8, 128 tokens): the error rises in 215 of 224 layers, for the weights alone (205
  layers) and for the activations alone (221), most in v_proj and up_proj (0.025 to 0.065). A
  block format already isolates an outlier in its own block of 16 to 64 values, and a float format
  has no outlier problem to begin with; a rotation spreads each outlier's energy into every
  block, where it raises the scale of values that were small, and it makes the weights' error
  dense where the model had placed small weights against large channels. Weight only integer
  quantization per row (W4A16 int) is also worse rotated (0.62 against 0.98 to 1.48).
* **So rotations are offered only where they help.** The pass is opt in, and a rotated model is
  refused (`RotationRefusedError`, TS1024, with these numbers in the message) by every block or
  float format (int4 grouped, nvfp4, mxfp4, mxfp8, fp8 at any granularity) and by weight only
  integers; only the per token integer activation formats (`w4a4_int_rtn`, `w8a8_int_rtn`,
  `qerror.ROTATION_FORMATS`) take it. The table above was measured before the refusal existed. A
  rotated 4 bit KV cache stays (it measurably lowers error, docs/KV_QUANT.md). The learned R1
  is the best rotation in the four integer columns; on NVFP4, MXFP4 and FP8 a random or the
  Hadamard rotation is slightly better than it, and no rotation beats none on those formats.

The learned R1 is trained on a layer local objective (each linear layer's W4A4 output error on
calibration inputs, DartQuant style) rather than SpinQuant's end to end loss, to train a 360M
model in a few GB of CPU memory; with the scale's gradient kept (a straight through rounding with
a detached scale has an exactly zero gradient along the rotation manifold, since an orthogonal Q
leaves x Q (W Q)^T unchanged: found while building, the first version did not learn).

## Not done or not verified

* One real model (SmolLM2-360M; the other local Llama style checkpoint, Qwen2.5-0.5B, is supported
  by the fold, 896 = Paley 224 x 4, but was not run). 766 scored positions: KL and perplexity
  ratios are for comparing rows, not a benchmark score.
* Format loss only: no rotated model ran on the quantized kernels (verdict a), and no online R4 or
  R3 kernel cost was measured inside a model. The fold supports Llama and Qwen2 layouts only (no
  LayerNorm models: their mean subtraction needs a different fold).
* Hadamard sizes are Sylvester and Paley I only (q + 1 for a prime q = 3 mod 4, times 2^k); 172 or
  156 times 2^k, which Quark lists, are refused.
* The proof checks the folded program on the tokens it is given (128 here); the fold is linear, so
  a correct fold holds on every input, but a placement whose error depends on the input (for
  example one that only matters past some position) is only as covered as the tokens are.
* The learned R1 uses round to nearest integers; it was not trained for NVFP4 or MXFP4, and
  SpinQuant's end to end objective was not run.
