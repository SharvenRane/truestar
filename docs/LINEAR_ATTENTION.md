# Linear attention and state space kernels

Author: Sharven Rane. First written 2026-11-24. Code: `src/truestar/linattn/`. Pain ledger row P53.
Tests: `tests/test_linattn.py`. Numbers: `scripts/bench_linattn.py`, `scripts/linattn_fingerprint.py`.

Mamba 2, Gated DeltaNet (Qwen3 Next, Qwen3.5), Kimi Delta Attention (Kimi Linear) and lightning
attention replace softmax attention with a recurrence over a matrix state. Libraries run it in
chunks: inside a chunk everything is a matrix product and only the state crosses chunk borders.
The chunked form is exact in real arithmetic for any chunk size, but libraries pick 64 or 16 and get
different bits, and nobody reports how the rounding grows along the chain of chunks. Truestar runs
the chunked form as its own kernels, with the chunk size as an explicit reduction order parameter,
checks every new call signature against a float64 sequential recurrence, and measures both.

## 1. The recurrence

Per batch, head and step t: query and key of width dk, value of width dv, state S of shape [dk, dv].

| part | definition |
|---|---|
| decay | `D_t = diag(exp(g_t))`, `g_t <= 0`; `g` is one value per head (`scalar`), one per key channel (`channel`) or absent (`none`) |
| rule `add` | `S_t = D_t S_{t-1} + k_t v_t^T` |
| rule `delta` | `S_t = (I - beta_t k_t k_t^T) D_t S_{t-1} + beta_t k_t v_t^T` |
| output | `o_t = scale * S_t^T q_t` |

| layer | rule | decay | mapping |
|---|---|---|---|
| Mamba 2 SSD (Mamba 2, Falcon H1, Nemotron H, Bamba, Zamba 2) | add | scalar | q = C, k = B, v = dt x, g = dt A (the D skip and gate stay outside) |
| Gated DeltaNet (Qwen3 Next, Qwen3.5) | delta | scalar | g = the per head log decay, beta = the write strength |
| Kimi Delta Attention (Kimi Linear) | delta | channel | g per key channel |
| gated linear attention | add | channel | |
| lightning attention, retention | add | scalar | a fixed per head decay |
| DeltaNet, plain linear attention | delta or add | none | |

Shapes follow the models' code: q, k `[B, T, H, dk]`, v `[B, T, H, dv]`, g `[B, T, H]` or
`[B, T, H, dk]`, beta `[B, T, H]`, initial and final state `[B, H, dk, dv]`.

```python
from truestar.linattn import linear_attention
o = linear_attention(q, k, v, g, beta, rule="delta")                       # Gated DeltaNet
o, state = linear_attention(q, k, v, g, rule="add", output_final_state=True)  # Mamba 2 SSD
```

## 2. Chunk size invariance, the proof

Cut the sequence into chunks of C steps (the last one padded with `k = v = 0`, `g = 0`, which leaves
the state unchanged and adds outputs that are cut off). Inside a chunk let `c_i = g_1 + ... + g_i`
(per head or per channel) and S the state before the chunk.

1. **Decay.** Unrolling `S_i = D_i S_{i-1} + k_i u_i^T` for i steps gives
   `S_i = e^{c_i} S + sum_{j <= i} (e^{c_i - c_j} * k_j) u_j^T`, where `*` scales the key's channels.
   This is the add rule with `u = v`, by induction on i. Every factor `e^{c_i - c_j}` with `j <= i`
   is at most 1 because every g is at most 0, and it is formed only where `j <= i`, so nothing
   overflows (a decay of -30 per step is tested).
2. **Output.** `o_i = scale * ((q_i * e^{c_i})^T S + sum_{j <= i} A_ij u_j)` with
   `A_ij = sum_d q_id k_jd e^{c_id - c_jd}`.
3. **Delta rule.** `S_i = (I - beta_i k_i k_i^T) D_i S_{i-1} + beta_i k_i v_i^T` equals
   `D_i S_{i-1} + k_i u_i^T` with the pseudo value `u_i = beta_i (v_i - (D_i S_{i-1})^T k_i)`. Using
   step 1 for `D_i S_{i-1}` (the sum stops at `j < i`) gives, for all i of the chunk at once,
   `(I + M) U = diag(beta) (V - Kc S)` with `M_ij = beta_i sum_d k_id k_jd e^{c_id - c_jd}` for `j < i`
   (zero otherwise) and `Kc_i = k_i * e^{c_i}`. `I + M` is unit lower triangular, so it is always
   invertible, and `U = Wv - Wk S` with `Wv = (I + M)^{-1} diag(beta) V` and
   `Wk = (I + M)^{-1} diag(beta) Kc`, which depend only on the chunk's own inputs.
4. **Next state.** Step 1 at `i = C`: `S' = e^{c_C} S + sum_j (k_j * e^{c_C - c_j}) u_j^T`.

Each step is an identity in real arithmetic, so one chunk step of length C equals C single steps,
and a chunking of any size is a composition of exact chunk steps: the outputs and the final state
are the same for every C, including C = 1 (the recurrence itself) and C >= T (one chunk). In float
arithmetic the products are summed in another order for each C, which is the only difference.

Measured (tests/test_linattn.py, float64, both rules, all three decay kinds, an initial state, 101
steps): every chunk size in 1, 2, 3, 7, 16, 64, 100, 128 equals the recurrence within 1e-13 of the
largest value; the measured spread was 2.6e-16 to 3.5e-15. Gradients for q, k, v, g, beta and the
initial state agree within 1e-11 (both rules, all decay kinds).

## 3. What finite precision does, measured

RTX 5070 Ti, strict float32 dots, B 2, T 2048, H 4, dk = dv = 128, scalar decay, per element error
(the kernel layer's measure, docs/KERNEL_IR.md section 7) against the float64 recurrence:

| rule, dtype | vs float64, chunk 16 / 32 / 64 | chunk pairs 16-32 / 16-64 / 32-64 | elements that differ between chunk sizes |
|---|---|---|---|
| add, float32 | 1.22e-6 / 1.39e-6 / 2.29e-6 | 1.43e-6 / 2.27e-6 / 1.51e-6 | 71% / 82% / 62% |
| delta, float32 | 1.41e-6 / 1.66e-6 / 2.14e-6 | 1.64e-6 / 2.03e-6 / 1.72e-6 | 74% / 84% / 66% |
| add, bfloat16 in and out | 3.89e-3 for all three | 5.5e-3 | under 1% (the output rounding hides most) |
| delta, bfloat16 in and out | 3.89e-3 for all three | 5.1e-3 to 6.3e-3 | under 1% |

So two libraries that pick chunk 16 and chunk 64 get other bits in most elements, about 2e-6 apart
in float32, which is the size of each one's own error against float64. Smaller chunks were closer
to float64 in every case measured. In float64 the chunked form differs from the sequential scan by
6.4e-16 to 8.0e-16 for chunk 1 to 128.

**Error along the chunk chain** (`chain_error`, T 16384, 1024 chunks of 16, float32, B 1, H 2,
dk = dv = 64): worst error per chunk, the least squares slope over the chunk index and the ratio of
the last quarter to the first quarter.

| rule | decay per step | worst | slope per chunk | last / first quarter | code |
|---|---|---|---|---|---|
| add | none | 1.0e-5 | 6.4e-9 | 7.0 | TS1062 |
| add | up to 0.01 | 1.1e-6 | 1.7e-11 | 1.02 | |
| add | up to 0.3 | 1.1e-6 | 1.9e-14 | 1.01 | |
| delta | none | 1.2e-6 | 1.5e-11 | 1.02 | |
| delta | up to 0.01 | 1.1e-6 | 7.2e-12 | 1.01 | |
| delta | up to 0.3 | 1.2e-6 | 6.2e-12 | 1.01 | |

Plain linear attention without decay is the one case that grows: its state is a sum of T outer
products with nothing forgetting the old rounding, so the error rises steadily along the chain
(7x over 16384 steps). Any decay, or the delta rule (which overwrites rather than adds), keeps the
error flat. `chain_error` reports TS1062 when the last quarter is above 4x the first.

## 4. The kernels

`triton_kernels.py`, two Triton kernels, both written from section 2:

* `_la_prep` (delta rule), one program per chunk and head: `M`, the inverse of `I + M` by forward
  substitution in registers, `Wk` and `Wv` stored as float32. All chunks run in parallel.
* `_la_fwd`, one program per block of value columns and head: walks the chunks in order, keeps its
  `[dk, BV]` slice of the state in float32 registers, and per chunk does five dots. The decays are
  applied as row scales after the dots, so q and k are the only key width operands (this halved the
  shared memory of the dk 128 case and made it compile).

Per key channel decay (gated linear attention, Kimi Delta Attention) has no Triton kernel yet: it
runs in the chunked PyTorch form (`chunked.py`), on CPU or CUDA, verified the same way.
`linear_attention(impl=None)` picks the kernels when they run the call and says what it ran in
`last_call()`; `impl="triton"` refuses the rest with TS1061, never a quiet switch.

**Launch config** (`triton_kernels.config`) depends on the call's dims only: chunk 16, 32 value
columns per program, 4 warps for a key width up to 64 and 8 for 128. The prep kernel always uses 4
warps. Measured on the RTX 5070 Ti over chunk 16, 32, 64, value blocks 16, 32, 64 and 4 or 8 warps
at T 4096 (add and delta, dk 64 and 128, B * H 8 and 32): the pinned config was the fastest of the
18 in 4 of 8 cases and within 1.05x to 1.71x in the others (worst: add, dk 128, B * H 8, 2.37 ms
against 1.39 ms for 16 columns). Chunk 64 with 64 columns was 11x to 21x slower than the fastest
config (the strict float32 dots spill registers). The GPU was shared with other runs during the
sweep, so single entries carry noise.

## 5. Determinism

* No atomics, no split along the sequence: every output element and state entry is computed by one
  program in a fixed order.
* The value block and the forward kernel's warp count do not change the bits (tested over 3 x 2
  configs, both rules). The prep kernel's warp count does (it sets the order of the substitution's
  reductions), so it is fixed in every mode.
* The chunk size changes the bits (section 3). `deterministic=True` (or None, following
  `torch.are_deterministic_algorithms_enabled()`) pins it to 16 and refuses another with TS1063.
* Since nothing reads the SM count, a row's result is independent of the GPU size, the batch it is
  in and the run. Measured 2026-11-24 with `scripts/linattn_fingerprint.py` (4 cases: SSD and
  Gated DeltaNet at T 4096 and 1000, ragged shapes, with and without decay): the outputs and final
  states were bitwise equal on an RTX 5070 Ti (70 SMs) and an RTX 5060 Laptop GPU (26 SMs), same
  Triton 3.7.1 and torch 2.11. A row alone and in a batch of 3, three simulated SM counts and two
  runs are bitwise equal in the tests. Other GPU architectures compile the same source to other
  instructions and are not covered (docs/CROSS_CHIP.md).

## 6. Verification and gradients

* Forward: the first call per shape, dtype, rule, decay kind and launch config runs the float64
  sequential recurrence on the same device and compares the output and the final state per element
  with the reduction floor. Tolerance: the kernel layer's base for the output dtype (float32 for
  float32 and float64 inputs, since the kernels compute in float32), TF32 when allowed, times
  `max(1, log2(T * dk * C) / 4)`. A mismatch raises `LinAttnMismatchError` (TS1060); a test
  perturbs one element of one step by 1e-3 of the largest output and is caught, and one breaks the
  chunked form by 0.1% on the last step and is caught. Measured errors were 4e-7 to 2.1e-6 in
  float32, against tolerances near 1e-4.
* Backward: the autograd gradient of the chunked PyTorch form at the same chunk size (state in
  float32, float64 for float64 inputs). There is no fused backward kernel yet. In float64 these
  gradients equal the recurrence's (section 2). `check_gradients` compares the op's gradients for
  every input with the float64 recurrence's; on CUDA in float32 both rules pass.

## 7. Speed

RTX 5070 Ti, float32 inputs, forward, median ms (triton.testing.do_bench), every Truestar row
checked against float64 first. Sequential is the recurrence one step at a time in float32 on the
GPU (the models' own slow path). Eager chunked is `chunked.py` with chunk 64, written from the math.

| shape (B, T, H, dk, dv) | sequential | eager chunked | Truestar, strict float32 | Truestar, TF32 dots |
|---|---|---|---|---|
| Mamba 2 2.7B layer, SSD (1, 4096, 80, 128, 64) | 437 | 9.97 | 6.13 (71x, 1.6x) | 3.47 |
| Qwen3 Next layer, Gated DeltaNet (1, 4096, 32, 128, 128) | 632 | 12.1 | 5.38 (117x, 2.3x) | 3.17 |
| Gated DeltaNet (4, 2048, 16, 64, 64) | 440 | 7.01 | 0.76 (580x, 9.2x) | 0.66 |
| SSD (4, 2048, 16, 64, 64) | 296 | 7.28 | 0.47 (629x, 15.5x) | 0.38 |

In brackets: times faster than sequential and than eager chunked. The GPU was shared with other
agents' jobs during these runs, which only ever adds time, so each entry is the minimum of three
runs of the script (single runs were up to 15x slower in a row another job hit). The strict float32 rows are the deterministic config; TF32 dots are opt in (`allow_tf32=True`) and
checked against the TF32 tolerance. The single sequence layers with dk 128 are bound by the FMA
dots of strict float32; float16 and bfloat16 tensor core dots are not used yet.

## 7b. Metal (Apple GPUs)

`metal_kernels.py` has the same two kernel split for MPS tensors, written in Metal from section 2:
`la_prep` (delta rule) per chunk and head, and `la_fwd` per block of value columns and head, the
state slice in threadgroup memory, chunk 16, every sum a serial loop in index order by one thread.
`linear_attention(impl=None)` picks them for MPS tensors with no or scalar decay and dk up to 128
(dv up to 256); `impl="metal"` refuses the rest with TS1061. 16 bit inputs are widened to float32
on the device. The gradient is the chunked form's, as for the Triton kernels.

Before 2026-11-25 every MPS call failed: the first call check ran the float64 recurrence on the
inputs' device, which has no float64. The check now copies to the CPU first (in two steps: MPS
`to("cpu", torch.float64)` in one call returns zeros on torch 2.14, docs/METAL.md).

Mac mini M2 (10 core GPU, torch 2.14.1), forward, float32, median ms over 30 calls after warmup,
every Metal row checked against float64 first (errors 3.6e-7 to 1.2e-6):

| rule, decay, (B, T, H, dk = dv) | chunked form on MPS | Metal kernels | chunked form on the M2 CPU |
|---|---|---|---|
| add, none, (1, 1024, 8, 64) | 6.88 | 2.19 (3.1x) | 5.18 |
| add, none, (2, 4096, 8, 64) | 42.4 | 8.73 (4.9x) | 47.5 |
| add, scalar, (1, 1024, 8, 64) | 7.15 | 2.50 (2.9x) | 5.14 |
| add, scalar, (2, 4096, 8, 64) | 42.4 | 12.8 (3.3x) | 47.2 |
| delta, none, (1, 1024, 8, 64) | 30.4 | 2.47 (12.3x) | 10.7 |
| delta, none, (2, 4096, 8, 64) | 268.6 | 16.5 (16.3x) | 86.1 |

The value block width (16 columns, the fastest of 8, 16, 32 and 64 measured) does not change the
bits (tested at 4, 16 and 64); the outputs repeat bit for bit run to run, and a batch entry alone
has the same bits as in a batch of 3 (tests/test_linattn.py). No SIMD group matrices yet.

## 8. Importer hook (refusal by name)

`truestar.linattn.refusal_lifted(kind, class_name)` is True when `coverage.LAYERS` maps the class to
a rule this package verifies and the importer registered a lowering for it. Since P60 the lowerings
of section 10 are registered, so it is True for `Mamba2Mixer`, `FalconH1Mixer`, `BambaMixer`,
`Zamba2MambaMixer` and `Qwen3_5GatedDeltaNet`, and False for every other class.

## 9. Limits (2026-11-24)

* No Triton kernel for per key channel decay; it runs in the chunked PyTorch form (slower, O(C^2 dk)
  memory per chunk).
* No fused backward: gradients come from autograd through the chunked PyTorch form.
* Head dims up to 256; dk 256 was not measured for speed.
* Mamba 3's complex valued state and MIMO recurrence are not covered.
* No importer lowering of the Hugging Face modules yet (section 8): models are still refused.

## 10. Importer lowering of Hugging Face layers (P60, 2026-11-24)

Code: `src/truestar/ir/hf_linattn.py` (the lowerings), `src/truestar/ir/linattn_ops.py` (the IR
ops), `src/truestar/linattn/torch_op.py` (`torch.ops.truestar.linear_attention`). Tests:
tests/test_hf_models.py, the `linear_attention` cases of tests/test_autodiff.py, test_forward.py and
test_vmap.py. Numbers: `scripts/bench_linattn_import.py`, saved in `results/linattn_import/`.

In transformers 5 every Mamba 2 SSD layer calls the module function `mamba2_chunk_scan` and every
Gated DeltaNet layer `torch_chunk_gated_delta_rule`. `import_module` and `export.export` replace that
function during torch.export only (`hf_linattn.lowered`) by one written from the same definition:
the time step softplus and clamp, the group repeat of B and C, the D skip, the head repeat and l2
norm of q and k and the float32 casts are transformers' own torch ops in the same order, and the
recurrence is `torch.ops.truestar.linear_attention` (SSD: rule add, q = C, k = B, v = dt x,
g = dt A; Gated DeltaNet: rule delta, scale 1/sqrt(dk)). The importer turns it into the IR op
`linear_attention`: reference the float64 sequential recurrence, reverse rule
`linear_attention_vjp` and forward rule `linear_attention_jvp` (references: autograd and forward AD
through the float64 recurrence), vmap by the exact per element loop, compiled as a library step
that runs section 4's kernels on CUDA and the chunked form on the CPU, checked on its first call.

Still refused, with the reason in the message: a class with no lowering (Kimi Delta Attention,
lightning attention, mLSTM, the layers of MoE families, which TS940 refuses first), a transformers
without the module function (4.48), a program exported without the lowering, and (TS1150) a call
with an initial recurrent state or a final state request (generation with a cache).

Measured, transformers 5.24.0 (from a local wheel on the ts env), torch 2.11, CPU, test configs of
`testing_hf.LOWERED_2026`:

| family | before | layers lowered | fp64 vs eager PyTorch | grad vs torch.autograd |
|---|---|---|---|---|
| Mamba 2 | refused TS942 | 2 | 6.4e-16 | 5.4e-16 |
| Falcon H1 | refused TS942 | 2 | 4.1e-16 | 1.9e-16 |
| Qwen3.5 | refused TS941 | 1 | 3.1e-16 | 4.7e-16 |
| Bamba | refused TS942 | 1 | 4.3e-16 | 3.4e-16 |
| Zamba 2 | refused TS942 | 2 | 4.6e-16 | 3.3e-16 |

All five compile on the CPU in float32 with no float64 fallback and export a verified artifact.

Speed, forward, float32, batch 4, 256 tokens, the test model, median of 10 calls after the checked
first call; eager is transformers' own PyTorch path (mamba_ssm, causal_conv1d and fla are not
installed, so this is not transformers' fastest mode), TF32 off on both sides. CUDA: RTX 5070 Ti,
0% utilization before and after the run (the GPU is shared, so single numbers carry noise).

| family | CPU eager / Truestar ms | CPU speedup | CUDA eager / Truestar ms | CUDA speedup |
|---|---|---|---|---|
| Mamba 2 | 4.71 / 11.94 | 0.39x | 2.60 / 2.21 | 1.18x |
| Falcon H1 | 409.0 / 165.1 | 2.48x | 17.5 / 25.0 | 0.70x |
| Qwen3.5 | 4.78 / 18.36 | 0.26x | 3.67 / 7.14 | 0.51x |
| Bamba | 4.60 / 18.12 | 0.25x | 2.87 / 6.24 | 0.46x |
| Zamba 2 | 6.89 / 21.18 | 0.33x | 3.61 / 8.03 | 0.45x |

So the lowering closes the coverage gap but is not a speed win at these tiny widths (32 wide
models, the whole program is a few ms of small kernels, and the per call overhead of the compiled
executor dominates); Falcon H1's default layer (128 heads, state 256) is the one large recurrence,
2.5x faster than transformers on the CPU. Larger layers are measured in section 7.

**Remeasured 2026-11-25 after the automatic CUDA graph** (docs/COMPILER.md, Automatic graphs: an
ordinary compiled call is replayed from a verified CUDA graph from its second call):
`scripts/bench_linattn_import.py --device cuda --skip-accuracy`, one Modal NVIDIA L4, torch 2.11.0,
transformers 5.24.1, same shapes (batch 4, 256 tokens, float32, TF32 off), median of 10 calls:

| family | CUDA eager / Truestar ms (L4) | CUDA speedup |
|---|---|---|
| Mamba 2 | 5.41 / 0.389 | 13.9x |
| Falcon H1 | 44.95 / 29.33 | 1.53x |
| Qwen3.5 | 6.75 / 0.530 | 12.7x |
| Bamba | 5.65 / 0.527 | 10.7x |
| Zamba 2 | 7.46 / 0.601 | 12.4x |

The CUDA gap of the table above was the compiled executor's per launch host cost, which the
automatic graph removes; the linear attention kernels were never the slow part. The eager side runs
without graphs (transformers' own PyTorch path, no mamba_ssm, causal_conv1d or fla), so these numbers
compare a graphed program against an ungraphed eager model. Falcon H1 is GPU bound at its default
layer size.

The CPU gap was profiled (`scripts/profile_decodeperf.py --cases linattn --device cpu`, the desktop
CPU, same shapes): Mamba 2 4.64 ms eager against 8.84 ms compiled, Qwen3.5 5.39 against 16.0 ms.
About half of Mamba 2's compiled call is the chunked PyTorch form of the recurrence (2.5 ms per layer:
16 chunks of 16 steps, a Python loop of small matrix products) and the rest is 35 generated CPU
kernels of about 0.1 ms each; Qwen3.5 is dominated by its 51 CPU kernels. A larger CPU chunk would
cut the loop but changes the bits of every CPU linear attention program (the chunk is part of the
summation order, section 4), so it was not changed here; nor were the CPU kernels. Not shipped.

Gaps found (not fixed here):

* Falcon H1 on CUDA at the test shape (batch 3, 6 tokens): the Triton kernel's own float64 check
  refuses (TS1060): state width 256 with per step decays down to -12.8, the float32 state is
  1.2e-4 off float64 against a tolerance of 7.3e-5. In isolation (dk 256, decays to -13) the kernel's
  state error is 3.8e-5 against 1.1e-5 for the chunked form, so the kernel's decay arithmetic loses
  more at strong decay (hypothesis, not proven: the exponent is a difference of two large float32
  running sums). The test expects the loud refusal (`testing_hf.LOWERED_CUDA_GAPS`).
* Two in place writes through one view (Qwen3.5's rotary sections) import exactly now, but the
  compiled CPU program of a small module with that pattern is refused by its first call check; the
  Qwen3.5 model itself passes its checks on CPU and CUDA.
* A `slice_scatter` with a step other than 1 is a kernel only when it writes one entry; the real
  Qwen3.5 rotary sections (11 entries) fall back to the float64 reference for that op.
