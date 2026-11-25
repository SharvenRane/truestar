# Multi head latent attention (DeepSeek V3 MLA)

Author: Sharven Rane. First written 2026-11-24.

Code in `src/truestar/mla/` (model and steps, Hugging Face lowering, the absorption proof, import
and compile, the fused latent attention kernel `latent.py`), `torch.empty` in `src/truestar/ir/importer.py`, the sparse selection refusal in
`src/truestar/ir/refusals.py`; tests in `tests/test_mla.py`; benchmark `scripts/bench_mla_decode.py`.

## What it is

DeepSeek V2 and V3 compress keys and values into one low rank latent per token: `kv_a_proj` gives
a latent c (`kv_lora_rank` numbers, RMS normed) and one RoPE key shared by every head
(`qk_rope_head_dim` numbers, the decoupled RoPE); `kv_b_proj` expands c into each head's key part
and value. The cache can hold c and the RoPE key instead of per head K and V.

* **Latent cache.** `cc [L, B, S, r]` and `rc [L, B, S, dr]`, fixed capacity S, written at int64
  positions `pos [T]` with `index_copy`; keys past a query's position are masked, so unwritten slots
  never count. One step program runs the prompt (T tokens) and every decode step (T = 1).
* **Expanded attention** (`mode="expand"`): `kv_b_proj` applied to the whole cache, then ordinary
  attention per head. This is Hugging Face's computation.
* **Absorbed attention** (`mode="absorb"`): with `kv_b_proj`'s weight viewed as `[H, dn + dv, r]`,
  `W_uk` (key part) moves into the query, `q_lat = q_nope W_uk`, and the attention runs on the cache
  itself as one KV head shared by all query heads (keys `[c, k_rope]`, values c, the original
  softmax scale); `W_uv` (value part) is applied to each head's output. Same function in real
  arithmetic, different rounding.
* **Fused latent attention** (`mode="fused"`, the decode default since 2026-11-25): the absorbed
  attention of each layer as one op, `torch.ops.truestar.mla_attention` (IR op `mla_attention`),
  section "Fused latent attention" below.
* **Interleaved RoPE** (`rope_interleave`, DeepSeek's layout) is folded into the weights as a
  permutation of the RoPE rows of `q_b_proj` and `kv_a_proj_with_mqa`, after which the half split
  rotation gives the same products.

## Use

```python
from truestar.mla import from_hf, compile_steps

model = from_hf(hf)                            # a DeepseekV3ForCausalLM with dense layers
steps = compile_steps(model, "cuda")           # prefill "expand", decode "fused"
tokens, logits = steps.generate(prompt, 16, capacity=64, model_cfg=model.c, device="cuda")
graphed = steps.decode.graphed(update={2: 1, 3: 2})   # optional: one CUDA graph per signature, cache in place
```

`compile_steps` first runs `prove_absorption` for each absorbed mode it builds (`absorb` or `fused`,
`prove_absorption(..., mode=)`), then imports each step
through `torch.export` (symbolic batch, token count and capacity) and compiles it with
`compile_program`, whose first call per shape signature is checked against the float64 reference
interpreter. `import_step(model, mode)` gives the IR program alone; `make_model(MLAConfig(...))`
builds a model with random weights without Hugging Face.

## The absorption proof (TS1201)

`prove_absorption(model, prompt, steps=4)` decodes greedily in float64 with the expanded attention
and runs the absorbed attention on the same inputs at every step. It compares, layer by layer, the
latent cache the step returns and the attention output, then the logits, each against `1e-9` of
the point's largest magnitude. A correct absorption measures about `1e-15` (4.1e-15 at the benchmark
config, 2.0e-15 at the test config). Any failing point raises `MLAAbsorptionError` (TS1201) naming
the step and the layer, and nothing is compiled. The tests break it on purpose two ways (the heads
of `W_uk` rolled by one, the heads of `W_uv` rolled by one); both fail at
`model.layers.0 attention output` of step 0.

## Fused latent attention (decode="fused", TS1460)

Profiling the absorbed decode on an L4 (`scripts/profile_decodeperf.py`, the benchmark config below)
showed where the compiled step lost to eager PyTorch in a CUDA graph: per layer the absorbed step
launches the latent query matmul, two concats (the query parts and the whole cache `[c, k_rope]`),
the attention over the concatenated cache (35.6 us per layer at batch 8, the slowest kernel after the
LM head), the `W_uv` matmul, a transposing copy and an out of place `index_copy` of the layer's
cache; then two stacks copy the whole cache again. With the automatic CUDA graph the step was GPU
bound (0.83 ms GPU per step at batch 1 against 0.65 ms for eager PyTorch's kernels).

`mla_attention(q_nope, q_rope, cc, rc, lat, kr, pos, W_uk, W_uv, scale)` does the layer's attention
in one Triton kernel per (sequence, head, query row) (`mla/latent.py`): `q_lat = q_nope W_uk[h]` in
registers, scores `(q_lat . c_p + q_rope . r_p) * scale` read straight from the latent and RoPE key
caches for the cached positions `p < pos[0]`, then this step's own new latents `j <= t`, an online
softmax in a fixed key order with float32 accumulation, and `o = (sum w_p c_p) W_uv[h]^T`. Nothing is
concatenated or copied, and the step writes the new latents of every layer into the caches once, at
the end (two `index_copy` per step instead of two per layer plus two stacks). For consecutive
positions (what `generate` passes) the visible keys are exactly the absorbed decode's
`p <= pos[t]` after its `index_copy`; `prove_absorption(mode="fused")` checks the fused step against
the expanded one in float64 (3.0e-15 at the test config, 4.1e-15 at the benchmark config), and the
compiled op is region checked against its float64 reference on its first call like every library
step. The CPU runs the same math with torch ops. No atomics, no split over keys: the bits depend on
the inputs only (two compiles bitwise equal, tests/test_mla.py). Shapes the kernel does not run
(latent rank above 1024, a head dim above 256, mismatched operands) raise TS1460 naming them;
`decode="absorb"` is the alternative.

## Hugging Face DeepSeek V3

Checked on transformers 5.24.1 (Linux, Python 3.12, on the GPU runs) and 5.20.2 (Windows, Python
3.10). The attention code of `DeepseekV3Attention` is the same in both apart from the cache keyword
being renamed.

* **Import through torch.export.** `DeepseekV3Attention.expand_kv` builds the keys with
  `new_empty` and two slice copies. `torch.empty` (`aten.empty.memory_format`, `aten.new_empty`,
  `aten.empty_like`) now imports as a tensor of NaN, what PyTorch writes into uninitialized memory
  under `torch.use_deterministic_algorithms(True)`: code that writes every entry before reading it
  imports exactly, and a read of an unwritten entry gives NaN in the result rather than a silent
  zero. An integer or bool `torch.empty` is refused (TS1203). With this, a dense `DeepseekV3ForCausalLM`
  imports and matches PyTorch float64 within 1e-13 (2.1e-16 measured) once the model's own float32
  casts are taken out (`hf_float64`: Hugging Face's RMSNorm and rotary tables round to float32 even
  in a float64 model, and its eager attention takes the softmax in float32; with them the imported
  program and eager PyTorch differ at 5e-8, the size of a float32 rounding).
* **`from_hf`** copies the weights into an `MLAModel`. Against `hf_float64(hf)` on two layers, 4
  heads, ranks 24 and 16: interleaved RoPE, half split RoPE, no query rank (`q_proj`), attention
  bias and YaRN scaling all match within 1e-13 in both modes (1.2e-15 to 4.3e-15 measured), and
  8 greedy tokens equal the Hugging Face model recomputing the whole sequence each step.
* **Refused, by stable code, before anything is built:**
  * mixture of experts layers: TS940, as before (use `first_k_dense_replace` at least the layer
    count for dense layers);
  * token sparse attention selection: **TS1202**, an indexer that keeps the top k keys per query
    (`DeepseekV32Indexer`, `DeepseekV4Indexer` and the indexers of GLM MoE DSA, GLM 5 Next,
    Mistral Large 4, HY V4, HY V4 VL, AXK2, Dots3 Note, MiniMax M3 VL and Qwen4 Exp in transformers
    5.24.1, each checked to call `topk`). It is checked before MoE, so DeepSeek V3.2 gets TS1202
    whether or not its MoE layers are kept. Not built: which keys a query reads is a discrete choice
    that a last bit difference can flip;
  * other MLA variants: **TS1200**, naming the layer and the reason: a class whose attention code
    differs from `DeepseekV3Attention` (DeepSeek V2's complex RoPE, Kimi, LongCat, MiniCPM3,
    Mistral 4 and the others in 5.24.1), a `kv_b_proj` bias, fewer KV heads than query heads, a
    sequence length dependent RoPE type (dynamic, longrope), an activation other than SiLU.
    `Glm4MoeLiteAttention` and `YoutuAttention` have the same `forward` source as
    `DeepseekV3Attention` in 5.24.1 but were not run, so they are refused too.

## Results (2026-11-24)

(Before the automatic CUDA graph of compiled calls and the fused decode; the current numbers are in
the next section.)

Imported steps against the same `MLAModel` run eagerly in float64 (test config: 2 layers, 4 heads,
r = 16, dr = 4, dn = 8, dv = 6, vocab 96), prompt 5 then 7 decode steps, batch 2 and batch 3 from the
same import: tokens equal, logits within 1e-13 (1.4e-15 expanded, 1.8e-15 absorbed). Compiled in
float32 (CPU and an L4): tokens equal the float64 tokens, worst logit error 3.8e-7 relative (CPU),
0 fallbacks.

Benchmark, `scripts/bench_mla_decode.py` on one Modal NVIDIA L4 (torch 2.11.0+cu130, transformers
5.24.1, float32, TF32 off everywhere): a DeepSeek V3 config with vocab 8192, width 512, 4 dense
layers, 8 heads, q rank 192, kv rank 128, dn 32, dr 16, dv 32, MLP 1408, interleaved RoPE, random
weights scaled 3x; prompt 64, 64 new tokens, capacity 192. Milliseconds per generated token over
the decode loop (prefill subtracted), median of 5 interleaved rounds after a warm up, one run:

| path | batch 1 | batch 8 |
|---|---|---|
| Hugging Face `DeepseekV3ForCausalLM`, eager sdpa, its dynamic cache | 7.28 | 7.61 |
| `MLAModel` step, eager PyTorch, expanded | 6.18 | 6.47 |
| `MLAModel` step, eager PyTorch, absorbed | 7.11 | 7.61 |
| the eager absorbed step captured in one `torch.cuda.CUDAGraph` | 0.62 | 1.00 |
| Truestar compiled, expanded | 14.03 | 13.75 |
| Truestar compiled, absorbed | 15.15 | 14.75 |
| Truestar compiled, absorbed, `graphed` | 0.86 | 1.21 |
| Truestar compiled, expanded, `graphed` | 0.90 | 1.36 |

Every path gave the same 64 tokens as the float64 model at both batch sizes. Reading it:

* At this size every step is host bound. A profile at batch 1 (torch.profiler, 16 steps) puts the
  GPU time of one decode step at 0.60 ms for the compiled absorbed step and 0.64 ms for the eager
  one; the compiled step launches 171 kernels against 291. The compiled executor's Python cost per
  launch is what makes it about 2x slower than eager PyTorch without graphs.
* With CUDA graphs on both sides the compiled step is 1.2x (batch 8) to 1.4x (batch 1) slower than
  PyTorch's. Part of the gap is the graphed call's index check read (one host wait per call) and
  the copy of the outputs; it was not broken down further.
* Absorption does not pay at this size: the cache is short (192) and the expansion small
  (`H (dn + dv)` = 512 per token against r = 128). Its point is the cache traffic of long contexts,
  which was not measured.
* A second container in an earlier run measured the non graphed paths 1.5x faster (4.2 to 5.0 ms
  eager, 9.1 to 10.3 ms compiled) with the same ratios: absolute numbers depend on the host CPU.

Cache size at capacity 192: latent 442,368 bytes at batch 1 (3,538,944 at batch 8), a per head K
and V cache 1,966,080 (15,728,640): 4.44x smaller, `H (dn + dr + dv) / (r + dr)` = 640 / 144. At
DeepSeek V3's own sizes (128 heads, r 512, dr 64, dn 128, dv 128) the same formula gives 71x.
Hugging Face 5.x caches the latent too (294,912 bytes after 128 tokens at batch 1, the same 144
numbers per token and layer).

Determinism (`scripts/bench_mla_decode.py --determinism`): the absorbed decode compiled twice per
process, batch 8, prompt 64, 64 tokens, sha256 of the tokens and logits, weights and prompt hashed
too (equal in every run). Two separate L4 containers:

| compile mode | container 1 | container 2 |
|---|---|---|
| `deterministic=True` | `a6d8ec0dcef7421f` twice | `a6d8ec0dcef7421f` twice |
| default (`deterministic=None`, follows `torch.are_deterministic_algorithms_enabled()`) | `75a244bf66c5d161` twice | `75a244bf66c5d161` twice |

Two compiles in one process were bitwise equal in every container (5 of 5 in the default mode). But
an earlier benchmark container gave `ec2babf2ae4797ed` in the default mode, so across processes the
default mode is not always bitwise equal; the default may choose timed library calls and split
contractions (docs/COMPILER.md). Use `deterministic=True` for bits that repeat across runs; it was
equal in 2 of 2 containers. The cause of the default mode's different run was not isolated.
`tests/test_mla.py` asserts equality of two compiles in one process on CUDA.

## Results (2026-11-25): automatic graphs and the fused decode

Same benchmark, config and GPU type (`scripts/bench_mla_decode.py 5`, one Modal NVIDIA L4, torch
2.11.0, transformers 5.24.1, float32, TF32 off), with main's automatic CUDA graph of ordinary compiled
calls (docs/COMPILER.md, Automatic graphs) and the fused decode. Milliseconds per generated token,
median of 5 interleaved rounds, one run:

| path | batch 1 | batch 8 |
|---|---|---|
| Hugging Face `DeepseekV3ForCausalLM`, eager sdpa | 7.22 | 7.45 |
| `MLAModel` step, eager PyTorch, expanded | 6.11 | 6.31 |
| `MLAModel` step, eager PyTorch, absorbed | 7.10 | 7.46 |
| Truestar compiled, expanded (ordinary call, automatic graph) | 1.01 | 1.40 |
| Truestar compiled, absorbed (ordinary call, automatic graph) | 0.90 | 1.24 |
| **Truestar compiled, fused** (ordinary call, automatic graph) | **0.76** | **0.97** |
| the eager absorbed step captured in one `torch.cuda.CUDAGraph` | 0.62 | 0.99 |
| Truestar compiled, absorbed, `graphed(update=...)` | 0.81 | 1.16 |
| Truestar compiled, expanded, `graphed(update=...)` | 0.90 | 1.31 |
| **Truestar compiled, fused, `graphed(update=...)`** | **0.68** | **0.85** |

Every path gave the same 64 tokens as the float64 model at both batch sizes; 0 fallbacks in every
program; two compiles of the absorbed decode bitwise equal (`tests/test_mla.py` checks the fused one).

* Like for like without graphs on the eager side, the compiled decode is now 9.3x (batch 1) and 7.7x
  (batch 8) faster than the eager absorbed step: the automatic graph removed the host cost that made
  it 2x slower on 2026-11-24 (the ordinary call without graphs is still 12 to 15 ms, host bound).
* Like for like with graphs on both sides (cache updated in place), the fused decode is 1.16x faster
  than eager PyTorch in a CUDA graph at batch 8 and 1.09x slower at batch 1 (the absorbed one was
  1.17x and 1.31x slower). Against the absorbed decode it is 1.17x (batch 1) to 1.28x (batch 8) faster
  as an ordinary call and 1.19x to 1.36x graphed.
* GPU time per step (`truestar profile`, automatic graphs off): 0.825 ms absorbed and 0.656 ms fused at
  batch 1, 1.217 and 0.819 ms at batch 8; 235 steps per call against 162. Eager PyTorch's kernels take
  0.647 ms (batch 1) and 1.006 ms (batch 8).
* What remains at batch 1: the generated kernels (82 in the absorbed step, mostly GEMV shaped
  projections with their fused elementwise ops) take 0.69 ms of GPU time, eager's whole step 0.65 ms;
  then the RoPE `cat`s of `rotate_half` and the casts. Not attempted here.

## Limits

* Dense layers only: a DeepSeek V3 with MoE layers is refused (TS940). The checks ran at tiny and
  small random configs; no pretrained checkpoint was loaded.
* The cache is the fixed capacity layout (one position vector for the batch); a paged latent cache
  and ragged batches were not built.
* The fused GPU attention shape takes head dims up to 256. DeepSeek V3's absorbed query is 576 wide
  (r + dr), so at real sizes `decode="absorb"` runs its attention as three kernels (scores, softmax,
  product); `decode="fused"` runs it as one (r up to 1024, tested at r 512, dn 128, dr 64, dv 128).
* The fused kernel runs one program per (sequence, head, query row) over all cached keys, with no
  split over keys: at small batch and few heads with very long caches it has few programs. It was
  measured at capacity 192 only.
* Prefill in absorbed mode works but is slower than expanded; `compile_steps` defaults to expanded
  prefill and fused decode.
* Speed is one config, one GPU, one run; see Results for what it does and does not show.
