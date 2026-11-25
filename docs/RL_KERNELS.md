# One kernel set for the RL trainer and sampler (`truestar.rlkernels`, P15 part b)

Author: Sharven Rane. First written 2026-11-24.

In RL post training the sampler computes logprobs in a prefill and then one token at a time
against a KV cache, in batches with padding; the trainer recomputes them in one teacher forced
forward. With ordinary kernels the three paths reduce in different orders (another GEMM config
for another row count, another attention split for one query than for many, another softmax
chunking for a padded row), so the logprobs differ at rounding level and the importance ratio is
not exactly 1 (docs/PAIN_RESEARCH_2.md, N3; docs/RL_PAIR.md measures it). `truestar.rlkernels`
is one set of Triton kernels used by all three paths, built so the logprobs are bitwise equal.

Code: `src/truestar/rlkernels/` (`kernels.py`, `mode.py`, `tp.py`, `llama.py`); tests in
`tests/test_rlkernels.py` and `tests/test_examples.py`; benchmark `scripts/bench_rlkernels.py`;
example pair `examples/rl_kernels_pair.py`.

## Use

```python
from truestar.rlkernels import rl_mode

with rl_mode():                      # trainer
    logits = model(tokens)
with rl_mode():                      # sampler: prefill, then decode steps against the cache
    out = model(prompt_ids, pos, key_valid, cache)
```

The model is plain PyTorch. Inside `rl_mode()` these calls go to the RL kernels; everything else
runs unchanged:

| PyTorch call | RL kernel |
|---|---|
| `F.linear`, `nn.Linear` | matmul, K split into 4 equal segments added as a binary tree |
| `torch.matmul`, `@`, `mm`, `bmm` | the same matmul (a 2 dim right operand is split like a weight, a batched one is not) |
| `F.scaled_dot_product_attention` (bool mask, `is_causal`, `enable_gqa`, `scale`) | one attention kernel for training forward, prefill and decode |
| `F.rms_norm`, `nn.RMSNorm`, `F.layer_norm`, `nn.LayerNorm` | row kernels |
| `softmax`, `log_softmax` in every spelling | anchored row kernels |
| `sum`, `mean` over one dim or all dims | fixed chunk row sum |

Elementwise ops, indexing, `cat`, embedding lookups and order free reductions (`max`, `argmax`,
`any`, integer sums) need no special kernel: their result per element does not depend on the
batch. Floating point reductions without an invariant kernel (`cumsum`, `var`, `std`, `norm`,
`logsumexp`, `einsum`, `addmm`, convolutions, `cross_entropy`, ...) raise `RLModeError` with the
call named; `rl_mode(strict=False)` runs them eagerly and counts them in `mode.eager_calls`. A
CPU tensor, an additive float attention mask, dropout or a value head dim other than the query's
is an error, never a quiet switch to another kernel.

Each routed call is an autograd function whose backward recomputes the eager PyTorch op on the
saved inputs, so the trainer trains on the RL forward (gradients within 1e-3 relative of eager in
the test; gradients are not claimed to be bitwise invariant).

From the command line, `truestar rlpair` checks a pair on these kernels:

```
truestar rlpair examples/rl_kernels_pair.py --device cuda --kernels rl --strict --tp 2
```

`--kernels rl` runs the trainer, the sampler, the module replays and the logprob computation in
`rl_mode()`. Exit 0 needs bitwise equal logprobs (with `--kernels rl` anything less exits 1) and,
with `--strict`, a sampler whose logprobs do not change with its batch. When the logprobs are not
bitwise equal, the kernels every module ran are recorded on both sides and the first op whose
kernel differs is named (TS185, with `module`, the sampler call and both kernels); if every module
ran the same kernels, TS187 says the difference is outside the kernel set. `--tp N` also compares
the trainer at TP 1 with the pair file's `tp_logits(model, tokens, N)` (TS186).

## How the kernels stay invariant

* **Constant tiles.** Tile shapes, block sizes and warp counts are module constants per dtype
  (matmul 64 x 64 x 32 in float32, 64 x 64 x 64 in bfloat16 and float16; attention 32 query rows
  by 16 keys; rows in chunks of 1024). Nothing is picked from the shapes of a call: no autotuning,
  no split K chosen by size, no cuBLAS heuristics.
* **Fixed order reductions.** Each reduction walks its axis in fixed size chunks accumulated in
  order into a vector of fixed width, summed once at the end. Chunks past the end and masked
  entries add exact zeros.
* **Anchored chunks.** Softmax rows and attention start their chunks at the first unmasked entry
  (the anchor), found with an exact integer min. A row left padded by p has its first real key at
  column p; its chunks hold the same logical keys as the unpadded trainer row, so the online
  softmax rescales at the same points. Trailing masked keys (a static cache's unwritten slots, the
  causal future) are skipped; they would only add exact zeros.
* **Segment tree for weights.** A weight contraction reduces K as 4 equal segments, each from a
  zero accumulator, added as `(s0 + s1) + (s2 + s3)` in registers. This is what makes tensor
  parallelism exact (below).
* **One binary per kernel and dtype.** Two things were found while building this, each with a
  test: (1) with the mask and causality as `constexpr` flags, the masked decode compiled to
  another binary than the causal training forward and gave other bits from position 2 on; they
  are runtime flags now. (2) With Triton's default alignment specialization, `sum` over an
  unaligned view (a slice one element into its buffer) compiled to a binary with another
  `tl.sum` order and gave other bits; the row and attention kernels are compiled with no value or
  alignment specialization, and the matmul copies an unaligned operand to aligned memory and
  never specializes on the row count. Compiler chosen mul plus add contraction is turned off
  (`enable_fp_fusion=False`); that one is defensive, no test failed with it on.

## Tensor parallel: a simulation on one device

TP 2 needs two GPUs; this machine has one. `truestar.rlkernels.tp` therefore SIMULATES it: both
ranks run one after another in one process on the same GPU, each with its own shard of every
weight, and the all reduce is a Python function that adds a list of tensors. No communication
library is involved. What it shows is the arithmetic a real TP run must reproduce:

* Column parallel layers (q, k, v, gate, up, the lm head over the vocab) change no reduction.
* Row parallel layers (o_proj, down_proj) split K. Rank r holds a contiguous half of K, which is
  two of the four segments: `matmul_partials` returns its subtree `(s0 + s1)` in float32, and
  `tp.all_reduce` adds the ranks' results in a fixed binary tree in rank order, which rebuilds
  `(s0 + s1) + (s2 + s3)`. The cast to the model dtype happens once, after the all reduce.

`llama.tp_train_logits(model, ids, tp)` is the trainer forward written this way. A control test
shows the tree is what matters: ranks that reduce their half of K in one sequential pass give
other bits than TP 1. A real multi GPU run would need a collective that keeps this order and
float32 payloads (for example an all gather of the partials and the same tree on every rank);
NCCL's ring all reduce does not. Not run on real hardware.

## Results (2026-11-24, RTX 5070 Ti, torch 2.11, CUDA 12.8, Triton 3.7.1, Windows)

Model: `truestar.rlkernels.llama`, Llama style (RMSNorm, rotary, 4 query heads on 2 KV heads
through SDPA with `enable_gqa`, SwiGLU, untied head; vocab 512, width 128, 2 layers; random
weights, built in code).

**Bitwise equal per token logprobs**, float32 and bfloat16, for every token of two target rows
(51 and 54 tokens; rows 6 to 70 tokens, so keys cross attention chunks) against the training
forward of the row alone, in six paths at batch sizes 1, 3 and 8 with the row first, in the
middle and last (84 comparisons per dtype, all bitwise equal):

| path | what runs |
|---|---|
| train | teacher forced forward of the right padded batch, causal attention without a mask |
| prefill dynamic / static | one forward of the left padded batch into a KV cache, key padding mask, cache appended or written into a fixed capacity buffer |
| decode dynamic / static | prompt prefill, then one token per step against the cache, rows in lockstep, finished rows fed a masked pad |
| decode static, prefill chunks of 3 | the prompt prefilled 3 tokens at a time |

Baseline, the same paths on PyTorch's own kernels: 78 of 84 comparisons not bitwise in float32
(max 9.5e-07) and 10 of 84 in bfloat16 (max 3.8e-03, the decode paths).

**TP 1 equals the TP 2 simulation bitwise** in float32 and bfloat16 (5 rows of the same model).

**`truestar rlpair --kernels rl --strict --tp 2`** on `examples/rl_kernels_pair.py` (two rows
left padded by 3 inside the sampler's batch): status identical, max difference 0, TP 1 vs TP 2
bitwise, batch composition bitwise at batches 2 and 7 (and 3 and 8 in the test). The same pair
without `--kernels rl`: not bitwise (max 4.8e-07), batch composition not bitwise at batch 7,
exit 1 under `--strict`.

**Swapped kernel named:** a sampler whose decode steps use a hand written attention (matmul,
softmax, matmul: the same math, another reduction order) is TS185 at `blocks.0.attn#0`, trainer
`attention:rl`, sampler `matmul:rl`, sampler call `blocks.0.attn#1` (the first decode step);
the logprob gap is only 4.8e-07, so a statistics only check calls it rounding. A sampler with
attention, or RMSNorm, swapped to the eager PyTorch kernel is named the same way.

### Speed

`scripts/bench_rlkernels.py --rounds 5`, microseconds per call, fastest of 5 interleaved rounds,
two runs on an idle GPU (the table gives the first; the second was within 5% except the float32
decode step, eager 2.89 ms). eager: PyTorch (cuBLAS, SDPA, ATen). fast: Truestar's compiled
executor on a one op program, tuned per shape. rl: the RL kernels.

| float32 | eager | fast | rl | rl / eager | rl / fast |
|---|---|---|---|---|---|
| linear 8 x 2048 x 2048 | 23.5 | 60.0 | 129.4 | 5.5 | 2.2 |
| linear 512 x 2048 x 2048 | 180.5 | 266.2 | 457.3 | 2.5 | 1.7 |
| linear 4096 x 2048 x 2048 | 1274.7 | 1888.8 | 3419.5 | 2.7 | 1.8 |
| attention prefill B4 H16 S512 D64 causal | 208.7 | 271.8 | 301.0 | 1.4 | 1.1 |
| attention decode B4 H16 cache 512 | 65.0 | | 79.8 | 1.2 | |
| rms_norm 4096 x 2048 | 72.6 | | 77.5 | 1.1 | |
| log_softmax 512 x 32000 | 169.4 | | 343.6 | 2.0 | |
| softmax 512 x 32000 | 168.1 | 314.9 | 352.2 | 2.1 | 1.1 |
| model training forward B4 T512 | 18555 | | 36274 | 2.0 | |
| model decode step B4, cache 512 | 2202 | | 5963 | 2.7 | |

| bfloat16 | eager | fast | rl | rl / eager | rl / fast |
|---|---|---|---|---|---|
| linear 8 x 2048 x 2048 | 12.3 | 22.0 | 35.7 | 2.9 | 1.6 |
| linear 512 x 2048 x 2048 | 49.8 | 52.7 | 58.9 | 1.2 | 1.1 |
| linear 4096 x 2048 x 2048 | 373.6 | 357.0 | 387.5 | 1.04 | 1.09 |
| attention prefill B4 H16 S512 D64 causal | 76.1 | 32.7 | 179.2 | 2.4 | 5.5 |
| attention decode B4 H16 cache 512 | 36.7 | | 63.6 | 1.7 | |
| rms_norm 4096 x 2048 | 16.1 | | 19.5 | 1.2 | |
| log_softmax 512 x 32000 | 96.9 | | 85.3 | 0.88 | |
| softmax 512 x 32000 | 109.1 | 71.4 | 90.7 | 0.83 | 1.3 |
| model training forward B4 T512 | 9404 | | 5598 | 0.60 | |
| model decode step B4, cache 512 | 2561 | | 3473 | 1.4 | |

The model rows use the same model at dim 1024, 4 layers, 16 heads on 4 KV heads, SwiGLU 2816,
vocab 32000. Reading them:

* bfloat16 contractions cost 4% to 18% over cuBLAS at 512 rows and up, 2.9x at 8 rows (decode).
  float32 contractions cost 2.5x to 5.5x: they use the FMA path (no TF32, which would round the
  inputs), and cuBLAS's float32 SIMT kernels are better tuned than one fixed Triton tile.
* bfloat16 attention prefill costs 2.4x over SDPA's flash kernel and 5.5x over Truestar's tuned
  attention: 32 x 16 tiles with unspecialized loads. Larger tiles measured slower during tuning
  (64 x 64 at B8: 2.2x slower than 32 x 16 in bfloat16, 18x in float32).
* The bfloat16 training forward is faster on the RL kernels (0.60x) because in torch 2.11 eager
  SDPA with `enable_gqa=True` ran the math path for this call (a softmax kernel and two batched
  GEMMs in the profile), not flash. Against an eager model that repeats the KV heads and gets
  flash attention the comparison would be worse for the RL kernels; that variant was not timed.
* The decode step is host bound on both sides at this size; the RL path adds a Python mode
  dispatch per routed call. Its float32 cost (2.1x to 2.7x) is mostly the small row contractions.

The P14 acceptance target ("under 20% overhead") is met only by the large bfloat16 contractions,
the norms and softmaxes; it is not met by float32 contractions, attention or the decode step.

## Tests that prove the tests work

Permanent tests that break something on purpose:

* `test_unanchored_attention_breaks_left_padding`: key chunks start at column 0 instead of the
  anchor; left padded prefill and decode no longer match the training forward.
* `test_tensor_parallel_without_the_segment_tree_is_not_bitwise`: rank partials reduced in one
  pass differ from TP 1.
* `test_segment_count_that_depends_on_the_batch_would_break_invariance`: a K split chosen by row
  count makes batch 1 and batch 8 rows differ.
* `test_eager_pytorch_kernels_are_not_bitwise_across_paths`: the baseline is really different.
* `test_rlpair_names_a_hand_written_decode_attention`, `..._an_eager_kernel_swapped_into_the_sampler`,
  `test_rlpair_reports_a_tp_mismatch`.

Manual breaks on 2026-11-24, each restored: Triton's default specialization turned back on failed
`test_unaligned_views_give_the_same_bits`; with `enable_fp_fusion=True` all 24 tests passed
(the setting is kept as a guard, not as a proven need).

## Limits

* CUDA through Triton only; one GPU model (RTX 5070 Ti) and one Triton version (3.7.1) were run.
  Tile constants were measured on this GPU; a different GPU may need others (any constant keeps
  the invariance, the speed is what changes).
* float16 kernels exist (same tiles as bfloat16) and pass the float64 comparison, but the end to
  end acceptance ran in float32 and bfloat16 only.
* The anchor is the first attended key of a query tile. Masks whose rows start at different keys
  for one sequence (sliding windows) do not keep the invariance; additive float masks are refused.
* A fully masked query row gives zeros.
* Tensor parallel is a one device simulation; no real collective was run. Pipeline and sequence
  parallelism are not covered.
* Gradients come from the eager backward and are not bitwise invariant.
* Ops outside the table run eagerly (elementwise) or are refused (reductions); a model with a
  custom fused kernel of its own is not covered.
* Only the small Llama style model built in code was run, no pretrained checkpoint and no real
  engine (vLLM, SGLang).
