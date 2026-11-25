# Decoder LLMs in the IR and the native front end

Author: Sharven Rane. First written 2026-11-24.

Pain rows served: P7 (export bakes in shapes: one decode program for every position), P5 (no
recompiles across positions, batch sizes and cache sizes), P15 (prefill and decode are the same
kind of program, ready for one kernel set). Code in `src/truestar/ir/` (shapes, reference,
importer, grad_rules) and `src/truestar/frontend/`; tests in `tests/test_llm_ops.py`; the native
example in `examples/llama_kv_decode.py`.

## What works

* A Llama style decoder (RMSNorm, rotary positions, grouped query attention with 4 query heads on
  2 KV heads, SwiGLU, lm head tied to the embedding; random weights) imports through `torch.export`
  as a prefill graph and as a decode step graph, in the export graph and in the core ATen graph, and
  matches PyTorch float64.
* The same model written natively (`@truestar.function`) runs the prompt and every decode step with
  one compiled program and decodes greedily token for token like the PyTorch model.
* The ops a sampler needs (sort, top k, cumsum based top p filtering, temperature, argmax,
  multinomial with a seed) exist as IR ops or lower onto them.

## New IR ops

Each is a new op name. No existing op changed meaning, so a consumer that does not know one of these
ops (the compiled executor, for instance) fails on its name instead of computing something else.

| Op | Meaning | Reference (float64, no fused PyTorch op) |
|---|---|---|
| `flip(x) dims` | reverse along dims | indexing with a reversed index |
| `sort(x) dim descending stable` | values and indices; equal values keep input order, NaN sorts last | rank counting: an entry's rank is the number of entries that go before it plus the equal ones at an earlier position |
| `topk(x) k dim largest sorted` | the first k of that stable sort | the same rank counting; the dim must have a fixed size |
| `argmax`, `argmin` `dim keepdim` | first index of the extreme; NaN counts as the extreme | comparison with the reduced extreme, then the smallest matching position |
| `amax`, `amin` `dims keepdim` | max or min over any dims; NaN propagates | a running comparison over each reduced dim |
| `index_copy(x, index, src) dim` | x with rows `index` replaced by src; the index is data | indexed assignment; repeated or out of range indices are an error |
| `multinomial(weights, offset?) seed num_samples` | categorical draws | inverse CDF, see Sampling |

Multi value ops (`sort`, `topk`) are built with `Builder.op_multi` and imported with `_emit_multi`.

## Lowered onto existing ops

* **Grouped and multi query attention.** `scaled_dot_product_attention(..., enable_gqa=True)` splits
  the query heads dim `[.., Hq, T, E]` into `[.., Hkv, n, T, E]` (n = `Hq / Hkv`), gives k and v a size
  1 dim there (`[.., Hkv, 1, S, E]`, and a mask with a heads dim the same way), runs `attention`, which
  broadcasts each KV head over its n query heads, and merges the heads back. Every reshape is a view,
  so the KV cache is never copied. Query head h reads KV head h // n, which is PyTorch's definition
  (a `repeat_interleave` of the KV heads). Until 2026-11-24 the KV heads were repeated with a reshape,
  an expand and a reshape, which materialized the cache n times per layer in compiled decode steps
  (the largest copies of a decode step, docs/QUALITY_BAR.md P38). `aten.repeat_interleave.self_int`
  still lowers to that reshape, expand, reshape. Head counts must be fixed; batch and sequence may be
  symbolic. Because it reuses `attention`, autodiff and the compiler see nothing new.
* **Masks and sliding windows.** An explicit bool or additive mask was already an operand of
  `attention`. A sliding window built from `arange` comparisons (the way Hugging Face builds it)
  imports as those comparisons. Natively, `ops.attention(..., is_causal=True, window=W)` builds the
  same mask (key j is kept for query i when `i - W < j <= i`).
* **Temperature.** `softmax(x / t)`; natively `ops.softmax(x, dim, temperature=t)` with a number or a
  scalar tensor.
* **Top p filtering.** `ops.top_p_filter(logits, p, min_keep=1)`: sort ascending, softmax, `cumsum`,
  mark tokens whose cumulative probability stays at or below `1 - p` (the last `min_keep` sorted
  tokens are always kept), put the mark back in the original order by a `gather` with the inverse
  permutation (a second `sort` of the indices), set the marked logits to minus infinity. This is
  Hugging Face's `TopPLogitsWarper`, which uses a scatter instead of the inverse permutation.
* **`max.dim`, `min.dim`** become `argmax` or `argmin` plus a `gather` of the values. `aten.amax`,
  `aten.amin`, `aten.max.default` and `aten.min.default` map to `amax` and `amin`, and the
  elementwise `aten.max.other` and `aten.min.other` to `maximum` and `minimum`, so the reductions
  import whichever spelling the torch version emits (CI on torch 2.14 met `aten.amax` in a graph that
  torch 2.11 exports without it).
* **`index_put`.** One integer index tensor (`x[:, :, pos] = k`) becomes `index_copy`. A boolean mask
  with a single value (`x[mask] = 0`) becomes `where`. Both keep every shape static.
* **Chunk and split along a symbolic dim.** Allowed when the piece count is static and every piece has
  the same size that is provably an integer, for example `chunk(2)` of a dim `2*T`: a reshape to
  `[.., 2, T, ..]` and one `select` per piece. Derived dims (`2*T` in `dynamic_shapes`) keep the root
  symbol's name, and the reference interpreter solves `2*T = 10` for `T`.

## Still refused, loudly

* Reading with a boolean mask, `x[mask]`: its size depends on the data. The import error names the
  node and suggests `torch.where` or `masked_fill`; the native front end refuses boolean indexing.
* `torch.multinomial` and other PyTorch random ops: their generator cannot be reproduced. Use the IR
  op `multinomial` (`ops.multinomial`) instead.
* `index_put` with several index tensors, with `accumulate=True`, or a mask with many values;
  scatter in general.
* Chunk or split along a symbolic dim into unequal pieces or a dim dependent count (`chunk(2)` of
  `T`); `topk` along a symbolic dim (k cannot be proved to fit).

## KV cache decoding

The cache has a fixed capacity `S` (a symbolic dim) and is an input and an output of the step. The
positions of the new tokens are an int64 tensor input `pos` (data, not a shape). The step writes
`k` and `v` at `pos` with `index_copy` and attends with the mask `arange(S) <= pos[:, None]`, so
unwritten slots never leak in. Nothing in the program depends on the position or the cache length,
so one program serves them all.

Native (`examples/llama_kv_decode.py`):

```python
@truestar.function
def forward(ids: Tensor["B", "T", "i64"], pos: Tensor["T", "i64"],
            k_cache: Tensor[L, "B", HKV, "S", HD, "f64"], v_cache: ..., embed: ..., ...):
    ...
    kc = ops.index_copy(kc, 2, pos, k)                    # inside each layer
    a = ops.attention(q, kc, vc, mask=keep, enable_gqa=True)
    ...
    nxt = ops.argmax(logits[:, ids.shape[1] - 1], dim=-1)  # greedy sampling in the program
    return logits, nxt, ks, vs
```

`forward(ids: i64[B, T], pos: i64[T], k_cache: f64[2, B, 2, S, 8], ...) -> (f64[B, T, 64], i64[B],
f64[2, B, 2, S, 8], f64[2, B, 2, S, 8])`, 133 nodes. The same program runs the prompt (T tokens)
and each decode step (T = 1). It is compiled once, at decoration; the 17 calls of a 16 token
generation compile nothing, for batch 2 with capacity 24 and batch 1 with capacity 40 alike.

Import: the PyTorch model's `forward(ids)` (prefill, returns logits and every layer's K and V) and
`step(ids, pos, k0, v0, k1, v1)` (decode, uses `index_copy` and `enable_gqa`) export with symbolic
`B`, `T` and `S`.

## Results (2026-11-24, torch 2.11, CPU, float64)

Greedy decoding for 16 tokens, against the plain `torch.nn` model recomputing the whole sequence at
every step in float64 (no cache on the PyTorch side, so the cache logic is checked, not shared):

| Path | Tokens | Worst logits relative error |
|---|---|---|
| native, batch 2, prompt 5, capacity 24 | equal | 1.9e-15 |
| native, batch 1, prompt 3, capacity 40 | equal | 1.7e-15 |
| imported prefill program, then 15 steps of one imported decode program | equal | below 1e-13 (asserted) |

The smallest gap between the top two logits over those steps is 3.0e-3, twelve orders of magnitude
above the error, so equal tokens is a real check. The random weights are scaled so attention and the
MLP outweigh the tied embedding: with the default init greedy decoding repeats one token and a
broken cache could go unnoticed. The generated sequences use more than 8 distinct tokens.

Imported programs against PyTorch float64, worst relative error over three shapes each:

| Graph | Prefill (B, T) = (2, 5), (3, 9), (1, 2) | Decode (B, S, position) = (2, 10, 3), (3, 17, 16), (1, 4, 0) |
|---|---|---|
| export graph (120 and 128 nodes) | 9.9e-16 | 9.6e-16 |
| core ATen graph (239 and 225 nodes) | 2.7e-15 | 9.6e-16 |

Every new op matches PyTorch float64 under 1e-10 (integer results exactly) in both graph forms with
a symbolic batch, at a second batch size from the same import: 19 import cases and 8 native
functions compared with the same source run eagerly.

## Paged KV cache (2026-11-24)

Code: `examples/llama_paged_decode.py` (model, host block allocator, generation loops), tests in
`tests/test_paged_kv.py`, benchmark `scripts/bench_decode.py`.

Layout: one pool per K and V for all layers and all sequences, `[L, P, HKV, HD]` with `P` = blocks
x block size. Each sequence owns a list of blocks anywhere in the pool (its block table, int64
`[B, M]`). One decode step, written in plain PyTorch and imported through `torch.export`:

1. writes the new K and V at their slots with `index_copy` on the pool's slot dim. The slot of
   position `p` of sequence `b` is `table[b, p // block] * block + p % block`, computed on the host
   (`BlockPool.slots`) the way a serving engine computes its slot mapping, because the IR has no
   integer floor division yet;
2. gathers each sequence's cache in position order with one `index_select` of the pool, index
   `table[:, :, None] * block + arange(block)` (integer arithmetic inside the program);
3. attends with the mask `arange(S) <= pos[:, :, None]`, positions per sequence (`pos` is `[B, T]`),
   so the slots past a sequence's own position, which can hold another sequence's data, never count.

Nothing new was needed in the IR: `index_copy`, `index_select`, integer `mul` and `add`, `arange`
and `attention` already existed. Both steps import with symbolic batch `B`, token count `T`, table
width `M` and pool size `P` (the dense step with `B`, `T` and capacity `S`): one program for the
prompt and every step, compiled once.

### Checked

| Check | Result |
|---|---|
| Paged against dense, float64 reference, 10 steps, blocks in shuffled order, unused pool slots filled with pi, table wider than needed | tokens equal, logits **bitwise equal** at every step |
| The same, compiled on CUDA in float32 (`compile_program`) | tokens equal, logits **bitwise equal**; 0 fallbacks in either program; the float32 tokens equal the float64 reference tokens; no recompile across positions |
| Paged step imported against the same PyTorch module run eagerly in float64 | tokens equal, logits within 1e-13 |
| Ragged batch: sequences of length 6 and 2 prefilled alone into one pool, then decoded together for 5 steps | each matches its own dense decode at batch 1: tokens equal, logits within 1e-13 |
| Weights copied from a `LlamaForCausalLM` built from a config (`from_hf`), paged decode in the reference against `generate` in float64 | 8 of 8 tokens equal per sequence |
| An out of range or repeated slot | `IRError` in the reference and in the compiled program (the compiled one clamps first, so no CUDA device assert) |

Why bitwise: masked scores become exactly 0 after the softmax and `0 * v` adds nothing, so the
gathered rows that differ from the dense cache contribute exactly zero, and the reduction runs over
the same `S` entries in the same order. This needs the unused slots to hold finite numbers (a NaN
there would survive `0 * NaN`); a fresh pool is zeros and a recycled block holds old finite K and V.

Tests that fail on purpose: a block table shifted by one block changes the tokens
(`test_a_wrong_block_table_changes_the_tokens`). Manual break, restored: a mask that lets each query
see one slot past its position failed 4 of the 7 tests (bitwise reference, ragged, compiled,
Hugging Face).

### Decode benchmark

`scripts/bench_decode.py`: `LlamaForCausalLM` built from a config (vocab 8192, width 512, 4 layers,
8 query and 2 KV heads, MLP 1408, random weights scaled 3x so greedy tokens vary), prompt 32, 64 new
tokens, block 16, capacity 96. Truestar steps compiled in float32 on CUDA; the eager rows run the
same dense step module in PyTorch and Hugging Face `generate` (sdpa, its dynamic cache). Milliseconds
per generated token, median of interleaved rounds after a warm up; two separate runs on the shared
RTX 5070 Ti (3 and 5 rounds), torch 2.11:

| batch | Truestar paged | Truestar dense | eager dense step | HF generate |
|---|---|---|---|---|
| 1 | 4.65, 5.71 | 3.11, 3.68 | 3.39, 2.98 | 3.42, 3.46 |
| 8 | 4.99, 5.41 | 3.70, 4.73 | 3.76, 6.72 | 3.30, 3.63 |

Agreement in every run: paged logits bitwise equal to dense, dense tokens 100% equal to `generate`.

Reading it: at this size the steps are small and host work per step matters most (a cProfile of
the compiled paged run is led by index checks, Triton launches and layout steps; the GPU time per
step was not measured separately). The compiled dense step is level with eager PyTorch and
`generate` within the noise of the shared GPU. The paged step costs 1 to 2 ms more per token: 8
gathers and 8 pool writes per step instead of 8 cache writes, plus the device side index checks.
Paging is about memory, not speed here: the pool holds only the blocks sequences use and sequences
of different lengths share it (the ragged test), where the dense cache reserves `B x S` per layer.

Found while building it:

* The index library steps first checked their indices with one host sync per op (16 per paged
  step), 8.2 ms per token at batch 1. The checks are now recorded on the device, read once per call,
  and a repeated index tensor (the same gather index in every layer) is checked once: 5.8 then 4.7 ms.
* `arange` with a symbolic end did not run in the compiled executor (a sympy symbol reached
  `torch.arange`); it now takes its length from the dim table.

Not done: in place pool updates (each step returns a new pool, value semantics; a donated buffer
would avoid the copy), a fused paged attention kernel that reads the block table directly (the
gather materializes `[B, S, HKV, HD]` per layer), CUDA graph capture of the step, prefix sharing
between sequences, and integer floor division in the IR (the slots come from the host).

## Sampling

`multinomial` draws category `k` for row `r` and sample `s` as the first index whose running weight
sum exceeds `u * total`, with

```
u = (splitmix64(splitmix64(splitmix64(seed) ^ offset) ^ (r * num_samples + s)) >> 11) * 2^-53
```

in 64 bit unsigned integer arithmetic. `seed` is an attribute; `offset` is an optional int64 scalar
operand (the decode step, for example), so one program gives fresh draws at every step and the same
draws for the same seed and offset on any backend. Zero weight categories are never drawn; a row
with a negative, infinite or NaN weight, or no positive weight, is an error. The test checks the
draws against `torch.searchsorted` with the same uniforms, determinism, and frequencies over 4000
draws within 0.032 of the weights. Limit: the running sum is float64; a backend summing in float32
can pick a neighbouring category when `u * total` falls within rounding of a boundary.

## Autodiff

Added to `grad_rules.py` and checked by finite differences in `tests/test_autodiff.py` like every
other rule: `flip` (flip the cotangent back), `index_copy` (the cotangent with the written rows
zeroed for x, its rows at the index for src), `amax` and `amin` (the cotangent goes to the entries
equal to the result, split evenly among ties, as PyTorch does). Grouped query attention needs no rule:
its lowering is reshape, expand and `attention`, which have rules; a test compares the gradient of a
GQA attention layer with `torch.autograd` (under 1e-10).

`sort` and `topk` have rules since 2026-11-24 (scatter style through the new `scatter_add` op, ties
in stable order; docs/AUTODIFF.md), and so do `gather` and `index_select`, so an embedding lookup
trains.
`argmax`, `argmin` and `multinomial` return integers and carry no gradient.

## Native front end additions

`ops.cos`, `sin`, `log`, `cast` (and the methods `.float()`, `.double()`, `.long()`), `arange`,
`rms_norm`, `cumsum`, `flip`, `argmax`, `argmin`, `amax`, `amin`, `sort`, `topk`, `gather`,
`index_select`, `embedding`, `index_copy`, `masked_fill`, `repeat_interleave`, `chunk`, `split`,
`multinomial`, `top_p_filter`; `ops.attention` gained `enable_gqa` and `window`, `ops.softmax` gained
`temperature`. The matching `torch.*` spellings are aliases, except `torch.multinomial`.

## Found on the way

`torch.export` keeps only one of two example inputs that share storage: exporting the decode step
with four KV caches made from one zeros tensor gave a graph that read the first cache four times,
and the imported program silently ignored the other three. `ir.import_module` now refuses example
inputs that share storage, naming them.

## Tests that prove the tests work

* `test_broken_reference_op_is_caught` breaks `flip` (one dim dropped), `sort` (order reversed),
  `topk` (rolled), `argmax` (clamped), `index_copy` (writes one slot later) and `amax` (second
  largest); each case then fails.
* `test_native_decode_with_a_stale_cache_diverges`: a cache write that does nothing changes the
  generated tokens. `test_broken_kv_cache_attention_is_caught_by_the_decode_comparison`: attention
  that ignores the mask fails the decode comparison.
* Manual breaks on 2026-11-24, each restored: the stable sort tie rule inverted failed
  `test_sort_is_stable_with_ties_and_nan_sorts_last`; `index_copy` writing `src` flipped along the
  dim failed 5 tests (three import cases, the native op test and native greedy decoding). The imported
  decode test did not fail on that break because a single position write is unchanged by a flip,
  which is why the native test writes a whole prompt.

## Limits

* The native example runs in the float64 reference interpreter. The imported dense and paged steps
  also run compiled (see Paged KV cache).
* The model in the tests is small (vocab 64, width 32, 2 layers) and has random weights; no
  pretrained checkpoint was run (no downloads).
* Only torch 2.11 was run here; mappings for the other spellings of the max and min reductions were
  written for torch 2.14 but not run against it.
* Rotary positions are computed in the program from `pos`. The native example's cache is fixed
  capacity; the paged cache exists for the imported model (examples/llama_paged_decode.py).
* `sort` and `topk` reference kernels compare every pair along the dim (memory and time grow with
  the square of its size), fine for tests, slow for a real vocabulary.
