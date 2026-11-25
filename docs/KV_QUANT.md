# Quantized KV cache for decode

`truestar.kvquant` stores a decode's keys and values as int8 or fp8 (e4m3) with one float32 scale
per head and per group of tokens (or below 8 bits as int4 or NVFP4, optionally Hadamard rotated:
section "Below 8 bits"), and runs decode attention over that cache with the dequantization
inside a split KV kernel. Accumulation stays in float32 and chunks merge in a fixed order.

```python
from truestar import kvquant as KQ

kc = KQ.quantize(k, "int8", group=32)      # k [B, Hkv, N, D] float16 or bfloat16
vc = KQ.quantize(v, "int8", group=32)
out = KQ.decode_attention(q, kc, vc)       # q [B, Hq, T, D], T up to 16, Hq a multiple of Hkv
```

## Formats

| format | stored value | scale per head and group | worst error of one stored value |
|---|---|---|---|
| `int8` | round(x / s), clamped to [-127, 127] | s = amax / 127 | s / 2, plus float32 rounding of x / s (below 2^-16 of s) |
| `fp8` | x / s as `float8_e4m3fn` | s = amax / 448 | 2^-4 of \|x\| in e4m3's normal range; 2^-10 times s below it |

`group` is 16, 32, 64 or 128 tokens. A paged pool (`quantize_pages`, [P, Hkv, page, D]) needs a
page size that is a multiple of the group, so no group spans two pages; `gather_pages` builds the
dense cache of a block table with the same values and scales a dense quantization holds.

KV memory per token (all KV heads, K and V, scales included) is `bytes_per_token`: for 8 KV heads of
dim 128, float16 4096 bytes, int8 and fp8 with groups of 32 2050 bytes (50.05%).

## The kernel

One program per batch, query head and chunk of keys loads blocks of 64 int8 or fp8 keys, multiplies
by their group scales in float32, computes the scores, the running max and sum (exp2) and the
output accumulator in float32, and stores its chunk's partial. A second kernel merges the chunks in
the order 0, 1, 2, ... and writes the output in q's dtype. A grouped query head reads its KV head
(`h // (Hq / Hkv)`) directly, so a GQA cache is never expanded.

Chunks (`chunk_for`, since 2026-11-25 the shapes decide, never the GPU): in deterministic mode
(`deterministic=True`, or `torch.use_deterministic_algorithms`) 256 keys from a context of 257,
so the bits depend on the context length only. Otherwise the smallest power of two chunk (at least
64 keys) that gives about 256 programs (a constant, not the SM count): one program per KV head
with the grouped kernel, else one per query head and row. Before 2026-11-25 contexts below 2048
keys never split and the default chunk followed the SM count.

Grouped query heads (`grouped_block`, since 2026-11-25): when a KV head serves 2 to 64 query rows
(query heads of the group times query rows) and keys are scaled per token, one program runs all of
them as the rows of one tile: `tl.dot` on the int8 or e4m3 codes (exact in float16 and bfloat16)
with float32 accumulation, the K scale applied per key to the scores and the V scale per block (a
block of 64, 32 or 16 keys lies in one V group) to P V. A KV block is read once for the whole
group, against once per query head before. Per channel keys and single row calls keep the one
row kernel. Both are checked against float64 at the first call.

## Error bound against the unquantized cache

`error_bound(q, kc, vc, k, v)` returns, per query row and output column, a proven upper bound on the
difference between attention over the quantized cache and attention over the unquantized one:

    |o' - o| <= (exp(2 E) - 1) * sum_j w_j |v_j - o| + max_j |v'_j - v_j|

with w and o the exact weights and output and E the largest change of a scaled score. Why:
o' - o = sum_j (w'_j - w_j)(v_j - o) + sum_j w'_j (v'_j - v_j), because both weight vectors sum to
one; a score change of at most E moves each weight by a factor within [exp(-2 E), exp(2 E)]; and the
w' sum to one. It is computed in float64 from the cache's own rounding errors, so it holds for that
cache, not for a typical one.

Measured (RTX 5060 Laptop GPU, q, k, v standard normal, 8 heads of dim 128, groups of 32):

| | int8 | fp8 |
|---|---|---|
| largest absolute error, batch 1 to 8, 2048 to 8192 keys | 8.2e-4 to 2.4e-3 | 2.2e-3 to 7.3e-3 |
| against the float16 cache through eager SDPA (4096 keys) | 36 times | 98 times |
| the proven bound on the same data | 0.08 to 0.10 | 0.39 to 0.41 |

The bound is 40 to 160 times the measured error: it assumes every rounding error lines up against
the output. tests/test_kvquant.py checks it on every element, with peaked scores as well.

## Verification and determinism

* The first call for each (shape, chunk, format, group, dtype) is compared with float64 attention
  over the dequantized cache, the exact values the kernel reads; a mismatch raises
  `KernelMismatchError`. Measured relative error 2.2e-4 to 3.4e-4.
* No atomics: every run gives the same bits. Deterministic chunks give the same bits with the SM
  count pinned to 26 and to 70, and a sequence decoded alone equals the same sequence in a batch.
  A paged and a dense cache give the same bits (tests/test_kvquant.py).
* Refused with TS701 (`KVQuantError`): an unknown format, a group other than 16 to 128, more than
  16 query rows, a head dim that is not a power of two up to 256, K and V of different formats, a
  chunk that is not a multiple of 64, a CPU tensor. Nothing falls back to another path.

## Per channel key scales

`quantize(k, fmt, group, channel=True)` keeps one scale per head, channel and block of `group`
tokens ([B, Hkv, N / group, D]); values always take one scale per token group (`decode_attention`
refuses per channel V with TS701). One scale per channel per block of D x g tokens costs the memory
of one per group of g tokens (for D 128 and g 32: blocks of 4096), so
`scripts/kvquant_channel_study.py` compares the two at equal memory (8 heads of dim 128, batch 1 and
8, 2048 and 8192 keys, largest absolute error against float64 over the unquantized cache):

| keys | int8: per channel wins | fp8: per channel wins |
|---|---|---|
| standard normal | 1 of 4 (error 5% higher to 20% lower) | 0 of 4 (error 8% to 18% higher) |
| 4 of 128 channels scaled by 20 | 4 of 4 (3.4x to 4.4x lower error) | 0 of 4 (1.04x to 2.5x higher error) |
| magnitudes drifting 0.2 to 5 along the sequence | 3 of 4 | 3 of 4 |

Per channel does not win every case, so the default stays one scale per token group and per channel
keys are an option; they pay off for int8 keys whose outliers sit in a few channels, the pattern
reported for trained LLMs (random data here, not a trained model). fp8 already spends its exponent
on magnitudes, and a block wide channel scale costs it resolution. In a paged pool the page must be
a multiple of the block, so per channel keys there need small blocks (more scale memory).

## Below 8 bits: int4 and NVFP4, rotated or not

`KQ.quantize(k, "int4" | "nvfp4", group, rot=False | True)` (code: `kvquant/low.py`). Research item
2.9 (TurboQuant, ICLR 2026; NVIDIA's NVFP4 KV cache, 2025-12; vLLM's TurboQuant study, 2026-05).

| format | element | per token and 16 channels | per head and group of tokens | bytes per token, 8 KV heads of 128 |
|---|---|---|---|---|
| float16 | | | | 4096 |
| int8, fp8 | 8 bits | | float32 | 2050 |
| `int4` | integer -7 to 7 | E4M3 scale, block amax / 7 | float32, group amax / (448 x 7) | 1154 (28.2%) |
| `nvfp4` | E2M1 | E4M3 scale, block amax / 6 | float32, group amax / (448 x 6) | 1154 (28.2%) |

Two channels share a byte (channel j low nibble, j + D / 2 high), so the kernel reads the two
halves of the head dim with no interleave. The second level scale is kept per head and token
group, not per tensor as NVFP4 weights do, so a growing cache never rescales what it stored and
a paged cache equals a dense one. Codes are round to nearest, ties to even, against the stored
scales, computed in float64 with the format tables of `truestar.quant.formats`;
`low.fake_quantize` gives the same values on the CPU for model studies (tested equal).

**Rotation** (`rot=True`, a power of two head dim up to 256): the cache stores k H and v H with H
the +1 / -1 Sylvester matrix (H H = D I), computed in float64 from the float16 cache (exact unless
a row's exponents spread over 2^38). Decode rotates the query by the fast Walsh kernel (float32,
`rotate/walsh_kernel.py`, checked on its first call, TS1023), scales the scores by 1 / D and
turns the output back by o H / D (a power of two). In exact arithmetic nothing changes:
(q H)(k H)^T / D = q k^T. The rotation is applied after RoPE, online, because RoPE does not commute
with a rotation of the head dim (folding it into W_q and W_k is the wrong placement the rotation
proof catches, docs/ROTATION.md).

**Same guarantees as int8 and fp8.** The first call per shape is checked against float64
attention over the stored values turned back to the unrotated basis (`QuantKV.original()`);
`error_bound` is computed from those same values, so the proven bound holds for a rotated cache
with no transfer argument (it is the same theorem on the values the cache holds). No atomics,
deterministic chunks ignore the SM count, a sequence alone equals it in a batch, paged equals
dense bitwise (tests/test_kvquant_low.py, also run on an L4, sm_89). Not supported (TS701): a
per channel scales, rotation of int8 or fp8. A float tail and a fixed capacity prefix
(``tail=``, ``prefix_len=``) are read as for int8 (2026-11-25), so the 4 bit formats also run in the
compiled decode step below.

Measured on a real model's cache (`scripts/kvquant_low_study.py`, results/rotation/kv_study.json):
SmolLM2-360M, 32 layers of 5 KV heads of dim 64, WikiText-2 test text, groups of 32 tokens.

*Per layer decode:* keys after RoPE and values of a 2048 token prefix, the last token's query, the
decode kernel over each cache against float64 attention over the float16 cache (largest absolute
error per layer; the largest output magnitude per layer is 0.26 to 5.7, median 1.6):

| cache | error, median of 32 layers | error, worst layer | proven bound, median | bound / error, smallest |
|---|---|---|---|---|
| int8 | 0.040 | 0.198 | 0.73 | 5.3 |
| fp8 | 0.106 | 0.479 | 6.1 | 16 |
| int4 | 0.243 | 1.10 | 62 | 119 |
| int4 rotated | 0.172 | 0.935 | 23 | 55 |
| nvfp4 | 0.325 | 1.72 | 119 | 84 |
| nvfp4 rotated | 0.256 | 0.808 | 52 | 35 |

The bound held on every layer and format (it is checked on every element by the tests). At 4 bits
it is 35 to 119 times the error: exp(2 E) - 1 grows fast with the score change E, so it is a
guarantee, not an estimate. Rotation lowers the median error by 29% (int4) and 21% (NVFP4) and the
median bound by 2.7x and 2.3x.

*Perplexity* with every layer's whole cache in the format (2 x 1024 tokens after the prefix, the
float32 model on the CPU, the cache values from the same quantizer the kernel reads):

| cache | perplexity | against the float16 cache | KV bytes per token, this model |
|---|---|---|---|
| float16 | 15.007 | | 40960 |
| int8 | 15.017 | +0.07% | 20520 (50.1%) |
| fp8 | 15.019 | +0.08% | 20520 |
| int4 | 15.738 | +4.9% | 11560 (28.2%) |
| int4 rotated | 15.269 | +1.7% | 11560 |
| nvfp4 | 15.542 | +3.6% | 11560 |
| nvfp4 rotated | 15.348 | +2.3% | 11560 |

*Speed, briefly* (RTX 5070 Ti, idle; 32 query and 8 KV heads of dim 128, one query row, random
data, `triton.testing.do_bench`, microseconds; float16 is `torch.nn.functional.
scaled_dot_product_attention` with `enable_gqa`, which is not a decode kernel and is shown only as
the plain PyTorch path):

| B, N | float16 SDPA | int8 | fp8 | int4 | int4 rotated | nvfp4 | nvfp4 rotated |
|---|---|---|---|---|---|---|---|
| 1, 4096 | 652 | 37 | 36 | 62 | 79 | 90 | 94 |
| 1, 16384 | 2527 | 124 | 110 | 217 | 221 | 330 | 332 |
| 8, 4096 | 5093 | 222 | 205 | 418 | 424 | 641 | 647 |
| 8, 16384 | 20327 | 834 | 916 | 1637 | 1640 | 2529 | 2537 |

The 4 bit kernels are 1.7x to 2.0x (int4) and 2.4x to 3.0x (NVFP4, whose E2M1 decode is a chain of
selects) slower than int8, though they read 44% less memory: the dequantization is on the
CUDA cores and dominates, the result vLLM's TurboQuant study reports (FP8 stays the best default;
4 bit costs throughput and buys capacity). Rotation adds two small launches: 27% at batch 1 and
4096 keys, under 2% from 16384 keys or batch 8. So 4 bit KV is for memory bound serving (1.78x the
tokens of int8 in the same memory), with rotation on, and int8 or fp8 otherwise.

## In a compiled decode (opt in)

`examples/llama_kvquant_decode.py`: `make_decode_step(model, kv_cache=None | "int8" | "fp8")`.
`None` is the dense step of examples/llama_paged_decode.py itself, so its program and bits are what
they were (the fingerprint's 9 compiled hashes are unchanged). With a format, each layer keeps a
quantized prefix of whole groups and a float tail of the newest tokens:

* the step writes the new token into the tail (`index_copy`) and calls
  `torch.ops.truestar.kvq_attention`, which the importer turns into the IR op `kvq_attention`
  (float64 reference: attention over the dequantized prefix and the live tail, grouped query heads
  mapped to their KV head) and the compiled executor runs as a library step through
  `decode_attention` (the kernel reads the tail as one more chunk, merged last);
* the host's `QuantCache.after_step` quantizes a full tail once, from its float values, and appends
  it to the prefix: no value is quantized twice, so the storage bounds above hold for every token;
* the program checks the step against the float64 reference on its first call, like every library
  step (a wrong softmax scale while building this was caught there, relative error 0.44);
* `layer_error_bounds` reports `error_bound` per layer for a step (the cache keeps the unquantized
  groups only when asked, `keep_orig=True`); deterministic mode is passed to the kernel.

Measured end to end (`scripts/bench_kvquant_decode.py`): the model of scripts/bench_decode.py
(vocab 8192, dim 512, 4 layers, 8 query and 2 KV heads of dim 64), random weights and cache contents,
32 decoded tokens, all programs float32 (compiled in float16 the dense step failed Truestar's own end
to end check on these weights and returned nothing). Tokens per second (median step) and KV memory:

| B | N | float32 cache | int8 | fp8 | KV MiB float32 / int8 and fp8 |
|---|---|---|---|---|---|
| 1 | 2048 | 198 | 173 | 201 | 8.1 / 2.2 |
| 1 | 8192 | 177 | 208 | 198 | 32.1 / 8.2 |
| 8 | 2048 | 1017 | 1407 | 1479 | 65.1 / 17.3 |
| 8 | 8192 | 422 | 1452 | 1490 | 257.1 / 65.4 |

KV bytes per token: 4096 float32, 2048 a float16 cache would take, 1026 int8 and fp8 (scales
included). At batch 1 every variant is bound by about 5 ms of host time per step, so the formats tie
within noise (int8 at 2048 keys 13% slower, at 8192 keys 18% faster). At batch 8 the quantized cache
is 1.4x to 3.5x faster; part of that is the dense step copying its whole cache every step (value
semantics), which the quantized step does only for its tail.

### 4 bit formats in the compiled decode (2026-11-25)

`make_decode_step(model, "int4" | "nvfp4", rot=False | True)` (and `import_quant_step(..., rot=)`,
`QuantCache.from_dense(..., rot=)`): the prefix is packed 4 bit data uint8 [L, B, HKV, N, HD / 2] with
its E4M3 block scales passed as uint8 bits [L, B, HKV, N, HD / 16], two inputs after `tlen`
(`LowQuantDecodeStep`). `kvq_attention` takes them as `k_bscale` and `v_bscale` and the attribute
`rot`; its float64 reference dequantizes with `kvquant.low.dequantize` and turns a rotated cache
back (`x H / D`), so the first call check compares against attention over exactly the values the
cache holds. The 4 bit kernel gained the float tail program of the int8 kernel (one more program per
row, merged last) and the fixed capacity prefix length. A rotated cache stores `x H`; its tail stays
in the natural basis in the step and is rotated in float32 (one fast Walsh launch each for K and V)
before the kernel reads it, so every key the kernel reads is in the cache's basis. A full tail is
quantized once by the host (`after_step`), with the rotation when the cache is rotated.

Tests (tests/test_kvquant_decode.py, one L4): the kernel with a tail and a fixed capacity prefix
against float64 for int4 and NVFP4, rotated or not, float16 and bfloat16, and bitwise equal to the
cache of exactly the live keys; the compiled step for all four within 2% of the eager float64 step
over 15 steps that cross a group fill, two calls bitwise equal; the fixed capacity step as one CUDA
graph equal bitwise to the growing step over 30 steps.

Measured (`scripts/bench_kvquant_decode.py --mode graphed --fixed --formats
int8,fp8,int4,int4_rot,nvfp4,nvfp4_rot`, one Modal NVIDIA L4, torch 2.11, float32 programs, groups of
32, 32 timed steps after 3, median microseconds per step, the float32 row is the dense step graphed
with its cache updated in place):

| B | N | float32 | int8 | fp8 | int4 | int4 rotated | nvfp4 | nvfp4 rotated |
|---|---|---|---|---|---|---|---|---|
| 1 | 2048 | 1020 | 851 | 845 | 876 | 903 | 902 | 927 |
| 1 | 8192 | 1874 | 963 | 973 | 1028 | 1054 | 1170 | 1211 |
| 8 | 2048 | 3052 | 1186 | 1171 | 1364 | 1383 | 1713 | 1725 |
| 8 | 8192 | 9866 | 1671 | 1629 | 2318 | 2331 | 3904 | 3894 |

KV bytes per token (4 layers, 2 KV heads of 64): 4096 float32, 1026 int8 and fp8, 578 int4 and
NVFP4; at B 8, N 8192 the cache takes 257.1, 65.6 and 37.4 MiB. So in the compiled decode the 4 bit
formats hold 1.78x the tokens of int8 in the same memory and 7.1x those of a float32 cache, and are
1.16x (B 1, N 2048) to 4.26x (B 8, N 8192) faster per step than the float32 cache, but 3% to 39%
slower than int8 (int4) and 6% to 134% slower (NVFP4, whose E2M1 decode is a chain of selects):
choose them for capacity, as the standalone measurements above say. Rotation costs at most 3.1%.

## Limits

* The quantized step is a separate decode step (one token per sequence); prefill runs the float
  dense step, and `QuantCache.from_dense` converts its cache. Every sequence of a batch has the same
  length (as in the dense step).
* The compiled executor never chooses the quantized cache by itself: the model opts in.
* Measured on one 26 SM laptop GPU only; fp8 needs a GPU and Triton with e4m3 loads (sm_89 and up).
