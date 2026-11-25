# Prefix cache check

A serving engine that caches a shared prefix (prefix caching, chunked prefill) runs the prefix once,
keeps its K and V, and runs only the new tokens against them. In exact arithmetic that is one full
prefill of the whole sequence. In floating point it is the same only when every reduction sees the
same values in the same order in both runs. `truestar prefixcheck` tells whether it is, and when it is
not, where the runs part and whether the cached path is merely rounded differently or less accurate.

## From files (any engine)

```
truestar prefixcheck full.pt cached.pt --reference ref.pt [--tolerance 1e-3] [--format agent]
```

Each file holds the logits at the same new tokens (`[..., tokens, vocab]`): a `.pt` tensor, a `.pt`
dict with `logits` and optionally `layers` (per layer tensors, such as the K and V caches, in model
order), or a `.npz` with `logits` and `layers/<name>` arrays. The reference is float64 logits of the
same tokens, when one exists. Exit codes: 0 bitwise equal, 1 they differ, 2 usage error.

| code | severity | meaning |
|---|---|---|
| TS704 | info | `prefix_cache_bitwise`: the cached run gives the full prefill's bits |
| TS702 | warning | `prefix_cache_divergent`: they differ; the first token and the first layer that differ, and the largest difference |
| TS703 | error | `prefix_cache_wrong`: against the reference the cached run's error exceeds the full prefill's by more than the tolerance (relative to the reference's largest value): a less accurate path, not only rounding |

From Python: `truestar.prefixcheck.compare(full, cached, full_layers, cached_layers, reference)` and
`check_files(...)` return a report with `bitwise`, `first_token`, `first_layer`, `layer_max_abs`,
`error_full_vs_reference`, `error_cached_vs_reference` and the codes.

## Truestar's own caches

`scripts/prefix_check_llama.py` runs a Llama style decoder (examples/llama_paged_decode.py) compiled in
deterministic mode, a full prefill against a cached prefix at several split points, with the K and V
caches of every layer and a float64 reference. Measured on an RTX 5060 Laptop GPU, float32:

| cache | result | against float64, cached / full |
|---|---|---|
| dense: prefix prefill, then the rest as one chunk | bitwise at splits 1, 37, 64, 120 of 160 and 2100 of 2200 | equal |
| paged: the same through a paged pool, blocks shuffled | bitwise at every split | equal |
| decode: the prefix, then one token at a time | bitwise at every split, also past 2048 keys | equal |
| int8 prefix (groups of 16), then one token at a time | TS702 from the first new token, largest difference 9e-4 to 1.4e-3 | 3.7e-4 to 5.7e-4 / 3.5e-7 to 3.8e-7 |
| fp8 prefix | TS702 and TS703, largest difference 2.9e-3 to 5.5e-3 | 1.2e-3 to 2.2e-3 / the same |

Why the dense, paged and decode caches are bitwise: in deterministic mode a contraction's row does not
depend on the row count (no split K), and this model's attention (a data mask, grouped query heads)
runs as separate score, softmax and weighted sum kernels, which reduce over keys in the same order for
a prefill row and a decode row. Where the fused attention kernel runs a decode with split KV (chunks
of 256 keys from 2048 keys in deterministic mode, docs/KERNEL_IR.md), the decode row is not the
prefill row's bits: `attention_case` in the script measures 3.05e-5 at 2200 keys in float16 (TS702, no
TS703: the decode is as accurate, 2.43e-4 against the prefill's 2.53e-4), and bitwise below 2048 keys.

## Limits

* No external engine's files have been checked yet; the file format is the interface for them.
* Float16 runs of the Llama model wait for mixed precision islands (laptop-fp16decode): without them
  the float16 program fails Truestar's own end to end check at long positions (float16 rotary
  angles). In float32 the same check failed once at position 2100 with a nearly empty cache
  (8.6e-4 against 6.3e-4): float32 rotary angles lose about 2.5e-4 rad there.
* The quantized steps need at least one full group in the prefix.
