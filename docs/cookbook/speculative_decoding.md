# Speculative decoding equal to greedy

Script: [examples/cookbook/speculative_decoding.py](https://github.com/SharvenRane/truestar/blob/main/examples/cookbook/speculative_decoding.py)

A small draft model proposes K tokens; the target scores all K in one call against its KV cache,
keeps the longest prefix it agrees with, and adds its own next token. The result equals plain
greedy decoding of the target only if the target's logits for a position are the same bits
whether that position was computed alone or inside a verify call of K tokens. The script checks
that instead of assuming it: it decodes both ways and compares every token and every logit row.

The model is a native `truestar.nn` decoder (token and position embeddings, transformer blocks
with a KV cache, a head). One native function serves the prompt, every one token step and every
verify call:

```python
@truestar.native(executor=CompiledExecutor("cpu", "float32", deterministic=True))
def step(p: m.spec, ids: Tensor["B", "T", "i64"], pos: Tensor["T", "i64"], cache: m.cache_spec()):
    logits, c2 = m(p, ids, pos, cache)
    return logits, ops.argmax(logits, dim=-1), c2

logits, nxt, cache = step(pt, ids, torch.arange(start, start + K), cache)   # verify K draft tokens
```

A rejected draft needs no cache rollback: attention reads only the slots at or before `pos`, and
the next call writes over the stale ones. Output:

```
greedy:      [9, 4, 0, 12, 29, 5, 31, 16, 1, 28, 5, 28, 5, 16, 0, 8, 5, 28, 5, 4, 5, 28, 9, 28]
speculative: [9, 4, 0, 12, 29, 5, 31, 16, 1, 28, 5, 28, 5, 16, 0, 8, 5, 28, 5, 4, 5, 28, 9, 28]
target calls: greedy 24, speculative 12 (12 of 44 draft tokens accepted)
tokens equal: True; logit rows bitwise equal: 24 of 24 (largest difference 0.0e+00)
OK: speculative decoding returned the greedy tokens bit for bit
```

## Why the executor matters

The compiled CPU executor computes each output element with its own reduction in a fixed order,
so a row's result does not depend on how many rows were in the call. On the float64 reference
executor (`@truestar.native` without an executor), which calls PyTorch's matmul, the same script
gave equal tokens but only 1 of 24 logit rows bitwise equal (differences up to 6.7e-16): equal
tokens there are luck at a near tie, not a guarantee.

## Limits

* Measured on this toy model on the CPU (24 tokens, K = 4, 2 blocks). On CUDA the attention and
  contraction kernels in deterministic mode keep a shape only config; whether a verify call of K
  rows and a one row call share it has not been measured here.
* The draft here shares the target's first block; acceptance (12 of 44) says nothing about real
  draft models. Speed is not measured.
* Sampling (temperature, top p) is not covered; greedy acceptance only.
