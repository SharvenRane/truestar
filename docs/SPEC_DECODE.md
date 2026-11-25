# Speculative decoding that never changes the output

`truestar.specdec` runs greedy speculative decoding: a small draft model proposes `k` tokens, the
target scores them in one call over `k + 1` query rows, and the round keeps the longest prefix the
target agrees with plus the target's own next token. In exact arithmetic that is the target's greedy
output. In floating point it is only when the target's logits for a row do not depend on how many
query rows the call has; on ordinary stacks they do, and the "lossless" output drifts (batched
implementations: arXiv 2510.22876; precision: arXiv 2609.15504; batch invariance as the prerequisite:
arXiv 2607.17283; verified speculation for determinism: LLM-42, arXiv 2601.17768).

```python
from truestar import specdec as SD
SD.check_rows(target, prompt, kc, vc, rows=k + 1)   # TS706 row invariant, TS705 not
toks, stats = SD.speculative(target, draft, prompt, steps, k, tk, tv, dk, dv)
assert torch.equal(toks, SD.greedy(target, prompt, steps, tk2, tv2))
```

`check_rows` feeds the same tokens as one call and one at a time from the same cache and compares
logits and caches bitwise. A Truestar program compiled in deterministic mode passes (no split K; each
row's reduction over keys runs the same way for one row and many, docs/PREFIX_CHECK.md). Eager
PyTorch does not, even in float64: about 1e-15 on CPU (`tests/test_specdec.py`).

| code | severity | meaning |
|---|---|---|
| TS706 | info | `spec_rows_invariant`: one row and the verify width give the same bits |
| TS705 | error | `spec_rows_variant`: they differ (first row and largest difference named); speculative decoding would change the tokens |

## Measured (RTX 5060 Laptop GPU, batch 1, deterministic float32)

A byte level target (8 layers, dim 384) and draft (2 layers, dim 128), trained in session on the
Python standard library's source (`scripts/specdec_pair.py`, about 4 minutes, rebuilt from its seed
anywhere), 8 held out prompts of 64 bytes, 128 new tokens (`scripts/bench_specdec.py`). Tokens were
bitwise equal to greedy decoding on every prompt at every `k`; row invariance TS706 at 3 to 7 rows.

| method | ordinary compiled calls (main) | CUDA graph replay (with laptop-fp16decode) | acceptance | tokens per target call |
|---|---|---|---|---|
| greedy (baseline) | 122 tokens/s | 702 tokens/s | | 1.00 |
| speculative k=2 | 151 (1.24x) | 926 (1.32x) | 0.74 | 2.46 |
| speculative k=3 | 160 (1.31x) | 987 (1.41x) | 0.66 | 2.96 |
| speculative k=4 | 159 (1.31x) | 973 (1.39x) | 0.60 | 3.36 |
| speculative k=6 | 152 (1.25x) | 1007 (1.43x) | 0.52 | 4.05 |

Main refuses a CUDA graph of the dense step (TS801, fixed on laptop-fp16decode); the graphed column
ran on main with that branch merged locally. The gain depends on the pair: a draft that agrees less
often gains less, and one that never agrees is slower than greedy (every round still pays `k` draft
calls). The tokens do not depend on the pair.

## Limits

* Greedy only. Speculative sampling keeps the target's distribution, not its samples: the accept test
  draws extra random numbers. A keyed sampler (one random number per position from a key) would make
  it equal to plain keyed sampling; not built.
* Batch 1: each sequence accepts its own number of tokens.
* No multi token prediction head drafting yet (no model here has one); a separate draft model only.
* The round loop reads the accepted count on the host once per round.
