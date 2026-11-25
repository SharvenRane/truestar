# Linear attention models

Script: [examples/cookbook/linear_attention.py](https://github.com/SharvenRane/truestar/blob/main/examples/cookbook/linear_attention.py)

Gated DeltaNet (Qwen3 Next, Qwen3.5), Mamba 2 SSD, Kimi Delta Attention and lightning attention
replace softmax attention with a recurrence over a matrix state. `truestar.linattn` runs the
chunked form of that recurrence as Truestar's own kernels (Triton on CUDA, chunked PyTorch
elsewhere) and checks the first call per signature against the float64 sequential recurrence
([LINEAR_ATTENTION.md](../LINEAR_ATTENTION.md)).

```python
from truestar.linattn import chain_error, chunk_invariance, last_call, linear_attention, recurrent

o = linear_attention(q, k, v, decay, beta, rule="delta")                     # Gated DeltaNet
o, state = linear_attention(q, k, v, decay, rule="add", output_final_state=True)   # Mamba 2 SSD
last_call()                                     # impl, config, checked against float64 on this call
chunk_invariance(q, k, v, decay, beta, rule="delta", chunks=(16, 32, 64))
chain_error(q, k, v, decay, beta, rule="delta", chunk=16)
```

Shapes follow the models' code: q, k `[B, T, H, dk]`, v `[B, T, H, dv]`, a log decay `[B, T, H]`
(scalar per head) or `[B, T, H, dk]` (per key channel, Kimi), beta `[B, T, H]`. Output:

```
Gated DeltaNet (delta rule, scalar decay)
  ran chunked with {'CHUNK': 16}, checked against float64 on this call: True
  relative error vs the float64 recurrence: 1.4e-07; final state (2, 4, 32, 32)
Mamba 2 SSD (add rule, scalar decay)
  ran chunked with {'CHUNK': 16}, checked against float64 on this call: True
  relative error vs the float64 recurrence: 1.3e-07; final state (2, 4, 32, 32)

chunk size as a reduction order (delta rule, float32):
  chunk  16: worst element error vs float64 5.0e-07
  chunk  32: worst element error vs float64 6.8e-07
  chunk  64: worst element error vs float64 6.2e-07
  chunks 16-32: bitwise equal False, 72% of elements differ, largest difference 3.0e-07
  chunks 16-64: bitwise equal False, 78% of elements differ, largest difference 3.0e-07
  chunks 32-64: bitwise equal False, 67% of elements differ, largest difference 3.6e-07
error along 16 chunks of 16: last quarter / first quarter = 1.03 (no growth warning)
```

The chunked form is exact in real arithmetic for any chunk size, so the chunk size is a reduction
order: in float32 every size is within 7e-7 of float64 here, and no two sizes give the same bits.
Fix the chunk (`chunk=64`) when you need the same bits as another run. `chain_error` reports
whether the error grows along the chain of chunks (TS1062 when the last quarter is well above the
first).

## Limits

* A torch level op, not yet a `truestar.nn` layer, and the importer refuses Hugging Face linear
  attention and SSM layers by name (TS941, TS942) instead of lowering them
  ([MODEL_COVERAGE.md](../MODEL_COVERAGE.md)).
* The delta rule assumes unit norm keys, as the models normalize them.
