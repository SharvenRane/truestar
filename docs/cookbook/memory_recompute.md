# Memory planning with recompute

Script: [examples/cookbook/memory_recompute.py](https://github.com/SharvenRane/truestar/blob/main/examples/cookbook/memory_recompute.py)

A compiled program knows every buffer it will allocate, so it can say how much memory a call needs
at a given batch size from shapes alone, before anything runs. With a budget, a training step
recomputes values its backward pass needs instead of storing them until the plan fits, and every
recompute keeps the bits ([MEMORY.md](../MEMORY.md)).

```python
from truestar.compiler import CompiledExecutor

ex = CompiledExecutor("cpu", "float32")
step = truestar.train_step(loss, optim.AdamW(lr=1e-3), executor=ex)
plan = ex.last.memory_plan({"B": 2048})          # no data: peak, the line at the peak, what is live
print(plan)

tight = CompiledExecutor("cpu", "float32", memory_budget=int(plan.total_bytes * 0.8),
                         memory_dims={"B": 2048}, recompute=True)
step = truestar.train_step(loss, optim.AdamW(lr=1e-3), executor=tight)
tight.last.recompute_report                      # each recomputed value, its bytes and source line
```

Output (a 4 layer MLP, batch 2048, float32):

```
memory plan: train_step_loss on cpu, float32, B=2048
  total 6.95 MiB: params and inputs 723.75 KiB, peak during the call 6.24 MiB
  without reuse (every value held to the end of the call) 10.01 MiB: 38% saved; in place steps 0
  peak at library %d_scatter_add: scatter_add (losses.py:37 in cross_entropy: nll = -ops.gather(logp, 1, ops.reshape(t, [n, 1])))
  live at the peak:
       1.00 MiB  %linear  layers.py:51 in forward: return ops.linear(x, p["w"], p["b"])
       1.00 MiB  %gelu  layers.py:221 in _act: return ops.gelu(x)
       ...

budget 5.56 MiB (80% of the plain step, 6.95 MiB)
planned with recompute: 4.95 MiB
  recomputed: %gelu (1.00 MiB, gelu) from layers.py:221 in _act: return ops.gelu(x), rerun before train_step_loss:%d_matmul_5
  recomputed: %gelu_2 (1.00 MiB, gelu) from layers.py:221 in _act: return ops.gelu(x), rerun before train_step_loss:%d_matmul_3

losses 2.3058 -> 2.2942; bitwise equal to the plain step: losses True, parameters True
```

A budget the plan cannot meet is refused before the first launch (TS420) with the peak's source
line, the largest values live there and the largest batch that fits. `recompute="mincut"`,
`"offload"` and `"mincut+offload"` choose what to recompute or move to host memory by a min cut
over the step ([MEMORY.md](../MEMORY.md), section on min cut and offload).

## Limits

* Pick the budget against `total_bytes` (inputs and parameters included). Recompute only shrinks
  what the backward pass stores; a step whose peak sits in the optimizer update (parameters and
  moments) cannot be helped by it.
* On the CPU the planner frees dead values but does not reuse buffers in place or take donated
  inputs; those run on CUDA ([MEMORY.md](../MEMORY.md) Limits).
* Index steps (gathers, from cross entropy) allocate range check temporaries the plan does not
  count; the plan prints a note when there are any.
