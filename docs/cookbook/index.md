# Cookbook

Short recipes, each one a script under `examples/cookbook/` that runs on a CPU in under a minute
and prints its own verdict. `tests/test_cookbook.py` runs every one of them with the GPU hidden.
Run any of them from a checkout:

```
python examples/cookbook/verify_float64.py
```

| recipe | question | Truestar API |
|---|---|---|
| [Verify a model against float64](verify_float64.md) | is my compiled model computing what it should | `@truestar.native`, `CompiledExecutor` |
| [Compare two chips](compare_chips.md) | what will this model give on a chip I do not own | `truestar.devsim.simulate("sim:...")` |
| [Deterministic training](deterministic_training.md) | does the same run give the same bits, and does a resumed run | `truestar.train_step`, `truestar.fit` |
| [Bitwise reproducible results across GPUs](cross_gpu_bitwise.md) | does another GPU give the same bits | `truestar fingerprint`, `truestar.fingerprint` |
| [Quantize safely](quantize_safely.md) | what do int8 and fp8 cost, and is the kernel right | `truestar.quant` |
| [Catch a runtime that lies](catch_lying_runtime.md) | does the shipped file compute the model in its runtime | `truestar.conform` |
| [Speculative decoding equal to greedy](speculative_decoding.md) | does speculation change the output | `truestar.nn` KV cache, `CompiledExecutor` |
| [Linear attention models](linear_attention.md) | Gated DeltaNet and Mamba 2 layers on a verified kernel | `truestar.linattn` |
| [Near tie diagnosis](near_tie.md) | is a changed prediction a bug or a tie | `truestar.devsim`, `truestar.quant.model.decisions` |
| [Memory planning with recompute](memory_recompute.md) | will this step fit, and how to make it fit | `memory_plan`, `memory_budget`, `recompute=True` |

Coming from PyTorch: [Switching to Truestar](../SWITCHING.md).
