# truestar.models: Hugging Face checkpoints built natively

Author: Sharven Rane. First written 2026-11-24. Code: `src/truestar/models/` (`config.py`,
`decoder.py`, `loader.py`, `runner.py`), the reader `src/truestar/interop/safetensors_np.py`, tests
`tests/test_models_native.py`, scripts `scripts/bench_models_native.py`,
`scripts/models_native_gpu.py`. Codes TS1390 to TS1397.

## Use

```python
import torch
from truestar import models

lm = models.from_pretrained("path/to/folder")    # config.json + model.safetensors, or an index and shards
logits = lm(ids)                                 # ids int64 [B, T]: a compiled native function
cache = lm.init_cache(B, capacity)
logits, cache = lm.step(ids, torch.arange(T), cache)     # prompt, then one token steps at their positions
out = lm.generate(ids, 16)                       # greedy through the cache

lm.model, lm.params                              # the truestar.nn layer and its parameter tree
lm.model(lm.params, ids, torch.arange(T))        # the same model eagerly, through PyTorch
```

`dtype=` picks the model precision ("f64", "f32", "bf16", "f16"; default: float64 for F64 files,
else float32). `executor=` picks where the programs run: the float64 reference interpreter by
default, `truestar.compiler.CompiledExecutor("cuda", "float32")` for Truestar kernels (the
executor checks its first run against the float64 reference itself).

What is not in the path: `torch.export`, the PyTorch exporter, the IR importer, and transformers
(the folder's `config.json` is read as JSON). The model is the `truestar.nn` layer
`models.DecoderLM`, whose `forward` is translated from source like any native function, so one
compile serves every batch size, prompt length, position and cache capacity.

**Local folders only.** A Hub id (`"meta-llama/Llama-3.2-1B"`) is refused with TS1397; nothing is
downloaded. Resolving a Hub id (for example through `huggingface_hub.snapshot_download`) is an
option left for the owner to approve.

## Families

| model_type | What the native decoder does differently |
|---|---|
| `llama` | rope scaling none, "linear" or "llama3"; `attention_bias`, `mlp_bias`; tied or untied head |
| `mistral` | sliding window over every layer when `sliding_window` is set |
| `qwen2` | q, k, v biases; tied head for the small models; `use_sliding_window` refused (TS1391) |
| `qwen3` | RMSNorm of q and k per head before the rotation; explicit `head_dim` |
| `phi3` | fused `qkv_proj` and `gate_up_proj` split by rows at load; `partial_rotary_factor`; sliding window; longrope refused |
| `gemma2` | `(1 + w)` RMSNorm in float32, embedding times sqrt(hidden) in the model dtype, sandwich norms, score scale `query_pre_attn_scalar ** -0.5`, attention and final logit capping `cap tanh(x / cap)`, sliding window on even layers (or `layer_types`) |
| `gemma3_text` | Gemma 2 without capping (unless set), plus q and k norms, sliding layers by `sliding_window_pattern` (or `layer_types`) with their own rope base `rope_local_base_freq` |

Not native yet: mixture of experts blocks (truestar.nn has none on main; Mixtral, Qwen3 MoE and
the like are refused with TS1390), multimodal Gemma 3 folders (TS1391), yarn, dynamic and longrope
scaling (TS1391), padding masks (one position vector for the batch).

Precision follows transformers: RMSNorm, the rotary angles and a capped attention's softmax run in
float32 for float32 and narrower models and cast back, as the Hugging Face modules do; in a
float64 model they run in float64. Rotary frequencies are computed with transformers' formula in
float32 arithmetic (the values a checkpoint was trained with).

## Loading checks

Before anything is returned, every stored tensor is matched to a leaf of the native parameter tree
and every problem is listed at once: TS1392 for a tensor the model needs and the folder lacks,
TS1393 for a stored tensor the model does not use (only `rotary_emb.inv_freq` and a tied
`lm_head.weight` are skipped, and named in `lm.info.ignored`), TS1394 for a shape that differs.
A sharded folder's index is checked as a whole first (TS1395): every shard exists, every name is
in the shard it is mapped to, no name is in two shards, no shard holds an unlisted tensor.
`lm.info` says what happened: tensors, bytes, shards, how many leaves are zero copy views of the
memory map and how many were converted (TS1396, info).

## The reader

`truestar.interop.safetensors_np` reads safetensors with NumPy only (a test runs it in a process
and checks torch was never imported). `open_checkpoint(path)` takes a folder, an index json or one
file; tensors are read only memory mapped, read on first touch. `t.raw()` is the stored bits,
`t.numpy()` the values (BF16 widened by a 16 bit shift, FP8 E4M3FN and E5M2 through 256 entry
tables, both exact; tested against torch's conversions for all 256 patterns), `t.torch()` a zero
copy torch view in the stored dtype. The header checks are those of the existing
`interop.safetensors` reader (duplicate names, byte lengths, overlaps, holes, buffer length).

## Verified (2026-11-24, torch 2.11, CPU)

`tests/test_models_native.py`, 24 tests (16 of them per family). For each family a tiny random model (3 layers, hidden 32,
4 query and 2 key/value heads, every weight including norms and biases perturbed so a swapped name
shows) is built from a transformers config in code, saved with `save_pretrained`, loaded natively
and compared with transformers in float64 with its float32 casts kept exact
(`ir.reference.exact_float32_casts`, the bar of `tests/test_hf_models.py`): the compiled native
function at two batch sizes and lengths (one compile), the eager layer, the prompt plus three
decode steps through the cache, and greedy generation against greedy recomputation. Bar 1e-10;
measured worst (`scripts/bench_models_native.py`):

| family | prefill | 4 decode steps | compiled CPU float32 vs transformers |
|---|---|---|---|
| llama (llama3 rope scaling, biases) | 1.2e-15 | 1.4e-15 | 3.0e-7 |
| llama tied (linear scaling) | 1.6e-15 | 1.6e-15 | 4.1e-7 |
| mistral (window 3) | 1.1e-15 | 1.4e-15 | 3.8e-7 |
| qwen2 | 1.8e-15 | 1.8e-15 | 3.7e-7 |
| qwen3 | 8.1e-16 | 1.3e-15 | 2.2e-7 |
| phi3 (window 4) | 1.3e-15 | 1.7e-15 | 3.1e-7 |
| gemma2 (window 3, both caps) | 1.9e-15 | 2.8e-15 | 1.6e-6 |
| gemma3 text (window 3, pattern 2, linear scaling) | 1.6e-15 | 1.7e-15 | 7.1e-7 |

The bar has teeth (tested): swapping one layer's q and o weights gives 1e-3 and above; moving the
rotary table by one float32 step fails 1e-10.

transformers versions: the installed 4.48.0 has no Qwen3 or Gemma 3, so those two tests skip
there (20 tests run); all 24 pass with transformers 4.53.3 (run from a `pip --target` folder, the
installed environment unchanged), the other families on both. On 4.53.3 the torch.export import
path (`ir.import_module`) refuses every one of these models (`lazy_load_decompositions`, 22 to 44
problems), while the native loader needs no change: that is the measured benefit of having no
exporter in the path.

On an L4 (Modal, `scripts/models_native_gpu.py`, no transformers in the image): a Llama 3 style
and a Gemma 2 style folder written from native weights, compiled to Truestar CUDA kernels; worst
error against the float64 reference: float32 1.6e-7 (llama) and 7.0e-7 (gemma2) for the prompt and
four decode steps, bfloat16 6.6e-3 and 2.5e-2 (weights rounded to bfloat16 included).
