# LoRA and QLoRA fine tuning

Author: Sharven Rane. First written 2026-11-24. Codes TS1370 to TS1377.
Code: `src/truestar/lora/__init__.py`, the Llama decoder and Hugging Face importer
`src/truestar/nn/llama.py`; tests `tests/test_lora.py`; benchmark `scripts/bench_lora.py`.
Built on truestar.nn (docs/NN.md), truestar.optim and `train_step` (docs/OPTIM.md) and the weight
formats of docs/QUANT.md.

## Use

```python
import truestar
from truestar import Tensor, lora, nn, ops, optim
from truestar.nn.llama import Llama, from_hf

net, p = from_hf(hf_model)                 # a transformers LlamaForCausalLM; or net = Llama(...), p = net.init(0)
lm = lora.wrap(net, targets=("q_proj", "v_proj"), rank=8, alpha=16, dropout=0.05)
# QLoRA: lora.wrap(net, ..., quant="int4:g64")  (int8, int4:g32/64/128, nvfp4, mxfp4)
base = lm.prepare_base(p)                  # frozen base tree (quantized with quant=)
a = lm.init(seed=0)                        # A keyed uniform, B zero: starts equal to the base model

@truestar.native
def loss(a: lm.spec, p: lm.base_spec, ids: Tensor["B", "T", "i64"], y: Tensor["B", "T", "i64"],
         training: bool = False) -> Tensor[()]:
    logits = lm(p, a, ids, training=training)
    return nn.cross_entropy(ops.reshape(logits, [-1, net.vocab]), ops.reshape(y, [-1]))

step = truestar.train_step(loss, optim.AdamW(lr=1e-3), params="a", static={"training": True},
                           executor="cuda")
ts = step.init(a)
ts, r = step(ts, base, ids, y)             # the base is an input, never differentiated or updated

merged = lm.merge(base, ts.params)         # plain weights for `net` (quantized weights decoded)
lora.verify_merge(lm, base, ts.params, ids)          # float64 check, TS1374 on failure
lora.save_adapters("run.lora.safetensors", lm, ts.params)
a = lora.load_adapters("run.lora.safetensors", lm)   # TS1373 for another config or base layout
```

## How it is built

* **Targets.** `wrap` walks the layer config tree and replaces every `nn.Linear` whose dotted path
  (`l0.self_attn.q_proj`) or last name matches a shell pattern (`q_proj`, `l1.*`, `*_proj`) with a
  `LoRALinear`. With `quant`, every other Linear except `quant_skip` (default `lm_head`) becomes a
  `LoRALinear` of rank 0: frozen, base weight in the format. The original config is not changed
  (tested); no match is TS1371 with the list of Linear paths.
* **Two trees.** The model is called as `lm(p, a, x)`. `merge_trees(p, a)` joins the base and
  adapter trees at compile time. It is a new kind of front end call, a structure function
  (`__truestar_structure__`): it runs as Python on the traced values and emits no op. So
  `train_step(params="a")` differentiates and updates the adapter leaves only, the optimizer slots
  and checkpoints hold adapters only, and the base tree is an ordinary input. The LoRA step program
  of the benchmark's small model has 785 nodes against 1182 for full fine tuning: the backward
  computes no frozen weight gradient.
* **The layer.** `y = x W^T + b + (alpha / r) dropout(x) A^T B^T`, PEFT's formula and initial
  values (A uniform on +-1/sqrt(d_in), drawn at the key `fold(seed, path.lora_a)`, B zero).
  Dropout is the keyed `ops.dropout`. Its mask depends on the step's key (`call_key(seed, step)`)
  and the call site only, so it is the same on every chip and after a resume. Training and
  evaluation are two configs (`lm.train_model`, `lm.eval_model`). The evaluation program has no
  random op, and the training program has exactly one per adapter (tested by counting).
* **QLoRA base weights** are the codes and scales of `truestar.quant.formats.quantize_weight`
  (docs/QUANT.md), decoded inside the program with ops that already have kernels: a cast of the
  packed bytes to int64, an `index_select` into a 256 entry table per nibble (the signed int4 value,
  or the E2M1 value), a second table for E4M3 or E8M0 block scales, a broadcast multiply. No bit
  operations are needed, and no new IR op. `verify_decode` checks that this decoder, run in
  float64, is bit identical to `QuantWeight.dequantize` for every quantized layer (TS1375). A test
  swaps the high and low nibble tables and is refused.
* **Merge** computes `W + (alpha / r) B A` in float64 from the decoded base weight and rounds once
  to the layer dtype. `verify_merge` runs the adapter model and the merged plain model in float64
  and refuses a relative difference above 1e-12 (TS1374). A test doubling one layer's alpha is refused.
* **Hugging Face import.** `truestar.nn.llama.from_hf` copies a `LlamaForCausalLM`'s weights into
  a truestar.nn Llama tree that keeps Hugging Face's module names, so the usual target names
  apply. It refuses rope scaling, attention or MLP biases, a non SiLU activation and a head dim
  other than hidden / heads. This is a one way importer. The model then trains natively.

## Verified (2026-11-24, torch 2.11, transformers 4.48, CPU, float64)

`tests/test_lora.py`, 18 tests:

| Check | Result |
|---|---|
| `from_hf` forward vs `LlamaForCausalLM` (untied and tied head), native and eager | below 1e-12 relative. transformers computes RMSNorm and the rotary angles in float32 even for a float64 model, so the test replaces those two modules' forwards with the same formula in float64; with them as shipped the difference is 8e-8 |
| LoRA loss and every adapter gradient (q, v and down projections, B random) vs a hand written torch LoRA module on the Hugging Face model, float64 autograd | loss 1e-13, gradients below 1e-11 relative |
| QLoRA, int8, int4:g32, int4:g64, nvfp4, mxfp4: loss and adapter gradients vs float64 autograd of the Hugging Face model holding the exactly dequantized weights | loss 1e-13, gradients below 1e-11 relative. The decoder is bit identical to the format for all 14 quantized layers, and the base tree is under half the float tree's bytes |
| Merged forward vs adapter forward, float64 (plain, int4:g32, nvfp4 base) | below 1e-13 relative. The merged weight equals `W + 2 B A` bit for bit |
| Training: 12 AdamW steps with dropout on the reference executor | loss below 0.8 of the first, two runs bitwise equal (adapters and loss history), base untouched, only adapter leaves carried |
| Adapter file | round trip bitwise, under 20 kB against a 0.3 MB base, TS1373 for another rank or base layout, TS1372 for a missing adapter |

The tests can fail. Removing the `alpha / r` scale fails 9 of 18. Swapping the FP4 nibble tables
fails 3 (both FP4 decode checks and the nvfp4 merge).

## Measured

`scripts/bench_lora.py`: a Llama style model with random weights from a config learns next token
= a fixed random permutation of the current token, AdamW lr 3e-3, the same batches for every
configuration. "carried" counts the bytes a step holds: trained tensors, their gradients, the two
AdamW slots and the frozen base.

### CPU (Windows desktop, shared, 12 steps, median after 3): vocab 128, dim 128, 2 blocks, 4 heads / 2 KV, ffn 256, batch 8 x 32; 328k params

| Configuration | loss first to last | step ms | first call s | carried bytes |
|---|---|---|---|---|
| truestar LoRA r8 q,v (compiled CPU) | 5.042 to 4.704 | 28.1 | 44 | 1.43 MB (adapters 28.7 kB) |
| truestar QLoRA int4:g64 (compiled CPU) | 5.046 to 4.702 | 38.7 | 6.9 | 0.43 MB |
| truestar full fine tuning (compiled CPU) | 5.042 to 1.931 | 45.7 | 18.6 | 5.25 MB |
| eager PyTorch full fine tuning | 5.042 to 1.931 | 8.1 | | 5.25 MB |
| hand written LoRA, eager PyTorch | 5.042 to 4.701 | 4.7 | | 1.43 MB |

Two truestar LoRA runs from the same seed: adapters bit identical, loss histories equal. Truestar
full fine tuning follows eager PyTorch to 1.4e-6 on the last loss after 12 steps in float32 (the
same weights and batches), which is an independent check of the whole compiled step. LoRA's loss
falls more slowly than full fine tuning's, as expected for a rank 8 adapter on two projections of
a randomly initialized model. The LoRA and hand written LoRA losses agree to 3e-3 after 12 steps;
their dropout masks differ (keyed against torch's generator).

### Modal L4 (40 steps, median after 8): vocab 512, dim 512, 4 blocks, 8 heads / 4 KV, ffn 1024, batch 16 x 64; 9.97M params

"steady peak" is the device memory allocated during steps 2 on, beyond what was allocated before step 2
(torch.cuda.max_memory_allocated). The carried state (params, slots) is held before the step and is in the
"carried" column, not in that delta. "first call peak" includes the compile and the first call's float64 check.

| Configuration | loss first to last | step ms | first call s | carried | steady peak | first call peak |
|---|---|---|---|---|---|---|
| truestar LoRA r8 q,v (compiled CUDA) | 6.386 to 5.207 | 39.4 (60.4 in an earlier run) | 76 | 40.8 MB (adapters 0.23 MB, AdamW 0.46 MB) | 138 MB | 455 MB |
| truestar QLoRA int4:g64 (compiled CUDA) | 6.389 to 5.212 | 56.9 (87.2) | 33 | 8.1 MB (base 7.2 MB against 39.9 MB) | 288 MB | 423 MB |
| truestar full fine tuning (compiled CUDA) | refused, see below | | | 159.5 MB | | |
| eager PyTorch full fine tuning | 6.386 to 1.855 | 14.2 (21.2) | 0.1 | 159.5 MB | 174 MB | 224 MB |
| hand written LoRA, eager PyTorch | 6.386 to 5.226 | 12.3 (18.9) | 0.03 | 40.8 MB | 131 MB | 172 MB |

* **Loss goes down** in every configuration that ran. LoRA's first loss equals eager PyTorch's to 8e-8 relative, the
  same model and weights through a separate implementation.
* **Bitwise determinism**: two truestar LoRA runs from the same seed (dropout on) gave bit identical adapters
  and equal loss histories on the L4, as on the CPU.
* **Adapter only memory**: the trained state (adapters, gradients, AdamW slots) is 0.92 MB against 159.5 MB
  for full fine tuning (174x less). The whole carried state is 40.8 MB with a float32 base and 8.1 MB with
  an int4:g64 base (QLoRA, 19.7x less than full fine tuning).
* **QLoRA's working memory is larger**: 288 MB steady against 138 MB. Each step decodes every quantized
  weight into a float temporary, and the decoder casts the packed bytes to int64 for `index_select`
  (8 bytes per two weights), so the decode temporaries are bigger than the float weights they produce.
  A fused decode and matmul kernel would remove both. Not built.
* **Speed**: on the L4 the compiled truestar LoRA step is 3.2x slower than the hand written eager PyTorch
  LoRA (39.4 against 12.3 ms), and QLoRA 4.6x. The step times moved by up to 1.5x between two runs on Modal,
  so they are preliminary.
* **AdamW's first update is ill conditioned in float32.** The LoRA step was refused by the compiled
  executor's end to end check on its first call (two outputs 2.3x and 5.6x over the float64 bar, where an
  independent float32 evaluation is itself 1.7x and 3.3x over). The benchmark runs the documented
  `accept_ill_conditioned=True` (TS782). Each such output is listed and warned about, and every other output
  keeps its bar. Not needed on the CPU.
* **Found, not fixed: the compiled full fine tuning step on CUDA is refused** by the same check:
  output `%sub_48` 4.1e-3 to 4.5e-3 relative against a bar of 1.9e-3, while the independent float32
  evaluation is 9.2e-4 off, within the bar. So this is not ill conditioning; the compiled CUDA program
  is less accurate than float32 should be on one parameter update. The same step passes on the CPU,
  where it follows eager PyTorch to 1.4e-6. Not isolated (docs/OPTIM.md measured the CUDA steps of
  smaller models). Owner decision: compiler follow up.


## Limits (2026-11-24)

* Speed: the compiled truestar step is slower than eager PyTorch on these small models (the
  compiler's kernel count and launch overhead, docs/OPTIM.md). LoRA is faster than full fine tuning
  inside truestar because the frozen weight gradients are not in the program.
* QLoRA decodes every quantized weight on every step into a float temporary of the full weight
  (the memory of the stored base shrinks, the per layer peak does not). A fused decode and matmul
  kernel is not built.
* `from_hf` covers Llama style decoders without rope scaling or biases. Imported IR programs
  (`ir.import_module`, docs/TRAINING.md) are not rewritten with adapters. The native model is the
  path.
* Adapter dropout follows PEFT (on the adapter input only). No DoRA, rsLoRA scaling, per layer
  rank or alpha, or embedding adapters yet. `merge` returns float weights and does not requantize.
* The eager call `lm(p, a, x, training=True)` with dropout needs a key, like every eager random
  layer. Use the native function or `training=False`.
