# Range check for low precision targets (`truestar range`, pain P21 draft)

Author: Sharven Rane. Written 2026-11-24. Research: pain research round 3, item R3
(and R8 for the precision manifest at the end).

A model trained in float32 or bfloat16 is deployed where activations are stored in float16 (Apple
Neural Engine, WebGPU f16, TensorRT fp16 engines, phone NPUs). An intermediate passes 65,504 and the
output becomes wrong, often with no NaN: onnxruntime#31626 (whisper small: the `Pow` of a
decomposed layer norm reached about 630,000, 49 of 1500 positions lost their content, the output
had zero Inf and zero NaN), PR #32294 (bge small en v1.5: cosine to fp32 of 0.63 to 0.83 before the
fix), onnxruntime#26732 (Gemma 3 270M unusable in fp16; the Hub workaround is 18 hand inserted Clip
nodes at +-32752). Today people find this by exporting, deploying, seeing garbage, adding
intermediates as graph outputs and counting infinities by hand.

`truestar range` predicts it on the CPU before deployment, names the op and the source line with
the margin in bits, proposes a remedy and verifies the remedy by emulating the target.

## Use

```
truestar range model.py --target fp16
truestar range model.py --target webgpu-f16 --format agent      # JSON for coding agents
truestar range model.py --target ane --headroom 2 --json range.json
```

`model.py` defines `get_model()` and `get_inputs()` as for `truestar check`, and optionally
`get_calibration()` returning a list of more forward argument tuples (same shapes). Exit codes: 0
clean, 1 a finding (also when a remedy was verified), 2 usage error.

Targets:

| target | storage | composite ops | notes |
|---|---|---|---|
| `fp16` | float16 | fused, float32 internals | TensorRT fp16, Core ML GPU |
| `ane` | float16 | fused | NaN becomes +infinity (arXiv 2606.22283) |
| `webgpu-f16` | float16 | decomposed, every intermediate float16 | unfused ONNX graphs below opset 17 |
| `bf16` | bfloat16 | fused | |
| `fp8-e4m3`, `fp8-e5m2` | float8, unscaled | fused | no per tensor scale; scaled FP8 is not modelled |

Every preset accumulates matmul, conv and reductions in float32 (each op is computed exactly and its
output rounded once). Constants (weights, filled tensors) beyond the range saturate to the largest
finite value, as float16 converters do; computed values overflow to infinity.

```python
from truestar.rangecheck import range_check
rep = range_check(model, inputs, target="fp16")
print(rep.to_text())
rep.sites        # where values leave the range, with op, module, file, line, peak, margin, static bound
rep.chosen       # the remedy that passed verification
```

## How it works

1. **Import once** to the tensor IR (`torch.export`, docs/IR.md). Every node keeps its ATen op,
   module path and user source line.
2. **Observe in float64.** One reference run per calibration input records, for every value, the
   largest finite magnitude and how many nonzero values fall below the format's smallest normal
   value. Composite ops are written out once with a recorder (`rangecheck/composite.py`), so the
   intermediates the source never names are measured too: x minus mean, its square and the
   variance inside layer and group norm, x squared and its mean inside RMS norm, the attention
   scores and the sum of exponentials. Each node is also replayed alone with its inputs rounded to
   the format, which gives the number of correct bits that op keeps by itself (cancellation).
3. **Static bounds.** Interval rules over the IR (`rangecheck/interval.py`) bound every value for
   any input inside the box spanned by the calibration inputs. A value whose bound fits is
   proven safe for every such input, not only for the inputs that ran. The bounds are used one way
   only: they can prove safety, never raise an alarm. Normalizations reset them (a layer norm output
   is at most sqrt(N - 1) |w| + |b|), so transformers get tight bounds after every norm.
4. **Emulate the target and its twin.** The target run rounds every stored value to the format.
   The twin rounds significands only and keeps float64's exponent range. Both see the same
   rounding; they differ only by range. The target is judged against the twin, not against
   float64, so the format's ordinary rounding is never called a range problem. A run passes when it
   has no new non finite values, no decision flip beyond the twin's (a flip is real when the float64
   gap between the two classes is larger than twice the twin's error on that row) and an output
   error within 2x the twin's.
5. **Sites.** `overflow` (TS301): an op's output peaks above the maximum. `hidden_overflow` (TS302):
   an intermediate of a composite op that the target runs decomposed. `near_overflow` (TS303):
   within the headroom (default 1 bit) and its static bound does not keep it under the maximum.
   `underflow` (TS304): at least half the nonzero values below the smallest normal. `precision_loss`
   (TS305): the op alone keeps fewer than 2 correct bits. Sites whose inputs are all inside the range
   are causes (`origin`); values that stay out of range further down are listed as carried. If the
   target passes, overflow and underflow sites are moved to information (TS309, no effect).
6. **Remedies, verified** (`rangecheck/remedy.py`), tried in this order, each by rerunning the
   emulation on every calibration input:
   * `rescale`: an exact power of two reparameterization. The out of range values (and the
     homogeneous ops that carry them: adds, linear layers, reshapes) are carried scaled by 2^-k.
     Weights and edges into that region are scaled, norms reading it take eps * 2^-2k (layer, RMS
     and group norm are invariant otherwise), other consumers get the value back times 2^k. In
     binary floating point a power of two scale is exact, so the rewritten program's float64 output
     must equal the original's (checked at 1e-9, measured 0 to 1.4e-16). No float32 op on the
     device and no run time cost once folded into the weights.
   * `fp32`: keep each site and every consumer of a value that is still out of range in float32.
   * `clamp`: clip out of range outputs to +-max * 2^-headroom (the Gemma Hub fix).

## Measured (2026-11-24, CPU, torch 2.11, transformers 4.48)

Ground truth is never the emulator: `rangecheck/groundtruth.py` runs the model twice with PyTorch's
own kernels, once in float64 recording every ATen op, once as `model.to(float16)` where every op is
fed the float64 outputs of the ops before it (cast to the dtype it really receives). An op whose
output then has NaN or infinity where float64 is finite overflows on its own. Feeding correct inputs
is what makes every site visible: in a chained run only the first one is, since everything after it
is already NaN. Casts only one run performs (PyTorch upcasts RMS norm internals of a half input)
are skipped when aligning the two op sequences. For `webgpu-f16` the ground truth executes layer
norm as the ONNX decomposition in float16. Sites are matched by module path and op family; the
ground truth counted are those whose inputs were all in range (the causes).

Models (tests/range_models.py, synthetic weights, nothing downloaded):

* **bge small style**: transformers `BertModel` with bge small en v1.5's shape (hidden 384, 12
  layers, 12 heads, FFN 1536; CLS pooling, L2 normalize), trained looking norms, and a seeded
  outlier channel on the FFN outputs of layers 0, 3 and 4 scaled to the peaks #32294 measured on the
  real activations (|x - mean| 256 to 258, 304 to 321, 297 to 304). Measured here: 256.7, 315 and
  309.
* **Gemma 3 style**: written in plain PyTorch (embedding times sqrt(640), (1 + w) RMS norms before
  and after attention and the MLP, QK norm, one KV head, GeGLU with tanh gelu, tied head, 18
  layers). Two outlier channels of the residual stream grow by about 9,000 per layer from layer 3
  (a stable positive GeGLU feed and a post feedforward norm weight of 520), peaking at 1.08e5.
* Clean: BERT base shape (768, 12 layers) and torchvision ResNet50 with batch norm statistics set
  from a calibration batch (as docs/NPU.md does).

Rerun with `PYTHONPATH=src python scripts/range_study.py` (2 CPU threads, PC shared with other jobs).
"true sites" are the ground truth causes, "found" the predicted causes matched to them; the baseline
column is what people do today: run the model in float16 (PyTorch, decomposed layer norm for
`webgpu-f16`) and look at the output cosine to float64, which says something is wrong but not where.

| model | target | status | true sites | found | missed | false pos. | analysis s | baseline cosine (PyTorch chained) | emulated cosine | remedy | remedied cosine min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bge small style | webgpu-f16 | remedied | 3 | 3 | 0 | 0 | 2.3 | 0.1552 | 0.1552 | rescale | 0.999996 |
| bge small style | fp16 | ok | 0 | 0 | 0 | 0 | 1.7 | 1.0000 | 1.0000 | - | - |
| Gemma 3 style | fp16 | remedied | 1 | 1 | 0 | 0 | 6.9 | NaN output (-1 by convention) | NaN output | rescale | 1.000000 |
| Gemma 3 style | ane | remedied | 1 | 1 | 0 | 0 | 6.4 | NaN output | infinite output | rescale | 1.000000 |
| Gemma 3 style | bf16 | ok | 0 | 0 | 0 | 0 | 3.8 | 1.0000 | 1.0000 | - | - |
| Gemma 3 style | fp8-e5m2 | remedied | not measured | | | | 6.6 | not measured | NaN output | rescale | 0.981721 |
| BERT base shape (clean) | fp16 | ok | 0 | 0 | 0 | 0 | 3.2 | 1.0000 | 1.0000 | - | - |
| BERT base shape (clean) | webgpu-f16 | ok | 0 | 0 | 0 | 0 | 3.0 | 1.0000 | 1.0000 | - | - |
| ResNet50 (clean) | fp16 | ok | 0 | 0 | 0 | 0 | 3.0 | 0.9984 | 0.9983 | - | - |
| examples/range_model.py | fp16 | remedied | 1 | 1 | 0 | 0 | 0.1 (warm) | NaN output | NaN output | rescale | 1.000000 |

Ground truth runs took 0.03 to 2.9 s. The analysis times include the import (1 to 2.4 s cold).

Reading the table:

* **Every true cause found, no false positive, no false alarm on the clean models** (9 model and
  target pairs with a ground truth: 6 of 6 causes found, 0 extra; the 5 pairs without a cause report `ok`
  with no site).
* **The emulator agrees with PyTorch.** On bge small in `webgpu-f16` the emulated cosine is 0.155157
  and PyTorch's own chained float16 run with decomposed layer norms gives 0.155164. The failure is
  the one #31626 describes: no NaN and no infinity at the output, the infinite variance makes the
  division return 0 and the row collapses to the norm's bias. With fused norms (`fp16`) the same
  model is clean, as on the CPU and WASM paths in the issue; the overflow inside the fused norm is
  listed only as information (TS309).
* **Names the op, line and margin.** bge: `(x - mean)^2` inside
  `bert.encoder.layer.{0,3,4}.output.LayerNorm`, modeling_bert.py:554, margins -0.01, -0.60 and
  -0.54 bits. Gemma: the residual add of `layers.10`, `x = x + self.post_feedforward_layernorm(...)`
  in the model file, margin -0.13 bits; 6 earlier adds are near overflows (TS303, margin +0.06 to
  +0.54 bits) and 14 later residual values are carried out of range.
* **The rescale is exact and wins.** Gemma: residual stream carried at 2^-2 (23 values, 24 weights
  or edges scaled, 23 norms with eps * 2^-4), float64 rewrite error 1.4e-16, emulated float16 back
  to the twin (error 4.6e-4 vs the twin's 4.8e-4, 0 flips). bge: 3 values at 2^-1, rewrite error 0,
  cosine min 0.999996. The acceptance bar of the research draft (cosine at least 0.9999 under fp16
  emulation) is met on both. Keeping 30 ops in float32 also passes; it costs float32 ops on the
  device.
* **The Hub workaround fails verification here.** Clipping the 15 out of range outputs at +-32752
  (what the Gemma 3 Hub weights ship) gives cosine 0.984 and changes the argmax of all 32 positions
  on this model: the clipped residual channels carry the information. A clip is accepted only when
  verification passes (on examples/range_model.py it does).
* **Static bounds prove most values safe.** 273 of 316 float values in bge small, 883 of 944 in the
  Gemma style decoder and 76 of 87 in the example are proven to stay in range for any input in the
  calibration box; only 12 of 175 in ResNet50 (bounds through stacks of convolutions grow
  quickly). Soundness is tested: every observed float64 peak and hidden intermediate is inside its
  bound on four models (tests/test_rangecheck.py).
* **Other range failures:** logits near 1e-5 (the docs/NPU.md MobileNetV3 case) are found as an
  underflow at the head (TS304) and fixed by keeping that one op in float32; the one pass variance of
  examples/check_model.py is a precision loss at its subtraction, check_model.py:20 (TS305, 0.1
  correct bits), reported as a warning since the twin has it too.
* **bf16 and fp8.** The Gemma style model is clean in bfloat16. In unscaled float8 e5m2 the rescale
  passes against a twin with 3 significand bits; cosine 0.98 is the format's own floor, not a range
  effect.

## Precision manifest in `truestar check` (R8)

`truestar check` now prints, per backend, what each op class really computed in the model's dtype,
measured with crafted inputs on that backend instead of read from flags (`truestar/precision_probe.py`):

```
  precision: linear float32 (24 bits, subnormals kept); bmm float32 (24 bits, subnormals kept); conv2d float32 (24 bits, subnormals kept)
```

* Input bits: x = 1 + 2^-m against ones. A kernel that rounds inputs to q significand bits loses the
  2^-m term for m >= q, so the first lost m is the number of bits used: 24 float32, 11 TF32, 8
  bfloat16 passes.
* Accumulator (16 bit dtypes): 2^p followed by 2^(p-1) ones; summed in the dtype every 1 is lost.
  A blocked reduction in the dtype can hide this, so "wider than the dtype" means no narrow
  accumulation was seen, not a proof.
* Subnormals: half the smallest normal value times one.
* The flags meant to control it (`float32_matmul_precision`, TF32 and reduced precision reduction
  switches on CUDA) are listed next to the measurement in the JSON.

Measured on this CPU: float32 24 bits on all three; float16 11 bits and bfloat16 8 bits with wider
accumulation; **bfloat16 conv2d flushes subnormals on the CPU while linear and bmm keep them**
(oneDNN). The TF32 and float16 accumulation labels are tested with kernels that round their inputs
to 11 bits and sum in float16; on a real GPU they are not yet verified.

## Limits

* Ground truth is PyTorch's CPU kernels. No model was run on a Neural Engine, a WebGPU device or a
  TensorRT engine; the targets are emulations whose rules (storage, decomposition, NaN to infinity,
  constant saturation) are stated above and come from the cited reports.
* The two real cases are rebuilt with synthetic weights scaled to the published activation peaks,
  not the real checkpoints (no downloads). The bge peaks match #32294's measurements; the Gemma
  peaks follow the reported overflow, not a measured profile of the real 270M model.
* Float16 accumulation inside matmul and reductions is not modelled (every preset accumulates in
  float32). Scaled FP8 (per tensor or per block scales) is not modelled; R4 covers it.
* Attention scores and softmax sums overflowing in a decomposed graph are detected but a rescale
  cannot reach them; they fall back to float32.
* Calibration inputs must share their shapes. Sites are as good as the calibration set: a value the
  calibration never drives high is only covered by its static bound, which is loose after deep conv
  stacks.
* The ground truth matcher pairs sites by module path and op family; two sites of the same family in
  one module are paired by count.
* The precision manifest's TF32 and float16 accumulation labels were tested with simulated kernels;
  they have not been run on a GPU in this work (CPU only).
