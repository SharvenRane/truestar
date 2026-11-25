# Engines and other frameworks as port candidates

Author: Sharven Rane. First written 2026-11-24.

Pain row P32 (docs/QUALITY_BAR.md). Research items 11 and 12 of
<research>/GAP_DEVICES_FRAMEWORKS.md, with their redesign sections. Code in
`src/truestar/engines/`, tests in `tests/test_engine_port.py` and `tests/test_xport.py`, measurements
by `scripts/engine_port_study.py`.

`truestar port` (docs/PORT_CHECK.md) compares two PyTorch modules. Two more candidate kinds:

```
truestar port reference.py gguf:model.gguf                      # llama.cpp running a GGUF file
truestar port reference.py gguf:model.gguf --kv-type f16 --format agent
truestar port reference.py jax:candidate.py                     # a JAX / Flax NNX port
truestar port reference.py keras:candidate.py --keras-backends jax,torch
```

The reference file is the one `truestar port` takes (`get_model()`, `get_inputs()`), plus optionally
`get_prompt()` (the text the input ids came from, for the tokenizer check) and `get_config()` (a
`truestar.engines.llama_ref.LlamaConfig` or a dict of its fields, when the model has no Hugging Face
style `.config`). Exit codes as for `truestar port`: 0 same model, 1 a finding, 2 usage error.

## 1. llama.cpp (`gguf:`)

### The problem

People see gibberish from a GGUF model and cannot tell whether the engine is wrong on their backend,
the quantization lost too much, or a setting is off. llama.cpp's own tool, `test-backend-ops`,
compares one op at a time on synthetic shapes against the CPU backend, so it cannot test the CPU
backend itself, cannot see a wrong setting or a badly converted tensor, and does not say which layer
of the real model breaks. A "GPU corruption" report later traced to sampler settings
(llama.cpp#26754) shows why the verdict must separate engine, quantization and settings.

### The redesign: three verdicts that never mix

The engine runs one teacher forced pass of the reference's token ids (no sampling, so the verdict
cannot depend on sampler settings). Every node on the path is read through llama.cpp's scheduler
evaluation callback (`cb_eval`), the mechanism behind its eval callback example; ggml calls it per
graph node, so no bisection by truncating the graph is needed.

* **(a) Engine.** Each step (RMS norm, each projection, rotary positions on q and k, attention, the
  residual adds, SiLU gate, the head) is replayed by Truestar in float64 **on the engine's own input
  for that step**, with the GGUF's weights decoded exactly and the GGUF's settings. So a step's
  error is its own, never inherited, and quantization of the weights cannot show up as an engine
  error. The rounding the engine is allowed to do is modeled exactly, from an explicit list of
  paths: activations rounded to Q8_0 blocks of 32 for Q8_0 and Q4_0 weights, to Q8_K blocks of 256
  for K quants, f32 or f16 activations for F16 weights, and for an f16 KV cache K and V (optionally
  q and the probabilities) rounded to f16. Each listed path is tried, the best fitting one is
  reported per weight type (`engine.paths`), and anything beyond it is the engine's own error
  (TS640). Threshold: 10 times the float32 replay's own error of that step, never below 1e-6.
* **(b) Conversion.** Every setting is compared by name (rope theta, rope scaling, norm eps, head and
  KV head counts, rotary dims, sizes; TS641), and every GGUF tensor is matched to a reference
  parameter **by content**, under the transforms a converter can apply: identity, transpose, the
  per head row permutation the llama architecture needs for its adjacent pair rotary convention,
  and both. A tensor in another layout than the architecture needs (a transposed square weight, q
  or k rows not permuted) or with other values (a factor applied twice) is TS642, with the norm
  ratio. Then a float64 replay of the GGUF against the GGUF with the found faults undone locates
  the first step each fault changes.
* **(c) Quantization.** From float64 replays only: the reference against the same model with the
  GGUF's decoded weights and the modeled activation rounding. Top 1 agreement, KL, flips beyond near
  ties (TS645) or none (TS644). No engine output enters it, so an engine bug can neither inflate nor
  hide the quantization loss.

The settings the engine actually used are read from its graph: the RMS_NORM eps, the ROPE frequency
base, frequency scale, dims and mode, the SOFT_MAX scale. If they differ from the file, the engine
ignored the file (reported as `where: engine`), which is a different bug from a file holding the
wrong value. The tokenizer check (`get_prompt()`) tokenizes the prompt with the GGUF's vocabulary in
the engine and names a missing or extra BOS (TS646).

The final logits are decomposed: `total` (engine against the reference model) next to `engine`
(engine against the replay of the GGUF), `conversion` (GGUF against the GGUF with faults undone) and
`quantization` (that against the reference), plus `rounding_effect`, how far the allowed activation
rounding alone moves the logits. End to end, activation rounding is discontinuous (a value on a
rounding boundary rounds either way after float32 noise upstream), so only an engine drift beyond
that effect with no step to blame is TS657.

### Example (the tiny test model, a GGUF with rope theta 500000 instead of 10000)

```
truestar port: reference ref.py vs engine theta.gguf (llama)
engine: llama-cpp-python 0.3.38 on cpu, kv cache f32; GGUF tensors: 21 F32
final logits vs reference: 2.53e-01; engine vs GGUF replay 3.90e-07, conversion 2.53e-01, quantization 8.81e-16
decisions vs reference: top 1 agreement 68.8% (5 of 16 flipped, 5 beyond near ties), KL 3.01e-01
engine verdict: 33 steps replayed on the engine's own inputs, all within threshold (worst ratio 0.11)
setting: rope theta is 500000 in the GGUF, 10000 in the reference
quantization verdict (float64 replays only): top 1 agreement 100.0% (0 of 16 flipped, 0 beyond near ties), KL 0.00e+00
observer check: tapped vs untapped logits max difference 0.00e+00
status: DIVERGENT (blame: settings)

CONVERSION: rope theta is 500000 in the GGUF, 10000 in the reference. First step affected: 0.q_rope
(rotary positions on q), relative change 4.36e-01 in the float64 replay.
```

The same model quantized to Q4_K_M by the engine, no fault: `status: OK (quantization loss only)`,
engine against the GGUF replay 1.23e-07, quantization 2.61e-01 (top 1 agreement 81.2%).

An engine fault (one node perturbed through the test hook, Q4_K_M file):

```
ENGINE: node ffn_up-1 (MUL_MAT), the MLP up projection of layer 1, computes differently from the
float64 replay of that step on the engine's own inputs (own error 2.13e-03, threshold 2.0e-06).
```

### Measured (2026-11-24, `scripts/engine_port_study.py`, 5 seeds, CPU, llama-cpp-python 0.3.38)

Tiny Llama style decoder (2 layers, width 256, 4 query and 2 KV heads, 320 tokens, 16 teacher
forced positions), 5 random seeds, every case in F32, F16 weights, Q8_0 and Q4_K_M (quantized by the
engine's own tool). The baselines use the same runs: **final** compares the final logits with the
reference, threshold the largest clean error of that file type; **naive** compares each engine node
with the reference model's own activation at that node (accumulated error, what a hand written per
layer dump does), threshold the largest clean error at that node for that file type.

| | Truestar | final logits | naive per layer |
|---|---|---|---|
| clean ports (4 types, KV f32 and f16, 5 seeds) | **0 of 40** false alarms, worst own error 0.13 of its threshold | 0 by construction | 0 by construction |
| file faults: rope theta, norm eps, rope scaling, transposed square weight, q or k not permuted, scaled weight, no BOS (7 kinds, 4 types, 5 seeds) | **140 of 140** caught, 140 with the right blame (settings, weights, tokenizer), 0 blamed on the engine | 116 of 140 flagged, none named | not run (names a node, not a cause) |
| engine faults: one node (8 kinds: projections, rotary, attention, norm) perturbed by 1e-4 to 1e-2 of its max (3 per type and seed) | **60 of 60** caught at the right node | 19 of 60 | 33 of 60 |
| engine faults in F32 / F16 / Q8_0 / Q4_K_M files | 15 / 15 / 15 / 15 of 15 | 12 / 3 / 2 / 2 | 15 / 7 / 8 / 3 |

The baselines lose engine faults exactly where quantization loss is large: in a Q4_K_M file the
clean final logits already differ from the reference by up to 0.31 (top 1 agreement down to 44%),
so an engine bug of 1e-4 at one node is invisible there, while its own error is far over its
threshold. Without the rounding model (the replay with f32 activations only), clean files false
alarm: Q8_0 15 steps over threshold (own errors up to 9.1e-3), Q4_K_M 15 (1.6e-2), F16 2 (2.5e-4);
a threshold loose enough for those would hide every engine fault below 1e-2.

Clean final error against the reference, largest over seeds: F32 4.5e-7, F16 9.8e-4, Q8_0 3.1e-2
(top 1 at least 93.8%), Q4_K_M 0.31; the engine's own part stays at float32 level in every type.
Time per check: median 0.09 s, worst 1.9 s (the first call, library load), of which the engine
pass is 4 ms; the rest is the float64 replays. Tapped and untapped logits were bitwise equal in
every CPU run.

### What it reads and writes

GGUF is read and decoded by Truestar from the published format description (`engines/gguf.py`:
header, typed metadata, tensor infos, F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K blocks). The
decoders are bitwise equal to the `gguf` package's on every Q8_0, Q4_K, Q6_K and F16 tensor of the
test files (tests compare them; the package is not a dependency). Truestar never quantizes weights
for the engine: quantized files come from the engine's own `llama_model_quantize`. GGUF is not one of
Truestar's own formats.

## 2. Other frameworks (`jax:`, `keras:`)

### The problem

The Keras 3 FAQ promises identical numerics across backends up to 1e-7 per function; users find the
exceptions in production. Ports between frameworks (a PyTorch model rewritten in Flax, a Keras model
meant to run on any backend) are checked by comparing final outputs, which says only "differs".

### The redesign

The candidate file defines `get_model(weights_path)`: it receives the reference's state dict as a
safetensors file (written by `truestar.interop`) and converts it the way the port's converter does.
It runs in a child process, one per framework and per Keras backend (Keras fixes its backend at
import), in float64 (JAX x64, Keras floatx float64), on the CPU. Activations leave the framework
through DLPack (`truestar.interop.from_dlpack`, zero copy and checked) and cross the process boundary
in safetensors files. Then as in `truestar port`: weights matched by content (`portcheck.match_weights`;
Flax and Keras kernels are stored `[in, out]`, matched as transposes), modules mapped when their
matched weights cover each other, every mapped candidate module replayed alone in its framework on
the reference module's recorded inputs, threshold 10 times the reference module's own float32
error, and the first module over threshold whose mapped modules inside all agree is named. A 2 D
weight stored in another orientation than the port's other 2 D weights adds the hint "the converter
likely missed a transpose there". With several Keras backends, a backend that diverges while another
matches is TS656.

**Found on the way: Keras computes below float64 on its jax and torch backends.** With
`keras.config.set_floatx("float64")` and float64 kernels, `Dense` returns float32 on the torch
backend: `keras/src/backend/common/dtypes.py` maps every 64 bit result type to 32 bit except on
TensorFlow ("aligned with JAX's default behavior"), and the torch `matmul` casts its result to that
type. On the jax backend the dtypes stay float64 but the results differ from float64 at 1e-7. The
check reports it (TS658, the modules named) and uses float32 noise thresholds there instead of
calling a correct port wrong.

### Measured

Same reference, 2 seeds. Clean ports: the Flax NNX port and the Keras port on its jax and torch
backends. Faults: in Flax a hardcoded eps, rope theta 500000, rotary pairs taken adjacent, a square
kernel loaded without its transpose; in Keras a hardcoded eps (both backends), an attention scale
applied twice on the torch backend only, a kernel without its transpose (both backends).

| | result |
|---|---|
| clean runs (Flax, Keras jax, Keras torch, and the Keras jax side of the one backend bug) | **0 of 8** false alarms |
| faults, counted per backend | **18 of 18** caught, 18 named at the right module |
| the one backend bug | keras:jax clean, keras:torch named at `blocks.0.attn`, TS656 |
| transposed kernel | named at the dense layer with the hint "missed a transpose" |
| final output alone | flags every fault as "differs" and names nothing |

Clean final error: Flax 2.1e-15 (float64 throughout), Keras 9.9e-8 on jax and 6.8e-7 on torch (the
float32 internals above, TS658). Time: median 6.1 s for a Flax check, 14.6 s for Keras with both
backends, almost all of it starting the framework processes; the isolated replay of 28 module pairs
takes 0.05 to 0.07 s per backend. The guard test shows why the replay is isolated: on the
candidate's own inputs, an untransposed kernel in layer 0 makes at least 10 later modules look
wrong; on the reference's inputs only that layer (and modules containing it) is over threshold.

## Limits (2026-11-24)

* llama.cpp: only the CPU build of llama-cpp-python 0.3.38 was run. CUDA and Vulkan builds were not
  installed (no prebuilt wheel with sm_120 kernels was tried), so the per backend use from the
  research item (CPU against Vulkan on the same GGUF) is not verified; the tap reader works through
  `ggml_backend_tensor_get`, which reads any backend, but GPU fusion with an eval callback is not
  measured. The injected test hook writes into host memory only.
* Only the llama architecture is replayed (RMSNorm, rotary positions, grouped query attention,
  SwiGLU). Other architectures are refused (TS650). Flash attention is off in the tapped engine
  run (the scores must be separate graph nodes); the verdict for a flash attention engine is not
  measured.
* Engine faults were injected through the eval callback (a perturbed node), not real llama.cpp
  bugs; a real before and after llama.cpp commit pair was not run.
* Graph node names are those of this llama.cpp build (`attn_norm-0`, `Qcur-0`, `kqv_out-0`, ...);
  another naming is refused (TS648), never guessed.
* Identical tensors (all ones norms) pair in order; the test models use random norms.
* Frameworks: Flax NNX 0.10.7 on JAX 0.6.2 and Keras 3.12.4 on its jax and torch backends, CPU only.
  TensorFlow and MLX were not installed (MLX needs macOS). Inline code without weights (a rotary
  function inside an attention module) is blamed at the enclosing module; there is no op level
  drilldown inside another framework. Only a Llama style decoder was ported; the Keras pooling
  caveat and Keras issues #24170 and #24248 from the research item were not reproduced.
* The tiny models are built in code (2 layers, width 256, 320 tokens); no published GGUF or
  checkpoint was run.
