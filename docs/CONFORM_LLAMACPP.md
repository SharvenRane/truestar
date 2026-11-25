# llama.cpp conformance (`truestar conform llamacpp`)

Author: Sharven Rane. First written 2026-11-24.

Pain row P32 (docs/QUALITY_BAR.md), the engine half. `truestar port reference.py gguf:model.gguf`
(docs/PORT_ENGINES.md) taps every graph node of llama-cpp-python's bundled engine. This command
answers the other question users ask: **does the llama.cpp build I actually run compute the model,
in every format and with every setting I use?** It runs an official llama.cpp build unmodified and
judges its logits against Truestar's float64 replay. Truestar is the reference here, never a plugin:
the feature exposes what the engine computes.

Code: `src/truestar/conform/llamacpp.py`, the driver `src/truestar/conform/llamacpp_driver.cpp`.
Tests: `tests/test_conform_llamacpp.py` (all but one run without llama.cpp). Measurements:
`scripts/llamacpp_conform_study.py`.

## Use

```
truestar conform llamacpp model.gguf --llama-cpp ~/llama.cpp
truestar conform llamacpp model.gguf --llama-cpp ~/llama.cpp --source hf_model_dir --format agent
truestar conform llamacpp hf_model_dir --llama-cpp ~/llama.cpp --formats f32,f16,q8_0,q4_k_m
truestar conform llamacpp model.gguf --threads 1,8 --batch 512,1 --flash off,on --kv f16
```

`--llama-cpp DIR` (or `TRUESTAR_LLAMA_CPP`) is a llama.cpp source tree built with cmake (shared
libraries, the default), or an install prefix with `lib/` and `include/`. A Hugging Face directory
(`config.json`, `*.safetensors`, a tokenizer the converter accepts) is converted by llama.cpp's own
`convert_hf_to_gguf.py` (f32, f16, q8_0) and `llama-quantize` (Q4_K_M, from the f32 file); Truestar
never quantizes a weight for the engine. Default settings: threads 1 and 4, batch (and ubatch) 512
and 7, flash attention off and on, KV cache f16 (llama.cpp's default): 8 runs per format. Default
prompts: 3 fixed token id sequences of 24 tokens (seed 1310, the first starting with the GGUF's BOS),
each teacher forced and then decoded greedily for 16 steps. Exit codes: 0 conforms, 1 a finding
(TS1311, TS1312, TS1316), 2 the check cannot run (TS1310, TS1319, bad arguments).

## How the engine is run

Truestar compiles its own small C++ driver against the `llama.h` and `ggml.h` of the build under test
and links it with that build's `libllama`, so every struct layout is the one of that exact version
(the context parameters change between releases; a ctypes binding would have to guess them). The
driver loads the GGUF, decodes the token ids with `llama_decode` in chunks of the requested batch,
and writes the float32 logits of every position (teacher forced) and of every greedy step (argmax,
lowest index on ties, then the chosen token is decoded alone). It sets threads, batch, ubatch, flash
attention (`LLAMA_FLASH_ATTN_TYPE_ENABLED` or `DISABLED`, never auto) and the KV cache type, and
nothing else. The binary is cached by the hash of its source, the headers and `libllama`. The
version recorded in every report: `git describe` of the tree when it is a checkout, a
`.truestar-source` note next to a tarball build, ggml's version, the system info line, and the
sha256 of `libllama`.

## The three verdicts

1. **Engine** (always). The GGUF's weights are decoded exactly to float64 (`engines/gguf.py`) and the
   model is replayed in float64 on the same tokens. The engine's logits are compared with that
   replay per position (relative L2 error). Quantization cannot appear in this verdict: both sides
   hold the same quantized weights. The tolerance per position is derived from two float64 side
   runs, not chosen: `tau = 10 x max(float32 replay error, rounding effect)`, never below 1e-6, where
   the rounding effect is the replay with the rounding llama.cpp's CPU backend is allowed to do:
   each matmul input rounded to its weight's dot product type (Q8_0 blocks for Q8_0 and Q4_0, Q8_K
   blocks for K quants, f16 for f16 weights), and with an f16 KV cache q, k, v and the attention
   probabilities rounded to f16 (the operands of the two attention products). A position over `tau`
   is TS1311.
2. **Greedy decoding.** The engine's own greedy tokens are replayed teacher forced, so each step is
   compared on the prefix the engine actually had. The first step where the engine's token differs
   from the replay's argmax is reported. It is a bug (TS1312) unless the replay's top two logits
   there are within twice the per element bound (10 times the larger per element change of the two
   side runs): then it is a near tie, ambiguous (TS1313), not a bug.
3. **Conversion and quantization** (with a source: a model directory, or `--source` for a GGUF).
   Conversion: every setting by name (sizes, head counts, eps, rope theta and scaling) and every
   tensor by content, under the layouts a converter can apply (`engine_port.match_weights`); a
   difference is TS1316. Quantization: float64 replays only, the GGUF's decoded weights against
   the source weights: relative error, top 1 agreement, flips beyond near ties (top two closer than
   1% of the logit standard deviation, as `truestar port` counts them), and the source's greedy
   continuation checked against the file. Always TS1314 (the numbers; when a conversion fault was found the
   message says the difference includes it); TS1315 when quantization
   alone changes decisions. No engine output enters this verdict.

Then the settings: the logits of every run are hashed. Runs with different bits are TS1318 when the
greedy tokens stay the same, TS1317 when the tokens change, with each change at a near tie or not.

## Codes

| code | severity | meaning |
|---|---|---|
| TS1310 | error, exit 2 | no usable llama.cpp build (directory, libllama, headers, a C++ compiler, or the engine run failed); the reason is named |
| TS1311 | error | engine logits beyond the derived tolerance of the file's own float64 replay |
| TS1312 | error | a greedy token differs from the replay where the top two are not a near tie |
| TS1313 | info | a greedy token differs only at a near tie: ambiguous |
| TS1314 | info | quantization alone, in float64 (always reported with a source) |
| TS1315 | warning | quantization alone changes decisions |
| TS1316 | error | the GGUF does not hold the source model (setting or tensor) |
| TS1317 | warning | greedy tokens change with engine settings |
| TS1318 | info | logits bits change with engine settings, tokens do not |
| TS1319 | error, exit 2 | model not covered (architecture, tensor type), malformed file, or llama.cpp's converter or quantizer failed |

## Measured (2026-11-24, `scripts/llamacpp_conform_study.py`)

Engine: llama.cpp v1.0.0 built from the official release source tarball
(`archive/refs/tags/v1.0.0.tar.gz`, sha256 `3f5f750e…532b`, release commit `f1f9a08f`), cmake
Release, CPU only (GGML_NATIVE), g++ 13.3, on a rented Linux server (16 cores, at most 4 threads
used). Converter: that release's `convert_hf_to_gguf.py` under transformers 5.24 and torch 2.11
(its requirements file pins transformers 4.57.6; the Llama and Qwen2 paths ran unchanged).

Models, all built in code: a sentencepiece unigram tokenizer (320 pieces) trained on generated
words, then a Llama (untied head) and a Qwen2 (q, k, v biases, tied head) decoder from
transformers with random weights and random norms (2 layers, width 256, 4 query and 2 KV heads,
MLP 512, vocab 320), seeds 0 and 1. Every model converted to f32, f16, q8_0 and Q4_K_M (Q4_K_M:
13 Q4_K, 3 Q6_K, 5 F32 tensors), every file run with the 8 default settings, 3 prompts of 24
tokens and 16 greedy steps each.

| | result |
|---|---|
| clean runs (2 architectures x 2 seeds x 4 formats x 8 settings = 128 runs, 9,216 teacher forced positions) | **0** positions over tolerance, worst error 0.137 of its tolerance; **0** TS1311, **0** TS1312 |
| greedy steps that differ from the float64 replay in clean runs | 72, all near ties (TS1313); none in f32 or f16 files |
| engine's largest per element deviation, in units of the modeled rounding effect | 1.73 (f32 1.41, f16 1.11, q8_0 1.73, Q4_K_M 1.61): the 10x bound holds with margin, a 2x bound would not |
| planted engine fault: the engine told to use rope base 500000 (and 10100) over the file's 10000 | **4 of 4** models TS1311 and TS1312, worst error 204 to 233 times the tolerance |
| planted conversion faults: GGUF converted from a config with rope theta 500000, or from a checkpoint with `layers.1.mlp.up_proj` scaled 1.5x, each in f32 and q8_0 | **16 of 16** TS1316 with the setting or the tensor named (norm ratio 1.5), **0** blamed on the engine (worst engine error 0.14 of its tolerance) |

Quantization alone, float64 replays, largest over the 4 models: f32 9.6e-11 (the GGUF stores eps and
theta in float32), f16 1.9e-3 (top 1 agreement at least 98.6%), q8_0 6.0e-2 (at least 91.7%, up to
5 flips beyond near ties of 72), Q4_K_M 0.61 (top 1 agreement 44% to 51%, 30 to 39 flips beyond near
ties, every greedy continuation changed). The engine's own part stays at the level its rounding
explains in every format: the total error against the source (Q4_K_M up to 0.61) is almost all
quantization.

**Does llama.cpp's answer change with settings?** On this build, on the CPU:

* Threads (1 or 4) never changed a bit, in any format.
* Flash attention on against off changed the bits in f32, f16 and Q4_K_M files on every model, and in
  q8_0 on 2 of 4 models (largest relative difference 5e-5 in f32, 7.5e-3 in q8_0).
* Batch 512 against 7 changed the bits only in Q4_K_M files (4 distinct results over 8 settings on
  every model), up to 2.7e-2 relative.
* The greedy tokens changed with settings in 2 of 16 model and format pairs (llama seed 1 Q4_K_M:
  batch 7 and flash attention each moved a token of one prompt; qwen2 seed 1 q8_0: flash attention),
  every change at a near tie (TS1317, all labeled near tie).

Cross check of the reference: Truestar's float64 replay of the source weights against transformers
running the same checkpoint in float64 (sdpa attention) differs by 2.8e-7 to 3.3e-7. transformers
computes the rotary angles and the RMSNorm variance in float32 even in a float64 model; replacing
both with float64 versions brings the difference to 1.8e-15 to 2.4e-15 on all 4 models. A second full run
reproduced every number above. Time: 13 to 133 s per model for 4 formats and 8 settings each (32
engine runs) on the shared server (a production box with other load), including the converter runs.

## Limits (2026-11-24)

* One build: llama.cpp v1.0.0, CPU backend, Linux x86. CUDA, Vulkan and Metal builds were not run
  (the GPU on the PC was reserved for a benchmark; the server has none). The driver sets
  `n_gpu_layers = 0`.
* The driver is compiled with a POSIX C++ compiler; Windows (MSVC) is refused with TS1310.
* The float64 replay covers the llama and qwen2 architectures (RMSNorm, rotary positions in the
  default and linear scaling, grouped query attention, SwiGLU, q k v biases). Others are TS1319.
  Tensor types: F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K.
* The rounding model is the CPU backend's (dot product types, f16 KV cache). A GPU backend rounds
  differently; its tolerance is not derived yet.
* The near tie zone follows the tolerance: with Q8_0 and Q4_K_M weights the modeled rounding moves a
  logit by up to about 0.13, so the per element bound reaches 1.3 and the ambiguous zone (a top two
  gap within twice the bound) is up to about 2.7 logits wide in these runs,
  and a flip inside it is never called a bug. A real engine bug smaller than that is caught
  only when it pushes a position over its tolerance.
* Prompts are token ids, so the tokenizer is not exercised (`truestar port ... gguf:` checks BOS
  handling with a text prompt).
* Planted faults only (an engine setting override, faulty conversions); no real llama.cpp bug, and
  no before and after pair of llama.cpp commits, was run. Tiny models only; no published GGUF.
