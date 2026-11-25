# Conform: GPU serving engines (vLLM, SGLang)

`truestar conform vllm` and `truestar conform sglang` show what a serving engine computes. The engine
serves a small Llama style decoder whose config and random weights are built in code (nothing is
downloaded). Truestar asks it for prompt log probabilities and greedy generations, then compares them
with its own float64 reference of the same weights, and compares the engine with itself under the
features engines call lossless: batching, prefix caching and chunked prefill.

## Run it

The engines need Linux and a CUDA GPU. The battery runs there and writes a results directory; the
judge runs anywhere, without a GPU:

```
python -m truestar.conform.serving_battery run --engine vllm --out DIR      # on the GPU host
truestar conform vllm DIR [--format agent]                                  # anywhere
```

On a machine that has the engine and a GPU, `truestar conform vllm` (no DIR) does both. Without them it
exits 2 with TS1320 and prints both commands. `truestar conform tensorrt-llm` exits 2 with TS1321 (see
"After launch"). `scripts/modal_serving.py vllm` runs the battery on a Modal L4 and copies DIR back.

The runs below are saved in `docs/data/serving/`, so this works on any machine today:

```
truestar conform vllm docs/data/serving/vllm_bfloat16
truestar conform sglang docs/data/serving/sglang_bfloat16
```

Exit codes: 0 conforms and deterministic, 1 a finding (including an engine that changes its own answer
under a lossless feature), 2 usage error (engine missing or refused, unreadable DIR, other weights).

## What is served and asked

* Model: Llama architecture, vocab 1024, hidden 512, 4 layers, 8 query heads and 2 key/value heads of
  64, MLP 1408, untied head. Weights come from NumPy's frozen legacy stream (seeded, the same draws on
  every NumPy and torch) and are rounded once to the serving dtype; the engine loads exactly those
  values. The judge rebuilds them and refuses a run whose weight hash differs (TS1322). A first run used
  torch's generator and was refused this way: torch 2.14 on the GPU host and torch 2.11 here drew
  different numbers.
* Prompts: 8 prompts of 112 token ids; the first 64 are shared, so a prefix cache can reuse them.
* Requests: greedy (temperature 0), 32 new tokens, end of sequence ignored, top 5 log probabilities at
  every prompt and generated position. The engines run with their tokenizer switched off.

Settings, each in a fresh engine process:

| setting | vLLM | SGLang |
|---|---|---|
| base | `enable_prefix_caching=False`, `enable_chunked_prefill=False`, `max_num_batched_tokens=2048` | `disable_radix_cache=True`, `chunked_prefill_size=-1` |
| prefix_cache | `enable_prefix_caching=True` | `disable_radix_cache=False` |
| chunked_prefill | `enable_chunked_prefill=True`, `max_num_batched_tokens=32` | `chunked_prefill_size=32` |

A 112 token prompt cannot be prefilled in one step under a 32 token budget, so the chunked setting
chunks every prompt. vLLM warns that disabling chunked prefill is not officially supported for this
model; its default is on, so `base` is the less usual configuration there.

Passes: `batch1` (each prompt alone), `batch1_repeat` (the same requests again), `batch8` (all prompts
in one call), `gen_only` (each prompt alone without prompt log probabilities). Both engines skip the
prefix cache for a request that asks for prompt log probabilities (both reported 0 cached tokens), so
the cache is judged on `gen_only`, where vLLM reported 512 and SGLang 888 cached tokens.

## How it is judged

Against float64, per position, the error is |engine log probability - float64 log probability| of the
token at that position. The tolerance is 10 times the RMS of the same error for a correct CPU run in the
serving dtype (every step rounded to bfloat16), and at least 0.05 nats for bfloat16. Generated tokens
are judged with the reference teacher forced on the engine's own generation, so one early difference
does not hide later ones. A choice that differs from the float64 argmax is a near tie (TS1325, either
choice is correct) when the float64 margin between the two tokens is at most twice the largest error
that correct bfloat16 run shows on any position; otherwise it is a real divergence (TS1324).

Against itself, every pair is compared bit for bit; the table gives the largest log probability change
(on positions with the same context) and the first generated token that changed, with whether float64
calls that position a near tie.

## Results: NVIDIA L4, bfloat16, 2026-11-24

vLLM 0.34.0 (torch 2.14.0+cu130), SGLang 0.5.23 (torch 2.14.1+cu130). Tolerance 0.439 nats (vLLM run)
and 0.44 nats (SGLang run); near tie margin 0.389 nats; the correct CPU bfloat16 run has RMS error
0.044 and largest error 0.195.

Against float64:

| engine | setting.pass | positions | over tolerance | RMS error | worst error | prompt top 1 real / tie | greedy real / tie |
|---|---|---|---|---|---|---|---|
| vLLM | base.batch1 | 1144 | 0 | 0.0204 | 0.0727 | 0 / 23 | 0 / 4 |
| vLLM | base.batch8 | 1144 | 0 | 0.0219 | 0.0736 | 0 / 23 | 0 / 11 |
| vLLM | prefix_cache.gen_only | 256 | 0 | 0.0189 | 0.0556 | n/a | 0 / 2 |
| vLLM | chunked_prefill.batch1 | 1144 | 0 | 0.0216 | 0.0701 | 0 / 24 | 0 / 7 |
| SGLang | base.batch1 | 1144 | 0 | 0.0225 | 0.079 | 0 / 8 | 0 / 5 |
| SGLang | base.batch8 | 1144 | 0 | 0.0228 | 0.109 | 0 / 15 | 0 / 5 |
| SGLang | prefix_cache.gen_only | 256 | 0 | 0.0223 | 0.0653 | n/a | 0 / 6 |
| SGLang | chunked_prefill.batch1 | 1144 | 0 | 0.0222 | 0.0809 | 0 / 15 | 0 / 5 |

Every disagreement with float64 is a near tie; in the vLLM base run the largest margin among them is
0.088 nats. Prompt top 1 counts repeat across prompts because the 64 token prefix is shared.

Against itself:

| engine | check | bitwise | largest change (nats) | generations changed (of 8) | first changed token |
|---|---|---|---|---|---|
| vLLM | same request twice | yes | 0 | 0 | none |
| vLLM | batch 1 vs batch 8 | no | 0.0789 | 5 | prompt 2, token 0 (near tie) |
| vLLM | prefix caching off vs on (cache read, 512 tokens) | no | 0.0612 | 1 | prompt 6, token 6 (near tie) |
| vLLM | chunked prefill off vs on | no | 0.0789 | 4 | prompt 2, token 0 (near tie) |
| SGLang | same request twice | yes | 0 | 0 | none |
| SGLang | batch 1 vs batch 8 | no | 0.1 | 5 | prompt 2, token 1 (near tie) |
| SGLang | prefix caching off vs on (cache read, 888 tokens) | no | 0.0617 | 4 | prompt 6, token 6 (near tie) |
| SGLang | chunked prefill off vs on | no | 0.1 | 3 | prompt 2, token 0 (near tie) |

Both engines are repeatable for an identical request in an identical engine, and both change bits, and
greedy text, when the same prompt is batched with others, read from a prefix cache, or prefilled in
chunks. Every change measured here is within the tolerance and starts at a near tie, so none is a wrong
answer; they are different correct answers, which is what a user comparing two runs sees.

## Does it catch a wrong engine

`--plant rope_theta=1000000.0` serves a config.json whose rotary base differs from the reference (a
classic conversion bug). On vLLM, L4 (`docs/data/serving/vllm_bfloat16_planted`): 623 of 1144 positions
over tolerance, RMS error 0.846 nats (0.020 clean), worst 3.36, 256 real prompt top 1 flips, and the
first real greedy divergence at the first generated token (TS1323, TS1324). The tests also plant a
layer's down projection at 1.5 times its value in a CPU stand in engine.

## Limits

* One small model, one GPU (L4), one dtype (bfloat16), one version of each engine. Other attention
  backends, CUDA graph sizes, speculative decoding, quantized weights, tensor parallelism and
  sampling with temperature are not covered.
* The tolerance comes from a CPU run that rounds every step to bfloat16, including softmax and norms;
  the engines keep more in float32, so their error is about half of it here. A fault smaller than
  10 times that RMS in a single position is not flagged.
* The near tie rule uses float64 margins; for an engine that is wrong (the planted run) some flips are
  also called ties.
* Prefix caching is only exercised for requests without prompt log probabilities, since both engines
  skip the cache for those; the report says when 0 tokens were read from the cache.

## After launch

TensorRT-LLM is refused by name (TS1321). Installing `tensorrt_llm` from pypi.nvidia.com on a slim
Modal image took about 9.5 minutes and then failed at import: pip resolved torch 2.14.0, and its
decoding operators did not load ("Decoding operators failed to load. This may be caused by an
incompatibility between PyTorch and TensorRT-LLM") with no CUDA toolkit in the image. It needs a pinned
torch and the NVIDIA container image; that is listed for after launch.

## Codes

| code | severity | meaning |
|---|---|---|
| TS1320 | error | serving_engine_unavailable: the engine is not installed here or there is no CUDA GPU (exit 2) |
| TS1321 | error | serving_engine_refused: named but not supported in this release (TensorRT-LLM) (exit 2) |
| TS1322 | error | serving_run_failed: a setting failed in the engine, or DIR cannot be judged |
| TS1323 | error | serving_logprob_divergent: log probabilities off float64 beyond the derived tolerance |
| TS1324 | error | serving_decision_divergent: a top 1 or greedy token differs from float64 beyond a near tie |
| TS1325 | info | serving_near_tie: differs from float64 only at near ties |
| TS1326 | warning | serving_repeat_differs: the same request twice gave different results |
| TS1327 | warning | serving_batch_differs: batch 1 vs batch 8 differ |
| TS1328 | warning | serving_prefix_cache_differs: prefix caching changes the results |
| TS1329 | warning | serving_chunked_prefill_differs: chunked prefill changes the results |

`truestar prefixcheck` ([PREFIX_CHECK.md](PREFIX_CHECK.md)) compares logits files from any engine;
this command drives the engine itself.
