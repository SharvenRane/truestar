# truestar serve: an LLM server whose answers do not change with batching

Author: Sharven Rane. First written 2026-11-24.

Code in `src/truestar/serve/` (model, engine, check, http, cli), tests in `tests/test_serve.py`,
GPU measurements by `scripts/bench_serve.py` (Modal L4). Codes TS1400 to TS1409.

## Why

vLLM 0.34.0 and SGLang 0.5.23 both give a request other bits when it is batched with other requests,
when its prefix comes from the prefix cache, and when its prompt is prefilled in chunks
(`truestar conform vllm/sglang`, docs/CONFORM_SERVING.md). Each change is small
and starts at a near tie, but greedy text changes, and a sampled request with a fixed seed is not
reproducible: the answer depends on who else was using the server. `truestar serve` is a small server
built so that cannot happen, and checks it bit for bit.

## What it is

```
truestar serve [--preset tiny|bench] [--device cpu|cuda|reference] [--port 8000] [--chunk N]
               [--no-prefix-cache] [--max-batch 32] [--max-rows 512]
truestar serve --check [--device cpu|cuda] [--format agent] [--json out.json]
```

* **Model.** A Llama style decoder (RMSNorm, rotary positions, grouped query attention, SwiGLU, untied
  head) built from a config in code with weights from NumPy's PCG64 generator: nothing is downloaded,
  and the weights are the same bits under every torch version. Presets: `tiny` (vocab 320, width 64,
  2 layers) and `bench` (vocab 8192, width 512, 4 layers, 8 query and 2 KV heads, MLP 1408; the decode
  benchmark's shape in docs/LLM.md). Tokens are UTF-8 bytes (ids 0 to 255); no tokenizer download.
* **One step program.** Every token is a row of a paged KV cache step,
  `step(ids [N,1], pos [N,1], slots [N,1], table [N,M], k_pool, v_pool)`, imported through
  `torch.export` with symbolic `N`, `M` and pool size, and compiled once by `compile_program` in
  float32 with `deterministic=True` (on `cuda`, or on `cpu` through numba), or run by the float64 IR
  interpreter (`--device reference`). A prefill is several rows of one sequence, a decode one row, so
  any mix of prefills, prefill chunks and decodes is one call. Each layer writes all rows' K and V into
  the pool before any row reads, so the rows of one prefill chunk see each other.
* **Continuous batching.** Each step takes one row for every running decode, then fills the token
  budget (`--max-rows`, and `--chunk` with chunked prefill) with prefill rows, first come first served.
  New requests join as soon as a sequence slot and their KV blocks are free; finished ones leave at
  once. The row count is padded to a power of two with rows that write to scratch slots, so the
  compiled program meets at most 10 row counts (all run once at start up, `Engine.warmup`).
* **KV blocks and the prefix cache.** A request reserves blocks for its prompt plus `max_tokens` at
  admission (no preemption). A full block is keyed by the chained hash of every token up to its end;
  a new request reuses the cached blocks of its prompt (leaving at least one prompt token to compute)
  and only computes the rest. A block nobody holds stays readable until it is reallocated, least
  recently released first.
* **Sampling.** Greedy is the first index of the largest logit. Sampled requests (temperature, top p)
  draw by inverse CDF with the uniform `uniform01(seed, i, 0)` for the request's i-th generated token:
  the IR `multinomial` scheme (docs/LLM.md, Sampling), keyed by the request's seed and the token index
  only, never by the batch slot, the step or the chip. Logprobs are a float64 log softmax of the float32
  logits with a fixed summation order.
* **HTTP.** Standard library `ThreadingHTTPServer`. `POST /v1/completions` (prompt as text or token
  ids, `max_tokens`, `temperature`, `top_p`, `seed`, `stop`, `logprobs` 0 to 20, `stream`),
  `POST /v1/chat/completions` (`messages`, `max_tokens` or `max_completion_tokens`, `logprobs`,
  `top_logprobs`, `stream`; a plain `<|role|>` template), `GET /v1/models`, `GET /health`. Streaming is
  server sent events ending in `data: [DONE]`. `usage.prompt_tokens_details.cached_tokens` reports the
  prefix cache. A parameter the server does not implement is refused by name (TS1404) unless it has
  its default value (`n: 1`, `frequency_penalty: 0`, ...), never ignored. A sampled request without a
  seed gets one, returned as `truestar.seed` with TS1405, so it can be replayed exactly.

## Why the answers cannot change

A row's arithmetic does not depend on the other rows of its call:

1. **Matrix products per row.** In deterministic mode no contraction splits K, and without split K a
   row's result is bitwise independent of the row count and of the launch config (docs/KERNEL_IR.md,
   6b). Norms, rotary positions and the SwiGLU are elementwise or per row.
2. **Attention over a fixed set of slots.** Every table row is `M = max_len / block` blocks wide,
   whatever the sequence's length or the batch, so a row's softmax and weighted sum always run over the
   same `M * block` slots in the same order. Slots past the row's position are masked: their softmax
   weight is exactly 0, and 0 times a finite value adds nothing (unused pool slots hold zeros or old,
   finite K and V; padding rows write only to scratch slots).
3. **The cache holds the same bits.** By 1 and 2, the K and V of a position are the same bits whether
   they were computed alone, batched, in a chunk, or by an earlier request that shared the prefix. So a
   cached block is exactly what the request would have computed, and so is every later row.
4. **Keyed draws.** The sampler sees the same logits and the same uniform, so it picks the same token.

The cost is the price of 2: attention always reads `max_len` slots, even for a short sequence. At this
size that is cheap; for long contexts a fused paged attention kernel with a fixed chunk order would be
the next step.

## The guarantee test

`truestar serve --check` (and `tests/test_serve.py::test_guarantee_and_float64_on_cpu`, and on CUDA
`test_guarantee_on_cuda_and_cpu_agree`) runs two requests, one greedy and one sampled with a seed,
both 24 tokens with the top 3 logprobs, in six scenarios and compares tokens, logprobs and top
logprobs with `==` against the alone run:

| scenario | what changes |
|---|---|
| alone | nothing: the baseline |
| batched with 7 others | 3 requests arrive with it, 4 more join at steps 2, 4, 5 and 9 while it decodes; greedy and sampled, prompt lengths 2 to 40 |
| prefix cache on | an earlier request with the same 19 token prefix ran first; 16 tokens come from the cache |
| chunked prefill (5 tokens) | its prompt is prefilled 5 tokens per step |
| batched, chunked (3 tokens) | both |
| batched, prefix cache, chunked (3 tokens) | all three |

Then the float64 check: the same greedy request through the float64 reference server (the same step
program in the IR interpreter) must give the float32 server's tokens (a difference is accepted only at
a near tie, relative gap below 1e-3), and the reference server must equal a plain causal forward of the
whole text with no KV cache at all (tokens equal, logprobs within 1e-9).

Tests that prove the tests work:

* `test_a_batch_dependent_step_is_caught`: a step whose logits move by 1e-6 times the row count (what
  split K or a batch dependent kernel choice does elsewhere) fails the batched scenarios (TS1401) and
  passes the alone one.
* `test_a_prefix_cache_keyed_without_tokens_is_caught`: blocks keyed by position only (another
  prompt's block is reused) fail the prefix cache scenarios and not the chunked one.

## Results (2026-11-24)

Tests: `tests/test_serve.py` 11 passed and 1 skipped (CUDA) on the local CPU, 12 passed on a Modal L4 (8 min 45 s, the CPU compile under numba dominates).

Local CPU (numba executor, `tiny`): all 12 scenario runs bitwise, float64 check agrees (float32
logprob error 2.5e-6, reference against no cache 1.1e-14), 0 fallbacks.

Modal L4, torch 2.11 (`scripts/bench_serve.py --hf`, raw numbers in
`docs/data/serve/l4_2026-11-24.json`):

| check on cuda (`tiny`) | result |
|---|---|
| `serve --check`, compiled, deterministic | TS1400 and TS1408: all 12 runs bitwise; float64 greedy tokens equal, float32 logprob error 6.7e-6 |
| planted: the same step compiled with `deterministic=False` (split K allowed) | TS1401: 10 of 10 non alone runs differ, from the first token (batched, prefix cache, chunked all break) |
| planted: attention sized to the batch's widest sequence instead of the fixed `M` | TS1401: the prefix cache runs of the greedy request differ from the first token (on the numba CPU executor the same change stays bitwise: its sums run sequentially, so trailing zeros change nothing) |
| cuda against the numba cpu executor, the same two requests | tokens equal for greedy and for keyed sampling; logprobs not bitwise, largest difference 5.2e-6 |
| `bench` preset, 64 greedy tokens against a teacher forced float64 forward | tokens equal, logprob error 3.1e-5 |

Both planted rows are why the design has its two rules: deterministic mode, and the fixed table width.

Throughput, `bench` preset, B requests of 32 prompt tokens, 64 greedy tokens each, all submitted at
once, median of 3 runs after warm up. "Request 0" says whether the first request's tokens and
logprobs are the bits it got at batch 1.

| batch | method | tokens/s | ms per step | request 0 vs alone |
|---|---|---|---|---|
| 1 | truestar serve, compiled cuda, deterministic | 74 | 13.6 | bitwise |
| 1 | truestar serve scheduler, eager PyTorch step | 145 | 6.9 | bitwise |
| 1 | eager PyTorch generate (static batch, dense cache) | 179 | 5.6 | bitwise |
| 1 | Hugging Face `generate` (sdpa), same weights | 143 | 7.0 | bitwise |
| 8 | truestar serve, compiled cuda, deterministic | 494 | 16.2 | bitwise |
| 8 | truestar serve scheduler, eager PyTorch step | 959 | 8.4 | tokens equal, logprobs differ |
| 8 | eager PyTorch generate | 1362 | 5.9 | tokens equal, logprobs differ |
| 8 | Hugging Face `generate` | 1027 | 7.8 | tokens equal, logprobs differ |
| 32 | truestar serve, compiled cuda, deterministic | 1628 | 19.1 | bitwise |
| 32 | truestar serve scheduler, eager PyTorch step | 2624 | 11.8 | tokens equal, logprobs differ |
| 32 | eager PyTorch generate | 5626 | 5.7 | tokens equal, logprobs differ |
| 32 | Hugging Face `generate` | 4337 | 7.4 | tokens equal, logprobs differ |

Reading it:

* **The guarantee is the benefit, speed is the cost.** Only the compiled deterministic step keeps
  request 0's bits at batch 8 and 32. Every eager path, including Truestar's own scheduler with the eager
  step, changes the logprobs once the batch grows (TS1409 for the baselines), as vLLM and SGLang do.
* **It is slower, 2.4x at batch 1 and 3.5x at batch 32 against the eager generate loop.** Per step the
  compiled call costs 13.6 to 19 ms against 5.6 to 5.9 ms. Known parts: every row gathers its whole
  `max_len` cache per layer (the price of the fixed width), the gather and pool write are index library
  steps with device side index checks (docs/LLM.md, paged decode: 1 to 2 ms more per token than dense
  on a 5070 Ti), no split K for small M, each step returns new pools (value semantics, no in place
  update), and the host builds the rows in Python. The step was not profiled per kernel on the L4.
* vLLM numbers (docs/CONFORM_SERVING.md) are for context only: a different
  model and setup, not comparable to these rows.

## Limits

* Small random weight models only; no checkpoint loading and no real tokenizer (bytes).
* One request per completion (`n` 1), no beam search, penalties or logit bias (refused by name), no
  preemption (blocks for `max_tokens` are reserved at admission), no speculative decoding (not on
  main).
* Attention reads all `max_len` slots per row; long contexts need a fused paged attention kernel with
  a fixed chunk order (the split KV decode kernel already fixes its chunking by context length,
  docs/KERNEL_IR.md 6b).
* Across chips: the keyed draws are identical everywhere and the L4 and the numba CPU gave the same
  tokens, but float32 logits are not the same bits on different executors, so a sampled token can
  differ when its uniform falls within rounding of a CDF boundary, and logprobs differ in the last
  bits (5e-6 measured).
* Logprobs are of the raw model distribution, before temperature and top p.
