# RL trainer and sampler pair check (`truestar rlpair`, pain P15)

In RL post training the rollout engine samples tokens and records their logprobs, and the trainer
recomputes the logprobs of the same tokens under the same weights. The two are different code: the
sampler decodes one token at a time with a KV cache, in batches with padding, often with fused
kernels and another dtype; the trainer runs one teacher forced forward over the whole sequence.
When their logprobs differ, the importance ratio trainer / sampler is biased and on policy training
becomes off policy, which can degrade or collapse a run (docs/PAIN_RESEARCH_2.md, N3).

`truestar rlpair` measures the gap and says where it comes from: which module first computes
differently for the same token position, and which ATen op inside it.

## Use

```
truestar rlpair pair.py                          # text report
truestar rlpair pair.py --format agent           # versioned JSON for agents
truestar rlpair pair.py --device cuda --batch-sizes 2,7,64 --strict --json out.json
truestar rlpair pair.py --device cuda --kernels rl --strict --tp 2   # on the shared RL kernels
```

`--kernels rl` runs both sides on `truestar.rlkernels` (one batch and position invariant kernel
set for the training forward, prefill and decode, docs/RL_KERNELS.md) and requires bitwise equal
logprobs; when they are not, the first op whose kernel differs between the trainer and the
sampler is named (TS185). `--tp N` compares the trainer at TP 1 with the pair file's
`tp_logits(model, tokens, N)` bitwise (TS186); in the shipped example that is a one device
simulation.

```python
from truestar.rlpair import pair_check
rep = pair_check(model, tokens, prompt_len, sample_fn, sample_model=sampler_copy, layout=layout)
print(rep.to_text())
```

`pair.py` defines:

| function | meaning |
|---|---|
| `get_model()` | the trainer's `nn.Module` |
| `get_tokens()` | `[B, T]` token ids: prompt plus the sampled completion |
| `get_prompt_len()` | P, the prompt length; tokens P to T minus 1 are scored |
| `sample_logits(model, tokens, prompt_len)` | runs the sampler and returns logits `[B, T - P, V]` (or logprobs `[B, T - P]`) it used for each scored token |
| `train_logits(model, tokens)` (optional) | the trainer forward, logits `[B, T, V]`; default `model(tokens)` |
| `get_sampler_model(model)` (optional) | the sampler's copy of the model (another dtype, another instance); default the same object |
| `get_layout()` (optional) | `{"rows": [...], "left_pad": [...]}`: where each trainer row sits in the sampler's batch and how many pad positions precede it |
| `tp_logits(model, tokens, tp)` (optional) | logits `[B, T, V]` of the trainer forward at tensor parallel size `tp`, for `--tp` |

The sampler may batch rows with other rows and left pad them; it must return logits for the trainer's
rows only. Module names must be the same on both sides (for an engine with other names, run
`truestar port` first, docs/PORT_CHECK.md).

Exit codes: 0 logprobs identical or equal to rounding, 1 a precision gap or a divergence, 2 usage
error. With `--strict`, 0 only for bitwise identical logprobs and a sampler whose logprobs do not
change with the batch.

## What it reports

* **Logprob statistics** over all scored tokens: max and mean absolute difference, the number and
  fraction over `--threshold` (default 1e-3), the importance ratio exp(trainer minus sampler) with
  min, max, mean, std, spread and the count outside 1 plus or minus `--clip` (default 0.2), a k3 KL
  estimate, and the worst positions (row, position, token, both logprobs).
* **Classification:** `identical` (bitwise), `match` (within rounding of the more precise dtype:
  1e-4 for float32), `precision` (every module stays within the band the sampler's dtype explains)
  or `divergent`.
* **Blamed module** with the kind of difference: `decode_path` (the module computes differently in
  incremental decoding than on the full sequence) or `implementation` (the sampler's module computes
  differently even on the full sequence: another implementation or other weights), its own error,
  the first position and decode step where it differs, and its source file and line.
* **Blamed op** inside it, with the scalar arguments of both paths and, for a slice or `arange` along
  the positions, the start the sampler used against the step's position (`offset +1`).
* **Inputs that differ:** a module that receives integer inputs (position ids, masks) whose values
  differ from the trainer's at the same position (TS197).
* **Outside the modules:** when every module matches but the logprobs do not, a sampling temperature
  is fitted to the trainer's logits and reported when it explains the gap (TS193).
* **Weights:** sampler parameters that are not the trainer's rounded to the sampler dtype (TS194).
* **Batch composition:** the sampler's logprobs for the first row alone against the same row at the
  first and last position of batches of `--batch-sizes` random rows, bitwise.

## Method

1. **Two recorded runs.** The trainer forward runs once with every module call recorded (inputs and
   outputs). The sampler runs once with every module call recorded and tagged with its decode step:
   each call of the step module (the model itself by default, `--step-module` otherwise) starts a
   step, and the length of its token input gives the step's chunk. Step i feeds c_i tokens at
   sampler positions o_i to o_i + c_i.
2. **Position mapping.** A tensor of a module call in step i is lined up with the trainer's tensor of
   the same call, dim by dim: a dim of size c_i is the step's chunk, a dim of size o_i + c_i is the
   cache prefix (all keys so far, as in an attention score matrix), a dim of size o_i is the cache
   before the step, dim 0 is the batch (trainer row b is sampler row `rows[b]`, shifted by
   `left_pad[b]` positions), and equal dims map one to one. A tensor that fits none of these is not
   compared and counted (TS196).
3. **Isolated replay.** For each module call, the sampler's inputs from every step are written into
   the trainer call's arguments at the mapped positions. Three runs of that full sequence follow:
   the trainer's module (reference), the sampler's module in its own dtype, and, when the dtypes
   differ, a copy of the trainer's module in the sampler dtype (the precision band). From these, a
   module's own error splits into the incremental path (sampler outputs against the sampler's module
   on the full sequence), the implementation (sampler's module against the trainer's on the same full
   sequence) and the band. A module is divergent when either part exceeds its threshold (1e-4 in
   float32; 2e-2 or 3 times the band in bfloat16), precision when only the dtype explains it. The
   first divergent module in execution order whose submodules are all clean is blamed.
4. **Op drilldown.** For a decode path divergence, the sampler is rerun with an ATen dispatch mode
   active only inside the blamed module's call at the first divergent step, and the same module is
   traced on the full sequence path. Ops are aligned by name, outputs are mapped with the rules of
   step 2, and each op whose mapped output differs is rerun alone with the full path's inputs written
   in at the mapped positions. The first op that still differs is named; if no aligned op differs on
   its own, the first op that only the sampling path runs is named (a temperature division). For an
   implementation divergence the regular op drilldown of `truestar check` runs between the two
   modules on the same inputs.

## Acceptance results (2026-11-24, CPU, torch 2.11; one CUDA check on the RTX 5070 Ti)

A two block GPT style decoder in plain `torch.nn` (vocabulary 64, width 32, 4 heads, rotary
positions, RMSNorm, SwiGLU MLP, random weights) with a KV cache decoding path, `tests/rl_models.py`.
Two sequences of 16 tokens, prompt 6, 20 scored tokens, 10 sampler steps.

| case | status | max abs logprob diff | blamed | op |
|---|---|---|---|---|
| correct sampler, float32 | match | 4.8e-07 (CUDA 2.4e-07) | none | |
| correct sampler, rows left padded by 3 in a batch of 4 | match | 4.8e-07 | none | |
| temperature 0.8 inside the `Head` module | divergent (TS190) | 0.347 | `head#0`, decode path, from position 0 | `aten.div.Tensor`, constant 0.8, sampling path only |
| temperature 0.7 applied to the logits outside the model | divergent (TS193) | 0.611 | none, temperature fitted 0.7000 (residual 6e-07) | |
| KV cache rotary offset off by one after prefill | divergent (TS190) | 0.0262 | `blocks.0.attn#0`, decode path, from position 6 (decode step 1) | `aten.slice.Tensor` starts at 7 for position 6, offset +1 |
| the same with left padding | divergent | 0.0262 | `blocks.0.attn#0` from position 6 | slice at sampler position 9 (trainer position 6), offset +1 |
| sampler in bfloat16, trainer float32 | precision (TS191) | 9.8e-03 (mean 3.1e-03, 17 of 20 over 1e-3, ratio 0.9946 to 1.0098) | none; largest own error 6.3e-03 against a band of 6.0e-03 | |
| bfloat16 sampler with the rotary bug | divergent | 0.0237 | `blocks.0.attn#0` | `aten.slice.Tensor` |
| position ids built from the cache offset (Hugging Face style) | divergent, TS197 | 0.0269 | `<model>#0` own code | `aten.arange.start` from 7 for position 6, offset +1 |
| one sampler weight 1% stale | divergent, TS194 | 1.5e-03 | `blocks.1.mlp.up#0`, implementation | weights named instead of an op |

Batch composition, the first row alone against batches of 2, 7 and 64 at the first and last position:

| backend, dtype | bitwise | max abs logprob diff |
|---|---|---|
| CPU float32 | 3 of 4 (only batch 2 at position 0 differs) | 4.8e-07 |
| CUDA float32 | 2 of 6: batch 64 bitwise, batches 2 and 7 not | 4.8e-07 |
| CPU bfloat16 sampler | 3 of 4 | 4.0e-03 at batch 7, position 0 |
| CUDA bfloat16 sampler | 6 of 6 | 0 |

So batch composition does change this sampler's logprobs on both the CPU and the GPU in float32,
at rounding level: bitwise equal logprobs need batch invariant kernels (P14, `truestar invariance`).

**Baseline.** Comparing logprobs alone (the vllm#21475 style check) gives the statistics in the first
part of the report and nothing else: every injected bug above shows up only as a logprob gap of 2e-3
to 0.6, with no module, no position and no op.

## Limits

* Both sides must run in one process as `nn.Module`s with the same module names. A remote engine
  (vLLM, SGLang) can be compared only by its returned logprobs, which gives the statistics but no
  localization.
* Steps are inferred from calls of one step module; a sampler that calls submodules directly, runs
  speculative decoding or chunked prefill with reordered rows needs `--step-module` and may not map.
* A dim that happens to have the size of the chunk or the cache can be mapped wrongly; tensors that
  fold the batch into another dim (B times S rows of a matmul input) are not compared at op level.
* Recording keeps every module input and output on the CPU, so memory grows with model size and
  sequence length; the check is meant for short sequences and small batches.
* Shared kernels for prefill, decode and the training forward are in docs/RL_KERNELS.md
  (`--kernels rl`); tensor parallel there is a one device simulation, no real multi GPU run.
