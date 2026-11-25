# Batch invariance (`truestar invariance`, pain P14)

The same prompt at temperature 0 gives different completions depending on what else is in the
batch (docs/PAIN_RESEARCH_2.md, N2). The cause is kernels whose reduction order depends on the batch
shape: a GEMM that picks another tile or split of K for another row count, an attention kernel that
splits the keys differently, a reduction that changes its tree. Each answer is correct to float
precision, but the bits differ, and greedy decoding amplifies one bit into another token.

`truestar invariance` asks one question per row: does this row get the same bits whatever else is in
the batch? When the answer is no, it names the module and the ATen op responsible, says whether a
torch flag fixes it (only after rerunning with that flag), and counts the prediction and token flips.

## Use

```
truestar invariance model.py                               # every backend on this machine
truestar invariance model.py --backends cpu,cuda --format agent
truestar invariance model.py --sizes 2,7,64 --rows 2 --decode 4 --seq-pad 5 --no-flags --json out.json
```

```python
from truestar.invariance import invariance
rep = invariance(model, inputs, backends=["cpu", "cuda"])
print(rep.to_text())
```

`model.py` defines `get_model()` and `get_inputs()` as for `truestar check`, and optionally
`get_batch_dim()` (default 0). Tensors whose size at the batch dim equals the example batch are
batched; other inputs (masks, tables) are passed unchanged. A single `[B, S]` integer input is
treated as token ids: then greedy decoding and sequence padding are checked too.

Exit codes: 0 every backend invariant, 1 some row is not, 2 usage error.

## Method

For each of the first `--rows` example rows, on each backend, in the model's own dtype:

1. The row runs alone (batch 1). That is the reference. There is no float64 here: the question is
   whether the same backend gives the same bits for the same row.
2. The row runs again in contexts that must not change its result: the same call repeated (run to
   run determinism), inside batches of 2, 7 and 64 filled with random rows of the same statistics,
   at the first, middle and last position, inside a batch of 8 padded with zero rows, and for token
   models with 5 pad tokens appended (causal attention must not let them change earlier positions).
   The row's slice of every output is compared bitwise and numerically: `bitwise`, `close` (within
   the dtype's float tolerance) or `different` (beyond it, a real dependence on other rows).
   Argmax flips over the last dim and greedy decoding token flips are counted.
3. Module drilldown, for up to four failing contexts: every module call is recorded in the alone run
   and in the context run, then replayed alone with the context's inputs except that the row's
   slice is replaced by what the module saw when the row ran alone. A module whose row output still
   differs computes that row differently because of the rest of the batch. The first such module
   whose submodules are all invariant is named: its own code or its own kernel is the cause.
4. Op drilldown inside each such module: ATen ops are traced in both runs and aligned by name, and
   each op whose row slice differs is rerun alone with the same substitution. The first op whose row
   output still differs is named. Ops that only copy bits (view, cat, select and the like) are never
   blamed for a float level difference, since they cannot round.
5. Flags: each flag set below is applied, the whole output check is rerun, and the flags are
   restored. A flag set is reported as a fix only when the rerun had every pair bitwise identical.

| backend | flag sets tried |
|---|---|
| CUDA | `deterministic` (use_deterministic_algorithms with CUBLAS_WORKSPACE_CONFIG, cudnn deterministic, no benchmark), `no_reduced_precision` (no TF32, no fp16 or bf16 reduced precision reductions), `math_sdpa` (flash, mem efficient and cudnn attention off), `all` |
| CPU | `deterministic`, `one_thread` (torch.set_num_threads(1)), `mkldnn_off`, `all` |

Locating the row: a tensor dim of size a alone and size n * a in a batch of n holds the row either at
`[p * a, (p + 1) * a)` (batch major, for example the B * S rows of a matmul) or at `p::n` (sequence
major, as inside `nn.MultiheadAttention` with `batch_first=False`); the candidate closest to the alone
tensor is used. A grown dim in the sequence padding context holds the row at `[0, a)`. A module or
op whose tensors cannot be lined up this way is skipped and counted in a note, never guessed.

## Agent JSON

`--format agent` prints one JSON document (schema_version 1, `command: "invariance"`):
`status` (`invariant`, `not_invariant`, `error`), `codes`, `issues`, `settings` (rows, contexts,
decode steps), and per backend: `status` (`invariant`, `batch_variant`, `nondeterministic`,
`batch_dependent`, `error`), `summary` (pairs, bitwise, close, different, argmax flips, token flip
contexts), `rows` (one entry per row and context with `bitwise`, `class`, `max_abs`, `rel_err`,
argmax flips and, for token models, both token lists and `token_flip_step`), `first_module` (with
source file and line), `first_op`, `non_invariant_ops` (module, op, contexts), `modules`,
`flag_trials` (`flags`, `settings`, `fixed`, `remaining`) and `notes`.

`next_actions` kinds: `use_batch_invariant_op` (the op, its module and a reference replacement from
`truestar.invariant_ops`), `set_flags` (only for a flag set verified to fix it, `verified: true`),
`flags_do_not_fix` (every set was tried and none fixed it), `fix_batch_dependence` (a real
dependence, with the user file and line), `check_flips`, `inspect_failure`.

| code | severity | slug | meaning |
|---|---|---|---|
| TS050 | warning | batch_variant | a row is not bitwise identical inside other batches, positions or padding |
| TS051 | warning | nondeterministic | the same row run twice alone gives different bits |
| TS052 | error | batch_dependent | a row's output depends on other rows beyond float tolerance |
| TS053 | warning | invariance_flips | the non invariance flips argmax predictions or greedy tokens |
| TS054 | warning | invariance_unattributed | not invariant, but no single module or op was named |

## What this machine shows (2026-11-24, RTX 5070 Ti, CUDA 12.8, torch 2.11, Windows)

Acceptance models, random weights, rows 0 and 1, 10 contexts (11 with sequence padding), greedy
decoding 4 tokens for the GPT:

| model | backend | status | bitwise pairs | max abs | op named | flag that fixes it |
|---|---|---|---|---|---|---|
| MLP 64, 256, 10 (float32) | CPU | batch_variant | 2/20 | 1.4e-7 | `aten.addmm` in fc1 and fc2 | none |
| MLP (float32) | CUDA | batch_variant | 2/20 | 6.7e-8 | `aten.addmm` in fc1 and fc2 | none |
| TransformerEncoder 2 layers (float32) | CPU | invariant | 20/20 | 0 | | |
| TransformerEncoder (float32) | CUDA | batch_variant | 2/20 | 9.5e-7 | `aten.addmm` in self_attn, linear1, linear2 | none |
| GPT decoder 2 blocks (float32) | CPU | batch_variant | 20/22 | 6.0e-7 | `aten._scaled_dot_product_flash_attention_for_cpu` (sequence padding only) | none |
| GPT decoder (float32) | CUDA | batch_variant | 2/22 | 6.0e-7 | `aten.addmm` in every Linear | none |
| GPT decoder (bfloat16) | CPU | batch_variant | 20/22 | 7.8e-3 | the CPU flash attention kernel (sequence padding only) | none |
| GPT, MLP, encoder (bfloat16) | CUDA | invariant | all | 0 | | |
| GPT built from `truestar.invariant_ops` (float32) | CPU and CUDA | invariant | 22/22 | 0 | | |

Findings, in plain terms:

* **CUDA float32 GEMM is not batch invariant even at batch 2:** a row alone uses a different kernel
  than the same row with one other row (in a direct probe also for K = 64, so this is kernel choice
  by row count, not only split K). No torch flag fixes it: deterministic algorithms make a fixed
  shape repeat its bits, they do not make a row independent of the shape.
* **CUDA bfloat16** was invariant for these small models, but a direct probe shows `aten.mm` with
  K = 4096 differs by up to 1.0 (one bf16 ulp at that magnitude) between batch 1 and batches 2, 7 and
  64, and `aten.addmm` differs at K = 512 and 4096: split K at large K. CUDA float32 `sum` over 512 or more
  elements also changes with the number of rows.
* **CPU float32 GEMM (MKL) is position dependent:** inside a batch of 7, row 0 differs from the
  alone result while row 6 does not. With a transposed right operand (what `nn.Linear` passes),
  half of the 64 positions in the same 64 row GEMM round differently from position 0.
* **CPU attention is sequence length dependent:** padding the sequence by 5 tokens changes earlier
  positions by up to 6e-7 (float32) and 7.8e-3 (bfloat16) through the CPU flash kernel, although
  causal masking makes the math identical.
* **CPU elementwise ops depend on their offset in the flat tensor:** `sigmoid`, `silu` and tanh
  GELU give other bits in the scalar tail loop than in the vectorized body (about 4% of elements
  differ), so a row whose length is not a multiple of the vector width changes with the batch; plain
  `exp`, `tanh`, `erf` and exact GELU did not. While building the reference ops, an elementwise op
  over 32768 element pieces was also seen to change with the thread split, and there
  `one_thread` was verified as the fix.
* **Layouts matter:** in the first version of the reference attention, a permuted operand was a view
  at batch 1 and a copy at batch 2, so the GEMM saw other strides and rounded differently. The
  checker found it (`Attn` named); the fix is to make operands contiguous.
* No argmax or greedy token flips were seen with these random weight models: the float level
  differences (up to 1e-6) never crossed a near tie in 4 decoded tokens. Flips are counted and are
  demonstrated with an injected batch dependence (tests).

Not measured: tensor parallel degree (one GPU here), other GPU models, ROCm and MPS. Cross GPU type
differences are what `truestar check` and `truestar record`/`diff` cover.

## Batch invariant reference ops (`truestar/invariant_ops.py`)

A proof of concept, not wired into the kernel layer. Two ways to fix each row's reduction order:

* **fixed tiling** (`matmul`, `linear`, `bmm` in `tile` mode): rows go through the vendor GEMM in
  tiles of exactly 64 rows (the last padded with zero rows), every call with the same shape and a
  row major right operand; batched matmuls run one matrix at a time (batched GEMM kernels change
  with the batch count: on CUDA float32 a per row `bmm` was not invariant). This relies on the
  vendor kernel being a function of the shape only, which the tests check on each backend.
* **explicit trees** (`tree_sum`, `tree_max`, `mean`, `softmax`, `layer_norm`, `rms_norm`, and
  `matmul`, `bmm`, `scaled_dot_product_attention` in `tree` mode): reductions in chunks of exactly 64
  elements, each a fixed pairwise tree of elementwise adds, chunks accumulated left to right.
  Elementwise IEEE adds and multiplies are exact per element, so these are invariant by
  construction, and padding with identity elements leaves them unchanged, which makes the tree
  attention also invariant to right padding of the sequence. `elementwise`, `gelu`, `silu`,
  `sigmoid` pad the flat tensor so that no element reaches a scalar tail.

Measured overhead (`measure_overhead()`, one decoder layer's shapes, 64 x 16 tokens, width 256,
seconds per call, best of 3):

| op | CPU float32 | CPU bfloat16 | CUDA float32 | CUDA bfloat16 |
|---|---|---|---|---|
| linear, tile 64 | 2.8x | 6.6x | 12.6x | 15.2x |
| linear, tree | 193x | 478x | 114x | 203x |
| causal attention, tree | 94x | 57x | 52x | 68x |
| softmax | 34x | 4.4x | 143x | 146x |
| layer norm | 22x | 22x | 75x | 69x |
| tanh GELU | 6.1x | 6.0x | 26x | 21x |

Whole GPT decoder above at batch 64: 1.74 ms to 22.7 ms on CPU (13x), 0.79 ms to 7.3 ms on CUDA
(9.2x). Tile size trades launches against padding: for a 1024 by 256 by 1024 linear, tile 16 is 9x
(CPU) and 30x (CUDA), tile 256 is 2.0x and 2.8x; for 16 rows tile 16 is cheapest. The default 64 is
a compromise. These numbers are Python level loops and trees over stock kernels; they show the cost
of fixing the order without a compiler, and they are far above the ledger's 20% bar, which is for
Truestar compiled kernels with an invariant reduction schedule (not built yet).

## Tests

tests/test_invariance.py (30 tests, CPU and the local GPU): row location, contexts, the three
acceptance models on each backend (asserted against a direct op probe, so a machine where
`F.linear` is invariant skips instead of failing), the clean invariant model, an injected op whose
output depends on the batch size (`BatchOpFault` on `aten.mul.Tensor`: module `scale#0` and the op
named, no flag fixes it), a real dependence (`BatchMix`: TS052, `aten.mean.dim`, argmax flips, user
file and line), greedy token flips, run to run drift (TS051, no op blamed), a flag verified as a
fix (`one_thread`), agent JSON and CLI, and every reference op for correctness and row invariance in
float32 and bfloat16. Breaking the op drilldown (return nothing) failed 11 tests; marking every
batch dependent module as its own cause failed 10; a permanent test removes the row substitution and
asserts the report is no longer the single faulty op.

## Training mode (`--train`, since 2026-11-24)

`invariance(..., train=True)` runs the model in training mode with keyed dropout (docs/RANDOM.md):
each mask is drawn once over the dims after dim 0 and shared by every row, so a row's mask does not
depend on its batch position. A two layer `nn.TransformerEncoder` with dropout 0.1 shows no batch
dependence (TS052) on CPU in training mode. Training batch norm normalizes a row with the statistics
of its batch, so it is reported batch dependent (TS052), which it is by definition; `BatchNorm1d`
cannot run one row alone and the report says so instead of failing.

## Limits

* The reference is the same backend at batch 1. A kernel wrong at every batch size is
  `truestar check`'s job.
* Op drilldown cannot see a row's position inside a same shape tensor: if a kernel is position
  dependent within a fixed tile, the module is named but no op (TS054).
* Sequence padding is checked only for token models with right padding and no attention mask
  argument; padding with a mask, left padding and KV cache paths are not.
* Random filler rows use the example's statistics; real traffic may hit other kernel paths.
* Tensor parallel invariance and the compiled invariant kernels from the P14 row are not built.
* Training mode shares one dropout mask across rows; a mask that is independent of the batch position
  for every layout (sequence first included) is not built.
