# Port check (`truestar port`, pain P16)

When a model is ported to a second implementation (a Hugging Face model reimplemented by hand, a JAX
or MLX port converted back to PyTorch, a serving engine's own model code, a checkpoint with a LoRA
adapter merged in), the port often computes a different model: a hardcoded eps, a missing square
root in the attention scale, rotary pairs rotated the wrong way, an extra cast, adapter weights
silently dropped by `load_state_dict(strict=False)`. People compare final logits, see that they
differ, and cannot tell where. See docs/PAIN_RESEARCH_2.md, section N4.

`truestar port` compares two `torch.nn.Module` implementations of the same model whose module names
and layouts differ, and names the first module whose own computation differs, with what differs
inside it.
Candidates that are not PyTorch modules (llama.cpp running a GGUF file, a JAX/Flax NNX or Keras 3
port) are in docs/PORT_ENGINES.md: `truestar port ref.py gguf:model.gguf`, `jax:cand.py`, `keras:cand.py`.

## Use

```
truestar port reference_model.py candidate_model.py
truestar port reference_model.py candidate_model.py --format agent    # JSON for coding agents
truestar port reference_model.py candidate_model.py --json port.json  # also save the JSON
```

`reference_model.py` defines `get_model()` and `get_inputs()` (optionally `get_kwargs()`), as for
`truestar check`. `candidate_model.py` defines `get_model()`, or `get_model(reference)`, which
receives the reference model so it can convert the weights the way the port's converter does.
Both models run on the reference's inputs. Exit codes: 0 same model, 1 different model or an
incomplete weight map, 2 usage error.

```python
from truestar.portcheck import port_check
report = port_check(reference_model, candidate_model, inputs)
print(report.to_text())
report.blamed            # PairResult: reference and candidate module, own error, threshold
report.op                # OpFinding: first differing ATen op, changed constants, missing ops
report.weights           # WeightMap: every edge, unmatched and unused weights
report.module_map        # ModuleMap: pairs, inlined modules, unmapped modules
```

## How the module map is inferred

Names are useless across ports, so the map comes from the weights.

1. **Weights by content.** Every reference parameter is matched to a candidate parameter, in this
   order: equal bytes; a transpose (Conv1D style storage); a row permutation (for example q and k
   rows permuted for interleaved rotary); a contiguous row block of a fused tensor, in either
   direction (q, k and v inside one `wqkv`, gate and up inside one `w13`); the same values in a
   lower dtype. What is left is matched to the closest tensor of the same shape (or to the unfilled
   rows of a fused tensor) if the relative difference is under 0.5, and reported as a value
   difference. If that difference equals `c * B @ A` for two unmatched reference tensors, the edge
   is recognised as a merged adapter and the factor `c` is reported. Identical tensors (norm weights
   that are all ones) are paired in the order the forward pass first reads them, found by tracing
   which parameter storages each ATen op reads.
2. **Modules by weights.** Reference module A maps to candidate module B when every matched weight
   under A has its counterpart under B and every matched weight under B comes from A. Wrapper chains
   with the same weights pair outermost first (so a LoRA wrapper pairs with the merged linear, not
   its base layer), tied copies pair in call order. A module whose weights are only a row block of
   a fused candidate module maps to it with an output slice (`q_proj` to columns 0:32 of `wqkv`);
   a row permuted linear compares its outputs through the same permutation. A module whose weights
   became plain parameters of a bigger module on the other side (an RMSNorm class written inline)
   is listed as inlined there; its computation is then part of that module's own code.
3. **Isolated replay.** Both models run in float64 on CPU. Each mapped candidate module is replayed
   alone with the reference module's recorded inputs (each candidate input tensor is paired with
   the reference input of the same shape, closest in value), so its error is its own. The threshold
   for each module is 10 times the error of the reference module itself computed in float32, never
   below 1e-9: two implementations that differ by less than float32 noise are the same model. A
   parent is replayed only if every mapped module inside it agrees, so the first module named
   differs in its own code.
4. **Op drilldown.** In the blamed pair both sides are traced at the ATen op level and aligned by op
   name (truestar.opdrill). The first candidate op whose output is wrong for the reference op's
   inputs is named, with the non tensor arguments that differ (eps, a slice step, a rotary base, a
   divisor), compute ops that only one side runs (a missing scale, an adapter path) and casts only
   the candidate runs. Layout ops such as views and splits are left out of the text, since fusion
   changes them freely; the JSON keeps them.
5. **Weights that do not line up are loud.** A weight one forward reads with no counterpart is
   TS162 (the dropped adapter case), a weight present on both sides that one forward never reads is
   TS163 (adapters loaded but never applied), a weight in one model only that is never read is TS169.

## Example

The reference is a Llama style decoder with Hugging Face names; the candidate fuses q, k and v and
writes RMSNorm inline, and hardcodes eps 1e-6 (tests/port_models.py):

```
truestar port: reference ref_model.py vs candidate cand_model.py
weights: 21 reference, 15 candidate, 21 matched (11 equal, 10 slice_of_candidate)
module map: 24 pairs, 5 reference modules inlined, 0 reference and 0 candidate modules unmapped
final output: relative error 7.20e-03 (DIVERGENT)
status: DIVERGENT

DIVERGENT: candidate layers.0 (BlockB) computes differently from reference model.layers.0
(DecoderLayer) given the same inputs (own error 1.24e-02, threshold 2.0e-06). Every mapped module
inside it agrees, so the difference is in its own forward code. It holds the weights of reference
module(s) written inline here: model.layers.0.input_layernorm, model.layers.0.post_attention_layernorm.
First differing op: aten.add.Tensor (op 3 of 92 in the candidate), own error 2.11e-02 for the
reference op's inputs; its non tensor arguments are [1e-06] where the reference has [1e-05]
(constant 1e-06 instead of 1e-05). Compute ops only in the reference: aten.mm.default x3.
```

Other findings from the test suite, each named at the module the bug is in:

| injected bug | named module (candidate) | explanation |
|---|---|---|
| eps 1e-6, norm as a class | `blocks.0.norm1` | `aten.add.Tensor`, constant 1e-06 instead of 1e-05 |
| rotary pairs interleaved without permuting q, k | `blocks.0.attn` | `aten.slice.Tensor`: every 2nd element where the reference takes a contiguous half |
| rotary base 500000 | `layers.0.attention` | `aten.pow.Scalar`, constant 500000 instead of 10000 |
| scale divides by head dim | `blocks.0.attn` | `aten.div.Tensor`, constant 8 instead of 2.82843, a missing square root |
| round trip through bfloat16 in a norm | `blocks.0.norm1` | casts only the candidate runs: to bfloat16, to float64 |
| LoRA dropped on conversion | `blocks.0.attn.wq` | TS162 lists the 8 lora_A and lora_B weights |
| LoRA loaded, never applied | `blocks.0.attn` | TS163, the q_lora and v_lora weights are never read |
| LoRA merged without its scaling | `layers.0.attention.wqkv`, columns 0:32 | rows 0:32 = reference + 1 * lora_B @ lora_A |
| one weight rounded to bfloat16 | `blocks.1.attn.wo` | TS165, values equal the reference rounded to bfloat16 |

Correct ports report clean: renamed modules, fused q, k, v with inline norms, interleaved rotary
with permuted q and k rows, a fused reference against a separate candidate, and LoRA merged
correctly into separate or fused weights.

## Agent JSON

`--format agent` prints one JSON document with the envelope of docs/AGENT_INTERFACE.md
(`schema_version` 1, `"command": "port"`) and these fields:

| field | meaning |
|---|---|
| `reference`, `candidate` | `name` and absolute `file` |
| `precision` | `"CPU float64"` |
| `status`, `exit_code`, `codes`, `issues` | as for `truestar check`; codes TS160 to TS169 |
| `output` | final output `rel_err`, `max_abs`, `nonfinite_mismatch`, `shape_mismatch` |
| `weights` | counts, `kinds` per edge kind, `reference_unmatched`, `candidate_unmatched`, `candidate_partial` (rows of fused tensors with no source), `reference_unused`, `candidate_unused`, every `edges` entry (`reference`, `candidate`, `kind`, `rows`, `note`, `values_differ`, `merged_adapter`) |
| `module_map` | `pairs` (`reference`, `candidate`, types, `view`, `rows`), `inlined`, `candidate_inlined`, `reference_unmapped`, `candidate_unmapped` |
| `blamed` | `reference`, `candidate` (call keys), types, `view`, `status`, `own_code`, `rel_err`, `accumulated_rel_err`, `threshold`, `ratio`, `inlined_reference_modules`, `source` and `reference_source` (file and line), `op` |
| `blamed.op` | `op`, `index`, `total`, `rel_err`, `argument_differences`, `only_in_reference`, `only_in_candidate`, `extra_casts`, `explanation` |
| `explanation` | the same sentence the text report prints |
| `modules_checked`, `modules_over_threshold` | replayed pairs, and the first 5 pairs over threshold |
| `next_actions` | `inspect_source` (candidate class file and line), `inspect_reference`, `inspect_op`, `load_missing_weights`, `use_or_remove_weights`, `wrap_in_module`, `rerun` |

## Limits

* Both models must take the same positional and keyword inputs. Inside, a candidate module input
  with no reference input of the same shape keeps the candidate's own value (named in the
  explanation), so a module whose layout differs (for example `[S, B, D]` against `[B, S, D]`)
  is compared but its inputs are not substituted.
* Fused tensors are recognised along dim 0 only (the usual layout of fused linear weights). A
  fused tensor whose parts are concatenated along another dim is reported as unmatched.
* Modules without weights (activations, a rotary module with no buffers) are never paired; their
  work counts as the enclosing pair's own code. Buffers are not matched.
* Both models must run in float64 on CPU; if not, both run in their own dtype (TS003) with looser
  thresholds.
* A merged adapter is recognised when its factors are plain tensors of the reference; quantized
  or DoRA style adapters are reported as value differences.
* Op drilldown aligns ops by name. When the candidate computes the same thing with entirely
  different ops (fused attention against explicit matmuls) the module is still named, but the op
  level explanation is the list of ops only one side runs.
* Agents can call it through the MCP tool `port_check` (docs/AGENT_INTERFACE.md) or the CLI.
