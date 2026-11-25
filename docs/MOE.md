# Mixture of experts

`truestar.moe` (2026-11-24) lowers routed top k MoE layers at import, runs them with a fixed tie rule
and a grouped expert kernel whose results do not depend on scheduling, and reports the tokens whose
expert choice is not determined in the working precision.

## What is covered

transformers 5.x writes a routed MoE block as a router child that returns `(logits, weights,
indices)` and an experts child holding stacked `gate_up_proj [E, 2I, H]` and `down_proj [E, H, I]`,
computing `act(gate) * up`. Truestar lowers every experts module of that form, checked on the
instance: no biases, not transposed, the default gate, an `act_fn`. Measured on transformers 5.24
with tiny random configs (`truestar.testing_hf.MOE_2026`, 4 experts top 2; DeepSeek also with 8
experts in 4 groups, top 2 groups):

| family | before (9a9ad96) | after: import, float64 error, compile (CPU, CUDA), train (gradient error) |
|---|---|---|
| Mixtral | refused TS940 | PASS 3.0e-16, PASS, PASS 3.0e-16 |
| Qwen3 MoE | refused TS940 | PASS 3.4e-16, PASS, PASS 5.0e-16 |
| OLMoE | refused TS940 | PASS 4.4e-16, PASS, PASS 6.5e-16 |
| DeepSeek V3 (shared plus routed) | refused TS940 | PASS 2.9e-16, PASS, PASS 2.9e-16 |
| DeepSeek V3, group limited routing | refused TS940 | PASS 3.1e-16, PASS, PASS 1.8e-16 |

The float64 error is the relative max difference between the imported program and transformers'
own eager experts, both in float64; the gradient error is the same for every parameter's gradient
(`scripts/op_coverage.py --sources hf --train --compile cpu`). Qwen3 Next and Kimi Linear have
covered MoE layers too; they stay refused, now with TS941, for their linear attention.

Not covered, refused with TS940 and the reason in the message: gpt-oss (transposed experts with
biases and a clamped SwiGLU), Llama 4 (a different weight layout), Nemotron H (per expert modules),
Switch Transformers and NLLB MoE (expert capacity dropping), and any layer exported outside the
lowering.

## How it is lowered

* **Importer**: `ir.import_module` runs `truestar.moe.lowered(model)` around torch.export. Each
  covered experts module then computes every expert on every token and keeps the routed ones with a
  mask, in expert order (`truestar.moe.static_experts`). That form has no data dependent shapes, so
  it imports into ops that already have a float64 reference, gradient rules and compiled kernels.
  Unrouted experts add an exact zero (a `where`, never `0 * y`). It costs E / k times the expert
  FLOPs of a routed implementation.
* **Routing**: the router imports as written. Its `topk` is the IR's `topk`, the first k of a stable
  sort. The tie rule, on every device: a larger score first; bitwise equal scores to the lower expert
  index; NaN above every number (`truestar.moe.TIE_RULE`).
* **Eager**: `truestar.moe.use_grouped(model)` runs each covered experts module with
  `grouped_experts` and every MoE block's `torch.topk` under the tie rule; `restore(model)` undoes it.
  `grouped_experts` sorts the (token, slot) pairs by expert with a stable sort, gathers each
  expert's tokens into one block, runs one matmul pair per expert, puts the results back by a
  permutation (no atomic adds) and sums the k slots in a fixed order. The first call per shape,
  dtype and device is checked against the same math in float64 (TS1181, `TRUESTAR_VERIFY` as for
  every kernel).

```python
from truestar import ir, moe
prog = ir.import_module(model, args)      # covered MoE layers lowered, the rest refused by name
moe.use_grouped(model)                    # eager: grouped experts, fixed tie rule
```

## Measured on one NVIDIA L4 (Modal, torch 2.11 cu130, TF32 off)

| check | result |
|---|---|
| grouped experts, float32 against float64 (256 tokens, 8 experts, top 2) | 2.4e-7 (CUDA), 6.6e-7 (CPU) relative |
| grouped experts, two runs, forward and every gradient | bitwise equal on CPU and on CUDA |
| compiled program (the 5 configs), two runs | bitwise equal on CPU and on CUDA |
| compiled program against float64 | at most 2.5e-7 (CUDA), 1.1e-7 (CPU) relative |
| compiled program, CPU against CUDA | not bitwise equal (different matmul kernels); same experts chosen |
| eager with `use_grouped`, CPU against CUDA | the same experts for every token in every layer of the 5 configs |

Speed of one experts call, median of 30 timed calls after 5 warm up calls, the float64 check off,
inputs drawn once (`scripts/moe_bench.py --only speed`), one run on one L4:

| config | dtype | grouped | transformers eager | transformers grouped_mm |
|---|---|---|---|---|
| 512 tokens, 8 experts, top 2, H 256, I 512 | float32 | 1.18 ms | 3.09 ms | 0.73 ms |
| same | bfloat16 | 1.19 ms | 3.12 ms | 0.73 ms |
| 8192 tokens, 32 experts, top 4, H 1024, I 1408 | float32 | 39.6 ms | 43.9 ms | 40.9 ms |
| same | bfloat16 | 10.0 ms | 15.9 ms | 10.5 ms |

The grouped kernel is faster than the library's eager loop at both sizes and slower than its
`grouped_mm` at the small one, where the one Python loop over experts dominates; whether `grouped_mm`
is bitwise reproducible was not measured. These are single runs, not checked against the roofline.

## Near tie router report

```python
from truestar.moe.ties import router_ties
rep = router_ties(model, args, dtype=torch.bfloat16)   # float64 on the CPU against bfloat16
print(rep.describe())
```

For each covered layer, the model runs in float64 on the CPU and in the working precision on its
device; hooks read each router's input in both. Per token and expert, the router logit's rounding
bound is `(2u + gamma_H(u_acc)) * sum_i |W_ei| |x_i|` plus `sum_i |W_ei| |x_wp,i - x_i|`, the error
that reached the router from earlier layers. Softmax routers compare logits (8 float32 ulps added for
the softmax), DeepSeek style routers `sigmoid(logit) + bias`, and group limited routing is tested at
the group boundary as well. A token is a near tie (TS1182) when its float64 gap between the k-th and
(k+1)-th score is at most the sum of their bounds. A token whose expert set changed in the working
run although it is not a near tie is TS1183; a router whose rule the report does not bound is listed
as TS1184.

On the 5 configs above: no near ties in float32; in bfloat16, 2 of 24 tokens (Mixtral), 5 of 24
(Qwen3 MoE), 4 of 24 (OLMoE), 2 of 12 (DeepSeek V3) and 1 of 12 (group limited). With a router
planted so two experts differ by about 1e-9 at the top 2 boundary, every token is flagged, for
example:

```
router near ties (float32 on cpu, bound from float64)
  gate: 8 tokens, top 2, 8 near ties, 7 expert set changes (0 not near ties), smallest gap/bound 8.97e-06
    TS1182 token 0 (0, 0): experts 2 and 1 differ by 3.820e-09 in float64, bound 1.398e-05 (flipped)
```

## Codes

TS1180 an expert index outside 0..E-1; TS1181 grouped experts differ from float64; TS1182 router near
tie; TS1183 routing change that is not a near tie; TS1184 router rule not bounded. They are defined in
`truestar.moe.codes` and move into `truestar.codes` with their rows in docs/AGENT_INTERFACE.md.
