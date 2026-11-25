# Training steps under default verification (2026-11-25, RTX 5070 Ti)

`PYTHONPATH=src python scripts/trainverify_bench.py --time --configs ... --json after.jsonl`

The fair benchmark's training workloads (research/fairbench, rebuilt in the script): MLP, encoder
layer (`tel`), resnet18, each with AdamW and SGD, float32 strict (no TF32), compiled with
`compile_program(step.program, "cuda", "float32")` and default verification. `plain` calls
`cp(inputs, update=upd)` every step (automatic CUDA graph from the second step), `graph` calls
`cp.graphed(update=upd)`, `off` is `plain` with `graphs="off"`. `llama` is a Llama style decoder
written in the script (dim 512, depth 4, 8 heads, vocab 4096, batch 4 by 128), not part of the fair
benchmark. Step time: median of 20 steps after 5 warm up (CUDA events); eager is the same model in
PyTorch with `torch.optim` fused. First call: the whole verified first call (compile, tuning, every
check, the derivation).

## Before and after

Before: `results/laptop-trainverify/before*.jsonl` (base 92277bb, laptop RTX 5060) and the fair
benchmark report (11 of 12 refused on an L4). After: `after.jsonl` here.

| config | before | after | Truestar ms | eager ms | first call s |
|---|---|---|---|---|---|
| mlp adamw plain | refused (fc1.weight_new 7.0e-4 > 6.25e-4) | pass | 4.20 | 3.60 | 65.7 |
| mlp adamw graph | refused (same) | pass | 4.13 | 3.59 | 68.1 |
| mlp sgd plain | pass | pass | 4.00 | 3.33 | 67.4 |
| mlp sgd graph | pass (fair bench: TS801) | pass | 3.98 | 3.33 | 65.2 |
| tel adamw plain | refused (in_proj_weight_new 1.33e-3 > 9.2e-4) | pass | 9.42 | 9.08 | 26.0 |
| tel adamw graph | refused (same) | pass | 9.37 | 9.02 | 24.4 |
| tel sgd plain | refused (linear1.weight_momentum 1.67e-2 > 8.7e-4) | pass | 9.32 | 8.94 | 24.4 |
| tel sgd graph | refused (same) | pass | 9.32 | 8.96 | 25.1 |
| resnet18 adamw plain | refused (conv1.weight_new 2.0e-2 > 2.2e-3) | pass (6 weights accepted by the cut) | 44.57 | 38.88 | 1325.9 |
| resnet18 adamw graph | refused (same) | pass (6 by the cut) | 38.60 | 51.92 | 1339.1 |
| resnet18 sgd plain | refused (layer3.1.conv1.weight_momentum 2.46e-3 > 2.08e-3) | not measured | | | |
| resnet18 sgd graph | refused (same) | not measured | | | |

Ordinary calls (`off`): mlp adamw 4.44 ms, mlp sgd 4.16, tel adamw 9.84, tel sgd 9.77. The
automatic graph replayed from the second step in every `plain` row except llama
(`auto_graph_replays` 2 of the 3 later steps; the first later step records).

The run was stopped by the host for low memory during resnet18 sgd plain (the resnet18
derivation's CPU fallback holds about 20 GB), so the two resnet18 SGD rows have no after number.

Llama: `llama adamw off` pass (29.34 ms, eager 8.26), `llama sgd plain` pass (32.22 ms, eager
7.82). `llama adamw plain` crashed in the eager timing (rotary tables were plain tensors, not
buffers; fixed in the script after the run, not rerun). `llama adamw graph` is refused with TS800:
a step copies a CPU tensor to the GPU on every call, so the step cannot be recorded, and the
automatic graph of `plain` falls back to ordinary calls for the same reason. An earlier run of
`llama adamw plain` without timing passed verification (before the derivation fixes it was refused:
`emb.weight_new`, 78 elements at 17.9x).

## What changed the verdicts

docs/TOLERANCE.md, "The compiled executor's first call checks": over the fixed bar the derived
bound decides; a call with `update=` or `donate=True` is judged by the derived bound alone (which
is what refuses a wrong decay sign or a wrong cast inside the bar); constant arithmetic is exact in
the error model (it made every Llama gradient's error model run NaN); the per element multiple is
the Student t quantile for 8 runs and the output's size; outputs computed from verified outputs
are judged from them (AdamW's first step sign flips).
