# Precision planner (`truestar compile --precision auto`)

Author: Sharven Rane. Written 2026-11-24. Code: `src/truestar/precplan/` (`units.py`,
`planner.py`, `output.py`, `cli.py`); tests in `tests/test_precplan.py`; study in
`scripts/precision_study.py`; example `examples/precision_model.py`.

Pain row: P33 (docs/QUALITY_BAR.md). Research item: GAP_COMPETITORS 4 and its Redesign section.
Never introduce items 1 (no silent fallback), 2 (no false alarms), 3 (no silent wrong results of
our own) and 6 (no cryptic errors).

## The problem

Choosing a precision per layer is how a model gets fast on a GPU, and it is where it silently
gets wrong. TensorRT 11 removed the builder's freedom to pick FP16 or FP32 per layer: the choice
now happens in a separate ModelOpt pass over an ONNX file (`modelopt.onnx.autocast`, then
`trtexec --stronglyTyped`). Users report in NVIDIA/TensorRT#4807 that this moved "precision
assignment away from the component that actually knows the target hardware", reduced ONNX
portability, "significantly increased dependency footprint" and ran about 20% slower than the
builder's own choice. The usual alternatives are a uniform dtype (all float16, all bfloat16), a
rule by class ("float16 except norms", torch.autocast's op lists), or both, judged by an accuracy
number that hides individual decisions changing (pain P2).

## The redesign

The tool that times the kernels picks the precision, per layer, and a plan is accepted only when
decision flips against float64 stay within the user's budget on held out inputs it never searched
on.

* **Correct.** Every candidate is measured against float64, not against float32 or a cosine. A
  layer whose values leave float16 or bfloat16 range or lose their digits there is excluded by
  `truestar range` on a float64 run of the IR (an analysis, not a lucky calibration batch).
  Quantized candidates carry both verdicts of `truestar.quant`: a kernel that fails its exact
  check on the layer's own inputs is removed as a kernel bug (TS664), never mistaken for format
  loss, and the format's own loss is measured from float64 references alone. The finished plan is
  verified end to end on held out inputs; if it fails there, the search tightens and runs again.
* **Efficient.** One knob instead of a sweep: a per layer error threshold tau. For each tau a
  dynamic program picks, per layer, the fastest candidate within tau, counting the casts between
  activation dtypes. Plans get faster as tau grows, so the fastest plan within budget is found by
  bisection over tau: O(log n) verification runs. Speed is measured, per layer and end to end,
  as GPU time from CUDA graph replay on this GPU, and timing differences under 5% (or 2 us) count
  as equal, in favour of the more accurate candidate.
* **Easy.** One flag on one command, the PyTorch model as it is (no ONNX rewrite, no second
  package), and a report that says per layer what was chosen and why the faster candidates were
  not.

## Use

```
truestar compile model.py --precision auto --max-flips 0.1%
truestar compile model.py --precision auto --max-flips 4 --plan plan.json --format agent
truestar compile model.py --precisions fp32,tf32,bf16,fp16      # no quantized kernels
```

`model.py` defines `get_model()` and `get_inputs()` (calibration inputs) as for `truestar
check`, optionally `get_calibration()` (more forward argument tuples) and `get_holdout()`
(forward argument tuples used only to verify the finished plan; without it the calibration inputs
are split in two halves). `--max-flips` is a count of calibration decisions (scaled to the held
out size) or a share. Exit codes: 0 plan within budget, 1 budget not reachable or a kernel bug
found, 2 usage error (TS669 without a CUDA GPU).

```python
from truestar import precplan

plan = precplan.plan(model, calibration, holdout, max_flips="0.5%")
print(plan.to_text())          # per layer choice and why; plan.as_dict() for JSON
plan.save("plan.json")
fast = precplan.apply(model, "plan.json")   # refuses another model's plan (TS666)
y = fast(x)
```

A decision is the argmax of the last output dim per row: a class, or a token per position for a
language model.

## Precisions

| precision | weights | activations | compute | for |
|---|---|---|---|---|
| `fp32` | float32 | float32 | strict float32 (TF32 off, also for convolutions, which PyTorch runs in TF32 by default) | every unit |
| `tf32` | float32 | float32 | TF32 tensor cores for this unit only | Linear, convolutions, attention modules |
| `bf16`, `fp16` | that dtype | that dtype | cuBLAS and cuDNN in that dtype | every unit |
| `int8w` | INT8 per output channel | float16 | Truestar weight only kernel (docs/QUANT.md) | Linear |
| `int4w` | INT4 per group of 128 (or 64, 32) | float16 | Truestar weight only kernel | Linear |
| `fp8` | FP8 E4M3 per channel | float16, quantized per token inside | Truestar FP8 kernel on the FP8 tensor cores | Linear |

A precision a layer cannot run (a shape the kernel does not take) is named (TS665) and never
replaced by something else.

## How it works

1. **Reference.** The model in float64 on the CPU, once per calibration and held out batch.
2. **Units.** A unit is a leaf module, or a module whose forward reads its children's weights
   instead of calling them (nn.MultiheadAttention reads `out_proj.weight`): the planner wraps every
   leaf, runs the model once, and merges any wrapped child that was never called into its parent.
   Units cast their inputs to their activation dtype and return in it; code between units runs
   in whatever dtype arrives, with PyTorch's promotion.
3. **Tie groups.** Code between units can mix dtypes in an op that does not promote (a query from
   a float16 layer times a key from a float32 layer). Probe runs give neighbouring units
   alternating dtypes; an error inside module M ties every unit under M to one activation dtype
   (TS667). Members of a group can still differ in precision within that dtype (an INT4 and an
   FP16 projection both produce float16).
4. **Range.** `truestar range` runs on the float64 IR for float16 and bfloat16. A site of overflow,
   near overflow, hidden overflow in a decomposed op, lost precision (fewer than 2 correct bits
   kept by an op) or a harmful underflow in a unit's module excludes that activation dtype for the
   unit (TS663); a site in code between units excludes it for every unit under that module. Fused
   norm internals that real kernels keep in float32 do not count.
5. **Candidates.** On the calibration inputs, with everything else in float32: each tie group in
   bfloat16 and float16, and each unit in each of its other precisions. Output error against float64
   and flips per candidate. For quantized candidates, verdict (a) is read from the layer's own
   first call check and verdict (b) is the float64 model with only that layer's format applied.
6. **Timing.** Each unit alone, on inputs of its recorded shapes in each candidate precision: GPU
   time from a CUDA graph of 5 calls, median of 7 replays. One cast per group input.
7. **Search.** Distinct candidate errors are the thresholds. For a threshold, the dynamic program
   runs over tie groups in execution order with the activation dtype as its state: cost is the
   sum of chosen unit times plus a cast wherever the dtype changes. The fastest plan (threshold
   infinite) is verified first; if it keeps the budget it is the answer. Otherwise float32
   everywhere is verified (if it fails, the budget is unreachable: TS662) and the threshold is
   bisected. Then up to 6 greedy refinements try the largest predicted savings one at a time,
   each verified.
8. **Verify.** Every plan that kept the calibration budget is timed end to end (CUDA graph
   replay); the fastest is run on the held out inputs. If it fails there, the calibration budget
   and the error threshold are both tightened below the failing plan's and the search runs again
   (TS661), up to 7 rounds; then the most accurate plan is gated, and only if that fails too is the
   budget unreachable (TS662).
9. **Report.** Per unit: the choice, its time, its error alone, its verdicts, and for each faster
   candidate why it was not taken. End to end: the plan and the baselines (all fp32, all tf32, all
   fp16, all bf16, fp16 except norms, torch.autocast fp16) with flips and GPU time.

## Measured

2026-11-24, RTX 5070 Ti (sm_120), torch 2.11, Triton 3.7.1, `scripts/precision_study.py`. The GPU
was shared with other jobs the whole time (other test suites, up to 15.8 of 16 GB used); rows
whose rounds disagreed are marked. GPU time per forward is the median of CUDA graph replay rounds
with the round range; eager is wall time per call without a graph. Flips are decisions that
differ from the float64 CPU reference, on calibration (cal) and held out (hold) inputs; the
budget is checked on both.

Models, all trained in code on synthetic tasks (no downloads): the MLP of the study (OnePassNorm
on readings near 30, Linear 1024, 4096, 4096, 100; 2 x 1024 rows each side; accuracy 0.52), the
Llama style decoder of docs/QUANT.md (dim 512, 4 layers, 8 heads on 4 KV heads, hidden 1536,
vocab 512, 800 steps; 2 x 8 x 256 tokens each side; next token accuracy 0.73), torchvision
resnet18 and a torchvision VisionTransformer (64 x 64, patch 8, 6 layers, width 384) on synthetic
class templates (2 x 256 images each side; accuracy 0.995 and 0.61).

**MLP, budget 0 flips** (planned in 5.4 s, 34 verification runs; plan 2 fp32, 1 tf32, 3 fp16):

| config | GPU us (range) | eager us | flips cal | flips hold |
|---|---|---|---|---|
| plan | 809 (800 to 810) | 847 | 0 | 0 |
| all fp32 | 1729 | 1779 | 0 | 0 |
| all tf32 | 1032 | 1090 | 2 | 0 |
| all fp16 | 605 | 682 | 359 | 324 |
| all bf16 | 530 | 588 | 422 | 380 |
| fp16 except norms | 604 | 663 | 359 | 324 |
| torch.autocast fp16 | 749 | 826 | 3 | 0 |

**MLP, budget 0.5%** (10 of 2048; planned in 4.4 s, 31 runs; plan 1 fp32, 4 fp16, 1 bf16):

| config | GPU us (range) | eager us | flips cal | flips hold |
|---|---|---|---|---|
| plan | 548 (543 to 548) | 620 | 8 | 4 |
| all fp32 | 1728 | 1812 | 0 | 0 |
| all tf32 | 1032 | 1078 | 2 | 0 |
| all fp16 | 601 | 670 | 359 | 324 |
| all bf16 | 531 | 588 | 422 | 380 |
| fp16 except norms | 605 | 675 | 359 | 324 |
| torch.autocast fp16 | 750 | 818 | 3 | 0 |

The planted layer: `norm` stays float32 in every plan; truestar range names the subtraction
(`aten.sub.Tensor`, precision_study.py:41, -0.9 correct bits in float16) for both 16 bit dtypes.
Every uniform 16 bit baseline and the rule by class flip a sixth of all decisions, because the
rule cannot see a norm that is not a torch norm class. torch.autocast keeps the norm in float32
(it only lowers matmuls) and is the only baseline near the budget; at 0.5% the plan is 1.37x
faster than it, at 0 flips only the plan and float32 hold the budget and the plan is 2.1x faster
than float32.

**Llama style decoder, budget 0 flips** (planned in 32.5 s: range 13 s, format loss 8 s; 192
verification runs; 4 tie groups, one per attention module; plan 5 fp32, 5 tf32, 30 fp16; the first
plan within budget on calibration flipped 1 held out decision, so the search tightened once,
TS661):

| config | GPU us (range) | eager us | flips cal | flips hold |
|---|---|---|---|---|
| plan | 1760 (1759 to 1761) | 3283 | 0 | 0 |
| all fp32 | 3520 | 4046 | 0 | 0 |
| all tf32 | 2241 | 2716 | 0 | 0 |
| all fp16 | 1682 | 3139 | 1 | 1 |
| all bf16 | 1849 | 2448 | 4 | 7 |
| fp16 except norms | 1788 | 3301 | 1 | 1 |
| torch.autocast fp16 | 1887 | 3055 | 1 | 1 |

At a zero flip budget the plan is 2.0x faster than float32 and 1.27x faster than the fastest
uniform baseline that also holds the budget (all tf32); all fp16 is 4% faster but flips a
decision on each side. Eagerly the plan is slower than all fp16 (3283 against 3139 us): 40 Python
wrappers per forward.

**Llama, budget 0.5%** (earlier run of the same code with the older timing, planning time not
comparable): plan 1661 us (1 fp32, 31 fp16, 7 fp8, 1 int8w), 7 and 7 flips of 2048 (allowed 10);
all fp16 1861 us, 1 and 1; all bf16 2029 us, 4 and 7; torch.autocast 2158 us; all fp32 3918 us.
The plan puts gate and up projections on the FP8 kernel (each passed verdict (a) at 0.90 of its
bound; verdict (b) says the format alone flips 4 to 6 decisions) and is 1.12x faster than all fp16.

**resnet18, budget 0** (52 units; plan 23 fp16, 28 bf16, 1 fp32): plan 2334 us (2331 to 2339),
all fp16 2931, all bf16 2806, torch.autocast 3242, fp16 except norms 3913, all tf32 4697 (one
round stalled to 1 s: a shared GPU), all fp32 6520; every config 0 flips on both sides. The
classifier is easy (accuracy 0.995), so the budget never binds and the plan is the fastest
measured: 1.20x over all bf16, 2.8x over float32. Planning took 130 s with other suites on the
GPU (timing 52 units in 3 to 4 precisions each).

**ViT, budget 0** (earlier run, older timing, planning 437 s under contention): plan all 58 units
fp16 6923 us (6791 to 7588), all fp16 6730, all bf16 7030 (4 and 1 flips, over budget), fp16
except norms 9160, torch.autocast 10456, all tf32 13735, all fp32 24222. Here a literal
`model.half()` is 3% faster than the plan and the planner says so (TS670): the class token and
position embedding belong to parent modules, stay float32 in the plan and cost two casts.

Reading the numbers:

* **The budget held on held out inputs in every run**, and twice the held out gate changed the
  answer (Llama at 0 flips, and the test where held out readings overflow float16).
* **Against the baselines:** where a uniform dtype breaks the budget (MLP, Llama at 0 flips) the
  plan is the fastest config that holds it, 1.27x to 2.1x over the next one. Where float16 is
  already safe (resnet18, ViT) the plan is float16 too and within a few percent of the literal
  uniform model, faster on resnet18, 3% slower on the ViT.
* **Speed of planning:** 4 to 5 s for the MLP and 32 s for the decoder on a quiet GPU; the CUDA
  graph timing first cost about 0.4 s per candidate (the clock warm up queued replays without
  waiting) until fixed, and minutes when other jobs filled the GPU's memory.
* **Tests can fail** (tests/test_precplan.py, 16 tests): with the range check skipped, the
  planted norm test fails (TS663 missing) while the budget test still passes, because the
  measured error alone also keeps the norm in float32: two independent reasons. With kernel
  verdicts ignored, the injected INT4 scale bug is used and its test fails. With the held out gate
  disabled, the overflow test fails. Without tie groups or without merging uncalled children, the
  unit tests fail.

## Limits

* **Not the compiled executor.** A plan runs as PyTorch modules (cuBLAS, cuDNN) plus Truestar's
  quantized kernels, wrapped per unit. Truestar's own compiled executor still takes one float
  dtype per program (docs/COMPILER.md); per node precision there is not built.
* **Eager overhead.** Each unit is a Python wrapper (a cast check and, for tf32, two flag
  writes). Under CUDA graph replay this costs nothing; run eagerly, a plan with many small units
  can be slower than a uniform model even when its GPU time is lower. Both numbers are reported.
* **Code between units keeps its dtype.** Parameters owned by a parent module (a ViT class token
  and position embedding) stay float32, so the activation stream after them is float32 again and
  the next unit casts back. A literal `model.half()` casts them too; on the ViT this is why all
  fp16 can be a little faster than the plan (TS670 says so when it happens).
* **Inputs used.** The range check runs on the first calibration batch (at most 16 rows) plus its
  static bounds; format loss (verdict b) on the first calibration batch (at most 128 rows). Values
  the calibration set never drives high are covered only by the static bounds.
* **The held out gate is used more than once** when tightening: the inputs never steer a search,
  but a plan that passes after several rounds has been selected against them a little. A third
  set would remove that; the planner does not ask for one.
* **Decisions are argmax of the last dim.** A regression output only gets its relative error;
  near ties are not separated from real flips (the budget counts every flip).
* **One GPU, shared.** RTX 5070 Ti (sm_120), torch 2.11, other jobs running. Timings are medians
  of CUDA graph replays with their round range; a round that stalls by more than 1.5x is measured
  again and a note says when the rounds still disagree. Planning time depends on how busy the GPU
  is (see the table).
* **Quantized candidates are Linear layers only**, weight only INT8 and INT4 and FP8 W8A8; NVFP4
  and MXFP4 are not offered yet. Weight only formats only win at small row counts (decode), so at
  prefill sizes the planner rarely picks them, as measured.
* **Not compared against TensorRT ModelOpt autocast**: ModelOpt is not installed and TensorRT
  is not in this venv. The baselines are what PyTorch users do: uniform dtypes, a rule by class,
  torch.autocast.
* **No MCP tool** for the planner yet; the CLI and the Python API are the interface.
