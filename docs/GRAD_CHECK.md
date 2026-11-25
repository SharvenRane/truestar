# Backward check (`truestar check --backward`)

Pain P18 (docs/PAIN_RESEARCH_2.md, N6): BF16 and FP8 runs diverge late with no NaN because a trusted
kernel's backward returns subtly wrong gradients. One report saw the FlashAttention 3 gradient norm
grow a thousandfold after 25B tokens. Loss and grad norm monitors fire late by design, and
`torch.autograd.detect_anomaly` only looks for NaN. The forward check (`truestar check`) cannot see a
bug that lives only in backward. This is its backward counterpart.

```
truestar check model.py --backward                       # every backend here
truestar check model.py --backward --backends cpu,cuda --steps 20 --format agent
```

```python
import truestar
report = truestar.grad_check(model, inputs, backends=["cpu", "cuda"], steps=20)
print(report.to_text())
doc = report.to_agent()        # the agent JSON below
```

## Method

1. **Reference.** The model runs in float64 on CPU, in eval mode (dropout off), forward and backward
   through a fixed loss: by default the mean squared error to a fixed random target, or the caller's
   `loss_fn(outputs, targets)`. Every module call is recorded with its inputs copied before forward,
   and a tensor hook on each output records the upstream gradient that reached it. Floating model
   inputs take gradients too, so a model without parameters is still checked.
2. **Isolated backward replay.** Each replayed call (leaves, and parents that run ops of their own once
   every module inside them matched, the same rule as the forward check) runs forward and backward
   alone with the reference inputs and the reference upstream gradients. That yields the gradient of
   every floating input and every trainable parameter of that call. It runs in float64 on CPU (the
   answer), and in the model dtype on each backend. The error a call shows is its own, never
   inherited from the modules after it.
3. **Thresholds per gradient tensor.** The precision's floor (float32 1e-4, TF32 5e-3 for modules that run a matmul or convolution, float16 5e-3,
   bfloat16 4e-2), never below 10x the larger of two noise floors measured on the same tensor:
   * the CPU replay in the model dtype, as in the forward check (dropped when it is itself far off,
     so a broken CPU kernel cannot hide a bug);
   * an emulated replay: float64 math with weights, inputs, upstream gradients and every op result
     rounded to the model dtype (and convolution inputs rounded to TF32 when the backend enables
     TF32). That is what a correct kernel which rounds once per op computes. It captures error that
     is intrinsic to the precision: ReLU masks and max pool routes that flip because inputs round,
     cancellation in reductions. It is also the only floor available when the backend under test is
     the CPU itself.

   Both floors are capped the way the forward check caps its CPU floor: a floor above 20x the
   precision floor is dropped. An emulated floor that large means the module's own math loses its
   digits in this precision (the one pass variance in examples/check_model.py), which is the
   divergence the forward check reports, so it must not lift the gradient threshold above it. The
   only exception is a gradient that is zero in exact math, whose value is rounding by nature.

   A gradient that is zero in exact math (attention's key bias, which softmax cancels) is measured
   against 1e-6 of the largest gradient of the same call instead of against itself.

   Where the emulated floor is dropped for being over the cap, the backend gradient is also
   compared with the emulated (correctly rounded) gradient itself, on the same scale (distance over
   the reference norm) and against the same threshold. If it lands there, it is OK and the report
   says so: a correct kernel in this precision computes that value, and the gap to float64 comes from
   rounding, typically a ReLU mask or max pool route that flips on a near tie. A wrong kernel does
   not land on the rounded run, so it stays flagged. Found on resnet18 in float16 at 32x32
   (2026-11-24): one draw has a layer4.1 ReLU input of -5.3e-4 in float64 and +1.2e-4 in float16,
   so the whole model replay is 1.1e-1 off float64 at layer4.1.bn2.weight while the float16 CPU
   kernels are 9.5e-4 from the rounded run.
   **Parents and near tie mask flips** (owner's rule, 2026-11-24). A parent (whole model) replay in
   float16 compounds every child's rounding, and where a ReLU input is a near tie the backend, the
   rounded run and float64 can each take another branch. Such a parent is not DIVERGENT when every
   leaf replay is clean and the deviation traces to those flips; it is SUSPECT (TS188) with a "near
   tie mask flip" note naming the ReLUs and the number of flipped elements. "Traces to" is checked
   (`src/truestar/gradties.py`): the parent's forward runs on the backend and in float64 recording
   every ReLU with its module; both must run the same ReLUs; every element whose mask differs must
   be a near tie (its float64 input within twice the precision floor of zero, relative to that
   ReLU's input RMS); and the rounded run is replayed again with the backend's masks forced into its
   ReLUs, where the backend's gradients must land within the usual thresholds. A wrong backward
   does not land there and stays DIVERGENT; if any leaf is not clean, the excuse is withdrawn.

4. **Forward versus backward.** The forward output of the same replay is compared too. A module whose
   gradients diverge while its forward matches has a backward bug (TS180). If the forward diverges as
   well, the report says so (TS182) and points at the forward check. The forward part uses the forward
   check's own threshold, so a module the forward check calls divergent is never fwd OK here, and its
   gradients are reported divergent too.
5. **Blame.** Gradients flow from the last module in forward order to the first, so the over
   threshold call that runs first in backward is named.
6. **Op drilldown.** The blamed call's backward alone is traced under a `TorchDispatchMode` (the
   forward runs outside it) on the reference and on the backend. The op sequences are aligned and
   each op whose output looks wrong is rerun alone with the reference inputs, inside a backward pass
   as it ran in the real graph. Each op's threshold also takes an emulated floor (the same inputs,
   exact math, result rounded) so cancellation in a sum is not blamed. The first op whose own error
   crosses its threshold is named together with the autograd node that ran it (for example
   `aten.mm.default` in `AddmmBackward0`).
7. **Training trend (`--steps N`).** N SGD steps from the same weights run on the float64 reference,
   on an emulated run (float64 math rounded to the model dtype, weight updates included), on CPU in
   the model dtype for accelerators, and on the backend. Per step it records the total and per module
   gradient norm. Low precision runs drift from float64 even when every kernel is right (small
   updates round away), so the backend is judged against the emulated run: when its gap at the last
   step is over the threshold, more than twice the gap at step one, and more than 10x the gap between
   two correct runs, the module is flagged (TS183). The reported failure mode is slow accumulation, and
   a wrong gradient that is too small to cross a threshold in one step can still show here.

Repros: with `--repro-dir`, the blamed call gets `grad_repro.py` (forward and backward of that module
alone with the saved inputs and upstream gradients, compared per gradient to float64; exit 1 means it
reproduced; needs only PyTorch) and, when an op was named, `op_repro.py` for that single op.

## Fault injection

`truestar.gradfault.BackwardOpFault(op, device_type, amount, kind, node, ramp)` corrupts one ATen op
only while autograd runs a backward pass, only on `device_type`, and never on float64 tensors, so the
reference stays exact even when the backend under test is the CPU. `kind="row_leak"` on
`aten._softmax_backward_data` adds `amount * y * rowsum(y * g)`, a softmax backward that leaks its row
sum term. `ramp` adds that much corruption per backward pass. The agent JSON lists active injectors in
`injected_faults`.

## Codes

| code | severity | slug | meaning |
|---|---|---|---|
| TS180 | error | grad_divergence | a module's own backward is wrong while its own forward matches |
| TS181 | warning | grad_suspect | own gradient error over threshold, under the divergent cutoff |
| TS182 | error | grad_forward_divergent | gradients diverge and the module's forward diverges too |
| TS183 | warning | grad_norm_drift | gradient norm drifts from a correct run over the training steps |
| TS184 | error | backward_replay_failed | the module raised when its backward was replayed alone |
| TS188 | warning | grad_near_tie_mask_flip | a parent is off float64 only through ReLU masks that flip at near ties (every leaf clean, a rounded run with the backend's masks matches) |

TS002 (model failed on the backend), TS001, TS005, TS020 and TS021 keep their meanings.

## Agent JSON

Same envelope as docs/AGENT_INTERFACE.md (`schema_version` 1, `command` `"check"`) plus
`"mode": "backward"` and `loss`. Each backend carries `blamed_module` with `gradient` (for example
`"param weight"` or `"input 0"`), `rel_err`, `threshold`, `ratio`, `norm_ratio`, `cosine`,
`forward_status`, `forward_rel_err`, `backward_only`, `source`, `user_source`; `blamed_op` with
`op`, `node`, `inputs`, `rel_err`, `threshold`; `grad_norm_ratio` per checked call; `trend`
(`total_ratio` per step, `control_ratio` per control, `flagged`); `repro`, `op_repro` and
`injected_faults`. `next_actions` kinds: `confirm_op_repro`, `confirm_repro`, `control_repro`,
`injected_fault`, `check_forward`, `mitigate_fp32`, `inspect_source`, `pin_torch`,
`watch_grad_norm`, `inspect_failure`, `accept_near_tie` (TS188: no kernel to fix).

## Measured (2026-11-24, RTX 5070 Ti, torch 2.11, CUDA 12.8, Windows)

* Clean models report OK on CPU and CUDA in float32 and bfloat16: MLP, a transformer block with its
  own softmax attention, `nn.TransformerEncoderLayer`, resnet18 at 32x32 (also float16), and resnet50
  at 64x64 on CUDA in float32 with TF32 convolutions, float16 and bfloat16.
* An `aten.mm` fault only in backward is blamed on `fc2#0`, op `aten.mm.default` in
  `AddmmBackward0`, while `truestar check` reports OK, on CPU (float32, bfloat16) and CUDA (bfloat16).
* A softmax backward row sum leak in a transformer block is blamed on the attention module's own
  code, op `aten._softmax_backward_data.default` in `SoftmaxBackward0`; its children are clean and the
  total gradient norm moves by 0.2% (cosine 0.9993), which a norm monitor would not notice.
* A backward error that grows by 0.002 per backward pass is not over threshold in the bfloat16 single
  step replay but is flagged by `--steps 20` (gap to a correct run 0.0075 at step 1, 0.049 at step 20).
* Scaled dot product attention backward on the GPU in bfloat16 at (2, 8, 1024, 64), causal and not,
  and (1, 16, 2048, 128): OK. On this machine PyTorch picks the memory efficient kernel; no flash
  attention kernel is available in this Windows build. Relative error of dq, dk, dv against float64:
  4.4e-3, 3.8e-3, 1.8e-3 (efficient), 4.6e-3, 4.5e-3, 3.8e-3 (cuDNN), 4.2e-3, 4.2e-3, 3.4e-3 (math),
  against an emulated correct bfloat16 floor near 1e-2; norm ratio within 1e-4, cosine 0.99999.

## Limits

* Gradients are judged by their whole tensor relative error only. The forward check also judges
  each output's worst element and worst slice (docs/TOLERANCE.md, "truestar check"); on gradients
  that measure failed 6 tests of tests/test_gradcheck.py, clean resnet18 backward checks among them
  (input and weight gradients are sparse by construction: max pool routes, ReLU masks, near ties),
  so it is off here until it is calibrated for gradients. A backward that is wrong in one row of a large gradient can therefore pass.
* Parent replays compound their children's rounding, and in float16 at tiny shapes (batch 2, 1x1
  spatial in layer4) a near tie that flips differently in the backend and in the rounded run gave a
  false TS180 at the parent. Measured on resnet18 float16 at 32x32 with portable uniform draws
  (`scripts/probe_fp16_parent_ties.py`: every weight, buffer and input from float64 `torch.rand`,
  seeds 0 to 79, CPU only): 4 of 80 draws DIVERGENT before the near tie rule, 0 after (those 4 are
  SUSPECT, 10 to 22 flipped elements each, all within 0.18% of the input RMS of zero). Injected
  faults stay DIVERGENT on all 80 draws before and after: a 30% `aten.threshold_backward` fault, a 30%
  `aten.convolution_backward` fault (both caught at a leaf), and a 15% wrong backward in the model's
  own code that only the whole model replay sees. Such draws still exit 1 (SUSPECT), by the owner's
  decision. Only ReLU masks are forced; a max pool route that flips is not explained this way.

* Training mode (`--train`, `train=True`, since 2026-11-24, docs/RANDOM.md): batch statistics,
  keyed dropout masks identical on every backend and in the backward, running statistic updates
  compared with the forward. A 10% fault in the batch norm training backward (CPU
  `native_batch_norm_backward`, CUDA `cudnn_batch_norm_backward`) is TS180 at the batch norm in
  training mode and invisible in eval mode. Training mode parents keep their measured noise floors
  uncapped (batch statistics couple every element: plain PyTorch float32 against float64 on resnet18
  at batch 16 is up to 1.1e-2 off, eval mode 6.8e-7); leaves keep the cap.
* Optimizers other than plain SGD, gradient scaling and FP8 are not modelled. FP8 needs an FP8
  rounding mode and FP8 kernels to test; neither exists here.
* Op drilldown replaces every tensor argument with the reference value, including Python scalars
  that dispatch wraps as tensors; a wrong constant in a custom `autograd.Function` backward is
  therefore blamed at the module (with its source), not at an op.
* Backward ops that only exist on the backend (fused attention backward kernels) cannot run on CPU,
  so they are checked through their module, not alone.
* The trend loop reuses one batch; real data shifts and learning rate schedules are not modelled.
* Cost: about two float64 replays per call plus the backend replay. resnet50 at 64x64 checks in about
  14 s (float32) and 26 s (bfloat16) on CUDA; no budget is set yet.
