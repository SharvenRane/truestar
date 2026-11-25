# `truestar fix`: verified remedies

`truestar check` finds the problem and hands over a repro. `truestar fix` goes one step further: it
applies a workaround without changing the user's code, reruns Truestar's own check with the
workaround in place, and only reports it as working when that check comes back clean (or
measurably better, with the numbers). Every remedy reports what it costs.

Code: `src/truestar/fix/` (`table.py` remedy table, `mode.py` how remedies are applied,
`engine.py` search and verification, `precision.py` INT8 layer selection, `output.py` text and
agent JSON, `cli.py` the command). Tests: `tests/test_fix.py`.

## Op remedies

```python
import truestar.fix

report = truestar.fix.fix(model, x)            # check, then find and verify remedies
print(report.to_text())

with truestar.fix.patched(report):             # scoped: only the blamed op, only on its device
    y = model(x)

handle = truestar.fix.apply(report, model=model)   # every forward of this model, until
handle.remove()                                     # handle.remove()
```

```
truestar fix model.py --backends cuda                  # text
truestar fix model.py --format agent                   # agent JSON on stdout
truestar fix model.py --format json --json full.json   # full JSON
truestar fix model.py --mechanism library              # kernel override instead of a mode
truestar fix --list                                    # the remedy table as JSON
```

The model file is the same as for `truestar check`, plus an optional `get_context()` returning a
context manager every run happens inside (for example a `truestar.testing.OpFault` that simulates
a bad kernel, see `examples/fix_demo.py`).

Exit codes: 0 when every backend was already clean or is resolved, 1 when a backend is improved,
unresolved or not applicable, 2 on usage errors. Exit 0 after a fix means the remedy works, not
that the kernel is correct: the upstream report is still listed.

### Method

For each backend where `truestar check` names an ATen op (`first_divergent_op`):

1. The candidate decompositions for that op are taken from the table in order.
2. Op level filter: the op alone is run on the backend with its saved reference inputs and the
   remedy applied, and compared with the float64 answer at the op's own threshold. A candidate
   that fails here is not tried further.
3. Model level verification: `truestar.check` is rerun on that backend with the remedy (and every
   remedy already accepted) applied. The candidate wins only when the module that was blamed is
   now OK and the remedy actually replaced at least one call during that check.
4. If the backend is still not clean and another op is now blamed, the loop repeats for it
   (at most 4 ops).
5. Cost: the forward pass on the backend is timed with and without the remedies (median of 20
   runs after 3 warm up runs by default).

Outcomes and codes:

| outcome | code | meaning |
|---|---|---|
| resolved | TS090 | the rerun check on this backend is clean with the remedies applied |
| improved | TS091 | the output error fell at least 2x or real flips fell, but the check is not clean |
| unresolved | TS092 | no candidate cleared the blamed module; the op repro path is the upstream report |
| not applicable | TS093 | no single op is blamed (module code, a device branch, a model that fails to run) |

### Mechanisms

* `mode` (default): a `TorchDispatchMode` that replaces the op only when its tensors are on the
  blamed device type. It sits above the backend kernels, so it also clears Python level faults
  such as `truestar.testing.OpFault` when entered inside them. Cost: Python dispatch on every op
  while active.
* `library`: a `torch.library` kernel on the device's dispatch key (CUDA, MPS, XPU, CPU), for the
  whole process. This is the right shape for a real kernel bug such as `aten.linear.default` on
  MPS, and only the replaced op pays a Python cost. It cannot clear a fault that a dispatch mode
  injects, because modes run before kernels; `fix(..., mechanism="library")` then reports
  unresolved honestly (tested). `remove()` restores the previous kernel.

Both act below autograd, so gradients keep the original op's derivative formula.

### Remedy table

| op | candidates in order | notes |
|---|---|---|
| `aten.addmm.default` | `mm_add`, `bmm_add`, `mul_sum` | `mul_sum` skipped above 2^26 product elements |
| `aten.linear.default` | `matmul_add`, `addmm`, `linear_nobias_add`, `mul_sum` | built in remedy for the macOS 26 MPS finding: `x @ w.T + b` was correct in the probe while `F.linear` with bias was not. `linear_nobias_add` calls linear itself, so mode only |
| `aten.mm.default` | `bmm`, `mul_sum` | |
| `aten.bmm.default` | `mm_loop`, `mul_sum` | loop limited to batch 4096 |
| `aten._softmax.default` | `explicit` (stable exp of x minus max), `explicit_fp32` | fp32 variant for float16 and bfloat16 only |
| `aten._log_softmax.default` | `explicit`, `explicit_fp32` | as above |
| `aten.native_layer_norm.default` | `explicit` | float32 statistics for half inputs |
| `aten.gelu.default` | `explicit` | erf or tanh form |
| `aten.convolution.default` | `im2col` | 2D, not transposed, any groups |

A test runs every entry against the real op in float64 on CPU (relative tolerance 1e-10), so a
table entry can never be the source of a wrong result. `fix(..., table=...)` takes a custom table.

### MPS (prepared, not run)

There is no Mac here. The MPS case uses the same code path as a test on CUDA: CUDA normally
decomposes `aten.linear` into `t` and `addmm` before any dispatch mode sees it, so the test
registers a CUDA kernel for `aten.linear` (as MPS has its own), faults it with
`OpFault("aten.linear.default", "cuda")`, and checks that `truestar fix` blames
`aten.linear.default` as an op the reference never runs, and resolves it with the first MPS table
entry, `matmul_add`. A second test makes that fused kernel itself wrong (a kernel level fault) and
resolves it with the `library` mechanism, which is what a real MPS run would use. On a Mac:
`truestar fix model.py --backends mps --mechanism library`. The upstream report draft is
`docs/upstream/mps_linear_bias_DRAFT.md`, listed in the report for that op and device.

## Precision remedies (INT8)

```python
plan = truestar.fix.select_precision(model, eval_x, calib_x, target_flips=4)
print(plan.to_text())      # also plan.as_dict(), plan.to_agent()
```

```
truestar fix model.py --int8 --target-flips 4    # get_calibration() optional in the model file
```

The INT8 model is simulated on CPU with fake quantization (weights per output channel symmetric
INT8, each quantized layer's input per tensor UINT8 from calibration min/max) and every prediction
is compared with the float64 model. Offline, no network. Then:

1. Units are the Conv and Linear layers, or any module names (such as the NPU study's block taps).
2. Ranking: from per layer error data when given (a mapping, a `LayerBlame` from
   `truestar.remote.agreement`, or rows; each tap is credited only with the error it adds over the
   largest before it), otherwise measured with one simulation per unit.
3. Units are kept in float in ranking order, simulating the whole model after each one, until the
   flips are at most the target.
4. Each kept unit is dropped again, least useful first, if the target still holds without it. The
   final set is simulated once more and that is what is reported.

Reported: flips and output error before and after, kept units and layers, and the share of
multiply accumulates kept in float. Codes TS094 (target met in simulation) and TS095 (not met).

### Vendor execution hook (not wired)

`PrecisionPlan.qai_hub_hint()` returns the layer names to leave out of INT8. The place it would be
used is the quantize step in `truestar.remote.qai_hub_quant` (`HubSession.quantize`), followed by
the existing compile and inference jobs and `truestar.remote.agreement` to measure the flips on the
phone. That module is not changed here. Whether AI Hub's quantize job accepts per layer exclusions,
and under which option name, has not been checked; it is an open item, not a claim.

## Measured (2026-11-24, RTX 5070 Ti shared with other jobs, torch 2.11, CUDA 12.8)

| case | before | after | remedy | forward cost |
|---|---|---|---|---|
| `examples/fix_demo.py`, OpFault 5% on addmm | output error 1.5e-1, 3 of 32 flips | 1.6e-7, 0 flips | `mm_add` | 0.267 to 0.263 ms (noise) |
| ResNet18 4x3x64x64, OpFault 5% on convolution | 5.2e-1 | 6.7e-7 | `im2col`, 79 calls | 3.86 to 5.46 ms (+41%) |
| ResNet18, fused CUDA linear plus OpFault 5% (MPS path) | 5.0e-2 | 3.7e-4 | `matmul_add` | within noise |
| ResNet18, fused CUDA linear kernel wrong by 5%, library mechanism | 5.0e-2 | 3.7e-4 | `matmul_add` | 1.85 to 1.74 ms (noise) |
| MLP with addmm, mm, bmm and mul all faulted | divergent | unresolved, TS092 | none of 3 passes the op filter | none |

Overhead on a clean ResNet18 (no fault, so this is what a real kernel bug would pay), median of 50:
addmm remedy +19% as a mode and +4% as a library kernel; convolution remedy +67% as a mode and +51%
as a library kernel (im2col is slower than cuDNN). The GPU was in use by other jobs, so single digit
percentages are within noise.

INT8, synthetic ResNet18 task of docs/NPU.md (256 inputs, CPU simulation): 15 flips against
float64 with every layer quantized (the S24 CPU had 16 in the study). Target 4: 6 of 21 layers in
float, 26.3% of MACs, 4 flips, 36 simulations, 52 s. Target 0: 18 of 21 layers, 80.9% of MACs,
0 flips, 100 simulations. Flips are not monotone in the kept set, which is why every step is
simulated rather than predicted.

## Limits

* Only the op `truestar check` names is remedied, one op per round, at most 4 rounds.
* An op remedy cannot fix a module's own device branch or code outside modules (TS093).
* The mode mechanism costs Python dispatch on every op while active; the library mechanism costs
  Python only for the replaced op but is process wide.
* Ops the reference never runs are verified against their float64 CPU result for the backend's own
  inputs, as in the op drilldown.
* MPS and ROCm are untested here; the MPS path is exercised on CUDA as described above.
* The INT8 simulation is the intended arithmetic, not a vendor runtime. On the phone NPU the study
  found extra rounding differences the simulation does not model, so a plan must be measured on
  the device before it is trusted there.
