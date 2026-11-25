# Record and diff across environments (P13)

Upgrading torch, CUDA, cuDNN, Triton or the driver can change a model's outputs with no error.
pytorch#181169 is the sharp example: torch 2.12 moved vLLM gsm8k accuracy from 0.54 to 0.48, bit
identical over three runs. People notice the end metric and then bisect versions by hand.

`truestar record` stores what every module saw and produced in the old environment.
`truestar diff` compares that against another recording or against the current environment,
names the first module whose own result changed, then the ATen op inside it, and says what kind of
change it is.

## Commands

```
truestar record model.py -o before.tsrec                 # first accelerator, else cpu
truestar record model.py -o before.tsrec --backend cuda --ops all
truestar record model.py -o small.tsrec --outputs-only   # outputs and hashes only
# ... upgrade ...
truestar diff before.tsrec --replay                      # replay here, every module alone
truestar diff before.tsrec after.tsrec                   # two recordings, no model code needed
truestar diff before.tsrec --replay --format agent       # versioned JSON with next_actions
```

`model.py` is the same file `truestar check` takes (`get_model()`, `get_inputs()`, optional
`get_kwargs()`). The recording names it, so `--replay` finds it again; `--model` overrides.

Exit codes. `record`: 0 written, 2 usage error. `diff`: 0 when the verdict is identical, noise or
precision change, 1 when it is divergent or the runs are not comparable, 2 usage error. With
`--strict`, a precision change also exits 1.

## What a recording holds

* Per module call: the inputs, copied before forward so in place ops cannot change them; the
  outputs; a hash of the inputs; a hash of the module's parameters and buffers; its nesting.
* Tensor identity: a tensor passed twice (`attn(y, y, y)`) is one object again on replay.
  nn.MultiheadAttention takes its fused kernel only when query, key and value are the same object,
  so losing identity would replay a different code path.
* With `--ops all` or `--ops name,name`: every ATen op run under those modules, with its inputs and
  outputs and the module path it ran under.
* The plain model output, from a run without hooks. Forward hooks switch off fused fast paths
  (TransformerEncoderLayer checks for them), so the instrumented run can take another code path.
  When the two disagree, diff reports TS071 and the baseline line uses the plain path.
* The environment fingerprint: Python, platform, torch and its git hash, CUDA runtime, HIP,
  cuDNN, Triton, numpy, driver (from nvidia-smi), GPU name and capability, TF32 and
  fp32_precision settings, reduced precision reductions, cuDNN benchmark and deterministic flags,
  deterministic algorithms, SDPA backends, thread count and the relevant environment variables.
  Each key has a kind: version, precision, algorithm or hardware.

Format: magic, a sorted key JSON header, deduplicated raw tensor blobs and a sha256 of the whole
file. Nothing depends on time, host or process, so identical runs give identical bytes (checked on
CPU and on CUDA). A flipped bit, truncation, a foreign file and an unknown version are each refused
with the reason (TS065). `--max-mb` bounds the size (TS070, default 1024 MiB).

Measured sizes, resnet18 at 2x3x64x64 on CUDA: 5.3 MiB without op traces, 52 MiB with `--ops
all` (op inputs include every weight), 7.5 MiB with `--ops all --outputs-only`. For a plain module
recording, outputs only saves little, because a module's inputs are mostly an earlier module's
outputs and are stored once.

## How diff decides

Replay (`--replay`, the main mode): every recorded module call is run alone in the current
environment with its recorded inputs, under the same do nothing hooks as the recording. Its own
change is the difference between its current output and its recorded output, so nothing is
inherited from earlier layers. Each changed module is also run on CPU in float64 and in its own
dtype, which calibrates the noise band exactly as `truestar check` does, and shows which of the
two results is closer to exact math.

Two recordings: a module whose inputs hash the same in both runs is isolated exactly, because
everything before it was bitwise identical. Modules after the first change see different inputs;
they are reported as not isolated (TS068), and their classes still count toward the verdict so a
worse change hidden downstream is never passed silently. An outputs only recording is replayed
this way too: the model runs once and modules are isolated by input hash.

Each module change gets one class:

| class | rule |
|---|---|
| identical | bitwise equal outputs |
| noise | relative L2 change within the dtype band (float32: 1e-4, never below 10x the calibrated CPU noise) |
| precision | beyond the noise band, within the band of a precision mode the fingerprints show was switched (TF32: 5e-3), on a CUDA device |
| divergent | anything else |

A module that ran submodules gets the parent allowance of `truestar check` (a suspect level change
still counts as the lower class), since it compounds its children's precision. A module owns its
change only when that change is worse than every change owned by a module inside it, so a block
is never blamed for its child. The blamed module is the first owner of the worst class.

Op drilldown runs inside the blamed module. With a recorded op trace, the module is traced again
here, its ops are aligned with the recorded ones by name, and each op that moved is fed its
recorded inputs alone: the first op whose own change is divergent (else precision) is named.
Without a trace, opdrill compares the module's ops with float64 CPU math in the current
environment, which names a wrong kernel but cannot compare against the old environment op by op;
the report then suggests recording with `--ops`.

`--repro-dir` (default `truestar_repro/`) receives `op_repro.py` and `op_payload.pt` for the
blamed op: the op, its recorded inputs and its recorded output, runnable with plain PyTorch.
Exit code 1 means the op's result moved beyond the band. That is the upstream bug report.

## Example

resnet18 recorded on an RTX 5070 Ti with TF32 off, replayed with TF32 on:

```
Verdict: PRECISION CHANGE: explained by a precision setting that differs
First module whose own result changed: layer1.0.conv1#0 (Conv2d)
  class precision, own change 2.98e-04 (noise band 1.0e-04, precision band 5.0e-03), measured by isolated replay
  vs float64 math: recorded 7.27e-07, current 2.98e-04 (the recorded result is closer to exact)
  op: aten.convolution.default (op 1 of 1), class precision, change 2.98e-04 vs band 1.0e-04, fed its recorded inputs
Modules: 54 identical, 0 noise, 11 precision, 0 divergent, 11 inherited
Final output change: 4.21e-04 (max abs 1.39e-03)
Baseline, torch.allclose(final outputs, rtol=1e-05, atol=1e-08): FAIL; it names no module and no op
[TS061] module layer1.0.conv1#0 (Conv2d) own output changed by 2.98e-04 (noise band 1.0e-04, precision band 5.0e-03); explained by flags.cudnn.allow_tf32, ...
```

The same recording replayed with `OpFault("aten.addmm.default", "cuda", amount=0.5)` active:

```
Verdict: DIVERGENT: a module's own result changed beyond what precision explains
First module whose own result changed: fc#0 (Linear)
  class divergent, own change 5.00e-01 (noise band 1.0e-04), measured by isolated replay
  vs float64 math: recorded 1.14e-07, current 5.00e-01 (the recorded result is closer to exact)
  op: aten.addmm.default (op 2 of 2), class divergent, change 5.00e-01 vs band 1.0e-04, fed its recorded inputs
Modules: 74 identical, 0 noise, 0 precision, 1 divergent, 1 inherited
Baseline, torch.allclose(final outputs, rtol=1e-05, atol=1e-08): FAIL; it names no module and no op
```

Record took 0.64 s and replay 0.31 s for resnet18 without op traces; the CLI replay with op traces
took 6.8 s including process start and the torch import.

## Agent JSON

`truestar diff ... --format agent` prints one JSON document with the envelope of
docs/AGENT_INTERFACE.md (`schema_version` 1, `command` "diff", strict JSON). Fields: `mode`,
`recorded`, `current`, `model` (`name`, `dtype`, `device`), `status` (`identical`, `noise`,
`precision`, `divergent`, `not_comparable` or `error`), `exit_code`, `codes`, `issues`,
`environment_changes` (`key`, `recorded`, `current`, `kind`), `precision_explained`,
`blamed_module` (key, class, rel_err, bands, method, float64 errors, `source`, `user_source`),
`blamed_op` (op, class, rel_err, band, `source` = `recorded_trace` or `float64_reference`,
inputs), `counts`, `modules_changed`, `final_output`, `baseline`, `op_repro`, `notes`,
`next_actions`. `truestar record --format agent` prints the recording path, size, sha256 and the
fingerprint.

`next_actions` kinds: `confirm_op_repro`, `pin_version` (per changed version key, with a pip
command for torch), `report_upstream`, `restore_precision_flag` (per changed precision key),
`inspect_source`, `injected_fault`, `record_op_trace`, `replay_for_isolation`, `check_weights`,
`check_inputs`, `inspect_replay_failure`.

## Codes

| code | severity | slug | meaning |
|---|---|---|---|
| TS060 | error | env_divergence | a module's own output changed beyond the dtype band and any precision explanation |
| TS061 | warning | env_precision_change | changed within the band of a precision setting that differs |
| TS062 | info | env_numerical_noise | changed within the dtype band |
| TS063 | info | env_changed | the fingerprints differ |
| TS064 | error | module_state_differs | different parameters or buffers: not the same model |
| TS065 | error | record_invalid | missing, truncated, modified or unknown version recording (exit 2) |
| TS066 | warning | call_set_differs | different module calls ran |
| TS067 | warning | replay_failed | a module call could not be replayed |
| TS068 | info | isolation_limited | changed modules saw different inputs; replay isolates every module |
| TS069 | error | inputs_differ | the two recordings fed different model inputs |
| TS070 | error | record_too_large | the recording would exceed `--max-mb` (exit 2) |
| TS071 | warning | instrumented_path_differs | recording hooks changed the model's code path |

## Limits

* Only forward passes in eval mode without gradients are recorded.
* Precision explanation is per run, not per op: a TF32 flag change explains a change in any CUDA
  module, including one that has no matmul or convolution. A change of TF32 size with no flag
  change (a new library picking reduced precision kernels, the pytorch#157274 pattern) is
  divergent, and TS060 says the size matches TF32.
* An OpFault on CPU also corrupts the float64 CPU replay, so the "closer to exact" line and the
  float64 op fallback are wrong in that test setup. Real CPU kernel regressions affect float64
  kernels only when they share code.
* Fused fast paths that forward hooks switch off (TransformerEncoderLayer) are not measured per
  module; TS071 flags the case, and the baseline line uses the plain path.
* Modules with random ops in eval mode, inputs that are not tensors or plain values (opaque
  objects are reported as TS067), sparse and quantized tensors are not replayed.
* Two real software versions were not compared on this machine (one venv, no downloads). The
  acceptance uses an injected op change and TF32, both within one process.
