# Profile: GPU time per compiled step, mapped to IR ops and source lines

Author: Sharven Rane. First written 2026-11-24. Code: `src/truestar/compiler/profile.py`, tests in
`tests/test_profile.py`, benchmark in `scripts/bench_profile.py`.

Research item R14 (GAP_PYTORCH_JAX.md) and competitor item 14 (GAP_COMPETITORS.md). Rows served:
P12 (agent output) and never introduce #4 (overhead is measured, so it has to be measurable per
kernel); P6 and P8 work uses it to see where time goes.

## Use

```
truestar profile model.py                         # text table
truestar profile model.py --format agent          # one JSON document on stdout
truestar profile model.py --crosscheck            # also compare with torch.profiler
truestar profile model.py --dtype float16 --tf32 --sentinels nan,div --iters 50 --json out.json
```

`model.py` defines `get_model()` and `get_inputs()` like every other command. The model is
imported to the IR, compiled for CUDA and called: once (compile, Triton compile, the float64
checks; reported apart as `first_call`), `--warmup` more times, then `--iters` timed calls.
From Python: `truestar.compiler.profile.profile_program(compiled_program, inputs, iters, warmup,
crosscheck)` returns the same data as a dict.

```
truestar profile: mlp_model on NVIDIA GeForce RTX 5070 Ti, float32, 30 calls
  per call: GPU busy 0.174 ms, steps sum to 0.177 ms in 3 distinct steps; as run 0.179 ms, host issue 0.121 ms
  events: 90 of 90 step runs timed
  first call: compile 0.09 s, first run 0.34 s (checks against float64 0.19 s), not in the times below

   share  ms/call calls  kind     shape         GB/s TFLOP/s  ops  [line]
   43.8%   0.0774     1  kernel   contraction    230     6.9  linear + relu  [linear.py:134, activation.py:143]
   33.0%   0.0584     1  kernel   contraction    154     4.5  linear  [linear.py:134]
   23.2%   0.0411     1  kernel   contraction    224     6.5  linear + gelu  [linear.py:134, activation.py:816]

  by source line (ms per call; shared = in a kernel fused with other lines):
    linear.py:134   own 0.0584  shared 0.1185  return F.linear(input, self.weight, self.bias) (module 2: Li
    activation.py:143   own 0.0000  shared 0.0774  return F.relu(input, inplace=self.inplace) (module 3: ReLU)
    activation.py:816   own 0.0000  shared 0.0411  return F.gelu(input, approximate=self.approximate) (module 1

  torch.profiler crosscheck: 90 of 90 Truestar kernel launches seen; GPU kernel time 0.170 ms per call against 0.177 ms from events
```

For an `nn.Sequential` of library layers the line is inside torch (`linear.py:134`) with the module
named after it; for a model with its own `forward` it is the user's line.

## How it measures, and why not torch.profiler

PyTorch's profiler collects GPU activity through CUPTI and maps it back to Python afterwards. Two
problems people hit:

* **Events can be dropped silently.** pytorch#192021 (closed 2026-08-19): kernel records whose
  timestamps fall before the capture window start are discarded, and "nothing in the trace file or
  the python API tells you the trace is incomplete". The maintainers traced it to a CUPTI bug fixed
  in CUDA 13.1; this machine runs torch 2.11 with CUDA 12.8.
* **Kernels do not map to lines.** Inductor's generated kernel names say which ops were fused, not
  which source lines, and a fused kernel's time has to be split by hand.

Truestar runs every step of a compiled program itself (docs/COMPILER.md), so the profile is built
from the runtime instead of a trace:

1. **CUDA events from the runtime's step loop** (the same loop that runs checks;
   `Plan._run_instrumented`). Every kernel, library kernel, fallback and layout copy is one step;
   control flow is timed through the steps of its regions, so a loop body step is counted once per
   iteration. Each timed call carries one event pair around **one** step (every run of that step in
   the call), and the calls cycle through the steps. Recording an event between every step was
   tried first: on this machine it stretched the transformer layer's GPU time by 12% (each event
   costs a few microseconds of GPU time on Windows), and the steps then summed to 1.12x the call.
2. **The GPU is kept busy.** A sleep kernel (`torch.cuda._sleep`, calibrated once) is queued before
   each timed call, three times longer than the host takes to issue the call. The steps then run
   back to back and each event pair measures the step's own GPU time, not a gap while the GPU
   waited for the next launch. Without it, the MLP above measured 0.310 ms of step time for a call
   whose GPU span is 0.179 ms. A step's time is that of the fastest call that timed it (the GPU is
   shared; the median is in `gpu_ms_median`).
3. **Accounting, never silent.** A first call counts how often each step runs; the timed calls
   must then time exactly that many runs (`steps_expected`), every event pair must have completed
   and give a finite time, and anything else is listed under `accounting.unaccounted` and is code
   TS510 (exit 1). `steps_timed == steps_run == steps_expected` in every test.
4. **Mapping.** Each step carries the IR nodes it covers (a fused kernel covers several) and each
   node its source (`file:line in function: code`). Steps list their lines; lines list the steps they
   are in, with `own` time (steps holding only that line) and `shared` time (fused with other lines),
   so fused time is never split by a guess.
5. **Totals that can be checked.** `gpu_busy_ms` is the same call timed with only two events around
   it (the GPU kept busy the same way); `step_gpu_ms` is the sum of the step events and must agree
   with it. `call_span_ms` is the call as it runs (with launch gaps) and `host_issue_ms` the host time
   to issue it; when the host takes longer than the GPU, `host_bound` is true.
6. **Cross check (`--crosscheck`).** The same calls run under `torch.profiler`; Truestar kernels are
   the Triton functions `kern` and `kern_loop`, and the runtime knows how many it launched, so a
   shortfall in the profiler's count is reported as TS511 with both numbers.

Bytes per step are the sizes of the tensors the step read and wrote (measured on its first run);
FLOPs are counted for `matmul`, `linear`, `attention` and `conv2d` from their shapes. Both give
achieved GB/s and TFLOP/s per step.

## Agent JSON

The usual envelope (`schema_version` 1, `tool`, `command: "profile"`, `status`, `exit_code`,
`model`, `codes`, `issues`, `next_actions`) plus:

| field | meaning |
|---|---|
| `totals` | `step_gpu_ms`, `gpu_busy_ms`, `instrumented_span_ms`, `event_overhead_ms`, `call_span_ms`, `host_issue_ms`, `host_bound`, `gpu_ms_without_line`, `steps` |
| `accounting` | `steps_run`, `steps_timed`, `unaccounted` (list of `{step, region}`) |
| `first_call` | `compile_s`, `first_call_s`, `verify_s` |
| `steps` | per step, slowest first: `step`, `kind`, `region`, `shape`, `ops`, `values`, `fused`, `impl`, `lines`, `calls`, `gpu_ms` (per call), `gpu_ms_per_launch`, `share`, `bytes`, `gb_per_s`, `flops`, `tflop_per_s` |
| `lines` | per source line: `file`, `line`, `text`, `gpu_ms` (own), `gpu_ms_shared`, `steps` |
| `crosscheck` | with `--crosscheck`: `truestar_kernels_launched`, `truestar_kernels_seen`, `dropped`, `truestar_kernel_ms_per_call`, `gpu_ms_per_call` |
| `sentinels` | the sentinel classes (compiled in checks) the program ran with |

`next_actions` holds `inspect_hot_step` with the file and line of the slowest step. Codes: TS510
(a step run without valid time), TS511 (torch.profiler saw fewer launches), TS512 (steps in the
float64 fallback, with their time).

## Measured

Measured 2026-11-24 with `scripts/bench_profile.py` (RTX 5070 Ti shared with other jobs, torch
2.11, CUDA 12.8, Windows; same models as the checks benchmark in docs/COMPILER.md; 30 timed calls
per step, three runs per model, `--crosscheck` on).

| model | steps (distinct) | step runs timed / expected | sum of steps (ms) | GPU busy call (ms) | difference | torch.profiler GPU time (ms) | Truestar kernel launches seen by torch.profiler |
|---|---|---|---|---|---|---|---|
| MLP | 4 | 120 / 120 | 0.2545, 0.2541, 0.2548 | 0.2537, 0.2548, 0.2540 | +0.3%, -0.3%, +0.4% | 0.2605, 0.2465, 0.2465 | 120 of 120 in all three runs |
| TransformerEncoderLayer | 28 | 840 / 840 | 0.8996, 0.8975, 0.9011 | 0.8885, 0.8859, 0.8872 | +1.2%, +1.3%, +1.6% | 0.8690, 0.9315, 0.8824 | 210 of 210 in all three runs |
| resnet18 | 41 | 1230 / 1230 | 3.0177, 3.0144, 3.0136 | 3.0966, 3.0894, 3.0706 | -2.5%, -2.4%, -1.9% | 3.0830, 3.0628, 3.0814 | 540, 540, then **539 of 540** |

* Every step run is accounted for on all three models, and every step maps to a source line
  (0 steps without one). Per step times sum to within 2.5% of the whole call's GPU busy time
  (target: 5%).
* **torch.profiler dropped a kernel event** in one of three resnet18 runs (539 of 540 Truestar
  launches), and in an earlier run of the transformer layer (209 of 210), with nothing in its
  output saying so. `truestar profile --crosscheck` reports it as TS511 with both counts; its own
  per step events had every run.
* torch.profiler's total differs from the busy call by 0.4% to 5.1% in either direction between runs
  of the same model, so it is a cross check, not the reference.
* Nsight Systems was not run (not installed).

## Limits

* CUDA only. The CPU executor has no profile yet.
* Each step is timed in its own calls, so a profile of n steps runs n times `--iters` calls (840
  calls for the transformer layer, about one second).
* Steps timed in different calls on a shared GPU: per step times are the fastest call, and the sum
  of steps is compared with the fastest busy call, both reported with medians.
* A library step (cuDNN convolution) can launch several CUDA kernels; it is timed as one step.
* Nsight Systems was not run for a third comparison (not installed here).
* The first call is reported as three numbers, not broken down per kernel.
