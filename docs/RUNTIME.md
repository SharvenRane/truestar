# Runtime: host overhead and CUDA graphs

Author: Sharven Rane. First written 2026-11-24. Code: `src/truestar/runtime/`, benchmark
`scripts/bench_overhead.py`, tests `tests/test_runtime.py`.

Question this answers: where does Python cost time when Truestar launches kernels, how much of it
CUDA graphs remove, and whether a Rust or C++ launcher would still pay off after that.

Short answer: Python dominates only for small kernels (GPU time under about 20 us). CUDA graph
capture removes it: a 50 kernel step drops from 1319 us to 418 us, which equals its GPU time.
After graphs the host spends 13 to 26 us per step, overlapped with 413 us of GPU work, so a native
launcher has nothing left to win on these workloads. No Rust or C++ now.

Compiled programs have their own graph runner since 2026-11-24, `cp.graphed()` (docs/COMPILER.md):
it records after the verified first call at each signature, keeps the index checks inside the graph
and updates a KV cache in place. `capture` below remains the generic layer for any callable.

## 1. API

```python
from truestar.runtime import capture

step = capture(fn, (x,))                         # any executor: CompiledKernel, CompiledGraph,
y = step(x2)                                     # the compiler's executor, plain PyTorch code
step = capture(fn, {"x": x, "w": w}, static=["w"])            # w bound by address, never copied
step = capture(fn, (x,), dynamic={0: 0}, buckets=[8, 16, 32]) # one graph per batch size
step = capture(fn, (x,), dynamic={0: 0}, buckets=[16, 64], pad=True)  # round up, slice outputs
```

The interface is one callable: `fn(*tensors)` or `fn(**tensors)` returning a tensor or a tuple,
list or dict of tensors. Nothing in the runtime knows about kernels, so the compiled executor
plugs in unchanged when it lands.

What a capture does:

1. Allocates static input buffers and runs `fn` on them twice on a side stream. The first run may
   compile and run Truestar's float64 verification. It is also checked for writes to its inputs.
   The second runs with `torch.cuda.set_sync_debug_mode("error")`, so any host sync is refused.
2. Records a third run into a `torch.cuda.CUDAGraph`, replays it once and compares every output
   bitwise with the eager run on the same inputs. Any difference refuses the capture.
3. Each call checks shapes, dtypes and devices, copies the inputs into the static buffers (one
   `torch._foreach_copy_` launch for all of them), replays, and returns outputs. With
   `outputs="copy"` (default) the outputs are fresh tensors, so a tensor the caller holds never
   changes. With `outputs="static"` the graph's own buffers come back; the next call of the
   captured function, in any bucket, overwrites them.

Inputs listed in `static=` are bound by address: the graph reads the caller's tensor directly, a
call must pass that same tensor, and in place updates are seen by the next replay. Passing the
buffers from `static_inputs()` also skips the copy.

### Refusals (all raise, none fall back silently)

| Cause | Error |
|---|---|
| `.item()`, `.cpu()`, boolean mask indexing, `nonzero`, `TRUESTAR_VERIFY=all` | `HostSyncError` |
| function writes to an input | `InputMutationError` |
| replay differs from the eager run (frozen Python state, randomness on the host) | `CaptureMismatchError` |
| CPU tensor or Python number as an input, non tensor output | `CaptureError` |
| shape outside the buckets, other dim changed, dtype or device changed, a bound input swapped | `BucketError` |
| padded result differs from the unpadded eager run | `CaptureMismatchError` |

### Buckets and their memory cost

Each bucket is its own graph with its own static inputs and outputs. Intermediates live in one
memory pool shared by all buckets of a captured function, which is safe because the buckets replay
one at a time on one stream and every buffer that must survive (static inputs and outputs) is held.
`memory()` reports per bucket the bytes held and the growth of the allocator reservation.
Measured: the 50 kernel MLP at 32x256 holds 0.09 MiB and reserves 2 MiB; at 1024x1024 it holds
12 MiB and reserves 36 MiB. Ten buckets of the larger one would hold about ten times the held
bytes; the reserved pool is shared, so it grows only to the largest bucket's need plus what the
held outputs pin.

With `pad=True` a call of size n runs in the smallest bucket of at least n: inputs are zero padded,
outputs sliced. That is only correct when rows along the padded dim are computed independently and
when the kernels' arithmetic does not depend on the padded size. The runtime does not trust this:
the first call at each new size is compared bitwise with an eager run at the real size, and refused
on any difference. Found while testing: Truestar's tile kernels choose their row block from the row
count below 16 rows, and the layer norm's reduction order then changes. Padding n = 1 or 2 into a
bucket of 16 is refused for that reason; n = 3 to 15 matched bitwise. A softmax over the padded dim
is refused for every size.

## 2. Measurements

RTX 5070 Ti, Windows 11 (WDDM driver), torch 2.11 (CUDA 12.8), Triton 3.7.1, float32 with TF32
off. Numbers are microseconds, the median of three separate benchmark processes, each value itself
the median of 7 to 9 repeats. Two other agents were using this GPU at the same time, so single runs
vary by up to 2x on some rows (seen on medium sizes); the medians below were stable across the
three runs to about 10%. Reproduce with
`PYTHONPATH=src python scripts/bench_overhead.py --runs 3` (about 3.5 minutes, one GPU process at a
time).

How each number is taken (`truestar.runtime.timing`): host time is the wall time to enqueue 200
calls with no synchronize, divided by 200; wall time adds one synchronize at the end; GPU time
records 20 calls into one CUDA graph and times its replay with CUDA events, so no host work is in
the window.

### 2.1 One kernel layer launch

Parts of `CompiledKernel.__call__`, each timed alone:

| case | gather | signature | cache lookup | verify mode | output alloc | Triton launch | of which C launcher | whole call, host | GPU | host / GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| gelu+res 64x512 | 0.5 | 0.7 | 0.1 | 0.3 | 2.1 | 12.4 | 7.2 | 17.2 | 1.8 | 9.8 |
| softmax 64x512 | 0.4 | 0.4 | 0.1 | 0.4 | 2.1 | 12.4 | 7.3 | 17.4 | 2.3 | 7.5 |
| layer norm 64x768 | 0.8 | 1.6 | 0.3 | 0.5 | 3.1 | 13.9 | 6.3 | 17.9 | 2.2 | 8.1 |
| causal attention 1x4x128x64 fp16 | 0.5 | 1.1 | 0.2 | 0.4 | 2.2 | 12.6 | 7.7 | 19.5 | 5.7 | 3.4 |
| gelu+res 4096x4096 | 0.8 | 0.7 | 0.2 | 0.4 | 2.0 | 9.7 | 4.9 | 15.6 | 258.7 | 0.1 |
| softmax 4096x4096 | 0.7 | 0.7 | 0.2 | 0.4 | 2.1 | 9.7 | 4.7 | 14.2 | 175.7 | 0.1 |
| layer norm 8192x1024 | 0.8 | 1.6 | 0.3 | 0.5 | 3.2 | 12.1 | 5.9 | 16.0 | 72.5 | 0.2 |
| causal attention 2x8x1024x64 fp16 | 0.8 | 1.8 | 0.4 | 0.5 | 3.0 | 14.5 | 6.5 | 32.8 | 54.5 | 0.6 |

"C launcher" is Triton's compiled launcher called directly with every argument precomputed: the
cost a native launcher would still pay (argument packing plus `cuLaunchKernel`). On this Windows
driver it is 5 to 8 us, more than the 2 to 4 us usually quoted for Linux.

Reading it:

* A launch costs 14 to 20 us of host time (one outlier at 33). About 7 us is the driver floor,
  5 to 7 us is Triton's Python binder and specialization key, and 4 to 6 us is Truestar's wrapper
  (argument gathering, signature tuple, output allocation, verify check). The kernel cache lookup
  itself is 0.1 to 0.4 us: the cache is not the problem.
* Small kernels run 2 to 6 us on the GPU, so host time is 3x to 10x the GPU time. Overhead
  dominates.
* Medium kernels run 55 to 260 us; host time is 0.1x to 0.6x and fully hidden by asynchronous
  execution. Overhead does not matter there.

Captured single kernels (wall per call):

| case | Truestar eager | captured, copy in and out | captured, static buffers | graph replay alone, host | PyTorch eager |
|---|---|---|---|---|---|
| gelu+res 64x512 | 17.2 | 23.4 | 8.4 | 6.6 | 81.4 (9 kernels) |
| softmax 64x512 | 17.4 | 26.7 | 7.7 | 6.3 | 8.8 |
| layer norm 64x768 | 18.0 | 27.4 | 10.1 | 14.2 | 11.5 |
| causal attention 1x4x128x64 | 19.6 | 25.4 | 10.1 | 6.9 | 11.1 |
| gelu+res 4096x4096 | 262.8 | 783.3 | 258.0 | 4.0 | 2165.5 |
| softmax 4096x4096 | 175.1 | 525.7 | 175.2 | 4.4 | 175.1 |

A graph of one kernel buys about 2x only with static buffers. With copies it loses: every copy
is one more launch at small sizes and one more pass over memory at large sizes (three passes for
gelu+res at 4096x4096). Graphs are for sequences, and large inputs belong in `static=` or in the
static buffers.

### 2.2 A 50 kernel step

Ten pre norm MLP blocks, each five kernel layer calls (layer norm, matmul, bias plus tanh GELU,
matmul, residual add), chained by hand in Python (`runtime/workloads.py`). PyTorch eager is the
same math (about 130 launches per step, cuBLAS matmuls); `torch.compile` fuses it further.

| 50 kernel MLP, us per step | 32x256 host | 32x256 wall | 1024x1024 host | 1024x1024 wall |
|---|---|---|---|---|
| Truestar eager | 1315 | 1319 | 916 | 7878 |
| Truestar captured, copy in and out | 26 | 418 | 53 | 7886 |
| Truestar captured, static buffers | 13 | 414 | 20 | 7947 |
| PyTorch eager | 1299 | 1301 | 3377 | 7929 |
| PyTorch eager captured by this layer | 60 | 246 | 132 | 7717 |
| `torch.compile(mode="reduce-overhead")` | 85 | 148 | 242 | 6994 |
| GPU time, Truestar kernels / PyTorch kernels | | 413 / 239 | | 7805 / 7832 |

Wins and losses:

* Small step: capture makes Truestar 3.2x faster (1319 to 418 us) and the step becomes GPU bound
  (418 against 413 us of GPU time). The same capture makes PyTorch eager 5.3x faster (1301 to
  246 us), so the layer is not Truestar specific.
* `torch.compile` `reduce-overhead` wins the small step at 148 us. That gap is not host overhead:
  Truestar's own kernels take 413 us of GPU time against 239 us for PyTorch's (cuBLAS matmuls in
  strict float32 at 32 rows beat Truestar's Triton matmul), and Inductor fuses the elementwise ops
  further. It is a kernel quality item for the autotuner, not a runtime item. Compile time of the
  `reduce-overhead` path was 2.5 s on the first run (warm Inductor cache on later runs).
* Medium step: every variant is GPU bound at 7.0 to 7.9 ms; capture changes nothing beyond noise
  and `torch.compile` is about 11% faster through fusion.
* Single small kernels: `torch.compile` `reduce-overhead` costs 42 to 60 us per call, slower than
  both eager PyTorch (10 to 12 us) and Truestar captured (24 to 31 us). Its guards and cudagraph
  tree bookkeeping cost more than one kernel's launch.

## 3. Would a Rust or C++ launcher still pay off?

No, not after graphs, on the numbers above.

* What is left on the host after capture: 13 us per 50 kernel step with static buffers, 26 us with
  input and output copies. Of that, 6 to 7 us is `CUDAGraph.replay()` itself, which a native
  launcher would also pay (it is `cudaGraphLaunch` on this driver), and the copies are each one
  launch (about 7 us here) in any language. A native wrapper could remove at most the Python
  part, roughly 6 to 12 us per step.
* That host time is 3% to 6% of the step's 413 us of GPU work and runs concurrently with it. Wall
  time is already the GPU time, so removing all of it would change wall time by zero.
* Without graphs, a native launcher would cut a launch from about 17.5 us to the 7 us C floor, a
  50 kernel step from 1319 us to about 350 us of host time. Capture already gets the step to
  13 to 26 us of host time from Python, so the native launcher is the weaker tool for the same job.

When to revisit (the condition, with the number to watch):

* Steps that cannot be captured: shapes outside any bucket, or per step control flow on host
  values. They run eagerly at 14 to 20 us per launch, 10 us of it Python. If a real model spends
  more than about 30% of its step there, act. The first fix is still in Python: have the kernel
  layer's `_Launch` call Triton's compiled launcher directly with prepacked arguments, which skips
  the 5 to 7 us binder per launch (that change belongs in `src/truestar/kernel/`, owned by the
  autotuning work, so it is recommended here and not done).
* Steps whose GPU time is below about 30 us even after capture (decode of a tiny model). Then the
  6 to 7 us replay floor and the copies are a real fraction. Removing input copies needs graph
  node pointer updates (`cudaGraphExecKernelNodeSetParams`), which PyTorch does not expose; that
  is the one thing a small C++ extension would buy. Until then, `static=` and the static buffers
  avoid the copies from Python.

## 4. Limits (2026-11-24)

* Inference only: inputs that require grad are refused, no backward capture.
* One stream. Buckets share a pool and must not replay concurrently from several streams.
* Dynamic dims share one size per call (batch or tokens); two independent dynamic sizes need
  separate captured functions.
* Host side randomness or Python state that changes per call is caught only when it changes the
  output of the capture's self check; a function that reads a Python value changed later without
  affecting that one comparison would replay the frozen value. Do not capture such functions.
* `set_sync_debug_mode` is a PyTorch prototype and does not see every sync; a sync it misses fails
  during recording instead, as a `CaptureError`.
* Only measured on Windows. The C launcher floor on Linux is expected to be lower, which makes
  graphs relatively less important there for medium kernels and changes nothing for the verdict.
