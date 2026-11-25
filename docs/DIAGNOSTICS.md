# Diagnostics: CUDA out of memory and distributed hangs

Pain ledger row P10 (docs/QUALITY_BAR.md). PyTorch reports an OOM with the requested size and the op
type, and a NCCL hang with a timeout on whichever rank noticed. Neither says which layer instance,
which rank or which line caused it. `truestar.diag` does.

## Out of memory: `truestar.diag.memory`

```python
from truestar.diag import track_memory

tracker = track_memory(model)          # hooks every module; history persists across steps
for batch in loader:
    with tracker:                      # or decorate the step function: @tracker
        loss = model(batch).mean()
        loss.backward()
        opt.step(); opt.zero_grad()

print(tracker.memory_timeline().to_text())   # per module memory for the latest step
```

On a CUDA OOM the tracker prints a report to stderr, stores it as `tracker.report`, attaches it to
the exception as `err.truestar_report` and re raises the original error unchanged. `report.to_dict()`
gives the same content as JSON friendly data for scripts and agents.

Example, a model where one of four identical MLP blocks blows up on the fourth step:

```
truestar memory: CUDA out of memory at step 4, forward of blocks.2.mlp (MLP, example_oom.py:8)
  failed allocation: 1024.00 GiB (device total 15.92 GiB, allocated 27.17 MiB, reserved 32.00 MiB, peak this step 27.17 MiB)
  failing line: example_oom.py:12 in forward: h = h.repeat(1, 1 << 22)       # an input dependent blow up
  module chain: <root Net> > blocks.2 > blocks.2.mlp
  top memory holders this step (memory kept after each module's forward):
      256.00 KiB  blocks.0.mlp.up  (64, 1024) float32
      256.00 KiB  blocks.0.mlp.act  (64, 1024) float32
      ...
  largest live CUDA tensors:
        1.00 MiB  grad of blocks.0.mlp.up.weight (1024, 256) float32
        ...
  trend: steps 1 to 3 peaked at 33.76 MiB, 33.76 MiB, 33.76 MiB (steady); step 4 asked for 1.00 TiB more: a spike in this step, not a slow leak
```

What the report contains:

* **Module instance and phase.** Forward attribution comes from forward pre and post hooks (the
  innermost module still running). Backward attribution comes from tensor gradient hooks: the
  gradient of a module's output arriving marks the start of its backward. Tensor hooks are used
  instead of `register_full_backward_hook` because the latter breaks models with in place
  activations. An OOM in the loss, the optimizer or other code outside module calls is reported as
  `outside modules` with the last event seen.
* **Failing line:** the deepest frame of the traceback outside torch.
* **Top holders:** memory kept by each leaf module's forward in the failing step, plus the largest
  live CUDA tensors labelled as parameter, gradient or the output of a named module.
* **Trend:** per step peaks of earlier steps, classified as steady (a spike in this step), varying
  (input dependent) or a slow leak (memory left after each step keeps growing).

The tracker resets the CUDA peak statistic at the start of every step; code that reads
`torch.cuda.max_memory_allocated` itself will see per step values while tracking is on.

## Distributed hangs: `truestar.diag.collectives`

```python
import torch.distributed as dist
from truestar.diag import record_collectives

dist.init_process_group("nccl")
with record_collectives(timeout=120):      # on every rank
    train()
```

Each rank appends one record per collective (sequence number, op, shapes, dtype, process group,
extra arguments such as `src` or the reduce op, and the Python call site) to `rank<N>.jsonl` in a
shared directory: `directory=...`, else `$TRUESTAR_COLLECTIVE_DIR`, else a folder under the temp dir.
For jobs on several machines the directory must be on a file system every node can see. A hung
collective cannot carry messages, so files are the channel.

A watchdog thread per rank notices a collective that made no progress for `timeout` seconds. It
writes a stall marker, every other rank's watchdog then dumps where its main thread is, and after
`settle` seconds the stalled rank reads all files and writes `diagnosis_rank<N>.txt` and `.json`.
The lowest stalled rank prints it. With `on_hang="exit"` (the default) the process then exits with
code 86, since a hung collective cannot be cancelled; `on_hang="report"` keeps waiting. A
collective that raises (for example the backend's own timeout) triggers the same diagnosis before
the error propagates. `diagnose(directory)` can also be called after the fact, from any process.

Wrapped: `all_reduce`, `all_gather`, `all_gather_into_tensor`, `broadcast`, `reduce`,
`reduce_scatter`, `reduce_scatter_tensor`, `barrier`, `gather`, `scatter`, `all_to_all`,
`all_to_all_single` and the `*_object` variants.

What the diagnosis decides, in this order:

1. **Crash.** A rank that left the recorder with an exception: the error, its line, and the
   collective the other ranks wait in.
2. **Disagreement.** Per process group, the first sequence number where ranks differ on op, shapes,
   dtype, arguments or call site. It recognises a skipped collective (the rank's call matches the
   others' next one), an extra collective, and two collectives in swapped order, and blames the
   minority. A difference in call site alone can be harmless, so it is reported only when no
   stronger disagreement exists.
3. **Behind.** Everyone agrees so far but a rank entered fewer collectives: finished, dead without a
   Python error, or still running elsewhere (with its current line from the stack dump).
4. **Stall** with full agreement: a slow rank, the network, or collectives outside the recorded API.

Example, three ranks where rank 1 skips an `all_reduce` inside `if has_aux:`:

```
truestar collectives: hang diagnosis (3 ranks, written by rank 0)
  trigger: collective seq 3 made no progress for 2 s
  verdict: rank 1 skipped collective seq 2 all_reduce (8,) float32 at diag_workers.py:42 that ranks 0, 2 called; rank 1 went on to all_reduce (8,) float32 at diag_workers.py:44, so every later collective is paired with the wrong one. Look for a data dependent branch around diag_workers.py:42.
  divergence starts at: <repo>\tests\diag_workers.py:42
      dist.all_reduce(aux)  # MARK:skip_target
  collectives entered: rank 0: 4, rank 1: 3, rank 2: 4
  behind: rank 1
  rank 0: waiting in seq 3 all_reduce (8,) float32 at diag_workers.py:44
  rank 1: finished
  rank 2: waiting in seq 3 all_reduce (8,) float32 at diag_workers.py:44
  last 6 collectives per rank:
    rank 0:
      seq    0 [world] barrier no tensors at diag_workers.py:73  done
      seq    1 [world] all_reduce (8,) float32 at diag_workers.py:38  done
      seq    2 [world] all_reduce (8,) float32 at diag_workers.py:42  done
      seq    3 [world] all_reduce (8,) float32 at diag_workers.py:44  WAITING
    ...
```

Note that rank 1 never hung: its next `all_reduce` silently paired with the others' skipped one.
That is why the hang shows up on the healthy ranks and why the per rank sequence matters.

Example, four ranks where rank 2 calls `broadcast` before `all_reduce`:

```
  verdict: ranks disagree on the op of collective seq 2: rank 2 called broadcast (4,) float32 src=0 at diag_workers.py:28, ranks 0, 1, 3 called all_reduce (4,) float32 at diag_workers.py:31. Either rank 2 skipped a collective or the ranks call them in a different order.
```

## Limits

* NCCL itself is untested: this machine has one GPU. The recorder wraps the Python API and is
  backend agnostic, but every test runs on gloo over CPU processes.
* Collectives issued from C++ are not recorded, notably DDP's gradient bucket all reduces and FSDP
  internals. A hang inside those shows as a stall with no disagreement.
* `async_op=True` collectives are recorded but not timed (completion happens at `wait()`).
* A mismatch that does not hang (collectives that silently pair up and all complete) is not
  reported live; call `diagnose(directory)` after the run to check.
* The OOM tests ask for 1 TiB so the allocator fails at once without filling the shared GPU. On
  Windows the NVIDIA driver can fall back to system memory, so a request a little above VRAM may
  succeed slowly instead of raising (a 20 GiB request succeeded on the 16 GB card here).
* Tracking runs Python hooks on every module call; it is a debugging mode, not meant to stay on in
  production training.
