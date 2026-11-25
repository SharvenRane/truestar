# Deterministic data loading and exact resume

Author: Sharven Rane. First written 2026-11-24.

Code: `src/truestar/data/` (loader, sources, collate, keyed augmentation, error codes) and
`src/truestar/train/checkpoint.py` (the checkpoint file). Tests: `tests/test_dataio.py`. Evidence:
`scripts/data_vs_torch.py`, `results/data_loader/data_vs_torch.json`.

## The problem

A training run is only reproducible when its batches are. With `torch.utils.data.DataLoader` the
same seed gives other augmentations for another `num_workers` (each worker gets its own generator),
the sampler order is tied to torch's global generator, and resuming in the middle of an epoch means
replaying the sampler by hand. Measured here: with `torch.manual_seed(0)` and one fixed sampler
generator, one epoch of the same augmentation gave 3 different sha256 digests for 0, 2 and 4 workers.

## Use

```python
from truestar.data import Loader, ArraySource, RandomCropFlip
from truestar import train as T

loader = Loader(ArraySource(x, y), 256, seed=0, rank=rank, world=world, transform=RandomCropFlip(4),
                workers=4, mode="process", pin_memory=True, device="cuda")
trainer = T.Trainer(step, lr=sched, seed=11)        # fills the step's dropout key from (seed, step)
for batch in loader:                                 # one epoch from the current position
    trainer.train_step(*batch)
    if trainer.step_count % 1000 == 0:
        trainer.save("run.tsx", loader=loader)       # params, optimizer state, seed, loader position
# later, in a new process
trainer.load("run.tsx", loader=loader)               # the next batch is the one the old run would take
```

## What decides a batch

Every batch is a pure function of the seed, the epoch, the rank, the world size, the batch number
and the data. Nothing depends on workers, threads, the OS or the machine.

* **Order.** Epoch `e` sorts the indices by `mix(K ^ i)` with `K = rng.stream_key(seed,
  "truestar.data/order", [e])`, the element bits of the keyed RNG (docs/RANDOM.md), in uint64
  arithmetic, stable sort. `shuffle=False` keeps the index order.
* **Sharding.** Rank `r` of `W` takes positions `r, r + W, r + 2W, ...` of the order. The ranks
  never overlap and together serve every sample. By default nothing is dropped, so the last batch
  of a rank may be short and ranks may differ by one batch. `drop_last=True` serves only as many
  positions as fill whole batches on every rank.
* **Augmentation keys.** A transform is called as `transform(sample, key)` with `key =
  rng.stream_key(seed, "truestar.data/sample", [e, i])` for dataset index `i`, so a sample gets the
  same augmentation whatever its batch, rank or worker. A transform with a `batch(samples, keys)`
  method gets whole batches from sources that read batches at once; it must give the same bits as
  the per sample call (tested for `RandomCropFlip`).
* **Workers.** `workers=k` threads or processes (`mode`) compute batches ahead, at most
  `k * prefetch` in flight, and the loader hands them out in batch order whatever order they finish.
* **Position.** `state_dict()` holds the settings, the dataset length and fingerprint, the epoch
  and the number of order positions consumed across all ranks. Only batches handed to the caller
  count, not prefetched ones. `load_state_dict` refuses other settings (TS1382) or other data
  (TS1383). With `elastic=True` a run may resume with another world size: the rest of the epoch is
  re sharded, still with no overlap and no drop, but the batches then differ from the uninterrupted
  run.

## Sources

| Source | Sample `i` |
|---|---|
| `ArraySource(*arrays)` | `(a[i], b[i], ...)` of numpy arrays or torch tensors; batches read with one fancy index |
| `MemmapSource("x.npy", ...)` or `MemmapSource((path, dtype, shape))` | the same, read through numpy memory maps, reopened in each worker process |
| `FolderSource(root, suffixes, reader, labels)` | `reader(path)`, files in the byte order of their UTF-8 relative paths with `/`, so the same on every OS; `labels=True` adds the index of the top level folder |
| `LinesSource(path, parse)`, `JsonLinesSource(path)` | one line (UTF-8, `\r\n` or `\n`), indexed once by byte offset, read by seeking |

Any object with `len` and `[i]` works too. `collate` stacks tuples field by field, dicts key by key
and arrays with `np.stack`; Python ints become int64 on every OS. `to_torch` wraps the arrays
without a copy, pins them when `pin_memory=True` and CUDA is present (`loader.pinned` says whether
it did) and moves them with `non_blocking`.

## Checkpoints

A checkpoint is a `.tsx` file (docs/EXPORT.md format version 1, extended, not duplicated): the
carried tensors as params of a program with no nodes, and the JSON state under
`metadata["truestar_checkpoint"]`. Checkpoint version 2 holds the params, the optimizer state, the
step count, the history, the schedule and configuration, the hash of the step program, the RNG
seed and key input name, and the loader position. Version 1 files still load.

* **Checksums.** A sha256 per tensor and over the whole file, checked on every load (TS1386).
* **Atomic writes.** Temporary file in the target folder, flush, `fsync`, rename. A crash leaves the
  old file or the new one.
* **Same state, same bytes.** Two runs that reach the same state write identical files.
* **Refusals.** Another step program, names, dtypes or shapes: TS1388. Another seed, or a loader
  passed to a checkpoint saved without its position: TS1389. Unknown checkpoint version: TS1387.

`truestar.train.checkpoint.save(path, tensors, state)` and `load(path)` serve code that does not
use `Trainer`.

## Measured

**Exact resume** (`test_exact_resume_n_plus_m_equals_uninterrupted`): an MLP with dropout imported
with `training=True`, label smoothed cross entropy, AdamW with weight decay, clipping and a warmup
cosine schedule, a shuffled loader with 2 worker threads and 6 batches per epoch. 4 steps, save,
new trainer and new loader in fresh objects, load, 6 more steps, against 10 straight steps across
the epoch boundary: every param and optimizer tensor `torch.equal`, the history equal, the final
checkpoint bytes equal. Three mutations each break a test: not restoring the loader position,
keying augmentation by batch number instead of sample, and handing out batches out of order.

**Determinism** (`results/data_loader/data_vs_torch.json`, 8192 samples of `[3, 32, 32]`, pad 4
crop and flip, batch 256): the epoch digest is identical for 0, 1, 2 and 4 threads, 2 and 4
processes, a second process, and a Linux machine (Modal, Python 3.12) against Windows 11 (Python
3.10): `dcd3b6da624d...`. The same 19 tests pass on Linux with an L4, including the pinned memory
transfer being bitwise equal to the CPU batch. torch's DataLoader gave 3 different digests for 3
worker counts.

**Throughput** (Windows 11, 24 logical CPUs shared with other jobs, 50,000 samples, batch 256,
samples per second of the second epoch; noisy within about 20 percent):

| case | truestar | torch DataLoader |
|---|---|---|
| plain, no per sample work, 0 workers | 564k | 226k |
| crop and flip, 0 workers (batched transform) | 118k | 77k |
| crop and flip, 0 workers (per sample calls) | 103k | 77k |
| crop and flip, 4 threads (batched) | 144k | no thread mode |
| crop and flip, 4 processes (batched) | 125k | 211k |
| crop and flip, 4 processes (per sample) | 127k | 211k |

The single process paths are faster than torch's because whole batches are read with one index
and keys are computed per batch. With 4 processes torch is faster: its workers return tensors in
shared memory, while truestar's process pool pickles each batch through a pipe (about 3 MB per
batch here). Threads only help when the per sample work releases the GIL (file reads, numpy on
large arrays); with small per sample numpy calls 4 threads were slower than 0 workers (52k against
103k).

## Limits

* Process workers send batches through pickling, not shared memory; that is the next step for
  multi process throughput.
* Without `drop_last`, ranks can differ by one batch per epoch. Collective training steps then need
  the caller to handle the extra batch, as with any no drop sharding.
* A source's fingerprint is cheap (array shapes, file names, file sizes), not a hash of the
  contents. Data changed in place with the same sizes is not detected.
* `elastic=True` resumes keep no overlap and no drop but are not batch for batch identical to the
  uninterrupted run, and are refused with `drop_last=True`.
