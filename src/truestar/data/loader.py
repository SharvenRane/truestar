"""The deterministic loader: the same batches on every machine, OS, worker count and worker mode.

A batch is a pure function of (seed, epoch, rank, world, batch number) and the data: the order comes
from the keyed RNG (``order.epoch_order``), the rank's positions from a stride over it
(``order.shard``), and a transform's randomness from the per sample key (``order.sample_key``).
Workers only compute those batches; the loader hands them out in batch order whatever order they
finish in. The position (epoch, samples consumed) is saved with ``state_dict`` and a resumed loader
serves exactly the batches the uninterrupted one would have.
"""

from __future__ import annotations

import collections
import concurrent.futures as cf
from typing import Any, Callable, Dict, Iterator, Optional

import numpy as np

from .collate import collate as default_collate, to_torch
from .errors import DataError, raise_
from .order import epoch_order, sample_keys, shard, usable
from .sources import ArraySource, fingerprint_of

STATE_VERSION = 1
MODES = ("thread", "process")


class _Job:
    """What a worker needs to build a batch from its sample indices. Picklable when the source, the
    transform and the collate function are."""

    def __init__(self, source, transform, collate, seed):
        self.source, self.transform, self.collate, self.seed = source, transform, collate, int(seed)
        has_batch = callable(getattr(source, "get_batch", None))
        # whole batch reads: no transform, or one with a ``batch(samples, keys)`` form; no custom collate
        self.batched = collate is None and has_batch and (transform is None or callable(getattr(transform, "batch", None)))

    def __call__(self, epoch: int, number: int, indices: np.ndarray):
        try:
            if self.batched:
                batch = self.source.get_batch(indices)
                if self.transform is None:
                    return batch
                return self.transform.batch(batch, sample_keys(self.seed, epoch, indices))
            if self.transform is None:
                samples = [self.source[int(i)] for i in indices]
            else:
                keys = sample_keys(self.seed, epoch, indices).tolist()
                samples = [self.transform(self.source[int(i)], k) for i, k in zip(indices, keys)]
            fn = self.collate or default_collate
            return fn(samples) if self.collate else fn(samples, f"epoch {epoch} batch {number}")
        except DataError:
            raise
        except BaseException as e:  # noqa: BLE001 - any failure is reported with where it happened
            shown = ", ".join(str(int(i)) for i in indices[:8]) + (", ..." if len(indices) > 8 else "")
            raise DataError("TS1384", f"loading epoch {epoch} batch {number} (samples {shown}) raised "
                                      f"{type(e).__name__}: {e}") from None


_WORKER_JOB: Optional[_Job] = None


def _init_worker(job: _Job) -> None:
    global _WORKER_JOB
    _WORKER_JOB = job


def _run_in_worker(epoch: int, number: int, indices: np.ndarray):
    return _WORKER_JOB(epoch, number, indices)


class Loader:
    """Batches of ``source`` for one data parallel rank.

    ``source``: a ``truestar.data`` source, any object with ``len`` and ``[i]``, or a numpy array /
    torch tensor (wrapped in ``ArraySource``). ``transform(sample, key)`` gets each sample with its
    int64 key (use it for random augmentation: ``rng.generate(..., key=key)``); a transform with a
    ``batch(samples, keys)`` method gets whole batches from sources with ``get_batch`` and must give
    the same bits as the per sample call. ``collate(samples)`` replaces the default stacking. ``workers`` threads or processes (``mode``) build batches ahead,
    at most ``prefetch`` per worker; the output order never depends on them. ``pin_memory`` puts the
    tensors in page locked memory when CUDA is available (``self.pinned`` says whether it did) and
    ``device`` moves them there without blocking.

    One ``for batch in loader`` serves the rest of the current epoch, then moves to the next epoch.
    """

    def __init__(self, source: Any, batch_size: int, *, seed: int = 0, shuffle: bool = True,
                 drop_last: bool = False, rank: int = 0, world: int = 1,
                 transform: Optional[Callable[[Any, int], Any]] = None,
                 collate: Optional[Callable[[list], Any]] = None, workers: int = 0, mode: str = "thread",
                 prefetch: int = 2, pin_memory: bool = False, device: Any = None, as_torch: bool = True):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise_("TS1380", f"batch_size must be an int of at least 1, got {batch_size!r}")
        if not isinstance(world, int) or world < 1:
            raise_("TS1380", f"world must be an int of at least 1, got {world!r}")
        if not isinstance(rank, int) or not 0 <= rank < world:
            raise_("TS1380", f"rank must be in [0, {world}), got {rank!r}")
        if mode not in MODES:
            raise_("TS1380", f"mode must be one of {MODES}, got {mode!r}")
        if not isinstance(workers, int) or workers < 0 or prefetch < 1:
            raise_("TS1380", f"workers must be an int of at least 0 and prefetch at least 1, got {workers!r}, "
                             f"{prefetch!r}")
        if not (hasattr(source, "__len__") and hasattr(source, "__getitem__")) or _is_array(source):
            source = ArraySource(source)
        self.source, self.batch_size, self.seed = source, batch_size, int(seed)
        self.shuffle, self.drop_last, self.rank, self.world = bool(shuffle), bool(drop_last), rank, world
        self.workers, self.mode, self.prefetch = workers, mode, int(prefetch)
        self.device, self.as_torch = device, as_torch
        self.pinned = False
        if pin_memory:
            import torch

            self.pinned = bool(torch.cuda.is_available())
        self._job = _Job(source, transform, collate, self.seed)
        self._pool: Optional[cf.Executor] = None
        self.epoch = 0
        self.position = 0           # positions of this epoch's order consumed across all ranks
        self._order_cache = (None, None)

    # ------------------------------------------------------------------ sizes and order
    def __len__(self) -> int:
        """Batches per epoch on this rank."""
        return self._rank_batches(0)

    @property
    def epoch_size(self) -> int:
        """Samples served per epoch across all ranks."""
        return usable(len(self.source), self.world, self.batch_size, self.drop_last)

    def _rank_batches(self, start: int) -> int:
        k = len(range(self.rank, max(self.epoch_size - start, 0), self.world))
        return -(-k // self.batch_size)

    def order(self, epoch: Optional[int] = None) -> np.ndarray:
        """The global sample order of ``epoch`` (default: the current one)."""
        e = self.epoch if epoch is None else int(epoch)
        if self._order_cache[0] != e:
            self._order_cache = (e, epoch_order(len(self.source), self.seed, e, self.shuffle))
        return self._order_cache[1]

    def set_epoch(self, epoch: int) -> None:
        """Start epoch ``epoch`` from its beginning."""
        self.epoch, self.position = int(epoch), 0

    # ------------------------------------------------------------------ iteration
    def _executor(self) -> Optional[cf.Executor]:
        if self.workers == 0:
            return None
        if self._pool is None:
            if self.mode == "thread":
                self._pool = cf.ThreadPoolExecutor(self.workers, thread_name_prefix="truestar-data")
            else:
                self._pool = cf.ProcessPoolExecutor(self.workers, initializer=_init_worker, initargs=(self._job,))
        return self._pool

    def _finish(self, batch):
        return to_torch(batch, self.pinned, self.device) if self.as_torch else batch

    def __iter__(self) -> Iterator[Any]:
        epoch, start = self.epoch, self.position
        mine = shard(self.order(epoch), self.rank, self.world, start, self.epoch_size)
        B, step = self.batch_size, self.batch_size * self.world
        jobs = [(epoch, k, mine[k * B:(k + 1) * B]) for k in range(-(-len(mine) // B))]
        pool = self._executor()
        if pool is None:
            results = (self._job(*j) for j in jobs)
        else:
            results = self._ordered(pool, jobs)
        for k, batch in enumerate(results):
            self.position = min(start + (k + 1) * step, self.epoch_size)
            yield self._finish(batch)
        if self.epoch == epoch:
            self.epoch, self.position = epoch + 1, 0

    def _ordered(self, pool: cf.Executor, jobs):
        """Results of ``jobs`` in job order with a bounded number in flight."""
        fn = self._job if self.mode == "thread" else _run_in_worker
        window = self.workers * self.prefetch
        pending: collections.deque = collections.deque()
        it = iter(jobs)
        try:
            for j in it:
                pending.append(pool.submit(fn, *j))
                if len(pending) >= window:
                    break
            while pending:
                fut = pending.popleft()
                nxt = next(it, None)
                if nxt is not None:
                    pending.append(pool.submit(fn, *nxt))
                yield fut.result()
        finally:
            for f in pending:
                f.cancel()

    def batches(self, epoch: int) -> Iterator[Any]:
        """Every batch of epoch ``epoch`` on this rank, without touching the saved position."""
        saved = (self.epoch, self.position)
        self.epoch, self.position = int(epoch), 0
        try:
            yield from self.__iter__()
        finally:
            self.epoch, self.position = saved

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ position
    def state_dict(self) -> Dict[str, Any]:
        """The position as plain JSON: settings, dataset identity, epoch and samples consumed."""
        return {"version": STATE_VERSION, "seed": self.seed, "batch_size": self.batch_size,
                "shuffle": self.shuffle, "drop_last": self.drop_last, "rank": self.rank, "world": self.world,
                "length": len(self.source), "fingerprint": fingerprint_of(self.source),
                "epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state: Dict[str, Any], elastic: bool = False) -> None:
        """Continue from ``state``. The settings must match, so the resume serves exactly the batches the
        uninterrupted loader would have. ``elastic=True`` allows another ``world``/``rank`` (and so
        another per rank batching): the remaining samples of the epoch are then re sharded, still with
        no overlap and no drop, but the batches differ from the uninterrupted run."""
        if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
            raise_("TS1382", f"not a loader state of version {STATE_VERSION}: {state!r:.120}")
        keys = ["seed", "batch_size", "shuffle", "drop_last"] + ([] if elastic else ["rank", "world"])
        mine = self.state_dict()
        bad = [f"{k} {state.get(k)!r} (saved) vs {mine[k]!r} (here)" for k in keys if state.get(k) != mine[k]]
        if bad:
            raise_("TS1382", "the saved loader position has other settings: " + "; ".join(bad))
        if state["length"] != mine["length"] or state["fingerprint"] != mine["fingerprint"]:
            raise_("TS1383", f"the dataset changed: saved length {state['length']} fingerprint "
                             f"{state['fingerprint'] or '-'}, here {mine['length']} {mine['fingerprint'] or '-'}")
        if elastic and state["world"] != self.world and self.drop_last:
            raise_("TS1382", "an elastic resume with drop_last=True and another world size would drop other "
                             "samples than the original run; resume with drop_last=False")
        self.epoch, self.position = int(state["epoch"]), int(state["position"])


def _is_array(x: Any) -> bool:
    if isinstance(x, np.ndarray):
        return True
    try:
        import torch

        return isinstance(x, torch.Tensor)
    except ImportError:  # pragma: no cover
        return False


__all__ = ["Loader", "STATE_VERSION"]
