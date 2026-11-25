"""The sample order of an epoch, the shard of a rank, and the key of a sample.

Every quantity here is a pure function of integers, computed with 64 bit unsigned integer
arithmetic that numpy performs identically on every machine and OS:

* The order of epoch ``e`` sorts the indices ``0 .. n-1`` by ``h_i = mix(K ^ i)`` with
  ``K = rng.stream_key(seed, "truestar.data/order", [e])``, the element bits of the keyed RNG
  (docs/RANDOM.md), ties broken by the index (a stable sort). Without shuffling the order is the
  identity.
* Rank ``r`` of ``W`` takes the positions ``r, r + W, r + 2W, ...`` of the order (after the
  ``start`` positions already consumed), so the ranks never overlap and together cover every
  sample. ``drop_last`` keeps only as many positions as fill whole batches on every rank.
* The key a transform receives for sample ``i`` in epoch ``e`` is
  ``rng.stream_key(seed, "truestar.data/sample", [e, i])``: it depends on the sample and the
  epoch only, never on the worker, the rank or the batch.
"""

from __future__ import annotations

import numpy as np

from .. import rng
from .errors import raise_

ORDER_SITE = "truestar.data/order"
SAMPLE_SITE = "truestar.data/sample"

_G = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31 = np.uint64(30), np.uint64(27), np.uint64(31)


def mix_array(z: np.ndarray) -> np.ndarray:
    """SplitMix64 on a uint64 array (wrapping arithmetic), equal to ``rng.mix`` element by element."""
    z = np.asarray(z, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = z + _G
        z = (z ^ (z >> _S30)) * _M1
        z = (z ^ (z >> _S27)) * _M2
    return z ^ (z >> _S31)


def epoch_order(n: int, seed: int, epoch: int, shuffle: bool = True) -> np.ndarray:
    """The sample indices of epoch ``epoch`` in the order they are served (int64, length n)."""
    if n < 0:
        raise_("TS1381", f"a dataset has a length of at least 0, got {n}")
    if not shuffle:
        return np.arange(n, dtype=np.int64)
    K = np.uint64(rng.stream_key(int(seed), ORDER_SITE, [int(epoch)]))
    h = mix_array(np.arange(n, dtype=np.uint64) ^ K)
    return np.argsort(h, kind="stable").astype(np.int64, copy=False)


def usable(n: int, world: int, batch_size: int, drop_last: bool) -> int:
    """How many positions of an epoch are served in total across the ranks."""
    if not drop_last:
        return n
    per = world * batch_size
    return (n // per) * per


def shard(order: np.ndarray, rank: int, world: int, start: int = 0, stop: int = None) -> np.ndarray:
    """Rank ``rank``'s positions of ``order[start:stop]``: every ``world``-th one, starting at its rank."""
    stop = len(order) if stop is None else stop
    return order[start:stop][rank::world]


def sample_key(seed: int, epoch: int, index: int) -> int:
    """The key a transform receives for dataset index ``index`` in epoch ``epoch`` (signed int64 form)."""
    return rng.as_i64(rng.stream_key(int(seed), SAMPLE_SITE, [int(epoch), int(index)]))


def sample_keys(seed: int, epoch: int, indices: np.ndarray) -> np.ndarray:
    """``sample_key`` for many indices at once (int64 array): ``mix(mix(K_e) ^ i)`` with
    ``K_e = rng.stream_key(seed, SAMPLE_SITE, [epoch])``, since folding an int is ``mix(mix(k) ^ i)``."""
    pre = np.uint64(rng.mix(rng.stream_key(int(seed), SAMPLE_SITE, [int(epoch)])))
    return mix_array(np.asarray(indices, dtype=np.int64).astype(np.uint64) ^ pre).view(np.int64)
