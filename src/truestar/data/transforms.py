"""Keyed augmentations: ``transform(sample, key)`` with every random choice drawn from ``key``.

They are plain numpy on the integer bits of the keyed RNG, so the result is the same on every
machine and in every worker. Use them as ``Loader(..., transform=RandomCropFlip(4))``.
"""

from __future__ import annotations

import numpy as np

from .. import rng
from .order import mix_array


def key_ints_batch(keys: np.ndarray, count: int, high: int, site: str = "truestar.data/aug") -> np.ndarray:
    """``key_ints`` for many keys at once: an int64 array ``[len(keys), count]``."""
    if not 1 <= high <= 1 << 31:
        raise ValueError(f"high must be in [1, 2**31], got {high}")
    k = np.asarray(keys, dtype=np.int64).reshape(-1).view(np.uint64)
    K = mix_array(mix_array(k) ^ np.uint64(rng.site_id(site)))          # rng.stream_key(key, site)
    h = mix_array(K[:, None] ^ np.arange(count, dtype=np.uint64)[None, :])
    with np.errstate(over="ignore"):
        return (((h >> np.uint64(32)) * np.uint64(high)) >> np.uint64(32)).astype(np.int64)


def key_ints(key: int, count: int, high: int, site: str = "truestar.data/aug") -> np.ndarray:
    """``count`` integers in ``[0, high)`` from ``key``: exactly ``rng.generate("randint", (count,),
    torch.int64, key, site, low=0, high=high)``, computed in numpy (element i: ``h = mix(K ^ i)``, then
    ``((h >> 32) * high) >> 32``)."""
    return key_ints_batch(np.asarray([rng.as_i64(int(key))], dtype=np.int64), count, high, site)[0]


_AUG, _FLIP = rng.site_id("truestar.data/aug"), rng.site_id("truestar.data/flip")


def _scalar_ints(key: int, site: int, count: int, high: int):
    """``key_ints`` in plain Python integers (faster for a handful of values)."""
    K = rng.mix(rng.mix(rng.as_u64(int(key))) ^ site)
    return [((rng.mix(K ^ i) >> 32) * high) >> 32 for i in range(count)]


class RandomCropFlip:
    """Pad an image ``[C, H, W]`` (or a ``(image, label, ...)`` sample) by ``pad`` zeros, crop it back to
    ``H x W`` at a keyed offset and mirror it horizontally with probability one half. ``batch`` does a
    whole batch at once with the same bits."""

    def __init__(self, pad: int = 4):
        self.pad = int(pad)

    def __call__(self, sample, key: int):
        img, rest = (sample[0], sample[1:]) if isinstance(sample, tuple) else (sample, None)
        c, h, w = img.shape
        p = self.pad
        dy, dx = _scalar_ints(key, _AUG, 2, 2 * p + 1)
        flip = _scalar_ints(key, _FLIP, 1, 2)[0]
        padded = np.zeros((c, h + 2 * p, w + 2 * p), dtype=img.dtype)
        padded[:, p:p + h, p:p + w] = img
        out = padded[:, dy:dy + h, dx:dx + w]
        out = np.ascontiguousarray(out[:, :, ::-1] if flip else out)
        return (out,) + tuple(rest) if rest is not None else out

    def batch(self, sample, keys: np.ndarray):
        imgs, rest = (sample[0], sample[1:]) if isinstance(sample, tuple) else (sample, None)
        n, c, h, w = imgs.shape
        p = self.pad
        off = key_ints_batch(keys, 2, 2 * p + 1)
        flip = key_ints_batch(keys, 1, 2, "truestar.data/flip")[:, 0].astype(bool)
        padded = np.zeros((n, c, h + 2 * p, w + 2 * p), dtype=imgs.dtype)
        padded[:, :, p:p + h, p:p + w] = imgs
        out = np.empty_like(imgs)
        for j, ((dy, dx), f) in enumerate(zip(off.tolist(), flip.tolist())):
            crop = padded[j, :, dy:dy + h, dx:dx + w]
            out[j] = crop[:, :, ::-1] if f else crop
        return (out,) + tuple(rest) if rest is not None else out


__all__ = ["RandomCropFlip", "key_ints", "key_ints_batch"]
