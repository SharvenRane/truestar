"""Deterministic Hadamard matrices and the fast Walsh transform, in float64.

A Hadamard matrix H of order n has entries +1 and -1 and H H^T = n I, so H / sqrt(n) is
orthogonal. Orders built here:

* n = 2^k: Sylvester, H_2n = [[H, H], [H, -H]] (symmetric).
* n = m 2^k where m = q + 1 for a prime q with q = 3 (mod 4): Paley's first construction for
  H_m, then the Kronecker product H_m x H_{2^k}. This covers the hidden sizes of the local test
  models (576 = 72 x 8, 960 = 60 x 16, 2560 = 20 x 128, 896 = 224 x 4, 1536 = 12 x 128).

Other orders raise RotationError (TS1022): no Hadamard matrix of that order is built, and nothing
falls back to a random rotation.

Every matrix is checked on construction: H H^T must equal n I exactly (integer arithmetic in
float64 is exact at these sizes).
"""

from __future__ import annotations

import functools
import math
from typing import Tuple

import torch

F64 = torch.float64


class RotationError(ValueError):
    """TS1022: a rotation was asked for a size or a model layout this pass does not support."""
    code = "TS1022"


class RotationRefusedError(RotationError):
    """TS1024: a rotated model was given to a format the rotation measurably makes worse."""
    code = "TS1024"


def _is_prime(q: int) -> bool:
    if q < 2:
        return False
    for p in range(2, int(math.isqrt(q)) + 1):
        if q % p == 0:
            return False
    return True


def factor(n: int) -> Tuple[int, int]:
    """(m, p) with n = m p, p a power of two and m 1 or q + 1 for a Paley prime q (q = 3 mod 4),
    choosing the smallest m. Raises RotationError when no such split exists."""
    if n < 1:
        raise RotationError(f"[TS1022] a Hadamard order must be positive, got {n}")
    p = 1
    while n % (2 * p) == 0:
        p *= 2
    # try m = n / p, n / (p / 2), ... (m must be 1, 2 or a multiple of 4 to be a Hadamard order)
    cands = []
    pp = p
    while pp >= 1:
        m = n // pp
        if m == 1 or m == 2 or (m % 4 == 0 and _is_prime(m - 1) and (m - 1) % 4 == 3):
            cands.append((m, pp))
        pp //= 2
    if not cands:
        raise RotationError(f"[TS1022] no Hadamard matrix of order {n} is built here: {n} is not "
                            f"2^k times q + 1 for a prime q = 3 (mod 4)")
    return min(cands)


@functools.lru_cache(maxsize=None)
def _paley(m: int) -> Tuple[Tuple[int, ...], ...]:
    """Paley I: for a prime q = 3 (mod 4), H = I + S with S = [[0, 1^T], [-1, Jacobsthal]]."""
    q = m - 1
    sq = {(x * x) % q for x in range(1, q)}
    chi = [0] + [1 if a in sq else -1 for a in range(1, q)]
    S = [[0] * m for _ in range(m)]
    for j in range(1, m):
        S[0][j] = 1
        S[j][0] = -1
    for i in range(q):
        for j in range(q):
            S[i + 1][j + 1] = chi[(j - i) % q]
    return tuple(tuple(S[i][j] + (1 if i == j else 0) for j in range(m)) for i in range(m))


def _sylvester(p: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=F64)
    while h.shape[0] < p:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


@functools.lru_cache(maxsize=32)
def _hadamard_cpu(n: int) -> torch.Tensor:
    m, p = factor(n)
    hp = _sylvester(p)
    if m == 1:
        h = hp
    elif m == 2:
        h = _sylvester(2 * p)
    else:
        h = torch.kron(torch.tensor(_paley(m), dtype=F64), hp)
    if not torch.equal(h @ h.T, n * torch.eye(n, dtype=F64)):
        raise RotationError(f"[TS1022] the Hadamard matrix of order {n} failed H H^T = n I")
    return h


def hadamard(n: int, device=None) -> torch.Tensor:
    """The +1 / -1 Hadamard matrix of order n (float64), H H^T = n I."""
    h = _hadamard_cpu(n)
    return h if device is None else h.to(device)


def orthogonal_hadamard(n: int, device=None) -> torch.Tensor:
    """H / sqrt(n): an orthogonal matrix (float64)."""
    return hadamard(n, device) / math.sqrt(n)


def walsh(x: torch.Tensor, normalize: bool = True, dtype=F64) -> torch.Tensor:
    """x @ H along the last dim, in O(n log n) for the power of two factor (butterflies in ``dtype``,
    float64 by default; float32 only for training) and one small matmul for the Paley factor;
    divided by sqrt(n) when ``normalize``.

    For n a power of two the butterflies only add and subtract, so on inputs whose sums fit
    53 bits (any float16 or bfloat16 row whose exponent spread is below 2^38) the result is exact."""
    n = x.shape[-1]
    m, p = factor(n)
    if m == 2:
        m, p = 1, 2 * p
    y = x.to(dtype).reshape(-1, m, p)
    h = 1
    while h < p:            # Sylvester butterflies on the last factor
        y = y.reshape(-1, m, p // (2 * h), 2, h)
        a, b = y[:, :, :, 0], y[:, :, :, 1]
        y = torch.stack([a + b, a - b], dim=3).reshape(-1, m, p)
        h *= 2
    if m > 1:               # (H_m x H_p): y[a', b] = sum_a Hm[a, a'] y[a, b]
        hm = torch.tensor(_paley(m), dtype=dtype, device=y.device)
        y = torch.einsum("ac,rab->rcb", hm, y)
    y = y.reshape(x.shape)
    return y / math.sqrt(n) if normalize else y
