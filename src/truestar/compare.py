"""Numerical comparison with tolerances that tell real divergence apart from float noise.

Two measures per tensor, both relative errors on the same scale as the threshold:

* aggregate: ||a - b|| / ||b|| over the whole tensor. It sees a broad accuracy loss and averages
  away per element rounding noise;
* local: the worst single element and the worst slice (every index of every dim: a row, a column,
  a channel, a batch item), each against its own expected magnitude. An element is measured against
  max(|b_i|, rms(b)); a slice of n elements against the norm of its values floored elementwise at
  rms(b) * max(LOCAL_FLOOR, 1 / sqrt(n)). A slice of small values is so judged on its own scale
  once it holds enough elements for rounding noise to average out, while an element that cancelled
  to near zero in exact math is judged on the tensor's scale. One wrong row in 1024 is about 0.03
  aggregate but about 1 local.

The local limit is LOCAL_FACTOR times the aggregate threshold, and the threshold itself is never
below the precision's floor nor 10x the error a correct CPU run shows on the same module. Measured
margins: docs/TOLERANCE.md, section "truestar check", and scripts/bench_localized.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

# Relative L2 error expected from correct float math in each precision.
# Values are deliberately generous: a clean model must report clean (QUALITY_BAR never introduce #2).
DTYPE_FLOOR = {
    torch.float64: 1e-10,
    torch.float32: 1e-4,
    torch.float16: 5e-3,
    torch.bfloat16: 4e-2,
    # fp8 matmul inputs (the tensor core profiles, docs/TENSOR_CORE_MODELS.md): e4m3 keeps 3
    # fraction bits, so a per tensor scaled matmul is off by a few percent when correct
    torch.float8_e4m3fn: 6e-2,
    torch.float8_e5m2: 1e-1,
}
TF32_FLOOR = 5e-3
# Above threshold * DIVERGENT_FACTOR the result is wrong, not just imprecise. In any precision a
# relative error above DIVERGENT_CAP is wrong, so low precision floors cannot hide real bugs.
DIVERGENT_FACTOR = 20.0
DIVERGENT_CAP = 0.1
# Localized faults (one wrong row, element, column or small value region). The worst element or
# slice may be LOCAL_FACTOR times the aggregate threshold before it is suspect; it is divergent by
# the aggregate rule applied to that limit, with LOCAL_DIVERGENT_CAP in place of DIVERGENT_CAP.
LOCAL_FACTOR = 8.0
LOCAL_DIVERGENT_CAP = 0.5
# A slice of n elements is measured against its values floored at rms * max(LOCAL_FLOOR, 1/sqrt(n)).
LOCAL_FLOOR = 1.0 / 32.0

OK, SUSPECT, DIVERGENT = "ok", "suspect", "divergent"
_RANK = {OK: 0, SUSPECT: 1, DIVERGENT: 2}


def worst(*statuses: str) -> str:
    return max(statuses, key=lambda s: _RANK[s]) if statuses else OK


def local_limit(threshold: float) -> float:
    """The largest local (element or slice) error that is still OK under `threshold`."""
    return threshold * LOCAL_FACTOR


@dataclass
class Diff:
    rel_err: float  # ||a - b|| / ||b|| over finite elements
    max_abs: float
    nonfinite_mismatch: bool  # NaN or Inf where the reference has none (or vice versa)
    shape_mismatch: bool
    note: str = ""
    # Worst element or slice error against its own magnitude (module docstring), and where it is,
    # for example "index 517 of dim 0 (64 elements)" or "element [3, 17]". 0.0 when not measured.
    local_err: float = 0.0
    local_at: str = ""

    def status(self, threshold: float) -> str:
        if self.shape_mismatch or self.nonfinite_mismatch:
            return DIVERGENT
        return worst(self.aggregate_status(threshold), self.local_status(threshold))

    def aggregate_status(self, threshold: float) -> str:
        if self.rel_err > min(threshold * DIVERGENT_FACTOR, max(DIVERGENT_CAP, threshold * 2)):
            return DIVERGENT
        if self.rel_err > threshold:
            return SUSPECT
        return OK

    def local_status(self, threshold: float) -> str:
        lim = local_limit(threshold)
        if self.local_err > min(lim * DIVERGENT_FACTOR, max(LOCAL_DIVERGENT_CAP, lim * 2)):
            return DIVERGENT
        if self.local_err > lim:
            return SUSPECT
        return OK

    def localized(self, threshold: float) -> bool:
        """True when the local measure, not the whole tensor, sets the status."""
        return _RANK[self.local_status(threshold)] > _RANK[self.aggregate_status(threshold)]

    def explain(self, threshold: float) -> str:
        """Why the status is what it is, naming the slice when the local measure decided."""
        if self.shape_mismatch or self.nonfinite_mismatch:
            return self.note or "NaN or infinity where the reference has none"
        if self.localized(threshold):
            return (f"error {self.local_err:.2e} at {self.local_at} vs local limit "
                    f"{local_limit(threshold):.1e} (whole tensor {self.rel_err:.2e})")
        return f"error {self.rel_err:.2e} vs threshold {threshold:.1e}"


def _local(d: torch.Tensor, b: torch.Tensor):
    """(worst element or slice error against its own magnitude, where it is). d and b are float64
    with every non finite place already set to zero."""
    n = d.numel()
    if n < 2:
        return 0.0, ""  # one element: the aggregate already is the element's own error
    d2 = d * d
    b2 = b * b
    ms = float(b2.mean())
    if ms == 0.0:
        return (0.0, "") if float(d2.max()) == 0.0 else (float("inf"), "a reference of zeros")
    e2 = d2 / b2.clamp_min(ms)
    best = float(e2.amax())  # argmax over a large tensor is slow on the CPU; locate it below
    slice_best, slice_at = 0.0, ""
    if d.ndim >= 2:
        shape = d.shape
        for k, size in enumerate(shape):
            if size < 2:
                continue
            ns = n // size
            f2 = ms * max(LOCAL_FLOOR * LOCAL_FLOOR, 1.0 / ns)
            lead = 1
            for s_ in shape[:k]:
                lead *= s_
            r = (d2.reshape(lead, size, -1).sum((0, 2))
                 / b2.clamp_min(f2).reshape(lead, size, -1).sum((0, 2)))
            j = int(r.argmax())
            v = float(r[j])
            if v > slice_best:
                slice_best, slice_at = v, f"index {j} of dim {k} ({ns} elements)"
    # Name the slice when it explains the worst element too (a whole wrong row has elements as bad
    # as the row); name the element when it stands alone.
    if slice_at and slice_best >= 0.25 * best:
        where = slice_at
    else:
        i = int((e2.reshape(-1) == best).nonzero()[0])
        idx = []
        for size in reversed(d.shape):
            idx.append(i % size)
            i //= size
        where = "element [" + ", ".join(str(x) for x in reversed(idx)) + "]"
    return max(best, slice_best) ** 0.5, where


def diff_tensors(actual: torch.Tensor, reference: torch.Tensor, local: bool = True) -> Diff:
    """Compare one tensor to its reference. Both are moved to CPU float64 first. local=False
    skips the per element and per slice measure (local_err stays 0)."""
    if actual.shape != reference.shape:
        return Diff(float("inf"), float("inf"), False, True,
                    note=f"shape {tuple(actual.shape)} vs reference {tuple(reference.shape)}")
    if not (actual.is_floating_point() or actual.is_complex()):
        same = bool(torch.equal(actual.cpu(), reference.cpu().to(actual.dtype)))
        return Diff(0.0 if same else float("inf"), 0.0 if same else float("inf"), False, False,
                    note="" if same else "integer or boolean values differ")
    a = actual.detach().cpu().to(torch.float64)
    b = reference.detach().cpu().to(torch.float64)
    if a.numel() == 0:
        return Diff(0.0, 0.0, False, False)
    d = a - b
    # Fast path: everything finite (the common case).
    stats = torch.stack([torch.linalg.vector_norm(d), torch.linalg.vector_norm(b), d.abs().max()])
    if bool(torch.isfinite(stats).all()) and bool(torch.isfinite(a).all()):
        num, den, mx = stats.tolist()
        if num == 0.0:
            return Diff(0.0, mx, False, False)
        if den == 0.0:
            return Diff(float("inf") if num > 1e-30 else 0.0, mx, False, False)
        loc, at = _local(d, b) if local and not d.is_complex() else (0.0, "")
        return Diff(num / den, mx, False, False, local_err=loc, local_at=at)
    fa, fb = torch.isfinite(a), torch.isfinite(b)
    nonfinite_mismatch = bool((fa != fb).any())
    if not nonfinite_mismatch and not bool(fb.all()):
        # Both have NaN/Inf in the same places: they must also agree on which kind.
        nonfinite_mismatch = not bool(torch.equal(torch.isnan(a), torch.isnan(b))) or not bool(
            torch.equal(torch.where(torch.isinf(a), a, 0.0), torch.where(torch.isinf(b), b, 0.0))
        )
    mask = fa & fb
    if not bool(mask.any()):
        return Diff(0.0, 0.0, nonfinite_mismatch, False)
    dm = torch.where(mask, a - b, 0.0)
    bm = torch.where(mask, b, 0.0)
    d = dm[mask]
    num = float(torch.linalg.vector_norm(d))
    den = float(torch.linalg.vector_norm(bm))
    if num == 0.0:
        return Diff(0.0, float(d.abs().max()), nonfinite_mismatch, False)
    if den == 0.0:
        return Diff(float("inf") if num > 1e-30 else 0.0, float(d.abs().max()),
                    nonfinite_mismatch, False)
    loc, at = _local(dm, bm) if local and not dm.is_complex() else (0.0, "")
    return Diff(num / den, float(d.abs().max()), nonfinite_mismatch, False,
                local_err=loc, local_at=at)


def diff_lists(actual: List[torch.Tensor], reference: List[torch.Tensor], local: bool = True) -> Diff:
    """Worst Diff over paired tensors: the worst aggregate, with the worst local error of any."""
    if len(actual) != len(reference):
        return Diff(float("inf"), float("inf"), False, True,
                    note=f"{len(actual)} tensors vs reference {len(reference)}")
    if not actual:
        return Diff(0.0, 0.0, False, False)
    diffs = [diff_tensors(x, y, local) for x, y in zip(actual, reference)]
    worst_d = max(diffs, key=lambda d: (d.shape_mismatch, d.nonfinite_mismatch, d.rel_err))
    i = max(range(len(diffs)), key=lambda j: diffs[j].local_err)
    loc = diffs[i]
    at = f"output {i}, {loc.local_at}" if len(diffs) > 1 and loc.local_at else loc.local_at
    return Diff(worst_d.rel_err, worst_d.max_abs, worst_d.nonfinite_mismatch, worst_d.shape_mismatch,
                worst_d.note, loc.local_err, at)


def threshold_for(dtype: torch.dtype, tf32: bool, noise_floor: Optional[float]) -> float:
    """Tolerance for one layer on one backend.

    Takes the precision's own floor, raised for TF32, and never below 10x the error a correct CPU
    float32 run shows on the same layer (the calibrated noise floor).
    """
    base = DTYPE_FLOOR.get(dtype, 1e-4)
    if tf32 and dtype == torch.float32:
        base = max(base, TF32_FLOOR)
    if noise_floor is not None and noise_floor == noise_floor:  # not NaN
        base = max(base, 10.0 * noise_floor)
    return base
