"""Built in verification: compare a compiled kernel's outputs with the float64 reference.

The error measure is per element: |out_i - ref_i| divided by max(|ref_i|, floor), where the floor is
a fraction of the output's rms (never above its largest magnitude, never below 1e-6), and the check
fails when the worst element is over the tolerance. The fraction is FLOOR_ELEMENTWISE for a kernel
with no reduction (each element is judged on its own magnitude down to 1/8 of the rms) and
FLOOR_REDUCE for one with a reduction (a sum that cancels to near zero carries the rounding of its
terms, on the scale of the output's rms). Checks that do not know the kernel (the compiled
executor's region and end to end checks) use FLOOR_DEFAULT. Before 2026-11-24 every element was
divided by the largest magnitude, so a small element could be completely wrong and pass. NaN and
infinity must appear at exactly the same places.

A kernel with a sum reduction whose element fails that measure is measured once more with each
element's denominator raised to the sum of its terms' magnitudes (``magnitude_kernel``): the
rounding error of a float32 sum is bounded by its length times the unit roundoff times the sum of
the absolute terms, so a sum whose terms cancel (a bias gradient summed over the batch, zero in
exact math behind batch norm) is judged by the size of what it added, element by element, while
an element whose terms are small is still judged on its own magnitude (2026-11-24, found by
training steps of docs/OPTIM.md).

The tolerance depends on the kernel's compute
precision (the lowest precision among its inputs and outputs, and TF32 when allowed) and grows
slowly with the length of the longest single reduction (log2 of its size, divided by 4).

The floors were measured on the 5455 passing checks of the kernel, compiler and executor tests
(docs/KERNEL_IR.md, Verification): the largest fraction a correct kernel needed was 0.011 without a
reduction, 0.52 with one and 1.29 in region checks.

Besides the size of the errors, their sign is tested (``truestar.bias``, docs/BIAS.md): after every
output passes the size check, an output whose errors lean one way beyond what unbiased round to
nearest allows (cluster robust t test over the groupings from ``error_groupings``, p below 1e-6,
and a mean lean above 1/64 of the tolerance) raises ``KernelBiasError`` (TS960).
``TRUESTAR_VERIFY_BIAS``: ``on`` (default) raises, ``warn`` warns, ``off`` skips the test;
``TRUESTAR_BIAS_LOG`` names a file that gets one JSON line per tested output (the false positive
survey of docs/BIAS.md reads it).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import ir

BASE_TOL = {"float64": 1e-9, "float32": 2e-5, "float16": 2e-3, "bfloat16": 1.6e-2}
TF32_TOL = 2e-3
SCALE_FLOOR = 1e-6
# Element floor as a fraction of rms(ref), see the module docstring.
FLOOR_ELEMENTWISE = 0.125
FLOOR_REDUCE = 2.0
FLOOR_DEFAULT = 4.0


class KernelMismatchError(AssertionError):
    """A compiled kernel disagrees with the float64 reference. Never silently ignored."""


class KernelBiasError(KernelMismatchError):
    """A compiled kernel's errors are within the size tolerance but lean one way (TS960)."""

    code = "TS960"


class KernelBiasWarning(UserWarning):
    """TRUESTAR_VERIFY_BIAS=warn: the TS960 finding as a warning instead of an error."""


@dataclass
class OutputCheck:
    name: str
    error: float
    tol: float
    detail: str = ""
    at: str = ""  # the worst element's index, for example "[3, 17]"

    @property
    def ok(self) -> bool:
        return not self.detail and self.error <= self.tol


def tolerance(k: ir.Kernel, env: Optional[Dict[str, int]] = None,
              allow_tf32: bool = False) -> float:
    """Tolerance for k given bound sizes env (symbols without a value count as 1024)."""
    dtypes = [b.dtype for b in k.inputs] + [o.dtype for o in k.outputs]
    prec = ir.lowest_precision(dtypes)
    base = BASE_TOL[prec]
    if allow_tf32 and prec == "float32":
        base = TF32_TOL
    env = env or {}
    length = 1
    for r in k.reduces():
        s = r.axis.size
        length = max(length, s if isinstance(s, int) else env.get(s.name, 1024))
    growth = max(1.0, math.log2(length) / 4.0)
    return base * growth


def element_floor(k: ir.Kernel) -> float:
    """The element floor fraction for k: FLOOR_REDUCE when it reduces, else FLOOR_ELEMENTWISE."""
    return FLOOR_REDUCE if list(k.reduces()) else FLOOR_ELEMENTWISE


def _where(i: int, shape) -> str:
    idx = []
    for size in reversed(tuple(shape)):
        idx.append(i % size)
        i //= size
    return "[" + ", ".join(str(x) for x in reversed(idx)) + "]"


def compare(name: str, out, ref, tol: float, floor: float = FLOOR_DEFAULT, mag=None) -> OutputCheck:
    """Compare one output with its reference. Torch tensors on one device are compared there
    (compare_torch, the same rules); anything else goes through NumPy on the CPU. `floor` is the
    element floor as a fraction of rms(ref) (module docstring). `mag`, when given, has the output's
    shape and holds each element's sum of term magnitudes; it raises that element's denominator."""
    if _both_device_tensors(out, ref):
        return compare_torch(name, out, ref, tol, floor, mag)
    return compare_numpy(name, out, ref, tol, floor, mag)


def magnitude_kernel(k: ir.Kernel) -> Optional[ir.Kernel]:
    """``k`` with the body of every sum reduction replaced by its absolute value, or None when k has
    no sum. Its outputs bound the size of what each sum added up."""
    memo: Dict = {}
    found = [False]

    def go(n):
        if n in memo:
            return memo[n]
        if isinstance(n, ir.Reduce):
            body = go(n.body)
            if n.op == "sum":
                found[0] = True
                body = ir.Unary("abs", body)
            r = ir.Reduce(n.op, n.axis, body)
        elif isinstance(n, ir.Unary):
            r = ir.Unary(n.op, go(n.x))
        elif isinstance(n, ir.Binary):
            r = ir.Binary(n.op, go(n.a), go(n.b))
        elif isinstance(n, ir.Where):
            r = ir.Where(go(n.cond), go(n.a), go(n.b))
        else:
            r = n
        memo[n] = r
        return r

    outs = [ir.Output(o.name, o.axes, go(o.expr), o.dtype) for o in k.outputs]
    return ir.Kernel(f"{k.name}.magnitude", outs) if found[0] else None


def _both_device_tensors(out, ref) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return (isinstance(out, torch.Tensor) and isinstance(ref, torch.Tensor)
            and out.device == ref.device and out.device.type != "cpu")


def compare_torch(name: str, out, ref, tol: float, floor: float = FLOOR_DEFAULT, mag=None) -> OutputCheck:
    """compare_numpy on torch tensors, step for step, in float64 on their device."""
    import torch
    out = out.detach().to(torch.float64)
    ref = ref.detach().to(torch.float64)
    if tuple(out.shape) != tuple(ref.shape):
        return OutputCheck(name, math.inf, tol,
                           f"shape {tuple(out.shape)} != reference {tuple(ref.shape)}")
    nan_o, nan_r = torch.isnan(out), torch.isnan(ref)
    n = int((nan_o != nan_r).sum().item())
    if n:
        return OutputCheck(name, math.inf, tol, f"NaN at {n} places that differ from the reference")
    inf_o, inf_r = torch.isinf(out), torch.isinf(ref)
    if bool((inf_o != inf_r).any().item()) or bool((out[inf_r] != ref[inf_r]).any().item()):
        return OutputCheck(name, math.inf, tol, "infinities differ from the reference")
    fin = ~(nan_r | inf_r)
    if not bool(fin.any().item()):
        return OutputCheck(name, 0.0, tol)
    r = torch.abs(torch.where(fin, ref, 0.0))
    diff = torch.abs(torch.where(fin, out - ref, 0.0))
    big = float(r.max().item())
    rms = math.sqrt(float((r * r).sum().item()) / int(fin.sum().item()))
    lo = max(min(floor * rms, big), SCALE_FLOOR)
    den = torch.clamp(r, min=lo)
    if mag is not None:
        den = torch.maximum(den, torch.abs(torch.as_tensor(mag, dtype=torch.float64, device=den.device))
                            .reshape(den.shape))
    e = diff / den
    i = int(torch.argmax(e).item())
    return OutputCheck(name, float(e.reshape(-1)[i].item()), tol, at=_where(i, e.shape))


def compare_numpy(name: str, out: np.ndarray, ref: np.ndarray, tol: float,
                  floor: float = FLOOR_DEFAULT, mag=None) -> OutputCheck:
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if out.shape != ref.shape:
        return OutputCheck(name, math.inf, tol, f"shape {out.shape} != reference {ref.shape}")
    nan_o, nan_r = np.isnan(out), np.isnan(ref)
    if (nan_o != nan_r).any():
        n = int((nan_o != nan_r).sum())
        return OutputCheck(name, math.inf, tol, f"NaN at {n} places that differ from the reference")
    inf_o, inf_r = np.isinf(out), np.isinf(ref)
    if (inf_o != inf_r).any() or (out[inf_r] != ref[inf_r]).any():
        return OutputCheck(name, math.inf, tol, "infinities differ from the reference")
    fin = ~(nan_r | inf_r)
    if not fin.any():
        return OutputCheck(name, 0.0, tol)
    r = np.abs(np.where(fin, ref, 0.0))
    diff = np.abs(np.subtract(out, ref, out=np.zeros_like(ref), where=fin))   # never inf - inf
    big = float(r.max())
    rms = math.sqrt(float((r * r).sum()) / int(fin.sum()))
    lo = max(min(floor * rms, big), SCALE_FLOOR)
    den = np.maximum(r, lo)
    if mag is not None:
        den = np.maximum(den, np.abs(np.asarray(mag, dtype=np.float64)).reshape(den.shape))
    e = diff / den
    i = int(np.argmax(e))
    return OutputCheck(name, float(e.reshape(-1)[i]), tol, at=_where(i, e.shape))


def check_outputs(k: ir.Kernel, backend: str, outs: Dict, refs: Dict, tol: float, magnitude=None,
                  tf32: bool = False, rounded=None):
    """Raise KernelMismatchError naming every output over ``tol``. ``magnitude``: a callable that
    returns the outputs of ``magnitude_kernel(k)`` on the same inputs; it is called only when an
    output fails without a NaN, infinity or shape difference, and that output is measured again
    against the magnitude of its sums' terms. ``rounded``: a callable returning the rounded run
    (reference.RoundedEvaluator) as NumPy arrays; for a kernel that computes in float32, an output over
    ``tol`` (no NaN, infinity or shape difference) that is within ``tol`` of the rounded run passes:
    the kernel computes the formula as a correct float32 kernel does, and the formula cancels."""
    bad = []
    for o in k.outputs:
        c = compare(o.name, outs[o.name], refs[o.name], tol, element_floor(k))
        if not c.ok:
            bad.append(c)
    f32_math = not any(o.dtype == "float64" for o in k.outputs)
    if bad and rounded is not None and f32_math and not any(c.detail for c in bad):
        derived = rounded()            # (rounded run, sigma) per output, docs/TOLERANCE.md
        if derived is not None:
            keep = []
            for c in bad:
                if not _within_derived(outs[c.name], refs[c.name], *derived[c.name]):
                    keep.append(c)
            bad = keep
    if bad and magnitude is not None and not any(c.detail for c in bad):
        mags = magnitude()
        if mags is not None:
            again = []
            for c in bad:
                c2 = compare(c.name, outs[c.name], refs[c.name], tol, element_floor(k), mags[c.name])
                if not c2.ok:
                    again.append(OutputCheck(c.name, c.error, c.tol,
                                             f"relative error {c.error:.3g} at element {c.at} > tolerance "
                                             f"{c.tol:.3g}, and {c2.error:.3g} at element {c2.at} against the "
                                             f"magnitude of its sums' terms", c.at))
            bad = again
    if bad:
        parts = []
        for c in bad:
            why = c.detail or (f"relative error {c.error:.3g} at element {c.at} > tolerance "
                               f"{c.tol:.3g}")
            parts.append(f"output {c.name}: {why}")
        raise KernelMismatchError(
            f"kernel {k.name!r} on {backend} does not match the float64 reference: "
            + "; ".join(parts)
            + f".\n{k.to_text()}\nThis is a bug in the Truestar code generator; the result was "
              f"not used. Set TRUESTAR_VERIFY=0 only to debug.")
    check_lean(k, backend, outs, refs, tol, magnitude, tf32)


def _within_derived(out, ref, rounded, sigma) -> bool:
    """The verdict of docs/TOLERANCE.md for one output: every element within lambda sigma + |rounded - ref|
    + 1 ulp of float32 (+ 1e-3 u rms(ref)), and the RMS error within 2 times the RMS of sigma + ulp/2 +
    |rounded - ref|. NaN and infinity must match float64 exactly."""
    o = to_host_f64(out).numpy() if hasattr(out, "detach") else np.asarray(out, dtype=np.float64)
    r = to_host_f64(ref).numpy() if hasattr(ref, "detach") else np.asarray(ref, dtype=np.float64)
    if o.shape != r.shape:
        return False
    fin = np.isfinite(r)
    if not np.array_equal(np.isfinite(o), fin) or not np.array_equal(np.isnan(o), np.isnan(r)):
        return False
    u = 2.0 ** -24
    a = np.abs(np.where(fin, r, 0.0))
    ulp = np.where(a > 0, np.spacing(a.astype(np.float32)).astype(np.float64), 2.0 ** -149)
    rms_ref = float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0
    dr = np.abs(np.where(fin, rounded - r, 0.0))
    err = np.abs(np.where(fin, o - r, 0.0))
    from ..tolerance import element_lambda          # sigma from 8 runs: Student t, docs/TOLERANCE.md
    bound = element_lambda(8, int(fin.sum())) * sigma + dr + ulp + 1e-3 * u * rms_ref
    if np.any(err > bound):
        return False
    if err.size == 0:
        return True
    lim = 2.0 * float(np.sqrt(np.mean((sigma + ulp / 2 + dr) ** 2)))
    return float(np.sqrt(np.mean(err ** 2))) <= lim


_FLOAT_OUT = ("float64", "float32", "float16", "bfloat16")


def bias_mode() -> str:
    """'on' (default): a biased kernel raises KernelBiasError. 'warn': it warns. 'off': no test."""
    v = os.environ.get("TRUESTAR_VERIFY_BIAS", "on").strip().lower()
    if v in ("0", "off", "false", "no"):
        return "off"
    if v in ("warn", "warning"):
        return "warn"
    return "on"


def error_groupings(k: ir.Kernel, o: ir.Output) -> List[Tuple[int, ...]]:
    """The output dims each rounding of ``o`` is shared along, as sets of dims held fixed.

    Every node of the expression that rounds (an arithmetic op, a reduction, a load of a float64
    buffer into a lower precision kernel) depends on some of the output's axes; output elements that
    agree on those axes read the same rounded value, so their errors may be correlated. The lean
    test treats each such set as a grouping (docs/BIAS.md). Pure constant arithmetic (Const and Dim
    only) is left out: its single rounding is shared by every element and is what the effect floor
    allows for. A non constant node that depends on no output axis (a global sum broadcast to every
    element) gives the empty set: one group, so that output is never called biased."""
    prec = ir.lowest_precision([b.dtype for b in k.inputs] + [x.dtype for x in k.outputs])
    pos = {a: i for i, a in enumerate(o.axes)}
    data: Dict = {}

    def has_data(n) -> bool:
        if n not in data:
            if isinstance(n, (ir.Load, ir.Index, ir.Placeholder)):
                data[n] = True
            else:
                data[n] = any(has_data(c) for c in n.children())
        return data[n]

    out = {tuple(range(len(o.axes)))}
    for n in ir.walk(o.expr):
        if isinstance(n, (ir.Const, ir.Dim, ir.Index, ir.Placeholder)):
            continue
        if isinstance(n, ir.Load):
            if not (n.buffer.dtype == "float64" and prec != "float64"):
                continue    # a load in the kernel's precision (or a widening one) is exact
        elif isinstance(n, ir.Where) or (isinstance(n, ir.Unary) and n.op in ("neg", "abs", "not")):
            continue        # selects or flips a sign: no rounding of its own
        elif not has_data(n):
            continue        # constant arithmetic: covered by the effect floor
        out.add(tuple(sorted(pos[a] for a in ir.free_axes(n) if a in pos)))
    return sorted(out)


def _is_rounded_copy(out, ref, dtype: str, frac: float = 0.0) -> bool:
    """Every element of ``out`` (all but a fraction ``frac``: the float32 math before the rounding can
    cross a midpoint) equals its float64 reference rounded to nearest even in ``dtype``. A rounding
    that truncates or leans differs on about half of the elements."""
    import torch
    o = torch.as_tensor(out).detach().cpu().to(torch.float64)
    r = torch.as_tensor(ref).detach().cpu().to(torch.float64)
    if o.shape != r.shape:
        return False
    rr = r.to(getattr(torch, dtype)).to(torch.float64)
    same = (o == rr) | (torch.isnan(o) & torch.isnan(rr))
    return bool(same.all()) if frac == 0.0 else float((~same).float().mean()) <= frac


def check_lean(k: ir.Kernel, backend: str, outs: Dict, refs: Dict, tol: float, magnitude=None,
               tf32: bool = False) -> list:
    """The lean test (truestar.bias) on every float output of ``k``; raise KernelBiasError (or warn,
    by ``bias_mode``) naming every output whose errors lean. Returns the LeanResults.

    An output that leans is measured once more with each element's denominator raised to the
    magnitude of its sums' terms (``magnitude``, as in check_outputs): rounding in a sum scales with
    what it added, so a sum whose terms cancel (crafted inputs such as the order probe's, or a bias
    gradient behind batch norm) is judged in those units, and only a lean there is refused.

    ``tf32``: the kernel was compiled with allow_tf32. TF32 tensor cores on the GPUs measured here
    truncate float32 operands to 10 bits instead of rounding them, so a TF32 contraction leans
    toward zero by itself (docs/BIAS.md, measured 6.2e-4 on a positive 1024 long matmul). That is
    the precision the caller chose, not a code generator bug: it is reported once per kernel as a
    KernelBiasWarning with code TS966 and the kernel is not refused."""
    mode = bias_mode()
    if mode == "off" or getattr(k, "directed_rounding", False):
        # directed_rounding: the kernel holds a cast the program marked as leaning on purpose (the
        # neighbour search of truestar.train stochastic rounding); its size check above still runs
        return []
    from .. import bias as B
    results, memo = [], []
    for o in k.outputs:
        if o.dtype not in _FLOAT_OUT:
            continue
        r = B.lean_test(o.name, outs[o.name], refs[o.name], tol * B.BIAS_FRACTION, element_floor(k),
                        error_groupings(k, o))
        if r.biased and magnitude is not None:
            if not memo:
                memo.append(magnitude())
            mags = memo[0]
            if mags is not None and o.name in mags:
                r = B.lean_test(o.name, outs[o.name], refs[o.name], tol * B.BIAS_FRACTION,
                                element_floor(k), error_groupings(k, o), mag=mags[o.name])
                r.note = "measured against the magnitude of its sums' terms"
        if r.biased and o.dtype in ("float16", "bfloat16") and _is_rounded_copy(
                outs[o.name], refs[o.name], o.dtype, 0.0 if isinstance(o.expr, ir.Load) else 1e-3):
            # every element (a pure cast) or all but 0.1% (float32 math before the rounding can cross
            # a midpoint) is the round to nearest even of its float64 reference: the rounding is
            # correct and the lean is the data's (values just under midpoints, such as sums of two
            # bfloat16 numbers scaled by a batch norm's 1 / sqrt(1 + eps)). A truncating rounding
            # differs on about half of the elements and stays refused.
            r.note = ("lean of the data, not the rounding: the output is the round to nearest even of its "
                      "reference (" + r.describe() + ")")
            r.leans = []
        results.append(r)
    log = os.environ.get("TRUESTAR_BIAS_LOG", "").strip()
    if log:
        import json
        with open(log, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps({"kernel": k.name, "backend": backend, "tol": tol, **r.as_dict()}) + "\n")
    bad = [r for r in results if r.biased]
    if bad:
        msg = (f"kernel {k.name!r} on {backend}: errors against the float64 reference are within the "
               f"size tolerance {tol:.3g} but lean one way (TS960): " + "; ".join(r.describe() for r in bad)
               + f".\n{k.to_text()}\nUnbiased round to nearest gives a mean error of zero up to the "
               f"stated noise; a lean adds up over training steps. The result was not used. "
               f"TRUESTAR_VERIFY_BIAS=warn reports it without refusing (docs/BIAS.md).")
        if tf32:
            if k.name not in _tf32_warned:
                _tf32_warned.add(k.name)
                import warnings
                warnings.warn(f"kernel {k.name!r} on {backend} (TS966, TF32): " + "; ".join(r.describe() for r in bad)
                              + ". TF32 operands are truncated to 10 bits on this GPU, so TF32 contractions lean "
                                "toward zero; compile without allow_tf32 where that matters (training). Not "
                                "refused (docs/BIAS.md).", KernelBiasWarning, stacklevel=3)
        elif mode == "warn":
            import warnings
            warnings.warn(msg, KernelBiasWarning, stacklevel=3)
        else:
            raise KernelBiasError(msg)
    return results


_tf32_warned: set = set()


def verify_mode() -> str:
    """'first' (default): check each compiled kernel on its first call per shape signature.
    'all': check every call. 'off': never (debugging only)."""
    v = os.environ.get("TRUESTAR_VERIFY", "first").strip().lower()
    if v in ("0", "off", "false", "no"):
        return "off"
    if v in ("all", "always", "2"):
        return "all"
    return "first"


def verify_device() -> str:
    """Where the float64 reference of a GPU kernel runs. 'auto' (default): on the kernel's own
    CUDA device when the check is big enough to gain from it (reference.GPU_MIN_WORK), else the
    NumPy interpreter on the CPU. 'gpu': on the device for every CUDA kernel. 'cpu': always the
    NumPy interpreter, as before 2026-11-24. On the device, an op torch lacks in float64 falls back
    to the CPU for that op only. Either way the math, the tolerances and the schedule are the same."""
    v = os.environ.get("TRUESTAR_VERIFY_DEVICE", "auto").strip().lower()
    if v in ("cpu", "numpy", "host"):
        return "cpu"
    if v in ("gpu", "cuda", "device"):
        return "gpu"
    return "auto"


def to_host_f64(t):
    """A tensor as float64 on the CPU, for the reference and the comparison. An MPS tensor is copied
    to the CPU first and widened there: on torch 2.14.1 the combined device and dtype copy
    ``t.to("cpu", torch.float64)`` from MPS gave values no float32 can hold (docs/METAL.md)."""
    import torch
    t = t.detach()
    if t.device.type == "mps":
        return t.cpu().to(torch.float64)
    return t.to("cpu", torch.float64)


def to_device(t, device, dtype=None):
    """``t.to(device, dtype)``, in two steps when MPS is on either side (the cast on the CPU side
    of the copy), never one combined device and dtype copy; see to_host_f64."""
    import torch
    dev = torch.device(device)
    if dev.type != "mps" and t.device.type != "mps":
        return t.to(dev, dtype) if dtype is not None else t.to(dev)
    if dtype is None or t.dtype == dtype:
        return t.to(dev)
    if t.device.type == "mps":
        return t.to(dtype).to(dev) if dev.type == "mps" else t.to(dev).to(dtype)
    return t.to(dtype).to(dev)
