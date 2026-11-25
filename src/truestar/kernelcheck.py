"""`truestar kernel ref.py cand.py`: check a hand or AI written kernel against a reference over many
inputs, with derived tolerances, and say exactly which property it breaks (R1).

A one input allclose with a hand picked tolerance passes kernels that are wrong at another shape,
on strided inputs, on large or offset values, on NaN, or now and then (races), and fails kernels
that are right but round differently. This check runs the candidate on a set of input variants
built from the example inputs and judges every output with `truestar.tolerance` (a per element
bound derived from float64, no rtol or atol):

* shapes: every distinct size of the example inputs is a symbol (equal sizes are tied); each symbol
  is moved by one up and down, and all of them to odd sizes, so tails, ragged blocks and masks at
  the last tile run. A shape the reference rejects is skipped; one the candidate rejects is TS201.
* values: large magnitudes, a large offset (cancellation), small values, a constant tensor (ties)
  and NaN in one element, on the first floating input.
* strides: every input with two or more dims as a transposed and as a sliced (step 2) view with the
  same values (TS204 when only the strided run is wrong).
* dtypes: the example dtype, plus any listed (`DTYPES` in ref.py or --dtypes).
* determinism: the candidate runs twice on the same inputs; different bits are TS203.
* unwritten output: on CUDA the caching allocator is filled with NaN before each candidate call, so
  elements a kernel never writes show up as NaN (TS205).
* input mutation: inputs are compared before and after the call (TS206).
* the reference runs after the candidate, in float64, so there is nothing to read back.

On the first failing variant, a candidate built from ATen ops is traced and each op is rerun alone
in float64 on its own inputs; the first op that is wrong on its own is named (blamed_op). Custom
kernel launches (Triton, CUDA) are not ATen ops; then the report says so and the element location
(a tail, a diagonal, one index) is the blame.

File contract. ref.py: `reference(*inputs)` (or a KernelBench `Model` with `get_init_inputs()`)
and `get_inputs()`; optional `DTYPES`. cand.py: `kernel(*inputs)` or `candidate(*inputs)` (or a
KernelBench `ModelNew`).
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .codes import Issue
from .compare import DIVERGENT, OK, SUSPECT, worst
from .tolerance import LAMBDA_RMS, CloseReport, _fn64, clear_cache, derive

FAMILIES = ("large", "offset", "small", "constant", "special")


class KernelFileError(Exception):
    """ref.py or cand.py is missing, fails to import, or lacks the functions (TS020)."""


# ----------------------------------------------------------------------------- variants


@dataclass
class Variant:
    name: str
    kind: str  # base, shape, family, stride, dtype
    inputs: tuple
    dims: Dict[str, int] = field(default_factory=dict)
    dtype: Optional[torch.dtype] = None
    note: str = ""


def _floating(inputs) -> List[int]:
    return [i for i, t in enumerate(inputs) if isinstance(t, torch.Tensor) and t.is_floating_point()]


def _symbols(inputs) -> Dict[int, str]:
    """Each distinct size above 1 in the tensor inputs becomes a symbol (equal sizes tied)."""
    sizes = []
    for t in inputs:
        if isinstance(t, torch.Tensor):
            for s in t.shape:
                if s > 1 and s not in sizes:
                    sizes.append(s)
    return {s: f"d{i}" for i, s in enumerate(sizes)}


def _regen(t: torch.Tensor, shape, g: torch.Generator) -> torch.Tensor:
    """A tensor of the new shape whose values look like t's (normal with t's mean and std; integers
    uniform within t's range)."""
    if t.is_floating_point():
        x = t.detach().double()
        mean = float(x.mean()) if x.numel() else 0.0
        std = float(x.std()) if x.numel() > 1 else 1.0
        if not std == std or std == 0:
            std = 1.0
        v = torch.randn(shape, generator=g, dtype=torch.float64) * std + mean
        return v.to(t.dtype).to(t.device)
    if t.dtype == torch.bool:
        return (torch.rand(shape, generator=g) < 0.5).to(t.device)
    lo, hi = (int(t.min()), int(t.max())) if t.numel() else (0, 1)
    return torch.randint(lo, hi + 1, shape, generator=g).to(t.dtype).to(t.device)


def _reshape_inputs(inputs, mapping: Dict[int, int], g) -> tuple:
    out = []
    for t in inputs:
        if isinstance(t, torch.Tensor):
            shape = tuple(mapping.get(s, s) for s in t.shape)
            out.append(_regen(t, shape, g))
        else:
            out.append(t)
    return tuple(out)


def shape_variants(inputs, seed: int = 0) -> List[Variant]:
    syms = _symbols(inputs)
    vs: List[Variant] = []
    g = torch.Generator().manual_seed(seed)
    for size, name in syms.items():
        for new in (size + 1, size - 1):
            if new < 1:
                continue
            vs.append(Variant(f"{name}={new} (was {size})", "shape",
                              _reshape_inputs(inputs, {size: new}, g), {name: new}))
    if syms:
        odd = {s: (s // 2) * 2 + 1 if s > 3 else s for s in syms}
        odd = {s: (n + 2 if n == s else n) for s, n in odd.items()}
        dims = {syms[s]: n for s, n in odd.items()}
        vs.append(Variant("all sizes odd " + ", ".join(f"{k}={v}" for k, v in dims.items()),
                          "shape", _reshape_inputs(inputs, odd, g), dims))
        small = {s: min(s, 7) if s > 7 else s for s in syms}
        if small != {s: s for s in syms}:
            dims = {syms[s]: n for s, n in small.items()}
            vs.append(Variant("all sizes small " + ", ".join(f"{k}={v}" for k, v in dims.items()),
                              "shape", _reshape_inputs(inputs, small, g), dims))
    return vs


def family_variants(inputs) -> List[Variant]:
    fl = _floating(inputs)
    if not fl:
        return []
    i = fl[0]
    x = inputs[i]
    xd = x.detach().double()
    rms = float(xd.pow(2).mean().sqrt()) if xd.numel() else 1.0
    rms = rms if rms > 0 else 1.0
    amax = float(xd.abs().max()) if xd.numel() else 1.0
    amax = amax if amax > 0 else 1.0

    def put(v):
        new = list(inputs)
        new[i] = v.to(x.dtype).to(x.device)
        return tuple(new)

    out = [
        Variant("large values (max |x| = 100)", "family", put(xd * (100.0 / amax))),
        Variant("large offset (x + 64 rms)", "family", put(xd + 64.0 * rms)),
        Variant("small values (x / 128)", "family", put(xd / 128.0)),
        Variant("constant tensor (ties)", "family", put(torch.full_like(xd, float(xd.mean())))),
    ]
    # NaN only: infinity is not in the default set because PyTorch's own float32 kernels disagree
    # with float64 there (CPU float32 GELU(+inf) is NaN, float64 gives inf), so a candidate that
    # copies PyTorch would be flagged.
    sp = xd.clone().reshape(-1)
    if sp.numel() >= 2:
        sp[sp.numel() // 2] = float("nan")
        out.append(Variant("NaN in one element", "family", put(sp.reshape(xd.shape))))
    return out


def _transposed(t: torch.Tensor) -> torch.Tensor:
    return t.transpose(-1, -2).contiguous().transpose(-1, -2)


def _sliced(t: torch.Tensor) -> torch.Tensor:
    big = torch.empty(t.shape[:-1] + (2 * t.shape[-1],), dtype=t.dtype, device=t.device)
    big = big.fill_(float("nan")) if t.is_floating_point() else big.zero_()
    v = big[..., ::2]
    v.copy_(t)
    return v


def stride_variants(inputs) -> List[Variant]:
    idx = [i for i, t in enumerate(inputs) if isinstance(t, torch.Tensor) and t.dim() >= 2
           and t.shape[-1] > 1 and t.shape[-2] > 1]
    if not idx:
        return []
    out = []
    for fn, what in ((_transposed, "transposed storage"), (_sliced, "sliced storage (step 2)")):
        new = list(inputs)
        for i in idx:
            new[i] = fn(inputs[i])
        out.append(Variant(f"non contiguous inputs: {what}", "stride", tuple(new)))
    return out


def dtype_variants(inputs, dtypes: Sequence[torch.dtype]) -> List[Variant]:
    fl = _floating(inputs)
    have = {inputs[i].dtype for i in fl}
    out = []
    for dt in dtypes:
        if dt in have and len(have) == 1:
            continue
        new = tuple(t.to(dt) if i in fl else t for i, t in enumerate(inputs))
        out.append(Variant(f"dtype {str(dt).replace('torch.', '')}", "dtype", new, dtype=dt))
    return out


# ----------------------------------------------------------------------------- running


def _poison(device: torch.device, nbytes: int) -> None:
    """Fill the CUDA caching allocator's free blocks with NaN so unwritten output shows. Blocks of
    1 KB to 1 MB land in the small pool's segments and one larger block in the large pool, the two
    places a kernel's output can be carved from."""
    if device.type != "cuda":
        return
    try:
        held = [torch.empty(2 ** k // 4, dtype=torch.float32, device=device) for k in range(10, 21)]
        held.append(torch.empty(max(2 ** 20, nbytes) // 4, dtype=torch.float32, device=device))
        for t in held:
            t.fill_(float("nan"))
        del held
    except Exception:
        pass


def _clone(inputs):
    return tuple(t.clone() if isinstance(t, torch.Tensor) else copy.deepcopy(t) for t in inputs)


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return bool(torch.equal(torch.nan_to_num(a), torch.nan_to_num(b))) and bool(
            torch.equal(torch.isnan(a), torch.isnan(b))) if a.is_floating_point() else bool(
            torch.equal(a, b))
    return True


def _outputs(o) -> List[torch.Tensor]:
    from .capture import flatten_tensors

    return list(flatten_tensors(o))


@dataclass
class VariantResult:
    name: str
    kind: str
    status: str  # ok, divergent, skipped, error
    code: str = ""
    message: str = ""
    worst_ulps: Optional[float] = None
    worst_ratio: Optional[float] = None
    location: str = ""
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        def num(x):
            return x if isinstance(x, (int, float)) and x == x and abs(x) != float("inf") else None
        return {"name": self.name, "kind": self.kind, "status": self.status, "code": self.code,
                "message": self.message, "worst_ulps": num(self.worst_ulps),
                "worst_ratio": num(self.worst_ratio), "location": self.location,
                "seconds": round(self.seconds, 4)}


@dataclass
class BlamedOp:
    op: str
    index: int
    total: int
    message: str

    def as_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "index": self.index, "total": self.total, "message": self.message}


@dataclass
class KernelReport:
    reference_name: str
    candidate_name: str
    device: str
    status: str = OK
    variants: List[VariantResult] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    blamed_op: Optional[BlamedOp] = None
    op_note: str = ""
    first_failure: Optional[VariantResult] = None
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)
    candidate_source: Optional[Dict[str, Any]] = None

    @property
    def numerics(self) -> str:
        """ok or divergent: the verdict on the numbers alone, without the TS202 and TS203
        verdicts (NaN handling, determinism)."""
        return DIVERGENT if any(v.status == DIVERGENT for v in self.variants) else OK

    @property
    def codes(self) -> List[str]:
        out: List[str] = []
        for i in self.issues:
            if i.code not in out:
                out.append(i.code)
        return out

    def add(self, code: str, message: str) -> None:
        if not any(i.code == code for i in self.issues):  # first of each code only
            self.issues.append(Issue(code, message, self.device))

    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for v in self.variants:
            c[v.status] = c.get(v.status, 0) + 1
        return c

    def to_text(self) -> str:
        lines = [f"truestar kernel: {self.candidate_name} against {self.reference_name} on "
                 f"{self.device}: {self.status.upper()}"]
        c = self.counts()
        lines.append(f"  {len(self.variants)} input variants: "
                     + ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
                     + f" ({self.seconds:.2f} s)")
        for i in self.issues:
            lines.append(f"  [{i.code}] {i.message}")
        if self.blamed_op is not None:
            lines.append(f"  blamed op: {self.blamed_op.op} (op {self.blamed_op.index + 1} of "
                         f"{self.blamed_op.total}): {self.blamed_op.message}")
        elif self.op_note:
            lines.append(f"  op blame: {self.op_note}")
        for v in self.variants:
            if v.status not in (OK,):
                lines.append(f"    {v.status:9s} {v.kind:6s} {v.name}: {v.message[:220]}")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "numerics": self.numerics, "codes": self.codes,
                "issues": [i.as_dict() for i in self.issues],
                "variants": [v.as_dict() for v in self.variants],
                "variant_counts": self.counts(),
                "first_failure": None if self.first_failure is None else self.first_failure.as_dict(),
                "blamed_op": None if self.blamed_op is None else self.blamed_op.as_dict(),
                "op_note": self.op_note, "seconds": round(self.seconds, 3),
                "notes": list(self.notes)}


def _call(fn, inputs, device: torch.device):
    _poison(device, sum(t.numel() * t.element_size() * 4 for t in inputs
                        if isinstance(t, torch.Tensor)))
    out = fn(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return out


# The lean test (truestar.bias, the TS960 statistic) runs on the example input only. Its null is
# that each element's rounding error has mean zero, which holds when the intermediate values spread
# over many rounding intervals. The value families squeeze them into a few: x / 128 puts a sigmoid
# just above 0.5, where bfloat16 rounds every value down to 0.5 (a correct unfused bfloat16 SiLU
# leans toward zero, measured), large values saturate it at 1, a constant tensor gives every
# element the same error (so an example input with a constant floating tensor is skipped too).


def _lean(tol, out) -> str:
    """The lean test of each floating output against float64; a message when one fires. Effect
    floor: BIAS_FRACTION of the compiled kernels' size tolerance for the output dtype
    (truestar.kernel.verify.BASE_TOL), the floor that test uses."""
    from . import bias as B
    from .kernel.verify import BASE_TOL

    for i, (o, r) in enumerate(zip(_outputs(out), tol.reference)):
        if not o.is_floating_point() or o.shape != r.shape or o.numel() < 2:
            continue
        floor = B.BIAS_FRACTION * BASE_TOL.get(str(o.dtype).replace("torch.", ""), 2e-5)
        lr = B.lean_test(f"output {i}", o.detach(), r, floor, 1.0)
        if lr.biased:
            return lr.describe()
    return ""


_LOW_PRECISION = (("TF32", r"allow_tf32\s*=\s*True|"
                           r"set_float32_matmul_precision\(\s*['\"](high|medium)"),
                  ("float16", r"\.half\(\)|__half|at::Half|torch\.float16|__float2half|half2"),
                  ("bfloat16", r"torch\.bfloat16|__nv_bfloat16|at::BFloat16|__float2bfloat16"))


def _reduced_precision_in_source(candidate) -> str:
    """Which reduced precision the candidate's source file asks for (TF32 matmuls, float16 or
    bfloat16 compute), or "" (a source scan; no source, no verdict)."""
    import re

    src = _source(candidate)
    try:
        text = open(src["file"], encoding="utf-8", errors="replace").read() if src else ""
    except OSError:
        return ""
    found = [name for name, pat in _LOW_PRECISION if re.search(pat, text)]
    return " and ".join(found)


def _nan_semantics(reference, inputs, out, device) -> str:
    """Name the NaN rule the candidate follows when it is one of IEEE 754-2019's: maximum and
    minimum propagate NaN (the float64 reference does), maximumNumber and minimumNumber return
    the other operand (CUDA fmaxf and fminf). A NaN dropped by maximumNumber is the reference run
    with that NaN as minus infinity; by minimumNumber, as plus infinity. The candidate matches one
    of them within the derived tolerance, or the message stays generic."""
    for value, rule in ((float("-inf"), "maximumNumber (fmaxf)"),
                        (float("inf"), "minimumNumber (fminf)")):
        new = tuple(torch.nan_to_num(t, nan=value) if isinstance(t, torch.Tensor)
                    and t.is_floating_point() else t for t in inputs)
        try:
            clear_cache()
            if derive(reference, new, like=out, device=device).check(out, explain=False).ok:
                return (f"NaN is dropped as IEEE 754-2019 {rule} does, while the reference "
                        f"propagates it as maximum and minimum do")
        except Exception:
            continue
    return ""


def _out_of_memory(exc: BaseException) -> bool:
    return isinstance(exc, getattr(torch, "OutOfMemoryError", MemoryError)) or isinstance(
        exc, MemoryError) or "out of memory" in str(exc).lower()


def _judge(reference, candidate, v: Variant, device: torch.device, report: KernelReport,
           base_ok: Dict[str, bool],
           lean: bool = False) -> Tuple[VariantResult, Optional[CloseReport]]:
    t0 = time.perf_counter()
    inputs = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in v.inputs)
    if v.kind == "stride":  # keep the layout: build it on the device
        inputs = v.inputs
    # The reference must accept the variant, or it is not a valid input of this kernel.
    try:
        with torch.no_grad():
            _fn64(reference, device)(*tuple(t.double() if isinstance(t, torch.Tensor)
                                            and t.is_floating_point() else t
                                            for t in _clone(inputs)))
    except Exception as exc:
        if _out_of_memory(exc):  # the machine ran out, not an input the reference rejects
            raise
        return VariantResult(v.name, v.kind, "skipped",
                             message=f"the reference rejects it ({type(exc).__name__}: "
                                     f"{str(exc)[:120]})", seconds=time.perf_counter() - t0), None
    before = _clone(inputs)
    try:
        with torch.no_grad():
            out = _call(candidate, inputs, device)
    except Exception as exc:
        if _out_of_memory(exc):
            raise
        return VariantResult(v.name, v.kind, DIVERGENT, "TS201",
                             f"the candidate raised where the reference runs: "
                             f"{type(exc).__name__}: {str(exc)[:200]}",
                             seconds=time.perf_counter() - t0), None
    mutated = [i for i, (a, b) in enumerate(zip(before, inputs)) if not _same(a, b)]
    if mutated:
        return VariantResult(v.name, v.kind, DIVERGENT, "TS206",
                             f"the candidate wrote into input(s) {mutated}",
                             seconds=time.perf_counter() - t0), None
    clear_cache()
    tol = derive(reference, before, like=out, device=device)
    rep = tol.check(out, explain=True, fn=reference, inputs=before)
    if rep.ok:
        w = rep.worst
        varied = all(t.numel() < 2 or bool(t.amax() != t.amin()) for t in before
                     if isinstance(t, torch.Tensor) and t.is_floating_point())
        msg = _lean(tol, out) if lean and v.kind == "base" and varied else ""
        if msg:
            wide = all(o.dtype in (torch.float32, torch.float64)
                       for o in _outputs(out) if o.is_floating_point())
            low = _reduced_precision_in_source(candidate) if wide else ""
            if low:  # the lean is what the source asks for: its own verdict, not a wrong number
                return VariantResult(v.name, v.kind, SUSPECT, "TS1301",
                                     f"the errors lean one way and the source computes in {low}: "
                                     + msg, worst_ulps=None if w is None else w.max_ulps,
                                     worst_ratio=None if w is None else w.worst_ratio,
                                     seconds=time.perf_counter() - t0), rep
            return VariantResult(v.name, v.kind, DIVERGENT, "TS1300",
                                 "within the derived tolerance, but the errors lean one way: " + msg,
                                 worst_ulps=None if w is None else w.max_ulps,
                                 worst_ratio=None if w is None else w.worst_ratio,
                                 seconds=time.perf_counter() - t0), rep
        return VariantResult(v.name, v.kind, OK, worst_ulps=None if w is None else w.max_ulps,
                             worst_ratio=None if w is None else w.worst_ratio,
                             seconds=time.perf_counter() - t0), rep
    w = rep.worst
    code = "TS200"
    msg = rep.message()
    nan_where_finite = any(o.nonfinite_mismatch for o in rep.outputs)
    if nan_where_finite and device.type == "cuda" and v.kind not in ("family", "stride"):
        outs = _outputs(out)
        if any(bool(torch.isnan(o).any()) for o in outs if o.is_floating_point()) and not any(
                bool(torch.isnan(t).any()) for t in before if isinstance(t, torch.Tensor)
                and t.is_floating_point()):
            code = "TS205"
            msg = ("output elements were never written (the NaN the allocator was filled with "
                   "shows through): " + msg)
    if code == "TS200" and nan_where_finite and v.kind == "family" and "NaN" in v.name:
        code = "TS202"
        finite_ok = all(not o.shape_mismatch and not o.bad and o.rms_ratio <= LAMBDA_RMS
                        for o in rep.outputs)
        if finite_ok:  # only the NaN went another way: its own verdict, not a wrong number
            sem = _nan_semantics(reference, before, out, device)
            if sem:
                msg = sem + "; " + msg
            return VariantResult(v.name, v.kind, SUSPECT, code, msg,
                                 worst_ulps=None if w is None else w.max_ulps,
                                 worst_ratio=None if w is None else w.worst_ratio,
                                 location=w.location if w is not None else "",
                                 seconds=time.perf_counter() - t0), rep
    if code == "TS200" and v.kind == "family" and "NaN" in v.name and not nan_where_finite:
        # An index or integer output (argmax, argmin) has no NaN to compare: the NaN rule shows
        # only as other values. It is the NaN verdict when the output is exactly what a NaN
        # dropping maximumNumber or minimumNumber gives.
        sem = _nan_semantics(reference, before, out, device)
        if sem:
            return VariantResult(v.name, v.kind, SUSPECT, "TS202", sem + "; " + msg,
                                 seconds=time.perf_counter() - t0), rep
    if code == "TS200" and v.kind == "stride" and base_ok.get("base", False):
        code = "TS204"
        msg = "wrong on non contiguous inputs while contiguous inputs are right: " + msg
    return VariantResult(v.name, v.kind, DIVERGENT, code, msg,
                         worst_ulps=None if w is None else w.max_ulps,
                         worst_ratio=None if w is None else w.worst_ratio,
                         location=w.location if w is not None else "",
                         seconds=time.perf_counter() - t0), rep


_CODE_TEXT = {
    "TS200": "wrong result beyond the derived tolerance",
    "TS201": "raises at a shape the reference accepts",
    "TS202": "NaN or infinity handled differently from the reference",
    "TS204": "wrong only on non contiguous inputs",
    "TS205": "leaves output elements unwritten",
    "TS206": "writes into its inputs",
    "TS1300": "errors within the tolerance lean one way",
    "TS1301": "errors lean one way, from the reduced precision its source asks for",
}


def blame_op(candidate, reference, inputs, device: torch.device) -> Tuple[Optional[BlamedOp], str]:
    """Trace the candidate's ATen ops; rerun each floating op alone in float64 on its own inputs
    and judge it with a derived tolerance. The first op that is wrong on its own is named."""
    from torch.utils._python_dispatch import TorchDispatchMode

    from .capture import map_tensors

    recs = []

    class Rec(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            saved = map_tensors((args, kwargs), lambda t: t.detach().clone())
            out = func(*args, **kwargs)
            outs = [o for o in _outputs(out) if o.is_floating_point()]
            mutable = func._schema.is_mutable or any(
                r.alias_info is not None for r in func._schema.returns)
            if outs and not mutable:
                recs.append((func, saved, out.detach().clone() if isinstance(out, torch.Tensor)
                             else type(out)(o.detach().clone() if isinstance(o, torch.Tensor)
                                            else o for o in out)))
            return out

    try:
        with torch.no_grad(), Rec():
            candidate(*_clone(inputs))
    except Exception as exc:
        return None, f"tracing the candidate failed ({type(exc).__name__}: {exc})"
    float_ops = [r for r in recs if str(r[0].overloadpacket) not in
                 ("aten.empty", "aten.empty_like", "aten.zeros", "aten.full", "aten.lift_fresh",
                  "aten._to_copy", "aten.clone", "aten.detach")]
    if not recs:
        return None, ("the candidate runs no ATen op that computes floating values: its work is "
                      "in custom kernel launches, so the element location is the blame")
    for i, (func, (a, k), out) in enumerate(float_ops):
        def one(*xs, _f=func, _k=k):
            return _f(*xs, **_k)
        try:
            clear_cache()
            tol = derive(one, a, like=out, device=device, cache=False)
            rep = tol.check(out, explain=False)
        except Exception:
            continue
        if not rep.ok:
            return BlamedOp(str(func), i, len(float_ops), rep.message()), ""
    return None, (f"each of the {len(float_ops)} ATen ops the candidate runs is right on its own "
                  "inputs: the bug is in how they are combined (indexing, slicing, the order or "
                  "precision of steps) or in a custom kernel launch")


def check_kernel(reference: Callable, candidate: Callable, example_inputs, *,
                 device=None, dtypes: Optional[Sequence[torch.dtype]] = None,
                 shapes: bool = True, families: bool = True, strides: bool = True,
                 repeats: int = 2, extra_variants: Sequence[tuple] = (), drill_ops: bool = True,
                 lean: bool = True,
                 reference_name: str = "reference", candidate_name: str = "candidate",
                 seed: int = 0) -> KernelReport:
    """Check `candidate` against `reference` over many inputs built from `example_inputs`.

    Both take the same positional inputs. Returns a KernelReport; status is ok, suspect
    (every number within tolerance, but nondeterministic, TS203, or NaN handled differently from
    float64 while every finite element is right, TS202) or divergent (a wrong number).
    `report.numerics` is the verdict on the numbers alone. `lean` (on by default) adds the lean
    test on the example input (TS1300, divergent)."""
    t0 = time.perf_counter()
    base = tuple(example_inputs) if not isinstance(example_inputs, torch.Tensor) \
        else (example_inputs,)
    if device is None:
        dev = next((t.device for t in base if isinstance(t, torch.Tensor)), torch.device("cpu"))
    else:
        dev = torch.device(device)
    base = tuple(t.to(dev) if isinstance(t, torch.Tensor) else t for t in base)
    report = KernelReport(reference_name, candidate_name, str(dev))
    variants = [Variant("example inputs", "base", base)]
    variants += [Variant(f"extra input set {i}", "extra", tuple(x)) for i, x in
                 enumerate(extra_variants)]
    if shapes:
        variants += shape_variants(base, seed)
    if families:
        variants += family_variants(base)
    if strides:
        variants += stride_variants(base)
    if dtypes:
        variants += dtype_variants(base, dtypes)
    base_ok: Dict[str, bool] = {}
    first_rep = None
    for v in variants:
        if v.kind == "stride":
            v.inputs = tuple(t.to(dev) if isinstance(t, torch.Tensor) and t.device != dev else t
                             for t in v.inputs)
        res, rep = _judge(reference, candidate, v, dev, report, base_ok, lean=lean)
        if v.kind == "base":
            base_ok["base"] = res.status == OK
        report.variants.append(res)
        if res.status == SUSPECT:  # TS202, TS1301
            report.status = worst(report.status, SUSPECT)
            report.add(res.code, f"{_CODE_TEXT.get(res.code, res.code)} on {v.name}: "
                                 f"{res.message[:400]}")
        if res.status == DIVERGENT:
            report.status = DIVERGENT
            if report.first_failure is None:
                report.first_failure = res
                first_rep = v
            report.add(res.code, f"{_CODE_TEXT.get(res.code, res.code)} on {v.name}: "
                                 f"{res.message[:400]}")
    # Determinism: the same inputs twice, bit for bit.
    for v in variants[:2] if repeats >= 2 else []:
        try:
            with torch.no_grad():
                outs = [_outputs(_call(candidate, _clone(v.inputs), dev)) for _ in range(repeats)]
        except Exception:
            continue
        differ = any(not _same(a, b) for o in outs[1:] for a, b in zip(outs[0], o))
        if differ:
            res = VariantResult(f"{v.name}, run {repeats} times", "repeat", SUSPECT, "TS203",
                                "the same inputs give different bits from run to run (a race or "
                                "an order that depends on timing)")
            report.variants.append(res)
            report.status = worst(report.status, SUSPECT)
            report.add("TS203", f"nondeterministic on {v.name}: {res.message}")
            break
    if report.first_failure is not None and drill_ops and first_rep is not None:
        try:
            report.blamed_op, report.op_note = blame_op(candidate, reference, first_rep.inputs, dev)
        except Exception as exc:
            report.op_note = f"op blame failed: {type(exc).__name__}: {exc}"
    skipped = [v for v in report.variants if v.status == "skipped"]
    if skipped:
        report.notes.append(f"{len(skipped)} variant(s) skipped because the reference rejects "
                            f"them (first: {skipped[0].name}: {skipped[0].message})")
    report.candidate_source = _source(candidate)
    report.seconds = time.perf_counter() - t0
    return report


def _source(fn) -> Optional[Dict[str, Any]]:
    target = type(fn) if isinstance(fn, nn.Module) else fn
    try:
        target = getattr(target, "forward", target) if isinstance(fn, nn.Module) else target
        f = inspect.getsourcefile(target)
        line = inspect.getsourcelines(target)[1]
        return {"file": os.path.abspath(f), "line": line,
                "qualname": getattr(target, "__qualname__", str(target))}
    except Exception:
        return None


# ----------------------------------------------------------------------------- files and CLI


def _load(path: str, tag: str):
    if not os.path.isfile(path):
        raise KernelFileError(f"{path} does not exist")
    name = f"truestar_kernel_{tag}_{abs(hash(os.path.abspath(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    d = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        raise KernelFileError(f"importing {path} failed: {type(exc).__name__}: {exc}") from exc
    return mod


def _dtype_list(v) -> List[torch.dtype]:
    if not v:
        return []
    items = v.split(",") if isinstance(v, str) else list(v)
    return [getattr(torch, str(x).strip().replace("torch.", "")) for x in items if str(x).strip()]


def load_pair(ref_path: str, cand_path: str, device=None):
    """(reference, candidate, example inputs, dtypes from ref.py) from the two files."""
    ref = _load(ref_path, "ref")
    cand = _load(cand_path, "cand")
    if not hasattr(ref, "get_inputs"):
        raise KernelFileError(f"{ref_path} must define get_inputs()")
    torch.manual_seed(0)
    inputs = ref.get_inputs()
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    dev = torch.device(device) if device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if hasattr(ref, "reference"):
        reference = ref.reference
    elif hasattr(ref, "Model"):
        init = ref.get_init_inputs() if hasattr(ref, "get_init_inputs") else []
        torch.manual_seed(0)
        reference = ref.Model(*init).to(dev).eval()
    else:
        raise KernelFileError(f"{ref_path} must define reference(*inputs) or a class Model")
    for name in ("kernel", "candidate"):
        if hasattr(cand, name):
            candidate = getattr(cand, name)
            break
    else:
        if hasattr(cand, "ModelNew"):
            init = ref.get_init_inputs() if hasattr(ref, "get_init_inputs") else []
            torch.manual_seed(0)
            candidate = cand.ModelNew(*init).to(dev).eval()
            if isinstance(reference, nn.Module):
                try:
                    candidate.load_state_dict(reference.state_dict(), strict=False)
                except Exception:
                    pass
        else:
            raise KernelFileError(f"{cand_path} must define kernel(*inputs), candidate(*inputs) "
                                  f"or a class ModelNew")
    return reference, candidate, inputs, _dtype_list(getattr(ref, "DTYPES", None)), dev


def report_to_agent(report: KernelReport, ref_path: Optional[str],
                    cand_path: Optional[str]) -> Dict[str, Any]:
    from .agent import _envelope, exit_code_for

    doc = _envelope(report.status, exit_code_for(report.status))
    doc["command"] = "kernel"
    doc.update({
        "reference": {"name": report.reference_name,
                      "file": os.path.abspath(ref_path) if ref_path else None},
        "candidate": {"name": report.candidate_name,
                      "file": os.path.abspath(cand_path) if cand_path else None,
                      "source": report.candidate_source},
        "device": report.device,
        "tolerance": "derived per element from float64 (truestar.tolerance)",
    })
    doc.update(report.as_dict())
    doc["next_actions"] = next_actions(report, ref_path, cand_path)
    return doc


def next_actions(report: KernelReport, ref_path=None, cand_path=None) -> List[Dict[str, Any]]:
    acts: List[Dict[str, Any]] = []
    f = report.first_failure
    src = report.candidate_source or {}
    if f is not None:
        where = f" Bad elements: {f.location}." if f.location else ""
        acts.append({"kind": "inspect_candidate", "file": src.get("file"), "line": src.get("line"),
                     "command": None, "code": f.code, "variant": f.name,
                     "description": f"{_CODE_TEXT.get(f.code, f.code)} on '{f.name}'.{where}"})
        if report.blamed_op is not None:
            acts.append({"kind": "inspect_op", "op": report.blamed_op.op, "command": None,
                         "description": f"{report.blamed_op.op} is wrong on its own inputs: "
                                        f"{report.blamed_op.message[:300]}"})
    if "TS203" in report.codes:
        acts.append({"kind": "make_deterministic", "command": None,
                     "description": "The candidate gives different bits for the same inputs. Look "
                                    "for atomics, unsynchronized shared memory or a reduction "
                                    "order that depends on scheduling."})
    if acts and ref_path and cand_path:
        acts.append({"kind": "rerun", "description": "Rerun after a fix; exit code 0 means every "
                                                     "variant is within the derived tolerance.",
                     "command": f'truestar kernel "{os.path.abspath(ref_path)}" '
                                f'"{os.path.abspath(cand_path)}" --format agent'})
    return acts


def cli(a) -> int:
    """`truestar kernel ref.py cand.py` (called from truestar.cli)."""
    import contextlib

    from .agent import dumps, error_to_agent

    agent = a.format == "agent"
    quiet = contextlib.redirect_stdout(sys.stderr) if agent else contextlib.nullcontext()
    failure = None
    with quiet:
        try:
            reference, candidate, inputs, dts, dev = load_pair(a.reference, a.candidate, a.device)
            dts = _dtype_list(a.dtypes) or dts
            report = check_kernel(reference, candidate, inputs, device=dev, dtypes=dts,
                                  shapes=not a.no_shapes, repeats=a.repeats,
                                  lean=not getattr(a, "no_lean", False),
                                  reference_name=os.path.basename(a.reference),
                                  candidate_name=os.path.basename(a.candidate))
        except KernelFileError as exc:
            failure = ("TS020", str(exc))
        except Exception as exc:
            failure = ("TS021", f"{type(exc).__name__}: {exc}")
    if failure is not None:
        if agent:
            doc = error_to_agent(failure[0], failure[1], a.reference)
            doc["command"] = "kernel"
            print(dumps(doc))
        else:
            print(f"error [{failure[0]}]: {failure[1]}", file=sys.stderr)
        return 2
    doc = report_to_agent(report, a.reference, a.candidate)
    if agent:
        print(dumps(doc))
    else:
        print(report.to_text())
    if getattr(a, "json_path", None):
        with open(a.json_path, "w", encoding="utf-8") as fh:
            fh.write(dumps(doc))
    return 0 if report.status == OK else 1
