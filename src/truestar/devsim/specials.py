"""`truestar probe --specials`: how each backend treats NaN, infinity, subnormals, minus zero and
casts, against IEEE 754.

Every case is one small op on a handful of values. Each backend's result is compared, element by
element, with two references:

* ``float64``: the same op in float64 on the CPU (exact IEEE semantics, no rounding to speak of);
* ``ieee``: IEEE arithmetic in the backend's own precision. For a real backend that is the op in
  the probe dtype with each op rounded once (simulated in float64, so the CPU's own kernels are not
  trusted as the reference); for a simulated backend it is its profile's ideal (same storage
  dtype, IEEE rules).

An element is classified as NaN, +inf, -inf, +0, -0, subnormal or finite. A result whose class
differs from ``ieee`` (or whose finite value is off by more than 1% relative) is a departure from
IEEE (TS523). A result that matches ``ieee`` but not ``float64`` differs only as far as the
precision explains (TS524, for example a float16 overflow to infinity). Float to integer casts of
NaN, infinity and out of range values are undefined in C and implementation defined in PyTorch:
they are listed, never judged, and flagged only when backends disagree with each other. The same
holds for results IEEE leaves to the implementation: an overflowing dot product is 0, NaN or
infinity depending on whether products round before they are summed, a mean of large values
overflows or not depending on the accumulator, and relu(-0) is -0 under IEEE 754-2008 maxNum but
+0 under IEEE 754-2019 maximum.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from .profiles import Profile, get_profile

NAN, INF = math.nan, math.inf


@dataclass(frozen=True)
class Case:
    name: str
    what: str  # the op, for the table
    fn: Callable[..., torch.Tensor]
    inputs: Callable[[torch.dtype], List[List[float]]]  # per probe dtype (subnormals differ)
    impl_defined: bool = False


def _tiny(dt):
    return torch.finfo(dt).tiny


def _big(dt):
    return math.sqrt(torch.finfo(dt).max) * 2.0


CASES: List[Case] = [
    Case("add_nan", "x + 1", lambda x: x + 1, lambda d: [[NAN, 1.0]]),
    Case("relu_specials", "relu(x)", torch.relu, lambda d: [[NAN, -INF, INF]]),
    Case("relu_minus_zero", "relu(x)", torch.relu, lambda d: [[-0.0]], impl_defined=True),
    Case("sigmoid_specials", "sigmoid(x)", torch.sigmoid, lambda d: [[NAN, INF, -INF]]),
    Case("tanh_specials", "tanh(x)", torch.tanh, lambda d: [[NAN, INF, -INF]]),
    Case("exp_specials", "exp(x)", torch.exp, lambda d: [[NAN, INF, -INF, 100.0]]),
    Case("log_edges", "log(x)", torch.log, lambda d: [[0.0, -0.0, -1.0, INF]]),
    Case("sqrt_minus_zero", "sqrt(x)", torch.sqrt, lambda d: [[-0.0, -1.0, INF]]),
    Case("div_by_zero", "1 / x", lambda x: 1.0 / x, lambda d: [[0.0, -0.0, INF, -INF]]),
    Case("mul_minus_zero", "x * 1", lambda x: x * 1.0, lambda d: [[-0.0, 0.0]]),
    Case("sum_nan", "sum(x)", lambda x: x.sum().reshape(1), lambda d: [[1.0, NAN, 2.0]]),
    Case("sum_inf_minus_inf", "sum(x)", lambda x: x.sum().reshape(1), lambda d: [[INF, -INF]]),
    Case("amax_nan", "amax(x)", lambda x: x.amax().reshape(1), lambda d: [[1.0, NAN, 2.0]]),
    Case("mean_overflow", "mean(x)", lambda x: x.mean().reshape(1),
         lambda d: [[torch.finfo(d).max * 0.9] * 4], impl_defined=True),
    Case("softmax_inf", "softmax(x)", lambda x: torch.softmax(x, 0), lambda d: [[INF, 1.0, 2.0]]),
    Case("softmax_all_neg_inf", "softmax(x)", lambda x: torch.softmax(x, 0), lambda d: [[-INF, -INF]]),
    Case("softmax_nan", "softmax(x)", lambda x: torch.softmax(x, 0), lambda d: [[NAN, 1.0]]),
    Case("layer_norm_nan", "layer_norm(x)", lambda x: F.layer_norm(x, x.shape),
         lambda d: [[1.0, NAN, 3.0, 4.0]]),
    Case("batch_norm_nan", "batch_norm(x) eval",
         lambda x: F.batch_norm(x.reshape(1, -1), torch.zeros(x.numel(), dtype=x.dtype, device=x.device),
                                torch.ones(x.numel(), dtype=x.dtype, device=x.device)).reshape(-1),
         lambda d: [[1.0, NAN, 3.0]]),
    Case("reshape_nan", "reshape(contiguous(x.t()))",
         lambda x: x.reshape(2, 2).t().contiguous().reshape(-1), lambda d: [[NAN, 1.0, 2.0, NAN]]),
    Case("matmul_overflow", "x @ y", lambda x, y: (x.reshape(1, -1) @ y.reshape(-1, 1)).reshape(-1),
         lambda d: [[_big(d), _big(d)], [_big(d), _big(d)]]),
    Case("matmul_inf_minus_inf", "x @ y",
         lambda x, y: (x.reshape(1, -1) @ y.reshape(-1, 1)).reshape(-1),
         lambda d: [[_big(d), _big(d)], [_big(d), -_big(d)]], impl_defined=True),
    Case("subnormal_pass", "x * 1", lambda x: x * 1.0, lambda d: [[_tiny(d) / 4, -_tiny(d) / 4]]),
    Case("subnormal_product", "x * 0.5", lambda x: x * 0.5, lambda d: [[_tiny(d), _tiny(d) * 0.75]]),
    Case("subnormal_matmul", "x @ y", lambda x, y: (x.reshape(1, -1) @ y.reshape(-1, 1)).reshape(-1),
         lambda d: [[_tiny(d) / 4], [1.0]]),
    Case("cast_to_f16_overflow", "x.to(float16)", lambda x: x.to(torch.float16),
         lambda d: [[70000.0, -70000.0, 65519.0, 65520.0]]),
    Case("cast_to_int_trunc", "x.to(int32)", lambda x: x.to(torch.int32), lambda d: [[2.5, -2.5, -0.0]]),
    Case("cast_to_int_specials", "x.to(int32)", lambda x: x.to(torch.int32),
         lambda d: [[NAN, INF, -INF, 3e9]], impl_defined=True),
]


def classify(v: float, dt: torch.dtype) -> str:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "+inf" if v > 0 else "-inf"
    if v == 0:
        return "-0" if math.copysign(1.0, v) < 0 else "+0"
    if dt.is_floating_point and abs(v) < torch.finfo(dt).tiny:
        return "subnormal"
    return "finite"


def _fmt(v: float) -> str:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v == 0:
        return "-0" if math.copysign(1.0, v) < 0 else "0"
    return f"{v:.4g}"


@dataclass
class Row:
    backend: str
    case: str
    what: str
    index: int
    inputs: str
    float64: float
    ieee: float
    got: float
    verdict: str  # departure, precision, impl_defined, impl_disagree

    def as_dict(self) -> Dict[str, Any]:
        return {"backend": self.backend, "case": self.case, "op": self.what, "index": self.index,
                "inputs": self.inputs, "float64": _fmt(self.float64), "ieee": _fmt(self.ieee),
                "got": _fmt(self.got), "verdict": self.verdict}


@dataclass
class ProbeReport:
    dtype: torch.dtype
    backends: List[str]
    rows: List[Row]
    cases: int
    seconds: float
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def departures(self) -> List[Row]:
        """Departures from IEEE plus implementation defined cases where backends disagree."""
        return [r for r in self.rows if r.verdict in ("departure", "impl_disagree")]

    @property
    def status(self) -> str:
        return "divergent" if self.departures else "ok"

    def codes(self) -> List[str]:
        out = []
        if self.departures:
            out.append("TS523")
        if any(r.verdict == "precision" for r in self.rows):
            out.append("TS524")
        if any(b.startswith("sim:") for b in self.backends):
            out.append("TS520")
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {"dtype": str(self.dtype).replace("torch.", ""), "backends": self.backends,
                "cases": self.cases, "seconds": round(self.seconds, 3), "status": self.status,
                "codes": self.codes(), "errors": dict(self.errors),
                "rows": [r.as_dict() for r in self.rows]}

    def to_text(self) -> str:
        dt = str(self.dtype).replace("torch.", "")
        lines = [f"truestar probe --specials ({dt}, {self.cases} cases, {len(self.backends)} "
                 f"backends, {self.seconds:.1f} s): departures from IEEE in the backend's own "
                 "precision"]
        for b in self.backends:
            rows = [r for r in self.rows if r.backend == b]
            dep = [r for r in rows if r.verdict == "departure"]
            prec = [r for r in rows if r.verdict == "precision"]
            if b in self.errors:
                lines.append(f"[ERROR] {b}: {self.errors[b]}")
                continue
            lines.append(f"[{'DEPARTS' if dep else 'IEEE'}] {b}: {len(dep)} departures from IEEE, "
                         f"{len(prec)} differences explained by precision")
            if dep:
                lines.append(f"  {'case':22} {'op':16} {'input':>18} {'float64':>9} {'ieee':>9} "
                             f"{'got':>9}")
            for r in dep:
                lines.append(f"  {r.case:22} {r.what[:16]:16} {r.inputs:>18} {_fmt(r.float64):>9} "
                             f"{_fmt(r.ieee):>9} {_fmt(r.got):>9}")
        impl: Dict[tuple, List[str]] = {}
        disagree = set()
        for r in self.rows:
            if r.verdict in ("impl_defined", "impl_disagree"):
                impl.setdefault((r.case, r.index, r.inputs), []).append(f"{r.backend}={_fmt(r.got)}")
                if r.verdict == "impl_disagree":
                    disagree.add((r.case, r.index, r.inputs))
        if impl:
            lines.append("left to the implementation by IEEE (listed; backends that disagree are "
                         "marked, since a model relying on either answer is not portable):")
            for key, vals in impl.items():
                mark = "DISAGREE " if key in disagree else "         "
                lines.append(f"  {mark}{key[0]} {key[2]}: " + ", ".join(vals))
        return "\n".join(lines)


def _run(fn, values: List[List[float]], dtype, device, ctx):
    import contextlib

    xs = [torch.tensor(v, dtype=dtype, device=device) for v in values]
    with torch.no_grad(), (ctx or contextlib.nullcontext()):
        out = fn(*xs)
    return out.detach().cpu()


def probe_specials(backends: Optional[List[str]] = None,
                   dtype: torch.dtype = torch.float32) -> ProbeReport:
    from ..backends import available_backends, resolve_backends
    from .mode import simulate
    from .profiles import PROFILES

    t0 = time.perf_counter()
    if backends is None:
        chosen = available_backends()
        chosen += resolve_backends(sorted(PROFILES), allow_sim=True)[0]
    else:
        chosen, _ = resolve_backends(backends, allow_sim=True)
    rows: List[Row] = []
    errors: Dict[str, str] = {}
    impl_vals: Dict[tuple, Dict[str, float]] = {}  # (case, index, inputs) -> backend -> value
    for b in chosen:
        prec = b.floor_dtype(dtype)
        if b.sim:
            ideal: Profile = get_profile(b.sim).ideal()
        else:
            from dataclasses import replace

            ideal = replace(get_profile("sim:ieee-f16"), name=f"ieee-{prec}", storage=None)
        for case in CASES:
            values = case.inputs(prec)
            try:
                ref64 = _run(case.fn, values, torch.float64, "cpu", None)
                ieee = _run(case.fn, values, dtype, "cpu", simulate(ideal))
                got = _run(case.fn, values, dtype, b.device, b.context())
            except Exception as exc:
                errors[b.name] = f"{case.name}: {type(exc).__name__}: {exc}"
                continue
            out_dt = got.dtype
            elementwise = got.numel() == len(values[0]) and len(values) == 1
            whole = ",".join(_fmt(v) for v in values[0])
            whole = whole if len(whole) <= 18 else whole[:15] + "..."
            for i, (r, e, g) in enumerate(zip(ref64.reshape(-1).tolist(), ieee.reshape(-1).tolist(),
                                              got.reshape(-1).tolist())):
                inp = _fmt(values[0][i]) if elementwise else whole
                cls_dt = out_dt if out_dt.is_floating_point else dtype
                if case.impl_defined:
                    # integers are compared by value, floats by class (NaN, inf, sign of zero)
                    impl_vals.setdefault((case.name, i, inp), {})[b.name] = (
                        _fmt(g) if not out_dt.is_floating_point else classify(g, cls_dt))
                    rows.append(Row(b.name, case.name, case.what, i, inp, r, e, g, "impl_defined"))
                    continue
                cg, ce, cr = classify(g, cls_dt), classify(e, cls_dt), classify(r, cls_dt)
                off = (cg == ce == "finite" and abs(g - e) > 1e-2 * abs(e))
                if cg != ce or off:
                    rows.append(Row(b.name, case.name, case.what, i, inp, r, e, g, "departure"))
                elif cg != cr:
                    rows.append(Row(b.name, case.name, case.what, i, inp, r, e, g, "precision"))
    for key, per in impl_vals.items():
        vals = set(per.values())
        if len(vals) > 1:
            for row in rows:
                if (row.case, row.index, row.inputs) == key and row.verdict == "impl_defined":
                    row.verdict = "impl_disagree"
    return ProbeReport(dtype, [b.name for b in chosen], rows, len(CASES),
                       time.perf_counter() - t0, errors)
