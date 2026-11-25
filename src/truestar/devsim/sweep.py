"""`truestar check --sweep NAME:LO..HI`: find the first size at which a backend diverges.

Size dependent bugs sit at cliffs: f16 integers stop being exact above 2048, OpenVINO's GPU gemm
gave 2048 for a sum of 2049 ones (openvinotoolkit/openvino#22613), sums of squares saturate once
they pass 65504 (openvino#39703). A check at the training size misses all of them.

Method
------
1. A ladder of sizes: LO, HI, and every power of two ``2^k`` and ``2^k + 1`` strictly between them
   (the two sides of every binary cliff). The model is checked at each size on every backend,
   cheaply: no op drilldown, no repro.
2. For each backend, the first ladder size that is not OK and the last OK size before it bound the
   cliff; bisection between them finds the first failing size in ``log2(gap)`` more checks, run
   on that backend only. This assumes one cliff between two ladder points, which the ladder makes
   likely but cannot prove; sizes found failing are always rechecked, never inferred.
3. At the first failing size the full check runs once more on that backend, with module and op
   blame and a repro, so the report names the size, the module and the op.

The model file opts in by taking the size as a keyword: ``get_model(n=...)`` and/or
``get_inputs(n=...)`` for ``--sweep n:512..8192``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..compare import OK

Build = Callable[[int], Tuple[Any, Any, Optional[Dict[str, Any]]]]


def parse_sweep(spec: str) -> Tuple[str, int, int]:
    """``"n:512..8192"`` -> ("n", 512, 8192)."""
    try:
        name, rng = spec.split(":", 1)
        lo, hi = rng.split("..", 1)
        lo_i, hi_i = int(lo), int(hi)
    except ValueError:
        raise ValueError(f"--sweep takes NAME:LO..HI, for example n:512..8192; got {spec!r}") from None
    if not name.isidentifier():
        raise ValueError(f"--sweep size name must be a Python identifier, got {name!r}")
    if not 1 <= lo_i < hi_i:
        raise ValueError(f"--sweep needs 1 <= LO < HI, got {lo_i}..{hi_i}")
    return name, lo_i, hi_i


def ladder(lo: int, hi: int) -> List[int]:
    sizes = {lo, hi}
    k = 1
    while 2 ** k < hi:
        for s in (2 ** k, 2 ** k + 1):
            if lo < s < hi:
                sizes.add(s)
        k += 1
    return sorted(sizes)


@dataclass
class BackendSweep:
    backend: str
    verdict: str = "clean"  # clean, cliff, fails_at_lo, error
    last_ok: Optional[int] = None
    first_bad: Optional[int] = None
    status_at_first_bad: str = OK
    module: Optional[str] = None
    module_type: Optional[str] = None
    op: Optional[str] = None
    own_error: Optional[float] = None
    output_error: Optional[float] = None
    threshold: Optional[float] = None
    repro_path: Optional[str] = None
    codes: List[str] = field(default_factory=list)
    checked: List[Tuple[int, str]] = field(default_factory=list)  # (size, status), in run order
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        fin = lambda x: None if x is None or not math.isfinite(x) else x
        return {"backend": self.backend, "verdict": self.verdict, "last_ok": self.last_ok,
                "first_bad": self.first_bad, "status_at_first_bad": self.status_at_first_bad,
                "module": self.module, "module_type": self.module_type, "op": self.op,
                "own_error": fin(self.own_error), "output_error": fin(self.output_error),
                "threshold": fin(self.threshold), "repro_path": self.repro_path,
                "codes": list(self.codes), "checked": [list(c) for c in self.checked],
                "error": self.error}


@dataclass
class SweepReport:
    model: str
    name: str
    lo: int
    hi: int
    results: List[BackendSweep]
    runs: int
    seconds: float

    @property
    def status(self) -> str:
        return "ok" if all(r.verdict == "clean" for r in self.results) else "divergent"

    def as_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "size_name": self.name, "lo": self.lo, "hi": self.hi,
                "ladder": ladder(self.lo, self.hi), "status": self.status, "checks_run": self.runs,
                "seconds": round(self.seconds, 3), "backends": [r.as_dict() for r in self.results]}

    def to_text(self) -> str:
        lines = [f"truestar check --sweep {self.name}:{self.lo}..{self.hi}: {self.model} "
                 f"({self.runs} checks, {self.seconds:.1f} s)"]
        for r in self.results:
            if r.verdict == "clean":
                lines.append(f"[OK] {r.backend}: clean at every size checked "
                             f"({', '.join(str(s) for s, _ in sorted(r.checked))})")
                continue
            if r.verdict == "error":
                lines.append(f"[ERROR] {r.backend}: {r.error}")
                continue
            where = (f"already at {self.name}={r.first_bad}" if r.verdict == "fails_at_lo" else
                     f"clean up to {self.name}={r.last_ok}; {r.status_at_first_bad.upper()} from "
                     f"{self.name}={r.first_bad}")
            lines.append(f"[{r.status_at_first_bad.upper()}] {r.backend}: {where}  "
                         + " ".join(r.codes))
            if r.module:
                own = f", own error {r.own_error:.1e} vs threshold {r.threshold:.1e}" \
                    if r.own_error is not None and r.threshold is not None else ""
                lines.append(f"  module {r.module} ({r.module_type}){own}")
            if r.op:
                lines.append(f"  op {r.op}")
            if r.output_error is not None:
                lines.append(f"  final output relative error at {self.name}={r.first_bad}: "
                             f"{r.output_error:.1e}")
            if r.repro_path:
                lines.append(f"  repro: python {r.repro_path}")
        return "\n".join(lines)


def sweep(build: Build, name: str, lo: int, hi: int, backends: List[str],
          model_name: str = "model", repro_dir: Optional[str] = None) -> SweepReport:
    """Bisect ``name`` from ``lo`` to ``hi`` to the first size where each backend diverges.
    ``build(size)`` returns (model, inputs, kwargs) at that size."""
    from ..check import check

    t0 = time.perf_counter()
    runs = 0
    status: Dict[Tuple[int, str], str] = {}

    def run(size: int, names: List[str]):
        nonlocal runs
        model, inputs, kwargs = build(size)
        runs += 1
        rep = check(model, inputs, kwargs=kwargs, backends=names, repro_dir=None,
                    model_name=model_name, drill_ops=False)
        for r in rep.results:
            status[(size, r.backend.name)] = r.status
        return rep

    lad = ladder(lo, hi)
    for s in lad:
        run(s, backends)
    names = [r for r in backends]
    results: List[BackendSweep] = []
    for b in names:
        key = b.strip().lower()
        res = BackendSweep(backend=key)
        seq = [(s, status.get((s, key), OK)) for s in lad]
        bad = [i for i, (_, st) in enumerate(seq) if st != OK]
        if not bad:
            res.checked = seq
            results.append(res)
            continue
        i = bad[0]
        if i == 0:
            res.verdict, res.first_bad = "fails_at_lo", lad[0]
        else:
            good, worse = lad[i - 1], lad[i]
            while worse - good > 1:
                mid = (good + worse) // 2
                run(mid, [key])
                if status[(mid, key)] == OK:
                    good = mid
                else:
                    worse = mid
            res.verdict, res.last_ok, res.first_bad = "cliff", good, worse
        # Full check at the first failing size: module, op and repro.
        model, inputs, kwargs = build(res.first_bad)
        runs += 1
        full = check(model, inputs, kwargs=kwargs, backends=[key], repro_dir=repro_dir,
                     model_name=f"{model_name}_{name}{res.first_bad}")
        r = full.results[0]
        res.status_at_first_bad = r.status
        res.codes = ["TS521" if res.verdict == "cliff" else "TS522"] + list(r.codes)
        fd = r.first_divergent
        if fd is not None:
            res.module, res.module_type, res.threshold = fd.key, fd.type_name, fd.threshold
            res.own_error = fd.isolated.rel_err if fd.isolated is not None else None
        if r.first_divergent_op is not None:
            res.op = r.first_divergent_op.op
        res.output_error = r.output.rel_err if r.output is not None else None
        res.repro_path = r.repro_path
        res.checked = sorted({(s, st) for (s, bk), st in status.items() if bk == key})
        if r.status == OK:  # the cheap run said otherwise: report it, never infer
            res.error = "the full check at this size was OK although the quick check was not"
        results.append(res)
    return SweepReport(model_name, name, lo, hi, results, runs, time.perf_counter() - t0)
