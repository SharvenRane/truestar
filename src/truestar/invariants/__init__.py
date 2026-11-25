"""Invariant sentinels: run time checks of identities the math guarantees (docs/TRAIN_NUMERICS.md).

    from truestar import invariants
    sp = invariants.instrument(program)                  # finds softmax, softmax gradient, gate rows
    run = invariants.Sentinels(sp)                       # or Sentinels(sp, compile_program(sp.program, "cuda"))
    outs = run(*inputs)                                  # raises InvariantError (TS1047..TS1049) on a break

    invariants.softmax_grad_rows(ds, p, dp)              # the same checks on any tensors (a fused kernel's)

Identities checked, each row along the reduced dim:

| kind | identity | code |
|---|---|---|
| ``prob_rows`` | a softmax row sums to 1 | TS1047 |
| ``softmax_grad_rows`` | ``ds = p (dp - delta)`` with ``delta = sum(p dp)`` sums to ``delta (1 - sum p)``, which is 0 up to the rounding of p | TS1048 |
| ``normalized_rows`` | ``w = x / sum(x)`` (router gates, renormalized top k) sums to 1 | TS1049 |

Tolerances are derived per row from the dtypes and the row's own magnitudes, never guessed:
``u_c`` is the unit of the arithmetic (float32 for 16 bit and 8 bit storage, the dtype otherwise),
``u_s`` the unit of the stored dtype, ``n`` the row length, and the check itself sums in float64.

* probabilities: ``|sum p - 1| <= 2 ((n + 3) u_c + 2 u_s)``: the shared denominator carries the
  n term to every entry, each entry adds its exp, its division and its store;
* softmax gradient: ``|sum ds - delta (1 - sum p)| <= 2 (((n + 1) u_c + u_s) sum|dp p| (|1 - sum p| +
  sum p) + (2 u_c + 2 u_s) sum|ds|)``: the error of delta reaches every entry through its p, and each
  entry adds a subtraction, a product and a store;
* normalized rows: ``|sum w - 1| <= 2 ((n + 3) u_c + 2 u_s) sum|x| / |sum x|``.

The failure these catch (arXiv 2609.37852, FP8 attention): a backward whose delta comes from other
probabilities than the ones it multiplies (a stale delta, a forward and backward that quantize P
differently) leaves ``sum ds`` off by about ``u_fp8 sum|dp p|``; float32 tolerances are about
``n 6e-8`` of that, so it fires on the first step it appears. A consistent FP8 path passes.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..codes import Issue
from ..ir.core import Node, Program, Value

F64 = torch.float64
SAFETY = 2.0

_UNIT = {torch.float64: 2.0 ** -53, torch.float32: 2.0 ** -24, torch.float16: 2.0 ** -11,
         torch.bfloat16: 2.0 ** -8, torch.float8_e4m3fn: 2.0 ** -4, torch.float8_e5m2: 2.0 ** -3}

CODES = {"prob_rows": "TS1047", "softmax_grad_rows": "TS1048", "normalized_rows": "TS1049"}


def unit(dtype: torch.dtype) -> float:
    return _UNIT.get(dtype, 2.0 ** -24)


def compute_unit(dtype: torch.dtype) -> float:
    """Arithmetic unit for values stored in ``dtype``: float32 registers below 32 bits (every
    Truestar kernel and the vendor libraries widen 16 and 8 bit values), the dtype otherwise."""
    return unit(torch.float64) if dtype == torch.float64 else 2.0 ** -24


class InvariantError(RuntimeError):
    """A math identity failed by more than its dtype derived tolerance."""

    def __init__(self, results: List["InvariantResult"]):
        self.results = [r for r in results if not r.ok]
        first = self.results[0]
        self.code = first.code
        super().__init__("\n".join(r.message() for r in self.results))

    def issues(self) -> List[Issue]:
        return [Issue(r.code, r.message()) for r in self.results]

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "results": [r.as_dict() for r in self.results]}


@dataclasses.dataclass
class InvariantResult:
    kind: str
    name: str                    # the checked value (an IR value name or the caller's label)
    rows: int
    checked: int                 # rows with finite entries (rows holding NaN or inf are the nan sentinel's)
    worst_residual: float
    worst_tolerance: float       # the tolerance of the row with the largest residual / tolerance
    worst_ratio: float           # max over rows of residual / tolerance
    first_bad_row: Optional[Tuple[int, ...]]
    bad_rows: int
    source: str = ""

    @property
    def ok(self) -> bool:
        return self.bad_rows == 0

    @property
    def code(self) -> str:
        return CODES[self.kind]

    def as_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d.update(ok=self.ok, code=self.code)
        return d

    def message(self) -> str:
        what = {"prob_rows": "probability rows do not sum to 1",
                "softmax_grad_rows": "softmax gradient rows do not sum to zero",
                "normalized_rows": "normalized rows (gates) do not sum to 1"}[self.kind]
        if self.ok:
            return (f"truestar {self.code} {self.kind} ok: {self.name}, {self.checked} rows, worst residual "
                    f"{self.worst_residual:.3g} at {self.worst_ratio:.2g} of its tolerance")
        at = f" at {self.source}" if self.source else ""
        return (f"truestar {self.code} invariant: {what}: {self.name}{at}: {self.bad_rows} of {self.checked} rows, "
                f"first row {self.first_bad_row}, worst residual {self.worst_residual:.3g} against a tolerance "
                f"of {self.worst_tolerance:.3g} ({self.worst_ratio:.3g} times it)")


def _rows(kind: str, name: str, residual: torch.Tensor, tol: torch.Tensor, finite: torch.Tensor,
          source: str = "") -> InvariantResult:
    residual = residual.to(F64).abs()
    tol = tol.to(F64)
    ratio = torch.where(finite, residual / torch.where(tol > 0, tol, torch.full_like(tol, 1e-300)),
                        torch.zeros_like(residual))
    bad = finite & (residual > tol)
    rows = residual.numel()
    if rows == 0:
        return InvariantResult(kind, name, 0, 0, 0.0, 0.0, 0.0, None, 0, source)
    flat = int(ratio.reshape(-1).argmax())
    idx = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat), ratio.shape)) if ratio.dim() else ()
    first = None
    if bool(bad.any()):
        f = int(bad.reshape(-1).nonzero()[0])
        first = tuple(int(i) for i in torch.unravel_index(torch.tensor(f), bad.shape)) if bad.dim() else ()
    return InvariantResult(kind, name, rows, int(finite.sum()), float(residual.reshape(-1)[flat]),
                           float(tol.reshape(-1)[flat]), float(ratio.reshape(-1)[flat]), first, int(bad.sum()),
                           source)


# ----------------------------------------------------------------------------- checks on tensors


def prob_tolerance(n: int, store: torch.dtype, compute: Optional[float] = None, check: float = 0.0) -> float:
    uc = compute if compute is not None else compute_unit(store)
    return SAFETY * ((n + 3) * uc + 2 * unit(store) + n * check)


def prob_rows(p: torch.Tensor, dim: int = -1, *, name: str = "p", compute_unit_: Optional[float] = None,
              raise_on_fail: bool = False) -> InvariantResult:
    """Rows of probabilities (a softmax output) sum to 1 within the dtype's tolerance."""
    n = p.shape[dim]
    s = p.detach().to(F64).sum(dim)
    finite = torch.isfinite(p.detach()).all(dim)
    tol = torch.full_like(s, prob_tolerance(n, p.dtype, compute_unit_))
    r = _rows("prob_rows", name, s - 1.0, tol, finite)
    if raise_on_fail and not r.ok:
        raise InvariantError([r])
    return r


def softmax_grad_rows(ds: torch.Tensor, p: torch.Tensor, dp: torch.Tensor, dim: int = -1, *, name: str = "ds",
                      delta: Optional[torch.Tensor] = None, compute_unit_: Optional[float] = None,
                      raise_on_fail: bool = False) -> InvariantResult:
    """``ds`` is the gradient of a softmax's input: ``p (dp - delta)``. Its rows sum to
    ``delta (1 - sum p)``. ``p`` and ``dp`` are the probabilities and their incoming gradient as the
    kernel read them; ``delta`` (keepdim or not) is the kernel's own row value when it exposes one
    (FlashAttention style kernels store it), else ``sum(p dp)`` in float64 is used."""
    dsd, pd, gd = ds.detach().to(F64), p.detach().to(F64), dp.detach().to(F64)
    n = ds.shape[dim]
    st = ds.dtype if unit(ds.dtype) >= unit(p.dtype) else p.dtype
    uc = compute_unit_ if compute_unit_ is not None else compute_unit(st)
    us = unit(st)
    if delta is None:
        dl = (gd * pd).sum(dim)
    else:
        dl = delta.detach().to(F64)
        if dl.dim() == ds.dim():
            dl = dl.squeeze(dim)
    sp = pd.sum(dim)
    gy = (gd * pd).abs().sum(dim)
    sabs = dsd.abs().sum(dim)
    resid = dsd.sum(dim) - dl * (1.0 - sp)
    tol = SAFETY * (((n + 1) * uc + us) * gy * ((1.0 - sp).abs() + sp) + (2 * uc + 2 * us) * sabs)
    finite = torch.isfinite(dsd).all(dim) & torch.isfinite(pd).all(dim) & torch.isfinite(gd).all(dim)
    r = _rows("softmax_grad_rows", name, resid, tol, finite)
    if raise_on_fail and not r.ok:
        raise InvariantError([r])
    return r


def normalized_rows(w: torch.Tensor, x: Optional[torch.Tensor] = None, dim: int = -1, *, name: str = "w",
                    compute_unit_: Optional[float] = None, raise_on_fail: bool = False) -> InvariantResult:
    """``w = x / sum(x)`` along ``dim`` (router gates after a top k renormalization) sums to 1.
    Without ``x`` the entries are taken as non negative (sum|x| / |sum x| = 1)."""
    n = w.shape[dim]
    s = w.detach().to(F64).sum(dim)
    if x is not None:
        xd = x.detach().to(F64)
        cond = xd.abs().sum(dim) / xd.sum(dim).abs().clamp_min(1e-300)
    else:
        cond = w.detach().to(F64).abs().sum(dim)
    tol = prob_tolerance(n, w.dtype, compute_unit_) * cond
    finite = torch.isfinite(w.detach()).all(dim)
    r = _rows("normalized_rows", name, s - 1.0, tol, finite)
    if raise_on_fail and not r.ok:
        raise InvariantError([r])
    return r


# ----------------------------------------------------------------------------- finding sites in a program


@dataclasses.dataclass
class Site:
    kind: str
    value: str                  # the checked value
    dim: int
    n: int                      # row length (static)
    dtype: torch.dtype
    source: str
    operands: Dict[str, str]    # role: value name (p, dp, delta, x)
    outputs: Dict[str, int] = dataclasses.field(default_factory=dict)   # role: index in program outputs


def _static_len(t, dim) -> Optional[int]:
    d = t.shape[dim]
    try:
        return int(d)
    except (TypeError, ValueError):
        return None


def find(program: Program) -> List[Site]:
    """Every softmax output, softmax gradient and x / sum(x) row normalization in ``program``."""
    producer: Dict[int, Node] = {}
    for node in program.nodes:
        for o in node.outputs:
            producer[id(o)] = node
    softmax = {id(n.output): n for n in program.nodes if n.op == "softmax"}
    sites: List[Site] = []

    def through(v):
        """Skip expands and reshapes that only broadcast a keepdim row value."""
        while v is not None and id(v) in producer and producer[id(v)].op in ("expand",):
            v = producer[id(v)].inputs[0]
        return v

    for node in program.nodes:
        if node.op == "softmax":
            x = node.output
            d = node.attrs["dim"] % max(x.type.rank, 1)
            n = _static_len(x.type, d)
            if n is not None and x.type.dtype.is_floating_point:
                sites.append(Site("prob_rows", x.name, d, n, x.type.dtype, node.source, {"p": x.name}))
        if node.op == "mul" and len(node.inputs) == 2:
            for yi, si in ((0, 1), (1, 0)):
                y, sub = node.inputs[yi], node.inputs[si]
                if y is None or id(y) not in softmax or sub is None or id(sub) not in producer:
                    continue
                sn = producer[id(sub)]
                if sn.op != "sub":
                    continue
                g, inner = sn.inputs[0], through(sn.inputs[1])
                if inner is None or id(inner) not in producer or producer[id(inner)].op != "sum":
                    continue
                sm = producer[id(inner)]
                d = softmax[id(y)].attrs["dim"] % max(y.type.rank, 1)
                dims = sm.attrs.get("dims") or []
                if [int(x) % max(y.type.rank, 1) for x in dims] != [d] or not sm.attrs.get("keepdim"):
                    continue
                prod = sm.inputs[0]
                if prod is None or id(prod) not in producer or producer[id(prod)].op != "mul":
                    continue
                pa = producer[id(prod)].inputs
                # delta = sum(dp * p'), with p' normally the same probabilities; any p' is accepted
                # here, so a delta read from other probabilities (stale, requantized) is checked too
                if id(g) not in (id(pa[0]), id(pa[1])):
                    continue
                n = _static_len(y.type, d)
                if n is None:
                    continue
                sites.append(Site("softmax_grad_rows", node.output.name, d, n, node.output.type.dtype, node.source,
                                  {"p": y.name, "dp": g.name, "delta": inner.name}))
                break
        if node.op == "div" and len(node.inputs) == 2:
            x, den = node.inputs
            den = through(den)
            if x is None or den is None or id(den) not in producer:
                continue
            sm = producer[id(den)]
            if sm.op != "sum" or sm.inputs[0] is not x or not sm.attrs.get("keepdim"):
                continue
            dims = sm.attrs.get("dims") or []
            if len(dims) != 1:
                continue
            d = int(dims[0]) % max(x.type.rank, 1)
            n = _static_len(x.type, d)
            if n is not None:
                sites.append(Site("normalized_rows", node.output.name, d, n, node.output.type.dtype, node.source,
                                  {"x": x.name}))
    return sites


@dataclasses.dataclass
class Instrumented:
    """``program`` with the check values as extra outputs after the original ones."""

    program: Program
    original_outputs: int
    sites: List[Site]


def instrument(program: Program, sites: Optional[Sequence[Site]] = None) -> Instrumented:
    """Append, for every site, the float64 row reductions its check needs as program outputs."""
    from ..train.edit import Editor

    sites = list(find(program) if sites is None else sites)
    e = Editor(program, program.name)
    e.source = "truestar.invariants"
    vals: Dict[str, Value] = {v.name: v for v in program.inputs + program.params}
    for node in program.nodes:
        for o in node.outputs:
            vals[o.name] = o
    outs = list(program.outputs)
    n0 = len(outs)

    def f64(name):
        return e.op("cast", vals[name], dtype=F64, name="inv_f64")

    def rowsum(v, d):
        return e.op("sum", v, dims=[d], keepdim=False, name="inv_rowsum")

    for s in sites:
        d = s.dim
        if s.kind == "prob_rows":
            s.outputs = {"sum": len(outs)}          # NaN or inf in a row shows in its sum
            outs.append(rowsum(f64(s.value), d))
        elif s.kind == "softmax_grad_rows":
            z, p, g = f64(s.value), f64(s.operands["p"]), f64(s.operands["dp"])
            gp = e.op("mul", g, p, name="inv_gp")
            s.outputs = {"sum": len(outs), "abs": len(outs) + 1, "psum": len(outs) + 2, "gp": len(outs) + 3,
                         "gpsum": len(outs) + 4}
            outs.append(rowsum(z, d))
            outs.append(rowsum(e.op("abs", z, name="inv_abs"), d))
            outs.append(rowsum(p, d))
            outs.append(rowsum(e.op("abs", gp, name="inv_abs"), d))
            outs.append(rowsum(gp, d))
        elif s.kind == "normalized_rows":
            x = f64(s.operands["x"])
            s.outputs = {"sum": len(outs), "xabs": len(outs) + 1, "xsum": len(outs) + 2}
            outs.append(rowsum(f64(s.value), d))
            outs.append(rowsum(e.op("abs", x, name="inv_abs"), d))
            outs.append(rowsum(x, d))
    prog = e.build(outs, name=f"{program.name}_invariants")
    return Instrumented(prog, n0, sites)


def evaluate(inst: Instrumented, outputs: Sequence[torch.Tensor], compute_unit_: Optional[float] = None
             ) -> List[InvariantResult]:
    """Check every site from the extra outputs of one call of ``inst.program``."""
    res = []
    for s in inst.sites:
        o = {k: outputs[i].detach().to(F64).cpu() for k, i in s.outputs.items()}
        out_dt = outputs[s.outputs["sum"]].dtype
        chk = unit(out_dt) if out_dt != F64 else 0.0      # a compiled program may sum the check in float32
        uc = compute_unit_ if compute_unit_ is not None else compute_unit(s.dtype)
        us = unit(s.dtype)
        n = s.n
        if s.kind == "prob_rows":
            tol = torch.full_like(o["sum"], prob_tolerance(n, s.dtype, uc, chk))
            res.append(_rows(s.kind, s.value, o["sum"] - 1.0, tol, torch.isfinite(o["sum"]), s.source))
        elif s.kind == "softmax_grad_rows":
            sp, gy, sabs = o["psum"], o["gp"], o["abs"]
            dl = o["gpsum"]
            resid = o["sum"] - dl * (1.0 - sp)
            tol = SAFETY * (((n + 1) * uc + us + n * chk) * gy * ((1.0 - sp).abs() + sp)
                            + (2 * uc + 2 * us + n * chk) * sabs)
            fin = torch.isfinite(o["sum"]) & torch.isfinite(sabs) & torch.isfinite(gy)
            res.append(_rows(s.kind, s.value, resid, tol, fin, s.source))
        else:
            cond = o["xabs"] / o["xsum"].abs().clamp_min(1e-300)
            tol = prob_tolerance(n, s.dtype, uc, chk) * cond
            res.append(_rows(s.kind, s.value, o["sum"] - 1.0, tol, torch.isfinite(o["sum"]), s.source))
    return res


class Sentinels:
    """Run an instrumented program and check its invariants on every call.

    ``runner`` is a callable taking the input list and returning the output list (a
    ``compile_program`` result); None runs the float64 reference interpreter. ``mode`` "raise" (the
    default) raises ``InvariantError`` and returns nothing from a failing call; "record" keeps the
    results in ``self.history`` and returns the outputs."""

    def __init__(self, inst: Instrumented, runner=None, mode: str = "raise", compute_unit_: Optional[float] = None):
        if mode not in ("raise", "record"):
            raise ValueError("mode is 'raise' or 'record'")
        self.inst, self.runner, self.mode, self.uc = inst, runner, mode, compute_unit_
        self.history: List[List[InvariantResult]] = []
        self.last: List[InvariantResult] = []

    def __call__(self, *inputs: torch.Tensor) -> List[torch.Tensor]:
        if self.runner is None:
            from ..ir.reference import run

            outs = run(self.inst.program, *inputs)
        else:
            outs = self.runner(list(inputs))
        self.last = evaluate(self.inst, outs, self.uc)
        self.history.append(self.last)
        if self.mode == "raise" and any(not r.ok for r in self.last):
            raise InvariantError(self.last)
        return list(outs[: self.inst.original_outputs])


__all__ = ["InvariantError", "InvariantResult", "Site", "Instrumented", "Sentinels", "find", "instrument",
           "evaluate", "prob_rows", "softmax_grad_rows", "normalized_rows", "prob_tolerance", "unit",
           "compute_unit"]
