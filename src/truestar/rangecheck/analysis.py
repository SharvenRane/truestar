"""`truestar range`: predict, before deployment, where a model leaves a low precision format's range,
propose a remedy and verify it.

Pipeline (docs/RANGE.md):

1. Import the model to the tensor IR once (torch.export).
2. ``observe``: one float64 run per calibration input records the peak of every value and of every
   hidden intermediate (x minus mean squared, mean of squares, attention scores), how much of each
   tensor falls below the format's normal range, and how many correct bits each op keeps when
   replayed alone with inputs rounded to the format.
3. ``propagate``: static interval bounds over the input box; a value whose bound fits with headroom
   is proven safe for every input in the box.
4. Emulate the target and its range unlimited twin (same significand bits, float64's exponents). The
   target is judged against the twin, so ordinary rounding is never called a range problem.
5. If the target fails, build remedies (exact power of two rescale, float32 regions, clamps), run each
   in emulation and keep the first that passes the same test.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..ir.core import Node, Program
from ..ir.reference import bind_symbols
from .composite import COMPOSITES, HARMFUL_HIDDEN
from .engine import Compare, NodeObs, compare, emulate, observe
from .formats import Target, get_target
from .interval import concrete_shapes, input_box, mag, propagate
from .remedy import RegionError, apply_rescale, find_region, fp32_region

# a hidden intermediate whose value is a square of the input: a rescale by s shrinks it by s^2
_SQUARED = {"(x - mean)^2", "variance", "x^2", "mean(x^2)"}
_PRECISION_BITS = 2.0     # an op keeping fewer correct bits than this, alone, loses all precision
_UNDERFLOW_SHARE = 0.5    # share of nonzero values below the smallest normal that makes a site
_ERR_FACTOR = 2.0         # a run passes when its error is within this factor of the twin's


@dataclass
class Where:
    aten: str = ""
    file: str = ""
    line: Optional[int] = None
    code: str = ""
    module: str = ""
    module_type: str = ""

    def short(self) -> str:
        loc = f"{self.file}:{self.line}" if self.file else ""
        mod = f"{self.module} ({self.module_type})" if self.module else "<model>"
        return f"{mod}, {self.aten or '?'}" + (f", {loc}" if loc else "")


_SRC = re.compile(r"^(?P<aten>\S+)(?: \(node %\S+\))?(?:, (?P<file>[^:,]+):(?P<line>\d+) in \w+: (?P<code>.*?))?"
                  r"(?: \(module (?P<module>[^:]+): (?P<type>[^)]+)\))?$")


def parse_source(src: str) -> Where:
    m = _SRC.match(src.strip()) if src else None
    if not m:
        return Where(code=src)
    return Where(m.group("aten") or "", m.group("file") or "", int(m.group("line")) if m.group("line") else None,
                 (m.group("code") or "").strip(), m.group("module") or "", m.group("type") or "")


@dataclass
class Site:
    """One place where the target leaves its format's range or precision."""

    kind: str               # overflow, hidden_overflow, near_overflow, underflow, precision_loss,
    #                         constant_saturated, hidden_in_fused
    node: Optional[Node]
    value: str              # IR value name, plus the hidden intermediate's name when hidden
    op: str
    where: Where
    peak: float             # observed float64 peak magnitude
    headroom_bits: float    # log2(max / peak); negative overflows by that many bits
    static_peak: float      # static bound (inf when no rule proves anything)
    static_headroom_bits: float
    share_below_normal: float = 0.0
    precision_bits: Optional[float] = None
    hidden: str = ""
    effect: str = "untested"  # harmful, benign, untested
    origin: bool = True     # every float input is inside the range: this op is where the value leaves it
    remedy: str = ""

    @property
    def proven(self) -> bool:
        return math.isfinite(self.static_headroom_bits) and self.static_headroom_bits >= 0

    def as_dict(self) -> Dict[str, Any]:
        w = self.where
        return {
            "kind": self.kind, "value": self.value, "op": self.op, "hidden": self.hidden or None,
            "aten": w.aten or None, "module": w.module or None, "module_type": w.module_type or None,
            "file": w.file or None, "line": w.line, "code": w.code or None,
            "peak": self.peak, "headroom_bits": self.headroom_bits,
            "static_peak": self.static_peak, "static_headroom_bits": self.static_headroom_bits,
            "proven_safe_statically": self.proven,
            "share_below_normal": self.share_below_normal, "precision_bits": self.precision_bits,
            "origin": self.origin, "effect": self.effect, "remedy": self.remedy or None,
        }


@dataclass
class Attempt:
    remedy: str             # rescale, fp32, clamp
    passed: bool
    result: Optional[Compare]
    detail: str
    cost: Dict[str, Any] = field(default_factory=dict)
    exact_rel_err: Optional[float] = None  # rescale only: float64 rewritten vs original

    def as_dict(self) -> Dict[str, Any]:
        return {"remedy": self.remedy, "passed": self.passed,
                "result": None if self.result is None else self.result.as_dict(),
                "detail": self.detail, "cost": self.cost, "exact_rel_err": self.exact_rel_err}


@dataclass
class RangeReport:
    model_name: str
    target: Target
    headroom_bits: float
    status: str                       # ok, at_risk, overflow, remedied
    sites: List[Site]
    info: List[Site]
    twin: Compare
    target_run: Compare
    attempts: List[Attempt]
    chosen: Optional[Attempt]
    counts: Dict[str, int]
    seconds: Dict[str, float]
    calibration: int

    @property
    def codes(self) -> List[str]:
        out: List[str] = []
        for c in [_KIND_CODE.get(s.kind) for s in self.sites] + ["TS309" for _ in self.info[:1]] + self._run_codes():
            if c and c not in out:
                out.append(c)
        return out

    def _run_codes(self) -> List[str]:
        codes = []
        if not _passes(self.target_run, self.twin):
            codes.append("TS306")
            if not any(s.kind in _HARMFUL_KINDS for s in self.sites):
                codes.append("TS308")
        if self.chosen is not None:
            codes.append("TS307")
        elif self.status == "overflow":
            codes.append("TS308")
        if self.status == "ok" and not self.sites:
            codes.append("TS300")
        return codes

    def to_text(self, top: int = 12) -> str:
        from .output import report_text

        return report_text(self, top)

    def to_agent(self, model_file: Optional[str] = None) -> Dict[str, Any]:
        from .output import report_agent

        return report_agent(self, model_file)


_KIND_CODE = {"overflow": "TS301", "hidden_overflow": "TS302", "near_overflow": "TS303",
              "underflow": "TS304", "precision_loss": "TS305"}
_HARMFUL_KINDS = ("overflow", "hidden_overflow", "underflow", "precision_loss")


def _bits(fmt, peak: float) -> float:
    return fmt.headroom_bits(peak)


def _passes(run: Compare, twin: Compare) -> bool:
    """A run is as good as the format allows: no new non finite values, no decision flips beyond
    the twin's, and an output error within _ERR_FACTOR of the twin's."""
    if run.nonfinite > twin.nonfinite:
        return False
    if run.real_flips > twin.real_flips:
        return False
    if not math.isfinite(run.rel_err):
        return False
    return run.rel_err <= _ERR_FACTOR * twin.rel_err + 1e-12


def _as_calibration(inputs) -> List[Tuple[torch.Tensor, ...]]:
    if isinstance(inputs, torch.Tensor):
        return [(inputs,)]
    if isinstance(inputs, tuple) and all(isinstance(t, torch.Tensor) for t in inputs):
        return [inputs]
    out = []
    for item in inputs:
        out.append((item,) if isinstance(item, torch.Tensor) else tuple(item))
    return out


def range_check(model, inputs, target: str = "fp16", headroom_bits: float = 1.0,
                calibration: Optional[Sequence] = None, remedies: bool = True,
                model_name: Optional[str] = None, program: Optional[Program] = None) -> RangeReport:
    """Run the range analysis. ``inputs`` are the forward arguments (as for `truestar check`);
    ``calibration`` optionally a list of more forward argument tuples of the same shapes."""
    from .. import ir

    tgt = get_target(target)
    fmt = tgt.fmt
    limit = fmt.max * 2.0 ** -headroom_bits
    secs: Dict[str, float] = {}
    t0 = time.perf_counter()
    calib = _as_calibration(inputs) + (_as_calibration(calibration) if calibration is not None else [])
    if program is None:
        with torch.no_grad():
            program = ir.import_module(model, calib[0])
    prog = program
    secs["import"] = time.perf_counter() - t0

    t = time.perf_counter()
    obs, leaf, refs = observe(prog, calib, tgt)
    secs["observe"] = time.perf_counter() - t

    t = time.perf_counter()
    bindings = bind_symbols(prog, calib[0])
    bounds = propagate(prog, input_box(calib), concrete_shapes(prog, bindings))
    secs["static"] = time.perf_counter() - t

    t = time.perf_counter()
    twins = [emulate(prog, c, tgt, "twin").outputs for c in calib]
    target_runs = [emulate(prog, c, tgt, "target") for c in calib]
    twin_cmp = compare(refs, twins, twins)
    tgt_cmp = compare(refs, [r.outputs for r in target_runs], twins)
    secs["emulate"] = time.perf_counter() - t

    sites, info = _find_sites(prog, obs, leaf, bounds, tgt, limit)
    counts = _counts(prog, obs, bounds, fmt, limit)
    attempts: List[Attempt] = []
    chosen: Optional[Attempt] = None
    ok = _passes(tgt_cmp, twin_cmp)
    harmful = [s for s in sites if s.kind in _HARMFUL_KINDS]
    t = time.perf_counter()
    if ok:
        # a range event that changes nothing on the calibration inputs is information, not a finding.
        # A precision loss is not a range event: the twin has it too, so it stays a warning.
        for s in harmful:
            if s.kind != "precision_loss":
                s.effect = "benign"
                sites.remove(s)
                info.append(s)
        status = "at_risk" if any(s.kind in ("near_overflow", "precision_loss") for s in sites) else "ok"
    else:
        status = "overflow"
        for s in harmful:
            s.effect = "harmful"
        if remedies and harmful:
            attempts = _remedies(prog, calib, refs, twins, twin_cmp, tgt, harmful, obs, limit, headroom_bits)
            chosen = next((a for a in attempts if a.passed), None)
            if chosen is not None:
                status = "remedied"
                for s in harmful:
                    s.remedy = chosen.remedy
    secs["remedies"] = time.perf_counter() - t
    secs["total"] = time.perf_counter() - t0
    name = model_name or prog.name
    return RangeReport(name, tgt, headroom_bits, status, sites, info, twin_cmp, tgt_cmp, attempts, chosen,
                       counts, secs, len(calib))


def _find_sites(prog: Program, obs: Dict[int, NodeObs], leaf, bounds, tgt: Target, limit: float):
    fmt = tgt.fmt
    sites: List[Site] = []
    info: List[Site] = []
    peak_of = {vid: st.peak for vid, st in leaf.items()}
    for n in prog.nodes:
        for out in n.outputs:
            peak_of[id(out)] = obs[id(n)].out.peak
    for n in prog.nodes:
        if not n.output.type.dtype.is_floating_point:
            continue
        o = obs[id(n)]
        st = mag(bounds.of(n.output))
        w = parse_source(n.source)
        const = n.op in ("full", "full_like")
        origin = all(x is None or not x.type.dtype.is_floating_point or peak_of.get(id(x), 0.0) <= fmt.max
                     for x in n.inputs)
        base = dict(node=n, value=n.output.name, op=n.op, where=w, static_peak=st,
                    static_headroom_bits=_bits(fmt, st), origin=origin)
        if o.out.peak > fmt.max:
            kind = "constant_saturated" if const else "overflow"
            (info if const else sites).append(Site(kind, peak=o.out.peak, headroom_bits=_bits(fmt, o.out.peak), **base))
        elif o.out.peak > limit and not (math.isfinite(st) and st <= fmt.max) and not const:
            sites.append(Site("near_overflow", peak=o.out.peak, headroom_bits=_bits(fmt, o.out.peak), **base))
        hid_bounds = bounds.hidden.get(id(n), {})
        for hname in HARMFUL_HIDDEN.get(n.op, ()):
            hs = o.hidden.get(hname)
            if hs is None or hs.peak <= fmt.max:
                continue
            hst = mag(hid_bounds.get(hname, (-math.inf, math.inf)))
            hb = dict(base, value=f"{n.output.name}.{hname}", static_peak=hst,
                      static_headroom_bits=_bits(fmt, hst))
            kind = "hidden_overflow" if tgt.decomposed else "hidden_in_fused"
            (sites if tgt.decomposed else info).append(
                Site(kind, peak=hs.peak, headroom_bits=_bits(fmt, hs.peak), hidden=hname, **hb))
        if o.out.nonzero and o.out.peak <= fmt.max and not const:
            share = o.out.below_normal / o.out.nonzero
            if share >= _UNDERFLOW_SHARE:
                sites.append(Site("underflow", peak=o.out.peak, headroom_bits=_bits(fmt, o.out.peak),
                                  share_below_normal=share, **base))
                continue
        explained = bool(sites) and sites[-1].node is n  # an overflow here already explains the loss
        if o.local_ok and o.local_err > 0 and o.out.peak <= fmt.max and not const and not explained:
            bits = -math.log2(o.local_err) if o.local_err > 0 else 53.0
            if bits < _PRECISION_BITS:
                sites.append(Site("precision_loss", peak=o.out.peak, headroom_bits=_bits(fmt, o.out.peak),
                                  precision_bits=bits, **base))
    names = {id(p): p for p in prog.params}
    for vid, stat in leaf.items():
        if vid in names and stat.peak > fmt.max:
            p = names[vid]
            info.append(Site("constant_saturated", None, p.name, "param", Where(), stat.peak,
                             _bits(fmt, stat.peak), stat.peak, _bits(fmt, stat.peak)))
    # causes first: sites whose inputs are in range, then values that stay out of range downstream
    sites.sort(key=lambda s: (not s.origin, s.kind == "near_overflow"))
    return sites, info


def _counts(prog: Program, obs, bounds, fmt, limit: float) -> Dict[str, int]:
    n_float = proven = observed_only = 0
    for n in prog.nodes:
        if not n.output.type.dtype.is_floating_point:
            continue
        n_float += 1
        st = mag(bounds.of(n.output))
        if math.isfinite(st) and st <= limit:
            proven += 1
        elif obs[id(n)].out.peak <= limit:
            observed_only += 1
    return {"float_values": n_float, "proven_safe_statically": proven,
            "safe_on_calibration_only": observed_only}


def _seeds(sites: Sequence[Site], obs, limit: float) -> Tuple[Dict[int, float], List[Site]]:
    """Values to rescale and the factor each needs; sites a rescale cannot help are returned apart."""
    seeds: Dict[int, float] = {}
    rest: List[Site] = []
    for s in sites:
        n = s.node
        if s.kind == "overflow":
            seeds[id(n.output)] = max(seeds.get(id(n.output), 1.0), s.peak / limit)
        elif s.kind == "hidden_overflow" and n.op in ("layer_norm", "rms_norm", "group_norm"):
            f = s.peak / limit
            f = math.sqrt(f) if s.hidden in _SQUARED else f
            x = n.inputs[0]
            if x.kind != "op":
                rest.append(s)
                continue
            seeds[id(x)] = max(seeds.get(id(x), 1.0), f)
        else:
            rest.append(s)
    return seeds, rest


def _run_all(prog, calib, refs, twins, tgt, fp32=None, clamps=None) -> Compare:
    outs = [emulate(prog, c, tgt, "target", fp32_nodes=fp32, clamps=clamps).outputs for c in calib]
    return compare(refs, outs, twins)


def _remedies(prog, calib, refs, twins, twin_cmp, tgt, harmful, obs, limit, headroom_bits) -> List[Attempt]:
    peaks: Dict[int, float] = {}
    for n in prog.nodes:
        for o in n.outputs:
            peaks[id(o)] = obs[id(n)].out.peak
    attempts: List[Attempt] = []

    # 1. exact power of two rescale (plus float32 for sites a rescale cannot reach)
    seeds, rest = _seeds(harmful, obs, limit)
    if seeds:
        try:
            region = find_region(prog, seeds, peaks, limit)
            new = apply_rescale(prog, region)
            exact = compare(refs, [_ref_run(new, c) for c in calib]).rel_err
            fp32 = {id(n) for n in _map_nodes(new, [s.node for s in rest])}
            res = _run_all(new, calib, refs, twins, tgt, fp32=fp32)
            ok = _passes(res, twin_cmp) and exact <= 1e-9
            detail = (f"carry {len(region.values)} values scaled by 2^-{region.k}: {len(region.entries)} "
                      f"weights or edges scaled in, {len(region.norm_exits)} norms take eps * 2^-{2 * region.k}, "
                      f"{len(region.exits)} edges scaled back")
            if rest:
                detail += f"; {len(rest)} other sites kept in float32"
            attempts.append(Attempt("rescale", ok, res, detail,
                                    {"scale_exponent": -region.k, "rescaled_values": len(region.values),
                                     "entries": len(region.entries), "exits": len(region.exits),
                                     "norm_eps_changes": len(region.norm_exits), "fp32_ops": len(fp32)},
                                    exact))
        except RegionError as exc:
            attempts.append(Attempt("rescale", False, None, f"not applicable: {exc}"))

    # 2. keep the sites and every consumer of an out of range value in float32
    over_nodes = [s.node for s in harmful]
    keep = fp32_region(prog, over_nodes, peaks, limit)
    res = _run_all(prog, calib, refs, twins, tgt, fp32=keep)
    attempts.append(Attempt("fp32", _passes(res, twin_cmp), res,
                            f"{len(keep)} ops in float32 of {sum(1 for n in prog.nodes if n.output.type.dtype.is_floating_point)}",
                            {"fp32_ops": len(keep)}))

    # 3. clamp every out of range op output to +-max * 2^-headroom (hidden sites: float32)
    c = tgt.fmt.max * 2.0 ** -headroom_bits
    clamps = {}
    for n in prog.nodes:
        if id(n) in keep and peaks.get(id(n.output), 0.0) > limit and n.op not in ("full", "full_like"):
            clamps[id(n)] = c
    hidden_nodes = {id(s.node) for s in harmful if s.kind != "overflow"}
    if clamps:
        res = _run_all(prog, calib, refs, twins, tgt, fp32=hidden_nodes, clamps=clamps)
        attempts.append(Attempt("clamp", _passes(res, twin_cmp), res,
                                f"{len(clamps)} outputs clipped to +-{c:g}"
                                + (f", {len(hidden_nodes)} ops in float32" if hidden_nodes else ""),
                                {"clamps": len(clamps), "clip": c, "fp32_ops": len(hidden_nodes)}))
    return attempts


def _ref_run(prog: Program, inputs) -> List[torch.Tensor]:
    from ..ir.reference import run

    return run(prog, *inputs)


def _map_nodes(new: Program, old_nodes: Sequence[Node]) -> List[Node]:
    """Nodes of a rewritten program that define the same values as ``old_nodes``."""
    want = {id(n.output) for n in old_nodes if n is not None}
    return [n for n in new.nodes if id(n.output) in want]
