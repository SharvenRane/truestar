"""Observation in float64 and emulation of a low precision target, both over the tensor IR.

``observe`` runs the float64 reference and records, for every value and for every hidden
intermediate of a composite op, the largest finite magnitude and how many nonzero values fall below
the format's normal and subnormal range. It also replays every node alone with its inputs rounded to
the target format (isolated replay, as `truestar check` does for modules) and records how many
correct bits that node's own output keeps.

``emulate`` runs the program as the target would: every float value (inputs, weights, constants,
op outputs and, for a decomposed target, every hidden intermediate) is rounded to the format, each op
is computed exactly from the rounded operands (float32 accumulation is modelled as exact), and
per node policies keep chosen nodes in float32 or clamp their outputs. The same walk in ``twin``
mode rounds significands only, with float64's exponent range, so twin and target differ exactly by
the range of the format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..ir.core import Node, Program, eval_dim
from ..ir.reference import BINDING_OPS, IMPLS, bind_symbols
from .composite import COMPOSITES, identity, no_record
from .formats import Target


def _is_float(t: torch.Tensor) -> bool:
    return t.dtype.is_floating_point


@dataclass
class Stat:
    """Range statistics of one value (or one hidden intermediate) over every calibration input."""

    peak: float = 0.0           # largest finite |x|
    nonfinite: int = 0          # NaN or infinity already in float64
    count: int = 0
    nonzero: int = 0
    below_normal: int = 0       # 0 < |x| < smallest normal of the format
    flushed: int = 0            # 0 < |x| < half the smallest subnormal: becomes 0
    min_nonzero: float = math.inf

    def add(self, t: torch.Tensor, fmt) -> None:
        t = t.detach()
        if t.numel() == 0:
            return
        a = t.abs()
        fin = torch.isfinite(a)
        self.nonfinite += int((~fin).sum())
        a = torch.where(fin, a, torch.zeros_like(a))
        self.peak = max(self.peak, float(a.max()))
        self.count += a.numel()
        nz = a > 0
        n_nz = int(nz.sum())
        self.nonzero += n_nz
        if n_nz:
            self.min_nonzero = min(self.min_nonzero, float(a[nz].min()))
            self.below_normal += int((nz & (a < fmt.min_normal)).sum())
            self.flushed += int((nz & (a < fmt.min_subnormal / 2)).sum())


@dataclass
class NodeObs:
    out: Stat = field(default_factory=Stat)
    hidden: Dict[str, Stat] = field(default_factory=dict)
    local_err: float = 0.0     # worst relative L2 error of the node replayed alone with rounded inputs
    local_ok: bool = True      # False if the isolated replay produced a non finite value


def _round_fn(target: Target, mode: str) -> Callable[[torch.Tensor], torch.Tensor]:
    fmt = target.fmt
    if mode == "twin":
        return fmt.round_mantissa
    if target.nan_to_inf:
        def r(x):
            y = fmt.round(x)
            return torch.where(torch.isnan(y), torch.full_like(y, math.inf), y)
        return r
    return fmt.round


_CONSTANT_OPS = ("full", "full_like")


def _saturate(x: torch.Tensor, fmt) -> torch.Tensor:
    """Constants (weights, scalars, filled tensors) beyond the format's range saturate to its largest
    finite value, as float16 converters do (onnxconverter-common clips initializers); computed values
    overflow instead. Without this the -3.4e38 attention mask constant of a float32 export would
    become -inf and 0 * -inf = NaN, a failure no real fp16 graph has."""
    if fmt is None:
        return x
    return x.clamp(-fmt.max, fmt.max)


def _fp32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).to(torch.float64)


def _call(node: Node, args, shape, bindings):
    fn = IMPLS[node.op]
    if node.op in BINDING_OPS:
        return fn(node.attrs, args, shape, bindings=bindings)
    return fn(node.attrs, args, shape)


def _shapes(node: Node, bindings):
    shapes = [tuple(eval_dim(d, bindings) for d in o.type.shape) for o in node.outputs]
    return shapes[0] if len(shapes) == 1 else shapes


def _float_out(node: Node) -> bool:
    return node.output.type.dtype.is_floating_point


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    nb = float(torch.linalg.vector_norm(b))
    if nb == 0:
        return 0.0 if float(torch.linalg.vector_norm(a)) == 0 else math.inf
    return float(torch.linalg.vector_norm(a - b)) / nb


def _setup(prog: Program, inputs):
    bindings = bind_symbols(prog, inputs)
    env: Dict[int, torch.Tensor] = {}
    for v, t in zip(prog.inputs, inputs):
        env[id(v)] = t.detach().to("cpu", torch.float64 if t.dtype.is_floating_point else t.dtype)
    for p in prog.params:
        d = prog.param_data[p.name]
        env[id(p)] = d.detach().to("cpu", torch.float64 if d.dtype.is_floating_point else d.dtype)
    return bindings, env


def observe(prog: Program, calibration: Sequence[Sequence[torch.Tensor]], target: Target,
            local: bool = True) -> Tuple[Dict[int, NodeObs], Dict[int, Stat], List[List[torch.Tensor]]]:
    """float64 run over every calibration input. Returns per node observations, per input or param
    value statistics, and the float64 outputs of each run."""
    obs: Dict[int, NodeObs] = {id(n): NodeObs() for n in prog.nodes}
    leaf: Dict[int, Stat] = {}
    fmt = target.fmt
    r = _round_fn(target, "target")
    outs_all = []
    for inputs in calibration:
        bindings, env = _setup(prog, inputs)
        for v in list(prog.inputs) + list(prog.params):
            if _is_float(env[id(v)]):
                leaf.setdefault(id(v), Stat()).add(env[id(v)], fmt)
        for node in prog.nodes:
            o = obs[id(node)]
            args = [None if x is None else env[id(x)] for x in node.inputs]
            shape = _shapes(node, bindings)
            comp = COMPOSITES.get(node.op)
            if comp is not None:
                def rec(name, t, o=o):
                    o.hidden.setdefault(name, Stat()).add(t, fmt)
                res = comp(node.attrs, args, identity, rec)
            else:
                res = _call(node, args, shape, bindings)
            results = list(res) if isinstance(res, (list, tuple)) else [res]
            for out, val in zip(node.outputs, results):
                if val.dtype != torch.float64 and val.dtype.is_floating_point:
                    val = val.to(torch.float64)
                env[id(out)] = val
            first = results[0]
            if _is_float(first):
                o.out.add(first, fmt)
                if local and node.op not in ("cond", "loop"):
                    rargs = [None if a is None else (r(a) if _is_float(a) else a) for a in args]
                    try:
                        if comp is not None and target.decomposed:
                            lr = comp(node.attrs, rargs, r, no_record)
                        else:
                            lr = _call(node, rargs, shape, bindings)
                        lr = list(lr)[0] if isinstance(lr, (list, tuple)) else lr
                        lr = r(lr.to(torch.float64))
                        fin = torch.isfinite(first)
                        if not bool(torch.isfinite(lr)[fin].all()):
                            o.local_ok = False
                        else:
                            o.local_err = max(o.local_err, _rel(lr[fin], first[fin]))
                    except Exception:
                        pass
        outs_all.append([env[id(v)] for v in prog.outputs])
    return obs, leaf, outs_all


@dataclass
class EmuResult:
    outputs: List[torch.Tensor]
    origins: List[Tuple[Node, int]]  # nodes whose output turned non finite from finite inputs, count


def emulate(prog: Program, inputs: Sequence[torch.Tensor], target: Target, mode: str = "target",
            fp32_nodes: Optional[set] = None, clamps: Optional[Dict[int, float]] = None,
            keep_env: bool = False) -> EmuResult:
    """Run ``prog`` as the target would. ``mode`` is "target" or "twin" (significand only).

    ``fp32_nodes``: ids of nodes computed and stored in float32 (their operands are read in float32,
    their composite internals are fused). ``clamps``: node id to C, the output clamped to [-C, C]
    before rounding."""
    fp32_nodes = fp32_nodes or set()
    clamps = clamps or {}
    r_t = _round_fn(target, mode)
    bindings, env = _setup(prog, inputs)
    cache: Dict[Tuple[int, str], torch.Tensor] = {}
    origins: List[Tuple[Node, int]] = []
    for v in prog.inputs:
        if _is_float(env[id(v)]):
            env[id(v)] = r_t(env[id(v)])
    param_ids = {id(p) for p in prog.params}
    for node in prog.nodes:
        wide = id(node) in fp32_nodes
        r = _fp32 if wide else r_t
        key = "w" if wide else "t"
        args = []
        for x in node.inputs:
            if x is None:
                args.append(None)
                continue
            t = env[id(x)]
            if _is_float(t):
                if id(x) in param_ids:
                    ck = (id(x), key)
                    if ck not in cache:
                        cache[ck] = r(_saturate(t, None if wide else target.fmt))
                    t = cache[ck]
                else:
                    t = r(t)
            args.append(t)
        shape = _shapes(node, bindings)
        comp = COMPOSITES.get(node.op)
        if comp is not None and target.decomposed and not wide:
            res = comp(node.attrs, args, r, no_record)
        else:
            res = _call(node, args, shape, bindings)
        results = list(res) if isinstance(res, (list, tuple)) else [res]
        fin_in = all(a is None or not _is_float(a) or bool(torch.isfinite(a).all()) for a in args)
        for i, (out, val) in enumerate(zip(node.outputs, results)):
            if _is_float(val):
                val = val.to(torch.float64)
                if i == 0 and node.op in _CONSTANT_OPS and not wide:
                    val = _saturate(val, target.fmt)
                if i == 0 and id(node) in clamps:
                    c = clamps[id(node)]
                    val = val.clamp(-c, c)
                val = r(val)
                if i == 0 and fin_in:
                    bad = int((~torch.isfinite(val)).sum())
                    if bad:
                        origins.append((node, bad))
            env[id(out)] = val
    res = EmuResult([env[id(v)] for v in prog.outputs], origins)
    if keep_env:
        res.env = env
    return res


# ----------------------------------------------------------------------------- comparison


@dataclass
class Compare:
    rel_err: float          # relative L2 error of every float output against float64
    cosine_min: float       # worst per row cosine similarity of the first float output
    cosine_mean: float
    nonfinite: int          # non finite output values where float64 is finite
    flips: int              # per row argmax changes of the first float output
    real_flips: int         # flips whose float64 margin is larger than the format's own noise
    rows: int

    def as_dict(self) -> Dict[str, float]:
        return {"rel_err": self.rel_err, "cosine_min": self.cosine_min, "cosine_mean": self.cosine_mean,
                "nonfinite": self.nonfinite, "flips": self.flips, "real_flips": self.real_flips,
                "rows": self.rows}


def _rows(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 0:
        return t.reshape(1, 1)
    if t.dim() == 1:
        return t.reshape(1, -1)
    return t.reshape(-1, t.shape[-1])


def compare(refs: Sequence[Sequence[torch.Tensor]], runs: Sequence[Sequence[torch.Tensor]],
            noise: Optional[Sequence[Sequence[torch.Tensor]]] = None) -> Compare:
    """Compare runs with the float64 outputs, over every calibration input.

    ``noise`` (the twin run) sets what counts as a near tie: a flip is real only when the float64
    gap between the two classes is larger than twice the twin's error on that row."""
    num = den = 0.0
    cos: List[float] = []
    nonfinite = flips = real = rows = 0
    for i, (ref, run) in enumerate(zip(refs, runs)):
        firsts = [(a, b) for a, b in zip(ref, run) if a.dtype.is_floating_point]
        for a, b in firsts:
            fin = torch.isfinite(a)
            bad = fin & ~torch.isfinite(b)
            nonfinite += int(bad.sum())
            bb = torch.where(torch.isfinite(b), b, torch.zeros_like(b))
            num += float(((bb - a)[fin] ** 2).sum()) + (math.inf if int(bad.sum()) else 0.0)
            den += float((a[fin] ** 2).sum())
        if not firsts:
            continue
        a, b = firsts[0]
        ra, rb = _rows(a), _rows(b)
        rb_f = torch.where(torch.isfinite(rb), rb, torch.zeros_like(rb))
        na = torch.linalg.vector_norm(ra, dim=1)
        nb = torch.linalg.vector_norm(rb_f, dim=1)
        c = (ra * rb_f).sum(1) / (na * nb).clamp_min(1e-300)
        c = torch.where(torch.isfinite(rb).all(1), c, torch.full_like(c, -1.0))
        cos.extend(float(x) for x in c)
        rows += ra.shape[0]
        if ra.shape[1] > 1:
            ia = ra.argmax(1)
            ib = torch.where(torch.isfinite(rb), rb, torch.full_like(rb, -math.inf)).argmax(1)
            ib = torch.where(torch.isfinite(rb).any(1), ib, torch.full_like(ib, -1))
            diff = ia != ib
            flips += int(diff.sum())
            if noise is not None:
                tw = _rows(noise[i][[j for j, t in enumerate(ref) if t.dtype.is_floating_point][0]])
                tol = 2 * (tw - ra).abs().amax(1)
            else:
                tol = torch.zeros(ra.shape[0], dtype=ra.dtype)
            for j in torch.nonzero(diff).flatten().tolist():
                if int(ib[j]) < 0:
                    real += 1
                    continue
                gap = float(ra[j, ia[j]] - ra[j, ib[j]])
                if gap > float(tol[j]):
                    real += 1
    rel = math.sqrt(num / den) if den > 0 else (0.0 if num == 0 else math.inf)
    return Compare(rel, min(cos) if cos else 1.0, sum(cos) / len(cos) if cos else 1.0,
                   nonfinite, flips, real, rows)
