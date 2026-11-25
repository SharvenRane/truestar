"""The near tie router report: tokens whose expert choice is not determined in the working precision.

A router picks a token's top k experts by comparing scores. When the k-th and the (k+1)-th score
differ by less than what rounding in the working precision can move them, the choice is a coin
toss decided by summation order, so two chips (or two batch sizes) may send the token to different
experts, and everything after that layer differs by far more than rounding. This report finds those
tokens before that happens, from a float64 run.

For each covered MoE layer (truestar.moe.coverage) the model runs twice: in float64 on the CPU (the
reference) and in the working precision on its device. A hook reads the router's input in both. Per
token and expert, the bound on the router logit's error in the working precision is

    b_e = (2 u + gamma_H(u_acc)) * sum_i |W_ei| |x_i|  +  sum_i |W_ei| |x_wp,i - x_i|

with x the float64 router input, x_wp the working precision one (the measured error that reached
the router from earlier layers), u the unit roundoff of the working dtype, gamma_H the dot product
bound of H terms in the accumulator. The selection score is the logit for softmax routers (softmax
keeps the order; 8 float32 ulps are added for its own rounding) and sigmoid(logit) + bias for
DeepSeek style routers (|sigmoid'| times b_e plus 4 float32 ulps of the score). A token is a near
tie (TS1182) when, in float64,

    score_(k) - score_(k+1) <= bound_(k) + bound_(k+1)

For group limited routing the same test runs on the group scores (the sum of a group's two best
scores, bounded by the sum of their bounds) at the topk_group boundary.

The report also compares the expert sets the two runs picked: a token whose set differs and is a
near tie is the expected consequence; a differing token that is not a near tie is a real routing
divergence (TS1183). Routers whose rule the report does not know are listed as unmeasured (TS1184),
never skipped silently.
"""

from __future__ import annotations

import copy
import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..kernel import verify as KV
from .coverage import scan

U = {torch.float64: 2.0 ** -53, torch.float32: 2.0 ** -24, torch.float16: 2.0 ** -11, torch.bfloat16: 2.0 ** -8}
U32 = 2.0 ** -24


@dataclass
class Tie:
    layer: str
    token: int            # flattened (batch, position) index, as the router sees it
    position: Tuple[int, ...]
    kth: int              # expert ranked k-th in float64
    next: int             # expert ranked (k+1)-th
    gap: float            # their float64 score difference
    bound: float          # the rounding bound it is compared with
    flipped: bool         # the working precision run picked a different expert set here
    group: bool = False   # a tie between groups (group limited routing), not experts

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__, position=list(self.position))


@dataclass
class Layer:
    path: str
    rule: str             # softmax, sigmoid_bias, or unmeasured
    tokens: int = 0
    k: int = 0
    ties: List[Tie] = field(default_factory=list)
    flips: List[int] = field(default_factory=list)          # tokens whose expert set differs
    real_flips: List[int] = field(default_factory=list)     # of those, not near ties
    min_ratio: float = math.inf     # smallest gap / bound over tokens (below 1 is a near tie)
    note: str = ""


@dataclass
class TieReport:
    dtype: torch.dtype
    device: str
    layers: List[Layer]

    @property
    def ties(self) -> List[Tie]:
        return [t for layer in self.layers for t in layer.ties]

    @property
    def codes(self) -> List[str]:
        out = []
        if self.ties:
            out.append("TS1182")
        if any(layer.real_flips for layer in self.layers):
            out.append("TS1183")
        if any(layer.rule == "unmeasured" for layer in self.layers):
            out.append("TS1184")
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {"dtype": str(self.dtype).replace("torch.", ""), "device": self.device, "codes": self.codes,
                "layers": [dict(path=layer.path, rule=layer.rule, tokens=layer.tokens, k=layer.k,
                                near_ties=len(layer.ties), flips=len(layer.flips), real_flips=layer.real_flips,
                                min_gap_over_bound=layer.min_ratio, note=layer.note,
                                ties=[t.as_dict() for t in layer.ties]) for layer in self.layers]}

    def describe(self) -> str:
        name = str(self.dtype).replace("torch.", "")
        lines = [f"router near ties ({name} on {self.device}, bound from float64)"]
        for layer in self.layers:
            if layer.rule == "unmeasured":
                lines.append(f"  {layer.path}: TS1184 unmeasured ({layer.note})")
                continue
            lines.append(f"  {layer.path}: {layer.tokens} tokens, top {layer.k}, {len(layer.ties)} near ties, "
                         f"{len(layer.flips)} expert set changes ({len(layer.real_flips)} not near ties), "
                         f"smallest gap/bound {layer.min_ratio:.3g}")
            for t in layer.ties:
                what = "groups" if t.group else "experts"
                lines.append(f"    TS1182 token {t.token} {tuple(t.position)}: {what} {t.kth} and {t.next} differ by "
                             f"{t.gap:.3e} in float64, bound {t.bound:.3e}{' (flipped)' if t.flipped else ''}")
            for tok in layer.real_flips:
                lines.append(f"    TS1183 token {tok}: expert set changed although the choice is not a near tie")
        return "\n".join(lines)


def _gamma(n: int, u: float) -> float:
    return n * u / (1 - n * u)


def _router_of(block: torch.nn.Module, experts: torch.nn.Module) -> Optional[torch.nn.Module]:
    E, _, H = experts.gate_up_proj.shape
    for name, child in block.named_children():
        w = getattr(child, "weight", None)
        if child is not experts and isinstance(w, torch.Tensor) and tuple(w.shape) == (E, H):
            return child
    return None


def router_rule(router: torch.nn.Module) -> Tuple[str, str]:
    """(rule, note): softmax, sigmoid_bias or unmeasured, read from the router's forward."""
    try:
        src = inspect.getsource(type(router).forward)
    except (OSError, TypeError):
        return "unmeasured", "the router's source is not readable"
    if not hasattr(router, "top_k"):
        return "unmeasured", "the router has no top_k"
    if "sigmoid" in src and hasattr(router, "e_score_correction_bias"):
        return "sigmoid_bias", ""
    if "softmax" in src and "sigmoid" not in src and "bias" not in src:
        return "softmax", ""
    return "unmeasured", f"{type(router).__name__}'s scoring rule is not one this report bounds"


def _capture(model: torch.nn.Module, routers: Dict[str, torch.nn.Module], args, kwargs):
    seen: Dict[str, List[Any]] = {}
    hooks = []
    for path, r in routers.items():
        def pre(_m, a, _p=path):
            seen.setdefault(_p, [None, None])[0] = a[0].detach().reshape(-1, a[0].shape[-1])

        def post(_m, a, out, _p=path):
            seen.setdefault(_p, [None, None])[1] = out[2].detach() if isinstance(out, tuple) and len(out) > 2 else None

        hooks.append(r.register_forward_pre_hook(pre))
        hooks.append(r.register_forward_hook(post))
    try:
        with torch.no_grad():
            model(*args, **kwargs)
    finally:
        for h in hooks:
            h.remove()
    return seen


def _scores(rule: str, router: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    if rule == "softmax":
        return z
    return torch.sigmoid(z) + router.e_score_correction_bias.detach().double()


def _score_bounds(rule: str, z: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    if rule == "softmax":
        return b + 8 * U32 * z.abs().amax(-1, keepdim=True).clamp_min(1.0)
    d = torch.sigmoid(z) * (1 - torch.sigmoid(z))
    return d * b + 4 * U32 * s.abs()


def _boundary(sel: torch.Tensor, bound: torch.Tensor, k: int, allowed: Optional[torch.Tensor] = None):
    """Per token: (ratio gap / bound, gap, bound, k-th index, (k+1)-th index) at the top k boundary,
    under the tie rule (stable descending order)."""
    s = sel if allowed is None else torch.where(allowed, sel, torch.full_like(sel, -math.inf))
    order = torch.sort(s, dim=-1, descending=True, stable=True).indices
    a, c = order[:, k - 1], order[:, k]
    rows = torch.arange(sel.shape[0])
    gap = s[rows, a] - s[rows, c]
    bnd = bound[rows, a] + bound[rows, c]
    return gap / bnd.clamp_min(1e-300), gap, bnd, a, c


def router_ties(model: torch.nn.Module, args, kwargs=None, dtype: Optional[torch.dtype] = None,
                device: Optional[str] = None) -> TieReport:
    """Run ``model`` in float64 on the CPU and in ``dtype`` on ``device`` (defaults: the model's own),
    and report every token whose top k expert choice is a near tie (module docstring)."""
    kwargs = kwargs or {}
    args = args if isinstance(args, (tuple, list)) else (args,)
    p0 = next(model.parameters())
    dtype = dtype or p0.dtype
    device = str(device or p0.device)
    covered, _ = scan(model)
    mods = dict(model.named_modules())
    routers, layers = {}, []
    for c in covered:
        parent = c.rsplit(".", 1)[0] if "." in c else ""
        r = _router_of(mods[parent], mods[c])
        if r is None:
            layers.append(Layer(parent, "unmeasured", note="no router with an [E, H] weight next to the experts"))
            continue
        name = next(n for n, m in mods.items() if m is r)
        rule, note = router_rule(r)
        layers.append(Layer(name, rule, k=int(getattr(r, "top_k", 0)), note=note))
        if rule != "unmeasured":
            routers[name] = r

    def cast(t, dt, dev):
        if dev == "cpu" and torch.is_tensor(t) and t.is_floating_point():
            return t.cpu().to(dt)   # off the device first, then widened (no combined copy, docs/METAL.md)
        return t.to(dev, dt) if torch.is_tensor(t) and t.is_floating_point() else (
            t.to(dev) if torch.is_tensor(t) else t)

    m64 = copy.deepcopy(model).cpu().to(torch.float64).eval()
    r64 = {n: dict(m64.named_modules())[n] for n in routers}
    ref = _capture(m64, r64, [cast(a, torch.float64, "cpu") for a in args],
                   {k: cast(v, torch.float64, "cpu") for k, v in kwargs.items()})
    mwp = copy.deepcopy(model).to(device, dtype).eval()
    rwp = {n: dict(mwp.named_modules())[n] for n in routers}
    got = _capture(mwp, rwp, [cast(a, dtype, device) for a in args],
                   {k: cast(v, dtype, device) for k, v in kwargs.items()})
    u = U[dtype]
    u_acc = U[torch.float32] if dtype in (torch.float16, torch.bfloat16) else u
    for layer in layers:
        if layer.path not in routers:
            continue
        r = routers[layer.path]
        x, _ = ref[layer.path]
        xw, idx_w = got.get(layer.path, (None, None))
        W = KV.to_host_f64(r.weight)
        H = W.shape[1]
        z = x @ W.t()
        absw = W.abs()
        b = (2 * u + _gamma(H, u_acc)) * (x.abs() @ absw.t())
        if xw is not None:
            b = b + (KV.to_host_f64(xw) - x).abs() @ absw.t()
        sel = _scores(layer.rule, r, z)
        bound = _score_bounds(layer.rule, z, b, sel)
        k, E = layer.k, W.shape[0]
        layer.tokens = int(x.shape[0])
        allowed = None
        if layer.rule == "sigmoid_bias" and getattr(r, "num_group", 1) > 1 and r.topk_group < r.num_group:
            G = r.num_group
            gs = sel.view(-1, G, E // G)
            top2 = torch.sort(gs, dim=-1, descending=True, stable=True)
            gsc = top2.values[..., :2].sum(-1)
            gb = torch.gather(bound.view(-1, G, E // G), 2, top2.indices[..., :2]).sum(-1)
            ratio, gap, bnd, a, c = _boundary(gsc, gb, r.topk_group)
            chosen = torch.sort(gsc, dim=-1, descending=True, stable=True).indices[:, :r.topk_group]
            gmask = torch.zeros_like(gsc, dtype=torch.bool).scatter_(1, chosen, True)
            allowed = gmask.unsqueeze(-1).expand(-1, G, E // G).reshape(-1, E)
            for t in (ratio <= 1).nonzero().flatten().tolist():
                layer.ties.append(Tie(layer.path, t, _position(t, args), int(a[t]), int(c[t]), float(gap[t]),
                                      float(bnd[t]), False, group=True))
            layer.min_ratio = min(layer.min_ratio, float(ratio.min()))
        if k >= (E if allowed is None else int(allowed[0].sum())):
            layer.note = "every expert is chosen (k equals the candidates): no boundary"
            continue
        ratio, gap, bnd, a, c = _boundary(sel, bound, k, allowed)
        want = torch.sort(torch.where(allowed, sel, torch.full_like(sel, -math.inf)) if allowed is not None else sel,
                          dim=-1, descending=True, stable=True).indices[:, :k]
        flipped = torch.zeros(layer.tokens, dtype=torch.bool)
        if idx_w is not None:
            gw = torch.sort(idx_w.to("cpu"), dim=-1).values
            flipped = (gw != torch.sort(want, dim=-1).values).any(-1)
        layer.flips = flipped.nonzero().flatten().tolist()
        near = ratio <= 1
        group_ties = {t.token for t in layer.ties}
        layer.real_flips = [t for t in layer.flips if not near[t] and t not in group_ties]
        for t in near.nonzero().flatten().tolist():
            layer.ties.append(Tie(layer.path, t, _position(t, args), int(a[t]), int(c[t]), float(gap[t]),
                                  float(bnd[t]), bool(flipped[t])))
        layer.min_ratio = min(layer.min_ratio, float(ratio.min()))
    return TieReport(dtype, device, layers)


def _position(t: int, args) -> Tuple[int, ...]:
    """(batch, position) of flattened token t when the first input is [batch, seq, ...]."""
    x = args[0] if args else None
    if torch.is_tensor(x) and x.dim() >= 2:
        return (t // x.shape[1], t % x.shape[1])
    return (t,)
