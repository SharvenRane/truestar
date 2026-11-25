"""Attention logit overflow proven ahead of time from the weights (docs/TRAIN_NUMERICS.md).

    from truestar import logitbound
    rep = logitbound.check_module(model, target="fp16")   # torch TransformerEncoderLayer stacks, HF Llama style
    print(rep.text())                                     # per head: upper bound, reachable value, verdict
    logitbound.guard(model, target="fp16")                # raises TS1053 when an overflow is reachable

For one head with query and key projections W_q, W_k [d_h, d] (biases b_q, b_k) reading an input
whose 2 norm is at most R, the logit of a query token y and a key token y' is

    s = scale (W_q y + b_q) . (W_k y' + b_k)
      = scale (y^T M y' + (W_k^T b_q) . y' + (W_q^T b_k) . y + b_q . b_k),   M = W_q^T W_k

so |s| <= scale (sigma_max(M) R^2 + (|W_k^T b_q| + |W_q^T b_k|) R + |b_q . b_k|), with sigma_max
from an SVD in float64 (exact up to float64 rounding, no sampling). R comes from the layer in front:
a LayerNorm gives ||y|| <= max|gamma| sqrt(d) + ||beta||, an RMSNorm max|gamma| sqrt(d) (eps only
shrinks it); without a norm the caller gives R (or an elementwise range, R = sqrt(d) max|x|).
Rotary position embeddings rotate q and k by orthogonal matrices that depend on the positions, so
with RoPE the bound is the product form scale ||q|| ||k|| <= scale (sigma(W_q) R + |b_q|)(sigma(W_k)
R + |b_k|), which holds for every offset. A per head RMSNorm on q and k (QK norm) bounds ||q|| by
max|gamma_q| sqrt(d_h) directly. A softcap c bounds |s| by c.

The bound is checked against a value the model really reaches: an input built to maximize the
logit (the top singular pair of M, then a few rounds of the exact best response of each token under
its norm's constraint) is run through the real modules, and its logit is reported as
``reachable``. So every head has three numbers: reachable <= true max <= upper. Verdicts per target
format (the logit's storage, and q and k entries when they are stored unscaled):

* ``safe`` (TS1051): the upper bound is inside the format's finite range: no input can overflow;
* ``may_overflow`` (TS1052, warning): the upper bound is outside, the reachable value inside;
* ``overflows`` (TS1053, error): a valid input reaches a logit outside the range; ``guard`` refuses.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..codes import Issue

F64 = torch.float64

LIMITS = {"fp16": 65504.0, "bf16": 3.3895313892515355e38, "fp8_e4m3": 448.0, "fp8_e5m2": 57344.0,
          "fp32": 3.4028234663852886e38}
# exp(s) of an unshifted softmax overflows in the format above log(max)
EXP_LIMITS = {k: math.log(v) for k, v in LIMITS.items()}


class LogitOverflowError(RuntimeError):
    def __init__(self, report: "LogitReport"):
        self.report = report
        self.code = "TS1053"
        bad = [h for h in report.heads if h.verdict == "overflows"]
        super().__init__("truestar TS1053 attention logits overflow " + report.target + ": "
                         + "; ".join(h.line(report.target) for h in bad[:8])
                         + (f"; and {len(bad) - 8} more heads" if len(bad) > 8 else ""))


@dataclasses.dataclass
class HeadBound:
    layer: str
    head: int
    kv_head: int
    input_bound: float          # R
    upper: float                # bound on |logit| over every input with ||y|| <= R
    reachable: float            # |logit| a constructed valid input reaches through the real modules
    q_entry_upper: float        # bound on any |q_i| (and below for k): unscaled storage of q and k
    k_entry_upper: float
    method: str                 # "bilinear", "product (rope)", "qk norm", "softcap"
    verdict: str = ""

    @property
    def gap(self) -> float:
        return self.upper / self.reachable if self.reachable > 0 else math.inf

    def line(self, target: str) -> str:
        return (f"{self.layer} head {self.head}: |logit| <= {self.upper:.4g} (reachable {self.reachable:.4g}, "
                f"gap {self.gap:.2f}x) vs {target} max {LIMITS[target]:.6g}: {self.verdict}")


@dataclasses.dataclass
class LogitReport:
    target: str
    heads: List[HeadBound]
    entries_checked: bool

    @property
    def worst(self) -> Optional[HeadBound]:
        return max(self.heads, key=lambda h: h.upper) if self.heads else None

    @property
    def verdict(self) -> str:
        vs = {h.verdict for h in self.heads}
        return "overflows" if "overflows" in vs else "may_overflow" if "may_overflow" in vs else "safe"

    def issues(self) -> List[Issue]:
        code = {"safe": "TS1051", "may_overflow": "TS1052", "overflows": "TS1053"}[self.verdict]
        w = self.worst
        msg = (f"{len(self.heads)} heads against {self.target}; worst {w.line(self.target)}" if w else "no heads")
        return [Issue(code, msg)]

    def text(self) -> str:
        lines = [f"attention logit bound against {self.target} (max {LIMITS[self.target]:.6g}): {self.verdict}"]
        for h in sorted(self.heads, key=lambda h: -h.upper):
            lines.append("  " + h.line(self.target) + f" [{h.method}, R {h.input_bound:.4g}]")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {"target": self.target, "verdict": self.verdict,
                "heads": [dataclasses.asdict(h) | {"gap": h.gap} for h in self.heads]}


def _verdict(h: HeadBound, target: str, entries: bool) -> str:
    lim = LIMITS[target]
    up = max(h.upper, h.q_entry_upper, h.k_entry_upper) if entries else h.upper
    if h.reachable > lim:
        return "overflows"
    if up > lim:
        return "may_overflow"
    return "safe"


# ----------------------------------------------------------------------------- input norm bounds


def norm_bound(kind: str, d: int, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None) -> float:
    """Largest 2 norm of the output of a LayerNorm ("layernorm") or RMSNorm ("rmsnorm") over d
    features. The normalized vector has 2 norm sqrt(d) * sqrt(var / (var + eps)) <= sqrt(d)."""
    g = 1.0 if weight is None else float(weight.detach().to(F64).abs().max())
    r = g * math.sqrt(d)
    if kind == "layernorm" and bias is not None:
        r += float(bias.detach().to(F64).norm())
    elif kind not in ("layernorm", "rmsnorm"):
        raise ValueError(f"norm kind is 'layernorm' or 'rmsnorm', got {kind!r}")
    return r


# ----------------------------------------------------------------------------- one head


def _spec(w: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(w.detach().to(F64).cpu(), ord=2)) if w.numel() else 0.0


def head_upper(wq: torch.Tensor, wk: torch.Tensor, R: float, scale: float, bq=None, bk=None, rope: bool = False,
               qk_norm: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, softcap: Optional[float] = None
               ) -> Tuple[float, float, float, str]:
    """(bound on |logit|, bound on |q_i|, bound on |k_i|, method) for one head."""
    wq, wk = wq.detach().to(F64).cpu(), wk.detach().to(F64).cpu()
    bq = None if bq is None else bq.detach().to(F64).cpu()
    bk = None if bk is None else bk.detach().to(F64).cpu()
    dh = wq.shape[0]
    rows_q = wq.norm(dim=1) * R + (bq.abs() if bq is not None else 0.0)
    rows_k = wk.norm(dim=1) * R + (bk.abs() if bk is not None else 0.0)
    q_ent, k_ent = float(rows_q.max()), float(rows_k.max())
    if qk_norm is not None:
        gq, gk = (float(g.detach().to(F64).abs().max()) for g in qk_norm)
        up = scale * gq * gk * dh
        q_ent, k_ent = gq * math.sqrt(dh), gk * math.sqrt(dh)
        method = "qk norm"
    elif rope:
        nq = _spec(wq) * R + (float(bq.norm()) if bq is not None else 0.0)
        nk = _spec(wk) * R + (float(bk.norm()) if bk is not None else 0.0)
        up = scale * nq * nk
        method = "product (rope)"
    else:
        M = wq.T @ wk
        up = _spec(M) * R * R
        if bq is not None:
            up += float((wk.T @ bq).norm()) * R
        if bk is not None:
            up += float((wq.T @ bk).norm()) * R
        if bq is not None and bk is not None:
            up += abs(float(bq @ bk))
        up *= scale
        method = "bilinear"
    if softcap is not None and softcap < up:
        up, method = float(softcap), "softcap"
    return up, q_ent, k_ent, method


def _best_response(a: torch.Tensor, kind: str, gamma: Optional[torch.Tensor], d: int) -> torch.Tensor:
    """The normalized vector z (mean 0 for a LayerNorm, 2 norm sqrt(d)) maximizing (gamma * z) . a."""
    v = a * (gamma if gamma is not None else 1.0)
    if kind == "layernorm":
        v = v - v.mean()
    n = float(v.norm())
    if n == 0:
        v = torch.randn_like(v)
        if kind == "layernorm":
            v = v - v.mean()
        n = float(v.norm())
    return v * (math.sqrt(d) / n)


def adversarial_pair(wq, wk, kind: str, gamma=None, beta=None, bq=None, bk=None, rounds: int = 8
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two normalized vectors (before the norm's gamma and beta) for a query and a key token whose
    logit is large: start from the top singular pair of W_q^T W_k, then alternate exact best
    responses. Maximizes |s| (the larger of the positive and negative attempt)."""
    wq, wk = wq.detach().to(F64).cpu(), wk.detach().to(F64).cpu()
    d = wq.shape[1]
    g = None if gamma is None else gamma.detach().to(F64).cpu()
    b = torch.zeros(d, dtype=F64) if beta is None else beta.detach().to(F64).cpu()
    bq = torch.zeros(wq.shape[0], dtype=F64) if bq is None else bq.detach().to(F64).cpu()
    bk = torch.zeros(wk.shape[0], dtype=F64) if bk is None else bk.detach().to(F64).cpu()
    U, S, Vh = torch.linalg.svd(wq.T @ wk)
    best = None
    for sign in (1.0, -1.0):
        zq = _best_response(U[:, 0], kind, g, d)
        zk = _best_response(sign * Vh[0], kind, g, d)
        for _ in range(rounds):
            yq = (g * zq if g is not None else zq) + b
            k = wk @ ((g * zk if g is not None else zk) + b) + bk
            zq = _best_response(sign * (wq.T @ k), kind, g, d)
            yq = (g * zq if g is not None else zq) + b
            q = wq @ yq + bq
            zk = _best_response(sign * (wk.T @ q), kind, g, d)
        yq = (g * zq if g is not None else zq) + b
        yk = (g * zk if g is not None else zk) + b
        s = abs(float((wq @ yq + bq) @ (wk @ yk + bk)))
        if best is None or s > best[0]:
            best = (s, zq, zk)
    return best[1], best[2]


# ----------------------------------------------------------------------------- whole models


@dataclasses.dataclass
class AttnSite:
    """What the bound needs from one attention layer."""

    name: str
    wq: torch.Tensor            # [heads * d_h, d]
    wk: torch.Tensor            # [kv_heads * d_h, d]
    heads: int
    kv_heads: int
    bq: Optional[torch.Tensor] = None
    bk: Optional[torch.Tensor] = None
    norm: Optional[str] = None  # "layernorm", "rmsnorm" or None (then input_bound is needed)
    gamma: Optional[torch.Tensor] = None
    beta: Optional[torch.Tensor] = None
    scale: Optional[float] = None
    rope: bool = False
    q_norm: Optional[torch.Tensor] = None
    k_norm: Optional[torch.Tensor] = None
    softcap: Optional[float] = None
    logits: Any = None          # callable(zq, zk) -> logits [heads] through the real modules (the reachable value)


def sites_of(model: torch.nn.Module) -> List[AttnSite]:
    """Attention layers Truestar can read: torch.nn.TransformerEncoderLayer and DecoderLayer (their
    self attention; the norm in front when norm_first, else none) and Hugging Face style decoder
    layers with ``self_attn.q_proj`` / ``k_proj`` and an ``input_layernorm`` (Llama, Mistral, Qwen2,
    Qwen3 with q_norm / k_norm). Other attention is not read (an empty list means nothing found)."""
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.TransformerEncoderLayer, torch.nn.TransformerDecoderLayer)):
            mha = mod.self_attn
            E, H = mha.embed_dim, mha.num_heads
            w = mha.in_proj_weight
            b = mha.in_proj_bias
            nf = bool(getattr(mod, "norm_first", False))
            site = AttnSite(name or "layer", w[:E], w[E:2 * E], H, H,
                            None if b is None else b[:E], None if b is None else b[E:2 * E],
                            "layernorm" if nf else None, mod.norm1.weight if nf else None,
                            mod.norm1.bias if nf else None, 1.0 / math.sqrt(E // H))
            if nf:
                site.logits = _torch_mha_logits(mod.norm1, mha)
            out.append(site)
        elif hasattr(mod, "self_attn") and hasattr(mod.self_attn, "q_proj") and hasattr(mod, "input_layernorm"):
            at = mod.self_attn
            cfg = getattr(at, "config", None)
            H = getattr(cfg, "num_attention_heads", None) or getattr(at, "num_heads")
            KV = getattr(cfg, "num_key_value_heads", None) or getattr(at, "num_key_value_heads", H)
            dh = getattr(at, "head_dim", at.q_proj.weight.shape[0] // H)
            ln = mod.input_layernorm
            kind = "layernorm" if isinstance(ln, torch.nn.LayerNorm) else "rmsnorm"
            qn = getattr(at, "q_norm", None)
            kn = getattr(at, "k_norm", None)
            softcap = getattr(cfg, "attn_logit_softcapping", None) if cfg is not None else None
            scale = getattr(at, "scaling", None) or 1.0 / math.sqrt(dh)
            site = AttnSite(name, at.q_proj.weight, at.k_proj.weight, H, KV, at.q_proj.bias, at.k_proj.bias, kind,
                            getattr(ln, "weight", None), getattr(ln, "bias", None), scale, rope=True,
                            q_norm=None if qn is None else qn.weight, k_norm=None if kn is None else kn.weight,
                            softcap=softcap)
            site.logits = _hf_logits(ln, at, H, KV, dh, scale)
            out.append(site)
    return out


def _torch_mha_logits(norm, mha):
    E, H = mha.embed_dim, mha.num_heads
    dh = E // H

    @torch.no_grad()
    def run(zq, zk):
        x = torch.stack([zq, zk]).to(mha.in_proj_weight.dtype).to(mha.in_proj_weight.device)
        y = norm(x)
        w, b = mha.in_proj_weight, mha.in_proj_bias
        q = torch.nn.functional.linear(y[0], w[:E], None if b is None else b[:E]).view(H, dh)
        k = torch.nn.functional.linear(y[1], w[E:2 * E], None if b is None else b[E:2 * E]).view(H, dh)
        return (q.to(F64) * k.to(F64)).sum(-1) / math.sqrt(dh)
    return run


def _hf_logits(ln, at, H, KV, dh, scale):
    @torch.no_grad()
    def run(zq, zk):
        """Both tokens at position 0 (RoPE is the identity there), through the real norm and
        projections (and q_norm / k_norm)."""
        dev, dt = at.q_proj.weight.device, at.q_proj.weight.dtype
        x = torch.stack([zq, zk]).to(dt).to(dev)
        y = ln(x)
        q = at.q_proj(y[0]).view(H, dh)
        k = at.k_proj(y[1]).view(KV, dh)
        if getattr(at, "q_norm", None) is not None:
            q = at.q_norm(q)
        if getattr(at, "k_norm", None) is not None:
            k = at.k_norm(k)
        k = k.repeat_interleave(H // KV, dim=0)
        return (q.to(F64) * k.to(F64)).sum(-1) * scale
    return run


def check_sites(sites: Sequence[AttnSite], target: str = "fp16", input_bound: Optional[float] = None,
                entries: bool = True) -> LogitReport:
    """Per head bounds of every site against ``target`` (fp16, bf16, fp8_e4m3, fp8_e5m2).
    ``entries``: also check q and k entries against the format (unscaled storage of q and k)."""
    if target not in LIMITS:
        raise ValueError(f"target is one of {', '.join(LIMITS)}, got {target!r}")
    heads = []
    for s in sites:
        d = s.wq.shape[1]
        dh = s.wq.shape[0] // s.heads
        scale = s.scale if s.scale is not None else 1.0 / math.sqrt(dh)
        if s.norm is not None:
            R = norm_bound(s.norm, d, s.gamma, s.beta if s.norm == "layernorm" else None)
        elif input_bound is not None:
            R = float(input_bound)
        else:
            raise ValueError(f"{s.name}: no norm in front of the attention, so the input bound R is needed "
                             "(input_bound=..., the largest 2 norm of a token vector it reads)")
        reach = None
        if s.logits is not None and s.norm is not None:
            reach = torch.zeros(s.heads, dtype=F64)
        for h in range(s.heads):
            kv = h // (s.heads // s.kv_heads)
            sl_q = slice(h * dh, (h + 1) * dh)
            sl_k = slice(kv * dh, (kv + 1) * dh)
            bq = None if s.bq is None else s.bq[sl_q]
            bk = None if s.bk is None else s.bk[sl_k]
            qn = (s.q_norm, s.k_norm) if s.q_norm is not None and s.k_norm is not None else None
            up, qe, ke, method = head_upper(s.wq[sl_q], s.wk[sl_k], R, scale, bq, bk, s.rope, qn, s.softcap)
            r = 0.0
            if reach is not None:
                zq, zk = adversarial_pair(s.wq[sl_q], s.wk[sl_k], s.norm, s.gamma, s.beta, bq, bk)
                r = float(s.logits(zq, zk)[h].abs())
                if s.softcap is not None:
                    r = min(r, float(s.softcap))
            hb = HeadBound(s.name, h, kv, R, up, r, qe, ke, method)
            hb.verdict = _verdict(hb, target, entries)
            heads.append(hb)
    return LogitReport(target, heads, entries)


def check_module(model: torch.nn.Module, target: str = "fp16", input_bound: Optional[float] = None,
                 entries: bool = True) -> LogitReport:
    sites = sites_of(model)
    if not sites:
        raise ValueError("no attention layer Truestar can read in this model (torch Transformer layers and "
                         "Hugging Face q_proj / k_proj decoder layers are read)")
    return check_sites(sites, target, input_bound, entries)


def guard(model: torch.nn.Module, target: str = "fp16", input_bound: Optional[float] = None,
          entries: bool = True) -> LogitReport:
    """Refuse (LogitOverflowError, TS1053) when a valid input reaches a logit outside ``target``;
    warn (TS1052) when only the upper bound is outside; return the report otherwise."""
    rep = check_module(model, target, input_bound, entries)
    if rep.verdict == "overflows":
        raise LogitOverflowError(rep)
    if rep.verdict == "may_overflow":
        warnings.warn("truestar TS1052 " + rep.issues()[0].message, RuntimeWarning, stacklevel=2)
    return rep


def steps_to_limit(history: Sequence[Tuple[int, float]], target: str = "fp16", window: int = 8) -> Optional[float]:
    """Steps until the bound reaches ``target``'s max at the growth rate of the last ``window``
    (step, bound) pairs (a straight line fit of log bound); None when it is not growing."""
    pts = [(float(s), math.log(b)) for s, b in history[-window:] if b > 0]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    var = sum((p[0] - mx) ** 2 for p in pts)
    if var == 0:
        return None
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / var
    if slope <= 0:
        return None
    last_s, last_y = pts[-1]
    return max(0.0, (math.log(LIMITS[target]) - last_y) / slope)


__all__ = ["LIMITS", "HeadBound", "LogitReport", "LogitOverflowError", "AttnSite", "norm_bound", "head_upper",
           "adversarial_pair", "sites_of", "check_sites", "check_module", "guard", "steps_to_limit"]
