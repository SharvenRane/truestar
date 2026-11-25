"""`truestar port reference.py gguf:model.gguf`: a PyTorch reference against llama.cpp running a GGUF.

Three verdicts that never mix (docs/PORT_ENGINES.md):

(a) **engine**: every engine graph node on the path is compared with Truestar's float64 replay of
    that one step, fed the engine's own inputs for the step, the GGUF's weights decoded exactly and
    the GGUF's settings, with the rounding the engine is allowed to do modeled exactly (activations
    rounded to the dot product type of the weights: Q8_0 blocks for Q8_0 weights, Q8_K blocks for K
    quants, f16 for f16 weights and an f16 KV cache). What remains is the engine's own error.
(b) **conversion**: the GGUF's settings and tensors against the reference's, statically (each
    setting by name, each tensor matched by content with its layout) and as a float64 replay of the
    GGUF against the GGUF with the found faults undone, so each fault is located at a step.
(c) **quantization**: float64 replays only, the reference against the same model with the GGUF's
    decoded weights and modeled activation rounding: top 1 agreement, KL, per layer change.

Settings the engine actually used are read from its graph (RMS_NORM eps, ROPE freq base, freq
scale, dims and mode, SOFT_MAX scale) and compared with the GGUF, so "the engine ignored a setting"
is told apart from "the file holds a wrong setting".
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..codes import CODES, ERROR, Issue, severity
from ..compare import DIVERGENT, OK, SUSPECT
from . import gguf, llama_ref as R
from .llama_ref import F64, LlamaConfig

ARCHS = ("llama",)

# expected relative difference of a decoded tensor from its float64 source, per type (measured on
# Gaussian weights in tests/test_engine_port.py, rounded up about 2x)
TYPE_BOUND = {"F32": 1e-6, "F16": 1e-3, "BF16": 1e-2, "Q8_0": 1.5e-2, "Q4_0": 0.2, "Q4_K": 0.15,
              "Q5_K": 0.08, "Q6_K": 0.04}
VALUES_DIFFER = 0.5      # closest tensor of the right shape but over the type bound
CONVERSION_FLOOR = 1e-9
REPLAY_EXACT = 1e-10     # the replay reproduces a float64 reference
REPLAY_FLOAT32 = 1e-5    # a reference that computes some parts in float32; an architecture mismatch is far above  # both sides of verdict (b) are float64
ENGINE_FLOOR = 1e-6      # the engine computes in float32
ENGINE_FACTOR = 10.0     # threshold = 10 x the float32 replay's own error, never below the floor

GGUF_ROLES = {"attn_norm": "attn_norm", "attn_q": "q", "attn_k": "k", "attn_v": "v", "attn_output": "o",
              "ffn_norm": "ffn_norm", "ffn_gate": "gate", "ffn_up": "up", "ffn_down": "down"}
GLOBAL_GGUF = {"token_embd": "embed", "output_norm": "norm", "output": "output"}

# step -> (engine node name, op); {i} is the layer
TAP_NAMES = {
    "embd": [("inp_embd", None), ("embd", "GET_ROWS")],
    "attn_norm": [("attn_norm-{i}", None)], "q": [("Qcur-{i}", "MUL_MAT")], "k": [("Kcur-{i}", "MUL_MAT")],
    "v": [("Vcur-{i}", "MUL_MAT")], "q_rope": [("Qcur-{i}", "ROPE")], "k_rope": [("Kcur-{i}", "ROPE")],
    "attn": [("kqv_out-{i}", None)], "attn_out": [("attn_out-{i}", None)], "ffn_inp": [("ffn_inp-{i}", None)],
    "ffn_norm": [("ffn_norm-{i}", None)], "gate": [("ffn_gate-{i}", None)], "up": [("ffn_up-{i}", None)],
    "act": [("ffn_swiglu-{i}", None), ("ffn_gate_par-{i}", None)], "down": [("ffn_out-{i}", None)],
    "out": [("l_out-{i}", None)], "result_norm": [("result_norm", None)], "logits": [("result_output", None)],
}
STEP_WORDS = {"embd": "token embedding lookup", "attn_norm": "attention RMS norm", "q": "q projection",
              "k": "k projection", "v": "v projection", "q_rope": "rotary positions on q",
              "k_rope": "rotary positions on k", "attn": "attention (scores, causal softmax, values)",
              "attn_out": "attention output projection", "ffn_inp": "attention residual add",
              "ffn_norm": "MLP RMS norm", "gate": "MLP gate projection", "up": "MLP up projection",
              "act": "SiLU gate times up", "down": "MLP down projection", "out": "MLP residual add",
              "result_norm": "final RMS norm", "logits": "output head"}


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(F64)
    b = b.to(F64)
    if a.shape != b.shape:
        return float("inf")
    n = b.norm().item()
    d = (a - b).norm().item()
    return d / n if n > 0 else (0.0 if d == 0 else float("inf"))


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2e}"


def permute_rows(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Rows of each head from the half convention to the adjacent pair convention (what the llama
    architecture in the engine rotates)."""
    out = w.shape[0]
    d = out // n_heads
    return w.reshape(n_heads, 2, d // 2, *w.shape[1:]).transpose(1, 2).reshape(w.shape)


def unpermute_rows(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    out = w.shape[0]
    d = out // n_heads
    return w.reshape(n_heads, d // 2, 2, *w.shape[1:]).transpose(1, 2).reshape(w.shape)


def unpermute_cols(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Activations [T, n_heads * d] from the adjacent pair order to the half order."""
    return unpermute_rows(x.T, n_heads).T


# ----------------------------------------------------------------------------- results


def _g(v: Any) -> str:
    return f"{v:.7g}" if isinstance(v, float) else str(v)


@dataclass
class SettingFinding:
    name: str
    reference: Any
    gguf: Any
    engine: Any = None  # what the engine graph shows it used (None when not observable)
    where: str = "conversion"  # "conversion": the file differs; "engine": the engine ignored the file

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "reference": self.reference, "gguf": self.gguf, "engine": self.engine,
                "where": self.where}

    def sentence(self) -> str:
        if self.where == "engine":
            return f"the engine used {self.name} {_g(self.engine)} although the GGUF says {_g(self.gguf)}"
        return f"{self.name} is {_g(self.gguf)} in the GGUF, {_g(self.reference)} in the reference"


@dataclass
class WeightEdge:
    gguf_name: str
    role: str
    type_name: str
    reference: Optional[str]
    kind: Optional[str]      # identity, transpose, rope_permuted, rope_permuted_transposed
    expected: str
    rel: Optional[float]
    bound: float
    verdict: str             # ok, layout, values, unmatched
    ratio: Optional[float] = None  # norm ratio when values differ

    def as_dict(self) -> Dict[str, Any]:
        return {"gguf": self.gguf_name, "role": self.role, "type": self.type_name, "reference": self.reference,
                "kind": self.kind, "expected": self.expected, "rel_err": _num(self.rel), "bound": self.bound,
                "verdict": self.verdict, "norm_ratio": _num(self.ratio)}

    def sentence(self) -> str:
        if self.verdict == "layout":
            if self.expected == "rope_permuted" and self.kind == "identity":
                return (f"{self.gguf_name} holds {self.reference} in the reference row order; the llama "
                        "architecture rotates adjacent pairs, so q and k rows must be permuted per head")
            return f"{self.gguf_name} holds {self.reference} {self.kind.replace('_', ' ')} (expected {self.expected.replace('_', ' ')})"
        if self.verdict == "values":
            return (f"{self.gguf_name} differs from {self.reference} by {self.rel:.3g} relative, more than "
                    f"{self.type_name} rounding ({self.bound:g}); norm ratio {self.ratio:.4g}")
        if self.verdict == "unmatched":
            return f"{self.gguf_name} matches no reference weight"
        return f"{self.gguf_name} = {self.reference}"


def _num(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


@dataclass
class StepResult:
    key: str
    layer: int
    step: str
    op: str = ""
    node: str = ""
    rel_err: float = 0.0
    threshold: float = 0.0
    weight_type: str = ""
    path: str = ""  # the allowed rounding path that fits the engine best

    @property
    def over(self) -> bool:
        return self.rel_err > self.threshold

    def as_dict(self) -> Dict[str, Any]:
        return {"step": self.key, "layer": self.layer, "what": STEP_WORDS.get(self.step, self.step),
                "engine_node": self.node, "engine_op": self.op, "rel_err": _num(self.rel_err),
                "threshold": _num(self.threshold), "weight_type": self.weight_type or None,
                "rounding_path": self.path or None, "over": self.over}


@dataclass
class Decisions:
    positions: int
    top1_agree: int
    flips_beyond_ties: int
    kl_mean: float

    def as_dict(self) -> Dict[str, Any]:
        return {"positions": self.positions, "top1_agreement": self.top1_agree / max(self.positions, 1),
                "flips": self.positions - self.top1_agree, "flips_beyond_near_ties": self.flips_beyond_ties,
                "kl_mean": _num(self.kl_mean)}

    def short(self) -> str:
        return (f"top 1 agreement {100.0 * self.top1_agree / max(self.positions, 1):.1f}% "
                f"({self.positions - self.top1_agree} of {self.positions} flipped, "
                f"{self.flips_beyond_ties} beyond near ties), KL {self.kl_mean:.2e}")


def decisions(cand: torch.Tensor, ref: torch.Tensor, tie: float = 0.01) -> Decisions:
    """Top 1 agreement per position; a flip is a near tie when the reference's top two logits are
    closer than `tie` times that position's logit standard deviation."""
    cand = cand.to(F64)
    ref = ref.to(F64)
    top = ref.topk(2, dim=-1).values
    gap = top[:, 0] - top[:, 1]
    near = gap < tie * ref.std(dim=-1)
    agree = cand.argmax(-1) == ref.argmax(-1)
    lp = torch.log_softmax(ref, -1)
    lq = torch.log_softmax(cand, -1)
    kl = max(0.0, (lp.exp() * (lp - lq)).sum(-1).mean().item())  # never below 0 from rounding
    return Decisions(int(ref.shape[0]), int(agree.sum()), int((~agree & ~near).sum()), kl)


@dataclass
class EnginePortReport:
    reference_name: str
    candidate_name: str
    arch: str = ""
    status: str = OK
    issues: List[Issue] = field(default_factory=list)
    settings: List[SettingFinding] = field(default_factory=list)
    observed: Dict[str, Any] = field(default_factory=dict)
    weights: List[WeightEdge] = field(default_factory=list)
    reference_unmatched: List[str] = field(default_factory=list)
    engine_steps: List[StepResult] = field(default_factory=list)
    conversion_steps: List[StepResult] = field(default_factory=list)
    decomposition: Dict[str, Optional[float]] = field(default_factory=dict)
    decisions_total: Optional[Decisions] = None
    decisions_engine: Optional[Decisions] = None
    decisions_quant: Optional[Decisions] = None
    quant_types: Dict[str, int] = field(default_factory=dict)
    tokenizer: Optional[Dict[str, Any]] = None
    observer: Optional[Dict[str, Any]] = None
    engine: Dict[str, Any] = field(default_factory=dict)
    seconds: Dict[str, float] = field(default_factory=dict)
    blame: Optional[str] = None  # "engine", "settings", "weights", "tokenizer", "quantization", None
    explanation: str = ""

    @property
    def codes(self) -> List[str]:
        out: List[str] = []
        for i in self.issues:
            if i.code not in out:
                out.append(i.code)
        return out

    def first_engine(self) -> Optional[StepResult]:
        return next((s for s in self.engine_steps if s.over), None)

    def first_conversion(self) -> Optional[StepResult]:
        return next((s for s in self.conversion_steps if s.over), None)

    def as_dict(self) -> Dict[str, Any]:
        fe = self.first_engine()
        fc = self.first_conversion()
        return {
            "arch": self.arch, "status": self.status, "codes": self.codes,
            "issues": [i.as_dict() for i in self.issues], "blame": self.blame,
            "settings": [s.as_dict() for s in self.settings], "observed_engine_settings": self.observed,
            "weights": {"edges": [w.as_dict() for w in self.weights],
                        "reference_unmatched": self.reference_unmatched, "types": self.quant_types},
            "engine_verdict": {"first_over_threshold": fe.as_dict() if fe else None,
                               "steps_checked": len(self.engine_steps),
                               "steps_over_threshold": [s.as_dict() for s in self.engine_steps if s.over][:5],
                               "worst_ratio": _num(max((s.rel_err / s.threshold for s in self.engine_steps),
                                                       default=None)),
                               "decisions_vs_gguf_replay": self.decisions_engine.as_dict() if self.decisions_engine else None},
            "conversion_verdict": {"first_over_threshold": fc.as_dict() if fc else None,
                                   "steps_over_threshold": [s.as_dict() for s in self.conversion_steps if s.over][:5]},
            "quantization_verdict": self.decisions_quant.as_dict() if self.decisions_quant else None,
            "decomposition": {k: _num(v) for k, v in self.decomposition.items()},
            "decisions_vs_reference": self.decisions_total.as_dict() if self.decisions_total else None,
            "tokenizer": self.tokenizer, "observer": self.observer, "engine": self.engine,
            "seconds": self.seconds, "explanation": self.explanation,
        }

    def to_text(self) -> str:
        L = [f"truestar port: reference {self.reference_name} vs engine {self.candidate_name} ({self.arch})"]
        types = ", ".join(f"{n} {k}" for k, n in sorted(self.quant_types.items()))
        L.append(f"engine: {self.engine.get('version', '?')} on {self.engine.get('device', 'cpu')}, "
                 f"kv cache {self.engine.get('kv_type')}; GGUF tensors: {types}")
        d = self.decomposition
        L.append(f"final logits vs reference: {_fmt(d.get('total'))}; engine vs GGUF replay {_fmt(d.get('engine'))}, "
                 f"conversion {_fmt(d.get('conversion'))}, quantization {_fmt(d.get('quantization'))}")
        if self.decisions_total:
            L.append(f"decisions vs reference: {self.decisions_total.short()}")
        fe = self.first_engine()
        worst = max((s.rel_err / s.threshold for s in self.engine_steps), default=0.0)
        L.append(f"engine verdict: {len(self.engine_steps)} steps replayed on the engine's own inputs, "
                 + (f"first over threshold {fe.key} ({fe.node} {fe.op}) own error {fe.rel_err:.2e}, "
                    f"threshold {fe.threshold:.1e}" if fe else f"all within threshold (worst ratio {worst:.2f})"))
        for s in self.settings:
            L.append(f"setting: {s.sentence()}")
        for w in self.weights:
            if w.verdict != "ok":
                L.append(f"weight: {w.sentence()}")
        if self.decisions_quant:
            L.append(f"quantization verdict (float64 replays only): {self.decisions_quant.short()}")
        if self.tokenizer and not self.tokenizer.get("equal", True):
            L.append(f"tokenizer: {self.tokenizer['message']}")
        if self.observer:
            L.append(f"observer check: tapped vs untapped logits max difference {self.observer['max_abs']:.2e}")
        if self.status == OK and self.blame == "quantization":
            L.append("status: OK (quantization loss only)")
        else:
            L.append(f"status: {self.status.upper()}" + (f" (blame: {self.blame})" if self.blame else ""))
        if self.explanation:
            L.append("")
            L.append(self.explanation)
        return "\n".join(L)


# ----------------------------------------------------------------------------- inputs


class EnginePortError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def gguf_config(g: gguf.GGUFFile) -> LlamaConfig:
    a = g.arch()
    md = g.metadata

    def k(name, default=None):
        v = md.get(f"{a}.{name}", default)
        if v is None:
            raise EnginePortError("TS650", f"GGUF metadata {a}.{name} is missing")
        return v
    dim = int(k("embedding_length"))
    heads = int(k("attention.head_count"))
    scale = 1.0
    if md.get(f"{a}.rope.scaling.type") == "linear":
        scale = float(md.get(f"{a}.rope.scaling.factor", 1.0))
    elif md.get(f"{a}.rope.scaling.type") not in (None, "none"):
        raise EnginePortError("TS650", f"rope scaling {md.get(f'{a}.rope.scaling.type')!r} is not replayed")
    vocab = int(md.get(f"{a}.vocab_size", len(md.get("tokenizer.ggml.tokens", [])) or 0))
    return LlamaConfig(vocab=vocab, dim=dim, n_layers=int(k("block_count")), n_heads=heads,
                       n_kv_heads=int(md.get(f"{a}.attention.head_count_kv", heads)), ffn=int(k("feed_forward_length")),
                       eps=float(k("attention.layer_norm_rms_epsilon")), theta=float(md.get(f"{a}.rope.freq_base", 10000.0)),
                       rope_scale=scale, head_dim=int(md.get(f"{a}.rope.dimension_count", dim // heads)),
                       tied="output.weight" not in g.tensors, context=int(md.get(f"{a}.context_length", 0)))


def _role(name: str) -> Optional[str]:
    if not name.endswith(".weight"):
        return None
    base = name[:-7]
    if base in GLOBAL_GGUF:
        return GLOBAL_GGUF[base]
    parts = base.split(".")
    if len(parts) == 3 and parts[0] == "blk" and parts[2] in GGUF_ROLES:
        return f"{int(parts[1])}.{GGUF_ROLES[parts[2]]}"
    return None


def _ref_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    seen = set()
    for n, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        out[n] = p.detach().to(F64).cpu()
    return out


def _transforms(r: torch.Tensor, role: str, cfg: LlamaConfig) -> Dict[str, torch.Tensor]:
    out = {"identity": r}
    if r.ndim == 2:
        out["transpose"] = r.T
        if role.endswith((".q", ".k")):
            nh = cfg.n_heads if role.endswith(".q") else cfg.n_kv_heads
            if r.shape[0] % nh == 0 and (r.shape[0] // nh) % 2 == 0:
                out["rope_permuted"] = permute_rows(r, nh)
                out["rope_permuted_transposed"] = permute_rows(r, nh).T
    return out


def _undo(g: torch.Tensor, kind: str, role: str, cfg: LlamaConfig) -> torch.Tensor:
    nh = cfg.n_heads if role.endswith(".q") else cfg.n_kv_heads
    if kind == "identity":
        return g
    if kind == "transpose":
        return g.T.contiguous()
    if kind == "rope_permuted":
        return unpermute_rows(g, nh)
    if kind == "rope_permuted_transposed":
        return unpermute_rows(g.T.contiguous(), nh)
    raise ValueError(kind)


def match_weights(G: Dict[str, Tuple[torch.Tensor, str]], ref: Dict[str, torch.Tensor], cfg_ref: LlamaConfig,
                  arch: str) -> Tuple[List[WeightEdge], List[str], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Match every GGUF tensor (role -> decoded tensor, type) to a reference parameter by content.

    Returns the edges, the unmatched reference names, the reference weights by role, and the GGUF
    weights with every found layout fault undone (by role)."""
    order = list(ref)
    used: set = set()
    edges: List[WeightEdge] = []
    w_ref: Dict[str, torch.Tensor] = {}
    w_fixed: Dict[str, torch.Tensor] = {}
    for role, (g, tname) in G.items():
        gname = next((k for k, v in _ROLE_NAMES.items() if v == role), role)
        expected = "rope_permuted" if (arch == "llama" and role.endswith((".q", ".k"))) else "identity"
        bound = TYPE_BOUND.get(tname, 0.2)
        best = None
        for rn in order:
            if rn in used:
                continue
            r = ref[rn]
            if r.numel() != g.numel():
                continue
            for kind, rt in _transforms(r, role, cfg_ref).items():
                if rt.shape != g.shape:
                    continue
                rel = _rel(g, rt)
                if rel >= VALUES_DIFFER:  # a converter that scales a tensor keeps its direction
                    c = float((g * rt).sum() / max(float((rt * rt).sum()), 1e-300))
                    if c > 0 and _rel(g, c * rt) <= bound:
                        rel = min(rel, VALUES_DIFFER * 0.999)
                # strict improvement keeps the first in reference order on ties (equal norm weights)
                if best is None or rel < best[0] - 1e-12 or (abs(rel - best[0]) <= 1e-12 and kind == expected and best[2] != expected):
                    best = (rel, rn, kind)
        if best is None or best[0] >= VALUES_DIFFER:
            edges.append(WeightEdge(gname, role, tname, best[1] if best else None, best[2] if best else None,
                                    expected, best[0] if best else None, bound, "unmatched"))
            continue
        rel, rn, kind = best
        used.add(rn)
        r = ref[rn]
        if rel <= bound:
            verdict = "ok" if kind == expected else "layout"
            ratio = None
        else:
            verdict = "values"
            rt = _transforms(r, role, cfg_ref)[kind]
            ratio = (g.norm() / rt.norm()).item()
            rel = _rel(g, rt)
        edges.append(WeightEdge(gname, role, tname, rn, kind, expected, rel, bound, verdict, ratio))
        w_ref[role] = r
        w_fixed[role] = _undo(g, kind, role, cfg_ref) if verdict != "values" else r
    unmatched = [n for n in order if n not in used]
    return edges, unmatched, w_ref, w_fixed


_ROLE_NAMES: Dict[str, str] = {}


def _gguf_weights(g: gguf.GGUFFile) -> Dict[str, Tuple[torch.Tensor, str]]:
    out: Dict[str, Tuple[torch.Tensor, str]] = {}
    _ROLE_NAMES.clear()
    for name, ti in g.tensors.items():
        role = _role(name)
        if role is None:
            continue
        if not gguf.decodable(ti.type_name):
            raise EnginePortError("TS651", f"{name} is {ti.type_name}, which Truestar does not decode exactly")
        out[role] = (torch.from_numpy(g.tensor(name)), ti.type_name)
        _ROLE_NAMES[name] = role
    return out


# ----------------------------------------------------------------------------- engine taps


def _find_tap(run, step: R.Step):
    for pat, op in TAP_NAMES[step.name]:
        name = pat.format(i=step.layer)
        for t in run.taps:
            if t.name == name and (op is None or t.op == op) and t.data is not None:
                return t
    return None


def _engine_acts(run, cfg: LlamaConfig, ids: Sequence[int], arch: str):
    T = len(ids)
    acts: Dict[str, torch.Tensor] = {"ids": torch.tensor(list(ids))}
    nodes: Dict[str, Any] = {}
    missing = []
    for st in R.steps(cfg):
        t = _find_tap(run, st)
        if t is None:
            missing.append(st.key)
            continue
        a = torch.from_numpy(t.data).reshape(T, -1)
        if arch == "llama" and st.name in ("q", "q_rope"):
            a = unpermute_cols(a, cfg.n_heads)
        if arch == "llama" and st.name in ("k", "k_rope"):
            a = unpermute_cols(a, cfg.n_kv_heads)
        acts[st.key] = a
        nodes[st.key] = t
    if missing:
        raise EnginePortError("TS648", "engine graph nodes not found for steps " + ", ".join(missing[:8])
                              + (" ..." if len(missing) > 8 else "")
                              + "; this llama.cpp build names its nodes differently")
    return acts, nodes


def _observed(run) -> Dict[str, Any]:
    obs: Dict[str, Any] = {}
    for t in run.taps:
        if t.op == "RMS_NORM":
            obs.setdefault("eps", []).append(t.op_float(0))
        elif t.op == "ROPE":
            obs.setdefault("theta", []).append(t.op_float(5))
            obs.setdefault("rope_scale", []).append(1.0 / t.op_float(6) if t.op_float(6) else None)
            obs.setdefault("rope_dims", []).append(t.op_params[1])
            obs.setdefault("rope_mode", []).append(t.op_params[2])
        elif t.op == "SOFT_MAX" and t.name.startswith("kq_soft_max"):
            obs.setdefault("attn_scale", []).append(t.op_float(0))
    return {k: sorted(set(round(x, 9) if isinstance(x, float) else x for x in v if x is not None))
            for k, v in obs.items()}


# ----------------------------------------------------------------------------- the check


def _f16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).to(torch.float16).to(F64)


QUANT_DOT = {"Q8_0": "Q8_0 blocks", "Q4_0": "Q8_0 blocks", "Q4_K": "Q8_K blocks", "Q5_K": "Q8_K blocks",
             "Q6_K": "Q8_K blocks"}


def _round(x: torch.Tensor, wt: str, path: str) -> torch.Tensor:
    if path.startswith("f32"):
        return x
    return torch.from_numpy(gguf.round_like_dot_type(x.detach().numpy(), wt))


def _paths(st: R.Step, wt: str, kv_type: str):
    """The rounding paths the engine is allowed to take for this step: (name, prepare, attention
    rounding). Only these; anything else is the engine's own error. Which one the CPU backend
    takes for F16 weights and an F16 cache depends on the batch, so each listed path is tried and
    the one taken is reported."""
    if st.kind == "matmul":
        if wt in QUANT_DOT:
            yield (f"activations in {QUANT_DOT[wt]}", lambda s, x: _round(x, wt, "q"), None)
            return
        yield ("f32 activations", None, None)
        if wt in ("F16", "BF16"):
            yield (f"activations rounded to {wt}", lambda s, x: _round(x, wt, "r"), None)
        return
    if st.kind == "attention" and kv_type == "f16":
        for extra in ((), ("q",), ("p",), ("q", "p")):
            names = ("k", "v") + extra
            yield ("f16 " + ", ".join(names), None, {n: _f16 for n in names})
        return
    yield ("f32", None, None)


def engine_port_check(reference: torch.nn.Module, gguf_path: str, ids: Sequence[int], *,
                      config: Optional[LlamaConfig] = None, prompt: Optional[str] = None,
                      kv_type: str = "f32", threads: int = 4, gpu_layers: int = 0,
                      mutate: Optional[Callable] = None, reference_name: str = "reference",
                      candidate_name: Optional[str] = None) -> EnginePortReport:
    from . import llamacpp

    t_all = time.perf_counter()
    rep = EnginePortReport(reference_name, candidate_name or gguf_path)
    ids = [int(i) for i in (ids.reshape(-1).tolist() if isinstance(ids, torch.Tensor) else ids)]

    # --- inputs: the file, the reference config and weights
    t0 = time.perf_counter()
    g = gguf.read(gguf_path)
    arch = g.arch()
    rep.arch = arch
    if arch not in ARCHS:
        raise EnginePortError("TS650", f"GGUF architecture {arch!r} is not replayed (supported: {', '.join(ARCHS)})")
    cfg_g = gguf_config(g)
    if config is None:
        if not hasattr(reference, "config"):
            raise EnginePortError("TS650", "the reference has no .config; pass a LlamaConfig (get_config() in the file)")
        config = LlamaConfig.from_hf(reference.config)
    cfg_r = config
    G = _gguf_weights(g)
    for _, (_, tn) in G.items():
        rep.quant_types[tn] = rep.quant_types.get(tn, 0) + 1
    ref_params = _ref_params(reference)
    edges, unmatched, w_ref, w_fixed = match_weights(G, ref_params, cfg_r, arch)
    rep.weights = edges
    rep.reference_unmatched = unmatched
    if cfg_g.tied and "output" not in w_ref and "embed" in w_ref:
        w_ref["output"] = w_ref["embed"]
        w_fixed["output"] = w_fixed["embed"]
    # GGUF semantics in reference coordinates: q and k rows unpermuted by the architecture's rule
    w_gguf: Dict[str, torch.Tensor] = {}
    for role, (gt, _) in G.items():
        if arch == "llama" and role.endswith(".q"):
            gt = unpermute_rows(gt, cfg_g.n_heads)
        elif arch == "llama" and role.endswith(".k"):
            gt = unpermute_rows(gt, cfg_g.n_kv_heads)
        w_gguf[role] = gt
    if cfg_g.tied:
        w_gguf["output"] = w_gguf["embed"]
    types = {role: tn for role, (_, tn) in G.items()}
    if cfg_g.tied:
        types["output"] = types["embed"]
    rep.seconds["decode_and_match"] = time.perf_counter() - t0

    # --- settings, statically
    for name, a, b in [("rope theta", cfg_r.theta, cfg_g.theta), ("norm eps", cfg_r.eps, cfg_g.eps),
                       ("rope scaling factor", cfg_r.rope_scale, cfg_g.rope_scale),
                       ("head count", cfg_r.n_heads, cfg_g.n_heads), ("kv head count", cfg_r.n_kv_heads, cfg_g.n_kv_heads),
                       ("rope dimension count", cfg_r.head_dim, cfg_g.head_dim), ("embedding length", cfg_r.dim, cfg_g.dim),
                       ("feed forward length", cfg_r.ffn, cfg_g.ffn), ("block count", cfg_r.n_layers, cfg_g.n_layers)]:
        if (isinstance(a, float) and not math.isclose(a, b, rel_tol=1e-6)) or (not isinstance(a, float) and a != b):
            rep.settings.append(SettingFinding(name, a, b))
    structural = {s.name for s in rep.settings} & {"head count", "kv head count", "rope dimension count",
                                                    "embedding length", "feed forward length", "block count"}
    t0 = time.perf_counter()
    # --- the reference replay reproduces the reference model?
    with torch.no_grad():
        import copy
        m64 = copy.deepcopy(reference).to(F64).eval()
        out = m64(torch.tensor([ids]))
        out = getattr(out, "logits", out)
        ref_logits = (out[0] if isinstance(out, (tuple, list)) else out).reshape(len(ids), -1).to(F64)
    missing_roles = [r for r in w_gguf if r not in w_ref]
    if missing_roles or unmatched:
        for r in missing_roles:
            e = next(e for e in edges if e.role == r)
            rep.issues.append(Issue("TS643", e.sentence()))
        for n in unmatched:
            rep.issues.append(Issue("TS643", f"reference weight {n} has no GGUF tensor"))
    if missing_roles:
        rep.status = DIVERGENT
        rep.blame = "weights"
        rep.explanation = ("The GGUF does not hold every reference weight, so the model cannot be replayed: "
                           + "; ".join(i.message for i in rep.issues[:4]) + ".")
        rep.seconds["total"] = time.perf_counter() - t_all
        return rep
    spec_ref = R.forward(cfg_r, w_ref, torch.tensor(ids))
    err = _rel(spec_ref["logits"], ref_logits)
    if not err < REPLAY_EXACT:
        if not err < REPLAY_FLOAT32:
            raise EnginePortError("TS650", f"Truestar's float64 llama replay differs from the reference model by "
                                           f"{err:.2e} (needs < {REPLAY_FLOAT32:g}); the reference is not a llama style "
                                           "decoder Truestar replays (check the config and the rotary convention)")
        # Hugging Face models keep parts in float32 even after .double() (rotary cos and sin)
        rep.engine["reference_note"] = (f"the reference model itself is float32 accurate in places: it differs from "
                                        f"the exact float64 replay by {err:.1e}")

    # --- the engine, tapped and untapped
    run = llamacpp.run(gguf_path, ids, threads=threads, kv_type=kv_type, gpu_layers=gpu_layers, tap=True, mutate=mutate)
    clean = llamacpp.run(gguf_path, ids, threads=threads, kv_type=kv_type, gpu_layers=gpu_layers, tap=False)
    rep.seconds["engine_tapped"] = run.seconds
    rep.seconds["engine_untapped"] = clean.seconds
    rep.engine = {**rep.engine, "version": llamacpp.engine_version(), "device": "cpu" if gpu_layers == 0 else f"gpu ({gpu_layers} layers)",
                  "kv_type": kv_type, "threads": threads, "flash_attention": False, "tokens": len(ids)}
    diff = np.abs(run.logits - clean.logits).max()
    rep.observer = {"max_abs": float(diff), "equal": bool(diff == 0)}
    if mutate is None and float(np.linalg.norm(run.logits - clean.logits)) > 1e-6 * float(np.linalg.norm(clean.logits)):
        rep.issues.append(Issue("TS649", f"tapped and untapped logits differ by {diff:.2e} (max abs)"))
    eng = torch.from_numpy(run.logits if mutate is not None else clean.logits).to(F64)

    # --- settings the engine actually used
    obs = _observed(run)
    rep.observed = obs
    for key, label, gval, rval in [("eps", "norm eps", cfg_g.eps, cfg_r.eps), ("theta", "rope theta", cfg_g.theta, cfg_r.theta),
                                   ("rope_scale", "rope scaling factor", cfg_g.rope_scale, cfg_r.rope_scale),
                                   ("rope_dims", "rope dimension count", cfg_g.head_dim, cfg_r.head_dim),
                                   ("attn_scale", "attention scale", 1.0 / math.sqrt(cfg_g.head_dim),
                                    1.0 / math.sqrt(cfg_r.head_dim))]:
        for v in obs.get(key, []):
            if not math.isclose(float(v), float(gval), rel_tol=1e-5):
                rep.settings.append(SettingFinding(label, rval, gval, v, "engine"))

    # --- tokenizer
    if prompt is not None:
        eids = llamacpp.tokenize(gguf_path, prompt)
        tok = {"equal": eids == ids, "reference": ids[:32], "engine": eids[:32]}
        if eids != ids:
            k = next((i for i, (a, b) in enumerate(zip(eids, ids)) if a != b), min(len(eids), len(ids)))
            bos = g.metadata.get("tokenizer.ggml.bos_token_id")
            why = ""
            if eids[1:] == ids and eids[:1] == [bos]:
                why = "the engine adds a BOS token the reference does not (tokenizer.ggml.add_bos_token)"
            elif ids[1:] == eids and ids[:1] == [bos]:
                why = "the reference starts with BOS and the engine does not add one (tokenizer.ggml.add_bos_token)"
            else:
                why = f"first difference at position {k}"
            tok["message"] = f"engine ids differ from the reference ids for the prompt: {why}"
            rep.issues.append(Issue("TS646", tok["message"]))
        rep.tokenizer = tok

    # --- verdict (a): the engine on its own inputs
    if structural:
        rep.status = DIVERGENT
        rep.blame = "settings"
        for s in rep.settings:
            rep.issues.append(Issue("TS641", s.sentence()))
        rep.explanation = ("The GGUF describes another architecture than the reference ("
                           + "; ".join(s.sentence() for s in rep.settings if s.name in structural)
                           + "), so no step can be replayed.")
        rep.seconds["total"] = time.perf_counter() - t_all
        return rep
    acts, nodes = _engine_acts(run, cfg_g, ids, arch)
    pos = torch.arange(len(ids))

    chosen: Dict[str, List[str]] = {}
    for st in R.steps(cfg_g):
        wt = types.get(R.weight_key(st) or "", "")
        best = None
        for path, prep, rnd in _paths(st, wt, kv_type):
            y64 = R.run_step(st, cfg_g, w_gguf, acts, pos, prep, F64, rnd)
            p32 = (lambda s, x, _p=prep: _p(s, x).float()) if prep else None
            r32 = {k: (lambda x, _f=f: _f(x).float()) for k, f in rnd.items()} if rnd else None
            y32 = R.run_step(st, cfg_g, w_gguf, acts, pos, p32, torch.float32, r32)
            thr = max(ENGINE_FACTOR * _rel(y32, y64), ENGINE_FLOOR)
            own = _rel(acts[st.key], y64)
            if best is None or own / thr < best[0] / best[1]:
                best = (own, thr, path)
        node = nodes[st.key]
        rep.engine_steps.append(StepResult(st.key, st.layer, st.name, node.op, node.name, best[0], best[1], wt, best[2]))
        if st.kind == "matmul":
            chosen.setdefault(wt, []).append(best[2])
    # the activation path the engine took most often per weight type drives the float64 replays
    mode = {wt: max(set(v), key=v.count) for wt, v in chosen.items()}
    rep.engine["paths"] = {wt: m for wt, m in mode.items()}
    if kv_type == "f16":
        rep.engine["paths"]["attention"] = max(set(s.path for s in rep.engine_steps if s.step == "attn"),
                                               key=[s.path for s in rep.engine_steps if s.step == "attn"].count)

    def prepare(step, x):
        wt = types[R.weight_key(step)]
        return _round(x, wt, mode.get(wt, ""))

    # --- verdict (b) and (c): float64 replays only
    attn_path = rep.engine["paths"].get("attention", "")
    rounding = {n: _f16 for n in attn_path[4:].split(", ")} if attn_path.startswith("f16 ") else None
    spec_gguf = R.forward(cfg_g, w_gguf, torch.tensor(ids), prepare, rounding)
    spec_fixed = R.forward(cfg_r, w_fixed, torch.tensor(ids), prepare, rounding)
    for st in R.steps(cfg_r):
        a = R.run_step(st, cfg_g, w_gguf, spec_fixed, pos, prepare, F64, rounding)
        b = spec_fixed[st.key]
        rep.conversion_steps.append(StepResult(st.key, st.layer, st.name, rel_err=_rel(a, b), threshold=CONVERSION_FLOOR,
                                               weight_type=types.get(R.weight_key(st) or "", "")))
    rep.seconds["replay"] = time.perf_counter() - t0
    rep.decomposition = {"total": _rel(eng, ref_logits), "engine": _rel(eng, spec_gguf["logits"]),
                         "conversion": _rel(spec_gguf["logits"], spec_fixed["logits"]),
                         "quantization": _rel(spec_fixed["logits"], ref_logits),
                         "reference_replay": err}
    rep.decisions_total = decisions(eng, ref_logits)
    rep.decisions_engine = decisions(eng, spec_gguf["logits"])
    rep.decisions_quant = decisions(spec_fixed["logits"], ref_logits)

    # --- codes and blame
    for s in rep.settings:
        rep.issues.append(Issue("TS641", s.sentence()))
    for w in edges:
        if w.verdict in ("layout", "values"):
            rep.issues.append(Issue("TS642", w.sentence()))
    fe = rep.first_engine()
    fc = rep.first_conversion()
    engine_settings = [s for s in rep.settings if s.where == "engine"]
    if fe is not None:
        msg = (f"engine node {fe.node} ({fe.op}) at step {fe.key}, {STEP_WORDS.get(fe.step, fe.step)}: own error "
               f"{fe.rel_err:.2e} on the engine's own inputs, threshold {fe.threshold:.1e}")
        if engine_settings:
            msg += "; " + "; ".join(s.sentence() for s in engine_settings)
        rep.issues.append(Issue("TS640", msg))
    q = rep.decisions_quant
    if q is not None and rep.quant_types.keys() - {"F32"}:
        qmsg = (f"quantization ({', '.join(sorted(rep.quant_types))}): {q.short()}, logits {rep.decomposition['quantization']:.2e} "
                "relative, from float64 replays only")
        rep.issues.append(Issue("TS645" if q.flips_beyond_ties else "TS644", qmsg))
    errors = [i for i in rep.issues if severity(i.code) == ERROR]
    # End to end, activation rounding is discontinuous: a value on a rounding boundary rounds either
    # way after float32 noise upstream, so the engine may drift from the replay by as much as the
    # allowed rounding itself moves the logits. Only a drift beyond that is unexplained.
    spec_exact = R.forward(cfg_g, w_gguf, torch.tensor(ids))
    allowed = max(1e-3, 2.0 * _rel(spec_gguf["logits"], spec_exact["logits"]))
    rep.decomposition["rounding_effect"] = _rel(spec_gguf["logits"], spec_exact["logits"])
    if not errors and rep.decomposition["engine"] > allowed:
        rep.issues.append(Issue("TS657", f"engine logits differ from the GGUF replay by {rep.decomposition['engine']:.2e}, "
                                         f"more than the allowed rounding explains ({allowed:.1e}), but no step is over "
                                         "its threshold"))
        errors = [rep.issues[-1]]
    rep.status = DIVERGENT if errors else OK
    rep.blame, rep.explanation = _explain(rep, fe, fc)
    rep.seconds["total"] = time.perf_counter() - t_all
    return rep


def _explain(rep: EnginePortReport, fe: Optional[StepResult], fc: Optional[StepResult]) -> Tuple[Optional[str], str]:
    parts = []
    blame = None
    conv_settings = [s for s in rep.settings if s.where == "conversion"]
    eng_settings = [s for s in rep.settings if s.where == "engine"]
    bad_w = [w for w in rep.weights if w.verdict in ("layout", "values")]
    if fe is not None:
        blame = "engine"
        s = (f"ENGINE: node {fe.node} ({fe.op}), the {STEP_WORDS.get(fe.step, fe.step)} of "
             + (f"layer {fe.layer}" if fe.layer >= 0 else "the head")
             + f", computes differently from the float64 replay of that step on the engine's own inputs "
               f"(own error {fe.rel_err:.2e}, threshold {fe.threshold:.1e})")
        if eng_settings:
            s += "; " + "; ".join(x.sentence() for x in eng_settings)
            blame = "engine settings"
        parts.append(s + ".")
    if conv_settings or bad_w:
        blame = blame or ("settings" if conv_settings else "weights")
        loc = (f" First step affected: {fc.key} ({STEP_WORDS.get(fc.step, fc.step)}), relative change "
               f"{fc.rel_err:.2e} in the float64 replay." if fc else "")
        parts.append("CONVERSION: " + "; ".join([x.sentence() for x in conv_settings] + [w.sentence() for w in bad_w])
                     + "." + loc)
    tok = rep.tokenizer
    if tok and not tok.get("equal", True):
        blame = blame or "tokenizer"
        parts.append("TOKENIZER: " + tok["message"] + ".")
    if not parts:
        q = rep.decisions_quant
        if rep.quant_types.keys() - {"F32"}:
            parts.append(f"Same model: the engine computes the GGUF within float32 rounding at every step; the "
                         f"only difference from the reference is quantization ({q.short()}).")
            blame = "quantization" if q and q.flips_beyond_ties else None
        else:
            parts.append("Same model: the engine computes the GGUF within float32 rounding at every step and the "
                         "GGUF holds the reference's weights and settings.")
    return blame, " ".join(parts)


# ----------------------------------------------------------------------------- agent JSON


def report_to_agent(rep: EnginePortReport, reference_file: Optional[str], gguf_file: str) -> Dict[str, Any]:
    import os

    from ..agent import _envelope, exit_code_for

    doc = _envelope(rep.status, exit_code_for(rep.status))
    doc["command"] = "port"
    doc["reference"] = {"name": rep.reference_name, "file": os.path.abspath(reference_file) if reference_file else None}
    doc["candidate"] = {"name": rep.candidate_name, "file": os.path.abspath(gguf_file), "kind": "gguf"}
    doc["precision"] = "engine float32 vs float64 replay"
    doc.update(rep.as_dict())
    acts = []
    fe = rep.first_engine()
    if fe is not None:
        acts.append({"kind": "report_engine_op", "command": None, "node": fe.node, "op": fe.op,
                     "description": f"The engine's {fe.op} node {fe.node} is wrong on its own inputs; reproduce it "
                                    "with the engine's per op tests for this op and shape and report it upstream."})
    for s in rep.settings:
        acts.append({"kind": "fix_setting", "command": None, "setting": s.name,
                     "description": ("Rewrite the GGUF metadata: " if s.where == "conversion" else
                                     "The engine ignored the file's value: ") + s.sentence() + "."})
    for w in rep.weights:
        if w.verdict in ("layout", "values", "unmatched"):
            acts.append({"kind": "fix_converter", "command": None, "tensor": w.gguf_name,
                         "description": "Fix the converter: " + w.sentence() + "."})
    if rep.tokenizer and not rep.tokenizer.get("equal", True):
        acts.append({"kind": "fix_tokenizer", "command": None, "description": rep.tokenizer["message"]})
    if acts and reference_file:
        acts.append({"kind": "rerun", "description": "Rerun after a fix; exit 0 means the engine runs the same model.",
                     "command": f'truestar port "{os.path.abspath(reference_file)}" "gguf:{os.path.abspath(gguf_file)}" --format agent'})
    doc["next_actions"] = acts
    return doc
