"""Rotations folded into the weights of a Llama style model, proven equivalent in float64 before
anything is quantized.

Placement (row vectors, ``nn.Linear`` computes x W^T; QuaRot and SpinQuant name them R1 to R4):

* norm fold: each RMSNorm weight g moves into the linears that read the norm (W diag(g)), and the
  norm's weight becomes 1. Needed for R1: RMSNorm(x Q) = RMSNorm(x) Q only without a per channel
  scale, because an orthogonal Q keeps the norm of x.
* R1 (offline, residual stream, d x d): embedding E Q; q, k, v, gate, up W Q; o and down Q^T W;
  lm head W Q (a tied lm head is untied first).
* R2 (offline, per attention head, hd x hd): value rows of each KV head R^T W_v, the matching
  input columns of o_proj W_o R (every query head of a group uses its KV head's R).
* R4 (online, down_proj input, n x n): down W H, and the input of down_proj is rotated at run
  time (``OnlineRotate``) by the Walsh transform.
* R3 (queries and keys after RoPE) is online and lives in the KV cache (truestar.kvquant,
  ``rot=True``), not here: RoPE does not commute with a rotation of the head dim.

``prove`` runs the original and the rotated model in float64 on the same tokens and compares, in
execution order, every point where the two must agree once the rotation is undone (each layer's
input, q, k, v, the attention output, o, gate, up, the down input and output, the logits). The
check passes when every point is within ``TOL`` (relative to the point's largest magnitude) of
the original. A wrong placement fails it and is named: the first failing point and the steps that
sit between it and the last point that held. Only then is the rotated model handed to the
quantizer (``RotationProofError`` TS1020 otherwise; a matrix that is not orthogonal is TS1021).

Steps that are wrong on purpose (``Plan.extra``, for the tests and the study): ``qk_pre_rope``
folds a head rotation into W_q and W_k (crosses RoPE); ``mlp_across_act`` rotates the gate and
up outputs and folds the inverse into down (crosses SiLU and the product).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch import nn

from .hadamard import RotationError, factor, orthogonal_hadamard, walsh

F64 = torch.float64
TOL = 1e-9          # relative, per checked point; a correct fold measures 1e-15 to 1e-13
ORTHO_TOL = 1e-12   # max |Q^T Q - I| accepted as orthogonal (float64)

Rot = Union[None, str, torch.Tensor]


class RotationProofError(AssertionError):
    """TS1020: the rotated model does not compute the original model in float64."""
    code = "TS1020"

    def __init__(self, report: "ProofReport"):
        super().__init__(report.message())
        self.report = report


class NotOrthogonalError(ValueError):
    """TS1021: a rotation matrix is not orthogonal to float64 accuracy."""
    code = "TS1021"


# ----------------------------------------------------------------------------- the model layout

@dataclass
class Arch:
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    inter: int
    layers: int
    tied: bool


def arch_of(model: nn.Module) -> Arch:
    """The layout of a Hugging Face Llama or Qwen2 causal LM; anything else is refused (TS1022)."""
    cfg = getattr(model, "config", None)
    mt = getattr(cfg, "model_type", None)
    if mt not in ("llama", "qwen2") or not hasattr(model, "model") or not hasattr(model, "lm_head"):
        raise RotationError(f"[TS1022] rotations are folded for Llama and Qwen2 causal LMs "
                            f"(model.layers, RMSNorm, SwiGLU), got {type(model).__name__}")
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return Arch(cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, hd,
                cfg.intermediate_size, cfg.num_hidden_layers,
                model.lm_head.weight is model.model.embed_tokens.weight)


def _hadamard_or(m: Rot, n: int, what: str) -> Optional[torch.Tensor]:
    if m is None:
        return None
    if isinstance(m, str):
        if m != "hadamard":
            raise RotationError(f"[TS1022] {what}: a rotation is 'hadamard', a matrix or None, got {m!r}")
        factor(n)
        return orthogonal_hadamard(n)
    m = m.detach().to(F64).cpu()
    if m.shape != (n, n):
        raise RotationError(f"[TS1022] {what}: the rotation must be {n} x {n}, got {tuple(m.shape)}")
    return m


def check_orthogonal(m: torch.Tensor, what: str) -> float:
    e = float((m.T @ m - torch.eye(m.shape[0], dtype=F64)).abs().max())
    if not e <= ORTHO_TOL:
        raise NotOrthogonalError(f"[TS1021] {what} is not orthogonal: max |Q^T Q - I| = {e:.3e} "
                                 f"(accepted {ORTHO_TOL:.0e})")
    return e


@dataclass
class Plan:
    """What to fold. Each rotation is ``"hadamard"`` (deterministic, the default), an orthogonal
    float matrix (for example a learned one, truestar.rotate.learned) or None."""
    r1: Rot = "hadamard"
    r2: Rot = "hadamard"
    r4: Rot = "hadamard"
    fold_norms: bool = True
    extra: Sequence[str] = ()

    def matrices(self, a: Arch) -> Dict[str, Optional[torch.Tensor]]:
        m = {"r1": _hadamard_or(self.r1, a.hidden, "R1"),
             "r2": _hadamard_or(self.r2, a.head_dim, "R2"),
             "r4": _hadamard_or(self.r4, a.inter, "R4")}
        for e in self.extra:
            if e not in ("qk_pre_rope", "mlp_across_act"):
                raise RotationError(f"[TS1022] unknown extra step {e!r}; known: qk_pre_rope, mlp_across_act")
        if "qk_pre_rope" in self.extra:
            m["qk"] = orthogonal_hadamard(a.head_dim)
        if "mlp_across_act" in self.extra:
            if self.r4 is not None:
                raise RotationError("[TS1022] mlp_across_act and R4 both rotate the down input: pass "
                                    "r4=None with mlp_across_act, or drop mlp_across_act from extra")
            m["mlp"] = orthogonal_hadamard(a.inter)
        for k, v in m.items():
            if v is not None:
                check_orthogonal(v, k.upper())
        return m


# ----------------------------------------------------------------------------- folding

def _per_head_rows(w: torch.Tensor, r: torch.Tensor, heads: int) -> torch.Tensor:
    """Output rows of each head block multiplied: y_h -> y_h R, so W_h -> R^T W_h."""
    hd = r.shape[0]
    return torch.einsum("ji,hjk->hik", r.to(w.dtype), w.reshape(heads, hd, -1)).reshape(w.shape)


def _per_head_cols(w: torch.Tensor, r: torch.Tensor, heads: int) -> torch.Tensor:
    """Input columns of each head block: x_h -> x_h R, so W[:, h] -> W[:, h] R."""
    hd = r.shape[0]
    return torch.einsum("ohj,jk->ohk", w.reshape(w.shape[0], heads, hd), r.to(w.dtype)).reshape(w.shape)


def fold_state(sd: Dict[str, torch.Tensor], a: Arch, m: Dict[str, Optional[torch.Tensor]],
               fold_norms: bool = True) -> Dict[str, torch.Tensor]:
    """The rotated weights, from a state dict of the original model (keys as in the Hugging Face
    model; an untied ``lm_head.weight`` is always returned). Differentiable in the matrices, so a
    learned rotation can be trained through it. Down weights are keyed ``...mlp.down_proj.weight``
    here; ``apply`` moves them under the online wrapper."""
    out = dict(sd)
    L = "model.layers.{}."
    dt = sd["model.embed_tokens.weight"].dtype
    cast = lambda t: None if t is None else t.to(dtype=dt, device=sd["model.embed_tokens.weight"].device)  # noqa: E731
    r1, r2, r4 = cast(m.get("r1")), cast(m.get("r2")), cast(m.get("r4"))
    qk, mlp = cast(m.get("qk")), cast(m.get("mlp"))
    out["lm_head.weight"] = sd.get("lm_head.weight", sd["model.embed_tokens.weight"])
    if fold_norms:
        for i in range(a.layers):
            p = L.format(i)
            g = sd[p + "input_layernorm.weight"]
            for n in ("q_proj", "k_proj", "v_proj"):
                out[p + f"self_attn.{n}.weight"] = sd[p + f"self_attn.{n}.weight"] * g[None, :]
            out[p + "input_layernorm.weight"] = torch.ones_like(g)
            g = sd[p + "post_attention_layernorm.weight"]
            for n in ("gate_proj", "up_proj"):
                out[p + f"mlp.{n}.weight"] = sd[p + f"mlp.{n}.weight"] * g[None, :]
            out[p + "post_attention_layernorm.weight"] = torch.ones_like(g)
        g = sd["model.norm.weight"]
        out["lm_head.weight"] = out["lm_head.weight"] * g[None, :]
        out["model.norm.weight"] = torch.ones_like(g)
    if r1 is not None:
        out["model.embed_tokens.weight"] = out["model.embed_tokens.weight"] @ r1
        out["lm_head.weight"] = out["lm_head.weight"] @ r1
        for i in range(a.layers):
            p = L.format(i)
            for n in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj",
                      "mlp.up_proj"):
                out[p + n + ".weight"] = out[p + n + ".weight"] @ r1
            for n in ("self_attn.o_proj", "mlp.down_proj"):
                out[p + n + ".weight"] = r1.T @ out[p + n + ".weight"]
                if p + n + ".bias" in out:
                    out[p + n + ".bias"] = out[p + n + ".bias"] @ r1
    for i in range(a.layers):
        p = L.format(i)
        if r2 is not None:
            out[p + "self_attn.v_proj.weight"] = _per_head_rows(out[p + "self_attn.v_proj.weight"], r2, a.kv_heads)
            if p + "self_attn.v_proj.bias" in out:
                out[p + "self_attn.v_proj.bias"] = (out[p + "self_attn.v_proj.bias"].reshape(a.kv_heads, -1) @ r2).reshape(-1)
            out[p + "self_attn.o_proj.weight"] = _per_head_cols(out[p + "self_attn.o_proj.weight"], r2, a.heads)
        if qk is not None:        # wrong on purpose: crosses RoPE
            out[p + "self_attn.q_proj.weight"] = _per_head_rows(out[p + "self_attn.q_proj.weight"], qk, a.heads)
            out[p + "self_attn.k_proj.weight"] = _per_head_rows(out[p + "self_attn.k_proj.weight"], qk, a.kv_heads)
        if mlp is not None:       # wrong on purpose: crosses SiLU and the product
            for n in ("gate_proj", "up_proj"):
                out[p + f"mlp.{n}.weight"] = mlp.T @ out[p + f"mlp.{n}.weight"]
            out[p + "mlp.down_proj.weight"] = out[p + "mlp.down_proj.weight"] @ mlp
        if r4 is not None:
            out[p + "mlp.down_proj.weight"] = out[p + "mlp.down_proj.weight"] @ r4
    return out


class OnlineRotate(nn.Module):
    """y = linear(x R): the run time rotation of a linear layer's input (R4). For a Hadamard R
    the transform is the Walsh transform (float64, O(n log n) on the power of two factor);
    otherwise a matmul by the stored matrix."""

    def __init__(self, linear: nn.Linear, r: Optional[torch.Tensor], hadamard: bool):
        super().__init__()
        self.linear = linear
        self.hadamard = hadamard
        self.register_buffer("r", None if hadamard else r.clone())

    @property
    def weight(self):
        return self.linear.weight

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        if self.hadamard:
            return walsh(x, dtype=x.dtype if x.dtype == torch.float32 else F64).to(x.dtype)
        return x @ self.r.to(x.dtype)

    def forward(self, x):
        return self.linear(self.rotate(x))


@dataclass
class Rotated:
    """A rotated model and how it was made (needed by ``prove``)."""
    model: nn.Module
    arch: Arch
    mats: Dict[str, Optional[torch.Tensor]]
    plan: Plan


@torch.no_grad()
def apply(model: nn.Module, plan: Optional[Plan] = None, dtype=F64, device=None) -> Rotated:
    """A rotated copy of ``model``, folded in float64 and returned in ``dtype`` (float64 by
    default: the program ``prove`` checks; casting it to float16 afterwards is a rounding of the
    weights like any other, outside the proof). Nothing is proven yet: call ``prove`` before
    quantizing it. ``device`` (default: the model's) is where the folding and the copy live."""
    plan = plan or Plan()
    a = arch_of(model)
    m = plan.matrices(a)
    dev = device or next(model.parameters()).device
    r = copy.deepcopy(model).to(device=dev, dtype=F64)
    if a.tied:
        r.lm_head.weight = nn.Parameter(r.model.embed_tokens.weight.detach().clone())
        r.config.tie_word_embeddings = False
    sd = {k: v for k, v in r.state_dict().items()}
    new = fold_state(sd, a, m, plan.fold_norms)
    for i in range(a.layers):
        mlp = r.model.layers[i].mlp
        if m["r4"] is not None:
            mlp.down_proj = OnlineRotate(mlp.down_proj, m["r4"], isinstance(plan.r4, str))
            k = f"model.layers.{i}.mlp.down_proj"
            new[k + ".linear.weight"] = new.pop(k + ".weight")
            if k + ".bias" in new:
                new[k + ".linear.bias"] = new.pop(k + ".bias")
            if not isinstance(plan.r4, str):
                new[k + ".r"] = m["r4"].to(new[k + ".linear.weight"])
    missing, unexpected = r.load_state_dict(new, strict=False, assign=True)
    bad = [k for k in missing if not k.endswith("rotary_emb.inv_freq")]
    if bad or unexpected:
        raise RotationError(f"[TS1022] folded weights do not match the model: missing {bad[:4]}, "
                            f"unexpected {unexpected[:4]}")
    r._truestar_rotated = True     # read by qerror: only the formats a rotation helps take it (TS1024)
    return Rotated(r.to(dtype), a, m, plan)


# ----------------------------------------------------------------------------- the proof

class _Norm64(nn.Module):
    """RMSNorm computed in the input's dtype (Hugging Face's casts to float32, which would make a
    float64 comparison measure float32 rounding)."""

    def __init__(self, norm: nn.Module):
        super().__init__()
        self.weight = norm.weight
        self.eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def norms64(m: nn.Module) -> List[tuple]:
    """Replace the RMSNorms of m in place by ones that compute in the input's dtype; returns what
    ``restore_norms`` needs to put them back."""
    swapped = []
    for name, mod in list(m.named_modules()):
        if type(mod).__name__.endswith("RMSNorm"):
            parent, _, leaf = name.rpartition(".")
            owner = m.get_submodule(parent) if parent else m
            setattr(owner, leaf, _Norm64(mod))
            swapped.append((owner, leaf, mod))
    return swapped


def restore_norms(swapped: List[tuple]) -> None:
    for owner, leaf, mod in swapped:
        setattr(owner, leaf, mod)


def to_float64(model: nn.Module, device="cpu") -> nn.Module:
    """A float64 copy (on ``device``) whose RMSNorms run in float64."""
    m = copy.deepcopy(model).to(device=device, dtype=F64)
    norms64(m)
    return m


@dataclass
class Point:
    name: str
    rel: float
    ok: bool


@dataclass
class ProofReport:
    points: List[Point]
    tol: float
    orthogonality: Dict[str, float]
    first_bad: Optional[str] = None
    suspects: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.first_bad is None

    @property
    def worst(self) -> float:
        return max(p.rel for p in self.points)

    def message(self) -> str:
        if self.ok:
            return (f"rotated model equals the original in float64 at {len(self.points)} points "
                    f"(worst relative difference {self.worst:.2e}, bound {self.tol:.0e})")
        p = next(x for x in self.points if x.name == self.first_bad)
        return (f"[TS1020] the rotated model differs from the original at {self.first_bad} "
                f"(relative {p.rel:.3e}, bound {self.tol:.0e}); the steps between the last point "
                f"that held and this one: {', '.join(self.suspects) or 'none'}")


def _points(a: Arch, m: Dict[str, Optional[torch.Tensor]], rotated: bool):
    """(name, module path, 'in' or 'out', map from the rotated value to the original basis,
    steps that act right before this point)."""
    ident = lambda t: t  # noqa: E731

    def right(r):            # rotated value is x R: undo with R^T
        return ident if r is None else (lambda t: t @ r.T.to(t))

    def heads(r, n):
        if r is None:
            return ident
        return lambda t: (t.reshape(*t.shape[:-1], n, r.shape[0]) @ r.T.to(t)).reshape(t.shape)

    r1, r2, r4, qk, mlp = (m.get(k) for k in ("r1", "r2", "r4", "qk", "mlp"))
    pts = []
    for i in range(a.layers):
        p = f"model.layers.{i}"
        pts.append((f"{p} input", p, "in", right(r1), ["R1 into the embedding" if i == 0 else
                                                       f"R1 around layer {i - 1}"]))
        pts.append((f"{p}.self_attn.q_proj", p + ".self_attn.q_proj", "out", heads(qk, a.heads),
                    [f"{p}.input_layernorm (norm fold into q, k, v)", "R1 into q_proj"] +
                    (["head rotation folded into q_proj"] if qk is not None else [])))
        pts.append((f"{p}.self_attn.k_proj", p + ".self_attn.k_proj", "out", heads(qk, a.kv_heads),
                    ["R1 into k_proj"] + (["head rotation folded into k_proj"] if qk is not None else [])))
        pts.append((f"{p}.self_attn.v_proj", p + ".self_attn.v_proj", "out", heads(r2, a.kv_heads),
                    ["R1 into v_proj"] + (["R2 into v_proj"] if r2 is not None else [])))
        pts.append((f"{p}.self_attn (attention output, o_proj input)", p + ".self_attn.o_proj", "in",
                    heads(r2, a.heads), [f"{p}.self_attn core: RoPE, scores, softmax"] +
                    (["head rotation of q and k folded before RoPE (RoPE does not commute with it)"]
                     if qk is not None else [])))
        pts.append((f"{p}.self_attn.o_proj", p + ".self_attn.o_proj", "out", right(r1),
                    ["R2 into o_proj", "R1 into o_proj"]))
        pts.append((f"{p}.mlp.gate_proj", p + ".mlp.gate_proj", "out", right(mlp),
                    [f"{p}.post_attention_layernorm (norm fold into gate, up)", "R1 into gate_proj"] +
                    (["rotation of the gate output"] if mlp is not None else [])))
        pts.append((f"{p}.mlp.up_proj", p + ".mlp.up_proj", "out", right(mlp),
                    ["R1 into up_proj"] + (["rotation of the up output"] if mlp is not None else [])))
        down = p + ".mlp.down_proj" + (".linear" if rotated and r4 is not None else "")
        pts.append((f"{p}.mlp.down_proj input", down, "in", right(r4 if r4 is not None else mlp),
                    [f"{p}.mlp act_fn(gate) * up"] + (["R4 online rotation"] if r4 is not None else [])))
        pts.append((f"{p}.mlp.down_proj", down, "out", right(r1),
                    (["R4 into down_proj"] if r4 is not None else []) + ["R1 into down_proj"]))
    pts.append(("logits", "lm_head", "out", ident, ["model.norm (norm fold into lm_head)", "R1 into lm_head"]))
    return pts


def _capture(model: nn.Module, pts, ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    caps: Dict[str, torch.Tensor] = {}
    hooks = []
    for name, path, where, _, _ in pts:
        mod = model.get_submodule(path)
        if where == "in":
            def pre(mod, args, kwargs, name=name):
                x = args[0] if args else kwargs.get("hidden_states")
                caps[name] = x.detach().clone()
            hooks.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        else:
            def post(mod, args, out, name=name):
                caps[name] = (out[0] if isinstance(out, tuple) else out).detach().clone()
            hooks.append(mod.register_forward_hook(post))
    try:
        model(input_ids=ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return caps


@torch.no_grad()
def prove(original: nn.Module, rot: Rotated, ids: torch.Tensor, tol: float = TOL,
          raise_on_error: bool = True) -> ProofReport:
    """Run both models in float64 with token ids [B, T] and compare every checked point (see the
    module docstring), on the rotated model's device (a float64 GPU matmul is still float64).
    Raises RotationProofError (TS1020) unless ``raise_on_error`` is False; the report lists every
    point either way."""
    a = rot.arch
    ortho = {k: check_orthogonal(v, k.upper()) for k, v in rot.mats.items() if v is not None}
    dev = next(rot.model.parameters()).device
    o64 = to_float64(original, dev)
    o64.config._attn_implementation = "eager"
    ids = ids.to(dev)
    co = _capture(o64, _points(a, rot.mats, False), ids)
    del o64
    r64 = rot.model if next(rot.model.parameters()).dtype == F64 else to_float64(rot.model, dev)
    swapped = norms64(r64) if r64 is rot.model else []
    impl = r64.config._attn_implementation
    r64.config._attn_implementation = "eager"
    pr = _points(a, rot.mats, True)
    try:
        cr = _capture(r64, pr, ids)
    finally:
        restore_norms(swapped)
        r64.config._attn_implementation = impl
    points, first, suspects, pending = [], None, [], []
    for name, _, _, back, steps in pr:
        pending += steps
        ref = co[name]
        got = back(cr[name])
        rel = float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-300))
        ok = rel <= tol
        points.append(Point(name, rel, ok))
        if ok:
            pending = []
        elif first is None:
            first, suspects = name, list(pending)
    rep = ProofReport(points, tol, ortho, first, suspects)
    if not rep.ok and raise_on_error:
        raise RotationProofError(rep)
    return rep
