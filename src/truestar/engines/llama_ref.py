"""Float64 reference of a Llama style decoder, written step by step so each step can be replayed alone.

Every function takes the inputs of one step and returns its output, in the reference layout: rows
are tokens, a projection's output is `[T, out]`, rotary positions rotate the two halves of each head
(the Hugging Face convention). The engine check feeds each step the engine's own input for that
step, so a step's error is its own and never inherited from an earlier one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

F64 = torch.float64


@dataclass
class LlamaConfig:
    vocab: int
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn: int
    eps: float = 1e-5
    theta: float = 10000.0
    rope_scale: float = 1.0  # linear position scaling factor (positions are divided by it)
    head_dim: Optional[int] = None
    tied: bool = False
    context: int = 2048

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_hf(cfg: Any) -> "LlamaConfig":
        """From a Hugging Face LlamaConfig like object (attribute names of that convention)."""
        scaling = getattr(cfg, "rope_scaling", None) or {}
        scale = 1.0
        if scaling:
            kind = scaling.get("rope_type", scaling.get("type"))
            if kind not in (None, "default", "linear"):
                raise ValueError(f"rope scaling {kind!r} is not replayed (default and linear are)")
            scale = float(scaling.get("factor", 1.0)) if kind == "linear" else 1.0
        theta = getattr(cfg, "rope_theta", None)
        if theta is None:
            theta = (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
        return LlamaConfig(vocab=cfg.vocab_size, dim=cfg.hidden_size, n_layers=cfg.num_hidden_layers,
                           n_heads=cfg.num_attention_heads,
                           n_kv_heads=getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads,
                           ffn=cfg.intermediate_size, eps=float(cfg.rms_norm_eps), theta=float(theta),
                           rope_scale=scale, head_dim=getattr(cfg, "head_dim", None),
                           tied=bool(getattr(cfg, "tie_word_embeddings", False)),
                           context=int(getattr(cfg, "max_position_embeddings", 2048)))


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float, dt=F64) -> torch.Tensor:
    x = x.to(dt)
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w.to(dt)


def linear(x: torch.Tensor, w: torch.Tensor, dt=F64) -> torch.Tensor:
    return x.to(dt) @ w.to(dt).T


def rope(x: torch.Tensor, pos: torch.Tensor, n_heads: int, head_dim: int, theta: float,
         scale: float = 1.0, dt=F64) -> torch.Tensor:
    """x: [T, n_heads * head_dim]; rotate the two halves of each head."""
    T = x.shape[0]
    h = x.to(dt).reshape(T, n_heads, head_dim)
    adt = dt
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=adt) / head_dim))
    ang = (pos.to(adt) / scale)[:, None] * inv[None, :]
    cos = torch.cos(ang).repeat(1, 2)[:, None, :].to(dt)
    sin = torch.sin(ang).repeat(1, 2)[:, None, :].to(dt)
    half = head_dim // 2
    rot = torch.cat([-h[..., half:], h[..., :half]], dim=-1)
    return (h * cos + rot * sin).reshape(T, n_heads * head_dim)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_heads: int, n_kv: int,
              head_dim: int, dt=F64, rounding: Optional[Dict[str, Callable]] = None) -> torch.Tensor:
    """Causal attention over T tokens; q [T, H*D], k and v [T, Hkv*D]; returns [T, H*D]."""
    T = q.shape[0]
    r = rounding or {}  # an f16 cache rounds k and v; some paths also round q and the probabilities
    q, k, v = (r[n](t) if n in r else t for n, t in (("q", q), ("k", k), ("v", v)))
    qh = q.to(dt).reshape(T, n_heads, head_dim).transpose(0, 1)
    rep = n_heads // n_kv
    kh = k.to(dt).reshape(T, n_kv, head_dim).transpose(0, 1).repeat_interleave(rep, dim=0)
    vh = v.to(dt).reshape(T, n_kv, head_dim).transpose(0, 1).repeat_interleave(rep, dim=0)
    s = qh @ kh.transpose(1, 2) / math.sqrt(head_dim)
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    if "p" in r:
        p = r["p"](p)
    return (p @ vh).transpose(0, 1).reshape(T, n_heads * head_dim)


def silu_mul(g: torch.Tensor, u: torch.Tensor, dt=F64) -> torch.Tensor:
    g = g.to(dt)
    return g * torch.sigmoid(g) * u.to(dt)


# Weight roles (Truestar's own names; the GGUF and the reference map onto these by content).
LAYER_ROLES = ("attn_norm", "q", "k", "v", "o", "ffn_norm", "gate", "up", "down")
GLOBAL_ROLES = ("embed", "norm", "output")


@dataclass
class Step:
    """One replayable step of layer `layer` (-1 for the embedding and the head)."""
    name: str           # e.g. "q_rope"
    layer: int
    inputs: List[str]   # activation keys this step reads
    role: Optional[str] = None  # weight role for projections and norms
    kind: str = ""      # "norm", "matmul", "rope", "attention", "add", "act", "embed"

    @property
    def key(self) -> str:
        return f"{self.layer}.{self.name}" if self.layer >= 0 else self.name


def steps(cfg: LlamaConfig) -> List[Step]:
    out = [Step("embd", -1, ["ids"], "embed", "embed")]
    for i in range(cfg.n_layers):
        src = "embd" if i == 0 else f"{i - 1}.out"
        P = lambda n: f"{i}.{n}"  # noqa: E731
        out += [
            Step("attn_norm", i, [src], "attn_norm", "norm"),
            Step("q", i, [P("attn_norm")], "q", "matmul"),
            Step("k", i, [P("attn_norm")], "k", "matmul"),
            Step("v", i, [P("attn_norm")], "v", "matmul"),
            Step("q_rope", i, [P("q")], None, "rope"),
            Step("k_rope", i, [P("k")], None, "rope"),
            Step("attn", i, [P("q_rope"), P("k_rope"), P("v")], None, "attention"),
            Step("attn_out", i, [P("attn")], "o", "matmul"),
            Step("ffn_inp", i, [src, P("attn_out")], None, "add"),
            Step("ffn_norm", i, [P("ffn_inp")], "ffn_norm", "norm"),
            Step("gate", i, [P("ffn_norm")], "gate", "matmul"),
            Step("up", i, [P("ffn_norm")], "up", "matmul"),
            Step("act", i, [P("gate"), P("up")], None, "act"),
            Step("down", i, [P("act")], "down", "matmul"),
            Step("out", i, [P("ffn_inp"), P("down")], None, "add"),
        ]
    last = f"{cfg.n_layers - 1}.out"
    out += [Step("result_norm", -1, [last], "norm", "norm"),
            Step("logits", -1, ["result_norm"], "output", "matmul")]
    return out


def weight_key(step: Step) -> Optional[str]:
    if step.role is None:
        return None
    return step.role if step.layer < 0 else f"{step.layer}.{step.role}"


def run_step(step: Step, cfg: LlamaConfig, W: Dict[str, torch.Tensor], acts: Dict[str, torch.Tensor],
             pos: torch.Tensor, prepare: Optional[Callable[[Step, torch.Tensor], torch.Tensor]] = None,
             dt=F64, rounding: Optional[Dict[str, Callable]] = None) -> torch.Tensor:
    """Run one step on `acts` (its inputs). `prepare(step, x)` rounds a matmul input the way the
    engine is allowed to (for example to Q8_0 blocks) before the float64 product."""
    a = [acts[k] for k in step.inputs]
    wk = weight_key(step)
    D = cfg.head_dim
    if step.kind == "embed":
        return W["embed"].to(dt)[a[0].long()]
    if step.kind == "norm":
        return rms_norm(a[0], W[wk], cfg.eps, dt)
    if step.kind == "matmul":
        x = prepare(step, a[0]) if prepare is not None else a[0]
        return linear(x, W[wk], dt)
    if step.kind == "rope":
        nh = cfg.n_heads if step.name == "q_rope" else cfg.n_kv_heads
        return rope(a[0], pos, nh, D, cfg.theta, cfg.rope_scale, dt)
    if step.kind == "attention":
        return attention(a[0], a[1], a[2], cfg.n_heads, cfg.n_kv_heads, D, dt, rounding)
    if step.kind == "add":
        return a[0].to(dt) + a[1].to(dt)
    if step.kind == "act":
        return silu_mul(a[0], a[1], dt)
    raise ValueError(f"unknown step kind {step.kind}")


def forward(cfg: LlamaConfig, W: Dict[str, torch.Tensor], ids: torch.Tensor,
            prepare: Optional[Callable] = None, rounding: Optional[Dict[str, Callable]] = None,
            ) -> Dict[str, torch.Tensor]:
    """Every activation of a teacher forced pass over `ids` (1 D), float64."""
    ids = ids.reshape(-1)
    pos = torch.arange(ids.shape[0])
    acts: Dict[str, torch.Tensor] = {"ids": ids}
    for st in steps(cfg):
        acts[st.key] = run_step(st, cfg, W, acts, pos, prepare, F64, rounding)
    return acts


def output_weight(W: Dict[str, torch.Tensor], cfg: LlamaConfig) -> torch.Tensor:
    return W["output"] if "output" in W else W["embed"]
