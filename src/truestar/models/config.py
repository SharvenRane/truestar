"""Reading a Hugging Face ``config.json`` into the plan of a native decoder (``DecoderConfig``).

Only the fields that change the math are read; each family's rules are spelled out here, and a
setting this reader does not implement is refused (TS1391) instead of being ignored. Nothing here
imports transformers: the folder's ``config.json`` is plain JSON.

Family rules (what transformers computes, read from its modeling files):

* llama, mistral, qwen2, qwen3, phi3: pre norm blocks, RMSNorm ``w * norm(x)`` with the norm in
  float32 (or wider), rotary position embedding in the rotate half layout, gated MLP.
  qwen2 has q, k and v biases; qwen3 normalizes q and k per head (RMSNorm over head_dim) before the
  rotation; phi3 stores q, k, v fused (``qkv_proj``) and gate and up fused (``gate_up_proj``),
  which the loader splits. mistral and phi3 attend in a sliding window when ``sliding_window`` is
  set.
* gemma2, gemma3 (text): RMSNorm ``(1 + w) * norm(x)`` computed in float32, the embedding
  multiplied by sqrt(hidden_size), norms after attention and after the MLP as well as before
  (sandwich), the score scale ``query_pre_attn_scalar ** -0.5``, gelu (tanh) MLP, sliding window
  layers (gemma2: even layers; gemma3: all but every ``sliding_window_pattern``-th, or
  ``layer_types`` when the config lists them). gemma2 caps the attention scores and the logits
  with ``cap * tanh(x / cap)``; gemma3 normalizes q and k per head, and its sliding layers use
  ``rope_local_base_freq`` with plain rotary frequencies.

Rotary frequencies are computed the way transformers computes them (float32 arithmetic, then the
``rope_scaling`` rule: none, "default", "linear" or "llama3"); other scalings (yarn, dynamic,
longrope) are refused with TS1391.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

FAMILIES = ("llama", "mistral", "qwen2", "qwen3", "phi3", "gemma2", "gemma3_text")

CODE_FAMILY = "TS1390"
CODE_FEATURE = "TS1391"


class ModelLoadError(ValueError):
    """A folder that cannot be built natively. ``code`` is the stable TS code, ``problems`` lists
    every problem found."""

    def __init__(self, code: str, summary: str, problems: Sequence[str] = (), suggestion: str = ""):
        self.code = code
        self.summary = summary
        self.problems = list(problems)
        self.suggestion = suggestion
        lines = [f"{code}: {summary}"] + [f"  {p}" for p in self.problems[:50]]
        if len(self.problems) > 50:
            lines.append(f"  ... and {len(self.problems) - 50} more")
        if suggestion:
            lines.append(f"  fix: {suggestion}")
        super().__init__("\n".join(lines))

    def to_dict(self):
        return {"code": self.code, "summary": self.summary, "problems": self.problems, "suggestion": self.suggestion}


@dataclass(frozen=True)
class RopeSpec:
    """One table of rotary frequencies: ``inv_freq`` (float32 values, as transformers holds them)."""

    base: float
    dim: int                      # the rotated part of the head (head_dim * partial_rotary_factor)
    scaling: Optional[Tuple[Tuple[str, Any], ...]] = None   # the rope_scaling dict, as sorted items

    def inv_freq(self):
        """float32 [dim / 2], with transformers' formula and float32 arithmetic."""
        import torch

        inv = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        sc = dict(self.scaling or ())
        kind = sc.get("rope_type", sc.get("type", "default"))
        if kind == "linear":
            inv = inv / float(sc["factor"])
        elif kind == "llama3":
            factor = float(sc["factor"])
            low, high = float(sc["low_freq_factor"]), float(sc["high_freq_factor"])
            old = float(sc["original_max_position_embeddings"])
            low_wl, high_wl = old / low, old / high
            wl = 2 * math.pi / inv
            scaled = torch.where(wl > low_wl, inv / factor, inv)
            smooth = (old / wl - low) / (high - low)
            smoothed = (1 - smooth) * scaled / factor + smooth * scaled
            medium = ~(wl < high_wl) * ~(wl > low_wl)
            inv = torch.where(medium, smoothed, scaled)
        return inv.to(torch.float32)


@dataclass(frozen=True)
class DecoderConfig:
    """Everything the native decoder needs, read from config.json (see the module docstring)."""

    family: str
    vocab: int
    dim: int
    depth: int
    heads: int
    kv_heads: int
    head_dim: int
    hidden: int
    act: str                      # "silu", "gelu", "gelu_tanh"
    eps: float
    norm_kind: str                # "llama": w * norm(x); "gemma": (1 + w) * norm(x), multiplied in float32
    qkv_bias: bool
    o_bias: bool
    mlp_bias: bool
    qk_norm: bool
    sandwich: bool                # gemma: norms after attention and after the MLP too
    scale: float                  # attention score scale
    attn_softcap: float           # 0: none
    final_softcap: float          # 0: none
    embed_scale: str              # "none", "model" (sqrt(dim) in the model dtype), "f32" (rounded to float32 first)
    tie: bool
    rot_dim: int
    rope: RopeSpec
    rope_local: Optional[RopeSpec]
    windows: Tuple[int, ...]      # per layer: 0 is full causal attention, w > 0 a sliding window of w keys
    local_rope: Tuple[bool, ...]  # per layer: use rope_local
    fused_qkv: bool               # phi3 checkpoint layout
    fused_gate_up: bool
    torch_dtype: str
    raw: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)


_ACTS = {"silu": "silu", "swish": "silu", "gelu": "gelu", "gelu_pytorch_tanh": "gelu_tanh", "gelu_tanh": "gelu_tanh",
         "gelu_new": "gelu_tanh"}


def read_config(path: str) -> Dict[str, Any]:
    p = os.path.join(path, "config.json") if os.path.isdir(path) else path
    if not os.path.isfile(p):
        raise ModelLoadError(CODE_FAMILY, f"{path}: no config.json", suggestion="point at a folder written by "
                             "save_pretrained (config.json and the safetensors weights)")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _rope_scaling(raw: Dict[str, Any], problems: List[str], key: str = "rope_scaling"):
    sc = raw.get(key)
    if not sc:
        return None
    kind = sc.get("rope_type", sc.get("type", "default"))
    if kind not in ("default", "linear", "llama3"):
        problems.append(f"{key} type {kind!r} is not implemented natively (implemented: default, linear, llama3)")
        return None
    need = {"linear": ("factor",), "llama3": ("factor", "low_freq_factor", "high_freq_factor",
                                              "original_max_position_embeddings")}.get(kind, ())
    for k in need:
        if k not in sc:
            problems.append(f"{key} type {kind} needs {k!r}")
    if float(sc.get("attention_factor", 1.0) or 1.0) != 1.0:
        problems.append(f"{key} attention_factor {sc['attention_factor']} is not implemented")
    return tuple(sorted((k, v) for k, v in sc.items() if isinstance(v, (int, float, str))))


def _flat_rope(raw: Dict[str, Any]) -> Dict[str, Any]:
    """config.json with transformers 5's ``rope_parameters`` written back in the 4.x keys this module
    reads: ``rope_theta`` and ``rope_scaling`` (the rest of the dict, when its type is not default),
    and for a per layer type dict (Gemma 3) the full attention entry as those and the sliding one as
    ``rope_local_base_freq``. A key the dict already has in the 4.x form is kept, so an explicit
    ``rope_scaling`` is never hidden. A folder written by transformers 4.x has no
    ``rope_parameters`` and is returned unchanged."""
    rp = raw.get("rope_parameters")
    if not isinstance(rp, dict) or not rp:
        return raw
    out = dict(raw)

    def put(params: Dict[str, Any], theta_key: str, scaling_key: Optional[str]):
        if "rope_theta" in params and theta_key not in raw:
            out[theta_key] = params["rope_theta"]
        if "partial_rotary_factor" in params and "partial_rotary_factor" not in raw:
            out["partial_rotary_factor"] = params["partial_rotary_factor"]
        rest = {k: v for k, v in params.items() if k not in ("rope_theta", "partial_rotary_factor")}
        key = scaling_key or "rope_local_scaling"
        if rest.get("rope_type", rest.get("type", "default")) != "default" and not raw.get(key):
            out[key] = rest

    if all(isinstance(v, dict) for v in rp.values()):
        if "full_attention" in rp:
            put(rp["full_attention"], "rope_theta", "rope_scaling")
        if "sliding_attention" in rp:
            put(rp["sliding_attention"], "rope_local_base_freq", None)
    else:
        put(rp, "rope_theta", "rope_scaling")
    return out


def decoder_config(raw: Dict[str, Any]) -> DecoderConfig:
    """The native plan for a config.json dict; TS1390 for another family, TS1391 for a setting
    that is not implemented."""
    fam = raw.get("model_type")
    if fam == "gemma3":
        raise ModelLoadError(CODE_FEATURE, "a multimodal Gemma 3 folder (model_type gemma3): the native decoder "
                             "is the text model only", suggestion="save the text model (Gemma3ForCausalLM, "
                             "model_type gemma3_text) and load that folder")
    if fam not in FAMILIES:
        raise ModelLoadError(CODE_FAMILY, f"model_type {fam!r} has no native Truestar implementation",
                             [f"native families: {', '.join(FAMILIES)}"],
                             "import the model through truestar.ir.import_module instead")
    problems: List[str] = []
    flat = _flat_rope(raw)
    g = flat.get
    if flat.get("rope_local_scaling"):
        problems.append(f"a rope scaling on the sliding layers ({flat['rope_local_scaling']}) is not implemented")
    dim, heads = int(g("hidden_size")), int(g("num_attention_heads"))
    kv = int(g("num_key_value_heads") or heads)
    hd = int(g("head_dim") or dim // heads)
    depth = int(g("num_hidden_layers"))
    gemma = fam in ("gemma2", "gemma3_text")
    act_name = g("hidden_activation") if gemma and g("hidden_activation") else g("hidden_act", "silu")
    act = _ACTS.get(act_name)
    if act is None:
        problems.append(f"activation {act_name!r} is not implemented (implemented: {', '.join(sorted(_ACTS))})")
        act = "silu"
    if heads % kv:
        problems.append(f"num_attention_heads {heads} is not a multiple of num_key_value_heads {kv}")
    prf = float(g("partial_rotary_factor", 1.0) or 1.0)
    rot = int(hd * prf)
    if rot % 2 or rot <= 0:
        problems.append(f"the rotated part of the head ({hd} x {prf}) must be a positive even size")
    scaling = _rope_scaling(flat, problems)
    rope = RopeSpec(float(g("rope_theta", 10000.0)), rot, scaling)
    rope_local = None
    window = int(g("sliding_window") or 0)
    if fam == "qwen2" or fam == "qwen3":
        if g("use_sliding_window"):
            problems.append("use_sliding_window is set: Qwen sliding window attention is not implemented natively")
        window = 0
    if fam == "gemma2":
        if g("layer_types"):
            sliding = [t == "sliding_attention" for t in g("layer_types")]
        else:
            sliding = [i % 2 == 0 for i in range(depth)]
        windows = tuple(window if s else 0 for s in sliding)
        local = tuple(False for _ in range(depth))
    elif fam == "gemma3_text":
        if g("layer_types"):
            sliding = [t == "sliding_attention" for t in g("layer_types")]
        else:
            pat = int(g("sliding_window_pattern", 6))
            sliding = [bool((i + 1) % pat) for i in range(depth)]
        windows = tuple(window if s else 0 for s in sliding)
        local = tuple(sliding)
        rope_local = RopeSpec(float(g("rope_local_base_freq", 10000.0)), rot, None)
    else:
        windows = tuple(window for _ in range(depth))
        local = tuple(False for _ in range(depth))
    if len(windows) != depth:
        problems.append(f"layer_types lists {len(windows)} layers, num_hidden_layers is {depth}")
    if gemma:
        scale = float(g("query_pre_attn_scalar", hd)) ** -0.5
    else:
        scale = hd ** -0.5
    attn_cap = float(g("attn_logit_softcapping") or 0.0) if gemma else 0.0
    final_cap = float(g("final_logit_softcapping") or 0.0) if gemma else 0.0
    if g("mlp_bias") and fam not in ("llama",):
        problems.append("mlp_bias is only read for llama")
    if problems:
        raise ModelLoadError(CODE_FEATURE, f"config.json ({fam}) asks for what the native decoder does not "
                             f"implement ({len(problems)} problem(s)):", problems,
                             "import the model through truestar.ir.import_module instead")
    return DecoderConfig(
        family=fam, vocab=int(g("vocab_size")), dim=dim, depth=depth, heads=heads, kv_heads=kv, head_dim=hd,
        hidden=int(g("intermediate_size")), act=act, eps=float(g("rms_norm_eps", 1e-6)),
        norm_kind="gemma" if gemma else "llama",
        qkv_bias=bool(g("attention_bias", False)) or fam == "qwen2",
        o_bias=bool(g("attention_bias", False)) and fam in ("llama", "mistral", "qwen3", "gemma2", "gemma3_text"),
        mlp_bias=bool(g("mlp_bias", False)), qk_norm=fam in ("qwen3", "gemma3_text"), sandwich=gemma,
        scale=scale, attn_softcap=attn_cap, final_softcap=final_cap,
        embed_scale="model" if gemma else "none",
        tie=bool(g("tie_word_embeddings", gemma)), rot_dim=rot, rope=rope, rope_local=rope_local,
        windows=windows, local_rope=local, fused_qkv=fam == "phi3", fused_gate_up=fam == "phi3",
        torch_dtype=str(g("torch_dtype") or "float32"), raw=dict(raw))
