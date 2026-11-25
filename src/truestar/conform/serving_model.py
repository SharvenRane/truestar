"""The decoder a serving engine is checked on, and Truestar's references for it (docs/CONFORM_SERVING.md).

Everything here is built in code: a Llama style config, weights drawn from a seeded CPU generator and
rounded once to the serving dtype, and token id prompts. Nothing is downloaded and no tokenizer is
needed (the engines run with their tokenizer switched off and take token ids).

The engine computes in its serving dtype from those rounded weights. The reference computes the same
weights in float64 on the CPU. A second CPU run in the serving dtype (every step rounded to it) is a
correct run in that dtype; its error against float64 sets the tolerance an engine is held to.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..engines import llama_ref as R

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class ServeSpec:
    """What the engine serves and what it is asked. Saved with every run, so the check side rebuilds
    the same weights and prompts and verifies their hashes."""
    vocab: int = 1024
    dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64
    ffn: int = 1408
    eps: float = 1e-5
    theta: float = 10000.0
    context: int = 2048
    dtype: str = "bfloat16"
    seed: int = 20261124
    n_prompts: int = 8
    prefix_len: int = 64       # every prompt starts with the same tokens (prefix caching can reuse them)
    suffix_len: int = 48
    max_tokens: int = 32       # greedy tokens generated per prompt
    top: int = 5               # top logprobs requested per position
    chunk: int = 32            # token budget per step when chunked prefill is on
    logit_scale: float = 3.0   # spread of the random head, so greedy choices are mostly not near ties

    def config(self) -> R.LlamaConfig:
        return R.LlamaConfig(vocab=self.vocab, dim=self.dim, n_layers=self.n_layers, n_heads=self.n_heads,
                             n_kv_heads=self.n_kv_heads, ffn=self.ffn, eps=self.eps, theta=self.theta,
                             head_dim=self.head_dim, context=self.context)

    def as_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "ServeSpec":
        known = {k: v for k, v in d.items() if k in ServeSpec.__dataclass_fields__}
        return ServeSpec(**known)


def hf_names(cfg: R.LlamaConfig) -> Dict[str, str]:
    """Truestar weight key -> Hugging Face Llama checkpoint name (what both engines load)."""
    m = {"embed": "model.embed_tokens.weight", "norm": "model.norm.weight", "output": "lm_head.weight"}
    hf = {"attn_norm": "input_layernorm", "q": "self_attn.q_proj", "k": "self_attn.k_proj",
          "v": "self_attn.v_proj", "o": "self_attn.o_proj", "ffn_norm": "post_attention_layernorm",
          "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
    for i in range(cfg.n_layers):
        for role, name in hf.items():
            m[f"{i}.{role}"] = f"model.layers.{i}.{name}.weight"
    return m


def weights(spec: ServeSpec) -> Dict[str, torch.Tensor]:
    """Seeded weights, rounded once to the serving dtype (the exact values the engine loads)."""
    cfg = spec.config()
    g = np.random.RandomState(spec.seed)  # frozen stream (NumPy NEP 19): same draws on every NumPy and torch
    randn = lambda *shape: torch.from_numpy(g.standard_normal(shape))  # noqa: E731
    D, H, KV, F = cfg.dim, cfg.head_dim * cfg.n_heads, cfg.head_dim * cfg.n_kv_heads, cfg.ffn

    def lin(out, inp, scale=1.0):
        return randn(out, inp) * (scale / math.sqrt(inp))

    W: Dict[str, torch.Tensor] = {"embed": randn(cfg.vocab, D)}
    for i in range(cfg.n_layers):
        W[f"{i}.attn_norm"] = 1.0 + 0.1 * randn(D)
        W[f"{i}.q"] = lin(H, D)
        W[f"{i}.k"] = lin(KV, D)
        W[f"{i}.v"] = lin(KV, D)
        W[f"{i}.o"] = lin(D, H)
        W[f"{i}.ffn_norm"] = 1.0 + 0.1 * randn(D)
        W[f"{i}.gate"] = lin(F, D)
        W[f"{i}.up"] = lin(F, D)
        W[f"{i}.down"] = lin(D, F)
    W["norm"] = 1.0 + 0.1 * randn(D)
    W["output"] = lin(cfg.vocab, D, spec.logit_scale)
    dt = DTYPES[spec.dtype]
    return {k: v.to(torch.float32).to(dt).contiguous() for k, v in W.items()}


def weights_sha256(W: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(W):
        t = W[k].contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.element_size() > 0 else b"")
    return h.hexdigest()


def prompts(spec: ServeSpec) -> List[List[int]]:
    """n_prompts prompts sharing a prefix of prefix_len tokens; ids 3.. (0 to 2 are left to specials)."""
    g = np.random.RandomState(spec.seed + 1)
    prefix = [int(x) for x in g.randint(3, spec.vocab, spec.prefix_len)]
    return [prefix + [int(x) for x in g.randint(3, spec.vocab, spec.suffix_len)]
            for _ in range(spec.n_prompts)]


def write_model_dir(spec: ServeSpec, path: str) -> str:
    """A Hugging Face style LlamaForCausalLM directory (config.json, model.safetensors)."""
    from safetensors.torch import save_file

    cfg = spec.config()
    os.makedirs(path, exist_ok=True)
    W = weights(spec)
    names = hf_names(cfg)
    save_file({names[k]: v for k, v in W.items()}, os.path.join(path, "model.safetensors"),
              metadata={"format": "pt"})
    conf = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "vocab_size": cfg.vocab,
            "hidden_size": cfg.dim, "intermediate_size": cfg.ffn, "num_hidden_layers": cfg.n_layers,
            "num_attention_heads": cfg.n_heads, "num_key_value_heads": cfg.n_kv_heads,
            "head_dim": cfg.head_dim, "max_position_embeddings": cfg.context, "rms_norm_eps": cfg.eps,
            "rope_theta": cfg.theta, "rope_scaling": None, "tie_word_embeddings": False,
            "hidden_act": "silu", "attention_bias": False, "mlp_bias": False,
            "torch_dtype": spec.dtype, "dtype": spec.dtype,
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(conf, f, indent=1)
    with open(os.path.join(path, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump({"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}, f)
    return weights_sha256(W)


def logprobs(spec: ServeSpec, W: Dict[str, torch.Tensor], ids: Sequence[int],
             dt: torch.dtype = torch.float64) -> torch.Tensor:
    """Log softmax of the next token distribution at every position of `ids`, [T, vocab], float64.

    dt float64 is the reference. Any other dt rounds every step's output to dt (a correct run in the
    engine's dtype); its logits are then turned into log probabilities in float32, as the engines do.
    """
    cfg = spec.config()
    ids_t = torch.as_tensor(list(ids), dtype=torch.long)
    pos = torch.arange(ids_t.shape[0])
    acts: Dict[str, torch.Tensor] = {"ids": ids_t}
    for st in R.steps(cfg):
        acts[st.key] = R.run_step(st, cfg, W, acts, pos, None, dt)
    logits = acts["logits"]
    if dt == torch.float64:
        return torch.log_softmax(logits, dim=-1)
    return torch.log_softmax(logits.float(), dim=-1).double()
