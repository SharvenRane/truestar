"""``from_pretrained``: a local Hugging Face folder (config.json plus safetensors, one file or
sharded) built into the native decoder of ``truestar.models.decoder``, without transformers and
without the PyTorch exporter.

Every checkpoint tensor is matched to a leaf of the native parameter tree before anything is
returned, and every problem is listed at once: a tensor the model needs that the folder lacks
(TS1392), a tensor the folder has that the model does not use (TS1393), a shape that differs
(TS1394). The sharded index is checked by the reader (TS1395). Converting the stored dtype to the
requested one is reported (TS1396, info). A path that is not a local folder is refused (TS1397):
this loader never downloads.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..interop import safetensors_np as stnp
from .config import DecoderConfig, ModelLoadError, decoder_config, read_config
from .decoder import DecoderLM

CODE_MISSING, CODE_UNEXPECTED, CODE_SHAPE, CODE_CAST, CODE_NOT_LOCAL = "TS1392", "TS1393", "TS1394", "TS1396", "TS1397"

_DT = {"f64": torch.float64, "f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16,
       "float64": torch.float64, "float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
_SHORT = {torch.float64: "f64", torch.float32: "f32", torch.bfloat16: "bf16", torch.float16: "f16"}
# names old checkpoints carry that hold no trained value (recomputed from the config)
_IGNORED_SUFFIXES = ("rotary_emb.inv_freq",)


@dataclass
class LoadInfo:
    """What the load did: the source, counts, conversions (TS1396) and ignored names."""

    source: str
    family: str
    dtype: str
    tensors: int = 0
    bytes: int = 0
    shards: int = 1
    zero_copy: int = 0                                   # leaves that are views of the memory map
    cast: Dict[str, int] = field(default_factory=dict)   # "BF16->f32": count  (TS1396)
    ignored: List[str] = field(default_factory=list)
    seconds: float = 0.0

    def findings(self) -> List[Dict[str, Any]]:
        if not self.cast:
            return []
        return [{"code": CODE_CAST, "severity": "info",
                 "message": "weights converted from the stored dtype: " +
                 ", ".join(f"{k} x{v}" for k, v in sorted(self.cast.items()))}]


def build_model(cfg: DecoderConfig, dtype: str) -> DecoderLM:
    """The native layer for a config, in ``dtype`` ("f64", "f32", "bf16", "f16")."""
    tdt = _DT[dtype]
    up = "f64" if tdt == torch.float64 else "f32"
    scale = 0.0
    if cfg.embed_scale == "model":
        scale = float(torch.tensor(cfg.dim ** 0.5, dtype=tdt))
    elif cfg.embed_scale == "f32":
        scale = float(torch.tensor(cfg.dim ** 0.5, dtype=torch.float32).to(tdt))
    return DecoderLM(vocab=cfg.vocab, dim=cfg.dim, depth=cfg.depth, heads=cfg.heads, kv_heads=cfg.kv_heads,
                     head_dim=cfg.head_dim, hidden=cfg.hidden, rot=cfg.rot_dim, scale=cfg.scale, windows=cfg.windows,
                     local_rope=cfg.local_rope, act=cfg.act, eps=cfg.eps, norm_kind=cfg.norm_kind,
                     qkv_bias=cfg.qkv_bias, o_bias=cfg.o_bias, mlp_bias=cfg.mlp_bias, qk_norm=cfg.qk_norm,
                     sandwich=cfg.sandwich, attn_softcap=cfg.attn_softcap, final_softcap=cfg.final_softcap,
                     embed_scale=scale, tie=cfg.tie, upcast=up, dtype=dtype)


def name_map(cfg: DecoderConfig) -> Dict[str, List[Tuple[str, Optional[Tuple[int, int]]]]]:
    """Checkpoint name -> [(tree path, row range or None)]. A fused tensor maps to several leaves,
    each taking a range of its rows."""
    out: Dict[str, List[Tuple[str, Optional[Tuple[int, int]]]]] = {}

    def one(src, dst):
        out[src] = [(dst, None)]

    one("model.embed_tokens.weight", "embed.w")
    one("model.norm.weight", "norm.w")
    if not cfg.tie:
        one("lm_head.weight", "head.w")
    q, kv = cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim
    for i in range(cfg.depth):
        s, d = f"model.layers.{i}.", f"l{i}."
        one(s + "input_layernorm.weight", d + "in_norm.w")
        one(s + "post_attention_layernorm.weight", d + "post_attn_norm.w")
        if cfg.sandwich:
            one(s + "pre_feedforward_layernorm.weight", d + "pre_ff_norm.w")
            one(s + "post_feedforward_layernorm.weight", d + "post_ff_norm.w")
        a = s + "self_attn."
        if cfg.fused_qkv:
            out[a + "qkv_proj.weight"] = [(d + "attn.q.w", (0, q)), (d + "attn.k.w", (q, q + kv)),
                                          (d + "attn.v.w", (q + kv, q + 2 * kv))]
        else:
            for n in "qkv":
                one(a + f"{n}_proj.weight", d + f"attn.{n}.w")
                if cfg.qkv_bias:
                    one(a + f"{n}_proj.bias", d + f"attn.{n}.b")
        one(a + "o_proj.weight", d + "attn.o.w")
        if cfg.o_bias:
            one(a + "o_proj.bias", d + "attn.o.b")
        if cfg.qk_norm:
            one(a + "q_norm.weight", d + "attn.q_norm.w")
            one(a + "k_norm.weight", d + "attn.k_norm.w")
        m = s + "mlp."
        if cfg.fused_gate_up:
            out[m + "gate_up_proj.weight"] = [(d + "mlp.w1", (0, cfg.hidden)), (d + "mlp.w3", (cfg.hidden, 2 * cfg.hidden))]
        else:
            one(m + "gate_proj.weight", d + "mlp.w1")
            one(m + "up_proj.weight", d + "mlp.w3")
        one(m + "down_proj.weight", d + "mlp.w2")
        if cfg.mlp_bias:
            if cfg.fused_gate_up:
                out[m + "gate_up_proj.bias"] = [(d + "mlp.b1", (0, cfg.hidden)), (d + "mlp.b3", (cfg.hidden, 2 * cfg.hidden))]
            else:
                one(m + "gate_proj.bias", d + "mlp.b1")
                one(m + "up_proj.bias", d + "mlp.b3")
            one(m + "down_proj.bias", d + "mlp.b2")
    return out


def _leaves(tree, prefix=""):
    for k, v in tree.items():
        if isinstance(v, dict):
            yield from _leaves(v, f"{prefix}{k}.")
        else:
            yield prefix + k, v


def _set(tree, path, value):
    *head, last = path.split(".")
    for h in head:
        tree = tree.setdefault(h, {})
    tree[last] = value


def load_params(cfg: DecoderConfig, model: DecoderLM, ck: stnp.Checkpoint, info: LoadInfo) -> Dict[str, Any]:
    want = dict(_leaves(model.spec))
    nm = name_map(cfg)
    canon = {n: n for n in ck.names()}
    missing, unexpected, shapes = [], [], []
    params: Dict[str, Any] = {}
    seen = set()
    for stored, name in canon.items():
        if name.endswith(_IGNORED_SUFFIXES) or (cfg.tie and name == "lm_head.weight"):
            info.ignored.append(stored)
            continue
        dests = nm.get(name)
        if dests is None:
            raw = ck[stored]
            unexpected.append(f"{CODE_UNEXPECTED} {stored}: {raw.dtype}{list(raw.shape)} is not used by the native "
                              f"{cfg.family} model")
            continue
        raw = ck[stored]
        rows = sum((r[1] - r[0]) if r else 0 for _, r in dests)
        full = [want[d] for d, _ in dests]
        ok = True
        if any(r is not None for _, r in dests):
            inner = tuple(full[0].shape[1:])
            if tuple(raw.shape) != (rows,) + inner:
                shapes.append(f"{CODE_SHAPE} {stored}: {list(raw.shape)} stored, the native model splits it into "
                              f"{' + '.join(str(list(t.shape)) for t in full)} rows")
                ok = False
        elif tuple(raw.shape) != tuple(full[0].shape):
            shapes.append(f"{CODE_SHAPE} {stored}: {list(raw.shape)} stored, {list(full[0].shape)} expected "
                          f"for {dests[0][0]}")
            ok = False
        seen.update(d for d, _ in dests)
        if not ok:
            continue
        t = raw.torch()
        info.tensors += 1
        info.bytes += raw.nbytes
        for d, r in dests:
            leaf_dt = want[d].dtype
            v = t if r is None else t[r[0]:r[1]]
            if v.dtype != leaf_dt:
                key = f"{raw.dtype}->{_SHORT.get(leaf_dt, str(leaf_dt))}"
                info.cast[key] = info.cast.get(key, 0) + 1
                v = v.to(leaf_dt)
            else:
                info.zero_copy += 1
            _set(params, d, v)
    rope_leaves = {"rope.inv": cfg.rope}
    if model.has_local:
        rope_leaves["rope_local.inv"] = cfg.rope_local
    for d, spec in rope_leaves.items():
        _set(params, d, spec.inv_freq())
        seen.add(d)
    for d in sorted(set(want) - seen):
        src = [s for s, ds in nm.items() if any(x == d for x, _ in ds)]
        missing.append(f"{CODE_MISSING} {src[0] if src else d}: needed for {d} {list(want[d].shape)}, not in the folder")
    problems = missing + unexpected + shapes
    if problems:
        code = CODE_MISSING if missing else CODE_UNEXPECTED if unexpected else CODE_SHAPE
        raise ModelLoadError(code, f"{ck.source} does not fit the native {cfg.family} model "
                             f"({len(problems)} problem(s), nothing was loaded):", problems,
                             "check that config.json belongs to these weights")
    return params


def from_pretrained(path: str, *, dtype: Optional[str] = None, executor=None) -> "PretrainedLM":
    """Build the native model of a local Hugging Face folder.

    ``dtype``: "f64", "f32", "bf16", "f16"; None takes the checkpoint's float dtype (F64 files give
    a float64 model, F32 float32; BF16, F16 and FP8 files give float32 unless asked). ``executor``:
    where the compiled programs run (default: the float64 reference interpreter; a
    ``truestar.compiler.CompiledExecutor`` for Truestar kernels). Hub ids are not fetched (TS1397)."""
    t0 = time.perf_counter()
    path = os.fspath(path)
    if not os.path.isdir(path):
        raise ModelLoadError(CODE_NOT_LOCAL, f"{path!r} is not a local folder",
                             ["from_pretrained reads a folder written by save_pretrained (config.json and safetensors); "
                              "it never downloads from the Hub"],
                             "download the model yourself (for example huggingface_hub.snapshot_download) and pass "
                             "the folder")
    cfg = decoder_config(read_config(path))
    ck = stnp.open_checkpoint(path)
    if dtype is None:
        counts: Dict[str, int] = {}
        for k in ck.names():
            d = ck[k].dtype
            if d in ("F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"):
                counts[d] = counts.get(d, 0) + ck[k].nbytes
        top = max(counts, key=counts.get) if counts else "F32"
        dtype = {"F64": "f64"}.get(top, "f32")
    if dtype not in _DT:
        raise ValueError(f"dtype must be one of f64, f32, bf16, f16, got {dtype!r}")
    dtype = _SHORT[_DT[dtype]]
    model = build_model(cfg, dtype)
    info = LoadInfo(ck.source, cfg.family, dtype, shards=len(ck.files))
    params = load_params(cfg, model, ck, info)
    info.seconds = time.perf_counter() - t0
    from .runner import PretrainedLM
    return PretrainedLM(model, params, cfg, info, executor)
