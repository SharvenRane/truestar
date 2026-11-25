"""LoRA and QLoRA fine tuning on truestar.nn models (docs/LORA.md).

    from truestar import lora, nn, optim
    from truestar.nn.llama import Llama, from_hf

    net, p = from_hf(hf_model)                               # or Llama(...), p = net.init(seed)
    lm = lora.wrap(net, targets=("q_proj", "v_proj"), rank=8, alpha=16, dropout=0.05)
    base = lm.prepare_base(p)                                # frozen; quantized with quant="int4:g64"
    a = lm.init(seed=0)                                      # the adapters: the only trained tree

    @truestar.native
    def loss(a: lm.spec, p: lm.base_spec, ids: Tensor["B", "T", "i64"], y: Tensor["B", "T", "i64"],
             training: bool = False) -> Tensor[()]:
        logits = lm(p, a, ids, training=training)
        return nn.cross_entropy(ops.reshape(logits, [-1, net.vocab]), ops.reshape(y, [-1]))

    step = truestar.train_step(loss, optim.AdamW(lr=1e-3), params="a", static={"training": True})
    ts = step.init(a)
    ts, r = step(ts, base, ids, y)                           # base is read, never differentiated
    merged = lm.merge(base, ts.params)                       # plain weights for `net`
    lora.save_adapters("run.lora.safetensors", lm, ts.params)

Design. ``wrap`` rewrites the layer config tree: every ``nn.Linear`` whose path matches a target
pattern becomes a ``LoRALinear`` (base weight, optional bias, A [r, d_in], B [d_out, r]), and with
``quant`` every other Linear except the skipped ones becomes a ``LoRALinear`` of rank 0 whose base
weight is stored in a Truestar weight format. The model is called with two trees, the frozen base
and the adapters; ``merge_trees`` joins them at compile time (it emits no op), so the training step
differentiates and updates the adapter tree only. Dropout on the adapter input is the keyed
``ops.dropout``: its mask depends on the step's key and the call site only, the same on every chip.

Codes TS1370 to TS1379 (``truestar codes``).
"""

from __future__ import annotations

import copy
import dataclasses
import fnmatch
import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from ..frontend import ops
from ..frontend.types import TensorSpec
from ..nn.core import Layer, Param, _dtype, fan_in_uniform, inline, layer, native_method, param_key, zeros
from ..nn.layers import Linear


class LoRAError(ValueError):
    """A LoRA configuration, tree or adapter file Truestar refuses; ``code`` is TS1370 to TS1379."""

    def __init__(self, message: str, code: str = "TS1370"):
        super().__init__(f"[{code}] {message}")
        self.code = code


QUANT_SPECS = ("int8", "int4:g32", "int4:g64", "int4:g128", "nvfp4", "mxfp4")


# ----------------------------------------------------------------------------- dequantization in the program


@inline
def _deq_int8(q, dt):
    n = q["scale"].shape[0]
    return ops.cast(q["codes"], dt) * ops.reshape(ops.cast(q["scale"], dt), [n, 1])


@inline
def _deq_int4(q, n, k, g, dt):
    idx = ops.reshape(ops.cast(q["codes"], "i64"), [n * (k // 2)])
    lo = ops.reshape(ops.index_select(q["lut_lo"], 0, idx), [n, k // g, g // 2])
    hi = ops.reshape(ops.index_select(q["lut_hi"], 0, idx), [n, k // g, g // 2])
    s = ops.reshape(ops.cast(q["scale"], dt), [n, k // g, 1])
    return ops.reshape(ops.cat([lo, hi], dim=2) * s, [n, k])


@inline
def _deq_fp4(q, n, k, g, nv, dt):
    idx = ops.reshape(ops.cast(q["codes"], "i64"), [n * (k // 2)])
    lo = ops.reshape(ops.index_select(q["lut_lo"], 0, idx), [n, k // 2, 1])
    hi = ops.reshape(ops.index_select(q["lut_hi"], 0, idx), [n, k // 2, 1])
    e = ops.reshape(ops.cat([lo, hi], dim=2), [n, k // g, g])
    s = ops.index_select(q["lut_s"], 0, ops.reshape(ops.cast(q["scale"], "i64"), [n * (k // g)]))
    if nv:
        s = s * ops.cast(q["ts"], dt)
    return ops.reshape(e * ops.reshape(s, [n, k // g, 1]), [n, k])


# ----------------------------------------------------------------------------- the adapted linear layer


@layer
class LoRALinear(Layer):
    """y = x W^T + b + (alpha / rank) * dropout(x) A^T B^T. W is frozen and either a float tensor
    (``p["w"]``) or a quantized weight (``p["qw"]``, decoded in the program). ``rank`` 0 is a frozen
    (possibly quantized) linear without an adapter. ``train`` turns the adapter dropout on: the
    training and the evaluation models are two configs, so the evaluation program has no random op."""

    d_in: int
    d_out: int
    bias: bool = True
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    quant: str = ""
    train: bool = False
    dtype: str = "f32"

    @property
    def scale(self) -> float:
        return float(self.alpha) / self.rank if self.rank else 0.0

    @property
    def qkind(self) -> str:
        return self.quant.split(":")[0] if self.quant else ""

    @property
    def group(self) -> int:
        if self.quant.startswith("int4:g"):
            return int(self.quant[6:])
        return {"nvfp4": 16, "mxfp4": 32}.get(self.quant, 0)

    def params(self):
        out = {"w": Param((self.d_out, self.d_in), fan_in_uniform(self.d_in))}
        if self.bias:
            out["b"] = Param((self.d_out,), fan_in_uniform(self.d_in))
        if self.rank:
            out["lora_a"] = Param((self.rank, self.d_in), fan_in_uniform(self.d_in))
            out["lora_b"] = Param((self.d_out, self.rank), zeros)
        return out

    def forward(self, p, x):
        if self.qkind == "":
            w = p["w"]
        elif self.qkind == "int8":
            w = _deq_int8(p["qw"], x.dtype)
        elif self.qkind == "int4":
            w = _deq_int4(p["qw"], self.d_out, self.d_in, self.group, x.dtype)
        else:
            w = _deq_fp4(p["qw"], self.d_out, self.d_in, self.group, self.qkind == "nvfp4", x.dtype)
        if self.bias:
            y = ops.linear(x, w, p["b"])
        else:
            y = ops.linear(x, w)
        if self.rank > 0:
            h = x
            if self.train and self.dropout > 0:
                h = ops.dropout(x, self.dropout)
            y = y + ops.linear(ops.linear(h, p["lora_a"]), p["lora_b"]) * self.scale
        return y


# ----------------------------------------------------------------------------- trees


def merge_trees(base, adapters):
    """The union of two nested dicts (the base tree and the adapter tree), computed at compile time:
    it only rearranges values and emits no op. A key present in both must hold dicts in both."""
    if not isinstance(adapters, dict):
        raise LoRAError("adapter tree node is not a dict", "TS1372")
    out = dict(base)
    for k, v in adapters.items():
        if k in out:
            if not (isinstance(out[k], dict) and isinstance(v, dict)):
                raise LoRAError(f"adapter entry {k!r} collides with a base tensor", "TS1372")
            out[k] = merge_trees(out[k], v)
        else:
            out[k] = v
    return out


merge_trees.__truestar_structure__ = True


def _get(tree, path: str):
    for k in path.split("."):
        tree = tree[k]
    return tree


def _set_copy(tree, path: str, value):
    """A copy of ``tree`` (dicts copied along ``path`` only) with ``path`` set to ``value``."""
    keys = path.split(".")
    out = dict(tree)
    node = out
    for k in keys[:-1]:
        node[k] = dict(node[k])
        node = node[k]
    node[keys[-1]] = value
    return out


def _leaves(tree, prefix=""):
    for k, v in tree.items():
        if isinstance(v, dict):
            yield from _leaves(v, f"{prefix}{k}.")
        else:
            yield prefix + k, v


def _matches(path: str, patterns: Sequence[str]) -> bool:
    last = path.rsplit(".", 1)[-1]
    return any(fnmatch.fnmatchcase(path, pat) or fnmatch.fnmatchcase(last, pat) for pat in patterns)


def _rewrite(lay: Layer, prefix: str, fn) -> Layer:
    """A copy of ``lay`` whose children are replaced by ``fn(path, child)`` (None keeps the child and
    recurses into it). Copies share nothing mutable with the original config."""
    new, changed = {}, False
    for name, child in lay.parts.items():
        path = prefix + name
        r = fn(path, child)
        if r is None:
            r = _rewrite(child, path + ".", fn)
        new[name] = r
        changed = changed or r is not child
    if not changed:
        return lay
    clone = copy.copy(lay)
    object.__setattr__(clone, "_parts", new)
    return clone


# ----------------------------------------------------------------------------- quantized base weights


def _luts(kind: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    from ..quant.formats import E2M1, E4M3, e8m0_decode

    b = torch.arange(256, dtype=torch.int64)
    lo, hi = b & 15, b >> 4
    if kind == "int4":
        out = {"lut_lo": torch.where(lo >= 8, lo - 16, lo).to(torch.float64),
               "lut_hi": torch.where(hi >= 8, hi - 16, hi).to(torch.float64)}
    else:
        out = {"lut_lo": E2M1.decode(lo), "lut_hi": E2M1.decode(hi)}
        s = E4M3.decode(b) if kind == "nvfp4" else e8m0_decode(b)
        out["lut_s"] = torch.nan_to_num(s, nan=0.0)   # NaN codes are never written by the quantizer
    return {k: v.to(dtype) for k, v in out.items()}


def quantize_leaf(w: torch.Tensor, spec: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """A float weight [N, K] as the leaves ``LoRALinear`` decodes: codes, scales and lookup tables.
    The codes and scales are exactly those of ``truestar.quant.formats.quantize_weight``."""
    from ..quant.formats import quantize_weight

    qw = quantize_weight(w, spec)
    kind = qw.fmt.kind
    if kind == "int8":
        return {"codes": qw.data.cpu(), "scale": qw.scale.cpu()}
    out = {"codes": qw.data.cpu()}
    if kind == "int4":
        out["scale"] = qw.scale.cpu()
    else:
        out["scale"] = qw.scale.view(torch.uint8).cpu() if qw.scale.dtype != torch.uint8 else qw.scale.cpu()
        if kind == "nvfp4":
            out["ts"] = qw.tensor_scale.reshape(1).to(torch.float32).cpu()
    out.update(_luts(kind, dtype))
    return out


def dequantize_leaf(q: Dict[str, torch.Tensor], spec: str, shape: Tuple[int, int]) -> torch.Tensor:
    """The float64 weight a quantized leaf stands for (``QuantWeight.dequantize``, independent of the
    in program decoder)."""
    from ..quant.formats import QuantWeight, parse

    fmt = parse(spec)
    scale = q["scale"]
    if fmt.kind == "nvfp4":
        scale = scale.view(torch.float8_e4m3fn)
    return QuantWeight(fmt, tuple(shape), q["codes"], scale, q.get("ts")).dequantize()


def _quant_spec_tree(lin: LoRALinear) -> Dict[str, TensorSpec]:
    dt = _dtype(lin.dtype)
    N, K = lin.d_out, lin.d_in
    kind, G = lin.qkind, lin.group
    if kind == "int8":
        return {"codes": TensorSpec((N, K), torch.int8), "scale": TensorSpec((N,), torch.float32)}
    out = {"codes": TensorSpec((N, K // 2), torch.uint8)}
    if kind == "int4":
        out["scale"] = TensorSpec((N, K // G), torch.float16)
    else:
        out["scale"] = TensorSpec((N, K // G), torch.uint8)
        if kind == "nvfp4":
            out["ts"] = TensorSpec((1,), torch.float32)
    for k in ("lut_lo", "lut_hi") + (("lut_s",) if kind in ("nvfp4", "mxfp4") else ()):
        out[k] = TensorSpec((256,), dt)
    return out


# ----------------------------------------------------------------------------- the wrapped model


class LoRAModel:
    """A model with LoRA adapters (built by ``wrap``). ``lm(p, a, x, training=False)`` calls the
    model on the base tree ``p`` and the adapter tree ``a``, eagerly or inside a native function."""

    def __init__(self, base: Layer, targets: Sequence[str], rank: int, alpha: float, dropout: float,
                 quant: Optional[str], quant_skip: Sequence[str]):
        if isinstance(targets, str):
            targets = (targets,)
        self.base, self.targets = base, tuple(targets)
        self.rank, self.alpha, self.dropout = int(rank), float(alpha), float(dropout)
        self.quant, self.quant_skip = quant or "", tuple(quant_skip)
        if self.rank < 1:
            raise LoRAError(f"rank must be at least 1, got {rank}", "TS1370")
        if not 0.0 <= self.dropout < 1.0:
            raise LoRAError(f"dropout must be in [0, 1), got {dropout}", "TS1370")
        if self.quant and self.quant not in QUANT_SPECS:
            raise LoRAError(f"quant {quant!r} is not a weight format LoRA supports; one of {', '.join(QUANT_SPECS)}",
                            "TS1376")
        self.adapted: Dict[str, LoRALinear] = {}     # path -> adapted layer (rank > 0)
        self.quantized: Dict[str, LoRALinear] = {}   # path -> layer whose base weight is quantized

        def make(train):
            def fn(path, child):
                if not isinstance(child, Linear):
                    return None
                hit = _matches(path, self.targets)
                q = self.quant if self.quant and not _matches(path, self.quant_skip) else ""
                if not hit and not q:
                    return None
                if q:
                    g = LoRALinear(child.d_in, child.d_out, child.bias, 1, 1.0, 0.0, q).group
                    if g and child.d_in % g:
                        raise LoRAError(f"{path}: input size {child.d_in} is not a multiple of the {q} block {g}; "
                                        f"skip it with quant_skip or pick a smaller group", "TS1376")
                r = LoRALinear(child.d_in, child.d_out, child.bias, self.rank if hit else 0,
                               self.alpha if hit else 0.0, self.dropout if hit else 0.0, q, train, child.dtype)
                if not train:
                    if hit:
                        self.adapted[path] = r
                    if q:
                        self.quantized[path] = r
                return r
            return fn

        self.eval_model = _rewrite(base, "", make(False))
        self.train_model = _rewrite(base, "", make(True)) if self.dropout > 0 else self.eval_model
        if not self.adapted:
            names = sorted(p for p, c in _named_layers(base) if isinstance(c, Linear))
            raise LoRAError(f"no Linear layer matches the targets {list(self.targets)}; the Linear layers are: "
                            f"{', '.join(names[:40])}", "TS1371")
        self.takes_training = bool(getattr(base, "uses_training", False))

    # -- trees
    @property
    def spec(self) -> Dict[str, Any]:
        """The adapter tree as an annotation."""
        out: Dict[str, Any] = {}
        for path, lin in self.adapted.items():
            dt = _dtype(lin.dtype)
            out = _set_tree(out, path, {"lora_a": TensorSpec((lin.rank, lin.d_in), dt),
                                        "lora_b": TensorSpec((lin.d_out, lin.rank), dt)})
        return out

    @property
    def base_spec(self) -> Dict[str, Any]:
        """The frozen base tree as an annotation: the model's spec with quantized weights replaced."""
        out = self.base.spec
        for path, lin in self.quantized.items():
            node = dict(_get(out, path))
            node.pop("w")
            node["qw"] = _quant_spec_tree(lin)
            out = _set_copy(out, path, node)
        return out

    def init(self, seed: int = 0) -> Dict[str, Any]:
        """Adapters before training: A uniform on +-1/sqrt(d_in) at the key fold(seed, path), B zero,
        so the adapted model starts equal to the base model."""
        out: Dict[str, Any] = {}
        for path, lin in self.adapted.items():
            dt = _dtype(lin.dtype)
            prm = lin.params()
            a = prm["lora_a"].init.draw((lin.rank, lin.d_in), dt, param_key(seed, path + ".lora_a"), path + ".lora_a")
            out = _set_tree(out, path, {"lora_a": a, "lora_b": torch.zeros((lin.d_out, lin.rank), dtype=dt)})
        return out

    def prepare_base(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """The frozen base tree from the model's float parameters: quantized where ``quant`` asks."""
        out = p
        for path, lin in self.quantized.items():
            node = dict(_get(p, path))
            w = node.pop("w")
            node["qw"] = quantize_leaf(w, lin.quant, _dtype(lin.dtype))
            out = _set_copy(out, path, node)
        return out

    def dequantized_base(self, base: Dict[str, Any], dtype=torch.float64) -> Dict[str, Any]:
        """The base tree with every quantized weight replaced by the float weight its format stands
        for (float64 by default): the model the QLoRA gradients are verified against."""
        out = base
        for path, lin in self.quantized.items():
            node = dict(_get(base, path))
            node["w"] = dequantize_leaf(node.pop("qw"), lin.quant, (lin.d_out, lin.d_in)).to(dtype)
            out = _set_copy(out, path, node)
        return _cast_tree(out, dtype)

    def merge(self, base: Dict[str, Any], adapters: Dict[str, Any], dtype: Optional[torch.dtype] = None):
        """Plain parameters for the original model: W + (alpha / rank) B A for every adapted layer,
        computed in float64 and rounded once to the layer's dtype (or ``dtype``). Quantized weights are
        decoded first, so the merged tree is float."""
        check_adapters(self, adapters)
        out = base
        for path, lin in self.quantized.items():
            node = dict(_get(out, path))
            node["w"] = dequantize_leaf(node.pop("qw"), lin.quant, (lin.d_out, lin.d_in))
            out = _set_copy(out, path, node)
        for path, lin in self.adapted.items():
            node = dict(_get(out, path))
            ad = _get(adapters, path)
            w64 = node["w"].to(torch.float64)
            delta = ad["lora_b"].to(torch.float64) @ ad["lora_a"].to(torch.float64)
            node["w"] = w64 + lin.scale * delta
            out = _set_copy(out, path, node)
        target = {path: dtype or _dtype(lin.dtype) for path, lin in {**self.quantized, **self.adapted}.items()}
        for path, dt in target.items():
            node = dict(_get(out, path))
            node["w"] = node["w"].to(dt)
            out = _set_copy(out, path, node)
        return out

    # -- calls
    def __truestar_inline__(self):
        return native_method(LoRAModel), (self,)

    def __call__(self, p, a, x, training: bool = False):
        return LoRAModel.forward(self, p, a, x, training)

    def forward(self, p, a, x, training=False):
        q = merge_trees(p, a)
        if training:
            m = self.train_model
        else:
            m = self.eval_model
        if self.takes_training:
            return m(q, x, training=training)
        return m(q, x)

    # -- numbers
    def config(self) -> Dict[str, Any]:
        return {"targets": list(self.targets), "rank": self.rank, "alpha": self.alpha, "dropout": self.dropout,
                "quant": self.quant, "quant_skip": list(self.quant_skip), "adapted": sorted(self.adapted),
                "quantized": sorted(self.quantized), "model": repr(self.base)}

    def num_adapter_params(self) -> int:
        return sum(lin.rank * (lin.d_in + lin.d_out) for lin in self.adapted.values())

    def __repr__(self):
        return (f"LoRAModel({type(self.base).__name__}, rank={self.rank}, alpha={self.alpha}, "
                f"dropout={self.dropout}, quant={self.quant or None}, {len(self.adapted)} adapted layers)")


def _named_layers(lay: Layer, prefix: str = ""):
    for name, child in lay.parts.items():
        yield prefix + name, child
        yield from _named_layers(child, f"{prefix}{name}.")


def _set_tree(tree, path, value):
    keys = path.split(".")
    out = dict(tree)
    node = out
    for k in keys[:-1]:
        node[k] = dict(node.get(k, {}))
        node = node[k]
    node[keys[-1]] = value
    return out


def _cast_tree(tree, dtype):
    return {k: (_cast_tree(v, dtype) if isinstance(v, dict) else
                (v.to(dtype) if v.is_floating_point() else v)) for k, v in tree.items()}


def wrap(model: Layer, targets: Sequence[str] = ("q_proj", "v_proj"), rank: int = 8, alpha: float = 16.0,
         dropout: float = 0.0, quant: Optional[str] = None, quant_skip: Sequence[str] = ("lm_head",)) -> LoRAModel:
    """Add LoRA adapters to every ``nn.Linear`` of ``model`` whose path (``l0.self_attn.q_proj``) or
    last name matches one of ``targets`` (shell patterns: ``q_proj``, ``l1.*``, ``*_proj``). With
    ``quant`` (one of QUANT_SPECS) every Linear not matching ``quant_skip`` keeps its base weight in
    that format (QLoRA). The model's own config is not changed."""
    if not isinstance(model, Layer):
        raise LoRAError(f"wrap needs a truestar.nn layer, got {type(model).__name__} (import a Hugging Face Llama "
                        "with truestar.nn.llama.from_hf first)", "TS1370")
    return LoRAModel(model, targets, rank, alpha, dropout, quant, quant_skip)


def check_adapters(lm: LoRAModel, adapters: Dict[str, Any]) -> None:
    """Refuse an adapter tree that does not fit ``lm``, naming every missing, extra or misshapen leaf."""
    want = dict(_leaves(lm.spec))
    have = dict(_leaves(adapters)) if isinstance(adapters, dict) else {}
    problems = [f"missing {k}" for k in want if k not in have]
    problems += [f"unexpected {k}" for k in have if k not in want]
    for k, s in want.items():
        if k in have:
            t = have[k]
            if tuple(t.shape) != tuple(s.shape) or t.dtype != s.dtype:
                problems.append(f"{k}: {t.dtype} {list(t.shape)}, expected {s.dtype} {list(s.shape)}")
    if problems:
        raise LoRAError("adapters do not fit this model: " + "; ".join(problems), "TS1372")


# ----------------------------------------------------------------------------- adapter files


def _base_hash(lm: LoRAModel) -> str:
    desc = [(k, str(s.dtype), [int(d) for d in s.shape]) for k, s in _leaves(lm.base.spec)]
    return hashlib.sha256(json.dumps(desc).encode()).hexdigest()


def save_adapters(path: str, lm: LoRAModel, adapters: Dict[str, Any]) -> str:
    """A safetensors file with the adapter tensors only (named by tree path) and, as metadata, the
    LoRA configuration and a hash of the base model's parameter layout."""
    from ..interop.safetensors import save_file

    check_adapters(lm, adapters)
    cfg = {k: v for k, v in lm.config().items() if k != "model"}
    meta = {"truestar.kind": "lora_adapters", "truestar.lora": json.dumps(cfg, sort_keys=True),
            "truestar.base": _base_hash(lm)}
    return save_file({k: v.detach().cpu().contiguous() for k, v in _leaves(adapters)}, path, meta)


def load_adapters(path: str, lm: LoRAModel) -> Dict[str, Any]:
    """Read adapters written by ``save_adapters`` for a model with the same LoRA configuration and base
    layout (TS1373 otherwise, naming what differs)."""
    from ..interop.safetensors import LoadReport, load_file

    rep = LoadReport(path)
    data = load_file(path, mmap=False, report=rep)
    meta = rep.metadata or {}
    if meta.get("truestar.kind") != "lora_adapters":
        raise LoRAError(f"{path} is not a truestar LoRA adapter file", "TS1373")
    cfg = json.loads(meta.get("truestar.lora", "{}"))
    mine = {k: v for k, v in lm.config().items() if k != "model"}
    diff = [f"{k}: file {cfg.get(k)!r}, model {mine[k]!r}" for k in mine if cfg.get(k) != mine[k]]
    if meta.get("truestar.base") != _base_hash(lm):
        diff.append("the base model's parameter layout differs")
    if diff:
        raise LoRAError(f"{path} was written for another LoRA model: " + "; ".join(diff), "TS1373")
    out: Dict[str, Any] = {}
    for k, v in data.items():
        out = _set_tree(out, k, v.clone())
    check_adapters(lm, out)
    return out


# ----------------------------------------------------------------------------- memory accounting


def tree_bytes(tree) -> int:
    return sum(v.numel() * v.element_size() for _, v in _leaves(tree))


def memory_report(lm: LoRAModel, base: Dict[str, Any], float_params: Dict[str, Any], optimizer_slots: int = 2
                  ) -> Dict[str, int]:
    """Bytes a step carries: trained tensors, their gradients and optimizer slots, and the frozen base,
    for LoRA (this model) and for full fine tuning of the float model."""
    a_bytes = sum(lin.rank * (lin.d_in + lin.d_out) * _dtype(lin.dtype).itemsize for lin in lm.adapted.values())
    full = tree_bytes(float_params)
    return {"lora_trained": a_bytes, "lora_grads_and_slots": a_bytes * (1 + optimizer_slots),
            "lora_frozen_base": tree_bytes(base), "lora_total": a_bytes * (2 + optimizer_slots) + tree_bytes(base),
            "full_trained": full, "full_grads_and_slots": full * (1 + optimizer_slots),
            "full_total": full * (2 + optimizer_slots)}


__all__ = ["LoRAError", "LoRALinear", "LoRAModel", "QUANT_SPECS", "check_adapters", "dequantize_leaf", "load_adapters",
           "memory_report", "merge_trees", "quantize_leaf", "save_adapters", "tree_bytes", "wrap"]


# ----------------------------------------------------------------------------- checks


def decode_weight(lin: LoRALinear, q: Dict[str, torch.Tensor], dtype=torch.float64) -> torch.Tensor:
    """The in program decoder of a quantized base weight, run eagerly."""
    if lin.qkind == "int8":
        return _deq_int8(q, dtype)
    luts = {k: (v.to(dtype) if k.startswith("lut") else v) for k, v in q.items()}
    if lin.qkind == "int4":
        return _deq_int4(luts, lin.d_out, lin.d_in, lin.group, dtype)
    return _deq_fp4(luts, lin.d_out, lin.d_in, lin.group, lin.qkind == "nvfp4", dtype)


def verify_decode(lm: LoRAModel, base: Dict[str, Any]) -> int:
    """Every quantized base weight, decoded the way the program decodes it (in float64), must equal
    the format's exact dequantization bit for bit (TS1375 otherwise). Returns the number checked."""
    bad = []
    for path, lin in lm.quantized.items():
        q = _get(base, path)["qw"]
        got = decode_weight(lin, q)
        want = dequantize_leaf(q, lin.quant, (lin.d_out, lin.d_in))
        if not torch.equal(got, want):
            n = int((got != want).sum())
            bad.append(f"{path} ({lin.quant}): {n} of {got.numel()} entries differ")
    if bad:
        raise LoRAError("the in program decoder disagrees with the format's dequantization: " + "; ".join(bad),
                        "TS1375")
    return len(lm.quantized)


def verify_merge(lm: LoRAModel, base: Dict[str, Any], adapters: Dict[str, Any], x: torch.Tensor,
                 tol: float = 1e-12) -> float:
    """Run the adapter model and the merged model in float64 on ``x`` and return the largest
    difference relative to the largest output magnitude; TS1374 when it exceeds ``tol``. Merging is
    exact in real arithmetic, so the difference is float64 rounding only."""
    b64, a64 = _cast_tree(base, torch.float64), _cast_tree(adapters, torch.float64)
    ref = lm.eval_model(merge_trees(b64, a64), x)
    merged = lm.merge(base, adapters, dtype=torch.float64)
    got = lm.base(_cast_tree(merged, torch.float64), x)
    err = float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-300))
    if not err <= tol:
        raise LoRAError(f"the merged model differs from the adapter model by {err:.3g} relative (bound {tol:g})",
                        "TS1374")
    return err


def assert_bitwise(a: Dict[str, Any], b: Dict[str, Any], what: str = "adapters") -> None:
    """Two adapter trees must be bit identical (TS1377 otherwise, naming the first leaf that differs)."""
    la, lb = dict(_leaves(a)), dict(_leaves(b))
    for k in la:
        x, y = la[k], lb.get(k)
        if y is None or x.dtype != y.dtype or x.shape != y.shape or \
                not torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)):
            raise LoRAError(f"{what} differ between two runs at {k}", "TS1377")


__all__ += ["assert_bitwise", "decode_weight", "verify_decode", "verify_merge"]
