"""Which mixture of experts layers Truestar lowers, and the precise refusal for the rest.

A routed MoE block in transformers 5.24 (Mixtral, Qwen3 MoE, OLMoE, DeepSeek V2 and V3, Dots1,
FlexOlmo and the other families written the same way) is a router child that returns
``(logits, weights, indices)`` and an experts child ``experts(hidden [T, H], indices [T, k],
weights [T, k])`` holding ``gate_up_proj [E, 2I, H]`` and ``down_proj [E, H, I]``. The block may
add a shared expert (DeepSeek) next to the routed ones.

An experts module is covered when all of these hold, checked on the instance:

* 3D ``gate_up_proj`` and ``down_proj`` of matching E, H and I, no biases, not transposed;
* the gate is the default ``act(gate) * up`` (gpt-oss's clamped SwiGLU overrides it);
* a callable ``act_fn``.

Lowering is per instance: ``lowered(model)`` replaces each covered experts module's forward by the
static form (experts.static_experts) for the duration of an export, and tells the refusal rule
(ir/refusals.py) which module paths and classes it has lowered. The router is imported as written:
its ``topk`` becomes the IR's ``topk``, a stable sort (routing.TIE_RULE). Anything MoE shaped that
is not covered keeps TS940, with the reason (``reason_for``). ``use_grouped(model)`` is the eager
counterpart: the experts run ``grouped_experts`` and every router's ``torch.topk`` follows the tie
rule, so the same inputs pick the same experts on every device.
"""

from __future__ import annotations

import contextlib
import threading
import types
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from .experts import grouped_experts, static_experts
from .routing import stable_topk

_state = threading.local()


def _default_gate(experts: torch.nn.Module) -> bool:
    gate = getattr(type(experts), "_apply_gate", None)
    return gate is None or getattr(gate, "__name__", "") == "_default_apply_gate"


def experts_reason(m: torch.nn.Module) -> Optional[str]:
    """None when ``m`` is an experts module Truestar lowers, else why not (one sentence)."""
    gu, dn = getattr(m, "gate_up_proj", None), getattr(m, "down_proj", None)
    if not (isinstance(gu, torch.Tensor) and isinstance(dn, torch.Tensor)):
        return ("its experts are not stored as stacked gate_up_proj and down_proj tensors (per expert "
                "modules or another layout)")
    if gu.dim() != 3 or dn.dim() != 3:
        return f"its expert weights are not 3D stacks (gate_up_proj {tuple(gu.shape)}, down_proj {tuple(dn.shape)})"
    if getattr(m, "is_transposed", False):
        return "its expert weights are stored transposed"
    E, I2, H = gu.shape
    if dn.shape[0] != E or dn.shape[1] != H or I2 != 2 * dn.shape[2]:
        return (f"gate_up_proj {tuple(gu.shape)} and down_proj {tuple(dn.shape)} are not [E, 2I, H] and "
                f"[E, H, I]")
    if getattr(m, "has_bias", False) or getattr(m, "gate_up_proj_bias", None) is not None:
        return "its experts have biases"
    if not _default_gate(m):
        return f"its gate is not act(gate) * up ({type(m).__name__}._apply_gate is overridden, e.g. a clamped SwiGLU)"
    if not callable(getattr(m, "act_fn", None)):
        return "its experts have no act_fn"
    return None


def _is_moe_class(name: str) -> bool:
    from ..ir.refusals import classify
    return classify(name) == "moe"


def _capacity(m: torch.nn.Module) -> bool:
    return any(hasattr(m, a) for a in ("expert_capacity", "capacity_factor"))


def scan(model: torch.nn.Module) -> Tuple[List[str], Dict[str, str]]:
    """(paths of covered experts modules, {path: reason} for every MoE layer that is not covered)."""
    covered, refused = [], {}
    for path, m in model.named_modules():
        if _capacity(m):
            refused[path] = ("expert capacity dropping: tokens over an expert's capacity are dropped, which "
                             "the lowering does not express")
            continue
        if hasattr(m, "gate_up_proj") or (type(m).__name__.endswith("Experts") and _is_moe_class(type(m).__name__)):
            why = experts_reason(m)
            if why is None:
                covered.append(path)
            else:
                refused[path] = why
    lifted = lifted_paths() or set()
    for c in covered:
        if c not in lifted:     # covered, but this export does not run the static form
            parent = c.rsplit(".", 1)[0] if "." in c else ""
            refused.setdefault(parent, "its experts are covered by truestar.moe, but this export did not run inside "
                                       "truestar.moe.lowered(model) (ir.import_module does)")
    return covered, refused


def lifted_paths() -> Optional[set]:
    """Module paths whose MoE refusal is lifted by an active ``lowered`` context (None outside one)."""
    return getattr(_state, "paths", None)


def lifted_classes() -> Optional[set]:
    return getattr(_state, "classes", None)


def reason_for(path: str) -> Optional[str]:
    reasons = getattr(_state, "reasons", None) or {}
    return reasons.get(path)


def _norm(path: str) -> str:
    """torch.export's module stack path (``L['self'].model.layers[0].mlp`` or ``model.layers.0.mlp``)
    as a named_modules path."""
    p = path.replace("L['self'].", "").replace("L['self']", "")
    for a, b in (("['", "."), ("']", ""), ("[", "."), ("]", "")):
        p = p.replace(a, b)
    return p.strip(".")


def is_lifted(path: str) -> bool:
    """True when an active ``lowered`` context lowered the MoE layer at ``path``."""
    paths = lifted_paths()
    return paths is not None and _norm(path) in paths


def _static_forward(self, hidden_states, top_k_index, top_k_weights):
    return static_experts(hidden_states, top_k_index, top_k_weights, self.gate_up_proj, self.down_proj, self.act_fn)


def _grouped_forward(self, hidden_states, top_k_index, top_k_weights):
    return grouped_experts(hidden_states, top_k_index, top_k_weights, self.gate_up_proj, self.down_proj, self.act_fn,
                           name=getattr(self, "_truestar_path", "experts"))


@contextlib.contextmanager
def lowered(model: torch.nn.Module) -> Iterator[Dict[str, object]]:
    """Lower every covered experts module of ``model`` to the static form while the block runs, and
    lift TS940 for those blocks (paths and classes). Restores the modules on exit."""
    covered, refused = scan(model)
    mods = dict(model.named_modules())
    paths, classes = set(), set()
    for c in covered:
        parent = c.rsplit(".", 1)[0] if "." in c else ""
        block = mods[parent]
        paths.add(c)
        paths.add(parent)
        classes.add(type(mods[c]).__name__)
        if _is_moe_class(type(block).__name__):
            classes.add(type(block).__name__)
        for name, child in block.named_children():   # the router and any other MoE named sibling
            if _is_moe_class(type(child).__name__) and name != "shared_experts":
                p = f"{parent}.{name}" if parent else name
                if p not in refused:
                    paths.add(p)
                    classes.add(type(child).__name__)
    refused = {p: r for p, r in refused.items() if p not in paths}
    saved = []
    for c in covered:
        m = mods[c]
        saved.append((m, m.__dict__.get("forward")))

        m.forward = types.MethodType(_static_forward, m)    # a bound method: deepcopy rebinds it
    prev = (lifted_paths(), lifted_classes(), getattr(_state, "reasons", None))
    _state.paths = (prev[0] or set()) | paths
    _state.classes = (prev[1] or set()) | classes
    _state.reasons = {**(prev[2] or {}), **refused}
    try:
        yield {"lowered": covered, "refused": refused}
    finally:
        _state.paths, _state.classes, _state.reasons = prev
        for m, f in saved:
            if f is None:
                m.__dict__.pop("forward", None)
            else:
                m.forward = f


def use_grouped(model: torch.nn.Module) -> List[str]:
    """Eager: run every covered experts module with ``grouped_experts`` and every MoE router's topk
    under the tie rule. Returns the experts paths. Undo with ``restore(model)``."""
    covered, _ = scan(model)
    mods = dict(model.named_modules())
    for c in covered:
        m = mods[c]

        m.__dict__["_truestar_saved_forward"] = m.__dict__.get("forward")
        m._truestar_path = c
        m.forward = types.MethodType(_grouped_forward, m)
        parent = c.rsplit(".", 1)[0] if "." in c else ""
        block = mods[parent]
        if "_truestar_saved_forward" not in block.__dict__:
            orig = block.forward

            def bfwd(*a, _orig=orig, **kw):
                with stable_topk():
                    return _orig(*a, **kw)

            block.__dict__["_truestar_saved_forward"] = block.__dict__.get("forward")
            block.forward = bfwd
    return covered


def restore(model: torch.nn.Module) -> None:
    for m in model.modules():
        if "_truestar_saved_forward" in m.__dict__:
            f = m.__dict__.pop("_truestar_saved_forward")
            if f is None:
                m.__dict__.pop("forward", None)
            else:
                m.forward = f
