"""Model families Truestar refuses by name, before any op is imported (coverage refresh, 2026-11-24).

Three kinds of layer in current models have no verified path in Truestar yet. A model that has one
is refused with a stable code that names the layer, its class and the first module path, so the
answer is never a silent CPU fallback or a number nobody checked:

* TS940 ``moe_routing_unsupported``: mixture of experts. The router's top k is a discrete choice
  (an ulp can change which experts a token reaches), experts are dispatched by data dependent
  gathers and grouped matmuls, and the gradient through routing has no verified rule. gpt-oss,
  Qwen3 MoE, Qwen3 Next, Qwen3.5 MoE, Llama 4, DeepSeek V3, Kimi Linear, OLMoE, Gemma 4 26B A4B,
  Granite 4 hybrids, Jamba and the other routed models.
* TS941 ``linear_attention_unsupported``: chunked linear attention and delta rule recurrences
  with no lowering (Kimi Delta Attention, lightning attention, mLSTM, and a Gated DeltaNet that
  ``hf_linattn`` does not cover).
* TS942 ``chunked_ssm_unsupported``: chunked selective state space scans with no lowering. The Mamba
  1 sequential scan is supported.
* TS1202 ``sparse_attention_selection_refused``: token sparse attention, where an indexer scores
  every key and keeps the top k per query (DeepSeek Sparse Attention in DeepSeek V3.2 and V4, and
  the indexers of GLM MoE DSA, GLM 5 Next, Mistral Large 4, HY V4, AXK2, Dots3 Note, MiniMax M3 VL
  and Qwen4 Exp in transformers 5.24.1). Which keys a query reads is a discrete choice an ulp can
  flip. Checked first: a model with it is refused by this code even when it also has MoE layers,
  because dropping the MoE layers (dense layers only) still leaves the indexer.
* TS1260 ``detection_postprocess_unsupported``: the eval forward of torchvision's detectors (Faster,
  Mask and Keypoint R-CNN, RetinaNet, FCOS, SSD). Proposal selection and post processing (top k over
  anchors, score threshold, removal of small boxes, non maximum suppression, RoI align on the kept
  boxes) return a number of boxes that depends on the values; torch.export stops on it
  (``GuardOnDataDependentSymNode``) and Truestar has no verified path for it. The dense part (the
  backbone with FPN and the RPN or detection head) and the second stage on pooled RoI features
  (box head and predictor, mask head and predictor) import, compile and train (vision and image
  generation coverage, docs/MODEL_COVERAGE.md).

Since P60 (2026-11-24) the Mamba 2 SSD layers of Mamba 2, Falcon H1, Bamba and Zamba 2 and the
Gated DeltaNet layer of Qwen3.5 are lowered to the verified ``linear_attention`` op instead
(``hf_linattn.py``): ``check_module`` skips a layer ``hf_linattn.lowerable`` covers, and
``check_exported`` skips a layer only when the exported graph really holds a
``truestar.linear_attention`` node inside it, so a program exported without the lowering is still
refused. The refusal message says why a layer was not lowered.

The rule reads module class names: the model's own modules in ``import_module``, and the module
stack torch.export records on every node in ``import_exported``. A class is matched by exact name
first, then by a name part (``SparseMoeBlock``, ``GatedDeltaNet``, ``Mamba2Mixer`` and so on), so a
new family built from the same layers is refused too. ``refusal_for`` gives the verdict without
raising, ``check_module`` and ``check_exported`` raise ``UnsupportedOpError`` with ``err.code``.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

import torch

from .core import UnsupportedOpError

# code, short name, what is not supported (used in the message)
KINDS = {
    "moe": ("TS940", "mixture of experts routing",
            "router top k, expert dispatch and the gradient through routing are not verified"),
    "linear": ("TS941", "chunked linear attention",
               "chunked linear attention and delta rule recurrences are imported only through a verified "
               "lowering to linear_attention (truestar.ir.hf_linattn)"),
    "ssm": ("TS942", "chunked selective state space scan",
            "chunked SSD scans are imported only through a verified lowering to linear_attention "
            "(truestar.ir.hf_linattn; the Mamba 1 sequential scan imports as it is)"),
    "sparse": ("TS1202", "token sparse attention selection",
               "the indexer's top k decides which keys each query reads, a discrete choice an ulp can flip; "
               "it is not built (the dense MLA attention of DeepSeek V3 is, truestar.mla)"),
    "detect": ("TS1260", "data dependent detection post processing",
               "proposal selection and post processing (top k over anchors, score threshold, removal of small "
               "boxes, non maximum suppression, RoI align on the kept boxes) return a number of boxes that "
               "depends on the values, which torch.export cannot trace with fixed shapes and Truestar has no "
               "verified path for"),
}

# what a refusal tells the user to do next, per kind (default: where refused and lowered layers are listed)
NEXT = {
    "detect": "Import the dense part instead (model.backbone with model.rpn.head, or model.head for RetinaNet, "
              "FCOS and SSD) and the second stage on pooled features (model.roi_heads.box_head and box_predictor, "
              "mask_head and mask_predictor); see docs/MODEL_COVERAGE.md (image generation, detection and "
              "segmentation).",
    "sparse": "Not planned: docs/MLA.md.",
}

# exact class names seen in transformers 5.24 (the sweep of docs/MODEL_COVERAGE.md)
EXACT = {
    "GptOssMLP": "moe", "GptOssExperts": "moe", "GptOssTopKRouter": "moe",
    "Qwen3MoeSparseMoeBlock": "moe", "Qwen3NextSparseMoeBlock": "moe", "Qwen3_5MoeSparseMoeBlock": "moe",
    "Llama4TextMoe": "moe", "Llama4TextExperts": "moe", "DeepseekV3MoE": "moe", "DeepseekV32MoE": "moe",
    "KimiSparseMoeBlock": "moe", "OlmoeSparseMoeBlock": "moe", "MiniMaxSparseMoeBlock": "moe",
    "Dots1MoE": "moe", "JambaSparseMoeBlock": "moe", "GraniteMoeHybridMoE": "moe", "NemotronHMoE": "moe",
    "Qwen3NextGatedDeltaNet": "linear", "Qwen3_5GatedDeltaNet": "linear", "Qwen3_5MoeGatedDeltaNet": "linear",
    "KimiDeltaAttention": "linear", "MiniMaxLightningAttention": "linear", "xLSTMLayer": "linear",
    "Mamba2Mixer": "ssm", "FalconH1Mixer": "ssm", "NemotronHMamba2Mixer": "ssm", "BambaMixer": "ssm",
    "Zamba2MambaMixer": "ssm", "GraniteMoeHybridMambaLayer": "ssm",
    "DeepseekV32Indexer": "sparse", "DeepseekV4Indexer": "sparse", "GlmMoeDsaIndexer": "sparse",
    "Glm5NextTextIndexer": "sparse", "MistralLarge4TextIndexer": "sparse", "HYV4Indexer": "sparse",
    "HYV4VLIndexer": "sparse", "AXK2Indexer": "sparse", "Dots3NoteTextIndexer": "sparse",
    "MiniMaxM3VLIndexer": "sparse", "Qwen4ExpTextQSAIndexer": "sparse",
}

# torchvision's detection modules whose eval forward selects boxes by value. Matched only with their
# module (``torchvision.models.detection...``): a class of the user's own named ``SSD`` is not one.
DETECT_MODULE = "torchvision.models.detection"
DETECT = {"RegionProposalNetwork", "RoIHeads", "GeneralizedRCNN", "FasterRCNN", "MaskRCNN", "KeypointRCNN",
          "RetinaNet", "FCOS", "SSD"}

# name parts that mark the same layers in families not listed above
PATTERNS = [
    (re.compile(r"(SparseMoe|MoeBlock|MoE$|Moe$|Experts$|TopKRouter$|MoeRouter$|ExpertRouter$|Top\d+Router$|"
                r"SparseMLP$)"), "moe"),
    (re.compile(r"(GatedDeltaNet|DeltaNet|DeltaAttention|LightningAttention|LinearAttention$)"), "linear"),
    (re.compile(r"Mamba2"), "ssm"),
    (re.compile(r"(SparseAttnIndexer|SAIndexer$|DsaIndexer$|TextIndexer$|AttentionIndexer$)"), "sparse"),
]


def _short(cls_name: str) -> str:
    return cls_name.rsplit(".", 1)[-1]


def classify(cls_name: str) -> Optional[str]:
    """The refused kind of a module class name (short, or qualified with its module), or None."""
    if "." in cls_name:
        mod, cls_name = cls_name.rsplit(".", 1)
        if cls_name in DETECT and (mod == DETECT_MODULE or mod.startswith(DETECT_MODULE + ".")):
            return "detect"
    if cls_name in EXACT:
        return EXACT[cls_name]
    for pat, kind in PATTERNS:
        if pat.search(cls_name):
            return kind
    return None


def _container_or_norm(path: str, cls: str, lifted) -> bool:
    """A class matched only by a name part (``Mamba2`` in ``Mamba2RMSNorm``, ``Mamba2Block``,
    ``Mamba2Model``) that is a normalization layer, or that holds a lowered layer: it computes no
    recurrence of its own once that layer is lowered, so it is not refused for it."""
    if cls.endswith(("Norm", "NormGated")):
        return True
    return any(p.startswith(path + ".") or not path for p, _ in lifted)


def refusal_for(modules: Iterable[Tuple[str, str]], lifted: Iterable[Tuple[str, str]] = ()
                ) -> Optional[Tuple[str, str, str, str]]:
    """(code, kind name, class, path) of the first refused (path, class name) pair: token sparse
    selection first, then MoE.
    ``lifted`` pairs (layers lowered to linear_attention) are not refused."""
    lifted = set(lifted)
    # per kind, the most specific layer: an exact class name first, then the deepest path, so a
    # message names the mixer or the expert block, not the model class that holds it
    from ..moe.coverage import is_lifted
    found = {}
    for path, cls in modules:
        kind = classify(cls)    # qualified names too: torchvision's detectors match with their module
        short = _short(cls)
        if kind is None or (kind in ("linear", "ssm") and (path, short) in lifted):
            continue
        if kind == "moe" and is_lifted(path):   # lowered by truestar.moe.lowered
            continue
        if kind in ("linear", "ssm") and short not in EXACT and _container_or_norm(path, short, lifted):
            continue
        rank = (short in EXACT or short in DETECT, path.count(".") + bool(path))
        if kind not in found or rank > found[kind][2]:
            found[kind] = (short, path, rank)
    for kind in ("sparse", "moe", "linear", "ssm", "detect"):
        if kind in found:
            code, what, _ = KINDS[kind]
            return code, what, found[kind][0], found[kind][1]
    return None


def _raise(name: str, verdict, others, detail: Optional[str] = None, reasons=None) -> None:
    code, what, cls, path = verdict
    why = next(v[2] for v in KINDS.values() if v[0] == code)
    if code == "TS940":
        from ..moe.coverage import reason_for
        reason = _near_reason(path, {**(reasons or {})}) or reason_for(path)
        if reason:
            why = (f"{reason}; Truestar lowers routed top k experts stored as stacked gate_up_proj [E, 2I, H] and "
                   f"down_proj [E, H, I] with act(gate) * up and no biases (truestar.moe)")
    extra = f" (also: {', '.join(sorted(others))})" if others else ""
    detail = f" Not lowered: {detail}." if detail else ""
    kind = next(k for k, v in KINDS.items() if v[0] == code)
    tail = NEXT.get(kind, "See docs/MODEL_COVERAGE.md (refused) and docs/LINEAR_ATTENTION.md (lowered layers).")
    err = UnsupportedOpError(
        f"[{code}] truestar.ir refuses {name}: it has {what} ({cls} at {path or 'the top module'}); {why}."
        f"{detail} Nothing was imported{extra}. {tail}")
    err.code = code
    raise err


def _near_reason(path, reasons):
    """The reason recorded for this layer, its children or its block's other children (the router
    and the experts of one block); capacity dropping first, as it is the deciding difference."""
    parent = path.rsplit(".", 1)[0] if "." in path else ""
    near = [r for p, r in reasons.items()
            if p == path or p.startswith(path + ".") or (parent and (p == parent or p.startswith(parent + ".")))]
    near.sort(key=lambda r: "capacity" not in r)
    return near[0] if near else None


def _kinds(modules, lifted=()):
    """Kind names of the refused layers in ``modules``, leaving out lowered MoE blocks and the
    ``lifted`` (path, class) pairs lowered to linear_attention."""
    from ..moe.coverage import is_lifted
    lifted = set(lifted)
    return {KINDS[k][1] for k in {classify(c) for p, c in modules
                                  if (p, _short(c)) not in lifted and not (classify(c) == "moe" and is_lifted(p))}
            if k is not None}


def check_module(model: torch.nn.Module, name: Optional[str] = None) -> None:
    from .hf_linattn import lowerable, unlowered_reason
    from ..moe.coverage import scan
    named = dict(model.named_modules())
    mods = [(p, f"{type(m).__module__}.{type(m).__qualname__}") for p, m in named.items()]
    lifted = lowerable(model)
    verdict = refusal_for(mods, lifted)
    if verdict is not None:
        detail = unlowered_reason(named[verdict[3]]) if verdict[0] in ("TS941", "TS942") else None
        _raise(name or type(model).__name__, verdict, _kinds(mods, lifted) - {verdict[1]}, detail,
               scan(model)[1])


def _stack_classes(graph_module: torch.fx.GraphModule):
    seen = set()
    for gm in [graph_module] + [m for m in graph_module.modules() if isinstance(m, torch.fx.GraphModule)]:
        for node in gm.graph.nodes:
            for path, cls in (node.meta.get("nn_module_stack") or {}).values():
                cls_name = cls if isinstance(cls, str) else f"{cls.__module__}.{cls.__qualname__}"
                key = (path, cls_name)       # qualified: torchvision's detection classes match with it
                if key not in seen:
                    seen.add(key)
                    yield key


def _lowered_in_graph(graph_module: torch.fx.GraphModule):
    """(path, class) of the linear and SSM layers that hold a torch.ops.truestar.linear_attention
    node: the layers hf_linattn lowered."""
    out = set()
    for gm in [graph_module] + [m for m in graph_module.modules() if isinstance(m, torch.fx.GraphModule)]:
        for node in gm.graph.nodes:
            if node.op != "call_function" or str(node.target) != "truestar.linear_attention.default":
                continue
            for path, cls in (node.meta.get("nn_module_stack") or {}).values():
                cls_name = (cls if isinstance(cls, str) else getattr(cls, "__name__", str(cls))).rsplit(".", 1)[-1]
                if classify(cls_name) in ("linear", "ssm"):
                    out.add((path, cls_name))
    return out


def check_exported(ep, name: str = "program") -> None:
    mods = list(_stack_classes(ep.graph_module))
    lifted = _lowered_in_graph(ep.graph_module)
    verdict = refusal_for(mods, lifted)
    if verdict is not None:
        detail = None
        if verdict[0] in ("TS941", "TS942"):
            from .hf_linattn import SPECS
            detail = ("the program was exported without Truestar's lowering (import the module with "
                      "truestar.ir.import_module, or export it inside truestar.ir.hf_linattn.lowered(model))"
                      if verdict[2] in SPECS else f"Truestar has no lowering of {verdict[2]}")
        _raise(name, verdict, _kinds(mods, lifted) - {verdict[1]}, detail)
