"""Ops and messages found by the text, retrieval, speech and vision language sweep (2026-11-24,
docs/MODEL_COVERAGE.md "Text, retrieval, speech and vision language models", scripts/text_coverage.py).

Lowered onto IR ops that already have a float64 reference, reverse, forward and batching rules and
kernels, so no new IR op and no new rule:

* ``_weight_norm(v, g, dim)``: ``v * g / ||v||``, the norm over every dim but ``dim`` (all dims for
  ``dim=-1``). wav2vec2 and HuBERT reparametrize their convolutional position embedding with it.
* ``sign``: ``(x > 0) - (x < 0)`` in the input's dtype (torch gives 0 for nan and +0 for -0, and so
  does this); ``frozen``, so its gradient is 0 as torch's. ``ceil``: ``-floor(-x)``. DeBERTa v2 and
  v3 bucket their relative positions on a log scale with both.

Two messages instead of a generic failure:

* a runtime check on a value read from a tensor (``torch._check`` of an ``.item()``): transformers
  checks that the image placeholder tokens match the image features this way in every vision
  language model. The IR has no runtime assertion, and leaving the check out would run an input the
  model rejects, so it is refused, naming the line and the switch transformers documents for it.
* torch.export failing on a data dependent size (``.tolist()``, ``.item()`` or a boolean selection
  inside the model): the user's line is named instead of torch's internal frame.
"""

from __future__ import annotations

import os
import traceback
from typing import Dict, List, Optional

import torch

from .core import UnsupportedOpError
from .importer import _describe, aten, source_line
from .importer_extra import _f, _op, _unsupported


@aten("aten._weight_norm.default")
def _weight_norm(c, a):
    v, g = c.tensor(a["v"]), c.tensor(a["g"])
    dim = a.get("dim", 0)
    r = v.type.rank
    if dim == -1:
        dims = list(range(r))
    else:
        dim %= r
        dims = [d for d in range(r) if d != dim]
    if v.type.dtype != g.type.dtype:
        _unsupported(c, f"weight norm with v in {v.type.dtype} and g in {g.type.dtype}")
    n = _op(c, "sqrt", _op(c, "sum", _op(c, "mul", v, v), dims=dims, keepdim=True)) if dims else _op(c, "abs", v)
    return _op(c, "mul", v, _op(c, "div", g, n))


@aten("aten.sign.default")
def _sign(c, a):
    x = c.tensor(a["self"])
    if x.type.dtype == torch.bool:
        return x
    pos = _op(c, "cast", _op(c, "gt", x, c.const(0, x)), dtype=x.type.dtype)
    neg = _op(c, "cast", _op(c, "lt", x, c.const(0, x)), dtype=x.type.dtype)
    s = _op(c, "sub", pos, neg)
    return _op(c, "frozen", s) if x.type.dtype.is_floating_point else s


@aten("aten.ceil.default")
def _ceil(c, a):
    from .importer import HANDLERS
    x = c.tensor(a["self"])
    if not x.type.dtype.is_floating_point:
        return x
    neg = _op(c, "neg", x)
    return _op(c, "neg", HANDLERS["aten.floor.default"](c, {"self": neg}))


@aten("aten.masked_scatter.default")
def _masked_scatter(c, a):
    """``self`` with the positions where ``mask`` is true filled, in row major order, by the leading
    elements of ``source``: the n-th true position (counted with a cumulative sum) reads element n
    of the flattened source. Vision language models merge the projected image features into the
    image placeholder positions of the text embeddings this way. Gradient: the self positions not
    filled, and to the source through the gather (scatter add)."""
    x, mask, src = c.tensor(a["self"]), c.tensor(a["mask"]), c.tensor(a["source"])
    shape = list(x.type.shape)
    if not all(isinstance(s, int) for s in shape + list(src.type.shape)):
        _unsupported(c, "masked_scatter with a symbolic size", "TS861")
    if mask.type.rank < x.type.rank:
        mask = _op(c, "reshape", mask, shape=[1] * (x.type.rank - mask.type.rank) + list(mask.type.shape))
    if list(mask.type.shape) != shape:
        mask = _op(c, "expand", mask, shape=shape)
    if mask.type.dtype != torch.bool:
        mask = _op(c, "ne", mask, c.const(0, mask))
    n = 1
    for s in shape:
        n *= s
    m = 1
    for s in src.type.shape:
        m *= s
    if m == 0:
        _unsupported(c, "masked_scatter with an empty source")
    flat = _op(c, "reshape", mask, shape=[n])
    pos = _op(c, "sub", _op(c, "cumsum", _op(c, "cast", flat, dtype=torch.int64), dim=0),
              c.const(1, _op(c, "cast", flat, dtype=torch.int64)))
    # before the first true position the count is -1: read element 0 there (not used). No upper clamp:
    # more true positions than source elements is an error in torch, and the out of range gather
    # raises here too instead of repeating an element
    pos = _op(c, "clamp", pos, min=0, max=None)
    s = _op(c, "reshape", src, shape=[m])
    if src.type.dtype != x.type.dtype:
        s = _op(c, "cast", s, dtype=x.type.dtype)
    filled = _op(c, "reshape", _op(c, "index_select", s, pos, dim=0), shape=shape)
    return _op(c, "where", mask, filled, x)


# ----------------------------------------------------------------------------- torch.vmap inside a model

_VMAP = {"_vmap_increment_nesting", "_vmap_decrement_nesting", "_add_batch_dim", "_remove_batch_dim",
         "lazy_load_decompositions"}


def has_vmap(ep) -> bool:
    """The exported graph keeps torch.vmap as functorch level markers (transformers builds every mask
    with a custom overlay, such as Gemma 3's bidirectional image blocks, through vmap). Rerunning the
    export's own ATen pass with an empty decomposition table removes them: the batched ops become
    plain ATen ops and every other op stays as exported."""
    return any(n.op == "call_function" and getattr(n.target, "__name__", "") in _VMAP
               and "functorch" in (getattr(n.target, "__module__", "") or "") for n in ep.graph.nodes)


# ----------------------------------------------------------------------------- runtime checks

_ASSERTS = ("aten._assert_scalar.default", "aten.sym_constrain_range_for_size.default",
            "aten._assert_async.msg", "aten._assert_async.default")
DISABLE_CHECK_ENV = "TRANSFORMERS_DISABLE_TORCH_CHECK"


def _is_tensor_node(n) -> bool:
    return isinstance(n, torch.fx.Node) and isinstance(n.meta.get("val"), torch.Tensor)


def runtime_checks(graph: torch.fx.Graph, target_name) -> Dict[torch.fx.Node, torch.fx.Node]:
    """Every node that only computes a runtime check on a value read from a tensor (the ``item`` and
    the Python number arithmetic and casts after it, and the assertion), mapped to that check's
    ``item`` node (the one the message names)."""
    chains = []
    for node in graph.nodes:
        if node.op != "call_function" or target_name(node) not in _ASSERTS:
            continue
        chain: List[torch.fx.Node] = [node]
        stack = [x for x in node.all_input_nodes]
        item = None
        while stack:
            n = stack.pop()
            if n in chain or _is_tensor_node(n) or n.op != "call_function":
                continue
            chain.append(n)
            if target_name(n) == "aten.item.default":
                item = item or n
                continue                      # its input is the tensor the value is read from
            stack.extend(n.all_input_nodes)
        if item is not None:
            chains.append((chain, item))
    # every node of the checks must feed only checks (a value also used elsewhere is real data); torch
    # adds its own range checks on the same .item(), so this is decided over all checks together
    every = {n for chain, _ in chains for n in chain}
    bad = {item for chain, item in chains if not all(set(n.users) <= every for n in chain)}
    out: Dict[torch.fx.Node, torch.fx.Node] = {}
    for chain, item in chains:
        if item not in bad:
            for n in chain:
                out[n] = item
    return out


def runtime_check_problem(item: torch.fx.Node) -> str:
    return (f"  runtime check on a value read from a tensor at {_describe(item)}: the model asserts a "
            "condition on a count computed from its inputs (torch._check of .item(); transformers checks "
            "this way that the image placeholder tokens match the image features). The IR has no runtime "
            "assertion and leaving the check out would run inputs the model rejects, so nothing is "
            f"imported. For a transformers model whose prompt layout you build, set {DISABLE_CHECK_ENV}=1 "
            "while exporting (it removes only the check; no value changes), or remove the check from "
            "the model")


# ----------------------------------------------------------------------------- torch.export failures

_DATA_DEPENDENT = ("GuardOnDataDependentSymNode", "PendingUnbackedSymbolNotFound")


def _model_frames(tb) -> List[traceback.FrameSummary]:
    """The frames outside torch and truestar.ir (the user's and the model library's), outermost first."""
    torch_dir = os.path.dirname(torch.__file__)
    ir_dir = os.path.dirname(__file__)
    frames = [f for f in traceback.extract_tb(tb)
              if not f.filename.startswith(torch_dir) and not f.filename.startswith(ir_dir)
              and "<frozen" not in f.filename and "/contextlib.py" not in f.filename.replace("\\", "/")]
    return frames


def explain_export_failure(err: BaseException, name: str) -> None:
    """Raise UnsupportedOpError naming the model line when torch.export failed on a data dependent
    size or branch; return (and let the caller re raise) for any other failure."""
    if type(err).__name__ not in _DATA_DEPENDENT:
        return
    frames = _model_frames(err.__traceback__)
    fr = frames[-1] if frames else None
    where = (f"{os.path.basename(fr.filename)}:{fr.lineno} in {fr.name}: `{(fr.line or '').strip()}`"
             if fr else "an unknown line")
    first = (str(err).strip().splitlines() or [""])[0][:200]
    # the size may have become unknown earlier (a boolean selection), so the calls leading here are named
    path = " < ".join(f"{os.path.basename(f.filename)}:{f.lineno} in {f.name}" for f in reversed(frames[-5:-1]))
    origin = f"; called from {path}" if path else ""
    raise UnsupportedOpError(
        f"torch.export cannot trace {name}: {where} reads a size or a branch condition from the values of "
        f"a tensor ({type(err).__name__}: {first}){origin}. torch.export traces with symbolic values, so that size "
        "is unknown; Truestar imports what torch.export traces, so nothing was imported. In the model: "
        "take the size as a Python int (fixed per program, for example an image grid), or keep the shape "
        "static (torch.where(mask, x, fill) instead of x[mask])") from err


__all__ = ["has_vmap", "runtime_checks", "runtime_check_problem", "explain_export_failure", "DISABLE_CHECK_ENV",
           "source_line"]
