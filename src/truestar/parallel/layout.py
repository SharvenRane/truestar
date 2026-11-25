"""Parallel layouts and tensor parallel plans for `truestar parallel` (docs/PARALLEL.md).

A layout names how one training step is spread over ranks: `tp` tensor parallel ranks (Linear
weights split by column or row), `dp` data parallel replicas (the batch split, gradients summed),
`ga` gradient accumulation micro batches per replica, `sp` sequence parallel (activations between a
row layer and the next column layer split along the sequence) and `pp` pipeline stages (only 1 is
built). Ranks are numbered tp first: rank = dp_rank * tp + tp_rank. `shard` keeps state sharded over
the dp replicas: 1 each replica holds a slice of the optimizer state and updates only that slice of
the parameters (gradients reduce scattered, parameters all gathered after the update), 2 also holds
only that slice of the parameters between steps (all gathered for the step's compute).

A plan maps module names to a style:

* `column` (or `column:K` for a weight fused from K equal parts, such as gate and up, or q, k and v):
  an nn.Linear whose output features are split over the tp ranks; each part is split separately.
* `row`: an nn.Linear whose input features are split; partial outputs are summed over tp, the bias is
  added once after the sum.
* `scatter`: with sp, the module's output is split along the sequence (the residual stream enters
  the sequence parallel region here).
* `gather`: with sp, the module's output is gathered along the sequence (the stream leaves it).
* `replicate`: every tp rank holds the whole module (the default for every unnamed module).
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class LayoutError(ValueError):
    """A layout or plan that cannot run (TS680). The message says what to change."""


@dataclasses.dataclass(frozen=True)
class Layout:
    tp: int = 1
    dp: int = 1
    pp: int = 1
    ga: int = 1
    sp: bool = False
    seq_dim: int = 1  # batch first models: [batch, sequence, features]
    shard: int = 0    # 0 replicated state, 1 optimizer state sharded over dp, 2 parameters too

    @property
    def world(self) -> int:
        return self.tp * self.dp

    def text(self) -> str:
        text = f"tp={self.tp},dp={self.dp},pp={self.pp},ga={self.ga},sp={int(self.sp)}"
        return text + (f",shard={self.shard}" if self.shard else "")

    def tp_rank(self, rank: int) -> int:
        return rank % self.tp

    def dp_rank(self, rank: int) -> int:
        return rank // self.tp

    def tp_group(self, rank: int) -> Tuple[int, ...]:
        d = self.dp_rank(rank)
        return tuple(d * self.tp + j for j in range(self.tp))

    def dp_group(self, rank: int) -> Tuple[int, ...]:
        t = self.tp_rank(rank)
        return tuple(i * self.tp + t for i in range(self.dp))


def parse_layout(text: str) -> Layout:
    """`tp=2,dp=2,pp=1,ga=2,sp=1,shard=1` (also `tp:2`). Unknown keys and values below 1 are refused."""
    vals: Dict[str, int] = {}
    for part in (p.strip() for p in str(text).split(",")):
        if not part:
            continue
        sep = "=" if "=" in part else ":" if ":" in part else None
        if sep is None:
            raise LayoutError(f"layout part {part!r} is not key=value (example: tp=2,dp=2,ga=2,sp=1)")
        k, v = (s.strip() for s in part.split(sep, 1))
        if k not in ("tp", "dp", "pp", "ga", "sp", "seq_dim", "shard"):
            raise LayoutError(f"unknown layout key {k!r}; known: tp, dp, pp, ga, sp, seq_dim, shard")
        if k in vals:
            raise LayoutError(f"layout key {k!r} given twice")
        try:
            vals[k] = int(v)
        except ValueError:
            raise LayoutError(f"layout {k}={v!r} is not an integer") from None
    for k in ("tp", "dp", "pp", "ga"):
        if vals.get(k, 1) < 1:
            raise LayoutError(f"layout {k}={vals[k]} must be at least 1")
    if vals.get("sp", 0) not in (0, 1):
        raise LayoutError("layout sp must be 0 or 1")
    if vals.get("shard", 0) not in (0, 1, 2):
        raise LayoutError("layout shard must be 0 (replicated), 1 (optimizer state sharded over dp) or 2 "
                          "(parameters sharded too)")
    lay = Layout(tp=vals.get("tp", 1), dp=vals.get("dp", 1), pp=vals.get("pp", 1), ga=vals.get("ga", 1),
                 sp=bool(vals.get("sp", 0)), seq_dim=vals.get("seq_dim", 1), shard=vals.get("shard", 0))
    if lay.shard and lay.dp == 1:
        raise LayoutError(f"shard={lay.shard} shards state over the dp replicas, so it needs dp of at least 2")
    if lay.pp != 1:
        raise LayoutError(f"pp={lay.pp}: pipeline parallel is not built; only pp=1 runs (nothing is "
                          "approximated by another layout)")
    if lay.sp and lay.tp == 1:
        raise LayoutError("sp=1 needs tp of at least 2 (sequence parallel splits the sequence over the "
                          "tensor parallel ranks)")
    if lay.world > 16:
        raise LayoutError(f"{lay.world} ranks: simulated layouts are capped at 16 ranks")
    return lay


@dataclasses.dataclass(frozen=True)
class Style:
    kind: str  # column, row, scatter, gather, replicate
    parts: int = 1

    def text(self) -> str:
        return f"{self.kind}:{self.parts}" if self.parts != 1 else self.kind


KINDS = ("column", "row", "scatter", "gather", "replicate")


def parse_style(s) -> Style:
    if isinstance(s, Style):
        return s
    text = str(s).strip()
    kind, _, parts = text.partition(":")
    if kind not in KINDS:
        raise LayoutError(f"plan style {text!r}: known styles are {', '.join(KINDS)} (column:K for a "
                          "weight fused from K parts)")
    if parts and kind != "column":
        raise LayoutError(f"plan style {text!r}: only column takes a part count")
    try:
        n = int(parts) if parts else 1
    except ValueError:
        raise LayoutError(f"plan style {text!r}: the part count must be an integer") from None
    if n < 1:
        raise LayoutError(f"plan style {text!r}: the part count must be at least 1")
    return Style(kind, n)


Plan = Dict[str, Style]


def leaf_order(model: nn.Module, inputs) -> List[str]:
    """Names of leaf modules in the order their forward finishes, from one no grad call."""
    order: List[str] = []
    handles = []
    for name, m in model.named_modules():
        if name and not list(m.children()):
            handles.append(m.register_forward_hook(lambda mod, a, o, n=name: order.append(n)))
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for h in handles:
            h.remove()
    return order


def auto_plan(model: nn.Module, inputs, tp: int) -> Plan:
    """Only pairs that are unambiguous: two nn.Linear leaves of the same parent, consecutive among the
    leaves that own parameters, the first widening (in != out) and the second taking exactly its
    output width back. That is an MLP up and down projection. Square projections (attention q, k, v
    and out) are never guessed: q followed by k has the same shapes as a real pair, so a guess
    could be wrong, and a wrong plan made by Truestar would be a false alarm. Name them in
    get_parallel_plan()."""
    mods = dict(model.named_modules())
    order = [n for n in leaf_order(model, inputs) if any(True for _ in mods[n].parameters(recurse=False))]
    plan: Plan = {}
    i = 0
    while i + 1 < len(order):
        a, b = mods[order[i]], mods[order[i + 1]]
        pa, pb = order[i].rpartition(".")[0], order[i + 1].rpartition(".")[0]
        if (isinstance(a, nn.Linear) and isinstance(b, nn.Linear) and pa == pb and order[i] not in plan
                and a.out_features != a.in_features and b.in_features == a.out_features
                and a.out_features % tp == 0):
            plan[order[i]], plan[order[i + 1]] = Style("column"), Style("row")
            i += 2
        else:
            i += 1
    return plan


def validate_plan(model: nn.Module, plan: Plan, layout: Layout) -> Plan:
    mods = dict(model.named_modules())
    out: Plan = {}
    for name, st in plan.items():
        st = parse_style(st)
        if name not in mods or name == "":
            raise LayoutError(f"plan names module {name!r}, which the model does not have")
        m = mods[name]
        if st.kind in ("column", "row"):
            if not isinstance(m, nn.Linear):
                raise LayoutError(f"plan: {name} is {type(m).__name__}; column and row need an nn.Linear")
            if st.kind == "column" and m.out_features % (layout.tp * st.parts):
                raise LayoutError(f"plan: {name} has {m.out_features} output features, not divisible by "
                                  f"tp={layout.tp} times {st.parts} parts")
            if st.kind == "row" and m.in_features % layout.tp:
                raise LayoutError(f"plan: {name} has {m.in_features} input features, not divisible by "
                                  f"tp={layout.tp}")
        if st.kind in ("scatter", "gather") and not layout.sp:
            raise LayoutError(f"plan: {name} is {st.kind}, which only means something with sp=1")
        out[name] = st
    if layout.tp > 1 and not any(s.kind in ("column", "row") for s in out.values()):
        raise LayoutError(f"tp={layout.tp} but the plan splits no Linear: define get_parallel_plan() "
                          "naming column and row layers")
    if layout.sp and not any(s.kind == "scatter" for s in out.values()):
        raise LayoutError("sp=1 needs one module marked scatter in the plan (where the residual stream "
                          "becomes sequence parallel), and one marked gather before the loss")
    return out


# ----------------------------------------------------------------------------- shards


def _part_rows(n: int, parts: int, tp: int, rank: int) -> List[slice]:
    size = n // parts
    per = size // tp
    return [slice(p * size + rank * per, p * size + (rank + 1) * per) for p in range(parts)]


def shard(t: torch.Tensor, style: Style, pname: str, tp: int, rank: int) -> torch.Tensor:
    """The local piece of a full parameter of a module with this style (pname: weight or bias)."""
    if style.kind == "column":
        return torch.cat([t[s] for s in _part_rows(t.shape[0], style.parts, tp, rank)], dim=0).clone()
    if style.kind == "row" and pname == "weight":
        return t.chunk(tp, dim=1)[rank].clone()
    return t.clone()


def is_sharded(style: Optional[Style], pname: str) -> bool:
    return style is not None and (style.kind == "column" or (style.kind == "row" and pname == "weight"))


def unshard(pieces: List[torch.Tensor], style: Style, pname: str) -> torch.Tensor:
    """Inverse of shard over the tp ranks' pieces (in tp rank order)."""
    tp = len(pieces)
    if style.kind == "column":
        parts = style.parts
        per = pieces[0].shape[0] // parts
        return torch.cat([pieces[r][p * per:(p + 1) * per] for p in range(parts) for r in range(tp)], dim=0)
    if style.kind == "row" and pname == "weight":
        return torch.cat(pieces, dim=1)
    raise ValueError("not a sharded parameter")
