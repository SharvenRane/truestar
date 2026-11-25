"""Port check (P16): does a second implementation compute the same model as the reference?

A port (a hand written reimplementation, a model converted back from JAX or MLX, a serving
engine's own model code, a merged adapter checkpoint) usually renames modules and changes layout,
so module names cannot be compared. Truestar infers the module map from the weights instead:

1. **Weight map.** Every parameter of the reference is matched to the candidate by content, not by
   name: equal bytes, a transpose, a row permutation (for example q and k rows permuted for a
   different rotary pairing), a row slice of a fused tensor (q, k and v inside one qkv weight, in
   either direction), a dtype cast, or, as a last resort, the closest tensor of the same shape
   (reported as a value difference, with an explanation when the difference is an adapter delta
   B @ A merged into the weight). Identical tensors (norm weights that are all ones) are paired
   by the order in which the forward pass first reads them. Weights with no counterpart, and
   weights the forward pass never reads, are reported loudly: that is the dropped LoRA case.
2. **Module map.** Reference module A maps to candidate module B when every matched weight under A
   lands under B and every matched weight under B comes from A. A module whose weights are only a
   slice of a fused candidate module (q_proj inside qkv) maps to it with an output slice. A
   reference module whose weights end up as plain parameters of a bigger candidate module (an
   RMSNorm class written inline in the block) is reported as inlined there.
3. **Isolated replay.** Both models run in float64 on CPU on the same inputs. Each mapped candidate
   module is then replayed alone with the reference module's inputs (each candidate input tensor is
   paired with the reference input of the same shape), so its error is its own, not inherited.
   Thresholds are calibrated per module: 10x the error of the reference module itself run in
   float32, never below 1e-9. A parent is replayed only when every mapped module inside it agrees,
   so the first module named is the one whose own computation differs.
4. **Op drilldown.** Inside the blamed pair, both sides are traced at the ATen op level, aligned by
   op name, and the first candidate op whose output is wrong for the reference op's inputs is
   named, together with the non tensor arguments that differ (a constant such as eps, a slice step,
   a rotary base) and the ops only one side runs (a missing scale, an extra cast).
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .capture import ROOT, Call, Recorder, flatten_tensors, map_tensors
from .codes import Issue
from .compare import DIVERGENT, OK, SUSPECT, Diff, diff_lists, worst

FLOOR = 1e-9  # both sides run in float64: anything above this and above float32 noise is a different model
NOISE_FACTOR = 10.0
FALLBACK_THRESHOLD = 1e-6  # when a module cannot be calibrated in float32
APPROX_MAX_REL = 0.5  # unrelated random tensors differ by about sqrt(2)
MERGE_RESIDUAL = 1e-3  # float32 rounding of a merged weight stays far below this
MAX_LIST = 8  # names listed per finding in messages


# ----------------------------------------------------------------------------- weight map


@dataclass
class Edge:
    """One reference parameter matched to one candidate parameter."""

    ref: str
    cand: str
    # equal, transpose, row_perm, slice_of_candidate (cand[a:b] == ref), slice_of_reference
    # (ref[a:b] == cand), cast, approx (values differ), merged (base weight with an adapter delta
    # folded in), adapter (an adapter factor A or B folded into cand)
    kind: str
    rows: Optional[Tuple[int, int]] = None
    perm: Optional[List[int]] = None  # cand rows = ref rows[perm]
    note: str = ""
    differs: bool = False  # matched by shape (or by position in a fused tensor), values differ
    merged: bool = False  # the difference is an adapter delta B @ A folded into the weight
    adapter: Optional[Tuple[str, str]] = None  # on a merged base edge: the (A, B) reference factors

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"reference": self.ref, "candidate": self.cand, "kind": self.kind}
        if self.rows is not None:
            d["rows"] = list(self.rows)
        if self.note:
            d["note"] = self.note
        if self.differs:
            d["values_differ"] = True
        if self.merged:
            d["merged_adapter"] = True
        return d


@dataclass
class WeightMap:
    edges: List[Edge]
    ref_unmatched: List[str]
    cand_unmatched: List[str]  # includes partially covered fused tensors, see cand_partial
    cand_partial: Dict[str, List[Tuple[int, int]]]  # rows of a fused candidate tensor with no source
    ref_unused: List[str]  # never read by the reference forward
    cand_unused: List[str]
    notes: List[str] = field(default_factory=list)

    def by_ref(self) -> Dict[str, List[Edge]]:
        out: Dict[str, List[Edge]] = defaultdict(list)
        for e in self.edges:
            out[e.ref].append(e)
        return out

    def by_cand(self) -> Dict[str, List[Edge]]:
        out: Dict[str, List[Edge]] = defaultdict(list)
        for e in self.edges:
            out[e.cand].append(e)
        return out

    def counts(self) -> Dict[str, int]:
        return dict(Counter(e.kind for e in self.edges))


def _bytes(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous().reshape(-1)
    if t.numel() == 0:
        return b""
    return t.view(torch.uint8).numpy().tobytes()


def _key(t: torch.Tensor) -> Tuple:
    return (str(t.dtype), tuple(t.shape), hashlib.blake2b(_bytes(t), digest_size=16).hexdigest())


def _row_hashes(t: torch.Tensor) -> List[str]:
    rows = t.detach().cpu().contiguous().reshape(t.shape[0], -1)
    return [hashlib.blake2b(_bytes(r), digest_size=12).hexdigest() for r in rows]


def _rows(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().reshape(t.shape[0], -1)


def _find_block(big: torch.Tensor, small: torch.Tensor, taken: List[Tuple[int, int]]) -> Optional[int]:
    """Offset o with big[o:o+len(small)] == small along dim 0, not overlapping `taken`."""
    if big.dtype != small.dtype or big.ndim != small.ndim or big.ndim == 0:
        return None
    if tuple(big.shape[1:]) != tuple(small.shape[1:]) or big.shape[0] <= small.shape[0]:
        return None
    n = small.shape[0]
    if n == 0:
        return None
    br, sr = _rows(big), _rows(small)
    hits = torch.nonzero((br == sr[0]).all(dim=1)).flatten().tolist()
    for o in hits:
        if o + n > big.shape[0]:
            continue
        if any(not (o + n <= a or o >= b) for a, b in taken):
            continue
        if torch.equal(br[o:o + n], sr):
            return o
    return None


def _rank(delta: torch.Tensor) -> Optional[int]:
    if delta.ndim != 2 or min(delta.shape) == 0:
        return None
    d = delta.detach().double()
    s = torch.linalg.svdvals(d)
    if s.numel() == 0 or float(s[0]) == 0.0:
        return 0
    return int((s > s[0] * 1e-10).sum())


def match_weights(ref_params: Dict[str, torch.Tensor], cand_params: Dict[str, torch.Tensor],
                  ref_order: Dict[str, int], cand_order: Dict[str, int]) -> WeightMap:
    """Match every reference parameter to the candidate by content (see the module docstring)."""
    edges: List[Edge] = []
    notes: List[str] = []
    ref_left = set(ref_params)
    cand_left = set(cand_params)
    inf = float("inf")
    rkey = lambda n: (ref_order.get(n, inf), n)
    ckey = lambda n: (cand_order.get(n, inf), n)

    # 1. Equal bytes. Identical tensors (all ones norms, tied embeddings) pair by first use order.
    groups_r: Dict[Tuple, List[str]] = defaultdict(list)
    groups_c: Dict[Tuple, List[str]] = defaultdict(list)
    for n, t in ref_params.items():
        groups_r[_key(t)].append(n)
    for n, t in cand_params.items():
        groups_c[_key(t)].append(n)
    for k in groups_r:
        if k not in groups_c:
            continue
        rs, cs = sorted(groups_r[k], key=rkey), sorted(groups_c[k], key=ckey)
        if len(rs) == 1:
            pairs = [(rs[0], c) for c in cs]
        elif len(cs) == 1:
            pairs = [(r, cs[0]) for r in rs]
        else:
            pairs = list(zip(rs, cs))
            if len(rs) != len(cs):
                notes.append(f"{len(rs)} identical reference tensors ({', '.join(rs[:3])}...) vs "
                             f"{len(cs)} in the candidate; paired by first use order, the rest "
                             "are unmatched.")
        for r, c in pairs:
            edges.append(Edge(r, c, "equal"))
            ref_left.discard(r)
            cand_left.discard(c)

    def cand_index() -> Dict[Tuple, List[str]]:
        idx: Dict[Tuple, List[str]] = defaultdict(list)
        for n in sorted(cand_left, key=ckey):
            idx[_key(cand_params[n])].append(n)
        return idx

    # 2. Transpose (Conv1D style [in, out] storage).
    idx = cand_index()
    for r in sorted(ref_left, key=rkey):
        t = ref_params[r]
        if t.ndim != 2:
            continue
        hits = [c for c in idx.get(_key(t.t()), []) if c in cand_left]
        if hits:
            edges.append(Edge(r, hits[0], "transpose"))
            ref_left.discard(r)
            cand_left.discard(hits[0])

    # 3. Row permutation (same rows in another order).
    sig_c: Dict[Tuple, List[str]] = defaultdict(list)
    rows_c: Dict[str, List[str]] = {}
    for c in sorted(cand_left, key=ckey):
        t = cand_params[c]
        if t.ndim >= 1 and t.shape[0] > 1:
            rows_c[c] = _row_hashes(t)
            sig_c[(str(t.dtype), tuple(t.shape), tuple(sorted(rows_c[c])))].append(c)
    for r in sorted(ref_left, key=rkey):
        t = ref_params[r]
        if t.ndim < 1 or t.shape[0] < 2:
            continue
        rh = _row_hashes(t)
        hits = [c for c in sig_c.get((str(t.dtype), tuple(t.shape), tuple(sorted(rh))), [])
                if c in cand_left]
        if not hits:
            continue
        c = hits[0]
        if len(set(rh)) != len(rh):
            continue  # repeated rows: the permutation is not defined, leave for later passes
        pos = {h: i for i, h in enumerate(rh)}
        perm = [pos[h] for h in rows_c[c]]
        edges.append(Edge(r, c, "row_perm", perm=perm,
                          note=f"rows permuted ({_perm_summary(perm)})"))
        ref_left.discard(r)
        cand_left.discard(c)

    # 4. Row slices of fused tensors, in both directions.
    cov_c: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for r in sorted(ref_left, key=rkey):
        t = ref_params[r]
        for c in sorted(cand_left, key=ckey):
            o = _find_block(cand_params[c], t, cov_c[c])
            if o is not None:
                edges.append(Edge(r, c, "slice_of_candidate", rows=(o, o + t.shape[0]),
                                  note=f"rows {o}:{o + t.shape[0]} of fused {c}"))
                cov_c[c].append((o, o + t.shape[0]))
                ref_left.discard(r)
                break
    for c, spans in cov_c.items():
        if spans and sum(b - a for a, b in spans) == cand_params[c].shape[0]:
            cand_left.discard(c)
    cov_r: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for c in sorted(cand_left, key=ckey):
        if cov_c.get(c):
            continue
        t = cand_params[c]
        for r in sorted(ref_left, key=rkey):
            o = _find_block(ref_params[r], t, cov_r[r])
            if o is not None:
                edges.append(Edge(r, c, "slice_of_reference", rows=(o, o + t.shape[0]),
                                  note=f"rows {o}:{o + t.shape[0]} of fused reference {r}"))
                cov_r[r].append((o, o + t.shape[0]))
                cand_left.discard(c)
                break
    for r, spans in cov_r.items():
        if spans and sum(b - a for a, b in spans) == ref_params[r].shape[0]:
            ref_left.discard(r)

    # 5. dtype cast (stored in a lower precision).
    for r in sorted(ref_left, key=rkey):
        t = ref_params[r]
        for c in sorted(cand_left, key=ckey):
            u = cand_params[c]
            if u.shape != t.shape or u.dtype == t.dtype:
                continue
            lo = u.dtype if u.element_size() < t.element_size() else t.dtype
            if torch.equal(t.detach().to(lo), u.detach().to(lo)):
                edges.append(Edge(r, c, "cast", note=f"stored as {_dt(u.dtype)}, reference "
                                                     f"{_dt(t.dtype)}"))
                ref_left.discard(r)
                cand_left.discard(c)
                break

    # 6. Closest tensor of the same shape: the same weight with different values. Rows of a fused
    # candidate tensor that no reference tensor filled exactly are tried the same way.
    approx: List[Tuple[Edge, torch.Tensor]] = []
    for c in sorted(cand_left, key=ckey):
        spans = cov_c.get(c)
        if not spans:
            continue
        for a, b in _gaps(sorted(spans), cand_params[c].shape[0]):
            u = cand_params[c].detach()[a:b]
            best, best_rel = None, APPROX_MAX_REL
            for r in sorted(ref_left, key=rkey):
                t = ref_params[r].detach()
                if tuple(t.shape) != tuple(u.shape) or t.numel() == 0:
                    continue
                den = float(torch.linalg.vector_norm(t.double()))
                rel = float(torch.linalg.vector_norm(u.double() - t.double())) / den if den else inf
                if rel < best_rel:
                    best, best_rel = r, rel
            if best is None:
                continue
            e = Edge(best, c, "slice_of_candidate", rows=(a, b), differs=True,
                     note=f"rows {a}:{b} of fused {c}, values differ, relative difference "
                          f"{best_rel:.2e}")
            edges.append(e)
            approx.append((e, u))
            cov_c[c].append((a, b))
            ref_left.discard(best)
        if sum(y - x for x, y in cov_c[c]) == cand_params[c].shape[0]:
            cand_left.discard(c)
    for r in sorted(ref_left, key=rkey):
        t = ref_params[r].detach().double()
        best, best_rel = None, APPROX_MAX_REL
        for c in sorted(cand_left, key=ckey):
            u = cand_params[c]
            if tuple(u.shape) != tuple(t.shape) or t.numel() == 0:
                continue
            den = float(torch.linalg.vector_norm(t))
            rel = float(torch.linalg.vector_norm(u.detach().double() - t)) / den if den else inf
            if rel < best_rel:
                best, best_rel = c, rel
        if best is None:
            continue
        u = cand_params[best].detach()
        note = f"values differ, relative difference {best_rel:.2e}"
        for lo in (torch.bfloat16, torch.float16):
            if ref_params[r].dtype != lo and torch.equal(ref_params[r].detach().to(lo).to(u.dtype), u):
                note = f"values equal the reference rounded to {_dt(lo)} (a converter cast)"
                break
        else:
            rk = _rank(u.double() - t)
            if rk is not None and rk < min(t.shape):
                note += f"; the difference has rank {rk} (a merged low rank delta?)"
        e = Edge(r, best, "approx", note=note, differs=True)
        edges.append(e)
        approx.append((e, u))
        ref_left.discard(r)
        cand_left.discard(best)

    # 7. An approx edge whose delta is c * B @ A for two unmatched reference tensors: a merged adapter.
    for e, u in approx:
        t, u = ref_params[e.ref].detach().double(), u.double()
        if t.ndim != 2:
            continue
        delta = u - t
        dn = float(torch.linalg.vector_norm(delta))
        if dn == 0.0:
            continue
        lefts = sorted(ref_left, key=rkey)
        found = None
        for b in lefts:
            B = ref_params[b].detach().double()
            if B.ndim != 2 or B.shape[0] != t.shape[0]:
                continue
            for a in lefts:
                A = ref_params[a].detach().double()
                if a == b or A.ndim != 2 or A.shape[1] != t.shape[1] or A.shape[0] != B.shape[1]:
                    continue
                ba = B @ A
                nn_ = float((ba * ba).sum())
                if nn_ == 0.0:
                    continue
                cf = float((delta * ba).sum()) / nn_
                if float(torch.linalg.vector_norm(delta - cf * ba)) <= MERGE_RESIDUAL * dn:
                    found = (a, b, cf)
                    break
            if found:
                break
        if found:
            a, b, cf = found
            if e.kind == "approx":
                e.kind = "merged"
            e.merged = True
            e.adapter = (a, b)
            where = e.cand + (f" rows {e.rows[0]}:{e.rows[1]}" if e.rows else "")
            e.note = f"{where} = reference + {cf:.6g} * {b} @ {a} (adapter merged into the weight)"
            edges.append(Edge(a, e.cand, "adapter", rows=e.rows, merged=True,
                              note="adapter factor folded into " + where))
            edges.append(Edge(b, e.cand, "adapter", rows=e.rows, merged=True,
                              note="adapter factor folded into " + where))
            ref_left.discard(a)
            ref_left.discard(b)

    partial = {}
    for c, spans in cov_c.items():
        if c in cand_left and spans:
            partial[c] = _gaps(sorted(spans), cand_params[c].shape[0])
    for r, spans in cov_r.items():
        if r in ref_left and spans:
            notes.append(f"reference {r}: rows {_gaps(sorted(spans), ref_params[r].shape[0])} "
                         "have no counterpart in the candidate.")
    return WeightMap(edges, sorted(ref_left, key=rkey), sorted(cand_left, key=ckey), partial,
                     [], [], notes)


def _gaps(spans: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
    out, pos = [], 0
    for a, b in spans:
        if a > pos:
            out.append((pos, a))
        pos = max(pos, b)
    if pos < n:
        out.append((pos, n))
    return out


def _perm_summary(perm: List[int]) -> str:
    head = ", ".join(str(p) for p in perm[:6])
    return f"candidate row i is reference row perm[i], perm = [{head}{', ...' if len(perm) > 6 else ''}]"


def _dt(d: torch.dtype) -> str:
    return str(d).replace("torch.", "")


# ----------------------------------------------------------------------------- use order


def _use_order(model: nn.Module, args: Tuple, kwargs: Dict) -> Dict[str, int]:
    """Parameter name -> index of the first op that reads it during forward (absent: never read)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    by_ptr: Dict[int, List[str]] = defaultdict(list)
    for n, p in model.named_parameters():
        if p.numel():
            by_ptr[p.untyped_storage().data_ptr()].append(n)
    order: Dict[str, int] = {}
    counter = [0]

    class Uses(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            for t in flatten_tensors((args, kwargs)):
                try:
                    ptr = t.untyped_storage().data_ptr()
                except Exception:
                    continue
                for n in by_ptr.get(ptr, ()):
                    order.setdefault(n, counter[0])
            counter[0] += 1
            return func(*args, **kwargs)

    with torch.no_grad(), Uses():
        model(*args, **kwargs)
    return order


# ----------------------------------------------------------------------------- module map


@dataclass
class ModulePair:
    ref: str  # module path, ROOT for the model itself
    cand: str
    ref_type: str
    cand_type: str
    view: str = "identity"  # identity, cand_slice, ref_slice, ref_perm
    rows: Optional[Tuple[int, int]] = None
    perm: Optional[List[int]] = None

    def as_dict(self) -> Dict[str, Any]:
        d = {"reference": self.ref, "candidate": self.cand, "reference_type": self.ref_type,
             "candidate_type": self.cand_type, "view": self.view}
        if self.rows is not None:
            d["rows"] = list(self.rows)
        return d


@dataclass
class ModuleMap:
    pairs: List[ModulePair]
    inlined: Dict[str, str]  # reference module -> candidate module holding its weights as plain params
    ref_unmapped: List[str]  # reference modules with weights but no counterpart
    cand_unmapped: List[str]
    cand_inlined: Dict[str, str] = field(default_factory=dict)  # the same, the other way round


def _closures(model: nn.Module) -> Tuple[Dict[str, FrozenSet[str]], Dict[str, FrozenSet[str]]]:
    canon = {id(p): n for n, p in model.named_parameters()}
    full, direct = {}, {}
    for name, mod in model.named_modules():
        key = name or ROOT
        full[key] = frozenset(canon[id(p)] for p in mod.parameters() if id(p) in canon)
        direct[key] = frozenset(canon[id(p)] for p in mod.parameters(recurse=False) if id(p) in canon)
    return full, direct


def _first_index(calls: List[Call]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in sorted(calls, key=lambda c: c.start):
        out.setdefault(c.name, c.start)
    return out


def build_module_map(ref: nn.Module, cand: nn.Module, wm: WeightMap, ref_calls: List[Call],
                     cand_calls: List[Call]) -> ModuleMap:
    fwd: Dict[str, set] = defaultdict(set)
    bwd: Dict[str, set] = defaultdict(set)
    for e in wm.edges:
        fwd[e.ref].add(e.cand)
        bwd[e.cand].add(e.ref)
    rfull, rdirect = _closures(ref)
    cfull, cdirect = _closures(cand)
    rclos = {m: frozenset(p for p in s if p in fwd) for m, s in rfull.items()}
    cclos = {m: frozenset(p for p in s if p in bwd) for m, s in cfull.items()}
    rtypes = {(n or ROOT): type(m).__name__ for n, m in ref.named_modules()}
    ctypes = {(n or ROOT): type(m).__name__ for n, m in cand.named_modules()}
    holders: Dict[str, List[str]] = defaultdict(list)  # cand param -> cand modules containing it
    for m, s in cclos.items():
        for p in s:
            holders[p].append(m)
    edge_kinds: Dict[Tuple[str, str], List[Edge]] = defaultdict(list)
    for e in wm.edges:
        edge_kinds[(e.ref, e.cand)].append(e)

    def covers(a: str, b: str) -> bool:
        sa, sb = rclos[a], cclos[b]
        return (all(fwd[p] & sb for p in sa) and all(bwd[t] & sa for t in sb))

    rstart, cstart = _first_index(ref_calls), _first_index(cand_calls)
    inf = float("inf")
    candidates: Dict[str, List[str]] = {}
    slice_pairs: List[ModulePair] = []
    for a, sa in rclos.items():
        if not sa:
            continue
        p0 = min(sa)
        pool = {b for t in fwd[p0] for b in holders.get(t, ())}
        ms = []
        for b in sorted(pool, key=lambda b: (cstart.get(b, inf), b)):
            if not covers(a, b):
                continue
            es = [e for p in sa for t in cclos[b] for e in edge_kinds.get((p, t), ())]
            kinds = {e.kind for e in es}
            if kinds == {"adapter"}:
                continue  # an adapter factor alone is not a module of the candidate
            if any(e.adapter and not set(e.adapter) <= sa for e in es):
                continue  # a base layer without its adapter cannot compute the merged weight
            if "slice_of_candidate" in kinds and kinds <= {"slice_of_candidate", "adapter"}:
                rows = {e.rows for e in es if e.kind == "slice_of_candidate"}
                if len(rows) == 1 and cdirect[b] == cclos[b]:
                    slice_pairs.append(ModulePair(a, b, rtypes[a], ctypes[b], "cand_slice",
                                                  rows=next(iter(rows))))
                continue
            if kinds == {"slice_of_reference"}:
                rows = {e.rows for e in es}
                if len(rows) == 1 and rdirect[a] == sa:
                    slice_pairs.append(ModulePair(a, b, rtypes[a], ctypes[b], "ref_slice",
                                                  rows=next(iter(rows))))
                continue
            ms.append(b)
        if ms:
            candidates[a] = ms

    # Pair ref modules that share the same candidate set in call order (outermost first for
    # wrapper chains, call order for tied copies). A candidate module is used once.
    groups: Dict[FrozenSet[str], List[str]] = defaultdict(list)
    for a, ms in candidates.items():
        groups[frozenset(ms)].append(a)
    used: set = set()
    pairs: List[ModulePair] = []
    for key, refs in groups.items():
        refs = sorted(refs, key=lambda a: (rstart.get(a, inf), a))
        cands = [b for b in sorted(key, key=lambda b: (cstart.get(b, inf), b)) if b not in used]
        for a, b in zip(refs, cands):
            used.add(b)
            pairs.append(_with_view(ModulePair(a, b, rtypes[a], ctypes[b]), rdirect[a],
                                    cdirect[b], rclos[a], cclos[b], edge_kinds))
    mapped_r = {p.ref for p in pairs} | {p.ref for p in slice_pairs}
    mapped_c = {p.cand for p in pairs} | {p.cand for p in slice_pairs}

    # Reference modules whose weights became plain parameters of a bigger candidate module.
    inlined: Dict[str, str] = {}
    for a, sa in rclos.items():
        if not sa or a in mapped_r:
            continue
        dst = set().union(*(fwd[p] for p in sa))
        owners = {m for m, s in cdirect.items() if dst & s}
        if len(owners) == 1:
            o = next(iter(owners))
            if dst <= cdirect[o]:
                inlined[a] = o
    cand_inlined: Dict[str, str] = {}
    for b, sb in cclos.items():
        if not sb or b in mapped_c:
            continue
        src = set().union(*(bwd[t] for t in sb))
        owners = {m for m, s in rdirect.items() if src & s}
        if len(owners) == 1:
            o = next(iter(owners))
            if src <= rdirect[o]:
                cand_inlined[b] = o
    ref_unmapped = [a for a, s in rclos.items() if s and a not in mapped_r and a not in inlined
                    and rdirect[a]]
    cand_unmapped = [b for b, s in cclos.items() if s and b not in mapped_c and b not in cand_inlined
                     and cdirect[b]]
    all_pairs = sorted(pairs + slice_pairs, key=lambda p: (rstart.get(p.ref, inf), p.ref,
                                                           cstart.get(p.cand, inf)))
    return ModuleMap(all_pairs, inlined, ref_unmapped, cand_unmapped, cand_inlined)


def _with_view(p: ModulePair, rdirect, cdirect, rclos, cclos, edge_kinds) -> ModulePair:
    """A leaf pair whose weights are row permuted compares outputs through the same permutation."""
    if rdirect != rclos or cdirect != cclos or not rdirect:
        return p
    es = [e for r in rdirect for c in cdirect for e in edge_kinds.get((r, c), ())]
    perms = {tuple(e.perm) for e in es if e.kind == "row_perm"}
    if es and len(perms) == 1 and all(e.kind == "row_perm" for e in es):
        p.view, p.perm = "ref_perm", list(next(iter(perms)))
    return p


# ----------------------------------------------------------------------------- results


@dataclass
class PairResult:
    pair: ModulePair
    ref_key: str
    cand_key: str
    own: Optional[Diff]  # candidate module alone with the reference inputs
    accumulated: Optional[Diff]  # inside the full candidate run
    threshold: float
    status: str
    replayed: bool
    own_code: bool  # the candidate module has children: the difference is in its own forward code
    error: str = ""
    inputs_kept: List[int] = field(default_factory=list)  # candidate inputs with no reference match

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.ref_key, "candidate": self.cand_key,
            "reference_type": self.pair.ref_type, "candidate_type": self.pair.cand_type,
            "view": self.pair.view, "status": self.status, "replayed": self.replayed,
            "own_code": self.own_code,
            "rel_err": _num(self.own.rel_err) if self.own else None,
            "accumulated_rel_err": _num(self.accumulated.rel_err) if self.accumulated else None,
            "threshold": self.threshold, "error": self.error or None,
        }


@dataclass
class OpFinding:
    op: Optional[str]  # first candidate op whose own output is wrong, or None
    index: int
    total: int
    rel_err: Optional[float]
    arg_diffs: List[Dict[str, Any]]  # aligned ops whose non tensor arguments differ
    only_reference: Dict[str, int]
    only_candidate: Dict[str, int]
    casts: List[str]  # dtypes of casts only the candidate runs
    explanation: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op, "index": self.index, "total": self.total, "rel_err": _num(self.rel_err),
            "argument_differences": self.arg_diffs, "only_in_reference": self.only_reference,
            "only_in_candidate": self.only_candidate, "extra_casts": self.casts,
            "explanation": self.explanation,
        }


@dataclass
class PortReport:
    reference_name: str
    candidate_name: str
    weights: WeightMap
    module_map: ModuleMap
    results: List[PairResult]
    output: Optional[Diff]
    output_threshold: float
    blamed: Optional[PairResult]
    op: Optional[OpFinding]
    explanation: str
    issues: List[Issue]
    notes: List[str]
    reference_params: int
    candidate_params: int
    blame_source: Optional[Dict[str, Any]] = None
    reference_source: Optional[Dict[str, Any]] = None

    @property
    def codes(self) -> List[str]:
        out: List[str] = []
        for i in self.issues:
            if i.code not in out:
                out.append(i.code)
        return out

    @property
    def status(self) -> str:
        from .codes import severity

        s = OK
        for i in self.issues:
            if severity(i.code) == "error":
                s = worst(s, DIVERGENT)
            elif severity(i.code) == "warning":
                s = worst(s, SUSPECT)
        return s

    def as_dict(self) -> Dict[str, Any]:
        wm, mm = self.weights, self.module_map
        return {
            "status": self.status,
            "codes": self.codes,
            "issues": [i.as_dict() for i in self.issues],
            "output": _diff_dict(self.output),
            "weights": {
                "reference_count": self.reference_params,
                "candidate_count": self.candidate_params,
                "matched": len(wm.edges),
                "kinds": wm.counts(),
                "reference_unmatched": wm.ref_unmatched,
                "candidate_unmatched": wm.cand_unmatched,
                "candidate_partial": {k: [list(s) for s in v] for k, v in wm.cand_partial.items()},
                "reference_unused": wm.ref_unused,
                "candidate_unused": wm.cand_unused,
                "edges": [e.as_dict() for e in wm.edges],
            },
            "module_map": {
                "pairs": [p.as_dict() for p in mm.pairs],
                "inlined": mm.inlined,
                "candidate_inlined": mm.cand_inlined,
                "reference_unmapped": mm.ref_unmapped,
                "candidate_unmapped": mm.cand_unmapped,
            },
            "blamed": self._blamed_dict(),
            "explanation": self.explanation,
            "modules_checked": sum(1 for r in self.results if r.replayed),
            "modules_over_threshold": [r.as_dict() for r in self.results if r.status != OK][:5],
            "notes": self.notes,
        }

    def _blamed_dict(self) -> Optional[Dict[str, Any]]:
        b = self.blamed
        if b is None:
            return None
        d = b.as_dict()
        d["ratio"] = _num(b.own.rel_err / b.threshold) if b.own and b.threshold else None
        d["inlined_reference_modules"] = sorted(
            a for a, o in self.module_map.inlined.items() if o == b.pair.cand)
        d["source"] = self.blame_source
        d["reference_source"] = self.reference_source
        d["op"] = self.op.as_dict() if self.op else None
        return d

    def to_text(self, top: int = 5) -> str:
        wm, mm = self.weights, self.module_map
        lines = [f"truestar port: reference {self.reference_name} vs candidate {self.candidate_name}"]
        kinds = ", ".join(f"{v} {k}" for k, v in sorted(wm.counts().items()))
        lines.append(f"weights: {self.reference_params} reference, {self.candidate_params} candidate, "
                     f"{len(wm.edges)} matched ({kinds or 'none'})")
        lines.append(f"module map: {len(mm.pairs)} pairs, {len(mm.inlined)} reference modules "
                     f"inlined, {len(mm.ref_unmapped)} reference and {len(mm.cand_unmapped)} "
                     "candidate modules unmapped")
        if self.output is not None:
            st = self.output.status(self.output_threshold)
            lines.append(f"final output: relative error {_fmt(self.output.rel_err)} "
                         f"({st.upper()})")
        lines.append(f"status: {self.status.upper()}")
        if self.explanation:
            lines.append("")
            lines.append(self.explanation)
        bad = [r for r in self.results if r.status != OK]
        if bad:
            lines.append("")
            lines.append("modules over threshold (own error with reference inputs):")
            for r in bad[:top]:
                own = _fmt(r.own.rel_err) if r.own else "n/a"
                lines.append(f"  {r.ref_key} -> {r.cand_key} ({r.pair.cand_type}) own {own} "
                             f"threshold {r.threshold:.1e} {r.status.upper()}")
        if self.issues:
            lines.append("")
            for i in self.issues:
                lines.append(f"[{i.code}] {i.message}")
        for n in self.notes:
            lines.append(f"note: {n}")
        return "\n".join(lines)


def _num(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _fmt(x: float) -> str:
    return f"{x:.2e}" if math.isfinite(x) else str(x)


def _diff_dict(d: Optional[Diff]) -> Optional[Dict[str, Any]]:
    if d is None:
        return None
    return {"rel_err": _num(d.rel_err), "max_abs": _num(d.max_abs),
            "nonfinite_mismatch": d.nonfinite_mismatch, "shape_mismatch": d.shape_mismatch}


# ----------------------------------------------------------------------------- running


def _to64(obj: Any) -> Any:
    return map_tensors(obj, lambda t: t.detach().to(torch.float64) if t.is_floating_point()
                       else t.detach())


def _to32(obj: Any) -> Any:
    return map_tensors(obj, lambda t: t.detach().to(torch.float32) if t.is_floating_point()
                       else t.detach())


def _clone(obj: Any) -> Any:
    return map_tensors(obj, torch.clone)


def _get(model: nn.Module, name: str) -> nn.Module:
    return model if name == ROOT else model.get_submodule(name)


def _record(model: nn.Module, args, kwargs):
    with torch.no_grad(), Recorder(model, keep_inputs=True) as rec:
        out = model(*_clone(args), **_clone(kwargs))
    return out, rec.calls


def _view(p: ModulePair, ref_out: List[torch.Tensor], cand_out: List[torch.Tensor]):
    """Bring the two output lists to the same layout, or None when the view does not apply."""
    if p.view == "identity":
        return ref_out, cand_out
    if not ref_out or not cand_out:
        return None
    r, c = ref_out[0], cand_out[0]
    if p.view == "cand_slice":
        a, b = p.rows
        if c.ndim == 0 or c.shape[-1] < b:
            return None
        return [r], [c[..., a:b]]
    if p.view == "ref_slice":
        a, b = p.rows
        if r.ndim == 0 or r.shape[-1] < b:
            return None
        return [r[..., a:b]], [c]
    if p.view == "ref_perm":
        if r.ndim == 0 or r.shape[-1] != len(p.perm):
            return None
        return [r[..., p.perm]], [c]
    return None


def _mapped_inputs(ref_call: Call, cand_call: Call):
    """The candidate call's arguments with each tensor replaced by the reference input of the same
    shape and dtype that is closest in value (equal when everything upstream agrees)."""
    ref_leaves = flatten_tensors((ref_call.args, ref_call.kwargs))
    kept: List[int] = []
    counter = [0]

    def pick(t: torch.Tensor) -> torch.Tensor:
        i = counter[0]
        counter[0] += 1
        same = [r for r in ref_leaves if r.shape == t.shape and r.dtype == t.dtype]
        if not same:
            kept.append(i)
            return t
        if len(same) == 1:
            return same[0]
        def dist(r):
            if not t.is_floating_point():
                return 0.0 if torch.equal(r, t) else 1.0
            return float(torch.linalg.vector_norm((r - t).double()))
        return min(same, key=dist)

    # flatten_tensors sorts dict keys, map_tensors does not: walk kwargs in sorted key order.
    args = map_tensors(cand_call.args, pick)
    kwargs = {k: map_tensors(cand_call.kwargs[k], pick) for k in sorted(cand_call.kwargs, key=str)}
    return args, kwargs, kept


def _replay(module: nn.Module, args, kwargs) -> List[torch.Tensor]:
    with torch.no_grad():
        out = module(*_clone(args), **_clone(kwargs))
    return [t.detach().cpu() for t in flatten_tensors(out)]


def port_check(reference: nn.Module, candidate: nn.Module, inputs: Any,
               kwargs: Optional[Dict[str, Any]] = None, reference_name: Optional[str] = None,
               candidate_name: Optional[str] = None, drill_ops: bool = True) -> PortReport:
    """Check that `candidate` computes the same model as `reference` (see the module docstring).

    Both models are copied, never modified, run in eval mode, in float64 on CPU, on `inputs`
    (a tensor or a tuple of positional arguments) and `kwargs`.
    """
    args = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    kwargs = dict(kwargs or {})
    rname = reference_name or type(reference).__name__
    cname = candidate_name or type(candidate).__name__
    notes: List[str] = []
    issues: List[Issue] = []

    ref_orig = {n: p.detach().cpu() for n, p in reference.named_parameters()}
    cand_orig = {n: p.detach().cpu() for n, p in candidate.named_parameters()}
    ref = copy.deepcopy(reference).cpu().eval()
    cand = copy.deepcopy(candidate).cpu().eval()
    try:
        ref, cand = ref.to(torch.float64), cand.to(torch.float64)
        a64, k64 = _to64(args), _to64(kwargs)
        ref_out, ref_calls = _record(ref, a64, k64)
        cand_out, cand_calls = _record(cand, a64, k64)
    except Exception as exc:
        issues.append(Issue("TS003", "float64 run failed, so both models run in their own dtype "
                                     f"and thresholds are looser: {type(exc).__name__}: {exc}"))
        ref = copy.deepcopy(reference).cpu().eval()
        cand = copy.deepcopy(candidate).cpu().eval()
        a64, k64 = args, kwargs
        ref_out, ref_calls = _record(ref, a64, k64)
        cand_out, cand_calls = _record(cand, a64, k64)

    ref_order = _use_order(ref, _clone(a64), _clone(k64))
    cand_order = _use_order(cand, _clone(a64), _clone(k64))
    wm = match_weights(ref_orig, cand_orig, ref_order, cand_order)
    wm.ref_unused = [n for n in ref_orig if n not in ref_order and ref_orig[n].numel()]
    wm.cand_unused = [n for n in cand_orig if n not in cand_order and cand_orig[n].numel()]
    notes.extend(wm.notes)
    mm = build_module_map(ref, cand, wm, ref_calls, cand_calls)

    # Weight findings.
    ref_unmatched = set(wm.ref_unmatched)
    cand_unmatched = set(wm.cand_unmatched)
    used_missing = [n for n in wm.ref_unmatched if n in ref_order]
    dead_ref = [n for n in wm.ref_unmatched if n not in ref_order]
    cand_extra = [n for n in wm.cand_unmatched if n in cand_order]
    dead_cand = [n for n in wm.cand_unmatched if n not in cand_order]
    if used_missing:
        issues.append(Issue("TS162", f"{len(used_missing)} reference weight(s) used by the reference "
                                     "forward have no counterpart in the candidate (dropped on "
                                     f"conversion?): {_names(used_missing)}"))
    if cand_extra:
        parts = []
        for n in cand_extra:
            if n in wm.cand_partial:
                gaps = ", ".join(f"{a}:{b}" for a, b in wm.cand_partial[n])
                parts.append(f"{n} rows {gaps}")
            else:
                parts.append(n)
        issues.append(Issue("TS162", f"{len(cand_extra)} candidate weight(s) the candidate forward "
                                     f"reads have no counterpart in the reference: {_names(parts)}"))
    by_ref, by_cand = wm.by_ref(), wm.by_cand()
    one_sided = []
    for n in wm.cand_unused:
        if n in cand_unmatched:
            continue
        srcs = [e.ref for e in by_cand.get(n, ()) if e.ref in ref_order]
        if srcs:
            one_sided.append(f"candidate {n} (reference {srcs[0]} is used)")
    for n in wm.ref_unused:
        if n in ref_unmatched:
            continue
        dsts = [e.cand for e in by_ref.get(n, ()) if e.cand in cand_order]
        if dsts:
            one_sided.append(f"reference {n} (candidate {dsts[0]} is used)")
    if one_sided:
        issues.append(Issue("TS163", "weight(s) present in both models but never read by one "
                                     f"forward pass (loaded but ignored?): {_names(one_sided, 4)}"))
    if dead_ref or dead_cand:
        issues.append(Issue("TS169", "weight(s) present in one model only and never read by its "
                                     f"forward: {_names(['reference ' + n for n in dead_ref] + ['candidate ' + n for n in dead_cand])}"))
    approx = [e for e in wm.edges if e.differs and not e.merged]
    if approx:
        issues.append(Issue("TS165", "weight(s) matched by shape only, values differ: "
                        + _names([f"{e.ref} -> {e.cand}: {e.note}" for e in approx], 4)))
    casts = [e for e in wm.edges if e.kind == "cast"]
    if casts:
        issues.append(Issue("TS165", "weight(s) stored in another dtype: "
                        + _names([f"{e.ref} -> {e.cand} ({e.note})" for e in casts], 4)))
    layout = [e for e in wm.edges if e.kind in ("transpose", "row_perm", "slice_of_candidate",
                                                "slice_of_reference", "merged")]
    if layout:
        issues.append(Issue("TS164", f"{len(layout)} weight(s) matched through a layout change "
                                     "(transpose, permutation, fused slice or merged adapter): "
                        + _names([f"{e.ref} -> {e.cand} ({'merged adapter' if e.merged else e.kind})"
                                  for e in layout], 4)))
    if mm.inlined or mm.ref_unmapped or mm.cand_unmapped or mm.cand_inlined:
        bits = []
        if mm.inlined:
            bits.append("reference modules whose weights live in a bigger candidate module "
                        "(written inline or merged): " + _names(
                [f"{a} -> {o}" for a, o in mm.inlined.items()], 4))
        if mm.cand_inlined:
            bits.append("candidate modules whose weights live in a bigger reference module: "
                        + _names([f"{b} -> {o}" for b, o in mm.cand_inlined.items()], 4))
        if mm.ref_unmapped:
            bits.append("reference modules with no counterpart: " + _names(mm.ref_unmapped, 4))
        if mm.cand_unmapped:
            bits.append("candidate modules with no counterpart: " + _names(mm.cand_unmapped, 4))
        issues.append(Issue("TS167", "; ".join(bits)))

    # Float32 copy of the reference, for per module thresholds.
    try:
        ref32 = copy.deepcopy(ref).to(torch.float32)
    except Exception:
        ref32 = None

    ref_by_name: Dict[str, List[Call]] = defaultdict(list)
    cand_by_name: Dict[str, List[Call]] = defaultdict(list)
    for c in ref_calls:
        ref_by_name[c.name].append(c)
    for c in cand_calls:
        cand_by_name[c.name].append(c)
    for lst in list(ref_by_name.values()) + list(cand_by_name.values()):
        lst.sort(key=lambda c: c.index)
    pairs_by_ref: Dict[str, List[ModulePair]] = defaultdict(list)
    for p in mm.pairs:
        pairs_by_ref[p.ref].append(p)
    cand_leaf = {(n or ROOT): next(iter(m.children()), None) is None for n, m in cand.named_modules()}
    noise_cache: Dict[str, float] = {}

    def threshold(c: Call) -> float:
        if c.key in noise_cache:
            return noise_cache[c.key]
        thr = FALLBACK_THRESHOLD
        if ref32 is not None:
            try:
                outs = _replay(_get(ref32, c.name), _to32(c.args), _to32(c.kwargs))
                noise = diff_lists(outs, c.outputs).rel_err
                if math.isfinite(noise):
                    thr = max(FLOOR, NOISE_FACTOR * noise)
            except Exception:
                pass
        noise_cache[c.key] = thr
        return thr

    results: List[PairResult] = []
    bad_key: Dict[str, bool] = {}
    bad_inside: Dict[str, bool] = {}
    ref_call_by_key = {c.key: c for c in ref_calls}
    mismatched_counts = []
    for rc in ref_calls:  # completion order: children before parents
        inner = any(bad_key.get(k, False) or bad_inside.get(k, False) for k in rc.children)
        bad_inside[rc.key] = inner
        for p in pairs_by_ref.get(rc.name, ()):
            cc_list = cand_by_name.get(p.cand, [])
            if rc.index >= len(cc_list):
                if rc.index == 0:
                    mismatched_counts.append(f"{p.cand} never ran")
                continue
            if rc.index == 0 and len(cc_list) != len(ref_by_name[rc.name]):
                mismatched_counts.append(f"{rc.name} ran {len(ref_by_name[rc.name])} times, "
                                         f"{p.cand} {len(cc_list)}")
            cc = cc_list[rc.index]
            thr = threshold(rc)
            viewed = _view(p, rc.outputs, cc.outputs)
            acc = diff_lists(viewed[1], viewed[0]) if viewed else None
            own_code = not cand_leaf.get(p.cand, True)
            if inner:
                results.append(PairResult(p, rc.key, cc.key, None, acc, thr, OK, False, own_code))
                continue
            cargs, ckw, kept = _mapped_inputs(rc, cc)
            err, own, status = "", None, OK
            try:
                outs = _replay(_get(cand, p.cand), cargs, ckw)
                v = _view(p, rc.outputs, outs)
                if v is None:
                    notes.append(f"{rc.key} -> {cc.key}: outputs cannot be compared through the "
                                 f"{p.view} view; skipped.")
                    continue
                own = diff_lists(v[1], v[0])
                status = own.status(thr)
            except Exception as exc:
                err, status = f"{type(exc).__name__}: {exc}", DIVERGENT
            res = PairResult(p, rc.key, cc.key, own, acc, thr, status, True, own_code, err, kept)
            results.append(res)
            if status != OK:
                bad_key[rc.key] = True
    if mismatched_counts:
        notes.append("call counts differ: " + "; ".join(mismatched_counts[:5]))

    order = {c.key: c.start for c in ref_calls}
    results.sort(key=lambda r: (order.get(r.ref_key, 0), r.cand_key))
    bad = [r for r in results if r.status != OK]
    divergent = [r for r in bad if r.status == DIVERGENT]
    blamed = divergent[0] if divergent else (bad[0] if bad else None)

    ref_flat = [t.detach().cpu() for t in flatten_tensors(ref_out)]
    cand_flat = [t.detach().cpu() for t in flatten_tensors(cand_out)]
    output = diff_lists(cand_flat, ref_flat)
    out_thr = threshold(ref_call_by_key[f"{ROOT}#0"]) if f"{ROOT}#0" in ref_call_by_key else FALLBACK_THRESHOLD
    out_status = output.status(out_thr)

    op = None
    explanation = ""
    blame_source = reference_source = None
    if blamed is not None:
        rc = ref_call_by_key[blamed.ref_key]
        cc = next(c for c in cand_calls if c.key == blamed.cand_key)
        if drill_ops and not blamed.error:
            cargs, ckw, _ = _mapped_inputs(rc, cc)
            try:
                op = drill_pair(_get(ref, rc.name), _get(cand, cc.name), rc.args, rc.kwargs,
                                cargs, ckw, blamed.threshold)
            except Exception as exc:
                notes.append(f"op drilldown failed on {cc.key}: {type(exc).__name__}: {exc}")
        try:
            from .agent import module_source

            blame_source = module_source(cand, cc.name)
            reference_source = module_source(ref, rc.name)
        except Exception:
            pass
        explanation = _explain(blamed, op, mm, wm, ref, cand, rname, cname)
        code = "TS160" if blamed.status == DIVERGENT else "TS161"
        own = _fmt(blamed.own.rel_err) if blamed.own else "n/a"
        more = len(bad) - 1
        issues.insert(0, Issue(code, f"candidate {blamed.cand_key} ({blamed.pair.cand_type}) computes "
                                     f"differently from reference {blamed.ref_key} "
                                     f"({blamed.pair.ref_type}) on the same inputs: own error "
                                     f"{own} vs threshold {blamed.threshold:.1e}"
                                     + (f"; {more} more module(s) over threshold" if more else "")))
        if blamed.error:
            issues.insert(1, Issue("TS168", f"candidate {blamed.cand_key} raised when replayed with "
                                            f"the reference inputs: {blamed.error}"))
    elif out_status != OK:
        issues.insert(0, Issue("TS166", "the final outputs differ (relative error "
                                        f"{_fmt(output.rel_err)}) but no mapped module is to blame: "
                                        "the difference is in code outside the mapped modules, or "
                                        "in modules the map could not pair (see TS167)."))
        explanation = ("The outputs differ but every mapped module agrees on its own inputs. Look "
                       "at code outside modules and at the unmapped modules listed under TS167.")
    elif used_missing or cand_extra or one_sided:
        explanation = "Every mapped module agrees, but the weight map is incomplete (see TS162, TS163)."
    return PortReport(rname, cname, wm, mm, results, output, out_thr, blamed, op, explanation,
                      issues, notes, len(ref_orig), len(cand_orig), blame_source, reference_source)


def _names(items: Sequence[str], limit: int = MAX_LIST) -> str:
    items = list(items)
    s = ", ".join(items[:limit])
    return s + (f" and {len(items) - limit} more" if len(items) > limit else "")


def _explain(b: PairResult, op: Optional[OpFinding], mm: ModuleMap, wm: WeightMap, ref: nn.Module,
             cand: nn.Module, rname: str, cname: str) -> str:
    own = _fmt(b.own.rel_err) if b.own else "n/a"
    s = (f"{b.status.upper()}: candidate {b.pair.cand} ({b.pair.cand_type}) computes differently "
         f"from reference {b.pair.ref} ({b.pair.ref_type}) given the same inputs "
         f"(own error {own}, threshold {b.threshold:.1e}).")
    if b.pair.view == "cand_slice":
        s += f" Compared on output columns {b.pair.rows[0]}:{b.pair.rows[1]} of the fused module."
    if b.own_code:
        s += " Every mapped module inside it agrees, so the difference is in its own forward code."
        inl = sorted(a for a, o in mm.inlined.items() if o == b.pair.cand)
        if inl:
            s += " It holds the weights of reference module(s) written inline here: " + _names(inl, 4) + "."
    if b.error:
        s += f" It raised when replayed: {b.error}."
    # Weights inside the blamed pair that have no counterpart, or are not read.
    rfull, _ = _closures(ref)
    cfull, _ = _closures(cand)
    rset, cset = rfull.get(b.pair.ref, frozenset()), cfull.get(b.pair.cand, frozenset())
    miss = [n for n in wm.ref_unmatched if n in rset]
    extra = [n for n in wm.cand_unmatched if n in cset]
    unused = [n for n in wm.cand_unused if n in cset]
    if miss:
        s += " Reference weights used here have no counterpart in the candidate: " + _names(miss, 4) + "."
    if extra:
        s += " Candidate weights here have no counterpart in the reference: " + _names(extra, 4) + "."
    if unused:
        s += " Candidate weights here are never read by its forward: " + _names(unused, 4) + "."
    diffs = [e for e in wm.edges if e.ref in rset and (e.differs or e.kind == "cast")]
    for e in diffs[:2]:
        s += f" Weight {e.ref} -> {e.cand}: {e.note}."
    if op is not None and op.explanation:
        s += " " + op.explanation
    if b.inputs_kept:
        s += (f" Candidate input(s) {b.inputs_kept} had no reference input of the same shape and "
              "kept the candidate's own values.")
    return s


# ----------------------------------------------------------------------------- op drilldown


# Ops that only move or reshape data. Fusion changes them freely, so the text explanation leaves
# them out of the "only on one side" lists (the JSON keeps every op).
_LAYOUT = {
    "aten.view", "aten._unsafe_view", "aten.t", "aten.transpose", "aten.expand", "aten.clone",
    "aten.unsqueeze", "aten.squeeze", "aten.reshape", "aten.permute", "aten.split",
    "aten.split_with_sizes", "aten.chunk", "aten.alias", "aten.detach", "aten.contiguous",
    "aten.select", "aten.slice", "aten.unbind", "aten.flatten", "aten._reshape_alias",
}


def _layout(name: str) -> bool:
    return name.rsplit(".", 1)[0] in _LAYOUT


def _slice_reading(c: List[Any], r: List[Any]) -> str:
    """aten.slice.Tensor(self, dim, start, end, step): say which elements each side takes."""
    def parts(v):
        v = list(v) + [None] * (4 - len(v))
        dim, start, end, step = v[:4]
        return dim, start or 0, end, step or 1
    cd, cs, ce, cst = parts(c)
    rd, rs, re_, rst = parts(r)
    if cst != rst:
        return (f"the candidate takes every {cst} element(s) from {cs} where the reference takes "
                f"every {rst} from {rs}" + (" to " + str(re_) if re_ is not None and re_ < 2**62 else "")
                + ": the elements are paired differently, as when rotary pairs are interleaved "
                  "on one side and split in halves on the other")
    if (cs, ce) != (rs, re_):
        return f"the candidate slices {cs}:{ce} where the reference slices {rs}:{re_}"
    return ""


def _scalars(obj: Any) -> List[Any]:
    out: List[Any] = []

    def walk(o: Any) -> None:
        if isinstance(o, torch.Tensor):
            return
        if isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for k in sorted(o, key=str):
                walk(o[k])
        elif isinstance(o, (int, float, bool, complex, str, torch.dtype, torch.device,
                            torch.memory_format, torch.layout)) or o is None:
            out.append(o)

    walk(obj)
    return out


def _show(v: Any) -> str:
    if isinstance(v, int) and not isinstance(v, bool) and v >= 2 ** 62:
        return "end"  # slice end meaning "to the end"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, torch.dtype):
        return _dt(v)
    return str(v)


def drill_pair(ref_mod: nn.Module, cand_mod: nn.Module, ref_args, ref_kwargs, cand_args,
               cand_kwargs, threshold: float) -> OpFinding:
    """Name what differs inside one blamed pair, at the ATen op level (both on CPU float64)."""
    from .opdrill import _align, _comparable, _isolate, trace_ops

    rops, t1 = trace_ops(ref_mod, _clone(ref_args), _clone(ref_kwargs))
    cops, t2 = trace_ops(cand_mod, _clone(cand_args), _clone(cand_kwargs))
    if t1 or t2:
        return OpFinding(None, 0, 0, None, [], {}, {}, [], "Op drilldown skipped: too many ops.")
    rops = [r for r in rops if _comparable(r)]
    cops = [r for r in cops if _comparable(r)]
    pairs = _align([r.name for r in rops], [r.name for r in cops])

    arg_diffs: List[Dict[str, Any]] = []
    raw: Dict[int, Tuple[List[Any], List[Any]]] = {}
    for i, j in sorted(pairs.items()):
        sc, sr = _scalars((cops[i].args, cops[i].kwargs)), _scalars((rops[j].args, rops[j].kwargs))
        if sc != sr:
            raw[i + 1] = (sc, sr)
            arg_diffs.append({"op": cops[i].name, "index": i + 1,
                              "candidate": [_show(v) for v in sc],
                              "reference": [_show(v) for v in sr]})
    only_r = Counter(r.name for r in rops) - Counter(c.name for c in cops)
    only_c = Counter(c.name for c in cops) - Counter(r.name for r in rops)
    casts = []
    for i, c in enumerate(cops):
        if i not in pairs and "_to_copy" in c.name and isinstance(c.kwargs.get("dtype"), torch.dtype):
            casts.append(_dt(c.kwargs["dtype"]))

    first, first_err = None, None
    for i, d in enumerate(cops):
        r = rops[pairs[i]] if i in pairs else None
        acc = diff_lists(d.outputs, r.outputs) if r is not None else None
        if acc is not None and acc.status(threshold) == OK:
            continue
        got = _isolate(d, r, "cpu", True, threshold, torch.float64)
        if got is None:
            continue
        iso, thr = got[0], got[1]
        if iso.status(thr) == OK:
            continue
        first, first_err = i, iso.rel_err
        break

    parts: List[str] = []
    if first is not None:
        d = cops[first]
        msg = (f"First differing op: {d.name} (op {first + 1} of {len(cops)} in the candidate), "
               f"own error {_fmt(first_err)} for the reference op's inputs")
        ad = next((a for a in arg_diffs if a["index"] == first + 1), None)
        if ad is not None:
            msg += (f"; its non tensor arguments are [{', '.join(ad['candidate'])}] where the "
                    f"reference has [{', '.join(ad['reference'])}]")
            const = (_slice_reading(*raw[first + 1]) if "aten.slice" in d.name
                     else _const_change(*raw[first + 1]))
            if const:
                msg += f" ({const})"
        parts.append(msg + ".")
    others = [a for a in arg_diffs if (first is None or a["index"] != first + 1)
              and not _layout(a["op"])]
    if others and first is None:
        a = others[0]
        parts.append(f"Op {a['op']} (op {a['index']}) has non tensor arguments "
                     f"[{', '.join(a['candidate'])}] where the reference has "
                     f"[{', '.join(a['reference'])}].")
    shown_r = [(k, v) for k, v in sorted(only_r.items()) if not _layout(k)]
    shown_c = [(k, v) for k, v in sorted(only_c.items()) if not _layout(k)]
    if shown_r:
        parts.append("Compute ops only in the reference: "
                     + ", ".join(f"{k} x{v}" for k, v in shown_r[:6]) + ".")
    if shown_c:
        parts.append("Compute ops only in the candidate: "
                     + ", ".join(f"{k} x{v}" for k, v in shown_c[:6]) + ".")
    if casts:
        uniq = list(dict.fromkeys(casts))
        msg = "The candidate runs casts the reference does not: to " + ", to ".join(uniq)
        low = [d for d in ("float8_e4m3fn", "float8_e5m2", "bfloat16", "float16", "float32")
               if d in uniq]
        if low:
            msg += f" (a round trip through {low[0]} loses precision)"
        parts.append(msg + ".")
    if first is None and not arg_diffs and not only_r and not only_c:
        parts.append("Both sides run the same ops with the same arguments; the difference comes "
                     "from tensor values, such as weights or buffers that differ.")
    return OpFinding(cops[first].name if first is not None else None,
                     (first + 1) if first is not None else 0, len(cops), first_err, arg_diffs,
                     dict(only_r), dict(only_c), casts, " ".join(parts))


def _const_change(c: List[Any], r: List[Any]) -> str:
    """A short reading of a single changed constant, from the raw argument values."""
    if len(c) != len(r):
        return ""
    ch = [(x, y) for x, y in zip(c, r) if x != y]
    if len(ch) != 1:
        return ""
    x, y = ch[0]
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))) or isinstance(x, bool):
        return f"{_show(x)} instead of {_show(y)}"
    fx, fy = float(x), float(y)
    x, y = _show(x), _show(y)
    if fy not in (0.0, 1.0) and fx != 0 and abs(fx - fy * fy) < 1e-6 * abs(fx):
        return f"constant {x} instead of {y}: {x} is {y} squared, a missing square root?"
    return f"constant {x} instead of {y}"


# ----------------------------------------------------------------------------- command line


class PortFileError(Exception):
    """A model file is missing, fails to import, or lacks get_model() (TS020)."""


def _import(path: str, alias: str):
    if not os.path.isfile(path):
        raise PortFileError(f"{path} not found")
    stem = os.path.splitext(os.path.basename(path))[0]
    name = stem if stem.isidentifier() and stem not in sys.modules else alias
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    d = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:
        sys.path.insert(0, d)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        raise PortFileError(f"importing {path} failed: {type(exc).__name__}: {exc}") from exc
    if not hasattr(mod, "get_model"):
        raise PortFileError(f"{path} must define get_model()")
    return mod


def load_pair(reference_path: str, candidate_path: str):
    """(reference, candidate, inputs, kwargs) from two model files.

    The reference file defines get_model() and get_inputs() (and optionally get_kwargs()). The
    candidate file defines get_model(), or get_model(reference) to convert the reference's weights,
    which is what a port's converter does. When the candidate also defines get_inputs(), the
    reference's inputs are still used, so both see the same tensors.
    """
    rmod = _import(reference_path, "truestar_port_reference")
    cmod = _import(candidate_path, "truestar_port_candidate")
    if not hasattr(rmod, "get_inputs"):
        raise PortFileError(f"{reference_path} must define get_inputs()")
    try:
        reference = rmod.get_model()
        inputs = rmod.get_inputs()
        kwargs = rmod.get_kwargs() if hasattr(rmod, "get_kwargs") else None
    except Exception as exc:
        raise PortFileError(f"calling get_model() / get_inputs() in {reference_path} failed: "
                            f"{type(exc).__name__}: {exc}") from exc
    try:
        n = len([p for p in inspect.signature(cmod.get_model).parameters.values()
                 if p.default is inspect.Parameter.empty and p.kind in
                 (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
        candidate = cmod.get_model(reference) if n >= 1 else cmod.get_model()
    except Exception as exc:
        raise PortFileError(f"calling get_model() in {candidate_path} failed: "
                            f"{type(exc).__name__}: {exc}") from exc
    return reference, candidate, inputs, kwargs


def report_to_agent(report: PortReport, reference_file: Optional[str],
                    candidate_file: Optional[str]) -> Dict[str, Any]:
    from .agent import _envelope, exit_code_for

    doc = _envelope(report.status, exit_code_for(report.status))
    doc["command"] = "port"
    body = report.as_dict()
    doc.update({
        "reference": {"name": report.reference_name,
                      "file": os.path.abspath(reference_file) if reference_file else None},
        "candidate": {"name": report.candidate_name,
                      "file": os.path.abspath(candidate_file) if candidate_file else None},
        "precision": "CPU float64",
    })
    doc.update(body)
    doc["next_actions"] = next_actions(report, reference_file, candidate_file)
    return doc


def next_actions(report: PortReport, reference_file: Optional[str] = None,
                 candidate_file: Optional[str] = None) -> List[Dict[str, Any]]:
    acts: List[Dict[str, Any]] = []
    b = report.blamed
    wm = report.weights
    if b is not None:
        src = report.blame_source or {}
        user = src.get("user") or src.get("module")
        if user:
            acts.append({"kind": "inspect_source", "file": user["file"], "line": user["line"],
                         "command": None,
                         "description": f"Read {user['qualname']} (candidate {b.pair.cand}) next to "
                                        f"the reference {b.pair.ref} ({b.pair.ref_type}); "
                                        "they differ on the same inputs."})
        rs = (report.reference_source or {}).get("user") or (report.reference_source or {}).get("module")
        if rs:
            acts.append({"kind": "inspect_reference", "file": rs["file"], "line": rs["line"],
                         "command": None,
                         "description": f"The reference implementation is {rs['qualname']}."})
        if report.op is not None and report.op.explanation:
            acts.append({"kind": "inspect_op", "command": None, "op": report.op.op,
                         "description": report.op.explanation})
    used_missing = [n for n in wm.ref_unmatched if n not in wm.ref_unused]
    if used_missing:
        acts.append({"kind": "load_missing_weights", "command": None, "weights": used_missing[:20],
                     "description": "The reference uses these weights and the candidate has no "
                                    "counterpart. Check the converter's key mapping and any "
                                    "load_state_dict(strict=False) that hides missing keys, or "
                                    "merge the adapter into the base weight."})
    unused = [n for n in wm.cand_unused if n not in wm.cand_unmatched]
    if unused:
        acts.append({"kind": "use_or_remove_weights", "command": None, "weights": unused[:20],
                     "description": "These candidate weights match reference weights but the "
                                    "candidate forward never reads them: the code path that applies "
                                    "them is missing."})
    if "TS166" in report.codes:
        acts.append({"kind": "wrap_in_module", "command": None,
                     "description": "Outputs differ but no mapped module is to blame. Move the "
                                    "functional code between modules into modules, or map the "
                                    "unmapped modules, and rerun."})
    if acts and reference_file and candidate_file:
        acts.append({"kind": "rerun", "description": "Rerun after a fix; exit code 0 means the "
                                                     "port computes the same model.",
                     "command": f'truestar port "{os.path.abspath(reference_file)}" '
                                f'"{os.path.abspath(candidate_file)}" --format agent'})
    return acts


def cli(a) -> int:
    """`truestar port reference.py candidate.py` (called from truestar.cli)."""
    import contextlib
    import json

    from .agent import dumps, error_to_agent

    if a.candidate.startswith(("gguf:", "jax:", "keras:")):
        from .engines.cli import cli as engine_cli  # docs/PORT_ENGINES.md

        return engine_cli(a)
    agent = a.format == "agent"
    quiet = contextlib.redirect_stdout(sys.stderr) if agent else contextlib.nullcontext()
    failure = None
    with quiet:
        try:
            reference, candidate, inputs, kwargs = load_pair(a.reference, a.candidate)
            report = port_check(reference, candidate, inputs, kwargs,
                                reference_name=os.path.basename(a.reference),
                                candidate_name=os.path.basename(a.candidate))
        except PortFileError as exc:
            failure = ("TS020", str(exc))
        except Exception as exc:
            failure = ("TS021", f"{type(exc).__name__}: {exc}")
    if failure is not None:
        if agent:
            doc = error_to_agent(failure[0], failure[1], a.reference)
            doc["command"] = "port"
            print(dumps(doc))
        else:
            print(f"error [{failure[0]}]: {failure[1]}", file=sys.stderr)
        return 2
    if agent:
        print(dumps(report_to_agent(report, a.reference, a.candidate)))
    else:
        print(report.to_text(top=a.top))
    if getattr(a, "json_path", None):
        with open(a.json_path, "w", encoding="utf-8") as f:
            f.write(dumps(report_to_agent(report, a.reference, a.candidate)))
    return 0 if report.status == OK else 1
