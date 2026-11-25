"""``truestar export``: import a model, prove the artifact matches PyTorch, then write it.

The artifact that gets verified is the artifact that gets written: the imported program is
serialized, parsed back from those bytes, and the parsed program is what runs against PyTorch. Only
if every check passes are the same program bytes written, together with the evidence.

Checks (all in float64, the IR reference interpreter against the PyTorch module itself):

1. the example inputs the user gave;
2. when the batch dim is symbolic, a second batch of a different size built from the example
   rows (reversed order, cycled, with small noise on float inputs), so a batch size baked into the
   graph shows up as a failure here rather than in production.

Python branches that torch.export would bake in are refused with the line of the branch.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as _dt
import hashlib
import inspect
import linecache
import os
import platform
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .. import ir
from ..ir import serialize

DEFAULT_TOLERANCE = 1e-9  # relative to the largest output magnitude; measured agreement is about 1e-15
DEFAULT_MAX_BATCH = 1024  # only the default size scale for picking the second batch; the batch is unbounded
BATCH_SYMBOL = "B"
EVIDENCE_KEY = "truestar_export"


class ExportError(Exception):
    """Export was refused. The message says why, where in the user's code, and what to try."""


class VerificationError(ExportError):
    """The imported program disagrees with PyTorch. No artifact was written."""

    def __init__(self, message: str, checks: List[Dict[str, Any]]):
        super().__init__(message)
        self.checks = checks


@dataclass
class ExportResult:
    path: Optional[str]
    program: ir.Program
    program_hash: str
    evidence: Dict[str, Any]
    artifact: bytes = field(repr=False, default=b"")

    def to_text(self) -> str:
        ev = self.evidence
        lines = [f"exported {ev['model']} -> {self.path or '(not written)'}",
                 f"  program hash {self.program_hash}",
                 f"  inputs: " + ", ".join(f"%{v.name}: {v.type}" for v in self.program.inputs),
                 f"  outputs: " + ", ".join(str(v.type) for v in self.program.outputs)]
        if ev.get("batch_range"):
            lo, hi = ev["batch_range"]
            lines.append(f"  batch symbol {ev['batch_symbol']} in [{lo}, {'inf' if hi is None else hi}]")
        elif not self.program.symbols:
            lines.append("  batch: fixed")
        if ev.get("dynamic_dims") or (self.program.symbols and not ev.get("batch_range")):
            lines.append("  symbols: " + ", ".join(f"{k} in [{lo}, {'inf' if hi is None else hi}]"
                                                   for k, (lo, hi) in self.program.symbols.items()))
        for c in ev["checks"]:
            lines.append(f"  check {c['name']}: shapes {c['input_shapes']}, max rel error {c['max_rel_error']:.2e} "
                         f"(bar {ev['tolerance']:.0e}) {'PASS' if c['passed'] else 'FAIL'}")
        for b in ev.get("branches", []):
            other = b.get("other_side")
            extra = (f", {other['side']} side never taken: checked against torch's branch, max rel error "
                     f"{other['max_rel_error']:.2e}" if other else "")
            lines.append(f"  branch {b['where']}: then taken {b['then_taken']}, else taken {b['else_taken']}{extra}")
        for w in ev.get("warnings", []):
            lines.append(f"  WARNING weak check, {w}")
        lines.append(f"  torch {ev['torch_version']}, truestar {ev['truestar_version']}, {ev['created_utc']}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------- helpers


def _as_args(args: Any) -> Tuple[Any, ...]:
    """The positional arguments: one tensor, or a tuple or list of arguments (each a tensor or a
    nested structure of tensors)."""
    if isinstance(args, torch.Tensor):
        return (args,)
    if isinstance(args, dict):
        raise ExportError("args is a dict; pass keyword arguments with kwargs=..., or wrap it: args=(d,)")
    return tuple(args)


def inputs_sha256(inputs: Sequence[torch.Tensor]) -> str:
    """sha256 over dtype, shape and raw bytes of every input, in order."""
    h = hashlib.sha256()
    for t in inputs:
        t = t.detach().cpu().contiguous()
        h.update(f"{t.dtype}|{list(t.shape)}|".encode())
        if t.numel():
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _to64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().double() if t.is_floating_point() else t.detach()


def _flatten(out: Any) -> List[Any]:
    from torch.utils._pytree import tree_flatten
    return tree_flatten(out)[0]


def _compare(got: Sequence[torch.Tensor], want: Sequence[Any]) -> Tuple[float, float, str]:
    """(max abs error, max relative error, problem). Relative error is per output, scaled by its largest magnitude."""
    if len(got) != len(want):
        return float("inf"), float("inf"), f"the program has {len(got)} outputs, PyTorch returned {len(want)}"
    max_abs = max_rel = 0.0
    for i, (g, w) in enumerate(zip(got, want)):
        if not isinstance(w, torch.Tensor):
            return float("inf"), float("inf"), f"PyTorch output {i} is a {type(w).__name__}, not a tensor"
        if tuple(g.shape) != tuple(w.shape):
            return float("inf"), float("inf"), f"output {i} has shape {list(g.shape)}, PyTorch gives {list(w.shape)}"
        if not w.is_floating_point():
            if not torch.equal(g.to(w.dtype), w):
                return float("inf"), float("inf"), f"integer or bool output {i} differs"
            continue
        g, w = g.double(), w.double()
        nan_g, nan_w = torch.isnan(g), torch.isnan(w)
        if not torch.equal(nan_g, nan_w):
            return float("inf"), float("inf"), f"output {i} has NaN in different places"
        g, w = torch.where(nan_g, 0.0, g), torch.where(nan_w, 0.0, w)
        inf_mask = torch.isinf(w) | torch.isinf(g)
        if inf_mask.any():
            if not torch.equal(g[inf_mask], w[inf_mask]):
                return float("inf"), float("inf"), f"output {i} has inf in different places"
            g, w = torch.where(inf_mask, 0.0, g), torch.where(inf_mask, 0.0, w)
        if w.numel() == 0:
            continue
        err = (g - w).abs().max().item()
        scale = max(w.abs().max().item(), 1e-30)
        max_abs = max(max_abs, err)
        max_rel = max(max_rel, err / scale)
    return max_abs, max_rel, ""


def _weakness(outputs: Sequence[Any], tolerance: float) -> str:
    """Why a passing check proves little: every float output is zero, or does not depend on the input row.

    Rows count as identical when they differ by less than the tolerance (relative to the output's
    largest magnitude): an input dependent error that small could not fail the check either.
    """
    floats = [o for o in outputs if isinstance(o, torch.Tensor) and o.is_floating_point() and o.numel()]
    if not floats:
        return ""
    if all(o.abs().max().item() == 0 for o in floats):
        return "every output is exactly zero, so this check cannot tell a right program from a wrong one"
    def same_rows(o):
        if o.dim() == 0 or o.shape[0] < 2:
            return False
        o = o.double()
        return (o - o[:1]).abs().max().item() <= tolerance * max(o.abs().max().item(), 1e-300)
    if all(same_rows(o) for o in floats):
        return ("the outputs are the same for every batch row (within the tolerance), so they do not depend "
                "on the input and this check cannot catch an input dependent error")
    return ""


def _second_batch(args: Sequence[torch.Tensor], b2: int) -> Tuple[torch.Tensor, ...]:
    """A batch of size b2 from the example rows: reversed, cycled, plus small noise on float inputs."""
    g = torch.Generator().manual_seed(0)
    out = []
    for a in args:
        b = a.shape[0]
        idx = torch.tensor([(b - 1 - i) % b for i in range(b2)])
        x = a.detach()[idx].clone()
        if x.is_floating_point():
            std = a.detach().double().std().item() if a.numel() > 1 else 1.0
            noise = torch.randn(x.shape, generator=g, dtype=torch.float64) * (0.05 * (std or 1.0))
            x = (x.double() + noise).to(a.dtype)
        out.append(x)
    return tuple(out)


def _pick_second_batch(b: int, lo: int, hi: Optional[int]) -> Optional[int]:
    hi = DEFAULT_MAX_BATCH if hi is None else hi
    for cand in (7, b + 5, b + 1, b - 1, lo):
        if cand != b and lo <= cand <= hi and cand >= 1:
            return cand
    return None


# ----------------------------------------------------------------------------- export failures


_TORCH_DIR = os.path.dirname(torch.__file__).replace("\\", "/").lower() + "/"


def _user_frame_from_message(msg: str) -> Optional[str]:
    m = re.search(r'The following call raised this error:\s*\n\s*File "([^"]+)", line (\d+), in (\S+)\n\s*(.*)', msg)
    if not m:
        return None
    path, line, fn, code = m.groups()
    return f"{os.path.basename(path)}:{line} in {fn}: {code.strip()}"


def _draft_guard_frames(model, args, dynamic_shapes) -> List[Tuple[str, str]]:
    """Use torch.export.draft_export to find which user lines added guards. Best effort."""
    try:
        from torch._logging.structured import INTERN_TABLE
        ep = torch.export.draft_export(model, tuple(args), dynamic_shapes=dynamic_shapes)
        report = ep._report
    except Exception:
        return []
    names = {v: k for k, v in INTERN_TABLE.items()}
    found = []
    for f in report.failures:
        if f.failure_type.name not in ("GUARD_ADDED", "DATA_DEPENDENT_ERROR"):
            continue
        data = f.data
        expr = str(data.get("expr", ""))
        for s, src in (data.get("symbol_to_sources") or {}).items():
            expr = re.sub(rf"\b{re.escape(str(s))}\b", str(src), expr)
        frames = data.get("user_stack") or []
        for fr in reversed(frames):
            path = names.get(fr.get("filename"), str(fr.get("filename")))
            if isinstance(path, str) and path.replace("\\", "/").lower().startswith(_TORCH_DIR):
                continue
            code = (fr.get("loc") or "").strip()
            if not code and isinstance(path, str) and os.path.isfile(path):
                code = linecache.getline(path, int(fr.get("line", 0))).strip()
            where = f"{os.path.basename(str(path))}:{fr.get('line')} in {fr.get('name')}: {code}"
            if (where, expr) not in found:
                found.append((where, expr))
            break
    return found


def _explain_export_failure(exc: Exception, name: str, model, args, dynamic_shapes) -> ExportError:
    msg = str(exc)
    kind = type(exc).__name__
    if "data-dependent" in msg or "DataDependent" in kind:
        where = _user_frame_from_message(msg)
        if where is None:
            frames = _draft_guard_frames(model, args, dynamic_shapes)
            where = frames[0][0] if frames else "(torch.export did not say where)"
        return ExportError(
            f"export refused: {name} has a Python branch on a tensor's value, which torch.export cannot "
            f"capture; exporting it would bake in whichever side the example inputs took.\n"
            f"  branch: {where}\n"
            "No artifact was written. Fix: compute both sides and pick with torch.where, or use torch.cond "
            "(torch.export captures it and truestar imports it as an IR cond with both sides verified), "
            "or write the function with @truestar.function, which keeps real if and loops.")
    if "Constraints violated" in msg:
        frames = _draft_guard_frames(model, args, dynamic_shapes)
        guards = [ln.strip()[2:] for ln in msg.splitlines() if ln.strip().startswith("- ")]
        lines = [f"export refused: {name} has code that only works for some batch sizes, so torch.export "
                 "cannot keep the batch dim symbolic; exporting it would freeze the batch or bake in a branch."]
        for where, expr in frames:
            lines.append(f"  branch: {where}" + (f"  (guard: {expr})" if expr else ""))
        if not frames:
            lines.append("  branch: (torch.export did not say which line; run torch.export.draft_export to see it)")
        for gd in guards:
            lines.append(f"  torch.export: {gd}")
        lines.append("No artifact was written. Fix: make that code batch independent, export a fixed batch "
                     "(--static-batch, dynamic_batch=False), or limit --max-batch to the range torch.export "
                     "suggests above.")
        return ExportError("\n".join(lines))
    first = "\n".join(msg.strip().splitlines()[:12])
    return ExportError(f"export refused: torch.export failed on {name} ({kind}):\n{first}\nNo artifact was written.")


def _control_flow_sites(ep) -> List[Tuple[str, str]]:
    """Higher order control flow nodes (cond, while_loop, map, scan) with the user line that made them."""
    sites = []
    for node in ep.graph.nodes:
        if node.op != "call_function" or "higher_order" not in type(node.target).__module__:
            continue
        frames = re.findall(r'File "([^"]+)", line (\d+), in (\S+)\n\s*(.*)', node.meta.get("stack_trace") or "")
        user = [f for f in frames if not f[0].startswith("<")
                and not f[0].replace("\\", "/").lower().startswith(_TORCH_DIR)]
        if user:
            path, line, fn, code = user[-1]
            where = f"{os.path.basename(path)}:{line} in {fn}: {code.strip()}"
        else:
            where = f"node %{node.name}"
        sites.append((str(node.target), where))
    return sites


# ----------------------------------------------------------------------------- export


def _is_tensor(x) -> bool:
    return isinstance(x, torch.Tensor)


def _tree64(obj):
    from torch.utils._pytree import tree_map
    return tree_map(lambda t: _to64(t) if isinstance(t, torch.Tensor) else t, obj)


def _call_tree(args: Tuple[Any, ...], kwargs: Dict[str, Any]):
    return (tuple(args), dict(kwargs))


def _resize_axis(x: torch.Tensor, axis: int, new: int, gen: torch.Generator) -> torch.Tensor:
    """x with dim ``axis`` resized to ``new`` entries, cycled from the example in reverse order, with
    small noise on floats (same recipe as the second batch)."""
    n = x.shape[axis]
    idx = torch.tensor([(n - 1 - i) % n for i in range(new)])
    y = x.detach().index_select(axis, idx).clone()
    if y.is_floating_point():
        std = x.detach().double().std().item() if x.numel() > 1 else 1.0
        y = (y.double() + torch.randn(y.shape, generator=gen, dtype=torch.float64) * (0.05 * (std or 1.0))).to(x.dtype)
    return y


def _register_dataclasses(model, args, kwargs) -> None:
    """torch.export needs every dataclass in the inputs or outputs registered as a pytree node.
    Truestar does it for the user: the input types are found in the call, the output types by one
    eager call of the model on the example."""
    from torch.utils._pytree import SUPPORTED_NODES

    found = []

    def visit(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            found.append(type(obj))
            for f in dataclasses.fields(obj):
                visit(getattr(obj, f.name))
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                visit(x)
        elif isinstance(obj, dict):
            for x in obj.values():
                visit(x)

    visit((args, kwargs))
    with torch.no_grad():
        visit(copy.deepcopy(model)(*args, **kwargs))
    for cls in dict.fromkeys(found):
        if cls not in SUPPORTED_NODES:
            torch.export.register_dataclass(cls, serialized_type_name=f"{cls.__module__}.{cls.__qualname__}")


def _walk_conds(prog: ir.Program):
    """Every cond node of a program, its regions included, in a fixed order."""
    for n in prog.nodes:
        if n.op == "cond":
            yield n
        for v in n.attrs.values():
            if isinstance(v, ir.Program):
                yield from _walk_conds(v)


class _BranchRecorder:
    """Counts the sides every cond takes while the checks run, and keeps the first operands seen so
    the side never taken can be run on them afterwards."""

    def __init__(self):
        self.seen: Dict[int, Dict[str, Any]] = {}

    def __call__(self, attrs, taken, operands, bindings):
        rec = self.seen.setdefault(id(attrs["then_branch"]), {"then": 0, "else": 0, "first": {}})
        rec["then" if taken else "else"] += 1
        rec["first"].setdefault(taken, ([t.clone() for t in operands], dict(bindings or {})))


def _verify_branches(shipped: ir.Program, imported: ir.Program, rec: _BranchRecorder, tolerance: float):
    """Branch coverage of the checks, and a check of each side the inputs never took: the IR region
    is run on the operands that reached the cond and compared with torch's own captured subgraph
    for that side, in float64. Returns (records, problems, warnings)."""
    from ..ir import importer, reference

    pairs = list(zip(_walk_conds(shipped), _walk_conds(imported)))
    out, problems, warnings = [], [], []
    for n_ship, n_imp in pairs:
        r = rec.seen.get(id(n_ship.attrs["then_branch"]), {"then": 0, "else": 0, "first": {}})
        where = n_ship.source or f"%{n_ship.output.name}"
        entry = {"where": where, "then_taken": r["then"], "else_taken": r["else"], "other_side": None}
        if not r["then"] and not r["else"]:
            warnings.append(f"the branch at {where} never ran in any check")
        elif not r["then"] or not r["else"]:
            side = "then" if not r["then"] else "else"
            ops, bindings = r["first"][side == "else"]
            region_s = n_ship.attrs[f"{side}_branch"]
            gm = importer.TORCH_REGIONS.get(id(n_imp.attrs[f"{side}_branch"]))
            if gm is None:
                warnings.append(f"the {side} side of the branch at {where} was never taken and has no torch "
                                "subgraph to check it against")
            else:
                try:
                    got = reference._run(region_s, ops, bindings)
                    with torch.no_grad():
                        want = gm(*[t.double() if t.is_floating_point() else t for t in ops])
                    want = list(want) if isinstance(want, (tuple, list)) else [want]
                    max_abs, max_rel, problem = _compare(got, want)
                except Exception as e:  # the side may fail on operands meant for the other side
                    max_rel, problem = float("inf"), f"{type(e).__name__}: {e}"
                passed = not problem and max_rel <= tolerance
                entry["other_side"] = {"side": side, "max_rel_error": max_rel, "passed": passed, "problem": problem}
                if not passed:
                    problems.append(f"the {side} side of the branch at {where}, never taken by the checks, disagrees "
                                    f"with torch's captured {side} branch ({problem or f'max rel error {max_rel:.3e}'})")
        out.append(entry)
    return out, problems, warnings


def export(model: torch.nn.Module, args: Any, output: Optional[str] = None, *, kwargs: Optional[Dict[str, Any]] = None,
           dynamic_batch: bool = True, max_batch: Optional[int] = None,
           dynamic_dims: Optional[Dict[str, Dict[int, str]]] = None, second_batch: Optional[int] = None,
           tolerance: float = DEFAULT_TOLERANCE, decompose: bool = False, name: Optional[str] = None) -> ExportResult:
    """Export ``model`` called as ``model(*args, **kwargs)`` to a verified artifact.

    args, kwargs: the example call. Arguments may be tensors or nested tuples, lists, dicts,
      NamedTuples and dataclasses of tensors; the structure of inputs and outputs is stored in the
      artifact and ``load(path)(...)`` takes and returns the same structure.
    dynamic_batch: keep dim 0 of every input tensor symbolic as ``B`` (in [1, max_batch], or with no
      upper bound when max_batch is None). Every input tensor must then have the same dim 0 size, at
      least 2 (torch.export specializes size 1 dims).
    dynamic_dims: more symbolic dims, by forward parameter name: ``{"x": {1: "T"}}``. Each is
      verified at a second length too.
    second_batch: batch size for the second check (default: 7, or a nearby size inside the range).
    tolerance: maximum relative error (per output, against its largest magnitude) between the IR
      reference and PyTorch, both in float64.
    output: path to write. None runs every check and returns the bytes without writing.

    Branches (``torch.cond``) are verified on both sides: a side no check takes is run on the
    operands that reached it and compared with torch's own captured branch.

    Raises ExportError (or VerificationError) and writes nothing when anything fails.
    """
    from .. import pytree
    from ..ir import importer, reference

    args = _as_args(args)
    kwargs = dict(kwargs or {})
    name = name or type(model).__name__
    dynamic_dims = {k: dict(v) for k, v in (dynamic_dims or {}).items()}
    try:
        leaves, call_spec = pytree.flatten(_call_tree(args, kwargs), _is_tensor)
    except pytree.StructureError as e:
        raise ExportError(f"export refused: the example call of {name} is not a structure of tensors: {e}") from None
    if not leaves:
        raise ExportError(f"export refused: the example call of {name} has no tensor inputs")
    try:
        bound = inspect.signature(model.forward).bind(*args, **kwargs)
    except TypeError as e:
        raise ExportError(f"export refused: the example call does not match {name}.forward: {e}") from None
    for pname, axes in dynamic_dims.items():
        if not isinstance(bound.arguments.get(pname), torch.Tensor):
            raise ExportError(f"dynamic_dims names {pname!r}, which is not a tensor argument of {name}.forward "
                              f"(arguments: {list(bound.arguments)})")
        for ax in axes:
            nd = bound.arguments[pname].dim()
            if not -nd <= ax < nd or (dynamic_batch and ax % nd == 0):
                raise ExportError(f"dynamic_dims {pname}:{ax} is not an axis of {pname} {list(bound.arguments[pname].shape)}"
                                  + (" other than the batch axis 0" if dynamic_batch else ""))

    dims: Dict[str, Any] = {}
    batch_size = None
    if dynamic_batch:
        sizes = {t.shape[0] if t.dim() else None for t in leaves}
        if None in sizes or len(sizes) != 1:
            raise ExportError(f"dynamic batch needs every input tensor to have the same dim 0 size, got "
                              f"{[list(t.shape) for t in leaves]}; pass dynamic_batch=False (--static-batch)")
        (batch_size,) = sizes
        if batch_size < 2:
            raise ExportError("the example batch is 1, and torch.export specializes size 1 dims, so the batch "
                              "would be frozen; give an example batch of 2 or more, or pass dynamic_batch=False")
        if max_batch is not None and batch_size > max_batch:
            raise ExportError(f"the example batch {batch_size} is above max_batch {max_batch}")
        dims[BATCH_SYMBOL] = (torch.export.Dim(BATCH_SYMBOL, min=1, max=max_batch) if max_batch is not None
                              else torch.export.Dim(BATCH_SYMBOL, min=1))
    for axes in dynamic_dims.values():
        for sym in axes.values():
            dims.setdefault(sym, torch.export.Dim(sym, min=1))
    named_axes: Dict[int, Dict[int, str]] = {}  # id(tensor) -> axis -> symbol name

    def spec_of(pname):
        def one(t):
            if not isinstance(t, torch.Tensor):
                return None
            axes = {0: BATCH_SYMBOL} if dynamic_batch else {}
            if isinstance(bound.arguments.get(pname), torch.Tensor) and t is bound.arguments[pname]:
                axes.update({ax % t.dim(): sym for ax, sym in dynamic_dims.get(pname, {}).items()})
            named_axes[id(t)] = axes
            return {ax: dims[sym] for ax, sym in axes.items()} or None
        return one

    def shapes_tree(obj, one):
        """torch.export's dynamic_shapes for one argument: the argument's structure with a spec per
        tensor (a dataclass is given as the list of its fields, as torch.export expects)."""
        if isinstance(obj, torch.Tensor):
            return one(obj)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return [shapes_tree(getattr(obj, f.name), one) for f in dataclasses.fields(obj)]
        if isinstance(obj, dict):
            return {k: shapes_tree(v, one) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [shapes_tree(v, one) for v in obj] if isinstance(obj, list) else                 tuple(shapes_tree(v, one) for v in obj)
        return None

    dynamic_shapes = None
    if dims:
        dynamic_shapes = {p: shapes_tree(v, spec_of(p)) for p, v in bound.arguments.items()}

    _register_dataclasses(model, args, kwargs)
    torch._dynamo.reset()  # a Dynamo cache left by eager control flow can specialize a dim
    try:
        from ..ir.hf_linattn import lowered
        from ..ir.importer import _export_tracking_constants

        # small tensors built from constants keep their values (per image sizes, ir/importer.py);
        # Mamba 2 SSD and Gated DeltaNet layers lower to linear_attention (P60)
        with lowered(model):
            ep = _export_tracking_constants(model, args, dynamic_shapes, kwargs)
    except Exception as e:  # torch.export raises many exception types
        raise _explain_export_failure(e, name, model, args, dynamic_shapes) from None
    if decompose:
        ep = ep.run_decompositions()
    try:
        in_spec = pytree.from_torch_spec(ep.call_spec.in_spec)
        out_spec = pytree.from_torch_spec(ep.call_spec.out_spec)
        out_node = next(n for n in ep.graph.nodes if n.op == "output").args[0]
        out_node = out_node if isinstance(out_node, (list, tuple)) else [out_node]
        out_spec = pytree.none_leaves(out_spec, [i for i, o in enumerate(out_node) if o is None])
    except pytree.StructureError as e:
        raise ExportError(f"export refused: {e}") from None
    ordered = pytree.match(in_spec, _call_tree(args, kwargs), _is_tensor)  # torch's input order
    user = [s.arg.name for s in ep.graph_signature.input_specs if s.kind.name == "USER_INPUT"]
    if len(user) != len(ordered):
        raise ExportError(f"internal error: torch.export kept {len(user)} of {len(ordered)} input tensors")
    dim_names = {(ph, ax): sym for ph, t in zip(user, ordered) for ax, sym in named_axes.get(id(t), {}).items()}
    importer.TORCH_REGIONS.clear()
    try:
        prog = ir.import_exported(ep, name=name, dim_names=dim_names)
    except ir.IRError as e:
        branches = _control_flow_sites(ep)
        extra = "".join(f"\n  captured control flow: torch.{op} at {where}" for op, where in branches)
        raise ExportError(f"export refused: truestar cannot import {name}:\n{e}{extra}\n"
                          "No artifact was written.") from None

    # Verify the program as parsed back from its own bytes: what is checked is what ships.
    blob = serialize.dumps(prog)
    shipped = serialize.loads(blob).program
    phash = serialize.program_hash(shipped)

    batch_range = None
    if dynamic_batch:
        if BATCH_SYMBOL not in shipped.symbols:
            raise ExportError(f"export refused: the batch dim of {name} did not stay symbolic in the imported "
                              "program; pass dynamic_batch=False to export a fixed batch")
        lo, hi = shipped.symbols[BATCH_SYMBOL]
        batch_range = [lo, hi]
    for sym in {s for a in dynamic_dims.values() for s in a.values()}:
        if sym not in shipped.symbols:
            raise ExportError(f"export refused: dim {sym} of {name} did not stay symbolic (torch.export "
                              "specialized it); the model only works for one size there")

    flat = list(leaves)  # in call_spec order
    gen = torch.Generator().manual_seed(0)
    check_sets = [("example", flat)]
    if dynamic_batch:
        b2 = second_batch if second_batch is not None else _pick_second_batch(batch_size, *batch_range)
        if b2 is None or b2 == batch_size:
            raise ExportError(f"no second batch size differs from {batch_size} inside [{batch_range[0]}, "
                              f"{batch_range[1] if batch_range[1] is not None else 'inf'}]")
        if not (batch_range[0] <= b2 <= (batch_range[1] or b2)):
            raise ExportError(f"second_batch {b2} is outside the batch range {batch_range}")
        check_sets.append(("second_batch", list(_second_batch(flat, b2))))
    for sym in sorted({s for a in dynamic_dims.values() for s in a.values()}):
        lo, hi = shipped.symbols[sym]
        cur = next(t.shape[ax] for t in flat for ax, s in named_axes.get(id(t), {}).items() if s == sym)
        new = next((n for n in (cur + 3, cur - 1, cur + 1) if n >= max(lo, 1) and (hi is None or n <= hi)), None)
        if new is None:
            continue
        varied = [t if not any(s == sym for s in named_axes.get(id(t), {}).values()) else
                  _resize_axis(t, next(ax for ax, s in named_axes[id(t)].items() if s == sym), new, gen) for t in flat]
        check_sets.append((f"second_{sym}", varied))

    m64 = copy.deepcopy(model).double()
    checks = []
    recorder = _BranchRecorder()
    for label, inputs in check_sets:
        call_args, call_kwargs = pytree.unflatten(call_spec, inputs)
        x64 = _tree64((call_args, call_kwargs))
        # float32 casts inside the model stay float64, as in the reference interpreter (see
        # ir.reference.exact_float32_casts); the count is kept in the evidence
        casts = ir.reference.exact_float32_casts()
        with torch.no_grad(), casts:
            want = pytree.flatten(m64(*x64[0], **x64[1]), _is_tensor)[0]  # None entries carry no tensor
        problem = ""
        reference.COND_TRACE = recorder
        try:
            got = ir.run(shipped, *pytree.match(in_spec, x64, _is_tensor))
        except ir.IRError as e:
            got, problem = [], f"the program cannot run on these inputs: {e}"
        finally:
            reference.COND_TRACE = None
        if problem:
            max_abs = max_rel = float("inf")
        else:
            max_abs, max_rel, problem = _compare(got, want)
        passed = not problem and max_rel <= tolerance
        weak = _weakness(want, tolerance) if passed else ""
        checks.append({"name": label, "input_shapes": [list(t.shape) for t in inputs],
                       "input_dtypes": [str(t.dtype).replace("torch.", "") for t in inputs],
                       "inputs_sha256": inputs_sha256(inputs), "max_abs_error": max_abs,
                       "max_rel_error": max_rel, "passed": passed, "problem": problem, "weak": weak,
                       "float32_casts_kept_float64": casts.count})
    branches, branch_problems, branch_warnings = _verify_branches(shipped, prog, recorder, tolerance)
    importer.TORCH_REGIONS.clear()
    failed = [c for c in checks if not c["passed"]]
    if failed or branch_problems:
        lines = [f"export refused: the imported program of {name} disagrees with PyTorch (float64), so no "
                 "artifact was written."]
        for c in checks:
            detail = c["problem"] or f"max rel error {c['max_rel_error']:.3e}, bar {tolerance:.0e}"
            lines.append(f"  {c['name']} {c['input_shapes']}: {'PASS' if c['passed'] else 'FAIL'} ({detail})")
        lines += [f"  {p}" for p in branch_problems]
        raise VerificationError("\n".join(lines), checks)

    from .. import __version__
    warnings = sorted({f"{c['name']}: {c['weak']}" for c in checks if c["weak"]}) + branch_warnings
    evidence = {
        "warnings": warnings,
        "verified": True,
        "model": name,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "reference": "truestar.ir.reference float64 against the PyTorch module in float64",
        "tolerance": tolerance,
        "checks": checks,
        "branches": branches,
        "batch_symbol": BATCH_SYMBOL if dynamic_batch else None,
        "batch_range": batch_range,
        "dynamic_dims": {p: {str(ax): s for ax, s in a.items()} for p, a in dynamic_dims.items()},
        "io": {"inputs": in_spec, "outputs": out_spec, "arg_names": list(bound.arguments)[:len(args)]},
        "graph": "core_aten" if decompose else "export",
        "program_hash": phash,
        "torch_version": torch.__version__,
        "truestar_version": __version__,
        "python_version": platform.python_version(),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
    }
    artifact = serialize.dumps(shipped, {EVIDENCE_KEY: evidence})
    if serialize.loads(artifact, verify=False).program_hash != phash:
        raise ExportError("internal error: the artifact's program hash changed after verification")
    path = None
    if output:
        serialize.write_bytes(artifact, output)
        path = os.path.abspath(output)
        with open(path, "rb") as f:
            if f.read() != artifact:
                raise ExportError(f"internal error: {path} does not hold the bytes that were written")
    return ExportResult(path, shipped, phash, evidence, artifact)
