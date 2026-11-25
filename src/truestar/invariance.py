"""`truestar invariance`: does one row's result depend on what else is in the batch?

Method
------
For each of the first few rows of the example batch, on each backend, in the model's own dtype:

1. The row is run alone (batch 1). That is the reference: there is no float64 here, the question
   is whether the *same* backend gives the *same bits* for the same row.
2. The row is run again in contexts that must not change its result: the same call repeated, inside
   batches of 2, 7 and 64 filled with other random rows at the first, middle and last position,
   inside a batch padded with zero rows, and (for token models) with the sequence right padded.
   The row's slice of every output is compared bitwise and numerically, and argmax flips and greedy
   decoding token flips are counted.
3. Drilldown for a context that is not bitwise identical: every module call is replayed alone with
   the context's recorded inputs, except that the row's slice is replaced by the inputs the module
   saw when the row ran alone. A module whose row output still differs computes that row
   differently because of the rest of the batch. The first such module whose submodules are all
   invariant is named (its own code, or its own kernel, is the cause).
4. Inside that module, the ATen ops are traced in both runs and aligned by name, and each op whose
   row slice differs is re-run alone with the same substitution of the row's inputs. The first op
   whose row output still differs is named (for example aten.mm.default with a split K kernel).
5. Torch flag sets (deterministic algorithms, no reduced precision reductions, math attention, one
   CPU thread) are applied one at a time and the whole output check is rerun, so a flag is only
   reported as a fix when it was seen to fix it.
"""

from __future__ import annotations

import contextlib
import copy
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .backends import Backend, resolve_backends
from .capture import ROOT, Recorder, flatten_tensors, map_tensors
from .codes import Issue
from .compare import diff_tensors, threshold_for

BITWISE, CLOSE, DIFFERENT = "bitwise", "close", "different"
INVARIANT, VARIANT, NONDETERMINISTIC, DEPENDENT, ERROR = (
    "invariant", "batch_variant", "nondeterministic", "batch_dependent", "error")
_RANK = {BITWISE: 0, CLOSE: 1, DIFFERENT: 2}
DEFAULT_SIZES = (2, 7, 64)
PAD_SIZE = 8
MAX_DRILL_CONTEXTS = 4
MAX_MODULES_DRILLED = 6
MAX_OPS_PER_MODULE = 4


# ----------------------------------------------------------------------------- contexts


@dataclass(frozen=True)
class Context:
    name: str  # "batch7@3"
    kind: str  # "repeat", "batch", "pad" (zero rows), "seqpad" (sequence right padded)
    size: int = 1  # batch size the row runs in
    position: int = 0  # the row's index inside that batch
    seq_pad: int = 0  # extra sequence positions (seqpad only)

    def as_dict(self) -> Dict[str, Any]:
        return {"context": self.name, "kind": self.kind, "batch_size": self.size,
                "position": self.position, "seq_pad": self.seq_pad}


def default_contexts(sizes: Sequence[int] = DEFAULT_SIZES, seq_pad: int = 0) -> List[Context]:
    out = [Context("repeat", "repeat")]
    for n in sizes:
        if n < 2:
            continue
        for p in sorted({0, n // 2, n - 1}):
            out.append(Context(f"batch{n}@{p}", "batch", n, p))
    out.append(Context(f"pad{PAD_SIZE}@0", "pad", PAD_SIZE, 0))
    if seq_pad > 0:
        out.append(Context(f"seqpad+{seq_pad}", "seqpad", 1, 0, seq_pad))
    return out


# ----------------------------------------------------------------------------- tensors


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.is_floating_point() or a.is_complex():
        ints = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
        it = ints.get(a.element_size())
        if it is not None and not a.is_complex():
            return bool(torch.equal(a.contiguous().view(it), b.contiguous().view(it)))
    return bool(torch.equal(a, b))


def _regions(alone: Sequence[int], ctx: Sequence[int], c: Context) -> List[Tuple]:
    """Candidate indices of the row's part of a context tensor, from its shape against the alone
    shape.

    Batch contexts: a dim of size a alone and size * a in the context holds the row. For a = 1
    (the batch dim itself) that is [position]. For a > 1 the batch was folded into another dim
    (B * S rows of a matmul, B * H heads): batch major layouts hold the row at
    [position * a, (position + 1) * a), sequence major ones (batch_first=False attention) at
    position::size, so both are candidates. Sequence padding: a dim that grew holds the row at
    [0, a). Empty when the row cannot be located.
    """
    if len(alone) != len(ctx):
        return []
    choices: List[List[slice]] = []
    for a, b in zip(alone, ctx):
        if a == b:
            choices.append([slice(None)])
        elif c.kind in ("batch", "pad") and a > 0 and b == a * c.size:
            opts = [slice(c.position * a, (c.position + 1) * a)]
            if a > 1:
                opts.append(slice(c.position, None, c.size))
            choices.append(opts)
        elif c.kind == "seqpad" and b > a:
            choices.append([slice(0, a)])
        else:
            return []
    out: List[Tuple] = [()]
    for opts in choices:
        out = [x + (o,) for x in out for o in opts][:8]
    return out


def _distance(x: torch.Tensor, y: torch.Tensor) -> float:
    if _bits_equal(x, y):
        return 0.0
    if not (x.is_floating_point() and y.is_floating_point()):
        return float((x != y).sum()) + 1.0
    d = (x.double() - y.double()).nan_to_num(nan=1e30, posinf=1e30, neginf=-1e30)
    return float(d.abs().sum()) + 1e-300


def _region(alone: torch.Tensor, ctx: torch.Tensor, c: Context):
    """The candidate region of `ctx` closest to `alone` (None when none exists)."""
    cands = _regions(alone.shape, ctx.shape, c)
    if len(cands) <= 1:
        return cands[0] if cands else None
    return min(cands, key=lambda r: _distance(ctx[r], alone))


def _row_part(t: torch.Tensor, alone: torch.Tensor, c: Context) -> Optional[torch.Tensor]:
    r = _region(alone, t, c)
    return None if r is None else t[r]


def _leaves(obj: Any) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []

    def walk(o):
        if isinstance(o, torch.Tensor):
            out.append(o)
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)

    walk(obj)
    return out


def _substitute(ctx_obj: Any, alone_obj: Any, c: Context) -> Optional[Any]:
    """Context arguments with the row's part of every tensor replaced by the alone arguments.

    None when the two argument structures do not line up (the row cannot be isolated)."""
    cl, al = _leaves(ctx_obj), _leaves(alone_obj)
    if len(cl) != len(al):
        return None
    repl = []
    for ct, at in zip(cl, al):
        if ct.dtype != at.dtype:
            return None
        r = _region(at, ct, c)
        if r is None:
            return None
        t = ct.clone()
        t[r] = at
        repl.append(t)
    it = iter(repl)
    return map_tensors(ctx_obj, lambda _t: next(it))


def _to(obj: Any, device: str) -> Any:
    return map_tensors(obj, lambda t: t.detach().to(device))


def _clone(obj: Any) -> Any:
    return map_tensors(obj, torch.clone)


def _cpu_outputs(out: Any) -> List[torch.Tensor]:
    return [t.detach().to("cpu", copy=True) for t in flatten_tensors(out)]


@dataclass
class Compared:
    cls: str  # bitwise, close, different
    max_abs: Optional[float]
    rel_err: Optional[float]
    located: bool = True


def _tolerance(dtype: torch.dtype, backend: Backend) -> float:
    return threshold_for(dtype, backend.tf32, None)


def _compare(ctx_out: List[torch.Tensor], alone_out: List[torch.Tensor], c: Context,
             tol: float) -> Compared:
    if len(ctx_out) != len(alone_out):
        return Compared(DIFFERENT, None, None, located=False)
    worst = Compared(BITWISE, 0.0, 0.0)
    for t, a in zip(ctx_out, alone_out):
        part = t if c.kind == "repeat" else _row_part(t, a, c)
        if part is None:
            return Compared(DIFFERENT, None, None, located=False)
        if _bits_equal(part, a):
            continue
        d = diff_tensors(part, a)
        cls = DIFFERENT if (d.shape_mismatch or d.nonfinite_mismatch or d.rel_err > tol) else CLOSE
        if (_RANK[cls], d.rel_err) > (_RANK[worst.cls], worst.rel_err or 0.0):
            worst = Compared(cls, d.max_abs, d.rel_err)
    return worst


def _argmax_flips(ctx_out: List[torch.Tensor], alone_out: List[torch.Tensor],
                  c: Context) -> Tuple[int, int]:
    """(flips, decisions) of argmax over the last dim of the first floating output."""
    for t, a in zip(ctx_out, alone_out):
        if not a.is_floating_point() or a.ndim < 2 or a.shape[-1] < 2:
            continue
        part = t if c.kind == "repeat" else _row_part(t, a, c)
        if part is None or part.shape != a.shape:
            return 0, 0
        pa, aa = part.reshape(-1, a.shape[-1]).argmax(-1), a.reshape(-1, a.shape[-1]).argmax(-1)
        return int((pa != aa).sum()), int(aa.numel())
    return 0, 0


# ----------------------------------------------------------------------------- inputs


@dataclass
class Inputs:
    args: Tuple
    kwargs: Dict[str, Any]
    batch_dim: int
    batch: int  # example batch size
    lm: bool = False  # a single [B, S] integer tensor (token ids) feeds the model


def _batched(t: torch.Tensor, inp: Inputs) -> bool:
    return t.ndim > inp.batch_dim and t.shape[inp.batch_dim] == inp.batch


def _filler(t: torch.Tensor, n: int, bd: int, g: torch.Generator, zeros: bool) -> torch.Tensor:
    shape = list(t.shape)
    shape[bd] = n
    if zeros:
        return torch.zeros(shape, dtype=t.dtype)
    if t.is_floating_point():
        tf = t.double()
        std = float(tf.std()) if t.numel() > 1 else 1.0
        return (torch.randn(shape, generator=g, dtype=torch.float64) * (std or 1.0)
                + float(tf.mean())).to(t.dtype)
    if t.dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=g).bool()
    lo, hi = int(t.min()), int(t.max())
    return torch.randint(lo, hi + 1, shape, generator=g, dtype=torch.int64).to(t.dtype)


def row_inputs(inp: Inputs, row: int) -> Tuple[Tuple, Dict]:
    f = lambda t: t.narrow(inp.batch_dim, row, 1).clone() if _batched(t, inp) else t
    return map_tensors(inp.args, f), map_tensors(inp.kwargs, f)


def context_inputs(inp: Inputs, row: int, c: Context, seed: int) -> Tuple[Tuple, Dict]:
    g = torch.Generator().manual_seed(seed)
    bd = inp.batch_dim

    def f(t: torch.Tensor) -> torch.Tensor:
        if not _batched(t, inp):
            return t
        one = t.narrow(bd, row, 1)
        if c.kind == "repeat":
            return one.clone()
        if c.kind == "seqpad":
            if t.ndim < 2:
                return one.clone()
            shape = list(one.shape)
            shape[1] = c.seq_pad
            return torch.cat([one, torch.zeros(shape, dtype=t.dtype)], dim=1)
        out = _filler(t, c.size, bd, g, zeros=c.kind == "pad")
        out.narrow(bd, c.position, 1).copy_(one)
        return out

    return map_tensors(inp.args, f), map_tensors(inp.kwargs, f)


# ----------------------------------------------------------------------------- results


@dataclass
class RowResult:
    row: int
    context: Context
    cmp: Compared
    argmax_flips: int = 0
    argmax_total: int = 0
    tokens_alone: Optional[List[int]] = None
    tokens_context: Optional[List[int]] = None

    @property
    def token_flip_step(self) -> Optional[int]:
        if self.tokens_alone is None or self.tokens_context is None:
            return None
        for i, (a, b) in enumerate(zip(self.tokens_alone, self.tokens_context)):
            if a != b:
                return i
        return None

    def as_dict(self) -> Dict[str, Any]:
        d = self.context.as_dict()
        d.update({
            "row": self.row,
            "bitwise": self.cmp.cls == BITWISE,
            "class": self.cmp.cls,
            "located": self.cmp.located,
            "max_abs": _num(self.cmp.max_abs),
            "rel_err": _num(self.cmp.rel_err),
            "argmax_flips": self.argmax_flips,
            "argmax_total": self.argmax_total,
        })
        if self.tokens_alone is not None:
            d["tokens_alone"] = self.tokens_alone
            d["tokens_context"] = self.tokens_context
            d["token_flip_step"] = self.token_flip_step
        return d


@dataclass
class ModuleFinding:
    key: str
    name: str
    type_name: str
    context: str
    cls: str
    max_abs: Optional[float]
    own: bool  # every submodule is invariant: the cause is this module's own code or kernel
    source: Optional[Dict[str, Any]] = None
    rel_err: Optional[float] = None  # the row's relative change from the batch context
    # The module's own rounding error for this row: its output alone in the model dtype against
    # the same call in float64 on CPU. A batch change far above it is more than float reordering.
    noise_floor: Optional[float] = None

    @property
    def score(self) -> float:
        """Batch change relative to the module's noise floor (the forward check's ratio)."""
        if self.rel_err is None:
            return float("inf")  # outputs could not be lined up: as bad as it gets
        floor = self.noise_floor
        return self.rel_err / max(floor or 1.2e-7, 1e-12)

    def as_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "name": self.name, "type": self.type_name,
                "context": self.context, "class": self.cls, "max_abs": _num(self.max_abs),
                "own": self.own, "source": self.source, "rel_err": _num(self.rel_err),
                "noise_floor": _num(self.noise_floor),
                "noise_ratio": _num(None if self.rel_err is None else self.score)}


@dataclass
class OpFinding:
    op: str
    module: str  # module key
    context: str
    index: int  # 1 based among the module's compared ops
    total: int
    cls: str
    max_abs: Optional[float]
    shapes_alone: List[List[int]]
    shapes_context: List[List[int]]
    matched: bool = True  # False: this op ran only in the batch context (a different kernel)

    def as_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "module": self.module, "context": self.context,
                "index": self.index, "total": self.total, "class": self.cls,
                "max_abs": _num(self.max_abs), "input_shapes_alone": self.shapes_alone,
                "input_shapes_context": self.shapes_context, "matched": self.matched}


@dataclass
class FlagTrial:
    name: str
    settings: List[str]
    fixed: bool
    remaining: int  # (row, context) pairs still not bitwise identical
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"flags": self.name, "settings": self.settings, "fixed": self.fixed,
                "remaining": self.remaining, "error": self.error or None}


@dataclass
class BackendInvariance:
    backend: Backend
    status: str = INVARIANT
    rows: List[RowResult] = field(default_factory=list)
    modules: List[ModuleFinding] = field(default_factory=list)
    ops: List[OpFinding] = field(default_factory=list)
    flag_trials: List[FlagTrial] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    error: str = ""

    def add(self, code: str, msg: str) -> None:
        self.issues.append(Issue(code, msg, self.backend.name))

    @property
    def codes(self) -> List[str]:
        out: List[str] = []
        for i in self.issues:
            if i.code not in out:
                out.append(i.code)
        return out

    @property
    def failing(self) -> List[RowResult]:
        return [r for r in self.rows if r.cmp.cls != BITWISE]

    @property
    def first_module(self) -> Optional[ModuleFinding]:
        # The own cause whose batch change stands furthest above its own noise floor: a float
        # level difference in an early Linear must not outrank a real dependence later on.
        own = [m for m in self.modules if m.own]
        return max(own, key=lambda m: (_RANK.get(m.cls, 0), m.score)) if own else None

    @property
    def first_op(self) -> Optional[OpFinding]:
        fm = self.first_module
        for o in self.ops:
            if fm is not None and o.module == fm.key:
                return o
        return self.ops[0] if self.ops else None

    @property
    def token_flips(self) -> int:
        return sum(1 for r in self.rows if r.token_flip_step is not None)

    @property
    def argmax_flips(self) -> int:
        return sum(r.argmax_flips for r in self.rows)

    def non_invariant_ops(self) -> List[Dict[str, Any]]:
        """Distinct (module, op) pairs named, with the contexts each was seen in."""
        seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for o in self.ops:
            k = (o.module, o.op)
            if k not in seen:
                seen[k] = {"module": o.module, "op": o.op, "class": o.cls, "contexts": [],
                           "max_abs": _num(o.max_abs), "matched": o.matched}
            if o.context not in seen[k]["contexts"]:
                seen[k]["contexts"].append(o.context)
        return list(seen.values())


@dataclass
class InvarianceReport:
    model_name: str
    dtype: torch.dtype
    contexts: List[Context]
    rows_checked: List[int]
    decode_steps: int
    results: List[BackendInvariance] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(r.status == ERROR for r in self.results):
            return ERROR
        return INVARIANT if all(r.status == INVARIANT for r in self.results) else "not_invariant"

    @property
    def codes(self) -> List[str]:
        out: List[str] = []
        for r in self.results:
            for c in r.codes:
                if c not in out:
                    out.append(c)
        return out

    def to_text(self) -> str:
        lines = [f"truestar invariance: {self.model_name} ({str(self.dtype).replace('torch.', '')}),"
                 f" rows {self.rows_checked}, {len(self.contexts)} contexts: "
                 + ", ".join(c.name for c in self.contexts)]
        for n in self.notes:
            lines.append(f"  note: {n}")
        for r in self.results:
            b = r.backend
            lines.append(f"\n[{b.name}] {b.description}: {r.status.upper()}")
            if r.error:
                lines.append(f"  error: {r.error}")
                continue
            ok = sum(1 for x in r.rows if x.cmp.cls == BITWISE)
            lines.append(f"  {ok}/{len(r.rows)} (row, context) pairs bitwise identical to the row alone")
            for x in r.failing[:12]:
                extra = ""
                if x.argmax_flips:
                    extra += f", argmax flips {x.argmax_flips}/{x.argmax_total}"
                if x.token_flip_step is not None:
                    extra += f", greedy tokens differ from step {x.token_flip_step}"
                ma = "n/a" if x.cmp.max_abs is None else f"{x.cmp.max_abs:.2e}"
                lines.append(f"    row {x.row} {x.context.name:<12} {x.cmp.cls:<9} max abs {ma}{extra}")
            if len(r.failing) > 12:
                lines.append(f"    ... {len(r.failing) - 12} more")
            fm = r.first_module
            if fm is not None:
                lines.append(f"  first batch dependent module: {fm.key} ({fm.type_name}) in {fm.context}")
            for o in r.non_invariant_ops():
                tag = "" if o["matched"] else " (runs only in the batch context)"
                lines.append(f"  op: {o['op']} in {o['module']}{tag}, contexts {', '.join(o['contexts'])}")
            for t in r.flag_trials:
                verdict = "FIXES it" if t.fixed else f"does not fix it ({t.remaining} pairs still differ)"
                if t.error:
                    verdict = f"failed: {t.error}"
                lines.append(f"  flags {t.name}: {verdict}")
            for n in r.notes:
                lines.append(f"  note: {n}")
            for i in r.issues:
                lines.append(f"  [{i.code}] {i.message}")
        for name, why in self.skipped:
            lines.append(f"\n[{name}] skipped: {why}")
        lines.append(f"\nstatus: {self.status}")
        return "\n".join(lines)


def _num(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if x == x and abs(x) != float("inf") else None


# ----------------------------------------------------------------------------- flags


def _cpu_flag_sets() -> Dict[str, List[Tuple[str, Callable, Callable]]]:
    def det():
        prev = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        return lambda: torch.use_deterministic_algorithms(prev)

    def one_thread():
        prev = torch.get_num_threads()
        torch.set_num_threads(1)
        return lambda: torch.set_num_threads(prev)

    def mkldnn_off():
        prev = torch.backends.mkldnn.enabled
        torch.backends.mkldnn.enabled = False
        return lambda: setattr(torch.backends.mkldnn, "enabled", prev)

    return {
        "deterministic": [("torch.use_deterministic_algorithms(True)", det)],
        "one_thread": [("torch.set_num_threads(1)", one_thread)],
        "mkldnn_off": [("torch.backends.mkldnn.enabled = False", mkldnn_off)],
    }


def _cuda_flag_sets() -> Dict[str, List[Tuple[str, Callable]]]:
    def setattr_flag(obj, attr, value):
        def apply():
            prev = getattr(obj, attr)
            setattr(obj, attr, value)
            return lambda: setattr(obj, attr, prev)
        return apply

    def det():
        prev_env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        prev = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)

        def undo():
            torch.use_deterministic_algorithms(prev)
            if prev_env is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev_env
        return undo

    m = torch.backends.cuda.matmul
    reduced = [("torch.backends.cuda.matmul.allow_tf32 = False", setattr_flag(m, "allow_tf32", False)),
               ("torch.backends.cudnn.allow_tf32 = False",
                setattr_flag(torch.backends.cudnn, "allow_tf32", False))]
    for attr in ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"):
        if hasattr(m, attr):
            reduced.append((f"torch.backends.cuda.matmul.{attr} = False", setattr_flag(m, attr, False)))

    def sdp(name):
        en = getattr(torch.backends.cuda, f"enable_{name}_sdp", None)
        is_on = getattr(torch.backends.cuda, f"{name}_sdp_enabled", None)
        if en is None or is_on is None:
            return None

        def apply():
            prev = is_on()
            en(False)
            return lambda: en(prev)
        return (f"torch.backends.cuda.enable_{name}_sdp(False)", apply)

    math_sdp = [x for x in (sdp("flash"), sdp("mem_efficient"), sdp("cudnn")) if x is not None]
    return {
        "deterministic": [("torch.use_deterministic_algorithms(True) with "
                           "CUBLAS_WORKSPACE_CONFIG=:4096:8", det),
                          ("torch.backends.cudnn.deterministic = True",
                           setattr_flag(torch.backends.cudnn, "deterministic", True)),
                          ("torch.backends.cudnn.benchmark = False",
                           setattr_flag(torch.backends.cudnn, "benchmark", False))],
        "no_reduced_precision": reduced,
        "math_sdpa": math_sdp,
    }


def flag_sets(backend: Backend) -> Dict[str, List[Tuple[str, Callable]]]:
    sets = _cuda_flag_sets() if backend.device.startswith("cuda") else _cpu_flag_sets()
    sets = {k: v for k, v in sets.items() if v}
    sets["all"] = [x for v in list(sets.values()) for x in v]
    return sets


@contextlib.contextmanager
def _applied(flags: List[Tuple[str, Callable]]) -> Iterator[None]:
    undo = []
    try:
        for _, apply in flags:
            undo.append(apply())
        yield
    finally:
        for u in reversed(undo):
            u()


# ----------------------------------------------------------------------------- core


def _model_dtype(model: nn.Module) -> torch.dtype:
    for p in list(model.parameters()) + list(model.buffers()):
        if p.is_floating_point():
            return p.dtype
    return torch.float32


def _cast(obj: Any, device: str, dtype: torch.dtype) -> Any:
    return map_tensors(obj, lambda t: t.detach().to(device=device, dtype=dtype)
                       if t.is_floating_point() else t.detach().to(device))


def _forward(model: nn.Module, args, kwargs) -> Any:
    with torch.no_grad():
        return model(*args, **kwargs)


def _greedy(model: nn.Module, tokens: torch.Tensor, steps: int, position: int) -> List[int]:
    """Greedy decoding of every row in `tokens`; returns the tokens generated for `position`."""
    out = []
    with torch.no_grad():
        for _ in range(steps):
            logits = flatten_tensors(model(tokens))[0]
            nxt = logits[:, -1].argmax(-1, keepdim=True).to(tokens.dtype)
            out.append(int(nxt[position, 0]))
            tokens = torch.cat([tokens, nxt], dim=1)
    return out


def _run_contexts(model: nn.Module, inp: Inputs, rows: List[int], contexts: List[Context],
                  device: str, dtype: torch.dtype, tol: float, seed: int,
                  decode_steps: int) -> List[RowResult]:
    results: List[RowResult] = []
    for r in rows:
        a_args, a_kw = row_inputs(inp, r)
        a_args, a_kw = _cast(a_args, device, dtype), _cast(a_kw, device, dtype)
        alone = _cpu_outputs(_forward(model, a_args, a_kw))
        tokens_alone = None
        if inp.lm and decode_steps > 0:
            tokens_alone = _greedy(model, _leaves(a_args)[0], decode_steps, 0)
        for i, c in enumerate(contexts):
            c_args, c_kw = context_inputs(inp, r, c, seed + 1000 * r + i)
            c_args, c_kw = _cast(c_args, device, dtype), _cast(c_kw, device, dtype)
            out = _cpu_outputs(_forward(model, c_args, c_kw))
            cmp = _compare(out, alone, c, tol)
            flips, total = _argmax_flips(out, alone, c)
            rr = RowResult(r, c, cmp, flips, total)
            if tokens_alone is not None and c.kind != "seqpad":
                rr.tokens_alone = tokens_alone
                rr.tokens_context = _greedy(model, _leaves(c_args)[0], decode_steps, c.position)
            results.append(rr)
    return results


def _replaying(name: str):
    """Training mode: an isolated replay of module ``name`` keys its dropout masks by that name."""
    from .trainmode import active

    tm = active()
    return tm.replay(f"{name}#replay", name) if tm is not None else contextlib.nullcontext()


def _record(model: nn.Module, args, kwargs):
    with torch.no_grad(), Recorder(model, keep_inputs=True) as rec:
        model(*args, **kwargs)
    return {c.key: c for c in rec.calls}, rec.calls


def _module_by_name(model: nn.Module, name: str) -> nn.Module:
    return model if name == ROOT else model.get_submodule(name)


def _descendants(calls) -> Dict[str, List[str]]:
    kids = {c.key: list(c.children) for c in calls}
    out: Dict[str, List[str]] = {}

    def walk(k):
        if k in out:
            return out[k]
        acc: List[str] = []
        for ch in kids.get(k, []):
            acc.append(ch)
            acc.extend(walk(ch))
        out[k] = acc
        return acc

    for k in kids:
        walk(k)
    return out


def drill_modules(model: nn.Module, a_args, a_kw, c_args, c_kw, c: Context, device: str,
                  tol: float, notes: List[str]) -> List[Tuple[ModuleFinding, Any, Any, Any, Any]]:
    """Module calls whose output for the row depends on the batch, replayed alone.

    Returns (finding, alone args, alone kwargs, substituted context args, kwargs) per module that
    is batch dependent, in execution order. Arguments are on `device`.
    """
    alone_calls, alone_list = _record(model, a_args, a_kw)
    ctx_calls, ctx_list = _record(model, c_args, c_kw)
    if set(alone_calls) != set(ctx_calls):
        only = sorted(set(ctx_calls) ^ set(alone_calls))[:5]
        notes.append(f"{c.name}: different modules ran in the batch context ({', '.join(only)}); "
                     "only modules that ran in both are replayed.")
    varying: Dict[str, Tuple] = {}
    skipped = 0
    for cc in sorted(ctx_list, key=lambda x: x.start):
        ca = alone_calls.get(cc.key)
        if ca is None:
            continue
        mod = _module_by_name(model, cc.name)
        sub = _substitute((cc.args, cc.kwargs), (ca.args, ca.kwargs), c)
        if sub is None or not _leaves((ca.args, ca.kwargs)):
            skipped += 1
            continue
        s_args, s_kw = _to(sub, device)
        r_args, r_kw = _to((ca.args, ca.kwargs), device)
        try:
            with _replaying(cc.name):
                ref = _cpu_outputs(_forward(mod, _clone(r_args), _clone(r_kw)))
            with _replaying(cc.name):
                got = _cpu_outputs(_forward(mod, _clone(s_args), _clone(s_kw)))
        except Exception as exc:  # a module that cannot be called on its own
            notes.append(f"{c.name}: replaying {cc.key} alone failed: {type(exc).__name__}: {exc}")
            continue
        cmp = _compare(got, ref, c, tol)
        if cmp.cls != BITWISE:
            varying[cc.key] = (cc, cmp, r_args, r_kw, s_args, s_kw)
    if skipped:
        notes.append(f"{c.name}: {skipped} module call(s) could not be replayed with the row "
                     "isolated (their inputs have no batch dimension that lines up).")
    desc = _descendants(ctx_list)
    out = []
    for key, (cc, cmp, r_args, r_kw, s_args, s_kw) in varying.items():
        own = not any(d in varying for d in desc.get(key, []))
        f = ModuleFinding(key, cc.name, cc.type_name, c.name, cmp.cls, cmp.max_abs, own,
                          rel_err=cmp.rel_err)
        if own:
            with _replaying(cc.name):
                f.noise_floor = _noise_floor(_module_by_name(model, cc.name), r_args, r_kw)
        out.append((f, r_args, r_kw, s_args, s_kw))
    return out


def _noise_floor(mod: nn.Module, r_args, r_kw) -> Optional[float]:
    """The module's own rounding error on the row alone: model dtype output against a float64 CPU
    replay of the same call. None when float64 cannot run it; then the dtype epsilon is used."""
    try:
        got = _cpu_outputs(_forward(mod, _clone(r_args), _clone(r_kw)))
        dt = next((t.dtype for t in got if t.is_floating_point()), None)
        if dt is None:
            return None
        m64 = copy.deepcopy(mod).to("cpu").to(torch.float64)
        up = lambda t: t.detach().cpu().to(torch.float64) if t.is_floating_point() else t.cpu()
        ref = _cpu_outputs(_forward(m64, map_tensors(r_args, up), map_tensors(r_kw, up)))
        errs = [diff_tensors(g, r).rel_err for g, r in zip(got, ref) if g.is_floating_point()]
        errs = [e for e in errs if e == e]
        return max(errs + [float(torch.finfo(dt).eps)])
    except Exception:
        return None


# Ops that only move or copy bits. They cannot change rounding, so they are never blamed for a
# float level difference (they can still be blamed for a real dependence, like x[0] for every row).
_MOVEMENT = {
    "aten.cat", "aten.stack", "aten.slice", "aten.select", "aten.view", "aten._unsafe_view",
    "aten.reshape", "aten.permute", "aten.transpose", "aten.t", "aten.expand", "aten.clone",
    "aten.copy", "aten.copy_", "aten.contiguous", "aten.unsqueeze", "aten.squeeze", "aten.split",
    "aten.split_with_sizes", "aten.unbind", "aten.chunk", "aten.narrow", "aten.new_zeros",
    "aten.zeros_like", "aten.new_full", "aten.full", "aten.constant_pad_nd", "aten.alias",
    "aten.detach", "aten.lift_fresh", "aten._to_copy", "aten.index_select", "aten.embedding",
    "aten.masked_fill", "aten.where", "aten.flatten", "aten.unflatten",
}
_RANDOM = ("aten.rand", "aten.bernoulli", "aten.normal", "aten.uniform", "aten.native_dropout",
           "aten.dropout", "aten.multinomial", "aten.exponential", "aten.poisson", "aten.randperm")


def _comparable(rec) -> bool:
    """Ops whose outputs can be compared. Unlike opdrill, ops merely tagged as seeded (attention
    with an optional dropout) are kept: only random number generators are skipped."""
    from .opdrill import _UNINITIALIZED

    if not rec.outputs:
        return False
    packet = str(rec.func.overloadpacket)
    return packet not in _UNINITIALIZED and not packet.startswith(_RANDOM)


def _scalars(obj: Any) -> List[Any]:
    """Non tensor arguments of an op call, in order."""
    out: List[Any] = []

    def walk(o):
        if isinstance(o, torch.Tensor):
            return
        if isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for k in sorted(o, key=str):
                out.append(k)
                walk(o[k])
        else:
            out.append(o if isinstance(o, (int, float, bool, str, type(None))) else str(o))

    walk(obj)
    return out


def _outs(func, outs: List[torch.Tensor]) -> List[torch.Tensor]:
    """Outputs to compare. Seeded ops (attention kernels with optional dropout) also return the
    RNG seed and offset they drew, which differ on every call: only their floating outputs count."""
    if torch.Tag.nondeterministic_seeded in getattr(func, "tags", ()):
        return [t for t in outs if t.is_floating_point()]
    return outs


def drill_ops(module: nn.Module, module_key: str, r_args, r_kw, s_args, s_kw, c: Context,
              device: str, tol: float, limit: int = MAX_OPS_PER_MODULE) -> List[OpFinding]:
    """ATen ops inside `module` whose own output for the row depends on the batch."""
    from . import opdrill

    name = module_key.rsplit("#", 1)[0]
    with _replaying(name):
        alone_ops, t1 = opdrill.trace_ops(module, _clone(r_args), _clone(r_kw))
    with _replaying(name):
        ctx_ops, t2 = opdrill.trace_ops(module, _clone(s_args), _clone(s_kw))
    if t1 or t2:
        return []
    alone_ops = [o for o in alone_ops if _comparable(o)]
    ctx_ops = [o for o in ctx_ops if _comparable(o)]
    pairs = opdrill._align([o.name for o in alone_ops], [o.name for o in ctx_ops])
    found: List[OpFinding] = []
    unmatched: List[OpFinding] = []
    shapes = lambda rec: [list(t.shape) for t in _leaves((rec.args, rec.kwargs))]
    for i, d in enumerate(ctx_ops):
        if i not in pairs:
            unmatched.append(OpFinding(d.name, module_key, c.name, i + 1, len(ctx_ops), CLOSE,
                                       None, [], shapes(d), matched=False))
            continue
        a = alone_ops[pairs[i]]
        if _scalars((d.args, d.kwargs)) != _scalars((a.args, a.kwargs)):
            continue  # not the same computation (a loop index, or sizes that encode the batch)
        if _compare(_outs(d.func, d.outputs), _outs(a.func, a.outputs), c, tol).cls == BITWISE:
            continue
        sub = _substitute((d.args, d.kwargs), (a.args, a.kwargs), c)
        if sub is None:
            continue
        try:
            got = _outs(d.func, opdrill._call(d.func, sub[0], sub[1], device))
            ref = _outs(a.func, opdrill._call(a.func, a.args, a.kwargs, device))
        except Exception:
            continue
        cmp = _compare(got, ref, c, tol)
        if cmp.cls == BITWISE:
            continue  # inherited from an earlier op
        if cmp.cls == CLOSE and str(d.func.overloadpacket) in _MOVEMENT:
            continue  # copies bits, cannot round: the row was located ambiguously
        found.append(OpFinding(d.name, module_key, c.name, i + 1, len(ctx_ops), cmp.cls,
                               cmp.max_abs, shapes(a), shapes(d)))
        if len(found) >= limit:
            break
    if not found and unmatched:
        # No matched op differs on its own, but the batch context ran ops the row alone did not:
        # a kernel chosen by shape (for example a different attention or GEMM path).
        return unmatched[:limit]
    return found


def _pick_drill_contexts(rows: List[RowResult], limit: int) -> List[RowResult]:
    """Failing (row, context) pairs to drill: one per context kind and size, worst first."""
    picked: List[RowResult] = []
    seen = set()
    for r in sorted((x for x in rows if x.cmp.cls != BITWISE),
                    key=lambda x: (-_RANK[x.cmp.cls], x.context.kind != "repeat")):
        k = (r.context.kind, r.context.size)
        if k in seen:
            continue
        seen.add(k)
        picked.append(r)
        if len(picked) >= limit:
            break
    return picked


def _parse_inputs(inputs, kwargs, batch_dim: int) -> Inputs:
    args = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    kwargs = dict(kwargs or {})
    leaves = [t for t in _leaves((args, kwargs)) if t.ndim > batch_dim]
    if not leaves:
        raise ValueError("invariance needs at least one tensor input with a batch dimension "
                         f"(dim {batch_dim})")
    batch = leaves[0].shape[batch_dim]
    inp = Inputs(args, kwargs, batch_dim, batch)
    batched = [t for t in _leaves((args, kwargs)) if _batched(t, inp)]
    inp.lm = (len(batched) == 1 and batch_dim == 0 and batched[0].ndim == 2
              and not batched[0].is_floating_point() and batched[0].dtype != torch.bool)
    return inp


def invariance(
    model: nn.Module,
    inputs,
    kwargs: Optional[Dict[str, Any]] = None,
    backends: Optional[List[str]] = None,
    sizes: Sequence[int] = DEFAULT_SIZES,
    rows: int = 2,
    batch_dim: int = 0,
    decode_steps: Optional[int] = None,
    seq_pad: Optional[int] = None,
    try_flags: bool = True,
    drill: bool = True,
    seed: int = 0,
    model_name: Optional[str] = None,
    train: bool = False,
) -> InvarianceReport:
    """Check that each row's result does not depend on the rest of the batch, on every backend.

    Args:
        model: any nn.Module; copied, never modified, run in eval mode, or in training mode with
            ``train=True`` (truestar.trainmode): dropout masks are keyed and drawn once per row
            shape, so a row's mask does not depend on its batch position; training batch norm
            normalizes with batch statistics, so it is named as batch dependent, which it is.
        inputs: a tensor or a sequence of positional arguments; tensors whose size at `batch_dim`
            equals the example batch size are treated as batched, others are passed unchanged.
        sizes: batch sizes to place the row in.
        rows: how many example rows to check.
        decode_steps: greedy tokens to decode for token models (default 4, 0 disables).
        seq_pad: right padding of the token sequence for token models (default 5, 0 disables).
        try_flags: rerun with torch flag sets to see which, if any, fix the non invariance.
    """
    inp = _parse_inputs(inputs, kwargs, batch_dim)
    if decode_steps is None:
        decode_steps = 4 if inp.lm else 0
    if seq_pad is None:
        seq_pad = 5 if inp.lm else 0
    if not inp.lm:
        decode_steps, seq_pad = 0, 0
    contexts = default_contexts(sizes, seq_pad)
    dtype = _model_dtype(model)
    chosen, skipped = resolve_backends(backends)
    row_ids = list(range(min(rows, inp.batch)))
    rep = InvarianceReport(model_name or type(model).__name__, dtype, contexts, row_ids,
                           decode_steps, skipped=skipped)
    from .trainmode import TrainMode, prepare

    base = prepare(model, train)
    tm = TrainMode(seed, per_row=True) if train else None
    if tm is not None:
        rep.notes.append("training mode: keyed dropout masks drawn once per row shape (a row's mask does not "
                         "depend on its batch position); training batch norm uses batch statistics, so a row "
                         "depends on its batch through it by definition.")
    with tm if tm is not None else contextlib.nullcontext():
        _run_backends(rep, base, inp, row_ids, contexts, dtype, seed, decode_steps, try_flags, drill, chosen)
    return rep


def _run_backends(rep, base, inp, row_ids, contexts, dtype, seed, decode_steps, try_flags, drill, chosen):
    for b in chosen:
        res = BackendInvariance(b)
        rep.results.append(res)
        try:
            _check_backend(res, base, inp, row_ids, contexts, dtype, seed, decode_steps,
                           try_flags, drill)
        except Exception as exc:
            bns = [n or "<model>" for n, m in base.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
            if base.training and bns and "value per channel when training" in str(exc):
                # batch norm cannot normalize one row by the statistics of that row alone: in training
                # mode the row's result is defined by the rest of its batch, which is the finding
                res.status = DEPENDENT
                res.add("TS052", f"training mode batch norm ({', '.join(bns[:5])}) normalizes each row with the "
                                 "statistics of its whole batch, so a row's result depends on the batch by "
                                 f"definition; a row alone cannot even run ({exc}). Check with train=False for "
                                 "the kernels, or use a per row norm (LayerNorm, GroupNorm) if rows must be "
                                 "independent")
                continue
            res.status = ERROR
            res.error = f"{type(exc).__name__}: {exc}"
            res.add("TS002", f"the model raised on {b.name}: {res.error}")


def _check_backend(res: BackendInvariance, base: nn.Module, inp: Inputs, row_ids, contexts,
                   dtype, seed, decode_steps, try_flags, drill) -> None:
    from .trainmode import active

    b = res.backend
    dev = copy.deepcopy(base).to(b.device)
    if active() is not None:
        active().attach(dev)
    tol = _tolerance(dtype, b)
    res.rows = _run_contexts(dev, inp, row_ids, contexts, b.device, dtype, tol, seed, decode_steps)
    failing = res.failing
    if not failing:
        res.status = INVARIANT
        return
    worst = max(failing, key=lambda x: _RANK[x.cmp.cls]).cmp.cls
    repeat_bad = any(r.context.kind == "repeat" for r in failing)
    batch_bad = [r for r in failing if r.context.kind != "repeat"]
    if worst == DIFFERENT:
        res.status = DEPENDENT
        x = next(r for r in failing if r.cmp.cls == DIFFERENT)
        res.add("TS052", f"row {x.row} in {x.context.name} differs beyond float tolerance "
                         f"(rel err {_fmt(x.cmp.rel_err)}): the result depends on other rows, "
                         "not only on summation order")
    elif repeat_bad and not batch_bad:
        res.status = NONDETERMINISTIC
    else:
        res.status = VARIANT
    if repeat_bad:
        res.add("TS051", "the same row run twice alone gives different bits (run to run "
                         "nondeterminism)")
    if batch_bad and worst != DIFFERENT:
        names = sorted({r.context.name for r in batch_bad}, key=lambda n: [c.name for c in contexts].index(n))
        res.add("TS050", f"{len(batch_bad)} of {len(res.rows)} (row, context) pairs are not "
                         f"bitwise identical to the row alone: {', '.join(names[:6])}")
    flips = res.argmax_flips
    if flips or res.token_flips:
        res.add("TS053", f"argmax flips: {flips} positions; greedy decoding differs in "
                         f"{res.token_flips} contexts")
    if drill:
        _drill(res, dev, inp, contexts, dtype, tol, seed)
        if not res.ops:
            res.add("TS054", "no single module or ATen op was named for the non invariance"
                    + ("" if res.modules else " (no module replayed alone differs)"))
    if try_flags:
        for name, flags in flag_sets(b).items():
            try:
                with _applied(flags):
                    again = _run_contexts(dev, inp, row_ids, contexts, b.device, dtype, tol,
                                          seed, decode_steps)
                remaining = sum(1 for r in again if r.cmp.cls != BITWISE)
                res.flag_trials.append(FlagTrial(name, [s for s, _ in flags], remaining == 0,
                                                 remaining))
            except Exception as exc:
                res.flag_trials.append(FlagTrial(name, [s for s, _ in flags], False, -1,
                                                 f"{type(exc).__name__}: {str(exc)[:200]}"))
        if b.device.startswith("cuda"):
            torch.cuda.synchronize()


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2e}"


def _drill(res: BackendInvariance, dev: nn.Module, inp: Inputs, contexts, dtype, tol,
           seed) -> None:
    from .agent import module_source

    b = res.backend
    seen_modules = set()
    for rr in _pick_drill_contexts(res.rows, MAX_DRILL_CONTEXTS):
        c = rr.context
        i = contexts.index(c)
        a_args, a_kw = _cast(row_inputs(inp, rr.row), b.device, dtype)
        c_args, c_kw = _cast(context_inputs(inp, rr.row, c, seed + 1000 * rr.row + i), b.device,
                             dtype)
        found = drill_modules(dev, a_args, a_kw, c_args, c_kw, c, b.device, tol, res.notes)
        own = [f for f in found if f[0].own]
        for f, *_ in found:
            if f.own:
                try:
                    f.source = module_source(dev, f.name)
                except Exception:
                    f.source = None
            res.modules.append(f)
        for f, r_args, r_kw, s_args, s_kw in own[:MAX_MODULES_DRILLED]:
            if (f.key, c.name) in seen_modules:
                continue
            seen_modules.add((f.key, c.name))
            mod = _module_by_name(dev, f.name)
            res.ops.extend(drill_ops(mod, f.key, r_args, r_kw, s_args, s_kw, c, b.device, tol))
    # keep modules in a stable order: own causes first, then by context order
    order = {c.name: i for i, c in enumerate(contexts)}
    res.modules.sort(key=lambda m: (not m.own, -_RANK.get(m.cls, 0), -m.score,
                                    order.get(m.context, 0)))


# ----------------------------------------------------------------------------- agent JSON


def _replacement(op: str) -> Optional[str]:
    table = {
        "aten.mm": "truestar.invariant_ops.matmul", "aten.addmm": "truestar.invariant_ops.linear",
        "aten.linear": "truestar.invariant_ops.linear", "aten.matmul": "truestar.invariant_ops.matmul",
        "aten.bmm": "truestar.invariant_ops.bmm", "aten.baddbmm": "truestar.invariant_ops.bmm",
        "aten._softmax": "truestar.invariant_ops.softmax",
        "aten.native_layer_norm": "truestar.invariant_ops.layer_norm",
        "aten.sum": "truestar.invariant_ops.tree_sum", "aten.mean": "truestar.invariant_ops.mean",
        "aten.gelu": "truestar.invariant_ops.gelu", "aten.silu": "truestar.invariant_ops.silu",
        "aten.sigmoid": "truestar.invariant_ops.sigmoid",
        "aten.exp": "truestar.invariant_ops.elementwise",
        "aten.tanh": "truestar.invariant_ops.elementwise",
    }
    for k, v in table.items():
        if op == k or op.startswith(k + "."):
            return v
    if "attention" in op or "sdpa" in op or "_transformer_encoder_layer_fwd" in op:
        return "truestar.invariant_ops.scaled_dot_product_attention"
    return None


def next_actions(rep: InvarianceReport, model_file: Optional[str] = None) -> List[Dict[str, Any]]:
    acts: List[Dict[str, Any]] = []
    for r in rep.results:
        b = r.backend.name
        if r.status in (INVARIANT, ERROR):
            if r.status == ERROR:
                acts.append({"kind": "inspect_failure", "backend": b, "command": None,
                             "description": f"The model raised on {b}: {r.error}"})
            continue
        if r.status == DEPENDENT:
            fm = r.first_module
            src = (fm.source or {}).get("user") if fm else None
            acts.append({
                "kind": "fix_batch_dependence", "backend": b, "command": None,
                "file": src.get("file") if src else None, "line": src.get("line") if src else None,
                "description": "A row's result depends on the other rows beyond float noise (for "
                               "example batch statistics, a mean over the batch, or cross row "
                               "attention). " + (f"Start at {fm.key} ({fm.type_name})." if fm else
                                                 "No module was named."),
            })
        for o in r.non_invariant_ops():
            if r.status == DEPENDENT and o["class"] == "different":
                continue
            rep_op = _replacement(o["op"])
            acts.append({
                "kind": "use_batch_invariant_op", "backend": b, "op": o["op"], "module": o["module"],
                "replacement": rep_op, "command": None,
                "description": (f"{o['op']} in {o['module']} gives a row different bits depending "
                                f"on the batch ({', '.join(o['contexts'][:3])}). Use a batch "
                                f"invariant kernel for it" + (f", for example {rep_op}" if rep_op else "")
                                + ", or keep the batch shape and the row's position fixed."),
            })
        fixes = [t for t in r.flag_trials if t.fixed and t.name != "all"]
        if fixes:
            for t in fixes:
                acts.append({"kind": "set_flags", "backend": b, "flags": t.name,
                             "settings": t.settings, "command": None, "verified": True,
                             "description": f"Verified by rerunning on {b}: setting "
                                            + "; ".join(t.settings) + " makes every checked row "
                                            "bitwise identical."})
        elif r.flag_trials:
            all_t = next((t for t in r.flag_trials if t.name == "all"), None)
            if all_t is not None and all_t.fixed:
                acts.append({"kind": "set_flags", "backend": b, "flags": "all",
                             "settings": all_t.settings, "command": None, "verified": True,
                             "description": "Only all tested flags together fix it on " + b + ": "
                                            + "; ".join(all_t.settings)})
            else:
                acts.append({"kind": "flags_do_not_fix", "backend": b, "command": None,
                             "verified": True,
                             "description": f"No tested torch flag set fixes it on {b} ("
                                            + ", ".join(t.name for t in r.flag_trials)
                                            + "). Deterministic flags make a fixed shape repeat "
                                            "the same bits; they do not make a row independent "
                                            "of the batch shape."})
        if r.token_flips or r.argmax_flips:
            acts.append({"kind": "check_flips", "backend": b, "command": None,
                         "description": f"Predictions changed: {r.argmax_flips} argmax flips, "
                                        f"greedy decoding differs in {r.token_flips} contexts. "
                                        "Evals and caches keyed on these outputs are not stable."})
    return acts


def to_agent(rep: InvarianceReport, model_file: Optional[str] = None) -> Dict[str, Any]:
    from . import __version__
    from .agent import SCHEMA_VERSION, _torch_version

    status = rep.status
    exit_code = 0 if status == INVARIANT else (2 if status == ERROR else 1)
    issues = [i.as_dict() for r in rep.results for i in r.issues]
    backends = []
    for r in rep.results:
        fm, fo = r.first_module, r.first_op
        ok = sum(1 for x in r.rows if x.cmp.cls == BITWISE)
        backends.append({
            "name": r.backend.name, "device": r.backend.device,
            "description": r.backend.description, "status": r.status, "codes": r.codes,
            "summary": {"pairs": len(r.rows), "bitwise": ok,
                        "close": sum(1 for x in r.rows if x.cmp.cls == CLOSE),
                        "different": sum(1 for x in r.rows if x.cmp.cls == DIFFERENT),
                        "argmax_flips": r.argmax_flips, "token_flip_contexts": r.token_flips},
            "rows": [x.as_dict() for x in r.rows],
            "first_module": fm.as_dict() if fm else None,
            "first_op": fo.as_dict() if fo else None,
            "non_invariant_ops": r.non_invariant_ops(),
            "modules": [m.as_dict() for m in r.modules[:20]],
            "flag_trials": [t.as_dict() for t in r.flag_trials],
            "notes": list(r.notes),
            "error": r.error or None,
        })
    return {
        "schema_version": SCHEMA_VERSION, "tool": "truestar", "command": "invariance",
        "truestar_version": __version__, "torch_version": _torch_version(),
        "status": status, "exit_code": exit_code,
        "model": {"name": rep.model_name,
                  "file": os.path.abspath(model_file) if model_file else None,
                  "dtype": str(rep.dtype).replace("torch.", "")},
        "settings": {"rows": rep.rows_checked, "contexts": [c.as_dict() for c in rep.contexts],
                     "decode_steps": rep.decode_steps},
        "codes": rep.codes, "issues": issues, "backends": backends,
        "skipped": [{"name": n, "reason": why, "code": "TS001", "severity": "info"}
                    for n, why in rep.skipped],
        "notes": list(rep.notes),
        "next_actions": next_actions(rep, model_file),
    }


# ----------------------------------------------------------------------------- command line


def run_cli(a) -> int:
    import json

    from .agent import dumps, error_to_agent
    from .backends import BackendUnavailable
    from .cli import ModelFileError, _load

    agent = a.format == "agent"
    quiet = contextlib.redirect_stdout(sys.stderr) if agent else contextlib.nullcontext()
    failure = None
    with quiet:
        try:
            sizes = [int(s) for s in a.sizes.split(",") if s]
            model, inputs, kwargs = _load(a.model)
            bd = 0
            for m in list(sys.modules.values())[::-1]:
                if getattr(m, "__file__", None) and os.path.abspath(m.__file__) == os.path.abspath(a.model):
                    bd = int(m.get_batch_dim()) if hasattr(m, "get_batch_dim") else 0
                    break
            rep = invariance(model, inputs, kwargs=kwargs,
                             backends=[s for s in a.backends.split(",") if s] if a.backends else None,
                             sizes=sizes, rows=a.rows, batch_dim=bd, decode_steps=a.decode,
                             seq_pad=a.seq_pad, try_flags=not a.no_flags,
                             model_name=os.path.splitext(os.path.basename(a.model))[0],
                             train=getattr(a, "train", False))
        except ModelFileError as exc:
            failure = ("TS020", str(exc))
        except BackendUnavailable as exc:
            failure = ("TS001", str(exc))
        except ValueError as exc:
            failure = ("TS021", str(exc))
    if failure is not None:
        if agent:
            doc = error_to_agent(failure[0], failure[1], a.model)
            doc["command"] = "invariance"
            print(dumps(doc))
        else:
            print(f"error [{failure[0]}]: {failure[1]}", file=sys.stderr)
        return 2
    doc = to_agent(rep, a.model)
    if agent:
        print(dumps(doc))
    else:
        print(rep.to_text())
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(doc, indent=2, allow_nan=False))
    return doc["exit_code"]
