"""`truestar rlpair`: do the RL trainer and the rollout sampler compute the same logprobs?

In RL post training the sampler (incremental decoding with a KV cache, batched and padded, often
another dtype or another engine) and the trainer (one teacher forced forward over the whole
sequence) should give the same logprob for every sampled token under the same weights. When they
do not, the importance ratio trainer / sampler is biased and training drifts off policy
(docs/PAIN_RESEARCH_2.md, N3). docs/RL_PAIR.md has the user guide.

Method
------
1. Logprobs. The trainer forward gives logits at positions P-1 .. T-2 for the tokens P .. T-1
   (P is the prompt length); the sampler returns its own logits (or logprobs) for the same tokens.
   Statistics: max and mean absolute logprob difference, the fraction of tokens over a threshold,
   the importance ratio exp(trainer - sampler) (min, max, std, the fraction outside the PPO clip
   range), a k3 KL estimate and the worst positions.
2. Position mapping. Both runs are recorded per module call. The sampler's calls of a step module
   (the model itself by default) mark the decode steps: step i feeds c_i tokens at sampler
   positions [o_i, o_i + c_i). A tensor of a module call in step i maps to the trainer's tensor of
   the same module call dim by dim: a dim of size c_i is the step's chunk, a dim of size o_i + c_i
   is the cache prefix (keys so far), a dim of size o_i the cache before the step, dim 0 the batch
   (sampler row and left padding per trainer row from the layout), equal dims map one to one.
3. Isolated replay per module. The sampler's inputs of a module, gathered over every step, are
   written into the trainer call's arguments at the mapped positions. The trainer's module then
   runs teacher forced on those inputs (reference), the sampler's module runs the same full
   sequence path in its own dtype, and when the dtypes differ a copy of the trainer's module in the
   sampler dtype gives the precision band. Comparing the sampler's incremental outputs with these
   separates three causes for every module: the incremental path computes differently (a decode
   path bug), the sampler's module computes differently even on the full sequence (another
   implementation or a weight bug), or only the dtype differs (precision). The first module in
   execution order whose own error is a real divergence and whose submodules are clean is blamed,
   with the first position where it differs.
4. Op drilldown in the blamed module: the sampler call at the first differing step is traced at
   the ATen level (by rerunning the sampler with a dispatch mode active only inside that call), as
   is the full sequence replay; ops are aligned by name, outputs mapped with the same position
   rules, and each op whose mapped output differs is rerun alone with the full path's inputs at
   the mapped positions. The first op that still differs is named, with its scalar arguments and,
   for a slice along positions, the start the sampler used against the position it should have.
5. Mismatch outside every module (no module differs): a sampling temperature applied to the
   sampler's logits is fitted and reported. Stale sampler weights are found by comparing
   parameters by name. Batch composition: the sampler runs the first row alone and inside batches
   of other rows, and its logprobs are compared bitwise.
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .capture import ROOT, Recorder, flatten_tensors, map_tensors
from .codes import Issue
from .compare import diff_tensors, threshold_for

# statuses
IDENTICAL, MATCH, PRECISION, DIVERGENT, ERROR = (
    "identical", "match", "precision", "divergent", "error")
# module classes
M_IDENTICAL, M_NOISE, M_PRECISION, M_DIVERGENT, M_UNMAPPED = (
    "identical", "noise", "precision", "divergent", "unmapped")
DECODE_PATH, IMPLEMENTATION = "decode_path", "implementation"

# Logprob difference that counts as rounding when both sides use this dtype.
LP_NOISE = {torch.float64: 1e-9, torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 5e-2}
# Relative error of an incremental against a full sequence computation in the same dtype that is
# still rounding (other reduction lengths and orders).
PATH_FLOOR = {torch.float64: 1e-10, torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 2e-2}
BAND_FACTOR = 3.0  # a sampler module within 3x the emulated dtype band is precision
MAX_DRILL_OPS = 5000

BATCH, FULL, CHUNK, PREFIX, PAST = "batch", "full", "chunk", "prefix", "past"


# ----------------------------------------------------------------------------- geometry


@dataclass
class Step:
    index: int
    offset: int  # sampler (padded) position of the first token fed in this step
    length: int  # tokens fed per row


@dataclass
class Geometry:
    """How sampler tensors line up with trainer tensors."""
    Bt: int  # trainer batch
    Bs: int  # sampler batch
    T: int  # trainer sequence length
    rows: List[int]  # sampler row of each trainer row
    pads: List[int]  # left padding of each trainer row in the sampler

    @property
    def uniform_pad(self) -> Optional[int]:
        return self.pads[0] if len(set(self.pads)) == 1 else None


def _dim_kinds(t_shape, s_shapes, steps: List[Step], g: Geometry) -> Optional[List[str]]:
    """Per dim: batch, full, chunk, prefix or past (see the module docstring). None when the
    tensors cannot be lined up."""
    nd = len(t_shape)
    if not s_shapes or any(len(s) != nd for s in s_shapes):
        return None
    kinds: List[str] = []
    for d in range(nd):
        tv, sv = t_shape[d], [s[d] for s in s_shapes]
        if d == 0 and tv == g.Bt and all(x == g.Bs for x in sv) and nd > 1:
            kinds.append(BATCH)
        elif tv == g.T and all(x == st.length for x, st in zip(sv, steps)):
            kinds.append(CHUNK)
        elif tv == g.T and all(x == st.offset + st.length for x, st in zip(sv, steps)):
            kinds.append(PREFIX)
        elif tv == g.T and all(x == st.offset and st.offset > 0 for x, st in zip(sv, steps)):
            kinds.append(PAST)
        elif all(x == tv for x in sv):
            kinds.append(FULL)
        else:
            return None
    positional = any(k in (CHUNK, PREFIX, PAST) for k in kinds)
    if positional and BATCH not in kinds and g.uniform_pad is None:
        return None  # per row padding but no batch dim to tell the rows apart
    return kinds


def _pairs(kinds: List[str], st: Step, g: Geometry) -> List[Tuple[tuple, tuple]]:
    """(trainer index, sampler index) slices for one step, one pair per trainer row (or one pair
    when the tensor has no batch dim)."""
    rows = list(range(g.Bt)) if BATCH in kinds else [None]
    out = []
    for b in rows:
        p = g.pads[b] if b is not None else (g.uniform_pad or 0)
        ti, si, empty = [], [], False
        for k in kinds:
            if k == BATCH:
                ti.append(slice(b, b + 1))
                si.append(slice(g.rows[b], g.rows[b] + 1))
            elif k == FULL:
                ti.append(slice(None))
                si.append(slice(None))
            else:
                lo, hi, base = {CHUNK: (st.offset, st.offset + st.length, st.offset),
                                PREFIX: (0, st.offset + st.length, 0),
                                PAST: (0, st.offset, 0)}[k]
                lo_t, hi_t = max(0, lo - p), min(g.T, hi - p)
                if hi_t <= lo_t:
                    empty = True
                    break
                ti.append(slice(lo_t, hi_t))
                si.append(slice(lo_t + p - base, hi_t + p - base))
        if not empty:
            out.append((tuple(ti), tuple(si)))
    return out


def _gather(t: torch.Tensor, s_list: List[torch.Tensor], kinds, steps, g) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mapped values of a trainer shaped tensor and the per step sampler tensors, flattened."""
    tv, sv = [], []
    for s, st in zip(s_list, steps):
        for ti, si in _pairs(kinds, st, g):
            tv.append(t[ti].reshape(-1))
            sv.append(s[si].reshape(-1))
    if not tv:
        return torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.float64)
    return torch.cat([x.double() if x.is_floating_point() else x for x in tv]), \
        torch.cat([x.double() if x.is_floating_point() else x for x in sv])


# ----------------------------------------------------------------------------- paths


def _paths(obj: Any, prefix: tuple = ()) -> List[Tuple[tuple, torch.Tensor]]:
    out: List[Tuple[tuple, torch.Tensor]] = []
    if isinstance(obj, torch.Tensor):
        out.append((prefix, obj))
    elif isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            out.extend(_paths(x, prefix + (i,)))
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            out.extend(_paths(obj[k], prefix + (k,)))
    return out


def _rebuild(obj: Any, repl: Dict[tuple, torch.Tensor], prefix: tuple = ()) -> Any:
    if isinstance(obj, torch.Tensor):
        return repl.get(prefix, obj)
    if isinstance(obj, tuple):
        return tuple(_rebuild(x, repl, prefix + (i,)) for i, x in enumerate(obj))
    if isinstance(obj, list):
        return [_rebuild(x, repl, prefix + (i,)) for i, x in enumerate(obj)]
    if isinstance(obj, dict):
        return {k: _rebuild(v, repl, prefix + (k,)) for k, v in obj.items()}
    return obj


def _path_name(path: tuple) -> str:
    if not path:
        return "input"
    head = "args" if path[0] == 0 else "kwargs"
    rest = "".join(f"[{p!r}]" if isinstance(p, str) else f"[{p}]" for p in path[1:])
    return head + rest


def _cast(obj: Any, device: str, dtype: Optional[torch.dtype]) -> Any:
    def fn(t: torch.Tensor) -> torch.Tensor:
        t = t.to(device)
        return t.to(dtype) if dtype is not None and t.is_floating_point() else t

    return map_tensors(obj, fn)


def _clone(obj: Any) -> Any:
    return map_tensors(obj, torch.clone)


def _cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", copy=True)


# ----------------------------------------------------------------------------- recording


@dataclass
class SCall:
    name: str
    type_name: str
    step: int
    local: int  # call index of this module inside the step
    global_index: int  # call index of this module over the whole sampler run
    args: Any
    kwargs: Dict[str, Any]
    outputs: List[torch.Tensor] = field(default_factory=list)


class _StepRecorder:
    """Records every module call of the sampler run with the decode step it belongs to."""

    def __init__(self, model: nn.Module, step_module: str, seq_dim: int = 1):
        self.model, self.step_module, self.seq_dim = model, step_module, seq_dim
        self.calls: List[SCall] = []
        self.steps: List[Tuple[int, int]] = []  # (batch, length) per step
        self.outside = 0
        self._stack: List[SCall] = []
        self._counts: Dict[str, int] = {}
        self._local: Dict[Tuple[int, str], int] = {}
        self._handles = []

    def __enter__(self):
        for name, mod in self.model.named_modules():
            label = name or ROOT
            self._handles.append(mod.register_forward_pre_hook(self._pre(label), with_kwargs=True))
            self._handles.append(mod.register_forward_hook(self._post(), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _pre(self, label: str):
        def hook(module, args, kwargs):
            if label == self.step_module:
                leaves = flatten_tensors((args, kwargs))
                if not leaves or leaves[0].dim() <= self.seq_dim:
                    raise ValueError(f"step module {label} takes no [batch, seq] tensor input")
                x = leaves[0]
                self.steps.append((int(x.shape[0]), int(x.shape[self.seq_dim])))
            step = len(self.steps) - 1
            gi = self._counts.get(label, 0)
            self._counts[label] = gi + 1
            li = self._local.get((step, label), 0)
            self._local[(step, label)] = li + 1
            call = SCall(label, type(module).__name__, step, li, gi,
                         map_tensors(args, _cpu), map_tensors(kwargs, _cpu))
            if step < 0:
                self.outside += 1
            self._stack.append(call)

        return hook

    def _post(self):
        def hook(module, args, kwargs, output):
            call = self._stack.pop()
            call.outputs = [_cpu(t) for t in flatten_tensors(output)]
            if call.step >= 0:
                self.calls.append(call)

        return hook


def _steps_from(raw: List[Tuple[int, int]]) -> List[Step]:
    out, off = [], 0
    for i, (_, c) in enumerate(raw):
        out.append(Step(i, off, c))
        off += c
    return out


# ----------------------------------------------------------------------------- results


@dataclass
class LogprobStats:
    tokens: int
    max_abs: float
    mean_abs: float
    threshold: float
    over: int
    ratio_min: float
    ratio_max: float
    ratio_mean: float
    ratio_std: float
    clip: float
    outside_clip: int
    kl_k3: float
    bitwise: bool
    worst: List[Dict[str, Any]]

    @property
    def over_fraction(self) -> float:
        return self.over / self.tokens if self.tokens else 0.0

    @property
    def ratio_spread(self) -> float:
        return self.ratio_max - self.ratio_min

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tokens": self.tokens, "bitwise": self.bitwise,
            "max_abs_diff": _num(self.max_abs), "mean_abs_diff": _num(self.mean_abs),
            "threshold": self.threshold, "over_threshold": self.over,
            "over_threshold_fraction": _num(self.over_fraction),
            "importance_ratio": {"min": _num(self.ratio_min), "max": _num(self.ratio_max),
                                 "mean": _num(self.ratio_mean), "std": _num(self.ratio_std),
                                 "spread": _num(self.ratio_spread), "clip": self.clip,
                                 "outside_clip": self.outside_clip},
            "kl_k3": _num(self.kl_k3),
            "worst": self.worst,
        }


@dataclass
class InputDiff:
    path: str
    integer: bool
    mismatches: int  # integer: values that differ; float: 0
    rel_err: Optional[float]
    first_position: Optional[int]
    sampler_value: Any = None
    trainer_value: Any = None

    def as_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "integer": self.integer, "mismatches": self.mismatches,
                "rel_err": _num(self.rel_err), "first_position": self.first_position,
                "sampler_value": self.sampler_value, "trainer_value": self.trainer_value}


@dataclass
class ModulePair:
    key: str  # trainer call key, e.g. "blocks.0.attn#0"
    name: str
    type_name: str
    start: int  # trainer execution order
    status: str
    kind: Optional[str] = None  # decode_path or implementation, for divergent
    total: Optional[float] = None  # sampler incremental vs trainer reference
    path: Optional[float] = None  # sampler incremental vs sampler module on the full sequence
    impl: Optional[float] = None  # sampler module full sequence vs trainer reference
    band: Optional[float] = None  # trainer module in the sampler dtype vs trainer reference
    threshold: float = 0.0
    path_threshold: float = 0.0
    first_position: Optional[int] = None
    first_step: Optional[int] = None
    steps: int = 0
    own: bool = False  # no divergent submodule
    input_diffs: List[InputDiff] = field(default_factory=list)
    note: str = ""
    source: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "name": self.name, "type": self.type_name, "status": self.status,
            "kind": self.kind, "own_code": self.own,
            "rel_err": _num(self.total), "path_rel_err": _num(self.path),
            "impl_rel_err": _num(self.impl), "precision_band": _num(self.band),
            "threshold": self.threshold, "path_threshold": _num(self.path_threshold),
            "first_position": self.first_position, "first_step": self.first_step,
            "steps": self.steps, "input_diffs": [d.as_dict() for d in self.input_diffs],
            "note": self.note or None, "source": self.source,
        }


@dataclass
class OpPair:
    op: str
    module: str
    step: int
    position: Optional[int]
    index: int  # 1 based among the sampler call's compared ops
    total: int
    matched: bool
    rel_err: Optional[float]
    threshold: float
    sampler_scalars: List[Any]
    trainer_scalars: Optional[List[Any]]
    input_shapes: List[List[int]]
    hint: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "module": self.module, "step": self.step,
                "position": self.position, "index": self.index, "total": self.total,
                "matched": self.matched, "rel_err": _num(self.rel_err),
                "threshold": self.threshold, "sampler_scalars": _jsonable(self.sampler_scalars),
                "trainer_scalars": _jsonable(self.trainer_scalars),
                "input_shapes": self.input_shapes, "hint": self.hint or None}


@dataclass
class BatchTrial:
    size: int
    position: int
    bitwise: bool
    max_abs: Optional[float]
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"batch_size": self.size, "position": self.position, "bitwise": self.bitwise,
                "max_abs_diff": _num(self.max_abs), "error": self.error or None}


@dataclass
class PairReport:
    model_name: str
    device: str
    train_dtype: torch.dtype
    sample_dtype: torch.dtype
    batch: int
    seq_len: int
    prompt_len: int
    stats: Optional[LogprobStats] = None
    status: str = ERROR
    lp_noise: float = 0.0
    modules: List[ModulePair] = field(default_factory=list)
    blamed: Optional[ModulePair] = None
    op: Optional[OpPair] = None
    temperature: Optional[Dict[str, Any]] = None
    weights: Optional[Dict[str, Any]] = None
    batch_trials: List[BatchTrial] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    layout: Optional[Dict[str, Any]] = None
    issues: List[Issue] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    error: str = ""
    seconds: float = 0.0
    kernels: Optional[str] = None  # "rl": both sides ran on the RL kernel set (truestar.rlkernels)
    kernel_swap: Optional[Dict[str, Any]] = None
    tp: Optional[Dict[str, Any]] = None

    def add(self, code: str, msg: str) -> None:
        self.issues.append(Issue(code, msg))

    @property
    def codes(self) -> List[str]:
        out: List[str] = []
        for i in self.issues:
            if i.code not in out:
                out.append(i.code)
        return out

    strict: bool = False

    @property
    def batch_invariant(self) -> Optional[bool]:
        done = [t for t in self.batch_trials if not t.error]
        return all(t.bitwise for t in done) if done else None

    @property
    def exit_code(self) -> int:
        if self.status == ERROR:
            return 2
        if self.tp is not None and not self.tp.get("bitwise"):
            return 1
        if self.kernels == "rl" and self.stats is not None and not self.stats.bitwise:
            return 1  # the RL kernel set promises bitwise equal logprobs
        if self.strict:  # bitwise or nothing: identical logprobs and a batch invariant sampler
            return 0 if self.status == IDENTICAL and self.batch_invariant is not False else 1
        return 0 if self.status in (IDENTICAL, MATCH) else 1

    def to_text(self, top: int = 5) -> str:
        return _text(self, top)

    def to_agent(self, pair_file: Optional[str] = None) -> Dict[str, Any]:
        return to_agent(self, pair_file)


def _num(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _jsonable(v: Any) -> Any:
    if v is None:
        return None
    out = []
    for x in v:
        if isinstance(x, float):
            out.append(x if math.isfinite(x) else str(x))
        elif isinstance(x, (int, bool, str)) or x is None:
            out.append(x)
        else:
            out.append(str(x))
    return out


def _dtype_of(model: nn.Module) -> torch.dtype:
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32


def _device_of(model: nn.Module) -> str:
    for p in model.parameters():
        return str(p.device)
    return "cpu"


def _short(d: torch.dtype) -> str:
    return str(d).replace("torch.", "")


# ----------------------------------------------------------------------------- logprobs


_REPLAY_CTX: Callable[[], Any] = contextlib.nullcontext  # rl_mode during a --kernels rl check


def _logprobs_from(out: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Logprobs of `targets` [B, N] from logits [B, N, V] (log_softmax in at least float32) or,
    when `out` is already [B, N], the logprobs themselves."""
    if out.dim() == 2 and tuple(out.shape) == tuple(targets.shape):
        return out.detach().float()
    if out.dim() != 3 or tuple(out.shape[:2]) != tuple(targets.shape):
        raise ValueError(f"expected logits [B, N, V] or logprobs [B, N] with [B, N] = "
                         f"{tuple(targets.shape)}, got shape {tuple(out.shape)}")
    z = out.detach()
    z = z.to(torch.promote_types(z.dtype, torch.float32))
    with _REPLAY_CTX():  # the RL kernels' log_softmax under --kernels rl
        return torch.log_softmax(z, dim=-1).gather(-1, targets.to(z.device)[..., None])[..., 0]


def logprob_stats(train_lp: torch.Tensor, sample_lp: torch.Tensor, tokens: Optional[torch.Tensor] = None,
                  prompt_len: int = 0, threshold: float = 1e-3, clip: float = 0.2,
                  top: int = 5) -> LogprobStats:
    """Mismatch statistics between trainer and sampler logprobs of the same tokens ([B, N])."""
    if train_lp.shape != sample_lp.shape:
        raise ValueError(f"trainer logprobs {tuple(train_lp.shape)} and sampler logprobs "
                         f"{tuple(sample_lp.shape)} differ in shape")
    a, b = train_lp.detach().cpu().double(), sample_lp.detach().cpu().double()
    d = a - b
    ad = d.abs().nan_to_num(nan=float("inf"))
    ratio = torch.exp(d)
    kl = (ratio - 1 - d)  # k3 estimator of KL(sampler || trainer), on sampler samples
    n = d.numel()
    worst = []
    if n:
        order = torch.argsort(ad.reshape(-1), descending=True)[:top]
        N = d.shape[1]
        for idx in order.tolist():
            r, j = divmod(idx, N)
            pos = prompt_len + j
            worst.append({
                "row": r, "position": pos,
                "token": int(tokens[r, pos]) if tokens is not None else None,
                "trainer_logprob": _num(a[r, j]), "sampler_logprob": _num(b[r, j]),
                "abs_diff": _num(ad[r, j]), "ratio": _num(ratio[r, j]),
            })
    bitwise = bool(torch.equal(train_lp.detach().cpu(), sample_lp.detach().cpu().to(train_lp.dtype))) \
        if train_lp.dtype == sample_lp.dtype else False
    return LogprobStats(
        tokens=n, max_abs=float(ad.max()) if n else 0.0, mean_abs=float(ad.mean()) if n else 0.0,
        threshold=threshold, over=int((ad > threshold).sum()),
        ratio_min=float(ratio.min()) if n else 1.0, ratio_max=float(ratio.max()) if n else 1.0,
        ratio_mean=float(ratio.mean()) if n else 1.0,
        ratio_std=float(ratio.std()) if n > 1 else 0.0, clip=clip,
        outside_clip=int(((ratio - 1).abs() > clip).sum()),
        kl_k3=float(kl.mean()) if n else 0.0, bitwise=bitwise, worst=worst,
    )


def fit_temperature(train_logits: torch.Tensor, sample_lp: torch.Tensor,
                    targets: torch.Tensor) -> Tuple[float, float]:
    """The temperature t for which log_softmax(trainer logits / t) best explains the sampler's
    logprobs: (t, max abs residual)."""
    z = train_logits.detach().cpu().double()
    s = sample_lp.detach().cpu().double()
    tg = targets.cpu()[..., None]

    def resid(t: float) -> float:
        return float((torch.log_softmax(z / t, -1).gather(-1, tg)[..., 0] - s).abs().mean())

    lo, hi = math.log(0.05), math.log(20.0)
    grid = [lo + (hi - lo) * i / 60 for i in range(61)]
    best = min(grid, key=lambda u: resid(math.exp(u)))
    a, b = best - (hi - lo) / 60, best + (hi - lo) / 60
    for _ in range(60):  # golden section in log t
        m1, m2 = a + 0.382 * (b - a), a + 0.618 * (b - a)
        if resid(math.exp(m1)) < resid(math.exp(m2)):
            b = m2
        else:
            a = m1
    t = math.exp((a + b) / 2)
    mx = float((torch.log_softmax(z / t, -1).gather(-1, tg)[..., 0] - s).abs().max())
    return t, mx


# ----------------------------------------------------------------------------- module pairs


def _module(model: nn.Module, name: str) -> nn.Module:
    return model if name == ROOT else model.get_submodule(name)


def _forward(mod: nn.Module, obj) -> List[torch.Tensor]:
    args, kwargs = obj
    with torch.no_grad(), _REPLAY_CTX():
        out = mod(*args, **kwargs)
    return [_cpu(t) for t in flatten_tensors(out)]


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    if not (a.is_floating_point() and b.is_floating_point()):
        return 0.0 if torch.equal(a, b) else float("inf")
    d = diff_tensors(a, b)
    return float("inf") if (d.nonfinite_mismatch or d.shape_mismatch) else d.rel_err


def _reconstruct(t_obj, s_calls: List[SCall], steps: List[Step], g: Geometry):
    """The trainer call's arguments with every tensor the sampler also passed (same path, mappable
    shape) replaced by the sampler's values at the mapped positions. Returns (object, diffs,
    mapped path count, unmapped path names)."""
    t_paths = _paths(t_obj)
    s_paths = [dict(_paths((c.args, c.kwargs))) for c in s_calls]
    repl: Dict[tuple, torch.Tensor] = {}
    diffs: List[InputDiff] = []
    unmapped: List[str] = []
    for path, t in t_paths:
        ss = [sp.get(path) for sp in s_paths]
        if any(s is None for s in ss):
            continue
        kinds = _dim_kinds(t.shape, [s.shape for s in ss], steps, g)
        if kinds is None or t.is_floating_point() != ss[0].is_floating_point():
            unmapped.append(_path_name(path))
            continue
        x = t.clone()
        for s, st in zip(ss, steps):
            for ti, si in _pairs(kinds, st, g):
                x[ti] = s[si].to(x.dtype)
        repl[path] = x
        # did the sampler feed this module something else than the trainer did?
        first_pos, mism, rel, sv, tv = None, 0, None, None, None
        if t.is_floating_point():
            tt, xx = _gather(t, ss, kinds, steps, g)
            rel = _rel(xx, tt)
        else:
            for s, st in zip(ss, steps):
                for ti, si in _pairs(kinds, st, g):
                    a, b = s[si], t[ti]
                    bad = a != b
                    if bool(bad.any()):
                        mism += int(bad.sum())
                        if first_pos is None:
                            first_pos = _first_position(kinds, st, g, ti, bad)
                            idx = bad.nonzero()[0].tolist()
                            sv, tv = a[tuple(idx)].item(), b[tuple(idx)].item()
            if mism:
                diffs.append(InputDiff(_path_name(path), True, mism, None, first_pos, sv, tv))
            continue
        if rel is not None and rel > 0:
            diffs.append(InputDiff(_path_name(path), False, 0, rel, None))
    return _rebuild(t_obj, repl, ()), diffs, len(repl), unmapped


def _first_position(kinds, st: Step, g: Geometry, ti: tuple, bad: torch.Tensor) -> Optional[int]:
    for d, k in enumerate(kinds):
        if k in (CHUNK, PREFIX, PAST):
            hits = bad.nonzero()
            if hits.numel():
                return int(ti[d].start + int(hits[:, d].min()))
    return None


def _module_pairs(rep: PairReport, train_model: nn.Module, sample_model: nn.Module, t_calls,
                  s_calls: List[SCall], steps: List[Step], g: Geometry, device: str,
                  t_dtype: torch.dtype, s_dtype: torch.dtype) -> List[ModulePair]:
    thr_t = threshold_for(t_dtype, _tf32(device), None)
    differ = t_dtype != s_dtype
    same_obj = train_model is sample_model
    by_key: Dict[Tuple[str, int], List[SCall]] = {}
    for c in s_calls:
        by_key.setdefault((c.name, c.local), []).append(c)
    t_index: Dict[str, int] = {}
    out: List[ModulePair] = []
    unmapped: List[str] = []
    for tc in sorted(t_calls, key=lambda c: c.start):
        k = t_index.get(tc.name, 0)
        t_index[tc.name] = k + 1
        sc = by_key.get((tc.name, k), [])
        mp = ModulePair(tc.key, tc.name, tc.type_name, tc.start, M_UNMAPPED, threshold=thr_t)
        out.append(mp)
        if not sc:
            mp.note = "not called by the sampler"
            unmapped.append(tc.key)
            continue
        st = [steps[c.step] for c in sc]
        mp.steps = len(st)
        x_obj, diffs, n_mapped, bad_paths = _reconstruct((tc.args, tc.kwargs), sc, st, g)
        mp.input_diffs = diffs
        if n_mapped == 0 and _paths((tc.args, tc.kwargs)):
            mp.note = "no input could be lined up with the sampler's"
            unmapped.append(tc.key)
            continue
        leaf_kinds = []
        for j, t in enumerate(tc.outputs):
            ss = [c.outputs[j] if j < len(c.outputs) else None for c in sc]
            if any(s is None for s in ss):
                leaf_kinds.append(None)
                continue
            leaf_kinds.append(_dim_kinds(t.shape, [s.shape for s in ss], st, g))
        if not any(k is not None for k in leaf_kinds):
            mp.note = "outputs could not be lined up with the sampler's"
            unmapped.append(tc.key)
            continue
        tmod, smod = _module(train_model, tc.name), _module(sample_model, tc.name)
        try:
            ref = _forward(tmod, _cast(_clone(x_obj), device, t_dtype))
        except Exception as exc:
            mp.note = f"trainer replay failed: {type(exc).__name__}: {exc}"
            unmapped.append(tc.key)
            continue
        full = ref
        if not (same_obj and not differ):
            try:
                full = _forward(smod, _cast(_clone(x_obj), device, s_dtype))
            except Exception as exc:
                full = None
                mp.note = f"the sampler's module does not run the full sequence path: {exc}"
        band = 0.0
        if differ:
            try:
                emu = copy.deepcopy(tmod).to(s_dtype)
                eo = _forward(emu, _cast(_clone(x_obj), device, s_dtype))
                band = _leaf_err(eo, ref, leaf_kinds, None, None, g, full_tensor=True)
            except Exception:
                band = 0.0
        inc = [[c.outputs[j] for c in sc] for j in range(len(tc.outputs))]

        def err(a_lists, b) -> float:
            return _leaf_err(a_lists, b, leaf_kinds, st, None, g)

        mp.total = err(inc, ref)
        mp.path = err(inc, full) if full is not None else None
        mp.impl = _leaf_err(full, ref, leaf_kinds, None, None, g, full_tensor=True,
                            ref_like=True) if full is not None else None
        mp.band = band if differ else None
        mp.path_threshold = max(PATH_FLOOR.get(s_dtype, 1e-4), BAND_FACTOR * band, thr_t)
        impl_thr = max(thr_t, BAND_FACTOR * band) if differ else thr_t
        bitwise = all(_bitwise_leaf(inc[j], ref[j], leaf_kinds[j], st, g)
                      for j in range(len(ref)) if leaf_kinds[j] is not None)
        if bitwise:
            mp.status = M_IDENTICAL
        elif mp.total <= thr_t:
            mp.status = M_NOISE
        elif mp.path is not None and mp.path > mp.path_threshold:
            mp.status, mp.kind = M_DIVERGENT, DECODE_PATH
        elif mp.impl is not None and mp.impl > impl_thr:
            mp.status, mp.kind = M_DIVERGENT, IMPLEMENTATION
        elif mp.path is None and mp.total > impl_thr:
            mp.status, mp.kind = M_DIVERGENT, IMPLEMENTATION
        elif differ:
            mp.status = M_PRECISION
        else:
            mp.status = M_NOISE
        if mp.status == M_DIVERGENT:
            base = full if (mp.kind == DECODE_PATH and full is not None) else ref
            thr = mp.path_threshold if mp.kind == DECODE_PATH else impl_thr
            for c, s_ in zip(sc, st):
                e = _leaf_err([[c.outputs[j]] for j in range(len(tc.outputs))], base, leaf_kinds,
                              [s_], None, g)
                if e > thr:
                    pad = g.uniform_pad if g.uniform_pad is not None else min(g.pads)
                    mp.first_step = s_.index
                    mp.first_position = max(0, s_.offset - pad)
                    break
    if unmapped:
        rep.notes.append(f"{len(unmapped)} trainer module call(s) could not be compared: "
                         + ", ".join(unmapped[:5]))
    return out


def _leaf_err(a, b, kinds, steps, _unused, g: Geometry, full_tensor: bool = False,
              ref_like: bool = False) -> float:
    """Worst relative error over output leaves. `a` is a list of per step lists of sampler
    tensors (incremental) unless `full_tensor`, when `a` holds trainer shaped tensors that are
    compared to `b` at the positions the sampler covers."""
    worst = 0.0
    for j, k in enumerate(kinds):
        if k is None or j >= len(b):
            continue
        if full_tensor:
            # compare full sequence tensors on the positions the sampler fed
            x, y = a[j], b[j]
            if x.shape != y.shape:
                return float("inf")
            cov = _covered(k, g, x.shape)
            e = _rel(x[cov].double() if x.is_floating_point() else x[cov],
                     y[cov].double() if y.is_floating_point() else y[cov])
        else:
            tv, sv = _gather(b[j], a[j], k, steps, g)
            e = _rel(sv, tv)
        worst = max(worst, e)
    return worst


def _covered(kinds, g: Geometry, shape) -> tuple:
    """Index of the trainer positions the sampler fed (all but the last token)."""
    idx = []
    for d, k in enumerate(kinds):
        if k in (CHUNK, PREFIX, PAST):
            idx.append(slice(0, g.T - 1))
        else:
            idx.append(slice(None))
    return tuple(idx)


def _bitwise_leaf(s_list, t, kinds, steps, g) -> bool:
    if kinds is None:
        return True
    for s, st in zip(s_list, steps):
        if s.dtype != t.dtype:
            return False
        for ti, si in _pairs(kinds, st, g):
            if not torch.equal(s[si], t[ti]):
                return False
    return True


def _tf32(device: str) -> bool:
    return device.startswith("cuda") and bool(torch.backends.cuda.matmul.allow_tf32)


def _assign_ownership(mods: List[ModulePair], t_calls) -> None:
    kids = {c.key: list(c.children) for c in t_calls}
    desc: Dict[str, List[str]] = {}

    def walk(k):
        if k in desc:
            return desc[k]
        acc = []
        for ch in kids.get(k, []):
            acc.append(ch)
            acc.extend(walk(ch))
        desc[k] = acc
        return acc

    bad = {m.key for m in mods if m.status == M_DIVERGENT}
    for m in mods:
        m.own = m.status == M_DIVERGENT and not any(d in bad for d in walk(m.key))


# ----------------------------------------------------------------------------- op drilldown


def _scalars(obj):
    from .invariance import _scalars as sc

    return sc(obj)


def _trace_sampler_call(sample_fn, sample_model, tokens, prompt_len, name: str, global_index: int):
    from . import opdrill

    mod = _module(sample_model, name)
    tracer = opdrill._make_tracer(MAX_DRILL_OPS)
    state = {"n": 0, "on": False}

    def pre(m, a):
        if state["n"] == global_index:
            tracer.__enter__()
            state["on"] = True
        state["n"] += 1

    def post(m, a, o):
        if state["on"]:
            tracer.__exit__(None, None, None)
            state["on"] = False

    h1, h2 = mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)
    try:
        with torch.no_grad():
            sample_fn(sample_model, tokens, prompt_len)
    finally:
        h1.remove()
        h2.remove()
        if state["on"]:
            tracer.__exit__(None, None, None)
    return tracer.ops, tracer.truncated


def _op_kinds(t: torch.Tensor, s: torch.Tensor, st: Step, g: Geometry):
    return _dim_kinds(t.shape, [s.shape], [st], g)


def _position_hint(rec, kinds_out, st: Step, g: Geometry) -> str:
    """For a slice or arange along the positions: the start the sampler used against the
    position of the step."""
    packet = str(rec.func.overloadpacket)
    if packet not in ("aten.slice", "aten.arange"):
        return ""
    ints = [x for x in _scalars((rec.args, rec.kwargs))
            if isinstance(x, int) and not isinstance(x, bool)]
    if packet == "aten.slice":
        if len(ints) < 2:
            return ""
        dim, start = ints[0], ints[1]
    else:
        if not ints:
            return ""
        dim, start = 0, ints[0]
    if kinds_out is None or dim >= len(kinds_out) or kinds_out[dim] != CHUNK:
        return ""
    off = start - st.offset
    if off == 0:
        return ""
    pad = g.uniform_pad
    where = (f"sampler position {st.offset} (trainer position {st.offset - pad})"
             if pad else f"position {st.offset}")
    what = "slices from" if packet == "aten.slice" else "counts positions from"
    return (f"{what} {start} for the step at {where}: offset {off:+d} (a position or KV cache "
            "offset bug)")


def drill_decode(sample_fn, sample_model, tokens, prompt_len, mp: ModulePair, s_call: SCall,
                 x_obj, st: Step, g: Geometry, device: str, s_dtype: torch.dtype,
                 thr: float, full_module: nn.Module) -> Tuple[Optional[OpPair], List[str]]:
    """First ATen op inside the blamed module's sampler call (at step `st`) whose own output
    differs from the full sequence path at the mapped positions."""
    from . import opdrill
    from .invariance import _MOVEMENT

    notes: List[str] = []
    s_ops, tr1 = _trace_sampler_call(sample_fn, sample_model, tokens, prompt_len, s_call.name,
                                     s_call.global_index)
    args, kwargs = _cast(_clone(x_obj), device, s_dtype)
    f_ops, tr2 = opdrill.trace_ops(full_module, args, kwargs, MAX_DRILL_OPS)
    if tr1 or tr2:
        return None, [f"op drilldown skipped for {mp.key}: more than {MAX_DRILL_OPS} ops"]
    s_ops = [o for o in s_ops if opdrill._comparable(o)]
    f_ops = [o for o in f_ops if opdrill._comparable(o)]
    pairs = opdrill._align([o.name for o in f_ops], [o.name for o in s_ops])
    pos = st.offset - (g.uniform_pad if g.uniform_pad is not None else min(g.pads))
    for i, so in enumerate(s_ops):
        if i not in pairs:
            continue
        fo = f_ops[pairs[i]]
        if len(so.outputs) != len(fo.outputs) or not so.outputs:
            continue
        ks = [_op_kinds(t, s, st, g) for t, s in zip(fo.outputs, so.outputs)]
        if any(k is None for k in ks):
            continue
        acc = max(_rel(*reversed(_gather(t, [s], k, [st], g))) for t, s, k in zip(fo.outputs, so.outputs, ks))
        if acc <= thr:
            continue
        # isolate: the sampler op on the full path's inputs at the mapped positions
        sl, fl = opdrill._leaves((so.args, so.kwargs)), opdrill._leaves((fo.args, fo.kwargs))
        if len(sl) != len(fl):
            continue
        new, ok = [], True
        for s, f in zip(sl, fl):
            k = _op_kinds(f, s, st, g)
            if k is None:
                ok = False
                break
            x = s.clone()
            for ti, si in _pairs(k, st, g):
                x[si] = f[ti].to(x.dtype)
            new.append(x)
        if not ok:
            continue
        it = iter(new)
        sub = opdrill._map((so.args, so.kwargs), lambda t: next(it))
        try:
            got = opdrill._call(so.func, sub[0], sub[1], device)
        except Exception:
            continue
        own = max(_rel(*reversed(_gather(t, [s], k, [st], g))) for t, s, k in zip(fo.outputs, got, ks))
        if own <= thr:
            continue  # inherited from an earlier op
        if str(so.func.overloadpacket) in _MOVEMENT and own < 1e-2:
            continue  # copies bits, cannot round
        return OpPair(so.name, mp.key, st.index, pos, i + 1, len(s_ops), True, own, thr,
                      _scalars((so.args, so.kwargs)), _scalars((fo.args, fo.kwargs)),
                      [list(t.shape) for t in sl], _position_hint(so, ks[0], st, g)), notes
    # Nothing matched differs on its own: ops only the sampling path runs are the likely cause.
    only = [(i, o) for i, o in enumerate(s_ops) if i not in pairs
            and str(o.func.overloadpacket) not in _MOVEMENT]
    if only:
        i, o = only[0]
        sc = _scalars((o.args, o.kwargs))
        consts = [x for x in sc if isinstance(x, float)]
        hint = f"runs only in the sampling path" + (
            f", with constant {consts[0]:g}" if consts else "")
        return OpPair(o.name, mp.key, st.index, pos, i + 1, len(s_ops), False, None, thr, sc,
                      None, [list(t.shape) for t in opdrill._leaves((o.args, o.kwargs))], hint), notes
    notes.append(f"op drilldown on {mp.key}: no single ATen op was confirmed wrong at step "
                 f"{st.index}.")
    return None, notes


def drill_impl(train_mod, sample_mod, mp: ModulePair, x_obj, device: str, s_dtype: torch.dtype,
               thr: float, step: Step) -> Tuple[Optional[OpPair], List[str]]:
    """The blamed module computes differently even on the full sequence: the regular op
    drilldown between the trainer's module and the sampler's module on the same inputs."""
    from . import opdrill

    ref_mod = copy.deepcopy(train_mod).to("cpu")
    args, kwargs = _cast(_clone(x_obj), "cpu", None)
    res, notes = opdrill.drill(ref_mod, sample_mod, mp.key, args, kwargs, device, s_dtype, thr,
                               device == "cpu")
    if res is None:
        return None, notes
    return OpPair(res.op, mp.key, step.index if step else 0, mp.first_position, res.index,
                  res.total, res.matched, res.isolated.rel_err, res.threshold,
                  _scalars((res.args, res.kwargs)), None, res.input_shapes), notes


# ----------------------------------------------------------------------------- main entry


def _layout(layout, Bt: int, Bs: int) -> Geometry:
    rows = list(range(Bt))
    pads = [0] * Bt
    if layout:
        if isinstance(layout, dict):
            rows = list(layout.get("rows", rows))
            pads = list(layout.get("left_pad", pads))
        else:
            rows, pads = list(layout.rows), list(layout.left_pad)
    if len(rows) != Bt or len(pads) != Bt:
        raise ValueError("layout rows and left_pad need one entry per trainer row")
    if any(r < 0 or r >= Bs for r in rows):
        raise ValueError(f"layout row index out of range for a sampler batch of {Bs}")
    return Geometry(Bt, Bs, 0, rows, pads)


def weight_check(train_model: nn.Module, sample_model: nn.Module) -> Optional[Dict[str, Any]]:
    """Parameters of the sampler that are not the trainer's (rounded to the sampler dtype)."""
    if train_model is sample_model:
        return None
    tp = dict(train_model.named_parameters())
    sp = dict(sample_model.named_parameters())
    stale, missing, worst = [], sorted(set(tp) - set(sp)), 0.0
    for n, p in tp.items():
        q = sp.get(n)
        if q is None:
            continue
        if q.shape != p.shape:
            stale.append(n)
            continue
        a = p.detach().to("cpu").to(q.dtype)
        b = q.detach().to("cpu")
        if not torch.equal(a, b):
            stale.append(n)
            worst = max(worst, float((a.double() - b.double()).abs().max()))
    return {"compared": len(set(tp) & set(sp)), "differ": stale, "missing_in_sampler": missing,
            "extra_in_sampler": sorted(set(sp) - set(tp)), "max_abs": worst}


def batch_check(sample_fn, sample_model, tokens: torch.Tensor, prompt_len: int,
                sizes: Sequence[int], seed: int = 1234) -> List[BatchTrial]:
    """Sampler logprobs of the first row alone versus inside batches of other rows."""
    row = tokens[:1]
    tg = row[:, prompt_len:]
    with torch.no_grad():
        alone = _logprobs_from(sample_fn(sample_model, row, prompt_len), tg)
    out: List[BatchTrial] = []
    g = torch.Generator().manual_seed(seed)
    vocab = int(tokens.max()) + 1
    for n in sizes:
        if n < 2:
            continue
        for p in sorted({0, n - 1}):
            filler = torch.randint(0, max(vocab, 2), (n, tokens.shape[1]), generator=g).to(tokens.device)
            filler[p] = row[0]
            try:
                with torch.no_grad():
                    got = _logprobs_from(sample_fn(sample_model, filler, prompt_len),
                                         filler[:, prompt_len:])[p:p + 1]
            except Exception as exc:
                out.append(BatchTrial(n, p, False, None, f"{type(exc).__name__}: {exc}"))
                continue
            same = bool(torch.equal(got.cpu(), alone.cpu()))
            out.append(BatchTrial(n, p, same, float((got.double() - alone.double()).abs().max().cpu())))
    return out


def pair_check(
    model: nn.Module,
    tokens: torch.Tensor,
    prompt_len: int,
    sample_fn: Callable,
    *,
    train_fn: Optional[Callable] = None,
    sample_model: Optional[nn.Module] = None,
    layout: Any = None,
    threshold: float = 1e-3,
    clip: float = 0.2,
    top: int = 5,
    batch_sizes: Sequence[int] = (2, 7),
    step_module: Optional[str] = None,
    localize: Optional[bool] = None,
    ops: bool = True,
    model_name: str = "model",
    strict: bool = False,
    kernels: Optional[str] = None,
    sampler_overrides: Optional[Dict[str, str]] = None,
    tp: Optional[int] = None,
    tp_fn: Optional[Callable] = None,
) -> PairReport:
    """Compare the trainer's teacher forced logprobs with the sampler's for the same tokens.

    `train_fn(model, tokens)` returns logits [B, T, V] (default `model(tokens)`).
    `sample_fn(sample_model, tokens, prompt_len)` runs the sampler (prefill plus decode with a KV
    cache, in any batch layout and dtype) and returns logits [B, T - prompt_len, V] or logprobs
    [B, T - prompt_len] for tokens[:, prompt_len:]. `sample_model` defaults to `model`. `layout`
    ({"rows": [...], "left_pad": [...]}) says where each trainer row sits in the sampler's batch
    when the sampler batches or pads; `step_module` is the module whose calls are the decode
    steps (default: the sampler model itself).

    `kernels="rl"` runs the trainer, the sampler and every replay inside
    `truestar.rlkernels.rl_mode()` (one batch and position invariant kernel set, CUDA only,
    docs/RL_KERNELS.md); when the logprobs are then not bitwise equal, the kernels each module ran
    on both sides are compared and the first op whose kernel differs is named (TS185).
    `sampler_overrides` (for example `{"attention": "eager"}`) runs those op families of the
    sampler on eager kernels, to inject a swap. `tp` with `tp_fn(model, tokens, tp)` (logits
    [B, T, V] of a simulated tensor parallel trainer forward) also compares the trainer's TP 1
    logprobs with TP `tp` bitwise (TS186).
    """
    global _REPLAY_CTX
    if kernels not in (None, "default", "rl"):
        raise ValueError(f"kernels must be 'default' or 'rl', got {kernels!r}")
    if kernels == "rl":
        from .rlkernels import available, rl_mode

        if not available():
            raise ValueError("--kernels rl needs CUDA and Triton (the RL kernel set has no CPU path)")
        train_fn0 = train_fn or (lambda m, x: m(x))
        sample_fn0 = sample_fn
        tp_fn0 = tp_fn

        def rl_train(m, x):
            with rl_mode():
                return train_fn0(m, x)

        def rl_sample(m, x, p):
            with rl_mode(overrides=sampler_overrides):
                return sample_fn0(m, x, p)

        def rl_tp(m, x, n):
            with rl_mode():
                return tp_fn0(m, x, n)

        old_ctx = _REPLAY_CTX
        _REPLAY_CTX = rl_mode
        try:
            rep = pair_check(model, tokens, prompt_len, rl_sample, train_fn=rl_train,
                             sample_model=sample_model, layout=layout, threshold=threshold,
                             clip=clip, top=top, batch_sizes=batch_sizes, step_module=step_module,
                             localize=localize, ops=ops, model_name=model_name, strict=strict,
                             tp=tp, tp_fn=rl_tp if tp_fn0 is not None else None)
        finally:
            _REPLAY_CTX = old_ctx
        rep.kernels = "rl"
        if rep.stats is not None and not rep.stats.bitwise:
            _kernel_swap(rep, model, sample_model if sample_model is not None else model,
                         tokens.to(_device_of(model)), prompt_len, train_fn0, sample_fn0,
                         sampler_overrides)
        return rep
    t0 = time.perf_counter()
    sample_model = model if sample_model is None else sample_model
    train_fn = train_fn or (lambda m, x: m(x))
    device = _device_of(model)
    t_dtype, s_dtype = _dtype_of(model), _dtype_of(sample_model)
    if tokens.dim() != 2:
        raise ValueError(f"tokens must be [batch, seq], got shape {tuple(tokens.shape)}")
    tokens = tokens.to(device)
    B, T = tokens.shape
    if not 0 < prompt_len < T:
        raise ValueError(f"prompt_len must be between 1 and {T - 1}")
    rep = PairReport(model_name, device, t_dtype, s_dtype, B, T, prompt_len)
    rep.strict = strict
    targets = tokens[:, prompt_len:]

    with torch.no_grad(), Recorder(model, keep_inputs=True) as trec:
        t_logits = train_fn(model, tokens)
    if t_logits.dim() != 3 or tuple(t_logits.shape[:2]) != (B, T):
        raise ValueError(f"train_fn must return logits [B, T, V] = [{B}, {T}, V], got "
                         f"{tuple(t_logits.shape)}")
    train_lp = _logprobs_from(t_logits[:, prompt_len - 1:T - 1], targets)

    step_name = step_module or ROOT
    with torch.no_grad(), _StepRecorder(sample_model, step_name) as srec:
        s_out = sample_fn(sample_model, tokens, prompt_len)
    sample_lp = _logprobs_from(s_out, targets)
    if sample_lp.shape != train_lp.shape:
        raise ValueError(f"sample_fn must return logits or logprobs for tokens[:, prompt_len:] "
                         f"= {tuple(train_lp.shape)}, got {tuple(s_out.shape)}")
    rep.stats = logprob_stats(train_lp, sample_lp, tokens, prompt_len, threshold, clip, top)
    rep.lp_noise = min(LP_NOISE.get(t_dtype, 1e-4), LP_NOISE.get(s_dtype, 1e-4))
    if srec.outside:
        rep.notes.append(f"{srec.outside} sampler module call(s) ran outside the step module "
                         f"{step_name} and were not mapped.")

    st = rep.stats
    if st.outside_clip:
        rep.add("TS198", f"{st.outside_clip}/{st.tokens} tokens have an importance ratio outside "
                         f"1 +- {clip} (min {st.ratio_min:.4g}, max {st.ratio_max:.4g})")

    rep.weights = weight_check(model, sample_model)
    if rep.weights and (rep.weights["differ"] or rep.weights["missing_in_sampler"]):
        w = rep.weights
        rep.add("TS194", f"{len(w['differ'])} sampler parameter(s) differ from the trainer's "
                         f"beyond {_short(s_dtype)} rounding (max abs {w['max_abs']:.3g}): "
                         + ", ".join(w["differ"][:5])
                         + (f"; missing in the sampler: {', '.join(w['missing_in_sampler'][:5])}"
                            if w["missing_in_sampler"] else ""))

    if st.bitwise:
        rep.status = IDENTICAL
    elif st.max_abs <= rep.lp_noise:
        rep.status = MATCH
        rep.add("TS192", f"logprobs differ only at rounding level: max |diff| {st.max_abs:.3g} "
                         f"<= {rep.lp_noise:g}")
    else:
        rep.status = DIVERGENT  # refined below

    steps = _steps_from(srec.steps)
    rep.steps = steps
    if steps:
        Bs = srec.steps[0][0]
        g = _layout(layout, B, Bs)
        g.T = T
        rep.layout = {"rows": g.rows, "left_pad": g.pads, "sampler_batch": Bs,
                      "steps": len(steps), "first_chunk": steps[0].length}
    else:
        g = None
        rep.notes.append(f"the step module {step_name} never ran in the sampler; positions cannot "
                         "be mapped, so no module is localized.")

    run_modules = (rep.status == DIVERGENT and localize is not False) or localize is True
    if run_modules and g is not None:
        mods = _module_pairs(rep, model, sample_model, trec.calls, srec.calls, steps, g, device,
                             t_dtype, s_dtype)
        _assign_ownership(mods, trec.calls)
        rep.modules = mods
        for m in mods:
            if m.status == M_DIVERGENT or m.input_diffs:
                try:
                    from .agent import module_source

                    m.source = module_source(model, m.name)
                except Exception:
                    m.source = None
        owners = [m for m in mods if m.own]
        rep.blamed = owners[0] if owners else None
        unm = [m for m in mods if m.status == M_UNMAPPED]
        if unm:
            rep.add("TS196", f"{len(unm)} module call(s) could not be mapped to sampler "
                             "positions: " + ", ".join(m.key for m in unm[:5]))
        int_diff = next((m for m in sorted(mods, key=lambda m: m.start)
                         if any(d.integer for d in m.input_diffs)), None)
        if int_diff is not None:
            d = next(d for d in int_diff.input_diffs if d.integer)
            rep.add("TS197", f"{int_diff.key} ({int_diff.type_name}) receives {d.path} that "
                             f"differs from the trainer's at position {d.first_position}: sampler "
                             f"{d.sampler_value}, trainer {d.trainer_value} ({d.mismatches} values)")
        if rep.status != DIVERGENT:
            pass  # localized on request only (localize=True); the logprobs already match
        elif rep.blamed is not None:
            b = rep.blamed
            where = (f" from position {b.first_position} (decode step {b.first_step})"
                     if b.first_position is not None else "")
            if b.kind == DECODE_PATH:
                why = (f"computes differently in the sampling path{where}: own error "
                       f"{b.path:.3g} against its own full sequence path (threshold "
                       f"{b.path_threshold:.1g})")
            else:
                why = (f"computes differently from the trainer's module on the same inputs"
                       f"{where}: own error {b.impl if b.impl is not None else b.total:.3g}")
                if b.band:
                    why += f" (dtype band {b.band:.2g})"
            rep.add("TS190", f"module {b.key} ({b.type_name}) {why}")
            own_w = [n for n in (rep.weights or {}).get("differ", [])
                     if b.name == ROOT or n.startswith(b.name + ".")]
            if own_w and b.kind == IMPLEMENTATION:
                rep.notes.append(f"{b.key} holds sampler weights that differ from the trainer's "
                                 f"({', '.join(own_w[:5])}); the weights, not an op, explain it.")
            elif ops:
                _drill_blamed(rep, model, sample_model, sample_fn, tokens, prompt_len, trec.calls,
                              srec.calls, steps, g, device, s_dtype)
        elif t_dtype != s_dtype and not any(m.status == M_DIVERGENT for m in mods):
            rep.status = PRECISION
            worst = max((m for m in mods if m.total is not None), key=lambda m: m.total,
                        default=None)
            rep.add("TS191", f"logprobs differ (max |diff| {st.max_abs:.3g}, mean "
                             f"{st.mean_abs:.3g}) only as far as the sampler's "
                             f"{_short(s_dtype)} against the trainer's {_short(t_dtype)} "
                             "explains: every module stays within its dtype band"
                             + (f" (largest own error {worst.key}: {worst.total:.3g}, band "
                                f"{worst.band:.3g})" if worst and worst.band is not None else ""))
        else:
            tau, resid = fit_temperature(t_logits[:, prompt_len - 1:T - 1], sample_lp, targets)
            rep.temperature = {"fitted": tau, "max_residual": resid,
                               "explains": bool(resid <= max(10 * rep.lp_noise, 1e-3)
                                                and abs(tau - 1.0) > 1e-3)}
            msg = ("logprobs differ but every module computes the same on the same positions: "
                   "the difference is in code outside the model's modules")
            if rep.temperature["explains"]:
                msg += (f"; the sampler's logprobs are log_softmax(logits / {tau:.4g}) of the "
                        f"trainer's logits (max residual {resid:.2g}): a sampling temperature "
                        "is applied before the logprobs")
            rep.add("TS193", msg)
    elif rep.status == DIVERGENT and t_dtype != s_dtype:
        rep.status = PRECISION
        rep.add("TS191", f"logprobs differ (max |diff| {st.max_abs:.3g}); the sampler runs in "
                         f"{_short(s_dtype)} and the trainer in {_short(t_dtype)} (not localized)")
    elif rep.status == DIVERGENT:
        rep.add("TS190", f"logprobs differ (max |diff| {st.max_abs:.3g}); not localized")

    if rep.weights and rep.weights["differ"] and rep.status in (IDENTICAL, MATCH, PRECISION):
        rep.status = DIVERGENT

    if tp:
        _tp_check(rep, model, tokens, prompt_len, tp, tp_fn, train_lp)

    if batch_sizes:
        try:
            rep.batch_trials = batch_check(sample_fn, sample_model, tokens, prompt_len, batch_sizes)
        except Exception as exc:
            rep.notes.append(f"batch composition check failed: {type(exc).__name__}: {exc}")
        bad = [t for t in rep.batch_trials if not t.bitwise and not t.error]
        if bad:
            w = max(bad, key=lambda t: t.max_abs or 0)
            rep.add("TS195", f"the sampler's logprobs for a row change with the batch it runs in: "
                             f"{len(bad)}/{len(rep.batch_trials)} batch contexts are not bitwise "
                             f"identical (max |diff| {w.max_abs:.3g} at batch {w.size}, position "
                             f"{w.position})")
    rep.seconds = time.perf_counter() - t0
    return rep


def _tp_check(rep: PairReport, model, tokens, prompt_len, tp: int, tp_fn, train_lp) -> None:
    """Trainer logprobs at TP 1 against a simulated TP `tp` forward, bitwise."""
    if tp_fn is None:
        rep.notes.append("--tp given but the pair file defines no tp_logits(model, tokens, tp); "
                         "tensor parallel not compared")
        return
    T = tokens.shape[1]
    with torch.no_grad():
        out = tp_fn(model, tokens, tp)
    lp = _logprobs_from(out[:, prompt_len - 1:T - 1], tokens[:, prompt_len:])
    same = bool(torch.equal(lp.detach().cpu(), train_lp.detach().cpu()))
    diff = float((lp.double() - train_lp.double()).abs().max().cpu())
    rep.tp = {"tp": tp, "bitwise": same, "max_abs": diff, "simulated": True}
    if not same:
        rep.add("TS186", f"trainer logprobs at TP 1 and TP {tp} (simulated on one device) are not "
                         f"bitwise equal: max |diff| {diff:.3g}")


def _kernel_seqs(trace) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key, op, kern in trace:
        out.setdefault(key or "?", []).append(f"{op}:{kern}")
    return out


def _kernel_swap(rep: PairReport, model, sample_model, tokens, prompt_len, train_fn, sample_fn,
                 overrides) -> None:
    """Logprobs differ on the RL kernels: compare the kernels each module ran in the trainer and
    in every sampler call of the same module, and name the first op whose kernel differs."""
    from .rlkernels import KernelTrace, rl_mode

    mt = rl_mode(trace=True)
    ms = rl_mode(trace=True, overrides=overrides)
    try:
        with torch.no_grad():
            with KernelTrace(model), mt:
                train_fn(model, tokens)
            with KernelTrace(sample_model), ms:
                sample_fn(sample_model, tokens, prompt_len)
    except Exception as exc:
        rep.notes.append(f"kernel trace failed: {type(exc).__name__}: {exc}")
        return
    t_seq, s_seq = _kernel_seqs(mt.trace), _kernel_seqs(ms.trace)
    order: List[str] = []
    for key, _, _ in mt.trace:
        if key is not None and key not in order:
            order.append(key)

    def call_index(k: str) -> int:
        tail = k.rsplit("#", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    for key in order:
        name = key.rsplit("#", 1)[0]
        tk = t_seq.get(key, [])
        for sk in sorted((k for k in s_seq if k.rsplit("#", 1)[0] == name), key=call_index):
            sv = s_seq[sk]
            if sv == tk:
                continue
            i = next((j for j, (a, b) in enumerate(zip(tk, sv)) if a != b), min(len(tk), len(sv)))
            tr = tk[i] if i < len(tk) else None
            sa = sv[i] if i < len(sv) else None
            rep.kernel_swap = {"module": key, "sampler_call": sk, "index": i, "trainer": tr,
                               "sampler": sa, "trainer_kernels": tk, "sampler_kernels": sv}
            rep.add("TS185", f"{key} runs {tr or 'nothing'} in the trainer but {sa or 'nothing'} "
                             f"in the sampler call {sk} (op {i} of the module): another kernel, "
                             "another reduction order")
            return
    eager = sorted({e.split(":")[0] for seqs in (t_seq, s_seq) for v in seqs.values()
                    for e in v if e.endswith(":eager")})
    rep.add("TS187", "logprobs are not bitwise equal although the trainer and the sampler ran the "
                     "same RL kernels in every module"
                     + (f" (eager calls: {', '.join(eager)})" if eager else "")
                     + ": the difference comes from ops outside the kernel set or from the inputs")


def _drill_blamed(rep, model, sample_model, sample_fn, tokens, prompt_len, t_calls, s_calls,
                  steps, g, device, s_dtype):
    b = rep.blamed
    tc = next(c for c in t_calls if c.key == b.key)
    k = int(b.key.rsplit("#", 1)[1])
    sc = [c for c in s_calls if c.name == b.name and c.local == k]
    x_obj, _, _, _ = _reconstruct((tc.args, tc.kwargs), sc, [steps[c.step] for c in sc], g)
    first = b.first_step if b.first_step is not None else sc[0].step
    call = next((c for c in sc if c.step == first), sc[0])
    smod = _module(sample_model, b.name)
    try:
        if b.kind == DECODE_PATH:
            op, notes = drill_decode(sample_fn, sample_model, tokens, prompt_len, b, call, x_obj,
                                     steps[call.step], g, device, s_dtype, b.path_threshold, smod)
        else:
            thr = max(b.threshold, BAND_FACTOR * (b.band or 0.0))
            op, notes = drill_impl(_module(model, b.name), smod, b, x_obj, device, s_dtype, thr,
                                   steps[call.step])
    except Exception as exc:
        op, notes = None, [f"op drilldown on {b.key} failed: {type(exc).__name__}: {exc}"]
    rep.op = op
    rep.notes.extend(notes)


# ----------------------------------------------------------------------------- output


def _fmt(x: Optional[float], spec: str = ".3g") -> str:
    return "n/a" if x is None else format(x, spec)


def _text(rep: PairReport, top: int) -> str:
    L = [f"truestar rlpair: {rep.model_name} on {rep.device}, trainer {_short(rep.train_dtype)}, "
         f"sampler {_short(rep.sample_dtype)}"]
    L.append(f"tokens {rep.batch} x {rep.seq_len}, prompt {rep.prompt_len}, "
             f"{rep.stats.tokens if rep.stats else 0} scored tokens, {len(rep.steps)} sampler steps"
             + (f", layout rows {rep.layout['rows']} left pad {rep.layout['left_pad']}"
                if rep.layout and any(rep.layout["left_pad"]) else ""))
    if rep.error:
        L.append(f"error: {rep.error}")
        return "\n".join(L)
    s = rep.stats
    L.append(f"logprobs: max |diff| {s.max_abs:.3g}, mean {s.mean_abs:.3g}, over {s.threshold:g}: "
             f"{s.over}/{s.tokens} ({100 * s.over_fraction:.1f}%), bitwise {'yes' if s.bitwise else 'no'}")
    L.append(f"importance ratio trainer/sampler: min {s.ratio_min:.6g}, max {s.ratio_max:.6g}, "
             f"std {s.ratio_std:.3g}, outside 1 +- {s.clip:g}: {s.outside_clip}/{s.tokens}, "
             f"KL k3 {s.kl_k3:.3g}")
    if s.worst and not s.bitwise:
        L.append("worst positions:")
        for w in s.worst[:top]:
            L.append(f"  row {w['row']} position {w['position']} token {w['token']}: trainer "
                     f"{_fmt(w['trainer_logprob'], '.6f')} sampler {_fmt(w['sampler_logprob'], '.6f')} "
                     f"diff {_fmt(w['abs_diff'])}")
    L.append(f"status: {rep.status.upper()}")
    for i in rep.issues:
        L.append(f"  [{i.code}] {i.message}")
    if rep.blamed is not None:
        src = (rep.blamed.source or {}).get("user") if rep.blamed.source else None
        if src:
            L.append(f"  source: {src['file']}:{src['line']} ({src['qualname']})")
    if rep.op is not None:
        o = rep.op
        L.append(f"  op: {o.op} (op {o.index} of {o.total} in {o.module}, step {o.step}"
                 + (f", own error {o.rel_err:.3g}" if o.rel_err is not None and math.isfinite(o.rel_err)
                    else (", own values differ" if o.rel_err is not None else "")) + ")"
                 + (f": {o.hint}" if o.hint else ""))
        L.append(f"      sampler scalars {_jsonable(o.sampler_scalars)}"
                 + (f", full path scalars {_jsonable(o.trainer_scalars)}" if o.trainer_scalars is not None else ""))
    if rep.modules:
        shown = [m for m in rep.modules if m.status in (M_DIVERGENT, M_PRECISION)]
        shown = sorted(shown, key=lambda m: -(m.total or 0))[:top]
        if shown:
            L.append("modules (own error against the trainer on the sampler's inputs):")
            for m in shown:
                L.append(f"  {m.key:<28} {m.type_name:<12} {m.status:<10} total {_fmt(m.total)} "
                         f"path {_fmt(m.path)} impl {_fmt(m.impl)}"
                         + (f" band {_fmt(m.band)}" if m.band is not None else ""))
    if rep.kernels:
        L.append(f"kernels: {rep.kernels} (truestar.rlkernels, one kernel set for trainer and sampler)")
    if rep.kernel_swap:
        k = rep.kernel_swap
        L.append(f"  kernel swap: {k['module']} op {k['index']}: trainer {k['trainer']}, sampler "
                 f"{k['sampler']} ({k['sampler_call']})")
    if rep.tp is not None:
        L.append(f"tensor parallel (simulated on one device): TP 1 vs TP {rep.tp['tp']} "
                 + ("bitwise" if rep.tp["bitwise"] else f"max |diff| {rep.tp['max_abs']:.3g}"))
    if rep.batch_trials:
        L.append("batch composition (sampler, first row alone vs inside a batch): "
                 + ", ".join(f"{t.size}@{t.position} "
                             + ("bitwise" if t.bitwise else (f"error" if t.error else f"{t.max_abs:.2g}"))
                             for t in rep.batch_trials))
    for n in rep.notes:
        L.append(f"  note: {n}")
    L.append(f"took {rep.seconds:.2f} s")
    return "\n".join(L)


def next_actions(rep: PairReport, pair_file: Optional[str] = None) -> List[Dict[str, Any]]:
    acts: List[Dict[str, Any]] = []
    b = rep.blamed
    if b is not None:
        src = (b.source or {}).get("user") if b.source else None
        acts.append({
            "kind": "inspect_source", "module": b.key, "command": None,
            "file": src.get("file") if src else None, "line": src.get("line") if src else None,
            "description": (f"{b.key} ({b.type_name}) is the first module whose own output differs "
                            + ("between incremental decoding and the full sequence"
                               if b.kind == DECODE_PATH else "between the sampler's and the "
                               "trainer's implementation")
                            + (f", from position {b.first_position}" if b.first_position is not None else "")
                            + (". Read its forward for code that only the cached path runs."
                               if b.kind == DECODE_PATH else
                               ". Compare the two implementations and their weights.")),
        })
    if rep.op is not None:
        o = rep.op
        kind = "fix_position_offset" if "offset" in (o.hint or "") else "inspect_op"
        acts.append({"kind": kind, "op": o.op, "module": o.module, "command": None,
                     "description": f"{o.op} in {o.module} at decode step {o.step}"
                                    + (f": {o.hint}" if o.hint else "")
                                    + f". Sampler scalars {_jsonable(o.sampler_scalars)}"
                                    + (f", full path {_jsonable(o.trainer_scalars)}"
                                       if o.trainer_scalars is not None else "") + "."})
    if any(i.code == "TS197" for i in rep.issues):
        acts.append({"kind": "check_position_ids", "command": None,
                     "description": next(i.message for i in rep.issues if i.code == "TS197")
                     + ". The sampler feeds other positions or masks than the trainer; fix the "
                       "code that builds them (cache length, padding offset)."})
    if rep.temperature and rep.temperature.get("explains"):
        tau = rep.temperature["fitted"]
        acts.append({"kind": "match_temperature", "temperature": tau, "command": None,
                     "description": f"The sampler's logprobs come from logits divided by {tau:.4g}. "
                                    "Return logprobs of the raw logits from the sampler, or divide "
                                    "the trainer's logits by the same temperature."})
    elif any(i.code == "TS193" for i in rep.issues):
        acts.append({"kind": "inspect_logprob_code", "command": None,
                     "description": "Every module matches; compare the code that turns logits "
                                    "into logprobs on both sides (temperature, top k or top p "
                                    "masking, log_softmax dtype)."})
    if rep.status == PRECISION:
        acts.append({"kind": "match_dtype", "command": None,
                     "description": f"Run the sampler in {_short(rep.train_dtype)} (or the trainer "
                                    f"forward in {_short(rep.sample_dtype)}) to remove the gap, or "
                                    "correct the off policy gap with truncated importance "
                                    "sampling on these ratios."})
    if rep.weights and rep.weights["differ"]:
        acts.append({"kind": "sync_weights", "params": rep.weights["differ"][:10], "command": None,
                     "description": "The sampler holds other weights than the trainer: sync the "
                                    "weights before the rollout (stale or partial weight update)."})
    if any(i.code == "TS195" for i in rep.issues):
        target = f'"{os.path.abspath(pair_file)}"' if pair_file else "<model.py>"
        acts.append({"kind": "check_batch_invariance", "command": f"truestar invariance {target}",
                     "description": "The sampler's logprobs depend on the batch. Run the batch "
                                    "invariance check on the model to name the op, and use batch "
                                    "invariant kernels in the sampler."})
    if rep.kernel_swap:
        k = rep.kernel_swap
        acts.append({"kind": "use_rl_kernels", "module": k["module"], "command": None,
                     "description": f"{k['module']} runs {k['trainer']} in the trainer and "
                                    f"{k['sampler']} in the sampler: run the same op on the same "
                                    "kernel on both sides (rl_mode, no override, no hand written "
                                    "decode path)."})
    if rep.tp is not None and not rep.tp.get("bitwise"):
        acts.append({"kind": "fix_all_reduce_order", "command": None,
                     "description": "TP logprobs differ from TP 1: split K into the kernel's fixed "
                                    "segments per rank and all reduce float32 partials in the fixed "
                                    "tree order (truestar.rlkernels.tp)."})
    if any(i.code == "TS196" for i in rep.issues) and b is None:
        acts.append({"kind": "check_mapping", "command": None,
                     "description": "Some module calls could not be lined up with sampler "
                                    "positions; pass a layout (rows, left_pad) or a step_module."})
    return acts


def to_agent(rep: PairReport, pair_file: Optional[str] = None) -> Dict[str, Any]:
    from . import __version__
    from .agent import SCHEMA_VERSION, _torch_version

    mods = rep.modules
    shown = [m for m in mods if m.status in (M_DIVERGENT, M_PRECISION) or m.input_diffs]
    return {
        "schema_version": SCHEMA_VERSION, "tool": "truestar", "command": "rlpair",
        "truestar_version": __version__, "torch_version": _torch_version(),
        "status": rep.status, "exit_code": rep.exit_code,
        "model": {"name": rep.model_name,
                  "file": os.path.abspath(pair_file) if pair_file else None,
                  "trainer_dtype": _short(rep.train_dtype), "sampler_dtype": _short(rep.sample_dtype),
                  "device": rep.device},
        "tokens": {"batch": rep.batch, "seq_len": rep.seq_len, "prompt_len": rep.prompt_len},
        "layout": rep.layout,
        "codes": rep.codes, "issues": [i.as_dict() for i in rep.issues],
        "logprobs": rep.stats.as_dict() if rep.stats else None,
        "strict": rep.strict,
        "batch_invariant": rep.batch_invariant,
        "lp_noise": rep.lp_noise,
        "blamed_module": rep.blamed.as_dict() if rep.blamed else None,
        "blamed_op": rep.op.as_dict() if rep.op else None,
        "temperature": rep.temperature,
        "weights": rep.weights,
        "modules_compared": len(mods),
        "module_status_counts": {s: sum(1 for m in mods if m.status == s) for s in
                                 (M_IDENTICAL, M_NOISE, M_PRECISION, M_DIVERGENT, M_UNMAPPED)},
        "modules": [m.as_dict() for m in sorted(shown, key=lambda m: m.start)[:20]],
        "batch_composition": [t.as_dict() for t in rep.batch_trials],
        "kernels": rep.kernels or "default",
        "kernel_swap": rep.kernel_swap,
        "tensor_parallel": rep.tp,
        "notes": list(rep.notes),
        "error": rep.error or None,
        "seconds": round(rep.seconds, 3),
        "next_actions": next_actions(rep, pair_file),
    }


# ----------------------------------------------------------------------------- command line


def _load_pair(path: str):
    """Load a pair file: get_model(), get_tokens(), get_prompt_len(), sample_logits(model, tokens,
    prompt_len); optional train_logits(model, tokens), get_sampler_model(model), get_layout()."""
    from .cli import ModelFileError

    if not os.path.isfile(path):
        raise ModelFileError(f"{path} not found")
    import importlib.util

    stem = os.path.splitext(os.path.basename(path))[0]
    name = stem if stem.isidentifier() and stem not in sys.modules else "truestar_user_pair"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        raise ModelFileError(f"importing {path} failed: {type(exc).__name__}: {exc}") from exc
    for fn in ("get_model", "get_tokens", "get_prompt_len", "sample_logits"):
        if not hasattr(mod, fn):
            raise ModelFileError(f"{path} must define {fn}()")
    return mod


def run_cli(a) -> int:
    import json

    from .agent import dumps, error_to_agent
    from .cli import ModelFileError

    agent = a.format == "agent"
    quiet = contextlib.redirect_stdout(sys.stderr) if agent else contextlib.nullcontext()
    failure = None
    rep = None
    with quiet:
        try:
            sizes = [int(s) for s in a.batch_sizes.split(",") if s] if a.batch_sizes else []
            mod = _load_pair(a.pair)
            try:
                model = mod.get_model()
                tokens = mod.get_tokens()
                P = int(mod.get_prompt_len())
                smodel = mod.get_sampler_model(model) if hasattr(mod, "get_sampler_model") else None
                layout = mod.get_layout() if hasattr(mod, "get_layout") else None
            except Exception as exc:
                raise ModelFileError(f"calling the pair file's functions failed: "
                                     f"{type(exc).__name__}: {exc}") from exc
            if a.device:
                if a.device.startswith("cuda") and not torch.cuda.is_available():
                    raise ValueError("--device cuda requested but CUDA is not available")
                model = model.to(a.device)
                if smodel is not None:
                    smodel = smodel.to(a.device)
            kernels = getattr(a, "kernels", None)
            tp = getattr(a, "tp", None)
            rep = pair_check(model, tokens, P, mod.sample_logits,
                             train_fn=getattr(mod, "train_logits", None), sample_model=smodel,
                             layout=layout, threshold=a.threshold, clip=a.clip, top=a.top,
                             batch_sizes=sizes, step_module=a.step_module, ops=not a.no_ops,
                             model_name=os.path.splitext(os.path.basename(a.pair))[0],
                             strict=a.strict, kernels=None if kernels == "default" else kernels,
                             tp=tp, tp_fn=getattr(mod, "tp_logits", None))
        except ModelFileError as exc:
            failure = ("TS020", str(exc))
        except ValueError as exc:
            failure = ("TS199", str(exc))
        except Exception as exc:
            failure = ("TS199", f"{type(exc).__name__}: {exc}")
    if failure is not None:
        if agent:
            doc = error_to_agent(failure[0], failure[1], a.pair)
            doc["command"] = "rlpair"
            print(dumps(doc))
        else:
            print(f"error [{failure[0]}]: {failure[1]}", file=sys.stderr)
        return 2
    doc = to_agent(rep, a.pair)
    if agent:
        print(dumps(doc))
    else:
        print(rep.to_text(top=a.top))
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(doc, indent=2, allow_nan=False))
    return rep.exit_code
