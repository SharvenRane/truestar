"""Static memory planning for compiled programs (docs/MEMORY.md).

Three parts:

* ``schedule(plan)``: liveness over the compiled steps, computed once per program (it does not
  depend on the symbol values). After each step the runtime drops every slot no later step reads,
  so the caching allocator can hand that storage to the next allocation. A kernel that is
  elementwise in one of its inputs (no reduction, every load of that input at the output's own
  index) writes its result into that input's buffer when the input dies at that step and nothing
  else can see it (a runtime guard checks reference counts before each such launch). Program inputs
  are reused only when the caller donates them.
* ``plan_memory(cp, dims)``: the exact bytes a call allocates, step by step, before anything is
  launched. A dry run on meta tensors gives each step's new storages and which outputs are views of
  which inputs; sizes come from the IR types at the given symbol values, rounded the way the CUDA
  caching allocator rounds (512 bytes). The result names the step, the op and the user's source
  line at the peak, and the values live there.
* ``check_budget``: a program whose planned peak exceeds ``memory_budget`` is refused before the
  first launch with ``MemoryBudgetError`` (TS420), naming the line at the peak and the largest
  value of the batch symbol that fits.

Value semantics make this exact: IR values never alias and are never mutated, so the only storage
sharing is what the compiled steps create themselves (views from layout ops, in place reuse), and
the planner sees all of it.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import sympy
import torch

from ..ir.core import IRError
from ..kernel import ir as kir

ALLOC_ROUND = 512          # the CUDA caching allocator rounds every block up to a multiple of 512 bytes
VIEW_OPS = ("reshape", "permute", "expand", "select", "slice", "select_dynamic")


class MemoryBudgetError(IRError):
    """TS420: the planned peak of a compiled program exceeds its memory budget. Raised before any
    kernel is launched. ``plan`` is the MemoryPlan that was over budget."""

    code = "TS420"

    def __init__(self, message: str, plan: "MemoryPlan", budget: int, fits: Optional[Dict[str, int]] = None):
        super().__init__(message)
        self.plan = plan
        self.budget = budget
        self.fits = fits or {}


# ----------------------------------------------------------------------------- sizes


def parse_bytes(v) -> int:
    """``6 * 2**30``, ``"6 GiB"``, ``"512MB"``, ``"1.5G"`` as a number of bytes."""
    if isinstance(v, bool):
        raise TypeError("a memory size cannot be a bool")
    if isinstance(v, (int, float)):
        return int(v)
    m = re.fullmatch(r"\s*([0-9.]+)\s*([kmgt]?)(i?)b?\s*", str(v).lower())
    if not m:
        raise ValueError(f"cannot read {v!r} as a memory size; use bytes or a string like '6 GiB'")
    num, unit, binary = float(m.group(1)), m.group(2), m.group(3)
    base = 1024 if (binary or unit) else 1
    if unit and not binary:
        base = 1000
    return int(num * base ** " kmgt".index(unit or " "))


def fmt_bytes(n: float) -> str:
    sign = "-" if n < 0 else ""
    n = abs(n)
    for unit, scale in (("GiB", 2 ** 30), ("MiB", 2 ** 20), ("KiB", 2 ** 10)):
        if n >= scale:
            return f"{sign}{n / scale:.2f} {unit}"
    return f"{sign}{int(n)} B"


_ROUND = [True]


def block_bytes(nbytes: int, device: str) -> int:
    """Bytes the allocator charges for a tensor of ``nbytes``: rounded to 512 on CUDA, 0 for empty.
    With rounding off (``plan_memory(..., rounding=False)``) the bytes requested, which is what
    torch's ``requested_bytes`` statistic counts."""
    if nbytes <= 0:
        return 0
    if device == "cuda" and _ROUND[0]:
        return ((nbytes + ALLOC_ROUND - 1) // ALLOC_ROUND) * ALLOC_ROUND
    return nbytes


_KSIZE_CACHE: Dict[str, Any] = {}


def eval_ksize(size, bindings: Dict[str, int]) -> int:
    """Value of a kernel size (an int, or a kernel symbol standing for an IR dim expression)."""
    if isinstance(size, int):
        return size
    from .lower import _SIZE_TAKEN
    text = _SIZE_TAKEN.get(size.name, size.name)
    expr = _KSIZE_CACHE.get(text)
    if expr is None:
        expr = _KSIZE_CACHE[text] = sympy.sympify(text)
    v = expr.subs({sympy.Symbol(n): x for n, x in bindings.items()})
    return int(v)


# ----------------------------------------------------------------------------- step reads and writes


def step_io(step, with_check: bool = True) -> Tuple[List[int], List[int]]:
    """(slots the step reads, slots it writes). Check reads count as reads, so a value a region check
    compares against lives until that check ran. Cached on the step (steps never change their slots)."""
    key = "_io_check" if with_check else "_io"
    hit = step.__dict__.get(key)
    if hit is None:
        hit = _step_io(step, with_check)
        step.__dict__[key] = hit
    return hit


def _step_io(step, with_check: bool) -> Tuple[List[int], List[int]]:
    from . import runtime as R
    s = step
    if isinstance(s, R.RecomputeStep):
        reads, writes = step_io(s.producer, with_check=False)
        return reads, writes
    if isinstance(s, R.OffloadStep):
        return [s.slot], [s.host]
    if isinstance(s, R.ReloadStep):
        return [s.host], [s.slot]
    if isinstance(s, (R.FenceStep, R.WaitStep)):
        return [s.slot], []
    if isinstance(s, R.KernelStep):
        reads, writes = list(s.in_slots), [s.out] + ([s.true_slot] if s.true_slot is not None else [])
    elif isinstance(s, R.AdapterStep):
        reads, writes = [s.src], [s.dst]
    elif isinstance(s, (R.ImplStep, R.CreateStep, R.LibraryStep, R.IntStep, R.RandomStep)):
        reads, writes = [x for x in s.in_slots if x is not None], [s.out]
    elif isinstance(s, (R.HostStep, R.IndexStep)):     # FallbackStep is a HostStep
        reads, writes = [x for x in s.in_slots if x is not None], list(s.outs)
    elif isinstance(s, R.DimScalarStep):
        reads, writes = [], [s.out]
    elif isinstance(s, R.CondStep):
        reads, writes = [s.pred] + list(s.args), list(s.outs)
    elif isinstance(s, R.LoopStep):
        reads, writes = list(s.carried) + list(s.captures), list(s.outs)
    else:
        raise IRError(f"the memory planner does not know step kind {type(s).__name__}")
    if with_check and s.check is not None:
        reads += [slot for _, _, slot, const in s.check.ext if slot is not None and const is None]
    return reads, writes


# ----------------------------------------------------------------------------- schedule (liveness)


@dataclass
class Schedule:
    """What the runtime does around each step, by step index. Independent of the symbol values."""

    frees: List[Tuple[int, ...]]              # slots dropped after the step
    inplace: Dict[int, int]                   # step index: position of the input whose buffer the output takes
    inplace_donated: Dict[int, int]           # the same, for program inputs, used only when donated
    stash: Dict[int, Tuple[int, ...]]         # recompute checks: slots copied before the step (check runs only)
    recompute_slots: Dict[int, Tuple[int, ...]]   # recompute step index: slots compared bitwise with the stash
    reuse: bool = True
    # steps whose output is a program output: when the call donates, the donated input's buffer is
    # taken before an intermediate's (a carried value keeps its address)
    donated_first: Set[int] = field(default_factory=set)
    # step index of an inplace_donated write: the program input whose buffer it writes over, also
    # through earlier in place writes (a chain of index_copy steps over one KV cache)
    donated_root: Dict[int, int] = field(default_factory=dict)


class _Union:
    """Union find over slots that may share storage, with the members of every group."""

    def __init__(self):
        self.parent: Dict[int, int] = {}
        self.members: Dict[int, List[int]] = {}

    def find(self, a: int) -> int:
        if a not in self.parent:
            self.parent[a] = a
            self.members[a] = [a]
            return a
        p = a
        while self.parent[p] != p:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb
            self.members[rb].extend(self.members.pop(ra))

    def group(self, a: int) -> List[int]:
        return self.members[self.find(a)]


def elementwise_inputs(ck) -> List[int]:
    """Input positions of a compiled kernel whose buffer its output may overwrite: one output, no
    reduction anywhere, the input has the output's shape and dtype, and every load of it is at the
    output's own index. Each output element is then computed only from loads at that element, by the
    program instance that stores it, before it stores it, so writing over the input is safe."""
    hit = _EW_CACHE.get(id(ck))
    if hit is not None and hit[0] is ck:
        return hit[1]
    res = _elementwise_inputs(ck)
    _EW_CACHE[id(ck)] = (ck, res)
    return res


_EW_CACHE: Dict[int, Tuple[Any, List[int]]] = {}


def _elementwise_inputs(ck) -> List[int]:
    k = ck.kernel
    if len(k.outputs) != 1:
        return []
    o = k.outputs[0]
    if kir.contains_reduce(o.expr):
        return []
    loads = [n for n in kir.walk(o.expr) if isinstance(n, kir.Load)]
    res = []
    for j, b in enumerate(k.inputs):
        if tuple(b.shape) != tuple(o.shape) or b.dtype != o.dtype:
            continue
        mine = [ld for ld in loads if ld.buffer.name == b.name]
        if mine and all(tuple(ld.index) == tuple(o.axes) for ld in mine):
            res.append(j)
    return res


def ranges(plan, steps=None) -> Tuple[Dict[int, List[Tuple[int, int]]], List[Tuple[List[int], List[int]]]]:
    """Live ranges of every slot: [(first event, last event)], one range per write. Program inputs
    and params start live at -1."""
    steps = plan.steps if steps is None else steps
    io = [step_io(s) for s in steps]
    live: Dict[int, List[Tuple[int, int]]] = {}
    for s in list(plan.in_slots) + [i for i, t in enumerate(plan.template) if t is not None]:
        live.setdefault(s, [(-1, -1)])
    for i, (reads, writes) in enumerate(io):
        for s in reads:
            if s not in live:
                raise IRError(f"internal: {plan.path} step {i} reads slot {s} before any step writes it")
            a, _ = live[s][-1]
            live[s][-1] = (a, i)
        for s in writes:
            live.setdefault(s, []).append((i, i))
    return live, io


def schedule(plan, reuse: bool = True) -> Schedule:
    """Liveness and in place decisions for one plan (a program or a region)."""
    from . import runtime as R
    n = len(plan.steps)
    sched = Schedule([()] * n, {}, {}, {}, {}, reuse)
    recompute_at = {i: s for i, s in enumerate(plan.steps) if isinstance(s, (R.RecomputeStep, R.ReloadStep))}
    live, io = ranges(plan)
    outs = set(plan.out_slots)
    if recompute_at:       # check runs keep a copy of a value before the step that drops it
        stash: Dict[int, List[int]] = {}
        for i, st in recompute_at.items():
            slots = tuple(io[i][1])
            sched.recompute_slots[i] = slots
            for s in slots:
                prev = [r for r in live[s] if r[0] < i]
                if prev:
                    stash.setdefault(prev[-1][1], []).append(s)
        sched.stash = {i: tuple(v) for i, v in stash.items()}
    if not reuse:
        return sched
    frees: List[List[int]] = [[] for _ in range(n)]
    for s, rs in live.items():
        for k, (a, b) in enumerate(rs):
            last = k == len(rs) - 1
            if last and s in outs:
                continue
            if b >= 0:
                frees[b].append(s)
    sched.frees = [tuple(f) for f in frees]
    # in place: union find over slots that may share storage (views, region passthrough)
    groups = _Union()
    for i, st in enumerate(plan.steps):
        reads, writes = io[i]
        if isinstance(st, R.ImplStep) and st.nodes[0].op in VIEW_OPS:
            groups.union(writes[0], reads[0])
        elif isinstance(st, (R.CondStep, R.LoopStep)):
            for w in writes:
                for r in reads:
                    groups.union(w, r)
    if plan.opts.device != "cuda":
        return sched
    template = {i for i, t in enumerate(plan.template) if t is not None}
    inputs = set(plan.in_slots)

    def dead_after(s: int, i: int) -> bool:
        for m in groups.group(s):
            for a, b in live.get(m, []):
                if a <= i < b:
                    return False
        return True

    for i, st in enumerate(plan.steps):
        if type(st) is R.IndexStep and st.node.op == "index_copy" and len(st.outs) == 1:
            eligible = [0]       # index_copy rewrites rows of its input 0 and keeps the rest
        elif type(st) is not R.KernelStep or st.scalar:
            continue
        else:
            eligible = elementwise_inputs(st.ck)
        scalar_ins = getattr(st, "scalar_ins", ())
        # A program output written over a donated input keeps its address from call to call: a
        # training step's new momentum takes the old momentum's buffer, not the gradient's, so the
        # carried state stays put (what a CUDA graph replay needs) at the same peak. Other steps
        # prefer an intermediate, as before.
        prefer_donated = st.out in outs and any(
            any(m in inputs for m in groups.group(st.in_slots[j])) for j in eligible if j not in scalar_ins)
        for j in eligible:
            if j in scalar_ins:
                continue
            s = st.in_slots[j]
            group = groups.group(s)
            # a view of a param or of a program output must never be written over; a view of a program
            # input only when the caller donates it
            if any(m in outs or m in template for m in group) or st.in_slots.count(s) > 1                     or not dead_after(s, i):
                continue
            if any(groups.find(x) == groups.find(s) for k, x in enumerate(st.in_slots) if k != j):
                continue
            groups.union(st.out, s)
            if any(m in inputs for m in group):   # a program input: only when the caller donates it
                sched.inplace_donated.setdefault(i, j)
                sched.donated_root.setdefault(i, next(m for m in group if m in inputs))
                continue
            sched.inplace.setdefault(i, j)   # an intermediate takes precedence over a donated input,
            if not prefer_donated:           # unless the output is a program output (above)
                break
        if prefer_donated and i in sched.inplace_donated:
            sched.donated_first.add(i)
    return sched


# ----------------------------------------------------------------------------- the runtime guard


def _counts(t: torch.Tensor) -> Tuple[int, int]:
    return sys.getrefcount(t), torch._C._storage_Use_Count(t.untyped_storage()._cdata)


def _owner_counts(t: torch.Tensor) -> Tuple[int, int]:
    return _counts(t)


def _calibrate() -> Tuple[int, int]:
    """The counts a tensor held by exactly one list slot shows through ``sole_owner``'s call path
    (they depend on the interpreter: measured, not assumed)."""
    env = [torch.empty(1)]
    return _owner_counts(env[0])


_SOLE = _calibrate()


def sole_owner(t: torch.Tensor) -> bool:
    """True when the runtime's slot is the only reference to ``t`` and to its storage, and ``t`` covers
    its whole storage: then writing over it cannot be seen by anyone. Called as ``sole_owner(env[s])``.
    A caller that kept the tensor, a view of it anywhere, or a second slot holding it, all raise one of
    the two counts, and the step then allocates a new buffer instead."""
    return (_counts(t) == _SOLE and t.storage_offset() == 0 and t.is_contiguous()
            and t.untyped_storage().nbytes() == t.numel() * t.element_size())


# ----------------------------------------------------------------------------- dry run (sizes, views)


@dataclass
class StepFacts:
    writes: List[Tuple[int, int, Optional[int]]]   # (slot, bytes of a new storage, or the slot it is a view of)
    extra: int = 0                                 # bytes allocated and freed inside the step
    region_peak: Optional[int] = None              # cond, loop: peak inside, relative to entry
    note: str = ""                                 # why this step is not exact, if it is not


class _Fake:
    """Stands in for the runtime when an index step runs on meta tensors. Its range check records
    the current node with every pending check; there is none in a dry run, and nothing is read back."""

    pending: list
    checked: dict
    node = None

    def __init__(self):
        self.pending, self.checked = [], {}


def _meta(shape, dtype) -> torch.Tensor:
    return torch.empty(tuple(int(x) for x in shape), dtype=dtype, device="meta")


def _dev_bytes(t: torch.Tensor, on_device: bool, device: str) -> int:
    return block_bytes(t.numel() * t.element_size(), device) if on_device else 0


def dry_run(plan, dims: List[int], bindings: Dict[str, int], metas: List[torch.Tensor]) -> Tuple[List[StepFacts], Dict[int, torch.Tensor]]:
    """Facts per step at these symbol values, from meta tensors only: nothing touches a device."""
    from . import runtime as R
    from .lower import kind_of
    opts = plan.opts
    dev = opts.device
    wdt = opts.torch_dtype
    env: Dict[int, torch.Tensor] = {}
    ondev: Dict[int, bool] = {}
    for i, t in enumerate(plan.template):
        if t is not None:
            env[i] = torch.empty_strided(tuple(t.shape), t.stride(), dtype=t.dtype, device="meta")
            ondev[i] = t.device.type == dev
    for s, m in zip(plan.in_slots, metas):
        env[s] = m
        ondev[s] = True
    facts: List[StepFacts] = []

    def shape_of(idx):
        return tuple(dims[i] for i in idx)

    def fresh(slot, t, on=True):
        env[slot] = t
        ondev[slot] = on
        return (slot, _dev_bytes(t, on, dev), None)

    def view(slot, t, src):
        env[slot] = t
        ondev[slot] = ondev.get(src, True)
        return (slot, 0, src)

    for st in plan.steps:
        f = StepFacts([])
        recompute = isinstance(st, R.RecomputeStep)
        if recompute:                 # the producer run again (never in place: that is a schedule decision)
            st = st.producer
        if isinstance(st, R.OffloadStep):       # host memory: not device bytes
            f.writes.append(fresh(st.host, env[st.slot], on=False))
        elif isinstance(st, R.ReloadStep):      # a new device buffer of the value
            f.writes.append(fresh(st.slot, env[st.host]))
        elif isinstance(st, (R.FenceStep, R.WaitStep)):
            pass
        elif isinstance(st, R.KernelStep):
            shape = tuple(eval_ksize(x, bindings) for x in st.out_sizes) or (1,)
            f.writes.append(fresh(st.out, _meta(shape, st.out_dtype or wdt)))
            if st.true_slot is not None:
                f.writes.append(fresh(st.true_slot, _meta(shape if st.out_sizes else (), torch.bool)))
        elif isinstance(st, R.AdapterStep):
            f.writes.append(fresh(st.dst, torch.empty(env[st.src].shape, dtype=st.dtype, device="meta")))
        elif isinstance(st, R.ImplStep):
            node = st.nodes[0]
            shape = st.static_shape if st.static_shape is not None else shape_of(st.shape_idx)
            args = [env[x] for x in st.in_slots]
            if node.op == "select_dynamic":
                d = node.attrs["dim"] % args[0].dim()
                out = args[0].select(d, 0)
            else:
                out = st.fn(st.attrs, args, shape)
            src = next((x for x, a in zip(st.in_slots, args) if torch._C._is_alias_of(out, a)), None)
            f.writes.append(view(st.out, out, src) if src is not None else fresh(st.out, out))
        elif isinstance(st, R.CreateStep):
            if st.op == "arange":
                n = dims[st.shape_idx[0]]
                out = torch.empty((n,), dtype=st.dtype, device="meta")
            else:
                out = _meta(shape_of(st.shape_idx), st.dtype)
            f.writes.append(fresh(st.out, out, torch.device(st.device).type == dev))
        elif isinstance(st, R.DimScalarStep):
            # cached per runtime after its first call: resident in a later call, never freed
            env[st.out] = torch.empty((), dtype=st.dtype, device="meta")
            ondev[st.out] = False
            f.writes.append((st.out, 0, None))
        elif isinstance(st, R.HostStep):      # also FallbackStep: float64 on the CPU, result moved over
            for slot, o, idx in zip(st.outs, st.node.outputs, st.shape_idx):
                k = kind_of(o.type.dtype)
                shape = shape_of(idx)
                if k == "f":
                    f.writes.append(fresh(slot, _meta(shape, wdt)))
                elif k == "b" or len(shape) > 0:
                    f.writes.append(fresh(slot, _meta(shape, o.type.dtype)))
                else:
                    f.writes.append(fresh(slot, _meta(shape, o.type.dtype), dev == "cpu"))
        elif isinstance(st, R.LibraryStep):
            args = [None if x is None else env[x] for x in st.in_slots]
            if st.op == "conv2d":
                out = torch.nn.functional.conv2d(args[0], args[1], args[2], **st.kw)
                f.note = "cuDNN workspace" if dev == "cuda" else ""
            elif st.op == "max_pool2d":
                out = torch.nn.functional.max_pool2d(args[0], **st.kw)
            else:
                out = torch.nn.functional.adaptive_avg_pool2d(args[0], **st.kw)
            f.writes.append(fresh(st.out, out.contiguous()))
        elif isinstance(st, R.IndexStep):
            res = st.fn(_Fake(), st.attrs, [None if x is None else env[x] for x in st.in_slots])
            for slot, r in zip(st.outs, res):
                f.writes.append(fresh(slot, r.contiguous()))
            f.note = "index range checks"
        elif isinstance(st, R.RandomStep):
            shape = shape_of(st.shape_idx)
            declared = st.node.output.type.dtype
            f.writes.append(fresh(st.out, _meta(shape, st.working)))
            if declared != st.working:        # drawn in the declared dtype, then cast
                f.extra = _dev_bytes(_meta(shape, declared), True, dev)
            f.note = "random generator temporaries"
        elif isinstance(st, R.IntStep):
            o = st.nodes[0].output
            f.writes.append(fresh(st.out, _meta(shape_of(st.shape_idx), o.type.dtype)))
        elif isinstance(st, R.CondStep):
            peaks, outs = [], None
            sub_metas = [env[x] for x in st.args]
            for branch in (st.then, st.other):
                sub = plan_region(branch, dims, bindings, sub_metas)
                peaks.append(sub.peak)
                if outs is None or sum(b for _, b, _ in sub.out_writes) > sum(b for _, b, _ in outs):
                    outs = sub.out_writes
                    out_metas = sub.out_metas
            f.region_peak = max(peaks)
            for slot, (_, b, src), m in zip(st.outs, outs, out_metas):
                f.writes.append(view(slot, m, st.args[src]) if src is not None else fresh(slot, m))
            f.note = "branch taken at run time: the larger branch is planned"
        elif isinstance(st, R.LoopStep):
            carried = [env[x] for x in st.carried]
            caps = [env[x] for x in st.captures]
            body = plan_region(st.body, dims, bindings, [torch.empty((), dtype=torch.int64, device="meta")]
                               + carried + caps)
            carried_new = sum(b for _, b, _ in body.out_writes)
            trips = dims[st.trip] if st.trip is not None else 2
            peak = body.peak
            if st.cond is not None:
                peak = max(peak, plan_region(st.cond, dims, bindings, carried + caps).peak)
            f.region_peak = ((carried_new if trips >= 2 else 0) + peak) if trips >= 1 else 0
            parent = [None] + list(st.carried) + list(st.captures)    # region input position: parent slot
            for k, (slot, (_, b, src), m) in enumerate(zip(st.outs, body.out_writes, body.out_metas)):
                if trips < 1:
                    f.writes.append(view(slot, m, st.carried[k]))
                elif src is None or parent[src] is None:
                    f.writes.append(fresh(slot, m))
                else:
                    f.writes.append(view(slot, m, parent[src]))
            f.note = "" if st.cond is None else "while loop: at least two iterations are planned"
        else:
            raise IRError(f"the memory planner does not know step kind {type(st).__name__}")
        facts.append(f)
    return facts, env


# ----------------------------------------------------------------------------- walk (the timeline)


@dataclass
class _Region:
    peak: int
    out_writes: List[Tuple[int, int, Optional[int]]]   # per region output: (slot, new bytes, region input position)
    out_metas: List[torch.Tensor]


def covers_of(env: Dict[int, torch.Tensor]) -> Dict[int, bool]:
    """Per slot: the tensor starts its storage, is contiguous and spans all of it (the guard's
    shape conditions, read from the dry run's meta tensors)."""
    return {s: (t.storage_offset() == 0 and t.is_contiguous()
                and t.untyped_storage().nbytes() == t.numel() * t.element_size()) for s, t in env.items()}


def plan_region(plan, dims, bindings, metas) -> _Region:
    facts, env = dry_run(plan, dims, bindings, metas)
    w = walk(plan, facts, plan.schedule, donated=False, covers=covers_of(env))
    pos = {s: k for k, s in enumerate(plan.in_slots)}
    outs = []
    for s in plan.out_slots:
        root = w.storage_root.get(s, s)
        if root in pos:
            outs.append((s, 0, pos[root]))
        else:
            outs.append((s, w.slot_bytes.get(s, 0), None))
    return _Region(w.peak, outs, [env[s] for s in plan.out_slots])


@dataclass
class Walk:
    peak: int = 0
    peak_step: int = -1
    timeline: List[int] = field(default_factory=list)           # live bytes during each step (with its extra)
    live_at_peak: List[Tuple[Tuple[int, ...], int]] = field(default_factory=list)   # (slots, bytes)
    end: int = 0                                                 # live after the call (outputs and their copies)
    storage_root: Dict[int, int] = field(default_factory=dict)   # slot: first slot of its storage
    slot_bytes: Dict[int, int] = field(default_factory=dict)
    inplace_steps: int = 0


def walk(plan, facts: List[StepFacts], sched: Schedule, donated: bool, clone_outputs: bool = False,
         input_bytes: Sequence[int] = (), want_live: bool = True, covers: Optional[Dict[int, bool]] = None) -> Walk:
    """Live bytes through one call, from the facts and the schedule, relative to before the call.
    Inputs and params exist before the call; a donated input that dies lowers the count."""
    w = Walk()
    sid: Dict[int, int] = {}
    holders: Dict[int, int] = {}
    size: Dict[int, int] = {}
    pinned: set = set()
    root: Dict[int, int] = {}
    nid = [0]

    def new(slot, nbytes, pin=False):
        k = nid[0]
        nid[0] += 1
        sid[slot], holders[k], size[k], root[k] = k, 1, nbytes, slot
        if pin:
            pinned.add(k)
        return k

    def release(slot):
        k = sid.pop(slot, None)
        if k is None:
            return 0
        holders[k] -= 1
        if holders[k] == 0 and k not in pinned:
            return size[k]
        return 0

    for i, t in enumerate(plan.template):
        if t is not None:
            new(i, 0, pin=True)
    for k, s in enumerate(plan.in_slots):
        if s not in sid:
            new(s, int(input_bytes[k]) if donated and k < len(input_bytes) else 0, pin=not donated)
    live = 0
    for i, f in enumerate(facts):
        ip = sched.inplace.get(i)
        if donated and (ip is None or i in sched.donated_first) and i in sched.inplace_donated:
            ip = sched.inplace_donated[i]
        step = plan.steps[i]
        if ip is not None:       # the runtime guard, mirrored: sole holder of a storage it covers
            k = sid.get(step.in_slots[ip])
            if k is None or holders[k] != 1 or k in pinned or (covers is not None and
                                                                not covers.get(step.in_slots[ip], True)):
                ip = None
        before = live
        for slot, nbytes, src in f.writes:
            live -= release(slot)   # a recompute step writes a slot again (normally dropped already)
            if ip is not None and slot == step.out:
                src = step.in_slots[ip]
                nbytes = 0
                w.inplace_steps += 1
            if src is not None and src in sid:
                k = sid[src]
                sid[slot] = k
                holders[k] += 1
            else:
                new(slot, nbytes)
                live += nbytes
            w.slot_bytes[slot] = nbytes
        during = live + f.extra
        if f.region_peak is not None:
            during = max(during, before + f.region_peak)
        w.timeline.append(during)
        if during > w.peak:
            w.peak, w.peak_step = during, i
            if want_live:
                w.live_at_peak = _alive(sid, size, pinned)
        for s in sched.frees[i] if sched.reuse else ():
            live -= release(s)
    # the call returns: every slot but the outputs is dropped, then aliased outputs are copied
    outs = set(plan.out_slots)
    for s in list(sid):
        if s not in outs:
            live -= release(s)
    if clone_outputs:
        before = live
        for s, alias in zip(plan.out_slots, plan.alias_out):
            if alias:
                live += w.slot_bytes.get(s, 0) or _bytes_of_storage(sid, size, s)
        if live > w.peak:
            w.peak, w.peak_step = live, len(facts)
        for s, alias in zip(plan.out_slots, plan.alias_out):
            if alias:
                live -= release(s)
    w.end = live
    w.storage_root = {s: root[k] for s, k in sid.items()}
    return w


def _alive(sid, size, pinned):
    alive: Dict[int, List[int]] = {}
    for s, k in sid.items():
        if k not in pinned and size[k] > 0:
            alive.setdefault(k, []).append(s)
    return sorted(((tuple(v), size[k]) for k, v in alive.items()), key=lambda x: -x[1])


def _bytes_of_storage(sid, size, s):
    return size.get(sid.get(s), 0)


# ----------------------------------------------------------------------------- the plan report


@dataclass
class MemoryPlan:
    """Planned device memory of one call of a compiled program at given symbol values."""

    program: str
    device: str
    dtype: str
    dims: Dict[str, int]
    peak_bytes: int                 # allocated during the call, above what existed before it
    resident_bytes: int             # params and inputs, on the device before the call
    output_bytes: int               # still allocated after the call (the outputs)
    hold_peak_bytes: int            # the same call with every value held to the end (no reuse)
    no_inplace_peak_bytes: int      # with liveness but without in place reuse
    peak_step: Dict[str, Any]
    live_at_peak: List[Dict[str, Any]]
    timeline: List[Dict[str, Any]]
    inplace_steps: int
    donated: bool
    notes: List[str] = field(default_factory=list)
    recompute: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        """Params, inputs and the peak of the call: what must fit on the device."""
        return self.resident_bytes + self.peak_bytes

    @property
    def saved_by_reuse(self) -> float:
        return 1.0 - self.peak_bytes / self.hold_peak_bytes if self.hold_peak_bytes else 0.0

    def top_steps(self, n: int = 5) -> List[Dict[str, Any]]:
        return sorted(self.timeline, key=lambda e: -e["live"])[:n]

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": "truestar.compiler.memory/1", "program": self.program, "device": self.device,
                "dtype": self.dtype, "dims": self.dims, "peak_bytes": self.peak_bytes,
                "resident_bytes": self.resident_bytes, "total_bytes": self.total_bytes,
                "output_bytes": self.output_bytes, "hold_peak_bytes": self.hold_peak_bytes,
                "no_inplace_peak_bytes": self.no_inplace_peak_bytes, "saved_by_reuse": self.saved_by_reuse,
                "peak_step": self.peak_step, "live_at_peak": self.live_at_peak, "inplace_steps": self.inplace_steps,
                "donated": self.donated, "notes": self.notes, "recompute": self.recompute, "timeline": self.timeline}

    def to_text(self, top: int = 5) -> str:
        dims = ", ".join(f"{k}={v}" for k, v in self.dims.items()) or "no symbols"
        ps = self.peak_step
        lines = [f"memory plan: {self.program} on {self.device}, {self.dtype}, {dims}",
                 f"  total {fmt_bytes(self.total_bytes)}: params and inputs {fmt_bytes(self.resident_bytes)}, "
                 f"peak during the call {fmt_bytes(self.peak_bytes)}",
                 f"  without reuse (every value held to the end of the call) {fmt_bytes(self.hold_peak_bytes)}: "
                 f"{100 * self.saved_by_reuse:.0f}% saved; in place steps {self.inplace_steps}"
                 + (", inputs donated" if self.donated else ""),
                 f"  peak at {ps.get('kind', '')} {ps.get('name', '')}: {', '.join(ps.get('ops', []))}"
                 + (f" ({ps['source']})" if ps.get("source") else "")]
        if self.live_at_peak:
            lines.append("  live at the peak:")
            for v in self.live_at_peak[:top]:
                lines.append(f"    {fmt_bytes(v['bytes']):>11}  {v['value']}"
                             + (f"  {v['source']}" if v.get("source") else ""))
        rec = [r for r in self.recompute if r.get("method") != "offload"]
        off = [r for r in self.recompute if r.get("method") == "offload"]
        if rec:
            lines.append(f"  recomputed: {len(rec)} values, "
                         f"{fmt_bytes(sum(r['bytes'] for r in rec))} not stored")
        if off:
            lines.append(f"  offloaded to host memory: {len(off)} values, "
                         f"{fmt_bytes(sum(r['bytes'] for r in off))} off the device between the passes")
        for nt in self.notes:
            lines.append(f"  note: {nt}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()


def _user_line(source: str) -> str:
    """The user's part of a node source ("model.py:12 in forward: ..."), else the whole source."""
    m = re.search(r"[\w.\-]+\.py:\d+ in [^:]+: .*", source or "")
    return m.group(0) if m else (source or "")


def _names(plan) -> Dict[int, Tuple[str, str]]:
    """slot: (value name, source line of its producer)."""
    L = plan.lowering
    out: Dict[int, Tuple[str, str]] = {}
    for lv in L.lvs:
        src = _user_line(lv.node.source) if lv.node is not None else ""
        name = ("%" + lv.value.name) if lv.value is not None else (lv.name or f"step{lv.id}")
        if lv.leaf and lv.value is not None:
            name = lv.value.ref
        if lv.slot is not None:
            out.setdefault(lv.slot, (name, src))
        if lv.kslot is not None and lv.kslot != lv.slot:
            out.setdefault(lv.kslot, (name + " (as float)", src))
    for st in plan.steps:
        for o in (o for n in st.nodes for o in n.outputs):
            lv = L.vals.get(id(o))
            if lv is not None and lv.slot is not None:
                out.setdefault(lv.slot, ("%" + o.name, _user_line(st.nodes[0].source)))
    return out


def _step_entry(step, live: int) -> Dict[str, Any]:
    src = next((_user_line(n.source) for n in step.nodes if n.source), "")
    return {"name": step.name, "kind": step.kind, "ops": step.ops(), "source": src, "live": live}


def input_metas(cp, dims_or_inputs) -> Tuple[Dict[str, int], List[torch.Tensor]]:
    """(symbol values, meta tensors of the inputs in their working form) from a dims dict or tensors."""
    from ..ir import reference as iref
    from ..ir.core import eval_dim, symbol
    from .lower import kind_of
    prog = cp.program
    opts = cp.opts
    if isinstance(dims_or_inputs, dict):
        values = {str(k): int(v) for k, v in dims_or_inputs.items()}
        missing = [s for s in cp.table.symbols if s not in values]
        if missing:
            raise ValueError(f"plan_memory needs a value for every symbol of {prog.name}; missing: "
                             f"{', '.join(missing)}")
        bind = {symbol(k): v for k, v in values.items()}
        shapes = [tuple(eval_dim(d, bind) for d in v.type.shape) for v in prog.inputs]
    else:
        ts = list(dims_or_inputs)
        b = iref.bind_symbols(prog, ts)
        values = {s: int(b[symbol(s)]) for s in cp.table.symbols}
        shapes = [tuple(t.shape) for t in ts]
    metas = []
    from .lower import input_storage_dtypes
    for v, shape, sdt in zip(prog.inputs, shapes, input_storage_dtypes(prog, opts)):
        k = kind_of(v.type.dtype)
        dt = sdt if k == "f" else v.type.dtype
        metas.append(torch.empty(shape, dtype=dt, device="meta"))
    return values, metas


def plan_memory(cp, dims_or_inputs=None, donate: bool = False, rounding: bool = True) -> MemoryPlan:
    prev = _ROUND[0]
    _ROUND[0] = rounding
    try:
        return _plan_memory(cp, dims_or_inputs, donate)
    finally:
        _ROUND[0] = prev


def _plan_memory(cp, dims_or_inputs=None, donate: bool = False) -> MemoryPlan:
    """The planned device memory of one call of ``cp`` (a CompiledProgram). ``dims_or_inputs`` is a
    dict of symbol values or the input tensors; None uses the largest declared value of every
    symbol. Nothing is launched and no device memory is touched. Inputs are assumed to arrive on the
    executor's device in its precision."""
    if dims_or_inputs is None:
        dims_or_inputs = max_dims(cp)
    values, metas = input_metas(cp, dims_or_inputs)
    plan = cp.plan
    dims = plan.table.evaluate(tuple(values[s] for s in plan.table.symbols))
    facts, env = dry_run(plan, dims, values, metas)
    cov = covers_of(env)
    clone = cp._clone
    dev = cp.opts.torch_device.type
    in_bytes = [block_bytes(m.numel() * m.element_size(), dev) for m in metas]
    w = walk(plan, facts, plan.schedule, donated=donate, clone_outputs=clone, input_bytes=in_bytes, covers=cov)
    hold = walk(plan, facts, Schedule(plan.schedule.frees, {}, {}, {}, {}, reuse=False), donated=False,
                clone_outputs=clone)
    noip = walk(plan, facts, Schedule(plan.schedule.frees, {}, {}, {}, {}, reuse=plan.schedule.reuse),
                donated=donate, clone_outputs=clone, input_bytes=in_bytes)
    resident = sum(block_bytes(t.numel() * t.element_size(), dev) for t in _distinct(plan.template)
                   if t.device.type == dev)
    resident += sum(block_bytes(m.numel() * m.element_size(), dev) for m in metas)
    names = _names(plan)
    if w.peak_step >= len(plan.steps):
        peak_step = {"name": "copies of aliased outputs", "kind": "return", "ops": [], "source": "", "live": w.peak}
    elif w.peak_step >= 0:
        peak_step = _step_entry(plan.steps[w.peak_step], w.peak)
    else:
        peak_step = {"name": "", "kind": "", "ops": [], "source": "", "live": 0}
    live = []
    for slots, nbytes in w.live_at_peak:
        nm = [names.get(s, (f"slot {s}", "")) for s in slots]
        live.append({"value": " = ".join(dict.fromkeys(n for n, _ in nm)), "bytes": nbytes,
                     "source": next((src for _, src in nm if src), "")})
    timeline = [_step_entry(st, b) for st, b in zip(plan.steps, w.timeline)]
    notes = []
    kinds = {}
    for f in facts:
        if f.note:
            kinds[f.note] = kinds.get(f.note, 0) + 1
    for nt, n in kinds.items():
        if nt == "cuDNN workspace":
            notes.append(f"{n} convolutions: cuDNN's workspace is allocated and freed inside the library call "
                         f"and is not planned (its size depends on the algorithm cuDNN picks)")
        elif nt == "random generator temporaries":
            notes.append(f"{n} random steps: the keyed generator's own temporaries are not planned")
        elif nt == "index range checks":
            notes.append(f"{n} index steps: their range check temporaries are not planned")
        else:
            notes.append(f"{n} steps: {nt}")
    return MemoryPlan(cp.program.name, dev, cp.opts.dtype, values, w.peak, resident, w.end, hold.peak, noip.peak,
                      peak_step, live, timeline, w.inplace_steps, donate, notes,
                      list(getattr(cp, "recompute_report", []) or []))


def _distinct(ts):
    seen = set()
    for t in ts:
        if t is None:
            continue
        k = t.untyped_storage().data_ptr() if t.device.type != "meta" else id(t)
        if k in seen:
            continue
        seen.add(k)
        yield t


def _finite(v) -> Optional[int]:
    """An upper bound as an int, or None when there is none (None, sympy oo, float inf)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if math.isfinite(f) else None


def max_dims(cp) -> Dict[str, int]:
    """The largest declared value of every symbol; raises when one has no upper bound."""
    out = {}
    for s in cp.table.symbols:
        lo, hi = cp.program.symbols.get(s, (1, None))
        hi = _finite(hi)
        if hi is None:
            raise ValueError(f"symbol {s} of {cp.program.name} has no upper bound: pass memory_dims={{'{s}': ...}} "
                             f"(or the inputs) to plan memory")
        out[s] = int(hi)
    return out


def largest_fit(cp, symbol: str, budget: int, dims: Dict[str, int], donate: bool = False) -> Optional[int]:
    """The largest value of ``symbol`` (others as in ``dims``) whose planned total fits ``budget``,
    or None when even its smallest value does not fit. A binary search over the planner, no launch."""
    lo, hi = cp.program.symbols.get(symbol, (1, None))
    hi = _finite(hi)
    lo = max(1, int(lo))
    hi = int(hi) if hi is not None else max(lo, int(dims.get(symbol, lo))) * 64

    def fits(v):
        d = dict(dims)
        d[symbol] = v
        return plan_memory(cp, d, donate).total_bytes <= budget

    if not fits(lo):
        return None
    if fits(hi):
        return hi
    a, b = lo, hi       # fits(a), not fits(b)
    while b - a > 1:
        m = (a + b) // 2
        if fits(m):
            a = m
        else:
            b = m
    return a


def check_budget(cp, budget: int, dims: Dict[str, int], donate: bool = False) -> MemoryPlan:
    """Plan at ``dims``; raise MemoryBudgetError (TS420) when the total exceeds ``budget``."""
    mp = plan_memory(cp, dims, donate)
    if mp.total_bytes <= budget:
        return mp
    fits = {}
    for s in cp.table.symbols:
        try:
            v = largest_fit(cp, s, budget, dims, donate)
        except Exception:  # noqa: BLE001 (the hint is best effort; the error is raised anyway)
            v = None
        if v is not None:
            fits[s] = v
    ps = mp.peak_step
    dims_text = ", ".join(f"{k}={v}" for k, v in mp.dims.items()) or "its fixed shapes"
    lines = [f"[TS420] compiled program {cp.program.name!r} on {cp.opts.device} ({cp.opts.dtype}) needs "
             f"{fmt_bytes(mp.total_bytes)} at {dims_text} (params and inputs {fmt_bytes(mp.resident_bytes)}, "
             f"peak during the call {fmt_bytes(mp.peak_bytes)}), over the memory budget of {fmt_bytes(budget)}.",
             f"  peak at {ps.get('kind', '')} {ps.get('name', '')}: {', '.join(ps.get('ops', []))}"
             + (f"\n    at {ps['source']}" if ps.get("source") else "")]
    if mp.live_at_peak:
        lines.append("  largest values live at the peak:")
        for v in mp.live_at_peak[:5]:
            lines.append(f"    {fmt_bytes(v['bytes']):>11}  {v['value']}" + (f"  ({v['source']})" if v["source"] else ""))
    if fits:
        lines.append("  fits the budget: " + ", ".join(f"{s} up to {v}" for s, v in fits.items()))
    lines.append("  try: a smaller batch, recompute=True for a training program (docs/MEMORY.md), or a larger budget.")
    lines.append("Nothing was launched.")
    raise MemoryBudgetError("\n".join(lines), mp, budget, fits)
