"""Sentinels: compiled in runtime checks for compiled programs (``CompiledExecutor(..., sentinels=...)``).

Detection is cheap and lives where the work is:

* every Truestar kernel is compiled with the armed check classes (kernel/checks.py) and writes a
  flag word only when a block holds a bad value;
* steps that run through torch (library kernels, the fallback, index and integer steps) get one
  reduction of their float output (``not isfinite`` or ``isnan``), or of an integer divisor;
* float program inputs get the same reduction once per call; params are checked once.

All flags of a call are read with one host read at the end of the call. Only when one fired is the
call run again in *locate* mode: the same steps, a read after every step, so the first step that
produced the bad value (in time, also inside loops) is found. That step's IR ops are then evaluated
in float64 on the inputs it really received, which names the op, its source line and the first
offending element. The result of the call is never returned.

Index bounds are always checked (since the paged KV cache work); with or without checks they now
raise ``SentinelError`` with the op, the line and the first offending index too.
"""

from __future__ import annotations

import math
import re
import weakref
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch

from ..ir.core import IRError, Node
from ..kernel import checks as kchecks
from .lower import kind_of

CODES = {"nan": "TS500", "inf": "TS501", "overflow": "TS502", "div": "TS503", "bounds": "TS504"}
WHAT = {"nan": "NaN", "inf": "an infinity", "overflow": "a value too large for the storage precision",
        "div": "a division by zero", "bounds": "an index out of bounds"}
F32_MAX = torch.finfo(torch.float32).max
PRIORITY = ("div", "overflow", "inf", "nan")   # a cause before what it leads to (0 / 0 is NaN)
_LINE = re.compile(r"([^\s,()]+\.py):(\d+)")


class SentinelError(IRError):
    """A compiled in check fired. The results of the call were not returned.

    Fields: ``check`` (nan, inf, overflow, div, bounds), ``code`` (TS500 to TS504), ``op``,
    ``value`` (the IR value), ``source`` (the IR node's source text), ``file`` and ``line`` (the
    user line, when the source names one), ``index`` (first offending element of ``value``, a
    tuple), ``bad_value`` (the value found there, or the bad index), ``count`` and ``numel`` (bad
    entries of ``value``), ``step`` (kernel or step name), ``fused_ops`` (every op in that kernel),
    ``region``, ``iterations`` (loop iteration per enclosing loop, outermost first)."""

    def __init__(self, message: str, **fields: Any):
        super().__init__(message)
        self.check = fields.get("check", "")
        self.code = CODES.get(self.check, "")
        for k in ("op", "value", "source", "file", "line", "index", "bad_value", "count", "numel", "step",
                  "fused_ops", "region", "iterations", "program", "device", "dtype"):
            setattr(self, k, fields.get(k))

    def to_dict(self) -> Dict[str, Any]:
        bad = self.bad_value
        if isinstance(bad, float) and not math.isfinite(bad):
            bad = str(bad)
        return {"code": self.code, "check": self.check, "op": self.op, "value": self.value,
                "source": self.source, "file": self.file, "line": self.line,
                "index": None if self.index is None else list(self.index), "bad_value": bad,
                "count": self.count, "numel": self.numel, "step": self.step, "fused_ops": self.fused_ops,
                "region": self.region, "iterations": self.iterations, "program": self.program,
                "device": self.device, "dtype": self.dtype, "message": str(self)}


def source_line(source: str) -> Tuple[Optional[str], Optional[int]]:
    """(file, line) of the last ``file.py:N`` in an IR node's source text (the user's line)."""
    hits = _LINE.findall(source or "")
    if not hits:
        return None, None
    f, n = hits[-1]
    return f, int(n)


def _first(mask: torch.Tensor) -> Tuple[Optional[Tuple[int, ...]], int]:
    """(multi index of the first True entry in row major order, number of True entries)."""
    mask = mask.detach()
    count = int(mask.sum())
    if count == 0:
        return None, 0
    flat = int(torch.argmax(mask.reshape(-1).to(torch.int8)))
    if mask.dim() == 0:
        return (), count
    idx = []
    for s in reversed(mask.shape):
        idx.append(flat % s)
        flat //= s
    return tuple(reversed(idx)), count


def bad_mask(t: torch.Tensor, check: str, dtype: torch.dtype, register_max: float = F32_MAX) -> torch.Tensor:
    """Entries of ``t`` that fail ``check``. ``t`` is either a value as the compiled program stored
    it, or its float64 reference: then an infinity means beyond the register range (float32), and
    overflow means beyond the storage dtype's range."""
    t = t.detach()
    if not t.dtype.is_floating_point:
        return torch.zeros(t.shape, dtype=torch.bool, device=t.device)
    if check == "nan":
        return torch.isnan(t)
    if check == "inf":
        return torch.isinf(t) | (t.abs() > register_max) if t.dtype == torch.float64 else torch.isinf(t)
    if check == "overflow":
        big = torch.finfo(dtype).max
        if t.dtype == torch.float64:
            return torch.isfinite(t) & (t.abs() > big)
        return torch.isinf(t)
    raise ValueError(check)


def _describe_node(node: Node) -> str:
    return f"{node.op} (%{node.output.name})" + (f" at {node.source}" if node.source else "")


# ----------------------------------------------------------------------------- the taint pass


def _step_io(step) -> Tuple[List[int], List[int]]:
    ins = [s for s in getattr(step, "in_slots", []) or [] if s is not None]
    if hasattr(step, "src"):
        ins = [step.src]
    outs = list(getattr(step, "outs", []) or [])
    for name in ("out", "dst"):
        s = getattr(step, name, None)
        if isinstance(s, int):
            outs.append(s)
    return ins, outs


def arm_inf(plan, classes: FrozenSet[str], opts, taint: Optional[Dict[int, FrozenSet[str]]] = None) -> None:
    """Disarm the inf class on kernels whose output can hold an infinity on purpose (an infinite
    constant such as a -inf attention mask that reaches the stored value, or a value derived from
    one). ``kernel.checks.may_be_inf`` decides from the kernel IR; ``exp(-inf)`` is 0, so a
    masked softmax stays armed. Such kernels are recompiled without the inf class and listed in the
    report; nan and div stay armed on them."""
    from .. import kernel as K
    taint = dict(taint or {})
    for i, t in enumerate(plan.template):
        if isinstance(t, torch.Tensor) and t.dtype.is_floating_point and t.numel():
            signs = set()
            if bool((t == math.inf).any()):
                signs.add(kchecks.POS)
            if bool((t == -math.inf).any()):
                signs.add(kchecks.NEG)
            if signs:
                taint[i] = frozenset(signs)
    for step in plan.steps:
        kind = step.kind
        if kind == "kernel":
            names = [b.name for b in step.spec.kernel.inputs]
            buf = {n: taint.get(s, frozenset()) for n, s in zip(names, step.in_slots)}
            memo: dict = {}
            signs = frozenset()
            for o in step.spec.kernel.outputs:
                signs = signs | kchecks.may_be_inf(o.expr, buf, memo)
            if signs:
                taint[step.out] = signs
                if "inf" in step.armed:
                    step.armed = step.armed - {"inf"}
                    step.disarmed = ("inf: the stored value can hold an infinity on purpose "
                                     "(an infinite constant or a value derived from one)")
                    step.ck = K.compile(step.spec.kernel, opts.device, allow_tf32=opts.allow_tf32,
                                        checks=step.armed, portable_tensor_cores=opts.portable_tensor_cores,
                                        **opts.hidden)
            continue
        if kind == "create":
            v = step.attrs.get("value") if step.op != "arange" else None
            if isinstance(v, float) and math.isinf(v):
                taint[step.out] = frozenset((kchecks.POS if v > 0 else kchecks.NEG,))
            continue
        if kind in ("cond", "loop"):
            if kind == "cond":
                pairs = [(step.then, step.args), (step.other, step.args)]
                outer = list(step.args)
            else:
                outer = list(step.carried) + list(step.captures)
                pairs = [(step.body, [None] + outer)]
                if step.cond is not None:
                    pairs.append((step.cond, outer))
            union = frozenset()
            for sub, args in pairs:
                inner = {s: taint.get(a, frozenset()) for s, a in zip(sub.in_slots, args) if a is not None}
                arm_inf(sub, classes, opts, inner)
                union = union | frozenset().union(*[inner.get(s, frozenset()) for s in sub.in_slots] or [frozenset()])
            if union:
                for s in step.outs:
                    taint[s] = union
            continue
        ins, outs = _step_io(step)
        signs = frozenset().union(*[taint.get(s, frozenset()) for s in ins] or [frozenset()])
        if signs:
            for s in outs:
                taint[s] = signs
        step.inf_tainted = bool(signs)


# ----------------------------------------------------------------------------- per program state


class ProgramChecks:
    """Flags, triggers and the locate run of one compiled program."""

    TORCH_KINDS = ("library", "fallback", "host")

    @property
    def cp(self):
        cp = self._cp()
        if cp is None:
            raise ReferenceError("the compiled program of these sentinels no longer exists")
        return cp

    def __init__(self, cp, classes: FrozenSet[str]):
        # a weak reference: the program owns its sentinels, so a strong one back would make a cycle
        # that keeps the program (its device params and buffers) alive until the garbage collector
        # runs, which other code that scans live tensors then sees
        self._cp = weakref.ref(cp)
        self.classes = classes
        self.sites: List[Any] = []              # kernel steps with armed classes, in build order
        for plan, step in cp.plan.all_steps():
            if step.kind == "kernel":
                if step.armed:
                    step.site = len(self.sites)
                    self.sites.append((plan, step))
            elif step.kind in self.TORCH_KINDS:
                step.trig = step.chk_kind = self._torch_trigger(step)
                if step.trig:
                    plan.has_trig = True
        if cp.opts.device == "cuda":
            for plan in {id(p): p for p, _ in cp.plan.all_steps()}.values():
                self._checks_at_loads(plan)
        self.flags: Optional[torch.Tensor] = None
        self.locating = False
        self.loops: Dict[Any, int] = {}
        self.order: List[Any] = []
        self.trig: List[torch.Tensor] = []
        self.watch_nan: List[torch.Tensor] = []
        self.watch_fin: List[torch.Tensor] = []
        self.ring: List[list] = []              # [flag buffer, per site views, event of its call]
        self.next = 0
        self.slot = None
        self.kernel_fired = False
        self.note = ""
        self.pending: List[tuple] = []          # deferred reads: (event, host copy, kernel words, args, call)
        self.calls = 0
        self.params_checked = False

    def _checks_at_loads(self, plan) -> None:
        """A torch step whose float output is read only by armed Truestar kernels (through views at
        most) is not reduced at the end of the call: the kernels check that buffer as they load it
        (a predicated store), which costs no extra pass over the tensor. Outputs read by anything
        else (a program output, control flow, another torch step) keep the end of call reduction."""
        from .. import kernel as K
        readers: Dict[int, List[Tuple[Any, int]]] = {}
        alias: Dict[int, int] = {}
        for step in plan.steps:
            if step.kind == "layout" and step.in_slots:
                alias[step.out] = alias.get(step.in_slots[0], step.in_slots[0])
        root = lambda sl: alias.get(sl, sl)  # noqa: E731
        for step in plan.steps:
            if step.kind == "layout":
                continue
            ins, _ = _step_io(step)
            if step.kind in ("cond", "loop"):
                ins = list(getattr(step, "args", []) or []) + list(getattr(step, "carried", []) or [])                     + list(getattr(step, "captures", []) or []) + ([step.pred] if hasattr(step, "pred") else [])
            for k, sl in enumerate(ins):
                readers.setdefault(root(sl), []).append((step, k))
        outs = {root(sl) for sl in plan.out_slots}
        changed: Dict[int, Any] = {}
        for step in plan.steps:
            trig = getattr(step, "trig", None)
            if step.kind not in self.TORCH_KINDS or not trig or trig[0] != "float":
                continue
            _, slots = _step_io(step)
            cls = ["nan"] if "nan" in self.classes else []
            if trig[1]:
                cls.append("inf")
            ok = bool(slots) and bool(cls)
            for sl in slots:
                rs = readers.get(sl, [])
                if sl in outs or not rs or any(r.kind != "kernel" or not r.armed for r, _ in rs):
                    ok = False
            if not ok:
                continue
            step.trig = None
            for sl in slots:
                for r, k in readers[sl]:
                    name = r.spec.kernel.inputs[k].name
                    loads = changed.setdefault(id(r), (r, {}))[1]
                    loads[name] = tuple(sorted(set(loads.get(name, ())) | set(cls)))
        opts = self.cp.opts
        for r, loads in changed.values():
            r.check_loads = loads
            r.ck = K.compile(r.spec.kernel, opts.device, allow_tf32=opts.allow_tf32, checks=r.armed,
                             check_loads=loads, portable_tensor_cores=opts.portable_tensor_cores,
                             **opts.hidden)
        plan.has_trig = any(getattr(st, "trig", None) for st in plan.steps)

    def _torch_trigger(self, step):
        """What one torch run step is checked for: ("float", inf armed) for float outputs, ("div", slot)
        for an integer remainder or division, None when nothing applies."""
        node = step.nodes[0] if step.nodes else None
        if node is None:
            return None
        if step.kind == "library" and node.op in ("remainder", "fmod", "div", "floor_divide") \
                and "div" in self.classes and len(node.inputs) > 1 and node.inputs[1] is not None \
                and kind_of(node.inputs[1].type.dtype) == "i":
            return ("div", None)
        floats = [o for o in node.outputs if kind_of(o.type.dtype) == "f"]
        if not floats or not ({"nan", "inf"} & self.classes):
            return None
        return ("float", "inf" in self.classes and not getattr(step, "inf_tainted", False))

    # ---- per call
    def _buffer(self):
        """A flag buffer for this call. On CUDA the buffers are pinned host memory that the kernels
        write through the unified address space: zeroing and reading them needs no GPU launch and no
        copy, only the call's event. A ring of buffers lets deferred reads overlap later calls."""
        n = len(self.sites) * kchecks.WORDS + 1     # the last word: any torch side trigger
        cuda = self.cp.opts.device == "cuda"
        if not self.ring:
            for _ in range(self.MAX_PENDING + 2):
                f = torch.zeros(n, dtype=torch.int32, pin_memory=cuda)
                views = [f[i * kchecks.WORDS:(i + 1) * kchecks.WORDS] for i in range(len(self.sites))]
                self.ring.append([f, views, None])
        slot = self.ring[self.next % len(self.ring)]
        self.next += 1
        if slot[2] is not None:            # still owned by a pending read: wait for its call
            slot[2].synchronize()
            slot[2] = None
        return slot

    def begin(self, rt, args: Sequence[torch.Tensor]) -> None:
        slot = self._buffer()
        self.slot = slot
        self.flags = slot[0]
        self.flags.zero_()                  # host memory: a memset, no launch
        for (_, step), v in zip(self.sites, slot[1]):
            step.chk_view = v
        self.trig = []
        self.watch_nan, self.watch_fin = [], []
        if not self.params_checked:
            self._check_params()
        # Inputs are not reduced per call: a NaN or Inf input reaches the first kernel or torch step
        # that reads it, whose check fires, and the locate run names the input as the origin.

    def _float_bad(self, t: torch.Tensor, inf: bool) -> None:
        """Watch a float tensor; no launch now. At the end of the call one multi tensor 2 norm
        (``torch._foreach_norm``, a few launches for every watched tensor) decides: the norm is NaN
        exactly when an entry is NaN, and not finite when an entry is NaN or infinite (or, rarely,
        when finite entries overflow the float32 norm: the locate run then finds nothing and the call
        returns normally).

        With the memory planner reusing buffers (the default), the norm is taken at once, so the watch
        never keeps a value alive past its last reader and the device memory plan holds as planned.
        With ``memory_reuse=False`` every value lives to the end of the call anyway, and up to
        FLUSH tensors are reduced together."""
        if self.cp.opts.memory_reuse:
            n = torch.linalg.vector_norm(t, dtype=torch.float32)
            self.trig.append(torch.isnan(n) if not inf else ~torch.isfinite(n))
            return
        (self.watch_fin if inf else self.watch_nan).append(t)
        if len(self.watch_fin) + len(self.watch_nan) >= self.FLUSH:
            self._flush()

    FLUSH = 128     # watched tensors kept alive at most (long loops): then reduced early

    def _flush(self) -> None:
        for lst, bad in ((self.watch_nan, torch.isnan), (self.watch_fin, lambda n: ~torch.isfinite(n))):
            if lst:
                norms = torch._foreach_norm(lst, 2, dtype=torch.float32)
                self.trig.append(bad(torch.stack(list(norms))).any())
                lst.clear()

    def after(self, step, env) -> None:
        """A torch run step just ran (non locate mode): queue its trigger, no host read."""
        kind, extra = step.trig
        node = step.nodes[0]
        if kind == "div":
            d = env[step.in_slots[1]]
            self.trig.append((d == 0).any())
            return
        slots = getattr(step, "outs", None) or [step.out]
        for o, s in zip(node.outputs, slots):
            if kind_of(o.type.dtype) == "f":
                t = env[s]
                if isinstance(t, torch.Tensor) and t.dtype.is_floating_point and t.numel():
                    self._float_bad(t, extra)

    def _fold_triggers(self) -> None:
        """Write the torch side triggers into the buffer's last word (one small copy, only when torch
        run steps were watched)."""
        self._flush()
        if self.trig:
            t = self.trig[0] if len(self.trig) == 1 else torch.stack([x.reshape(()) for x in self.trig]).any()
            self.flags[-1:].copy_(t.reshape(1), non_blocking=True)

    def _event(self):
        if self.cp.opts.device != "cuda":
            return None
        ev = torch.cuda.Event()
        ev.record()
        return ev

    def fired(self) -> bool:
        """Wait for the call's GPU work, then read every flag from host memory."""
        self._fold_triggers()
        ev = self._event()
        if ev is not None:
            ev.synchronize()
        hits = self.flags.tolist()
        self.kernel_fired = any(hits[:-1])
        return any(hits)

    # ---- deferred reads (sentinel_read="deferred")
    MAX_PENDING = 4

    def defer(self, args, call: int) -> None:
        """Queue this call's flags for a later read: only an event, no host sync and no copy (the
        kernels write the pinned buffer directly). Read at the start of a later call or by read_sentinels()."""
        self._fold_triggers()
        ev = self._event()
        self.slot[2] = ev
        self.pending.append((ev, self.slot, self.flags.numel() - 1, list(args), call))

    def poll(self, block: bool = False):
        """The oldest pending call whose flags fired, or None. Done calls that are clean are dropped;
        with block (or too many pending) the reads wait for the GPU."""
        while self.pending:
            ev, host, nk, args, call = self.pending[0]
            if ev is not None and not (block or len(self.pending) > self.MAX_PENDING) and not ev.query():
                return None
            if ev is not None:
                ev.synchronize()
            self.pending.pop(0)
            vals = host[0].tolist()
            if host[2] is ev:
                host[2] = None
            if any(vals):
                self.kernel_fired = any(vals[:nk])
                return args, call
        return None

    def _check_params(self) -> None:
        self.params_checked = True
        prog = self.cp.program
        for p in prog.params:
            data = prog.param_data[p.name]
            if not data.dtype.is_floating_point:
                continue
            for check in ("nan", "inf"):
                if check in self.classes:
                    mask = torch.isnan(data) if check == "nan" else torch.isinf(data)
                    idx, count = _first(mask)
                    if idx is not None and check == "nan":     # an infinite param is a mask on purpose
                        raise self._error(check, op="param", value=p.name, source=f"parameter {p.name}",
                                          index=idx, bad=float(data[idx]) if idx else float(data),
                                          count=count, numel=data.numel(), step="params", fused=[],
                                          region=self.cp.program.name, why=f"parameter @{p.name} holds {WHAT[check]}")


    # ---- locate
    def locate(self, plan, args, rt):
        """Run the call again with a read after every step and raise for the first step that
        produced a bad value. Returns only if the flag did not fire again."""
        if self.cp.opts.device == "cuda":
            torch.cuda.synchronize()
        self.flags.zero_()
        self.trig = []
        self.watch_nan, self.watch_fin = [], []
        self.locating = True
        self.loops = {}
        self.order = []
        try:
            if {"nan", "inf"} & self.classes:
                for v, t, kind in zip(self.cp.program.inputs, args, self.cp._kinds):
                    if kind != "f":
                        continue
                    for check in ("nan", "inf"):
                        if check in self.classes:
                            idx, count = _first(bad_mask(t, check, t.dtype))
                            if idx is not None:
                                raise self._error(check, op="input", value=v.name, source=f"program input %{v.name}",
                                                  index=idx, bad=float(t[idx]), count=count, numel=t.numel(),
                                                  step="inputs", fused=[], region=self.cp.program.name,
                                                  why=f"program input %{v.name} holds {WHAT[check]}")
            plan.run(args, rt)
        finally:
            self.locating = False
        return None

    def inspect(self, plan, step, env, rt) -> None:
        """Locate mode: called after every step."""
        if step.kind == "kernel" and getattr(step, "site", None) is not None:
            if self.cp.opts.device == "cuda":
                torch.cuda.synchronize()
            words = step.chk_view.tolist()
            for check in PRIORITY:
                if words[kchecks.SLOT[check]]:
                    raise self._attribute_kernel(plan, step, env, rt, check)
        elif step.kind in self.TORCH_KINDS and getattr(step, "chk_kind", None):
            kind, inf = step.chk_kind      # also when its check moved to the kernels that load it
            node = step.nodes[0]
            if kind == "div":
                d = env[step.in_slots[1]]
                idx, count = _first(d == 0)
                if idx is not None:
                    raise self._error("div", op=node.op, value=node.inputs[1].name, source=node.source,
                                      index=idx, bad=0, count=count, numel=d.numel(), step=step.name,
                                      fused=[node.op], region=plan.path,
                                      why=f"integer {node.op} divides by zero")
                return
            slots = getattr(step, "outs", None) or [step.out]
            for o, s in zip(node.outputs, slots):
                t = env[s]
                if not (isinstance(t, torch.Tensor) and t.dtype.is_floating_point):
                    continue
                for check in ("nan", "inf"):
                    if check not in self.classes or (check == "inf" and not inf):
                        continue
                    idx, count = _first(bad_mask(t, check, t.dtype))
                    if idx is not None:
                        raise self._error(check, op=node.op, value=o.name, source=node.source, index=idx,
                                          bad=float(t[idx]), count=count, numel=t.numel(), step=step.name,
                                          fused=[node.op], region=plan.path,
                                          why=f"{node.op} (%{o.name}) produced {WHAT[check]} ({step.kind} step)")

    def _attribute_kernel(self, plan, step, env, rt, check: str) -> SentinelError:
        """Name the op of a fused kernel that produced the bad value: its ops in float64 on the
        inputs the kernel received, first op (program order) whose own result is bad."""
        out = env[step.out]
        dtype = self.cp.opts.torch_dtype
        nodes = list(step.nodes)
        fused = [n.op for n in nodes]
        local = None
        if step.check is not None:
            try:
                local = step.check.reference(env, rt)
            except Exception:  # noqa: BLE001 (attribution is best effort; the error is still raised)
                local = None
        blamed, idx, count, numel, bad, note = None, None, 0, out.numel(), None, ""
        if local is not None:
            order = ([check] if check != "nan" else ["nan", "inf"])
            for c in order:
                for node in nodes:
                    if check == "div":
                        hit = self._div_hit(node, local)
                        if hit is not None:
                            blamed, (idx, count, numel, bad) = node, hit
                            break
                        continue
                    for o in node.outputs:
                        ref = local.get(id(o))
                        if ref is None or not ref.dtype.is_floating_point:
                            continue
                        mask = bad_mask(ref, c, dtype)
                        i, n = _first(mask)
                        if i is not None:
                            blamed, idx, count, numel, bad = node, i, n, ref.numel(), float(ref[i])
                            if c != check:
                                note = (f" Its float64 value exceeds the float32 range here ({bad:.3g}), "
                                        f"so it is Inf in the kernel and gives NaN later in the same kernel.")
                            break
                    if blamed is not None:
                        break
                if blamed is not None or check == "div":
                    break
        if blamed is None:     # nothing in float64: the precision of the kernel itself (or no reference)
            blamed = nodes[-1] if nodes else None
            mask = bad_mask(out, "nan" if check == "nan" else "inf", dtype) if check != "div" else None
            if mask is not None:
                idx, count = _first(mask)
                numel = out.numel()
                bad = float(out[idx]) if idx is not None else None
            if check in ("overflow", "inf") and blamed is not None and local is not None:
                ref = local.get(id(blamed.output))
                if ref is not None and idx is not None and ref.shape == out.shape:
                    bad = float(ref[idx])
                    note = f" The float64 result there is {bad:.6g}, beyond the {self.cp.opts.dtype} range."
        value = blamed.output.name if blamed is not None else ""
        op = blamed.op if blamed is not None else ""
        what = f"{blamed.op} (%{blamed.output.name})" if blamed is not None else f"kernel {step.name}"
        why = (f"{what} divides by zero." if check == "div" else f"{what} produced {WHAT[check]}.") + note
        return self._error(check, op=op, value=value, source=blamed.source if blamed is not None else "",
                           index=idx, bad=bad, count=count, numel=numel, step=step.name, fused=fused,
                           region=plan.path, why=why)

    @staticmethod
    def _div_hit(node: Node, local):
        if node.op == "div" and len(node.inputs) > 1:
            den = local.get(id(node.inputs[1]))
        elif node.op == "reciprocal":
            den = local.get(id(node.inputs[0]))
        else:
            return None
        if den is None:
            return None
        out = local.get(id(node.output))
        mask = (den == 0)
        if out is not None and mask.shape != out.shape:
            mask = mask.expand(out.shape) if mask.dim() <= out.dim() else mask
        idx, count = _first(mask)
        if idx is None:
            return None
        bad = float(out[idx]) if out is not None and out.shape == mask.shape else 0.0
        return idx, count, mask.numel(), bad

    def _error(self, check, *, op, value, source, index, bad, count, numel, step, fused, region, why) -> SentinelError:
        f, line = source_line(source or "")
        iters = [k for k in self.loops.values()] if self.loops else []
        cp = self.cp
        where = f"{f}:{line}" if f else (source or "unknown source")
        idx_text = "" if index is None else f"index {tuple(index)} of %{value}"
        bad_text = "" if bad is None else f" (value {bad:.6g})" if isinstance(bad, float) else f" (value {bad})"
        lines = [f"truestar {CODES[check]} {check} sentinel: {why}",
                 f"  at: {source or where}",
                 f"  first offending element: {idx_text}{bad_text}; {count} of {numel} entries"
                 if index is not None else "  first offending element: unknown",
                 f"  step: {step}" + (f", fused ops: {' + '.join(fused)}" if fused else "")
                 + f"; region {region}" + (f"; loop iterations {iters}" if iters else ""),
                 f"  program {cp.program.name!r} on {cp.opts.device} ({cp.opts.dtype}). "
                 + (f"The results of this call were not returned." if not self.note else
                    f"Found by a deferred read{self.note}; that call's results were already returned.")]
        return SentinelError("\n".join(lines), check=check, op=op, value=value, source=source, file=f, line=line,
                          index=None if index is None else tuple(index), bad_value=bad, count=count, numel=numel,
                          step=step, fused_ops=list(fused), region=region, iterations=iters,
                          program=cp.program.name, device=cp.opts.device, dtype=cp.opts.dtype)


def bounds_error(op: str, node: Optional[Node], index: torch.Tensor, size: int, unique: bool) -> SentinelError:
    """The always on index check fired: name the op, its line and the first bad index."""
    idx = index.detach().long()
    bad = (idx < -size) | (idx >= size)
    if unique and idx.numel() > 1:
        w = torch.where(idx < 0, idx + size, idx).reshape(-1)
        first = {}
        rep = torch.zeros(w.shape, dtype=torch.bool, device=w.device)
        for k, v in enumerate(w.tolist()):
            if v in first:
                rep[k] = True
            first.setdefault(v, k)
        bad = bad | rep.reshape(idx.shape)
    pos, count = _first(bad)
    val = int(idx[pos]) if pos is not None else None
    source = node.source if node is not None else ""
    f, line = source_line(source)
    what = "out of range or repeated" if unique else "out of range"
    msg = (f"{op}: an index is {what} for a dim of size {size} (the results of this call were not returned)\n"
           f"truestar TS504 bounds sentinel: index {val} at position {pos} of the index tensor"
           f" ({count} of {idx.numel()} bad)" + (f"\n  at: {f}:{line}  [{source}]" if f else
                                                 (f"\n  at: {source}" if source else "")))
    return SentinelError(msg, check="bounds", op=op, value=node.output.name if node is not None else "",
                      source=source, file=f, line=line, index=pos, bad_value=val, count=count,
                      numel=idx.numel(), step=f"%{node.output.name}" if node is not None else op,
                      fused_ops=[op], region="", iterations=[])
