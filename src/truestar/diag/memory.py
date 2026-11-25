"""Name the module instance behind a CUDA out of memory error.

PyTorch's OOM message says how much memory was requested and how much is free, but not which layer
asked, whether it was the forward or the backward pass, or whether memory had been creeping up for
many steps. :class:`MemoryTracker` hooks every module of a model and keeps, per call, the allocated
memory before and after, the peak and the size of what the module produced. When a CUDA OOM passes
through it, it prints and stores an :class:`OOMReport` and lets the original error continue.

    tracker = track_memory(model)
    for batch in loader:
        with tracker:                       # or decorate the step function with @tracker
            loss = model(batch).sum()
            loss.backward()
    print(tracker.memory_timeline().to_text())

On OOM the report is printed to stderr, stored as ``tracker.report`` and attached to the exception as
``err.truestar_report``. Nothing is swallowed: the original exception is re raised unchanged.

Forward attribution uses module forward pre and post hooks. Backward attribution uses tensor
gradient hooks: the gradient of a module's output arriving marks the start of that module's
backward, the gradient of its input marks the end. Tensor hooks are used instead of
``register_full_backward_hook`` because the latter breaks models with in place activations.
"""

from __future__ import annotations

import gc
import inspect
import linecache
import os
import re
import sys
import traceback
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

_TORCH_DIR = os.path.dirname(torch.__file__)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def fmt_bytes(n: Optional[float]) -> str:
    if n is None:
        return "-"
    sign = "-" if n < 0 else ""
    n = abs(float(n))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{sign}{n:.0f} {unit}" if unit == "B" else f"{sign}{n:.2f} {unit}"
        n /= 1024
    return f"{sign}{n:.2f} TiB"


_UNITS = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3, "TiB": 1024 ** 4,
          "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3, "TB": 1024 ** 4}


def _parse_request(msg: str) -> Tuple[Optional[int], str]:
    m = re.search(r"Tried to allocate ([0-9.]+) ?([KMGT]?i?B)", msg)
    if not m:
        return None, ""
    return int(float(m.group(1)) * _UNITS.get(m.group(2), 1)), f"{m.group(1)} {m.group(2)}"


def is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower() and "CUDA" in str(exc)


try:
    from torch._subclasses.fake_tensor import is_fake as _is_fake
except ImportError:  # older torch
    def _is_fake(t: torch.Tensor) -> bool:
        return False


def _owns_device_memory(t: torch.Tensor) -> bool:
    """False for tensors that report a CUDA device but hold no device memory: the fake tensors
    torch.compile and torch.export keep in their graphs (they stay live after a compile, and reading
    their data pointer warns, and in newer torch raises) and meta tensors."""
    return not t.is_meta and not _is_fake(t)


def _flatten(obj: Any) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, torch.Tensor):
            out.append(o)
        elif isinstance(o, (list, tuple)):
            stack.extend(reversed(o))
        elif isinstance(o, dict):
            stack.extend(reversed(list(o.values())))
    return out


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _shape(t: torch.Tensor) -> str:
    return "(" + ", ".join(str(s) for s in t.shape) + ")"


def _source_of(module: nn.Module) -> str:
    cls = type(module)
    try:
        fn = cls.forward
        path = inspect.getsourcefile(fn) or inspect.getfile(fn)
        line = inspect.getsourcelines(fn)[1]
        return f"{path}:{line}"
    except (TypeError, OSError):
        try:
            return inspect.getfile(cls)
        except TypeError:
            return "<unknown>"


def _user_frame(tb) -> Optional[Tuple[str, int, str, str]]:
    """Deepest traceback frame outside torch and truestar.diag: the line that asked for memory."""
    best = None
    for frame, lineno in traceback.walk_tb(tb):
        path = os.path.abspath(frame.f_code.co_filename)
        if path.startswith(_TORCH_DIR) or path.startswith(_THIS_DIR):
            continue
        best = (path, lineno, frame.f_code.co_name, linecache.getline(path, lineno).strip())
    return best


# ----------------------------------------------------------------------------- records


@dataclass
class Event:
    step: int
    phase: str          # "forward" or "backward"
    module: str
    edge: str           # "start" or "end"
    allocated: int
    peak: int
    delta: int = 0      # allocated change across the call (end events)
    out_bytes: int = 0
    out_shapes: str = ""


@dataclass
class StepStats:
    step: int
    start_allocated: int
    peak: int = 0
    end_allocated: int = 0
    failed: bool = False


@dataclass
class MemoryTimeline:
    step: int
    events: List[Event]
    stats: Optional[StepStats]

    def to_text(self, top: int = 10) -> str:
        lines = [f"truestar memory timeline, step {self.step}"]
        if self.stats is not None:
            s = self.stats
            lines.append(f"  start {fmt_bytes(s.start_allocated)}, peak {fmt_bytes(s.peak)}, "
                         f"end {fmt_bytes(s.end_allocated)}" + ("  (failed: out of memory)" if s.failed else ""))
        lines.append(f"  {'phase':<9}{'module':<32}{'allocated':>12}{'change':>12}{'peak':>12}  output")
        for e in self.events:
            if e.edge != "end":
                continue
            lines.append(f"  {e.phase:<9}{e.module[:31]:<32}{fmt_bytes(e.allocated):>12}"
                         f"{fmt_bytes(e.delta):>12}{fmt_bytes(e.peak):>12}  {e.out_shapes}")
        ends = [e for e in self.events if e.edge == "end"]
        if ends:
            biggest = sorted(ends, key=lambda e: e.delta, reverse=True)[:top]
            lines.append("  largest increases: " + ", ".join(
                f"{e.phase} {e.module} {fmt_bytes(e.delta)}" for e in biggest if e.delta > 0))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "stats": None if self.stats is None else vars(self.stats),
                "events": [vars(e) for e in self.events]}


@dataclass
class OOMReport:
    step: int
    phase: str                      # "forward", "backward" or "outside modules"
    module: Optional[str]           # qualified name of the module instance
    module_type: Optional[str]
    module_source: Optional[str]    # file:line of that module's forward
    module_chain: List[str]
    failed_bytes: Optional[int]
    failed_text: str
    user_line: Optional[Tuple[str, int, str, str]]  # (file, line, function, code)
    allocated: int
    reserved: int
    capacity: int
    step_peak: int
    top_modules: List[Tuple[str, int, str]]          # (module, bytes kept by its forward, output shapes)
    top_tensors: List[Tuple[str, str, str, int]]     # (label, shape, dtype, bytes)
    trend: List[StepStats]
    trend_verdict: str
    context: str = ""
    error: str = ""

    def to_text(self) -> str:
        where = self.phase if self.module is None else f"{self.phase} of {self.module}"
        lines = [f"truestar memory: CUDA out of memory at step {self.step}, {where}"
                 + (f" ({self.module_type}, {self.module_source})" if self.module else "")]
        if self.context:
            lines.append(f"  context: {self.context}")
        lines.append(f"  failed allocation: {self.failed_text or 'unknown size'} "
                     f"(device total {fmt_bytes(self.capacity)}, allocated {fmt_bytes(self.allocated)}, "
                     f"reserved {fmt_bytes(self.reserved)}, peak this step {fmt_bytes(self.step_peak)})")
        if self.user_line:
            path, line, fn, code = self.user_line
            lines.append(f"  failing line: {path}:{line} in {fn}: {code}")
        if len(self.module_chain) > 1:
            lines.append("  module chain: " + " > ".join(self.module_chain))
        if self.top_modules:
            lines.append("  top memory holders this step (memory kept after each module's forward):")
            for name, b, shapes in self.top_modules:
                lines.append(f"    {fmt_bytes(b):>12}  {name}  {shapes}")
        if self.top_tensors:
            lines.append("  largest live CUDA tensors:")
            for label, shape, dtype, b in self.top_tensors:
                lines.append(f"    {fmt_bytes(b):>12}  {label} {shape} {dtype}")
        lines.append(f"  trend: {self.trend_verdict}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in vars(self).items() if k != "trend"}
        d["trend"] = [vars(s) for s in self.trend]
        return d


# ----------------------------------------------------------------------------- tracker


class MemoryTracker(ContextDecorator):
    """Tracks GPU memory per module call; reports and re raises CUDA OOM errors.

    Use as a context manager around each training step, or as a decorator on the step function.
    History (per step peaks and per module events) persists across uses of the same tracker, which
    is what makes the trend in the report possible. A new step starts at each call of the root
    module's forward unless :meth:`step` is called by hand.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None, top: int = 5,
                 keep_steps: int = 3, verbose: bool = True, stream=None):
        self.model = model
        if device is None:
            p = next((p for p in model.parameters() if p.is_cuda), None)
            device = p.device if p is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self._cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.top = top
        self.keep_steps = keep_steps
        self.verbose = verbose
        self.stream = stream
        self.report: Optional[OOMReport] = None
        self.steps: List[StepStats] = []
        self._events: Dict[int, List[Event]] = {}
        self._handles: List[Any] = []
        self._names: Dict[int, str] = {}
        self._mods: Dict[str, nn.Module] = {}
        self._manual_steps = False
        self._closed = 0
        self._active = False
        self._reset_step_state()

    # -- memory probes
    def _alloc(self) -> int:
        return torch.cuda.memory_allocated(self.device) if self._cuda else 0

    def _peak(self) -> int:
        return torch.cuda.max_memory_allocated(self.device) if self._cuda else 0

    def _reset_step_state(self) -> None:
        self._fwd_stack: List[Tuple[str, int]] = []
        self._bwd_active: List[str] = []
        self._bwd_start_alloc: Dict[str, int] = {}
        self._bwd_current: Optional[str] = None
        self._bwd_callback_queued = False
        self._bwd_last_done: Optional[str] = None
        self._last: str = ""
        self._retained: Dict[str, int] = {}
        self._out_shapes: Dict[str, str] = {}
        self._producers: Dict[int, str] = {}

    # -- steps
    def step(self) -> None:
        """Start a new step by hand (turns off the automatic step per root forward)."""
        self._manual_steps = True
        self._new_step()

    def _close_step(self) -> None:
        if self.steps and self._closed != self.steps[-1].step:
            s = self.steps[-1]
            for n in list(self._bwd_active):  # backward started but no input gradient marked its end
                self._bwd_end([n])
            s.peak = max(s.peak, self._peak())
            s.end_allocated = self._alloc()
            self._closed = s.step

    def _new_step(self) -> None:
        self._close_step()
        n = (self.steps[-1].step + 1) if self.steps else 1
        if self._cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        self.steps.append(StepStats(step=n, start_allocated=self._alloc()))
        if len(self.steps) > 10000:
            del self.steps[: len(self.steps) - 10000]
        self._events[n] = []
        for old in [k for k in self._events if k <= n - self.keep_steps]:
            del self._events[old]
        self._reset_step_state()

    def _event(self, phase: str, name: str, edge: str, **kw) -> None:
        if not self.steps:
            self._new_step()
        ev = Event(self.steps[-1].step, phase, name, edge, self._alloc(), self._peak(), **kw)
        self._events[self.steps[-1].step].append(ev)

    # -- hooks
    def _pre(self, mod, args):
        if not self._active:
            return
        name = self._names[id(mod)]
        if mod is self.model and not self._fwd_stack and not self._manual_steps:
            self._new_step()
        self._fwd_stack.append((name, self._alloc()))
        self._event("forward", name, "start")
        self._last = f"forward of {name} started"

    def _post(self, mod, args, out):
        if not self._active:
            return
        name = self._names[id(mod)]
        before = None
        while self._fwd_stack:
            n, a = self._fwd_stack.pop()
            if n == name:
                before = a
                break
        now = self._alloc()
        delta = 0 if before is None else now - before
        outs = _flatten(out)
        shapes = " ".join(f"{_shape(t)} {str(t.dtype).replace('torch.', '')}" for t in outs[:3])
        if len(outs) > 3:
            shapes += f" +{len(outs) - 3} more"
        self._retained[name] = self._retained.get(name, 0) + delta
        self._out_shapes[name] = shapes
        for t in outs:
            if t.is_cuda:
                try:
                    self._producers[t.untyped_storage().data_ptr()] = name
                except RuntimeError:
                    pass
        self._event("forward", name, "end", delta=delta, out_bytes=sum(_tensor_bytes(t) for t in outs),
                    out_shapes=shapes)
        self._last = f"forward of {name} finished"
        if torch.is_grad_enabled():
            for t in outs:
                if t.requires_grad:
                    self._attach(t, "_truestar_bwd_start", name, self._bwd_start)
            for t in _flatten(args):
                if t.requires_grad:
                    self._attach(t, "_truestar_bwd_end", name, self._bwd_end)

    def _attach(self, t: torch.Tensor, attr: str, name: str, fn) -> None:
        names = getattr(t, attr, None)
        if names is None:
            names = []
            try:
                t.register_hook(lambda g, names=names: fn(names))
                setattr(t, attr, names)
            except RuntimeError:
                return
        names.append(name)

    def _bwd_start(self, names: List[str]) -> None:
        if not self._active or not names:
            return
        # Hooks for one tensor are listed innermost module first (child forward ends before parent).
        for n in reversed(names):
            self._bwd_active.append(n)
            self._bwd_start_alloc[n] = self._alloc()
            self._event("backward", n, "start")
        self._bwd_current = names[0]
        self._last = f"backward of {names[0]} started"
        if not self._bwd_callback_queued:
            # Runs when this backward pass finishes (the hook DDP uses), so a later OOM in the
            # optimizer or the loss is not blamed on the last module whose backward started.
            try:
                torch.autograd.Variable._execution_engine.queue_callback(self._bwd_done)
                self._bwd_callback_queued = True
            except Exception:
                pass

    def _bwd_done(self) -> None:
        self._bwd_callback_queued = False
        if not self._active:
            return
        last = self._bwd_current
        for n in list(self._bwd_active):
            self._bwd_end([n])
        self._bwd_current = None
        self._last = "backward pass finished" + (f" (last module: {last})" if last else "")

    def _bwd_end(self, names: List[str]) -> None:
        if not self._active:
            return
        for n in names:
            if n in self._bwd_active:
                self._bwd_active.remove(n)
                a0 = self._bwd_start_alloc.pop(n, self._alloc())
                self._event("backward", n, "end", delta=self._alloc() - a0)
                if self._bwd_current == n:
                    self._bwd_current = None
                self._bwd_last_done = n
                self._last = f"backward of {n} finished"

    # -- context manager
    def __enter__(self) -> "MemoryTracker":
        self.report = None
        self._names = {}
        self._mods = {}
        for name, m in self.model.named_modules():
            label = name or f"<root {type(m).__name__}>"
            self._names[id(m)] = label
            self._mods[label] = m
            self._handles.append(m.register_forward_pre_hook(self._pre))
            self._handles.append(m.register_forward_hook(self._post))
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc is not None and is_cuda_oom(exc):
                if self.steps:
                    self.steps[-1].failed = True
                self.report = self._build_report(exc, tb)
                try:
                    exc.truestar_report = self.report
                except Exception:
                    pass
                if self.verbose:
                    print(self.report.to_text(), file=self.stream or sys.stderr, flush=True)
            else:
                self._close_step()
        finally:
            self._active = False
            for h in self._handles:
                h.remove()
            self._handles = []
            self._fwd_stack = []
        return False  # never swallow the error

    # -- report
    def _build_report(self, exc: BaseException, tb) -> OOMReport:
        msg = str(exc)
        failed_bytes, failed_text = _parse_request(msg)
        step = self.steps[-1].step if self.steps else 0
        chain: List[str] = []
        context = ""
        if self._fwd_stack:
            phase = "forward"
            chain = [n for n, _ in self._fwd_stack]
            module = chain[-1]
        elif self._bwd_current is not None or self._bwd_active:
            phase = "backward"
            module = self._bwd_current or self._bwd_active[-1]
            chain = [module]
        else:
            phase = "outside modules"
            module = None
            context = (f"no module call was running; last event: {self._last}" if self._last
                       else "no module call was running")
        mod = self._mods.get(module) if module else None
        cap = torch.cuda.get_device_properties(self.device).total_memory if self._cuda else 0
        # Leaf modules only: a parent's figure is the sum of its children's.
        top_modules = sorted(((n, b, self._out_shapes.get(n, "")) for n, b in self._retained.items()
                              if b > 0 and n in self._mods and not any(True for _ in self._mods[n].children())),
                             key=lambda x: x[1], reverse=True)[: self.top]
        trend = [s for s in self.steps[:-1]][-10:]
        return OOMReport(
            step=step, phase=phase, module=module,
            module_type=type(mod).__name__ if mod is not None else None,
            module_source=_source_of(mod) if mod is not None else None,
            module_chain=chain, failed_bytes=failed_bytes, failed_text=failed_text,
            user_line=_user_frame(tb), allocated=self._alloc(),
            reserved=torch.cuda.memory_reserved(self.device) if self._cuda else 0,
            capacity=cap, step_peak=self._peak(), top_modules=top_modules,
            top_tensors=self._live_tensors(), trend=trend,
            trend_verdict=self._verdict(trend, failed_bytes, step), context=context,
            error=msg.splitlines()[0] if msg else type(exc).__name__,
        )

    def _live_tensors(self) -> List[Tuple[str, str, str, int]]:
        if not self._cuda:
            return []
        params: Dict[int, str] = {}
        for n, p in self.model.named_parameters():
            if p.is_cuda:
                params[p.untyped_storage().data_ptr()] = f"param {n}"
                if p.grad is not None and p.grad.is_cuda:
                    params[p.grad.untyped_storage().data_ptr()] = f"grad of {n}"
        seen: Dict[int, Tuple[str, str, str, int]] = {}
        # Unreachable reference cycles still in gc's lists are not holders: collect them first, so
        # the list names what is really alive (and their device memory is freed).
        gc.collect()
        for o in gc.get_objects():
            try:
                if not issubclass(type(o), torch.Tensor) or not o.is_cuda or not _owns_device_memory(o):
                    continue
                st = o.untyped_storage()
                ptr, nb = st.data_ptr(), st.nbytes()
            except Exception:
                continue
            if ptr in seen or nb == 0:
                continue
            label = params.get(ptr) or (f"output of {self._producers[ptr]}" if ptr in self._producers else "tensor")
            seen[ptr] = (label, _shape(o), str(o.dtype).replace("torch.", ""), nb)
        return sorted(seen.values(), key=lambda x: x[3], reverse=True)[: self.top]

    @staticmethod
    def _verdict(trend: List[StepStats], failed: Optional[int], step: int) -> str:
        ask = f"; step {step} asked for {fmt_bytes(failed)} more" if failed else ""
        if not trend:
            return f"no earlier tracked steps{ask}"
        peaks = [s.peak for s in trend]
        ends = [s.end_allocated for s in trend]
        first, last = trend[0].step, trend[-1].step
        shown = ", ".join(fmt_bytes(p) for p in peaks[-5:])
        growing = len(ends) >= 3 and all(b > a for a, b in zip(ends, ends[1:])) and ends[-1] - ends[0] > 1 << 20
        if growing:
            return (f"memory left allocated after each step grew from {fmt_bytes(ends[0])} to "
                    f"{fmt_bytes(ends[-1])} over steps {first} to {last}: a slow leak (something keeps "
                    f"tensors across steps, for example a loss kept with its graph){ask}")
        steady = max(peaks) <= 1.1 * min(peaks) + (1 << 20)
        if steady:
            spike = failed is not None and failed > max(peaks)
            return (f"steps {first} to {last} peaked at {shown} (steady){ask}"
                    + (": a spike in this step, not a slow leak" if spike else ""))
        return f"steps {first} to {last} peaked at {shown} (varies by step, likely input dependent){ask}"

    # -- summaries
    def memory_timeline(self, step: int = -1) -> MemoryTimeline:
        """Per module memory events of one step (default: the latest)."""
        if not self.steps:
            return MemoryTimeline(0, [], None)
        st = self.steps[step] if step < 0 else next(s for s in self.steps if s.step == step)
        return MemoryTimeline(st.step, list(self._events.get(st.step, [])), st)


def track_memory(model: nn.Module, **kw) -> MemoryTracker:
    """Create a :class:`MemoryTracker`; use it as ``with tracker:`` or ``@tracker``."""
    return MemoryTracker(model, **kw)


def memory_timeline(tracker: MemoryTracker, step: int = -1) -> MemoryTimeline:
    return tracker.memory_timeline(step)
