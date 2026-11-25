"""Compiled programs: the steps a lowered program runs, and the checks built into them.

A compiled program is a flat list of steps over a slot array:

* kernel: one fused Truestar kernel (Triton on the GPU, numba on the CPU);
* layout: views and copies (reshape, permute, expand, select, slice, cat), no arithmetic;
* create: constant tensors (full, arange);
* library: convolution and pooling through torch, a declared library kernel;
* random: the keyed generator's own kernel (Triton on the GPU, numba on the CPU, docs/RANDOM.md),
  checked bit for bit against the reference interpreter on the first run;
* host: integer bookkeeping on the CPU (a dim as a scalar, a loop counter);
* fallback: the float64 reference implementation of an op that has no kernel lowering yet;
* cond, loop: control flow whose regions are compiled programs of the same kind.

Verification. On the first run of every program and of every region, each kernel group and each
library step is compared with the float64 reference implementation of the IR ops it covers, on
the inputs that step really received. The first run of the whole program is also compared end to
end with the reference interpreter. A mismatch raises ``ProgramMismatchError`` naming the region
and the op; the wrong result is never returned.

Where the float64 reference runs (since 2026-11-24): for a CUDA program, on the same GPU when the
check is big enough (``check_device``), else on the CPU as before. Same implementations, same
tolerances, same schedule; only float64 rounding differs (docs/COMPILER.md, Verification).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .. import kernel as K
from ..ir import reference as iref
from ..ir.core import IRError, Node, Program, Value, symbol
from ..kernel import reference as kref
from ..kernel import verify as kverify
from ..kernel.verify import KernelMismatchError
from .lower import DimTable, KernelSpec, Lowering, NotLowerable, Options, kind_of

STATS = {"programs": 0, "kernels": 0, "regions": 0}
STALL_CHECK_FROM = 256     # while loops: first iteration at which a stalled loop is looked for
STALL_CHECK_EVERY = 256


class IllConditionedWarning(UserWarning):
    """TS782: an output over the end to end float64 bar was accepted because the caller asked for it
    (accept_ill_conditioned=True) and an independent float32 evaluation of the program is as far off
    float64 (within 4x). Each accepted output is named with its numbers and listed in
    ``CompiledProgram.conditioning``."""

    code = "TS782"


class ProgramMismatchError(AssertionError):
    """A compiled program disagrees with the float64 reference interpreter. Names the region and op."""

    def __init__(self, message: str, *, program: str = "", region: str = "", op: str = "", value: str = "",
                 ops: Sequence[str] = (), error: float = math.nan, tolerance: float = math.nan, stage: str = "",
                 near_ties: Sequence[Dict[str, Any]] = ()):
        super().__init__(message)
        # end to end only: when non empty, the mismatch is fully explained by these comparisons, decided
        # the other way than float64 on values within rounding of a tie (CompiledProgram._branch_flips)
        self.near_ties = list(near_ties)
        self.ill_conditioned = bool(self.near_ties)
        self.program = program
        self.region = region
        self.op = op
        self.value = value
        self.ops = list(ops)
        self.error = error
        self.tolerance = tolerance
        self.stage = stage


class FallbackNotAllowed(IRError):
    """The program needs the fallback path and the executor was created with allow_fallback=False."""


class RecomputeMismatchError(ProgramMismatchError):
    """TS421: a recomputed value is not bitwise equal to the value it replaces. Names the value,
    the ops and the source line. The result is not returned."""

    code = "TS421"


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> Tuple[bool, int]:
    """(equal, number of differing entries), comparing bits (NaN payloads and signed zeros too)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, -1
    if a.dtype.is_floating_point:
        view = {8: torch.int64, 4: torch.int32, 2: torch.int16, 1: torch.int8}[a.element_size()]
        a, b = a.contiguous().view(view), b.contiguous().view(view)
    ne = a != b
    n = int(ne.sum())
    return n == 0, n


# ----------------------------------------------------------------------------- runtime context


class Runtime:
    """Per call facts shared by every step: evaluated dims, symbol bindings, the verify mode."""

    __slots__ = ("dims", "values", "bindings", "mode", "opts", "stats", "cache", "pending", "checked", "chk",
                 "node", "timer", "update", "decisions")

    def __init__(self, table: DimTable, values: Tuple[int, ...], mode: str, opts: Options):
        self.values = values
        self.dims = table.evaluate(values)
        self.bindings = {symbol(s): v for s, v in zip(table.symbols, values)}
        self.mode = mode
        self.opts = opts
        self.stats = {"checks": 0, "max_error": 0.0, "seconds": 0.0, "gpu_checks": 0}
        self.cache: Dict[Any, torch.Tensor] = {}   # constants that depend only on the symbol values
        self.pending: List[Tuple[torch.Tensor, str]] = []   # index checks read once, after the run
        self.checked: Dict[Any, torch.Tensor] = {}          # index tensors already checked in this run
        self.chk = None      # compiler.checks.ProgramChecks when compiled in checks are on
        self.node = None     # the IR node of the index step running (for bounds messages)
        self.timer = None    # compiler.profile timer when profiling
        self.update = frozenset()   # env slots of inputs the caller lets be written in place (update=)
        # during a verified run: id of every top level float comparison's value -> the decision this
        # run took (recorded by the region checks, from the values the step really received)
        self.decisions: Optional[Dict[int, Optional[torch.Tensor]]] = None

    def raise_pending(self) -> None:
        """One host sync for every index check of the run (IndexStep): raise if any failed."""
        pending, self.pending = self.pending, []
        self.checked = {}
        if not pending:
            return
        flags = torch.stack([f[0].reshape(()) for f in pending])
        if flags.is_cuda and torch.cuda.is_current_stream_capturing():
            # inside a CUDA graph capture (CompiledTrainStep(graph=True)): no host read is allowed.
            # The flags are copied to pinned host memory inside the graph and read after each replay
            # (CAPTURED_PENDING, train/compiled.py), so a bad index in a replay still raises.
            host = torch.zeros(len(pending), dtype=torch.int32, pin_memory=True)
            host.copy_(flags.to(torch.int32), non_blocking=True)
            CAPTURED_PENDING.append((host, [p[1:] for p in pending]))
            return
        if bool(flags.any()):
            from .checks import bounds_error
            hit = next(f for f, h in zip(pending, flags.tolist()) if h)
            _, msg, op, node, index, size, unique = hit
            raise bounds_error(op, node, index, size, unique)


# (pinned host flags, pending entries) recorded by Runtime.raise_pending during a CUDA graph capture
CAPTURED_PENDING: List[Tuple[torch.Tensor, list]] = []


def _ref_form(t: torch.Tensor, kind: str, device: Optional[torch.device] = None) -> torch.Tensor:
    """The tensor a step read, in the form the reference implementation takes (float64, on the CPU
    unless ``device`` says where the reference runs)."""
    t = t.detach()
    if device is None:
        if kind == "f":
            return kverify.to_host_f64(t)
        if kind == "b":
            return (t != 0).cpu() if t.dtype != torch.bool else t.cpu()
        return t.cpu()
    if kind == "f":
        return t.to(device, torch.float64)
    if kind == "b":
        return (t != 0).to(device) if t.dtype != torch.bool else t.to(device)
    return t.to(device)


GPU_BUDGET_MB = 2048     # largest estimated float64 working set of one check run on the GPU


def gpu_budget_bytes() -> int:
    try:
        return int(float(os.environ.get("TRUESTAR_VERIFY_GPU_MB", GPU_BUDGET_MB)) * 2 ** 20)
    except ValueError:
        return GPU_BUDGET_MB * 2 ** 20


def check_device(opts: Options, work: int, peak_elements: int, like: torch.Tensor) -> Optional[torch.device]:
    """Where one region check's float64 reference runs: ``like``'s CUDA device, or None for the CPU.

    The rule of the kernel layer (TRUESTAR_VERIFY_DEVICE): ``cpu`` always the CPU; ``gpu`` the GPU for
    every check of a CUDA program; ``auto`` (default) the GPU when the check's work reaches
    ``kernel.reference.GPU_MIN_WORK`` elements, because below that the CPU takes a few milliseconds and
    the GPU's first float64 use in a process costs 0.1 to 0.2 s. In every mode a check whose estimated
    float64 working set exceeds TRUESTAR_VERIFY_GPU_MB (default 2048) stays on the CPU, so the check
    never competes with the program for GPU memory."""
    if opts.device != "cuda" or like.device.type != "cuda":
        return None
    mode = kverify.verify_device()
    if mode == "cpu" or (mode == "auto" and work < kref.GPU_MIN_WORK):
        return None
    if 8 * peak_elements > gpu_budget_bytes():
        return None
    return like.device


def saturate_scalar(value: float, dtype: torch.dtype) -> float:
    """A finite float64 number beyond the working dtype's range becomes that dtype's largest finite
    number of the same sign, not an infinity. Hugging Face masks are ``torch.finfo(float64).min``
    in a float64 import and are multiplied by 0 (Mistral's sliding window): an infinity would give
    NaN where the float32 model, which uses ``torch.finfo(float32).min``, gives 0. NaN and infinities
    are kept."""
    big = torch.finfo(dtype).max
    if math.isfinite(value) and abs(value) > big:
        return math.copysign(big, value)
    return value


def saturate(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``saturate_scalar`` for every entry of a float tensor (only when one is out of range)."""
    if t.dtype.itemsize <= torch.empty((), dtype=dtype).itemsize or t.numel() == 0:
        return t
    big = torch.finfo(dtype).max
    finite = torch.isfinite(t)
    if not bool((finite & (t.abs() > big)).any()):
        return t
    return torch.where(finite, t.clamp(-big, big), t)


def _to_working(t: torch.Tensor, opts: Options) -> torch.Tensor:
    """A reference result (or a user input) in the form compiled steps pass around."""
    if t.dtype.is_floating_point:
        return kverify.to_device(saturate(t, opts.torch_dtype), opts.torch_device, opts.torch_dtype)
    if t.dtype == torch.bool or t.dim() > 0:
        return t.to(opts.torch_device)
    return t.cpu()   # integer scalars stay on the host


# ----------------------------------------------------------------------------- steps


class Step:
    kind = "step"
    check = None

    def __init__(self, name: str, nodes: Sequence[Node] = ()):
        self.name = name
        self.nodes = list(nodes)

    def ops(self) -> List[str]:
        return [n.op for n in self.nodes]

    def run(self, env: List[Any], rt: Runtime) -> None:
        raise NotImplementedError

    def entry(self) -> Dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "ops": self.ops(),
                "values": [o.name for n in self.nodes for o in n.outputs]}


class KernelStep(Step):
    kind = "kernel"

    def __init__(self, name, nodes, ck, in_slots, out, true_slot, final):
        super().__init__(name, nodes)
        self.ck = ck
        self.in_slots = in_slots
        self.out = out
        self.true_slot = true_slot
        self.final = final
        self.scalar = False
        self.scalar_ins: List[int] = []
        self.spec = None
        self.out_dtype = None   # float32 for a mixed precision island, else the working dtype
        self.armed = frozenset()     # compiled in check classes of this kernel
        self.disarmed = ""           # why a requested class is not armed here
        self.site = None
        self.chk_view = None         # this kernel's flag words in the program's flag buffer

    def run(self, env, rt):
        if self.scalar_ins:
            args = [env[s] for s in self.in_slots]
            for i in self.scalar_ins:
                args[i] = args[i].reshape(1)
        else:
            args = [env[s] for s in self.in_slots]
        out = self.ck(*args) if self.chk_view is None else self.ck(*args, _chk=self.chk_view)
        if self.scalar:
            out = out.reshape(())
        env[self.out] = out
        if self.true_slot is not None:
            env[self.true_slot] = out != 0

    def run_into(self, env, rt, j: int):
        """Write the result into the buffer of input ``j`` (the memory planner proved that safe)."""
        args = [env[s] for s in self.in_slots]
        for i in self.scalar_ins:
            args[i] = args[i].reshape(1)
        if self.chk_view is None:
            out = self.ck.call_into(args[j], *args)
        else:
            out = self.ck.call_into(args[j], *args, _chk=self.chk_view)
        env[self.out] = out
        if self.true_slot is not None:
            env[self.true_slot] = out != 0

    def entry(self):
        e = super().entry()
        e["shape"] = self.ck.plan_kind
        e["key"] = self.ck.key
        if self.ck.backend == "metal":     # the Metal loop shape and tiles (docs/METAL.md)
            e["metal"] = self.ck.art.plan.config()
        if self.armed or self.disarmed:
            e["checks"] = sorted(self.armed)
            if self.disarmed:
                e["checks_disarmed"] = self.disarmed
        return e


class RecomputeStep(Step):
    """A value computed again before a later reader instead of being kept alive (docs/MEMORY.md).
    It runs the same compiled step as the original: the same kernel binary, the same launch config
    and inputs that are themselves either kept or recomputed bit for bit, so the result is bitwise
    equal. Check runs verify that: the stored value is kept aside and compared bitwise."""

    kind = "recompute"

    def __init__(self, producer: "Step"):
        super().__init__(f"recompute {producer.name}", producer.nodes)
        self.producer = producer

    def run(self, env, rt):
        self.producer.run(env, rt)


_COPY_STREAMS: Dict[Any, Any] = {}


def _copy_stream(device):
    st = _COPY_STREAMS.get(device)
    if st is None:
        st = _COPY_STREAMS[device] = torch.cuda.Stream(device=device)
    return st


class OffloadStep(Step):
    """Starts copying a stored value to pinned host memory on a side stream (docs/MEMORY.md, host
    offload). The device buffer stays until the FenceStep after it; the copy is exact."""

    kind = "offload"

    def __init__(self, slot: int, host: int, name: str):
        super().__init__(f"offload {name}")
        self.slot, self.host, self.done = slot, host, None

    def run(self, env, rt):
        t = env[self.slot]
        cs = _copy_stream(t.device)
        cs.wait_stream(torch.cuda.current_stream(t.device))
        with torch.cuda.stream(cs):
            h = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device="cpu", pin_memory=True)
            h.copy_(t, non_blocking=True)
            self.done = torch.cuda.Event()
            self.done.record(cs)
        env[self.host] = h


class FenceStep(Step):
    """The main stream waits for an offload copy; the schedule drops the device buffer after it."""

    kind = "fence"

    def __init__(self, offload: OffloadStep):
        super().__init__(f"fence {offload.name[8:]}")
        self.offload, self.slot = offload, offload.slot

    def run(self, env, rt):
        torch.cuda.current_stream(env[self.slot].device).wait_event(self.offload.done)


class ReloadStep(Step):
    """Starts copying an offloaded value back into a new device buffer on the side stream."""

    kind = "reload"

    def __init__(self, offload: OffloadStep, device):
        super().__init__(f"reload {offload.name[8:]}")
        self.offload, self.slot, self.host, self.device, self.done = offload, offload.slot, offload.host, device, None

    def run(self, env, rt):
        h = env[self.host]
        main = torch.cuda.current_stream(self.device)
        d = torch.empty_strided(h.shape, h.stride(), dtype=h.dtype, device=self.device)   # on the main stream
        cs = _copy_stream(d.device)
        cs.wait_stream(main)
        with torch.cuda.stream(cs):
            d.copy_(h, non_blocking=True)
            self.done = torch.cuda.Event()
            self.done.record(cs)
        env[self.slot] = d

    def sync(self, env) -> None:
        torch.cuda.current_stream(self.device).wait_event(self.done)


class WaitStep(Step):
    """The main stream waits for a reload, right before the first reader of the value."""

    kind = "wait"

    def __init__(self, reload: ReloadStep):
        super().__init__(f"wait {reload.name[7:]}")
        self.reload, self.slot = reload, reload.slot

    def run(self, env, rt):
        self.reload.sync(env)


class AdapterStep(Step):
    """Casts a bool or integer tensor to the float form kernels load."""

    kind = "adapter"

    def __init__(self, name, src, dst, dtype, device):
        super().__init__(name)
        self.src, self.dst, self.dtype, self.device = src, dst, dtype, device

    def run(self, env, rt):
        env[self.dst] = env[self.src].to(device=self.device, dtype=self.dtype)


class ImplStep(Step):
    """Runs the reference implementation of a layout op directly on the working tensors."""

    kind = "layout"

    def __init__(self, name, node, in_slots, out, shape_idx, static_shape):
        super().__init__(name, [node])
        self.fn = iref.IMPLS[node.op]
        self.attrs = node.attrs
        self.in_slots = in_slots
        self.out = out
        self.shape_idx = shape_idx
        self.static_shape = static_shape

    def run(self, env, rt):
        shape = self.static_shape
        if shape is None:
            dims = rt.dims
            shape = tuple(dims[i] for i in self.shape_idx)
        try:
            env[self.out] = self.fn(self.attrs, [env[s] for s in self.in_slots], shape)
        except IRError as e:
            if self.nodes[0].op != "select_dynamic":
                raise
            from .checks import bounds_error
            x, i = env[self.in_slots[0]], env[self.in_slots[1]]
            d = self.attrs["dim"] % x.dim()
            raise bounds_error("select_dynamic", self.nodes[0], i.reshape(1), x.shape[d], False) from e


class CreateStep(Step):
    kind = "create"

    def __init__(self, name, node, in_slots, out, shape_idx, dtype, device):
        super().__init__(name, [node])
        self.attrs = node.attrs
        self.op = node.op
        self.in_slots = in_slots
        self.out = out
        self.shape_idx = shape_idx
        self.dtype = dtype
        self.device = device

    def run(self, env, rt):
        a = self.attrs
        if self.op == "arange":
            end = a["end"]
            if not isinstance(end, (int, float)):   # a symbolic end (arange(S)): its length from the dim table
                n = rt.dims[self.shape_idx[0]]
                end = a["start"] + n * a["step"]
            env[self.out] = torch.arange(a["start"], end, a["step"], dtype=self.dtype, device=self.device)
            return
        dims = rt.dims
        value = a["value"]
        if self.dtype.is_floating_point and isinstance(value, float):
            value = saturate_scalar(value, self.dtype)
        env[self.out] = torch.full(tuple(dims[i] for i in self.shape_idx), value, dtype=self.dtype,
                                   device=self.device)


class HostStep(Step):
    """Integer bookkeeping on the CPU through the reference implementation (exact)."""

    kind = "host"

    def __init__(self, name, node, in_slots, outs, shape_idx, opts):
        super().__init__(name, [node])
        self.node = node
        self.fn = iref.IMPLS[node.op]
        self.in_slots = in_slots
        self.outs = outs
        self.shape_idx = shape_idx
        self.opts = opts
        self.binding = node.op in iref.BINDING_OPS
        self.in_kinds = [None if x is None else kind_of(x.type.dtype) for x in node.inputs]

    def run(self, env, rt):
        args = [None if s is None else _ref_form(env[s], k) for s, k in zip(self.in_slots, self.in_kinds)]
        dims = rt.dims
        shapes = [tuple(dims[i] for i in idx) for idx in self.shape_idx]
        shape = shapes[0] if len(shapes) == 1 else shapes
        res = self.fn(self.node.attrs, args, shape, bindings=rt.bindings) if self.binding \
            else self.fn(self.node.attrs, args, shape)
        res = list(res) if isinstance(res, (list, tuple)) else [res]
        if len(res) != len(self.outs):
            raise IRError(f"reference {self.node.op} produced {len(res)} values, the node defines {len(self.outs)}")
        for s, r in zip(self.outs, res):
            env[s] = _to_working(r, self.opts)


class DimScalarStep(Step):
    """The run time value of a dim as a scalar tensor, read from the evaluated dim table."""

    kind = "host"

    def __init__(self, name, node, out, index, dtype, device):
        super().__init__(name, [node])
        self.out, self.index, self.dtype, self.device = out, index, dtype, device

    def run(self, env, rt):
        t = rt.cache.get(self)
        if t is None:
            t = rt.cache[self] = torch.tensor(rt.dims[self.index], dtype=self.dtype, device=self.device)
        env[self.out] = t


class FallbackStep(HostStep):
    """An op with no kernel lowering: its float64 reference implementation on the CPU. Listed in the
    compile report with the reason; correct and slow."""

    kind = "fallback"

    def __init__(self, name, node, in_slots, outs, shape_idx, opts, reason):
        super().__init__(name, node, in_slots, outs, shape_idx, opts)
        self.reason = reason

    def entry(self):
        e = super().entry()
        e["reason"] = self.reason
        e["source"] = self.node.source
        e["path"] = "float64 reference implementation on the CPU"
        return e


def _pair(v):
    if isinstance(v, int):
        return v, v
    v = list(v)
    return (v[0], v[0]) if len(v) == 1 else (v[0], v[1])


class LibraryStep(Step):
    """Convolution and pooling through torch: a declared library kernel, checked like any other."""

    kind = "library"

    def __init__(self, name, node, in_slots, out, opts, reason=""):
        super().__init__(name, [node])
        self.node = node
        self.reason = reason or "asked for with library="
        self.in_slots = in_slots
        self.out = out
        self.op = node.op
        a = node.attrs
        self.widen = opts.device == "cpu" and opts.dtype != "float32"   # torch has no half conv on CPU
        self.dtype = opts.torch_dtype
        self.no_tf32 = opts.device == "cuda" and opts.dtype == "float32" and not opts.allow_tf32
        if self.op == "conv2d":
            self.kw = dict(stride=_pair(a.get("stride", 1)), padding=_pair(a.get("padding", 0)),
                           dilation=_pair(a.get("dilation", 1)), groups=a.get("groups", 1))
            self.impl = "torch.nn.functional.conv2d" + (" (cuDNN)" if opts.device == "cuda" else "")
        elif self.op == "max_pool2d":
            k = _pair(a["kernel_size"])
            self.kw = dict(kernel_size=k, stride=_pair(a.get("stride") or a["kernel_size"]),
                           padding=_pair(a.get("padding", 0)), dilation=_pair(a.get("dilation", 1)),
                           ceil_mode=bool(a.get("ceil_mode", False)))
            self.impl = "torch.nn.functional.max_pool2d"
        elif self.op == "avg_pool2d":
            k = _pair(a["kernel_size"])
            self.kw = dict(kernel_size=k, stride=_pair(a.get("stride") or a["kernel_size"]),
                           padding=_pair(a.get("padding", 0)), ceil_mode=bool(a.get("ceil_mode", False)),
                           count_include_pad=bool(a.get("count_include_pad", True)),
                           divisor_override=a.get("divisor_override"))
            self.impl = "torch.nn.functional.avg_pool2d"
        else:
            self.kw = dict(output_size=tuple(_pair(a["output_size"])))
            self.impl = "torch.nn.functional.adaptive_avg_pool2d"

    def run(self, env, rt):
        args = [None if s is None else env[s] for s in self.in_slots]
        if self.widen:
            args = [None if t is None else t.float() for t in args]
        if self.op == "conv2d":
            if self.no_tf32 and torch.backends.cudnn.allow_tf32:
                torch.backends.cudnn.allow_tf32 = False
                try:
                    out = F.conv2d(args[0], args[1], args[2], **self.kw)
                finally:
                    torch.backends.cudnn.allow_tf32 = True
            else:
                out = F.conv2d(args[0], args[1], args[2], **self.kw)
        elif self.op == "max_pool2d":
            out = F.max_pool2d(args[0], **self.kw)
        elif self.op == "avg_pool2d":
            out = F.avg_pool2d(args[0], **self.kw)
        else:
            out = F.adaptive_avg_pool2d(args[0], **self.kw)
        env[self.out] = out.to(self.dtype) if self.widen else out

    def entry(self):
        e = super().entry()
        e["impl"] = self.impl
        e["reason"] = self.reason
        return e


class RandomStep(Step):
    """A ``random`` op through the keyed generator's compiled kernel on the working device. The draw
    is made in the op's declared dtype (the bits the reference interpreter makes too), then cast to
    the working dtype if that differs."""

    kind = "random"

    def __init__(self, name, node, in_slots, out, shape_idx, opts):
        super().__init__(name, [node])
        from .. import rng

        self.node = node
        self.in_slots = in_slots
        self.out = out
        self.shape_idx = shape_idx
        self.attrs = node.attrs
        self.device = opts.torch_device
        declared = node.output.type.dtype
        self.working = opts.torch_dtype if declared.is_floating_point else declared
        have = rng.available_backends(str(self.device))
        self.backend = "triton" if "triton" in have else ("numba" if "numba" in have else "torch")

    def run(self, env, rt):
        from .. import rng

        a = self.attrs
        vals = [int(env[s].item()) for s in self.in_slots]
        shape = tuple(rt.dims[i] for i in self.shape_idx)
        if self.node.op == "random_batched":  # vmap of random: row k has the batch index k as one more stream
            t = rng.generate_batch(a["dist"], shape, a["dtype"], vals[0], a["site"], vals[1:], p=a.get("p"),
                                   low=a.get("low", 0), high=a.get("high"), device=self.device)
        else:
            t = rng.generate(a["dist"], shape, a["dtype"], vals[0], a["site"], vals[1:], p=a.get("p"),
                             low=a.get("low", 0), high=a.get("high"), device=self.device, backend=self.backend)
        env[self.out] = t if t.dtype == self.working else t.to(self.working)

    def entry(self):
        e = super().entry()
        e["impl"] = "truestar.rng (torch, batched)" if self.node.op == "random_batched" else f"truestar.rng ({self.backend})"
        e["site"] = self.attrs.get("site_name")
        return e


def _checked(rt, op: str, index: torch.Tensor, size: int, unique: bool = False) -> torch.Tensor:
    """``index`` with negative entries wrapped and every entry clamped into [0, size), so the torch
    op never reads or writes out of bounds (a bad CUDA index would end the process with a device
    assert). Whether any entry was out of [-size, size), or repeated with ``unique``, is recorded on
    the device and read once after the whole run (``Runtime.raise_pending``): no host sync per op."""
    key = (id(index), size, unique)
    hit = rt.checked.get(key)
    if hit is not None and hit[0] is index:   # the same index tensor read again (every layer's gather)
        return hit[1]
    src = index
    index = index.long()
    if index.numel() == 0:
        return index
    bad = (index < -size).any() | (index >= size).any()
    index = torch.where(index < 0, index + size, index).clamp(0, size - 1)
    if unique and index.numel() > 1:
        s = torch.sort(index.reshape(-1)).values
        bad = bad | (s[1:] == s[:-1]).any()
    rt.pending.append((bad, f"{op}: an index is {'out of range or repeated' if unique else 'out of range'} "
                            f"for a dim of size {size}", op, rt.node, src, size, unique))
    rt.checked[key] = (src, index)
    return index


class _Deterministic:
    """torch.use_deterministic_algorithms(True) for one call (scatter_add_ on CUDA has a
    deterministic implementation under it), then the caller's setting again."""

    def __enter__(self):
        self.prev = torch.are_deterministic_algorithms_enabled()
        self.warn = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)

    def __exit__(self, *exc):
        torch.use_deterministic_algorithms(self.prev, warn_only=self.warn)
        return False


def _index_sort(rt, a, args):
    (x,) = args
    s = torch.sort(x, dim=a.get("dim", -1), descending=bool(a.get("descending", False)), stable=True)
    return [s.values, s.indices]


def _index_topk(rt, a, args):
    # a stable sort, then the first k: the reference's tie order (torch.topk does not promise one)
    (x,) = args
    d = a.get("dim", -1) % x.dim()
    s = torch.sort(x, dim=d, descending=bool(a.get("largest", True)), stable=True)
    k = a["k"]
    return [s.values.narrow(d, 0, k).contiguous(), s.indices.narrow(d, 0, k).contiguous()]


def _index_gather(rt, a, args):
    x, i = args
    d = a["dim"] % x.dim()
    return [torch.gather(x, d, _checked(rt, "gather", i, x.shape[d]))]


def _index_scatter_add(rt, a, args):
    x, i, src = args
    d = a["dim"] % x.dim()
    i = _checked(rt, "scatter_add", i, x.shape[d])
    with _Deterministic():
        return [x.clone().scatter_add_(d, i, src.to(x.dtype))]


def _index_select(rt, a, args):
    x, i = args
    d = a["dim"] % x.dim()
    return [torch.index_select(x, d, _checked(rt, "index_select", i.reshape(-1), x.shape[d]))]


def _index_copy(rt, a, args):
    # indices are checked unique, so every row is written once and the order of writes cannot matter
    x, i, src = args
    d = a["dim"] % x.dim()
    return [x.index_copy(d, _checked(rt, "index_copy", i, x.shape[d], unique=True), src.to(x.dtype))]


def _index_index(rt, a, args):
    x, idx = args[0], list(args[1:])
    return [x[tuple(slice(None) if i is None else _checked(rt, "index", i, x.shape[d]) for d, i in enumerate(idx))]]


def _index_kvq_attention(rt, a, args):
    """kvq_attention through truestar.kvquant's split KV kernel (float16 or bfloat16 math, float32
    accumulation, fixed order merge); its first run is region checked against the float64
    reference like every library step. Meta tensors (the memory planner) get a meta result."""
    q, kd, ks, vd, vs, tk, tv, tl = args[:8]
    pl = args[8] if len(args) > 8 else None
    kb, vb = (args[9], args[10]) if len(args) > 10 and args[9] is not None else (None, None)
    if q.device.type == "meta":
        return [torch.empty(q.shape, dtype=q.dtype, device="meta")]
    from .. import kvquant as KQ
    half = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
    if a["fmt"] == "fp8":
        kd, vd = kd.view(torch.float8_e4m3fn), vd.view(torch.float8_e4m3fn)
    if kb is not None:                  # int4 / nvfp4: E4M3 block scales passed as their uint8 bits
        rot = bool(a.get("rot", False))
        kc = KQ.QuantKV(kd, ks.float(), a["fmt"], a["group"], False, kb.view(torch.float8_e4m3fn), rot)
        vc = KQ.QuantKV(vd, vs.float(), a["fmt"], a["vgroup"], False, vb.view(torch.float8_e4m3fn), rot)
    else:
        kc = KQ.QuantKV(kd, ks.float(), a["fmt"], a["group"], bool(a["channel"]))
        vc = KQ.QuantKV(vd, vs.float(), a["fmt"], a["vgroup"])
    det = getattr(getattr(rt, "opts", None), "deterministic", None)   # None follows torch
    out = KQ.decode_attention(q.to(half), kc, vc, scale=a.get("scale") or None, deterministic=det,
                              verify=False, tail=(tk.to(half), tv.to(half), tl.reshape(1).to(torch.int32)),
                              prefix_len=None if pl is None else pl.reshape(1).to(torch.int32))
    return [out.to(q.dtype)]


def _index_mla_attention(rt, a, args):
    """mla_attention through truestar.mla.latent: the fused Triton kernel on CUDA, the same math in
    torch ops on the CPU; region checked against the float64 reference on its first run."""
    q = args[0]
    if q.device.type == "meta":
        return [torch.empty(q.shape[:3] + (args[8].shape[1],), dtype=q.dtype, device="meta")]
    from ..mla import latent
    fn = latent.latent_attention if q.device.type == "cuda" else latent.composite
    return [fn(*args, float(a["scale"]))]


def _linattn_args(a, args):
    return [None if x is None else x for x in args[:5]], dict(rule=a["rule"], scale=float(a["scale"]))


def _index_linear_attention(rt, a, args):
    """linear_attention (P60) through truestar.linattn: the Triton kernels on CUDA, the chunked
    PyTorch form on the CPU, the chunk pinned when deterministic. Region checked against the float64
    recurrence on the first run like every library step."""
    (q, k, v, g, beta), kw = _linattn_args(a, args)
    if q.device.type == "meta":
        return [torch.empty(v.shape, dtype=v.dtype, device="meta")]
    from ..linattn.op import linear_attention
    det = getattr(getattr(rt, "opts", None), "deterministic", None)
    return [linear_attention(q, k, v, g, beta, deterministic=det, **kw)]


def _index_linear_attention_vjp(rt, a, args):
    """The cotangent of operand ``wrt``: autograd through the chunked form at chunk 16."""
    (q, k, v, g, beta), kw = _linattn_args(a, args)
    i = a["wrt"]
    x = args[i]
    if x.device.type == "meta":
        return [torch.empty(x.shape, dtype=x.dtype, device="meta")]
    from ..linattn.chunked import chunked
    from ..linattn.triton_kernels import DET_CHUNK
    with torch.inference_mode(False), torch.enable_grad():
        leaves = [None if t is None else t.detach().clone().requires_grad_(j == i)
                  for j, t in enumerate((q, k, v, g, beta))]
        o, _ = chunked(*leaves, chunk=DET_CHUNK, **kw)
        (gx,) = torch.autograd.grad(o, [leaves[i]], args[5].to(o.dtype), allow_unused=True)
    return [torch.zeros_like(x) if gx is None else gx.to(x.dtype)]


def _index_linear_attention_jvp(rt, a, args):
    """The tangent of o: forward AD (torch.func.jvp) through the chunked form at chunk 16."""
    (q, k, v, g, beta), kw = _linattn_args(a, args)
    if q.device.type == "meta":
        return [torch.empty(v.shape, dtype=v.dtype, device="meta")]
    from ..linattn.chunked import chunked
    from ..linattn.triton_kernels import DET_CHUNK
    ins = [q, k, v, g, beta]
    present = [j for j, x in enumerate(ins) if x is not None]

    def f(*xs):
        full = list(ins)
        for j, x in zip(present, xs):
            full[j] = x
        return chunked(*full, chunk=DET_CHUNK, **kw)[0]
    tans = tuple(torch.zeros_like(ins[j]) if args[5 + j] is None else args[5 + j].to(ins[j].dtype) for j in present)
    with torch.inference_mode(False):
        return [torch.func.jvp(f, tuple(ins[j] for j in present), tans)[1].to(v.dtype)]


def _index_attention_grad(rt, a, args):
    """attention_grad (ir/attention_grad.py) through compiler/attention_bwd.py: Truestar's Triton
    kernels on CUDA (fixed order sums, no atomics), float32 torch blocks on the CPU."""
    from .attention_bwd import attention_grad
    tf32 = bool(getattr(getattr(rt, "opts", None), "allow_tf32", False))
    return [attention_grad(*args[:5], float(a["scale"]), bool(a["is_causal"]), tf32=tf32)]


INDEX_IMPLS = {"attention_grad": _index_attention_grad, "sort": _index_sort, "topk": _index_topk, "gather": _index_gather,
               "scatter_add": _index_scatter_add, "index_select": _index_select, "index_copy": _index_copy,
               "index": _index_index, "kvq_attention": _index_kvq_attention,
               "linear_attention": _index_linear_attention,
               "mla_attention": _index_mla_attention,
               "linear_attention_vjp": _index_linear_attention_vjp,
               "linear_attention_jvp": _index_linear_attention_jvp}


class IndexStep(Step):
    """Sorting and data movement (sort, topk, gather, scatter_add, index_select, index_copy, index)
    through torch on the executor's device. Entries are copied, never computed, except the sums of
    repeated scatter_add indices (deterministic, in torch's order). Checked against the reference on
    the first run like a library kernel."""

    kind = "library"

    def __init__(self, name, node, in_slots, outs, device: str = ""):
        super().__init__(name, [node])
        from . import gather_native as GN
        self.node = node
        self.fn = INDEX_IMPLS[node.op]
        self.attrs = node.attrs
        self.in_slots = in_slots
        self.outs = outs
        self.impl = f"torch ({node.op})"
        if node.op == "attention_grad":
            self.impl = "truestar attention_grad (" + ("triton" if device == "cuda" else "float32 torch blocks") + ")"
        # Truestar's own kernel (compiler/gather_native.py) for the reads it takes; None keeps torch
        self.native = None if GN.static_reason(node, device) else GN.backend_for(device)
        if self.native:
            self.impl = f"truestar {node.op} ({self.native})"
        self.torch_runs = 0      # calls whose tensors the native kernel did not take (shown in the entry)

    def run(self, env, rt):
        rt.node = self.node
        args = [None if s is None else env[s] for s in self.in_slots]
        if self.native:
            from . import gather_native as GN
            got = (GN.scatter_add(self.attrs, args, self.native) if self.node.op == "scatter_add"
                   else GN.gather(self.node.op, self.attrs, args, self.native))
            if got is not None:
                out, flag, src = got
                rt.pending.append((flag, f"{self.node.op}: an index is out of range", self.node.op, rt.node,
                                   src, _index_size(self.node.op, self.attrs, args), False))
                env[self.outs[0]] = out
                return
            self.torch_runs += 1
        res = self.fn(rt, self.attrs, args)
        for s, r in zip(self.outs, res):
            env[s] = r

    @property
    def out(self) -> int:
        return self.outs[0]

    def run_into(self, env, rt, j: int):
        """``index_copy`` written into the buffer of its input 0 (the memory planner proved it dead
        after this step, or the caller declared it updated in place): the KV cache of a decode step
        is then written at its new positions only, not copied whole. Same entries, same order of
        writes (indices are checked unique), so the result is bitwise the out of place one."""
        if self.node.op != "index_copy" or j != 0:
            raise RuntimeError(f"index step {self.node.op} has no in place form for input {j}")
        rt.node = self.node
        x, i, src = [env[s] for s in self.in_slots]
        d = self.attrs["dim"] % x.dim()
        x.index_copy_(d, _checked(rt, "index_copy", i, x.shape[d], unique=True), src.to(x.dtype))
        env[self.outs[0]] = x

    def entry(self):
        e = super().entry()
        e["impl"] = self.impl if not (self.native and self.torch_runs) else             f"{self.impl}, torch ({self.node.op}) on {self.torch_runs} calls"
        return e


def _index_size(op, attrs, args) -> int:
    """The size of the input dim the first index tensor reads (for the out of range message)."""
    x = args[0]
    if op in ("index_select", "gather", "scatter_add"):
        return x.shape[attrs["dim"] % x.dim()]
    k = next(k for k, t in enumerate(args[1:]) if t is not None)
    return x.shape[k]


class IntStep(Step):
    """Integer tensor arithmetic run by the reference implementation on the device's own tensors
    (plain torch integer ops, exact on any device). Checked against the CPU reference on the first
    run, with zero tolerance."""

    kind = "library"

    def __init__(self, name, node, in_slots, out, shape_idx, device: str = ""):
        super().__init__(name, [node])
        from .coverage_rules import DEVICE_INT_IMPLS
        from . import gather_native as GN
        self.node = node
        self.fn = DEVICE_INT_IMPLS.get(node.op, iref.IMPLS[node.op])
        self.attrs = node.attrs
        self.in_slots = in_slots
        self.out = out
        self.shape_idx = shape_idx
        self.impl = f"torch integer {node.op}"
        self.native = None if GN.static_reason(node, device) else GN.backend_for(device)
        if self.native:
            self.impl = f"truestar {node.op} ({self.native})"
        self.torch_runs = 0

    def run(self, env, rt):
        dims = rt.dims
        if self.native:
            from . import gather_native as GN
            x = env[self.in_slots[0]]
            got = GN.arg_extreme(self.node.op, self.attrs, x, None, self.native)
            if got is not None:
                env[self.out] = got
                return
            self.torch_runs += 1
        env[self.out] = self.fn(self.attrs, [None if s is None else env[s] for s in self.in_slots],
                                tuple(dims[i] for i in self.shape_idx))

    def entry(self):
        e = super().entry()
        e["impl"] = self.impl
        return e


class CondStep(Step):
    kind = "cond"

    def __init__(self, name, node, pred, args, outs, then, other):
        super().__init__(name, [node])
        self.pred, self.args, self.outs, self.then, self.other = pred, args, outs, then, other

    def run(self, env, rt):
        plan = self.then if bool(env[self.pred].item()) else self.other
        res = plan.run([env[s] for s in self.args], rt)
        for s, r in zip(self.outs, res):
            env[s] = r

    def entry(self):
        e = super().entry()
        e["regions"] = {"then_branch": self.then.report_entries(), "else_branch": self.other.report_entries()}
        return e


class LoopStep(Step):
    kind = "loop"

    def __init__(self, name, node, carried, captures, outs, body, cond, trip, limit):
        super().__init__(name, [node])
        self.carried, self.captures, self.outs = carried, captures, outs
        self.body, self.cond, self.trip, self.limit = body, cond, trip, limit

    def run(self, env, rt):
        carried = [env[s] for s in self.carried]
        caps = [env[s] for s in self.captures]
        body = self.body
        track = rt.chk is not None and rt.chk.locating
        if track:
            rt.chk.loops[self] = 0
        if self.cond is None:
            for k in range(max(0, rt.dims[self.trip])):
                if track:
                    rt.chk.loops[self] = k
                carried = body.run([torch.tensor(k, dtype=torch.int64)] + carried + caps, rt)
        else:
            k = 0
            older = old = None
            while bool(self.cond.run(carried + caps, rt)[0].item()):
                if self.limit is not None and k >= self.limit:
                    raise IRError(f"loop ran more than max_iterations={self.limit} times")
                if k >= STALL_CHECK_FROM and k % STALL_CHECK_EVERY < 3:
                    # A condition that needs more precision than the working dtype never turns false:
                    # the carried values reach a fixed point or a two cycle. Say so instead of spinning.
                    if any(p is not None and all(torch.equal(a, b) for a, b in zip(carried, p))
                           for p in (old, older)):
                        raise IRError(
                            f"loop {self.name} cannot terminate in {rt.opts.dtype} on {rt.opts.device}: after "
                            f"{k} iterations the carried values repeat exactly while the condition stays "
                            f"true. The condition needs more precision than {rt.opts.dtype}; use float32, "
                            f"loosen the threshold or give the loop max_iterations.")
                    older, old = old, carried
                if track:
                    rt.chk.loops[self] = k
                carried = body.run([torch.tensor(k, dtype=torch.int64)] + carried + caps, rt)
                k += 1
        if track:
            rt.chk.loops.pop(self, None)
        for s, r in zip(self.outs, carried):
            env[s] = r

    def entry(self):
        e = super().entry()
        e["regions"] = {"body": self.body.report_entries()}
        if self.cond is not None:
            e["regions"]["cond"] = self.cond.report_entries()
        return e


# ----------------------------------------------------------------------------- checks


_CONTRACTIONS = {"matmul": (0, -1), "linear": (0, -1), "attention": (1, -2)}   # input, dim summed over


class GroupCheck:
    """Compare the values a kernel group (or a library step) produced with the float64 reference
    implementation of the IR ops it covers, run on the inputs the group really received."""

    def __init__(self, nodes, ext, outs, shape_idx, tol):
        self.nodes = nodes          # IR nodes in program order
        self.ext = ext              # [(value, kind, slot or None, exact constant or None)]
        self.outs = outs            # [(value, kind, slot)]
        self.shape_idx = shape_idx  # per node: per output: indices into the dim table
        self.tol = tol
        self.device = None          # where the last run's reference ran (None: the CPU)
        # values the GPU run may free after node i: read for the last time there, not compared
        keep = {id(v) for v, _, _ in outs}
        last: Dict[int, int] = {}
        for i, node in enumerate(nodes):
            for x in node.inputs:
                if x is not None:
                    last[id(x)] = i
            for o in node.outputs:
                last.setdefault(id(o), i)
        self.dead_after: List[List[int]] = [[] for _ in nodes]
        for vid, i in last.items():
            if vid not in keep:
                self.dead_after[i].append(vid)

    def size(self, env, rt) -> Tuple[int, int]:
        """(work, peak): elements of the largest step of the reference (an output, or a contraction's
        output times its summed length), and a working set estimate (the inputs plus four times the
        largest value)."""
        dims = rt.dims
        shapes: Dict[int, Tuple[int, ...]] = {}
        peak = 0
        for v, kind, slot, const in self.ext:
            t = const if const is not None else env[slot]
            shapes[id(v)] = tuple(t.shape)
            peak += t.numel()
        work = big = 1
        for node, idxs in zip(self.nodes, self.shape_idx):
            for o, idx in zip(node.outputs, idxs):
                sh = tuple(dims[i] for i in idx)
                shapes[id(o)] = sh
                n = math.prod(sh)
                big = max(big, n)
                inner = 1
                if node.op in _CONTRACTIONS:
                    i, d = _CONTRACTIONS[node.op]
                    s = shapes.get(id(node.inputs[i])) if node.inputs[i] is not None else None
                    inner = s[d] if s and len(s) >= -d else 1
                elif node.op == "conv2d":
                    s = shapes.get(id(node.inputs[1]))
                    inner = math.prod(s[1:]) if s else 1
                work = max(work, n * inner)
        return work, peak + 4 * big

    def reference(self, env, rt, device: Optional[torch.device] = None, drop: bool = False) -> Dict[int, torch.Tensor]:
        """Every value of the group in float64: on the CPU, or on ``device`` with the same
        implementations (``drop`` frees values no later node reads and no comparison needs)."""
        local: Dict[int, torch.Tensor] = {}
        for v, kind, slot, const in self.ext:
            if const is not None:
                local[id(v)] = const if device is None else const.to(device)
            else:
                local[id(v)] = _ref_form(env[slot], kind, device)
        dims = rt.dims
        for i, (node, idxs) in enumerate(zip(self.nodes, self.shape_idx)):
            args = [None if x is None else local[id(x)] for x in node.inputs]
            shapes = [tuple(dims[i] for i in idx) for idx in idxs]
            shape = shapes[0] if len(shapes) == 1 else shapes
            if device is None:
                res = iref.IMPLS[node.op](node.attrs, args, shape)
            else:
                res = iref.call_on_device(node.op, node.attrs, args, shape, device)
            del args
            res = list(res) if isinstance(res, (list, tuple)) else [res]
            for o, r in zip(node.outputs, res):
                local[id(o)] = r
                if rt.decisions is not None and id(o) in rt.decisions:
                    rt.decisions[id(o)] = r.detach().to("cpu")   # the decision this run took
            if drop:
                for vid in self.dead_after[i]:
                    local.pop(vid, None)
        return local

    def _within_term_magnitude(self, v, env, rt, slot) -> bool:
        """For a value made by a sum like op (``_TERM_MAGNITUDE_OPS``) that failed the plain measure:
        is its error within the tolerance relative to the magnitude of the terms it added? A sum
        whose terms cancel (a bias gradient behind batch norm, zero in exact math) keeps only
        rounding noise, which the plain measure divides by the tiny result (kernel/verify.py)."""
        node = next((n for n in self.nodes if any(o is v for o in n.outputs)), None)
        if node is None or node.op not in _TERM_MAGNITUDE_OPS or node.extra_outputs:
            return False
        full = self.reference(env, rt)
        args = [None if x is None else full[id(x)].abs() for x in node.inputs]
        ref = full[id(v)]
        mag = iref.IMPLS[node.op](node.attrs, args, tuple(int(d) for d in ref.shape))
        return _compare(v.name, env[slot], ref, self.tol, mag).ok

    def _within_derived(self, v, env, rt, slot, opts) -> bool:
        """For a float value over the fixed tolerance: is it within the bound derived for this region
        (truestar.tolerance, docs/TOLERANCE.md)? The region's IR ops run again in float64 from its
        inputs, rounded to the working dtype, and with every op given the error a correct kernel can
        make there, propagated to the value. A formula that cancels (1 - exp(x) for a tiny x, written
        by hand) is far from float64 in every correct float32 execution and passes; a wrong kernel is
        far from all of these runs and does not."""
        if opts.dtype == "float64" or opts.mixed or not _derivable(self.nodes):   # mixed=True: its own declared precision bars
            return False
        from .. import tolerance as TL
        ins = tuple(const if const is not None else env[sl] for _, _, sl, const in self.ext)
        ext, nodes, shape_idx, dims = self.ext, self.nodes, self.shape_idx, rt.dims

        def fn(*xs):
            local = {}
            for (vv, kind, _, _), x in zip(ext, xs):
                local[id(vv)] = x if kind == "f" else _ref_form(x, kind, None).to(x.device)
            for node, idxs in zip(nodes, shape_idx):
                args = [None if q is None else local[id(q)] for q in node.inputs]
                shapes = [tuple(dims[i] for i in idx) for idx in idxs]
                res = iref.IMPLS[node.op](node.attrs, args, shapes[0] if len(shapes) == 1 else shapes)
                res = list(res) if isinstance(res, (list, tuple)) else [res]
                for o, r in zip(node.outputs, res):
                    local[id(o)] = r
            return local[id(v)]
        try:
            rep = TL.close_report(fn, ins, env[slot], dtype=opts.torch_dtype,
                                  tf32=bool(opts.allow_tf32) and opts.device == "cuda")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False
        except Exception:  # noqa: BLE001 (no derivation: the fixed tolerance stays the verdict)
            return False
        return bool(rep.ok)

    def run(self, plan: "Plan", step: Step, env, rt) -> None:
        t0 = time.perf_counter()
        dev = check_device(plan.opts, *self.size(env, rt), env[self.outs[0][2]])
        local = None
        if dev is not None:
            try:
                local = self.reference(env, rt, dev, drop=True)
            except torch.cuda.OutOfMemoryError:     # the GPU is shared: this check goes to the CPU
                local = None
                torch.cuda.empty_cache()
                dev = None
        if local is None:
            local = self.reference(env, rt)
        else:
            rt.stats["gpu_checks"] += 1
        self.device = dev
        rt.stats["checks"] += 1
        for v, kind, slot in self.outs:
            c = _compare(v.name, env[slot], local[id(v)], 0.0 if kind != "f" else self.tol)
            if not c.ok and kind == "f" and not c.detail and self._within_term_magnitude(v, env, rt, slot):
                continue
            if not c.ok and kind == "f" and not c.detail and self._within_derived(v, env, rt, slot, plan.opts):
                rt.stats["derived_accepts"] = rt.stats.get("derived_accepts", 0) + 1
                continue
            if (not c.ok and kind == "f" and plan.opts.mixed and c.detail.startswith("infinities")
                    and env[slot].dtype in (torch.float16, torch.bfloat16)):
                # mixed=True: a value declared in 16 bits overflows where that dtype overflows (the
                # float16 loss scaler relies on it); compare with the reference sent through the
                # same range, every finite element still under the kernel tolerance
                c = _compare(v.name, env[slot], _overflow_like(local[id(v)], env[slot].dtype), self.tol)
                if c.ok:
                    continue
            if c.error == c.error and c.error != math.inf:
                rt.stats["max_error"] = max(rt.stats["max_error"], c.error)
            if not c.ok:
                if dev is not None:              # every value again, for naming the op
                    local = self.reference(env, rt, dev)
                raise plan.mismatch(step, self, v, c, local, rt)
        rt.stats["seconds"] += time.perf_counter() - t0


def _compare(name, out: torch.Tensor, ref: torch.Tensor, tol: float, mag: Optional[torch.Tensor] = None):
    """kernel.verify.compare: on the reference's GPU when the reference is there (compare_torch),
    else in NumPy on the CPU. The same rules either way. ``mag``: each element's sum of term
    magnitudes (``_within_term_magnitude``)."""
    if ref.device.type != "cpu":
        m = None if mag is None else mag.to(ref.device)
        return kverify.compare(name, out.detach().to(ref.device), ref.detach(), tol, mag=m)
    m = None if mag is None else kverify.to_host_f64(mag).numpy()
    return kverify.compare(name, kverify.to_host_f64(out).numpy(), kverify.to_host_f64(ref).numpy(), tol, mag=m)


# Ops whose result is a sum of products of their operands: the same op on the operands' absolute
# values is the sum of the terms' magnitudes, which bounds the rounding error of any summation order.
_TERM_MAGNITUDE_OPS = ("sum", "mean", "matmul", "linear", "conv2d")


# mixed=True end to end bar: this many times the distance from float64 of an independent evaluation
# rounded to the declared dtypes (CompiledProgram._mixed_bars). Measured margins: docs/TRAINING.md.
MIXED_FACTOR = 4.0


def _overflow_like(ref: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``ref`` with every element that rounds to infinity in ``dtype`` replaced by that infinity
    (round to nearest: |x| >= max + ulp(max) / 2); the other elements keep their float64 value."""
    fi = torch.finfo(dtype)
    thr = fi.max + fi.max / (2.0 - fi.eps) * fi.eps / 2.0
    inf = torch.full_like(ref, math.inf)
    return torch.where(ref.abs() >= thr, torch.where(ref > 0, inf, -inf), ref)


def _why(c) -> str:
    return c.detail or f"relative error {c.error:.3g} > tolerance {c.tol:.3g}"


# Comparisons whose decision a compiled run records on its verified call (CompiledProgram._branch_flips).
_DECISION_OPS = ("gt", "ge", "lt", "le", "eq", "ne")
# Ops whose kernels compute below the working precision by design (kvq_attention reads a quantized KV
# cache with its own error bound, region checked by its own tolerance): the derived tolerance models
# float32 kernels only, so a program or region holding one keeps the fixed bar as its verdict.
_UNMODELED_OPS = frozenset({"kvq_attention"})


def _derivable(nodes) -> bool:
    return not any(n.op in _UNMODELED_OPS for n in nodes)


def _describe_flips(flips: Sequence[Dict[str, Any]], most: int = 3) -> str:
    parts = []
    for f in flips[:most]:
        src = f" ({f['source']})" if f.get("source") else ""
        parts.append(f"%{f['value']} = {f['op']}(%{f['lhs']}, %{f['rhs']}) on {f['count']} element(s), first at "
                     f"{f['at']} where float64 has {f['lhs_value']:.3g} against {f['rhs_value']:.3g}, "
                     f"{f['margin']:.2g} of the operands' rms apart{src}")
    if len(flips) > most:
        parts.append(f"and {len(flips) - most} more")
    return "; ".join(parts)


# ----------------------------------------------------------------------------- plans


def _inplace_input(step, ip: int, ipd: int, own: bool, update, env, dfirst: bool = False) -> int:
    """The input of step to write its result over, or -1. ip: an intermediate the planner
    proved dead (written over when the runtime holds its only reference); ipd: a program input,
    only when the caller donated the inputs (same guard) or declared this one updated in place
    (update=: the caller asked for the write, so a reference it keeps does not stop it).
    dfirst: the step writes a program output and the planner put the donated input first, so the
    output keeps the donated buffer's address from call to call (a CUDA graph replay of a training
    step needs its carried state to stay put); it falls back to ip when that input is shared."""
    from .memory import sole_owner
    if dfirst and own and ipd >= 0 and sole_owner(env[step.in_slots[ipd]]):
        return ipd
    if ip >= 0:
        return ip if sole_owner(env[step.in_slots[ip]]) else -1
    if ipd < 0:
        return -1
    s = step.in_slots[ipd]
    # the input itself, or a value whose buffer already is that input's (an earlier in place write of
    # an update= input: the planner proved it dead after this step, the caller asked for the write)
    if s in update or getattr(step, "donated_root", None) in update:
        t = env[s]
        return ipd if t.is_contiguous() and t.storage_offset() == 0 else -1
    return ipd if own and sole_owner(env[s]) else -1


class Plan:
    """A compiled program or region: steps over a slot array."""

    has_trig = False     # a step of this plan has a torch side check trigger (compiler/checks.py)

    def __init__(self, prog: Program, path: str, opts: Options, table: DimTable):
        self.prog = prog
        self.path = path
        self.opts = opts
        self.table = table
        self.steps: List[Step] = []
        self.template: List[Any] = []
        self.in_slots: List[int] = []
        self.out_slots: List[int] = []
        self.alias_out: List[bool] = []
        self.islands: List[tuple] = []    # (name, dtype, shape) kept in float32 in a 16 bit program
        self.checked = False
        self.lowering: Optional[Lowering] = None
        self.schedule = None
        self._ops: List[Tuple[Step, Tuple[int, ...], int, int, bool]] = []

    def set_schedule(self, sched) -> None:
        """Install the memory schedule (compiler.memory.schedule): what to drop after each step and
        which kernels write over a dead input."""
        self.schedule = sched
        # (step, frees, in place input, in place donated input, the donated one goes first)
        self._ops = [(st, sched.frees[i] if sched.reuse else (), sched.inplace.get(i, -1),
                      sched.inplace_donated.get(i, -1), i in sched.donated_first) for i, st in enumerate(self.steps)]
        roots = getattr(sched, "donated_root", {})
        for i, st in enumerate(self.steps):
            st.donated_root = roots.get(i)

    def run(self, args: List[torch.Tensor], rt: Runtime, own: bool = False) -> List[torch.Tensor]:
        """Run the steps. ``own``: the caller handed the input tensors over (donation); ``args`` is
        emptied so the steps hold the only references the runtime has."""
        from .memory import sole_owner
        env = self.template.copy()
        for s, a in zip(self.in_slots, args):
            env[s] = a
        if own:
            args.clear()
        check = rt.mode == "all" or (rt.mode == "first" and not self.checked)
        sched = self.schedule
        cur = None
        if rt.timer is not None or (rt.chk is not None and (self.has_trig or rt.chk.locating)):
            return self._run_instrumented(env, rt, check, own)
        try:
            if check:
                stash: Dict[int, torch.Tensor] = {}
                for i, (cur, frees, ip, ipd, dfirst) in enumerate(self._ops):
                    for s in sched.stash.get(i, ()):
                        stash[s] = env[s].clone()          # a recompute check compares against this copy
                    j = _inplace_input(cur, ip, ipd, own, rt.update, env, dfirst)
                    if j >= 0:
                        before = env[cur.in_slots[j]].clone() if cur.check is not None else None
                        cur.run_into(env, rt, j)
                        if before is not None:              # the region check reads the input as it was
                            env[cur.in_slots[j]] = before
                    else:
                        cur.run(env, rt)
                    if cur.check is not None:
                        cur.check.run(self, cur, env, rt)
                    recompute = sched.recompute_slots.get(i)
                    if recompute:
                        self.recompute_check(cur, recompute, env, stash, rt)
                    for s in frees:
                        env[s] = None
                self.checked = True
            else:
                for cur, frees, ip, ipd, dfirst in self._ops:
                    j = _inplace_input(cur, ip, ipd, own, rt.update, env, dfirst)
                    if j >= 0:
                        cur.run_into(env, rt, j)
                    else:
                        cur.run(env, rt)
                    for s in frees:
                        env[s] = None
        except KernelMismatchError as e:
            raise self.kernel_mismatch(cur, e, env, rt) from e
        return [env[s] for s in self.out_slots]

    def _run_instrumented(self, env, rt, check: bool, own: bool) -> List[torch.Tensor]:
        """The step loop of ``run`` with compiled in checks (rt.chk) and or the profiler's timer
        (rt.timer): the same schedule (values dropped after their last reader, kernels writing over
        a dead input, recompute checks). The locate run of the sentinels keeps every value and runs
        nothing in place, so the inputs a step received are still there to name the op."""
        from .memory import sole_owner
        chk, timer = rt.chk, rt.timer
        locating = chk is not None and chk.locating
        sched = self.schedule
        stash: Dict[int, torch.Tensor] = {}
        cur = None
        try:
            for i, (cur, frees, ip, ipd, dfirst) in enumerate(self._ops):
                if check:
                    for s in sched.stash.get(i, ()):
                        stash[s] = env[s].clone()
                if timer is not None:
                    timer.start(self, cur)
                j = -1 if locating else _inplace_input(cur, ip, ipd, own, rt.update, env, dfirst)
                if j >= 0:
                    before = env[cur.in_slots[j]].clone() if (check and cur.check is not None) else None
                    cur.run_into(env, rt, j)
                    if before is not None:
                        env[cur.in_slots[j]] = before
                else:
                    cur.run(env, rt)
                if timer is not None:
                    timer.stop(self, cur, env)
                if chk is not None:
                    if locating:
                        chk.inspect(self, cur, env, rt)
                    elif getattr(cur, "trig", None):
                        chk.after(cur, env)
                if check:
                    if cur.check is not None:
                        cur.check.run(self, cur, env, rt)
                    recompute = sched.recompute_slots.get(i)
                    if recompute:
                        self.recompute_check(cur, recompute, env, stash, rt)
                if not locating:
                    for s in frees:
                        env[s] = None
            if check:
                self.checked = True
        except KernelMismatchError as e:
            raise self.kernel_mismatch(cur, e, env, rt) from e
        return [env[s] for s in self.out_slots]

    def recompute_check(self, step, slots, env, stash, rt) -> None:
        """A recomputed value must equal the stored one bit for bit (TS421)."""
        if isinstance(step, ReloadStep):
            step.sync(env)
        for s in slots:
            stored = stash.pop(s, None)
            if stored is None:
                continue
            ok, n = bitwise_equal(env[s], stored)
            if ok:
                continue
            if isinstance(step, ReloadStep):
                msg = (f"[TS1112] compiled program {self.program_name!r} on {self.opts.device}: the value reloaded "
                       f"from host memory ({step.name}) is not bitwise equal to the stored value ({n} of "
                       f"{stored.numel()} entries differ). The result was not returned. This is a bug in Truestar.")
                raise RecomputeMismatchError(msg, program=self.program_name, region=self.path, op="offload",
                                             value=step.name, ops=[], error=float(n), tolerance=0.0, stage="offload")
            node = step.nodes[-1]
            ops = ", ".join(f"{nd.op} (%{nd.output.name}" + (f", {nd.source}" if nd.source else "") + ")"
                            for nd in step.nodes)
            msg = (f"[TS421] compiled program {self.program_name!r} on {self.opts.device} ({self.opts.dtype}): the "
                   f"recomputed value of {step.producer.name!r} is not bitwise equal to the stored value "
                   f"({n} of {stored.numel()} entries differ).\n  region {self.path}, recomputed {ops}\n"
                   f"Recomputing must give the same bits (same kernel, same launch config, same inputs); "
                   f"something changed an input or the kernel between the two runs. The result was not "
                   f"returned. This is a bug in Truestar, not in your program.")
            raise RecomputeMismatchError(msg, program=self.program_name, region=self.path, op=node.op,
                                     value=node.output.name, ops=step.ops(), error=float(n), tolerance=0.0,
                                     stage="recompute")

    # ---- reporting
    def report_entries(self) -> List[Dict[str, Any]]:
        return [s.entry() for s in self.steps]

    def all_steps(self):
        for s in self.steps:
            yield self, s
            for sub in (getattr(s, "then", None), getattr(s, "other", None), getattr(s, "body", None),
                        getattr(s, "cond", None)):
                if isinstance(sub, Plan):
                    yield from sub.all_steps()

    # ---- mismatch errors
    @property
    def program_name(self) -> str:
        """Name of the whole program this plan belongs to (a region's path starts with it)."""
        return self.path.split("/")[0]

    def _describe(self, step: Step) -> str:
        ops = ", ".join(f"{n.op} (%{n.output.name}" + (f", {n.source}" if n.source else "") + ")"
                        for n in step.nodes)
        return f"region {self.path}, {step.kind} {step.name!r} covering {ops}"

    def mismatch(self, step, check, value, c, local, rt) -> ProgramMismatchError:
        blamed, note = self._drill(step, check, local, rt)
        op = blamed.op if blamed is not None else step.nodes[-1].op
        msg = (f"compiled program {self.program_name!r} on {self.opts.device} ({self.opts.dtype}) does not match "
               f"the float64 reference interpreter.\n  {self._describe(step)}\n  value %{value.name}: {_why(c)}\n"
               f"  op: {op}. {note}\nThe result was not returned. This is a bug in the Truestar compiler, "
               f"not in your program. Set TRUESTAR_VERIFY=0 only to debug.")
        return ProgramMismatchError(msg, program=self.program_name, region=self.path, op=op, value=value.name,
                                    ops=step.ops(), error=c.error, tolerance=c.tol, stage="region")

    def kernel_mismatch(self, step, err, env, rt) -> ProgramMismatchError:
        blamed, note = None, "The kernel level check failed before the region check ran."
        check = step.check if step is not None else None
        if check is not None:
            try:
                blamed, note = self._drill(step, check, check.reference(env, rt), rt)
            except Exception as e:  # noqa: BLE001 (the drill is best effort; the mismatch is still raised)
                note = f"The op could not be isolated ({type(e).__name__}: {e})."
        op = blamed.op if blamed is not None else (step.nodes[-1].op if step is not None and step.nodes else "")
        where = self._describe(step) if step is not None else f"region {self.path}"
        msg = (f"compiled program {self.program_name!r} on {self.opts.device} ({self.opts.dtype}): a kernel does not "
               f"match its float64 reference.\n  {where}\n  op: {op}. {note}\n  kernel check: {err}")
        return ProgramMismatchError(msg, program=self.program_name, region=self.path, op=op,
                                    value=step.nodes[-1].output.name if step is not None and step.nodes else "",
                                    ops=step.ops() if step is not None else [], stage="kernel")

    def _drill(self, step, check, local, rt) -> Tuple[Optional[Node], str]:
        """Compile each op of the group alone, feed it the reference inputs and name the first op that
        is wrong on its own."""
        if len(check.nodes) == 1:
            return check.nodes[0], "It is the only op in this region."
        opts = self.opts
        for node in check.nodes:
            try:
                ins = [x for x in dict.fromkeys(x for x in node.inputs if x is not None)]
                params = [x for x in ins if self.lowering.is_param(x) and self.prog.param_data[x.name].dim() == 0]
                ins = [x for x in ins if x not in params]
                mini = Program(f"{self.prog.name}.{node.output.name}", ins, params,
                               {p.name: self.prog.param_data[p.name] for p in params}, [node],
                               list(node.outputs), dict(self.prog.symbols))
                table = DimTable(self.table.symbols)
                plan = build_plan(mini, opts, table, f"{self.path}/isolated %{node.output.name}", count=False)
                sub = Runtime(table, rt.values, "off", opts)
                args = [_to_working(local[id(x)], opts) for x in ins]
                refs = [_ref_form(a, kind_of(x.type.dtype)) for a, x in zip(args, ins)]
                want = iref.IMPLS[node.op](node.attrs, [None if x is None else (
                    self.prog.param_data[x.name] if x in params else refs[ins.index(x)]) for x in node.inputs],
                    tuple(local[id(node.output)].shape))
                got = plan.run(args, sub)[0]
            except KernelMismatchError:
                return node, "Compiled alone on the reference inputs, this op fails its kernel check."
            except Exception as e:  # noqa: BLE001
                return None, f"The ops could not be isolated ({type(e).__name__}: {e})."
            kind = kind_of(node.output.type.dtype)
            c = _compare(node.output.name, got, want, 0.0 if kind != "f" else check.tol)
            if not c.ok:
                return node, (f"Compiled alone on the reference inputs, this op is wrong by itself "
                              f"({_why(c)}).")
        return None, ("Every op of the region matches when compiled alone, so the fault is in how they were "
                      "fused.")


# ----------------------------------------------------------------------------- building plans


def _conv_tolerance(node: Node, opts: Options) -> float:
    base = kverify.BASE_TOL[opts.dtype]
    if opts.allow_tf32 and opts.dtype == "float32" and opts.device == "cuda":
        base = kverify.TF32_TOL
    length = 1
    if node.op == "conv2d":
        # a symbolic size (the batch, in the weight gradient's conv) counts as 1024, as in the
        # kernel tolerance (kernel/verify.py)
        w = node.inputs[1].type.shape
        for d in w[1:4]:
            length *= int(d) if isinstance(d, int) else 1024
    return base * max(1.0, math.log2(max(length, 1)) / 4.0)


class _Builder:
    """Builds the steps that are not kernels while ``Lowering`` walks the nodes."""

    library_ops = ("conv2d", "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d")

    def __init__(self, plan: Plan, L: Lowering):
        self.plan = plan
        self.L = L
        self.opts = plan.opts
        self.table = plan.table
        self.steps: List[Tuple[float, Step]] = []
        self.alias: set = set()

    def _name(self, node: Node) -> str:
        return f"%{node.output.name}"

    def _outs(self, node: Node, alias: bool = False) -> List[int]:
        slots = []
        for o in node.outputs:
            lv = self.L.leaf(o)
            slots.append(lv.slot)
            if alias:
                self.alias.add(lv.slot)
        return slots

    def _ins(self, node: Node, true_typed: bool = True) -> List[Optional[int]]:
        return [None if x is None else self.L.need(x, true_typed) for x in node.inputs]

    def _shapes(self, node: Node):
        return [self.table.shape(o.type.shape) for o in node.outputs]

    def layout(self, L, node):
        ins = self._ins(node)
        shape_idx = self._shapes(node)[0]
        static = tuple(int(d) for d in node.output.type.shape) \
            if all(isinstance(d, int) for d in node.output.type.shape) else None
        pos = L.tick()
        out = self._outs(node, alias=True)[0]
        self.steps.append((pos, ImplStep(self._name(node), node, ins, out, shape_idx, static)))

    def create(self, L, node):
        ins = self._ins(node)
        dtype = node.attrs.get("dtype") or node.output.type.dtype
        if dtype.is_floating_point:
            # mixed=True: the working precision, or its float32 island; otherwise the working precision
            dtype = getattr(torch, L.fdtype_of(dtype)) if self.opts.mixed else self.opts.torch_dtype
        device = "cpu" if (not dtype.is_floating_point and dtype != torch.bool
                           and node.output.type.rank == 0) else self.opts.torch_device
        shape_idx = self._shapes(node)[0]
        pos = L.tick()
        out = self._outs(node)[0]
        self.steps.append((pos, CreateStep(self._name(node), node, ins, out, shape_idx, dtype, device)))

    def host(self, L, node):
        if node.op == "dim_scalar":
            dtype = node.attrs["dtype"]
            is_float = dtype.is_floating_point
            index = self.table.add(node.attrs["expr"])
            pos = L.tick()
            self.steps.append((pos, DimScalarStep(self._name(node), node, self._outs(node)[0], index,
                                                  self.opts.torch_dtype if is_float else dtype,
                                                  self.opts.torch_device if is_float else "cpu")))
            return
        ins = self._ins(node)
        pos = L.tick()
        self.steps.append((pos, HostStep(self._name(node), node, ins, self._outs(node), self._shapes(node),
                                         self.opts)))

    def fallback(self, L, node, reason):
        if node.op not in iref.IMPLS:
            raise IRError(f"{node.op} cannot be compiled ({reason}) and has no reference implementation")
        ins = self._ins(node)
        pos = L.tick()
        self.steps.append((pos, FallbackStep(self._name(node), node, ins, self._outs(node), self._shapes(node),
                                             self.opts, reason)))

    def library(self, L, node, reason=""):
        ins = self._ins(node)
        pos = L.tick()
        out = self._outs(node)[0]
        step = LibraryStep(self._name(node), node, ins, out, self.opts, reason)
        ext = [(x, "f", s, None) for x, s in dict(zip([x for x in node.inputs if x is not None],
                                                       [s for s in ins if s is not None])).items()]
        step.check = GroupCheck([node], ext, [(node.output, "f", out)], [self._shapes(node)],
                                _conv_tolerance(node, self.opts))
        self.steps.append((pos, step))

    def random(self, L, node):
        ins = self._ins(node)
        pos = L.tick()
        out = self._outs(node)[0]
        shapes = self._shapes(node)
        step = RandomStep(self._name(node), node, ins, out, shapes[0], self.opts)
        if node.output.type.dtype.is_floating_point and self.opts.mixed:
            step.working = getattr(torch, L.fdtype_of(node.output.type.dtype))   # a float32 island keeps float32 draws
        kind = kind_of(node.output.type.dtype)
        declared = node.output.type.dtype
        # bit exact against the reference unless the working dtype is narrower than the declared one
        tol = 0.0 if (kind != "f" or step.working == declared) else 2.0 ** -8
        ext = [(x, "i", sl, None) for x, sl in zip(node.inputs, ins)]
        step.check = GroupCheck([node], ext, [(node.output, kind, out)], [shapes], tol)
        self.steps.append((pos, step))

    def index(self, L, node):
        ins = self._ins(node)
        pos = L.tick()
        outs = self._outs(node)
        step = IndexStep(self._name(node), node, ins, outs, self.opts.device)
        ext = [(x, kind_of(x.type.dtype), s, None)
               for x, s in dict(zip([x for x in node.inputs if x is not None],
                                    [s for s in ins if s is not None])).items()]
        tol = kverify.BASE_TOL[self.opts.dtype]
        if node.op == "kvq_attention":     # half precision math: the float16 tolerance, grown with N
            tol = max(tol, kverify.BASE_TOL["float16"] * 2)
        if node.op.startswith("linear_attention"):   # the T step chain, as truestar.linattn grows it
            from ..linattn.op import tolerance as la_tol
            q = node.inputs[0].type.shape
            T, dk = (d if isinstance(d, int) else 4096 for d in (q[1], q[3]))
            dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(
                self.opts.dtype, torch.float32)
            tol = max(tol, la_tol(dt, T, dk, 16) * (1 if node.op == "linear_attention" else 10))
        if node.op == "attention_grad":   # sums over T rows and D columns: the kernel layer's growth
            from .attention_bwd import tolerance_length
            tol = tol * max(1.0, math.log2(tolerance_length(node.inputs[0].type.shape)) / 4.0)
        step.check = GroupCheck([node], ext, [(o, kind_of(o.type.dtype), s) for o, s in zip(node.outputs, outs)],
                                [self._shapes(node)], tol)
        self.steps.append((pos, step))

    def int_device(self, L, node):
        ins = self._ins(node)
        pos = L.tick()
        out = self._outs(node)[0]
        step = IntStep(self._name(node), node, ins, out, self._shapes(node)[0], self.opts.device)
        ext = [(x, kind_of(x.type.dtype), s, None)
               for x, s in dict(zip([x for x in node.inputs if x is not None],
                                    [s for s in ins if s is not None])).items()]
        step.check = GroupCheck([node], ext, [(node.output, "i", out)], [self._shapes(node)], 0.0)
        self.steps.append((pos, step))

    def control(self, L, node):
        a = node.attrs
        name = self._name(node)
        base = f"{self.plan.path}/{node.op} {name}"
        if node.op == "cond":
            then = build_plan(a["then_branch"], self.opts, self.table, base + "/then_branch")
            other = build_plan(a["else_branch"], self.opts, self.table, base + "/else_branch")
            pred = L.need(node.inputs[0], true_typed=False)
            args = [L.need(x) for x in node.inputs[1:]]
            pos = L.tick()
            self.steps.append((pos, CondStep(name, node, pred, args, self._outs(node, alias=True), then, other)))
            return
        n = a["n_carried"]
        body = build_plan(a["body"], self.opts, self.table, base + "/body")
        cond = None if a.get("cond") is None else build_plan(a["cond"], self.opts, self.table, base + "/cond")
        trip = None if cond is not None else self.table.add(a["trip_count"])
        ins = [L.need(x) for x in node.inputs]
        pos = L.tick()
        self.steps.append((pos, LoopStep(name, node, ins[:n], ins[n:], self._outs(node, alias=True), body, cond,
                                         trip, a.get("max_iterations"))))


def build_plan(prog: Program, opts: Options, table: DimTable, path: str, count: bool = True) -> Plan:
    """Lower ``prog`` (a program or a region) and compile its kernels."""
    from .coverage_rules import lower_composites
    prog = lower_composites(prog)   # group, RMS and vector norms as their verified primitive ops
    plan = Plan(prog, path, opts, table)
    L = plan.lowering = Lowering(prog, opts, table, path)
    B = _Builder(plan, L)
    L.lower_nodes(B)
    specs = L.fuse()
    steps = list(B.steps)
    carried_tol: Dict[int, float] = {}
    for spec in specs:
        steps.append((spec.root.pos, _kernel_step(plan, L, spec, carried_tol, count)))
    for lv in L.adapters:
        dtype = torch.float32 if lv.kind == "i" else opts.torch_dtype
        steps.append((lv.pos + 0.5, AdapterStep(f"%{lv.name}", lv.slot, lv.kslot, dtype, opts.torch_device)))
    steps.sort(key=lambda t: t[0])
    plan.steps = [s for _, s in steps]
    plan.template = [None] * L.nslots
    from .lower import kn_weights, store_kn
    plan.kn_params = kn_weights(prog, opts)    # linear weights stored [K, N] (prepacked)
    for p in prog.params:
        lv = L.vals[id(p)]
        data = prog.param_data[p.name].detach()
        if lv.kind == "f" and lv.fdtype and lv.fdtype != opts.dtype:   # an island param
            plan.template[lv.slot] = kverify.to_device(data, opts.torch_device, getattr(torch, lv.fdtype))
        else:
            plan.template[lv.slot] = _to_working(data, opts)
            if p.name in plan.kn_params:
                plan.template[lv.slot] = store_kn(plan.template[lv.slot])
    plan.islands = [(lv.name or (lv.value.name if lv.value is not None else f"v{lv.id}"), lv.fdtype,
                     tuple(lv.shape or ())) for lv in L.islands()]
    for lv, data in L.derived_params:
        plan.template[lv.slot] = kverify.to_device(saturate(data, opts.torch_dtype), opts.torch_device, opts.torch_dtype)
    plan.in_slots = [L.vals[id(v)].slot for v in prog.inputs]
    alias = B.alias | set(plan.in_slots) | {L.vals[id(p)].slot for p in prog.params}
    plan.out_slots = [L.vals[id(v)].slot for v in prog.outputs]
    plan.alias_out = [s in alias for s in plan.out_slots]
    from .memory import schedule
    plan.set_schedule(schedule(plan, opts.memory_reuse))
    if count:
        STATS["regions"] += 1
    return plan


def _kernel_step(plan: Plan, L: Lowering, spec: KernelSpec, carried_tol: Dict[int, float], count: bool) -> Step:
    opts = plan.opts
    armed = opts.checks if count else frozenset()
    # cuBLAS may run a float32 contraction kernel when measured faster (kernel/library.py); never in
    # deterministic mode, where a row's bits must not depend on the batch size
    ck = K.compile(spec.kernel, opts.device, allow_tf32=opts.allow_tf32, checks=armed,
                   deterministic=bool(opts.deterministic),
                   library=opts.cublas and not opts.deterministic,
                   portable_tensor_cores=opts.portable_tensor_cores, **opts.hidden)
    if count:
        STATS["kernels"] += 1
    root = spec.root
    in_slots = [L.kernel_slot(a) for a in spec.inputs]
    out = L.kernel_slot(root)
    true_slot = root.slot if (root.kind != "f" and root.need_true) else None
    step = KernelStep(spec.kernel.name, spec.nodes, ck, in_slots, out, true_slot, root.value is not None)
    step.out_dtype = getattr(torch, root.fdtype) if root.kind == "f" and root.fdtype else None
    step.scalar = len(root.shape) == 0
    step.out_sizes = tuple(root.shape)      # kernel sizes, for the memory planner
    step.scalar_ins = [i for i, a in enumerate(spec.inputs) if len(a.shape) == 0]
    step.spec = spec
    step.armed = frozenset(armed)
    tol = kverify.tolerance(spec.kernel, None, allow_tf32=opts.allow_tf32)
    tol += sum(carried_tol.get(a.id, 0.0) for a in spec.inputs)
    if root.value is None:
        carried_tol[root.id] = tol
        return step
    produced = {id(o) for n in spec.nodes for o in n.outputs}
    ext, seen = [], set()
    for n in spec.nodes:
        for x in n.inputs:
            if x is None or id(x) in produced or id(x) in seen:
                continue
            seen.add(id(x))
            lv = L.vals[id(x)]
            kind = kind_of(x.type.dtype)
            if lv.const is not None and L.is_param(x):
                data = plan.prog.param_data[x.name]
                # the kernel inlines the constant saturated to the working dtype (lower_nodes), as every
                # other param is in the plan template; the reference reads the same number (a mask
                # value of torch.finfo(float64).min, T5's decoder, was compared with 1.8e308)
                ext.append((x, kind, None, saturate(data.double(), plan.opts.torch_dtype) if kind == "f" else data))
            else:
                ext.append((x, kind, lv.slot if lv.slot is not None else lv.kslot, None))
    shape_idx = [[plan.table.shape(o.type.shape) for o in n.outputs] for n in spec.nodes]
    step.check = GroupCheck(spec.nodes, ext, [(root.value, root.kind, out)], shape_idx, tol)
    return step


# ----------------------------------------------------------------------------- whole programs


def verify_mode(override: Optional[str] = None) -> str:
    if override is not None:
        v = str(override).lower()
        return "off" if v in ("0", "off", "false", "no") else ("all" if v in ("all", "always", "2") else "first")
    return kverify.verify_mode()


def torch_allows_tf32() -> bool:
    """True when PyTorch would run a float32 matmul on CUDA with TF32 right now.

    Reads ``torch.backends.cuda.matmul.fp32_precision`` (torch 2.9 and later), which every way of
    setting the precision updates: ``allow_tf32``, ``fp32_precision`` (matmul or the generic
    ``torch.backends.fp32_precision``) and ``torch.set_float32_matmul_precision`` ("high" and
    "medium" both mean TF32 on CUDA). ``allow_tf32`` itself raises once the two APIs were mixed, so
    it is only the fallback for older torch."""
    getter = getattr(torch._C, "_get_fp32_precision_getter", None)
    if getter is not None:
        try:
            return getter("cuda", "matmul") == "tf32"
        except RuntimeError:
            pass
    try:
        return bool(torch.backends.cuda.matmul.allow_tf32)
    except RuntimeError:
        return False


# per precision variant state of a CompiledProgram that follows PyTorch's TF32 setting
_VARIANT_FIELDS = ("opts", "plan", "_bound", "_clone", "verified", "report", "max_error",
                   "max_region_error", "compile_seconds", "kernel_cache", "recompute_report",
                   "memory_plan_at_compile", "checks")   # checks: the sentinels of the variant


class CompiledProgram:
    """A program compiled for one device and one float precision. Call it with the input tensors.

    With ``opts.allow_tf32`` None the program follows PyTorch's float32 matmul precision
    (torch_allows_tf32) at every call, like eager PyTorch: a call made under the other setting runs,
    and on its first use compiles and verifies, the variant built for it. Each variant is checked
    against float64 with its own tolerance. ``follows_tf32`` says which mode is on.

    With ``opts.deterministic`` None the program follows ``torch.are_deterministic_algorithms_enabled()``
    the same way: deterministic variants never split K, so a contraction's result does not depend on
    the batch size (``follows_deterministic``)."""

    def __init__(self, prog: Program, opts: Options, verify: Optional[str] = None,
                 allow_fallback: bool = True, memory_budget=None, memory_dims: Optional[Dict[str, int]] = None,
                 recompute=False):
        from . import memory as M
        self.memory_budget = None if memory_budget is None else M.parse_bytes(memory_budget)
        self.memory_dims = None if memory_dims is None else {str(k): int(v) for k, v in memory_dims.items()}
        # recompute: False, True or "greedy" (recompute.py), "mincut", "offload", "mincut+offload" (remat.py)
        if recompute is True:
            recompute = "greedy"
        if recompute not in (False, None, "greedy", "mincut", "offload", "mincut+offload"):
            raise ValueError(f"recompute={recompute!r}: True (greedy), 'greedy', 'mincut', 'offload' or "
                             "'mincut+offload'")
        self.recompute = recompute or False
        if recompute and self.memory_budget is None:
            raise ValueError("recompute=True needs a memory_budget: recomputation recomputes values only "
                             "until the planned peak fits the budget")
        from ..ir.custom_vjp import inline_custom_vjp
        prog = inline_custom_vjp(prog)  # a user gradient only matters to autodiff; run the primal
        self.program = prog
        self.verify = verify
        self.allow_fallback = allow_fallback
        self.table = DimTable(list(prog.symbols))
        self._kinds = [kind_of(v.type.dtype) for v in prog.inputs]
        from .lower import input_storage_dtypes
        self._in_dtypes = [d if k == "f" else None for d, k in zip(input_storage_dtypes(prog, opts), self._kinds)]
        self._end_to_end_seconds = 0.0
        self.calls = 0
        self.derived_unavailable = False  # the derivation could not run: the fixed bar was the verdict
        self.derived_accepts: List[Dict[str, Any]] = []   # outputs over the fixed bar, within the derived bound
        self.end_to_end = "auto"  # "derived" / "bar" / "auto": how the end to end check judges (compile_program)
        self._writes_state = False  # this call passes update= or donate=True (end_to_end="auto" is strict)
        # automatic CUDA graph replay (docs/COMPILER.md, Automatic graphs): from the second ordinary
        # call at a signature, the call is replayed from a graph recorded after a verified call and
        # checked bitwise against it; compile_program(graphs="off") or TRUESTAR_GRAPHS=off opts out
        self.auto_graphs = os.environ.get("TRUESTAR_GRAPHS", "auto").strip().lower() not in ("off", "0", "no", "false")
        self._autos: Dict[tuple, Any] = {}     # update spec -> GraphedProgram
        self._auto_seen: Dict[tuple, int] = {}
        self._auto_off: Dict[tuple, str] = {}
        self._in_auto = False
        self.follows_tf32 = opts.allow_tf32 is None
        self.follows_deterministic = opts.deterministic is None
        self._tf32_matters = opts.device == "cuda" and opts.dtype == "float32"
        # a pinned precision must give the same bits whatever PyTorch says, so only a program that
        # follows PyTorch's setting may use cuBLAS (which always runs in PyTorch's current precision)
        opts = replace(opts, cublas=self.follows_tf32, **self._wanted(opts))
        self._variants: Dict[Tuple[bool, bool], Dict[str, Any]] = {}
        self.compile_cache: Dict[str, Any] = {"verify": "not run"}   # what the first call took from truestar.compile_cache
        self._build(opts)

    def _build(self, opts: Options) -> None:
        t0 = time.perf_counter()
        before = (K.stats.memory_hits, K.stats.disk_hits, K.stats.misses)
        prog = self.program
        self.opts = opts
        self.plan = build_plan(prog, opts, self.table, prog.name)
        self._bound: Dict[Tuple, Runtime] = {}
        self._clone = any(self.plan.alias_out)
        self.verified = False
        self.max_error = 0.0            # end to end, against the float64 reference, on the first inputs
        self.mixed_bounds: List[Any] = []   # mixed=True: per output declared precision bars (_mixed_bars)
        self.max_region_error = 0.0     # largest error of one region against its own reference
        self.end_to_end_device = None   # where the last end to end reference ran
        # outputs accepted end to end with accept_ill_conditioned=True (TS782, docs/COMPILER.md
        # Verification): an independent float32 evaluation is as far from float64
        self.conditioning: List[Dict[str, Any]] = []
        self.compile_seconds = time.perf_counter() - t0
        after = (K.stats.memory_hits, K.stats.disk_hits, K.stats.misses)
        self.kernel_cache = {"memory_hits": after[0] - before[0], "disk_hits": after[1] - before[1],
                             "misses": after[2] - before[2]}
        STATS["programs"] += 1
        self.checks = None
        if opts.checks:
            from .checks import ProgramChecks, arm_inf
            if "inf" in opts.checks:
                arm_inf(self.plan, opts.checks, opts)
            self.checks = ProgramChecks(self, opts.checks)
        from .report import CompileReport
        self.report = CompileReport.from_program(self)
        if not self.allow_fallback and self.report.fallbacks:
            raise FallbackNotAllowed(
                f"{prog.name} needs the fallback path for: "
                + "; ".join(f"{f['ops'][0]} {f['name']} ({f['reason']})" for f in self.report.fallbacks))
        # memory, per variant: the budget is checked now when the symbol values to plan for are known
        # (memory_dims, else the largest declared value of every symbol), else at the first call with a
        # new shape; either way before anything of this variant is launched
        from . import memory as M
        self.recompute_report: List[Dict[str, Any]] = []
        self.memory_plan_at_compile = None
        if self.memory_budget is not None:
            dims = self.memory_dims
            if dims is None:
                try:
                    dims = M.max_dims(self)
                except ValueError:
                    dims = None       # an unbounded symbol: checked at the first call of each new shape
            if dims is not None:
                if self.recompute == "greedy":
                    from .recompute import apply_recompute
                    self.recompute_report = apply_recompute(self, self.memory_budget, dims)
                elif self.recompute:
                    from .remat import apply_remat
                    self.recompute_report = apply_remat(self, self.memory_budget, dims, self.recompute)
                if self.recompute:
                    self.report = CompileReport.from_program(self)
                self.memory_plan_at_compile = M.check_budget(self, self.memory_budget, dims)
            elif self.recompute:
                raise ValueError(f"recompute=True needs the symbol values to plan for: pass memory_dims= "
                                 f"(symbols {', '.join(self.table.symbols)} have no upper bound)")

    def memory_plan(self, dims_or_inputs=None, donate: bool = False, rounding: bool = True):
        """The planned device memory of one call (compiler.memory.MemoryPlan), before any launch.
        ``dims_or_inputs``: symbol values ({"B": 64}) or the input tensors; None plans at the largest
        declared value of every symbol."""
        from .memory import plan_memory
        return plan_memory(self, dims_or_inputs, donate, rounding)

    def _wanted(self, opts: Options) -> Dict[str, bool]:
        """The settings PyTorch asks for right now, for the options that follow it."""
        out = {}
        if self.follows_tf32:
            out["allow_tf32"] = self._tf32_matters and torch_allows_tf32()
        if self.follows_deterministic:
            out["deterministic"] = torch.are_deterministic_algorithms_enabled()
        return out

    def _switch(self, want: Dict[str, bool]) -> None:
        """Make the variant for these settings current, compiling it on first use."""
        key = lambda o: (bool(o.allow_tf32), bool(o.deterministic))  # noqa: E731
        if self.checks is not None and self.checks.pending:
            self._poll_checks(True)       # deferred sentinel reads of this variant, before leaving it
        self._variants[key(self.opts)] = {f: getattr(self, f) for f in _VARIANT_FIELDS}
        opts = replace(self.opts, **want)
        saved = self._variants.get(key(opts))
        if saved is None:
            self._build(opts)
        else:
            for f, v in saved.items():
                setattr(self, f, v)

    # ---- calls
    def _bind(self, inputs, key) -> Runtime:
        bindings = iref.bind_symbols(self.program, inputs)
        values = tuple(int(bindings[symbol(s)]) for s in self.table.symbols)
        if self.memory_budget is not None:       # before the first launch at this shape
            from .memory import check_budget
            check_budget(self, self.memory_budget, dict(zip(self.table.symbols, values)))
        rt = Runtime(self.table, values, "off", self.opts)
        self._bound[key] = rt
        return rt

    binds_symbols = True   # the front end may skip its own per call shape check (see bind)

    AUTO_GRAPHS_MAX = 8      # graphs kept per program; further signatures run as ordinary calls

    def _auto_call(self, inputs, update=None) -> Optional[List[torch.Tensor]]:
        """Outputs replayed from this signature's CUDA graph, or None for an ordinary call: the first
        call at a signature, a signature whose recording was refused (a step that syncs with the host,
        such as a random op reading its key, a replay that was not bitwise equal: TS800, TS801), a
        program that checks every call (TRUESTAR_VERIFY=all), one compiled with ``memory_budget`` or
        ``recompute`` or with ``sentinel_read="deferred"``, an input that is not a CUDA tensor, or
        more than AUTO_GRAPHS_MAX signatures. ``auto_graph_refusals`` says why a signature was not.

        With ``update`` (a training step updating its parameters and optimizer state in place) the
        updated inputs are bound by address, as ``graphed(update=)`` binds them: a training loop passes
        the same state tensors every step. The recording is checked bitwise against the ordinary call,
        the updated inputs included (TS801); a later call with other tensors for them gets TS802 from
        the graph before anything is written, and that signature runs as ordinary calls from then on."""
        from .graphs import BucketError, CaptureError, CaptureMismatchError, GraphedProgram, HostSyncError
        if verify_mode(self.verify) == "all":
            return None
        # a graph keeps its own memory pool for the call's working set: a program compiled under a
        # memory budget or with recompute stays on its planned memory; a deferred sentinel read is
        # the ordinary call's own schedule
        if self.memory_budget is not None or self.recompute or self.opts.sentinel_read == "deferred":
            return None
        if not all(isinstance(t, torch.Tensor) and t.device.type == "cuda" for t in inputs):
            return None
        want = self._wanted(self.opts) if (self.follows_tf32 or self.follows_deterministic) else {}
        spec = tuple(sorted((int(i), int(j)) for i, j in dict(update or {}).items()))
        key = (tuple(sorted(want.items())), tuple((tuple(t.shape), t.dtype, t.device) for t in inputs), spec)
        if key in self._auto_off:
            return None
        seen = self._auto_seen.get(key, 0)
        self._auto_seen[key] = seen + 1
        if seen == 0:
            return None                  # the first call is the ordinary verified call
        gp = self._autos.get(spec)
        if gp is None:
            gp = self._autos[spec] = GraphedProgram(self, update=dict(spec) or None)
        held = sum(len(g._entries) for g in self._autos.values())
        if held >= self.AUTO_GRAPHS_MAX and not any(k[2] == key[1] for k in gp._entries):
            self._auto_off[key] = f"more than {self.AUTO_GRAPHS_MAX} signatures"
            return None
        self._in_auto = True
        try:
            return gp(list(inputs))
        except (CaptureError, CaptureMismatchError, HostSyncError, BucketError) as e:
            self._auto_off[key] = str(e).splitlines()[0][:200]
            return None
        finally:
            self._in_auto = False

    @property
    def auto_graph_refusals(self) -> Dict[tuple, str]:
        return dict(self._auto_off)

    @property
    def auto_graph_replays(self) -> int:
        return sum(g.replays for g in self._autos.values())

    @property
    def _auto(self):
        """The automatic graphs of calls without ``update`` (None before the first one)."""
        return self._autos.get(())

    def bind(self, inputs: Sequence[torch.Tensor]) -> None:
        """Check input shapes against the declared types. Raises ShapeError; costs a dict lookup
        once a shape has been seen."""
        key = tuple([t.shape for t in inputs])
        if key not in self._bound:
            self._bind(inputs, key)

    def __call__(self, inputs: Sequence[torch.Tensor], donate: bool = False,
                 update: Optional[Dict[int, int]] = None) -> List[torch.Tensor]:
        """Run the program. ``donate=True`` hands the input tensors over: they are taken out of the
        ``inputs`` list, and a kernel may write its result into a dead input's buffer. An input the
        caller still references elsewhere is detected (reference counts) and never written over.

        ``update={i: j}``: input ``i`` is updated in place to hold output ``j`` (a KV cache written
        at new positions), and output ``j`` comes back as that same tensor. When the step producing
        output ``j`` can write into input ``i`` (``index_copy`` of it, or an elementwise kernel, and
        nothing reads input ``i`` afterwards) it does, else output ``j`` is copied into it at the end;
        ``last_update_inplace`` says which. Input ``i`` must be a contiguous tensor on the executor's
        device in its precision. The first call's checks compare against the input as it was."""
        if (self.auto_graphs and not self._in_auto and not donate
                and self.opts.device == "cuda"):
            outs = self._auto_call(inputs, update)
            if outs is not None:
                return outs
        upd = self._update_spec(inputs, update, donate)
        self._writes_state = bool(upd) or bool(donate)
        if self.follows_tf32 or self.follows_deterministic:
            want = self._wanted(self.opts)
            if any(getattr(self.opts, k) != v for k, v in want.items()):
                self._switch(want)
        key = tuple([t.shape for t in inputs])
        rt = self._bound.get(key)
        if rt is None:
            rt = self._bind(inputs, key)
        opts = self.opts
        dev, dt = opts.torch_device, opts.torch_dtype
        args = [kverify.to_device(t, dev, idt) if k == "f" else t.to(dev)
                for t, k, idt in zip(inputs, self._kinds, self._in_dtypes)]
        mode = verify_mode(self.verify)
        rt.mode = mode
        self.calls += 1
        rt.pending, rt.checked = [], {}
        verifying = not (mode == "off" or (mode == "first" and self.verified))
        rt.update = frozenset(self.plan.in_slots[i] for i in upd)
        targets = {i: inputs[i] for i in upd}
        if upd and verifying:      # the end to end check reads the inputs as they were before the writes
            inputs = [t.detach().clone() if i in upd else t for i, t in enumerate(inputs)]
        if donate:
            if not isinstance(inputs, list):
                raise TypeError("donate=True needs the inputs as a list: the executor takes the tensors out of it")
            refs = [t.detach().clone() for t in inputs] if verifying else None   # for the end to end check
            inputs.clear()
            inputs = refs
        chk = self.checks
        if chk is not None:
            if chk.pending:
                self._poll_checks(False)
            rt.chk = chk
            chk.begin(rt, args)
            if donate:          # a locate run needs the inputs again: keep this call's own references
                args = list(args)
        try:
            if not verifying:
                if chk is not None and self.opts.sentinel_read == "deferred":
                    outs = self.plan.run(list(args), rt, own=donate)
                    chk.defer(args, self.calls)
                else:
                    outs = self.plan.run(list(args) if chk is not None else args, rt, own=donate)
                    self._read_checks(args, rt)
                rt.raise_pending()
            else:
                outs = self._first_call(inputs, args, rt, donate, upd, mode)
        finally:
            rt.chk = None
            rt.update = frozenset()
        if self._clone:
            outs = [o if (a and j in upd.values()) else (o.clone() if a else o)
                    for j, (o, a) in enumerate(zip(outs, self.plan.alias_out))]
        if upd:
            inplace = True
            for i, j in upd.items():
                t, o = targets[i], outs[j]
                if tuple(o.shape) != tuple(t.shape):
                    raise ValueError(f"update: output {j} has shape {tuple(o.shape)}, input {i} {tuple(t.shape)}")
                if o.data_ptr() != t.data_ptr():
                    t.copy_(o)
                    inplace = False
                outs[j] = t
            self.last_update_inplace = inplace
        return outs

    def graphed(self, static: Sequence[int] = (), update: Optional[Dict[int, int]] = None, outputs: str = "copy"):
        """This program replayed from CUDA graphs, one per input signature, each recorded after the
        ordinary verified first call at that signature (compiler.graphs.GraphedProgram)."""
        from .graphs import GraphedProgram
        return GraphedProgram(self, static, update, outputs)

    def _update_spec(self, inputs, update, donate) -> Dict[int, int]:
        if not update:
            return {}
        if donate:
            raise ValueError("update= and donate=True cannot be combined: update names the inputs written in place")
        upd = {int(i): int(j) for i, j in dict(update).items()}
        outs = self.program.outputs
        if len(set(upd.values())) != len(upd):
            raise ValueError(f"update={update}: two inputs cannot hold the same output")
        for i, j in upd.items():
            if not 0 <= i < len(inputs) or not 0 <= j < len(outs):
                raise ValueError(f"update={update}: the program has {len(inputs)} inputs and {len(outs)} outputs")
            t = inputs[i]
            want = tuple(int(d) for d in t.shape) if isinstance(t, torch.Tensor) else None
            dev = self.opts.torch_device
            if (not isinstance(t, torch.Tensor) or t.device.type != dev.type
                    or (dev.index is not None and t.device.index != dev.index)
                    or (self._kinds[i] == "f" and t.dtype != self._in_dtypes[i]) or not t.is_contiguous()):
                raise TypeError(f"update={update}: input {i} must be a contiguous tensor on {self.opts.torch_device} "
                                f"in {self.opts.dtype}, got {getattr(t, 'dtype', type(t))} on {getattr(t, 'device', '-')}")
            if outs[j].type.dtype != self.program.inputs[i].type.dtype or want is None:
                raise TypeError(f"update={update}: output {j} and input {i} have different dtypes")
        return upd

    def _end_to_end_device(self, outs, rt) -> Optional[torch.device]:
        """The end to end reference runs on the GPU when a region check of this call did (the
        program is big enough to gain), or for every CUDA program with TRUESTAR_VERIFY_DEVICE=gpu."""
        if self.opts.device != "cuda":
            return None
        like = next((o for o in outs if isinstance(o, torch.Tensor) and o.device.type == "cuda"), None)
        if like is None:
            return None
        mode = kverify.verify_device()
        if mode == "gpu" or (mode == "auto" and rt.stats["gpu_checks"] > 0):
            return like.device
        return None

    def _read_checks(self, args, rt) -> None:
        """Compiled in checks: one host read of every flag; on a hit, the locate run raises
        SentinelError naming the op, its line and the first offending element."""
        chk = rt.chk
        if chk is None or not chk.fired():
            return
        self._locate(args, rt, "")

    def _poll_checks(self, block: bool) -> None:
        """sentinel_read="deferred": read the flags of earlier calls that have finished on the GPU (all of
        them with block) and raise for the first that fired."""
        chk = self.checks
        hit = chk.poll(block)
        if hit is None:
            return
        args, call = hit
        chk.pending.clear()
        key = tuple([t.shape for t in args])
        rt = self._bound.get(key) or self._bind(args, key)
        rt.chk = chk
        try:
            self._locate(args, rt, f" (deferred read: the flags of call {call} were read at call {self.calls})")
        finally:
            rt.chk = None

    def read_sentinels(self) -> None:
        """With sentinel_read="deferred": wait for every pending call and raise SentinelError if one fired."""
        if self.checks is not None:
            self._poll_checks(True)

    def _locate(self, args, rt, note: str) -> None:
        chk = rt.chk
        chk.note = note
        rt.pending, rt.checked = [], {}
        saved = rt.mode
        rt.mode = "off"
        env_verify = os.environ.get("TRUESTAR_VERIFY")
        os.environ["TRUESTAR_VERIFY"] = "0"   # the kernel layer's own check: off for the locate run
        try:
            chk.locate(self.plan, args, rt)
        finally:
            rt.mode = saved
            if env_verify is None:
                os.environ.pop("TRUESTAR_VERIFY", None)
            else:
                os.environ["TRUESTAR_VERIFY"] = env_verify
        if not chk.kernel_fired:
            return      # a conservative trigger of a torch step (a sum that overflowed): nothing is bad
        from .checks import SentinelError
        raise SentinelError(f"truestar check: a compiled in check fired in {self.program.name!r} but did not fire "
                         f"again when the call was repeated step by step (a nondeterministic value?). The results "
                         f"of this call were not returned.", check="nan", program=self.program.name)

    def _first_call(self, inputs, args, rt, own: bool, upd: Dict[int, int], mode: str) -> List[torch.Tensor]:
        """A verified call, through the persistent compile cache (truestar.compile_cache): a program
        record made by a verified call on these same inputs is accepted when this call, run without
        the checks, launches the recorded kernel binaries and returns the recorded output bits;
        otherwise (or with no record) the call is verified and its result recorded."""
        from .. import compile_cache as CC
        key = None
        if mode == "first" and not own and rt.chk is None and not self.memory_budget:
            try:
                key = CC.verify_key(self, inputs, upd, mode)
            except Exception:   # noqa: BLE001 (a program the serializer refuses is not cached)
                key = None
        self.compile_cache = {"verify": "off" if key is None else "miss", "key": key}
        if key is not None:
            rec = CC.load(key)
            if rec is not None:
                outs = self._replay_verified(rec, inputs, args, rt)
                if outs is not None:
                    return outs
                opts = self.opts           # rerun from the inputs as they were before the replay wrote any
                args = [kverify.to_device(t, opts.torch_device, idt) if k == "f" else t.to(opts.torch_device)
                        for t, k, idt in zip(inputs, self._kinds, self._in_dtypes)]
            CC.stats.misses += 1
        with K.collect_checks() as col:
            outs = self._verified_run(inputs, args, rt, own=own)
        if key is not None:
            try:
                CC.store(key, {"program": self.program.name, "outputs": CC.tensor_digest(outs),
                               "kernels": sorted(col.seen), "max_error": self.max_error,
                               "max_region_error": self.max_region_error,
                               "end_to_end_device": self.end_to_end_device, "conditioning": self.conditioning})
                self.compile_cache["verify"] = "stored"
            except OSError as e:     # a read only or full disk: the call itself succeeded
                import warnings
                warnings.warn(f"[TS1353] truestar compile cache: could not write the program record ({e})",
                              CC.CompileCacheWarning, stacklevel=4)
        return outs

    def _replay_verified(self, rec, inputs, args, rt) -> Optional[List[torch.Tensor]]:
        """Run without checks; accept the record only for the same kernel binaries and output bits."""
        from .. import compile_cache as CC
        rt.mode = "off"
        try:
            with K.collect_checks(defer=True) as col:
                outs = self.plan.run(list(args), rt)
            rt.raise_pending()
        finally:
            rt.mode = "first"
        if sorted(col.seen) != rec.get("kernels") or CC.tensor_digest(outs) != rec.get("outputs"):
            CC.stats.mismatched += 1
            self.compile_cache.update(verify="mismatch", code="TS1354")
            return None
        col.accept()
        self.plan.checked = True
        self.verified = True
        self.max_error = max(self.max_error, float(rec.get("max_error", 0.0)))
        self.max_region_error = max(self.max_region_error, float(rec.get("max_region_error", 0.0)))
        self.end_to_end_device = rec.get("end_to_end_device")
        cond = list(rec.get("conditioning") or [])
        if cond:
            self.conditioning.extend(cond)
            import warnings
            warnings.warn(IllConditionedWarning(
                f"[TS782] compiled program {self.program.name!r}: {len(cond)} output(s) over the float64 bar "
                f"accepted as ill conditioned (accept_ill_conditioned=True, from the compile cache record): "
                + "; ".join(f"%{a['output']} error {a['error']:.3g}, bar {a['tolerance']:.3g}" for a in cond)),
                stacklevel=4)
        CC.stats.hits += 1
        self.compile_cache.update(verify="hit", code="TS1350")
        return outs

    def _verified_run(self, inputs, args, rt, own: bool = False) -> List[torch.Tensor]:
        rt.decisions = {id(n.outputs[0]): None for n in self.program.nodes
                        if n.op in _DECISION_OPS and any(x is not None and x.type.dtype.is_floating_point
                                                         for x in n.inputs)} or None
        try:
            return self._verified_run_body(inputs, args, rt, own)
        finally:
            rt.decisions = None

    def _verified_run_body(self, inputs, args, rt, own: bool) -> List[torch.Tensor]:
        rt.stats["checks"], rt.stats["max_error"], rt.stats["gpu_checks"] = 0, 0.0, 0
        try:
            outs = self.plan.run(list(args) if rt.chk is not None else args, rt, own=own)
        except ProgramMismatchError:
            # a NaN or an overflow of the working precision also differs from the float64
            # reference: when a check fired, name it (the locate run) instead of a compiler bug
            if rt.chk is not None:
                self._read_checks(args, rt)
            raise
        self._read_checks(args, rt)
        rt.raise_pending()
        t0 = time.perf_counter()
        dev = self._end_to_end_device(outs, rt)
        refs = None
        if dev is not None:
            try:
                refs = iref.run(self._reference_program(), *[t.detach() for t in inputs], device=dev)
            except torch.cuda.OutOfMemoryError:     # the GPU is shared: the CPU runs this check
                refs, dev = None, None
                torch.cuda.empty_cache()
        if refs is None:
            refs = iref.run(self._reference_program(), *[t.detach().cpu() for t in inputs])
        self.end_to_end_device = "cpu" if dev is None else str(dev)
        tol = self.end_to_end_tolerance()
        worst = 0.0
        f32ref = None
        derived = None                      # the derived verdict, computed at the first output over the bar
        # strict (every float output by its derived bound): end_to_end="derived", or the default "auto" on
        # a call that writes the caller's state in place (update= or donate=True: a training step), whose
        # error would carry into every later step
        strict = self.end_to_end == "derived" or (self.end_to_end == "auto" and self._writes_state)
        accepted: List[Dict[str, Any]] = []
        mixed_bars = self._mixed_bars(inputs, refs) if self.opts.mixed else None
        mixed_flips = None
        for i, (v, o, r) in enumerate(zip(self.program.outputs, outs, refs)):
            kind = kind_of(v.type.dtype)
            otol = tol if mixed_bars is None or mixed_bars[i] is None else mixed_bars[i]
            c = _compare(v.name, o, r, 0.0 if kind != "f" else otol)
            if mixed_bars is not None and mixed_bars[i] is not None:
                self.mixed_bounds[i]["error"] = c.error
                if (not c.ok and not math.isfinite(c.error) and o.dim() == 0
                        and not bool(torch.isfinite(o).all()) and not bool(torch.isfinite(r).all())):
                    # a scalar the declared precision overflows in both runs (the gradient norm of a
                    # float16 step the loss scaler skips): inf in one, NaN in the other is the same verdict
                    self.mixed_bounds[i]["overflow"] = True
                    continue
            if mixed_bars is not None and mixed_bars[i] is not None and not c.ok and math.isfinite(c.error):
                # A 16 bit value within rounding of a tie (a relu input near 0 whose float32 sum
                # rounds to the other sign) takes the other branch than float64, and everything
                # downstream changes. Accepted only when the float64 reference taking this run's
                # decisions matches EVERY output within its declared precision bar and every flipped
                # decision is on a near tie (TS1481, listed in near_ties).
                if mixed_flips is None:
                    flips, replay = self._branch_flips(inputs, outs, tol, rt.decisions, bars=mixed_bars)
                    ok = bool(flips) and replay is not None and math.isfinite(replay) and all(
                        f["max_margin"] <= tol for f in flips)
                    mixed_flips = (flips, ok)
                if mixed_flips[1]:
                    self.mixed_bounds[i]["near_tie"] = True
                    continue
                cause = ""
                if mixed_flips[0]:
                    cause = (f"\n  {sum(f['count'] for f in mixed_flips[0])} branch decision(s) differ from float64 "
                             f"({_describe_flips(mixed_flips[0])}), but the float64 reference taking them does not "
                             f"match every output within its bar, or a decision is not on a near tie.")
                raise ProgramMismatchError(
                    f"[TS1480] compiled mixed precision program {self.program.name!r} on {self.opts.device} "
                    f"({self.opts.dtype} with float32 islands) does not match the float64 reference "
                    f"interpreter end to end.\n  output %{v.name} (declared {str(v.type.dtype).replace('torch.', '')}): "
                    f"{_why(c)}. The bar is derived for the declared mixed precision: {MIXED_FACTOR:g}x the distance from "
                    f"float64 of an independent float32 evaluation that rounds every value to its declared dtype "
                    f"(the reference interpreter on emulate_precision(program)), and at least "
                    f"{self._mixed_floor():.3g}; that evaluation is {self.mixed_bounds[i]['emulated_error']:.3g} off "
                    f"on this output.\n  op: {self._producer(v)}. A value computed or stored in a narrower dtype "
                    f"than declared (a wrong cast, master weights or optimizer state rounded to 16 bits) or a "
                    f"missing or doubled loss scale lands here.{cause}\nThe result was not returned.",
                    program=self.program.name, region=self.program.name, op=self._producer(v), value=v.name,
                    error=c.error, tolerance=c.tol, stage="end_to_end")
            if strict and kind == "f":
                # strict: the derived bound decides for every float output, also one within the fixed
                # bar (a wrong weight decay sign is 2e-5 off, well within it, and 33 times its bound)
                if derived is None:
                    derived = self._derived_verdict(inputs, outs) or {}
                d = derived.get(i)
                if d is not None and not d.ok and self._cut_verdict(i, inputs, outs, derived, v, c.error):
                    worst = max(worst, c.error)
                    continue
                if d is not None:
                    if d.ok:
                        worst = max(worst, c.error)
                        continue
                    c.detail = (f"relative error {c.error:.3g}, beyond the bound derived from the program "
                                f"(docs/TOLERANCE.md): " + (d.detail or f"{d.bad} of {d.total} elements, worst "
                                                            f"{d.worst_ratio:.3g}x its bound"))
            e_err = math.nan
            if not c.ok and kind == "f" and math.isfinite(c.error):
                # Ill conditioned outputs (AdamW's first step divides by |g|; a relu mask of a value
                # within rounding of 0) are far from float64 in any correct execution in this
                # precision. Measure how far one is: the same IR program evaluated by the reference
                # interpreter in float32 (torch's float32 kernels, independent of the compiled
                # kernels, which never vouch for themselves). The default stays strict and only
                # reports the number; accept_ill_conditioned=True accepts within 4x of it.
                if f32ref is None:
                    f32ref = self._float32_reference(inputs, dev) or []
                if f32ref:
                    e_err = _compare(v.name, f32ref[i], r, tol).error
                if (self.opts.accept_ill_conditioned and math.isfinite(e_err) and e_err > 0
                        and c.error <= 4.0 * e_err):
                    accepted.append({"output": v.name, "error": c.error, "tolerance": c.tol,
                                     "float32_reference_error": e_err})
                    worst = max(worst, c.error)
                    continue
                # The fixed bar (BASE_TOL times 4 sqrt(kernels)) knows nothing about the program. Over
                # it, the output is judged by the bound derived from the program itself: how far a
                # correct execution in this precision can be on these inputs, op by op.
                if derived is None and not strict:
                    derived = self._derived_verdict(inputs, outs) or {}
                d = None if strict else derived.get(i)
                if d is not None and not d.ok and self._cut_verdict(i, inputs, outs, derived, v, c.error):
                    worst = max(worst, c.error)
                    continue
                if d is not None and d.ok:
                    self.derived_accepts.append({"output": v.name, "error": c.error, "fixed_bar": c.tol,
                                                 "worst_ratio": d.worst_ratio, "rms_ratio": d.rms_ratio})
                    worst = max(worst, c.error)
                    continue
                if d is not None:
                    c.detail = (_why(c) + "; and beyond the bound derived from the program (docs/TOLERANCE.md): "
                                + (d.detail or f"{d.bad} of {d.total} elements, worst {d.worst_ratio:.3g}x its bound"))
            if not c.ok:
                if math.isfinite(e_err):
                    cond = (f"\n  the reference interpreter in float32 (an independent float32 evaluation of "
                            f"the same program) is itself {e_err:.3g} off float64 on this output"
                            + (", so the output is ill conditioned in float32; "
                               "accept_ill_conditioned=True accepts outputs within 4x of that distance "
                               "(TS782, docs/COMPILER.md)" if e_err > c.tol else ", within the bar")
                            + ".")
                else:
                    cond = ""
                flips, replay = self._branch_flips(inputs, outs, tol, rt.decisions)
                # explained: the float64 reference with this run's decisions matches, and every element
                # decided the other way is a near tie: its operands are within the end to end bar of
                # each other, relative to their rms (a deviation the bar itself accepts for that value)
                explained = (bool(flips) and replay is not None and replay <= tol
                             and all(f["max_margin"] <= tol for f in flips))
                regions = (f"Every region matched its own reference on these inputs (largest region error "
                           f"{rt.stats['max_error']:.3g} over {rt.stats['checks']} checks)")
                if explained:
                    cause = (f"\n  ill conditioned in {self.opts.dtype} at a near tie: this run took "
                             f"{sum(f['count'] for f in flips)} branch decision(s) the other way than float64, "
                             f"each on operands within {tol:.3g} of their rms of each other (a near tie): "
                             f"{_describe_flips(flips)}.\n  The float64 "
                             f"reference taking this run's decisions matches every output (largest relative error "
                             f"{replay:.3g} <= tolerance {tol:.3g}), so the difference is the tie and not the "
                             f"compiled kernels: a correct {self.opts.dtype} evaluation can fall on either side of "
                             f"it. {regions}.")
                else:
                    cause = ""
                    if flips and replay is not None and replay > tol:
                        cause = (f"\n  {sum(f['count'] for f in flips)} branch decision(s) differ from float64 "
                                 f"({_describe_flips(flips)}), but the float64 reference taking them still differs "
                                 f"from this output (relative error {replay:.3g}), so they do not explain it.")
                    elif flips:
                        far = max(f["max_margin"] for f in flips)
                        cause = (f"\n  {sum(f['count'] for f in flips)} branch decision(s) differ from float64 "
                                 f"({_describe_flips(flips)}) and explain this output, but not all on a near tie: "
                                 f"operands up to {far:.3g} of their rms apart (bar {tol:.3g}), more than rounding "
                                 f"moves a value that passed its check.")
                    cause += (f"\n  op: {self._producer(v)}. {regions}, so either the error grows across regions "
                              f"in this precision (try float32, or a branch decided on a near tie) or values are "
                              f"wired wrongly between regions.")
                raise ProgramMismatchError(
                    f"compiled program {self.program.name!r} on {self.opts.device} ({self.opts.dtype}) does not "
                    f"match the float64 reference interpreter end to end.\n  region {self.program.name} (whole "
                    f"program), output %{v.name}: {_why(c)}{cond}{cause}\nThe result was not returned.",
                    program=self.program.name, region=self.program.name, op=self._producer(v), value=v.name,
                    error=c.error, tolerance=c.tol, stage="end_to_end", near_ties=flips if explained else ())
            worst = max(worst, c.error)
        if accepted:
            self.conditioning.extend(accepted)
            import warnings
            warnings.warn(IllConditionedWarning(
                f"[TS782] compiled program {self.program.name!r}: {len(accepted)} output(s) over the float64 bar "
                f"accepted as ill conditioned (accept_ill_conditioned=True): "
                + "; ".join(f"%{a['output']} error {a['error']:.3g}, bar {a['tolerance']:.3g}, float32 reference "
                            f"{a['float32_reference_error']:.3g}" for a in accepted)), stacklevel=3)
        self.max_error = max(self.max_error, worst)
        self.max_region_error = max(self.max_region_error, rt.stats["max_error"])
        self.verified = True
        del refs
        if dev is not None or rt.stats["gpu_checks"]:
            torch.cuda.empty_cache()                # give the checks' float64 memory back to other jobs
        self._end_to_end_seconds += time.perf_counter() - t0
        return outs

    @property
    def verify_seconds(self) -> float:
        """Time spent in program level checks so far: region checks plus end to end checks."""
        return self._end_to_end_seconds + sum(rt.stats["seconds"] for rt in self._bound.values())

    @property
    def region_verify_seconds(self) -> float:
        """Time spent in region checks so far (each kernel group and library step against the float64
        reference of the IR ops it covers)."""
        return sum(rt.stats["seconds"] for rt in self._bound.values())

    @property
    def end_to_end_verify_seconds(self) -> float:
        """Time spent comparing whole program outputs with the reference interpreter so far."""
        return self._end_to_end_seconds

    @property
    def kernel_verify_seconds(self) -> float:
        """Time the kernel layer spent checking each kernel against its own float64 reference."""
        return sum(s.ck.verify_seconds for _, s in self.plan.all_steps() if s.kind == "kernel")

    def _cut_verdict(self, i: int, inputs, outs, derived, v, error: float) -> bool:
        """Output i failed its bound derived from the program inputs. Is it right given the other
        outputs it is computed from? An AdamW step's new weight is p - lr m / (sqrt(v) + eps) of its
        new moments m and v, both outputs: where the gradient is within its own error of 0, a correct
        execution may flip the sign of the update (a bimodal outcome the error model runs, normal
        draws around float64, sample too rarely to bound), and m and v say which way it went.

        The cuts are the float outputs output i depends on that passed their own derived bounds. The
        part of the program between the program inputs and the cuts and output i is run again in
        float64 with the compiled values of the cuts in place of their float64 ones, and output i is
        judged by the bound derived for that part (the same rules). Sound by composition: the cuts are
        right (their bounds), and output i is what the program computes from them (this bound). A
        wrong optimizer formula is still refused (it is in the part judged here), and a wrong
        gradient is refused at the cut. No cut, a control flow op or a failure to derive: False."""
        if (self.opts.dtype == "float64" or self.opts.mixed      # mixed=True: its own declared precision bars
                or not _derivable(self.program.nodes)):
            return False
        prog = self.program
        ok_out = {id(prog.outputs[j]): j for j, d in derived.items()
                  if j != i and d is not None and d.ok and prog.outputs[j].type.dtype.is_floating_point}
        producer = {}
        for node in prog.nodes:
            for o in node.outputs:
                producer[id(o)] = node
        need, cuts, stack, seen = [], [], [prog.outputs[i]], set()
        # the target itself is never a cut: walk from its producer
        while stack:
            val = stack.pop()
            if id(val) in seen:
                continue
            seen.add(id(val))
            if val is not prog.outputs[i] and id(val) in ok_out:
                cuts.append(ok_out[id(val)])
                continue
            node = producer.get(id(val))
            if node is None:
                continue                     # a program input or param
            if node.extra_outputs or any(isinstance(a, Program) for a in node.attrs.values()):
                return False
            need.append(node)
            stack.extend(x for x in node.inputs if x is not None)
        if not cuts:
            return False
        order = {id(n): k for k, n in enumerate(prog.nodes)}
        nodes = sorted({id(n): n for n in need}.values(), key=lambda n: order[id(n)])
        cut_idx = sorted(set(cuts))
        cut_vals = [prog.outputs[j] for j in cut_idx]
        sub = Program(f"{prog.name}_cut_{v.name}", list(prog.inputs) + cut_vals, list(prog.params),
                      prog.param_data, nodes, [prog.outputs[i]], dict(prog.symbols))
        from .. import tolerance as TL
        dev = outs[i].device if outs[i].device.type == "cuda" else None

        def fn(*xs):
            return iref.run(sub, *xs, device=dev)[0]
        try:
            ins = tuple(t.detach() for t in inputs) + tuple(outs[j].detach() for j in cut_idx)
            rep = TL.close_report(fn, ins, outs[i], dtype=self.opts.torch_dtype, explain=False,
                                  tf32=bool(self.opts.allow_tf32) and self.opts.device == "cuda")
        except Exception:  # noqa: BLE001 (no verdict from the cut: the output stays refused)
            return False
        o = rep.outputs[0]
        if o.ok:
            self.derived_accepts.append({"output": v.name, "error": error, "via_outputs": [c.name for c in cut_vals],
                                         "worst_ratio": o.worst_ratio, "rms_ratio": o.rms_ratio})
        return bool(o.ok)

    def _derived_verdict(self, inputs, outs) -> Optional[Dict[int, Any]]:
        """The derived tolerance (truestar.tolerance, docs/TOLERANCE.md) of the whole program on these
        inputs: the reference interpreter in float64, the same with every op result rounded to the
        working dtype, and error model runs in which every op result gets the error a correct kernel
        can make, propagated through the rest of the program. {output index: OutputReport} for the
        float outputs, or None where it cannot run (a float64 program, an op the reference
        interpreter cannot run in float64 on this device). It never accepts what the derivation does
        not: an output passes only within its per element bound and its RMS limit."""
        if (self.opts.dtype == "float64" or self.opts.mixed      # mixed=True: its own declared precision bars
                or not _derivable(self.program.nodes)):
            return None
        from .. import tolerance as TL
        fl = [k for k, v in enumerate(self.program.outputs) if v.type.dtype.is_floating_point]
        if not fl:
            return None
        prog = self.program

        def fn(*xs):
            # the float64 runs on the device the derivation works on (the outputs' CUDA device: a
            # resnet18 step took 20 minutes of CPU here), the CPU when the device runs out of memory
            dev = next((x.device for x in xs if isinstance(x, torch.Tensor) and x.device.type == "cuda"), None)
            r = iref.run(prog, *xs, device=dev)
            r = r if isinstance(r, (list, tuple)) else [r]
            return tuple(r[k] for k in fl)

        def attempt(where, memo=True):
            ins = tuple(t.detach() if where is None else t.detach().to(where) for t in inputs)
            got = tuple(outs[k] if where is None else outs[k].to(where) for k in fl)
            return TL.close_report(fn, ins, got, dtype=self.opts.torch_dtype, explain=False, memo=memo,
                                   tf32=bool(self.opts.allow_tf32) and self.opts.device == "cuda")
        try:
            rep = None
            for where, memo in ((None, True), (None, False), (torch.device("cpu"), True)):
                try:
                    rep = attempt(where, memo)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
            if rep is None:
                raise RuntimeError("no device could hold the derivation")
        except Exception:  # noqa: BLE001 (no derivation: the fixed bar stays the verdict, as before)
            self.derived_unavailable = True
            return None
        self._derived_report = rep
        return {k: o for k, o in zip(fl, rep.outputs)}

    def _reference_program(self) -> Program:
        """The program the end to end float64 check runs: the program itself, or with ``mixed=True``
        the program with a range check after every value declared in 16 bits (emulate_precision
        range_only): exact arithmetic that overflows where the declared dtype overflows, so a float16
        loss scaler skips the steps the declared precision skips."""
        if not self.opts.mixed:
            return self.program
        if getattr(self, "_range_ref", None) is None:
            from ..train.amp import LOW_DTYPES, emulate_precision
            self._range_ref = emulate_precision(self.program, LOW_DTYPES, range_only=True)
        return self._range_ref

    def _mixed_floor(self) -> float:
        """The float32 end to end bar of this program (its kernel count), the least a float32 island
        output is held to."""
        return 4.0 * kverify.BASE_TOL["float32"]

    def _mixed_bars(self, inputs, refs) -> List[Optional[float]]:
        """End to end bars derived for the declared mixed precision (``mixed=True``). For every float
        output declared 32 bits or wider, two evaluations of the declared precision independent of
        the compiled kernels: the reference interpreter on ``emulate_precision(program)`` with
        float32 arithmetic (torch's float32 CPU kernels, every value rounded to its declared dtype:
        the 16 bit casts, contractions and activations of the policy, float32 everywhere else), once
        rounding to nearest and once rounding every 16 bit value toward zero (each at most one unit
        in the last place away). ``e`` is the larger of the first one's distance from the float64
        reference (what the declared precision costs) and the distance between the two (how far one
        ulp of every 16 bit value moves the output: an output fed by few 16 bit values, such as a
        bias gradient, can land one ulp away in any correct run). The bar is
        ``max(MIXED_FACTOR e, 4 x the float32 kernel bar)`` (over 0.1 the output is listed as ill
        conditioned in the declared precision). Outputs declared in the
        working precision keep the working precision bar (None here). A master weight rounded to 16
        bits, a dropped float32 island or a missing unscale moves an output far beyond it.
        ``self.mixed_bounds`` records each output's numbers."""
        from ..train.amp import emulate_precision
        floor = self._mixed_floor()
        saved = iref.COMPUTE_DTYPE
        iref.COMPUTE_DTYPE = torch.float32
        cpu_in = [t.detach().cpu() for t in inputs]
        try:
            if getattr(self, "_emu_prog", None) is None:
                self._emu_prog = emulate_precision(self.program)
                self._emu_rtz = emulate_precision(self.program, rounding="toward_zero")
            emu = iref.run(self._emu_prog, *cpu_in)
            rtz = iref.run(self._emu_rtz, *cpu_in)
        except Exception as exc:  # noqa: BLE001 (no emulation: refuse rather than fall back to the loose bar)
            iref.COMPUTE_DTYPE = saved
            raise ProgramMismatchError(
                f"[TS1480] compiled mixed precision program {self.program.name!r}: the declared precision "
                f"evaluation that derives the end to end bars failed ({type(exc).__name__}: {exc}); the "
                f"result is not returned.", program=self.program.name, stage="end_to_end") from exc
        iref.COMPUTE_DTYPE = saved
        bars: List[Optional[float]] = []
        self.mixed_bounds = []
        for v, e, z, r in zip(self.program.outputs, emu, rtz, refs):
            dt = v.type.dtype
            if not dt.is_floating_point or torch.finfo(dt).bits < 32:
                bars.append(None)
                self.mixed_bounds.append(None)
                continue
            err = _compare(v.name, e, r.detach().cpu(), 0.0).error
            ulp = _compare(v.name, z, e, 0.0).error
            worst = max(err, ulp)
            if not math.isfinite(worst):
                worst = 0.1
            # no cap: an output the declared precision itself leaves this far from float64 is ill
            # conditioned in it (a bias gradient behind batch norm cancels to rounding noise); any correct
            # evaluation is that far off, so the bar follows, and the output is listed (ill_conditioned)
            bar = max(floor, MIXED_FACTOR * worst)
            bars.append(bar)
            self.mixed_bounds.append({"output": v.name, "emulated_error": err, "ulp_sensitivity": ulp, "bar": bar,
                                      "ill_conditioned": bar > 0.1})
        return bars

    def _float32_reference(self, inputs, dev) -> Optional[List[torch.Tensor]]:
        """The reference interpreter computing in float32 instead of float64: the same IR ops, run
        by torch's own float32 kernels (accumulation in float32 included), on the program's own
        device (the CPU when that runs out of memory). It is an independent correct execution in this
        precision, so its distance from float64 says how ill conditioned each output is. The program's
        device, not the device the float64 end to end check happened to run on (``auto`` puts small
        checks on the CPU), so the number in the error and in the 4x acceptance does not depend on
        where that check ran (test_region_gpu_verify compares both reports). Only for float32
        programs (None otherwise, and the check stays as strict as before). ``dev`` is unused."""
        if self.opts.dtype != "float32":
            return None
        saved = iref.COMPUTE_DTYPE
        iref.COMPUTE_DTYPE = torch.float32
        try:
            if self.opts.device == "cuda":
                try:
                    return iref.run(self.program, *[t.detach() for t in inputs], device=self.opts.torch_device)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
            return iref.run(self.program, *[t.detach().cpu() for t in inputs])
        except Exception:  # noqa: BLE001 (an op without a float32 path: no allowance, the check stays strict)
            return None
        finally:
            iref.COMPUTE_DTYPE = saved

    def _producer(self, v: Value) -> str:
        for n in self.program.nodes:
            if any(o is v for o in n.outputs):
                return n.op
        return "input"

    def _branch_flips(self, inputs, outs, tol: float, decisions, bars=None) -> Tuple[List[Dict[str, Any]], Optional[float]]:
        """After an end to end mismatch: the comparisons whose decision in this run (recorded by the
        region checks) differs from float64's, and the largest relative error of the outputs against the
        float64 reference that takes this run's decisions (None when there is no such decision).

        A comparison of float values within rounding of a tie (a relu mask of an input at
        1e-8 of its rms) can go either way in a correct float32 run, and everything downstream of it
        changes. If the float64 reference with the same decisions matches the output, the decisions
        explain the whole difference. Only runs after a failed end to end check (CPU, two reference runs)."""
        taken = {vid: t for vid, t in (decisions or {}).items() if t is not None}
        if not taken:
            return [], None
        cpu_in = [t.detach().cpu() for t in inputs]
        _, named = iref.run(self._reference_program(), *cpu_in, return_all=True)
        known: Dict[int, torch.Tensor] = {id(v): t for v, t in zip(self.program.inputs, cpu_in)}
        for p in self.program.params:
            known[id(p)] = self.program.param_data[p.name]
        flips: List[Dict[str, Any]] = []
        for n in self.program.nodes:
            if not n.outputs or id(n.outputs[0]) not in taken:
                continue
            o = n.outputs[0]
            ref, got = named[o.name], taken[id(o)].to(torch.bool)
            if tuple(got.shape) != tuple(ref.shape):
                continue
            diff = got != ref.to(torch.bool)
            count = int(diff.sum())
            if not count:
                continue
            at = tuple(int(i) for i in diff.nonzero()[0])
            ops = [named[x.name] if x.name in named else known.get(id(x)) for x in n.inputs[:2]]
            if any(t is None for t in ops):
                continue
            a, b = (torch.broadcast_to(t.to(torch.float64), ref.shape) for t in ops)
            lhs, rhs = float(a[at]), float(b[at])
            scale = max(float(t.to(torch.float64).pow(2).mean().sqrt()) for t in ops)
            gap = float((a - b)[diff].abs().max())
            flips.append({"value": o.name, "op": n.op, "lhs": n.inputs[0].name, "rhs": n.inputs[1].name,
                          "count": count, "at": list(at), "lhs_value": lhs, "rhs_value": rhs,
                          "margin": abs(lhs - rhs) / scale if scale > 0 else math.inf,
                          "max_margin": gap / scale if scale > 0 else math.inf, "source": n.source})
        if not flips:
            return [], None
        replay = iref.run(self._reference_program(), *cpu_in, overrides=taken)
        worst, seen = 0.0, []
        for v, o, r in zip(self.program.outputs, outs, replay):
            t = tol if bars is None or bars[len(seen)] is None else bars[len(seen)]
            seen.append(v)
            c = _compare(v.name, o, r, 0.0 if kind_of(v.type.dtype) != "f" else t)
            if bars is not None and not c.ok:
                return flips, math.inf
            worst = max(worst, c.error if c.error == c.error else math.inf)
        return flips, worst

    def end_to_end_tolerance(self) -> float:
        checked = sum(1 for _, s in self.plan.all_steps() if s.kind in ("kernel", "library"))
        base = kverify.BASE_TOL[self.opts.dtype]
        if self.opts.allow_tf32 and self.opts.dtype == "float32" and self.opts.device == "cuda":
            base = kverify.TF32_TOL
        return min(0.1, base * 4.0 * max(1.0, math.sqrt(checked)))

    @property
    def num_kernels(self) -> int:
        return sum(1 for _, s in self.plan.all_steps() if s.kind == "kernel")
