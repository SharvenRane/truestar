"""CUDA graph replay of a compiled program, one graph per input signature (docs/COMPILER.md, CUDA
graphs).

A compiled call issues one launch per step from Python: 14 to 20 us of host time each. A decode
step of a Llama style block is 39 steps whose GPU work is a few microseconds each, so the host is
the bottleneck there by a factor of several. ``cp.graphed()`` records the steps of one call into a
``torch.cuda.CUDAGraph`` per input signature and replays it: one launch per call.

What is kept from the ordinary call, and how:

* The first call at a new signature is the ordinary call: compile, Triton compile, every float64
  check (kernel, region, end to end). Only then is the graph recorded, from steps that were
  verified at exactly this signature.
* The recording is replayed once at once and compared bitwise with that first call (outputs and
  every input updated in place); any difference refuses the graph (``CaptureMismatchError``).
* Index checks (``index_copy``, ``index_select``, gathers) are recorded in the graph: their flags
  are copied to pinned host memory inside the graph, and every replay waits for the GPU and reads
  them before returning. A bad index raises the same error as the ordinary call and the results are
  not returned. That wait is the only host sync per call.
* A step that syncs with the host refuses the recording (``HostSyncError``), never runs eagerly in
  its place. ``TRUESTAR_VERIFY=all`` (a check every call) is refused.
* Compiled in sentinels (docs/COMPILER.md, Sentinels) stay armed: each graph owns a pinned flag buffer
  that its kernels write, the torch step triggers are folded into it inside the graph, and every
  replay reads it after the GPU finishes; a fired flag runs the ordinary locate run on this call's
  inputs, which raises ``SentinelError`` naming the op (an input updated in place has then been
  written once more at the same positions).
* A PyTorch precision setting the program follows (TF32, deterministic algorithms) selects its
  variant before the signature is looked up, so a changed setting gets its own verified graph.

Inputs listed in ``static=`` and every input of ``update=`` are bound by address: the graph reads
the caller's tensor, every call must pass that same tensor, and an input of ``update=`` is written
in place (a KV cache). Other inputs are copied into the graph's buffers, one launch for all.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from ..runtime.graphs import BucketError, CaptureError, CaptureMismatchError, HostSyncError, bitwise_equal


class _Entry:
    __slots__ = ("graph", "static_in", "outs", "host", "pending", "done", "key", "flags", "views", "chk")


class GraphedProgram:
    """``cp.graphed(static=(), update=None, outputs="copy")``: see the module docstring.

    ``outputs="copy"`` (default) returns fresh tensors; ``"static"`` returns the graph's own output
    buffers, which the next replay overwrites. Outputs named in ``update`` are always the updated
    input tensors themselves."""

    def __init__(self, cp, static: Sequence[int] = (), update: Optional[Dict[int, int]] = None,
                 outputs: str = "copy"):
        if cp.opts.device != "cuda":
            raise ValueError("graphed(): CUDA graphs need a program compiled for 'cuda'")
        if outputs not in ("copy", "static"):
            raise ValueError(f"outputs must be 'copy' or 'static', not {outputs!r}")
        self.cp = cp
        self.update = {int(i): int(j) for i, j in dict(update or {}).items()}
        self.static = set(int(i) for i in static) | set(self.update)
        self.outputs = outputs
        self._entries: Dict[tuple, _Entry] = {}
        self._pool = None
        self.captures = 0
        self.replays = 0

    # ---- calls
    def __call__(self, inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        from .runtime import verify_mode
        cp = self.cp
        if verify_mode(cp.verify) == "all":
            raise HostSyncError("[TS800] graphed(): TRUESTAR_VERIFY=all checks every call against float64 on the "
                                "host; a CUDA graph replays without checks. Use verify='first' or 'off'.")
        if cp.follows_tf32 or cp.follows_deterministic:
            want = cp._wanted(cp.opts)
            if any(getattr(cp.opts, k) != v for k, v in want.items()):
                cp._switch(want)
        inputs = list(inputs)
        for i, t in enumerate(inputs):
            if not isinstance(t, torch.Tensor) or t.device.type != "cuda":
                raise CaptureError(f"graphed(): input {i} must be a CUDA tensor, got "
                                   f"{getattr(t, 'device', type(t).__name__)}")
        key = (bool(cp.opts.allow_tf32), bool(cp.opts.deterministic),
               tuple((tuple(t.shape), t.dtype, t.device) for t in inputs))
        e = self._entries.get(key)
        if e is None:
            return self._first(key, inputs)
        return self._replay(e, inputs)

    def _first(self, key, inputs):
        """The ordinary verified call at this signature, then the recording."""
        cp = self.cp
        for i in sorted(self.static):
            # a bound input is read as it is: the ordinary call converts a float input to the dtype it
            # is stored in, a graph cannot, so the caller's tensor must already be in it
            want = cp._in_dtypes[i]
            if want is not None and inputs[i].dtype != want:
                raise CaptureError(f"[TS802] graphed(): input {i} is bound by address and must already be "
                                   f"{str(want).replace('torch.', '')} (the dtype the program stores it in), "
                                   f"got {str(inputs[i].dtype).replace('torch.', '')}")
        before = {i: inputs[i].detach().clone() for i in self.update}
        outs = cp(list(inputs), update=self.update or None)
        after = {i: inputs[i].detach().clone() for i in self.update}
        e = self._capture(key, inputs, outs, before, after)
        self._entries[key] = e
        return outs

    def _arm(self, rt, e_flags):
        """Point the program's sentinels at this graph's own pinned flag buffer."""
        chk = self.cp.checks
        if chk is None:
            return
        flags, views = e_flags
        chk.slot = [flags, views, None]
        chk.flags = flags
        for (_, step), v in zip(chk.sites, views):
            step.chk_view = v
        chk.trig, chk.watch_nan, chk.watch_fin = [], [], []
        rt.chk = chk

    def _flag_buffer(self):
        from ..kernel import checks as kchecks
        chk = self.cp.checks
        if chk is None:
            return None
        n = len(chk.sites) * kchecks.WORDS + 1
        f = torch.zeros(n, dtype=torch.int32, pin_memory=True)
        return f, [f[i * kchecks.WORDS:(i + 1) * kchecks.WORDS] for i in range(len(chk.sites))]

    def _run(self, rt, static_in, flags=None):
        cp = self.cp
        if flags is not None:
            self._arm(rt, flags)
        rt.pending, rt.checked = [], {}
        rt.update = frozenset(cp.plan.in_slots[i] for i in self.update)
        try:
            outs = cp.plan.run(list(static_in), rt, own=False)
            # an update the plan could not alias in place ends, in the ordinary call, with a copy into
            # the caller's tensor (CompiledProgram.__call__); the same copy here is recorded in the
            # graph, so a replay writes the bound input as the verified call did
            for i, j in self.update.items():
                if outs[j].data_ptr() != static_in[i].data_ptr():
                    static_in[i].copy_(outs[j])
                    outs[j] = static_in[i]
            if rt.chk is not None:
                rt.chk._fold_triggers()       # torch step triggers into the flag buffer, in the graph
        finally:
            rt.update = frozenset()
            rt.chk = None
        pending, rt.pending, rt.checked = rt.pending, [], {}
        return outs, pending

    def _capture(self, key, inputs, outs1, before, after) -> _Entry:
        cp = self.cp
        dev, dt = cp.opts.torch_device, cp.opts.torch_dtype
        static_in = []
        for i, (t, k) in enumerate(zip(inputs, cp._kinds)):
            if i in self.static:
                static_in.append(t)
            else:
                idt = cp._in_dtypes[i] or dt
                static_in.append((t.to(dev, idt) if k == "f" else t.to(dev)).detach().clone())
        rt = cp._bound[tuple([t.shape for t in inputs])]
        mode = rt.mode
        rt.mode = "off"

        def restore():
            for i, v in before.items():
                inputs[i].copy_(v)

        flags = self._flag_buffer()
        try:
            restore()
            side = torch.cuda.Stream(device=static_in[0].device if static_in else None)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):          # allocator and Triton warm on the capture stream
                _, pending = self._run(rt, static_in, flags)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            host = torch.zeros(max(1, len(pending)), dtype=torch.int32, pin_memory=True)
            restore()
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(g, pool=self._pool):
                    outs_g, pending = self._run(rt, static_in, flags)
                    if pending:
                        iflags = torch.stack([f[0].reshape(()) for f in pending]).to(torch.int32)
                        host[: len(pending)].copy_(iflags, non_blocking=True)
            except RuntimeError as err:
                if "captur" in str(err).lower() or "sync" in str(err).lower():
                    raise HostSyncError(f"[TS800] graphed(): a step of {cp.program.name!r} syncs with the host and "
                                        f"cannot be recorded in a CUDA graph: {err}") from err
                raise
            restore()
            if flags is not None:
                flags[0].zero_()
            g.replay()
            torch.cuda.synchronize()
            if flags is not None and bool(flags[0].any()):
                raise CaptureMismatchError(f"[TS801] graphed(): a sentinel fired in the recorded graph of "
                                           f"{cp.program.name!r} but not in the verified call; refused")
        finally:
            rt.mode = mode
        # the recording must give exactly the first call's results
        for j, (a, b) in enumerate(zip(outs_g, outs1)):
            if j in self.update.values():
                continue
            if not bitwise_equal(a, b):
                raise CaptureMismatchError(f"[TS801] graphed(): the CUDA graph of {cp.program.name!r} gave other bits than "
                                           f"the verified call for output {j}; refused")
        for i, v in after.items():
            if not bitwise_equal(inputs[i], v):
                raise CaptureMismatchError(f"[TS801] graphed(): the CUDA graph of {cp.program.name!r} updated input {i} "
                                           f"differently from the verified call; refused")
        e = _Entry()
        e.graph, e.static_in, e.outs, e.key = g, static_in, outs_g, key
        e.host, e.pending = host, [p[1:] for p in pending]
        e.flags, e.chk = flags, self.cp.checks
        e.done = torch.cuda.Event() if (pending or flags is not None) else None
        self.captures += 1
        return e

    def _replay(self, e: _Entry, inputs):
        cp = self.cp
        dst, src = [], []
        for i, (t, s) in enumerate(zip(inputs, e.static_in)):
            if i in self.static:
                if t.data_ptr() != s.data_ptr() or t.shape != s.shape:
                    raise BucketError(f"[TS802] graphed(): input {i} is bound by address; pass the tensor the graph was "
                                      f"recorded with")
            else:
                dst.append(s)
                src.append(t)
        if dst:
            torch._foreach_copy_(dst, src)
        if e.flags is not None:
            e.flags[0].zero_()             # pinned host memory: a memset, the last replay is done
        e.graph.replay()
        self.replays += 1
        if e.done is not None:
            e.done.record()
            e.done.synchronize()
            if e.flags is not None and bool(e.flags[0].any()):
                self._located(e, inputs)
            n = len(e.pending)
            if n and bool(e.host[:n].any()):
                from .checks import bounds_error
                hit = next(p for p, h in zip(e.pending, e.host[:n].tolist()) if h)
                _, op, node, index, size, unique = hit
                raise bounds_error(op, node, index, size, unique)
        outs = []
        inv = {j: i for i, j in self.update.items()}
        for j, o in enumerate(e.outs):
            if j in inv:
                outs.append(inputs[inv[j]])
            else:
                outs.append(o.clone() if self.outputs == "copy" else o)
        return outs

    def _located(self, e, inputs):
        """A sentinel fired in a replay: the ordinary locate run on these inputs names the op."""
        cp = self.cp
        rt = cp._bound[tuple([t.shape for t in inputs])]
        rt.chk = e.chk
        e.chk.flags = e.flags[0]
        try:
            cp._locate([t for t in inputs], rt, " (in a CUDA graph replay)")
        finally:
            rt.chk = None
        from .checks import SentinelError
        raise SentinelError(f"truestar check: a compiled in check fired in a CUDA graph replay of "
                            f"{cp.program.name!r}. The results of this call were not returned.",
                            check="nan", program=cp.program.name)

    @property
    def num_graphs(self) -> int:
        return len(self._entries)
