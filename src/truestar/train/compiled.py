"""A training step program compiled for a device, with its params and state carried on the device.

``CompiledTrainStep`` owns the carried values (params, optimizer state) as device tensors and calls the
compiled step program once per step, donating them, so the update writes the new values over the old
buffers (docs/MEMORY.md, Donation). With ``graph=True`` the step is replayed from one CUDA graph after
its checked first call: the host cost of issuing every kernel disappears (docs/TRAINING.md, Compiled).

    cs = CompiledTrainStep(step, "cuda", "float32", graph=True)
    for x, y in loader:
        loss, grad_norm, nonfinite = cs(x, y, lr=3e-4)
    params = cs.params()

Correctness rules:

* The first call is the compiled program's checked call (every kernel and region against float64).
  The graph is captured only after ``warmup`` checked or plain calls, and its first replay is
  compared bit for bit with a plain call on the same inputs and state (both from a copy of the
  state), so a replay that computes anything else raises before it is used.
* A replay must read and write the same addresses on every step. Carried values the step writes
  over their own buffer (donation) stay where they are; the others (the 0 dim step count, a state
  value updated inside a reduction kernel) are copied back into their buffer inside the graph and
  listed in ``copied_back``. If a value that was written in place before capture moves during
  capture, capture raises ``GraphCaptureError`` (TS780) naming it and restores the state from the
  copy: never a silent fallback.
* A new batch is copied into the graph's static input buffers; a batch of another shape or dtype
  than the captured one raises (TS781) instead of replaying with stale shapes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..ir.core import IRError
from .step import TrainStep

N_HEAD = 3   # loss, grad_norm, nonfinite


class GraphCaptureError(IRError):
    """TS780: the step cannot be replayed from one CUDA graph (a carried value is not updated in
    place, or the replay did not reproduce a plain call bit for bit)."""

    code = "TS780"


class GraphInputError(IRError):
    """TS781: a batch given to a captured step differs in shape, dtype or device from the captured one."""

    code = "TS781"


class CompiledTrainStep:
    """``step`` (from ``build_train_step``) compiled by ``truestar.compiler``. See the module docstring."""

    def __init__(self, step: TrainStep, device: str = "cuda", dtype: str = "float32", graph: bool = False,
                 warmup: int = 2, **compile_kw):
        from ..compiler import compile_program
        self.step_def = step
        self.device = device
        if step.amp is not None:
            # a mixed precision step: compile in the policy's dtype with float32 islands for the master
            # weights, optimizer state and loss (compile_program mixed=True); "float32" asks for that
            if dtype in ("float32", torch.float32):
                dtype = str(step.amp.dtype).replace("torch.", "")
            compile_kw.setdefault("mixed", True)
        self.cp = compile_program(step.program, device, dtype, **compile_kw)
        self.names = [v.name for v in step.program.inputs]
        self.carried = list(step.carried)
        wdtype = getattr(torch, str(dtype).replace("torch.", ""))
        # a mixed step stores each carried value in its own dtype (float32 islands); a float32 step keeps
        # every float value in the working dtype, as before
        stored = dict(zip(self.names, self.cp._in_dtypes)) if step.amp is not None else {}
        self.state: Dict[str, torch.Tensor] = {}
        for n in self.carried:
            t = step.initial[n].detach()
            dt = (stored.get(n) or wdtype) if t.dtype.is_floating_point else t.dtype
            self.state[n] = t.to(device, dt).clone()
        self.graph = graph
        self.warmup = warmup
        self.calls = 0
        self._g: Optional[torch.cuda.CUDAGraph] = None
        self._static: Dict[str, torch.Tensor] = {}
        self._head: List[torch.Tensor] = []
        self._dtype = stored.get("lr") or wdtype
        self._aliased: set = set()
        self.copied_back: List[str] = []   # carried values a replay copies back (not written in place)
        self._pending: list = []           # index checks recorded in the graph (read after a replay)

    # -- plain calls

    def _inputs(self, batch: Dict[str, torch.Tensor], lr: torch.Tensor, state: Dict[str, torch.Tensor]):
        ins = []
        for n in self.names:
            if n in batch:
                ins.append(batch[n])
            elif n == "lr":
                ins.append(lr)
            else:
                ins.append(state.pop(n))
        return ins

    def _plain(self, batch, lr, state) -> List[torch.Tensor]:
        outs = self.cp(self._inputs(batch, lr, state), donate=True)
        for n, t in zip(self.carried, outs[N_HEAD:]):
            state[n] = t
        return outs[:N_HEAD]

    def _batch(self, args) -> Dict[str, torch.Tensor]:
        names = self.step_def.batch_inputs
        if len(args) != len(names):
            raise TypeError(f"the step takes {len(names)} batch tensors ({', '.join(names)}), got {len(args)}")
        return {n: a.to(self.device) for n, a in zip(names, args)}

    def _lr(self, lr) -> torch.Tensor:
        return torch.as_tensor(lr, dtype=self._dtype).to(self.device)

    def __call__(self, *batch: torch.Tensor, lr: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step. Returns (loss, grad_norm, nonfinite) as device tensors (no host sync)."""
        b = self._batch(batch)
        self.calls += 1
        if getattr(self.step_def, "sr_seed", None) is not None:
            # weights="bfloat16_sr": the stochastic rounding key of step k (0 based) is
            # call_key(sr_seed, k), a fixed schedule, so every run and backend rounds the same way
            from .. import rng
            b["sr.key"] = torch.tensor(rng.as_i64(rng.call_key(self.step_def.sr_seed, self.calls - 1)),
                                       dtype=torch.int64, device=self.device)
        if self._g is not None:
            return self._replay(b, lr)
        if self.graph and self.calls > self.warmup:
            self._capture(b, lr)
            return self._replay(b, lr)
        ptrs = {n: t.data_ptr() for n, t in self.state.items()}
        head = self._plain(b, self._lr(lr), self.state)
        # which carried values the step writes over their own buffer (donation, docs/MEMORY.md)
        self._aliased = {n for n, t in self.state.items() if t.data_ptr() == ptrs[n]}
        return tuple(head)

    def params(self) -> Dict[str, torch.Tensor]:
        """The current params by their model names (views of the carried buffers, of the flat group
        buffer with ``param_group=True``; copy them to keep them)."""
        src = self._static if self._g is not None else self.state
        own = {n: src[n] for n in self.step_def.param_names}
        if self.step_def.layout is None:
            return own
        return {n: t for n, t in self.step_def.unflatten(own).items() if n in self.step_def.layout}

    def carried_state(self) -> Dict[str, torch.Tensor]:
        src = self._static if self._g is not None else self.state
        return {n: src[n] for n in self.carried}

    # -- the graph

    def _capture(self, batch, lr) -> None:
        backup = {n: t.clone() for n, t in self.state.items()}
        static_b = {n: t.clone() for n, t in batch.items()}
        static_lr = self._lr(lr)
        ptrs = {n: t.data_ptr() for n, t in self.state.items()}
        # a plain call on a copy of the state, for the bitwise comparison with the first replay
        ref_state = {n: t.clone() for n, t in self.state.items()}
        ref_head = self._plain(static_b, static_lr, ref_state)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        # values the step does not write in place are held here and copied back inside the graph
        held = {n: self.state[n] for n in self.carried if n not in self._aliased}
        state = self.state
        self.state = {}
        from ..compiler import runtime as _rt
        _rt.CAPTURED_PENDING.clear()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(g, stream=stream):
                head = self._plain(static_b, static_lr, state)
                for n, t in held.items():
                    t.copy_(state[n])
                    state[n] = t
        torch.cuda.current_stream().wait_stream(stream)
        # index range checks (an embedding's token ids) recorded in the graph: read after each replay
        self._pending = list(_rt.CAPTURED_PENDING)
        _rt.CAPTURED_PENDING.clear()
        moved = [n for n in self.carried if state[n].data_ptr() != ptrs[n]]
        if moved:
            del g, head
            self.state = backup
            raise GraphCaptureError(
                f"[TS780] the training step cannot be replayed from a CUDA graph: {len(moved)} carried "
                f"value(s) moved to another buffer during capture although the plain call before wrote them "
                f"in place ({', '.join(moved[:5])}{', ...' if len(moved) > 5 else ''}), so a replay would read "
                f"last step's values from freed memory. The state was restored; run with graph=False.")
        self.copied_back = sorted(held)
        # the state tensors are now the graph's own buffers; the first replay runs the step for real
        for n, t in backup.items():
            state[n].copy_(t)
        del backup
        self._g, self._static, self._head = g, {**state, **static_b, "lr": static_lr}, list(head)
        self._g.replay()
        torch.cuda.synchronize()
        self._check_pending()
        bad = [n for n in self.carried if not _same_bits(state[n], ref_state[n])]
        bad += [k for k, a, r in zip(("loss", "grad_norm", "nonfinite"), self._head, ref_head) if not _same_bits(a, r)]
        if bad:
            self._g = None
            raise GraphCaptureError(f"[TS780] the first replay of the captured training step differs bit for bit "
                                    f"from a plain call on the same inputs and state in {', '.join(bad[:5])}; the "
                                    f"replay is not used.")
        self._replayed_once = True

    def _replay(self, batch, lr) -> Tuple[torch.Tensor, ...]:
        if getattr(self, "_replayed_once", False):
            self._replayed_once = False      # the capture's verified replay was this step
            return tuple(self._head)
        for n, t in batch.items():
            s = self._static[n]
            if t.shape != s.shape or t.dtype != s.dtype or t.device != s.device:
                raise GraphInputError(f"[TS781] batch input {n!r} is {tuple(t.shape)} {t.dtype} on {t.device}; the "
                                      f"captured step takes {tuple(s.shape)} {s.dtype} on {s.device}. Build "
                                      f"another CompiledTrainStep for that shape, or use graph=False.")
            s.copy_(t)
        self._static["lr"].fill_(float(lr))
        if self._pending:
            for host, _ in self._pending:
                host.zero_()
        self._g.replay()
        if self._pending:
            torch.cuda.current_stream().synchronize()   # one host sync per step, only with index checks
            self._check_pending()
        return tuple(self._head)

    def _check_pending(self) -> None:
        """Raise the index range error a replay recorded (the plain call raises the same one)."""
        for host, entries in getattr(self, "_pending", ()):
            if bool(host.any()):
                from ..compiler.checks import bounds_error
                _, op, node, index, size, unique = next(e for e, h in zip(entries, host.tolist()) if h)
                raise bounds_error(op, node, index, size, unique)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.is_floating_point:
        w = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.contiguous().view(w), b.contiguous().view(w))
    return torch.equal(a, b)
