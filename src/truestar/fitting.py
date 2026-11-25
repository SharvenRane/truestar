"""One compiled, verified program per training step, built from a native loss function.

    net = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 1))

    @truestar.native
    def loss(p: net.spec, x: Tensor["B", 16], y: Tensor["B", 1]) -> Tensor[()]:
        return nn.mse(net(p, x), y)

    step = truestar.train_step(loss, optim.AdamW(lr=1e-3), clip=1.0)
    ts = step.init(net.init(seed=0))
    for x, y in batches:
        ts, r = step(ts, x, y)          # r.loss, r.grad_norm, r.lr
    step.save("run.safetensors", ts)    # later: ts = step.load("run.safetensors")

``train_step`` composes, in IR, the loss program, its gradient (the autodiff pass), the non finite
check, the global norm and clipping, the optimizer's per step scalars and per tensor update, and
with float16 the loss scale update, into one program. Each step is one call of that program; no
Python runs between its ops. The program is whatever executor runs it: the float64 reference
interpreter by default, or ``executor="cpu"`` / ``"cuda"`` for the compiled executor, which checks
the compiled program against the reference on its first call (docs/COMPILER.md).

Everything a step reads is an explicit input: parameters, optimizer slots, model state (batch norm
statistics), the batch, the learning rate and update number (computed on the host from the step
count, so a schedule never recompiles), and the random key (``rng.call_key(seed, step)``, so step n
draws the same dropout masks after a resume and on every backend). See docs/OPTIM.md.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from . import pytree, rng
from .frontend.function import Executor, NativeFunction, ReferenceExecutor, emit_call
from .ir.core import IRError, Program, Value
from .train.edit import Editor
from .train.optim import any_nonfinite


class FitError(IRError):
    """A training step could not be built or run; the message says what to change. ``code`` is one
    of TS760 to TS766 (``truestar codes``)."""

    def __init__(self, message: str, code: str = "TS762"):
        super().__init__(f"[{code}] {message}")
        self.code = code


class NonFiniteStep(FitError):
    """A NaN or Inf in the loss or a gradient (TS763). The step was not applied; ``report`` names the
    first op whose result is not finite (its operands are finite)."""

    def __init__(self, message: str, report: Optional[Dict[str, Any]] = None):
        super().__init__(message, "TS763")
        self.report = report or {}


@dataclasses.dataclass
class TrainState:
    """Everything a step carries to the next one. Nothing else is hidden anywhere."""

    params: Any
    opt: Dict[str, Any]
    step: int = 0
    model_state: Any = None
    amp: Optional[Dict[str, torch.Tensor]] = None   # {"scale": f32[], "growth": f64[]} with float16


@dataclasses.dataclass
class StepResult:
    step: int
    loss: float
    grad_norm: float
    lr: float
    skipped: bool = False
    aux: Any = None
    seconds: float = 0.0


class _StepEditor(Editor):
    """An editor that the front end's translator can emit into (it expects a builder's names)."""

    def _fresh(self, base: str) -> str:
        return self.fresh(base)

    def op_multi(self, *a, **k):
        raise FitError("control flow inside an optimizer rule is not supported")


def _resolve_executor(executor) -> Executor:
    if executor is None or executor == "reference":
        return ReferenceExecutor()
    if isinstance(executor, str):
        from .compiler import CompiledExecutor
        dev, _, prec = executor.partition(":")
        return CompiledExecutor(dev, dtype=prec or "float32")
    if hasattr(executor, "prepare"):
        return executor
    raise FitError(f"executor must be 'reference', 'cpu', 'cuda' (optionally ':float16' or ':bfloat16') or an "
                   f"Executor, got {executor!r}")


def _leaf_indices(sig, name: str) -> List[int]:
    return [i for i, (path, _) in enumerate(sig.params)
            if path == name or path.startswith(name + ".") or path.startswith(name + "[")]


def _is_t(x) -> bool:
    return isinstance(x, torch.Tensor)


def _prune(prog: Program, name: str) -> Program:
    """`prog` without the nodes, params and inputs its outputs do not read."""
    need = {id(v) for v in prog.outputs}
    keep = []
    for node in reversed(prog.nodes):
        if any(id(o) in need for o in node.outputs):
            keep.append(node)
            need.update(id(x) for x in node.inputs if x is not None)
    keep.reverse()
    params = [v for v in prog.params if id(v) in need]
    inputs = [v for v in prog.inputs if id(v) in need]
    used = {str(d) for v in inputs + params + [o for n in keep for o in n.outputs] for d in v.type.shape
            if not isinstance(d, int)}
    syms = {k: v for k, v in prog.symbols.items() if k in used}  # the update program has no batch size
    out = Program(name, inputs, params, {v.name: prog.param_data[v.name] for v in params}, keep,
                  list(prog.outputs), syms)
    out.verify()
    return out


class Step:
    """A training step program built from a native loss function (see the module docstring)."""

    def __init__(self, loss_fn: NativeFunction, optimizer, *, params: str = "p", state: Optional[str] = None,
                 clip: Optional[float] = None, precision: Optional[str] = None, loss_scale="auto",
                 clip_value: Optional[float] = None, clip_norm_type: float = 2.0, executor=None,
                 static: Optional[Dict[str, Any]] = None, loop_memory: str = "auto",
                 data_parallel=None, accumulate: int = 1, bucket_bytes: int = 25 << 20, check_replicas: int = 0):
        from . import optim as _optim
        from .nn.core import native_method
        from .train import ops as train_ops

        if not isinstance(loss_fn, NativeFunction):
            raise FitError(f"train_step needs a @truestar.native loss function, got {type(loss_fn).__name__}", "TS760")
        if not isinstance(optimizer, _optim.Optimizer):
            raise FitError(f"optimizer must be a truestar.optim optimizer, got {type(optimizer).__name__}")
        if clip is not None and not clip > 0:
            raise FitError(f"clip must be a positive max norm, got {clip}")
        if clip_value is not None and not clip_value > 0:
            raise FitError(f"clip_value must be positive, got {clip_value}")
        if clip_value is not None and clip is not None:
            raise FitError("pass clip (a max gradient norm) or clip_value (an entry bound), not both")
        if float(clip_norm_type) not in (1.0, 2.0, math.inf):
            raise FitError(f"clip_norm_type must be 1, 2 or inf, got {clip_norm_type}")
        self.loss_fn, self.optimizer, self.clip = loss_fn, optimizer, clip
        self.clip_value, self.clip_norm_type = clip_value, float(clip_norm_type)
        if int(accumulate) < 1:
            raise FitError(f"accumulate must be at least 1, got {accumulate}")
        self.dp, self.accumulate = data_parallel, int(accumulate)
        self.bucket_bytes, self.check_replicas = int(bucket_bytes), int(check_replicas)
        self.split = data_parallel is not None or self.accumulate > 1
        self.params_name, self.state_name = params, state
        self.static = dict(static or {})
        c = loss_fn.specialize(**self.static)
        self.compiled_loss = c
        sig = c.signature
        if params not in loss_fn.trees and params not in loss_fn.specs:
            raise FitError(f"{loss_fn.name} has no tensor parameter {params!r} (its tensor parameters: "
                           f"{list(loss_fn.specs) + list(loss_fn.trees)}); pass params=<name>", "TS760")
        if state is not None and state not in loss_fn.trees:
            raise FitError(f"{loss_fn.name} has no structured parameter {state!r} for the model state", "TS761")
        P = c.program
        out0 = P.outputs[0]
        if out0.type.rank != 0 or not out0.type.dtype.is_floating_point:
            raise FitError(f"{loss_fn.name} must return the loss (a float scalar) first; its first result is {out0.type}", "TS760")
        p_idx = _leaf_indices(sig, params)
        s_idx = _leaf_indices(sig, state) if state is not None else []
        n_leaf_inputs = len(sig.params)
        self._p_names = [P.inputs[i].name for i in p_idx]
        self._s_names = [P.inputs[i].name for i in s_idx]
        self._param_tree = loss_fn.trees.get(params)
        self._state_tree = loss_fn.trees.get(state) if state is not None else None
        # results of the loss function: loss, then the new model state (when ``state``), then aux
        out_tree = sig.out_tree
        n_out = len(P.outputs)
        if state is not None:
            ok = (out_tree is not None and out_tree["kind"] == "tuple" and len(out_tree["children"]) >= 2
                  and pytree.describe(out_tree["children"][1]) == pytree.describe(self._state_tree))
            if not ok:
                raise FitError(f"with state={state!r}, {loss_fn.name} must return (loss, new {state}) with new "
                               f"{state} shaped like {pytree.describe(self._state_tree)}; it returns "
                               f"{pytree.describe(out_tree) if out_tree else 'one tensor'}", "TS761")
            n_state_out = len(s_idx)
            for j, i in enumerate(s_idx):
                if not P.outputs[1 + j].type.same_as(P.inputs[i].type):
                    raise FitError(f"new {sig.params[i][0]} is {P.outputs[1 + j].type}, the input is {P.inputs[i].type}", "TS761")
            aux_children = out_tree["children"][2:]
        else:
            n_state_out = 0
            aux_children = out_tree["children"][1:] if out_tree is not None and out_tree["kind"] == "tuple" else []
        self._aux_tree = {"kind": "tuple", "children": aux_children} if aux_children else None
        n_aux = n_out - 1 - n_state_out

        # mixed precision
        self.precision = precision
        self.scaler = None
        prog = P
        if precision not in (None, "f32", "float32"):
            from .train.amp import AutocastPolicy, LossScaler, autocast
            low = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "f16": torch.float16,
                   "float16": torch.float16}.get(precision)
            if low is None:
                raise FitError(f"precision must be 'f32', 'bf16' or 'f16', got {precision!r}")
            policy = AutocastPolicy(low, fp32_ops=frozenset({"log_softmax", "softmax", "layer_norm"}))
            try:
                prog = autocast(prog, policy)
            except IRError as e:
                raise FitError(f"mixed precision: {e}", "TS766") from None
            if loss_scale == "auto":
                loss_scale = LossScaler() if low == torch.float16 else None
            self.scaler = loss_scale if loss_scale and getattr(loss_scale, "enabled", True) else None
        scaled = self.scaler is not None
        if scaled:
            e0 = Editor(prog)
            sc = e0.input("amp_scale", torch.float32, ())
            e0.source = "truestar.fit loss scaling"
            s_loss = e0.op("mul", prog.outputs[0], e0.op("cast", sc, dtype=prog.outputs[0].type.dtype),
                           name="scaled_loss")
            prog = e0.build([s_loss] + list(prog.outputs))
        n_fwd = len(prog.outputs)
        G = train_ops.grad(prog, self._p_names, of=[0], loop_memory=loop_memory, name=f"grad_{P.name}")
        if self.split:
            why = []
            if state is not None:
                why.append(f"model state {state!r} (each replica's statistics would come from its own rows; a "
                           "synchronized statistic is not built)")
            if c.key_input:
                why.append("random ops (each replica would draw its own masks for its rows, not the single "
                           "device step's masks)")
            if scaled:
                why.append("float16 loss scaling (the skip decision is not synchronized over replicas yet)")
            if why:
                raise FitError("data parallel or accumulated steps refuse " + "; ".join(why) + ", because the "
                               "result would not be the single device step", "TS823")

        e = _StepEditor(G, f"train_step_{P.name}")
        e.source = "truestar.fit"
        by_name = {v.name: v for v in G.inputs}
        pvals = [by_name[n] for n in self._p_names]
        loss_v = G.outputs[1] if scaled else G.outputs[0]
        grads = list(G.outputs[n_fwd:])
        self.grad_program = None
        if self.split:
            # two programs: forward and backward per micro batch, then (after the sums over micro batches
            # and replicas) the update with the summed loss and gradients as inputs
            self.grad_program = G
            loss_v = e.input("sync_loss", loss_v.type.dtype, loss_v.type.shape)
            grads = [e.input(f"sync_grad_{i}", g.type.dtype, g.type.shape) for i, g in enumerate(grads)]
        lr = e.input("lr", torch.float64, ())
        t = e.input("t", torch.float64, ())
        slots = list(optimizer.slots)
        slot_vals = {s: [e.input(f"opt_{s}_{v.name}", v.type.dtype, v.type.shape) for v in pvals] for s in slots}
        tracker = None
        if scaled:
            scale = by_name["amp_scale"]
            tracker = e.input("amp_growth", torch.float64, ())
            inv = e.op("div", e.const(1.0, torch.float32), scale, name="inv_scale")
            grads = [e.op("mul", g, e.op("cast", inv, dtype=g.type.dtype)) for g in grads]
            nonfinite = any_nonfinite(e, grads)
        else:
            nonfinite = any_nonfinite(e, [loss_v] + grads)
        e.source = "truestar.fit gradient norm"
        if clip is not None:
            grads, norm = emit_call(e, _optim.clip_by_global_norm, grads, float(clip), None, self.clip_norm_type)
        else:
            norm = emit_call(e, _optim.global_norm, grads, self.clip_norm_type)
            if clip_value is not None:
                grads = emit_call(e, _optim.clip_by_value, grads, float(clip_value))
        k = emit_call(e, native_method(type(optimizer), "scalars"), optimizer, t, lr)
        new_p, new_slots = [], {s: [] for s in slots}
        for i, (pv, g) in enumerate(zip(pvals, grads)):
            q, sl = emit_call(e, native_method(type(optimizer), "leaf"), optimizer, pv, g,
                              {s: slot_vals[s][i] for s in slots}, k)
            if not q.type.same_as(pv.type):
                raise FitError(f"internal: the update of {pv.name} is {q.type}, the param is {pv.type}")
            new_p.append(q)
            for s in slots:
                new_slots[s].append(sl[s])
        carried_new = new_p + [v for s in slots for v in new_slots[s]]
        carried_old = pvals + [v for s in slots for v in slot_vals[s]]
        extra = []
        if scaled:
            from .train.step import _scale_update
            e.source = "truestar.fit skip the step on overflow"
            carried_new = [e.op("where", nonfinite, a, b) for a, b in zip(carried_old, carried_new)]
            extra = _scale_update(e, self.scaler, scale, tracker, nonfinite)
        state_out = list(G.outputs[(2 if scaled else 1):(2 if scaled else 1) + n_state_out])
        aux_out = list(G.outputs[(2 if scaled else 1) + n_state_out:(2 if scaled else 1) + n_state_out + n_aux])
        self._n_fwd = n_fwd
        if self.split:
            self.program = _prune(e.build([loss_v, norm, nonfinite] + carried_new + extra), f"update_{P.name}")
        else:
            self.program: Program = e.build([loss_v, norm, nonfinite] + carried_new + state_out + aux_out + extra)
        self._n_params, self._slots = len(pvals), slots
        self._n_state, self._n_aux = n_state_out, n_aux
        self._key_input = c.key_input
        self._n_leaf_inputs = n_leaf_inputs
        ident = self.program.to_text() + json.dumps(self.config(), sort_keys=True, default=str)
        if self.split:
            ident = G.to_text() + ident
        self.hash = hashlib.sha256(ident.encode("utf-8")).hexdigest()
        self.executor = _resolve_executor(executor)
        self._run = self.executor.prepare(self.program)
        self._grad_run = self.executor.prepare(G) if self.split else None

    # ------------------------------------------------------------------ state

    def init(self, params, model_state=None) -> TrainState:
        """The state before the first step: optimizer slots are zeros, the step count 0."""
        if self.state_name is not None and model_state is None:
            raise FitError(f"this step carries model state {self.state_name!r}: pass init(params, model_state)", "TS761")
        amp = None
        if self.scaler is not None:
            amp = {"scale": torch.tensor(float(self.scaler.init_scale), dtype=torch.float32),
                   "growth": torch.tensor(0.0, dtype=torch.float64)}
        return TrainState(params, self.optimizer.init(params), 0, model_state, amp)

    # ------------------------------------------------------------------ one step

    def _inputs(self, ts: TrainState, args, kwargs) -> List[torch.Tensor]:
        named = self._named_inputs(ts, args, kwargs)
        return [named[v.name] for v in self.program.inputs]

    def _named_inputs(self, ts: TrainState, args, kwargs) -> Dict[str, torch.Tensor]:
        fn = self.loss_fn
        call = dict(kwargs)
        call[self.params_name] = ts.params
        if self.state_name is not None:
            call[self.state_name] = ts.model_state
        call.update(self.static)
        try:
            bound = fn.py_signature.bind(*args, **call) if not args else self._bind_positional(args, call)
        except TypeError as e:
            raise FitError(f"bad batch for {fn.name}: {e}", "TS760") from None
        bound.apply_defaults()
        calls = fn.calls
        try:
            tensors = fn._flat_arguments(self.compiled_loss, bound)
        finally:
            fn.calls = calls
        if self._key_input:   # step n draws with call_key(seed, n), whatever ran before
            tensors[-1] = torch.tensor(rng.as_i64(rng.call_key(fn.seed, ts.step)), dtype=torch.int64)
        named = {v.name: x for v, x in zip(self.compiled_loss.program.inputs, tensors)}
        named["lr"] = torch.tensor(self.optimizer.lr_at(ts.step), dtype=torch.float64)
        named["t"] = torch.tensor(float(ts.step + 1), dtype=torch.float64)
        pnames = self._p_names
        for s in self._slots:
            leaves = pytree.match(self._param_tree, ts.opt[s], _is_t, f"opt[{s!r}]") if self._param_tree else [ts.opt[s]]
            for n, x in zip(pnames, leaves):
                named[f"opt_{s}_{n}"] = x
        if self.scaler is not None:
            named["amp_scale"] = ts.amp["scale"]
            named["amp_growth"] = ts.amp["growth"]
        return named

    def _bind_positional(self, args, call):
        """Bind the batch positionally to the loss parameters that are neither params nor state."""
        names = [n for n in self.loss_fn.py_signature.parameters
                 if n not in (self.params_name, self.state_name) and n not in self.static]
        if len(args) > len(names):
            raise TypeError(f"{len(args)} batch values for {len(names)} batch parameters {names}")
        for n, a in zip(names, args):
            if n in call:
                raise TypeError(f"{n} given twice")
            call[n] = a
        return self.loss_fn.py_signature.bind(**call)

    def _micro_batches(self, batch, batch_kw):
        """The local batch split into `accumulate` equal micro batches along dim 0."""
        k = self.accumulate
        if k == 1:
            return [(batch, batch_kw)]

        def split(x, name):
            if not isinstance(x, torch.Tensor) or x.dim() == 0:
                return [x] * k
            if x.shape[0] % k:
                raise FitError(f"batch value {name} has {x.shape[0]} rows, which does not split into accumulate={k} "
                               "equal micro batches (the mean of micro batch means is the batch mean only when "
                               "they are equal)", "TS823")
            return list(x.chunk(k, dim=0))
        cols = [split(x, f"#{i}") for i, x in enumerate(batch)]
        kws = {n: split(x, n) for n, x in batch_kw.items()}
        return [(tuple(c[m] for c in cols), {n: v[m] for n, v in kws.items()}) for m in range(k)]

    def _replicas_hold_equal_batches(self, batch, batch_kw) -> None:
        """TS1210 when the replicas hold batches of different row counts. The sum over replicas is
        divided by accumulate * world, which is the whole batch mean only when every replica holds
        the same rows (a batch of 32 split over 3 ranks gives 11, 11 and 10). One small all gather per
        step, before any compute, so every replica raises the same error and none waits in a sum."""
        named = [(f"#{i}", x) for i, x in enumerate(batch)] + list(batch_kw.items())
        named = [(n, x) for n, x in named if isinstance(x, torch.Tensor) and x.dim() > 0]
        if not named:
            return
        mine = torch.tensor([[x.shape[0] for _, x in named]], dtype=torch.int64)
        got = self.dp.all_gather(mine, tuple(range(self.dp.world)), 0, "batch rows per replica (equal shards check)")
        for r in range(1, got.shape[0]):
            bad = (got[r] != got[0]).nonzero()
            if bad.numel():
                i = int(bad[0])
                rows = [int(v) for v in got[:, i]]
                world, total = self.dp.world, sum(rows)
                fix = (f"give every replica {total // world} rows" if total % world == 0 else
                       f"{total} rows do not split into {world} equal shards: pad or drop {total % world} of them")
                raise FitError(f"replica {r} holds {rows[r]} rows of batch value {named[i][0]} but replica 0 holds "
                               f"{rows[0]} (rows per replica: {rows}). The data parallel step divides the sum over "
                               f"replicas by their count, which is the whole batch mean only for equal shards; {fix}",
                               "TS1210")

    def _run_split(self, ts: TrainState, batch, batch_kw):
        """Forward and backward per micro batch, sums over micro batches and replicas, then the update."""
        from .parallel.native import reduce_mean, replicas_agree

        if self.dp is not None and self.dp.world > 1:
            self._replicas_hold_equal_batches(batch, batch_kw)
        G = self.grad_program
        n_g = len(G.outputs) - self._n_fwd
        loss_sum, grad_sums, aux = None, None, None
        named = None
        for mb, (b, kw) in enumerate(self._micro_batches(batch, batch_kw)):
            named = self._named_inputs(ts, b, kw)
            gin = [named[v.name] for v in G.inputs]
            if getattr(self._grad_run, "binds_symbols", False):
                self._grad_run.bind(gin)
            outs = self._grad_run(gin)
            loss, grads = outs[0], list(outs[self._n_fwd:self._n_fwd + n_g])
            if mb == 0:
                loss_sum, grad_sums = loss.clone(), [g.clone() for g in grads]
                aux = [o.detach() for o in outs[1:self._n_fwd]]
            else:
                loss_sum = loss_sum + loss
                grad_sums = [a + g for a, g in zip(grad_sums, grads)]
        world = 1 if self.dp is None else self.dp.world
        count = float(self.accumulate * world)
        loss_m, grad_m = reduce_mean(self.dp, loss_sum, grad_sums, count, self.bucket_bytes)
        named["sync_loss"] = loss_m
        for i, g in enumerate(grad_m):
            named[f"sync_grad_{i}"] = g
        uin = [named[v.name] for v in self.program.inputs]
        if getattr(self._run, "binds_symbols", False):
            self._run.bind(uin)
        outs = self._run(uin)
        if self.dp is not None and self.check_replicas and (ts.step + 1) % self.check_replicas == 0:
            replicas_agree(self.dp, self._p_names, outs[3:3 + self._n_params], ts.step)
        return outs, aux, uin

    def __call__(self, ts: TrainState, *batch, **batch_kw) -> Tuple[TrainState, StepResult]:
        t0 = time.perf_counter()
        if self.split:
            outs, aux_split, inputs = self._run_split(ts, batch, batch_kw)
        else:
            inputs = self._inputs(ts, batch, batch_kw)
            if getattr(self._run, "binds_symbols", False):
                self._run.bind(inputs)
            outs = self._run(inputs)
        loss, norm, bad = outs[0], outs[1], bool(outs[2])
        k = 3
        new_carried = [o.to(v.type.dtype) for o, v in zip(outs[k:], self.program.outputs[k:])]
        npar = self._n_params
        lr = self.optimizer.lr_at(ts.step)
        if bad and self.scaler is None:
            from .train.trainer import first_nonfinite
            report = first_nonfinite(self.program, [x.detach().cpu() for x in inputs])
            raise NonFiniteStep(f"training step {ts.step}: the loss or a gradient is not finite (loss "
                                f"{float(loss):.4g}, grad norm {float(norm):.4g}); the step was not applied. "
                                f"{report.get('message', '')}", report)
        params = self._unflatten_params(new_carried[:npar])
        opt = {}
        for j, s in enumerate(self._slots):
            opt[s] = self._unflatten_params(new_carried[npar * (j + 1):npar * (j + 2)])
        k2 = npar * (1 + len(self._slots))
        model_state = ts.model_state
        if self._n_state:
            model_state = pytree.unflatten(self._state_tree, new_carried[k2:k2 + self._n_state])
        k3 = k2 + self._n_state
        aux = None
        if self._n_aux:
            leaves = aux_split if self.split else [o.detach() for o in outs[k + k3:k + k3 + self._n_aux]]
            aux = pytree.unflatten(self._aux_tree, leaves)
            aux = aux[0] if len(aux) == 1 else aux
        amp = ts.amp
        if self.scaler is not None:
            amp = {"scale": new_carried[-2], "growth": new_carried[-1]}
        new = TrainState(params, opt, ts.step + 1, model_state, amp)
        return new, StepResult(ts.step, float(loss), float(norm), lr, bool(bad), aux, time.perf_counter() - t0)

    def _unflatten_params(self, leaves):
        return pytree.unflatten(self._param_tree, leaves) if self._param_tree is not None else leaves[0]

    # ------------------------------------------------------------------ checkpoints

    def _named(self, ts: TrainState) -> Dict[str, torch.Tensor]:
        out = {}

        def add(prefix, tree, spec):
            leaves = pytree.match(spec, tree, _is_t, prefix) if spec is not None else [tree]
            paths = pytree.leaf_paths(spec, prefix) if spec is not None else [prefix]
            for pth, x in zip(paths, leaves):
                out[pth] = x.detach().to("cpu").contiguous()

        add("params", ts.params, self._param_tree)
        for s in self._slots:
            add(f"opt.{s}", ts.opt[s], self._param_tree)
        if self.state_name is not None:
            add("model_state", ts.model_state, self._state_tree)
        if ts.amp is not None:
            out["amp.scale"] = ts.amp["scale"].detach().cpu()
            out["amp.growth"] = ts.amp["growth"].detach().cpu()
        return out

    def config(self) -> Dict[str, Any]:
        cfg = {"loss": self.loss_fn.name, "optimizer": self.optimizer.config(), "clip": self.clip,
               "precision": self.precision, "params": self.params_name, "state": self.state_name}
        if self.clip_value is not None or self.clip_norm_type != 2.0:
            cfg["clip_value"], cfg["clip_norm_type"] = self.clip_value, self.clip_norm_type
        if self.split:
            cfg["accumulate"] = self.accumulate
            cfg["data_parallel"] = None if self.dp is None else self.dp.world
        return cfg

    def save(self, path: str, ts: TrainState) -> str:
        """A safetensors file with every carried tensor (params, optimizer slots, model state, loss
        scale) and, as metadata, the step count, the step program's hash and the configuration."""
        from .interop.safetensors import save_file

        meta = {"truestar.kind": "train_state", "truestar.step": str(int(ts.step)), "truestar.program": self.hash,
                "truestar.config": json.dumps(self.config(), sort_keys=True, default=str)}
        return save_file(self._named(ts), path, meta)

    def load(self, path: str, device=None) -> TrainState:
        """Read a checkpoint written by ``save`` of the same step program. Refuses another program
        (different model, loss, optimizer settings or precision) and names every missing, extra,
        misshapen or mistyped tensor at once."""
        from .interop.safetensors import LoadReport, load_file

        rep = LoadReport(path)
        data = load_file(path, mmap=False, report=rep)
        meta = rep.metadata or {}
        if meta.get("truestar.kind") != "train_state":
            raise FitError(f"{path} is not a truestar training checkpoint (no truestar.kind=train_state metadata)", "TS765")
        if meta.get("truestar.program") != self.hash:
            raise FitError(f"{path} was written by another step program (hash {meta.get('truestar.program', '?')[:12]}, "
                           f"this one {self.hash[:12]}); its config: {meta.get('truestar.config')}", "TS764")
        step = int(meta["truestar.step"])
        like = self._named(self._template())
        problems = []
        for k, ref in like.items():
            if k not in data:
                problems.append(f"missing {k}")
            elif data[k].dtype != ref.dtype or tuple(data[k].shape) != tuple(ref.shape):
                problems.append(f"{k}: {data[k].dtype} {list(data[k].shape)}, expected {ref.dtype} {list(ref.shape)}")
        problems += [f"unexpected {k}" for k in data if k not in like]
        if problems:
            raise FitError(f"{path} does not match this step: " + "; ".join(problems), "TS765")
        dev = device

        def get(k):
            x = data[k].clone()
            return x if dev is None else x.to(dev)

        def tree(prefix, spec):
            if spec is None:
                return get(prefix)
            return pytree.unflatten(spec, [get(pth) for pth in pytree.leaf_paths(spec, prefix)])

        params = tree("params", self._param_tree)
        opt = {s: tree(f"opt.{s}", self._param_tree) for s in self._slots}
        ms = tree("model_state", self._state_tree) if self.state_name is not None else None
        amp = {"scale": get("amp.scale"), "growth": get("amp.growth")} if "amp.scale" in data else None
        return TrainState(params, opt, step, ms, amp)

    def _template(self) -> TrainState:
        sig = self.compiled_loss.signature
        types = dict(sig.params)

        def zeros_tree(name, spec):
            if spec is None:
                t = types[name]
                return torch.zeros(tuple(int(d) for d in t.shape), dtype=t.dtype)
            paths = pytree.leaf_paths(spec, name)
            return pytree.unflatten(spec, [torch.zeros(tuple(int(d) for d in types[q].shape), dtype=types[q].dtype)
                                           for q in paths])

        p = zeros_tree(self.params_name, self._param_tree)
        ms = zeros_tree(self.state_name, self._state_tree) if self.state_name is not None else None
        return self.init(p, ms)


def train_step(loss_fn: NativeFunction, optimizer, *, params: str = "p", state: Optional[str] = None,
               clip: Optional[float] = None, precision: Optional[str] = None, loss_scale="auto", executor=None,
               static: Optional[Dict[str, Any]] = None, loop_memory: str = "auto", data_parallel=None,
               accumulate: int = 1, bucket_bytes: int = 25 << 20, check_replicas: int = 0,
               clip_value: Optional[float] = None, clip_norm_type: float = 2.0) -> Step:
    """Build the one program training step of ``loss_fn`` (see the module docstring).

    params: the loss parameter holding the trained tree. state: a loss parameter holding model
    state; the loss then returns ``(loss, new_state)``. clip: max global gradient norm (``clip_norm_type``
    2, 1 or inf, as torch's clip_grad_norm_). clip_value: clamp every gradient entry to +-clip_value
    instead (torch's clip_grad_value_; the reported grad_norm is taken before it).
    precision: "f32" (default), "bf16" or "f16" (autocast of linear, matmul, conv2d and attention,
    with dynamic loss scaling for f16). executor: "reference" (default, float64), "cpu", "cuda",
    "cuda:bfloat16" and so on, or an Executor. static: values of the loss function's static
    parameters (for example ``{"training": True}``).

    data_parallel: a communicator over the replicas (``truestar.parallel.devices.DeviceComm``, docs/TRAINING.md,
    Multi GPU): each rank passes its own rows, the step averages loss and gradients over ranks in fixed
    buckets before clipping and the update, so every replica takes the single device step of the whole
    batch. accumulate: split the batch into that many equal micro batches, one forward and backward each.
    check_replicas: every that many steps, compare parameter fingerprints across replicas (TS824)."""
    return Step(loss_fn, optimizer, params=params, state=state, clip=clip, precision=precision,
                loss_scale=loss_scale, executor=executor, static=static, loop_memory=loop_memory,
                data_parallel=data_parallel, accumulate=accumulate, bucket_bytes=bucket_bytes,
                check_replicas=check_replicas, clip_value=clip_value, clip_norm_type=clip_norm_type)


def fit(step: Step, state: TrainState, batches: Iterable, *, steps: Optional[int] = None, log_every: int = 0,
        checkpoint: Optional[str] = None, checkpoint_every: int = 0,
        on_step: Optional[Callable[[TrainState, StepResult], None]] = None) -> Tuple[TrainState, List[StepResult]]:
    """Run ``step`` over ``batches`` (each a tuple of positional batch values or a dict of keyword
    ones) for at most ``steps`` steps. Writes ``checkpoint`` every ``checkpoint_every`` steps and at
    the end when given. Returns the final state and one StepResult per step."""
    history: List[StepResult] = []
    for i, batch in enumerate(batches):
        if steps is not None and i >= steps:
            break
        if isinstance(batch, dict):
            state, r = step(state, **batch)
        else:
            state, r = step(state, *(batch if isinstance(batch, (tuple, list)) else (batch,)))
        history.append(r)
        if log_every and (r.step % log_every == 0):
            print(f"step {r.step:6d}  loss {r.loss:.6f}  grad norm {r.grad_norm:.4f}  lr {r.lr:.3g}"
                  + ("  skipped" if r.skipped else ""))
        if checkpoint and checkpoint_every and state.step % checkpoint_every == 0:
            step.save(checkpoint, state)
        if on_step is not None:
            on_step(state, r)
    if checkpoint:
        step.save(checkpoint, state)
    return state, history


__all__ = ["Step", "TrainState", "StepResult", "FitError", "NonFiniteStep", "train_step", "fit"]
