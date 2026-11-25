"""One IR program per training step: forward, loss, backward, gradient averaging, clipping, the
optimizer update and loss scaling, with no Python between ops.

Layout of the step program (``TrainStep.program``):

| Part | Content |
|---|---|
| inputs | the model inputs, the loss targets, ``lr`` (f64, 0 dim), every trainable param (by name), every state value (``TrainStep.state_names``) |
| outputs | ``loss``, ``grad_norm`` (before clipping, after unscaling), ``nonfinite`` (bool: a NaN or Inf in the gradients, or in the loss), the new params, the new state in ``state_names`` order |

State is the optimizer state (``step``, moments) followed, with loss scaling, by ``amp.scale``
(f32) and ``amp.growth_tracker`` (f64). The caller feeds the new params and state back in as the
next step's inputs; :class:`~truestar.train.trainer.Trainer` does that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from ..ir.core import IRError, Program, Value
from . import ops
from .amp import AutocastPolicy, LossScaler, autocast, stochastic_round
from .distributed import all_reduce_mean
from .edit import Editor, _numel, inputs_to_group, params_to_inputs
from .losses import Loss, attach_loss
from .optim import ROW, Optimizer, any_nonfinite, clip_by_global_norm, global_norm

N_HEAD = 3  # loss, grad_norm, nonfinite


@dataclass
class TrainStep:
    program: Program
    batch_inputs: List[str]
    param_names: List[str]
    state_names: List[str]
    initial: Dict[str, torch.Tensor]
    optimizer: Optimizer
    loss: Loss
    clip_grad_norm: Optional[float] = None
    amp: Optional[AutocastPolicy] = None
    scaler: Optional[LossScaler] = None
    data_parallel: bool = False
    # param_group=True: every param is a view of one flat input; name -> (group, offset, shape)
    layout: Optional[Dict[str, Tuple[str, int, tuple]]] = None
    # the batch input holding the int64 key every random op reads (docs/RANDOM.md), when the model has one
    key_input: Optional[str] = None

    def unflatten(self, values: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Carried values by the original param names: with a param group, ``<param>`` and
        ``<param>.exp_avg`` (and so on) are views of the group's flat buffers; other values (the step
        count, loss scale) are passed through. Without a group, ``values`` unchanged."""
        if self.layout is None:
            return dict(values)
        grp = next(iter(self.layout.values()))[0]
        out = {k: t for k, t in values.items() if k != grp and not k.startswith(grp + ".")}
        for n, (_, off, shape) in self.layout.items():
            numel = 1
            for d in shape:
                numel *= d
            for k, t in values.items():
                if k == grp or k.startswith(grp + "."):
                    out[n + k[len(grp):]] = t[off:off + numel].view(shape)
        return out

    @property
    def carried(self) -> List[str]:
        """Names of the values fed back from one step to the next: params, then state."""
        return self.param_names + self.state_names

    def config(self) -> Dict[str, Any]:
        return {"optimizer": self.optimizer.config(), "loss": self.loss.config(),
                "clip_grad_norm": self.clip_grad_norm, "amp": self.amp.config() if self.amp else None,
                "scaler": self.scaler.config() if self.scaler else None, "data_parallel": self.data_parallel,
                "batch_inputs": self.batch_inputs, "params": self.param_names, "state": self.state_names}


def _resolve_params(model: Program, params) -> List[str]:
    if isinstance(params, torch.nn.Module):
        names = [n for n, p in params.named_parameters() if p.requires_grad]
    else:
        names = list(params)
    if not names:
        raise IRError("nothing to train: params is empty")
    return names


def _low_params(prog: Program, names: Sequence[str], dtype: torch.dtype) -> Program:
    """A copy of ``prog`` whose params ``names`` are declared (and stored) in ``dtype``; every reader
    reads a float32 cast of the low precision param, so the model's ops keep their types."""
    from ..ir.core import Node, TensorType
    repl: Dict[int, Value] = {}
    casts: List[Node] = []
    params, data = [], dict(prog.param_data)
    taken = {v.name for v in prog.inputs + prog.params} | {o.name for n in prog.nodes for o in n.outputs}
    for p in prog.params:
        if p.name not in names:
            params.append(p)
            continue
        low = Value(p.name, TensorType(dtype, p.type.shape), "param")
        params.append(low)
        data[p.name] = prog.param_data[p.name].to(dtype)
        wide = f"{p.name}_wide"
        while wide in taken:
            wide += "_"
        taken.add(wide)
        out = Value(wide, p.type, "op")
        casts.append(Node("cast", (low,), {"dtype": p.type.dtype}, out, "truestar.train bfloat16 weights"))
        repl[id(p)] = out
    nodes = casts + [Node(n.op, tuple(repl.get(id(x), x) if x is not None else None for x in n.inputs),
                          dict(n.attrs), n.output, n.source, n.extra_outputs)
                     if any(x is not None and id(x) in repl for x in n.inputs) else n for n in prog.nodes]
    out = Program(prog.name, list(prog.inputs), params, data, nodes,
                  [repl.get(id(v), v) for v in prog.outputs], dict(prog.symbols))
    out.verify()
    return out


def build_train_step(model: Program, loss: Loss, optimizer: Optimizer,
                     params: Union[Sequence[str], torch.nn.Module], *, clip_grad_norm: Optional[float] = None,
                     amp: Optional[AutocastPolicy] = None, scaler: Union[LossScaler, None, str] = "auto",
                     data_parallel: bool = False, name: Optional[str] = None, param_group: bool = False,
                     weights: str = "float32", sr_seed: int = 0) -> TrainStep:
    """Compose forward, loss, gradient and update into one program.

    model: a forward program (for example from ``ir.import_module``) whose outputs feed ``loss``.
    params: names of the params to train, or the ``nn.Module`` the program was imported from (its
      ``named_parameters`` with ``requires_grad``). Other params stay constants.
    clip_grad_norm: clip by global norm (after unscaling and after the data parallel average).
    amp: a mixed precision policy. The model's float32 params stay float32 master weights.
    scaler: dynamic loss scaling. "auto" means a default ``LossScaler()`` for float16 and none for
      bfloat16 or without ``amp``.
    data_parallel: average loss and gradients across ``torch.distributed`` ranks inside the program.
    param_group: keep every param in one flat buffer ``params.flat`` (the model reads views of it),
      concatenate the gradients once, and run the optimizer on the flat buffers: one kernel per
      updated value (the params, each moment) for the whole model, one norm and one non finite
      check, instead of a handful per param. The state is ``params.flat.exp_avg`` and so on;
      ``TrainStep.unflatten`` gives per param views. Needs one float dtype and fixed param shapes.
    weights: "float32" (master weights, the default) or "bfloat16_sr" (with a bfloat16 policy): the
      params are stored in bfloat16 and there is no float32 copy; the optimizer state stays float32,
      the update is computed in float32 from the bfloat16 weight and the gradient, and the new weight
      is rounded to bfloat16 stochastically (``stochastic_round``), keyed by the int64 program input
      ``sr.key`` (the caller passes ``truestar.rng.call_key(sr_seed, step)``; CompiledTrainStep
      does) and a site per param, so the rounding is the same on every backend and every run.
    """
    names = _resolve_params(model, params)
    if weights not in ("float32", "bfloat16_sr"):
        raise IRError(f"weights must be 'float32' or 'bfloat16_sr', not {weights!r}")
    sr = weights == "bfloat16_sr"
    if sr:
        if amp is None or amp.dtype != torch.bfloat16:
            raise IRError("weights='bfloat16_sr' stores bfloat16 weights: pass amp=AutocastPolicy(torch.bfloat16)")
        if param_group or data_parallel or clip_grad_norm is not None:
            raise IRError("weights='bfloat16_sr' does not support param_group, data_parallel or clip_grad_norm yet")
    if scaler == "auto":
        scaler = LossScaler() if amp is not None and amp.dtype == torch.float16 else None
    if scaler is not None and not scaler.enabled:
        scaler = None
    if scaler is not None and amp is None:
        raise IRError("loss scaling needs a mixed precision policy (amp=AutocastPolicy(...))")
    missing = [n for n in names if n not in model.param_data]
    if missing:
        raise IRError(f"{model.name} has no param(s) {missing}; params: {sorted(model.param_data)}")
    if sr:
        bad = [n for n in names if model.param_data[n].dtype != torch.float32]
        if bad:
            raise IRError(f"weights='bfloat16_sr' starts from float32 params; {bad} are not float32")
        model = _low_params(model, names, torch.bfloat16)
    initial = {n: model.param_data[n].detach().clone() for n in names}
    if amp is not None and not sr:
        low = [n for n in names if initial[n].dtype != torch.float32]
        if low:
            raise IRError(f"mixed precision keeps float32 master weights, but {low} are "
                          f"{[str(initial[n].dtype) for n in low]}; import a float32 model")

    g, target_names, n_fwd = loss_and_grad(model, loss, names, amp, scaled=scaler is not None)
    layout = None
    if param_group:
        group = "params.flat"
        g, offs = inputs_to_group(g, names, group, multiple=ROW)
        layout = {n: (group, o, s) for n, (o, s) in offs.items()}
        total = next(v for v in g.inputs if v.name == group).type.shape[0]
        flat0 = torch.cat([initial[n].reshape(-1) for n in names])
        initial = {group: torch.cat([flat0, flat0.new_zeros(total - flat0.numel())])}
    e = Editor(g, name or f"train_step_{model.name}")
    loss_v = g.outputs[n_fwd - 1]
    grads = list(g.outputs[n_fwd:])
    by_name = {v.name: v for v in g.inputs}
    if param_group:
        e.source = "truestar.train param group gradient"
        flat = [e.op("reshape", gr, shape=[_numel(gr.type.shape)], name=f"{n}.grad_flat")
                for n, gr in zip(names, grads)]
        tail = total - sum(_numel(gr.type.shape) for gr in grads)
        if tail:       # the group's padding has a zero gradient, so it stays zero in every update
            flat.append(e.op("full", shape=[tail], value=0.0, dtype=grads[0].type.dtype, name="params.flat.pad"))
        grads = [e.op("cat", *flat, dim=0, name="params.flat.grad")]
        names = [group]
    pvals = [by_name[n] for n in names]
    lr = e.input("lr", torch.float64, ())
    sr_key = e.input("sr.key", torch.int64, ()) if sr else None
    stored = pvals
    if sr:
        # the optimizer works on float32 views of the bfloat16 weights and on float32 gradients
        e.source = "truestar.train bfloat16 weights"
        pvals = [e.op("cast", p, dtype=torch.float32, name=f"{p.name}_f32") for p in stored]
        grads = [e.op("cast", gr, dtype=torch.float32, name=f"{gr.name}_f32") for gr in grads]
    spec = optimizer.state_spec(pvals)
    state = {s: e.input(s, t.dtype, t.shape) for s, t, _ in spec}
    state_names = [s for s, _, _ in spec]
    init_state = optimizer.init_state(pvals)
    scale = tracker = None
    if scaler is not None:
        scale = by_name["amp.scale"]
        tracker = e.input("amp.growth_tracker", torch.float64, ())
        state_names += ["amp.scale", "amp.growth_tracker"]
        init_state["amp.scale"] = torch.tensor(scaler.init_scale, dtype=torch.float32)
        init_state["amp.growth_tracker"] = torch.tensor(0.0, dtype=torch.float64)

    if data_parallel:
        reduced = all_reduce_mean(e, [loss_v] + grads)
        loss_v, grads = reduced[0], reduced[1:]
    if scaler is not None:
        e.source = "truestar.train unscale gradients"
        inv = e.op("div", e.const(1.0, torch.float32), scale, name="inv_scale")
        grads = [e.op("mul", gr, inv, name=f"{gr.name}_unscaled") for gr in grads]
        nonfinite = any_nonfinite(e, grads)
    else:
        nonfinite = any_nonfinite(e, [loss_v] + grads)
    norm = global_norm(e, grads)
    if clip_grad_norm is not None:
        grads, norm = clip_by_global_norm(e, grads, clip_grad_norm, norm)
    new_p, new_s = optimizer.emit(e, pvals, grads, state, lr)
    if sr:
        e.source = "truestar.train stochastic rounding"
        new_p = [stochastic_round(e, q, sr_key, f"truestar.train.sr.{n}") for n, q in zip(names, new_p)]
        pvals = stored
    new_state = [new_s[s] for s, _, _ in spec]
    if scaler is not None:
        e.source = "truestar.train skip the step on overflow"
        new_p = [e.op("where", nonfinite, p, q, name=f"{p.name}_kept") for p, q in zip(pvals, new_p)]
        new_state = [e.op("where", nonfinite, state[s], v, name=f"{s}_kept") for s, v in zip(state_names, new_state)]
        new_state += _scale_update(e, scaler, scale, tracker, nonfinite)
    for s, v in zip(state_names, new_state):
        want = state[s].type if s in state else (scale.type if s == "amp.scale" else tracker.type)
        if not v.type.same_as(want):
            raise IRError(f"internal error: new state {s} is {v.type}, its input is {want}")
    for p, q in zip(pvals, new_p):
        if not q.type.same_as(p.type):
            raise IRError(f"internal error: new {p.name} is {q.type}, the param is {p.type}")
    batch = [v.name for v in model.inputs] + target_names
    order = batch + ["lr"] + (["sr.key"] if sr else []) + names + state_names
    by_input = {v.name: v for v in e.inputs}
    if sorted(order) != sorted(by_input):
        raise IRError(f"internal error: step inputs {sorted(by_input)} do not match the layout {order}")
    e.inputs = [by_input[n] for n in order]
    prog = e.build([loss_v, norm, nonfinite] + list(new_p) + list(new_state))
    initial.update(init_state)
    ts = TrainStep(prog, batch, names, state_names, initial, optimizer, loss, clip_grad_norm, amp, scaler,
                   data_parallel, layout, getattr(model, "key", None))
    ts.sr_seed = sr_seed if sr else None   # weights="bfloat16_sr": the key of step k is rng.call_key(sr_seed, k)
    return ts


def loss_and_grad(model: Program, loss: Loss, names: Sequence[str], amp: Optional[AutocastPolicy] = None,
                  scaled: bool = False):
    """The forward, loss and backward part of the step as its own program.

    inputs: the model inputs, the loss targets, the params ``names``, and ``amp.scale`` when
    ``scaled``. outputs: the loss that is differentiated (the scaled loss when ``scaled``), the
    unscaled loss when ``scaled``, then one gradient per name (of the differentiated loss).
    Returns the program, the target input names and the number of loss outputs."""
    fwd = autocast(model, amp) if amp is not None else model
    lp, target_names = attach_loss(fwd, loss, cast_outputs_to=torch.float32 if amp is not None else None)
    lp = params_to_inputs(lp, names)
    if scaled:
        e = Editor(lp)
        scale = e.input("amp.scale", torch.float32, ())
        e.source = "truestar.train loss scaling"
        s = e.op("mul", lp.outputs[0], scale, name="scaled_loss")
        lp = e.build([s, lp.outputs[0]])
    return ops.grad(lp, list(names), of=[0]), target_names, len(lp.outputs)


def _scale_update(e: Editor, s: LossScaler, scale: Value, tracker: Value, found: Value) -> List[Value]:
    e.source = "truestar.train loss scale update"
    f32 = torch.float32
    nxt = e.op("add", tracker, e.const(1.0, tracker.type.dtype), name="growth_tracker_next")
    grow = e.op("eq", nxt, e.const(float(s.growth_interval), tracker.type.dtype), name="scale_grow")
    grown = e.op("mul", scale, e.const(s.growth_factor, f32), name="scale_grown")
    # finite in float32 (GradScaler's rule), written as grown - grown == 0 (inf - inf is NaN) so no
    # float32 max constant is needed: a 16 bit working precision would saturate it (mixed=True)
    grown_ok = e.op("eq", e.op("sub", grown, grown), e.const(0.0, f32))
    grown = e.op("where", grown_ok, grown, scale)
    backed = e.op("mul", scale, e.const(s.backoff_factor, f32), name="scale_backoff")
    new_scale = e.op("where", found, backed, e.op("where", grow, grown, scale), name="amp_scale_new")
    zero = e.const(0.0, tracker.type.dtype)
    new_tracker = e.op("where", found, zero, e.op("where", grow, zero, nxt), name="amp_growth_tracker_new")
    return [new_scale, new_tracker]
