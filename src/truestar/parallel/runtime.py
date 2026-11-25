"""One training step on one simulated rank, and the single process reference step.

The rank step is written from the definitions, not from any framework: shard the Linear weights the
plan names, wrap those modules' forward with the collective their split needs, run `ga` micro
batches with every micro loss normalized by the global count (so uneven token counts stay right),
sum the gradients of sequence parallel weights over tp, sum every gradient over dp in one bucket,
clip by the global norm (sharded weights counted once per shard, replicated ones once), and step a
torch optimizer on the local pieces (SGD and AdamW are elementwise, so a shard steps exactly as the
same entries of the whole tensor would).

`faults` switches in one known bug each (docs/PARALLEL.md lists them with the issues they come
from). They exist to show what the check reports; a run without faults is the correct program.
"""

from __future__ import annotations

import dataclasses
import traceback
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .comm import Comm, FromRow, GatherOut, GatherSeq, ScatterSeqReduce, SplitSeq, ToColumn
from .layout import Layout, Plan, Style, is_sharded, shard

FAULTS: Dict[str, str] = {
    "dp_sum_of_means": "each replica divides its loss by its own count and the gradient all reduce sums "
                       "over dp: the step is dp times too large (step size class of Megatron-LM#5974, "
                       "verl#4710)",
    "dp_average_twice": "the gradient all reduce averages over dp although every loss is already divided "
                        "by the global count: the step is dp times too small",
    "ga_divided_twice": "micro losses are divided by ga and the accumulated gradients are divided by ga "
                        "again (the accumulation normalization class of transformers#34694, #35204)",
    "ga_mean_of_means": "each micro batch loss is a mean over its own count, then averaged over micro "
                        "batches: wrong when micro batches hold different token counts (transformers#34694)",
    "missing_tp_sync": "gradients of weights that run on a sequence shard (sequence parallel norms) are "
                       "not summed over tp (Megatron-LM#7452 class: composition bugs with no error)",
    "column_input_grad_not_reduced": "the input gradient of column layers is not summed over tp",
    "row_bias_every_rank": "row layers add their bias on every tp rank before the sum (tp times the bias)",
    "fused_split_as_block": "a weight fused from parts (gate and up, q k v) is split as one block, so a "
                            "rank holds whole parts instead of its slice of each",
    "row_split_dim0": "row layer weights split along the output dimension instead of the input dimension",
    "clip_local_norm": "gradient clipping uses each rank's local shard norm instead of the global norm "
                       "(pytorch#149768 class)",
    "shard_stale_params": "with sharded state (shard=1 or 2), each replica writes back only the slice it "
                          "updated and keeps its stale copy of the other slices (the parameter all gather "
                          "after the update is missing)",
}


class MSE:
    """Squared error summed over elements, divided by the element count."""

    def sum(self, out: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return ((out - tgt) ** 2).sum()

    def count(self, tgt: torch.Tensor) -> torch.Tensor:
        return torch.tensor(float(tgt.numel()), dtype=torch.float64)


class CrossEntropy:
    """Cross entropy over the last dim, summed over counted targets, divided by their number."""

    def __init__(self, ignore_index: int = -100) -> None:
        self.ignore_index = ignore_index

    def sum(self, out, tgt):
        return F.cross_entropy(out.reshape(-1, out.shape[-1]), tgt.reshape(-1), ignore_index=self.ignore_index,
                               reduction="sum")

    def count(self, tgt):
        return (tgt != self.ignore_index).sum().to(torch.float64)


@dataclasses.dataclass
class Call:
    out: List[torch.Tensor]
    grad: List[Optional[torch.Tensor]]
    hint: Tuple  # ("rep",) or ("split", dim)


@dataclasses.dataclass
class RankRecord:
    rank: int
    tp_rank: int
    dp_rank: int
    loss: float = float("nan")
    calls: Dict[str, List[Call]] = dataclasses.field(default_factory=dict)
    order: List[str] = dataclasses.field(default_factory=list)
    init: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    grads: Dict[str, Optional[torch.Tensor]] = dataclasses.field(default_factory=dict)
    updated: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    owner: Dict[str, Tuple[str, Optional[str], str]] = dataclasses.field(default_factory=dict)
    seq_partial: Set[str] = dataclasses.field(default_factory=set)
    clip_norm: Optional[float] = None
    log: list = dataclasses.field(default_factory=list)
    error: Optional[str] = None
    error_module: Optional[str] = None
    error_where: Optional[str] = None
    device: Optional[str] = None      # real ranks: "rank 1 on host cuda:1 (Tesla T4, nccl)"
    seconds: float = 0.0              # real ranks: wall time of this rank's step

    def strip_for_transfer(self) -> "RankRecord":
        self.log = [tuple(x) for x in self.log]
        return self


def _tensors(o) -> List[torch.Tensor]:
    if isinstance(o, torch.Tensor):
        return [o]
    if isinstance(o, (tuple, list)):
        return [t for x in o for t in _tensors(x)]
    if isinstance(o, dict):
        return [t for x in o.values() for t in _tensors(x)]
    return []


def _split_batch(x, n: int, i: int):
    if isinstance(x, torch.Tensor) and x.dim() > 0:
        if x.shape[0] % n:
            raise ValueError(f"batch of {x.shape[0]} rows does not split into {n} equal pieces")
        return x.chunk(n, dim=0)[i]
    return x


def make_optimizer(params, name: str, lr: float) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    raise ValueError(f"optimizer {name!r}: sgd or adamw")


class _Recorder:
    """Forward hooks on every module: outputs, their gradients, the call order, the module stack
    (for naming where an exception happened) and, under sequence parallel, the region state."""

    def __init__(self, model: nn.Module, rec: RankRecord, plan: Plan, layout: Layout) -> None:
        self.rec, self.plan, self.layout = rec, plan, layout
        self.stack: List[str] = []
        self.in_seq = False
        self.handles = []
        for name, m in model.named_modules():
            n = name or "<model>"
            self.handles.append(m.register_forward_pre_hook(lambda mod, a, n=n: self._pre(n, mod)))
            self.handles.append(m.register_forward_hook(lambda mod, a, o, n=n: self._post(n, mod, o)))

    def _pre(self, n, mod):
        self.stack.append((n, self.in_seq))
        st = self.plan.get(n)
        if st is not None and st.kind == "column":
            self.in_seq = False
        elif st is not None and st.kind == "row":
            if self.layout.sp and getattr(mod, "bias", None) is not None:
                self.rec.seq_partial.add(f"{n}.bias")  # added after the reduce scatter, on a shard
        elif self.in_seq and not list(mod.children()):
            for pn, _ in mod.named_parameters(recurse=False):
                self.rec.seq_partial.add(f"{n}.{pn}")

    def _post(self, n, mod, out):
        seq_at_pre = False
        if self.stack and self.stack[-1][0] == n:
            seq_at_pre = self.stack.pop()[1]
        st = self.plan.get(n)
        if self.layout.sp and st is not None:
            if st.kind in ("row", "scatter"):
                self.in_seq = True
            elif st.kind == "gather":
                self.in_seq = False
        if st is None and seq_at_pre and self.in_seq and list(mod.children()):
            for pn, _ in mod.named_parameters(recurse=False):  # a parent's own weights, used on the shard
                self.rec.seq_partial.add(f"{n}.{pn}")
        if st is not None and st.kind == "column":
            hint = ("split", -1, st.parts)
        elif self.in_seq:
            hint = ("split", self.layout.seq_dim)
        else:
            hint = ("rep",)
        ts = _tensors(out)
        call = Call([t.detach().clone() for t in ts], [None] * len(ts), hint)
        self.rec.calls.setdefault(n, []).append(call)
        self.rec.order.append(n)
        for k, t in enumerate(ts):
            if t.requires_grad:
                t.register_hook(lambda g, c=call, k=k: c.grad.__setitem__(k, g.detach().clone()))

    def remove(self):
        for h in self.handles:
            h.remove()


def _where(exc: BaseException) -> str:
    frames = traceback.extract_tb(exc.__traceback__)
    pkg = __file__.rsplit("parallel", 1)[0]
    user = [f for f in frames if "torch" not in f.filename.replace("\\", "/").split("/site-packages/")[-1][:6]
            and not f.filename.startswith(pkg)]
    f = (user or frames)[-1]
    return f"{f.filename}:{f.lineno} in {f.name}: {(f.line or '').strip()}"


def _install_plan(model: nn.Module, plan: Plan, layout: Layout, comm: Comm, rank: int, faults: Set[str],
                  rec: RankRecord) -> None:
    mods = dict(model.named_modules())
    tpr, tpg = layout.tp_rank(rank), layout.tp_group(rank)
    for name, st in plan.items():
        m = mods[name]
        if st.kind in ("column", "row"):
            for pn in ("weight", "bias"):
                p = getattr(m, pn, None)
                if p is None:
                    continue
                if st.kind == "column" and "fused_split_as_block" in faults:
                    local = shard(p.data, Style("column", 1), pn, layout.tp, tpr)
                elif st.kind == "row" and pn == "weight" and "row_split_dim0" in faults:
                    local = p.data.chunk(layout.tp, dim=0)[tpr].clone()
                else:
                    local = shard(p.data, st, pn, layout.tp, tpr)
                setattr(m, pn, nn.Parameter(local))
        if st.kind == "column":
            def fwd(x, m=m, name=name):
                if layout.sp:
                    x = GatherSeq.apply(x, comm, tpg, layout.seq_dim, f"{name} input all gather",
                                        "column_input_grad_not_reduced" not in faults)
                else:
                    x = ToColumn.apply(x, comm, tpg, f"{name} input gradient all reduce",
                                       "column_input_grad_not_reduced" not in faults)
                return F.linear(x, m.weight, m.bias)
            m.forward = fwd
        elif st.kind == "row":
            def fwd(x, m=m, name=name):
                bias_first = "row_bias_every_rank" in faults
                y = F.linear(x, m.weight, m.bias if bias_first else None)
                if layout.sp:
                    y = ScatterSeqReduce.apply(y, comm, tpg, layout.seq_dim, f"{name} output reduce scatter")
                else:
                    y = FromRow.apply(y, comm, tpg, f"{name} output all reduce")
                return y if (bias_first or m.bias is None) else y + m.bias
            m.forward = fwd
        elif st.kind in ("scatter", "gather"):
            orig = m.forward
            fn = SplitSeq if st.kind == "scatter" else GatherOut

            def fwd(*a, orig=orig, fn=fn, name=name, kind=st.kind, **k):
                return fn.apply(orig(*a, **k), comm, tpg, layout.seq_dim, f"{name} output {kind}")
            m.forward = fwd


def run_rank(comm: Comm, model: nn.Module, inputs: Sequence, targets, loss, layout: Layout, plan: Plan,
             optimizer: str, lr: float, clip: float, faults: Sequence[str], abort: Callable[[], None],
             sync: Optional[dict] = None) -> RankRecord:
    """One step on one rank. `sync` (real ranks): {"bucket_bytes": int, "overlap": bool} sums the dp
    gradients in fixed buckets started during the backward pass (gradsync.py); None keeps the one
    bucket after the backward pass that the simulated ranks use."""
    from .gradsync import GradientSync
    from .sharded import FlatShard

    rank = comm.rank
    faults = set(faults)
    rec = RankRecord(rank, layout.tp_rank(rank), layout.dp_rank(rank))
    recorder = None
    gsync = None
    try:
        _install_plan(model, plan, layout, comm, rank, faults, rec)
        mods = dict(model.named_modules())
        for pname, p in model.named_parameters():
            mname, _, short = pname.rpartition(".")
            st = plan.get(mname)
            rec.owner[pname] = (mname, st.text() if st else None, short)
            rec.init[pname] = p.detach().clone()
        dpg, tpg = layout.dp_group(rank), layout.tp_group(rank)
        dpr = layout.dp_rank(rank)
        dev = next((p.device for p in model.parameters()), torch.device("cpu"))
        shard_state = FlatShard(list(model.named_parameters()), dpr, layout.dp) if layout.shard else None
        if layout.shard == 2:
            # between steps this replica holds only its slice; the step gathers the whole for compute
            rest = shard_state.mine(shard_state.flat_params()).clone()
            shard_state.release_params()
            full = comm.all_gather(rest, dpg, 0, "parameter all gather over dp for compute (sharded parameters)")
            shard_state.write_params(full)
        if sync is not None and layout.dp > 1 and not layout.shard:
            gsync = GradientSync(list(model.named_parameters()), comm, dpg,
                                 bucket_bytes=int(sync.get("bucket_bytes", 25 << 20)),
                                 overlap=bool(sync.get("overlap", True)))
        recorder = _Recorder(model, rec, plan, layout)
        x_rank = [_split_batch(x, layout.dp, dpr) for x in inputs]
        y_rank = _split_batch(targets, layout.dp, dpr)
        local_count = loss.count(y_rank)
        global_count = comm.all_reduce(local_count, dpg, "loss count all reduce")
        total = torch.zeros((), dtype=torch.float64, device=dev)
        for mb in range(layout.ga):
            xs = [_split_batch(x, layout.ga, mb) for x in x_rank]
            ys = _split_batch(y_rank, layout.ga, mb)
            out = model(*xs)
            s = loss.sum(out, ys)
            if "ga_mean_of_means" in faults:
                micro = s / loss.count(ys) / (layout.ga * layout.dp)
            elif "dp_sum_of_means" in faults:
                micro = s / local_count
            else:
                micro = s / global_count
            if gsync is not None and mb < layout.ga - 1:
                with gsync.no_sync():
                    micro.backward()
            else:
                micro.backward()
            total = total + micro.detach()
        recorder.remove()
        recorder = None
        rec.loss = float(comm.all_reduce(total, dpg, "loss all reduce"))
        params = [(n, p) for n, p in model.named_parameters()]
        if gsync is not None:
            # buckets started during the last backward; sums over dp commute with the tp sum below
            gsync.finish(scale=(1.0 / layout.dp) if "dp_average_twice" in faults else None)
            gsync.remove()
        # 1. weights that ran on a sequence shard hold partial gradients: sum over tp
        if layout.sp and "missing_tp_sync" not in faults:
            for n, p in params:
                if n in rec.seq_partial and p.grad is not None:
                    p.grad.copy_(comm.all_reduce(p.grad, tpg, f"{n} gradient all reduce over tp (sp)"))
        if "ga_divided_twice" in faults and layout.ga > 1:
            for _, p in params:
                if p.grad is not None:
                    p.grad.div_(layout.ga)
        with_grad = [(n, p) for n, p in params if p.grad is not None]
        if layout.shard:
            _sharded_update(comm, rec, model, params, with_grad, shard_state, layout, plan, optimizer, lr, clip,
                            faults, tpg, dpg)
            rec.log = list(comm.log)
            return rec
        # 2. one bucket, one all reduce over dp
        if layout.dp > 1 and with_grad and gsync is None:
            flat = torch.cat([p.grad.reshape(-1) for _, p in with_grad])
            flat = comm.all_reduce(flat, dpg, f"gradient bucket all reduce over dp ({flat.numel()} entries)")
            if "dp_average_twice" in faults:
                flat = flat / layout.dp
            i = 0
            for _, p in with_grad:
                k = p.grad.numel()
                p.grad.copy_(flat[i:i + k].view_as(p.grad))
                i += k
        for n, p in params:
            rec.grads[n] = None if p.grad is None else p.grad.detach().clone()
        # 3. clipping by the global norm
        if clip > 0:
            sq = torch.zeros((), dtype=torch.float64, device=dev)
            for n, p in with_grad:
                mname, style, short = rec.owner[n]
                sharded = is_sharded(plan.get(mname), short)
                if sharded or rec.tp_rank == 0 or "clip_local_norm" in faults:
                    sq = sq + p.grad.double().pow(2).sum()
            if "clip_local_norm" not in faults:
                sq = comm.all_reduce(sq, tpg, "gradient norm all reduce over tp")
            norm = float(sq.sqrt())
            rec.clip_norm = norm
            coef = min(clip / (norm + 1e-6), 1.0)
            for _, p in with_grad:
                p.grad.mul_(coef)
        opt = make_optimizer([p for _, p in params], optimizer, lr)
        opt.step()
        for n, p in params:
            rec.updated[n] = p.detach().clone()
    except BaseException as exc:  # noqa: BLE001 - every failure is reported, then the others are released
        from .comm import CollectiveAborted

        if isinstance(exc, CollectiveAborted):
            rec.error = f"aborted: {exc}"
        else:
            rec.error = f"{type(exc).__name__}: {exc}"
            rec.error_module = recorder.stack[-1][0] if (recorder and recorder.stack) else None
            rec.error_where = _where(exc)
        abort()
    finally:
        if recorder is not None:
            recorder.remove()
        if gsync is not None:
            gsync.remove()
    rec.log = list(comm.log)
    return rec


def _sharded_update(comm, rec, model, params, with_grad, sh, layout, plan, optimizer, lr, clip, faults, tpg, dpg):
    """Steps 2 to 4 with state sharded over dp (sharded.py): reduce scatter, the norm of the slices,
    the optimizer on this replica's slice, all gather of the updated parameters."""
    gflat = sh.flat_grads()
    gs = comm.reduce_scatter(gflat, dpg, 0, f"gradient reduce scatter over dp ({gflat.numel()} entries, sharded "
                                            "state)")
    if "dp_average_twice" in faults:
        gs = gs / layout.dp
    full = comm.all_gather(gs, dpg, 0, "gradient all gather over dp (record only, not part of the step)")
    for (n, p), g in zip(sh.params, sh.pieces(full)):
        rec.grads[n] = None if p.grad is None else g.detach().clone()
    for n, p in params:
        rec.grads.setdefault(n, None)
    coef = 1.0
    if clip > 0:
        def counted(n):
            mname, _, short = rec.owner[n]
            return is_sharded(plan.get(mname), short) or rec.tp_rank == 0 or "clip_local_norm" in faults
        mask = sh.mine(sh.flat_mask(counted))
        sq = (gs.double().pow(2) * mask.double()).sum()
        sq = comm.all_reduce(sq, dpg, "gradient norm all reduce over dp (sharded state)")
        if "clip_local_norm" not in faults:
            sq = comm.all_reduce(sq, tpg, "gradient norm all reduce over tp")
        norm = float(sq.sqrt())
        rec.clip_norm = norm
        coef = min(clip / (norm + 1e-6), 1.0)
    pflat = sh.flat_params()
    piece = nn.Parameter(sh.mine(pflat).clone())
    piece.grad = gs * coef
    make_optimizer([piece], optimizer, lr).step()
    if "shard_stale_params" in faults:
        new = pflat.clone()
        new[sh.lo:sh.hi] = piece.detach()
    else:
        new = comm.all_gather(piece.detach(), dpg, 0, "parameter all gather over dp (sharded state)")
    if layout.shard == 1:
        sh.write_params(new.clone())
    for (n, _), t in zip(sh.params, sh.pieces(new)):
        rec.updated[n] = t.detach().clone()
    for n, p in params:
        if n not in rec.updated:
            rec.updated[n] = p.detach().clone()
    if layout.shard == 2:
        sh.release_params()  # only the slice `piece` stays until the next step gathers it


def run_reference(model: nn.Module, inputs: Sequence, targets, loss, optimizer: str, lr: float,
                  clip: float) -> RankRecord:
    """The single device program: the whole batch at once, plain torch, no collectives."""
    rec = RankRecord(0, 0, 0)
    for pname, p in model.named_parameters():
        mname, _, short = pname.rpartition(".")
        rec.owner[pname] = (mname, None, short)
        rec.init[pname] = p.detach().clone()
    recorder = _Recorder(model, rec, {}, Layout())
    try:
        out = model(*inputs)
        lval = loss.sum(out, targets) / loss.count(targets)
        lval.backward()
    finally:
        recorder.remove()
    rec.loss = float(lval.detach())
    params = list(model.named_parameters())
    for n, p in params:
        rec.grads[n] = None if p.grad is None else p.grad.detach().clone()
    if clip > 0:
        rec.clip_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in params if p.grad is not None], clip))
    make_optimizer([p for _, p in params], optimizer, lr).step()
    for n, p in params:
        rec.updated[n] = p.detach().clone()
    return rec
