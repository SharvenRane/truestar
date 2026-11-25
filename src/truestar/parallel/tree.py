"""Tensor parallel steps bitwise equal to the one device step: one canonical reduction tree.

A tensor parallel step differs from the one device step only in the order of its sums: a row layer
adds `tp` partial products that each rank formed over its slice of the input features, a column
layer adds `tp` partial input gradients. Floating point addition is not associative, so the usual
check (docs/PARALLEL.md) compares in float64 within a bound. Here every contraction of a linear
layer, in both arms, walks one canonical binary tree that depends only on the full shapes:

* the reduction axis of length L (L per part for a weight fused from P parts) is cut into W equal
  blocks (`ways` W, odd: 1 unless tp has an odd factor) and each block into G equal contiguous
  groups, G the largest power of two up to `cap` that divides L / W: W * G groups in all;
* a group sum is a pairwise tree over its products (`leaf="exact"`: products are single IEEE
  multiplies, pairs are (2i, 2i+1), an odd last element is carried up), or one matmul over the
  group (`leaf="block"`, faster, equal between arms only when the library kernel's result for one
  element does not depend on the other dimensions);
* the W * G group sums are added by the pairwise tree, and the P part sums after them. The first
  log2(G) levels of that tree pair within a block (their counts are even), so it sums each block's
  G groups and then the W block sums.

A rank of a layout tp = W * 2^a (2^a dividing G) holds G / 2^a aligned groups of a split axis inside
one block, so its pairwise tree over them is exactly a subtree of the canonical tree; the cross rank
step all gathers the tp subtree roots and finishes the same tree (its first a levels complete each
block, then the W block sums). The one device step uses the same W: tree_check gives it the W of
the tp it is compared with, so tp=3 and tp=6 equal the one device step with W=3, and every power of
two tp equals the one with W=1 (the default, the tree before W existed). Forward of row layers, backward input
gradients of column layers, weight and bias gradients (over the tokens, which tp does not split)
and the clipping norm (squares in float64, grouped along the shard axis) all use it, so the sharded
step and the one device step perform the same additions in the same order: bitwise equal by
construction, for every tp = W * 2^a whose 2^a divides G, on any device whose elementwise float multiply and add
are IEEE round to nearest (with `leaf="exact"`).

Scope: tp layouts (dp=1, ga=1, sp=0). Data parallel and gradient accumulation split the batch, so
their equality is a batch invariance question for every op of the model (norms, attention), not
only for the linear layers; it is not claimed here.

Codes: TS1100 (bitwise equal), TS1101 (a value differs, named), TS1102 (layout or shapes the tree
cannot split), TS1103 (the plain step, F.linear and rank order sums, differs: recorded for contrast).
"""

from __future__ import annotations

import copy
import dataclasses
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .comm import Comm, ThreadComm, _Hub
from .layout import Layout, Plan, Style, is_sharded, shard

LEAVES = ("exact", "tiled", "block")


class TreeLayoutError(ValueError):
    """TS1102: a layout or a layer the canonical tree cannot split."""


# ----------------------------------------------------------------------------- the tree


def odd_part(n: int) -> int:
    """n without its factors of two: the W of a tp layout (3 for tp=3 and tp=6, 1 for powers of two)."""
    while n > 1 and n % 2 == 0:
        n //= 2
    return n


def axis_groups(n: int, cap: int, ways: int = 1) -> int:
    """Canonical group count of a reduction axis of length n: `ways` blocks, each of the largest power
    of two up to `cap` that divides n / ways. 0 when n does not split into `ways` equal blocks."""
    if n % ways:
        return 0
    return ways * group_count(n // ways, cap)


def group_count(n: int, cap: int) -> int:
    """Largest power of two up to `cap` that divides `n` (1 if `n` is odd)."""
    g = 1
    while g * 2 <= cap and n % (g * 2) == 0:
        g *= 2
    return g


def ptree(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Pairwise sum along `dim`: level by level, entries (2i, 2i+1) added, an odd last one carried."""
    dim = dim % t.dim()
    while t.shape[dim] > 1:
        n = t.shape[dim]
        e = n - (n % 2)
        pairs = t.narrow(dim, 0, e).unflatten(dim, (e // 2, 2))
        s = pairs.select(dim + 1, 0) + pairs.select(dim + 1, 1)
        if n % 2:
            s = torch.cat([s, t.narrow(dim, e, 1)], dim=dim)
        t = s
    return t.squeeze(dim)


def _rows_per_chunk(m: int, k: int, n: int, budget_elems: int = 1 << 24) -> int:
    return max(1, min(m, budget_elems // max(1, k * n)))


def canon_tile(full: int, cap: int, tile_cap: int = 64) -> int:
    """Tile edge for one output dimension of a contraction, from its full (unsharded) length: a power
    of two up to `tile_cap` that divides full // cap, so it divides every rank's piece too."""
    return group_count(full // cap if full % cap == 0 else full, tile_cap)


def contract_groups(a: torch.Tensor, b: torch.Tensor, parts: int, groups: int, leaf: str,
                    tile: Tuple[int, int] = (0, 0)) -> torch.Tensor:
    """a [M, L], b [L, N], the L axis laid out as `parts` equal blocks each cut into `groups` equal
    groups: returns the group sums [parts, groups, M, N]. Leaves: exact (pairwise tree of single
    multiplies), block (one matmul per group, shapes as they come), tiled (one batched matmul whose
    entries are the canonical output tiles `tile` = (rows, cols) of every group, contiguous, so every
    entry has the same shape and layout in every arm; only the batch count differs)."""
    m, length = a.shape
    n = b.shape[1]
    per = length // parts
    gs = per // groups
    if leaf == "tiled":
        tr, tc = tile
        out = []
        for p in range(parts):
            ap = a[:, p * per:(p + 1) * per].reshape(m // tr, tr, groups, gs).permute(2, 0, 1, 3).contiguous()
            bp = b[p * per:(p + 1) * per].reshape(groups, gs, n // tc, tc).permute(0, 2, 1, 3).contiguous()
            c = torch.matmul(ap.unsqueeze(2), bp.unsqueeze(1))          # [G, M/tr, N/tc, tr, tc]: one call
            out.append(c.permute(0, 1, 3, 2, 4).reshape(groups, m, n))
        return torch.stack(out)
    if leaf == "exact":
        from . import tree_kernel

        if tree_kernel.usable(a, b, gs):
            chunks = tree_kernel.chunk_group_sums(a, b, parts, groups)     # first levels, in registers
            return ptree(chunks, 2) if chunks.shape[2] > 1 else chunks[:, :, 0]
    out = a.new_empty((parts, groups, m, n))
    for p in range(parts):
        for g in range(groups):
            lo = p * per + g * gs
            ag, bg = a[:, lo:lo + gs], b[lo:lo + gs]
            if leaf == "block":
                out[p, g] = ag @ bg
                continue
            step = _rows_per_chunk(m, gs, n)
            for r in range(0, m, step):
                prod = ag[r:r + step].unsqueeze(2) * bg.unsqueeze(0)   # [rows, gs, N], one rounding each
                out[p, g, r:r + step] = ptree(prod, 1)
    return out


def tree_rows(t: torch.Tensor, groups: int) -> torch.Tensor:
    """Sum of a [M, ...] tensor over its rows by the canonical tree (groups of rows, then groups)."""
    m = t.shape[0]
    gs = m // groups
    return ptree(torch.stack([ptree(t[g * gs:(g + 1) * gs], 0) for g in range(groups)]), 0)


@dataclasses.dataclass
class TreeSpec:
    """How one nn.Linear reduces: role full (one device, or replicated), column or row."""
    role: str = "full"
    parts: int = 1                       # output parts of a fused column weight
    k_groups: int = 1                    # canonical groups of the input features
    n_groups: int = 1                    # canonical groups per output part
    cap: int = 8
    ways: int = 1                        # odd block count W of the split axes (tp's odd factor)
    leaf: str = "exact"
    tk: int = 1                          # canonical tile edges of the input and output features
    tn: int = 1
    tp: int = 1
    comm: Optional[Comm] = None
    group: Tuple[int, ...] = (0,)
    name: str = ""

    def gather(self, t: torch.Tensor, what: str) -> torch.Tensor:
        return self.comm.all_gather(t.unsqueeze(0).contiguous(), self.group, 0, f"{self.name} {what} (tree)")


class _TreeLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, spec: TreeSpec):
        ctx.spec = spec
        ctx.save_for_backward(x, w)
        ctx.has_bias = b is not None
        x2 = x.reshape(-1, x.shape[-1])
        tm = canon_tile(x2.shape[0], 1)
        if spec.role == "row":
            local = contract_groups(x2, w.t(), 1, spec.k_groups // spec.tp, spec.leaf, (tm, spec.tn))
            root = ptree(local[0], 0)
            y = ptree(spec.gather(root, "row output partial sums"), 0)
        else:
            y = ptree(contract_groups(x2, w.t(), 1, spec.k_groups, spec.leaf, (tm, spec.tn))[0], 0)
        if b is not None:
            y = y + b
        return y.reshape(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, gy):
        spec = ctx.spec
        x, w = ctx.saved_tensors
        x2 = x.reshape(-1, x.shape[-1])
        g2 = gy.reshape(-1, gy.shape[-1]).contiguous()
        mg = group_count(g2.shape[0], spec.cap)
        tm = canon_tile(g2.shape[0], 1)
        gx = gw = gb = None
        if ctx.needs_input_grad[0]:
            if spec.role == "column":
                local = contract_groups(g2, w, spec.parts, spec.n_groups // spec.tp, spec.leaf, (tm, spec.tk))
                roots = ptree(local, 1)                                     # [P, M, K]
                whole = spec.gather(roots, "column input gradient partial sums (backward)")
                gx = ptree(ptree(whole, 0), 0)
            else:
                gx = ptree(ptree(contract_groups(g2, w, spec.parts, spec.n_groups, spec.leaf, (tm, spec.tk)), 1), 0)
            gx = gx.reshape(x.shape)
        if ctx.needs_input_grad[1]:
            gw = ptree(contract_groups(g2.t().contiguous(), x2, 1, mg, spec.leaf, (spec.tn, spec.tk))[0], 0)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            gb = tree_rows(g2, mg)
        return gx, gw, gb, None


def tree_linear(x, w, b, spec: TreeSpec):
    return _TreeLinear.apply(x, w, b, spec)


# ----------------------------------------------------------------------------- installing


def _split_error(name: str, what: str, n: int, groups: int, tp: int, cap: int, ways: int) -> str:
    if groups == 0:
        return (f"[TS1102] {name}: {n} {what} do not split into {ways} equal blocks (tp={tp} has the odd factor "
                f"{ways}); use a width divisible by {ways}")
    return (f"[TS1102] {name}: {n} {what} give {groups} canonical groups ({ways} block(s) of {groups // ways}, "
            f"cap {cap}), which tp={tp} does not divide; raise cap or use a width with a larger power of two factor")


def _check_split(name: str, st: Style, mod: nn.Linear, tp: int, cap: int, ways: int = 1) -> Tuple[int, int]:
    """Canonical group counts of the input and output features. Only the split axis of a sharded
    layer is cut into `ways` blocks; replicated layers keep the tree they have at tp=1."""
    kin, kout = mod.in_features, mod.out_features // st.parts
    kg = axis_groups(kin, cap, ways if st.kind == "row" else 1)
    ng = axis_groups(kout, cap, ways if st.kind == "column" else 1)
    if st.kind == "row" and (kg == 0 or kg % tp):
        raise TreeLayoutError(_split_error(name, "input features", kin, kg, tp, cap, ways))
    if st.kind == "column" and (ng == 0 or ng % tp):
        raise TreeLayoutError(_split_error(name, "output features per part", kout, ng, tp, cap, ways))
    return kg, ng


def install(model: nn.Module, plan: Plan, *, tp: int = 1, rank: int = 0, comm: Optional[Comm] = None,
            group: Tuple[int, ...] = (0,), cap: int = 8, leaf: str = "exact",
            ways: Optional[int] = None) -> Dict[str, TreeSpec]:
    """Route every nn.Linear of `model` through the canonical tree. With tp > 1 the plan's column and
    row layers are sharded for `rank` (the pieces truestar.parallel.layout.shard gives) and their
    cross rank sums finish the tree; tp=1 is the one device program (the plan still sets the fused
    part count, which is part of the canonical tree). `ways` is the odd block count W of the split
    axes, by default tp's odd factor; the one device run compared with tp=3 passes ways=3."""
    if leaf not in LEAVES:
        raise ValueError(f"leaf {leaf!r}: one of {LEAVES}")
    ways = odd_part(tp) if ways is None else int(ways)
    if tp < 1 or ways < 1 or odd_part(ways) != ways or (tp > 1 and odd_part(tp) != ways):
        raise TreeLayoutError(f"[TS1102] tp={tp} with ways={ways}: a tp = W * 2^a layout needs W = "
                              f"{odd_part(max(tp, 1))} odd blocks per split axis")
    specs: Dict[str, TreeSpec] = {}
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        st = plan.get(name) or Style("replicate")
        st_parts = st.parts if st.kind == "column" else 1
        kg, ng = _check_split(name, Style(st.kind, st_parts), m, tp, cap, ways)
        role = st.kind if (tp > 1 and st.kind in ("column", "row")) else "full"
        tn = group_count(canon_tile(m.out_features // st_parts, cap), 64)
        spec = TreeSpec(role, st_parts, kg, ng, cap, ways, leaf, canon_tile(m.in_features, cap), tn, tp, comm, group,
                        name)
        if role != "full":
            for pn in ("weight", "bias"):
                p = getattr(m, pn, None)
                if p is not None:
                    setattr(m, pn, nn.Parameter(shard(p.data, st, pn, tp, rank)))

        def fwd(x, m=m, spec=spec):
            return tree_linear(x, m.weight, m.bias, spec)   # a row bias is added once, after the finished sum
        m.forward = fwd
        specs[name] = spec
    return specs


def tree_sq_norm(named_grads: Sequence[Tuple[str, torch.Tensor]], plan: Plan, *, tp: int = 1,
                 comm: Optional[Comm] = None, group: Tuple[int, ...] = (0,), cap: int = 8,
                 ways: int = 1) -> torch.Tensor:
    """Squared global gradient norm in float64 by the canonical tree: per parameter, squares grouped
    along its shard axis (parts, then groups, each group flattened row major), then the parameters in
    registration order. A rank computes its subtree roots and finishes them with its tp peers."""
    vals = []
    for pname, g in named_grads:
        mname, _, short = pname.rpartition(".")
        st = plan.get(mname)
        sq = g.double() ** 2
        if st is None or not is_sharded(st, short):
            vals.append(ptree(sq.reshape(-1), 0))
            continue
        axis = 0 if st.kind == "column" else 1
        parts = st.parts if st.kind == "column" else 1
        full_len = sq.shape[axis] * tp                      # length of the split axis in the whole tensor
        groups = axis_groups(full_len // parts, cap, ways)
        local_groups = groups // tp
        per = sq.shape[axis] // parts
        gs = per // local_groups
        roots = []
        for p in range(parts):
            gsum = [ptree(sq.narrow(axis, p * per + j * gs, gs).reshape(-1), 0) for j in range(local_groups)]
            roots.append(ptree(torch.stack(gsum), 0))
        roots_t = torch.stack(roots)                         # [P]
        if tp > 1:
            whole = comm.all_gather(roots_t.unsqueeze(0), group, 0, f"{pname} gradient norm partial sums (tree)")
            roots_t = ptree(whole, 0)
        vals.append(ptree(roots_t, 0))
    if not vals:
        return torch.zeros((), dtype=torch.float64)
    return ptree(torch.stack(vals), 0)


# ----------------------------------------------------------------------------- one step


@dataclasses.dataclass
class StepRecord:
    loss: Optional[torch.Tensor] = None
    out: Optional[torch.Tensor] = None
    grads: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    updated: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    norm: Optional[float] = None
    error: Optional[str] = None
    seconds: float = 0.0
    device: str = ""


def _make_opt(params, name: str, lr: float):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, foreach=False)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01, foreach=False)
    raise ValueError(f"optimizer {name!r}: sgd or adamw")


def run_step(model: nn.Module, inputs: Sequence, targets, loss, plan: Plan, *, optimizer: str = "adamw",
             lr: float = 1e-2, clip: float = 0.0, tree: bool = True, tp: int = 1, rank: int = 0,
             comm: Optional[Comm] = None, group: Tuple[int, ...] = (0,), cap: int = 8,
             leaf: str = "exact", ways: Optional[int] = None) -> StepRecord:
    """One training step (forward, loss sum over its count, backward, optional global norm clipping,
    optimizer step). `tree=False` is the plain step: F.linear and, under tp, the rank order sums of
    truestar.parallel.runtime (for contrast with the tree)."""
    rec = StepRecord()
    t0 = time.perf_counter()
    ways = odd_part(tp) if ways is None else int(ways)
    if tree:
        install(model, plan, tp=tp, rank=rank, comm=comm, group=group, cap=cap, leaf=leaf, ways=ways)
    elif tp > 1:
        from .runtime import RankRecord, _install_plan
        _install_plan(model, plan, Layout(tp=tp), comm, rank, set(), RankRecord(rank, rank, 0))
    out = model(*inputs)
    lval = loss.sum(out, targets) / loss.count(targets).to(out.dtype)
    lval.backward()
    params = [(n, p) for n, p in model.named_parameters()]
    rec.loss, rec.out = lval.detach().clone(), out.detach().clone()
    for n, p in params:
        if p.grad is not None:
            rec.grads[n] = p.grad.detach().clone()
    if clip > 0:
        with_grad = [(n, p.grad) for n, p in params if p.grad is not None]
        if tree:
            # one device: the plan's grouping with tp=1 is one rank holding every group, the same tree
            sq = tree_sq_norm(with_grad, plan, tp=tp, comm=comm, group=group, cap=cap, ways=ways)
        else:
            sq = sum(g.double().pow(2).sum() for _, g in with_grad)
            if tp > 1:
                sq = comm.all_reduce(sq, group, "gradient norm all reduce over tp")
        rec.norm = float(sq.sqrt())
        coef = min(clip / (rec.norm + 1e-6), 1.0)
        for _, p in params:
            if p.grad is not None:
                p.grad.mul_(coef)
    _make_opt([p for _, p in params], optimizer, lr).step()
    for n, p in params:
        rec.updated[n] = p.detach().clone()
    rec.seconds = time.perf_counter() - t0
    return rec


def _run_threads(model, inputs, targets, loss, plan, tp, kw, timeout=120.0) -> List[StepRecord]:
    """Ranks as threads of this process. CPU only: CUDA backward passes of all threads run on one
    autograd engine thread per device, so a rank's backward collective would wait forever for a peer
    whose backward is queued behind it. CUDA ranks run as processes (ranks="gloo")."""
    if any(p.is_cuda for p in model.parameters()):
        raise ValueError("thread ranks are CPU only (one CUDA autograd thread per device serializes the "
                         "ranks' backward passes); use ranks='gloo' with a model file")
    hub = _Hub(timeout)
    recs: List[Optional[StepRecord]] = [None] * tp

    def body(r):
        try:
            comm = ThreadComm(hub, r, tp)
            recs[r] = run_step(copy.deepcopy(model), inputs, targets, loss, plan, tp=tp, rank=r, comm=comm,
                               group=tuple(range(tp)), **kw)
        except BaseException as exc:  # noqa: BLE001 - reported per rank, the peers released
            recs[r] = StepRecord(error=f"{type(exc).__name__}: {exc}")
            hub.abort()

    ths = [threading.Thread(target=body, args=(r,), daemon=True) for r in range(tp)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout + 10)
    return [r if r is not None else StepRecord(error="the rank did not finish") for r in recs]


class HostGlooComm(Comm):
    """torch.distributed gloo between processes (or machines); tensors cross through host memory, so
    a CUDA rank works with gloo. Only all_gather is used by the tree."""

    def __init__(self, rank: int, world: int) -> None:
        super().__init__()
        self.rank, self.world = rank, world

    def all_gather(self, t, group, dim, purpose):
        import torch.distributed as dist

        self._note("all_gather", purpose, group, t)
        h = t.detach().contiguous().cpu()
        bufs = [torch.empty_like(h) for _ in group]
        dist.all_gather(bufs, h)
        return torch.cat(bufs, dim=dim).to(t.device)

    def all_reduce(self, t, group, purpose):   # the plain step's rank order sum, for contrast only
        g = self.all_gather(t.unsqueeze(0), group, 0, purpose)
        acc = g[0].clone()
        for x in g[1:]:
            acc = acc + x
        return acc


def _rank_record(rec: StepRecord) -> StepRecord:
    cpu = lambda d: {k: v.cpu() for k, v in d.items()}  # noqa: E731
    return StepRecord(rec.loss.cpu(), rec.out.cpu(), cpu(rec.grads), cpu(rec.updated), rec.norm, rec.error,
                      rec.seconds, rec.device)


def gloo_rank(rank: int, world: int, payload_path: str, init_method: str, out_path: str,
              device: Optional[str] = None, timeout: float = 300.0) -> StepRecord:
    """One tree rank as its own process: builds the model from the payload's model file, loads its
    weights, joins gloo at `init_method` (file:// or tcp://host:port), runs the step, saves the
    record to `out_path` (CPU tensors). `device` overrides the payload's (a machine's own GPU)."""
    import datetime

    import torch.distributed as dist

    from ..cli import _load

    pay = torch.load(payload_path, weights_only=False)
    dev = device or pay["device"]
    try:
        model, _, _ = _load(pay["model_file"])
        if str(dev).startswith("cuda"):
            torch.cuda.set_device(torch.device(dev))
        model = model.to(device=dev, dtype=pay["dtype"])
        model.load_state_dict(pay["state"])
        dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world,
                                timeout=datetime.timedelta(seconds=timeout))
        mv = lambda x: x.to(dev) if torch.is_tensor(x) else x  # noqa: E731
        comm = HostGlooComm(rank, world)
        rec = run_step(model, [mv(x) for x in pay["inputs"]], mv(pay["targets"]), pay["loss"], pay["plan"],
                       tp=world, rank=rank, comm=comm, group=tuple(range(world)), **pay["kw"])
        rec.device = (f"{dev} ({torch.cuda.get_device_name(torch.device(dev))})" if str(dev).startswith("cuda")
                      else str(dev))
        rec = _rank_record(rec)
    except BaseException as exc:  # noqa: BLE001 - reported by the parent
        rec = StepRecord(error=f"{type(exc).__name__}: {exc}")
    torch.save(rec, out_path)
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass
    return rec


def _spawned_rank(rank, world, payload_path, init_method, directory, device=None):
    import os
    import sys

    rec = gloo_rank(rank, world, payload_path, init_method, os.path.join(directory, f"rank{rank}.pt"), device)
    sys.stdout.flush()
    os._exit(0 if rec.error is None else 3)


def write_payload(path: str, model_file: str, model: nn.Module, inputs, targets, loss, plan, kw, device, dtype):
    torch.save({"model_file": model_file, "state": {k: v.cpu() for k, v in model.state_dict().items()},
                "inputs": [x.cpu() if torch.is_tensor(x) else x for x in inputs], "targets": targets.cpu(),
                "loss": loss, "plan": plan, "kw": kw, "device": device, "dtype": dtype}, path)


def rank_devices(device: str, tp: int) -> List[str]:
    """One device per rank process: "cuda" on a machine with at least tp GPUs gives rank r cuda:r (each
    rank on its own GPU, as a real tp job runs); otherwise every rank uses `device` as given."""
    if str(device) == "cuda" and torch.cuda.is_available() and torch.cuda.device_count() >= tp:
        return [f"cuda:{r}" for r in range(tp)]
    return [str(device)] * tp


def _run_gloo(model_file, model, inputs, targets, loss, plan, tp, kw, device, dtype, timeout=300.0):
    import os
    import tempfile

    import torch.multiprocessing as mp

    from .comm import file_init_method

    with tempfile.TemporaryDirectory(prefix="truestar_tree_") as d:
        pay = os.path.join(d, "payload.pt")
        write_payload(pay, os.path.abspath(model_file), model, inputs, targets, loss, plan, kw, device, dtype)
        ctx = mp.get_context("spawn")
        init = file_init_method(os.path.join(d, "init"))
        devs = rank_devices(device, tp)
        procs = [ctx.Process(target=_spawned_rank, args=(r, tp, pay, init, d, devs[r])) for r in range(tp)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout)
            if p.is_alive():
                p.kill()
        recs = []
        for r in range(tp):
            f = os.path.join(d, f"rank{r}.pt")
            recs.append(torch.load(f, weights_only=False) if os.path.exists(f)
                        else StepRecord(error=f"rank {r} wrote no result (exit {procs[r].exitcode})"))
        return recs


# ----------------------------------------------------------------------------- comparison


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> Tuple[bool, int]:
    """Bit for bit (signed zeros and NaN payloads included): (equal, entries that differ)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, max(a.numel(), b.numel())
    ia = {torch.float64: torch.int64, torch.float32: torch.int32, torch.float16: torch.int16,
          torch.bfloat16: torch.int16}[a.dtype]
    d = (a.contiguous().view(ia) != b.contiguous().view(ia)).sum().item()
    return d == 0, int(d)


def max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest absolute difference relative to the largest magnitude of the one device value."""
    a, b = a.double(), b.double()
    scale = float(b.abs().max()) if b.numel() else 0.0
    return float((a - b).abs().max()) / scale if scale > 0 else float((a - b).abs().max(initial=0.0))


@dataclasses.dataclass
class TreeReport:
    tp: int
    leaf: str
    dtype: str
    device: str
    bitwise: bool
    findings: List[Tuple[str, str]]
    compared: int
    differing: int
    plain_differing: Optional[int] = None
    plain_compared: Optional[int] = None
    plain_max_rel: Optional[float] = None
    seconds: Dict[str, float] = dataclasses.field(default_factory=dict)
    devices: List[str] = dataclasses.field(default_factory=list)   # where each rank ran (gloo processes)

    @property
    def codes(self) -> List[str]:
        return [c for c, _ in self.findings]

    def to_text(self) -> str:
        head = (f"truestar tree tp: tp={self.tp}, leaf {self.leaf}, {self.dtype} on {self.device}: "
                f"{self.compared} values compared bit for bit, {self.differing} differ")
        lines = [head] + [f"  [{c}] {m}" for c, m in self.findings]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self) | {"codes": self.codes}


def compare(ref: StepRecord, recs: List[StepRecord], plan: Plan, tp: int) -> Tuple[int, int, List[str], int]:
    """Every rank's loss, output, gradient and updated piece against the one device values, sharded
    the same way. Returns (compared, differing, first few names that differ, worst relative difference)."""
    compared = differing = 0
    worst = 0.0
    names: List[str] = []

    def one(what, a, b):
        nonlocal compared, differing, worst
        compared += 1
        ok, _ = bits_equal(a, b)
        if not ok:
            differing += 1
            if a.shape == b.shape and " gradient " not in what:   # an exact zero gradient has no scale
                worst = max(worst, max_rel(a, b))
            if len(names) < 5:
                names.append(what)

    for r, rec in enumerate(recs):
        one(f"rank {r} loss", rec.loss, ref.loss)
        one(f"rank {r} output", rec.out, ref.out)
        for field in ("grads", "updated"):
            for pname, full in getattr(ref, field).items():
                mname, _, short = pname.rpartition(".")
                st = plan.get(mname)
                want = shard(full, st, short, tp, r) if is_sharded(st, short) else full
                got = getattr(rec, field).get(pname)
                if got is None:
                    compared += 1
                    differing += 1
                    names.append(f"rank {r} {field} {pname} missing")
                    continue
                one(f"rank {r} {'gradient' if field == 'grads' else 'updated weight'} {pname}", got, want)
    return compared, differing, names, worst


def tree_check(model: nn.Module, inputs, *, tp: int = 2, plan: Optional[Dict] = None, targets=None, loss=None,
               optimizer: str = "adamw", lr: float = 1e-2, clip: float = 1.0, dtype=torch.float32,
               device: str = "cpu", leaf: str = "exact", cap: int = 8, plain: bool = True, ranks: str = "threads",
               model_file: Optional[str] = None) -> TreeReport:
    """One training step one device and under tp (ranks "threads": CPU threads of this process;
    "gloo": one process per rank, CPU or CUDA, needs the model file), both through the
    canonical tree, compared bit for bit. `plain` also runs the plain step (F.linear, rank order
    sums) both ways and records how far it is (TS1103)."""
    from .layout import validate_plan
    from .runtime import MSE

    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
    model = copy.deepcopy(model).to(device=device, dtype=dtype)
    inputs = [x.to(device=device, dtype=dtype) if torch.is_tensor(x) and x.is_floating_point()
              else (x.to(device) if torch.is_tensor(x) else x) for x in inputs]
    plan = validate_plan(model, dict(plan or {}), Layout(tp=tp))
    loss = loss or MSE()
    if targets is None:
        with torch.no_grad():
            o = model(*inputs)
        targets = torch.randn(o.shape, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    targets = targets.to(device=device, dtype=dtype) if targets.is_floating_point() else targets.to(device)
    kw = dict(optimizer=optimizer, lr=lr, clip=clip, cap=cap, leaf=leaf, ways=odd_part(tp))
    findings: List[Tuple[str, str]] = []
    secs: Dict[str, float] = {}
    try:
        ref = run_step(copy.deepcopy(model), inputs, targets, loss, plan, tree=True, tp=1, **kw)
    except TreeLayoutError as e:
        return TreeReport(tp, leaf, str(dtype), device, False, [("TS1102", str(e))], 0, 0)
    secs["one_device_tree"] = ref.seconds
    ref = _rank_record(ref)

    def ranks_run(tree):
        if ranks == "gloo":
            if model_file is None:
                raise ValueError("ranks='gloo' needs model_file (each process builds the model from it)")
            return _run_gloo(model_file, model, inputs, targets, loss, plan, tp, dict(tree=tree, **kw), device, dtype)
        return _run_threads(model, inputs, targets, loss, plan, tp, dict(tree=tree, **kw))
    recs = ranks_run(True)
    errs = [r.error for r in recs if r.error]
    if errs:
        code = "TS1102" if any("TS1102" in e for e in errs) else "TS1101"
        return TreeReport(tp, leaf, str(dtype), device, False, [(code, errs[0])], 0, 0, seconds=secs)
    secs["tp_tree_max_rank"] = max(r.seconds for r in recs)
    compared, differing, names, worst = compare(ref, recs, plan, tp)
    if differing == 0:
        findings.append(("TS1100", f"the tp={tp} step is bitwise equal to the one device step: loss, output, "
                                   f"every gradient and every updated weight on every rank ({compared} values)"))
    else:
        findings.append(("TS1101", f"{differing} of {compared} values differ bitwise, first: {', '.join(names)} "
                                   f"(worst relative difference {worst:.1e})"))
    rep = TreeReport(tp, leaf, str(dtype), device, differing == 0, findings, compared, differing, seconds=secs,
                     devices=[r.device for r in recs])
    if plain:
        pref = _rank_record(run_step(copy.deepcopy(model), inputs, targets, loss, plan, tree=False, tp=1, **kw))
        precs = ranks_run(False)
        secs["one_device_plain"] = pref.seconds
        if all(r.error is None for r in precs):
            secs["tp_plain_max_rank"] = max(r.seconds for r in precs)
            pc, pd, pn, pw = compare(pref, precs, plan, tp)
            rep.plain_compared, rep.plain_differing, rep.plain_max_rel = pc, pd, pw
            if pd:
                findings.append(("TS1103", f"for contrast, the plain step (F.linear, rank order sums) differs from "
                                           f"its one device step in {pd} of {pc} values (worst relative difference {pw:.1e}), "
                                           f"first: {', '.join(pn[:3])}"))
    return rep
