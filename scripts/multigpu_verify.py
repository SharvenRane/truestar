"""Real multi GPU runs for docs/PARALLEL.md and docs/TRAINING.md (Multi GPU): verification and speed.

    PYTHONPATH=src python scripts/multigpu_verify.py --out results.json            # every GPU of this machine
    PYTHONPATH=src python scripts/multigpu_verify.py --device cpu --size small     # 2 CPU gloo ranks (debug)
    # across machines, one rank per machine (rank 0 listens on the port):
    PYTHONPATH=src python scripts/multigpu_verify.py --dist-master 192.0.2.10:29511 --dist-world 2 --dist-rank 0
    PYTHONPATH=src python scripts/multigpu_verify.py --dist-master 192.0.2.10:29511 --dist-world 2 --dist-rank 1

Phases (each one rank process per device, torch.distributed between them):

* dp: data parallel training of a pre norm transformer (torch.nn), AdamW, clipping, several steps.
  The 2 rank step against (a) one GPU on the whole batch (tolerance) and (b) one GPU computing the
  same per replica gradients and summing them in rank order (bitwise, the "same reduction tree"
  reference), with gradient accumulation 1 and 2, fixed and native order, overlap on and off.
* speed: time per step, one GPU on the whole batch against 2 GPUs, overlap on and off, two bucket
  sizes, fixed and native order.
* shard: optimizer state sharded over the 2 replicas (sharded.py) against the unsharded data
  parallel run, several steps; optimizer state memory per rank.
* tp: a column then row MLP block on 2 GPUs in float32 against one GPU (values and gradients), and
  its time.
* native: `truestar.train_step(..., data_parallel=comm)` on the compiled executor against one GPU.
* layouts (local only): `parallel_check(ranks="devices")` in float64 for tp, dp, sp, sharded state
  and every injected fault.

No code from any other framework's data or tensor parallel implementation is used.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))

SIZES = {
    # din, d, ff, head_dim, blocks, dout, batch, seq, steps
    "small": dict(din=16, d=32, ff=64, head_dim=8, n=2, dout=8, batch=8, seq=8, steps=3, timed=3,
                  tp_d=64, tp_ff=256, tp_tokens=64, mlp=64, mlp_batch=32),
    "lan": dict(din=64, d=256, ff=704, head_dim=64, n=2, dout=64, batch=16, seq=64, steps=3, timed=6,
                tp_d=1024, tp_ff=4096, tp_tokens=1024, mlp=256, mlp_batch=128),
    "l4": dict(din=256, d=512, ff=1408, head_dim=64, n=4, dout=256, batch=32, seq=128, steps=3, timed=6,
               tp_d=2048, tp_ff=8192, tp_tokens=4096, mlp=1024, mlp_batch=512),
    "t4": dict(din=256, d=512, ff=1408, head_dim=64, n=4, dout=256, batch=32, seq=128, steps=4, timed=10,
               tp_d=2048, tp_ff=8192, tp_tokens=4096, mlp=1024, mlp_batch=512),
}


def _setup_determinism():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _sync(dev):
    import torch

    if str(dev).startswith("cuda"):
        torch.cuda.synchronize(dev)


def _model(cfg, dev):
    import torch

    import parallel_model as PM

    torch.manual_seed(0)
    return PM.Model(din=cfg["din"], d=cfg["d"], ff=cfg["ff"], head_dim=cfg["head_dim"], n=cfg["n"],
                    dout=cfg["dout"]).to(dev)


def _batches(cfg, n):
    import torch

    g = torch.Generator().manual_seed(7)
    return [(torch.randn(cfg["batch"], cfg["seq"], cfg["din"], generator=g),
             torch.randn(cfg["batch"], cfg["seq"], cfg["dout"], generator=g)) for _ in range(n)]


def _opt(params):
    import torch

    return torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01, foreach=False)


def _clip(params):
    import torch

    return float(torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=False))


def _flat(params):
    import torch

    return torch.cat([p.detach().reshape(-1).float().cpu() for p in params])


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def single_run(cfg, dev, steps, world=None, ga=1):
    """One device. world None: the whole batch, one backward per micro batch of the whole batch.
    world W: the same per replica gradient sums a W rank run computes, added in rank order."""
    import torch

    m = _model(cfg, dev)
    ps = list(m.parameters())
    opt = _opt(ps)
    losses, flats = [], []
    for x, y in _batches(cfg, steps):
        x, y = x.to(dev), y.to(dev)
        n = float(y.numel())
        if world is None:
            m.zero_grad(set_to_none=True)
            tot = 0.0
            for xm, ym in zip(x.chunk(ga), y.chunk(ga)):
                loss = ((m(xm) - ym) ** 2).sum() / n
                loss.backward()
                tot += float(loss.detach())
        else:
            accs, tot = [], 0.0
            for xr, yr in zip(x.chunk(world), y.chunk(world)):
                m.zero_grad(set_to_none=True)
                for xm, ym in zip(xr.chunk(ga), yr.chunk(ga)):
                    loss = ((m(xm) - ym) ** 2).sum() / n
                    loss.backward()
                    tot += float(loss.detach())
                accs.append([p.grad.clone() for p in ps])
            for i, p in enumerate(ps):
                acc = accs[0][i].clone()
                for a in accs[1:]:
                    acc = acc + a[i]
                p.grad = acc
        if not flats:
            grads0 = torch.cat([p.grad.reshape(-1).float().cpu() for p in ps])
        _clip(ps)
        opt.step()
        losses.append(tot)
        flats.append(_flat(ps))
    return losses, flats, grads0


def dp_run(cfg, comm, dev, steps, ga=1, overlap=True, bucket=25 << 20, timed=False):
    import torch

    from truestar.parallel.gradsync import GradientSync

    m = _model(cfg, dev)
    ps = list(m.parameters())
    opt = _opt(ps)
    group = tuple(range(comm.world))
    sync = GradientSync(list(m.named_parameters()), comm, group, bucket_bytes=bucket, overlap=overlap)
    losses, flats, times = [], [], []
    for x, y in _batches(cfg, steps):
        x, y = x.to(dev), y.to(dev)
        n = float(y.numel())
        xr, yr = x.chunk(comm.world)[comm.rank], y.chunk(comm.world)[comm.rank]
        _sync(dev)
        t0 = time.perf_counter()
        m.zero_grad(set_to_none=True)
        tot = torch.zeros((), device=dev)
        micro = list(zip(xr.chunk(ga), yr.chunk(ga)))
        for i, (xm, ym) in enumerate(micro):
            loss = ((m(xm) - ym) ** 2).sum() / n
            if i < len(micro) - 1:
                with sync.no_sync():
                    loss.backward()
            else:
                loss.backward()
            tot = tot + loss.detach()
        sync.finish()
        if not flats and not timed:
            grads0 = torch.cat([p.grad.reshape(-1).float().cpu() for p in ps])
        _clip(ps)
        opt.step()
        _sync(dev)
        times.append(time.perf_counter() - t0)
        lt = comm.all_reduce(tot, group, "loss")
        losses.append(float(lt))
        if not timed:
            flats.append(_flat(ps))
    sync.remove()
    return losses, flats, times, len(sync.buckets), (None if timed else grads0)


def single_timed(cfg, dev, steps):
    import torch

    m = _model(cfg, dev)
    ps = list(m.parameters())
    opt = _opt(ps)
    times = []
    for x, y in _batches(cfg, steps):
        x, y = x.to(dev), y.to(dev)
        _sync(dev)
        t0 = time.perf_counter()
        m.zero_grad(set_to_none=True)
        loss = ((m(x) - y) ** 2).sum() / float(y.numel())
        loss.backward()
        _clip(ps)
        opt.step()
        _sync(dev)
        times.append(time.perf_counter() - t0)
    return times


def _med(ts, skip=2):
    ts = ts[skip:] if len(ts) > skip + 1 else ts
    return statistics.median(ts)


def phase_dp(cfg, comm, dev, out):
    import torch.distributed as dist

    steps = cfg["steps"]
    res = {}
    for ga in (1, 2):
        for order in ("fixed", "native"):
            comm.order = order
            for overlap in (True, False):
                losses, flats, _, nb, g0 = dp_run(cfg, comm, dev, steps, ga=ga, overlap=overlap,
                                                  bucket=(1 << 20) if cfg is SIZES["t4"] else 4096)
                res[(ga, order, overlap)] = (losses, flats, nb, g0)
    comm.order = "fixed"
    dist.barrier()
    if comm.rank == 0:
        rows = []
        for ga in (1, 2):
            ref_tree = single_run(cfg, dev, steps, world=comm.world, ga=ga)
            ref_whole = single_run(cfg, dev, steps, world=None, ga=1)
            for (g, order, overlap), (losses, flats, nb, g0) in res.items():
                if g != ga:
                    continue
                bit = all(a.equal(b) for a, b in zip(flats, ref_tree[1])) and g0.equal(ref_tree[2])
                rows.append({"ga": ga, "order": order, "overlap": overlap, "buckets": nb,
                             "bitwise_vs_same_tree_single_gpu": bit,
                             "worst_param_rel_vs_same_tree": max(_rel(a, b) for a, b in zip(flats, ref_tree[1])),
                             "worst_param_rel_vs_whole_batch_single_gpu":
                                 max(_rel(a, b) for a, b in zip(flats, ref_whole[1])),
                             "step1_grad_rel_vs_whole_batch": _rel(g0, ref_whole[2]),
                             "loss_rel_vs_whole": max(abs(a - b) / abs(b) for a, b in zip(losses, ref_whole[0]))})
        # run to run: the same dp configuration twice
        out["dp"] = rows
    else:
        out["dp"] = None
    dist.barrier()
    a = dp_run(cfg, comm, dev, steps, ga=1, overlap=True)[1]
    b = dp_run(cfg, comm, dev, steps, ga=1, overlap=True)[1]
    if comm.rank == 0:
        out["dp_run_to_run_bitwise"] = all(x.equal(y) for x, y in zip(a, b))
    other = comm.gather_objects(a[-1], dst=0)
    if comm.rank == 0:
        out["dp_replicas_bitwise"] = all(o.equal(other[0]) for o in other)


def phase_speed(cfg, comm, dev, out):
    import torch.distributed as dist

    steps = cfg["timed"] + 2
    rows = {}
    if comm.rank == 0:
        t = single_timed(cfg, dev, steps)
        rows["single_gpu_whole_batch_ms"] = 1e3 * _med(t)
    dist.barrier()
    for order in ("fixed", "native"):
        comm.order = order
        for overlap in (True, False):
            for bucket_mb in (1, 25):
                _, _, t, nb, _ = dp_run(cfg, comm, dev, steps, overlap=overlap, bucket=bucket_mb << 20, timed=True)
                ms = comm.gather_objects(1e3 * _med(t), dst=0)
                if comm.rank == 0:
                    rows[f"dp{comm.world}_{order}_overlap_{'on' if overlap else 'off'}_bucket_{bucket_mb}mb_ms"] = \
                        max(ms)
                    rows[f"dp{comm.world}_{order}_overlap_{'on' if overlap else 'off'}_bucket_{bucket_mb}mb_buckets"] = nb
    comm.order = "fixed"
    if comm.rank == 0:
        params = sum(p.numel() for p in _model(cfg, dev).parameters())
        rows["params"] = params
        rows["global_batch_tokens"] = cfg["batch"] * cfg["seq"]
        best = min(v for k, v in rows.items() if k.endswith("_ms") and k.startswith("dp"))
        rows["speedup_best_dp_vs_single"] = rows["single_gpu_whole_batch_ms"] / best
        out["speed"] = rows


def phase_shard(cfg, comm, dev, out):
    import torch

    from truestar.parallel.sharded import FlatShard

    steps = cfg["steps"]
    group = tuple(range(comm.world))
    res = {}
    for clip in (False, True):
        for mode in ("replicated", "sharded"):
            m = _model(cfg, dev)
            ps = list(m.parameters())
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)
            before = torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0
            sh = FlatShard(list(m.named_parameters()), comm.rank, comm.world)
            if mode == "sharded":
                piece = torch.nn.Parameter(sh.mine(sh.flat_params()).clone())
                opt = _opt([piece])
            else:
                opt = _opt(ps)
            flats = []
            for x, y in _batches(cfg, steps):
                x, y = x.to(dev), y.to(dev)
                xr, yr = x.chunk(comm.world)[comm.rank], y.chunk(comm.world)[comm.rank]
                m.zero_grad(set_to_none=True)
                (((m(xr) - yr) ** 2).sum() / float(y.numel())).backward()
                if mode == "replicated":
                    flat = sh.flat_grads()
                    red = comm.all_reduce(flat, group, "grads")
                    for p, g in zip(ps, sh.pieces(red)):
                        p.grad.copy_(g)
                    if clip:
                        _clip(ps)
                    opt.step()
                else:
                    gs = comm.reduce_scatter(sh.flat_grads(), group, 0, "grads")
                    if clip:
                        sq = comm.all_reduce(gs.double().pow(2).sum(), group, "norm")
                        coef = min(1.0 / (float(sq.sqrt()) + 1e-6), 1.0)
                        gs = gs * coef
                    piece.grad = gs
                    opt.step()
                    sh.write_params(comm.all_gather(piece.detach(), group, 0, "params"))
                flats.append(_flat(ps))
            state = sum(t.numel() * t.element_size() for st in opt.state.values() for t in st.values()
                        if torch.is_tensor(t) and t.dim() > 0)
            peak = (torch.cuda.max_memory_allocated(dev) - before) if dev.type == "cuda" else 0
            res[(clip, mode)] = (flats, state, peak)
    if comm.rank == 0:
        rows = []
        for clip in (False, True):
            a, b = res[(clip, "sharded")], res[(clip, "replicated")]
            rows.append({"clip": clip, "bitwise_vs_replicated": all(x.equal(y) for x, y in zip(a[0], b[0])),
                         "worst_param_rel": max(_rel(x, y) for x, y in zip(a[0], b[0])),
                         "optimizer_state_bytes_sharded": a[1], "optimizer_state_bytes_replicated": b[1],
                         "peak_bytes_sharded": a[2], "peak_bytes_replicated": b[2]})
        out["shard"] = rows


def phase_tp(cfg, comm, dev, out):
    import torch
    import torch.nn.functional as F

    from truestar.parallel.comm import FromRow, ToColumn

    d, ff, T = cfg["tp_d"], cfg["tp_ff"], cfg["tp_tokens"]
    group = tuple(range(comm.world))
    g = torch.Generator().manual_seed(11)
    w1 = torch.randn(ff, d, generator=g) / d ** 0.5
    w2 = torch.randn(d, ff, generator=g) / ff ** 0.5
    x = torch.randn(T, d, generator=g)
    r, n = comm.rank, comm.world
    w1l = w1.chunk(n, dim=0)[r].to(dev).requires_grad_()
    w2l = w2.chunk(n, dim=1)[r].to(dev).requires_grad_()
    xd = x.to(dev).requires_grad_()
    res = {}
    for order in ("fixed", "native"):
        comm.order = order
        times = []
        for it in range(cfg["timed"] + 2):
            for t in (w1l, w2l, xd):
                t.grad = None
            _sync(dev)
            t0 = time.perf_counter()
            h = F.gelu(F.linear(ToColumn.apply(xd, comm, group, "tp input grad", True), w1l))
            y = FromRow.apply(F.linear(h, w2l), comm, group, "tp row output")
            y.square().sum().backward()
            _sync(dev)
            times.append(time.perf_counter() - t0)
        res[order] = (1e3 * _med(times), y.detach().cpu(), xd.grad.cpu(), w1l.grad.cpu(), w2l.grad.cpu())
    comm.order = "fixed"
    gw1 = comm.gather_objects(res["fixed"][3], dst=0)
    gw2 = comm.gather_objects(res["fixed"][4], dst=0)
    if comm.rank == 0:
        W1, W2, X = w1.to(dev).requires_grad_(), w2.to(dev).requires_grad_(), x.to(dev).requires_grad_()
        times = []
        for it in range(cfg["timed"] + 2):
            for t in (W1, W2, X):
                t.grad = None
            _sync(dev)
            t0 = time.perf_counter()
            y = F.linear(F.gelu(F.linear(X, W1)), W2)
            y.square().sum().backward()
            _sync(dev)
            times.append(time.perf_counter() - t0)
        ms, yp, gx, _, _ = res["fixed"]
        out["tp"] = {"single_gpu_ms": 1e3 * _med(times), f"tp{comm.world}_fixed_ms": ms,
                     f"tp{comm.world}_native_ms": res["native"][0],
                     "out_rel": _rel(yp, y.detach().cpu()), "input_grad_rel": _rel(gx, X.grad.cpu()),
                     "w1_grad_rel": _rel(torch.cat(gw1, 0), W1.grad.cpu()),
                     "w2_grad_rel": _rel(torch.cat(gw2, 1), W2.grad.cpu()),
                     "shape": f"tokens {T}, d {d}, ff {ff} (column then row, gelu)"}


def phase_native(cfg, comm, dev, out, executor):
    import torch
    import torch.distributed as dist

    import truestar
    from truestar import Tensor, nn, optim

    h = cfg["mlp"]
    net = nn.Sequential(nn.Linear(64, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 10))

    globals().update(Tensor=Tensor, nn=nn, net=net)  # the annotations are read from module globals

    @truestar.native
    def loss(p: net.spec, x: Tensor["B", 64], y: Tensor["B", "i64"]) -> Tensor[()]:
        return nn.cross_entropy(net(p, x), y)

    g = torch.Generator().manual_seed(5)
    B = cfg["mlp_batch"] // comm.world * comm.world   # equal shards for any world (TS1210)
    data = [(torch.randn(B, 64, generator=g), torch.randint(0, 10, (B,), generator=g)) for _ in range(cfg["steps"] + 2)]

    def leaves(t):
        return [x for k in sorted(t) for x in leaves(t[k])] if isinstance(t, dict) else [t]

    def to_dev(tree):
        return {k: to_dev(v) for k, v in tree.items()} if isinstance(tree, dict) else tree.to(dev)

    def run(step, batches):
        ts = step.init(to_dev(net.init(seed=0)))
        ts.opt = {k: to_dev(v) for k, v in ts.opt.items()}
        times, hist = [], []
        for x, y in batches:
            _sync(dev)
            t0 = time.perf_counter()
            ts, r = step(ts, x.to(dev), y.to(dev))
            _sync(dev)
            times.append(time.perf_counter() - t0)
            hist.append(r.loss)
        return torch.cat([p.detach().reshape(-1).cpu() for p in leaves(ts.params)]), hist, times

    t0 = time.perf_counter()
    dp = truestar.train_step(loss, optim.AdamW(lr=1e-3), clip=1.0, executor=executor, data_parallel=comm,
                             check_replicas=1)
    shards = [(x.chunk(comm.world)[comm.rank], y.chunk(comm.world)[comm.rank]) for x, y in data]
    pdp, hdp, tdp = run(dp, shards)
    build_dp = time.perf_counter() - t0
    ms_dp = comm.gather_objects(1e3 * _med(tdp), dst=0)
    dist.barrier()
    err = None
    if comm.rank == 0:
      try:
        tree = truestar.train_step(loss, optim.AdamW(lr=1e-3), clip=1.0, executor=executor, accumulate=comm.world)
        ptree, htree, _ = run(tree, data)
        whole = truestar.train_step(loss, optim.AdamW(lr=1e-3), clip=1.0, executor=executor)
        pw, hw, tw = run(whole, data)
        out["native"] = {"executor": executor, "steps": len(data),
                         "bitwise_vs_single_gpu_accumulate": bool(pdp.equal(ptree)),
                         "param_rel_vs_single_gpu_accumulate": _rel(pdp, ptree),
                         "param_rel_vs_single_gpu_whole_batch": _rel(pdp, pw),
                         "loss_rel_vs_whole": max(abs(a - b) / abs(b) for a, b in zip(hdp, hw)),
                         "dp_ms": max(ms_dp), "single_whole_ms": 1e3 * _med(tw),
                         "first_steps_s_dp": build_dp, "replica_check": "passed every step (TS824 not raised)"}
      except Exception as exc:  # noqa: BLE001 - reported after the barrier, so the peers do not hang
        err = exc
    dist.barrier()
    if err is not None:
        raise err


def _tx_plan(cfg, tp):
    plan = {}
    for i in range(cfg["n"]):
        p = f"blocks.{i}."
        plan.update({p + "attn.q": "column", p + "attn.k": "column", p + "attn.v": "column", p + "attn.o": "row",
                     p + "gate_up": "column:2", p + "down": "row"})
    return plan


def phase_tpx(cfg, comm, dev, out):
    """Tensor parallel training steps of the transformer: q, k, v and gate_up split by columns, o and
    down by rows (truestar.parallel.runtime, the layout runtime's sharded modules on DeviceComm), AdamW,
    float32, the whole batch on every rank. Against one GPU: every parameter after every step
    (gathered and unsharded), and the time per step, fixed and native order."""
    import torch

    from truestar.parallel.layout import Layout, is_sharded, unshard, validate_plan
    from truestar.parallel.runtime import RankRecord, _install_plan

    n, r = comm.world, comm.rank
    if cfg["d"] % n or (cfg["d"] // cfg["head_dim"]) % n or cfg["ff"] % n:
        out["tpx"] = {"skipped": f"d={cfg['d']}, heads={cfg['d'] // cfg['head_dim']}, ff={cfg['ff']} do not split over tp={n}"}
        return
    steps = cfg["timed"] + 2
    batches = _batches(cfg, steps)

    def run(model, opt):
        flats, times = [], []
        for x, y in batches:
            x, y = x.to(dev), y.to(dev)
            _sync(dev)
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            (((model(x) - y) ** 2).sum() / float(y.numel())).backward()
            opt.step()
            _sync(dev)
            times.append(time.perf_counter() - t0)
            if len(times) in (1, steps):   # after the first and the last step (keeps the gather small)
                flats.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        return flats, times

    rows = {}
    for order in ("fixed", "native"):
        comm.order = order
        m = _model(cfg, dev)
        plan = validate_plan(m, _tx_plan(cfg, n), Layout(tp=n))
        _install_plan(m, plan, Layout(tp=n), comm, r, set(), RankRecord(r, r, 0))
        flats, times = run(m, _opt(list(m.parameters())))
        allr = comm.gather_objects((flats, 1e3 * _med(times)), dst=0)
        if r == 0:
            rows[order] = allr
    comm.order = "fixed"
    if r == 0:
        m1 = _model(cfg, dev)
        single, t1 = run(m1, _opt(list(m1.parameters())))
        res = {"tp": n, "single_ms": 1e3 * _med(t1), "params_after_steps": steps,
               "shape": f"{cfg['n']} blocks, d {cfg['d']}, ff {cfg['ff']}, heads {cfg['d'] // cfg['head_dim']}, "
                        f"batch {cfg['batch']} x {cfg['seq']} tokens"}
        for order, allr in rows.items():
            worst, worst_name, first_step, other = 0.0, None, 0.0, 0.0
            for s_i in range(2):
                for k, ref in single[s_i].items():
                    mname, _, short = k.rpartition(".")
                    st = plan.get(mname)
                    if is_sharded(st, short):
                        got = unshard([a[0][s_i][k] for a in allr], st, short)
                    else:
                        got = allr[0][0][s_i][k]
                    e = _rel(got.float(), ref.float())
                    if s_i == 0:
                        first_step = max(first_step, e)
                    if not k.endswith("attn.k.bias"):
                        other = max(other, e)
                    if e > worst:
                        worst, worst_name = e, f"{k} after step {(1, steps)[s_i]}"
            res[order] = {"tp_ms": max(a[1] for a in allr), "speedup": res["single_ms"] / max(a[1] for a in allr),
                          "worst_param_rel_vs_single": worst, "worst_at": worst_name,
                          "first_step_param_rel": first_step,
                          # the key bias has an exactly zero true gradient (softmax ignores a constant added to
                          # every key), so AdamW turns its rounding noise into full steps; the rest:
                          "worst_param_rel_without_key_bias": other}
        out["tpx"] = res


def rank_main(rank, world, init_method, device, size, phases, out_path, backend):
    _setup_determinism()
    import torch

    from truestar.parallel.devices import DeviceComm, join, leave

    info = join(rank, world, init_method, device=device, backend=backend, timeout=600)
    import torch.distributed as dist

    groups = {tuple(range(world)): dist.new_group(list(range(world)))}
    comm = DeviceComm(rank, world, info.device, info.backend, groups, order="fixed")
    dev = torch.device(info.device)
    cfg = SIZES[size]
    out = {"ranks": comm.gather_objects(info.text(), dst=0), "size": size, "torch": torch.__version__,
           "cuda": torch.version.cuda, "nccl": (str(torch.cuda.nccl.version())
                                                if info.backend == "nccl" else None)}
    errors = {}
    for ph in phases:
        t0 = time.perf_counter()
        try:
            if ph == "dp":
                phase_dp(cfg, comm, dev, out)
            elif ph == "speed":
                phase_speed(cfg, comm, dev, out)
            elif ph == "shard":
                phase_shard(cfg, comm, dev, out)
            elif ph == "tp":
                phase_tp(cfg, comm, dev, out)
            elif ph == "tpx":
                phase_tpx(cfg, comm, dev, out)
            elif ph == "native":
                phase_native(cfg, comm, dev, out, "cuda" if dev.type == "cuda" else "reference")
        except Exception as exc:  # noqa: BLE001 - reported per phase, the others still run
            import traceback

            errors[ph] = f"rank {rank}: {type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"
        out.setdefault("seconds", {})[ph] = time.perf_counter() - t0
    errs = comm.gather_objects(errors, dst=0)
    if rank == 0:
        out["errors"] = {k: v for e in errs for k, v in e.items()}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=str)
    leave()


def layouts(model_file, device, out, clean=None, with_faults=True):
    """parallel_check on real ranks: clean layouts and every injected fault."""
    import parallel_model as PM

    from truestar.parallel import FAULTS, parallel_check, parse_layout

    rows = []
    clean = clean or ["dp=2", "dp=2,ga=2", "tp=2", "tp=2,sp=1", "tp=2,ga=2,sp=1", "dp=2,shard=1", "dp=2,ga=2,shard=2"]
    for lay in clean:
        for order in ("fixed", "native"):
            L = parse_layout(lay)
            t0 = time.perf_counter()
            r = parallel_check(PM.get_model(), PM.get_inputs(), L, plan=PM.get_parallel_plan(L) if L.tp > 1 else None,
                               optimizer="adamw", clip=0.5, ranks="devices", device=device, model_file=model_file,
                               order=order, sync={"bucket_bytes": 4096, "overlap": True})
            rows.append({"layout": lay, "order": order, "status": r.status, "codes": r.codes, "worst": r.worst,
                         "seconds": time.perf_counter() - t0, "devices": r.devices})
    out["layouts"] = rows
    if not with_faults:
        return
    import torch

    from truestar.parallel import CrossEntropy

    tests = os.path.abspath(os.path.join(HERE, "..", "tests"))
    sys.path.insert(0, tests)
    import parallel_models as M

    cls_file = os.path.join(tempfile.mkdtemp(prefix="truestar_cls_"), "padded_classifier.py")
    with open(cls_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join([
            "import sys", f"sys.path.insert(0, {tests!r})", "import torch", "import parallel_models as M",
            "from truestar.parallel import CrossEntropy",
            "def get_model():", "    return M.Classifier()",
            "def get_inputs():", "    return torch.randn(8, 6, 8, generator=torch.Generator().manual_seed(3))",
            "def get_targets():", "    return M.padded_targets()",
            "def get_loss():", "    return CrossEntropy()", ""]))
    faults = []
    for f in list(FAULTS):
        lay = "tp=2,ga=2,sp=1" if f in ("missing_tp_sync", "column_input_grad_not_reduced", "row_bias_every_rank",
                                        "fused_split_as_block", "row_split_dim0", "clip_local_norm") else \
            ("dp=2,shard=1" if f == "shard_stale_params" else "dp=2,ga=4")
        if f == "ga_mean_of_means":
            lay = "dp=2,ga=2"
            r = parallel_check(M.Classifier(), torch.randn(8, 6, 8, generator=torch.Generator().manual_seed(3)), lay,
                               targets=M.padded_targets(), loss=CrossEntropy(), ranks="devices", device=device,
                               model_file=cls_file, faults=[f], sync={"bucket_bytes": 4096, "overlap": True})
        else:
            L = parse_layout(lay)
            r = parallel_check(PM.get_model(), PM.get_inputs(), L, plan=PM.get_parallel_plan(L) if L.tp > 1 else None,
                               optimizer="adamw", clip=0.5, ranks="devices", device=device, model_file=model_file,
                               faults=[f], sync={"bucket_bytes": 4096, "overlap": True})
        first = r.first()
        faults.append({"fault": f, "layout": lay, "status": r.status, "code": first.code if first else None,
                       "module": first.module if first else None, "collective": first.collective if first else None})
    out["layouts"] = rows
    out["faults"] = faults


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--size", default="t4", choices=tuple(SIZES))
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--phases", default="dp,speed,shard,tp,native,layouts")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--out", default="multigpu_results.json")
    ap.add_argument("--layouts", help="semicolon separated layouts for the layouts phase (default: the 2 rank list)")
    ap.add_argument("--no-faults", action="store_true", help="layouts phase: clean layouts only")
    ap.add_argument("--dist-master")
    ap.add_argument("--dist-world", type=int)
    ap.add_argument("--dist-rank", type=int)
    a = ap.parse_args()
    phases = [p for p in a.phases.split(",") if p]
    rank_phases = [p for p in phases if p != "layouts"]
    if a.dist_master:
        rank_main(a.dist_rank, a.dist_world, f"tcp://{a.dist_master}", a.device, a.size, rank_phases, a.out, a.backend)
        if a.dist_rank == 0:
            print(open(a.out, encoding="utf-8").read())
        return
    import torch.multiprocessing as mp

    from truestar.parallel.comm import file_init_method
    from truestar.parallel.devices import local_devices

    devs = local_devices(a.world, a.device)
    res = {}
    if rank_phases:
        with tempfile.TemporaryDirectory() as d:
            part = os.path.join(d, "out.json")
            init = file_init_method(os.path.join(d, "init"))
            ps = [mp.get_context("spawn").Process(target=rank_main,
                                                  args=(r, a.world, init, devs[r], a.size, rank_phases, part, a.backend))
                  for r in range(a.world)]
            for p in ps:
                p.start()
            for p in ps:
                p.join(7200)
            if os.path.exists(part):
                res = json.load(open(part, encoding="utf-8"))
            else:
                res = {"error": f"rank processes exited with {[p.exitcode for p in ps]} and wrote nothing"}
    if "layouts" in phases:
        mf = os.path.join(HERE, "..", "examples", "parallel_model.py")
        try:
            layouts(mf, a.device, res, a.layouts.split(";") if a.layouts else None, not a.no_faults)
        except Exception as exc:  # noqa: BLE001
            import traceback

            res.setdefault("errors", {})["layouts"] = traceback.format_exc()[-3000:]
            print(exc, file=sys.stderr)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
