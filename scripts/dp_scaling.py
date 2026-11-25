"""Multi GPU scaling of a training step: Truestar's data parallel gradient sync (native and fixed order)
against torch DistributedDataParallel and FSDP2, and Truestar's column then row MLP against torch tensor
parallel, at 1, 2 and 4 GPUs, with a breakdown of where the data parallel step's time goes.

    PYTHONPATH=src python scripts/dp_scaling.py --shapes small,big --worlds 1,2,4 --out scaling.json

Every arm trains the same model (examples/parallel_model.py) on the same global batch with the same
optimizer (AdamW, foreach) and clipping, float32, TF32 off, deterministic algorithms on. A world of
W GPUs gives each rank 1 / W of the batch (strong scaling). Times are medians over the timed steps,
the slowest rank's median per arm.

Data parallel arms:
* ts_native: GradientSync (gradsync.py), buckets of `--bucket-mb`, overlap with backward, NCCL all
  reduce;
* ts_fixed: the same with the fixed order sum (all to all, rank order fold, all gather): bitwise
  equal to one GPU adding the same per replica gradients in rank order, and run to run;
* ts_fixed_gather: the older fixed order (all gather of every gradient, fold of all of it);
* ddp: torch.nn.parallel.DistributedDataParallel, its defaults plus gradient_as_bucket_view;
* fsdp: torch FSDP2 (fully_shard per block and on the root).

An arm "ts_native:4:1" runs with 4 MiB buckets and a first bucket of 1 MiB; "ddp:4" with bucket_cap_mb 4.

Breakdown for the Truestar arms at W > 1: the step with no gradient sync (compute), the bucket
reductions alone (communication), the host time until every kernel is launched, and the overlap
(compute + communication - step) / communication.

Correctness kept in the timed runs: for ts_fixed the replicas end bitwise equal and a second run is
bitwise equal to the first (scripts/multigpu_verify.py checks it bitwise against one GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))

SHAPES = {
    # the "l4" model of multigpu_verify.py: 13.1 M parameters, 4096 tokens a step
    "small": dict(din=256, d=512, ff=1408, head_dim=64, n=4, dout=256, batch=32, seq=128),
    # 103 M parameters, 8192 tokens a step: compute is not negligible next to the gradient sum
    "big": dict(din=256, d=1024, ff=2816, head_dim=64, n=8, dout=256, batch=32, seq=256),
}
TP_SHAPES = {"small": dict(tokens=4096, d=2048, ff=8192), "big": dict(tokens=8192, d=4096, ff=16384)}


def _det():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _model(cfg, dev):
    import torch

    import parallel_model as PM

    torch.manual_seed(0)
    return PM.Model(din=cfg["din"], d=cfg["d"], ff=cfg["ff"], head_dim=cfg["head_dim"], n=cfg["n"],
                    dout=cfg["dout"]).to(dev)


def _data(cfg, steps, rank, world, dev):
    import torch

    g = torch.Generator().manual_seed(7)
    out = []
    for _ in range(steps):
        x = torch.randn(cfg["batch"], cfg["seq"], cfg["din"], generator=g)
        y = torch.randn(cfg["batch"], cfg["seq"], cfg["dout"], generator=g)
        out.append((x.chunk(world)[rank].to(dev), y.chunk(world)[rank].to(dev), float(y.numel())))
    return out


def _med(ts, skip):
    return statistics.median(ts[skip:])


def _fingerprint(params):
    import torch

    return torch.stack([torch.stack([p.detach().double().sum(), p.detach().double().abs().sum()]) for p in params])


def run_arm(arm, cfg, comm, dev, warm, timed, bucket_mb, sync_mode="on"):
    """One arm's timed steps. sync_mode "on" (normal), "off" (no gradient sum: the compute alone)."""
    import torch
    import torch.distributed as dist

    from truestar.parallel.gradsync import GradientSync

    world, rank = comm.world, comm.rank
    # "ts_native:4:1": bucket 4 MiB, first bucket 1 MiB; "ddp:4": bucket_cap_mb 4
    arm, *sizes = arm.split(":")
    if sizes:
        bucket_mb = float(sizes[0])
    first_mb = float(sizes[1]) if len(sizes) > 1 else None
    m = _model(cfg, dev)
    group = tuple(range(world))
    sync = None
    model = m
    if arm == "ddp" and world > 1:
        from torch.nn.parallel import DistributedDataParallel

        model = DistributedDataParallel(m, device_ids=[dev.index], bucket_cap_mb=bucket_mb,
                                        gradient_as_bucket_view=True)
    elif arm == "fsdp" and world > 1:
        from torch.distributed.fsdp import fully_shard

        for b in m.blocks:
            fully_shard(b)
        fully_shard(m)
    elif arm.startswith("ts_") and world > 1 and sync_mode == "on":
        comm.order = "native" if arm == "ts_native" else "fixed"
        comm.fixed_algo = "gather" if arm == "ts_fixed_gather" else "scatter"
        sync = GradientSync(list(m.named_parameters()), comm, group, bucket_bytes=int(bucket_mb * 2 ** 20), overlap=True,
                            first_bucket_bytes=None if first_mb is None else int(first_mb * 2 ** 20))
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01, foreach=True)
    data = _data(cfg, warm + timed, rank, world, dev)
    times, host = [], []
    for x, y, n in data:
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model(x)
        if arm in ("ddp", "fsdp") and world > 1:
            loss = ((out - y) ** 2).mean()          # their gradient is the mean over replicas
        else:
            loss = ((out - y) ** 2).sum() / n       # Truestar sums: each replica divides by the global count
        loss.backward()
        if sync is not None:
            sync.finish()
        torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True)
        opt.step()
        host.append(time.perf_counter() - t0)
        torch.cuda.synchronize(dev)
        times.append(time.perf_counter() - t0)
    nb = len(sync.buckets) if sync is not None else None
    if sync is not None:
        sync.remove()
    fp = _fingerprint(m.parameters()) if arm not in ("fsdp",) else None
    del model, opt, data
    return {"ms": 1e3 * _med(times, warm), "host_ms": 1e3 * _med(host, warm), "buckets": nb}, fp, m


def comm_only(cfg, comm, dev, bucket_mb, order, algo, reps=6):
    """The bucket reductions of one step alone (no backward): started in bucket order, then waited."""
    import torch

    from truestar.parallel.gradsync import GradientSync

    m = _model(cfg, dev)
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    comm.order, comm.fixed_algo = order, algo
    sync = GradientSync(list(m.named_parameters()), comm, tuple(range(comm.world)), bucket_bytes=bucket_mb << 20,
                        overlap=False)
    ts = []
    for _ in range(reps + 2):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        sync.finish()
        torch.cuda.synchronize(dev)
        ts.append(time.perf_counter() - t0)
    sync.remove()
    return 1e3 * _med(ts, 2)


def tp_arms(tcfg, comm, dev, warm, timed):
    """Column then row MLP (gelu) forward and backward: Truestar's ToColumn / FromRow in both orders,
    and torch tensor parallel (ColwiseParallel, RowwiseParallel) on the same shards."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from truestar.parallel.comm import FromRow, ToColumn

    d, ff, T = tcfg["d"], tcfg["ff"], tcfg["tokens"]
    world, rank = comm.world, comm.rank
    group = tuple(range(world))
    g = torch.Generator().manual_seed(11)
    w1 = torch.randn(ff, d, generator=g) / d ** 0.5
    w2 = torch.randn(d, ff, generator=g) / ff ** 0.5
    x = torch.randn(T, d, generator=g).to(dev)
    res = {}

    def timeit(fn):
        ts = []
        for _ in range(warm + timed):
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize(dev)
            ts.append(time.perf_counter() - t0)
        return 1e3 * _med(ts, warm)

    w1l = w1.chunk(world, 0)[rank].to(dev).requires_grad_()
    w2l = w2.chunk(world, 1)[rank].to(dev).requires_grad_()
    xd = x.clone().requires_grad_()
    for order, algo, name in (("native", "scatter", "ts_native"), ("fixed", "scatter", "ts_fixed"),
                              ("fixed", "gather", "ts_fixed_gather")):
        comm.order, comm.fixed_algo = order, algo

        def step():
            for t in (w1l, w2l, xd):
                t.grad = None
            h = F.gelu(F.linear(ToColumn.apply(xd, comm, group, "tp in", True), w1l))
            FromRow.apply(F.linear(h, w2l), comm, group, "tp out").square().sum().backward()
        res[name] = timeit(step) if world > 1 or name == "ts_native" else None

    if world > 1:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

        mesh = init_device_mesh("cuda", (world,))
        mlp = nn.Sequential(nn.Linear(d, ff, bias=False), nn.GELU(), nn.Linear(ff, d, bias=False)).to(dev)
        with torch.no_grad():
            mlp[0].weight.copy_(w1.to(dev))
            mlp[2].weight.copy_(w2.to(dev))
        parallelize_module(mlp, mesh, {"0": ColwiseParallel(), "2": RowwiseParallel()})
        xt = x.clone().requires_grad_()

        def tstep():
            xt.grad = None
            for p in mlp.parameters():
                p.grad = None
            mlp(xt).square().sum().backward()
        res["torch_tp"] = timeit(tstep)
    return res


def rank_main(rank, world, init, shapes, arms, warm, timed, bucket_mb, out_path, with_tp):
    _det()
    import torch
    import torch.distributed as dist

    from truestar.parallel.devices import DeviceComm, join, leave

    info = join(rank, world, init, device=f"cuda:{rank}", backend="nccl" if world > 1 else "auto", timeout=300)
    dev = torch.device(info.device)
    groups = {tuple(range(world)): dist.new_group(list(range(world))) if world > 1 else None}
    comm = DeviceComm(rank, world, info.device, info.backend, groups)
    out = {"world": world, "gpu": info.device_name, "torch": torch.__version__}
    errors = {}
    for shape in shapes:
        cfg = SHAPES[shape]
        res = {"params": sum(p.numel() for p in _model(cfg, dev).parameters()),
               "global_tokens": cfg["batch"] * cfg["seq"], "arms": {}}
        fps = {}
        for arm in (arms if world > 1 else ["single"]):
            try:
                r, fp, _ = run_arm(arm, cfg, comm, dev, warm, timed, bucket_mb)
                allr = comm.gather_objects(r, dst=0) if world > 1 else [r]
                if rank == 0:
                    res["arms"][arm] = {"ms": max(a["ms"] for a in allr), "host_ms": max(a["host_ms"] for a in allr),
                                        "buckets": r["buckets"]}
                if fp is not None and world > 1:
                    every = comm.gather_objects(fp.cpu(), dst=0)
                    if rank == 0:
                        res["arms"][arm]["replicas_bitwise"] = all(e.equal(every[0]) for e in every)
                fps[arm] = fp
            except Exception as exc:  # noqa: BLE001 - one arm's failure is reported, the others run
                import traceback

                errors[f"{shape}/{arm}"] = f"{type(exc).__name__}: {exc} {traceback.format_exc()[-1200:]}"
                if world > 1:
                    dist.barrier()
            torch.cuda.empty_cache()
        if world > 1 and fps.get("ts_fixed") is not None:
            # a second run of the same steps: bitwise run to run
            _, fp2, _ = run_arm("ts_fixed", cfg, comm, dev, warm, timed, bucket_mb)
            if rank == 0:
                res["ts_fixed_run_to_run_bitwise"] = bool(fp2.equal(fps["ts_fixed"]))
        if world > 1:
            bd = {}
            r, _, _ = run_arm("ts_native", cfg, comm, dev, warm, timed, bucket_mb, sync_mode="off")
            allr = comm.gather_objects(r, dst=0)
            bd["compute_ms"] = max(a["ms"] for a in allr) if rank == 0 else None
            bd["compute_host_ms"] = max(a["host_ms"] for a in allr) if rank == 0 else None
            for order, algo in (("native", "scatter"), ("fixed", "scatter"), ("fixed", "gather")):
                ms = comm_only(cfg, comm, dev, bucket_mb, order, algo)
                allm = comm.gather_objects(ms, dst=0)
                if rank == 0:
                    bd[f"comm_{order}_{algo}_ms"] = max(allm)
            if rank == 0:
                grad_bytes = 4 * res["params"]
                bd["grad_mb"] = grad_bytes / 2 ** 20
                bd["ring_bus_gb_s_native"] = 2 * (world - 1) / world * grad_bytes / (bd["comm_native_scatter_ms"] / 1e3) / 1e9
                for arm, comm_key in (("ts_native", "comm_native_scatter_ms"), ("ts_fixed", "comm_fixed_scatter_ms"),
                                      ("ts_fixed_gather", "comm_fixed_gather_ms")):
                    if arm in res["arms"]:
                        c, s = bd[comm_key], res["arms"][arm]["ms"]
                        bd[f"{arm}_overlap"] = (bd["compute_ms"] + c - s) / c
                res["breakdown"] = bd
        out[shape] = res
        if with_tp:
            try:
                tr = tp_arms(TP_SHAPES[shape], comm, dev, warm, timed)
                allt = comm.gather_objects(tr, dst=0) if world > 1 else [tr]
                if rank == 0:
                    out[f"tp_{shape}"] = {k: (max(a[k] for a in allt) if allt[0][k] is not None else None)
                                          for k in tr}
                    out[f"tp_{shape}"]["shape"] = TP_SHAPES[shape]
            except Exception as exc:  # noqa: BLE001
                import traceback

                errors[f"tp_{shape}"] = f"{type(exc).__name__}: {exc} {traceback.format_exc()[-1200:]}"
    errs = comm.gather_objects(errors, dst=0) if world > 1 else [errors]
    if rank == 0:
        out["errors"] = {k: v for e in errs for k, v in e.items()}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=str)
    leave()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="small,big")
    ap.add_argument("--worlds", default="1,2,4")
    ap.add_argument("--arms", default="ts_native,ts_fixed,ts_fixed_gather,ddp,fsdp")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--timed", type=int, default=8)
    ap.add_argument("--bucket-mb", type=int, default=25)
    ap.add_argument("--no-tp", action="store_true")
    ap.add_argument("--verify", default="", help="also run scripts/multigpu_verify.py with these phases at the "
                                                 "largest world (e.g. dp,tp)")
    ap.add_argument("--out", default="dp_scaling.json")
    a = ap.parse_args()
    import torch
    import torch.multiprocessing as mp

    from truestar.parallel.comm import file_init_method

    shapes = [s for s in a.shapes.split(",") if s]
    arms = [s for s in a.arms.split(",") if s]
    worlds = [int(w) for w in a.worlds.split(",") if w and int(w) <= torch.cuda.device_count()]
    result = {"bucket_mb": a.bucket_mb, "warm": a.warm, "timed": a.timed}
    for w in worlds:
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as d:
            part = os.path.join(d, "out.json")
            init = file_init_method(os.path.join(d, "init"))
            ps = [mp.get_context("spawn").Process(target=rank_main, args=(r, w, init, shapes, arms, a.warm, a.timed,
                                                                         a.bucket_mb, part, not a.no_tp))
                  for r in range(w)]
            for p in ps:
                p.start()
            for p in ps:
                p.join(900)
            codes = [p.exitcode for p in ps]
            result[f"world{w}"] = json.load(open(part, encoding="utf-8")) if os.path.exists(part) else {"exit": codes}
        result[f"world{w}"]["seconds"] = time.perf_counter() - t0
        print(f"[dp_scaling] world {w} done in {time.perf_counter() - t0:.0f}s exit {codes}", flush=True)
    if a.verify and worlds:
        w = max(worlds)
        with tempfile.TemporaryDirectory() as d:
            vo = os.path.join(d, "v.json")
            t0 = time.perf_counter()
            p = subprocess.run([sys.executable, os.path.join(HERE, "multigpu_verify.py"), "--world", str(w), "--size",
                                "l4", "--phases", a.verify, "--out", vo], capture_output=True, text=True)
            result["verify"] = json.load(open(vo, encoding="utf-8")) if os.path.exists(vo) else \
                {"exit": p.returncode, "stderr": p.stderr[-3000:]}
            result["verify_seconds"] = time.perf_counter() - t0
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=str)
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
