"""`truestar parallel --ranks devices`: the layout step on real ranks, one process per device.

Same rank program as the simulated ranks (runtime.run_rank), with a DeviceComm (devices.py): NCCL
between GPUs on Linux, gloo elsewhere (Windows, CPU ranks, two machines). Both arms stay float64, so
the 1e-9 threshold of docs/PARALLEL.md applies unchanged; what is new is that every collective is a
real torch.distributed call between processes on real devices, and the gradient sync is the bucketed
one that starts during the backward pass.

Two ways to start the ranks:

* local: the parent spawns `world` processes, one per GPU of this machine (cuda:0 to cuda:N-1), or
  CPU processes with `device="cpu"`; they meet over a file store;
* several machines: every machine runs the same command with its own `--dist-rank`, and they meet
  at `--dist-master host:port` (rank 0 listens there). Rank 0 sends the float64 weights, inputs and
  targets to the others, gathers every rank's record and writes the report.
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Optional

import torch


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, list):
        return [to_cpu(v) for v in x]
    if isinstance(x, tuple):
        return tuple(to_cpu(v) for v in x)
    if isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    return x


def rec_to_cpu(rec):
    for calls in rec.calls.values():
        for c in calls:
            c.out = [to_cpu(t) for t in c.out]
            c.grad = [None if g is None else to_cpu(g) for g in c.grad]
    rec.init = to_cpu(rec.init)
    rec.grads = {k: (None if v is None else to_cpu(v)) for k, v in rec.grads.items()}
    rec.updated = to_cpu(rec.updated)
    rec.log = [tuple(x) for x in rec.log]
    return rec


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x


def _rank_step(rank, world, info, pay, model, order, sync):
    from .devices import DeviceComm, make_groups
    from .runtime import RankRecord, run_rank

    lay = pay["layout"]
    try:
        groups = make_groups(lay)
        comm = DeviceComm(rank, world, info.device, info.backend, groups, order=order)
        dev = torch.device(info.device)
        model = model.double().eval().to(dev)
        model.load_state_dict(pay["state"])
        inputs = to_device(pay["inputs"], dev)
        targets = to_device(pay["targets"], dev)
        if info.device.startswith("cuda"):
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        rec = run_rank(comm, model, inputs, targets, pay["loss"], lay, pay["plan"], pay["optimizer"], pay["lr"],
                       pay["clip"], pay["faults"], abort=lambda: None, sync=sync)
        if info.device.startswith("cuda"):
            torch.cuda.synchronize(dev)
        rec.seconds = time.perf_counter() - t0
    except BaseException as exc:  # noqa: BLE001 - reported by rank 0 as this rank's failure
        rec = RankRecord(rank, lay.tp_rank(rank), lay.dp_rank(rank))
        rec.error = f"{type(exc).__name__}: {exc}"
    rec.device = info.text()
    return rec_to_cpu(rec)


def local_rank_main(rank: int, world: int, directory: str, timeout: float, device: str, backend: str, order: str,
                    sync) -> None:
    """Entry point of one spawned local rank (must live in a module for the spawn start method)."""
    from ..cli import _load
    from .comm import file_init_method
    from .devices import join, leave
    from .runtime import RankRecord

    pay = torch.load(os.path.join(directory, "payload.pt"), weights_only=False)
    out = os.path.join(directory, f"rank{rank}.pt")
    try:
        info = join(rank, world, file_init_method(os.path.join(directory, "init")), device=device, backend=backend,
                    timeout=timeout)
        model, _, _ = _load(pay["model_file"])
        rec = _rank_step(rank, world, info, pay, model, order, sync)
    except BaseException as exc:  # noqa: BLE001
        lay = pay["layout"]
        rec = RankRecord(rank, lay.tp_rank(rank), lay.dp_rank(rank))
        rec.error = f"{type(exc).__name__}: {exc}"
    torch.save(rec, out)
    leave()
    sys.stdout.flush()
    os._exit(0 if rec.error is None else 3)


def run_local(model_file, payload: dict, world: int, devices: List[str], timeout: float, backend: str, order: str,
              sync) -> list:
    """Spawn one process per device in `devices` and return their records in rank order."""
    import tempfile

    import torch.multiprocessing as mp

    from .runtime import RankRecord

    lay = payload["layout"]
    with tempfile.TemporaryDirectory(prefix="truestar_ranks_") as d:
        torch.save(dict(payload, model_file=os.path.abspath(model_file)), os.path.join(d, "payload.pt"))
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=local_rank_main, args=(r, world, d, timeout, devices[r], backend, order, sync))
                 for r in range(world)]
        for p in procs:
            p.start()
        t0 = time.time()
        failed_at = None
        try:
            while any(p.is_alive() for p in procs) and time.time() - t0 < timeout + 120:
                if failed_at is None and any(p.exitcode == 3 for p in procs):
                    failed_at = time.time()
                if failed_at is not None and time.time() - failed_at > 5.0:
                    break
                time.sleep(0.05)
        finally:
            for p in procs:
                if p.is_alive():
                    p.kill()
                p.join(timeout=10)
        recs = []
        for r in range(world):
            f = os.path.join(d, f"rank{r}.pt")
            if os.path.exists(f):
                recs.append(torch.load(f, weights_only=False))
            else:
                rr = RankRecord(r, lay.tp_rank(r), lay.dp_rank(r))
                rr.error = ("aborted: another rank failed, so this rank's collective never completed"
                            if failed_at is not None else
                            f"the rank process exited with code {procs[r].exitcode} and wrote no result")
                recs.append(rr)
        return recs


def run_multi_node(model, model_file, payload: Optional[dict], rank: int, world: int, master: str, device: str,
                   timeout: float, backend: str, order: str, sync):
    """This process is rank `rank` of a job spread over machines. Rank 0 passes `payload` and gets every
    record back (rank order); the other ranks get None."""
    from .devices import DeviceComm, join, leave

    info = join(rank, world, f"tcp://{master}", device=device, backend=backend, timeout=timeout)
    try:
        boot = DeviceComm(rank, world, info.device, info.backend, {}, order=order)
        pay = boot.broadcast_object(payload if rank == 0 else None, src=0)
        rec = _rank_step(rank, world, info, pay, model, order, sync)
        recs = boot.gather_objects(rec, dst=0)
    finally:
        leave()
    return recs
