"""Entry point of one gloo rank process for `parallel_check(..., ranks="gloo")` (spawned, so it must
live in a module). The process builds the model from the model file, loads the parent's float64
weights, joins a gloo group over a file store, creates every tp and dp group in the same order as
every other rank, runs the rank step and saves its record."""

from __future__ import annotations

import datetime
import os
import sys

import torch


def gloo_rank(rank: int, world: int, directory: str, timeout: float) -> None:
    import torch.distributed as dist

    from ..cli import _load
    from .comm import GlooComm, file_init_method
    from .runtime import RankRecord, run_rank

    pay = torch.load(os.path.join(directory, "payload.pt"), weights_only=False)
    lay = pay["layout"]
    out = os.path.join(directory, f"rank{rank}.pt")
    try:
        model, _, _ = _load(pay["model_file"])
        model = model.double().eval()
        model.load_state_dict(pay["state"])
        dist.init_process_group("gloo", init_method=file_init_method(os.path.join(directory, "init")),
                                rank=rank, world_size=world,
                                timeout=datetime.timedelta(seconds=timeout))
        groups = {}
        for d in range(lay.dp):
            g = tuple(d * lay.tp + j for j in range(lay.tp))
            groups[g] = dist.new_group(list(g)) if len(g) > 1 else None
        for t in range(lay.tp):
            g = tuple(i * lay.tp + t for i in range(lay.dp))
            groups[g] = dist.new_group(list(g)) if len(g) > 1 else None
        comm = GlooComm(rank, world, groups)
        rec = run_rank(comm, model, pay["inputs"], pay["targets"], pay["loss"], lay, pay["plan"], pay["optimizer"],
                       pay["lr"], pay["clip"], pay["faults"], abort=lambda: None)
    except BaseException as exc:  # noqa: BLE001 - reported by the parent as this rank's failure
        rec = RankRecord(rank, lay.tp_rank(rank), lay.dp_rank(rank))
        rec.error = f"{type(exc).__name__}: {exc}"
    torch.save(rec, out)
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass
    sys.stdout.flush()
    os._exit(0 if rec.error is None else 3)  # a rank whose peer failed may still block in gloo
