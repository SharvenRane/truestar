"""Worker processes for the data parallel tests in tests/test_train.py (spawned, so they must live
in a module). Each rank runs the same training step program on its shard of every batch."""

import datetime
import os
import sys

import torch
import torch.distributed as dist

from truestar.diag.collectives import CollectiveRecorder
from truestar.parallel.comm import file_init_method
from truestar.train import Trainer


def dp_worker(rank: int, world: int, directory: str, step, batches, steps_for_rank, timeout: float) -> None:
    dist.init_process_group("gloo", init_method=file_init_method(os.path.join(directory, "init")),
                            rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=60))  # backstop only
    rec = CollectiveRecorder(directory=os.path.join(directory, "rec"), timeout=timeout, settle=1.0,
                             on_hang="exit", last_n=6)
    trainer = Trainer(step)
    with rec:
        for k in range(steps_for_rank[rank]):
            shard = [b.chunk(world, dim=0)[rank] for b in batches[k]]
            trainer.train_step(*shard)  # MARK:dp_step
    torch.save({"history": trainer.history, "state": trainer.state},
               os.path.join(directory, f"result_rank{rank}.pt"))
    dist.destroy_process_group()
    sys.exit(0)
