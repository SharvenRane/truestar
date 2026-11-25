"""Worker processes for tests/test_diag_collectives.py (spawned, so they must live in a module).

Lines the tests look up are tagged with ``# MARK:<name>``; the tests find their line numbers by
scanning this file, so editing the file does not break the assertions.
"""

import datetime
import os
import sys

import torch
import torch.distributed as dist

from truestar.diag.collectives import CollectiveRecorder
from truestar.parallel.comm import file_init_method


def _init(rank: int, world: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=file_init_method(init_file), rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=30))  # backstop only


def scenario_order(rank: int):
    """Rank 2 calls broadcast before all_reduce; every other rank calls all_reduce first."""
    x = torch.ones(4)
    dist.all_reduce(x)  # MARK:order_warmup
    if rank == 2:
        dist.broadcast(x, src=0)  # MARK:order_bad
        dist.all_reduce(x)
    else:
        dist.all_reduce(x)  # MARK:order_good
        dist.broadcast(x, src=0)


def scenario_skip(rank: int):
    """A data dependent branch: rank 1 has no auxiliary loss this step and skips its all_reduce."""
    grads = torch.ones(8)
    dist.all_reduce(grads)  # MARK:skip_grads
    aux = torch.full((8,), float(rank))
    has_aux = rank != 1  # stands in for "this batch produced an auxiliary loss"
    if has_aux:
        dist.all_reduce(aux)  # MARK:skip_target
    stats = torch.zeros(8)
    dist.all_reduce(stats)  # MARK:skip_after


def scenario_crash(rank: int):
    """Rank 2 hits a bad batch and raises; the others wait in the next all_reduce."""
    x = torch.ones(4)
    dist.all_reduce(x)  # MARK:crash_first
    dist.all_reduce(x)  # MARK:crash_second
    if rank == 2:
        raise ValueError("bad batch: label 7 out of range")  # MARK:crash_raise
    dist.all_reduce(x)  # MARK:crash_wait


def scenario_clean(rank: int):
    x = torch.ones(4)
    for _ in range(3):
        dist.all_reduce(x)
    dist.broadcast(x, src=0)
    dist.barrier()


SCENARIOS = {"order": scenario_order, "skip": scenario_skip, "crash": scenario_crash, "clean": scenario_clean}


def worker(rank: int, world: int, directory: str, scenario: str, timeout: float) -> None:
    _init(rank, world, os.path.join(directory, "init"))
    rec = CollectiveRecorder(directory=os.path.join(directory, "rec"), timeout=timeout, settle=1.0,
                             on_hang="exit", last_n=6)
    with rec:
        dist.barrier()  # MARK:start_barrier
        SCENARIOS[scenario](rank)
    # The clean scenario ends here. Ranks that hang never get here: the watchdog exits them.
    dist.destroy_process_group()
    sys.exit(0)
