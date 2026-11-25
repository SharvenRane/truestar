"""Measurements for docs/PARALLEL.md, "Bitwise tensor parallel by one canonical tree" (P55).

Part "check": the example transformer one step at tp=2 and tp=4, threads (CPU) and gloo processes
(CPU and CUDA), both leaves, AdamW and SGD, clipping 1.0: bitwise verdict of the tree, and how far
the plain step (F.linear, rank order sums) is from its own one device step.

Part "speed": one nn.Linear forward and backward at M tokens by K by N, F.linear against the tree
with each leaf, on the CPU and on CUDA (short: the GPU is shared).

    PYTHONPATH=src python scripts/tree_tp_study.py check|speed [--cuda] [--out results/tree_tp.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))

from truestar.parallel.layout import Layout  # noqa: E402
from truestar.parallel.tree import TreeSpec, canon_tile, group_count, tree_check, tree_linear  # noqa: E402

MODEL = os.path.join(HERE, "..", "examples", "parallel_model.py")


WIDE = """import sys
sys.path.insert(0, {examples!r})
import torch
import parallel_model as pm
from parallel_model import get_parallel_plan


def get_model():
    torch.manual_seed(0)
    return pm.Model(din=64, d=256, ff=512, head_dim=32, n=2, dout=32)


def get_inputs():
    return torch.randn(8, 16, 64, generator=torch.Generator().manual_seed(1))
"""


def part_check(cuda: bool):
    import tempfile

    from truestar.cli import _load
    import parallel_model as pm

    tmp = tempfile.mkdtemp(prefix="truestar_tree_wide_")
    wide = os.path.join(tmp, "tree_wide_model.py")
    with open(wide, "w", encoding="utf-8") as f:
        f.write(WIDE.format(examples=os.path.join(HERE, "..", "examples")))
    rows = []
    for label, mf in (("example d16", MODEL), ("wide d256", wide)):
        model, x, _ = _load(mf)
        configs = [("threads", "cpu")] + [("gloo", "cpu")] + ([("gloo", "cuda")] if cuda else [])
        for ranks, dev in configs:
            for tp in (2, 4):
                for leaf in ("exact", "tiled", "block"):
                    for opt in ("adamw", "sgd"):
                        if ranks == "gloo" and opt == "sgd":
                            continue
                        if label.startswith("wide") and (leaf == "exact" and dev == "cpu" and ranks == "gloo"):
                            continue
                        t = time.perf_counter()
                        r = tree_check(model, x, tp=tp, plan=pm.get_parallel_plan(Layout(tp=tp)), leaf=leaf,
                                       optimizer=opt, device=dev, ranks=ranks, model_file=mf, clip=1.0)
                        row = {"model": label, "ranks": ranks, "device": dev, "tp": tp, "leaf": leaf, "optimizer": opt,
                               "bitwise": r.bitwise, "compared": r.compared, "differing": r.differing,
                               "plain_differing": r.plain_differing, "plain_max_rel": r.plain_max_rel, "codes": r.codes,
                               "wall_s": time.perf_counter() - t, "seconds": r.seconds}
                        print(json.dumps(row), flush=True)
                        rows.append(row)
    return rows


def _time(fn, dev, n):
    fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return best


def part_speed(cuda: bool):
    rows = []
    devs = ["cpu"] + (["cuda"] if cuda else [])
    for dev in devs:
        for (m, k, n) in ((256, 512, 512), (1024, 1024, 1024)):
            g = torch.Generator().manual_seed(0)
            x = torch.randn(m, k, generator=g).to(dev).requires_grad_(True)
            w = torch.randn(n, k, generator=g).div(k ** 0.5).to(dev).requires_grad_(True)
            b = torch.zeros(n, device=dev, requires_grad=True)
            gy = torch.randn(m, n, generator=g).to(dev)

            def plain():
                y = F.linear(x, w, b)
                torch.autograd.grad(y, (x, w, b), gy)

            res = {"device": dev, "M": m, "K": k, "N": n, "plain_ms": 1e3 * _time(plain, dev, 5)}
            for leaf in ("block", "tiled", "exact"):
                spec = TreeSpec("full", 1, group_count(k, 8), group_count(n, 8), 8, leaf=leaf, tk=canon_tile(k, 8),
                                tn=canon_tile(n, 8))

                def tree(spec=spec):
                    y = tree_linear(x, w, b, spec)
                    torch.autograd.grad(y, (x, w, b), gy)
                reps = 3 if (leaf == "exact" and dev == "cpu") else 5
                res[f"{leaf}_ms"] = 1e3 * _time(tree, dev, reps)
                res[f"{leaf}_x"] = res[f"{leaf}_ms"] / res["plain_ms"]
                if leaf == "exact" and dev == "cuda":
                    # the same leaf in torch ops, without the Triton chunk kernel (tree_kernel.py)
                    os.environ["TRUESTAR_TREE_KERNEL"] = "0"
                    try:
                        res["exact_torch_ms"] = 1e3 * _time(tree, dev, 3)
                    finally:
                        os.environ.pop("TRUESTAR_TREE_KERNEL", None)
                    res["exact_torch_x"] = res["exact_torch_ms"] / res["plain_ms"]
            print(json.dumps(res), flush=True)
            rows.append(res)
    return rows


def part_machines(rank: int, master: str, leaf: str):
    """tp=2 with one rank per machine (gloo over the network, CPU). Each machine builds the same model,
    batch and targets from the example file and seeds; rank 0 also runs the one device step and
    compares both ranks' records with it bit for bit."""
    import datetime

    import torch.distributed as dist

    from truestar.cli import _load
    from truestar.parallel.runtime import MSE
    from truestar.parallel.tree import HostGlooComm, _rank_record, compare, run_step
    import parallel_model as pm

    model, x, _ = _load(MODEL)
    model = model.float()
    inputs = [x.float()]
    plan = {k: __import__("truestar.parallel.layout", fromlist=["parse_style"]).parse_style(v)
            for k, v in pm.get_parallel_plan(Layout(tp=2)).items()}
    with torch.no_grad():
        o = model(*inputs)
    targets = torch.randn(o.shape, generator=torch.Generator().manual_seed(7), dtype=torch.float64).float()
    kw = dict(optimizer="adamw", lr=1e-2, clip=1.0, cap=8, leaf=leaf)
    from truestar.parallel.devices import pin_gloo_interface
    pin_gloo_interface(master.split(":")[0])
    dist.init_process_group("gloo", init_method=f"tcp://{master}", rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=120))
    import copy
    import platform
    rec = _rank_record(run_step(copy.deepcopy(model), inputs, targets, MSE(), plan, tp=2, rank=rank,
                                comm=HostGlooComm(rank, 2), group=(0, 1), **kw))
    got = [None, None]
    dist.all_gather_object(got, {"rec": rec, "cpu": platform.processor(), "host": platform.node(),
                                 "torch": torch.__version__})
    dist.destroy_process_group()
    if rank != 0:
        print("rank 1 done", flush=True)
        return []
    ref = _rank_record(run_step(copy.deepcopy(model), inputs, targets, MSE(), plan, tp=1, **kw))
    compared, differing, names, worst = compare(ref, [g["rec"] for g in got], plan, 2)
    row = {"part": "machines", "leaf": leaf, "compared": compared, "differing": differing, "first": names,
           "ranks": [{k: g[k] for k in ("cpu", "host", "torch")} for g in got], "one_device_on": got[0]["cpu"]}
    print(json.dumps(row), flush=True)
    return [row]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["check", "speed", "machines"])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--master", default="192.0.2.10:29517")
    ap.add_argument("--leaf", default="exact")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.part == "machines":
        rows = part_machines(a.rank, a.master, a.leaf)
    else:
        rows = part_check(a.cuda) if a.part == "check" else part_speed(a.cuda)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"part": a.part, "torch": torch.__version__, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
