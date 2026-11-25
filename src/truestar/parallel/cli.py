"""`truestar parallel model.py --layout tp=2,dp=2,ga=2,sp=1`: the command line side of truestar.parallel.

The model file is the one `truestar check` takes (get_model, get_inputs) plus optional get_targets()
(default: a fixed random target, mean squared error), get_loss() (an object with sum(out, target) and
count(target), such as truestar.parallel.CrossEntropy()) and get_parallel_plan(layout) (module name to
style; without it only unambiguous MLP pairs are split). Exit codes: 0 the layout computes the single
device step, 1 a finding, 2 usage error (TS680 for a layout or plan that cannot run).
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import sys

EXIT_USAGE = 2


def add_parser(sub) -> None:
    r = sub.add_parser("parallel", help="check one training step under a parallel layout against the single "
                                        "device step, on simulated ranks (docs/PARALLEL.md)")
    r.add_argument("model", help="python file defining get_model() and get_inputs()")
    r.add_argument("--layout", required=True, help="tp=N,dp=N,ga=N,sp=0|1,pp=1 (example tp=2,dp=2)")
    r.add_argument("--optimizer", default="sgd", choices=("sgd", "adamw"))
    r.add_argument("--lr", type=float, default=0.1)
    r.add_argument("--clip", type=float, default=0.0, help="clip gradients to this global norm (0: off)")
    r.add_argument("--ranks", default="threads", choices=("threads", "gloo", "devices"),
                   help="threads (default, one process), gloo (one CPU process per rank, simulated) or devices "
                        "(one process per real device: NCCL between GPUs on Linux, gloo elsewhere)")
    r.add_argument("--device", default="cuda", choices=("cuda", "cpu"),
                   help="with --ranks devices: one GPU per rank (default) or CPU processes")
    r.add_argument("--backend", default="auto", choices=("auto", "nccl", "gloo"))
    r.add_argument("--order", default="fixed", choices=("fixed", "native"),
                   help="sums over ranks: fixed (all gather, then a rank order sum; bitwise reproducible) or "
                        "native (the backend's all reduce)")
    r.add_argument("--bucket-mb", type=float, default=25.0, help="gradient bucket size with --ranks devices")
    r.add_argument("--no-overlap", action="store_true",
                   help="reduce gradient buckets after the backward pass instead of during it")
    r.add_argument("--dist-master", metavar="HOST:PORT",
                   help="spread the ranks over machines: rank 0's address (it listens on PORT)")
    r.add_argument("--dist-world", type=int, help="number of ranks over all machines (with --dist-master)")
    r.add_argument("--dist-rank", type=int, help="this machine's rank (with --dist-master); rank 0 writes the report")
    r.add_argument("--inject", action="append", default=[], metavar="FAULT",
                   help="switch in a known layout bug to see the check name it (docs/PARALLEL.md lists them)")
    r.add_argument("--timeout", type=float, default=60.0, help="seconds a collective may wait")
    r.add_argument("--json", dest="json_path", help="also write the agent JSON here")
    r.add_argument("--format", choices=("text", "agent"), default="text")


def run(a) -> int:
    from ..agent import dumps, error_to_agent

    agent = a.format == "agent"

    def fail(code: str, msg: str) -> int:
        if agent:
            doc = error_to_agent(code, msg, a.model)
            doc["command"] = "parallel"
            print(dumps(doc))
        else:
            print(f"error [{code}]: {msg}", file=sys.stderr)
        return EXIT_USAGE

    quiet = contextlib.redirect_stdout(sys.stderr) if agent else contextlib.nullcontext()
    failure, rep = None, None
    with quiet:  # in agent mode stdout carries only the JSON document
        failure, rep = _check(a)
    if failure is not None and failure[0] == "served":
        if agent:
            from ..agent import _envelope
            doc = _envelope("served", 0)
            doc.update({"command": "parallel", "next_actions": [],
                        "message": "this rank ran its step and sent its record to rank 0, which writes the report"})
            print(dumps(doc))
        return 0
    if failure is not None:
        return fail(*failure)
    doc = rep.to_agent(a.model)
    if agent:
        print(dumps(doc))
    else:
        print(rep.to_text())
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(doc, indent=2, allow_nan=False))
    return 1 if rep.status == "divergent" else 0


def _check(a):
    from ..cli import ModelFileError, _load
    from . import LayoutError, parallel_check, parse_layout
    from .devices import DevicesUnavailable, RendezvousError

    dist = None
    if a.dist_master or a.dist_world is not None or a.dist_rank is not None:
        if not (a.dist_master and a.dist_world and a.dist_rank is not None):
            return ("TS680", "--dist-master, --dist-world and --dist-rank go together"), None
        if a.ranks != "devices":
            return ("TS680", "--dist-master needs --ranks devices"), None
        dist = {"master": a.dist_master, "world": a.dist_world, "rank": a.dist_rank}
    sync = {"bucket_bytes": int(a.bucket_mb * (1 << 20)), "overlap": not a.no_overlap}
    if dist is not None and dist["rank"] != 0:
        from .check import serve_rank
        try:
            print(serve_rank(a.model, dist, device=a.device, backend=a.backend, order=a.order, sync=sync,
                             timeout=a.timeout), file=sys.stderr)
        except DevicesUnavailable as exc:
            return ("TS821", str(exc)), None
        except RendezvousError as exc:
            return ("TS822", str(exc)), None
        return ("served", ""), None

    try:
        layout = parse_layout(a.layout)
    except LayoutError as exc:
        return ("TS680", str(exc)), None
    try:
        model, inputs, kwargs = _load(a.model)
    except ModelFileError as exc:
        return ("TS020", str(exc)), None
    if kwargs:
        return ("TS021", "truestar parallel takes positional tensor inputs only; get_kwargs() is not supported"), None
    mod = next((m for m in list(sys.modules.values())
                if getattr(m, "__file__", None) and os.path.abspath(m.__file__) == os.path.abspath(a.model)), None)
    targets = mod.get_targets() if hasattr(mod, "get_targets") else None
    loss = mod.get_loss() if hasattr(mod, "get_loss") else None
    plan = None
    if hasattr(mod, "get_parallel_plan"):
        fn = mod.get_parallel_plan
        plan = fn(layout) if inspect.signature(fn).parameters else fn()
    try:
        rep = parallel_check(model, inputs, layout, targets=targets, loss=loss, plan=plan, optimizer=a.optimizer,
                             lr=a.lr, clip=a.clip, faults=a.inject, ranks=a.ranks, model_file=a.model,
                             timeout=a.timeout, model_name=os.path.splitext(os.path.basename(a.model))[0],
                             device=a.device, backend=a.backend, order=a.order, sync=sync, dist=dist)
    except LayoutError as exc:
        return ("TS680", str(exc)), None
    except DevicesUnavailable as exc:
        return ("TS821", str(exc)), None
    except RendezvousError as exc:
        return ("TS822", str(exc)), None
    return None, rep
