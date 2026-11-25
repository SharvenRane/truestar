"""Side by side: torch.compile (Dynamo) versus the Truestar native front end on small programs.

Counts graph breaks with torch._dynamo.explain and compiles with a counting backend (every graph
Dynamo hands to a backend is one compile; the backend then runs the graph eagerly, so the numbers
measure Dynamo itself, not Inductor). CPU only. Prints a JSON report.

    PYTHONPATH=src python scripts/frontend_vs_torch_compile.py
"""

from __future__ import annotations

import json
import os
import time
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")

import torch  # noqa: E402

# With the GPU hidden, torch still reports CUDA as available with 0 devices and Dynamo then crashes
# while probing Triton (IndexError in get_device_properties). Make the CPU only setup consistent.
torch.cuda.is_available = lambda: False

import torch._dynamo as dynamo  # noqa: E402
from torch._dynamo.utils import counters  # noqa: E402

import truestar  # noqa: E402
from truestar import Tensor, ops  # noqa: E402
from truestar.frontend import FrontendShapeError  # noqa: E402

torch.manual_seed(0)


class Counting:
    def __init__(self):
        self.graphs = 0
        self.nodes = []

    def __call__(self, gm, example_inputs):
        self.graphs += 1
        self.nodes.append(sum(1 for n in gm.graph.nodes if n.op == "call_function"))
        return gm.forward


def explain(fn, *args):
    dynamo.reset()
    e = dynamo.explain(fn)(*args)
    return {"graphs": e.graph_count, "graph_breaks": e.graph_break_count,
            "break_reasons": sorted({str(r.reason).splitlines()[0][:160] for r in e.break_reasons})}


def count_compiles(fn, inputs, dynamic=None):
    dynamo.reset()
    counters.clear()
    be = Counting()
    cf = torch.compile(fn, backend=be, dynamic=dynamic)
    t0 = time.perf_counter()
    for args in inputs:
        cf(*args)
    return {"calls": len(inputs), "backend_compiles": be.graphs, "fx_ops_per_graph": be.nodes,
            "frames_compiled": int(counters["frames"]["ok"]),
            "graph_breaks_hit": int(sum(counters["unimplemented"].values())),
            "seconds": round(time.perf_counter() - t0, 3)}


def fullgraph_error(fn, *args):
    dynamo.reset()
    try:
        torch.compile(fn, backend=Counting(), fullgraph=True)(*args)
        return None
    except Exception as e:  # report what users see
        return f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"


def timed_decorate(make):
    t0 = time.perf_counter()
    f = make()
    return f, round(time.perf_counter() - t0, 4)


# ----------------------------------------------------------------------------- program 1: data branch


def branch_torch(x, w):
    h = x @ w
    if h.sum() > 0:
        h = torch.relu(h)
    else:
        h = -h * 0.5
    return h


def branch_torch_cond(x, w):
    # the manual rewrite PyTorch recommends for data dependent branches
    h = x @ w
    return torch.cond(h.sum() > 0, lambda h: torch.relu(h), lambda h: -h * 0.5, (h,))


def make_branch():
    @truestar.function
    def branch(x: Tensor["B", 64], w: Tensor[64, 32]):
        h = x @ w
        if h.sum() > 0:
            h = ops.relu(h)
        else:
            h = -h * 0.5
        return h
    return branch


# ----------------------------------------------------------------------------- program 2: loop over T


def rnn_torch(x, wx, wh):
    h = torch.zeros(x.shape[0], 32)
    for t in range(x.shape[1]):
        h = torch.tanh(x[:, t] @ wx + h @ wh)
    return h


def make_rnn():
    @truestar.function
    def rnn(x: Tensor["B", "T", 16], wx: Tensor[16, 32], wh: Tensor[32, 32]):
        h = ops.zeros((x.shape[0], 32))
        for t in range(x.shape[1]):
            h = ops.tanh(x[:, t] @ wx + h @ wh)
        return h
    return rnn


# ----------------------------------------------------------------------------- program 3: shape bug


def buggy_torch(x, w):
    h = torch.relu(x @ w)
    if h.mean() > 10.0:   # rare on normal data
        h = h @ w         # bug: [B, 32] @ [64, 32]
    return h


def make_buggy():
    @truestar.function
    def buggy(x: Tensor["B", 64], w: Tensor[64, 32]):
        h = ops.relu(x @ w)
        if h.mean() > 10.0:
            h = h @ w
        return h
    return buggy


# ----------------------------------------------------------------------------- control: branch free MLP


def mlp_torch(x, w1, w2):
    return torch.softmax(torch.relu(x @ w1) @ w2, dim=-1)


def make_mlp():
    @truestar.function
    def mlp(x: Tensor["B", 64], w1: Tensor[64, 64], w2: Tensor[64, 10]):
        return ops.softmax(ops.relu(x @ w1) @ w2, dim=-1)
    return mlp


def main():
    report = {"torch": torch.__version__, "device": "cpu"}
    batches = [2, 3, 4, 8, 16, 1]

    # 1. data dependent branch, data alternates sign so both branches run
    w = torch.randn(64, 32)
    ins = [((1 if i % 2 else -1) * torch.ones(b, 64) + 0.1 * torch.randn(b, 64), w) for i, b in enumerate(batches)]
    f, dt = timed_decorate(make_branch)
    before = f.compile_count
    for x, ww in ins:
        f(x, ww)
    report["data_branch"] = {
        "torch_explain": explain(branch_torch, *ins[0]),
        "torch_fullgraph": fullgraph_error(branch_torch, *ins[0]),
        "torch_default": count_compiles(branch_torch, ins),
        "torch_dynamic_true": count_compiles(branch_torch, ins, dynamic=True),
        "torch_cond_rewrite_fullgraph": fullgraph_error(branch_torch_cond, *ins[0]) or "ok",
        "torch_cond_rewrite": count_compiles(branch_torch_cond, ins),
        "truestar": {"graph_breaks": 0, "compiles": f.compile_count, "compiles_during_calls": f.compile_count - before,
                     "decorate_seconds": dt, "ir_ops": f.program.op_counts()},
    }

    # 2. loop over a symbolic length
    wx, wh = torch.randn(16, 32), torch.randn(32, 32) * 0.1
    lengths = list(range(1, 13))
    ins = [(torch.randn(4, T, 16), wx, wh) for T in lengths]
    f, dt = timed_decorate(make_rnn)
    for args in ins:
        f(*args)
    report["rnn_loop"] = {
        "lengths": lengths,
        "torch_explain_T5": explain(rnn_torch, torch.randn(4, 5, 16), wx, wh),
        "torch_default": count_compiles(rnn_torch, ins),
        "torch_dynamic_true": count_compiles(rnn_torch, ins, dynamic=True),
        "recompile_limit": getattr(dynamo.config, "recompile_limit", getattr(dynamo.config, "cache_size_limit", None)),
        "truestar": {"graph_breaks": 0, "compiles": f.compile_count, "decorate_seconds": dt,
                     "ir_ops": f.program.op_counts()},
    }

    # 3. shape bug in a rarely taken branch
    w = torch.randn(64, 32)
    normal = [(torch.randn(b, 64), w) for b in (2, 4, 8) for _ in range(10)]
    dynamo.reset()
    cf = torch.compile(buggy_torch, backend=Counting())
    errors_normal = 0
    for x, ww in normal:
        try:
            cf(x, ww)
        except Exception:
            errors_normal += 1
    try:
        cf(torch.full((2, 64), 5.0), torch.ones(64, 32))
        trigger = None
    except Exception as e:
        import re
        inner = re.search(r"got RuntimeError\('([^']*)'\)", str(e))
        trigger = f"{type(e).__name__} at the call: " + (inner.group(1) if inner else str(e).splitlines()[0][:200])
    try:
        make_buggy()
        ts_err = None
    except FrontendShapeError as e:
        ts_err = {"raised_at": "decoration (before any data)", "line": e.line, "op": e.op,
                  "operands": [f"{n}: {t}" for n, t in e.operands], "suggestion": e.suggestion}
    report["shape_bug"] = {
        "torch_compile_calls_on_normal_data": len(normal), "torch_compile_errors_on_normal_data": errors_normal,
        "torch_compile_on_triggering_data": trigger,
        "torch_explain": explain(buggy_torch, *normal[0]),
        "truestar": ts_err,
    }

    # control: branch free MLP, where torch.compile does fine
    w1, w2 = torch.randn(64, 64), torch.randn(64, 10)
    ins = [(torch.randn(b, 64), w1, w2) for b in batches]
    f, dt = timed_decorate(make_mlp)
    for args in ins:
        f(*args)
    report["mlp_control"] = {
        "torch_explain": explain(mlp_torch, *ins[0]),
        "torch_default": count_compiles(mlp_torch, ins),
        "torch_dynamic_true": count_compiles(mlp_torch, ins, dynamic=True),
        "truestar": {"graph_breaks": 0, "compiles": f.compile_count, "decorate_seconds": dt},
    }
    report["batches"] = batches
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
