"""Speed of Truestar's gather and argmax kernels (compiler/gather_native.py) against the torch step
they replace (runtime.INDEX_IMPLS with its index check, coverage_rules.DEVICE_INT_IMPLS) and against
eager torch (the bare op, no check), per op at model sizes and in whole compiled programs.

usage: PYTHONPATH=src python scripts/bench_gather_native.py [--device cuda|cpu] [--json out.json]

Per op: median of ``--reps`` timed calls after warm up (CUDA events on the GPU, perf_counter on the
CPU), inputs made once and reused by all three. Whole programs: one program compiled twice, with the
native kernels and with them switched off (the torch step), median per call.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from truestar.compiler import gather_native as GN
from truestar.compiler import runtime as RT
from truestar.compiler.coverage_rules import DEVICE_INT_IMPLS


class _Rt:
    def __init__(self):
        self.pending, self.checked, self.node = [], {}, None


def _time(fn, device, reps, warm=5):
    for _ in range(warm):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        return statistics.median(ts)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


def op_cases(dev):
    g = torch.Generator().manual_seed(0)

    def r(*s):
        return torch.randn(*s, generator=g).to(dev)

    def ri(hi, *s):
        return torch.randint(0, hi, s, generator=g).to(dev)
    big = dev == "cuda"
    V, D, B, T = (50257, 768, 8, 512) if big else (30522, 768, 4, 128)
    w, ids = r(V, D), ri(V, B, T)
    yield ("embedding (index, GPT-2 vocab)", "index", {}, [w, ids], lambda: w[ids])
    w2, ids2 = r(30522, 768), ri(30522, 32 if big else 8, 128)
    yield ("embedding (index, BERT vocab)", "index", {}, [w2, ids2], lambda: w2[ids2])
    N, C = (4096, 32000) if big else (512, 32000)
    lsm, y = r(N, C), ri(C, N, 1)
    yield ("cross entropy class pick (gather)", "gather", {"dim": 1}, [lsm, y], lambda: torch.gather(lsm, 1, y))
    x = r(8, 64, 112, 112) if big else r(2, 64, 56, 56)
    iw = torch.arange(x.shape[3] * 2, device=dev) // 2
    yield ("nearest upsample x2 (index_select dim 3)", "index_select", {"dim": 3}, [x, iw],
           lambda: torch.index_select(x, 3, iw))
    xp = r(8, 64, 56, 56) if big else r(2, 64, 56, 56)
    ip = torch.tensor([2, 1] + list(range(56)) + [54, 53], device=dev)
    yield ("reflect pad 2 (index_select dim 3)", "index_select", {"dim": 3}, [xp, ip],
           lambda: torch.index_select(xp, 3, ip))
    xa = r(8, 512, 14, 14) if big else r(2, 512, 14, 14)
    rr = torch.tensor([[2 * i, 2 * i + 1] for i in range(7)], device=dev).reshape(7, 1, 2, 1)
    cc = torch.tensor([[2 * i, 2 * i + 1] for i in range(7)], device=dev).reshape(1, 7, 1, 2)
    yield ("adaptive max pool taps (index, 2 tensors)", "index", {}, [xa, None, None, rr, cc],
           lambda: xa[:, :, rr, cc])
    taps = r(8, 512, 7, 7, 4) if big else r(2, 512, 7, 7, 4)
    first = ri(4, *taps.shape[:-1], 1)
    yield ("adaptive max pool pick (gather dim 4)", "gather", {"dim": 4}, [taps, first],
           lambda: torch.gather(taps, 4, first))
    yield ("adaptive max pool first max (argmax dim -1)", "argmax", {"dim": -1, "keepdim": True}, [taps],
           lambda: torch.argmax(taps, dim=-1, keepdim=True))
    D2 = 512
    bag_rows, bag_ids = r(640, D2), (torch.arange(640, device=dev) // 10).reshape(640, 1).expand(640, D2)
    yield ("embedding bag sum, 64 bags of 10 (scatter_add)", "scatter_add", {"dim": 0},
           [torch.zeros(64, D2, device=dev), bag_ids, bag_rows],
           lambda: torch.zeros(64, D2, device=dev).scatter_add_(0, bag_ids, bag_rows))
    cnt_ids = ri(30522, 4096 if big else 512)
    ones = torch.ones(cnt_ids.shape[0], device=dev)
    yield ("token counts over a vocabulary (scatter_add, 1 column)", "scatter_add", {"dim": 0},
           [torch.zeros(30522, device=dev), cnt_ids, ones],
           lambda: torch.zeros(30522, device=dev).scatter_add_(0, cnt_ids, ones))
    lg = r(64, 32000) if big else r(16, 32000)
    yield ("greedy decode (argmax over vocab)", "argmax", {"dim": -1, "keepdim": False}, [lg],
           lambda: torch.argmax(lg, dim=-1))


def bench_ops(dev, reps):
    be = GN.backend_for(dev)
    rows = []
    for name, op, attrs, args, eager in op_cases(dev):
        rt = _Rt()
        if op in GN.ARG_OPS:
            def native():
                return GN.arg_extreme(op, attrs, args[0], None, be)

            def old():
                return DEVICE_INT_IMPLS[op](attrs, args, None)
            ok = torch.equal(native(), old())
        else:
            def native():
                if op == "scatter_add":
                    return GN.scatter_add(attrs, args, be)
                return GN.gather(op, attrs, args, be)

            def old():
                rt.checked = {}
                return RT.INDEX_IMPLS[op](rt, attrs, args)
            if native() is None:      # a shape the kernel leaves to the torch step
                rows.append({"case": name, "op": op, "truestar": "torch step (gated)"})
                print(json.dumps(rows[-1]), flush=True)
                continue
            ok = torch.equal(native()[0], old()[0]) if op != "scatter_add" else \
                bool((native()[0] - old()[0]).abs().max() <= 1e-4)
        t_new, t_old, t_eager = (_time(f, dev, reps) for f in (native, old, eager))
        rows.append({"case": name, "op": op, "equal": bool(ok), "truestar_us": round(t_new, 1),
                     "torch_step_us": round(t_old, 1), "eager_us": round(t_eager, 1),
                     "vs_torch_step": round(t_old / t_new, 2), "vs_eager": round(t_eager / t_new, 2)})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def programs(dev):
    import torch.nn as nn
    import torch.nn.functional as F
    from truestar import ir

    class LM(nn.Module):
        def __init__(self, v=8192, d=256):
            super().__init__()
            self.emb = nn.Embedding(v, d)
            self.out = nn.Linear(d, v)

        def forward(self, ids, y):
            h = self.emb(ids).mean(dim=1)
            return F.cross_entropy(self.out(h), y), self.out(h).argmax(dim=-1)

    g = torch.Generator().manual_seed(0)
    B = 64 if dev == "cuda" else 16
    ids, y = torch.randint(0, 8192, (B, 128), generator=g), torch.randint(0, 8192, (B,), generator=g)
    m = LM()
    yield "embedding + cross entropy + argmax", ir.import_module(m, (ids, y)), [ids, y]
    pool = nn.Sequential(nn.ReflectionPad2d(2), nn.Conv2d(16, 16, 3), nn.AdaptiveMaxPool2d((5, 5)),
                         nn.Upsample(scale_factor=2, mode="nearest"))
    x = torch.randn(8 if dev == "cuda" else 2, 16, 40, 40, generator=g)
    yield "reflect pad + conv + adaptive max pool + nearest upsample", ir.import_module(pool, (x,)), [x]
    bag = nn.EmbeddingBag(8192, 256, mode="mean")
    bids = torch.randint(0, 8192, (B, 32), generator=g)
    yield "embedding bag (mean)", ir.import_module(bag, (bids,)), [bids]


def bench_programs(dev, reps):
    from truestar.compiler import compile_program
    rows = []
    keep = GN.backend_for
    for name, prog, xs in programs(dev):
        xs = [x.to(dev) for x in xs]
        res = {}
        for mode in ("truestar", "torch_step"):
            GN.backend_for = keep if mode == "truestar" else (lambda d: None)
            cp = compile_program(prog, dev, "float32")
            outs = cp(xs)
            impl = sorted({st.get("impl", "") for st in cp.report.library if st["ops"][0] in GN.GATHER_OPS + GN.ARG_OPS + GN.SCATTER_OPS})
            res[mode] = (_time(lambda: cp(xs), dev, reps), impl, [o.detach().cpu() for o in outs])
        GN.backend_for = keep
        same = all(torch.equal(a, b) for a, b in zip(res["truestar"][2], res["torch_step"][2]))
        rows.append({"program": name, "truestar_us": round(res["truestar"][0], 1),
                     "torch_step_us": round(res["torch_step"][0], 1),
                     "vs_torch_step": round(res["torch_step"][0] / res["truestar"][0], 2),
                     "outputs_bitwise_equal": same, "impl_truestar": res["truestar"][1],
                     "impl_torch_step": res["torch_step"][1]})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--json")
    a = ap.parse_args()
    env = {"device": a.device, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name() if a.device == "cuda" else None,
           "threads": torch.get_num_threads(), "statistic": f"median of {a.reps}"}
    print(json.dumps(env), flush=True)
    out = {"env": env, "ops": bench_ops(a.device, a.reps), "programs": bench_programs(a.device, max(10, a.reps // 2))}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
