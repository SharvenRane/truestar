"""Speed and accuracy of order free sums (docs/ORDER_FREE_SUM.md) against plain float32 sums.

Run: PYTHONPATH=src python scripts/bench_order_free_sum.py [--out results/order_free_sum/bench.json]

Measures on the local CUDA GPU (keep runs short, the GPU is shared):
* kernel IR row sums (tile single and tile looped shapes), softmax and layer norm compiled with
  and without order_free, each with its default launch config and tuned;
* a whole tensor sum: the two kernel Triton path against torch.sum;
* bits: whether each result is equal across launch configs (osum must be, plain may not be);
* accuracy: worst relative error against a float64 sum for both.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from truestar import kernel as K  # noqa: E402
from truestar.kernel import ordersum as O  # noqa: E402


def timed(fn, reps=100):
    for _ in range(3):
        fn()
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


def bits(t):
    return t.contiguous().view(torch.int32)


def rowsum_kernel():
    M, N = K.sym("M"), K.sym("N")
    x = K.buffer("x", (M, N), "float32")
    i, r = K.axis("i", M), K.axis("r", N)
    return K.kernel("rowsum", [K.output("y", (i,), K.reduce_sum(r, x[i, r]))])


def softmax_kernel():
    M, N = K.sym("M"), K.sym("N")
    x = K.buffer("x", (M, N), "float32")
    i, j = K.axis("i", M), K.axis("j", N)
    return K.kernel("softmax", [K.output("y", (i, j), K.softmax_weight(x[i, j], over=j))])


def layernorm_graph_kernel():
    M, N = K.sym("M"), K.sym("N")
    x = K.buffer("x", (M, N), "float32")
    i, j, r1, r2 = K.axis("i", M), K.axis("j", N), K.axis("r1", N), K.axis("r2", N)
    mean = K.reduce_sum(r1, x[i, r1]) / K.dim(N)
    d = x[i, r2] - mean
    var = K.reduce_sum(r2, d * d) / K.dim(N)
    return K.kernel("layernorm", [K.output("y", (i, j), (x[i, j] - mean) * K.rsqrt(var + 1e-5))])


def configs(kind):
    if kind == "single":
        return [dict(BR=1, num_warps=4), dict(BR=4, num_warps=8), dict(BR=8, num_warps=4)]
    return [dict(BC=1024, num_warps=4), dict(BC=2048, num_warps=8), dict(BC=4096, num_warps=8)]


def bench_kernel(name, k, shape, out_name="y"):
    rows, n = shape
    torch.manual_seed(0)
    x = torch.randn(rows, n, device="cuda") * torch.exp2(torch.randint(-8, 8, (rows, n), device="cuda").float())
    kind = "single" if n <= 4096 else "looped"
    res = {"kernel": name, "shape": list(shape), "loop_shape": f"tile_{kind}"}
    for mode in ("plain", "order_free"):
        of = mode == "order_free"
        outs = []
        for c in configs(kind):
            fn = K.compile(k, "cuda", order_free=of, _config=c)
            o = fn(x=x)
            outs.append(o[out_name] if isinstance(o, dict) else o)
        equal = all(torch.equal(bits(outs[0]), bits(o)) for o in outs[1:])
        fdef = K.compile(k, "cuda", order_free=of, tune=False)
        t_def = timed(lambda: fdef(x=x))
        ftun = K.compile(k, "cuda", order_free=of, tune=True, deterministic=True)
        t_tun = timed(lambda: ftun(x=x))
        res[mode] = {"us_default_config": round(t_def, 2), "us_tuned_deterministic": round(t_tun, 2),
                     "bits_equal_over_configs": equal, "configs": len(outs)}
    res["slowdown_default"] = round(res["order_free"]["us_default_config"] / res["plain"]["us_default_config"], 3)
    res["slowdown_tuned_deterministic"] = round(res["order_free"]["us_tuned_deterministic"]
                                                / res["plain"]["us_tuned_deterministic"], 3)
    return res


def accuracy(rows=256, n=4096):
    torch.manual_seed(3)
    out = {}
    cases = {
        "gaussian": torch.randn(rows, n),
        "wide_exponents": torch.randn(rows, n) * torch.exp2(torch.randint(-30, 30, (rows, n)).float()),
        "cancelling": torch.cat([a := torch.randn(rows, n // 2) * 1e4, -a + torch.randn(rows, n // 2) * 1e-2], 1),
    }
    for name, x in cases.items():
        ref = x.double().sum(1)
        scale = x.double().abs().sum(1)
        e_plain = ((x.cuda().sum(1).cpu().double() - ref).abs() / scale).max().item()
        e_os = ((O.order_free_sum(x.cuda(), 1).cpu().double() - ref).abs() / scale).max().item()
        bound_ok = bool(((O.order_free_sum(x, 1).double() - ref).abs()
                         <= O.error_bound(x, 1) + O.order_free_sum(x, 1).abs().double() * 2.0 ** -24).all())
        out[name] = {"plain_cuda_sum_err_over_sum_abs": e_plain, "osum_err_over_sum_abs": e_os,
                     "osum_within_proven_bound": bound_ok}
    return out


def whole_sum(n):
    torch.manual_seed(1)
    x = torch.randn(n, device="cuda")
    ref = O.order_free_sum_cuda(x)
    same = all(torch.equal(bits(ref), bits(O.order_free_sum_cuda(x, block=b, programs=p)))
               for b, p in ((256, 1), (512, 7), (1024, 70), (2048, 1000), (4096, 33)))
    perm = torch.randperm(n, device="cuda")
    same_perm = torch.equal(bits(ref), bits(O.order_free_sum_cuda(x[perm])))
    plain_perm = torch.equal(bits(x.sum()), bits(x[perm].sum()))
    return {"n": n, "us_torch_sum": round(timed(lambda: x.sum()), 2),
            "us_order_free_sum_cuda": round(timed(lambda: O.order_free_sum_cuda(x)), 2),
            "us_order_free_sum_torch_ops": round(timed(lambda: O.order_free_sum(x, 0), reps=10), 2),
            "bits_equal_over_5_block_and_program_counts": same,
            "bits_equal_after_permutation": same_perm,
            "torch_sum_bits_equal_after_permutation": plain_perm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("results", "order_free_sum", "bench.json"))
    a = ap.parse_args()
    doc = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "kernels": []}
    for name, k in (("rowsum", rowsum_kernel()), ("softmax", softmax_kernel()),
                    ("layernorm", layernorm_graph_kernel())):
        for shape in ((4096, 1024), (1024, 4096), (64, 65536)):
            r = bench_kernel(name, k, shape)
            print(json.dumps(r))
            doc["kernels"].append(r)
    doc["whole_sum"] = [whole_sum(n) for n in (1 << 20, (1 << 24) - 1)]
    print(json.dumps(doc["whole_sum"]))
    doc["accuracy"] = accuracy()
    print(json.dumps(doc["accuracy"]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)


if __name__ == "__main__":
    main()
