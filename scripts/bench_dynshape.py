"""Dynamic shapes against one static compile per shape (docs/DYNAMIC_SHAPES.md).

For an MLP over 20 batch sizes and a transformer encoder layer and a decoder prefill block over 20
sequence lengths: compile count and total compile time (Truestar compile, Triton variants and the
first verified call), steady step time at each shape, and bitwise equality of every output.

  python scripts/bench_dynshape.py [--device cuda] [--out results/dynshape.json]

Each arm runs in its own process with empty Truestar and Triton caches, so no arm reuses another's
compiled kernels."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

SIZES = {"mlp": [1, 2, 3, 5, 7, 8, 12, 16, 17, 24, 31, 32, 40, 48, 63, 64, 80, 100, 127, 128],
         "encoder": [3, 5, 8, 11, 16, 20, 27, 32, 40, 50, 64, 77, 96, 100, 128, 150, 200, 256, 300, 384],
         "prefill": [3, 5, 8, 11, 16, 20, 27, 32, 40, 50, 64, 77, 96, 100, 128, 150, 200, 256, 300, 384]}


def build(name):
    import torch
    from torch import nn
    torch.manual_seed(0)
    if name == "mlp":
        m = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256), nn.Softmax(-1))
        return m.eval(), [{0: "B"}], {"B": (1, 128)}, lambda n: (n, 256)
    if name == "encoder":
        m = nn.TransformerEncoderLayer(256, 8, 1024, dropout=0.0, batch_first=True)
        return m.eval(), [{1: "T"}], {"T": (2, 512)}, lambda n: (4, n, 256)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    from test_dynamic_shapes import Prefill
    return Prefill(256, 8).eval(), [{1: "T"}], {"T": (2, 512)}, lambda n: (4, n, 256)


def _sync(dev):
    import torch
    if dev == "cuda":
        torch.cuda.synchronize()


def _step_ms(f, x, dev, reps):
    import torch
    for _ in range(3):
        f(x)
    _sync(dev)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f(x)
        _sync(dev)
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    del torch
    return ts[len(ts) // 2]


def arm(name, mode, dev, verify, out_path):
    import torch
    import truestar.ir as ir
    from truestar.compiler import compile_program
    from truestar.compiler.dynamic import compile_dynamic
    variants = [0]
    try:
        import triton

        def hook(**kw):
            variants[0] += 1
            return False
        triton.knobs.runtime.jit_post_compile_hook = hook
    except Exception:
        pass
    model, axes, dims, shape = build(name)
    sizes = SIZES[name]
    g = torch.Generator().manual_seed(1)
    xs = {n: torch.randn(*shape(n), generator=g) for n in sizes}
    res = {"name": name, "mode": mode, "verify": verify, "sizes": sizes, "per_shape": []}
    t_all = time.perf_counter()
    if mode == "dynamic":
        t = time.perf_counter()
        dp = compile_dynamic(model, (xs[sizes[len(sizes) // 2]],), axes=axes, dims=dims, device=dev, verify=verify)
        res["truestar_compile_s"] = time.perf_counter() - t
        res["compiles"] = 1
        for n in sizes:
            x = [xs[n].to(dev)]
            v0, t = variants[0], time.perf_counter()
            y = dp(x)[0]
            _sync(dev)
            first = time.perf_counter() - t
            res["per_shape"].append({"n": n, "first_s": first, "triton_variants": variants[0] - v0,
                                     "bucket": dp.bucket_of(dp.bind(x)),
                                     "step_ms": _step_ms(dp, x, dev, 20)})
            torch.save(y.cpu(), os.path.join(out_path, f"{name}_{n}.pt"))
        res["kernel_checks_skipped"] = dp.stats.kernel_checks_skipped
        res["buckets"] = len(dp.stats.buckets)
    else:
        res["compiles"] = 0
        res["truestar_compile_s"] = 0.0
        for n in sizes:
            x = [xs[n].to(dev)]
            v0, t = variants[0], time.perf_counter()
            cp = compile_program(ir.import_module(model, (xs[n],)), dev, deterministic=True)
            tc = time.perf_counter() - t
            y = cp(x)[0]
            _sync(dev)
            first = time.perf_counter() - t
            res["compiles"] += 1
            res["truestar_compile_s"] += tc
            res["per_shape"].append({"n": n, "first_s": first, "import_compile_s": tc,
                                     "triton_variants": variants[0] - v0, "step_ms": _step_ms(cp, x, dev, 20)})
            ref = os.path.join(out_path, f"{name}_{n}.pt")
            if os.path.exists(ref):
                d = torch.load(ref)
                yi = y.cpu().contiguous().view(torch.int32)
                res["per_shape"][-1]["bit_differences"] = int((yi != d.contiguous().view(torch.int32)).sum())
                res["per_shape"][-1]["max_abs_diff"] = float((y.cpu() - d).abs().max())
    res["total_first_s"] = sum(r["first_s"] for r in res["per_shape"])
    res["wall_s"] = time.perf_counter() - t_all
    res["triton_variants"] = variants[0]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", default="mlp,encoder,prefill")
    ap.add_argument("--arm", default=None)   # internal: name,mode,verify,tensor dir
    a = ap.parse_args()
    if a.arm:
        name, mode, verify, tdir = a.arm.split(",")
        print("RESULT " + json.dumps(arm(name, mode, a.device, verify, tdir)), flush=True)
        return
    rows = []
    for name in a.models.split(","):
        tdir = tempfile.mkdtemp(prefix="dynshape_out_")
        for mode, verify in (("dynamic", "bucket"), ("dynamic", "shape"), ("static", "shape")):
            env = dict(os.environ, TRUESTAR_CACHE_DIR=tempfile.mkdtemp(prefix="ts_k_"),
                       TRITON_CACHE_DIR=tempfile.mkdtemp(prefix="ts_t_"),
                       TRUESTAR_TUNE_DIR=tempfile.mkdtemp(prefix="ts_tune_"))
            p = subprocess.run([sys.executable, __file__, "--device", a.device, "--arm", f"{name},{mode},{verify},{tdir}"],
                               env=env, capture_output=True, text=True)
            line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
            if not line:
                print(p.stdout[-3000:], p.stderr[-3000:])
                raise SystemExit(f"arm {name} {mode} failed")
            r = json.loads(line[0][7:])
            rows.append(r)
            ps = r["per_shape"]
            bits = [x.get("bit_differences") for x in ps]
            print(f"{name:8s} {mode:7s} verify={verify:6s} compiles={r['compiles']:2d} "
                  f"triton_variants={r['triton_variants']:3d} truestar_compile={r['truestar_compile_s']:.2f}s "
                  f"total_first_calls={r['total_first_s']:.2f}s wall={r['wall_s']:.1f}s "
                  f"step_ms_median_over_shapes={sorted(x['step_ms'] for x in ps)[len(ps) // 2]:.3f}"
                  + (f" bit_differences={bits}" if mode == "static" else ""), flush=True)
        st = rows[-1]["per_shape"]
        dy = rows[-3]["per_shape"]
        ratio = [d["step_ms"] / s["step_ms"] for d, s in zip(dy, st)]
        print(f"{name:8s} step time dynamic/static per shape: min {min(ratio):.3f} median "
              f"{sorted(ratio)[len(ratio) // 2]:.3f} max {max(ratio):.3f}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)
    for r in rows:      # compact per shape lines (the remote runner keeps only the tail of stdout)
        print(r["name"], r["mode"], r["verify"], " ".join(
            f"{x['n']}:{x['first_s']:.2f}s/{x['triton_variants']}v/{x['step_ms']:.3f}ms"
            + (f"/{x['bit_differences']}b" if "bit_differences" in x else "") for x in r["per_shape"]))


if __name__ == "__main__":
    main()
