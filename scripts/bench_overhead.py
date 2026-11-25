"""Host overhead of the kernel layer launch path, and what CUDA graph capture removes.

    PYTHONPATH=src python scripts/bench_overhead.py                 # all phases, one at a time
    PYTHONPATH=src python scripts/bench_overhead.py --phase kernels # one phase

Phases (each a separate process, run one after another so only one GPU process runs at a time):

  kernels  per launch host time of single kernel layer kernels (elementwise chain, softmax, layer
           norm, causal attention) at a small and a medium size, split into its parts (argument
           gathering, signature, cache lookup, verify mode check, output allocation, Triton's
           Python launch, Triton's C launcher alone), against the kernel's GPU time
  chain    a 50 kernel step (ten pre norm MLP blocks, five kernel layer calls each) at a small and a
           medium size: Truestar eager, Truestar captured, PyTorch eager, PyTorch eager captured by
           the same layer, torch.compile mode="reduce-overhead"
  single   the four single kernels at the small size: Truestar eager and captured against PyTorch
           eager and torch.compile mode="reduce-overhead"

All numbers are microseconds, medians of several repeats (see truestar.runtime.timing). The GPU
is shared with other work during development runs, so single numbers can be noisy; medians are
reported and the run is meant to be repeated.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time


def _setup():
    import torch
    os.environ.setdefault("TRUESTAR_VERIFY", "first")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def _cases(small: bool):
    from truestar.runtime import workloads as W
    if small:
        return [("gelu+res 64x512 fp32", lambda: W.gelu_residual(64, 512)),
                ("softmax 64x512 fp32", lambda: W.softmax(64, 512)),
                ("layer norm 64x768 fp32", lambda: W.layer_norm(64, 768)),
                ("causal attn 1x4x128x64 fp16", lambda: W.attention(1, 4, 128, 64))]
    return [("gelu+res 4096x4096 fp32", lambda: W.gelu_residual(4096, 4096)),
            ("softmax 4096x4096 fp32", lambda: W.softmax(4096, 4096)),
            ("layer norm 8192x1024 fp32", lambda: W.layer_norm(8192, 1024)),
            ("causal attn 2x8x1024x64 fp16", lambda: W.attention(2, 8, 1024, 64))]


def _raw_launcher(L, args):
    """Triton's compiled C launcher with all arguments precomputed: the floor a native launcher
    pays per kernel (argument packing in C plus cuLaunchKernel)."""
    from triton.runtime import driver
    kern = L.fn.run(*args, grid=L.grid, warmup=True, **L.meta)
    device = driver.active.get_current_device()
    stream = driver.active.get_current_stream(device)
    binder = L.fn.device_caches[device][4]
    bound, _, _ = binder(*args, **L.meta)
    vals = tuple(bound.values())
    g = tuple(L.grid) + (1,) * (3 - len(L.grid))
    lm = kern.launch_metadata(L.grid, stream, *vals)

    def run():
        kern.run(g[0], g[1], g[2], stream, kern.function, kern.packed_metadata, lm, None, None,
                 *vals)
    return run


def phase_kernels(args):
    torch = _setup()
    from truestar.kernel import verify
    from truestar.runtime import capture, timing
    rows = []
    for small in (True, False):
        for name, build in _cases(small):
            ck, tfn, ins = build()
            ck(**ins)  # compile, first call verification
            torch.cuda.synchronize()
            got = ck._gather((), ins)
            sig = ck._signature(got)
            L = ck._fast[sig]
            outs = [torch.empty(shape, dtype=dt, device=L.device)
                    for (_, _, shape, _), dt in zip(L.out_specs, L.torch_dtypes)]
            launch_args = [got[n] for n in L.in_order] + outs + list(L.int_args)
            raw = _raw_launcher(L, launch_args)

            def triton_launch():
                L.fn[L.grid](*launch_args, **L.meta)

            def alloc():
                return [torch.empty(shape, dtype=dt, device=L.device)
                        for (_, _, shape, _), dt in zip(L.out_specs, L.torch_dtypes)]

            r = dict(case=name, size="small" if small else "medium")
            r["gather_us"] = timing.host_only(lambda: ck._gather((), ins))
            r["signature_us"] = timing.host_only(lambda: ck._signature(got))
            r["lookup_us"] = timing.host_only(lambda: ck._fast.get(sig))
            r["verify_check_us"] = timing.host_only(verify.verify_mode)
            r["alloc_us"] = timing.host_only(alloc)
            r["triton_launch_us"] = timing.host_and_wall(triton_launch)["host_us"]
            r["c_launcher_us"] = timing.host_and_wall(raw)["host_us"]
            hw = timing.host_and_wall(lambda: ck(**ins))
            r["truestar_host_us"], r["truestar_wall_us"] = hw["host_us"], hw["wall_us"]
            r["gpu_us"] = timing.device_time(lambda: ck(**ins))
            th = timing.host_and_wall(lambda: tfn(**ins))
            r["torch_host_us"], r["torch_wall_us"] = th["host_us"], th["wall_us"]
            r["torch_gpu_us"] = timing.device_time(lambda: tfn(**ins))
            cf = capture(ck, ins)
            hw = timing.host_and_wall(lambda: cf(**ins))
            r["captured_copy_host_us"], r["captured_copy_wall_us"] = hw["host_us"], hw["wall_us"]
            cs = capture(ck, ins, outputs="static")
            sin = dict(zip(ins, cs.static_inputs()))
            hw = timing.host_and_wall(lambda: cs(**sin))
            r["captured_static_host_us"], r["captured_static_wall_us"] = hw["host_us"], hw["wall_us"]
            g = cs._entries[None].graph
            r["replay_host_us"] = timing.host_and_wall(g.replay)["host_us"]
            rows.append(r)
            del cf, cs, g
            torch.cuda.empty_cache()
    return rows


def _compile_ro(torch, fn):
    torch._dynamo.reset()
    return torch.compile(fn, mode="reduce-overhead")


def phase_chain(args):
    torch = _setup()
    from truestar.runtime import bitwise_equal, capture, timing
    from truestar.runtime.workloads import MLPStack
    rows = []
    sizes = [("small", 32, 256), ("medium", 1024, 1024)]
    for label, M, H in sizes:
        m = MLPStack(M, H, blocks=10)
        x = m.x
        m.step(x)
        torch.cuda.synchronize()
        n = 8  # 8 steps = 400 launches, below the launch queue depth
        reps = 9
        r = dict(case=f"50 kernel MLP {M}x{H} fp32", size=label, kernels=m.kernels_per_step)

        def put(prefix, fn):
            hw = timing.host_and_wall(fn, n=n, repeats=reps)
            r[prefix + "_host_us"], r[prefix + "_wall_us"] = hw["host_us"], hw["wall_us"]

        put("truestar", lambda: m.step(x))
        r["truestar_gpu_us"] = timing.device_time(lambda: m.step(x), k=4)
        cf = capture(m.step, (x,))
        assert bitwise_equal(cf(x), m.step(x))
        put("truestar_captured", lambda: cf(x))
        cs = capture(m.step, (x,), outputs="static")
        (sx,) = cs.static_inputs()
        put("truestar_captured_static", lambda: cs(sx))
        r["graph_bytes"] = cf.memory()[None]["allocated"]
        r["graph_reserved_bytes"] = cf.memory()[None]["reserved"]
        put("torch", lambda: m.torch_step(x))
        r["torch_gpu_us"] = timing.device_time(lambda: m.torch_step(x), k=4)
        ct = capture(m.torch_step, (x,))
        put("torch_captured", lambda: ct(x))
        t0 = time.perf_counter()
        cc = _compile_ro(torch, m.torch_step)
        for _ in range(3):
            cc(x)
        torch.cuda.synchronize()
        r["compile_ro_first_s"] = time.perf_counter() - t0
        put("compile_ro", lambda: cc(x))
        err = (cc(x).double() - m.torch_step(x).double()).abs().max().item()
        r["compile_ro_maxabs_vs_eager"] = err
        rows.append(r)
        del cf, cs, ct, cc
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    return rows


def phase_single(args):
    torch = _setup()
    from truestar.runtime import capture, timing
    rows = []
    for name, build in _cases(True):
        ck, tfn, ins = build()
        ck(**ins)
        r = dict(case=name)
        r["truestar_wall_us"] = timing.host_and_wall(lambda: ck(**ins))["wall_us"]
        cf = capture(ck, ins)
        r["truestar_captured_wall_us"] = timing.host_and_wall(lambda: cf(**ins))["wall_us"]
        r["torch_wall_us"] = timing.host_and_wall(lambda: tfn(**ins))["wall_us"]
        cc = _compile_ro(torch, tfn)
        for _ in range(3):
            cc(**ins)
        torch.cuda.synchronize()
        r["compile_ro_wall_us"] = timing.host_and_wall(lambda: cc(**ins))["wall_us"]
        rows.append(r)
        del cf, cc
    torch._dynamo.reset()
    return rows


PHASES = {"kernels": phase_kernels, "chain": phase_chain, "single": phase_single}


def _median_rows(runs):
    """Element wise median across runs of the same phase (rows line up by position)."""
    import statistics
    out = []
    for rows in zip(*runs):
        r = dict(rows[0])
        for k, v in rows[0].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                r[k] = statistics.median(row[k] for row in rows)
        out.append(r)
    return out


def _fmt(v):
    return f"{v:.1f}" if isinstance(v, float) else str(v)


def report(results):
    out = []
    if "kernels" in results:
        out.append("### Per launch host time vs GPU time (us)\n")
        out.append("| case | gather | signature | lookup | verify check | alloc | Triton launch | "
                   "C launcher | Truestar call host | GPU | host/GPU | captured static wall | "
                   "graph replay host |")
        out.append("|" + "---|" * 14)
        for r in results["kernels"]:
            out.append("| " + " | ".join(_fmt(x) for x in (
                r["case"], r["gather_us"], r["signature_us"], r["lookup_us"],
                r["verify_check_us"], r["alloc_us"], r["triton_launch_us"], r["c_launcher_us"],
                r["truestar_host_us"], r["gpu_us"], r["truestar_host_us"] / r["gpu_us"],
                r["captured_static_wall_us"], r["replay_host_us"])) + " |")
        out.append("\n| case | Truestar wall | captured copy wall | captured static wall | "
                   "PyTorch host | PyTorch wall | PyTorch GPU |")
        out.append("|" + "---|" * 7)
        for r in results["kernels"]:
            out.append("| " + " | ".join(_fmt(x) for x in (
                r["case"], r["truestar_wall_us"], r["captured_copy_wall_us"],
                r["captured_static_wall_us"], r["torch_host_us"], r["torch_wall_us"],
                r["torch_gpu_us"])) + " |")
    if "chain" in results:
        out.append("\n### 50 kernel step (us per step)\n")
        out.append("| case | impl | host | wall |")
        out.append("|---|---|---|---|")
        for r in results["chain"]:
            for impl, key in (("Truestar eager", "truestar"),
                              ("Truestar captured, copy out", "truestar_captured"),
                              ("Truestar captured, static", "truestar_captured_static"),
                              ("PyTorch eager", "torch"),
                              ("PyTorch eager captured (this layer)", "torch_captured"),
                              ("torch.compile reduce-overhead", "compile_ro")):
                out.append(f"| {r['case']} | {impl} | {_fmt(r[key + '_host_us'])} | "
                           f"{_fmt(r[key + '_wall_us'])} |")
            out.append(f"| {r['case']} | GPU time Truestar / PyTorch | | "
                       f"{_fmt(r['truestar_gpu_us'])} / {_fmt(r['torch_gpu_us'])} |")
            out.append(f"| {r['case']} | graph memory held / reserved MiB | | "
                       f"{r['graph_bytes'] / 2**20:.2f} / {r['graph_reserved_bytes'] / 2**20:.2f} |")
            out.append(f"| {r['case']} | torch.compile first calls s, max abs vs eager | | "
                       f"{r['compile_ro_first_s']:.1f}, {r['compile_ro_maxabs_vs_eager']:.1e} |")
    if "single" in results:
        out.append("\n### Single small kernels, wall us per call\n")
        out.append("| case | Truestar eager | Truestar captured | PyTorch eager | "
                   "torch.compile reduce-overhead |")
        out.append("|---|---|---|---|---|")
        for r in results["single"]:
            out.append("| " + " | ".join(_fmt(x) for x in (
                r["case"], r["truestar_wall_us"], r["truestar_captured_wall_us"],
                r["torch_wall_us"], r["compile_ro_wall_us"])) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES))
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--json", help="also write raw results to this file")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat every phase this many times (separate processes, one after "
                         "another) and report the median of each number across runs")
    args = ap.parse_args()
    if args.child:
        print("JSON" + json.dumps(PHASES[args.phase](args)))
        return
    env = dict(os.environ)
    env.setdefault("TRUESTAR_CACHE_DIR", os.path.join(tempfile.gettempdir(), "truestar_bench_rt"))
    runs = {}
    for k in range(args.runs):
        for ph in ([args.phase] if args.phase else list(PHASES)):
            t = time.perf_counter()
            p = subprocess.run([sys.executable, __file__, "--phase", ph, "--child"], env=env,
                               capture_output=True, text=True)
            line = [ln for ln in p.stdout.splitlines() if ln.startswith("JSON")]
            if p.returncode != 0 or not line:
                print(p.stdout[-3000:], p.stderr[-5000:])
                raise SystemExit(f"phase {ph} failed")
            runs.setdefault(ph, []).append(json.loads(line[-1][4:]))
            print(f"run {k + 1} phase {ph}: {time.perf_counter() - t:.0f} s", file=sys.stderr)
    results = {ph: _median_rows(rs) for ph, rs in runs.items()}
    import torch
    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    print(report(results))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"median": results, "runs": runs}, f, indent=1)


if __name__ == "__main__":
    main()
