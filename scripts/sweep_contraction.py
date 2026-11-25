"""Time every contraction tile of the tuner's grid, at every split K count, on this GPU.

    PYTHONPATH=src python scripts/sweep_contraction.py --json sweep.json
    PYTHONPATH=src python scripts/sweep_contraction.py --dtype float16 --json sweep_half.json
    PYTHONPATH=src python scripts/sweep_contraction.py --tf32 --shapes 1x4096x4096,512x1024x256

Every case is ``relu(x @ W.T + b)`` (as in scripts/bench_matmul.py) in ``--dtype`` (float32,
float16 or bfloat16; ``--tf32`` for float32 on TF32 tensor cores). For each shape the script
launches the kernel with each (BM, BN, BK, warps, stages) of ``CONTRACTION_GRID`` for the operand
class (and the half class's KTAIL off tile) at each split K count from 1 to 16 that
``check_config`` accepts, times it as a CUDA graph of 20 calls (median of ``--rounds`` rounds, one
graph alive at a time), and prints the default config's time next to the best one found.

Each shape runs in its own child process, so a crash of the CUDA context loses one shape, not the
sweep; a shape whose child dies is run again one config per child process, and the configs that
crash are recorded with the error. Results are written after every shape and an existing
``--json`` is resumed (shapes already in it are skipped). Use the output to set the defaults for a
GPU with a different SM count; the script changes nothing itself.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile

SHAPES = [(1, 4096, 4096), (4, 2048, 2048), (16, 4096, 1024), (32, 4096, 4096), (64, 2048, 1024),
          (64, 2048, 2048), (64, 1000, 2048), (256, 4096, 1024), (512, 1024, 256), (512, 256, 1024),
          (1024, 1024, 1024), (2048, 4096, 1024), (64, 512, 8192), (16, 1024, 16384),
          (128, 256, 8192), (8192, 768, 256), (8192, 1024, 256), (8192, 256, 1024)]


def configs_for(tb, p, env, cls):
    out = []
    grid = [tb._cfg(*g) for g in tb.CONTRACTION_GRID[cls]]
    if cls == "half":
        grid.append(tb._cfg(128, 128, 64, 4, 3, ktail=False))
    for g in grid:
        for split in (1, 2, 4, 8, 16):
            c = tb._fit_shared(p, dict(g, SPLIT_K=split))
            if c not in out and tb.check_config(p, env, c) is None:
                out.append(c)
    return out


def run_shape(shape, dtype, tf32, rounds, only=None):
    """Time the configs of one shape in this process; `only` limits it to config indices."""
    import torch
    import triton.testing
    from truestar import kernel as K
    from truestar.kernel import Graph
    from truestar.kernel import triton_backend as tb

    torch.backends.cuda.matmul.allow_tf32 = tf32
    M, N, Kd = shape
    dt = getattr(torch, dtype)
    g = Graph()
    x_ = g.input("x", (K.sym("M"), Kd), dtype)
    w_ = g.input("w", (N, Kd), dtype)
    b_ = g.input("b", (N,), dtype)
    (_, kern), = g.fuse({"y": g.relu(x_ @ g.transpose(w_, 0, 1) + b_)})
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(M, Kd, device="cuda", generator=gen).to(dt)
    w = (torch.randn(N, Kd, device="cuda", generator=gen) / Kd ** 0.5).to(dt)
    b = torch.randn(N, device="cuda", generator=gen).to(dt)
    base = K.compile(kern, "cuda", allow_tf32=tf32, tune=False)
    base(x=x, w=w, b=b)
    default = base.launch_configs()[0][1]
    p = base.art.plan
    env = {} if isinstance(p.m.size, int) else {p.m.size.name: M}
    cls = tb._contraction_class(p)
    configs = [default] + [c for c in configs_for(tb, p, env, cls) if c != default]
    if only is not None:
        configs = [configs[i] for i in only]

    def graph_of(fn):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            for _ in range(20):
                fn()
        return cg

    fns, failed = [], []
    for c in configs:
        try:
            ck = K.compile(kern, "cuda", allow_tf32=tf32, tune=False, _config=c)
            fn = (lambda ck=ck: ck(x=x, w=w, b=b))
            fn()
            torch.cuda.synchronize()
            fns.append((c, fn))
        except Exception as e:                                    # recorded, never hidden
            failed.append({"config": c, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    times = [[] for _ in fns]
    for _ in range(rounds):
        for i, (_, fn) in enumerate(fns):
            cg = graph_of(fn)
            times[i].append(triton.testing.do_bench(cg.replay, warmup=3, rep=20,
                                                    return_mode="median") / 20 * 1e3)
            del cg
        torch.cuda.empty_cache()
    res = [{"config": c, "us": statistics.median(t)} for (c, _), t in zip(fns, times)]
    return {"shape": [M, N, Kd], "default": default, "n_configs": len(configs), "results": res,
            "failed": failed}


def child(argv):
    out = subprocess.run([sys.executable, __file__] + argv, capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        tail = (out.stderr.strip().splitlines() or ["no output"])[-3:]
        return None, f"exit {out.returncode}: " + " | ".join(tail)[:500]
    return json.loads(line[7:]), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", help="comma separated MxNxK list (default: the built in sweep)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--json")
    ap.add_argument("--child-shape", help=argparse.SUPPRESS)
    ap.add_argument("--child-only", help=argparse.SUPPRESS)
    a = ap.parse_args()
    os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_sweep_"))
    os.environ["TRUESTAR_VERIFY"] = "first"
    if a.child_shape:
        shape = tuple(int(v) for v in a.child_shape.split("x"))
        only = json.loads(a.child_only) if a.child_only else None
        print("RESULT " + json.dumps(run_shape(shape, a.dtype, a.tf32, a.rounds, only)), flush=True)
        return

    import torch
    props = torch.cuda.get_device_properties(0)
    shapes = SHAPES if not a.shapes else [tuple(int(v) for v in s.split("x")) for s in a.shapes.split(",")]
    common = ["--dtype", a.dtype, "--rounds", str(a.rounds)] + (["--tf32"] if a.tf32 else [])
    doc = {"gpu": props.name, "sms": props.multi_processor_count, "dtype": a.dtype, "tf32": a.tf32,
           "rows": []}
    if a.json and os.path.exists(a.json):
        with open(a.json, encoding="utf-8") as f:
            doc = json.load(f)
    done = {tuple(r["shape"]) for r in doc["rows"]}
    for shape in shapes:
        if shape in done:
            continue
        tag = "x".join(map(str, shape))
        row, err = child(["--child-shape", tag] + common)
        if row is None:          # the context died: one config per child, to find the culprit
            print(f"{tag}: shape child failed ({err}); rerunning config by config", flush=True)
            probe, err2 = child(["--child-shape", tag, "--child-only", "[0]"] + common)
            n = probe["n_configs"] if probe else 0
            row = {"shape": list(shape), "default": probe["default"] if probe else None,
                   "n_configs": n, "results": [], "failed": [], "shape_child_error": err}
            for i in range(n):
                r, e = child(["--child-shape", tag, "--child-only", json.dumps([i])] + common)
                if r is None:
                    row["failed"].append({"config_index": i, "error": e})
                else:
                    row["results"] += r["results"]
                    row["failed"] += r["failed"]
        row["results"].sort(key=lambda r: r["us"])
        doc["rows"].append(row)
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=1)
        dflt = next((r["us"] for r in row["results"] if r["config"] == row["default"]), None)
        if row["results"] and dflt is not None:
            bc = row["results"][0]["config"]
            print(f"{tag}: default {dflt:.1f} us, best {row['results'][0]['us']:.1f} us "
                  f"({bc['BM']}x{bc['BN']}x{bc['BK']} w{bc['num_warps']} s{bc['num_stages']} "
                  f"split {bc['SPLIT_K']}{'' if bc['KTAIL'] else ' no ktail'}), default / best "
                  f"{dflt / row['results'][0]['us']:.2f}, {len(row['failed'])} failed", flush=True)
        else:
            print(f"{tag}: no timing for the default, {len(row['failed'])} failed", flush=True)


if __name__ == "__main__":
    main()
