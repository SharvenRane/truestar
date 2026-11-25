"""P17 measurements: cold registration, first call, warm calls and torch.compile for three native
custom ops. Run each phase in a fresh process:

    PYTHONPATH=src python scripts/customop_bench.py            # prints one JSON object

Each phase uses a fresh, empty kernel cache directory, so first call numbers include code generation
and the Triton or numba compile. Warm numbers are medians over repeated calls.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap

CHILD = textwrap.dedent('''
    import json, os, shutil, statistics, sys, time, warnings
    import torch
    import torch.nn.functional as F
    dev = sys.argv[1]
    # model widths per device: the numba CPU backend is a correctness backend (plain loop nests),
    # so the CPU run uses smaller widths
    D, HID, N = (512, 2048, 4096) if dev == "cuda" else (128, 512, 1024)
    def log(m):
        print(m, file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    import truestar
    from truestar import Tensor, ops
    t1 = time.perf_counter()

    @truestar.torch_op
    def gelu_mlp(x: Tensor["B", D], w1: Tensor[HID, D], b1: Tensor[HID], w2: Tensor[D, HID]) -> Tensor["B", D]:
        h = ops.gelu(ops.linear(x, w1, b1), approximate="tanh")
        return h @ w2.T + x
    t2 = time.perf_counter()

    @truestar.torch_op
    def softcap_attention(q: Tensor["B", "H", "S", 64], k: Tensor["B", "H", "S", 64],
                          v: Tensor["B", "H", "S", 64]) -> Tensor["B", "H", "S", 64]:
        s = (q @ k.transpose(-1, -2)) * 0.125
        s = 30.0 * ops.tanh(s / 30.0)
        return ops.softmax(s, dim=-1) @ v
    t3 = time.perf_counter()

    @truestar.torch_op
    def rms_norm_residual(x: Tensor["B", N], r: Tensor["B", N], w: Tensor[N]) -> Tensor["B", N]:
        h = x + r
        return h * ops.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6) * w
    t4 = time.perf_counter()

    def t_mlp(x, w1, b1, w2): return F.gelu(F.linear(x, w1, b1), approximate="tanh") @ w2.T + x
    def t_attn(q, k, v):
        s = (q @ k.transpose(-1, -2)) * 0.125
        return torch.softmax(30.0 * torch.tanh(s / 30.0), -1) @ v
    def t_rms(x, r, w):
        h = x + r
        return h * torch.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6) * w

    sizes = {"cpu": {"mlp": 64, "attn": (1, 2, 64), "rms": 64}, "cuda": {"mlp": 4096, "attn": (2, 16, 1024), "rms": 8192}}[dev]
    B, H, S = sizes["attn"]
    cases = {
        "gelu_mlp": (gelu_mlp, t_mlp, [(sizes["mlp"], D), (HID, D), (HID,), (D, HID)]),
        "softcap_attention": (softcap_attention, t_attn, [(B, H, S, 64)] * 3),
        "rms_norm_residual": (rms_norm_residual, t_rms, [(sizes["rms"], N), (sizes["rms"], N), (N,)]),
    }

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    def timeit(fn, n):
        ts = []
        for _ in range(n):
            sync(); t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
        return statistics.median(ts) * 1e3

    res = {"device": dev, "torch": torch.__version__, "import_truestar_s": t1 - t0,
           "register_s": {"gelu_mlp": t2 - t1, "softcap_attention": t3 - t2, "rms_norm_residual": t4 - t3},
           "compilers_on_path": [c for c in ("cl", "nvcc", "gcc", "g++", "clang") if shutil.which(c)], "ops": {}}
    for name, (op, ref, shapes) in cases.items():
        log(f"{dev} {name}")
        xs = [(torch.randn(*s, device=dev) * (0.05 if name == "gelu_mlp" and k > 0 else 1.0)).requires_grad_()
              for k, s in enumerate(shapes)]
        sync(); t = time.perf_counter(); y = op(*xs); sync(); first = time.perf_counter() - t
        log(f"  forward first call {first:.2f} s")
        sync(); t = time.perf_counter(); y.sum().backward(); sync(); first_bwd = time.perf_counter() - t
        log(f"  backward first call {first_bwd:.2f} s")
        with torch.no_grad():
            warm = timeit(lambda: op(*xs), 20)
            eager = timeit(lambda: ref(*xs), 20)
            x64 = [x.detach().double() for x in xs]
            err_ts = float((op(*xs).double() - ref(*x64)).abs().max() / ref(*x64).abs().max())
            err_torch = float((ref(*xs).double() - ref(*x64)).abs().max() / ref(*x64).abs().max())
        torch._dynamo.reset()
        ex = torch._dynamo.explain(lambda *a: op(*a) * 2.0)(*[x.detach() for x in xs])
        res["ops"][name] = {"first_call_s": first, "first_backward_s": first_bwd, "warm_ms": warm,
                            "torch_eager_ms": eager, "rel_err_vs_f64": err_ts, "torch_f32_rel_err_vs_f64": err_torch,
                            "graph_breaks": ex.graph_break_count, "graphs": ex.graph_count,
                            "forward_path": op.plan(dev).path, "backward_path": op.plan(dev, True).path,
                            "kernels": [op.plan(dev).kernels, op.plan(dev, True).kernels]}
    # inductor with the op inside, forward and backward
    op, ref, shapes = cases["rms_norm_residual"]
    xs = [torch.randn(*s, device=dev).requires_grad_() for s in shapes]
    torch._dynamo.reset()
    try:
        cm = torch.compile(lambda *a: op(*a).relu().sum(), backend="inductor", fullgraph=True)
        cm(*xs).backward()
        res["inductor_fullgraph_backward"] = "ok"
    except Exception as e:
        res["inductor_fullgraph_backward"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    print(json.dumps(res))
''')


def main():
    out = {}
    import torch
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        cache = tempfile.mkdtemp(prefix="truestar_cold_")
        env = dict(os.environ, TRUESTAR_CACHE_DIR=cache, NUMBA_CACHE_DIR=os.path.join(cache, "numba"),
                   TRITON_CACHE_DIR=os.path.join(cache, "triton"))
        with tempfile.TemporaryDirectory() as d:
            script = os.path.join(d, "child.py")
            with open(script, "w", encoding="utf-8") as f:
                f.write(CHILD)
            p = subprocess.run([sys.executable, script, dev], env=env, stdout=subprocess.PIPE, text=True, timeout=1800)
        shutil.rmtree(cache, ignore_errors=True)
        if p.returncode != 0:
            out[dev] = {"error": f"exit {p.returncode}"}
        else:
            out[dev] = json.loads(p.stdout.strip().splitlines()[-1])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
