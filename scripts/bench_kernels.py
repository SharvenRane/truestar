"""Benchmark Truestar kernels against PyTorch eager, torch.compile, SDPA and FlexAttention.

    PYTHONPATH=src python scripts/bench_kernels.py            # full run, about 10 minutes
    PYTHONPATH=src python scripts/bench_kernels.py --quick    # smaller sizes, one extra size

Cases:
  (a) elementwise chain (tanh GELU of x plus a residual, eight ops), float32
  (b) causal attention, float16, at S 2048, 4096, 8192 and head dims 64 and 128 (batch times
      heads shrinks as S grows: B2 H16 at 2048, B1 H8 at 4096, B1 H2 at 8192)
  (c) causal differential attention (Ye et al. 2024), float16: two softmaxes with a learned
      per head lambda. FlexAttention's score_mod and mask_mod act on one score at a time, so it
      needs two calls plus a combine; Truestar fuses the whole thing into one kernel.

Truestar runs twice per case: with tuning off (the fixed default launch config) and with the
default auto tuning, which searches launch configs once per kernel, shape bucket, dtype and device
and stores the winner in the on disk cache. Compile times are cold: every phase of the run shares
fresh, empty cache directories (Truestar, Triton and Inductor), and each case's kernels are new to
them. A last process reuses those directories to measure a warm start.

Each phase is its own process and only one GPU process runs at a time. Run times are the median
of five rounds; each round spins every implementation for 0.2 s (the GPU drops its clocks when
idle) and then takes the median of triton.testing.do_bench (L2 flushed before every call). The
rounds interleave the implementations, so slow drift in clocks or a neighbour on a shared GPU
hits all of them alike; the spread column is (max - min) / median over the five rounds. Every
implementation's error is measured against a float64 computation of the same math. Truestar's
own float64 verification runs on the first call of every tuned or default config; its time is
reported and excluded from compile and run times.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

ROUNDS = 5


def _common():
    import torch
    import triton.testing

    def spin(fn, s=0.2):
        t = time.perf_counter()
        while time.perf_counter() - t < s:
            fn()
        torch.cuda.synchronize()

    def rounds(fns):
        """{name: [ms] * ROUNDS}, interleaving the implementations round by round."""
        res = {n: [] for n in fns}
        for _ in range(ROUNDS):
            for n, fn in fns.items():
                spin(fn)
                res[n].append(float(triton.testing.do_bench(fn, warmup=25, rep=100,
                                                            return_mode="median")))
        return res

    def timed_first(fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        return out, time.perf_counter() - t

    def err(out, ref):
        o = out.double()
        return ((o - ref).abs().max() / ref.abs().max()).item()

    return spin, rounds, timed_first, err


class Case:
    """Rows of one case; run functions are collected and timed together at the end."""

    def __init__(self, name):
        self.name = name
        self.rows = []
        self.fns = {}

    def add(self, impl, fn, kernels, compile_s, e, note="", **extra):
        self.rows.append(dict(case=self.name, impl=impl, kernels=kernels, compile_s=compile_s,
                              err=e, note=note, **extra))
        self.fns[impl] = fn

    def finish(self, rounds):
        times = rounds(self.fns)
        for r in self.rows:
            ts = sorted(times[r["impl"]])
            med = ts[len(ts) // 2]
            r["run_ms"] = med
            r["runs_ms"] = times[r["impl"]]
            r["spread"] = (ts[-1] - ts[0]) / med
        return self.rows


def _truestar(case, build, inputs, ref, timed_first, err, edit=None, tuned=True):
    """Compile, verify and register one Truestar row (tuning off or auto)."""
    from truestar.kernel import tune
    os.environ["TRUESTAR_VERIFY"] = "0"
    g, outs = build()
    tune_before = tune.stats.seconds
    t = time.perf_counter()
    fn = g.compile(outs, "cuda", tune=None if tuned else False)
    timed_first(lambda: fn(**inputs))
    compile_s = time.perf_counter() - t
    tune_s = tune.stats.seconds - tune_before
    os.environ["TRUESTAR_VERIFY"] = "first"
    t = time.perf_counter()
    out = list(fn(**inputs).values())[0]
    import torch
    torch.cuda.synchronize()
    verify_s = time.perf_counter() - t
    cfgs = [c for _, ck in fn.kernels for _, c, _ in ck.launch_configs()]
    srcs = [s for _, ck in fn.kernels for _, _, s in ck.launch_configs()]
    cfg_txt = "; ".join(" ".join(f"{k}={v}" for k, v in c.items()) for c in cfgs)
    impl = "truestar tuned" if tuned else "truestar tuning off"
    note = f"config {cfg_txt} ({', '.join(srcs)})"
    extra = dict(tune_s=tune_s, verify_s=verify_s, compile_wo_tune_s=compile_s - tune_s)
    if edit is not None:
        g2, outs2 = edit()
        os.environ["TRUESTAR_VERIFY"] = "0"
        t = time.perf_counter()
        fn2 = g2.compile(outs2, "cuda", tune=None if tuned else False)
        timed_first(lambda: fn2(**inputs))
        extra["edit_s"] = time.perf_counter() - t
        os.environ["TRUESTAR_VERIFY"] = "first"
    case.add(impl, lambda: fn(**inputs), fn.num_kernels, compile_s, err(out, ref), note, **extra)
    return out


def _torch_compiled(case, f, inputs, ref, timed_first, err, impl, kernels="?", edit=None):
    import torch
    torch._dynamo.reset()
    cf = torch.compile(f)
    out, first = timed_first(lambda: cf(**inputs))
    extra = {}
    if edit is not None:
        cf2 = torch.compile(edit)
        _, extra["edit_s"] = timed_first(lambda: cf2(**inputs))
    case.add(impl, lambda: cf(**inputs), kernels, first, err(out, ref), **extra)


def _eager(case, f, inputs, ref, timed_first, err, impl, kernels="?"):
    out, _ = timed_first(lambda: f(**inputs))
    case.add(impl, lambda: f(**inputs), kernels, 0.0, err(out, ref))


def _ref_attn(q, k, v, scale):
    """float64 causal attention, one head at a time so S 8192 fits in memory."""
    import torch
    B, H, S, _ = q.shape
    out = torch.empty(B, H, S, v.shape[-1], dtype=torch.float64, device=q.device)
    mask = torch.ones(S, S, dtype=torch.bool, device=q.device).tril()
    for b in range(B):
        for h in range(H):
            s = (q[b, h].double() @ k[b, h].double().T) * scale
            out[b, h] = torch.softmax(s.masked_fill(~mask, float("-inf")), -1) @ v[b, h].double()
    return out


def _attn_builder(B, H, D, sc):
    from truestar import kernel as K
    from truestar.kernel import Graph

    def build(scale=sc):
        g = Graph()
        Sy = K.sym("S")
        gq, gk, gv = (g.input(n, (B, H, Sy, D), "float16") for n in "qkv")
        s = (gq @ gk.T) * scale
        s = g.pointwise(lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF), s, with_index=True)
        return g, {"o": g.softmax(s) @ gv}
    return build


def _attention_case(B, H, S, D, spin, rounds, timed_first, err, full):
    import torch
    import torch.nn.functional as F
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    torch.manual_seed(0)
    dev = "cuda"
    scale = D ** -0.5
    q, k, v = (torch.randn(B, H, S, D, device=dev, dtype=torch.float16) for _ in range(3))
    ins = {"q": q, "k": k, "v": v}
    ref = _ref_attn(q, k, v, scale)
    case = Case(f"(b) causal attention B{B} H{H} S{S} D{D} fp16")
    build = _attn_builder(B, H, D, scale)
    edit = (lambda: build(scale * 0.5)) if full else None
    _truestar(case, build, ins, ref, timed_first, err, tuned=False)
    _truestar(case, build, ins, ref, timed_first, err, edit=edit, tuned=True)
    if full:
        mask = torch.ones(S, S, dtype=torch.bool, device=dev).tril()

        def attn_math(q, k, v, scale=scale):
            s = (q @ k.transpose(-1, -2)) * scale
            s = s.masked_fill(~mask, float("-inf"))
            return torch.softmax(s.float(), -1).to(q.dtype) @ v

        _eager(case, attn_math, ins, ref, timed_first, err, "eager math", kernels="5")
        _torch_compiled(case, attn_math, ins, ref, timed_first, err, "torch.compile math",
                        edit=lambda q, k, v: attn_math(q, k, v, scale * 0.5))
    _eager(case, lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True), ins,
           ref, timed_first, err, "SDPA (hand written flash)", kernels=1)
    bm = create_block_mask(lambda b, h, i, j: j <= i, None, None, S, S, device=dev)

    def flex(q, k, v):
        return flex_attention(q, k, v, block_mask=bm, scale=scale)

    _torch_compiled(case, flex, ins, ref, timed_first, err, "torch.compile FlexAttention",
                    kernels=1)
    return case.finish(rounds)


def _phase_cold(args):
    import torch
    from truestar import kernel as K
    from truestar.kernel import Graph
    import torch.nn.functional as F
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    spin, rounds, timed_first, err = _common()
    torch.manual_seed(0)
    dev = "cuda"
    rows = []

    # ------------------------------------------------------------------ (a) elementwise chain
    M, N = (2048, 2048) if args.quick else (8192, 4096)
    x = torch.randn(M, N, device=dev)
    r = torch.randn(M, N, device=dev)
    ins = {"x": x, "r": r}

    def gelu_res(x, r, c=0.044715):
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + c * x * x * x))) + r

    ref = gelu_res(x.double(), r.double())

    def build_a(c=0.044715):
        g = Graph()
        gx = g.input("x", (K.sym("M"), K.sym("N")))
        gr = g.input("r", (K.sym("M"), K.sym("N")))
        y = 0.5 * gx * (1.0 + g.tanh(0.7978845608 * (gx + c * gx * gx * gx))) + gr
        return g, {"y": y}

    case = Case(f"(a) gelu+residual {M}x{N} fp32")
    _truestar(case, build_a, ins, ref, timed_first, err, tuned=False)
    _truestar(case, build_a, ins, ref, timed_first, err, edit=lambda: build_a(0.05))
    _eager(case, gelu_res, ins, ref, timed_first, err, "eager", kernels=9)
    _torch_compiled(case, gelu_res, ins, ref, timed_first, err, "torch.compile", kernels=1,
                    edit=lambda x, r: gelu_res(x, r, 0.05))
    rows += case.finish(rounds)
    del x, r, ins, ref

    # ------------------------------------------------------------------ (b) causal attention
    B, H, S, D = (1, 8, 1024, 64) if args.quick else (2, 16, 2048, 64)
    rows += _attention_case(B, H, S, D, spin, rounds, timed_first, err, full=True)

    # ------------------------------------------------------------------ (c) differential
    Dh = D // 2  # each component's query/key dim; values keep the full head dim (2 * Dh)
    dt = torch.float16
    q1, k1, q2, k2 = (torch.randn(B, H, S, Dh, device=dev, dtype=dt) for _ in range(4))
    v = torch.randn(B, H, S, 2 * Dh, device=dev, dtype=dt)
    lam = (torch.rand(1, H, 1, 1, device=dev) * 0.8).float()
    sc = Dh ** -0.5
    mask = torch.ones(S, S, dtype=torch.bool, device=dev).tril()
    ins = {"q1": q1, "k1": k1, "q2": q2, "k2": k2, "v": v, "lam": lam}

    def diff_math(q1, k1, q2, k2, v, lam, sc=sc):
        def p(q, k):
            s = ((q @ k.transpose(-1, -2)) * sc).masked_fill(~mask, float("-inf"))
            return torch.softmax(s.float(), -1)
        return ((p(q1, k1) - lam * p(q2, k2)).to(v.dtype)) @ v

    def diff_sdpa(q1, k1, q2, k2, v, lam):
        a1 = F.scaled_dot_product_attention(q1, k1, v, is_causal=True, scale=sc)
        a2 = F.scaled_dot_product_attention(q2, k2, v, is_causal=True, scale=sc)
        return a1 - lam.to(v.dtype) * a2

    def diff_ref(q1, k1, q2, k2, v, lam):
        def p(q, k):
            s = ((q.double() @ k.double().transpose(-1, -2)) * sc).masked_fill(~mask,
                                                                               float("-inf"))
            return torch.softmax(s, -1)
        return (p(q1, k1) - lam.double() * p(q2, k2)) @ v.double()

    ref = diff_ref(**ins)

    def build_c(s_=sc):
        g = Graph()
        Sy = K.sym("S")
        gq1, gk1, gq2, gk2 = (g.input(n, (B, H, Sy, Dh), "float16")
                              for n in ("q1", "k1", "q2", "k2"))
        gv = g.input("v", (B, H, Sy, 2 * Dh), "float16")
        glam = g.input("lam", (1, H, 1, 1), "float32")
        causal = lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF)  # noqa: E731
        p1 = g.softmax(g.pointwise(causal, (gq1 @ gk1.T) * s_, with_index=True))
        p2 = g.softmax(g.pointwise(causal, (gq2 @ gk2.T) * s_, with_index=True))
        return g, {"o": (p1 - glam * p2) @ gv}

    case = Case(f"(c) differential attention B{B} H{H} S{S} D{Dh}x2 fp16")
    _truestar(case, build_c, ins, ref, timed_first, err, tuned=False)
    _truestar(case, build_c, ins, ref, timed_first, err, edit=lambda: build_c(sc * 0.5))
    _eager(case, diff_math, ins, ref, timed_first, err, "eager math")
    _eager(case, diff_sdpa, ins, ref, timed_first, err, "eager 2x SDPA + combine", kernels=3)
    _torch_compiled(case, diff_math, ins, ref, timed_first, err, "torch.compile math",
                    edit=lambda **kw: diff_math(**kw, sc=sc * 0.5))
    bm = create_block_mask(lambda b, h, i, j: j <= i, None, None, S, S, device=dev)

    def flex2(q1, k1, q2, k2, v, lam):
        a1 = flex_attention(q1, k1, v, block_mask=bm, scale=sc)
        a2 = flex_attention(q2, k2, v, block_mask=bm, scale=sc)
        return a1 - lam.to(v.dtype) * a2

    _torch_compiled(case, flex2, ins, ref, timed_first, err,
                    "torch.compile 2x FlexAttention + combine")
    rows += case.finish(rounds)
    return rows


def _phase_attn(args):
    spin, rounds, timed_first, err = _common()
    return _attention_case(args.B, args.H, args.S, args.D, spin, rounds, timed_first, err,
                           full=False)


def _phase_warm(args):
    """Same cache dirs: how long a new process takes to compile and launch the same kernels."""
    import torch
    from truestar.kernel import tune
    _, _, timed_first, _ = _common()
    os.environ["TRUESTAR_VERIFY"] = "0"
    res = {}
    B, H, S, D = (1, 8, 1024, 64) if args.quick else (2, 16, 2048, 64)
    q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.float16) for _ in range(3))
    mask = torch.ones(S, S, dtype=torch.bool, device="cuda").tril()
    build = _attn_builder(B, H, D, D ** -0.5)
    t = time.perf_counter()
    g, outs = build()
    fn = g.compile(outs, "cuda")
    timed_first(lambda: fn(q=q, k=k, v=v))
    res["truestar attention warm start s"] = time.perf_counter() - t
    res["truestar attention warm start: tuning searches"] = tune.stats.searches
    res["truestar attention warm start: configs read from disk"] = tune.stats.cache_hits

    def attn_math(q, k, v):
        s = (q @ k.transpose(-1, -2)) * D ** -0.5
        return torch.softmax(s.masked_fill(~mask, float("-inf")).float(), -1).to(q.dtype) @ v

    t = time.perf_counter()
    timed_first(lambda: torch.compile(attn_math)(q, k, v))
    res["torch.compile attention math warm start s"] = time.perf_counter() - t
    return res


def _fmt(x, nd=2):
    return f"{x:.{nd}f}" if x else "0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--child", choices=["cold", "attn", "warm"])
    ap.add_argument("--S", type=int, default=2048)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--json", help="also write all rows to this file")
    args = ap.parse_args()
    if args.child:
        out = {"cold": _phase_cold, "attn": _phase_attn, "warm": _phase_warm}[args.child](args)
        print("JSON" + json.dumps(out))
        return
    cache = tempfile.mkdtemp(prefix="truestar_bench_")
    env = dict(os.environ)
    env["TRUESTAR_CACHE_DIR"] = os.path.join(cache, "truestar")
    env["TRITON_CACHE_DIR"] = os.path.join(cache, "triton")
    env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache, "inductor")
    # B * H * S^2 is held at the (b) size so the float64 checks (CPU, quadratic in S) stay near
    # a minute per phase
    sizes = [(1, 8, 2048, 128)] if args.quick else [
        (2, 16, 2048, 128), (1, 8, 4096, 64), (1, 8, 4096, 128), (1, 2, 8192, 64),
        (1, 2, 8192, 128)]
    phases = [("cold", [])] + [("attn", ["--B", str(b), "--H", str(h), "--S", str(s), "--D",
                                         str(d)]) for b, h, s, d in sizes]
    phases.append(("warm", []))
    rows, warm = [], {}
    t_all = time.perf_counter()
    for phase, extra in phases:  # sequential: one GPU process at a time
        cmd = [sys.executable, __file__, "--child", phase] + extra
        cmd += ["--quick"] if args.quick else []
        t = time.perf_counter()
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("JSON")]
        if p.returncode != 0 or not line:
            print(p.stdout[-3000:], p.stderr[-5000:])
            raise SystemExit(f"{phase} {extra} phase failed")
        print(f"phase {phase} {' '.join(extra)}: {time.perf_counter() - t:.0f} s", file=sys.stderr)
        res = json.loads(line[-1][4:])
        if phase == "warm":
            warm = res
        else:
            rows += res
    import torch
    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}  "
          f"(total {time.perf_counter() - t_all:.0f} s)")
    print()
    print("| case | implementation | kernels | compile s | tune s | verify s | edit recompile s "
          "| run ms (median of 5) | spread | max rel err vs f64 | notes |")
    print("|" + "---|" * 11)
    for r in rows:
        print(f"| {r['case']} | {r['impl']} | {r['kernels']} | {_fmt(r['compile_s'])} | "
              f"{_fmt(r.get('tune_s', 0))} | {_fmt(r.get('verify_s', 0), 1)} | "
              f"{_fmt(r.get('edit_s', 0))} | {r['run_ms']:.3f} | {100 * r['spread']:.1f}% | "
              f"{r['err']:.2e} | {r['note']} |")
    print()
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"], {})[r["impl"]] = r
    print("| case | truestar tuned ms | best other | its ms | truestar speedup |")
    print("|---|---|---|---|---|")
    for c, impls in by_case.items():
        ours = impls["truestar tuned"]["run_ms"]
        others = {k: v["run_ms"] for k, v in impls.items() if not k.startswith("truestar")}
        best = min(others, key=others.get)
        print(f"| {c} | {ours:.3f} | {best} | {others[best]:.3f} | {others[best] / ours:.2f}x |")
    print()
    for k_, v_ in warm.items():
        print(f"{k_}: {v_:.2f}" if isinstance(v_, float) else f"{k_}: {v_}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": rows, "warm": warm}, f, indent=1)


if __name__ == "__main__":
    main()
