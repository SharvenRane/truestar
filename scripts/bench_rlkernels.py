"""Speed cost of the RL kernel set (truestar.rlkernels) against eager PyTorch and Truestar's fast
compiled kernels.

    PYTHONPATH=src python scripts/bench_rlkernels.py              # ops and the model, both dtypes
    PYTHONPATH=src python scripts/bench_rlkernels.py --rounds 5 --json out.json

Three columns per case, all on the same GPU in one process:

* eager: PyTorch's own kernels (cuBLAS, SDPA flash or memory efficient attention, ATen norms and
  softmax). Fast, not batch invariant.
* fast: Truestar's compiled executor (`compile_program`, Triton kernels with tuned per shape
  configs) on a one op IR program; only for the ops it has (linear, causal attention, softmax).
* rl: the RL kernel set, the same call the trainer and the sampler make under `rl_mode()`.

The model cases run the Llama style decoder of `truestar.rlkernels.llama` at a larger size
(dim 1024, 4 layers, 16 heads on 4 KV heads, SwiGLU 2816, vocab 32000; random weights): the
training forward over B x T tokens and one decode step against a cache of T keys, eager against
`rl_mode()`.

Timing: CUDA events around 20 calls after warm up, per round; the GPU may be shared, so each case
keeps the fastest of `--rounds` rounds, interleaved across columns (other jobs only add time).
"""

import argparse
import json
import os
import sys
import tempfile
import time

import torch
import torch.nn.functional as F

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_rlbench_"))

from truestar.rlkernels import kernels as K  # noqa: E402
from truestar.rlkernels import llama as L  # noqa: E402
from truestar.rlkernels import rl_mode  # noqa: E402


def timer(fn, n=20):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1000.0  # microseconds


def fast_kernel(kind, dtype, shapes):
    """A one op IR program compiled by Truestar's compiler, or None."""
    from truestar.compiler import compile_program
    from truestar.ir.core import Builder

    name = {torch.float32: "float32", torch.bfloat16: "bfloat16"}[dtype]
    b = Builder(f"bench_{kind}")
    if kind == "linear":
        (M, Kd), (N, _) = shapes
        x = b.input("x", torch.float32, ("M", Kd))
        w = b.param("w", torch.randn(N, Kd) * 0.02)
        prog = b.build([b.op("linear", x, w, None)])
    elif kind == "attention":
        q = b.input("q", torch.float32, ("B", shapes[0][1], "S", shapes[0][3]))
        k = b.input("k", torch.float32, ("B", shapes[0][1], "S", shapes[0][3]))
        v = b.input("v", torch.float32, ("B", shapes[0][1], "S", shapes[0][3]))
        prog = b.build([b.op("attention", q, k, v, None, scale=None, is_causal=True)])
    elif kind == "softmax":
        x = b.input("x", torch.float32, ("R", shapes[0][1]))
        prog = b.build([b.op("softmax", x, dim=-1)])
    else:
        return None
    # Speed only: Truestar checks these kernels against float64 in its own suite; the float64 check
    # of a 4096 x 2048 or B8 S512 call would hold over a GiB on the shared GPU.
    cp = compile_program(prog, "cuda", name, verify="off")
    return cp


def op_cases(dtype):
    g = torch.Generator(device="cuda").manual_seed(0)

    def r(*s):
        return torch.randn(*s, device="cuda", generator=g).to(dtype)

    cases = []
    for M in (8, 512, 4096):
        x, w = r(M, 2048), r(2048, 2048) * 0.02
        cases.append((f"linear {M}x2048x2048", lambda x=x, w=w: F.linear(x, w),
                      lambda x=x, w=w: K.linear(x, w), ("linear", [(M, 2048), (2048, 2048)], [x]), w))
    q, k, v = r(4, 16, 512, 64), r(4, 16, 512, 64), r(4, 16, 512, 64)
    cases.append(("attention prefill B4 H16 S512 D64 causal",
                  lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
                  lambda: K.attention(q, k, v, is_causal=True),
                  ("attention", [tuple(q.shape)], [q, k, v]), None))
    qd = r(4, 16, 1, 64)
    mask = torch.ones(4, 1, 1, 512, dtype=torch.bool, device="cuda")
    cases.append(("attention decode B4 H16 cache 512 D64",
                  lambda: F.scaled_dot_product_attention(qd, k, v, attn_mask=mask),
                  lambda: K.attention(qd, k, v, attn_mask=mask), None, None))
    xn, wn = r(4096, 2048), r(2048)
    cases.append(("rms_norm 4096x2048", lambda: F.rms_norm(xn, (2048,), wn, 1e-5),
                  lambda: K.rms_norm(xn, (2048,), wn, 1e-5), None, None))
    xs = r(512, 32000)
    cases.append(("log_softmax 512x32000", lambda: torch.log_softmax(xs, -1),
                  lambda: K.log_softmax(xs, -1), None, None))
    cases.append(("softmax 512x32000", lambda: torch.softmax(xs, -1), lambda: K.softmax(xs, -1),
                  ("softmax", [tuple(xs.shape)], [xs]), None))
    return cases


def model_cases(dtype):
    cfg = L.Config(vocab=32000, dim=1024, layers=4, heads=16, kv_heads=4, hidden=2816, max_len=2048)
    m = L.make_model(device="cuda", dtype=dtype, config=cfg)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.mul_(0.02 / 0.06)
    B, T = 4, 512
    ids = torch.randint(0, cfg.vocab, (B, T), device="cuda")
    valid = torch.ones(B, T + 1, dtype=torch.bool, device="cuda")
    pos = torch.full((B, 1), T, device="cuda")
    tok = torch.randint(0, cfg.vocab, (B, 1), device="cuda")

    def train():
        with torch.no_grad():
            return m(ids)

    def decode_eager():
        c = cache_eager
        c.seen = T
        with torch.no_grad():
            return m(tok, pos, valid, c)

    cache_eager = L.KVCache(cfg.layers, "static", capacity=T + 1)
    with torch.no_grad():
        m(ids, torch.arange(T, device="cuda").expand(B, T), valid[:, :T], cache_eager)

    def rl(fn):
        def run():
            with rl_mode():
                return fn()
        return run

    return [("model training forward B4 T512", train, rl(train)),
            ("model decode step B4 cache 512", decode_eager, rl(decode_eager))], m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--json")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    for dname in a.dtypes.split(","):
        dtype = getattr(torch, dname)
        cases = op_cases(dtype)
        fast = {}
        for name, _, _, spec, _ in cases:
            if spec is None:
                continue
            try:
                cp = fast_kernel(spec[0], dtype, spec[1])
                ins = spec[2]
                cp(ins)  # compile, tune, verify once
                fast[name] = (lambda cp=cp, ins=ins: cp(ins))
            except Exception as exc:  # report, do not hide
                print(f"  fast kernel for {name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        best = {}
        for _ in range(a.rounds):
            for name, eager, rlf, _, _ in cases:
                for col, fn in (("eager", eager), ("rl", rlf), ("fast", fast.get(name))):
                    if fn is None:
                        continue
                    t = timer(fn)
                    best[(name, col)] = min(best.get((name, col), float("inf")), t)
        mcases, model = model_cases(dtype)
        for _ in range(a.rounds):
            for name, eager, rlf in mcases:
                for col, fn in (("eager", eager), ("rl", rlf)):
                    t = timer(fn, n=5)
                    best[(name, col)] = min(best.get((name, col), float("inf")), t)
        del model
        torch.cuda.empty_cache()
        names = [c[0] for c in cases] + [c[0] for c in mcases]
        print(f"\n{dname}: microseconds per call (fastest of {a.rounds} rounds), rl / eager, rl / fast")
        print(f"{'case':<44} {'eager':>10} {'fast':>10} {'rl':>10} {'rl/eager':>9} {'rl/fast':>8}")
        for n in names:
            e, f, r = best.get((n, "eager")), best.get((n, "fast")), best.get((n, "rl"))
            row = {"dtype": dname, "case": n, "eager_us": e, "fast_us": f, "rl_us": r,
                   "rl_over_eager": r / e if e else None, "rl_over_fast": r / f if f else None}
            rows.append(row)
            print(f"{n:<44} {e:>10.1f} {(f'{f:.1f}' if f else '-'):>10} {r:>10.1f} {r / e:>9.2f} "
                  f"{(f'{r / f:.2f}' if f else '-'):>8}")
    print(f"\npeak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
