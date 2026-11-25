"""Strict float32 contraction: weight layout and deterministic K chunks against cuBLAS.

    PYTHONPATH=src python scripts/bench_perfmm.py [--shapes MxNxK,...] [--rounds 5] [--json out.json]
        [--tools F.linear,det_nk,det_kn,def_nk,def_kn,lin_det,lin_def]

Every case is ``relu(x @ W.T + b)`` in float32, TF32 off on both sides (scripts/bench_matmul.py).
Tools: ``F.linear`` (cuBLAS plus relu), ``det_*`` / ``def_*`` the kernel layer in deterministic and
default mode fed the weight stored [N, K] (``nk``) or [K, N] (``kn``, the view ``w_kn.t()``), and
``lin_*`` a ``truestar.compile`` of ``torch.nn.Linear`` + relu, which is where prepacking applies.
Each tool is a CUDA graph of 20 calls; rounds are interleaved; the median of the rounds is
reported (FAIR_BENCH_2026_11/METHOD.md: idle gate before the run, median of 5 rounds). Two calls
of every tool must give equal bits, and every output is checked against float64.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get("TRUESTAR_FAIR_BENCH", "fair_bench"), "scripts"))  # unpublished fair benchmark
import probe_ieee_mm as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--tools", default="F.linear,det_nk,det_kn,def_nk,def_kn")
    ap.add_argument("--json")
    args = ap.parse_args()
    import perf_gate
    gate = perf_gate.wait_idle()
    torch = P.setup()
    import torch.nn.functional as F
    from truestar import kernel as K
    tools = args.tools.split(",")
    print(f"gpu {torch.cuda.get_device_name(0)}, SMs "
          f"{torch.cuda.get_device_properties(0).multi_processor_count}", flush=True)
    rows = []
    for M, N, Kd in P.parse_shapes(args.shapes):
        x, w, b = P.data(torch, M, N, Kd)
        ref = torch.relu(x.double() @ w.double().T + b.double())
        wv = w.t().contiguous().t()
        fns = {}
        kern = P.kernel_of(N, Kd)
        cks = {}
        for t in tools:
            if t == "F.linear":
                fns[t] = lambda: torch.relu(F.linear(x, w, b))
            elif t[:3] in ("det", "def"):
                det = t.startswith("det")
                if det not in cks:
                    cks[det] = K.compile(kern, "cuda", tune=False, deterministic=det)
                ww = w if t.endswith("nk") else wv
                fns[t] = (lambda ck=cks[det], ww=ww: ck(x=x, w=ww, b=b))
            elif t.startswith("lin"):   # an imported nn.Linear + relu: the weight is a constant
                from truestar import ir
                from truestar.compiler import compile_program
                lin = torch.nn.Linear(Kd, N)
                with torch.no_grad():
                    lin.weight.copy_(w.cpu())
                    lin.bias.copy_(b.cpu())
                mod = torch.nn.Sequential(lin, torch.nn.ReLU()).eval()
                prog = ir.import_module(mod, (x.cpu(),), name=f"lin{M}x{N}x{Kd}")
                cp = compile_program(prog, "cuda", "float32", deterministic=(t == "lin_det"),
                                     allow_tf32=False)
                fns[t] = (lambda cp=cp: cp([x])[0])
        outs, same, errs = {}, {}, {}
        for t, f in fns.items():
            with torch.no_grad():
                o1 = f().clone()
                o2 = f().clone()
            outs[t] = o1
            same[t] = torch.equal(o1, o2)
            errs[t] = P.rel_err(o1, ref)
        with torch.no_grad():
            med, mn, redone = perf_gate.timed(torch, fns, args.rounds)
        cub = med.get("F.linear")
        cfg = {d: cks[d].launch_configs()[0][1] for d in cks}
        twin = lambda t: (t[4:] if t.startswith("lin") else t[:3]) + "_nk"  # noqa: E731
        bits = {t: torch.equal(outs[t], outs[twin(t)]) for t in outs
                if t != twin(t) and twin(t) in outs}
        row = dict(shape=[M, N, Kd], median_us=med, min_us=mn, rounds_redone=redone, error=errs, run_to_run_equal=same,
                   bits_equal_to_nk=bits, configs={str(k): v for k, v in cfg.items()})
        rows.append(row)
        print(f"{M}x{N}x{Kd} " + "  ".join(
            f"{t} {med[t]:.1f}" + (f" ({cub / med[t]:.2f}x)" if cub and t != "F.linear" else "")
            + ("" if same[t] else " RUNS DIFFER") for t in med)
            + f" | err " + " ".join(f"{t} {e:.1e}" for t, e in errs.items())
            + (" | layout bits " + ("same" if all(bits.values()) else f"DIFFER {bits}") if bits else "")
            + f" | cfg {row['configs']}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(gpu=torch.cuda.get_device_name(0), gate=gate, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
