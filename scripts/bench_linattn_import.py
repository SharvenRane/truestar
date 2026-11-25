"""Mamba 2 SSD and Gated DeltaNet families through the importer lowering (pain row P60).

For each family of ``truestar.testing_hf.LOWERED_2026``:

* before: the import with the lowering switched off (``hf_linattn.SPECS`` emptied), which is the
  refusal by name main had (TS941, TS942);
* after: import of the float64 model, the largest error of ``ir.run`` against eager PyTorch float64
  (transformers' float32 casts made exact, as in tests/test_hf_models.py) at batch 2 and 5, and of
  the one step gradient of every weight against torch.autograd;
* speed, one small config: the test model on a longer sequence, float32, eager PyTorch forward
  (transformers' own PyTorch path: no mamba_ssm, causal_conv1d or fla is installed) against the
  Truestar program compiled for the same device, same weights and inputs, median of ``--reps``
  calls after a warm up call (the compiled program's first call is its float64 check).

    python scripts/bench_linattn_import.py --device cpu --seq 256 --out results/linattn_import/cpu.json

Needs a transformers with the families (5.24 was measured, docs/LINEAR_ATTENTION.md section 10).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import time

import torch

from truestar import ir
from truestar import testing_hf as th
from truestar.ir import hf_linattn


def _rel(got, want):
    scale = want.abs().max().item() or 1.0
    return (got.double() - want.double()).abs().max().item() / scale


def _before(name):
    model, args = th.build(name, batch=2)
    saved = dict(hf_linattn.SPECS)
    hf_linattn.SPECS.clear()
    try:
        ir.import_module(copy.deepcopy(model).double(), args)
        return "imports"
    except Exception as e:   # noqa: BLE001
        return f"refused {getattr(e, 'code', type(e).__name__)}"
    finally:
        hf_linattn.SPECS.update(saved)


def _after(name):
    from truestar.ir.autodiff import grad
    model, args = th.build(name, batch=2)
    m64 = copy.deepcopy(model).double()
    prog = ir.import_module(m64, args, dynamic_shapes=({0: torch.export.Dim("B", min=1, max=64)},))
    fwd = 0.0
    for batch, seed in ((2, 1), (5, 2)):
        _, xs = th.build(name, batch=batch, seed=seed)
        with torch.no_grad(), ir.reference.exact_float32_casts():
            want = m64(*xs)
        got = ir.run(prog, *xs)
        fwd = max(fwd, max(_rel(g, w) for g, w in zip(got, want)))
    _, xs = th.build(name, batch=2, seed=3)
    sd = m64.state_dict(keep_vars=True)
    names = [v.name for v in prog.params if v.name in sd and isinstance(sd[v.name], torch.nn.Parameter)
             and sd[v.name].requires_grad]
    out0 = ir.run(prog, *xs)[0]
    cot = torch.randn(out0.shape, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    got = ir.run(grad(prog, wrt=names, of=[0], seed="input"), *xs, cot)[len(prog.outputs):]
    m64.zero_grad(set_to_none=True)
    with ir.reference.exact_float32_casts():
        m64(*xs)[0].backward(cot)
    scale = max(sd[n].grad.abs().max().item() for n in names if sd[n].grad is not None)
    gerr = max((g - (sd[n].grad if sd[n].grad is not None else torch.zeros_like(g))).abs().max().item() / scale
               for n, g in zip(names, got))
    return {"layers_lowered": prog.op_counts().get("linear_attention", 0), "fp64_max_rel_err": fwd,
            "grad_max_rel_err": gerr}


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _speed(name, device, seq, batch, reps):
    from truestar.compiler import compile_program
    torch.manual_seed(0)
    model, _ = th.build(name, batch=batch)
    ids = torch.randint(3, th.VOCAB, (batch, seq), generator=torch.Generator().manual_seed(1))
    prog = ir.import_module(copy.deepcopy(model).double(), (ids,))
    eager = model.to(device)
    x = ids.to(device)
    res = {"seq": seq, "batch": batch, "device": device}
    with torch.no_grad():
        eager(x)
        ts = []
        for _ in range(reps):
            _sync(device)
            t = time.perf_counter()
            want = eager(x)[0]
            _sync(device)
            ts.append(time.perf_counter() - t)
    res["eager_ms"] = 1e3 * statistics.median(ts)
    cp = compile_program(prog, device, "float32")
    try:
        out = cp([x])[0]                       # first call: checked against the float64 reference
    except Exception as e:   # noqa: BLE001  (a refused compiled program is reported, not timed)
        res["truestar"] = f"refused: {str(e).splitlines()[0][:160]}"
        k = [ln for ln in str(e).splitlines() if "TS1060" in ln]
        res["reason"] = k[0].strip()[:300] if k else ""
        return res
    res["fallbacks"] = len(cp.report.fallbacks)
    res["rel_err_vs_eager_fp32"] = _rel(out.cpu(), want.cpu())
    ts = []
    for _ in range(reps):
        _sync(device)
        t = time.perf_counter()
        cp([x])
        _sync(device)
        ts.append(time.perf_counter() - t)
    res["truestar_ms"] = 1e3 * statistics.median(ts)
    res["speedup"] = res["eager_ms"] / res["truestar_ms"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--families", default=",".join(th.LOWERED_2026))
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import transformers
    rows = {}
    for name in a.families.split(","):
        r = {"before": _before(name)}
        if not a.skip_accuracy:
            r.update(_after(name))
        r["speed"] = _speed(name, a.device, a.seq, a.batch, a.reps)
        rows[name] = r
        print(name, json.dumps(r), flush=True)
    meta = {"transformers": transformers.__version__, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name() if a.device == "cuda" else None, "tf32": False}
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"meta": meta, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
