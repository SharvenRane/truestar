"""Measurements for docs/CONFORM.md: every artifact and runtime here, against the usual check.

    PYTHONPATH=src python scripts/conform_study.py [--out DIR]

Baseline: what the LiteRT PyTorch quickstart does, `numpy.allclose(torch_output, artifact_output,
atol=1e-5, rtol=1e-5)` on one input (here the example input), which says pass or fail and nothing
else. Truestar: `conform` with its defaults (8 samples). For each row: whether each one flags the
file, Truestar's status, codes, the module it blames with the cause, the off target nodes it names,
and the seconds conform took (the reference, all samples, localization and the bundle).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))

import conform_models as cm  # noqa: E402
from truestar.conform import conform  # noqa: E402
from truestar.conform import runtimes as R  # noqa: E402


def baseline(model, x, rt_name, path):
    """One input, one allclose, as the quickstart does."""
    from truestar.conform.artifact import load

    art = load(path)
    rt = R.make(rt_name)
    s = rt.open(art.data, art)
    try:
        out = s.run([x.numpy()])[0]
    finally:
        s.close()
    with torch.no_grad():
        ref = model(x).numpy()
    return not bool(np.allclose(ref, out, atol=1e-5, rtol=1e-5))


def cases(d):
    x = cm.example()
    m = cm.make_classifier()
    mc = cm.make_classifier(cumsum=True)
    torch.manual_seed(0)
    q = cm.QuantClassifier().eval()
    out = []
    have_onnx = R.make("ort:cpu").unavailable() is None
    if have_onnx:
        clean = cm.export_onnx(m, x, os.path.join(d, "clf.onnx"))
        out.append(("onnx clean", m, clean))
        out.append(("onnx gelu as tanh (exporter decomposition)", m, cm.edit_onnx(clean, os.path.join(d, "g.onnx"), "gelu_tanh")))
        out.append(("onnx Det node (no GPU kernel)", m, cm.edit_onnx(clean, os.path.join(d, "det.onnx"), "det_node")))
        qc = cm.export_onnx(q, x, os.path.join(d, "q.onnx"))
        out.append(("onnx INT8 fake quant clean", q, qc))
        out.append(("onnx INT8 round to floor in q2", q, cm.edit_onnx(qc, os.path.join(d, "qf.onnx"), "int8_floor")))
    if R.make("tflite:interp").unavailable() is None:
        out.append(("tflite clean", m, cm.build_tflite(m, x, os.path.join(d, "clf.tflite"))))
        out.append(("tflite cumsum (outside XNNPACK)", mc, cm.build_tflite(mc, x, os.path.join(d, "cum.tflite"))))
        out.append(("tflite layer norm eps outside sqrt", m, cm.build_tflite(m, x, os.path.join(d, "eps.tflite"), ("layernorm_eps_outside",))))
        out.append(("tflite gelu as tanh", m, cm.build_tflite(m, x, os.path.join(d, "gt.tflite"), ("gelu_tanh",))))
    if R.make("pte:cpu").unavailable() is None:
        out.append(("pte XNNPACK, cumsum model", mc, cm.build_pte(mc, x, os.path.join(d, "m.pte"))))
    return x, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    d = a.out or tempfile.mkdtemp(prefix="truestar_conform_study_")
    os.makedirs(d, exist_ok=True)
    x, rows = cases(d)
    print(f"torch {torch.__version__}; artifacts in {d}")
    print("| artifact | runtime | allclose flags | truestar | codes | blamed (cause) | off target nodes | s |")
    print("|---|---|---|---|---|---|---|---|")
    for name, model, path in rows:
        fmt = os.path.splitext(path)[1][1:]
        fmt = {"onnx": "onnx", "tflite": "tflite", "pte": "pte"}[fmt]
        names = [n for n, (f, _) in R.RUNTIMES.items() if f == fmt]
        t0 = time.perf_counter()
        rep = conform(model, x, path, runtimes=[n for n in names if R.make(n).unavailable() is None],
                      golden=path + ".tsgold")
        dt = time.perf_counter() - t0
        for r in rep.results:
            if r.status == "unavailable":
                continue
            try:
                base = "yes" if baseline(model, x, r.runtime, path) else "no"
            except Exception as exc:
                base = f"error {type(exc).__name__}"
            codes = ",".join(sorted({c for c, _ in r.issues})) or "-"
            blamed = f"{r.blamed.module} ({r.blamed.cause})" if r.blamed else "-"
            off = ", ".join(f"{p['op']}@{p['module'] or 'top'}" for p in (r.residency or {}).get("off_target", [])) or "-"
            prec = f" {r.precision.get('observed')}" if r.precision.get("observed") not in (None, "float32") else ""
            print(f"| {name} | {r.runtime} | {base} | {r.status}{prec} | {codes} | {blamed} | {off} | "
                  f"{r.seconds:.2f} |")
        print(f"<!-- {name}: conform total {dt:.2f} s -->")


if __name__ == "__main__":
    main()
