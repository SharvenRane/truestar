"""Real INT8 agreement and per layer blame study on hosted phones (Qualcomm AI Hub).

    PYTHONPATH=src python scripts/npu_study.py --out npu_study.json

Needs qai-hub and a configured API token. Every AI Hub step is cached (see HubCache), so a rerun
with nothing changed submits no jobs; the printed job log says what was submitted and what reused.

What it runs, with the deterministic model from truestar.remote.synthetic (ResNet18, 20 classes,
128x128 inputs, seed 0):

1. INT8 w8a8 agreement: float ONNX compile, AI Hub quantize job (64 calibration inputs), one TFLite
   compile, then all eval inputs in one inference job per (device, compute unit). Host runs of the
   same QDQ model with ONNX Runtime (simulated, and with integer kernels) are added for contrast.
2. Float TFLite on the first device's NPU and CPU, for contrast with INT8.
3. Per layer blame: the model with block level taps as extra outputs, NPU vs CPU control on the
   first device, as a float model and as an INT8 model. For INT8, each unit is also compared tap by
   tap with ONNX Runtime running the same tapped QDQ model on this PC (no extra jobs).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from truestar.remote.agreement import RunLogits, layer_blame  # noqa: E402
from truestar.remote.qai_hub_quant import (DEFAULT_DEVICES, HubSession, download_onnx,  # noqa: E402
                                           int8_agreement_study, layer_blame_study,
                                           match_outputs, onnx_and_quantized, onnxruntime_logits,
                                           onnxruntime_outputs)
from truestar.remote.synthetic import build_task  # noqa: E402


def host_layer_compare(ls, x):
    """Which phone unit follows the intended INT8 model? Compare each, tap by tap, with ONNX
    Runtime running the same tapped QDQ model on this PC with every Q/DQ kept (simulated INT8)."""
    names = ls.taps + ["output"]
    with tempfile.TemporaryDirectory() as td:
        sim = onnxruntime_outputs(download_onnx(ls.source_model, td), x, optimized=False)
    sim = match_outputs(sim, [r.shape for r in [ls.reference[-1]] + ls.reference[:-1]])
    sim = sim[1:] + sim[:1]
    out = {}
    for unit, arrs in ls.outputs.items():
        b = layer_blame(names, arrs, sim, ls.reference, quantized=True)
        print(f"{unit} vs host simulated INT8 (same QDQ model):")
        print(b.to_text())
        out[unit] = {"first_over_threshold": b.first.name if b.first else None,
                     "rows": [r.as_dict() for r in b.rows]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write all results as JSON here")
    ap.add_argument("--devices", default=",".join(DEFAULT_DEVICES))
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=16, help="inputs for the per layer runs")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--skip", default="", help="comma list of: agreement,float,layers,host")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    devices = a.devices.split(",")
    t0 = time.time()
    task = build_task(n_eval=a.n_eval)
    sess = HubSession(cache_dir=a.cache_dir)
    results = {"devices": devices, "n_eval": a.n_eval, "num_classes": task.num_classes,
               "image_size": task.image_size}

    if "agreement" not in skip:
        host = []
        if "host" not in skip:
            _onnx, q, _ = onnx_and_quantized(sess, task.model, task.calib_x, devices[0])
            with tempfile.TemporaryDirectory() as td:
                path = download_onnx(q, td)
                for opt, label in ((False, "int8 host ORT simulated"), (True, "int8 host ORT kernels")):
                    try:
                        host.append(RunLogits(label, onnxruntime_logits(path, task.eval_x, opt), "int8"))
                    except Exception as exc:  # noqa: BLE001
                        print(f"{label} failed: {exc}")
        units = [(devices[0], "npu"), (devices[0], "cpu")]
        for d in devices[1:]:
            units += [(d, "npu"), (d, "cpu")]
        float_units = [] if "float" in skip else [(devices[0], "npu"), (devices[0], "cpu")]
        study = int8_agreement_study(task.model, task.eval_x, task.eval_y, task.calib_x,
                                     devices=devices, units=units, session=sess, host_runs=host,
                                     float_units=float_units)
        print(study.to_text())
        results["agreement"] = study.as_dict()

    if "layers" not in skip:
        x = task.eval_x[:a.n_layer]
        for quantized in (False, True):
            ls = layer_blame_study(task.model, x, device=devices[0], quantized=quantized,
                                   calib_x=task.calib_x, session=sess)
            print(ls.to_text())
            results["layers_int8" if quantized else "layers_float"] = ls.as_dict()
            if quantized and "host" not in skip:
                results["layers_int8_vs_host"] = host_layer_compare(ls, x)

    results["job_log"] = {"submitted": sess.log.submitted, "reused": sess.log.reused}
    results["seconds"] = round(time.time() - t0, 1)
    print(f"jobs submitted this run: {len(sess.log.submitted)}, reused: {len(sess.log.reused)}, "
          f"{results['seconds']} s")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
