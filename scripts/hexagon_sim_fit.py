"""Fit and measure sim:hexagon against real Snapdragon NPU outputs (docs/DEVICES.md section 7, P59).

Inputs are the saved outputs of the Qualcomm AI Hub battery (docs/QAIHUB.md): a bundle directory
with ``summary.json`` and one ``<run>.npz`` per run.

    python scripts/hexagon_sim_fit.py fit BUNDLE [--run float__s26__tflite__npu]
    python scripts/hexagon_sim_fit.py measure BUNDLE
    python scripts/hexagon_sim_fit.py fixture BUNDLE tests/data/hexagon_npu_taps.npz

``fit`` recomputes the GELU curve in ``truestar.devsim.hexagon`` from the MLP's first layer: the
device's GELU input is the float16 linear layer's unrounded accumulator (fused), its output is
``x * phi(x)``; ``phi`` is fitted per sign as a line below |x| = 0.6 and a degree 5 polynomial
above, on every other input sorted by value. The other half is held out and scored.
``measure`` prints, per run and tap, the error of sim:ieee-f16 and sim:hexagon against the device.
``fixture`` writes the small set of device taps the tests use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from truestar.conform import qaihub_battery as B
from truestar.conform.qaihub import _to_shape, match_outputs

FP16_RUNS = ("float__s26__tflite__npu", "float__s26__dlc__npu", "float__xelite__dlc__npu",
             "float__xelite__onnx__npu")
BREAK = 0.6
FIT_LIMIT = 3.0


def r16(v):
    return np.asarray(v, np.float64).astype(np.float16).astype(np.float64)


def device_taps(bundle: str, run: str):
    f64 = B.float_reference()
    shapes = {t: f64[t].shape for t in B.FLOAT_TAPS}
    z = dict(np.load(os.path.join(bundle, run + ".npz")))
    m = match_outputs(z, "float", shapes)
    return {t: r16(_to_shape(m[t], shapes[t])) for t in B.FLOAT_TAPS if t in m}


def fused_preactivation():
    """fc1 of the battery MLP on float16 rounded input, weight and bias, summed exactly."""
    arrs = B.inputs("float")
    fc1 = B.float_model().mlp.fc1
    x = r16(arrs["x_mlp"])
    w = r16(fc1.weight.detach().numpy())
    b = r16(fc1.bias.detach().numpy())
    return x @ w.T + b


def split(x):
    order = np.argsort(x.ravel(), kind="stable")
    train = np.zeros(x.size, bool)
    train[order[::2]] = True
    return train.reshape(x.shape)


def fit(x, dev, train):
    out = {}
    for side, sel in (("neg", x < -1e-3), ("pos", x > 1e-3)):
        s = sel & train & (np.abs(x) < FIT_LIMIT)
        a, phi = np.abs(x[s]), dev[s] / x[s]
        # weight by the inverse of the output's float16 half ulp, seen through phi = out / x
        w = a / np.maximum(np.abs(np.spacing(dev[s].astype(np.float16)).astype(np.float64)), 1e-12)
        lo = a < BREAK
        out[side] = (np.polyfit(a[lo], phi[lo], 1, w=w[lo]), np.polyfit(a[~lo], phi[~lo], 5, w=w[~lo]))
    return out


def rel_l2(a, ref):
    n = np.linalg.norm(ref)
    return float(np.linalg.norm(a - ref) / n) if n else float(np.linalg.norm(a - ref))


def cmd_fit(args):
    from truestar.devsim import hexagon as H

    x = fused_preactivation()
    dev = device_taps(args.bundle, args.run)["mlp.fc1_gelu"]
    train = split(x)
    coef = fit(x, dev, train)
    for side, (lo, hi) in coef.items():
        print(f"{side}: line {np.array2string(lo, separator=', ', precision=17)}")
        print(f"{side}: degree 5 {np.array2string(hi, separator=', ', precision=17)}")
    test = ~train
    shipped = r16(H.gelu(torch.from_numpy(x)).numpy())
    textbook = r16(x * 0.5 * (1 + torch.erf(torch.from_numpy(x) / np.sqrt(2)).numpy()))
    rounded_in = r16(torch.nn.functional.gelu(torch.from_numpy(r16(x))).numpy())
    unfused = r16(H.gelu(torch.from_numpy(r16(x))).numpy())
    print(json.dumps({
        "held_out_points": int(test.sum()),
        "shipped_curve": {"rel_l2": rel_l2(shipped[test], dev[test]),
                          "bit_equal": float(np.mean(shipped[test] == dev[test]))},
        "shipped_curve_of_float16_input": {"rel_l2": rel_l2(unfused[test], dev[test]),
                                           "bit_equal": float(np.mean(unfused[test] == dev[test]))},
        "textbook_gelu_of_float16_input": {"rel_l2": rel_l2(rounded_in[test], dev[test]),
                                           "bit_equal": float(np.mean(rounded_in[test] == dev[test]))},
        "textbook_gelu_of_fused_input": {"rel_l2": rel_l2(textbook[test], dev[test]),
                                         "bit_equal": float(np.mean(textbook[test] == dev[test]))},
    }, indent=1))


def cmd_measure(args):
    want = B.float_reference()
    sims = {s: B.simulated(s) for s in ("sim:ieee-f16", "sim:hexagon")}
    rows = []
    for run in FP16_RUNS:
        if not os.path.exists(os.path.join(args.bundle, run + ".npz")):
            continue
        dev = device_taps(args.bundle, run)
        for t in B.FLOAT_TAPS:
            row = {"run": run, "tap": t, "device_to_f64": rel_l2(dev[t], want[t])}
            for s, out in sims.items():
                row[s] = {"to_device": rel_l2(r16(out[t]), dev[t]),
                          "bit_equal": float(np.mean(r16(out[t]) == dev[t]))}
            rows.append(row)
    print(json.dumps(rows, indent=1))
    print("\n| run | tap | sim:ieee-f16 to device | sim:hexagon to device | ratio | bit equal before | after |",
          file=sys.stderr)
    print("|---|---|---|---|---|---|---|", file=sys.stderr)
    for r in rows:
        a, b = r["sim:ieee-f16"], r["sim:hexagon"]
        ratio = a["to_device"] / b["to_device"] if b["to_device"] else float("inf")
        print(f"| {r['run']} | {r['tap']} | {a['to_device']:.2e} | {b['to_device']:.2e} | {ratio:.2f}x | "
              f"{a['bit_equal']:.1%} | {b['bit_equal']:.1%} |", file=sys.stderr)


def cmd_fixture(args):
    s26 = device_taps(args.bundle, "float__s26__tflite__npu")
    xe = device_taps(args.bundle, "float__xelite__dlc__npu")
    np.savez_compressed(
        args.out,
        s26_fc1_gelu=s26["mlp.fc1_gelu"].astype(np.float16),
        s26_fc2_relu=s26["mlp.fc2_relu"].astype(np.float16),
        s26_enc_out=s26["enc.out"].astype(np.float16),
        s26_conv_stem_c01=s26["conv.stem"][:, :2].astype(np.float16),
        s26_conv_out=s26["conv.out"].astype(np.float16),
        xelite_fc1_gelu=xe["mlp.fc1_gelu"].astype(np.float16),
    )
    print(args.out, os.path.getsize(args.out), "bytes")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("bundle")
    f.add_argument("--run", default="float__s26__tflite__npu")
    m = sub.add_parser("measure")
    m.add_argument("bundle")
    x = sub.add_parser("fixture")
    x.add_argument("bundle")
    x.add_argument("out")
    args = ap.parse_args(argv)
    {"fit": cmd_fit, "measure": cmd_measure, "fixture": cmd_fixture}[args.cmd](args)


if __name__ == "__main__":
    main()
