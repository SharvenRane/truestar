"""Cited cases: can the integer reference tell the conventions apart, and does it name the right
one? (`truestar intref cases`, docs/INTREF.md)

1. CMSIS-NN#214: TFLite 2.21 was reported to use single rounding for FULLY_CONNECTED and double
   rounding for MUL. TensorFlow is not installed here, so the "device" is the reported behaviour
   (convention `tflite_2_21_py`), run on random inputs (what a user would try first) and on
   `distinguishing_case` inputs. The question is whether the per op verdict recovers it.
2. ONNX Runtime as a real oracle (when installed): QLinearConv, QLinearMatMul, QuantizeLinear (with
   exact ties), com.microsoft QLinearMul and QLinearAdd, CPU provider, graph optimizations on.
3. How many convention pairs each op's inputs separate, random against distinguishing inputs.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from .conventions import documented_conventions
from .match import distinguishing_case, match_op, pairs_distinguished
from .ops import run_op

FC_PARAMS = dict(x_zp=-5, sx=0.0173, sw=0.00731, sy=0.0412, y_zp=4)
MUL_PARAMS = dict(a_zp=-5, sa=0.0173, b_zp=9, sb=0.0291, sy=0.0377, y_zp=-3)
ADD_PARAMS = dict(a_zp=-3, sa=0.031, b_zp=5, sb=0.017, sy=0.044, y_zp=2)
CONV_PARAMS = dict(x_zp=-5, sx=0.0173, sw=0.00731, sy=0.0412, y_zp=4, padding=0)
# sa * sb / sy = 2^-5 exactly: products land on exact ties, so tie rules become visible.
MUL_DYADIC = dict(a_zp=0, sa=0.25, b_zp=0, sb=0.125, sy=1.0, y_zp=0)
REQ_PARAMS = dict(x_zp=-5, sx=0.0173, sy=0.0412, y_zp=4)


def _random_case(op: str, rng) -> Dict[str, np.ndarray]:
    i8 = lambda *s: rng.integers(-128, 128, s).astype(np.int8)  # noqa: E731
    if op == "fully_connected":
        return {"x": i8(64, 32), "w": rng.integers(-127, 128, (16, 32)).astype(np.int8),
                "b": rng.integers(-2000, 2000, 16).astype(np.int32)}
    if op == "conv":
        return {"x": i8(1, 4, 10, 10), "w": rng.integers(-127, 128, (8, 4, 3, 3)).astype(np.int8),
                "b": rng.integers(-2000, 2000, 8).astype(np.int32)}
    if op in ("mul", "add"):
        return {"a": i8(4096), "b": i8(4096)}
    return {"x": i8(4096)}


PARAMS = {"fully_connected": FC_PARAMS, "mul": MUL_PARAMS, "add": ADD_PARAMS, "conv": CONV_PARAMS,
          "requantize": REQ_PARAMS}


def cmsis_214(seed: int = 0) -> dict:
    """Per op verdicts for a device that behaves as CMSIS-NN#214 reports for TFLite 2.21."""
    rng = np.random.default_rng(seed)
    convs = documented_conventions()
    out = {}
    for op in ("fully_connected", "mul"):
        p = PARAMS[op]
        for kind, case in (("random", _random_case(op, rng)), ("distinguishing", distinguishing_case(op, p))):
            dev = run_op(op, case, p, "tflite_2_21_py")
            v = match_op(op, case, p, dev, convs)
            out[f"{op}/{kind}"] = v.as_dict()
    fc = set(out["fully_connected/distinguishing"]["matched"])
    mu = set(out["mul/distinguishing"]["matched"])
    single_build = [c for c in ("cmsis_nn", "cmsis_nn_single") if c in fc and c in mu]
    out["summary"] = {
        "fully_connected_matches": sorted(fc), "mul_matches": sorted(mu),
        "one_cmsis_build_matches_both": bool(single_build),
        "text": (f"FULLY_CONNECTED matches {', '.join(sorted(fc))}; MUL matches {', '.join(sorted(mu))}; "
                 + ("no single CMSIS-NN build matches both ops, as the issue reports"
                    if not single_build else f"{single_build} matches both"))}
    return out


def pair_table(seed: int = 0) -> List[dict]:
    """Convention pairs separated per op, random against distinguishing inputs."""
    rng = np.random.default_rng(seed)
    convs = documented_conventions()
    rows = []
    for label, op, p in ([(o, o, PARAMS[o]) for o in ("conv", "fully_connected", "mul", "add", "requantize")]
                         + [("mul, dyadic scales", "mul", MUL_DYADIC)]):
        r = pairs_distinguished(op, _random_case(op, rng), p, convs)
        d = pairs_distinguished(op, distinguishing_case(op, p, convs), p, convs)
        rows.append({"op": label, "pairs": r["pairs"], "random": r["distinguished"],
                     "distinguishing": d["distinguished"],
                     "identical_groups": [g for g in d["groups"] if len(g) > 1]})
    return rows


# ----------------------------------------------------------------------------- ONNX Runtime oracle


def _ort_run(nodes, feeds, inits):
    import onnxruntime as ort
    from onnx import helper as h
    from onnx import numpy_helper as nh

    tp = {np.dtype(np.int8): 3, np.dtype(np.uint8): 2, np.dtype(np.float32): 1}
    g = h.make_graph(nodes, "g", [h.make_tensor_value_info(k, tp[v.dtype], v.shape) for k, v in feeds.items()],
                     [h.make_empty_tensor_value_info("y")],
                     [nh.from_array(np.asarray(v), k) for k, v in inits.items()])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17), h.make_opsetid("com.microsoft", 1)])
    m.ir_version = 8
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    return s.run(None, feeds)[0]


def onnxruntime_cases(seed: int = 0) -> Optional[dict]:
    """Run real ONNX Runtime kernels and ask which convention each matches. None if ORT or onnx
    is not installed."""
    try:
        import onnxruntime  # noqa: F401
        from onnx import helper as h
    except ImportError:
        return None
    import onnxruntime as ort_mod

    rng = np.random.default_rng(seed)
    f32 = np.float32
    convs = documented_conventions()
    out = {"onnxruntime": ort_mod.__version__}
    # QLinearConv: uint8 activations, int8 per channel weights.
    x = rng.integers(0, 256, (2, 8, 12, 12)).astype(np.uint8)
    w = rng.integers(-127, 128, (16, 8, 3, 3)).astype(np.int8)
    b = rng.integers(-2000, 2000, 16).astype(np.int32)
    sw = rng.uniform(0.002, 0.02, 16).astype(f32)
    got = _ort_run([h.make_node("QLinearConv", ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz", "b"], ["y"],
                                pads=[1, 1, 1, 1])], {"x": x},
                   dict(xs=f32(0.0213), xz=np.uint8(121), w=w, ws=sw, wz=np.zeros(16, np.int8),
                        ys=f32(0.173), yz=np.uint8(97), b=b))
    p = dict(x_zp=121, sx=float(f32(0.0213)), sw=sw.astype(np.float64), sy=float(f32(0.173)),
             y_zp=97, padding=1, out_signed=False)
    out["QLinearConv"] = match_op("conv", {"x": x, "w": w, "b": b}, p, got, convs).as_dict()
    # QLinearMatMul
    a = rng.integers(-128, 128, (4, 32, 64)).astype(np.int8)
    bb = rng.integers(-128, 128, (64, 24)).astype(np.int8)
    got = _ort_run([h.make_node("QLinearMatMul", ["a", "as", "az", "b", "bs", "bz", "ys", "yz"], ["y"])],
                   {"a": a}, dict(**{"as": f32(0.05)}, az=np.int8(3), b=bb, bs=f32(0.011),
                                  bz=np.int8(0), ys=f32(0.9), yz=np.int8(-5)))
    p = dict(a_zp=3, b_zp=0, sa=float(f32(0.05)), sb=float(f32(0.011)), sy=float(f32(0.9)), y_zp=-5)
    out["QLinearMatMul"] = match_op("matmul", {"a": a, "b": bb}, p, got, convs).as_dict()
    # QuantizeLinear with exact ties (x / scale = k + 0.5)
    xf = np.concatenate([(np.arange(-300, 300) + 0.5) * f32(0.0625),
                         rng.normal(0, 3, 4000)]).astype(f32)
    got = _ort_run([h.make_node("QuantizeLinear", ["x", "s", "z"], ["y"])], {"x": xf},
                   dict(s=f32(0.0625), z=np.int8(1)))
    out["QuantizeLinear"] = match_op("quantize", {"x": xf}, dict(scale=0.0625, zp=1), got, convs).as_dict()
    # Contrib QLinearMul and QLinearAdd on every int8 pair.
    case = distinguishing_case("mul", MUL_PARAMS)
    for name, op, pp in (("QLinearMul", "mul", MUL_PARAMS), ("QLinearAdd", "add", ADD_PARAMS)):
        inits = dict(**{"as": f32(pp["sa"])}, az=np.int8(pp["a_zp"]), bs=f32(pp["sb"]),
                     bz=np.int8(pp["b_zp"]), ys=f32(pp["sy"]), yz=np.int8(pp["y_zp"]))
        pf = {k: (float(f32(v)) if k.startswith("s") else v) for k, v in pp.items()}
        got = _ort_run([h.make_node(name, ["a", "as", "az", "b", "bs", "bz", "ys", "yz"], ["y"],
                                    domain="com.microsoft")], {"a": case["a"], "b": case["b"]}, inits)
        out[name] = match_op(op, case, pf, got, convs).as_dict()
    return out


def run_cases(seed: int = 0, with_onnxruntime: bool = True) -> dict:
    t0 = time.time()
    res = {"cmsis_nn_214": cmsis_214(seed), "pairs": pair_table(seed)}
    res["onnxruntime"] = onnxruntime_cases(seed) if with_onnxruntime else None
    res["seconds"] = round(time.time() - t0, 2)
    return res


def cases_text(res: dict) -> str:
    c = res["cmsis_nn_214"]
    lines = ["truestar intref cases", "", "CMSIS-NN#214 (device = TFLite 2.21 as reported):"]
    for k in ("fully_connected/random", "fully_connected/distinguishing", "mul/random", "mul/distinguishing"):
        lines.append(f"  {k:32s} {c[k]['code']}  {c[k]['verdict']}")
    lines.append("  " + c["summary"]["text"])
    lines += ["", "convention pairs separated (documented conventions):",
              "  op                 pairs  random  distinguishing  identical on every input tried"]
    for r in res["pairs"]:
        lines.append(f"  {r['op']:18s} {r['pairs']:5d} {r['random']:7d} {r['distinguishing']:15d}  "
                     + "; ".join(", ".join(g) for g in r["identical_groups"]))
    o = res.get("onnxruntime")
    if o:
        lines += ["", f"ONNX Runtime {o['onnxruntime']} CPU kernels as the device:"]
        for k, v in o.items():
            if k != "onnxruntime":
                lines.append(f"  {k:15s} {v['code']}  {v['verdict']}")
    else:
        lines += ["", "ONNX Runtime not installed: oracle cases skipped"]
    lines.append(f"\n{res['seconds']:.1f} s")
    return "\n".join(lines)
