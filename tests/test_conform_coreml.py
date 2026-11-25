"""truestar conform coreml (docs/COREML.md): the check half on constructed bundles, anywhere; the
collect half only on a Mac with coremltools."""

import json
import os
import platform

import numpy as np
import pytest

from truestar.conform import coreml as C


@pytest.fixture(scope="module")
def refs():
    return C.references("mlp")


def _ops(preferred, supported, n=4):
    return [{"op": "linear", "preferred": preferred, "supported": supported} for _ in range(n)] + \
           [{"op": "const", "preferred": "CPU", "supported": ["CPU"]}]


def _bundle(tmp_path, runs, refs, hashes=None):
    s = {"battery": "coreml", "battery_version": C.BATTERY_VERSION, "seed": C.SEED,
         "input_hashes": hashes or {"mlp": C.input_hash(C.build("mlp")[1])}, "runs": {}}
    for name, (row, y) in runs.items():
        s["runs"][name] = row
        if y is not None:
            np.savez(os.path.join(tmp_path, name + ".npz"), y=np.asarray(y, dtype=np.float32))
    with open(os.path.join(tmp_path, "summary.json"), "w") as f:
        json.dump(s, f)
    return str(tmp_path)


def _row(prec, units, ops, **kw):
    return dict({"model": "mlp", "precision": prec, "units": units, "ops": ops, "run_to_run_equal": True,
                 "median_ms": 1.0}, **kw)


def test_battery_is_portable_and_hashed():
    m, x = C.build("mlp")
    m2, x2 = C.build("mlp")
    assert C.input_hash(x) == C.input_hash(x2)
    w = m[0].weight.detach().numpy()
    assert np.array_equal(w, m2[0].weight.detach().numpy())
    q, _ = C.build("mlp_int8")
    wq = q[0].weight.detach().double().numpy()
    scale = np.abs(wq).max(axis=1, keepdims=True) / 127.0
    assert np.allclose(wq / scale, np.rint(wq / scale), atol=1e-3)       # on the int8 grid


def test_verdicts_and_placement(tmp_path, refs):
    rng = np.random.default_rng(0)
    big = refs["f64"] + 0.05 * np.abs(refs["f64"]).max() * rng.standard_normal(refs["f64"].shape)
    runs = {
        "fp32_cpu": (_row("FLOAT32", "CPU_ONLY", _ops("CPU", ["CPU", "GPU"])), refs["f32"]),
        "fp32_gpu_fallback": (_row("FLOAT32", "CPU_AND_GPU", _ops("CPU", ["CPU", "GPU"])), refs["f32"]),
        "fp32_ne": (_row("FLOAT32", "CPU_AND_NE", _ops("CPU", ["CPU", "GPU"])), refs["f32"]),
        "fp32_at_fp16": (_row("FLOAT32", "CPU_AND_GPU", _ops("GPU", ["CPU", "GPU"])), refs["sim16"]),
        "fp16_ne": (_row("FLOAT16", "CPU_AND_NE", _ops("NE", ["CPU", "GPU", "NE"])), refs["sim16"]),
        "fp16_bad": (_row("FLOAT16", "ALL", _ops("NE", ["NE"]), run_to_run_equal=False), big),
        "failed": ({"model": "mlp", "precision": "FLOAT16", "units": "ALL", "error": "run: boom"}, None),
    }
    d = _bundle(tmp_path, runs, refs)
    vs, top = C.check(d)
    assert top == []
    v = {x.run: x for x in vs}
    codes = {k: {c for c, _ in x.issues} for k, x in v.items()}
    assert v["fp32_cpu"].verdict == "float32" and codes["fp32_cpu"] == set()
    assert "TS1160" in codes["fp32_gpu_fallback"] and v["fp32_gpu_fallback"].placement["off_requested"] == 4
    assert "TS1161" in codes["fp32_ne"]
    assert v["fp32_at_fp16"].verdict == "float16" and "TS1162" in codes["fp32_at_fp16"]
    assert v["fp16_ne"].verdict == "float16" and codes["fp16_ne"] == set()
    assert v["fp16_bad"].verdict == "beyond_float16" and {"TS1163", "TS1167"} <= codes["fp16_bad"]
    assert v["failed"].verdict == "failed" and "TS1164" in codes["failed"]
    assert C.is_finding(vs, top)
    assert "fp32_gpu_fallback" in C.to_text(vs, top)


def test_input_hash_mismatch_and_bad_bundle(tmp_path, refs):
    d = _bundle(tmp_path, {"r": (_row("FLOAT32", "CPU_ONLY", []), refs["f32"])}, refs, hashes={"mlp": "0" * 64})
    vs, top = C.check(d)
    assert vs == [] and [c for c, _ in top] == ["TS1166"]
    with open(os.path.join(d, "summary.json")) as f:
        s = json.load(f)
    s["battery_version"] = 99
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(s, f)
    with pytest.raises(C.CoreMLBundleError):
        C.check(d)


def test_cli_exit_codes(tmp_path, refs):
    from truestar.cli import main
    d = _bundle(tmp_path, {"ok": (_row("FLOAT32", "CPU_ONLY", _ops("CPU", ["CPU"])), refs["f32"])}, refs)
    assert main(["conform", "coreml", d, "--format", "agent"]) == 0
    assert main(["conform", "coreml", str(tmp_path / "missing"), "--format", "agent"]) == 2
    assert main(["conform", "coreml", "--format", "agent"]) == 2


def test_codes_are_registered():
    from truestar import codes
    for c in ("TS1160", "TS1161", "TS1162", "TS1163", "TS1164", "TS1165", "TS1166", "TS1167", "TS1168"):
        assert c in codes.CODES and codes.slug(c).startswith("coreml_")


@pytest.mark.skipif(platform.system() != "Darwin", reason="collecting needs macOS with Core ML")
def test_collect_on_a_mac(tmp_path):
    pytest.importorskip("coremltools")
    d = C.collect(str(tmp_path), models=("mlp",), precisions=("FLOAT16",), units=("CPU_ONLY", "CPU_AND_GPU"),
                  reps=2, log=None)
    vs, top = C.check(d)
    assert top == [] and len(vs) == 2
    # judged, not failed: on an M2 (macOS 26.6) the FLOAT16 CPU run is beyond correct float16 (TS1163)
    assert all(v.verdict != "failed" and v.metrics and v.placement["compute_ops"] > 0 for v in vs), \
        [v.as_dict() for v in vs]
