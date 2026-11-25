"""Minimal repro: ONNX Runtime's WebGPU EP accumulates some float16 MatMul shapes in float16.

One MatMul node, float16 inputs, compared with float64 numpy. Measured with onnxruntime 1.29.0 and
onnxruntime-ep-webgpu 0.6.0 on Windows 11 (RTX 5070 Ti and Radeon integrated, Dawn D3D12 and
Vulkan); results in docs/WEBGPU.md. Nothing is sent anywhere.

    python scripts/webgpu_f16_matmul_repro.py                 # default EP options
    python scripts/webgpu_f16_matmul_repro.py --fp32-acc      # enableMatmulFp32Accumulation=1
    python scripts/webgpu_f16_matmul_repro.py --backend Vulkan --adapter 0

Exit code 1 when any shape is off by more than 1e-2 relative (float16 with a float32 accumulator
stays near 2e-4 on these shapes).
"""

import argparse
import ctypes
import sys

import numpy as np
import onnxruntime as ort
import onnxruntime_ep_webgpu as ep
from onnx import TensorProto, helper

EP = "WebGpuExecutionProvider"
SHAPES = [(1, 4096, 64), (1, 8192, 64), (1, 8193, 64), (1, 12288, 64), (8, 8192, 64),
          (8, 8193, 64), (64, 1024, 64), (64, 4096, 64)]


def model(m, k, n):
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT16, [m, k])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT16, [k, n])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [m, n])
    g = helper.make_graph([helper.make_node("MatMul", ["a", "b"], ["y"])], "mm", [a, b], [y])
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]).SerializeToString()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="D3D12", choices=["D3D12", "Vulkan", "Metal"])
    ap.add_argument("--adapter", type=int, default=0)
    ap.add_argument("--fp32-acc", action="store_true")
    a = ap.parse_args()
    if a.backend == "Vulkan" and sys.platform == "win32":
        ctypes.WinDLL("vulkan-1.dll")  # the EP's Dawn finds no Vulkan adapter without it
    ort.set_default_logger_severity(3)
    ort.register_execution_provider_library(EP, ep.get_library_path())
    dev = [d for d in ort.get_ep_devices() if d.ep_name == EP][:1]  # one device: see WEBGPU.md
    opts = {"dawnBackendType": a.backend, "adapterIndex": str(a.adapter)}
    if a.fp32_acc:
        opts["enableMatmulFp32Accumulation"] = "1"
    rng = np.random.default_rng(0)
    worst = 0.0
    for m, k, n in SHAPES:
        so = ort.SessionOptions()
        so.add_provider_for_devices(dev, opts)
        s = ort.InferenceSession(model(m, k, n), sess_options=so)
        if EP not in s.get_providers():
            print("WebGPU EP not created; ONNX Runtime fell back to", s.get_providers())
            return 2
        x = rng.random((m, k)).astype(np.float16)  # positive, like activations after GELU
        w = rng.random((k, n)).astype(np.float16)
        got = s.run(None, {"a": x, "b": w})[0].astype(np.float64)
        ref = x.astype(np.float64) @ w.astype(np.float64)
        rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        worst = max(worst, rel)
        print(f"M={m:3d} K={k:5d} N={n}: relative error {rel:.2e}"
              + ("   <- float16 accumulation" if rel > 1e-2 else ""))
    print(f"options {opts}: worst {worst:.2e}")
    return 1 if worst > 1e-2 else 0


if __name__ == "__main__":
    sys.exit(main())
