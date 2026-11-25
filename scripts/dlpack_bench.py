"""DLPack import: cost and failure behaviour of truestar.interop.from_dlpack against torch.from_dlpack.

Run: PYTHONPATH=src python scripts/dlpack_bench.py

Each failure case runs in its own process, because stock torch.from_dlpack can kill the process.
Prints one JSON document.
"""

import json
import subprocess
import sys
import time

import numpy as np
import torch

from truestar.interop import from_dlpack

CASES = {
    "negative strides": "np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::-1]",
    "read only": "(lambda a: (setattr(a.flags, 'writeable', False), a)[1])(np.ones((2, 2), np.float32))",
    "big endian": "np.ones(3, dtype='>f4')",
    "cuda to numpy (consumer is numpy)": None,
    "not an array": "[1.0, 2.0]",
}


def run_case(name, expr, which):
    if expr is None:
        code = ("import numpy as np, torch\n"
                "try:\n    np.from_dlpack(torch.ones(3, device='cuda'))\n    print('accepted')\n"
                "except Exception as e:\n    print(type(e).__name__ + ': ' + str(e).splitlines()[0])\n")
        if which == "truestar":
            code = ("import numpy as np, torch\nfrom truestar.interop import from_dlpack\n"
                    "try:\n    from_dlpack(torch.ones(3, device='cuda'), device='cpu')\n    print('accepted')\n"
                    "except Exception as e:\n    print(type(e).__name__ + ': ' + str(e).splitlines()[0])\n")
    else:
        call = "torch.from_dlpack(x)" if which == "torch" else "from_dlpack(x)"
        code = ("import numpy as np, torch\nfrom truestar.interop import from_dlpack\n"
                f"x = {expr}\n"
                f"try:\n    t = {call}\n    print('accepted, tensor writable: ' + str(not t.is_inference()))\n"
                "except Exception as e:\n    print(type(e).__name__ + ': ' + str(e).splitlines()[0])\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        return f"process died (exit code {r.returncode})"
    return r.stdout.strip()[:200]


def timeit(fn, n=2000):
    for _ in range(50):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6


def main():
    doc = {"torch": torch.__version__, "numpy": np.__version__, "overhead_us": {}, "zero_copy": {}, "failures": {}}
    a = np.random.default_rng(0).standard_normal((1024, 1024)).astype(np.float32)
    doc["overhead_us"]["numpy 1024x1024 f32, torch.from_dlpack"] = round(timeit(lambda: torch.from_dlpack(a)), 2)
    doc["overhead_us"]["numpy 1024x1024 f32, truestar from_dlpack"] = round(timeit(lambda: from_dlpack(a)), 2)
    doc["zero_copy"]["numpy"] = from_dlpack(a).data_ptr() == a.ctypes.data
    if torch.cuda.is_available():
        c = torch.randn(1024, 1024, device="cuda")
        doc["overhead_us"]["cuda 1024x1024 f32, torch.from_dlpack"] = round(timeit(lambda: torch.from_dlpack(c)), 2)
        doc["overhead_us"]["cuda 1024x1024 f32, truestar from_dlpack"] = round(timeit(lambda: from_dlpack(c)), 2)

        class Foreign:  # not a torch tensor, so the full DLPack path runs
            def __dlpack_device__(self):
                return c.__dlpack_device__()

            def __dlpack__(self, stream=None, max_version=None):
                return c.__dlpack__(stream=stream, max_version=max_version)

        f = Foreign()
        doc["overhead_us"]["cuda foreign producer, torch.from_dlpack"] = round(timeit(lambda: torch.from_dlpack(f)), 2)
        doc["overhead_us"]["cuda foreign producer, truestar from_dlpack"] = round(timeit(lambda: from_dlpack(f)), 2)
        doc["zero_copy"]["cuda"] = from_dlpack(f).data_ptr() == c.data_ptr()
    for name, expr in CASES.items():
        doc["failures"][name] = {"torch": run_case(name, expr, "torch"), "truestar": run_case(name, expr, "truestar")}
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
