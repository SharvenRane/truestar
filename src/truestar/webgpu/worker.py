"""WebGPU worker process: runs ONNX graphs on ONNX Runtime's WebGPU execution provider.

Runs in its own process, started by :mod:`truestar.webgpu.client`, for three reasons:

* ONNX Runtime's WebGPU EP owns one Dawn device per process ("use a separate process for each
  adapter", onnxruntime-ep-webgpu README), so ``webgpu:d3d12:0`` and ``webgpu:d3d12:1`` in one check
  need two processes;
* a Dawn device loss or driver crash ends this process, not the user's check, and is reported as
  the backend failing;
* the worker needs only numpy and onnxruntime with the WebGPU plugin, not torch, so a different
  Python can host it (``TRUESTAR_WEBGPU_PYTHON``).

Protocol: length prefixed pickles. The parent writes ``(op, payload)`` to stdin, the worker answers
``("ok", result)`` or ``("error", text)`` on its original stdout. Native code (Dawn, the Vulkan
loader, ONNX Runtime) also prints to stdout, so file descriptor 1 is pointed at stderr before
anything native loads; stderr goes to a log file the parent reads for EP error text.

Every run reports where each node ran, read from ONNX Runtime's profile, because ONNX Runtime
silently falls back to its CPU EP when the WebGPU EP fails to initialise (it prints "EP Error ...
Falling back to ['CPUExecutionProvider'] and retrying" and builds the session anyway).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import struct
import sys
import tempfile
import time
import traceback

EP_NAME = "WebGpuExecutionProvider"


def _read_msg(stream):
    head = stream.read(8)
    if len(head) < 8:
        return None
    (n,) = struct.unpack("<Q", head)
    data = stream.read(n)
    return pickle.loads(data)


def _write_msg(stream, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)
    stream.flush()


class Worker:
    def __init__(self, backend_type: str, adapter_index: int, ep_options: dict):
        self.backend_type = backend_type
        self.adapter_index = adapter_index
        self.ep_options = dict(ep_options)
        self.ort = None
        self.device = None
        self.sessions = {}
        self.tmp = tempfile.mkdtemp(prefix="truestar_webgpu_")
        self.info = {}

    # ------------------------------------------------------------------ setup

    def _adapters(self):
        """Dawn's adapter list for the requested backend, from wgpudawn when it is installed.

        ONNX Runtime does not expose the adapter it picked; its log says "selected physical adapter
        index i of n". wgpudawn enumerates with the same Dawn, so its list for this backend type
        names adapter i. When wgpudawn is missing the adapter is reported as unknown."""
        try:
            import wgpudawn as wgpu  # noqa: F401
        except Exception as exc:
            return None, f"wgpudawn not importable ({type(exc).__name__}); adapter not named"
        try:
            found = []
            for a in wgpu.gpu.enumerate_adapters_sync():
                info = a.info
                if str(info.get("backend_type", "")).lower() != self.backend_type.lower():
                    continue
                found.append({
                    "device": info.get("device", ""),
                    "vendor": info.get("vendor", ""),
                    "adapter_type": info.get("adapter_type", ""),
                    "backend_type": info.get("backend_type", ""),
                    "architecture": info.get("architecture", ""),
                    "driver": info.get("description", ""),
                    "shader_f16_in_wgpudawn": "shader-f16" in a.features,
                })
            return found, ""
        except Exception as exc:
            return None, f"adapter enumeration failed: {type(exc).__name__}: {exc}"

    def hello(self, _payload):
        preload = ""
        if self.backend_type == "Vulkan" and sys.platform == "win32":
            # Measured with onnxruntime-ep-webgpu 0.6.0 on Windows 11: Dawn inside the EP finds
            # no Vulkan adapter ("Failed to get a WebGPU adapter: No supported adapters") unless
            # vulkan-1.dll is already loaded in the process; with it loaded, both Vulkan adapters
            # appear. ONNX Runtime then falls back to its CPU EP without raising.
            try:
                import ctypes

                ctypes.WinDLL("vulkan-1.dll")
                preload = "vulkan-1.dll loaded before the EP (its Dawn finds no Vulkan adapter otherwise)"
            except OSError as exc:
                preload = f"vulkan-1.dll could not be loaded: {exc}"
        adapters, adapter_note = self._adapters()
        import numpy as np  # noqa: F401
        import onnxruntime as ort

        self.ort = ort
        ort.set_default_logger_severity(3)
        info = {"onnxruntime": ort.__version__, "backend_type": self.backend_type,
                "adapter_index": self.adapter_index, "adapters": adapters,
                "adapter_note": adapter_note, "python": sys.executable, "preload": preload}
        try:
            import onnxruntime_ep_webgpu as ep
        except Exception as exc:
            info["unavailable"] = (f"onnxruntime_ep_webgpu is not importable in {sys.executable} "
                                   f"({type(exc).__name__}: {exc}); install onnxruntime-ep-webgpu "
                                   "or point TRUESTAR_WEBGPU_PYTHON at a Python that has it")
            self.info = info
            return info
        info["ep_version"] = getattr(ep, "__version__", "") or _dist_version("onnxruntime-ep-webgpu")
        try:
            ort.register_execution_provider_library(EP_NAME, ep.get_library_path())
        except Exception as exc:
            info["unavailable"] = f"registering the WebGPU EP library failed: {exc}"
            self.info = info
            return info
        devs = [d for d in ort.get_ep_devices() if d.ep_name == EP_NAME]
        info["ep_devices"] = len(devs)
        if not devs:
            info["unavailable"] = "the WebGPU EP loaded but listed no device"
            self.info = info
            return info
        # The EP lists the same WebGPU device more than once on some machines (twice here). Passing
        # all of them makes session creation fail, after which ONNX Runtime silently runs the
        # graph on its CPU EP. The README says the EP accepts one device; pass exactly one.
        self.device = devs[0]
        if adapters is not None:
            if self.adapter_index < len(adapters):
                info["adapter"] = adapters[self.adapter_index]
            else:
                info["adapter"] = None
        # Probe: a one node graph must run on the WebGPU EP.
        from onnx import TensorProto, helper

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
        g = helper.make_graph([helper.make_node("Relu", ["x"], ["y"], name="probe")], "probe", [x], [y])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        res = self.run({"model": m.SerializeToString(), "feeds": {"x": np.ones(4, np.float32)},
                        "provider": "webgpu", "profile": True})
        if res.get("fallback"):
            info["unavailable"] = ("the WebGPU EP could not be created, and ONNX Runtime fell back "
                                   "to its CPU EP: " + (res.get("ep_error") or "no error text"))
        self.info = info
        return info

    # ------------------------------------------------------------------ sessions

    def _options(self, provider: str, profile: bool, tag: str):
        ort = self.ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if profile:
            so.enable_profiling = True
            so.profile_file_prefix = os.path.join(self.tmp, f"prof_{tag}")
        if provider == "webgpu":
            opts = {"dawnBackendType": self.backend_type, "adapterIndex": str(self.adapter_index)}
            opts.update(self.ep_options)
            so.add_provider_for_devices([self.device], opts)
        return so

    def _session(self, model: bytes, provider: str, profile: bool):
        h = hashlib.sha256(model).hexdigest()[:16]
        key = (h, provider, profile)
        if key in self.sessions and not profile:
            return self.sessions[key], ""
        mark = _stderr_size()
        so = self._options(provider, profile, h)
        prov = None if provider == "webgpu" else ["CPUExecutionProvider"]
        sess = self.ort.InferenceSession(model, sess_options=so, providers=prov)
        ep_error = _stderr_since(mark)
        if not profile:
            self.sessions[key] = sess
        return sess, ep_error

    def run(self, payload):
        """Run one graph. payload: model (bytes), feeds (name -> ndarray), provider ("webgpu" or
        "cpu"), profile (record where each node ran), repeat (extra runs to measure run to run
        determinism), outputs (names or None)."""
        import numpy as np

        provider = payload.get("provider", "webgpu")
        profile = bool(payload.get("profile", False))
        sess, ep_error = self._session(payload["model"], provider, profile)
        providers = list(sess.get_providers())
        names = payload.get("outputs")
        t0 = time.perf_counter()
        outs = sess.run(names, payload["feeds"])
        elapsed = time.perf_counter() - t0
        repeat_equal = None
        for _ in range(int(payload.get("repeat", 0))):
            again = sess.run(names, payload["feeds"])
            same = all(np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc"
                       else np.array_equal(a, b) for a, b in zip(outs, again))
            repeat_equal = same if repeat_equal is None else (repeat_equal and same)
        nodes = None
        if profile:
            path = sess.end_profiling()
            nodes = _profile_nodes(path)
            try:
                os.remove(path)
            except OSError:
                pass
        out_names = [o.name for o in sess.get_outputs()] if names is None else list(names)
        return {
            "outputs": [np.asarray(o) for o in outs],
            "output_names": out_names,
            "providers": providers,
            "fallback": provider == "webgpu" and EP_NAME not in providers,
            "ep_error": ep_error,
            "nodes": nodes,
            "seconds": elapsed,
            "repeat_equal": repeat_equal,
        }


def _profile_nodes(path: str):
    """[(node name, op type, provider)] for every node the profile saw run, in order."""
    with open(path, "r", encoding="utf-8") as f:
        events = json.load(f)
    seen, out = set(), []
    for e in events:
        if e.get("cat") != "Node":
            continue
        name = e.get("name", "")
        if not name.endswith("_kernel_time"):
            continue
        node = name[: -len("_kernel_time")]
        args = e.get("args", {})
        if node in seen:
            continue
        seen.add(node)
        out.append((node, args.get("op_name", ""), args.get("provider", "")))
    return out


def _dist_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return ""


_LOG = None


def _stderr_size() -> int:
    try:
        sys.stderr.flush()
        return os.fstat(2).st_size
    except OSError:
        return 0


def _stderr_since(mark: int) -> str:
    """Text native code wrote to stderr (the log file) since mark: ONNX Runtime's EP errors."""
    if _LOG is None:
        return ""
    try:
        sys.stderr.flush()
        with open(_LOG, "rb") as f:
            f.seek(mark)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    keep = [ln for ln in text.splitlines() if "EP Error" in ln or "Falling back" in ln
            or "Failed" in ln or "error" in ln.lower()]
    return "\n".join(keep)[-4000:]


def main() -> None:
    global _LOG
    proto_out = os.fdopen(os.dup(1), "wb")
    proto_in = sys.stdin.buffer
    os.dup2(2, 1)  # native prints to stdout now land in the stderr log, never in the protocol
    _LOG = os.environ.get("TRUESTAR_WEBGPU_LOG")
    backend_type = sys.argv[1] if len(sys.argv) > 1 else "D3D12"
    adapter_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ep_options = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    w = Worker(backend_type, adapter_index, ep_options)
    while True:
        msg = _read_msg(proto_in)
        if msg is None or msg[0] == "close":
            break
        op, payload = msg
        try:
            result = getattr(w, op)(payload)
            _write_msg(proto_out, ("ok", result))
        except Exception as exc:  # reported to the parent, never swallowed
            _write_msg(proto_out, ("error", f"{type(exc).__name__}: {exc}\n"
                                            + traceback.format_exc(limit=4)))


if __name__ == "__main__":
    main()
