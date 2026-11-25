"""Windows ML worker process: runs ONNX graphs on one Windows ML execution provider.

Same protocol and the same reasons for a separate process as the WebGPU worker
(:mod:`truestar.webgpu.worker`): length prefixed pickles on stdin and stdout, native output sent to a
log file, one EP device per process, and a driver crash ends this process rather than the check.

Where the EP library comes from, in this order (the first that gives a ready library wins, and the
source is reported):

1. the Windows ML EP catalog (`winui3.microsoft.windows.ai.machinelearning`), when the EP is
   installed and ready on this PC;
2. a bring your own EP package (Windows ML documents both routes): `onnxruntime-ep-nv-tensorrt-rtx-*`
   for NvTensorRtRtx, `onnxruntime-ep-webgpu` for WebGPU;
3. DirectML only as the built in `DmlExecutionProvider` of an ONNX Runtime build that has it
   (onnxruntime-directml or onnxruntime-windowsml).

The worker never downloads an EP on its own: the catalog download goes through Windows Update and
installs system wide, so it runs only from :func:`truestar.winml.ensure_ready`.
"""

from __future__ import annotations

import json
import os
import sys
import time

# onnxruntime must load before the WinRT projection: winrt-runtime ships its own msvcp140.dll,
# and with it loaded first onnxruntime's pybind module fails to initialise (measured on Windows 11
# with winrt-runtime 3.2.1 and onnxruntime-gpu 1.29.0: "A dynamic link library (DLL)
# initialization routine failed").
import onnxruntime as ort  # noqa: E402

from ..webgpu.worker import _profile_nodes, _read_msg, _stderr_since, _stderr_size, _write_msg  # noqa: E402
from ..webgpu import worker as _wg  # noqa: E402

# short name: (EP name in the catalog and ONNX Runtime, bring your own module or None)
EPS = {
    "nvtrtx": ("NvTensorRTRTXExecutionProvider", "onnxruntime_ep_nv_tensorrt_rtx"),
    "webgpu": ("WebGpuExecutionProvider", "onnxruntime_ep_webgpu"),
    "dml": ("DmlExecutionProvider", None),
    "openvino": ("OpenVINOExecutionProvider", None),
    "qnn": ("QNNExecutionProvider", None),
    "vitisai": ("VitisAIExecutionProvider", None),
    "amdgpu": ("AMDGPUExecutionProvider", None),
    "migraphx": ("MIGraphXExecutionProvider", None),
}
_STATES = {0: "Ready", 1: "NotReady", 2: "NotPresent"}


def catalog_providers(ensure: str = "", timeout: float = 0.0):
    """[(name, ready state, library path)] from the Windows ML catalog, or (None, reason).

    `ensure`: EP name to make ready first (download through Windows Update when NotPresent)."""
    try:
        from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
            InitializeOptions, initialize)
        import winui3.microsoft.windows.ai.machinelearning as winml
    except Exception as exc:
        return None, (f"the Windows ML projection is not importable ({type(exc).__name__}: {exc}); "
                      "pip install wasdk-Microsoft.Windows.AI.MachineLearning "
                      "wasdk-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap")
    try:
        ctx = initialize(options=InitializeOptions.NONE)  # never shows UI
        ctx.__enter__()
    except Exception as exc:
        return None, f"the Windows App Runtime did not initialise: {type(exc).__name__}: {exc}"
    try:
        cat = winml.ExecutionProviderCatalog.get_default()
        provs = list(cat.find_all_providers())
        report = {}
        for p in provs:
            if ensure and p.name.lower() == ensure.lower():
                t0 = time.time()
                r = p.ensure_ready_async().get()
                report = {"status": int(r.status), "diagnostic": str(r.diagnostic_text or ""),
                          "extended_error": str(r.extended_error), "seconds": round(time.time() - t0, 1)}
        out = [(p.name, _STATES.get(int(p.ready_state), str(p.ready_state)), str(p.library_path or ""))
               for p in provs]
        return out, report
    except Exception as exc:
        return None, f"the Windows ML catalog failed: {type(exc).__name__}: {exc}"


class Worker(_wg.Worker):
    def __init__(self, short: str, device_index: int, ep_options: dict):
        super().__init__("winml", device_index, ep_options)
        self.short = short
        self.ep_name = EPS[short][0]
        self.builtin = False

    def hello(self, _payload):
        import numpy as np

        self.ort = ort
        ort.set_default_logger_severity(3)
        info = {"onnxruntime": ort.__version__, "ep": self.ep_name, "short": self.short,
                "device_index": self.adapter_index, "python": sys.executable,
                "onnxruntime_package": _onnxruntime_package()}
        cat, extra = catalog_providers()
        info["catalog"] = cat
        info["catalog_note"] = extra if isinstance(extra, str) else ""
        lib, source = "", ""
        if cat:
            for name, state, path in cat:
                if name.lower() == self.ep_name.lower() and state == "Ready" and path:
                    lib, source = path, "Windows ML EP catalog"
        mod = EPS[self.short][1]
        if not lib and mod:
            try:
                m = __import__(mod)
                lib, source = m.get_library_path(), f"bring your own EP package {mod}"
                info["ep_version"] = getattr(m, "__version__", "") or _wg._dist_version(mod.replace("_", "-"))
            except Exception as exc:
                info["byo_note"] = f"{mod} not importable ({type(exc).__name__})"
        if not lib and self.short == "dml":
            if "DmlExecutionProvider" in ort.get_available_providers():
                self.builtin, source = True, f"built into {info['onnxruntime_package']}"
        info["source"] = source
        if not lib and not self.builtin:
            states = {n: s for n, s, _ in (cat or [])}
            st = next((s for n, s in states.items() if n.lower() == self.ep_name.lower()), None)
            why = [f"{self.ep_name} is not available here"]
            if cat is None:
                why.append(info["catalog_note"])
            elif st is None and self.short != "dml":
                why.append("the Windows ML catalog does not offer it for this PC's hardware")
            else:
                why.append(f"Windows ML catalog state {st} (truestar.winml.ensure_ready('{self.short}') "
                           "downloads it through Windows Update)")
            if self.short == "dml":
                why.append("DmlExecutionProvider exists only in the onnxruntime-directml and "
                           "onnxruntime-windowsml builds, and this Python has "
                           f"{info['onnxruntime_package']} {ort.__version__}")
            if info.get("byo_note"):
                why.append(info["byo_note"])
            info["unavailable"] = "; ".join(w for w in why if w)
            self.info = info
            return info
        info["library_path"] = lib
        if lib:
            try:
                ort.register_execution_provider_library(self.ep_name, lib)
            except Exception as exc:
                if "already" not in str(exc).lower():
                    info["unavailable"] = f"registering {self.ep_name} from {lib} failed: {exc}"
                    self.info = info
                    return info
            devs = [d for d in ort.get_ep_devices() if d.ep_name == self.ep_name]
            info["ep_devices"] = [{"type": str(d.device.type).split(".")[-1],
                                   "vendor": d.ep_vendor,
                                   "description": dict(d.device.metadata).get("Description", ""),
                                   "ep_metadata": dict(d.ep_metadata)} for d in devs]
            if self.adapter_index >= len(devs):
                info["unavailable"] = (f"{self.ep_name} lists {len(devs)} device(s); index "
                                       f"{self.adapter_index} does not exist")
                self.info = info
                return info
            self.device = devs[self.adapter_index]
            info["device"] = info["ep_devices"][self.adapter_index]
        # Probe: a one node graph must run on the EP (ONNX Runtime falls back to its CPU EP
        # without raising when an EP cannot take a graph).
        from onnx import TensorProto, helper

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 4])
        g = helper.make_graph([helper.make_node("MatMul", ["x", "x"], ["y"], name="probe")], "probe", [x], [y])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 8
        try:
            res = self.run({"model": m.SerializeToString(), "feeds": {"x": np.eye(4, dtype=np.float32)},
                            "provider": "winml", "profile": True})
            if res.get("fallback"):
                info["unavailable"] = (f"{self.ep_name} could not be created and ONNX Runtime fell back "
                                       "to its CPU EP: " + (res.get("ep_error") or "no error text"))
        except Exception as exc:
            info["unavailable"] = f"{self.ep_name} failed on a one node probe: {type(exc).__name__}: {exc}"
        self.info = info
        return info

    def _options(self, provider: str, profile: bool, tag: str):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if profile:
            so.enable_profiling = True
            so.profile_file_prefix = os.path.join(self.tmp, f"prof_{tag}")
        if provider == "winml" and not self.builtin:
            so.add_provider_for_devices([self.device], dict(self.ep_options))
        return so

    def _session(self, model: bytes, provider: str, profile: bool):
        import hashlib

        h = hashlib.sha256(model).hexdigest()[:16]
        key = (h, provider, profile)
        if key in self.sessions and not profile:
            return self.sessions[key], ""
        mark = _stderr_size()
        so = self._options(provider, profile, h)
        if provider == "winml" and self.builtin:
            opts = dict(self.ep_options)
            opts.setdefault("device_id", str(self.adapter_index))
            prov = [("DmlExecutionProvider", opts), "CPUExecutionProvider"]
        elif provider == "winml":
            prov = None
        else:
            prov = ["CPUExecutionProvider"]
        sess = ort.InferenceSession(model, sess_options=so, providers=prov)
        ep_error = _stderr_since(mark)
        if not profile:
            self.sessions[key] = sess
        return sess, ep_error

    def run(self, payload):
        payload = dict(payload, provider=payload.get("provider", "winml"))
        res = super().run(payload)
        # the base class judges fallback by the WebGPU EP name; judge it by this EP
        res["fallback"] = payload.get("provider", "winml") == "winml" and self.ep_name not in res["providers"]
        return res

    def ensure(self, payload):
        """Make an EP ready through the Windows ML catalog (may download through Windows Update)."""
        cat, report = catalog_providers(ensure=self.ep_name)
        return {"catalog": cat, "report": report}


def _onnxruntime_package() -> str:
    for name in ("onnxruntime-windowsml", "onnxruntime-directml", "onnxruntime-gpu", "onnxruntime"):
        if _wg._dist_version(name):
            return name
    return "onnxruntime"


def main() -> None:
    proto_out = os.fdopen(os.dup(1), "wb")
    proto_in = sys.stdin.buffer
    os.dup2(2, 1)
    _wg._LOG = os.environ.get("TRUESTAR_WEBGPU_LOG")
    short = sys.argv[1] if len(sys.argv) > 1 else "nvtrtx"
    index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ep_options = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    w = Worker(short, index, ep_options)
    while True:
        msg = _read_msg(proto_in)
        if msg is None or msg[0] == "close":
            break
        op, payload = msg
        try:
            _write_msg(proto_out, ("ok", getattr(w, op)(payload)))
        except Exception as exc:
            import traceback

            _write_msg(proto_out, ("error", f"{type(exc).__name__}: {exc}\n" + traceback.format_exc(limit=4)))


if __name__ == "__main__":
    main()
