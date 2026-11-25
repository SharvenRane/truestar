"""Runtimes that run a deployed artifact in its own engine, with residency and fingerprints.

Names are `<format>:<target>`:

| runtime | what runs | residency read from |
|---|---|---|
| ort:cpu | ONNX Runtime, CPU provider | all nodes on the CPU by construction |
| ort:cuda | ONNX Runtime, providers [CUDA, CPU] as users write them | one profiled run (provider per node) |
| ort:webgpu | ONNX Runtime with the WebGPU plugin provider (onnxruntime-ep-webgpu), CPU allowed | one profiled run |
| tflite:interp | LiteRT `Interpreter(model)`: XNNPACK delegate applied by default | the delegate partitions in the execution plan |
| tflite:ref | LiteRT `Interpreter` without default delegates (builtin kernels) | all on builtin kernels |
| tflite:cpu | LiteRT `CompiledModel`, CPU accelerator | `is_fully_accelerated`, then one cut per operator |
| tflite:gpu | LiteRT `CompiledModel`, GPU accelerator, its default precision | as tflite:cpu |
| tflite:gpu-f32 | as tflite:gpu with `enforce_f32` | as tflite:cpu |
| tflite:npu | LiteRT `CompiledModel`, NPU with CPU fallback (as the LiteRT docs write it) | LiteRT's own log (compiler plugin, dispatch library, partitions), then one compile per operator; DISPATCH_OP partitions of an AOT file |
| pte:cpu | ExecuTorch runtime (`executorch.runtime`) | kernel calls and delegate calls in the program |
| tsx:reference | Truestar's IR reference interpreter (float64, cast to the declared dtypes) | trivially resident |
| tsx:cpu, tsx:cuda | Truestar's compiled executor | trivially resident |

A runtime that cannot run here is refused with the reason, never replaced by another one.
"""

from __future__ import annotations

import contextlib
import ctypes
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .artifact import ONNX, PTE, TFLITE, TSX, Artifact

RUNTIMES: Dict[str, Tuple[str, str]] = {
    "ort:cpu": (ONNX, "cpu"),
    "ort:cuda": (ONNX, "cuda"),
    "ort:webgpu": (ONNX, "webgpu"),
    "tflite:interp": (TFLITE, "interp"),
    "tflite:ref": (TFLITE, "ref"),
    "tflite:cpu": (TFLITE, "cpu"),
    "tflite:gpu": (TFLITE, "gpu"),
    "tflite:gpu-f32": (TFLITE, "gpu-f32"),
    "tflite:npu": (TFLITE, "npu"),
    "pte:cpu": (PTE, "cpu"),
    "tsx:reference": (TSX, "reference"),
    "tsx:cpu": (TSX, "cpu"),
    "tsx:cuda": (TSX, "cuda"),
}

# Which runtimes run by default for each format (the ones a user of that format reaches first).
DEFAULTS = {ONNX: ["ort:cpu", "ort:cuda", "ort:webgpu"],
            TFLITE: ["tflite:interp", "tflite:cpu", "tflite:gpu"],
            PTE: ["pte:cpu"], TSX: ["tsx:reference", "tsx:cuda"]}


class RuntimeUnavailable(RuntimeError):
    pass


class RuntimeFailed(RuntimeError):
    pass


@dataclass
class Placement:
    node: str
    op: str
    module: str
    where: str


@dataclass
class Residency:
    requested: str  # the device the user asked for: cpu, gpu, ...
    known: bool  # placement was read, not assumed
    method: str
    nodes: int = 0
    off_target: List[Placement] = field(default_factory=list)  # ran somewhere else
    notes: List[str] = field(default_factory=list)
    instrumented: bool = False  # reading it needed an instrumented run (profiling)
    findings: List[Tuple[str, str]] = field(default_factory=list)  # (code, message) for the report

    @property
    def resident(self) -> bool:
        return self.known and not self.off_target

    def as_dict(self) -> Dict[str, Any]:
        return {"requested": self.requested, "known": self.known, "method": self.method,
                "nodes": self.nodes, "resident": self.resident, "instrumented": self.instrumented,
                "off_target": [vars(p) for p in self.off_target], "notes": list(self.notes)}


def _dist(name: str) -> Optional[str]:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _file_sha(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------- sessions

class Session:
    """One loaded artifact (or cut) in one runtime. run() takes and returns numpy arrays in the
    artifact's input and output order."""

    def __init__(self, run: Callable[[List[np.ndarray]], List[np.ndarray]], info: Dict[str, Any],
                 close: Optional[Callable[[], None]] = None):
        self._run = run
        self.info = info
        self._close = close

    def run(self, feeds: List[np.ndarray]) -> List[np.ndarray]:
        return [np.array(o, copy=True) for o in self._run(feeds)]

    def close(self) -> None:
        if self._close:
            self._close()


class Runtime:
    name = ""
    format = ""
    target = ""
    requested = "cpu"

    def __init__(self, name: str, options: Optional[Dict[str, Any]] = None):
        self.name = name
        self.format, self.target = RUNTIMES[name]
        self.options = dict(options or {})

    def unavailable(self) -> Optional[str]:
        return None

    def open(self, data: bytes, artifact: Artifact, *, profile: bool = False) -> Session:
        raise NotImplementedError

    def residency(self, artifact: Artifact, session: Session) -> Residency:
        return Residency(self.requested, False, "not exposed by this runtime")

    def fingerprint(self) -> Dict[str, Any]:
        from ..recdiff.fingerprint import runtime_fingerprint

        return runtime_fingerprint(self.name, self._versions())

    def _versions(self) -> Dict[str, Any]:
        return {}


# ----------------------------------------------------------------------------- ONNX Runtime

_FLOAT_TYPES = ("float", "double", "float16", "bfloat16")


class OrtRuntime(Runtime):
    def __init__(self, name, options=None):
        super().__init__(name, options)
        self.requested = {"cpu": "cpu", "cuda": "cuda", "webgpu": "gpu"}[self.target]

    def unavailable(self):
        try:
            import onnxruntime as ort
        except Exception as exc:
            return f"onnxruntime is not installed ({type(exc).__name__})"
        if self.target == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
            return (f"onnxruntime {ort.__version__} here has no CUDA provider (installed providers: "
                    f"{', '.join(ort.get_available_providers())}); install onnxruntime-gpu")
        if self.target == "webgpu":
            try:
                self._webgpu_device()
            except RuntimeUnavailable as exc:
                return str(exc)
        return None

    def _webgpu_device(self):
        import onnxruntime as ort

        try:
            import onnxruntime_ep_webgpu as W
        except Exception:
            raise RuntimeUnavailable("the WebGPU plugin provider is not installed "
                                     "(pip install onnxruntime-ep-webgpu, needs onnxruntime 1.24.4 or later)")
        if not hasattr(ort, "register_execution_provider_library"):
            raise RuntimeUnavailable(f"onnxruntime {ort.__version__} cannot load plugin providers")
        try:
            if not getattr(OrtRuntime, "_webgpu_registered", False):
                ort.register_execution_provider_library("webgpu", W.get_library_path())
                OrtRuntime._webgpu_registered = True
        except Exception as exc:
            if "already" not in str(exc).lower():
                raise RuntimeUnavailable(f"the WebGPU plugin provider did not register: {exc}")
            OrtRuntime._webgpu_registered = True
        devs = [d for d in ort.get_ep_devices() if d.ep_name == W.get_ep_name()]
        if not devs:
            raise RuntimeUnavailable("the WebGPU plugin provider found no GPU adapter")
        want = self.options.get("adapter")
        if want is not None:
            devs = [d for d in devs if want.lower() in str(d.device.metadata.get("Description", "")).lower()]
            if not devs:
                raise RuntimeUnavailable(f"no WebGPU adapter matches {want!r}")
        # first adapter in the provider's own order (high performance first), as users get it
        return devs[0]

    def open(self, data, artifact, *, profile=False):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        for k, v in self.options.get("session_config", {}).items():
            so.add_session_config_entry(k, v)
        prof_dir = None
        if profile:
            prof_dir = tempfile.mkdtemp(prefix="truestar_conform_")
            so.enable_profiling = True
            so.profile_file_prefix = os.path.join(prof_dir, "profile")
        info: Dict[str, Any] = {}
        try:
            if self.target == "webgpu":
                dev = self._webgpu_device()
                so.add_provider_for_devices([dev], {})
                info["adapter"] = dev.device.metadata.get("Description", "")
                sess = ort.InferenceSession(data, so)
            else:
                prov = (["CUDAExecutionProvider", "CPUExecutionProvider"] if self.target == "cuda"
                        else ["CPUExecutionProvider"])
                sess = ort.InferenceSession(data, so, providers=prov)
        except RuntimeUnavailable:
            raise
        except Exception as exc:
            raise RuntimeFailed(f"ONNX Runtime could not load the model: {type(exc).__name__}: {exc}") from exc
        info["providers"] = sess.get_providers()
        if self.target == "cuda" and "CUDAExecutionProvider" in info["providers"]:
            opts = sess.get_provider_options().get("CUDAExecutionProvider", {})
            info["tf32"] = str(opts.get("use_tf32", "0")) == "1"
        names = [i.name for i in sess.get_inputs()]
        outs = [o.name for o in sess.get_outputs()]
        state = {"profile": None}

        def run(feeds):
            res = sess.run(outs, {n: np.ascontiguousarray(f) for n, f in zip(names, feeds)})
            if prof_dir is not None and state["profile"] is None:
                state["profile"] = sess.end_profiling()
            return res

        def close():
            if prof_dir:
                shutil.rmtree(prof_dir, ignore_errors=True)

        s = Session(run, info, close)
        s.state = state
        s.ort_session = sess
        return s

    def residency(self, artifact, session):
        provs = session.info.get("providers", [])
        n = artifact.node_count()
        if self.target == "cpu":
            return Residency("cpu", True, "CPU provider only", n)
        want = "CUDAExecutionProvider" if self.target == "cuda" else "WebGpuExecutionProvider"
        if want not in provs:
            r = Residency(self.requested, True, "session providers", n, notes=[
                f"{want} did not load (session providers: {', '.join(provs)}): every node ran on the CPU"])
            r.off_target = [Placement("(all)", "(all)", "", "CPUExecutionProvider")]
            return r
        path = session.state.get("profile")
        r = Residency(self.requested, path is not None, "one profiled run (provider per node)", n,
                      instrumented=True)
        if path is None:
            r.notes.append("the profile was not written; placement unknown")
            return r
        try:
            with open(path, encoding="utf-8") as f:
                events = json.load(f)
        except Exception as exc:
            r.known = False
            r.notes.append(f"profile unreadable: {exc}")
            return r
        by_name = {}
        for i in range(n):
            ni = artifact.node_info(i)
            by_name[ni["node"]] = ni
        seen = {}
        for e in events:
            if e.get("cat") != "Node" or not str(e.get("name", "")).endswith("_kernel_time"):
                continue
            a = e.get("args", {})
            prov = a.get("provider", "")
            if prov == want or str(a.get("op_name", "")).startswith("Memcpy"):
                continue
            floating = any(k in _FLOAT_TYPES for o in a.get("output_type_shape", []) for k in o)
            if not floating:
                continue  # integer shape arithmetic ORT keeps on the CPU on purpose
            node = e["name"][: -len("_kernel_time")]
            meta = by_name.get(node, {"op": a.get("op_name", "?"), "module": ""})
            seen[node] = Placement(node, a.get("op_name", meta["op"]), meta.get("module", ""), prov)
        r.off_target = list(seen.values())
        return r

    def _versions(self):
        import onnxruntime as ort

        v = {"onnxruntime": ort.__version__,
             "onnxruntime.package": _dist("onnxruntime-gpu") and "onnxruntime-gpu" or "onnxruntime"}
        if self.target == "webgpu":
            v["onnxruntime-ep-webgpu"] = _dist("onnxruntime-ep-webgpu")
            with contextlib.suppress(Exception):
                v["adapter"] = self._webgpu_device().device.metadata.get("Description", "")
        if self.target == "cuda":
            with contextlib.suppress(Exception):
                import onnxruntime.capi._pybind_state as C  # build info: CUDA version it was built for

                v["onnxruntime.build"] = C.get_build_info() if hasattr(C, "get_build_info") else None
        return v


# ----------------------------------------------------------------------------- LiteRT (TFLite)

_DXC_STATE: Dict[str, Any] = {}


def find_dxc() -> Optional[str]:
    """dxcompiler.dll for LiteRT's WebGPU accelerator on Windows (it fails with Windows error 87
    without one). Looked up in TRUESTAR_DXC_DIR, then next to the ONNX Runtime WebGPU plugin, then
    on PATH."""
    cands = []
    if os.environ.get("TRUESTAR_DXC_DIR"):
        cands.append(os.environ["TRUESTAR_DXC_DIR"])
    with contextlib.suppress(Exception):
        import importlib.util

        spec = importlib.util.find_spec("onnxruntime_ep_webgpu")
        if spec and spec.origin:
            cands.append(os.path.dirname(spec.origin))
    cands += os.environ.get("PATH", "").split(os.pathsep)
    for d in cands:
        p = os.path.join(d, "dxcompiler.dll")
        if d and os.path.isfile(p):
            return p
    return None


def _preload_dxc() -> Optional[str]:
    if platform.system() != "Windows":
        return None
    if "path" in _DXC_STATE:
        return _DXC_STATE["path"]
    p = find_dxc()
    if p:
        with contextlib.suppress(Exception):
            os.add_dll_directory(os.path.dirname(p))
        try:
            ctypes.WinDLL(p)
        except OSError:
            p = None
    _DXC_STATE["path"] = p
    return p


@contextlib.contextmanager
def _quiet_fd2(into: Optional[str] = None):
    """LiteRT logs to the C stderr; keep it out of the JSON on stdout and the user's terminal.
    `into`: a file path that receives the log instead of the null device (the NPU residency is
    read from it)."""
    try:
        fd = sys.stderr.fileno()
    except Exception:
        yield
        return
    saved = os.dup(fd)
    win = platform.system() == "Windows"
    with open(into or os.devnull, "a" if into else "w") as null:
        os.dup2(null.fileno(), fd)
        old_handle = None
        if win:  # native loggers write to the process stderr handle, not to fd 2
            import msvcrt

            k32 = ctypes.windll.kernel32
            old_handle = k32.GetStdHandle(-12)
            k32.SetStdHandle(-12, msvcrt.get_osfhandle(null.fileno()))
        try:
            yield
        finally:
            if win:
                ctypes.windll.kernel32.SetStdHandle(-12, old_handle)
            os.dup2(saved, fd)
            os.close(saved)


def _log_path() -> str:
    fd, p = tempfile.mkstemp(prefix="truestar_litert_", suffix=".log")
    os.close(fd)
    return p


def _read_log(path: str) -> str:
    try:
        with open(path, "rb") as f:
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    with contextlib.suppress(OSError):
        os.remove(path)
    return text


DISPATCH_OP = "CUSTOM:DISPATCH_OP"  # the custom op LiteRT puts NPU bytecode behind (libLiteRt)
_PARTITIONED = re.compile(r"Partitioned subgraph<(\d+)>, selected (\d+) ops, from a total of (\d+) ops")


def npu_log_facts(env_log: str, model_log: str) -> Dict[str, Any]:
    """What LiteRT's own log says about the NPU path, from the environment creation and one
    CompiledModel creation. The LiteRT Intel page: is_fully_accelerated() alone is not sufficient,
    "if the dispatch library never loaded, ops were fully offloaded to XNNPACK/CPU, not the NPU"."""
    text = env_log + "\n" + model_log
    facts: Dict[str, Any] = {"npu_registered": None, "plugin_loaded": False, "plugin_error": "",
                             "partitions": [], "dispatch_error": ""}
    if "NPU accelerator could not be loaded" in text:
        facts["npu_registered"] = False
        m = re.search(r"NPU accelerator could not be loaded and registered: ([^\r\n]*)", text)
        facts["npu_error"] = m.group(1).strip().rstrip(".") if m else ""
    elif re.search(r"RegisterAccelerator: ptr=\S+, name=\S*NPU", text, re.I):
        facts["npu_registered"] = True
    if re.search(r"Loaded plugin at", text):
        facts["plugin_loaded"] = True
    m = re.search(r"Failed to load plugin at: ([^\r\n]*?) with error", text)
    if m:
        facts["plugin_error"] = f"compiler plugin {os.path.basename(m.group(1).strip())} did not load"
        mm = re.search(r"Failed to load library '[^']*': ([^\r\n]*)", text)
        if mm:
            facts["plugin_error"] += f" ({mm.group(1).strip().rstrip('.')}: a library it links against, "\
                                     "for example openvino.dll for the Intel plugin, is missing)"
    if "No compiler plugin found" in text:
        facts.setdefault("plugin_error", "")
        facts["no_plugin"] = True
    facts["partitions"] = [(int(a), int(b), int(c)) for a, b, c in _PARTITIONED.findall(text)]
    m = re.search(r"([^\r\n]*Dispatch[^\r\n]*(?:fail|error|not found)[^\r\n]*)", text, re.I)
    if m:
        facts["dispatch_error"] = m.group(1).strip()[:300]
    return facts


_LITERT_ENV: Dict[str, Any] = {}


def _litert_env():
    """One LiteRT environment for every CompiledModel: each new environment loads and logs every
    accelerator again, and destroys them again when collected."""
    if "env" not in _LITERT_ENV:
        from ai_edge_litert.environment import Environment, EnvironmentOptions

        import ai_edge_litert

        log = _log_path()
        with _quiet_fd2(log):
            _LITERT_ENV["env"] = Environment.create(options=EnvironmentOptions(
                runtime_path=os.path.dirname(os.path.abspath(ai_edge_litert.__file__))))
        _LITERT_ENV["log"] = _read_log(log)
        import atexit

        def drop():
            env = _LITERT_ENV.pop("env", None)
            _release(env)

        atexit.register(drop)
    return _LITERT_ENV["env"]


def _release(*objs) -> None:
    import gc

    with _quiet_fd2():
        for o in objs:
            del o
        gc.collect()


class TfliteRuntime(Runtime):
    def __init__(self, name, options=None):
        super().__init__(name, options)
        self.requested = ("gpu" if self.target.startswith("gpu") else
                          "npu" if self.target == "npu" else "cpu")

    def unavailable(self):
        try:
            import ai_edge_litert  # noqa: F401
        except Exception as exc:
            return f"ai-edge-litert is not installed ({type(exc).__name__})"
        if self.target == "npu":
            from ai_edge_litert.hardware_accelerator import HardwareAccelerator as HA

            if not hasattr(HA, "NPU"):
                return "this ai-edge-litert has no NPU accelerator option (LiteRT 2.x CompiledModel needed)"
        if self.target.startswith("gpu"):
            if platform.system() == "Windows" and _preload_dxc() is None:
                return ("LiteRT's GPU accelerator (WebGPU on Direct3D 12) needs dxcompiler.dll and none "
                        "was found (set TRUESTAR_DXC_DIR, or install onnxruntime-ep-webgpu which ships one)")
        return None

    def _interp(self, data, preserve=False):
        from ai_edge_litert import interpreter as I

        kw = {}
        if self.target == "ref":
            kw["experimental_op_resolver_type"] = I.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        with _quiet_fd2():
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                it = I.Interpreter(model_content=data, experimental_preserve_all_tensors=preserve, **kw)
                it.allocate_tensors()
        return it

    def open(self, data, artifact, *, profile=False, preserve=False):
        if self.target in ("interp", "ref"):
            try:
                it = self._interp(data, preserve)
            except Exception as exc:
                raise RuntimeFailed(f"LiteRT Interpreter could not load the model: {exc}") from exc
            ins = [d["index"] for d in it.get_input_details()]
            outs = [d["index"] for d in it.get_output_details()]

            def run(feeds):
                for i, f in zip(ins, feeds):
                    it.set_tensor(i, np.ascontiguousarray(f))
                it.invoke()
                return [it.get_tensor(o) for o in outs]

            s = Session(run, {"engine": "Interpreter", "preserve_all_tensors": preserve},
                        lambda: _release())
            s.interpreter = it
            return s
        from ai_edge_litert.compiled_model import CompiledModel
        from ai_edge_litert.gpu_options import GpuOptions
        from ai_edge_litert.hardware_accelerator import HardwareAccelerator as HA
        from ai_edge_litert.options import Options

        if self.target == "cpu":
            opts = Options(HA.CPU)
        elif self.target == "npu":
            opts = Options(HA.NPU | HA.CPU)  # NPU with CPU fallback, as LiteRT's NPU docs write it
        else:
            _preload_dxc()
            opts = Options(HA.GPU, gpu_options=GpuOptions(enforce_f32=self.target == "gpu-f32"))
        model_log = _log_path()
        try:
            env = _litert_env()
            with _quiet_fd2(model_log):
                cm = CompiledModel.from_buffer(data, options=opts, environment=env)
                full = bool(cm.is_fully_accelerated())
        except Exception as exc:
            raise RuntimeFailed(f"LiteRT CompiledModel ({self.target}) could not compile the model: "
                                f"{str(exc).encode('ascii', 'replace').decode()[:300]}") from exc
        ib = cm.create_input_buffers(0)
        ob = cm.create_output_buffers(0)
        g = getattr(artifact, "graph", None)
        if g is not None and data == artifact.data:
            out_specs = [(g.shape(i), g.dtype(i)) for i in g.outputs]
        else:
            from .tflite_graph import TfliteGraph

            gg = TfliteGraph(data)
            out_specs = [(gg.shape(i), gg.dtype(i)) for i in gg.outputs]

        def run(feeds):
            ib, ob, cm = state["ib"], state["ob"], state["cm"]
            for b, f in zip(ib, feeds):
                b.write(np.ascontiguousarray(f))
            with _quiet_fd2():
                cm.run_by_index(0, ib, ob)
            return [b.read(int(np.prod(shp)) if shp else 1, dt).reshape(shp) for b, (shp, dt) in zip(ob, out_specs)]

        state = {"cm": cm, "ib": ib, "ob": ob}

        def close():
            objs = [state.pop(k) for k in list(state)]
            _release(*objs)

        info = {"engine": "CompiledModel", "fully_accelerated": full}
        mlog = _read_log(model_log)
        if self.target == "npu":
            info["npu"] = npu_log_facts(_LITERT_ENV.get("log", ""), mlog)
        s = Session(run, info, close)
        return s

    def npu_selected(self, data: bytes) -> Optional[Tuple[int, int]]:
        """(ops LiteRT's compiler plugin put on the NPU, ops in the graph) for one model, from its
        log; None when no plugin partitioned anything."""
        from ai_edge_litert.compiled_model import CompiledModel
        from ai_edge_litert.hardware_accelerator import HardwareAccelerator as HA
        from ai_edge_litert.options import Options

        log = _log_path()
        try:
            env = _litert_env()
            with _quiet_fd2(log):
                cm = CompiledModel.from_buffer(data, options=Options(HA.NPU | HA.CPU), environment=env)
            _release(cm)
        except Exception:
            _read_log(log)
            return None
        parts = npu_log_facts("", _read_log(log))["partitions"]
        if not parts:
            return None
        return sum(p[1] for p in parts), sum(p[2] for p in parts)

    def fully_accelerated(self, data: bytes) -> Optional[bool]:
        from ai_edge_litert.compiled_model import CompiledModel
        from ai_edge_litert.gpu_options import GpuOptions
        from ai_edge_litert.hardware_accelerator import HardwareAccelerator as HA
        from ai_edge_litert.options import Options

        opts = (Options(HA.CPU) if self.target == "cpu" else
                Options(HA.GPU, gpu_options=GpuOptions(enforce_f32=self.target == "gpu-f32")))
        try:
            env = _litert_env()
            with _quiet_fd2():
                cm = CompiledModel.from_buffer(data, options=opts, environment=env)
                full = bool(cm.is_fully_accelerated())
            _release(cm)
            return full
        except Exception:
            return None

    def residency(self, artifact, session):
        g = artifact.graph
        n = len(g.ops)
        if self.target == "ref":
            return Residency("cpu", True, "builtin kernels only (no delegate)", n)
        if self.target == "interp":
            it = session.interpreter
            details = it._get_ops_details()
            delegates = [d for d in details if d.get("op_name") == "DELEGATE"]
            covered = set()
            prod = g.producers()
            for d in delegates:
                stop = {int(t) for t in d["inputs"]}
                stack = [int(t) for t in d["outputs"]]
                while stack:
                    t = stack.pop()
                    if t in stop or t not in prod or prod[t] in covered:
                        continue
                    covered.add(prod[t])
                    stack.extend(x for x in g.ops[prod[t]].inputs if x >= 0)
            r = Residency("cpu", True, f"XNNPACK delegate partitions ({len(delegates)}) in the execution plan", n)
            for op in g.ops:
                if op.index not in covered:
                    info = artifact.node_info(op.index)
                    r.off_target.append(Placement(info["node"], op.name, info["module"], "builtin kernel"))
            if not delegates:
                r.notes.append("no delegate applied: the whole model ran on builtin kernels")
            return r
        if self.target == "npu":
            return self._npu_residency(artifact, session)
        full = session.info.get("fully_accelerated")
        r = Residency(self.requested, True, "is_fully_accelerated, then one compile per operator", n)
        if full:
            return r
        # The runtime says only "not fully accelerated". Compile every operator alone to name them.
        for op in g.ops:
            fa = self.fully_accelerated(artifact.single_op(op.index))
            if fa is False or fa is None:
                info = artifact.node_info(op.index)
                r.off_target.append(Placement(info["node"], op.name, info["module"],
                                              "cpu fallback" if fa is False else "could not compile alone"))
        if not r.off_target:
            r.notes.append("not fully accelerated, but every operator alone is: the fallback comes "
                           "from how operators were grouped")
        return r

    def _npu_residency(self, artifact, session) -> Residency:
        """Where an NPU request really ran. Three cases:

        * AOT file (DISPATCH_OP custom ops): the operators outside the dispatch partitions are read
          from the file (TS1126); if the dispatch path did not load, everything ran on the CPU;
        * a plain file that LiteRT's compiler plugin partitioned: the operators it did not select
          are found by compiling each operator alone (TS1124);
        * no plugin and no dispatch library: LiteRT ran the whole model on XNNPACK while
          is_fully_accelerated() said True (TS1125)."""
        g = artifact.graph
        n = len(g.ops)
        facts = session.info.get("npu") or {}
        full = session.info.get("fully_accelerated")
        r = Residency("npu", True, "LiteRT log (NPU accelerator, compiler plugin, partitions), "
                                   "DISPATCH_OP partitions of the file, one compile per operator", n)
        r.notes.append(f"is_fully_accelerated() = {full}")
        dispatch = [op for op in g.ops if op.name == DISPATCH_OP]
        why = "; ".join(x for x in (
            ("NPU accelerator not registered" + (f": {facts.get('npu_error')}" if facts.get("npu_error") else ""))
            if facts.get("npu_registered") is False else "",
            facts.get("plugin_error", ""), facts.get("dispatch_error", "")) if x)

        def everything(where):
            for op in g.ops:
                info = artifact.node_info(op.index)
                r.off_target.append(Placement(info["node"], op.name, info["module"], where))

        if dispatch:
            outside = [op for op in g.ops if op.name != DISPATCH_OP]
            r.findings.append(("TS1126", f"ahead of time compiled file: {len(dispatch)} NPU dispatch "
                               f"partition(s), {len(outside)} operator(s) outside them that run on the "
                               "CPU or GPU fallback" + (": " + ", ".join(o.name for o in outside[:8])
                                                       if outside else "")))
            if facts.get("npu_registered") is False:
                r.findings.append(("TS1125", "the file carries NPU bytecode but the NPU dispatch "
                                   f"library did not load ({why}); LiteRT could not run the dispatch "
                                   "partitions here"))
                everything("no NPU dispatch")
                return r
            for op in outside:
                info = artifact.node_info(op.index)
                r.off_target.append(Placement(info["node"], op.name, info["module"], "outside NPU partitions"))
            return r
        parts = facts.get("partitions") or []
        if not parts:
            r.findings.append(("TS1125", "the NPU was requested (HardwareAccelerator.NPU | CPU) but no "
                               "NPU compiler plugin partitioned the graph"
                               + (f" ({why})" if why else "") + ": every operator ran on the XNNPACK CPU "
                               f"fallback while is_fully_accelerated() returned {full}"))
            everything("XNNPACK CPU fallback")
            return r
        selected, total = sum(p[1] for p in parts), sum(p[2] for p in parts)
        r.notes.append(f"compiler plugin selected {selected} of {total} operators for the NPU")
        if selected < total:
            for op in g.ops:
                one = self.npu_selected(artifact.single_op(op.index))
                if not one or one[0] < one[1]:
                    info = artifact.node_info(op.index)
                    r.off_target.append(Placement(info["node"], op.name, info["module"], "not selected for the NPU"))
            r.findings.append(("TS1124", f"LiteRT put {selected} of {total} operators on the NPU; the "
                               "rest ran on the fallback"))
        return r

    def _versions(self):
        v = {"ai-edge-litert": _dist("ai-edge-litert")}
        if self.target == "npu":
            with contextlib.suppress(Exception):
                facts = npu_log_facts(_LITERT_ENV.get("log", ""), "")
                v["npu.registered"] = facts.get("npu_registered")
        if self.target.startswith("gpu"):
            v["dxcompiler"] = _file_sha(_preload_dxc())
            v["gpu.precision"] = "f32 enforced" if self.target == "gpu-f32" else "accelerator default"
        return v


# ----------------------------------------------------------------------------- ExecuTorch

class PteRuntime(Runtime):
    def unavailable(self):
        try:
            import executorch.runtime  # noqa: F401
        except Exception as exc:
            return f"executorch is not installed ({type(exc).__name__})"
        return None

    def open(self, data, artifact, *, profile=False):
        from executorch.runtime import Runtime as ETRuntime

        try:
            with _quiet_fd2():
                rt = ETRuntime.get()
                prog = rt.load_program(bytes(data))
                meth = prog.load_method("forward")
        except Exception as exc:
            raise RuntimeFailed(f"ExecuTorch could not load the program: {type(exc).__name__}: {exc}") from exc
        import torch

        def run(feeds):
            with _quiet_fd2():
                outs = meth.execute([torch.from_numpy(np.ascontiguousarray(f)) for f in feeds])
            return [o.detach().cpu().numpy() if hasattr(o, "numpy") else np.asarray(o) for o in outs]

        info = {"backends": list(getattr(rt.backend_registry, "registered_backend_names", []))}
        return Session(run, info)

    def residency(self, artifact, session):
        ins = artifact.instructions()
        backends = sorted({i["backend"] for i in ins if i["kind"] == "delegate"})
        r = Residency("delegate " + ", ".join(backends) if backends else "cpu", True,
                      "kernel and delegate calls in the program", len(ins))
        registered = set(session.info.get("backends", []))
        for b in backends:
            if registered and b not in registered:
                r.notes.append(f"delegate {b} is not registered in this runtime build")
        if backends:
            for k, i in enumerate(ins):
                if i["kind"] == "kernel":
                    r.off_target.append(Placement(f"instruction {k}", i["op"], "", "portable kernel"))
            r.notes.append(f"{sum(1 for i in ins if i['kind'] == 'delegate')} delegate call(s) to "
                           f"{', '.join(backends)}")
        else:
            r.notes.append("no delegate in the program: every op runs on portable kernels")
        return r

    def _versions(self):
        return {"executorch": _dist("executorch")}


# ----------------------------------------------------------------------------- Truestar export

class TsxRuntime(Runtime):
    def __init__(self, name, options=None):
        super().__init__(name, options)
        self.requested = "cuda" if self.target == "cuda" else "cpu"

    def unavailable(self):
        if self.target == "cuda":
            import torch

            if not torch.cuda.is_available():
                return "CUDA is not available to torch here"
        return None

    def open(self, data, artifact, *, profile=False):
        import torch

        from ..export import runner as R

        try:
            r = R.Runner(artifact.runner.artifact, executor=self.target)
        except Exception as exc:
            raise RuntimeFailed(f"Truestar executor {self.target}: {exc}") from exc

        def run(feeds):
            ins = [torch.from_numpy(np.ascontiguousarray(f)) for f in feeds]
            if self.target == "cuda":
                ins = [t.cuda() for t in ins]
            return [o.detach().cpu().numpy() for o in r(*ins)]

        return Session(run, {"executor": self.target})

    def residency(self, artifact, session):
        return Residency(self.requested, True, "single executor, no partitioning", artifact.node_count())

    def _versions(self):
        from .. import __version__

        return {"truestar": __version__}


def make(name: str, options: Optional[Dict[str, Any]] = None) -> Runtime:
    if name not in RUNTIMES:
        raise KeyError(name)
    fmt = RUNTIMES[name][0]
    cls = {ONNX: OrtRuntime, TFLITE: TfliteRuntime, PTE: PteRuntime, TSX: TsxRuntime}[fmt]
    return cls(name, options)
