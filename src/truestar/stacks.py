"""Compilers and runtimes as check backends: Inductor, ONNX Runtime and TensorRT.

A stack backend runs the user's model through a third party compiler or runtime instead of eager
PyTorch. Names (one `--backends` list mixes them freely with cpu, cuda and so on):

    cpu+inductor, cuda+inductor        torch.compile with the Inductor backend on that device
    rocm+inductor                      the same on an AMD GPU (needs a ROCm build of torch)
    ort:cpu, ort:cuda                  torch.onnx.export (dynamo exporter), then ONNX Runtime with the
                                       CPU provider, or the CUDA provider with CPU allowed as fallback
    trt:fp32, trt:tf32, trt:fp16       torch.onnx.export, then a TensorRT engine: strict float32 (TF32
                                       off), TensorRT's default float32 (TF32 on), or FP16 enabled
    metal                              Truestar's own compiler: torch.export, the Truestar IR, then
                                       the compiled executor with Metal kernels on the MPS device
                                       (docs/METAL.md)

Each stack exposes a Runner that compiles one nn.Module (or one ATen op wrapped in a module) and
returns an Engine: a callable that takes CPU tensors and returns CPU tensors in the order
`truestar.capture.flatten_tensors` uses, so every comparison in check.py applies unchanged.

The engine for the whole model is built exactly as a user would build it: no extra outputs, no
debug flags, nothing that changes fusion. See stackcheck.py for how a mismatch is localized.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import platform
import re
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

from .backends import Backend
from .capture import flatten_tensors, map_tensors

# name: (kind, option)
STACKS: Dict[str, Tuple[str, str]] = {
    "cpu+inductor": ("inductor", "cpu"),
    "cuda+inductor": ("inductor", "cuda"),
    "rocm+inductor": ("inductor", "rocm"),   # the same on an AMD GPU (ROCm build of torch, docs/ROCM.md)
    "ort:cpu": ("ort", "cpu"),
    "ort:cuda": ("ort", "cuda"),
    "trt:fp32": ("trt", "fp32"),
    "trt:tf32": ("trt", "tf32"),
    "trt:fp16": ("trt", "fp16"),
    "metal": ("truestar", "metal"),
}


class StackError(RuntimeError):
    """A stack could not compile or run a module (export failed, engine build failed, ...)."""


def is_stack_name(name: str) -> bool:
    return "+" in name or ":" in name or name in STACKS


@dataclass
class StackBackend(Backend):
    """A Backend whose results come from a compiler or runtime rather than eager PyTorch."""

    kind: str = ""  # "inductor", "ort", "trt", "truestar"
    option: str = ""  # device for inductor and ort, precision for trt
    # Test and study hooks (never set from the command line): see Runner subclasses.
    options: Dict[str, Any] = field(default_factory=dict)

    def compute_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """The precision the stack computes in, which sets the tolerance. Only trt:fp16 lowers a
        float32 or float64 model."""
        if self.kind == "trt" and self.option == "fp16" and dtype in (torch.float32,
                                                                       torch.float64):
            return torch.float16
        return dtype

    def make_runner(self) -> "Runner":
        if self.kind == "inductor":
            return InductorRunner(self)
        if self.kind == "ort":
            return OrtRunner(self)
        if self.kind == "truestar":
            return TruestarRunner(self)
        return TrtRunner(self)

    def as_dict(self) -> Dict:
        d = super().as_dict()
        d["stack"] = {"kind": self.kind, "option": self.option}
        return d


# ----------------------------------------------------------------------------- discovery


def _cxx_compiler() -> Optional[str]:
    names = ("cl",) if platform.system() == "Windows" else ("c++", "g++", "clang++")
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def _cuda_ok() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def probe(name: str) -> Tuple[Optional[StackBackend], str]:
    """(backend, "") when the stack can run here, else (None, the reason)."""
    key = name.strip().lower()
    if key not in STACKS:
        return None, ("unknown backend; stacks are " + ", ".join(STACKS))
    kind, opt = STACKS[key]
    if kind == "truestar":
        from .kernel import metal_backend

        why = metal_backend.unavailable_reason()
        if why is not None:
            return None, why
        return StackBackend(name=key, device="mps",
                            description=f"Truestar compiled executor, Metal kernels on the MPS device, "
                                        f"torch {torch.__version__}",
                            kind=kind, option=opt), ""
    if kind == "inductor":
        if opt in ("cuda", "rocm"):
            hip = getattr(torch.version, "hip", None)
            if opt == "rocm" and not hip:
                return None, (f"torch {torch.__version__} is not a ROCm build (torch.version.hip is None); "
                              f"rocm+inductor needs a ROCm wheel of torch and an AMD GPU")
            if not _cuda_ok():
                return None, "no AMD GPU visible to torch" if opt == "rocm" else "no CUDA device"
            try:
                import triton  # noqa: F401
            except Exception:
                return None, "Inductor on CUDA needs triton, which is not importable"
            dev = torch.cuda.get_device_name(0)
            return StackBackend(name=key, device="cuda:0",
                                description=f"torch.compile (Inductor) on {dev}, torch "
                                            f"{torch.__version__}" + (f", HIP {hip}" if hip else ""),
                                tf32=bool(torch.backends.cuda.matmul.allow_tf32
                                          or torch.backends.cudnn.allow_tf32),
                                kind=kind, option=opt), ""
        cxx = _cxx_compiler()
        if cxx is None:
            return None, ("Inductor on CPU needs a C++ compiler ("
                          + ("cl from Visual Studio Build Tools" if platform.system() == "Windows"
                             else "g++ or clang++") + ") on PATH; none was found")
        return StackBackend(name=key, device="cpu",
                            description=f"torch.compile (Inductor) on CPU, torch {torch.__version__}",
                            kind=kind, option=opt), ""
    try:
        import onnx  # noqa: F401
        import onnxscript  # noqa: F401
    except Exception:
        return None, "the ONNX exporter needs the onnx and onnxscript packages"
    if kind == "ort":
        try:
            import onnxruntime as ort
        except Exception:
            return None, "onnxruntime is not installed"
        provs = ort.get_available_providers()
        if opt == "cuda":
            if "CUDAExecutionProvider" not in provs:
                return None, ("this onnxruntime build has no CUDAExecutionProvider (install "
                              "onnxruntime-gpu); available: " + ", ".join(provs))
            if not _cuda_ok():
                return None, "no CUDA device"
            desc = f"ONNX Runtime {ort.__version__}, CUDA provider (CPU fallback allowed)"
            return StackBackend(name=key, device="cuda:0", description=desc, kind=kind,
                                option=opt), ""
        return StackBackend(name=key, device="cpu",
                            description=f"ONNX Runtime {ort.__version__}, CPU provider",
                            kind=kind, option=opt), ""
    try:
        import tensorrt as trt
    except Exception:
        return None, "tensorrt is not installed"
    if not _cuda_ok():
        return None, "no CUDA device"
    label = {"fp32": "float32, TF32 off", "tf32": "float32 with TF32 (TensorRT default)",
             "fp16": "FP16 enabled, TF32 off"}[opt]
    return StackBackend(name=key, device="cuda:0",
                        description=f"TensorRT {trt.__version__}, {label}",
                        tf32=opt == "tf32", kind=kind, option=opt), ""


def available_stacks() -> List[StackBackend]:
    out = []
    for name in STACKS:
        b, _ = probe(name)
        if b is not None:
            out.append(b)
    return out


# ----------------------------------------------------------------------------- shared helpers


def _tensor_leaves(obj: Any) -> List[torch.Tensor]:
    from torch.utils import _pytree as pytree

    return [x for x in pytree.tree_leaves(obj) if isinstance(x, torch.Tensor)]


class OpModule(nn.Module):
    """One ATen op call as a module whose inputs are the call's tensors, so that a stack can compile
    the op alone. Non tensor arguments are baked in as constants."""

    def __init__(self, func, args: Tuple, kwargs: Dict, first_only: bool = False):
        super().__init__()
        self.first_only = first_only
        from torch.utils import _pytree as pytree

        leaves, self._spec = pytree.tree_flatten((args, kwargs))
        self._slots = [i for i, x in enumerate(leaves) if isinstance(x, torch.Tensor)]
        self._consts = [None if isinstance(x, torch.Tensor) else x for x in leaves]
        self.func = func

    def forward(self, *tensors):
        from torch.utils import _pytree as pytree

        leaves = list(self._consts)
        for i, t in zip(self._slots, tensors):
            leaves[i] = t
        a, k = pytree.tree_unflatten(leaves, self._spec)
        out = self.func(*a, **k)
        if self.first_only and isinstance(out, (tuple, list)):
            return out[0]
        return out


def op_inputs(args: Tuple, kwargs: Dict) -> Tuple:
    return tuple(_tensor_leaves((args, kwargs)))


def _retarget_devices(obj: Any, device: str) -> Any:
    if isinstance(obj, torch.device):
        return torch.device(device)
    if isinstance(obj, tuple):
        return tuple(_retarget_devices(x, device) for x in obj)
    if isinstance(obj, list):
        return [_retarget_devices(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _retarget_devices(v, device) for k, v in obj.items()}
    return obj


class Engine:
    """A compiled module. Call with CPU args and kwargs; returns CPU tensors (flatten order)."""

    def __init__(self, run: Callable, info: Optional[Dict[str, Any]] = None,
                 close: Optional[Callable] = None):
        self._run = run
        self.info: Dict[str, Any] = info if info is not None else {}
        # (code, message) findings about the engine itself, such as nodes that ran on the CPU.
        self.findings: List[Tuple[str, str]] = []
        self._close = close

    def __call__(self, args: Tuple, kwargs: Dict) -> List[torch.Tensor]:
        return self._run(args, kwargs)

    def close(self) -> None:
        if self._close is not None:
            self._close()
            self._close = None


class Runner:
    """Compiles modules for one stack. One runner serves a whole check, so caches it holds (the
    TensorRT builder and timing cache, for example) are shared by every engine it builds."""

    def __init__(self, backend: StackBackend):
        self.backend = backend
        self.options = dict(backend.options)
        self.compiles = 0

    def compile(self, module: nn.Module, args: Tuple, kwargs: Dict, *,
                whole_model: bool = False) -> Engine:
        raise NotImplementedError

    def compile_op(self, func, args: Tuple, kwargs: Dict) -> Engine:
        """Compile one ATen op alone; the engine is called like the op. An op with several
        outputs that the stack cannot return all of (TensorRT takes only the normalized output of
        LayerNormalization, not mean and rstd) is retried returning its first output only, and
        info["first_output_only"] says so."""
        try:
            eng = self.compile(OpModule(func, args, kwargs), op_inputs(args, kwargs), {})
        except StackError:
            eng = self.compile(OpModule(func, args, kwargs, first_only=True),
                               op_inputs(args, kwargs), {})
            eng.info["first_output_only"] = True
        inner = eng._run
        eng._run = lambda a, k: inner(op_inputs(a, k), {})
        return eng

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------------- Inductor

_KERNEL_DEF = re.compile(r"^\s*(\w+) = async_compile\.(?:triton|cpp_pybinding|cpp|cuda)\(")
_ORIGIN = re.compile(r"#\s*Topologically Sorted Source Nodes: \[(.*?)\], Original ATen: \[(.*?)\]")
_EXTERN = re.compile(r"^\s*(extern_kernels\.\w+|torch\.ops\.\w+\.\w+(?:\.\w+)?)\(")


def inductor_kernels(sources: List[str]) -> List[Dict[str, Any]]:
    """Kernels in Inductor's generated code, each with the ATen ops it was generated from.

    Inductor writes `# Topologically Sorted Source Nodes: [...], Original ATen: [...]` above every
    kernel definition and every extern call (library GEMMs, convolutions). Returns
    [{"kernel": name, "aten": [...], "nodes": [...], "extern": bool}] in definition order."""
    out: List[Dict[str, Any]] = []
    seen = set()
    for src in sources:
        pending = None
        for line in src.splitlines():
            m = _ORIGIN.search(line)
            if m:
                nodes = [x.strip() for x in m.group(1).split(",") if x.strip()]
                aten = [x.strip() for x in m.group(2).split(",") if x.strip()]
                pending = (nodes, aten)
                continue
            if pending is None:
                continue
            k = _KERNEL_DEF.match(line)
            e = None if k else _EXTERN.match(line)
            if k or e:
                name = k.group(1) if k else e.group(1)
                key = (name, tuple(pending[1]))
                if key not in seen:
                    seen.add(key)
                    out.append({"kernel": name, "aten": pending[1], "nodes": pending[0],
                                "extern": e is not None})
                pending = None
    return out


class InductorRunner(Runner):
    """torch.compile(module) with the Inductor backend and its default settings.

    Dynamo's recompile limit is raised and hitting it raises instead of quietly running eager code,
    so every engine here is really compiled. Graph breaks are counted and reported: code between
    graphs runs eagerly, exactly as it does for the user."""

    def __init__(self, backend: StackBackend):
        super().__init__(backend)
        self.device = "cuda:0" if backend.option in ("cuda", "rocm") else "cpu"
        self.mode = self.options.get("mode")

    def _patch(self):
        import torch._dynamo.config as dcfg

        return dcfg.patch(recompile_limit=1_000_000, accumulated_recompile_limit=1_000_000,
                          fail_on_recompile_limit_hit=True)

    def compile(self, module: nn.Module, args: Tuple, kwargs: Dict, *,
                whole_model: bool = False) -> Engine:
        from torch._dynamo.utils import counters
        from torch._inductor.utils import run_and_get_code

        mod = copy.deepcopy(module).to(self.device).eval()
        kw = {"mode": self.mode} if self.mode else {}
        compiled = torch.compile(mod, backend="inductor", **kw)
        state = {"first": True}
        info: Dict[str, Any] = {"kernels": [], "graph_breaks": 0}
        self.compiles += 1

        def move(o):
            return _retarget_devices(map_tensors(o, lambda t: t.to(self.device)), self.device)

        def run(a, k):
            a2, k2 = move(a), move(k)
            with torch.no_grad(), self._patch():
                if state["first"]:
                    breaks0 = sum(counters["graph_break"].values())
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        out, codes = run_and_get_code(compiled, *a2, **k2)
                    info["kernels"] = inductor_kernels(codes)
                    info["graph_breaks"] = sum(counters["graph_break"].values()) - breaks0
                    state["first"] = False
                else:
                    out = compiled(*a2, **k2)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            return [t.detach().cpu() for t in flatten_tensors(out)]

        def close():
            nonlocal compiled, mod
            compiled = mod = None

        return Engine(run, info, close)

    def compile_op(self, func, args: Tuple, kwargs: Dict) -> Engine:
        return super().compile_op(func, _retarget_devices(args, self.device),
                                  _retarget_devices(kwargs, self.device))


# ----------------------------------------------------------------------------- Truestar's compiler


class TruestarRunner(Runner):
    """The module through Truestar's own compiler: torch.export into the Truestar IR, then the
    compiled executor on the Metal device. Every kernel and region is checked against float64 on its
    first run by the executor itself; ops with no kernel lowering run through the float64 reference
    on the CPU, and the engine reports them as TS400 instead of letting them pass as Metal results."""

    def __init__(self, backend: StackBackend):
        super().__init__(backend)
        self.device = backend.option

    def compile(self, module: nn.Module, args: Tuple, kwargs: Dict, *,
                whole_model: bool = False) -> Engine:
        from . import ir
        from .compiler import compile_program

        if kwargs:
            raise StackError("the Truestar stack takes positional tensor inputs only")
        if not all(isinstance(a, torch.Tensor) for a in args):
            raise StackError("the Truestar stack takes tensor inputs only")
        floats = [a.dtype for a in args if a.is_floating_point()]
        dt = floats[0] if floats else torch.float32
        if dt not in (torch.float32, torch.float16, torch.bfloat16):
            raise StackError(f"[TS841] the Metal device has no {str(dt).replace('torch.', '')} "
                             f"arithmetic; check the model in float32, float16 or bfloat16")
        mod = copy.deepcopy(module).eval()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prog = ir.import_module(mod, tuple(args))
            cp = compile_program(prog, self.device, str(dt).replace("torch.", ""))
        except Exception as exc:
            raise StackError(f"Truestar could not compile the module: {type(exc).__name__}: {exc}") from exc
        self.compiles += 1
        rep = cp.report
        info: Dict[str, Any] = {"kernels": [k.get("shape") for k in rep.kernels],
                                "fallbacks": len(rep.fallbacks), "library": len(rep.library)}
        def run(a, k):
            outs = cp([t for t in a])
            if self.device == "metal":
                torch.mps.synchronize()
            return [t.detach().cpu() for t in flatten_tensors(outs)]

        def close():
            nonlocal cp
            cp = None

        eng = Engine(run, info, close)
        if rep.fallbacks:
            ops = sorted({str(o) for f in rep.fallbacks for o in f.get("ops", [])})
            eng.findings.append(("TS400", f"{len(rep.fallbacks)} op(s) ran through the float64 reference "
                                          f"on the CPU, not on Metal: {', '.join(ops)}"))
        return eng


# ----------------------------------------------------------------------------- ONNX export


@dataclass
class Exported:
    data: bytes
    inputs: List[str]
    outputs: List[str]
    out_spec: Any
    nodes: Dict[str, Dict[str, str]]  # node name -> {"op": op_type, "module": module path}


def _module_path(node) -> str:
    for p in node.metadata_props:
        if p.key == "pkg.torch.onnx.name_scopes":
            try:
                scopes = [s for s in json.loads(p.value.replace("'", '"')) if s]
            except Exception:
                return ""
            # cumulative paths, ['', 'blocks', 'blocks.0', 'blocks.0.fc', 'linear']: the module is
            # the last one before the op (joining them gave 'blocks.blocks.0.blocks.0.fc')
            paths = scopes[:-1]
            return paths[-1] if paths else ""
    return ""


def export_onnx(module: nn.Module, args: Tuple, kwargs: Dict,
                translations: Optional[Dict] = None) -> Exported:
    """torch.onnx.export with the default (dynamo) exporter, captured output kept quiet."""
    sink = io.StringIO()
    extra = {"custom_translation_table": translations} if translations else {}
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(sink):
            warnings.simplefilter("ignore")
            prog = torch.onnx.export(module, tuple(args), kwargs=dict(kwargs) or None,
                                     dynamo=True, verbose=False, **extra)
    except Exception as exc:
        raise StackError(f"torch.onnx.export failed: {type(exc).__name__}: "
                         f"{str(exc).strip().splitlines()[0] if str(exc).strip() else ''}") from exc
    proto = prog.model_proto
    init = {i.name for i in proto.graph.initializer}
    inputs = [i.name for i in proto.graph.input if i.name not in init]
    outputs = [o.name for o in proto.graph.output]
    try:
        out_spec = prog.exported_program.call_spec.out_spec
    except Exception:
        out_spec = None
    nodes = {n.name: {"op": n.op_type, "module": _module_path(n)} for n in proto.graph.node}
    return Exported(proto.SerializeToString(), inputs, outputs, out_spec, nodes)


def _restructure(outs: List[torch.Tensor], out_spec) -> List[torch.Tensor]:
    """Outputs in graph order back to the order flatten_tensors gives for the eager output."""
    if out_spec is None:
        return outs
    from torch.utils import _pytree as pytree

    try:
        if out_spec.num_leaves != len(outs):
            return outs
        return flatten_tensors(pytree.tree_unflatten(list(outs), out_spec))
    except Exception:
        return outs


def _feed(exp: Exported, args: Tuple, kwargs: Dict) -> List[torch.Tensor]:
    ts = _tensor_leaves((args, kwargs))
    if len(ts) != len(exp.inputs):
        raise StackError(f"the exported graph takes {len(exp.inputs)} inputs but the call passes "
                         f"{len(ts)} tensors")
    return ts


# ----------------------------------------------------------------------------- ONNX Runtime

# Ops ORT deliberately keeps on the CPU in a GPU session to compute shapes. A node on the CPU that
# produces floating point data is not one of those: it is the model's math running off the GPU.
_FLOAT_TYPES = ("float", "double", "float16", "bfloat16")


class OrtRunner(Runner):
    """ONNX Runtime session per exported module.

    ort:cuda asks for [CUDAExecutionProvider, CPUExecutionProvider], what users write. ORT then
    places nodes the CUDA provider lacks on the CPU with no error; the whole model session is
    profiled once to read where every node actually ran (`provider` per node), and floating point
    nodes on the CPU become a finding (TS400). If the CUDA provider did not load at all
    (`get_providers()` lacks it), the session silently ran everything on the CPU (TS401)."""

    def __init__(self, backend: StackBackend):
        super().__init__(backend)
        self.providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                          if backend.option == "cuda" else ["CPUExecutionProvider"])

    def compile(self, module: nn.Module, args: Tuple, kwargs: Dict, *,
                whole_model: bool = False) -> Engine:
        import onnxruntime as ort

        exp = export_onnx(module, args, kwargs, self.options.get("translations"))
        self.compiles += 1
        so = ort.SessionOptions()
        so.log_severity_level = 3
        for k, v in self.options.get("session_config", {}).items():
            so.add_session_config_entry(k, v)
        prof_dir = None
        if whole_model and self.backend.option == "cuda":
            prof_dir = tempfile.mkdtemp(prefix="truestar_ort_")
            so.enable_profiling = True
            so.profile_file_prefix = os.path.join(prof_dir, "profile")
        try:
            sess = ort.InferenceSession(exp.data, so, providers=self.providers)
        except Exception as exc:
            raise StackError(f"ONNX Runtime could not load the exported model: "
                             f"{type(exc).__name__}: {exc}") from exc
        got = sess.get_providers()
        info: Dict[str, Any] = {"providers": got, "onnx_nodes": len(exp.nodes)}
        if self.backend.option == "cuda":
            opts = sess.get_provider_options().get("CUDAExecutionProvider", {})
            info["tf32"] = str(opts.get("use_tf32", "0")) == "1"
        eng = Engine(lambda a, k: None, info)
        if self.backend.option == "cuda" and "CUDAExecutionProvider" not in got:
            eng.findings.append((
                "TS401", "ONNX Runtime did not load the CUDA provider (session providers: "
                + ", ".join(got) + "), so the whole model ran on the CPU while the GPU was "
                "requested. Check the CUDA and cuDNN libraries onnxruntime-gpu needs."))
        names = [i.name for i in sess.get_inputs()]
        state = {"profiled": prof_dir is None}

        def run(a, k):
            ts = _feed(exp, a, k)
            feed = {}
            for n, t in zip(exp.inputs, ts):
                if t.dtype == torch.bfloat16:
                    raise StackError("ONNX Runtime from Python takes no bfloat16 inputs")
                feed[n] = t.detach().cpu().contiguous().numpy()
            feed = {n: feed[n] for n in names if n in feed}
            outs = sess.run(None, feed)
            if not state["profiled"]:
                state["profiled"] = True
                eng.findings.extend(_placement_findings(sess.end_profiling(), exp))
                shutil.rmtree(prof_dir, ignore_errors=True)
            return _restructure([torch.from_numpy(o.copy()) if hasattr(o, "copy")
                                 else torch.as_tensor(o) for o in outs], exp.out_spec)

        eng._run = run
        return eng


def _placement_findings(profile_path: str, exp: Exported) -> List[Tuple[str, str]]:
    """Floating point nodes that ORT ran on the CPU in a CUDA session, from its profile."""
    try:
        with open(profile_path, encoding="utf-8") as f:
            events = json.load(f)
    except Exception as exc:
        return [("TS400", f"could not read where ORT placed nodes (profile unreadable: {exc}); "
                          "silent CPU fallback was not ruled out")]
    on_cpu: Dict[str, Dict[str, str]] = {}
    for e in events:
        if e.get("cat") != "Node" or not str(e.get("name", "")).endswith("_kernel_time"):
            continue
        a = e.get("args", {})
        if a.get("provider") != "CPUExecutionProvider" or str(a.get("op_name", "")).startswith(
                "Memcpy"):
            continue
        outs = a.get("output_type_shape", [])
        floating = any(k in _FLOAT_TYPES for o in outs for k in o)
        if not floating:
            continue  # shape arithmetic ORT keeps on the CPU on purpose
        node = e["name"][: -len("_kernel_time")]
        meta = exp.nodes.get(node)
        if meta is None:  # renamed by an ORT rewrite: match by op type when it is unique
            same = [m for m in exp.nodes.values() if m["op"] == a.get("op_name")]
            meta = same[0] if len(same) == 1 else {"op": a.get("op_name", "?"), "module": ""}
        on_cpu[node] = {"op": a.get("op_name", meta["op"]), "module": meta.get("module", "")}
    if not on_cpu:
        return []
    parts = []
    for node, m in list(on_cpu.items())[:8]:
        where = f" in module {m['module']}" if m["module"] else ""
        parts.append(f"{m['op']} (node {node}{where})")
    return [("TS400", f"{len(on_cpu)} floating point node(s) ran on the CPU in a CUDA session, "
                      "with no error from ONNX Runtime: " + ", ".join(parts)
             + ". Their results are the CPU provider's, not the GPU's, and each one adds a "
               "device copy.")]


# ----------------------------------------------------------------------------- TensorRT

def _trt_dtype(trt, dt) -> torch.dtype:
    table = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
             trt.int64: torch.int64, trt.bool: torch.bool, trt.int8: torch.int8,
             trt.bfloat16: torch.bfloat16, trt.uint8: torch.uint8}
    if dt not in table:
        raise StackError(f"TensorRT tensor type {dt} has no torch equivalent here")
    return table[dt]


class TrtRunner(Runner):
    """ONNX export then a TensorRT engine, built with one Builder and one timing cache per check.

    Sharing the builder and the timing cache cuts a small engine build from about 15 s to under
    1 s here, and makes the per module engines pick the tactics the whole model engine timed. The
    whole model engine is built as users build it: graph outputs only, no layer marked as an extra
    output (marking every tensor as an output, as Polygraphy's `--trt-outputs mark all` does,
    stops TensorRT from fusing through those tensors and so checks a different engine).

    Test hook: options["layer_hook"](network, config) runs before the build (used to force one
    layer type to FP16 to prove a precision fault is caught)."""

    def __init__(self, backend: StackBackend):
        super().__init__(backend)
        import tensorrt as trt

        self.trt = trt
        self.errors: List[str] = []
        runner = self

        class _Log(trt.ILogger):
            def __init__(self):
                trt.ILogger.__init__(self)

            def log(self, severity, msg):
                if severity <= trt.ILogger.ERROR:
                    runner.errors.append(msg)

        self.logger = _Log()
        self.builder = trt.Builder(self.logger)
        self.runtime = trt.Runtime(self.logger)
        self.cache = self.builder.create_builder_config().create_timing_cache(b"")

    def _config(self):
        trt = self.trt
        cfg = self.builder.create_builder_config()
        if self.backend.option != "tf32":
            cfg.clear_flag(trt.BuilderFlag.TF32)
        if self.backend.option == "fp16":
            cfg.set_flag(trt.BuilderFlag.FP16)
        cfg.set_timing_cache(self.cache, False)
        cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        return cfg

    def compile(self, module: nn.Module, args: Tuple, kwargs: Dict, *,
                whole_model: bool = False) -> Engine:
        trt = self.trt
        exp = export_onnx(module, args, kwargs, self.options.get("translations"))
        self.compiles += 1
        self.errors.clear()
        net = self.builder.create_network(0)
        parser = trt.OnnxParser(net, self.logger)
        if not parser.parse(exp.data):
            errs = [parser.get_error(i).desc() for i in range(parser.num_errors)]
            raise StackError("TensorRT could not parse the exported model: " + "; ".join(errs[:3]))
        cfg = self._config()
        hook = self.options.get("layer_hook")
        if hook is not None:
            hook(net, cfg)
        ser = self.builder.build_serialized_network(net, cfg)
        if ser is None:
            raise StackError("TensorRT engine build failed: " + "; ".join(self.errors[-3:]))
        engine = self.runtime.deserialize_cuda_engine(ser)
        ctx = engine.create_execution_context()
        in_types = {n: _trt_dtype(trt, engine.get_tensor_dtype(n)) for n in exp.inputs}
        out_types = {n: _trt_dtype(trt, engine.get_tensor_dtype(n)) for n in exp.outputs}
        info: Dict[str, Any] = {"layers": None}

        def layers() -> List[str]:
            try:
                insp = engine.create_engine_inspector()
                data = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
                return [l.get("Name", "") if isinstance(l, dict) else str(l)
                        for l in data.get("Layers", [])]
            except Exception:
                return []

        info["layer_names"] = layers

        def run(a, k):
            ts = _feed(exp, a, k)
            keep = []
            for n, t in zip(exp.inputs, ts):
                d = t.detach().to("cuda", dtype=in_types[n]).contiguous()
                keep.append(d)
                ctx.set_input_shape(n, tuple(d.shape))
                ctx.set_tensor_address(n, d.data_ptr())
            outs = []
            for n in exp.outputs:
                shape = tuple(ctx.get_tensor_shape(n))
                o = torch.empty(shape, dtype=out_types[n], device="cuda")
                outs.append(o)
                ctx.set_tensor_address(n, o.data_ptr())
            stream = torch.cuda.current_stream()
            if not ctx.execute_async_v3(stream.cuda_stream):
                raise StackError("TensorRT execution failed: " + "; ".join(self.errors[-3:]))
            stream.synchronize()
            return _restructure([o.cpu() for o in outs], exp.out_spec)

        def close():
            nonlocal ctx, engine
            ctx = engine = None

        return Engine(run, info, close)
