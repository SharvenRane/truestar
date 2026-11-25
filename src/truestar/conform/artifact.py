"""Deployed artifact formats: detection, graph units mapped to modules, cuts and float64 meaning.

A *unit* is the part of the artifact that came from one PyTorch module: the nodes (ONNX) or
operators (TFLite) whose source location names that module, with the tensors entering and leaving
it. Conformance cuts a unit out as a model of its own, runs it in the same runtime on the float64
reference inputs of that module and so measures the unit's own error, with nothing inherited and
without instrumenting the run that is judged.

What each format allows:

| format | units mapped from | cut | float64 meaning of the file |
|---|---|---|---|
| .onnx | `pkg.torch.onnx.name_scopes` metadata (torch.onnx dynamo exporter), else `/a/b/Op` node names | onnx.utils.Extractor | onnx.reference evaluator on a float64 copy |
| .tflite | tensor names `<module path>/<name>` | tflite_graph.cut | tflite_graph.evaluate64 (listed ops) |
| .pte | not mappable: the program keeps no module names | no | no |
| .tsx | not mapped (the reference interpreter is the meaning) | no | the IR reference interpreter, float64 |
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ONNX, TFLITE, PTE, TSX, COREML = "onnx", "tflite", "pte", "tsx", "coreml"


class ArtifactError(RuntimeError):
    """The artifact cannot be read or its format is not supported here (code TS620)."""


@dataclass
class Unit:
    module: str  # module path in the PyTorch model ("" is the top module)
    nodes: List[int]  # node or operator indices, in graph order
    inputs: List[str]  # tensor ids entering the unit (no constants)
    outputs: List[str]  # tensor ids leaving the unit
    ops: List[str] = field(default_factory=list)  # op type per node


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect(path: str) -> str:
    if not os.path.exists(path):
        raise ArtifactError(f"{path} not found")
    if os.path.isdir(path):
        if path.rstrip("/\\").endswith(".mlpackage") or os.path.isfile(os.path.join(path, "Manifest.json")):
            return COREML
        raise ArtifactError(f"{path} is a directory, not a model file")
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as f:
        head = f.read(16)
    if head[:8] == b"TRUESTAR":
        return TSX
    if head[4:8] == b"TFL3":
        return TFLITE
    if head[4:6] == b"ET" or ext == ".pte":
        return PTE
    if ext == ".mlmodel":
        return COREML
    if ext == ".onnx":
        return ONNX
    raise ArtifactError(f"{path}: unknown artifact format (extension {ext or 'none'}); conform reads "
                        ".onnx, .tflite, .pte and .tsx files")


COREML_REASON = ("Core ML artifacts (.mlpackage, .mlmodel) can only be run by Apple's Core ML "
                 "framework, which exists on macOS and iOS only; coremltools on Windows has no "
                 "wheel (pip builds 9.0 from source) and its predict() needs macOS. Run conform on a "
                 "Mac, or check the .onnx or .tflite export of the same model here.")


class Artifact:
    """A loaded artifact. Subclasses implement the format specific parts."""

    format = ""
    can_cut = False
    can_mean = False

    def __init__(self, path: str, data: bytes):
        self.path = path
        self.data = data
        self.sha256 = _sha256(data)

    # signature: (name, shape, numpy dtype) per input and output
    def input_specs(self) -> List[Tuple[str, Tuple[int, ...], Any]]:
        raise NotImplementedError

    def output_names(self) -> List[str]:
        raise NotImplementedError

    def node_count(self) -> int:
        return 0

    def node_info(self, i: int) -> Dict[str, str]:
        return {}

    def units(self) -> List[Unit]:
        return []

    def cut(self, unit: Unit) -> bytes:
        raise ArtifactError(f"{self.format} artifacts cannot be cut")

    def meaning64(self, feeds: Dict[str, np.ndarray], wanted: Sequence[str],
                  unit: Optional[Unit] = None) -> Optional[List[np.ndarray]]:
        return None

    # values cross into and out of the file as real numbers; a format with quantized tensors maps
    # them to and from its integer codes
    def encode(self, tid: str, a: np.ndarray, dtype) -> np.ndarray:
        return a.astype(dtype) if a.dtype.kind == "f" else a

    def decode(self, tid: str, a: np.ndarray) -> np.ndarray:
        return a

    def encode_inputs(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        return arrays

    def decode_outputs(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        return arrays

    def describe(self) -> Dict[str, Any]:
        return {"path": self.path, "format": self.format, "sha256": self.sha256,
                "bytes": len(self.data), "nodes": self.node_count()}


# ----------------------------------------------------------------------------- ONNX

def _scope_module(node) -> Optional[str]:
    for p in node.metadata_props:
        if p.key == "pkg.torch.onnx.name_scopes":
            try:
                scopes = json.loads(p.value.replace("'", '"'))
            except Exception:
                return None
            # cumulative paths: ['', 'blocks', 'blocks.0', 'blocks.0.fc', 'linear']
            paths = [s for s in scopes[:-1] if s]
            return paths[-1] if paths else ""
    name = node.name or ""
    if name.startswith("/") and name.count("/") >= 2:  # TorchScript exporter: /blocks.0/fc/Gemm
        return ".".join(p for p in name.strip("/").split("/")[:-1] if p)
    return None


_ONNX_NP = {1: np.float32, 2: np.uint8, 3: np.int8, 5: np.int16, 6: np.int32, 7: np.int64,
            9: np.bool_, 10: np.float16, 11: np.float64}


class OnnxArtifact(Artifact):
    format = ONNX
    can_cut = True
    can_mean = True

    def __init__(self, path: str, data: bytes):
        super().__init__(path, data)
        try:
            import onnx
        except ImportError:
            from ..deps import MissingDependencyError

            raise MissingDependencyError("onnx", "Reading a .onnx file") from None
        try:
            self.model = onnx.load_from_string(data)
        except Exception as exc:
            raise ArtifactError(f"{path}: not a readable ONNX model ({type(exc).__name__}: {exc})") from exc
        g = self.model.graph
        self.inits = {i.name for i in g.initializer}
        self.const_out = {o for n in g.node if n.op_type == "Constant" for o in n.output}

    def input_specs(self):
        out = []
        for i in self.model.graph.input:
            if i.name in self.inits:
                continue
            t = i.type.tensor_type
            shape = tuple(int(d.dim_value) if d.HasField("dim_value") else -1 for d in t.shape.dim)
            out.append((i.name, shape, _ONNX_NP.get(t.elem_type, np.float32)))
        return out

    def output_names(self):
        return [o.name for o in self.model.graph.output]

    def node_count(self):
        return len(self.model.graph.node)

    def node_info(self, i):
        n = self.model.graph.node[i]
        return {"node": n.name, "op": n.op_type, "module": _scope_module(n) or ""}

    def units(self) -> List[Unit]:
        nodes = list(self.model.graph.node)
        mod = [_scope_module(n) for n in nodes]
        if all(m is None for m in mod):
            return []
        produced_by = {o: i for i, n in enumerate(nodes) for o in n.output}
        consumers: Dict[str, List[int]] = {}
        for i, n in enumerate(nodes):
            for t in n.input:
                consumers.setdefault(t, []).append(i)
        graph_out = set(self.output_names())
        paths = set()
        for m in mod:
            if m:
                parts = m.split(".")
                for k in range(1, len(parts) + 1):
                    paths.add(".".join(parts[:k]))
        units = []
        for p in sorted(paths, key=lambda s: (min(i for i, m in enumerate(mod) if m and (m == s or m.startswith(s + "."))), s)):
            idx = [i for i, m in enumerate(mod) if m and (m == p or m.startswith(p + "."))]
            # skip Constant nodes: they are folded into the cut as they are
            ids = set(idx)
            ins, outs = [], []
            for i in idx:
                for t in nodes[i].input:
                    if not t or t in self.inits or t in ins:
                        continue
                    src = produced_by.get(t)
                    if src is not None and (src in ids or nodes[src].op_type == "Constant"):
                        continue
                    ins.append(t)
                for t in nodes[i].output:
                    if t in graph_out or any(c not in ids for c in consumers.get(t, [])):
                        if t not in outs:
                            outs.append(t)
            units.append(Unit(p, idx, ins, outs, [nodes[i].op_type for i in idx]))
        return units

    def cut(self, unit: Unit) -> bytes:
        import onnx
        from onnx.utils import Extractor

        try:
            m = Extractor(self.model).extract_model(list(unit.inputs), list(unit.outputs))
        except Exception as exc:
            raise ArtifactError(f"could not cut module {unit.module or '(top)'} out of the ONNX "
                                f"graph: {type(exc).__name__}: {exc}") from exc
        return m.SerializeToString()

    def _double_model(self, model):
        """A float64 copy: float initializers, constants, casts and declared types widened."""
        import copy

        from onnx import TensorProto, numpy_helper

        m = copy.deepcopy(model)
        g = m.graph
        F, D = TensorProto.FLOAT, TensorProto.DOUBLE
        for init in g.initializer:
            if init.data_type in (F, TensorProto.FLOAT16):
                arr = numpy_helper.to_array(init).astype(np.float64)
                init.CopyFrom(numpy_helper.from_array(arr, init.name))
        for vi in list(g.input) + list(g.output) + list(g.value_info):
            if vi.type.tensor_type.elem_type in (F, TensorProto.FLOAT16):
                vi.type.tensor_type.elem_type = D
        for n in g.node:
            for a in n.attribute:
                if n.op_type in ("Cast", "CastLike") and a.name == "to" and a.i in (F, TensorProto.FLOAT16):
                    a.i = D
                if a.type == a.TENSOR and a.t.data_type in (F, TensorProto.FLOAT16):
                    arr = numpy_helper.to_array(a.t).astype(np.float64)
                    a.t.CopyFrom(numpy_helper.from_array(arr, a.t.name))
        return m

    def meaning64(self, feeds, wanted, unit=None):
        try:
            from onnx.reference import ReferenceEvaluator
        except Exception:
            return None
        try:
            base = self.model if unit is None else __import__("onnx").load_from_string(self.cut(unit))
            ev = ReferenceEvaluator(self._double_model(base))
            f64 = {k: (np.asarray(v, dtype=np.float64) if np.asarray(v).dtype.kind == "f" else np.asarray(v))
                   for k, v in feeds.items()}
            outs = ev.run(list(wanted), f64)
            return [np.asarray(o, dtype=np.float64) for o in outs]
        except Exception:
            return None


# ----------------------------------------------------------------------------- TFLite

class TfliteArtifact(Artifact):
    format = TFLITE
    can_cut = True
    can_mean = True

    def __init__(self, path: str, data: bytes):
        super().__init__(path, data)
        from .tflite_graph import TfliteGraph

        try:
            self.graph = TfliteGraph(data)
        except ImportError:
            from ..deps import MissingDependencyError

            raise MissingDependencyError("ai_edge_litert", "Reading a .tflite file",
                                         package="ai-edge-litert") from None
        except Exception as exc:
            raise ArtifactError(f"{path}: not a readable TFLite model ({type(exc).__name__}: {exc})") from exc

    def tid(self, i: int) -> str:
        return f"{i}:{self.graph.tensor_name(i)}"

    @staticmethod
    def index(tid: str) -> int:
        return int(tid.split(":", 1)[0])

    def input_specs(self):
        g = self.graph
        return [(g.tensor_name(i), g.shape(i), g.dtype(i)) for i in g.inputs]

    def output_names(self):
        return [self.graph.tensor_name(i) for i in self.graph.outputs]

    def node_count(self):
        return len(self.graph.ops)

    def _op_module(self, op) -> Optional[str]:
        names = [self.graph.tensor_name(t) for t in op.outputs]
        for n in names:
            first = n.split(";")[0]
            if "/" in first:
                return first.rsplit("/", 1)[0].replace("/", ".")
        return None

    def node_info(self, i):
        op = self.graph.ops[i]
        return {"node": self.graph.tensor_name(op.outputs[0]) if op.outputs else str(i), "op": op.name,
                "module": self._op_module(op) or ""}

    def units(self) -> List[Unit]:
        g = self.graph
        mod = [self._op_module(op) for op in g.ops]
        if all(m is None for m in mod):
            return []
        consumers: Dict[int, List[int]] = {}
        for op in g.ops:
            for t in op.inputs:
                consumers.setdefault(t, []).append(op.index)
        prod = g.producers()
        graph_out = set(g.outputs)
        paths = set()
        for m in mod:
            if m:
                parts = m.split(".")
                for k in range(1, len(parts) + 1):
                    paths.add(".".join(parts[:k]))
        units = []
        for p in sorted(paths, key=lambda s: (min(i for i, m in enumerate(mod) if m and (m == s or m.startswith(s + "."))), s)):
            idx = [i for i, m in enumerate(mod) if m and (m == p or m.startswith(p + "."))]
            ids = set(idx)
            ins, outs = [], []
            for i in idx:
                for t in g.ops[i].inputs:
                    if t < 0 or g.constant(t) is not None or self.tid(t) in ins:
                        continue
                    if prod.get(t) in ids:
                        continue
                    ins.append(self.tid(t))
                for t in g.ops[i].outputs:
                    if t in graph_out or any(c not in ids for c in consumers.get(t, [])):
                        if self.tid(t) not in outs:
                            outs.append(self.tid(t))
            units.append(Unit(p, idx, ins, outs, [g.ops[i].name for i in idx]))
        return units

    def _encode_index(self, i: int, a: np.ndarray, dtype=None) -> np.ndarray:
        g = self.graph
        if g.quant(i) is not None and np.asarray(a).dtype.kind == "f":
            return g.quantize(i, a)
        return super().encode("", np.asarray(a), dtype if dtype is not None else g.dtype(i))

    def _decode_index(self, i: int, a: np.ndarray) -> np.ndarray:
        g = self.graph
        a = np.asarray(a)
        return g.dequantize(i, a) if g.quant(i) is not None and a.dtype.kind in "iu" else a

    def encode(self, tid, a, dtype):
        return self._encode_index(self.index(tid), a, dtype)

    def decode(self, tid, a):
        return self._decode_index(self.index(tid), a)

    def encode_inputs(self, arrays):
        return [self._encode_index(i, a) for i, a in zip(self.graph.inputs, arrays)] + list(arrays[len(self.graph.inputs):])

    def decode_outputs(self, arrays):
        return [self._decode_index(i, a) for i, a in zip(self.graph.outputs, arrays)] + list(arrays[len(self.graph.outputs):])

    def cut(self, unit: Unit) -> bytes:
        return self.graph.cut(unit.nodes, [self.index(t) for t in unit.inputs],
                              [self.index(t) for t in unit.outputs])

    def single_op(self, i: int) -> bytes:
        g = self.graph
        op = g.ops[i]
        ins = [t for t in op.inputs if t >= 0 and g.constant(t) is None]
        return g.cut([i], ins, op.outputs)

    def meaning64(self, feeds, wanted, unit=None):
        g = self.graph
        if unit is None:
            f = {g.inputs[k]: v for k, v in enumerate(feeds.values())} if not all(
                ":" in k for k in feeds) else {self.index(k): v for k, v in feeds.items()}
            return g.evaluate64(f, [self.index(w) if ":" in w else g.outputs[self.output_names().index(w)]
                                    for w in wanted])
        f = {self.index(k): v for k, v in feeds.items()}
        return g.evaluate64(f, [self.index(w) for w in wanted], unit.nodes)


# ----------------------------------------------------------------------------- ExecuTorch

class PteArtifact(Artifact):
    format = PTE

    def __init__(self, path: str, data: bytes):
        super().__init__(path, data)
        self.program = None
        try:
            from executorch.exir._serialize._program import deserialize_pte_binary

            p = deserialize_pte_binary(data)
            self.program = getattr(p, "program", p)
        except ImportError:
            from ..deps import MissingDependencyError

            raise MissingDependencyError("executorch", "Reading a .pte file") from None
        except Exception as exc:
            raise ArtifactError(f"{path}: not a readable ExecuTorch program ({type(exc).__name__}: {exc})") from exc

    def plan(self):
        return self.program.execution_plan[0]

    def input_specs(self):
        return []  # taken from the loaded method at run time

    def output_names(self):
        return [f"output{i}" for i in range(len(self.plan().outputs))]

    def instructions(self) -> List[Dict[str, Any]]:
        """Each instruction: a kernel call (operator name) or a delegate call (backend id)."""
        plan = self.plan()
        out = []
        for c in plan.chains:
            for ins in c.instructions:
                a = ins.instr_args
                kind = type(a).__name__
                if kind == "KernelCall":
                    op = plan.operators[a.op_index]
                    nm = op.name + (f".{op.overload}" if getattr(op, "overload", "") else "")
                    out.append({"kind": "kernel", "op": nm})
                elif kind == "DelegateCall":
                    out.append({"kind": "delegate", "backend": plan.delegates[a.delegate_index].id})
        return out

    def node_count(self):
        return len(self.instructions())


# ----------------------------------------------------------------------------- Truestar export

class TsxArtifact(Artifact):
    format = TSX
    can_mean = True

    def __init__(self, path: str, data: bytes):
        super().__init__(path, data)
        from ..export import runner as R

        try:
            self.runner = R.load(path)
        except Exception as exc:
            raise ArtifactError(f"{path}: not a readable Truestar artifact ({type(exc).__name__}: {exc})") from exc

    def input_specs(self):
        import torch

        out = []
        for v in self.runner.program.inputs:
            shape = tuple(int(d) if isinstance(d, int) else -1 for d in v.type.shape)
            out.append((v.name, shape, torch.empty(0, dtype=v.type.dtype).numpy().dtype.type))
        return out

    def output_names(self):
        return [v.name for v in self.runner.program.outputs]

    def node_count(self):
        return len(self.runner.program.nodes)

    def meaning64(self, feeds, wanted, unit=None):
        import torch

        from ..export import runner as R

        r = R.Runner(self.runner.artifact, executor="reference", cast_outputs=False)
        ins = [torch.from_numpy(np.ascontiguousarray(v)) for v in feeds.values()]
        outs = r(*ins)
        return [o.detach().cpu().double().numpy() for o in outs]


def load(path: str) -> Artifact:
    fmt = detect(path)
    if fmt == COREML:
        raise ArtifactError(COREML_REASON)
    with open(path, "rb") as f:
        data = f.read()
    return {ONNX: OnnxArtifact, TFLITE: TfliteArtifact, PTE: PteArtifact, TSX: TsxArtifact}[fmt](path, data)
