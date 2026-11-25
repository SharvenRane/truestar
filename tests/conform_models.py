"""Models, artifact builders and injected faults for tests/test_conform.py and scripts/conform_study.py.

Every artifact is built here from a model defined in code (no downloads):

* ONNX: torch.onnx.export (dynamo exporter), then optional edits of the file itself:
  `gelu_tanh` (the exporter decomposes gelu as the tanh approximation, nodes keep their module
  scope), `int8_floor` (the quantize step of a fake INT8 layer rounds toward minus infinity instead
  of to nearest), `det_node` (adds a Det node that the GPU providers lack, for residency);
* TFLite: truestar.conform.tflite_build (faults `gelu_tanh`, `layernorm_eps_outside`);
* ExecuTorch: torch.export, to_edge_transform_and_lower with the XNNPACK partitioner.
"""

from __future__ import annotations

import contextlib
import copy
import io
import os
import warnings

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        return x + self.ln(torch.nn.functional.gelu(self.fc(x)))


class Classifier(nn.Module):
    """Two residual blocks with LayerNorm and gelu, then a 10 way head: logits [N, 10]."""

    def __init__(self, d_in=16, d=32, classes=10, cumsum=False):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.blocks = nn.Sequential(Block(d), Block(d))
        self.head = nn.Linear(d, classes)
        self.cumsum = cumsum

    def forward(self, x):
        h = self.blocks(torch.relu(self.inp(x)))
        if self.cumsum:
            h = torch.cumsum(h, dim=1) * 0.1
        return self.head(h)


class FakeQuant(nn.Module):
    """Per tensor INT8 fake quantization: round half to even, clamp to [-128, 127]."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        q = torch.clamp(torch.round(x / self.scale), -128, 127)
        return q * self.scale


class QuantClassifier(nn.Module):
    def __init__(self, d_in=16, d=32, classes=10):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.q1 = FakeQuant(0.05)
        self.fc = nn.Linear(d, d)
        self.q2 = FakeQuant(0.05)
        self.head = nn.Linear(d, classes)

    def forward(self, x):
        h = self.q1(torch.relu(self.inp(x)))
        h = self.q2(torch.relu(self.fc(h)))
        return self.head(h)


def make_classifier(seed=0, **kw):
    torch.manual_seed(seed)
    return Classifier(**kw).eval()


def example(batch=64, d_in=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, d_in, generator=g)


# ----------------------------------------------------------------------------- ONNX

def export_onnx(model, x, path):
    sink = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(sink):
        warnings.simplefilter("ignore")
        prog = torch.onnx.export(model, (x,), dynamo=True, verbose=False)
    proto = prog.model_proto
    with open(path, "wb") as f:
        f.write(proto.SerializeToString())
    return path


def _scope_of(node):
    return [p for p in node.metadata_props]


def edit_onnx(path, out_path, fault):
    import onnx
    from onnx import helper, numpy_helper
    import numpy as np

    m = onnx.load(path)
    g = m.graph
    nodes = list(g.node)
    new_nodes = []
    k = 0
    for n in nodes:
        if fault == "gelu_tanh" and n.op_type == "Gelu":
            x, y = n.input[0], n.output[0]
            meta = list(n.metadata_props)
            p = f"{n.name}_tanh"
            c = {"c0": 0.044715, "c1": float(np.sqrt(2.0 / np.pi)), "one": 1.0, "half": 0.5}
            names = {}
            for key, v in c.items():
                nm = f"{p}_{key}"
                g.initializer.append(numpy_helper.from_array(np.asarray(v, dtype=np.float32), nm))
                names[key] = nm
            steps = [("Mul", [x, x], "x2"), ("Mul", [f"{p}_x2", x], "x3"),
                     ("Mul", [f"{p}_x3", names["c0"]], "a"), ("Add", [x, f"{p}_a"], "b"),
                     ("Mul", [f"{p}_b", names["c1"]], "c"), ("Tanh", [f"{p}_c"], "t"),
                     ("Add", [f"{p}_t", names["one"]], "u"), ("Mul", [x, f"{p}_u"], "v"),
                     ("Mul", [f"{p}_v", names["half"]], None)]
            for op, ins, out in steps:
                nn_ = helper.make_node(op, ins, [y if out is None else f"{p}_{out}"], name=f"{p}_{op}_{k}")
                k += 1
                nn_.metadata_props.extend(meta)
                new_nodes.append(nn_)
            continue
        if fault == "int8_floor" and n.op_type == "Round":
            scopes = {q.key: q.value for q in n.metadata_props}.get("pkg.torch.onnx.name_scopes", "")
            if "q2" in scopes:
                f = helper.make_node("Floor", list(n.input), list(n.output), name=n.name + "_floor")
                f.metadata_props.extend(n.metadata_props)
                new_nodes.append(f)
                continue
        new_nodes.append(n)
    if fault == "det_node":
        # one Det node on a [N, 2, 2] view of the head output, added back as zero: the value does
        # not change, but the GPU providers have no Det kernel, so ORT places it on the CPU
        out = g.output[0].name
        mid = out + "_pre_det"
        for n in new_nodes:
            for i, o in enumerate(n.output):
                if o == out:
                    n.output[i] = mid
        shp = numpy_helper.from_array(np.asarray([-1, 2, 2], dtype=np.int64), "det_shape")
        sl = numpy_helper.from_array(np.asarray([0, 4], dtype=np.int64), "det_slice")
        zero = numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "det_zero")
        g.initializer.extend([shp, zero])
        st = numpy_helper.from_array(np.asarray([0], dtype=np.int64), "det_start")
        en = numpy_helper.from_array(np.asarray([4], dtype=np.int64), "det_end")
        ax = numpy_helper.from_array(np.asarray([1], dtype=np.int64), "det_axis")
        one = numpy_helper.from_array(np.asarray([-1, 1], dtype=np.int64), "det_col")
        g.initializer.extend([st, en, ax, one])
        meta = list(new_nodes[-1].metadata_props)
        extra = [helper.make_node("Slice", [mid, "det_start", "det_end", "det_axis"], ["det_in4"], name="det_slice_node"),
                 helper.make_node("Reshape", ["det_in4", "det_shape"], ["det_in"], name="det_reshape"),
                 helper.make_node("Det", ["det_in"], ["det_val"], name="det"),
                 helper.make_node("Reshape", ["det_val", "det_col"], ["det_col_v"], name="det_col_reshape"),
                 helper.make_node("Mul", ["det_col_v", "det_zero"], ["det_z"], name="det_times_zero"),
                 helper.make_node("Add", [mid, "det_z"], [out], name="det_add")]
        for e in extra:
            e.metadata_props.extend(meta)
        new_nodes.extend(extra)
    del g.node[:]
    g.node.extend(new_nodes)
    onnx.save(m, out_path)
    return out_path


# ----------------------------------------------------------------------------- TFLite and ExecuTorch

def build_tflite(model, x, path, faults=()):
    from truestar.conform.tflite_build import build_tflite as b

    with open(path, "wb") as f:
        f.write(b(model, (x,), faults))
    return path


def build_pte(model, x, path, delegate=True):
    scripts = os.path.join(os.path.dirname(os.__file__), "..", "Scripts")
    import shutil
    import sys

    if "FLATC_EXECUTABLE" not in os.environ:
        cand = os.path.join(os.path.dirname(sys.executable), "flatc.exe")
        if os.path.isfile(cand):
            os.environ["FLATC_EXECUTABLE"] = cand
        elif shutil.which("flatc"):
            os.environ["FLATC_EXECUTABLE"] = shutil.which("flatc")
    from executorch.exir import to_edge_transform_and_lower

    ep = torch.export.export(model.eval(), (x,))
    parts = []
    if delegate:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        parts = [XnnpackPartitioner()]
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        prog = to_edge_transform_and_lower(ep, partitioner=parts).to_executorch()
    with open(path, "wb") as f:
        f.write(prog.buffer)
    return path


def write_model_file(path, factory="make_classifier", kwargs="", batch=64):
    """A model.py for the CLI: get_model() and get_inputs() from this module."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(path, "w", encoding="utf-8") as f:
        f.write("import sys\n"
                f"sys.path.insert(0, {here!r})\n"
                "import conform_models as cm\n\n"
                f"def get_model():\n    return cm.{factory}({kwargs})\n\n"
                f"def get_inputs():\n    return cm.example({batch})\n")
    return path
