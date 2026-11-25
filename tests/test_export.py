"""Tests for the on disk IR format (truestar.ir.serialize) and truestar export (P7)."""

import copy
import datetime
import json
import os
import struct
import subprocess
import sys
import textwrap
from dataclasses import dataclass

import pytest
import sympy
import torch
from torch import nn

from truestar import ir
from truestar.cli import main as cli_main
from truestar.export import ExportError, VerificationError, export, inputs_sha256, load
from truestar.export.runner import Runner
from truestar.ir import reference
from truestar.ir import serialize as S
from truestar.ir.core import Node, Program, TensorType, Value, symbol

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
BATCH = ({0: torch.export.Dim("B", min=1, max=64)},)


def randomize_norms(model: nn.Module, seed: int = 0) -> nn.Module:
    """Non trivial batch norm statistics, so a normalization bug cannot hide behind the identity."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=g) * 0.1)
                m.running_var.copy_(torch.rand(m.running_var.shape, generator=g) + 0.5)
                m.weight.copy_(1 + 0.2 * torch.randn(m.weight.shape, generator=g))
                m.bias.copy_(0.1 * torch.randn(m.bias.shape, generator=g))
    return model


def tv(name: str) -> nn.Module:
    import torchvision
    torch.manual_seed(0)
    return randomize_norms(torchvision.models.get_model(name, weights=None)).eval()


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-300)).item()


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 5)

    def forward(self, x):
        y = self.pool(torch.relu(self.bn(self.conv(x)))).flatten(1)
        return self.fc(y)


def small_cnn() -> nn.Module:
    torch.manual_seed(0)
    return randomize_norms(SmallCNN()).eval()


@pytest.fixture(scope="module")
def resnet_prog():
    model = tv("resnet18")
    return model, ir.import_module(model, (torch.randn(2, 3, 64, 64),), dynamic_shapes=BATCH)


# ----------------------------------------------------------------------------- format: round trip


def test_round_trip_is_exact(resnet_prog):
    model, prog = resnet_prog
    blob = S.dumps(prog)
    art = S.loads(blob)
    back = art.program
    assert back.to_text() == prog.to_text()
    assert back.symbols == prog.symbols
    assert [p.name for p in back.params] == [p.name for p in prog.params]
    for p in prog.params:
        a, b = prog.param_data[p.name], back.param_data[p.name]
        assert a.dtype == b.dtype and torch.equal(a, b), p.name
    for n0, n1 in zip(prog.nodes, back.nodes):
        assert n0.op == n1.op and n0.attrs == n1.attrs and n0.source == n1.source
    x = torch.randn(3, 3, 64, 64, dtype=torch.float64)
    assert torch.equal(ir.run(prog, x)[0], ir.run(back, x)[0])
    assert S.dumps(back) == blob  # read then write gives the same bytes
    assert art.format_version == S.FORMAT_VERSION and art.program_hash == S.program_hash(prog)


def _all_types_program():
    """A program whose attributes use every encodable type, on an op whose rule ignores extras."""
    b = ir.Builder("attrs")
    x = b.input("x", torch.float32, ("B", 4))
    w = b.param("w", torch.randn(6, 4))
    h = b.param("h", torch.tensor([1.5, -2.0], dtype=torch.bfloat16))
    m = b.param("m", torch.tensor([True, False, True]))
    y = b.op("linear", x, w, None)
    y = b.op("reshape", y, shape=[2, 3 * symbol("B")])
    sub_b = ir.Builder("region")
    s_in = sub_b.input("t", torch.float32, (3,))
    sub = sub_b.build([sub_b.op("relu", s_in)])
    attrs = {
        "none": None, "t": True, "f": False, "int": -7, "big": 2 ** 70, "float": 0.1, "neg0": -0.0,
        "inf": float("inf"), "ninf": float("-inf"), "nan": float("nan"), "str": "a \"quoted\" ü",
        "list": [1, [2.5, None]], "tuple": (1, ("x",)), "dict": {"b": 1, "a": (2,)},
        "dtype": torch.bfloat16, "expr": 4 * symbol("B") + 1, "floor": sympy.floor(symbol("B") / 2),
        "ref": x, "region": sub, "tensor": torch.arange(6, dtype=torch.int32).reshape(2, 3),
        "empty": torch.empty(0, 3),
    }
    z = b.op("relu", y, **attrs)
    return b.build([z]), attrs, (h, m)


def _same(a, b):
    if isinstance(a, float) and a != a:
        return isinstance(b, float) and b != b
    if isinstance(a, float):
        return type(b) is float and a == b and str(a) == str(b)  # -0.0 keeps its sign
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, Program):
        return a.to_text() == b.to_text()
    if isinstance(a, Value):
        return isinstance(b, Value) and a.name == b.name and a.type.same_as(b.type)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, sympy.Basic):
        return isinstance(b, sympy.Basic) and sympy.simplify(a - b) == 0
    return type(a) is type(b) and a == b


def test_every_attribute_type_round_trips():
    prog, attrs, _ = _all_types_program()
    back = S.loads(S.dumps(prog)).program
    got = back.nodes[-1].attrs
    for k, v in attrs.items():
        assert _same(v, got[k]), k
    assert got["ref"] is back.inputs[0]  # a value reference resolves to the loaded value itself
    for p in prog.params:
        assert torch.equal(prog.param_data[p.name], back.param_data[p.name])
        assert back.param_data[p.name].dtype == prog.param_data[p.name].dtype
    assert str(back.nodes[1].output.type) == "f32[2, 3*B]"


def test_new_ops_serialize_without_code_changes():
    """An op this truestar does not define still writes and reads back; verification names it."""
    prog, _, _ = _all_types_program()
    last = prog.nodes[-1]
    out = Value("future", last.output.type, "op")
    prog.nodes.append(Node("future_scan", (last.output, None), {"steps": 3, "body": _all_types_program()[0]},
                           out, "x.py:1"))
    prog.outputs = [out]
    blob = S.dumps(prog)
    back = S.loads(blob, verify=False).program
    assert back.nodes[-1].op == "future_scan" and back.nodes[-1].attrs["steps"] == 3
    with pytest.raises(S.ArtifactError, match=r"future_scan.*does not define"):
        S.loads(blob)


@pytest.mark.parametrize("bad, kind", [(object(), "object"), ({1, 2}, "set"), (torch.device("cpu"), "device"),
                                       ({1: "x"}, "key"), (torch.Size([2]), "Size")])
def test_unencodable_attribute_is_a_named_error(bad, kind):
    b = ir.Builder("bad")
    x = b.input("x", torch.float32, (2,))
    b.op("relu", x)
    prog = b.build([b.nodes[-1].output])
    prog.nodes[-1].attrs["weird"] = bad
    with pytest.raises(S.SerializeError) as e:
        S.dumps(prog)
    msg = str(e.value)
    assert "'weird'" in msg and "relu node %relu" in msg


def test_unknown_structural_field_is_refused():
    @dataclass(eq=False)
    class RegionNode(Node):
        regions: tuple = ()

    b = ir.Builder("p")
    x = b.input("x", torch.float32, (2,))
    y = b.op("relu", x)
    prog = b.build([y])
    n = prog.nodes[0]
    prog.nodes[0] = RegionNode(n.op, n.inputs, n.attrs, n.output, n.source, regions=(1,))
    with pytest.raises(S.SerializeError, match=r"regions.*bump FORMAT_VERSION"):
        S.dumps(prog)


def test_metadata_must_be_plain_json(resnet_prog):
    _, prog = resnet_prog
    with pytest.raises(S.SerializeError, match="metadata"):
        S.dumps(prog, {"err": float("nan")})


# ----------------------------------------------------------------------------- format: determinism


def test_deterministic_bytes_across_imports():
    model = small_cnn()
    x = torch.randn(2, 3, 16, 16)
    a = S.dumps(ir.import_module(model, (x,), dynamic_shapes=BATCH))
    b = S.dumps(ir.import_module(copy.deepcopy(model), (torch.randn(2, 3, 16, 16),), dynamic_shapes=BATCH))
    assert a == b
    meta = S.dumps(ir.import_module(model, (x,), dynamic_shapes=BATCH), {"note": "x"})
    assert meta != a and S.loads(meta).program_hash == S.loads(a).program_hash


def test_program_hash_changes_with_any_weight():
    model = small_cnn()
    x = torch.randn(2, 3, 16, 16)
    h0 = S.program_hash(ir.import_module(model, (x,), dynamic_shapes=BATCH))
    with torch.no_grad():
        model.fc.bias[0] += 1e-7
    assert S.program_hash(ir.import_module(model, (x,), dynamic_shapes=BATCH)) != h0


# ----------------------------------------------------------------------------- format: bad files


@pytest.fixture(scope="module")
def small_blob():
    return S.dumps(ir.import_module(small_cnn(), (torch.randn(2, 3, 16, 16),), dynamic_shapes=BATCH))


def _with_version(blob: bytes, version: int) -> bytes:
    return blob[:8] + struct.pack("<I", version) + blob[12:]


def test_unknown_versions_are_loud(small_blob):
    with pytest.raises(S.ArtifactError, match=r"version 2.*newer truestar"):
        S.loads(_with_version(small_blob, 2))
    with pytest.raises(S.ArtifactError, match="unsupported artifact format version 0"):
        S.loads(_with_version(small_blob, 0))


def test_not_an_artifact(small_blob, tmp_path):
    with pytest.raises(S.ArtifactError, match="magic"):
        S.loads(b"PK\x03\x04" + small_blob[4:])
    with pytest.raises(S.ArtifactError, match="only 10 bytes"):
        S.loads(small_blob[:10])
    p = tmp_path / "torch.pt"
    torch.save({"w": torch.zeros(3)}, p)
    with pytest.raises(S.ArtifactError, match="not a truestar artifact"):
        S.load(str(p))


def test_corrupted_weight_is_detected_and_named(small_blob):
    hlen = struct.unpack_from("<Q", small_blob, 16)[0]
    header = json.loads(small_blob[24:24 + hlen])
    fc = next(p for p in header["program"]["params"] if p["name"] == "fc.weight")
    data_start = 24 + hlen + ((-(24 + hlen)) % 64)
    pos = data_start + fc["data"]["offset"] + 5
    bad = bytearray(small_blob)
    bad[pos] ^= 0x01  # one bit
    with pytest.raises(S.ArtifactError, match=r"corrupted.*@fc\.weight"):
        S.loads(bytes(bad))


def test_corrupted_header_and_truncation_are_detected(small_blob):
    bad = bytearray(small_blob)
    i = small_blob.index(b'"relu"')
    bad[i + 1:i + 5] = b"gelu"
    with pytest.raises(S.ArtifactError, match="corrupted"):
        S.loads(bytes(bad))
    with pytest.raises(S.ArtifactError, match="corrupted"):
        S.loads(small_blob[:-1])
    with pytest.raises(S.ArtifactError, match="corrupted"):
        S.loads(small_blob + b"\0")


def _refresh_checksum(body: bytes) -> bytes:
    import hashlib
    return body + hashlib.sha256(body).digest()


def test_tampering_with_a_valid_checksum_is_still_caught(small_blob):
    """A writer that changes weights and refreshes the file checksum still breaks the program hash."""
    hlen = struct.unpack_from("<Q", small_blob, 16)[0]
    data_start = 24 + hlen + ((-(24 + hlen)) % 64)
    body = bytearray(small_blob[:-32])
    body[data_start + 3] ^= 0xFF
    with pytest.raises(S.ArtifactError, match="program hash"):
        S.loads(_refresh_checksum(bytes(body)))
    # a header edit that keeps the hash field but changes the program
    body = small_blob[:-32].replace(b'"relu"', b'"gelu"', 1)
    with pytest.raises(S.ArtifactError, match="program hash"):
        S.loads(_refresh_checksum(body))


# ----------------------------------------------------------------------------- export


class _ThreeOperandEinsum(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(3, 5))
        self.c = nn.Parameter(torch.randn(6, 2))

    def forward(self, y):
        return torch.einsum("ij,bjk,kl->bil", self.a, y, self.c)


@pytest.mark.skipif(not torch.backends.opt_einsum.is_available(),
                    reason="the bug needs the opt_einsum package (installed with JAX)")
def test_export_keeps_batch_dynamic_through_a_three_operand_einsum(tmp_path):
    """With opt_einsum installed, torch.einsum's contraction path search compares sizes and used to
    specialize the batch to the example size; the exporter traces with the search off."""
    torch.manual_seed(0)
    model = _ThreeOperandEinsum().eval()
    path = tmp_path / "einsum.tsx"
    assert torch.backends.opt_einsum.enabled
    res = export(model, torch.randn(2, 5, 6), str(path))
    assert torch.backends.opt_einsum.enabled  # the setting is restored for the user
    assert [str(v.type) for v in res.program.inputs] == ["f32[B, 5, 6]"]
    run = load(str(path))
    for b in (7, 1):
        y = torch.randn(b, 5, 6)
        with torch.no_grad():
            want = model(y)
        assert rel(run(y)[0], want) < 1e-5


def test_export_dynamic_batch_preserved(tmp_path):
    """P7: export at batch 2, run at batch 7 and 1, same answers as PyTorch."""
    model = tv("resnet18")
    path = tmp_path / "r18.tsx"
    res = export(model, torch.randn(2, 3, 64, 64), str(path))
    assert [str(v.type) for v in res.program.inputs] == ["f32[B, 3, 64, 64]"]
    assert res.evidence["batch_range"] == [1, None]  # unbounded since 2026-11-24 (was [1, 1024])
    run = load(str(path))
    for b in (7, 1):
        x = torch.randn(b, 3, 64, 64)
        with torch.no_grad():
            want = model(x)
            want64 = copy.deepcopy(model).double()(x.double())
        got = run(x)[0]
        assert got.dtype == torch.float32 and got.shape == (b, 1000)
        assert rel(got, want) < 1e-5
        assert rel(Runner(run.artifact, cast_outputs=False)(x)[0], want64) < 1e-12


@pytest.mark.parametrize("name", ["resnet18", "mobilenet_v3_small", "mobilenet_v2", "efficientnet_b0",
                                  "squeezenet1_1", "mnasnet0_5"])
def test_torchvision_classifier_exports_and_round_trips(name, tmp_path):
    model = tv(name)
    path = tmp_path / f"{name}.tsx"
    res = export(model, torch.randn(2, 3, 64, 64), str(path))
    assert all(c["passed"] and c["max_rel_error"] < 1e-12 for c in res.evidence["checks"])
    run = load(str(path))
    assert run.artifact.program_hash == res.program_hash
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        want = model(x)
    assert rel(run(x)[0], want) < 1e-5


def test_googlenet_exports_with_an_input_dependent_check(tmp_path):
    """googlenet's init (std 0.01 convolutions, default batch norm statistics) gives the same output
    for every input, so its export check was WEAK; with batch norm statistics measured on random
    inputs (as scripts/export_zoo.py does) it exports with no warning and round trips."""
    import torchvision
    torch.manual_seed(0)
    model = torchvision.models.get_model("googlenet", weights=None).eval()
    res = export(model, torch.randn(2, 3, 64, 64), str(tmp_path / "weak.tsx"))
    assert res.evidence["warnings"]                       # the cause: input independent outputs
    for bn in model.modules():
        if isinstance(bn, nn.BatchNorm2d):
            bn.reset_running_stats()
            bn.momentum = None
    model.train()
    with torch.no_grad():
        for _ in range(4):
            model(torch.randn(8, 3, 64, 64))
    model.eval()
    path = tmp_path / "googlenet.tsx"
    res = export(model, torch.randn(2, 3, 64, 64), str(path))
    assert res.evidence["warnings"] == []
    assert all(c["passed"] and c["max_rel_error"] < 1e-12 for c in res.evidence["checks"])
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        want = model(x)
    assert want.std(0).max() > 1e-3
    assert rel(load(str(path))(x)[0], want) < 1e-4


def test_evidence_is_recorded(tmp_path):
    model = small_cnn()
    x = torch.randn(2, 3, 16, 16)
    path = tmp_path / "m.tsx"
    res = export(model, x, str(path))
    ev = S.load(str(path)).metadata["truestar_export"]
    assert ev == res.evidence and ev["verified"] is True
    assert ev["torch_version"] == torch.__version__
    assert ev["program_hash"] == res.program_hash == S.load(str(path)).program_hash
    datetime.datetime.fromisoformat(ev["created_utc"])
    first, second = ev["checks"]
    assert first["name"] == "example" and first["inputs_sha256"] == inputs_sha256([x])
    assert first["input_shapes"] == [[2, 3, 16, 16]] and second["input_shapes"] == [[7, 3, 16, 16]]
    assert max(first["max_rel_error"], second["max_rel_error"]) <= ev["tolerance"]


def test_same_model_same_program_hash(tmp_path):
    model = small_cnn()
    a = export(model, torch.randn(2, 3, 16, 16))
    b = export(copy.deepcopy(model), torch.randn(4, 3, 16, 16))
    assert a.program_hash == b.program_hash
    assert S.dumps(a.program) == S.dumps(b.program)


def test_disagreeing_program_is_never_written(tmp_path, monkeypatch):
    """Never ship an unverified artifact: a wrong reference relu makes export refuse and write nothing."""
    orig = reference.IMPLS["relu"]
    monkeypatch.setitem(reference.IMPLS, "relu", lambda attrs, args, shape: orig(attrs, args, shape) * 1.001)
    path = tmp_path / "bad.tsx"
    with pytest.raises(VerificationError, match="disagrees with PyTorch") as e:
        export(small_cnn(), torch.randn(2, 3, 16, 16), str(path))
    assert not path.exists() and not list(tmp_path.iterdir())
    assert all(not c["passed"] for c in e.value.checks)


def test_disagreement_only_at_another_batch_is_caught(tmp_path, monkeypatch):
    """A program that is right at the export batch and wrong elsewhere fails the second batch check."""
    orig = reference.IMPLS["linear"]

    def wrong_beyond_2(attrs, args, shape):
        out = orig(attrs, args, shape)
        return out if shape[0] <= 2 else out + 1.0

    monkeypatch.setitem(reference.IMPLS, "linear", wrong_beyond_2)
    path = tmp_path / "bad.tsx"
    with pytest.raises(VerificationError) as e:
        export(small_cnn(), torch.randn(2, 3, 16, 16), str(path))
    assert [c["passed"] for c in e.value.checks] == [True, False]
    assert not path.exists()


class ValueBranch(nn.Module):
    def forward(self, x):
        if x.mean() > 0:
            return torch.relu(x)
        return -x


class BatchBranch(nn.Module):
    def forward(self, x):
        if x.shape[0] > 4:
            return x * 2
        return x + 1


class HardBatch(nn.Module):
    def forward(self, x):
        return x.reshape(2, -1).sum(1)


class CondBranch(nn.Module):
    def forward(self, x):
        return torch.cond(x.mean() > 0, lambda t: torch.relu(t), lambda t: -t, (x,))


def test_value_branch_is_refused_naming_the_line(tmp_path):
    path = tmp_path / "v.tsx"
    with pytest.raises(ExportError) as e:
        export(ValueBranch(), torch.full((2, 3), 3.0), str(path))
    msg = str(e.value)
    assert "branch on a tensor's value" in msg and "if x.mean() > 0:" in msg and "test_export.py:" in msg
    assert not path.exists()


def test_batch_branch_is_refused_naming_the_line(tmp_path):
    path = tmp_path / "b.tsx"
    with pytest.raises(ExportError) as e:
        export(BatchBranch(), torch.full((2, 3), 3.0), str(path))
    msg = str(e.value)
    assert "if x.shape[0] > 4:" in msg and "batch" in msg
    assert not path.exists()


def test_hard_coded_batch_is_refused():
    with pytest.raises(ExportError, match="batch"):
        export(HardBatch(), torch.randn(2, 6))


@pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
                    reason="torch 2.11 dynamo crashes on torch.cond when CUDA_VISIBLE_DEVICES is empty; use -1")
def test_captured_branch_is_exported_correctly_or_refused(tmp_path):
    """torch.cond is captured by torch.export and, since 2026-11-24, imported as an IR cond: both
    sides run correctly after export (the refusal path this test also allowed is gone)."""
    path = tmp_path / "c.tsx"
    res = export(CondBranch(), torch.full((2, 3), 3.0), str(path))
    assert res.evidence["branches"][0]["other_side"]["passed"]
    run = load(str(path))
    for x in (torch.full((7, 3), 3.0), torch.full((7, 3), -3.0)):
        assert torch.equal(run(x)[0], CondBranch()(x))


def test_static_batch_is_fixed_and_enforced(tmp_path):
    model = small_cnn()
    path = tmp_path / "s.tsx"
    res = export(model, torch.randn(2, 3, 16, 16), str(path), dynamic_batch=False)
    assert str(res.program.inputs[0].type) == "f32[2, 3, 16, 16]" and len(res.evidence["checks"]) == 1
    with pytest.raises(ir.ShapeError):
        load(str(path))(torch.randn(3, 3, 16, 16))


def test_batch_of_one_example_is_rejected():
    with pytest.raises(ExportError, match="batch is 1"):
        export(small_cnn(), torch.randn(1, 3, 16, 16))


def test_runner_checks_input_dtype_and_range(tmp_path):
    path = tmp_path / "m.tsx"
    export(small_cnn(), torch.randn(2, 3, 16, 16), str(path), max_batch=8)
    run = load(str(path))
    with pytest.raises(ir.IRError, match="float64 but the artifact declares f32"):
        run(torch.randn(2, 3, 16, 16, dtype=torch.float64))
    with pytest.raises(ir.ShapeError, match="outside its range"):
        run(torch.randn(9, 3, 16, 16))


# ----------------------------------------------------------------------------- standalone runner


MODEL_SRC = '''
import torch
from torch import nn


class SecretNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        y = torch.nn.functional.hardswish(self.bn(self.conv(x)))
        return self.fc(y.mean(dim=(2, 3))).softmax(-1)


def get_model():
    torch.manual_seed(0)
    m = SecretNet().eval()
    with torch.no_grad():
        m.bn.running_mean.uniform_(-0.2, 0.2)
        m.bn.running_var.uniform_(0.5, 1.5)
    return m


def get_inputs():
    torch.manual_seed(1)
    return torch.randn(2, 3, 12, 12)
'''

RUNNER_SCRIPT = r'''
import importlib.abc, sys
BLOCKED = {"secretnet_model_src", "torchvision"}

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked import of {name}")
        return None

sys.meta_path.insert(0, Block())
import torch
def _forbidden(*a, **k):
    raise RuntimeError("torch.export / TorchScript must not be used by the runner")
torch.export.export = torch.export.load = torch.jit.load = _forbidden
from truestar.export.runner import load
run = load(sys.argv[1])
x = torch.load(sys.argv[2], weights_only=True)
torch.save(run(x), sys.argv[3])
assert not [m for m in sys.modules if m.split(".")[0] in BLOCKED]
print("ran", run.signature())
'''


def _load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _subprocess_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    env.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    if env["CUDA_VISIBLE_DEVICES"] == "":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def test_artifact_runs_in_a_fresh_process_without_the_model_class(tmp_path):
    src = tmp_path / "secretnet_model_src.py"
    src.write_text(MODEL_SRC)
    mod = _load_module(str(src), "secretnet_model_src")
    model = mod.get_model()
    path = tmp_path / "secret.tsx"
    export(model, mod.get_inputs(), str(path))
    x = torch.randn(5, 3, 12, 12)
    with torch.no_grad():
        want = model(x)
    sys.modules.pop("secretnet_model_src", None)
    src.unlink()  # the model's source is gone before the runner starts
    work = tmp_path / "elsewhere"
    work.mkdir()
    torch.save(x, work / "in.pt")
    script = work / "run_it.py"
    script.write_text(RUNNER_SCRIPT)
    proc = subprocess.run([sys.executable, str(script), str(path), "in.pt", "out.pt"], cwd=str(work),
                          env=_subprocess_env(), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "ran SecretNet(%x: f32[B, 3, 12, 12]) -> (f32[B, 3])" in proc.stdout
    (got,) = torch.load(work / "out.pt", weights_only=True)
    assert got.shape == (5, 3) and rel(got, want) < 1e-6


def test_cli_export_and_run(tmp_path, capsys):
    src = tmp_path / "cli_model.py"
    src.write_text(MODEL_SRC)
    path = tmp_path / "cli.tsx"
    assert cli_main(["export", str(src), "-o", str(path)]) == 0
    assert "check second_batch" in capsys.readouterr().out
    torch.save([torch.randn(6, 3, 12, 12)], tmp_path / "in.pt")
    assert cli_main(["run", str(path), str(tmp_path / "in.pt"), "-o", str(tmp_path / "out.pt")]) == 0
    assert "output 0: float32[6, 3]" in capsys.readouterr().out
    assert torch.load(tmp_path / "out.pt", weights_only=True)[0].shape == (6, 3)
    bad = tmp_path / "branchy.py"
    bad.write_text(textwrap.dedent('''
        import torch
        class M(torch.nn.Module):
            def forward(self, x):
                if x.sum() > 0:
                    return x
                return -x
        def get_model():
            return M()
        def get_inputs():
            return torch.ones(2, 3)
    '''))
    assert cli_main(["export", str(bad), "-o", str(tmp_path / "no.tsx")]) == 1
    assert "if x.sum() > 0:" in capsys.readouterr().err
    assert not (tmp_path / "no.tsx").exists()


def test_vacuous_checks_are_flagged_not_hidden():
    """A model whose outputs are all zero passes any comparison; the evidence must say so."""
    model = small_cnn()
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
    res = export(model, torch.randn(2, 3, 16, 16))
    assert res.evidence["warnings"] and "exactly zero" in res.evidence["warnings"][0]
    assert "WARNING weak check" in res.to_text()
    assert export(small_cnn(), torch.randn(2, 3, 16, 16)).evidence["warnings"] == []


# ----------------------------------------------------------------------------- native front end programs

import truestar  # noqa: E402
from truestar import Tensor, ops  # noqa: E402


@truestar.function
def native_branch(x: Tensor["B", 8], w: Tensor[8, 4]) -> Tensor["B", 4]:
    h = x @ w
    if h.sum() > 0:
        h = ops.relu(h)
        k = 1.0
    else:
        h = -h * 0.5
        k = 3.0
    return h * k


@truestar.function
def native_loop(x: Tensor["B", "T", 8], wx: Tensor[8, 16], wh: Tensor[16, 16]) -> Tensor["B", 16]:
    h = ops.zeros((x.shape[0], 16), dtype=x.dtype)
    for t in range(1, x.shape[1]):
        row = x[:, t] - x[:, t - 1]
        if row.sum() > 0:
            h = ops.tanh(row @ wx + h @ wh)
        else:
            h = h * 0.5
    return h


def _regions(prog):
    out = []
    for r in prog.regions():
        out.append(r)
        out.extend(_regions(r))
    return out


def _round_trip_native(fn):
    prog = fn.program
    blob = S.dumps(prog)
    back = S.loads(blob).program
    assert back.to_text() == prog.to_text()
    assert S.dumps(back) == blob
    assert len(_regions(back)) == len(_regions(prog)) > 0
    for n0, n1 in zip(prog.nodes, back.nodes):
        assert [v.name for v in n0.outputs] == [v.name for v in n1.outputs]
    return prog, back


def test_native_data_dependent_if_round_trips():
    prog, back = _round_trip_native(native_branch)
    assert back.op_counts()["cond"] == 1
    cond = next(n for n in back.nodes if n.op == "cond")
    assert isinstance(cond.attrs["then_branch"], Program) and isinstance(cond.attrs["else_branch"], Program)
    torch.manual_seed(0)
    w = torch.randn(8, 4, dtype=torch.float64)
    taken = set()
    for x in (torch.ones(3, 8, dtype=torch.float64), -torch.ones(5, 8, dtype=torch.float64)):
        taken.add(bool((x @ w).sum() > 0))
        a, b = ir.run(prog, x, w), ir.run(back, x, w)
        assert all(torch.equal(p, q) for p, q in zip(a, b))
    assert taken == {True, False}


def test_native_symbolic_loop_round_trips():
    prog, back = _round_trip_native(native_loop)
    loop = next(n for n in back.nodes if n.op == "loop")
    trip = loop.attrs["trip_count"]
    assert isinstance(trip, sympy.Basic) and trip.free_symbols == {symbol("T")}
    orig = next(n for n in prog.nodes if n.op == "loop").attrs["trip_count"]
    assert sympy.simplify(trip - orig) == 0
    assert len(loop.outputs) == len(next(n for n in prog.nodes if n.op == "loop").outputs)
    torch.manual_seed(1)
    wx, wh = torch.randn(8, 16, dtype=torch.float64), 0.2 * torch.randn(16, 16, dtype=torch.float64)
    for B, T in ((1, 1), (2, 4), (3, 9)):
        x = torch.randn(B, T, 8, dtype=torch.float64)
        a, b = ir.run(prog, x, wx, wh), ir.run(back, x, wx, wh)
        assert all(torch.equal(p, q) for p, q in zip(a, b))


def test_native_program_runs_in_a_fresh_process(tmp_path):
    """A native function with a loop and a data dependent if, saved and run with no front end source."""
    path = tmp_path / "native.tsx"
    S.save(native_loop.program, str(path))
    torch.manual_seed(2)
    x, wx, wh = torch.randn(2, 6, 8), torch.randn(8, 16), 0.2 * torch.randn(16, 16)
    want = ir.run(native_loop.program, x, wx, wh)[0]
    torch.save([x, wx, wh], tmp_path / "in.pt")
    code = ("import sys, torch\nfrom truestar.export.runner import load\n"
            "r = load(sys.argv[1])\nx = torch.load(sys.argv[2], weights_only=True)\n"
            "torch.save(r(*x), sys.argv[3])\n"
            "assert 'test_export' not in sys.modules\n")
    proc = subprocess.run([sys.executable, "-c", code, str(path), str(tmp_path / "in.pt"), str(tmp_path / "out.pt")],
                          cwd=str(tmp_path), env=_subprocess_env(), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    (got,) = torch.load(tmp_path / "out.pt", weights_only=True)
    assert got.dtype == torch.float32 and rel(got, want) < 1e-6
