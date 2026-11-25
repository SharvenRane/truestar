"""LiteRT's unified NPU path as a conform target (`tflite:npu`, research item 6.2).

The LiteRT NPU path falls back to the CPU on its own and `is_fully_accelerated()` still says True
(LiteRT's Intel NPU page says so). The runtime reads LiteRT's own log to say what really ran:
TS1125 nothing reached an NPU, TS1124 part of the graph did, TS1126 an AOT compiled file's dispatch
partitions read from the file. This PC has no NPU, so the real run must give TS1125; the AOT and
partial cases are built from files and logs.
"""
import flatbuffers
import pytest
import torch
from torch import nn

pytest.importorskip("ai_edge_litert")

from truestar.conform import artifact as A  # noqa: E402
from truestar.conform import conform  # noqa: E402
from truestar.conform import runtimes as R  # noqa: E402
from truestar.conform.tflite_build import build_tflite  # noqa: E402

# Lines LiteRT 2.4.0 printed on this PC (Windows 11, Ryzen 9 9900X, no NPU), kept verbatim.
ENV_LOG = ("WARNING: [litert/runtime/accelerators/npu_registry.cc:34] NPU accelerator could not be "
           "loaded and registered: kLiteRtStatusErrorNotFound.\n")
MODEL_LOG = ("WARNING: [litert/compiler/plugin/compiler_plugin.cc:337] Failed to load plugin at: C:/x/"
             "vendors/intel_openvino/compiler/LiteRtCompilerPlugin_IntelOpenvino.dll with error: ERROR: "
             "[unknown:0]\n\u2514 Could not load shared library C:/x/LiteRtCompilerPlugin_IntelOpenvino.dll: "
             "Failed to load library 'C:\\x\\LiteRtCompilerPlugin_IntelOpenvino.dll': The specified "
             "module could not be found..\nWARNING: [litert/runtime/compiled_model.cc:2035] Failed to "
             "apply compiler plugins: No compiler plugin found\n")
# The partition line format, from the string in libLiteRt.dll
PART_LOG = ("INFO: [compiler_plugin.cc:690] Partitioned subgraph<0>, selected 2 ops, from a total of 3 "
            "ops. resulted in 1 partitions.\n")


def _mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).eval(), torch.randn(4, 16)


def test_log_facts_no_npu():
    f = R.npu_log_facts(ENV_LOG, MODEL_LOG)
    assert f["npu_registered"] is False and f["npu_error"] == "kLiteRtStatusErrorNotFound"
    assert "LiteRtCompilerPlugin_IntelOpenvino.dll did not load" in f["plugin_error"]
    assert f["partitions"] == []


def test_log_facts_partition_line():
    assert R.npu_log_facts("", PART_LOG)["partitions"] == [(0, 2, 3)]


def test_real_npu_request_on_this_pc(tmp_path):
    m, x = _mlp()
    p = tmp_path / "m.tflite"
    p.write_bytes(build_tflite(m, (x,)))
    rep = conform(m, (x,), str(p), runtimes=["tflite:npu", "tflite:cpu"])
    npu = next(r for r in rep.results if r.runtime == "tflite:npu")
    codes = [c for c, _ in npu.issues]
    facts = R.make("tflite:npu")
    if "TS1125" not in codes:  # a machine with a working NPU path: then it must name the partitions
        assert npu.residency["notes"] and any("selected" in n for n in npu.residency["notes"])
        return
    assert "TS628" in codes and len(npu.residency["off_target"]) == 3
    assert "is_fully_accelerated() returned True" in dict(npu.issues)["TS1125"]
    cpu = next(r for r in rep.results if r.runtime == "tflite:cpu")
    assert not [c for c, _ in cpu.issues if c.startswith("TS11")]
    assert facts.requested == "npu"


def _aot_like(data: bytes, dispatch_ops) -> bytes:
    """The same file with some operators replaced by LiteRT's DISPATCH_OP custom op (what an AOT
    compile leaves in place of the NPU partitions; the bytecode itself is not needed to read it)."""
    from ai_edge_litert import schema_py_generated as S

    m = S.ModelT.InitFromPackedBuf(bytearray(data), 0)
    oc = S.OperatorCodeT()
    oc.builtinCode = S.BuiltinOperator.CUSTOM
    oc.deprecatedBuiltinCode = S.BuiltinOperator.CUSTOM
    oc.customCode = "DISPATCH_OP"
    m.operatorCodes.append(oc)
    for i in dispatch_ops:
        m.subgraphs[0].operators[i].opcodeIndex = len(m.operatorCodes) - 1
        m.subgraphs[0].operators[i].builtinOptions = None
        m.subgraphs[0].operators[i].builtinOptionsType = 0
    b = flatbuffers.Builder(1 << 16)
    b.Finish(m.Pack(b), file_identifier=b"TFL3")
    return bytes(b.Output())


def _art(tmp_path, data):
    p = tmp_path / "aot.tflite"
    p.write_bytes(data)
    return A.load(str(p))


def test_aot_file_partitions_read_statically(tmp_path):
    m, x = _mlp()
    art = _art(tmp_path, _aot_like(build_tflite(m, (x,)), [0, 1]))
    rt = R.make("tflite:npu")
    s = R.Session(lambda f: [], {"fully_accelerated": True, "npu": {"npu_registered": True, "partitions": []}})
    r = rt.residency(art, s)
    codes = [c for c, _ in r.findings]
    assert codes == ["TS1126"] and "2 NPU dispatch partition(s), 1 operator(s) outside" in r.findings[0][1]
    assert [p.op for p in r.off_target] == ["FULLY_CONNECTED"]


def test_aot_file_without_dispatch_library_is_ts1125(tmp_path):
    m, x = _mlp()
    art = _art(tmp_path, _aot_like(build_tflite(m, (x,)), [0, 1, 2]))
    s = R.Session(lambda f: [], {"fully_accelerated": True,
                                 "npu": R.npu_log_facts(ENV_LOG, MODEL_LOG)})
    r = R.make("tflite:npu").residency(art, s)
    assert [c for c, _ in r.findings] == ["TS1126", "TS1125"] and len(r.off_target) == 3


def test_partial_partition_names_the_left_out_op(tmp_path, monkeypatch):
    m, x = _mlp()
    data = build_tflite(m, (x,))
    art = _art(tmp_path, data)
    rt = R.make("tflite:npu")
    names = [op.name for op in art.graph.ops]

    def one(d):  # the plugin takes the fully connected layers and leaves RELU on the CPU
        from truestar.conform.tflite_graph import TfliteGraph

        return (0, 1) if TfliteGraph(d).ops[0].name == "RELU" else (1, 1)

    monkeypatch.setattr(rt, "npu_selected", one)
    s = R.Session(lambda f: [], {"fully_accelerated": True, "npu": R.npu_log_facts("", PART_LOG)})
    r = rt.residency(art, s)
    assert [c for c, _ in r.findings] == ["TS1124"]
    assert [p.op for p in r.off_target] == ["RELU"] and names.count("FULLY_CONNECTED") == 2
