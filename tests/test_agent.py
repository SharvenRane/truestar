"""Agent interface: --format agent JSON, stable codes, and the MCP server over real stdio."""

import json
import os
import subprocess
import sys
import textwrap
import threading

import pytest
import torch

import truestar
from truestar import agent, codes
from truestar.cli import main
from truestar.compare import DIVERGENT, OK
from truestar.mcp_server import PROTOCOL_VERSIONS, TOOLS, sdk_available

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
_ACCEL = next((b for b in truestar.available_backends() if b.name != "cpu"), None)
ACCEL = _ACCEL.name if _ACCEL else None
ACCEL_TYPE = torch.device(_ACCEL.device).type if _ACCEL else None
needs_accel = pytest.mark.skipif(ACCEL is None, reason="needs a GPU backend")

CLEAN = """
import torch
def get_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
def get_inputs():
    torch.manual_seed(1)
    return torch.randn(5, 4)
"""

# A user class whose own forward is wrong on the accelerator: blame must point at this file.
FAULTY = """
import torch
from torch import nn


class Scaler(nn.Module):
    def forward(self, x):
        if x.device.type == "{accel}":
            return x * 1.5
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.bad = Scaler()
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(self.bad(self.fc1(x)))


def get_model():
    torch.manual_seed(0)
    return Net()


def get_inputs():
    torch.manual_seed(1)
    print("this print must not corrupt agent JSON")
    return torch.randn(16, 8)
"""


def _write(tmp_path, name, body):
    f = tmp_path / name
    f.write_text(textwrap.dedent(body))
    return f


def _agent(argv, capsys):
    code = main(argv)
    out = capsys.readouterr().out
    doc = json.loads(out)  # stdout must be exactly one JSON document
    return code, doc


def _assert_shape(doc):
    """Fields an agent depends on, with their types. Breaking any of these needs a schema bump."""
    assert doc["schema_version"] == agent.SCHEMA_VERSION == 1
    assert doc["tool"] == "truestar" and doc["command"] == "check"
    assert doc["status"] in ("ok", "suspect", "divergent", "error")
    assert doc["exit_code"] in (0, 1, 2)
    assert isinstance(doc["codes"], list) and isinstance(doc["issues"], list)
    assert isinstance(doc["backends"], list) and isinstance(doc["next_actions"], list)
    for i in doc["issues"]:
        assert set(i) >= {"code", "slug", "severity", "backend", "message"}
        assert i["code"] in codes.CODES
    for b in doc["backends"]:
        assert set(b) >= {"name", "device", "status", "codes", "output", "decisions",
                          "blamed_module", "repro", "modules_over_threshold", "error"}
        assert set(b["decisions"]) == {"total", "flips", "near_tie_flips"}
    for a in doc["next_actions"]:
        assert set(a) >= {"kind", "description", "command"}
    # Strict JSON: no NaN or Infinity anywhere.
    json.dumps(doc, allow_nan=False)


# ----------------------------------------------------------------------------- codes


def test_codes_are_stable_and_documented():
    # Released codes never change meaning; this list only grows.
    expected = {"TS001": "backend_unavailable", "TS002": "model_failed_on_backend",
                "TS003": "reference_degraded", "TS004": "module_set_differs",
                "TS005": "reference_failed", "TS010": "divergence", "TS011": "suspect",
                "TS012": "output_divergent_unattributed", "TS013": "prediction_flips",
                "TS014": "module_replay_failed", "TS020": "model_file_invalid",
                "TS021": "usage_error"}
    for c, s in expected.items():
        assert codes.slug(c) == s
    doc = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "AGENT_INTERFACE.md"), encoding="utf-8").read()
    for c in codes.CODES:
        assert c in doc, f"{c} missing from docs/AGENT_INTERFACE.md"
    with pytest.raises(ValueError):
        codes.Issue("TS999", "no such code")


# ----------------------------------------------------------------------------- agent JSON


def test_agent_json_clean_model(tmp_path, capsys):
    f = _write(tmp_path, "clean.py", CLEAN)
    code, doc = _agent(["check", str(f), "--backends", "cpu", "--no-repro", "--format", "agent"],
                       capsys)
    _assert_shape(doc)
    assert code == 0 and doc["exit_code"] == 0 and doc["status"] == "ok"
    assert doc["codes"] == [] and doc["next_actions"] == []
    assert doc["model"]["file"] == str(f.resolve()) and doc["model"]["dtype"] == "float32"
    (cpu,) = doc["backends"]
    assert cpu["name"] == "cpu" and cpu["status"] == "ok" and cpu["blamed_module"] is None
    assert cpu["output"]["rel_err"] < 1e-5 and cpu["modules_checked"] >= 2


def test_text_output_stays_the_default(tmp_path, capsys):
    f = _write(tmp_path, "clean.py", CLEAN)
    assert main(["check", str(f), "--backends", "cpu", "--no-repro"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("truestar check: clean") and "overall: OK" in out


@needs_accel
def test_agent_json_blames_user_source_line(tmp_path, capsys):
    f = _write(tmp_path, "faulty.py", FAULTY.format(accel=ACCEL_TYPE))
    code, doc = _agent(["check", str(f), "--backends", f"cpu,{ACCEL}", "--repro-dir",
                        str(tmp_path / "rep"), "--format", "agent"], capsys)
    _assert_shape(doc)
    assert code == 1 and doc["exit_code"] == 1 and doc["status"] == DIVERGENT
    assert "TS010" in doc["codes"]
    cpu, acc = doc["backends"]
    assert cpu["status"] == OK and cpu["codes"] == []
    assert acc["status"] == DIVERGENT and "TS010" in acc["codes"]
    blamed = acc["blamed_module"]
    assert blamed["name"] == "bad" and blamed["type"] == "Scaler"
    assert blamed["rel_err"] > blamed["threshold"] and blamed["ratio"] > 1
    src = blamed["source"]
    assert src["user_code"] is True and os.path.samefile(src["file"], f)
    line = f.read_text().splitlines()[src["line"] - 1]
    assert line.startswith("class Scaler"), line
    assert blamed["user_source"]["qualname"].endswith("Scaler")
    # The repro command in the JSON really reproduces, and the control run passes.
    assert acc["repro"] and os.path.isfile(acc["repro"]["path"])
    kinds = [a["kind"] for a in doc["next_actions"]]
    assert kinds[:2] == ["confirm_repro", "control_repro"] and "inspect_source" in kinds
    insp = next(a for a in doc["next_actions"] if a["kind"] == "inspect_source")
    assert insp["line"] == src["line"]
    env = dict(os.environ, PYTHONPATH=SRC)
    r1 = subprocess.run(acc["repro"]["command"], shell=True, capture_output=True, text=True, env=env)
    r0 = subprocess.run(acc["repro"]["control_command"], shell=True, capture_output=True, text=True,
                        env=env)
    assert r1.returncode == 1, r1.stdout + r1.stderr
    assert r0.returncode == 0, r0.stdout + r0.stderr


def _scale_on_accel(module, inputs, output):
    return output * 1.5 if output.device.type == ACCEL_TYPE else output


@needs_accel
def test_library_module_divergence_suggests_pin_and_upstream(tmp_path):
    """A fault inside a stock torch class: no user class to edit, so pin torch and report."""
    from torch import nn

    m = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    m[1].register_forward_hook(_scale_on_accel)  # a stock nn.Linear that is wrong on the GPU
    rep = truestar.check(m, torch.randn(16, 8), backends=["cpu", ACCEL], repro_dir=str(tmp_path))
    doc = agent.report_to_agent(rep)
    acc = doc["backends"][1]
    assert acc["blamed_module"]["name"] == "1" and acc["blamed_module"]["type"] == "Linear"
    assert acc["blamed_module"]["source"]["qualname"] == "torch.nn.modules.linear.Linear"
    assert acc["blamed_module"]["source"]["user_code"] is False
    kinds = [a["kind"] for a in doc["next_actions"]]
    assert "pin_torch" in kinds and "report_upstream" in kinds, doc["next_actions"]
    pin = next(a for a in doc["next_actions"] if a["kind"] == "pin_torch")
    assert pin["current"] == "torch==" + torch.__version__.split("+")[0]


def test_usage_errors_emit_agent_json_with_exit_2(tmp_path, capsys):
    code, doc = _agent(["check", str(tmp_path / "missing.py"), "--format", "agent"], capsys)
    _assert_shape(doc)
    assert code == 2 and doc["exit_code"] == 2 and doc["status"] == "error"
    assert doc["codes"] == ["TS020"]

    bad = _write(tmp_path, "nofn.py", "x = 1\n")
    code, doc = _agent(["check", str(bad), "--format", "agent"], capsys)
    assert code == 2 and doc["codes"] == ["TS020"] and "get_model" in doc["issues"][0]["message"]

    clean = _write(tmp_path, "clean.py", CLEAN)
    present = {b.name for b in truestar.available_backends()}
    missing = next(n for n in ("mps", "xpu", "cuda") if n not in present)
    code, doc = _agent(["check", str(clean), "--backends", missing, "--format", "agent"], capsys)
    assert code == 2 and doc["codes"] == ["TS001"] and "cpu" in doc["available"]
    assert doc["next_actions"][0]["command"].endswith("--format agent")

    code, doc = _agent(["check", str(clean), "--format", "agent", "--bogus"], capsys)
    assert code == 2 and doc["codes"] == ["TS021"]
    assert doc["model"]["file"] == str(clean.resolve())


def test_text_mode_usage_error_is_exit_2_with_code(tmp_path, capsys):
    assert main(["check", str(tmp_path / "missing.py")]) == 2
    assert "TS020" in capsys.readouterr().err


def test_model_failure_on_backend_is_ts002():
    class Breaks(torch.nn.Module):
        calls = 0  # on the class: check() runs deep copies of the model

        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)

        def forward(self, x):
            Breaks.calls += 1
            if Breaks.calls > 1:  # reference run passes, the backend run fails
                raise RuntimeError("kernel not implemented")
            return self.fc(x)

    rep = truestar.check(Breaks(), torch.randn(2, 4), backends=["cpu"])
    doc = agent.report_to_agent(rep)
    _assert_shape(doc)
    assert doc["status"] == DIVERGENT and doc["backends"][0]["codes"] == ["TS002"]
    assert doc["next_actions"][0]["kind"] == "inspect_failure"
    assert "TS002" in rep.to_text()


def test_flips_are_ts013():
    class Flip(torch.nn.Module):
        n = 0  # on the class: check() runs deep copies of the model

        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            Flip.n += 1
            y = self.fc(x)
            return y.flip(1) if Flip.n > 1 else y  # right on the reference run only

    rep = truestar.check(Flip(), torch.randn(16, 4), backends=["cpu"])
    doc = agent.report_to_agent(rep)
    _assert_shape(doc)
    b = doc["backends"][0]
    assert b["status"] != OK and b["decisions"]["flips"] > 0
    assert "TS013" in b["codes"]


def test_agent_document_has_no_nan_or_inf():
    doc = agent.error_to_agent("TS021", "x")
    doc["probe"] = agent._num(float("inf"))
    assert doc["probe"] is None and agent._num(float("nan")) is None
    agent.dumps(doc)
    with pytest.raises(ValueError):
        agent.dumps({"x": float("inf")})


# ----------------------------------------------------------------------------- MCP over stdio


class Client:
    """A minimal MCP client: one JSON object per line over the server's stdin and stdout."""

    def __init__(self, impl):
        env = dict(os.environ, PYTHONPATH=SRC, PYTHONIOENCODING="utf-8")
        self.p = subprocess.Popen([sys.executable, "-m", "truestar", "mcp", "--impl", impl],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, env=env)
        self.next_id = 0
        # A hung server fails the test instead of hanging the suite.
        self.watchdog = threading.Timer(300, self.p.kill)
        self.watchdog.daemon = True
        self.watchdog.start()

    def send(self, obj):
        self.p.stdin.write((json.dumps(obj) + "\n").encode())
        self.p.stdin.flush()

    def request(self, method, params=None):
        self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        line = self.p.stdout.readline()
        assert line, self.p.stderr.read().decode(errors="replace")
        resp = json.loads(line)
        assert resp["id"] == self.next_id, resp
        return resp

    def call(self, name, arguments):
        resp = self.request("tools/call", {"name": name, "arguments": arguments})
        return resp["result"]

    def close(self):
        self.watchdog.cancel()
        self.p.stdin.close()
        self.p.wait(timeout=30)
        self.p.stdout.close()
        self.p.stderr.close()


IMPLS = ["builtin"] + (["sdk"] if sdk_available() else [])


@pytest.fixture(params=IMPLS)
def client(request):
    c = Client(request.param)
    init = c.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                    "clientInfo": {"name": "pytest", "version": "0"}})
    assert init["result"]["protocolVersion"] in PROTOCOL_VERSIONS
    assert init["result"]["serverInfo"]["name"] == "truestar"
    assert "tools" in init["result"]["capabilities"]
    c.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    yield c
    c.close()


def test_mcp_lists_every_tool(client):
    tools = client.request("tools/list")["result"]["tools"]
    assert [t["name"] for t in tools] == ["list_backends", "check_model", "run_repro",
                                          "grad_check", "port_check", "invariance_check",
                                          "record", "diff", "export_model", "verify_kernel"]
    for t in tools:
        assert t["inputSchema"]["type"] == "object" and t["description"]
        assert set(t["inputSchema"].get("required", [])) <= set(t["inputSchema"]["properties"])


def test_mcp_list_backends(client):
    res = client.call("list_backends", {})
    assert res["isError"] is False
    names = [b["name"] for b in res["structuredContent"]["backends"]]
    assert names[0] == "cpu"
    assert json.loads(res["content"][0]["text"]) == res["structuredContent"]


def test_mcp_check_model_clean_returns_agent_json(client, tmp_path):
    f = _write(tmp_path, "clean.py", CLEAN)
    res = client.call("check_model", {"model_path": str(f), "backends": ["cpu"],
                                      "write_repro": False})
    assert res["isError"] is False, res
    doc = res["structuredContent"]
    _assert_shape(doc)
    assert doc["status"] == "ok" and doc["exit_code"] == 0


def test_mcp_check_model_missing_file_is_tool_error(client, tmp_path):
    res = client.call("check_model", {"model_path": str(tmp_path / "nope.py")})
    assert res["isError"] is True and res["structuredContent"]["codes"] == ["TS020"]


def test_mcp_rejects_bad_arguments_and_non_repro_scripts(client, tmp_path):
    res = client.call("check_model", {"model_path": 3})
    assert res["isError"] is True
    res = client.call("check_model", {"model_path": "x.py", "surprise": 1})
    assert res["isError"] is True
    evil = tmp_path / "repro.py"
    evil.write_text("print('not a truestar repro')\n")
    res = client.call("run_repro", {"repro_path": str(evil)})
    assert res["isError"] is True and "not written by truestar" in res["structuredContent"]["error"]


def test_mcp_unknown_tool_and_method(client):
    if client.p.args[-1] == "builtin":
        resp = client.request("tools/call", {"name": "nope", "arguments": {}})
        assert resp["error"]["code"] == -32602
        assert client.request("no/such/method")["error"]["code"] == -32601
    else:  # the official SDK reports an unknown tool as a tool error result
        assert client.call("nope", {})["isError"] is True
    # The server is still alive afterwards.
    assert client.request("ping")["result"] == {}


@needs_accel
def test_mcp_check_then_run_repro_round_trip(client, tmp_path):
    f = _write(tmp_path, "faulty.py", FAULTY.format(accel=ACCEL_TYPE))
    res = client.call("check_model", {"model_path": str(f), "backends": ["cpu", ACCEL]})
    doc = res["structuredContent"]
    _assert_shape(doc)
    assert res["isError"] is False and doc["status"] == DIVERGENT and doc["exit_code"] == 1
    acc = doc["backends"][1]
    assert acc["blamed_module"]["name"] == "bad"
    path = acc["repro"]["path"]
    assert os.path.dirname(os.path.dirname(path)) == str(tmp_path / "truestar_repro")
    on_acc = client.call("run_repro", {"repro_path": path})
    assert on_acc["isError"] is False and on_acc["structuredContent"]["reproduced"] is True, on_acc
    on_cpu = client.call("run_repro", {"repro_path": path, "device": "cpu"})
    assert on_cpu["structuredContent"]["passed"] is True
    assert on_cpu["structuredContent"]["reproduced"] is False


def test_builtin_parse_error_keeps_server_alive():
    c = Client("builtin")
    try:
        c.p.stdin.write(b"this is not json\n")
        c.p.stdin.flush()
        resp = json.loads(c.p.stdout.readline())
        assert resp["error"]["code"] == -32700 and resp["id"] is None
        assert c.request("ping")["result"] == {}
    finally:
        c.close()


def _fake_mcp(monkeypatch, server_cls):
    """Install a stand in `mcp` package whose lowlevel Server is `server_cls`."""
    import types as pytypes

    names = ("mcp", "mcp.server", "mcp.server.lowlevel", "mcp.server.stdio")
    mods = {n: pytypes.ModuleType(n) for n in names}
    mods["mcp"].server = mods["mcp.server"]
    mods["mcp.server"].lowlevel = mods["mcp.server.lowlevel"]
    mods["mcp.server"].stdio = mods["mcp.server.stdio"]
    mods["mcp.server.lowlevel"].Server = server_cls
    for n, m in mods.items():
        monkeypatch.setitem(sys.modules, n, m)


def test_mcp_sdk_api_detection(monkeypatch, capsys):
    """The sdk transport serves both official SDK shapes: 1.x registers handlers with the
    list_tools()/call_tool() decorators, 2.x takes them as on_list_tools=/on_call_tool= constructor
    arguments. A Server with neither falls back to builtin under auto, and --impl sdk names the cause."""
    from truestar import mcp_server

    class ServerOther:  # neither shape
        def __init__(self, name):
            pass

    _fake_mcp(monkeypatch, ServerOther)
    assert not mcp_server.sdk_available()
    assert "needs mcp 1.10 or later" in mcp_server.sdk_problem()
    monkeypatch.setattr(mcp_server, "serve_builtin", lambda: "builtin")
    monkeypatch.setattr(mcp_server, "serve_sdk", lambda: "sdk")
    assert mcp_server.serve("auto") == "builtin"
    assert mcp_server.serve("sdk") == 2
    err = capsys.readouterr().err
    assert "needs mcp 1.10 or later" in err and "--impl builtin" in err, err

    class Server2x:  # handlers are constructor arguments, no decorators
        def __init__(self, name, *, version="", on_list_tools=None, on_call_tool=None):
            pass

    class Server1x(ServerOther):
        def list_tools(self):
            pass

        def call_tool(self, *, validate_input=True):
            pass

    for cls, api in ((Server2x, "2.x"), (Server1x, "1.x")):
        _fake_mcp(monkeypatch, cls)
        assert mcp_server._sdk_api(cls) == api
        assert mcp_server.sdk_available() and mcp_server.sdk_problem() is None
        assert mcp_server.serve("auto") == "sdk"


def test_tool_schemas_match_handlers():
    from truestar.mcp_server import _HANDLERS

    assert [t["name"] for t in TOOLS] == list(_HANDLERS)


def test_mcp_server_and_codes_do_not_load_torch():
    """Each torch import with CUDA commits gigabytes; the long lived server must stay light."""
    code = ("import sys, truestar, truestar.mcp_server, truestar.cli, truestar.agent;"
            "truestar.agent.error_to_agent('TS021', 'x');"
            "assert 'torch' not in sys.modules, 'torch loaded'")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=SRC))
    assert p.returncode == 0, p.stderr
    from truestar.compare import OK as compare_ok

    assert agent.OK == compare_ok


def test_agent_json_includes_blamed_op_and_op_repro(tmp_path):
    import torch
    from torch import nn

    import truestar
    from truestar.agent import report_to_agent
    from truestar.testing import OpFault

    accel = next((b for b in truestar.available_backends() if b.name != "cpu"), None)
    if accel is None:
        import pytest
        pytest.skip("needs a GPU backend")
    dev_type = torch.device(accel.device).type
    lin = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
    seen = []
    from torch.utils._python_dispatch import TorchDispatchMode

    class Rec(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            seen.append(str(func))
            return func(*args, **(kwargs or {}))

    with torch.no_grad(), Rec():
        nn.Linear(4, 3).to(accel.device)(torch.randn(2, 4, device=accel.device))
    op_name = "aten.linear.default" if "aten.linear.default" in seen else "aten.addmm.default"
    with OpFault(op_name, dev_type, amount=0.5):
        rep = truestar.check(lin, torch.randn(8, 16), backends=["cpu", accel.name],
                             repro_dir=str(tmp_path))
    doc = report_to_agent(rep)
    b = next(x for x in doc["backends"] if x["name"] == accel.name)
    assert b["blamed_op"]["op"] == op_name
    assert b["blamed_op"]["ratio"] > 1
    assert b["op_repro"] and b["op_repro"]["path"].endswith("op_repro.py")
    kinds = [a["kind"] for a in doc["next_actions"]]
    assert kinds[0] == "confirm_op_repro"


# ----------------------------------------------------------------------------- MCP: newer commands

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")


def _example(tmp_path, *names):
    """Copy examples into tmp_path, so repros, recordings and artifacts land there."""
    import shutil

    out = []
    for n in names:
        shutil.copy(os.path.join(EXAMPLES, n), tmp_path / n)
        out.append(str(tmp_path / n))
    return out if len(out) > 1 else out[0]


def _envelope_ok(doc, command):
    assert doc["schema_version"] == 1 and doc["tool"] == "truestar"
    assert doc["command"] == command
    assert isinstance(doc["codes"], list) and isinstance(doc["next_actions"], list)
    json.dumps(doc, allow_nan=False)


def test_mcp_grad_check_blames_backward_only_bug(client, tmp_path):
    model = _example(tmp_path, "backward_model.py")
    res = client.call("grad_check", {"model_path": model, "backends": ["cpu"]})
    doc = res["structuredContent"]
    assert res["isError"] is False, res
    _envelope_ok(doc, "check")
    assert doc["mode"] == "backward" and doc["status"] == "divergent" and doc["exit_code"] == 1
    assert "TS180" in doc["codes"]
    cpu = doc["backends"][0]
    assert cpu["blamed_module"]["name"] == "act" and cpu["blamed_module"]["backward_only"] is True
    assert os.path.dirname(os.path.dirname(cpu["repro"]["path"])) == str(tmp_path / "truestar_repro")
    # The forward check of the same model is clean: the bug lives only in backward.
    fwd = client.call("check_model", {"model_path": model, "backends": ["cpu"],
                                      "write_repro": False})
    assert fwd["structuredContent"]["status"] == "ok"


def test_mcp_port_check_names_the_module_and_constant(client, tmp_path):
    ref, cand = _example(tmp_path, "port_reference.py", "port_candidate.py")
    res = client.call("port_check", {"reference_path": ref, "candidate_path": cand})
    doc = res["structuredContent"]
    assert res["isError"] is False, res
    _envelope_ok(doc, "port")
    assert doc["status"] == "divergent" and "TS160" in doc["codes"]
    assert doc["blamed"]["candidate"].startswith("blocks.0.norm")
    assert doc["blamed"]["op"]["op"] == "aten.add.Tensor"
    missing = client.call("port_check", {"reference_path": ref,
                                         "candidate_path": str(tmp_path / "nope.py")})
    assert missing["isError"] is True and missing["structuredContent"]["exit_code"] == 2


def test_mcp_verify_kernel_names_the_shape(client, tmp_path):
    ref, cand = _example(tmp_path, "kernel_reference.py", "kernel_candidate.py")
    res = client.call("verify_kernel", {"reference_path": ref, "candidate_path": cand,
                                        "device": "cpu"})
    doc = res["structuredContent"]
    assert res["isError"] is False, res
    _envelope_ok(doc, "kernel")
    assert doc["status"] == "divergent" and doc["codes"][0] == "TS200"
    assert doc["first_failure"]["kind"] == "shape"
    missing = client.call("verify_kernel", {"reference_path": ref,
                                            "candidate_path": str(tmp_path / "nope.py")})
    assert missing["isError"] is True and missing["structuredContent"]["exit_code"] == 2


def test_mcp_invariance_check_finds_batch_dependence(client, tmp_path):
    model = _example(tmp_path, "invariance_model.py")
    res = client.call("invariance_check", {"model_path": model, "backends": ["cpu"],
                                           "sizes": [2, 7], "rows": 1, "try_flags": False})
    doc = res["structuredContent"]
    assert res["isError"] is False, res
    _envelope_ok(doc, "invariance")
    assert doc["status"] == "not_invariant" and "TS052" in doc["codes"]
    assert doc["backends"][0]["first_module"]["key"].startswith("norm")
    bad = client.call("invariance_check", {"model_path": model, "sizes": [1]})
    assert bad["isError"] is True


def test_mcp_record_then_diff_round_trip(client, tmp_path):
    model = _example(tmp_path, "record_diff_model.py")
    rec = str(tmp_path / "before.tsrec")
    res = client.call("record", {"model_path": model, "output_path": rec, "backend": "cpu",
                                 "ops": "all"})
    assert res["isError"] is False, res
    _envelope_ok(res["structuredContent"], "record")
    assert res["structuredContent"]["recording"]["path"] == rec and os.path.isfile(rec)

    same = client.call("diff", {"recorded_path": rec})
    assert same["isError"] is False, same
    _envelope_ok(same["structuredContent"], "diff")
    assert same["structuredContent"]["status"] == "identical"

    # The same model with the simulated upgrade switched on, given as the replay model.
    upgraded = tmp_path / "upgraded.py"
    src = open(model, encoding="utf-8").read()
    upgraded.write_text(src.replace('os.environ.get("TRUESTAR_EXAMPLE_UPGRADE") == "1"', "True"))
    moved = client.call("diff", {"recorded_path": rec, "model_path": str(upgraded),
                                 "write_repro": False})
    doc = moved["structuredContent"]
    assert moved["isError"] is False, moved
    assert doc["status"] == "divergent" and doc["exit_code"] == 1 and "TS060" in doc["codes"]
    assert doc["blamed_module"]["key"] == "norm#0"

    second = str(tmp_path / "after.tsrec")
    assert client.call("record", {"model_path": model, "output_path": second,
                                  "backend": "cpu"})["isError"] is False
    two = client.call("diff", {"recorded_path": rec, "current_path": second})
    assert two["isError"] is False and two["structuredContent"]["status"] == "identical"


def test_mcp_record_and_diff_touch_only_truestar_files(client, tmp_path):
    model = _example(tmp_path, "record_diff_model.py")
    precious = tmp_path / "notes.txt"
    precious.write_text("keep me")
    res = client.call("record", {"model_path": model, "output_path": str(precious)})
    assert res["isError"] is True and "refusing to overwrite" in res["structuredContent"]["error"]
    assert precious.read_text() == "keep me"
    res = client.call("diff", {"recorded_path": str(precious)})
    assert res["isError"] is True and "not a recording" in res["structuredContent"]["error"]
    res = client.call("diff", {"recorded_path": str(tmp_path / "missing.tsrec")})
    assert res["isError"] is True


BRANCHY = """
import torch


class ValueBranch(torch.nn.Module):
    def forward(self, x):
        if x.mean() > 0:
            return torch.relu(x)
        return -x


def get_model():
    return ValueBranch()


def get_inputs():
    return 3 * torch.ones(2, 3)
"""


def test_mcp_export_model_writes_or_refuses(client, tmp_path):
    model = _example(tmp_path, "export_model.py")
    out = str(tmp_path / "small_cnn.tsx")
    res = client.call("export_model", {"model_path": model, "output_path": out})
    doc = res["structuredContent"]
    assert res["isError"] is False, res
    _envelope_ok(doc, "export")
    assert doc["status"] == "written" and doc["exit_code"] == 0
    art = doc["artifact"]
    assert art["path"] == out and os.path.getsize(out) == art["bytes"]
    assert len(art["program_hash"]) == 64 and art["evidence"]["verified"] is True
    assert [c["name"] for c in art["evidence"]["checks"]] == ["example", "second_batch"]
    assert doc["next_actions"][0]["kind"] == "run_artifact"
    # Writing again over its own artifact is allowed and gives the same program.
    again = client.call("export_model", {"model_path": model, "output_path": out})
    assert again["structuredContent"]["artifact"]["program_hash"] == art["program_hash"]

    branchy = _write(tmp_path, "branchy.py", BRANCHY)
    refused_out = tmp_path / "branchy.tsx"
    res = client.call("export_model", {"model_path": str(branchy),
                                       "output_path": str(refused_out)})
    doc = res["structuredContent"]
    assert res["isError"] is False and doc["status"] == "refused" and doc["exit_code"] == 1, doc
    assert "branch" in doc["stderr"] and not refused_out.exists()

    precious = tmp_path / "weights.bin"
    precious.write_bytes(b"not an artifact")
    res = client.call("export_model", {"model_path": model, "output_path": str(precious)})
    assert res["isError"] is True and precious.read_bytes() == b"not an artifact"
