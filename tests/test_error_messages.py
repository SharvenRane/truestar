"""Error messages name the cause and what to do next (scripts/audit_codes.py lists the rest).

Each test triggers one refusal on the CPU and checks the message tells the user how to get past it,
not only what went wrong.
"""
import json
import textwrap

import pytest
import torch
from torch import nn

from truestar.cli import ModelFileError, _load, main


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(p)


def test_ts020_missing_get_inputs_says_what_each_function_returns(tmp_path):
    path = _write(tmp_path, "no_inputs.py", """
        import torch
        def get_model():
            return torch.nn.Linear(2, 2)
    """)
    with pytest.raises(ModelFileError) as e:
        _load(path)
    msg = str(e.value)
    assert "get_inputs()" in msg
    assert "nn.Module" in msg and "tuple of forward arguments" in msg


def test_ts021_fingerprint_compare_with_one_file_shows_the_usage(tmp_path, capsys):
    one = tmp_path / "a.json"
    one.write_text("{}", encoding="utf-8")
    assert main(["fingerprint", "compare", str(one), "--format", "agent"]) == 2
    doc = json.loads(capsys.readouterr().out)
    msg = doc["issues"][0]["message"]
    assert doc["issues"][0]["code"] == "TS021"
    assert "got 1" in msg and "truestar fingerprint compare A.json B.json" in msg


def test_ts021_conform_with_get_kwargs_says_how_to_pass_them(tmp_path, capsys):
    path = _write(tmp_path, "kw.py", """
        import torch
        class M(torch.nn.Module):
            def forward(self, x, scale=1.0):
                return x * scale
        def get_model():
            return M()
        def get_inputs():
            return torch.ones(2)
        def get_kwargs():
            return {"scale": 2.0}
    """)
    assert main(["conform", path, str(tmp_path / "unused.onnx"), "--format", "agent"]) == 2
    doc = json.loads(capsys.readouterr().out)
    msg = [i["message"] for i in doc["issues"] if i["code"] == "TS021"][0]
    assert "get_kwargs()" in msg and "get_inputs()" in msg and "positional" in msg


def test_ts1022_unknown_extra_step_lists_the_known_steps():
    from truestar.rotate import fold as R

    arch = R.Arch(hidden=8, heads=2, kv_heads=2, head_dim=4, inter=16, layers=1, tied=False)
    with pytest.raises(R.RotationError) as e:
        R.Plan(extra=("qk_pre_rop",)).matrices(arch)
    assert "qk_pre_rope" in str(e.value) and "mlp_across_act" in str(e.value)


def test_ts1022_mlp_across_act_with_r4_says_which_to_drop():
    from truestar.rotate import fold as R

    arch = R.Arch(hidden=8, heads=2, kv_heads=2, head_dim=4, inter=16, layers=1, tied=False)
    with pytest.raises(R.RotationError) as e:
        R.Plan(extra=("mlp_across_act",)).matrices(arch)
    assert "r4=None" in str(e.value)
    R.Plan(r4=None, extra=("mlp_across_act",)).matrices(arch)  # the advice works


def test_ts680_unknown_ranks_mode_names_the_choices():
    from truestar.parallel.check import parallel_check
    from truestar.parallel.layout import LayoutError

    with pytest.raises(LayoutError) as e:
        parallel_check(nn.Linear(2, 2), torch.zeros(2, 2), "dp=2", ranks="thread")
    assert "unknown ranks mode 'thread'" in str(e.value) and "threads, gloo or devices" in str(e.value)


def test_ts821_unknown_process_group_backend_names_the_choices():
    from truestar.parallel.devices import DevicesUnavailable, pick_backend

    with pytest.raises(DevicesUnavailable) as e:
        pick_backend("cpu", "mpi")
    assert "unknown process group backend 'mpi'" in str(e.value) and "auto, nccl or gloo" in str(e.value)


def test_ts1080_cuda_entry_on_a_cpu_tensor_names_what_it_got_and_the_portable_entry():
    from truestar.kernel.ordersum import OrderFreeSumError, order_free_sum, order_free_sum_cuda

    x = torch.ones(4, dtype=torch.float32)
    with pytest.raises(OrderFreeSumError) as e:
        order_free_sum_cuda(x)
    msg = str(e.value)
    assert "got torch.float32 on cpu" in msg and "order_free_sum(" in msg
    assert float(order_free_sum(x.reshape(-1))) == 4.0  # the advice runs on the CPU
