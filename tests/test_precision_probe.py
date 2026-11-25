"""Per backend precision manifest in `truestar check` (truestar.precision_probe, research item R8)."""

import torch
import torch.nn as nn

from truestar import precision_probe as pp
from truestar.agent import report_to_agent
from truestar.check import check


def _round_bits(t: torch.Tensor, bits: int) -> torch.Tensor:
    m, e = torch.frexp(t.double())
    return torch.ldexp(torch.round(m * 2 ** bits) / 2 ** bits, e).to(t.dtype)


def test_cpu_float32_uses_all_24_bits():
    m = pp.manifest("cpu", torch.float32)
    assert set(m["ops"]) == {"linear", "bmm", "conv2d"}
    for e in m["ops"].values():
        assert e["input_bits"] == 24 and e["effective"] == "float32"
    assert m["flags"]["float32_matmul_precision"] in ("highest", "high", "medium")


def test_half_dtypes_report_their_bits_and_accumulator():
    for dt, bits in ((torch.float16, 11), (torch.bfloat16, 8)):
        m = pp.manifest("cpu", dt)
        for e in m["ops"].values():
            assert e["input_bits"] == bits and "accumulator" in e


def test_probe_names_tf32_and_narrow_accumulation(monkeypatch):
    """Tests can fail: kernels that round inputs to TF32 or accumulate in float16 must be named."""
    def tf32_linear(x, w):
        return (_round_bits(x, 11).double() @ _round_bits(w, 11).double()).float()

    def fp16_seq_sum(x, w):
        acc = torch.zeros((), dtype=torch.float16)
        for a, b in zip(x, w):
            acc = (acc + a * b).to(torch.float16)
        return acc

    monkeypatch.setattr(pp, "_ops", lambda: {"linear": tf32_linear})
    e = pp.manifest("cpu", torch.float32)["ops"]["linear"]
    assert e["input_bits"] == 11 and e["effective"] == "tf32"
    monkeypatch.setattr(pp, "_ops", lambda: {"bmm": fp16_seq_sum})
    e = pp.manifest("cpu", torch.float16)["ops"]["bmm"]
    assert e["accumulator"] == "dtype (ones lost)"


def test_check_report_and_agent_json_carry_the_manifest():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    rep = check(model, torch.randn(3, 8), backends=["cpu"])
    assert "precision: linear float32 (24 bits" in rep.to_text()
    doc = report_to_agent(rep)
    assert doc["backends"][0]["precision"]["ops"]["linear"]["input_bits"] == 24
