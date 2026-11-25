"""Localized faults: one wrong row in 1024, one wrong element, a wrong last column, a wrong region
of small values. The whole tensor relative error (the scoring before 2026-11-24) dilutes each of
them below its divergent bar; the per element and per slice measure of compare.py must catch
them, in float32, float16 and bfloat16, and stay quiet on the clean model.

Every check test runs twice: with the local measure (must be divergent and blame the fault) and
with it switched off (the old scoring, which must not reach divergent), so each test proves it
would have failed before the change.
"""

import pytest
import torch
from torch import nn

import truestar
from truestar import compare
from truestar.compare import DIVERGENT, OK, SUSPECT, diff_tensors, threshold_for
from truestar.testing import DeviceFault

_ACCEL = next((b for b in truestar.available_backends() if b.name != "cpu"), None)
ACCEL = _ACCEL.name if _ACCEL else None
ACCEL_TYPE = torch.device(_ACCEL.device).type if _ACCEL else None
needs_accel = pytest.mark.skipif(ACCEL is None, reason="needs a GPU backend (cuda, rocm, mps, xpu)")

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
KINDS = ["row", "element", "last_col", "region"]


class Wide(nn.Module):
    """A 1024 x 1024 layer whose first 32 channels are 32 times smaller than the rest."""

    def __init__(self, fault=None):
        super().__init__()
        self.fc1 = nn.Linear(256, 1024)
        g = torch.ones(1024)
        g[:32] = 1.0 / 32.0
        self.register_buffer("gain", g)
        self.fault = fault if fault is not None else nn.Identity()
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        return self.fc2(self.fault(self.fc1(x) * self.gain))


def _amount(kind, dk):
    """Half precision: the part is completely wrong (zeroed, or negated for the region). Float32:
    5 percent wrong, still 25x its threshold but diluted below the old divergent bar."""
    if dk == "fp32":
        return 0.05
    return -2.0 if kind == "region" else -1.0


def _fault(device_type, kind, dk):
    return DeviceFault(device_type, kind, _amount(kind, dk), 517 if kind == "row" else 32)


def _check(model, x, backend, old=False, monkeypatch=None):
    if old:
        monkeypatch.setattr(compare.Diff, "local_status", lambda self, threshold: OK)
    return truestar.check(model, x, backends=[backend], drill_ops=False).results[0]


# ----------------------------------------------------------------------------- diff_tensors


def _rounded(b, dtype):
    """What a correct kernel returns: the exact value rounded once to the working dtype."""
    return b.to(dtype).to(torch.float64)


def _inject(a, b, kind, dk):
    f = 1.0 + _amount(kind, dk)
    a = a.clone()
    if kind == "row":
        a[517] = b[517] * f
    elif kind == "element":
        i = int(b.abs().argmax())
        a.view(-1)[i] = b.view(-1)[i] * f
    elif kind == "last_col":
        a[:, -1] = b[:, -1] * f
    else:
        a[:, :32] = b[:, :32] * f
    return a


@pytest.mark.parametrize("dk", list(DTYPES))
@pytest.mark.parametrize("kind", KINDS)
def test_diff_catches_localized_fault_the_aggregate_misses(kind, dk):
    torch.manual_seed(0)
    b = torch.randn(1024, 1024, dtype=torch.float64)
    b[:, :32] /= 32.0
    dtype = DTYPES[dk]
    thr = threshold_for(dtype, False, None)
    a = _inject(_rounded(b, dtype), b, kind, dk)
    d = diff_tensors(a, b)
    assert d.aggregate_status(thr) != DIVERGENT, (d.rel_err, thr)  # the old scoring misses it
    assert d.status(thr) == DIVERGENT, (d.local_err, d.local_at)
    assert d.localized(thr)
    if kind == "row":
        assert d.local_at == "index 517 of dim 0 (1024 elements)", d.local_at
    if kind == "last_col":
        assert d.local_at == "index 1023 of dim 1 (1024 elements)", d.local_at
    if kind == "region":
        assert "of dim 1" in d.local_at and int(d.local_at.split()[1]) < 32, d.local_at


@pytest.mark.parametrize("dk", list(DTYPES))
@pytest.mark.parametrize("shape", [(1024, 1024), (4, 10), (2, 3, 17, 5), (7,), (1, 4096)])
def test_diff_correct_rounding_is_ok(shape, dk):
    """A correctly rounded result is OK in every shape, with small value regions and values that
    are near zero, including when it is only one or two elements per slice."""
    torch.manual_seed(1)
    b = torch.randn(*shape, dtype=torch.float64)
    b.view(-1)[: max(1, b.numel() // 32)] *= 1e-3
    dtype = DTYPES[dk]
    thr = threshold_for(dtype, False, None)
    d = diff_tensors(_rounded(b, dtype), b)
    assert d.status(thr) == OK, (d.rel_err, d.local_err, d.local_at, thr)
    assert d.local_err < compare.local_limit(thr) / 4


def test_diff_local_handles_nonfinite_and_zero_references():
    b = torch.randn(64, 64, dtype=torch.float64)
    b[3, 4] = float("inf")
    a = b.clone()
    a[10] = 0.0  # a zeroed row with an infinity elsewhere: the slow path measures it too
    d = diff_tensors(a, b)
    assert not d.nonfinite_mismatch
    assert d.status(1e-4) == DIVERGENT and d.local_at == "index 10 of dim 0 (64 elements)"
    z = torch.zeros(8, 8)
    assert diff_tensors(z, z).status(1e-4) == OK
    assert diff_tensors(torch.ones(1), torch.ones(1) * 2).local_err == 0.0  # one element: aggregate only


def test_diff_lists_keeps_the_worst_local_error_of_any_output():
    b1, b2 = torch.randn(256, 256), torch.randn(64, 64)
    a2 = b2.clone()
    a2[5] = 0.0
    big = b1 * 1.001  # worst aggregate, but uniform
    d = compare.diff_lists([big, a2], [b1, b2])
    assert d.rel_err == pytest.approx(diff_tensors(a2, b2).rel_err)
    assert d.local_err == pytest.approx(1.0, rel=1e-6)
    assert d.local_at == "output 1, index 5 of dim 0 (64 elements)"


# ----------------------------------------------------------------------------- truestar check


@pytest.mark.parametrize("dk", list(DTYPES))
@pytest.mark.parametrize("kind", KINDS)
def test_check_blames_localized_fault_on_cpu(kind, dk, monkeypatch):
    """`lowp` corrupts every run that is not float64, so the CPU is the faulty backend."""
    torch.manual_seed(0)
    x = torch.randn(1024, 256)
    model = Wide(_fault("lowp", kind, dk)).eval().to(DTYPES[dk])
    r = _check(model, x, "cpu")
    assert r.status == DIVERGENT, [i.message for i in r.issues]
    assert r.first_divergent is not None and r.first_divergent.name == "fault"
    assert [l.name for l in r.layers if l.status != OK] == ["fault"]
    msg = next(i.message for i in r.issues if i.code == "TS010")
    assert "local limit" in msg and " at " in msg, msg
    if kind == "row":
        assert "index 517 of dim 0" in msg, msg
    old = _check(model, x, "cpu", old=True, monkeypatch=monkeypatch)
    assert old.status in (OK, SUSPECT), [i.message for i in old.issues]


@needs_accel
@pytest.mark.parametrize("dk", list(DTYPES))
@pytest.mark.parametrize("kind", KINDS)
def test_check_blames_localized_fault_on_accelerator(kind, dk, monkeypatch):
    torch.manual_seed(0)
    x = torch.randn(1024, 256)
    model = Wide(_fault(ACCEL_TYPE, kind, dk)).eval().to(DTYPES[dk])
    r = _check(model, x, ACCEL)
    assert r.status == DIVERGENT, [i.message for i in r.issues]
    assert r.first_divergent is not None and r.first_divergent.name == "fault"
    old = _check(model, x, ACCEL, old=True, monkeypatch=monkeypatch)
    assert old.status in (OK, SUSPECT), [i.message for i in old.issues]


@pytest.mark.parametrize("dk", list(DTYPES))
def test_clean_wide_model_is_clean(dk):
    torch.manual_seed(0)
    rep = truestar.check(Wide().eval().to(DTYPES[dk]), torch.randn(1024, 256),
                         backends=["cpu", ACCEL] if ACCEL else ["cpu"], drill_ops=False)
    for r in rep.results:
        assert r.status == OK, rep.to_text()
        worst = max(l.isolated.local_err / compare.local_limit(l.threshold)
                    for l in r.layers if l.isolated is not None)
        assert worst < 0.5, worst


def test_agent_json_names_the_slice():
    from truestar.agent import report_to_agent

    torch.manual_seed(0)
    rep = truestar.check(Wide(_fault("lowp", "row", "bf16")).eval().to(torch.bfloat16),
                         torch.randn(1024, 256), backends=["cpu"], drill_ops=False)
    doc = report_to_agent(rep)
    b = doc["backends"][0]["blamed_module"]
    assert b["localized"] is True and b["local_at"].startswith("index 517 of dim 0"), b
    assert b["local_err"] > b["local_limit"]
