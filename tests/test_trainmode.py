"""Training mode in the checkers (truestar.trainmode): check, grad_check and invariance with
train=True. Clean models report clean, keyed dropout masks are the same on every backend, the running
statistic update is compared, and training only faults that eval mode cannot see are caught."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

import truestar
from truestar import trainmode
from truestar.check import ReferenceFailed
from truestar.invariance import invariance
from truestar.testing import OpFault

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
BN_FWD = {"cpu": "aten.native_batch_norm.default", "cuda": "aten.cudnn_batch_norm.default"}
BN_BWD = {"cpu": ("aten.native_batch_norm_backward.default", lambda a, k: bool(a[7])),
          "cuda": ("aten.cudnn_batch_norm_backward.default", None)}
TRAINING = lambda a, k: bool(a[5])  # noqa: E731 (batch norm forward: the training flag)


def conv_bn():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.Dropout(0.3), nn.Flatten(),
                         nn.Linear(8 * 6 * 6, 10), nn.BatchNorm1d(10))


def encoder():
    torch.manual_seed(0)
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(32, 4, 64, dropout=0.1, batch_first=True), 2)


class SdpaAttention(nn.Module):
    """need_weights=False: nn.MultiheadAttention calls scaled_dot_product_attention with dropout_p."""

    def __init__(self):
        super().__init__()
        self.a = nn.MultiheadAttention(32, 4, dropout=0.2, batch_first=True)

    def forward(self, x):
        return self.a(x, x, x, need_weights=False)[0]


X_IMG = torch.randn(4, 3, 8, 8, generator=torch.Generator().manual_seed(1))
X_SEQ = torch.randn(4, 5, 32, generator=torch.Generator().manual_seed(2))


@pytest.mark.parametrize("name,make,x", [("conv_bn", conv_bn, X_IMG), ("encoder", encoder, X_SEQ),
                                         ("sdpa_dropout", SdpaAttention, X_SEQ)])
def test_clean_models_report_clean_in_training_mode(name, make, x):
    model = make().train()
    before = copy.deepcopy(model.state_dict())
    rep = truestar.check(model, x, train=True)
    assert rep.status == "ok", rep.to_text()
    assert any("training mode" in n for n in rep.notes)
    g = truestar.grad_check(model, x, train=True)
    assert g.status == "ok", g.to_text()
    # the caller's model is never changed, not even its running statistics
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_keyed_masks_are_the_same_on_every_backend():
    model = conv_bn().train()
    outs = []
    for dev in DEVICES:
        m = copy.deepcopy(model)[:4].to(dev)   # up to and through the dropout
        with trainmode.TrainMode(seed=5) as tm:
            tm.attach(m)
            with torch.no_grad():
                h = m(X_IMG.to(dev))
        outs.append(h.cpu())
        assert tm.masks == 1
    for o in outs[1:]:
        assert torch.equal(o == 0, outs[0] == 0)
        assert torch.allclose(o, outs[0], atol=1e-5)
    with trainmode.TrainMode(seed=6) as tm:     # another seed, another mask
        m = tm.attach(copy.deepcopy(model)[:4])
        with torch.no_grad():
            other = m(X_IMG)
    assert not torch.equal(other == 0, outs[0] == 0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("target", ["output", "state"])
def test_training_only_forward_fault_is_caught_in_training_mode_only(device, target):
    """A batch norm kernel wrong only when training=True: in its output, or in its running statistic
    update (the forward output stays right). Eval mode never runs that path, so it reports clean."""
    model = conv_bn()
    found = {}
    for train in (False, True):
        fault = OpFault(BN_FWD[device], device, 0.1, when=TRAINING, target=target, skip_float64=True)
        with fault:
            rep = truestar.check(model, X_IMG, backends=[device], train=train)
        r = rep.results[0]
        found[train] = (r.status, r.first_divergent.key if r.first_divergent else None, fault.hits)
        if train:
            if device == "cpu" and target == "output":
                assert r.first_divergent_op is not None and r.first_divergent_op.op == BN_FWD["cpu"]
    assert found[False] == ("ok", None, 0)
    assert found[True][:2] == ("divergent", "1#0") and found[True][2] > 0


@pytest.mark.parametrize("device", DEVICES)
def test_training_only_backward_fault_is_caught_by_grad_check(device):
    op, when = BN_BWD[device]
    model = conv_bn()
    found = {}
    for train in (False, True):
        with OpFault(op, device, 0.1, when=when, skip_float64=True):
            rep = truestar.grad_check(model, X_IMG, backends=[device], train=train)
        r = rep.results[0]
        found[train] = (r.status, r.first_divergent.key if r.first_divergent else None, r.codes)
        if train and device == "cpu":
            assert r.first_divergent_op is not None and r.first_divergent_op.op == op
    assert found[False] == ("ok", None, [])
    assert found[True][0] == "divergent" and found[True][1] in ("1#0", "6#0") and "TS180" in found[True][2]


def test_unkeyed_random_ops_are_refused_not_compared():
    class Noisy(nn.Module):
        def forward(self, x):
            return x * torch.bernoulli(torch.full_like(x, 0.5))

    with pytest.raises(ReferenceFailed, match="random generator"):
        truestar.check(nn.Sequential(nn.Linear(4, 4), Noisy()), torch.randn(3, 4), train=True)
    # eval mode keeps its old behaviour for models with no randomness
    assert truestar.check(nn.Sequential(nn.Linear(4, 4)), torch.randn(3, 4)).status == "ok"


def test_invariance_in_training_mode():
    rep = invariance(encoder(), X_SEQ, backends=["cpu"], train=True, try_flags=False)
    r = rep.results[0]
    assert r.status != "error", r.error
    assert "TS052" not in r.codes          # keyed per row masks: no false batch dependence
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(16, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 4))
    rep = invariance(mlp, torch.randn(8, 16), backends=["cpu"], train=True, try_flags=False)
    r = rep.results[0]
    assert r.status == "batch_dependent" and "TS052" in r.codes
    assert any("batch norm" in i.message for i in r.issues)


# ----------------------------------------------------------------------------- tests that the tests can fail


def test_broken_keying_is_caught(monkeypatch):
    """Keys that depend on the dtype (so the float64 reference draws another mask than the run under
    test) must make a clean dropout model look divergent: the clean result above is not luck."""
    real = trainmode.TrainMode.dropout

    def by_dtype(self, input, p=0.5, training=True, inplace=False):
        self.seed = trainmode.rng.fold(0, str(input.dtype))
        return real(self, input, p, training, inplace)

    monkeypatch.setattr(trainmode.TrainMode, "dropout", by_dtype)
    rep = truestar.check(conv_bn(), X_IMG, backends=["cpu"], train=True)
    assert rep.status == "divergent"
    assert rep.results[0].first_divergent.key == "3#0"


def test_state_restore_is_needed(monkeypatch):
    """Without loading the recorded state, isolated replays start from statistics other calls
    already updated, and the running statistic comparison must notice."""
    model = conv_bn()
    assert truestar.check(model, X_IMG, backends=["cpu"], train=True).status == "ok"
    monkeypatch.setattr(trainmode, "load_state", lambda module, saved: None)
    rep = truestar.check(model, X_IMG, backends=["cpu"], train=True)
    assert rep.status == "divergent"
