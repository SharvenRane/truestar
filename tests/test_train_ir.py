"""Training mode in the IR: batch norm with explicit running statistics and keyed dropout, through the
importer, the reference interpreter, autodiff and the compiled executor (docs/RANDOM.md)."""

from __future__ import annotations

import copy
import importlib.util
import os

import pytest
import torch
from torch import nn

from truestar import ir, rng
from truestar.ir.autodiff import grad
from truestar.ir.core import UnsupportedOpError

CUDA = torch.cuda.is_available()
HERE = os.path.dirname(os.path.abspath(__file__))


def _bn_model():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 6 * 6, 5),
                      nn.BatchNorm1d(5, momentum=0.3)).double().train()
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.running_mean.uniform_(-1, 1)
            mod.running_var.uniform_(0.5, 2)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.5, 0.5)
    return m


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


@pytest.mark.parametrize("decompose", [False, True], ids=["export", "core_aten"])
def test_training_batch_norm_matches_torch_with_explicit_state(decompose):
    m = _bn_model()
    p = ir.import_module(copy.deepcopy(m), (torch.randn(3, 3, 8, 8, dtype=torch.float64),), training=True,
                         decompose=decompose)
    assert sorted(t for t, _, _ in p.state) == sorted(
        ["1.running_mean", "1.running_var", "1.num_batches_tracked",
         "5.running_mean", "5.running_var", "5.num_batches_tracked"])
    assert p.key is None                     # no dropout: no key input
    n_user = len(p.outputs) - len(p.state)
    state = ir.initial_state(p)
    ref = copy.deepcopy(m)
    for step in range(4):
        x = torch.randn(3, 3, 8, 8, dtype=torch.float64)
        outs = ir.run(p, x, *state)
        assert _rel(outs[0], ref(x)) < 1e-12
        state = list(outs[n_user:])
        bufs = ref.state_dict()
        for (t, _, _), s in zip(p.state, state):
            assert _rel(s, bufs[t]) < 1e-12, (step, t)
        assert int(state[[t for t, _, _ in p.state].index("1.num_batches_tracked")]) == step + 1


def test_training_batch_norm_gradients_match_torch_autograd():
    m = _bn_model()
    x = torch.randn(3, 3, 8, 8, dtype=torch.float64)

    class Loss(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            y = self.net(x)
            return (y * y).mean()

    p = ir.import_module(Loss(copy.deepcopy(m)).train(), (x,), training=True)
    wrt = [f"net.{n}" for n, _ in m.named_parameters()]
    g = grad(p, wrt=wrt, of=[0])
    outs = ir.run(g, x, *ir.initial_state(p))
    ref = copy.deepcopy(m)
    loss = (ref(x) ** 2).mean()
    loss.backward()
    grads = outs[len(p.outputs):]
    # the conv bias in front of batch norm has an exactly zero gradient (both sides give 1e-20 noise),
    # so errors are measured against the largest gradient of the model
    scale = max(float(prm.grad.abs().max()) for prm in ref.parameters())
    for (name, prm), gv in zip(ref.named_parameters(), grads):
        assert float((gv - prm.grad).abs().max()) < 1e-12 * scale, name


def test_training_batch_norm_with_a_symbolic_batch():
    m = nn.Sequential(nn.Linear(6, 4), nn.BatchNorm1d(4)).double().train()
    B = torch.export.Dim("B", min=2, max=64)
    p = ir.import_module(copy.deepcopy(m), (torch.randn(5, 6, dtype=torch.float64),),
                         dynamic_shapes=({0: B},), training=True)
    assert "dim_scalar" in p.op_counts()        # n / (n - 1) is a run time value of B
    ref = copy.deepcopy(m)
    state = ir.initial_state(p)
    for b in (2, 7, 33):
        x = torch.randn(b, 6, dtype=torch.float64)
        outs = ir.run(p, x, *state)
        assert _rel(outs[0], ref(x)) < 1e-12
        state = list(outs[1:])
        bufs = ref.state_dict()
        for (t, _, _), s in zip(p.state, state):
            assert _rel(s, bufs[t]) < 1e-12


def test_resnet18_training_steps_match_torch():
    pytest.importorskip("torchvision")
    spec = importlib.util.spec_from_file_location("train_mode_resnet18",
                                                  os.path.join(HERE, "..", "scripts", "train_mode_resnet18.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    worst, control = mod.main(steps=3, resync=True)
    assert worst < 1e-11
    # free running at this learning rate is chaotic: two correct float64 runs drift apart too
    assert control > worst


def test_keyed_dropout_import():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 16), nn.Dropout(0.25), nn.ReLU(), nn.Linear(16, 4)).train()
    x = torch.randn(64, 8)
    for decompose in (False, True):
        p = ir.import_module(m, (x,), training=True, decompose=decompose)
        assert p.key is not None and p.inputs[-1].name == p.key
        nodes = [n for n in p.nodes if n.op == "random"]
        assert len(nodes) == 1 and nodes[0].attrs["site_name"] == "1/dropout#0"
        assert nodes[0].attrs["site"] == rng.site_id("1/dropout#0")
        _, named = ir.run(p, x, torch.tensor(1), return_all=True)
        mask = named[nodes[0].output.name]
        out1 = ir.run(p, x, torch.tensor(1))[0]
        assert torch.equal(out1, ir.run(p, x, torch.tensor(1))[0])          # same key, same result
        assert not torch.equal(out1, ir.run(p, x, torch.tensor(2))[0])      # new key, new mask
        expect = rng.generate("bernoulli", (64, 16), torch.bool, 1, "1/dropout#0", p=0.75)
        assert torch.equal(mask, expect)
        assert abs(float(expect.double().mean()) - 0.75) < 0.05


def test_training_graphs_are_refused_without_training_true():
    m = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)).train()
    with pytest.raises(UnsupportedOpError, match="training=True"):
        ir.import_module(m, (torch.randn(3, 4),))
    with pytest.raises(UnsupportedOpError, match="training=True"):
        ir.import_module(nn.Sequential(nn.Dropout(0.5)).train(), (torch.randn(3, 4),))
    with pytest.raises(UnsupportedOpError, match="model.train"):
        ir.import_module(m.eval(), (torch.randn(3, 4),), training=True)
    # momentum=None (a cumulative average) fails inside torch.export itself; the builder refuses it too
    from truestar.ir.core import Builder
    from truestar.ir.train_ops import batch_norm_train

    bld = Builder("bn")
    x, rm, rv = bld.input("x", torch.float32, (3, 4)), bld.input("m", torch.float32, (4,)), bld.input("v", torch.float32, (4,))
    with pytest.raises(UnsupportedOpError, match="momentum=None"):
        batch_norm_train(lambda op, ins, **kw: bld.op(op, *ins, **kw), None, x, None, None, rm, rv, None, 1e-5)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
def test_compiled_dropout_draws_the_reference_mask(device):
    from truestar.compiler import CompiledExecutor

    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 16), nn.Dropout(0.3), nn.ReLU(), nn.Linear(16, 4), nn.Dropout(0.5)).train()
    p = ir.import_module(m, (torch.randn(5, 8),), training=True)
    x = torch.randn(5, 8)
    ex = CompiledExecutor(device, verify="all", allow_fallback=False)
    run = ex.prepare(p)
    for key in (7, -3, 2 ** 40):
        ref = ir.run(p, x, torch.tensor(key))[0]
        out = run([x.to(device), torch.tensor(key)])[0].cpu()
        assert torch.equal(out == 0, ref == 0)            # identical masks
        assert _rel(out, ref) < 1e-5
    rep = run.report
    assert rep.counts()["random"] == 2 and rep.counts()["fallback"] == 0
    assert "truestar.rng" in rep.to_text()
