"""Mixture of experts (truestar.moe): routing tie rule, grouped experts, importer lowering, training,
the near tie router report and the TS940 reasons for layers that are not covered.

The tiny block below has the transformers 5.x layout (a router returning (logits, weights, indices),
experts holding stacked gate_up_proj [E, 2I, H] and down_proj [E, H, I]); its eager experts loop over
experts with index_add like the library's eager implementation, so it is the eager reference. The
transformers families run when the installed transformers has them (5.x), else skip.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from truestar import ir
from truestar.ir.core import UnsupportedOpError
from truestar.moe import experts as X
from truestar.moe import grouped_experts, lowered, reference_experts, restore, route_topk, stable_topk, use_grouped
from truestar.moe.ties import router_ties


class TinyTopKRouter(nn.Module):
    def __init__(self, E, H, k):
        super().__init__()
        self.top_k, self.weight = k, nn.Parameter(torch.randn(E, H) * 0.3)

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.reshape(-1, self.weight.shape[1]), self.weight)
        probs = torch.softmax(logits, dim=-1)
        top, idx = torch.topk(probs, self.top_k, dim=-1)
        return logits, top / top.sum(-1, keepdim=True), idx


class TinyExperts(nn.Module):
    def __init__(self, E, H, inter):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * inter, H) * 0.2)
        self.down_proj = nn.Parameter(torch.randn(E, H, inter) * 0.2)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for e in range(self.gate_up_proj.shape[0]):
            pos, tok = torch.where((top_k_index == e).t())
            if tok.numel() == 0:
                continue
            gate, up = F.linear(hidden_states[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
            y = F.linear(self.act_fn(gate) * up, self.down_proj[e]) * top_k_weights[tok, pos, None]
            out.index_add_(0, tok, y.to(out.dtype))
        return out


class TinySparseMoeBlock(nn.Module):
    def __init__(self, E=4, H=16, inter=8, k=2):
        super().__init__()
        self.gate, self.experts = TinyTopKRouter(E, H, k), TinyExperts(E, H, inter)

    def forward(self, x):
        B, S, H = x.shape
        flat = x.reshape(-1, H)
        _, w, idx = self.gate(flat)
        return self.experts(flat, idx, w).reshape(B, S, H)


class TinyMoeModel(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.inp = nn.Linear(16, 16)
        self.moe = TinySparseMoeBlock(**kw)

    def forward(self, x):
        h = self.inp(x)
        return h + self.moe(h)


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    return TinyMoeModel(**kw).eval(), (torch.randn(2, 5, 16),)


# ----------------------------------------------------------------------------- routing


def test_route_topk_tie_rule():
    s = torch.tensor([[1.0, 3.0, 3.0, 2.0, 3.0], [float("nan"), 0.0, 1.0, 1.0, -1.0]])
    v, i = route_topk(s, 3)
    assert i.tolist() == [[1, 2, 4], [0, 2, 3]]       # equal scores: lower index first; NaN ranks first
    with stable_topk():
        assert torch.topk(s, 3).indices.tolist() == i.tolist()
        assert s.topk(2, dim=-1, sorted=False).indices.tolist() == i[:, :2].tolist()
    # the IR's topk follows the same rule
    ref = ir.reference.IMPLS["topk"] if hasattr(ir.reference, "IMPLS") else None
    if ref is not None:
        assert ref({"k": 3, "dim": -1, "largest": True}, [s], None)[1].tolist() == i.tolist()


# ----------------------------------------------------------------------------- grouped experts


def _expert_args(T=37, E=5, H=16, inter=8, k=2, dtype=torch.float64, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, H, generator=g, dtype=dtype)
    idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])
    w = torch.rand(T, k, generator=g, dtype=dtype)
    gu = torch.randn(E, 2 * inter, H, generator=g, dtype=dtype) * 0.3
    dn = torch.randn(E, H, inter, generator=g, dtype=dtype) * 0.3
    return x, idx, w, gu, dn


def test_grouped_matches_float64_and_eager_loop():
    x, idx, w, gu, dn = _expert_args()
    act = nn.SiLU()
    got = grouped_experts(x, idx, w, gu, dn, act)
    assert (got - reference_experts(x, idx, w, gu, dn, act)).abs().max() < 1e-13
    e = TinyExperts(5, 16, 8).double()
    e.gate_up_proj.data, e.down_proj.data = gu, dn
    assert (got - e(x, idx, w)).abs().max() < 1e-13
    x32, w32, gu32, dn32 = (t.float() for t in (x, w, gu, dn))
    X.reset()
    n = X.STATS["checks"]
    grouped_experts(x32, idx, w32, gu32, dn32, act)
    grouped_experts(x32, idx, w32, gu32, dn32, act)
    assert X.STATS["checks"] == n + 1 and X.last_call()["checked"] is False   # first call per signature


def test_grouped_is_bitwise_reproducible_forward_and_backward():
    x, idx, w, gu, dn = (t.float() if t.is_floating_point() else t for t in _expert_args(T=64, E=6, k=3))

    def run():
        leaves = [t.clone().requires_grad_(True) for t in (x, w, gu, dn)]
        out = grouped_experts(leaves[0], idx, leaves[1], leaves[2], leaves[3], nn.SiLU())
        out.backward(torch.ones_like(out))
        return [out.detach()] + [t.grad for t in leaves]

    a, b = run(), run()
    assert all(torch.equal(p, q) for p, q in zip(a, b))


def test_grouped_gradients_match_eager_float64():
    x, idx, w, gu, dn = _expert_args()
    e = TinyExperts(5, 16, 8).double()
    e.gate_up_proj.data, e.down_proj.data = gu.clone(), dn.clone()
    lx, lw = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
    e(lx, idx, lw).pow(2).sum().backward()
    gx, gw = lx.grad, lw.grad
    lx, lw = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
    lgu, ldn = gu.clone().requires_grad_(True), dn.clone().requires_grad_(True)
    grouped_experts(lx, idx, lw, lgu, ldn, nn.SiLU()).pow(2).sum().backward()
    for got, want in ((lx.grad, gx), (lw.grad, gw), (lgu.grad, e.gate_up_proj.grad), (ldn.grad, e.down_proj.grad)):
        assert (got - want).abs().max() <= 1e-12 * max(1.0, want.abs().max().item())


def test_grouped_mismatch_and_bad_index_are_named(monkeypatch):
    x, idx, w, gu, dn = (t.float() if t.is_floating_point() else t for t in _expert_args(T=9))
    bad = idx.clone()
    bad[3, 1] = 7
    with pytest.raises(ValueError, match="TS1180"):
        grouped_experts(x, bad, w, gu, dn, nn.SiLU())
    real = X.grouped_experts_unchecked
    monkeypatch.setattr(X, "grouped_experts_unchecked", lambda *a: real(*a) * 1.01)
    X.reset()
    with pytest.raises(X.MoEMismatchError, match="TS1181"):
        grouped_experts(x, idx, w, gu, dn, nn.SiLU())
    X.reset()


def test_use_grouped_eager_matches_and_restores():
    model, args = _model()
    with torch.no_grad():
        want = model(*args)
        assert use_grouped(model) == ["moe.experts"]
        got = model(*args)
        restore(model)
        again = model(*args)
    assert (got - want).abs().max() < 1e-5 and torch.equal(again, want)
    assert "forward" not in model.moe.experts.__dict__


# ----------------------------------------------------------------------------- importer and training


def test_import_lowers_the_block_and_matches_float64():
    model, args = _model()
    prog = ir.import_module(model, args)
    m64 = copy.deepcopy(model).double()
    a64 = tuple(a.double() for a in args)
    p64 = ir.import_module(m64, a64)
    with torch.no_grad():
        want = m64(*a64)
    got = ir.run(p64, *a64)
    got = got[0] if isinstance(got, (list, tuple)) else got
    assert (got - want).abs().max() < 1e-13
    assert prog is not None and "forward" not in model.moe.experts.__dict__   # the module is restored


def test_outside_the_lowering_the_layer_is_refused_by_name():
    """The eager experts loop has data dependent shapes; only the lowered form exports."""
    from truestar.ir.refusals import check_module
    model, args = _model()
    with pytest.raises(UnsupportedOpError, match=r"TS940.*did not run inside truestar.moe.lowered") as err:
        check_module(model, "tiny")
    assert err.value.code == "TS940"
    with lowered(model):
        check_module(model, "tiny")
        ep = torch.export.export(model, args, strict=False)
        ir.import_exported(ep, name="tiny")
    with pytest.raises(UnsupportedOpError, match="TS940"):      # the same program, outside the context
        ir.import_exported(ep, name="tiny")


def test_ir_gradients_match_eager_float64():
    from truestar.ir.autodiff import grad
    model, args = _model()
    m64 = copy.deepcopy(model).double()
    a64 = tuple(a.double() for a in args)
    prog = ir.import_module(m64, a64)
    sd = dict(m64.named_parameters())
    names = [v.name for v in prog.params if v.name in sd]
    assert "moe.experts.gate_up_proj" in names and "moe.gate.weight" in names
    step = grad(prog, wrt=names, of=[0], seed="input")
    cot = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    got = ir.run(step, *a64, cot)
    g = dict(zip(names, got[len(prog.outputs):]))
    m64(*a64).backward(cot)
    for nm in names:
        assert (g[nm] - sd[nm].grad).abs().max() <= 1e-12 * max(1.0, sd[nm].grad.abs().max().item()), nm


# ----------------------------------------------------------------------------- refusals


def test_uncovered_variants_keep_ts940_with_the_reason():
    from truestar.ir.refusals import check_module

    def refused(mutate):
        model, _ = _model()
        mutate(model.moe)
        with pytest.raises(UnsupportedOpError) as err:
            check_module(model, "tiny")
        assert err.value.code == "TS940"
        return str(err.value)

    assert "capacity" in refused(lambda b: setattr(b.gate, "expert_capacity", 4))
    assert "biases" in refused(lambda b: setattr(b.experts, "has_bias", True))
    assert "transposed" in refused(lambda b: setattr(b.experts, "is_transposed", True))
    assert "not [E, 2I, H]" in refused(lambda b: setattr(b.experts, "down_proj", nn.Parameter(torch.randn(4, 8, 16))))


# ----------------------------------------------------------------------------- near tie router report


def test_router_ties_planted_and_clear():
    torch.manual_seed(0)
    model = TinySparseMoeBlock().eval()
    x = torch.randn(1, 8, 16).abs()
    r = torch.randn(16) * 0.01
    with torch.no_grad():
        model.gate.weight[0], model.gate.weight[3], model.gate.weight[1] = 0.5, -0.5, r
        model.gate.weight[2] = r * (1 + 0.2)
    clear = router_ties(model, (x,), dtype=torch.float32)
    assert clear.codes == [] and clear.layers[0].rule == "softmax" and clear.layers[0].min_ratio > 1
    with torch.no_grad():
        model.gate.weight[2] = r * (1 + 1e-7)        # experts 1 and 2 now differ by about 1e-9
    rep = router_ties(model, (x,), dtype=torch.float32)
    assert "TS1182" in rep.codes and len(rep.ties) == 8
    t = rep.ties[0]
    assert t.layer == "gate" and {t.kth, t.next} == {1, 2} and t.gap < t.bound and t.position == (0, 0)
    assert "TS1183" not in rep.codes                  # every changed choice is a near tie
    assert "TS1182 token 0 (0, 0): experts" in rep.describe()


# ----------------------------------------------------------------------------- transformers families


def _families():
    from truestar import testing_hf
    return testing_hf.available(getattr(testing_hf, "MOE_2026", {}))


@pytest.mark.parametrize("name", ["mixtral", "qwen3_moe", "olmoe", "deepseek_v3", "deepseek_v3_groups"])
def test_transformers_families_import_and_match_float64(name):
    pytest.importorskip("transformers")
    if name not in _families():
        pytest.skip("this transformers has no such family")
    from truestar import testing_hf
    model, args = testing_hf.build(name)
    if not any(hasattr(m, "gate_up_proj") for m in model.modules()):
        pytest.skip("this transformers stores experts per module (before 5.x)")
    m64 = copy.deepcopy(model).double()
    a64 = tuple(a.double() if a.is_floating_point() else a for a in args)
    p64 = ir.import_module(m64, a64)
    with torch.no_grad(), ir.reference.exact_float32_casts():   # the routers' .float() casts stay float64
        want = m64(*a64)[0]
    got = ir.run(p64, *a64)[0]
    assert (got - want).abs().max() / want.abs().max() < 1e-12
    assert router_ties(model, args, dtype=torch.float32).layers


def test_moe_codes_are_registered():
    from truestar.codes import CODES
    from truestar.moe.codes import MOE_CODES
    assert sorted(MOE_CODES) == [f"TS{n}" for n in range(1180, 1185)] and set(MOE_CODES) <= set(CODES)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_grouped_bitwise_and_same_experts_as_cpu():
    x, idx, w, gu, dn = (t.float() if t.is_floating_point() else t for t in _expert_args(T=200, E=8, k=2))
    act = nn.SiLU()
    X.reset()
    dev = [t.cuda() for t in (x, idx, w, gu, dn)]
    a = grouped_experts(*dev, act)
    assert X.last_call()["checked"] and torch.equal(a, grouped_experts(*dev, act))
    model, args = _model()
    picks = {}
    for d in ("cpu", "cuda"):
        m = copy.deepcopy(model).to(d)
        use_grouped(m)
        seen = []
        h = m.moe.gate.register_forward_hook(lambda _m, _a, o: seen.append(o[2].cpu()))
        with torch.no_grad():
            m(args[0].to(d))
        h.remove()
        picks[d] = seen[0]
    assert torch.equal(picks["cpu"], picks["cuda"])
