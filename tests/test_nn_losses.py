"""truestar.nn losses added 2026-11-25: values and gradients against torch.nn.functional in float64,
eager and compiled from one definition, and the float32 compiled values against float64."""

import pytest
import torch
import torch.nn.functional as F

import truestar
from truestar import Tensor, nn, ops

F64 = torch.float64
G = torch.Generator().manual_seed(0)
X = torch.randn(6, 5, generator=G, dtype=F64) * 3
Y = torch.randn(6, 5, generator=G, dtype=F64)
P01 = torch.rand(6, 5, generator=G, dtype=F64)
B01 = (torch.rand(6, 5, generator=G, dtype=F64) > 0.5).to(F64)
PW = torch.rand(5, generator=G, dtype=F64) * 3


@truestar.native
def f_bce_logits(x: Tensor[6, 5, "f64"], y: Tensor[6, 5, "f64"]) -> Tensor["f64"]:
    return nn.binary_cross_entropy_with_logits(x, y)


@truestar.native
def f_bce_logits_pw(x: Tensor[6, 5, "f64"], y: Tensor[6, 5, "f64"], w: Tensor[5, "f64"]) -> Tensor["f64"]:
    return nn.binary_cross_entropy_with_logits(x, y, pos_weight=w, reduction="sum")


@truestar.native
def f_bce(x: Tensor[6, 5, "f64"], y: Tensor[6, 5, "f64"]) -> Tensor["f64"]:
    return nn.binary_cross_entropy(ops.sigmoid(x), y)


@truestar.native
def f_l1(x: Tensor[6, 5, "f64"], y: Tensor[6, 5, "f64"]) -> Tensor["f64"]:
    return nn.l1(x, y)


@truestar.native
def f_smooth(x: Tensor[6, 5, "f64"], y: Tensor[6, 5, "f64"]) -> Tensor["f64"]:
    return nn.smooth_l1(x, y, beta=0.7, reduction="sum")


@truestar.native
def f_huber(x: Tensor[6, 5, "f64"], y: Tensor[6, 5, "f64"]) -> Tensor["f64"]:
    return nn.huber(x, y, delta=1.5)


@truestar.native
def f_kl(x: Tensor[6, 5, "f64"], y: Tensor[6, 5, "f64"]) -> Tensor["f64"]:
    return nn.kl_div(ops.log_softmax(x, dim=-1), ops.softmax(y, dim=-1))


@truestar.native
def f_kl_logt(x: Tensor[6, 5, "f64"], y: Tensor[6, 5, "f64"]) -> Tensor["f64"]:
    return nn.kl_div(ops.log_softmax(x, dim=-1), ops.log_softmax(y, dim=-1), reduction="sum", log_target=True)


@truestar.native
def f_cos(x: Tensor[6, 5, "f64"], y: Tensor[6, 5, "f64"]) -> Tensor["f64"]:
    return ops.mean(nn.cosine_similarity(x, y))


def _bce_t(x, y):
    return F.binary_cross_entropy(torch.sigmoid(x), y)


CASES = {
    "bce_logits": (f_bce_logits, lambda x, y: F.binary_cross_entropy_with_logits(x, y), (X, B01)),
    "bce_logits_soft": (f_bce_logits, lambda x, y: F.binary_cross_entropy_with_logits(x, y), (X, P01)),
    "bce": (f_bce, _bce_t, (X, P01)),
    "l1": (f_l1, F.l1_loss, (X, Y)),
    "smooth_l1": (f_smooth, lambda x, y: F.smooth_l1_loss(x, y, beta=0.7, reduction="sum"), (X, Y)),
    "huber": (f_huber, lambda x, y: F.huber_loss(x, y, delta=1.5), (X, Y)),
    "kl": (f_kl, lambda x, y: F.kl_div(F.log_softmax(x, -1), F.softmax(y, -1), reduction="batchmean"), (X, Y)),
    "kl_log_target": (f_kl_logt, lambda x, y: F.kl_div(F.log_softmax(x, -1), F.log_softmax(y, -1), reduction="sum",
                                                        log_target=True), (X, Y)),
    "cosine": (f_cos, lambda x, y: F.cosine_similarity(x, y, dim=-1).mean(), (X, Y)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_value_and_gradient_match_torch_in_float64(name):
    fn, ref, (x, y) = CASES[name]
    xt = x.clone().requires_grad_(True)
    want = ref(xt, y)
    want.backward()
    got = fn(x, y)                                   # compiled
    eager = fn.python(x, y)                            # the same source run as Python through PyTorch
    assert abs(float(got) - float(want)) <= 1e-13 * max(1.0, abs(float(want))), (name, float(got), float(want))
    assert abs(float(eager) - float(want)) <= 1e-13 * max(1.0, abs(float(want)))
    _, g = truestar.grad(fn, wrt="x")(x, y)
    assert torch.allclose(g, xt.grad, rtol=1e-11, atol=1e-14), (name, float((g - xt.grad).abs().max()))


def test_pos_weight_matches_torch():
    xt = X.clone().requires_grad_(True)
    want = F.binary_cross_entropy_with_logits(xt, P01, pos_weight=PW, reduction="sum")
    want.backward()
    assert abs(float(f_bce_logits_pw(X, P01, PW)) - float(want)) <= 1e-13 * abs(float(want))
    _, g = truestar.grad(f_bce_logits_pw, wrt="x")(X, P01, PW)
    assert torch.allclose(g, xt.grad, rtol=1e-11, atol=1e-14)


def test_nll_matches_torch_with_and_without_ignore_index():
    logp = F.log_softmax(X, -1)
    t = torch.tensor([0, 4, 2, 2, 1, 3])
    assert abs(float(nn.nll(logp, t)) - float(F.nll_loss(logp, t))) < 1e-15
    t2 = torch.tensor([0, -100, 2, -100, 1, 3])
    assert abs(float(nn.nll(logp, t2, ignore_index=-100)) - float(F.nll_loss(logp, t2))) < 1e-15


HAVE_NUMBA = __import__("importlib").util.find_spec("numba") is not None
needs_numba = pytest.mark.skipif(not HAVE_NUMBA, reason="needs numba (the compiled CPU executor computes in real float32)")


def _cpu32():
    from truestar.compiler import CompiledExecutor

    return CompiledExecutor("cpu", "float32", verify="off")   # raw float32 results, no float64 rescue


def l1p(x: Tensor["N", "f32"]) -> Tensor["N", "f32"]:
    return nn.log1p(x)


def sp32(x: Tensor["N", "f32"]) -> Tensor["N", "f32"]:
    return nn.softplus(x)


@needs_numba
def test_log1p_and_softplus_are_accurate_in_compiled_float32():
    """Compiled float32 against float64 torch.log1p: within 2 float32 ulps relative everywhere
    (the plain log(1 + x) is off by up to 100% below the float32 epsilon)."""
    x = torch.cat([-torch.logspace(-12, -0.31, 200, dtype=F64), torch.logspace(-12, 4, 400, dtype=F64)])
    f = truestar.native(l1p, executor=_cpu32())
    got = f(x.float())
    assert got.dtype == torch.float32
    got = got.double()
    want = torch.log1p(x.float().double())
    rel = ((got - want).abs() / want.abs()).max().item()
    assert rel < 2.4e-7, rel
    naive = torch.log(1 + x.float()).double()
    assert ((naive - want).abs() / want.abs()).max().item() > 0.5
    z = torch.linspace(-80, 80, 801, dtype=F64)
    s = truestar.native(sp32, executor=_cpu32())(z.float()).double()
    sw = F.softplus(z.float().double())
    assert ((s - sw).abs() / sw).max().item() < 5e-7


def bce32(x: Tensor["N", "f32"], y: Tensor["N", "f32"]) -> Tensor["N", "f32"]:
    return nn.binary_cross_entropy_with_logits(x, y, reduction="none")


@needs_numba
def test_float32_bce_with_logits_against_float64_at_large_margins():
    """Confident correct predictions (loss about exp(-|x|)): the compiled float32 loss keeps its
    relative accuracy to 1e-6 where log(1 + exp(-|x|)) in float32 returns 0."""
    x = torch.linspace(5, 60, 200, dtype=F64)
    y = torch.ones_like(x)
    got = truestar.native(bce32, executor=_cpu32())(x.float(), y.float()).double()
    want = F.binary_cross_entropy_with_logits(x.float().double(), y, reduction="none")
    assert ((got - want).abs() / want).max().item() < 1e-6
    naive = torch.log(1 + torch.exp(-x.float())).double()
    assert (naive == 0).sum().item() > 100


def test_reduction_none_and_bce_at_the_clamp():
    p = torch.tensor([0.0, 1.0, 0.5], dtype=F64)
    y = torch.tensor([1.0, 0.0, 1.0], dtype=F64)
    got = nn.binary_cross_entropy(p, y, reduction="none")
    assert torch.equal(got, F.binary_cross_entropy(p, y, reduction="none"))
    assert nn.l1(X, Y, reduction="none").shape == X.shape
