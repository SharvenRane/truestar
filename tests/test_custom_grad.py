"""User written gradients (``@fn.grad``), verified when they are registered (docs/AUTODIFF.md)."""

import time
import warnings

import pytest
import torch

import truestar
from truestar import Tensor, ops
from truestar import ir
from truestar.frontend import CustomGradientError, FrontendShapeError
from truestar.ir import reference, serialize
from truestar.ir.autodiff import grad
from truestar.ir.custom_vjp import has_custom_vjp, inline_custom_vjp

F64 = torch.float64


def rel_err(a, b) -> float:
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-300)).item()


# ----------------------------------------------------------------------------- a correct rule


@truestar.function
def softplus(x: Tensor["N"]) -> Tensor["N"]:
    # the textbook form; exp overflows in the branch that is not taken for large x
    return ops.where(x > 20.0, x, ops.log(1.0 + ops.exp(x)))


@softplus.grad
def _softplus_grad(x, g):
    return g * ops.sigmoid(x)


def test_correct_rule_is_verified_by_finite_differences():
    rep = softplus.grad_report()
    assert rep.verified
    # log has an autodiff rule since the op coverage work, so the rule is checked twice: against
    # autodiff of the forward and against finite differences
    assert rep.autodiff is None
    (c,) = rep.inputs
    assert c.name == "x" and not c.wrong and c.fd_directions >= 3
    assert c.autodiff_error is not None and c.autodiff_error < 1e-12
    assert c.fd_error < 1e-8
    assert rep.sizes == [{"N": 3}, {"N": 2}]


def test_rule_is_what_grad_differentiates_and_survives_where_autograd_gives_nan():
    x = torch.tensor([-3.0, 0.5, 1000.0], dtype=torch.float32)
    prog = softplus.program
    assert [n.op for n in prog.nodes] == ["custom_vjp"]
    gp = grad(prog, ["x"], seed="ones")
    y, gx = ir.run(gp, x)
    assert rel_err(gx, torch.sigmoid(x.double())) < 1e-12
    assert torch.isfinite(gx).all()
    # the same source through torch autograd: the where gotcha gives nan at x = 1000
    xt = x.double().requires_grad_()
    torch.where(xt > 20, xt, torch.log(1 + torch.exp(xt))).sum().backward()
    assert torch.isnan(xt.grad[2])


def test_rule_with_autodiff_available_is_checked_entrywise():
    @truestar.function
    def gated(x: Tensor["B", 4], w: Tensor[4]) -> Tensor["B", 4]:
        return x * ops.tanh(x * w)

    @gated.grad
    def _(x, w, g, out):
        t = ops.tanh(x * w)
        d = (1.0 - t * t) * g * x
        return g * t + d * w, (d * x).sum(0)

    rep = gated.grad_report()
    assert rep.autodiff is None
    for c in rep.inputs:
        assert c.autodiff_error < 1e-13 and c.fd_error < 1e-8, c


def test_call_inside_another_function_uses_the_rule():
    @truestar.function
    def loss(a: Tensor["N"], w: Tensor["N"]) -> Tensor[()]:
        return softplus(a * w - 0.5).sum()

    prog = loss.program
    assert prog.op_counts().get("custom_vjp") == 1
    gp = grad(prog, ["w"])
    a, w = torch.randn(7, dtype=F64), torch.randn(7, dtype=F64)
    _, gw = ir.run(gp, a, w)
    wt = w.clone().requires_grad_()
    torch.nn.functional.softplus(a * wt - 0.5).sum().backward()
    assert rel_err(gw, wt.grad) < 1e-12


# ----------------------------------------------------------------------------- wrong rules are refused


def test_sign_error_is_refused_naming_the_input():
    @truestar.function
    def f(x: Tensor["N"]) -> Tensor["N"]:
        return ops.tanh(x)

    with pytest.raises(CustomGradientError) as ei:
        @f.grad
        def _(x, g):
            t = ops.tanh(x)
            return -g * (1.0 - t * t)
    e = ei.value
    assert "wrong for x" in str(e) and "autodiff" in str(e) and "finite differences" in str(e)
    d = e.to_dict()
    assert d["code"] == "wrong_gradient" and d["operands"][0]["name"] == "gradient of x"
    assert f._custom_grad is None  # nothing was attached


def test_missing_term_is_refused_only_for_the_wrong_input():
    @truestar.function
    def f(x: Tensor["N"], y: Tensor["N"]) -> Tensor["N"]:
        return x * ops.exp(y) + y * y

    with pytest.raises(CustomGradientError) as ei:
        @f.grad
        def _(x, y, g):
            return g * ops.exp(y), g * x * ops.exp(y)   # forgot the 2 y term of y
    msg = str(ei.value)
    assert "wrong for y (" in msg and "x (" not in msg


def test_missing_term_is_refused_by_finite_differences_alone():
    @truestar.function
    def f(x: Tensor["N"]) -> Tensor["N"]:
        return ops.log(1.0 + x * x)     # the refusal names finite differences (log also has an autodiff rule)

    with pytest.raises(CustomGradientError, match=r"wrong for x \(.*finite differences"):
        @f.grad
        def _(x, g):
            return g * x / (1.0 + x * x)  # missing the factor 2


def test_wrong_broadcast_is_refused_before_anything_runs():
    @truestar.function
    def affine(x: Tensor["B", 8], b: Tensor[8]) -> Tensor["B", 8]:
        return x + b

    with pytest.raises(FrontendShapeError) as ei:
        @affine.grad
        def _(x, b, g):
            return g, g                  # the bias gradient must be summed over B
    e = ei.value
    assert e.code == "wrong_gradient_shape"
    assert "gradient returned for b has type f32[B, 8] but b is f32[8]" in str(e)
    assert ".sum(0)" in e.suggestion


def test_signature_errors():
    @truestar.function
    def f(x: Tensor["N"]) -> Tensor["N"]:
        return x * x

    with pytest.raises(CustomGradientError, match="not parameters of f"):
        f.grad(lambda_free_rule_extra_name)
    with pytest.raises(CustomGradientError, match="must take g"):
        f.grad(rule_without_g)


def lambda_free_rule_extra_name(x, y, g):
    return g


def rule_without_g(x):
    return x


def test_wrong_number_of_gradients():
    @truestar.function
    def f(x: Tensor["N"], y: Tensor["N"]) -> Tensor["N"]:
        return x * y

    with pytest.raises(FrontendShapeError, match="must return 2"):
        @f.grad
        def _(x, y, g):
            return g * y


# ----------------------------------------------------------------------------- opt out


def test_straight_through_estimator_needs_unverified_reason():
    @truestar.function
    def binarize(x: Tensor["N"]) -> Tensor["N"]:
        return ops.where(x > 0.0, ops.ones_like(x), ops.zeros_like(x))

    with pytest.raises(CustomGradientError, match="wrong for x"):
        @binarize.grad
        def _(x, g):
            return g

    @binarize.grad(unverified="straight through estimator")
    def _(x, g):
        return g

    rep = binarize.grad_report()
    assert not rep.verified and rep.unverified_reason == "straight through estimator"
    gp = grad(binarize.program, ["x"], seed="ones")
    assert torch.equal(ir.run(gp, torch.tensor([-1.0, 2.0]))[1], torch.ones(2, dtype=F64))
    with pytest.raises(Exception, match="non empty reason"):
        binarize.grad(unverified="")(lambda x, g: g)


def test_domain_for_restricted_inputs():
    @truestar.function
    def lg(x: Tensor["N"]) -> Tensor["N"]:
        return ops.log(x)

    with pytest.raises(CustomGradientError, match="not finite"):
        @lg.grad
        def _(x, g):
            return g / x

    @lg.grad(domain={"x": (0.5, 3.0)})
    def _(x, g):
        return g / x
    assert lg.grad_report().verified


# ----------------------------------------------------------------------------- the rest of the stack


def test_executors_and_serialization():
    prog = softplus.program
    flat = inline_custom_vjp(prog)
    assert has_custom_vjp(prog) and not has_custom_vjp(flat)
    x = torch.randn(5, dtype=F64)
    assert torch.equal(ir.run(prog, x)[0], ir.run(flat, x)[0])
    art = serialize.loads(serialize.dumps(prog))
    assert [n.op for n in art.program.nodes] == ["custom_vjp"]
    g1 = ir.run(grad(art.program, ["x"], seed="ones"), x)[1]
    assert rel_err(g1, torch.sigmoid(x)) < 1e-12


def test_torch_op_backward_uses_the_rule():
    op = truestar.torch_op(softplus, name="ts_softplus_custom_grad")
    x = torch.tensor([-1.0, 0.25, 50.0], dtype=torch.float32, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        op(x).sum().backward()
        torch.library.opcheck(op.overload, (torch.randn(4, requires_grad=True),))
    assert rel_err(x.grad, torch.sigmoid(x.detach())) < 1e-6
    assert torch.isfinite(x.grad).all()


def test_verification_cost_under_one_second():
    @truestar.function
    def f(x: Tensor["B", 16], w: Tensor[16, 16]) -> Tensor["B", 16]:
        return ops.tanh(x @ w)

    t0 = time.perf_counter()

    @f.grad
    def _(x, w, g, out):
        d = g * (1.0 - out * out)
        return d @ w.T, x.T @ d
    assert time.perf_counter() - t0 < 1.0
    assert f.grad_report().seconds < 1.0


def test_compiled_paths_run_the_primal_and_the_rule():
    pytest.importorskip("numba")
    from truestar.compiler import CompiledExecutor

    @truestar.function(executor=CompiledExecutor("cpu"))
    def f(x: Tensor["B", 16], w: Tensor[16, 16]) -> Tensor["B", 16]:
        return ops.tanh(x @ w)

    @f.grad
    def _(x, w, g, out):
        d = g * (1.0 - out * out)
        return d @ w.T, x.T @ d

    x, w = torch.randn(4, 16), torch.randn(16, 16)
    assert rel_err(f(x, w), torch.tanh(x @ w)) < 1e-5
    op = truestar.torch_op(f, name="ts_tanh_mm_custom_grad")
    xr, wr = x.clone().requires_grad_(), w.clone().requires_grad_()
    op(xr, wr).sum().backward()
    assert op.plan("cpu").path == "kernel" and op.plan("cpu", backward=True).path == "kernel"
    xt, wt = x.clone().requires_grad_(), w.clone().requires_grad_()
    torch.tanh(xt @ wt).sum().backward()
    assert rel_err(xr.grad, xt.grad) < 1e-5 and rel_err(wr.grad, wt.grad) < 1e-5
