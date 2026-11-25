"""truestar.value_and_grad with JAX's has_aux convention, against jax-like expectations and torch
autograd in float64."""

import pytest
import torch

import truestar
from truestar import Tensor, ops
from truestar.frontend import InputError

F64 = torch.float64


@truestar.native
def loss_only(w: Tensor[3, 2, "f64"], x: Tensor["B", 3, "f64"]) -> Tensor["f64"]:
    return ops.sum(ops.tanh(x @ w) ** 2)


@truestar.native
def loss_aux(w: Tensor[3, 2, "f64"], x: Tensor["B", 3, "f64"]):
    y = ops.tanh(x @ w)
    return ops.sum(y ** 2), {"y": y, "mean": ops.mean(y)}


def _ref(w, x):
    wt = w.clone().requires_grad_(True)
    y = torch.tanh(x @ wt)
    loss = (y ** 2).sum()
    loss.backward()
    return loss.detach(), y.detach(), wt.grad


W = torch.randn(3, 2, dtype=F64, generator=torch.Generator().manual_seed(0))
X = torch.randn(5, 3, dtype=F64, generator=torch.Generator().manual_seed(1))


def test_value_and_grad_without_aux():
    loss, g = truestar.value_and_grad(loss_only, wrt="w")(W, X)
    rl, _, rg = _ref(W, X)
    assert abs(float(loss) - float(rl)) < 1e-13 and torch.allclose(g, rg, rtol=1e-12, atol=0)


def test_value_and_grad_with_aux_returns_the_aux_structure_undifferentiated():
    (loss, aux), g = truestar.value_and_grad(loss_aux, wrt="w", has_aux=True)(W, X)
    rl, ry, rg = _ref(W, X)
    assert abs(float(loss) - float(rl)) < 1e-13
    assert torch.allclose(aux["y"], ry, rtol=1e-13, atol=0) and abs(float(aux["mean"]) - float(ry.mean())) < 1e-15
    assert torch.allclose(g, rg, rtol=1e-12, atol=0)


def test_wrong_has_aux_is_refused_with_the_fix():
    with pytest.raises(InputError, match="has_aux=True"):
        truestar.value_and_grad(loss_aux, wrt="w")(W, X)
    with pytest.raises(InputError, match="pair"):
        truestar.value_and_grad(loss_only, wrt="w", has_aux=True)(W, X)
