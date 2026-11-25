"""Random and training ops in the native front end (docs/RANDOM.md): implicit keys from the program
seed and call count, site names, loops, explicit keys, errors, compiled backends, batch norm state."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import truestar
from truestar import Tensor, ops, rng
from truestar.frontend.errors import FrontendError

CUDA = torch.cuda.is_available()


@truestar.function(seed=3)
def noisy(x: Tensor["B", 16]):
    h = ops.dropout(x, 0.25)
    return h, ops.randn(h.shape), ops.rand((5,)), ops.randint(0, 10, (7,)), ops.bernoulli((9,), 0.3)


def test_implicit_keys_follow_seed_and_call_count():
    assert noisy.sites == ("noisy/dropout#0", "noisy/randn#0", "noisy/rand#0", "noisy/randint#0",
                           "noisy/bernoulli#0")
    x = torch.ones(4, 16)
    noisy.calls = 0
    first = noisy(x)
    second = noisy(x)
    assert noisy.calls == 2
    assert not torch.equal(first[1], second[1])            # a new call draws new numbers
    noisy.calls = 0
    again = noisy(x)
    assert all(torch.equal(a, b) for a, b in zip(first, again))   # replaying call 0 is exact
    # every value is a pure function of (key, site, index): reproduce call 0 eagerly
    k = noisy.key_for(0)
    assert k == rng.call_key(3, 0)
    keep = rng.generate("bernoulli", (4, 16), torch.bool, k, "noisy/dropout#0", p=0.75)
    assert torch.equal(first[0] != 0, keep)
    assert torch.equal(first[1], rng.generate("normal", (4, 16), torch.float32, k, "noisy/randn#0").double())
    assert torch.equal(first[2], ops.rand((5,), key=k, name="noisy/rand#0").double())
    assert torch.equal(first[3], ops.randint(0, 10, (7,), key=k, name="noisy/randint#0"))
    assert torch.equal(first[4], ops.bernoulli((9,), 0.3, key=k, name="noisy/bernoulli#0"))


def test_same_seed_same_numbers_other_seed_other_numbers():
    def make(seed):
        @truestar.function(seed=seed)
        def f(x: Tensor[8]):
            return x + ops.randn((8,))
        return f
    x = torch.zeros(8)
    assert torch.equal(make(1)(x), make(1)(x))
    assert not torch.equal(make(1)(x), make(2)(x))


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
def test_compiled_backends_draw_the_reference_bits(device):
    from truestar.compiler import CompiledExecutor

    def body(x: Tensor["B", 16]):
        h = ops.dropout(x, 0.5)
        return h, ops.randn(h.shape), ops.rand((5,)), ops.randint(-3, 3, (7,))

    # one source, so the same sites; one compiled, one on the reference interpreter
    f = truestar.function(seed=11, executor=CompiledExecutor(device, verify="all", allow_fallback=False))(body)
    g = truestar.function(seed=11)(body)

    for b in (1, 3, 8):
        x = torch.randn(b, 16)
        got, ref = f(x.to(device)), g(x)
        assert torch.equal(got[0].cpu() == 0, ref[0] == 0)
        for a, r in zip(got[1:], ref[1:]):
            assert torch.equal(a.cpu().to(r.dtype), r), "a compiled draw differs from the reference"


def test_symbolic_loop_iterations_draw_fresh_numbers():
    @truestar.function(seed=5)
    def acc(x: Tensor["B", "T", 4]):
        h = ops.zeros((4,))
        for t in range(x.shape[1]):
            h = h * 10.0 + ops.randint(0, 10, (4,)).float()
        return h

    out = acc(torch.zeros(1, 3, 4))
    k = acc.key_for(0)
    digits = [rng.generate("randint", (4,), torch.int64, k, "acc/randint#0", (t,), low=0, high=10) for t in range(3)]
    assert torch.equal(out, (digits[0] * 100 + digits[1] * 10 + digits[2]).double())
    assert not torch.equal(digits[0], digits[1])


def test_unrolled_loops_and_inlined_calls_get_their_own_sites():
    @truestar.function
    def inner(x: Tensor["B", 8]):
        return ops.dropout(x, 0.5)

    @truestar.function
    def outer(x: Tensor["B", 8]):
        y = inner(x)
        for i in range(2):
            y = y + ops.rand((8,), name="noise")
        return inner(y)

    assert outer.sites == ("outer/inner#0/dropout#0", "outer/noise[0]", "outer/noise[1]",
                           "outer/inner#1/dropout#0")


def test_explicit_keys():
    @truestar.function
    def f(x: Tensor[8], k: Tensor["i64"]):
        return ops.rand((8,), key=k, name="a"), ops.rand((8,), key=42, name="a2")

    a, b = f(torch.zeros(8), torch.tensor(7))
    assert torch.equal(a, rng.generate("uniform", (8,), torch.float32, 7, "f/a").double())
    assert torch.equal(b, rng.generate("uniform", (8,), torch.float32, 42, "f/a2").double())
    assert f.program.key is None            # no implicitly keyed op: no hidden key input


def test_errors_are_clear():
    with pytest.raises(FrontendError, match="duplicate_site|share the site"):
        @truestar.function
        def dup(x: Tensor[8]):
            return ops.rand((8,), name="a") + ops.rand((8,), name="a")
    with pytest.raises(FrontendError) as e:
        @truestar.function
        def torch_rand(x: Tensor[8]):
            return torch.rand(8) + x
    assert "ops.rand" in e.value.to_dict()["suggestion"]
    with pytest.raises(TypeError, match="needs key="):
        ops.randn((3,))
    with pytest.raises(FrontendError, match="dropout probability"):
        @truestar.function
        def p_one(x: Tensor[8]):
            return ops.dropout(x, 1.0)


def test_dropout_gradient_is_the_scaled_mask():
    from truestar.ir import run
    from truestar.ir.autodiff import grad

    @truestar.function(seed=9)
    def f(x: Tensor[64]):
        return ops.sum(ops.dropout(x, 0.25))

    g = grad(f.program, wrt=["x"])
    k = torch.tensor(rng.as_i64(f.key_for(0)))
    loss, dx = run(g, torch.randn(64), k)
    keep = rng.generate("bernoulli", (64,), torch.bool, f.key_for(0), "f/dropout#0", p=0.75)
    assert torch.allclose(dx, keep.double() / 0.75, rtol=0, atol=1e-15)


def test_eval_dropout_and_torch_alias():
    @truestar.function
    def f(x: Tensor[8]):
        return F.dropout(x, 0.5, False) + ops.dropout(x, 0.5, training=False)

    x = torch.randn(8)
    assert torch.equal(f(x), (x + x).double())
    assert f.sites == ()


def test_batch_norm_training_returns_new_state():
    @truestar.function
    def bn(x: Tensor["B", 4, 5, 5], w: Tensor[4], b: Tensor[4], m: Tensor[4], v: Tensor[4]):
        return ops.batch_norm(x, w, b, m, v, training=True, momentum=0.2)

    @truestar.function
    def bn_eval(x: Tensor["B", 4, 5, 5], w: Tensor[4], b: Tensor[4], m: Tensor[4], v: Tensor[4]):
        return ops.batch_norm(x, w, b, m, v)

    torch.manual_seed(0)
    w, b, m, v = torch.rand(4) + 0.5, torch.randn(4), torch.randn(4), torch.rand(4) + 0.5
    for n in (2, 3, 9):
        x = torch.randn(n, 4, 5, 5)
        y, m2, v2 = bn(x, w, b, m, v)
        rm, rv = m.double().clone(), v.double().clone()
        ref = F.batch_norm(x.double(), rm, rv, w.double(), b.double(), True, 0.2, 1e-5)
        assert (y - ref).abs().max() < 1e-12
        assert (m2 - rm).abs().max() < 1e-14 and (v2 - rv).abs().max() < 1e-14
        assert torch.equal(m, m) and not torch.equal(m2.float(), m)       # inputs are never mutated
        ye = bn_eval(x, w, b, m, v)
        assert (ye - F.batch_norm(x.double(), m.double(), v.double(), w.double(), b.double(), False)).abs().max() < 1e-12
    # the eager form returns the same triple
    y, m2, v2 = ops.batch_norm(x.double(), w.double(), b.double(), m.double(), v.double(), training=True, momentum=0.2)
    rm, rv = m.double().clone(), v.double().clone()
    F.batch_norm(x.double(), rm, rv, w.double(), b.double(), True, 0.2, 1e-5)
    assert torch.equal(m2, rm) and torch.equal(v2, rv)
