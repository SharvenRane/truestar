"""Operator local bounds for comparing two stacks (truestar.opbounds, docs/TOLERANCE.md)."""
import math

import torch

from truestar import opbounds as B
from truestar.codes import CODES


def _two_orders_matmul(a, b):
    """Two correct float32 matmuls that sum K in different orders (forward and reversed chunks)."""
    one = a @ b
    k = a.shape[1]
    perm = torch.randperm(k, generator=torch.Generator().manual_seed(1))
    other = torch.zeros_like(one)
    for c in torch.split(perm, 7):
        other = other + a[:, c] @ b[c]
    return one, other


def test_codes():
    assert CODES["TS1098"][1] == "local_bound_look_closer"
    assert CODES["TS1099"][1] == "local_bound_not_rounding"


def test_probable_grows_with_sqrt_n():
    u = 2.0 ** -24
    for n in (100, 10_000, 1_000_000):
        assert B.gamma_probable(n, u, 0.999) < B.gamma(n, u)
    r = B.gamma_probable(10 ** 6, u, 0.999) / B.gamma_probable(10 ** 4, u, 0.999)
    assert 8 < r < 13            # about sqrt(100) = 10, the log factor of lam moves it a little


def test_reordered_matmul_is_rounding_and_fault_is_not():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(64, 2048, generator=g), torch.randn(2048, 48, generator=g)
    one, other = _two_orders_matmul(a, b)
    assert not torch.equal(one, other)
    bd = B.bound_matmul(a, b, acc="float32", out_dtype="float32")
    v = B.compare("matmul", bd, one, other)
    assert v.verdict == "rounding" and v.code is None and v.probable_ratio < 1
    assert v.worst_ratio < v.probable_ratio / 5     # the probable bound is much tighter
    bad = other.clone()
    bad[3, 4] += 0.01 * float((a[3].abs() * b[:, 4].abs()).sum())   # 1% of the absolute sum
    v = B.compare("matmul", bd, one, bad)
    assert v.verdict == "not_rounding" and v.code == "TS1099" and v.beyond_worst == 1


def test_narrower_accumulation_than_declared_is_flagged():
    g = torch.Generator().manual_seed(2)
    a, b = torch.randn(32, 4096, generator=g), torch.randn(4096, 32, generator=g)
    exact = a.double() @ b.double()
    half_acc = torch.zeros(32, 32, dtype=torch.float16)   # a running float16 accumulator
    for k0 in range(0, 4096, 64):
        half_acc = (half_acc.float() + a[:, k0:k0 + 64] @ b[k0:k0 + 64]).half()
    half_acc = half_acc.float()
    v = B.compare("matmul", B.bound_matmul(a, b, acc="float32"), exact.float(), half_acc)
    assert v.code in ("TS1098", "TS1099")


def test_sum_mean_softmax_elementwise():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(16, 5000, generator=g)
    perm = torch.randperm(5000, generator=g)
    s1, s2 = x.sum(1), x[:, perm].sum(1)
    assert B.compare("sum", B.bound_sum(x, 1), s1, s2).verdict == "rounding"
    m1, m2 = x.mean(1), x[:, perm].mean(1)
    assert B.compare("mean", B.bound_sum(x, 1, mean=True), m1, m2).verdict == "rounding"
    y1 = torch.softmax(x, 1)
    y2 = torch.softmax(x.double(), 1).float()
    assert B.compare("softmax", B.bound_softmax(x, 1), y1, y2).verdict == "rounding"
    e1, e2 = torch.exp(x), torch.exp(x.double()).float()
    assert B.compare("exp", B.bound_elementwise(torch.exp, x, ulps=2), e1, e2).verdict == "rounding"
    e3 = e2.clone()
    e3[0, 0] = e3[0, 0] * (1 + 1e-5)
    assert B.compare("exp", B.bound_elementwise(torch.exp, x, ulps=2), e1, e3).code == "TS1099"


def test_softmax_bound_covers_both_exp_errors():
    """A correct kernel whose exp is 2 ulps (4 u) high on x_0 and 4 u low on the others is
    8 u y_0 (1 - y_0) off on y_0 (scripts/bound_crosscheck.py found the old bound, which counted
    one exp error, under that first order worst case on short rows)."""
    u = 2.0 ** -24
    x = torch.tensor([[-6.0, 0.0], [-3.0, 1.0], [-8.0, -1.0]], dtype=torch.float32)
    x64 = x.double()
    e = torch.exp(x64)
    sign = torch.tensor([1.0, -1.0], dtype=torch.float64)
    e_hat = e * (1 + 4 * u * sign)
    y_hat = e_hat / e_hat.sum(1, keepdim=True)
    y = torch.softmax(x64, 1)
    err = (y_hat - y).abs()
    bound = B.bound_softmax(x, 1, acc="float32", exp_ulps=2.0).worst
    assert bool((err[:, 0] <= bound[:, 0]).all())
    assert bool((err[:, 0] > y[:, 0] * (4 * u + B.gamma(2, u) + u)).all())  # the old bound


def test_probable_bound_holds_on_random_sums():
    """Coverage check of the probabilistic shape: over 2000 reordered float32 sums of 4096 terms,
    no difference exceeds the pairwise probable bound at 0.999 confidence."""
    g = torch.Generator().manual_seed(4)
    x = torch.randn(2000, 4096, generator=g) + 0.3
    seq = torch.zeros(2000)
    for j in range(0, 4096, 512):     # chunked sequential, a different tree from torch.sum
        seq = seq + x[:, j:j + 512].sum(1)
    v = B.compare("sum", B.bound_sum(x, 1, confidence=0.999), x.sum(1), seq)
    assert v.beyond_probable == 0 and not math.isinf(v.probable_ratio)
