"""Sliced float64 matmul on int8 tensor cores (kernel/f64emu.py, docs/SLICED_F64.md)."""
from fractions import Fraction

import pytest
import torch

from truestar import kernel as K
from truestar.kernel import f64emu as F
from truestar.kernel import reference as R

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _exact(a, b, pts):
    A, Bt = a.cpu().tolist(), b.cpu().t().tolist()
    return [sum(Fraction(x) * Fraction(y) for x, y in zip(A[i], Bt[j])) for i, j in pts]


def _cases():
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(300, 700, dtype=torch.float64, device="cuda", generator=g)
    b = torch.randn(700, 260, dtype=torch.float64, device="cuda", generator=g)
    yield "gaussian", a, b
    ea = torch.randint(-40, 40, a.shape, device="cuda", generator=g).double()
    eb = torch.randint(-40, 40, b.shape, device="cuda", generator=g).double()
    yield "wide exponents", a * torch.exp2(ea), b * torch.exp2(eb)
    c = a.clone()
    c[:, ::2] *= 1e-200          # half of every row far below the rest
    yield "tiny halves", c, b
    yield "ints", torch.randint(-1000, 1000, a.shape, device="cuda", generator=g).double(), \
        torch.randint(-1000, 1000, b.shape, device="cuda", generator=g).double()


@pytest.mark.parametrize("engine", ["triton", "int_mm"])
def test_every_entry_within_the_native_float64_guarantee(engine):
    for name, a, b in _cases():
        C, info = F.sliced_matmul(a, b, engine=engine, check_cpu=True)
        assert info.worst_bound_ratio <= 1.0 and info.cpu_checked, name
        g = torch.Generator().manual_seed(1)
        pts = list(zip(torch.randint(0, a.shape[0], (25,), generator=g).tolist(),
                       torch.randint(0, b.shape[1], (25,), generator=g).tolist()))
        for (i, j), ex in zip(pts, _exact(a, b, pts)):
            scale = Fraction(float((a[i].abs() @ b[:, j].abs()).item()))
            err = abs(Fraction(float(C[i, j])) - ex)
            assert err <= Fraction(F.gamma(a.shape[1])) * scale * (1 + Fraction(1, 10 ** 6)), (name, i, j)


def test_integer_matrices_are_exact():
    _, a, b = list(_cases())[3]
    C, _ = F.sliced_matmul(a, b)
    assert torch.equal(C, a @ b)        # every product and sum below 2**53: exact both ways


def test_engines_agree_bitwise():
    _, a, b = next(_cases())
    c1, i1 = F.sliced_matmul(a, b, engine="triton", slices=8, check_cpu=False)
    c2, i2 = F.sliced_matmul(a, b, engine="int_mm", slices=8, check_cpu=False)
    # both add the same exact integers per weight, smallest weight first: equal unless the Triton
    # path cut K into more chunks than one (not at K = 700)
    assert torch.equal(c1, c2)


def test_cpu_check_catches_a_wrong_result(monkeypatch):
    _, a, b = next(_cases())
    real = F._triton_products

    def broken(ad, bd, s, C, **kw):
        n = real(ad, bd, s, C, **kw)
        C[5, 7] += 1.0
        return n

    monkeypatch.setattr(F, "_triton_products", broken)
    F._checked_shapes.clear()
    with pytest.raises(F.SlicedF64Mismatch) as e:
        # row 5 must be among the sampled rows: sample all of them
        monkeypatch.setattr(F, "_cpu_check", lambda *args, **kw: real_check(*args, rows=a.shape[0], **kw))
        F.sliced_matmul(a, b, engine="triton")
    assert e.value.code == "TS1085"


real_check = F._cpu_check


def test_refusals():
    with pytest.raises(F.SlicedF64Error):
        F.sliced_matmul(torch.zeros(4, 4, dtype=torch.float64), torch.zeros(4, 4, dtype=torch.float64))
    a = torch.zeros(4, 4, dtype=torch.float64, device="cuda")
    a[0, 0] = float("inf")
    with pytest.raises(F.SlicedF64Error) as e:
        F.sliced_matmul(a, a)
    assert e.value.code == "TS1086"


def test_sliced_einsum_layouts():
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(3, 40, 64, dtype=torch.float64, device="cuda", generator=g)
    y = torch.randn(3, 64, 24, dtype=torch.float64, device="cuda", generator=g)
    for spec, u, v in (("bik,bkj->bij", x, y), ("bik,bkj->bji", x, y), ("ik,kj->ij", x[0], y[0]),
                       ("ki,kj->ij", x[0].t().contiguous(), y[0])):
        got = F.sliced_einsum(spec, u, v)
        assert torch.allclose(got, torch.einsum(spec, u, v), rtol=1e-13, atol=1e-13), spec
    assert F.sliced_einsum("ijk,kl->il", x, y[0]) is None   # j summed out of one operand only


def test_kernel_verification_uses_it_when_asked(monkeypatch):
    monkeypatch.setenv("TRUESTAR_VERIFY_F64", "sliced")
    monkeypatch.setenv("TRUESTAR_VERIFY_DEVICE", "gpu")
    monkeypatch.setattr(R, "SLICED_MIN_WORK", 1)
    M, N, Kd = 256, 192, 512
    a = K.buffer("a", (M, Kd), "float32")
    b = K.buffer("b", (Kd, N), "float32")
    i, j, r = K.axis("i", M), K.axis("j", N), K.axis("r", Kd)
    k = K.kernel("mm", [K.output("y", (i, j), K.reduce_sum(r, a[i, r] * b[r, j]))])
    before = R.sliced_stats["used"]
    fn = K.compile(k, "cuda")
    fn(a=torch.randn(M, Kd, device="cuda"), b=torch.randn(Kd, N, device="cuda"))
    assert R.sliced_stats["used"] > before
