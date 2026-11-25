"""Kernel verification judges each element against its own magnitude (kernel/verify.py).

Before 2026-11-24 the error of every element was divided by the output's largest magnitude, so a
kernel could be completely wrong on its small elements and pass. Each test here shows the old rule
passing the broken kernel (the floor fractions set to infinity give exactly the old measure) and the
new rule failing it, and the correct kernel passing the new rule.
"""

import math

import numpy as np
import pytest
import torch

from truestar import kernel as K
from truestar.kernel import verify


def _has_triton():
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


BACKENDS = ["cpu"] + (["cuda"] if _has_triton() else [])
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def _old_rule(monkeypatch):
    """The size measure before 2026-11-24, alone: the lean test (docs/BIAS.md) is switched off too,
    since it refuses some of these broken kernels by itself (a wrong factor on small values leans
    one way: the float16 exp and the row sum cases on the CPU)."""
    monkeypatch.setattr(verify, "FLOOR_ELEMENTWISE", math.inf)
    monkeypatch.setattr(verify, "FLOOR_REDUCE", math.inf)
    monkeypatch.setenv("TRUESTAR_VERIFY_BIAS", "off")


# ----------------------------------------------------------------------------- compare


@pytest.mark.parametrize("dt", list(DTYPES))
def test_compare_catches_wrong_small_elements(dt):
    rng = np.random.default_rng(0)
    ref = np.exp(rng.normal(0.0, 2.0, 4096))  # magnitudes from about 1e-3 to 1e3
    tol = verify.BASE_TOL[dt]
    out = ref.copy()
    small = ref < 0.5 * tol * ref.max()  # wrong by 100% there: still under tol of the largest value
    assert small.sum() > 20
    out[small] *= 2.0
    old = verify.compare_numpy("y", out, ref, tol, math.inf)
    assert old.ok, old  # the old measure divides by the largest value: passes
    new = verify.compare_numpy("y", out, ref, tol, verify.FLOOR_ELEMENTWISE)
    assert not new.ok and new.error > 10 * tol, new
    i = int(new.at.strip("[]"))
    assert small[i], new.at
    # rounding to the working precision itself passes, with room to spare
    tdt = DTYPES[dt]
    rounded = torch.from_numpy(ref).to(tdt).double().numpy()
    ok = verify.compare_numpy("y", rounded, ref, tol, verify.FLOOR_ELEMENTWISE)
    assert ok.ok and ok.error < tol / 2, ok


def test_compare_torch_and_numpy_agree_on_the_element_rule():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    rng = np.random.default_rng(1)
    ref = np.exp(rng.normal(0.0, 2.0, (64, 33)))
    out = ref * (1 + 1e-4 * rng.normal(size=ref.shape))
    out[ref < 0.01] *= 1.5
    for floor in (verify.FLOOR_ELEMENTWISE, verify.FLOOR_REDUCE, math.inf):
        a = verify.compare_numpy("y", out, ref, 1e-3, floor)
        b = verify.compare_torch("y", torch.tensor(out, device="cuda"), torch.tensor(ref, device="cuda"),
                                 1e-3, floor)
        assert a.ok == b.ok and a.at == b.at and abs(a.error - b.error) <= 1e-12 * max(1.0, a.error)


def test_element_floor_never_looser_than_the_old_measure():
    ref = np.array([1.0, 1.0, 1.0, 1.0])
    out = ref + np.array([0.0, 0.0, 0.0, 1e-3])
    # flat output: rms equals the largest value, so the floor is capped there and the measure is
    # the old one exactly
    for floor in (verify.FLOOR_ELEMENTWISE, verify.FLOOR_REDUCE, verify.FLOOR_DEFAULT, math.inf):
        assert verify.compare_numpy("y", out, ref, 1.0, floor).error == pytest.approx(1e-3)


# ----------------------------------------------------------------------------- compiled kernels


def _exp_kernel(dt):
    M = K.sym("M")
    x = K.buffer("x", (M,), dt)
    i = K.axis("i", M)
    return K.kernel("expsmall", [K.output("y", (i,), K.exp(x[i]))])


def _wrong_below(limit, factor, be):
    """exp, wrong by `factor` where its result is below `limit` (a small value tail bug)."""
    if be == "cuda":
        return lambda e: f"tl.where(tl.exp({e}) < {limit}, tl.exp({e}) * {factor}, tl.exp({e}))"
    return lambda e: f"(math.exp({e}) * {factor} if math.exp({e}) < {limit} else math.exp({e}))"


@pytest.mark.parametrize("be", BACKENDS)
@pytest.mark.parametrize("dt", list(DTYPES))
def test_elementwise_kernel_wrong_only_on_small_values_is_caught(be, dt, monkeypatch):
    torch.manual_seed(0)
    x = (2.0 * torch.randn(4096)).to(be).to(DTYPES[dt])
    # completely wrong (3x) below 0.5 in half precision; 1.5x below 0.02 in float32
    limit, factor = (0.02, 1.5) if dt == "float32" else (0.5, 3.0)
    brk = {"exp": _wrong_below(limit, factor, be)}
    monkeypatch.setenv("TRUESTAR_VERIFY", "all")
    good = K.compile(_exp_kernel(dt), be)
    good(x=x)  # the correct kernel passes the element rule
    with pytest.raises(K.KernelMismatchError, match="at element"):
        K.compile(_exp_kernel(dt), be, _break=brk)(x=x)
    _old_rule(monkeypatch)
    K.compile(_exp_kernel(dt), be, _break=brk)(x=x)  # the old measure passes the broken kernel


def _rowsum_kernel(dt):
    M = K.sym("M")
    x = K.buffer("x", (M, 32), dt)
    i, r = K.axis("i", M), K.axis("r", 32)
    return K.kernel("rowsum", [K.output("y", (i,), K.reduce_sum(r, x[i, r]) * 1.0)])


def _wrong_small_mul(limit, factor, be):
    if be == "cuda":
        return lambda a, b: (f"tl.where(tl.abs({a}) < {limit}, ({a}) * ({b}) * {factor}, "
                             f"({a}) * ({b}))")
    return lambda a, b: f"(({a}) * ({b}) * {factor} if abs({a}) < {limit} else ({a}) * ({b}))"


@pytest.mark.parametrize("be", BACKENDS)
def test_reduction_kernel_wrong_on_small_rows_is_caught(be, monkeypatch):
    """Row sums where a few rows are 1000x larger than the rest (a peaky output, like attention
    scores): the rows of ordinary size are 0.5 percent wrong. Against the largest value that is
    under the float32 tolerance; against the floor of twice the output rms it is not."""
    torch.manual_seed(0)
    x = torch.randn(1024, 32)
    x[:4] *= 1000.0
    x = x.to(be)
    brk = {"mul": _wrong_small_mul(100.0, 1.005, be)}
    monkeypatch.setenv("TRUESTAR_VERIFY", "all")
    K.compile(_rowsum_kernel("float32"), be)(x=x)
    with pytest.raises(K.KernelMismatchError, match="rowsum"):
        K.compile(_rowsum_kernel("float32"), be, _break=brk)(x=x)
    _old_rule(monkeypatch)
    K.compile(_rowsum_kernel("float32"), be, _break=brk)(x=x)


def test_matching_infinities_compare_without_a_numpy_warning():
    # A masked attention row holds -inf on both sides. The difference used to be taken over every
    # element before masking, so -inf - -inf printed "RuntimeWarning: invalid value encountered in
    # subtract" on a clean run.
    import warnings

    out = np.array([1.0, -np.inf, 2.0, np.inf])
    ref = np.array([1.0, -np.inf, 2.0, np.inf])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chk = verify.compare_numpy("y", out, ref, 1e-6)
    assert chk.error == 0.0
