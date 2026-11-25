"""The kernel check's second measure for sums (kernel/verify.py): a sum whose terms cancel is judged
against the magnitude of its terms, so float32 rounding noise passes, while an error of a size no
summation order can produce is still refused."""

import numpy as np
import pytest

from truestar.kernel import ir, reference, verify


def _row_sum():
    n, m = ir.sym("N"), ir.sym("M")
    x = ir.buffer("x", (n, m), "float32")
    i, r = ir.axis("i", n), ir.axis("r", m)
    return ir.kernel("row_sum", [ir.output("o", [i], ir.reduce_sum(r, ir.Load(x, (i, r))))])


def _cancelling_rows(rng):
    """Rows of +-1e3 values that sum to nearly zero: the float64 result is tiny."""
    a = rng.standard_normal((8, 512)) * 1e3
    a[:, 256:] = -a[:, :256]
    a[:, 0] += 1e-4
    return a


def test_magnitude_kernel_sums_absolute_terms():
    k = _row_sum()
    mk = verify.magnitude_kernel(k)
    x = np.array([[1.0, -2.0, 3.0], [-1.0, 1.0, 0.0]])
    assert np.allclose(reference.run_reference(mk, {"x": x})["o"], [6.0, 2.0])
    no_sum = ir.kernel("neg", [ir.output("o", [ir.axis("i", 3)], -ir.Load(ir.buffer("y", (3,)), (ir.axis("i", 3),)))])
    assert verify.magnitude_kernel(no_sum) is None


def test_float32_rounding_of_a_cancelling_sum_passes():
    rng = np.random.default_rng(0)
    x = _cancelling_rows(rng)
    k = _row_sum()
    ref = reference.run_reference(k, {"x": x})
    out = {"o": x.astype(np.float32).sum(axis=1, dtype=np.float32).astype(np.float64)}
    tol = verify.tolerance(k, {"M": 512})
    plain = verify.compare("o", out["o"], ref["o"], tol)
    assert not plain.ok                           # the plain measure divides by the tiny result
    verify.check_outputs(k, "numpy", out, ref, tol, lambda: reference.run_reference(verify.magnitude_kernel(k), {"x": x}))


def test_a_real_error_in_a_sum_is_still_refused():
    rng = np.random.default_rng(1)
    x = _cancelling_rows(rng)
    k = _row_sum()
    ref = reference.run_reference(k, {"x": x})
    mag = reference.run_reference(verify.magnitude_kernel(k), {"x": x})["o"]
    wrong = {"o": ref["o"] + 1e-3 * mag}          # 1e-3 of the terms' magnitude: no rounding does that
    with pytest.raises(verify.KernelMismatchError, match="magnitude of its sums' terms"):
        verify.check_outputs(k, "numpy", wrong, ref, verify.tolerance(k, {"M": 512}),
                             lambda: reference.run_reference(verify.magnitude_kernel(k), {"x": x}))
    dropped = {"o": ref["o"] - x[:, :1].sum(axis=1)}  # a term left out of the sum
    with pytest.raises(verify.KernelMismatchError):
        verify.check_outputs(k, "numpy", dropped, ref, verify.tolerance(k, {"M": 512}),
                             lambda: reference.run_reference(verify.magnitude_kernel(k), {"x": x}))


def test_magnitude_is_per_element_so_a_small_wrong_element_still_fails():
    """Rows with large cancelling terms next to a row of small terms: the large rows' magnitudes do
    not lend a denominator to the small row, so a 50% error there is refused (element measure of
    tests/test_kernel_verify_elements.py, kept through the magnitude pass)."""
    rng = np.random.default_rng(2)
    x = _cancelling_rows(rng)
    x[3] = rng.random(512) * 1e-3          # small, all positive: no cancellation in this row
    k = _row_sum()
    ref = reference.run_reference(k, {"x": x})
    out = {"o": x.astype(np.float32).sum(axis=1, dtype=np.float32).astype(np.float64)}
    mag = lambda: reference.run_reference(verify.magnitude_kernel(k), {"x": x})  # noqa: E731
    verify.check_outputs(k, "numpy", out, ref, verify.tolerance(k, {"M": 512}), mag)   # rounding only: passes
    out["o"][3] *= 1.5
    with pytest.raises(verify.KernelMismatchError, match=r"element \[3\]"):
        verify.check_outputs(k, "numpy", out, ref, verify.tolerance(k, {"M": 512}), mag)
