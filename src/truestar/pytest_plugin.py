"""pytest helper for derived tolerances.

Enable it in a conftest.py with `pytest_plugins = ["truestar.pytest_plugin"]`, then:

    def test_my_softmax(assert_derived_close):
        x = torch.randn(64, 1000, dtype=torch.float16)
        assert_derived_close(lambda t: torch.softmax(t, -1), x, my_softmax(x))

The fixture is `truestar.assert_close`: no rtol or atol, the tolerance is derived from a float64
run of the reference function (docs/TOLERANCE.md). A failure is reported as an ordinary assertion
with the bad elements, their error in ulps, where they sit and a policy that would explain them.
Loading the plugin does not import torch; the first use does.
"""

import pytest


@pytest.fixture
def assert_derived_close():
    from .tolerance import assert_close

    return assert_close


@pytest.fixture
def derived_close_report():
    """`truestar.close_report`: the same check, returning the report instead of raising."""
    from .tolerance import close_report

    return close_report
