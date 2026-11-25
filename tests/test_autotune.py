"""`@truestar.verified_autotune` (R7): every Triton autotune config is checked against float64
before it is timed; a wrong config is removed with a named reason, never picked."""

import warnings

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and Triton")


def _mm(tuned, a, b, kmask):
    import triton

    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))  # noqa: E731
    tuned[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1), KMASK=kmask)
    return c


def _tuned(configs=None):
    import truestar
    from truestar.kernelbench_triton import AUTOTUNE_MM_CONFIGS, autotune_mm_kernel

    return truestar.verified_autotune(configs or AUTOTUNE_MM_CONFIGS, key=["M", "N", "K"],
                                      reference=lambda a, b: a @ b, inputs=["a_ptr", "b_ptr"],
                                      outputs=["c_ptr"], warmup=5, rep=10)(autotune_mm_kernel)


def _inputs(K):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(256, K, generator=g).cuda().half(),
            torch.randn(K, 192, generator=g).cuda().half())


def test_wrong_configs_are_removed_with_a_reason_and_never_picked():
    from truestar.autotune import VerifiedAutotuneWarning
    from truestar.tolerance import close_report

    a, b = _inputs(1040)  # only BK=16 divides 1040; the kernel has no K mask
    tuned = _tuned()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", VerifiedAutotuneWarning)
        c = _mm(tuned, a, b, False)
    assert close_report(lambda x, y: x @ y, (a, b), c).ok
    rep = next(iter(tuned.reports.values()))
    removed = {r.config for r in rep.rejected}
    assert len(removed) == 4 and all("BK=16" not in r for r in removed)
    assert all(r.code == "TS207" and "elements" in r.reason for r in rep.rejected)
    assert "BK=16" in rep.best
    assert sum("[TS207]" in str(x.message) for x in w) == 4


def test_every_config_right_keeps_every_config():
    a, b = _inputs(1024)
    tuned = _tuned()
    _mm(tuned, a, b, False)
    rep = next(iter(tuned.reports.values()))
    assert not rep.rejected and all(r.ms is not None for r in rep.results)


def test_no_right_config_raises_instead_of_returning_a_wrong_answer():
    import triton

    from truestar.autotune import VerifiedAutotuneError

    a, b = _inputs(1040)
    wrong_only = [triton.Config({"BM": 64, "BN": 64, "BK": 32}, num_warps=4)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(VerifiedAutotuneError, match="no autotune config gives the right"):
            _mm(_tuned(wrong_only), a, b, False)


def test_bad_argument_names_are_refused():
    import truestar
    from truestar.kernelbench_triton import autotune_softmax_kernel

    with pytest.raises(ValueError, match="not an argument"):
        truestar.verified_autotune([], key=["N"], reference=lambda x: x, inputs=["x"],
                                   outputs=["y_ptr"])(autotune_softmax_kernel)
