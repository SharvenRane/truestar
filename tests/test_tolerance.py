"""Derived tolerances (`truestar.assert_close`, R6): correct kernels pass, wrong ones fail with a
named reason, against hand picked tolerances that do one or the other."""

import math

import pytest
import torch

import truestar
from truestar import kernelbench
from truestar.tolerance import (LAMBDA_RMS, ToleranceError, clear_cache, close_report, derive,
                                policy_for)

pytest_plugins = ["truestar.pytest_plugin"]


def _inputs(case, draw=0):
    g = torch.Generator().manual_seed(1000 + draw)
    return case.make_inputs(case.dims, case.dtype, g)


VISIBLE = [c for c in kernelbench.cases() if not c.needs]
# A constant off by 1e-3 in float16 is about one to two ulps: within what a correct float16 kernel
# that rounds twice shows, so it is out of reach by design (docs/TOLERANCE.md). The bfloat16 last
# row tile scaled by 1 + 2^-7 is about one bfloat16 ulp: a measured miss (docs/KERNEL_CHECK.md,
# held out bound 5), not a design choice.
BELOW_PRECISION = {"mm_scale_1e-3_fp16", "mm_last_row_tile_scale_bf16"}


@pytest.mark.parametrize("case", [c for c in VISIBLE if c.correct], ids=lambda c: c.name)
def test_correct_kernels_pass(case):
    for draw in range(2):
        clear_cache()
        x = _inputs(case, draw)
        rep = close_report(case.reference, x, case.candidate(*x))
        assert rep.ok, rep.message()
        assert rep.consistent_with_noise


@pytest.mark.parametrize("case", [c for c in VISIBLE if not c.correct
                                  and c.name not in BELOW_PRECISION], ids=lambda c: c.name)
def test_wrong_kernels_fail(case):
    clear_cache()
    x = _inputs(case)
    rep = close_report(case.reference, x, case.candidate(*x))
    assert not rep.ok, f"{case.name} ({case.bug}) passed: {rep.message()}"


def test_rates_against_torch_defaults():
    """The R6 acceptance numbers in small: zero false fails where torch's default assert_close
    false fails, and fewer false passes than a loose allclose."""
    ours_ff = torch_ff = ours_fp = loose_fp = 0
    for case in VISIBLE:
        x = _inputs(case)
        out = case.candidate(*x)
        same = case.reference(*x)
        clear_cache()
        ok = close_report(case.reference, x, out).ok
        try:
            torch.testing.assert_close(out, same)
            t_ok = True
        except AssertionError:
            t_ok = False
        loose = bool(torch.allclose(out.double(), same.double(), rtol=1e-2, atol=1e-2))
        if case.correct:
            ours_ff += not ok
            torch_ff += not t_ok
        else:
            ours_fp += ok
            loose_fp += loose
    assert ours_ff == 0
    assert torch_ff >= 5  # blocked fp32 GEMM, split K, attention, offset layer norm ...
    assert ours_fp <= len(BELOW_PRECISION)
    assert loose_fp >= 4  # wrong scale, wrong eps, tanh GELU, a constant off by 1e-5


def test_rounded_run_always_passes():
    """The correctly rounded run is a correct implementation by definition."""
    from truestar.gradcheck import _rounding_mode

    x = torch.randn(32, 300, dtype=torch.float64).half()
    f = lambda t: torch.softmax(t * 3, -1) @ t.transpose(0, 1)  # noqa: E731
    with _rounding_mode(torch.float16):
        r = f(x.double())
    rep = close_report(f, (x,), r.half())
    assert rep.ok, rep.message()


def test_failure_message_names_location_ulps_and_cause():
    case = next(c for c in VISIBLE if c.name == "mm_fp16_accumulation")
    x = _inputs(case)
    with pytest.raises(ToleranceError) as ei:
        truestar.assert_close(case.reference, x, case.candidate(*x))
    msg = str(ei.value)
    assert "ulp of float16" in msg and "float16 accumulation" in msg
    assert "Not reordering noise" in msg
    assert ei.value.report.worst.bad > 0

    tail = next(c for c in VISIBLE if c.name == "mm_column_tail_fp16")
    x = _inputs(tail)
    rep = close_report(tail.reference, x, tail.candidate(*x))
    assert "tail of dim 1 (indices 192 to 199 of 200)" in rep.message()

    edge = next(c for c in VISIBLE if c.name == "attn_mask_edge_fp16")
    x = _inputs(edge)
    rep = close_report(edge.reference, x, edge.candidate(*x))
    assert "indices 128 to 190 of dim 2" in rep.message(), rep.message()

    tf = next(c for c in VISIBLE if c.name == "mm_tf32_fp32")
    x = _inputs(tf)
    rep = close_report(tf.reference, x, tf.candidate(*x))
    assert "TF32" in rep.hint


def test_tf32_policy_follows_flags_and_override():
    assert not policy_for(torch.float32, "cpu").tf32_mm
    assert policy_for(torch.float32, "cpu", tf32=True).tf32_mm
    assert not policy_for(torch.float16, "cuda", tf32=True).tf32_mm
    case = next(c for c in VISIBLE if c.name == "mm_tf32_fp32")
    x = _inputs(case)
    assert close_report(case.reference, x, case.candidate(*x), tf32=True).ok


def test_nonfinite_handling():
    x = torch.randn(8, 16)
    x[0, 3] = float("nan")
    f = lambda t: t * 2  # noqa: E731
    assert close_report(f, (x,), x * 2).ok
    y = x * 2
    y[1, 1] = float("nan")
    rep = close_report(f, (x,), y)
    assert not rep.ok and rep.outputs[0].nonfinite_mismatch == 1
    y = x * 2
    y[0, 3] = 0.0  # NaN dropped where the reference has one
    assert not close_report(f, (x,), y).ok


def test_overflow_the_dtype_cannot_resolve_is_not_a_failure():
    """exp(12) overflows float16 in a per op rounded run but the float64 product does not: a
    kernel that keeps a float32 intermediate (finite) and one that does not (inf) both pass."""
    x = torch.full((4,), 12.0, dtype=torch.float16)
    f = lambda t: torch.exp(t) * 1e-3  # noqa: E731
    good = (torch.exp(x.float()) * 1e-3).half()
    assert close_report(f, (x,), good).ok
    assert close_report(f, (x,), torch.exp(x) * 1e-3).outputs[0].unresolved == 4


def test_tuple_outputs_module_and_shape_mismatch():
    lin = torch.nn.Linear(64, 32).half()
    x = torch.randn(8, 64).half()
    out = (lin.float()(x.float())).half()
    assert close_report(lin, (x,), out).ok
    f = lambda t: (t.sum(-1), t.amax(-1))  # noqa: E731
    t = torch.randn(4, 100)
    assert close_report(f, (t,), f(t)).ok
    rep = close_report(f, (t,), (t.sum(-1), t.amax(-1)[:3]))
    assert not rep.ok and rep.outputs[1].shape_mismatch


def test_derived_reference_is_cached_per_input_signature():
    clear_cache()
    a, b = torch.randn(64, 2048), torch.randn(2048, 64)
    t1 = derive(kernelbench.mm_ref, (a, b))
    t2 = derive(kernelbench.mm_ref, (a, b))
    assert t1 is t2
    a.add_(0)  # an in place write bumps the version: derive again
    assert derive(kernelbench.mm_ref, (a, b)) is not t1


def test_cache_never_serves_another_tensor_at_the_same_address():
    """A new input can land at a freed input's address with the same version counter; it must not
    get that input's derived reference (found by the R1 benchmark: a stale reference)."""
    clear_cache()
    f = torch.nn.functional.gelu
    for seed in range(6):
        x = torch.randn(33, 127, generator=torch.Generator().manual_seed(seed))
        assert close_report(f, (x,), f(x)).ok
        del x


def test_cache_keeps_no_input_or_module_alive():
    """The cache holds its inputs and fn weakly: once the caller drops them, they and the entry's
    float64 references are freed by reference counting, with no garbage collection."""
    import gc
    import weakref
    from truestar import tolerance as T
    clear_cache()
    lin = torch.nn.Linear(64, 32)
    x = torch.randn(8, 64)
    tol = derive(lin, (x,))
    assert derive(lin, (x,)) is tol and len(T._CACHE) == 1      # still a hit while both live
    refs = [weakref.ref(o) for o in (lin, x, tol)]
    gc.disable()
    try:
        del lin, x, tol
        assert all(r() is None for r in refs) and len(T._CACHE) == 0
    finally:
        gc.enable()


def test_flatten_tensors_keeps_order_and_no_reference():
    import gc
    import weakref
    from truestar.capture import flatten_tensors
    a, b, c, d = (torch.zeros(1) for _ in range(4))
    assert [id(t) for t in flatten_tensors([a, {"y": c, "x": (b,)}, [[d]]])] == [id(a), id(b), id(c), id(d)]
    r = weakref.ref(a)
    gc.disable()
    try:
        flatten_tensors(([a], {"k": a}))
        del a
        assert r() is None                  # the call left no cycle holding the tensor
    finally:
        gc.enable()


def test_reduced_precision_split_k_and_gated_activations_are_modelled():
    """cuBLAS may round split K partial sums to float16 when PyTorch allows it (the default);
    float32 GELU cancels to about 4 u |x| where the output is tiny. Both are correct."""
    p = policy_for(torch.float16, "cuda")
    assert p.reduced_precision_reduction == bool(
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    x = torch.linspace(-6, 6, 4001)
    assert close_report(torch.nn.functional.gelu, (x,), torch.nn.functional.gelu(x)).ok


def test_cancellation_inputs_do_not_false_fail():
    """Running sums that grow and then cancel (all positive, then all negative) are the worst case
    for accumulation; a sequential float32 kernel there must still pass."""
    k = 4096
    a = torch.cat([torch.rand(16, k // 2) + 1, -(torch.rand(16, k // 2) + 1)], dim=1)
    b = torch.rand(k, 8) + 1
    seq = torch.zeros(16, 8)
    for k0 in range(k):
        seq = seq + a[:, k0:k0 + 1] * b[k0:k0 + 1]
    rep = close_report(kernelbench.mm_ref, (a, b), seq)
    assert rep.ok, rep.message()


def test_pytest_fixture(assert_derived_close):
    x = torch.randn(16, 300, dtype=torch.float16)
    assert_derived_close(lambda t: torch.softmax(t, -1), (x,), torch.softmax(x.float(), -1).half())
    with pytest.raises(AssertionError):
        bad = torch.softmax(x.float(), -1).half()
        bad[:, -1] = 0
        assert_derived_close(lambda t: torch.softmax(t, -1), (x,), bad)


def test_lambda_rms_is_the_documented_limit():
    assert LAMBDA_RMS == 2.0 and math.isfinite(LAMBDA_RMS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_tf32_and_cublas_pass_under_the_device_policy():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(256, 2048, generator=g).cuda()
    b = torch.randn(2048, 200, generator=g).cuda()
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        out = a @ b
        assert close_report(kernelbench.mm_ref, (a, b), out).ok
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    assert not close_report(kernelbench.mm_ref, (a, b), out).ok  # IEEE policy: TF32 is wrong
    h = (a.half(), b.half())
    assert close_report(kernelbench.mm_ref, h, h[0] @ h[1]).ok


def _repro():
    import importlib.util
    import os

    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts",
                     "repro_heldout_misses.py")
    spec = importlib.util.spec_from_file_location("repro_heldout_misses", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", ["sum, constant tensor", "mean, constant tensor",
                                  "cumsum, constant tensor", "masked cumsum, constant tensor",
                                  "instance norm"])
def test_held_out_bound_misses_are_derived_now(name):
    """Held out false fails (sequential float32 sums of a constant tensor, two pass instance norm)
    pass by the derived model, and the same kernels with a small real fault still fail."""
    ref, cand, x = _repro().cases()[name]
    clear_cache()
    out = cand(*x)
    assert close_report(ref, x, out).ok
    if name == "instance norm":
        bad = out.clone()
        bad[:, -1] = bad[:, -1] * (1 + 2 ** -10)  # one channel scaled by 0.1%
    else:
        bad = out * (1 + 2 ** -12)  # 2.4e-4: a stale scale, far below allclose 1e-2
    clear_cache()
    assert not close_report(ref, x, bad).ok
