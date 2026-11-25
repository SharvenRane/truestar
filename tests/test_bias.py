"""The lean test on rounding errors (truestar.bias, docs/BIAS.md).

The size checks bound how large an error is; these tests show kernels whose errors are small but
lean one way (a truncating store, round toward zero accumulation, a biased exp) passing the size
check and being refused by the lean test (TS960), and correct rounding never being refused: round
to nearest in every precision, a constant rounded once, a softmax whose row shares one rounded
denominator, and real compiled CPU kernels.
"""

import math

import numpy as np
import pytest
import torch
from torch import nn

from truestar import bias as B
from truestar import kernel as K
from truestar.kernel import ir, reference, verify


def _rn(x, dt):
    return torch.from_numpy(np.asarray(x, dtype=np.float64)).to(dt).double().numpy()


def _trunc(x, keep_bits):
    """Round toward zero to a float32 with only ``keep_bits`` of its 23 stored mantissa bits."""
    a = torch.from_numpy(np.asarray(x, dtype=np.float32).copy()).view(torch.int32)
    mask = ~((1 << (23 - keep_bits)) - 1)
    return (a & mask).view(torch.float32).double().numpy()


def _rz32(x):
    """Round toward zero to float32 (from float64)."""
    f = np.asarray(x, dtype=np.float64).astype(np.float32)
    over = np.abs(f.astype(np.float64)) > np.abs(x)
    f[over] = np.nextafter(f[over], np.float32(0))
    return f


# ----------------------------------------------------------------------------- the statistic


def test_t_pvalue_matches_closed_forms():
    for t in (0.0, 0.5, 1.0, 3.0, 10.0, 100.0):
        assert math.isclose(B.t_pvalue(t, 1), 1 - 2 / math.pi * math.atan(t), rel_tol=1e-9, abs_tol=1e-15)
        assert math.isclose(B.t_pvalue(t, 2), 1 - t / math.sqrt(2 + t * t), rel_tol=1e-9, abs_tol=1e-15)
    assert math.isclose(B.t_pvalue(5.0, 1e8), math.erfc(5 / math.sqrt(2)), rel_tol=1e-12)
    assert B.t_pvalue(math.inf, 10) == 0.0 and B.t_pvalue(float("nan"), 10) == 1.0


@pytest.mark.parametrize("dt,tol", [(torch.float32, 2e-5), (torch.float16, 2e-3), (torch.bfloat16, 1.6e-2)])
def test_round_to_nearest_never_fires(dt, tol):
    worst = 0.0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        ref = np.exp(rng.normal(0.0, 2.0, (48, 96))) * rng.choice([-1.0, 1.0], (48, 96))
        r = B.lean_test("y", _rn(ref, dt), ref, tol * B.BIAS_FRACTION, verify.FLOOR_ELEMENTWISE, [(0, 1)])
        assert not r.biased, r.describe()
        worst = max(worst, r.margin())
    assert worst < 0.5


def test_a_constant_rounded_once_is_inside_the_effect_floor():
    """x * fl32(1/3): every element shares the constant's rounding, a lean no grouping can tell from
    bias; it is what the effect floor allows for (under 1 u against a floor of 5.2 u)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 128))
    out = (x.astype(np.float32) * np.float32(1 / 3)).astype(np.float64)
    r = B.lean_test("y", out, x / 3, 2e-5 * B.BIAS_FRACTION, verify.FLOOR_ELEMENTWISE, [(0, 1)])
    assert not r.biased
    out_lean = [l for l in r.leans if l.direction == B.OUTWARD][0]
    assert out_lean.p < 1e-6               # significant: the shared rounding is real ...
    assert abs(out_lean.mean) < r.floor     # ... and below the floor


def test_a_shared_row_rounding_needs_its_grouping():
    """Rows that share one rounded value (a softmax denominator perturbed by 3 u per row) have
    correlated errors: treated as independent elements they look significant; grouped by row, the
    test sees 64 independent rows and stays quiet."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(64, 512))
    e = np.exp(x - x.max(1, keepdims=True))
    ref = e / e.sum(1, keepdims=True)
    shared = 1 + 3 * 2.0 ** -24 * rng.choice([-1.0, 1.0], (64, 1)) * rng.uniform(0.5, 1, (64, 1))
    out = _rn(ref * shared, torch.float32)
    floor = 1e-9                           # take the effect floor out, so only significance decides
    naive = B.lean_test("y", out, ref, floor, 2.0, [(0, 1)])
    grouped = B.lean_test("y", out, ref, floor, 2.0, [(0,), (0, 1)])
    assert min(l.p for l in naive.leans) < 1e-6
    assert all(l.p > 1e-6 for l in grouped.leans) and not grouped.biased
    assert {l.grouping for l in grouped.leans} == {"dims (0,)"}


def test_blackbox_groupings_skip_few_groups():
    assert B.all_groupings((4, 64, 32)) == [(1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]


# ----------------------------------------------------------------------------- kernel groupings


def test_error_groupings_follow_the_kernel_ir():
    n, m = ir.sym("N"), ir.sym("M")
    x = ir.buffer("x", (n, m), "float32")
    i, j = ir.axis("i", n), ir.axis("j", m)
    sm = ir.kernel("sm", [ir.output("y", [i, j], ir.softmax_weight(x[i, j] * 0.125, j))])
    assert verify.error_groupings(sm, sm.outputs[0]) == [(0,), (0, 1)]       # per row denominators
    ew = ir.kernel("ew", [ir.output("y", [i, j], ir.exp(x[i, j] * (1.0 / 3.0)))])
    assert verify.error_groupings(ew, ew.outputs[0]) == [(0, 1)]             # the constant is left out
    b = ir.buffer("b", (m,), "float64")
    bc = ir.kernel("bc", [ir.output("y", [i, j], x[i, j] + b[j], "float32")])
    assert verify.error_groupings(bc, bc.outputs[0]) == [(0, 1), (1,)]       # float64 load rounds per column
    r = ir.axis("r", m)
    g = ir.kernel("g", [ir.output("y", [i, j], x[i, j] / ir.reduce_sum(r, ir.reduce_sum(ir.axis("q", n), x[ir.axis("q", n), r])))])
    assert () in verify.error_groupings(g, g.outputs[0])                    # global sum: one group


# ----------------------------------------------------------------------------- injected biased kernels


def _kernel_inputs(rng, shape):
    return {"x": rng.normal(0.0, 1.0, shape).astype(np.float32).astype(np.float64)}


def _assert_caught(k, outs, refs, tol, monkeypatch):
    monkeypatch.setenv("TRUESTAR_VERIFY_BIAS", "off")
    verify.check_outputs(k, "test", outs, refs, tol)          # the size check passes it
    monkeypatch.setenv("TRUESTAR_VERIFY_BIAS", "on")
    with pytest.raises(verify.KernelBiasError, match="TS960"):
        verify.check_outputs(k, "test", outs, refs, tol)


def test_truncating_bfloat16_store_is_caught(monkeypatch):
    n = ir.sym("N")
    x = ir.buffer("x", (n, 256), "float32")
    i, j = ir.axis("i", n), ir.axis("j", 256)
    k = ir.kernel("gelu_bf16", [ir.output("y", [i, j], x[i, j] * ir.sigmoid(x[i, j] * 1.702), "bfloat16")])
    inp = _kernel_inputs(np.random.default_rng(0), (32, 256))
    refs = reference.run_reference(k, inp)
    tol = verify.tolerance(k, {"N": 32})
    good = {"y": _rn(refs["y"], torch.bfloat16)}
    verify.check_outputs(k, "test", good, refs, tol)          # round to nearest: passes both checks
    bad = {"y": _trunc(refs["y"], 7)}                          # bfloat16 by dropping 16 bits
    _assert_caught(k, bad, refs, tol, monkeypatch)


def test_round_toward_zero_accumulation_is_caught(monkeypatch):
    """A softmax denominator style row sum of positive terms, every partial sum rounded toward zero."""
    m = 1024
    x = ir.buffer("x", (64, m), "float32")
    i, r = ir.axis("i", 64), ir.axis("r", m)
    k = ir.kernel("rowsum_exp", [ir.output("s", [i], ir.reduce_sum(r, ir.exp(x[i, r])))])
    inp = _kernel_inputs(np.random.default_rng(1), (64, m))
    refs = reference.run_reference(k, inp)
    terms = np.exp(inp["x"]).astype(np.float32)
    acc = np.zeros(64, np.float32)
    for c in range(m):
        acc = _rz32(acc.astype(np.float64) + terms[:, c].astype(np.float64))
    tol = verify.tolerance(k)
    rn = np.zeros(64, np.float32)
    for c in range(m):
        rn = (rn + terms[:, c]).astype(np.float32)
    verify.check_outputs(k, "test", {"s": rn.astype(np.float64)}, refs, tol)   # round to nearest passes
    _assert_caught(k, {"s": acc.astype(np.float64)}, refs, tol, monkeypatch)


def test_biased_exp_is_caught(monkeypatch):
    """exp(x) as 2^(x * c) with c = log2(e) made 2^-20 too small: always high for the x <= 0 of a
    stable softmax, by under half the tolerance."""
    x = ir.buffer("x", (128, 128), "float32")
    i, j = ir.axis("i", 128), ir.axis("j", 128)
    k = ir.kernel("exp_shifted", [ir.output("y", [i, j], ir.exp(x[i, j]))])
    rng = np.random.default_rng(2)
    inp = {"x": -np.abs(rng.normal(0.0, 3.0, (128, 128))).astype(np.float32).astype(np.float64)}
    refs = reference.run_reference(k, inp)
    c = np.float32(np.log2(np.e) * (1 - 2.0 ** -20))
    out = np.exp2(inp["x"].astype(np.float32) * c).astype(np.float32).astype(np.float64)
    tol = verify.tolerance(k)
    assert verify.compare("y", out, refs["y"], tol, verify.FLOOR_ELEMENTWISE).error < tol / 2
    _assert_caught(k, {"y": out}, refs, tol, monkeypatch)


def test_warn_mode_reports_without_refusing(monkeypatch, tmp_path):
    n = ir.sym("N")
    x = ir.buffer("x", (n, 256), "float32")
    i, j = ir.axis("i", n), ir.axis("j", 256)
    k = ir.kernel("cp_bf16", [ir.output("y", [i, j], x[i, j] * 3.0, "bfloat16")])
    inp = _kernel_inputs(np.random.default_rng(4), (16, 256))
    refs = reference.run_reference(k, inp)
    log = tmp_path / "lean.jsonl"
    monkeypatch.setenv("TRUESTAR_VERIFY_BIAS", "warn")
    monkeypatch.setenv("TRUESTAR_BIAS_LOG", str(log))
    with pytest.warns(verify.KernelBiasWarning, match="toward zero"):
        verify.check_outputs(k, "test", {"y": _trunc(refs["y"], 7)}, refs, verify.tolerance(k, {"N": 16}))
    line = log.read_text().splitlines()[0]
    assert '"biased": true' in line and '"kernel": "cp_bf16"' in line


# ----------------------------------------------------------------------------- real compiled kernels


def _numba():
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _numba(), reason="the CPU kernel backend needs numba")
@pytest.mark.parametrize("dt", ["float32", "float16", "bfloat16"])
def test_compiled_cpu_kernels_pass_and_a_truncating_one_is_refused(dt, monkeypatch, tmp_path):
    log = tmp_path / "lean.jsonl"
    monkeypatch.setenv("TRUESTAR_BIAS_LOG", str(log))
    n = ir.sym("N")
    x = ir.buffer("x", (n, 192), dt)
    i, j = ir.axis("i", n), ir.axis("j", 192)
    tdt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dt]
    kernels = [
        ir.kernel(f"sm_{dt}", [ir.output("y", [i, j], ir.softmax_weight(x[i, j] * 0.125, j))]),
        ir.kernel(f"gelu_{dt}", [ir.output("y", [i, j], 0.5 * x[i, j] * (1.0 + ir.erf(x[i, j] * 0.7071067811865476)))]),
        ir.kernel(f"rs_{dt}", [ir.output("y", [i], ir.reduce_sum(j, x[i, j] * x[i, j]))]),
    ]
    xt = torch.randn(96, 192, generator=torch.Generator().manual_seed(0)).to(tdt)
    for k in kernels:
        K.compile(k, "cpu")(xt)                               # verified on the first call, lean test on
    lines = log.read_text().splitlines()
    assert len(lines) == len(kernels) and all('"biased": false' in l for l in lines)

    from truestar.kernel import compiler as KC
    k = ir.kernel(f"scale_{dt}_t", [ir.output("y", [i, j], x[i, j] * 1.7)])
    fn = K.compile(k, "cpu")
    real = KC.CompiledKernel._run_cpu

    def truncating(self, got, L, chk=None):
        """The kernel's float32 value stored by dropping bits instead of rounding: to bfloat16
        (7 bits) or float16 (10 bits) by truncation, or for float32 a store that drops 5 bits."""
        outs = real(self, got, L) if chk is None else real(self, got, L, chk)
        keep = {"float32": 18, "float16": 10, "bfloat16": 7}[dt]
        y32 = got["x"].double().numpy() * 1.7
        return {"y": torch.from_numpy(_trunc(y32, keep)).to(outs["y"].dtype)}
    monkeypatch.setattr(KC.CompiledKernel, "_run_cpu", truncating)
    with pytest.raises(verify.KernelBiasError, match="toward zero"):
        fn(xt)


# ----------------------------------------------------------------------------- truestar check


class _Truncating(nn.Module):
    """A linear layer whose float32 result loses its low 8 mantissa bits (round toward zero), as a
    backend kernel with a truncating store would; float64 (the reference) is left exact."""

    def __init__(self, bits=15):
        super().__init__()
        self.lin = nn.Linear(64, 64)
        self.bits = bits

    def forward(self, x):
        y = self.lin(x)
        if y.dtype == torch.float32:
            mask = ~((1 << (23 - self.bits)) - 1)
            y = (y.view(torch.int32) & mask).view(torch.float32)
        return torch.relu(y)


def test_check_flags_a_leaning_module_the_size_check_passes():
    import truestar
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(32, 64), nn.GELU(), _Truncating(), nn.Linear(64, 8))
    rep = truestar.check(m, torch.randn(64, 32), backends=["cpu"], drill_ops=False)
    r = rep.results[0]
    assert "TS961" in r.codes and "TS010" not in r.codes and "TS011" not in r.codes, rep.to_text()
    issue = [i for i in r.issues if i.code == "TS961"][0]
    assert "2#0 (_Truncating)" in issue.message and "toward zero" in issue.message
    lay = {l.key: l for l in r.layers}
    assert not any(l.bias and l.bias["biased"] for k, l in lay.items() if "_Truncating" not in l.type_name
                   and l.type_name != "Sequential"), [(k, l.bias) for k, l in lay.items()]


def test_check_clean_model_has_no_lean():
    import truestar
    torch.manual_seed(1)
    m = nn.Sequential(nn.Linear(32, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 128), nn.Softmax(-1))
    rep = truestar.check(m, torch.randn(64, 32), backends=["cpu"], drill_ops=False)
    assert "TS961" not in rep.codes, rep.to_text()
    assert any(l.bias is not None for l in rep.results[0].layers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU with Triton")
def test_tf32_contraction_leans_and_is_reported_not_refused(recwarn):
    """Measured on the RTX 5070 Ti (Triton 3.7.1): TF32 operands are truncated, so a positive matmul
    leans toward zero by about 6e-4, eight times the effect floor; strict float32 does not lean."""
    pytest.importorskip("triton")
    M, N, Kd = 128, 128, 1024
    a, b = K.buffer("a", (M, Kd)), K.buffer("b", (Kd, N))
    i, j, k = K.axis("i", M), K.axis("j", N), K.axis("k", Kd)
    g = torch.Generator().manual_seed(0)
    A, Bm = torch.rand(M, Kd, generator=g).cuda(), torch.rand(Kd, N, generator=g).cuda()
    strict = K.kernel("lean_mm_ieee", [K.output("c", (i, j), K.reduce_sum(k, a[i, k] * b[k, j]))])
    K.compile(strict, "cuda")(a=A, b=Bm)
    assert not [w for w in recwarn if issubclass(w.category, verify.KernelBiasWarning)]
    tf = K.kernel("lean_mm_tf32", [K.output("c", (i, j), K.reduce_sum(k, a[i, k] * b[k, j]))])
    K.compile(tf, "cuda", allow_tf32=True)(a=A, b=Bm)          # not refused
    ws = [w for w in recwarn if issubclass(w.category, verify.KernelBiasWarning)]
    assert len(ws) == 1 and "TS966" in str(ws[0].message) and "toward zero" in str(ws[0].message)
