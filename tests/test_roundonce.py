"""Round once elementwise math (docs/ROUND_ONCE.md): correct rounding, portable bits, kernels.

The exhaustive float16 and bfloat16 comparison against mpmath (all 65,536 inputs per format and
function) runs in scripts/roundonce_measure.py small; it found 0 mismatches and pinned the SHA256
of the 65,536 output bits per format and function in PINNED below. The fast tests here recompute
those bits and check the hash, check a random subset against mpmath directly, and check the
float32 hard cases.
"""
import hashlib
import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")
mpmath = pytest.importorskip("mpmath")

sys.path.insert(0, os.path.dirname(__file__))
import roundonce_ref as ref  # noqa: E402

from truestar import kernel as K  # noqa: E402
from truestar.kernel import ir, roundonce as R  # noqa: E402

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")


def _metal():
    from truestar.kernel import metal_backend as mb
    return mb.available()


METAL = _metal()
needs_metal = pytest.mark.skipif(not METAL, reason="needs Metal (macOS with MPS)")

# sha256 of R.apply(name, x).view(int16) over all 65,536 bit patterns of the format, proven
# equal to the mpmath correctly rounded result by scripts/roundonce_measure.py small
PINNED = {
    "bfloat16/erf": "0393597ecba230755bf07c2b04c57db98ff01a446991051090555521246f48ba",
    "bfloat16/exp": "3c4c43fcea32e4bed52a347938952d81fae5472b0496385521199e722c96a360",
    "bfloat16/log": "f64b723b63f38ea2e92206a702166b5f7da8dc1b5fe9846af85ae11b0e3daff8",
    "bfloat16/rsqrt": "5b33d67ca2d5fd385a60584d8740a2bb1e715ea29b8c6af870d788f4ad0e2be0",
    "bfloat16/sigmoid": "68d3a27df0b8afccefecb5f0e49bc2df62a4f6dc201cff1b51f33807b6623bef",
    "bfloat16/tanh": "93b86a43386e016707432e45f56fda72e4312b1c5cd68ccbb588ed7c9cacf0c4",
    "float16/erf": "9bdc79d64641a2d79a6876a3d58abf82a3492032d27f119ffe75797dd0a7dba5",
    "float16/exp": "608c213c696b69ed1068ffad77c072bb7b6077f54f58ae277d4e54278020f342",
    "float16/log": "322c994520bd9cc2f82db2af6dd2a93063983c1b8f6603b19b207ad75f17f99a",
    "float16/rsqrt": "87d0a874cbf78a0e82326401468f258c0f7e28d56f2f65182ca1f9a4499187fb",
    "float16/sigmoid": "a1c114a164cf529daac76bff9657e9d4828d3617bf5eb7398732bf78730c252b",
    "float16/tanh": "fe29dbdff0de41fbffd617d5648cdcab89d76b8291b9e2e9ac08a6fc97f0089c",
}


def all16(fmt):
    return torch.arange(65536, dtype=torch.int32).to(torch.int16).view(getattr(torch, fmt))


def same(a, b):
    """Equal as IEEE values including the sign of zero; any NaN equals any NaN."""
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)


def nbits_differ(p, q):
    """Positions where the bit patterns differ (NaN payloads and signs included)."""
    w = torch.int16 if p.element_size() == 2 else torch.int32
    return int((p.view(w) != q.view(w)).sum())


# ----------------------------------------------------------------------------- constants

def test_constants_are_correctly_rounded():
    with mpmath.workprec(300):
        ln2 = mpmath.log(2)
        assert R.LN2_HI == float(mpmath.mpf(R.LN2_HI))
        assert abs(mpmath.mpf(R.LN2_HI) + mpmath.mpf(R.LN2_LO) - ln2) < mpmath.mpf(2) ** -85
        # 32 significant bits: k * LN2_HI is exact for |k| < 2^21
        m, _ = math.frexp(R.LN2_HI)
        assert (m * 2 ** 32) == int(m * 2 ** 32)
        assert R.TWO_OVER_SQRTPI == float(2 / mpmath.sqrt(mpmath.pi))
        assert R.SQRT2 == float(mpmath.sqrt(2))
    assert R.EXP_C[5] == float(mpmath.mpf(1) / 120)
    assert R.ERF_C[7] == 2 / 15


def test_erf_series_is_long_enough():
    """The tail of the positive series beyond ERF_N terms is below 2^-60 of the sum at a = 4."""
    y = mpmath.mpf(16)
    terms = [mpmath.mpf(1)]
    for n in range(1, 200):
        terms.append(terms[-1] * 2 * y / (2 * n + 1))
    assert sum(terms[R.ERF_N + 1:]) / sum(terms) < mpmath.mpf(2) ** -60


# ----------------------------------------------------------------------------- 16 bit formats

@pytest.mark.parametrize("fmt", ["float16", "bfloat16"])
@pytest.mark.parametrize("name", R.FUNCTIONS)
def test_16bit_exhaustive_bits_match_the_proven_hash(fmt, name):
    key = f"{fmt}/{name}"
    if key not in PINNED:
        pytest.skip("no pinned hash")
    y = R.apply(name, all16(fmt))
    assert hashlib.sha256(y.view(torch.int16).numpy().tobytes()).hexdigest() == PINNED[key]


@pytest.mark.parametrize("fmt", ["float16", "bfloat16"])
@pytest.mark.parametrize("name", R.FUNCTIONS)
def test_16bit_random_subset_against_mpmath(fmt, name):
    g = torch.Generator().manual_seed(7)
    x = all16(fmt)[torch.randint(0, 65536, (400,), generator=g)]
    y = R.apply(name, x)
    for xv, yv in zip(x.tolist(), y.tolist()):
        assert same(yv, ref.reference(name, xv, fmt)), (name, fmt, xv, yv)


# ----------------------------------------------------------------------------- float32

HARD = {
    "exp": [0.0, -0.0, 1.0, -1.0, 88.72283, 88.72284, -87.33654, -103.27893, -103.97208, -104.0,
            1e-8, -1e-8, 1e-40, 0.34657359, 0.5, 2.0, 10.0, -50.0],
    "log": [1.0, 0.99999994, 1.0000001, 2.0, 0.5, 1.1754944e-38, 1e-45, 1e-40, 3.4028235e38,
            0.70710677, 1.4142135, 10.0, 1e-10],
    "tanh": [0.0, -0.0, 1e-45, -1e-45, 1e-20, 0.49999997, 0.5, 0.50000006, 9.010913, -9.01092,
             20.0, 1e-3, 3.0],
    "erf": [0.0, -0.0, 1e-45, 1e-38, 1e-10, 0.5, 1.0, 2.0, 3.9192059, 3.9999998, 4.0, -2.5],
    "rsqrt": [1.0, 4.0, 2.0, 0.25, 1e-45, 1.1754944e-38, 3.4028235e38, 0.0, -0.0, 3.0],
    "sigmoid": [0.0, -0.0, 16.635532, 17.0, -87.33654, -103.27893, -104.0, 88.72283, 1e-8],
}


@pytest.mark.parametrize("name", R.FUNCTIONS)
def test_float32_hard_cases_against_mpmath(name):
    xs = []
    for v in HARD[name]:
        b = torch.tensor([v], dtype=torch.float32).view(torch.int32)
        xs += [b + d for d in (-2, -1, 0, 1, 2)]
    x = torch.cat(xs).view(torch.float32)
    x = torch.cat([x, -x, torch.tensor([math.inf, -math.inf, math.nan])])
    y = R.apply(name, x)
    for xv, yv in zip(x.tolist(), y.tolist()):
        assert same(yv, ref.reference(name, xv, "float32")), (name, xv, yv)


@pytest.mark.parametrize("name", R.FUNCTIONS)
def test_float32_random_sample_against_mpmath(name):
    g = torch.Generator().manual_seed(11)
    bits = torch.randint(-(1 << 31), 1 << 31, (300,), generator=g, dtype=torch.int64)
    x = bits.to(torch.int32).view(torch.float32)
    y = R.apply(name, x)
    for xv, yv in zip(x.tolist(), y.tolist()):
        assert same(yv, ref.reference(name, xv, "float32")), (name, xv, yv)


def test_special_values_and_signed_zeros():
    x = torch.tensor([0.0, -0.0, math.inf, -math.inf, math.nan], dtype=torch.float32)
    sgn = lambda t: [math.copysign(1.0, v) for v in t.tolist()]  # noqa: E731
    assert R.tanh(x)[:4].tolist() == [0.0, 0.0, 1.0, -1.0] and sgn(R.tanh(x)[:2]) == [1.0, -1.0]
    assert R.erf(x)[:4].tolist() == [0.0, 0.0, 1.0, -1.0] and sgn(R.erf(x)[:2]) == [1.0, -1.0]
    assert R.exp(x)[:4].tolist() == [1.0, 1.0, math.inf, 0.0]
    assert R.log(x)[:3].tolist() == [-math.inf, -math.inf, math.inf] and math.isnan(R.log(x)[3])
    assert R.rsqrt(x)[:3].tolist() == [math.inf, -math.inf, 0.0] and math.isnan(R.rsqrt(x)[3])
    assert R.sigmoid(x)[:4].tolist() == [0.5, 0.5, 1.0, 0.0]
    for name in R.FUNCTIONS:
        assert math.isnan(R.apply(name, x)[4])


def test_round_to_handles_subnormals_ties_and_overflow():
    vals = [1.0 + 2.0 ** -11, 1.0 + 3 * 2.0 ** -11, 2.0 ** -25, 3 * 2.0 ** -26, 65520.0, 65519.99,
            -2.0 ** -26, 2.0 ** -24 * 1.5, -0.0]
    y = torch.tensor(vals, dtype=torch.float64)
    got = R.round_to(y, torch.float16).tolist()
    for v, gv in zip(vals, got):
        assert same(gv, ref._round_float(v, "float16")), (v, gv)
    # float32: the arithmetic rounding agrees with the IEEE cast, subnormals included
    g = torch.Generator().manual_seed(3)
    y = torch.randn(1 << 16, generator=g, dtype=torch.float64) * torch.exp2(
        torch.randint(-160, 130, (1 << 16,), generator=g).to(torch.float64))
    assert nbits_differ(R.round_to(y, torch.float32), y.to(torch.float32)) == 0


def test_cpu_sqrt_is_correctly_rounded():
    """The CPU evaluator's sqrt (NumPy) is the IEEE one; torch's CPU float64 sqrt is not on every
    build (docs/ROUND_ONCE.md), which is why the evaluator does not use it."""
    g = torch.Generator().manual_seed(5)
    a = torch.rand(4000, generator=g, dtype=torch.float64)
    s = R._TorchOps(torch, True).sqrt(a)
    assert all(sv == math.sqrt(av) for av, sv in zip(a.tolist(), s.tolist()))


def test_dtype_refusals():
    with pytest.raises(R.RoundOnceError) as e:
        R.exp(torch.zeros(3, dtype=torch.int32))
    assert e.value.code == "TS1091"
    with pytest.raises(R.RoundOnceError) as e:
        R.exp(torch.zeros(3), dtype=torch.float64)
    assert e.value.code == "TS1091"
    with pytest.raises(R.RoundOnceError) as e:
        R.apply("cos", torch.zeros(3))
    assert e.value.code == "TS1092"


# ----------------------------------------------------------------------------- portability

@needs_cuda
@pytest.mark.parametrize("name", R.FUNCTIONS)
def test_cpu_and_cuda_give_the_same_bits(name):
    for fmt in ("float16", "bfloat16"):
        x = all16(fmt)
        assert nbits_differ(R.apply(name, x), R.apply(name, x.cuda()).cpu()) == 0
    g = torch.Generator().manual_seed(13)
    x = torch.randint(-(1 << 31), 1 << 31, (1 << 18,), generator=g, dtype=torch.int64)
    x = x.to(torch.int32).view(torch.float32)
    assert nbits_differ(R.apply(name, x), R.apply(name, x.cuda()).cpu()) == 0


# ----------------------------------------------------------------------------- kernels

def _unary_kernel(name, fmt_in, fmt_out="float32"):
    N = K.sym("N")
    i = K.axis("i", N)
    xb = K.buffer("x", (N,), fmt_in)
    return K.kernel(f"ro_{name}_{fmt_in}_{fmt_out}",
                    [K.output("y", (i,), getattr(ir, name)(xb[i]), fmt_out)])


def _plus_zero(t):
    """The kernel layer stores -0 as +0 (its broadcast adds a zero block); compare modulo that."""
    return torch.where(t == 0, torch.zeros_like(t), t)


def _numba():
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.parametrize("backend", ["cuda", "cpu", "metal"])
@pytest.mark.parametrize("name", R.FUNCTIONS)
def test_kernel_bits_equal_the_torch_evaluator(name, backend, monkeypatch):
    if backend == "cuda" and not CUDA:
        pytest.skip("needs CUDA")
    if backend == "cpu" and not _numba():
        pytest.skip("needs numba")
    if backend == "metal" and not METAL:
        pytest.skip("needs Metal")
    monkeypatch.setenv("TRUESTAR_VERIFY", "0")   # exp overflows float32 where float64 does not
    dev = {"cuda": "cuda", "cpu": "cpu", "metal": "mps"}[backend]
    g = torch.Generator().manual_seed(17)
    bits = torch.randint(-(1 << 31), 1 << 31, (1 << 16,), generator=g, dtype=torch.int64)
    for fmt, x in (("float16", all16("float16")), ("bfloat16", all16("bfloat16")),
                   ("float32", bits.to(torch.int32).view(torch.float32))):
        fn = K.compile(_unary_kernel(name, fmt), backend, math="round_once", tune=False)
        got = fn(x=x.to(dev)).cpu()
        want = R.apply(name, x, torch.float32)
        assert nbits_differ(_plus_zero(got), _plus_zero(want)) == 0, (name, fmt)


@needs_cuda
def test_triton_round_once_modules_launch_without_fusion():
    from truestar.kernel import triton_backend as tb
    fn = K.compile(_unary_kernel("erf", "float32"), "cuda", math="round_once", tune=False)
    assert fn.art.module.TS_FP_FUSION is False
    assert "_ro_erf" in fn.art.source and "enable_fp_fusion" not in fn.art.source
    env = {"s0": 1024}
    _, _, _, meta = tb.prepare(fn.art.module, fn.ck, fn.art.plan, env)
    assert meta.get("enable_fp_fusion") is False
    plain = K.compile(_unary_kernel("erf", "float32"), "cuda", tune=False)
    assert "enable_fp_fusion" not in tb.prepare(plain.art.module, plain.ck, plain.art.plan, env)[3]
    assert plain.art.key != fn.art.key


@needs_cuda
def test_round_once_kernels_are_still_verified_against_float64():
    M, N = K.sym("M"), K.sym("N")
    x = K.buffer("x", (M, N), "float32")
    i, j = K.axis("i", M), K.axis("j", N)
    e = K.tanh(x[i, j]) * K.sigmoid(x[i, j]) + ir.erf(x[i, j] * 0.5) + K.log(K.exp(x[i, j]) + 1.0)
    k = K.kernel("ro_mix", [K.output("y", (i, j), e + K.rsqrt(K.abs_(x[i, j]) + 1.0))])
    xt = torch.randn(64, 300, device="cuda")
    fn = K.compile(k, "cuda", math="round_once", tune=False)
    fn(x=xt)                                       # verified on the first call
    sm = K.kernel("ro_softmax", [K.output("y", (i, j), K.softmax_weight(x[i, j] * 0.5, over=j))])
    K.compile(sm, "cuda", math="round_once", tune=False)(x=xt)
    # a broken round once exp is caught by the same check
    bad = K.compile(k, "cuda", math="round_once", tune=False, _break={"exp": lambda a: f"({a}) * 1.01"})
    with pytest.raises(K.KernelMismatchError):
        bad(x=xt)


# ----------------------------------------------------------------------------- refusals

def test_unknown_mode_is_refused():
    with pytest.raises(R.RoundOnceError) as e:
        K.compile(_unary_kernel("exp", "float32"), "cpu", math="fast")
    assert e.value.code == "TS1090"


def test_metal_source_has_the_soft_float64_helpers():
    """Apple GPUs have no float64: the Metal helpers run the same sequence on ulong bits."""
    from truestar.kernel import metal_backend as mb
    R.check_kernel(_unary_kernel("exp", "float32"), "metal")          # accepted (TS1089 before)
    src, _ = mb.generate(ir.canonicalize(_unary_kernel("erf", "float32"))[0], {"math": "round_once"})
    assert "_ro_erf(" in src and "ts_dmul(" in src and "double" not in src
    assert src.count("{") == src.count("}")
    plain, _ = mb.generate(ir.canonicalize(_unary_kernel("erf", "float32"))[0], {})
    assert "_ro_erf" not in plain and "ts_erf(" in plain


def _f64_bits(a):
    import numpy as np
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def _soft_f64_operands(n, seed):
    """Edge values (zeros, subnormals, the largest finite, infinities, NaN, ties) and random bit
    patterns and random normal doubles, as uint64 bits."""
    import numpy as np
    rng = np.random.default_rng(seed)
    edge = np.array([0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308, 2.225073858507201e-308,
                     1.7976931348623157e308, -1.7976931348623157e308, np.inf, -np.inf, np.nan, 1.0, -1.0,
                     0.5, 1.5, 2.0 ** -1074 * 3, 1.0 + 2.0 ** -52, 1.0 - 2.0 ** -53, 3.0, 0.1, 1e300, 1e-300])
    rb = rng.integers(0, 1 << 63, size=n, dtype=np.uint64) | (rng.integers(0, 2, size=n, dtype=np.uint64) << 63)
    nb = _f64_bits(rng.normal(size=n) * np.exp2(rng.integers(-60, 60, size=n)))
    a = np.concatenate([np.repeat(_f64_bits(edge), len(edge)), rb, nb])
    b = np.concatenate([np.tile(_f64_bits(edge), len(edge)), rng.permutation(rb), rng.permutation(nb)])
    return a, b


@needs_metal
def test_soft_float64_matches_the_cpu_bit_for_bit():
    import numpy as np
    from truestar.kernel.softf64_metal import SOURCE
    src = "#include <metal_stdlib>\nusing namespace metal;\n" + SOURCE + """
kernel void ops(device const ulong* a [[buffer(0)]], device const ulong* b [[buffer(1)]],
                device ulong* o [[buffer(2)]], device int* c [[buffer(3)]], device float* f [[buffer(4)]],
                uint i [[thread_position_in_grid]]) {
    ulong x = a[i], y = b[i];
    o[8 * i + 0] = ts_dadd(x, y);
    o[8 * i + 1] = ts_dsub(x, y);
    o[8 * i + 2] = ts_dmul(x, y);
    o[8 * i + 3] = ts_ddiv(x, y);
    o[8 * i + 4] = ts_dsqrt(x);
    o[8 * i + 5] = ts_dfloor(x);
    o[8 * i + 6] = ts_dabs(x);
    o[8 * i + 7] = ts_dneg(x);
    c[3 * i + 0] = ts_dlt(x, y) ? 1 : 0;
    c[3 * i + 1] = ts_dle(x, y) ? 1 : 0;
    c[3 * i + 2] = ts_deq(x, y) ? 1 : 0;
    f[i] = ts_dtof32(x);
}
"""
    lib = torch.mps.compile_shader(src)
    a, b = _soft_f64_operands(200_000, 3)
    n = a.size
    ta = torch.from_numpy(a.view(np.int64)).to("mps")
    tb = torch.from_numpy(b.view(np.int64)).to("mps")
    o = torch.zeros(8 * n, dtype=torch.int64, device="mps")
    c = torch.zeros(3 * n, dtype=torch.int32, device="mps")
    f = torch.zeros(n, dtype=torch.float32, device="mps")
    lib.ops(ta, tb, o, c, f, threads=[n, 1, 1], group_size=[256, 1, 1])
    o = o.cpu().numpy().view(np.uint64).reshape(n, 8)
    c = c.cpu().numpy().reshape(n, 3)
    f = f.cpu().numpy()
    x, y = a.view(np.float64), b.view(np.float64)
    with np.errstate(all="ignore"):
        want = [x + y, x - y, x * y, x / y, np.sqrt(x), np.floor(x), np.abs(x), -x]
        cmpw = [x < y, x <= y, x == y]
        fw = x.astype(np.float32)
    for k, w in enumerate(want):
        got = o[:, k].view(np.float64)
        bad = ~((got.view(np.uint64) == _f64_bits(w)) | (np.isnan(got) & np.isnan(w)))
        assert not bad.any(), (k, a[bad][:3], b[bad][:3], got[bad][:3], w[bad][:3])
    for k, w in enumerate(cmpw):
        assert np.array_equal(c[:, k].astype(bool), w), k
    fb = ~((f.view(np.uint32) == fw.view(np.uint32)) | (np.isnan(f) & np.isnan(fw)))
    assert not fb.any(), (x[fb][:3], f[fb][:3], fw[fb][:3])
    assert (f[np.isnan(f)].view(np.uint32) == 0x7FC00000).all()


@needs_metal
def test_metal_round_once_float32_bits_equal_the_cpu_on_a_million_inputs(monkeypatch):
    """Every function on 2^20 random float32 bit patterns plus the hard cases: the Mac's GPU gives
    the CPU evaluator's bits (the correctly rounded ones), and sqrt is correctly rounded too."""
    monkeypatch.setenv("TRUESTAR_VERIFY", "0")
    g = torch.Generator().manual_seed(29)
    x = torch.randint(-(1 << 31), 1 << 31, (1 << 20,), generator=g, dtype=torch.int64).to(torch.int32).view(torch.float32)
    for name in R.FUNCTIONS:
        xs = torch.cat([x, torch.tensor([xin for xin, _ in R.EXCEPTIONS.get(name, [])] or [1.0], dtype=torch.float32)])
        got = K.compile(_unary_kernel(name, "float32"), "metal", math="round_once", tune=False)(x=xs.to("mps")).cpu()
        assert nbits_differ(_plus_zero(got), _plus_zero(R.apply(name, xs, torch.float32))) == 0, name
    pos = x.abs()
    got = K.compile(_unary_kernel("sqrt", "float32"), "metal", math="round_once", tune=False)(x=pos.to("mps")).cpu()
    want = torch.sqrt(pos)
    want = torch.where(torch.isnan(want), torch.tensor(float("nan")), want)    # round once NaN is canonical
    assert nbits_differ(_plus_zero(got), _plus_zero(want)) == 0
    assert (got[torch.isnan(got)].view(torch.int32) == 0x7FC00000).all()


@needs_metal
def test_metal_round_once_kernels_are_verified_and_a_break_is_caught():
    M, N = K.sym("M"), K.sym("N")
    x = K.buffer("x", (M, N), "float32")
    i, j = K.axis("i", M), K.axis("j", N)
    e = K.tanh(x[i, j]) * K.sigmoid(x[i, j]) + ir.erf(x[i, j] * 0.5) + K.log(K.exp(x[i, j]) + 1.0)
    k = K.kernel("ro_mix", [K.output("y", (i, j), e + K.rsqrt(K.abs_(x[i, j]) + 1.0))])
    xt = torch.randn(64, 300).to("mps")
    K.compile(k, "metal", math="round_once")(x=xt)          # verified on the first call
    sm = K.kernel("ro_softmax", [K.output("y", (i, j), K.softmax_weight(x[i, j] * 0.5, over=j))])
    K.compile(sm, "metal", math="round_once")(x=xt)
    bad = K.compile(k, "metal", math="round_once", _break={"exp": lambda a: f"(precise::exp({a}) * 1.01f)"})
    with pytest.raises(K.KernelMismatchError):
        bad(x=xt)


def test_float64_buffers_and_ops_without_a_form_are_refused():
    with pytest.raises(R.RoundOnceError) as e:
        K.compile(_unary_kernel("exp", "float64"), "cpu", math="round_once")
    assert e.value.code == "TS1091"
    N = K.sym("N")
    i = K.axis("i", N)
    xb = K.buffer("x", (N,), "float32")
    k = K.kernel("ro_cos", [K.output("y", (i,), ir.Unary("cos", xb[i]))])
    with pytest.raises(R.RoundOnceError) as e:
        K.compile(k, "cpu", math="round_once")
    assert e.value.code == "TS1092"


@needs_cuda
def test_attention_loop_shape_is_refused():
    g = K.Graph()
    S, D = K.sym("S"), 16
    q, kk, v = (g.input(n, (2, S, D), "float32") for n in "qkv")
    with pytest.raises(R.RoundOnceError) as e:
        g.compile({"o": g.softmax((q @ kk.T) * 0.25) @ v}, "cuda", math="round_once", tune=False)
    assert e.value.code == "TS1093"
