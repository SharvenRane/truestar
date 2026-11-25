"""Exact tensor core blocks (kernel/tcexact.py): the NumPy twin, the proof's bounds, and the fp16
engine's exactness on every tensor core generation model. CPU only; the GPU bits are checked by
scripts/tcexact_bench.py and the cross chip fingerprint."""
import numpy as np
import pytest
import torch

from truestar.devsim.tensorcore import GENERATIONS, unit_matmul
from truestar.kernel import tcexact as X


def _vals(shape, dtype, seed, spread=0):
    rs = np.random.RandomState(seed)
    x = rs.standard_normal(shape).astype(np.float32)
    if spread:
        x *= (2.0 ** rs.randint(-spread, spread + 1, size=shape)).astype(np.float32)
    t = torch.from_numpy(x).to({"float16": torch.float16, "bfloat16": torch.bfloat16,
                                "tf32": torch.float32}[dtype])
    return t.float().numpy()


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "tf32"])
@pytest.mark.parametrize("kb", [16, 64, 128])
def test_block_accuracy(dtype, kb):
    a = _vals((32, kb), dtype, 1)
    b = _vals((kb, 24), dtype, 2)
    tf = dtype == "tf32"
    if tf:
        a, b = X.round_tf32(a), X.round_tf32(b)
    r = X.emulate_block(a, b, tf).astype(np.float64)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    den = np.abs(a).astype(np.float64) @ np.abs(b).astype(np.float64)
    # dropped digits: at most 2**(e - 21) per operand value, e the block exponent (< 2x the block
    # maximum), plus the pairs of weight 3 and 4 and three float32 roundings
    bound = (np.abs(a).max(1)[:, None] * np.abs(b).sum(0)[None, :]
             + np.abs(a).sum(1)[:, None] * np.abs(b).max(0)[None, :]) * 2.0 ** -18 + 4 * 2.0 ** -24 * den
    assert np.all(np.abs(r - ref) <= bound)


def test_exact_when_values_fit_the_digits():
    # integers below 2**7 in float16: one digit each, the block result is the exact dot product
    rs = np.random.RandomState(3)
    a = rs.randint(-64, 65, size=(16, 64)).astype(np.float32)
    b = rs.randint(-64, 65, size=(64, 16)).astype(np.float32)
    assert np.array_equal(X.emulate_block(a, b), (a.astype(np.int64) @ b.astype(np.int64)).astype(np.float32))


def test_matmul_order_and_padding():
    a = _vals((8, 200), "float16", 4)
    b = _vals((200, 8), "float16", 5)
    got = X.emulate_matmul(a, b, 64)
    acc = np.zeros((8, 8), np.float32)
    for k0 in range(0, 200, 64):
        aa = np.zeros((8, 64), np.float32)
        bb = np.zeros((64, 8), np.float32)
        w = min(64, 200 - k0)
        aa[:, :w], bb[:w] = a[:, k0:k0 + w], b[k0:k0 + w]
        acc = acc + X.emulate_block(aa, bb)
    assert np.array_equal(got.view(np.int32), acc.view(np.int32))


def test_tf32_rounding_nearest_even():
    x = np.array([1 + 2 ** -11, 1 + 3 * 2 ** -11, 1 + 2 ** -11 + 2 ** -20, -(1 + 2 ** -11), 3.4e38,
                  np.inf, 0.0], dtype=np.float32)
    want = np.array([1.0, 1 + 4 * 2 ** -11, 1 + 2 ** -10, -1.0, np.inf, np.inf, 0.0], dtype=np.float32)
    got = X.round_tf32(x)
    assert np.array_equal(got[:4], want[:4])
    assert got[5] == np.inf and got[6] == 0.0


def test_zero_rows_and_wide_exponents():
    a = _vals((16, 64), "bfloat16", 6, spread=60)
    b = _vals((64, 16), "bfloat16", 7, spread=60)
    a[3] = 0.0
    b[:, 5] = 0.0
    r = X.emulate_block(a, b)
    assert np.all(np.isfinite(r))
    assert np.all(r[3] == 0) and np.all(r[:, 5] == 0)


def test_engine_choice_and_refusal():
    assert X.engine_for(16) == "fp16"
    assert X.engine_for(32) == "int8" and X.engine_for(128) == "int8"
    for kb in (8, 256):
        with pytest.raises(X.PortableTensorCoreError) as e:
            X.engine_for(kb)
        assert e.value.code == "TS1332"


def _fp16_digit_planes(a, b):
    """The digits of one block, as the kernel cuts them."""
    return X.block_digits(a, 1)[0], X.block_digits(b, 0)[0]


@pytest.mark.parametrize("gen", ["a100", "ada", "h100", "b200", "sm120", "v100"])
def test_fp16_engine_sum_exact_on_every_generation_model(gen):
    """The proof, executed: the chained fp16 digit dot products of the fp16 engine, run through the
    bit accurate tensor core model of each generation (devsim/tensorcore.py), equal the exact integer
    sums, including the worst case of every digit at its largest magnitude."""
    unit = GENERATIONS[gen]["fp16"]
    rs = np.random.RandomState(8)
    cases = []
    for kb in (16, 64, 128):
        a = _vals((16, kb), "float16", 9 + kb)
        b = _vals((kb, 16), "float16", 10 + kb)
        cases.append(_fp16_digit_planes(a, b))
        worst = [np.full((16, kb), 127.0), np.full((16, kb), -64.0), np.full((16, kb), 64.0)]
        cases.append((worst, [np.full((kb, 16), 127.0), np.full((kb, 16), 64.0), np.full((kb, 16), -64.0)]))
        rnd = [rs.randint(-64, 65, size=(16, kb)).astype(np.float64) for _ in range(3)]
        cases.append((rnd, [rs.randint(-64, 65, size=(kb, 16)).astype(np.float64) for _ in range(3)]))
    for da, db in cases:
        da = [np.asarray(d, np.float64) for d in da]
        db = [np.asarray(d, np.float64) for d in db]
        for pairs in (((0, 0),), ((0, 1), (1, 0)), ((0, 2), (1, 1), (2, 0))):
            c = np.full((16, 16), 12582912.0)   # the kernel starts every chain at 1.5 * 2**23
            exact = 0
            for p, q in pairs:
                c = unit_matmul(da[p], db[q], c, unit, "fp16")
                exact = exact + da[p].astype(np.int64) @ db[q].astype(np.int64)
            assert np.array_equal(c - 12582912.0, exact.astype(np.float64)), (gen, pairs)


def test_narrow_window_would_break_it():
    """A unit with a 10 bit window loses bits on the same kind of digits: the window constant
    matters."""
    from dataclasses import replace
    unit = replace(GENERATIONS["ada"]["fp16"], frac_bits=10)
    a = np.full((16, 64), 64.0)
    a[:, 1::2] = 1.0
    b = np.full((64, 16), 63.0)
    c = unit_matmul(a, b, None, unit, "fp16")
    assert not np.array_equal(c, (a.astype(np.int64) @ b.astype(np.int64)).astype(np.float64))


def test_bounds():
    X.check_bounds()
    assert X.TC_FRAC_BITS_MIN == 24 and X.SUM_BITS - 1 < X.TC_FRAC_BITS_MIN


def test_digits_balanced_and_exact():
    rs = np.random.RandomState(12)
    t = (rs.uniform(-127, 127, size=10000)).astype(np.float32)
    t[:4] = [127.0, -127.0, 0.5, -0.5]
    d = X.digits(t)
    assert np.abs(d[0]).max() <= 127 and all(np.abs(x).max() <= 64 for x in d[1:])
    back = d[0] + d[1] / 128.0 + d[2] / 16384.0
    assert np.all(np.abs(back - t.astype(np.float64)) <= 0.5 / 16384.0)


def _mm_kernel(dtype):
    from truestar import kernel as K
    from truestar.kernel import Graph, ir
    g = Graph()
    a = g.input("a", (K.sym("M"), K.sym("K")), dtype)
    b = g.input("b", (K.sym("K"), K.sym("N")), dtype)
    (_, kern), = Graph.fuse(g, {"c": a @ b})
    return ir.canonicalize(kern)[0]


@pytest.mark.parametrize("dtype,tf32,exact", [("float16", False, True), ("bfloat16", False, True),
                                               ("float32", True, True), ("float32", False, False)])
def test_contraction_source_and_pinned_block(monkeypatch, dtype, tf32, exact):
    from truestar.kernel import triton_backend as tb
    monkeypatch.setattr(tb, "device_limits", lambda: (tb.REF_SMS, tb.REF_SMEM, tb.REF_MEMORY))
    opts = {"portable_tc": True, "deterministic": True}
    if tf32:
        opts["allow_tf32"] = True
    src, p = tb.generate(_mm_kernel(dtype), opts)
    assert ("_tsx_xdot(" in src.split("def kern(")[1]) == exact
    assert p.portable
    env = {"s0": 4096, "s1": 4096, "s2": 4096}
    cands = tb.candidates(p, env)
    assert all(c["BK"] == X.PORTABLE_BK and c["SPLIT_K"] == 1 for c in cands)
    assert tb.default_config(p, env)["BK"] == X.PORTABLE_BK
    bad = dict(cands[0], BK=32)
    assert tb.check_config(p, env, bad) is not None
    # plain mode keeps the tuned K block and tl.dot
    src2, p2 = tb.generate(_mm_kernel(dtype), {"allow_tf32": True} if tf32 else {})
    assert "_tsx_xdot(" not in src2.split("def kern(")[1]


def test_compile_refusals():
    from truestar import kernel as K
    k = _mm_kernel("float16")
    with pytest.raises(X.PortableTensorCoreError) as e:
        K.compile(k, "cpu", portable_tensor_cores=True)
    assert e.value.code == "TS1330"
    from truestar.codes import CODES
    assert all(c in CODES for c in X.CODES)
