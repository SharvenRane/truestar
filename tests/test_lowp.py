"""Low precision formats: MXFP8 tables and scale rules, the FP8 variant trap (OCP versus FNUZ), and
keyed stochastic rounding (docs/QUANT.md)."""

import json
import math
import struct

import numpy as np
import pytest
import torch

from truestar import codes, rng
from truestar.quant import formats as F
from truestar.quant import fp8variant as V
from truestar.quant import mx
from truestar.quant import sround as S

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
ml_dtypes = pytest.importorskip("ml_dtypes")


def _same(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return bool(((a == b) | (np.isnan(a) & np.isnan(b))).all())


# ----------------------------------------------------------------------------- tables


@pytest.mark.parametrize("codec,ref", [
    (F.E2M1, "float4_e2m1fn"),
    (F.E4M3, "float8_e4m3fn"), (F.E5M2, "float8_e5m2"), (F.E4M3FNUZ, "float8_e4m3fnuz"),
    (F.E5M2FNUZ, "float8_e5m2fnuz"), (F.E4M3B11FNUZ, "float8_e4m3b11fnuz"),
    (F.E4M3IEEE, "float8_e4m3")])
def test_every_code_matches_ml_dtypes(codec, ref):
    n = 1 << codec.bits
    want = np.arange(n, dtype=np.uint8).view(getattr(ml_dtypes, ref)).astype(np.float64)
    assert _same(codec.decode(torch.arange(n)).numpy(), want)


@pytest.mark.parametrize("codec,dt", [(F.E4M3FNUZ, torch.float8_e4m3fnuz),
                                      (F.E5M2FNUZ, torch.float8_e5m2fnuz)])
def test_fnuz_tables_match_torch(codec, dt):
    c = torch.arange(256, dtype=torch.uint8)
    assert _same(c.view(dt).double().numpy(), codec.decode(c.long()).numpy())


def test_ocp_mx_element_facts():
    # OCP MX v1.1 section 5.3: max normal and emax of every element format
    assert F.E2M1.max_finite == 6.0
    assert [mx.emax(c) for c in (F.E4M3, F.E5M2, F.E2M1)] == [8, 15, 2]
    assert F.E4M3FNUZ.max_finite == 240.0 and F.E5M2FNUZ.max_finite == 57344.0
    assert math.isnan(F.E4M3FNUZ.values[0x80]) and F.E4M3.values[0x80] == 0.0
    # the trap: a normal E4M3 code is worth half under FNUZ
    for c in range(0x08, 0x7F):
        assert F.E4M3FNUZ.values[c] * 2 == F.E4M3.values[c]


@pytest.mark.parametrize("codec,ref", [(F.E4M3FNUZ, "float8_e4m3fnuz")])
def test_encode_is_round_to_nearest_even(codec, ref):
    """Every midpoint between neighbours (the ties) and random points, against ml_dtypes' cast."""
    mags = sorted({abs(v) for v in codec.values if math.isfinite(v)})
    mids = [(a + b) / 2 for a, b in zip(mags, mags[1:])]
    g = torch.Generator().manual_seed(0)
    x = torch.cat([torch.tensor(mids + mags, dtype=torch.float64),
                   torch.rand(4000, generator=g, dtype=torch.float64) * codec.max_finite])
    x = torch.cat([x, -x])
    got = codec.decode(codec.encode(x)).numpy()
    want = x.numpy().astype(np.float32).astype(getattr(ml_dtypes, ref)).astype(np.float64)
    assert _same(np.where(got == 0, 0.0, got), np.where(want == 0, 0.0, want))


# ----------------------------------------------------------------------------- MX scale rules


def test_floor_rule_is_the_existing_mxfp4_scale():
    g = torch.Generator().manual_seed(1)
    am = torch.rand(5000, generator=g, dtype=torch.float64) * 2.0 ** torch.randint(-30, 30, (5000,), generator=g)
    am = torch.cat([am, torch.tensor([0.0, 1.0, 6.0, 7.999, 8.0, 2.0 ** -126])])
    assert torch.equal(mx.scale_exponent(am, F.E2M1, "floor") + 127, F.mx_scale_code(am))


@pytest.mark.parametrize("fmt", list(mx.MX_FORMATS))
def test_scale_rules_and_clipping(fmt):
    g = torch.Generator().manual_seed(2)
    x = torch.randn(256, 512, generator=g) * torch.exp(torch.randn(256, 1, generator=g) * 3)
    codec = mx.element_codec(fmt)
    res = {r: mx.quantize(x, fmt, r) for r in mx.SCALE_RULES}
    assert res["floor_adj"].clipped(x) == 0 and res["ceil"].clipped(x) == 0
    assert res["floor"].clipped(x) > 0                         # the v1.0 rule clips some maxima
    # ceil: the smallest power of two X with amax / X <= max (exact check)
    am = x.double().reshape(256, -1, 32).abs().amax(-1)
    X = F.e8m0_decode(res["ceil"].scale_codes)
    assert bool((am <= X * codec.max_finite).all()) and bool((am > X / 2 * codec.max_finite).all())
    for r in mx.SCALE_RULES:
        d = res[r].dequantize()
        # exact decode: element value times a power of two, and the codes are RTN of x / X
        assert not mx.check_quantizer(x, res[r].codes, res[r].scale_codes, fmt, r)
        assert d.dtype == torch.float64
    assert mx.matching_rules(x, res["ceil"].scale_codes, fmt)[-1] == "ceil"


def test_mx_quantizer_check_names_the_rule_and_the_element():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(8, 256, generator=g)
    q = mx.quantize(x, "mxfp8_e4m3", "ceil")
    out = mx.check_quantizer(x, q.codes, q.scale_codes, "mxfp8_e4m3", "floor")
    assert out and out[0].code == "TS1008" and "follow the" in out[0].message and "ceil" in out[0].message
    q = mx.quantize(x, "mxfp8_e5m2")
    c = q.codes.clone()
    c[3, 17] ^= 1
    out = mx.check_quantizer(x, c, q.scale_codes, "mxfp8_e5m2")
    assert [i.code for i in out] == ["TS1009"] and "flat index 785" in out[0].message


def test_mx_refuses_bad_input():
    with pytest.raises(F.FormatError, match="multiple of 32"):
        mx.quantize(torch.randn(4, 48), "mxfp8")
    with pytest.raises(F.FormatError, match="unknown MX format"):      # MXFP6 is not shipped
        mx.element_codec("mxfp6_e2m3")
    with pytest.raises(F.FormatError, match="NaN or Inf"):
        mx.quantize(torch.tensor([math.inf] + [0.0] * 31), "mxfp4")


@cuda
@pytest.mark.parametrize("fa,fb", [("mxfp8_e4m3", "mxfp8_e4m3"), ("mxfp8_e5m2", "mxfp8_e5m2"),
                                   ("mxfp8_e4m3", "mxfp8_e5m2")])
def test_mx_matmul_passes_verdict_a(fa, fb):
    from truestar.quant import mx_kernels as K
    g = torch.Generator().manual_seed(4)
    x = torch.randn(70, 384, generator=g)
    w = torch.randn(96, 384, generator=g) * 0.05
    qa, qb = mx.quantize(x, fa), mx.quantize(w, fb)
    qa = mx.MXTensor(qa.fmt, qa.codes.cuda(), qa.scale_codes.cuda())
    qb = mx.MXTensor(qb.fmt, qb.codes.cuda(), qb.scale_codes.cuda())
    y = K.matmul(qa, qb)
    assert K.verdict(qa, qb, y) <= 1.0
    bad = mx.MXTensor(qb.fmt, qb.codes, torch.roll(qb.scale_codes, 1, dims=1))   # wrong scales
    assert K.verdict(qa, qb, K.matmul(qa, bad)) > 1.0


# ----------------------------------------------------------------------------- FP8 variants


def test_variant_names():
    assert V.variant_of_name("torch.float8_e4m3fnuz") == "e4m3fnuz"
    assert V.variant_of_name("FLOAT8E4M3FN") == "e4m3fn"
    assert V.variant_of_name("f8E5M2FNUZ") == "e5m2fnuz"
    assert V.variant_of_name("F8_E4M3") == "e4m3fn"                 # safetensors
    assert V.variant_of_name("fp8e4b8") == "e4m3fnuz"               # Triton
    assert V.variant_of_name("__hip_fp8_e4m3_fnuz") == "e4m3fnuz"
    assert V.variant_of_name("float8_e4m3") == "e4m3ieee"           # ml_dtypes: not OCP
    for amb in ("e4m3", "fp8", "fp8_e4m3", "e5m2"):
        assert V.variant_of_name(amb) is None and V.is_ambiguous_name(amb), amb
    assert V.target_family("gfx942") == V.FNUZ and V.target_family("MI300X") == V.FNUZ
    assert V.target_family("gfx950") == V.OCP and V.target_family("sm_120") == V.OCP
    assert V.target_family("sm_80") is None and V.target_family("gfx1100") is None


def _scaled_fp8(dt, n=4096, seed=5):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g)
    mxv = torch.finfo(dt).max
    x = x / x.abs().max() * mxv                                    # amax maps to the max code
    return x.to(dt)


def test_bits_tell_the_variants_apart():
    ocp = _scaled_fp8(torch.float8_e4m3fn)
    fnuz = _scaled_fp8(torch.float8_e4m3fnuz)
    tiny = torch.tensor([-1e-9, 1.0]).to(torch.float8_e4m3fn)         # a negative zero
    assert V.bit_evidence(fnuz, "e4m3").guess == V.FNUZ
    assert V.bit_evidence(torch.cat([ocp.view(torch.uint8), tiny.view(torch.uint8)]), "e4m3").guess == V.OCP
    assert V.bit_evidence(_scaled_fp8(torch.float8_e5m2fnuz), "e5m2").guess == V.FNUZ


def test_mismatch_conflict_and_inference_findings():
    fnuz = _scaled_fp8(torch.float8_e4m3fnuz)
    ocp = _scaled_fp8(torch.float8_e4m3fn)
    r = V.check_tensor("float8_e4m3fn", ocp, "gfx942", "w")
    assert [i.code for i in r.issues] == ["TS1000"] and "half its OCP value" in r.issues[0].message
    assert not V.check_tensor("float8_e4m3fnuz", fnuz, "gfx942").issues
    assert not V.check_tensor("float8_e4m3fn", ocp, "gfx950").issues
    # FNUZ bits saved under an OCP name: the bits disagree with the name
    r = V.check_tensor("F8_E4M3", fnuz.view(torch.uint8), "gfx950", "w")
    assert "TS1003" in [i.code for i in r.issues]
    # no usable name: inferred, then compared
    r = V.check_tensor("e4m3", fnuz.view(torch.uint8), "sm_90")
    assert [i.code for i in r.issues] == ["TS1001", "TS1002", "TS1000"]
    # "fp8" does not even say E4M3 or E5M2, so the bits cannot be read
    assert [i.code for i in V.check_tensor("fp8", fnuz.view(torch.uint8)).issues] == ["TS1001", "TS1004"]
    r = V.check_tensor("e4m3", torch.tensor([0, 1, 2], dtype=torch.uint8))
    assert [i.code for i in r.issues] == ["TS1001", "TS1004"]
    assert [i.code for i in V.check_tensor("float8_e5m2", None, "gfx1100").issues] == ["TS1005"]


def test_safetensors_and_onnx_files(tmp_path):
    fnuz = _scaled_fp8(torch.float8_e4m3fnuz, 256)
    # a safetensors file written by hand (the format is an 8 byte header length, JSON, data)
    data = fnuz.view(torch.uint8).numpy().tobytes()
    header = json.dumps({"w": {"dtype": "F8_E4M3FNUZ", "shape": [256], "data_offsets": [0, 256]},
                         "b": {"dtype": "F8_E4M3", "shape": [256], "data_offsets": [256, 512]}}).encode()
    p = tmp_path / "m.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + data + data)
    reps = V.check_safetensors(str(p), "gfx950")
    got = {r.tensor: [i.code for i in r.issues] for r in reps}
    # b: named OCP (as the target) but holding FNUZ bits: the conflict is the finding
    assert got == {"w": ["TS1000"], "b": ["TS1003"]}
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    t = helper.make_tensor("w8", TensorProto.FLOAT8E4M3FNUZ, [256], data, raw=True)
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])], [t])
    q = tmp_path / "m.onnx"
    onnx.save(helper.make_model(graph), str(q))
    reps = V.check_onnx(str(q), "sm_120")
    assert [i.code for i in V.issues(reps)] == ["TS1000"]


def test_mi350_tf32_profile_is_bf16():
    from truestar.devsim import get_profile, simulate
    p = get_profile("sim:mi350-tf32")
    assert p.floor_dtype == torch.bfloat16
    g = torch.Generator().manual_seed(6)
    a, b = torch.randn(16, 64, generator=g), torch.randn(64, 8, generator=g)
    with simulate(p):
        y = a @ b
    want = (a.bfloat16().double() @ b.bfloat16().double()).float()
    assert torch.equal(y, want)


# ----------------------------------------------------------------------------- keyed SR


TGTS = ["e2m1", "e4m3", "e5m2", "bf16", "fp16"]


def _sr_input(n=3000, seed=9):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g) * torch.exp(torch.randn(n, generator=g) * 2)
    x[:8] = torch.tensor([0.0, -0.0, 1e-42, -1e-30, 500.0, -7.0, 3e38, 2.0 ** -149])
    return x


@pytest.mark.parametrize("t", TGTS)
def test_sr_backends_bitwise_equal_cpu(t):
    x = _sr_input()
    a = S.round_python(x, t, 11, "g", 2)
    assert torch.equal(a, S.round_torch(x, t, 11, "g", 2))
    e = torch.randint(-6, 6, x.shape, generator=torch.Generator().manual_seed(1))
    assert torch.equal(S.round_python(x, t, 11, "g", 2, e), S.round_torch(x, t, 11, "g", 2, e))


@cuda
@pytest.mark.parametrize("t", TGTS)
def test_sr_bitwise_equal_on_gpu(t):
    x = _sr_input(20000, 10)
    ref = S.round_torch(x, t, 11, "g", 2)
    assert torch.equal(S.round_torch(x.cuda(), t, 11, "g", 2).cpu(), ref)
    assert torch.equal(S.round_triton(x.cuda(), t, 11, "g", 2).cpu(), ref)
    e = torch.randint(-6, 6, x.shape, generator=torch.Generator().manual_seed(2))
    assert torch.equal(S.round_triton(x.cuda(), t, 11, "g", 2, e.cuda()).cpu(),
                       S.round_torch(x, t, 11, "g", 2, e))
    assert not S.verify_codes(S.round_triton(x.cuda(), t, 11, "g", 2), x, t, 11, "g", 2)


def test_sr_random_bits_are_truestar_rng_bits():
    K = S.stream(11, "g", 2)
    bits = [rng.mix(K ^ i) >> 32 for i in range(64)]
    u = rng.generate("uniform", (64,), torch.float64, 11, "g", streams=(2,), backend="torch")
    assert [int(v) >> 21 for v in (u * 2.0 ** 53).long().tolist()] == bits


def test_sr_is_keyed_and_deterministic():
    x = torch.rand(4096) * 3
    a = S.round_codes(x, "e2m1", 1, "s", 0)
    assert torch.equal(a, S.round_codes(x, "e2m1", 1, "s", 0))
    assert not torch.equal(a, S.round_codes(x, "e2m1", 1, "s", 1))
    assert not torch.equal(a, S.round_codes(x, "e2m1", 2, "s", 0))
    # exact values never move; values in range land on a neighbour
    ex = torch.tensor([0.5, 1.0, 1.5, 3.0, -6.0, 0.0])
    assert torch.equal(F.E2M1.decode(S.round_codes(ex, "e2m1", 1)), ex.double())
    v = S.round_values(x, "e2m1", 1)
    lo = F.E2M1.decode(F.E2M1.encode(x.double()))
    assert bool(((v - x.double()).abs() <= 2.0).all())


@pytest.mark.parametrize("t", TGTS)
def test_sr_is_unbiased(t):
    g = torch.Generator().manual_seed(12)
    hi = {"e2m1": 5.9, "e4m3": 440.0, "e5m2": 5e4}.get(t, 1e4)
    x = (torch.rand(2000, generator=g) * 2 - 1) * hi
    bt = S.bias_test(x, t, draws=200)
    assert bt.ok, bt
    assert abs(bt.var_ratio - 1) < 0.1
    assert not S.bias_issue(bt, t)


def test_bias_test_catches_a_biased_rounder(monkeypatch):
    x = (torch.rand(2000, generator=torch.Generator().manual_seed(13)) * 2 - 1) * 5.0
    real = S.round_values

    def skewed(x, t, key, site="sround", step=0, scale_exp=None, backend="auto"):
        v = real(x, t, key, site, step, scale_exp, backend)
        return torch.where(torch.rand(v.shape, generator=torch.Generator().manual_seed(step)) < 0.05,
                           F.E2M1.decode(F.E2M1.encode(v + 0.5 * torch.sign(v))), v)

    monkeypatch.setattr(S, "round_values", skewed)
    bt = S.bias_test(x, "e2m1", draws=200)
    assert not bt.ok
    assert [i.code for i in S.bias_issue(bt, "e2m1")] == ["TS1007"]


def test_verify_codes_names_a_flipped_code():
    x = _sr_input()
    c = S.round_torch(x, "e4m3", 3)
    c[100] ^= 1
    out = S.verify_codes(c, x, "e4m3", 3)
    assert [i.code for i in out] == ["TS1006"] and "flat index 100" in out[0].message


def test_sr_accumulation_does_not_stall():
    """Adding 1e-3 to a bfloat16 accumulator at 1.0 ten thousand times: round to nearest stalls
    (1e-3 is under half a bfloat16 ulp at 1), keyed SR tracks the float64 sum 11.0."""
    acc_rn = torch.ones(256, dtype=torch.bfloat16)
    acc_sr = torch.ones(256, dtype=torch.bfloat16)
    for s in range(10000):
        acc_rn = (acc_rn.float() + 1e-3).bfloat16()
        acc_sr = S.to_dtype(acc_sr.float() + 1e-3, torch.bfloat16, key=5, site="acc", step=s,
                            backend="torch")
    assert float(acc_rn.float().mean()) == 1.0
    assert abs(float(acc_sr.float().mean()) - 11.0) < 0.1


def test_mx_keyed_rounding_same_on_every_device():
    x = torch.randn(16, 128)
    a = mx.quantize(x, "mxfp4", rounding="keyed", key=3, step=1)
    assert torch.equal(a.scale_codes, mx.quantize(x, "mxfp4").scale_codes)
    if torch.cuda.is_available():
        b = mx.quantize(x.cuda(), "mxfp4", rounding="keyed", key=3, step=1)
        assert torch.equal(a.codes, b.codes.cpu())
    with pytest.raises(F.FormatError, match="needs a key"):
        mx.quantize(x, "mxfp4", rounding="keyed")


def test_codes_registered():
    want = {"TS1000": "fp8_variant_mismatch", "TS1001": "fp8_variant_ambiguous_name",
            "TS1002": "fp8_variant_inferred", "TS1003": "fp8_variant_bits_conflict",
            "TS1004": "fp8_variant_undetermined", "TS1005": "fp8_target_unknown",
            "TS1006": "sround_bits_differ", "TS1007": "sround_biased",
            "TS1008": "mx_scale_differs", "TS1009": "mx_element_differs"}
    for c, s in want.items():
        assert codes.slug(c) == s
