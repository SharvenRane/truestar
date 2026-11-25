"""Quantized kernels: exact formats, the kernel verdict (a) and the format verdict (b) (docs/QUANT.md)."""

from __future__ import annotations

import math

import pytest
import torch

from truestar import codes
from truestar.quant import formats as F

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="quantized kernels need CUDA")


# ----------------------------------------------------------------------------- formats (CPU)


@pytest.mark.parametrize("name,dt", [("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)])
def test_fp8_tables_and_rounding_match_torch_casts(name, dt):
    c = F.FP8[name]
    codes_ = torch.arange(256, dtype=torch.uint8)
    tv = codes_.view(dt).double()
    mv = c.decode(codes_)
    assert bool(((tv == mv) | (tv.isnan() & mv.isnan())).all())
    fin = mv[torch.isfinite(mv)].unique()
    mids = (fin[1:] + fin[:-1]) / 2                        # every tie
    g = torch.Generator().manual_seed(0)
    x = torch.cat([torch.randn(200_000, generator=g, dtype=torch.float64) * c.max_finite / 8,
                   mids, -mids]).clamp(-c.max_finite, c.max_finite)
    assert bool((c.decode(c.encode(x)) == x.to(dt).double()).all())


def test_e2m1_values_ties_saturation_and_nan():
    assert F.E2M1.values[:8] == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, 100.0, -0.25, -5.0],
                     dtype=torch.float64)
    want = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 0.0, -4.0]   # ties to the even code
    assert F.E2M1.decode(F.E2M1.encode(x)).tolist() == want
    with pytest.raises(F.FormatError):
        F.E2M1.encode(torch.tensor([math.nan], dtype=torch.float64))


def test_pow2_and_correctly_rounded_division_are_exact():
    e = torch.arange(-126, 128)
    assert F.pow2(e).tolist() == [2.0 ** k for k in range(-126, 128)]
    assert F.e8m0_decode(torch.tensor([127, 0, 254, 255])).tolist()[:3] == [1.0, 2.0 ** -127, 2.0 ** 127]
    y = torch.rand(100_000, dtype=torch.float32) * 1000
    for d in (448.0, 57344.0, 2688.0):
        assert bool((F.div_f32_rn(y, d) == y / d).all())     # CPU torch division is correct


def test_parse_and_refusals():
    assert F.parse("int4").group == 128 and F.parse("int4:g32").group == 32
    assert F.parse("fp8_e5m2:tensor").scale == "tensor"
    assert F.parse("nvfp4").group == 16 and F.parse("mxfp4").group == 32
    for bad in ("int3", "int4:g48", "fp8_e3m4", "fp8_e4m3:row", "nvfp8"):
        with pytest.raises(F.FormatError):
            F.parse(bad)
    with pytest.raises(F.FormatError):
        F.quantize_weight(torch.randn(8, 100), "int4:g32")       # K not a multiple of the group


@pytest.mark.parametrize("spec", F.SPECS)
def test_quantize_weight_layout_and_exact_dequantize(spec):
    g = torch.Generator().manual_seed(1)
    w = torch.randn(24, 256, generator=g) * 0.1
    w[3] = 0                                                       # an all zero channel
    q = F.quantize_weight(w, spec)
    d = q.dequantize()
    assert d.dtype == torch.float64 and d.shape == w.shape
    assert bool((d[3] == 0).all())
    rel = float((d - w.double()).norm() / w.double().norm())
    assert rel < {"int8": 0.01, "fp8": 0.06}.get(q.fmt.kind, 0.15), rel
    c = q.codes()
    if q.fmt.kind == "int4":
        assert int(c.min()) >= -8 and int(c.max()) <= 7
        # layout: byte j of a group holds inputs j (low nibble) and j + G/2 (high nibble)
        G = q.fmt.group
        b = int(q.data[0, 0])
        lo, hi = b & 15, b >> 4
        assert (lo - 16 if lo >= 8 else lo) == int(c[0, 0])
        assert (hi - 16 if hi >= 8 else hi) == int(c[0, G // 2])
    if q.fmt.kind in ("nvfp4", "mxfp4"):
        b = int(q.data[0, 0])
        assert (b & 15) == int(c[0, 0]) and (b >> 4) == int(c[0, 1])   # input 2i in the low nibble


def test_codes_registered():
    for c, s in {"TS460": "quant_kernel_error", "TS461": "quant_quantizer_error",
                 "TS462": "quant_format_flips", "TS463": "quant_format_loss",
                 "TS464": "quant_format_unsupported"}.items():
        assert codes.slug(c) == s


# ----------------------------------------------------------------------------- kernels (CUDA)


def _xw(M, K, N, dtype=torch.float16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, device="cuda", generator=g).to(dtype)
    w = torch.randn(N, K, device="cuda", generator=g) * 0.05
    return x, w


@cuda
@pytest.mark.parametrize("spec", F.SPECS)
@pytest.mark.parametrize("M", [1, 33, 200])
def test_every_format_kernel_passes_verdict_a(spec, M):
    from truestar.quant import verify as V
    x, w = _xw(M, 512, 200)                         # N = 200: a ragged last tile
    chk = V.verify_kernel(x, F.quantize_weight(w, spec))
    assert chk.ok, chk.message()
    assert chk.ratio <= 1.0 and chk.quantizer_mismatches == 0 and chk.code is None


@cuda
@pytest.mark.parametrize("spec", ["int8", "int4:g64"])
def test_weight_only_kernels_in_bfloat16(spec):
    from truestar.quant import verify as V
    x, w = _xw(70, 256, 96, torch.bfloat16)
    chk = V.verify_kernel(x, F.quantize_weight(w, spec))
    assert chk.ok, chk.message()


@cuda
def test_activation_quantizers_are_exact():
    """FP8 and both FP4 quantizers round once (round to odd before the format's rounding), so
    every code equals the format's rule; scales are correctly rounded."""
    from truestar.quant import kernels as Q, verify as V
    g = torch.Generator(device="cuda").manual_seed(3)
    x = (torch.randn(256, 512, device="cuda", generator=g)
         * torch.logspace(-6, 1, 256, device="cuda")[:, None]).half()
    x[5, :64] = 0
    for act in (Q.quant_act_fp8(x, "e4m3"), Q.quant_act_fp8(x, "e5m2"),
                Q.quant_act_fp4(x, "nvfp4"), Q.quant_act_fp4(x, "mxfp4")):
        bad, amb = V.check_quantizer(x, act)
        assert bad == 0, (act.fmt.spec, bad, amb)


@cuda
def test_broken_quantizer_output_is_ts461():
    from truestar.quant import kernels as Q, verify as V
    x, w = _xw(16, 256, 64)
    qw = F.quantize_weight(w, "mxfp4")
    y, act = Q.quantized_matmul(x, qw, return_act=True)
    act.data[0, 0] ^= 0x01                            # one wrong activation code
    chk = V.check_matmul(x, qw, y, act)
    assert chk.code == "TS461" and chk.quantizer_mismatches >= 1


INJECTED = [
    ("int8", "channel_scale_shift", "a wrong scale"),
    ("int4:g128", "group_scale_shift", "got the scale of group 1"),
    ("int4:g128", "last_group_scale", "group 7 got the scale of group 6"),
    ("int4:g128", "group_boundary_half", "the kernel uses groups of 64"),
    ("int4:g32", "nibble_swap", "low and high nibbles"),
    ("fp8_e4m3:channel", "tensor_scale_for_channel", "a wrong scale"),
    ("fp8_e4m3:channel", "fp16_accumulate", "not finite"),
    ("nvfp4", "block_scale_shift", "got the scale of block 1"),
    ("nvfp4", "block32_for_nvfp4", "a wrong scale index"),
    ("nvfp4", "nibble_swap", "two codes of each byte"),
    ("mxfp4", "block_scale_shift", "a wrong scale index"),
]


@cuda
@pytest.mark.parametrize("spec,bug,named", INJECTED)
def test_injected_kernel_bug_is_verdict_a_and_named(spec, bug, named):
    from truestar.quant import verify as V
    x, w = _xw(64, 1024, 128)
    qw = F.quantize_weight(w, spec)
    chk = V.verify_kernel(x, qw, inject=bug, layer="mlp.down")
    assert chk.code == "TS460" and not chk.ok
    assert chk.ratio > 100, chk.ratio
    text = " ".join(chk.diagnosis)
    assert named in text, text
    assert "mlp.down" in chk.message()
    # verdict (b) is computed from float64 references only: the same with or without the bug
    loss = V.layer_format_loss(x, w, qw)
    assert loss == V.layer_format_loss(x, w, F.quantize_weight(w, spec))


@cuda
def test_subtle_bug_hides_in_format_loss_against_the_float_baseline():
    """The usual check (quantized output against the float output) sees the last group bug as
    a small change inside INT4's own loss; verdict (a) fails it by orders of magnitude."""
    from truestar.quant import kernels as Q, verify as V
    x, w = _xw(64, 4096, 256)
    qw = F.quantize_weight(w, "int4:g128")
    yf = x.double() @ w.double().T
    rel = lambda y: float((y.double() - yf).norm() / yf.norm())
    good = rel(Q.quantized_matmul(x, qw))
    bad = rel(Q.quantized_matmul(x, qw, inject="last_group_scale"))
    assert abs(bad - good) < 0.1 * good                # baseline moves under 10% of format loss
    chk = V.verify_kernel(x, qw, inject="last_group_scale")
    assert chk.ratio > 10 and chk.worst_case_ratio > 1   # past even the rigorous worst case bound


@cuda
def test_no_silent_fallback():
    from truestar.quant import kernels as Q
    qw = F.quantize_weight(torch.randn(64, 128), "int4:g64")
    with pytest.raises(Q.QuantKernelError, match="CUDA only"):
        Q.quantized_matmul(torch.randn(4, 128, dtype=torch.float16), qw)
    qg = F.quantize_weight(torch.randn(64, 128, device="cuda"), "int8")
    with pytest.raises(Q.QuantKernelError, match="float16 or bfloat16"):
        Q.quantized_matmul(torch.randn(4, 128, device="cuda"), qg)
    with pytest.raises(ValueError, match="unknown injected bug"):
        Q.quantized_matmul(torch.randn(4, 128, device="cuda").half(), qg, inject="nope")


# ----------------------------------------------------------------------------- model level


@cuda
def test_model_verdicts_and_injected_layer_named():
    from truestar.quant import model as QM
    from truestar.rlkernels import llama as L
    cfg = L.Config(vocab=256, dim=128, layers=2, heads=4, kv_heads=2, hidden=256, max_len=64)
    m = L.make_model(seed=0, device="cuda", dtype=torch.float16, config=cfg)
    ids = torch.randint(0, 256, (2, 17), device="cuda", generator=torch.Generator(device="cuda").manual_seed(0))
    res = QM.compare(m, ids, ["int4:g64", "nvfp4"])
    for r in res["formats"]:
        assert r["kernel_verdict"]["ok"], r["kernel_verdict"]["failing"]
        assert r["kernel_verdict"]["layers_checked"] == 14
        assert set(r["codes"]) <= {"TS462", "TS463"}
    clean = res["formats"][0]
    bug = QM.compare(m, ids, ["int4:g64"], inject={"blocks.1.mlp.down_proj": "last_group_scale"},
                     layer_inputs=False)["formats"][0]
    failing = bug["kernel_verdict"]["failing"]
    assert [f["layer"] for f in failing] == ["blocks.1.mlp.down_proj"]
    assert "TS460" in bug["codes"]
    assert bug["format_verdict"]["delta_ce"] == clean["format_verdict"]["delta_ce"]
    assert bug["format_verdict"]["flips"] == clean["format_verdict"]["flips"]


@cuda
def test_quant_linear_raises_on_a_kernel_bug():
    from truestar.quant import model as QM, verify as V
    lin = torch.nn.Linear(256, 64, bias=True).cuda().half()
    ql = QM.QuantLinear(lin, "int4:g128", "proj")
    x = torch.randn(3, 5, 256, device="cuda", dtype=torch.float16)
    y = ql(x)
    assert y.shape == (3, 5, 64) and ql.checks[-1].ok
    ql.inject = "group_scale_shift"
    with pytest.raises(V.QuantKernelMismatchError, match="proj"):
        ql(torch.randn(7, 256, device="cuda", dtype=torch.float16))   # a new shape: checked
