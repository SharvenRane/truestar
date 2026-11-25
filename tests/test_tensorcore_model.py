"""Bit accurate tensor core models (docs/TENSOR_CORE_MODELS.md): hand computed cases from the
published behaviour, the sim: profiles wired into check, and bitwise agreement with a local sm_120
GPU when one is present."""

import time

import numpy as np
import pytest
import torch
from torch import nn

from truestar.check import check
from truestar.devsim import get_profile, simulate
from truestar.devsim.profiles import tensor_core_profiles
from truestar.devsim.tensorcore import (FORMATS, MatrixUnit, UnmodelledFormat, round_to, split,
                                        unit_for, unit_matmul)


def dot(unit, a, b, c=0.0, fmt="fp16"):
    return float(unit_matmul(np.array([a], dtype=float), np.array(b, dtype=float)[:, None],
                             np.array([[c]]), unit, fmt)[0, 0])


def pad(v, n):
    return list(v) + [0.0] * (n - len(v))


# ----------------------------------------------------------------------------- number formats


def test_split_and_reject():
    neg, sig, ex = split(np.array([1.5, -2.0 ** -24, 0.0]), FORMATS["fp16"])
    assert list(neg) == [False, True, False]
    assert list(sig) == [1536, 1, 0] and list(ex) == [0, -14, -14]  # 2^-24 is subnormal in fp16
    with pytest.raises(ValueError, match="not a finite fp16"):
        split(np.array([1.0 + 2.0 ** -12]), FORMATS["fp16"])


def test_round_to_rounds_once():
    # torch's float64 to float16 cast goes through float32 and lands on the tie: 0x1.f4p+3
    x = np.array([float.fromhex("0x1.f420008cp+3")])
    assert float(round_to(x, FORMATS["fp16"], "rne")[0]) == float.fromhex("0x1.f44p+3")
    assert float(round_to(x, FORMATS["fp16"], "rz")[0]) == float.fromhex("0x1.f40p+3")
    big = np.array([1e39, -1e39])
    assert list(round_to(big, FORMATS["fp32"], "rne")) == [np.inf, -np.inf]
    assert list(round_to(big, FORMATS["fp32"], "rz")) == [FORMATS["fp32"].maxval, -FORMATS["fp32"].maxval]
    assert float(round_to(np.array([2.0 ** -150 * 1.5]), FORMATS["fp32"], "rne")[0]) == 2.0 ** -149


# ----------------------------------------------------------------------------- hand computed cases


def test_products_stay_unnormalized_v100():
    """arXiv 2512.07004 section 4.1.1: 2.25 as 1.5 x 1.5 keeps exponent 0, so two 2^-23 products
    survive; as 1 x 2.25 it has exponent 1 and they are truncated; as c it also truncates them."""
    v100 = unit_for("v100", "fp16")
    small = ([2.0 ** -12, 2.0 ** -12], [2.0 ** -11, 2.0 ** -11])
    assert dot(v100, [1.5] + small[0] + [0], [1.5] + small[1] + [0]) == 2.25 + 2.0 ** -22
    assert dot(v100, [1.0] + small[0] + [0], [2.25] + small[1] + [0]) == 2.25
    assert dot(v100, [0.0] + small[0] + [0], [0.0] + small[1] + [0], c=2.25) == 2.25


def test_alignment_truncates_toward_zero():
    """1 - 2^-26: the negative term is below the 25 bit grid and dropped (sign magnitude), so the
    result is 1. A two's complement cut would give 1 - 2^-25, truncated to 1 - 2^-24."""
    u = unit_for("sm120", "fp16")
    assert dot(u, pad([1.0, -2.0 ** -12], 16), pad([1.0, 2.0 ** -14], 16)) == 1.0
    rd = MatrixUnit("rd", ("fp16",), "fp32", 16, 25, cut="rd")
    assert dot(rd, pad([1.0, -2.0 ** -12], 16), pad([1.0, 2.0 ** -14], 16)) == 1.0 - 2.0 ** -24


def test_group_boundary_and_chaining():
    """1 + 2^-24 + 2^-24 is 1 + 2^-23 only when both small products share a group with the 1;
    across groups each partial truncates back to 1."""
    u = unit_for("sm120", "fp16")
    for j in range(2, 32):
        a = pad([1.0, 2.0 ** -12], 32)
        b = pad([1.0, 2.0 ** -12], 32)
        a[j], b[j] = 2.0 ** -12, 2.0 ** -12
        want = 1.0 + 2.0 ** -23 if j < 16 else 1.0
        assert dot(u, a, b) == want, j
    a100 = unit_for("a100", "fp16")  # groups of 8
    for j, want in ((7, 1.0 + 2.0 ** -23), (8, 1.0)):
        a = pad([1.0, 2.0 ** -12], 16)
        b = pad([1.0, 2.0 ** -12], 16)
        a[j], b[j] = 2.0 ** -12, 2.0 ** -12
        assert dot(a100, a, b) == want, j


def test_exponent_floor_a100_and_h100():
    """Section 4.1.2 and 4.1.6: with every product tiny and c = 0 the alignment exponent is held
    at -132 (A100) or -133 (H100); a subnormal c moves it to -126."""
    s6 = sum(2.0 ** i for i in range(-155, -149))
    s7 = sum(2.0 ** i for i in range(-156, -149))

    def bf(prods, n):  # products p as (p * 2^70) x 2^-70, both bf16 values
        return pad([p * 2.0 ** 70 for p in prods], n), pad([2.0 ** -70] * len(prods), n)

    a100, h100 = unit_for("a100", "bf16"), unit_for("h100", "bf16")
    assert dot(a100, *bf([s6, 2.0 ** -156, 2.0 ** -156], 8), fmt="bf16") == 2.0 ** -149
    assert dot(a100, *bf([s6, 2.0 ** -156, 2.0 ** -157, 2.0 ** -157], 8), fmt="bf16") == 0.0
    assert dot(a100, *bf([2.0 ** -150, 2.0 ** -151, 2.0 ** -151], 8), c=2.0 ** -127,
               fmt="bf16") == 2.0 ** -127
    assert dot(h100, *bf([s7, 2.0 ** -157, 2.0 ** -158, 2.0 ** -158], 16), fmt="bf16") == 2.0 ** -149
    assert dot(h100, *bf([s7, 2.0 ** -157 + 2.0 ** -158, 2.0 ** -159, 2.0 ** -159], 16),
               fmt="bf16") == 0.0


def test_narrow_fp8_accumulator():
    """Ada and Hopper wgmma keep 13 fraction bits for fp8: c = 1 plus two 2^-14 products stays 1
    (section 4.1.4). B200 and sm_120 keep 25 and return 1 + 2^-13."""
    a = pad([2.0 ** -7, 2.0 ** -7], 32)
    b = pad([2.0 ** -7, 2.0 ** -7], 32)
    assert dot(unit_for("ada", "e4m3"), a[:16], b[:16], c=1.0, fmt="e4m3") == 1.0
    assert dot(unit_for("h100", "e4m3"), a, b, c=1.0, fmt="e4m3") == 1.0
    assert dot(unit_for("b200", "e4m3"), a, b, c=1.0, fmt="e4m3") == 1.0 + 2.0 ** -13
    assert dot(unit_for("sm120", "e4m3"), a, b, c=1.0, fmt="e4m3") == 1.0 + 2.0 ** -13


def test_hopper_mma_sync_fp8_interleaves_pairs():
    """Section 4.1.6: 1, 2^-24 and a third 2^-24 at position j give 1 + 2^-23 only when j is in
    the same half (alternating pairs) as positions 0 and 1."""
    u = unit_for("h100-mma", "e5m2")
    for j in range(2, 32):
        a = pad([1.0, 2.0 ** -12], 32)
        b = pad([1.0, 2.0 ** -12], 32)
        a[j], b[j] = 2.0 ** -12, 2.0 ** -12
        same = (j // 2) % 2 == 0
        assert dot(u, a, b, fmt="e5m2") == (1.0 + 2.0 ** -23 if same else 1.0), j


def test_zero_is_plus_zero_and_specials():
    u = unit_for("sm120", "fp16")
    assert np.signbit(dot(u, pad([-0.0], 16), pad([1.0], 16), c=-0.0)) == False  # noqa: E712
    assert np.isnan(dot(u, pad([np.inf], 16), pad([0.0], 16)))
    assert dot(u, pad([np.inf], 16), pad([1.0], 16)) == np.inf


def test_unmodelled_format_and_batching():
    with pytest.raises(UnmodelledFormat):
        unit_matmul(np.ones((2, 4)), np.ones((4, 2)), None, unit_for("v100", "fp16"), "bf16")
    with pytest.raises(UnmodelledFormat):
        unit_for("v100", "e4m3")
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 5, 40)).astype(np.float16).astype(float)
    b = rng.standard_normal((3, 40, 6)).astype(np.float16).astype(float)
    u = unit_for("h100", "fp16")
    both = unit_matmul(a, b, None, u, "fp16")
    for i in range(3):
        assert np.array_equal(both[i], unit_matmul(a[i], b[i], None, u, "fp16"))


def test_speed_256_cubed():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((256, 256)).astype(np.float16).astype(float)
    b = rng.standard_normal((256, 256)).astype(np.float16).astype(float)
    t = time.perf_counter()
    d = unit_matmul(a, b, None, unit_for("sm120", "fp16"), "fp16")
    assert time.perf_counter() - t < 10.0
    assert np.abs(d - a @ b).max() < 1e-3


# ----------------------------------------------------------------------------- sim: profiles


def test_profiles_are_cited_and_only_sm120_is_validated():
    profs = {p.name: p for p in tensor_core_profiles()}
    assert {"sim:h100-fp8", "sim:b200-fp16", "sim:a100-bf16", "sim:sm120-fp16"} <= set(profs)
    assert {n for n, p in profs.items() if p.validated} == {
        "sim:sm120-fp16", "sim:sm120-bf16", "sim:sm120-tf32", "sim:sm120-fp8",
        "sim:ada-fp16", "sim:ada-bf16", "sim:ada-tf32"}  # Ada fitted on an L4, not its fp8
    for p in profs.values():
        unit_for(p.mma, p.mma_fmt)  # every profile names a modelled data path
        assert p.ideal().mma == "exact"


def test_simulate_routes_mm_through_the_unit():
    torch.manual_seed(0)
    x = torch.randn(6, 48).half().float()
    w = torch.randn(48, 5).half().float()
    with simulate("sim:sm120-fp16"):
        y = x @ w
    want = unit_matmul(x.double().numpy(), w.double().numpy(), None, unit_for("sm120", "fp16"),
                       "fp16")
    assert np.array_equal(y.double().numpy(), want)
    with simulate(get_profile("sim:sm120-fp16").ideal()):
        z = x @ w
    exact = round_to(x.double().numpy() @ w.double().numpy(), FORMATS["fp32"], "rne")
    assert np.array_equal(z.double().numpy(), exact)


def test_check_codes_on_tensor_core_profiles():
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).eval()
    x = torch.randn(8, 16)
    r = check(mlp, x, backends=["sim:h100-fp8"], drill_ops=False).results[0]
    assert r.status == "ok" and {"TS520", "TS1094", "TS1097"} <= set(r.codes)
    r = check(mlp, x, backends=["sim:sm120-bf16"], drill_ops=False).results[0]
    assert r.status == "ok" and "TS1097" in r.codes and "TS1094" not in r.codes
    cnn = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Flatten(), nn.Linear(4 * 6 * 6, 3)).eval()
    r = check(cnn, torch.randn(2, 3, 8, 8), backends=["sim:h100-fp16"], drill_ops=False).results[0]
    assert "TS1096" in r.codes and "TS525" not in r.codes


# ----------------------------------------------------------------------------- the real GPU


def _sm120():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (12, 0)


@pytest.mark.skipif(not _sm120(), reason="needs an sm_120 GPU (RTX 50 series)")
@pytest.mark.parametrize("dtype,fmt", [(torch.float16, "fp16"), (torch.bfloat16, "bf16")])
@pytest.mark.parametrize("k", [16, 64])
def test_cublas_matches_sm120_bitwise(dtype, fmt, k):
    """65,536 dot products through cuBLAS (fp32 output), compared bit for bit."""
    g = torch.Generator().manual_seed(k)
    a = (torch.randn(256, k, generator=g) * torch.exp2(torch.randint(-6, 6, (256, k), generator=g)))
    b = (torch.randn(k, 256, generator=g) * torch.exp2(torch.randint(-6, 6, (k, 256), generator=g)))
    a, b = a.to(dtype), b.to(dtype)
    got = torch.mm(a.cuda(), b.cuda(), out_dtype=torch.float32).cpu()
    want = unit_matmul(a.double().numpy(), b.double().numpy(), None, unit_for("sm120", fmt), fmt)
    same = got.view(torch.int32) == torch.from_numpy(want.astype(np.float32)).view(torch.int32)
    assert int(same.sum()) == same.numel()


@pytest.mark.skipif(not _sm120(), reason="needs an sm_120 GPU (RTX 50 series)")
def test_scaled_mm_fp8_matches_sm120_bitwise():
    g = torch.Generator().manual_seed(3)
    a = torch.randn(128, 64, generator=g).to(torch.float8_e4m3fn)
    b = torch.randn(128, 64, generator=g).to(torch.float8_e4m3fn)  # B is passed column major
    one = torch.tensor(1.0, device="cuda")
    got = torch._scaled_mm(a.cuda(), b.cuda().t(), one, one, out_dtype=torch.float32).cpu()
    want = unit_matmul(a.double().numpy(), b.double().numpy().T, None, unit_for("sm120", "e4m3"),
                       "e4m3")
    assert np.array_equal(got.numpy().view(np.int32), want.astype(np.float32).view(np.int32))
