"""Launch defaults that follow the GPU's SM count, shared memory and memory (triton_backend).

* On 70 SMs (the RTX 5070 Ti the defaults were measured on) every default config and every tuning
  candidate is the one main chose before: a frozen table of 168 cases (strict float32, TF32,
  float16, bfloat16, each also deterministic) in tests/data/contraction_defaults_70sm.json.
* Below FEW_SMS SMs strict float32 follows the 26 SM measurements (scripts/sweep_contraction.py).
* Deterministic mode never splits K, on any SM count, so its bits stay independent of the GPU's
  size; configs without split K give the same bits whatever the tile.
* Shared memory budget, split K workspace and local memory follow the device's limits.
"""

import json
import os
import pathlib
import tempfile

import pytest
import torch

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_kcache_"))

from truestar import kernel as K  # noqa: E402
from truestar.kernel import Graph, ir  # noqa: E402
from truestar.kernel import triton_backend as tb  # noqa: E402

TABLE = pathlib.Path(__file__).parent / "data" / "contraction_defaults_70sm.json"


def _has_triton():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


gpu = pytest.mark.skipif(not _has_triton(), reason="needs CUDA and Triton")


def linear_kernel(N, Kd, dtype="float32"):
    g = Graph()
    x = g.input("x", (K.sym("M"), Kd), dtype)
    w = g.input("w", (N, Kd), dtype)
    b = g.input("b", (N,), dtype)
    (_, kern), = g.fuse({"y": g.relu(x @ g.transpose(w, 0, 1) + b)})
    return kern


def plan(N, Kd, dtype="float32", tf32=False, det=False):
    opts = {}
    if tf32:
        opts["allow_tf32"] = True
    if det:
        opts["deterministic"] = True
    return tb.generate(ir.canonicalize(linear_kernel(N, Kd, dtype))[0], opts)[1]


@pytest.fixture
def limits(monkeypatch):
    """Pin device_limits to (sms, shared memory, total memory)."""
    def pin(sms, smem=tb.REF_SMEM, memory=tb.REF_MEMORY):
        monkeypatch.setattr(tb, "device_limits", lambda: (sms, smem, memory))
    return pin


def test_seventy_sms_give_exactly_the_measured_defaults(limits):
    limits(tb.REF_SMS)
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    assert len(table) == 168
    for key, (default, cands) in table.items():
        variant, shape = key.split(" ")
        M, N, Kd = (int(v) for v in shape.split("x"))
        p = plan(N, Kd, variant.split("-")[0], tf32="tf32" in variant, det=variant.endswith("-det"))
        env = {p.m.size.name: M}
        assert tb.default_config(p, env) == default, key
        assert tb.candidates(p, env) == cands, key


def test_waves_scale_seventy_sm_thresholds():
    assert tb._waves(64, 70) == 64 and tb._waves(32, 70) == 32
    assert tb._waves(64, 26) == 24 and tb._waves(32, 26) == 12
    assert tb._waves(64, 132) == 121 and tb._waves(1, 4) == 1


@pytest.mark.parametrize("shape,cfg", [
    # (M, N, K) -> (BM, BN, BK, warps, stages, SPLIT_K): the measured best or within 9% of it
    ((1, 4096, 4096), (16, 128, 32, 2, 4, 4)),
    ((16, 4096, 1024), (16, 128, 32, 2, 4, 4)),
    ((16, 1024, 16384), (16, 128, 32, 2, 4, 16)),
    ((32, 4096, 4096), (32, 64, 32, 2, 4, 2)),
    ((64, 2048, 1024), (64, 128, 16, 4, 3, 8)),
    ((512, 1024, 256), (64, 128, 16, 4, 3, 1)),
    ((1024, 1024, 1024), (128, 128, 16, 4, 3, 1)),
    ((2048, 4096, 1024), (128, 256, 16, 8, 3, 1)),
    ((8192, 256, 1024), (128, 128, 16, 4, 3, 1)),
])
def test_few_sms_follow_the_26_sm_measurements(limits, shape, cfg):
    limits(26)
    M, N, Kd = shape
    p = plan(N, Kd)
    c = tb.default_config(p, {p.m.size.name: M})
    assert (c["BM"], c["BN"], c["BK"], c["num_warps"], c["num_stages"], c["SPLIT_K"]) == cfg


@pytest.mark.parametrize("det,shape,cfg", [
    # 58 SMs (an L4): (M, N, K) -> (BM, BN, BK, warps, stages, SPLIT_K), the L4 sweep's best tile
    # wherever the 70 SM rules do not split K; where they split, their config is kept
    (True, (1, 4096, 4096), (16, 32, 64, 2, 3, 1)),     # K chunks, one program per tile
    (True, (4096, 4096, 4096), (128, 256, 16, 8, 3, 1)),   # K chunks: 128 x 256 measured best on the L4
    (True, (16, 1024, 16384), (16, 64, 64, 2, 3, 4)),   # K chunks split across programs: 70 SM rule
    (True, (64, 2048, 2048), (64, 128, 16, 4, 3, 4)),
    (True, (128, 256, 8192), (64, 128, 16, 4, 3, 16)),
    (True, (64, 2048, 1024), (64, 16, 32, 1, 3, 1)),     # K 1024: one chain, the 58 SM tile
    (True, (512, 1024, 256), (128, 64, 16, 4, 3, 1)),   # a wave of 128 x 64 tiles: 70 SM rule
    (True, (1024, 1024, 1024), (256, 128, 16, 8, 3, 1)),
    (True, (8192, 256, 1024), (64, 128, 16, 4, 3, 1)),
    (False, (64, 2048, 2048), (64, 128, 16, 4, 3, 8)),   # split K: 70 SM rule
    (False, (16, 1024, 16384), (16, 64, 64, 2, 3, 4)),
    (False, (1024, 1024, 1024), (256, 128, 16, 8, 3, 1)),
])
def test_mid_sms_follow_the_58_sm_measurements(limits, det, shape, cfg):
    limits(58)
    M, N, Kd = shape
    p = plan(N, Kd, det=det)
    c = tb.default_config(p, {p.m.size.name: M})
    assert (c["BM"], c["BN"], c["BK"], c["num_warps"], c["num_stages"], c["SPLIT_K"]) == cfg


@pytest.mark.parametrize("dtype,tf32,shape,cfg", [
    # float16 and bfloat16 share the rules; the measured best on 26 SMs unless noted
    ("float16", False, (2048, 4096, 1024), (64, 128, 64, 4, 3, 1)),
    ("bfloat16", False, (8192, 256, 1024), (64, 128, 64, 4, 3, 1)),
    ("bfloat16", False, (8192, 768, 256), (64, 64, 64, 4, 3, 1)),       # K 256: 70 SM rule
    ("float16", False, (1024, 1024, 1024), (64, 64, 64, 4, 3, 1)),      # under 5 waves
    ("float16", False, (64, 512, 8192), (32, 64, 128, 4, 3, 8)),
    ("bfloat16", False, (64, 1000, 2048), (32, 64, 128, 4, 3, 4)),      # 1.03x the best
    ("float16", False, (64, 2048, 1024), (32, 64, 128, 4, 3, 1)),       # K short: 70 SM rule
    ("float16", False, (256, 4096, 1024), (64, 64, 64, 4, 3, 1)),       # under 1024 rows
    ("float32", True, (1024, 1024, 1024), (64, 128, 32, 4, 3, 1)),
    ("float32", True, (64, 2048, 2048), (64, 64, 32, 4, 3, 4)),
    ("float32", True, (128, 256, 8192), (64, 64, 32, 4, 3, 16)),
    ("float32", True, (32, 4096, 4096), (32, 64, 64, 4, 3, 4)),
    ("float32", True, (4, 2048, 2048), (16, 32, 64, 4, 3, 4)),
    ("float32", True, (8192, 1024, 256), (64, 64, 32, 4, 3, 1)),        # K short: 70 SM rule
])
def test_few_sms_half_and_tf32_follow_the_26_sm_measurements(limits, dtype, tf32, shape, cfg):
    limits(26)
    M, N, Kd = shape
    p = plan(N, Kd, dtype, tf32)
    c = tb.default_config(p, {p.m.size.name: M})
    assert (c["BM"], c["BN"], c["BK"], c["num_warps"], c["num_stages"], c["SPLIT_K"]) == cfg
    p.deterministic = True                 # without split K the tall K rules give way
    d = tb.default_config(p, {p.m.size.name: M})
    assert d["SPLIT_K"] == 1
    if cfg[5] > 1:
        limits(70)
        assert d == tb.default_config(p, {p.m.size.name: M})


@pytest.mark.parametrize("sms", [8, 26, 47, 70, 132])
def test_deterministic_mode_splits_k_only_into_its_fixed_chunks(limits, sms):
    """Deterministic mode on any SM count: one chain over K (SPLIT_K 1) except strict float32 with
    K above DET_K_MIN, whose K chunks follow K alone and may run in up to one program each."""
    limits(sms)
    for M, N, Kd in [(1, 4096, 4096), (16, 1024, 16384), (64, 2048, 2048), (128, 256, 8192), (1024, 1024, 1024)]:
        for dtype, tf32 in (("float32", False), ("float32", True), ("float16", False)):
            p = plan(N, Kd, dtype, tf32, det=True)
            env = {p.m.size.name: M}
            cfgs = [tb.default_config(p, env)] + tb.candidates(p, env)
            if dtype == "float32" and not tf32 and Kd > tb.DET_K_MIN:
                assert p.kchunk == tb.DET_K_CHUNK
                assert all(c["SPLIT_K"] <= -(-Kd // p.kchunk) for c in cfgs)
                assert all(tb.check_config(p, env, c) is None for c in cfgs)
            else:
                assert p.kchunk == 0
                assert all(c["SPLIT_K"] == 1 for c in cfgs)


def test_pruning_counts_programs_in_waves_of_this_gpu(limits):
    p = plan(256, 1024)
    env = {p.m.size.name: 512}            # 128 x 256 tiles: 4 programs, 64 x 128: 16
    grid = [tb._cfg(*g) for g in tb.CONTRACTION_GRID["ieee"]]
    limits(70)
    many = tb._prune_contraction(p, env, grid)
    limits(8)                              # 8 SMs: 16 programs fill two waves, 4 is half a wave
    few = tb._prune_contraction(p, env, grid)
    assert any(c["BM"] == 128 and c["BN"] == 256 for c in few)
    assert few != many


def test_shared_memory_budget_follows_the_device(limits):
    p = plan(4096, 4096, "float16")
    cfg = tb._cfg(128, 128, 64, 4, 5)       # 5 stages of 128 x 64 and 64 x 128 halves: 160 KiB
    limits(70, smem=101376)
    big = tb._fit_shared(p, cfg)
    limits(70, smem=49152)                  # a GPU with 48 KiB per program
    small = tb._fit_shared(p, cfg)
    need = lambda c: 2 * (c["BM"] * c["BK"] + c["BK"] * c["BN"]) * c["num_stages"]  # noqa: E731
    assert need(big) <= 101376 - tb.SMEM_RESERVE and need(small) <= 49152 - tb.SMEM_RESERVE
    assert need(small) < need(big)
    assert tb.SMEM_BUDGET == 96 * 1024      # the reference budget is unchanged


def test_split_k_workspace_stays_a_small_share_of_gpu_memory(limits):
    p = plan(4096, 16384)
    env = {p.m.size.name: 2048}
    limits(26, memory=8 * 1024 ** 3)
    roomy = tb._split_k(p, env, 128, 256, 10 ** 6)
    limits(26, memory=512 * 1024 ** 2)      # 32 MiB of workspace allowed: one 2048 x 4096 slab
    tight = tb._split_k(p, env, 128, 256, 10 ** 6)
    assert roomy > tight >= 1
    assert tight * tb.workspace_elems(p, env) * 4 <= 512 * 1024 ** 2 // tb.WORKSPACE_SHARE


def test_device_limits_without_a_visible_gpu_are_the_reference(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert tb.device_limits() == (tb.REF_SMS, tb.REF_SMEM, tb.REF_MEMORY)


# ----------------------------------------------------------------------------- on the GPU


def data(M, N, Kd):
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(M, Kd, device="cuda", generator=gen)
    w = torch.randn(N, Kd, device="cuda", generator=gen) / Kd ** 0.5
    b = torch.randn(N, device="cuda", generator=gen)
    return x, w, b


@gpu
def test_deterministic_bits_do_not_depend_on_the_sm_class(limits):
    """The deterministic default chosen for 26 SMs and the one chosen for 70 SMs are different tiles
    without split K, and give the same bits: a deterministic result does not depend on the GPU's
    size."""
    kern = linear_kernel(4096, 1024)
    x, w, b = data(64, 4096, 1024)
    p = plan(4096, 1024, det=True)
    env = {p.m.size.name: 64}
    limits(26)
    few = tb.default_config(p, env)
    limits(70)
    many = tb.default_config(p, env)
    assert few != many and few["SPLIT_K"] == many["SPLIT_K"] == 1
    outs = [K.compile(kern, "cuda", deterministic=True, _config=c)(x=x, w=w, b=b) for c in (few, many)]
    assert torch.equal(outs[0], outs[1])


@gpu
@pytest.mark.parametrize("shape", [(1, 4096, 1024), (64, 2048, 1024), (300, 1024, 1024)])
def test_mid_sm_deterministic_tiles_keep_the_bits(limits, shape):
    """The 58 SM deterministic tiles (64 x 16 x 32 with one warp, 16 x 32 x 64) give
    the bits of the 70 SM tiles, and match float64 within the float32 tolerance."""
    M, N, Kd = shape
    kern = linear_kernel(N, Kd)
    x, w, b = data(M, N, Kd)
    p = plan(N, Kd, det=True)
    env = {p.m.size.name: M}
    limits(58)
    mid = tb.default_config(p, env)
    limits(70)
    ref = tb.default_config(p, env)
    assert mid != ref and mid["SPLIT_K"] == ref["SPLIT_K"] == 1
    outs = [K.compile(kern, "cuda", deterministic=True, _config=c)(x=x, w=w, b=b) for c in (mid, ref)]
    assert torch.equal(outs[0], outs[1])
    exact = torch.relu(x.double() @ w.double().T + b.double())
    assert ((outs[0].double() - exact).abs().max() / exact.abs().max()).item() < 2e-5


@gpu
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_half_deterministic_bits_do_not_depend_on_the_sm_class(limits, dtype):
    """As above for half precision tensor core dots: 64 x 128 x 64 (26 SMs) and 64 x 64 x 64 (70
    SMs) tiles of a 2048 x 4096 x 1024 linear give the same bits."""
    kern = linear_kernel(4096, 1024, dtype)
    x, w, b = (t.to(getattr(torch, dtype)) for t in data(2048, 4096, 1024))
    p = plan(4096, 1024, dtype, det=True)
    env = {p.m.size.name: 2048}
    limits(26)
    few = tb.default_config(p, env)
    limits(70)
    many = tb.default_config(p, env)
    assert few != many and few["SPLIT_K"] == many["SPLIT_K"] == 1
    outs = [K.compile(kern, "cuda", deterministic=True, _config=c)(x=x, w=w, b=b) for c in (few, many)]
    assert torch.equal(outs[0], outs[1])


def _spilling_config(kern, x, w, b):
    """A grid config whose compiled kernel spills on this GPU, with its local bytes per thread."""
    for g in tb.CONTRACTION_GRID["ieee"]:
        ck = K.compile(kern, "cuda", tune=False, _config=tb._cfg(*g))
        ck(x=x, w=w, b=b)
        L = list(ck._fast.values())[0]
        local = max(k.n_spills * 4 for k in L.fn.device_caches[x.device.index or 0][0].values()
                    if k.metadata.num_warps == g[3])
        if local > 0:
            return tb._cfg(*g), local
    return None, 0


@gpu
def test_a_config_needing_too_much_local_memory_is_refused_before_launch(monkeypatch):
    kern = linear_kernel(4096, 1024)
    x, w, b = data(256, 4096, 1024)
    cfg, local = _spilling_config(kern, x, w, b)
    if cfg is None:
        pytest.skip("no grid config spills registers on this GPU")
    monkeypatch.setattr(tb, "LOCAL_MAX_PER_THREAD", local - 1)
    with pytest.raises(tb.ConfigResourceError, match="TS700"):
        K.compile(kern, "cuda", tune=False, _config=cfg)(x=x, w=w, b=b)
    # a config that does not spill still runs, and its answer is right
    out = K.compile(kern, "cuda", tune=False, _config=tb._cfg(16, 64, 64, 2, 3))(x=x, w=w, b=b)
    ref = torch.relu(x.double() @ w.double().T + b.double())
    assert ((out.double() - ref).abs().max() / ref.abs().max()).item() < 1e-5


@gpu
def test_the_tuner_skips_a_refused_candidate_with_its_reason(monkeypatch, tmp_path):
    from truestar.kernel import tune
    monkeypatch.setenv("TRUESTAR_CACHE_DIR", str(tmp_path / "cache"))   # no tuned config on disk
    K.clear_memory_cache()
    kern = linear_kernel(4096, 1024)
    x, w, b = data(256, 4096, 1024)
    cfg, local = _spilling_config(kern, x, w, b)
    if cfg is None:
        pytest.skip("no grid config spills registers on this GPU")
    monkeypatch.setattr(tb, "LOCAL_MAX_PER_THREAD", local - 1)
    safe = tb._cfg(16, 64, 64, 2, 3)
    ck = K.compile(kern, "cuda", tune=True)
    monkeypatch.setattr(tb, "candidates", lambda p_, env: [safe, cfg])
    before = len(tune.stats.log)
    ck(x=x, w=w, b=b)
    rec = tune.stats.log[before]
    assert any("TS700" in why for _, why in rec["failed"])
    assert ck.launch_configs()[0][1] == safe
