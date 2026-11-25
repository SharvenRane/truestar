"""AMD GPUs (ROCm, HIP) in the kernel layer, the compiled executor and the check backends.

Three groups, each with its own skip reason:

* always: the target table, the limits override, TF32 and MFMA options, the relief ladder, the
  codes, and the TS883 refusal of "rocm" on a torch without HIP;
* compile only, wherever Triton's AMD backend is importable (no GPU needed): every case of
  truestar.kernel.rocm_cases compiles for gfx942, gfx1100 and gfx1103 with the target's wave size,
  CUs and LDS, without spilling registers, and the resource guards refuse what does not fit;
* on an AMD GPU with a ROCm build of torch: every case runs and matches float64, the compiled
  executor runs on "rocm", and deterministic results are bitwise stable.
"""

import os
import tempfile

import pytest
import torch

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_kcache_"))

from truestar import codes  # noqa: E402
from truestar import kernel as K  # noqa: E402
from truestar.kernel import ir, rocm, rocm_cases  # noqa: E402
from truestar.kernel import triton_backend as tb  # noqa: E402

COMPILE_OK, COMPILE_WHY = rocm.compile_available()
HIP_OK, HIP_WHY = rocm.hip_available()
compile_only = pytest.mark.skipif(not COMPILE_OK, reason=f"AMD compile path unavailable: {COMPILE_WHY}")
on_amd = pytest.mark.skipif(not HIP_OK, reason=f"needs an AMD GPU and a ROCm build of torch: {HIP_WHY}")
CASES = {name: (k, shapes, kw) for name, k, shapes, kw in rocm_cases.cases()}


def _plan(name, opts=None):
    k = CASES[name][0]
    ck = ir.canonicalize(k)[0]
    env = {}
    _, shapes, _ = CASES[name]
    from truestar.kernel import reference
    user = reference.bind_sizes(k, shapes)
    ck, in_map, _ = ir.canonicalize(k)
    for b in k.inputs:
        cb = next(x for x in ck.inputs if x.name == in_map[b.name])
        for su, sc in zip(b.shape, cb.shape):
            if isinstance(sc, ir.Sym):
                env[sc.name] = user[su.name]
    return tb.generate(ck, opts or {})[1], env


# ----------------------------------------------------------------------------- always


def test_target_table():
    t = rocm.target("gfx942")
    assert (t.wave, t.cus, t.lds, t.matrix, t.xf32) == (64, 304, 65536, "mfma", True)
    assert t.scratch_per_lane == 8387584 // 64
    for a, cus in (("gfx1100", 96), ("gfx1103", 12)):
        r = rocm.target(a)
        assert (r.wave, r.cus, r.lds, r.matrix, r.xf32) == (32, cus, 65536, "wmma", False)
        assert r.scratch_per_lane == 8388352 // 32
    assert rocm.target("GFX942:sramecc+:xnack-") is t        # torch's gcnArchName form


def test_unknown_target_is_ts880():
    with pytest.raises(rocm.UnknownTarget, match=r"\[TS880\].*gfx9999"):
        rocm.target("gfx9999")
    assert rocm.UnknownTarget.code == "TS880"


def test_codes_exist():
    for c in ("TS880", "TS881", "TS882", "TS883", "TS884", "TS885", "TS886"):
        assert c in codes.CODES
    assert codes.severity("TS883") == "error" and codes.severity("TS885") == "info"


def test_target_limits_override_and_restore():
    before = tb.device_limits()
    with tb.target_limits(304, 65536, 192 * 2 ** 30):
        assert tb.device_limits() == (304, 65536, 192 * 2 ** 30)
        assert tb.smem_budget() == 65536 - tb.SMEM_RESERVE
        with tb.target_limits(12, 65536, 8 * 2 ** 30):
            assert tb.device_limits()[0] == 12
        assert tb.device_limits()[0] == 304
    assert tb.device_limits() == before


def test_defaults_follow_cu_count():
    """The SM aware defaults read CUs on AMD: the small M strict float32 linear splits K to fill
    304 CUs and to fill 12 (below FEW_SMS); deterministic mode splits only into its K chunks (fixed by K), the same on both."""
    p, env = _plan("linear_m16_k8192_float32")
    pd, _ = _plan("linear_m16_k8192_float32", {"deterministic": True})
    got = {}
    for arch in ("gfx942", "gfx1103"):
        t = rocm.target(arch)
        with tb.target_limits(t.cus, t.lds, t.memory):
            got[arch] = tb.default_config(p, env)
            assert pd.kchunk == tb.DET_K_CHUNK          # K 8192: chunks fixed by K, not by the CUs
            assert all(c["SPLIT_K"] <= 8192 // pd.kchunk for c in [tb.default_config(pd, env)] + tb.candidates(pd, env))
    assert got["gfx942"]["SPLIT_K"] > 1 and got["gfx1103"]["SPLIT_K"] > 1
    assert got["gfx942"] != got["gfx1103"]


def test_lds_budget_trims_stages():
    p, env = _plan("linear_1024_float32")
    big = dict(BM=128, BN=128, BK=64, GROUP=8, num_warps=8, num_stages=4, KTAIL=True, SPLIT_K=1)
    on_nv = tb._fit_shared(p, big)
    with tb.target_limits(304, 65536, 192 * 2 ** 30):
        on_amd = tb._fit_shared(p, big)
    need = lambda c: (c["BM"] + c["BN"]) * c["BK"] * 4 * c["num_stages"]  # noqa: E731
    assert need(on_amd) <= 65536 - tb.SMEM_RESERVE < need(big)
    assert need(on_amd) <= need(on_nv)


def test_tf32_only_where_the_target_has_it():
    o, notes = rocm.kernel_opts(rocm.target("gfx942"), {"allow_tf32": True})
    assert o == {"allow_tf32": True, "amd_arch": "gfx942"} and notes == []
    for a in ("gfx1100", "gfx1103"):
        o, notes = rocm.kernel_opts(rocm.target(a), {"allow_tf32": True, "deterministic": True})
        assert o == {"deterministic": True, "amd_arch": a}
        assert len(notes) == 1 and notes[0].startswith("[TS884]")


def test_mfma_instruction_option():
    cdna, rdna = rocm.target("gfx942"), rocm.target("gfx1100")
    for kind in ("contraction", "attention", "conv"):
        assert rocm.launch_options(cdna, kind) == {"matrix_instr_nonkdim": 16}
        assert rocm.launch_options(rdna, kind) == {}
    assert rocm.launch_options(cdna, "tile") == {} and rocm.launch_options(cdna, "window") == {}


@pytest.mark.parametrize("name", ["linear_m16_k8192_float32", "linear_m16_k8192_float32_det",
                                  "linear_1024_float16", "attention_causal_d128_float16", "conv3x3_float32"])
@pytest.mark.parametrize("arch", ["gfx942", "gfx1100"])
def test_relief_ladder_keeps_split_k_and_valid_configs(name, arch):
    t = rocm.target(arch)
    p, env = _plan(name, {"deterministic": True} if name.endswith("_det") else None)
    with tb.target_limits(t.cus, t.lds, t.memory):
        d = tb.default_config(p, env)
        ladder = rocm.relief_ladder(p, env, d, t)
    assert ladder and d not in ladder
    for c in ladder:
        assert tb.check_config(p, env, c) is None
        assert c["num_warps"] * t.wave <= 1024
        assert c.get("SPLIT_K", 1) == d.get("SPLIT_K", 1)


def test_relief_ladder_empty_for_tile_and_window():
    t = rocm.target("gfx1100")
    for name in ("softmax_float32", "maxpool3x3_s2_float32"):
        p, env = _plan(name)
        assert rocm.relief_ladder(p, env, tb.default_config(p, env), t) == []


@pytest.mark.skipif(HIP_OK, reason="this machine has ROCm: the refusal is for machines without it")
def test_rocm_refused_without_hip():
    k = CASES["pointwise_float32"][0]
    with pytest.raises(rocm.RocmUnavailable, match=r"\[TS883\]"):
        K.compile(k, "rocm")
    from truestar.compiler import CompiledExecutor
    with pytest.raises(rocm.RocmUnavailable, match=r"\[TS883\]"):
        CompiledExecutor("rocm")
    with pytest.raises(rocm.RocmUnavailable):
        rocm.current_target()
    from truestar.stacks import probe
    b, why = probe("rocm+inductor")
    assert b is None and ("ROCm build" in why or "AMD GPU" in why)
    from truestar.backends import BackendUnavailable, resolve_backends
    with pytest.raises(BackendUnavailable, match="rocm"):
        resolve_backends(["rocm"])


# ----------------------------------------------------------------------------- compile only


@compile_only
@pytest.mark.parametrize("arch", ["gfx942", "gfx1100", "gfx1103"])
@pytest.mark.parametrize("name", list(CASES))
def test_every_loop_shape_compiles_for_amd(name, arch):
    k, shapes, kw = CASES[name]
    b = rocm.build(k, arch, shapes, **kw)
    t = b.target
    expect = {"pointwise": "tile", "softmax": "tile", "row": "tile", "linear": "contraction",
              "attention": "attention", "conv3x3": "conv", "maxpool3x3": "window"}[name.split("_")[0]]
    assert b.plan_kind == expect
    for f in b.functions:
        assert f.hsaco_bytes > 0
        assert f.lanes == f.num_warps * t.wave <= 1024
        assert f.lds <= t.lds
        assert f.scratch == 0 and f.vgpr_spills == 0, (name, arch, b.config, f)
    assert b.notes[-1].startswith("[TS885]")
    if kw.get("deterministic"):
        assert len(b.functions) == (2 if b.config["SPLIT_K"] > 1 else 1)   # K chunks may split (fixed by K)
    if name == "linear_m16_k8192_float32":
        assert b.config["SPLIT_K"] > 1 and [f.name for f in b.functions] == ["kern", "kern_splitk"]
    if kw.get("allow_tf32"):
        assert b.tf32 == t.xf32
        assert ('input_precision="tf32"' in b.source) == t.xf32
        assert any(n.startswith("[TS884]") for n in b.notes) == (not t.xf32)
    if b.plan_kind in rocm.DOT_SHAPES and t.matrix == "mfma":
        assert b.functions[0].options.get("matrix_instr_nonkdim") == 16


@compile_only
def test_wave32_default_spills_and_ladder_relieves():
    """Measured: the NVIDIA derived default for a float16 linear spills on wave32 RDNA3 (128
    lanes per 4 wave program) and not on wave64 CDNA; the ladder's first step (8 waves) does not."""
    k, shapes, kw = CASES["linear_1024_float16"]
    with tb.target_limits(96, 65536, 24 * 2 ** 30):
        p, env = _plan("linear_1024_float16")
        d = tb.default_config(p, env)
    forced = rocm.build(k, "gfx1100", shapes, config=d)
    assert forced.functions[0].scratch > 0
    relieved = rocm.build(k, "gfx1100", shapes)
    assert relieved.config != d and relieved.functions[0].scratch == 0
    assert any(n.startswith("[TS886]") for n in relieved.notes)
    assert rocm.build(k, "gfx942", shapes).functions[0].scratch == 0


@compile_only
def test_lds_over_limit_is_ts882():
    k, shapes, kw = CASES["linear_1024_float32"]
    cfg = dict(BM=128, BN=128, BK=64, GROUP=8, num_warps=8, num_stages=3, KTAIL=True, SPLIT_K=1)
    with pytest.raises(rocm.LdsError, match=r"\[TS882\].*LDS"):
        rocm.build(k, "gfx942", shapes, config=cfg)


@compile_only
def test_candidates_build_or_are_refused_with_a_reason():
    k, shapes, kw = CASES["linear_1024_float16"]
    res = rocm.build_candidates(k, "gfx942", shapes, **kw)
    assert len(res) > 1
    built = [r for _, r in res if isinstance(r, rocm.AmdBuild)]
    assert built and built[0].config == res[0][0]
    for c, r in res:
        if not isinstance(r, rocm.AmdBuild):
            assert any(code in str(r) for code in ("TS700", "TS881", "TS882")), (c, r)


# ----------------------------------------------------------------------------- on an AMD GPU


@on_amd
def test_current_target_reads_the_device():
    t = rocm.current_target()
    assert t.arch.startswith("gfx") and t.wave in (32, 64) and t.cus > 0 and t.lds > 0
    sms, smem, mem = tb.device_limits()
    assert sms == t.cus and mem == t.memory


@on_amd
@pytest.mark.parametrize("name", list(CASES))
def test_every_case_runs_and_matches_float64(name):
    """Every kernel is checked against float64 on its first call by the kernel layer itself; this
    also compares with the reference interpreter directly."""
    from truestar.kernel import reference
    k, shapes, kw = CASES[name]
    gen = torch.Generator().manual_seed(0)
    dts = {b.name: getattr(torch, b.dtype) for b in k.inputs}
    xs = {n: torch.randn(*s, generator=gen).to(dts[n]).cuda() for n, s in shapes.items()}
    fn = K.compile(k, "rocm", allow_tf32=kw.get("allow_tf32", False), deterministic=kw.get("deterministic", False),
                   tune=False)     # the default config (and its relief on wave32); the tuner has its own tests
    out = fn(**xs)
    if not isinstance(out, dict):
        out = {k.outputs[0].name: out}
    ref = reference.run_reference_torch(k, xs, xs[next(iter(xs))].device)    # float64 on the GPU
    for o, got in out.items():
        want = ref[o].double()
        err = ((got.double() - want).abs().max() / want.abs().max().clamp_min(1e-6)).item()
        tol = {"float32": 2e-3 if kw.get("allow_tf32") else 1e-4, "float16": 2e-2, "bfloat16": 6e-2}
        assert err < tol[str(got.dtype).split(".")[-1]], (name, err)


@on_amd
def test_deterministic_contraction_is_batch_independent_on_amd():
    k, shapes, kw = CASES["linear_m16_k8192_float32_det"]
    fn = K.compile(k, "rocm", deterministic=True)
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(300, 8192, generator=gen).cuda()
    w = torch.randn(1024, 8192, generator=gen).cuda()
    b = torch.randn(1024, generator=gen).cuda()
    big = fn(x=x, w=w, b=b)
    small = fn(x=x[:7].contiguous(), w=w, b=b)
    again = fn(x=x, w=w, b=b)
    assert torch.equal(big, again)
    assert torch.equal(big[:7], small)


@on_amd
def test_compiled_executor_runs_on_rocm():
    import torch.nn as nn
    from truestar import ir as tir
    from truestar.compiler import CompiledExecutor, compile_program
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8)).eval()
    B = torch.export.Dim("B", min=1, max=64)
    prog = tir.import_module(model, (torch.randn(2, 16),), dynamic_shapes=({0: B},))
    cp = compile_program(prog, "rocm", "float32", deterministic=True)
    x = torch.randn(5, 16)
    y = cp([x])[0]
    assert y.device.type == "cuda" and cp.verified
    want = model.double()(x.double())
    assert (y.double().cpu() - want).abs().max().item() / want.abs().max().item() < 1e-4
    assert torch.equal(y, cp([x])[0])
    assert CompiledExecutor("rocm").name == "compiled:rocm:float32"


# ----------------------------------------------------------------------------- relief path, emulated


def _cuda_ok():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available() and torch.cuda.device_count() > 0 and not rocm.hip_version()
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_ok(), reason="emulates the AMD launch path on an NVIDIA GPU: needs CUDA and Triton")
def test_relief_path_of_the_launcher_runs_and_verifies(monkeypatch):
    """The hardware code path of the relief ladder (CompiledKernel._prepare and _relieve), run on
    an NVIDIA GPU with the AMD target faked and the default reported as spilling: the first ladder
    config is launched, verified against float64 at its first call and recorded as the source."""
    from truestar.kernel import compiler as kc
    k, shapes, kw = CASES["linear_1024_float16"]
    fn = K.compile(k, "cuda", tune=False)
    fn.opts["amd_arch"] = "gfx1100"
    t = rocm.target("gfx1100")
    monkeypatch.setattr(rocm, "current_target", lambda device=None: t)
    monkeypatch.setattr(rocm, "launch_options", lambda target, kind: {})
    seen = []
    orig = kc.CompiledKernel._set_launch

    def spy(self, got, L, env, cfg, ints):
        seen.append(cfg)
        return orig(self, got, L, env, cfg, ints)

    monkeypatch.setattr(kc.CompiledKernel, "_set_launch", spy)
    monkeypatch.setattr(kc, "_spills", lambda ks: 64 if seen[-1] is None else 0)
    gen = torch.Generator().manual_seed(0)
    xs = {n: (torch.randn(*s, generator=gen) * 0.5).half().cuda() for n, s in shapes.items()}
    y = fn(**xs)
    L = next(iter(fn._fast.values()))
    assert seen[0] is None and len(seen) == 2
    assert L.config_source == "default, spill relief" and L.config == seen[1]
    assert seen[1]["num_warps"] == 2 * tb.default_config(fn.art.plan, L.env)["num_warps"]
    ref = torch.relu(xs["x"].double() @ xs["w"].double().T + xs["b"].double())
    assert ((y.double() - ref).abs().max() / ref.abs().max()).item() < 2e-3
