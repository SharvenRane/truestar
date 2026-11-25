"""Autotuning of GPU launch configs and the attention schedule rules.

Every config the tuner can pick is run and checked (by the built in float64 verification and by
an independent float64 PyTorch computation), so whichever config wins is correct.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import warnings

import pytest
import torch

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_kcache_"))

from truestar import kernel as K  # noqa: E402
from truestar.kernel import Graph, compiler, tune  # noqa: E402
from truestar.kernel import triton_backend as tb  # noqa: E402


def _has_triton():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


HAS_GPU = _has_triton()
pytestmark = pytest.mark.skipif(not HAS_GPU, reason="needs CUDA and Triton")

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def close(out, ref, tol):
    out = out.detach().double().cpu()
    ref = ref.detach().double().cpu()
    assert out.shape == ref.shape
    err = (out - ref).abs().max().item()
    scale = max(ref.abs().max().item(), 1e-6)
    assert err / scale <= tol, f"relative error {err / scale:.3g} > {tol}"


TOL = {torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 3e-2}


@pytest.fixture
def fresh_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUESTAR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TRUESTAR_TUNE_BUDGET", "60")
    K.clear_memory_cache()
    tune.stats.reset()
    yield tmp_path / "cache"
    K.clear_memory_cache()


def _attn(dtype, B=1, H=2, D=64, mask="causal", Dv=None):
    g = Graph()
    N = K.sym("N")
    q = g.input("q", (B, H, N, D), dtype)
    k = g.input("k", (B, H, N, D), dtype)
    v = g.input("v", (B, H, N, Dv or D), dtype)
    s = (q @ k.T) * (D ** -0.5)
    if mask == "causal":
        s = g.pointwise(lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF), s, with_index=True)
    elif mask == "window":
        s = g.pointwise(lambda s, b, h, i, j: K.where((j <= i) & (j > i - 70), s, K.NEG_INF), s,
                        with_index=True)
    return g, {"o": g.softmax(s) @ v}


def _attn_kernel(dtype, **kw):
    g, outs = _attn(dtype, **kw)
    parts = g.fuse(outs)
    assert len(parts) == 1
    return parts[0][1]


def _torch_attn(q, k, v, mask):
    q, k, v = q.double().cpu(), k.double().cpu(), v.double().cpu()
    n = q.shape[-2]
    s = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    i = torch.arange(n)[:, None]
    j = torch.arange(n)[None, :]
    if mask == "causal":
        s = s.masked_fill(~(j <= i), float("-inf"))
    elif mask == "window":
        s = s.masked_fill(~((j <= i) & (j > i - 70)), float("-inf"))
    return torch.softmax(s, -1) @ v


def _qkv(n, dtype, B=1, H=2, D=64, Dv=None, seed=3):
    torch.manual_seed(seed)
    mk = lambda d: torch.randn(B, H, n, d).to("cuda", dtype)  # noqa: E731
    return mk(D), mk(D), mk(Dv or D)


def _gelu_kernel():
    g = Graph()
    x = g.input("x", (K.sym("M"), K.sym("N")))
    r = g.input("r", (K.sym("M"), K.sym("N")))
    y = 0.5 * x * (1.0 + g.tanh(0.79788 * (x + 0.044715 * x * x * x))) + r
    return g.fuse({"y": y})[0][1]


# ----------------------------------------------------------------------------- the tuner


CHILD = textwrap.dedent("""
    import json, sys, warnings
    import torch
    from truestar import kernel as K
    from truestar.kernel import Graph, tune
    g = Graph()
    N = K.sym("N")
    q, k, v = (g.input(n, (1, 2, N, 64), "float16") for n in "qkv")
    s = g.pointwise(lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF), (q @ k.T) * 0.125,
                    with_index=True)
    fn = g.compile({"o": g.softmax(s) @ v}, "cuda", tune=True)
    torch.manual_seed(0)
    t = [torch.randn(1, 2, 384, 64, device="cuda", dtype=torch.float16) for _ in range(3)]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn(q=t[0], k=t[1], v=t[2])
    ck = fn.kernels[0][1]
    (_, cfg, src), = ck.launch_configs()
    print("JSON" + json.dumps(dict(searches=tune.stats.searches, hits=tune.stats.cache_hits,
                                   rejected=tune.stats.rejected, src=src, cfg=cfg,
                                   tune_s=ck.tune_seconds, verify_s=ck.verify_seconds,
                                   warnings=[str(x.message) for x in w])))
""")


def _child(cache):
    env = dict(os.environ, TRUESTAR_CACHE_DIR=str(cache), PYTHONPATH=SRC,
               TRUESTAR_TUNE_BUDGET="60")
    p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True,
                       timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("JSON")][-1]
    return json.loads(line[4:])


def test_tuning_picks_a_config_and_a_second_process_reuses_it(tmp_path):
    cache = tmp_path / "cache"
    first = _child(cache)
    assert first["searches"] == 1 and first["hits"] == 0 and first["src"] == "tuned"
    assert first["tune_s"] > 0 and first["verify_s"] > 0  # the tuned config was verified
    files = list((cache / "triton" / "tune").glob("*.json"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text())
    assert rec["config"] == first["cfg"] and rec["timed"] and rec["tune_seconds"] > 0
    second = _child(cache)
    assert second["searches"] == 0 and second["hits"] == 1 and second["src"] == "cache"
    assert second["cfg"] == first["cfg"] and second["tune_s"] == 0
    assert second["verify_s"] > 0  # a config read from disk is still verified on first use
    assert not second["warnings"]


def test_same_bucket_shares_one_search_and_a_new_bucket_tunes(fresh_cache):
    kern = _attn_kernel("float16")
    fn = K.compile(kern, "cuda", tune=True)
    for n in (300, 384):  # both round up to 512
        q, k, v = _qkv(n, torch.float16)
        close(fn(q=q, k=k, v=v), _torch_attn(q, k, v, "causal"), TOL[torch.float16])
    assert tune.stats.searches == 1
    q, k, v = _qkv(600, torch.float16)  # bucket 1024
    fn(q=q, k=k, v=v)
    assert tune.stats.searches == 2
    srcs = [s for _, _, s in fn.launch_configs()]
    assert srcs == ["tuned", "cache", "tuned"]  # the 384 launch reused the 300 search


def test_constant_edit_reuses_tuned_config_but_not_the_binary(fresh_cache):
    def kern(c):
        g = Graph()
        x = g.input("x", (K.sym("M"), K.sym("N")))
        return g.fuse({"y": g.tanh(x * c) + 1.0})[0][1]
    x = torch.randn(300, 700, device="cuda")
    f1 = K.compile(kern(0.5), "cuda", tune=True)
    close(f1(x=x), torch.tanh(x.double() * 0.5) + 1, 1e-5)
    f2 = K.compile(kern(0.25), "cuda", tune=True)
    assert f2.key != f1.key and f2.tune_base_key == f1.tune_base_key
    close(f2(x=x), torch.tanh(x.double() * 0.25) + 1, 1e-5)  # verified on its first call
    assert tune.stats.searches == 1 and f2.launch_configs()[0][2] == "cache"
    assert f2.verify_seconds > 0
    # a structural edit is a new tuning key
    g = Graph()
    xx = g.input("x", (K.sym("M"), K.sym("N")))
    f3 = K.compile(g.fuse({"y": g.sigmoid(xx * 0.5) + 1.0})[0][1], "cuda", tune=True)
    assert f3.tune_base_key != f1.tune_base_key


@pytest.mark.parametrize("damage",["garbage", "edited_config", "bad_config", "wrong_key",
                                    "old_version", "truncated"])
def test_corrupt_cached_config_is_detected_and_retuned(fresh_cache, damage):
    kern = _gelu_kernel()
    x, r = torch.randn(256, 512, device="cuda"), torch.randn(256, 512, device="cuda")
    K.compile(kern, "cuda", tune=True)(x=x, r=r)
    assert tune.stats.searches == 1
    (path,) = (fresh_cache / "triton" / "tune").glob("*.json")
    good = json.loads(path.read_text())
    rec = dict(good)
    if damage == "garbage":
        path.write_bytes(b"\x00\xffnot json")
    elif damage == "truncated":
        path.write_text(path.read_text()[:40])
    else:
        if damage == "edited_config":
            rec["config"] = dict(good["config"], num_warps=2)  # digest no longer matches
        elif damage == "bad_config":  # a self consistent file whose config cannot launch
            rec["config"] = dict(good["config"], BC=48)
            rec["digest"] = tune._digest(rec["key"], rec["config"])
        elif damage == "wrong_key":
            rec["key"] = "0" * 32
        elif damage == "old_version":
            rec["version"] = tune.TUNE_VERSION + 99
        path.write_text(json.dumps(rec))
    K.clear_memory_cache()
    with pytest.warns(K.TuneCacheWarning):
        fn = K.compile(kern, "cuda", tune=True)
        out = fn(x=x, r=r)
    assert tune.stats.rejected == 1 and tune.stats.searches == 2
    (_, cfg, src), = fn.launch_configs()
    assert src == "tuned" and tb.check_config(fn.art.plan, {"s0": 256, "s1": 512}, cfg) is None
    healed = json.loads(path.read_text())
    assert healed["digest"] == tune._digest(healed["key"], healed["config"])
    ref = 0.5 * x * (1 + torch.tanh(0.79788 * (x + 0.044715 * x ** 3))) + r
    close(out, ref, 1e-5)
    # and the rewritten file is read back without a warning
    K.clear_memory_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fn = K.compile(kern, "cuda", tune=True)
        fn(x=x, r=r)
    assert fn.launch_configs()[0][2] == "cache"


def test_modes_auto_off_and_budget(fresh_cache, monkeypatch):
    kern = _gelu_kernel()
    small = [torch.randn(64, 64, device="cuda") for _ in range(2)]
    big = [torch.randn(4096, 4096, device="cuda") for _ in range(2)]
    fn = K.compile(kern, "cuda")  # auto: only launches big enough to matter (2^24 elements) are tuned
    fn(x=small[0], r=small[1])
    fn(x=big[0], r=big[1])
    assert [s for _, _, s in fn.launch_configs()] == ["default", "tuned"]
    monkeypatch.setenv("TRUESTAR_TUNE", "0")
    K.clear_memory_cache()
    fn = K.compile(kern, "cuda")
    fn(x=big[0], r=big[1])
    assert fn.launch_configs()[0][2] == "default"
    assert fn.launch_configs()[0][1] == tb.default_config(fn.art.plan, {"s0": 4096, "s1": 4096})
    # a zero budget still runs: only the default is timed, the rest are skipped and reported
    monkeypatch.setenv("TRUESTAR_TUNE", "auto")
    monkeypatch.setenv("TRUESTAR_TUNE_BUDGET", "0")
    K.clear_memory_cache()
    tune.stats.reset()
    q, k, v = _qkv(256, torch.float16)
    fa = K.compile(_attn_kernel("float16"), "cuda", tune=True)
    close(fa(q=q, k=k, v=v), _torch_attn(q, k, v, "causal"), TOL[torch.float16])
    rec = tune.stats.log[-1]
    assert len(rec["timed"]) == 1 and rec["skipped_for_budget"] == len(
        tb.candidates(fa.art.plan, {"s0": 256})) - 1
    assert fa.launch_configs()[0][1] == tb.default_config(fa.art.plan, {"s0": 256})


def test_check_config_rejects_bad_configs():
    kern = _attn_kernel("float16")
    from truestar.kernel import ir
    p = tb.plan(ir.canonicalize(kern)[0])
    env = {"s0": 1000}
    good = tb.default_config(p, env)
    assert tb.check_config(p, env, good) is None
    for c in tb.candidates(p, env):
        assert tb.check_config(p, env, c) is None, c
    bad = [None, [], dict(good, BM=48), dict(good, BN=0), dict(good, LPT=1),
           dict(good, num_warps=3), dict(good, num_stages=0), dict(good, num_stages=True),
           dict(good, extra=1), {k_: v_ for k_, v_ in good.items() if k_ != "BN"},
           dict(good, BM="64"), dict(good, BM=512)]
    for c in bad:
        assert tb.check_config(p, env, c) is not None, c
    with pytest.raises(ValueError):
        tb.prepare(None, None, p, env, config=dict(good, BM=48))


# ----------------------------------------------------------------------------- correctness of
# every config the tuner may pick, across shapes, dtypes and masks


@pytest.mark.parametrize("dtype,mask", [(torch.float16, "causal"), (torch.float16, "window"),
                                        (torch.float16, "none"), (torch.bfloat16, "causal"),
                                        (torch.float32, "causal")])
def test_every_attention_candidate_is_correct(dtype, mask):
    name = str(dtype).replace("torch.", "")
    kern = _attn_kernel(name, mask=mask)
    from truestar.kernel import ir
    p = tb.plan(ir.canonicalize(kern)[0])
    n = 333  # several query and key blocks, ragged last block
    q, k, v = _qkv(n, dtype, seed=n)
    ref = _torch_attn(q, k, v, mask)
    for cfg in tb.candidates(p, {"s0": n}):
        fn = K.compile(kern, "cuda", _config=cfg)  # first call verifies this config
        close(fn(q=q, k=k, v=v), ref, TOL[dtype])
        assert fn.launch_configs()[0][1:] == (cfg, "forced")


@pytest.mark.parametrize("D,Dv", [(128, 128), (32, 96)])
def test_attention_candidates_other_head_dims(D, Dv):
    kern = _attn_kernel("float16", D=D, Dv=Dv)
    from truestar.kernel import ir
    p = tb.plan(ir.canonicalize(kern)[0])
    q, k, v = _qkv(300, torch.float16, D=D, Dv=Dv)
    ref = _torch_attn(q, k, v, "causal")
    for cfg in tb.candidates(p, {"s0": 300}):
        close(K.compile(kern, "cuda", _config=cfg)(q=q, k=k, v=v), ref, TOL[torch.float16])


def test_differential_attention_candidates():
    g = Graph()
    N = K.sym("N")
    q1, k1, q2, k2 = (g.input(n, (1, 2, N, 32), "float16") for n in ("q1", "k1", "q2", "k2"))
    v = g.input("v", (1, 2, N, 64), "float16")
    lam = g.input("lam", (1, 2, 1, 1), "float32")
    causal = lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF)  # noqa: E731
    p1 = g.softmax(g.pointwise(causal, (q1 @ k1.T) * 32 ** -0.5, with_index=True))
    p2 = g.softmax(g.pointwise(causal, (q2 @ k2.T) * 32 ** -0.5, with_index=True))
    (_, kern), = g.fuse({"o": (p1 - lam * p2) @ v})
    from truestar.kernel import ir
    p = tb.plan(ir.canonicalize(kern)[0])
    torch.manual_seed(5)
    n = 290
    ins = {nm: torch.randn(1, 2, n, 32, device="cuda", dtype=torch.float16)
           for nm in ("q1", "k1", "q2", "k2")}
    ins["v"] = torch.randn(1, 2, n, 64, device="cuda", dtype=torch.float16)
    ins["lam"] = torch.rand(1, 2, 1, 1, device="cuda") * 0.8
    ref = (_attn_probs(ins["q1"], ins["k1"]) - ins["lam"].double().cpu()
           * _attn_probs(ins["q2"], ins["k2"])) @ ins["v"].double().cpu()
    for cfg in tb.candidates(p, {"s0": n}):
        close(K.compile(kern, "cuda", _config=cfg)(**ins), ref, TOL[torch.float16])


def _attn_probs(q, k):
    q, k = q.double().cpu(), k.double().cpu()
    n = q.shape[-2]
    s = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    s = s.masked_fill(~torch.ones(n, n, dtype=torch.bool).tril(), float("-inf"))
    return torch.softmax(s, -1)


@pytest.mark.parametrize("shape", [(1000, 3000), (7, 70000), (4096, 1)])
def test_every_tile_candidate_is_correct(shape):
    kern = _gelu_kernel()
    from truestar.kernel import ir
    p = tb.plan(ir.canonicalize(kern)[0])
    torch.manual_seed(1)
    x, r = torch.randn(*shape, device="cuda"), torch.randn(*shape, device="cuda")
    ref = 0.5 * x.double() * (1 + torch.tanh(0.79788 * (x.double() + 0.044715 * x.double() ** 3))) \
        + r.double()
    env = {"s0": shape[0], "s1": shape[1]}
    for cfg in tb.candidates(p, env):
        close(K.compile(kern, "cuda", _config=cfg)(x=x, r=r), ref, 1e-5)


@pytest.mark.parametrize("n", [300, 6000])
def test_every_reduction_candidate_is_correct(n):
    g = Graph()
    x = g.input("x", (K.sym("M"), K.sym("N")))
    (_, kern), = g.fuse({"y": g.softmax(x * 0.5)})
    from truestar.kernel import ir
    p = tb.plan(ir.canonicalize(kern)[0])
    torch.manual_seed(2)
    xt = torch.randn(37, n, device="cuda")
    ref = torch.softmax(xt.double() * 0.5, -1)
    for cfg in tb.candidates(p, {"s0": 37, "s1": n}):
        close(K.compile(kern, "cuda", _config=cfg)(x=xt), ref, 1e-5)


@pytest.mark.parametrize("dtype,tf32", [(torch.float16, False), (torch.bfloat16, False),
                                        (torch.float32, False), (torch.float32, True)])
@pytest.mark.parametrize("m", [300, 5])
def test_every_contraction_candidate_is_correct(dtype, tf32, m):
    g = Graph()
    name = str(dtype).replace("torch.", "")
    a = g.input("a", (K.sym("M"), K.sym("K")), name)
    b = g.input("b", (K.sym("K"), K.sym("N")), name)
    (_, kern), = g.fuse({"c": g.relu(a @ b)})
    from truestar.kernel import ir
    p = tb.plan(ir.canonicalize(kern)[0])
    p.tf32 = tf32
    torch.manual_seed(4)
    at = torch.randn(m, 200, device="cuda").to(dtype)
    bt = torch.randn(200, 170, device="cuda").to(dtype)
    ref = torch.relu(at.double() @ bt.double())
    env = {"s0": m, "s1": 200, "s2": 170}
    cands = tb.candidates(p, env)
    assert 2 <= len(cands) <= 1 + tb.CANDIDATES_MAX + 3
    # every config the tuner could ever time, not only the ones kept for this shape, plus split K
    # with 1 to 16 chunks (16 chunks of K 200 leaves some chunks empty) and both K loop forms
    cls = "half" if dtype != torch.float32 else ("tf32" if tf32 else "ieee")
    full = [tb._cfg(*g) for g in tb.CONTRACTION_GRID[cls]]
    full += [tb._cfg(64, 128, 32, 4, 3, split=s) for s in (2, 4, 16)]
    full += [tb._cfg(16, 64, 32, 4, 3, split=8, ktail=False), dict(cands[0], KTAIL=False)]
    for cfg in cands + full:
        cfg = tb._fit_shared(p, cfg)
        close(K.compile(kern, "cuda", allow_tf32=tf32, _config=cfg)(a=at, b=bt), ref,
              5e-3 if tf32 else TOL[dtype])


def test_contraction_default_config_follows_operands_and_tf32(monkeypatch):
    # the rules below are the 70 SM ones; fewer SMs are covered in tests/test_sm_aware.py
    monkeypatch.setattr(tb, "device_limits", lambda: (tb.REF_SMS, tb.REF_SMEM, tb.REF_MEMORY))
    g = Graph()
    for name in ("float32", "float16"):
        a = g.input("a" + name, (K.sym("M"), K.sym("K")), name)
        b = g.input("b" + name, (K.sym("K"), K.sym("N")), name)
        (_, kern), = Graph.fuse(g, {"c": a @ b})
        from truestar.kernel import ir
        for tf32 in (False, True):
            src, p = tb.generate(ir.canonicalize(kern)[0], {"allow_tf32": True} if tf32 else {})
            assert p.tf32 == tf32
            env = {"s0": 64, "s1": 2048, "s2": 2048}
            small = tb.default_config(p, env)
            big = tb.default_config(p, {"s0": 4096, "s1": 4096, "s2": 4096})
            assert big["BM"] >= 64 and big["BN"] >= 64 and big["SPLIT_K"] == 1
            # a small M gets enough programs: short row tiles, or split K (strict float32)
            progs = tb._programs(64, 2048, small["BM"], small["BN"]) * small["SPLIT_K"]
            assert progs >= 64 and (small["BM"] <= 32 or small["SPLIT_K"] > 1)
            p.deterministic = True
            det = tb.default_config(p, env)
            assert det["SPLIT_K"] == 1 and det["BM"] <= 32
            assert all(c["SPLIT_K"] == 1 for c in tb.candidates(p, env))
            assert tb.check_config(p, env, dict(det, SPLIT_K=2)) is not None
            p.deterministic = False
            if name == "float32" and not tf32:
                assert 'input_precision="ieee"' in src
            if name == "float32" and tf32:
                assert 'input_precision="tf32"' in src


# ----------------------------------------------------------------------------- schedule rules


def test_interior_bounds_of_causal_and_window_masks():
    from truestar.kernel import ir
    p = tb.plan(ir.canonicalize(_attn_kernel("float16"))[0])
    assert tb._interior_bounds(p) == (None, 0)  # keys j <= start_m need no mask
    p = tb.plan(ir.canonicalize(_attn_kernel("float16", mask="window"))[0])
    assert tb._interior_bounds(p) == (-69, 0)   # j > i - 70 means j >= i - 69
    p = tb.plan(ir.canonicalize(_attn_kernel("float16", mask="none"))[0])
    assert tb._interior_bounds(p) == (None, None)


def test_interior_loop_has_no_masks_and_scores_use_exp2():
    kern = _attn_kernel("float16")
    fn = K.compile(kern, "cuda", tune=False)
    src = fn.source
    split = src.split("if SPLIT:")[1].split("else:")[0]
    interior = split.split("for n0 in range(")[1]
    assert "mask=" not in interior and "tl.where(" not in interior.replace(
        "tl.where(mnew0 == float('-inf'), 0.0, mnew0)", "")
    assert "tl.exp2(" in src and "tl.exp(" not in src
    assert repr(0.125 * tb.LOG2E) in src  # log2(e) folded into the softmax scale


def test_lpt_and_split_switches_agree():
    kern = _attn_kernel("float16", mask="window")
    q, k, v = _qkv(500, torch.float16, seed=9)
    ref = _torch_attn(q, k, v, "window")
    from truestar.kernel import ir
    p = tb.plan(ir.canonicalize(kern)[0])
    base = tb.default_config(p, {"s0": 500})
    outs = []
    for lpt in (False, True):
        for split in (False, True):
            fn = K.compile(kern, "cuda", _config=dict(base, LPT=lpt, SPLIT=split))
            out = fn(q=q, k=k, v=v)
            close(out, ref, TOL[torch.float16])
            outs.append(out.float())
    for o in outs[1:]:
        assert (o - outs[0]).abs().max().item() < 2e-3


def test_tune_key_needs_no_triton(monkeypatch):
    """CPU only installs have no triton: computing a tune key must not import it."""
    import sys

    from truestar.kernel import tune as _tune

    monkeypatch.setitem(sys.modules, "triton", None)  # import triton now raises ImportError
    k1 = _tune.tune_key("k", {"B": 3, "N": 100}, ["float32"], "cpu")
    k2 = _tune.tune_key("k", {"B": 4, "N": 100}, ["float32"], "cpu")
    assert k1 == k2 and len(k1) == 32


# ----------------------------------------------------------------------------- first call cost


def _linear_kernel(N, Kd, epilogue):
    g = Graph()
    x = g.input("x", (K.sym("M"), Kd), "float32")
    w = g.input("w", (N, Kd), "float32")
    b = g.input("b", (N,), "float32")
    return g.fuse({"y": epilogue(g, x @ g.transpose(w, 0, 1) + b)})[0][1]


def _linear_data(M, N, Kd, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(M, Kd, device="cuda"), torch.randn(N, Kd, device="cuda") / Kd ** 0.5,
            torch.randn(N, device="cuda"))


def test_contraction_candidates_are_pruned_by_shape():
    """Fewer Triton compiles per search: tiles that cannot fill the GPU, or that waste most of
    their rows, are not timed; the default always is."""
    kern = _linear_kernel(2048, 2048, lambda g, y: g.relu(y))
    from truestar.kernel import ir
    p = tb.plan(ir.canonicalize(kern)[0])
    for M in (1, 16, 64, 896, 8192):
        env = {"s0": M}
        cands = tb.candidates(p, env)
        assert cands[0] == tb.default_config(p, env)
        assert len(cands) <= 1 + tb.CANDIDATES_MAX + 2
        assert len({tuple(sorted(c.items())) for c in cands}) == len(cands)
        if M <= 16:
            assert all(c["BM"] <= 32 for c in cands[1:tb.CANDIDATES_MAX + 1])
        if M == 8192:
            assert all(tb._programs(M, 2048, c["BM"], c["BN"]) >= 32 for c in cands)


def test_shape_cache_shares_a_config_between_epilogues(fresh_cache):
    """linear + relu tunes; linear + tanh with the same operands and sizes reads that config
    without a search, also from a later process (the file is on disk)."""
    x, w, b = _linear_data(1024, 1024, 1024)
    relu = K.compile(_linear_kernel(1024, 1024, lambda g, y: g.relu(y)), "cuda", tune=True)
    close(relu(x=x, w=w, b=b), torch.relu(x.double() @ w.double().T + b.double()), 1e-4)
    assert tune.stats.searches == 1
    tanh = K.compile(_linear_kernel(1024, 1024, lambda g, y: g.tanh(y)), "cuda", tune=True)
    close(tanh(x=x, w=w, b=b), torch.tanh(x.double() @ w.double().T + b.double()), 1e-4)
    assert tune.stats.searches == 1
    (_, cfg_r, src_r), = relu.launch_configs()
    (_, cfg_t, src_t), = tanh.launch_configs()
    assert (src_r, src_t) == ("tuned", "shape cache") and cfg_r == cfg_t
    # another weight layout is another shape: searched on its own
    g = Graph()
    xx = g.input("x", (K.sym("M"), 1024), "float32")
    ww = g.input("w", (1024, 1024), "float32")
    other = K.compile(g.fuse({"y": g.relu(xx @ ww)})[0][1], "cuda", tune=True)
    other(x=x, w=w)
    assert tune.stats.searches == 2


def test_nearby_row_counts_share_one_search(fresh_cache):
    """The row count of a contraction is bucketed by powers of four: 1100 and 4000 rows share
    the search done at 3000, 5000 rows searches again."""
    ck = K.compile(_linear_kernel(512, 1024, lambda g, y: g.relu(y)), "cuda", tune=True)
    for M in (3000, 1100, 4000):
        x, w, b = _linear_data(M, 512, 1024)
        close(ck(x=x, w=w, b=b), torch.relu(x.double() @ w.double().T + b.double()), 1e-4)
    assert tune.stats.searches == 1
    x, w, b = _linear_data(5000, 512, 1024)
    ck(x=x, w=w, b=b)
    assert tune.stats.searches == 2


def test_new_batch_size_reuses_a_compiled_variant(fresh_cache, monkeypatch):
    """A launch too small to tune reuses a config this kernel already compiled when the model
    says it is within tb.REUSE_SLACK of the default, so a new batch size compiles no Triton variant; it is
    still verified at the new shape, and it gives the same bits as the default would (no split K
    involved, results do not depend on the config)."""
    monkeypatch.setenv("TRUESTAR_TUNE", "auto")
    kern = _linear_kernel(768, 256, lambda g, y: g.relu(y))
    ck = K.compile(kern, "cuda")
    from truestar.kernel import ir
    p = ck.art.plan
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    x, w, b = _linear_data(8192, 768, 256)
    ck(x=x, w=w, b=b)                                # below 2^32 work: the default, compiled
    srcs = [s for _, _, s in ck.launch_configs()]
    assert srcs == ["default"]
    tuned = ck.launch_configs()[0][1]                # the config compiled at 8192 rows
    x7 = x[:896].contiguous()
    d = tb.default_config(p, {"s0": 896})
    expect_reuse = d != tuned and tb.estimate(p, {"s0": 896}, tuned, sms) <= tb.REUSE_SLACK * tb.estimate(
        p, {"s0": 896}, d, sms)
    out = ck(x=x7, w=w, b=b)
    close(out, torch.relu(x7.double() @ w.double().T + b.double()), 1e-4)
    cfg7, src7 = ck.launch_configs()[1][1:]
    if expect_reuse:
        assert src7 == "reused" and cfg7 == tuned
    else:
        assert src7 == "default" and cfg7 == d
    ref = K.compile(kern, "cuda", _config=d)(x=x7, w=w, b=b)
    if tuned["SPLIT_K"] == 1 and d["SPLIT_K"] == 1:
        assert torch.equal(out, ref)
    # 1300 rows: the default would be a 128 x 128 tile (a new variant); a compiled one (128 x 256
    # or 128 x 64) is rated within REUSE_SLACK of it: reused, and verified at the new shape
    x2 = x[:1300].contiguous()
    assert tb.default_config(p, {"s0": 1300}) not in (tuned, d)
    out2 = ck(x=x2, w=w, b=b)
    close(out2, torch.relu(x2.double() @ w.double().T + b.double()), 1e-4)
    cfg2, src2 = ck.launch_configs()[2][1:]
    assert src2 == "reused" and cfg2 in (tuned, d)
    assert ck.verify_seconds > 0


def test_reuse_never_picks_a_much_slower_config():
    kern = _linear_kernel(1024, 4096, lambda g, y: g.relu(y))
    from truestar.kernel import ir
    p = tb.plan(ir.canonicalize(kern)[0])
    big = tb.default_config(p, {"s0": 4096})
    small = tb.default_config(p, {"s0": 7})
    # the 128 row tile at 7 rows runs one wave of 8 programs over all of K: far slower
    assert tb.estimate(p, {"s0": 7}, big, 70) > tb.REUSE_SLACK * tb.estimate(p, {"s0": 7}, small, 70)


def test_reuse_is_only_for_small_launches(fresh_cache, monkeypatch):
    """A launch above REUSE_MAX_WORK compiles its own default even when a compiled config would
    do: it runs long and often enough to repay the compile."""
    monkeypatch.setenv("TRUESTAR_TUNE", "auto")
    ck = K.compile(_linear_kernel(768, 256, lambda g, y: g.relu(y)), "cuda")
    x, w, b = _linear_data(8192, 768, 256)
    ck(x=x, w=w, b=b)
    x2 = x[:2000].contiguous()                      # 3.9e8 multiply adds, above 2^28
    assert tb.work(ck.art.plan, {"s0": 2000}) >= tb.REUSE_MAX_WORK
    ck(x=x2, w=w, b=b)
    cfg, src = ck.launch_configs()[1][1:]
    assert src == "default" and cfg == tb.default_config(ck.art.plan, {"s0": 2000})
