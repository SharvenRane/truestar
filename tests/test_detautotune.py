"""Deterministic autotuning: only parameters that keep the summation order (docs/KERNEL_IR.md 6b, P49)."""

import pytest
import torch

from truestar import kernel as K
from truestar.kernel import Graph, tune
from truestar.kernel import triton_backend as tb

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def attention(dtype="float32", causal=False):
    g = Graph()
    T, N = K.sym("T"), K.sym("N")
    q = g.input("q", (1, 2, T, 64), dtype)
    k, v = (g.input(n, (1, 2, N, 64), dtype) for n in "kv")
    s = (q @ k.T) * 0.125
    if causal:
        s = g.pointwise(lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF), s, with_index=True)
    (_, kern), = g.fuse({"o": g.softmax(s) @ v})
    return kern


def softmax(cols=1000):
    g = Graph()
    x = g.input("x", (K.sym("R"), cols), "float32")
    (_, kern), = g.fuse({"y": g.softmax(x)})
    return kern


def plan_env(ck):
    return ck.art.plan, next(iter(ck._fast.values())).env


@gpu
def test_order_candidates_differ_from_the_default_only_in_order_free_parameters():
    for kern, ins in ((attention(), {n: torch.randn(1, 2, 300, 64, device="cuda") for n in "qkv"}),
                      (softmax(), {"x": torch.randn(64, 1000, device="cuda")})):
        ck = K.compile(kern, "cuda", tune=False, deterministic=True)
        ck(**ins)
        p, env = plan_env(ck)
        d = tb.default_config(p, env)
        free = tb.ORDER_FREE[tb.config_kind(p, env)]
        cands = tb.order_candidates(p, env)
        assert cands[0] == d and len(cands) > 1
        for c in cands:
            assert {k for k in c if c[k] != d.get(k)} <= set(free)
            assert tb.order_key(p, env, c) == tb.order_key(p, env, d)


@gpu
@pytest.mark.parametrize("dtype,causal", [("float32", True), ("float16", False)])
def test_order_tuned_attention_gives_the_default_bits(dtype, causal, monkeypatch, tmp_path):
    monkeypatch.setenv("TRUESTAR_CACHE_DIR", str(tmp_path))
    K.clear_memory_cache()
    kern = attention(dtype, causal)
    gen = torch.Generator(device="cuda").manual_seed(0)
    ins = {n: torch.randn(1, 2, 300, 64, device="cuda", generator=gen).to(getattr(torch, dtype)) for n in "qkv"}
    ck = K.compile(kern, "cuda", tune=True, deterministic=True)
    out = ck(**ins)
    (_, cfg, src), = ck.launch_configs()
    p, env = plan_env(ck)
    d = tb.default_config(p, env)
    assert src == "order tuned" and tb.order_key(p, env, cfg) == tb.order_key(p, env, d)
    assert torch.equal(out, K.compile(kern, "cuda", _config=d)(**ins))
    # a second process reads the choice back from the tuning cache and keeps the bits
    K.clear_memory_cache()
    again = K.compile(kern, "cuda", tune=True, deterministic=True)
    assert torch.equal(again(**ins), out)
    K.clear_memory_cache()


@gpu
def test_a_choice_that_changes_the_bits_is_refused(monkeypatch, tmp_path):
    """The bitwise check behind the order argument: a search that picked a config with other warps
    (which changes tl.sum's tree) keeps the default instead."""
    monkeypatch.setenv("TRUESTAR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRUESTAR_TUNE", "on")
    K.clear_memory_cache()
    kern = softmax(1000)
    x = torch.randn(64, 1000, device="cuda", generator=torch.Generator(device="cuda").manual_seed(0))
    probe = K.compile(kern, "cuda", tune=False, deterministic=True)
    ref = probe(x=x)
    p, env = plan_env(probe)
    d = tb.default_config(p, env)
    other = [dict(d, num_warps=w) for w in (2, 4, 8, 16) if w != d["num_warps"]]
    bad = next(c for c in other if not torch.equal(K.compile(kern, "cuda", _config=c)(x=x), ref))
    monkeypatch.setattr(tb, "order_candidates", lambda p, env: [tb.default_config(p, env), bad])
    real = tune.search
    monkeypatch.setattr(tune, "search", lambda make, cands, budget=None: (cands[1], real(make, cands, budget)[1]))
    n = len(tune.stats.log)
    ck = K.compile(kern, "cuda", tune=True, deterministic=True)
    assert torch.equal(ck(x=x), ref)
    assert ck.launch_configs()[0][1] == d
    assert tune.stats.log[n]["order_check"] == "failed: kept the default"
    K.clear_memory_cache()


@gpu
def test_a_cached_config_that_changes_the_order_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUESTAR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRUESTAR_TUNE", "on")
    K.clear_memory_cache()
    kern = attention()
    ins = {n: torch.randn(1, 2, 300, 64, device="cuda") for n in "qkv"}
    probe = K.compile(kern, "cuda", tune=False, deterministic=True)
    probe(**ins)
    p, env = plan_env(probe)
    d = tb.default_config(p, env)
    bad = dict(d, BN=64 if d["BN"] != 64 else 32)
    real_load = tune.load
    monkeypatch.setattr(tune, "load", lambda key, check: bad if check(bad) is None else real_load(key, check))
    ck = K.compile(kern, "cuda", tune=True, deterministic=True)
    ck(**ins)
    (_, cfg, src), = ck.launch_configs()
    assert cfg["BN"] == d["BN"] and src == "order tuned"
    K.clear_memory_cache()
