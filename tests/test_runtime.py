"""CUDA graph capture (truestar.runtime): replay correctness, buckets, refusals, stale outputs.

Every replay is compared bitwise with the eager launches of the same function on the same inputs.
"""

import os
import tempfile

import pytest
import torch

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_rtcache_"))


def _has_gpu():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="needs CUDA and Triton")

if _has_gpu():
    from truestar.kernel import Graph  # noqa: E402
    from truestar import kernel as K  # noqa: E402
    from truestar.runtime import (BucketError, CaptureError, CaptureMismatchError,  # noqa: E402
                                  HostSyncError, InputMutationError, bitwise_equal, capture,
                                  timing)
    from truestar.runtime import workloads as W  # noqa: E402


def rn(*shape, seed, dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, device="cuda", generator=g).to(dtype)


@pytest.fixture(scope="module")
def mlp():
    return W.MLPStack(rows=16, hidden=128, blocks=10)


# ----------------------------------------------------------------------------- correctness


@pytest.mark.parametrize("case", ["gelu", "softmax", "layer_norm", "attention"])
def test_replay_bitwise_equal_to_eager(case):
    ck, _, ins = {"gelu": lambda: W.gelu_residual(64, 512),
                  "softmax": lambda: W.softmax(64, 1000),
                  "layer_norm": lambda: W.layer_norm(64, 768),
                  "attention": lambda: W.attention(1, 4, 128, 64)}[case]()
    cf = capture(ck, ins)
    assert cf.num_graphs == 1
    for seed in (1, 2, 3):
        new = {k: rn(*v.shape, seed=seed + 10 * i, dtype=v.dtype) for i, (k, v) in
               enumerate(ins.items())}
        assert bitwise_equal(cf(**new), ck(**new))


def test_fifty_kernel_chain_bitwise(mlp):
    assert mlp.kernels_per_step == 50
    cf = capture(mlp.step, (mlp.x,))
    for seed in (1, 2):
        x = rn(16, 128, seed=seed)
        assert bitwise_equal(cf(x), mlp.step(x))


def test_torch_eager_function_captures(mlp):
    """The layer wraps any executor, here plain PyTorch code."""
    cf = capture(mlp.torch_step, (mlp.x,))
    x = rn(16, 128, seed=4)
    assert bitwise_equal(cf(x), mlp.torch_step(x))


def test_named_inputs_and_dict_outputs():
    g = Graph()
    x = g.input("x", (K.sym("M"), 256))
    cg = g.compile({"s": g.softmax(x), "e": g.exp(x) + 1.0}, "cuda")
    cf = capture(cg, {"x": rn(32, 256, seed=0)})
    x2 = rn(32, 256, seed=5)
    out, ref = cf(x=x2), cg(x=x2)
    assert set(out) == {"s", "e"}
    assert all(bitwise_equal(out[k], ref[k]) for k in ref)
    assert bitwise_equal(cf(x2)["s"], ref["s"])  # positional also works for named inputs


def test_no_stale_outputs_after_input_change():
    ck, _, ins = W.layer_norm(32, 256)
    cf = capture(ck, ins)
    a = {"x": rn(32, 256, seed=1), "w": ins["w"], "b": ins["b"]}
    b = {"x": rn(32, 256, seed=2), "w": rn(256, seed=3), "b": rn(256, seed=4)}
    ya = cf(**a)
    yb = cf(**b)
    assert bitwise_equal(yb, ck(**b))
    assert bitwise_equal(ya, ck(**a)), "an earlier output changed after the next call"
    assert not bitwise_equal(ya, yb)
    a["x"].mul_(2.0)  # in place change of a caller tensor must be seen by the next call
    assert bitwise_equal(cf(**a), ck(**a))


def test_static_outputs_are_reused_buffers():
    ck, _, ins = W.softmax(16, 128)
    cf = capture(ck, ins, outputs="static")
    y1 = cf(x=rn(16, 128, seed=1))
    x2 = rn(16, 128, seed=2)
    y2 = cf(x=x2)
    assert y1.data_ptr() == y2.data_ptr()
    assert bitwise_equal(y1, ck(x=x2)), "static mode returns the graph buffer, as documented"


def test_passing_static_inputs_skips_the_copy():
    ck, _, ins = W.softmax(16, 128)
    cf = capture(ck, ins, outputs="static")
    (sx,) = cf.static_inputs()
    sx.copy_(rn(16, 128, seed=9))
    assert bitwise_equal(cf(x=sx), ck(x=sx.clone()))


def test_inputs_bound_by_address():
    """Weights bound by address are never copied; in place updates are seen, other tensors refused."""
    ck, _, ins = W.layer_norm(32, 256)
    w, b = ins["w"].clone(), ins["b"].clone()
    cf = capture(ck, {"x": ins["x"], "w": w, "b": b}, static=["w", "b"],
                 dynamic={"x": 0}, buckets=[32, 64]).capture_all()
    x = rn(64, 256, seed=3)
    assert bitwise_equal(cf(x=x, w=w, b=b), ck(x=x, w=w, b=b))
    w.mul_(0.5)
    assert bitwise_equal(cf(x=x, w=w, b=b), ck(x=x, w=w, b=b))
    with pytest.raises(BucketError, match="bound by address"):
        cf(x=x, w=w.clone(), b=b)
    with pytest.raises(ValueError, match="both dynamic and static"):
        capture(ck, ins, static=["x"], dynamic={"x": 0}, buckets=[32])


def test_several_inputs_and_outputs_change_together():
    """Batched input and output copies: every input reaches the graph, every output is fresh."""
    def f(a, b, c):
        return a * b, b + c, c - a
    xs = [rn(16, 32, seed=i) for i in range(3)]
    cf = capture(f, xs)
    ys = [rn(16, 32, seed=10 + i) for i in range(3)]
    o1 = cf(*ys)
    o2 = cf(*xs)
    assert isinstance(o1, tuple) and len(o1) == 3
    for got, want in zip(o1, f(*ys)):
        assert bitwise_equal(got, want)
    for got, want in zip(o2, f(*xs)):
        assert bitwise_equal(got, want)


# ----------------------------------------------------------------------------- buckets


def test_exact_buckets_one_graph_each(mlp):
    cf = capture(mlp.step, (mlp.x,), dynamic={0: 0}, buckets=[8, 16, 32])
    assert cf.num_graphs == 1  # the example's size
    for n in (8, 32, 16, 8):
        x = rn(n, 128, seed=n)
        assert bitwise_equal(cf(x), mlp.step(x))
    assert cf.num_graphs == 3
    mem = cf.memory()
    assert set(mem) == {8, 16, 32} and all(m["allocated"] > 0 for m in mem.values())
    with pytest.raises(BucketError, match="not one of the captured buckets"):
        cf(rn(12, 128, seed=0))


def test_padded_buckets_bitwise(mlp):
    """Sizes of 16 rows and up: the tile kernels' block shape no longer depends on the row count."""
    cf = capture(mlp.step, (rn(16, 128, seed=0),), dynamic={0: 0}, buckets=[16, 32, 64], pad=True)
    for n in (17, 32, 20, 40, 64, 17):
        x = rn(n, 128, seed=100 + n)
        y = cf(x)
        assert y.shape == (n, 128)
        assert bitwise_equal(y, mlp.step(x))
    assert cf.num_graphs == 3
    with pytest.raises(BucketError, match="outside the buckets"):
        cf(rn(65, 128, seed=0))


def test_padded_elementwise_any_size():
    ck, _, ins = W.gelu_residual(8, 256)
    cf = capture(ck, ins, dynamic={"x": 0, "r": 0}, buckets=[8, 64], pad=True)
    for n in (1, 3, 8, 9, 63):
        x, r = rn(n, 256, seed=n), rn(n, 256, seed=n + 1)
        assert bitwise_equal(cf(x=x, r=r), ck(x=x, r=r))


def test_padding_never_returns_a_different_answer(mlp):
    """Below 16 rows the layer norm block shape follows the row count, which changes its reduction
    order. Padding then either matches bitwise or is refused; it never answers differently."""
    cf = capture(mlp.step, (rn(16, 128, seed=0),), dynamic={0: 0}, buckets=[16], pad=True)
    for n in range(1, 16):
        x = rn(n, 128, seed=200 + n)
        try:
            y = cf(x)
        except CaptureMismatchError:
            with pytest.raises(CaptureMismatchError):  # stays refused, never cached as checked
                cf(x)
            continue
        assert bitwise_equal(y, mlp.step(x))


def test_capture_all_largest_first():
    ck, _, ins = W.gelu_residual(8, 256)
    cf = capture(ck, ins, dynamic={"x": 0, "r": 0}, buckets=[8, 16, 64]).capture_all()
    assert cf.num_graphs == 3
    x, r = rn(64, 256, seed=1), rn(64, 256, seed=2)
    assert bitwise_equal(cf(x=x, r=r), ck(x=x, r=r))


def test_padding_refused_when_rows_interact():
    """softmax over the padded dim: padding changes the answer, so the first call refuses."""
    g = Graph()
    x = g.input("x", (K.sym("M"), 64))
    ck = g.compile({"y": g.softmax(x, axis=0)}, "cuda").kernels[0][1]
    cf = capture(ck, {"x": rn(8, 64, seed=0)}, dynamic={"x": 0}, buckets=[8], pad=True)
    with pytest.raises(CaptureMismatchError, match="not independent along the padded dim"):
        cf(x=rn(5, 64, seed=1))


def test_shape_dtype_and_device_changes_refused():
    ck, _, ins = W.layer_norm(32, 256)
    cf = capture(ck, ins, dynamic={"x": 0}, buckets=[32])
    with pytest.raises(BucketError, match="shape"):
        cf(x=rn(32, 256, seed=0), w=rn(128, seed=0), b=ins["b"])
    with pytest.raises(BucketError, match="float16"):
        cf(x=rn(32, 256, seed=0, dtype=torch.float16), w=ins["w"], b=ins["b"])
    with pytest.raises(BucketError, match="dynamic dims disagree"):
        cf2 = capture(lambda a, b: a + b, (rn(8, 64, seed=0), rn(8, 64, seed=1)),
                      dynamic={0: 0, 1: 0}, buckets=[8, 16])
        cf2(rn(8, 64, seed=0), rn(16, 64, seed=1))


# ----------------------------------------------------------------------------- refusals


def test_host_sync_refused():
    def f(x):
        return x * (x.sum().item() + 1.0)
    with pytest.raises(HostSyncError, match="synchronizes"):
        capture(f, (rn(8, 8, seed=0),))


def test_data_dependent_shape_refused():
    def f(x):
        return x[x > 0]
    with pytest.raises(HostSyncError):
        capture(f, (rn(8, 8, seed=0),))


def test_verify_all_refused(monkeypatch):
    ck, _, ins = W.softmax(16, 96)
    monkeypatch.setenv("TRUESTAR_VERIFY", "all")
    with pytest.raises(HostSyncError, match="TRUESTAR_VERIFY=all"):
        capture(ck, ins)


def test_cpu_and_non_tensor_inputs_refused():
    with pytest.raises(CaptureError, match="on cpu"):
        capture(lambda x: x + 1, (torch.ones(4),))
    with pytest.raises(CaptureError, match="functools.partial"):
        capture(lambda x, s: x * s, (rn(4, seed=0), 2.0))
    with pytest.raises(CaptureError, match="must return a tensor"):
        capture(lambda x: 3, (rn(4, seed=0),))


def test_input_mutation_refused():
    def f(x):
        x.add_(1.0)
        return x * 2
    with pytest.raises(InputMutationError, match="writes to input"):
        capture(f, (rn(8, seed=0),))


def test_frozen_host_state_detected():
    """A function whose result depends on Python state that changes per call cannot be frozen."""
    state = {"n": 0}

    def f(x):
        state["n"] += 1
        return x * float(state["n"])
    with pytest.raises(CaptureMismatchError, match="differs from the eager run"):
        capture(f, (rn(8, seed=0),))


def test_bad_arguments():
    x = rn(8, 4, seed=0)
    with pytest.raises(ValueError, match="need buckets"):
        capture(lambda x: x + 1, (x,), dynamic={0: 0})
    with pytest.raises(ValueError, match="without dynamic"):
        capture(lambda x: x + 1, (x,), buckets=[8])
    with pytest.raises(ValueError, match="warmup"):
        capture(lambda x: x + 1, (x,), warmup=1)


# ----------------------------------------------------------------------------- timing helpers


def test_timing_helpers_report_positive_numbers():
    ck, _, ins = W.softmax(64, 512)
    hw = timing.host_and_wall(lambda: ck(**ins), n=20, repeats=3, spin=0.0)
    dev = timing.device_time(lambda: ck(**ins), k=5, repeats=3, spin=0.0)
    assert hw["host_us"] > 0 and hw["wall_us"] >= hw["host_us"] * 0.5 and dev > 0
