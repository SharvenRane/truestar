"""truestar.nn: every layer eagerly, compiled and differentiated, against torch.nn in float64
(docs/NN.md). Also: keyed initialization, explicit state, training mode as a compile time value,
KV cache decoding, and errors that name the leaf."""

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F

import truestar
from truestar import Tensor, nn, ops

F64 = torch.float64
TOL = 1e-12


def rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def tree_map(fn, t):
    return {k: tree_map(fn, v) for k, v in t.items()} if isinstance(t, dict) else fn(t)


def leaves(t, prefix=""):
    if isinstance(t, dict):
        for k, v in t.items():
            yield from leaves(v, f"{prefix}{k}.")
    else:
        yield prefix[:-1], t


def grads_vs_torch(fn_native, fn_torch, p, *args):
    """Value and gradient of a native scalar function against torch.autograd on the same params.
    Each leaf's error is relative to that leaf's largest gradient entry, floored at 1e-6 of the
    largest entry of any leaf: a gradient that is zero in exact math (attention's key bias, which
    softmax cancels) is rounding noise on both sides and is measured against the model's scale."""
    val, g = truestar.grad(fn_native, wrt="p")(p, *args)
    pt = tree_map(lambda t: t.detach().clone().requires_grad_(), p)
    lt = fn_torch(pt, *args)
    lt.backward()
    worst = rel(torch.as_tensor(val), lt.detach())
    ours, theirs = dict(leaves(g)), dict(leaves(pt))
    assert set(ours) == set(theirs)
    scale = max(float(t.grad.abs().max()) for t in theirs.values())
    for path, a in ours.items():
        b = theirs[path].grad
        worst = max(worst, float((a - b).abs().max()) / max(float(b.abs().max()), 1e-6 * scale))
    return worst


# ----------------------------------------------------------------------------- dense layers


def test_linear_eager_compiled_and_gradient_match_torch():
    lin = nn.Linear(6, 4, dtype="f64")
    p = lin.init(seed=1)
    x = torch.randn(5, 6, dtype=F64)

    @truestar.native
    def f(p: lin.spec, x: Tensor["B", 6, "f64"]) -> Tensor["f64"]:
        return ops.sum(ops.tanh(lin(p, x)))

    ref = torch.tanh(F.linear(x, p["w"], p["b"])).sum()
    assert rel(f(p, x), ref) < TOL
    assert rel(lin(p, x), F.linear(x, p["w"], p["b"])) == 0.0          # eager is torch itself
    assert grads_vs_torch(f, lambda q, x: torch.tanh(F.linear(x, q["w"], q["b"])).sum(), p, x) < TOL
    assert set(nn.Linear(6, 4, bias=False).spec) == {"w"}


def test_embedding_any_id_shape_and_gradient():
    emb = nn.Embedding(10, 3, dtype="f64")
    p = emb.init(seed=2)
    ids = torch.tensor([[1, 4, 4], [9, 0, 1]])

    @truestar.native
    def f(p: emb.spec, ids: Tensor["B", "T", "i64"]) -> Tensor["f64"]:
        e = emb(p, ids)
        return ops.sum(e * e * e)

    assert rel(f(p, ids), (F.embedding(ids, p["w"]) ** 3).sum()) < TOL
    assert grads_vs_torch(f, lambda q, ids: (F.embedding(ids, q["w"]) ** 3).sum(), p, ids) < TOL


# ----------------------------------------------------------------------------- convolution, pools, norms


def test_cnn_layers_forward_and_gradient_match_torch():
    net = nn.Sequential(nn.Conv2d(2, 4, 3, stride=1, padding=1, dtype="f64"), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(4, 4, 3, padding=1, groups=2, dtype="f64"), nn.Activation("silu"), nn.AvgPool2d(2),
                        nn.Flatten(), nn.Linear(16, 3, dtype="f64"))
    p = net.init(seed=3)
    x = torch.randn(3, 2, 8, 8, dtype=F64)

    def ref(q, x):
        h = F.max_pool2d(F.relu(F.conv2d(x, q["0"]["w"], q["0"]["b"], padding=1)), 2)
        h = F.avg_pool2d(F.silu(F.conv2d(h, q["3"]["w"], q["3"]["b"], padding=1, groups=2)), 2)
        return F.linear(h.flatten(1), q["7"]["w"], q["7"]["b"])

    @truestar.native
    def f(p: net.spec, x: Tensor["B", 2, 8, 8, "f64"]) -> Tensor["f64"]:
        y = net(p, x)
        return ops.sum(y * y)

    assert rel(net(p, x), ref(p, x)) < TOL
    assert grads_vs_torch(f, lambda q, x: (ref(q, x) ** 2).sum(), p, x) < TOL
    assert list(net.spec) == ["0", "3", "7"]   # only layers with params have an entry


def test_global_avg_pool_layer_norm_rms_norm():
    ln, rms, gap = nn.LayerNorm(5, dtype="f64"), nn.RMSNorm(5, dtype="f64"), nn.GlobalAvgPool()
    p = {"ln": ln.init(1), "rms": rms.init(2)}
    p["ln"]["w"] = p["ln"]["w"] + torch.randn(5, dtype=F64) * 0.1
    p["rms"]["w"] = p["rms"]["w"] + torch.randn(5, dtype=F64) * 0.1
    x = torch.randn(2, 5, 3, 4, dtype=F64)

    @truestar.native
    def f(p: {"ln": ln.spec, "rms": rms.spec}, x: Tensor["B", 5, 3, 4, "f64"]) -> Tensor["f64"]:
        h = gap(x)
        a = ln(p["ln"], h)
        b = rms(p["rms"], h)
        return ops.sum(a * b * a)

    def ref(q, x):
        h = x.mean((2, 3))
        a = F.layer_norm(h, [5], q["ln"]["w"], q["ln"]["b"], 1e-5)
        b = F.rms_norm(h, [5], q["rms"]["w"], 1e-6)
        return (a * b * a).sum()

    assert rel(f(p, x), ref(p, x)) < TOL
    assert grads_vs_torch(f, ref, p, x) < TOL


def test_batch_norm_state_is_explicit_and_matches_torch():
    bn = nn.BatchNorm(3, momentum=0.2, dtype="f64")
    p, s = bn.init(0), bn.init_state()
    p["w"] = torch.tensor([1.5, 0.5, 2.0], dtype=F64)
    x = torch.randn(4, 3, 5, 5, dtype=F64) * 2 + 1

    @truestar.native
    def train(p: bn.spec, s: bn.state_spec, x: Tensor["B", 3, 5, 5, "f64"]):
        return bn(p, s, x, training=True)

    @truestar.native
    def evaluate(p: bn.spec, s: bn.state_spec, x: Tensor["B", 3, 5, 5, "f64"]):
        return bn(p, s, x)

    tb = torch.nn.BatchNorm2d(3, momentum=0.2).double()
    with torch.no_grad():
        tb.weight.copy_(p["w"])
    y, s2 = train(p, s, x)
    yt = tb(x)
    assert rel(y, yt) < TOL
    assert rel(s2["mean"], tb.running_mean) < TOL and rel(s2["var"], tb.running_var) < TOL
    assert torch.equal(s["mean"], torch.zeros(3))                     # the input state is unchanged
    tb.eval()
    ye, s3 = evaluate(p, s2, x)
    assert rel(ye, tb(x)) < TOL and s3["mean"] is not None
    # training is a compile time value: the eval program has no batch statistics in it
    assert train.program.op_counts().get("batch_norm", 0) == 0
    assert evaluate.program.op_counts().get("batch_norm") == 1


# ----------------------------------------------------------------------------- dropout and training mode


def test_dropout_is_keyed_and_absent_from_the_eval_program():
    drop = nn.Dropout(0.5)

    @truestar.native(seed=7)
    def f(x: Tensor["B", 64, "f64"], training: bool = False) -> Tensor["B", 64, "f64"]:
        return drop(x, training=training)

    x = torch.ones(3, 64, dtype=F64)
    assert torch.equal(f(x), x)
    assert f.specialize(training=False).program.op_counts().get("random", 0) == 0
    f.calls = 0
    a = f(x, training=True)
    f.calls = 0
    b = f(x, training=True)
    assert torch.equal(a, b)                                          # same call number, same mask
    assert set(a.unique().tolist()) == {0.0, 2.0}
    c = f(x, training=True)                                           # next call, new mask
    assert not torch.equal(a, c)
    # eager: the key must be given, and the same key gives the same bits as truestar.rng
    with pytest.raises(TypeError, match="key="):
        drop(x, training=True)
    e1, e2 = drop(x, training=True, key=11), drop(x, training=True, key=11)
    assert torch.equal(e1, e2)


# ----------------------------------------------------------------------------- attention


def _torch_mha(q_, x, heads, kv_heads, causal, mask=None):
    b, t, d = x.shape
    hd = d // heads
    q = F.linear(x, q_["q"]["w"], q_["q"]["b"]).view(b, t, heads, hd).transpose(1, 2)
    k = F.linear(x, q_["k"]["w"], q_["k"]["b"]).view(b, t, kv_heads, hd).transpose(1, 2)
    v = F.linear(x, q_["v"]["w"], q_["v"]["b"]).view(b, t, kv_heads, hd).transpose(1, 2)
    keep = None
    if causal:
        keep = torch.ones(t, t, dtype=torch.bool).tril()
    if mask is not None:
        m = mask[:, None, None, :]
        keep = m if keep is None else keep & m
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep, enable_gqa=kv_heads != heads)
    return F.linear(a.transpose(1, 2).reshape(b, t, d), q_["o"]["w"], q_["o"]["b"])


@pytest.mark.parametrize("kv_heads,causal,padded", [(None, True, False), (2, True, True), (1, False, True)])
def test_attention_masks_and_grouped_heads_match_torch(kv_heads, causal, padded):
    att = nn.MultiHeadAttention(16, 4, kv_heads=kv_heads, causal=causal, dtype="f64")
    p = att.init(seed=5)
    x = torch.randn(2, 6, 16, dtype=F64)
    mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])

    if padded:
        @truestar.native
        def f(p: att.spec, x: Tensor["B", "T", 16, "f64"], m: Tensor["B", "T", "bool"]) -> Tensor["f64"]:
            y = att(p, x, mask=m)
            return ops.sum(y * y)
        args = (x, mask)
    else:
        @truestar.native
        def f(p: att.spec, x: Tensor["B", "T", 16, "f64"]) -> Tensor["f64"]:
            y = att(p, x)
            return ops.sum(y * y)
        args = (x,)
    kvh = att.n_kv
    ref = lambda q, x, m=None: (_torch_mha(q, x, 4, kvh, causal, m) ** 2).sum()  # noqa: E731
    assert rel(f(p, *args), ref(p, *args)) < TOL
    assert grads_vs_torch(f, ref, p, *args) < 1e-10


def test_kv_cache_decode_equals_the_full_forward_with_one_compile():
    blk = nn.TransformerBlock(16, 4, kv_heads=2, norm="rms", act="silu", dtype="f64")
    p = blk.init(seed=6)
    x = torch.randn(2, 9, 16, dtype=F64)

    @truestar.native
    def full(p: blk.spec, x: Tensor["B", "T", 16, "f64"]) -> Tensor["B", "T", 16, "f64"]:
        return blk(p, x)

    @truestar.native
    def step(p: blk.spec, x: Tensor["B", "T", 16, "f64"], pos: Tensor["T", "i64"], cache: blk.attn.cache_spec("B", "S")):
        return blk(p, x, cache=cache, pos=pos)

    want = full(p, x)
    cache = blk.attn.init_cache(2, 12)
    cache = tree_map(lambda t: t.double(), cache)
    y0, cache = step(p, x[:, :5], torch.arange(5), cache)
    outs = [y0]
    for t in range(5, 9):
        y, cache = step(p, x[:, t:t + 1], torch.tensor([t]), cache)
        outs.append(y)
    assert rel(torch.cat(outs, 1), want) < TOL
    assert step.compile_count == 1   # prompt and every decode step: one program, any capacity


# ----------------------------------------------------------------------------- initialization


def test_init_is_keyed_by_path_not_by_construction_order():
    a = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    p1, p2 = a.init(seed=0), a.init(seed=0)
    assert all(torch.equal(x, y) for (_, x), (_, y) in zip(leaves(p1), leaves(p2)))
    assert not torch.equal(a.init(seed=1)["0"]["w"], p1["0"]["w"])
    # a layer inserted elsewhere does not change this layer's draw (its path "0" is the same)
    b = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8), nn.Linear(8, 2))
    assert torch.equal(b.init(seed=0)["0"]["w"], p1["0"]["w"])
    # the draw is truestar.rng at fold(seed, path): reproducible from the key alone, any device
    from truestar import rng
    key = nn.param_key(0, "0.w")
    u = rng.generate("uniform", (8, 4), torch.float32, key, "0.w", backend="python")
    assert torch.equal(((u * 2 - 1) * (1 / math.sqrt(4))), p1["0"]["w"])
    # scale: torch.nn.Linear's default bound 1/sqrt(fan_in), and normal std for embeddings
    w = nn.Linear(256, 512).init(3)["w"]
    assert float(w.abs().max()) <= 1 / 16 and float(w.abs().max()) > 0.99 / 16
    e = nn.Embedding(1000, 64, std=0.02).init(3)["w"]
    assert abs(float(e.std()) - 0.02) < 0.001 and abs(float(e.mean())) < 0.001


def test_layers_are_immutable_configs_and_hashable():
    lin = nn.Linear(3, 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        lin.d_out = 5
    assert lin == nn.Linear(3, 2) and hash(lin) == hash(nn.Linear(3, 2))
    blk = nn.TransformerBlock(8, 2)
    assert blk.attn is blk.parts["attn"] and blk.num_params() == sum(t.numel() for _, t in leaves(blk.init(0)))
    with pytest.raises(ValueError, match="multiple of heads"):
        nn.MultiHeadAttention(10, 3)


def test_wrong_parameter_tree_names_the_leaf():
    net = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))

    @truestar.native
    def f(p: net.spec, x: Tensor["B", 4]) -> Tensor["B", 2]:
        return net(p, x)

    p = net.init(0)
    bad = {"0": p["0"], "1": {"w": p["1"]["w"], "bias": p["1"]["b"]}}
    with pytest.raises(truestar.frontend.InputError, match=r"p\['1'\].*missing key 'b'"):
        f(bad, torch.randn(2, 4))
    with pytest.raises(truestar.frontend.InputError, match=r"p\['0'\]\['w'\]"):
        f({"0": {"w": p["0"]["w"].T.contiguous(), "b": p["0"]["b"]}, "1": p["1"]}, torch.randn(2, 4))


def test_a_shape_bug_inside_a_layer_is_reported_at_decoration():
    net = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 2))   # second layer expects 4 inputs, gets 3
    with pytest.raises(truestar.frontend.FrontendShapeError) as ei:
        @truestar.native
        def f(p: net.spec, x: Tensor["B", 4]) -> Tensor["B", 2]:
            return net(p, x)
    assert "linear" in str(ei.value) and "layers.py" in str(ei.value)


# ----------------------------------------------------------------------------- ops added for the library


def test_expm1_is_accurate_near_zero_and_differentiable():
    x = torch.tensor([-30.0, -1.0, -0.5, -0.49, -1e-3, -1e-9, -1e-300, 0.0, 1e-12, 1e-4, 0.3, 0.5, 0.51, 2.0, 40.0],
                     dtype=F64)

    @truestar.native
    def f(x: Tensor["N", "f64"]) -> Tensor["N", "f64"]:
        return ops.expm1(x)

    y = f(x)
    want = torch.expm1(x)
    r = ((y - want).abs() / want.abs().clamp_min(1e-308))
    assert float(r.max()) < 4e-16, r
    g = truestar.grad(_expm1_sum, wrt="x")(x)[1]
    assert rel(g, torch.exp(x)) < 1e-15


@truestar.native
def _expm1_sum(x: Tensor["N", "f64"]) -> Tensor["f64"]:
    return ops.sum(ops.expm1(x))


def test_cross_entropy_matches_torch_with_ignore_index_and_smoothing():
    logits = torch.randn(3, 5, 7, dtype=F64) * 3
    y = torch.randint(0, 7, (3, 5))
    y[0, 1] = -100
    y[2, 4] = -100

    @truestar.native
    def f(z: Tensor["B", "T", 7, "f64"], y: Tensor["B", "T", "i64"]) -> Tensor["f64"]:
        return nn.cross_entropy(z, y, ignore_index=-100, label_smoothing=0.1)

    want = F.cross_entropy(logits.reshape(-1, 7), y.reshape(-1), ignore_index=-100, label_smoothing=0.1)
    assert rel(f(logits, y), want) < TOL
    assert rel(nn.cross_entropy(logits, y, ignore_index=-100, label_smoothing=0.1), want) < TOL
    zt = logits.clone().requires_grad_()
    F.cross_entropy(zt.reshape(-1, 7), y.reshape(-1), ignore_index=-100, label_smoothing=0.1).backward()
    _, g = truestar.grad(f, wrt="z")(logits, y)
    assert rel(g, zt.grad) < TOL
    big = torch.tensor([[1e4, -1e4, 0.0]], dtype=F64)

    @truestar.native
    def h(z: Tensor["B", 3, "f64"], y: Tensor["B", "i64"]) -> Tensor["f64"]:
        return nn.cross_entropy(z, y)

    assert math.isfinite(float(h(big, torch.tensor([1]))))   # 2e4, not inf: log_softmax is one stable op
