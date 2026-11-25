"""Dynamic shapes (compiler/dynamic.py, docs/DYNAMIC_SHAPES.md): one compile for every batch size and
sequence length in a declared range, kernel checks per bucket, and results bitwise equal to the
static compile of the same shape. Every result is also compared with float64."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_kcache_"))

from truestar import ir  # noqa: E402
from truestar.compiler import STATS, compile_program  # noqa: E402
from truestar.compiler.dynamic import Buckets, DynamicShapeError, compile_dynamic  # noqa: E402
from truestar.kernel import compiler as kcomp  # noqa: E402


def _has_gpu():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


DEVICES = ["cpu"] + (["cuda"] if _has_gpu() else [])
g = torch.Generator().manual_seed(0)


def rnd(*shape):
    return torch.randn(*shape, generator=g)


class Prefill(nn.Module):
    """A decoder block run over a whole prompt: pre norm causal attention and a GELU MLP."""

    def __init__(self, d=32, h=4):
        super().__init__()
        self.h = h
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.up, self.down = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).split(d, -1)
        sh = lambda z: z.reshape(b, t, self.h, d // self.h).transpose(1, 2)  # noqa: E731
        a = nn.functional.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, t, d))
        return x + self.down(nn.functional.gelu(self.up(self.n2(x))))


def _models():
    torch.manual_seed(0)
    return {
        "mlp": (nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 16), nn.Softmax(-1)).eval(),
                [{0: "B"}], {"B": (1, 128)}, lambda n: (n, 16), [1, 3, 8, 17, 64, 128]),
        "encoder": (nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True).eval(),
                    [{1: "T"}], {"T": (2, 256)}, lambda n: (2, n, 32), [2, 5, 16, 33, 100, 256]),
        "prefill": (Prefill().eval(), [{1: "T"}], {"T": (2, 256)}, lambda n: (2, n, 32), [2, 7, 16, 31, 64, 200]),
    }


def _rel(a, b):
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["mlp", "encoder", "prefill"])
def test_one_compile_bitwise_static_and_float64(device, name):
    model, axes, dims, shape, sizes = _models()[name]
    dp = compile_dynamic(model, (rnd(*shape(sizes[1])),), axes=axes, dims=dims, device=device)
    rows = dp.prove([(rnd(*shape(n)),) for n in sizes])     # TS1426 on any differing element
    assert [r["differing"] for r in rows] == [0] * len(sizes)
    before = dict(STATS)
    for n in sizes:
        x = rnd(*shape(n))
        assert _rel(dp([x])[0], model.double()(x.double())) < 2e-4
        model.float()
    assert dp.stats.compiles == 1 and dict(STATS) == before


@pytest.mark.parametrize("device", DEVICES)
def test_no_recompile_across_a_sweep(device):
    model, axes, dims, shape, _ = _models()["encoder"]
    dp = compile_dynamic(model, (rnd(*shape(8)),), axes=axes, dims=dims, device=device)
    dp([rnd(*shape(8))])
    before = dict(STATS)
    for n in range(2, 42, 2):
        dp([rnd(*shape(n))])
    assert dict(STATS) == before
    assert dp.stats.buckets == [(("T", b),) for b in (2, 4, 8, 16, 32, 64)]


@pytest.mark.parametrize("device", DEVICES)
def test_bucket_verification_checks_each_binary_once_per_bucket(device):
    model, axes, dims, shape, _ = _models()["mlp"]
    per_shape = compile_dynamic(model, (rnd(*shape(4)),), axes=axes, dims=dims, device=device, verify="shape")
    per_bucket = compile_dynamic(model, (rnd(*shape(4)),), axes=axes, dims=dims, device=device, verify="bucket")
    for n in (33, 40, 47, 64):            # one power of two bucket
        x = rnd(*shape(n))
        assert torch.equal(per_shape([x])[0], per_bucket([x])[0])
    assert per_shape.stats.kernel_checks_skipped == 0
    assert per_bucket.stats.kernel_checks_skipped > 0
    assert per_bucket.stats.buckets == [(("B", 64),)]


def test_binary_identity_follows_triton_specialization():
    """Two launches share a check only when Triton would run the same binary for both."""
    class FakeParam:
        def __init__(self, name, c=False):
            self.name, self.is_constexpr = name, c

    class FakeFn:
        __name__ = "kern"
        params = [FakeParam("p_x"), FakeParam("p_y"), FakeParam("st_x_0"), FakeParam("S_B"), FakeParam("BR", True)]
        do_not_specialize = ["S_B"]

    class L:
        fn, meta, in_order, post, lib, has_ws = FakeFn(), {"BR": 16}, ["x"], None, None, False

    x = torch.zeros(64)
    L.int_args = [32, 7]
    a = kcomp._binary_id(L, {"x": x}, "triton")
    L.int_args = [32, 9]                     # an unspecialized size changes: same binary
    assert kcomp._binary_id(L, {"x": x}, "triton") == a
    L.int_args = [33, 7]                     # a stride stops dividing by 16: another binary
    assert kcomp._binary_id(L, {"x": x}, "triton") != a
    L.int_args = [32, 7]
    assert kcomp._binary_id(L, {"x": x[1:]}, "triton") != a     # a misaligned pointer
    L.meta = {"BR": 32}
    assert kcomp._binary_id(L, {"x": x}, "triton") != a         # another constexpr
    assert kcomp._binary_id(L, {"x": x}, "cpu") == ("cpu",)


def test_buckets_policy():
    b = Buckets.make("pow2", {"T": (1, 512)})
    assert [b.of("T", n) for n in (1, 2, 3, 17, 64, 65, 512)] == [1, 2, 4, 32, 64, 128, 512]
    u = Buckets.make({"T": [16, 128, 512]}, {"T": (1, 512)})
    assert [u.of("T", n) for n in (1, 16, 17, 200, 512)] == [16, 16, 128, 512, 512]
    for bad in ([16, 8, 512], [16, 128], []):
        with pytest.raises(DynamicShapeError, match="TS1422"):
            Buckets.make({"T": bad}, {"T": (1, 512)})
    with pytest.raises(DynamicShapeError, match="TS1422"):
        Buckets.make({"T": [16, 512]}, {"T": (1, None)})        # unbounded: a list cannot cover it
    with pytest.raises(DynamicShapeError, match="TS1424"):
        Buckets.make({"S": "pow2"}, {"T": (1, 512)})


def test_refusals():
    model, axes, dims, shape, _ = _models()["mlp"]
    x = (rnd(*shape(4)),)
    with pytest.raises(DynamicShapeError, match="TS1421"):
        compile_dynamic(model, x, axes=axes, dims=dims, device="cpu", deterministic=False)
    with pytest.raises(DynamicShapeError, match="TS1423"):
        compile_dynamic(model, x, axes=axes, dims=dims, device="cpu", library=["conv2d"])
    with pytest.raises(DynamicShapeError, match="TS1424"):
        compile_dynamic(model, x, axes=[{0: "B"}], dims={"T": (1, 8)}, device="cpu")
    with pytest.raises(DynamicShapeError, match="TS1424"):
        compile_dynamic(model, None, axes=axes, dims=dims, device="cpu")
    with pytest.raises(DynamicShapeError, match="TS1424"):
        compile_dynamic(ir.import_module(model, x), device="cpu")          # nothing symbolic
    dp = compile_dynamic(model, x, axes=axes, dims=dims, device="cpu")
    with pytest.raises(DynamicShapeError, match="TS1420"):
        dp([rnd(129, 16)])
    with pytest.raises(DynamicShapeError, match="TS1420"):
        dp([rnd(4, 15)])
    prog_dp = compile_dynamic(dp.program, device="cpu")
    with pytest.raises(DynamicShapeError, match="TS1427"):
        prog_dp.prove([x])
    # bitwise_static=False lifts the pins (and their refusals)
    free = compile_dynamic(model, x, axes=axes, dims=dims, device="cpu", deterministic=False, bitwise_static=False)
    x5 = rnd(5, 16)
    assert torch.equal(free([x5])[0], free([x5])[0])


def test_prove_names_a_difference():
    model, axes, dims, shape, _ = _models()["mlp"]
    dp = compile_dynamic(model, (rnd(*shape(4)),), axes=axes, dims=dims, device="cpu")
    real = dp.cp

    def off_by_one_ulp(inputs, **kw):
        out = real(inputs, **kw)
        return [torch.nextafter(out[0], torch.full_like(out[0], 2.0))] + out[1:]
    dp.cp = off_by_one_ulp
    with pytest.raises(DynamicShapeError, match="TS1426"):
        dp.prove([(rnd(*shape(6)),)])


def test_torch_export_spec_is_read_back():
    model, _, _, shape, _ = _models()["encoder"]
    T = torch.export.Dim("T", min=2, max=64)
    dp = compile_dynamic(model, (rnd(*shape(8)),), dynamic_shapes=({1: T},), device="cpu")
    assert dp.buckets.ranges == {"T": (2, 64)}
    with pytest.raises(DynamicShapeError, match="TS1420"):
        dp([rnd(*shape(65))])


def test_row_block_does_not_follow_a_static_row_count():
    """A row reduction's BR is the same for a static and a symbolic row count (it fixes the layout
    tl.sum reduces each row in): found on an L4, where batch 3 compiled static differed from the
    symbolic compile in 64 of 192 softmax outputs."""
    from truestar import kernel as K
    from truestar.kernel import ir as kir
    from truestar.kernel import triton_backend as tb

    def softmax_plan(rows):
        x = K.buffer("x", (rows, 64))
        i, j = K.axis("i", rows), K.axis("j", 64)
        return tb.plan(kir.canonicalize(K.kernel("r", [K.output("y", (i, j), K.softmax_weight(x[i, j], over=j))]))[0])
    sym = softmax_plan(K.sym("B"))
    for n in (1, 2, 3, 8, 15, 16, 64):
        assert tb.default_config(softmax_plan(n), {}) == tb.default_config(sym, {"s0": n})


def test_conv_over_symbolic_channels_is_refused():
    """The one loop shape choice that reads whether a size is symbolic (a conv's tap loop) is refused
    when the declared range reaches below the static threshold (TS1425)."""
    from types import SimpleNamespace as NS

    from truestar import kernel as K
    from truestar.compiler import dynamic
    from truestar.kernel import triton_conv

    chain = (K.axis("c", K.sym("C")), K.axis("u", 3), K.axis("v", 3))
    plan = triton_conv.ConvPlan(None, chain, None, None, (), (), ())
    step = NS(name="conv0", ck=NS(art=NS(plan=plan)))
    cp = NS(plan=NS(all_steps=lambda: [(None, step)]))
    with pytest.raises(DynamicShapeError, match="TS1425"):
        dynamic._refuse_order_changes(cp, {"C": (3, 64)})
    dynamic._refuse_order_changes(cp, {"C": (16, 64)})         # every count takes the tap loop statically too
