"""Transformer inference speed work (P38): batch axes folded into contraction rows, cuBLAS as a
measured and verified implementation of float32 contraction kernels, KV caches updated in place
(``update=``) and CUDA graph replay of compiled programs (``cp.graphed()``).

Every speed path must give the results of the path it replaces: bitwise where the arithmetic is
the same (in place writes, graph replay), within the float64 checks otherwise (fold, cuBLAS), and a
broken implementation must be caught by the same checks.
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_kcache_"))

from truestar import ir  # noqa: E402
from truestar import kernel as K  # noqa: E402
from truestar.compiler import compile_program  # noqa: E402
from truestar.kernel import library  # noqa: E402
from truestar.kernel import ir as kir  # noqa: E402
from truestar.kernel import triton_backend as tb  # noqa: E402
from truestar.kernel.verify import KernelMismatchError  # noqa: E402


def _has_triton():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_triton(), reason="needs CUDA and Triton")


@pytest.fixture(autouse=True)
def _strict_fp32():
    prev = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    yield
    torch.backends.cuda.matmul.fp32_precision = prev


class Block(nn.Module):
    """A small decoder block with a KV cache (value semantics, as torch.export needs)."""

    def __init__(self, dim=64, heads=4, kv=2, hidden=96):
        super().__init__()
        self.h, self.kv, self.hd = heads, kv, dim // heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, kv * self.hd, bias=False)
        self.wv = nn.Linear(dim, kv * self.hd, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)

    def forward(self, x, pos, kc, vc):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        kc = kc.index_copy(2, pos, k)
        vc = vc.index_copy(2, pos, v)
        keep = torch.arange(kc.shape[2], device=x.device) <= pos[:, None]
        a = F.scaled_dot_product_attention(q, kc, vc, attn_mask=keep, enable_gqa=True)
        x = x + self.wo(a.transpose(1, 2).reshape(B, T, D))
        return x + self.w2(F.relu(self.w1(x))), kc, vc


S = 32


def _block_program():
    torch.manual_seed(0)
    m = Block().eval()
    ex = (torch.randn(2, 1, 64), torch.tensor([5]), torch.randn(2, 2, S, 16), torch.randn(2, 2, S, 16))
    Bd = torch.export.Dim("B", min=1, max=64)
    prog = ir.import_module(m, ex, dynamic_shapes=({0: Bd}, None, {0: Bd}, {0: Bd}), name="block")
    return m, prog


def _step_inputs(B, pos, seed=1):
    g = torch.Generator().manual_seed(seed)
    kc = torch.randn(B, 2, S, 16, generator=g)
    vc = torch.randn(B, 2, S, 16, generator=g)
    return [torch.randn(B, 1, 64, generator=g), torch.tensor([pos]), kc, vc]


def _cuda(xs):
    return [x.cuda() for x in xs]


# ----------------------------------------------------------------------------- fold


def _batched_linear(rows_first=True):
    """relu(x[B, T, K] @ w[N, K].T + b): the right factor has no batch axis, so it folds."""

    x = kir.buffer("x", (K.sym("B"), K.sym("T"), 48))
    w = kir.buffer("w", (40, 48))
    b = kir.buffer("b", (40,))
    i, t, n, k = K.axis("i", K.sym("B")), K.axis("t", K.sym("T")), K.axis("n", 40), K.axis("k", 48)
    e = kir.relu(kir.reduce_sum(k, x[i, t, k] * w[n, k]) + b[n])
    return kir.kernel("bl", [kir.output("y", (i, t, n), e)])


def test_batch_axes_fold_into_rows():
    kern = _batched_linear()
    ck, _, _ = kir.canonicalize(kern)
    p = tb.plan(ck)
    assert p.kind == "contraction" and len(p.fold) == 1 and p.batch == ()
    ckern = K.compile(kern, "cuda")
    w, b = torch.randn(40, 48, device="cuda"), torch.randn(40, device="cuda")
    for B, T in ((1, 1), (8, 1), (3, 5), (17, 2)):
        x = torch.randn(B, T, 48, device="cuda")
        y = ckern(x=x, w=w, b=b)                 # checked against float64 on every new shape
        ref = torch.relu(x.double() @ w.double().T + b.double())
        assert torch.allclose(y.double(), ref, rtol=1e-5, atol=1e-5)
        L = list(ckern._fast.values())[-1]
        assert L.grid[1] == 1                    # one program grid over B x T rows, not one per B


def test_attention_like_contraction_does_not_fold():

    a = kir.buffer("a", (4, 8, 16))
    bb = kir.buffer("bb", (4, 16, 8))
    i, m, n, k = K.axis("i", 4), K.axis("m", 8), K.axis("n", 8), K.axis("k", 16)
    kern = kir.kernel("bmm", [kir.output("y", (i, m, n), kir.reduce_sum(k, a[i, m, k] * bb[i, k, n]))])
    ck, _, _ = kir.canonicalize(kern)
    p = tb.plan(ck)
    assert p.fold == () and len(p.batch) == 1   # the right factor depends on the batch axis
    x, y = torch.randn(4, 8, 16, device="cuda"), torch.randn(4, 16, 8, device="cuda")
    out = K.compile(kern, "cuda")(a=x, bb=y)
    assert torch.allclose(out.double(), x.double() @ y.double(), rtol=1e-5, atol=1e-5)


# ----------------------------------------------------------------------------- cuBLAS forms


def _linear_kernel(epi):

    x = kir.buffer("x", (K.sym("M"), 64))
    w = kir.buffer("w", (48, 64))
    bias = kir.buffer("bias", (48,))
    r = kir.buffer("r", (K.sym("M"), 48))
    m, n, k = K.axis("m", K.sym("M")), K.axis("n", 48), K.axis("k", 64)
    acc = kir.reduce_sum(k, x[m, k] * w[n, k])
    e = {"plain": acc, "bias": acc + bias[n], "relu": kir.relu(acc + bias[n]), "res": r[m, n] + acc,
         "res_bias": r[m, n] + (acc + bias[n]), "gelu": kir.tanh(acc + bias[n]),
         "scaled": acc * 2.0}[epi]
    used = {"plain": ("x", "w"), "bias": ("x", "w", "bias"), "relu": ("x", "w", "bias"),
            "res": ("x", "w", "r"), "res_bias": ("x", "w", "r", "bias"), "gelu": ("x", "w", "bias"),
            "scaled": ("x", "w")}[epi]
    return kir.kernel(f"lin_{epi}", [kir.output("y", (m, n), e)]), used


@pytest.mark.parametrize("epi,ok", [("plain", True), ("bias", True), ("relu", True), ("res", True),
                                    ("res_bias", True), ("gelu", False), ("scaled", False)])
def test_library_forms(epi, ok):
    kern, used = _linear_kernel(epi)
    ck, _, _ = kir.canonicalize(kern)
    form = library.library_form(ck, tb.plan(ck))
    assert (form is not None) == ok
    if not ok:
        return
    x, w = torch.randn(33, 64, device="cuda"), torch.randn(48, 64, device="cuda")
    t = {"x": x, "w": w, "bias": torch.randn(48, device="cuda"), "r": torch.randn(33, 48, device="cuda")}
    y = K.compile(kern, "cuda", _config={"LIB": 1})(**{n: t[n] for n in used})   # verified
    ref = x.double() @ w.double().T
    if "bias" in used:
        ref = ref + t["bias"].double()
    if "r" in used:
        ref = ref + t["r"].double()
    if epi == "relu":
        ref = torch.relu(ref)
    assert torch.allclose(y.double(), ref, rtol=1e-5, atol=1e-5)


def test_a_broken_library_call_is_caught(monkeypatch):
    kern, used = _linear_kernel("bias")
    real = library.run

    def wrong(form, canon, out, mats=None):
        real(form, canon, out, mats)
        out.mul_(1.001)                       # 0.1%: far above the float32 tolerance
        return out
    monkeypatch.setattr(library, "run", wrong)
    x, w, b = torch.randn(16, 64, device="cuda"), torch.randn(48, 64, device="cuda"), torch.randn(48, device="cuda")
    with pytest.raises(KernelMismatchError):
        K.compile(kern, "cuda", _config={"LIB": 1})(x=x, w=w, bias=b)


def test_library_follows_the_precision_it_was_compiled_for():
    """A strict float32 kernel whose choice is cuBLAS runs the Triton launch while PyTorch says
    TF32: cuBLAS would round the operands to TF32, the kernel must not."""
    kern, _ = _linear_kernel("bias")
    ckern = K.compile(kern, "cuda", _config={"LIB": 1})
    x = torch.randn(64, 64, device="cuda") * 3
    w, b = torch.randn(48, 64, device="cuda"), torch.randn(48, device="cuda")
    ref = x.double() @ w.double().T + b.double()
    y = ckern(x=x, w=w, bias=b)
    L = list(ckern._fast.values())[0]
    assert L.lib is not None and L.alt is None
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    y2 = ckern(x=x, w=w, bias=b)
    assert L.alt is not None                      # the Triton launch ran
    err = float((y2.double() - ref).abs().max() / ref.abs().max())
    assert err < 1e-5                             # strict float32, not TF32 (about 1e-3 here)
    assert float((y.double() - ref).abs().max() / ref.abs().max()) < 1e-5


def test_kernel_layer_default_never_uses_the_library():
    kern, _ = _linear_kernel("bias")
    ckern = K.compile(kern, "cuda")
    x, w, b = torch.randn(512, 64, device="cuda"), torch.randn(48, 64, device="cuda"), torch.randn(48, device="cuda")
    ckern(x=x, w=w, bias=b)
    assert all(not c.get("LIB") for _, c, _ in ckern.launch_configs())


def _mlp_program():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256)).eval()
    return m, ir.import_module(m, (torch.randn(4, 256),),
                               dynamic_shapes=({0: torch.export.Dim("B", min=1, max=4096)},))


def test_compiled_program_choice_is_verified_and_deterministic_mode_never_uses_it():
    m, prog = _mlp_program()
    x = torch.randn(2048, 256)
    ref = m.double()(x.double())
    cp = compile_program(prog, "cuda", "float32")
    y = cp([x.cuda()])[0]
    assert torch.allclose(y.double().cpu(), ref, rtol=1e-4, atol=1e-5)
    cpd = compile_program(prog, "cuda", "float32", deterministic=True)
    cpd([x.cuda()])
    for _, s in cpd.plan.all_steps():
        if s.kind == "kernel":
            assert not s.ck.library
            assert all(not c.get("LIB") for _, c, _ in s.ck.launch_configs())


# ----------------------------------------------------------------------------- update= (in place KV cache)


def test_update_writes_the_cache_in_place_bitwise():
    _, prog = _block_program()
    cp = compile_program(prog, "cuda", "float32")
    xs = _cuda(_step_inputs(3, 7))
    ref = cp([t.clone() for t in xs])
    kc, vc = xs[2].clone(), xs[3].clone()
    ptr = kc.data_ptr()
    outs = cp([xs[0], xs[1], kc, vc], update={2: 1, 3: 2})
    assert cp.last_update_inplace
    assert outs[1] is kc and outs[2] is vc and kc.data_ptr() == ptr
    for a, b in zip(outs, ref):
        assert torch.equal(a, b)


def test_update_copies_when_the_output_is_not_writable_in_place():
    """Output 0 is not produced from input 0: it is computed elsewhere, then copied into it."""
    _, prog = _block_program()
    cp = compile_program(prog, "cuda", "float32")
    xs = _cuda(_step_inputs(2, 3))
    ref = cp([t.clone() for t in xs])
    x = xs[0].clone()
    outs = cp([x, xs[1], xs[2].clone(), xs[3].clone()], update={0: 0})
    assert not cp.last_update_inplace
    assert outs[0] is x and torch.equal(x, ref[0])


def test_update_rejects_bad_targets():
    _, prog = _block_program()
    cp = compile_program(prog, "cuda", "float32")
    xs = _cuda(_step_inputs(2, 3))
    with pytest.raises(TypeError):
        cp([xs[0], xs[1], xs[2].cpu(), xs[3]], update={2: 1})
    with pytest.raises(ValueError):
        cp(xs, update={2: 1, 3: 1})


# ----------------------------------------------------------------------------- graphs


def test_graphed_decode_is_bitwise_the_ordinary_call():
    _, prog = _block_program()
    cp = compile_program(prog, "cuda", "float32")
    plain = compile_program(prog, "cuda", "float32")
    g = cp.graphed(update={2: 1, 3: 2})
    for B in (1, 4):
        xs = _cuda(_step_inputs(B, 0, seed=B))
        kc, vc = xs[2].clone(), xs[3].clone()
        pk, pv = xs[2].clone(), xs[3].clone()
        for pos in range(4):
            x = torch.randn(B, 1, 64, device="cuda")
            p = torch.tensor([pos], device="cuda")
            outs = g([x, p, kc, vc])
            ref = plain([x, p, pk, pv])
            pk, pv = ref[1], ref[2]
            assert torch.equal(outs[0], ref[0]) and torch.equal(kc, pk) and torch.equal(vc, pv)
            assert outs[1] is kc
    assert g.num_graphs == 2 and g.replays == 6 and g.captures == 2


def test_graphed_bad_index_raises_and_bound_inputs_are_checked():
    from truestar.runtime.graphs import BucketError
    _, prog = _block_program()
    cp = compile_program(prog, "cuda", "float32")
    g = cp.graphed(update={2: 1, 3: 2})
    xs = _cuda(_step_inputs(2, 1))
    kc, vc = xs[2], xs[3]
    g([xs[0], xs[1], kc, vc])
    with pytest.raises(Exception, match="index_copy"):
        g([xs[0], torch.tensor([S + 3], device="cuda"), kc, vc])
    with pytest.raises(BucketError):
        g([xs[0], xs[1], kc.clone(), vc])


def test_graphed_refuses_verify_all(monkeypatch):
    from truestar.runtime.graphs import HostSyncError
    _, prog = _block_program()
    cp = compile_program(prog, "cuda", "float32", verify="all")
    with pytest.raises(HostSyncError):
        cp.graphed()(_cuda(_step_inputs(2, 1)))


def test_graphed_follows_the_tf32_setting():
    m, prog = _mlp_program()
    cp = compile_program(prog, "cuda", "float32")
    g = cp.graphed()
    x = torch.randn(64, 256, device="cuda")
    a = g([x])[0]
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    b = g([x])[0]
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    c = g([x])[0]
    assert g.num_graphs == 2 and torch.equal(a, c)
    ref = m.double().cuda()(x.double())
    assert float((a.double() - ref).abs().max() / ref.abs().max()) < 1e-5
    assert float((b.double() - ref).abs().max() / ref.abs().max()) < 1e-2


def test_a_pinned_precision_never_uses_the_library():
    """cuBLAS runs in PyTorch's current precision, so a program with allow_tf32 pinned keeps its own
    kernels: its bits must not depend on what PyTorch says (tests/test_compiler_tf32.py)."""
    _, prog = _mlp_program()
    cp = compile_program(prog, "cuda", "float32", allow_tf32=False)
    cp([torch.randn(2048, 256).cuda()])
    for _, s in cp.plan.all_steps():
        if s.kind == "kernel":
            assert not s.ck.library


def test_graphed_program_carries_the_sentinel_flags():
    """With compiled in sentinels the graph keeps its own flag buffer: a clean replay is bitwise the
    ordinary call, and a NaN in a later replay raises SentinelError instead of returning."""
    from truestar.compiler.checks import SentinelError
    _, prog = _block_program()
    cp = compile_program(prog, "cuda", "float32", sentinels="nan")
    plain = compile_program(prog, "cuda", "float32", sentinels="nan")   # same kernels (no cuBLAS with sentinels)
    g = cp.graphed(update={2: 1, 3: 2})
    xs = _cuda(_step_inputs(2, 1))
    kc, vc = xs[2].clone(), xs[3].clone()
    pk, pv = xs[2].clone(), xs[3].clone()
    for pos in range(3):
        x = torch.randn(2, 1, 64, device="cuda")
        p = torch.tensor([pos], device="cuda")
        outs = g([x, p, kc, vc])
        ref = plain([x, p, pk, pv])
        pk, pv = ref[1], ref[2]
        assert torch.equal(outs[0], ref[0]) and torch.equal(kc, pk)
    assert g.replays == 2 and g._entries and next(iter(g._entries.values())).flags is not None
    bad = torch.randn(2, 1, 64, device="cuda")
    bad[1, 0, 5] = float("nan")
    with pytest.raises(SentinelError):
        g([bad, torch.tensor([4], device="cuda"), kc, vc])
