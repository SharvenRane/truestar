"""Importer changes from the text, retrieval, speech and vision language sweep (ir/importer_text.py,
docs/MODEL_COVERAGE.md): the legacy empty cat of the transformers KV cache, weight norm, sign and
ceil, the refusal of a runtime check on an .item(), and the model line named when torch.export
fails on a data dependent size. Plain torch modules, no transformers needed."""

import pytest
import torch
from torch import nn

from truestar import ir
from truestar.ir.autodiff import grad
from truestar.ir.core import IRError, UnsupportedOpError


def _rel(got, want):
    scale = want.double().abs().max().item() or 1.0
    return (got.double() - want.double()).abs().max().item() / scale


def _check_value_and_grad(model, x, wrt_input=True):
    """Import the float64 model, compare the value, and the input or weight gradient against
    torch.autograd under a random cotangent."""
    model = model.double().eval()
    x = x.double()
    prog = ir.import_module(model, (x,))
    want = model(x)
    got = ir.run(prog, x)
    got = got[0] if isinstance(got, (list, tuple)) else got
    assert got.shape == want.shape
    assert _rel(got, want) < 1e-12
    cot = torch.randn(want.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    sd = model.state_dict(keep_vars=True)
    if wrt_input:
        names = [prog.inputs[0].name]
        leaf = x.clone().requires_grad_(True)
        model(leaf).backward(cot)
        wants = [leaf.grad]
    else:
        names = [v.name for v in prog.params if v.name in sd and isinstance(sd[v.name], nn.Parameter)]
        model.zero_grad(set_to_none=True)
        model(x).backward(cot)
        wants = [sd[n].grad for n in names]
    assert names
    gs = ir.run(grad(prog, wrt=names, of=[0], seed="input"), x, cot)[len(prog.outputs):]
    for n, g, w in zip(names, gs, wants):
        w = torch.zeros_like(g) if w is None else w
        assert _rel(g, w) < 1e-12 or w.abs().max() == g.abs().max() == 0, n
    return prog


class _CacheCat(nn.Module):
    """The transformers DynamicLayer pattern: an empty one dimensional tensor, then the first keys cat onto it."""

    def forward(self, k):
        empty = torch.tensor([], dtype=k.dtype, device=k.device)
        keys = torch.cat([empty, k], dim=-2)
        keys = torch.cat([keys, k * 2.0], dim=-2)
        return keys * 3.0


def test_cat_skips_a_legacy_empty_tensor_like_torch():
    """Before: ShapeError 'dim -2 is out of range for rank 1' (every encoder decoder and VLM decode
    step of the sweep)."""
    x = torch.randn(2, 4, 3, 8, generator=torch.Generator().manual_seed(0))
    prog = _check_value_and_grad(_CacheCat(), x)
    assert prog.outputs[0].type.shape[-2] == 6


def test_cat_of_only_legacy_empties_stays_a_plain_cat():
    class M(nn.Module):
        def forward(self, x):
            e = torch.tensor([], dtype=x.dtype)
            return torch.cat([e, e], dim=0).sum() + x

    x = torch.randn(3, dtype=torch.float64)
    prog = ir.import_module(M().double(), (x,))
    out = ir.run(prog, x)
    assert torch.equal(out[0] if isinstance(out, (list, tuple)) else out, x)


@pytest.mark.parametrize("dim", [0, 2, None])
def test_weight_norm_value_and_weight_gradients(dim):
    """wav2vec2 and HuBERT reparametrize the convolutional position embedding with weight norm over
    dim 2. Before: unsupported op aten._weight_norm.default."""
    torch.manual_seed(0)
    conv = nn.Conv1d(8, 8, kernel_size=5, padding=2, groups=2)
    conv = nn.utils.parametrizations.weight_norm(conv, name="weight", dim=dim)
    x = torch.randn(2, 8, 11, generator=torch.Generator().manual_seed(2))
    _check_value_and_grad(conv, x, wrt_input=False)
    _check_value_and_grad(conv, x, wrt_input=True)


class _Buckets(nn.Module):
    """DeBERTa's log bucketed relative positions: sign of an integer distance, ceil of a log."""

    def forward(self, x):
        n = x.shape[-1]
        rel = torch.arange(n)[:, None] - torch.arange(n)[None, :]
        s = torch.sign(rel)
        mid = 2
        a = torch.where((rel < mid) & (rel > -mid), torch.tensor(mid - 1).type_as(rel), torch.abs(rel))
        logp = torch.ceil(torch.log(a / mid) / torch.log(torch.tensor((8 - 1) / mid)) * (mid - 1)) + mid
        bucket = torch.where(a <= mid, rel.type_as(logp), logp * s)
        return x @ (bucket.to(x.dtype) + torch.sign(x).transpose(0, 1) @ torch.ceil(x * 3.0))


def test_sign_and_ceil_match_torch_with_zero_gradient():
    x = torch.randn(7, 7, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    x[0, 0], x[1, 1], x[2, 2] = 0.0, -0.0, -0.25
    _check_value_and_grad(_Buckets(), x)


def test_sign_and_ceil_edge_values_bit_for_bit():
    class M(nn.Module):
        def forward(self, x):
            return torch.sign(x), torch.ceil(x)

    x = torch.tensor([-0.0, 0.0, -2.5, 3.0, -0.5, 0.5, 2.0 ** 53 + 2, float("nan"), float("inf"), -float("inf")],
                     dtype=torch.float64)
    prog = ir.import_module(M(), (x,))
    for g, w in zip(ir.run(prog, x), M()(x)):
        assert torch.equal(torch.signbit(g), torch.signbit(w))
        assert torch.equal(torch.isnan(g), torch.isnan(w))
        ok = ~torch.isnan(w)
        assert torch.equal(g[ok], w[ok])


class _Checked(nn.Module):
    """transformers' placeholder check: a count read with .item() and asserted with torch._check."""

    def forward(self, ids, feats):
        n = (ids == 7).sum()
        torch._check(n.item() * feats.shape[-1] == feats.numel())
        return feats * 2.0


def test_runtime_check_on_an_item_is_refused_with_its_line():
    """Before: three unrelated 'unsupported op' lines (aten.item, cast_symbool_to_symint_guardless,
    eq). After: one problem naming the check, the user's line and the switch transformers documents."""
    ids = torch.tensor([[7, 7, 1, 2]])
    feats = torch.randn(1, 2, 3, dtype=torch.float64)
    with pytest.raises(UnsupportedOpError) as info:
        ir.import_module(_Checked(), (ids, feats))
    msg = str(info.value)
    assert "runtime check on a value read from a tensor" in msg
    assert "test_importer_text.py" in msg and "TRANSFORMERS_DISABLE_TORCH_CHECK=1" in msg
    assert "cast_symbool_to_symint_guardless" not in msg and "1 problem(s)" in msg


class _DataDependent(nn.Module):
    def forward(self, x):
        if (x > 0).sum().item() > 2:
            return x * 2.0
        return x


def test_export_failure_on_a_data_dependent_branch_names_the_model_line():
    x = torch.randn(4, dtype=torch.float64)
    with pytest.raises(UnsupportedOpError) as info:
        ir.import_module(_DataDependent(), (x,))
    msg = str(info.value)
    assert msg.startswith("torch.export cannot trace _DataDependent: test_importer_text.py:")
    assert "if (x > 0).sum().item() > 2:" in msg and "nothing was imported" in msg
    assert type(info.value.__cause__).__name__ == "GuardOnDataDependentSymNode"


class _Merge(nn.Module):
    """The vision language merge: projected image features scattered into the placeholder rows."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(6, 8)

    def forward(self, emb):
        ids = torch.tensor([[9, 7, 7, 3, 7], [7, 1, 7, 7, 2]])
        feats = self.proj(torch.arange(36, dtype=emb.dtype).reshape(6, 6) / 18.0 - 1.0)   # 6 placeholders
        mask = (ids == 7).unsqueeze(-1).expand_as(emb)
        return emb.masked_scatter(mask, feats) * 1.5


def test_masked_scatter_value_and_gradients():
    """Before: unsupported op aten.masked_scatter.default (LLaVA, LLaVA-NeXT, Gemma 3)."""
    torch.manual_seed(0)
    emb = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(4))
    _check_value_and_grad(_Merge(), emb, wrt_input=True)
    _check_value_and_grad(_Merge(), emb, wrt_input=False)


def test_masked_scatter_with_too_few_source_elements_raises_like_torch():
    class M(nn.Module):
        def forward(self, x):
            return x.masked_scatter(x > 0, torch.zeros(2, dtype=x.dtype))

    x = torch.tensor([1.0, 2.0, -1.0, 3.0], dtype=torch.float64)
    with pytest.raises(RuntimeError):
        M()(x)
    prog = ir.import_module(M(), (torch.tensor([1.0, -2.0, -1.0, 3.0], dtype=torch.float64),))
    with pytest.raises(IRError, match="index out of range"):     # loud, as torch's error
        ir.run(prog, x)


def test_cache_cat_compiles_on_cpu_with_its_unused_empty_constant():
    """The empty cache tensor stays an unused zero sized param; the compiler's leaf for it used to
    raise TypeError ('NoneType' object is not iterable) for every decode step of the sweep."""
    pytest.importorskip("numba")
    from truestar.compiler import compile_program

    x = torch.randn(2, 4, 3, 8, generator=torch.Generator().manual_seed(0))
    prog = ir.import_module(_CacheCat().double(), (x.double(),))
    cp = compile_program(prog, "cpu", "float32")
    out = cp([x])[0]
    assert _rel(out, _CacheCat()(x.double())) < 1e-6
    assert not cp.report.fallbacks


def test_integer_clamp_runs_on_the_device_not_the_float64_fallback():
    """DeBERTa clamps its integer relative positions; before, each clamp was a float64 CPU fallback."""
    pytest.importorskip("numba")
    from truestar.compiler import compile_program

    class M(nn.Module):
        def forward(self, x):
            n = x.shape[-1]
            rel = (torch.arange(n)[:, None] - torch.arange(n)[None, :] + 2).clamp(0, 3)
            return torch.gather(x, 1, rel.expand(x.shape[0], n, n).reshape(x.shape[0], -1)[:, :n])

    x = torch.randn(2, 5, generator=torch.Generator().manual_seed(1))
    prog = ir.import_module(M(), (x.double(),))
    cp = compile_program(prog, "cpu", "float32")
    assert torch.allclose(cp([x])[0].double(), M()(x.double()), rtol=0, atol=1e-6)
    assert not [f for f in cp.report.fallbacks if "clamp" in f["ops"]]


class _VmapMask(nn.Module):
    """A mask built with torch.vmap inside forward (transformers' masks with a custom overlay, such as
    Gemma 3's bidirectional image blocks), then combined with ``&`` while its inputs are read again."""

    def forward(self, x):
        m = torch.vmap(lambda r: r > r.mean())(x)
        k = (x > 0) & m
        return torch.where(k, x, -x) + m.to(x.dtype) * x


def test_vmap_inside_the_model_imports_through_the_aten_pass():
    """Before: one 'unsupported op' line per functorch marker (_vmap_increment_nesting, _add_batch_dim,
    lazy_load_decompositions, ...)."""
    x = torch.randn(3, 4, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    _check_value_and_grad(_VmapMask(), x)


def test_and_of_tensors_is_not_taken_for_an_in_place_op():
    """aten.__and__ ends in an underscore; the in place rule took it for a mutation of its first input
    and refused any program that read that input afterwards."""
    class M(nn.Module):
        def forward(self, x):
            a = x > 0
            b = x < 1
            return torch.where(a & b, x, 0.0 * x) + a.to(x.dtype) - (a | b).to(x.dtype)

    x = torch.randn(10, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    ep = torch.export.export(M(), (x,), strict=False)
    assert any("__and__" in str(n.target) for n in ep.graph.nodes)
    prog = ir.import_exported(ep, name="and")
    out = ir.run(prog, x)
    assert torch.equal(out[0] if isinstance(out, (list, tuple)) else out, M()(x))


def test_compiled_where_with_a_float64_min_scalar_matches_its_check():
    """T5's decoder masks its position bias with torch.finfo(dtype).min, a Python scalar: the kernel
    inlined it saturated to float32 while its check's reference used 1.8e308, so the compiled
    program was refused (ProgramMismatchError, relative error 1) although it was right."""
    pytest.importorskip("numba")
    from truestar.compiler import compile_program

    class M(nn.Module):
        def forward(self, x):
            n = x.shape[-1]
            causal = torch.arange(n)[None, :] <= torch.arange(n)[:, None]
            bias = torch.where(causal, x[:, :1] * 0.5, torch.finfo(x.dtype).min)  # T5: masked position bias
            return torch.nn.functional.scaled_dot_product_attention(x, x, x, attn_mask=bias)

    x = torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(7))
    prog = ir.import_module(M(), (x.double(),))
    out = compile_program(prog, "cpu", "float32")([x])[0]
    want = M()(x)                                   # the float32 model uses finfo(float32).min
    assert _rel(out, want) < 1e-6


def test_integer_pad_runs_on_the_device_not_the_float64_fallback():
    """Gemma 3 pads its integer image block ids with -1 and its bool image mask with False; before,
    each was a float64 CPU fallback."""
    pytest.importorskip("numba")
    from truestar.compiler import compile_program

    class M(nn.Module):
        def forward(self, x):
            ids = (x > 0).long().cumsum(-1)
            ids = torch.nn.functional.pad(ids, (0, 2), value=-1)
            img = torch.nn.functional.pad(x > 0.5, (1, 1), value=False)          # a bool pad too
            same = (ids[:, :-2] == ids[:, 2:]) & img[:, 1:-1]
            return x.sum(-1, keepdim=True) * same.to(x.dtype)

    x = torch.randn(2, 6, generator=torch.Generator().manual_seed(2))
    prog = ir.import_module(M(), (x.double(),))
    cp = compile_program(prog, "cpu", "float32")
    assert _rel(cp([x])[0], M()(x.double())) < 1e-6
    assert not [f for f in cp.report.fallbacks if "pad" in f["ops"]], cp.report.fallbacks
