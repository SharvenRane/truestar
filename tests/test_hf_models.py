"""Hugging Face models built from configs (random weights, nothing downloaded): truestar check, IR
import, compile and export. See docs/MODEL_COVERAGE.md and scripts/hf_coverage.py."""

import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from truestar import ir  # noqa: E402
from truestar.ir.core import UnsupportedOpError  # noqa: E402
from truestar import testing_hf  # noqa: E402
from truestar.testing_hf import MODELS, build  # noqa: E402

# the 2025 to 2026 families this transformers knows (all 28 on 5.24), minus the measured gaps
CURRENT = sorted(n for n in testing_hf.available(testing_hf.MODELS_2026) if n not in testing_hf.KNOWN_GAPS_2026)
# Mamba 2 SSD and Gated DeltaNet families lowered to linear_attention (P60): on transformers 5.24
# all five; on 4.48 none (no module function to lower), and those are refused below instead
LOWERED = sorted(testing_hf.lowered_families())
NAMES = sorted(MODELS) + CURRENT + LOWERED
REFUSED = sorted(testing_hf.available(testing_hf.REFUSED_2026))
NOT_LOWERED = sorted(set(testing_hf.available(testing_hf.LOWERED_2026)) - set(LOWERED))
CUDA = torch.cuda.is_available()
TRANSFORMERS_MAJOR = int(transformers.__version__.split(".")[0])


def _rel(got, want):
    scale = want.abs().max().item() or 1.0
    return (got.double() - want.double()).abs().max().item() / scale


FIXED = set(testing_hf.FIXED_BATCH_2026)


def _import(name, batch=2):
    model, args = build(name, batch=batch)
    m64 = copy.deepcopy(model).double()
    a64 = tuple(a.double() if a.is_floating_point() else a for a in args)
    if name in FIXED:     # a per image Python loop: one program per batch size
        return m64, ir.import_module(m64, a64)
    B = torch.export.Dim("B", min=1, max=64)
    return m64, ir.import_module(m64, a64, dynamic_shapes=({0: B},))


_PROGRAMS = {}


def _program(name):
    if name not in _PROGRAMS:
        _PROGRAMS[name] = _import(name)
    return _PROGRAMS[name]


@pytest.mark.parametrize("name", NAMES)
def test_import_matches_pytorch_float64_at_two_batch_sizes(name):
    m64, prog = _program(name)
    assert ("B" in prog.symbols) != (name in FIXED)
    for batch, seed in ((2, 1), (5, 2)):
        if name in FIXED and batch != 2:
            m64, prog = _import(name, batch)
        _, xs = build(name, batch=batch, seed=seed)
        xs = tuple(x.double() if x.is_floating_point() else x for x in xs)
        with torch.no_grad(), ir.reference.exact_float32_casts():
            want = m64(*xs)
        got = ir.run(prog, *xs)
        got = got if isinstance(got, (list, tuple)) else [got]
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert g.shape == w.shape
            assert _rel(g, w) < 1e-10, (name, batch, _rel(g, w))


@pytest.mark.parametrize("name", NAMES)
def test_export_writes_a_verified_artifact_that_runs(name, tmp_path):
    from truestar import export

    model, args = build(name, batch=2)
    path = tmp_path / f"{name}.tsx"
    if name == "gemma2" and TRANSFORMERS_MAJOR < 5:
        # transformers 4.x builds Gemma 2's embedding scale as torch.tensor(sqrt(width), dtype=the
        # model's dtype): the float32 export holds sqrt(32) rounded to float32 (1.7e-8 off), the
        # float64 model the exact value, and the 1e-9 export bar refuses the artifact (2.1e-9).
        # transformers 5 keeps it in a buffer, which the float64 copy converts, and the export passes.
        with pytest.raises(export.VerificationError, match="max rel error"):
            export.export(copy.deepcopy(model), args, output=str(path))
        return
    export.export(copy.deepcopy(model), args, output=str(path), dynamic_batch=name not in FIXED)
    runner = export.load(str(path))
    _, xs = build(name, batch=2 if name in FIXED else 3, seed=4)
    got = runner(*xs)
    got = got if isinstance(got, (list, tuple)) else [got]
    with torch.no_grad():
        want = model(*xs)
    # the artifact runs the float64 reference; the model here is float32
    assert _rel(got[0], want[0]) < 1e-4


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("name", NAMES)
def test_compiled_on_cuda_matches_the_reference_on_first_call(name):
    from truestar.compiler import compile_program

    _, prog = _program(name)
    cp = compile_program(prog, "cuda", "float32")
    if name in testing_hf.LOWERED_CUDA_GAPS:      # refused loudly by the kernel's own check, never a quiet result
        from truestar.compiler.runtime import ProgramMismatchError
        _, xs = build(name, batch=3, seed=5)
        with pytest.raises(ProgramMismatchError, match="TS1060"):
            cp(list(xs))
        return
    for batch, seed in ((3, 5), (2, 6)):
        if name in FIXED and batch != 2:
            continue
        _, xs = build(name, batch=batch, seed=seed)
        outs = cp(list(xs))             # verified against ir.run on the first call (raises otherwise)
        want = ir.run(prog, *[x.double() if x.is_floating_point() else x for x in xs])
        want = want if isinstance(want, (list, tuple)) else [want]
        assert _rel(outs[0].cpu(), want[0]) < 1e-3
    assert cp.report.kernels
    # since the op coverage work (2026-11-24) nothing runs on the float64 CPU reference
    assert not cp.report.fallbacks, [f["ops"] for f in cp.report.fallbacks]


@pytest.mark.parametrize("name", sorted(FIXED & set(CURRENT)))
def test_per_image_vision_encoders_compile_on_the_cpu_with_no_fallback(name):
    """SigLIP2 and Pixtral vision (per image sizes from constants, ir/importer_vision.py): the CPU
    compile is verified against ir.run on the first call and nothing falls back to the reference."""
    pytest.importorskip("numba")
    from truestar.compiler import compile_program

    _, prog = _program(name)
    cp = compile_program(prog, "cpu", "float32")
    _, xs = build(name, batch=2, seed=7)
    outs = cp(list(xs))
    want = ir.run(prog, *[x.double() if x.is_floating_point() else x for x in xs])
    want = want if isinstance(want, (list, tuple)) else [want]
    assert _rel(outs[0].cpu(), want[0]) < 1e-4
    assert not cp.report.fallbacks, [f["ops"] for f in cp.report.fallbacks]


@pytest.mark.parametrize("name", NAMES)
def test_one_training_step_gradient_matches_torch_autograd(name):
    """The gradient of a random projection of output 0 with respect to every trainable weight:
    truestar.ir.grad of the imported program against torch.autograd, both float64 (bar 1e-10
    relative to the largest gradient entry). Whisper's sinusoidal positions are frozen in torch
    (requires_grad False) and are left out."""
    from truestar.ir.autodiff import grad

    m64, prog = _program(name)
    _, xs = build(name, batch=2, seed=3)
    xs = tuple(x.double() if x.is_floating_point() else x for x in xs)
    # tied weights (GPT2's head, T5's and Mamba's embeddings) are one tensor stored under one name
    sd = m64.state_dict(keep_vars=True)
    params = {v.name: sd[v.name] for v in prog.params
              if v.name in sd and isinstance(sd[v.name], torch.nn.Parameter) and sd[v.name].requires_grad}
    names = list(params)
    trainable = {id(p) for p in m64.parameters() if p.requires_grad}
    assert {id(p) for p in params.values()} == trainable
    out0 = ir.run(prog, *xs)
    out0 = (out0 if isinstance(out0, (list, tuple)) else [out0])[0]
    cot = torch.randn(out0.shape, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    got = ir.run(grad(prog, wrt=names, of=[0], seed="input"), *xs, cot)[len(prog.outputs):]
    m64.zero_grad(set_to_none=True)
    with ir.reference.exact_float32_casts():
        m64(*xs)[0].backward(cot)
    # Gemma 3n at these inputs: torch's own float64 gradient has NaN entries (2 to 1024 per weight
    # in the last layer's AltUp and Laurel and the per layer inputs); Truestar's must have NaN at
    # exactly the same entries and agree everywhere else
    scale = max(params[n].grad.nan_to_num(0.0).abs().max().item() for n in names if params[n].grad is not None)
    for n, g in zip(names, got):
        want = params[n].grad
        want = torch.zeros_like(g) if want is None else want
        assert torch.equal(torch.isnan(g), torch.isnan(want)), (name, n, "NaN entries differ")
        ok = ~torch.isnan(want)
        assert (g[ok] - want[ok]).abs().max().item() / scale < 1e-10 if ok.any() else True, (name, n)


@pytest.mark.parametrize("name", ["llama", "bert"])
def test_truestar_check_is_clean(name):
    from truestar.check import check

    model, args = build(name, batch=2)
    rep = check(model, args, backends=["cpu", "cuda"] if CUDA else ["cpu"], drill_ops=False)
    assert rep.status.upper() == "OK", rep.codes


def test_float32_islands_are_real_and_kept_exact_only_in_the_comparison():
    """Llama's rotary embedding computes in float32 inside a float64 model; the imported program's
    reference is exact, so the plain float64 copy differs by float32 rounding and the comparison mode
    removes exactly that."""
    m64, prog = _program("llama")
    _, xs = build("llama", batch=2, seed=1)
    with torch.no_grad():
        plain = m64(*xs)[0]
    mode = ir.reference.exact_float32_casts()
    with torch.no_grad(), mode:
        exact = m64(*xs)[0]
    got = ir.run(prog, *xs)[0]
    assert mode.count > 0
    assert 1e-9 < _rel(got, plain) < 1e-5
    assert _rel(got, exact) < 1e-12


def test_wrappers_are_inlined_and_an_enabled_autocast_is_refused():
    class M(torch.nn.Module):
        def forward(self, x):
            with torch.no_grad():
                y = x * 2.0
            with torch.autocast("cpu", enabled=False):
                z = y + 1.0
            return z.sin()

    x = torch.randn(3, 4, dtype=torch.float64)
    prog = ir.import_module(M(), (x,))
    assert torch.allclose(ir.run(prog, x)[0], M()(x), rtol=0, atol=1e-15)
    assert "wrap" not in " ".join(n.op for n in prog.nodes)

    class A(torch.nn.Module):
        def forward(self, x):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                return x @ x.T

    with pytest.raises(UnsupportedOpError, match="autocast"):
        ir.import_module(A(), (torch.randn(3, 4),))


def test_in_place_bitwise_or_on_a_mask_imports_with_alias_semantics():
    class M(torch.nn.Module):
        def forward(self, x):
            m = x > 0
            m.bitwise_or_(x < -1)
            return torch.where(m, x, torch.zeros_like(x))

    x = torch.randn(4, 5, dtype=torch.float64)
    prog = ir.import_module(M(), (x,))
    assert torch.equal(ir.run(prog, x)[0], M()(x))

    class Bad(torch.nn.Module):
        def forward(self, x):
            m = x > 0
            old = m
            m.bitwise_or_(x < -1)
            return torch.where(old, x, -x) + m.to(x.dtype)

    # ``old`` is the same tensor, so it sees the change too; the import must agree
    prog = ir.import_module(Bad(), (x,))
    assert torch.equal(ir.run(prog, x)[0], Bad()(x))


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_float64_min_mask_times_zero_stays_zero_in_float32():
    """A float64 import keeps torch.finfo(float64).min; the float32 executor saturates it to
    torch.finfo(float32).min, so mask * 0 is 0 (as in the float32 model), not -inf * 0 = NaN."""
    from truestar.compiler import compile_program

    lo = torch.finfo(torch.float64).min

    class M(torch.nn.Module):
        def forward(self, x):
            mask = torch.full((x.shape[-1], x.shape[-1]), lo, dtype=x.dtype)
            keep = torch.ones_like(mask).triu(1)
            mask = mask * keep
            return torch.softmax(x @ x.transpose(-1, -2) + mask, dim=-1)

    x = torch.randn(2, 5, 5, dtype=torch.float64)
    prog = ir.import_module(M(), (x,))
    out = compile_program(prog, "cuda", "float32")([x.float()])[0]
    assert torch.isfinite(out).all()
    assert _rel(out.cpu(), M()(x)) < 1e-5


def test_export_refuses_t5_when_float32_islands_are_compared_as_float64(monkeypatch):
    """Without the exact cast mode the float64 copy of T5 really computes its layer norm in float32,
    and the export bar (1e-9) must catch the 7e-8 difference: the verification is not toothless."""
    from truestar import export

    class Null:
        count = 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ir.reference, "exact_float32_casts", Null)
    model, args = build("t5_encoder", batch=2)
    with pytest.raises(export.VerificationError, match="max rel error"):
        export.export(model, args, output=None)


# ----------------------------------------------------------------------------- coverage refresh (2025 to 2026)


@pytest.mark.parametrize("name", REFUSED)
def test_moe_linear_attention_and_chunked_ssm_families_are_refused_by_name(name):
    """gpt-oss, Qwen3 MoE, Qwen3 Next, Llama 4, DeepSeek V3, Kimi Linear, OLMoE, Nemotron H: refused
    before torch.export with TS940 (mixture of experts first, also when they have Gated DeltaNet or
    SSD layers), the layer class and its path named; the export path refuses the same model too."""
    model, args = build(name, batch=2)
    with pytest.raises(UnsupportedOpError) as info:
        ir.import_module(model, args)
    code = info.value.code
    assert code in ("TS940", "TS941", "TS942"), str(info.value)
    msg = str(info.value)
    assert msg.startswith(f"[{code}]") and " at model" in msg and "Nothing was imported" in msg
    # Qwen3 Next's and Kimi Linear's MoE layers are lowered (truestar.moe); their linear attention is not
    want = {"qwen3_next": "TS941", "kimi_linear": "TS941"}.get(name, "TS940")
    assert code == want, msg
    try:
        ep = torch.export.export(model, args, strict=False)
    except Exception:   # noqa: BLE001  (a few of these do not export at all; import_module refused first)
        return
    with pytest.raises(UnsupportedOpError) as info2:
        ir.import_exported(ep, name=name)
    # exported outside truestar.moe.lowered, a covered MoE layer is refused first (TS940)
    assert info2.value.code == ("TS940" if name in ("qwen3_next", "kimi_linear") else code)


@pytest.mark.parametrize("name", sorted(n for n in testing_hf.KNOWN_GAPS_2026 if n in testing_hf.available(
    testing_hf.MODELS_2026)))
def test_known_gaps_still_fail_as_documented(name):
    if TRANSFORMERS_MAJOR < 5:
        pytest.skip("the gaps were measured on transformers 5.24 (Pixtral's 4.x vision model imports)")
    model, args = build(name, batch=2)
    with pytest.raises(Exception):     # noqa: B017  (UnsupportedOpError or torch.export's own error)
        ir.import_module(copy.deepcopy(model).double(), tuple(a.double() if a.is_floating_point() else a
                                                               for a in args))


def test_current_families_are_all_measured():
    """On a transformers that knows them all (5.24 in docs/MODEL_COVERAGE.md), every 2025 to 2026
    family is either tested above or listed as a gap with its reason."""
    known = testing_hf.available(testing_hf.MODELS_2026)
    assert set(known) == set(CURRENT) | (set(testing_hf.KNOWN_GAPS_2026) & set(known))
    if len(known) == len(testing_hf.MODELS_2026):
        assert len(CURRENT) == 28


# ----------------------------------------------------------------------------- lowered linear attention (P60)


@pytest.mark.parametrize("name", NOT_LOWERED)
def test_ssd_and_gated_deltanet_families_without_a_lowering_stay_refused_with_the_reason(name):
    """On a transformers without the module function the lowering replaces (4.48 computes the scan
    inline in the layer), Mamba 2, Bamba and the others stay refused, and the message says why."""
    model, args = build(name, batch=2)
    with pytest.raises(UnsupportedOpError) as info:
        ir.import_module(model, args)
    assert info.value.code == testing_hf.LOWERED_CODES[name], str(info.value)
    assert "Not lowered:" in str(info.value) and "Nothing was imported" in str(info.value)


@pytest.mark.parametrize("name", LOWERED)
def test_lowered_family_has_one_linear_attention_op_per_layer_and_compiles_on_cpu(name):
    """Each SSD or Gated DeltaNet layer is one linear_attention IR op; the CPU float32 program
    compiles with no float64 fallback and its first call is checked against the reference."""
    from truestar.compiler import compile_program
    from truestar.ir.hf_linattn import lowerable

    m64, prog = _program(name)
    assert prog.op_counts().get("linear_attention") == len(lowerable(m64)) > 0
    cp = compile_program(prog, "cpu", "float32")
    _, xs = build(name, batch=3, seed=5)
    outs = cp(list(xs))
    want = ir.run(prog, *xs)
    assert _rel(outs[0].cpu(), want[0]) < 1e-4
    assert not cp.report.fallbacks, [f["ops"] for f in cp.report.fallbacks]


def test_lowering_is_undone_after_export_and_a_plain_export_is_still_refused():
    """The replacement lives in the transformers module only during import_module's export; a
    program exported without it holds the layer's own chunked code and is refused with the reason."""
    if "mamba2" not in LOWERED:
        pytest.skip("needs a transformers with the Mamba 2 module function (5.x)")
    import sys
    from truestar.ir import hf_linattn

    model, args = build("mamba2", batch=2)
    mod = sys.modules[type(model.model.backbone.layers[0].mixer).__module__]
    before = getattr(mod, hf_linattn.SSD_FN)
    ir.import_module(copy.deepcopy(model), args)
    assert getattr(mod, hf_linattn.SSD_FN) is before
    ep = torch.export.export(model, args)
    with pytest.raises(UnsupportedOpError) as info:
        ir.import_exported(ep, name="mamba2")
    assert info.value.code == "TS942" and "exported without Truestar's lowering" in str(info.value)
    with hf_linattn.lowered(model):
        ep = torch.export.export(model, args)
    assert getattr(mod, hf_linattn.SSD_FN) is before
    ir.import_exported(ep, name="mamba2")     # the lowered export imports


def test_a_call_with_a_recurrent_state_is_refused_with_ts1150():
    """Generation with a cache passes an initial state or asks for the final one: not lowered."""
    from truestar.ir import hf_linattn

    x = torch.randn(1, 3, 2, 4)
    with pytest.raises(UnsupportedOpError) as info:
        hf_linattn.ssd_chunk_scan(x, torch.rand(1, 3, 2), -torch.ones(2), torch.randn(1, 3, 1, 2),
                                  torch.randn(1, 3, 1, 2), 4, initial_states=torch.zeros(1, 2, 4, 2))
    assert info.value.code == "TS1150"
    with pytest.raises(UnsupportedOpError) as info:
        hf_linattn.gdn_chunk(x, x, x, -torch.rand(1, 3, 2), torch.rand(1, 3, 2), output_final_state=True)
    assert info.value.code == "TS1150"


def test_a_view_written_twice_imports_and_a_stale_view_is_still_refused():
    """Qwen3.5's rotary embedding writes two slices of ``f[0]`` through the same view, one after the
    other: the view holds the first write, so the second is exact (P60). A view taken before a write
    through another view of the same tensor is stale and stays refused."""
    class Twice(torch.nn.Module):
        def forward(self, x):
            f = x * 2.0
            t = f[0]
            t[..., 1:3:3] = f[1, ..., 1:3:3]
            t[..., 2:6:3] = f[2, ..., 2:6:3]
            return torch.cat((t, t), dim=-1)

    class Stale(torch.nn.Module):
        def forward(self, x):
            f = x * 2.0
            t = f[0]
            u = f[1]
            u[..., :2] = 1.0
            t[..., :2] = f[2, ..., :2]
            return t + f

    x = torch.randn(3, 2, 6, dtype=torch.float64)
    got = ir.run(ir.import_module(Twice(), (x,)), x)
    assert torch.equal(got[0] if isinstance(got, (list, tuple)) else got, Twice()(x))
    # compiled on the CPU this pattern is refused by the first call check today (P60 status entry)
    with pytest.raises(UnsupportedOpError, match="value semantics|take the view again"):
        ir.import_module(Stale(), (x,))
