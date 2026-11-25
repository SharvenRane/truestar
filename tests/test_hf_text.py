"""Text, retrieval, speech and vision language models built from configs (random weights, nothing
downloaded): import against PyTorch float64, the compiled CPU float32 program verified on its first
call, and one training step's gradient against torch.autograd in float64. See
docs/MODEL_COVERAGE.md ("Text, retrieval, speech and vision language models") and
scripts/text_coverage.py."""

import copy
import warnings

import pytest
import torch

transformers = pytest.importorskip("transformers")
if int(transformers.__version__.split(".")[0]) < 5:
    pytest.skip("measured on transformers 5.24 (the 4.x cache, mask and vision language code differ)",
                allow_module_level=True)

from truestar import ir  # noqa: E402
from truestar import testing_hf_text as T  # noqa: E402

KNOWN = T.available()
NAMES = sorted(n for n in KNOWN if n not in T.KNOWN_GAPS_TEXT)
GAPS = sorted(n for n in KNOWN if n in T.KNOWN_GAPS_TEXT)


def _to64(xs):
    return tuple(x.double() if x.is_floating_point() else x for x in xs)


def _rel(got, want):
    scale = want.double().abs().max().item() or 1.0
    return (got.double() - want.double()).abs().max().item() / scale


_PROGRAMS = {}


def _program(name):
    if name not in _PROGRAMS:
        model, args = T.build(name, batch=2)
        m64 = copy.deepcopy(model).double()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prog = ir.import_module(m64, _to64(args), dynamic_shapes=T.dynamic_shapes(model.kind, args))
        _PROGRAMS[name] = (m64, prog)
    return _PROGRAMS[name]


def _batches(name, dynamic, fixed):
    return dynamic if T.kind_of(name) in T.DYNAMIC_BATCH else fixed


@pytest.mark.parametrize("name", NAMES)
def test_import_matches_pytorch_float64(name):
    m64, prog = _program(name)
    assert ("B" in prog.symbols) == (T.kind_of(name) in T.DYNAMIC_BATCH)
    for batch, seed in _batches(name, ((2, 1), (5, 2)), ((2, 1),)):
        _, xs = T.build(name, batch=batch, seed=seed)
        xs = _to64(xs)
        with torch.no_grad(), ir.reference.exact_float32_casts():
            want = m64(*xs)
        got = ir.run(prog, *xs)
        got = got if isinstance(got, (list, tuple)) else [got]
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert g.shape == w.shape
            assert _rel(g, w) < 1e-10, (name, batch, _rel(g, w))


@pytest.mark.parametrize("name", NAMES)
def test_compiled_cpu_float32_is_verified_on_first_call_with_no_fallback(name):
    from truestar.compiler import compile_program

    _, prog = _program(name)
    cp = compile_program(prog, "cpu", "float32")
    for batch, seed in _batches(name, ((3, 5), (2, 6)), ((2, 5),)):
        _, xs = T.build(name, batch=batch, seed=seed)
        outs = cp(list(xs))             # verified against the float64 reference on the first call
        want = ir.run(prog, *_to64(xs))
        want = want if isinstance(want, (list, tuple)) else [want]
        assert _rel(outs[0], want[0]) < 1e-3
    assert cp.report.kernels
    assert not cp.report.fallbacks, [f["ops"] for f in cp.report.fallbacks]


@pytest.mark.parametrize("name", NAMES)
def test_one_training_step_gradient_matches_torch_autograd(name):
    """Every trainable weight (a tied weight once) against torch.autograd, both float64, bar 1e-10
    relative to the largest gradient entry; output 0 (the decode step's logits, the sentence
    embedding, the audio hidden states) under a random cotangent."""
    from truestar.ir.autodiff import grad

    m64, prog = _program(name)
    _, xs = T.build(name, batch=2, seed=3)
    xs = _to64(xs)
    sd = m64.state_dict(keep_vars=True)
    params = {v.name: sd[v.name] for v in prog.params
              if v.name in sd and isinstance(sd[v.name], torch.nn.Parameter) and sd[v.name].requires_grad}
    names = list(params)
    assert {id(p) for p in params.values()} == {id(p) for p in m64.parameters() if p.requires_grad}
    out0 = ir.run(prog, *xs)
    out0 = (out0 if isinstance(out0, (list, tuple)) else [out0])[0]
    cot = torch.randn(out0.shape, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    got = ir.run(grad(prog, wrt=names, of=[0], seed="input"), *xs, cot)[len(prog.outputs):]
    m64.zero_grad(set_to_none=True)
    with ir.reference.exact_float32_casts():
        m64(*xs)[0].backward(cot)
    scale = max(params[n].grad.abs().max().item() for n in names if params[n].grad is not None)
    for n, g in zip(names, got):
        want = params[n].grad
        want = torch.zeros_like(g) if want is None else want
        assert (g - want).abs().max().item() / scale < 1e-10, (name, n)


@pytest.mark.parametrize("name", GAPS)
def test_known_gaps_are_refused_with_the_model_line(name):
    """torch.export cannot trace these at a fixed image size; the refusal names the model line
    (before: torch's GuardOnDataDependentSymNode with only torch internal frames on top)."""
    from truestar.ir.core import UnsupportedOpError

    model, args = T.build(name, batch=2)
    with pytest.raises(UnsupportedOpError) as info:
        ir.import_module(copy.deepcopy(model).double(), _to64(args),
                         dynamic_shapes=T.dynamic_shapes(model.kind, args))
    msg = str(info.value)
    assert msg.startswith("torch.export cannot trace TextWrap: ") and "nothing was imported" in msg
    assert type(info.value.__cause__).__name__ == "GuardOnDataDependentSymNode"


def test_the_placeholder_count_check_is_refused_by_name_when_left_on():
    """LLaVA with transformers' runtime check on: one problem naming the check and the switch."""
    from truestar.ir.core import UnsupportedOpError

    if "llava" not in KNOWN:
        pytest.skip("needs the llava model type")
    model, args = T.build("llava", batch=2)
    model.disable_check = False
    with pytest.raises(UnsupportedOpError) as info:
        ir.import_module(copy.deepcopy(model).double(), _to64(args))
    msg = str(info.value)
    assert "runtime check on a value read from a tensor" in msg and "modeling_llava.py" in msg
    assert "TRANSFORMERS_DISABLE_TORCH_CHECK=1" in msg and "1 problem(s)" in msg


def test_every_family_is_measured():
    """On a transformers that knows them all (5.24 in docs/MODEL_COVERAGE.md), every family is
    either tested above or listed as a gap with its reason."""
    assert set(KNOWN) == set(NAMES) | set(GAPS)
    assert set(T.KNOWN_GAPS_TEXT) <= set(T.MODELS_TEXT)
    for name in T.MODELS_TEXT:
        assert T.kind_of(name) in ("seq2seq", "speech_decode", "embed", "audio", "vlm")
