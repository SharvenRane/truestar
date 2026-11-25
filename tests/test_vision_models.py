"""Image generation, detection and segmentation models with random weights (truestar.testing_vision,
docs/MODEL_COVERAGE.md "Image generation, detection and segmentation"): import against PyTorch
float64, compile for the CPU in float32 (verified against the float64 reference on the first call,
no float64 CPU fallback), and one training step's gradient against torch.autograd in float64.

The models need diffusers, torchvision or transformers; the measured run used diffusers 0.43.0,
torchvision 0.26.0 and transformers 5.24.0 on torch 2.11 (CPU)."""

import copy

import pytest
import torch

from truestar import ir
from truestar import testing_vision as tv
from truestar.ir.core import UnsupportedOpError

COVERED = sorted(n for n in tv.available({**tv.GEN, **tv.VISION}) if n not in tv.KNOWN_GAPS)
DETECTORS = sorted(tv.available(tv.DETECTORS))
GAPS = sorted(n for n in tv.KNOWN_GAPS if n in tv.available(tv.ALL))


def _rel(got, want):
    scale = want.double().abs().max().item() or 1.0
    return (got.double() - want.double()).abs().max().item() / scale


def _f64(xs):
    return tuple(x.double() if x.is_floating_point() else x for x in xs)


_PROGRAMS = {}


def _program(name):
    if name not in _PROGRAMS:
        model, args = tv.build(name, batch=2)
        m64 = copy.deepcopy(model).double()
        _PROGRAMS[name] = (model, m64, ir.import_module(m64, _f64(args)))
    return _PROGRAMS[name]


@pytest.mark.parametrize("name", COVERED)
def test_import_matches_pytorch_float64(name):
    _, m64, prog = _program(name)
    _, xs = tv.build(name, batch=2, seed=1)
    xs = _f64(xs)
    with torch.no_grad(), ir.reference.exact_float32_casts():
        want = m64(*xs)
    got = ir.run(prog, *xs)
    got = got if isinstance(got, (list, tuple)) else [got]
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g.shape == w.shape
        assert _rel(g, w) < 1e-10, (name, _rel(g, w))


@pytest.mark.parametrize("name", COVERED)
def test_compiled_cpu_float32_is_verified_on_first_call_with_no_fallback(name):
    pytest.importorskip("numba")
    from truestar.compiler import compile_program

    model, _, prog = _program(name)
    cp = compile_program(prog, "cpu", "float32")
    _, xs = tv.build(name, batch=2, seed=5)
    outs = cp(list(xs))             # checked against the float64 reference on the first call (raises otherwise)
    with torch.no_grad():
        want = model(*xs)
    assert _rel(outs[0], want[0]) < 1e-3
    assert cp.report.kernels
    assert not cp.report.fallbacks, [f["ops"] for f in cp.report.fallbacks]


@pytest.mark.parametrize("name", COVERED)
def test_one_training_step_gradient_matches_torch_autograd(name):
    """The gradient of a random projection of output 0 with respect to every trainable weight (the
    inputs for the schedulers, which have none): truestar.ir.grad against torch.autograd, both in
    float64, bar 1e-10 relative to the largest gradient entry."""
    from truestar.ir.autodiff import grad

    _, m64, prog = _program(name)
    _, xs = tv.build(name, batch=2, seed=3)
    xs = _f64(xs)
    sd = m64.state_dict(keep_vars=True)
    params = {v.name: sd[v.name] for v in prog.params
              if v.name in sd and isinstance(sd[v.name], torch.nn.Parameter) and sd[v.name].requires_grad}
    names = list(params)
    on_inputs = not names
    if on_inputs:
        names = [v.name for v, x in zip(prog.inputs, xs) if x.is_floating_point()]
    out0 = ir.run(prog, *xs)
    out0 = (out0 if isinstance(out0, (list, tuple)) else [out0])[0]
    cot = torch.randn(out0.shape, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    got = ir.run(grad(prog, wrt=names, of=[0], seed="input"), *xs, cot)[len(prog.outputs):]
    m64.zero_grad(set_to_none=True)
    leaves = tuple(x.clone().requires_grad_(True) if x.is_floating_point() else x for x in xs)
    with ir.reference.exact_float32_casts():
        m64(*leaves)[0].backward(cot)
    if not on_inputs:
        # every weight torch gives a gradient is in the program (the VAE encoder's program does not
        # use the decoder, SAM with a point prompt leaves its mask and box embeddings unused)
        used = {id(p) for p in m64.parameters() if p.grad is not None}
        assert used <= {id(p) for p in params.values()}, name
    wants = [x.grad for x in leaves if x.is_floating_point()] if on_inputs else [params[n].grad for n in names]
    wants = [torch.zeros_like(g) if w is None else w for g, w in zip(got, wants)]
    scale = max(w.abs().max().item() for w in wants) or 1.0
    for n, g, w in zip(names, got, wants):
        assert (g - w).abs().max().item() / scale < 1e-10, (name, n)


@pytest.mark.parametrize("name", DETECTORS)
def test_full_detectors_are_refused_by_name(name):
    """Faster R-CNN, RetinaNet and Mask R-CNN in eval mode: TS1260 before torch.export, naming the
    module that selects boxes by value; torch.export itself cannot trace them."""
    model, args = tv.build(name, batch=1)
    with pytest.raises(UnsupportedOpError) as info:
        ir.import_module(model, args)
    msg = str(info.value)
    assert info.value.code == "TS1260" and msg.startswith("[TS1260]")
    want = "(RetinaNet at model)" if name == "retinanet" else "(RegionProposalNetwork at model.rpn)"
    assert want in msg and "Nothing was imported" in msg
    with pytest.raises(Exception, match="data-dependent|GuardOnDataDependent"):
        torch.export.export(model, args, strict=False)


@pytest.mark.parametrize("name", GAPS)
def test_known_gaps_still_fail_as_documented(name):
    model, args = tv.build(name, batch=2)
    with pytest.raises(Exception, match="data-dependent|GuardOnDataDependent"):
        ir.import_module(copy.deepcopy(model).double(), _f64(args))


def test_every_family_is_measured():
    """With every library present (the measured run): 16 programs covered, 3 detectors refused, the
    Euler step a known gap."""
    known = tv.available(tv.ALL)
    assert set(known) == set(COVERED) | set(DETECTORS) | set(GAPS)
    if len(known) == len(tv.ALL):
        assert (len(COVERED), len(DETECTORS), len(GAPS)) == (16, 3, 1)
