"""The importer changes of the vision and image generation coverage (ir/importer_vision.py, the cat
cast in ir/importer.py, the TS1260 refusal in ir/refusals.py; docs/MODEL_COVERAGE.md). Small
programs only: the models themselves are in tests/test_vision_models.py."""

import pytest
import torch

from truestar import ir
from truestar.ir.autodiff import grad
from truestar.ir.core import UnsupportedOpError

F64 = torch.float64


def _imp(fn, *xs, module=None):
    class M(torch.nn.Module):
        def forward(self, *a):
            return fn(*a)
    m = module if module is not None else M()
    return ir.import_module(m, xs), m


def _ops(prog):
    return [n.op for n in prog.nodes]


def test_concat_and_concatenate_are_cat():
    x, y = torch.randn(2, 3, dtype=F64), torch.randn(2, 5, dtype=F64)
    for f in (lambda a, b: torch.concat([a, b], dim=1), lambda a, b: torch.concatenate([a, b], dim=1)):
        prog, m = _imp(f, x, y)
        assert torch.equal(ir.run(prog, x, y)[0], m(x, y))


def test_div_with_a_rounding_mode_of_constants_is_folded_to_torchs_quotient():
    """DETR's frequency table: torch.div(arange, 2, rounding_mode="floor"). Also the cases where
    Python's floor division differs from floor(a / b) (1 // 0.1 is 9, floor(1 / 0.1) is 10), trunc
    of negatives, and integers: the folded constant is torch's own quotient."""
    def f(v):
        t = torch.arange(8, dtype=v.dtype)
        a = torch.tensor([1.0, -1.0, 7.5, -7.5, 0.3], dtype=v.dtype)
        b = torch.tensor([0.1, 0.1, 2.0, 2.0, 0.1], dtype=v.dtype)
        i = torch.arange(-5, 5)
        return (v * torch.div(t, 2, rounding_mode="floor").sum(), torch.div(a, b, rounding_mode="floor"),
                torch.div(a, b, rounding_mode="trunc"), torch.div(i, 3, rounding_mode="floor"),
                torch.div(i, 3, rounding_mode="trunc"), torch.floor_divide(i, 3))
    x = torch.randn(3, dtype=F64)
    prog, m = _imp(f, x)
    got, want = ir.run(prog, x), m(x)
    for g, w in zip(got, want):
        assert g.dtype == w.dtype and torch.equal(g, w), (g, w)
    assert want[1][0].item() == 9.0          # the case floor(a / b) gets wrong
    assert sum(p.name.startswith("div_") for p in prog.params) == 6      # six folded quotients


def test_div_with_a_rounding_mode_of_an_input_is_refused_by_name():
    x = torch.randn(4, dtype=F64)
    with pytest.raises(UnsupportedOpError) as info:
        _imp(lambda v: torch.div(v, 0.1, rounding_mode="floor"), x)
    assert info.value.code == "TS860"
    assert "rounding_mode 'floor' of a value that depends on an input" in str(info.value)


def test_div_rounding_mode_none_is_the_plain_division():
    x, y = torch.randn(4, dtype=F64), torch.rand(4, dtype=F64) + 0.5
    prog, m = _imp(lambda a, b: torch.div(a, b, rounding_mode=None), x, y)
    assert torch.equal(ir.run(prog, x, y)[0], m(x, y))


def test_cat_of_mixed_dtypes_is_cast_to_torchs_result_dtype():
    """SAM joins float points with a padding point of another dtype; torch.cat promotes. The IR cat
    gets inputs of one dtype (casts first), so the compiler's layout step takes it."""
    x = torch.randn(2, 3, dtype=F64)
    i = torch.arange(6).reshape(2, 3)
    prog, m = _imp(lambda a, b: torch.cat([a, b], dim=0), x, i)
    got = ir.run(prog, x, i)[0]
    assert got.dtype == F64 and torch.equal(got, m(x, i))
    cat = [n for n in prog.nodes if n.op == "cat"][0]
    assert len({v.type.dtype for v in cat.inputs}) == 1


def test_cat_of_mixed_dtypes_compiles_without_fallback():
    pytest.importorskip("numba")
    from truestar.compiler import compile_program
    x = torch.randn(2, 3)
    i = torch.arange(6).reshape(2, 3)
    prog, m = _imp(lambda a, b: torch.cat([a * 2.0, b], dim=0), x, i)
    cp = compile_program(prog, "cpu", "float32")
    out = cp([x, i])                       # verified against the float64 reference on the first call
    assert torch.allclose(out[0], m(x, i))
    assert not cp.report.fallbacks, [f["ops"] for f in cp.report.fallbacks]


def test_constant_pow_and_cumsum_are_folded_at_import():
    """Flux's RoPE frequencies (a scalar base, a tensor exponent from arange) and SAM's position grid
    (cumsum of ones) depend on no input and no weight: folded to constants, equal to the float64
    reference, no pow or cumsum node left."""
    def f(v):
        freqs = 1.0 / (10000.0 ** (torch.arange(0, 8, 2, dtype=v.dtype) / 8))
        grid = torch.ones(4, 4, dtype=v.dtype).cumsum(0) - 0.5
        return (v[:, None] * freqs, v.sum() * grid)
    x = torch.randn(3, dtype=F64)
    prog, m = _imp(f, x)
    for g, w in zip(ir.run(prog, x), m(x)):
        assert (g - w).abs().max().item() <= 1e-15 * w.abs().max().item()
    assert "pow" not in _ops(prog) and "cumsum" not in _ops(prog)


def test_pow_and_cumsum_of_an_input_or_a_weight_are_not_folded():
    """A value that reaches an input or a model parameter keeps its node: its gradient and its per
    call value stay."""
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.e = torch.nn.Parameter(torch.tensor([0.5, 1.5, 2.0], dtype=F64))

        def forward(self, v):
            return (2.0 ** self.e) * v.cumsum(0)

    m = M()
    x = torch.randn(3, dtype=F64)
    prog, _ = _imp(None, x, module=m)
    assert "pow" in _ops(prog) and "cumsum" in _ops(prog)
    names = [v.name for v in prog.params if v.name in dict(m.named_parameters())]
    assert names == ["e"]
    step = grad(prog, wrt=names, of=[0], seed="input")
    cot = torch.randn(3, dtype=F64)
    got = ir.run(step, x, cot)[len(prog.outputs):][0]
    (m(x) * cot).sum().backward()
    assert torch.allclose(got, m.e.grad, rtol=1e-12, atol=0)


def test_detector_modules_are_refused_by_name():
    """torchvision's detectors in eval mode: refused with TS1260 before torch.export, the module class
    and path named, and the dense part suggested."""
    tvd = pytest.importorskip("torchvision.models.detection")
    from truestar.ir.refusals import refusal_for

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m

        def forward(self, x):
            return self.model(list(x))[0]["boxes"]

    m = tvd.retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=3, min_size=32, max_size=32)
    with pytest.raises(UnsupportedOpError) as info:
        ir.import_module(Wrap(m).eval(), (torch.rand(1, 3, 32, 32),))
    msg = str(info.value)
    assert info.value.code == "TS1260" and msg.startswith("[TS1260]")
    assert "(RetinaNet at model)" in msg and "Nothing was imported" in msg and "model.head" in msg
    # the R-CNN family names its region proposal network; a backbone alone is not refused, and the
    # class names match only in torchvision's detection package (a user's own SSD is not a detector)
    d = "torchvision.models.detection."
    assert refusal_for([("model", d + "faster_rcnn.FasterRCNN"), ("model.rpn", d + "rpn.RegionProposalNetwork"),
                        ("model.roi_heads", d + "roi_heads.RoIHeads")])[2:] == ("RegionProposalNetwork", "model.rpn")
    assert refusal_for([("backbone", d + "backbone_utils.BackboneWithFPN"), ("head", d + "rpn.RPNHead"),
                        ("box_head", d + "faster_rcnn.TwoMLPHead")]) is None
    assert refusal_for([("1", "mymodels.SSD"), ("2", "SSD"), ("3", "RetinaNet")]) is None
