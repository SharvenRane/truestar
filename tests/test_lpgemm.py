"""truestar.lpgemm: FP8 and NVFP4 training GEMMs with float64 verdicts per GEMM (format loss and
kernel error), the W / W^T invariant, per layer gradients against float64 and the gradient error
planner (docs/TRAIN_NUMERICS.md)."""

import pytest
import torch

from truestar import lpgemm as G

CUDA = torch.cuda.is_available()
cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")
F64 = torch.float64


def _make():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 384), torch.nn.GELU(),
                               torch.nn.Linear(384, 128))


def _batch():
    g = torch.Generator().manual_seed(1)
    return torch.randn(256, 256, generator=g), torch.randint(0, 128, (256,), generator=g)


LOSS = torch.nn.functional.cross_entropy
# measured dW error against float64 on this model (docs/TRAIN_NUMERICS.md), with headroom
DW_LIMIT = {"bf16": 0.02, "fp8": 0.12, "fp8_channel": 0.12, "fp8_hybrid": 0.18, "nvfp4_fwd": 0.35}


@cuda
@pytest.mark.parametrize("recipe", list(G.RECIPES))
def test_every_gemm_passes_its_kernel_verdict_and_gradients_are_measured(recipe):
    x, y = _batch()
    rep = G.grad_report(_make().double(), G.convert(_make().cuda(), recipe), LOSS, x, y)
    assert len(rep.layers) == 3
    for l in rep.layers:
        assert {g.which for g in l.gemms} == {"fwd", "dgrad", "wgrad"}
        for g in l.gemms:
            assert g.ok, (l.layer, g)
        assert l.dw_error < DW_LIMIT[recipe], (l.layer, l.dw_error)
    assert abs(rep.loss - rep.loss_f64) < 0.02 * rep.loss_f64
    codes = {i.code for i in rep.issues()}
    if recipe == "fp8_channel":
        assert codes == {"TS1043"}
        assert all(not l.consistent for l in rep.layers)
    else:
        assert codes == set(), rep.issues()


@cuda
@pytest.mark.parametrize("recipe,which,bug", [("fp8_channel", "fwd", "tensor_scale_for_channel"),
                                              ("fp8", "wgrad", "fp16_accumulate"),
                                              ("fp8_channel", "dgrad", "tensor_scale_for_channel"),
                                              ("nvfp4_fwd", "fwd", "block_scale_shift")])
def test_injected_kernel_bugs_are_ts1042_and_not_format_loss(recipe, which, bug):
    x, y = _batch()
    clean = G.grad_report(_make().double(), G.convert(_make().cuda(), recipe), LOSS, x, y)
    m = G.convert(_make().cuda(), recipe)
    G.layers(m)[1].inject = {which: bug}
    rep = G.grad_report(_make().double(), m, LOSS, x, y)
    rec = [g for g in rep.layers[1].gemms if g.which == which][0]
    ref = [g for g in clean.layers[1].gemms if g.which == which][0]
    assert not rec.ok and rec.kernel_ratio > 1.0
    assert "TS1042" in {i.code for i in rep.issues()}
    if which == "fwd":   # the forward's operands are the same, so the format verdict is unchanged
        assert abs(rec.format_loss - ref.format_loss) < 1e-12


@cuda
def test_strict_mode_refuses_the_wrong_gemm_before_its_result_spreads():
    """A float16 accumulating forward kernel overflows; strict mode raises TS1042 at that GEMM."""
    x, y = _batch()
    m = G.convert(_make().cuda(), "fp8")
    for l in G.layers(m):
        l.strict = True
    G.layers(m)[0].inject = {"fwd": "fp16_accumulate"}
    with pytest.raises(G.LowPrecisionKernelError) as e:
        LOSS(m(x.cuda() * 50), y.cuda())
    assert e.value.code == "TS1042" and e.value.record.which == "fwd"


def test_transposed_weight_quantization_invariant():
    """Per tensor FP8 and 16 x 16 NVFP4 tiles quantize W and W^T to the same values; per channel FP8
    and 1 x 16 NVFP4 blocks do not (most entries differ)."""
    w = torch.randn(128, 256, generator=torch.Generator().manual_seed(2))
    assert G.transpose_consistency(w, "fp8_tensor") == (0, 0.0)
    assert G.transpose_consistency(w, "nvfp4_2d") == (0, 0.0)
    n, worst = G.transpose_consistency(w, "fp8_channel")
    assert n > 0.5 * w.numel() and worst > 0.01
    n, worst = G.transpose_consistency(w, "nvfp4")
    assert n > 0.5 * w.numel() and worst > 0.05


@cuda
def test_nvfp4_forward_and_its_backward_use_one_weight():
    """The NVFP4 recipe's input gradient multiplies by exactly the forward's dequantized weight
    (block values exact in bf16, the tensor scale applied once), checked bitwise."""
    x, y = _batch()
    rep = G.grad_report(_make().double(), G.convert(_make().cuda(), "nvfp4_fwd"), LOSS, x, y)
    assert all(l.consistent for l in rep.layers)


@cuda
def test_planner_keeps_the_layers_over_budget_in_bf16():
    x, y = _batch()
    kept, hist = G.plan(_make, "fp8_hybrid", LOSS, x, y, budget=1.0)
    assert kept == [] and len(hist) == 1
    errs = {l.layer: l.dw_error for l in hist[0].layers}
    budget = sorted(errs.values())[1] * 1.001     # only the smallest error is within it
    kept, hist = G.plan(_make, "fp8_hybrid", LOSS, x, y, budget=budget)
    assert 1 <= len(kept) <= 2
    final = {l.layer: l.dw_error for l in hist[-1].layers}
    assert all(v <= budget for v in final.values()), final
