"""Tests for the Stage 2 IR: importer, shape rules, reference interpreter."""

import copy
import math
import warnings

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import Dim

from truestar import ir
from truestar.ir import reference, shapes

warnings.filterwarnings("ignore", category=FutureWarning)

TOL = 1e-10


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-300)).item()


def randomize_norms(model: nn.Module, seed: int = 0) -> nn.Module:
    """Give every batch norm and layer norm non trivial statistics and affine params.

    A freshly initialised batch norm is nearly the identity (mean 0, var 1), which would hide a
    wrong formula. Random stats make each normalization actually change the values.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=g) * 0.1)
                m.running_var.copy_(torch.rand(m.running_var.shape, generator=g) + 0.5)
            if isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm)) and m.weight is not None:
                m.weight.copy_(1 + 0.2 * torch.randn(m.weight.shape, generator=g))
                m.bias.copy_(0.1 * torch.randn(m.bias.shape, generator=g))
    return model


def torch_f64(model: nn.Module, *args):
    with torch.no_grad():
        out = copy.deepcopy(model).double()(*[a.double() for a in args])
    return out[0] if isinstance(out, tuple) else out


BATCH = ({0: Dim("B", min=1, max=64)},)


def _resnet18():
    import torchvision
    torch.manual_seed(0)
    return randomize_norms(torchvision.models.resnet18().eval()), (2, 3, 64, 64)


def _mobilenet_v3_small():
    import torchvision
    torch.manual_seed(0)
    return randomize_norms(torchvision.models.mobilenet_v3_small().eval()), (2, 3, 64, 64)


def _mobilenet_v2():
    import torchvision
    torch.manual_seed(0)
    return randomize_norms(torchvision.models.mobilenet_v2().eval()), (2, 3, 64, 64)


def _encoder_layer_relu():
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(32, 4, 64, batch_first=True).eval()
    return randomize_norms(layer), (2, 7, 32)


def _encoder_layer_gelu_prenorm():
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(32, 4, 64, batch_first=True, activation="gelu", norm_first=True).eval()
    return randomize_norms(layer), (2, 7, 32)


def _encoder_two_layers():
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(32, 4, 64, batch_first=True, activation="gelu")
    enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False).eval()
    return randomize_norms(enc), (2, 7, 32)


MODELS = {
    "resnet18": _resnet18,
    "mobilenet_v3_small": _mobilenet_v3_small,
    "mobilenet_v2": _mobilenet_v2,
    "encoder_layer_relu": _encoder_layer_relu,
    "encoder_layer_gelu_prenorm": _encoder_layer_gelu_prenorm,
    "encoder_two_layers": _encoder_two_layers,
}

_cache = {}


def imported(name: str, decompose: bool):
    key = (name, decompose)
    if key not in _cache:
        model, shape = MODELS[name]()
        x = torch.randn(*shape, generator=torch.Generator().manual_seed(1))
        prog = ir.import_module(model, (x,), dynamic_shapes=BATCH, decompose=decompose)
        _cache[key] = (model, shape, prog)
    return _cache[key]


# ----------------------------------------------------------------------------- model zoo


@pytest.mark.parametrize("decompose", [False, True], ids=["export_ir", "core_aten"])
@pytest.mark.parametrize("name", list(MODELS))
def test_model_matches_pytorch_float64(name, decompose):
    model, shape, prog = imported(name, decompose)
    x = torch.randn(*shape, generator=torch.Generator().manual_seed(2))
    out = ir.run(prog, x)[0]
    ref = torch_f64(model, x)
    assert out.dtype == torch.float64
    assert out.shape == ref.shape
    err = rel_err(out, ref)
    assert err < TOL, f"{name}: relative error {err:.3e}"


@pytest.mark.parametrize("name", ["resnet18", "encoder_layer_relu", "encoder_two_layers"])
def test_symbolic_batch_two_sizes_one_import(name):
    model, shape, prog = imported(name, False)
    assert str(prog.inputs[0].type).startswith("f32[B, ")
    assert prog.symbols == {"B": (1, 64)}
    for batch in (1, 5):
        x = torch.randn(batch, *shape[1:], generator=torch.Generator().manual_seed(batch))
        out = ir.run(prog, x)[0]
        assert out.shape[0] == batch
        assert rel_err(out, torch_f64(model, x)) < TOL


def test_symbolic_shapes_flow_through_reshapes():
    _, _, prog = imported("encoder_layer_relu", False)
    text = prog.to_text(show_params=False)
    assert "4*B" in text and "5*B" not in text  # heads x batch is tracked as an expression
    assert "7*B" in text  # sequence x batch after the attention reshape


def test_symbol_range_and_input_shape_checked_at_run():
    _, shape, prog = imported("resnet18", False)
    with pytest.raises(ir.ShapeError, match=r"outside its range \[1, 64\]"):
        ir.run(prog, torch.randn(65, *shape[1:]))
    with pytest.raises(ir.ShapeError) as e:
        ir.run(prog, torch.randn(2, 4, 64, 64))
    assert "f32[B, 3, 64, 64]" in str(e.value) and "[2, 4, 64, 64]" in str(e.value)


def test_printer_shows_program_and_shapes():
    _, _, prog = imported("resnet18", False)
    text = str(prog)
    assert text.startswith("program ResNet(%x: f32[B, 3, 64, 64]) -> (f32[B, 1000])")
    assert "symbol B in [1, 64]" in text
    assert "@conv1.weight: f32[64, 3, 7, 7]" in text
    assert "= conv2d(%x, @conv1.weight, _) stride=[2, 2] padding=[3, 3]" in text
    assert ": f32[B, 64, 32, 32]" in text
    assert text.rstrip().endswith("return %linear")


# ----------------------------------------------------------------------------- shape errors (P9)


def test_shape_error_at_import_names_op_shapes_and_line():
    import torchvision
    model = torchvision.models.resnet18().eval()
    ep = torch.export.export(model, (torch.randn(2, 3, 64, 64),), dynamic_shapes=BATCH)
    with pytest.raises(ir.ShapeError) as e:
        ir.import_exported(ep, input_shapes={"x": ["B", 4, 64, 64]})
    msg = str(e.value)
    assert e.value.op == "conv2d"
    assert "f32[B, 4, 64, 64]" in msg and "f32[64, 3, 7, 7]" in msg
    assert "aten.conv2d.default" in msg and "resnet.py" in msg and "module conv1: Conv2d" in msg


def test_shape_error_transformer_feature_dim():
    model, _, _ = imported("encoder_layer_relu", False)
    ep = torch.export.export(model, (torch.randn(2, 7, 32),), dynamic_shapes=BATCH)
    with pytest.raises(ir.ShapeError) as e:
        ir.import_exported(ep, input_shapes={"src": ["B", 7, 48]})
    assert e.value.op == "linear"
    assert "f32[7, B, 48]" in str(e.value) and "f32[96, 32]" in str(e.value)


def test_shape_error_in_builder():
    b = ir.Builder()
    x = b.input("x", torch.float32, ("B", 512))
    w = b.param("w", torch.randn(1000, 256))
    with pytest.raises(ir.ShapeError) as e:
        b.op("linear", x, w, None)
    assert "f32[B, 512]" in str(e.value) and "f32[1000, 256]" in str(e.value)


def test_symbolic_dims_must_be_provably_equal():
    b = ir.Builder()
    x = b.input("x", torch.float32, ("B", 3))
    y = b.input("y", torch.float32, ("C", 3))
    z = b.input("z", torch.float32, (1, 3))
    b.op("add", x, z)  # broadcasting against 1 is fine
    b.op("add", x, x)
    with pytest.raises(ir.ShapeError, match="provably equal"):
        b.op("add", x, y)
    with pytest.raises(ir.ShapeError, match="element count"):
        b.op("reshape", x, shape=[ir.symbol("B") + 1, 3])


def test_verify_catches_a_corrupted_program():
    b = ir.Builder()
    x = b.input("x", torch.float32, ("B", 4))
    y = b.op("relu", x)
    prog = b.build([y])
    prog.nodes[0].output.type = ir.TensorType(torch.float32, (ir.symbol("B"), 5))
    with pytest.raises(ir.ShapeError, match="does not match the shape rule"):
        prog.verify()


# ----------------------------------------------------------------------------- unsupported ops


class _Unsupported(nn.Module):
    def forward(self, x):
        # cumsum, then flip and sort were the examples until they became supported (2026-11-24)
        y = x + torch.trace(x[:, :2].t() @ x[:, :2])
        return torch.logcumsumexp(y, dim=1)


def test_unsupported_ops_fail_loudly_with_all_names():
    ep = torch.export.export(_Unsupported(), (torch.randn(2, 5),))
    with pytest.raises(ir.UnsupportedOpError) as e:
        ir.import_exported(ep)
    msg = str(e.value)
    assert "aten.trace.default" in msg and "aten.logcumsumexp.default" in msg
    assert "test_ir.py" in msg and "nothing was skipped" in msg


def test_training_mode_fails_loudly():
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.Dropout(0.5)).train()
    ep = torch.export.export(model, (torch.randn(2, 3, 8, 8),))
    with pytest.raises(ir.UnsupportedOpError, match="eval"):
        ir.import_exported(ep)


class _BadAlias(nn.Module):
    def forward(self, x):
        a = x + 1
        b = a.relu_()
        return a * 2 + b


def test_inplace_ops_become_pure_or_are_rejected():
    """An in place ATen op becomes a pure op only when nobody reads the old value; else a loud error."""
    x = torch.randn(3, 4)
    ep = torch.export.export(_BadAlias(), (x,))
    prog = ir.import_exported(ep)  # torch.export already points later reads at the relu_ result
    assert rel_err(ir.run(prog, x)[0], _BadAlias()(x.double())) < TOL

    # Rewire one later read to the pre mutation value, as a graph with a real alias read would have it.
    relu = next(n for n in ep.graph.nodes if str(n.target) == "aten.relu_.default")
    mul = next(n for n in ep.graph.nodes if str(n.target) == "aten.mul.Tensor")
    mul.replace_input_with(relu, relu.args[0])
    with pytest.raises(ir.UnsupportedOpError, match="mutates %add, which %mul reads afterwards"):
        ir.import_exported(ep)


# ----------------------------------------------------------------------------- op coverage


class _OpZoo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 3, stride=2, padding=2, dilation=2, groups=2)
        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)
        self.lin = nn.Linear(8, 6)
        self.register_buffer("mask", torch.randn(5, 5, generator=torch.Generator().manual_seed(3)))

    def forward(self, x, q):
        h = self.dw(self.conv(x))
        h = F.silu(h) + torch.sigmoid(h) - torch.tanh(h) * F.gelu(h, approximate="tanh")
        h = F.relu6(h) / 3.0 + F.hardtanh(h, -0.5, 0.5)
        h = self.pool(h)
        g = F.adaptive_avg_pool2d(h, (3, 2))
        s = torch.cat([g.flatten(1), g.mean(dim=(2, 3)), h.sum(dim=-1).sum(dim=-1, keepdim=False)], dim=1)
        s = s[:, 1:13:2] * 2.0 - 1.0
        s = 1.0 - s
        a = F.softmax(s, dim=-1)
        att = F.scaled_dot_product_attention(q, q * 0.5, q, attn_mask=self.mask)
        att = att + F.scaled_dot_product_attention(q, q, q, is_causal=True, scale=0.3)
        t = self.lin(att.transpose(1, 2).reshape(q.shape[0], -1, 8)).mean(dim=1)
        return a + t


def test_op_zoo_matches_pytorch_float64():
    torch.manual_seed(0)
    model = _OpZoo().eval()
    x = torch.randn(2, 4, 17, 15)
    q = torch.randn(2, 2, 5, 4)
    for decompose in (False, True):
        ep = torch.export.export(model, (x, q))
        if decompose:
            ep = ep.run_decompositions()
        prog = ir.import_exported(ep)
        err = rel_err(ir.run(prog, x, q)[0], torch_f64(model, x, q))
        assert err < TOL, f"decompose={decompose}: {err:.3e}"


def test_every_ir_op_has_a_shape_rule_and_a_reference():
    assert set(shapes.RULES) == set(reference.IMPLS)


@pytest.mark.parametrize("cfg", [
    dict(stride=1, padding=0, dilation=1, groups=1),
    dict(stride=2, padding=1, dilation=1, groups=1),
    dict(stride=(2, 1), padding=(0, 2), dilation=(2, 1), groups=3),
    dict(stride=1, padding=1, dilation=1, groups=6),
])
def test_reference_conv2d_matches_textbook_definition(cfg):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 6, 9, 8, generator=g, dtype=torch.float64)
    w = torch.randn(6, 6 // cfg["groups"], 3, 2, generator=g, dtype=torch.float64)
    b = torch.randn(6, generator=g, dtype=torch.float64)
    attrs = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
    t = shapes.infer("conv2d", [ir.TensorType(x.dtype, x.shape), ir.TensorType(w.dtype, w.shape),
                                ir.TensorType(b.dtype, b.shape)], attrs)
    out = reference.IMPLS["conv2d"](attrs, [x, w, b], tuple(t.shape))
    assert rel_err(out, F.conv2d(x, w, b, **cfg)) < 1e-13


def test_broken_reference_op_is_caught(monkeypatch):
    """Guard: the model tests must fail if a reference op is wrong (unbiased variance in layer norm)."""
    model, shape, prog = imported("encoder_layer_relu", False)
    x = torch.randn(*shape)

    def wrong_layer_norm(attrs, args, out_shape):
        xx, w, b = args
        mu = xx.mean(dim=-1, keepdim=True)
        y = (xx - mu) / torch.sqrt(xx.var(dim=-1, unbiased=True, keepdim=True) + attrs["eps"])
        return y * w + b

    monkeypatch.setitem(reference.IMPLS, "layer_norm", wrong_layer_norm)
    assert rel_err(ir.run(prog, x)[0], torch_f64(model, x)) > TOL


def test_reference_checks_its_own_output_types(monkeypatch):
    """A kernel that returns the wrong shape is an error, never a silent result."""
    _, shape, prog = imported("resnet18", False)
    monkeypatch.setitem(reference.IMPLS, "adaptive_avg_pool2d", lambda a, args, s: args[0])
    with pytest.raises(ir.IRError, match="adaptive_avg_pool2d produced"):
        ir.run(prog, torch.randn(*shape))
