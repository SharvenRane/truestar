"""Nested inputs and outputs in the native front end, and truestar.grad over them.

Structures: dict, list and tuple annotations, dataclasses and NamedTuples (no registration),
returned dicts and structures, mismatches named by leaf path, export round trip of the program, and
gradients with the structure of their parameter, through loops and branches.
"""

import dataclasses
from typing import NamedTuple

import pytest
import torch

import truestar
from truestar import Tensor, ops
from truestar.frontend import FrontendError, InputError

F64 = torch.float64


@dataclasses.dataclass
class Params:
    w: Tensor[4, 3]
    b: Tensor[3]


class Out(NamedTuple):
    y: Tensor["B", 3]
    norm: Tensor[()]


@truestar.function
def layer(p: Params, x: Tensor["B", 4], extra: {"scale": Tensor[()], "shift": [Tensor[3], Tensor[3]]}) -> Out:
    h = x @ p.w + p.b
    h = h * extra["scale"] + extra["shift"][0] - extra["shift"][1]
    for k, v in {"a": h, "b": h * 2.0}.items():
        h = h + v
    return Out(y=ops.tanh(h), norm=(h * h).sum())


def layer_torch(p, x, extra):
    h = x @ p.w + p.b
    h = h * extra["scale"] + extra["shift"][0] - extra["shift"][1]
    h = h + h + h * 2.0  # the dict values are taken before the loop: h0 + h0 + 2 h0
    return torch.tanh(h), (h * h).sum()


def _args(B=5, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=dtype)  # noqa: E731
    return (Params(r(4, 3) * 0.3, r(3)), r(B, 4),
            {"scale": torch.tensor(0.5, dtype=dtype), "shift": [r(3), r(3)]})


def test_structured_arguments_and_result():
    p, x, ex = _args()
    out = layer(p, x, ex)
    assert type(out) is Out
    p64 = Params(p.w.double(), p.b.double())
    ex64 = {"scale": ex["scale"].double(), "shift": [t.double() for t in ex["shift"]]}
    y, n = layer_torch(p64, x.double(), ex64)
    assert (out.y - y).abs().max() < 1e-12 and abs(out.norm - n) / n < 1e-12
    sig = str(layer.signature)
    assert "p.w: f32[4, 3]" in sig and "extra['shift'][1]: f32[3]" in sig and "Out(y=T, norm=T)" in sig
    assert [v.name for v in layer.program.inputs] == ["p_w", "p_b", "x", "extra_scale", "extra_shift_0", "extra_shift_1"]


@pytest.mark.parametrize("change,message", [
    (lambda p, x, e: (p, x, {"scale": e["scale"], "shfit": e["shift"]}),
     "extra['shift']: missing key 'shift'; unexpected key 'shfit' (expected keys: 'scale', 'shift')"),
    (lambda p, x, e: (p, x, {"scale": e["scale"], "shift": e["shift"][:1]}), "extra['shift']: expected 2 items, got 1"),
    (lambda p, x, e: ({"w": p.w, "b": p.b}, x, e), "p: expected a Params, got a dict"),
    (lambda p, x, e: (Params(p.w.double(), p.b), x, e), "p.w must have dtype f32"),
    (lambda p, x, e: (Params(p.w, torch.zeros(4)), x, e), "p.b shape does not match"),
    (lambda p, x, e: (p, x, {"scale": e["scale"], "shift": [e["shift"][0], None]}), "extra['shift'][1]: expected a tensor, got None"),
])
def test_mismatch_names_the_leaf(change, message):
    with pytest.raises(InputError) as err:
        layer(*change(*_args()))
    assert message in str(err.value)


def test_returned_dict_and_annotation_check():
    @truestar.function
    def heads(x: Tensor["B", 4]) -> {"mean": Tensor["B"], "max": Tensor["B"]}:
        return {"mean": x.mean(-1), "max": ops.amax(x, -1)}

    out = heads(torch.arange(8.0).reshape(2, 4))
    assert list(out) == ["mean", "max"] and out["max"].tolist() == [3.0, 7.0]

    with pytest.raises(FrontendError, match="declares {'mean': T} but the function returns {'avg': T}"):
        @truestar.function
        def renamed(x: Tensor["B", 4]) -> {"mean": Tensor["B"]}:
            return {"avg": x.mean(-1)}


def test_structure_not_made_of_tensors_is_refused_at_decoration():
    with pytest.raises(FrontendError, match="every leaf of a structured parameter must be annotated Tensor"):
        @truestar.function
        def mixed(x: {"a": Tensor[3], "n": 3}) -> Tensor[3]:
            return x["a"]


def test_structured_program_exports_and_round_trips(tmp_path):
    from truestar.ir import serialize
    from truestar import ir
    blob = serialize.dumps(layer.program)
    back = serialize.loads(blob).program
    p, x, ex = _args(B=3)
    flat = [p.w, p.b, x, ex["scale"], *ex["shift"]]
    for a, b in zip(ir.run(back, *flat), ir.run(layer.program, *flat)):
        assert torch.equal(a, b)


# ----------------------------------------------------------------------------- truestar.grad


@truestar.function
def rnn_loss(params: {"wx": Tensor[4, 5], "wh": Tensor[5, 5]}, x: Tensor["B", "T", 4], y: Tensor["B", 5]) -> Tensor[()]:
    h = ops.zeros((x.shape[0], 5))
    for t in range(x.shape[1]):
        h = ops.tanh(x[:, t] @ params["wx"] + h @ params["wh"])
    if h.mean() > 0:
        h = h * 2.0
    return ((h - y) * (h - y)).mean()


def rnn_loss_torch(params, x, y):
    h = torch.zeros(x.shape[0], 5, dtype=x.dtype)
    for t in range(x.shape[1]):
        h = torch.tanh(x[:, t] @ params["wx"] + h @ params["wh"])
    if h.mean() > 0:
        h = h * 2.0
    return ((h - y) ** 2).mean()


@pytest.mark.parametrize("T", [1, 4, 9])
@pytest.mark.parametrize("memory", ["auto", "recompute", "sqrt"])
def test_grad_has_the_structure_of_its_parameter(T, memory):
    g = torch.Generator().manual_seed(T)
    params = {"wx": torch.randn(4, 5, generator=g) * 0.5, "wh": torch.randn(5, 5, generator=g) * 0.4}
    x, y = torch.randn(3, T, 4, generator=g), torch.randn(3, 5, generator=g)
    loss, grads = truestar.grad(rnn_loss, wrt="params", loop_memory=memory)(params, x, y)
    p64 = {k: v.double().requires_grad_() for k, v in params.items()}
    want = rnn_loss_torch(p64, x.double(), y.double())
    want.backward()
    assert abs(loss.item() - want.item()) < 1e-12
    assert set(grads) == {"wx", "wh"}
    for k in grads:
        assert (grads[k] - p64[k].grad).abs().max().item() < 1e-12


def test_grad_with_respect_to_several_arguments_and_a_dataclass():
    @truestar.function
    def loss(p: Params, x: Tensor["B", 4]) -> Tensor[()]:
        return ops.tanh(x @ p.w + p.b).sum()

    p, x, _ = _args(dtype=torch.float32)
    value, (gp, gx) = truestar.grad(loss, wrt=["p", "x"])(p, x)
    assert isinstance(gp, Params) and gx.shape == x.shape
    w, b, xx = (t.double().requires_grad_() for t in (p.w, p.b, x))
    torch.tanh(xx @ w + b).sum().backward()
    assert (gp.w - w.grad).abs().max() < 1e-12 and (gp.b - b.grad).abs().max() < 1e-12
    assert (gx - xx.grad).abs().max() < 1e-12


def test_grad_needs_a_scalar_and_known_names():
    with pytest.raises(InputError, match="tensor parameters of rnn_loss are"):
        truestar.grad(rnn_loss, wrt="weights")
    with pytest.raises(InputError, match="return a float scalar first"):
        truestar.grad(layer, wrt="x")(*_args())
