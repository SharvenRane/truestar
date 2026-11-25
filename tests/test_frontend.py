"""Tests for the native front end (@truestar.function): translation, shape errors, control flow."""

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import truestar
from truestar import Tensor, ops
from truestar import ir
from truestar.frontend import (Executor, FrontendError, FrontendShapeError, InputError, ReferenceExecutor,
                               StaticValueError, UnsupportedConstructError, compile_count)
from truestar.ir import reference

TOL = 1e-10
HERE = Path(__file__).read_text(encoding="utf-8").splitlines()


def line_of(marker: str) -> int:
    """Line number (1 based) of the unique line in this file that ends with the marker comment."""
    hits = [i + 1 for i, ln in enumerate(HERE) if ln.rstrip().endswith(f"# {marker}")]
    assert len(hits) == 1, f"marker {marker} found {len(hits)} times"
    return hits[0]


def rel_err(a, b) -> float:
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-300)).item()


def f64(*ts):
    return [t.double() for t in ts]


def check(fn, *inputs, tol=TOL):
    """Run the compiled function and the same source eagerly in PyTorch float64; they must agree."""
    got = fn(*inputs)
    want = fn.python(*[t.double() if isinstance(t, torch.Tensor) and t.is_floating_point() else t
                       for t in inputs])
    got = got if isinstance(got, tuple) else (got,)
    want = want if isinstance(want, tuple) else (want,)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert tuple(g.shape) == tuple(torch.as_tensor(w).shape)
        assert rel_err(g, w) < tol, f"{fn.name}: {rel_err(g, w):.3e}"
    return got


g = torch.Generator().manual_seed(0)


def rnd(*shape, scale=1.0):
    return torch.randn(*shape, generator=g) * scale


# ----------------------------------------------------------------------------- straight line code


@truestar.function
def mlp(x: Tensor["B", 16, "f32"], w1: Tensor[16, 32], b1: Tensor[32], w2: Tensor[32, 8]) -> Tensor["B", 8]:
    h = ops.gelu(x @ w1 + b1)
    h = ops.layer_norm(h, [32], eps=1e-5)
    return ops.softmax(h @ w2, dim=-1)


def test_mlp_matches_pytorch_and_hand_written_reference():
    x, w1, b1, w2 = rnd(4, 16), rnd(16, 32), rnd(32), rnd(32, 8)
    got = check(mlp, x, w1, b1, w2)[0]
    xd, w1d, b1d, w2d = f64(x, w1, b1, w2)
    want = torch.softmax(F.layer_norm(F.gelu(xd @ w1d + b1d), [32], eps=1e-5) @ w2d, dim=-1)
    assert rel_err(got, want) < TOL
    assert str(mlp.signature) == "mlp(x: f32[B, 16], w1: f32[16, 32], b1: f32[32], w2: f32[32, 8]) -> f32[B, 8]"


@truestar.function
def elementwise_zoo(x: Tensor["B", 6], y: Tensor["B", 6]):
    a = ops.relu(x) * 2.0 - y / 3 + 1
    b = ops.exp(-ops.abs(x)) + ops.sigmoid(y) + ops.tanh(x) + ops.silu(y)
    c = ops.sqrt(ops.abs(x) + 1.0) + ops.rsqrt(y * y + 1.0) + x ** 2 + ops.clamp(y, -0.5, 0.5)
    d = ops.maximum(x, y) - ops.minimum(x, 0.0) + ops.where(x > y, x, y * 0.5)
    m = x.mean(dim=1, keepdim=True) + y.sum(1, keepdim=True) / x.shape[1]
    return a + b + c + d - m, (x > 0).any(dim=1), (x > -10.0).all()


def test_elementwise_and_reductions_match_pytorch():
    check(elementwise_zoo, rnd(5, 6), rnd(5, 6))


@truestar.function
def shapes_zoo(x: Tensor["B", 4, 6], w: Tensor[6, 4]):
    y = x.transpose(1, 2)                 # [B, 6, 4]
    z = y.reshape(x.shape[0], 24)
    p = x.permute(0, 2, 1) * ops.transpose(x, 1, 2)
    c = ops.cat([x, x * 2.0], dim=2)      # [B, 4, 12]
    s = c[:, 1:3, ::2] + c[:, 0:2, 1::2]  # [B, 2, 6]
    first = x[:, 0] @ w                   # [B, 4]
    e = x[..., -1].unsqueeze(1).squeeze(1)
    lin = ops.linear(x, w.T, None)        # [B, 4, 4]
    return z, p, s, first, e, lin, x[0].sum() + x.mT.sum()


def test_shape_ops_and_indexing_match_pytorch():
    check(shapes_zoo, rnd(3, 4, 6), rnd(6, 4))


@truestar.function
def attn(q: Tensor["B", 2, "L", 8], k: Tensor["B", 2, "L", 8], v: Tensor["B", 2, "L", 8]):
    causal = ops.attention(q, k, v, is_causal=True)
    return causal + torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(8), dim=-1) @ v


def test_attention_and_torch_aliases_match_pytorch():
    for L in (3, 5):
        check(attn, rnd(2, 2, L, 8), rnd(2, 2, L, 8), rnd(2, 2, L, 8))
    assert attn.compile_count == 1


# ----------------------------------------------------------------------------- static control flow


@truestar.function
def static_branches(x: Tensor["B", 128], scale: float = 2.0, depth: int = 3, act: str = "relu"):
    if x.shape[1] == 128 and depth > 0:      # shape fact and static args: decided at compile time
        x = x * scale
    else:
        x = x + undefined_name_in_dead_branch  # noqa: F821 (never translated)
    for i in range(depth):                   # static trip count: unrolled
        x = x + float(i)
    for s in (0.5, 0.25):
        x = x * s
    x = ops.relu(x) if act == "relu" else ops.tanh(x)
    return x


def test_static_if_and_unrolled_for():
    x = rnd(3, 128)
    check(static_branches, x)
    ops_in_ir = static_branches.program.op_counts()
    assert "cond" not in ops_in_ir and "loop" not in ops_in_ir
    assert ops_in_ir["add"] == 3 and ops_in_ir["relu"] == 1


def test_static_args_compile_once_per_value():
    x = rnd(2, 128)
    before = static_branches.compile_count
    for _ in range(3):
        out = static_branches(x, act="tanh", depth=1)
        assert rel_err(out, static_branches.python(x.double(), act="tanh", depth=1)) < TOL
    assert static_branches.compile_count == before + 1
    assert "tanh" in static_branches.specialize(act="tanh", depth=1).program.op_counts()


# ----------------------------------------------------------------------------- data dependent control flow


@truestar.function
def data_branch(x: Tensor["B", 8], w: Tensor[8, 4]) -> Tensor["B", 4]:
    h = x @ w
    if h.sum() > 0:
        h = ops.relu(h)
        k = 1.0
    else:
        h = -h * 0.5
        k = 3.0
    return h * k


@truestar.function
def early_return(x: Tensor["N"]):
    if x.mean() > 0:
        return x * 2.0, x.sum()
    return -x, x.mean()


@truestar.function
def select_lowering(x: Tensor["B", 4], y: Tensor["B", 4]):
    return x if x.sum() > y.sum() else y * 2.0


@truestar.function
def nested_branches(x: Tensor["B", 4]):
    s = x.sum()
    if s > 0:
        if x[:, 0].sum() > 0:
            y = x + 1.0
        else:
            y = x - 1.0
    else:
        y = x * 0.0
    return y


def test_data_dependent_if_both_branches():
    w = rnd(8, 4)
    pos = torch.ones(2, 8)
    for x in (pos, -pos):
        h = x.double() @ w.double()
        taken = bool(h.sum() > 0)
        out = check(data_branch, x, w)[0]
        assert rel_err(out, F.relu(h) if taken else -h * 1.5) < TOL
    seen = set()
    for i in range(12):
        x = rnd(3, 8)
        seen.add(bool((x @ w).sum() > 0))
        check(data_branch, x, w)
    assert seen == {True, False}
    assert data_branch.program.op_counts()["cond"] == 1
    assert data_branch.compile_count == 1


def test_early_return_both_paths():
    check(early_return, torch.tensor([1.0, 2.0, -0.5]))
    check(early_return, torch.tensor([-1.0, -2.0, 0.5]))
    assert early_return.program.op_counts()["cond"] == 1


def test_ternary_on_tensor_data_lowers_to_select():
    assert "where" in select_lowering.program.op_counts()
    assert "cond" not in select_lowering.program.op_counts()
    a, b = rnd(3, 4), rnd(3, 4)
    check(select_lowering, a, b)
    check(select_lowering, a + 10, b)
    check(select_lowering, a - 10, b)


def test_nested_data_dependent_branches():
    for x in (torch.ones(2, 4), -torch.ones(2, 4), torch.tensor([[-1.0, 3, 3, 3], [-1.0, 3, 3, 3]])):
        check(nested_branches, x)


# ----------------------------------------------------------------------------- loops


@truestar.function
def rnn(x: Tensor["B", "T", 8], wx: Tensor[8, 16], wh: Tensor[16, 16]) -> Tensor["B", 16]:
    h = ops.zeros((x.shape[0], 16), dtype=x.dtype)
    for t in range(x.shape[1]):
        h = ops.tanh(x[:, t] @ wx + h @ wh)
    return h


@truestar.function
def loop_with_branch(x: Tensor["T", 4]):
    acc = ops.zeros((4,))
    count = 0
    for t in range(1, x.shape[0]):
        row = x[t] - x[t - 1]
        if row.sum() > 0:
            acc = acc + row
        else:
            acc = acc - row * 0.5
        count = count + 1
    return acc, count


@truestar.function
def halve_until_small(x: Tensor["N"]):
    steps = 0
    while x.abs().sum() > 1.0:
        x = x * 0.5
        steps = steps + 1
    return x, steps


@truestar.function
def newton_sqrt(a: Tensor["N"]):
    y = a * 0.0 + 1.0
    err = ((y * y - a) / a).abs().sum()
    while err > 1e-13:
        y = 0.5 * (y + a / y)
        err = ((y * y - a) / a).abs().sum()
    return y


def test_symbolic_for_loop_matches_pytorch():
    wx, wh = rnd(8, 16), rnd(16, 16, scale=0.2)
    for B, T in ((1, 1), (2, 5), (3, 9)):
        x = rnd(B, T, 8)
        out = check(rnn, x, wx, wh)[0]
        h = torch.zeros(B, 16, dtype=torch.float64)
        for t in range(T):
            h = torch.tanh(x[:, t].double() @ wx.double() + h @ wh.double())
        assert rel_err(out, h) < TOL
    assert rnn.program.op_counts() == {"full": 1, "loop": 1}
    assert rnn.compile_count == 1


def test_loop_with_offset_start_and_inner_branch():
    for T in (1, 2, 6):
        check(loop_with_branch, rnd(T, 4))


def test_while_loop_on_tensor_data():
    for x in (torch.tensor([0.1, 0.2]), torch.tensor([10.0, -30.0, 5.0]), torch.tensor([100.0] * 7)):
        got = check(halve_until_small, x)
        assert int(got[1]) == halve_until_small.python(x.double())[1]
    a = torch.tensor([2.0, 9.0, 0.25, 1e4])
    assert rel_err(check(newton_sqrt, a)[0], torch.sqrt(a.double())) < 1e-12


# ----------------------------------------------------------------------------- composition


@truestar.function
def block(x: Tensor["B", 16], w: Tensor[16, 16]) -> Tensor["B", 16]:
    return x + ops.relu(x @ w)


@truestar.function
def two_blocks(x: Tensor["B", 16], w1: Tensor[16, 16], w2: Tensor[16, 16]):
    y = block(block(x, w1), w2)
    if y.mean() > 0:
        y = block(y, w1)
    return y


def test_calling_another_native_function_inlines_it():
    x, w1, w2 = rnd(3, 16), rnd(16, 16, scale=0.3), rnd(16, 16, scale=0.3)
    for sign in (1.0, -1.0):
        xs = x * sign + sign
        xd, w1d, w2d = f64(xs, w1, w2)
        blk = lambda a, w: a + torch.relu(a @ w)  # noqa: E731
        y = blk(blk(xd, w1d), w2d)
        want = blk(y, w1d) if y.mean() > 0 else y
        assert rel_err(two_blocks(xs, w1, w2), want) < TOL
    counts = two_blocks.program.op_counts()
    assert counts["matmul"] == 2 and counts["cond"] == 1


# ----------------------------------------------------------------------------- one compile, every size


def test_one_compile_serves_every_batch_size():
    @truestar.function
    def layer(x: Tensor["B", 16], w: Tensor[16, 8]) -> Tensor["B", 8]:
        h = x @ w
        if h.mean() > 0:
            h = ops.relu(h)
        return h

    total_before = compile_count()
    assert layer.compile_count == 1
    w = rnd(16, 8)
    for B in (1, 2, 3, 7, 64, 5, 1):
        check(layer, rnd(B, 16), w)
    assert layer.compile_count == 1
    assert compile_count() == total_before

    wx, wh = rnd(8, 16), rnd(16, 16, scale=0.2)
    before = rnn.compile_count
    for B, T in ((1, 3), (4, 1), (2, 17), (8, 4)):
        check(rnn, rnd(B, T, 8), wx, wh)
    assert rnn.compile_count == before == 1


# ----------------------------------------------------------------------------- shape errors


def test_matmul_shape_error_names_line_op_shapes_and_fix():
    with pytest.raises(FrontendShapeError) as ei:
        @truestar.function
        def bad(x: Tensor["B", 128, "f32"], w: Tensor[64, 128, "f32"]):
            h = ops.relu(x)
            return h @ w  # E:matmul
    e = ei.value
    assert e.line == line_of("E:matmul")
    assert e.file.endswith("test_frontend.py") and e.function == "bad"
    assert e.op == "matmul" and e.code == "shape_mismatch"
    assert [str(t) for _, t in e.operands] == ["f32[B, 128]", "f32[64, 128]"]
    assert e.suggestion == "w looks transposed: try h @ w.T"
    text = str(e)
    assert f"test_frontend.py:{e.line} in bad" in text and "return h @ w" in text
    d = e.to_dict()
    json.dumps(d)
    assert d["line"] == e.line and d["op"] == "matmul" and d["code"] == "shape_mismatch"
    assert d["operands"][0] == {"name": "h", "type": "f32[B, 128]", "dtype": "f32", "shape": ["B", "128"]}
    assert d["suggestion"].endswith("h @ w.T") and d["schema"] == "truestar.frontend.error/1"


def test_broadcast_error_and_untaken_branch_checked_at_definition():
    with pytest.raises(FrontendShapeError) as ei:
        @truestar.function
        def bad(x: Tensor["B", 4], y: Tensor[4, "B"]):
            if x.sum() > 1e30:  # practically never true at run time, still checked now
                x = x + y  # E:bcast
            return x
    e = ei.value
    assert e.line == line_of("E:bcast") and e.op == "add"
    assert "transposes of each other" in e.suggestion


def test_symbolic_dims_must_match_exactly():
    with pytest.raises(FrontendShapeError) as ei:
        @truestar.function
        def bad(x: Tensor["B", 8], y: Tensor["C", 8]):
            return x - y  # E:symbolic
    assert ei.value.line == line_of("E:symbolic")
    assert "provably equal" in ei.value.message


def test_shape_error_inside_symbolic_loop_body():
    with pytest.raises(FrontendShapeError) as ei:
        @truestar.function
        def bad(x: Tensor["B", "T", 8], wx: Tensor[8, 16], wh: Tensor[8, 16]):
            h = ops.zeros((x.shape[0], 16))
            for t in range(x.shape[1]):
                h = ops.tanh(x[:, t] @ wx + h @ wh)  # E:loopbody
            return h
    e = ei.value
    assert e.line == line_of("E:loopbody") and e.op == "matmul"
    assert [str(t) for _, t in e.operands] == ["f32[B, 16]", "f32[8, 16]"]


def test_loop_carried_type_change_is_an_error():
    with pytest.raises(FrontendShapeError) as ei:
        @truestar.function
        def bad(x: Tensor["B", "T", 8]):
            h = ops.zeros((x.shape[0], 8))
            for t in range(x.shape[1]):  # E:carried
                h = ops.cat([h, x[:, t]], dim=1)
            return h
    e = ei.value
    assert e.line == line_of("E:carried") and e.code == "loop_type_change"
    assert [str(t) for _, t in e.operands] == ["f32[B, 8]", "f32[B, 16]"]


def test_branch_type_mismatch_is_an_error():
    with pytest.raises(FrontendShapeError) as ei:
        @truestar.function
        def bad(x: Tensor["B", 8]):
            if x.sum() > 0:  # E:branches
                y = x
            else:
                y = x.sum(dim=1)
            return y
    e = ei.value
    assert e.line == line_of("E:branches") and e.code == "branch_type_mismatch"
    assert [str(t) for _, t in e.operands] == ["f32[B, 8]", "f32[B]"]


def test_return_annotation_and_annotated_assignment_are_checked():
    with pytest.raises(FrontendShapeError) as ei:
        @truestar.function
        def bad(x: Tensor["B", 8], w: Tensor[8, 4]) -> Tensor["B", 8]:
            return x @ w  # E:retann
    assert ei.value.line == line_of("E:retann") and ei.value.code == "annotation_mismatch"
    with pytest.raises(FrontendShapeError) as ei:
        @truestar.function
        def bad2(x: Tensor["B", 8], w: Tensor[8, 4]):
            h: Tensor["B", 8] = x @ w  # E:annassign
            return h
    assert ei.value.line == line_of("E:annassign")


def test_linear_and_layer_norm_errors_suggest_a_fix():
    with pytest.raises(FrontendShapeError) as ei:
        @truestar.function
        def bad(x: Tensor["B", 16], w: Tensor[16, 4]):
            return ops.linear(x, w)  # E:linear
    assert ei.value.line == line_of("E:linear") and "[out, in]" in ei.value.suggestion
    with pytest.raises(FrontendShapeError) as ei:
        @truestar.function
        def bad2(x: Tensor["B", 16]):
            return ops.layer_norm(x, [8])  # E:ln
    assert ei.value.line == line_of("E:ln") and "[16]" in ei.value.suggestion


def test_condition_on_non_scalar_tensor_is_an_error():
    with pytest.raises(FrontendShapeError) as ei:
        @truestar.function
        def bad(x: Tensor["B", 8]):
            if x > 0:  # E:ambiguous
                x = -x
            return x
    e = ei.value
    assert e.line == line_of("E:ambiguous") and e.code == "ambiguous_condition"
    assert ".any()" in e.suggestion


def test_branch_on_symbolic_dim_is_not_silently_specialized():
    with pytest.raises(StaticValueError) as ei:
        @truestar.function
        def bad(x: Tensor["B", 8]):
            if x.shape[0] == 1:  # E:symbranch
                return x * 2.0
            return x
    assert ei.value.line == line_of("E:symbranch") and "B" in ei.value.message


def test_undefined_after_one_sided_branch_or_loop():
    with pytest.raises(FrontendError) as ei:
        @truestar.function
        def bad(x: Tensor["B", 8]):
            if x.sum() > 0:
                y = x
            return y  # E:onesided
    assert ei.value.line == line_of("E:onesided") and ei.value.code == "undefined_name"


# ----------------------------------------------------------------------------- unsupported constructs


UNSUPPORTED = {
    "try": ("try blocks", "U:try"),
    "lambda": ("lambda", "U:lambda"),
    "listcomp": ("list comprehensions", "U:listcomp"),
    "break": ("break", "U:break"),
    "setitem": ("assignment to a subscript", "U:setitem"),
    "print": ("calls to print", "U:print"),
    "loop_return": ("return inside a loop", "U:loopret"),
    "iter_tensor": ("iterating over a tensor", "U:itertensor"),
    "fstring": ("f strings", "U:fstring"),
}


def _make_unsupported(kind):
    if kind == "try":
        def f(x: Tensor["B", 4]):
            try:  # U:try
                y = x
            except Exception:
                y = x
            return y
    elif kind == "lambda":
        def f(x: Tensor["B", 4]):
            g = lambda t: t  # noqa: E731  # U:lambda
            return x
    elif kind == "listcomp":
        def f(x: Tensor["B", 4]):
            ys = [x for _ in range(2)]  # U:listcomp
            return x
    elif kind == "break":
        def f(x: Tensor["B", 4]):
            for i in range(3):
                break  # U:break
            return x
    elif kind == "setitem":
        def f(x: Tensor["B", 4]):
            x[0] = 1.0  # U:setitem
            return x
    elif kind == "print":
        def f(x: Tensor["B", 4]):
            print(x)  # U:print
            return x
    elif kind == "loop_return":
        def f(x: Tensor["T", 4]):
            for t in range(x.shape[0]):
                return x  # U:loopret
            return x
    elif kind == "iter_tensor":
        def f(x: Tensor["T", 4]):
            for row in x:  # U:itertensor
                x = x + row
            return x
    elif kind == "fstring":
        def f(x: Tensor["T", 4]):
            s = f"{x}"  # U:fstring
            return x
    return f


@pytest.mark.parametrize("kind", sorted(UNSUPPORTED))
def test_unsupported_construct_fails_loudly_with_line(kind):
    what, marker = UNSUPPORTED[kind]
    with pytest.raises(UnsupportedConstructError) as ei:
        truestar.function(_make_unsupported(kind))
    e = ei.value
    assert e.line == line_of(marker), str(e)
    assert what in e.message, e.message
    assert e.to_dict()["code"] == "unsupported_construct"


# ----------------------------------------------------------------------------- calling


def test_call_time_input_errors():
    w = rnd(8, 4)
    with pytest.raises(InputError, match="dtype"):
        data_branch(rnd(2, 8).double(), w)
    with pytest.raises(InputError, match="does not match"):
        data_branch(rnd(2, 9), w)
    with pytest.raises(InputError):
        data_branch(rnd(2, 8), "not a tensor")


def test_executor_seam_is_pluggable():
    calls = {"prepare": 0, "run": 0}

    class Counting(Executor):
        name = "counting"

        def prepare(self, program):
            calls["prepare"] += 1
            inner = ReferenceExecutor().prepare(program)

            def run(inputs):
                calls["run"] += 1
                return inner(inputs)
            return run

    @truestar.function(executor=Counting())
    def f(x: Tensor["B", 4]):
        return ops.relu(x) + 1.0

    for B in (1, 3, 6):
        check(f, rnd(B, 4))
    assert calls == {"prepare": 1, "run": 3}
    assert f.executor.name == "counting"


def test_lazy_compile_and_symbol_ranges():
    @truestar.function(dims={"B": (1, 4)}, eager=False)
    def f(x: Tensor["B", 4]):
        return x * 2.0

    assert f.compile_count == 0
    f(rnd(4, 4))
    assert f.compile_count == 1
    with pytest.raises(InputError, match="outside its range"):
        f(rnd(5, 4))


# ----------------------------------------------------------------------------- IR control flow ops


def test_ir_cond_and_loop_built_by_hand_and_verified():
    then = ir.Builder("then")
    t_in = then.input("x", torch.float32, ("N",))
    then_p = then.build([then.op("relu", t_in)])
    other = ir.Builder("else")
    e_in = other.input("x", torch.float32, ("N",))
    else_p = other.build([other.op("neg", e_in)])
    b = ir.Builder("p")
    x = b.input("x", torch.float32, ("N",))
    s = b.op("sum", x, dims=[], keepdim=False)
    zero = b.param("zero", torch.tensor(0.0))
    pred = b.op("gt", s, zero)
    (y,) = b.op_multi("cond", pred, x, then_branch=then_p, else_branch=else_p)
    prog = b.build([y])
    assert "cond(" in prog.to_text() and "then_branch(" in prog.to_text()
    v = torch.tensor([1.0, -2.0, 3.0])
    assert torch.equal(ir.run(prog, v)[0], torch.relu(v.double()))
    assert torch.equal(ir.run(prog, -v)[0], v.double())

    bad = ir.Builder("else")
    bad_in = bad.input("x", torch.float32, ("N",))
    bad_p = bad.build([bad.op("sum", bad_in, dims=[], keepdim=False)])
    with pytest.raises(ir.ShapeError, match="then_branch but"):
        b.op_multi("cond", pred, x, then_branch=then_p, else_branch=bad_p)
    with pytest.raises(ir.ShapeError, match="exactly one"):
        b.op_multi("loop", x, n_carried=1, body=then_p)


def test_reference_checks_loop_types_and_select_bounds():
    body = ir.Builder("body")
    i = body.input("i", torch.int64, ())
    h = body.input("h", torch.float32, ("N",))
    body_p = body.build([body.op("add", h, body.param("one", torch.tensor(1.0)))])
    b = ir.Builder("p")
    x = b.input("x", torch.float32, ("N", "T"))
    (y,) = b.op_multi("loop", b.op("select", x, dim=1, index=0), n_carried=1, body=body_p,
                      trip_count=ir.symbol("T") - 1)
    prog = b.build([y])
    v = torch.zeros(3, 4)
    assert torch.equal(ir.run(prog, v)[0], torch.full((3,), 3.0, dtype=torch.float64))
    assert torch.equal(ir.run(prog, torch.zeros(3, 1))[0], torch.zeros(3, dtype=torch.float64))
    del i


# ----------------------------------------------------------------------------- tests that prove the tests work


def test_broken_cond_reference_is_caught(monkeypatch):
    """Guard: if cond always took the then branch, the both branch comparisons must fail."""
    monkeypatch.setitem(reference.IMPLS, "cond",
                        lambda attrs, args, shapes, bindings=None: reference._run(attrs["then_branch"], args[1:], bindings))
    with pytest.raises(AssertionError):
        check(data_branch, -torch.ones(2, 8), torch.ones(8, 4))


def test_broken_loop_reference_is_caught(monkeypatch):
    """Guard: a loop that stops one iteration early must fail the RNN comparison."""
    real = reference.IMPLS["loop"]

    def short(attrs, args, shapes, bindings=None):
        attrs = dict(attrs, trip_count=attrs["trip_count"] - 1) if attrs.get("trip_count") is not None else attrs
        return real(attrs, args, shapes, bindings=bindings)

    monkeypatch.setitem(reference.IMPLS, "loop", short)
    with pytest.raises(AssertionError):
        check(rnn, rnd(2, 4, 8), rnd(8, 16), rnd(16, 16, scale=0.2))
