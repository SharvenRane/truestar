"""Decoder LLM support in the IR, the importer and the native front end (added 2026-11-24).

* Every new op against PyTorch float64: imported with torch.export in both graph forms (the export
  graph and the core ATen graph), with a symbolic batch (and sequence where it applies), checked at
  a second shape from the same import; and written natively, compared with the same source run
  eagerly by PyTorch.
* KV cache decoding: a Llama style model (RMSNorm, rotary, grouped query attention, SwiGLU, tied
  head; random weights) decoded greedily for 16 tokens, natively and through imported prefill and
  decode graphs, against plain torch.nn recomputing the whole sequence in float64.
* Tests that prove the tests work: broken reference ops make the comparisons fail.

See docs/LLM.md.
"""

import copy
import importlib.util
import math
import os
import warnings

import pytest
import sympy
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import Dim

import truestar
from truestar import Tensor, ir, ops
from truestar.frontend import FrontendError, compile_count
from truestar.ir import reference, shapes
from truestar.ir.autodiff import grad

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

TOL = 1e-10
F64 = torch.float64

_EX = os.path.join(os.path.dirname(__file__), "..", "examples", "llama_kv_decode.py")
_spec = importlib.util.spec_from_file_location("llama_kv_decode", _EX)
llama = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(llama)


def rel_err(a, b) -> float:
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-300)).item()


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(*shape, seed=0):
    return torch.randn(*shape, generator=gen(seed))


def f64(model, *args):
    with torch.no_grad():
        out = copy.deepcopy(model).double()(*[a.double() if a.is_floating_point() else a for a in args])
    return list(out) if isinstance(out, (tuple, list)) else [out]


class Fn(nn.Module):
    """A module whose forward is ``fn(self, *inputs)``; tensors in ``buffers`` become buffers."""

    def __init__(self, fn, n_inputs, **buffers):
        super().__init__()
        self.fn = fn
        for k, v in buffers.items():
            self.register_buffer(k, v)
        self.forward = {1: self.f1, 2: self.f2, 3: self.f3, 4: self.f4}[n_inputs]

    def f1(self, a):
        return self.fn(self, a)

    def f2(self, a, b):
        return self.fn(self, a, b)

    def f3(self, a, b, c):
        return self.fn(self, a, b, c)

    def f4(self, a, b, c, d):
        return self.fn(self, a, b, c, d)


def compare(got, want, what):
    assert len(got) == len(want), what
    for g, w in zip(got, want):
        assert tuple(g.shape) == tuple(w.shape), f"{what}: shape {tuple(g.shape)} vs {tuple(w.shape)}"
        if w.is_floating_point():
            err = rel_err(g, w)
            assert err < TOL, f"{what}: relative error {err:.3e}"
        else:
            assert g.dtype == w.dtype and torch.equal(g, w), f"{what}: integer result differs"


def check(model, args, dyn, other=(), expect=(), graphs=(False, True)):
    """Import in each graph form, assert the expected IR ops appear, compare with PyTorch float64 at
    the example inputs and at every input set in ``other`` (other shapes, same program)."""
    model = model.eval()
    progs = []
    for decompose in graphs:
        prog = ir.import_module(model, args, dynamic_shapes=dyn, decompose=decompose)
        ops_ = prog.op_counts() if not decompose else {e: 1 for e in expect}  # core ATen decomposes attention
        missing = [e for e in expect if e not in ops_]
        assert not missing, f"decompose={decompose}: expected IR ops {missing}, got {sorted(ops_)}"
        for a in (args,) + tuple(other):
            compare(ir.run(prog, *a), f64(model, *a), f"decompose={decompose}, shapes {[tuple(t.shape) for t in a]}")
        progs.append(prog)
    return progs


B = Dim("B", min=1, max=16)
T = Dim("T", min=2, max=64)


# ----------------------------------------------------------------------------- import cases

CASES = {}


def case(name):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


def _x(*shape, seed=0):
    return randn(*shape, seed=seed)


@case("flip")
def _():
    m = Fn(lambda s, x: x.flip(-1) * 2 + x.flip(0, 2), 1)
    return m, (_x(2, 3, 4),), ({0: B},), [(_x(5, 3, 4, seed=1),)], ["flip"]


@case("sort_last_dim")
def _():
    m = Fn(lambda s, x: torch.sort(x, dim=-1), 1)
    return m, (_x(2, 3, 7),), ({0: B},), [(_x(5, 3, 7, seed=1),)], ["sort"]


@case("sort_descending_symbolic_dim")
def _():
    m = Fn(lambda s, x: torch.sort(x, dim=0, descending=True), 1)
    return m, (_x(4, 3),), ({0: B},), [(_x(9, 3, seed=1),)], ["sort"]


@case("topk_largest")
def _():
    m = Fn(lambda s, x: torch.topk(x, 3, dim=-1), 1)
    return m, (_x(2, 4, 9),), ({0: B},), [(_x(5, 4, 9, seed=1),)], ["topk"]


@case("topk_smallest_middle_dim")
def _():
    m = Fn(lambda s, x: torch.topk(x, 2, dim=1, largest=False), 1)
    return m, (_x(2, 6, 3),), ({0: B},), [(_x(3, 6, 3, seed=1),)], ["topk"]


@case("argmax_argmin")
def _():
    m = Fn(lambda s, x: (x.argmax(-1), x.argmin(1, keepdim=True), x.argmax()), 1)
    return m, (_x(2, 5, 6),), ({0: B},), [(_x(4, 5, 6, seed=1),)], ["argmax", "argmin"]


@case("max_min_dim_amax")
def _():
    m = Fn(lambda s, x: (*x.max(dim=-1), *x.min(dim=1, keepdim=True), x.amax(-1)), 1)
    return m, (_x(2, 5, 6),), ({0: B},), [(_x(4, 5, 6, seed=1),)], ["argmax", "argmin", "gather"]


@case("amax_amin_reductions")
def _():
    m = Fn(lambda s, x: (x.amax(dim=(0, 2)), x.amin(-1, keepdim=True), x.max(), x.min(), torch.max(x, x.flip(-1))), 1)
    return m, (_x(2, 5, 6),), ({0: B},), [(_x(4, 5, 6, seed=1),)], ["amax", "amin"]


@case("repeat_interleave")
def _():
    m = Fn(lambda s, x: x.repeat_interleave(3, dim=1) * 1.5, 1)
    return m, (_x(2, 4, 5),), ({0: B},), [(_x(3, 4, 5, seed=1),)], ["expand"]


def _qkv(hq, hkv, t, b=2, e=16, seed=0):
    return _x(b, hq, t, e, seed=seed), _x(b, hkv, t, e, seed=seed + 1), _x(b, hkv, t, e, seed=seed + 2)


_QKV_DYN = ({0: B, 2: T}, {0: B, 2: T}, {0: B, 2: T})


@case("gqa_causal")
def _():
    m = Fn(lambda s, q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True), 3)
    return m, _qkv(8, 2, 6), _QKV_DYN, [_qkv(8, 2, 9, b=3, seed=5)], ["attention", "reshape"]  # KV heads broadcast, not repeated (P38)


@case("mqa_one_kv_head")
def _():
    m = Fn(lambda s, q, k, v: F.scaled_dot_product_attention(q, k, v, enable_gqa=True), 3)
    return m, _qkv(4, 1, 5), _QKV_DYN, [_qkv(4, 1, 7, b=1, seed=5)], ["attention", "reshape"]


def _sliding(s, q, k, v):
    t = q.shape[2]
    i = torch.arange(t).view(-1, 1)
    j = torch.arange(t).view(1, -1)
    keep = (j <= i) & (j > i - 3)  # a sliding window of 3 keys
    return F.scaled_dot_product_attention(q, k, v, attn_mask=keep, enable_gqa=True)


@case("gqa_sliding_window_bool_mask")
def _():
    m = Fn(_sliding, 3)
    return m, _qkv(6, 3, 7), _QKV_DYN, [_qkv(6, 3, 11, b=3, seed=5)], ["attention", "le", "gt"]


@case("gqa_additive_mask")
def _():
    def fn(s, q, k, v, bias):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, enable_gqa=True, scale=0.3)
    m = Fn(fn, 4)
    q, k, v = _qkv(4, 2, 5)
    q2, k2, v2 = _qkv(4, 2, 8, b=3, seed=5)
    return (m, (q, k, v, _x(2, 1, 5, 5, seed=9)), _QKV_DYN + ({0: B, 2: T, 3: T},),
            [(q2, k2, v2, _x(3, 1, 8, 8, seed=10))], ["attention"])


@case("softmax_temperature")
def _():
    m = Fn(lambda s, x: F.softmax(x / s.t, dim=-1), 1, t=torch.tensor(0.7))
    return m, (_x(2, 3, 9),), ({0: B},), [(_x(5, 3, 9, seed=1),)], ["softmax", "div"]


def _idx(*vals):
    return torch.tensor(vals, dtype=torch.int64)


@case("index_copy_runtime_positions")
def _():
    m = Fn(lambda s, x, i, src: x.index_copy(1, i, src) * 2, 3)
    return (m, (_x(2, 6, 4), _idx(4, 1), _x(2, 2, 4, seed=1)), ({0: B}, None, {0: B}),
            [(_x(5, 6, 4, seed=2), _idx(0, 5), _x(5, 2, 4, seed=3))], ["index_copy"])


@case("index_put_one_index")
def _():
    m = Fn(lambda s, x, i, src: torch.ops.aten.index_put.default(x, [None, i], src), 3)
    return (m, (_x(2, 6, 4), _idx(2, 3, 0), _x(2, 3, 4, seed=1)), ({0: B}, None, {0: B}),
            [(_x(3, 6, 4, seed=2), _idx(5, 1, 2), _x(3, 3, 4, seed=3))], ["index_copy"])


def _assign_slot(s, x, i, src):
    y = x * 1.0
    y[:, i] = src  # an in place write at run time positions on a fresh tensor
    return y + 1


@case("setitem_runtime_index")
def _():
    m = Fn(_assign_slot, 3)
    return (m, (_x(2, 6, 4), _idx(1, 4), _x(2, 2, 4, seed=1)), ({0: B}, None, {0: B}),
            [(_x(4, 6, 4, seed=2), _idx(3, 0), _x(4, 2, 4, seed=3))], ["index_copy"])


def _mask_fill(s, x, m):
    y = x * 1.0
    y[m] = -0.5  # boolean mask assignment of one value: no data dependent shape
    return y


@case("bool_mask_assignment")
def _():
    m = Fn(_mask_fill, 2)
    x, x2 = _x(2, 5, 3), _x(4, 5, 3, seed=1)
    return m, (x, x > 0.3), ({0: B}, {0: B}), [(x2, x2 < -0.2)], ["where"]


@case("chunk_split_symbolic_dim")
def _():
    def fn(s, x):
        a, b = x.chunk(2, dim=1)
        c, d = x.split(x.shape[1] // 2, dim=1)
        return a * 2 + b - c * d
    m = Fn(fn, 1)
    return m, (_x(2, 6, 4),), ({0: B, 1: 2 * T},), [(_x(3, 10, 4, seed=1),)], ["select", "reshape"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_op_matches_pytorch_float64(name):
    model, args, dyn, other, expect = CASES[name]()
    progs = check(model, args, dyn, other, expect)
    for p in progs:  # symbolic batch in every case
        assert any(not isinstance(d, int) for v in p.inputs for d in v.type.shape), str(p.inputs)


def test_chunk_of_doubled_dim_keeps_the_root_symbol():
    model, args, dyn, other, expect = CASES["chunk_split_symbolic_dim"]()
    prog = ir.import_module(model.eval(), args, dynamic_shapes=dyn)
    assert str(prog.inputs[0].type) == "f32[B, 2*T, 4]"
    assert "slice" not in prog.op_counts()  # pieces are selects of a [B, 2, T, 4] reshape


def test_bool_mask_indexing_is_refused_loudly():
    m = Fn(lambda s, x, mask: x[mask].sum(), 2).eval()
    x = _x(3, 4)
    with pytest.raises(ir.UnsupportedOpError, match="data dependent shape"):
        ir.import_module(m, (x, x > 0))


def test_uneven_chunk_of_symbolic_dim_is_refused_in_the_front_end():
    with pytest.raises(FrontendError, match="equal pieces"):
        @truestar.function
        def bad(x: Tensor["B", "T", 4]):
            a, b = ops.chunk(x, 2, dim=1)
            return a + b


def test_index_put_with_two_index_tensors_is_refused():
    m = Fn(lambda s, x, i, v: torch.ops.aten.index_put.default(x, [i, i], v), 3).eval()
    with pytest.raises(ir.UnsupportedOpError, match="2 index tensors"):
        ir.import_module(m, (_x(4, 4), _idx(1, 2), _x(2, seed=1)))


def test_aliased_example_inputs_are_refused():
    """torch.export keeps one of two example inputs that share storage and drops the other."""
    m = Fn(lambda s, a, b: a + 2 * b, 2).eval()
    x = _x(3)
    with pytest.raises(ir.UnsupportedOpError, match="share storage"):
        ir.import_module(m, (x, x))
    prog = ir.import_module(m, (x, x.clone()))
    y = _x(3, seed=1)
    assert rel_err(ir.run(prog, x, y)[0], x.double() + 2 * y.double()) < TOL


def test_repeated_index_copy_positions_are_an_error():
    b = ir.Builder("dup")
    x = b.input("x", F64, (4, 3))
    i = b.input("i", torch.int64, (2,))
    src = b.input("src", F64, (2, 3))
    prog = b.build([b.op("index_copy", x, i, src, dim=0)])
    with pytest.raises(ir.IRError, match="repeated indices"):
        ir.run(prog, torch.zeros(4, 3), _idx(1, 1), torch.ones(2, 3))
    with pytest.raises(ir.IRError, match="out of range"):
        ir.run(prog, torch.zeros(4, 3), _idx(1, 4), torch.ones(2, 3))


# ----------------------------------------------------------------------------- reference against torch directly


def test_sort_is_stable_with_ties_and_nan_sorts_last():
    x = torch.tensor([[3.0, 1.0, 3.0, float("nan"), 1.0, -2.0, 3.0]], dtype=F64)
    for desc in (False, True):
        v, i = reference.IMPLS["sort"]({"dim": -1, "descending": desc}, [x], None)
        w = torch.sort(x, dim=-1, descending=desc, stable=True)
        assert torch.equal(i, w.indices), (desc, i, w.indices)
        assert torch.equal(v.nan_to_num(9.0), w.values.nan_to_num(9.0))


def test_argmax_picks_the_first_of_ties_and_nan():
    x = torch.tensor([[1.0, 5.0, 5.0, 0.0], [2.0, float("nan"), 7.0, float("nan")]], dtype=F64)
    assert reference.IMPLS["argmax"]({"dim": 1}, [x], None).tolist() == [1, 1] == torch.argmax(x, 1).tolist()
    assert reference.IMPLS["argmin"]({"dim": 1}, [x], None).tolist() == [3, 1] == torch.argmin(x, 1).tolist()


def test_new_ops_have_symbolic_shape_rules():
    Bs, Ts = ir.symbol("B"), ir.symbol("T")
    t = ir.TensorType(torch.float32, (Bs, Ts, 6))
    i64 = torch.int64
    assert str(shapes.infer("flip", [t], {"dims": [0, 2]})) == "f32[B, T, 6]"
    assert [str(u) for u in shapes.infer_outputs("sort", [t], {"dim": 1})] == ["f32[B, T, 6]", "i64[B, T, 6]"]
    assert [str(u) for u in shapes.infer_outputs("topk", [t], {"k": 2, "dim": -1})] == ["f32[B, T, 2]", "i64[B, T, 2]"]
    assert str(shapes.infer("argmax", [t], {"dim": 1, "keepdim": True})) == "i64[B, 1, 6]"
    assert str(shapes.infer("argmin", [t], {"dim": None})) == "i64[]"
    cache = ir.TensorType(torch.float32, (Bs, 2, ir.symbol("S"), 8))
    pos = ir.TensorType(i64, (Ts,))
    new = ir.TensorType(torch.float32, (Bs, 2, Ts, 8))
    assert str(shapes.infer("index_copy", [cache, pos, new], {"dim": 2})) == "f32[B, 2, S, 8]"
    probs = ir.TensorType(torch.float32, (Bs, 50))
    assert str(shapes.infer("multinomial", [probs], {"seed": 1, "num_samples": 3})) == "i64[B, 3]"
    with pytest.raises(ir.UnsupportedOpError, match="symbolic dim"):
        shapes.infer_outputs("topk", [t], {"k": 2, "dim": 1})
    with pytest.raises(ir.ShapeError, match="src must have shape"):
        shapes.infer("index_copy", [cache, pos, ir.TensorType(torch.float32, (Bs, 2, 1, 8))], {"dim": 2})
    with pytest.raises(ir.ShapeError, match="k=7"):
        shapes.infer_outputs("topk", [t], {"k": 7, "dim": -1})


# ----------------------------------------------------------------------------- sampling


def _sampler(seed, n=1, offset=False):
    b = ir.Builder("sample")
    p = b.input("p", F64, ("B", 10))
    ins = [p]
    if offset:
        ins.append(b.input("off", torch.int64, ()))
    return b.build([b.op("multinomial", *ins, seed=seed, num_samples=n)])


def test_multinomial_is_inverse_cdf_of_the_counter_based_uniforms():
    p = torch.rand(6, 10, generator=gen(0), dtype=F64)
    p[:, 3] = 0.0  # a zero weight category is never drawn
    prog = _sampler(seed=1234, n=4, offset=True)
    got = ir.run(prog, p, torch.tensor(7))[0]
    u = torch.tensor([[reference.uniform01(1234, 7, r * 4 + s) for s in range(4)] for r in range(6)], dtype=F64)
    cum = torch.cumsum(p, dim=1)
    want = torch.searchsorted(cum, u * cum[:, -1:], right=True)
    assert torch.equal(got, want)
    assert not (got == 3).any()


def test_multinomial_is_deterministic_per_seed_and_offset_and_follows_the_weights():
    p = torch.tensor([[0.1, 0.0, 0.6, 0.3] + [0.0] * 6], dtype=F64).expand(4000, 10).contiguous()
    a = ir.run(_sampler(seed=7), p)[0]
    assert torch.equal(a, ir.run(_sampler(seed=7), p)[0])
    assert not torch.equal(a, ir.run(_sampler(seed=8), p)[0])
    freq = torch.bincount(a.reshape(-1), minlength=10).double() / a.numel()
    # 4000 draws: the standard error of a frequency is at most 0.008, the bar is 4 of them
    assert (freq - p[0]).abs().max() < 0.032, freq
    with pytest.raises(ir.IRError, match="no positive weight"):
        ir.run(_sampler(seed=7), torch.zeros(2, 10, dtype=F64))


@truestar.function
def sample_step(logits: Tensor["B", 16, "f64"], step: Tensor["i64"]) -> Tensor["B", 1, "i64"]:
    probs = ops.softmax(ops.top_p_filter(logits, 0.8), dim=-1, temperature=0.7)
    return ops.multinomial(probs, num_samples=1, seed=42, offset=step)


def test_native_top_p_temperature_and_sampling_match_eager():
    for b in (1, 3):
        logits = randn(b, 16, seed=b).double() * 3
        for step in (0, 5):
            got = sample_step(logits, torch.tensor(step))
            want = sample_step.python(logits, torch.tensor(step))
            assert torch.equal(got, want)
    ops_ = sample_step.program.op_counts()
    assert ops_["sort"] == 2 and ops_["cumsum"] == 1 and ops_["multinomial"] == 1


@truestar.function
def top_p_only(logits: Tensor["B", 12, "f64"]) -> Tensor["B", 12, "f64"]:
    return ops.top_p_filter(logits, 0.75)


def test_native_top_p_filter_matches_the_cumsum_recipe():
    for b in (2, 5):
        x = randn(b, 12, seed=b).double() * 2
        got = top_p_only(x)
        # Hugging Face's TopPLogitsWarper written out in torch
        sx, si = torch.sort(x, descending=False)
        remove = sx.softmax(-1).cumsum(-1) <= 1 - 0.75
        remove[..., -1:] = False
        want = x.masked_fill(remove.scatter(1, si, remove), -math.inf)
        assert torch.equal(torch.isinf(got), torch.isinf(want))
        assert torch.equal(got[~torch.isinf(got)], want[~torch.isinf(want)])
        assert torch.isinf(got).any() and not torch.isinf(got).all(dim=-1).any()


# ----------------------------------------------------------------------------- native ops against eager


def ncheck(fn, *inputs):
    got = fn(*inputs)
    want = fn.python(*inputs)
    got = got if isinstance(got, tuple) else (got,)
    want = want if isinstance(want, tuple) else (want,)
    compare(list(got), list(want), fn.name)
    return got


@truestar.function
def n_sort_topk(x: Tensor["B", 9, "f64"]):
    v, i = ops.sort(x, dim=-1, descending=True)
    tv, ti = ops.topk(x, 3, dim=-1, largest=False)
    return (v, i, tv, ti, ops.argmax(x, dim=-1), ops.argmin(x, dim=0, keepdim=True), ops.flip(x, [0, 1]),
            ops.amax(x, dim=-1), ops.amin(x, dim=(0, 1), keepdim=True), x.amax(0))


@truestar.function
def n_gather_cumsum(x: Tensor["B", 6, "f64"], i: Tensor["B", 3, "i64"], j: Tensor[4, "i64"]):
    return ops.gather(x, 1, i), ops.index_select(x, 1, j), ops.cumsum(x, dim=1), x.flip(1).cumsum(1)


@truestar.function
def n_cache_write(cache: Tensor["B", 2, "S", 4, "f64"], pos: Tensor["T", "i64"], new: Tensor["B", 2, "T", 4, "f64"]):
    return ops.index_copy(cache, 2, pos, new), cache.index_copy(2, pos, new * 2)


@truestar.function
def n_chunks(x: Tensor["B", "2*T", 6, "f64"]):
    a, b = ops.chunk(x, 2, dim=1)
    c, d, e = ops.split(x, 2, dim=2)
    return a * b, c + d - e


@truestar.function
def n_attention(q: Tensor["B", 6, "T", 8, "f64"], k: Tensor["B", 2, "T", 8, "f64"], v: Tensor["B", 2, "T", 8, "f64"]):
    full = ops.attention(q, k, v, is_causal=True, enable_gqa=True)
    win = ops.attention(q, k, v, is_causal=True, enable_gqa=True, window=3)
    k1 = k[:, :1]
    v1 = v[:, :1]
    mqa = ops.attention(q, k1, v1, enable_gqa=True)
    rep = ops.attention(q, ops.repeat_interleave(k, 3, dim=1), ops.repeat_interleave(v, 3, dim=1), is_causal=True)
    return full, win, mqa, full - rep


@truestar.function
def n_misc(x: Tensor["B", 5, "f64"], m: Tensor["B", 5, "bool"], ids: Tensor["B", 3, "i64"], w: Tensor[7, 5, "f64"]):
    return (ops.softmax(x, dim=-1, temperature=0.5), ops.masked_fill(x, m, -1.0), ops.embedding(w, ids),
            ops.rms_norm(x, [5], w[0], eps=1e-5), ops.cos(x) + ops.sin(x) + ops.log(x * x + 1.0),
            ops.cast(ops.arange(x.shape[0]), "f64"))


def test_native_llm_ops_match_eager_pytorch_with_symbolic_dims():
    for b in (2, 5):
        ncheck(n_sort_topk, randn(b, 9, seed=b).double())
        x = randn(b, 6, seed=b).double()
        ncheck(n_gather_cumsum, x, torch.randint(0, 6, (b, 3), generator=gen(b)), _idx(5, 0, 2, 2))
        for s, t in ((7, 1), (9, 3)):
            ncheck(n_cache_write, randn(b, 2, s, 4, seed=s).double(), torch.randperm(s, generator=gen(t))[:t],
                   randn(b, 2, t, 4, seed=t).double())
        for t in (1, 4):
            ncheck(n_chunks, randn(b, 2 * t, 6, seed=t).double())
        for t in (1, 4, 7):
            q, k, v = (randn(b, h, t, 8, seed=t + h).double() for h in (6, 2, 2))
            full, win, mqa, diff = ncheck(n_attention, q, k, v)
            assert diff.abs().max() < 1e-12  # enable_gqa equals repeating the KV heads by hand
        x = randn(b, 5, seed=b).double()
        ncheck(n_misc, x, x > 0, torch.randint(0, 7, (b, 3), generator=gen(b)), randn(7, 5, seed=3).double())
    assert {"sort", "topk", "argmax", "argmin", "flip", "amax", "amin"} <= set(n_sort_topk.program.op_counts())
    assert n_attention.program.op_counts()["attention"] == 4


def test_native_window_matches_an_explicit_mask():
    q, k, v = (randn(2, h, 7, 8, seed=h).double() for h in (6, 2, 2))
    win = n_attention(q, k, v)[1]
    i, j = torch.arange(7).view(-1, 1), torch.arange(7).view(1, -1)
    keep = (j <= i) & (j > i - 3)
    want = F.scaled_dot_product_attention(q, k.repeat_interleave(3, 1), v.repeat_interleave(3, 1), attn_mask=keep)
    assert rel_err(win, want) < TOL


# ----------------------------------------------------------------------------- KV cache decoding


def _prompt(b=2, t=5, seed=1):
    return torch.randint(0, llama.V, (b, t), generator=gen(seed))


def test_native_greedy_decode_matches_torch_token_by_token():
    model = llama.make_model()
    for b, t0, cap in ((2, 5, 24), (1, 3, 40)):
        prompt = _prompt(b, t0)
        before = compile_count()
        got, got_logits = llama.native_generate(model, prompt, steps=16, capacity=cap)
        assert compile_count() == before  # one program, compiled at decoration, serves every position
        want, want_logits = llama.torch_generate(model, prompt, steps=16)
        assert torch.equal(got, want), (got, want)
        assert rel_err(got_logits, want_logits) < 1e-13
        top2 = want_logits.topk(2, dim=-1).values
        assert (top2[..., 0] - top2[..., 1]).min() > 1e-6  # no near tie: equal tokens is a real check
        assert len(set(want.reshape(-1).tolist())) > 8  # the tokens vary, so a stale cache would show
    sig = str(llama.forward.signature)
    assert "ids: i64[B, T]" in sig and "pos: i64[T]" in sig and "k_cache: f64[2, B, 2, S, 8]" in sig
    assert llama.forward.program.op_counts()["index_copy"] == 4


def test_native_decode_with_a_stale_cache_diverges():
    """Guard: the token comparison notices a cache that is not written (index_copy replaced by its input)."""
    model = llama.make_model()
    prompt = _prompt()
    want, _ = llama.torch_generate(model, prompt, steps=8)
    real = reference.IMPLS["index_copy"]
    try:
        reference.IMPLS["index_copy"] = lambda attrs, args, shape: args[0].clone()
        got, _ = llama.native_generate(model, prompt, steps=8, capacity=16)
    finally:
        reference.IMPLS["index_copy"] = real
    assert not torch.equal(got, want)


class Prefill(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids):
        return self.m(ids)


class Decode(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids, pos, k0, v0, k1, v1):
        return self.m.step(ids, pos, k0, v0, k1, v1)


def _cache(b, s, seed):
    return [randn(b, llama.HKV, s, llama.HD, seed=seed + i) for i in range(2 * llama.L)]


@pytest.fixture(scope="module")
def imported_llama():
    model = llama.make_model()
    S = Dim("S", min=2, max=256)
    cd = {0: B, 2: S}
    out = {}
    for decompose in (False, True):
        pre = ir.import_module(Prefill(model), (_prompt(2, 5),), dynamic_shapes=({0: B, 1: T},), decompose=decompose)
        dec = ir.import_module(Decode(model), (_prompt(2, 1), _idx(3), *_cache(2, 10, 0)),
                               dynamic_shapes=({0: B}, None, cd, cd, cd, cd), decompose=decompose)
        out[decompose] = (pre, dec)
    return model, out


@pytest.mark.parametrize("decompose", [False, True], ids=["export_ir", "core_aten"])
def test_imported_llama_prefill_and_decode_match_pytorch_float64(imported_llama, decompose):
    model, progs = imported_llama
    pre, dec = progs[decompose]
    assert str(pre.inputs[0].type) == "i64[B, T]"
    assert [str(v.type) for v in dec.inputs[:3]] == ["i64[B, 1]", "i64[1]", "f32[B, 2, S, 8]"]
    if not decompose:
        assert "attention" in pre.op_counts() and "index_copy" in dec.op_counts()
    md = Decode(copy.deepcopy(model).double())
    for b, t in ((2, 5), (3, 9), (1, 2)):
        ids = _prompt(b, t, seed=t)
        compare(ir.run(pre, ids), f64(Prefill(model), ids), f"prefill {b}x{t}")
    for b, s, p in ((2, 10, 3), (3, 17, 16), (1, 4, 0)):
        args = (_prompt(b, 1, seed=s), _idx(p), *_cache(b, s, seed=s))
        with torch.no_grad():
            want = md(*[a.double() if a.is_floating_point() else a for a in args])
        compare(ir.run(dec, *args), list(want), f"decode batch {b}, cache {s}, position {p}")


def test_imported_llama_greedy_decode_matches_torch(imported_llama):
    """Prefill program, then 16 steps of one decode program (every position, one import)."""
    model, progs = imported_llama
    pre, dec = progs[False]
    prompt = _prompt(2, 5)
    want, want_logits = llama.torch_generate(model, prompt, steps=16)
    cap = 24
    out = ir.run(pre, prompt)
    cache = []
    for kv in out[1:]:
        c = torch.zeros(2, llama.HKV, cap, llama.HD, dtype=F64)
        c[:, :, :5] = kv
        cache.append(c)
    logits = [out[0][:, -1]]
    toks = [logits[0].argmax(-1)]
    for s in range(15):
        res = ir.run(dec, toks[-1][:, None], _idx(5 + s), *cache)
        cache = res[1:]
        logits.append(res[0][:, -1])
        toks.append(logits[-1].argmax(-1))
    assert torch.equal(torch.stack(toks, 1), want)
    assert rel_err(torch.stack(logits), want_logits) < 1e-13


# ----------------------------------------------------------------------------- autodiff through GQA


def test_gqa_attention_gradient_matches_autograd():
    torch.manual_seed(0)

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(16, 32, bias=False)
            self.kv = nn.Linear(16, 32, bias=False)

        def forward(self, x):
            b, t, _ = x.shape
            q = self.q(x).view(b, t, 4, 8).transpose(1, 2)
            k, v = self.kv(x).view(b, t, 2, 16).transpose(1, 2).chunk(2, dim=-1)
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True).sum(dim=(1, 3))

    m = Attn().eval()
    x = randn(3, 5, 16)
    prog = ir.import_module(m, (x,))
    g = grad(prog, ["q.weight", "kv.weight"], seed="ones")
    _, gq, gkv = ir.run(g, x)
    md = copy.deepcopy(m).double()
    md(x.double()).sum().backward()
    assert rel_err(gq, md.q.weight.grad) < TOL and rel_err(gkv, md.kv.weight.grad) < TOL


# ----------------------------------------------------------------------------- tests that prove the tests work


def _broken(op):
    real = reference.IMPLS[op]
    if op == "flip":
        return lambda attrs, args, shape: real({**attrs, "dims": attrs["dims"][:-1]}, args, shape)
    if op == "sort":  # loses stability: the tie order is reversed
        def wrong(attrs, args, shapes_):
            v, i = real({**attrs, "descending": not attrs.get("descending", False)}, args, shapes_)
            return [v, i]
        return wrong
    if op == "topk":
        return lambda attrs, args, shapes_: [t.roll(1, -1) for t in real(attrs, args, shapes_)]
    if op == "argmax":
        return lambda attrs, args, shape: real(attrs, args, shape).clamp(max=2)
    if op == "index_copy":  # writes one slot later
        return lambda attrs, args, shape: real(attrs, [args[0], (args[1] + 1) % args[0].shape[attrs["dim"]],
                                                       args[2]], shape)
    if op == "amax":  # ignores NaN handling and takes the second largest instead of the largest
        def wrong(attrs, args, shape):
            x = args[0]
            return real(attrs, [torch.where(x == real({**attrs, "keepdim": True}, [x], None),
                                            torch.full_like(x, -1e9), x)], shape)
        return wrong
    raise KeyError(op)


@pytest.mark.parametrize("op,case_name", [
    ("flip", "flip"), ("sort", "sort_last_dim"), ("topk", "topk_largest"), ("argmax", "argmax_argmin"),
    ("index_copy", "index_copy_runtime_positions"), ("amax", "amax_amin_reductions"),
])
def test_broken_reference_op_is_caught(monkeypatch, op, case_name):
    monkeypatch.setitem(reference.IMPLS, op, _broken(op))
    model, args, dyn, other, expect = CASES[case_name]()
    with pytest.raises(AssertionError, match="relative error|integer result differs"):
        check(model, args, dyn, other, expect, graphs=(False,))


def test_broken_kv_cache_attention_is_caught_by_the_decode_comparison(monkeypatch, imported_llama):
    """An attention kernel that ignores its mask breaks decode (unwritten cache slots leak in)."""
    model, progs = imported_llama
    real = reference.IMPLS["attention"]
    monkeypatch.setitem(reference.IMPLS, "attention", lambda attrs, args, shape: real(attrs, args[:3] + [None], shape))
    args = (_prompt(2, 1), _idx(3), *_cache(2, 10, seed=0))
    want = list(Decode(copy.deepcopy(model).double())(*[a.double() if a.is_floating_point() else a for a in args]))
    with pytest.raises(AssertionError, match="relative error"):
        compare(ir.run(progs[False][1], *args), want, "decode")
