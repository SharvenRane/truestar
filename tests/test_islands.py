"""Mixed precision islands: values a model declares in a wider float dtype than the program's own (an
explicit ``.float()``) keep float32 storage and math inside a float16 or bfloat16 program, with a
rounding back only where the model casts; params read through such a cast are stored in float32.

* A Llama decode with float32 rotary angles (examples/llama_paged_decode.py, ``rope_f32``) compiled
  in float16 passes Truestar's own checks, which refuse it without the islands, and its new key
  matches float64 at long positions; imported from a float64 model or from a float16 one.
* The report lists the islands; a program without explicit wider casts has none, and its kernels
  are the ones it had (no float32 buffer appears).
"""

import copy
import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

import llama_paged_decode as P  # noqa: E402
from truestar import ir  # noqa: E402
from truestar.compiler import compile_program  # noqa: E402


def cfg(rope_f32):
    return P.Cfg(vocab=256, dim=128, heads=4, kv_heads=2, head_dim=32, layers=2, ffn=256, block=16,
                 rope_f32=rope_f32)


def model(rope_f32):
    torch.manual_seed(0)
    return P.Llama(cfg(rope_f32)).eval()


def import_dense(m, dtype, capacity=4096):
    from torch.export import Dim
    c = m.c
    step = P.DenseStep(m.to(dtype))
    B, T, S = Dim("B", min=1, max=64), Dim("T", min=1, max=512), Dim("S", min=2, max=1 << 15)
    kc = torch.zeros(c.layers, 2, c.kv_heads, 16, c.head_dim, dtype=dtype)
    return ir.import_module(step, (torch.zeros(2, 3, dtype=torch.int64), torch.arange(3), kc, kc.clone()),
                            dynamic_shapes=({0: B, 1: T}, {0: T}, {1: B, 3: S}, {1: B, 3: S}))


def run_at(cp, c, pos, N=4096):
    kc = torch.zeros(c.layers, 1, c.kv_heads, N, c.head_dim, device="cuda", dtype=torch.float16)
    ids = torch.randint(0, c.vocab, (1, 1), device="cuda", generator=torch.Generator(device="cuda").manual_seed(pos))
    return ids, cp([ids, torch.tensor([pos], device="cuda"), kc, kc.clone()])


def test_without_islands_the_float16_program_is_over_the_fixed_bar():
    """Before the derived bound the end to end check refused this program. Its outputs are over the
    fixed bar; the bound derived from the program allows what every op rounded to float16 computes
    (docs/TOLERANCE.md), so the call is accepted only by it and lists every such output."""
    m = model(False)
    cp = compile_program(import_dense(m, torch.float64), "cuda", "float16")
    assert cp.report.islands == []
    run_at(cp, m.c, 3000)
    assert cp.derived_accepts and all(a["error"] > a.get("fixed_bar", 0.0) for a in cp.derived_accepts)


@pytest.mark.parametrize("src", [torch.float64, torch.float16])
def test_float32_rope_forms_islands_and_passes_the_checks(src):
    m = model(True)
    prog = import_dense(m, src)
    cp = compile_program(prog, "cuda", "float16")
    names = [n for n, _, _ in cp.report.islands]
    assert names and all(d == "float32" for _, d, _ in cp.report.islands)
    if src == torch.float64:
        assert any("inv_freq" in n for n in names)          # the param read through .float()
    assert "mixed precision islands" in cp.report.to_text()
    assert cp.report.to_dict()["islands"]
    ids, (lg, k16, _) = run_at(cp, m.c, 3000)              # first call and end to end checks pass
    assert torch.isfinite(lg).all()


def test_new_key_matches_float64_at_long_positions():
    m = model(True)
    cp = compile_program(import_dense(m, torch.float64), "cuda", "float16")
    ref_step = P.DenseStep(model(True).double()).cuda()
    for pos in (300, 4000):
        ids, (lg, k16, _) = run_at(cp, m.c, pos)
        kc = torch.zeros(m.c.layers, 1, m.c.kv_heads, 4096, m.c.head_dim, device="cuda", dtype=torch.float64)
        with torch.no_grad():
            _, k64, _ = ref_step(ids, torch.tensor([pos], device="cuda"), kc, kc.clone())
        err = ((k16[:, :, :, pos].double() - k64[:, :, :, pos]).abs().max() / k64[:, :, :, pos].abs().max()).item()
        assert err < 3e-3


def test_a_pure_16_bit_program_has_no_islands_and_unchanged_kernels():
    net = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 8)).eval()
    prog = ir.import_module(net, (torch.randn(4, 32),))
    for dt in ("float16", "bfloat16", "float32"):
        cp = compile_program(prog, "cuda", dt)
        assert cp.report.islands == []
        assert "mixed precision islands" not in cp.report.to_text() or dt != "float32"
        for k in cp.report.kernels:
            assert "float32" not in str(k.get("dtypes", "")) or dt == "float32"
        cp([torch.randn(4, 32, device="cuda")])


def test_a_float32_program_is_untouched_by_wider_declared_values():
    m = model(True)
    cp = compile_program(import_dense(m, torch.float64), "cuda", "float32")
    assert cp.report.islands == []
    run_at(cp, m.c, 100)


def test_graphed_llama_step_with_an_update_the_plan_cannot_alias():
    """The dense Llama step's cache update cannot be aliased in place by the plan, so the ordinary
    call copies the result into the caller's tensor; a CUDA graph of it must record that copy (it was
    refused with TS801 before) and give the ordinary call's bits on every replay."""
    for rope in (False, True):
        m = model(rope)
        prog = import_dense(m, torch.float64)
        cp, plain = compile_program(prog, "cuda", "float16"), compile_program(prog, "cuda", "float16")
        g = cp.graphed(update={2: 1, 3: 2})
        c = m.c
        kc = torch.zeros(c.layers, 1, c.kv_heads, 64, c.head_dim, device="cuda", dtype=torch.float16)
        vc, pk, pv = kc.clone(), kc.clone(), kc.clone()
        ids = torch.zeros(1, 1, dtype=torch.long, device="cuda")
        for pos in range(5):
            p = torch.tensor([pos], device="cuda")
            outs = g([ids, p, kc, vc])
            ref = plain([ids, p, pk, pv])
            pk, pv = ref[1], ref[2]
            assert torch.equal(outs[0], ref[0]) and torch.equal(kc, pk) and torch.equal(vc, pv)
            assert outs[1] is kc
            ids = outs[0][:, -1].argmax(-1)[:, None]
        assert g.captures == 1 and g.replays == 4


def test_flat_step_writes_its_cache_in_place_and_matches_the_dense_step():
    """examples/llama_paged_decode.py FlatStep: one cache buffer written by an index_copy per layer.
    With update= the plan writes every layer's index_copy over the caller's buffer (a chain the
    runtime used to break after the first write), the logits are the dense step's bit for bit, and a
    CUDA graph of it replays those bits."""
    m = model(True)
    c = m.c
    S = 64
    flat = compile_program(P.import_flat_step(m, capacity=16), "cuda", "float16")
    dense = compile_program(import_dense(m, torch.float64), "cuda", "float16")
    g = compile_program(P.import_flat_step(m, capacity=16), "cuda", "float16").graphed(update={3: 1, 4: 2})
    kf = torch.zeros(c.layers * S, 1, c.kv_heads, c.head_dim, device="cuda", dtype=torch.float16)
    vf, kg, vg = kf.clone(), kf.clone(), kf.clone()
    kd = torch.zeros(c.layers, 1, c.kv_heads, S, c.head_dim, device="cuda", dtype=torch.float16)
    vd = kd.clone()
    ids = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    for pos in range(5):
        p = torch.tensor([pos], device="cuda")
        rows = P.flat_rows(c, p, S)
        of = flat([ids, p, rows, kf, vf], update={3: 1, 4: 2})
        assert flat.last_update_inplace and of[1] is kf
        og = g([ids, p, rows, kg, vg])
        od = dense([ids, p, kd, vd], update={2: 1, 3: 2})
        assert torch.equal(of[0], od[0]) and torch.equal(og[0], of[0]) and torch.equal(kg, kf)
        ids = of[0][:, -1].argmax(-1)[:, None]
    assert g.captures == 1


# ---- float64 islands in float32 programs (P63, scripts/rope_long_positions.py)

def model64(rope_f64):
    torch.manual_seed(0)
    c = P.Cfg(vocab=256, dim=128, heads=4, kv_heads=2, head_dim=32, layers=3, ffn=256, block=16, rope_f64=rope_f64)
    return P.Llama(c).eval()


def decode_2100(cp, c, check=True):
    kc = torch.zeros(c.layers, 2, c.kv_heads, 2200, c.head_dim, device="cuda")
    return cp([torch.zeros(2, 1, dtype=torch.long, device="cuda"), torch.tensor([2100], device="cuda"), kc, kc.clone()])


def test_float32_angles_are_over_the_fixed_bar_at_position_2100():
    """The failure found by the prefix cache check: float32 rotary angles at position 2100 put the new
    key row 8.6e-4 off the float64 reference, above the fixed bar. The end to end check refused it
    before the derived bound; now the new key rows are accepted only by the bound derived from the
    program (float32 angles are what the model declares) and listed."""
    m = model64(False)
    ir_, _ = P.import_steps(m, capacity=2200, max_capacity=8192)
    cp = compile_program(ir_, "cuda", "float32", deterministic=True)
    decode_2100(cp, m.c)
    assert any("stack" in a["output"] for a in cp.derived_accepts)


def test_float64_angles_are_a_float64_island_and_pass_at_position_2100():
    m = model64(True)
    ir_, _ = P.import_steps(m, capacity=2200, max_capacity=8192, trace_dtype=torch.float32)
    cp = compile_program(ir_, "cuda", "float32", deterministic=True)
    _, kc, _ = decode_2100(cp, m.c)                      # passes the end to end check
    assert cp.report.islands and {d for _, d, _ in cp.report.islands} == {"float64"}
    with torch.no_grad():
        kc0 = torch.zeros(m.c.layers, 2, m.c.kv_heads, 2200, m.c.head_dim, dtype=torch.float64, device="cuda")
        _, kref, _ = P.DenseStep(copy.deepcopy(m)).double().cuda()(
            torch.zeros(2, 1, dtype=torch.long, device="cuda"), torch.tensor([2100], device="cuda"), kc0, kc0.clone())
    row, ref = kc[:, :, :, 2100].double(), kref[:, :, :, 2100]
    assert ((row - ref).abs().max() / ref.abs().max()).item() < 2e-6     # float32 angles: 4.3e-5


def test_a_float32_import_without_float64_values_has_no_islands():
    m = model64(False)
    ir_, _ = P.import_steps(m, capacity=64, trace_dtype=torch.float32)
    assert compile_program(ir_, "cuda", "float32").report.islands == []
