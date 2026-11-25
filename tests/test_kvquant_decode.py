"""The compiled executor running a Llama decode that opts into a quantized KV cache
(examples/llama_kvquant_decode.py, IR op kvq_attention).

* kv_cache=None is the dense step of examples/llama_paged_decode.py itself.
* int8, fp8, and (CFG4, head dim 32) int4 and nvfp4 rotated or not: the compiled step has the kvq_attention library step and no fallbacks, passes the
  first call check against the float64 reference (attention over the dequantized cache), equals the
  eager float64 step within float16 rounding, crosses group boundaries (each full tail quantized
  once), gives the same bits run to run, and in deterministic mode the same bits with the SM count
  pinned to 26 and to 70.
"""

import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

import llama_kvquant_decode as Q  # noqa: E402
import llama_paged_decode as P  # noqa: E402
from truestar import kvquant as KQ  # noqa: E402

CFG = P.Cfg(vocab=128, dim=64, heads=4, kv_heads=2, head_dim=16, layers=2, ffn=128, block=16)
GROUP = 16


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return P.Llama(CFG).eval()


@pytest.fixture(scope="module")
def prefilled(model):
    from truestar.compiler import compile_program
    b, n = 2, 37
    dense, _ = P.import_steps(model, capacity=n)
    cd = compile_program(dense, "cuda", "float32")
    prompt = torch.randint(0, CFG.vocab, (b, n), generator=torch.Generator().manual_seed(3)).cuda()
    kc = torch.zeros(CFG.layers, b, CFG.kv_heads, n, CFG.head_dim, device="cuda")
    logits, kc, vc = cd([prompt, torch.arange(n, device="cuda"), kc, kc.clone()])
    return logits[:, -1].argmax(-1)[:, None], kc, vc, n


def test_none_is_the_dense_step_unchanged(model):
    step = Q.make_decode_step(model, None)
    assert type(step) is P.DenseStep and step.model is model
    with pytest.raises(KQ.KVQuantError, match="TS701"):
        Q.make_decode_step(model, "int16")


@pytest.mark.parametrize("fmt", ["int8", "fp8"])
def test_compiled_quantized_decode(model, prefilled, fmt):
    from truestar.compiler import compile_program
    ids0, kc, vc, n = prefilled
    prog = Q.import_quant_step(model, fmt, GROUP)
    cq = compile_program(prog, "cuda", "float32")
    assert len(cq.report.fallbacks) == 0
    assert any("kvq_attention" in str(e) for e in cq.report.library)
    qc = Q.QuantCache.from_dense(kc, vc, n, fmt, GROUP)
    assert qc.kd.shape[3] == 32 and qc.tlen == 5
    eager = Q.QuantDecodeStep(copy.deepcopy(model).double().cuda(), fmt, GROUP)
    ids = ids0
    for s in range(15):                    # the tail fills at step 11: one group quantized
        ins = [ids, torch.tensor([n + s], device="cuda")] + qc.inputs("cuda")
        lg, tk, tv = cq(ins)              # the first call is checked against float64
        again = cq(ins)[0]
        assert torch.equal(lg, again)
        ref = eager(*[x.double() if x.is_floating_point() else x for x in ins])[0]
        assert ((lg.double() - ref).abs().max() / ref.abs().max()).item() < 2e-2
        qc.after_step(tk, tv)
        ids = lg[:, -1].argmax(-1)[:, None]
    assert qc.kd.shape[3] == 48 and qc.tlen == 4


def test_layer_error_bounds_are_reported(model, prefilled):
    from truestar.compiler import compile_program
    ids, kc, vc, n = prefilled
    cq = compile_program(Q.import_quant_step(model, "int8", GROUP, return_q=True), "cuda", "float32")
    qc = Q.QuantCache.from_dense(kc, vc, n, "int8", GROUP, keep_orig=True)
    out = cq([ids, torch.tensor([n], device="cuda")] + qc.inputs("cuda"))
    qc.after_step(out[1], out[2])
    bounds = Q.layer_error_bounds(out[3], qc)
    assert len(bounds) == CFG.layers and all(0 < b < 1 for b in bounds)


def test_deterministic_decode_has_the_same_bits_on_26_and_70_sms(model, prefilled, monkeypatch):
    from truestar.compiler import compile_program
    ids, kc, vc, n = prefilled
    prog = Q.import_quant_step(model, "int8", GROUP)
    qc = Q.QuantCache.from_dense(kc, vc, n, "int8", GROUP)
    ins = [ids, torch.tensor([n], device="cuda")] + qc.inputs("cuda")
    outs = []
    for sms in (26, 70):
        monkeypatch.setattr(KQ, "_sms", lambda sms=sms: sms)
        outs.append(compile_program(prog, "cuda", "float32", deterministic=True)(ins)[0])
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("fmt", ["int8", "fp8"])
def test_fixed_capacity_step_is_one_graph_for_the_whole_decode(model, prefilled, fmt):
    from truestar.compiler import compile_program
    ids, kc, vc, n = prefilled
    grow = compile_program(Q.import_quant_step(model, fmt, GROUP), "cuda", "float32")
    g = compile_program(Q.import_quant_step(model, fmt, GROUP, fixed=True), "cuda", "float32").graphed(
        static=(2, 3, 4, 5), update={6: 1, 7: 2})
    qa = Q.QuantCache.from_dense(kc, vc, n, fmt, GROUP)
    qb = Q.QuantCache.from_dense(kc, vc, n, fmt, GROUP, capacity=128)
    for s in range(30):                    # crosses two group fills
        pos = torch.tensor([n + s], device="cuda")
        la, ta, va = grow([ids, pos] + qa.inputs("cuda"))
        lb, tb, vb = g([ids, pos] + qb.inputs("cuda"))
        assert torch.equal(la, lb)
        qa.after_step(ta, va)
        qb.after_step(tb, vb)
        ids = lb[:, -1].argmax(-1)[:, None]
    assert g.captures == 1 and qb.plen == qa.kd.shape[3]


def test_a_statically_bound_input_in_another_dtype_is_refused(model, prefilled):
    from truestar.compiler import compile_program
    from truestar.runtime.graphs import CaptureError
    ids, kc, vc, n = prefilled
    g = compile_program(Q.import_quant_step(model, "int8", GROUP, fixed=True), "cuda", "float16").graphed(
        static=(2, 3, 4, 5), update={6: 1, 7: 2})
    qc = Q.QuantCache.from_dense(kc, vc, n, "int8", GROUP, capacity=128)   # float32 tails: stored float16
    with pytest.raises(CaptureError, match="TS802"):
        g([ids, torch.tensor([n], device="cuda")] + qc.inputs("cuda"))


# ----------------------------------------------------------------------------- 4 bit prefix, float tail

CFG4 = P.Cfg(vocab=128, dim=64, heads=4, kv_heads=2, head_dim=32, layers=2, ffn=128, block=16)
LOW = [(f, r) for f in ("int4", "nvfp4") for r in (False, True)]


@pytest.fixture(scope="module")
def prefilled4():
    from truestar.compiler import compile_program
    torch.manual_seed(0)
    m = P.Llama(CFG4).eval()
    b, n = 2, 37
    cd = compile_program(P.import_steps(m, capacity=n)[0], "cuda", "float32")
    prompt = torch.randint(0, CFG4.vocab, (b, n), generator=torch.Generator().manual_seed(3)).cuda()
    kc = torch.zeros(CFG4.layers, b, CFG4.kv_heads, n, CFG4.head_dim, device="cuda")
    logits, kc, vc = cd([prompt, torch.arange(n, device="cuda"), kc, kc.clone()])
    return m, logits[:, -1].argmax(-1)[:, None], kc, vc, n


def test_unknown_format_is_refused(model):
    with pytest.raises(KQ.KVQuantError, match="TS701"):
        Q.make_decode_step(model, "int2")
    with pytest.raises(KQ.KVQuantError, match="TS701"):
        Q.QuantDecodeStep(model, "int8", rot=True)


@pytest.mark.parametrize("fmt,rot", LOW)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_kernel_with_float_tail_and_prefix_len_matches_float64(fmt, rot, dtype):
    g = torch.Generator(device="cuda").manual_seed(5)
    B, H, N, D, C = 2, 2, 96, 64, 32
    k = torch.randn(B, H, N, D, device="cuda", generator=g).to(dtype)
    v = torch.randn(B, H, N, D, device="cuda", generator=g).to(dtype)
    tk = torch.randn(B, H, C, D, device="cuda", generator=g).to(dtype)
    tv = torch.randn(B, H, C, D, device="cuda", generator=g).to(dtype)
    q = torch.randn(B, 2 * H, 1, D, device="cuda", generator=g).to(dtype)
    kc, vc = KQ.quantize(k, fmt, 32, rot=rot), KQ.quantize(v, fmt, 32, rot=rot)
    for tl, pl in ((7, None), (0, None), (C, 64), (3, 32)):
        tail = (tk, tv, torch.tensor([tl], dtype=torch.int32, device="cuda"))
        plen = None if pl is None else torch.tensor([pl], dtype=torch.int32, device="cuda")
        out = KQ.decode_attention(q, kc, vc, tail=tail, prefix_len=plen, verify=False, chunk=64)
        ref = KQ.reference(q, kc, vc, tail=tail, prefix_len=plen)
        err = ((out.double() - ref).abs().max() / ref.abs().max()).item()
        assert err < (2e-3 if dtype == torch.float16 else 1.6e-2), (tl, pl, err)
        if pl is not None:      # a fixed capacity prefix is bitwise the cache of its live keys
            short = KQ.QuantKV(kc.data[:, :, :pl], kc.scale[:, :, :pl // 32], fmt, 32, bscale=kc.bscale[:, :, :pl], rot=rot)
            vshort = KQ.QuantKV(vc.data[:, :, :pl], vc.scale[:, :, :pl // 32], fmt, 32, bscale=vc.bscale[:, :, :pl], rot=rot)
            assert torch.equal(out, KQ.decode_attention(q, short, vshort, tail=tail, verify=False, chunk=64))


@pytest.mark.parametrize("fmt,rot", LOW)
def test_compiled_4bit_decode_with_float_tail(prefilled4, fmt, rot):
    from truestar.compiler import compile_program
    m, ids0, kc, vc, n = prefilled4
    cq = compile_program(Q.import_quant_step(m, fmt, GROUP, rot=rot), "cuda", "float32")
    assert len(cq.report.fallbacks) == 0
    qc = Q.QuantCache.from_dense(kc, vc, n, fmt, GROUP, rot=rot)
    assert qc.kd.shape[3] == 32 and qc.kd.shape[4] == CFG4.head_dim // 2 and qc.tlen == 5
    eager = Q.make_decode_step(copy.deepcopy(m).double().cuda(), fmt, GROUP, rot=rot)
    ids = ids0
    for s in range(15):                    # the tail fills at step 11: one group quantized to 4 bits
        ins = [ids, torch.tensor([n + s], device="cuda")] + qc.inputs("cuda")
        lg, tk, tv = cq(ins)              # the first call is checked against float64
        assert torch.equal(lg, cq(ins)[0])
        ref = eager(*[x.double() if x.is_floating_point() else x for x in ins])[0]
        assert ((lg.double() - ref).abs().max() / ref.abs().max()).item() < 2e-2
        qc.after_step(tk, tv)
        ids = lg[:, -1].argmax(-1)[:, None]
    assert qc.kd.shape[3] == 48 and qc.kb.shape[3] == 48 and qc.tlen == 4


@pytest.mark.parametrize("fmt,rot", [("int4", True), ("nvfp4", False)])
def test_fixed_capacity_4bit_step_is_one_graph_and_equals_the_growing_one(prefilled4, fmt, rot):
    from truestar.compiler import compile_program
    m, ids, kc, vc, n = prefilled4
    grow = compile_program(Q.import_quant_step(m, fmt, GROUP, rot=rot), "cuda", "float32")
    g = compile_program(Q.import_quant_step(m, fmt, GROUP, fixed=True, rot=rot), "cuda", "float32").graphed(
        static=(2, 3, 4, 5, 9, 10), update={6: 1, 7: 2})
    qa = Q.QuantCache.from_dense(kc, vc, n, fmt, GROUP, rot=rot)
    qb = Q.QuantCache.from_dense(kc, vc, n, fmt, GROUP, capacity=128, rot=rot)
    for s in range(30):                    # crosses two group fills
        pos = torch.tensor([n + s], device="cuda")
        la, ta, va = grow([ids, pos] + qa.inputs("cuda"))
        lb, tb, vb = g([ids, pos] + qb.inputs("cuda"))
        assert torch.equal(la, lb)
        qa.after_step(ta, va)
        qb.after_step(tb, vb)
        ids = lb[:, -1].argmax(-1)[:, None]
    assert g.captures == 1 and qb.plen == qa.kd.shape[3]
