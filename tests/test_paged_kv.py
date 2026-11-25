"""Paged KV cache decoding (examples/llama_paged_decode.py): paged against dense in the float64
reference and compiled on CUDA, ragged batches, a Hugging Face generate check, and failures."""

import importlib.util
import math
import os
import sys

import pytest
import torch

from truestar import ir

_EX = os.path.join(os.path.dirname(__file__), "..", "examples", "llama_paged_decode.py")
_spec = importlib.util.spec_from_file_location("llama_paged_decode", _EX)
P = importlib.util.module_from_spec(_spec)
sys.modules["llama_paged_decode"] = P    # dataclasses look their module up while the class is built
_spec.loader.exec_module(P)

CUDA = torch.cuda.is_available()
C = P.Cfg()
_STEPS = {}


def steps():
    if "s" not in _STEPS:
        model = P.make_model(C)
        _STEPS["s"] = (model,) + P.import_steps(model)
    return _STEPS["s"]


def _prompt(b, t, seed=1):
    return torch.randint(0, C.vocab, (b, t), generator=torch.Generator().manual_seed(seed))


def _order(n, seed=2):
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()


def test_paged_equals_dense_bitwise_in_the_reference():
    """Scattered blocks, a pool full of finite garbage (pi), a table wider than needed: logits are
    bitwise equal to the dense cache at every step, and the program is one import for every step."""
    _, dense, paged = steps()
    prompt = _prompt(2, 5)
    td, ld = P.generate_dense(lambda *a: ir.run(dense, *a), prompt, 10, C, capacity=24)
    tp, lp = P.generate_paged(lambda *a: ir.run(paged, *a), prompt, 10, C, n_blocks=16, width=6,
                              order=_order(16), fill=math.pi)
    assert torch.equal(td, tp)
    assert torch.equal(ld, lp)
    assert len(set(td.reshape(-1).tolist())) > 6   # the decode really moves


def test_paged_decode_matches_the_torch_model_eagerly():
    model, _, paged = steps()
    m64 = P.copy.deepcopy(model).double()
    prompt = _prompt(2, 4, seed=3)
    with torch.no_grad():
        te, le = P.generate_paged(m64.paged_step, prompt, 6, C, 12, 4, order=_order(12, 4))
    tp, lp = P.generate_paged(lambda *a: ir.run(paged, *a), prompt, 6, C, 12, 4, order=_order(12, 4))
    assert torch.equal(te, tp)
    assert ((le - lp).abs().max() / le.abs().max()).item() < 1e-13


def test_ragged_batch_matches_each_sequence_alone():
    """Two sequences of different lengths decode in one batch from one pool; each matches its own
    dense decode at batch 1 (per sequence positions are what the dense step cannot express)."""
    _, dense, paged = steps()
    a, b = _prompt(1, 6, seed=5), _prompt(1, 2, seed=6)
    pool = P.BlockPool(20, C.block, _order(20, 7))
    kp = torch.full((C.layers, 20 * C.block, C.kv_heads, C.head_dim), 0.25, dtype=torch.float64)
    vp = kp.clone()
    nxt, lens = [], []
    for prompt in (a, b):          # prefill each sequence alone into the shared pool
        s = pool.add_sequence()
        n = prompt.shape[1]
        pool.reserve(s, n)
        pos = torch.arange(n)[None]
        logits, kp, vp = ir.run(paged, prompt, pos, torch.tensor([pool.slots(s, range(n))]),
                                pool.table([s], 5), kp, vp)
        nxt.append(int(logits[0, -1].argmax()))
        lens.append(n)
    got = [[nxt[0]], [nxt[1]]]
    got_logits = [[], []]
    for step in range(5):          # then decode both together
        for s in (0, 1):
            pool.reserve(s, lens[s] + step + 1)
        ids = torch.tensor([[got[0][-1]], [got[1][-1]]])
        pos = torch.tensor([[lens[0] + step], [lens[1] + step]])
        slots = torch.tensor([pool.slots(0, [lens[0] + step]), pool.slots(1, [lens[1] + step])])
        logits, kp, vp = ir.run(paged, ids, pos, slots, pool.table([0, 1], 5), kp, vp)
        for s in (0, 1):
            got[s].append(int(logits[s, -1].argmax()))
            got_logits[s].append(logits[s, -1])
    for s, prompt in enumerate((a, b)):
        want, want_logits = P.generate_dense(lambda *x: ir.run(dense, *x), prompt, 6, C, capacity=20)
        assert want[0].tolist() == got[s]
        g = torch.stack(got_logits[s])
        w = want_logits[1:, 0]
        assert ((g - w).abs().max() / w.abs().max()).item() < 1e-13


def test_a_wrong_block_table_changes_the_tokens():
    """Tests can fail: a block table pointing one block off reads another sequence's cache."""
    _, dense, paged = steps()
    prompt = _prompt(2, 5)
    td, _ = P.generate_dense(lambda *a: ir.run(dense, *a), prompt, 8, C, capacity=24)

    def shifted(ids, pos, slots, table, kp, vp):
        return ir.run(paged, ids, pos, slots, (table + 1) % 16, kp, vp)

    tp, _ = P.generate_paged(shifted, prompt, 8, C, 16, 6, order=_order(16), fill=math.pi)
    assert not torch.equal(td, tp)


def test_out_of_range_slot_is_an_error_in_the_reference_and_compiled():
    from truestar.compiler import compile_program

    _, _, paged = steps()
    ids = _prompt(1, 2)
    pos = torch.arange(2)[None]
    kp = torch.zeros(C.layers, 8 * C.block, C.kv_heads, C.head_dim, dtype=torch.float64)
    bad_slots = torch.tensor([[0, 8 * C.block]])
    table = torch.tensor([[0, 1]])
    with pytest.raises(ir.IRError, match="out of range"):
        ir.run(paged, ids, pos, bad_slots, table, kp, kp.clone())
    cp = compile_program(paged, "cuda" if CUDA else "cpu", "float32")
    cp([ids, pos, torch.tensor([[0, 1]]), table, kp.float(), kp.float()])   # a good call first
    with pytest.raises(ir.IRError, match="index_copy: an index is out of range or repeated"):
        cp([ids, pos, bad_slots, table, kp.float(), kp.float()])
    with pytest.raises(ir.IRError, match="repeated"):
        cp([ids, pos, torch.tensor([[3, 3]]), table, kp.float(), kp.float()])


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_compiled_paged_equals_compiled_dense_bitwise_on_cuda():
    from truestar.compiler import STATS, compile_program

    _, dense, paged = steps()
    cd = compile_program(dense, "cuda", "float32")
    cp = compile_program(paged, "cuda", "float32")
    prompt = _prompt(2, 5)
    order = _order(16)
    run_d = lambda: P.generate_dense(lambda *a: cd(list(a)), prompt, 10, C, 24, device="cuda",  # noqa: E731
                                     dtype=torch.float32)
    run_p = lambda: P.generate_paged(lambda *a: cp(list(a)), prompt, 10, C, 16, 6, order=order,  # noqa: E731
                                     device="cuda", dtype=torch.float32, fill=math.pi)
    td, ld = run_d()
    tp, lp = run_p()
    assert torch.equal(td, tp) and torch.equal(ld, lp)
    assert not cd.report.fallbacks and not cp.report.fallbacks
    ref, _ = P.generate_dense(lambda *a: ir.run(dense, *a), prompt, 10, C, capacity=24)
    assert torch.equal(td.cpu(), ref)                 # float32 compiled tokens = float64 reference tokens
    before = dict(STATS)
    run_p()
    assert STATS["regions"] == before["regions"]       # no recompiles across positions


def test_weights_from_hugging_face_decode_like_generate():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                   tie_word_embeddings=False, attn_implementation="sdpa")
    torch.manual_seed(0)
    hf = transformers.LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in hf.model.layers.parameters():
            if p.dim() == 2:
                p.mul_(3.0)
    mine = P.from_hf(hf)
    c = P.Cfg(**{**mine.c.__dict__, "block": 4})
    mine.c = c
    _, paged = P.import_steps(mine)
    prompt = _prompt(2, 5, seed=9)
    tp, _ = P.generate_paged(lambda *a: ir.run(paged, *a), prompt, 8, c, 16, 4, order=_order(16))
    hf64 = hf.double()
    with torch.no_grad():
        out = hf64.generate(prompt, max_new_tokens=8, min_new_tokens=8, do_sample=False, pad_token_id=0,
                            eos_token_id=None)
    assert torch.equal(out[:, 5:], tp)
