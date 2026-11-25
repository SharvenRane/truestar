"""Speculative decoding equal to greedy decoding (truestar.specdec, docs/SPEC_DECODE.md)."""

import copy
import os
import sys

import pytest
import torch

from truestar import codes
from truestar import specdec as SD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples"))
import llama_paged_decode as P  # noqa: E402

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CAP = 96


def model(seed, layers=2, dim=64):
    torch.manual_seed(seed)
    c = P.Cfg(vocab=64, dim=dim, heads=4, kv_heads=2, head_dim=dim // 4, layers=layers, ffn=2 * dim, block=16)
    return P.Llama(c).eval()


def cache(m, device="cpu", dtype=torch.float64):
    k = torch.zeros(m.c.layers, 1, m.c.kv_heads, CAP, m.c.head_dim, device=device, dtype=dtype)
    return k, torch.zeros_like(k)


def eager(m):
    m = copy.deepcopy(m).double()

    def step(ids, pos, kc, vc):
        with torch.no_grad():
            return m.dense_step(ids, pos, kc, vc)
    return step


def test_codes():
    assert codes.slug("TS705") == "spec_rows_variant" and codes.slug("TS706") == "spec_rows_invariant"


@pytest.mark.parametrize("same_draft", [True, False])
@pytest.mark.parametrize("k", [1, 3, 5])
def test_speculative_tokens_are_greedy_tokens_float64(same_draft, k):
    t, d = model(0), model(0 if same_draft else 1, layers=1, dim=32 if not same_draft else 64)
    if same_draft:
        d = t
    prompt = torch.randint(0, 64, (1, 9), generator=torch.Generator().manual_seed(4))
    ref = SD.greedy(eager(t), prompt, 40, *cache(t))
    out, st = SD.speculative(eager(t), eager(d), prompt, 40, k, *cache(t), *cache(d))
    assert torch.equal(out, ref)
    if same_draft:
        assert st["acceptance"] == 1.0          # the draft is the target: every proposal is kept
    assert st["rounds"] <= 39


def test_a_row_dependent_step_is_ts705():
    t = model(0)
    inner = eager(t)

    def skewed(ids, pos, kc, vc):        # a kernel whose rounding follows the row count
        lg, kc, vc = inner(ids, pos, kc, vc)
        return lg + 1e-9 * ids.shape[1], kc, vc
    prompt = torch.randint(0, 64, (1, 9), generator=torch.Generator().manual_seed(4))
    bad = SD.check_rows(skewed, prompt, *cache(t), rows=4)
    assert not bad.invariant and bad.codes == ["TS705"] and bad.first_row == 0


def test_eager_pytorch_is_not_row_invariant_even_in_float64():
    """The reason speculative decoding is lossless only in exact arithmetic: eager PyTorch's matmuls
    round a 4 row call differently from 4 one row calls (measured about 1e-15 in float64 on CPU)."""
    t = model(0)
    prompt = torch.randint(0, 64, (1, 9), generator=torch.Generator().manual_seed(4))
    r = SD.check_rows(eager(t), prompt, *cache(t), rows=4)
    assert not r.invariant and r.codes == ["TS705"] and r.max_abs < 1e-12


@gpu
def test_compiled_deterministic_target_is_row_invariant_and_speculative_is_exact():
    from truestar.compiler import compile_program
    t, d = model(0, layers=3), model(1, layers=1, dim=32)
    cps = {}
    for name, m in (("t", t), ("d", d)):
        ir, _ = P.import_steps(m, capacity=CAP)       # traced at batch 2: B stays symbolic
        cp = compile_program(ir, "cuda", "float32", deterministic=True)
        cps[name] = (lambda cp: (lambda ids, pos, kc, vc: cp([ids, pos, kc, vc])))(cp)
    prompt = torch.randint(0, 64, (1, 12), generator=torch.Generator().manual_seed(5))
    for rows in (2, 4, 7):
        r = SD.check_rows(cps["t"], prompt, *cache(t, "cuda", torch.float32), rows=rows)
        assert r.invariant and r.codes == ["TS706"]
    ref = SD.greedy(cps["t"], prompt, 48, *cache(t, "cuda", torch.float32))
    for k in (2, 4):
        out, _ = SD.speculative(cps["t"], cps["d"], prompt, 48, k, *cache(t, "cuda", torch.float32),
                                *cache(d, "cuda", torch.float32))
        assert torch.equal(out, ref)
