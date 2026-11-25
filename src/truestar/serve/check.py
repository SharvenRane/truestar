"""The serving guarantee, checked: a request gives the same tokens and logprobs, bit for bit, alone,
batched with 7 other requests (some arriving while it decodes), with the prefix cache on and off, and
with chunked prefill on and off. Plus the float64 check: the float32 server's greedy tokens equal the
float64 reference server's, except at a near tie, and the reference server equals a plain causal
forward with no cache at all.

    truestar serve --check [--device cpu|cuda] [--format agent]
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .engine import Engine, Request, StepRunner
from .model import ServeConfig, TinyLM, config


def drive(engine: Engine, arrivals: Sequence[Tuple[int, Request]]) -> List:
    """Submit each request at its step index, run until everything finished; the sequences in order."""
    pending = sorted(enumerate(arrivals), key=lambda x: x[1][0])
    seqs: Dict[int, object] = {}
    step = 0
    while pending or engine.running or engine.waiting:
        while pending and pending[0][1][0] <= step:
            i, (_, r) = pending.pop(0)
            seqs[i] = engine.submit(replace(r))
        engine.step()
        step += 1
    return [seqs[i] for i in range(len(arrivals))]


def _requests(c: ServeConfig, seed: int = 11):
    g = np.random.Generator(np.random.PCG64(seed))
    tok = lambda n: [int(t) for t in g.integers(0, 256, n)]  # noqa: E731
    shared = tok(2 * c.block + 3)                           # two full blocks and a bit
    targets = [Request(shared + tok(7), max_tokens=24, rid="greedy", top_logprobs=3),
               Request(shared + tok(4), max_tokens=24, temperature=0.8, top_p=0.9, seed=1234, rid="sampled",
                       top_logprobs=3)]
    others = [Request(tok(n), max_tokens=m, temperature=t, seed=s)
              for n, m, t, s in ((5, 20, 0.0, 0), (31, 9, 1.0, 7), (2, 30, 0.0, 0), (17, 14, 0.7, 3),
                                 (40, 6, 0.0, 0), (9, 25, 1.2, 9), (23, 12, 0.0, 0))]
    warm = Request(shared + tok(6), max_tokens=4, rid="warm")
    return targets, others, warm


def _key(s) -> Tuple:
    return tuple(s.out), tuple(s.logprobs), tuple(tuple(t) for t in s.top)


def check_invariance(model: TinyLM, runner: StepRunner, reference: Optional[StepRunner] = None) -> Dict:
    c = model.c
    targets, others, warm = _requests(c)
    runs: Dict[str, Dict] = {}

    def scenario(name, prefix_cache, chunk, batched):
        out = {}
        for t in targets:
            eng = Engine(model, runner=runner, prefix_cache=prefix_cache, chunk=chunk, max_batch=8)
            arrivals = []
            if prefix_cache:
                drive(eng, [(0, warm)])                      # puts the shared prefix in the cache
            if batched:
                arrivals = [(0, o) for o in others[:3]] + [(0, t)] + [(2, others[3]), (4, others[4]),
                                                                       (5, others[5]), (9, others[6])]
            else:
                arrivals = [(0, t)]
            seqs = drive(eng, arrivals)
            s = next(x for x in seqs if x.req.rid == t.rid)
            out[t.rid] = {"key": _key(s), "cached_tokens": s.cached, "steps": eng.steps}
        runs[name] = out

    scenario("alone", False, None, False)
    scenario("batched with 7 others", False, None, True)
    scenario("prefix cache on", True, None, False)
    scenario("chunked prefill (5 tokens)", False, 5, False)
    scenario("batched, chunked (3 tokens)", False, 3, True)
    scenario("batched, prefix cache, chunked (3 tokens)", True, 3, True)
    base = runs["alone"]
    rows, ok = [], True
    for name, out in runs.items():
        for rid, r in out.items():
            same = r["key"] == base[rid]["key"]
            ok &= same
            first = None
            if not same:
                a, b = base[rid]["key"][0], r["key"][0]
                la, lb = base[rid]["key"][1], r["key"][1]
                first = next((i for i in range(min(len(a), len(b))) if a[i] != b[i] or la[i] != lb[i]), None)
            rows.append({"scenario": name, "request": rid, "bitwise": same, "first_difference": first,
                         "cached_tokens": r["cached_tokens"], "tokens": len(r["key"][0])})
    rep = {"executor": runner.executor, "invariant": ok, "rows": rows,
           "codes": ["TS1400"] if ok else ["TS1401"]}
    if reference is not None:
        rep["float64"] = check_reference(model, runner, reference, targets[0])
        rep["codes"].append("TS1402" if not rep["float64"]["ok"] else "TS1408")
    return rep


def check_reference(model: TinyLM, runner: StepRunner, reference: StepRunner, req: Request,
                    tie: float = 1e-3) -> Dict:
    """Greedy tokens of the float32 server against the float64 reference server (same step program, IR
    interpreter), and the reference server against a plain causal forward of the whole text."""
    e32 = Engine(model, runner=runner, prefix_cache=False)
    e64 = Engine(model, runner=reference, prefix_cache=False)
    s32 = drive(e32, [(0, req)])[0]
    s64 = drive(e64, [(0, req)])[0]
    # the reference server against no cache at all: teacher forced on its own tokens
    with torch.no_grad():
        full = model.full_logits(torch.tensor([req.prompt + s64.out[:-1]], dtype=torch.int64))[0]
    p = len(req.prompt)
    nocache = full[p - 1:].argmax(-1).tolist()
    lp_full = torch.log_softmax(full[p - 1:], -1)
    ref_lp_err = max(abs(lp_full[i, t].item() - s64.logprobs[i]) for i, t in enumerate(s64.out))
    first = next((i for i, (a, b) in enumerate(zip(s32.out, s64.out)) if a != b), None)
    gap = None
    if first is not None:
        row = full[p - 1 + first]
        gap = abs(row[s32.out[first]].item() - row[s64.out[first]].item()) / row.abs().max().item()
    n = first if first is not None else len(s32.out)
    lp_err = max((abs(a - b) for a, b in zip(s32.logprobs[:n], s64.logprobs[:n])), default=0.0)
    ok = nocache == s64.out and ref_lp_err < 1e-9 and (first is None or gap < tie)
    return {"ok": ok, "tokens_equal": first is None, "first_difference": first, "tie_gap": gap,
            "float32_logprob_max_error": lp_err, "reference_vs_no_cache_tokens_equal": nocache == s64.out,
            "reference_vs_no_cache_logprob_error": ref_lp_err, "tokens": len(s32.out)}


def run_check(preset: str = "tiny", device: str = "cpu", with_reference: bool = True) -> Dict:
    c = config(preset)
    model = TinyLM(c)
    runner = StepRunner(model, device)
    ref = StepRunner(model, "reference") if with_reference else None
    rep = check_invariance(model, runner, ref)
    rep["fallbacks"] = runner.fallbacks
    rep["preset"] = preset
    return rep
