"""truestar serve: continuous batching, prefix cache, chunked prefill, keyed sampling, the HTTP subset,
and the guarantee (the same request gives the same bits alone, batched, cached and chunked)."""

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch

pytest.importorskip("numba")

from truestar.ir.reference import uniform01  # noqa: E402
from truestar.serve import engine as E  # noqa: E402
from truestar.serve.check import _requests, check_invariance, drive  # noqa: E402
from truestar.serve.engine import Engine, Request, ServeError, StepRunner, sample  # noqa: E402
from truestar.serve.model import TinyLM, config  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return TinyLM(config("tiny"))


@pytest.fixture(scope="module")
def cpu(model):
    return StepRunner(model, "cpu")


@pytest.fixture(scope="module")
def ref(model):
    return StepRunner(model, "reference")


def test_guarantee_and_float64_on_cpu(model, cpu, ref):
    rep = check_invariance(model, cpu, ref)
    assert rep["invariant"] and rep["codes"] == ["TS1400", "TS1408"], rep
    assert len(rep["rows"]) == 12 and all(r["bitwise"] for r in rep["rows"])
    cached = [r for r in rep["rows"] if "prefix cache" in r["scenario"]]
    assert cached and all(r["cached_tokens"] == 2 * model.c.block for r in cached)   # the cache was really read
    f = rep["float64"]
    assert f["tokens_equal"] and f["reference_vs_no_cache_tokens_equal"]
    assert f["float32_logprob_max_error"] < 1e-4 and f["reference_vs_no_cache_logprob_error"] < 1e-9
    assert cpu.fallbacks == 0


class RowCountLeak:
    """A planted fault: logits that depend on how many rows share the call (what split K or a batch
    dependent kernel choice does in other engines)."""

    def __init__(self, inner):
        self.inner, self.c, self.executor = inner, inner.c, "planted"

    def pools(self, n):
        return self.inner.pools(n)

    def __call__(self, ids, *rest):
        lg, kp, vp = self.inner(ids, *rest)
        return lg + 1e-6 * ids.shape[0], kp, vp


def test_a_batch_dependent_step_is_caught(model, cpu):
    rep = check_invariance(model, RowCountLeak(cpu))
    assert rep["codes"] == ["TS1401"] and not rep["invariant"]
    bad = {r["scenario"] for r in rep["rows"] if not r["bitwise"]}
    assert "batched with 7 others" in bad and "alone" not in bad


def test_a_prefix_cache_keyed_without_tokens_is_caught(model, cpu, monkeypatch):
    """A planted prefix cache fault: blocks keyed by position only, so a cached block of another
    prompt is reused. The prefix cache scenarios must fail."""
    monkeypatch.setattr(E, "chain", lambda prev, toks: E.hashlib.blake2b(prev + b"x", digest_size=16).digest())
    rep = check_invariance(model, cpu)
    bad = {r["scenario"] for r in rep["rows"] if not r["bitwise"]}
    assert "TS1401" in rep["codes"] and "prefix cache on" in bad and "chunked prefill (5 tokens)" not in bad


def test_keyed_sampling_depends_on_seed_and_index_only():
    g = np.random.Generator(np.random.PCG64(0))
    lg = g.standard_normal(50)
    a = [sample(lg, 0.9, 1.0, 77, i) for i in range(40)]
    assert a == [sample(lg.copy(), 0.9, 1.0, 77, i) for i in range(40)]
    assert a != [sample(lg, 0.9, 1.0, 78, i) for i in range(40)]
    # the draw is the IR multinomial's uniform: the inverse CDF at uniform01(seed, index, 0)
    w = np.exp((lg - lg.max()) / 0.9)
    cum = np.cumsum(w)
    assert a[5] == int(np.searchsorted(cum, uniform01(77, 5, 0) * cum[-1], side="right"))
    # frequencies follow the distribution
    p = w / w.sum()
    n = 4000
    counts = np.bincount([sample(lg, 0.9, 1.0, 5, i) for i in range(n)], minlength=50) / n
    assert np.abs(counts - p).max() < 0.03
    assert sample(lg, 0.0, 1.0, 0, 0) == int(np.argmax(lg))
    # top p keeps the smallest set whose mass reaches p
    top = {sample(lg, 1.0, 0.3, 9, i) for i in range(300)}
    order = np.argsort(-w)
    k = int(np.searchsorted(np.cumsum(w[order]) / w.sum(), 0.3)) + 1
    assert top <= set(order[:k].tolist())


def test_continuous_batching_admits_and_retires(model, cpu):
    eng = Engine(model, runner=cpu, max_batch=3, chunk=4)
    seqs = [eng.submit(Request([1 + i] * (3 + 2 * i), max_tokens=4 + i)) for i in range(6)]
    peak = 0
    while eng.step() or eng.waiting:
        peak = max(peak, len(eng.running))
    assert peak == 3 and all(s.done.is_set() and len(s.out) == s.req.max_tokens for s in seqs)
    assert all(s.finish == "length" for s in seqs)
    assert eng.blocks.available() == model.c.n_blocks       # every block released (cached ones are idle)


def test_prefix_cache_blocks_are_evicted_and_reused(model, cpu):
    from dataclasses import replace

    small = TinyLM(replace(model.c, n_blocks=8))
    eng = Engine(small, runner=cpu, max_batch=1)      # same weights and step program, a pool of 8 blocks
    outs = []
    for i in range(5):    # each request fills 4 of the 8 blocks; the cache must give up old blocks
        s = drive(eng, [(0, Request([10 + i] * 25, max_tokens=6))])[0]
        outs.append(s.out)
    again = drive(eng, [(0, Request([14] * 25, max_tokens=6))])[0]
    assert again.cached == 24 and again.out == outs[4]
    fresh = drive(Engine(small, runner=cpu, max_batch=1, prefix_cache=False), [(0, Request([10] * 25, max_tokens=6))])[0]
    assert fresh.out == outs[0]


def test_requests_are_refused_by_name(model, cpu):
    eng = Engine(model, runner=cpu)
    with pytest.raises(ServeError) as e:
        eng.submit(Request([1] * 120, max_tokens=20))
    assert e.value.code == "TS1403"
    for bad in (Request([], 4), Request([model.c.vocab], 4), Request([1], 0), Request([1], 4, top_p=0.0)):
        with pytest.raises(ServeError) as e:
            eng.submit(bad)
        assert e.value.code == "TS1404"


def test_stop_strings_and_stop_ids(model, cpu):
    base = drive(Engine(model, runner=cpu), [(0, Request(list(b"hello"), max_tokens=12))])[0]
    stop = bytes([base.out[3]])
    s = drive(Engine(model, runner=cpu), [(0, Request(list(b"hello"), max_tokens=12, stop_bytes=(stop,)))])[0]
    first = next(i for i, t in enumerate(base.out) if bytes([t]) == stop) if base.out[3] < 256 else None
    if first is not None:
        assert s.finish == "stop" and s.out == base.out[:first + 1]
    s2 = drive(Engine(model, runner=cpu), [(0, Request(list(b"hello"), max_tokens=12, stop_ids=(base.out[2],)))])[0]
    assert s2.finish == "stop" and s2.out[-1] == base.out[2] and s2.out == base.out[:len(s2.out)]


# ----------------------------------------------------------------------------- HTTP


@pytest.fixture(scope="module")
def server(model, cpu):
    from truestar.serve.http import serve

    eng = Engine(model, runner=cpu, max_batch=8, chunk=6)
    srv = serve(eng, "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    eng.stop()
    srv.server_close()


def _post(url, body, raw=False):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
            return r.status, (data if raw else json.loads(data))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_completions_match_the_engine_alone(model, cpu, server):
    body = {"model": "truestar-tiny", "prompt": "The cache", "max_tokens": 10, "temperature": 0.7, "seed": 42,
            "logprobs": 2}
    code, out = _post(server + "/v1/completions", body)
    assert code == 200 and out["object"] == "text_completion"
    ch = out["choices"][0]
    alone = drive(Engine(model, runner=cpu, prefix_cache=False),
                  [(0, Request(list(b"The cache"), max_tokens=10, temperature=0.7, seed=42, top_logprobs=2))])[0]
    assert ch["logprobs"]["token_ids"] == alone.out and ch["logprobs"]["token_logprobs"] == alone.logprobs
    assert len(ch["logprobs"]["top_logprobs"]) == 10 and ch["finish_reason"] == "length"
    assert out["usage"] == {"prompt_tokens": 9, "completion_tokens": 10, "total_tokens": 19,
                            "prompt_tokens_details": {"cached_tokens": 0}}


def test_http_concurrent_requests_give_the_alone_bits(model, cpu, server):
    prompts = [f"request number {i} " * (1 + i % 3) for i in range(8)]
    results = [None] * 8

    def one(i):
        results[i] = _post(server + "/v1/completions", {"prompt": prompts[i], "max_tokens": 12, "temperature": 1.0,
                                                        "seed": i, "logprobs": 0})[1]

    th = [threading.Thread(target=one, args=(i,)) for i in range(8)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    for i in (0, 5):
        alone = drive(Engine(model, runner=cpu, prefix_cache=False),
                      [(0, Request(list(prompts[i].encode()), max_tokens=12, temperature=1.0, seed=i))])[0]
        lp = results[i]["choices"][0]["logprobs"]
        assert lp["token_ids"] == alone.out and lp["token_logprobs"] == alone.logprobs


def test_http_chat_stream_and_refusals(server):
    msgs = [{"role": "user", "content": "hi"}]
    code, out = _post(server + "/v1/chat/completions", {"messages": msgs, "max_tokens": 6, "temperature": 0,
                                                         "logprobs": True, "top_logprobs": 1})
    assert code == 200 and out["object"] == "chat.completion"
    content = out["choices"][0]["logprobs"]["content"]
    assert len(content) == 6 and all(len(c["top_logprobs"]) == 1 for c in content)
    # greedy: the chosen token is the top one
    assert all(c["top_logprobs"][0]["logprob"] == c["logprob"] for c in content)
    code, raw = _post(server + "/v1/chat/completions", {"messages": msgs, "max_tokens": 6, "temperature": 0,
                                                         "stream": True}, raw=True)
    events = [json.loads(l[6:]) for l in raw.decode().split("\n\n") if l.startswith("data: {")]
    assert raw.decode().rstrip().endswith("data: [DONE]")
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    assert text == out["choices"][0]["message"]["content"] and events[-1]["choices"][0]["finish_reason"] == "length"
    # sampled without a seed: one is assigned and returned, and replaying it gives the same text
    code, a = _post(server + "/v1/completions", {"prompt": "x", "max_tokens": 5})
    assert code == 200 and a["truestar"]["codes"] == ["TS1405"]
    code, b = _post(server + "/v1/completions", {"prompt": "x", "max_tokens": 5, "seed": a["truestar"]["seed"]})
    assert b["choices"][0]["text"] == a["choices"][0]["text"] and "codes" not in b["truestar"]
    for bad in ({"prompt": "x", "n": 2}, {"prompt": "x", "frequency_penalty": 0.5}, {"prompt": "x", "bogus": 1},
                {"prompt": "x", "max_tokens": 500}):
        code, err = _post(server + "/v1/completions", bad)
        assert code == 400 and err["error"]["code"] in ("TS1404", "TS1403")
    code, ok = _post(server + "/v1/completions", {"prompt": "x", "n": 1, "frequency_penalty": 0, "max_tokens": 2})
    assert code == 200


@pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.device_count() > 0), reason="needs CUDA (run on Modal L4)")
def test_guarantee_on_cuda_and_cpu_agree(model, cpu):
    cuda = StepRunner(model, "cuda")
    rep = check_invariance(model, cuda, StepRunner(model, "reference"))
    assert rep["codes"] == ["TS1400", "TS1408"], rep
    targets, _, _ = _requests(model.c)
    for t in targets:   # every chip: keyed draws are the same; tokens equal unless logits sit at a boundary
        a = drive(Engine(model, runner=cuda), [(0, t)])[0]
        b = drive(Engine(model, runner=cpu), [(0, t)])[0]
        assert a.out == b.out
        assert max(abs(x - y) for x, y in zip(a.logprobs, b.logprobs)) < 1e-4
