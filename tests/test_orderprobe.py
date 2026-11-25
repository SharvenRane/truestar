"""The order probe (truestar.orderprobe, docs/ORDERPROBE.md): recover the summation tree of a
reduction from its float32 outputs, describe it in words, and replay it bit for bit."""

import json

import numpy as np
import pytest
import torch

from truestar import orderprobe as OP


def _seq(ix):
    t = ix[0]
    for i in ix[1:]:
        t = (t, i)
    return t


def _pw(ix):
    if len(ix) == 1:
        return ix[0]
    h = len(ix) // 2
    return (_pw(ix[:h]), _pw(ix[h:]))


N = 64
TREES = {
    "sequential, one term at a time in index order": _seq(list(range(N))),
    "from the last index down": _seq(list(range(N))[::-1]),
    "pairwise: a balanced binary tree": _pw(list(range(N))),
    "pairwise over 8 contiguous blocks of 8; inside each block: sequential":
        _pw([_seq(list(range(b, b + 8))) for b in range(0, N, 8)]),
    "split into 4 contiguous chunks of 16; inside each chunk: sequential":
        _seq([_seq(list(range(b, b + 16))) for b in range(0, N, 16)]),
    "8 interleaved accumulators (term i goes to accumulator i mod 8); each: sequential":
        _pw([_seq(list(range(l, N, 8))) for l in range(8)]),
    "a balanced tree over interleaved halves": None,
}


def _halving(ix):
    if len(ix) == 1:
        return ix[0]
    return (_halving(ix[0::2]), _halving(ix[1::2]))


TREES["a balanced tree over interleaved halves"] = _halving(list(range(N)))


def _sum_target(tree, n=N, rows=16, acc=np.float32):
    return OP.Target("sum", "synthetic", n, rows, run=lambda X: OP.replay(tree, X.astype(np.float32), acc))


@pytest.mark.parametrize("words", list(TREES))
def test_sum_trees_are_recovered_described_and_replayed(words):
    tree = TREES[words]
    r = OP.probe(_sum_target(tree))
    assert OP.canonical(r.tree) == OP.canonical(tree)
    assert r.description.startswith(words) or words in r.description, r.description
    assert r.verified and r.accumulator_bits == 24 and r.codes == ["TS962"]


@pytest.mark.parametrize("words", ["sequential, one term", "pairwise: a balanced", "8 interleaved accumulators"])
def test_softmax_denominator_trees_are_recovered(words):
    tree = next(t for w, t in TREES.items() if w.startswith(words))

    def run(X):
        e = np.exp(X - X.max(1, keepdims=True)).astype(np.float32)
        s = OP.replay(tree, e).astype(np.float32)
        return (e / s[:, None]).astype(np.float64)
    r = OP.probe(OP.Target("softmax", "synthetic", N, 64, run=run))
    assert OP.canonical(r.tree) == OP.canonical(tree), r.description


def test_a_wider_accumulator_is_found():
    """A float64 accumulator absorbs nothing at the float32 scale; the probe retries at 53 bits."""
    tree = _pw([_seq(list(range(b, b + 8))) for b in range(0, 32, 8)])
    tg = OP.Target("sum", "synthetic", 32, 8,
                   run=lambda X: OP.replay(tree, X.astype(np.float32), np.float64).astype(np.float32).astype(np.float64))
    r = OP.probe(tg)
    assert r.accumulator_bits == 53 and OP.canonical(r.tree) == OP.canonical(tree) and r.verified


def test_a_run_dependent_order_is_reported_not_described():
    rng = np.random.default_rng(0)

    def run(X):           # a fresh random split every call, as atomics give
        k = int(rng.integers(1, N - 1))
        t = (_seq(list(range(k))), _seq(list(range(k, N))))
        return OP.replay(t, X.astype(np.float32))
    r = OP.probe(OP.Target("sum", "synthetic", N, 16, run=run))
    assert not r.repeatable and "TS963" in r.codes and r.tree is None


def test_a_method_that_is_no_tree_is_unresolved():
    def kahan(X):
        X = X.astype(np.float32)
        s = np.zeros(len(X), np.float32)
        c = np.zeros(len(X), np.float32)
        for k in range(X.shape[1]):
            y = (X[:, k] - c).astype(np.float32)
            t = (s + y).astype(np.float32)
            c = ((t - s).astype(np.float32) - y).astype(np.float32)
            s = t
        return s.astype(np.float64)
    r = OP.probe(OP.Target("sum", "synthetic", 32, 8, run=kahan))
    assert "TS964" in r.codes and "TS962" not in r.codes


def test_replay_rounds_once_per_node():
    x = np.array([[2.0 ** 24, 1.0, 1.0]], dtype=np.float32)
    assert OP.replay(((0, 1), 2), x)[0] == 2.0 ** 24           # each 1 is lost on its own
    assert OP.replay((0, (1, 2)), x)[0] == 2.0 ** 24 + 2       # together they survive
    assert OP.replay((0, 1, 2), x)[0] == 2.0 ** 24 + 2         # one rounding for all three


def test_torch_cpu_sum_and_linear_replay_bit_for_bit():
    r = OP.probe("sum", "torch:cpu", n=256, rows=8)
    assert r.verified and r.repeatable and r.accumulator_bits == 24, r.to_text()
    lin = OP.probe("linear", "torch:cpu", n=96, rows=16, cols=8)
    assert lin.verified, lin.to_text()
    assert lin.product_model, lin.to_text()


def test_truestar_cpu_kernel_accumulates_in_float64():
    """docs/KERNEL_IR.md: the numba backend adds in float64 inside its loops; the probe finds it,
    and the probe's cancelling inputs pass the kernel check (size and lean, measured against the
    magnitude of the sum's terms)."""
    pytest.importorskip("numba")
    r = OP.probe("sum", "truestar:cpu", n=64, rows=8)
    assert r.accumulator_bits == 53 and r.verified, r.to_text()
    assert r.description.startswith("sequential")


def test_compare_files_names_a_different_order(tmp_path):
    a = OP.probe(_sum_target(TREES["pairwise: a balanced binary tree"]))
    b = OP.probe(_sum_target(_seq(list(range(N)))))
    a.label = b.label = "row sum"
    for p, r, dev in ((tmp_path / "a.json", a, "gpu A"), (tmp_path / "b.json", b, "gpu B")):
        p.write_text(json.dumps({"env": {"device": dev}, "results": [r.as_dict()]}))
    lines, codes = OP.compare_files(str(tmp_path / "a.json"), str(tmp_path / "b.json"))
    assert codes == ["TS965"] and any("row sum: different orders" in l for l in lines), lines
    lines, codes = OP.compare_files(str(tmp_path / "a.json"), str(tmp_path / "a.json"))
    assert codes == [] and "same order" in lines[-1]


def test_cli_runs_and_writes_json(tmp_path, capsys):
    from truestar.cli import main
    out = tmp_path / "o.json"
    code = main(["orderprobe", "--op", "sum", "--n", "48", "--rows", "4", "--backends", "torch:cpu",
                 "--json", str(out)])
    assert code == 0
    text = capsys.readouterr().out
    assert "order:" in text and "TS962" in text
    d = json.loads(out.read_text())
    assert d["results"][0]["verified"] is True and d["results"][0]["tree"] is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_cublas_linear_on_the_gpu_replays_with_fused_products():
    r = OP.probe("linear", "torch:cuda", n=512, rows=128, cols=128)
    assert r.verified and r.repeatable, r.to_text()
    assert r.product_model.startswith("fused multiply add"), r.to_text()


MPS = torch.backends.mps.is_available()


@pytest.mark.skipif(not MPS, reason="needs an MPS device")
@pytest.mark.parametrize("backend", ["torch:mps", "truestar:mps"])
def test_mps_and_metal_orders_are_recovered_and_replayed(backend):
    r = OP.probe("sum", backend, n=64, rows=16)
    assert r.verified and r.repeatable and r.accumulator_bits == 24, r.to_text()
    r = OP.probe("linear", backend, n=64, rows=16, cols=16)
    assert r.verified and "sequential" in r.to_text(), r.to_text()
