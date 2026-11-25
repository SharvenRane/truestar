"""Prefix cache check (truestar.prefixcheck, `truestar prefixcheck`, docs/PREFIX_CHECK.md)."""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from truestar import codes
from truestar import prefixcheck as PC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def logits(seed=0, shape=(2, 10, 50)):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def test_codes_are_registered():
    assert codes.slug("TS702") == "prefix_cache_divergent"
    assert codes.slug("TS703") == "prefix_cache_wrong"
    assert codes.slug("TS704") == "prefix_cache_bitwise"


def test_bitwise_runs_report_ts704():
    a = logits()
    rep = PC.compare(a, a.clone())
    assert rep.bitwise and rep.codes == ["TS704"] and rep.first_token is None


def test_divergence_names_the_first_token_and_layer():
    a = logits()
    b = a.clone()
    b[1, 6, 3] += 1e-4
    layers_a = {"layer0.k": torch.ones(3), "layer1.k": torch.ones(3), "layer2.k": torch.ones(3)}
    layers_b = {"layer0.k": torch.ones(3), "layer1.k": torch.ones(3) + 1e-6, "layer2.k": torch.ones(3) + 1e-5}
    rep = PC.compare(a, b, layers_a, layers_b)
    assert not rep.bitwise and rep.first_token == 6 and rep.first_layer == "layer1.k"
    assert rep.codes == ["TS702"] and abs(rep.max_abs - 1e-4) < 1e-6
    assert "first layer that differs: layer1.k" in rep.to_text()


def test_a_less_accurate_cached_run_is_ts703():
    ref = logits().double()
    full = (ref + 1e-7).float()
    cached = (ref + 1e-1).float()
    rep = PC.compare(full, cached, reference=ref)
    assert "TS703" in rep.codes and rep.error_cached > rep.error_full
    near = PC.compare(full, (ref - 1e-7).float(), reference=ref)
    assert "TS703" not in near.codes


def test_files_and_the_command(tmp_path):
    a = logits()
    b = a.clone()
    b[0, 2, 0] += 1.0
    torch.save({"logits": a, "layers": {"l0": torch.zeros(2)}}, tmp_path / "full.pt")
    torch.save({"logits": b, "layers": {"l0": torch.zeros(2)}}, tmp_path / "cached.pt")
    np.savez(tmp_path / "same.npz", logits=a.numpy(), **{"layers/l0": np.zeros(2)})
    rep = PC.check_files(str(tmp_path / "full.pt"), str(tmp_path / "cached.pt"))
    assert rep.first_token == 2
    assert PC.check_files(str(tmp_path / "full.pt"), str(tmp_path / "same.npz")).bitwise
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    run = lambda *x: subprocess.run([sys.executable, "-m", "truestar", "prefixcheck", *x],  # noqa: E731
                                    capture_output=True, text=True, env=env)
    p = run(str(tmp_path / "full.pt"), str(tmp_path / "same.npz"))
    assert p.returncode == 0 and "BITWISE" in p.stdout
    p = run(str(tmp_path / "full.pt"), str(tmp_path / "cached.pt"), "--format", "agent")
    assert p.returncode == 1 and '"TS702"' in p.stdout
    p = run(str(tmp_path / "full.pt"), str(tmp_path / "missing.pt"))
    assert p.returncode == 2


@gpu
def test_truestar_dense_paged_and_decode_caches_give_the_full_prefill_bits():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import prefix_check_llama as H
    rows = H.run(length=96, splits=(1, 40, 80), caches=("dense", "paged", "dense-decode", "int8"))
    for r in rows:
        if r["cache"] in ("dense", "paged", "dense-decode"):
            assert r["bitwise"], r
        else:          # a quantized prefix is not the full prefill's cache: rounding, named
            assert not r["bitwise"] and r["issues"][0]["code"] == "TS702" and r["first_token"] == 0


@gpu
def test_a_split_kv_decode_differs_from_the_prefill_row_and_is_named():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import prefix_check_llama as H
    long_ = H.attention_case(keys=2200)
    assert not long_["bitwise"] and 256 in long_["kv_chunks"]
    assert [i["code"] for i in long_["issues"]] == ["TS702"]          # rounding, not less accurate
    assert H.attention_case(keys=1000)["bitwise"]
