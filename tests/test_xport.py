"""Cross framework port check: JAX/Flax NNX and Keras 3 candidates against a PyTorch reference
(docs/PORT_ENGINES.md, pain row P32). Each candidate runs in its own process in float64; weights
cross by safetensors, activations by DLPack."""

import json
import os
import subprocess
import sys

import importlib.util

import pytest
import torch

# only the child processes import the frameworks (keras picks its backend at import)
if not all(importlib.util.find_spec(m) for m in ("jax", "flax", "keras")):
    pytest.skip("jax, flax and keras are needed", allow_module_level=True)

import engine_models as em  # noqa: E402
from truestar.engines import xport  # noqa: E402

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)
IDS = torch.tensor([em.reference_tokenize(em.PROMPT)])


@pytest.fixture(scope="module")
def model():
    return em.build()


def cand(tmp_path, module, **kw):
    """A candidate file that builds the port with fault switches."""
    f = tmp_path / f"cand_{module}_{abs(hash(repr(sorted(kw.items()))))}.py"
    args = ", ".join(f"{k}={v!r}" for k, v in kw.items())
    f.write_text(f"import sys\nsys.path.insert(0, {TESTS!r})\nfrom {module} import build\n"
                 f"def get_model(w):\n    return build(w{', ' + args if args else ''})\n", encoding="utf-8")
    return str(f)


def run(model, kind, path, backends=None):
    return xport.framework_port_check(model, IDS, kind, path, backends=backends)


def test_correct_flax_port_is_clean(model):
    r = run(model, "jax", os.path.join(TESTS, "xport_flax.py"))
    b = r.backends[0]
    assert r.status == "ok" and not r.codes, r.to_text()
    assert b.output_rel_err < 1e-13 and len(b.pairs) >= 25
    assert b.edge_kinds == {"equal": 6, "transpose": 15}  # Flax kernels are [in, out]
    assert not b.lower_precision


@pytest.mark.parametrize("kw,module,words", [
    ({"eps": 1e-6}, "blocks.0.attn_norm", None),
    ({"theta": 500000.0}, "blocks.0.attn", None),
    ({"pairs": True}, "blocks.0.attn", None),
    ({"untransposed": ("model.layers.1.self_attn.o_proj.weight",)}, "blocks.1.attn.wo", "missed a transpose"),
])
def test_flax_port_bugs_are_named_at_their_module(model, tmp_path, kw, module, words):
    r = run(model, "jax", cand(tmp_path, "xport_flax", **kw))
    b = r.backends[0]
    assert r.status == "divergent" and "TS653" in r.codes, r.to_text()
    assert b.blamed.candidate == module
    if words:
        assert words in b.hint
    # the final output alone says only "differs"
    assert b.output_rel_err > 1e-4


def test_correct_keras_port_is_clean_on_jax_and_torch(model):
    r = run(model, "keras", os.path.join(TESTS, "xport_keras.py"), ["jax", "torch"])
    assert r.status == "ok", r.to_text()
    assert [b.status for b in r.backends] == ["ok", "ok"]
    assert all(len(b.pairs) >= 25 for b in r.backends)
    # Keras forces 64 bit results to 32 bit outside TensorFlow: reported, not a false alarm
    assert "TS658" in r.codes and all(b.lower_precision for b in r.backends)
    assert "float32 noise" in r.explanation


def test_a_bug_on_one_keras_backend_is_named_with_the_backend(model, tmp_path):
    r = run(model, "keras", cand(tmp_path, "xport_keras", backend_bug="torch"), ["jax", "torch"])
    jx, th = r.backends
    assert jx.status == "ok" and th.status == "divergent", r.to_text()
    assert th.blamed.candidate == "blocks.0.attn"
    assert "TS656" in r.codes


def test_keras_eps_is_named_on_every_backend(model, tmp_path):
    r = run(model, "keras", cand(tmp_path, "xport_keras", eps=1e-6), ["jax", "torch"])
    assert [b.blamed.candidate for b in r.backends] == ["blocks.0.n1", "blocks.0.n1"], r.to_text()


def _downstream_over(b):
    """Pairs over threshold that neither are the blamed module nor contain it."""
    k = b.blamed.reference
    return [p.candidate for p in b.pairs if p.status != "ok" and p.reference != k
            and not (p.reference == "<model>" or k.startswith(p.reference + "."))]


def test_isolated_replay_keeps_inherited_error_out(model, tmp_path, monkeypatch):
    """With the reference's inputs only the faulty module (and modules containing it) is over
    threshold. Guard: on the candidate's own inputs every later module inherits the error."""
    path = cand(tmp_path, "xport_flax", untransposed=("model.layers.0.self_attn.o_proj.weight",))
    b = run(model, "jax", path).backends[0]
    assert b.blamed.candidate == "blocks.0.attn.wo" and _downstream_over(b) == []
    monkeypatch.setattr(xport, "_substitute", lambda ref_inputs, t: t)
    b = run(model, "jax", path).backends[0]
    assert len(_downstream_over(b)) >= 10


def test_missing_framework_is_loud(model, tmp_path):
    with pytest.raises(xport.FrameworkError) as e:
        run(model, "keras", os.path.join(TESTS, "xport_keras.py"), ["no_such_backend"])
    assert e.value.code in ("TS652", "TS021")


def test_cli_agent_json(tmp_path):
    ref = tmp_path / "ref.py"
    ref.write_text(f"import sys\nsys.path.insert(0, {TESTS!r})\nimport torch, engine_models as em\n"
                   "def get_model():\n    return em.build()\n"
                   "def get_inputs():\n    return torch.tensor([em.reference_tokenize(em.PROMPT)])\n", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"), PYTHONIOENCODING="utf-8")
    bad = cand(tmp_path, "xport_flax", theta=500000.0)
    p = subprocess.run([sys.executable, "-m", "truestar", "port", str(ref), "jax:" + bad, "--format", "agent"],
                       cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert p.returncode == 1, p.stdout + p.stderr
    doc = json.loads(p.stdout)
    assert doc["command"] == "port" and doc["candidate"]["kind"] == "jax" and "TS653" in doc["codes"]
    assert doc["backends"][0]["blamed"]["candidate"] == "blocks.0.attn"
    assert doc["next_actions"][0]["kind"] == "inspect_candidate_module"
