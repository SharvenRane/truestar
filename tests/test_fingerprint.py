"""`truestar fingerprint` and the deterministic launch configs it relies on (docs/CROSS_CHIP.md).

What one machine can check, and is checked here:

* run to run: two fingerprints on this GPU (one in process, one from the command line in a fresh
  process) give the same hash for every deterministic row, and every row's two calls agree;
* batch invariance of deterministic rows on the MLP (no split K, config free contractions);
* deterministic mode tunes a kernel whose bits depend on its launch config (attention, reductions)
  only over the parameters that keep its summation order (source ``order tuned`` or ``pinned``),
  and the bits are the default config's; contractions are still tuned freely;
* ``compare`` names a changed setting, a changed environment key, a changed claimed hash (TS743,
  exit 1), different inputs (TS744) and a file that is not a fingerprint (TS749, exit 2);
* a weights or inputs file changed after it was written is refused (TS740).

What only two machines can check (not run here; see ``test_deterministic_rows_match_another_gpu``):
bitwise equality of deterministic rows across GPUs, architectures, drivers and Triton versions. That
test runs only when ``TRUESTAR_FP_OTHER`` names a fingerprint JSON written on another machine.
"""

import copy
import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_kcache_"))

from truestar import kernel as K  # noqa: E402
from truestar.fingerprint import compare, compare_files, load, run  # noqa: E402
from truestar.fingerprint import data as fpdata  # noqa: E402
from truestar.fingerprint.models import CASES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")


def _has_triton():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


HAS_GPU = _has_triton()
needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="needs CUDA and Triton")
SMALL = dict(cases=["mlp", "llama_step"], dtypes=["float32"], batches=[1, 7], arms=["truestar_det", "eager"])


def _cli(*argv, cwd=None):
    env = dict(os.environ, PYTHONPATH=SRC, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-m", "truestar", "fingerprint", *argv], cwd=cwd, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)


# ----------------------------------------------------------------- documents without a GPU

def _doc(gpu="GPU A", h="aa", tf32=False):
    st = {"torch_deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False,
          "matmul_allow_tf32": tf32, "cudnn_allow_tf32": tf32}
    row = {"batch": 7, "outputs": [h * 32], "again": [h * 32], "run_to_run_equal": True, "row0": [h * 32]}
    return {"schema": "truestar.fingerprint/1", "fresh_cache": True,
            "env": {"gpu": gpu, "capability": "12.0", "driver": "1", "torch": "2.11", "git": {"commit": "c"}},
            "settings": {"tune_mode": "auto"},
            "data": {"mlp": {"weights_sha256": "w", "input_tensor_sha256": ["i"]}},
            "groups": [{"case": "mlp", "dtype": "float32", "arm": "truestar_det", "claim": "bitwise",
                        "settings": st, "truestar_deterministic": True, "rows": [row], "error": None,
                        "configs": [{"step": "s", "shapes": [["b0", [7, 512]]], "config": {"BM": 16},
                                     "source": "default"}]},
                       {"case": "mlp", "dtype": "float32", "arm": "eager", "claim": "none", "settings": st,
                        "rows": [copy.deepcopy(row)], "error": None}]}


def test_compare_identical_documents_holds_the_claim():
    cmp = compare(_doc(), _doc())
    assert cmp.claim_ok and not cmp.env_diff and not cmp.settings_diff
    assert {r["status"] for r in cmp.rows} == {"equal"}


def test_compare_names_a_changed_setting_and_environment():
    b = _doc(gpu="GPU B", tf32=True)
    b["settings"]["tune_mode"] = "off"
    cmp = compare(_doc(), b)
    keys = {(d["where"], d["key"]) for d in cmp.settings_diff}
    assert ("global", "tune_mode") in keys
    assert ("truestar_det float32", "matmul_allow_tf32") in keys
    assert ("eager float32", "cudnn_allow_tf32") in keys
    assert [d["key"] for d in cmp.env_diff] == ["gpu"]
    codes = {i["code"] for i in cmp.issues}
    assert {"TS745", "TS747"} <= codes and "TS743" not in codes
    assert "tune_mode" in cmp.to_text() and "matmul_allow_tf32" in cmp.to_text()


def test_compare_flags_a_changed_claimed_hash_and_shows_the_config_difference():
    b = _doc()
    b["groups"][0]["rows"][0]["outputs"] = ["bb" * 32]
    b["groups"][0]["configs"][0]["config"] = {"BM": 32}
    cmp = compare(_doc(), b)
    assert not cmp.claim_ok
    bad = [r for r in cmp.rows if r["status"] == "different"]
    assert len(bad) == 1 and bad[0]["arm"] == "truestar_det" and bad[0]["configs_differ"]
    assert bad[0]["config_diffs"][0]["a"]["config"] == {"BM": 16}
    assert "TS743" in {i["code"] for i in cmp.issues}
    # an arm without a claim that differs is recorded, not a finding
    c = _doc()
    c["groups"][1]["rows"][0]["outputs"] = ["cc" * 32]
    cmp = compare(_doc(), c)
    assert cmp.claim_ok and "TS746" in {i["code"] for i in cmp.issues}


def test_compare_refuses_to_compare_different_inputs():
    b = _doc()
    b["data"]["mlp"]["input_tensor_sha256"] = ["other"]
    b["groups"][0]["rows"][0]["outputs"] = ["bb" * 32]
    cmp = compare(_doc(), b)
    assert {r["status"] for r in cmp.rows} == {"not_comparable"}
    assert "TS744" in {i["code"] for i in cmp.issues} and cmp.claim_ok


def test_compare_command_line_exit_codes(tmp_path):
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    a.write_text(json.dumps(_doc()))
    changed = _doc(tf32=True)
    changed["groups"][0]["rows"][0]["outputs"] = ["bb" * 32]
    b.write_text(json.dumps(changed))
    c.write_text(json.dumps({"something": "else"}))
    p = _cli("compare", str(a), str(a))
    assert p.returncode == 0, p.stdout + p.stderr
    p = _cli("compare", str(a), str(b), "--format", "agent")
    assert p.returncode == 1, p.stdout + p.stderr
    doc = json.loads(p.stdout)
    assert "TS743" in doc["codes"] and "TS745" in doc["codes"]
    assert any(d["key"] == "matmul_allow_tf32" for d in doc["comparison"]["settings_diff"])
    p = _cli("compare", str(a), str(c), "--format", "agent")
    assert p.returncode == 2 and json.loads(p.stdout)["codes"] == ["TS749"]


def test_legacy_files_are_read():
    for name in ("fp_desktop_5070ti.json", "fp_laptop_5060.json"):
        path = os.path.join(ROOT, "docs", "cross_chip", name)
        doc = load(path)
        assert {g["case"] for g in doc["groups"]} == {"legacy_mlp", "legacy_transformer", "legacy_resnet18"}
    cmp = compare_files(os.path.join(ROOT, "docs", "cross_chip", "fp_desktop_5070ti.json"),
                        os.path.join(ROOT, "docs", "cross_chip", "fp_laptop_5060.json"))
    assert cmp.claim_ok                     # the first run: compiled outputs equal on both GPUs
    eager = [r for r in cmp.rows if r["arm"] == "eager"]
    assert [r["status"] for r in eager if r["case"] == "legacy_transformer"] == ["different"]


def test_a_changed_data_file_is_refused(tmp_path):
    case = CASES["legacy_mlp"]
    _, xs, rec = fpdata.prepare(case, str(tmp_path))
    assert rec["source"] == "written"
    _, xs2, rec2 = fpdata.prepare(case, str(tmp_path))
    assert rec2["source"] == "loaded" and rec2["inputs_sha256"] == rec["inputs_sha256"]
    bad = [x.clone() for x in xs]
    bad[0][0, 0] += 1.0
    torch.save(bad, os.path.join(tmp_path, "legacy_mlp.inputs.pt"))
    with pytest.raises(fpdata.DataMismatch, match="does not match the manifest"):
        fpdata.prepare(case, str(tmp_path))


def test_weights_and_inputs_match_the_pinned_hashes():
    """Every machine must build the same bits. PyTorch's CPU generator did not (Windows desktop
    against the Kaggle Linux host), so the cases use the portable generator; on each machine this
    test checks its own bits against the hashes pinned on the desktop. Run on a second machine it is
    the cross machine check of the data."""
    from truestar.fingerprint.models import GOLDEN

    assert set(GOLDEN) == {n for n, c in CASES.items() if not c.legacy}
    for name, (w, i) in GOLDEN.items():
        case = CASES[name]
        assert fpdata.state_sha(fpdata.build(case).state_dict()) == w, name
        assert fpdata.inputs_sha(case.make_inputs(max(case.batches))) == i, name


def test_portable_draws_are_exact():
    from truestar.fingerprint.models import portable_normal, portable_uniform

    u = portable_uniform(3, 1, 100000)
    assert ((u * (1 << 24)) % 1 == 0).all() and 0 <= u.min() and u.max() < 1
    assert abs(u.mean() - 0.5) < 0.01
    x = portable_normal(3, 2, (50000,))
    assert abs(float(x.mean())) < 0.02 and abs(float(x.std()) - 1) < 0.02
    assert torch.equal(portable_normal(3, 2, (7,)), x[:7])     # a prefix is the smaller draw


def test_committed_results_say_what_cross_chip_md_says():
    """docs/CROSS_CHIP.md quotes these files; if they change, the page must change with them."""
    d = os.path.join(ROOT, "docs", "cross_chip")
    desk, t37, t36 = (load(os.path.join(d, f)) for f in
                      ("fp2_rtx5070ti.json", "fp2_t4_triton371.json", "fp2_t4_triton360.json"))
    assert desk["env"]["capability"] == "12.0" and t37["env"]["capability"] == "7.5"
    assert desk["env"]["triton"] == t37["env"]["triton"] == "3.7.1" and t36["env"]["triton"] == "3.6.0"
    cmp = compare(desk, t37)
    assert all(x["weights_equal"] and x["inputs_equal"] for x in cmp.data if not x["case"].startswith("legacy"))

    def status(c, case, dtype, arm="truestar_det"):
        return [r["status"] for r in c.rows if (r["case"], r["dtype"], r["arm"]) == (case, dtype, arm)]
    for case in ("mlp", "transformer", "llama_step"):
        assert status(cmp, case, "float32") == ["equal"] * 4, case
    assert "equal" not in status(cmp, "resnet18", "float32")
    assert status(compare(t36, t37), "transformer", "float32") == ["different"] * 4
    assert status(compare(desk, t36), "mlp", "float32") == ["equal"] * 4


# ----------------------------------------------------------------- on this GPU

@needs_gpu
def test_fingerprint_is_stable_run_to_run_on_this_gpu(tmp_path):
    data_dir = str(tmp_path / "data")
    first = run(data_dir=data_dir, **SMALL)
    assert not [i for i in first["issues"] if i["code"] in ("TS741", "TS742", "TS750")], first["issues"]
    assert all(first["data"][c]["golden_match"] for c in ("mlp", "llama_step"))
    for g in first["groups"]:
        assert all(r["run_to_run_equal"] for r in g["rows"]), g
    det = {g["case"]: g for g in first["groups"] if g["arm"] == "truestar_det"}
    assert det["mlp"]["batch_invariant_row0"] is True
    assert all(e is not None and e < 1e-5 for r in det["mlp"]["rows"] for e in r["max_rel_error_vs_float64"])
    env = first["env"]
    for key in ("gpu", "capability", "sm_count", "driver", "torch", "triton", "cuda_runtime", "cudnn"):
        assert key in env
    assert env["git"]["commit"]
    # a second fingerprint from the command line, in a fresh process, same data files
    out = tmp_path / "cli.json"
    p = _cli("--cases", "mlp,llama_step", "--dtypes", "float32", "--batches", "1,7", "--arms",
             "truestar_det,eager", "--data-dir", data_dir, "--json", str(out))
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    second = load(str(out))
    assert all(second["data"][c]["source"] == "loaded" for c in ("mlp", "llama_step"))
    cmp = compare(first, second)
    assert cmp.claim_ok
    assert {r["status"] for r in cmp.rows if r["arm"] == "truestar_det"} == {"equal"}


@needs_gpu
def test_deterministic_mode_tunes_only_order_keeping_parameters(monkeypatch, tmp_path):
    """Tuning by timing picks per GPU and per run; for attention the pick changes the bits."""
    from truestar.kernel import Graph, tune
    from truestar.kernel import triton_backend as tb

    monkeypatch.setenv("TRUESTAR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TRUESTAR_TUNE_BUDGET", "60")
    K.clear_memory_cache()
    g = Graph()
    N = K.sym("N")
    q, k, v = (g.input(n, (1, 2, N, 64), "float32") for n in "qkv")
    (_, attn), = g.fuse({"o": g.softmax((q @ k.T) * 0.125) @ v})
    gen = torch.Generator(device="cuda").manual_seed(0)
    args = {n: torch.randn(1, 2, 512, 64, device="cuda", generator=gen) for n in "qkv"}

    det = K.compile(attn, "cuda", tune=True, deterministic=True)
    out_det = det(**args)
    plan = det.art.plan
    env = {s.name: 512 for s in K.ir.canonicalize(attn)[0].symbols()}
    (_, cfg, src), = det.launch_configs()
    assert src in ("pinned", "order tuned")
    assert tb.order_key(plan, env, cfg) == tb.order_key(plan, env, tb.default_config(plan, env))
    assert torch.equal(out_det, K.compile(attn, "cuda", _config=tb.default_config(plan, env))(**args))
    searches = tune.stats.searches
    fast = K.compile(attn, "cuda", tune=True)
    fast(**args)
    assert fast.launch_configs()[0][2] == "tuned" and tune.stats.searches == searches + 1

    # the property the pin protects: another attention config gives other bits for the same inputs
    outs = [K.compile(attn, "cuda", _config=c)(**args) for c in tb.candidates(plan, env)[1:]]
    assert any(not torch.equal(o, out_det) for o in outs)

    # a reduction is pinned too; a contraction (config free without split K) is still tuned
    g = Graph()
    x = g.input("x", (K.sym("M"), 4096), "float32")
    (_, red), = g.fuse({"y": g.softmax(x)})
    r = K.compile(red, "cuda", tune=True, deterministic=True)
    xr = torch.randn(64, 4096, device="cuda")
    yr = r(x=xr)
    (_, rcfg, rsrc), = r.launch_configs()
    rplan, renv = r.art.plan, next(iter(r._fast.values())).env
    assert rsrc in ("pinned", "order tuned")
    assert tb.order_key(rplan, renv, rcfg) == tb.order_key(rplan, renv, tb.default_config(rplan, renv))
    assert torch.equal(yr, K.compile(red, "cuda", _config=tb.default_config(rplan, renv))(x=xr))
    g = Graph()
    x = g.input("x", (K.sym("M"), 512), "float32")
    w = g.input("w", (256, 512), "float32")
    (_, lin), = g.fuse({"y": x @ g.transpose(w, 0, 1)})
    c = K.compile(lin, "cuda", tune=True, deterministic=True)
    c(x=torch.randn(64, 512, device="cuda"), w=torch.randn(256, 512, device="cuda"))
    (_, cfg, src), = c.launch_configs()
    assert src == "tuned" and cfg["SPLIT_K"] == 1
    K.clear_memory_cache()


@needs_gpu
@pytest.mark.skipif(not os.environ.get("TRUESTAR_FP_OTHER"),
                    reason="cross machine only: set TRUESTAR_FP_OTHER to a fingerprint JSON from another GPU")
def test_deterministic_rows_match_another_gpu(tmp_path):
    other = load(os.environ["TRUESTAR_FP_OTHER"])
    cases = sorted({g["case"] for g in other["groups"]})
    dtypes = sorted({g["dtype"] for g in other["groups"]})
    here = run(cases=cases, dtypes=dtypes, arms=["truestar_det"], data_dir=str(tmp_path / "data"))
    cmp = compare(other, here)
    assert cmp.claim_ok, cmp.to_text()
