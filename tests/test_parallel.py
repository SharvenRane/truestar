"""`truestar parallel` (docs/PARALLEL.md, ledger row P34): one training step under a layout against the
single device step, on simulated ranks (threads, and gloo processes).

Each injected fault is a bug class from the issues cited in docs/PARALLEL.md; each must be named at
the right module or collective in one step, and correct layouts must be clean.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

import parallel_models as M
from truestar import codes
from truestar.parallel import CrossEntropy, FAULTS, LayoutError, auto_plan, parallel_check, parse_layout

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")

CLEAN = [
    ("tp=2", "tiny", False, "sgd", 0.0),
    ("tp=2,dp=2", "tiny", False, "sgd", 0.0),
    ("tp=2,dp=2", "tiny", False, "adamw", 0.5),
    ("tp=4", "tiny", False, "sgd", 1.0),
    ("tp=2,dp=1,ga=2,sp=1", "tiny", True, "sgd", 0.0),
    ("tp=2,dp=2,ga=2,sp=1", "tiny", True, "adamw", 0.5),
    ("dp=4,ga=2", "tiny", False, "sgd", 0.0),
    ("dp=2,ga=4", "tiny", False, "adamw", 0.0),
]


def _run(layout, sp=False, faults=(), optimizer="sgd", clip=0.0, **kw):
    return parallel_check(M.tiny(), M.tiny_inputs(), layout, plan=M.tiny_plan(sp=sp) if "tp" in layout else None,
                          optimizer=optimizer, clip=clip, faults=faults, **kw)


@pytest.mark.parametrize("layout,model,sp,opt,clip", CLEAN)
def test_correct_layouts_are_clean(layout, model, sp, opt, clip):
    r = _run(layout, sp=sp, optimizer=opt, clip=clip)
    assert r.status == "ok", r.to_text()
    assert r.codes == ["TS690"]
    assert all(v < 1e-10 for v in r.worst.values()), r.worst
    assert r.compared["parameter gradients"] == 31 and r.compared["updated weights"] == 31


def test_auto_plan_pairs_only_unambiguous_mlp_projections():
    plan = auto_plan(M.tiny().double(), (M.tiny_inputs().double(),), 2)
    assert plan == {}, plan  # square q, k, v, o are never guessed; gate_up (2 ff) to down (ff) is not a width pair
    mlp = M.MLP()
    plan = auto_plan(mlp, (torch.randn(4, 10),), 2)
    assert {k: v.text() for k, v in plan.items()} == {"up": "column", "down": "row"}
    r = parallel_check(mlp, torch.randn(8, 10), "tp=2,dp=2,ga=2")
    assert r.status == "ok" and r.plan_source.startswith("found automatically"), r.to_text()


def test_uneven_token_counts_are_right_under_dp_and_ga():
    """Global count normalization: padded rows make micro batches and replicas hold different token counts."""
    x = torch.randn(8, 6, 8, generator=torch.Generator().manual_seed(3))
    r = parallel_check(M.Classifier(), x, "tp=2,dp=2,ga=2", targets=M.padded_targets(), loss=CrossEntropy())
    assert r.status == "ok", r.to_text()


def test_dp_sum_of_means_is_named_as_per_replica_normalization_with_the_gradient_factor():
    r = _run("tp=2,dp=2,ga=1", faults=["dp_sum_of_means"])
    f = r.first()
    assert f.code == "TS682" and "sum of per replica means" in f.message, r.to_text()
    assert "(2 times the single device gradients" in f.message


def test_dp_average_twice_is_a_one_over_dp_factor():
    r = _run("dp=4", faults=["dp_average_twice"])
    f = r.first()
    assert f.code == "TS683" and "0.25 times" in f.message and "1/dp" in f.message, r.to_text()


def test_accumulation_divided_twice_is_a_one_over_ga_factor():
    r = _run("tp=2,ga=4", faults=["ga_divided_twice"])
    f = r.first()
    assert f.code == "TS683" and "0.25 times" in f.message and "1/ga" in f.message, r.to_text()
    assert f.collective == "gradient accumulation"


def test_equal_dp_and_ga_factors_are_reported_as_ambiguous():
    r = _run("dp=2,ga=2", faults=["ga_divided_twice"])
    f = r.first()
    assert f.code == "TS683" and "1/dp or 1/ga" in f.message, r.to_text()


def test_mean_of_micro_batch_means_is_named_as_loss_normalization():
    x = torch.randn(8, 6, 8, generator=torch.Generator().manual_seed(3))
    r = parallel_check(M.Classifier(), x, "dp=1,ga=4", targets=M.padded_targets(), loss=CrossEntropy(),
                       faults=["ga_mean_of_means"])
    f = r.first()
    assert f.code == "TS682" and "mean of per micro batch means" in f.message, r.to_text()


def test_mean_of_means_is_not_flagged_when_counts_are_equal():
    """Without padding the wrong normalization equals the right one: no finding is correct."""
    x = torch.randn(8, 6, 8, generator=torch.Generator().manual_seed(3))
    y = torch.randint(0, 7, (8, 6), generator=torch.Generator().manual_seed(4))
    r = parallel_check(M.Classifier(), x, "dp=2,ga=2", targets=y, loss=CrossEntropy(), faults=["ga_mean_of_means"])
    assert r.status == "ok", r.to_text()


def test_missing_tp_sync_of_sequence_parallel_norms_is_named_per_weight():
    r = _run("tp=2,dp=1,ga=2,sp=1", sp=True, faults=["missing_tp_sync"])
    f = r.first()
    assert f.code == "TS684" and f.module == "blocks.0.norm1", r.to_text()
    assert "sum over tp equals the single device gradient" in f.message and "sequence shard" in f.message
    flagged = {x.module for x in r.findings if x.code == "TS684"}
    assert {"blocks.0.norm1", "blocks.0.norm2", "blocks.1.norm1", "blocks.1.norm2", "norm"} <= flagged


def test_missing_backward_all_reduce_of_a_column_input_is_blamed_at_the_column_layer():
    r = _run("tp=2", faults=["column_input_grad_not_reduced"])
    f = r.first()
    assert f.code == "TS685" and f.module in ("blocks.1.mlp.gate_up",), r.to_text()
    assert f.collective == "blocks.1.mlp.gate_up input gradient all reduce (backward)"


def test_row_bias_added_on_every_rank_is_named_with_the_bias_copies():
    r = _run("tp=2,dp=2", faults=["row_bias_every_rank"])
    f = r.first()
    assert f.code == "TS681" and f.module == "blocks.0.attn.o", r.to_text()
    assert "1 extra copies of the bias" in f.message


def test_fused_weight_split_as_one_block_names_the_split_it_got():
    r = _run("tp=2", faults=["fused_split_as_block"])
    f = r.first()
    assert f.code == "TS681" and f.module == "blocks.0.mlp.gate_up", r.to_text()
    assert "a split along dim 0 as one block" in f.message and "column:2" in f.message


def test_row_weight_split_on_the_wrong_dim_fails_loudly_at_the_module():
    r = _run("tp=2", faults=["row_split_dim0"])
    f = r.first()
    assert f.code == "TS689" and f.module == "blocks.0.attn.o", r.to_text()


def test_clipping_by_the_local_norm_is_named_with_both_norms():
    r = _run("tp=2,dp=2", faults=["clip_local_norm"], clip=0.5)
    f = r.first()
    assert f.code == "TS687" and "local shard's norm" in f.message, r.to_text()
    assert f.collective == "gradient norm all reduce over tp"


def test_every_fault_is_caught_and_only_by_its_own_finding():
    """The table in docs/PARALLEL.md: each fault, one step, the first finding's code."""
    want = {"dp_sum_of_means": "TS682", "dp_average_twice": "TS683", "ga_divided_twice": "TS683",
            "missing_tp_sync": "TS684", "column_input_grad_not_reduced": "TS685", "row_bias_every_rank": "TS681",
            "fused_split_as_block": "TS681", "row_split_dim0": "TS689", "clip_local_norm": "TS687"}
    for fault, code in want.items():
        r = _run("tp=2,dp=2,ga=4,sp=1", sp=True, faults=[fault], clip=0.5)
        assert r.status == "divergent" and r.first().code == code, (fault, r.to_text())
    # ga_mean_of_means needs uneven counts (its own test); shard_stale_params needs shard=1 or 2
    # (tests/test_multigpu.py::test_missing_parameter_gather_after_a_sharded_update_is_named)
    assert set(want) | {"ga_mean_of_means", "shard_stale_params"} == set(FAULTS)


def test_a_model_that_reads_the_sequence_length_from_its_shard_fails_by_name():
    class Bad(M.Attention):
        def forward(self, x):
            B, T, _ = x.shape  # under sequence parallel this is the local sequence piece
            q, k, v = self.q(x), self.k(x), self.v(x)
            h = q.shape[-1] // self.head_dim
            q = q.view(B, T, h, self.head_dim)
            return self.o(q.reshape(B, T, -1))

    m = M.tiny()
    m.blocks[0].attn = Bad(16, 4)
    r = parallel_check(m, M.tiny_inputs(), "tp=2,sp=1", plan=M.tiny_plan(sp=True))
    f = r.first()
    assert f.code == "TS689" and f.module == "blocks.0.attn" and "test_parallel.py" in f.message, r.to_text()


def test_layout_errors_are_refused_by_name():
    for bad, part in [("tp=2,pp=2", "pipeline parallel is not built"), ("tp=0", "at least 1"),
                      ("sp=1", "needs tp"), ("tq=2", "unknown layout key"), ("tp=2,tp=2", "given twice")]:
        with pytest.raises(LayoutError, match=part):
            parse_layout(bad)
    with pytest.raises(LayoutError, match="must divide it"):
        parallel_check(M.tiny(), M.tiny_inputs(B=6), "dp=4", plan=None)
    with pytest.raises(LayoutError, match="splits no Linear"):
        parallel_check(M.tiny(), M.tiny_inputs(), "tp=2")
    with pytest.raises(LayoutError, match="not divisible"):
        parallel_check(M.tiny(), M.tiny_inputs(), "tp=3", plan=M.tiny_plan())
    with pytest.raises(LayoutError, match="scatter"):
        parallel_check(M.tiny(), M.tiny_inputs(), "tp=2,sp=1", plan=M.tiny_plan())
    with pytest.raises(LayoutError, match="unknown fault"):
        parallel_check(M.tiny(), M.tiny_inputs(), "dp=2", faults=["nope"])


def test_agent_document_carries_the_blame():
    r = _run("tp=2,dp=2", faults=["row_bias_every_rank"])
    doc = r.to_agent("model.py")
    json.dumps(doc, allow_nan=False)
    assert doc["command"] == "parallel" and doc["status"] == "divergent" and doc["exit_code"] == 1
    assert doc["blamed_module"] == "blocks.0.attn.o" and doc["simulated"] is True
    assert "TS681" in doc["codes"] and "TS690" in doc["codes"]
    assert all(c in codes.CODES for c in doc["codes"])
    assert doc["next_actions"][0]["kind"] == "fix_layout"


needs_gloo = pytest.mark.skipif(not (torch.distributed.is_available() and torch.distributed.is_gloo_available()),
                                reason="needs torch.distributed with gloo")


def test_file_init_method_survives_the_url_round_trip_torch_does(tmp_path):
    """torch adds rank and world size with urlunparse before it opens the store; the path it opens
    must be the path given, with no part of it taken as a host (four slashes on POSIX lose /tmp)."""
    from urllib.parse import urlparse, urlunparse
    from urllib.request import url2pathname

    from truestar.parallel.comm import file_init_method

    path = os.path.join(str(tmp_path), "init")
    parsed = urlparse(urlunparse(urlparse(file_init_method(path))._replace(query="rank=0&world_size=2")))
    if sys.platform == "win32":  # torch's Windows handler joins netloc and path
        opened = os.path.normpath(url2pathname(parsed.netloc + parsed.path))
    else:
        assert parsed.netloc == "", parsed
        opened = parsed.path
    assert os.path.normcase(opened) == os.path.normcase(os.path.abspath(path))


@needs_gloo
def test_gloo_processes_give_the_same_verdicts_as_threads(tmp_path):
    """Four real gloo processes: the clean layout is clean and an injected fault is named the same way."""
    f = os.path.join(HERE, "parallel_models_file.py")
    from truestar.cli import _load

    model, x, _ = _load(f)
    import parallel_models_file as PF

    ok = parallel_check(model, x, "tp=2,dp=2,ga=2,sp=1", plan=PF.get_parallel_plan(parse_layout("sp=1,tp=2")),
                        ranks="gloo", model_file=f)
    assert ok.status == "ok", ok.to_text()
    bad = parallel_check(model, x, "tp=2,dp=2", plan=PF.get_parallel_plan(parse_layout("tp=2")), ranks="gloo", model_file=f,
                         faults=["column_input_grad_not_reduced"])
    assert bad.first().code == "TS685" and bad.first().module == "blocks.1.mlp.gate_up", bad.to_text()


def _cli(*argv):
    env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + HERE, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-m", "truestar", *argv], capture_output=True, text=True, env=env,
                          encoding="utf-8", errors="replace", timeout=300)


def test_cli_clean_and_faulty():
    f = os.path.join(HERE, "parallel_models_file.py")
    p = _cli("parallel", f, "--layout", "tp=2,dp=2,ga=2,sp=1")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "overall: OK" in p.stdout
    p = _cli("parallel", f, "--layout", "tp=2,dp=2,ga=2,sp=1", "--inject", "missing_tp_sync", "--format", "agent")
    assert p.returncode == 1 and p.stdout.startswith("{"), p.stdout + p.stderr
    doc = json.loads(p.stdout)
    assert doc["codes"][0] == "TS684" and doc["blamed_module"] == "blocks.0.norm1"
    p = _cli("parallel", f, "--layout", "tp=2,pp=2", "--format", "agent")
    assert p.returncode == 2, p.stdout + p.stderr
    assert json.loads(p.stdout)["issues"][0]["code"] == "TS680"
