"""Real multi process hangs on CPU with gloo: the diagnosis must name the rank, sequence and line."""

import json
import os
import time

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

import diag_workers
from truestar.diag.collectives import HANG_EXIT_CODE, diagnose

pytestmark = pytest.mark.skipif(not (dist.is_available() and dist.is_gloo_available()),
                                reason="needs torch.distributed with gloo")

WORKERS = os.path.abspath(diag_workers.__file__)
TIMEOUT = 2.0      # watchdog timeout per collective, seconds
DEADLINE = 60.0    # hard limit per scenario before every worker is killed


def mark(name: str) -> int:
    with open(WORKERS, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if f"# MARK:{name}" in line:
                return i
    raise KeyError(name)


def run(tmp_path, scenario: str, world: int):
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=diag_workers.worker, args=(r, world, str(tmp_path), scenario, TIMEOUT))
             for r in range(world)]
    for p in procs:
        p.start()
    t0 = time.time()
    try:
        while any(p.is_alive() for p in procs) and time.time() - t0 < DEADLINE:
            time.sleep(0.1)
    finally:
        for p in procs:  # never leave a worker behind, whatever happened
            if p.is_alive():
                p.kill()
            p.join(timeout=10)
    assert not any(p.is_alive() for p in procs)
    elapsed = time.time() - t0
    assert elapsed < DEADLINE, f"workers had to be killed after {elapsed:.0f} s: the watchdog never fired"
    rec = tmp_path / "rec"
    reports = {}
    for r in range(world):
        p = rec / f"diagnosis_rank{r}.json"
        if p.exists():
            reports[r] = json.loads(p.read_text(encoding="utf-8"))
    texts = {r: (rec / f"diagnosis_rank{r}.txt").read_text(encoding="utf-8") for r in reports}
    return [p.exitcode for p in procs], reports, texts, diagnose(str(rec)), elapsed


def _same_file(path: str) -> bool:
    return os.path.normcase(os.path.abspath(path)) == os.path.normcase(WORKERS)


def _waited_and_failed(codes, texts, r: int) -> bool:
    """A waiting rank ends by the watchdog, or by the collective's own error when the backend sees
    the peer go away first (gloo on Linux reports a closed peer at once; on Windows it blocks until
    the watchdog). Either way it must fail and must have written the diagnosis."""
    if codes[r] == HANG_EXIT_CODE:
        return True
    return codes[r] not in (0, None) and r in texts and ") raised " in texts[r]


def test_mismatched_collective_order(tmp_path):
    codes, reports, texts, d, elapsed = run(tmp_path, "order", world=4)
    assert reports, f"no rank wrote a diagnosis; exit codes {codes}"
    for r, rep in reports.items():  # every rank that noticed reached the same conclusion
        assert rep["kind"] == "mismatch", texts[r]
        assert rep["rank"] == 2, texts[r]
        assert rep["seq"] == 2, texts[r]  # seq 0 is the start barrier, seq 1 the warm up all_reduce
        assert _same_file(rep["file"]) and rep["line"] == mark("order_bad"), texts[r]
    assert d.kind == "mismatch" and d.rank == 2 and d.seq == 2 and d.line == mark("order_bad")
    text = next(iter(texts.values()))
    assert f"diag_workers.py:{mark('order_bad')}" in text and f"diag_workers.py:{mark('order_good')}" in text
    assert "rank 2 called broadcast" in text
    assert all(c in (HANG_EXIT_CODE, 0) for c in codes), codes


def test_rank_skips_collective_in_if(tmp_path):
    codes, reports, texts, d, elapsed = run(tmp_path, "skip", world=3)
    assert reports, f"no rank wrote a diagnosis; exit codes {codes}"
    for r, rep in reports.items():
        assert rep["kind"] == "skip", texts[r]
        assert rep["rank"] == 1, texts[r]
        assert rep["seq"] == 2, texts[r]
        assert _same_file(rep["file"]) and rep["line"] == mark("skip_target"), texts[r]
        assert rep["behind"] == [1], texts[r]
    assert d.kind == "skip" and d.rank == 1 and d.seq == 2 and d.line == mark("skip_target")
    text = next(iter(texts.values()))
    assert "rank 1 skipped collective seq 2" in text
    # rank 1 never hung (its collectives silently paired with the wrong ones), the others did
    assert codes[1] == 0 and _waited_and_failed(codes, texts, 0) and _waited_and_failed(codes, texts, 2), codes


def test_rank_crashes(tmp_path):
    codes, reports, texts, d, elapsed = run(tmp_path, "crash", world=3)
    assert reports, f"no rank wrote a diagnosis; exit codes {codes}"
    assert 2 not in reports  # the crashed rank does not diagnose; the waiting ones do
    for r, rep in reports.items():
        assert rep["kind"] == "crash", texts[r]
        assert rep["rank"] == 2, texts[r]
        assert rep["seq"] == 3, texts[r]  # the all_reduce rank 2 never reached
        assert _same_file(rep["file"]) and rep["line"] == mark("crash_raise"), texts[r]
    assert d.kind == "crash" and d.rank == 2 and d.seq == 3 and d.line == mark("crash_raise")
    text = next(iter(texts.values()))
    assert "rank 2 crashed with ValueError: bad batch" in text
    assert f"diag_workers.py:{mark('crash_wait')}" in text  # where the others wait
    assert codes[2] not in (0, HANG_EXIT_CODE) and _waited_and_failed(codes, texts, 0), codes


def test_clean_run_reports_nothing(tmp_path):
    codes, reports, texts, d, elapsed = run(tmp_path, "clean", world=2)
    assert codes == [0, 0]
    assert reports == {}
    assert d.kind == "ok", d.to_text()
    assert elapsed < 30
