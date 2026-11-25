"""Truestar's derived bounds against cached FPTaylor sound bounds (scripts/bound_crosscheck.py).

FPTaylor is not needed here: its results for the corpus are cached in
tests/data/bound_crosscheck_fptaylor.json, keyed by the exact FPTaylor input text, so a change to
the corpus or to the op error model shows as a stale entry. Regenerate where FPTaylor exists:

    PYTHONPATH=src python scripts/bound_crosscheck.py --fptaylor <path>/fptaylor \\
        --write-fixture tests/data/bound_crosscheck_fptaylor.json
"""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("bound_crosscheck",
                                               os.path.join(HERE, "..", "scripts", "bound_crosscheck.py"))
bc = importlib.util.module_from_spec(_spec)
sys.modules["bound_crosscheck"] = bc  # dataclasses look their module up there
_spec.loader.exec_module(bc)

with open(bc.FIXTURE) as fh:
    FIXTURE = json.load(fh)["results"]


@pytest.fixture(scope="module")
def rows():
    return bc.run()[0]


def test_fixture_matches_the_corpus():
    """Every formula FPTaylor can express has a cached result for its current input text."""
    stale = []
    for f in bc.corpus():
        for dt in f.dtypes:
            text = bc.fptaylor_input(f, dt)
            if text is None:
                continue
            got = FIXTURE.get(f"{f.name}/{dt}")
            if not got or got.get("sha") != bc._sha(text):
                stale.append(f"{f.name}/{dt}")
    assert not stale, f"regenerate the fixture (FPTaylor input changed): {stale}"


def test_no_truestar_bound_under_a_sound_bound(rows):
    """Truestar's largest bound over the box is never under FPTaylor's sound bound; the 64 run
    error model and truestar.opbounds are never under the first order worst case at any grid
    point."""
    bad = bc.bugs(rows)
    assert not bad, bc.table(bad)


def test_every_expressible_row_was_compared(rows):
    for r in rows:
        if r.formula in ("erf", "gelu_erf"):
            assert r.fptaylor_abs is None  # FPTaylor has no erf: pointwise check only
            continue
        if r.fptaylor_abs is None:
            # FPTaylor itself failed on every piece of the box (recorded with its message):
            # only the half precision AdamW parameter update ("num_of_float: inf")
            assert (r.formula, r.output) == ("adamw_step", 0) and r.dtype != "float32", (r.formula, r.dtype)
            assert "error" in FIXTURE[f"{r.formula}/{r.dtype}"]
            continue
        assert r.fptaylor_abs > 0, (r.formula, r.dtype, r.output)


def test_cancelling_kernel_is_rejected(rows):
    """1 - exp(x) checked against the bound of -expm1(x) must fall outside it: the formula
    cancels near 0: its relative error (FPTaylor) is far above Truestar's relative bound for
    expm1, and pointwise its first order worst case is above that bound."""
    sel = [r for r in rows if r.formula == "one_minus_exp_vs_expm1"]
    assert len(sel) == 3
    for r in sel:
        assert "REL_BELOW" in r.flags and "MODEL_BELOW" in r.flags, r.flags
        assert r.fptaylor_rel > 10 * r.truestar_rel


def test_sampled_bound_dips_are_rare(rows):
    """The default 8 run sigma estimate is random per element: at a few grid points the bound is
    under the first order worst case of its own model (none with 64 runs, checked above). Measured
    2026-11-25: at most 4.9e-4 of the points of any row, lowest ratio 0.81."""
    worst = max(r.point_below_frac for r in rows if not r.expected_below)
    assert worst <= 1e-3, worst
    assert min(r.point_min for r in rows if not r.expected_below) > 0.5


def test_fptaylor_input_rounds_every_op_in_its_format():
    x = bc.V("x")
    with bc.acc():
        y = bc.cast(bc.exp(x) + bc.C(1.0))
    f = bc.Formula("t", [("x", -1.0, 1.0)], [y], lambda t: t)
    text = bc.fptaylor_input(f, "bfloat16")
    assert "rnd[32,ne,4,-24,-150](exp(x))" in text  # float32 accumulation, 4 u exp
    assert "rnd[16,ne,1,-8,-134](t" in text  # the cast to bfloat16 keeps its rounding
    assert "real x in [-1.0, 1.0]" in text


def test_parse_fptaylor_output():
    out = ("Problem: out0\n\nOptimization lower bounds for error models:\n"
           "The absolute error (exact) model: 7.5e-3 (0x1p-7) (suboptimality = 4.0%)\n\n"
           "Absolute error (exact): 7.812501e-3 (0x1p-7)\n"
           "Relative error (exact): 3.981338e-3 (0x1p-8)\n\nProblem: out1\n\n"
           "Absolute error (exact): 9.536744e-7 (0x1p-20)\n")
    p = bc.parse_fptaylor(out)
    assert p["out0"] == {"abs": 7.812501e-3, "rel": 3.981338e-3, "abs_low": 7.5e-3}
    assert p["out1"] == {"abs": 9.536744e-7, "rel": None, "abs_low": None}
