"""The Lean proofs (proofs/lean) are instantiated at tcexact's constants.

* proofs/lean/TruestarProofs/Constants.lean must be the file scripts/gen_lean_constants.py writes
  from the current src/truestar/kernel/tcexact.py, and its values must equal the imported module's;
* with `lake` on PATH: `lake build` passes, `lake exe tccheck` agrees with the Python simulator on
  every probe vector, and a weakened TC_FRAC_BITS_MIN or accumulator start makes the build fail.
  Skipped where Lean is not installed (Lean runs on the server, see docs/PROOFS.md).
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from truestar.kernel import tcexact as X

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "proofs" / "lean"
CONSTS = LEAN / "TruestarProofs" / "Constants.lean"


def _gen():
    spec = importlib.util.spec_from_file_location("gen_lean_constants",
                                                  ROOT / "scripts" / "gen_lean_constants.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generated_file_is_current():
    g = _gen()
    assert CONSTS.read_text(encoding="utf-8") == g.render(g.read_constants()), \
        "Constants.lean is stale: run scripts/gen_lean_constants.py"


def test_lean_values_equal_python_constants():
    text = CONSTS.read_text(encoding="utf-8")
    vals = {m.group(1): int(m.group(2)) for m in re.finditer(r"^def (\w+) : \w+ := (-?\d+)$", text, re.M)}
    assert vals == {
        "TC_FRAC_BITS_MIN": X.TC_FRAC_BITS_MIN,
        "ACC_START": int(X._RN),
        "TOP_MAX": X.TOP_MAX,
        "LOW_MAX": X.LOW_MAX,
        "DIGIT_BITS": X.DIGIT_BITS,
        "DIGITS": X.DIGITS,
        "SUM_BITS": X.SUM_BITS,
        "PORTABLE_BK": X.PORTABLE_BK,
        "MAX_KB": X.MAX_KB,
    }
    assert X._RN == 1.5 * 2 ** 23


def test_fixture_present():
    import json
    data = json.loads((LEAN / "fixtures" / "tc_probes.json").read_text(encoding="utf-8"))
    fams = {p["family"] for p in data["probes"]}
    assert {"xchip", "random", "digits", "variants"} <= fams
    assert len(data["probes"]) >= 200 and len(data["fp8"]) == 256


needs_lake = pytest.mark.skipif(shutil.which("lake") is None, reason="Lean (lake) not installed")


def _lake(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["lake", *args], cwd=cwd, capture_output=True, text=True, timeout=1800)


@needs_lake
def test_lake_build_and_cross_check():
    r = _lake(LEAN, "build")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "sorry" not in (r.stdout + r.stderr)
    r = _lake(LEAN, "exe", "tccheck", "fixtures/tc_probes.json")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "mismatches: 0" in r.stdout


@needs_lake
@pytest.mark.parametrize("name,value", [("TC_FRAC_BITS_MIN", "23"), ("TC_FRAC_BITS_MIN", "22"),
                                        ("ACC_START", "8388608"), ("ACC_START", "16000000"),
                                        ("MAX_KB", "256")])
def test_weakened_constant_fails_build(tmp_path, name, value):
    work = tmp_path / "lean"
    shutil.copytree(LEAN, work, ignore=shutil.ignore_patterns("fixtures"))
    c = work / "TruestarProofs" / "Constants.lean"
    text, n = re.subn(rf"^(def {name} : \w+ := )-?\d+$", rf"\g<1>{value}", c.read_text(), flags=re.M)
    assert n == 1
    c.write_text(text)
    r = _lake(work, "build", "TruestarProofs.Instances")
    assert r.returncode != 0, f"build passed with {name} = {value}"
