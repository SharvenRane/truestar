"""Persistent compile cache (truestar.compile_cache): a verified first call is recorded; a later
process on the same inputs reuses the result only for the same kernel binaries and output bits;
bad entries are ignored and rebuilt; the size cap evicts least recently used entries; opt out."""

import json
import os
import subprocess
import sys
import warnings

import pytest
import torch
import torch.nn as nn

from truestar import compile_cache as CC
from truestar import ir
from truestar import kernel as K
from truestar.compiler import compile_program

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(16, 32), nn.Linear(32, 8)

    def forward(self, x):
        return self.b(torch.tanh(self.a(x)))


def _program(seed=0):
    torch.manual_seed(seed)
    m = _Net().eval()
    x = torch.randn(4, 16)
    return ir.import_module(m, (x,), name="ccnet"), x


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUESTAR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("TRUESTAR_COMPILE_CACHE", raising=False)
    monkeypatch.delenv("TRUESTAR_VERIFY", raising=False)
    K.clear_memory_cache()
    CC.stats.reset()
    yield tmp_path / "cache"
    K.clear_memory_cache()


def _new_process_like(prog, x, device="cpu"):
    """A compile and first call with nothing loaded in this process (the disk cache stays)."""
    K.clear_memory_cache()
    cp = compile_program(prog, device, "float32")
    out = cp([x.to(device)])[0]
    return cp, out


def _bits(t):
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def test_record_then_hit_bitwise_equal_and_no_checks(cache):
    prog, x = _program()
    cp1, out1 = _new_process_like(prog, x)
    assert cp1.compile_cache["verify"] == "stored"
    assert cp1.verify_seconds + cp1.kernel_verify_seconds > 0
    assert len(list((cache / "programs").glob("*.json"))) == 1
    cp2, out2 = _new_process_like(prog, x)
    assert cp2.compile_cache["verify"] == "hit" and cp2.compile_cache["code"] == "TS1350"
    assert cp2.verify_seconds == 0 and cp2.kernel_verify_seconds == 0
    assert cp2.verified and cp2.max_error == cp1.max_error
    assert _bits(out1) == _bits(out2)
    # later calls of the hit program are not checked again
    out3 = cp2([x])[0]
    assert cp2.kernel_verify_seconds == 0 and _bits(out3) == _bits(out1)


def test_other_inputs_or_weights_are_verified(cache):
    prog, x = _program()
    _new_process_like(prog, x)
    cp, _ = _new_process_like(prog, x + 1.0)          # other input values: other key
    assert cp.compile_cache["verify"] == "stored" and cp.kernel_verify_seconds > 0
    prog2, _ = _program(seed=1)                        # same structure, other weights
    cp, _ = _new_process_like(prog2, x)
    assert cp.compile_cache["verify"] == "stored" and cp.kernel_verify_seconds > 0
    # an in place edit of a weight after hashing changes the key too
    next(iter(prog.param_data.values())).mul_(2.0)
    cp, _ = _new_process_like(prog, x)
    assert cp.compile_cache["verify"] == "stored"


def test_corrupt_record_is_ignored_and_rebuilt(cache):
    prog, x = _program()
    _, out1 = _new_process_like(prog, x)
    path = next((cache / "programs").glob("*.json"))
    path.write_bytes(path.read_bytes()[:40])           # truncated mid write by a crash
    with pytest.warns(CC.CompileCacheWarning, match="TS1351"):
        cp, out2 = _new_process_like(prog, x)
    assert cp.compile_cache["verify"] == "stored" and cp.kernel_verify_seconds > 0
    assert CC.stats.rejected == 1 and _bits(out1) == _bits(out2)
    json.loads(path.read_text())                       # rewritten whole
    cp, _ = _new_process_like(prog, x)
    assert cp.compile_cache["verify"] == "hit"
    # wrong format version and a body edited without its digest
    rec = json.loads(path.read_text())
    rec["body"]["max_error"] = 0.0
    path.write_text(json.dumps(rec))
    with pytest.warns(CC.CompileCacheWarning, match="digest"):
        cp, _ = _new_process_like(prog, x)
    assert cp.compile_cache["verify"] == "stored"
    rec = json.loads(path.read_text())
    rec["format"] = 999
    path.write_text(json.dumps(rec))
    with pytest.warns(CC.CompileCacheWarning, match="format"):
        cp, _ = _new_process_like(prog, x)
    assert cp.compile_cache["verify"] == "stored"


def test_record_that_does_not_reproduce_is_verified_in_full(cache):
    """A consistent record whose outputs (or kernel list) differ from this call's: verify in full."""
    prog, x = _program()
    _, out1 = _new_process_like(prog, x)
    path = next((cache / "programs").glob("*.json"))
    rec = json.loads(path.read_text())
    for field, bad in (("outputs", "0" * 64), ("kernels", ["not a kernel"])):
        body = dict(rec["body"], **{field: bad})
        path.write_text(json.dumps({"format": CC.FORMAT, "body": body, "digest": CC._digest(body)}))
        cp, out2 = _new_process_like(prog, x)
        assert cp.compile_cache["verify"] == "stored" and cp.compile_cache["code"] == "TS1354"
        assert cp.kernel_verify_seconds > 0 and cp.verified
        assert _bits(out1) == _bits(out2)
    assert CC.stats.mismatched == 2


def test_wrong_kernel_on_replay_is_caught(cache, monkeypatch):
    """A kernel that returns other bits than when the record was made is checked, and refused."""
    prog, x = _program()
    _new_process_like(prog, x)
    orig = K.CompiledKernel._run_cpu

    def bad(self, got, L, *a):
        outs = orig(self, got, L, *a)
        return {n: v * 1.01 for n, v in outs.items()}
    monkeypatch.setattr(K.CompiledKernel, "_run_cpu", bad)
    from truestar.compiler import ProgramMismatchError
    with pytest.raises(ProgramMismatchError):
        _new_process_like(prog, x)
    assert CC.stats.mismatched == 1


def test_opt_out(cache, monkeypatch):
    monkeypatch.setenv("TRUESTAR_COMPILE_CACHE", "0")
    prog, x = _program()
    cp, _ = _new_process_like(prog, x)
    cp2, _ = _new_process_like(prog, x)
    assert cp.compile_cache["verify"] == cp2.compile_cache["verify"] == "off"
    assert cp2.kernel_verify_seconds > 0
    assert not (cache / "programs").exists()


def test_verify_all_is_never_skipped(cache, monkeypatch):
    prog, x = _program()
    _new_process_like(prog, x)
    monkeypatch.setenv("TRUESTAR_VERIFY", "all")
    cp, _ = _new_process_like(prog, x)
    assert cp.compile_cache["verify"] == "off" and cp.kernel_verify_seconds > 0


def test_trim_evicts_least_recently_used(cache):
    d = cache / "programs"
    d.mkdir(parents=True)
    for i in range(10):
        p = d / f"{i}.json"
        p.write_bytes(b"x" * 1000)
        os.utime(p, (1000 + i, 1000 + i))
    with pytest.warns(CC.CompileCacheWarning, match="TS1352"):
        n = CC.trim(limit=5000)
    left = sorted(int(p.stem) for p in d.glob("*.json"))
    assert n == 6 and left == [6, 7, 8, 9]       # the four most recent, under 90% of the cap
    assert CC.trim(limit=5000) == 0


def test_cli_info_and_clear(cache):
    prog, x = _program()
    _new_process_like(prog, x)
    env = dict(os.environ, PYTHONPATH=SRC)
    p = subprocess.run([sys.executable, "-m", "truestar", "cache", "info"], env=env, capture_output=True,
                       text=True, check=True)
    doc = json.loads(p.stdout)
    assert doc["dir"] == str(cache) and doc["entries"]["programs"]["count"] == 1 and doc["bytes"] > 0
    p = subprocess.run([sys.executable, "-m", "truestar", "cache", "clear"], env=env, capture_output=True,
                       text=True, check=True)
    doc = json.loads(p.stdout)
    assert doc["deleted"] > 0 and doc["after"]["bytes"] == 0


_CHILD = """
import hashlib, json, sys, torch, torch.nn as nn
from truestar import ir
from truestar.compiler import compile_program
torch.manual_seed(0)
m = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 8)).eval()
x = torch.randn(4, 16)
dev = sys.argv[1]
cp = compile_program(ir.import_module(m, (x,)), dev, "float32")
out = cp([x.to(dev)])[0]
print(json.dumps({"verify": cp.compile_cache["verify"], "checks": cp.verify_seconds + cp.kernel_verify_seconds,
                  "sha": hashlib.sha256(out.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}))
"""


def _child(device, cache):
    env = dict(os.environ, PYTHONPATH=SRC, TRUESTAR_CACHE_DIR=str(cache))
    env.pop("TRITON_CACHE_DIR", None)
    p = subprocess.run([sys.executable, "-c", _CHILD, device], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"))])
def test_new_process_hit_bitwise_equal(cache, device):
    a = _child(device, cache)
    b = _child(device, cache)
    assert a["verify"] == "stored" and a["checks"] > 0
    assert b["verify"] == "hit" and b["checks"] == 0
    assert a["sha"] == b["sha"]
    if device == "cuda":     # Triton's binaries are kept under the Truestar cache
        assert any((cache / "triton" / "bin").iterdir())
