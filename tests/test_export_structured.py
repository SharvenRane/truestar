"""Export of control flow, nested inputs and outputs, keyword arguments and more symbolic dims.

* torch.cond exports with both sides verified: the side no check takes is run on the operands that
  reached it and compared with torch's own captured branch;
* a scan RNN exports with a symbolic length and batch, each verified at a second size;
* dict, list, tuple, NamedTuple and dataclass inputs and outputs, and keyword arguments, round trip
  through the artifact (also in a fresh process), and a mismatch names the exact leaf;
* the batch symbol has no upper bound unless one is asked for.
"""

import collections
import dataclasses
import os
import subprocess
import sys
import textwrap
import warnings

import pytest
import torch
from torch import nn
from torch._higher_order_ops.scan import scan

from truestar import pytree
from truestar.cli import _parse_dynamic, main as cli_main
from truestar.export import ExportError, VerificationError, export, load
from truestar.ir import reference

warnings.filterwarnings("ignore")

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def rel(a, b):
    s = max(a.abs().max().item(), b.abs().max().item(), 1e-300)
    return (a - b).abs().max().item() / s


# ----------------------------------------------------------------------------- control flow


class CondBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x):
        return torch.cond(x.mean() > 0, lambda t: torch.relu(self.lin(t)), lambda t: -t * 0.5, (x,))


def test_cond_exports_with_both_sides_verified(tmp_path):
    torch.manual_seed(0)
    m = CondBranch().eval()
    path = tmp_path / "c.tsx"
    res = export(m, torch.rand(2, 3) + 0.5, str(path))   # every check takes the then side
    (b,) = res.evidence["branches"]
    assert b["then_taken"] >= 2 and b["else_taken"] == 0
    assert b["other_side"]["side"] == "else" and b["other_side"]["passed"]
    assert b["other_side"]["max_rel_error"] < 1e-12
    assert "test_export_structured.py" in b["where"]
    assert "else side never taken: checked against torch's branch" in res.to_text()
    run = load(str(path))
    for x in (torch.rand(7, 3) + 0.5, -torch.rand(5, 3) - 0.5):
        with torch.no_grad():
            assert rel(run(x)[0], m(x)) < 1e-6


def test_wrong_untaken_branch_is_caught_and_nothing_written(tmp_path, monkeypatch):
    """The else side uses neg; break neg in the reference. No check takes that side, so only the
    branch check can see it, and it must refuse."""
    monkeypatch.setitem(reference.IMPLS, "neg", lambda attrs, args, shape: args[0] * 1.001)
    path = tmp_path / "bad.tsx"
    with pytest.raises(VerificationError) as err:
        export(CondBranch().eval(), torch.rand(2, 3) + 0.5, str(path))
    assert "never taken by the checks, disagrees with torch's captured else branch" in str(err.value)
    assert not path.exists()


class ScanRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.wx = nn.Parameter(torch.randn(4, 5) * 0.5)
        self.wh = nn.Parameter(torch.randn(5, 5) * 0.4)

    def forward(self, xs, h0):
        def step(h, x):
            h2 = torch.tanh(x @ self.wx + h @ self.wh)
            return h2, h2.clone()
        return scan(step, h0, xs)


def test_scan_rnn_exports_with_symbolic_length_and_batch(tmp_path):
    torch.manual_seed(1)
    m = ScanRNN()
    path = tmp_path / "rnn.tsx"
    res = export(m, (torch.randn(6, 2, 4), torch.randn(2, 5)), str(path), dynamic_batch=False,
                 dynamic_dims={"xs": {0: "T", 1: "B"}, "h0": {0: "B"}})
    names = [c["name"] for c in res.evidence["checks"]]
    assert names == ["example", "second_B", "second_T"]
    assert [str(v.type) for v in res.program.inputs] == ["f32[T, B, 4]", "f32[B, 5]"]
    run = load(str(path))
    xs, h0 = torch.randn(13, 3, 4), torch.randn(3, 5)
    got = run(xs, h0)
    want = m(xs, h0)
    assert rel(got[0], want[0]) < 1e-6 and rel(got[1], want[1]) < 1e-6


def test_specialized_dynamic_dim_is_refused():
    class FixedLen(nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], 2, 3).sum(1)
    with pytest.raises(ExportError, match="x"):
        export(FixedLen(), torch.randn(2, 6), dynamic_dims={"x": {1: "T"}})


# ----------------------------------------------------------------------------- pytrees


Out = collections.namedtuple("Out", ["logits", "aux"])


@dataclasses.dataclass
class Batch:
    x: torch.Tensor
    mask: torch.Tensor


class Structured(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, batch, extras, *, scale):
        h = self.lin(batch.x) * batch.mask
        return {"out": Out(logits=h * scale, aux=[extras[0] + 1, extras[1].sum(1)]), "none": None}


def _structured_call(b=2):
    return ((Batch(torch.randn(b, 4), torch.ones(b, 3)), (torch.randn(b, 2), torch.randn(b, 5))),
            {"scale": torch.full((b, 1), 2.0)})


def test_nested_inputs_outputs_and_kwargs_round_trip(tmp_path):
    torch.manual_seed(2)
    m = Structured().eval()
    args, kwargs = _structured_call()
    path = tmp_path / "s.tsx"
    res = export(m, args, str(path), kwargs=kwargs)
    io = res.evidence["io"]
    assert io["arg_names"] == ["batch", "extras"]
    assert pytree.leaf_paths(io["outputs"]) == ["['out'].logits", "['out'].aux[0]", "['out'].aux[1]"]
    run = load(str(path))
    args, kwargs = _structured_call(5)
    got = run(*args, **kwargs)
    with torch.no_grad():
        want = m(*args, **kwargs)
    assert set(got) == {"out", "none"} and got["none"] is None
    assert type(got["out"]).__name__ == "Out" and got["out"]._fields == ("logits", "aux")
    assert rel(got["out"].logits, want["out"].logits) < 1e-6
    for g, w in zip(got["out"].aux, want["out"].aux):
        assert rel(g, w) < 1e-6
    flat = run(*[t for t in pytree.match(io["inputs"], (args, kwargs), lambda t: isinstance(t, torch.Tensor))])
    assert isinstance(flat, list) and len(flat) == 3  # the flat form every artifact accepts


@pytest.mark.parametrize("bad,where", [
    (lambda a, k: ((a[0], (a[1][0],)), k), "extras: expected 2 items, got 1"),
    (lambda a, k: ((a[0], a[1]), {"scal": k["scale"]}), "scale: missing keyword argument scale; unexpected keyword argument scal"),
    (lambda a, k: (({"x": a[0].x, "mask": a[0].mask}, a[1]), k), "batch: expected a Batch, got a dict"),
    (lambda a, k: ((Batch(a[0].x.double(), a[0].mask), a[1]), k), "batch.x: is torch.float64"),
    (lambda a, k: ((Batch(a[0].x[:, :3], a[0].mask), a[1]), k), "batch.x: has shape [2, 3]"),
])
def test_structure_mismatch_names_the_leaf(tmp_path, bad, where):
    m = Structured().eval()
    args, kwargs = _structured_call()
    path = tmp_path / "s.tsx"
    export(m, args, str(path), kwargs=kwargs)
    a, k = bad(*_structured_call())
    with pytest.raises(pytree.StructureError) as err:
        load(str(path))(*a, **k)
    assert where in str(err.value)


def test_renamed_dict_key_names_both_keys():
    spec = pytree.flatten({"params": {"w": torch.ones(1), "b": torch.ones(1)}}, torch.is_tensor)[1]
    with pytest.raises(pytree.StructureError) as err:
        pytree.match(spec, {"params": {"w": torch.ones(1), "bias": torch.ones(1)}}, torch.is_tensor)
    msg = str(err.value)
    assert msg.splitlines()[0] == ("['params']['b']: missing key 'b'; unexpected key 'bias' "
                                   "(expected keys: 'w', 'b')")


FRESH = textwrap.dedent('''
    import sys, torch
    from truestar.export import load
    run = load(sys.argv[1])
    out = run({"a": torch.ones(3, 2), "b": [torch.ones(3, 2) * 2]}, k=torch.ones(3, 1))
    print(sorted(out), type(out["pair"]).__name__, out["pair"][0].shape, float(out["sum"].sum()))
''')


def test_structured_artifact_in_a_fresh_process(tmp_path):
    class DictModel(nn.Module):
        def forward(self, d, k):
            return {"sum": d["a"] + d["b"][0] * k, "pair": (d["a"] * 2, d["b"][0])}
    path = tmp_path / "d.tsx"
    export(DictModel(), ({"a": torch.randn(2, 2), "b": [torch.randn(2, 2)]},), str(path), kwargs={"k": torch.randn(2, 1)})
    script = tmp_path / "go.py"
    script.write_text(FRESH)
    env = dict(os.environ, PYTHONPATH=SRC, CUDA_VISIBLE_DEVICES="-1")
    proc = subprocess.run([sys.executable, str(script), str(path)], env=env, capture_output=True, text=True,
                          timeout=300, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.split() == ["['pair',", "'sum']", "tuple", "torch.Size([3,", "2])", "18.0"]


# ----------------------------------------------------------------------------- unbounded dims, CLI


def test_batch_is_unbounded_by_default_and_bounded_on_request(tmp_path):
    m = nn.Linear(3, 2)
    path = tmp_path / "u.tsx"
    res = export(m, torch.randn(2, 3), str(path))
    assert res.evidence["batch_range"] == [1, None] and "in [1, inf]" in res.to_text()
    x = torch.randn(5000, 3)
    with torch.no_grad():
        assert rel(load(str(path))(x)[0], m(x)) < 1e-6
    assert export(m, torch.randn(2, 3), max_batch=64).evidence["batch_range"] == [1, 64]


def test_cli_dynamic_flag(tmp_path, capsys):
    assert _parse_dynamic(["xs:0=T,1=B", "h0:0=B", "y:2"]) == {"xs": {0: "T", 1: "B"}, "h0": {0: "B"},
                                                             "y": {2: "y_2"}}
    with pytest.raises(ValueError, match="ARG:AXIS"):
        _parse_dynamic(["xs"])
    src = tmp_path / "rnn_model.py"
    src.write_text(textwrap.dedent('''
        import torch
        from torch._higher_order_ops.scan import scan
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(4, 4) * 0.3)
            def forward(self, xs, h0, *, gain):
                def step(h, x):
                    h2 = torch.tanh(x + h @ self.w) * gain
                    return h2, h2.clone()
                return scan(step, h0, xs)[1]
        def get_model():
            return M()
        def get_inputs():
            return (torch.randn(5, 2, 4), torch.randn(2, 4))
        def get_kwargs():
            return {"gain": torch.ones(2, 1)}
    '''))
    path = tmp_path / "rnn.tsx"
    rc = cli_main(["export", str(src), "-o", str(path), "--static-batch", "--dynamic", "xs:0=T"])
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "check second_T" in out.out
    torch.save({"xs": torch.randn(9, 2, 4), "h0": torch.randn(2, 4), "gain": torch.ones(2, 1)}, tmp_path / "in.pt")
    # inputs.pt as a dict runs as keyword arguments: positional parameters cannot go there, so this fails by name
    assert cli_main(["run", str(path), str(tmp_path / "in.pt")]) == 2
    assert "exported with 2 positional argument(s)" in capsys.readouterr().err


def test_torch_spec_conversion_does_not_use_deprecated_children_specs():
    # torch 2.11 deprecated TreeSpec.children_specs; reading it printed a FutureWarning on every
    # export and conform run.
    import warnings

    import torch.utils._pytree as tpt

    from truestar import pytree

    _, ts = tpt.tree_flatten((torch.zeros(1), [torch.zeros(2), None], {"a": torch.zeros(3)}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = pytree.from_torch_spec(ts)
    assert spec["kind"] == "tuple" and len(spec["children"]) == 3
