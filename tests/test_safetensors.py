"""safetensors weights (truestar.interop.safetensors): format, validation, checks against the model."""

import glob
import json
import os
import struct

import pytest
import torch

import truestar
from truestar import Tensor, ops
from truestar import ir
from truestar.interop import (LoadReport, SafetensorsError, load_file, load_weights, read_header, save_file,
                              save_weights)

st = pytest.importorskip("safetensors.torch")


def weights_dict():
    g = torch.Generator().manual_seed(0)
    return {"w1": torch.randn(8, 4, generator=g), "b1": torch.randn(8, generator=g),
            "half": torch.randn(3, 5, generator=g).half(), "bf": torch.randn(7, generator=g).bfloat16(),
            "ids": torch.arange(5), "mask": torch.tensor([True, False, True]), "scalar": torch.tensor(2.5),
            "empty": torch.zeros(0, 3)}


def test_round_trip_with_the_safetensors_package(tmp_path):
    w = weights_dict()
    ours, theirs = str(tmp_path / "ours.safetensors"), str(tmp_path / "theirs.safetensors")
    save_file(w, ours, {"format": "pt"})
    st.save_file(w, theirs, {"format": "pt"})
    for path in (ours, theirs):
        a, b = load_file(path), st.load_file(path)
        assert set(a) == set(b) == set(w)
        for k in w:
            assert a[k].dtype == b[k].dtype == w[k].dtype and torch.equal(a[k], b[k]) and torch.equal(a[k], w[k])
    assert read_header(ours)[1] == {"format": "pt"}


def test_bytes_match_the_package_writer(tmp_path):
    w = weights_dict()
    ours, theirs = str(tmp_path / "ours.safetensors"), str(tmp_path / "theirs.safetensors")
    save_file(w, ours, {"format": "pt"})
    st.save_file(w, theirs, {"format": "pt"})
    with open(ours, "rb") as a, open(theirs, "rb") as b:
        assert a.read() == b.read()


def test_a_real_hub_checkpoint_reads_bit_for_bit():
    files = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--openai-community--gpt2/"
                                                "snapshots/*/model.safetensors")))
    if not files:
        pytest.skip("no cached GPT2 safetensors file")
    from safetensors import safe_open
    ours = load_file(files[0])
    with safe_open(files[0], "pt") as f:
        keys = list(f.keys())
        assert sorted(keys) == sorted(ours)
        for k in keys:
            assert torch.equal(f.get_tensor(k), ours[k]), k


def test_mmap_is_copy_on_write(tmp_path):
    path = str(tmp_path / "w.safetensors")
    save_file({"a": torch.zeros(1024)}, path)
    with open(path, "rb") as f:
        before = f.read()
    t = load_file(path)["a"]
    t += 1                            # writes go to private pages, never to the file
    with open(path, "rb") as f:
        assert f.read() == before
    assert torch.equal(load_file(path)["a"], torch.zeros(1024))
    del t


def raw_file(path, header, data: bytes):
    hb = json.dumps(header).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + data)


def test_every_problem_is_listed_at_once(tmp_path):
    path = str(tmp_path / "bad.safetensors")
    hb = ('{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
          '"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
          '"b":{"dtype":"F32","shape":[3],"data_offsets":[8,16]},'
          '"c":{"dtype":"Q4","shape":[1],"data_offsets":[16,17]},'
          '"d":{"dtype":"F32","shape":[1],"data_offsets":[20,24]}}').encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + bytes(30))
    with pytest.raises(SafetensorsError) as ei:
        load_file(path)
    p = "\n".join(ei.value.problems)
    assert "a: the name appears more than once" in p
    assert "b: F32[3] needs 12 bytes but data_offsets [8, 16] hold 8" in p
    assert "c: unknown dtype 'Q4'" in p
    assert "d: a hole of 12 bytes" in p
    assert "the data buffer is 30 bytes (trailing bytes)" in p
    with pytest.raises(Exception):   # the package rejects it too
        st.load_file(path)


def test_overlap_truncation_and_header_errors(tmp_path):
    path = str(tmp_path / "x.safetensors")
    raw_file(path, {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                    "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}}, bytes(12))
    with pytest.raises(SafetensorsError, match="overlap"):
        load_file(path)
    raw_file(path, {"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}, bytes(10))
    with pytest.raises(SafetensorsError, match="truncated"):
        load_file(path)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 10 ** 12))
    with pytest.raises(SafetensorsError, match="100 MB"):
        load_file(path)


def test_unaligned_tensor_is_copied_and_named(tmp_path):
    path = str(tmp_path / "u.safetensors")
    x = torch.tensor([1.5, -2.0, 3.25])
    raw_file(path, {"byte": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
                    "x": {"dtype": "F32", "shape": [3], "data_offsets": [1, 13]}},
             bytes([7]) + x.numpy().tobytes())
    rep = LoadReport(path)
    out = load_file(path, report=rep)
    assert torch.equal(out["x"], x) and rep.copied == ["x"]
    assert torch.equal(st.load_file(path)["x"], x)


# ----------------------------------------------------------------------------- into a native function


@truestar.function
def mlp(x: Tensor["B", 4], w1: Tensor["H", 4], b1: Tensor["H"], w2: Tensor[3, "H"]) -> Tensor["B", 3]:
    return ops.linear(ops.tanh(ops.linear(x, w1, b1)), w2)


def mlp_weights(h=6):
    g = torch.Generator().manual_seed(1)
    return {"w1": torch.randn(h, 4, generator=g), "b1": torch.randn(h, generator=g),
            "w2": torch.randn(3, h, generator=g)}


def test_load_into_native_function_and_run(tmp_path):
    path = str(tmp_path / "mlp.safetensors")
    w = mlp_weights()
    save_weights(path, w, check=mlp)
    loaded = load_weights(path, into=mlp)
    assert loaded.data_inputs == ["x"] and loaded.check_errors is not None and max(loaded.check_errors) < 1e-12
    x = torch.randn(5, 4)
    assert torch.equal(mlp(x, **loaded), mlp(x, **w))


def test_native_function_mismatches_listed_together(tmp_path):
    path = str(tmp_path / "bad.safetensors")
    w = mlp_weights()
    w["w2"] = torch.randn(3, 7)            # H = 7 here, 6 in w1
    w["b1"] = w["b1"].double()
    w["extra"] = torch.zeros(1)
    save_file(w, path)
    with pytest.raises(SafetensorsError) as ei:
        load_weights(path, into=mlp)
    p = "\n".join(ei.value.problems)
    assert "extra: in the file but not in the function" in p
    assert "b1: dtype F64 in the file, F32 in the function" in p
    assert "w2: dim 1 is H = 7, but b1 has H = 6" in p   # names bind in sorted order
    assert ei.value.code == "safetensors_mismatch"


def test_swapped_same_shape_tensors_refused_by_the_reference_record(tmp_path):
    @truestar.function
    def two(x: Tensor["B", 4], wa: Tensor[4, 4], wb: Tensor[4, 4]) -> Tensor["B", 4]:
        return ops.tanh(x @ wa) @ wb

    g = torch.Generator().manual_seed(2)
    w = {"wa": torch.randn(4, 4, generator=g), "wb": torch.randn(4, 4, generator=g)}
    path = str(tmp_path / "two.safetensors")
    save_weights(path, w, check=two)
    meta = read_header(path)[1]
    swapped = str(tmp_path / "swapped.safetensors")
    save_file({"wa": w["wb"], "wb": w["wa"]}, swapped, meta)   # same names, shapes and dtypes
    with pytest.raises(SafetensorsError) as ei:
        load_weights(swapped, into=two)
    assert ei.value.code == "safetensors_check_failed" and "swapped" in ei.value.suggestion
    transposed = str(tmp_path / "t.safetensors")
    save_file({"wa": w["wa"].T.contiguous(), "wb": w["wb"]}, transposed, meta)
    with pytest.raises(SafetensorsError, match="do not reproduce"):
        load_weights(transposed, into=two)
    assert load_weights(swapped, into=two, verify=False)["wa"] is not None


# ----------------------------------------------------------------------------- into a module and a program


def make_module():
    torch.manual_seed(3)
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)).eval()


def test_module_transposed_weight_torch_accepts_truestar_refuses(tmp_path):
    m = make_module()
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    path = str(tmp_path / "m.safetensors")
    save_weights(path, sd, check=m, check_inputs=(torch.randn(3, 4),))
    meta = read_header(path)[1]
    bad = dict(sd)
    bad["0.weight"] = sd["0.weight"].T.contiguous()
    badpath = str(tmp_path / "bad.safetensors")
    save_file(bad, badpath, meta)
    m2 = make_module()
    m2.load_state_dict(st.load_file(badpath), strict=True)        # PyTorch: accepted, silently wrong
    assert not torch.equal(m2[0].weight, m[0].weight)
    m3 = make_module()
    with torch.no_grad():
        m3[0].weight.zero_()
    with pytest.raises(SafetensorsError, match="do not reproduce"):
        load_weights(badpath, into=m3)
    assert torch.equal(m3[0].weight, torch.zeros(4, 4))            # nothing was loaded
    load_weights(path, into=m3)
    assert torch.equal(m3[0].weight, m[0].weight)


def test_module_missing_unexpected_and_dtype(tmp_path):
    m = make_module()
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    del sd["2.bias"]
    sd["0.weight"] = sd["0.weight"].half()
    sd["stray"] = torch.zeros(2)
    path = str(tmp_path / "m.safetensors")
    save_file(sd, path)
    with pytest.raises(SafetensorsError) as ei:
        load_weights(path, into=make_module())
    p = "\n".join(ei.value.problems)
    assert "2.bias: the module needs F32[2]" in p and "stray: in the file" in p and "0.weight: dtype F16" in p


def test_ir_program_params(tmp_path):
    m = make_module()
    x = torch.randn(2, 4)
    prog = ir.import_module(m, (x,))
    m2 = make_module()
    with torch.no_grad():
        for p in m2.parameters():
            p.mul_(-1.5)
    path = str(tmp_path / "p.safetensors")
    save_file({k: v.detach().clone() for k, v in m2.state_dict().items()}, path)
    loaded = load_weights(path, into=prog)
    got = ir.run(loaded.program, x)[0]
    assert torch.allclose(got, m2(x).double(), atol=1e-6)
    assert not torch.allclose(ir.run(prog, x)[0], got)      # the original program is unchanged


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_load_to_cuda(tmp_path):
    path = str(tmp_path / "w.safetensors")
    w = weights_dict()
    save_file(w, path)
    out = load_file(path, device="cuda")
    assert all(t.device.type == "cuda" for t in out.values())
    assert torch.equal(out["w1"].cpu(), w["w1"])


def test_tied_weights_need_one_name_in_the_file(tmp_path):
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(10, 4)
            self.head = torch.nn.Linear(4, 10, bias=False)
            self.head.weight = self.emb.weight

        def forward(self, ids):
            return self.head(self.emb(ids))

    src = Tied().eval()
    path = str(tmp_path / "tied.safetensors")
    save_weights(path, {"emb.weight": src.emb.weight.detach().clone()}, check=src,
                 check_inputs=(torch.tensor([[1, 2, 9]]),))
    dst = Tied().eval()
    res = load_weights(path, into=dst)
    assert res.tied == {"head.weight": "emb.weight"}
    assert dst.head.weight is dst.emb.weight and torch.equal(dst.emb.weight, src.emb.weight)
