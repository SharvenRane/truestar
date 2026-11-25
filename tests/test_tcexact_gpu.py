"""Portable tensor cores on a GPU (sm_80 and newer): compiled contraction kernels equal the NumPy
emulation bit for bit, for every launch config the tuner may pick. Skipped without CUDA."""
import numpy as np
import pytest
import torch

from truestar import kernel as K
from truestar.kernel import Graph
from truestar.kernel import tcexact as X
from truestar.kernel import triton_backend as tb

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() == 0
                                or torch.cuda.get_device_capability() < (8, 0), reason="needs sm_80+")

TD = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _out(x):
    return x["c"] if isinstance(x, dict) else x


def _mm(dtype):
    g = Graph()
    a = g.input("a", (K.sym("M"), K.sym("K")), dtype)
    b = g.input("b", (K.sym("K"), K.sym("N")), dtype)
    return g.fuse({"c": a @ b})[0][1]


@pytest.mark.parametrize("dtype,tf32", [("float16", False), ("bfloat16", False), ("float32", True)])
@pytest.mark.parametrize("shape", [(37, 50, 200), (128, 96, 1024)])
def test_contraction_equals_emulation_for_every_config(dtype, tf32, shape):
    M, N, Kd = shape
    rs = np.random.RandomState(M + N)
    a = torch.from_numpy(rs.standard_normal((M, Kd)).astype(np.float32)).to(TD[dtype])
    b = torch.from_numpy(rs.standard_normal((Kd, N)).astype(np.float32)).to(TD[dtype])
    want = X.emulate_matmul(a.float().numpy(), b.float().numpy(), X.PORTABLE_BK, tf32)
    want = torch.from_numpy(want).to(TD[dtype]).float().numpy()   # stored in the operand dtype
    kern = _mm(dtype)
    ck = K.compile(kern, "cuda", allow_tf32=tf32, portable_tensor_cores=True)
    ag, bg = a.cuda(), b.cuda()
    got = _out(ck(a=ag, b=bg)).float().cpu().numpy()
    assert np.array_equal(got.view(np.int32), want.view(np.int32))
    for cfg in ((16, 32, 64, 4, 3), (64, 64, 64, 4, 2), (128, 64, 64, 8, 2)):
        c = tb._cfg(*cfg)
        other = K.compile(kern, "cuda", allow_tf32=tf32, portable_tensor_cores=True, _config=c)
        out = _out(other(a=ag, b=bg)).float().cpu().numpy()
        assert np.array_equal(out.view(np.int32), want.view(np.int32)), cfg


def test_plain_mode_unchanged():
    kern = _mm("float16")
    ck = K.compile(kern, "cuda")
    a = torch.randn(64, 128, device="cuda").half()
    b = torch.randn(128, 32, device="cuda").half()
    torch.testing.assert_close(_out(ck(a=a, b=b)).float(), (a.float() @ b.float()), rtol=2e-3, atol=2e-3)
