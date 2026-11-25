"""The Triton chunk kernel of the exact tree leaf (parallel/tree_kernel.py) against the torch arm of the
same leaf: bitwise, for chunk widths 2 to 64, chunk counts with and without a carried odd chunk, fused
parts, strided operands, wide exponents and subnormal products; and a tree_linear forward and backward
with the kernel bitwise equal to the one without it. CUDA only.
"""

import os

import pytest
import torch

from truestar.parallel import tree_kernel
from truestar.parallel.tree import TreeSpec, contract_groups, group_count, tree_linear

cuda = pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.device_count() > 0), reason="needs CUDA")


def _wide(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) * torch.exp(3 * torch.randn(shape, generator=g))


@pytest.fixture
def torch_arm(monkeypatch):
    def run(fn):
        monkeypatch.setenv("TRUESTAR_TREE_KERNEL", "0")
        try:
            return fn()
        finally:
            monkeypatch.delenv("TRUESTAR_TREE_KERNEL")
    return run


def test_chunk_width():
    assert [tree_kernel.chunk_width(n) for n in (1, 3, 6, 96, 128, 256, 1000)] == [1, 1, 2, 32, 64, 64, 8]


@cuda
@pytest.mark.parametrize("m,length,n,parts,groups", [
    (37, 96, 45, 1, 1),       # C 32, 3 chunks: the upper tree carries one
    (64, 512, 64, 1, 8),      # C 64, one chunk per group
    (16, 6, 8, 1, 1),         # C 2, 3 chunks
    (33, 128, 70, 2, 4),      # two fused parts, C 16
    (130, 1024, 129, 1, 8),   # C 64, 2 chunks, edge tiles on both sides
    (5, 40, 3, 1, 1),         # C 8, 5 chunks
])
def test_kernel_is_bitwise_the_torch_leaf(torch_arm, m, length, n, parts, groups):
    if not tree_kernel.available():
        pytest.skip(f"no Triton kernel: {tree_kernel._FAILED}")
    a = _wide((m, length), 1).cuda()
    w = _wide((n, length), 2).cuda()
    b = w.t()                                              # strided, as the forward passes it
    got = contract_groups(a, b, parts, groups, "exact")
    want = torch_arm(lambda: contract_groups(a, b, parts, groups, "exact"))
    assert got.shape == want.shape == (parts, groups, m, n)
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))


@cuda
def test_subnormal_products_are_kept(torch_arm):
    if not tree_kernel.available():
        pytest.skip("no Triton kernel")
    a = torch.full((16, 64), 1e-20, device="cuda")
    b = torch.full((64, 16), 1e-20, device="cuda")       # products 1e-40, subnormal in float32
    got = contract_groups(a, b, 1, 1, "exact")
    want = torch_arm(lambda: contract_groups(a, b, 1, 1, "exact"))
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))
    assert float(got.abs().min()) > 0


@cuda
def test_tree_linear_forward_and_backward_bitwise(torch_arm):
    if not tree_kernel.available():
        pytest.skip("no Triton kernel")
    k, n, m = 256, 192, 48
    spec = TreeSpec("full", 1, group_count(k, 8), group_count(n, 8), 8, leaf="exact")
    x0 = _wide((m, k), 3).cuda()
    w0 = (_wide((n, k), 4) / k ** 0.5).cuda()
    b0 = torch.randn(n).cuda()
    gy = _wide((m, n), 5).cuda()

    def run():
        x, w, b = (t.clone().requires_grad_() for t in (x0, w0, b0))
        y = tree_linear(x, w, b, spec)
        return (y,) + torch.autograd.grad(y, (x, w, b), gy)

    got = run()
    want = torch_arm(run)
    for g, h in zip(got, want):
        assert torch.equal(g.view(torch.int32), h.view(torch.int32))
