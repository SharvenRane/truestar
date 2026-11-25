"""The importer traces einsum the same way whether or not opt_einsum is installed.

With opt_einsum present (jax installs it) torch.einsum picks a contraction path from the example
shapes while tracing, which specialized a symbolic batch dim and broke test_ir_ops[einsum]
(found 2026-11-24 after installing jax into the ml venv)."""

import torch

from truestar.ir.importer import plain_einsum


def test_plain_einsum_switches_off_and_restores():
    be = torch.backends.opt_einsum
    before = be.enabled
    with plain_einsum():
        assert be.enabled is False
    assert be.enabled == before


def test_symbolic_batch_survives_export_with_opt_einsum_enabled():
    class M(torch.nn.Module):
        def forward(self, a, b, c):
            return torch.einsum("bij,bjk,bkl->bil", a, b, c)

    B = torch.export.Dim("B")
    args = (torch.randn(2, 3, 4), torch.randn(2, 4, 5), torch.randn(2, 5, 6))
    old = torch.backends.opt_einsum.enabled
    torch.backends.opt_einsum.enabled = True
    try:
        with plain_einsum():
            ep = torch.export.export(M(), args, dynamic_shapes=({0: B}, {0: B}, {0: B}))
    finally:
        torch.backends.opt_einsum.enabled = old
    x = (torch.randn(7, 3, 4), torch.randn(7, 4, 5), torch.randn(7, 5, 6))
    assert torch.allclose(ep.module()(*x), M()(*x))
