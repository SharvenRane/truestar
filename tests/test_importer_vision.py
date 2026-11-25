"""The importer's ops for per image size vision encoders (ir/importer_vision.py), each against eager
PyTorch in float64 on small modules. SigLIP2 and Pixtral themselves: tests/test_hf_models.py."""

import pytest
import torch
import torch.nn.functional as F

from truestar import ir
from truestar.ir.core import UnsupportedOpError
from truestar.ir.importer_vision import aa_matrix


def _rel(got, want):
    return ((got.double() - want.double()).abs().max() / want.double().abs().max().clamp_min(1e-300)).item()


def _check(model, *args, tol=1e-12):
    prog = ir.import_module(model, args)
    with torch.no_grad():
        want = model(*args)
    got = ir.run(prog, *args)
    got = got[0] if isinstance(got, (list, tuple)) else got
    assert got.shape == want.shape
    assert _rel(got, want) < tol, _rel(got, want)
    return prog


@pytest.mark.parametrize("size_in,size_out", [((4, 4), (6, 6)), ((4, 5), (7, 3)), ((9, 8), (3, 4)),
                                              ((5, 5), (5, 5)), ((3, 7), (11, 2))])
def test_antialiased_bilinear_matches_torch(size_in, size_out):
    """Up and down sampling (the filter widens when downsampling) and the borders."""
    x = torch.randn(2, 3, *size_in, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    want = F.interpolate(x, size=size_out, mode="bilinear", antialias=True, align_corners=False)
    mh, mw = aa_matrix(size_in[0], size_out[0]), aa_matrix(size_in[1], size_out[1])
    assert _rel(mh @ x @ mw.T, want) < 1e-14

    class M(torch.nn.Module):
        def forward(self, x):
            return F.interpolate(x, size=size_out, mode="bilinear", antialias=True)

    _check(M(), x)


def test_sizes_read_from_a_constant_tensor_import():
    """A size read back with .tolist() from a tensor the model builds from constants (SigLIP2's
    spatial_shapes, Pixtral's image_sizes): exported with the value known, sliced exactly."""
    class M(torch.nn.Module):
        def forward(self, x):
            sizes = torch.tensor([[x.shape[-2], x.shape[-1]]]).expand(x.shape[0], 2)
            out = []
            for i in range(x.shape[0]):
                h, w = (sizes[i] // 2).tolist()
                out.append(x[i, :, :h, :w].flatten(1).T)
            return torch.cat(out, 0)

    _check(M(), torch.randn(2, 3, 6, 4, dtype=torch.float64))


def test_item_of_an_input_dependent_value_is_refused_by_name():
    class M(torch.nn.Module):
        def forward(self, x, n):
            return x[: n.item()] * 2

    with pytest.raises(Exception) as info:      # torch.export's own error, or TS861 at import
        ir.import_module(M(), (torch.randn(5, 3, dtype=torch.float64), torch.tensor(3)))
    assert isinstance(info.value, UnsupportedOpError) or "data-dependent" in str(info.value) \
        or "DataDependent" in type(info.value).__name__


def test_floor_divide_baddbmm_transpose_and_empty():
    class M(torch.nn.Module):
        def forward(self, x, a, b, k):
            q = torch.floor_divide(k, 3)                       # integers, negative too
            y = torch.baddbmm(x, a, b, beta=0.5, alpha=2.0)
            z = torch.empty(y.shape, dtype=y.dtype)
            z[:] = y
            return z.flatten(1).T, q

    g = torch.Generator().manual_seed(1)
    args = (torch.randn(2, 3, 4, dtype=torch.float64, generator=g), torch.randn(2, 3, 5, dtype=torch.float64, generator=g),
            torch.randn(2, 5, 4, dtype=torch.float64, generator=g), torch.tensor([-7, -3, 0, 2, 8, 9]))
    prog = ir.import_module(M(), args)
    with torch.no_grad():
        want = M()(*args)
    got = ir.run(prog, *args)
    assert _rel(got[0], want[0]) < 1e-14
    assert torch.equal(got[1], want[1])


def test_all_prod_and_div_rounding_modes():
    """The ops of transformers 5.24's loop free SigLIP2 resize (checks with all and prod, positions
    split with a floor division of constants) and integer floor and trunc division of an input, against
    torch. A float division with a rounding mode of an input is refused (TS860, tests/test_vision_ops.py):
    floor(a / b) is not torch's floor division."""
    class M(torch.nn.Module):
        def forward(self, x, k):
            sizes = torch.tensor([[2, 3], [3, 2]]).float()[:, None]
            pos = torch.arange(6, dtype=torch.float32)[None, :, None]
            rows = torch.div(pos, sizes[..., 1:], rounding_mode="floor")
            ok = (sizes.prod(-1, keepdim=True) > 0).all()
            return (x * rows.sum() * ok, torch.div(k, -3, rounding_mode="floor"),
                    torch.floor_divide(k, 2), torch.div(k, 3, rounding_mode="trunc"),
                    k.prod(0), x.prod(1))

    g = torch.Generator().manual_seed(2)
    args = (torch.randn(4, 3, dtype=torch.float64, generator=g) * 3, torch.tensor([[-7, 5], [4, -2], [3, 3]]))
    prog = ir.import_module(M(), args)
    with torch.no_grad():
        want = M()(*args)
    got = ir.run(prog, *args)
    for g_, w in zip(got, want):
        assert g_.dtype == w.dtype and g_.shape == w.shape
        assert torch.equal(g_, w) if not w.is_floating_point() else _rel(g_, w) < 1e-14
