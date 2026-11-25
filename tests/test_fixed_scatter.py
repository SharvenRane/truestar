"""Fixed order sums by all to all, fold and all gather (`fixed_algo="scatter"`, devices.py) against the
all gather and fold of every rank's whole tensor (`"gather"`) and against one process adding the same
pieces in rank order: bitwise, on 2, 3 and 4 real gloo processes on the CPU, for sizes that do and
do not split into equal pieces, for the blocking and the started (bucket) all reduce and for the
reduce scatter. The gradient sync built on it is checked bitwise in test_multirank.py.
"""

import os

import pytest
import torch

needs_gloo = pytest.mark.skipif(not (torch.distributed.is_available() and torch.distributed.is_gloo_available()),
                                reason="needs torch.distributed with gloo")

SIZES = (1, 7, 12, 1000, 4099)


def _piece(rank, size, k):
    g = torch.Generator().manual_seed(1000 * k + rank)
    # wide exponents so a different order of the additions shows in the last bits
    return torch.randn(size, generator=g) * torch.exp(4 * torch.randn(size, generator=g))


def _rank(rank, world, d):
    from truestar.parallel.comm import file_init_method
    from truestar.parallel.devices import DeviceComm, join, leave

    info = join(rank, world, file_init_method(os.path.join(d, "init")), device="cpu", timeout=60)
    import torch.distributed as dist

    g = tuple(range(world))
    groups = {g: dist.new_group(list(g))}
    scat = DeviceComm(rank, world, "cpu", info.backend, groups, order="fixed", fixed_algo="scatter")
    gath = DeviceComm(rank, world, "cpu", info.backend, groups, order="fixed", fixed_algo="gather")
    out = {}
    for k, size in enumerate(SIZES):
        x = _piece(rank, size, k)
        before = x.clone()
        out[f"ar_scatter_{size}"] = scat.all_reduce(x, g, "t")
        out[f"ar_gather_{size}"] = gath.all_reduce(x, g, "t")
        out[f"start_scatter_{size}"] = scat.all_reduce_start(x.clone(), g, "t", donate=True).wait()
        out[f"input_kept_{size}"] = bool(x.equal(before))
        m = _piece(rank, world * 3 * 5, k).view(5, world * 3)
        out[f"rs_scatter_{size}"] = scat.reduce_scatter(m, g, 1, "t")
        out[f"rs_gather_{size}"] = gath.reduce_scatter(m, g, 1, "t")
    torch.save(out, os.path.join(d, f"r{rank}.pt"))
    leave()


@needs_gloo
@pytest.mark.parametrize("world", [2, 3, 4])
def test_scatter_fixed_sum_is_bitwise_the_rank_order_sum(tmp_path, world):
    import torch.multiprocessing as mp

    ps = [mp.get_context("spawn").Process(target=_rank, args=(r, world, str(tmp_path))) for r in range(world)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(180)
    assert [p.exitcode for p in ps] == [0] * world
    got = [torch.load(os.path.join(tmp_path, f"r{r}.pt")) for r in range(world)]
    for k, size in enumerate(SIZES):
        pieces = [_piece(r, size, k) for r in range(world)]
        ref = pieces[0].clone()
        for p in pieces[1:]:
            ref = ref + p
        mats = [_piece(r, world * 3 * 5, k).view(5, world * 3) for r in range(world)]
        mref = mats[0].clone()
        for m in mats[1:]:
            mref = mref + m
        for r in range(world):
            o = got[r]
            assert o[f"ar_scatter_{size}"].equal(ref), (world, size, r)
            assert o[f"ar_gather_{size}"].equal(ref)
            assert o[f"start_scatter_{size}"].equal(ref)
            assert o[f"input_kept_{size}"]
            want = mref.chunk(world, dim=1)[r]
            assert o[f"rs_scatter_{size}"].equal(want)
            assert o[f"rs_gather_{size}"].equal(want)
    # the reference order matters: some other order gives other bits for these inputs
    if world >= 3:
        pieces = [_piece(r, SIZES[-1], len(SIZES) - 1) for r in range(world)]
        ref = pieces[0].clone()
        for p in pieces[1:]:
            ref = ref + p
        rev = pieces[-1].clone()
        for p in reversed(pieces[:-1]):
            rev = rev + p
        assert not rev.equal(ref)
