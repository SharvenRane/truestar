"""Data parallel, sharded state and the gradient sync beyond 2 ranks (ledger row P66): 1, 3 and 4 real
gloo processes on the CPU, each the same DeviceComm a GPU job uses (docs/TRAINING.md, Multi GPU).

Bitwise where the design promises it (fixed order sums against one process that adds the same per
replica pieces in rank order; sharded state against replicated state), within 1e-12 against the
whole batch step otherwise; replicas that hold unequal batches (TS1210) and reduce scatters whose
dim does not split (TS1211) are refused on every rank, so no rank waits in a sum.
"""

import os
import sys
import threading

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

needs_gloo = pytest.mark.skipif(not (torch.distributed.is_available() and torch.distributed.is_gloo_available()),
                                reason="needs torch.distributed with gloo")


def _spawn(target, world, d, *args, timeout=240):
    import torch.multiprocessing as mp

    ps = [mp.get_context("spawn").Process(target=target, args=(r, world, d) + args) for r in range(world)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(timeout)
        if p.is_alive():
            p.kill()
    return [p.exitcode for p in ps]


def _join(rank, world, d):
    from truestar.parallel.comm import file_init_method
    from truestar.parallel.devices import DeviceComm, join

    info = join(rank, world, file_init_method(os.path.join(d, "init")), device="cpu", timeout=60)
    import torch.distributed as dist

    g = tuple(range(world))
    return DeviceComm(rank, world, "cpu", info.backend, {g: dist.new_group(list(g)) if world > 1 else None})


# ----------------------------------------------------------------------------- native step


def _native_rank(rank, world, d, rows):
    sys.path.insert(0, HERE)
    import test_multigpu as T
    from truestar.fitting import FitError
    from truestar.parallel.devices import leave

    comm = _join(rank, world, d)
    truestar, optim, net, loss, _ = T._native()
    g = torch.Generator().manual_seed(3)
    data = [(torch.randn(rows, 8, generator=g), torch.randint(0, 3, (rows,), generator=g)) for _ in range(3)]
    step = truestar.train_step(loss, optim.AdamW(lr=1e-2), clip=1.0, data_parallel=comm, bucket_bytes=200,
                               check_replicas=1)
    ts = step.init(net.init(seed=0))
    out = {}
    try:
        for x, y in data:
            ts, r = step(ts, x.chunk(world)[rank], y.chunk(world)[rank])
        out = {"params": ts.params, "loss": r.loss}
    except FitError as e:
        out = {"error": str(e)}
    torch.save(out, os.path.join(d, f"r{rank}.pt"))
    leave()


def _native_reference(rows, world):
    sys.path.insert(0, HERE)
    import test_multigpu as T

    truestar, optim, net, loss, _ = T._native()
    g = torch.Generator().manual_seed(3)
    data = [(torch.randn(rows, 8, generator=g), torch.randint(0, 3, (rows,), generator=g)) for _ in range(3)]
    ref = {}
    for acc in (1, world):
        step = truestar.train_step(loss, optim.AdamW(lr=1e-2), clip=1.0, accumulate=acc)
        ts = step.init(net.init(seed=0))
        for x, y in data:
            ts, r = step(ts, x, y)
        ref[acc] = ts.params
    return ref, T._leaves


@needs_gloo
@pytest.mark.parametrize("world", [1, 3, 4])
def test_native_data_parallel_equals_one_process_for_any_world(tmp_path, world):
    """world processes, one micro batch each: bitwise the one process step with `world` micro batches
    (the same left to right sum), within 1e-12 of the whole batch step, replicas identical."""
    rows = 24
    codes = _spawn(_native_rank, world, str(tmp_path), rows)
    assert codes == [0] * world, codes
    got = [torch.load(os.path.join(str(tmp_path), f"r{r}.pt"), weights_only=False) for r in range(world)]
    assert all("error" not in g for g in got), [g.get("error") for g in got]
    ref, leaves = _native_reference(rows, world)
    for g in got[1:]:
        assert all(torch.equal(a, b) for a, b in zip(leaves(g["params"]), leaves(got[0]["params"])))
    assert all(torch.equal(a, b) for a, b in zip(leaves(got[0]["params"]), leaves(ref[world])))
    for a, b in zip(leaves(got[0]["params"]), leaves(ref[1])):
        assert float((a - b).abs().max()) <= 1e-12 * float(b.abs().max())


@needs_gloo
def test_native_data_parallel_refuses_unequal_replica_batches_on_every_rank(tmp_path):
    """16 rows over 3 replicas are 6, 6 and 4: the sum over replicas divided by 3 is not the batch mean.
    Before TS1210 this trained silently on the wrong mean; now every rank raises the same error."""
    codes = _spawn(_native_rank, 3, str(tmp_path), 16)
    assert codes == [0, 0, 0], codes
    got = [torch.load(os.path.join(str(tmp_path), f"r{r}.pt"), weights_only=False) for r in range(3)]
    for g in got:
        assert "TS1210" in g.get("error", ""), g
        assert "rows per replica: [6, 6, 4]" in g["error"] and "pad or drop 1" in g["error"], g["error"]


# ----------------------------------------------------------------------------- torch.nn gradient sync and sharding


def _model():
    sys.path.insert(0, HERE)
    import parallel_models as PM

    torch.manual_seed(0)
    return PM.Tiny(d=24, ff=48, head_dim=4, act=torch.tanh).double()


def _batch(rows):
    g = torch.Generator().manual_seed(9)
    return torch.randn(rows, 4, 12, generator=g, dtype=torch.float64), torch.randn(rows, 4, 6, generator=g,
                                                                                    dtype=torch.float64)


def _sync_rank(rank, world, d, order):
    from truestar.parallel.devices import leave
    from truestar.parallel.gradsync import GradientSync
    from truestar.parallel.sharded import FlatShard

    comm = _join(rank, world, d)
    comm.order = order
    group = tuple(range(world))
    x, y = _batch(12)
    n = float(y.numel())
    xr, yr = x.chunk(world)[rank], y.chunk(world)[rank]
    out = {}
    for sharded in (False, True):
        m = _model()
        ps = [p for p in m.parameters()]
        if not sharded:
            sync = GradientSync(list(m.named_parameters()), comm, group, bucket_bytes=4096, overlap=True)
        if sharded:
            sh = FlatShard(list(m.named_parameters()), rank, world)
            flat = sh.flat_params().clone()
            opt = torch.optim.AdamW([torch.nn.Parameter(sh.mine(flat).clone())], lr=1e-2, foreach=False)
        else:
            opt = torch.optim.AdamW(ps, lr=1e-2, foreach=False)
        for _ in range(3):
            for p in ps:
                p.grad = None
            (((m(xr) - yr) ** 2).sum() / n).backward()
            if sharded:   # each replica receives the sum of its own slice only
                gs = comm.reduce_scatter(sh.flat_grads(), group, 0, "gradient reduce scatter")
                piece = opt.param_groups[0]["params"][0]
                piece.grad = gs
                opt.step()
                full = comm.all_gather(piece.detach(), group, 0, "parameter all gather")
                sh.write_params(full.clone())
            else:
                sync.finish()
                opt.step()
        out["sharded" if sharded else "replicated"] = torch.cat([p.detach().reshape(-1) for p in m.parameters()])
    try:
        comm.reduce_scatter(torch.ones(world * 5 + 1), group, 0, "an uneven flat vector")
        out["uneven"] = "no error"
    except ValueError as e:
        out["uneven"] = str(e)
    torch.save(out, os.path.join(d, f"s{rank}.pt"))
    leave()


@needs_gloo
@pytest.mark.parametrize("world,order", [(3, "fixed"), (4, "fixed"), (3, "native")])
def test_gradient_sync_and_sharded_state_for_three_and_four_ranks(tmp_path, world, order):
    codes = _spawn(_sync_rank, world, str(tmp_path), order)
    assert codes == [0] * world, codes
    got = [torch.load(os.path.join(str(tmp_path), f"s{r}.pt"), weights_only=False) for r in range(world)]
    # same tree: one process computes each replica's gradient and adds them in rank order
    m = _model()
    ps = list(m.parameters())
    opt = torch.optim.AdamW(ps, lr=1e-2, foreach=False)
    x, y = _batch(12)
    n = float(y.numel())
    whole = _model()
    wopt = torch.optim.AdamW(list(whole.parameters()), lr=1e-2, foreach=False)
    for _ in range(3):
        acc = None
        for xr, yr in zip(x.chunk(world), y.chunk(world)):
            m.zero_grad(set_to_none=True)
            (((m(xr) - yr) ** 2).sum() / n).backward()
            gs = [p.grad.clone() for p in ps]
            acc = gs if acc is None else [a + g for a, g in zip(acc, gs)]
        for p, a in zip(ps, acc):
            p.grad = a
        opt.step()
        whole.zero_grad(set_to_none=True)
        (((whole(x) - y) ** 2).sum() / n).backward()
        wopt.step()
    tree = torch.cat([p.detach().reshape(-1) for p in ps])
    ref = torch.cat([p.detach().reshape(-1) for p in whole.parameters()])
    for g in got:
        assert torch.equal(g["replicated"], got[0]["replicated"])    # replicas identical
        assert "TS1211" in g["uneven"] and f"{world * 5 + 1} entries" in g["uneven"], g["uneven"]
    if order == "fixed":   # gloo's own all reduce over 3 ranks is not the rank order sum (TS825)
        assert torch.equal(got[0]["replicated"], tree)
    # Sharded state sums bitwise the same gradients, but torch's CPU AdamW gives an element other bits
    # near the end of a tensor (vector body against scalar tail; 1 entry here), and a replica's slice
    # ends elsewhere than the parameters do. So on the CPU it is checked within the bound; the GPU
    # run (scripts/multigpu_verify.py, phase shard) checks it bit for bit.
    from truestar.parallel.check import TOL   # float64: the layout check's 1e-9
    for k in ("replicated", "sharded"):
        assert float((got[0][k] - ref).abs().max()) <= TOL * float(ref.abs().max()), k
        assert float((got[0][k] - tree).abs().max()) <= TOL * float(tree.abs().max()), k


def test_reduce_scatter_that_does_not_split_is_refused_on_simulated_ranks():
    from truestar.parallel.comm import ThreadComm, UnevenSplit, _Hub

    hub = _Hub(10.0)
    errs = [None] * 3

    def run(r):
        try:
            ThreadComm(hub, r, 3).reduce_scatter(torch.ones(10), (0, 1, 2), 0, "test")
        except UnevenSplit as e:
            errs[r] = str(e)

    ts = [threading.Thread(target=run, args=(r,)) for r in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(20)
    assert all(e and "TS1211" in e and "10 entries" in e for e in errs), errs
