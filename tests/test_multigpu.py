"""Real multi GPU ranks (ledger row P39, docs/PARALLEL.md Real ranks, docs/TRAINING.md Multi GPU).

On a machine without two GPUs the real rank code runs as separate CPU processes joined by gloo, the
same DeviceComm, gradient buckets and sharded update a GPU job uses. Tests marked `two_gpus` run
only where at least two CUDA devices exist (the Kaggle T4 x2 kernel runs them).
"""

import json
import os
import subprocess
import sys
import threading

import pytest
import torch

import parallel_models as M
from truestar import codes
from truestar.parallel import LayoutError, parallel_check, parse_layout

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
MODEL_FILE = os.path.join(HERE, "parallel_models_file.py")

needs_gloo = pytest.mark.skipif(not (torch.distributed.is_available() and torch.distributed.is_gloo_available()),
                                reason="needs torch.distributed with gloo")
two_gpus = pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.device_count() >= 2),
                              reason="needs two CUDA devices")


def test_codes_of_the_real_rank_work_exist():
    for c in ("TS820", "TS821", "TS822", "TS823", "TS824", "TS825"):
        assert c in codes.CODES


def test_shard_layout_key():
    assert parse_layout("dp=2,shard=1").shard == 1
    assert parse_layout("tp=2,dp=2,shard=2").text().endswith(",shard=2")
    assert parse_layout("tp=2").text() == "tp=2,dp=1,pp=1,ga=1,sp=0"  # unchanged without sharding
    with pytest.raises(LayoutError, match="needs dp of at least 2"):
        parse_layout("tp=2,shard=1")
    with pytest.raises(LayoutError, match="shard must be"):
        parse_layout("dp=2,shard=3")


@pytest.mark.parametrize("layout,sp", [("dp=2,shard=1", False), ("dp=4,ga=2,shard=2", False),
                                       ("tp=2,dp=2,shard=1", False), ("tp=2,dp=2,ga=2,sp=1,shard=2", True)])
def test_sharded_state_computes_the_single_device_step(layout, sp):
    r = parallel_check(M.tiny(), M.tiny_inputs(), layout, plan=M.tiny_plan(sp=sp) if "tp" in layout else None,
                       optimizer="adamw", clip=0.5)
    assert r.status == "ok", r.to_text()
    assert r.worst["update"] < 1e-9


def test_missing_parameter_gather_after_a_sharded_update_is_named():
    r = parallel_check(M.tiny(), M.tiny_inputs(), "dp=2,shard=1", optimizer="adamw", faults=["shard_stale_params"])
    f = r.first()
    assert f.code == "TS687" and f.collective == "parameter all gather over dp (sharded state)", r.to_text()


def test_clipping_on_the_local_norm_is_still_named_with_sharded_state():
    r = parallel_check(M.tiny(), M.tiny_inputs(), "tp=2,dp=2,shard=2", plan=M.tiny_plan(), optimizer="adamw",
                       clip=0.5, faults=["clip_local_norm"])
    assert r.first().code == "TS687" and r.first().collective == "gradient norm all reduce over tp", r.to_text()


def _thread_ranks(world, body):
    from truestar.parallel.comm import ThreadComm, _Hub

    hub = _Hub(30.0)
    out = [None] * world
    errs = []

    def run(r):
        try:
            out[r] = body(ThreadComm(hub, r, world))
        except BaseException as exc:  # noqa: BLE001
            errs.append(exc)
            hub.abort()

    ts = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    if errs:
        raise errs[0]
    return out


def test_gradient_buckets_sum_every_gradient_and_skip_accumulation_micro_batches():
    """Small buckets (many of them), a parameter without a gradient, and two micro batches of which only
    the last starts the sync: every rank ends with the sum of every rank's accumulated gradients."""
    from truestar.parallel.gradsync import GradientSync

    def body(comm):
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3)).double()
        unused = torch.nn.Parameter(torch.ones(4, dtype=torch.float64))
        params = list(m.named_parameters()) + [("unused", unused)]
        sync = GradientSync(params, comm, (0, 1, 2), bucket_bytes=64, overlap=True)
        g = torch.Generator().manual_seed(10 + comm.rank)
        x = torch.randn(4, 6, generator=g, dtype=torch.float64)
        with sync.no_sync():
            m(x[:2]).square().sum().backward()
        m(x[2:]).square().sum().backward()
        local = [p.grad.clone() for p in m.parameters()]
        sync.finish()
        sync.remove()
        return local, [p.grad.clone() for p in m.parameters()], unused.grad, len(sync.buckets)

    res = _thread_ranks(3, body)
    assert res[0][3] >= 3  # really several buckets
    for r in range(3):
        assert res[r][2] is None
        for i, g in enumerate(res[r][1]):
            want = res[0][0][i] + res[1][0][i] + res[2][0][i]
            assert torch.equal(g, want) or float((g - want).abs().max()) < 1e-14


def test_sharded_flat_layout_round_trips_with_padding():
    from truestar.parallel.sharded import FlatShard

    ps = [("a", torch.nn.Parameter(torch.arange(5.0))), ("b", torch.nn.Parameter(torch.arange(6.0).reshape(2, 3)))]
    sh = [FlatShard(ps, r, 3) for r in range(3)]
    assert sh[0].per == 4 and sh[0].padded == 12
    flat = sh[0].flat_params()
    assert torch.equal(torch.cat([s.mine(flat) for s in sh])[:11], torch.cat([torch.arange(5.0), torch.arange(6.0)]))
    sh[0].write_params(flat * 2)
    assert torch.equal(ps[1][1].data, torch.arange(6.0).reshape(2, 3) * 2)


def test_replica_fingerprints_catch_a_replica_that_drifted():
    from truestar.fitting import FitError
    from truestar.parallel.native import replicas_agree

    def body(comm):
        p = [torch.ones(3), torch.zeros(2)]
        if comm.rank == 1:
            p[1] = p[1] + 1e-7
        try:
            replicas_agree(comm, ["w", "b"], p, 4)
        except FitError as e:
            return str(e)
        return None

    msgs = _thread_ranks(2, body)
    assert msgs[0] and "TS824" in msgs[0] and "replica 1" in msgs[0] and " b " in msgs[0]


# ----------------------------------------------------------------------------- native step


def _native():
    import truestar
    from truestar import Tensor, nn, optim

    net = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 3))
    globals().update(Tensor=Tensor, net=net, nn=nn)

    @truestar.native
    def loss(p: net.spec, x: Tensor["B", 8], y: Tensor["B", "i64"]) -> Tensor[()]:
        return nn.cross_entropy(net(p, x), y)

    g = torch.Generator().manual_seed(3)
    data = [(torch.randn(16, 8, generator=g), torch.randint(0, 3, (16,), generator=g)) for _ in range(4)]
    return truestar, optim, net, loss, data


def _leaves(t):
    return [x for k in sorted(t) for x in _leaves(t[k])] if isinstance(t, dict) else [t]


def test_native_accumulated_step_equals_the_whole_batch_step():
    truestar, optim, net, loss, data = _native()
    out = []
    for acc in (1, 4):
        step = truestar.train_step(loss, optim.AdamW(lr=1e-2), clip=1.0, accumulate=acc)
        ts = step.init(net.init(seed=0))
        for x, y in data:
            ts, r = step(ts, x, y)
        out.append((ts, r))
    for a, b in zip(_leaves(out[0][0].params), _leaves(out[1][0].params)):
        assert float((a - b).abs().max()) <= 1e-12 * float(b.abs().max())
    assert abs(out[0][1].loss - out[1][1].loss) < 1e-12


def test_native_data_parallel_refuses_what_would_not_be_the_single_device_step():
    import truestar
    from truestar import Tensor, nn, optim
    from truestar.fitting import FitError

    net = nn.Sequential(nn.Linear(8, 16), nn.Dropout(0.2), nn.Linear(16, 3))
    globals().update(Tensor=Tensor, net=net, nn=nn)

    @truestar.native
    def loss(p: net.spec, x: Tensor["B", 8], y: Tensor["B", "i64"], training: bool = True) -> Tensor[()]:
        return nn.cross_entropy(net(p, x, training=training), y)

    with pytest.raises(FitError, match="TS823.*random ops"):
        truestar.train_step(loss, optim.SGD(lr=0.1), accumulate=2, static={"training": True})
    _, _, net2, loss2, data = _native()
    step = truestar.train_step(loss2, optim.SGD(lr=0.1), accumulate=3)
    with pytest.raises(FitError, match="TS823"):
        step(step.init(net2.init(seed=0)), *data[0])  # 16 rows do not split into 3


def _native_rank(rank, world, d):
    sys.path.insert(0, HERE)
    import test_multigpu as T
    from truestar.parallel.comm import file_init_method
    from truestar.parallel.devices import DeviceComm, join, leave

    info = join(rank, world, file_init_method(os.path.join(d, "init")), device="cpu", timeout=60)
    import torch.distributed as dist

    comm = DeviceComm(rank, world, "cpu", info.backend, {(0, 1): dist.new_group([0, 1])})
    truestar, optim, net, loss, data = T._native()
    step = truestar.train_step(loss, optim.AdamW(lr=1e-2), clip=1.0, data_parallel=comm, accumulate=2,
                               bucket_bytes=200, check_replicas=1)
    ts = step.init(net.init(seed=0))
    for x, y in data:
        ts, r = step(ts, x.chunk(world)[rank], y.chunk(world)[rank])
    torch.save({"params": ts.params, "loss": r.loss}, os.path.join(d, f"r{rank}.pt"))
    leave()


@needs_gloo
def test_native_data_parallel_on_two_processes_equals_one_process(tmp_path):
    """2 gloo processes, 2 micro batches each, several buckets: the same parameters as one process with
    4 micro batches (same reduction tree), bit for bit, and the whole batch step within 1e-12."""
    import torch.multiprocessing as mp

    truestar, optim, net, loss, data = _native()
    ref = {}
    for acc in (1, 4):
        step = truestar.train_step(loss, optim.AdamW(lr=1e-2), clip=1.0, accumulate=acc)
        ts = step.init(net.init(seed=0))
        for x, y in data:
            ts, r = step(ts, x, y)
        ref[acc] = (ts.params, r.loss)
    ps = [mp.get_context("spawn").Process(target=_native_rank, args=(r, 2, str(tmp_path))) for r in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(240)
    got = [torch.load(os.path.join(str(tmp_path), f"r{r}.pt"), weights_only=False) for r in range(2)]
    for a, b in zip(_leaves(got[0]["params"]), _leaves(got[1]["params"])):
        assert torch.equal(a, b)  # replicas identical
    for a, b in zip(_leaves(got[0]["params"]), _leaves(ref[4][0])):
        assert torch.equal(a, b)
    for a, b in zip(_leaves(got[0]["params"]), _leaves(ref[1][0])):
        assert float((a - b).abs().max()) <= 1e-12 * float(b.abs().max())


# ----------------------------------------------------------------------------- real rank processes


def test_more_gpu_ranks_than_gpus_are_refused_by_name():
    from truestar.parallel.devices import DevicesUnavailable, local_devices

    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    with pytest.raises(DevicesUnavailable):
        local_devices(n + 1, "cuda")


def test_a_rendezvous_that_never_completes_is_named_with_its_address():
    from truestar.parallel.devices import RendezvousError, free_port, join

    port = free_port()
    with pytest.raises(RendezvousError, match=f"127.0.0.1:{port}"):
        join(1, 2, f"tcp://127.0.0.1:{port}", device="cpu", backend="gloo", timeout=3)


@needs_gloo
@pytest.mark.parametrize("layout", ["dp=2", "tp=2,sp=1", "dp=2,ga=2,shard=2"])
def test_layout_check_on_real_rank_processes_is_clean(layout):
    L = parse_layout(layout)
    model, x = M.tiny(), M.tiny_inputs()
    r = parallel_check(model, x, L, plan=M.tiny_plan(sp=L.sp) if L.tp > 1 else None, optimizer="adamw", clip=0.5,
                       ranks="devices", device="cpu", model_file=MODEL_FILE, sync={"bucket_bytes": 2048, "overlap": True})
    assert r.status == "ok", r.to_text()
    assert r.codes == ["TS820"] and r.real and len(r.devices) == L.world + 1


@needs_gloo
def test_a_fault_on_real_rank_processes_is_named_as_on_threads():
    r = parallel_check(M.tiny(), M.tiny_inputs(), "dp=2,ga=4", ranks="devices", device="cpu", model_file=MODEL_FILE,
                       faults=["ga_divided_twice"], sync={"bucket_bytes": 2048, "overlap": True})
    f = r.first()
    assert f.code == "TS683" and f.collective == "gradient accumulation", r.to_text()


@needs_gloo
def test_cli_real_ranks_agent_document():
    env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + HERE, PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, "-m", "truestar", "parallel", MODEL_FILE, "--layout", "tp=2,sp=1",
                        "--ranks", "devices", "--device", "cpu", "--format", "agent"], capture_output=True, text=True,
                       env=env, encoding="utf-8", errors="replace", timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    doc = json.loads(p.stdout)
    assert doc["simulated"] is False and doc["codes"] == ["TS820"] and len(doc["devices"]) == 3


@two_gpus
@pytest.mark.parametrize("layout", ["dp=2", "tp=2,ga=2,sp=1", "dp=2,shard=2"])
def test_layout_check_on_two_gpus(layout):
    L = parse_layout(layout)
    for order in ("fixed", "native"):
        r = parallel_check(M.tiny(), M.tiny_inputs(), L, plan=M.tiny_plan(sp=L.sp) if L.tp > 1 else None,
                           optimizer="adamw", clip=0.5, ranks="devices", device="cuda", model_file=MODEL_FILE,
                           order=order)
        assert r.status == "ok", r.to_text()
        assert all("cuda" in d for d in r.devices[:2])


@two_gpus
def test_fault_named_on_two_gpus():
    r = parallel_check(M.tiny(), M.tiny_inputs(), "tp=2,sp=1", plan=M.tiny_plan(sp=True), ranks="devices",
                       device="cuda", model_file=MODEL_FILE, faults=["missing_tp_sync"])
    assert r.first().code == "TS684" and r.first().module == "blocks.0.norm1", r.to_text()
