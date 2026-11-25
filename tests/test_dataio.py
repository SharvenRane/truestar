"""truestar.data: keyed order, sharding, workers, resume, sources; checkpoints and the exact resume."""

import hashlib
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from truestar import ir, rng
from truestar import train as T
from truestar.data import (ArraySource, DataError, FolderSource, JsonLinesSource, LinesSource, Loader,
                           MemmapSource, RandomCropFlip, epoch_order, key_ints, mix_array, sample_key)
from truestar.data.order import sample_keys
from truestar.data.errors import DATA_CODES
from truestar.train import checkpoint as ckpt

CUDA = torch.cuda.is_available()


def _digest(batches):
    h = hashlib.sha256()
    for b in batches:
        for t in (b if isinstance(b, tuple) else (b,)):
            h.update(str(t.dtype).encode() + str(tuple(t.shape)).encode())
            h.update(t.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _images(n=97):
    g = np.random.default_rng(3)
    return g.standard_normal((n, 3, 8, 8)).astype(np.float32), np.arange(n, dtype=np.int64) % 10


# ----------------------------------------------------------------------------- order and sharding


def test_mix_array_is_the_rng_mixer_and_the_order_is_pinned():
    z = [0, 1, 2 ** 63, 2 ** 64 - 1, 0x1234567890ABCDEF]
    assert mix_array(np.asarray(z, dtype=np.uint64)).tolist() == [rng.mix(v) for v in z]
    o = epoch_order(1000, 42, 0)
    assert sorted(o.tolist()) == list(range(1000))
    # pinned: the order is a pure function of (n, seed, epoch), the same on every machine and OS
    assert o[:8].tolist() == PINNED_ORDER
    assert not np.array_equal(o, epoch_order(1000, 42, 1))
    assert not np.array_equal(o, epoch_order(1000, 43, 0))
    assert np.array_equal(epoch_order(10, 42, 0, shuffle=False), np.arange(10))
    # the keyed ints are the RNG's randint draw
    assert np.array_equal(key_ints(77, 50, 9), rng.generate("randint", (50,), torch.int64, 77,
                                                            "truestar.data/aug", low=0, high=9).numpy())
    from truestar.data.transforms import _AUG, _scalar_ints
    assert _scalar_ints(77, _AUG, 50, 9) == key_ints(77, 50, 9).tolist()
    assert sample_key(1, 2, 3) == rng.as_i64(rng.stream_key(1, "truestar.data/sample", [2, 3]))
    idx = np.asarray([0, 3, 2 ** 40, 17])
    assert sample_keys(1, 2, idx).tolist() == [sample_key(1, 2, int(i)) for i in idx]


PINNED_ORDER = [940, 673, 662, 964, 733, 35, 272, 327]


@pytest.mark.parametrize("world", [1, 2, 3, 5])
@pytest.mark.parametrize("drop_last", [False, True])
def test_shards_cover_every_sample_once(world, drop_last):
    n, B = 103, 4
    x = np.arange(n, dtype=np.int64)
    seen, counts = [], []
    for r in range(world):
        ld = Loader(x, B, seed=9, rank=r, world=world, drop_last=drop_last)
        bs = [b.numpy() for b in ld]
        assert len(bs) == len(ld)
        counts.append(len(bs))
        if drop_last:
            assert all(len(b) == B for b in bs)
        seen += np.concatenate(bs).tolist()
    assert len(seen) == len(set(seen))                                 # no overlap
    if drop_last:
        assert len(seen) == (n // (world * B)) * world * B and len(set(counts)) == 1
    else:
        assert sorted(seen) == list(range(n))                          # no drop


# ----------------------------------------------------------------------------- workers


def test_same_batches_for_every_worker_count_and_mode():
    x, y = _images()
    want = None
    for workers, mode in [(0, "thread"), (1, "thread"), (3, "thread"), (2, "process")]:
        with Loader(ArraySource(x, y), 8, seed=5, transform=RandomCropFlip(2), workers=workers, mode=mode) as ld:
            got = _digest(list(ld) + list(ld))          # two epochs
        want = want or got
        assert got == want, (workers, mode)
    # the batched paths (no transform; RandomCropFlip.batch) equal per sample reads plus collate
    per_sample = lambda s: (np.stack([u for u, _ in s]), np.stack([v for _, v in s]))
    assert _digest(Loader(ArraySource(x, y), 8, seed=5)) == _digest(Loader(ArraySource(x, y), 8, seed=5,
                                                                             collate=per_sample))
    aug = RandomCropFlip(2)
    ld = Loader(ArraySource(x, y), 8, seed=5, transform=aug, collate=per_sample)
    assert not ld._job.batched and _digest(list(ld) + list(ld)) == want
    assert Loader(ArraySource(x, y), 8, transform=aug)._job.batched


def test_augmentation_depends_on_the_sample_not_the_position():
    x, y = _images(40)
    one = {int(l): im for im, l in zip(*[t.numpy() for t in next(iter(Loader(ArraySource(x, np.arange(40)), 40, seed=1,
                                                                                    transform=RandomCropFlip(2))))])}
    for r in range(2):   # sharded with another batch size: the same sample gets the same augmentation
        for im, i in zip(*[t.numpy() for t in next(iter(Loader(ArraySource(x, np.arange(40)), 20, seed=1, rank=r, world=2,
                                                                transform=RandomCropFlip(2))))]):
            assert np.array_equal(one[int(i)], im)


def test_worker_and_collate_errors_name_the_batch():
    def bad(sample, key):
        if int(sample) == 7:
            raise ValueError("broken sample")
        return sample

    with pytest.raises(DataError, match=r"TS1384.*epoch 0 batch 1 .*broken sample"):
        list(Loader(np.arange(20), 4, shuffle=False, transform=bad, workers=2))
    ragged = [np.zeros(3), np.zeros(4)]
    with pytest.raises(DataError, match=r"TS1385.*shapes"):
        list(Loader(ragged, 2, shuffle=False))
    with pytest.raises(DataError, match="TS1380"):
        Loader(np.arange(3), 0)
    with pytest.raises(DataError, match="TS1380"):
        Loader(np.arange(3), 1, rank=2, world=2)


# ----------------------------------------------------------------------------- resume


def test_resume_mid_epoch_gives_the_uninterrupted_batches():
    x, y = _images()
    mk = lambda: Loader(ArraySource(x, y), 6, seed=2, rank=1, world=2, transform=RandomCropFlip(2), workers=2)
    straight = mk()
    want = list(straight) + list(straight)
    a = mk()
    got = []
    for k, b in enumerate(a):
        got.append(b)
        if k == 2:
            break                         # interrupted with prefetched batches in flight
    state = json.loads(json.dumps(a.state_dict()))
    a.close()
    b = mk()
    b.load_state_dict(state)
    got += list(b)                        # rest of epoch 0
    got += list(b)                        # epoch 1
    assert len(got) == len(want) and _digest(got) == _digest(want)
    # the same loader object continues from its position too
    c = mk()
    first = [next(iter(c))]
    rest = list(c)
    assert _digest(first + rest) == _digest(want[:len(c)])


def test_resume_refuses_other_settings_or_data():
    st = Loader(np.arange(10), 2, seed=1).state_dict()
    with pytest.raises(DataError, match="TS1382.*seed"):
        Loader(np.arange(10), 2, seed=2).load_state_dict(st)
    with pytest.raises(DataError, match="TS1382.*batch_size"):
        Loader(np.arange(10), 3, seed=1).load_state_dict(st)
    with pytest.raises(DataError, match="TS1383"):
        Loader(np.arange(11), 2, seed=1).load_state_dict(st)


def test_elastic_resume_keeps_no_overlap_no_drop():
    n = 50
    old = [Loader(np.arange(n), 3, seed=4, rank=r, world=2) for r in range(2)]
    seen = []
    for ld in old:
        it = iter(ld)
        for _ in range(2):
            seen += next(it).tolist()
    st = old[0].state_dict()
    assert st["position"] == 2 * 3 * 2
    for r in range(3):
        ld = Loader(np.arange(n), 3, seed=4, rank=r, world=3)
        ld.load_state_dict(st, elastic=True)
        for b in ld:
            seen += b.tolist()
    assert sorted(seen) == list(range(n))


# ----------------------------------------------------------------------------- sources


def test_sources(tmp_path):
    x = np.arange(60, dtype=np.float32).reshape(20, 3)
    np.save(tmp_path / "x.npy", x)
    x.tofile(tmp_path / "x.raw")
    for src in (MemmapSource(str(tmp_path / "x.npy")), MemmapSource((str(tmp_path / "x.raw"), "float32", (20, 3)))):
        assert len(src) == 20 and np.array_equal(src[4], x[4])
        got = torch.cat(list(Loader(src, 7, seed=0, workers=2, mode="process"))).numpy()
        assert np.array_equal(got, x[epoch_order(20, 0, 0)])
    for cls, name in ((1, "b.bin"), (0, "a.bin"), (1, "a.bin"), (0, "z.bin")):
        d = tmp_path / "img" / f"class{cls}"
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(f"{cls}{name}".encode())
    (tmp_path / "img" / "class0" / "skip.txt").write_text("x")
    fs = FolderSource(str(tmp_path / "img"), suffixes=[".bin"], labels=True)
    assert fs.files == ["class0/a.bin", "class0/z.bin", "class1/a.bin", "class1/b.bin"]
    assert fs[2] == (b"1a.bin", 1) and fs.classes == ["class0", "class1"]
    p = tmp_path / "t.jsonl"
    p.write_bytes(b'{"a": 1}\r\n\n{"a": 2}\n{"a": 3}')
    js = JsonLinesSource(str(p))
    assert len(js) == 3 and [js[i]["a"] for i in range(3)] == [1, 2, 3]
    first = next(iter(Loader(js, 2, shuffle=False, as_torch=False)))
    assert list(first) == ["a"] and first["a"].tolist() == [1, 2] and first["a"].dtype == np.int64
    lines = LinesSource(str(p), skip_empty=False)
    assert len(lines) == 4 and lines[1] == ""
    with pytest.raises(DataError, match="TS1381"):
        ArraySource(np.zeros(3), np.zeros(4))
    with pytest.raises(DataError, match="TS1381"):
        FolderSource(str(tmp_path / "nothing"))


@pytest.mark.skipif(not CUDA, reason="pinned memory needs CUDA")
def test_pinned_transfer_is_bitwise_the_cpu_batch():
    x, y = _images()
    cpu = list(Loader(ArraySource(x, y), 8, seed=1))
    gpu = Loader(ArraySource(x, y), 8, seed=1, pin_memory=True, device="cuda", workers=2)
    assert gpu.pinned
    for a, b in zip(cpu, gpu):
        assert b[0].is_cuda and torch.equal(a[0], b[0].cpu()) and torch.equal(a[1], b[1].cpu())


# ----------------------------------------------------------------------------- checkpoints


def test_checkpoint_file_checksums_versions_and_atomic_write(tmp_path):
    p = str(tmp_path / "c.tsx")
    t = {"w": torch.arange(6, dtype=torch.float32).reshape(2, 3), "step": torch.tensor(3)}
    ckpt.save(p, t, {"note": "x"})
    assert os.listdir(tmp_path) == ["c.tsx"]                       # no temporary file left behind
    got, meta = ckpt.load(p)
    assert torch.equal(got["w"], t["w"]) and meta["note"] == "x" and meta["version"] == ckpt.CHECKPOINT_VERSION
    raw = bytearray(open(p, "rb").read())
    raw[-40] ^= 1
    with pytest.raises(DataError, match="TS1386"):
        ckpt.loads(bytes(raw))
    from truestar.ir import serialize

    blob = serialize.dumps(serialize.loads(open(p, "rb").read()).program,
                           {ckpt.CHECKPOINT_KEY: {"version": 99}})
    with pytest.raises(DataError, match="TS1387"):
        ckpt.loads(blob)
    assert set(DATA_CODES) == {f"TS{c}" for c in range(1380, 1390)}


class DropMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.Dropout(0.25), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, x):
        return self.net(x)


def _trainer(seed=11):
    torch.manual_seed(0)
    m = DropMLP().double().train()
    prog = ir.import_module(m, (torch.randn(5, 8, dtype=torch.float64),), training=True,
                            dynamic_shapes=({0: torch.export.Dim("B", max=64)},))
    step = T.build_train_step(prog, T.CrossEntropy(label_smoothing=0.1), T.AdamW(1e-2, weight_decay=0.1),
                              params=m, clip_grad_norm=1.0)
    return T.Trainer(step, lr=T.WarmupCosine(1e-2, warmup_steps=2, total_steps=20), seed=seed)


def _loader():
    g = np.random.default_rng(0)
    x = g.standard_normal((45, 8))
    y = (np.arange(45) % 4).astype(np.int64)
    return Loader(ArraySource(x, y), 8, seed=3, workers=2)


def _forever(ld):
    while True:
        yield from ld


def test_exact_resume_n_plus_m_equals_uninterrupted(tmp_path):
    N, M = 4, 6                                   # 6 batches per epoch: the run crosses an epoch boundary
    straight, ld = _trainer(), _loader()
    for _, b in zip(range(N + M), _forever(ld)):
        straight.train_step(*b)
    first, ld1 = _trainer(), _loader()
    for _, b in zip(range(N), _forever(ld1)):
        first.train_step(*b)
    path = str(tmp_path / "run.tsx")
    first.save(path, loader=ld1)
    resumed, ld2 = _trainer(), _loader()
    resumed.load(path, loader=ld2)
    assert resumed.step_count == N and ld2.state_dict() == ld1.state_dict()
    for _, b in zip(range(M), _forever(ld2)):
        resumed.train_step(*b)
    for n in straight.state:
        assert torch.equal(straight.state[n], resumed.state[n]), n
    assert straight.history == resumed.history
    assert straight.checkpoint_bytes(ld) == resumed.checkpoint_bytes(ld2)
    # the key input is filled from the seed: a different seed draws other dropout masks
    other = _trainer(seed=12)
    with pytest.raises(DataError, match="TS1389"):
        other.load(path)
    # a loader passed to a checkpoint saved without one is refused
    p2 = str(tmp_path / "noloader.tsx")
    first.save(p2)
    with pytest.raises(DataError, match="TS1389"):
        _trainer().load(p2, loader=_loader())
    with pytest.raises(DataError, match="TS1388"):
        T.Trainer(T.build_train_step(*_other_step())).load(path)


def _other_step():
    m = nn.Sequential(nn.Linear(8, 4)).double()
    prog = ir.import_module(m, (torch.randn(5, 8, dtype=torch.float64),))
    return prog, T.MSE(), T.SGD(0.1), m
