"""Real CUDA out of memory errors, provoked on purpose: the report must name the module instance.

Each OOM asks the caching allocator for far more than the card holds (1 TiB on a 16 GB card), so
the error is the real torch.OutOfMemoryError but nothing is actually allocated and the shared GPU
is not filled. Models are tiny (a few MiB).
"""

import gc
import io
import os

import pytest
import torch
from torch import nn

from truestar.diag import memory_timeline, track_memory

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
pytestmark = needs_cuda

HUGE = 1024 ** 4  # 1 TiB: fails at once (a few times VRAM can be slow, or even succeed, on Windows sysmem fallback)
THIS = os.path.abspath(__file__)


def mark(name: str) -> int:
    with open(THIS, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if f"# MARK:{name}" in line and "def mark" not in line and "mark(" not in line:
                return i
    raise KeyError(name)


@pytest.fixture(autouse=True)
def _free_gpu():
    yield
    gc.collect()
    torch.cuda.empty_cache()


class MLP(nn.Module):
    """Identical instances; only one is told to blow up, and only at one step."""

    def __init__(self, d: int, blow_at_step: int = 0):
        super().__init__()
        self.up = nn.Linear(d, 2 * d)
        self.act = nn.ReLU(inplace=True)
        self.down = nn.Linear(2 * d, d)
        self.blow_at_step = blow_at_step
        self.calls = 0

    def forward(self, x):  # MARK:mlp_forward
        self.calls += 1
        h = self.act(self.up(x))
        if self.calls == self.blow_at_step:
            # stands in for an input dependent buffer, for example attention on a very long sequence
            scratch = torch.empty(HUGE, dtype=torch.uint8, device=x.device)  # MARK:huge_alloc
            h = h + scratch[:1].float()
        return self.down(h)


class Block(nn.Module):
    def __init__(self, d: int, blow_at_step: int = 0):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mlp = MLP(d, blow_at_step)

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class Net(nn.Module):
    def __init__(self, d: int = 128, n: int = 4, blow_block: int = -1, blow_at_step: int = 0):
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, blow_at_step if i == blow_block else 0) for i in range(n))
        self.head = nn.Linear(d, 10)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x)


class _BoomFn(torch.autograd.Function):
    step = 0
    blow_at = 0

    @staticmethod
    def forward(ctx, x):
        return x * 2

    @staticmethod
    def backward(ctx, g):
        if _BoomFn.step == _BoomFn.blow_at:
            torch.empty(HUGE, dtype=torch.uint8, device=g.device)
        return g * 2


class Boom(nn.Module):
    """Cheap forward, a backward that needs a huge buffer at one step."""

    def forward(self, x):
        return _BoomFn.apply(x)


class BackwardNet(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True))
        self.mid = nn.Sequential(nn.Linear(d, d), Boom(), nn.ReLU(inplace=True))
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        return self.out(self.mid(self.stem(x)))


def _train(model, tracker, steps, batch=32, d=128):
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(steps):
        with tracker:
            loss = model(torch.randn(batch, d, device="cuda")).square().mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)


def test_forward_oom_names_the_instance_and_reraises():
    model = Net(blow_block=2, blow_at_step=4).cuda()
    out = io.StringIO()
    tracker = track_memory(model, stream=out)
    with pytest.raises(torch.OutOfMemoryError) as ei:
        _train(model, tracker, steps=6)
    rep = tracker.report
    assert rep is not None and ei.value.truestar_report is rep  # same report, original error type
    assert "Tried to allocate 1024.00 GiB" in str(ei.value)  # the original message is untouched
    assert rep.module == "blocks.2.mlp"  # the instance, not just the type: blocks.0/1/3.mlp are MLPs too
    assert rep.module_type == "MLP"
    assert rep.phase == "forward"
    assert rep.step == 4
    assert rep.module_source == f"{THIS}:{mark('mlp_forward')}"
    assert rep.user_line[0] == THIS and rep.user_line[1] == mark("huge_alloc")
    assert rep.failed_bytes == HUGE
    assert rep.module_chain == ["<root Net>", "blocks.2", "blocks.2.mlp"]
    # trend: three good steps before the failing one, all about the same size
    assert [s.step for s in rep.trend] == [1, 2, 3] and all(s.peak > 0 for s in rep.trend)
    assert "steady" in rep.trend_verdict and "spike" in rep.trend_verdict
    # memory holders at that moment: modules that already ran this step, and live tensors
    names = [n for n, _, _ in rep.top_modules]
    assert names and all(n.startswith(("blocks.0", "blocks.1", "blocks.2")) for n in names)
    assert any(label.startswith("param ") for label, *_ in rep.top_tensors)
    text = out.getvalue()
    assert "forward of blocks.2.mlp (MLP" in text and f"test_diag_memory.py:{mark('huge_alloc')}" in text
    assert rep.to_dict()["module"] == "blocks.2.mlp"
    # tracker removed its hooks on the way out
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_backward_oom_names_the_module_whose_backward_ran():
    _BoomFn.step, _BoomFn.blow_at = 0, 3
    model = BackwardNet().cuda()
    tracker = track_memory(model, verbose=False)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    with pytest.raises(torch.OutOfMemoryError):
        for step in range(1, 6):
            _BoomFn.step = step
            with tracker:
                loss = model(torch.randn(16, 64, device="cuda")).square().mean()
                loss.backward()  # MARK:backward_call
                opt.step()
                opt.zero_grad(set_to_none=True)
    rep = tracker.report
    assert rep.phase == "backward", rep.to_text()
    assert rep.module == "mid.1", rep.to_text()  # the Boom instance inside mid
    assert rep.module_type == "Boom"
    assert rep.step == 3
    assert [s.step for s in rep.trend] == [1, 2]


def test_oom_outside_modules_with_a_leak_trend():
    model = Net(d=64, n=2).cuda()
    tracker = track_memory(model, verbose=False)
    kept = []  # the classic leak: something from every step is kept alive
    with pytest.raises(torch.OutOfMemoryError):
        for step in range(1, 8):
            with tracker:
                loss = model(torch.randn(8, 64, device="cuda")).square().mean()
                loss.backward()
                kept.append(torch.empty(2 * 1024 ** 2, dtype=torch.uint8, device="cuda"))  # 2 MiB per step
                if step == 6:
                    torch.empty(HUGE, dtype=torch.uint8, device="cuda")  # MARK:outside_alloc
    rep = tracker.report
    del kept
    assert rep.phase == "outside modules" and rep.module is None
    assert rep.user_line[1] == mark("outside_alloc")
    assert "slow leak" in rep.trend_verdict, rep.trend_verdict
    assert "last event" in rep.context


def test_memory_timeline_for_a_step():
    model = Net(d=64, n=2).cuda()
    tracker = track_memory(model)
    _train(model, tracker, steps=2, d=64)
    tl = memory_timeline(tracker)
    assert tl.step == 2
    fwd = [e for e in tl.events if e.phase == "forward" and e.edge == "end"]
    bwd = [e for e in tl.events if e.phase == "backward" and e.edge == "end"]
    assert {e.module for e in fwd} >= {"blocks.0.mlp.up", "blocks.1.mlp.down", "head", "<root Net>"}
    assert any(e.module == "blocks.0.mlp" for e in bwd) and any(e.module == "head" for e in bwd)
    up = next(e for e in fwd if e.module == "blocks.0.mlp.up")
    assert up.out_shapes == "(32, 128) float32" and up.delta > 0
    assert tl.stats.peak >= tl.stats.start_allocated > 0
    text = tl.to_text()
    assert "blocks.0.mlp.up" in text and "backward" in text
    assert tracker.memory_timeline(1).step == 1
    assert tracker.report is None


def test_decorator_form_and_other_errors_pass_through():
    model = Net(d=64, n=2, blow_block=1, blow_at_step=2).cuda()
    tracker = track_memory(model, verbose=False)

    @tracker
    def step(x):
        model(x).sum().backward()

    step(torch.randn(4, 64, device="cuda"))
    with pytest.raises(torch.OutOfMemoryError) as ei:
        step(torch.randn(4, 64, device="cuda"))
    assert ei.value.truestar_report.module == "blocks.1.mlp" and ei.value.truestar_report.step == 2

    @tracker
    def bad(x):
        model(x)
        raise ValueError("not memory")

    with pytest.raises(ValueError):
        bad(torch.randn(4, 64, device="cuda"))
    assert tracker.report is None  # only OOMs produce a report
