"""Real ranks: one process per device, joined by torch.distributed (NCCL on Linux GPUs, gloo otherwise).

`DeviceComm` has the interface of the simulated communicators in comm.py (`all_reduce`,
`all_gather`, `reduce_scatter`, each logged with its purpose), so the layout runtime of
`truestar parallel` runs unchanged on real GPUs, plus asynchronous bucket reductions for the
gradient sync (gradsync.py).

Two reduction orders:

* `order="fixed"` (the default): every element of a sum over ranks is the left to right sum of the
  ranks' values in rank order, the same on every rank. The result is bitwise the same on every rank,
  from run to run, and equal to the sum one device computes when it adds the same pieces in rank
  order. Two ways to get it (`fixed_algo`), bitwise equal to each other:
  - `"scatter"` (the default): cut the flat tensor into `world` equal pieces, an all to all hands
    rank r piece r of every rank, rank r adds those in rank order, and an all gather returns the
    summed pieces. Each rank sends and receives 2 (world - 1) / world times the tensor, the bytes of
    a ring all reduce, and adds only its own piece.
  - `"gather"`: an all gather of the whole tensor and the rank order sum of all of it on every rank;
    world - 1 times the tensor on the wire and world - 1 full additions. Kept for comparison.
* `order="native"`: the backend's own all reduce (NCCL picks ring or tree by size and topology).
  For 2 ranks every element is one addition `a + b`, which is commutative in IEEE arithmetic, so it
  equals the fixed order bitwise; for 3 or more ranks the association depends on the algorithm and
  is not the rank order sum.

With gloo and CUDA tensors every collective goes through a host copy (gloo's CUDA support is partial
and Windows has no NCCL); `staged` in the log says so.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import socket
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .comm import Comm, check_even_split


class RendezvousError(RuntimeError):
    """The ranks could not join (TS822): a port that does not answer, a firewall, a wrong address."""


class DevicesUnavailable(RuntimeError):
    """Fewer devices than ranks, or the requested backend is missing (TS821)."""


@dataclasses.dataclass
class RankInfo:
    rank: int
    world: int
    device: str
    backend: str
    device_name: str
    host: str

    def text(self) -> str:
        return f"rank {self.rank} on {self.host} {self.device} ({self.device_name}, {self.backend})"


def pick_backend(device: str, backend: str = "auto") -> str:
    import torch.distributed as dist

    if backend not in ("auto", "nccl", "gloo"):
        raise DevicesUnavailable(f"unknown process group backend {backend!r}; use auto, nccl or gloo")
    if backend == "auto":
        return "nccl" if device.startswith("cuda") and dist.is_nccl_available() else "gloo"
    if backend == "nccl" and not dist.is_nccl_available():
        raise DevicesUnavailable("backend nccl is not available in this torch build (Windows has none); use gloo")
    if backend == "gloo" and not dist.is_gloo_available():
        raise DevicesUnavailable("backend gloo is not available in this torch build")
    return backend


def device_name(device: str) -> str:
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(torch.device(device))
    return "cpu"


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def route_ip(host: str) -> Optional[str]:
    """This machine's address on the route to `host` (no packet is sent: a UDP connect only picks it)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((socket.gethostbyname(host), 9))
            return s.getsockname()[0]
    except OSError:
        return None


def pin_gloo_interface(master_host: str) -> Optional[str]:
    """Point gloo at the interface that reaches rank 0. Gloo otherwise binds the first address the host
    name resolves to, which on a machine with a WSL or VPN adapter is often one the other machine cannot
    reach (measured: the desktop's name resolved to 172.30.32.1, its WSL adapter, first, and the job
    hung until the timeout). Sets GLOO_SOCKET_IFNAME unless the user set it; needs psutil to map the
    address to an interface name, else leaves gloo's choice alone. Returns the interface used."""
    if os.environ.get("GLOO_SOCKET_IFNAME") or master_host in ("127.0.0.1", "localhost"):
        return os.environ.get("GLOO_SOCKET_IFNAME")
    ip = route_ip(master_host)
    try:
        import psutil
    except ImportError:
        return None
    for name, addrs in psutil.net_if_addrs().items():
        if ip and any(a.family == socket.AF_INET and a.address == ip for a in addrs):
            os.environ["GLOO_SOCKET_IFNAME"] = name
            return name
    return None


def join(rank: int, world: int, init_method: str, *, device: str = "cpu", backend: str = "auto",
         timeout: float = 120.0) -> RankInfo:
    """init_process_group for this process. `device` is "cpu", "cuda" (this process's local GPU) or
    "cuda:N". A rendezvous that does not complete raises RendezvousError naming the address."""
    import torch.distributed as dist

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise DevicesUnavailable(f"rank {rank}: device {device} requested but CUDA is not available here")
        if device == "cuda":
            device = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.set_device(torch.device(device))
    be = pick_backend(device, backend)
    if be == "gloo" and init_method.startswith("tcp://"):
        pin_gloo_interface(init_method[len("tcp://"):].rsplit(":", 1)[0].strip("[]"))
    kw = {}
    if be == "nccl":
        kw["device_id"] = torch.device(device)
    try:
        dist.init_process_group(be, init_method=init_method, rank=rank, world_size=world,
                                timeout=datetime.timedelta(seconds=timeout), **kw)
    except Exception as exc:  # noqa: BLE001 - every rendezvous failure is reported the same way
        raise RendezvousError(f"rank {rank} of {world} could not join {init_method} ({type(exc).__name__}: {exc}). "
                              "Check that rank 0's address is reachable from this machine and that its port is "
                              "open in the firewall; with gloo across machines, GLOO_SOCKET_IFNAME names the interface that reaches "
                              f"rank 0 (now {os.environ.get('GLOO_SOCKET_IFNAME') or 'unset'})") from None
    return RankInfo(rank, world, device, be, device_name(device), socket.gethostname())


def make_groups(layout) -> Dict[Tuple[int, ...], object]:
    """Every tp group and every dp group, created in the same order on every rank (new_group is
    collective). A group of one rank needs no process group."""
    import torch.distributed as dist

    groups: Dict[Tuple[int, ...], object] = {}
    for d in range(layout.dp):
        g = tuple(d * layout.tp + j for j in range(layout.tp))
        groups[g] = dist.new_group(list(g)) if len(g) > 1 else None
    for t in range(layout.tp):
        g = tuple(i * layout.tp + t for i in range(layout.dp))
        if g not in groups:
            groups[g] = dist.new_group(list(g)) if len(g) > 1 else None
    return groups


class Pending:
    """An asynchronous bucket reduction; `wait()` returns the reduced tensor on the caller's device."""

    def __init__(self, comm: "DeviceComm", work, out: torch.Tensor, parts: Optional[List[torch.Tensor]],
                 device: torch.device, stream=None) -> None:
        self.comm, self.work, self.out, self.parts, self.device = comm, work, out, parts, device
        self.stream = stream

    def wait(self) -> torch.Tensor:
        if self.work is not None:
            self.work.wait()
        if self.stream is not None:
            cur = torch.cuda.current_stream(self.out.device)
            cur.wait_stream(self.stream)
            self.out.record_stream(cur)
        if self.parts is not None:
            res = self.comm._fold(self.parts)
        else:
            res = self.out
        return res.to(self.device, non_blocking=False)


class DeviceComm(Comm):
    def __init__(self, rank: int, world: int, device: str, backend: str, groups: Dict[Tuple[int, ...], object],
                 order: str = "fixed", fixed_algo: str = "scatter") -> None:
        super().__init__()
        if order not in ("fixed", "native"):
            raise ValueError(f"order {order!r}: fixed or native")
        if fixed_algo not in ("scatter", "gather"):
            raise ValueError(f"fixed_algo {fixed_algo!r}: scatter or gather")
        self.rank, self.world, self.groups, self.order, self.backend = rank, world, groups, order, backend
        self.fixed_algo = fixed_algo
        self._side = None
        self.device = torch.device(device)
        self.staged = backend == "gloo" and self.device.type == "cuda"
        self.wire = torch.device("cpu") if (backend == "gloo") else self.device

    def _pg(self, group: Sequence[int]):
        return self.groups[tuple(group)]

    def _on_wire(self, t: torch.Tensor) -> torch.Tensor:
        return t.detach().to(self.wire).contiguous().clone()

    @staticmethod
    def _fold(parts: List[torch.Tensor]) -> torch.Tensor:
        acc = parts[0].clone()
        for p in parts[1:]:
            acc = acc + p
        return acc

    def _fixed_scatter(self, x: torch.Tensor, group: Sequence[int]) -> torch.Tensor:
        """Rank order sum of `x` over `group`: all to all, a fold of this rank's piece, all gather.

        Element e lands in piece e // c (c = ceil(numel / n)); member j of the group receives piece j
        of every member in group order and adds them left to right, the same additions the gather fold
        makes for that element, so both give the same bits. A tail that does not fill the last piece
        is padded with zeros and cut off again (the padded entries are never read)."""
        import torch.distributed as dist

        n, pg = len(group), self._pg(group)
        flat = x.reshape(-1)
        size = flat.numel()
        c = -(-size // n)
        if c * n != size:
            flat = torch.cat([flat, flat.new_zeros(c * n - size)])
        recv = torch.empty_like(flat)
        dist.all_to_all_single(recv, flat, group=pg)
        rows = recv.view(n, c)
        mine = rows[0] + rows[1]
        for j in range(2, n):
            mine = mine + rows[j]
        out = torch.empty_like(flat)
        dist.all_gather_into_tensor(out, mine, group=pg)
        return (out[:size] if c * n != size else out).view(x.shape)

    def _side_stream(self):
        if self._side is None:
            self._side = torch.cuda.Stream(device=self.device)
        return self._side

    def _note(self, op, purpose, group, t):
        super()._note(op + (" (staged through the host)" if self.staged else ""), purpose, group, t)

    def all_reduce(self, t, group, purpose):
        import torch.distributed as dist

        self._note("all_reduce", purpose, group, t)
        if len(group) > 1 and self.order == "fixed" and self.fixed_algo == "scatter":
            # reads its input only, so no defensive copy
            return self._fixed_scatter(t.detach().to(self.wire).contiguous(), group).to(t.device)
        x = self._on_wire(t)
        if len(group) > 1:
            if self.order == "fixed":
                bufs = [torch.empty_like(x) for _ in group]
                dist.all_gather(bufs, x, group=self._pg(group))
                x = self._fold(bufs)
            else:
                dist.all_reduce(x, group=self._pg(group))
        return x.to(t.device)

    def all_gather(self, t, group, dim, purpose):
        import torch.distributed as dist

        self._note("all_gather", purpose, group, t)
        x = self._on_wire(t)
        if len(group) == 1:
            return x.to(t.device)
        bufs = [torch.empty_like(x) for _ in group]
        dist.all_gather(bufs, x, group=self._pg(group))
        return torch.cat(bufs, dim=dim).to(t.device)

    def reduce_scatter(self, t, group, dim, purpose):
        import torch.distributed as dist

        n = len(group)
        idx = list(group).index(self.rank)
        check_even_split(t, group, dim, purpose)
        if n == 1:
            self._note("reduce_scatter", purpose, group, t)
            return t.detach().clone()
        if self.order == "fixed" and self.fixed_algo == "scatter":
            # the scatter half of the fixed sum: piece j of the flat tensor (dim moved first) is
            # chunk j along dim, so this rank's folded piece is its reduce scatter result
            import torch.distributed as dist

            self._note("reduce_scatter", purpose, group, t)
            x = t.detach().movedim(dim, 0).contiguous().to(self.wire)
            recv = torch.empty_like(x)
            dist.all_to_all_single(recv, x, group=self._pg(group))
            rows = recv.view(n, -1)
            mine = rows[0] + rows[1]
            for j in range(2, n):
                mine = mine + rows[j]
            mine = mine.view((x.shape[0] // n,) + tuple(x.shape[1:]))
            return mine.movedim(0, dim).contiguous().to(t.device)
        if self.order == "fixed" or self.backend != "nccl":
            # fixed: every rank sums all pieces in rank order; gloo has no reduce scatter of its own
            full = self.all_reduce(t, group, purpose)
            return full.chunk(n, dim=dim)[idx].contiguous()
        self._note("reduce_scatter", purpose, group, t)
        x = t.detach().movedim(dim, 0).contiguous().to(self.wire)
        out = torch.empty((x.shape[0] // n,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
        dist.reduce_scatter_tensor(out, x, group=self._pg(group))
        return out.movedim(0, dim).contiguous().to(t.device)

    def all_reduce_start(self, t: torch.Tensor, group: Sequence[int], purpose: str, *,
                         donate: bool = False) -> Pending:
        """Start the sum of `t` over `group` without waiting (a gradient bucket). `donate=True`: the
        caller hands `t` over (a bucket buffer it leaves alone until `wait()`), so a reduction on the
        wire device works on it directly instead of on a copy."""
        import torch.distributed as dist

        self._note("all_reduce (async)", purpose, group, t)
        x = t.detach() if (donate and t.device == self.wire and t.is_contiguous()) else self._on_wire(t)
        if len(group) == 1:
            return Pending(self, None, x, None, t.device)
        if self.order == "fixed" and self.fixed_algo == "scatter":
            if x.device.type == "cuda":
                # all to all, fold and all gather on a side stream: the fold waits for the all to all
                # there, not on the stream the backward pass is still computing on
                side, cur = self._side_stream(), torch.cuda.current_stream(x.device)
                side.wait_stream(cur)
                with torch.cuda.stream(side):
                    res = self._fixed_scatter(x, group)
                x.record_stream(side)
                return Pending(self, None, res, None, t.device, stream=side)
            return Pending(self, None, self._fixed_scatter(x, group), None, t.device)
        if self.order == "fixed":
            bufs = [torch.empty_like(x) for _ in group]
            work = dist.all_gather(bufs, x, group=self._pg(group), async_op=True)
            return Pending(self, work, x, bufs, t.device)
        work = dist.all_reduce(x, group=self._pg(group), async_op=True)
        return Pending(self, work, x, None, t.device)

    def gather_objects(self, obj, dst: int = 0):
        """Every rank's picklable `obj` on rank `dst` (a list in rank order), None elsewhere."""
        import torch.distributed as dist

        out = [None] * self.world if self.rank == dst else None
        dist.gather_object(obj, out, dst=dst)
        return out

    def broadcast_object(self, obj, src: int = 0):
        import torch.distributed as dist

        box = [obj]
        dist.broadcast_object_list(box, src=src)
        return box[0]


def leave() -> None:
    import torch.distributed as dist

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def local_devices(world: int, device: str) -> List[str]:
    """One device per local rank: "cuda" gives cuda:0..world-1 and needs that many GPUs; "cpu" gives
    cpu for every rank."""
    if device == "cpu":
        return ["cpu"] * world
    if not torch.cuda.is_available():
        raise DevicesUnavailable(f"{world} GPU ranks requested but CUDA is not available on this machine")
    n = torch.cuda.device_count()
    if n < world:
        raise DevicesUnavailable(f"{world} GPU ranks need {world} GPUs; this machine has {n} "
                                 f"({', '.join(torch.cuda.get_device_name(i) for i in range(n))}). Two ranks never "
                                 "share one GPU here, because that would not be a multi GPU run")
    return [f"cuda:{i}" for i in range(world)]


__all__ = ["DeviceComm", "Pending", "RankInfo", "RendezvousError", "DevicesUnavailable", "join", "leave",
           "make_groups", "pick_backend", "free_port", "local_devices", "device_name"]
