"""Say which rank, which collective and which source line caused a distributed hang.

A NCCL or gloo hang shows up as every rank waiting until a watchdog timeout, with a message that
names an op and a sequence number on the rank that noticed, not the rank that caused it. The usual
causes are ranks calling collectives in a different order, one rank skipping a collective inside a
data dependent ``if``, and one rank crashing while the others wait for it.

:class:`CollectiveRecorder` wraps the ``torch.distributed`` collectives on every rank and appends one
record per call (sequence number, op, shapes, dtype, process group and the Python call site) to a
per rank file in a shared directory. A hung collective cannot be used to talk to the other ranks,
so files are the channel. A watchdog thread notices when a collective has made no progress for
``timeout`` seconds, asks the other ranks to dump where they are, reads every rank's recent records
and prints a :class:`Diagnosis`: which rank is behind, the first sequence number where ranks
disagree on op, shape or call site, and the source line where the divergence started.

    with record_collectives(timeout=60):   # on every rank, after init_process_group
        train()

The directory defaults to ``$TRUESTAR_COLLECTIVE_DIR`` or a folder under the system temp dir; for
jobs spanning several machines it must be on a file system every node can see.
"""

from __future__ import annotations

import inspect
import json
import linecache
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

_TORCH_DIR = os.path.dirname(torch.__file__)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

WRAPPED = [
    "all_reduce", "all_gather", "all_gather_into_tensor", "broadcast", "reduce", "reduce_scatter",
    "reduce_scatter_tensor", "barrier", "gather", "scatter", "all_to_all", "all_to_all_single",
    "all_gather_object", "broadcast_object_list", "gather_object", "scatter_object_list",
]
STALL_MARKER = "STALL"
HANG_EXIT_CODE = 86

# Directories of libraries that issue collectives on the user's behalf (truestar.train runs the
# gradient all_reduce from inside the IR interpreter). Their frames are skipped like torch's, so a
# record names the user's line that started the step.
LIBRARY_DIRS: List[str] = []


def skip_frames_from(directory: str) -> None:
    """Treat frames under ``directory`` as library frames when recording call sites."""
    d = os.path.abspath(directory)
    if d not in LIBRARY_DIRS:
        LIBRARY_DIRS.append(d)


def _user_site(frame) -> Tuple[str, int, str, str]:
    f = frame
    while f is not None:
        path = os.path.abspath(f.f_code.co_filename)
        if not (path.startswith(_TORCH_DIR) or path.startswith(_THIS_DIR)
                or any(path.startswith(d) for d in LIBRARY_DIRS)):
            return path, f.f_lineno, f.f_code.co_name, linecache.getline(path, f.f_lineno).strip()
        f = f.f_back
    return "<unknown>", 0, "", ""


def _user_stack(frame, limit: int = 8) -> List[str]:
    out = []
    for fs in traceback.extract_stack(frame)[::-1]:
        path = os.path.abspath(fs.filename)
        if path.startswith(_THIS_DIR) or "threading.py" in path:
            continue
        out.append(f"{path}:{fs.lineno} in {fs.name}: {(fs.line or '').strip()}")
        if len(out) >= limit:
            break
    return out


def _tensors(obj: Any, out: List[torch.Tensor]) -> None:
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            _tensors(o, out)


def _group_info(group) -> Tuple[str, List[int]]:
    if group is None or group is dist.GroupMember.WORLD:
        return "world", list(range(dist.get_world_size()))
    try:
        ranks = list(dist.get_process_group_ranks(group))
    except Exception:
        ranks = []
    return "ranks " + ",".join(map(str, ranks)), ranks


# ----------------------------------------------------------------------------- recorder


class CollectiveRecorder:
    """Records collectives on this rank and diagnoses stalls. Use one per rank, as a context manager.

    Args:
        directory: shared folder for the per rank files (created if missing).
        timeout: seconds a collective may make no progress before the watchdog diagnoses a hang.
        last_n: how many recent collectives per rank the diagnosis shows.
        on_hang: ``"exit"`` ends the process with ``exit_code`` after printing (a hung collective
            cannot be cancelled), ``"report"`` prints and keeps waiting.
        settle: seconds to wait after asking other ranks to dump their state before reading it.
    """

    def __init__(self, directory: Optional[str] = None, timeout: float = 300.0, last_n: int = 10,
                 on_hang: str = "exit", exit_code: int = HANG_EXIT_CODE, settle: float = 1.0,
                 stream=None, max_records: int = 20000):
        if on_hang not in ("exit", "report"):
            raise ValueError("on_hang must be 'exit' or 'report'")
        self.directory = directory or os.environ.get("TRUESTAR_COLLECTIVE_DIR") or os.path.join(
            tempfile.gettempdir(), "truestar_collectives", os.environ.get("MASTER_PORT", "default"))
        self.timeout = timeout
        self.last_n = last_n
        self.on_hang = on_hang
        self.exit_code = exit_code
        self.settle = settle
        self.stream = stream
        self.max_records = max_records
        self.diagnosis: Optional[Diagnosis] = None
        self.rank = -1
        self._lock = threading.RLock()
        self._local = threading.local()
        self._originals: Dict[Tuple[Any, str], Any] = {}
        self._seq = 0
        self._gseq: Dict[str, int] = {}
        self._inflight: Optional[Tuple[int, float]] = None
        self._recent: Deque[str] = deque(maxlen=max(last_n * 8, 128))
        self._lines = 0
        self._reported = False
        self._dumped = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._collective_error: Optional[BaseException] = None

    # -- files
    @property
    def path(self) -> str:
        return os.path.join(self.directory, f"rank{self.rank}.jsonl")

    def _write(self, rec: Dict[str, Any]) -> None:
        rec.setdefault("t", time.time())
        line = json.dumps(rec)
        with self._lock:
            self._lines += 1
            if rec["kind"] in ("call", "done"):
                self._recent.append(line)
            if self._lines > self.max_records:  # rotate: keep the header and the recent calls
                self._fh.close()
                with open(self.path, "w", encoding="utf-8") as f:
                    f.write(self._header + "\n" + "\n".join(self._recent) + "\n")
                self._fh = open(self.path, "a", encoding="utf-8")
                self._lines = len(self._recent) + 1
                if rec["kind"] in ("call", "done"):
                    return
            self._fh.write(line + "\n")
            self._fh.flush()

    # -- wrapping
    def _wrap(self, name: str, orig):
        sig = inspect.signature(orig)
        rec = self

        def wrapper(*args, **kwargs):
            if getattr(rec._local, "depth", 0) > 0:  # nested call inside another collective
                return orig(*args, **kwargs)
            try:
                bound = sig.bind_partial(*args, **kwargs).arguments
            except TypeError:
                bound = {}
            group_key, group_ranks = _group_info(bound.get("group"))
            ts: List[torch.Tensor] = []
            for k, v in bound.items():
                if k not in ("group", "op", "async_op"):
                    _tensors(v, ts)
            extra = {}
            for k in ("src", "dst", "group_src", "group_dst"):
                if bound.get(k) is not None:
                    extra[k] = bound[k]
            if bound.get("op") is not None:
                extra["op"] = str(bound["op"]).split(".")[-1]
            if bound.get("object_list") is not None and not ts:
                extra["n_objects"] = len(bound["object_list"])
            site = _user_site(sys._getframe(1))
            with rec._lock:
                seq = rec._seq
                rec._seq += 1
                gseq = rec._gseq.get(group_key, 0)
                rec._gseq[group_key] = gseq + 1
                is_async = bool(bound.get("async_op"))
                rec._write({
                    "kind": "call", "seq": seq, "gseq": gseq, "op": name,
                    "shapes": [list(t.shape) for t in ts[:8]],
                    "dtype": str(ts[0].dtype).replace("torch.", "") if ts else None,
                    "group": group_key, "group_ranks": group_ranks, "extra": extra,
                    "file": site[0], "line": site[1], "func": site[2], "code": site[3],
                    "async": is_async,
                })
                if not is_async:
                    rec._inflight = (seq, time.time())
            rec._local.depth = 1
            try:
                out = orig(*args, **kwargs)
            except BaseException as e:
                rec._local.depth = 0
                rec._inflight = None
                rec._collective_error = e
                rec._write({"kind": "error", "seq": seq, "error": f"{type(e).__name__}: {str(e)[:300]}"})
                rec._trigger(f"collective seq {seq} ({name}) raised {type(e).__name__}", exit_process=False)
                raise
            rec._local.depth = 0
            with rec._lock:
                rec._inflight = None
                rec._write({"kind": "done", "seq": seq})
            return out

        wrapper.__wrapped__ = orig
        wrapper.__name__ = name
        return wrapper

    def _patch(self) -> None:
        import torch.distributed.distributed_c10d as c10d
        for name in WRAPPED:
            orig = getattr(c10d, name, None)
            if orig is None:
                continue
            w = self._wrap(name, orig)
            for mod in (dist, c10d):
                if getattr(mod, name, None) is orig:
                    self._originals[(mod, name)] = orig
                    setattr(mod, name, w)

    def _unpatch(self) -> None:
        for (mod, name), orig in self._originals.items():
            setattr(mod, name, orig)
        self._originals = {}

    # -- watchdog
    def _watch(self) -> None:
        marker = os.path.join(self.directory, STALL_MARKER)
        while not self._stop.wait(0.1):
            inflight = self._inflight
            if inflight is not None and not self._reported and time.time() - inflight[1] > self.timeout:
                self._trigger(f"collective seq {inflight[0]} made no progress for {self.timeout:g} s",
                              exit_process=self.on_hang == "exit")
            if not self._dumped and os.path.exists(marker):
                self._dump_stack()

    def _dump_stack(self) -> None:
        self._dumped = True
        frame = sys._current_frames().get(self._main_ident)
        stack = _user_stack(frame) if frame is not None else []
        inflight = self._inflight
        self._write({"kind": "stack", "stack": stack, "inflight": inflight[0] if inflight else None})

    def _trigger(self, reason: str, exit_process: bool) -> None:
        with self._lock:
            if self._reported:
                return
            self._reported = True
        self._write({"kind": "stall", "reason": reason,
                     "seq": self._inflight[0] if self._inflight else None})
        try:
            open(os.path.join(self.directory, STALL_MARKER), "a").close()
        except OSError:
            pass
        if not self._dumped:
            self._dump_stack()
        time.sleep(self.settle)
        d = diagnose(self.directory, last_n=self.last_n)
        d.reporter = self.rank
        d.reason = reason
        self.diagnosis = d
        text = d.to_text()
        try:
            with open(os.path.join(self.directory, f"diagnosis_rank{self.rank}.txt"), "w", encoding="utf-8") as f:
                f.write(text + "\n")
            with open(os.path.join(self.directory, f"diagnosis_rank{self.rank}.json"), "w", encoding="utf-8") as f:
                json.dump(d.to_dict(), f, indent=1)
        except OSError:
            pass
        # Print once per job: the lowest rank that saw the stall prints, the rest only write files.
        reporters = [r for r, st in d.states.items() if st.stalled]
        if not reporters or self.rank == min(reporters):
            print(text, file=self.stream or sys.stderr, flush=True)
        if exit_process:
            try:
                sys.stderr.flush()
                sys.stdout.flush()
            finally:
                os._exit(self.exit_code)

    # -- context manager
    def __enter__(self) -> "CollectiveRecorder":
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("truestar collectives: call torch.distributed.init_process_group first")
        self.rank = dist.get_rank()
        os.makedirs(self.directory, exist_ok=True)
        if self.rank == 0:
            try:
                os.remove(os.path.join(self.directory, STALL_MARKER))
            except OSError:
                pass
        for stale in (f"diagnosis_rank{self.rank}.txt", f"diagnosis_rank{self.rank}.json"):
            try:
                os.remove(os.path.join(self.directory, stale))
            except OSError:
                pass
        self._header = json.dumps({"kind": "start", "rank": self.rank, "world": dist.get_world_size(),
                                   "pid": os.getpid(), "host": socket.gethostname(), "t": time.time(),
                                   "backend": str(dist.get_backend())})
        self._fh = open(self.path, "w", encoding="utf-8")
        self._fh.write(self._header + "\n")
        self._fh.flush()
        self._lines = 1
        self._main_ident = threading.get_ident()
        self._patch()
        self._stop.clear()
        self._thread = threading.Thread(target=self._watch, name="truestar-collective-watchdog", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc is not None and exc is not self._collective_error:
                site = ("<unknown>", 0, "", "")
                frames = list(traceback.walk_tb(tb))
                for frame, lineno in frames[::-1]:
                    path = os.path.abspath(frame.f_code.co_filename)
                    if not (path.startswith(_TORCH_DIR) or path.startswith(_THIS_DIR)):
                        site = (path, lineno, frame.f_code.co_name, linecache.getline(path, lineno).strip())
                        break
                self._write({"kind": "crash", "after_seq": self._seq - 1,
                             "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                             "file": site[0], "line": site[1], "func": site[2], "code": site[3]})
            elif exc is None:
                self._write({"kind": "finished", "calls": self._seq})
        finally:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2)
            self._unpatch()
            self._fh.close()
        return False


def record_collectives(**kw) -> CollectiveRecorder:
    """Create a :class:`CollectiveRecorder` (see its arguments)."""
    return CollectiveRecorder(**kw)


# ----------------------------------------------------------------------------- diagnosis


@dataclass
class RankState:
    rank: int
    header: Optional[Dict[str, Any]] = None
    calls: List[Dict[str, Any]] = field(default_factory=list)
    done: set = field(default_factory=set)
    crash: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    stalled: bool = False
    finished: bool = False
    stack: List[str] = field(default_factory=list)

    @property
    def last_call(self) -> Optional[Dict[str, Any]]:
        return self.calls[-1] if self.calls else None

    @property
    def waiting_in(self) -> Optional[Dict[str, Any]]:
        c = self.last_call
        if c is not None and c["seq"] not in self.done and not c.get("async"):
            return c
        return None

    def pid_alive(self) -> Optional[bool]:
        if not self.header or self.header.get("host") != socket.gethostname():
            return None
        pid = self.header.get("pid")
        try:
            if sys.platform == "win32":
                import ctypes
                h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
                if not h:
                    return False
                code = ctypes.c_ulong()
                ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
                ctypes.windll.kernel32.CloseHandle(h)
                return code.value == 259  # STILL_ACTIVE
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _sig(c: Dict[str, Any]) -> Tuple:
    return (c["op"], json.dumps(c["shapes"]), c["dtype"], json.dumps(c.get("extra", {}), sort_keys=True))


def _full(c: Dict[str, Any]) -> Tuple:
    return _sig(c) + (c["file"], c["line"])


def describe(c: Optional[Dict[str, Any]], site: bool = True) -> str:
    if c is None:
        return "nothing"
    shapes = " ".join(str(tuple(s)) for s in c["shapes"]) or "no tensors"
    extra = "".join(f" {k}={v}" for k, v in c.get("extra", {}).items())
    s = f"{c['op']} {shapes} {c['dtype'] or ''}{extra}".rstrip()
    if site:
        s += f" at {os.path.basename(c['file'])}:{c['line']}"
    return s


def _ranks(rs) -> str:
    rs = sorted(rs)
    return ("rank " if len(rs) == 1 else "ranks ") + ", ".join(map(str, rs))


@dataclass
class Diagnosis:
    kind: str                         # crash, skip, extra, mismatch, behind, stall, ok
    rank: Optional[int]               # the rank blamed
    seq: Optional[int]                # sequence number (in the process group) where it went wrong
    file: Optional[str]               # source file of the divergence or crash
    line: Optional[int]
    message: str
    group: str = "world"
    behind: List[int] = field(default_factory=list)
    world: int = 0
    states: Dict[int, RankState] = field(default_factory=dict)
    last_n: int = 10
    reporter: Optional[int] = None
    reason: str = ""

    def to_text(self) -> str:
        head = f"truestar collectives: hang diagnosis ({self.world} ranks"
        if self.reporter is not None:
            head += f", written by rank {self.reporter}"
        head += ")"
        lines = [head]
        if self.reason:
            lines.append(f"  trigger: {self.reason}")
        lines.append(f"  verdict: {self.message}")
        if self.file:
            lines.append(f"  divergence starts at: {self.file}:{self.line}")
            code = linecache.getline(self.file, self.line or 0).strip()
            if code:
                lines.append(f"      {code}")
        counts = {r: len(st.calls) for r, st in self.states.items()}
        if counts and len(set(counts.values())) > 1:
            lines.append("  collectives entered: " + ", ".join(f"rank {r}: {n}" for r, n in sorted(counts.items())))
        if self.behind:
            lines.append(f"  behind: {_ranks(self.behind)}")
        for r, st in sorted(self.states.items()):
            w = st.waiting_in
            state = ("crashed" if st.crash else "finished" if st.finished else
                     f"waiting in seq {w['gseq']} {describe(w)}" if w is not None else
                     "running, not in a collective")
            lines.append(f"  rank {r}: {state}")
            if st.crash:
                lines.append(f"      {st.crash['error']} at {st.crash['file']}:{st.crash['line']}")
            if st.stack and w is None and not st.finished and not st.crash:
                lines.append(f"      now at {st.stack[0]}")
        lines.append(f"  last {self.last_n} collectives per rank:")
        for r, st in sorted(self.states.items()):
            lines.append(f"    rank {r}:")
            for c in st.calls[-self.last_n:]:
                status = "done" if c["seq"] in st.done else ("async" if c.get("async") else "WAITING")
                lines.append(f"      seq {c['gseq']:>4} [{c['group']}] {describe(c)}  {status}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "rank": self.rank, "seq": self.seq, "file": self.file, "line": self.line,
            "message": self.message, "group": self.group, "behind": self.behind, "world": self.world,
            "reporter": self.reporter, "reason": self.reason,
            "ranks": {str(r): {"calls": st.calls[-self.last_n:], "done": sorted(st.done & {c["seq"] for c in st.calls[-self.last_n:]}),
                               "crash": st.crash, "finished": st.finished, "stalled": st.stalled,
                               "stack": st.stack} for r, st in self.states.items()},
        }


def load_states(directory: str) -> Tuple[Dict[int, RankState], int]:
    states: Dict[int, RankState] = {}
    world = 0
    for fn in os.listdir(directory):
        if not (fn.startswith("rank") and fn.endswith(".jsonl")):
            continue
        try:
            r = int(fn[4:-6])
        except ValueError:
            continue
        st = RankState(r)
        with open(os.path.join(directory, fn), encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # a line cut off mid write
                k = rec.get("kind")
                if k == "start":
                    st.header = rec
                    world = max(world, rec.get("world", 0))
                elif k == "call":
                    st.calls.append(rec)
                elif k == "done":
                    st.done.add(rec["seq"])
                elif k == "crash":
                    st.crash = rec
                elif k == "error":
                    st.error = rec
                elif k == "stall":
                    st.stalled = True
                elif k == "finished":
                    st.finished = True
                elif k == "stack":
                    st.stack = rec.get("stack", [])
        st.calls.sort(key=lambda c: c["seq"])
        states[r] = st
    return states, max(world, len(states))


def diagnose(directory: str, last_n: int = 10) -> Diagnosis:
    """Read every rank's records in ``directory`` and explain the hang."""
    states, world = load_states(directory)
    base = dict(states=states, world=world, last_n=last_n)
    for r in range(world):
        if r not in states:
            states[r] = RankState(r)

    # 1. A rank that crashed outside a collective is the cause; the others wait for it.
    crashed = [r for r, st in states.items() if st.crash]
    if crashed:
        r = min(crashed, key=lambda r: states[r].crash["t"])
        c = states[r].crash
        waiting = {o: st.waiting_in for o, st in states.items() if st.waiting_in is not None}
        wseq = None
        wtxt = ""
        if waiting:
            w = next(iter(waiting.values()))
            wseq = w["gseq"]
            wtxt = f"; {_ranks(waiting)} wait in seq {wseq} {describe(w)}, which rank {r} never reached"
        return Diagnosis("crash", r, wseq, c["file"], c["line"],
                         f"rank {r} crashed with {c['error']} at {os.path.basename(c['file'])}:{c['line']} "
                         f"after {c['after_seq'] + 1} collectives{wtxt}.", behind=[r], **base)

    # 2. First sequence number, per process group, where the members disagree.
    groups: Dict[str, List[int]] = {}
    for st in states.values():
        for c in st.calls:
            groups.setdefault(c["group"], c["group_ranks"] or list(range(world)))
    weak: Optional[Diagnosis] = None
    for gkey, members in groups.items():
        seqs = {r: [c for c in states[r].calls if c["group"] == gkey] for r in members if r in states}
        if not seqs:
            continue
        # Records may be trimmed to a recent window; align on group sequence numbers.
        by_g = {r: {c["gseq"]: c for c in cs} for r, cs in seqs.items()}
        lo = max((min(d) for d in by_g.values() if d), default=0)
        hi = max((max(d) for d in by_g.values() if d), default=-1)
        for k in range(lo, hi + 1):
            present = {r: d[k] for r, d in by_g.items() if k in d}
            if len(present) < len(by_g):
                break  # some rank has not reached k: handled below as behind
            if len(Counter(_full(c) for c in present.values())) == 1:
                continue
            d = _explain_disagreement(gkey, k, present, by_g)
            if d.kind != "site":
                d.states, d.world, d.last_n = states, world, last_n
                return d
            if weak is None or (d.seq or 0) < (weak.seq or 0):
                weak = d
            # A call site difference alone can be benign (same collective written twice); keep looking.
    if weak is not None:
        weak.kind = "mismatch"
        weak.states, weak.world, weak.last_n = states, world, last_n
        return weak

    # 3. Everyone agrees so far: someone is behind or stuck outside collectives.
    counts = {r: (st.calls[-1]["seq"] + 1 if st.calls else 0) for r, st in states.items()}
    if counts and len(set(counts.values())) > 1:
        lo = min(counts.values())
        behind = sorted(r for r, n in counts.items() if n == lo)
        ahead = next(st for r, st in states.items() if counts[r] > lo)
        nxt = next((c for c in ahead.calls if c["seq"] == lo), None)
        parts = []
        for r in behind:
            st = states[r]
            alive = st.pid_alive()
            where = (f"; it is now at {st.stack[0]}" if st.stack else "")
            if st.finished:
                parts.append(f"rank {r} finished after {lo} collectives")
            elif alive is False:
                parts.append(f"rank {r} died without a Python error (killed or native crash) after {lo} collectives")
            else:
                parts.append(f"rank {r} is still running but has not called collective seq {lo}{where}")
        msg = (f"{'; '.join(parts)}. The other ranks wait in seq {nxt['gseq']} {describe(nxt)}."
               if nxt else "; ".join(parts) + ".")
        r0 = behind[0]
        last = states[r0].last_call
        return Diagnosis("behind", r0, nxt["gseq"] if nxt else lo,
                         nxt["file"] if nxt else (last["file"] if last else None),
                         nxt["line"] if nxt else (last["line"] if last else None),
                         msg, behind=behind, **base)
    waiting = {r: st.waiting_in for r, st in states.items() if st.waiting_in is not None}
    if waiting:
        w = next(iter(waiting.values()))
        return Diagnosis("stall", None, w["gseq"], w["file"], w["line"],
                         f"all ranks agree on every collective so far; {_ranks(waiting)} wait in seq "
                         f"{w['gseq']} {describe(w)} with no disagreement recorded (slow rank, network or "
                         f"a collective outside the recorded API, for example DDP's C++ reducer).", **base)
    return Diagnosis("ok", None, None, None, None, "no disagreement and no stalled collective recorded.", **base)


def _explain_disagreement(gkey, k, present, by_g) -> Diagnosis:
    groups: Dict[Tuple, List[int]] = {}
    for r, c in present.items():
        groups.setdefault(_full(c), []).append(r)
    sizes = sorted(len(v) for v in groups.values())
    tie = len(sizes) > 1 and sizes[-1] == sizes[-2]
    # Try the order, skip and extra explanations, smallest group of ranks first.
    for key, rs in sorted(groups.items(), key=lambda kv: (len(kv[1]), -min(kv[1]))):
        others = sorted(o for o in present if o not in rs)
        if not others or not all(_full(present[x]) == _full(present[others[0]]) for x in others):
            continue
        mine, theirs = by_g[rs[0]], by_g[others[0]]
        nxt_mine = mine.get(k + 1)
        nxt_theirs = theirs.get(k + 1)
        if (nxt_mine and nxt_theirs and _full(nxt_mine) == _full(theirs[k])
                and _full(mine[k]) == _full(nxt_theirs)):
            c = mine[k]
            return Diagnosis("mismatch", rs[0], k, c["file"], c["line"],
                             f"collectives are called in a different order: at seq {k} {_ranks(rs)} called "
                             f"{describe(c)} then {describe(nxt_mine)}, while {_ranks(others)} called them "
                             f"the other way round.", group=gkey)
        if nxt_theirs is not None and _full(mine[k]) == _full(nxt_theirs):
            skipped = theirs[k]
            return Diagnosis("skip", rs[0], k, skipped["file"], skipped["line"],
                             f"{_ranks(rs)} skipped collective seq {k} {describe(skipped)} that "
                             f"{_ranks(others)} called; {_ranks(rs)} went on to {describe(mine[k])}, "
                             f"so every later collective is paired with the wrong one. Look for a data "
                             f"dependent branch around {os.path.basename(skipped['file'])}:{skipped['line']}.",
                             group=gkey, behind=sorted(rs))
        if nxt_mine is not None and _full(nxt_mine) == _full(theirs[k]):
            extra = mine[k]
            return Diagnosis("extra", rs[0], k, extra["file"], extra["line"],
                             f"{_ranks(rs)} made an extra collective at seq {k}: {describe(extra)} that "
                             f"{_ranks(others)} never call; they called {describe(theirs[k])}. Look for a "
                             f"data dependent branch around {os.path.basename(extra['file'])}:{extra['line']}.",
                             group=gkey, behind=sorted(others))
    # Plain mismatch: blame the minority (on an even split, the side without the lowest rank).
    if tie:
        major_ranks = groups[_full(present[min(present)])]
    else:
        major_ranks = max(groups.values(), key=len)
    minority = sorted(r for r in present if r not in major_ranks)
    r = minority[0]
    mc, oc = present[r], present[major_ranks[0]]
    same_sig = _sig(mc) == _sig(oc)
    what = "call site" if same_sig else ("op" if mc["op"] != oc["op"] else "shape, dtype or arguments")
    msg = (f"ranks disagree on the {what} of collective seq {k}: {_ranks(minority)} called {describe(mc)}, "
           f"{_ranks(major_ranks)} called {describe(oc)}.")
    if not same_sig:
        msg += f" Either {_ranks(minority)} skipped a collective or the ranks call them in a different order."
    else:
        msg += " Same op and shapes from different lines can be harmless; check both branches."
    if tie:
        msg += " (Even split: blame shown for the side without the lowest rank.)"
    return Diagnosis("site" if same_sig else "mismatch", r, k, mc["file"], mc["line"], msg, group=gkey)
