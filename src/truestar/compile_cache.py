"""Persistent compile cache: what a new process needs to reach its first result without compiling,
tuning or verifying again (docs/COMPILER.md, Compile cache).

Everything lives under one directory, ``TRUESTAR_CACHE_DIR`` (default ``~/.cache/truestar/kernels``)::

    triton/k_<kernel key>.py       generated kernel source (kernel/compiler.py, keyed on the kernel IR)
    triton/tune/<tune key>.json    launch config and cuBLAS choices per shape bucket (kernel/tune.py)
    triton/bin/<hash>/...          Triton's compiled binaries: TRITON_CACHE_DIR points here unless the
                                   user set it
    numba/, metal/                 the CPU and Metal backends' generated sources
    programs/<key>.json            one record per verified first call (below)

A program record holds the RESULT of a program's first call verification: the bitwise digest of its
outputs, the kernel launches it checked (kernel key, shapes, dtypes, launch config), the largest end
to end and region errors and the conditioning notes. Its key covers everything that check depends on:
the IR program with its weights (``ir.serialize.program_hash``), the compile options, the input
values, shapes and dtypes of that call, the ``update`` spec, the verify mode and device rule, the
cuDNN and matmul precision flags, the Truestar sources of the compiler, kernel and IR packages (a
content hash, not the version number), Truestar's codegen and tuning versions, torch, Triton and
CUDA versions and the GPU. A later process makes the same call on the same inputs, runs it without
the checks and accepts the record only when its outputs are bitwise equal to the recorded ones AND
it launched exactly the recorded kernel binaries: same inputs, same binaries, same bits, so every
check would see what it saw. Any difference (other inputs, a retuned config, other outputs) runs the
full verification, as without the cache. A call on inputs not seen before is always verified.

Safe by construction: every write goes to a temporary file renamed over the target (atomic on one
file system); every record carries a format version and a sha256 of its content, and a record that
fails to parse, has another version or a wrong digest is ignored with a TS1351 warning and
rebuilt; the directory is capped (``TRUESTAR_CACHE_MAX_MB``, default 4096) by deleting the least
recently used entries (TS1352). ``TRUESTAR_COMPILE_CACHE=0`` turns the cross process reuse off
(program records, the Triton directory, tuned configs read from disk). ``truestar cache info`` and
``truestar cache clear`` show and empty it.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

FORMAT = 1
DEFAULT_MAX_MB = 4096


class CompileCacheWarning(UserWarning):
    """A cache entry was unreadable (ignored and rebuilt) or the cache was trimmed."""


_lock = threading.Lock()
_triton_installed: Optional[str] = None
_src_digest: Optional[str] = None
_hash_memo: Dict[int, Tuple[Any, tuple, str]] = {}
_trimmed = False


def enabled() -> bool:
    return os.environ.get("TRUESTAR_COMPILE_CACHE", "1").strip().lower() not in ("0", "off", "false", "no")


def root() -> Path:
    from .kernel.compiler import cache_dir
    return cache_dir()


def programs_dir() -> Path:
    return root() / "programs"


def triton_dir() -> Path:
    return root() / "triton" / "bin"


def max_bytes() -> int:
    try:
        return int(float(os.environ.get("TRUESTAR_CACHE_MAX_MB", DEFAULT_MAX_MB)) * 2 ** 20)
    except ValueError:
        return DEFAULT_MAX_MB * 2 ** 20


def install_triton_dir() -> Optional[str]:
    """Point Triton's binary cache into the Truestar cache, unless the user chose a directory
    (TRITON_CACHE_DIR set) or the cache is off. Process wide: Triton reads the variable when it
    compiles. Returns the directory Triton uses for Truestar kernels, or None for its default."""
    global _triton_installed
    if not enabled():
        return os.environ.get("TRITON_CACHE_DIR")
    want = str(triton_dir())
    cur = os.environ.get("TRITON_CACHE_DIR")
    if cur is not None and cur != _triton_installed:
        return cur                      # the user's choice
    if cur != want:                     # first use, or TRUESTAR_CACHE_DIR changed since
        os.makedirs(want, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = want
        _triton_installed = want
    return want


# ----------------------------------------------------------------------------- keys


def source_digest() -> str:
    """sha256 over the Truestar sources a compiled program and its checks depend on (the compiler,
    kernel and IR packages), so an edit of any of them never reuses a record made before it."""
    global _src_digest
    if _src_digest is None:
        base = Path(__file__).parent
        h = hashlib.sha256()
        for pkg in ("compiler", "kernel", "ir"):
            for p in sorted((base / pkg).rglob("*.py")):
                h.update(p.relative_to(base).as_posix().encode() + b"\0")
                h.update(p.read_bytes())
        h.update((base / "compile_cache.py").read_bytes())
        _src_digest = h.hexdigest()
    return _src_digest


def environment(device: str) -> Dict[str, Any]:
    import torch
    from . import __version__
    from .kernel import compiler as KC, tune
    env: Dict[str, Any] = {"format": FORMAT, "truestar": __version__, "sources": source_digest(),
                           "codegen": KC.CODEGEN_VERSION, "tune": tune.TUNE_VERSION,
                           "torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        import triton
        env["triton"] = triton.__version__
    except ImportError:
        env["triton"] = None
    if device in ("cuda", "gpu", "rocm", "hip") and torch.cuda.is_available():
        i = torch.cuda.current_device()
        env["gpu"] = [torch.cuda.get_device_name(i), list(torch.cuda.get_device_capability(i))]
    return env


def _identity(prog) -> tuple:
    """What must not have changed for a memoized hash to still hold: the node list and every
    weight tensor's storage and in place version counter."""
    data = getattr(prog, "param_data", {}) or {}
    return (len(prog.nodes), tuple((n, t.data_ptr(), t._version, tuple(t.shape), str(t.dtype))
                                   for n, t in sorted(data.items())))


def program_hash(prog) -> str:
    """ir.serialize.program_hash (structure and weights), memoized per Program object while its
    nodes and weights are unchanged (an in place edit of a weight bumps its version counter)."""
    hit = _hash_memo.get(id(prog))
    if hit is not None and hit[0] is prog and hit[1] == _identity(prog):
        return hit[2]
    from .ir.serialize import program_hash as ph
    h = ph(prog)
    remember_hash(prog, h)
    return h


def remember_hash(prog, h: str) -> None:
    """Record a program hash already computed (ir.serialize.loads checked it against the file)."""
    if len(_hash_memo) > 64:
        _hash_memo.clear()
    _hash_memo[id(prog)] = (prog, _identity(prog), h)


def tensor_digest(ts: Iterable[Any]) -> str:
    """sha256 of dtype, shape and the bytes of every tensor (moved to the host)."""
    import torch
    h = hashlib.sha256()
    for t in ts:
        if isinstance(t, torch.Tensor):
            t = t.detach()
            h.update(f"{t.dtype}|{tuple(t.shape)}|".encode())
            c = t.cpu().contiguous()
            if c.numel():
                h.update(memoryview(c.reshape(-1).view(torch.uint8).numpy()))
        else:
            h.update(repr(t).encode())
        h.update(b";")
    return h.hexdigest()


def _flags() -> Dict[str, Any]:
    import torch
    from .kernel import verify
    return {"verify_device": verify.verify_device(), "gpu_mb": os.environ.get("TRUESTAR_VERIFY_GPU_MB"),
            "cudnn": [torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark,
                      torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32],
            "matmul": str(torch.get_float32_matmul_precision()),
            "library": os.environ.get("TRUESTAR_LIBRARY"), "tune": os.environ.get("TRUESTAR_TUNE")}


def verify_key(cp, inputs, update: Dict[int, int], mode: str) -> Optional[str]:
    """The program record key of this first call, or None when it may not be cached."""
    o = cp.opts
    if not enabled() or o.hidden or o.checks:
        return None
    text = json.dumps({"env": environment(o.device), "program": program_hash(cp.program),
                       "opts": repr(o), "verify": mode, "update": sorted(update.items()),
                       "flags": _flags(), "inputs": tensor_digest(inputs),
                       "budget": [cp.memory_budget, cp.memory_dims, cp.recompute]},
                      sort_keys=True, default=repr)
    return hashlib.sha256(text.encode()).hexdigest()[:40]


# ----------------------------------------------------------------------------- records


def _digest(body: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _warn(msg: str) -> None:
    warnings.warn(f"[TS1351] truestar compile cache: {msg}; ignored, it is rebuilt", CompileCacheWarning,
                  stacklevel=4)


def load(key: str) -> Optional[Dict[str, Any]]:
    """The record for key, or None (missing; or unreadable: warned TS1351, counted, deleted)."""
    path = programs_dir() / f"{key}.json"
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        rec = json.loads(raw.decode("utf-8"))
        if not isinstance(rec, dict):
            raise ValueError("not a JSON object")
        if rec.get("format") != FORMAT:
            raise ValueError(f"format {rec.get('format')!r}, expected {FORMAT}")
        body = rec.get("body")
        if not isinstance(body, dict) or body.get("key") != key:
            raise ValueError("key does not match the file name")
        if rec.get("digest") != _digest(body):
            raise ValueError("digest mismatch (edited or truncated)")
    except (ValueError, UnicodeDecodeError) as e:
        stats.rejected += 1
        _warn(f"record {path} is unreadable ({e})")
        try:
            path.unlink()
        except OSError:
            pass
        return None
    try:
        os.utime(path)                  # least recently used order for trim()
    except OSError:
        pass
    return body


def store(key: str, body: Dict[str, Any]) -> None:
    d = programs_dir()
    d.mkdir(parents=True, exist_ok=True)
    body = dict(body, key=key, created=time.time())
    rec = {"format": FORMAT, "body": body, "digest": _digest(body)}
    path = d / f"{key}.json"
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    stats.stores += 1
    trim_once()


class Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.hits = self.misses = self.rejected = self.stores = self.mismatched = 0
        self.evicted = 0


stats = Stats()


# ----------------------------------------------------------------------------- size cap and CLI


def _entries(base: Path) -> List[Tuple[float, int, Path]]:
    """(last use, bytes, path) per evictable entry: every file outside triton/bin and every
    directory of Triton's (one compiled kernel each)."""
    out: List[Tuple[float, int, Path]] = []
    if not base.exists():
        return out
    tb = base / "triton" / "bin"
    for dirpath, dirnames, filenames in os.walk(base):
        p = Path(dirpath)
        if p == tb:
            for dn in dirnames:
                q = p / dn
                size, last = 0, 0.0
                for f in q.iterdir():
                    try:
                        s = f.stat()
                    except OSError:
                        continue
                    size += s.st_size
                    last = max(last, s.st_mtime, s.st_atime)
                out.append((last, size, q))
            dirnames[:] = []
            continue
        for fn in filenames:
            f = p / fn
            try:
                s = f.stat()
            except OSError:
                continue
            out.append((max(s.st_mtime, s.st_atime), s.st_size, f))
    return out


def info() -> Dict[str, Any]:
    base = root()
    ents = _entries(base)
    kinds: Dict[str, List[int]] = {}
    for _, size, p in ents:
        rel = p.relative_to(base).as_posix()
        k = ("programs" if rel.startswith("programs/") else "triton binaries" if rel.startswith("triton/bin/")
             else "tuned configs" if rel.startswith("triton/tune/") else "kernel sources")
        n = kinds.setdefault(k, [0, 0])
        n[0] += 1
        n[1] += size
    return {"dir": str(base), "enabled": enabled(), "triton_binaries": os.environ.get("TRITON_CACHE_DIR") or (str(triton_dir()) if enabled() else None),
            "bytes": sum(s for _, s, _ in ents), "max_bytes": max_bytes(), "format": FORMAT,
            "entries": {k: {"count": v[0], "bytes": v[1]} for k, v in sorted(kinds.items())}}


def _remove(p: Path) -> None:
    import shutil
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            p.unlink()
        except OSError:
            pass


def trim(limit: Optional[int] = None) -> int:
    """Delete least recently used entries until the cache is under 90% of its cap (when over the
    cap). Returns the number of entries deleted."""
    limit = max_bytes() if limit is None else limit
    ents = _entries(root())
    total = sum(s for _, s, _ in ents)
    if total <= limit:
        return 0
    n = 0
    for last, size, p in sorted(ents, key=lambda e: e[0]):
        if total <= 0.9 * limit:
            break
        _remove(p)
        total -= size
        n += 1
    stats.evicted += n
    warnings.warn(f"[TS1352] truestar compile cache over {limit / 2 ** 20:.0f} MiB: deleted the {n} least "
                  f"recently used entries", CompileCacheWarning, stacklevel=3)
    return n


def trim_once() -> None:
    """trim() at most once per process (a directory walk), after the first record is written."""
    global _trimmed
    with _lock:
        if _trimmed:
            return
        _trimmed = True
    trim()


def clear() -> int:
    """Delete the whole cache directory's content. Returns the number of entries deleted."""
    from .kernel import compiler as KC
    ents = _entries(root())
    for _, _, p in ents:
        _remove(p)
    KC.clear_memory_cache()
    return len(ents)
