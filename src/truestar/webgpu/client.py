"""Start and talk to a WebGPU worker process (see :mod:`truestar.webgpu.worker`)."""

from __future__ import annotations

import atexit
import json
import os
import pickle
import struct
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class WorkerDied(RuntimeError):
    """The worker process exited (a Dawn device loss or a driver crash ends it)."""


def worker_python() -> str:
    """The Python that hosts the worker: TRUESTAR_WEBGPU_PYTHON, else this one."""
    return os.environ.get("TRUESTAR_WEBGPU_PYTHON") or sys.executable


class WorkerClient:
    def __init__(self, backend_type: str, adapter_index: int,
                 ep_options: Optional[Dict[str, str]] = None, timeout: float = 600.0,
                 module: str = "truestar.webgpu.worker", python: Optional[str] = None,
                 label: str = "WebGPU"):
        # module, python, label: the same protocol hosts other workers (truestar.winml)
        self.label = label
        self.backend_type = backend_type
        self.adapter_index = adapter_index
        self.ep_options = dict(ep_options or {})
        self.timeout = timeout
        fd, self.log_path = tempfile.mkstemp(prefix="truestar_webgpu_", suffix=".log")
        os.close(fd)
        self._log = open(self.log_path, "wb")
        env = dict(os.environ)
        env["PYTHONPATH"] = _SRC + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["TRUESTAR_WEBGPU_LOG"] = self.log_path
        env.setdefault("PYTHONIOENCODING", "utf-8")
        self.proc = subprocess.Popen(
            [python or worker_python(), "-m", module, backend_type, str(adapter_index),
             json.dumps(self.ep_options)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, env=env)
        self.info: Dict[str, Any] = {}
        atexit.register(self.close)

    def call(self, op: str, payload: Any = None) -> Any:
        if self.proc.poll() is not None:
            raise WorkerDied(self._died_message())
        data = pickle.dumps((op, payload), protocol=pickle.HIGHEST_PROTOCOL)
        try:
            self.proc.stdin.write(struct.pack("<Q", len(data)))
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
            head = self.proc.stdout.read(8)
            if len(head) < 8:
                raise WorkerDied(self._died_message())
            (n,) = struct.unpack("<Q", head)
            status, result = pickle.loads(self.proc.stdout.read(n))
        except (BrokenPipeError, OSError) as exc:
            raise WorkerDied(self._died_message() + f" ({exc})") from exc
        if status != "ok":
            raise RuntimeError(f"{self.label} worker: {result}")
        return result

    def _died_message(self) -> str:
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass
        tail = self.log_tail()
        return (f"the {self.label} worker process exited with code {self.proc.returncode} "
                f"(a device loss or driver crash ends it). Log tail: {tail}")

    def log_tail(self, n: int = 1500) -> str:
        try:
            self._log.flush()
            with open(self.log_path, "rb") as f:
                text = f.read().decode("utf-8", "replace")
        except OSError:
            return ""
        lines = [ln for ln in text.splitlines()
                 if "loader_get_json" not in ln and "AgilitySDK" not in ln]
        return "\n".join(lines)[-n:]

    def close(self) -> None:
        if getattr(self, "proc", None) is None:
            return
        try:
            if self.proc.poll() is None:
                data = pickle.dumps(("close", None))
                self.proc.stdin.write(struct.pack("<Q", len(data)) + data)
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        for s in (self.proc.stdin, self.proc.stdout):
            try:
                s.close()
            except Exception:
                pass
        try:
            self._log.close()
            os.remove(self.log_path)
        except OSError:
            pass
        self.proc = None
