"""WebNN as a `truestar check` backend, in a headless browser (research item 6.3, docs/WEBGPU.md "WebNN").

Backend names: ``webnn`` (the browser's default device), ``webnn:cpu``, ``webnn:gpu``, ``webnn:npu``
(the `deviceType` asked of `navigator.ml.createContext`).

The model is exported to ONNX (the shared path of :mod:`truestar.webgpu.onnx_check`), translated
node by node into WebNN operations by :mod:`truestar.webnn.translate` and run in Chrome or Edge with
WebNN enabled, headless: the browser is started by Playwright with ``headless=True`` and never shows
a window. The page is served from a routed https origin inside the browser (a secure context, which
WebNN requires); nothing goes over the network. The CPU control runs the same ONNX graph on ONNX
Runtime's CPU EP in this process.

Which native stack the browser used (Windows ML EPs, LiteRT, Core ML) is the browser's choice for the
device type; the page cannot read it, so the report names the device type and the browser version.

Environment: ``TRUESTAR_WEBNN_CHANNEL`` (``chrome`` default, or ``msedge``, or ``chromium`` for
Playwright's own build).
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..backends import Backend, BackendUnavailable

PREFIX = "webnn"
DEVICES = ("cpu", "gpu", "npu")
ORIGIN = "https://truestar-webnn.invalid/"
FLAG = "--enable-features=WebMachineLearningNeuralNetwork"
EP_NAME = "WebNN"  # what the profile reports for translated nodes

_BROWSERS: Dict[str, "BrowserWorker"] = {}
_PROBED: Dict[str, Tuple[Optional[Backend], str]] = {}


def parse_spec(name: str) -> str:
    """'webnn:gpu' -> 'gpu'; 'webnn' -> '' (the browser default)."""
    parts = name.strip().lower().split(":")
    if parts[0] != PREFIX or len(parts) > 2:
        raise ValueError(f"not a WebNN backend name: {name!r} (use webnn, webnn:cpu, webnn:gpu, webnn:npu)")
    if len(parts) == 2 and parts[1] not in DEVICES:
        raise ValueError(f"unknown WebNN device type {parts[1]!r}; use one of {', '.join(DEVICES)}")
    return parts[1] if len(parts) == 2 else ""


def _decode(r: Dict[str, Any]) -> np.ndarray:
    dt = {"float32": np.float32, "int32": np.int32, "int64": np.int64, "uint8": np.uint8,
          "int8": np.int8}[r["dtype"]]
    return np.frombuffer(base64.b64decode(r["data"]), dtype=dt).reshape(r["shape"]).copy()


class BrowserWorker:
    """Answers the shared `run` call: provider "webnn" in the page, "cpu" on ONNX Runtime."""

    ep_name = EP_NAME

    def __init__(self, device: str):
        self.device = device
        self.channel = os.environ.get("TRUESTAR_WEBNN_CHANNEL", "chrome")
        self.pw = self.browser = self.page = None
        self.info: Dict[str, Any] = {}

    def start(self) -> Dict[str, Any]:
        from playwright.sync_api import sync_playwright

        self.pw = sync_playwright().start()
        kw = {} if self.channel == "chromium" else {"channel": self.channel}
        # headless=True: Playwright starts the browser with --headless; no window is ever shown.
        self.browser = self.pw.chromium.launch(headless=True, args=[FLAG], **kw)
        self.page = self.browser.new_page()
        script = open(os.path.join(os.path.dirname(__file__), "page.js"), encoding="utf-8").read()
        self.page.route(ORIGIN + "**", lambda r: r.fulfill(
            status=200, content_type="text/html",
            body=f"<!doctype html><html><head><script>{script}</script></head><body>truestar</body></html>"))
        self.page.goto(ORIGIN)
        info = self.page.evaluate("d => window.truestarWebnn.info(d)", self.device)
        info["browser"] = f"{self.channel} {self.browser.version}"
        info["device_type"] = self.device or "default"
        self.info = info
        return info

    def log_tail(self, n: int = 0) -> str:
        return ""

    def call(self, op: str, payload: Any = None) -> Any:
        if op == "hello":
            return self.info
        if op != "run":
            raise RuntimeError(f"WebNN worker: unknown call {op}")
        feeds = {k: np.ascontiguousarray(v) for k, v in payload["feeds"].items()}
        names = payload.get("outputs")
        if payload.get("provider") == "cpu":
            import onnxruntime as ort

            sess = ort.InferenceSession(payload["model"], providers=["CPUExecutionProvider"])
            t0 = time.perf_counter()
            outs = sess.run(names, feeds)
            return {"outputs": outs, "output_names": [o.name for o in sess.get_outputs()] if names is None else list(names),
                    "providers": ["CPUExecutionProvider"], "fallback": False, "ep_error": "",
                    "nodes": None, "seconds": time.perf_counter() - t0, "repeat_equal": None}
        from .translate import translate

        desc = translate(payload["model"], feeds, names or ())
        enc = {k: base64.b64encode(v.astype(np.float32).tobytes()).decode("ascii") for k, v in feeds.items()}
        repeat = int(payload.get("repeat", 0))
        res = self.page.evaluate("([d, desc, f, r]) => window.truestarWebnn.run(d, desc, f, r)",
                                 [self.device, desc, enc, repeat])
        runs = [[_decode(o) for o in run] for run in res["runs"]]
        outs = runs[0]
        repeat_equal = None
        for again in runs[1:]:
            same = all(np.array_equal(a, b, equal_nan=True) for a, b in zip(outs, again))
            repeat_equal = same if repeat_equal is None else (repeat_equal and same)
        nodes = [(n["name"], n["type"], EP_NAME) for n in desc["nodes"]] if payload.get("profile") else None
        self.context_retries = getattr(self, "context_retries", 0) + int(res.get("contextRetries", 0))
        return {"outputs": outs, "output_names": list(desc["outputs"]), "providers": [EP_NAME],
                "fallback": False, "ep_error": "", "nodes": nodes, "seconds": res["ms"] / 1000.0,
                "repeat_equal": repeat_equal}

    def close(self) -> None:
        for obj in (self.browser, self.pw):
            try:
                if obj is self.pw and obj is not None:
                    obj.stop()
                elif obj is not None:
                    obj.close()
            except Exception:
                pass
        self.pw = self.browser = self.page = None


def get_worker(name: str) -> BrowserWorker:
    key = name.strip().lower()
    w = _BROWSERS.get(key)
    if w is not None and w.page is not None:
        return w
    w = BrowserWorker(parse_spec(name))
    _BROWSERS[key] = w
    w.start()
    return w


def close_browsers() -> None:
    for w in list(_BROWSERS.values()):
        w.close()
    _BROWSERS.clear()
    _PROBED.clear()


def probe(name: str) -> Tuple[Optional[Backend], str]:
    key = name.strip().lower()
    if key in _PROBED:
        return _PROBED[key]
    try:
        device = parse_spec(name)
    except ValueError as exc:
        return None, str(exc)
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
        import playwright  # noqa: F401
    except Exception as exc:
        return None, (f"{type(exc).__name__}: {exc}; the WebNN backend needs onnx, onnxruntime (the CPU "
                      "control) and playwright with Chrome or Edge installed")
    try:
        w = get_worker(name)
        info = w.info
    except Exception as exc:
        result = (None, f"the headless browser did not start: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        _PROBED[key] = result
        return result
    if not info.get("hasML"):
        result = (None, f"{info.get('browser')}: navigator.ml is missing (WebNN not enabled in this build)")
    elif not info.get("context"):
        result = (None, f"{info.get('browser')}: [TS1129] createContext(deviceType={device or 'default'}) "
                        f"failed: {info.get('error', '')}")
    else:
        desc = (f"WebNN in headless {info['browser']}, deviceType {device or 'default'} "
                f"(the browser picks the native stack)")
        result = (Backend(name=key, device="webnn", description=desc, kind="webnn", info=dict(info),
                          notes=["WebNN has no float64; the graph is translated node by node by "
                                 "Truestar and run in float32"]), "")
    if result[0] is None:
        get_worker(name).close()
        _BROWSERS.pop(key, None)
    _PROBED[key] = result
    return result


def resolve(name: str) -> Backend:
    b, reason = probe(name)
    if b is None:
        raise BackendUnavailable(f"Requested backend not available on this machine: {name}. Reason: {reason}", [name])
    return b


_SCRIPT = '''"""Repro generated by truestar check for the WebNN backend.

{what}
Backend:  {backend} ({description})
Expected: relative error below {threshold:.1e} versus {expected_from}.

Run:  python {script}       (WebNN in a headless browser through truestar.webnn; no window opens)
Exit code 1 means the divergence reproduced.
"""
import os
import sys

import numpy as np
import truestar.webnn as W

HERE = os.path.dirname(os.path.abspath(__file__))
w = W.get_worker({backend!r})
feeds = dict(np.load(os.path.join(HERE, "{inputs}")))
expected = np.load(os.path.join(HERE, "{expected}"))
outs = w.call("run", {{"model": open(os.path.join(HERE, "{onnx}"), "rb").read(), "feeds": feeds,
                      "provider": "webnn"}})["outputs"]
W.close_browsers()
worst = 0.0
for i, got in enumerate(outs):
    ref = expected["out%d" % i].astype(np.float64)
    got = got.astype(np.float64)
    den = np.linalg.norm(ref)
    rel = np.linalg.norm(got - ref) / den if den else float(np.linalg.norm(got - ref))
    worst = max(worst, rel)
    print("output", i, "relative error %.3e" % rel)
sys.exit(1 if worst > {threshold!r} else 0)
'''


def write_repro(folder, stem, backend, data, feeds, expected, threshold, what, expected_from) -> str:
    os.makedirs(folder, exist_ok=True)
    onnx_name, in_name, exp_name, script = (f"{stem}.onnx", f"{stem}_inputs.npz", f"{stem}_expected.npz", f"{stem}.py")
    with open(os.path.join(folder, onnx_name), "wb") as f:
        f.write(data)
    np.savez(os.path.join(folder, in_name), **feeds)
    np.savez(os.path.join(folder, exp_name), **{f"out{i}": np.asarray(e, dtype=np.float64) for i, e in enumerate(expected)})
    path = os.path.join(folder, script)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_SCRIPT.format(what=what, backend=backend.name, description=backend.description,
                               threshold=threshold, expected_from=expected_from, script=script,
                               onnx=onnx_name, inputs=in_name, expected=exp_name))
    return path


def check_backend(res, base, args, kwargs, dtype, ref_calls, ref_flat, noise, op_counting,
                  repro_dir, model_name, drill_ops: bool = True) -> None:
    """Fill a check.BackendResult for a WebNN backend: refuse untranslatable graphs (TS1127), then
    the shared ONNX path."""
    import torch

    from ..compare import SUSPECT
    from ..webgpu.onnx_check import EpTarget, ExportFailed, check_backend as onnx_check_backend, export
    from .translate import Unsupported, translate

    b = res.backend
    if dtype != torch.float32:
        res.status = SUSPECT
        res.add("TS1127", f"the WebNN backend runs float32 graphs only; this check is in {dtype}")
        return
    worker = get_worker(b.name)
    try:
        data, feeds, _, _ = export(base, args, kwargs, dtype)
        translate(data, feeds)
    except Unsupported as exc:
        res.status = SUSPECT
        res.add("TS1127", f"nothing ran on {b.name}: {exc}")
        return
    except ExportFailed:
        pass  # the shared path reports TS562
    target = EpTarget("webnn", EP_NAME, f"WebNN ({b.info.get('device_type')})", fallback_code="TS1129",
                      cpu_nodes_code="TS1127", nondeterministic_code="TS1128", repro=write_repro)
    worker.context_retries = 0
    onnx_check_backend(res, base, args, kwargs, dtype, ref_calls, ref_flat, noise, op_counting,
                       repro_dir, model_name, drill_ops, target=target, worker=worker)
    if worker.context_retries:
        res.notes.append(f"the browser lost the WebNN context {worker.context_retries} time(s); each graph "
                         "was rerun once on a new context")
