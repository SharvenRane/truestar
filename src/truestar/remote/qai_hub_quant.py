"""INT8 agreement and per layer blame on phone NPUs through Qualcomm AI Hub.

Two studies, both built from the same cached steps:

* `int8_agreement_study`: quantize a model to w8a8 with AI Hub's quantize job, compile it for TFLite,
  run every input on the NPU and the CPU of one or more hosted phones, and count per input top 1
  agreement between every pair of runs (and the float64 reference), next to aggregate accuracy.
* `layer_blame_study`: wrap the model so chosen intermediate tensors are extra outputs, run it on
  the NPU and on the CPU of the same phone, and name the first tap where the NPU leaves the CPU
  control by more than a float16 floor (float models) or one INT8 step (quantized models).

Cost control. All inputs of a run go in one inference job (AI Hub takes a list of samples), input
sets are uploaded once as datasets, and every compile, quantize, upload and inference is cached in a
JSON file keyed by a fingerprint of the model (class, structure, every parameter and buffer), the
input specs, the device and the options. A rerun with nothing changed submits zero jobs; inference
outputs are also kept on disk as .npz so it needs no network at all.

The qai_hub client is passed in (or imported on first use), so tests drive all of this with a fake.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .agreement import (LayerBlame, PairAgreement, RunLogits, agreement_matrix, agreement_text,
                        layer_blame)

# Extra outputs per model. Each tap is one more tensor per input to download and one more
# point where the compiler must materialize a tensor it might otherwise fuse away, so keep it small.
MAX_TAPS = 12
CACHE_VERSION = 1
DEFAULT_DEVICES = ("Samsung Galaxy S24 (Family)", "Samsung Galaxy S25 (Family)")


# ----------------------------------------------------------------------------- fingerprints


def _h() -> "hashlib._Hash":
    return hashlib.sha256()


def model_fingerprint(model: nn.Module, specs: Dict[str, Tuple[Tuple[int, ...], str]],
                      extra: str = "") -> str:
    """Stable hash of what the compiler sees: structure, every tensor's bytes, the input specs."""
    h = _h()
    h.update(f"v{CACHE_VERSION}|{type(model).__module__}.{type(model).__qualname__}|".encode())
    h.update(repr(model).encode())
    for k, v in sorted(model.state_dict().items()):
        t = v.detach().cpu().contiguous()
        h.update(f"|{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.float().numpy().tobytes())
    h.update(json.dumps(sorted((k, list(s), d) for k, (s, d) in specs.items())).encode())
    h.update(extra.encode())
    return h.hexdigest()[:24]


def data_fingerprint(entries: Dict[str, List[np.ndarray]]) -> str:
    h = _h()
    for name in sorted(entries):
        h.update(f"|{name}|{len(entries[name])}|".encode())
        for a in entries[name]:
            a = np.ascontiguousarray(a)
            h.update(f"{a.dtype}{a.shape}".encode())
            h.update(a.tobytes())
    return h.hexdigest()[:24]


def cache_key(*parts: Any) -> str:
    h = _h()
    h.update("|".join(str(p) for p in parts).encode())
    return h.hexdigest()[:24]


# ----------------------------------------------------------------------------- cache


class HubCache:
    """Job ids and outputs from earlier runs, so reruns submit nothing.

    Layout: `<dir>/hub_cache.json` maps a key to the ids AI Hub returned; `<dir>/outputs/<key>.npz`
    holds downloaded inference outputs.
    """

    def __init__(self, directory: Optional[str] = None):
        directory = directory or os.environ.get("TRUESTAR_CACHE") or os.path.join(
            os.path.expanduser("~"), ".cache", "truestar", "qai_hub")
        self.dir = directory
        os.makedirs(os.path.join(self.dir, "outputs"), exist_ok=True)
        self.path = os.path.join(self.dir, "hub_cache.json")
        self.data: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.data.get(key)

    def put(self, key: str, value: Dict[str, Any]) -> None:
        self.data[key] = value
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    def drop(self, key: str) -> None:
        if key in self.data:
            del self.data[key]
            self.put("__last_drop__", {"key": key})

    def output_path(self, key: str) -> str:
        return os.path.join(self.dir, "outputs", f"{key}.npz")


# ----------------------------------------------------------------------------- taps


class TapWrapper(nn.Module):
    """Returns (model output, tap_1, ..., tap_k) where each tap is a named module's output.

    Taps are cloned in the hook so an in place op later in the model cannot overwrite them. The
    hook stores the tensor on the module that produced it, so deep copies of the wrapper (made for
    tracing and for the float64 reference) capture into their own modules.
    """

    def __init__(self, model: nn.Module, names: Sequence[str]):
        super().__init__()
        names = list(names)
        if len(names) > MAX_TAPS:
            raise ValueError(f"{len(names)} taps requested; at most {MAX_TAPS} (MAX_TAPS)")
        mods = dict(model.named_modules())
        missing = [n for n in names if n not in mods]
        if missing:
            raise ValueError(f"no module named {missing}")
        self.model = model
        self.names = names
        for n in set(names):
            mods[n].register_forward_hook(_tap_hook)

    def forward(self, *args):
        mods = dict(self.model.named_modules())
        for n in self.names:
            mods[n]._truestar_tap = None
        out = self.model(*args)
        taps = [mods[n]._truestar_tap for n in self.names]
        missing = [n for n, t in zip(self.names, taps) if t is None]
        if missing:
            raise RuntimeError(f"taps never ran: {missing}")
        return (out, *taps)


def _tap_hook(mod: nn.Module, _inp, out) -> None:
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"tap {type(mod).__name__} returns {type(out).__name__}, not a tensor")
    mod._truestar_tap = out.clone()


_SKIP_TOP = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
             nn.LayerNorm, nn.GroupNorm, nn.Dropout)


def default_taps(model: nn.Module, args: Sequence[torch.Tensor], max_taps: int = MAX_TAPS) -> List[str]:
    """Block level taps in forward order: children of top level containers and top level modules
    that are not a bare conv or norm (tapping those would stop the compiler folding norm into conv).
    The last module (the head, already the model output) is left out. Evenly thinned to `max_taps`.
    """
    cands: List[Tuple[str, nn.Module]] = []
    for name, child in model.named_children():
        if isinstance(child, (nn.Sequential, nn.ModuleList)):
            cands += [(f"{name}.{n}", c) for n, c in child.named_children()]
        elif not isinstance(child, _SKIP_TOP):
            cands.append((name, child))
    order: List[str] = []
    hooks = []
    for n, m in cands:
        hooks.append(m.register_forward_hook(
            lambda _m, _i, out, n=n: order.append(n) if isinstance(out, torch.Tensor) and n not in order else None))
    try:
        with torch.no_grad():
            model.eval()(*args)
    finally:
        for h in hooks:
            h.remove()
    order = order[:-1]
    if len(order) > max_taps:
        idx = np.unique(np.round(np.linspace(0, len(order) - 1, max_taps)).astype(int))
        order = [order[i] for i in idx]
    return order


# ----------------------------------------------------------------------------- output handling


def _natural(name: str) -> Tuple:
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name))


def outputs_in_order(out: Dict[str, List[np.ndarray]]) -> List[np.ndarray]:
    """AI Hub output dict -> one array per output, samples concatenated on axis 0.

    Names are sorted naturally ("output_2" before "output_10"); a plain sort would scramble taps.
    """
    res = []
    for name in sorted(out, key=_natural):
        parts = [np.asarray(a) for a in out[name]]
        res.append(np.concatenate(parts, axis=0) if parts[0].ndim else np.stack(parts))
    return res


def align(arr: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Match a runtime output to the reference shape, undoing a channel last layout if needed."""
    arr = np.asarray(arr)
    if tuple(arr.shape) == tuple(shape):
        return arr
    if arr.ndim == 4 and len(shape) == 4 and tuple(np.transpose(arr, (0, 3, 1, 2)).shape) == tuple(shape):
        return np.transpose(arr, (0, 3, 1, 2))
    if arr.size == int(np.prod(shape)):
        return arr.reshape(shape)
    raise ValueError(f"output shape {arr.shape} does not match reference {tuple(shape)}")


def match_outputs(got: List[np.ndarray], ref_shapes: List[Tuple[int, ...]]) -> List[np.ndarray]:
    """Pair runtime outputs with reference outputs: in order when every shape fits, else greedily
    by shape in order (runtimes sometimes reorder outputs)."""
    if len(got) != len(ref_shapes):
        raise ValueError(f"{len(got)} outputs, expected {len(ref_shapes)}")
    try:
        return [align(g, s) for g, s in zip(got, ref_shapes)]
    except ValueError:
        pass
    left = list(range(len(got)))
    res = []
    for s in ref_shapes:
        for i in left:
            try:
                res.append(align(got[i], s))
                left.remove(i)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"no output fits shape {s}; got {[g.shape for g in got]}")
    return res


# ----------------------------------------------------------------------------- session


def _id(obj: Any, *names: str) -> str:
    for n in names:
        v = getattr(obj, n, None)
        if v:
            return str(v)
    raise AttributeError(f"{type(obj).__name__} has none of {names}")


@dataclass
class JobLog:
    submitted: List[Tuple[str, str]] = field(default_factory=list)  # (kind, job id)
    reused: List[Tuple[str, str]] = field(default_factory=list)


class HubSession:
    """Cached AI Hub steps. Every method returns the same thing on a rerun without a new job."""

    def __init__(self, hub: Any = None, cache: Optional[HubCache] = None, cache_dir: Optional[str] = None):
        if hub is None:
            import qai_hub as hub  # noqa: F811  imported here so tests and the ml env never need it
        self.hub = hub
        self.cache = cache or HubCache(cache_dir)
        self.log = JobLog()

    # datasets -------------------------------------------------------------
    def dataset(self, entries: Dict[str, List[np.ndarray]], name: str = "truestar") -> Tuple[Any, str]:
        """(dataset, data fingerprint). Uploaded once per distinct content."""
        fp = data_fingerprint(entries)
        key = cache_key("dataset", fp)
        hit = self.cache.get(key)
        if hit:
            try:
                ds = self.hub.get_dataset(hit["dataset_id"])
                self.log.reused.append(("dataset", hit["dataset_id"]))
                return ds, fp
            except Exception:  # noqa: BLE001  expired or deleted: upload again
                pass
        ds = self.hub.upload_dataset(entries, name=name)
        self.cache.put(key, {"dataset_id": _id(ds, "dataset_id", "id")})
        self.log.submitted.append(("dataset", _id(ds, "dataset_id", "id")))
        return ds, fp

    # compile --------------------------------------------------------------
    def compile(self, source: Any, source_key: str, device: str, options: str,
                input_specs: Optional[Dict] = None, name: Optional[str] = None) -> Any:
        key = cache_key("compile", source_key, device, options)
        hit = self.cache.get(key)
        if hit:
            self.log.reused.append(("compile", hit["job_id"]))
            return self._model(hit)
        kw = {"model": source, "device": self.hub.Device(device), "options": options}
        if input_specs is not None:
            kw["input_specs"] = input_specs
        if name:
            kw["name"] = name
        job = self.hub.submit_compile_job(**kw)
        self.log.submitted.append(("compile", job.job_id))
        target = job.get_target_model()
        if target is None:
            raise RuntimeError(f"compile job {job.job_id} failed: {_status(job)}")
        self.cache.put(key, {"job_id": job.job_id, "model_id": _id(target, "model_id", "id")})
        return target

    def quantize(self, onnx_model: Any, calib: Any, calib_fp: str, name: Optional[str] = None) -> Any:
        mid = _id(onnx_model, "model_id", "id")
        key = cache_key("quantize", mid, calib_fp, "w8a8")
        hit = self.cache.get(key)
        if hit:
            self.log.reused.append(("quantize", hit["job_id"]))
            return self._model(hit)
        job = self.hub.submit_quantize_job(
            model=onnx_model, calibration_data=calib, weights_dtype=self.hub.QuantizeDtype.INT8,
            activations_dtype=self.hub.QuantizeDtype.INT8, name=name)
        self.log.submitted.append(("quantize", job.job_id))
        target = job.get_target_model()
        if target is None:
            raise RuntimeError(f"quantize job {job.job_id} failed: {_status(job)}")
        self.cache.put(key, {"job_id": job.job_id, "model_id": _id(target, "model_id", "id")})
        return target

    def _model(self, hit: Dict[str, Any]) -> Any:
        return self.hub.get_model(hit["model_id"])

    def job_id_for(self, kind: str, *parts: Any) -> Optional[str]:
        hit = self.cache.get(cache_key(kind, *parts))
        return hit["job_id"] if hit else None

    # inference ------------------------------------------------------------
    def infer(self, target: Any, device: str, dataset: Any, data_fp: str, options: str,
              name: Optional[str] = None) -> Tuple[Dict[str, List[np.ndarray]], str]:
        """(outputs as {name: [array per sample]}, job id)."""
        mid = _id(target, "model_id", "id")
        key = cache_key("infer", mid, device, data_fp, options)
        hit = self.cache.get(key)
        path = self.cache.output_path(key)
        if hit and os.path.exists(path):
            self.log.reused.append(("inference", hit["job_id"]))
            return _load_outputs(path), hit["job_id"]
        if hit:
            job = self.hub.get_job(hit["job_id"])
            self.log.reused.append(("inference", hit["job_id"]))
        else:
            kw = {"model": target, "device": self.hub.Device(device), "inputs": dataset, "options": options}
            if name:
                kw["name"] = name
            job = self.hub.submit_inference_job(**kw)
            self.log.submitted.append(("inference", job.job_id))
            self.cache.put(key, {"job_id": job.job_id})
        out = job.download_output_data()
        if out is None:
            self.cache.drop(key)
            raise RuntimeError(f"inference job {job.job_id} failed: {_status(job)}")
        _save_outputs(path, out)
        return {k: [np.asarray(a) for a in v] for k, v in out.items()}, job.job_id


def _status(job: Any) -> str:
    try:
        return str(job.get_status().message)
    except Exception:  # noqa: BLE001
        return "unknown status"


def _save_outputs(path: str, out: Dict[str, List[np.ndarray]]) -> None:
    flat = {}
    for k, v in out.items():
        for i, a in enumerate(v):
            flat[f"{k}::{i:06d}"] = np.asarray(a)
    np.savez_compressed(path, **flat)


def _load_outputs(path: str) -> Dict[str, List[np.ndarray]]:
    res: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    with np.load(path) as z:
        for k in z.files:
            name, i = k.rsplit("::", 1)
            res.setdefault(name, []).append((int(i), z[k]))
    return {k: [a for _, a in sorted(v, key=lambda t: t[0])] for k, v in res.items()}


# ----------------------------------------------------------------------------- shared prep


def _specs(args: Sequence[torch.Tensor]) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    return {f"input_{i}": ((1,) + tuple(a.shape[1:]), "float32") for i, a in enumerate(args)}


def _entries(x: torch.Tensor, name: str = "input_0") -> Dict[str, List[np.ndarray]]:
    """One sample per entry, batch dimension 1: AI Hub runs each list element separately."""
    arr = x.detach().cpu().float().numpy()
    return {name: [np.ascontiguousarray(arr[i:i + 1]) for i in range(arr.shape[0])]}


def reference_outputs(model: nn.Module, x: torch.Tensor) -> List[np.ndarray]:
    """float64 CPU reference, one array per model output, all samples stacked."""
    m = copy.deepcopy(model).cpu().double().eval()
    with torch.no_grad():
        out = m(x.double())
    outs = out if isinstance(out, (tuple, list)) else (out,)
    return [o.numpy() for o in outs]


def _trace(model: nn.Module, x: torch.Tensor):
    base = copy.deepcopy(model).cpu().float().eval()
    with torch.no_grad():
        return base, torch.jit.trace(base, (x[:1].float(),), check_trace=False)


def onnx_and_quantized(sess: HubSession, model: nn.Module, calib_x: torch.Tensor, device: str,
                       tag: str = "") -> Tuple[Any, Any, Dict[str, str]]:
    """(float ONNX model, w8a8 QDQ ONNX model, job ids). Cached."""
    base, traced = _trace(model, calib_x)
    specs = _specs([calib_x])
    fp = model_fingerprint(base, specs, tag)
    onnx_model = sess.compile(traced, fp, device, "--target_runtime onnx", input_specs=specs,
                              name=f"truestar-onnx{tag}")
    calib_entries = _entries(calib_x)
    calib, calib_fp = sess.dataset(calib_entries, name="truestar-calib")
    q = sess.quantize(onnx_model, calib, calib_fp, name=f"truestar-w8a8{tag}")
    jobs = {"onnx_compile": sess.job_id_for("compile", fp, device, "--target_runtime onnx") or "",
            "quantize": sess.job_id_for("quantize", _id(onnx_model, "model_id", "id"), calib_fp, "w8a8") or ""}
    return onnx_model, q, jobs


# ----------------------------------------------------------------------------- study 1: agreement


@dataclass
class AgreementStudy:
    runs: List[RunLogits]
    pairs: List[PairAgreement]
    labels: np.ndarray
    jobs: Dict[str, str]
    notes: List[str] = field(default_factory=list)
    log: Optional[JobLog] = None

    def to_text(self) -> str:
        lines = [agreement_text(self.runs, self.pairs, self.labels)]
        for n in self.notes:
            lines.append(f"note: {n}")
        lines.append("jobs: " + ", ".join(f"{k} {v}" for k, v in self.jobs.items() if v))
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "n": int(len(self.labels)),
            "accuracy": {r.name: r.accuracy(self.labels) for r in self.runs},
            "output_step": {r.name: r.step for r in self.runs},
            "pairs": [p.as_dict() for p in self.pairs],
            "jobs": self.jobs,
            "notes": self.notes,
        }


def int8_agreement_study(model: nn.Module, x: torch.Tensor, labels: torch.Tensor, calib_x: torch.Tensor,
                         devices: Sequence[str] = DEFAULT_DEVICES,
                         units: Sequence[Tuple[str, str]] = (),
                         session: Optional[HubSession] = None,
                         host_runs: Sequence[RunLogits] = (),
                         float_units: Sequence[Tuple[str, str]] = ()) -> AgreementStudy:
    """Quantize to w8a8, compile one TFLite model, run every input on each (device, unit).

    `units` lists (device, compute unit) pairs to run; default is NPU and CPU on the first device
    and NPU on the others. `float_units` optionally adds runs of the float TFLite model for contrast.
    `host_runs` are extra runs computed elsewhere (for example ONNX Runtime on the PC) added to the
    comparison as is.
    """
    sess = session or HubSession()
    devices = list(devices)
    if not units:
        units = [(devices[0], "npu"), (devices[0], "cpu")] + [(d, "npu") for d in devices[1:]]
    ref = reference_outputs(model, x)[0]
    runs = [RunLogits("float64 reference", ref, "float64")]
    jobs: Dict[str, str] = {}
    notes: List[str] = []

    onnx_model, q, j = onnx_and_quantized(sess, model, calib_x, devices[0])
    jobs.update(j)
    ds, data_fp = sess.dataset(_entries(x), name="truestar-eval")
    compiled: Dict[str, Any] = {}

    def tflite_for(src: Any, src_key: str, device: str, kind: str) -> Any:
        k = (kind, device)
        if k not in compiled:
            compiled[k] = sess.compile(src, src_key, device, "--target_runtime tflite",
                                       name=f"truestar-{kind}-tflite")
            jobs[f"{kind}_tflite_compile[{device}]"] = sess.job_id_for(
                "compile", src_key, device, "--target_runtime tflite") or ""
        return compiled[k]

    qkey = "qmodel:" + _id(q, "model_id", "id")
    for device, unit in units:
        # One TFLite file serves every device: compile once (for the first device) and reuse it.
        name = f"int8 {_short(device)} {unit}"
        try:
            try:
                target = tflite_for(q, qkey, devices[0], "int8")
                out, jid = sess.infer(target, device, ds, data_fp, f"--compute_unit {unit}",
                                      name=f"truestar-int8-{unit}")
            except Exception as first:  # noqa: BLE001
                if device == devices[0]:
                    raise
                # The shared file did not run there: compile for this device and try once more.
                notes.append(f"{name}: shared TFLite failed ({first}); compiled for this device")
                target = tflite_for(q, qkey, device, "int8")
                out, jid = sess.infer(target, device, ds, data_fp, f"--compute_unit {unit}",
                                      name=f"truestar-int8-{unit}")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{name} failed: {type(exc).__name__}: {exc}")
            continue
        logits = match_outputs(outputs_in_order(out)[:1], [ref.shape])[0]
        runs.append(RunLogits(name, logits, "int8", jobs={"inference": jid}))
        jobs[f"inference[{name}]"] = jid

    if float_units:
        base, traced = _trace(model, x)
        fkey = model_fingerprint(base, _specs([x]))
        for device, unit in float_units:
            target = sess.compile(traced, fkey, devices[0], "--target_runtime tflite",
                                  input_specs=_specs([x]), name="truestar-float-tflite")
            jobs["float_tflite_compile"] = sess.job_id_for("compile", fkey, devices[0],
                                                           "--target_runtime tflite") or ""
            name = f"float {_short(device)} {unit}"
            try:
                out, jid = sess.infer(target, device, ds, data_fp, f"--compute_unit {unit}",
                                      name=f"truestar-float-{unit}")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{name} failed: {type(exc).__name__}: {exc}")
                continue
            logits = match_outputs(outputs_in_order(out)[:1], [ref.shape])[0]
            runs.append(RunLogits(name, logits, "float16" if unit == "npu" else "float32",
                                  jobs={"inference": jid}))
            jobs[f"inference[{name}]"] = jid

    runs.extend(host_runs)
    pairs = agreement_matrix(runs)
    return AgreementStudy(runs=runs, pairs=pairs, labels=labels.cpu().numpy(), jobs=jobs, notes=notes,
                          log=sess.log)


def _short(device: str) -> str:
    return device.replace("Samsung Galaxy ", "").replace(" (Family)", "")


# ----------------------------------------------------------------------------- study 2: per layer


@dataclass
class LayerStudy:
    taps: List[str]
    blame: LayerBlame
    final: Dict[str, float]
    jobs: Dict[str, str]
    device: str
    quantized: bool
    log: Optional[JobLog] = None
    # Kept for follow up comparisons (not serialized): per unit outputs in tap order, the float64
    # reference in tap order, and the QDQ ONNX model for quantized studies.
    outputs: Dict[str, List[np.ndarray]] = field(default_factory=dict, repr=False)
    reference: List[np.ndarray] = field(default_factory=list, repr=False)
    source_model: Any = field(default=None, repr=False)

    def to_text(self) -> str:
        kind = "INT8 w8a8" if self.quantized else "float"
        lines = [f"per layer blame, {kind} model on {self.device}: NPU vs CPU control on the same phone"]
        lines.append(self.blame.to_text())
        lines.append("jobs: " + ", ".join(f"{k} {v}" for k, v in self.jobs.items() if v))
        return "\n".join(lines)

    def as_dict(self) -> dict:
        f = self.blame.first
        return {"device": self.device, "quantized": self.quantized, "taps": self.taps,
                "first_over_threshold": f.name if f else None,
                "rows": [r.as_dict() for r in self.blame.rows], "final": self.final, "jobs": self.jobs}


def layer_blame_study(model: nn.Module, x: torch.Tensor, taps: Optional[Sequence[str]] = None,
                      device: str = DEFAULT_DEVICES[0], quantized: bool = False,
                      calib_x: Optional[torch.Tensor] = None,
                      session: Optional[HubSession] = None,
                      device_unit: str = "npu", control_unit: str = "cpu") -> LayerStudy:
    """Run the model with taps as extra outputs on `device_unit` and `control_unit` of one phone and
    report the first tap where they part by more than the precision allows."""
    sess = session or HubSession()
    model = copy.deepcopy(model).cpu().float().eval()
    taps = list(taps) if taps is not None else default_taps(model, (x[:1],))
    wrapped = TapWrapper(model, taps).eval()
    ref = reference_outputs(wrapped, x)
    shapes = [r.shape for r in ref]
    jobs: Dict[str, str] = {}
    tag = "|taps:" + ",".join(taps)
    if quantized:
        if calib_x is None:
            raise ValueError("quantized=True needs calib_x")
        _onnx, q, j = onnx_and_quantized(sess, wrapped, calib_x, device, tag=tag)
        jobs.update(j)
        src, src_key, specs = q, "qmodel:" + _id(q, "model_id", "id"), None
    else:
        base, traced = _trace(wrapped, x)
        src, src_key, specs = traced, model_fingerprint(base, _specs([x]), tag), _specs([x])
    target = sess.compile(src, src_key, device, "--target_runtime tflite", input_specs=specs,
                          name="truestar-taps-tflite")
    jobs["tflite_compile"] = sess.job_id_for("compile", src_key, device, "--target_runtime tflite") or ""
    ds, data_fp = sess.dataset(_entries(x), name="truestar-taps")
    res = {}
    for unit in (device_unit, control_unit):
        out, jid = sess.infer(target, device, ds, data_fp, f"--compute_unit {unit}",
                              name=f"truestar-taps-{unit}")
        jobs[f"inference[{unit}]"] = jid
        res[unit] = match_outputs(outputs_in_order(out), shapes)
    in_tap_order = {u: v[1:] + v[:1] for u, v in res.items()}
    ref_order = ref[1:] + ref[:1]
    blame = layer_blame(taps + ["output"], in_tap_order[device_unit], in_tap_order[control_unit],
                        ref_order, quantized)
    final = {"device_vs_control": blame.rows[-1].rel_vs_control,
             "device_vs_reference": blame.rows[-1].rel_vs_reference,
             "control_vs_reference": blame.rows[-1].control_vs_reference}
    return LayerStudy(taps=taps, blame=blame, final=final, jobs=jobs, device=device,
                      quantized=quantized, log=sess.log, outputs=in_tap_order, reference=ref_order,
                      source_model=src if quantized else None)


# ----------------------------------------------------------------------------- host runs


def download_onnx(model: Any, directory: str) -> str:
    """Download an AI Hub ONNX model and return the path of the .onnx file (zips are unpacked)."""
    os.makedirs(directory, exist_ok=True)
    path = model.download(os.path.join(directory, "model"))
    if zipfile.is_zipfile(path):
        out = os.path.join(directory, "unzipped")
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
        for root, _dirs, files in os.walk(out):
            for f in files:
                if f.endswith(".onnx"):
                    return os.path.join(root, f)
        raise FileNotFoundError(f"no .onnx inside {path}")
    return path


def onnxruntime_outputs(onnx_path: str, x: torch.Tensor, optimized: bool) -> List[np.ndarray]:
    """Run a QDQ ONNX model on this PC with ONNX Runtime's CPU provider; every output, samples
    stacked on axis 0.

    optimized=False keeps every QuantizeLinear / DequantizeLinear pair, so the math is float with
    rounding at each quantization point: the intended INT8 model, simulated. optimized=True lets
    ONNX Runtime fuse them into integer kernels (QLinearConv and friends), like a deployment would.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimized
                                   else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    so.intra_op_num_threads = 4
    s = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    arr = x.detach().cpu().float().numpy()
    per = [s.run(None, {name: arr[i:i + 1]}) for i in range(arr.shape[0])]
    return [np.concatenate([p[k] for p in per], axis=0) for k in range(len(per[0]))]


def onnxruntime_logits(onnx_path: str, x: torch.Tensor, optimized: bool) -> np.ndarray:
    """First output of `onnxruntime_outputs`."""
    return onnxruntime_outputs(onnx_path, x, optimized)[0]
