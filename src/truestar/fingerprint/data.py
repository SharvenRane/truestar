"""Weights and inputs of every case, written to files once and hash checked on every load.

``prepare(case, data_dir)`` builds the model and its inputs on the CPU from fixed seeds, saves them
as ``<case>.weights.pt`` and ``<case>.inputs.pt`` and records three hashes in ``manifest.json``: the
sha256 of each file and the sha256 over the tensors' raw bytes. A later run with the same directory
loads the files instead of rebuilding them and refuses (TS740) when a file's bytes no longer match
the manifest. The tensor hashes go into the fingerprint, so two machines can prove they started from
the same bits even though ``torch.save`` files are not byte identical across platforms.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Tuple

import torch

from .models import GOLDEN, Case, build

MANIFEST = "manifest.json"


class DataMismatch(Exception):
    """A saved weights or inputs file does not match its manifest (TS740)."""


def tensor_sha(t: torch.Tensor) -> str:
    """sha256 over the raw bytes of a tensor (any dtype, 0 dim included)."""
    t = t.detach().contiguous().cpu().reshape(-1)
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"").hexdigest()


def state_sha(sd: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sd:
        h.update(name.encode())
        h.update(tensor_sha(sd[name]).encode())
    return h.hexdigest()


def inputs_sha(xs: Tuple[torch.Tensor, ...]) -> str:
    h = hashlib.sha256()
    for x in xs:
        h.update(str(x.dtype).encode())
        h.update(repr(tuple(x.shape)).encode())
        h.update(tensor_sha(x).encode())
    return h.hexdigest()


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(data_dir: str) -> dict:
    path = os.path.join(data_dir, MANIFEST)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        try:
            doc = json.load(f)
        except json.JSONDecodeError as exc:
            raise DataMismatch(f"{path} is not valid JSON ({exc}); delete the directory to rebuild it") from exc
    return doc if isinstance(doc, dict) else {}


def _write_manifest(data_dir: str, doc: dict) -> None:
    path = os.path.join(data_dir, MANIFEST)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def prepare(case: Case, data_dir: str):
    """(model on the CPU in eval mode, inputs at the largest batch, record) for ``case``.

    The record holds the tensor hashes that go into the fingerprint and whether the files were
    written now or loaded."""
    os.makedirs(data_dir, exist_ok=True)
    wpath = os.path.join(data_dir, f"{case.name}.weights.pt")
    ipath = os.path.join(data_dir, f"{case.name}.inputs.pt")
    manifest = _read_manifest(data_dir)
    entry = manifest.get(case.name)
    model = build(case)
    if entry and os.path.isfile(wpath) and os.path.isfile(ipath):
        for path, key in ((wpath, "weights_file_sha256"), (ipath, "inputs_file_sha256")):
            got = _file_sha(path)
            if got != entry.get(key):
                raise DataMismatch(f"{path}: sha256 {got[:16]}... does not match the manifest "
                                   f"({str(entry.get(key))[:16]}...); the file changed after it was written. "
                                   f"Delete {data_dir} to rebuild it from the seeds")
        sd = torch.load(wpath, map_location="cpu", weights_only=True)
        xs = tuple(torch.load(ipath, map_location="cpu", weights_only=True))
        source = "loaded"
    else:
        torch.save(model.state_dict(), wpath)
        xs = case.make_inputs(max(case.batches))
        torch.save(list(xs), ipath)
        sd = torch.load(wpath, map_location="cpu", weights_only=True)
        xs = tuple(torch.load(ipath, map_location="cpu", weights_only=True))
        source = "written"
    model.load_state_dict(sd)
    model.eval()
    rec = {"weights_sha256": state_sha(model.state_dict()), "inputs_sha256": inputs_sha(xs),
           "input_tensor_sha256": [tensor_sha(x) for x in xs],
           "source": source}
    golden = GOLDEN.get(case.name)
    rec["golden_match"] = None if golden is None else (rec["weights_sha256"], rec["inputs_sha256"]) == golden
    if entry and source == "loaded":
        for key in ("weights_sha256", "inputs_sha256"):
            if entry.get(key) != rec[key]:
                raise DataMismatch(f"{case.name}: the loaded {key.split('_')[0]} hash {rec[key][:16]}... does "
                                   f"not match the manifest ({str(entry.get(key))[:16]}...)")
    manifest[case.name] = dict(rec, weights_file_sha256=_file_sha(wpath), inputs_file_sha256=_file_sha(ipath))
    manifest[case.name].pop("source", None)
    manifest[case.name].pop("golden_match", None)
    _write_manifest(data_dir, manifest)
    return model, xs, rec
