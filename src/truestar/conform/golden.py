"""Golden vector bundles (`.tsgold`): what a deployed artifact must compute, replayable anywhere.

A bundle is a zip of `.npy` arrays and one `manifest.json`, readable with numpy alone:

* `inputs/s{k}_{i}.npy`: sample k, input i, in the artifact's input dtype;
* `expected/s{k}_{j}.npy`: output j of sample k from the PyTorch model in float64;
* `bound/s{k}_{j}.npy`: the derived per element bound (|out - expected| allowed for a correct float32
  implementation, docs/TOLERANCE.md);
* `runtime/{name}/s{k}_{j}.npy`: what each runtime produced when the bundle was written, with its
  fingerprint in the manifest (the upgrade baseline);
* `units/{u}/...`: for each module mapped into the artifact, its float64 reference inputs and
  outputs and each runtime's output of the module cut alone, so a rerun names the first module
  whose own result changed without the model code;
* `check_outputs.py`: a numpy only checker for teams that run the artifact on their own device.

Entries are written in a fixed order with a fixed timestamp, so the same content gives the same
bytes. Reading checks every file's sha256 against the manifest and refuses a damaged bundle (TS632).
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FORMAT = "truestar-golden"
FORMAT_VERSION = 1
_TS = (2026, 1, 1, 0, 0, 0)

RULE = ("An output conforms when its relative L2 error against `expected` (over finite elements) "
        "is at most `threshold`, NaN and infinity sit where `expected` has them, and the shape "
        "matches. `bound` gives the per element limit of a correct float32 implementation; "
        "elements beyond it are reported, the verdict uses the relative L2 rule.")

CHECKER = '''"""Check a device's outputs against a Truestar golden bundle with numpy only.

    python check_outputs.py bundle.tsgold outputs.npz

outputs.npz holds s{k}_{j} arrays: output j of sample k, as the device computed it. Exit 0 when
every output conforms, 1 otherwise.
"""
import io, json, sys, zipfile
import numpy as np


def main(bundle, outputs):
    z = zipfile.ZipFile(bundle)
    man = json.loads(z.read("manifest.json"))
    got = np.load(outputs)
    bad = 0
    for o in man["outputs"]:
        for k in range(man["samples"]):
            key = "s%d_%d" % (k, o["index"])
            exp = np.load(io.BytesIO(z.read("expected/" + key + ".npy"))).astype(np.float64)
            bnd = np.load(io.BytesIO(z.read("bound/" + key + ".npy")))
            if key not in got:
                print(key, "missing"); bad += 1; continue
            a = got[key].astype(np.float64)
            if a.shape != exp.shape:
                print(key, "shape", a.shape, "expected", exp.shape); bad += 1; continue
            fa, fe = np.isfinite(a), np.isfinite(exp)
            if (fa != fe).any():
                print(key, "NaN or inf where expected has none"); bad += 1; continue
            d = (a - exp)[fe]
            den = np.linalg.norm(exp[fe])
            rel = float(np.linalg.norm(d) / den) if den else float(np.linalg.norm(d))
            beyond = int((np.abs(a - exp)[fe] > bnd[fe]).sum())
            ok = rel <= o["threshold"]
            bad += 0 if ok else 1
            print(key, "ok" if ok else "FAIL", "rel %.3g (limit %.3g)" % (rel, o["threshold"]),
                  "%d elements beyond the derived bound" % beyond)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
'''


class GoldenError(RuntimeError):
    """The bundle is missing, damaged or of an unknown version (TS632)."""


def _npy(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(a), allow_pickle=False)
    return buf.getvalue()


class Writer:
    def __init__(self):
        self.files: Dict[str, bytes] = {}

    def array(self, name: str, a: np.ndarray) -> str:
        self.files[name] = _npy(np.asarray(a))
        return name

    def write(self, path: str, manifest: Dict[str, Any]) -> None:
        manifest = dict(manifest)
        manifest["format"] = FORMAT
        manifest["format_version"] = FORMAT_VERSION
        manifest["rule"] = RULE
        self.files["check_outputs.py"] = CHECKER.encode("utf-8")
        manifest["sha256"] = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(self.files.items())}
        body = json.dumps(manifest, sort_keys=True, indent=1, allow_nan=False).encode("utf-8")
        tmp = path + ".tmp"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ["manifest.json"] + sorted(self.files):
                info = zipfile.ZipInfo(name, date_time=_TS)
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, body if name == "manifest.json" else self.files[name])
        import os

        os.replace(tmp, path)


class Bundle:
    def __init__(self, path: str):
        self.path = path
        try:
            self.zip = zipfile.ZipFile(path)
            self.manifest = json.loads(self.zip.read("manifest.json"))
        except FileNotFoundError as exc:
            raise GoldenError(f"{path} not found") from exc
        except Exception as exc:
            raise GoldenError(f"{path} is not a golden bundle ({type(exc).__name__}: {exc})") from exc
        m = self.manifest
        if m.get("format") != FORMAT:
            raise GoldenError(f"{path} is not a golden bundle (format {m.get('format')!r})")
        if m.get("format_version") != FORMAT_VERSION:
            raise GoldenError(f"{path} has bundle version {m.get('format_version')}; this truestar reads "
                              f"version {FORMAT_VERSION}")
        names = set(self.zip.namelist()) - {"manifest.json"}
        want = m.get("sha256", {})
        if set(want) != names:
            raise GoldenError(f"{path}: files differ from the manifest (missing "
                              f"{sorted(set(want) - names)[:3]}, extra {sorted(names - set(want))[:3]})")
        for name, h in want.items():
            if hashlib.sha256(self.zip.read(name)).hexdigest() != h:
                raise GoldenError(f"{path}: {name} is damaged (sha256 differs from the manifest)")

    def array(self, name: str) -> np.ndarray:
        return np.load(io.BytesIO(self.zip.read(name)), allow_pickle=False)

    def inputs(self, k: int) -> List[np.ndarray]:
        return [self.array(f) for f in self.manifest["inputs"][k]]

    def expected(self, k: int) -> List[np.ndarray]:
        return [self.array(f"expected/s{k}_{o['index']}.npy") for o in self.manifest["outputs"]]

    def runtime_outputs(self, runtime: str, k: int) -> Optional[List[np.ndarray]]:
        r = self.manifest.get("runtimes", {}).get(runtime)
        if not r or not r.get("outputs"):
            return None
        return [self.array(f) for f in r["outputs"][k]]
