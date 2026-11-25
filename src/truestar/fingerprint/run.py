"""Run the fingerprint: sha256 of every output of every case, dtype, batch size and arm.

Arms (each a group of rows, one row per batch size):

* ``truestar_det``: the compiled executor with ``deterministic=True`` (no split K; launch configs
  tuned only for kernels whose bits do not depend on the config, docs/KERNEL_IR.md 6b), torch
  deterministic algorithms on for the library calls inside the program. The only arm whose bits are
  claimed: equal run to run here, and equal across GPUs under the conditions in docs/CROSS_CHIP.md,
  which ``truestar fingerprint compare`` checks.
* ``truestar_default``: the compiled executor with ``deterministic=False`` and PyTorch's defaults:
  split K and autotuning by timing are on. Recorded and labelled, never claimed.
* ``eager``: PyTorch eager with deterministic algorithms on (cuDNN deterministic, no benchmark).
* ``inductor``: ``torch.compile(model)`` with PyTorch's defaults.

dtypes: ``float32`` (TF32 off everywhere), ``tf32`` (float32 storage, TF32 dot products and
convolutions allowed in every arm), ``float16``, ``bfloat16``.
"""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from typing import Dict, List, Optional, Sequence

import torch

from . import data as D
from .models import BATCHES, CASES, DTYPES, Case

SCHEMA = "truestar.fingerprint/1"

ARMS: Dict[str, dict] = {
    "truestar_det": {"kind": "truestar", "deterministic": True, "torch_deterministic": True,
                     "claim": "bitwise", "label": "Truestar compiled, deterministic=True"},
    "truestar_default": {"kind": "truestar", "deterministic": False, "torch_deterministic": False,
                         "claim": "none", "label": "Truestar compiled, deterministic=False (split K and "
                                                  "tuning by timing on)"},
    "eager": {"kind": "eager", "torch_deterministic": True, "claim": "none",
              "label": "PyTorch eager, deterministic algorithms on"},
    "inductor": {"kind": "inductor", "torch_deterministic": False, "claim": "none",
                 "label": "torch.compile default (Inductor), PyTorch defaults"},
    # opt in (--arms): half and TF32 dot products as exact tensor core blocks (docs/PORTABLE_TC.md)
    "truestar_portable": {"kind": "truestar", "deterministic": True, "torch_deterministic": True,
                          "portable": True, "claim": "bitwise",
                          "label": "Truestar compiled, portable_tensor_cores=True"},
}
DEFAULT_ARMS = ("truestar_det", "truestar_default", "eager", "inductor")
ARM_NAMES = tuple(ARMS)
LEGACY_ARMS = ("truestar_det", "eager")


def _torch_dtype(dtype: str) -> torch.dtype:
    return {"float32": torch.float32, "tf32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[dtype]


@contextlib.contextmanager
def torch_settings(deterministic: bool, tf32: bool):
    """PyTorch's determinism and TF32 switches for one arm, restored afterwards."""
    cudnn, matmul = torch.backends.cudnn, torch.backends.cuda.matmul
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        saved = (torch.are_deterministic_algorithms_enabled(),
                 torch.is_deterministic_algorithms_warn_only_enabled(), cudnn.deterministic,
                 cudnn.benchmark, matmul.allow_tf32, cudnn.allow_tf32)
        torch.use_deterministic_algorithms(deterministic)
        cudnn.deterministic = deterministic
        cudnn.benchmark = False
        matmul.allow_tf32 = tf32
        cudnn.allow_tf32 = tf32
    try:
        yield current_settings()
    finally:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
            cudnn.deterministic, cudnn.benchmark = saved[2], saved[3]
            matmul.allow_tf32, cudnn.allow_tf32 = saved[4], saved[5]


def current_settings() -> dict:
    cudnn, matmul = torch.backends.cudnn, torch.backends.cuda.matmul
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": bool(cudnn.deterministic), "cudnn_benchmark": bool(cudnn.benchmark),
                "matmul_allow_tf32": bool(matmul.allow_tf32), "cudnn_allow_tf32": bool(cudnn.allow_tf32)}


def _git() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    out = {"commit": None, "dirty": None, "source": None}
    try:
        top = subprocess.run(["git", "-C", here, "rev-parse", "--show-toplevel"], capture_output=True,
                             text=True, timeout=10)
        if top.returncode == 0:
            root = top.stdout.strip()
            sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                                 timeout=10).stdout.strip()
            st = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "src"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
            if sha:
                return {"commit": sha, "dirty": bool(st), "source": "git"}
    except (OSError, subprocess.SubprocessError):
        pass
    if os.environ.get("TRUESTAR_GIT_COMMIT"):   # a copy of the source without .git (embedded runs)
        out.update(commit=os.environ["TRUESTAR_GIT_COMMIT"], source="TRUESTAR_GIT_COMMIT",
                   dirty=os.environ.get("TRUESTAR_GIT_DIRTY", "unknown"))
    return out


def _driver() -> Optional[str]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        p = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=20)
        return p.stdout.strip().splitlines()[0].strip() if p.returncode == 0 and p.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def environment() -> dict:
    import numpy

    import truestar

    env = {"truestar": truestar.__version__, "git": _git(), "python": platform.python_version(),
           "platform": platform.platform(), "cpu": platform.processor() or platform.machine(),
           "torch": torch.__version__, "numpy": numpy.__version__, "cuda_runtime": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
           "driver": _driver()}
    try:
        import triton

        env["triton"] = triton.__version__
    except ImportError:
        env["triton"] = None
    try:
        import torchvision

        env["torchvision"] = torchvision.__version__
    except ImportError:
        env["torchvision"] = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(gpu=p.name, capability=f"{p.major}.{p.minor}", sm_count=p.multi_processor_count,
                   gpu_memory_mib=p.total_memory // (1 << 20))
    else:
        env.update(gpu=None, capability=None, sm_count=None, gpu_memory_mib=None)
    return env


def global_settings() -> dict:
    from ..kernel import tune
    from ..kernel import verify as kverify

    return {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "TRUESTAR_TUNE": os.environ.get("TRUESTAR_TUNE"), "tune_mode": tune.mode(None),
            "tune_budget_s": tune.budget_seconds(), "verify_mode": kverify.verify_mode()}


def _outputs(x) -> List[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return [x]
    out = []
    for t in x:
        out += _outputs(t) if isinstance(t, (list, tuple)) else ([t] if isinstance(t, torch.Tensor) else [])
    return out


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> Optional[float]:
    g, w = got.detach().cpu().double(), want.detach().cpu().double()
    if g.shape != w.shape:
        return None
    scale = float(w.abs().max()) if w.numel() else 0.0
    err = float((g - w).abs().max()) if w.numel() else 0.0
    if err != err:      # NaN in the output
        return float("inf")
    return err / scale if scale > 0 else err


def _cast(xs, case: Case, dtype: str, device: str):
    td = _torch_dtype(dtype)
    return [x.to(device, td) if f else x.to(device) for x, f in zip(xs, _float_flags(case, len(xs)))]


def _float_flags(case: Case, n: int):
    flags = tuple(case.float_inputs)
    return flags + (True,) * (n - len(flags)) if len(flags) < n else flags[:n]


def _err_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:1500]


class _Truestar:
    """One compiled program per (case, dtype, arm)."""

    def __init__(self, prog, dtype: str, deterministic: bool, portable: bool = False):
        from ..compiler import compile_program

        t0 = time.perf_counter()
        self.cp = compile_program(prog, "cuda", "float32" if dtype == "tf32" else dtype,
                                  allow_tf32=dtype == "tf32", deterministic=deterministic,
                                  portable_tensor_cores=portable)
        self.compile_seconds = time.perf_counter() - t0

    def __call__(self, xs):
        return self.cp(list(xs))

    def info(self) -> dict:
        rep = self.cp.report
        lib = sorted({op for e in rep.library for op in e.get("ops", [])})
        fb = sorted({op for e in rep.fallbacks for op in e.get("ops", [])})
        return {"compile_seconds": round(self.compile_seconds, 3), "library_ops": lib, "fallback_ops": fb,
                "kernels": len(rep.kernels), "fp32_precision": rep.fp32_precision,
                "max_error_vs_float64_own_check": getattr(self.cp, "max_error", None)}

    def configs(self) -> List[dict]:
        out, seen = [], set()
        for step in self.cp.plan.steps:
            ck = getattr(step, "ck", None)
            if ck is None or not hasattr(ck, "launch_configs"):
                continue
            for sig, cfg, src in ck.launch_configs():
                shapes = [[n, list(s)] for n, s in sig]
                key = (step.name, repr(shapes))
                if key in seen:
                    continue
                seen.add(key)
                out.append({"step": step.name, "plan": ck.plan_kind, "shapes": shapes, "config": dict(cfg),
                            "source": src})
        return out


def _import(model, xs, case: Case):
    from .. import ir

    B = torch.export.Dim("B", min=1, max=512)
    example = tuple(x[:2].clone() for x in xs)
    return ir.import_module(model, example, dynamic_shapes=tuple({0: B} for _ in example), name=case.name)


def _reference(model, xs, case: Case) -> List[torch.Tensor]:
    import copy

    m = copy.deepcopy(model).double()
    with torch.no_grad():
        args = [x.double() if f else x for x, f in zip(xs, _float_flags(case, len(xs)))]
        return [o.detach() for o in _outputs(m(*args))]


def run_group(case: Case, model, xs, ref, prog, dtype: str, arm: str, batches: Sequence[int],
              log=None) -> dict:
    spec = ARMS[arm]
    g = {"case": case.name, "dtype": dtype, "arm": arm, "claim": spec["claim"], "label": spec["label"],
         "truestar_deterministic": spec.get("deterministic"), "rows": [], "error": None}
    runner = None
    with torch_settings(spec["torch_deterministic"], dtype == "tf32") as st:
        g["settings"] = st
        try:
            if spec["kind"] == "truestar":
                if prog is None:
                    raise RuntimeError(f"import failed: {case.name} could not be imported into the IR")
                runner = _Truestar(prog, dtype, spec["deterministic"], bool(spec.get("portable")))
            elif spec["kind"] == "eager":
                import copy

                runner = copy.deepcopy(model).to("cuda", _torch_dtype(dtype))
            else:
                import copy

                torch._dynamo.reset()
                runner = torch.compile(copy.deepcopy(model).to("cuda", _torch_dtype(dtype)))
        except Exception as exc:         # recorded, never hidden
            g["error"] = _err_text(exc)
            g["traceback"] = traceback.format_exc()[-3000:]
            return g
        for n in batches:
            row = {"batch": n}
            t0 = time.perf_counter()
            try:
                args = _cast([x[:n].contiguous() for x in xs], case, dtype, "cuda")
                with torch.no_grad():
                    first = [o.detach().clone() for o in _outputs(runner(args) if spec["kind"] == "truestar"
                                                                   else runner(*args))]
                    torch.cuda.synchronize()
                    again = [o.detach().clone() for o in _outputs(runner(args) if spec["kind"] == "truestar"
                                                                   else runner(*args))]
                    torch.cuda.synchronize()
                row["outputs"] = [D.tensor_sha(o) for o in first]
                row["again"] = [D.tensor_sha(o) for o in again]
                row["run_to_run_equal"] = row["outputs"] == row["again"]
                row["row0"] = [D.tensor_sha(o[0]) for o in first]
                row["max_rel_error_vs_float64"] = [_rel_err(o, r[:n]) for o, r in zip(first, ref)]
                row["output_dtypes"] = [str(o.dtype).replace("torch.", "") for o in first]
            except Exception as exc:     # recorded, never hidden
                row["error"] = _err_text(exc)
            row["seconds"] = round(time.perf_counter() - t0, 3)
            g["rows"].append(row)
            if log:
                log(f"  {case.name:<18} {dtype:<8} {arm:<16} batch {n:>3}: "
                    + (f"ERROR {row['error'][:120]}" if "error" in row else
                       f"{row['outputs'][0][:12]} run to run {'equal' if row['run_to_run_equal'] else 'DIFFERENT'}"))
        if spec["kind"] == "truestar":
            g["compile"] = runner.info()
            g["configs"] = runner.configs()
            import hashlib
            import json

            g["configs_sha256"] = hashlib.sha256(json.dumps(
                [[c["step"], c["shapes"], c["config"]] for c in g["configs"]], sort_keys=True).encode()).hexdigest()
    ok = [r for r in g["rows"] if "row0" in r]
    if len(ok) > 1:
        g["batch_invariant_row0"] = all(r["row0"] == ok[0]["row0"] for r in ok)
    del runner
    torch.cuda.empty_cache()
    return g


def run(cases: Optional[Sequence[str]] = None, dtypes: Optional[Sequence[str]] = None,
        batches: Optional[Sequence[int]] = None, arms: Optional[Sequence[str]] = None,
        data_dir: Optional[str] = None, fresh_cache: bool = False, log=None) -> dict:
    """Run the fingerprint and return the JSON document (see docs/CROSS_CHIP.md).

    ``cases``, ``dtypes``, ``batches`` and ``arms`` filter the full matrix (None: everything; the
    legacy cases always run at batch 4 in float32 with the truestar_det and eager arms only).
    ``data_dir`` holds the weights and inputs files (default: a ``truestar_fingerprint`` folder in
    the temp directory). ``fresh_cache=True`` compiles and tunes into an empty kernel cache, so no
    launch config comes from an earlier process."""
    if not torch.cuda.is_available():
        raise RuntimeError("truestar fingerprint needs a CUDA GPU")
    was_init = torch.cuda.is_initialized()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    unknown = [c for c in (cases or []) if c not in CASES]
    if unknown:
        raise ValueError(f"unknown case(s) {', '.join(unknown)}; cases: {', '.join(CASES)}")
    bad = [d for d in (dtypes or []) if d not in DTYPES]
    if bad:
        raise ValueError(f"unknown dtype(s) {', '.join(bad)}; dtypes: {', '.join(DTYPES)}")
    bad = [a for a in (arms or []) if a not in ARMS]
    if bad:
        raise ValueError(f"unknown arm(s) {', '.join(bad)}; arms: {', '.join(ARMS)}")
    data_dir = data_dir or os.path.join(tempfile.gettempdir(), "truestar_fingerprint")
    old_cache = os.environ.get("TRUESTAR_CACHE_DIR")
    cache_tmp = tempfile.mkdtemp(prefix="truestar_fp_cache_") if fresh_cache else None
    if cache_tmp:
        os.environ["TRUESTAR_CACHE_DIR"] = cache_tmp
    doc = {"schema": SCHEMA, "env": environment(), "settings": global_settings(),
           "cuda_initialized_before_cublas_config": was_init, "fresh_cache": bool(fresh_cache),
           "arms": {k: {"label": v["label"], "claim": v["claim"]} for k, v in ARMS.items()},
           "data": {}, "groups": []}
    t_start = time.perf_counter()
    try:
        for name in (cases or list(CASES)):
            case = CASES[name]
            model, xs, rec = D.prepare(case, data_dir)
            doc["data"][name] = dict(rec, note=case.note, input_shapes=[list(x.shape) for x in xs],
                                     legacy=case.legacy)
            try:
                prog = _import(model, xs, case)
                import_error = None
            except Exception as exc:
                prog, import_error = None, _err_text(exc)
            doc["data"][name]["import_error"] = import_error
            ref = _reference(model, xs, case)
            case_batches = [b for b in case.batches if batches is None or b in batches or case.legacy]
            case_dtypes = [d for d in case.dtypes if dtypes is None or d in dtypes or case.legacy]
            case_arms = [a for a in (arms or DEFAULT_ARMS) if not case.legacy or a in LEGACY_ARMS]
            for dtype in case_dtypes:
                for arm in case_arms:
                    doc["groups"].append(run_group(case, model, xs, ref, prog, dtype, arm, case_batches, log))
    finally:
        if cache_tmp:
            if old_cache is None:
                os.environ.pop("TRUESTAR_CACHE_DIR", None)
            else:
                os.environ["TRUESTAR_CACHE_DIR"] = old_cache
            shutil.rmtree(cache_tmp, ignore_errors=True)
    doc["seconds"] = round(time.perf_counter() - t_start, 1)
    doc["issues"] = issues(doc)
    return doc


def issues(doc: dict) -> List[dict]:
    """Codes for one fingerprint: TS741 a claimed row was not equal run to run, TS742 an arm or row
    failed (recorded, not a finding about the bits), TS750 a case's data differs from the pinned hashes."""
    out = []
    for name, d in doc.get("data", {}).items():
        if d.get("golden_match") is False:
            out.append({"code": "TS750", "message": f"{name}: the weights or inputs built here differ from the "
                                                    f"pinned hashes (models.GOLDEN); outputs from this machine "
                                                    f"cannot be compared with other machines' for this case"})
    for g in doc["groups"]:
        where = f"{g['case']} {g['dtype']} {g['arm']}"
        if g.get("error"):
            out.append({"code": "TS742", "message": f"{where}: {g['error']}"})
        for r in g["rows"]:
            if "error" in r:
                out.append({"code": "TS742", "message": f"{where} batch {r['batch']}: {r['error']}"})
            elif g["claim"] == "bitwise" and not r["run_to_run_equal"]:
                out.append({"code": "TS741", "message": f"{where} batch {r['batch']}: two calls gave different "
                                                        f"bits in deterministic mode"})
    return out


def to_text(doc: dict) -> str:
    env = doc["env"]
    git = env.get("git") or {}
    lines = [f"truestar fingerprint: {env.get('gpu')} (sm_{str(env.get('capability')).replace('.', '')}, "
             f"{env.get('sm_count')} SMs), driver {env.get('driver')}, CUDA {env.get('cuda_runtime')}, "
             f"torch {env.get('torch')}, Triton {env.get('triton')}, cuDNN {env.get('cudnn')}",
             f"commit {str(git.get('commit'))[:12]}{' (dirty)' if git.get('dirty') is True else ''}, "
             f"{env.get('platform')}", ""]
    for g in doc["groups"]:
        head = f"{g['case']:<18} {g['dtype']:<8} {g['arm']:<16}"
        if g.get("error"):
            lines.append(f"{head} ERROR {g['error'][:160]}")
            continue
        cells = []
        for r in g["rows"]:
            if "error" in r:
                cells.append(f"b{r['batch']}=ERROR")
            else:
                mark = "" if r["run_to_run_equal"] else "!"
                cells.append(f"b{r['batch']}={r['outputs'][0][:8]}{mark}")
        inv = g.get("batch_invariant_row0")
        lines.append(f"{head} {' '.join(cells)}" + ("" if inv is None else f"  row0 batch invariant: {inv}"))
    lines.append("")
    lines.append("! marks a row whose two calls differed. Only truestar_det rows are claimed bitwise; "
                 "the other arms are recorded for comparison.")
    for i in doc.get("issues", []):
        lines.append(f"[{i['code']}] {i['message']}")
    return "\n".join(lines)
