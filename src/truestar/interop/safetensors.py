"""safetensors weights, read and written without the safetensors package, checked against the model.

Format (https://github.com/huggingface/safetensors): 8 bytes little endian u64 header length N,
N bytes of UTF 8 JSON (an object, may be padded with spaces), then the data buffer. Each JSON entry
maps a tensor name to ``{"dtype", "shape", "data_offsets": [begin, end]}``, offsets relative to the
buffer, little endian, row major, densely packed. ``__metadata__`` is an optional str to str map.

Reading (``load_file``) validates the whole header first and reports every problem at once:
duplicate names (Python's ``json`` would keep the last one silently), unknown dtypes, a byte length
that does not match the shape, overlapping ranges, holes, a buffer longer or shorter than the
tensors, a header over 100 MB. Tensors are views of a copy on write memory map (no read until
touched, no write ever reaches the file); a tensor whose offset is not aligned to its dtype is
copied, and ``LoadReport.copied`` names it.

``load_weights(path, into=...)`` checks the file against what it is loaded into before anything is
assigned, and lists every missing, unexpected, mistyped and misshapen tensor at once:

* a native function (``@truestar.function``): its tensor parameters; symbolic dims must bind
  consistently across all weights; parameters not in the file are the data inputs;
* an IR program: its params (the importer names them like the module's ``state_dict``);
* a ``torch.nn.Module``: its ``state_dict``. No silent dtype cast (``cast=True`` allows it).

Shape checks cannot see two tensors of one shape swapped, or a square weight stored transposed.
``save_weights(path, weights, check=model)`` therefore records the model's outputs on inputs that
are regenerated exactly from a formula (no random generator, so every machine gets the same bits),
and ``load_weights`` reruns them and refuses weights that do not reproduce the outputs.
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import sympy
import torch

MAX_HEADER = 100 * 1024 * 1024
CHECK_KEY = "truestar.check"
CHECK_VALUES = 4096   # output values stored per output, at most (evenly strided over the output)

DTYPES: Dict[str, torch.dtype] = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U64": torch.uint64, "U32": torch.uint32, "U16": torch.uint16, "U8": torch.uint8, "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2, "C64": torch.complex64,
}
NAMES = {v: k for k, v in DTYPES.items()}


class SafetensorsError(ValueError):
    """A file or a load that is wrong. ``problems`` lists every one found; ``code`` is stable."""

    def __init__(self, code: str, summary: str, problems: Sequence[str] = (), suggestion: str = ""):
        self.code = code
        self.summary = summary
        self.problems = list(problems)
        self.suggestion = suggestion
        lines = [summary] + [f"  {p}" for p in self.problems[:50]]
        if len(self.problems) > 50:
            lines.append(f"  ... and {len(self.problems) - 50} more")
        if suggestion:
            lines.append(f"  fix: {suggestion}")
        super().__init__("\n".join(lines))

    def to_dict(self):
        return {"code": self.code, "summary": self.summary, "problems": self.problems, "suggestion": self.suggestion}


# --------------------------------------------------------------------------- header


@dataclass(frozen=True)
class Entry:
    name: str
    dtype: torch.dtype
    shape: Tuple[int, ...]
    begin: int
    end: int


def _no_duplicates(pairs):
    out, dup = {}, []
    for k, v in pairs:
        if k in out:
            dup.append(k)
        out[k] = v
    if dup:
        out["\0duplicates"] = dup
    return out


def read_header(path: str) -> Tuple[Dict[str, Entry], Dict[str, str], int, int]:
    """Validate the header. Returns (entries, metadata, data start offset, data length)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise SafetensorsError("safetensors_truncated", f"{path} is {size} bytes, shorter than the 8 byte header length")
        (n,) = struct.unpack("<Q", head)
        if n > MAX_HEADER:
            raise SafetensorsError("safetensors_bad_header", f"{path}: header length {n} is over the 100 MB limit "
                                   "(not a safetensors file, or a damaged one)")
        if 8 + n > size:
            raise SafetensorsError("safetensors_truncated", f"{path}: header length {n} runs past the end of the "
                                   f"{size} byte file")
        raw = f.read(n)
    if not raw.startswith(b"{"):
        raise SafetensorsError("safetensors_bad_header", f"{path}: the header does not start with '{{'")
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SafetensorsError("safetensors_bad_header", f"{path}: the header is not UTF 8 JSON: {e}") from None
    if not isinstance(obj, dict):
        raise SafetensorsError("safetensors_bad_header", f"{path}: the header is not a JSON object")
    problems: List[str] = []
    for d in obj.pop("\0duplicates", []):
        problems.append(f"{d}: the name appears more than once in the header")
    meta = obj.pop("__metadata__", None) or {}
    if not isinstance(meta, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()):
        problems.append("__metadata__ must map strings to strings")
        meta = {}
    data_len = size - 8 - n
    entries: Dict[str, Entry] = {}
    for name, e in obj.items():
        if not isinstance(e, dict) or set(e) - {"dtype", "shape", "data_offsets"} or len(e) != 3:
            problems.append(f"{name}: entry must have exactly dtype, shape and data_offsets, got {e!r}"[:200])
            continue
        dt = DTYPES.get(e["dtype"])
        if dt is None:
            problems.append(f"{name}: unknown dtype {e['dtype']!r} (known: {sorted(DTYPES)})")
            continue
        shape, off = e["shape"], e["data_offsets"]
        if not (isinstance(shape, list) and all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape)):
            problems.append(f"{name}: shape must be a list of non negative ints, got {shape!r}")
            continue
        if not (isinstance(off, list) and len(off) == 2 and all(isinstance(x, int) and not isinstance(x, bool)
                                                                and x >= 0 for x in off) and off[0] <= off[1]):
            problems.append(f"{name}: data_offsets must be [begin, end] with 0 <= begin <= end, got {off!r}")
            continue
        want = math.prod(shape) * torch.empty((), dtype=dt).element_size()
        if off[1] - off[0] != want:
            problems.append(f"{name}: {e['dtype']}{shape} needs {want} bytes but data_offsets {off} hold "
                            f"{off[1] - off[0]}")
            continue
        entries[name] = Entry(name, dt, tuple(shape), off[0], off[1])
    pos = 0
    for en in sorted(entries.values(), key=lambda x: (x.begin, x.end)):
        if en.begin < pos:
            problems.append(f"{en.name}: bytes [{en.begin}, {en.end}) overlap the tensor before it (ends at {pos})")
        elif en.begin > pos:
            problems.append(f"{en.name}: a hole of {en.begin - pos} bytes before it (at {pos})")
        pos = max(pos, en.end)
    if pos != data_len:
        problems.append(f"the tensors cover {pos} bytes but the data buffer is {data_len} bytes"
                        + (" (truncated file)" if pos > data_len else " (trailing bytes)"))
    if problems:
        raise SafetensorsError("safetensors_invalid", f"{path} is not a valid safetensors file "
                               f"({len(problems)} problem(s), nothing was loaded):", problems)
    return entries, meta, 8 + n, data_len


# --------------------------------------------------------------------------- load and save


@dataclass
class LoadReport:
    path: str
    tensors: int = 0
    bytes: int = 0
    mmap: bool = True
    copied: List[str] = field(default_factory=list)   # unaligned tensors that had to be copied
    seconds: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)


def load_file(path: str, *, device="cpu", mmap: bool = True, report: Optional[LoadReport] = None
              ) -> Dict[str, torch.Tensor]:
    """Every tensor in the file, after the header is validated. With ``mmap`` (default) CPU tensors are
    views of a copy on write memory map; other devices get a copy there."""
    t0 = time.perf_counter()
    entries, meta, start, data_len = read_header(path)
    rep = report if report is not None else LoadReport(path)
    rep.mmap = mmap
    rep.metadata = meta
    if data_len == 0:
        buf = torch.empty(0, dtype=torch.uint8)
    elif mmap:
        buf = torch.from_numpy(np.memmap(path, dtype=np.uint8, mode="c", offset=start, shape=(data_len,)))
    else:
        with open(path, "rb") as f:
            f.seek(start)
            buf = torch.frombuffer(bytearray(f.read(data_len)), dtype=torch.uint8)
    dev = torch.device(device)
    out: Dict[str, torch.Tensor] = {}
    for name, e in entries.items():
        size = torch.empty((), dtype=e.dtype).element_size()
        if e.end == e.begin:
            t = torch.empty(e.shape, dtype=e.dtype)
        else:
            raw = buf[e.begin:e.end]
            if e.begin % size:
                raw = raw.clone()
                rep.copied.append(name)
            t = raw.view(e.dtype).reshape(e.shape)
        out[name] = t if dev.type == "cpu" else t.to(dev)
        rep.tensors += 1
        rep.bytes += e.end - e.begin
    rep.seconds = time.perf_counter() - t0
    return out


def save_file(tensors: Mapping[str, torch.Tensor], path: str, metadata: Optional[Mapping[str, str]] = None) -> str:
    """Write a safetensors file atomically (temporary file, then rename). Tensors are ordered by
    dtype size, largest first, then name, so every tensor is aligned to its dtype."""
    problems = []
    for k, t in tensors.items():
        if not isinstance(k, str) or k == "__metadata__":
            problems.append(f"{k!r}: names must be strings other than __metadata__")
        elif not isinstance(t, torch.Tensor):
            problems.append(f"{k}: not a tensor ({type(t).__name__})")
        elif t.dtype not in NAMES:
            problems.append(f"{k}: dtype {t.dtype} has no safetensors name")
        elif t.layout != torch.strided:
            problems.append(f"{k}: layout {t.layout} cannot be saved")
    meta = dict(metadata or {})
    if not all(isinstance(a, str) and isinstance(b, str) for a, b in meta.items()):
        problems.append("metadata must map strings to strings")
    if problems:
        raise SafetensorsError("safetensors_cannot_save", f"cannot write {path}:", problems)
    order = sorted(tensors, key=lambda k: (-tensors[k].element_size(), k))
    header: Dict[str, Any] = {}
    if meta:
        header["__metadata__"] = dict(sorted(meta.items()))
    pos = 0
    for k in order:
        t = tensors[k]
        nb = t.numel() * t.element_size()
        header[k] = {"dtype": NAMES[t.dtype], "shape": list(t.shape), "data_offsets": [pos, pos + nb]}
        pos += nb
    hb = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    hb += b" " * (-len(hb) % 8)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for k in order:
            t = tensors[k].detach()
            if t.numel():
                f.write(t.to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy().data)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- reference inputs


def weyl(shape: Sequence[int], dtype: torch.dtype, salt: int, int_high: int = 2) -> torch.Tensor:
    """Inputs from a formula, the same bits on every machine: entry i of input number ``salt`` is
    ``frac(0.5 + (i + 7919 salt) / golden ratio)`` mapped to [-1, 1), computed in float64."""
    n = math.prod(shape)
    i = torch.arange(n, dtype=torch.float64) + 7919.0 * salt
    u = torch.remainder(0.5 + i * 0.6180339887498949, 1.0)
    if dtype.is_floating_point:
        return (u * 2.0 - 1.0).reshape(shape).to(dtype)
    if dtype == torch.bool:
        return (u > 0.5).reshape(shape)
    return torch.floor(u * int_high).to(dtype).reshape(shape)


_OUT_TOL = {torch.float64: 1e-9, torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 6e-2}


def _hexes(t: torch.Tensor) -> Tuple[List[int], List[str]]:
    flat = t.detach().cpu().to(torch.float64).reshape(-1)
    step = max(1, math.ceil(flat.numel() / CHECK_VALUES))
    idx = list(range(0, flat.numel(), step))
    return idx, [float(flat[i]).hex() for i in idx]


def _run_target(into, weights: Mapping[str, torch.Tensor], spec: Dict[str, Any]) -> List[torch.Tensor]:
    inputs = [weyl(x["shape"], DTYPES[x["dtype"]], k, x.get("int_high", 2)) for k, x in enumerate(spec["inputs"])]
    if isinstance(into, torch.nn.Module):
        with torch.no_grad():
            from torch.func import functional_call
            out = functional_call(into, dict(weights), tuple(inputs), strict=False)
    else:
        kw = {x["name"]: t for x, t in zip(spec["inputs"], inputs)}
        out = into(**kw, **weights)
    if hasattr(out, "to_tuple"):  # Hugging Face ModelOutput
        out = out.to_tuple()
    outs = list(out) if isinstance(out, (tuple, list)) else [out]
    return [o for o in outs if isinstance(o, torch.Tensor)]


def make_check(into, weights: Mapping[str, torch.Tensor], inputs: Optional[Sequence[torch.Tensor]] = None,
               sizes: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """The reference record: input shapes and the outputs of ``into`` with ``weights`` on regenerated
    inputs. Modules need example ``inputs`` (only their shapes and dtypes are used)."""
    if isinstance(into, torch.nn.Module):
        if into.training:
            raise SafetensorsError("safetensors_check_training", "the module is in training mode, so its outputs are "
                                   "not a function of its weights alone", suggestion="call .eval() before saving")
        stray = sorted(set(weights) - set(into.state_dict()))
        if stray:
            raise SafetensorsError("safetensors_check_names", "the weights hold names the module does not have:",
                                   stray, "rename or drop them before saving with check=")
        if inputs is None:
            raise SafetensorsError("safetensors_check_inputs", "a module check needs example inputs for their "
                                   "shapes and dtypes", suggestion="pass check_inputs=(x,)")
        spec_inputs = [{"name": f"arg{k}", "dtype": NAMES[t.dtype], "shape": list(t.shape),
                        **({"int_high": int(t.max()) + 1} if not t.dtype.is_floating_point and t.numel() else {})}
                       for k, t in enumerate(inputs)]
    else:
        sig = into.signature
        bind = _bind_function(into, weights, sizes or {})
        spec_inputs = []
        for n, t in sig.params:
            if n in weights:
                continue
            shape = [_eval(d, bind) for d in t.shape]
            spec_inputs.append({"name": n, "dtype": NAMES[t.dtype], "shape": shape})
    spec = {"version": 1, "generator": "weyl", "inputs": spec_inputs}
    outs = _run_target(into, weights, spec)
    declared = [t.dtype for t in into.signature.outputs] if not isinstance(into, torch.nn.Module) else [o.dtype for o in outs]
    spec["outputs"] = []
    for o, dt in zip(outs, declared):
        idx, vals = _hexes(o)
        spec["outputs"].append({"shape": list(o.shape), "dtype": NAMES.get(dt, str(dt)),
                                "stride": idx[1] - idx[0] if len(idx) > 1 else 1, "values": vals})
    return spec


def _eval(d, bind) -> int:
    from ..ir.core import eval_dim, symbol
    return int(eval_dim(d, {symbol(k): v for k, v in bind.items()}))


def _bind_function(fn, weights: Mapping[str, Any], sizes: Dict[str, int]) -> Dict[str, int]:
    bind: Dict[str, int] = dict(sizes)
    for n, t in fn.signature.params:
        w = weights.get(n)
        if w is None:
            continue
        for d, s in zip(t.shape, w.shape):
            if isinstance(d, sympy.Symbol):
                bind.setdefault(str(d), int(s))
    for s in fn.signature.symbols:
        bind.setdefault(s, 2)
    return bind


def run_check(into, weights: Mapping[str, torch.Tensor], spec: Dict[str, Any]) -> List[float]:
    """Rerun the record; returns the relative error per output. Raises on a mismatch."""
    outs = _run_target(into, weights, spec)
    if len(outs) != len(spec["outputs"]):
        raise SafetensorsError("safetensors_check_failed", f"the model returns {len(outs)} outputs, the record has "
                               f"{len(spec['outputs'])}")
    errs, problems = [], []
    for k, (o, r) in enumerate(zip(outs, spec["outputs"])):
        if list(o.shape) != r["shape"]:
            problems.append(f"output {k}: shape {list(o.shape)}, recorded {r['shape']}")
            errs.append(float("inf"))
            continue
        flat = o.detach().cpu().to(torch.float64).reshape(-1)
        got = flat[::r["stride"]][:len(r["values"])]
        want = torch.tensor([float.fromhex(v) for v in r["values"]], dtype=torch.float64)
        fin = torch.isfinite(want)
        if not torch.equal(fin, torch.isfinite(got)):
            err = float("inf")
        else:
            scale = max(float(want[fin].abs().max()) if fin.any() else 0.0, 1e-12)
            err = float((got[fin] - want[fin]).abs().max()) / scale if fin.any() else 0.0
        tol = _OUT_TOL.get(DTYPES.get(r["dtype"], torch.float32), 1e-4)
        errs.append(err)
        if err > tol:
            problems.append(f"output {k} {r['dtype']}{r['shape']}: relative error {err:.2e} against the recorded "
                            f"output (tolerance {tol:.0e})")
    if problems:
        raise SafetensorsError("safetensors_check_failed", "the loaded weights do not reproduce the outputs recorded "
                               "when the file was saved:", problems,
                               "every shape matched, so look for tensors of one shape swapped, a square weight "
                               "stored transposed, a wrong dtype conversion or a damaged file")
    return errs


def save_weights(path: str, weights: Mapping[str, torch.Tensor], *, check=None, check_inputs=None,
                 sizes: Optional[Dict[str, int]] = None, metadata: Optional[Mapping[str, str]] = None) -> str:
    """``save_file`` plus, with ``check=model``, the reference record in ``__metadata__``."""
    meta = dict(metadata or {})
    if check is not None:
        meta[CHECK_KEY] = json.dumps(make_check(check, weights, check_inputs, sizes), separators=(",", ":"))
    return save_file(weights, path, meta)


# --------------------------------------------------------------------------- load into a model


@dataclass
class LoadedWeights(Mapping[str, torch.Tensor]):
    """The checked tensors (a mapping, so ``fn(x, **w)`` works) and what was checked."""

    tensors: Dict[str, torch.Tensor]
    data_inputs: List[str]
    report: LoadReport
    check_errors: Optional[List[float]]   # per output; None when the file has no record
    program: Any = None                  # for an IR program: the program with these params
    tied: Dict[str, str] = field(default_factory=dict)   # module names filled through a tied weight

    def __getitem__(self, k):
        return self.tensors[k]

    def __iter__(self):
        return iter(self.tensors)

    def __len__(self):
        return len(self.tensors)


def _targets(into) -> Tuple[str, Dict[str, Tuple[torch.dtype, tuple]]]:
    from ..frontend.function import NativeFunction
    from ..ir.core import Program
    if isinstance(into, NativeFunction):
        return "function", {n: (t.dtype, tuple(t.shape)) for n, t in into.signature.params}
    if isinstance(into, Program):
        return "program", {p.name: (p.type.dtype, tuple(p.type.shape)) for p in into.params}
    if isinstance(into, torch.nn.Module):
        return "module", {k: (v.dtype, tuple(v.shape)) for k, v in into.state_dict().items()}
    raise TypeError(f"load_weights loads into a native function, an IR program or a torch.nn.Module, "
                    f"got {type(into).__name__}")


def _fmt(dtype, shape) -> str:
    return f"{NAMES.get(dtype, dtype)}{[str(d) for d in shape]}".replace("'", "")


def load_weights(path: str, into, *, device="cpu", mmap: bool = True, strict: bool = True, cast: bool = False,
                 verify: bool = True, assign: bool = False) -> LoadedWeights:
    """Load ``path`` into a native function, an IR program or a module, after checking every tensor.

    strict: missing or unexpected tensors are errors (for a function only unexpected ones: the
    parameters not in the file are its data inputs). cast: allow a dtype conversion. verify: rerun
    the reference record if the file has one. assign: a module takes the loaded tensors themselves
    (zero copy from the memory map) instead of copying into its parameters."""
    rep = LoadReport(path)
    tensors = load_file(path, device=device, mmap=mmap, report=rep)
    meta = rep.metadata
    kind, targets = _targets(into)
    problems = []
    for n in sorted(set(tensors) - set(targets)):
        problems.append(f"{n}: in the file but not in the {kind} ({_fmt(tensors[n].dtype, tensors[n].shape)})")
    missing = sorted(set(targets) - set(tensors))
    tied: Dict[str, str] = {}
    if kind == "module":   # tied weights (an lm head sharing the embedding): one name in the file is enough
        sd = into.state_dict(keep_vars=True)
        by_ptr: Dict[Tuple[int, int, tuple], List[str]] = {}
        for k, v in sd.items():
            by_ptr.setdefault((v.data_ptr(), v.storage_offset(), tuple(v.shape)), []).append(k)
        for names in by_ptr.values():
            present = [n for n in names if n in tensors]
            for n in names:
                if n not in tensors and present and sd[n].numel():
                    tied[n] = present[0]
        missing = [n for n in missing if n not in tied]
    if kind != "function" and strict:
        for n in missing:
            problems.append(f"{n}: the {kind} needs {_fmt(*targets[n])}, the file does not have it")
    bind: Dict[str, Tuple[int, str]] = {}
    for n in sorted(set(tensors) & set(targets)):
        t, (dt, shape) = tensors[n], targets[n]
        if t.dtype != dt and not cast:
            problems.append(f"{n}: dtype {NAMES.get(t.dtype)} in the file, {NAMES.get(dt, dt)} in the {kind} "
                            "(pass cast=True to convert)")
        if len(shape) != t.dim():
            problems.append(f"{n}: shape {list(t.shape)} in the file, {_fmt(dt, shape)} in the {kind}")
            continue
        for i, (d, s) in enumerate(zip(shape, t.shape)):
            if isinstance(d, int) or getattr(d, "is_Integer", False):
                if int(d) != s:
                    problems.append(f"{n}: shape {list(t.shape)} in the file, {_fmt(dt, shape)} in the {kind} "
                                    f"(dim {i})")
                    break
            elif isinstance(d, sympy.Symbol):
                sym = str(d)
                if sym in bind and bind[sym][0] != s:
                    problems.append(f"{n}: dim {i} is {sym} = {s}, but {bind[sym][1]} has {sym} = {bind[sym][0]}")
                bind.setdefault(sym, (int(s), n))
    if problems:
        raise SafetensorsError("safetensors_mismatch", f"{path} does not fit the {kind} ({len(problems)} problem(s), "
                               "nothing was loaded):", problems)
    if cast:
        tensors = {n: (t.to(targets[n][0]) if n in targets and t.dtype != targets[n][0] else t) for n, t in tensors.items()}
    out = LoadedWeights(tensors, missing if kind == "function" else [], rep, None)
    if kind == "program":
        from ..ir.core import Program
        data = dict(into.param_data)
        data.update(tensors)
        out.program = Program(into.name, list(into.inputs), list(into.params), data, list(into.nodes),
                              list(into.outputs), dict(into.symbols))
    spec = meta.get(CHECK_KEY)
    if verify and spec is not None:
        target = out.program if kind == "program" else into
        if kind == "program":
            raise SafetensorsError("safetensors_check_unsupported", "reference records are checked for functions "
                                   "and modules; this file's record cannot run on an IR program",
                                   suggestion="pass verify=False, or load into the function or module")
        out.check_errors = run_check(target, tensors, json.loads(spec))
    if kind == "module":
        res = into.load_state_dict(tensors, strict=False, assign=assign)
        left = sorted(set(res.missing_keys) - set(tied))
        if strict and (left or res.unexpected_keys):   # checked above, so this is an internal error
            raise SafetensorsError("safetensors_mismatch", "load_state_dict disagreed with the check",
                                   left + list(res.unexpected_keys))
        out.tied = tied
    return out
