"""`truestar record`: run a model once and store what every module saw and produced.

Per module call the recording holds the inputs (copied before forward, so in place ops cannot
change them), the outputs, a hash of the module's parameters and buffers, and a hash of the
inputs. On request it also holds an ATen op trace for chosen modules: every op's inputs and
outputs, in order, with the module path it ran under. The environment fingerprint
(fingerprint.py) is stored next to it.

`outputs_only=True` keeps only module outputs, input hashes and the model's own inputs, for a
much smaller file. Such a recording can still be diffed: a module whose inputs hash the same in
both runs is isolated exactly, because everything before it was bitwise identical.
"""

from __future__ import annotations

import copy
import hashlib
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch
from torch import nn

from ..capture import ROOT, Call, Recorder, flatten_tensors
from ..opdrill import _comparable, _cpu_copy, _leaves
from . import store as S
from .fingerprint import fingerprint

DEFAULT_MAX_BYTES = 1 << 30  # 1 GiB
MAX_TRACED_OPS = 50000


class RecordTooLarge(Exception):
    """The recording would exceed the size bound (code TS070)."""


def _model_dtype(model: nn.Module) -> torch.dtype:
    from ..check import _model_dtype as md

    return md(model)


def state_hashes(model: nn.Module) -> Dict[str, str]:
    """Hash of each module's parameters and buffers (its whole subtree), keyed by module name."""
    per: Dict[str, str] = {}
    for full, t in list(model.named_parameters()) + list(model.named_buffers()):
        per[full] = S.tensor_hash(t.detach())
    out: Dict[str, str] = {}
    for name, _ in model.named_modules():
        prefix = name + "." if name else ""
        h = hashlib.sha256()
        for full in sorted(k for k in per if k.startswith(prefix)):
            h.update(f"{full[len(prefix):]}={per[full]};".encode())
        out[name or ROOT] = h.hexdigest()[:32]
    return out


class AliasRecorder(Recorder):
    """capture.Recorder whose input copies keep tensor identity: a tensor passed twice is one
    copied object, so the recording knows `attn(x, x, x)` from `attn(x, y, z)`."""

    def _pre(self, name: str, leaf: bool):
        def hook(module, args, kwargs):
            self.in_hook = True
            try:
                idx = self._counts.get(name, 0)
                self._counts[name] = idx + 1
                parent = self._stack[-1] if self._stack else None
                call = Call(
                    key=f"{name}#{idx}", name=name, type_name=type(module).__name__, index=idx,
                    start=self._starts, depth=len(self._stack),
                    parent=parent.key if parent else None, is_leaf=leaf,
                )
                self._starts += 1
                if self.keep_inputs:
                    memo: Dict[int, Any] = {}
                    call.args = S.alias_map(args, _cpu_copy, memo)
                    call.kwargs = S.alias_map(kwargs, _cpu_copy, memo)
                if parent is not None:
                    parent.children.append(call.key)
                self._stack.append(call)
            finally:
                self.in_hook = False
            return None

        return hook


class _OpTracer:
    """Dispatch mode that records ATen ops run under selected modules (see Recorder stack)."""

    def __init__(self, rec: Recorder, select: Optional[set]):
        from torch.utils._python_dispatch import TorchDispatchMode

        tracer = self
        self.ops: List[Dict[str, Any]] = []
        self.truncated = False

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                stack = rec._stack
                if rec.in_hook or not stack or (
                        select is not None and not any(c.name in select for c in stack)):
                    return func(*args, **kwargs)
                if len(tracer.ops) >= MAX_TRACED_OPS:
                    tracer.truncated = True
                    return func(*args, **kwargs)
                memo: Dict[int, Any] = {}
                a = S.alias_map(args, _cpu_copy, memo)
                k = S.alias_map(kwargs, _cpu_copy, memo)
                out = func(*args, **kwargs)
                tracer.ops.append({"func": func, "path": [c.key for c in stack], "args": a,
                                   "kwargs": k, "outputs": [_cpu_copy(t) for t in _leaves(out)]})
                return out

        self.mode = Mode()


class Recording:
    """A recording in memory: the header (calls, env, ops) and the tensor blobs."""

    def __init__(self, header: Dict[str, Any], entries: List[Dict[str, Any]], blob_of,
                 raw: bytes, path: Optional[str] = None):
        self.header = header
        self.entries = entries
        self._blob_of = blob_of
        self.raw = raw
        self.path = path
        self._cache: Dict[int, torch.Tensor] = {}
        self.calls: List[Dict[str, Any]] = header["calls"]
        self.by_key = {c["key"]: c for c in self.calls}
        self.ops: List[Dict[str, Any]] = header.get("ops_trace", [])

    # -- access
    @property
    def env(self) -> Dict[str, Any]:
        return self.header["env"]

    @property
    def dtype(self) -> torch.dtype:
        return S.dtype_from(self.header["model"]["dtype"])

    @property
    def device(self) -> str:
        return self.header["device"]

    @property
    def outputs_only(self) -> bool:
        return self.header["mode"] == "outputs_only"

    def tensor(self, i: int) -> torch.Tensor:
        if i not in self._cache:
            self._cache[i] = self._blob_of(i)
        return self._cache[i]

    def outputs(self, call: Dict[str, Any]) -> List[torch.Tensor]:
        return [self.tensor(i) for i in call["out"]]

    def inputs(self, call: Dict[str, Any]):
        """(args, kwargs) as CPU tensors; raises store.Unreplayable when not stored."""
        return S.decode_call(call["args"], call["kwargs"], self.tensor)

    def ops_under(self, key: str) -> List[Dict[str, Any]]:
        return [o for o in self.ops if key in o["path"]]

    # -- file
    def save(self, path: str) -> str:
        with open(path, "wb") as f:
            f.write(self.raw)
        self.path = path
        return path

    @property
    def nbytes(self) -> int:
        return sum(e["nbytes"] for e in self.entries)


def load(path: str) -> Recording:
    if not os.path.isfile(path):
        raise S.RecordError(f"{path} not found")
    with open(path, "rb") as f:
        raw = f.read()
    header, blobs = S.loads(raw, path)
    entries = header["blobs"]
    return Recording(header, entries, lambda i: S.blob_tensor(entries[i], blobs), raw=raw,
                     path=os.path.abspath(path))


def _from_store(header: Dict[str, Any], st: S.BlobStore) -> Recording:
    raw = S.dumps(header, st)
    h, blobs = S.loads(raw)
    return Recording(h, h["blobs"], lambda i: S.blob_tensor(h["blobs"][i], blobs), raw=raw)


def record(
    model: nn.Module,
    inputs: Union[torch.Tensor, Sequence[Any]],
    kwargs: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    outputs_only: bool = False,
    ops: Union[None, str, Iterable[str]] = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    model_name: Optional[str] = None,
    model_file: Optional[str] = None,
) -> Recording:
    """Run `model` once on `device` and return the recording (save it with .save(path)).

    ops: None (no op traces), "all", or module names whose ATen ops are traced (with everything
    they call). The model is copied, never modified, and run in eval mode without gradients.
    """
    from ..check import _cast_inputs, _sync

    args = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    kwargs = dict(kwargs or {})
    dtype = _model_dtype(model)
    base = copy.deepcopy(model).cpu().eval()
    states = state_hashes(base)
    m = copy.deepcopy(base).to(device)
    select: Optional[set]
    if ops is None or ops == "none":
        select, ops_label = set(), "none"
    elif ops == "all":
        select, ops_label = None, "all"
    else:
        names = [n for n in ([ops] if isinstance(ops, str) else list(ops)) if n]
        known = {n or ROOT for n, _ in base.named_modules()}
        missing = [n for n in names if n not in known]
        if missing:
            raise ValueError(f"--ops names modules that do not exist: {', '.join(missing)}")
        select, ops_label = set(names), sorted(names)

    dev_args = _cast_inputs(args, device, dtype)
    dev_kwargs = _cast_inputs(kwargs, device, dtype)
    # The plain run, with no hooks: what production computes. Forward hooks switch off fused
    # fast paths (TransformerEncoderLayer checks for them), so the instrumented run below can
    # take another code path; the header says whether the two agree bitwise.
    plain_args, plain_kwargs = S.alias_map((dev_args, dev_kwargs), torch.clone)
    with torch.no_grad():
        plain = [t.detach().cpu() for t in flatten_tensors(m(*plain_args, **plain_kwargs))]
    _sync(device)
    with torch.no_grad(), AliasRecorder(m, keep_inputs=True, count_ops=True) as rec:
        tracer = _OpTracer(rec, select) if select != set() else None
        if tracer is not None:
            with tracer.mode:
                out = m(*dev_args, **dev_kwargs)
        else:
            out = m(*dev_args, **dev_kwargs)
    _sync(device)
    del out

    st = S.BlobStore()
    calls = []
    for c in rec.calls:
        keep = not outputs_only or c.name == ROOT
        a_h, k_h = S.encode_call(c.args, c.kwargs, None)
        if keep:
            a_t, k_t = S.encode_call(c.args, c.kwargs, st)
        calls.append({
            "key": c.key, "name": c.name, "type": c.type_name, "index": c.index,
            "start": c.start, "depth": c.depth, "parent": c.parent, "leaf": c.is_leaf,
            "own_ops": c.own_ops, "children": list(c.children),
            "state": states.get(c.name, ""),
            "in_hash": input_hash(a_h, k_h),
            "args": a_t if keep else a_h,
            "kwargs": k_t if keep else k_h,
            "out": [st.add(t) for t in c.outputs],
        })
        if st.nbytes > max_bytes:
            raise RecordTooLarge(_too_large(st.nbytes, max_bytes, outputs_only))
    root = next(c for c in rec.calls if c.name == ROOT)
    plain_idx = [st.add(t) for t in plain]
    same_path = S.hashes(plain) == S.hashes(root.outputs)
    op_list = []
    if tracer is not None:
        for o in tracer.ops:
            func = o["func"]
            if not o["outputs"] or not _comparable(_Shim(func, o["outputs"])):
                continue
            a_h, k_h = S.encode_call(o["args"], o["kwargs"], None)
            if not outputs_only:
                a_t, k_t = S.encode_call(o["args"], o["kwargs"], st)
            op_list.append({
                "name": str(func), "path": o["path"],
                "in_hash": input_hash(a_h, k_h),
                "args": a_h if outputs_only else a_t,
                "kwargs": k_h if outputs_only else k_t,
                "out": [st.add(t) for t in o["outputs"]],
            })
            if st.nbytes > max_bytes:
                raise RecordTooLarge(_too_large(st.nbytes, max_bytes, outputs_only))
    header = {
        "format": S.FORMAT, "version": S.VERSION,
        "truestar_version": _version(),
        "model": {"name": model_name or type(model).__name__, "class": type(model).__qualname__,
                  "file": os.path.abspath(model_file) if model_file else None,
                  "dtype": S.dtype_name(dtype)},
        "device": str(device),
        "mode": "outputs_only" if outputs_only else "full",
        "ops": ops_label,
        "ops_truncated": bool(tracer.truncated) if tracer is not None else False,
        "op_counting": bool(rec.op_counting),
        "env": fingerprint(),
        "plain_out": plain_idx,
        "instrumented_matches_plain": same_path,
        "calls": calls,
        "ops_trace": op_list,
    }
    return _from_store(header, st)


class _Shim:
    """Just enough of an OpRecord for opdrill._comparable."""

    def __init__(self, func, outputs):
        self.func = func
        self.outputs = outputs


def input_hash(args_tree, kwargs_tree) -> str:
    """Hash of call arguments encoded with store=None (tensors as content hashes)."""
    return S.tree_hash({"args": args_tree, "kwargs": kwargs_tree}, [])


def _too_large(n: int, limit: int, outputs_only: bool) -> str:
    msg = (f"the recording would be over {n / 2**20:.0f} MiB, above the limit of "
           f"{limit / 2**20:.0f} MiB.")
    if not outputs_only:
        msg += " Use --outputs-only to store only module outputs and hashes,"
    return msg + " use smaller inputs, trace fewer ops, or raise --max-mb."


def _version() -> str:
    from .. import __version__

    return __version__
