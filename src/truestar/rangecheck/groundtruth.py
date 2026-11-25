"""Ground truth for measuring `truestar range`: PyTorch's own low precision execution, op by op.

This does not use the IR, the emulator or any Truestar rounding. The model is run twice:

1. In float64, recording every ATen op's outputs.
2. As the low precision model (``model.to(dtype)``, PyTorch CPU kernels), where every op is fed the
   float64 outputs of the ops before it, cast to the dtype that op really receives (values past the
   format's range saturate, as an earlier op that overflowed is its own site). An op whose output then
   has NaN or infinity where float64 is finite is an overflow site: it overflows on its own, with
   correct inputs. Feeding correct inputs is what lets every site show up; in a plain chained run
   only the first one does, because everything after it is already NaN.

Casts that only one run performs (PyTorch upcasts RMS norm internals of a half input to float32)
are skipped when the two op sequences are aligned. With ``decompose_norms`` layer norms are executed
as the ONNX decomposition (ReduceMean, Sub, Pow, ReduceMean, Add, Sqrt, Div, Mul, Add) in the low
precision dtype, as an unfused WebGPU graph runs them; each step is checked.

``chained_run`` is the baseline people use today: run the whole model in low precision and compare
the final output (cosine), which cannot name an op.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

_CASTS = ("aten._to_copy.default", "aten.to.dtype", "aten.to.dtype_layout")
_LN = ("aten.native_layer_norm.default",)


@dataclass
class GroundSite:
    module: str
    op: str
    count: int
    inputs_in_range: bool = True  # False: an input was already past the range (saturated here)

    def family(self) -> str:
        return op_family(self.op)


@dataclass
class GroundTruth:
    sites: List[GroundSite]
    ops: int
    misaligned: int
    unsupported: List[str] = field(default_factory=list)


def op_family(name: str) -> str:
    """Coarse op family shared by ATen names and IR ops, for matching sites across the two."""
    n = name.replace("aten.", "").split(".")[0].lstrip("_")
    if n.startswith("layer_norm") or n == "native_layer_norm":
        return "layer_norm"
    if n in ("linear", "addmm", "mm", "bmm", "matmul", "baddbmm"):
        return "matmul"
    if "rms_norm" in n:
        return "rms_norm"
    if "scaled_dot_product" in n or n == "attention":
        return "attention"
    if n in ("add", "add_"):
        return "add"
    return n


class _ModuleStack:
    def __init__(self, model):
        self.stack: List[str] = []
        self.handles = []
        for name, mod in model.named_modules():
            self.handles.append(mod.register_forward_pre_hook(self._push(name)))
            self.handles.append(mod.register_forward_hook(self._pop))

    def _push(self, name):
        def hook(mod, args):
            self.stack.append(name)
        return hook

    def _pop(self, mod, args, out):
        self.stack.pop()

    def top(self) -> str:
        return self.stack[-1] if self.stack else ""

    def remove(self):
        for h in self.handles:
            h.remove()


class _Record(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops: List[Tuple[str, List[torch.Tensor]]] = []

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        out = func(*args, **(kwargs or {}))
        flat, _ = tree_flatten(out)
        self.ops.append((str(func), [t.detach().clone() if isinstance(t, torch.Tensor) else None for t in flat]))
        return out


def _sat_cast(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype.is_floating_point and t.dtype.is_floating_point:
        fmax = torch.finfo(dtype).max
        return t.clamp(-fmax, fmax).to(dtype)
    return t.to(dtype)


def _bad(lp: torch.Tensor, ref: Optional[torch.Tensor]) -> int:
    if ref is None or not lp.dtype.is_floating_point or lp.shape != ref.shape:
        return 0
    return int((~torch.isfinite(lp) & torch.isfinite(ref)).sum())


class _Replay(TorchDispatchMode):
    def __init__(self, ref_ops, stack: _ModuleStack, decompose_norms: bool):
        super().__init__()
        self.saturated = False
        self.ref = ref_ops
        self.i = 0
        self.src: Dict[int, Tuple[int, int]] = {}
        self.keep: List[torch.Tensor] = []
        self.stack = stack
        self.decompose = decompose_norms
        self.sites: List[GroundSite] = []
        self.misaligned = 0
        self.ops = 0
        self.unsupported: List[str] = []

    def _sub(self, a):
        if isinstance(a, torch.Tensor) and id(a) in self.src:
            j, k = self.src[id(a)]
            r = self.ref[j][1][k]
            if r is not None and r.shape == a.shape:
                if a.dtype.is_floating_point and r.dtype.is_floating_point:
                    fin = r[torch.isfinite(r)]
                    if fin.numel() and float(fin.abs().max()) > torch.finfo(a.dtype).max:
                        self.saturated = True
                return _sat_cast(r, a.dtype)
        return a

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        # align: casts only one side performs are skipped
        while self.i < len(self.ref) and self.ref[self.i][0] != name and self.ref[self.i][0] in _CASTS:
            self.i += 1
        aligned = self.i < len(self.ref) and self.ref[self.i][0] == name
        if name in _CASTS and not aligned:
            out = func(*args, **kwargs)
            src = args[0] if args and isinstance(args[0], torch.Tensor) else None
            if src is not None and id(src) in self.src:
                self.src[id(out)] = self.src[id(src)]
                self.keep.append(out)
            return out
        if not aligned:
            self.misaligned += 1
            return func(*args, **kwargs)
        idx = self.i
        self.i += 1
        self.ops += 1
        self.saturated = False
        sargs = tree_map(self._sub, args)
        skw = tree_map(self._sub, kwargs)
        refs = self.ref[idx][1]
        try:
            if self.decompose and name in _LN:
                out = self._layer_norm(sargs, refs)
            else:
                out = func(*sargs, **skw)
        except Exception as exc:  # an op PyTorch has no low precision CPU kernel for
            self.unsupported.append(f"{name}: {type(exc).__name__}")
            out = func(*args, **kwargs)
        flat, _ = tree_flatten(out)
        if flat and isinstance(flat[0], torch.Tensor) and not (self.decompose and name in _LN):
            n = _bad(flat[0], refs[0] if refs else None)
            if n:
                self.sites.append(GroundSite(self.stack.top(), name, n, not self.saturated))
        for k, t in enumerate(flat):
            if isinstance(t, torch.Tensor):
                self.src[id(t)] = (idx, k)
                self.keep.append(t)
        return out

    def _layer_norm(self, args, refs):
        x, shape, w, b, eps = args
        dims = tuple(range(x.dim() - len(shape), x.dim()))
        mod = self.stack.top()
        steps = []
        mu = x.mean(dims, keepdim=True)
        d = x - mu
        sq = d.pow(2)
        steps.append(("layer_norm.pow", sq))
        var = sq.mean(dims, keepdim=True)
        steps.append(("layer_norm.reducemean", var))
        sd = (var + torch.tensor(eps, dtype=x.dtype)).sqrt()
        y = d / sd
        if w is not None:
            y = y * w
        if b is not None:
            y = y + b
        for nm, t in steps:
            n = int((~torch.isfinite(t)).sum())
            if n and bool(torch.isfinite(x).all()):
                self.sites.append(GroundSite(mod, nm, n, not self.saturated))
                break
        return (y, mu.float(), (1.0 / sd).float())


def ground_truth(model: torch.nn.Module, inputs: Sequence, dtype: torch.dtype = torch.float16,
                 decompose_norms: bool = False) -> GroundTruth:
    """Every op that overflows on its own in PyTorch's ``dtype`` execution of ``model``."""
    inputs = tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,)
    m64 = copy.deepcopy(model).double().eval()
    rec = _Record()
    with torch.no_grad(), rec:
        m64(*[t.double() if t.dtype.is_floating_point else t for t in inputs])
    mlp = copy.deepcopy(model).to(dtype).eval()
    stack = _ModuleStack(mlp)
    rep = _Replay(rec.ops, stack, decompose_norms)
    try:
        with torch.no_grad(), rep:
            mlp(*[_sat_cast(t, dtype) if t.dtype.is_floating_point else t for t in inputs])
    finally:
        stack.remove()
    return GroundTruth(rep.sites, rep.ops, rep.misaligned, rep.unsupported)


class _Decompose(TorchDispatchMode):
    """Layer norms run as their ONNX decomposition in the low precision dtype (chained, unchecked)."""

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        if str(func) in _LN and args[0].dtype in (torch.float16, torch.bfloat16):
            x, shape, w, b, eps = args
            dims = tuple(range(x.dim() - len(shape), x.dim()))
            mu = x.mean(dims, keepdim=True)
            d = x - mu
            sd = (d.pow(2).mean(dims, keepdim=True) + torch.tensor(eps, dtype=x.dtype)).sqrt()
            y = d / sd
            y = y * w if w is not None else y
            y = y + b if b is not None else y
            return (y, mu.float(), (1.0 / sd).float())
        return func(*args, **(kwargs or {}))


def chained_run(model: torch.nn.Module, inputs: Sequence, dtype: torch.dtype = torch.float16,
                decompose_norms: bool = False) -> Dict[str, float]:
    """The baseline: run the model in ``dtype`` end to end and compare the output with float64."""
    import contextlib

    inputs = tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,)
    with torch.no_grad():
        ref = copy.deepcopy(model).double().eval()(*[t.double() if t.dtype.is_floating_point else t for t in inputs])
        with (_Decompose() if decompose_norms else contextlib.nullcontext()):
            out = copy.deepcopy(model).to(dtype).eval()(*[_sat_cast(t, dtype) if t.dtype.is_floating_point else t
                                                          for t in inputs])
    ref = ref.reshape(-1, ref.shape[-1]).double()
    out = out.reshape(-1, out.shape[-1]).double()
    fin = torch.isfinite(out).all(1)
    o = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    cos = (ref * o).sum(1) / (ref.norm(dim=1) * o.norm(dim=1)).clamp_min(1e-300)
    cos = torch.where(fin, cos, torch.full_like(cos, -1.0))
    return {"cosine_min": float(cos.min()), "cosine_mean": float(cos.mean()),
            "nonfinite": int((~torch.isfinite(out)).sum()),
            "flips": int((ref.argmax(1) != torch.where(torch.isfinite(out), out, torch.full_like(out, -math.inf)).argmax(1)).sum())}


def match(predicted: Sequence, truth: Sequence[GroundSite]) -> Dict[str, object]:
    """Match predicted sites (rangecheck Site objects of kind overflow or hidden_overflow) with
    ground truth sites by module path and op family."""
    def pkey(s):
        fam = op_family(s.where.aten or s.op) if not s.hidden else op_family(s.op)
        return (s.where.module, fam)

    def tkey(g):
        fam = g.op.split(".")[0] if g.op.startswith("layer_norm.") else g.family()
        return (g.module, fam)

    pk = {}
    for s in predicted:
        pk.setdefault(pkey(s), []).append(s)
    tk = {}
    for g in truth:
        tk.setdefault(tkey(g), []).append(g)
    found = missed_n = extra_n = 0
    missed, extra = [], []
    for k in sorted(set(pk) | set(tk)):
        p, t = len(pk.get(k, [])), len(tk.get(k, []))
        found += min(p, t)
        if t > p:
            missed_n += t - p
            missed.append(k)
        if p > t:
            extra_n += p - t
            extra.append(k)
    return {"true_sites": len(truth), "predicted": len(predicted), "found": found,
            "missed": missed_n, "false_positives": extra_n, "missed_keys": missed, "extra_keys": extra}
