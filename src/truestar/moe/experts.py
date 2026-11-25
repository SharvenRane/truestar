"""Expert computation for routed mixture of experts layers: one grouped form, one static form.

Both compute, for every token t and each of its k routed slots j (expert ``e = idx[t, j]``,
weight ``w[t, j]``)::

    y[t, j] = down[e] @ (act(gate[e] @ x[t]) * (up[e] @ x[t]))      gate_up[e] = [gate[e]; up[e]]
    out[t]  = sum_j w[t, j] * y[t, j]

* ``grouped_experts`` (eager, any device): one stable sort of the (token, slot) pairs by expert
  gathers each expert's tokens into one contiguous block, one matmul pair per expert runs the block,
  and a permutation (a pure gather, no atomic adds) puts the results back in slot order. The
  combine is a fixed sum over slots j = 0, 1, ..., k-1 (best routed expert first). Nothing depends
  on thread scheduling, so two runs give the same bits, forward and backward (the only gradient
  accumulation, over a token's k slots, is a plain reduction). The first call per shape, dtype and
  device is checked against the same math in float64 on the CPU (``TRUESTAR_VERIFY`` as for every
  Truestar kernel); a mismatch raises ``MoEMismatchError`` (TS1181).
* ``static_experts`` (for the importer): every expert runs on every token and a mask keeps the
  routed ones, ``out = sum_e where(routed(t, e), w(t, e) * y_e(t), 0)`` in expert order. It has no
  data dependent shapes, so torch.export traces it into ops the IR already verifies (float64
  reference, gradients, compile). It is exact against the grouped form up to the order of the sum
  (expert order here, slot order there); unrouted experts add an exact zero, never ``0 * y``, so an
  overflowing unrouted expert cannot leak a NaN. It costs E / k times the expert FLOPs.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from ..kernel import verify as KV


class MoEMismatchError(KV.KernelMismatchError):
    """The grouped experts disagree with the float64 computation (TS1181). Never silently ignored."""

    code = "TS1181"


STATS = {"checks": 0}
_verified: set = set()
_last: Dict[str, object] = {}


def last_call() -> Dict[str, object]:
    return dict(_last)


def reset() -> None:
    _verified.clear()
    _last.clear()


def _apply(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, act: Callable) -> torch.Tensor:
    gate, up = F.linear(x, gate_up).chunk(2, dim=-1)
    return F.linear(act(gate) * up, down)


def grouped_experts_unchecked(x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, gate_up: torch.Tensor,
                              down: torch.Tensor, act: Callable) -> torch.Tensor:
    """The grouped computation without the float64 check (``grouped_experts`` adds it)."""
    T, H = x.shape
    k = idx.shape[1]
    E = gate_up.shape[0]
    flat = idx.reshape(-1)
    # (token, slot) pairs grouped by expert; within an expert in (token, slot) order
    order = torch.sort(flat, stable=True).indices
    counts = torch.bincount(flat, minlength=E).tolist()
    # every token repeated k times (a broadcast, whose gradient is a plain sum over the k slots)
    xs = x.unsqueeze(1).expand(T, k, H).reshape(T * k, H).index_select(0, order)
    blocks = []
    start = 0
    for e, n in enumerate(counts[:E]):
        if n:
            blocks.append(_apply(xs.narrow(0, start, n), gate_up[e], down[e], act))
        start += n
    if start != T * k:
        raise ValueError(f"[TS1180] expert indices outside 0..{E - 1}: {T * k - start} of {T * k} slots")
    ys = torch.cat(blocks, 0) if blocks else xs.new_zeros(0, down.shape[1])
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    y = ys.index_select(0, inv).view(T, k, -1)          # slot order again: a permutation, no adds
    out = y[:, 0] * w[:, 0, None]
    for j in range(1, k):
        out = out + y[:, j] * w[:, j, None]
    return out.to(x.dtype)


def reference_experts(x, idx, w, gate_up, down, act) -> torch.Tensor:
    """The same math in float64 on the CPU, one token slot at a time (the check's reference)."""
    x, w, gu, dn = (KV.to_host_f64(t) for t in (x, w, gate_up, down))
    idx = idx.detach().to("cpu")
    out = torch.zeros(x.shape[0], dn.shape[1], dtype=torch.float64)
    for j in range(idx.shape[1]):
        for e in idx[:, j].unique().tolist():
            rows = (idx[:, j] == e).nonzero().squeeze(1)
            out[rows] += w[rows, j, None] * _apply(x[rows], gu[e], dn[e], act)
    return out


def tolerance(dtype: torch.dtype, H: int, inter: int, k: int, tf32: bool = False) -> float:
    """Per element tolerance: the kernel layer's base for the dtype, TF32 when allowed, grown with
    log2 of the longest dot product chain (hidden, then intermediate width, then k slots)."""
    name = {torch.float64: "float64", torch.float32: "float32", torch.float16: "float16",
            torch.bfloat16: "bfloat16"}[dtype]
    base = KV.TF32_TOL if tf32 and name == "float32" else KV.BASE_TOL[name]
    return base * max(1.0, math.log2(max(2, H) * max(2, inter) * max(2, k)) / 4.0)


def _tf32(x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype == torch.float32 and (
        torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != "highest")


def grouped_experts(x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, gate_up: torch.Tensor,
                    down: torch.Tensor, act: Callable, name: str = "experts") -> torch.Tensor:
    """Grouped, bitwise reproducible routed experts (see the module docstring); checked against
    float64 on the first call per signature."""
    out = grouped_experts_unchecked(x, idx, w, gate_up, down, act)
    sig = (tuple(x.shape), tuple(idx.shape), tuple(gate_up.shape), x.dtype, w.dtype, str(x.device))
    mode = KV.verify_mode()
    checked = mode == "all" or (mode == "first" and sig not in _verified)
    _last.clear()
    _last.update(impl="grouped", checked=checked, signature=sig)
    if checked:
        STATS["checks"] += 1
        with torch.no_grad():
            ref = reference_experts(x, idx, w, gate_up, down, act)
        tol = tolerance(x.dtype, x.shape[1], down.shape[2], idx.shape[1], _tf32(x))
        c = KV.compare_torch("out", out.detach().cpu(), ref, tol, KV.FLOOR_REDUCE)
        _last["error"] = c.error
        if not c.ok:
            raise MoEMismatchError(
                f"[TS1181] {name}: grouped experts differ from float64 by {c.error:.3e} at {c.at} (tolerance "
                f"{tol:.3e}); tokens {x.shape[0]}, experts {gate_up.shape[0]}, k {idx.shape[1]}, {x.dtype} on "
                f"{x.device}.")
        _verified.add(sig)
    return out


def static_experts(x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, gate_up: torch.Tensor,
                   down: torch.Tensor, act: Callable) -> torch.Tensor:
    """Every expert on every token, the routed ones kept by a mask (the importer's form)."""
    out = None
    for e in range(gate_up.shape[0]):
        hit = idx == e                                               # [T, k]; at most one slot per token
        we = torch.where(hit, w, torch.zeros_like(w)).sum(-1, keepdim=True)   # exact: one term or none
        y = _apply(x, gate_up[e], down[e], act)
        c = torch.where(hit.any(-1, keepdim=True), y * we, torch.zeros_like(y)).to(x.dtype)  # as eager: per expert
        out = c if out is None else out + c
    return out.to(x.dtype)
