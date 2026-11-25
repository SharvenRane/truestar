"""``linear_attention``: the verified chunked recurrence as one differentiable op.

* Implementation: the Triton kernels (triton_kernels.py) for CUDA tensors whose decay is none or
  scalar; the chunked PyTorch form (chunked.py) for CPU tensors and per key channel decay. The
  choice is never silent: ``impl="triton"`` refuses (TS1061) what the kernels do not run,
  ``impl="chunked"`` forces the PyTorch form, ``impl=None`` picks the kernels when they run the
  call, and ``last_call()`` names the implementation, the launch config and whether this call was
  checked.
* Verification: the first call per shape, dtype, rule, decay kind and launch config in a process
  runs the float64 sequential recurrence (reference.py) on the same device and compares the
  output and the final state per element (``kernel.verify.compare_torch``, the reduction floor).
  A mismatch raises ``LinAttnMismatchError`` (TS1060). ``TRUESTAR_VERIFY=all`` checks every call,
  ``0`` none, as for every Truestar kernel.
* Backward: the gradient is the autograd gradient of the chunked PyTorch form at the same chunk
  size, with the state in float32 (float64 for float64 inputs). The first backward per signature
  is checked against the float64 gradient of the same form, which tests/test_linattn.py shows
  equals the gradient of the sequential recurrence to about 1e-13.
* ``deterministic=True`` (or None, following ``torch.are_deterministic_algorithms_enabled()``)
  pins the launch config (triton_kernels.config): the result depends on the inputs only, not on
  the GPU's SM count, the batch or anything the caller tunes.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from ..kernel import verify as KV
from . import metal_kernels as MK
from . import triton_kernels as TK
from .chunked import chunked
from .reference import LinAttnError, check_args, decay_kind, recurrent


class LinAttnMismatchError(KV.KernelMismatchError):
    """The chunked kernel disagrees with the float64 recurrence (TS1060). Never silently ignored."""

    code = "TS1060"


_verified: set = set()
_last: Dict[str, object] = {}
STATS = {"checks": 0, "grad_checks": 0}


def last_call() -> Dict[str, object]:
    """What the last ``linear_attention`` call ran: impl, config, checked, errors."""
    return dict(_last)


def reset() -> None:
    _verified.clear()
    _last.clear()


def tolerance(dtype: torch.dtype, T: int, dk: int, chunk: int, allow_tf32: bool = False) -> float:
    """Per element tolerance against float64: the kernel layer's base for the output dtype (float32
    arithmetic inside, so 16 bit outputs get their own base), TF32 when allowed, grown with log2 of
    the longest dependency chain (the T steps times a key width of sums) as the kernel layer grows
    it with a reduction's length."""
    name = {torch.float64: "float64", torch.float32: "float32", torch.float16: "float16",
            torch.bfloat16: "bfloat16"}[dtype]
    base = KV.BASE_TOL[name]
    if name == "float64":
        base = KV.BASE_TOL["float32"]  # the kernels compute in float32 whatever the inputs
    if allow_tf32 and name in ("float32", "float64"):
        base = KV.TF32_TOL
    return base * max(1.0, math.log2(max(2, T) * max(2, dk) * max(2, chunk)) / 4.0)


def _signature(impl, q, k, v, g, beta, rule, s0, cfg):
    def d(x):
        return None if x is None else (tuple(x.shape), str(x.dtype), str(x.device))
    return (impl, rule, d(q), d(k), d(v), d(g), d(beta), d(s0), tuple(sorted((cfg or {}).items())))


def _check(name, o, S, q, k, v, g, beta, rule, scale, s0, tol, cfg, impl):
    """Compare o and the final state with the float64 recurrence; raise on a mismatch."""
    STATS["checks"] += 1
    if q.device.type == "mps":
        # the MPS device has no float64: the reference runs on the CPU, as the Metal kernels' does.
        # Copied first, then converted: MPS to("cpu", float64) in one call returns zeros on torch
        # 2.14 (docs/METAL.md); here the copies stay float32 until recurrent converts them.
        q, k, v, g, beta, s0, o, S = (None if t is None else t.cpu() for t in (q, k, v, g, beta, s0, o, S))
    with torch.no_grad():
        o_ref, s_ref = recurrent(q, k, v, g, beta, rule=rule, scale=scale, initial_state=s0)
    errors = {}
    for what, out, ref in (("o", o, o_ref), ("final_state", S, s_ref)):
        if out is None:
            continue
        c = KV.compare_torch(what, out, ref, tol, KV.FLOOR_REDUCE)
        errors[what] = c.error
        if not c.ok:
            raise LinAttnMismatchError(
                f"[TS1060] {name} ({impl}, rule {rule}, decay {decay_kind(g, k)}, config {cfg}): output "
                f"{what} differs from the float64 recurrence by {c.error:.3e} at {c.at} (tolerance "
                f"{tol:.3e}){' ' + c.detail if c.detail else ''}. Shapes q {tuple(q.shape)}, v {tuple(v.shape)}.")
    return errors


class _KernelFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, s0, rule, scale, cfg, allow_tf32):
        o, S = TK.forward(q, k, v, g, beta, rule=rule, scale=scale, initial_state=s0, cfg=cfg,
                          allow_tf32=allow_tf32)
        ctx.save_for_backward(q, k, v, g, beta, s0)
        ctx.rule, ctx.scale, ctx.chunk = rule, scale, cfg["CHUNK"]
        return o, S

    @staticmethod
    def backward(ctx, do, dS):
        return _chunked_backward(ctx, do, dS) + (None, None, None, None)


class _MetalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, s0, rule, scale, cfg, allow_tf32):
        o, S = MK.forward(q, k, v, g, beta, rule=rule, scale=scale, initial_state=s0)
        ctx.save_for_backward(q, k, v, g, beta, s0)
        ctx.rule, ctx.scale, ctx.chunk = rule, scale, cfg["CHUNK"]
        return o, S

    @staticmethod
    def backward(ctx, do, dS):
        return _chunked_backward(ctx, do, dS) + (None, None, None, None)


def _chunked_backward(ctx, do, dS):
    q, k, v, g, beta, s0 = ctx.saved_tensors
    ins = [q, k, v, g, beta, s0]
    need = [x is not None and ctx.needs_input_grad[i] for i, x in enumerate(ins)]
    with torch.enable_grad():
        leaves = [x.detach().requires_grad_(n) if x is not None else None for x, n in zip(ins, need)]
        o, S = chunked(*leaves[:5], rule=ctx.rule, chunk=ctx.chunk, scale=ctx.scale, initial_state=leaves[5])
        outs, grads_out = [o], [do]
        if dS is not None:
            outs.append(S)
            grads_out.append(dS.to(S.dtype))
        wanted = [x for x, n in zip(leaves, need) if n]
        got = torch.autograd.grad(outs, wanted, grads_out, allow_unused=True) if wanted else []
    it = iter(got)
    res = []
    for x, n in zip(ins, need):
        if not n:
            res.append(None)
            continue
        gx = next(it)
        res.append(torch.zeros_like(x) if gx is None else gx.to(x.dtype))
    return tuple(res)


def linear_attention(q, k, v, g=None, beta=None, *, rule: str = "add", scale: Optional[float] = None,
                     initial_state: Optional[torch.Tensor] = None, chunk: Optional[int] = None,
                     deterministic: Optional[bool] = None, impl: Optional[str] = None,
                     allow_tf32: bool = False, output_final_state: bool = False):
    """Chunked linear attention / state space recurrence (reference.py has the math and shapes).

    Returns ``o`` [B, T, H, dv] in v's dtype, or ``(o, final_state)`` with
    ``output_final_state=True`` (float32, float64 for float64 inputs). ``scale`` defaults to 1
    (Mamba 2 and the delta rule models fold their scale into q). Any chunk size gives the same
    result in exact arithmetic; ``deterministic`` pins it to triton_kernels.DET_CHUNK."""
    B, T, H, dk, dv = check_args(q, k, v, g, beta, rule, initial_state)
    kind = decay_kind(g, k)
    scale = 1.0 if scale is None else float(scale)
    if deterministic is None:
        deterministic = torch.are_deterministic_algorithms_enabled()
    reason = TK.kernel_supports(kind, dk, dv)
    on_cuda = q.is_cuda
    on_mps = q.device.type == "mps"
    mreason = MK.kernel_supports(kind, dk, dv, q.dtype) if on_mps else "not an MPS tensor"
    if impl is None:
        if on_cuda and TK.available() and reason is None:
            impl = "triton"
        elif on_mps and MK.available() and mreason is None and (chunk in (None, MK.CHUNK)):
            impl = "metal"
        else:
            impl = "chunked"
    if impl not in ("triton", "metal", "chunked"):
        raise LinAttnError(f"[TS1061] impl must be 'triton', 'metal', 'chunked' or None, got {impl!r}")
    if impl == "metal":
        if not on_mps or not MK.available():
            raise LinAttnError("[TS1061] impl='metal' needs MPS tensors on a Mac; the chunked PyTorch form "
                               "runs everywhere (impl='chunked')")
        if mreason is not None:
            raise LinAttnError(f"[TS1061] impl='metal' cannot run this call: {mreason}; use impl='chunked'")
        if chunk not in (None, MK.CHUNK):
            raise LinAttnError(f"[TS1061] the Metal kernels run chunk {MK.CHUNK}, got chunk={chunk}")
    if impl == "triton":
        if not on_cuda or not TK.available():
            raise LinAttnError("[TS1061] impl='triton' needs CUDA tensors and Triton; the chunked PyTorch form "
                               "runs everywhere (impl='chunked')")
        if reason is not None:
            raise LinAttnError(f"[TS1061] impl='triton' cannot run this call: {reason}; use impl='chunked'")
        if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise LinAttnError(f"[TS1061] the Triton kernels take float32, float16 or bfloat16, got {q.dtype}; "
                               "float64 runs in the chunked form (impl='chunked')")
        cfg = TK.config(B, H, dk, dv, deterministic=deterministic, chunk=chunk)
        TK.check_config(cfg, dk, dv, deterministic)
        o, S = _KernelFn.apply(q, k, v, g, beta, initial_state, rule, scale, cfg, allow_tf32)
        c = cfg["CHUNK"]
    elif impl == "metal":
        cfg = MK.config(dk, dv)
        o, S = _MetalFn.apply(q, k, v, g, beta, initial_state, rule, scale, cfg, False)
        c = cfg["CHUNK"]
    else:
        if deterministic and chunk not in (None, TK.DET_CHUNK):
            raise LinAttnError(f"[TS1063] deterministic mode pins the chunk to {TK.DET_CHUNK}; got chunk={chunk}")
        c = TK.DET_CHUNK if chunk is None else chunk
        cfg = {"CHUNK": c}
        o, S = chunked(q, k, v, g, beta, rule=rule, chunk=c, scale=scale, initial_state=initial_state)

    mode = KV.verify_mode()
    sig = _signature(impl, q, k, v, g, beta, rule, initial_state, cfg)
    checked, errors = False, {}
    if mode == "all" or (mode == "first" and sig not in _verified):
        tol = tolerance(v.dtype, T, dk, c, allow_tf32 and impl == "triton")
        if impl == "chunked" and v.dtype == torch.float64:
            tol = KV.BASE_TOL["float64"] * max(1.0, math.log2(max(2, T) * max(2, dk) * max(2, c)) / 4.0)
        errors = _check("linear_attention", o.detach(), S.detach(), q.detach(), k.detach(), v.detach(),
                        None if g is None else g.detach(), None if beta is None else beta.detach(), rule, scale,
                        None if initial_state is None else initial_state.detach(), tol, cfg, impl)
        _verified.add(sig)
        checked = True
    _last.clear()
    _last.update({"impl": impl, "config": dict(cfg), "deterministic": bool(deterministic), "checked": checked,
                  "errors": errors, "decay": kind, "rule": rule})
    return (o, S) if output_final_state else o


def check_gradients(q, k, v, g=None, beta=None, *, rule="add", scale=1.0, initial_state=None, chunk=None,
                    deterministic=None, impl=None, seed: int = 0):
    """Gradients of ``linear_attention`` (impl as chosen) against the float64 gradients of the
    sequential recurrence, for a random upstream gradient. Returns {input name: per element error}
    with the tolerance under ``"tol"``. The inputs are not modified."""
    STATS["grad_checks"] += 1
    names = ["q", "k", "v", "g", "beta", "initial_state"]
    ins = [q, k, v, g, beta, initial_state]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    B, T, H, dk, dv = check_args(q, k, v, g, beta, rule, initial_state)
    do = torch.randn(B, T, H, dv, generator=gen, dtype=torch.float64).to(q.device)
    lv = [None if x is None else x.detach().clone().requires_grad_(True) for x in ins]
    o = linear_attention(*lv[:5], rule=rule, scale=scale, initial_state=lv[5], chunk=chunk,
                         deterministic=deterministic, impl=impl)
    torch.autograd.backward(o, do.to(o.dtype))
    l64 = [None if x is None else x.detach().to(torch.float64).requires_grad_(True) for x in ins]
    o64, _ = recurrent(*l64[:5], rule=rule, scale=scale, initial_state=l64[5])
    torch.autograd.backward(o64, do)
    tol = 10 * tolerance(q.dtype, T, dk, chunk or TK.DET_CHUNK)
    if q.dtype == torch.float64:
        tol = KV.BASE_TOL["float64"] * 10
    out = {"tol": tol}
    for n, a, r in zip(names, lv, l64):
        if a is None:
            continue
        c = KV.compare_torch(n, a.grad, r.grad, tol, KV.FLOOR_REDUCE)
        out[n] = c.error
    return out
