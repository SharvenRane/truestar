"""The float64 sequential reference for linear attention and state space recurrences.

One recurrence covers the layers of docs/LINEAR_ATTENTION.md. Per batch, head and step t, with a
key and query of width dk, a value of width dv and a state S of shape [dk, dv]:

* decay ``D_t = diag(exp(g_t))``: ``g_t`` is one log decay per head (``scalar``: Mamba 2 SSD,
  Gated DeltaNet, lightning attention) or one per key channel (``channel``: gated linear attention,
  Kimi Delta Attention), never positive; ``g`` None means no decay.
* rule ``add``:   ``S_t = D_t S_{t-1} + k_t v_t^T``
* rule ``delta``: ``S_t = (I - beta_t k_t k_t^T) D_t S_{t-1} + beta_t k_t v_t^T``
* output ``o_t = scale * S_t^T q_t`` (the state after step t).

Shapes (the layout of the models' own code): q and k ``[B, T, H, dk]``, v ``[B, T, H, dv]``, g
``[B, T, H]`` or ``[B, T, H, dk]``, beta ``[B, T, H]``, initial state ``[B, H, dk, dv]``.

The reference is the recurrence itself, one step at a time, in float64 on the inputs' device.
It shares no code with the chunked forms it checks (chunked.py and the Triton kernels).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

RULES = ("add", "delta")


class LinAttnError(ValueError):
    """A call the linear attention kernels refuse (bad shapes, rule, dtype or device), TS1061."""

    code = "TS1061"


def decay_kind(g: Optional[torch.Tensor], k: torch.Tensor) -> str:
    """'none', 'scalar' (one log decay per head) or 'channel' (one per key channel)."""
    if g is None:
        return "none"
    if g.dim() == 3:
        return "scalar"
    if g.dim() == 4 and g.shape[-1] == k.shape[-1]:
        return "channel"
    raise LinAttnError(f"[TS1061] g must be [B, T, H] or [B, T, H, dk={k.shape[-1]}], got {tuple(g.shape)}")


def check_args(q, k, v, g, beta, rule, initial_state) -> Tuple[int, int, int, int, int]:
    """Shape checks shared by every implementation. Returns (B, T, H, dk, dv)."""
    if rule not in RULES:
        raise LinAttnError(f"[TS1061] rule must be one of {RULES}, got {rule!r}")
    if q.dim() != 4 or k.shape != q.shape:
        raise LinAttnError(f"[TS1061] q and k must both be [B, T, H, dk], got {tuple(q.shape)} and {tuple(k.shape)}")
    B, T, H, dk = q.shape
    if v.dim() != 4 or tuple(v.shape[:3]) != (B, T, H):
        raise LinAttnError(f"[TS1061] v must be [B={B}, T={T}, H={H}, dv], got {tuple(v.shape)}")
    dv = v.shape[-1]
    kind = decay_kind(g, k)
    if kind != "none" and tuple(g.shape[:3]) != (B, T, H):
        raise LinAttnError(f"[TS1061] g must start with [B={B}, T={T}, H={H}], got {tuple(g.shape)}")
    if rule == "delta":
        if beta is None or tuple(beta.shape) != (B, T, H):
            raise LinAttnError(f"[TS1061] the delta rule needs beta of shape [B={B}, T={T}, H={H}], got "
                               f"{None if beta is None else tuple(beta.shape)}")
    elif beta is not None:
        raise LinAttnError("[TS1061] beta is only used by the delta rule; pass rule='delta' or beta=None")
    if initial_state is not None and tuple(initial_state.shape) != (B, H, dk, dv):
        raise LinAttnError(f"[TS1061] initial_state must be [B={B}, H={H}, dk={dk}, dv={dv}], got "
                           f"{tuple(initial_state.shape)}")
    return B, T, H, dk, dv


def recurrent(q, k, v, g=None, beta=None, *, rule: str = "add", scale: float = 1.0,
              initial_state: Optional[torch.Tensor] = None, dtype=torch.float64):
    """The recurrence one step at a time. Returns (o [B, T, H, dv], final state [B, H, dk, dv]).

    ``dtype`` is the arithmetic: float64 for the reference. Any other dtype gives the plain
    sequential scan in that precision (the baseline the chunked kernels are measured against).
    Differentiable: the float64 gradients of this function are the gradient reference."""
    B, T, H, dk, dv = check_args(q, k, v, g, beta, rule, initial_state)
    kind = decay_kind(g, k)
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    if kind != "none":
        g = g.to(dtype)
    if beta is not None:
        beta = beta.to(dtype)
    if initial_state is None:
        S = torch.zeros(B, H, dk, dv, dtype=dtype, device=q.device)
    else:
        S = initial_state.to(dtype)
    outs = []
    for t in range(T):
        if kind == "scalar":
            S = S * torch.exp(g[:, t])[:, :, None, None]
        elif kind == "channel":
            S = S * torch.exp(g[:, t])[:, :, :, None]
        kt = k[:, t]
        if rule == "delta":
            pred = torch.einsum("bhk,bhkv->bhv", kt, S)
            u = beta[:, t, :, None] * (v[:, t] - pred)
        else:
            u = v[:, t]
        S = S + kt[:, :, :, None] * u[:, :, None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], S) * scale)
    o = torch.stack(outs, dim=1) if outs else v.new_zeros(B, 0, H, dv)
    return o, S
