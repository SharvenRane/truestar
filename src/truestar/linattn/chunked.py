"""The chunked form of the recurrence in plain PyTorch, written from the math (docs/LINEAR_ATTENTION.md).

The sequence is cut into chunks of ``chunk`` steps. Inside a chunk everything is a matrix product;
only the state is carried from chunk to chunk. With ``c_i`` the running sum of the log decays
inside the chunk (``c_i = g_1 + ... + g_i``, per head or per key channel) and S the state before
the chunk:

* every step's own contribution decays to step i by ``exp(c_i - c_j)`` (never above 1, since every
  g is at most 0), and the start state by ``exp(c_i)``;
* rule ``add``: ``o_i = scale * ((q_i * exp(c_i)) S + sum_{j <= i} A_ij v_j)`` with
  ``A_ij = sum_d q_id k_jd exp(c_id - c_jd)``;
* rule ``delta``: the step is an ``add`` step with the pseudo value
  ``u_i = beta_i (v_i - k_i^T D_i S_{i-1})``. Writing ``D_i S_{i-1}`` out gives the unit lower
  triangular system ``(I + M) U = diag(beta) (V - Kc S)`` with
  ``M_ij = beta_i sum_d k_id k_jd exp(c_id - c_jd)`` for ``j < i`` and ``Kc_i = k_i * exp(c_i)``,
  so ``U = Wv - Wk S`` where ``Wv`` and ``Wk`` solve that system for ``beta V`` and ``beta Kc``.
  ``Wv`` and ``Wk`` depend on the chunk's own inputs only;
* the next state is ``exp(c_C) S + sum_j (k_j * exp(c_C - c_j)) u_j^T``.

Each identity is exact in real arithmetic for every chunk size, so the output does not depend on
``chunk`` except through rounding (tests/test_linattn.py measures how much it does). The state is
carried in float32 for 16 and 32 bit inputs and in float64 for float64 inputs. Everything here is
differentiable by autograd; it is the CPU path, the backward of the Triton kernels, and the
"eager PyTorch" speed baseline.
"""

from __future__ import annotations

from typing import Optional

import torch

from .reference import LinAttnError, check_args, decay_kind


def _state_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def chunked(q, k, v, g=None, beta=None, *, rule: str = "add", chunk: int = 64, scale: float = 1.0,
            initial_state: Optional[torch.Tensor] = None, state_dtype: Optional[torch.dtype] = None):
    """Chunked recurrence. Returns (o [B, T, H, dv] in v's dtype, final state [B, H, dk, dv])."""
    B, T, H, dk, dv = check_args(q, k, v, g, beta, rule, initial_state)
    if not isinstance(chunk, int) or chunk < 1:
        raise LinAttnError(f"[TS1061] chunk must be a positive int, got {chunk!r}")
    kind = decay_kind(g, k)
    sd = state_dtype or _state_dtype(v.dtype)
    C = chunk
    N = -(-T // C) if T else 0
    pad = N * C - T

    def prep(x):  # [B, T, H, ...] -> [B, H, N, C, ...] in the state dtype, zero padded at the end
        x = x.to(sd).transpose(1, 2)
        if pad:
            x = torch.nn.functional.pad(x, (0, 0) * (x.dim() - 3) + (0, pad))
        return x.reshape(B, H, N, C, *x.shape[3:])

    # padding steps have k = v = 0 and g = 0: they leave the state unchanged and only add outputs
    # that are cut off below, so padding is exact
    qc, kc, vc = prep(q), prep(k), prep(v)
    if kind == "none":
        cum = torch.zeros(B, H, N, C, dtype=sd, device=q.device)
    elif kind == "scalar":
        cum = torch.cumsum(prep(g[..., None])[..., 0], dim=3)         # [B, H, N, C]
    else:
        cum = torch.cumsum(prep(g), dim=3)                            # [B, H, N, C, dk]
    if kind == "channel":
        decay_q = torch.exp(cum)                                      # [B, H, N, C, dk]
        decay_end = torch.exp(cum[:, :, :, -1:, :] - cum)             # to the chunk's end
        diff = cum[:, :, :, :, None, :] - cum[:, :, :, None, :, :]    # c_i - c_j, [.., C, C, dk]
    else:
        decay_q = torch.exp(cum)[..., None]
        decay_end = torch.exp(cum[:, :, :, -1:] - cum)[..., None]
        diff = (cum[:, :, :, :, None] - cum[:, :, :, None, :])[..., None]
    idx = torch.arange(C, device=q.device)
    lower = idx[:, None] >= idx[None, :]                              # j <= i
    strict = idx[:, None] > idx[None, :]                              # j < i
    # exp only where j <= i, so a masked entry can never overflow and give inf * 0
    neg = torch.tensor(float("-inf"), dtype=sd, device=q.device)
    pair = torch.exp(torch.where(lower[..., None], diff, neg))        # [B, H, N, C, C, dk or 1]
    if kind == "channel":
        A = torch.einsum("bhnid,bhnjd,bhnijd->bhnij", qc, kc, pair)
    else:
        A = torch.einsum("bhnid,bhnjd->bhnij", qc, kc) * pair[..., 0]
    q_start = qc * decay_q                                            # q_i * exp(c_i)
    k_end = kc * decay_end                                            # k_j * exp(c_C - c_j)
    if kind == "channel":
        state_decay = torch.exp(cum[:, :, :, -1, :])[..., None]       # [B, H, N, dk, 1]
    else:
        state_decay = torch.exp(cum[:, :, :, -1])[..., None, None]    # [B, H, N, 1, 1]

    if rule == "delta":
        bc = prep(beta[..., None])                                    # [B, H, N, C, 1]
        if kind == "channel":
            kk = torch.einsum("bhnid,bhnjd,bhnijd->bhnij", kc, kc, pair)
        else:
            kk = torch.einsum("bhnid,bhnjd->bhnij", kc, kc) * pair[..., 0]
        M = torch.where(strict, kk, torch.zeros((), dtype=sd, device=q.device)) * bc
        eye = torch.eye(C, dtype=sd, device=q.device)
        rhs = torch.cat([bc * vc, bc * (kc * decay_q)], dim=-1)       # [B, H, N, C, dv + dk]
        W = torch.linalg.solve_triangular(eye + M, rhs, upper=False, unitriangular=True)
        Wv, Wk = W[..., :dv], W[..., dv:]

    S = (torch.zeros(B, H, dk, dv, dtype=sd, device=q.device) if initial_state is None
         else initial_state.to(sd))
    outs = []
    for n in range(N):
        U = vc[:, :, n] if rule == "add" else Wv[:, :, n] - Wk[:, :, n] @ S
        outs.append((q_start[:, :, n] @ S + A[:, :, n] @ U) * scale)
        S = state_decay[:, :, n] * S + k_end[:, :, n].transpose(-1, -2) @ U
    if N:
        o = torch.stack(outs, dim=2).reshape(B, H, N * C, dv)[:, :, :T]
    else:
        o = torch.zeros(B, H, 0, dv, dtype=sd, device=q.device)
    return o.transpose(1, 2).to(v.dtype), S
