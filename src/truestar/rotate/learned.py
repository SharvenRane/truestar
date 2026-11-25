"""A learned residual rotation (R1), trained on a layer local quantization objective.

The rotation is Q = H C(A): H the deterministic Hadamard of the hidden size and C(A) the Cayley
transform (I - A)(I + A)^-1 of a skew symmetric A, so Q stays orthogonal for every A (SpinQuant
trains R1 on the Stiefel manifold for the same reason). A starts at zero (Q = H).

Objective (calibration, layer local like DartQuant's, not SpinQuant's end to end loss, so that a
360M model trains in a few GB of CPU memory): for every linear layer that reads the residual
stream (q, k, v, gate, up) the relative squared error of fq(X Q) fq(W Q)^T against X W^T, and for
every layer that writes it (o, down) the error of fq(A) fq(Q^T W)^T against A W^T Q, on inputs
captured from the norm folded model with R2 and R4 already applied. fq is symmetric round to
nearest (weights per output channel, activations per token, ``bits`` bits): straight through
rounding, true gradient of the scale (see ``fq``). Each layer's term is backpropagated on its own, so memory is one layer's.

After training, Q is recomputed in float64 from A and handed to ``fold.apply`` like any other
matrix, and ``fold.prove`` checks it before anything is quantized: a learned matrix gets no
exemption from the proof.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch

from . import fold
from .hadamard import orthogonal_hadamard

F64 = torch.float64
READ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj")
WRITE = ("self_attn.o_proj", "mlp.down_proj")


def cayley(a: torch.Tensor) -> torch.Tensor:
    s = (a - a.T) / 2
    i = torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    return torch.linalg.solve(i + s, i - s)


def fq(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric round to nearest per row (last dim). The rounding has a straight through
    gradient; the scale (row amax / qmax) keeps its true gradient. That is the only signal a
    rotation gets: an orthogonal Q leaves x Q (W Q)^T unchanged, so with a detached scale the
    gradient along the rotation manifold would be exactly zero, and the error moves only through
    how the rotation changes each row's largest magnitude."""
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
    z = x / s
    r = z + (torch.round(z) - z).detach()
    return torch.clamp(r, -qmax, qmax) * s


@torch.no_grad()
def _capture(model, ids, a: fold.Arch):
    caps: Dict[str, torch.Tensor] = {}
    hooks = []
    for i in range(a.layers):
        for n in READ + WRITE:
            path = f"model.layers.{i}.{n}"
            mod = model.get_submodule(path)
            if n == "mlp.down_proj" and isinstance(mod, fold.OnlineRotate):
                mod = mod.linear
            hooks.append(mod.register_forward_hook(
                lambda m, inp, out, p=path: caps.__setitem__(p, inp[0].reshape(-1, inp[0].shape[-1]).clone())))
    try:
        model(input_ids=ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return caps


def _weight(model, path):
    mod = model.get_submodule(path)
    return (mod.linear if isinstance(mod, fold.OnlineRotate) else mod).weight.detach()


def learn_r1(model, calib: torch.Tensor, steps: int = 60, lr: float = 3e-3, bits: int = 4,
             device: str = "cpu", r2="hadamard", r4="hadamard",
             log: Optional[List[Dict]] = None) -> torch.Tensor:
    """Train R1 for ``model`` (a Llama or Qwen2 causal LM) on token ids ``calib`` [B, T]; returns
    the orthogonal float64 matrix (d x d)."""
    a = fold.arch_of(model)
    base = fold.apply(model, fold.Plan(r1=None, r2=r2, r4=r4), device=device)   # float64, norms folded
    work = base.model.to(dtype=torch.float32)
    caps = _capture(work, calib.to(device), a)
    layers = []
    for i in range(a.layers):
        terms = []
        for n in READ + WRITE:
            p = f"model.layers.{i}.{n}"
            w, x = _weight(work, p).clone(), caps.pop(p)
            terms.append((n in READ, w, x, (x @ w.T).detach()))
        layers.append(terms)
    del work, base
    h = orthogonal_hadamard(a.hidden).to(device=device, dtype=torch.float32)
    A = torch.zeros(a.hidden, a.hidden, device=device, requires_grad=True)
    opt = torch.optim.Adam([A], lr=lr)
    t0 = time.time()
    for step in range(steps + 1):
        opt.zero_grad()
        q = h @ cayley(A)
        total = 0.0
        for terms in layers:
            loss = 0.0
            for reads, w, x, y in terms:
                if reads:
                    yq = fq(x @ q, bits) @ fq(w @ q, bits).T
                    ref = y
                else:
                    yq = fq(x, bits) @ fq(q.T @ w, bits).T
                    ref = y @ q
                loss = loss + (yq - ref).pow(2).sum() / ref.pow(2).sum()
            if step < steps:
                loss.backward(retain_graph=True)
            total += float(loss.detach())
        if log is not None:
            log.append({"step": step, "loss": total, "s": time.time() - t0})
        if step == steps:
            break
        opt.step()
    with torch.no_grad():
        return orthogonal_hadamard(a.hidden) @ cayley(A.detach().to("cpu", F64))
