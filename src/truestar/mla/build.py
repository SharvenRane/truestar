"""Import the MLA steps into the Truestar IR and compile them.

``import_step`` exports one step (``Step(model, mode)``) with symbolic batch ``B``, token count ``T``
and cache capacity ``S`` and imports it in float64: one program for the prompt and every decode
step. ``compile_steps`` builds the prefill program (expanded attention by default) and the decode
program (``"fused"`` by default since 2026-11-25: the absorbed attention as one fused latent kernel
per layer, mla/latent.py; ``"absorb"`` and ``"expand"`` on request; an absorbed mode only after
``prove_absorption`` held in float64), each compiled with ``truestar.compiler.compile_program``,
whose first call per shape signature is checked against the float64 reference interpreter.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .model import MODES, MLAModel, Step
from .prove import AbsorptionReport, prove_absorption


def import_step(model: MLAModel, mode: str, batch: int = 2, prompt: int = 3, capacity: int = 16,
                max_capacity: int = 1 << 16, max_tokens: int = 4096):
    from torch.export import Dim

    from .. import ir

    c = model.c
    m64 = Step(copy.deepcopy(model).double(), mode)
    B, T, S = Dim("B", min=1, max=1024), Dim("T", min=1, max=max_tokens), Dim("S", min=2, max=max_capacity)
    cc = torch.zeros(c.layers, batch, capacity, c.kv_rank, dtype=torch.float64)
    rc = torch.zeros(c.layers, batch, capacity, c.rope, dtype=torch.float64)
    ids = torch.zeros(batch, prompt, dtype=torch.int64)
    return ir.import_module(m64, (ids, torch.arange(prompt), cc, rc),
                            dynamic_shapes=({0: B, 1: T}, {0: T}, {1: B, 2: S}, {1: B, 2: S}), name=f"mla_{mode}_step")


@dataclass
class MLASteps:
    prefill: Any           # CompiledProgram (or the IR program when device is None)
    decode: Any
    prefill_mode: str
    decode_mode: str
    proof: Optional[AbsorptionReport]

    def generate(self, prompt: torch.Tensor, steps: int, capacity: int, model_cfg, device="cuda", dtype=torch.float32):
        from .model import generate

        def wrap(p):
            return lambda *a: p(list(a))

        return generate(wrap(self.prefill), model_cfg, prompt, steps, capacity, decode=wrap(self.decode),
                        device=device, dtype=dtype)


def compile_steps(model: MLAModel, device: str = "cuda", dtype: str = "float32", prefill: str = "expand",
                  decode: str = "fused", proof_prompt: Optional[torch.Tensor] = None, **compile_kw) -> MLASteps:
    """Prove (when either mode is ``absorb`` or ``fused``), import and compile the prefill and decode
    steps. ``decode="fused"`` runs each layer's attention as one fused latent kernel (mla/latent.py)."""
    from ..compiler import compile_program

    for mode in (prefill, decode):
        if mode not in MODES:
            raise ValueError(f"[TS1200] unknown MLA mode {mode!r}; the modes are {MODES}")
    proof = None
    absorbed = [m for m in (decode, prefill) if m in ("absorb", "fused")]
    if absorbed:
        if proof_prompt is None:
            proof_prompt = torch.randint(0, model.c.vocab, (2, 5), generator=torch.Generator().manual_seed(0))
        for m in dict.fromkeys(absorbed):                   # raises MLAAbsorptionError (TS1201)
            proof = prove_absorption(model, proof_prompt, mode=m)
    progs = {mode: import_step(model, mode) for mode in {prefill, decode}}
    cps = {mode: compile_program(p, device, dtype, **compile_kw) for mode, p in progs.items()}
    return MLASteps(cps[prefill], cps[decode], prefill, decode, proof)


__all__ = ["import_step", "compile_steps", "MLASteps"]
