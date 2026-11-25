"""Chunked linear attention and state space recurrences, verified against float64 (pain P53).

docs/LINEAR_ATTENTION.md has the math, the chunk size invariance proof, the measured differences
in finite precision and the speed. Entry points:

* ``linear_attention(q, k, v, g, beta, rule=...)``: the op (Triton on CUDA, chunked PyTorch
  elsewhere), checked against the float64 recurrence on its first call per signature.
* ``recurrent``: the float64 sequential reference (also the plain sequential scan baseline).
* ``chunked``: the chunked form in PyTorch, written from the math.
* ``chunk_invariance`` and ``chain_error``: the measurements behind the invariance report.
* ``refusal_lifted(kind)``: the hook the importer's refusal by name (TS941, TS942) consults.
"""

from .chunked import chunked  # noqa: F401
from .coverage import LAYERS, refusal_lifted  # noqa: F401
from .invariance import chain_error, chunk_invariance  # noqa: F401
from .op import LinAttnMismatchError, check_gradients, last_call, linear_attention  # noqa: F401
from .reference import RULES, LinAttnError, recurrent  # noqa: F401
from .triton_kernels import available  # noqa: F401

__all__ = ["linear_attention", "recurrent", "chunked", "chunk_invariance", "chain_error", "check_gradients",
           "last_call", "refusal_lifted", "LAYERS", "LinAttnError", "LinAttnMismatchError", "RULES", "available"]
