"""Importer lowerings of Hugging Face Mamba 2 SSD and Gated DeltaNet layers (pain row P60).

These layers used to be refused by name (TS941, TS942, ir/refusals.py). Their projections, causal
convolution, gates and norms are ordinary ops the importer already takes; only the chunked
recurrence was missing. In transformers 5 each layer computes that recurrence by calling one module
level function: ``mamba2_chunk_scan`` (Mamba 2 SSD) or ``torch_chunk_gated_delta_rule`` (Gated
DeltaNet). ``lowered(model)`` replaces that function, for the duration of torch.export only, by a
function written here from the same definition that computes the recurrence with
``torch.ops.truestar.linear_attention`` and everything around it (the time step softplus and clamp,
the B and C group repeat, the D skip, the l2 norm of q and k, the head repeat) with the same torch
ops in the same order as transformers. The exported graph then has one ``linear_attention`` node per
layer, which the importer turns into the IR op of linattn_ops.py: float64 reference = the sequential
recurrence, compiled = the kernels of truestar.linattn.

What is lowered (``SPECS``) is what was measured end to end (tests/test_hf_models.py,
scripts/bench_linattn_import.py): Mamba 2, Falcon H1, Bamba and Zamba 2 (SSD) and Qwen3.5 (Gated
DeltaNet), on transformers 5.24. Everything else stays refused with a precise reason:

* a layer class with no entry here (Kimi Delta Attention, lightning attention, mLSTM, the SSD and
  Gated DeltaNet layers of mixture of experts families, which TS940 refuses first anyway);
* a transformers whose module has no such function (4.x computes the scan inline in the layer);
* TS1150 at export time: a call the lowering does not cover, an initial recurrent state or a final
  state output (generation with a cache). The forward without a cache is covered.

The replacement is installed in the transformers module only inside ``lowered`` and restored on
exit, also on an error; it is not thread safe (two threads exporting the same family at once share
the module function).
"""

from __future__ import annotations

import contextlib
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .core import UnsupportedOpError

SSD_FN = "mamba2_chunk_scan"
GDN_FN = "torch_chunk_gated_delta_rule"

# class name: (module function it calls for the chunked recurrence, the parameters that function
# must take, so a changed signature is refused instead of silently mis-mapped)
SPECS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "Mamba2Mixer": (SSD_FN, ("hidden_states", "dt", "A", "B", "C", "chunk_size", "D", "dt_bias",
                             "initial_states", "dt_softplus", "dt_limit", "return_final_states")),
    "FalconH1Mixer": (SSD_FN, ()),
    "BambaMixer": (SSD_FN, ()),
    "Zamba2MambaMixer": (SSD_FN, ()),
    "Qwen3_5GatedDeltaNet": (GDN_FN, ("query", "key", "value", "g", "beta", "chunk_size", "initial_state",
                                      "output_final_state", "use_qk_l2norm_in_kernel")),
}
for _c in ("FalconH1Mixer", "BambaMixer", "Zamba2MambaMixer"):
    SPECS[_c] = (SSD_FN, SPECS["Mamba2Mixer"][1])


def _refuse(fn: str, what: str):
    err = UnsupportedOpError(
        f"[TS1150] truestar.ir lowers transformers' {fn} to linear_attention for the forward without a "
        f"cache only; this call has {what}. Import the model without a cache (use_cache=False) or keep "
        "the layer in PyTorch.")
    err.code = "TS1150"
    raise err


def _op():
    from ..linattn import torch_op  # noqa: F401  (registers torch.ops.truestar.linear_attention)
    return torch.ops.truestar.linear_attention


def ssd_chunk_scan(hidden_states, dt, A, B, C, chunk_size=None, D=None, dt_bias=None, initial_states=None,
                   dt_softplus=False, dt_limit=(0.0, float("inf")), return_final_states=False, **kwargs):
    """transformers' ``mamba2_chunk_scan`` with the scan as ``linear_attention`` (rule add, scalar
    decay): q = C, k = B, v = dt x, g = dt A per head, then the D skip. Every op before and after
    the scan, its dtype and its order, is transformers' own (the chunk size and the extra keyword
    arguments do not change the result there either)."""
    if initial_states is not None:
        _refuse(SSD_FN, "an initial recurrent state")
    if return_final_states:
        _refuse(SSD_FN, "a request for the final recurrent state")
    num_heads, num_groups = hidden_states.shape[2], B.shape[2]
    if dt_bias is not None:
        dt = dt + dt_bias.to(dt.dtype)
    if dt_softplus:
        dt = F.softplus(dt)
    dt = torch.clamp(dt, min=dt_limit[0], max=dt_limit[1])
    hidden_states = hidden_states.float()
    B = B.float().repeat_interleave(num_heads // num_groups, dim=2, output_size=num_heads)
    C = C.float().repeat_interleave(num_heads // num_groups, dim=2, output_size=num_heads)
    D_residual = None if D is None else D[..., None] * hidden_states
    v = hidden_states * dt[..., None].float()
    g = A.to(v.dtype) * dt.float()
    out = _op()(C, B, v, g, None, "add", 1.0)
    return out if D_residual is None else out + D_residual


def _l2norm(x, eps: float = 1e-6):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def gdn_chunk(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, **kwargs):
    """transformers' ``torch_chunk_gated_delta_rule`` as ``linear_attention`` (rule delta, scalar
    decay g, write strength beta, scale 1/sqrt(dk)), with its head repeat, float32 casts and l2
    norm of q and k."""
    if initial_state is not None:
        _refuse(GDN_FN, "an initial recurrent state")
    if output_final_state:
        _refuse(GDN_FN, "a request for the final recurrent state")
    initial_dtype = query.dtype
    num_k_heads, num_v_heads = key.shape[2], value.shape[2]
    if num_k_heads != num_v_heads:
        query = query.repeat_interleave(num_v_heads // num_k_heads, dim=2)
        key = key.repeat_interleave(num_v_heads // num_k_heads, dim=2)
    query, key, value, beta, g = (x.to(torch.float32) for x in (query, key, value, beta, g))
    if use_qk_l2norm_in_kernel:
        query, key = _l2norm(query), _l2norm(key)
    out = _op()(query, key, value, g, beta, "delta", query.shape[-1] ** -0.5)
    return out.to(initial_dtype), None


REPLACEMENTS = {SSD_FN: ssd_chunk_scan, GDN_FN: gdn_chunk}


def unlowered_reason(module: torch.nn.Module) -> Optional[str]:
    """None when ``lowered`` covers this layer, else why not (used in the refusal message)."""
    import inspect
    cls = type(module).__name__
    spec = SPECS.get(cls)
    if spec is None:
        return (f"Truestar has no lowering of {cls} (lowered: {', '.join(sorted(SPECS))}; "
                "truestar.ir.hf_linattn)")
    fn, params = spec
    mod = sys.modules.get(type(module).__module__)
    target = getattr(mod, fn, None)
    if target is None:
        ver = getattr(sys.modules.get("transformers"), "__version__", "?")
        return (f"{cls} in transformers {ver} has no module function {fn} to lower (the lowering targets the "
                "transformers 5 layout, measured on 5.24)")
    have = set(inspect.signature(target).parameters)
    missing = [p for p in params if p not in have]
    if missing:
        return f"{type(module).__module__}.{fn} no longer takes {missing}; the lowering would mis-map it"
    return None


def lowerable(model: torch.nn.Module) -> List[Tuple[str, str]]:
    """(path, class) of the layers ``lowered`` replaces in ``model``."""
    return [(p, type(m).__name__) for p, m in model.named_modules()
            if type(m).__name__ in SPECS and unlowered_reason(m) is None]


@contextlib.contextmanager
def lowered(model: torch.nn.Module) -> Iterator[List[Tuple[str, str]]]:
    """Install the replacements for every lowerable layer of ``model``; yields their (path, class)."""
    layers = lowerable(model)
    saved = {}
    for _, m in model.named_modules():
        cls = type(m).__name__
        if cls in SPECS and unlowered_reason(m) is None:
            mod = sys.modules[type(m).__module__]
            fn = SPECS[cls][0]
            saved.setdefault((mod, fn), getattr(mod, fn))
    try:
        for (mod, fn) in saved:
            setattr(mod, fn, REPLACEMENTS[fn])
        yield layers
    finally:
        for (mod, fn), old in saved.items():
            setattr(mod, fn, old)


def _register() -> None:
    from ..linattn import coverage
    for cls in SPECS:
        if cls in coverage.LAYERS:
            coverage.register_lowering(cls, REPLACEMENTS[SPECS[cls][0]])


_register()
