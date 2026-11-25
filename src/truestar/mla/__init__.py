"""Multi head latent attention (DeepSeek V2 and V3 style MLA) with a latent KV cache.

    from truestar.mla import from_hf, compile_steps

    model = from_hf(hf_deepseek_v3)            # dense layers; MoE (TS940), sparse selection (TS1202) refused
    steps = compile_steps(model, "cuda")       # prefill expanded, decode absorbed (proven in float64 first)
    tokens, logits = steps.generate(prompt, 16, capacity=64, model_cfg=model.c)

The cache holds the compressed latent and the shared RoPE key (``kv_rank + qk_rope_head_dim`` per
token and layer), not per head K and V. See docs/MLA.md.
"""

from .build import MLASteps, compile_steps, import_step
from .hf import MLAUnsupported, from_hf, hf_float64
from .model import MODES, MLAConfig, MLAModel, Step, generate, make_model
from .prove import AbsorptionReport, MLAAbsorptionError, prove_absorption

__all__ = ["MLAConfig", "MLAModel", "Step", "MODES", "make_model", "generate", "from_hf", "hf_float64",
           "MLAUnsupported", "prove_absorption", "AbsorptionReport", "MLAAbsorptionError", "import_step",
           "compile_steps", "MLASteps"]
