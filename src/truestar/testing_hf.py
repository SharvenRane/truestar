"""Small Hugging Face models built from config classes, with random weights. Nothing is downloaded.

Used by tests/test_hf_models.py and scripts/hf_coverage.py (docs/MODEL_COVERAGE.md). Each entry
builds the model from a ``transformers`` config constructed in code, wraps it so ``forward`` takes
tensors and returns a tuple of tensors (the logits or hidden states), and gives example inputs.

    from truestar.testing_hf import MODELS, build
    model, args = build("llama")

``transformers`` is optional for Truestar; importing this module without it raises ImportError.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn

import transformers  # noqa: F401  (ImportError here when transformers is not installed)

VOCAB = 96
SEQ = 6


class _Wrap(nn.Module):
    """``forward(*tensors) -> tuple of tensors``: the outputs Truestar compares."""

    def __init__(self, model: nn.Module, kind: str):
        super().__init__()
        self.model = model
        self.kind = kind

    def forward(self, x):
        if self.kind == "lm":
            return (self.model(input_ids=x, use_cache=False).logits,)
        if self.kind == "encoder_ids":
            return (self.model(input_ids=x).last_hidden_state,)
        if self.kind == "vision":
            return (self.model(pixel_values=x).last_hidden_state,)
        if self.kind == "vision_pooled":   # hidden states and the attention pooling head
            out = self.model(pixel_values=x)
            return (out.last_hidden_state, out.pooler_output)
        if self.kind == "siglip2":         # patches flattened by the caller, every patch valid, a 4 x 4 grid
            b, ch, h, w = x.shape
            p = self.model.config.patch_size
            patches = x.reshape(b, ch, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(b, -1, ch * p * p)
            mask = torch.ones(b, patches.shape[1], dtype=torch.long, device=x.device)
            shapes = torch.tensor([[h // p, w // p]], device=x.device).expand(b, 2)
            out = self.model(pixel_values=patches, pixel_attention_mask=mask, spatial_shapes=shapes)
            return (out.last_hidden_state, out.pooler_output)
        if self.kind == "pixtral":         # every image at the full size
            sizes = torch.tensor([[x.shape[-2], x.shape[-1]]], device=x.device).expand(x.shape[0], 2)
            return (self.model(pixel_values=x, image_sizes=sizes).last_hidden_state,)
        if self.kind == "text_pooled":     # CLIP text: hidden states and the end of text pooling
            # CLIP builds its causal mask with torch.full in the default dtype and fills it with the
            # minimum of the model's dtype, which overflows float32 for a float64 copy; the default
            # dtype follows the model's for the call (a Python setting, nothing in the graph)
            old = torch.get_default_dtype()
            torch.set_default_dtype(self.model.get_input_embeddings().weight.dtype)
            try:
                out = self.model(input_ids=x)
            finally:
                torch.set_default_dtype(old)
            return (out.last_hidden_state, out.pooler_output)
        if self.kind == "speech":          # Whisper: encoder on the features, decoder on fixed ids
            ids = torch.arange(4, device=x.device).expand(x.shape[0], 4)
            return (self.model(input_features=x, decoder_input_ids=ids, use_cache=False).last_hidden_state,)
        raise ValueError(self.kind)


def _ids(batch: int, seq: int = SEQ, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, seq), generator=g)


def _llama():
    c = transformers.LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                 attn_implementation="sdpa")
    return transformers.LlamaForCausalLM(c), "lm"


def _mistral():
    c = transformers.MistralConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                   sliding_window=4, attn_implementation="sdpa")
    return transformers.MistralForCausalLM(c), "lm"


def _qwen2():
    c = transformers.Qwen2Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                 attn_implementation="sdpa")
    return transformers.Qwen2ForCausalLM(c), "lm"


def _gpt2():
    c = transformers.GPT2Config(vocab_size=VOCAB, n_embd=32, n_layer=2, n_head=4, n_positions=64,
                                attn_implementation="sdpa")
    return transformers.GPT2LMHeadModel(c), "lm"


def _bert():
    c = transformers.BertConfig(vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                                intermediate_size=64, max_position_embeddings=64, attn_implementation="sdpa")
    return transformers.BertModel(c, add_pooling_layer=False), "encoder_ids"


def _vit():
    c = transformers.ViTConfig(image_size=32, patch_size=8, num_channels=3, hidden_size=32, num_hidden_layers=2,
                               num_attention_heads=4, intermediate_size=64, attn_implementation="sdpa")
    return transformers.ViTModel(c, add_pooling_layer=False), "vision"


def _t5_encoder():
    c = transformers.T5Config(vocab_size=VOCAB, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4,
                              relative_attention_num_buckets=8, relative_attention_max_distance=16)
    return transformers.T5EncoderModel(c), "encoder_ids"


def _whisper():
    c = transformers.WhisperConfig(vocab_size=VOCAB, num_mel_bins=8, encoder_layers=2, decoder_layers=2,
                                   encoder_attention_heads=4, decoder_attention_heads=4, d_model=32,
                                   encoder_ffn_dim=64, decoder_ffn_dim=64, max_source_positions=8,
                                   max_target_positions=16, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                                   decoder_start_token_id=1, attn_implementation="sdpa")
    return transformers.WhisperModel(c), "speech"


def _clip_text():
    c = transformers.CLIPTextConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                    num_attention_heads=4, max_position_embeddings=16,
                                    attn_implementation="sdpa")
    return transformers.CLIPTextModel(c), "text_pooled"


def _clip_vision():
    c = transformers.CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, image_size=32, patch_size=8,
                                      attn_implementation="sdpa")
    return transformers.CLIPVisionModel(c), "vision"


def _deit():
    c = transformers.DeiTConfig(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=2,
                                num_attention_heads=4, intermediate_size=64, attn_implementation="sdpa")
    return transformers.DeiTModel(c, add_pooling_layer=False), "vision"


def _swin():
    c = transformers.SwinConfig(image_size=32, patch_size=4, embed_dim=16, depths=[2, 2], num_heads=[2, 4],
                                window_size=4)
    return transformers.SwinModel(c, add_pooling_layer=False), "vision"


def _convnext():
    c = transformers.ConvNextConfig(num_stages=2, hidden_sizes=[16, 32], depths=[1, 1], image_size=32)
    return transformers.ConvNextModel(c), "vision"


def _mobilevit():
    c = transformers.MobileViTConfig(image_size=64, patch_size=2, hidden_sizes=[32, 40, 48],
                                     neck_hidden_sizes=[8, 16, 16, 24, 24, 32, 64], num_attention_heads=4)
    return transformers.MobileViTModel(c, expand_output=True), "vision"


def _mamba():
    c = transformers.MambaConfig(vocab_size=VOCAB, hidden_size=32, state_size=4, num_hidden_layers=2, expand=2,
                                 conv_kernel=4)
    return transformers.MambaForCausalLM(c), "lm"


# Image side per vision model (default 32)
IMAGE_SIZE = {"mobilevit": 64}

MODELS: Dict[str, Callable[[], Tuple[nn.Module, str]]] = {
    "llama": _llama, "mistral": _mistral, "qwen2": _qwen2, "gpt2": _gpt2, "bert": _bert, "vit": _vit,
    "t5_encoder": _t5_encoder,
    # added with the op coverage work (2026-11-24, scripts/op_coverage.py)
    "whisper": _whisper, "clip_text": _clip_text, "clip_vision": _clip_vision, "deit": _deit, "swin": _swin,
    "convnext": _convnext, "mobilevit": _mobilevit, "mamba": _mamba,
}

# Models measured by scripts/op_coverage.py that are not yet in the tests (none today).
EXTRA_MODELS: Dict[str, Callable[[], Tuple[nn.Module, str]]] = {}


# ----------------------------------------------------------------------------- 2025 to 2026 families
#
# Added with the coverage refresh (2026-11-24, docs/MODEL_COVERAGE.md "Current models"). These need a
# recent transformers (the sweep ran 5.24.0); on an older one a family whose model type is unknown
# is left out by ``available()``. Configs come from ``AutoConfig.for_model`` with the same small
# sizes for every family (vocab 96, width 32, 2 layers, 4 query and 2 KV heads of 8, MLP 64) plus
# the settings that make the family's new parts live at that size: a sliding window shorter than
# the sequence, one local and one global layer, soft capping, partial rotary, per layer embeddings,
# KV shared by the last layer.

_LM = dict(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
           num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False, pad_token_id=0,
           bos_token_id=1, eos_token_id=2)
_TWO = ["sliding_attention", "full_attention"]
# KV sharing needs a layer of the same type to share from: local, global, and a global that shares
_THREE = dict(num_hidden_layers=3, layer_types=["sliding_attention", "full_attention", "full_attention"],
              num_kv_shared_layers=1)


def _model_types():
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    return set(CONFIG_MAPPING_NAMES)


def _auto_lm(model_type: str, attn: str = "sdpa", experts: Optional[str] = None, **over):
    def make():
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.for_model(model_type, **{**_LM, **over})
        # transformers before 5.0 has no experts implementations (its experts are per expert modules)
        kw = {"experts_implementation": experts} if experts and int(transformers.__version__.split(".")[0]) >= 5 else {}
        return AutoModelForCausalLM.from_config(cfg, attn_implementation=attn, **kw), "lm"
    make.model_type = model_type
    return make


def _auto_vision(model_type: str, cls: str, kind: str = "vision", min_transformers: int = 0, **over):
    def make():
        from transformers import AutoConfig
        base = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                    image_size=32, patch_size=8, num_channels=3)
        cfg = AutoConfig.for_model(model_type, **{**base, **over})
        return getattr(transformers, cls)._from_config(cfg, attn_implementation="sdpa"), kind
    make.model_type = model_type
    # Pixtral's 4.x forward takes a list of images with no batch ("Batching/padding not supported")
    make.min_transformers = min_transformers
    return make


def _convnextv2():
    c = transformers.ConvNextV2Config(num_stages=2, hidden_sizes=[16, 32], depths=[1, 1], image_size=32)
    return transformers.ConvNextV2Model(c), "vision"


_convnextv2.model_type = "convnextv2"


MODELS_2026: Dict[str, Callable[[], Tuple[nn.Module, str]]] = {
    # Gemma 2: local and global layers, attention and final logit soft capping (eager attention:
    # SDPA has no soft cap), the query scale from query_pre_attn_scalar
    "gemma2": _auto_lm("gemma2", attn="eager", sliding_window=4, layer_types=_TWO, attn_logit_softcapping=50.0,
                       final_logit_softcapping=30.0, query_pre_attn_scalar=8),
    # Gemma 3: QK norm, local and global layers with their own RoPE bases
    "gemma3": _auto_lm("gemma3_text", sliding_window=4, layer_types=_TWO, final_logit_softcapping=30.0,
                       query_pre_attn_scalar=8),
    # Gemma 3n: AltUp, Laurel, per layer embeddings, KV shared by the last layer, Gaussian top k
    # activation sparsity in the first layer
    "gemma3n": _auto_lm("gemma3n_text", sliding_window=4, hidden_size_per_layer_input=8,
                        vocab_size_per_layer_input=VOCAB, altup_num_inputs=2, laurel_rank=4,
                        activation_sparsity_pattern=[0.95, 0.0, 0.0], intermediate_size=[64, 64, 64], **_THREE),
    # Gemma 4 (dense, E2B style): per layer embeddings, KV shared by the last layer, proportional
    # partial RoPE and a wider head on the global layer
    "gemma4": _auto_lm("gemma4_text", sliding_window=4, hidden_size_per_layer_input=8,
                       vocab_size_per_layer_input=VOCAB, per_layer_config={"01": {"head_dim": 16}, "02": {"head_dim": 16}},
                       final_logit_softcapping=30.0, **_THREE),
    "qwen3": _auto_lm("qwen3"),                                         # QK norm
    # Phi-4 mini is the phi3 model type with 3/4 of each head rotated
    "phi4_mini": _auto_lm("phi3", rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                                    "partial_rotary_factor": 0.75}),
    "smollm3": _auto_lm("smollm3", no_rope_layer_interval=2),           # every second layer without RoPE
    "olmo2": _auto_lm("olmo2"),                                         # norms after attention and MLP, QK norm
    "olmo3": _auto_lm("olmo3", sliding_window=4, layer_types=_TWO),
    "granite": _auto_lm("granite", embedding_multiplier=12.0, logits_scaling=8.0, residual_multiplier=0.22,
                        attention_multiplier=0.0625),
    "cohere2": _auto_lm("cohere2", sliding_window=4, layer_types=_TWO, logit_scale=0.5),  # parallel block
    "glm4": _auto_lm("glm4"),                                           # half rotary, sandwich norms
    "ernie4_5": _auto_lm("ernie4_5"),
    "exaone4": _auto_lm("exaone4", sliding_window=4, layer_types=_TWO),
    "seed_oss": _auto_lm("seed_oss"),
    "apertus": _auto_lm("apertus"),                                     # xIELU activation, QK norm
    "arcee": _auto_lm("arcee"),                                         # ReLU squared MLP
    "ministral3": _auto_lm("ministral3"),                               # YaRN with the Llama 4 query scale
    "hunyuan_dense": _auto_lm("hunyuan_v1_dense"),                      # QK norm
    "lfm2": _auto_lm("lfm2", layer_types=["conv", "full_attention"], block_auto_adjust_ff_dim=False,
                     block_ff_dim=64),                                  # gated short convolution blocks
    "bitnet": _auto_lm("bitnet"),                                       # ReLU squared, sub layer norms
    # vision encoders of 2025 to 2026 multimodal models
    "siglip_vision": _auto_vision("siglip_vision_model", "SiglipVisionModel", kind="vision_pooled"),
    "siglip2_vision": _auto_vision("siglip2_vision_model", "Siglip2VisionModel", kind="siglip2", num_patches=16),
    "dinov3_vit": _auto_vision("dinov3_vit", "DINOv3ViTModel", num_register_tokens=2, use_gated_mlp=True,
                               layerscale_value=0.1),
    "aimv2_vision": _auto_vision("aimv2_vision_model", "Aimv2VisionModel", kind="vision_pooled"),
    "smolvlm_vision": _auto_vision("smolvlm_vision", "SmolVLMVisionTransformer"),
    "pixtral_vision": _auto_vision("pixtral", "PixtralVisionModel", kind="pixtral", head_dim=8,
                                   min_transformers=5),
    "convnextv2": _convnextv2,                                          # global response norm
}


# Families of MODELS_2026 that do not import yet, with the measured reason (tests check they still
# fail the same way, so a fix shows up as a test to update)
KNOWN_GAPS_2026: Dict[str, str] = {}   # SigLIP2 and Pixtral vision import since 2026-11-25 (ir/importer_vision.py)

# Families that import only at a fixed batch, with the reason: their forward loops over the images in
# Python, so torch.export unrolls it per image and a symbolic batch has no loop count; the per image
# grid sizes come from a tensor built from constants (ir/importer_vision.py)
FIXED_BATCH_2026 = {
    "siglip2_vision": "the position embedding is resized per image in a Python loop over spatial_shapes",
    "pixtral_vision": "each image's patch grid is sliced in a Python loop over image_sizes",
}


def _dinov3_vit():
    model, kind = _auto_vision("dinov3_vit", "DINOv3ViTModel", num_register_tokens=2, use_gated_mlp=True,
                               layerscale_value=0.1)()
    model.embeddings.mask_token.requires_grad_(False)    # read only with bool_masked_pos, which is not given
    return model, kind


_dinov3_vit.model_type = "dinov3_vit"
MODELS_2026["dinov3_vit"] = _dinov3_vit


def _mamba2():
    c = transformers.Mamba2Config(vocab_size=VOCAB, hidden_size=32, state_size=4, num_hidden_layers=2, expand=2,
                                  conv_kernel=4, num_heads=8, head_dim=8, n_groups=1, chunk_size=4)
    return transformers.Mamba2ForCausalLM(c), "lm"


_mamba2.model_type = "mamba2"

# Families Truestar refuses by name (truestar.ir.refusals, codes TS940 to TS942): mixture of experts
# routing, chunked linear attention and delta rule recurrences, chunked selective state space scans.
# Built here so the sweep shows each refusal on the real model.
_MOE = dict(num_experts=4, num_local_experts=4, num_experts_per_tok=2, moe_intermediate_size=16)
REFUSED_2026: Dict[str, Callable[[], Tuple[nn.Module, str]]] = {
    "gpt_oss": _auto_lm("gpt_oss", attn="eager", sliding_window=4, layer_types=_TWO, num_local_experts=4,
                        num_experts_per_tok=2),
    "qwen3_next": _auto_lm("qwen3_next", layer_types=["linear_attention", "full_attention"], **_MOE),
    "llama4_text": _auto_lm("llama4_text", num_local_experts=4, num_experts_per_tok=1, intermediate_size_mlp=64),
    "kimi_linear": _auto_lm("kimi_linear"),
    "nemotron_h": _auto_lm("nemotron_h"),
}

# Mamba 2 SSD and Gated DeltaNet families (pain row P60): refused by name (TS941, TS942) until
# 2026-11-24, now lowered to the verified linear_attention op (truestar.ir.hf_linattn) where the
# installed transformers has the module function the lowering replaces (5.x; on 4.48 they are still
# refused, with the reason). Small configs: chunk 4 so a 6 token sequence crosses chunk borders in
# transformers' own chunked form, 8 SSM heads of width 8, state 4, one SSD group.
LOWERED_2026: Dict[str, Callable[[], Tuple[nn.Module, str]]] = {
    "mamba2": _mamba2,
    "falcon_h1": _auto_lm("falcon_h1"),
    # Qwen3.5: Gated DeltaNet with 2 key heads shared by 4 value heads of width 8; attention heads of 16
    # with half rotated (the real split is a quarter of 256), and the multimodal
    # rotary sections scaled to the head (the full size split [11, 11, 10] leaves width 8 heads with
    # empty sections, which no real checkpoint has)
    "qwen3_5": _auto_lm("qwen3_5_text", layer_types=["linear_attention", "full_attention"],
                        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
                        linear_value_head_dim=8, linear_conv_kernel_dim=4, head_dim=16,
                        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                                         "mrope_section": [2, 1, 1], "mrope_interleaved": True}),
    "bamba": _auto_lm("bamba", attn_layer_indices=[1], mamba_n_heads=8, mamba_d_head=8, mamba_n_groups=1,
                      mamba_d_state=4, mamba_chunk_size=4, mamba_expand=2),
    "zamba2": _auto_lm("zamba2", layers_block_type=["mamba", "hybrid"], n_mamba_heads=8, mamba_d_state=4,
                       mamba_ngroups=1, chunk_size=4, mamba_expand=2, attention_head_dim=8, num_mem_blocks=1,
                       use_shared_attention_adapter=False, adapter_rank=4),
}
# Lowered families whose compiled CUDA program is refused, with the measured reason (the test checks
# they still fail loudly, so a fix shows up as a test to update)
LOWERED_CUDA_GAPS = {
    "falcon_h1": "TS1060: the default Falcon H1 layer has a state width of 256 and per step decays down to "
                 "-12.8 (A up to 128); the Triton kernel's float32 state is 1.2e-4 off float64 there against a "
                 "tolerance of 7.3e-5, so its own first call check refuses (the chunked form passes on the CPU)",
}
LOWERED_CODES = {"mamba2": "TS942", "falcon_h1": "TS942", "qwen3_5": "TS941", "bamba": "TS942", "zamba2": "TS942"}


def lowered_families() -> Dict[str, Callable]:
    """The LOWERED_2026 families whose every linear and SSM layer the installed transformers lets
    truestar.ir.hf_linattn lower (so the import is not refused)."""
    from .ir.hf_linattn import lowerable
    from .ir.refusals import refusal_for
    out = {}
    for name, make in available(LOWERED_2026).items():
        torch.manual_seed(0)
        try:
            model, _ = make()
        except Exception:   # noqa: BLE001  (a config this transformers cannot build: not lowered here)
            continue
        if refusal_for([(p, type(m).__name__) for p, m in model.named_modules()], lowerable(model)) is None:
            out[name] = make
    return out


# Routed MoE families Truestar lowers (truestar.moe; before MoE support each was refused with TS940).
# 4 experts, top 2, unless the family's point needs more (DeepSeek's group limited routing). HF's
# "eager" experts (a loop over experts with index_add) are the eager reference: its default grouped_mm
# has no float64 kernel.
MOE_2026: Dict[str, Callable[[], Tuple[nn.Module, str]]] = {
    "mixtral": _auto_lm("mixtral", experts="eager", num_local_experts=4, num_experts_per_tok=2),
    "qwen3_moe": _auto_lm("qwen3_moe", experts="eager", **_MOE),
    "olmoe": _auto_lm("olmoe", experts="eager", num_experts=4, num_experts_per_tok=2),
    "deepseek_v3": _auto_lm("deepseek_v3", experts="eager", n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
                            first_k_dense_replace=1, kv_lora_rank=8, q_lora_rank=8, qk_rope_head_dim=8,
                            qk_nope_head_dim=8, v_head_dim=8, n_group=1, topk_group=1, num_key_value_heads=4),
    # group limited routing: 8 experts in 4 groups, the 2 best groups, then the top 2 experts in them
    "deepseek_v3_groups": _auto_lm("deepseek_v3", experts="eager", n_routed_experts=8, num_experts_per_tok=2,
                                   moe_intermediate_size=16, first_k_dense_replace=1, kv_lora_rank=8,
                                   q_lora_rank=8, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8, n_group=4, num_key_value_heads=4,
                                   topk_group=2),
}


def available(table) -> Dict[str, Callable]:
    """The entries of ``table`` whose model type this transformers knows."""
    known = _model_types()
    major = int(transformers.__version__.split(".")[0])
    return {k: v for k, v in table.items() if getattr(v, "model_type", None) in known
            and getattr(v, "min_transformers", 0) <= major}


def example_inputs(kind: str, batch: int, seed: int = 0, size: int = 32) -> Tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    if kind in ("vision", "vision_pooled", "siglip2", "pixtral"):
        return (torch.randn(batch, 3, size, size, generator=g),)
    if kind == "speech":   # [B, mel bins, 2 * max_source_positions]
        return (torch.randn(batch, 8, 16, generator=g),)
    return (_ids(batch, seed=seed),)


def build(name: str, batch: int = 2, seed: int = 0, dtype: torch.dtype = torch.float32):
    """(wrapped model in eval mode, example args). Weights come from ``torch.manual_seed(seed)``."""
    torch.manual_seed(seed)
    model, kind = {**MOE_2026, **REFUSED_2026, **LOWERED_2026, **MODELS_2026, **EXTRA_MODELS, **MODELS}[name]()
    model = _Wrap(model, kind).eval().to(dtype)
    return model, example_inputs(kind, batch, seed, IMAGE_SIZE.get(name, 32))
