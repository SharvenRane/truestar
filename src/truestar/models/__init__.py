"""truestar.models: Hugging Face decoder checkpoints built natively (docs/MODELS.md).

    from truestar import models
    lm = models.from_pretrained("path/to/folder")       # config.json + safetensors (one file or shards)
    logits = lm(ids)                                     # a compiled native function, no torch.export
    out = lm.generate(ids, 16)                           # greedy, through the KV cache

Families: Llama (incl. Llama 3 rope scaling), Mistral, Qwen2, Qwen3, Phi 3, Gemma 2, Gemma 3 text.
Only local folders are read; nothing is downloaded (TS1397).
"""

from .config import FAMILIES, DecoderConfig, ModelLoadError, RopeSpec, decoder_config, read_config
from .decoder import DecoderAttention, DecoderBlock, DecoderLM, DecoderNorm, RopeTable, RotateHalf
from .loader import LoadInfo, build_model, from_pretrained, name_map
from .runner import PretrainedLM, native_functions

__all__ = ["FAMILIES", "DecoderConfig", "ModelLoadError", "RopeSpec", "decoder_config", "read_config",
           "DecoderAttention", "DecoderBlock", "DecoderLM", "DecoderNorm", "RopeTable", "RotateHalf",
           "LoadInfo", "build_model", "from_pretrained", "name_map", "PretrainedLM", "native_functions"]
