"""Text, retrieval, speech and vision language models built from configs, with random weights.

The second coverage table next to ``truestar.testing_hf`` (docs/MODEL_COVERAGE.md, "Text, retrieval,
speech and vision language models"). Nothing is downloaded: every config is constructed in code at a
tiny size (vocab 96, width 32, 2 layers, 4 heads of 8, MLP 64), and the wrapper makes ``forward``
take tensors and return a tuple of tensors, output 0 being the one a training step differentiates.

What each wrapper runs, as the model is used in practice:

* ``seq2seq``: T5, BART, Marian. The encoder, the decoder over a 3 token prefix with the cache on,
  then one decode step that reads the self attention cache and the cross attention cache.
  Outputs: the step's logits, the prefix logits.
* ``embed``: BERT, RoBERTa, XLM-RoBERTa, DistilBERT, ModernBERT, DeBERTa v3. Padded batches (row i
  has ``i % 3`` padding tokens at the end, set to the family's pad id) and the sentence embedding
  head: the mean of the last hidden state over the attention mask, then L2 normalized.
  Outputs: the embedding, the last hidden state.
* ``audio``: wav2vec2, HuBERT. The convolutional feature extractor on the raw waveform, the
  convolutional position embedding (weight norm) and the encoder.
* ``speech_decode``: Whisper, as ``seq2seq`` with log mel features.
* ``vlm``: Qwen2.5-VL, Qwen3-VL, LLaVA, LLaVA-NeXT, Gemma 3, Idefics3, SmolVLM. One image per row
  at a fixed size: the image placeholder tokens are put in front of the text ids, the model merges
  the projected image features into those positions, runs the prefill with the cache on and then
  one decode step. The prompt layout (where the image tokens are) is fixed per program, as for a
  processor that always sees one image of this size, and so is the batch. Image metadata the model
  reads as sizes is given as Python ints where the model takes them (LLaVA-NeXT's ``image_sizes``).
  transformers' runtime check that the placeholder count matches the image features is an
  ``.item()`` the IR has no assertion for; the wrapper builds the prompt with the right count and
  sets ``TRANSFORMERS_DISABLE_TORCH_CHECK=1`` for the call (without it the import is refused by
  name, tests/test_hf_text.py). Outputs: the step's logits, the prefill logits.

    from truestar.testing_hf_text import MODELS_TEXT, build
    model, args = build("t5")
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

import torch
from torch import nn

import transformers  # noqa: F401  (ImportError here when transformers is not installed)

VOCAB = 96
SEQ = 6
PREFIX = 3          # decoder prefix length before the decode step
STEP_TOKEN = 5      # the token fed to the decode step
IMAGE = 32          # image side for every vision language model
TEXT = 4            # text tokens after the image tokens
WAVE = 400          # raw audio samples (wav2vec2, HuBERT): 19 frames after the feature extractor
# ids that are never drawn as text, so a placeholder can never be confused with a word
IMAGE_TOKEN, VIDEO_TOKEN, START_TOKEN, END_TOKEN = 95, 94, 93, 92
DISABLE_CHECK_ENV = "TRANSFORMERS_DISABLE_TORCH_CHECK"


def _model_types():
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    return set(CONFIG_MAPPING_NAMES)


def _first_param_dtype(model: nn.Module) -> torch.dtype:
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32


class TextWrap(nn.Module):
    """``forward(*tensors) -> tuple of tensors`` for the kinds in the module docstring."""

    def __init__(self, model: nn.Module, kind: str, pad_id: int = 0, layout: List[int] = (), family: str = ""):
        super().__init__()
        self.model = model
        self.kind = kind
        self.pad_id = pad_id
        self.layout = list(layout)       # the prompt tokens in front of the text ids (vlm)
        self.family = family
        self.disable_check = True        # vlm: transformers' placeholder count check (see forward)

    # ------------------------------------------------------------------ encoder decoder
    def _decode(self, enc_kwargs):
        b = next(iter(enc_kwargs.values())).shape[0]
        dev = next(iter(enc_kwargs.values())).device
        prefix = torch.arange(PREFIX, device=dev).expand(b, PREFIX)
        out = self.model(**enc_kwargs, decoder_input_ids=prefix, use_cache=True)
        enc = out.encoder_last_hidden_state
        step = torch.full((b, 1), STEP_TOKEN, dtype=torch.long, device=dev)
        nxt = self.model(encoder_outputs=(enc,), decoder_input_ids=step, past_key_values=out.past_key_values,
                         use_cache=True)
        return (nxt.logits, out.logits)

    # ------------------------------------------------------------------ vision language
    def _vlm(self, ids, pixels):
        m, c, f = self.model, self.model.config, self.family
        b, dev = ids.shape[0], ids.device
        lay = torch.tensor(self.layout, dtype=torch.long, device=dev)
        full = torch.cat([lay.expand(b, -1), ids], 1)
        n = full.shape[1]
        kw = {}
        if f in ("qwen2_5_vl", "qwen3_vl"):
            # the processor's patch layout: the image repeated over the temporal patch, patches grouped by
            # the 2 x 2 spatial merge, one row per patch of C * T * P * P values
            vc = c.vision_config
            p, t, ms = vc.patch_size, vc.temporal_patch_size, vc.spatial_merge_size
            gh = gw = IMAGE // p
            x = pixels.unsqueeze(1).expand(b, t, 3, IMAGE, IMAGE)
            x = x.reshape(b, 1, t, 3, gh // ms, ms, p, gw // ms, ms, p)
            x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).reshape(b * gh * gw, 3 * t * p * p)
            kw["pixel_values"] = x
            kw["image_grid_thw"] = torch.tensor([[1, gh, gw]] * b, dtype=torch.long, device=dev)
            types = [1 if tok == IMAGE_TOKEN else 0 for tok in self.layout] + [0] * ids.shape[1]
            kw["mm_token_type_ids"] = torch.tensor([types] * b, dtype=torch.int32, device=dev)
            m.model.rope_deltas = None
        elif f == "llava_next":
            kw["pixel_values"] = torch.stack([pixels, pixels], 1)      # base image and the one 32 x 32 crop
            kw["image_sizes"] = [[IMAGE, IMAGE]] * b       # Python ints: the anyres grid is fixed per program
        elif f in ("idefics3", "smolvlm"):
            kw["pixel_values"] = pixels.unsqueeze(1)                    # one image per row
        elif f == "gemma3":
            kw["pixel_values"] = pixels
            types = [1 if tok == IMAGE_TOKEN else 0 for tok in self.layout] + [0] * ids.shape[1]
            kw["token_type_ids"] = torch.tensor([types] * b, dtype=torch.long, device=dev)
        else:
            kw["pixel_values"] = pixels
        out = m(input_ids=full, use_cache=True, **kw)
        step = torch.full((b, 1), STEP_TOKEN, dtype=torch.long, device=dev)
        skw = {}
        if f == "gemma3":
            skw["token_type_ids"] = torch.zeros((b, 1), dtype=torch.long, device=dev)
        if f in ("qwen2_5_vl", "qwen3_vl"):
            skw["mm_token_type_ids"] = torch.zeros((b, 1), dtype=torch.int32, device=dev)
        nxt = m(input_ids=step, past_key_values=out.past_key_values, use_cache=True,
                cache_position=torch.tensor([n], device=dev), **skw)
        return (nxt.logits, out.logits)

    def forward(self, x, y=None):
        xs = (x,) if y is None else (x, y)
        if self.kind == "seq2seq":
            return self._decode({"input_ids": xs[0]})
        if self.kind == "speech_decode":
            return self._decode({"input_features": xs[0]})
        if self.kind == "embed":     # sentence embedding: masked mean pooling, then L2 normalization
            ids, mask = xs
            ids = torch.where(mask.bool(), ids, torch.full_like(ids, self.pad_id))
            h = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
            w = mask.unsqueeze(-1).to(h.dtype)
            mean = (h * w).sum(1) / w.sum(1).clamp(min=1e-9)
            return (torch.nn.functional.normalize(mean, p=2.0, dim=1), h)
        if self.kind == "audio":
            return (self.model(input_values=xs[0]).last_hidden_state,)
        if self.kind == "vlm":
            # the wrapper builds the prompt with exactly as many placeholders as the image has features,
            # so transformers' runtime count check (an .item() the IR cannot import, refused by name
            # otherwise) is switched off with its documented variable for the call; no value changes
            old = os.environ.get(DISABLE_CHECK_ENV)
            os.environ[DISABLE_CHECK_ENV] = "1" if self.disable_check else (old or "0")
            try:
                return self._vlm(*xs)
            finally:
                if old is None:
                    os.environ.pop(DISABLE_CHECK_ENV, None)
                else:
                    os.environ[DISABLE_CHECK_ENV] = old
        raise ValueError(self.kind)


# ----------------------------------------------------------------------------- configs

_ENC = dict(vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
            max_position_embeddings=64)
_DEC = dict(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False, pad_token_id=0,
            bos_token_id=1, eos_token_id=2)
_VIS = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, image_size=IMAGE,
            patch_size=8, num_channels=3)
_BART = dict(vocab_size=VOCAB, d_model=32, encoder_layers=2, decoder_layers=2, encoder_attention_heads=4,
             decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64, max_position_embeddings=64,
             pad_token_id=0, bos_token_id=1, eos_token_id=2)


def _entry(model_type: str, fn: Callable):
    fn.model_type = model_type
    return fn


def _t5():
    # Flan-T5 style (T5 v1.1): gated GELU feed forward, untied head
    c = transformers.T5Config(vocab_size=VOCAB, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_decoder_layers=2,
                              num_heads=4, relative_attention_num_buckets=8, relative_attention_max_distance=16,
                              feed_forward_proj="gated-gelu", tie_word_embeddings=False, decoder_start_token_id=0,
                              pad_token_id=0, eos_token_id=1)
    return transformers.T5ForConditionalGeneration(c), "seq2seq", {}


def _bart():
    c = transformers.BartConfig(**_BART, decoder_start_token_id=2, attn_implementation="sdpa")
    return transformers.BartForConditionalGeneration(c), "seq2seq", {}


def _marian():
    c = transformers.MarianConfig(**{**_BART, "eos_token_id": 1}, decoder_start_token_id=0,
                                  attn_implementation="sdpa")
    return transformers.MarianMTModel(c), "seq2seq", {}


def _whisper_decode():
    c = transformers.WhisperConfig(vocab_size=VOCAB, num_mel_bins=8, encoder_layers=2, decoder_layers=2,
                                   encoder_attention_heads=4, decoder_attention_heads=4, d_model=32,
                                   encoder_ffn_dim=64, decoder_ffn_dim=64, max_source_positions=8,
                                   max_target_positions=16, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                                   decoder_start_token_id=1, attn_implementation="sdpa")
    return transformers.WhisperForConditionalGeneration(c), "speech_decode", {}


def _bert():
    c = transformers.BertConfig(**_ENC, attn_implementation="sdpa")
    return transformers.BertModel(c, add_pooling_layer=False), "embed", {"pad_id": 0}


def _roberta():
    c = transformers.RobertaConfig(**{**_ENC, "max_position_embeddings": 66}, pad_token_id=1,
                                   attn_implementation="sdpa")
    return transformers.RobertaModel(c, add_pooling_layer=False), "embed", {"pad_id": 1}


def _xlm_roberta():
    c = transformers.XLMRobertaConfig(**{**_ENC, "max_position_embeddings": 66}, pad_token_id=1,
                                      attn_implementation="sdpa")
    return transformers.XLMRobertaModel(c, add_pooling_layer=False), "embed", {"pad_id": 1}


def _distilbert():
    c = transformers.DistilBertConfig(vocab_size=VOCAB, dim=32, n_layers=2, n_heads=4, hidden_dim=64,
                                      max_position_embeddings=64, pad_token_id=0, attn_implementation="sdpa")
    return transformers.DistilBertModel(c), "embed", {"pad_id": 0}


def _modernbert():
    # layer 0 global, layer 1 local with a window of 4 (two tokens each side), shorter than the sequence
    c = transformers.ModernBertConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, max_position_embeddings=64, global_attn_every_n_layers=2,
                                      local_attention=4, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                                      cls_token_id=1, sep_token_id=2, attn_implementation="sdpa")
    return transformers.ModernBertModel(c), "embed", {"pad_id": 0}


def _deberta_v3():
    # DeBERTa v3 settings: disentangled content to position and position to content attention with log
    # bucketed relative positions (8 buckets at this size), shared keys, layer normed relative embeddings
    c = transformers.DebertaV2Config(**_ENC, relative_attention=True, position_buckets=8, max_relative_positions=-1,
                                     pos_att_type=["p2c", "c2p"], norm_rel_ebd="layer_norm", share_att_key=True,
                                     position_biased_input=False, pad_token_id=0, type_vocab_size=0)
    return transformers.DebertaV2Model(c), "embed", {"pad_id": 0}


_AUDIO = dict(vocab_size=32, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
              conv_dim=(16, 16, 16), conv_stride=(5, 2, 2), conv_kernel=(10, 3, 3), conv_bias=False,
              num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4, mask_time_prob=0.0,
              mask_feature_prob=0.0, apply_spec_augment=False)


def _wav2vec2():
    # wav2vec2 base: group norm in the first convolution, post layer norm encoder
    c = transformers.Wav2Vec2Config(**_AUDIO, feat_extract_norm="group", do_stable_layer_norm=False)
    return transformers.Wav2Vec2Model(c), "audio", {}


def _hubert():
    # HuBERT large style: layer norm in every convolution, pre layer norm (stable) encoder
    c = transformers.HubertConfig(**_AUDIO, feat_extract_norm="layer", do_stable_layer_norm=True)
    return transformers.HubertModel(c), "audio", {}


def _vlm_entry(model_type: str, build_cfg: Callable[[], object], cls: str, layout: List[int]):
    def make():
        cfg = build_cfg()
        return getattr(transformers, cls)._from_config(cfg, attn_implementation="sdpa"), "vlm", \
            {"layout": layout, "family": model_type}
    return _entry(model_type, make)


def _cfg(model_type: str, **kw):
    from transformers import AutoConfig
    return AutoConfig.for_model(model_type, **kw)


def _llava_cfg(model_type: str, **extra):
    def f():
        return _cfg(model_type, vision_config={"model_type": "clip_vision_model", **_VIS},
                    text_config={"model_type": "llama", **_DEC}, image_token_index=IMAGE_TOKEN,
                    vision_feature_layer=-1, **extra)
    return f


def _qwen25_cfg():
    return _cfg("qwen2_5_vl",
                text_config={**_DEC, "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                                          "mrope_section": [1, 1, 2]}},
                vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4, patch_size=8,
                                   spatial_merge_size=2, temporal_patch_size=2, window_size=16, out_hidden_size=32,
                                   fullatt_block_indexes=[1]),
                image_token_id=IMAGE_TOKEN, video_token_id=VIDEO_TOKEN, vision_start_token_id=START_TOKEN,
                vision_end_token_id=END_TOKEN)


def _qwen3_cfg():
    return _cfg("qwen3_vl",
                text_config={**_DEC, "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                                          "mrope_section": [1, 1, 2], "mrope_interleaved": True}},
                vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4, patch_size=8,
                                   spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=32,
                                   num_position_embeddings=16, deepstack_visual_indexes=[0]),
                image_token_id=IMAGE_TOKEN, video_token_id=VIDEO_TOKEN, vision_start_token_id=START_TOKEN,
                vision_end_token_id=END_TOKEN)


def _gemma3_cfg():
    return _cfg("gemma3",
                text_config={**_DEC, "sliding_window": 4, "layer_types": ["sliding_attention", "full_attention"],
                             "query_pre_attn_scalar": 8},
                vision_config={"model_type": "siglip_vision_model", **_VIS, "vision_use_head": False},  # as Gemma 3
                mm_tokens_per_image=4, boi_token_index=START_TOKEN, eoi_token_index=END_TOKEN,
                image_token_index=IMAGE_TOKEN)


def _idefics3_cfg(model_type: str, vision_type: str):
    def f():
        return _cfg(model_type, vision_config={"model_type": vision_type, **_VIS},
                    text_config={"model_type": "llama", **_DEC}, image_token_id=IMAGE_TOKEN, scale_factor=2)
    return f


# 32 x 32 image, patch 8: a 4 x 4 grid of patches
_QWEN_LAYOUT = [START_TOKEN] + [IMAGE_TOKEN] * 4 + [END_TOKEN]           # 2 x 2 merge: 4 tokens
_LLAVA_LAYOUT = [IMAGE_TOKEN] * 16                                       # every patch, class token dropped
_LLAVA_NEXT_LAYOUT = [IMAGE_TOKEN] * 36                                  # base 16 + crop 16 + 4 row ends
_GEMMA3_LAYOUT = [START_TOKEN] + [IMAGE_TOKEN] * 4 + [END_TOKEN]         # pooled to mm_tokens_per_image
_IDEFICS3_LAYOUT = [IMAGE_TOKEN] * 4                                     # pixel shuffle by 2


MODELS_TEXT: Dict[str, Callable] = {
    # (1) encoder decoder, with the cross attention cache in the decode step
    "t5": _entry("t5", _t5),
    "bart": _entry("bart", _bart),
    "marian": _entry("marian", _marian),
    # (2) embedding and retrieval encoders, mean pooled
    "bert_embed": _entry("bert", _bert),
    "roberta_embed": _entry("roberta", _roberta),
    "xlm_roberta_embed": _entry("xlm-roberta", _xlm_roberta),
    "distilbert_embed": _entry("distilbert", _distilbert),
    "modernbert_embed": _entry("modernbert", _modernbert),
    "deberta_v3_embed": _entry("deberta-v2", _deberta_v3),
    # (3) speech
    "wav2vec2": _entry("wav2vec2", _wav2vec2),
    "hubert": _entry("hubert", _hubert),
    "whisper_decode": _entry("whisper", _whisper_decode),
    # (4) vision language models, image tokens merged into the text, prefill and one decode step
    "qwen2_5_vl": _vlm_entry("qwen2_5_vl", _qwen25_cfg, "Qwen2_5_VLForConditionalGeneration", _QWEN_LAYOUT),
    "qwen3_vl": _vlm_entry("qwen3_vl", _qwen3_cfg, "Qwen3VLForConditionalGeneration", _QWEN_LAYOUT),
    "llava": _vlm_entry("llava", _llava_cfg("llava"), "LlavaForConditionalGeneration", _LLAVA_LAYOUT),
    "llava_next": _vlm_entry("llava_next", _llava_cfg("llava_next", image_grid_pinpoints=[[IMAGE, IMAGE]]),
                             "LlavaNextForConditionalGeneration", _LLAVA_NEXT_LAYOUT),
    "gemma3_vlm": _vlm_entry("gemma3", _gemma3_cfg, "Gemma3ForConditionalGeneration", _GEMMA3_LAYOUT),
    "idefics3": _vlm_entry("idefics3", _idefics3_cfg("idefics3", "idefics3_vision"),
                           "Idefics3ForConditionalGeneration", _IDEFICS3_LAYOUT),
    "smolvlm": _vlm_entry("smolvlm", _idefics3_cfg("smolvlm", "smolvlm_vision"),
                          "SmolVLMForConditionalGeneration", _IDEFICS3_LAYOUT),
}

# Families that do not go through every stage, with the measured reason (tests keep them failing as
# documented, so a fix shows up as a test to update). Measured 2026-11-24 on transformers 5.24: in
# each, torch.export itself cannot trace the model at a fixed image size, because the image layout is
# read from tensor values inside the model; Truestar refuses with the model line named
# (ir/importer_text.explain_export_failure).
KNOWN_GAPS_TEXT: Dict[str, str] = {
    "qwen2_5_vl": "the vision encoder's window attention reads the patch grid from image_grid_thw with "
                  "int() (transformers vision_utils.py:162, get_vision_window_index); the patch split "
                  "per image and the 3D rope index also use .tolist() of tensors",
    "qwen3_vl": "the vision encoder's attention splits the patches per image with "
                "torch.split(t, lengths.tolist()) from image_grid_thw (modeling_qwen3_vl.py:304)",
    "idefics3": "all zero padding images are dropped with a boolean selection pixel_values[real_images_inds] "
                "(modeling_idefics3.py:557), so the image count is data dependent and the vision "
                "attention guards on it",
    "smolvlm": "as Idefics3: pixel_values[real_images_inds] (modeling_smolvlm.py:480)",
}

# Kinds whose program has a symbolic batch (the vision language wrappers fix the batch: the patch
# rows of Qwen's pixel input are batch times patches, and the prompt layout is a constant per row)
DYNAMIC_BATCH = {"seq2seq", "speech_decode", "embed", "audio"}


def available(table=None) -> Dict[str, Callable]:
    """The entries whose model type this transformers knows."""
    known = _model_types()
    table = MODELS_TEXT if table is None else table
    return {k: v for k, v in table.items() if getattr(v, "model_type", None) in known}


def example_inputs(kind: str, batch: int, seed: int = 0) -> Tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    if kind == "seq2seq":
        return (torch.randint(3, VOCAB - 6, (batch, SEQ), generator=g),)
    if kind == "speech_decode":          # [B, mel bins, 2 * max_source_positions]
        return (torch.randn(batch, 8, 16, generator=g),)
    if kind == "embed":                  # row i ends in i % 3 padding positions
        ids = torch.randint(3, VOCAB - 6, (batch, SEQ), generator=g)
        mask = torch.ones(batch, SEQ, dtype=torch.long)
        for i in range(batch):
            if i % 3:
                mask[i, SEQ - i % 3:] = 0
        return (ids, mask)
    if kind == "audio":
        return (torch.randn(batch, WAVE, generator=g),)
    if kind == "vlm":
        return (torch.randint(3, VOCAB - 6, (batch, TEXT), generator=g),
                torch.randn(batch, 3, IMAGE, IMAGE, generator=g))
    raise ValueError(kind)


def dynamic_shapes(kind: str, args):
    """``dynamic_shapes`` for ``ir.import_module``: the batch dimension of every input, or None."""
    if kind not in DYNAMIC_BATCH:
        return None
    B = torch.export.Dim("B", min=2, max=64)
    return dict(zip(("x", "y"), ({0: B} for _ in args)))


def build(name: str, batch: int = 2, seed: int = 0, dtype: torch.dtype = torch.float32):
    """(wrapped model in eval mode, example args). Weights come from ``torch.manual_seed(seed)``."""
    torch.manual_seed(seed)
    model, kind, extra = MODELS_TEXT[name]()
    model = TextWrap(model, kind, **extra).eval().to(dtype)
    return model, example_inputs(kind, batch, seed)


def kind_of(name: str) -> str:
    """The wrapper kind of an entry, without building the model twice in a test."""
    return {"t5": "seq2seq", "bart": "seq2seq", "marian": "seq2seq", "whisper_decode": "speech_decode",
            "wav2vec2": "audio", "hubert": "audio"}.get(name, "embed" if name.endswith("_embed") else "vlm")
