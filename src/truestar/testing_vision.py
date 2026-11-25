"""Small image generation, detection and segmentation models with random weights. Nothing is downloaded.

Used by tests/test_vision_models.py and scripts/op_coverage.py (source ``gen``), docs/MODEL_COVERAGE.md
section "Image generation, detection and segmentation". Each entry builds a model from a config in
code (``diffusers``, ``torchvision`` or ``transformers``), wraps it so ``forward`` takes tensors and
returns a tuple of tensors, and gives example inputs whose first dim is the batch.

    from truestar.testing_vision import GEN, VISION, build
    model, args = build("sd_unet")

The libraries are optional: a table entry whose library is missing is left out by ``available()``.
"""

from __future__ import annotations

import importlib.util
from typing import Callable, Dict, Tuple

import torch
from torch import nn


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _gen(batch: int, seed: int):
    return torch.Generator().manual_seed(seed)


# ----------------------------------------------------------------------------- image generation (diffusers)
#
# Latents 4 x 8 x 8, text tokens 6 of width 32, two resolution levels of widths 32 and 64, group norm
# of 8 groups, heads of 8: every block type of the full size model is live at that size.

TEXT = 6


class _Unet(nn.Module):
    """``forward(latents, timestep, text[, pooled text, time ids]) -> (noise prediction,)``."""

    def __init__(self, unet: nn.Module, sdxl: bool):
        super().__init__()
        self.unet, self.sdxl = unet, sdxl

    def forward(self, sample, timestep, text, text_embeds=None, time_ids=None):
        added = {"text_embeds": text_embeds, "time_ids": time_ids} if self.sdxl else None
        return (self.unet(sample, timestep, encoder_hidden_states=text, added_cond_kwargs=added,
                          return_dict=False)[0],)


def _sd_unet():
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1, block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32, attention_head_dim=8, norm_num_groups=8)
    return _Unet(unet, sdxl=False)


def _sdxl_unet():
    # SDXL: the pooled text embedding and the six size and crop numbers enter the time embedding
    # ("text_time"), linear projections in the transformer blocks, two transformer layers per block
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1, block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32, attention_head_dim=8, norm_num_groups=8, transformer_layers_per_block=(1, 2),
        use_linear_projection=True, addition_embed_type="text_time", addition_time_embed_dim=8,
        projection_class_embeddings_input_dim=16 + 6 * 8)
    return _Unet(unet, sdxl=True)


def _unet_inputs(sdxl: bool):
    def make(batch: int, seed: int):
        g = _gen(batch, seed)
        args = (torch.randn(batch, 4, 8, 8, generator=g), torch.rand(batch, generator=g) * 999,
                torch.randn(batch, TEXT, 32, generator=g))
        if sdxl:
            args += (torch.randn(batch, 16, generator=g), torch.rand(batch, 6, generator=g) * 64)
        return args
    return make


class _DiT(nn.Module):
    """DiT (class conditioned, adaptive layer norm zero): ``forward(latents, timestep, labels)``."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sample, timestep, labels):
        return (self.model(sample, timestep=timestep, class_labels=labels, return_dict=False)[0],)


def _dit():
    from diffusers import DiTTransformer2DModel
    return _DiT(DiTTransformer2DModel(num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=8,
                                      num_layers=2, sample_size=8, patch_size=2, num_embeds_ada_norm=10,
                                      norm_num_groups=8))


def _dit_inputs(batch: int, seed: int):
    g = _gen(batch, seed)
    return (torch.randn(batch, 4, 8, 8, generator=g), torch.rand(batch, generator=g) * 999,
            torch.randint(0, 10, (batch,), generator=g))


class _Flux(nn.Module):
    """Flux: double stream and single stream blocks, 3 axis RoPE from position ids, packed latents."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        h = w = 4
        img = torch.zeros(h, w, 3)
        img[..., 1] = torch.arange(h)[:, None]
        img[..., 2] = torch.arange(w)[None, :]
        self.register_buffer("img_ids", img.reshape(h * w, 3), persistent=False)
        self.register_buffer("txt_ids", torch.zeros(TEXT, 3), persistent=False)

    def forward(self, latents, text, pooled, timestep):
        return (self.model(hidden_states=latents, encoder_hidden_states=text, pooled_projections=pooled,
                           timestep=timestep, img_ids=self.img_ids, txt_ids=self.txt_ids, return_dict=False)[0],)


def _flux():
    from diffusers import FluxTransformer2DModel
    return _Flux(FluxTransformer2DModel(patch_size=1, in_channels=16, num_layers=1, num_single_layers=1,
                                        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=32,
                                        pooled_projection_dim=16, guidance_embeds=False, axes_dims_rope=(2, 2, 4)))


def _flux_inputs(batch: int, seed: int):
    g = _gen(batch, seed)
    return (torch.randn(batch, 16, 16, generator=g), torch.randn(batch, TEXT, 32, generator=g),
            torch.randn(batch, 16, generator=g), torch.rand(batch, generator=g))


def _vae_model():
    from diffusers import AutoencoderKL
    return AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2,
                         up_block_types=("UpDecoderBlock2D",) * 2, block_out_channels=(16, 32), layers_per_block=1,
                         latent_channels=4, norm_num_groups=8, sample_size=16)


class _VaeEncode(nn.Module):
    """The posterior's mean and log variance (sampling it is the caller's noise)."""

    def __init__(self):
        super().__init__()
        self.vae = _vae_model()

    def forward(self, image):
        dist = self.vae.encode(image).latent_dist
        return (dist.mean, dist.logvar)


class _VaeDecode(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = _vae_model()

    def forward(self, latents):
        return (self.vae.decode(latents, return_dict=False)[0],)


class _SchedulerStep(nn.Module):
    """One sampler step at a fixed step of a 10 step schedule, as a pipeline calls it:
    ``forward(noise prediction, latents) -> (scaled model input, previous latents, predicted clean
    latents)``. No weights: the scheduler's tables become constants. They follow the module's dtype
    like weights (``.double()`` converts them), so the float64 copy computes the step in float64;
    diffusers keeps them in float32. DDIM takes the timestep as a Python int, as after ``int(t)``
    (indexing the tables with the timestep tensor is a data dependent read torch.export cannot
    trace); Euler takes the timestep tensor and a preset step index."""

    STEP = 3
    TABLES = ("alphas_cumprod", "final_alpha_cumprod", "sigmas")

    def __init__(self, kind: str):
        super().__init__()
        import diffusers
        self.kind = kind
        cls = diffusers.DDIMScheduler if kind == "ddim" else diffusers.EulerDiscreteScheduler
        self.s = cls(num_train_timesteps=1000, beta_schedule="scaled_linear")
        self.s.set_timesteps(10)
        self.t = int(self.s.timesteps[self.STEP])

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        for k in self.TABLES:
            v = getattr(self.s, k, None)
            if torch.is_tensor(v) and v.is_floating_point():
                setattr(self.s, k, fn(v))
        return self

    def forward(self, eps, sample):
        s = self.s
        if self.kind == "ddim":
            scaled = s.scale_model_input(sample, self.t)          # the identity for DDIM
            out = s.step(eps, self.t, sample, eta=0.0)
        else:
            s._step_index = self.STEP          # Euler keeps a step index and advances it in step()
            t = s.timesteps[self.STEP]         # a tensor, as Euler requires
            scaled = s.scale_model_input(sample, t)               # sample / sqrt(sigma^2 + 1)
            out = s.step(eps, t, sample)
        return (scaled, out.prev_sample, out.pred_original_sample)


def _latent_pair(batch: int, seed: int):
    g = _gen(batch, seed)
    return (torch.randn(batch, 4, 8, 8, generator=g), torch.randn(batch, 4, 8, 8, generator=g))


def _image(size: int):
    def make(batch: int, seed: int):
        return (torch.rand(batch, 3, size, size, generator=_gen(batch, seed)),)
    return make


def _latents(batch: int, seed: int):
    return (torch.randn(batch, 4, 8, 8, generator=_gen(batch, seed)),)


# name: (library, builder, inputs(batch, seed))
GEN: Dict[str, Tuple[str, Callable[[], nn.Module], Callable]] = {
    "sd_unet": ("diffusers", _sd_unet, _unet_inputs(False)),
    "sdxl_unet": ("diffusers", _sdxl_unet, _unet_inputs(True)),
    "dit": ("diffusers", _dit, _dit_inputs),
    "flux": ("diffusers", _flux, _flux_inputs),
    "vae_encode": ("diffusers", _VaeEncode, _image(16)),
    "vae_decode": ("diffusers", _VaeDecode, _latents),
    "ddim_step": ("diffusers", lambda: _SchedulerStep("ddim"), _latent_pair),
    "euler_step": ("diffusers", lambda: _SchedulerStep("euler"), _latent_pair),
}


# ----------------------------------------------------------------------------- detection and segmentation

DET_SIZE = 64


class _Seg(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return (self.model(x)["out"],)


def _fcn():
    from torchvision.models.segmentation import fcn_resnet50
    return _Seg(fcn_resnet50(weights=None, weights_backbone=None, num_classes=4, aux_loss=False))


def _deeplabv3():
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    return _Seg(deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=4, aux_loss=False))


def _det_kwargs():
    # the transform resizes to a fixed size (no resize at DET_SIZE) and keeps few proposals
    return dict(weights=None, weights_backbone=None, num_classes=5, min_size=DET_SIZE, max_size=DET_SIZE)


def _fasterrcnn():
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    return fasterrcnn_mobilenet_v3_large_fpn(**_det_kwargs(), rpn_pre_nms_top_n_test=50, rpn_post_nms_top_n_test=20)


def _retinanet():
    from torchvision.models.detection import retinanet_resnet50_fpn
    return retinanet_resnet50_fpn(**_det_kwargs())


def _maskrcnn():
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    return maskrcnn_resnet50_fpn(**_det_kwargs(), rpn_pre_nms_top_n_test=50, rpn_post_nms_top_n_test=20)


class _Det(nn.Module):
    """The full eval forward, boxes, scores and labels of the first image (post processing included)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(list(x))[0]
        return tuple(out[k] for k in sorted(out))


class _DetDense(nn.Module):
    """Backbone, FPN and the dense heads on an image of the fixed size, normalized as the detector's
    transform does: everything before the data dependent proposal selection. Faster and Mask R-CNN:
    the RPN objectness and box deltas per level; RetinaNet: the class logits and box regression of
    every anchor. Holds only those modules (the detector's selection and post processing classes are
    refused by name, TS1260)."""

    def __init__(self, model):
        super().__init__()
        t = model.transform
        self.register_buffer("mean", torch.tensor(t.image_mean)[:, None, None], persistent=False)
        self.register_buffer("std", torch.tensor(t.image_std)[:, None, None], persistent=False)
        self.backbone = model.backbone
        self.rpn = hasattr(model, "rpn")
        self.head = model.rpn.head if self.rpn else model.head

    def forward(self, x):
        feats = self.backbone((x - self.mean) / self.std)
        feats = list(feats.values()) if isinstance(feats, dict) else [feats]
        if self.rpn:
            obj, deltas = self.head(feats)
            return tuple(obj) + tuple(deltas)
        out = self.head(feats)
        return (out["cls_logits"], out["bbox_regression"])


class _RoiHeads(nn.Module):
    """The second stage on pooled RoI features: box head and predictor (and the mask head and
    predictor for Mask R-CNN), ``forward(box features[N, C, 7, 7][, mask features[N, C, 14, 14]])``."""

    def __init__(self, model):
        super().__init__()
        r = model.roi_heads
        self.box_head, self.box_predictor = r.box_head, r.box_predictor
        self.mask_head, self.mask_predictor = r.mask_head, r.mask_predictor

    def forward(self, box_feats, mask_feats=None):
        cls, box = self.box_predictor(self.box_head(box_feats))
        if mask_feats is None:
            return (cls, box)
        return (cls, box, self.mask_predictor(self.mask_head(mask_feats)))


def _dense(make):
    return lambda: _DetDense(make())


def _roi_inputs(mask: bool):
    def make(batch: int, seed: int):
        g = _gen(batch, seed)
        n = 2 * batch            # RoIs, the batch dim of the second stage
        args = (torch.randn(n, 256, 7, 7, generator=g),)
        if mask:
            args += (torch.randn(n, 256, 14, 14, generator=g),)
        return args
    return make


class _Detr(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(pixel_values=x)
        return (out.logits, out.pred_boxes)


def _detr():
    import transformers
    backbone = transformers.ResNetConfig(embedding_size=16, hidden_sizes=[16, 32], depths=[1, 1],
                                         layer_type="basic", out_features=["stage2"])
    cfg = transformers.DetrConfig(use_timm_backbone=False, backbone_config=backbone, use_pretrained_backbone=False,
                                  d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=4,
                                  decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64,
                                  num_queries=8, num_labels=5)
    return _Detr(transformers.DetrForObjectDetection(cfg))


class _Sam(nn.Module):
    """SAM: the ViT image encoder, the prompt encoder on one point per image and the two way mask
    decoder; ``forward(pixels, points[B, 1, 1, 2]) -> (masks, iou scores)``."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, points):
        labels = torch.ones(points.shape[:3], dtype=torch.long, device=x.device)
        out = self.model(pixel_values=x, input_points=points, input_labels=labels, multimask_output=True)
        return (out.pred_masks, out.iou_scores)


def _sam():
    import transformers
    vision = transformers.SamVisionConfig(hidden_size=32, output_channels=16, num_hidden_layers=2,
                                          num_attention_heads=4, image_size=64, patch_size=8, mlp_dim=64,
                                          window_size=2, global_attn_indexes=[1], num_pos_feats=8)
    prompt = transformers.SamPromptEncoderConfig(hidden_size=16, image_size=64, patch_size=8, mask_input_channels=4)
    decoder = transformers.SamMaskDecoderConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                                                mlp_dim=32, iou_head_hidden_dim=16)
    cfg = transformers.SamConfig(vision_config=vision.to_dict(), prompt_encoder_config=prompt.to_dict(),
                                 mask_decoder_config=decoder.to_dict())
    return _Sam(transformers.SamModel(cfg))


def _sam_inputs(batch: int, seed: int):
    g = _gen(batch, seed)
    return (torch.rand(batch, 3, 64, 64, generator=g), torch.rand(batch, 1, 1, 2, generator=g) * 64)


VISION: Dict[str, Tuple[str, Callable[[], nn.Module], Callable]] = {
    "fcn_resnet50": ("torchvision", _fcn, _image(DET_SIZE)),
    "deeplabv3_mobilenet": ("torchvision", _deeplabv3, _image(DET_SIZE)),
    # dense parts of the detectors: backbone, FPN and the first stage heads
    "fasterrcnn_mobilenet_dense": ("torchvision", _dense(_fasterrcnn), _image(DET_SIZE)),
    "retinanet_dense": ("torchvision", _dense(_retinanet), _image(DET_SIZE)),
    "maskrcnn_dense": ("torchvision", _dense(_maskrcnn), _image(DET_SIZE)),
    # the second stage on pooled RoI features
    "fasterrcnn_mobilenet_roi": ("torchvision", lambda: _RoiHeads(_fasterrcnn()), _roi_inputs(False)),
    "maskrcnn_roi": ("torchvision", lambda: _RoiHeads(_maskrcnn()), _roi_inputs(True)),
    "detr": ("transformers", _detr, _image(DET_SIZE)),
    "sam": ("transformers", _sam, _sam_inputs),
}

# the full eval forward of the detectors, proposal selection and post processing included
DETECTORS: Dict[str, Tuple[str, Callable[[], nn.Module], Callable]] = {
    "fasterrcnn_mobilenet": ("torchvision", lambda: _Det(_fasterrcnn()), _image(DET_SIZE)),
    "retinanet": ("torchvision", lambda: _Det(_retinanet()), _image(DET_SIZE)),
    "maskrcnn": ("torchvision", lambda: _Det(_maskrcnn()), _image(DET_SIZE)),
}

ALL = {**GEN, **VISION, **DETECTORS}


# Entries that need a newer library than their table's: Flux's transformer arrived in diffusers 0.30, and
# transformers 4.x's SAM mask decoder branches in Python on a tensor value (torch.export stops on it;
# measured on 4.48), which 5.x no longer does. Below these versions the entry is not available.
MIN_VERSION: Dict[str, Tuple[str, str]] = {"flux": ("diffusers", "0.30"), "sam": ("transformers", "5.0")}


def _new_enough(name: str) -> bool:
    if name not in MIN_VERSION:
        return True
    import importlib.metadata
    import re

    def release(v: str) -> Tuple[int, ...]:     # "0.43.0.dev0" -> (0, 43, 0)
        return tuple(int(p) for p in re.findall(r"\d+", v.split("+")[0])[:3])

    lib, low = MIN_VERSION[name]
    try:
        return release(importlib.metadata.version(lib)) >= release(low)
    except importlib.metadata.PackageNotFoundError:
        return False


def available(table) -> Dict[str, Tuple]:
    """The entries of ``table`` whose library is installed (at ``MIN_VERSION`` where one is listed)."""
    return {k: v for k, v in table.items() if _has(v[0]) and _new_enough(k)}


def build(name: str, batch: int = 2, seed: int = 0, dtype: torch.dtype = torch.float32):
    """(wrapped model in eval mode, example args). Weights come from ``torch.manual_seed(seed)``."""
    _, make, inputs = ALL[name]
    torch.manual_seed(seed)
    model = make().eval().to(dtype)
    args = tuple(a.to(dtype) if a.is_floating_point() else a for a in inputs(batch, seed))
    return model, args

# Entries that do not import, with the measured reason (tests check they still fail the same way, so
# a fix shows up as a test to update)
KNOWN_GAPS = {
    "euler_step": "torch.export itself fails: EulerDiscreteScheduler.step branches in Python on a tensor value "
                  "(gamma = ... if s_tmin <= sigma <= s_tmax, sigma read from the sigmas table), "
                  "GuardOnDataDependentSymNode",
}
