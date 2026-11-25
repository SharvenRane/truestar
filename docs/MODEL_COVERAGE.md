# Hugging Face model coverage

Author: Sharven Rane. First written 2026-11-24.

Which real model families go through every Truestar stage, measured on models built from
`transformers` config classes in code with random weights (no download of any weight or config
file). Code: `src/truestar/testing_hf.py` (the configs), `scripts/hf_coverage.py` (the sweep),
`tests/test_hf_models.py` (fast tests per model, 67 since the op coverage work; with the current
families and refusals 185 on transformers 5.24, of which the CUDA ones skip without a GPU).
The ATen op coverage of these models, the torchvision zoo and the torch.nn layers is measured by
`scripts/op_coverage.py`, section "ATen op coverage" below.

```
PYTHONPATH=src python scripts/hf_coverage.py              # every model, CUDA when present
PYTHONPATH=src python scripts/hf_coverage.py --models llama,t5_encoder --no-cuda
```

Pain rows served: P7 (export of real models), P5 (one program for every batch size), P4 (the same
`truestar check` on real architectures), P46 (current model families).

## Current models (coverage refresh, 2026-11-24, transformers 5.24.0)

The first sweep below ran on transformers 4.48, which predates every 2025 to 2026 family. This
refresh reran it on **transformers 5.24.0** (the latest stable release, PyPI 2026-11-21), torch
2.11.0 CPU, torchvision 0.26.0, numba 0.68.1, Python 3.12, in a separate venv on the Linux server
(`/srv/truestar/venv-cov26`, CPU only). The Windows envs were not changed: `ml` and `ts` have
transformers 4.48.0, `cv` has 5.17.0 (checked 2026-11-24). Random weights, configs built in code,
nothing downloaded.

```
PYTHONPATH=src python scripts/op_coverage.py --sources hf,nn,cur --train --compile cpu --json after.json
PYTHONPATH=src python scripts/op_coverage.py --sources tv --train --compile cpu --json after_tv.json
```

**Families added** (`truestar.testing_hf.MODELS_2026`, 28; vocab 96, width 32, 2 or 3 layers, 4
query and 2 KV heads of 8, MLP 64, plus the settings that make each family's new parts live at that
size):

| Family (model type) | What is new in it, exercised here |
|---|---|
| Gemma 2 (`gemma2`) | local and global layers (window 4 of 6 tokens), attention and final logit soft capping (eager attention, softmax in float32), query scale from `query_pre_attn_scalar` |
| Gemma 3 (`gemma3_text`) | QK norm, local and global layers with their own RoPE bases, final soft capping |
| Gemma 3n (`gemma3n_text`) | AltUp, Laurel, per layer embeddings, KV shared by the last layer, Gaussian top k activation sparsity (`icdf(0.95)` in the forward) |
| Gemma 4 dense (`gemma4_text`) | per layer embeddings, KV shared by the last layer, proportional partial RoPE and a wider head on the global layers |
| Qwen3 (`qwen3`), Hunyuan dense (`hunyuan_v1_dense`), Apertus (`apertus`) | QK norm; Apertus: the xIELU activation |
| Phi-4 mini (`phi3`, 3/4 partial rotary), GLM 4 (`glm4`, half rotary, sandwich norms) | partial rotary |
| SmolLM3 (`smollm3`) | every second layer without RoPE |
| OLMo 2 and 3 (`olmo2`, `olmo3`) | norms after attention and MLP, QK norm; OLMo 3 local and global layers |
| Granite (`granite`), Cohere 2 (`cohere2`), EXAONE 4 (`exaone4`) | embedding, residual, attention and logit multipliers; Cohere 2's parallel block; local and global layers |
| ERNIE 4.5, Seed OSS, Arcee (ReLU squared), BitNet (ReLU squared, sub layer norms) | dense decoders of 2025 |
| Ministral 3 (`ministral3`) | YaRN with the Llama 4 style query scale (`floor` of the positions) |
| LFM2 (`lfm2`) | gated short convolution blocks next to attention |
| SigLIP vision, SigLIP2 vision, DINOv3 ViT, AIMv2 vision, SmolVLM vision, Pixtral vision, ConvNeXt V2 | the vision encoders of current multimodal models: `padding="valid"` patch convolutions, attention pooling heads, 2D RoPE with register tokens and layer scale (DINOv3), bucketized patch positions (SmolVLM), global response norm |

**Before and after.** Before: origin/main e96cd2c with the same model list (the fixture files
`testing_hf.py`, `testing_ops.py` and `scripts/op_coverage.py` of this branch copied onto it, so
the list is identical; CLIP text's wrapper fix is in the fixture). After: this branch. Both on
transformers 5.24.0, CPU. "compile" is `compile_program(prog, "cpu", "float32")` (numba kernels,
first call verified against the float64 reference) with no float64 CPU fallback; "train" is one
training step's gradient against torch.autograd in float64 (bar 1e-8, random cotangent).

| Models on transformers 5.24 | before (e96cd2c): import / compile / train | after: import / compile / train |
|---|---|---|
| the 15 Hugging Face models of the first sweep | 10 / 10 / 10 (Llama, Mistral, Qwen2, GPT-2 and Whisper refused at `aten.diff`) | 15 / 15 / 15 |
| the 28 families of 2025 to 2026 | 2 / 2 / 2 (AIMv2, ConvNeXt V2) | 26 / 26 / 26 |
| the 11 MoE, linear attention and chunked SSM families | 0, each refused with a list of unknown ops (one did not export) | 0, every one refused by name with TS940, TS941 or TS942 |
| the 20 ops and patterns of `testing_ops.current_ops` | 2 / 2 / 2 | 20 / 20 / 20 |
| the 92 torch.nn layers and losses | 92 / 92 / 91 | 92 / 92 / 91 (Hardsigmoid: torch's float32 1/6, unchanged) |
| imported program against PyTorch float64, worst | 3.1e-15 | 3.1e-15 (MobileViT); new families at most 1.3e-15 (Gemma 4) |

The headline number of the first sweep ("170 of 170") was measured on transformers 4.48 and stays
true there (the tests pass on both versions). On transformers 5.24 it was not true before this
branch: every decoder of the list, old or new, was refused at import, because transformers 5 calls
`torch.diff` on the position ids of every decoder to find packed sequences.

CUDA: not run for the new families (the PC GPU was in use by another benchmark; marked pending).
The CPU rows above use the same kernel IR and verification as the CUDA backend.

**Lowered since P60 (2026-11-24).** Mamba 2, Falcon H1, Qwen3.5, Bamba and Zamba 2 were refused here
(TS941, TS942); their SSD and Gated DeltaNet layers are now lowered to the verified
`linear_attention` op (`truestar.ir.hf_linattn`, docs/LINEAR_ATTENTION.md section 10). On
transformers 5.24 all five import, match PyTorch float64 within 6.1e-16 and train one step equal to
torch.autograd; on 4.48 (no module function to lower) they stay refused with the reason.

**Refused, planned** (`truestar.ir.refusals`, before any op is imported, on `import_module` and on
`import_exported`; the message names the layer class and its module path):

| Family | Code | Layer named in the refusal |
|---|---|---|
| gpt-oss | TS940 MoE routing | `GptOssTopKRouter` (also has attention sinks and 128 token windows, which import) |
| Qwen3 MoE | TS940 | `Qwen3MoeSparseMoeBlock` |
| Qwen3 Next | TS940 (also TS941) | `Qwen3NextSparseMoeBlock`, `Qwen3NextGatedDeltaNet` |
| Llama 4 | TS940 | `Llama4TextExperts` |
| DeepSeek V3 | TS940 | `DeepseekV3MoE` (MLA itself is not refused separately) |
| Kimi Linear | TS940 (also TS941) | `KimiLinearExperts`, `KimiDeltaAttention` |
| OLMoE | TS940 | `OlmoeSparseMoeBlock` |
| Nemotron H | TS940 (also TS942) | `NemotronHMoE` |

The rule matches exact class names first and then name parts (`SparseMoe`, `Experts`, `TopKRouter`,
`GatedDeltaNet`, `DeltaAttention`, `LightningAttention`, `Mamba2`), so a new family built from the
same layers is refused too; the Mamba 1 sequential scan (`MambaMixer`) and look alike names are
not. Planned: MoE with the near tie routing report (<research>/LIVE_RESEARCH_2026_11.md
4.3), chunked linear attention and SSD with chunk size invariance (5.2).

**Former gaps, closed 2026-11-25** (`ir/importer_vision.py`): SigLIP2 vision and Pixtral vision
were refused (SigLIP2: `item`, data dependent sizes, antialiased interpolation; Pixtral:
`torch.export` itself failed on `.item()` of `image_sizes`). Both read each image's patch grid from
a small integer tensor; when the model builds it from constants (a fixed image size, as the test
wrapper does), `import_module` and `truestar export` export again with torch's constant tracking
for tensors up to 256 elements, so every such size is a known integer. The reads torch.export
keeps after that only feed its own assertions and are dropped; a read of a value that depends on
an input is refused (TS861). Added lowerings: `_upsample_bilinear2d_aa` (constant matrices of
torch's antialiased triangle filter, within 1e-14 of torch up and down), `baddbmm`, `numpy_T`,
`floor_divide`, `empty` (zeros), and for transformers 5.24's loop free SigLIP2 resize `all`,
`prod.dim_int` and `div` with `rounding_mode`. Measured on transformers 5.24.0 (CPU, Modal) and
5.17.0 (CPU, local): both import and match PyTorch in float64 at batch 2 and 5 (below 1e-10),
export a verified artifact, compile on the CPU with no fallback and train one step equal to
torch.autograd in float64. Limit: a fixed batch (`testing_hf.FIXED_BATCH_2026`), because the 5.x
forward loops over the images in Python (SigLIP2 up to 5.17, Pixtral) and torch.export unrolls it;
Pixtral needs transformers 5 (the 4.x forward takes no batch).

**Ops added** (`ir/importer_current.py`, every one lowered onto IR ops that already have a float64
reference, reverse, forward and batching rules and kernels, so no new IR op and no new kernel rule):

| ATen op | Needed by | Lowering |
|---|---|---|
| `diff` | every transformers 5 decoder (packed sequence check) | `x[1:] - x[:-1]` (`ne` for booleans), with prepend and append |
| `square` | Arcee, BitNet | `x * x` |
| `floor` | Ministral 3 | truncation through int64 minus one where above x, below 2^52; `frozen` (gradient 0) |
| `bitwise_not` | the Mamba 2 family (refused) | `logical_not`, `-x - 1` on integers |
| `meshgrid`, `tile` | DINOv3 | reshapes and expands |
| `bucketize` | SmolVLM | the count of boundaries below each value (int64 or int32) |
| `erfinv` | Gemma 3n | the constant subgraph is run at import and folded in float64; of an input dependent value refused (TS860) |
| `std`, `var` | Gemma 3n | `sum((x - mean)^2) / (N - correction)` |
| `multiply`, `detach_`, `_is_all_true` | Gemma 3n (torch.distributions) | `mul`, identity, `all` |
| `conv2d.padding` and string padding | SigLIP, SmolVLM | `valid` is 0, `same` symmetric or an explicit pad first |
| `sum` and `mean` with a dtype, `sum` of booleans, `softmax` with a dtype | SmolVLM, Gemma 2 | cast first (as torch); a bool sum is int64 |

Two importer changes: an in place op on a slice of a tensor reached through identity aliases
(`contiguous`, a `type_as` or `to` that changes nothing, an earlier in place op) is rewritten as for
a fresh tensor (Gemma 3n's `corrected[1:] += x`), with the alias family check rooted at the fresh
tensor; and `exact_float32_casts` also keeps `dtype=torch.float32` arguments of functions given a
float64 tensor (Gemma 2's softmax) in float64, and creates float32 constants in float32 before
converting them (Gemma 3n's sparsity target), so the comparison matches the export's constants.

Patterns of the new families that needed no new op: attention sinks (a learned logit column joined
to the softmax and dropped after it), logit soft capping (`tanh(x / c) * c`), sliding windows
shorter than the sequence, QK norm, partial and proportional RoPE, per layer embeddings, KV shared
across layers, xIELU. `testing_ops.current_ops` checks sinks and soft capping alone (value,
gradient, vmap, forward mode).

Found on the torch side: Gemma 3n's float64 gradient from torch.autograd has NaN entries at the
test inputs (seed 3, batch 2: up to 1024 entries per weight, in the last layer's AltUp and Laurel
weights and the per layer inputs). Truestar's gradient has NaN at exactly the same entries and
agrees elsewhere; the cause was not traced.

## Image generation, detection and segmentation (2026-11-24)

The models most people run for images, built from configs in code with random weights (nothing
downloaded): `src/truestar/testing_vision.py` (tables `GEN`, `VISION`, `DETECTORS`,
`KNOWN_GAPS`). Run on the Linux server, CPU only: torch 2.11.0, torchvision 0.26.0, transformers
5.24.0, **diffusers 0.43.0** (installed into `/srv/truestar/venv-cov26` for this, PyPI latest on
2026-11-24), numba 0.68.1, Python 3.12.

```
PYTHONPATH=src python scripts/op_coverage.py --sources gen --compile cpu --train --json gen.json
PYTHONPATH=src python -m pytest -q tests/test_vision_models.py tests/test_vision_ops.py
```

Same bar as the 2025 to 2026 families above: "import" is `ir.import_module` of the float64 model
against PyTorch float64 (bar 1e-10 relative in the tests); "compile" is `compile_program(prog,
"cpu", "float32")`, whose first call is verified against the float64 reference, with **no float64
CPU fallback** allowed; "train" is one training step's gradient with respect to every trainable
weight (the inputs for the two scheduler steps, which have no weights) against torch.autograd in
float64 (bar 1e-10 in the tests, 1e-8 in the sweep).

**Sizes.** Diffusion: latents 4 x 8 x 8, text 6 tokens of width 32, two UNet levels of widths 32
and 64, group norm of 8 groups, heads of 8. Detection and segmentation: images 3 x 64 x 64, the
torchvision models at their real widths (`num_classes` 4 or 5), the detectors' transform set to
64 so no resize happens. DETR: a two stage ResNet backbone of widths 16 and 32, width 32, one
encoder and one decoder layer, 8 queries. SAM: a 64 x 64 image, patch 8, width 32, two ViT layers
(one windowed, one global), prompt and mask decoder width 16, one point prompt per image.

**Before and after.** Before: origin/main 9a9ad96 with this branch's `testing_vision.py` and
`scripts/op_coverage.py` copied onto it (the same model list). After: this branch. Each cell: import
(worst error against PyTorch float64) / compile CPU float32 / train (worst gradient error).

| Program (`testing_vision`) | Model | before | after |
|---|---|---|---|
| `sd_unet` | `UNet2DConditionModel`: cross attention down and up blocks, time embedding | PASS 2.6e-15 / PASS / PASS 2.1e-15 | same |
| `sdxl_unet` | the same class, SDXL style: `text_time` added conditioning (pooled text and six size numbers), linear projections, two transformer layers in a block | unsupported: `aten.concat` | PASS 1.4e-15 / PASS / PASS 1.5e-15 |
| `dit` | `DiTTransformer2DModel`: patches, adaptive layer norm zero, class labels | PASS 3.4e-16 / PASS / PASS 4.3e-16 | same |
| `flux` | `FluxTransformer2DModel`: double and single stream blocks, 3 axis RoPE, pooled text | PASS 3.4e-16 / PASS with 3 float64 CPU fallbacks (`pow`) / PASS 3.4e-16 | PASS 3.4e-16 / PASS, no fallback / PASS 3.4e-16 |
| `vae_encode` | `AutoencoderKL.encode`: posterior mean and log variance | PASS 2.7e-15 / PASS / PASS 2.7e-15 | same |
| `vae_decode` | `AutoencoderKL.decode` | PASS 2.5e-15 / PASS / PASS 2.2e-15 | same |
| `ddim_step` | `DDIMScheduler`: `scale_model_input` and `step` (eta 0) | PASS 1.1e-15 / PASS / PASS 0 | same |
| `euler_step` | `EulerDiscreteScheduler`: `scale_model_input` and `step` | torch.export fails | same (known gap, below) |
| `fcn_resnet50` | torchvision `fcn_resnet50`, `out` | PASS 5.5e-15 / PASS / PASS 2.2e-15 | same |
| `deeplabv3_mobilenet` | torchvision `deeplabv3_mobilenet_v3_large` (ASPP), `out` | PASS 1.1e-15 / PASS / PASS 2.5e-16 | same |
| `fasterrcnn_mobilenet_dense` | `fasterrcnn_mobilenet_v3_large_fpn`: normalize, backbone with FPN, RPN head (6 outputs) | PASS 3.0e-15 / PASS / PASS 8.9e-18 | same |
| `fasterrcnn_mobilenet_roi` | its box head and predictor on pooled RoI features | PASS 1.7e-15 / PASS / PASS 9.6e-16 | same |
| `retinanet_dense` | `retinanet_resnet50_fpn`: backbone with FPN, class and box heads of every anchor | PASS 6.5e-15 / PASS / PASS 4.0e-15 | same |
| `maskrcnn_dense` | `maskrcnn_resnet50_fpn`: backbone with FPN, RPN head (10 outputs) | PASS 7.5e-15 / PASS / PASS 3.6e-15 | same |
| `maskrcnn_roi` | its box head and predictor, mask head and predictor on pooled features | PASS 3.9e-15 / PASS / PASS 1.1e-15 | same |
| `detr` | transformers `DetrForObjectDetection` (ResNet backbone, sine positions, encoder, decoder, class and box heads) | unsupported: `aten.div.Tensor_mode` | PASS 2.1e-15 / PASS / PASS 6.1e-16 |
| `sam` | transformers `SamModel`: ViT encoder (windowed and global attention, relative positions), prompt encoder, two way mask decoder | PASS 1.5e-15 / PASS with 3 float64 CPU fallbacks (`cumsum` twice, `cat` of mixed dtypes) / PASS 1.2e-15 | PASS 1.5e-15 / PASS, no fallback / PASS 1.2e-15 |
| `fasterrcnn_mobilenet`, `retinanet`, `maskrcnn` | the full eval forward (boxes, labels, scores, masks) | torch.export fails: `GuardOnDataDependentSymNode` (`Eq(4*u1, 0)`; RetinaNet: a sum of `Min(1000, u_i)`) | refused by name, TS1260 |

Totals over the 16 programs that can be covered: before 14 import, 12 compile with no fallback,
14 train; after 16, 16, 16. The 82 distinct ATen ops of these models all import and all have a
float64 reference, reverse, forward and batching rule.

**Fixed on this branch** (each with a test in `tests/test_vision_ops.py` that fails on main):

| Gap on main | Models | Fix |
|---|---|---|
| `aten.concat` not imported | SDXL UNet (its `text_time` embedding joins the pooled text and the time ids with `torch.concat`) | `concat` and `concatenate` are `cat` (`ir/importer_vision.py`) |
| `aten.div.Tensor_mode` not imported | DETR (sine position embedding: `torch.div(arange, 2, rounding_mode="floor")`) | A quotient of constants is computed at import with `torch.div` itself; of an input it is refused with TS860 and the reason: torch's float floor division is Python's `//`, which differs from `floor(a / b)` (`1 // 0.1` is 9, `floor(1 / 0.1)` is 10) and the IR has no exact remainder to build it from. `floor_divide` the same; `rounding_mode=None` is `div` |
| float64 CPU fallback: `pow` with a tensor exponent | Flux (RoPE frequencies `theta ** (arange / dim)`, computed in the forward in float64) | A `pow` or `cumsum` whose value depends on no input and no model parameter is computed once at import by the float64 reference, the computation the reference does on every call, and held as a constant. A value that reaches an input or a weight keeps its node and its gradient (tested) |
| float64 CPU fallback: `cumsum` | SAM (image wide position grid, `ones(...).cumsum(0)`) | as above |
| float64 CPU fallback: `cat of mixed dtypes` | SAM (`_embed_points` joins the points with a padding point of another dtype) | The importer casts every `cat` input to the dtype torch.cat promotes to (from the exported node), so the IR cat is homogeneous and runs as a layout step |
| full detectors: `torch.export` fails with `GuardOnDataDependentSymNode: Could not guard on data-dependent expression Eq(4*u1, 0)` | Faster R-CNN, Mask R-CNN, RetinaNet | Refused by name before torch.export: **TS1260** `detection_postprocess_unsupported`, naming the module (`RegionProposalNetwork at model.rpn`, `RetinaNet at model`) and saying what to import instead |

**Refused, by name** (`truestar.ir.refusals`, TS1260; `import_module` and `import_exported`):
the eval forward of torchvision's detectors (`GeneralizedRCNN`, `FasterRCNN`, `MaskRCNN`,
`KeypointRCNN`, `RegionProposalNetwork`, `RoIHeads`, `RetinaNet`, `FCOS`, `SSD`). Proposal
selection and post processing (top k over anchors, score threshold, removal of small boxes, NMS,
RoI align on the kept boxes) return a number of boxes that depends on the values. What is covered
instead, measured above: the backbone with FPN and the RPN or RetinaNet head (`*_dense`, every
level's outputs), and the second stage on pooled RoI features (`*_roi`: box head and predictor,
and Mask R-CNN's mask head and predictor). The classes match only with their module
(`torchvision.models.detection`), so a backbone or head taken from these models (`BackboneWithFPN`,
`RPNHead`, `RetinaNetHead`, `TwoMLPHead`) is not refused, and neither is a class of the user's own
named `SSD` (the existing look alike test of the MoE refusals has one).

**Known gap** (`testing_vision.KNOWN_GAPS`, a test keeps it failing as documented): the Euler
step. `EulerDiscreteScheduler.step` branches in Python on a tensor value (`gamma = ... if s_tmin
<= sigma <= s_tmax`, where `sigma` is read from the `sigmas` table), so torch.export itself stops
(`GuardOnDataDependentSymNode`); nothing reaches Truestar. The DDIM step, whose branches read Python
numbers, passes.

Notes on the programs:

* The scheduler steps are measured as a pipeline calls them, at step 3 of a 10 step schedule
  (`scaled_linear` betas, 1000 training steps): `scale_model_input`, `step`, both outputs. DDIM
  takes the timestep as a Python int: indexing the tables with the timestep tensor is a data
  dependent read torch.export cannot trace. diffusers keeps the tables in float32; the wrapper
  converts them with the module's dtype like weights, so the float64 copy computes the step in
  float64 (without that, the torch float64 copy is 8.5e-8 off exact math, a float32 island).
* The VAE is measured as two programs: `encode` returns the posterior's mean and log variance
  (sampling from it is the caller's noise), `decode` maps latents to an image.
* Flux: one double stream and one single stream block, `patch_size` 1, 16 latent channels, the
  3 axis RoPE from fixed image and text position ids (`axes_dims_rope` 2, 2, 4), no guidance
  embedding.
* DETR's and SAM's post processing (`post_process_object_detection`, mask upscaling and
  thresholding) are processor functions outside the model's forward and were not measured.
* CUDA: not run (the PC GPU was in use by another benchmark). The CPU rows use the same kernel IR
  and verification as the CUDA backend.

## Models

transformers 4.48.0, torch 2.11, attention `sdpa` where the config takes it. Each model is wrapped
so `forward` takes one tensor and returns the logits or last hidden state.

| Name | Class | Size | Input |
|---|---|---|---|
| llama | `LlamaForCausalLM` | vocab 96, width 32, 2 layers, 4 query and 2 KV heads, MLP 64 | ids [B, 6] |
| mistral | `MistralForCausalLM` | as llama, sliding window 4 (shorter than the sequence, so the window mask is live) | ids [B, 6] |
| qwen2 | `Qwen2ForCausalLM` | as llama (q, k, v biases) | ids [B, 6] |
| gpt2 | `GPT2LMHeadModel` | vocab 96, width 32, 2 layers, 4 heads | ids [B, 6] |
| bert | `BertModel`, no pooler | vocab 96, width 32, 2 layers, 4 heads | ids [B, 6] |
| vit | `ViTModel`, no pooler | 32x32 image, patch 8, width 32, 2 layers | pixels [B, 3, 32, 32] |
| t5_encoder | `T5EncoderModel` | vocab 96, d_model 32, 2 layers, 4 heads, 8 relative position buckets | ids [B, 6] |
| whisper | `WhisperModel` (encoder and decoder) | 8 mel bins, 16 frames, width 32, 2 + 2 layers, 4 heads; decoder ids `arange(4)` | features [B, 8, 16] |
| clip_text | `CLIPTextModel` | vocab 96, width 32, 2 layers, 4 heads; hidden states and the end of text pooling | ids [B, 6] |
| clip_vision | `CLIPVisionModel` | 32x32 image, patch 8, width 32, 2 layers | pixels [B, 3, 32, 32] |
| deit | `DeiTModel`, no pooler | as vit (class and distillation tokens) | pixels [B, 3, 32, 32] |
| swin | `SwinModel`, no pooler | 32x32 image, patch 4, embed 16, depths 2 and 2, window 4 (shifted windows) | pixels [B, 3, 32, 32] |
| convnext | `ConvNextModel` | 2 stages, widths 16 and 32, one block each | pixels [B, 3, 32, 32] |
| mobilevit | `MobileViTModel` | 64x64 image, patch 2, hidden 32, 40, 48 | pixels [B, 3, 64, 64] |
| mamba | `MambaForCausalLM` (the sequential scan, no fused kernels) | vocab 96, width 32, state 4, 2 layers, conv kernel 4 | ids [B, 6] |

The last eight were added with the op coverage work (2026-11-24). Before it, Whisper and Mamba did
not import (`aten.conv1d`, `aten.softplus`), CLIP text did not import (its causal mask is a `full`
without a dtype, which the importer typed with the import time default dtype instead of the dtype
torch.export recorded), and none of the Hugging Face models except ViT, DeiT and MobileViT could be
differentiated (no reverse rule for `index`, `pow`, `cast`, `pad`); every one of the 15 now imports,
compiles on CUDA with no float64 CPU fallback, exports, and trains one step with gradients equal to
torch.autograd in float64 (`test_one_training_step_gradient_matches_torch_autograd`, bar 1e-10).
CLIP's float64 copy cannot run in plain PyTorch (the same `full` overflows float32), so its wrapper
sets the default dtype to the model's for the call.

## Stages

1. **check**: `truestar check` on CPU and CUDA (`check(model, args, backends=["cpu", "cuda"])`).
2. **import**: `ir.import_module` of the float64 model with a symbolic batch, then `ir.run` at batch
   2 and 5 against PyTorch float64. Bar 1e-10 relative to the largest output.
3. **compile**: `compile_program(program, "cuda", "float32")`, called at batch 3. The first call is
   checked against the float64 reference interpreter (kernel, region and end to end), so a pass
   means verified. The count of steps that took the float64 CPU fallback is given.
4. **export**: `truestar.export.export` (float64 verification at the example batch and a second
   batch, bar 1e-9), the artifact written and run.

## Results

Before: main at ae8e73f. After: this branch. RTX 5070 Ti, CUDA 12.8, Windows.

| Model | check before / after | import before | import after (worst error) | compile before | compile after | export before | export after |
|---|---|---|---|---|---|---|---|
| llama | OK / OK | refused: `wrap_with_set_grad_enabled` | PASS 2.7e-16 | blocked by import | PASS, 1 fallback (`triu`) | refused, same op | PASS |
| mistral | OK / OK | refused: `aten.bitwise_or_.Tensor` | PASS 2.6e-16 | blocked by import | PASS, 0 fallbacks | refused, same op | PASS |
| qwen2 | OK / OK | refused: `wrap_with_set_grad_enabled` | PASS 5.7e-16 | blocked by import | PASS, 1 fallback (`triu`) | refused, same op | PASS |
| gpt2 | OK / OK | PASS 4.6e-16 | PASS 4.6e-16 | error: float64 mask value overflows float32 in `full` | PASS, 0 fallbacks | PASS | PASS |
| bert | OK / OK | PASS 2.5e-16 | PASS 2.5e-16 | PASS, 5 fallbacks (`cast`, `index`) | PASS, 1 fallback (cast to an integer) | PASS | PASS |
| vit | OK / OK | PASS 7.2e-16 | PASS 7.2e-16 | PASS, 0 fallbacks | PASS, 0 fallbacks | PASS | PASS |
| t5_encoder | OK / OK | 1.1e-7 against the float64 copy (see float32 islands) | PASS 3.5e-16 | PASS, 28 fallbacks (`abs add cast index log minimum mul pow sub where`) | PASS, 2 fallbacks (casts to an integer) | refused (7e-8, bar 1e-9) | PASS |

`truestar check` passed on all seven before and after: it compares backends with the model as is,
so the import gaps did not touch it.

First blocking problem per model before, and what fixed it:

| Problem | Models | Fix |
|---|---|---|
| `torch.no_grad()` and `torch.autocast(enabled=False)` around the rotary embedding export as higher order ops (`wrap_with_set_grad_enabled`, `wrap_with_autocast`) | llama, qwen2 (and mistral behind its first error) | The importer inlines their body graphs into a new graph before importing (`_inline_wrappers`); the exported program is not changed. Neither changes a value in inference. An enabled autocast region changes dtypes and is refused by name |
| In place `bitwise_or_` on the sliding window mask | mistral | Mapped like the other in place ops (`bitwise_and_`, `bitwise_or_`, `__iand__`, `__ior__` to `logical_and` and `logical_or`); the driver's check that nobody reads the old value applies |
| Float32 islands: the rotary embedding and T5's layer norm call `.float()` on purpose, so the float64 copy of the model computes those parts in float32, 1e-7 away from exact math, while the imported program (whose reference treats a cast to float32 as exact) is right | llama, mistral, qwen2, t5_encoder | `ir.reference.exact_float32_casts()`: a `TorchFunctionMode` under which `.float()`, `.to(torch.float32)` and `.type(torch.float32)` convert to float64, so PyTorch computes the same exact math. The exporter runs its PyTorch side in it and records the count per check (`float32_casts_kept_float64` in the evidence). 16 bit casts are untouched (the reference rounds those). The 1e-9 export bar is unchanged; a test removes the mode and checks that the exporter refuses T5 again |
| `torch.finfo(float64).min` as a mask value overflows float32: `torch.full` raised, and an infinity would give NaN where Mistral multiplies the mask by 0 | gpt2, llama, mistral, qwen2 | Constants, `full` values and weights beyond the working dtype's range become its largest finite number of the same sign, as in the float32 model |

Then the compiled fallbacks, which pass but run on the CPU in float64: `sort topk gather
scatter_add index_select index_copy index` now run through torch on the device (index library
steps), `cos sin log abs reciprocal cast pow` got kernel rules, and integer index arithmetic runs on
the device (docs/COMPILER.md). Left: `triu` (the causal mask of Llama and Qwen2, built once per
call), casts to an integer dtype (BERT's attention mask, T5's relative position buckets). Both are
gone since the op coverage work (`triu` is a kernel, integer casts run on the device), and the
models added with it compile with no fallback too (section below).

## Tests

`tests/test_hf_models.py`: per model, the import against PyTorch float64 at two batch sizes (1e-10),
the export written and run, and the compiled CUDA program at two batch sizes (verified on the first
call); `truestar check` on Llama and BERT; the float32 islands (present without the mode, gone with
it); the wrapper inlining and the refusal of an enabled autocast; the in place mask; the float64
mask times zero staying zero in float32; and the exporter refusing T5 without the exact cast mode.

Manual break on 2026-11-24, restored: `exact_float32_casts` returning float32 for `.float()` failed
6 tests (four imports, the T5 export and the islands test).

Added with the op coverage work: the eight new models in every test above, the compiled CUDA test
now asserts no fallback, and `test_one_training_step_gradient_matches_torch_autograd` (every
trainable weight, tied weights once, against torch.autograd in float64, bar 1e-10). 67 tests.

## Limits

* Small random models only. No pretrained checkpoint was run (no downloads), so the coverage says
  the graphs import, compile and export, not that a released model's numbers were seen.
* Models whose `forward` needs a cache object, `generate`, or `output_attentions` are not wrapped;
  the decode path is covered by docs/LLM.md (paged and dense caches) with a model built from the
  same weights as `LlamaForCausalLM`.
* Encoder decoder T5 (the decoder with cross attention) was not tried, only the encoder.
* Float32 rotary angles at long positions: the angle `position * frequency` is rounded in float32
  before `cos`, so the compiled float32 program differs from the float64 reference by that rounding
  (about 1e-3 at position 32768); the end to end check reports it as growth across regions. The
  float32 PyTorch model rounds the same way.

## ATen op coverage (`scripts/op_coverage.py`, 2026-11-24)

The question this answers: which ATen ops do real models produce, and for each one, what does
Truestar have. Code: `scripts/op_coverage.py` (measurement), `src/truestar/testing_ops.py` (the
torch.nn layer list), tests `tests/test_op_coverage.py`, `tests/test_compiler_coverage.py`,
`tests/test_nn_more.py`.

```
PYTHONPATH=src python scripts/op_coverage.py --compile cuda --train --json results/op_coverage/after.json
PYTHONPATH=src python scripts/op_coverage.py --from-json results/op_coverage/after.json --write-doc
PYTHONPATH=src python scripts/op_coverage.py --sources nn --only LSTM,Upsample_bicubic --train
```

**The model list.** The torchvision classifier zoo (80 models, the 63 under 100M parameters at a
64 pixel input, 224 for ViT, Swin and MaxViT, 299 for Inception v3), the 15 Hugging Face models
above, and 92 torch.nn modules: one per layer (convolutions 1D to 3D, grouped, depthwise and
transposed; group, instance, batch, layer, RMS and local response norms; nearest, linear,
bilinear and bicubic upsampling; pixel and channel shuffles; 1D and 3D pools; reflect, replicate,
circular and zero padding; embeddings and embedding bags; RNN, LSTM, GRU and their cells; multi head
attention, a transformer decoder layer; bilinear, unfold, fold, cosine similarity, pairwise distance;
23 activations) and the common losses (MSE, L1, smooth L1, Huber, cross entropy with and without
label smoothing, NLL, BCE, BCE with logits, KL divergence). Random weights, nothing downloaded.

**What is measured, per model.** The default `torch.export` graph (ops inside control flow regions
too); the import with `ir.import_exported`; the imported float64 program against the PyTorch module
in float64; `--compile`: the compiled float32 program on CUDA (first call verified against the
reference) and the ops that took the float64 CPU fallback; `--train`: the gradient of a random
projection of output 0 with respect to every trainable weight (every float input for a module
without weights) against torch.autograd in float64, bar 1e-8.

**What is reported, per ATen op.** The number of models and uses, whether the importer takes it, the
IR ops its import produced, and for those IR ops: a float64 reference, a reverse rule (or a
lowering, or no gradient needed), a forward rule, a batching rule, and how the compiler runs it:
`kernel` (a rule of the loop IR, one rule set for the Triton CUDA and the numba CPU backends),
`kernel (lowered)` (rewritten into ops with kernels before lowering), `layout` and `create` (no
arithmetic), `library` (torch on the device: cuDNN convolution only when asked), `torch on device`
(sorting and gathers), `integer on device` (integer results such as casts and argmax), `fallback`
(the float64 reference on the CPU).

**Before and after.** Before: main at 7aca8a9 (the same script run on a copy of that commit, with
the same model list; `results/op_coverage/before.json`). After: this branch
(`results/op_coverage/after.json`). RTX 5070 Ti, CUDA 12.8, Windows, torch 2.11, transformers 4.48.

| | before (7aca8a9) | after |
|---|---|---|
| models exported (63 torchvision, 15 Hugging Face, 92 torch.nn) | 170 | 170 |
| models that import | 103 | 170 |
| torchvision: import / compile with no fallback / one training step equal to torch | 63 / 57 / 50 | 63 / 63 / 63 |
| Hugging Face: import / compile with no fallback / training step | 13 / 7 / 3 | 15 / 15 / 15 |
| torch.nn layers and losses: import / compile with no fallback / training step | 27 / 22 / 17 | 92 / 92 / 91 (Hardsigmoid: torch's float32 1/6) |
| imported program against PyTorch float64 (bar 1e-10) | 102 of 103 (CLIP text: float64 import failed) | 170 of 170, worst 4.1e-15 |
| distinct ATen ops in the list | 161 | 161 |
| ATen ops the importer takes | 108 | 161 |
| ATen ops whose IR ops all have a float64 reference | 108 | 161 |
| ... a reverse rule (or a lowering, or no gradient needed) | 88 | 161 |
| ... a forward rule | 103 | 161 |
| ... a batching rule | 108 | 161 |
| ... only native compiled steps (kernels, lowered to kernels, layout, creation, or no IR op at all) | 96 | 151 |
| ... part of the work as a torch step on the device (gathers, integer results) | 2 | 10 |
| ATen ops whose IR ops can take the float64 CPU fallback | 10 | 0 |
| compiled float64 CPU fallbacks measured, ops (models) | pad (8), slice_scatter (7), cast (3), vector_norm (3), group_norm (3), triu (2), argmax (1), rms_norm (1) | none |

The 10 ATen ops that are not native only run part of their work as torch steps on the device, never
on the CPU: the gathers of embeddings and embedding bags, advanced indexing, the class pick of the
cross entropy and NLL losses, nearest upsampling, the index padding modes and the adaptive max pool's
taps (`index`, `index_select`, `gather`: the loop IR has no data dependent load), and `argmax` (an
integer result).

Since the gather kernels (2026-11-24, `compiler/gather_native.py`, docs/COMPILER.md) these reads,
`argmax`, `argmin` and `scatter_add` run as Truestar's own Triton (CUDA) and numba (CPU) kernels:
rebuilt from the saved run (`--from-json results/op_coverage/after.json`), 161 of the 161 ops compile
to native steps only. What still takes the torch step at run time, named in the step's report entry:
integer `argmax`, half precision on the CPU, Metal, mask index tensors, and a `scatter_add` whose
index is over 64 times longer than its column count (token counts over a vocabulary, measured slower
natively). The integer arithmetic of `embedding_bag` offsets stays an exact integer step.
Every model of the list also compiles on the CPU (numba) with no fallback for the Hugging Face and
torch.nn models (107 of 107, `results/op_coverage/after_cpu_hfnn.json`); the torchvision zoo was
compiled on CUDA only here.

Gaps closed, by frequency in the list (the before run's refusals and missing rules):

* Import: `conv1d` (Whisper, Mamba and two layers), `embedding_bag`, `broadcast_tensors` (every
  pointwise loss), `log_softmax`, `softplus` (Mamba), transposed convolutions, 3D pools, interpolation,
  reflect, replicate and circular padding, the recurrent layers and cells, 13 activations and 9 losses:
  53 ATen ops, all lowered onto existing IR ops (docs/IR.md).
* Reverse rules: `index` (18 models: every embedding), `pad` (8), `pow` (6), `avg_pool2d` (6),
  `cast` (5), `slice_scatter` (3), `vector_norm` (3), `group_norm` (3), `rms_norm`, `abs`, and the
  rest of the float ops (docs/AUTODIFF.md).
* Compiled fallbacks: `pad` (8 models), `slice_scatter` (7), integer casts (3), `vector_norm` (3),
  `group_norm` (3), `triu` (2), `argmax`, `rms_norm` (docs/COMPILER.md).
* Wrong gradients found by the sweep and fixed: BERT's word embedding gradient in the padding row
  (`padding_idx` was dropped, 0.36 relative), mobilenet_v2's gradient at its ReLU6 bounds (hardtanh
  imported as clamp, whose gradient passes the bounds; torch's hardtanh sends none; 260 thousand
  exact zeros reach a ReLU6 at the default init, 0.35 relative). CLIP text did not import in float64
  (creation ops took the default dtype at import time).
* Found on the torch side, not fixed: torch's float64 CPU backward of `hardsigmoid` multiplies by the
  float32 constant 1/6 (0.1666666716337204), 3e-8 relative; it is the one remaining FAIL in the train
  column (Truestar's slope is exactly 1/6, tests/test_op_coverage.py checks it against the exact value).

Not measured in that run: the CPU compile of the torchvision zoo (`--compile cpu` ran on the Hugging Face and
torch.nn models: 107 of 107 with no fallback); the core ATen graph (`decompose=True`) of the new
models; pretrained weights; models over 100M parameters.

**The generated tables below are the coverage refresh run** (transformers 5.24.0, CPU, section
"Current models" above), not the 4.48 run of the table above: 63 torchvision models (now also
compiled on the CPU: 63 of 63 with no fallback, 63 of 63 training steps equal to torch), 54 Hugging
Face configs (15 of the first sweep, 28 current families, 11 refused by name), 92 torch.nn layers and
losses, 20 current ops and patterns. "Changed since the earlier run" compares with e96cd2c on the
same list and the same transformers. Saved runs: `results/op_coverage_2026/` (`before_all.json` and
`after_all.json`, the per source runs, the venv's `pip freeze`). The torchvision after run was made at
021e009 (this branch before its test commits; no importer change after it touches those models),
the rest at df36a59. An integer use of an op that runs exactly on the device (positions, packed
sequence ids) is listed as `name[int]` and counted as "integer on device"; the 4.48 run above
counted those as kernels or fallbacks. Refused models are left out of the per op table (their ops
are not claimed either way). The 7 ATen ops that did not import all came from SigLIP2 vision, which
imports since 2026-11-25 (see "Former gaps" above; these generated tables were not rerun).

```
PYTHONPATH=src python scripts/op_coverage.py --from-json results/op_coverage_2026/after_tv.json,results/op_coverage_2026/after_hfnncur.json --write-doc --before results/op_coverage_2026/before_all.json
```

<!-- op_coverage:begin -->

228 models exported, 216 import, 11 refused by name (TS940 to TS942). 197 distinct ATen ops: 190 import; of all ops, 190 have a float64 reference for every IR op they produce, 190 a reverse rule, 190 a forward rule, 190 a batching rule, 167 compile to native steps only (kernels, layout or creation, or nothing at all), 23 use a torch step on the device for part of their work, 0 can take the float64 CPU fallback.

### Changed since the earlier run

| ATen op | models | before: import, reverse, forward, compiled as | after |
|---|---|---|---|
| `aten.__and__.Tensor` | 28 | yes, n/a, n/a, n/a | yes, yes, yes, kernel |
| `aten.new_ones.default` | 28 | yes, n/a, n/a, n/a | yes, yes, yes, create |
| `aten.diff.default` | 28 | NO, no import, no import, no import | yes, yes, yes, integer on device, kernel, layout |
| `aten.eq.Tensor` | 27 | yes, n/a, n/a, n/a | yes, yes, yes, kernel |
| `aten.cumsum.default` | 26 | yes, n/a, n/a, n/a | yes, yes, yes, integer on device |
| `aten.cos.default` | 25 | yes, n/a, n/a, n/a | yes, yes, yes, kernel |
| `aten.sin.default` | 25 | yes, n/a, n/a, n/a | yes, yes, yes, kernel |
| `aten.gt.Tensor` | 8 | yes, n/a, n/a, n/a | yes, yes, yes, kernel |
| `aten.conv2d.padding` | 5 | NO, no import, no import, no import | yes, yes, yes, kernel |
| `aten.repeat.default` | 4 | yes, n/a, n/a, n/a | yes, yes, yes, layout |
| `aten.where.ScalarOther` | 4 | yes, yes, yes, kernel | yes, yes, yes, integer on device, kernel |
| `aten.sum.dim_IntList` | 4 | yes, n/a, n/a, n/a | yes, yes, yes, integer on device, kernel |
| `aten.where.self` | 4 | yes, yes, yes, integer on device | yes, yes, yes, integer on device, kernel |
| `aten.repeat_interleave.self_int` | 3 | yes, n/a, n/a, n/a | yes, yes, yes, layout |
| `aten.detach_.default` | 3 | NO, no import, no import, no import | yes, n/a, n/a, n/a |
| `aten._is_all_true.default` | 3 | NO, no import, no import, no import | yes, yes, yes, kernel |
| `aten.square.default` | 3 | NO, no import, no import, no import | yes, yes, yes, kernel |
| `aten.bucketize.Tensor` | 3 | NO, no import, no import, no import | yes, yes, yes, integer on device, kernel, layout |
| `aten.meshgrid.indexing` | 3 | NO, no import, no import, no import | yes, yes, yes, layout |
| `aten.std.dim` | 3 | NO, no import, no import, no import | yes, yes, yes, kernel |
| `aten.copy_.default` | 2 | yes, n/a, n/a, n/a | yes, yes, yes, kernel |
| `aten.mul_.Tensor` | 2 | yes, n/a, n/a, n/a | yes, yes, yes, kernel |
| `aten.multiply.Tensor` | 2 | NO, no import, no import, no import | yes, yes, yes, kernel |
| `aten.bmm.default` | 2 | yes, n/a, n/a, n/a | yes, yes, yes, kernel |
| `aten.floor.default` | 2 | NO, no import, no import, no import | yes, yes, yes, kernel |
| `aten.min.other` | 2 | yes, yes, yes, integer on device | yes, yes, yes, integer on device, kernel |
| `aten.reciprocal.default` | 2 | yes, n/a, n/a, n/a | yes, yes, yes, kernel |
| `aten.erfinv.default` | 2 | NO, no import, no import, no import | yes, n/a, n/a, n/a |
| `aten.full.default` | 2 | yes, n/a, n/a, n/a | yes, yes, yes, create |
| `aten.ones.default` | 2 | yes, n/a, n/a, n/a | yes, yes, yes, create |
| `aten.tile.default` | 2 | NO, no import, no import, no import | yes, yes, yes, layout |
| `aten.addmm.default` | 1 | yes, n/a, n/a, n/a | yes, yes, yes, kernel |
| `aten.arange.start` | 1 | yes, n/a, n/a, n/a | yes, yes, yes, create |
| `aten.expm1.default` | 1 | yes, n/a, n/a, n/a | yes, yes, yes, kernel |
| `aten.split.Tensor` | 1 | yes, n/a, n/a, n/a | yes, yes, yes, layout |
| `aten.sqrt.default` | 1 | yes, n/a, n/a, n/a | yes, yes, yes, kernel |
| `aten.bitwise_not.default` | 1 | NO, no import, no import, no import | yes, yes, yes, kernel |
| `aten.logical_and.default` | 1 | yes, n/a, n/a, n/a | yes, yes, yes, kernel |
| `aten.var.correction` | 1 | NO, no import, no import, no import | yes, yes, yes, kernel |

### Per ATen op

| ATen op | models | uses | import | IR ops | f64 ref | reverse | forward | batching | compiled as |
|---|---|---|---|---|---|---|---|---|---|
| `aten.linear.default` | 106 | 1725 | yes | linear | yes | yes | yes | yes | kernel |
| `aten.mul.Tensor` | 79 | 1684 | yes | mul mul[int] | yes | yes | yes | yes | integer on device, kernel |
| `aten.add.Tensor` | 76 | 1560 | yes | add add[int] | yes | yes | yes | yes | integer on device, kernel |
| `aten.conv2d.default` | 74 | 5068 | yes | conv2d | yes | yes | yes | yes | kernel |
| `aten.flatten.using_ints` | 65 | 74 | yes | reshape | yes | yes | yes | yes | layout |
| `aten.view.default` | 55 | 1332 | yes | reshape | yes | yes | yes | yes | layout |
| `aten.transpose.int` | 55 | 903 | yes | permute | yes | yes | yes | yes | layout |
| `aten.adaptive_avg_pool2d.default` | 53 | 527 | yes | adaptive_avg_pool2d | yes | yes | yes | yes | kernel |
| `aten.batch_norm.default` | 52 | 3935 | yes | batch_norm | yes | yes | yes | yes | kernel |
| `aten.reshape.default` | 52 | 1013 | yes | reshape | yes | yes | yes | yes | layout |
| `aten.cat.default` | 51 | 589 | yes | cat | yes | yes | yes | yes | layout |
| `aten.unsqueeze.default` | 50 | 1207 | yes | reshape | yes | yes | yes | yes | layout |
| `aten.slice.Tensor` | 44 | 1565 | yes | slice | yes | yes | yes | yes | layout |
| `aten.relu_.default` | 43 | 2603 | yes | relu | yes | yes | yes | yes | kernel |
| `aten.mean.dim` | 41 | 251 | yes | mean | yes | yes | yes | yes | kernel |
| `aten.scaled_dot_product_attention.default` | 41 | 109 | yes | attention | yes | yes | yes | yes | kernel |
| `aten.expand.default` | 40 | 219 | yes | expand | yes | yes | yes | yes | layout |
| `aten._assert_tensor_metadata.default` | 39 | 749 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.sub.Tensor` | 38 | 117 | yes | sub sub[int] | yes | yes | yes | yes | integer on device, kernel |
| `aten.index.Tensor` | 37 | 283 | yes | index | yes | yes | yes | yes | torch on device |
| `aten.embedding.default` | 37 | 45 | yes | index reshape frozen ne where full ... | yes | yes | yes | yes | create, kernel, layout, torch on device |
| `aten.to.dtype` | 36 | 621 | yes | cast | yes | yes | yes | yes | kernel |
| `aten.arange.default` | 34 | 201 | yes | arange | yes | yes | yes | yes | create |
| `aten.ne.Scalar` | 34 | 92 | yes | ne | yes | yes | yes | yes | kernel |
| `aten.alias.default` | 32 | 100 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.layer_norm.default` | 29 | 581 | yes | layer_norm | yes | yes | yes | yes | kernel |
| `aten.pow.Tensor_Scalar` | 29 | 291 | yes | pow | yes | yes | yes | yes | kernel |
| `aten.gelu.default` | 28 | 326 | yes | gelu | yes | yes | yes | yes | kernel |
| `aten.to.dtype_layout` | 28 | 91 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.__and__.Tensor` | 28 | 88 | yes | logical_and | yes | yes | yes | yes | kernel |
| `aten.new_ones.default` | 28 | 44 | yes | full | yes | yes | yes | yes | create |
| `aten.diff.default` | 28 | 35 | yes | slice cat sub[int] sub | yes | yes | yes | yes | integer on device, kernel, layout |
| `aten.dropout.default` | 27 | 699 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.add_.Tensor` | 27 | 628 | yes | add slice_scatter add[int] | yes | yes | yes | yes | integer on device, kernel |
| `aten.neg.default` | 27 | 97 | yes | neg | yes | yes | yes | yes | kernel |
| `aten.to.device` | 27 | 37 | yes | cast | yes | yes | yes | yes | kernel |
| `aten.eq.Tensor` | 27 | 35 | yes | eq | yes | yes | yes | yes | kernel |
| `aten.le.Tensor` | 27 | 34 | yes | le | yes | yes | yes | yes | kernel |
| `aten.contiguous.default` | 26 | 349 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.cumsum.default` | 26 | 33 | yes | cumsum[int] | yes | yes | yes | yes | integer on device |
| `aten.sigmoid.default` | 25 | 525 | yes | sigmoid | yes | yes | yes | yes | kernel |
| `aten.rsqrt.default` | 25 | 168 | yes | rsqrt | yes | yes | yes | yes | kernel |
| `aten.cos.default` | 25 | 29 | yes | cos | yes | yes | yes | yes | kernel |
| `aten.sin.default` | 25 | 29 | yes | sin | yes | yes | yes | yes | kernel |
| `aten.permute.default` | 24 | 989 | yes | permute | yes | yes | yes | yes | layout |
| `aten.silu.default` | 22 | 86 | yes | silu | yes | yes | yes | yes | kernel |
| `aten.max_pool2d.default` | 22 | 43 | yes | max_pool2d | yes | yes | yes | yes | kernel |
| `aten.select.int` | 21 | 532 | yes | select | yes | yes | yes | yes | layout |
| `aten.div.Tensor` | 19 | 155 | yes | div | yes | yes | yes | yes | kernel |
| `aten.dropout_.default` | 16 | 16 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.softmax.int` | 15 | 159 | yes | softmax | yes | yes | yes | yes | kernel |
| `aten.relu.default` | 14 | 139 | yes | relu | yes | yes | yes | yes | kernel |
| `aten.pad.default` | 13 | 148 | yes | pad index_select | yes | yes | yes | yes | kernel, torch on device |
| `aten.matmul.default` | 12 | 282 | yes | matmul | yes | yes | yes | yes | kernel |
| `aten.silu_.default` | 10 | 1023 | yes | silu | yes | yes | yes | yes | kernel |
| `aten.lift_fresh_copy.default` | 9 | 493 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.chunk.default` | 9 | 81 | yes | slice | yes | yes | yes | yes | layout |
| `aten.squeeze.dim` | 9 | 33 | yes | reshape | yes | yes | yes | yes | layout |
| `aten.tanh.default` | 9 | 18 | yes | tanh | yes | yes | yes | yes | kernel |
| `aten.eq.Scalar` | 8 | 56 | yes | eq | yes | yes | yes | yes | kernel |
| `aten.ge.Scalar` | 8 | 12 | yes | ge | yes | yes | yes | yes | kernel |
| `aten.zeros.default` | 8 | 12 | yes | full | yes | yes | yes | yes | create |
| `aten.flip.default` | 8 | 8 | yes | flip | yes | yes | yes | yes | kernel |
| `aten.gt.Tensor` | 8 | 8 | yes | gt | yes | yes | yes | yes | kernel |
| `aten.masked_fill.Scalar` | 7 | 110 | yes | where | yes | yes | yes | yes | kernel |
| `aten.roll.default` | 7 | 110 | yes | slice cat | yes | yes | yes | yes | layout |
| `aten.stack.default` | 7 | 16 | yes | reshape cat | yes | yes | yes | yes | layout |
| `aten.gt.Scalar` | 7 | 8 | yes | gt | yes | yes | yes | yes | kernel |
| `aten.fill_.Tensor` | 6 | 486 | yes | slice_scatter expand | yes | yes | yes | yes | kernel, layout |
| `aten.clamp.default` | 6 | 65 | yes | clamp | yes | yes | yes | yes | kernel |
| `aten.new_zeros.default` | 6 | 54 | yes | full | yes | yes | yes | yes | create |
| `aten.unflatten.int` | 6 | 29 | yes | reshape | yes | yes | yes | yes | layout |
| `aten.avg_pool2d.default` | 6 | 25 | yes | avg_pool2d | yes | yes | yes | yes | kernel |
| `aten.broadcast_tensors.default` | 6 | 6 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.type_as.default` | 5 | 98 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.split_with_sizes.default` | 5 | 12 | yes | slice | yes | yes | yes | yes | layout |
| `aten.conv1d.default` | 5 | 7 | yes | reshape conv2d | yes | yes | yes | yes | kernel, layout |
| `aten.conv2d.padding` | 5 | 5 | yes | conv2d pad | yes | yes | yes | yes | kernel |
| `aten.linalg_vector_norm.default` | 4 | 122 | yes | vector_norm | yes | yes | yes | yes | kernel (lowered) |
| `aten.exp.default` | 4 | 64 | yes | exp | yes | yes | yes | yes | kernel |
| `aten.clone.default` | 4 | 63 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.repeat.default` | 4 | 7 | yes | reshape expand | yes | yes | yes | yes | layout |
| `aten.where.ScalarOther` | 4 | 6 | yes | where where[int] | yes | yes | yes | yes | integer on device, kernel |
| `aten.sum.dim_IntList` | 4 | 5 | yes | cast sum[int] sum | yes | yes | yes | yes | integer on device, kernel |
| `aten.where.self` | 4 | 5 | yes | where where[int] | yes | yes | yes | yes | integer on device, kernel |
| `aten.arange.start_step` | 4 | 4 | yes | arange | yes | yes | yes | yes | create |
| `aten.embedding_bag.padding_idx` | 4 | 4 | yes | reshape full arange scatter_add le cast ... | yes | yes | yes | yes | create, integer on device, kernel, layout, torch on device |
| `aten.clamp_min.default` | 3 | 120 | yes | clamp | yes | yes | yes | yes | kernel |
| `aten.expand_as.default` | 3 | 120 | yes | expand | yes | yes | yes | yes | layout |
| `aten.zero_.default` | 3 | 60 | yes | full_like slice_scatter | yes | yes | yes | yes | create, kernel |
| `aten.hardsigmoid.default` | 3 | 18 | yes | hardsigmoid | yes | yes | yes | yes | kernel |
| `aten.repeat_interleave.self_int` | 3 | 9 | yes | reshape expand | yes | yes | yes | yes | layout |
| `aten.detach_.default` | 3 | 7 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.softplus.default` | 3 | 7 | yes | add mul where relu abs neg ... | yes | yes | yes | yes | create, kernel |
| `aten._is_all_true.default` | 3 | 6 | yes | logical_not any | yes | yes | yes | yes | kernel |
| `aten.square.default` | 3 | 5 | yes | mul | yes | yes | yes | yes | kernel |
| `aten.bucketize.Tensor` | 3 | 4 | yes | reshape cast sum[int] ge gt | yes | yes | yes | yes | integer on device, kernel, layout |
| `aten.log_softmax.int` | 3 | 3 | yes | log_softmax | yes | yes | yes | yes | kernel |
| `aten.meshgrid.indexing` | 3 | 3 | yes | reshape expand | yes | yes | yes | yes | layout |
| `aten.std.dim` | 3 | 3 | yes | mean sub mul sum div sqrt | yes | yes | yes | yes | kernel |
| `aten.hardswish_.default` | 2 | 40 | yes | hardswish | yes | yes | yes | yes | kernel |
| `aten.copy_.default` | 2 | 4 | yes | slice_scatter | yes | yes | yes | yes | kernel |
| `aten.mul_.Tensor` | 2 | 4 | yes | mul | yes | yes | yes | yes | kernel |
| `aten.multiply.Tensor` | 2 | 4 | yes | mul | yes | yes | yes | yes | kernel |
| `aten.bmm.default` | 2 | 3 | yes | matmul | yes | yes | yes | yes | kernel |
| `aten.floor.default` | 2 | 3 | yes | cast gt sub abs lt ne ... | yes | yes | yes | yes | kernel |
| `aten.log.default` | 2 | 3 | yes | log | yes | yes | yes | yes | kernel |
| `aten.min.other` | 2 | 3 | yes | minimum minimum[int] | yes | yes | yes | yes | integer on device, kernel |
| `aten.reciprocal.default` | 2 | 3 | yes | reciprocal | yes | yes | yes | yes | kernel |
| `aten.abs.default` | 2 | 2 | yes | abs[int] abs | yes | yes | yes | yes | integer on device, kernel |
| `aten.avg_pool3d.default` | 2 | 2 | yes | reshape avg_pool2d | yes | yes | yes | yes | kernel, layout |
| `aten.conv_transpose2d.input` | 2 | 2 | yes | conv2d_input_grad reshape add | yes | yes | yes | yes | kernel, layout |
| `aten.cross_entropy_loss.default` | 2 | 2 | yes | full_like sum reshape mul where neg ... | yes | yes | yes | yes | create, integer on device, kernel, layout, torch on device |
| `aten.erfinv.default` | 2 | 2 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.full.default` | 2 | 2 | yes | full | yes | yes | yes | yes | create |
| `aten.instance_norm.default` | 2 | 2 | yes | group_norm | yes | yes | yes | yes | kernel (lowered) |
| `aten.lstm.input` | 2 | 2 | yes | slice sigmoid mul linear add tanh ... | yes | yes | yes | yes | kernel, layout |
| `aten.lt.Scalar` | 2 | 2 | yes | lt | yes | yes | yes | yes | kernel |
| `aten.ones.default` | 2 | 2 | yes | full | yes | yes | yes | yes | create |
| `aten.tile.default` | 2 | 2 | yes | reshape expand | yes | yes | yes | yes | layout |
| `aten.upsample_bilinear2d.vec` | 2 | 2 | yes | permute matmul | yes | yes | yes | yes | kernel, layout |
| `aten.einsum.default` | 1 | 44 | yes | matmul permute | yes | yes | yes | yes | kernel, layout |
| `aten.hardtanh_.default` | 1 | 35 | yes | clamp | yes | yes | yes | yes | kernel |
| `aten.swapaxes.default` | 1 | 22 | yes | permute | yes | yes | yes | yes | layout |
| `aten.addmm.default` | 1 | 8 | yes | matmul add | yes | yes | yes | yes | kernel |
| `aten._assert_scalar.default` | 1 | 2 | NO |  | no import | no import | no import | no import | no import |
| `aten.all.default` | 1 | 2 | NO |  | no import | no import | no import | no import | no import |
| `aten.arange.start` | 1 | 2 | yes | arange | yes | yes | yes | yes | create |
| `aten.expm1.default` | 1 | 2 | yes | mul add clamp exp sub ge ... | yes | yes | yes | yes | kernel |
| `aten.item.default` | 1 | 2 | NO |  | no import | no import | no import | no import | no import |
| `aten.prod.dim_int` | 1 | 2 | NO |  | no import | no import | no import | no import | no import |
| `aten.split.Tensor` | 1 | 2 | yes | slice | yes | yes | yes | yes | layout |
| `aten.sqrt.default` | 1 | 2 | yes | sqrt | yes | yes | yes | yes | kernel |
| `cast_symbool_to_symint_guardless` | 1 | 2 | NO |  | no import | no import | no import | no import | no import |
| `aten.abs_.default` | 1 | 1 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.adaptive_avg_pool1d.default` | 1 | 1 | yes | reshape adaptive_avg_pool2d | yes | yes | yes | yes | kernel, layout |
| `aten.adaptive_avg_pool3d.default` | 1 | 1 | yes | reshape adaptive_avg_pool2d | yes | yes | yes | yes | kernel, layout |
| `aten.adaptive_max_pool1d.default` | 1 | 1 | yes | reshape adaptive_max_pool2d | yes | yes | yes | yes | layout, lowered: torch on device |
| `aten.argmax.default` | 1 | 1 | yes | argmax | yes | yes | yes | yes | integer on device |
| `aten.avg_pool1d.default` | 1 | 1 | yes | reshape avg_pool2d | yes | yes | yes | yes | kernel, layout |
| `aten.baddbmm.default` | 1 | 1 | NO |  | no import | no import | no import | no import | no import |
| `aten.bilinear.default` | 1 | 1 | yes | reshape permute matmul mul sum add | yes | yes | yes | yes | kernel, layout |
| `aten.binary_cross_entropy.default` | 1 | 1 | yes | mul add clamp log neg where ... | yes | yes | yes | yes | create, kernel |
| `aten.binary_cross_entropy_with_logits.default` | 1 | 1 | yes | mul add neg where relu abs ... | yes | yes | yes | yes | create, kernel |
| `aten.bitwise_not.default` | 1 | 1 | yes | logical_not | yes | yes | yes | yes | kernel |
| `aten.celu.default` | 1 | 1 | yes | mul add where clamp exp sub ... | yes | yes | yes | yes | kernel |
| `aten.channel_shuffle.default` | 1 | 1 | yes | reshape permute | yes | yes | yes | yes | layout |
| `aten.clamp_.default` | 1 | 1 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.col2im.default` | 1 | 1 | yes | reshape conv2d_input_grad | yes | yes | yes | yes | kernel, layout |
| `aten.conv3d.default` | 1 | 1 | yes | reshape permute slice select conv2d add ... | yes | yes | yes | yes | kernel, layout |
| `aten.conv_transpose1d.default` | 1 | 1 | yes | reshape conv2d_input_grad add | yes | yes | yes | yes | kernel, layout |
| `aten.conv_transpose3d.input` | 1 | 1 | yes | reshape full cat slice permute select ... | yes | yes | yes | yes | create, kernel, layout |
| `aten.cosine_similarity.default` | 1 | 1 | yes | mul sum sqrt clamp div | yes | yes | yes | yes | kernel |
| `aten.div.Tensor_mode` | 1 | 1 | NO |  | no import | no import | no import | no import | no import |
| `aten.div_.Tensor` | 1 | 1 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.elu.default` | 1 | 1 | yes | mul add where clamp exp sub ... | yes | yes | yes | yes | kernel |
| `aten.full_like.default` | 1 | 1 | yes | full_like | yes | yes | yes | yes | create |
| `aten.gather.default` | 1 | 1 | yes | gather | yes | yes | yes | yes | torch on device |
| `aten.glu.default` | 1 | 1 | yes | slice sigmoid mul | yes | yes | yes | yes | kernel, layout |
| `aten.group_norm.default` | 1 | 1 | yes | group_norm | yes | yes | yes | yes | kernel (lowered) |
| `aten.gru.input` | 1 | 1 | yes | slice add linear sigmoid mul select ... | yes | yes | yes | yes | kernel, layout |
| `aten.gru_cell.default` | 1 | 1 | yes | slice add linear sigmoid mul tanh ... | yes | yes | yes | yes | kernel, layout |
| `aten.hardshrink.default` | 1 | 1 | yes | abs gt full_like where | yes | yes | yes | yes | create, kernel |
| `aten.hardswish.default` | 1 | 1 | yes | hardswish | yes | yes | yes | yes | kernel |
| `aten.hardtanh.default` | 1 | 1 | yes | clamp | yes | yes | yes | yes | kernel |
| `aten.huber_loss.default` | 1 | 1 | yes | mul sub abs lt where mean | yes | yes | yes | yes | kernel |
| `aten.im2col.default` | 1 | 1 | yes | slice reshape pad cat | yes | yes | yes | yes | kernel, layout |
| `aten.kl_div.default` | 1 | 1 | yes | full_like where mul gt log sub ... | yes | yes | yes | yes | create, kernel |
| `aten.l1_loss.default` | 1 | 1 | yes | sub abs mean | yes | yes | yes | yes | kernel |
| `aten.le.Scalar` | 1 | 1 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.leaky_relu.default` | 1 | 1 | yes | gt mul where | yes | yes | yes | yes | kernel |
| `aten.log_sigmoid.default` | 1 | 1 | yes | add mul neg where relu abs ... | yes | yes | yes | yes | create, kernel |
| `aten.logical_and.default` | 1 | 1 | yes | logical_and | yes | yes | yes | yes | kernel |
| `aten.lstm_cell.default` | 1 | 1 | yes | slice sigmoid mul linear add tanh | yes | yes | yes | yes | kernel, layout |
| `aten.lt.Tensor` | 1 | 1 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.max_pool1d.default` | 1 | 1 | yes | reshape max_pool2d | yes | yes | yes | yes | kernel, layout |
| `aten.max_pool3d.default` | 1 | 1 | yes | reshape max_pool2d | yes | yes | yes | yes | kernel, layout |
| `aten.mish.default` | 1 | 1 | yes | mul add where relu abs neg ... | yes | yes | yes | yes | create, kernel |
| `aten.mse_loss.default` | 1 | 1 | yes | sub mul mean | yes | yes | yes | yes | kernel |
| `aten.nll_loss_nd.default` | 1 | 1 | yes | full_like reshape sum ne where[int] gather ... | yes | yes | yes | yes | create, integer on device, kernel, layout, torch on device |
| `aten.pairwise_distance.default` | 1 | 1 | yes | sub add vector_norm | yes | yes | yes | yes | kernel, kernel (lowered) |
| `aten.pixel_shuffle.default` | 1 | 1 | yes | reshape permute | yes | yes | yes | yes | layout |
| `aten.pixel_unshuffle.default` | 1 | 1 | yes | reshape permute | yes | yes | yes | yes | layout |
| `aten.prelu.default` | 1 | 1 | yes | reshape gt mul where | yes | yes | yes | yes | kernel, layout |
| `aten.rms_norm.default` | 1 | 1 | yes | rms_norm | yes | yes | yes | yes | kernel (lowered) |
| `aten.rnn_tanh.input` | 1 | 1 | yes | linear select reshape add tanh cat | yes | yes | yes | yes | kernel, layout |
| `aten.rsub.Scalar` | 1 | 1 | yes |  | n/a | n/a | n/a | n/a | n/a |
| `aten.selu.default` | 1 | 1 | yes | mul add where clamp exp sub ... | yes | yes | yes | yes | kernel |
| `aten.smooth_l1_loss.default` | 1 | 1 | yes | sub mul abs lt where mean | yes | yes | yes | yes | kernel |
| `aten.softshrink.default` | 1 | 1 | yes | where sub add full_like gt lt | yes | yes | yes | yes | create, kernel |
| `aten.threshold.default` | 1 | 1 | yes | gt full_like where | yes | yes | yes | yes | create, kernel |
| `aten.unbind.int` | 1 | 1 | yes | select | yes | yes | yes | yes | layout |
| `aten.upsample_bicubic2d.vec` | 1 | 1 | yes | permute matmul | yes | yes | yes | yes | kernel, layout |
| `aten.upsample_linear1d.vec` | 1 | 1 | yes | matmul | yes | yes | yes | yes | kernel |
| `aten.upsample_nearest1d.vec` | 1 | 1 | yes | index_select | yes | yes | yes | yes | torch on device |
| `aten.upsample_nearest2d.vec` | 1 | 1 | yes | index_select | yes | yes | yes | yes | torch on device |
| `aten.var.correction` | 1 | 1 | yes | mean sub mul sum div | yes | yes | yes | yes | kernel |

### Per model

| model | ATen ops | import | f64 error | compile fallbacks | train step (grad vs torch) |
|---|---|---|---|---|---|
| tv/alexnet | 7 | PASS | 1.9e-15 | none | PASS 2.1e-16 |
| tv/convnext_base | 9 | PASS | 2.9e-15 | none | PASS 9.0e-16 |
| tv/convnext_large | | skipped (over 100M params) | | | |
| tv/convnext_small | 9 | PASS | 2.2e-15 | none | PASS 9.3e-16 |
| tv/convnext_tiny | 9 | PASS | 2.3e-15 | none | PASS 1.1e-15 |
| tv/densenet121 | 9 | PASS | 1.2e-15 | none | PASS 9.9e-16 |
| tv/densenet161 | 9 | PASS | 1.5e-15 | none | PASS 1.0e-15 |
| tv/densenet169 | 9 | PASS | 1.5e-15 | none | PASS 6.5e-16 |
| tv/densenet201 | 9 | PASS | 1.6e-15 | none | PASS 7.5e-16 |
| tv/efficientnet_b0 | 10 | PASS | 2.2e-15 | none | PASS 1.4e-16 |
| tv/efficientnet_b1 | 10 | PASS | 2.9e-15 | none | PASS 1.4e-16 |
| tv/efficientnet_b2 | 10 | PASS | 3.2e-15 | none | PASS 1.4e-16 |
| tv/efficientnet_b3 | 10 | PASS | 2.8e-15 | none | PASS 1.4e-16 |
| tv/efficientnet_b4 | 10 | PASS | 3.5e-15 | none | PASS 2.1e-16 |
| tv/efficientnet_b5 | 10 | PASS | 3.2e-15 | none | PASS 1.4e-16 |
| tv/efficientnet_b6 | 10 | PASS | 3.3e-15 | none | PASS 1.8e-16 |
| tv/efficientnet_b7 | 10 | PASS | 3.0e-15 | none | PASS 1.8e-16 |
| tv/efficientnet_v2_l | | skipped (over 100M params) | | | |
| tv/efficientnet_v2_m | 10 | PASS | 4.5e-15 | none | PASS 1.4e-16 |
| tv/efficientnet_v2_s | 10 | PASS | 3.1e-15 | none | PASS 1.4e-16 |
| tv/googlenet | 9 | PASS | 0.0e+00 | none | PASS 8.8e-18 |
| tv/inception_v3 | 10 | PASS | 1.3e-15 | none | PASS 2.5e-15 |
| tv/maxvit_t | 24 | PASS | 7.2e-16 | none | PASS 7.5e-16 |
| tv/mnasnet0_5 | 7 | PASS | 1.5e-15 | none | PASS 3.9e-16 |
| tv/mnasnet0_75 | 7 | PASS | 2.0e-15 | none | PASS 4.1e-16 |
| tv/mnasnet1_0 | 7 | PASS | 1.8e-15 | none | PASS 4.3e-16 |
| tv/mnasnet1_3 | 7 | PASS | 1.5e-15 | none | PASS 4.0e-16 |
| tv/mobilenet_v2 | 8 | PASS | 1.6e-15 | none | PASS 1.8e-16 |
| tv/mobilenet_v3_large | 12 | PASS | 2.5e-15 | none | PASS 8.8e-18 |
| tv/mobilenet_v3_small | 12 | PASS | 1.7e-15 | none | PASS 8.8e-18 |
| tv/regnet_x_16gf | 7 | PASS | 1.9e-15 | none | PASS 1.3e-15 |
| tv/regnet_x_1_6gf | 7 | PASS | 1.1e-15 | none | PASS 3.5e-16 |
| tv/regnet_x_32gf | | skipped (over 100M params) | | | |
| tv/regnet_x_3_2gf | 7 | PASS | 1.2e-15 | none | PASS 1.1e-15 |
| tv/regnet_x_400mf | 7 | PASS | 8.3e-16 | none | PASS 4.6e-16 |
| tv/regnet_x_800mf | 7 | PASS | 9.6e-16 | none | PASS 2.5e-16 |
| tv/regnet_x_8gf | 7 | PASS | 1.5e-15 | none | PASS 1.6e-15 |
| tv/regnet_y_128gf | | skipped (over 100M params) | | | |
| tv/regnet_y_16gf | 10 | PASS | 1.8e-15 | none | PASS 3.5e-16 |
| tv/regnet_y_1_6gf | 10 | PASS | 7.9e-16 | none | PASS 2.8e-16 |
| tv/regnet_y_32gf | | skipped (over 100M params) | | | |
| tv/regnet_y_3_2gf | 10 | PASS | 1.3e-15 | none | PASS 3.2e-16 |
| tv/regnet_y_400mf | 10 | PASS | 7.2e-16 | none | PASS 2.1e-16 |
| tv/regnet_y_800mf | 10 | PASS | 7.5e-16 | none | PASS 1.8e-16 |
| tv/regnet_y_8gf | 10 | PASS | 1.1e-15 | none | PASS 3.0e-16 |
| tv/resnet101 | 8 | PASS | 2.7e-15 | none | PASS 2.1e-15 |
| tv/resnet152 | 8 | PASS | 3.3e-15 | none | PASS 3.2e-15 |
| tv/resnet18 | 8 | PASS | 1.7e-15 | none | PASS 1.6e-15 |
| tv/resnet34 | 8 | PASS | 1.9e-15 | none | PASS 1.4e-15 |
| tv/resnet50 | 8 | PASS | 2.3e-15 | none | PASS 2.5e-15 |
| tv/resnext101_32x8d | 8 | PASS | 1.5e-15 | none | PASS 7.0e-16 |
| tv/resnext101_64x4d | 8 | PASS | 1.1e-15 | none | PASS 5.6e-16 |
| tv/resnext50_32x4d | 8 | PASS | 1.4e-15 | none | PASS 6.3e-16 |
| tv/shufflenet_v2_x0_5 | 11 | PASS | 1.1e-16 | none | PASS 7.0e-17 |
| tv/shufflenet_v2_x1_0 | 11 | PASS | 5.6e-17 | none | PASS 1.1e-16 |
| tv/shufflenet_v2_x1_5 | 11 | PASS | 1.1e-16 | none | PASS 7.0e-17 |
| tv/shufflenet_v2_x2_0 | 11 | PASS | 1.6e-16 | none | PASS 7.0e-17 |
| tv/squeezenet1_0 | 7 | PASS | 1.3e-15 | none | PASS 1.1e-15 |
| tv/squeezenet1_1 | 7 | PASS | 7.3e-16 | none | PASS 9.0e-16 |
| tv/swin_b | 31 | PASS | 1.7e-15 | none | PASS 8.2e-16 |
| tv/swin_s | 31 | PASS | 8.5e-16 | none | PASS 7.1e-16 |
| tv/swin_t | 31 | PASS | 1.4e-15 | none | PASS 7.3e-16 |
| tv/swin_v2_b | 41 | PASS | 1.3e-15 | none | PASS 1.5e-15 |
| tv/swin_v2_s | 41 | PASS | 1.4e-15 | none | PASS 2.2e-15 |
| tv/swin_v2_t | 41 | PASS | 1.7e-15 | none | PASS 2.2e-15 |
| tv/vgg11 | | skipped (over 100M params) | | | |
| tv/vgg11_bn | | skipped (over 100M params) | | | |
| tv/vgg13 | | skipped (over 100M params) | | | |
| tv/vgg13_bn | | skipped (over 100M params) | | | |
| tv/vgg16 | | skipped (over 100M params) | | | |
| tv/vgg16_bn | | skipped (over 100M params) | | | |
| tv/vgg19 | | skipped (over 100M params) | | | |
| tv/vgg19_bn | | skipped (over 100M params) | | | |
| tv/vit_b_16 | 18 | PASS | 0.0e+00 | none | PASS 3.1e-15 |
| tv/vit_b_32 | 18 | PASS | 0.0e+00 | none | PASS 5.3e-15 |
| tv/vit_h_14 | | skipped (over 100M params) | | | |
| tv/vit_l_16 | | skipped (over 100M params) | | | |
| tv/vit_l_32 | | skipped (over 100M params) | | | |
| tv/wide_resnet101_2 | | skipped (over 100M params) | | | |
| tv/wide_resnet50_2 | 8 | PASS | 2.4e-15 | none | PASS 3.3e-15 |
| hf/llama | 34 | PASS | 3.0e-16 | none | PASS 4.0e-16 |
| hf/mistral | 35 | PASS | 3.0e-16 | none | PASS 2.9e-16 |
| hf/qwen2 | 34 | PASS | 3.7e-16 | none | PASS 4.3e-16 |
| hf/gpt2 | 30 | PASS | 2.3e-16 | none | PASS 2.8e-16 |
| hf/bert | 16 | PASS | 3.1e-16 | none | PASS 2.4e-16 |
| hf/vit | 16 | PASS | 6.9e-16 | none | PASS 3.4e-16 |
| hf/t5_encoder | 32 | PASS | 4.2e-16 | none | PASS 4.1e-16 |
| hf/whisper | 30 | PASS | 5.6e-16 | none | PASS 5.5e-16 |
| hf/clip_text | 21 | PASS | 6.1e-16 | none | PASS 6.1e-16 |
| hf/clip_vision | 17 | PASS | 6.9e-16 | none | PASS 4.3e-16 |
| hf/deit | 16 | PASS | 8.2e-16 | none | PASS 7.8e-16 |
| hf/swin | 29 | PASS | 4.2e-16 | none | PASS 7.7e-16 |
| hf/convnext | 8 | PASS | 5.9e-16 | none | PASS 8.6e-16 |
| hf/mobilevit | 17 | PASS | 3.1e-15 | none | PASS 6.4e-18 |
| hf/mamba | 25 | PASS | 2.6e-16 | none | PASS 2.8e-16 |
| hf/gemma2 | 43 | PASS | 4.6e-16 | none | PASS 8.4e-16 |
| hf/gemma3 | 38 | PASS | 7.4e-16 | none | PASS 9.5e-16 |
| hf/gemma3n | 62 | PASS | 8.4e-16 | none | PASS 1.3e-15 |
| hf/gemma4 | 39 | PASS | 1.3e-15 | none | PASS 8.7e-15 |
| hf/qwen3 | 34 | PASS | 3.2e-16 | none | PASS 4.9e-16 |
| hf/phi4_mini | 37 | PASS | 2.9e-16 | none | PASS 3.9e-16 |
| hf/smollm3 | 34 | PASS | 3.0e-16 | none | PASS 3.6e-16 |
| hf/olmo2 | 34 | PASS | 5.4e-16 | none | PASS 5.4e-16 |
| hf/olmo3 | 35 | PASS | 5.2e-16 | none | PASS 7.6e-16 |
| hf/granite | 35 | PASS | 4.0e-16 | none | PASS 3.1e-16 |
| hf/cohere2 | 37 | PASS | 3.0e-16 | none | PASS 2.4e-16 |
| hf/glm4 | 39 | PASS | 6.8e-16 | none | PASS 1.0e-15 |
| hf/ernie4_5 | 37 | PASS | 3.0e-16 | none | PASS 4.1e-16 |
| hf/exaone4 | 35 | PASS | 3.8e-16 | none | PASS 5.0e-16 |
| hf/seed_oss | 35 | PASS | 3.1e-16 | none | PASS 5.4e-16 |
| hf/apertus | 38 | PASS | 3.7e-16 | none | PASS 5.5e-16 |
| hf/arcee | 35 | PASS | 3.1e-16 | none | PASS 2.2e-16 |
| hf/ministral3 | 37 | PASS | 3.0e-16 | none | PASS 3.1e-16 |
| hf/hunyuan_dense | 34 | PASS | 3.9e-16 | none | PASS 5.4e-16 |
| hf/lfm2 | 37 | PASS | 2.7e-16 | none | PASS 2.1e-16 |
| hf/bitnet | 35 | PASS | 8.2e-16 | none | PASS 9.8e-16 |
| hf/siglip_vision | 24 | PASS | 6.3e-16 | none | PASS 8.5e-16 |
| hf/siglip2_vision | 59 | NO: `<built-in`, `<function`, `aten._assert_scalar.default`, `aten.all.default`, `aten.baddbmm.default`, `aten.div.Tensor_mode`, `aten.item.default`, `aten.prod.dim_int` |  |  |  |
| hf/dinov3_vit | 29 | PASS | 7.1e-16 | none | PASS 3.5e-16 |
| hf/aimv2_vision | 18 | PASS | 6.8e-16 | none | PASS 4.7e-16 |
| hf/smolvlm_vision | 31 | PASS | 8.7e-16 | none | PASS 3.5e-16 |
| hf/pixtral_vision | | torch.export failed: GuardOnDataDependentSymNode: Could not extract specialized integer from data-dependent expression u0 (unhinted: u0).  (Size-like symbols: none) | | | |
| hf/convnextv2 | 10 | PASS | 7.0e-16 | none | PASS 6.4e-16 |
| hf/gpt_oss | 0 | refused TS940: mixture of experts routing (`GptOssTopKRouter`) |  |  |  |
| hf/qwen3_moe | 0 | refused TS940: mixture of experts routing (`Qwen3MoeSparseMoeBlock`) |  |  |  |
| hf/qwen3_next | 0 | refused TS940: mixture of experts routing (`Qwen3NextSparseMoeBlock`) |  |  |  |
| hf/qwen3_5 | 0 | refused TS941: chunked linear attention (`Qwen3_5GatedDeltaNet`) |  |  |  |
| hf/llama4_text | 0 | refused TS940: mixture of experts routing (`Llama4TextExperts`) |  |  |  |
| hf/deepseek_v3 | 0 | refused TS940: mixture of experts routing (`DeepseekV3MoE`) |  |  |  |
| hf/kimi_linear | 0 | refused TS940: mixture of experts routing (`KimiLinearExperts`) |  |  |  |
| hf/olmoe | 0 | refused TS940: mixture of experts routing (`OlmoeSparseMoeBlock`) |  |  |  |
| hf/mamba2 | 0 | refused TS942: chunked selective state space scan (`Mamba2Mixer`) |  |  |  |
| hf/falcon_h1 | 0 | refused TS942: chunked selective state space scan (`FalconH1Mixer`) |  |  |  |
| hf/nemotron_h | 0 | refused TS940: mixture of experts routing (`NemotronHMoE`) |  |  |  |
| nn/Conv1d | 1 | PASS | 2.6e-17 | none | PASS 1.5e-16 |
| nn/Conv1d_depthwise | 1 | PASS | 0.0e+00 | none | PASS 3.0e-16 |
| nn/Conv2d_grouped | 1 | PASS | 1.7e-16 | none | PASS 5.4e-16 |
| nn/Conv2d_depthwise | 1 | PASS | 6.1e-17 | none | PASS 1.8e-16 |
| nn/Conv3d | 1 | PASS | 3.7e-16 | none | PASS 1.4e-15 |
| nn/ConvTranspose1d | 1 | PASS | 0.0e+00 | none | PASS 2.2e-16 |
| nn/ConvTranspose2d | 1 | PASS | 0.0e+00 | none | PASS 3.8e-16 |
| nn/ConvTranspose2d_grouped | 1 | PASS | 0.0e+00 | none | PASS 6.3e-16 |
| nn/ConvTranspose3d | 1 | PASS | 0.0e+00 | none | PASS 7.9e-16 |
| nn/GroupNorm | 1 | PASS | 1.1e-16 | none | PASS 4.5e-16 |
| nn/InstanceNorm1d | 1 | PASS | 9.0e-17 | none | PASS 1.0e-16 |
| nn/InstanceNorm2d | 1 | PASS | 2.6e-16 | none | PASS 5.9e-16 |
| nn/BatchNorm1d | 1 | PASS | 6.5e-17 | none | PASS 1.5e-16 |
| nn/BatchNorm3d | 1 | PASS | 1.1e-16 | none | PASS 3.3e-16 |
| nn/LayerNorm | 1 | PASS | 1.8e-16 | none | PASS 2.6e-16 |
| nn/RMSNorm | 1 | PASS | 1.8e-16 | none | PASS 1.9e-16 |
| nn/LocalResponseNorm | 8 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/Upsample_nearest | 1 | PASS | 0.0e+00 | none | PASS 1.4e-16 |
| nn/Upsample_bilinear | 1 | PASS | 0.0e+00 | none | PASS 2.8e-16 |
| nn/Upsample_bilinear_ac | 1 | PASS | 1.6e-16 | none | PASS 9.4e-17 |
| nn/Upsample_nearest1d | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/Upsample_linear1d | 1 | PASS | 0.0e+00 | none | PASS 1.2e-16 |
| nn/Upsample_bicubic | 1 | PASS | 0.0e+00 | none | PASS 6.2e-16 |
| nn/PixelShuffle | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/PixelUnshuffle | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/ChannelShuffle | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/MaxPool1d | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/AvgPool1d | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/AdaptiveAvgPool1d | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/AdaptiveMaxPool1d | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/MaxPool3d | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/AvgPool3d | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/AdaptiveAvgPool3d | 1 | PASS | 7.4e-17 | none | PASS 1.5e-16 |
| nn/ReflectionPad2d | 1 | PASS | 0.0e+00 | none | PASS 1.5e-16 |
| nn/ReplicationPad2d | 1 | PASS | 0.0e+00 | none | PASS 1.6e-16 |
| nn/ReflectionPad1d | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/ZeroPad2d | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/CircularPad2d | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/Embedding | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/Embedding_padding_idx | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/Embedding_scale_grad_by_freq | 1 | PASS | 0.0e+00 | none | PASS 6.0e-17 |
| nn/EmbeddingBag_mean | 3 | PASS | 0.0e+00 | none | PASS 9.8e-17 |
| nn/EmbeddingBag_sum | 3 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/EmbeddingBag_offsets | 1 | PASS | 0.0e+00 | none | PASS 6.6e-17 |
| nn/EmbeddingBag_padding_max | 3 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/RNN_tanh | 2 | PASS | 0.0e+00 | none | PASS 2.7e-16 |
| nn/LSTM | 2 | PASS | 9.3e-17 | none | PASS 1.5e-16 |
| nn/LSTM_2layer_bi | 2 | PASS | 1.7e-16 | none | PASS 2.1e-16 |
| nn/GRU | 2 | PASS | 0.0e+00 | none | PASS 1.3e-16 |
| nn/LSTMCell | 2 | PASS | 0.0e+00 | none | PASS 4.5e-17 |
| nn/GRUCell | 2 | PASS | 0.0e+00 | none | PASS 3.4e-17 |
| nn/MultiheadAttention | 10 | PASS | 1.5e-16 | none | PASS 1.5e-16 |
| nn/TransformerDecoderLayer | 15 | PASS | 2.7e-16 | none | PASS 4.2e-16 |
| nn/Bilinear | 1 | PASS | 0.0e+00 | none | PASS 9.9e-17 |
| nn/Unfold | 1 | PASS | 0.0e+00 | none | PASS 1.7e-16 |
| nn/Fold | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/CosineSimilarity | 3 | PASS | 1.2e-16 | none | PASS 1.4e-16 |
| nn/PairwiseDistance | 3 | PASS | 0.0e+00 | none | PASS 7.0e-17 |
| nn/PReLU | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/ELU | 1 | PASS | 2.7e-17 | none | PASS 1.5e-16 |
| nn/SELU | 1 | PASS | 1.0e-16 | none | PASS 2.5e-16 |
| nn/CELU | 1 | PASS | 2.7e-17 | none | PASS 1.5e-16 |
| nn/LeakyReLU | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/Softplus | 1 | PASS | 1.1e-16 | none | PASS 1.7e-16 |
| nn/Mish | 1 | PASS | 2.7e-17 | none | PASS 1.7e-16 |
| nn/LogSigmoid | 1 | PASS | 1.4e-16 | none | PASS 2.9e-16 |
| nn/Softsign | 3 | PASS | 0.0e+00 | none | PASS 3.9e-17 |
| nn/Hardshrink | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/Softshrink | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/Tanhshrink | 2 | PASS | 0.0e+00 | none | PASS 2.1e-16 |
| nn/Threshold | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/GLU | 1 | PASS | 0.0e+00 | none | PASS 4.8e-17 |
| nn/Softmin | 2 | PASS | 1.7e-16 | none | PASS 2.9e-16 |
| nn/LogSoftmax | 1 | PASS | 1.5e-16 | none | PASS 1.6e-16 |
| nn/Softmax2d | 1 | PASS | 1.2e-16 | none | PASS 2.5e-16 |
| nn/SiLU | 1 | PASS | 5.5e-17 | none | PASS 1.8e-16 |
| nn/Hardswish | 1 | PASS | 0.0e+00 | none | PASS 1.7e-16 |
| nn/Hardsigmoid | 1 | PASS | 0.0e+00 | none | FAIL 3.0e-08 |
| nn/GELU | 1 | PASS | 5.4e-17 | none | PASS 1.7e-16 |
| nn/ReLU6 | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/Sigmoid | 1 | PASS | 1.1e-16 | none | PASS 1.4e-16 |
| nn/Tanh | 1 | PASS | 0.0e+00 | none | PASS 1.4e-16 |
| nn/MSELoss | 3 | PASS | 0.0e+00 | none | PASS 1.3e-16 |
| nn/L1Loss | 3 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/SmoothL1Loss | 3 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/HuberLoss | 3 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| nn/CrossEntropyLoss | 1 | PASS | 1.2e-16 | none | PASS 0.0e+00 |
| nn/CrossEntropyLoss_smooth | 1 | PASS | 0.0e+00 | none | PASS 8.4e-17 |
| nn/NLLLoss | 2 | PASS | 1.2e-16 | none | PASS 0.0e+00 |
| nn/BCEWithLogitsLoss | 3 | PASS | 0.0e+00 | none | PASS 1.7e-16 |
| nn/BCELoss | 3 | PASS | 0.0e+00 | none | PASS 1.7e-16 |
| nn/KLDivLoss | 5 | PASS | 0.0e+00 | none | PASS 1.5e-16 |
| op/diff | 3 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| op/diff_n2_append | 2 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| op/square | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| op/floor | 2 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| op/bitwise_not | 4 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| op/meshgrid_xy | 5 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| op/meshgrid_ij | 5 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| op/tile | 1 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| op/bucketize | 4 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| op/bucketize_right | 4 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| op/std_var | 4 | PASS | 2.1e-16 | none | PASS 9.3e-17 |
| op/gaussian_cut | 14 | PASS | 0.0e+00 | none | PASS 3.3e-17 |
| op/multiply | 2 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| op/sum_of_bool | 6 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| op/renamed_slice_add | 6 | PASS | 0.0e+00 | none | PASS 0.0e+00 |
| op/Conv2d_same_odd | 1 | PASS | 1.1e-16 | none | PASS 6.2e-16 |
| op/Conv2d_same_even | 1 | PASS | 1.2e-16 | none | PASS 6.2e-16 |
| op/Conv2d_valid | 1 | PASS | 1.1e-16 | none | PASS 3.2e-16 |
| op/logit_softcap | 3 | PASS | 0.0e+00 | none | PASS 8.3e-17 |
| op/attention_sinks | 11 | PASS | 1.2e-16 | none | PASS 1.6e-16 |

<!-- op_coverage:end -->

## Text, retrieval, speech and vision language models (2026-11-24, transformers 5.24.0)

The models most ML users run besides decoder LLMs, built from configs in code with random weights
(nothing downloaded) in `truestar.testing_hf_text.MODELS_TEXT`, measured by
`scripts/text_coverage.py` and tested by `tests/test_hf_text.py`. Same venv and bar as the current
models section above: transformers 5.24.0, torch 2.11.0 CPU, Linux server, CPU only. "import" is
`ir.import_module` of the float64 model against PyTorch float64 (bar 1e-10, at batch 2 and 5 where
the batch is symbolic); "compile" is `compile_program(prog, "cpu", "float32")`, first call verified
against the float64 reference, no float64 CPU fallback allowed; "train" is one training step's
gradient for every trainable weight against torch.autograd in float64 (bar 1e-10 in the tests).

```
PYTHONPATH=src python scripts/text_coverage.py --json text.json
PYTHONPATH=src python -m pytest -q tests/test_hf_text.py
```

What each wrapper runs (sizes: vocab 96, width 32, 2 layers, 4 heads of 8, MLP 64):

* encoder decoder: the encoder, the decoder over a 3 token prefix with the cache on, then one decode
  step that reads the self attention cache and the cross attention cache (output 0 is the step's
  logits). Batch symbolic.
* embedding encoders: padded batches (row i ends in `i % 3` pad tokens, the family's pad id) and the
  sentence embedding head, the masked mean of the last hidden state, L2 normalized. Batch symbolic.
* speech: wav2vec2 and HuBERT on 400 raw samples (three convolutions to 19 frames, the weight normed
  convolutional position embedding, the encoder); Whisper as the encoder decoder above on log mel
  features.
* vision language: one 32 x 32 image per row, the placeholder tokens in front of 4 text ids, the
  image features merged into those positions, the prefill with the cache on and one decode step.
  The prompt layout and the batch are fixed per program. transformers' runtime check that the
  placeholder count matches the features is switched off with `TRANSFORMERS_DISABLE_TORCH_CHECK=1`
  for the call (the wrapper builds the right count; with the check on, the import is refused by
  name, see below).

| Family (model type) | What is exercised | import (worst) | compile cpu float32 | train | Status |
|---|---|---|---|---|---|
| T5, Flan style (`t5`) | relative position buckets, gated GELU, decoder self and cross attention cache, decode step | PASS 7.2e-16 | PASS | PASS | fixed |
| BART (`bart`) | learned positions with offset, cross attention cache, decode step | PASS 4.2e-16 | PASS | PASS | fixed |
| Marian (`marian`) | sinusoidal positions (frozen), cross attention cache, decode step | PASS 4.3e-16 | PASS | PASS | fixed |
| BERT (`bert`) | padded batch, mean pooling, L2 norm | PASS 2.9e-16 | PASS | PASS | pass |
| RoBERTa (`roberta`) | positions from the pad id (cumsum), mean pooling | PASS 4.7e-16 | PASS | PASS | pass |
| XLM-RoBERTa (`xlm-roberta`) | as RoBERTa | PASS 4.7e-16 | PASS | PASS | pass |
| DistilBERT (`distilbert`) | no token types, mean pooling | PASS 4.3e-16 | PASS | PASS | pass |
| ModernBERT (`modernbert`) | RoPE, one global and one local (window 4) layer, GeGLU, mean pooling | PASS 4.0e-16 | PASS | PASS | pass |
| DeBERTa v3 (`deberta-v2`, v3 settings) | disentangled c2p and p2c attention, log bucketed relative positions, shared keys | PASS 3.9e-16 | PASS | PASS | fixed |
| wav2vec2 base (`wav2vec2`) | convolutional feature extractor with group norm, weight normed position convolution, post norm encoder | PASS 1.1e-15 | PASS | PASS | fixed |
| HuBERT large style (`hubert`) | feature extractor with layer norm in every convolution, stable (pre norm) encoder | PASS 1.3e-15 | PASS | PASS | fixed |
| Whisper decode (`whisper`) | encoder, decoder prefix, decode step with the cross attention cache | PASS 6.0e-16 | PASS | PASS | fixed |
| LLaVA (`llava`) | CLIP vision tower, projector, 16 image tokens merged (`masked_scatter`), Llama prefill and decode step | PASS 7.9e-16 | PASS | PASS | fixed |
| LLaVA-NeXT (`llava_next`) | anyres: base image and one crop, unpadding, row end tokens (36 image tokens), decode step | PASS 8.7e-16 | PASS | PASS | fixed |
| Gemma 3 multimodal (`gemma3`) | SigLIP tower (no pooling head, as Gemma 3), pooled to 4 tokens, bidirectional image block mask (built with `torch.vmap`), local and global layers, decode step | PASS 6.2e-16 | PASS | PASS | fixed |
| Qwen2.5-VL (`qwen2_5_vl`) | window attention vision encoder, 3D rope | refused | | | refused by name |
| Qwen3-VL (`qwen3_vl`) | deepstack vision encoder, interleaved 3D rope | refused | | | refused by name |
| Idefics3 (`idefics3`) | pixel shuffle connector | refused | | | refused by name |
| SmolVLM (`smolvlm`) | as Idefics3 | refused | | | refused by name |

Compile worst errors against the float64 reference: 6.7e-8 to 2.6e-7 (float32). Train worst:
2.7e-16 to 9.8e-16. Counts: 19 families, 15 pass every stage (5 needed nothing, 10 fixed here),
4 refused by name. `tests/test_hf_text.py`: 51 tests on transformers 5.24, all passing (skipped
on transformers 4.x, where the cache, mask and vision language code differ and nothing was measured);
`tests/test_importer_text.py`: 17 tests with plain torch modules.

**Refused by name** (`KNOWN_GAPS_TEXT`, tests check the refusal): torch.export itself cannot trace
these at a fixed image size, because the model reads the image layout from tensor values. Truestar
now names the model line instead of torch's internal frame:

| Family | Where torch.export stops |
|---|---|
| Qwen2.5-VL | `int(grid_t)` of `image_grid_thw` in the window attention index (transformers `vision_utils.py:162`, `get_vision_window_index`); the per image split and the 3D rope index also call `.tolist()` |
| Qwen3-VL | `torch.split(t, lengths.tolist())` per image in the vision attention (`modeling_qwen3_vl.py:304`) |
| Idefics3, SmolVLM | padding images dropped with `pixel_values[real_images_inds]` (`modeling_idefics3.py:557`, `modeling_smolvlm.py:480`): the image count becomes data dependent and the vision attention guards on it. The message names the attention line it fails at and the calls leading there |

The message starts `torch.export cannot trace TextWrap: modeling_qwen3_vl.py:304 in forward:` with
the code line, says it reads a size or a branch condition from the values of a tensor, gives torch's
first error line and the calls leading there, says nothing was imported, and suggests taking the
size as a Python int (fixed per program) or keeping the shape static.

**Fixed** (before: origin/main 9a9ad96 with the same fixture; every test in
`tests/test_importer_text.py` but one guard fails on 9a9ad96 and passes here, with plain torch
modules, so on any transformers):

| Problem | Families | Fix |
|---|---|---|
| `cat` of the transformers KV cache: every layer starts as `torch.tensor([])` and the first keys are cat onto it; torch skips a one dimensional empty tensor next to tensors of another rank, the importer raised `ShapeError: dim -2 is out of range for rank 1` | T5, BART, Marian, Whisper decode, every VLM decode step | `aten.cat` skips such tensors as torch does (dtype promotion kept) |
| the empty cache constant stays as an unused zero sized param; the compiler's leaf for it raised `TypeError: 'NoneType' object is not iterable` | the same | a zero sized leaf keeps no shape (`shape_of_leaf` already meant this) |
| `finfo(float64).min` as a `where` scalar (T5's masked position bias) was inlined into the kernel saturated to float32 but its check's reference used 1.8e308: `ProgramMismatchError`, relative error 1, on a right kernel | T5 | the check's reference reads the same saturated constant, as every non scalar param already did |
| `aten._weight_norm` | wav2vec2, HuBERT | `v * g / norm(v)` over every dim but `dim` |
| `aten.sign`, `aten.ceil` | DeBERTa v2 and v3 | `(x > 0) - (x < 0)` (frozen), `-floor(-x)`; edge values bit for bit (nan, -0, inf, 2^53) |
| integer `clamp` and `pad` ran on the float64 CPU fallback | DeBERTa (relative positions), Gemma 3 (image block ids) | run on the device by the reference implementation, checked exactly, as integer `maximum` and `minimum` already were |
| `aten.masked_scatter` | LLaVA, LLaVA-NeXT, Gemma 3 (the image feature merge) | the n-th true position reads source element n (cumulative sum and a gather); too few source elements raise, as in torch |
| `torch.vmap` inside the model left functorch markers (`_vmap_increment_nesting`, `_add_batch_dim`, ...) in the graph, one "unsupported op" line each | Gemma 3 (transformers builds every mask with a custom overlay through vmap) | `import_exported` reruns the export's ATen pass with an empty decomposition table when they are present; the other ops stay as exported |
| `aten.__and__` and `aten.__or__` (`a & b`) were taken for in place ops (the name ends in `_`): a program reading `a` afterwards was refused as a mutation | Gemma 3, any `a & b` | only `__iand__` style dunder names are in place |
| transformers' placeholder count check (`torch._check` of an `.item()`) gave three or more unrelated "unsupported op" lines | every VLM | one problem naming the check, the model line and `TRANSFORMERS_DISABLE_TORCH_CHECK=1` |
| torch.export failing on a data dependent size showed torch's own error | Qwen2.5-VL, Qwen3-VL, Idefics3, SmolVLM | the refusal names the model line and the calls leading to it |

LLaVA-NeXT also needed `image_sizes` as Python ints in the wrapper (as a tensor, the anyres grid is
read with `.tolist()` and torch.export cannot trace it); with ints the grid is fixed per program.

**Limits.** Tiny random models only, no pretrained checkpoint. CUDA not run (the PC GPU was in use
by another benchmark). The vision language programs fix the batch, the image size and the prompt
layout; a second image size is a second program. The placeholder count check is off in both the
program and the float64 copy it is compared with. ModernBERT's unpadded (flash attention) path is
not exercised: the sweep uses SDPA with a padding mask. An output that is itself a float64 mask
value (1.8e308) still fails the whole program check in float32, which cannot hold it; inside
attention it is fine.
