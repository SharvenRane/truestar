"""A deterministic image classifier with meaningful outputs, built without downloading anything.

Random weights alone make a poor study model: activations drift in scale, logits sit near zero and
almost every input gets the same class. This module builds a ResNet18 whose predictions behave like
a trained model's, so that agreement counts and near ties mean something:

1. Data. `num_classes` prototype images are smooth random patterns (a seeded 3x6x6 tensor upsampled
   bilinearly to the image size). A sample is its class prototype plus smooth noise plus pixel
   noise, with a per sample noise strength spread over a range, so some inputs are easy and some
   sit near a decision boundary. Labels are the prototype index.
2. Backbone. torchvision ResNet18 with seeded random conv weights and seeded batch norm affine
   parameters (gamma in [0.5, 1.5], beta in [-0.1, 0.1]). Every batch norm's running mean and
   variance are then set from a calibration batch (one forward pass in train mode with
   momentum=None), so every layer sees unit scale activations, as in a trained network.
3. Head. The final Linear is a nearest class mean classifier on the 512 dim pooled features of a
   separate training draw: logit_c = alpha * ((mu_c - mu) . (f - mu) - |mu_c - mu|^2 / 2). alpha is
   chosen so the logits have a standard deviation of `logit_std` (a few units) on that draw.

Everything depends only on the seeds and arguments, so the model and data are bit identical across
runs on the same torch version, and model fingerprints stay stable for the job cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SyntheticTask:
    model: nn.Module
    eval_x: torch.Tensor
    eval_y: torch.Tensor
    calib_x: torch.Tensor
    num_classes: int
    image_size: int


def _prototypes(num_classes: int, size: int, g: torch.Generator) -> torch.Tensor:
    low = torch.randn(num_classes, 3, 6, 6, generator=g)
    return F.interpolate(low, size=(size, size), mode="bilinear", align_corners=False)


def _samples(protos: torch.Tensor, n: int, g: torch.Generator, noise=(0.25, 1.0)):
    c, _, s, _ = protos.shape
    y = torch.arange(n) % c
    y = y[torch.randperm(n, generator=g)]
    strength = noise[0] + (noise[1] - noise[0]) * torch.rand(n, 1, 1, 1, generator=g)
    smooth = F.interpolate(torch.randn(n, 3, 8, 8, generator=g), size=(s, s), mode="bilinear",
                           align_corners=False)
    pixel = 0.3 * torch.randn(n, 3, s, s, generator=g)
    x = protos[y] + strength * smooth + pixel
    return x.float(), y


def _features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    m = model
    h = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    h = m.layer4(m.layer3(m.layer2(m.layer1(h))))
    return torch.flatten(m.avgpool(h), 1)


def build_task(num_classes: int = 20, image_size: int = 128, n_eval: int = 256, n_calib: int = 64,
               n_train_per_class: int = 16, logit_std: float = 2.5, seed: int = 0) -> SyntheticTask:
    import torchvision

    g = torch.Generator().manual_seed(seed)
    protos = _prototypes(num_classes, image_size, g)
    train_x, train_y = _samples(protos, num_classes * n_train_per_class, g)
    calib_x, _ = _samples(protos, n_calib, g)
    eval_x, eval_y = _samples(protos, n_eval, g)

    torch.manual_seed(seed)
    model = torchvision.models.resnet18(num_classes=num_classes)
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            with torch.no_grad():
                mod.weight.copy_(0.5 + torch.rand(mod.num_features, generator=g))
                mod.bias.copy_(0.2 * torch.rand(mod.num_features, generator=g) - 0.1)
            mod.reset_running_stats()
            mod.momentum = None  # cumulative average: one batch gives exactly its own statistics
    model.train()
    with torch.no_grad():
        _features(model, calib_x)
    model.eval()
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.momentum = 0.1

    with torch.no_grad():
        f = _features(model, train_x).double()
        mu = f.mean(0)
        mus = torch.stack([f[train_y == c].mean(0) for c in range(num_classes)]) - mu
        w = mus
        b = -(mus @ mu) - 0.5 * (mus * mus).sum(1)
        raw = f @ w.T + b
        alpha = logit_std / float(raw.std())
        model.fc.weight.copy_((alpha * w).float())
        model.fc.bias.copy_((alpha * b).float())
    return SyntheticTask(model=model, eval_x=eval_x, eval_y=eval_y, calib_x=calib_x,
                         num_classes=num_classes, image_size=image_size)
