"""One small module per torch.nn layer and loss, with example inputs (random, seeded; nothing downloaded).

Used by scripts/op_coverage.py (the ``nn`` source of docs/MODEL_COVERAGE.md) and tests/test_op_coverage.py.

    from truestar.testing_ops import nn_layers
    for name, make in nn_layers():
        module, args = make()
"""

from __future__ import annotations

from typing import Any, List, Tuple

import torch
from torch import nn


class _Call(nn.Module):
    """A function of tensors as a module, for losses and functional layers."""

    def __init__(self, fn, **mods):
        super().__init__()
        self.fn = fn
        for k, v in mods.items():
            setattr(self, k, v)

    def forward(self, *xs):
        return self.fn(self, *xs)


def _r(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def nn_layers() -> List[Tuple[str, Any]]:
    """(name, () -> (module, args)) for the torch.nn layers and losses the models of the list need."""
    x4 = lambda: (_r(2, 4, 8, 8),)    # noqa: E731
    x3 = lambda: (_r(2, 4, 10),)      # noqa: E731
    x5 = lambda: (_r(2, 4, 4, 6, 6),)  # noqa: E731
    seq = lambda: (_r(2, 5, 4),)      # noqa: E731
    tgt = torch.tensor([1, 0, 3])
    L = [
        ("Conv1d", lambda: (nn.Conv1d(4, 6, 3, padding=1), x3())),
        ("Conv1d_depthwise", lambda: (nn.Conv1d(4, 4, 4, groups=4, padding=3), x3())),
        ("Conv2d_grouped", lambda: (nn.Conv2d(4, 6, 3, groups=2, padding=1), x4())),
        ("Conv2d_depthwise", lambda: (nn.Conv2d(4, 8, 3, groups=4, stride=2, padding=1), x4())),
        ("Conv3d", lambda: (nn.Conv3d(4, 3, 3, padding=1), x5())),
        ("ConvTranspose1d", lambda: (nn.ConvTranspose1d(4, 3, 3, stride=2, padding=1, output_padding=1), x3())),
        ("ConvTranspose2d", lambda: (nn.ConvTranspose2d(4, 3, 3, stride=2, padding=1, output_padding=1), x4())),
        ("ConvTranspose2d_grouped", lambda: (nn.ConvTranspose2d(4, 6, 4, stride=2, padding=1, groups=2), x4())),
        ("ConvTranspose3d", lambda: (nn.ConvTranspose3d(4, 2, 2, stride=2), x5())),
        ("GroupNorm", lambda: (nn.GroupNorm(2, 4), x4())),
        ("InstanceNorm1d", lambda: (nn.InstanceNorm1d(4, affine=True), x3())),
        ("InstanceNorm2d", lambda: (nn.InstanceNorm2d(4, affine=True), x4())),
        ("BatchNorm1d", lambda: (nn.BatchNorm1d(4), x3())),
        ("BatchNorm3d", lambda: (nn.BatchNorm3d(4), x5())),
        ("LayerNorm", lambda: (nn.LayerNorm(8), x4())),
        ("RMSNorm", lambda: (nn.RMSNorm(8), x4())),
        ("LocalResponseNorm", lambda: (nn.LocalResponseNorm(3), x4())),
        ("Upsample_nearest", lambda: (nn.Upsample(scale_factor=2, mode="nearest"), x4())),
        ("Upsample_bilinear", lambda: (nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), x4())),
        ("Upsample_bilinear_ac", lambda: (nn.Upsample(size=(5, 7), mode="bilinear", align_corners=True), x4())),
        ("Upsample_nearest1d", lambda: (nn.Upsample(scale_factor=2, mode="nearest"), x3())),
        ("Upsample_linear1d", lambda: (nn.Upsample(scale_factor=2, mode="linear"), x3())),
        ("Upsample_bicubic", lambda: (nn.Upsample(scale_factor=2, mode="bicubic"), x4())),
        ("PixelShuffle", lambda: (nn.PixelShuffle(2), (_r(2, 8, 3, 3),))),
        ("PixelUnshuffle", lambda: (nn.PixelUnshuffle(2), x4())),
        ("ChannelShuffle", lambda: (nn.ChannelShuffle(2), x4())),
        ("MaxPool1d", lambda: (nn.MaxPool1d(3, 2, padding=1), x3())),
        ("AvgPool1d", lambda: (nn.AvgPool1d(3, 2, padding=1), x3())),
        ("AdaptiveAvgPool1d", lambda: (nn.AdaptiveAvgPool1d(3), x3())),
        ("AdaptiveMaxPool1d", lambda: (nn.AdaptiveMaxPool1d(3), x3())),
        ("MaxPool3d", lambda: (nn.MaxPool3d(2), x5())),
        ("AvgPool3d", lambda: (nn.AvgPool3d(2), x5())),
        ("AdaptiveAvgPool3d", lambda: (nn.AdaptiveAvgPool3d(1), x5())),
        ("ReflectionPad2d", lambda: (nn.ReflectionPad2d(2), x4())),
        ("ReplicationPad2d", lambda: (nn.ReplicationPad2d(2), x4())),
        ("ReflectionPad1d", lambda: (nn.ReflectionPad1d(2), x3())),
        ("ZeroPad2d", lambda: (nn.ZeroPad2d((1, 2, 0, 1)), x4())),
        ("CircularPad2d", lambda: (nn.CircularPad2d(1), x4())),
        ("Embedding", lambda: (nn.Embedding(10, 4), (torch.tensor([[1, 2, 3], [4, 5, 9]]),))),
        ("Embedding_padding_idx", lambda: (nn.Embedding(10, 4, padding_idx=2), (torch.tensor([[1, 2, 3], [2, 5, 2]]),))),
        ("Embedding_scale_grad_by_freq", lambda: (nn.Embedding(10, 4, scale_grad_by_freq=True, padding_idx=0),
                                                  (torch.tensor([[1, 2, 1], [0, 1, 9]]),))),
        ("EmbeddingBag_mean", lambda: (nn.EmbeddingBag(10, 4, mode="mean"), (torch.tensor([[1, 2, 3], [4, 5, 9]]),))),
        ("EmbeddingBag_sum", lambda: (nn.EmbeddingBag(10, 4, mode="sum"), (torch.tensor([[1, 2, 3], [4, 5, 9]]),))),
        ("EmbeddingBag_offsets", lambda: (_Call(lambda s, ids, off: s.m(ids, off), m=nn.EmbeddingBag(10, 4, mode="mean")),
                                          (torch.tensor([1, 2, 3, 4, 9, 0, 5]), torch.tensor([0, 2, 5])))),
        ("EmbeddingBag_padding_max", lambda: (nn.EmbeddingBag(10, 4, mode="max", padding_idx=0),
                                              (torch.tensor([[1, 0, 3], [0, 0, 0], [4, 5, 9]]),))),
        ("RNN_tanh", lambda: (nn.RNN(4, 6, batch_first=True), seq())),
        ("LSTM", lambda: (nn.LSTM(4, 6, batch_first=True), seq())),
        ("LSTM_2layer_bi", lambda: (nn.LSTM(4, 6, num_layers=2, bidirectional=True, batch_first=True), seq())),
        ("GRU", lambda: (nn.GRU(4, 6, batch_first=True), seq())),
        ("LSTMCell", lambda: (nn.LSTMCell(4, 6), (_r(3, 4),))),
        ("GRUCell", lambda: (nn.GRUCell(4, 6), (_r(3, 4),))),
        ("MultiheadAttention", lambda: (_Call(lambda s, x: s.m(x, x, x, need_weights=False)[0],
                                              m=nn.MultiheadAttention(4, 2, batch_first=True)), seq())),
        ("TransformerDecoderLayer", lambda: (_Call(lambda s, x: s.m(x, x), m=nn.TransformerDecoderLayer(
            4, 2, 8, batch_first=True, dropout=0.0)), seq())),
        ("Bilinear", lambda: (_Call(lambda s, a: s.m(a, a), m=nn.Bilinear(4, 4, 3)), (_r(3, 4),))),
        ("Unfold", lambda: (nn.Unfold(3, padding=1), x4())),
        ("Fold", lambda: (nn.Fold((4, 4), 2, stride=2), (_r(2, 8, 4),))),
        ("CosineSimilarity", lambda: (_Call(lambda s, a: s.m(a, a * 2 + 1), m=nn.CosineSimilarity(1)), (_r(3, 4),))),
        ("PairwiseDistance", lambda: (_Call(lambda s, a: s.m(a, a * 2 + 1), m=nn.PairwiseDistance()), (_r(3, 4),))),
        ("PReLU", lambda: (nn.PReLU(4), x4())),
    ]
    acts = ["ELU", "SELU", "CELU", "LeakyReLU", "Softplus", "Mish", "LogSigmoid", "Softsign", "Hardshrink",
            "Softshrink", "Tanhshrink", "Threshold", "GLU", "Softmin", "LogSoftmax", "Softmax2d", "SiLU",
            "Hardswish", "Hardsigmoid", "GELU", "ReLU6", "Sigmoid", "Tanh"]
    for a in acts:
        def make(a=a):
            cls = getattr(nn, a)
            m = cls(0.5, 0.0) if a == "Threshold" else cls(dim=-1) if a in ("Softmin", "LogSoftmax") else cls()
            return m, x4()
        L.append((a, make))
    losses = [
        ("MSELoss", lambda s, x: nn.functional.mse_loss(x, x.flip(0))),
        ("L1Loss", lambda s, x: nn.functional.l1_loss(x, x.flip(0))),
        ("SmoothL1Loss", lambda s, x: nn.functional.smooth_l1_loss(x, x.flip(0))),
        ("HuberLoss", lambda s, x: nn.functional.huber_loss(x, x.flip(0), delta=0.5)),
        ("CrossEntropyLoss", lambda s, x: nn.functional.cross_entropy(x, tgt)),
        ("CrossEntropyLoss_smooth", lambda s, x: nn.functional.cross_entropy(x, tgt, label_smoothing=0.1)),
        ("NLLLoss", lambda s, x: nn.functional.nll_loss(x.log_softmax(-1), tgt)),
        ("BCEWithLogitsLoss", lambda s, x: nn.functional.binary_cross_entropy_with_logits(x, x.flip(0).sigmoid())),
        ("BCELoss", lambda s, x: nn.functional.binary_cross_entropy(x.sigmoid(), x.flip(0).sigmoid())),
        ("KLDivLoss", lambda s, x: nn.functional.kl_div(x.log_softmax(-1), x.flip(0).softmax(-1),
                                                        reduction="batchmean")),
    ]
    for name, fn in losses:
        L.append((name, lambda fn=fn: (_Call(fn), (_r(3, 4),))))
    return L


# ----------------------------------------------------------------------------- 2025 to 2026 model patterns


class _Sinks(nn.Module):
    """Attention with a learned sink logit per head (gpt-oss): one more column in each softmax row,
    dropped after the softmax, so every row's weights sum to less than one."""

    def __init__(self, heads=2, dim=4):
        super().__init__()
        self.h, self.d = heads, dim
        self.qkv = nn.Linear(heads * dim, 3 * heads * dim)
        self.sinks = nn.Parameter(torch.randn(heads, generator=torch.Generator().manual_seed(3)))

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        logits = q @ k.transpose(-1, -2) / self.d ** 0.5
        sink = self.sinks.reshape(1, self.h, 1, 1).expand(b, self.h, t, 1)
        w = torch.softmax(torch.cat([logits, sink], dim=-1), dim=-1)[..., :-1]
        return (w @ v).transpose(1, 2).reshape(b, t, self.h * self.d)


def _renamed_slice_add(s, x):
    """Gemma 3n's AltUp correction: an in place add on a slice of a tensor reached through
    contiguous and type_as after an in place add."""
    y = x * 2.0
    y += x
    z = y.contiguous().type_as(x)
    z[1:] += x[0]
    return z


def _gaussian_cut(s, x):
    """Gemma 3n's activation sparsity: keep what lies above mean + std * icdf(0.95), the icdf of a
    constant evaluated in the forward."""
    std_mult = torch.distributions.normal.Normal(0, 1).icdf(torch.tensor(0.95, dtype=x.dtype))
    cut = x.mean(-1, keepdim=True) + x.std(-1, keepdim=True, unbiased=False) * std_mult
    return torch.relu(x - cut)


def current_ops() -> List[Tuple[str, Any]]:
    """(name, () -> (module, args)) for the ATen ops and attention patterns the 2025 to 2026 families
    need (coverage refresh, ir/importer_current.py): each new op alone, and the patterns that are
    compositions of existing ops (attention sinks, logit soft capping), so their gradients are
    checked too."""
    x4 = lambda: (_r(2, 4, 8, 8),)    # noqa: E731
    x3 = lambda: (_r(3, 4, 5),)       # noqa: E731
    bounds = torch.tensor([-1.0, -0.25, 0.0, 0.5, 1.5])
    return [
        ("diff", lambda: (_Call(lambda s, x: x.diff(dim=-1, prepend=x[..., :1] * 0.5)), x3())),
        ("diff_n2_append", lambda: (_Call(lambda s, x: torch.diff(x, n=2, dim=1, append=x[:, :2])), x3())),
        ("square", lambda: (_Call(lambda s, x: torch.square(x)), x3())),
        ("floor", lambda: (_Call(lambda s, x: x * torch.floor(x * 3.0)), x3())),
        ("bitwise_not", lambda: (_Call(lambda s, x: torch.where(~(x > 0), x, -2.0 * x)), x3())),
        ("meshgrid_xy", lambda: (_Call(lambda s, x: (lambda g: g[0] * 2.0 + g[1])(
            torch.meshgrid(x[0, 0, :3], x[1, 1, :4], indexing="xy"))), x3())),
        ("meshgrid_ij", lambda: (_Call(lambda s, x: torch.stack(torch.meshgrid(
            x[0, 0, :3], x[1, 1, :4], indexing="ij"), dim=0).sum(0)), x3())),
        ("tile", lambda: (_Call(lambda s, x: x.tile((2, 1, 3))), x3())),
        ("bucketize", lambda: (_Call(lambda s, x: x * torch.bucketize(x, bounds).to(x.dtype)), x3())),
        ("bucketize_right", lambda: (_Call(lambda s, x: x * torch.bucketize(x, bounds, right=True).to(x.dtype)),
                                     x3())),
        ("std_var", lambda: (_Call(lambda s, x: x.std(-1, keepdim=True) + x.var(dim=(1, 2), correction=0)[:, None, None]),
                             x3())),
        ("gaussian_cut", lambda: (_Call(_gaussian_cut), x3())),
        ("multiply", lambda: (_Call(lambda s, x: torch.multiply(x, x.flip(0))), x3())),
        ("sum_of_bool", lambda: (_Call(lambda s, x: (x > 0).sum(-1).to(x.dtype)[..., None] * x), x3())),
        ("renamed_slice_add", lambda: (_Call(_renamed_slice_add), x3())),
        ("Conv2d_same_odd", lambda: (nn.Conv2d(4, 4, 3, padding="same"), x4())),
        ("Conv2d_same_even", lambda: (nn.Conv2d(4, 4, 4, padding="same", dilation=1), x4())),
        ("Conv2d_valid", lambda: (nn.Conv2d(4, 4, 3, padding="valid"), x4())),
        ("logit_softcap", lambda: (_Call(lambda s, x: torch.tanh(x / 3.0) * 3.0), x3())),
        ("attention_sinks", lambda: (_Sinks(), (_r(2, 5, 8),))),
    ]
