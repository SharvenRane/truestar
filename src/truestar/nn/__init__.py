"""truestar.nn: a layer library written natively for the Truestar front end (docs/NN.md).

    from truestar import nn, Tensor
    import truestar

    net = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 1))
    p = net.init(seed=0)                       # an explicit tree of tensors

    @truestar.native
    def loss(p: net.spec, x: Tensor["B", 16], y: Tensor["B", 1]) -> Tensor[()]:
        return nn.mse(net(p, x), y)

Layers are immutable configs; parameters and state are trees the caller holds; training mode is a
compile time argument; random draws are keyed.
"""

from .core import (Init, Inline, Layer, Param, constant, fan_in_uniform, fans, gain, inline, kaiming_normal,
                   kaiming_uniform, layer, normal, ones, param_key, trunc_normal, uniform, xavier_normal,
                   xavier_uniform, zeros)
from .layers import (GELU, MLP, Activation, AvgPool2d, BatchNorm, Conv2d, Dropout, Embedding, Flatten, GlobalAvgPool,
                     LayerNorm, Linear, MaxPool2d, MultiHeadAttention, ReLU, RMSNorm, Sequential, SiLU, Tanh,
                     TransformerBlock)
from .layers_more import (ALiBi, ConvTranspose2d, Conv1d, EmbeddingBag, GatedMLP, GEGLU, GroupNorm, GRU, InstanceNorm,
                          LSTM, PixelShuffle, Rotary, SwiGLU, Upsample)
from .summary import Summary, count, summary
from .losses import (accuracy, binary_cross_entropy, binary_cross_entropy_with_logits, cosine_similarity, cross_entropy,
                     huber, kl_div, l1, log1p, mse, nll, smooth_l1, softplus)

__all__ = [
    "Init", "Inline", "Layer", "Param", "constant", "fan_in_uniform", "inline", "layer", "normal", "ones",
    "param_key", "uniform", "zeros", "fans", "gain", "kaiming_normal", "kaiming_uniform", "trunc_normal",
    "xavier_normal", "xavier_uniform",
    "GELU", "MLP", "Activation", "AvgPool2d", "BatchNorm", "Conv2d", "Dropout", "Embedding", "Flatten",
    "GlobalAvgPool", "LayerNorm", "Linear", "MaxPool2d", "MultiHeadAttention", "ReLU", "RMSNorm", "Sequential",
    "SiLU", "Tanh", "TransformerBlock", "accuracy", "cross_entropy", "mse",
    "ALiBi", "ConvTranspose2d", "Conv1d", "EmbeddingBag", "GatedMLP", "GEGLU", "GroupNorm", "GRU", "InstanceNorm",
    "LSTM", "PixelShuffle", "Rotary", "SwiGLU", "Upsample",
    "binary_cross_entropy", "binary_cross_entropy_with_logits", "cosine_similarity", "huber", "kl_div", "l1", "log1p",
    "nll", "smooth_l1", "softplus",
    "Summary", "count", "summary",
]
