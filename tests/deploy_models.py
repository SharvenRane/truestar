"""Small models for truestar export --to (docs/EXPORT.md, "Deploy targets"): random weights,
deterministic, no downloads."""

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, d_in=16, d_hidden=32, d_out=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.head = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.head(x)


class DecoderBlock(nn.Module):
    """One pre norm causal self attention block with its MLP, as in small GPT style decoders."""

    def __init__(self, d=32, heads=4):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q = q.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
        k = k.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
        v = v.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (d // self.heads) ** 0.5)
        mask = torch.ones(t, t, dtype=torch.bool).tril()
        att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(y)
        return x + self.down(F.gelu(self.up(self.ln2(x))))


class TinyDecoder(nn.Module):
    """Token embedding, learned positions, two decoder blocks, final norm and a tied free head:
    the forward pass of a small decoder only language model on token ids."""

    def __init__(self, vocab=64, d=32, t_max=16, layers=2):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(t_max, d)
        self.blocks = nn.ModuleList(DecoderBlock(d) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        t = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(t))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


def _randomize(model: nn.Module, seed: int) -> nn.Module:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=g) * 0.1)
                m.running_var.copy_(torch.rand(m.running_var.shape, generator=g) + 0.5)
                m.weight.copy_(1 + 0.2 * torch.randn(m.weight.shape, generator=g))
                m.bias.copy_(0.1 * torch.randn(m.bias.shape, generator=g))
            if isinstance(m, nn.LayerNorm):
                m.weight.copy_(1 + 0.2 * torch.randn(m.weight.shape, generator=g))
                m.bias.copy_(0.1 * torch.randn(m.bias.shape, generator=g))
    return model.eval()


def mlp():
    torch.manual_seed(0)
    return _randomize(MLP(), 0), (torch.randn(4, 16),)


def resnet18(size=64):
    import torchvision

    torch.manual_seed(0)
    return _randomize(torchvision.models.resnet18(num_classes=10), 1), (torch.randn(2, 3, size, size),)


def encoder_layer():
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0,
                                       batch_first=True)
    return _randomize(layer, 2), (torch.randn(2, 8, 32),)


def decoder():
    torch.manual_seed(0)
    return _randomize(TinyDecoder(), 3), (torch.randint(0, 64, (2, 8)),)


ZOO = {"mlp": mlp, "resnet18": resnet18, "encoder_layer": encoder_layer, "decoder": decoder}
