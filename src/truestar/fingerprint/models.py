"""The models a fingerprint runs, with weights and inputs that are the same bits on every machine.

PyTorch's CPU random draws are not: the first T4 run (2026-11-24) found that ``torch.manual_seed``
followed by the default initializers and ``torch.randn`` with a seeded CPU generator gave other bits
on a Kaggle Linux x86 CPU than on the Windows desktop (every case's weights and inputs differed),
so its outputs could not be compared. The cases here therefore overwrite every parameter and draw
every input with ``portable_uniform``: a counter based integer hash (splitmix64 in numpy uint64,
wrapping arithmetic) whose top 24 bits become ``k / 2**24`` exactly, and normal like inputs as the
exact float64 sum of 12 such draws minus 6, rounded once to float32. No libm call, no vectorized
kernel and no platform dependent path is involved. Inputs are drawn at the largest batch size;
smaller batches take the first rows. ``data.py`` hashes both, and ``GOLDEN`` pins the hashes.

The ``legacy_*`` cases are the three models of the first cross chip run (batch 4, float32), built
exactly as that run built them (PyTorch's CPU generator), so the new fingerprint can be compared
with its two JSON files. They are portable only between machines whose CPU draws agree (the two
Windows machines of the first run did; the Kaggle Linux CPU did not).
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCHES = (1, 7, 64, 256)
DTYPES = ("float32", "tf32", "float16", "bfloat16")
LEGACY_BATCH = 4


@dataclass(frozen=True)
class Case:
    name: str
    build: Callable[[], nn.Module]            # after torch.manual_seed(seed), on the CPU
    make_inputs: Callable[[int], Tuple[torch.Tensor, ...]]   # inputs at batch n, CPU generator
    batches: Tuple[int, ...] = BATCHES
    dtypes: Tuple[str, ...] = DTYPES
    seed: int = 0
    legacy: bool = False
    note: str = ""
    float_inputs: Tuple[bool, ...] = field(default=(True,))  # which inputs take the working dtype


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


_M64 = (1 << 64) - 1


def _splitmix(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def portable_uniform(seed: int, stream: int, n: int) -> np.ndarray:
    """n float64 values k / 2**24 in [0, 1), the same bits on every platform."""
    key = np.uint64(((seed * 1_000_003 + stream) * 0x2545F4914F6CDD1D) & _M64)
    with np.errstate(over="ignore"):
        x = _splitmix(np.arange(n, dtype=np.uint64) ^ key)
    return (x >> np.uint64(40)).astype(np.float64) / float(1 << 24)


def portable_normal(seed: int, stream: int, shape) -> torch.Tensor:
    """Mean 0, variance 1 float32 values: 12 uniforms summed exactly in float64, minus 6."""
    n = int(np.prod(shape)) if len(shape) else 1
    u = portable_uniform(seed, stream, 12 * n).reshape(n, 12)
    return torch.from_numpy((u.sum(axis=1) - 6.0).astype(np.float32).reshape(shape))


def portable_ints(seed: int, stream: int, shape, high: int) -> torch.Tensor:
    n = int(np.prod(shape))
    u = portable_uniform(seed, stream, n)
    return torch.from_numpy(np.minimum((u * high).astype(np.int64), high - 1).reshape(shape))


def portable_init(model: nn.Module, seed: int) -> nn.Module:
    """Overwrite every parameter with portable draws: matrices and kernels U(-1, 1) / sqrt(fan in),
    one dimensional weights (norm scales) 1 + 0.1 U(-1, 1), biases 0.1 U(-1, 1). Buffers (batch
    norm statistics) keep their constant defaults."""
    with torch.no_grad():
        for i, (name, p) in enumerate(model.named_parameters()):
            u = torch.from_numpy((portable_uniform(seed, 1000 + i, p.numel()) * 2.0 - 1.0).astype(np.float32))
            u = u.reshape(p.shape)
            if p.dim() >= 2:
                p.copy_(u * np.float32(1.0 / np.sqrt(p[0].numel())))
            elif name.endswith("weight"):
                p.copy_(1.0 + np.float32(0.1) * u)
            else:
                p.copy_(np.float32(0.1) * u)
    return model


# ---- MLP: split K at small batches outside deterministic mode

def _mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(512, 2048), nn.GELU(approximate="tanh"), nn.Linear(2048, 2048),
                         nn.ReLU(), nn.Linear(2048, 10))


def _mlp_inputs(n: int):
    return (portable_normal(101, 0, (n, 512)),)


# ---- TransformerEncoderLayer: at batch 256 the feed forward contraction (16384 x 1024 x 256) and
# the attention reach the autotuning thresholds (2^32 and 2^28 multiply adds)

def _transformer() -> nn.Module:
    return nn.TransformerEncoderLayer(256, 8, 1024, dropout=0.0, batch_first=True)


def _transformer_inputs(n: int):
    return (portable_normal(102, 0, (n, 64, 256)),)


# ---- resnet18 at 64 x 64: convolutions are library calls (cuDNN) inside the compiled program

def _resnet18() -> nn.Module:
    import torchvision

    return torchvision.models.resnet18()


def _resnet_inputs(n: int):
    return (portable_normal(103, 0, (n, 3, 64, 64)),)


# ---- a small Llama style decoder step: RMSNorm, rotary positions, grouped query attention over a
# KV cache, SwiGLU, untied head. One new token per row attends to the cache and itself.

DIM, HEADS, KV_HEADS, HEAD_DIM, HIDDEN, VOCAB, LAYERS, PAST = 256, 8, 2, 32, 768, 1024, 2, 63


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) + 0.1 * torch.randn(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def _rotate(x, pos: int):
    half = x.shape[-1] // 2
    inv = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = pos * inv
    cos = torch.cos(ang).to(x.dtype).to(x.device)
    sin = torch.sin(ang).to(x.dtype).to(x.device)
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = _RMSNorm(DIM)
        self.q = nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
        self.k = nn.Linear(DIM, KV_HEADS * HEAD_DIM, bias=False)
        self.v = nn.Linear(DIM, KV_HEADS * HEAD_DIM, bias=False)
        self.o = nn.Linear(HEADS * HEAD_DIM, DIM, bias=False)
        self.mlp_norm = _RMSNorm(DIM)
        self.gate = nn.Linear(DIM, HIDDEN, bias=False)
        self.up = nn.Linear(DIM, HIDDEN, bias=False)
        self.down = nn.Linear(HIDDEN, DIM, bias=False)

    def forward(self, x, past_k, past_v):
        B = x.shape[0]
        h = self.attn_norm(x)
        q = _rotate(self.q(h).view(B, 1, HEADS, HEAD_DIM).transpose(1, 2), PAST)
        k = _rotate(self.k(h).view(B, 1, KV_HEADS, HEAD_DIM).transpose(1, 2), PAST)
        v = self.v(h).view(B, 1, KV_HEADS, HEAD_DIM).transpose(1, 2)
        keys = torch.cat([past_k, k], dim=2)
        values = torch.cat([past_v, v], dim=2)
        rep = HEADS // KV_HEADS
        kk = keys.repeat_interleave(rep, dim=1)
        vv = values.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, kk.transpose(-1, -2)) * (1.0 / math.sqrt(HEAD_DIM))
        att = torch.matmul(torch.softmax(scores, dim=-1), vv)
        x = x + self.o(att.transpose(1, 2).reshape(B, 1, HEADS * HEAD_DIM))
        h = self.mlp_norm(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x, keys, values


class LlamaStep(nn.Module):
    """One decode step: token ids [B, 1] and a cache [B, KV_HEADS, PAST, HEAD_DIM] per layer in;
    logits [B, VOCAB] and the grown caches out."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList(_Block() for _ in range(LAYERS))
        self.norm = _RMSNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, ids, k0, v0, k1, v1):
        x = self.embed(ids)
        x, k0, v0 = self.blocks[0](x, k0, v0)
        x, k1, v1 = self.blocks[1](x, k1, v1)
        return self.head(self.norm(x))[:, 0], k0, v0, k1, v1


def _llama_inputs(n: int):
    ids = portable_ints(104, 0, (n, 1), VOCAB)
    caches = tuple(portable_normal(104, 1 + i, (n, KV_HEADS, PAST, HEAD_DIM)) for i in range(2 * LAYERS))
    return (ids,) + caches


# ---- the first cross chip run's models (batch 4, float32 only)

def _legacy_inputs(shape):
    return lambda n: (torch.randn(n, *shape, generator=_gen(1234)),)


CASES: Dict[str, Case] = {c.name: c for c in [
    Case("mlp", _mlp, _mlp_inputs, note="512 -> 2048 -> 2048 -> 10; split K at batch 1 and 7 outside "
                                        "deterministic mode"),
    Case("transformer", _transformer, _transformer_inputs,
         note="TransformerEncoderLayer(256, 8, 1024), 64 tokens; batch 256 reaches the tuning thresholds"),
    Case("resnet18", _resnet18, _resnet_inputs, note="torchvision resnet18, 3 x 64 x 64; convolutions "
                                                    "run in cuDNN inside the compiled program"),
    Case("llama_step", LlamaStep, _llama_inputs, float_inputs=(False, True, True, True, True),
         note=f"{LAYERS} layer Llama style decode step, dim {DIM}, {HEADS} heads, {KV_HEADS} KV "
              f"heads, cache {PAST}, vocab {VOCAB}"),
    Case("legacy_mlp", lambda: nn.Sequential(nn.Linear(256, 512), nn.GELU(approximate="tanh"),
                                             nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10)),
         _legacy_inputs((256,)), batches=(LEGACY_BATCH,), dtypes=("float32",), legacy=True,
         note="first cross chip run's mlp"),
    Case("legacy_transformer",
         lambda: nn.TransformerEncoderLayer(128, 4, 512, dropout=0.0, batch_first=True),
         _legacy_inputs((32, 128)), batches=(LEGACY_BATCH,), dtypes=("float32",), legacy=True,
         note="first cross chip run's transformer"),
    Case("legacy_resnet18", _resnet18, _legacy_inputs((3, 64, 64)), batches=(LEGACY_BATCH,),
         dtypes=("float32",), legacy=True, note="first cross chip run's resnet18"),
]}


def build(case: Case) -> nn.Module:
    torch.manual_seed(case.seed)
    m = case.build().eval()
    return m if case.legacy else portable_init(m, zlib.crc32(case.name.encode()))


# sha256 of the weights and of the inputs (at the largest batch) of every portable case, measured
# on the Windows desktop (2026-11-24); data.prepare compares against it
GOLDEN: Dict[str, Tuple[str, str]] = {
    "mlp": ("e8a5ce94bc6355fb855d3bd50ad451f759ede388ea92cf715cd8cc80e0409e10",
            "1e220417baf2fa71f2d2f4e0432d1db1d100f49e084631791a1014de22fd02e3"),
    "transformer": ("b106ccb2ac9de63885569bea1c8474c2f3a47fcde7ba4905d3fd684035be2652",
                    "416d9a062b2796513c2c9f2917e9dc6edc898f8e8e950096102fce9e7db79602"),
    "resnet18": ("28df4ae0448c1d878f4caf33818999d619837b6197903adc77e470fd396b6aff",
                 "0ac2c3cd49ec8c83244cd13d2d823e79d8197bd5721715b1cb8f1fcb96651ffc"),
    "llama_step": ("aba5cf334b796a6d4559377cb1fe0bdaed291955bd06875d683e8264cccdc744",
                   "bb707eb8879630dedb147e0a493ad2c7517f6d56096028df5b789bc8a8059579"),
}


def case_names() -> List[str]:
    return list(CASES)
