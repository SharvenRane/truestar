"""One kernel set for the RL trainer and sampler (pain P15 part b, docs/RL_KERNELS.md).

`rl_mode()` routes a plain PyTorch model's contractions, attention, norms, softmaxes and sums to
batch and position invariant Triton kernels, so the training forward, the prefill and token by
token decoding against a KV cache give bitwise equal logprobs. `tp` simulates tensor parallel
splits on one device with a fixed all reduce order. CUDA only.
"""

from .kernels import RLKernelError, available  # noqa: F401
from .mode import KERNEL_OPS, KernelTrace, RLModeError, rl_mode  # noqa: F401

__all__ = ["rl_mode", "RLModeError", "RLKernelError", "KernelTrace", "KERNEL_OPS", "available"]
