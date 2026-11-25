"""Training mode for the checkers (``check``, ``grad_check``, ``invariance`` with ``train=True``).

PyTorch runs a model differently in training mode, and that is the forward every training step
actually uses: batch norm normalizes with batch statistics and updates its running statistics in
place, and dropout draws masks from a global generator whose bits differ from device to device. A
checker that compares backends must therefore make two things equal on every backend:

* **Dropout masks.** Inside ``TrainMode`` the functional dropout PyTorch modules call
  (``torch.nn.functional.dropout``, used by ``nn.Dropout``, ``nn.MultiheadAttention`` and the
  Transformer layers) is replaced by Truestar's keyed dropout (truestar.rng): the mask of a call is a
  pure function of (seed, module name, the module's call index, the dropout's ordinal inside that
  call, element index), so the reference, each backend and each isolated replay of a module draw the
  same mask. Any other random ATen op (``bernoulli_``, ``native_dropout``, ``randn`` and so on) that
  still runs is refused with ``TrainModeError``: comparing backends that drew different random
  numbers would be a false alarm, and checking with the randomness switched off would be a silent
  change of the model.
* **State.** The state every module's subtree holds (batch norm ``running_mean``, ``running_var``,
  ``num_batches_tracked``) is recorded at the start of each call in the reference run, loaded
  before that call is replayed alone, and the state after the call is compared like an output. So a
  module is judged on its training forward and on its running statistic update.

The module name of a call is the recorder's (``<model>`` for the root, ``layer1.0.bn1`` and so on).
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from . import rng

ROOT = "<model>"

# ATen ops that draw from PyTorch's generator. Under TrainMode they are an error (see module doc).
_RANDOM_OPS = {"bernoulli", "bernoulli_", "native_dropout", "rand", "rand_like", "randn", "randn_like", "randint",
               "randint_like", "randperm", "normal", "normal_", "uniform_", "exponential_", "geometric_",
               "cauchy_", "log_normal_", "multinomial", "random_", "_fused_dropout", "poisson", "rrelu_with_noise",
               "_standard_gamma", "binomial", "dirichlet"}
_SDPA_DROPOUT = {"_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention",
                 "_scaled_dot_product_cudnn_attention", "_scaled_dot_product_flash_attention_for_cpu",
                 "_flash_attention_forward", "_efficient_attention_forward"}

_ACTIVE: List["TrainMode"] = []


class TrainModeError(RuntimeError):
    """A random op Truestar cannot key ran during a training mode check."""


def active() -> Optional["TrainMode"]:
    return _ACTIVE[-1] if _ACTIVE else None


def _join(prefix: str, name: str) -> str:
    if prefix == ROOT:
        return name or ROOT
    return f"{prefix}.{name}" if name else prefix


class _Guard(TorchDispatchMode):
    """Refuses random ops the model runs (inside a module call); the checker's own random targets
    and inputs, drawn outside any module, are left alone."""

    def __init__(self, tm: "TrainMode"):
        super().__init__()
        self.tm = tm

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if not self.tm._stack:
            return func(*args, **kwargs)
        name = str(func.overloadpacket).split(".")[-1]
        if name in _RANDOM_OPS:
            raise TrainModeError(
                f"{func} draws from PyTorch's random generator during a training mode check; its bits differ "
                "between devices, so the backends cannot be compared. Truestar keys torch.nn.functional.dropout "
                "(nn.Dropout, nn.MultiheadAttention, Transformer layers); replace this op by a keyed one "
                "(truestar.rng.dropout, truestar.rng.generate) or check this model with train=False")
        if name in _SDPA_DROPOUT:
            p = args[3] if len(args) > 3 and isinstance(args[3], float) else kwargs.get("dropout_p", 0.0)
            if p:
                raise TrainModeError(f"{func} runs attention dropout (dropout_p={p}) inside a fused kernel with "
                                     "PyTorch's generator; pass dropout_p=0 and apply nn.Dropout to the weights, "
                                     "or check with train=False")
        return func(*args, **kwargs)


class TrainMode:
    """Context manager: keyed dropout, state recording and state restore for training mode checks.

    Use one instance for a whole check. ``attach(model)`` names the modules of each full model copy;
    the reference run records with ``recording=True``; ``replay(call_key, name)`` wraps an isolated
    replay of one module call (or a drilldown that runs copies of that module).
    """

    def __init__(self, seed: int = 0, per_row: bool = False):
        self.seed = rng.as_u64(seed)
        # per_row: one mask over the dims after dim 0, shared by every row, so a row's mask does not
        # depend on where it sits in the batch (what the batch invariance check needs)
        self.per_row = per_row
        self.recording = False
        self.start_counts: Dict[str, Dict[str, int]] = {}
        self.state_in: Dict[str, List[torch.Tensor]] = {}
        self._roots: Dict[int, Dict[int, str]] = {}
        self._names: Optional[Dict[int, str]] = None
        self._counts: Dict[str, int] = {}
        self._stack: List[Optional[list]] = []
        self._replay: Optional[Tuple[str, str]] = None
        self._handles = []
        self._saved = None
        self._guard = None
        self.masks = 0
        self._dropout = lambda input, p=0.5, training=True, inplace=False: self.dropout(input, p, training, inplace)
        self._sdpa = lambda *a, **k: self.attention(*a, **k)

    # ---------------------------------------------------------------- setup

    def attach(self, model: nn.Module) -> nn.Module:
        """Name the modules of a full model copy as the recorder does."""
        self._roots[id(model)] = {id(m): (n or ROOT) for n, m in model.named_modules()}
        return model

    def __enter__(self) -> "TrainMode":
        import torch.nn.functional as F

        self._saved = (F.dropout, F.scaled_dot_product_attention)
        F.dropout = self._dropout
        F.scaled_dot_product_attention = self._sdpa
        self._handles = [nn.modules.module.register_module_forward_pre_hook(self._pre),
                         nn.modules.module.register_module_forward_hook(self._post)]
        self._guard = _Guard(self)
        self._guard.__enter__()
        _ACTIVE.append(self)
        return self

    def __exit__(self, *exc) -> None:
        import torch.nn.functional as F

        _ACTIVE.remove(self)
        self._guard.__exit__(*exc)
        for h in self._handles:
            h.remove()
        F.dropout, F.scaled_dot_product_attention = self._saved

    @contextlib.contextmanager
    def replay(self, call_key: str, name: str):
        """Inside: every outermost module call is call ``call_key`` of module ``name``, starting
        from the state and the call counts the reference run had when that call started."""
        old = self._replay
        self._replay = (call_key, name)
        try:
            yield self
        finally:
            self._replay = old

    # ---------------------------------------------------------------- hooks

    def _pre(self, module, args):
        if not self._stack:
            if self._replay is not None:
                key, name = self._replay
                self._names = {id(m): _join(name, n) for n, m in module.named_modules()}
                self._counts = dict(self.start_counts.get(key, {}))
                saved = self.state_in.get(key)
                if saved is not None:
                    load_state(module, saved)
            else:
                self._names = self._roots.get(id(module))
                self._counts = {}
        name = None if self._names is None else self._names.get(id(module))
        if name is None:
            self._stack.append(None)
            return None
        idx = self._counts.get(name, 0)
        key = f"{name}#{idx}"
        if self.recording and self._replay is None:
            self.start_counts[key] = dict(self._counts)
            self.state_in[key] = state_tensors(module)
        self._counts[name] = idx + 1
        self._stack.append([name, idx, 0])
        return None

    def _post(self, module, args, output):
        if self._stack:
            self._stack.pop()
        return None

    # ---------------------------------------------------------------- keyed dropout

    def attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                  enable_gqa=False):
        """scaled_dot_product_attention; with dropout_p > 0 written out as softmax, keyed dropout on
        the weights, matmul (the fused kernels draw their own mask from PyTorch's generator)."""
        orig = self._saved[1]
        if not dropout_p:
            return orig(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal, scale=scale,
                        enable_gqa=enable_gqa)
        if enable_gqa:
            raise TrainModeError("attention dropout with enable_gqa=True is not keyed yet; check with train=False")
        L, S = query.shape[-2], key.shape[-2]
        s = (1.0 / (query.shape[-1] ** 0.5)) if scale is None else scale
        w = (query @ key.transpose(-2, -1)) * s
        if is_causal:
            keep = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
            w = w.masked_fill(~keep, float("-inf"))
        if attn_mask is not None:
            w = w.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else w + attn_mask
        w = torch.softmax(w, dim=-1)
        return self.dropout(w, dropout_p) @ value

    def dropout(self, input: torch.Tensor, p: float = 0.5, training: bool = True, inplace: bool = False):
        if not training or p == 0.0:
            return input
        if p < 0.0 or p > 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        frame = next((f for f in reversed(self._stack) if f is not None), None)
        if frame is None:
            raise TrainModeError("dropout ran outside any module of the checked model; wrap it in a module so "
                                 "its mask can be keyed by the module's name")
        key = rng.fold(rng.fold(self.seed, f"{frame[0]}#{frame[1]}"), frame[2])
        frame[2] += 1
        self.masks += 1
        if p == 1.0:
            out = torch.zeros_like(input)
        else:
            shape = tuple(input.shape)
            if self.per_row and input.dim() >= 1:
                shape = (1,) + shape[1:]
            with _disable_current_modes():   # the generator kernel is not an op of the model
                keep = rng.generate("bernoulli", shape, torch.bool, key, "dropout", p=1.0 - p,
                                    device=input.device)
            out = torch.where(keep, input * (1.0 / (1.0 - p)), torch.zeros((), dtype=input.dtype,
                                                                            device=input.device))
        if inplace:
            with torch.no_grad():
                input.copy_(out)
            return input
        return out



# ----------------------------------------------------------------------------- state


def state_tensors(module: nn.Module) -> List[torch.Tensor]:
    """CPU copies of every buffer in the module's subtree, in named_buffers order. Taken outside any
    dispatch mode: saving and loading state is the checker's work, not an op of the model."""
    with _disable_current_modes():
        return [b.detach().to("cpu", copy=True) for _, b in module.named_buffers()]


def load_state(module: nn.Module, saved: List[torch.Tensor]) -> None:
    bufs = [b for _, b in module.named_buffers()]
    if len(bufs) != len(saved):
        raise RuntimeError(f"state of {len(saved)} buffers recorded, the module has {len(bufs)}")
    with torch.no_grad(), _disable_current_modes():
        for b, s in zip(bufs, saved):
            b.copy_(s.to(device=b.device, dtype=b.dtype))


def train_outputs(module: nn.Module) -> List[torch.Tensor]:
    """The state after a call, compared like extra outputs (empty outside TrainMode)."""
    return state_tensors(module) if active() is not None else []


def prepare(model: nn.Module, train: bool) -> nn.Module:
    """A CPU copy of ``model`` in the requested mode (the checkers never change the caller's model)."""
    import copy

    m = copy.deepcopy(model).cpu()
    return m.train() if train else m.eval()


def describe(train: bool) -> str:
    return ("training mode: batch statistics with running statistic updates compared as outputs, keyed "
            "dropout masks identical on every backend") if train else "eval mode"


__all__ = ["TrainMode", "TrainModeError", "active", "state_tensors", "load_state", "train_outputs", "prepare"]
