"""Training on the Truestar IR: losses, optimizers, mixed precision and data parallel training, all
expressed as IR programs, and a small trainer that runs one step program per step.

See docs/TRAINING.md.
"""

from . import ops  # registers cast, log_softmax and all_reduce
from .amp import AutocastPolicy, EmulatedReferenceExecutor, LossScaler, autocast, emulate_precision
from .edit import Editor, params_to_inputs
from .losses import MSE, CausalLM, CrossEntropy, Loss, attach_loss, cross_entropy
from .ops import grad
from .optim import (SGD, Adam, AdamW, Lion, Optimizer, any_nonfinite, clip_by_global_norm, global_norm,
                    update_program)
from .step import TrainStep, build_train_step
from .trainer import Constant, NonFiniteError, StepResult, Trainer, WarmupCosine, first_nonfinite

__all__ = [
    "AutocastPolicy", "EmulatedReferenceExecutor", "LossScaler", "autocast", "emulate_precision",
    "Editor", "params_to_inputs", "MSE", "CausalLM", "CrossEntropy", "Loss", "attach_loss", "cross_entropy",
    "grad", "SGD", "Adam", "AdamW", "Lion", "Optimizer", "any_nonfinite", "clip_by_global_norm", "global_norm",
    "update_program", "TrainStep", "build_train_step", "Constant", "NonFiniteError", "StepResult", "Trainer",
    "WarmupCosine", "first_nonfinite",
]
