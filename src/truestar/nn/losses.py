"""Losses as inlined native functions: called inside a native function they become IR; called
eagerly they run through PyTorch. Each returns a scalar.

``cross_entropy`` reads the log probability of the target class with one ``gather`` instead of a one
hot product, uses ``log_softmax`` (one op with its own gradient rule, no differentiation through
the max shift), and with ``ignore_index`` divides by the number of counted rows, which is computed
in the program, so one program serves every batch size.
"""

from __future__ import annotations

from ..frontend import ops
from .core import inline


@inline
def mse(pred, target):
    """mean((pred - target)^2)."""
    d = pred - target
    return ops.mean(d * d)


@inline
def cross_entropy(logits, targets, ignore_index=None, label_smoothing=0.0):
    """Mean negative log likelihood of int64 ``targets`` [...] under ``logits`` [..., C].

    ``ignore_index``: rows whose target equals it count for nothing (the mean is over the other
    rows). ``label_smoothing`` eps: (1 - eps) * nll + eps * mean over classes of -log p, as
    torch.nn.functional.cross_entropy defines it."""
    C = logits.shape[-1]
    n = 1
    for d in targets.shape:
        n = n * d
    logp = ops.reshape(ops.log_softmax(logits, dim=-1), [n, C])
    t = ops.reshape(targets, [n])
    if ignore_index is None:
        nll = -ops.gather(logp, 1, ops.reshape(t, [n, 1]))
        loss = ops.mean(nll)
        if label_smoothing > 0.0:
            smooth = -ops.mean(logp)
            loss = (1.0 - label_smoothing) * loss + label_smoothing * smooth
        return loss
    keep = t != ignore_index
    safe = ops.where(keep, t, ops.zeros_like(t))
    w = ops.cast(ops.reshape(keep, [n, 1]), logits.dtype)
    nll = -ops.gather(logp, 1, ops.reshape(safe, [n, 1])) * w
    count = ops.sum(w)
    loss = ops.sum(nll) / count
    if label_smoothing > 0.0:
        smooth = -ops.sum(ops.mean(logp, dim=1, keepdim=True) * w) / count
        loss = (1.0 - label_smoothing) * loss + label_smoothing * smooth
    return loss


@inline
def accuracy(logits, targets):
    """Fraction of rows whose largest logit is the target (a float scalar, no gradient)."""
    hit = ops.argmax(logits, dim=-1) == targets
    return ops.mean(ops.cast(hit, logits.dtype))


__all__ = ["mse", "cross_entropy", "accuracy"]


# ----------------------------------------------------------------------------- more losses (2026-11-25)


@inline
def log1p(x):
    """log(1 + x), accurate near 0 (``ops.log(1 + x)`` loses every digit below the float epsilon):
    on [-1/2, 1/2] the series 2 atanh(z), z = x / (2 + x), |z| <= 1/3 (18 odd terms, below 1e-17
    relative); log(1 + x) outside, where 1 + x is exact or far from 1. The direct branch reads 1
    inside the series interval, so it stays finite there and its gradient through the where is 0."""
    cl = ops.clamp(x, min=-0.5, max=0.5)
    z = cl / (cl + 2.0)
    z2 = z * z
    acc = ops.full_like(x, 1.0 / 37.0)
    for k in range(35, 1, -2):
        acc = z2 * acc + (1.0 / k)
    series = z * 2.0 * (z2 * acc + 1.0)
    inside = ops.abs(x) <= 0.5
    outer = ops.where(inside, ops.ones_like(x), x)
    return ops.where(inside, series, ops.log(outer + 1.0))


@inline
def softplus(x):
    """log(1 + exp(x)) = max(x, 0) + log1p(exp(-|x|)): finite for every x, accurate for x << 0."""
    return ops.relu(x) + log1p(ops.exp(-ops.abs(x)))


@inline
def _reduce(x, reduction):
    if reduction == "mean":
        return ops.mean(x)
    if reduction == "sum":
        return ops.sum(x)
    return x


@inline
def binary_cross_entropy_with_logits(logits, target, pos_weight=None, reduction="mean"):
    """-(w y log s(x) + (1 - y) log(1 - s(x))) with s the sigmoid and w = ``pos_weight``, computed as
    (1 - y) x + (1 + (w - 1) y) softplus(-x), as torch.nn.functional defines it."""
    sp = softplus(-logits)
    if pos_weight is None:
        loss = (1.0 - target) * logits + sp
    else:
        loss = (1.0 - target) * logits + ((pos_weight - 1.0) * target + 1.0) * sp
    return _reduce(loss, reduction)


@inline
def binary_cross_entropy(probs, target, reduction="mean"):
    """-(y log p + (1 - y) log(1 - p)) with each log clamped at -100, as torch does (so p = 0 or 1
    gives a finite loss); log(1 - p) is ``log1p(-p)``."""
    lp = ops.clamp(ops.log(probs), min=-100.0)
    lq = ops.clamp(log1p(-probs), min=-100.0)
    return _reduce(-(target * lp + (1.0 - target) * lq), reduction)


@inline
def l1(pred, target, reduction="mean"):
    """|pred - target|."""
    return _reduce(ops.abs(pred - target), reduction)


@inline
def smooth_l1(pred, target, beta=1.0, reduction="mean"):
    """0.5 d^2 / beta where |d| < beta, else |d| - beta / 2 (beta 0 is L1): torch's smooth_l1_loss."""
    a = ops.abs(pred - target)
    if beta == 0.0:
        return _reduce(a, reduction)
    return _reduce(ops.where(a < beta, a * a * (0.5 / beta), a - 0.5 * beta), reduction)


@inline
def huber(pred, target, delta=1.0, reduction="mean"):
    """0.5 d^2 where |d| <= delta, else delta (|d| - delta / 2): torch's huber_loss."""
    a = ops.abs(pred - target)
    return _reduce(ops.where(a <= delta, a * a * 0.5, (a - 0.5 * delta) * delta), reduction)


@inline
def nll(logp, targets, ignore_index=None):
    """Mean of -logp[row, target] over int64 ``targets`` [...] with ``logp`` [..., C] log
    probabilities (torch's nll_loss); ``ignore_index`` rows count for nothing."""
    C = logp.shape[-1]
    n = 1
    for d in targets.shape:
        n = n * d
    lp = ops.reshape(logp, [n, C])
    t = ops.reshape(targets, [n])
    if ignore_index is None:
        return ops.mean(-ops.gather(lp, 1, ops.reshape(t, [n, 1])))
    keep = t != ignore_index
    safe = ops.where(keep, t, ops.zeros_like(t))
    w = ops.cast(ops.reshape(keep, [n, 1]), logp.dtype)
    return ops.sum(-ops.gather(lp, 1, ops.reshape(safe, [n, 1])) * w) / ops.sum(w)


@inline
def kl_div(logq, target, reduction="batchmean", log_target=False):
    """KL(target || q) pointwise as torch's kl_div: target (log target - logq), 0 where target is 0
    (``log_target``: target holds log probabilities, exp(t) (t - logq)). "batchmean" divides the sum
    by the first dim, the true KL per sample."""
    if log_target:
        pw = ops.exp(target) * (target - logq)
    else:
        pos = target > 0.0
        safe = ops.where(pos, target, ops.ones_like(target))
        pw = ops.where(pos, target * (ops.log(safe) - logq), ops.zeros_like(target))
    if reduction == "batchmean":
        return ops.sum(pw) / logq.shape[0]
    return _reduce(pw, reduction)


@inline
def cosine_similarity(a, b, dim=-1, eps=1e-8):
    """a . b / sqrt(max(|a|^2 |b|^2, eps^2)) along ``dim``: torch's cosine_similarity."""
    num = ops.sum(a * b, dim=dim)
    den = ops.sum(a * a, dim=dim) * ops.sum(b * b, dim=dim)
    return num / ops.sqrt(ops.clamp(den, min=eps * eps))


__all__ += ["log1p", "softplus", "binary_cross_entropy_with_logits", "binary_cross_entropy", "l1", "smooth_l1",
            "huber", "nll", "kl_div", "cosine_similarity"]
