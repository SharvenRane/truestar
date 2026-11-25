"""The ``random_batched`` IR op: what ``vmap`` makes of a keyed ``random`` op.

    %r = random_batched(%key, %s0, ...) dist=normal shape=[V, 4, 8] dtype=f32 site=... : f32[V, 4, 8]

Row k is bit for bit ``random(%key, %s0, ..., k)`` with shape ``shape[1:]``: the draw the op makes
alone with the batch index as one more stream, exactly as if the batch were a loop whose index is a
stream (docs/RANDOM.md, docs/TRANSFORMS.md). So every element has its own stream, independent of the
others, and each element can be reproduced alone. It is a new op name rather than a new attribute of
``random`` so that a consumer that does not know it fails on its name instead of drawing one shared
stream. The reference and the compiled step both use ``truestar.rng.generate_batch``, which is
vectorized over the batch (per row keys from one ``mix`` over ``arange``).
"""

from __future__ import annotations

import torch

from . import reference, shapes
from .core import ShapeError, TensorType

OP = "random_batched"


@shapes.rule(OP)
def _shape(types, attrs):
    t = shapes.RULES["random"](types, attrs)
    if t.rank < 1:
        raise ShapeError(OP, "shape needs a leading batch dim", [("key", types[0])] if types else [])
    return t


@reference.impl(OP)
def _run(attrs, args, shape):
    from .. import rng

    key = int(args[0].item())
    streams = [int(a.item()) for a in args[1:]]
    out = rng.generate_batch(attrs["dist"], shape, attrs["dtype"], key, attrs["site"], streams, p=attrs.get("p"),
                             low=attrs.get("low", 0), high=attrs.get("high"), device=args[0].device)
    return out.to(reference.COMPUTE_DTYPE) if out.dtype.is_floating_point else out
