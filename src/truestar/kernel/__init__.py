"""Truestar kernel layer: a loop level IR, template free fusion, Triton and numba backends, a
compile cache keyed on the IR, and built in float64 verification of every compiled kernel.

    from truestar import kernel as K
    M, N = K.sym("M"), K.sym("N")
    x = K.buffer("x", (M, N))
    i, j, r = K.axis("i", M), K.axis("j", N), K.axis("r", N)
    y = K.output("y", (i, j), K.softmax_weight(x[i, j], over=j))
    fn = K.compile(K.kernel("softmax", [y]), "cuda")
    out = fn(x=torch.randn(64, 1000, device="cuda"))

See docs/KERNEL_IR.md.
"""

from .ir import (Axis, Buffer, Const, Dim, Expr, Index, IRError, Kernel, Load, NEG_INF, Output,
                 Reduce, Sym, abs_, axis, buffer, canonicalize, const, dim, eq, exp, index, kernel,
                 log, maximum, minimum, ne, order_free_sum, output, reduce_max, reduce_min, reduce_sum, relu,
                 rsqrt, sigmoid, softmax_weight, sqrt, Strided, strided, sym, tanh, Term,
                 to_order_free, where)
from .compiler import CompiledKernel, cache_dir, clear_memory_cache, collect_checks, compile, kernel_key, stats
from .tune import TuneCacheWarning
from .tune import stats as tune_stats
from .reference import ShapeError, run_reference
from .verify import KernelBiasError, KernelMismatchError, tolerance
from .graph import Graph, Tensor, CompiledGraph
from .triton_backend import Unschedulable
