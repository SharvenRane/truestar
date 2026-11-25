"""Metal backend: schedule a kernel into a Metal loop shape and generate Metal Shading Language.

The structure analysis is the one the Triton backend uses (``triton_backend.plan``, which reads only
the IR, never imports Triton), so fusion makes the same kernels for "metal" as for "cuda". Each
structural shape maps to a Metal loop shape written here from the IR:

* ``pointwise`` (tile plan without reductions): one thread per output element.
* ``rows`` (tile plan with reductions): one threadgroup per row; every reduction is computed by
  the group together (each thread a strided part in index order, then a fixed tree in threadgroup
  memory), and the lanes of the row are shared out over the threads.
* ``contraction``: 32 x 32 output tiles per threadgroup of 128 threads, the factors (with any
  elementwise prologue) staged through threadgroup memory 16 deep in K, the product either with
  SIMD group matrices (8 x 8 multiply accumulate, four per SIMD group) or, where the device does
  not run those correctly, with plain multiply adds; then the epilogue per element.
* ``loops`` (attention, conv and window plans): one thread per output element, every reduction a
  serial loop placed at the shallowest depth where its axes are bound, as the CPU backend does.
  Correct for every kernel the GPU plan accepts; not fast for large attention or convolution.

Arithmetic runs in float32 (Metal has no float64): loads widen float16 and bfloat16 to float, every
reduction accumulates in float, stores round to the output dtype. The source starts with
``#pragma METAL fp math_mode(safe)`` and calls the ``precise::`` math functions, NaN tests read the
bits, and division by zero, ``log``, ``sqrt``, ``rsqrt`` and ``max``/``min`` follow the reference's
NaN and infinity rules explicitly, so the result does not depend on the compiler's math mode.
Metal has no ``erf``; ``ts_erf`` is two fitted pieces (x P(x^2) to 1, 1 - exp(-x^2) Q(1/x) to 4).

Every reduction runs in a fixed order for a given kernel, shape and device, so results are bitwise
reproducible run to run (no atomics anywhere).

Kernels are compiled at run time with ``torch.mps.compile_shader`` and launched on the MPS tensors
themselves, in torch's MPS command stream: no copy, no extra package. See docs/METAL.md.
"""

from __future__ import annotations

import math
import os
import re
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from . import ir
from .ir import (Axis, Binary, Const, Dim, Expr, Index, Kernel, Load, Reduce, Strided, Unary, Where,
                 free_axes)

GROUP = 256          # threads per group of the pointwise and loops shapes, at most for rows
MM_THREADS = 128     # threads per contraction group: 4 SIMD groups of 32
BM = BN = 32         # contraction output tile
BK = 16              # contraction K slab staged per step
MAX_THREADS = 2 ** 31 - 1   # grid and offsets are 32 bit
SERIAL_MAX = 64      # static reductions up to this length run serially per output element
ATT_GROUP = 64       # threads per attention group: one query row, one key per thread per chunk
ATT_MAX_D = 256      # attention head dims up to this run in the attention shape (static only)
ATT_MAX_TERMS = 4    # softmax terms per attention kernel the attention shape takes
ATT_TILE_MEM = 28 * 1024  # threadgroup bytes the SIMD group attention shape may use (Metal allows 32 KB)

MSL_TYPE = {"float32": "float", "float16": "half", "bfloat16": "bfloat"}
FLT_MAX = 3.4028234663852886e38


class MetalError(RuntimeError):
    code = "TS843"


class MetalUnavailable(MetalError):
    """TS840: the Metal backend was asked for on a machine that cannot run it (not macOS, no MPS
    device, or a torch without ``torch.mps.compile_shader``). Nothing falls back to another device."""
    code = "TS840"


class MetalUnsupported(ValueError):
    """TS841, TS842, TS845: a kernel or option the Metal backend refuses before any launch."""

    def __init__(self, code: str, msg: str):
        super().__init__(f"[{code}] {msg}")
        self.code = code


class MetalCompileError(MetalError):
    """TS843: the Metal compiler rejected generated source (a Truestar bug, never the user's)."""
    code = "TS843"


# =========================================================================== availability

_lock = threading.Lock()
_simd_state: Dict[str, bool] = {}


def unavailable_reason() -> Optional[str]:
    """Why Metal kernels cannot run here, or None when they can."""
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return "no MPS device (Metal kernels need macOS on Apple silicon with a torch built with MPS)"
    if not hasattr(getattr(torch, "mps", None), "compile_shader"):
        return (f"torch {torch.__version__} has no torch.mps.compile_shader, which compiles the "
                f"generated Metal source; use torch 2.6 or later")
    return None


def available() -> bool:
    return unavailable_reason() is None


def require(what: str = "the Metal backend") -> None:
    why = unavailable_reason()
    if why is not None:
        raise MetalUnavailable(f"[TS840] {what} cannot run on this machine: {why}. It never falls "
                               f"back to another device; use 'cpu' or 'cuda' instead.")


_SIMD_PROBE = """#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void probe(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                  device float* c [[buffer(2)]], uint3 lt [[thread_position_in_threadgroup]]) {
    simdgroup_float8x8 x;
    simdgroup_float8x8 y;
    simdgroup_float8x8 z = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_load(x, a, 8);
    simdgroup_load(y, b, 8);
    simdgroup_multiply_accumulate(z, x, y, z);
    simdgroup_store(z, c, 8);
}
"""


def simdgroup_probe() -> Tuple[bool, str]:
    """(usable, reason): compile and run one 8 x 8 SIMD group matrix product and compare it with
    float64. Cached per process."""
    with _lock:
        if "ok" in _simd_state:
            return _simd_state["ok"], _simd_state.get("why", "")
        ok, why = _run_simd_probe()
        _simd_state["ok"], _simd_state["why"] = ok, why
        return ok, why


def _run_simd_probe() -> Tuple[bool, str]:
    import torch
    try:
        lib = torch.mps.compile_shader(_SIMD_PROBE)
        g = torch.Generator().manual_seed(7)
        a = torch.randn(8, 8, generator=g)
        b = torch.randn(8, 8, generator=g)
        c = torch.zeros(8, 8, device="mps")
        lib.probe(a.to("mps"), b.to("mps"), c, threads=[32, 1, 1], group_size=[32, 1, 1])
        got = c.cpu().double()
    except Exception as e:  # compiler or runtime refusal
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    want = a.double() @ b.double()
    err = float((got - want).abs().max() / want.abs().max())
    if not err < 1e-5:
        return False, f"8 x 8 product off by {err:.2e} relative"
    return True, ""


def simdgroup_enabled() -> bool:
    """Whether contraction kernels use SIMD group matrices: TRUESTAR_METAL_SIMDGROUP=0 or 1 forces
    the choice, otherwise the probe decides (TS844 notes a device where it says no)."""
    env = os.environ.get("TRUESTAR_METAL_SIMDGROUP", "").strip().lower()
    if env in ("0", "off", "false", "no"):
        return False
    if env in ("1", "on", "true", "yes"):
        return True
    ok, why = simdgroup_probe()
    if not ok and not _simd_state.get("warned"):
        _simd_state["warned"] = True
        import warnings
        warnings.warn(f"[TS844] SIMD group matrices are not usable on this Metal device ({why}); contraction "
                      f"kernels use threadgroup tiles with plain multiply adds", RuntimeWarning, stacklevel=2)
    return ok


# =========================================================================== plans

@dataclass
class MetalPlan:
    kind: str                 # reported shape: "tile", "contraction" or "loops"
    shape: str                # Metal loop shape: pointwise, rows, contraction, loops
    base: object              # structural plan from triton_backend.plan
    group: int                # threads per threadgroup
    simdgroup: bool = False   # contraction: SIMD group matrices
    tile: Optional[Tuple[int, int]] = None   # attention_simd: (query rows, keys) per tile
    via: str = ""             # loops: the structural shape the kernel had (attention, conv, window)

    def config(self) -> Dict:
        c = {"shape": self.shape, "group": self.group}
        if self.shape == "contraction":
            t, bk = self.tile if self.tile else (BM, BK)
            c.update(BM=t, BN=t, BK=bk, simdgroup=self.simdgroup)
        if self.tile is not None and self.shape != "contraction":
            c.update(BQ=self.tile[0], BKV=self.tile[1])
        return c


def _pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


def _rows_group(rsize) -> int:
    if isinstance(rsize, int):
        return max(32, min(GROUP, _pow2(rsize)))
    return GROUP


def plan(k: Kernel, simdgroup: bool = True, attention: str = "auto", mm: str = "") -> MetalPlan:
    """The Metal loop shape for kernel k. Raises triton_backend.Unschedulable (the fusion pass then
    splits the kernel, exactly as for CUDA) or MetalUnsupported."""
    for b in k.inputs:
        if b.dtype == "float64":
            raise MetalUnsupported("TS841", f"kernel {k.name}: input {b.name} is float64; Metal has no "
                                            f"float64 arithmetic (use float32, float16 or bfloat16)")
    for o in k.outputs:
        if o.dtype == "float64":
            raise MetalUnsupported("TS841", f"kernel {k.name}: output {o.name} is float64; Metal has no "
                                            f"float64 arithmetic (use float32, float16 or bfloat16)")
    from . import triton_backend as tb   # IR analysis only; triton itself is not imported
    base = tb.plan(k)
    if base.kind == "tile":
        if not base.reduces:
            return MetalPlan("tile", "pointwise", base, GROUP)
        if base.lane is None and isinstance(base.rsize, int) and base.rsize <= SERIAL_MAX:
            # one value per output element over a short static reduction (pooling windows): a
            # thread per element beats a threadgroup per element
            return MetalPlan("tile", "loops", base, GROUP, via="tile")
        return MetalPlan("tile", "rows", base, _rows_group(base.rsize))
    if base.kind == "contraction":
        if getattr(base, "fold", ()) and not (len(base.fold) == 1 and not base.batch and _fold_on()):
            # the CUDA plan folds batch axes the right factor does not depend on into the rows;
            # the Metal contraction keeps every batch axis on the grid's z dimension
            import dataclasses
            o = k.outputs[0]
            keep = set(base.batch) | set(base.fold)
            base = dataclasses.replace(base, batch=tuple(x for x in o.axes if x in keep), fold=())
        t, bk = BM, BK
        if simdgroup and mm:
            a, _, b = mm.partition("x")
            if a.isdigit() and b.isdigit() and int(a) in (32, 64) and int(b) in (16, 32):
                t, bk = int(a), int(b)
        return MetalPlan("contraction", "contraction", base, MM_THREADS, simdgroup=bool(simdgroup), tile=(t, bk))
    if base.kind == "attention" and attention and attention != "loops" and _attention_ok(base):
        mode, _, bq = attention.partition(":")
        sizes = (int(bq),) if bq.isdigit() and int(bq) in (8, 16, 32) else None
        few = isinstance(base.i.size, int) and base.i.size < 8 and mode == "auto"
        if mode in ("auto", "simd") and simdgroup and not few:
            st = _att_simd_tile(base, sizes)
            if st is not None:
                return MetalPlan("attention", "attention_simd", base, 32 * (st[0] // 8), simdgroup=True, tile=st)
        g = int(attention[1:]) if attention.startswith("g") else ATT_GROUP
        return MetalPlan("attention", "attention", base, g)
    return MetalPlan("loops", "loops", base, GROUP, via=base.kind)


ATT_GROUPS = (32, 64, 128, 256)


ATT_MODES = ("auto", "simd", "rows", "loops")


def mm_option() -> str:
    """Contraction tile and K slab for measurements ("64x16" from TRUESTAR_METAL_MM), part of the
    cache key; "" is the default."""
    return os.environ.get("TRUESTAR_METAL_MM", "").strip()


def attention_option() -> str:
    """The attention choice of this process, a part of every Metal kernel's cache key. Modes
    (TRUESTAR_METAL_ATTENTION): "auto" (default: SIMD group matrices where the kernel and device
    allow and the query axis is not a handful of rows, else one query row per group), "simd",
    "rows" (one query row per group) and "loops" (the serial loops shape). Each falls back to rows,
    then loops, where it does not apply. TRUESTAR_METAL_ATT_GROUP picks the threads of rows and
    TRUESTAR_METAL_ATT_TILE the query rows of simd (8, 16 or 32), for measurements."""
    mode = os.environ.get("TRUESTAR_METAL_ATTENTION", "").strip().lower()
    g = os.environ.get("TRUESTAR_METAL_ATT_GROUP", "").strip()
    bq = os.environ.get("TRUESTAR_METAL_ATT_TILE", "").strip()
    if mode not in ATT_MODES:
        mode = "auto"
    if mode != "loops" and g.isdigit() and int(g) in ATT_GROUPS:
        return f"g{int(g)}"
    if mode in ("auto", "simd") and bq in ("8", "16", "32"):
        return f"{mode}:{bq}"
    return mode


def _att_simd_tile(ap, sizes=None) -> Optional[Tuple[int, int]]:
    """(query rows, keys) per group of the SIMD group matrix attention shape: one softmax term, the
    contraction and head dims static multiples of 8; query tile = key chunk, 8 rows per SIMD group."""
    if len(ap.terms) != 1:
        return None
    C, D = ap.terms[0].c.size, ap.d.size
    if not (isinstance(C, int) and isinstance(D, int) and C % 8 == 0 and D % 8 == 0 and 0 < C <= 256):
        return None
    if sizes is None:   # measured on an M2: 32 rows for head dims to 32, 16 from 64 (more groups per core)
        sizes = (32, 16, 8) if C + D < 128 else (16, 8)
    for B in sizes:
        words = B * (C + D) + B * B + B
        if 4 * words <= ATT_TILE_MEM:      # Q and the output are staged through the K and V area
            return B, B
    return None


def _attention_ok(ap) -> bool:
    """The attention shape takes a static head dim up to ATT_MAX_D and up to ATT_MAX_TERMS softmax
    terms; other attention kernels run in the loops shape."""
    return isinstance(ap.d.size, int) and 0 < ap.d.size <= ATT_MAX_D and 0 < len(ap.terms) <= ATT_MAX_TERMS


# =========================================================================== code generation

HEADER = """#include <metal_stdlib>
using namespace metal;
#pragma METAL fp math_mode(safe)

static inline bool ts_isnan(float x) { return (as_type<uint>(x) & 0x7fffffffu) > 0x7f800000u; }
static inline float ts_max(float a, float b) { return ts_isnan(a) ? a : (ts_isnan(b) ? b : (a >= b ? a : b)); }
static inline float ts_min(float a, float b) { return ts_isnan(a) ? a : (ts_isnan(b) ? b : (a <= b ? a : b)); }
static inline float ts_div(float a, float b) {
    if (b != 0.0f) return a / b;
    if (ts_isnan(a) || ts_isnan(b) || a == 0.0f) return NAN;
    return ((a > 0.0f) == !signbit(b)) ? INFINITY : -INFINITY;
}
static inline float ts_log(float x) { return x > 0.0f ? precise::log(x) : (x == 0.0f ? -INFINITY : NAN); }
static inline float ts_sqrt(float x) { return x >= 0.0f ? precise::sqrt(x) : NAN; }
static inline float ts_rsqrt(float x) {   // 1 / sqrt(x) as the reference: -0 gives -inf
    return x > 0.0f ? precise::rsqrt(x) : (x == 0.0f ? (signbit(x) ? -INFINITY : INFINITY) : NAN);
}
static inline float ts_sigmoid(float x) {
    if (ts_isnan(x)) return x;
    return x >= 0.0f ? 1.0f / (1.0f + precise::exp(-x)) : precise::exp(x) / (1.0f + precise::exp(x));
}
static inline float ts_erf(float x) {
    // piecewise fits written for Truestar (least squares on Chebyshev nodes, float64, then checked
    // in float32): x P(x^2) for |x| <= 1, 1 - exp(-x^2) Q(1/x) for 1 < |x| < 4 (Q fits
    // exp(x^2) erfc(x)), 1 beyond. Worst relative error against float64 erf 1.9e-7 on a dense grid,
    // every float32 in [-4, 4] checked on the M2 (docs/METAL.md).
    if (ts_isnan(x)) return x;
    float ax = fabs(x);
    float r;
    if (ax <= 1.0f) {
        const float z = ax * ax;
        const float p = (((((((((-0x1.c3d8d20000000p-24f * z + 0x1.9b83b00000000p-20f) * z + -0x1.f097de0000000p-17f) * z + 0x1.f946f20000000p-14f) * z + -0x1.c028d40000000p-11f) * z + 0x1.565ba40000000p-8f) * z + -0x1.b82ce20000000p-6f) * z + 0x1.ce2f220000000p-4f) * z + -0x1.8127460000000p-2f) * z + 0x1.20dd760000000p+0f);
        r = ax * p;
    } else if (ax < 4.0f) {
        const float t = 1.0f / ax;
        const float q = ((((((((((((((((-0x1.ebea0e0000000p-14f * t + 0x1.19a84a0000000p-6f) * t + -0x1.f3a2800000000p-4f) * t + 0x1.7387de0000000p-2f) * t + -0x1.098e8a0000000p-1f) * t + 0x1.6a2a740000000p-3f) * t + 0x1.1fa0f60000000p-1f) * t + -0x1.9f5b700000000p-1f) * t + -0x1.4ed45e0000000p-5f) * t + 0x1.5438400000000p+0f) * t + -0x1.d2ae2e0000000p+0f) * t + 0x1.30bcc40000000p+0f) * t + -0x1.b167f60000000p-3f) * t + -0x1.f0e77a0000000p-3f) * t + -0x1.3ccc460000000p-8f) * t + 0x1.210bb60000000p-1f) * t + -0x1.878de20000000p-17f);
        r = 1.0f - precise::exp(-(ax * ax)) * q;
    } else {
        r = 1.0f;   // erfc(4) is 1.5e-8, below half a float32 ulp of 1
    }
    return x < 0.0f ? -r : r;
}
"""

OSUM_HEADER = """
// order free sums (docs/ORDER_FREE_SUM.md): every step on integers, so the result does not depend
// on the device's float conversions or subnormal handling
static inline int ts_os_e(float x) {        // the term's exponent field for the row anchor
    int e = (int)((as_type<uint>(x) >> 23) & 255u);
    return e == 255 ? 1 : max(e, 1);
}
static inline long ts_os_q(float x, int emax) {   // round_half_even(x / 2**L), L = emax - 165
    uint b = as_type<uint>(x);
    int e = (int)((b >> 23) & 255u);
    if (e == 255) return 0;
    long m = (long)(b & 0x7fffffu);
    if (e != 0) m |= 0x800000l;
    int sh = max(e, 1) - emax + 15;
    long q;
    if (sh >= 0) {
        q = m << sh;
    } else {
        int r = -sh;
        if (r >= 26) {
            q = 0;
        } else {
            long t = m >> r;
            long rem = m & ((1l << r) - 1);
            long hf = 1l << (r - 1);
            q = (rem > hf || (rem == hf && (t & 1) != 0)) ? t + 1 : t;
        }
    }
    return (b >> 31) != 0u ? -q : q;
}
static inline int ts_os_flag(float x) {     // 4: a NaN, 2: a +inf, 1: a -inf
    uint b = as_type<uint>(x);
    if (((b >> 23) & 255u) != 255u) return 0;
    return (b & 0x7fffffu) != 0u ? 4 : ((b >> 31) != 0u ? 1 : 2);
}
static inline ulong ts_os_rne(ulong m, int r) {   // m / 2**r rounded half to even, r >= 1
    if (r >= 64) return 0;
    ulong t = m >> r;
    ulong rem = m & ((1ul << r) - 1ul);
    ulong hf = 1ul << (r - 1);
    return (rem > hf || (rem == hf && (t & 1ul) != 0ul)) ? t + 1ul : t;
}
static inline float ts_os_finish(long acc, int emax, int flags) {
    // the int64 total rounded to float32 (to nearest even), then scaled by 2**L, rounded again
    // only where the result is subnormal: the definition's two IEEE steps, done on the bits
    float f = 0.0f;
    if (acc != 0) {
        uint sign = acc < 0 ? 0x80000000u : 0u;
        ulong a = acc < 0 ? (ulong)(-acc) : (ulong)acc;
        int nb = 64 - (int)clz(a);
        ulong m;
        int E = emax - 165;
        if (nb > 24) {
            m = ts_os_rne(a, nb - 24);
            E += nb - 24;
            if (m == (1ul << 24)) { m >>= 1; E += 1; }
        } else {
            m = a << (24 - nb);
            E -= 24 - nb;
        }
        int eb = E + 150;
        uint bits;
        if (eb >= 255) {
            bits = 0x7f800000u;
        } else if (eb >= 1) {
            bits = ((uint)eb << 23) | (uint)(m - (1ul << 23));
        } else {
            bits = (uint)ts_os_rne(m, 1 - eb);
        }
        f = as_type<float>(sign | bits);
    }
    if ((flags & 1) != 0) f = -INFINITY;
    if ((flags & 2) != 0) f = INFINITY;
    if ((flags & 4) != 0 || (flags & 3) == 3) f = NAN;
    return f;
}
"""

RO_OPS = ("exp", "log", "tanh", "erf", "rsqrt", "sigmoid", "sqrt")   # roundonce.FUNCTIONS and sqrt

_UNARY = {
    "neg": "(-{0})", "exp": "precise::exp({0})", "log": "ts_log({0})", "sqrt": "ts_sqrt({0})",
    "rsqrt": "ts_rsqrt({0})", "sigmoid": "ts_sigmoid({0})", "tanh": "precise::tanh({0})",
    "abs": "fabs({0})", "not": "(!{0})", "erf": "ts_erf({0})", "cos": "precise::cos({0})",
    "sin": "precise::sin({0})",
}
_BINARY_SYM = {"add": "+", "sub": "-", "mul": "*", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
               "eq": "==", "ne": "!=", "and": "&&", "or": "||"}
_INIT = {"sum": "0.0f", "max": "(-INFINITY)", "min": "INFINITY"}


def _lit(v) -> str:
    """A float32 (or bool) literal for a constant of the IR. Values beyond the float32 range
    saturate to its largest finite number, as the runtime does with float64 masks."""
    if isinstance(v, bool):
        return "true" if v else "false"
    v = float(v)
    if math.isnan(v):
        return "NAN"
    if math.isinf(v):
        return "INFINITY" if v > 0 else "(-INFINITY)"
    if abs(v) > FLT_MAX:
        v = math.copysign(FLT_MAX, v)
    import numpy as np
    f = float(np.float32(v))
    if math.isinf(f):            # rounds up past the largest float32
        f = math.copysign(FLT_MAX, v)
    s = repr(f)
    if "e" not in s and "." not in s and "inf" not in s:
        s += ".0"
    return f"({s}f)" if f < 0 or s.startswith("-") else f"{s}f"


def _combine(op: str, acc: str, v: str) -> str:
    if op == "sum":
        return f"({acc} + {v})"
    return f"ts_max({acc}, {v})" if op == "max" else f"ts_min({acc}, {v})"


def _cname(name: str) -> str:
    return re.sub(r"\W", "_", name)


def param_layout(k: Kernel) -> List[Tuple]:
    """Order of the int32 parameter buffer: symbols (by name), then the strides of every input
    (by name, each dim), then the strides of every output (in order, each dim)."""
    out: List[Tuple] = [("sym", s.name) for s in sorted(k.symbols(), key=lambda s: s.name)]
    for b in sorted(k.inputs, key=lambda b: b.name):
        out += [("in", b.name, d) for d in range(len(b.shape))]
    for o in k.outputs:
        out += [("out", o.name, d) for d in range(len(o.axes))]
    return out


def _pending(roots: Sequence[Expr], bound: Set[Axis], values: Dict) -> List[Reduce]:
    """Reductions the roots need that can be computed with the bound axes and are not computed
    yet, nested ones first; reductions that need a deeper axis are left for inside its loop."""
    out: List[Reduce] = []
    seen = set()

    def go(n):
        if n in seen or n in values:
            return
        seen.add(n)
        if isinstance(n, Reduce):
            go(n.body)
            if free_axes(n) <= bound:
                out.append(n)
            return
        for c in n.children():
            go(c)

    for r in roots:
        go(r)
    return out


class _Em:
    def __init__(self, k: Kernel, opts: Dict):
        self.k = k
        self.opts = opts
        self.lines: List[str] = []
        self.ind = 1
        self.n = 0
        self.var: Dict[Axis, str] = {}
        self.values: Dict[Expr, str] = {}
        self.brk = opts.get("_break", {}) or {}
        self.round_once = opts.get("math") == "round_once"

    def line(self, s: str):
        self.lines.append("    " * self.ind + s)

    def open(self, s: str):
        self.line(s)
        self.ind += 1

    def close(self, s: str = "}"):
        self.ind -= 1
        self.line(s)

    def fresh(self, p: str) -> str:
        self.n += 1
        return f"{p}{self.n}"

    @staticmethod
    def size(s) -> str:
        return str(s) if isinstance(s, int) else f"S_{_cname(s.name)}"

    @staticmethod
    def in_stride(b, d) -> str:
        return f"st_{_cname(b)}_{d}"

    @staticmethod
    def out_stride(o, d) -> str:
        return f"ost_{_cname(o)}_{d}"

    # ------------------------------------------------------------------ expressions

    def ex(self, e: Expr) -> str:
        v = self.values.get(e)
        if v is not None:
            return v
        if isinstance(e, Const):
            return _lit(e.value)
        if isinstance(e, Dim):
            return f"float({self.size(e.size)})"
        if isinstance(e, Index):
            return f"float({self.var[e.axis]})"
        if isinstance(e, Load):
            code = self.load(e)
        elif isinstance(e, Unary):
            x = self.ex(e.x)
            if e.op in self.brk:
                code = self.brk[e.op](x)
            elif self.round_once and e.op in RO_OPS:
                code = f"_ro_{e.op}({x})"      # round once math: float64 on the bits, rounded once
            else:
                code = _UNARY[e.op].format(x)
        elif isinstance(e, Binary):
            a, b = self.ex(e.a), self.ex(e.b)
            if e.op in self.brk:
                code = self.brk[e.op](a, b)
            elif e.op in _BINARY_SYM:
                code = f"({a} {_BINARY_SYM[e.op]} {b})"
            elif e.op == "div":
                code = f"ts_div({a}, {b})"
            elif e.op == "maximum":
                code = f"ts_max({a}, {b})"
            elif e.op == "minimum":
                code = f"ts_min({a}, {b})"
            else:
                raise ir.IRError(f"metal backend cannot emit binary op {e.op}")
        elif isinstance(e, Where):
            c, a, b = self.ex(e.cond), self.ex(e.a), self.ex(e.b)
            code = f"({c} ? {a} : {b})"
        elif isinstance(e, Reduce):
            raise ir.IRError("internal: reduction used before it was computed")
        else:
            raise ir.IRError(f"metal backend cannot emit {type(e).__name__}")
        t = self.fresh("t")
        self.line(f"const auto {t} = {code};")
        self.values[e] = t
        return t

    def strided(self, a: Strided, size) -> Tuple[str, List[str]]:
        """(index, validity conditions) of a strided index; axes are never negative, so C integer
        division and remainder of a term mean the same as the reference's floor rules."""
        parts = []
        for t in a.terms:
            v = self.var[t.axis] if t.mul == 1 else f"({self.var[t.axis]} * {t.mul})"
            if t.div != 1:
                v = f"({v} / {t.div})"
            if t.mod:
                v = f"({v} % {t.mod})"
            parts.append(v if t.coeff == 1 else f"({t.coeff} * {v})")
        num = "(" + " + ".join(parts + [str(a.offset)]) + ")"
        conds = []
        if a.exact != 1:
            conds += [f"{num} >= 0", f"{num} % {a.exact} == 0"]
            num = f"({num} / {a.exact})"
        if not a.always_valid(size):
            conds += [f"{num} >= 0", f"{num} < {self.size(size)}"]
        return num, conds

    def load(self, e: Load) -> str:
        b = e.buffer
        terms, conds = [], []
        for d, a in enumerate(e.index):
            if isinstance(a, Axis):
                idx = self.var[a]
            elif isinstance(a, int):
                if a == 0:
                    continue
                idx = str(a)
            else:
                idx, c = self.strided(a, b.shape[d])
                conds += c
            terms.append(f"{idx} * {self.in_stride(b.name, d)}")
        off = " + ".join(terms) if terms else "0"
        val = f"float({_cname(b.name)}[{off}])"
        if conds and e.pad is not None:
            return f"(({' && '.join(conds)}) ? {val} : {_lit(e.pad)})"
        return val

    # ------------------------------------------------------------------ reductions

    def serial_reduce(self, r: Reduce, bound: Set[Axis]):
        if r.op == "osum":
            return self.serial_osum(r, bound)
        acc = self.fresh("acc")
        self.line(f"float {acc} = {_INIT[r.op]};")
        lv = self.fresh("r")
        self.var[r.axis] = lv
        self.open(f"for (int {lv} = 0; {lv} < {self.size(r.axis.size)}; ++{lv}) {{")
        saved = dict(self.values)
        inner = bound | {r.axis}
        for sub in _pending([r.body], inner, self.values):
            self.serial_reduce(sub, inner)
        v = self.ex(r.body)
        self.line(f"{acc} = {_combine(r.op, acc, v)};")
        self.values = saved
        self.close()
        self.values[r] = acc

    def _osum_pass(self, r: Reduce, bound: Set[Axis], head: str, step):
        lv = self.fresh("r")
        self.var[r.axis] = lv
        self.open(head.format(lv=lv))
        saved = dict(self.values)
        inner = bound | {r.axis}
        for sub in _pending([r.body], inner, self.values):
            self.serial_reduce(sub, inner)
        v = self.fresh("ov")
        self.line(f"const float {v} = float({self.ex(r.body)});")
        step(v)
        self.values = saved
        self.close()

    def serial_osum(self, r: Reduce, bound: Set[Axis]):
        """Order free sum by one thread: the anchor in a first pass, the int64 total in a second."""
        em, tot, fl = self.fresh("oe"), self.fresh("ot"), self.fresh("of")
        self.line(f"int {em} = 1;")
        self.line(f"long {tot} = 0;")
        self.line(f"int {fl} = 0;")
        n = self.size(r.axis.size)
        head = "for (int {lv} = 0; {lv} < " + str(n) + "; ++{lv}) {{"
        self._osum_pass(r, bound, head, lambda v: self.line(f"{em} = max({em}, ts_os_e({v}));"))
        self._osum_pass(r, bound, head, lambda v: (self.line(f"{tot} = {tot} + ts_os_q({v}, {em});"),
                                                   self.line(f"{fl} = {fl} | ts_os_flag({v});")))
        acc = self.fresh("acc")
        self.line(f"const float {acc} = ts_os_finish({tot}, {em}, {fl});")
        self.values[r] = acc

    def group_osum(self, r: Reduce, bound: Set[Axis], G: int):
        """Order free sum by the threadgroup: the anchor is a max and the total an int64 sum, both
        exact, so the threads' partials may be combined in any tree."""
        em, tot, fl = self.fresh("oe"), self.fresh("ot"), self.fresh("of")
        self.line(f"int {em} = 1;")
        self.line(f"long {tot} = 0;")
        self.line(f"int {fl} = 0;")
        n = self.size(r.axis.size)
        head = "for (int {lv} = ts_lid; {lv} < " + str(n) + "; {lv} += " + str(G) + ") {{"
        self._osum_pass(r, bound, head, lambda v: self.line(f"{em} = max({em}, ts_os_e({v}));"))
        for var, arr, comb in ((em, "ts_shi", "max({a}, {b})"),):
            self._int_tree(var, arr, comb, G)
        self._osum_pass(r, bound, head, lambda v: (self.line(f"{tot} = {tot} + ts_os_q({v}, {em});"),
                                                   self.line(f"{fl} = {fl} | ts_os_flag({v});")))
        self._int_tree(tot, "ts_shl", "{a} + {b}", G)
        self._int_tree(fl, "ts_shi", "{a} | {b}", G)
        acc = self.fresh("acc")
        self.line(f"const float {acc} = ts_os_finish({tot}, {em}, {fl});")
        self.values[r] = acc

    def _int_tree(self, var: str, arr: str, comb: str, G: int):
        self.line(f"{arr}[ts_lid] = {var};")
        self.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
        self.open(f"for (int ts_s = {G // 2}; ts_s > 0; ts_s >>= 1) {{")
        self.line(f"if (ts_lid < ts_s) {arr}[ts_lid] = " + comb.format(a=f"{arr}[ts_lid]", b=f"{arr}[ts_lid + ts_s]") + ";")
        self.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
        self.close()
        self.line(f"{var} = {arr}[0];")
        self.line("threadgroup_barrier(mem_flags::mem_threadgroup);")

    def group_reduce(self, r: Reduce, bound: Set[Axis], G: int):
        if r.op == "osum":
            return self.group_osum(r, bound, G)
        """One reduction by the whole threadgroup: thread t folds elements t, t + G, ... in order,
        then a fixed tree in threadgroup memory. Every thread gets the result."""
        acc = self.fresh("red")
        self.line(f"float {acc};")
        self.open("{")
        part = self.fresh("p")
        self.line(f"float {part} = {_INIT[r.op]};")
        lv = self.fresh("r")
        self.var[r.axis] = lv
        self.open(f"for (int {lv} = ts_lid; {lv} < {self.size(r.axis.size)}; {lv} += {G}) {{")
        saved = dict(self.values)
        inner = bound | {r.axis}
        for sub in _pending([r.body], inner, self.values):
            self.serial_reduce(sub, inner)
        v = self.ex(r.body)
        self.line(f"{part} = {_combine(r.op, part, v)};")
        self.values = saved
        self.close()
        self.line(f"ts_sh[ts_lid] = {part};")
        self.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
        self.open(f"for (int ts_s = {G // 2}; ts_s > 0; ts_s >>= 1) {{")
        self.line(f"if (ts_lid < ts_s) ts_sh[ts_lid] = {_combine(r.op, 'ts_sh[ts_lid]', 'ts_sh[ts_lid + ts_s]')};")
        self.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
        self.close()
        self.line(f"{acc} = ts_sh[0];")
        self.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
        self.close()
        self.values[r] = acc

    # ------------------------------------------------------------------ pieces

    def decompose(self, axes: Sequence[Axis], flat: str):
        """Axis variables from a row major flat index."""
        if not axes:
            return
        g = self.fresh("g")
        self.line(f"int {g} = {flat};")
        for d, a in enumerate(reversed(axes)):
            v = self.fresh("i")
            self.var[a] = v
            if d == len(axes) - 1:
                self.line(f"const int {v} = {g};")
            else:
                self.line(f"const int {v} = {g} % {self.size(a.size)};")
                self.line(f"{g} = {g} / {self.size(a.size)};")

    def store(self, o: ir.Output, v: str):
        terms = [f"{self.var[a]} * {self.out_stride(o.name, d)}" for d, a in enumerate(o.axes)]
        off = " + ".join(terms) if terms else "0"
        self.line(f"{_cname(o.name)}[{off}] = {MSL_TYPE[o.dtype]}(float({v}));")

    def prelude(self):
        for i, p in enumerate(param_layout(self.k)):
            if p[0] == "sym":
                self.line(f"const int S_{_cname(p[1])} = ts_P[{i}];")
            elif p[0] == "in":
                self.line(f"const int {self.in_stride(p[1], p[2])} = ts_P[{i}];")
            else:
                self.line(f"const int {self.out_stride(p[1], p[2])} = ts_P[{i}];")

    def product(self, axes: Sequence[Axis]) -> str:
        return " * ".join(self.size(a.size) for a in axes) if axes else "1"


def _signature(k: Kernel, extra: Sequence[str]) -> str:
    args, i = [], 0
    for b in sorted(k.inputs, key=lambda b: b.name):
        args.append(f"device const {MSL_TYPE[b.dtype]}* {_cname(b.name)} [[buffer({i})]]")
        i += 1
    for o in k.outputs:
        args.append(f"device {MSL_TYPE[o.dtype]}* {_cname(o.name)} [[buffer({i})]]")
        i += 1
    args.append(f"device const int* ts_P [[buffer({i})]]")
    args += list(extra)
    return "kernel void kern(\n    " + ",\n    ".join(args) + ")"


def _gen_elementwise(k: Kernel, p: MetalPlan, em: _Em):
    """pointwise and loops: one thread per output element."""
    axes = k.outputs[0].axes
    em.line(f"const int ts_total = {em.product(axes)};")
    em.line("if ((int)ts_gid.x >= ts_total) return;")
    em.decompose(axes, "(int)ts_gid.x")
    bound = set(axes)
    exprs = [o.expr for o in k.outputs]
    for r in _pending(exprs, bound, em.values):
        em.serial_reduce(r, bound)
    for o in k.outputs:
        em.store(o, em.ex(o.expr))


def _gen_rows(k: Kernel, p: MetalPlan, em: _Em):
    tp = p.base
    G = p.group
    em.line("const int ts_lid = (int)ts_lt.x;")
    em.line(f"threadgroup float ts_sh[{G}];")
    if any(r.op == "osum" for r in k.reduces()):
        em.line(f"threadgroup int ts_shi[{G}];")
        em.line(f"threadgroup long ts_shl[{G}];")
    em.decompose(tp.rows, "(int)ts_tg.x")
    bound = set(tp.rows)
    for r in tp.reduces:                     # dependency order, inner first
        if r not in em.values:
            em.group_reduce(r, bound, G)
    exprs = [o.expr for o in k.outputs]
    if tp.lane is not None:
        lv = em.fresh("i")
        em.var[tp.lane] = lv
        em.open(f"for (int {lv} = ts_lid; {lv} < {em.size(tp.lane.size)}; {lv} += {G}) {{")
        saved = dict(em.values)
        inner = bound | {tp.lane}
        for r in _pending(exprs, inner, em.values):
            em.serial_reduce(r, inner)
        for o in k.outputs:
            em.store(o, em.ex(o.expr))
        em.values = saved
        em.close()
    else:
        em.open("if (ts_lid == 0) {")
        for o in k.outputs:
            em.store(o, em.ex(o.expr))
        em.close()


def _fold_on() -> bool:
    return os.environ.get("TRUESTAR_METAL_MM_FOLD", "1") != "0"


def _plain_load(em, expr, row_axis, col_axis, fold=None):
    """(buffer, dim of row, dim of col, other dims) when expr is a plain load whose last dim is
    col_axis (tile rows contiguous in memory) and whose other axes are bound; else None."""
    if not isinstance(expr, Load) or expr.pad is not None:
        return None
    idx = expr.index
    if not all(isinstance(a, Axis) for a in idx) or idx.count(row_axis) != 1 or idx.count(col_axis) != 1:
        return None
    if idx[-1] != col_axis:
        return None
    if any(a not in em.var for a in idx if a not in (row_axis, col_axis, fold)):
        return None
    if fold is not None and idx.count(fold) != 1:
        return None
    return expr.buffer, idx.index(row_axis), len(idx) - 1, [(d, a) for d, a in enumerate(idx)
                                                            if a not in (row_axis, col_axis, fold)]


def _fold_uniform(em, info, expr, fold):
    """Runtime condition that a folded row index r = f * M + m addresses the buffer as r * (stride
    of m), so the merged rows are evenly strided ("true" without a fold)."""
    if fold is None:
        return "true"
    b, rd = info[0], info[1]
    fd = expr.index.index(fold)
    return f"({em.in_stride(b.name, fd)} == ts_Mm * {em.in_stride(b.name, rd)})"


def _mm_direct(k, p, em, BM, BN, FR):
    """Open the direct path of a SIMD group contraction: on a tile inside the output whose float32
    operands are plain loads and whose K is a multiple of 8, every SIMD group loads its 8 x 8 blocks
    straight from device memory (no threadgroup staging, no barrier in the K loop). The blocks and
    the multiply accumulate sequence (K in steps of 8, ascending) are the staged path's, so the bits
    are the same. Emits `if (direct) { ... } else {` and returns True, or emits nothing and returns
    False when the operands do not qualify."""
    cp = p.base
    if not p.simdgroup or os.environ.get("TRUESTAR_METAL_MM_DIRECT", "1") == "0":
        return False
    fold = cp.fold[0] if getattr(cp, "fold", ()) else None
    ia = _plain_load(em, cp.a, cp.m, cp.k, fold)
    ibk, ibt = _plain_load(em, cp.b, cp.k, cp.n), _plain_load(em, cp.b, cp.n, cp.k)
    ib = ibk or ibt
    if ia is None or ib is None or ia[0].dtype != "float32" or ib[0].dtype != "float32":
        return False

    def base(info):
        b, _, _, others = info
        return " + ".join([f"{em.var[a]} * {em.in_stride(b.name, d)}" for d, a in others] or ["0"])
    ab, bb = em.fresh("da"), em.fresh("db")
    em.line(f"const int {ab} = {base(ia)};")
    em.line(f"const int {bb} = {base(ib)};")
    sa = em.in_stride(ia[0].name, ia[1])
    sb = em.in_stride(ib[0].name, ib[1])
    em.open(f"if (ts_m0 + {BM} <= ts_M && ts_n0 + {BN} <= ts_N && ts_K % 8 == 0 && {_fold_uniform(em, ia, cp.a, fold)}) {{")
    pa, pb = _cname(ia[0].name), _cname(ib[0].name)
    em.open("for (int ts_k0 = 0; ts_k0 < ts_K; ts_k0 += 8) {")
    for u in range(FR):
        em.line(f"simdgroup_float8x8 ts_a{u};")
        em.line(f"simdgroup_load(ts_a{u}, {pa} + {ab} + (ts_m0 + ts_sr + {8 * u}) * {sa} + ts_k0, (ulong){sa});")
    for v in range(FR):
        em.line(f"simdgroup_float8x8 ts_b{v};")
        if ibk is not None:
            em.line(f"simdgroup_load(ts_b{v}, {pb} + {bb} + ts_k0 * {sb} + ts_n0 + ts_sc + {8 * v}, (ulong){sb});")
        else:
            em.line(f"simdgroup_load(ts_b{v}, {pb} + {bb} + (ts_n0 + ts_sc + {8 * v}) * {sb} + ts_k0, (ulong){sb}, "
                    f"ulong2(0, 0), true);")
    for u in range(FR):
        for v in range(FR):
            c = f"ts_c{u * FR + v}"
            em.line(f"simdgroup_multiply_accumulate({c}, ts_a{u}, ts_b{v}, {c});")
    em.close()
    em.close()
    em.open("else {")
    return True


def _gen_contraction(k: Kernel, p: MetalPlan, em: _Em):
    cp = p.base
    o = k.outputs[0]
    T = MM_THREADS
    BM, BK = p.tile if p.tile else (globals()["BM"], globals()["BK"])
    BN = BM
    FR = BM // 16                        # 8 x 8 blocks per SIMD group and dim (2 x 2 SIMD groups)
    em.line("const int ts_lid = (int)ts_lt.x;")
    em.line(f"threadgroup float ts_A[{BM * BK}];")
    em.line(f"threadgroup float ts_B[{BK * BN}];")
    em.line(f"threadgroup float ts_C[{BM * BN}];")
    em.decompose(cp.batch, "(int)ts_tg.z")
    em.line(f"const int ts_m0 = (int)ts_tg.y * {BM};")
    em.line(f"const int ts_n0 = (int)ts_tg.x * {BN};")
    fold = cp.fold[0] if getattr(cp, "fold", ()) else None
    if fold is not None:     # rows are (folded batch entry, m), row major
        em.line(f"const int ts_Mm = {em.size(cp.m.size)};")
        em.line(f"const int ts_M = {em.size(fold.size)} * ts_Mm;")
    else:
        em.line(f"const int ts_M = {em.size(cp.m.size)};")
    em.line(f"const int ts_N = {em.size(cp.n.size)};")
    em.line(f"const int ts_K = {em.size(cp.k.size)};")

    def bind_row(r):
        if fold is None:
            em.var[cp.m] = r
            return
        f, m = em.fresh("rf"), em.fresh("rm")
        em.line(f"const int {f} = {r} / ts_Mm;")
        em.line(f"const int {m} = {r} % ts_Mm;")
        em.var[fold], em.var[cp.m] = f, m

    if p.simdgroup:
        for u in range(FR * FR):     # unrolled: matrices in registers, never indexed at run time
            em.line(f"simdgroup_float8x8 ts_c{u} = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);")
        em.line(f"const int ts_sr = (int)(ts_sg / 2) * {BM // 2};")
        em.line(f"const int ts_sc = (int)(ts_sg % 2) * {BN // 2};")
    else:
        em.line("float ts_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};")
        em.line("const int ts_tr = ts_lid / 4;")
        em.line("const int ts_tc = (ts_lid % 4) * 8;")
    direct = _mm_direct(k, p, em, BM, BN, FR)
    em.open(f"for (int ts_k0 = 0; ts_k0 < ts_K; ts_k0 += {BK}) {{")

    def stage(dst, n_elems, cols, row_base, col_base, row_lim, col_lim, expr, row_axis, col_axis):
        em.open(f"for (int ts_e = ts_lid; ts_e < {n_elems}; ts_e += {T}) {{")
        ri, ci = em.fresh("u"), em.fresh("u")
        em.line(f"const int {ri} = {row_base} + ts_e / {cols};")
        em.line(f"const int {ci} = {col_base} + ts_e % {cols};")
        v = em.fresh("v")
        em.line(f"float {v} = 0.0f;")
        em.open(f"if ({ri} < {row_lim} && {ci} < {col_lim}) {{")
        saved = dict(em.values)
        if row_axis == cp.m:
            bind_row(ri)
        else:
            em.var[row_axis] = ri
        em.var[col_axis] = ci
        x = em.ex(expr)
        em.line(f"{v} = float({x});")
        em.values = saved
        em.close()
        em.line(f"{dst}[ts_e] = {v};")
        em.close()

    def plain(expr, row_axis, col_axis):
        """(buffer, dim of row, dim of col, other dims) when expr is a plain load whose last dim is
        the tile's column axis, so a tile row is contiguous in memory; else None."""
        if not isinstance(expr, Load) or expr.pad is not None:
            return None
        idx = expr.index
        if not all(isinstance(a, Axis) for a in idx) or idx.count(row_axis) != 1 or idx.count(col_axis) != 1:
            return None
        if idx[-1] != col_axis:
            return None
        if any(a not in em.var for a in idx if a not in (row_axis, col_axis)):
            return None
        return expr.buffer, idx.index(row_axis), len(idx) - 1, [(d, a) for d, a in enumerate(idx)
                                                                if a not in (row_axis, col_axis)]

    def vstage(dst, rows, cols, row_base, col_base, row_lim, col_lim, info, fallback, uni="true"):
        """Stage a rows x cols tile of a plain load with 4 wide vector reads when the tile is inside
        the buffer and its rows are 16 byte aligned; else element by element (fallback). Same
        values either way, so the arithmetic and its order do not change."""
        b, rd, _, others = info
        t = MSL_TYPE[b.dtype]
        base = " + ".join([f"{em.var[a]} * {em.in_stride(b.name, d)}" for d, a in others] or ["0"])
        rs = em.in_stride(b.name, rd)
        ok = em.fresh("vok")
        em.line(f"const int {ok}_b = {base};")
        em.line(f"const bool {ok} = ({row_base} + {rows} <= {row_lim}) && ({col_base} + {cols} <= {col_lim}) "
                f"&& ({rs} % 4 == 0) && ({ok}_b % 4 == 0) && {uni};")
        em.open(f"if ({ok}) {{")
        em.open(f"for (int ts_e = ts_lid; ts_e < {rows * cols // 4}; ts_e += {T}) {{")
        em.line(f"const int ts_r = ts_e / {cols // 4};")
        em.line(f"const int ts_c = (ts_e % {cols // 4}) * 4;")
        em.line(f"const {t}4 ts_v4 = *((device const {t}4*)({_cname(b.name)} + {ok}_b + ({row_base} + ts_r) * {rs} "
                f"+ {col_base} + ts_c));")
        for j in range(4):
            em.line(f"{dst}[ts_r * {cols} + ts_c + {j}] = float(ts_v4[{j}]);")
        em.close()
        em.close()
        em.open("else {")
        fallback()
        em.close()

    ia, ib_kn, ib_nk = (_plain_load(em, cp.a, cp.m, cp.k, fold), plain(cp.b, cp.k, cp.n),
                        plain(cp.b, cp.n, cp.k))
    vec = os.environ.get("TRUESTAR_METAL_MM_VEC", "1") != "0"
    sa = lambda: stage("ts_A", BM * BK, BK, "ts_m0", "ts_k0", "ts_M", "ts_K", cp.a, cp.m, cp.k)
    sb = lambda: stage("ts_B", BK * BN, BN, "ts_k0", "ts_n0", "ts_K", "ts_N", cp.b, cp.k, cp.n)
    if vec and ia is not None:
        vstage("ts_A", BM, BK, "ts_m0", "ts_k0", "ts_M", "ts_K", ia, sa, _fold_uniform(em, ia, cp.a, fold))
    else:
        sa()
    if vec and ib_kn is not None:
        vstage("ts_B", BK, BN, "ts_k0", "ts_n0", "ts_K", "ts_N", ib_kn, sb)
    elif vec and ib_nk is not None and p.simdgroup:
        # weight [n, k] (nn.Linear): stage the tile as [n][k] and load its blocks transposed
        def sbt():
            stage("ts_B", BN * BK, BK, "ts_n0", "ts_k0", "ts_N", "ts_K", cp.b, cp.n, cp.k)
        vstage("ts_B", BN, BK, "ts_n0", "ts_k0", "ts_N", "ts_K", ib_nk, sbt)
    else:
        sb()
    bt = vec and ib_kn is None and ib_nk is not None and p.simdgroup
    em.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    if p.simdgroup:
        em.open(f"for (int ts_kk = 0; ts_kk < {BK}; ts_kk += 8) {{")
        for u in range(FR):
            em.line(f"simdgroup_float8x8 ts_a{u};")
            em.line(f"simdgroup_load(ts_a{u}, ts_A, {BK}, ulong2(ts_kk, ts_sr + {8 * u}));")
        for v in range(FR):
            em.line(f"simdgroup_float8x8 ts_b{v};")
            if bt:
                em.line(f"simdgroup_load(ts_b{v}, ts_B, {BK}, ulong2(ts_kk, ts_sc + {8 * v}), true);")
            else:
                em.line(f"simdgroup_load(ts_b{v}, ts_B, {BN}, ulong2(ts_sc + {8 * v}, ts_kk));")
        for u in range(FR):
            for v in range(FR):
                c = f"ts_c{u * FR + v}"
                em.line(f"simdgroup_multiply_accumulate({c}, ts_a{u}, ts_b{v}, {c});")
        em.close()
    else:
        em.open(f"for (int ts_kk = 0; ts_kk < {BK}; ++ts_kk) {{")
        em.line(f"const float ts_av = ts_A[ts_tr * {BK} + ts_kk];")
        em.open("for (int ts_c = 0; ts_c < 8; ++ts_c) {")
        em.line(f"ts_acc[ts_c] = ts_acc[ts_c] + ts_av * ts_B[ts_kk * {BN} + ts_tc + ts_c];")
        em.close()
        em.close()
    em.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    em.close()
    if direct:
        em.close()
    if p.simdgroup:
        for u in range(FR):
            for v in range(FR):
                em.line(f"simdgroup_store(ts_c{u * FR + v}, ts_C, {BN}, ulong2(ts_sc + {8 * v}, ts_sr + {8 * u}));")
    else:
        em.open("for (int ts_c = 0; ts_c < 8; ++ts_c) {")
        em.line(f"ts_C[ts_tr * {BN} + ts_tc + ts_c] = ts_acc[ts_c];")
        em.close()
    em.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    em.open(f"for (int ts_e = ts_lid; ts_e < {BM * BN}; ts_e += {T}) {{")
    ri, ci = em.fresh("u"), em.fresh("u")
    em.line(f"const int {ri} = ts_m0 + ts_e / {BN};")
    em.line(f"const int {ci} = ts_n0 + ts_e % {BN};")
    em.open(f"if ({ri} < ts_M && {ci} < ts_N) {{")
    saved = dict(em.values)
    bind_row(ri)
    em.var[cp.n] = ci
    em.line("const float ts_cv = ts_C[ts_e];")
    em.values[cp.red] = "ts_cv"
    em.store(o, em.ex(o.expr))
    em.values = saved
    em.close()
    em.close()


def _tg_reduce(em: _Em, v: str, op: str, G: int) -> str:
    """Every thread's value v folded by the threadgroup in a fixed tree; every thread gets it."""
    r = em.fresh("red")
    em.line(f"ts_sh[ts_lid] = {v};")
    em.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    em.open(f"for (int ts_s = {G // 2}; ts_s > 0; ts_s >>= 1) {{")
    em.line(f"if (ts_lid < ts_s) ts_sh[ts_lid] = {_combine(op, 'ts_sh[ts_lid]', 'ts_sh[ts_lid + ts_s]')};")
    em.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    em.close()
    em.line(f"const float {r} = ts_sh[0];")
    em.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    return r


def _att_rows(k: Kernel, ap) -> List[Axis]:
    return [a for a in k.outputs[0].axes if a != ap.d]


def _gen_attention_simd(k: Kernel, p: MetalPlan, em: _Em):
    """B query rows per threadgroup, 8 per SIMD group, chunks of B keys, two passes. Pass 1: per
    chunk the scores S = Q K^T with SIMD group 8 x 8 matrix products (Q held in registers, the key
    chunk staged in threadgroup memory), the score expression and mask per element, the running row
    max and sum (four lanes per row, combined with fixed xor shuffles). Pass 2: the same scores
    again, P = exp(S - max) and O += P V with matrix products; O / sum at the end. The output is
    never rescaled, so a NaN stays in its own row. Every order is fixed by the shape (no atomics),
    so bits repeat run to run and a row's bits do not depend on the other rows or the batch."""
    from .ir import top_reduces
    ap = p.base
    term = ap.terms[0]
    B, BK = p.tile
    NSG = B // 8
    G = p.group
    o = k.outputs[0]
    C, D = term.c.size, ap.d.size
    W = BK // 4                      # score columns per lane in the per element phase
    dot = top_reduces(term.full_score)[0]
    em.line("const int ts_lid = (int)ts_lt.x;")
    em.line("const int ts_sl = (int)ts_sli;")
    em.line(f"threadgroup float ts_KV[{B * (C + D)}];")
    em.line(f"threadgroup float ts_S[{B * BK}];")
    em.line(f"threadgroup float ts_Lr[{B}];")
    em.line("threadgroup float* ts_K = ts_KV;")
    em.line(f"threadgroup float* ts_V = ts_KV + {BK * C};")
    em.line(f"const int ts_I = {em.size(ap.i.size)};")
    em.line(f"const int ts_N = {em.size(ap.j.size)};")
    em.line(f"const int ts_NIT = (ts_I + {B - 1}) / {B};")
    em.line("const int ts_it = (int)ts_tg.x % ts_NIT;")
    em.decompose(list(ap.batch), "(int)ts_tg.x / ts_NIT")
    em.line(f"const int ts_i0 = ts_it * {B};")
    # Q tile through threadgroup memory into registers
    em.open(f"for (int ts_e = ts_lid; ts_e < {B * C}; ts_e += {G}) {{")
    em.line(f"const int ts_iq = ts_i0 + ts_e / {C};")
    em.line("float ts_x = 0.0f;")
    em.open("if (ts_iq < ts_I) {")
    saved = dict(em.values)
    em.var[ap.i], em.var[term.c] = "ts_iq", f"(ts_e % {C})"
    em.line(f"ts_x = float({em.ex(term.q)});")
    em.values = saved
    em.close()
    em.line("ts_KV[ts_e] = ts_x;")
    em.close()
    em.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    em.line(f"simdgroup_float8x8 ts_qm[{C // 8}];")
    em.line(f"for (int ts_cb = 0; ts_cb < {C // 8}; ++ts_cb) simdgroup_load(ts_qm[ts_cb], ts_KV, {C}, "
            f"ulong2(ts_cb * 8, ts_sg * 8));")
    em.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    em.line("const int ts_rr = ts_sl / 4;")
    em.line(f"const int ts_c0 = (ts_sl % 4) * {W};")
    em.line("const int ts_r = (int)ts_sg * 8 + ts_rr;")
    em.line("const int ts_iq = ts_i0 + ts_r;")
    lo = "0" if ap.lo_off is None else f"max(0, ts_i0 + ({ap.lo_off}))"
    hi = "ts_N" if ap.hi_off is None else f"min(ts_N, ts_i0 + {B} + ({ap.hi_off}))"
    em.line(f"const int ts_jlo = {lo};")
    em.line(f"const int ts_jhi = {hi};")
    em.line("float ts_m = -INFINITY;")
    em.line("float ts_l = 0.0f;")

    def stage(name, n_cols, expr, col_axis):
        info = _plain_load(em, expr, ap.j, col_axis) if os.environ.get("TRUESTAR_METAL_ATT_VEC", "1") != "0" else None
        if info is not None and n_cols % 4 == 0:
            # a key chunk of a plain load: 4 wide reads when the chunk is inside the buffer and its
            # rows are 16 byte aligned (the same values as element by element)
            b, rd, _, others = info
            t = MSL_TYPE[b.dtype]
            base = " + ".join([f"{em.var[a]} * {em.in_stride(b.name, d)}" for d, a in others] or ["0"])
            rs = em.in_stride(b.name, rd)
            ok = em.fresh("vok")
            em.line(f"const int {ok}_b = {base};")
            em.open(f"if (ts_j0 + {BK} <= ts_N && {rs} % 4 == 0 && {ok}_b % 4 == 0) {{")
            em.open(f"for (int ts_e = ts_lid; ts_e < {BK * n_cols // 4}; ts_e += {G}) {{")
            em.line(f"const int ts_rv = ts_e / {n_cols // 4};")
            em.line(f"const int ts_cv = (ts_e % {n_cols // 4}) * 4;")
            em.line(f"const {t}4 ts_v4 = *((device const {t}4*)({_cname(b.name)} + {ok}_b + (ts_j0 + ts_rv) * {rs} + ts_cv));")
            for q in range(4):
                em.line(f"{name}[ts_rv * {n_cols} + ts_cv + {q}] = float(ts_v4[{q}]);")
            em.close()
            em.close()
            em.open("else {")
            stage_scalar(name, n_cols, expr, col_axis)
            em.close()
            return
        stage_scalar(name, n_cols, expr, col_axis)

    def stage_scalar(name, n_cols, expr, col_axis):
        em.open(f"for (int ts_e = ts_lid; ts_e < {BK * n_cols}; ts_e += {G}) {{")
        em.line(f"const int ts_jk = ts_j0 + ts_e / {n_cols};")
        em.line("float ts_x = 0.0f;")
        em.open("if (ts_jk < ts_N) {")
        sv = dict(em.values)
        em.var[ap.j], em.var[col_axis] = "ts_jk", f"(ts_e % {n_cols})"
        em.line(f"ts_x = float({em.ex(expr)});")
        em.values = sv
        em.close()
        em.line(f"{name}[ts_e] = ts_x;")
        em.close()

    def scores():
        """S tile of this SIMD group into ts_S, then each lane's W scores (expression and mask)."""
        em.line(f"simdgroup_float8x8 ts_sm[{BK // 8}];")
        em.line(f"for (int ts_nb = 0; ts_nb < {BK // 8}; ++ts_nb) ts_sm[ts_nb] = "
                f"make_filled_simdgroup_matrix<float, 8, 8>(0.0f);")
        em.open(f"for (int ts_cb = 0; ts_cb < {C // 8}; ++ts_cb) {{")
        em.open(f"for (int ts_nb = 0; ts_nb < {BK // 8}; ++ts_nb) {{")
        em.line("simdgroup_float8x8 ts_kt;")
        em.line(f"simdgroup_load(ts_kt, ts_K, {C}, ulong2(ts_cb * 8, ts_nb * 8), true);")
        em.line("simdgroup_multiply_accumulate(ts_sm[ts_nb], ts_qm[ts_cb], ts_kt, ts_sm[ts_nb]);")
        em.close()
        em.close()
        em.line(f"for (int ts_nb = 0; ts_nb < {BK // 8}; ++ts_nb) simdgroup_store(ts_sm[ts_nb], ts_S, {BK}, "
                f"ulong2(ts_nb * 8, ts_sg * 8));")
        em.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
        em.line(f"float ts_sv[{W}];")
        em.open(f"for (int ts_w = 0; ts_w < {W}; ++ts_w) {{")
        em.line("const int ts_jk = ts_j0 + ts_c0 + ts_w;")
        em.line(f"const float ts_dv = ts_S[ts_r * {BK} + ts_c0 + ts_w];")
        em.line("float ts_x = -INFINITY;")
        em.open("if (ts_iq < ts_I && ts_jk < ts_N) {")
        sv = dict(em.values)
        em.var[ap.i], em.var[ap.j] = "ts_iq", "ts_jk"
        em.values[dot] = "ts_dv"
        em.line(f"ts_x = float({em.ex(term.full_score)});")
        em.values = sv
        em.close()
        em.line("ts_sv[ts_w] = ts_x;")
        em.close()

    # pass 1: row max and sum
    em.open(f"for (int ts_j0 = ts_jlo; ts_j0 < ts_jhi; ts_j0 += {BK}) {{")
    stage("ts_K", C, term.kk, term.c)
    em.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    scores()
    em.line("float ts_cm = -INFINITY;")
    em.line(f"for (int ts_w = 0; ts_w < {W}; ++ts_w) ts_cm = ts_max(ts_cm, ts_sv[ts_w]);")
    em.line("ts_cm = ts_max(ts_cm, simd_shuffle_xor(ts_cm, 1));")
    em.line("ts_cm = ts_max(ts_cm, simd_shuffle_xor(ts_cm, 2));")
    em.line("const float ts_mn = ts_max(ts_m, ts_cm);")
    # nothing seen yet (every score so far -inf): keep the sum at zero, never exp(-inf + inf)
    em.line("const bool ts_none = ts_mn == -INFINITY;")
    em.line("float ts_ps = 0.0f;")
    em.line(f"for (int ts_w = 0; ts_w < {W}; ++ts_w) ts_ps = ts_ps + (ts_none ? 0.0f : precise::exp(ts_sv[ts_w] - ts_mn));")
    em.line("ts_ps = ts_ps + simd_shuffle_xor(ts_ps, 1);")
    em.line("ts_ps = ts_ps + simd_shuffle_xor(ts_ps, 2);")
    em.line("ts_l = ts_l * (ts_none ? 1.0f : precise::exp(ts_m - ts_mn)) + ts_ps;")
    em.line("ts_m = ts_mn;")
    em.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    em.close()
    em.line("if (ts_sl % 4 == 0) ts_Lr[ts_r] = ts_l;")
    # pass 2: O += exp(S - max) V
    em.line(f"simdgroup_float8x8 ts_om[{D // 8}];")
    em.line(f"for (int ts_dn = 0; ts_dn < {D // 8}; ++ts_dn) ts_om[ts_dn] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);")
    em.line("const bool ts_none = ts_m == -INFINITY;")
    em.open(f"for (int ts_j0 = ts_jlo; ts_j0 < ts_jhi; ts_j0 += {BK}) {{")
    stage("ts_K", C, term.kk, term.c)
    stage("ts_V", D, ap.v, ap.d)
    em.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    scores()
    em.line(f"for (int ts_w = 0; ts_w < {W}; ++ts_w) ts_S[ts_r * {BK} + ts_c0 + ts_w] = "
            f"ts_none ? 0.0f : precise::exp(ts_sv[ts_w] - ts_m);")
    em.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    em.open(f"for (int ts_kb = 0; ts_kb < {BK // 8}; ++ts_kb) {{")
    em.line("simdgroup_float8x8 ts_pm;")
    em.line(f"simdgroup_load(ts_pm, ts_S, {BK}, ulong2(ts_kb * 8, ts_sg * 8));")
    em.open(f"for (int ts_dn = 0; ts_dn < {D // 8}; ++ts_dn) {{")
    em.line("simdgroup_float8x8 ts_vm;")
    em.line(f"simdgroup_load(ts_vm, ts_V, {D}, ulong2(ts_dn * 8, ts_kb * 8));")
    em.line("simdgroup_multiply_accumulate(ts_om[ts_dn], ts_pm, ts_vm, ts_om[ts_dn]);")
    em.close()
    em.close()
    em.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    em.close()
    # O / sum, the epilogue, the store
    em.line(f"for (int ts_dn = 0; ts_dn < {D // 8}; ++ts_dn) simdgroup_store(ts_om[ts_dn], ts_KV, {D}, "
            f"ulong2(ts_dn * 8, ts_sg * 8));")
    em.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    em.open(f"for (int ts_e = ts_lid; ts_e < {B * D}; ts_e += {G}) {{")
    em.line(f"const int ts_ro = ts_e / {D};")
    em.line("const int ts_io = ts_i0 + ts_ro;")
    em.open("if (ts_io < ts_I) {")
    saved = dict(em.values)
    em.var[ap.i], em.var[ap.d] = "ts_io", f"(ts_e % {D})"
    c = em.ex(term.coef)
    red = em.fresh("att")
    em.line(f"const float {red} = ({c}) * ts_div(ts_KV[ts_e], ts_Lr[ts_ro]);")
    em.values[ap.red] = red
    em.store(o, em.ex(o.expr))
    em.values = saved
    em.close()
    em.close()


def _gen_attention(k: Kernel, p: MetalPlan, em: _Em):
    """One threadgroup per query row (batch axes and the query axis). Keys go in chunks of G, one
    key per thread: its scores (the dot product a serial loop over the contraction axis), the
    chunk's max and sum folded by the group in a fixed tree, the running max, sum and output
    rescaled as in flash attention, then each thread adds the chunk's weighted values for its head
    dim elements, keys in index order. No atomics; the order depends only on G and the shape, so the
    bits repeat run to run and do not depend on the number of rows."""
    ap = p.base
    G = p.group
    o = k.outputs[0]
    D = ap.d.size
    ND = -(-D // G)
    T = len(ap.terms)
    em.line("const int ts_lid = (int)ts_lt.x;")
    em.line(f"threadgroup float ts_sh[{G}];")
    em.line(f"threadgroup float ts_W[{T * G}];")
    rows = _att_rows(k, ap)
    em.decompose(rows, "(int)ts_tg.x")
    bound = set(rows)
    em.line(f"const int ts_N = {em.size(ap.j.size)};")
    for t in range(T):
        em.line(f"float ts_m{t} = -INFINITY;")
        em.line(f"float ts_l{t} = 0.0f;")
        em.line(f"float ts_acc{t}[{ND}] = {{" + ", ".join(["0.0f"] * ND) + "};")
    em.open(f"for (int ts_j0 = 0; ts_j0 < ts_N; ts_j0 += {G}) {{")
    em.line("const int ts_jv = ts_j0 + ts_lid;")
    for t in range(T):
        em.line(f"float ts_s{t} = -INFINITY;")
    em.open("if (ts_jv < ts_N) {")
    saved = dict(em.values)
    em.var[ap.j] = "ts_jv"
    inner = bound | {ap.j}
    for t, term in enumerate(ap.terms):
        for r in _pending([term.full_score], inner, em.values):
            em.serial_reduce(r, inner)
        em.line(f"ts_s{t} = float({em.ex(term.full_score)});")
    em.values = saved
    em.close()
    for t in range(T):
        cm = _tg_reduce(em, f"ts_s{t}", "max", G)
        mn, al, pv = em.fresh("mn"), em.fresh("al"), em.fresh("pw")
        em.line(f"const float {mn} = ts_max(ts_m{t}, {cm});")
        # nothing seen yet (every score so far -inf): keep the sums at zero, never exp(-inf + inf)
        em.line(f"const float {al} = ({mn} == -INFINITY) ? 1.0f : precise::exp(ts_m{t} - {mn});")
        em.line(f"const float {pv} = ({mn} == -INFINITY) ? 0.0f : precise::exp(ts_s{t} - {mn});")
        em.line(f"ts_W[{t * G} + ts_lid] = {pv};")
        cs = _tg_reduce(em, pv, "sum", G)
        em.line(f"ts_l{t} = ts_l{t} * {al} + {cs};")
        em.line(f"ts_m{t} = {mn};")
        em.line(f"for (int ts_q = 0; ts_q < {ND}; ++ts_q) ts_acc{t}[ts_q] = ts_acc{t}[ts_q] * {al};")
    em.line(f"const int ts_jn = min({G}, ts_N - ts_j0);")
    em.open(f"for (int ts_q = 0; ts_q < {ND}; ++ts_q) {{")
    em.line(f"const int ts_dv = ts_lid + ts_q * {G};")
    em.open(f"if (ts_dv < {D}) {{")
    em.open("for (int ts_jj = 0; ts_jj < ts_jn; ++ts_jj) {")
    em.line("const int ts_jk = ts_j0 + ts_jj;")
    saved = dict(em.values)
    em.var[ap.j], em.var[ap.d] = "ts_jk", "ts_dv"
    vv = em.ex(ap.v)
    for t in range(T):
        em.line(f"ts_acc{t}[ts_q] = ts_acc{t}[ts_q] + ts_W[{t * G} + ts_jj] * {vv};")
    em.values = saved
    em.close()
    em.close()
    em.close()
    em.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
    em.close()
    em.open(f"for (int ts_q = 0; ts_q < {ND}; ++ts_q) {{")
    em.line(f"const int ts_dv = ts_lid + ts_q * {G};")
    em.open(f"if (ts_dv < {D}) {{")
    saved = dict(em.values)
    em.var[ap.d] = "ts_dv"
    parts = []
    for t, term in enumerate(ap.terms):
        c = em.ex(term.coef)
        parts.append(f"({c}) * ts_div(ts_acc{t}[ts_q], ts_l{t})")
    red = em.fresh("att")
    em.line(f"const float {red} = " + " + ".join(parts) + ";")
    em.values[ap.red] = red
    em.store(o, em.ex(o.expr))
    em.values = saved
    em.close()
    em.close()


def generate(k: Kernel, opts: Optional[Dict] = None) -> Tuple[str, MetalPlan]:
    """(Metal source, plan) for kernel k. Works on any machine: compiling needs macOS."""
    opts = dict(opts or {})
    p = plan(k, simdgroup=opts.get("simdgroup", True), attention=opts.get("attention", "auto"),
             mm=opts.get("mm", ""))
    em = _Em(k, opts)
    em.prelude()
    if p.shape in ("pointwise", "loops"):
        extra = ["uint3 ts_gid [[thread_position_in_grid]]"]
        _gen_elementwise(k, p, em)
    elif p.shape == "rows":
        extra = ["uint3 ts_tg [[threadgroup_position_in_grid]]",
                 "uint3 ts_lt [[thread_position_in_threadgroup]]"]
        _gen_rows(k, p, em)
    elif p.shape == "attention_simd":
        extra = ["uint3 ts_tg [[threadgroup_position_in_grid]]",
                 "uint3 ts_lt [[thread_position_in_threadgroup]]",
                 "uint ts_sg [[simdgroup_index_in_threadgroup]]",
                 "uint ts_sli [[thread_index_in_simdgroup]]"]
        _gen_attention_simd(k, p, em)
    elif p.shape == "attention":
        extra = ["uint3 ts_tg [[threadgroup_position_in_grid]]",
                 "uint3 ts_lt [[thread_position_in_threadgroup]]"]
        _gen_attention(k, p, em)
    else:
        extra = ["uint3 ts_tg [[threadgroup_position_in_grid]]",
                 "uint3 ts_lt [[thread_position_in_threadgroup]]"]
        if p.simdgroup:
            extra.append("uint ts_sg [[simdgroup_index_in_threadgroup]]")
        _gen_contraction(k, p, em)
    head = HEADER
    if opts.get("math") == "round_once":
        from . import roundonce
        used = sorted({n.op for o in k.outputs for n in ir.walk(o.expr) if isinstance(n, Unary) and n.op in RO_OPS})
        if used:
            head = head + roundonce.metal_source(used)
    if any(r.op == "osum" for r in k.reduces()):
        head = head + OSUM_HEADER
    if p.shape in ("contraction", "attention_simd") and p.simdgroup:
        head = head.replace("using namespace metal;", "#include <metal_simdgroup_matrix>\nusing namespace metal;", 1)
    note = f"// truestar metal kernel {k.name}: {p.shape}" + (f" (structure: {p.via})" if p.via else "") + "\n"
    src = note + head + "\n" + _signature(k, extra) + " {\n" + "\n".join(em.lines) + "\n}\n"
    return src, p


# =========================================================================== run time

def load(src: str, kernel_name: str = "", path: str = ""):
    """Compile Metal source with torch.mps.compile_shader; the kernel function is ``kern``."""
    require()
    import torch
    try:
        lib = torch.mps.compile_shader(src)
        fn = getattr(lib, "kern")
    except Exception as e:
        where = f" (source: {path})" if path else ""
        raise MetalCompileError(f"[TS843] the Metal compiler rejected the generated source of kernel "
                                f"{kernel_name}{where}. This is a Truestar bug: please report it with "
                                f"the source file.\n{e}") from None
    return lib, fn


def _contiguous(shape) -> List[int]:
    st, acc = [], 1
    for x in reversed(shape):
        st.append(acc)
        acc *= int(x)
    return list(reversed(st))


def params(k: Kernel, env: Dict[str, int], in_shapes: Dict[str, tuple],
           out_shapes: Dict[str, tuple]) -> List[int]:
    """Values of the parameter buffer (param_layout order) for one launch. Inputs are passed
    contiguous, so their strides follow from their shapes."""
    vals = []
    for p in param_layout(k):
        if p[0] == "sym":
            vals.append(int(env[p[1]]))
        elif p[0] == "in":
            vals.append(_contiguous(in_shapes[p[1]])[p[2]])
        else:
            vals.append(_contiguous(out_shapes[p[1]])[p[2]])
    for v in vals:
        if v > MAX_THREADS:
            raise MetalUnsupported("TS845", f"kernel {k.name}: a size or stride of {v} needs more than 32 "
                                            f"bit offsets, which Metal kernels use")
    return vals or [0]


def grid(p: MetalPlan, k: Kernel, env: Dict[str, int]) -> Tuple[List[int], List[int], int]:
    """(threads, group size, work items) for one launch; work 0 means nothing to launch."""
    def sz(s):
        return s if isinstance(s, int) else env[s.name]

    def prod(axes):
        n = 1
        for a in axes:
            n *= sz(a.size)
        return n

    if p.shape in ("pointwise", "loops"):
        total = prod(k.outputs[0].axes)
        n = -(-total // p.group) * p.group
        th, work = [n, 1, 1], total
    elif p.shape == "rows":
        rows = prod(p.base.rows)
        th, work = [rows * p.group, 1, 1], rows
    elif p.shape == "attention":
        rows = prod(_att_rows(k, p.base))
        th, work = [rows * p.group, 1, 1], rows * sz(p.base.d.size)
    elif p.shape == "attention_simd":
        nit = -(-sz(p.base.i.size) // p.tile[0])
        groups = prod(p.base.batch) * nit
        th, work = [groups * p.group, 1, 1], groups
    else:
        cp = p.base
        M, N, B = sz(cp.m.size) * prod(getattr(cp, "fold", ())), sz(cp.n.size), prod(cp.batch)
        t = p.tile[0] if p.tile else BM
        gx, gy = -(-N // t), -(-M // t)
        th, work = [gx * MM_THREADS, gy, B], M * N * B
    if any(t > MAX_THREADS for t in th):
        raise MetalUnsupported("TS845", f"kernel {k.name}: a launch of {th} threads exceeds the 32 bit grid "
                                        f"the Metal backend uses")
    return th, [p.group, 1, 1], work
