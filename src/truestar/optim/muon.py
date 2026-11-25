"""Muon: momentum, then a Newton Schulz orthogonalization of each 2D update, with its
orthogonality measured against float64 (docs/OPTIM.md "Muon").

    opt = optim.Muon(lr=0.02, weight_decay=0.01)          # 2D leaves: Muon; other leaves: AdamW
    params, state = opt.update(params, grads, state, step=0)
    reports = opt.check(params, grads, state, step=0)       # one OrthoReport per 2D leaf

The step of a 2D leaf (rows m, cols n), with mu the momentum:

    buf = mu buf + (1 - mu) g                  # an average, as torch.optim.Muon's lerp computes
    u   = (1 - mu) g + mu buf   (Nesterov)     or u = buf
    X0  = u / max(||u||_F, eps)                # the norm in the parameter dtype, then cast to ns_dtype
    X   = a X + (b A + c A A) X,  A = X X^T    # ns_steps times, on the short side (X^T when m > n)
    p   = p (1 - lr wd) - lr r X,  r = sqrt(max(1, m / n)) ("original") or 0.2 sqrt(max(m, n))

Exact arithmetic maps every singular value s of X0 through the odd polynomial
q(x) = a x + b x^3 + c x^5, ns_steps times, and keeps the singular vectors. So the float64 result of
the iteration is known in closed form from one SVD of u: ``U q^k(S / ||u||) V^T``. That splits what
an orthogonality error is made of into two parts Truestar reports separately:

* ``intended``: how far the exact iteration is from the polar factor U V^T (the coefficients
  3.4445, -4.7750, 2.0315 deliberately do not converge: singular values end in about [0.68, 1.12]);
* ``rounding``: how far the computed update is from the exact iteration (bfloat16 matmuls, the
  kernel), checked against a first order bound derived from the dtype and the spectrum.

A wrong kernel or a typo in a coefficient changes ``rounding`` far beyond its bound (tested).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from .. import pytree
from ..frontend import ops
from ..nn.core import inline
from . import Optimizer, _check_beta, _leaves, _schedule

F64 = torch.float64

# The quintic of torch.optim.Muon's default, maximal slope at zero (the polynomial is the published
# Muon choice; these three numbers are a mathematical constant, not code).
DEFAULT_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
# A convergent choice: the cubic Newton Schulz map 1.5 x - 0.5 x^3 (fixed point 1, slope 0 there),
# written as a quintic with c = 0. Slower, but it converges to the polar factor.
CONVERGENT_COEFFICIENTS = (1.5, -0.5, 0.0)

_UNIT = {torch.float64: 2.0 ** -53, torch.float32: 2.0 ** -24, torch.float16: 2.0 ** -11,
         torch.bfloat16: 2.0 ** -8}
_NS_DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32, "f64": torch.float64}


@inline
def newton_schulz(x, a, b, c, steps, eps, dtype):
    """``steps`` iterations of X <- a X + (b A + c A A) X with A = X X^T, after X <- X / ||X||_F,
    in ``dtype``; the norm is taken in the input's dtype. Works on the short side (rows <= cols).
    Each matmul result is stored in ``dtype``; the two combinations (b A + c A A and a X + B X)
    are formed in float32 and stored once, as a GEMM epilogue does (torch.optim.Muon's addmm)."""
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    wide = torch.float64 if dtype == torch.float64 else torch.float32
    n = ops.sqrt(ops.sum(x * x))
    y = ops.cast(x / ops.clamp(n, min=eps), dtype)
    for _ in range(steps):
        g = ops.matmul(y, y.T)
        h = ops.cast(ops.cast(g, wide) * b + ops.cast(ops.matmul(g, g), wide) * c, dtype)
        y = ops.cast(ops.cast(y, wide) * a + ops.cast(ops.matmul(h, y), wide), dtype)
    if tall:
        y = y.T
    return y


def lr_ratio(rows: int, cols: int, adjust: str) -> float:
    if adjust == "original":
        return math.sqrt(max(1.0, rows / cols))
    if adjust == "match_rms_adamw":
        return 0.2 * math.sqrt(max(rows, cols))
    if adjust == "none":
        return 1.0
    raise ValueError(f"adjust_lr must be 'original', 'match_rms_adamw' or 'none', got {adjust!r}")


@dataclasses.dataclass(frozen=True)
class Muon(Optimizer):
    """Muon for 2D leaves, AdamW for every other leaf (biases, norms, 1D or 4D tensors).

    ``ns_dtype`` is the dtype of the iteration: "bf16" (torch.optim.Muon's), "f16", "f32" or "f64".
    The AdamW part uses ``adam_lr`` (default ``lr``), ``adam_b1``, ``adam_b2``, ``adam_eps`` and
    ``adam_weight_decay`` (default ``weight_decay``). Slots: ``m`` (Muon's momentum buffer for 2D
    leaves, AdamW's first moment otherwise) and ``v`` (AdamW's second moment; unused and kept at
    zero for 2D leaves, which costs one parameter sized buffer per matrix)."""

    lr: Any = 0.02
    weight_decay: float = 0.1
    momentum: float = 0.95
    nesterov: bool = True
    ns_coefficients: Tuple[float, float, float] = DEFAULT_COEFFICIENTS
    ns_steps: int = 5
    eps: float = 1e-7
    adjust_lr: str = "original"
    ns_dtype: str = "bf16"
    adam_lr: Any = None
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    adam_eps: float = 1e-8
    adam_weight_decay: Optional[float] = None
    slots = ("m", "v")

    def __post_init__(self):
        _schedule(self.lr)
        if self.adam_lr is not None:
            _schedule(self.adam_lr)
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}")
        if len(self.ns_coefficients) != 3:
            raise ValueError("ns_coefficients must be three numbers (a, b, c)")
        if not 0 <= int(self.ns_steps) <= 50:
            raise ValueError(f"ns_steps must be in [0, 50], got {self.ns_steps}")
        if self.ns_dtype not in _NS_DTYPES:
            raise ValueError(f"ns_dtype must be one of {', '.join(_NS_DTYPES)}, got {self.ns_dtype!r}")
        lr_ratio(2, 1, self.adjust_lr)
        _check_beta("adam_b1", self.adam_b1)
        _check_beta("adam_b2", self.adam_b2)
        object.__setattr__(self, "ns_coefficients", tuple(float(c) for c in self.ns_coefficients))

    @property
    def ns_torch_dtype(self) -> torch.dtype:
        return _NS_DTYPES[self.ns_dtype]

    @property
    def adam_wd(self) -> float:
        return self.weight_decay if self.adam_weight_decay is None else float(self.adam_weight_decay)

    def adam_lr_at(self, step: int) -> float:
        return float(_schedule(self.adam_lr if self.adam_lr is not None else self.lr)(step))

    def scalars(self, t, lr):
        c1 = -ops.expm1(t * math.log(self.adam_b1)) if self.adam_b1 > 0.0 else ops.ones_like(t)
        c2 = -ops.expm1(t * math.log(self.adam_b2)) if self.adam_b2 > 0.0 else ops.ones_like(t)
        alr = lr * self.adam_lr_scale
        return {"lr": lr, "decay": 1.0 - lr * self.weight_decay, "astep": alr / c1, "c2s": ops.sqrt(c2),
                "adecay": 1.0 - alr * self.adam_wd}

    @property
    def adam_lr_scale(self) -> float:
        """AdamW's learning rate as a multiple of Muon's (a constant ratio, so one schedule drives
        both: ``adam_lr`` is read at step 0 against ``lr`` at step 0)."""
        if self.adam_lr is None:
            return 1.0
        base = self.lr_at(0)
        return self.adam_lr_at(0) / base if base > 0 else 1.0

    def leaf(self, p, g, s, k):
        dt = p.dtype
        if p.ndim == 2:
            mu = self.momentum
            buf = s["m"] * mu + g * (1.0 - mu)
            if self.nesterov:
                u = g * (1.0 - mu) + buf * mu
            else:
                u = buf
            a, b, c = self.ns_coefficients
            o = ops.cast(newton_schulz(u, a, b, c, self.ns_steps, self.eps, self.ns_torch_dtype), dt)
            if self.adjust_lr == "original":
                ratio = math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
            elif self.adjust_lr == "match_rms_adamw":
                ratio = 0.2 * math.sqrt(max(p.shape[0], p.shape[1]))
            else:
                ratio = 1.0
            step = ops.cast(k["lr"], dt) * ratio
            q = p * ops.cast(k["decay"], dt) - o * step
            return q, {"m": buf, "v": s["v"]}
        if self.adam_wd != 0.0:
            p = p * ops.cast(k["adecay"], dt)
        m = s["m"] + (g - s["m"]) * (1.0 - self.adam_b1)
        v = s["v"] * self.adam_b2 + g * g * (1.0 - self.adam_b2)
        denom = ops.sqrt(v) / ops.cast(k["c2s"], dt) + self.adam_eps
        return p - (m / denom) * ops.cast(k["astep"], dt), {"m": m, "v": v}

    # ------------------------------------------------------------------ checks against float64

    def ns_input(self, g: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(the matrix the iteration orthogonalizes, the new buffer), eagerly in the leaf's dtype."""
        mu = self.momentum
        buf = m * mu + g * (1.0 - mu)
        u = g * (1.0 - mu) + buf * mu if self.nesterov else buf
        return u, buf

    def orthogonalize(self, u: torch.Tensor) -> torch.Tensor:
        a, b, c = self.ns_coefficients
        return newton_schulz(u, a, b, c, self.ns_steps, self.eps, self.ns_torch_dtype)

    def check(self, params, grads, state, step: int, leaves: Optional[List[int]] = None,
              raise_on_fail: bool = False) -> List["OrthoReport"]:
        """An ``OrthoReport`` for the orthogonalized update of every 2D leaf of update ``step``
        (state as before the update). Eager: the iteration runs in ``ns_dtype`` through PyTorch.
        ``raise_on_fail`` raises MuonOrthogonalityError (TS1040) when a report is not ok."""
        ps, _ = _leaves(params)
        gs, _ = _leaves(grads)
        ms, _ = _leaves(state["m"])
        out = []
        for i, (p, g, m) in enumerate(zip(ps, gs, ms)):
            if p.ndim != 2 or (leaves is not None and i not in leaves):
                continue
            u, _ = self.ns_input(g, m)
            o = self.orthogonalize(u)
            out.append(orthogonality_report(u, o, self.ns_coefficients, self.ns_steps, self.eps,
                                            self.ns_torch_dtype, name=f"leaf {i}"))
        if raise_on_fail and any(not r.ok for r in out):
            raise MuonOrthogonalityError(out)
        return out

    def recover_update(self, p_old: torch.Tensor, p_new: torch.Tensor, step: int) -> torch.Tensor:
        """The orthogonalized update a step applied to the 2D leaf ``p_old`` (for example inside a
        compiled step program): (p_old (1 - lr wd) - p_new) / (lr r), in float64. Exact up to the
        rounding of the parameter update itself (|p| u / (lr r) per entry)."""
        lr = self.lr_at(step)
        r = lr_ratio(p_old.shape[0], p_old.shape[1], self.adjust_lr)
        return (p_old.to(F64) * (1.0 - lr * self.weight_decay) - p_new.to(F64)) / (lr * r)

    def recover_resolution(self, p_old: torch.Tensor, step: int) -> float:
        """Frobenius norm of the error ``recover_update`` adds: the parameter's own rounding."""
        lr = self.lr_at(step)
        r = lr_ratio(p_old.shape[0], p_old.shape[1], self.adjust_lr)
        u = _UNIT.get(p_old.dtype, 2.0 ** -24)
        return 2.0 * u * float(p_old.to(F64).norm()) / (lr * r)


# ----------------------------------------------------------------------------- the float64 verdict


class MuonOrthogonalityError(RuntimeError):
    """TS1040: a Muon update is off the float64 result of its own iteration by more than its
    dtype's rounding."""

    def __init__(self, reports: List["OrthoReport"]):
        self.reports = [r for r in reports if not r.ok]
        self.code = "TS1040"
        super().__init__("truestar TS1040 Muon orthogonalization: " + "; ".join(r.line() for r in self.reports))


@dataclasses.dataclass
class OrthoReport:
    """The orthogonality of one Newton Schulz result ``o`` of the matrix ``u``, against float64.

    Norms are Frobenius norms divided by sqrt(rank) (so 1.0 is the size of an orthogonal factor),
    except ``ortho_error``, the largest distance of a singular value of ``o`` from 1."""

    name: str
    shape: Tuple[int, int]
    dtype: str
    rank: int
    ortho_error: float          # max_i |sigma_i(o) - 1| over the rank of u
    sv_range: Tuple[float, float]
    polar_distance: float       # ||o - U V^T|| / sqrt(rank)
    intended: float             # ||exact iteration - U V^T|| / sqrt(rank): the algorithm's choice
    intended_sv_range: Tuple[float, float]
    rounding: float             # ||o - exact iteration|| / sqrt(rank): what the dtype and kernel add
    estimate: float             # typical rounding for this dtype and spectrum (errors spread evenly)
    bound: float                # first order worst case of ``rounding`` (errors aimed at the worst direction)
    amplification: float        # product over steps of the worst case gain

    @property
    def tolerance(self) -> float:
        return min(ESTIMATE_FACTOR * self.estimate, self.bound)

    @property
    def ok(self) -> bool:
        """Rounding within 4 times the typical estimate (and so within the worst case bound)."""
        return self.rounding <= self.tolerance

    def issue(self):
        from ..codes import Issue

        return Issue("TS1041" if self.ok else "TS1040", self.line())

    def as_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["ok"] = self.ok
        d["tolerance"] = self.tolerance
        return d

    def line(self) -> str:
        v = "ok" if self.ok else "EXCEEDS BOUND"
        return (f"{self.name} {self.shape[0]}x{self.shape[1]} {self.dtype}: orthogonality error "
                f"{self.ortho_error:.3g} (sv {self.sv_range[0]:.3f}..{self.sv_range[1]:.3f}); intended "
                f"{self.intended:.3g}, rounding {self.rounding:.3g} vs tolerance {self.tolerance:.3g} "
                f"(estimate {self.estimate:.3g}, worst case {self.bound:.3g}) {v}")


def _poly(x, a, b, c):
    return x * (a + x * x * (b + c * x * x))


def exact_iteration(u: torch.Tensor, coefficients, steps: int, eps: float):
    """The float64 result of the iteration in exact arithmetic, from one SVD of ``u``:
    ``(U q^k(S / max(||u||_F, eps)) V^T, U V^T, S, singular value history)``."""
    a, b, c = coefficients
    u64 = u.detach().to(F64).cpu()
    U, S, Vh = torch.linalg.svd(u64, full_matrices=False)
    n = max(float(u64.norm()), eps)
    x = S / n
    hist = [x]
    for _ in range(steps):
        x = _poly(x, a, b, c)
        hist.append(x)
    return (U * x) @ Vh, U @ Vh, S, hist


def _lipschitz(x: torch.Tensor, a: float, b: float, c: float, full_rank: bool) -> float:
    """The first order gain of one step at singular values x in the Frobenius norm. In the SVD
    basis the derivative of X -> U q(S) V^T scales each component of a perturbation by a divided
    difference (q(s_i) - q(s_j)) / (s_i - s_j), by (q(s_i) + q(s_j)) / (s_i + s_j), or (off the
    row space) by q(s_i) / s_i. The first lies in the range of q' on [min s, max s] (mean value
    theorem), the other two are averages of q(s_i) / s_i. A rank deficient input adds s = 0."""
    if not x.numel():
        return abs(a)
    lo = float(x.min()) if full_rank else 0.0
    grid = torch.linspace(lo, float(x.max()), 4097, dtype=F64)
    slope = float((a + 3 * b * grid ** 2 + 5 * c * grid ** 4).abs().max())
    xs = x[x > 0]
    ratio = float((a + b * xs ** 2 + c * xs ** 4).abs().max()) if xs.numel() else abs(a)
    return max(slope, ratio, abs(a) if not full_rank else 0.0)


def rounding_bound(hist: List[torch.Tensor], coefficients, rows: int, cols: int, dtype: torch.dtype,
                   input_dtype: Optional[torch.dtype] = None, full_rank: bool = True) -> Tuple[float, float]:
    """First order bound on ||computed - exact||_F of the iteration in ``dtype`` (storage unit u),
    with products accumulated in float32 for 16 bit dtypes (every Truestar kernel and cuBLAS do;
    a kernel that accumulates in 16 bits exceeds it) and in the dtype otherwise, from the exact
    singular values of each iterate (``hist``, over the rank). Returns (bound, amplification).

    Per step: the rounding of A = X X^T, of B = b A + c A A, of B X and of a X + B X (each a storage
    rounding u times its norm plus an inner product error k u_acc |.||.|), plus the error carried
    from earlier steps times the step's first order gain (``_lipschitz``). The start adds the
    norm and division in the input dtype and the cast. Doubled for second order terms. A first order
    estimate, not a proof for every input; measured against bfloat16, float16, float32 and float64
    runs in tests/test_muon.py."""
    a, b, c = coefficients
    u = _UNIT.get(dtype, 2.0 ** -24)
    ui = _UNIT.get(input_dtype, u) if input_dtype is not None else u
    acc = min(u, 2.0 ** -24)
    m, n = min(rows, cols), max(rows, cols)
    ga, gm = n * acc, m * acc
    e = (u + 2.0 * ui + n * ui) * float(torch.sqrt((hist[0] ** 2).sum()))
    amp = 1.0
    for x in hist[:-1]:
        fx = float(torch.sqrt((x ** 2).sum()))
        fa = float(torch.sqrt((x ** 4).sum()))
        fa2 = float(torch.sqrt((x ** 8).sum()))
        a2 = float((x ** 2).max()) if x.numel() else 0.0
        fb = float(torch.sqrt(((b * x ** 2 + c * x ** 4) ** 2).sum()))
        fbx = float(torch.sqrt(((b * x ** 3 + c * x ** 5) ** 2).sum()))
        fnew = float(torch.sqrt((_poly(x, a, b, c) ** 2).sum()))
        e_a = u * fa + ga * fx * fx
        e_b = abs(c) * (u * fa2 + gm * fa * fa) + u * (abs(b) * fa + 2 * abs(c) * fa2) + u * fb             + (abs(b) + 2 * abs(c) * a2) * e_a
        e_y = u * fbx + gm * fb * fx + math.sqrt(a2) * e_b
        delta = u * (abs(a) * fx + fbx) + u * fnew + e_y
        lip = _lipschitz(x, a, b, c, full_rank)
        e = lip * e + delta
        amp *= lip
    return 2.0 * e, amp


def _rms_gain(x: torch.Tensor, a: float, b: float, c: float, rows: int, cols: int) -> float:
    """Root mean square of the per component first order gains of one step (the divided
    differences of ``_lipschitz`` for every pair, and q(s)/s off the row space). A rounding error
    spread evenly over the components grows by this factor; one aimed at the worst direction grows
    by ``_lipschitz``. Pairs are sampled when the rank is above 1024."""
    m, n = min(rows, cols), max(rows, cols)
    xs = x[x > 0]
    if not xs.numel():
        return abs(a)
    if xs.numel() > 1024:
        xs = xs[torch.linspace(0, xs.numel() - 1, 1024).long()]
    qx = _poly(xs, a, b, c)
    d = xs[:, None] - xs[None, :]
    same = d.abs() < 1e-12
    dq = (qx[:, None] - qx[None, :]) / torch.where(same, torch.ones_like(d), d)
    dq = torch.where(same, (a + 3 * b * xs ** 2 + 5 * c * xs ** 4)[:, None].expand_as(d), dq)
    sq = (qx[:, None] + qx[None, :]) / (xs[:, None] + xs[None, :])
    ratio = qx / xs
    k = xs.numel()
    inside = float(((dq ** 2 + sq ** 2) / 2).mean()) * k * k
    off = float((ratio ** 2).mean()) * k * (n - k) + (m - k) * n * a * a
    return math.sqrt((inside + off) / (m * n))


def rounding_estimate(hist: List[torch.Tensor], coefficients, rows: int, cols: int, dtype: torch.dtype,
                      input_dtype: Optional[torch.dtype] = None) -> float:
    """Typical size of ||computed - exact||_F when rounding errors are spread evenly (each storage
    rounding u/sqrt(3) times its norm, inner products sqrt(k) u_acc), carried by ``_rms_gain``.
    Storage roundings per step are those of ``newton_schulz``: A, A A, B, B X and the new X.
    The verdict allows 4 times this; ``rounding_bound`` is the worst case."""
    a, b, c = coefficients
    u = _UNIT.get(dtype, 2.0 ** -24) / math.sqrt(3.0)
    ui = (_UNIT.get(input_dtype, u) if input_dtype is not None else u) / math.sqrt(3.0)
    acc = min(_UNIT.get(dtype, 2.0 ** -24), 2.0 ** -24)
    m, n = min(rows, cols), max(rows, cols)
    e2 = ((u ** 2 + 2 * ui ** 2) * float((hist[0] ** 2).sum()))
    for x in hist[:-1]:
        fx2 = float((x ** 2).sum())
        fa2 = float((x ** 4).sum())
        fb2 = float(((b * x ** 2 + c * x ** 4) ** 2).sum())
        fbx2 = float(((b * x ** 3 + c * x ** 5) ** 2).sum())
        fnew2 = float((_poly(x, a, b, c) ** 2).sum())
        a2 = float((x ** 2).max()) if x.numel() else 0.0
        fa4 = float((x ** 8).sum())
        e_a2 = u * u * fa2 + n * acc * acc * fx2 * fx2 / max(m, 1)
        e_b2 = u * u * (c * c * fa4 + fb2) + (abs(b) + 2 * abs(c) * a2) ** 2 * e_a2
        e_y2 = u * u * fbx2 + m * acc * acc * fb2 * fx2 / max(m, 1) + a2 * e_b2
        delta2 = u * u * fnew2 + e_y2
        g = _rms_gain(x, a, b, c, rows, cols)
        e2 = g * g * e2 + delta2
    return math.sqrt(e2)


ESTIMATE_FACTOR = 4.0


def orthogonality_report(u: torch.Tensor, o: torch.Tensor, coefficients=DEFAULT_COEFFICIENTS, steps: int = 5,
                         eps: float = 1e-7, dtype: Optional[torch.dtype] = None, name: str = "update",
                         extra_error: float = 0.0) -> OrthoReport:
    """Measure ``o`` (a Newton Schulz result of ``u``, any dtype and device) against float64.
    ``extra_error`` is a Frobenius error already present in ``o`` and allowed on top of the bound
    (``Muon.recover_resolution`` for an update read back from parameters)."""
    if u.dim() != 2 or tuple(o.shape) != tuple(u.shape):
        raise ValueError(f"u and o must be 2D and of one shape, got {tuple(u.shape)} and {tuple(o.shape)}")
    dtype = dtype or o.dtype
    exact, polar, S, hist = exact_iteration(u, coefficients, steps, eps)
    tol = float(S.max()) * max(u.shape) * _UNIT.get(u.dtype, 2.0 ** -52) if S.numel() else 0.0
    r = int((S > tol).sum())
    o64 = o.detach().to(F64).cpu()
    if not bool(torch.isfinite(o64).all()):
        inf = math.inf
        return OrthoReport(name, (int(u.shape[0]), int(u.shape[1])), str(dtype).replace("torch.", ""), r, inf,
                           (inf, inf), inf, 0.0, (0.0, 0.0), inf, 0.0, 0.0, 0.0)
    U, _, Vh = torch.linalg.svd(u.detach().to(F64).cpu(), full_matrices=False)
    polar = U[:, :r] @ Vh[:r]       # the polar factor on the rank of u (null directions are not defined)
    scale = math.sqrt(max(r, 1))
    sv = torch.linalg.svdvals(o64)[:r]
    ex_sv = hist[-1][:r]
    bound, amp = rounding_bound([h[:r] for h in hist], coefficients, u.shape[0], u.shape[1], dtype,
                                input_dtype=u.dtype, full_rank=r == min(u.shape))
    return OrthoReport(
        name=name, shape=(int(u.shape[0]), int(u.shape[1])), dtype=str(dtype).replace("torch.", ""), rank=r,
        ortho_error=float((sv - 1).abs().max()) if r else 0.0,
        sv_range=(float(sv.min()), float(sv.max())) if r else (0.0, 0.0),
        polar_distance=float((o64 - polar).norm()) / scale,
        intended=float((exact - polar).norm()) / scale,
        intended_sv_range=(float(ex_sv.min()), float(ex_sv.max())) if r else (0.0, 0.0),
        rounding=float((o64 - exact).norm()) / scale,
        estimate=(rounding_estimate([h[:r] for h in hist], coefficients, u.shape[0], u.shape[1], dtype,
                                    input_dtype=u.dtype) + extra_error) / scale,
        bound=(bound + extra_error) / scale,
        amplification=amp)


__all__ = ["Muon", "OrthoReport", "MuonOrthogonalityError", "newton_schulz", "orthogonality_report", "exact_iteration", "rounding_bound", "rounding_estimate",
           "lr_ratio", "DEFAULT_COEFFICIENTS", "CONVERGENT_COEFFICIENTS"]
