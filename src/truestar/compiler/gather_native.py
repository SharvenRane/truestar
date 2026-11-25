"""Truestar's own kernels for the data dependent reads of a compiled program.

``index``, ``index_select`` and ``gather`` (embeddings, advanced indexing, the class pick of the
cross entropy and NLL losses, nearest upsampling, the index padding modes, the adaptive max pool's
taps) and ``argmax`` / ``argmin`` (an integer result) used to run as torch steps on the executor's
device. They now run as kernels written here: Triton on CUDA, numba on the CPU. The torch step stays
only for what these kernels do not take (named by ``plan``), and the step's report entry says which
one ran.

One plan covers the three reads. Every output element at coordinates ``c`` (row major over the
output shape) reads the input at

    offset(c) = sum_j c_j * xs_j + sum_k wrap(idx_k[sum_j c_j * is_kj], n_k) * xm_k

where ``xs_j`` is the input's stride carried by output dim ``j`` (0 for a dim made by the index
tensors), ``is_kj`` the stride of index tensor ``k`` over output dim ``j`` (0 where it broadcasts) and
``xm_k`` the input stride of the dim it indexes, of size ``n_k``. ``wrap`` adds ``n_k`` to a negative
index; an index outside ``[-n_k, n_k)`` sets a flag (read once after the run, like the torch step's
check) and is clamped so no kernel reads out of bounds. The kernels copy bits (a float is moved as
the integer of its width), so the result is exactly the input's entries: deterministic, and equal
to the reference wherever the input is.

``argmax`` / ``argmin`` follow the reference's definition: the first index of the extreme, a NaN
counts as the extreme (its first index wins). Every output element scans its row in index order.
"""

from __future__ import annotations

import functools
from typing import List, Optional, Sequence, Tuple

import torch

R_MAX = 6          # output rank the kernels take
K_MAX = 3          # index tensors per read
_BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
_ARG_FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_ARG_FLOATS_CPU = (torch.float32, torch.float64)
GATHER_OPS = ("index", "index_select", "gather")
ARG_OPS = ("argmax", "argmin")
SCATTER_OPS = ("scatter_add",)


# ----------------------------------------------------------------------------- availability


@functools.lru_cache(maxsize=None)
def _have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:     # noqa: BLE001  (not installed, or a broken install: the torch step runs)
        return False


def backend_for(device: str) -> Optional[str]:
    """The kernel family for an executor device, or None (that device keeps the torch step)."""
    if device == "cuda" and _have("triton"):
        return "triton"
    if device == "cpu" and _have("numba"):
        return "numba"
    return None


def static_reason(node, device: str) -> Optional[str]:
    """Why ``node`` keeps the torch step on ``device``, from what is known when compiling, or None
    when Truestar's kernel runs it. Shapes and strides are checked again at run time."""
    op = node.op
    if op not in GATHER_OPS + ARG_OPS + SCATTER_OPS:
        return f"{op} has no Truestar kernel"
    if backend_for(device) is None:
        return f"no Truestar gather kernel for the {device} device"
    if any(o.type.rank > R_MAX for o in node.outputs) or node.inputs[0].type.rank > R_MAX:
        return f"rank over {R_MAX}"
    if op in ARG_OPS:
        dt = str(node.inputs[0].type.dtype).replace("torch.", "")
        ok = ("float32", "float64") if device == "cpu" else ("float16", "bfloat16", "float32", "float64")
        if dt not in ok:
            return f"{op} of {dt}"
        return None
    if op in SCATTER_OPS:
        dt = str(node.inputs[0].type.dtype).replace("torch.", "")
        ok = ("float32", "float64") if device == "cpu" else ("float16", "bfloat16", "float32", "float64")
        if dt not in ok:
            return f"{op} of {dt}"
        if str(node.inputs[1].type.dtype).replace("torch.", "") not in ("int64", "int32", "int16", "int8"):
            return "an index tensor that is not an integer tensor"
        return None
    idx = [x for x in node.inputs[1:] if x is not None]
    if len(idx) > K_MAX:
        return f"more than {K_MAX} index tensors"
    if any(str(x.type.dtype).replace("torch.", "") not in ("int64", "int32", "int16", "int8") for x in idx):
        return "an index tensor that is not an integer tensor"
    return None


# ----------------------------------------------------------------------------- the plan of a read


class Plan:
    """How one read addresses its input (see the module text). ``pos``: where its index tensors are
    among the op's index arguments; ``flat``: index_select's index is read flattened; ``launch``:
    the kernel's launch parameters, computed on the first call of this signature."""
    __slots__ = ("out_shape", "xs", "idx", "istr", "xm", "n", "pos", "flat", "launch")

    def __init__(self, out_shape, xs, idx, istr, xm, n, pos=(0,), flat=False):
        self.out_shape, self.xs, self.idx, self.istr, self.xm, self.n = out_shape, xs, idx, istr, xm, n
        self.pos, self.flat, self.launch = tuple(pos), flat, None

    def index_tensors(self, idx):
        ts = [idx[k] for k in self.pos]
        return [ts[0].reshape(-1)] if self.flat else ts


def _bcast_strides(t: torch.Tensor, rank: int) -> List[int]:
    """Strides of ``t`` broadcast (right aligned) to ``rank`` dims: 0 where it has size 1 or no dim."""
    st = [0] * (rank - t.dim()) + [0 if s == 1 else st for s, st in zip(t.shape, t.stride())]
    return st


def plan(op: str, attrs, x: torch.Tensor, idx: Sequence[Optional[torch.Tensor]]) -> Tuple[Optional[Plan], str]:
    """The plan of one read, or (None, why the torch step runs it)."""
    rank = x.dim()
    if op == "index_select":
        (i,) = idx
        d = attrs["dim"] % rank
        if i.dim() > 1:
            if not i.is_contiguous():
                return None, "a strided index tensor of rank over 1"
            i = i.reshape(-1)
        m = i.numel()
        out = tuple(x.shape[:d]) + (m,) + tuple(x.shape[d + 1:])
        xs = list(x.stride())
        xs[d] = 0
        ist = [0] * rank
        ist[d] = i.stride(0) if i.dim() == 1 else 0
        return Plan(out, xs, [i], [ist], [x.stride(d)], [x.shape[d]], flat=idx[0].dim() > 1), ""
    if op == "gather":
        (i,) = idx
        d = attrs["dim"] % rank
        if i.dim() != rank or any(i.shape[k] > x.shape[k] for k in range(rank) if k != d):
            return None, "an index tensor whose shape does not fit the input"
        xs = list(x.stride())
        xs[d] = 0
        return Plan(tuple(i.shape), xs, [i], [list(i.stride())], [x.stride(d)], [x.shape[d]]), ""
    # index: torch's advanced indexing
    if len(idx) > rank:
        return None, "more index entries than input dims"
    pos = [k for k, t in enumerate(idx) if t is not None]
    ts = [idx[k] for k in pos]
    if not ts:
        return None, "no index tensor"
    try:
        bs = tuple(torch.broadcast_shapes(*[t.shape for t in ts]))
    except RuntimeError:
        return None, "index tensors that do not broadcast"
    rest = [k for k in range(rank) if k not in pos]
    adjacent = pos == list(range(pos[0], pos[-1] + 1))
    if adjacent:
        before = [k for k in rest if k < pos[0]]
        after = [k for k in rest if k > pos[-1]]
        out = tuple(x.shape[k] for k in before) + bs + tuple(x.shape[k] for k in after)
        xs = [x.stride(k) for k in before] + [0] * len(bs) + [x.stride(k) for k in after]
        lead, trail = len(before), len(after)
    else:
        out = bs + tuple(x.shape[k] for k in rest)
        xs = [0] * len(bs) + [x.stride(k) for k in rest]
        lead, trail = 0, len(rest)
    istr = [[0] * lead + _bcast_strides(t, len(bs)) + [0] * trail for t in ts]
    return Plan(out, xs, ts, istr, [x.stride(k) for k in pos], [x.shape[k] for k in pos], pos=pos), ""


def _bits(t: torch.Tensor) -> torch.Tensor:
    """``t`` viewed as unsigned or signed integers of its width (a bit copy cannot change a NaN)."""
    if t.dtype == torch.bool:
        return t.view(torch.uint8)
    return t.view(_BITS[t.element_size()])


def _padded(vals: Sequence[int], fill: int) -> List[int]:
    return [fill] * (R_MAX - len(vals)) + list(vals)


# ----------------------------------------------------------------------------- Triton


_K = {}


def _triton_gather():
    if "g" in _K:
        return _K["g"]
    import triton
    import triton.language as tl

    @triton.jit
    def ts_gather(px, po, pflag, p0, p1, p2, pm, rows, C,
                  K: tl.constexpr, ROWIDX: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
        # pm: S0..S4 (row dims), X0..X4, XC (input strides), then per index tensor 5 row strides,
        # its stride over the last dim, the input stride it moves and the size it indexes.
        # ROWIDX: every index tensor is constant along the last dim (one index load per row).
        r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
        c = (tl.program_id(1) * BC + tl.arange(0, BC)).to(tl.int64)
        rm = r < rows
        msk = rm[:, None] & (c < C)[None, :]
        t = r
        S4 = tl.load(pm + 4)
        c4 = t % S4
        t = t // S4
        S3 = tl.load(pm + 3)
        c3 = t % S3
        t = t // S3
        S2 = tl.load(pm + 2)
        c2 = t % S2
        t = t // S2
        S1 = tl.load(pm + 1)
        c1 = t % S1
        c0 = t // S1
        off = (c0 * tl.load(pm + 5) + c1 * tl.load(pm + 6) + c2 * tl.load(pm + 7) + c3 * tl.load(pm + 8)
               + c4 * tl.load(pm + 9))[:, None] + c[None, :] * tl.load(pm + 10)
        bad = tl.zeros((BR, BC), dtype=tl.int1)
        if K >= 1:
            io = c0 * tl.load(pm + 11) + c1 * tl.load(pm + 12) + c2 * tl.load(pm + 13) \
                + c3 * tl.load(pm + 14) + c4 * tl.load(pm + 15)
            n = tl.load(pm + 18)
            if ROWIDX:
                i = tl.load(p0 + io, mask=rm, other=0).to(tl.int64)
                bad = bad | (rm & ((i < -n) | (i >= n)))[:, None]
                i = tl.where(i < 0, i + n, i)
                i = tl.minimum(tl.maximum(i, 0), n - 1)
                off = off + (i * tl.load(pm + 17))[:, None]
            else:
                i = tl.load(p0 + io[:, None] + c[None, :] * tl.load(pm + 16), mask=msk, other=0).to(tl.int64)
                bad = bad | (msk & ((i < -n) | (i >= n)))
                i = tl.where(i < 0, i + n, i)
                i = tl.minimum(tl.maximum(i, 0), n - 1)
                off = off + i * tl.load(pm + 17)
        if K >= 2:
            io = c0 * tl.load(pm + 19) + c1 * tl.load(pm + 20) + c2 * tl.load(pm + 21) \
                + c3 * tl.load(pm + 22) + c4 * tl.load(pm + 23)
            n = tl.load(pm + 26)
            if ROWIDX:
                i = tl.load(p1 + io, mask=rm, other=0).to(tl.int64)
                bad = bad | (rm & ((i < -n) | (i >= n)))[:, None]
                i = tl.where(i < 0, i + n, i)
                i = tl.minimum(tl.maximum(i, 0), n - 1)
                off = off + (i * tl.load(pm + 25))[:, None]
            else:
                i = tl.load(p1 + io[:, None] + c[None, :] * tl.load(pm + 24), mask=msk, other=0).to(tl.int64)
                bad = bad | (msk & ((i < -n) | (i >= n)))
                i = tl.where(i < 0, i + n, i)
                i = tl.minimum(tl.maximum(i, 0), n - 1)
                off = off + i * tl.load(pm + 25)
        if K >= 3:
            io = c0 * tl.load(pm + 27) + c1 * tl.load(pm + 28) + c2 * tl.load(pm + 29) \
                + c3 * tl.load(pm + 30) + c4 * tl.load(pm + 31)
            n = tl.load(pm + 34)
            if ROWIDX:
                i = tl.load(p2 + io, mask=rm, other=0).to(tl.int64)
                bad = bad | (rm & ((i < -n) | (i >= n)))[:, None]
                i = tl.where(i < 0, i + n, i)
                i = tl.minimum(tl.maximum(i, 0), n - 1)
                off = off + (i * tl.load(pm + 33))[:, None]
            else:
                i = tl.load(p2 + io[:, None] + c[None, :] * tl.load(pm + 32), mask=msk, other=0).to(tl.int64)
                bad = bad | (msk & ((i < -n) | (i >= n)))
                i = tl.where(i < 0, i + n, i)
                i = tl.minimum(tl.maximum(i, 0), n - 1)
                off = off + i * tl.load(pm + 33)
        v = tl.load(px + off, mask=msk, other=0)
        tl.store(po + r[:, None] * C + c[None, :], v, mask=msk)
        if K >= 1:
            fz = tl.zeros((BR, BC), dtype=tl.int64)
            tl.store(pflag + fz, (fz + 1).to(tl.int8), mask=bad)

    _K["g"] = ts_gather
    return ts_gather


def _triton_arg():
    if "a" in _K:
        return _K["a"]
    import triton
    import triton.language as tl

    @triton.jit
    def ts_arg_extreme(px, po, pm, total, N,
                       LARGEST: tl.constexpr, F64: tl.constexpr, BO: tl.constexpr, BN: tl.constexpr):
        S0 = tl.load(pm + 0)
        S1 = tl.load(pm + 1)
        S2 = tl.load(pm + 2)
        S3 = tl.load(pm + 3)
        S4 = tl.load(pm + 4)
        X0 = tl.load(pm + 5)
        X1 = tl.load(pm + 6)
        X2 = tl.load(pm + 7)
        X3 = tl.load(pm + 8)
        X4 = tl.load(pm + 9)
        XN = tl.load(pm + 10)
        o = (tl.program_id(0) * BO + tl.arange(0, BO)).to(tl.int64)
        om = o < total
        t = o
        c4 = t % S4
        t = t // S4
        c3 = t % S3
        t = t // S3
        c2 = t % S2
        t = t // S2
        c1 = t % S1
        c0 = t // S1
        base = c0 * X0 + c1 * X1 + c2 * X2 + c3 * X3 + c4 * X4
        BIG = N
        best_nan = tl.full((BO,), BIG, tl.int64)
        best_idx = tl.full((BO,), BIG, tl.int64)
        if F64:
            best_val = tl.full((BO,), float("-inf"), tl.float64)
        else:
            best_val = tl.full((BO,), float("-inf"), tl.float32)
        for n0 in range(0, N, BN):
            n = n0 + tl.arange(0, BN).to(tl.int64)
            m = om[:, None] & (n < N)[None, :]
            x = tl.load(px + base[:, None] + n[None, :] * XN, mask=m, other=0.0)
            if F64:
                x = x.to(tl.float64)
            else:
                x = x.to(tl.float32)
            if not LARGEST:
                x = -x
            isnan = (x != x) & m
            nn = n[None, :] + tl.zeros((BO, BN), tl.int64)
            best_nan = tl.minimum(best_nan, tl.min(tl.where(isnan, nn, BIG), axis=1))
            v = tl.where(isnan | (~m), float("-inf"), x)
            cm = tl.max(v, axis=1)
            first = tl.min(tl.where((v == cm[:, None]) & m & (~isnan), nn, BIG), axis=1)
            take = (cm > best_val) | ((best_idx == BIG) & (first < BIG))
            best_val = tl.where(take, cm, best_val)
            best_idx = tl.where(take, first, best_idx)
        res = tl.where(best_nan < BIG, best_nan, best_idx)
        tl.store(po + o, res, mask=om)

    _K["a"] = ts_arg_extreme
    return ts_arg_extreme


def _triton_scatter():
    if "s" in _K:
        return _K["s"]
    import triton
    import triton.language as tl

    @triton.jit
    def ts_scatter_add(po, pi, ps, pflag, pm, cols, M, F64: tl.constexpr, BCOL: tl.constexpr):
        # one lane per column (the coordinates off the scatter dim); every lane walks its index in
        # order, so each sum is taken in the same order on every run and every device
        S0 = tl.load(pm + 0)
        S1 = tl.load(pm + 1)
        S2 = tl.load(pm + 2)
        S3 = tl.load(pm + 3)
        S4 = tl.load(pm + 4)
        col = (tl.program_id(0) * BCOL + tl.arange(0, BCOL)).to(tl.int64)
        cm = col < cols
        t = col
        c4 = t % S4
        t = t // S4
        c3 = t % S3
        t = t // S3
        c2 = t % S2
        t = t // S2
        c1 = t % S1
        c0 = t // S1
        ob = c0 * tl.load(pm + 5) + c1 * tl.load(pm + 6) + c2 * tl.load(pm + 7) + c3 * tl.load(pm + 8) \
            + c4 * tl.load(pm + 9)
        ib = c0 * tl.load(pm + 10) + c1 * tl.load(pm + 11) + c2 * tl.load(pm + 12) + c3 * tl.load(pm + 13) \
            + c4 * tl.load(pm + 14)
        sb = c0 * tl.load(pm + 15) + c1 * tl.load(pm + 16) + c2 * tl.load(pm + 17) + c3 * tl.load(pm + 18) \
            + c4 * tl.load(pm + 19)
        OD = tl.load(pm + 20)
        ID = tl.load(pm + 21)
        QD = tl.load(pm + 22)
        N = tl.load(pm + 23)
        bad = tl.zeros((BCOL,), dtype=tl.int1)
        for m in range(0, M):
            j = tl.load(pi + ib + m * ID, mask=cm, other=0).to(tl.int64)
            bad = bad | (cm & ((j < -N) | (j >= N)))
            j = tl.where(j < 0, j + N, j)
            j = tl.minimum(tl.maximum(j, 0), N - 1)
            o = ob + j * OD
            v = tl.load(ps + sb + m * QD, mask=cm, other=0.0)
            cur = tl.load(po + o, mask=cm, other=0.0)
            if F64:
                acc = cur.to(tl.float64) + v.to(tl.float64)
            else:
                acc = cur.to(tl.float32) + v.to(tl.float32)
            tl.store(po + o, acc.to(po.dtype.element_ty), mask=cm)
        fz = tl.zeros((BCOL,), dtype=tl.int64)
        tl.store(pflag + fz, (fz + 1).to(tl.int8), mask=bad)

    _K["s"] = ts_scatter_add
    return ts_scatter_add


def _np2(v: int) -> int:
    p = 1
    while p < v:
        p *= 2
    return p


def _gather_triton(p: Plan, x: torch.Tensor, idx, out: torch.Tensor, flag: torch.Tensor) -> None:
    if p.launch is None:
        import triton
        shape = list(p.out_shape)
        C = shape[-1]
        rows = out.numel() // C
        S = _padded(shape[:-1], 1)[1:]                         # 5 row dims, padded at the front
        X = _padded(p.xs[:-1], 0)[1:] + [p.xs[-1]]
        args = []
        for k in range(K_MAX):
            if k < len(p.istr):
                st = p.istr[k]
                args += _padded(st[:-1], 0)[1:] + [st[-1], p.xm[k], p.n[k]]
            else:
                args += [0] * 7 + [1]
        pm = _meta(S + X + args, x.device)
        if pm is None:
            raise _NoKernel("a new shape while a CUDA graph is captured")
        rowidx = all(st[-1] == 0 for st in p.istr)
        BC = min(_np2(C), 1024 if rowidx else 256)
        BR = max(1, min(1024, 4096 // BC))
        p.launch = ((triton.cdiv(rows, BR), triton.cdiv(C, BC)), pm, rows, C, rowidx, BR, BC,
                    8 if BR * BC >= 4096 and rowidx else 4)
    grid, pm, rows, C, rowidx, BR, BC, nw = p.launch
    i0 = idx[0] if len(idx) > 0 else flag
    i1 = idx[1] if len(idx) > 1 else flag
    i2 = idx[2] if len(idx) > 2 else flag
    _triton_gather()[grid](_bits(x), _bits(out), flag, i0, i1, i2, pm, rows, C, K=len(idx), ROWIDX=rowidx,
                           BR=BR, BC=BC, num_warps=nw)


_META = {}


def _meta(vals, device):
    """The shape metadata of a launch as a device tensor, made once per distinct shape (no host to
    device copy on later calls; none at all while a CUDA graph is being captured)."""
    key = (device, tuple(int(v) for v in vals))
    t = _META.get(key)
    if t is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return None
        if len(_META) > 4096:
            _META.clear()
        t = _META[key] = torch.tensor(key[1], dtype=torch.int64, device=device)
    return t


def _arg_triton(x, out, largest, dim) -> None:
    import triton
    rank = x.dim()
    if dim is None:
        if not x.is_contiguous():
            raise _NoKernel("a strided input without a dim")
        sizes, strides, N, XN = [], [], x.numel(), 1
    else:
        d = dim % rank
        sizes = [x.shape[k] for k in range(rank) if k != d]
        strides = [x.stride(k) for k in range(rank) if k != d]
        N, XN = x.shape[d], x.stride(d)
    total = out.numel()
    if total == 0 or N == 0:
        raise _NoKernel("an empty input")
    S = _padded(sizes, 1)[1:]
    X = _padded(strides, 0)[1:]
    pm = _meta(S + X + [XN], x.device)
    if pm is None:
        raise _NoKernel("a new shape while a CUDA graph is captured")
    # few long rows (a vocabulary): one program per row with wide chunks; many short rows: a block
    BN = min(_np2(N), 4096 if total < 2048 else 512)
    BO = max(1, min(256, 4096 // BN))
    while BO > 1 and triton.cdiv(total, BO) < 512:
        BO //= 2
    grid = (triton.cdiv(total, BO),)
    _triton_arg()[grid](x, out, pm, total, N, LARGEST=bool(largest), F64=x.dtype == torch.float64,
                        BO=BO, BN=BN, num_warps=8 if BO * BN >= 4096 else 4)


# ----------------------------------------------------------------------------- numba


@functools.lru_cache(maxsize=None)
def _numba_kernels():
    import numba
    import numpy as np

    @numba.njit(parallel=True, cache=True)
    def ts_gather_cpu(xf, of, flag, i0, i1, i2, rows, C, S, X, XC, IS, ISC, IM, IN, K, rowidx):
        for r in numba.prange(rows):
            t = r + 0
            base = 0
            b0 = 0
            b1 = 0
            b2 = 0
            for j in range(S.shape[0] - 1, -1, -1):
                cj = t % S[j]
                t = t // S[j]
                base += cj * X[j]
                b0 += cj * IS[0, j]
                b1 += cj * IS[1, j]
                b2 += cj * IS[2, j]
            if rowidx:        # every index constant along the last dim: one read per row, a row copy
                off = base
                for k in range(K):
                    if k == 0:
                        i = np.int64(i0[b0])
                    elif k == 1:
                        i = np.int64(i1[b1])
                    else:
                        i = np.int64(i2[b2])
                    n = IN[k]
                    if i < -n or i >= n:
                        flag[0] = 1
                    if i < 0:
                        i += n
                    if i < 0:
                        i = 0
                    if i > n - 1:
                        i = n - 1
                    off += i * IM[k]
                if XC == 1:
                    of[r * C:(r + 1) * C] = xf[off:off + C]
                else:
                    for c in range(C):
                        of[r * C + c] = xf[off + c * XC]
                continue
            for c in range(C):
                off = base + c * XC
                for k in range(K):
                    if k == 0:
                        i = np.int64(i0[b0 + c * ISC[0]])
                    elif k == 1:
                        i = np.int64(i1[b1 + c * ISC[1]])
                    else:
                        i = np.int64(i2[b2 + c * ISC[2]])
                    n = IN[k]
                    if i < -n or i >= n:
                        flag[0] = 1
                    if i < 0:
                        i += n
                    if i < 0:
                        i = 0
                    if i > n - 1:
                        i = n - 1
                    off += i * IM[k]
                of[r * C + c] = xf[off]

    @numba.njit(parallel=True, cache=True)
    def ts_arg_cpu(xf, of, total, N, S, X, XN, largest):
        for o in numba.prange(total):
            t = o + 0
            base = 0
            for j in range(S.shape[0] - 1, -1, -1):
                cj = t % S[j]
                t = t // S[j]
                base += cj * X[j]
            best = -1
            bv = 0.0
            for n in range(N):
                v = xf[base + n * XN]
                if v != v:
                    best = n
                    break
                if not largest:
                    v = -v
                if best < 0 or v > bv:
                    best = n
                    bv = v
            of[o] = best

    @numba.njit(parallel=True, cache=True)
    def ts_scatter_cpu(of, ixf, sf, flag, cols, M, S, OS, IS, QS, OD, ID, QD, N):
        for col in numba.prange(cols):
            t = col + 0
            ob = 0
            ib = 0
            sb = 0
            for j in range(S.shape[0] - 1, -1, -1):
                cj = t % S[j]
                t = t // S[j]
                ob += cj * OS[j]
                ib += cj * IS[j]
                sb += cj * QS[j]
            for m in range(M):
                k = np.int64(ixf[ib + m * ID])
                if k < -N or k >= N:
                    flag[0] = 1
                if k < 0:
                    k += N
                if k < 0:
                    k = 0
                if k > N - 1:
                    k = N - 1
                o = ob + k * OD
                of[o] = of[o] + sf[sb + m * QD]

    return ts_gather_cpu, ts_arg_cpu, ts_scatter_cpu




def _flat(t: torch.Tensor):
    """A 1D numpy view of every element ``t`` can address from its first one (no copy)."""
    span = 1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride())) if t.numel() else 1
    return torch.as_strided(t, (span,), (1,), t.storage_offset()).numpy()


def _gather_numba(p: Plan, x: torch.Tensor, idx, out: torch.Tensor, flag: torch.Tensor) -> None:
    import numpy as np
    if p.launch is None:
        shape = list(p.out_shape)
        K = len(p.istr)
        S = np.array(shape[:-1] or [1], np.int64)
        X = np.array(p.xs[:-1] or [0], np.int64)
        IS = np.zeros((K_MAX, len(S)), np.int64)
        ISC = np.zeros(K_MAX, np.int64)
        IM = np.zeros(K_MAX, np.int64)
        IN = np.ones(K_MAX, np.int64)
        for k in range(K):
            if len(shape) > 1:
                IS[k, :] = p.istr[k][:-1]
            ISC[k] = p.istr[k][-1]
            IM[k], IN[k] = p.xm[k], p.n[k]
        p.launch = (out.numel() // shape[-1], shape[-1], S, X, p.xs[-1], IS, ISC, IM, IN, K,
                    bool(K) and all(int(v) == 0 for v in ISC[:K]))
    rows, C, S, X, XC, IS, ISC, IM, IN, K, rowidx = p.launch
    dummy = np.zeros(1, np.int64)
    ix = [_flat(idx[k]) if k < K else dummy for k in range(K_MAX)]
    _numba_kernels()[0](_flat(_bits(x)), _bits(out).reshape(-1).numpy(), flag.numpy(), ix[0], ix[1], ix[2],
                        rows, C, S, X, XC, IS, ISC, IM, IN, K, rowidx)


def _arg_numba(x, out, largest, dim) -> None:
    import numpy as np
    a = _numba_kernels()[1]
    rank = x.dim()
    if dim is None:
        if not x.is_contiguous():
            raise _NoKernel("a strided input without a dim")
        sizes, strides, N, XN = [1], [0], x.numel(), 1
    else:
        d = dim % rank
        sizes = [x.shape[k] for k in range(rank) if k != d] or [1]
        strides = [x.stride(k) for k in range(rank) if k != d] or [0]
        N, XN = x.shape[d], x.stride(d)
    total = out.numel()
    if total == 0 or N == 0:
        raise _NoKernel("an empty input")
    a(_flat(x), out.reshape(-1).numpy(), total, N, np.array(sizes, np.int64), np.array(strides, np.int64),
      XN, bool(largest))


# ----------------------------------------------------------------------------- entry points


class _NoKernel(Exception):
    """This call's tensors are outside what the kernels take: the torch step runs it."""


_PLANS = {}


def gather(op: str, attrs, args, backend: str):
    """(result, flag, index read first) of one read by Truestar's kernel, or None when the torch step
    must run it (the caller then does). ``flag`` is a 1 element int8 tensor, nonzero when an index
    was out of range. Plans are kept per signature (shapes, strides, dtypes, devices)."""
    x, idx = args[0], args[1:]
    key = (op, attrs.get("dim"), backend, x.dtype, x.device, tuple(x.shape), x.stride(),
           tuple(None if t is None else (t.dtype, t.device, tuple(t.shape), t.stride()) for t in idx))
    p = _PLANS.get(key)
    if p is None:
        p = _PLANS[key] = _new_plan(op, attrs, x, idx) or False
        if len(_PLANS) > 4096:
            _PLANS.clear()
    if p is False:
        return None
    ts = p.index_tensors(idx)
    out = torch.empty(p.out_shape, dtype=x.dtype, device=x.device)
    flag = torch.zeros(1, dtype=torch.int8, device=x.device)
    if out.numel() == 0:
        return out, flag, ts[0]
    try:
        if backend == "triton":
            _gather_triton(p, x, ts, out, flag)
        else:
            _gather_numba(p, x, ts, out, flag)
    except _NoKernel:
        return None
    return out, flag, ts[0]


def _new_plan(op, attrs, x, idx):
    if x.device.type == "meta" or x.numel() == 0 or any(t is not None and t.device != x.device for t in idx):
        return None
    if any(t is not None and (t.dtype in (torch.bool, torch.uint8) or t.dtype.is_floating_point) for t in idx):
        return None
    p, _why = plan(op, attrs, x, list(idx))
    if p is None or len(p.out_shape) == 0 or len(p.out_shape) > R_MAX or len(p.idx) > K_MAX:
        return None
    p.idx = None          # the plan keeps no tensor of the call that made it
    return p


def arg_extreme(op: str, attrs, x: torch.Tensor, shape, backend: str):
    """argmax / argmin by Truestar's kernel, shaped like the reference's result, or None."""
    if x.device.type == "meta" or x.dim() == 0 or x.dtype not in (_ARG_FLOATS if backend == "triton"
                                                                  else _ARG_FLOATS_CPU):
        return None
    dim = attrs.get("dim")
    keep = bool(attrs.get("keepdim", False))
    rank = x.dim()
    if dim is None:
        oshape = (1,) * rank if keep else ()
    else:
        d = dim % rank
        oshape = tuple(1 if k == d else s for k, s in enumerate(x.shape)) if keep else \
            tuple(s for k, s in enumerate(x.shape) if k != d)
    out = torch.empty(oshape, dtype=torch.int64, device=x.device)
    try:
        if backend == "triton":
            _arg_triton(x, out, op == "argmax", dim)
        else:
            _arg_numba(x, out, op == "argmax", dim)
    except _NoKernel:
        return None
    return out


def scatter_add(attrs, args, backend: str):
    """(result, flag, index) of ``scatter_add`` by Truestar's kernels (a copy of the input, then every
    column's index walked in order: the sums of repeated indices are taken in index order, the CPU
    order of torch, on every device and run), or None when the torch step must run it."""
    x, i, src = args
    if x.device.type == "meta" or i.device != x.device or src.device != x.device or x.dim() == 0 \
            or x.dim() > R_MAX or i.dtype in (torch.bool, torch.uint8) or i.dtype.is_floating_point:
        return None
    if x.dtype not in (_ARG_FLOATS if backend == "triton" else _ARG_FLOATS_CPU) or src.dtype != x.dtype:
        return None
    rank = x.dim()
    d = attrs["dim"] % rank
    if i.dim() != rank or src.dim() != rank or any(i.shape[k] > src.shape[k] for k in range(rank)) \
            or any(i.shape[k] > x.shape[k] for k in range(rank) if k != d):
        return None
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    flag = torch.zeros(1, dtype=torch.int8, device=x.device)
    if out.numel() == 0:
        return out, flag, i
    copy = Plan(tuple(x.shape), list(x.stride()), [], [], [], [])
    off = [k for k in range(rank) if k != d]
    S = [i.shape[k] for k in off]
    OS = [out.stride(k) for k in off]
    IS = [i.stride(k) for k in off]
    QS = [src.stride(k) for k in off]
    cols = 1
    for v in S:
        cols *= v
    M = i.shape[d]
    if M > 64 * max(cols, 1):
        # one lane walks each column in order: a long index over few columns (token counts over a
        # vocabulary) is slower than torch's deterministic scatter (measured 0.55x on an L4)
        return None
    try:
        if backend == "triton":
            import triton
            _gather_triton(copy, x, [], out, flag)
            if cols and M:
                pm = _meta(_padded(S, 1)[1:] + _padded(OS, 0)[1:] + _padded(IS, 0)[1:] + _padded(QS, 0)[1:]
                           + [out.stride(d), i.stride(d), src.stride(d), x.shape[d]], x.device)
                if pm is None:
                    raise _NoKernel("a new shape while a CUDA graph is captured")
                BCOL = min(256, max(32, _np2(cols)))
                _triton_scatter()[(triton.cdiv(cols, BCOL),)](out, i, src, flag, pm, cols, M,
                                                               F64=x.dtype == torch.float64, BCOL=BCOL,
                                                               num_warps=max(1, min(8, BCOL // 32)))
        else:
            import numpy as np
            _gather_numba(copy, x, [], out, flag)
            if cols and M:
                sc = _numba_kernels()[2]
                sc(out.reshape(-1).numpy(), _flat(i), _flat(src), flag.numpy(), cols, M,
                   np.array(S or [1], np.int64), np.array(OS or [0], np.int64), np.array(IS or [0], np.int64),
                   np.array(QS or [0], np.int64), out.stride(d), i.stride(d), src.stride(d), x.shape[d])
    except _NoKernel:
        return None
    return out, flag, i
