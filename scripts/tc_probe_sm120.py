"""Measure the local GPU's matrix unit and compare it bit for bit with truestar.devsim.tensorcore.

Usage (one short GPU run per invocation):

    python scripts/tc_probe_sm120.py fit --dtype fp16
    python scripts/tc_probe_sm120.py validate --dtype bf16 --k 32 --count 1000000 --path triton
    python scripts/tc_probe_sm120.py features

``fit`` runs a designed ensemble through a one tile Triton kernel and scores every parameter set
of a grid (group size, fraction bits, cut, final rounding, exponent floor); ``validate`` compares
one unit against the GPU on random dot products through Triton ``tl.dot`` (one MMA chain, the
accumulator loaded from C), cuBLAS (``torch.mm(..., out_dtype=float32)``) or ``torch._scaled_mm``
for fp8; ``features`` prints the paper style probe vectors and what the GPU returned.
Results are appended as JSON lines to docs/data/tensor_core_sm120.jsonl.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from truestar.devsim.tensorcore import MatrixUnit, unit_matmul  # noqa: E402
from truestar.devsim.tensorcore import unit_for  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "data", "tensor_core_sm120.jsonl")
TDT = {"fp16": torch.float16, "bf16": torch.bfloat16, "e4m3": torch.float8_e4m3fn,
       "e5m2": torch.float8_e5m2, "tf32": torch.float32}

# ----------------------------------------------------------------------------- GPU paths

_KERNEL = None


def _triton_kernel():
    global _KERNEL
    if _KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def tile_mma(a_ptr, b_ptr, c_ptr, d_ptr, K: tl.constexpr, BM: tl.constexpr,
                     BN: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr,
                     ACC16: tl.constexpr):
            # a: [B, BM, K], b: [B, K, BN], c, d: [B, BM, BN]; one program per batch entry. The
            # accumulator starts at C and walks K in tiles of BK, in order.
            pid = tl.program_id(0)
            rm = tl.arange(0, BM)
            rn = tl.arange(0, BN)
            rk = tl.arange(0, BK)
            acc = tl.load(c_ptr + pid * BM * BN + rm[:, None] * BN + rn[None, :])
            if ACC16:
                acc = acc.to(tl.float16)
            for k0 in tl.static_range(0, K, BK):
                a = tl.load(a_ptr + pid * BM * K + rm[:, None] * K + (k0 + rk)[None, :])
                b = tl.load(b_ptr + pid * K * BN + (k0 + rk)[:, None] * BN + rn[None, :])
                if ACC16:
                    acc = tl.dot(a, b, acc, input_precision=PREC, out_dtype=tl.float16)
                else:
                    acc = tl.dot(a, b, acc, input_precision=PREC)
            tl.store(d_ptr + pid * BM * BN + rm[:, None] * BN + rn[None, :], acc.to(tl.float32))

        _KERNEL = tile_mma
    return _KERNEL


def gpu_triton(a, b, c, dtype, bk=None, acc16=False):
    """a [B, 16, K], b [B, K, 16], c [B, 16, 16] float64 numpy -> float32 numpy via tl.dot."""
    bsz, bm, k = a.shape
    bn = b.shape[-1]
    bk = bk or k
    ta = torch.from_numpy(a).to(TDT[dtype]).cuda().contiguous()
    tb = torch.from_numpy(b).to(TDT[dtype]).cuda().contiguous()
    tc = torch.from_numpy(c).to(torch.float32).cuda().contiguous()
    td = torch.empty_like(tc)
    _triton_kernel()[(bsz,)](ta, tb, tc, td, K=k, BM=bm, BN=bn, BK=bk,
                            PREC="tf32" if dtype == "tf32" else "ieee", ACC16=acc16)
    torch.cuda.synchronize()
    return td.cpu().numpy()


def gpu_cublas(a2, b2, dtype):
    ta = torch.from_numpy(a2).to(TDT[dtype]).cuda()
    tb = torch.from_numpy(b2).to(TDT[dtype]).cuda()
    torch.backends.cuda.matmul.allow_tf32 = dtype == "tf32"  # tf32 inputs are already rounded
    try:
        return torch.mm(ta, tb, out_dtype=torch.float32).cpu().numpy()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False


def gpu_scaled_mm(a2, b2, dtype, fast):
    ta = torch.from_numpy(a2).to(TDT[dtype]).cuda()
    tb = torch.from_numpy(b2.T.copy()).to(TDT[dtype]).cuda().t()  # column major B
    one = torch.tensor(1.0, device="cuda")
    return torch._scaled_mm(ta, tb, one, one, out_dtype=torch.float32,
                            use_fast_accum=fast).cpu().numpy()


# ----------------------------------------------------------------------------- inputs


def _q(x, dtype):
    if dtype == "tf32":  # round to 10 fraction bits, nearest even, through the fp32 bit pattern
        u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
        u = (u + 0xFFF + ((u >> 13) & 1)) & ~np.uint64(0x1FFF)
        return u.astype(np.uint32).view(np.float32).astype(np.float64)
    return torch.from_numpy(np.ascontiguousarray(x)).to(TDT[dtype]).to(torch.float64).numpy()


def ensemble(rng, dtype, shape_a, shape_b, shape_c, kind):
    """Random inputs of one ensemble kind, already representable in the input format."""
    if dtype in ("e4m3", "e5m2"):
        lo, hi = (-6, 7) if dtype == "e4m3" else (-12, 14)
    elif dtype == "fp16":
        lo, hi = -10, 10
    elif dtype == "tf32":
        lo, hi = -40, 40
    else:
        lo, hi = -40, 40

    def wide(shape, lo, hi):
        sig = 1 + rng.random(shape)
        return np.where(rng.random(shape) < .5, -1, 1) * sig * np.exp2(rng.integers(lo, hi, shape))

    if kind == "normal":
        a, b = rng.standard_normal(shape_a), rng.standard_normal(shape_b)
        c = np.zeros(shape_c)
    elif kind == "normal_c":
        a, b = rng.standard_normal(shape_a), rng.standard_normal(shape_b)
        c = rng.standard_normal(shape_c) * np.exp2(rng.integers(-4, 6, shape_c))
    elif kind == "wide":
        a, b = wide(shape_a, lo, hi), wide(shape_b, lo, hi)
        c = np.where(rng.random(shape_c) < .5, 0, wide(shape_c, 2 * lo, 2 * hi))
    elif kind == "cancel":
        # pairs that nearly cancel, the remainder below the largest term's grid
        a = rng.standard_normal(shape_a)
        b = rng.standard_normal(shape_b)
        a[..., 1::2] = -a[..., 0::2] * (1 + np.exp2(-rng.integers(6, 12, a[..., 1::2].shape)))
        c = rng.standard_normal(shape_c) * np.exp2(-rng.integers(0, 20, shape_c))
    elif kind == "tiny":  # products in the fp32 subnormal range (bf16 only reaches there)
        # exponents -82 to -67: every product below 2^-133, where the alignment floor acts
        a = wide(shape_a, -82, -66)
        b = wide(shape_b, -82, -66)
        c = np.where(rng.random(shape_c) < .7, 0, wide(shape_c, -149, -128))
    else:
        raise KeyError(kind)
    c = c.astype(np.float32).astype(np.float64)
    return _q(a, dtype), _q(b, dtype), c


KINDS = {"fp16": ["normal", "normal_c", "wide", "cancel"],
         "bf16": ["normal", "normal_c", "wide", "cancel", "tiny"],
         "e4m3": ["normal", "normal_c", "wide", "cancel"],
         "e5m2": ["normal", "normal_c", "wide", "cancel"],
         "tf32": ["normal", "normal_c", "wide", "cancel", "tiny"]}


def model_batched(a, b, c, unit, dtype):
    return unit_matmul(a, b, c, unit, dtype)


model_flat = model_batched


def split_k(a, b, unit, dtype, slices):
    """K cut into equal slices, each slice a unit chain from zero, slices added in order in fp32
    with round to nearest even (a split K GEMM schedule)."""
    from truestar.devsim.tensorcore import FORMATS, round_to
    k = a.shape[-1]
    step = k // max(1, slices)
    acc = None
    for k0 in range(0, k, step):
        part = unit_matmul(a[:, k0:k0 + step], b[k0:k0 + step], None, unit, dtype)
        acc = part if acc is None else round_to(acc + part, FORMATS["fp32"], "rne")
    return acc


def bits_equal(x, y):
    return x.astype(np.float32).view(np.int32) == y.astype(np.float32).view(np.int32)


def record(row):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    row["gpu"] = torch.cuda.get_device_name(0)
    row["torch"] = torch.__version__
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


# ----------------------------------------------------------------------------- commands


def cmd_fit(args):
    rng = np.random.default_rng(args.seed)
    k = args.k
    bsz = args.batches
    sets = []
    kinds = KINDS[args.dtype] if args.acc == "fp32" else ["normal", "normal_c", "cancel"]
    for kind in kinds:
        a, b, c = ensemble(rng, args.dtype, (bsz, 16, k), (bsz, k, 16), (bsz, 16, 16), kind)
        if args.acc == "fp16":
            c = _q(c, "fp16")  # an fp16 accumulator starts from fp16 values (no overflow here)
        got = gpu_triton(a, b, c, args.dtype, acc16=args.acc == "fp16")
        sets.append((kind, a, b, c, got))
    fp8 = args.dtype in ("e4m3", "e5m2")
    groups = ([4, 8, 16] if args.dtype == "tf32" else [8, 16, 32]) if not fp8 else [8, 16, 32]
    fracs = range(22, 28) if not fp8 else [12, 13, 14, 23, 24, 25, 26]
    floors = [None, -132, -133, -134] if args.dtype in ("bf16", "tf32") else [None]
    scores = []
    finals = ["rz", "rne"] if args.acc == "fp32" else ["rz", "rne", "via32"]
    for g, fb, cut, fin, fl in itertools.product(groups, fracs, ["rz", "rd"], finals, floors):
        unit = MatrixUnit("fit", (args.dtype,), args.acc, g, fb, cut=cut, final=fin, floor_exp=fl)
        per = {}
        for kind, a, b, c, got in sets:
            per[kind] = int(bits_equal(model_batched(a, b, c, unit, args.dtype), got).sum())
        total = sum(per.values())
        scores.append((total, dict(group=g, frac_bits=fb, cut=cut, final=fin, floor_exp=fl), per))
    scores.sort(key=lambda s: -s[0])
    n = sum(s[1].size for s in [(None, x[4]) for x in sets])
    for total, params, per in scores[:8]:
        print(f"{total}/{n}", params, per)
    best = scores[0]
    tied = [s[1] for s in scores if s[0] == best[0]]
    record({"cmd": "fit", "dtype": args.dtype, "acc": args.acc, "k": k, "outputs": n, "best_match": best[0],
            "best": best[1], "tied_best": tied, "per_kind": best[2],
            "runner_up": [{"match": s[0], **s[1]} for s in scores[len(tied):len(tied) + 3]]})


def cmd_validate(args):
    rng = np.random.default_rng(args.seed)
    unit = unit_for(args.gen, args.dtype, acc=args.acc)
    if args.group:  # an instruction with a shorter k (m16n8k8) sums fewer products at once
        from dataclasses import replace
        unit = replace(unit, group=args.group)
    k = args.k
    total = match = 0
    per = {}
    side = 0
    t0 = time.time()
    if args.path == "triton":
        kinds = KINDS[args.dtype] if args.acc == "fp32" else ["normal", "normal_c", "cancel"]
        bsz = max(1, args.count // (256 * len(kinds)))
        for kind in kinds:
            a, b, c = ensemble(rng, args.dtype, (bsz, 16, k), (bsz, k, 16), (bsz, 16, 16), kind)
            if args.acc == "fp16":
                c = _q(c, "fp16")
            got = gpu_triton(a, b, c, args.dtype, bk=args.bk, acc16=args.acc == "fp16")
            want = model_flat(a, b, c, unit, args.dtype)
            eq = bits_equal(want, got)
            per[kind] = [int(eq.sum()), int(eq.size)]
            total += eq.size
            match += int(eq.sum())
    else:
        # sides a multiple of 16: an odd leading dimension sends cuBLAS to another kernel family
        # (measured: bf16 with side 447 mismatched), and _scaled_mm requires it
        side = args.side or 16 * int(round((args.count / len(KINDS[args.dtype])) ** 0.5 / 16))
        for kind in KINDS[args.dtype]:
            a, b, _ = ensemble(rng, args.dtype, (side, k), (k, side), (1, 1), kind)
            if args.path == "cublas":
                got = gpu_cublas(a, b, args.dtype)
            else:
                got = gpu_scaled_mm(a, b, args.dtype, args.fast)
            want = split_k(a, b, unit, args.dtype, args.split)
            eq = bits_equal(want, got)
            per[kind] = [int(eq.sum()), int(eq.size)]
            total += eq.size
            match += int(eq.sum())
    record({"cmd": "validate", "gen": args.gen, "dtype": args.dtype, "acc": args.acc, "k": k,
            "path": args.path + ("" if args.path != "scaled_mm" else f"(fast={args.fast})")
            + (f"(bk={args.bk})" if args.bk else "") + (f"(side={side})" if args.path != "triton" else "") + (f"(group={args.group})" if args.group else "") + (f"(split={args.split})" if args.split > 1 else ""),
            "match": match, "outputs": total,
            "per_kind": per, "seconds": round(time.time() - t0, 1)})


KF = 64  # inner size of the feature probes


def cmd_features(args):
    """Paper style probe vectors, one dot product each (row 0 of a 16 x 16 tile)."""
    probes = []
    z = [0.0] * KF
    def put(vals_a, vals_b, c=0.0, note=""):
        a = (list(vals_a) + z)[:KF]
        b = (list(vals_b) + z)[:KF]
        probes.append((a, b, c, note))
    t = 2.0 ** -12
    put([1.5, t, t, t], [1.5, 2 ** -11 + 2 ** -12, 2 ** -13, 2 ** -13],
        note="denormalized product: 2.25 + 2^-23 + 2^-24 + 2^-25 + 2^-25, 25 fraction bits keep 2^-22")
    put([1.0, t, t, t], [2.25, 2 ** -11 + 2 ** -12, 2 ** -13, 2 ** -13],
        note="the same 2.25 as 1 x 2.25: product exponent 1, grid 2^-24")
    put([1.0] + [t] * 15, [1.0] + [2 ** -12] * 15,
        note="1 + 15 x 2^-24: one group of 16 keeps every 2^-24 (25 fraction bits)")
    put([1.0, -t], [1.0, 2 ** -14], note="1 - 2^-26: cut toward zero gives 1 - 2^-24 or 1?")
    put([0.0, t, t], [2 ** 15, 2 ** -12, 2 ** -12], note="zero times a large number takes no part")
    for j in range(2, KF):  # group size: 1 + 2^-24 + 2^-24 is 1 + 2^-23 only inside one group
        a = [1.0, t] + [0.0] * (KF - 2)
        b = [1.0, 2 ** -12] + [0.0] * (KF - 2)
        a[j], b[j] = t, 2 ** -12
        probes.append((a, b, 0.0, f"1 + 2^-24 (position 1) + 2^-24 (position {j})"))
    A = np.zeros((len(probes), 16, KF))
    B = np.zeros((len(probes), KF, 16))
    C = np.zeros((len(probes), 16, 16))
    for i, (a, b, c, _) in enumerate(probes):
        A[i, 0, :] = a
        B[i, :, 0] = b
        C[i, 0, 0] = c
    A, B = _q(A, "fp16"), _q(B, "fp16")
    got = gpu_triton(A, B, C, "fp16")[:, 0, 0]
    unit = unit_for("sm120", "fp16")
    want = model_batched(A, B, C, unit, "fp16")[:, 0, 0]
    rows = []
    for (a, b, c, note), g, w in zip(probes, got, want):
        rows.append({"probe": note, "gpu": float(g).hex(), "model": float(w).hex(),
                     "equal": bool(bits_equal(np.array(g), np.array(w)))})
        print(f"{'OK ' if rows[-1]['equal'] else 'BAD'} gpu={float(g).hex():24s} model={float(w).hex():24s} {note}")
    record({"cmd": "features", "rows": rows})


def cmd_fitall(args):
    """Every fit in one process (one cloud GPU session): the device name is in each record."""
    for d in args.dtypes.split(","):
        try:
            cmd_fit(argparse.Namespace(dtype=d, k=args.k, batches=args.batches, seed=1, acc="fp32"))
        except Exception as exc:  # a format the device or Triton cannot run
            record({"cmd": "fit", "dtype": d, "error": f"{type(exc).__name__}: {exc}"[:300]})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--dtype", default="fp16")
    f.add_argument("--k", type=int, default=32)
    f.add_argument("--batches", type=int, default=64)
    f.add_argument("--acc", default="fp32", choices=["fp32", "fp16"])
    f.add_argument("--seed", type=int, default=1)
    v = sub.add_parser("validate")
    v.add_argument("--gen", default="sm120")
    v.add_argument("--dtype", default="fp16")
    v.add_argument("--k", type=int, default=32)
    v.add_argument("--bk", type=int, default=None)
    v.add_argument("--acc", default="fp32", choices=["fp32", "fp16"])
    v.add_argument("--count", type=int, default=1_000_000)
    v.add_argument("--path", default="triton", choices=["triton", "cublas", "scaled_mm"])
    v.add_argument("--fast", action="store_true")
    v.add_argument("--split", type=int, default=1, help="model a split K schedule (cublas paths)")
    v.add_argument("--group", type=int, default=0, help="override the unit's group size")
    v.add_argument("--side", type=int, default=0, help="matrix side for cublas and scaled_mm")
    v.add_argument("--seed", type=int, default=2)
    sub.add_parser("features")
    fa = sub.add_parser("fitall")
    fa.add_argument("--dtypes", default="fp16,bf16,tf32,e4m3,e5m2")
    fa.add_argument("--k", type=int, default=32)
    fa.add_argument("--batches", type=int, default=64)
    args = ap.parse_args()
    {"fit": cmd_fit, "validate": cmd_validate, "features": cmd_features,
     "fitall": cmd_fitall}[args.cmd](args)


if __name__ == "__main__":
    main()
