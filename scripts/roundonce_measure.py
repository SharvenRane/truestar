"""Measurements for round once elementwise math (docs/ROUND_ONCE.md).

    PYTHONPATH=src python scripts/roundonce_measure.py small     # float16 and bfloat16, exhaustive
    PYTHONPATH=src python scripts/roundonce_measure.py f32err    # float64 error, float32 sample
    PYTHONPATH=src python scripts/roundonce_measure.py sweep     # all 2^32 float32 inputs, GPU
    PYTHONPATH=src python scripts/roundonce_measure.py native    # torch CPU vs CUDA float32
    PYTHONPATH=src python scripts/roundonce_measure.py kernels   # Triton vs numba vs torch bits
    PYTHONPATH=src python scripts/roundonce_measure.py speed     # elements per second on CUDA

Every subcommand merges its numbers into docs/data/roundonce_results.json. The mpmath reference
(tests/roundonce_ref.py) runs in a process pool on the CPU; the GPU runs are short chunks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import roundonce_ref as ref  # noqa: E402

RESULTS = os.path.join(ROOT, "docs", "data", "roundonce_results.json")
CACHE = os.environ.get("ROUNDONCE_CACHE", os.path.join(os.environ.get("TEMP", "/tmp"), "roundonce_cache"))
FUNCS = ("exp", "log", "tanh", "erf", "rsqrt", "sigmoid")


def save(section: str, data):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    allr = {}
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            allr = json.load(f)
    allr[section] = data
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(allr, f, indent=1, sort_keys=True, allow_nan=True)
    print(json.dumps(data, indent=1, allow_nan=True)[:4000])


# ---------------------------------------------------------------------------- reference in a pool

def _ref_chunk(args):
    name, xs, fmt = args
    return [ref.reference(name, x, fmt) for x in xs]


def _relerr_chunk(args):
    name, xs, ys = args
    return [ref.relative_error(name, x, y) for x, y in zip(xs, ys)]


def _ulp_chunk(args):
    name, xs, ys, fmt = args
    return [ref.ulp_error(name, x, y, fmt) for x, y in zip(xs, ys)]


def pool_map(fn, name, cols, workers=None, chunk=2000):
    n = len(cols[0])
    jobs = [(name, *[list(c[i:i + chunk]) for c in cols]) for i in range(0, n, chunk)]
    out = []
    with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 2)) as ex:
        for part in ex.map(fn, jobs):
            out.extend(part)
    return out


def reference_array(name, xs: np.ndarray, fmt: str, tag: str) -> np.ndarray:
    os.makedirs(CACHE, exist_ok=True)
    h = hashlib.sha256(xs.tobytes()).hexdigest()[:16]
    path = os.path.join(CACHE, f"ref_{tag}_{name}_{fmt}_{h}.npy")
    if os.path.exists(path):
        return np.load(path)
    vals = pool_map(_ref_chunk, name, [xs.astype(np.float64).tolist(), [fmt] * len(xs)]
                    ) if False else None
    jobs = [(name, xs[i:i + 2000].astype(np.float64).tolist(), fmt) for i in range(0, len(xs), 2000)]
    vals = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        for part in ex.map(_ref_chunk, jobs):
            vals.extend(part)
    arr = np.array(vals, dtype=np.float64)
    np.save(path, arr)
    return arr


def same_bits(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise: equal as IEEE values including the sign of zero, any NaN equal to any NaN."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    eq = (a == b) & (np.signbit(a) == np.signbit(b))
    return both_nan | eq


# ---------------------------------------------------------------------------- 16 bit exhaustive

def all16(torch, fmt):
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    return bits.view(getattr(torch, fmt))


def cmd_small(a):
    import torch
    from truestar.kernel import roundonce as R
    out = {}
    for fmt in ("float16", "bfloat16"):
        x = all16(torch, fmt)
        xs = x.to(torch.float64).numpy()
        for name in FUNCS:
            t0 = time.time()
            r = reference_array(name, xs, fmt, "all16")
            tref = time.time() - t0
            ro_cpu = R.apply(name, x).to(torch.float64).numpy()
            ro_gpu = R.apply(name, x.cuda()).cpu().to(torch.float64).numpy()
            # kernel semantics: round to float32 first, then the store rounds to fmt
            v64 = R.value64(name, x.cuda())
            twice = R.round_to(R.round_to(v64, torch.float32).to(torch.float64), getattr(torch, fmt))
            twice = twice.cpu().to(torch.float64).numpy()
            rec = {"mismatch_cpu": int((~same_bits(ro_cpu, r)).sum()),
                   "mismatch_cuda": int((~same_bits(ro_gpu, r)).sum()),
                   "cpu_vs_cuda": int((~same_bits(ro_cpu, ro_gpu)).sum()),
                   "cpu_vs_cuda_raw_bits": int((R.apply(name, x).view(torch.int16)
                                                != R.apply(name, x.cuda()).cpu().view(torch.int16)).sum()),
                   "via_float32_mismatch": int((~same_bits(twice, r)).sum()),
                   "reference_seconds": round(tref, 1)}
            # native torch in the same dtype, for scale
            fn = {"exp": torch.exp, "log": torch.log, "tanh": torch.tanh, "erf": torch.erf,
                  "rsqrt": torch.rsqrt, "sigmoid": torch.sigmoid}[name]
            for dev in ("cpu", "cuda"):
                try:
                    nat = fn(x.to(dev)).cpu().to(torch.float64).numpy()
                    rec[f"native_{dev}_mismatch"] = int((~same_bits(nat, r)).sum())
                except RuntimeError as e:
                    rec[f"native_{dev}_mismatch"] = f"not supported: {str(e)[:60]}"
            nc = fn(x).to(torch.float64).numpy()
            ng = fn(x.cuda()).cpu().to(torch.float64).numpy()
            rec["native_cpu_vs_cuda"] = int((~same_bits(nc, ng)).sum())
            rec["sha256_bits"] = hashlib.sha256(R.apply(name, x).view(torch.int16).numpy().tobytes()).hexdigest()
            out[f"{fmt}/{name}"] = rec
            print(fmt, name, rec, flush=True)
    save("small_exhaustive", out)


# ---------------------------------------------------------------------------- float32 error sample

THRESH = {
    "exp": [0.0, 88.72283, -87.33654, -103.27893, -103.97208, 0.34657359, -0.34657359, 1.0, 709.0],
    "log": [1.0, 0.70710677, 1.4142135, 2.0, 0.5, 1.1754944e-38, 1.4e-45, 3.4028235e38],
    "tanh": [0.0, 0.5, -0.5, 9.010913, -9.010913, 20.0, 1e-4],
    "erf": [0.0, 3.9192059, -3.9192059, 4.0, -4.0, 1e-4, 1.0, 2.0],
    "rsqrt": [1.0, 4.0, 0.25, 1.1754944e-38, 1.4e-45, 3.4028235e38, 2.0],
    "sigmoid": [0.0, 16.635532, -87.33654, -103.27893, -103.97208, 88.72283, 1e-4],
}


def f32_sample(name, per_binade=160, around=600, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    # every binade (exponent field 0..254) of both signs: random mantissas
    for e in range(255):
        man = rng.integers(0, 1 << 23, size=per_binade, dtype=np.int64)
        for s in (0, 1):
            parts.append(((s << 31) | (e << 23) | man).astype(np.uint32))
    # +- `around` neighbours of each threshold, and the extremes
    for t in THRESH[name]:
        for v in (t, -t):
            b = np.array([v], dtype=np.float32).view(np.uint32)[0].astype(np.int64)
            parts.append(np.clip(b + np.arange(-around, around + 1), 0, 0xFFFFFFFF).astype(np.uint32))
    special = np.array([0x0, 0x80000000, 0x1, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000,
                        0xFF800000, 0x7FC00000, 0x00800000, 0x007FFFFF], dtype=np.uint32)
    parts.append(special)
    u = np.unique(np.concatenate(parts))
    return u.view(np.float32)


def cmd_f32err(a):
    import torch
    from truestar.kernel import roundonce as R
    out = {}
    for name in (a.only or FUNCS):
        xs = f32_sample(name)
        x = torch.from_numpy(xs.copy())
        v_cpu = R.value64(name, x).numpy()
        v_gpu = R.value64(name, x.cuda()).cpu().numpy()
        y32 = R.apply(name, x.cuda()).cpu().to(torch.float64).numpy()
        t0 = time.time()
        rel = np.array(pool_map(_relerr_chunk, name, [xs.astype(np.float64).tolist(), v_gpu.tolist()]))
        r32 = reference_array(name, xs, "float32", "f32s")
        # the relative bound matters where the float64 value is a normal float64 above 2^-1000
        live = np.isfinite(v_gpu) & (np.abs(v_gpu) > 2.0 ** -1000)
        if name == "erf":
            # |x| >= 4 returns exactly +-1 on purpose: erfc(4) = 1.5e-8 is below half an ulp of 1 in
            # float32 (2^-25 = 3.0e-8), so every format rounds erf there to +-1 (analytic, not sampled)
            live &= np.abs(xs) < 4.0
        rr = rel[live & np.isfinite(rel)]
        bad_rel = int((~np.isfinite(rel[live])).sum())
        worst = int(np.argmax(np.where(live & np.isfinite(rel), rel, -1)))
        rec = {"n": int(len(xs)), "max_rel_error": float(rr.max()),
               "max_rel_error_in_2^-53": float(rr.max() / 2.0 ** -53),
               "p99_rel_error_in_2^-53": float(np.quantile(rr, 0.99) / 2.0 ** -53),
               "worst_input": float(xs[worst]), "nonfinite_rel": bad_rel,
               "float32_mismatch": int((~same_bits(y32, r32)).sum()),
               "value64_cpu_vs_cuda": int((v_cpu.view(np.int64) != v_gpu.view(np.int64)).sum()),
               "seconds": round(time.time() - t0, 1)}
        out[name] = rec
        print(name, rec, flush=True)
    prev = {}
    if os.path.exists(RESULTS):
        prev = json.load(open(RESULTS, encoding="utf-8")).get("float32_error_sample", {})
    prev.update(out)
    save("float32_error_sample", prev)


# ---------------------------------------------------------------------------- exhaustive float32

def _midpoint_distance(torch, y, r32):
    """Relative distance from float64 values y to the nearest float32 rounding midpoint around
    their float32 rounding r32 (both float64 tensors)."""
    rf = r32.to(torch.float32)
    lo = torch.nextafter(rf, torch.full_like(rf, -math.inf)).to(torch.float64)
    hi = torch.nextafter(rf, torch.full_like(rf, math.inf)).to(torch.float64)
    big = 2.0 ** 128
    # beyond the largest float32 the next value is "2^128" for midpoint purposes
    hi = torch.where(torch.isinf(hi) & (hi > 0), torch.full_like(hi, big), hi)
    lo = torch.where(torch.isinf(lo) & (lo < 0), torch.full_like(lo, -big), lo)
    r = torch.where(torch.isinf(r32), torch.sign(r32) * big, r32)
    hi = torch.where(torch.isinf(r32) & (r32 > 0), torch.full_like(hi, big), hi)
    lo = torch.where(torch.isinf(r32) & (r32 < 0), torch.full_like(lo, -big), lo)
    m1 = (r + lo) * 0.5
    m2 = (r + hi) * 0.5
    d = torch.minimum((y - m1).abs(), (y - m2).abs())
    return d / y.abs()


def cmd_sweep(a):
    import torch
    from truestar.kernel import roundonce as R
    bounds = json.load(open(RESULTS, encoding="utf-8"))["float32_error_sample"]
    out = {}
    chunk = 1 << 24
    for name in (a.only or FUNCS):
        emax = bounds[name]["max_rel_error"]
        B = max(16 * emax, 2.0 ** -50)
        flagged = []
        t0 = time.time()
        n_live = 0
        for start in range(0, 1 << 32, chunk):
            bits = torch.arange(start, start + chunk, dtype=torch.int64, device="cuda")
            x = (bits - (bits >= (1 << 31)).to(torch.int64) * (1 << 32)).to(torch.int32).view(torch.float32)
            y = R.value64(name, x)
            r = R.round_to(y, torch.float32).to(torch.float64)
            ok = torch.isfinite(y) & (y.abs() > 2.0 ** -1000) & ~torch.isnan(x)
            if name == "erf":
                ok &= x.abs() < 4.0      # erf is +-1 there in float32, proven analytically
            n_live += int(ok.sum())
            d = _midpoint_distance(torch, y, r)
            f = ok & (d <= B)
            if bool(f.any()):
                flagged.append(x[f].cpu().numpy())
            del bits, x, y, r, d, f, ok
        torch.cuda.synchronize()
        tgpu = time.time() - t0
        fl = np.concatenate(flagged) if flagged else np.zeros(0, np.float32)
        refv = reference_array(name, fl, "float32", "flag") if len(fl) else np.zeros(0)
        got = R.apply(name, torch.from_numpy(fl.copy()).cuda()).cpu().to(torch.float64).numpy()
        mism = ~same_bits(got, refv)
        rec = {"bound_B": B, "bound_B_in_2^-53": B / 2.0 ** -53, "measured_max_rel": emax,
               "inputs": 1 << 32, "live_inputs": n_live, "flagged": int(len(fl)),
               "flagged_mismatch": int(mism.sum()),
               "mismatch_inputs": [float(v) for v in fl[mism][:50]],
               "gpu_seconds": round(tgpu, 1)}
        out[name] = rec
        print(name, rec, flush=True)
    prev = {}
    if os.path.exists(RESULTS):
        prev = json.load(open(RESULTS, encoding="utf-8")).get("float32_exhaustive_sweep", {})
    prev.update(out)
    save("float32_exhaustive_sweep", prev)


# ---------------------------------------------------------------------------- native CPU vs CUDA

def _native(torch, name):
    return {"exp": torch.exp, "log": torch.log, "tanh": torch.tanh, "erf": torch.erf,
            "rsqrt": torch.rsqrt, "sigmoid": torch.sigmoid}[name]


def _ulp32(torch, y64):
    """ulp of float32 at |y64| (float64), with float32 subnormal spacing below 2^-126."""
    a = y64.abs().clamp(min=2.0 ** -126)
    e = torch.floor(torch.log2(a))
    # guard log2 rounding at binade edges
    e = torch.where(torch.exp2(e) > a, e - 1, e)
    e = torch.where(torch.exp2(e + 1) <= a, e + 1, e)
    return torch.exp2(e - 23)


def cmd_native(a):
    """Exhaustive over all 2^32 float32 inputs: torch native CPU vs CUDA bits, and each one's max
    ulp error against round once's float64 value (whose own error is below 2^-48 relative, so the
    ulp numbers are good to about 1e-8 ulp)."""
    import torch
    from truestar.kernel import roundonce as R
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    out = {}
    chunk = 1 << 24
    for name in (a.only or FUNCS):
        fn = _native(torch, name)
        diff = 0
        worst = {"cpu": 0.0, "cuda": 0.0}
        worst_x = {"cpu": None, "cuda": None}
        wrong = {"cpu": 0, "cuda": 0}
        ro_wrong = 0
        diff_examples = []
        t0 = time.time()
        for start in range(a.first * chunk, min(1 << 32, a.last * chunk), chunk):
            bits = torch.arange(start, start + chunk, dtype=torch.int64)
            x = (bits - (bits >= (1 << 31)).to(torch.int64) * (1 << 32)).to(torch.int32).view(torch.float32)
            xc = x.cuda()
            yc = fn(x)
            yg = fn(xc)
            ygc = yg.cpu()
            d = yc.view(torch.int32) != ygc.view(torch.int32)
            d &= ~(torch.isnan(yc) & torch.isnan(ygc))
            nd = int(d.sum())
            diff += nd
            if nd and len(diff_examples) < 10:
                diff_examples += [float(v) for v in x[d][:10 - len(diff_examples)]]
            r32 = R.round_to(R.value64(name, xc), torch.float32)   # correctly rounded (sweep)
            y64 = R.value64(name, xc, patched=False)                 # within 2^-48 relative
            ulp = _ulp32(torch, y64)
            fin = torch.isfinite(y64)
            for dev, yy in (("cpu", yc.cuda()), ("cuda", yg)):
                same = (yy.view(torch.int32) == r32.view(torch.int32)) | (torch.isnan(yy) & torch.isnan(r32))
                wrong[dev] += int((~same).sum())
                yy64 = yy.to(torch.float64)
                err = torch.where(fin & torch.isfinite(yy64), (yy64 - y64).abs() / ulp,
                                  torch.zeros_like(y64))
                # an infinity where the true value is finite (or the reverse) counts as inf
                bad = (fin != torch.isfinite(yy64)) & ~torch.isnan(y64)
                bad &= ~(torch.isinf(yy64) & (r32.to(torch.float64) == yy64))   # overflow rounding
                if bool(bad.any()):
                    err = torch.where(bad, torch.full_like(err, math.inf), err)
                m = float(err.max())
                if m > worst[dev]:
                    worst[dev] = m
                    worst_x[dev] = float(xc[int(err.argmax())])
            del bits, x, xc, yc, yg, ygc, y64, r32, ulp
        torch.cuda.synchronize()
        rec = {"inputs": 1 << 32, "native_cpu_vs_cuda_bits_differ": diff,
               "native_cpu_vs_cuda_examples": diff_examples,
               "native_cpu_max_ulp": worst["cpu"], "native_cpu_worst_input": worst_x["cpu"],
               "native_cuda_max_ulp": worst["cuda"], "native_cuda_worst_input": worst_x["cuda"],
               "native_cpu_not_correctly_rounded": wrong["cpu"],
               "native_cuda_not_correctly_rounded": wrong["cuda"],
               "seconds": round(time.time() - t0, 1)}
        out[name] = rec
        print(name, rec, flush=True)
    prev = {}
    if os.path.exists(RESULTS):
        prev = json.load(open(RESULTS, encoding="utf-8")).get("float32_native_exhaustive", {})
    prev.update(out)
    save("float32_native_exhaustive", prev)


def cmd_ro_portable(a):
    """Round once on the CPU vs on CUDA, float32: a large random sample of bit patterns."""
    import torch
    from truestar.kernel import roundonce as R
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    g = torch.Generator().manual_seed(1)
    n = a.n
    out = {}
    for name in (a.only or FUNCS):
        diff = 0
        t0 = time.time()
        for _ in range(max(1, n // (1 << 22))):
            bits = torch.randint(-(1 << 31), 1 << 31, (1 << 22,), generator=g, dtype=torch.int64)
            x = bits.to(torch.int32).view(torch.float32)
            yc = R.apply(name, x)
            yg = R.apply(name, x.cuda()).cpu()
            diff += int((yc.view(torch.int32) != yg.view(torch.int32)).sum())   # raw bits, NaN too
        out[name] = {"inputs": max(1, n // (1 << 22)) * (1 << 22), "cpu_vs_cuda_bits_differ": diff,
                     "seconds": round(time.time() - t0, 1)}
        print(name, out[name], flush=True)
    save("float32_roundonce_cpu_vs_cuda", out)


# ---------------------------------------------------------------------------- kernels

def _kernel(K, name, dtype_in, dtype_out):
    N = K.sym("N")
    x = K.buffer("x", (N,), dtype_in)
    i = K.axis("i", N)
    f = {"exp": K.exp, "log": K.log, "tanh": K.tanh, "erf": K.ir.erf if hasattr(K, "ir") else None,
         "rsqrt": K.rsqrt, "sigmoid": K.sigmoid}[name]
    if f is None:
        from truestar.kernel import ir
        f = ir.erf
    return K.kernel(f"ro_{name}_{dtype_in}_{dtype_out}", [K.output("y", (i,), f(x[i]), dtype_out)])


def cmd_kernels(a):
    os.environ["TRUESTAR_VERIFY"] = "0"   # exp overflows float32 where the float64 reference does not
    import torch
    from truestar import kernel as K
    from truestar.kernel import ir, roundonce as R
    g = torch.Generator().manual_seed(2)
    out = {}
    for name in FUNCS:
        f = getattr(ir, name)
        N = K.sym("N")
        i = K.axis("i", N)
        rec = {}
        for fmt in ("float16", "bfloat16", "float32"):
            xb = K.buffer("x", (N,), fmt)
            if fmt == "float32":
                bits = torch.randint(-(1 << 31), 1 << 31, (1 << 22,), generator=g, dtype=torch.int64)
                x = bits.to(torch.int32).view(torch.float32)
            else:
                x = all16(torch, fmt)
            # float32 output: the kernel's own working format, compared with the torch module
            k = K.kernel(f"ro_{name}_{fmt}", [K.output("y", (i,), f(xb[i]), "float32")])
            want = R.apply(name, x, torch.float32)
            tri = K.compile(k, "cuda", math="round_once", tune=False)(x=x.cuda()).cpu()
            cpu = K.compile(k, "cpu", math="round_once")(x=x)
            def nd(p, q):
                # raw bits, NaN included; the kernel layer stores -0 as +0, so zeros compare as +0
                p = torch.where(p == 0, torch.zeros_like(p), p)
                q = torch.where(q == 0, torch.zeros_like(q), q)
                return int((p.view(torch.int32) != q.view(torch.int32)).sum())
            rec[fmt] = {"n": int(x.numel()), "triton_vs_torch": nd(tri, want),
                        "numba_vs_torch": nd(cpu, want), "triton_vs_numba": nd(tri, cpu)}
            if fmt != "float32":
                k2 = K.kernel(f"ro_{name}_{fmt}_s", [K.output("y", (i,), f(xb[i]), fmt)])
                t2 = K.compile(k2, "cuda", math="round_once", tune=False)(x=x.cuda()).cpu()
                direct = R.apply(name, x)
                bad = ((t2 != direct) & ~(torch.isnan(t2) & torch.isnan(direct)))   # values, not NaN bits
                rec[fmt]["store_in_fmt_vs_direct_rounding"] = int(bad.sum())
        out[name] = rec
        print(name, rec, flush=True)
    save("kernel_bitwise", out)


# ---------------------------------------------------------------------------- speed

def cmd_speed(a):
    import torch
    from truestar import kernel as K
    from truestar.kernel import ir, roundonce as R
    out = {}

    def bench(fn, reps=20):
        fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(5):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(reps):
                fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) / reps / 1e3)
        return float(np.median(ts))

    for name in FUNCS:
        f = getattr(ir, name)
        N = K.sym("N")
        i = K.axis("i", N)
        xb = K.buffer("x", (N,), "float32")
        k = K.kernel(f"sp_{name}", [K.output("y", (i,), f(xb[i]), "float32")])
        kd = K.compile(k, "cuda", tune=False)
        kr = K.compile(k, "cuda", math="round_once", tune=False)
        nat = _native(torch, name)
        rec = {}
        for lg in (20, 24, 26):
            n = 1 << lg
            x = (torch.rand(n, device="cuda") * 4 - 2) if name not in ("log", "rsqrt") else torch.rand(n, device="cuda") * 4 + 1e-3
            kd(x=x)
            kr(x=x)  # first calls verify against float64; not timed
            yk = torch.empty_like(x)
            t_nat = bench(lambda: nat(x))
            t_def = bench(lambda: kd(x=x))
            t_ro = bench(lambda: kr(x=x))
            t_mod = bench(lambda: R.apply(name, x), reps=3) if lg <= 24 else None
            rec[f"2^{lg}"] = {"torch_native_Geps": n / t_nat / 1e9, "truestar_default_Geps": n / t_def / 1e9,
                              "truestar_round_once_Geps": n / t_ro / 1e9,
                              "roundonce_torch_eager_Geps": (n / t_mod / 1e9) if t_mod else None,
                              "round_once_over_native_time": t_ro / t_nat}
        out[name] = rec
        print(name, rec, flush=True)
    out["device"] = torch.cuda.get_device_name(0)
    save("speed_cuda", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["small", "f32err", "sweep", "native", "roportable", "kernels", "speed"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--n", type=int, default=1 << 24)
    ap.add_argument("--first", type=int, default=0, help="native: first chunk of 2^24 inputs")
    ap.add_argument("--last", type=int, default=256, help="native: end chunk (exclusive)")
    a = ap.parse_args()
    {"small": cmd_small, "f32err": cmd_f32err, "sweep": cmd_sweep, "native": cmd_native,
     "roportable": cmd_ro_portable, "kernels": cmd_kernels, "speed": cmd_speed}[a.cmd](a)


if __name__ == "__main__":
    main()
