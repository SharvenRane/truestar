"""A fixed numerics battery for accelerators that are not on this machine (docs/TPU.md).

This file is self contained on purpose: it imports only numpy and the standard library at module
level, so the same file runs as a Kaggle TPU kernel, on a local GPU through PyTorch, and inside
`truestar conform tpu` as the source of the inputs and the float64 reference.

Inputs are never drawn from a framework's random generator (those give other bits on other
hosts). Every tensor comes from a splitmix64 counter hash in uint64 arithmetic, turned into floats
by exact operations only: a 24 bit integer over 2**23 minus 1 for float32 tensors, a 7 bit integer
over 2**6 minus 1 for bfloat16 tensors, then a power of two scale. Every value is therefore exact in
its dtype and the same on every machine; `input_hashes` pins the bytes.

    python tpu_battery.py OUT_DIR                     run with whatever is here (JAX on a TPU first)
    python tpu_battery.py OUT_DIR --runner torch      PyTorch: cpu, cuda, cuda_tf32 (local GPU)
    python tpu_battery.py OUT_DIR --runner jax        JAX on its default device and on its CPU

Output: OUT_DIR/summary.json and OUT_DIR/<backend>/<case>.npy (bfloat16 results are stored as
float32, which is exact).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np

BATTERY_VERSION = 1
MASTER_SEED = 20261124
REPEATS = 5  # calls of one compiled program; a second compile adds one more call

_GOLD = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def stream_seed(case: str, name: str) -> int:
    h = hashlib.sha256(f"{MASTER_SEED}:{case}:{name}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def _splitmix(seed: int, n: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + (np.arange(1, n + 1, dtype=np.uint64) * _GOLD)
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        return z ^ (z >> np.uint64(31))


def draw(case: str, name: str, shape, kind: str = "f32", log2_scale: int = 0) -> np.ndarray:
    """Values exact in float32 ('f32') or in bfloat16 ('bf16'), as a float32 array."""
    n = int(np.prod(shape))
    z = _splitmix(stream_seed(case, name), n)
    if kind == "f32":
        v = (z >> np.uint64(40)).astype(np.float64) / 2.0 ** 23 - 1.0  # 24 bits
    elif kind == "bf16":
        v = (z >> np.uint64(57)).astype(np.float64) / 2.0 ** 6 - 1.0  # 7 bits
    elif kind == "wide":  # float32 exact values over 2**-20 .. 2**20: a sum's accumulation order shows
        u = (z >> np.uint64(40)).astype(np.float64) / 2.0 ** 23 - 1.0
        e = ((z & np.uint64(63)).astype(np.int64) % 41) - 20
        v = np.ldexp(u, e)
    elif kind == "pos":  # (k + 1) / 2**24 in (0, 1], exact in float32
        v = ((z >> np.uint64(40)).astype(np.float64) + 1.0) / 2.0 ** 24
    else:
        raise ValueError(kind)
    return np.ldexp(v, log2_scale).astype(np.float32).reshape(shape)


# ---------------------------------------------------------------------------------------------
# cases: name -> (op, dtype, precision, tensor specs). The precision field is JAX's matmul
# precision ("default", "high", "highest"); a PyTorch backend runs every case in its own mode.
# ---------------------------------------------------------------------------------------------

def _mlp_specs(kind):
    return {"x": ((128, 512), kind, 0), "w1": ((512, 2048), kind, -5), "b1": ((2048,), kind, -4),
            "w2": ((2048, 512), kind, -6), "b2": ((512,), kind, -4)}


def _enc_specs():
    s = {"x": ((4, 128, 256), "f32", 0)}
    for w in ("wq", "wk", "wv", "wo"):
        s[w] = ((256, 256), "f32", -4)
        s["b" + w[1]] = ((256,), "f32", -5)
    s.update({"w1": ((256, 1024), "f32", -4), "b1": ((1024,), "f32", -5),
              "w2": ((1024, 256), "f32", -5), "b2": ((256,), "f32", -5),
              "g1": ((256,), "f32", -3), "c1": ((256,), "f32", -4),
              "g2": ((256,), "f32", -3), "c2": ((256,), "f32", -4)})
    return s


_MM = {"a": ((512, 1024), "f32", 0), "b": ((1024, 512), "f32", 0)}
_MM16 = {"a": ((512, 1024), "bf16", 0), "b": ((1024, 512), "bf16", 0)}

CASES = {
    "mm_f32_default": ("matmul", "float32", "default", _MM),
    "mm_f32_high": ("matmul", "float32", "high", _MM),
    "mm_f32_highest": ("matmul", "float32", "highest", _MM),
    "mm_bf16": ("matmul", "bfloat16", "default", _MM16),
    "mm_bf16_acc_f32": ("matmul_acc32", "bfloat16", "default", _MM16),
    "softmax_f32": ("softmax", "float32", "default", {"x": ((256, 4096), "f32", 3)}),
    "layernorm_f32": ("layernorm", "float32", "default",
                      {"x": ((256, 1024), "f32", 2), "g": ((1024,), "f32", 0), "c": ((1024,), "f32", -1)}),
    "mlp_f32_default": ("mlp", "float32", "default", _mlp_specs("f32")),
    "mlp_f32_highest": ("mlp", "float32", "highest", _mlp_specs("f32")),
    "mlp_bf16": ("mlp", "bfloat16", "default", _mlp_specs("bf16")),
    "encoder_f32_default": ("encoder", "float32", "default", _enc_specs()),
    "encoder_f32_highest": ("encoder", "float32", "highest", _enc_specs()),
    "sum_f32_uniform": ("sum", "float32", "default", {"v": ((1 << 24,), "f32", 0)}),
    "sum_f32_wide": ("sum", "float32", "default", {"v": ((1 << 22,), "wide", 0)}),
    "exp_f32": ("exp", "float32", "default", {"x": ((1 << 20,), "f32", 3)}),
    "tanh_f32": ("tanh", "float32", "default", {"x": ((1 << 20,), "f32", 2)}),
    "log_f32": ("log", "float32", "default", {"x": ((1 << 20,), "pos", 0)}),
    "rsqrt_f32": ("rsqrt", "float32", "default", {"x": ((1 << 20,), "pos", 0)}),
}


def inputs(case: str) -> dict:
    specs = CASES[case][3]
    return {k: draw(case, k, shape, kind, sc) for k, (shape, kind, sc) in specs.items()}


def input_hash(arrs: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(arrs):
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrs[k], dtype=np.float32).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# one definition of each op over a small ops table, so the reference and every backend compute
# the same formula
# ---------------------------------------------------------------------------------------------

def _softmax(o, x):
    m = o.max_last(x)
    e = o.exp(x - m)
    return e / o.sum_last(e)


def _layernorm(o, x, g, c, eps=1e-5):
    mu = o.sum_last(x) / x.shape[-1]
    d = x - mu
    var = o.sum_last(d * d) / x.shape[-1]
    return d / o.sqrt(var + eps) * g + c


def _gelu(o, x):
    return 0.5 * x * (1.0 + o.erf(x * (1.0 / math.sqrt(2.0))))


def _mlp(o, t):
    h = _gelu(o, o.mm(t["x"], t["w1"]) + t["b1"])
    return o.mm(h, t["w2"]) + t["b2"]


def _encoder(o, t):
    x = t["x"]
    bsz, seq, d = x.shape
    heads, dh = 4, d // 4

    def split(y):
        return o.heads(y, bsz, seq, heads, dh)

    q = split(o.mm(x, t["wq"]) + t["bq"])
    k = split(o.mm(x, t["wk"]) + t["bk"])
    v = split(o.mm(x, t["wv"]) + t["bv"])
    p = _softmax(o, o.mm(q, o.tr(k)) * (1.0 / 8.0))
    ctx = o.merge(o.mm(p, v), bsz, seq, d)
    h = _layernorm(o, x + o.mm(ctx, t["wo"]) + t["bo"], t["g1"], t["c1"])
    f = o.mm(o.relu(o.mm(h, t["w1"]) + t["b1"]), t["w2"]) + t["b2"]
    return _layernorm(o, h + f, t["g2"], t["c2"])


def compute(o, op: str, t: dict):
    if op in ("matmul", "matmul_acc32"):
        return o.mm(t["a"], t["b"], acc32=op == "matmul_acc32")
    if op == "softmax":
        return _softmax(o, t["x"])
    if op == "layernorm":
        return _layernorm(o, t["x"], t["g"], t["c"])
    if op == "mlp":
        return _mlp(o, t)
    if op == "encoder":
        return _encoder(o, t)
    if op == "sum":
        return o.sum_all(t["v"])
    if op in ("exp", "tanh", "log"):
        return getattr(o, op)(t["x"])
    if op == "rsqrt":
        return o.rsqrt(t["x"])
    raise ValueError(op)


class NumpyRef:
    """float64 reference on the (exact) float32 or bfloat16 input values."""

    def mm(self, a, b, acc32=False):
        return np.matmul(a, b)

    def exp(self, x):
        return np.exp(x)

    def tanh(self, x):
        return np.tanh(x)

    def log(self, x):
        return np.log(x)

    def sqrt(self, x):
        return np.sqrt(x)

    def rsqrt(self, x):
        return 1.0 / np.sqrt(x)

    def erf(self, x):
        return np.vectorize(math.erf, otypes=[np.float64])(x)

    def relu(self, x):
        return np.maximum(x, 0.0)

    def max_last(self, x):
        return x.max(axis=-1, keepdims=True)

    def sum_last(self, x):
        return x.sum(axis=-1, keepdims=True)

    def sum_all(self, v):
        return np.array(math.fsum(v.tolist()))

    def tr(self, x):
        return np.swapaxes(x, -1, -2)

    def heads(self, y, b, s, h, dh):
        return y.reshape(b, s, h, dh).transpose(0, 2, 1, 3)

    def merge(self, y, b, s, d):
        return y.transpose(0, 2, 1, 3).reshape(b, s, d)


def reference(case: str, arrs: dict = None) -> np.ndarray:
    arrs = inputs(case) if arrs is None else arrs
    op = CASES[case][0]
    return np.asarray(compute(NumpyRef(), op, {k: v.astype(np.float64) for k, v in arrs.items()}),
                      dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------

def _abs_products(case: str, arrs: dict) -> np.ndarray:
    return np.matmul(np.abs(arrs["a"].astype(np.float64)), np.abs(arrs["b"].astype(np.float64)))


def metrics(case: str, out: np.ndarray, ref: np.ndarray, arrs: dict) -> dict:
    """Error of one output against the float64 reference.

    eff_bits is the precision the result behaves as having: for a matmul -log2 of the worst
    |error| / (|A| |B|) over elements, for a sum -log2(|error| / sum |v|). float32 rounding with any
    summation order stays above log2(1/(n eps)); one bfloat16 pass per product sits near 8.
    """
    op, dtype = CASES[case][0], CASES[case][1]
    out64 = np.asarray(out, dtype=np.float64).reshape(ref.shape)
    d = np.abs(out64 - ref)
    m = {"max_abs": float(d.max()), "rel_l2": float(np.linalg.norm(d) / max(np.linalg.norm(ref), 1e-300)),
         "nonfinite": int((~np.isfinite(out64)).sum())}
    floor = 1e-3 * float(np.sqrt(np.mean(ref * ref))) if ref.size > 1 else abs(float(ref))
    m["max_abs_over_rms"] = float(d.max() / max(float(np.sqrt(np.mean(ref * ref))), 1e-300))
    m["max_rel"] = float((d / np.maximum(np.abs(ref), max(floor, 1e-300))).max())
    if dtype == "float32" and op in ("exp", "tanh", "log", "rsqrt"):
        sp = np.spacing(np.abs(ref).astype(np.float32)).astype(np.float64)
        m["max_ulp"] = float((d / sp).max())
    if op in ("matmul", "matmul_acc32"):
        s = _abs_products(case, arrs)
        m["eff_bits"] = float(-np.log2(max(float((d / s).max()), 2.0 ** -60)))
    if op == "sum":
        s = float(np.abs(arrs["v"].astype(np.float64)).sum())
        m["eff_bits"] = float(-np.log2(max(float(d.max()) / s, 2.0 ** -60)))
    return m


def _digest(raw: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------
# runners: each returns {case: {"out": float array, "hashes": [..], "note": str}}
# ---------------------------------------------------------------------------------------------

def _run_cases(prepare, call, to_numpy, raw_bytes, recompile, cases):
    res = {}
    for case in cases:
        arrs = inputs(case)
        try:
            args = prepare(case, arrs)
            fn = call(case)
            outs = [fn(args) for _ in range(REPEATS)]
            outs.append(recompile(case)(args))
            hashes = [_digest(raw_bytes(o)) for o in outs]
            res[case] = {"out": to_numpy(outs[0]), "hashes": hashes}
        except Exception as exc:  # a case the backend cannot run is recorded, not fatal
            res[case] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    return res


class _JaxOps:
    def __init__(self, jnp, lax, precision, dtype):
        self.jnp, self.lax, self.p, self.dt = jnp, lax, precision, dtype

    def mm(self, a, b, acc32=False):
        if acc32:
            return self.jnp.matmul(a, b, precision=self.p, preferred_element_type=self.jnp.float32)
        return self.jnp.matmul(a, b, precision=self.p)

    def exp(self, x):
        return self.jnp.exp(x)

    def tanh(self, x):
        return self.jnp.tanh(x)

    def log(self, x):
        return self.jnp.log(x)

    def sqrt(self, x):
        return self.jnp.sqrt(x)

    def rsqrt(self, x):
        return self.lax.rsqrt(x)

    def erf(self, x):
        return self.lax.erf(x)

    def relu(self, x):
        return self.jnp.maximum(x, 0)

    def max_last(self, x):
        return x.max(axis=-1, keepdims=True)

    def sum_last(self, x):
        return x.sum(axis=-1, keepdims=True)

    def sum_all(self, v):
        return v.sum()

    def tr(self, x):
        return self.jnp.swapaxes(x, -1, -2)

    def heads(self, y, b, s, h, dh):
        return y.reshape(b, s, h, dh).transpose(0, 2, 1, 3)

    def merge(self, y, b, s, d):
        return y.transpose(0, 2, 1, 3).reshape(b, s, d)


def run_jax(device_kind: str = "default", cases=None):
    import jax
    import jax.numpy as jnp
    from jax import lax

    dev = jax.devices()[0] if device_kind == "default" else jax.devices("cpu")[0]
    prec = {"default": lax.Precision.DEFAULT, "high": lax.Precision.HIGH, "highest": lax.Precision.HIGHEST}

    def make(case):
        op, dtype, p, _ = CASES[case]
        o = _JaxOps(jnp, lax, prec[p], dtype)
        return lambda t: compute(o, op, t)

    def prepare(case, arrs):
        dt = jnp.bfloat16 if CASES[case][1] == "bfloat16" else jnp.float32
        return {k: jax.device_put(jnp.asarray(v).astype(dt), dev) for k, v in arrs.items()}

    def call(case):
        f = jax.jit(make(case))
        return lambda a: jax.block_until_ready(f(a))

    def recompile(case):
        g = make(case)
        f = jax.jit(lambda t: g(t))  # a new function object: traced and compiled again
        return lambda a: jax.block_until_ready(f(a))

    def to_numpy(x):
        return np.asarray(jnp.asarray(x, dtype=jnp.float32))

    def raw(x):
        return np.ascontiguousarray(np.asarray(x)).reshape(-1).view(np.uint8)

    out = _run_cases(prepare, call, to_numpy, raw, recompile, cases or list(CASES))
    info = {"framework": "jax", "jax": jax.__version__, "device": str(dev), "platform": dev.platform,
            "device_kind": getattr(dev, "device_kind", ""), "n_devices": len(jax.devices())}
    try:
        import jaxlib

        info["jaxlib"] = jaxlib.__version__
    except Exception:
        pass
    return out, info


class _TorchOps:
    def __init__(self, torch, dtype, acc32_float):
        self.t, self.dt, self.acc32_float = torch, dtype, acc32_float

    def mm(self, a, b, acc32=False):
        if acc32:  # bfloat16 values, products and sums in float32
            return self.t.matmul(a.float(), b.float())
        return self.t.matmul(a, b)

    def exp(self, x):
        return self.t.exp(x)

    def tanh(self, x):
        return self.t.tanh(x)

    def log(self, x):
        return self.t.log(x)

    def sqrt(self, x):
        return self.t.sqrt(x)

    def rsqrt(self, x):
        return self.t.rsqrt(x)

    def erf(self, x):
        return self.t.erf(x)

    def relu(self, x):
        return self.t.relu(x)

    def max_last(self, x):
        return x.amax(dim=-1, keepdim=True)

    def sum_last(self, x):
        return x.sum(dim=-1, keepdim=True)

    def sum_all(self, v):
        return v.sum()

    def tr(self, x):
        return x.transpose(-1, -2)

    def heads(self, y, b, s, h, dh):
        return y.reshape(b, s, h, dh).permute(0, 2, 1, 3)

    def merge(self, y, b, s, d):
        return y.permute(0, 2, 1, 3).reshape(b, s, d)


def run_torch(device: str = "cpu", tf32: bool = False, cases=None):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    def prepare(case, arrs):
        dt = torch.bfloat16 if CASES[case][1] == "bfloat16" else torch.float32
        return {k: torch.from_numpy(v.copy()).to(device=device, dtype=dt) for k, v in arrs.items()}

    def call(case):
        op, dtype = CASES[case][0], CASES[case][1]
        o = _TorchOps(torch, dtype, True)

        def f(t):
            with torch.no_grad():
                y = compute(o, op, t)
            if device != "cpu":
                torch.cuda.synchronize()
            return y
        return f

    def to_numpy(x):
        return x.detach().float().cpu().numpy()

    def raw(x):
        x = x.detach().cpu().reshape(-1)
        return (x.view(torch.int16) if x.dtype == torch.bfloat16 else x).numpy().view(np.uint8)

    out = _run_cases(prepare, call, to_numpy, raw, call, cases or list(CASES))
    info = {"framework": "torch", "torch": torch.__version__, "device": device, "tf32": tf32}
    if device != "cpu":
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return out, info


def run_torch_xla(cases=None):
    """PyTorch/XLA on the TPU, matmul and sum cases only, default and highest precision."""
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    res = {}
    for mode in ("default", "highest"):
        try:
            import torch_xla.backends

            torch_xla.backends.set_mat_mul_precision(mode)
            ok = True
        except Exception as exc:
            ok = False
            err = f"set_mat_mul_precision: {type(exc).__name__}: {exc}"
        for case in cases or ("mm_f32_default", "mm_bf16", "sum_f32_uniform"):
            key = f"{case}@xla_{mode}"
            if not ok:
                res[key] = {"error": err}
                continue
            arrs = inputs(case)
            dt = torch.bfloat16 if CASES[case][1] == "bfloat16" else torch.float32
            t = {k: torch.from_numpy(v.copy()).to(dtype=dt).to(dev) for k, v in arrs.items()}
            o = _TorchOps(torch, CASES[case][1], True)
            outs, hashes = [], []
            for _ in range(REPEATS):
                y = compute(o, CASES[case][0], t)
                y = y.cpu()
                outs.append(y)
                yy = y.view(torch.int16) if y.dtype == torch.bfloat16 else y
                hashes.append(_digest(yy.reshape(-1).numpy().view(np.uint8)))
            res[key] = {"out": outs[0].float().numpy(), "hashes": hashes}
    info = {"framework": "torch_xla", "torch": torch.__version__, "torch_xla": torch_xla.__version__,
            "device": str(dev)}
    return res, info


# ---------------------------------------------------------------------------------------------

def _case_of(key: str) -> str:
    return key.split("@", 1)[0]


def write(out_dir: str, backends: dict, extra: dict = None) -> dict:
    """backends: {name: (results, info)}. Writes .npy files and summary.json, returns the summary."""
    os.makedirs(out_dir, exist_ok=True)
    refs, arrs_c = {}, {}
    summary = {"battery_version": BATTERY_VERSION, "master_seed": MASTER_SEED,
               "cases": {c: {"op": v[0], "dtype": v[1], "precision": v[2]} for c, v in CASES.items()},
               "input_hashes": {}, "host": {"python": sys.version.split()[0], "platform": platform.platform(),
                                            "numpy": np.__version__},
               "backends": {}, **(extra or {})}
    for c in CASES:
        arrs_c[c] = inputs(c)
        summary["input_hashes"][c] = input_hash(arrs_c[c])
    for name, (res, info) in backends.items():
        bdir = os.path.join(out_dir, name)
        os.makedirs(bdir, exist_ok=True)
        rows = {}
        for key, r in res.items():
            c = _case_of(key)
            if "error" in r:
                rows[key] = {"error": r["error"]}
                continue
            if c not in refs:
                refs[c] = reference(c, arrs_c[c])
            np.save(os.path.join(bdir, key.replace("@", "__") + ".npy"), np.asarray(r["out"], np.float32))
            m = metrics(c, r["out"], refs[c], arrs_c[c])
            m["hashes"] = r["hashes"]
            m["bitwise_stable"] = len(set(r["hashes"])) == 1
            rows[key] = m
        summary["backends"][name] = {"info": info, "results": rows}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    return summary


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    runner = "auto"
    if "--runner" in argv:
        i = argv.index("--runner")
        runner = argv[i + 1]
        del argv[i:i + 2]
    out_dir = argv[0] if argv else "battery_out"
    t0 = time.time()
    backends = {}
    if runner in ("auto", "jax"):
        try:
            import jax

            plat = jax.devices()[0].platform
            backends[f"jax_{plat}"] = run_jax("default")
            print("jax", plat, "done", round(time.time() - t0, 1), flush=True)
            if plat != "cpu":
                backends["jax_cpu"] = run_jax("cpu")
                print("jax cpu done", round(time.time() - t0, 1), flush=True)
        except ImportError:
            if runner == "jax":
                raise
    if runner == "torch_xla" or (runner == "auto" and os.environ.get("TS_TORCH_XLA") == "1"):
        try:
            backends["torch_xla_tpu"] = run_torch_xla()
        except Exception as exc:
            backends["torch_xla_tpu"] = ({"_": {"error": f"{type(exc).__name__}: {exc}"[:500]}},
                                         {"framework": "torch_xla"})
        print("torch_xla done", round(time.time() - t0, 1), flush=True)
    if runner == "torch":
        import torch

        backends["torch_cpu"] = run_torch("cpu")
        if torch.cuda.is_available():
            backends["torch_cuda"] = run_torch("cuda", tf32=False)
            backends["torch_cuda_tf32"] = run_torch("cuda", tf32=True)
    s = write(out_dir, backends, {"wall_seconds": round(time.time() - t0, 1)})
    for b, v in s["backends"].items():
        for k, r in v["results"].items():
            if "error" in r:
                print(f"{b:16s} {k:28s} ERROR {r['error'][:100]}")
            else:
                print(f"{b:16s} {k:28s} rel_l2={r['rel_l2']:.3e} max_abs={r['max_abs']:.3e} "
                      f"bits={r.get('eff_bits', float('nan')):.1f} stable={r['bitwise_stable']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
