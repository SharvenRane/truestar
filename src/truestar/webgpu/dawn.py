"""Single op probes on Dawn directly (wgpudawn): our own WGSL compute shaders, no ONNX Runtime.

Why: ONNX Runtime's WebGPU EP picks its own shaders, tiling and accumulation; when a node is off
there, it is unclear whether the EP's kernel or the device's arithmetic is at fault. These probes
run textbook shaders whose every arithmetic step is known, so each result has two references:

* the float64 answer (how far the precision itself is from exact math), and
* the same algorithm emulated step by step in numpy in the same precision with round to nearest
  even (the ``ideal``: what a correct device computes with this exact shader).

A device is only blamed (TS566) when it is off from the emulation, i.e. when its own arithmetic
(rounding mode, flush to zero, exp accuracy, fused multiply add) differs from IEEE semantics by more
than the bar for the precision. Error against float64 is reported as information.

Probes: ``matmul`` (one invocation per output, sequential k loop), ``matmul_f32acc`` (f16 storage,
f32 accumulator; f16 only), ``sum`` (one invocation per row, sequential), ``sum_tree`` (workgroup
tree reduction of 256 lanes), ``softmax`` (max, exp, sum, divide per row).

Each backend runs in its own process (Dawn's instance options can be set once per process):
``python -m truestar.webgpu.dawn --backend Vulkan --adapter 0 --dtype f16``. Use :func:`run_probes`
from Python; it starts that process and returns its JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import numpy as np

# Bars on the error against the same precision emulation. WGSL allows f32 exp an error of
# 3 + 2|x| ULP and leaves f16 rounding implementation defined, so the bars sit far above one ULP
# but far below a wrong accumulator or a flushed range.
BAR = {"f32": 1e-5, "f16": 4e-3}
_NP = {"f32": np.float32, "f16": np.float16}


# ----------------------------------------------------------------------------- shaders


def _header(dtype: str) -> str:
    return "enable f16;\n" if dtype == "f16" else ""


def shader_matmul(dtype: str, acc: str) -> str:
    return _header(dtype) + f"""
struct Dims {{ m: u32, n: u32, k: u32, pad: u32 }};
@group(0) @binding(0) var<storage, read> a: array<{dtype}>;
@group(0) @binding(1) var<storage, read> b: array<{dtype}>;
@group(0) @binding(2) var<storage, read_write> c: array<{dtype}>;
@group(0) @binding(3) var<uniform> d: Dims;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
    let i = g.x; let j = g.y;
    if (i >= d.m || j >= d.n) {{ return; }}
    var s: {acc} = {acc}(0);
    for (var p: u32 = 0u; p < d.k; p = p + 1u) {{
        s = s + {acc}(a[i * d.k + p]) * {acc}(b[p * d.n + j]);
    }}
    c[i * d.n + j] = {dtype}(s);
}}
"""


def shader_sum(dtype: str) -> str:
    return _header(dtype) + f"""
struct Dims {{ m: u32, n: u32, k: u32, pad: u32 }};
@group(0) @binding(0) var<storage, read> a: array<{dtype}>;
@group(0) @binding(2) var<storage, read_write> c: array<{dtype}>;
@group(0) @binding(3) var<uniform> d: Dims;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
    let i = g.x;
    if (i >= d.m) {{ return; }}
    var s: {dtype} = {dtype}(0);
    for (var p: u32 = 0u; p < d.k; p = p + 1u) {{ s = s + a[i * d.k + p]; }}
    c[i] = s;
}}
"""


def shader_sum_tree(dtype: str) -> str:
    return _header(dtype) + f"""
struct Dims {{ m: u32, n: u32, k: u32, pad: u32 }};
@group(0) @binding(0) var<storage, read> a: array<{dtype}>;
@group(0) @binding(2) var<storage, read_write> c: array<{dtype}>;
@group(0) @binding(3) var<uniform> d: Dims;
var<workgroup> buf: array<{dtype}, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {{
    let row = w.x; let t = l.x;
    var s: {dtype} = {dtype}(0);
    for (var p: u32 = t; p < d.k; p = p + 256u) {{ s = s + a[row * d.k + p]; }}
    buf[t] = s;
    workgroupBarrier();
    var width: u32 = 128u;
    loop {{
        if (width == 0u) {{ break; }}
        if (t < width) {{ buf[t] = buf[t] + buf[t + width]; }}
        workgroupBarrier();
        width = width / 2u;
    }}
    if (t == 0u) {{ c[row] = buf[0]; }}
}}
"""


def shader_softmax(dtype: str) -> str:
    return _header(dtype) + f"""
struct Dims {{ m: u32, n: u32, k: u32, pad: u32 }};
@group(0) @binding(0) var<storage, read> a: array<{dtype}>;
@group(0) @binding(2) var<storage, read_write> c: array<{dtype}>;
@group(0) @binding(3) var<uniform> d: Dims;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
    let i = g.x;
    if (i >= d.m) {{ return; }}
    var mx: {dtype} = a[i * d.k];
    for (var p: u32 = 1u; p < d.k; p = p + 1u) {{ mx = max(mx, a[i * d.k + p]); }}
    var s: {dtype} = {dtype}(0);
    for (var p: u32 = 0u; p < d.k; p = p + 1u) {{
        let e = exp(a[i * d.k + p] - mx);
        c[i * d.k + p] = e;
        s = s + e;
    }}
    for (var p: u32 = 0u; p < d.k; p = p + 1u) {{ c[i * d.k + p] = c[i * d.k + p] / s; }}
}}
"""


# ----------------------------------------------------------------------------- emulation


def emulate_matmul(a, b, acc_dtype, out_dtype, variant: str = "separate"):
    """Sequential k loop. variant "separate": the product and the sum each round to the
    accumulator (what the WGSL source says); "fma": product plus sum rounded once (a fused
    multiply add, which a shader compiler may contract to); "f16_product": the product rounds to
    float16 before an f32 sum (the conversions to f32 dropped)."""
    s = np.zeros((a.shape[0], b.shape[1]), dtype=acc_dtype)
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    for p in range(a.shape[1]):
        prod = a64[:, p:p + 1] * b64[p:p + 1, :]  # exact for f32 and f16 inputs
        if variant == "fma":
            s = (s.astype(np.float64) + prod).astype(acc_dtype)
            continue
        if variant == "f16_product":
            prod = prod.astype(np.float16)
        s = (s + prod.astype(acc_dtype)).astype(acc_dtype)
    return s.astype(out_dtype)


def round_toward_zero_f16(x) -> np.ndarray:
    """float32 values to float16 rounded toward zero (numpy rounds to nearest even)."""
    x = np.asarray(x, np.float32)
    r = x.astype(np.float16)
    over = np.abs(r.astype(np.float32)) > np.abs(x)
    return np.where(over, np.nextafter(r, np.float16(0)), r).astype(np.float16)


def emulate_sum(a, dt):
    s = np.zeros(a.shape[0], dtype=dt)
    for p in range(a.shape[1]):
        s = (s + a[:, p]).astype(dt)
    return s


def emulate_sum_tree(a, dt):
    m, k = a.shape
    lanes = np.zeros((m, 256), dtype=dt)
    for start in range(0, k, 256):
        chunk = a[:, start:start + 256]
        lanes[:, :chunk.shape[1]] = (lanes[:, :chunk.shape[1]] + chunk).astype(dt)
    width = 128
    while width:
        lanes[:, :width] = (lanes[:, :width] + lanes[:, width:2 * width]).astype(dt)
        width //= 2
    return lanes[:, 0]


def emulate_softmax(a, dt):
    mx = a.max(axis=1, keepdims=True)
    e = np.exp((a - mx).astype(dt)).astype(dt)
    s = np.zeros((a.shape[0], 1), dtype=dt)
    for p in range(a.shape[1]):
        s = (s + e[:, p:p + 1]).astype(dt)
    return (e / s).astype(dt)


def _rel(x, ref) -> float:
    x = np.asarray(x, np.float64)
    ref = np.asarray(ref, np.float64)
    fin = np.isfinite(ref)
    if not np.array_equal(fin, np.isfinite(x)):
        return float("inf")
    den = np.linalg.norm(ref[fin])
    num = np.linalg.norm((x - ref)[fin])
    return 0.0 if num == 0 else (float(num / den) if den else float("inf"))


# ----------------------------------------------------------------------------- device side


class _Dawn:
    def __init__(self, backend: str, adapter: int, dtype: str):
        import wgpudawn as wgpu
        from wgpudawn.backends.dawn import extras

        toggles = []
        if dtype == "f16" and backend == "Vulkan":
            # Dawn hides shader-f16 on NVIDIA Vulkan unless this toggle is on (measured here: the
            # RTX 5070 Ti reports no shader-f16 on Vulkan without it, and with it, it does).
            toggles.append("vulkan_enable_f16_on_nvidia")
        extras.set_instance_extras(backends=[backend], enable_toggles=toggles)
        ads = [a for a in wgpu.gpu.enumerate_adapters_sync()
               if a.info["backend_type"] == backend]
        if adapter >= len(ads):
            raise RuntimeError(f"Dawn lists {len(ads)} {backend} adapter(s); index {adapter} "
                               "does not exist")
        self.adapter = ads[adapter]
        self.info = {k: self.adapter.info.get(k, "") for k in
                     ("device", "vendor", "adapter_type", "backend_type", "description")}
        self.info["toggles"] = toggles
        self.info["shader_f16"] = "shader-f16" in self.adapter.features
        if dtype == "f16" and not self.info["shader_f16"]:
            raise RuntimeError(f"{self.info['device']} on Dawn {backend} reports no shader-f16")
        feats = ["shader-f16"] if dtype == "f16" else []
        self.device = self.adapter.request_device_sync(required_features=feats)
        self.wgpu = wgpu

    def run(self, code: str, a: np.ndarray, b: Optional[np.ndarray], out_count: int, out_dtype,
            dims, groups) -> np.ndarray:
        wgpu, dev = self.wgpu, self.device
        U = wgpu.BufferUsage
        buf_a = dev.create_buffer_with_data(data=np.ascontiguousarray(a).tobytes(),
                                            usage=U.STORAGE)
        buf_b = None if b is None else dev.create_buffer_with_data(
            data=np.ascontiguousarray(b).tobytes(), usage=U.STORAGE)
        nbytes = max(16, out_count * np.dtype(out_dtype).itemsize)
        nbytes = (nbytes + 3) // 4 * 4
        buf_c = dev.create_buffer(size=nbytes, usage=U.STORAGE | U.COPY_SRC)
        buf_d = dev.create_buffer_with_data(data=np.array(dims, np.uint32).tobytes(),
                                            usage=U.UNIFORM)
        sm = dev.create_shader_module(code=code)
        pipe = dev.create_compute_pipeline(layout="auto",
                                           compute={"module": sm, "entry_point": "main"})
        bg = dev.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            {"binding": i, "resource": {"buffer": buf, "offset": 0, "size": buf.size}}
            for i, buf in enumerate((buf_a, buf_b, buf_c, buf_d)) if buf is not None])
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(*groups)
        cp.end()
        dev.queue.submit([enc.finish()])
        raw = dev.queue.read_buffer(buf_c)
        return np.frombuffer(bytes(raw), dtype=out_dtype)[:out_count].copy()


def probe_all(backend: str, adapter: int, dtype: str, seed: int = 0,
              sizes: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    sizes = dict({"m": 64, "n": 64, "k": 1024, "rows": 64, "len": 4096, "sm_len": 1024},
                 **(sizes or {}))
    dt = _NP[dtype]
    dev = _Dawn(backend, adapter, dtype)
    rng = np.random.default_rng(seed)
    results: List[Dict[str, Any]] = []

    def record(name, got, emu, exact, note="", bar=None, variants=None):
        bar = BAR[dtype] if bar is None else bar
        e_emu, e64 = _rel(got, emu), _rel(got, exact)
        bits_equal = float(np.mean(got == np.asarray(emu, dtype=got.dtype)))
        matches = {}
        for vname, v in (variants or {}).items():
            matches[vname] = float(np.mean(got == np.asarray(v, dtype=got.dtype)))
        # A variant other than the textbook one that matches the device bit for bit explains
        # the difference: the device does something WGSL permits (contraction to a fused
        # multiply add, a rounding direction for f16 conversions), reported, not blamed.
        explained = [k for k, f in matches.items() if f >= 0.99 and k != "separate"]
        results.append({"probe": name, "dtype": dtype, "rel_err_vs_emulation": e_emu,
                        "rel_err_emulation_vs_float64": _rel(emu, exact),
                        "rel_err_vs_float64": e64, "fraction_equal_to_emulation": bits_equal,
                        "fraction_equal_to_variant": matches, "explained_by": explained,
                        "bar": bar, "divergent": bool(e_emu > bar and not explained), "note": note})

    m, n, k = sizes["m"], sizes["n"], sizes["k"]
    a = rng.standard_normal((m, k)).astype(dt)
    b = rng.standard_normal((k, n)).astype(dt)
    exact = a.astype(np.float64) @ b.astype(np.float64)
    got = dev.run(shader_matmul(dtype, dtype), a, b, m * n, dt, (m, n, k, 0),
                  ((m + 7) // 8, (n + 7) // 8, 1)).reshape(m, n)
    record("matmul", got, emulate_matmul(a, b, dt, dt), exact,
           f"[{m},{k}] x [{k},{n}], accumulator {dtype}",
           variants={"separate": emulate_matmul(a, b, dt, dt),
                     "fma": emulate_matmul(a, b, dt, dt, "fma")})
    if dtype == "f16":
        got = dev.run(shader_matmul("f16", "f32"), a, b, m * n, dt, (m, n, k, 0),
                      ((m + 7) // 8, (n + 7) // 8, 1)).reshape(m, n)
        record("matmul_f32acc", got, emulate_matmul(a, b, np.float32, dt), exact,
               "f16 storage, f32 accumulator", bar=BAR["f32"] * 10,
               variants={"separate": emulate_matmul(a, b, np.float32, dt),
                         "f16_product": emulate_matmul(a, b, np.float32, dt, "f16_product"),
                         "rtz_cast": round_toward_zero_f16(
                             emulate_matmul(a, b, np.float32, np.float32))})
    rows, ln = sizes["rows"], sizes["len"]
    x = rng.random((rows, ln)).astype(dt)  # positive: the error of a sum grows with its length
    exact = x.astype(np.float64).sum(axis=1)
    got = dev.run(shader_sum(dtype), x, None, rows, dt, (rows, 0, ln, 0),
                  ((rows + 63) // 64, 1, 1))
    record("sum", got, emulate_sum(x, dt), exact, f"{rows} rows of {ln} positive values, sequential")
    got = dev.run(shader_sum_tree(dtype), x, None, rows, dt, (rows, 0, ln, 0), (rows, 1, 1))
    record("sum_tree", got, emulate_sum_tree(x, dt), exact, "256 lane workgroup tree reduction")
    sl = sizes["sm_len"]
    z = (rng.standard_normal((rows, sl)) * 4).astype(dt)
    z64 = z.astype(np.float64)
    e = np.exp(z64 - z64.max(axis=1, keepdims=True))
    exact = e / e.sum(axis=1, keepdims=True)
    got = dev.run(shader_softmax(dtype), z, None, rows * sl, dt, (rows, 0, sl, 0),
                  ((rows + 63) // 64, 1, 1)).reshape(rows, sl)
    record("softmax", got, emulate_softmax(z, dt), exact, f"{rows} rows of {sl}, logits std 4")
    return {"adapter": dev.info, "backend": backend, "adapter_index": adapter, "dtype": dtype,
            "results": results}


def run_probes(backend: str = "D3D12", adapter: int = 0, dtype: str = "f32",
               python: Optional[str] = None, timeout: float = 300.0) -> Dict[str, Any]:
    """Run every probe in a fresh process; returns {"adapter", "results"} or {"error"}."""
    from .client import _SRC, worker_python

    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    p = subprocess.run([python or worker_python(), "-m", "truestar.webgpu.dawn", "--backend",
                        backend, "--adapter", str(adapter), "--dtype", dtype],
                       capture_output=True, text=True, env=env, timeout=timeout)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        return {"error": f"probe process exited {p.returncode}: "
                         + (p.stderr.strip().splitlines() or ["no output"])[-1]}
    return json.loads(lines[-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m truestar.webgpu.dawn")
    ap.add_argument("--backend", default="D3D12", choices=["D3D12", "Vulkan", "Metal"])
    ap.add_argument("--adapter", type=int, default=0)
    ap.add_argument("--dtype", default="f32", choices=["f32", "f16"])
    a = ap.parse_args(argv)
    try:
        doc = probe_all(a.backend, a.adapter, a.dtype)
    except Exception as exc:
        doc = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(doc))
    if "error" in doc:
        return 2
    return 1 if any(r["divergent"] for r in doc["results"]) else 0


if __name__ == "__main__":
    sys.exit(main())
