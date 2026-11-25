"""`truestar conform llamacpp <gguf or hf dir>`: what an official llama.cpp build computes, judged in float64.

The engine runs unmodified: Truestar compiles its own small driver (`llamacpp_driver.cpp`) against
the llama.h of the build under test, links it with that build's libllama, and reads the float32
logits llama_decode produced for fixed token ids, teacher forced and greedy, under each engine
setting (threads, batch, flash attention). Nothing is tapped, sampled or patched.

Three verdicts that never mix (docs/CONFORM_LLAMACPP.md):

* **engine**: the engine's logits against Truestar's float64 replay of the GGUF's own weights,
  decoded exactly (`engines/gguf.py`). Quantization cannot show up here: both sides hold the same
  quantized weights. The tolerance per position is derived, not chosen: 10 times the larger of the
  float32 replay's own error and the effect of the rounding the engine is allowed to do (activations
  rounded to the dot product type of each weight, q, k, v and the probabilities rounded to f16 for
  an f16 KV cache). Greedy decoding is replayed teacher forced on the engine's own tokens; the first
  step where the engine picks another token than the float64 replay is a bug, unless the replay's
  top two logits are within twice the per element rounding bound there (a near tie, ambiguous).
* **conversion** (with a source checkpoint): every GGUF setting by name and every tensor by content
  against the source weights (`engines/engine_port.match_weights`).
* **quantization** (with a source checkpoint): float64 replays only, the GGUF's decoded weights
  against the source weights. No engine output enters it.

Then the settings: do the engine's bits, or its greedy tokens, change with threads, batch or flash
attention?
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..codes import Issue
from ..engines import gguf
from ..engines import llama_ref as R
from ..engines.llama_ref import F64, LlamaConfig

ARCHS = ("llama", "qwen2")
FORMATS = ("f32", "f16", "q8_0", "q4_k_m")
FACTOR = 10.0           # tolerance = 10 x the larger of float32 noise and the allowed rounding effect
FLOOR = 1e-6            # never below this relative error (the engine computes in float32)
QUANT_TIE = 0.01        # a quantization flip where the source's top two are closer than 1% of the logit std is a near tie
DEFAULT_THREADS = (1, 4)
DEFAULT_BATCH = (512, 7)
DEFAULT_FLASH = (False, True)
FINDINGS = ("TS1311", "TS1312", "TS1316")
UNUSABLE = ("TS1310", "TS1319")

HF_LAYER = {"input_layernorm.weight": "attn_norm", "self_attn.q_proj.weight": "q", "self_attn.k_proj.weight": "k",
            "self_attn.v_proj.weight": "v", "self_attn.o_proj.weight": "o",
            "post_attention_layernorm.weight": "ffn_norm", "mlp.gate_proj.weight": "gate",
            "mlp.up_proj.weight": "up", "mlp.down_proj.weight": "down"}
HF_BIAS = {"self_attn.q_proj.bias": "q", "self_attn.k_proj.bias": "k", "self_attn.v_proj.bias": "v"}
HF_GLOBAL = {"model.embed_tokens.weight": "embed", "model.norm.weight": "norm", "lm_head.weight": "output"}


class LlamaCppError(Exception):
    """The check cannot run: TS1310 (no usable llama.cpp build) or TS1319 (unsupported model, or a
    llama.cpp tool failed)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _rel_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.to(F64), b.to(F64)
    n = b.norm(dim=-1)
    return (a - b).norm(dim=-1) / torch.where(n > 0, n, torch.ones_like(n))


# ----------------------------------------------------------------------------- weights


@dataclass
class Weights:
    """A Llama or Qwen2 style decoder in Truestar's reference layout (q and k rows in the half
    rotation order), float64, with the GGUF tensor type of every weight ("source" for a checkpoint)."""
    cfg: LlamaConfig
    arch: str
    W: Dict[str, torch.Tensor]
    B: Dict[str, torch.Tensor] = field(default_factory=dict)
    types: Dict[str, str] = field(default_factory=dict)
    raw: Dict[str, torch.Tensor] = field(default_factory=dict)  # source tensors by checkpoint name


def load_hf_dir(path: str) -> Weights:
    from ..interop.safetensors import load_file

    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        raise LlamaCppError("TS1319", f"{path} has no config.json (a Hugging Face model directory is expected)")
    with open(cfg_path, encoding="utf-8") as f:
        c = json.load(f)
    arch = str(c.get("model_type", ""))
    if arch not in ARCHS:
        raise LlamaCppError("TS1319", f"model_type {arch!r} is not replayed in float64 (llama and qwen2 are)")
    files = sorted(n for n in os.listdir(path) if n.endswith(".safetensors"))
    if not files:
        raise LlamaCppError("TS1319", f"{path} has no .safetensors weights")
    raw: Dict[str, torch.Tensor] = {}
    for n in files:
        raw.update(load_file(os.path.join(path, n)))
    try:
        cfg = LlamaConfig.from_hf(SimpleNamespace(**c))
    except (AttributeError, ValueError) as exc:
        raise LlamaCppError("TS1319", f"config.json: {exc}") from exc
    W: Dict[str, torch.Tensor] = {}
    B: Dict[str, torch.Tensor] = {}
    unknown = []
    for name, t in raw.items():
        t64 = t.to(F64)
        if name in HF_GLOBAL:
            W[HF_GLOBAL[name]] = t64
            continue
        parts = name.split(".", 3)
        if len(parts) == 4 and parts[0] == "model" and parts[1] == "layers":
            i, rest = int(parts[2]), parts[3]
            if rest in HF_LAYER:
                W[f"{i}.{HF_LAYER[rest]}"] = t64
                continue
            if rest in HF_BIAS:
                B[f"{i}.{HF_BIAS[rest]}"] = t64
                continue
        unknown.append(name)
    if unknown:
        raise LlamaCppError("TS1319", f"tensors the float64 replay does not know: {', '.join(sorted(unknown)[:6])}")
    cfg.tied = "output" not in W
    return Weights(cfg, arch, W, B, {k: "source" for k in W}, raw)


def load_gguf(path: str) -> Weights:
    from ..engines.engine_port import EnginePortError, _gguf_weights, gguf_config, unpermute_rows

    try:
        g = gguf.read(path)
    except (gguf.GGUFError, OSError, struct.error) as exc:
        raise LlamaCppError("TS1319", f"{path}: {exc}") from exc
    arch = g.arch()
    if arch not in ARCHS:
        raise LlamaCppError("TS1319", f"GGUF architecture {arch!r} is not replayed in float64 (llama and qwen2 are)")
    try:
        cfg = gguf_config(g)
        G = _gguf_weights(g)
    except EnginePortError as exc:
        raise LlamaCppError("TS1319", exc.message if hasattr(exc, "message") else str(exc)) from exc
    W = {}
    types = {}
    for role, (t, tname) in G.items():
        if arch == "llama" and role.endswith((".q", ".k")):  # the converter permutes q and k rows per head
            t = unpermute_rows(t, cfg.n_heads if role.endswith(".q") else cfg.n_kv_heads)
        W[role], types[role] = t.to(F64), tname
    B = {}
    for name, ti in g.tensors.items():
        parts = name.split(".")
        if len(parts) == 4 and parts[0] == "blk" and parts[3] == "bias":
            role = {"attn_q": "q", "attn_k": "k", "attn_v": "v"}.get(parts[2])
            if role is None:
                raise LlamaCppError("TS1319", f"bias {name} is not replayed")
            B[f"{int(parts[1])}.{role}"] = torch.from_numpy(g.tensor(name)).to(F64)
    missing = [k for k in ("embed", "norm") if k not in W]
    if missing:
        raise LlamaCppError("TS1319", f"{path} has no {', '.join(missing)} tensor")
    return Weights(cfg, arch, W, B, types)


# ----------------------------------------------------------------------------- float64 replay


def _f16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).to(torch.float16).to(x.dtype)


def forward(wt: Weights, ids: Sequence[int], dt=F64, round_acts: bool = False, kv: str = "f32") -> torch.Tensor:
    """Logits [T, vocab] of a teacher forced pass. With `round_acts`, every matmul input is rounded
    the way the ggml CPU backend feeds that weight type, and for an f16 KV cache q, k, v and the
    attention probabilities are rounded to f16 (the operands of the two attention products)."""
    cfg = wt.cfg
    ids_t = torch.as_tensor(list(ids), dtype=torch.long)
    pos = torch.arange(ids_t.shape[0])
    D = cfg.head_dim

    def lin(x, role):
        if round_acts and wt.types.get(role, "F32") not in ("F32", "source"):
            x = torch.from_numpy(gguf.round_like_dot_type(x.detach().to(torch.float32).numpy(), wt.types[role])).to(dt)
        y = R.linear(x, wt.W[role], dt)
        if role in wt.B:
            y = y + wt.B[role].to(dt)
        return y

    rounding = {n: _f16 for n in ("q", "k", "v", "p")} if (round_acts and kv == "f16") else None
    h = wt.W["embed"].to(dt)[ids_t]
    for i in range(cfg.n_layers):
        x = R.rms_norm(h, wt.W[f"{i}.attn_norm"], cfg.eps, dt)
        q = R.rope(lin(x, f"{i}.q"), pos, cfg.n_heads, D, cfg.theta, cfg.rope_scale, dt)
        k = R.rope(lin(x, f"{i}.k"), pos, cfg.n_kv_heads, D, cfg.theta, cfg.rope_scale, dt)
        v = lin(x, f"{i}.v")
        a = R.attention(q, k, v, cfg.n_heads, cfg.n_kv_heads, D, dt, rounding)
        h = h + lin(a, f"{i}.o")
        x = R.rms_norm(h, wt.W[f"{i}.ffn_norm"], cfg.eps, dt)
        h = h + lin(R.silu_mul(lin(x, f"{i}.gate"), lin(x, f"{i}.up"), dt), f"{i}.down")
    x = R.rms_norm(h, wt.W["norm"], cfg.eps, dt)
    return lin(x, "output" if "output" in wt.W else "embed").to(F64)


@dataclass
class Replay:
    """The float64 replay of one token sequence, with what bounds the engine on it."""
    l64: torch.Tensor   # [T, V]
    tau: torch.Tensor   # per position relative tolerance
    bound: torch.Tensor  # per position per element bound (absolute)
    gap: torch.Tensor   # top 1 minus top 2 of l64


def replay(wt: Weights, ids: Sequence[int], kv: str) -> Replay:
    l64 = forward(wt, ids)
    l32 = forward(wt, ids, dt=torch.float32)
    lr = forward(wt, ids, round_acts=True, kv=kv)
    tau = FACTOR * torch.maximum(_rel_rows(l32, l64), _rel_rows(lr, l64)).clamp_min(FLOOR)
    bound = FACTOR * torch.maximum((l32 - l64).abs().amax(-1), (lr - l64).abs().amax(-1))
    top = l64.topk(2, dim=-1).values
    return Replay(l64, tau, bound, top[:, 0] - top[:, 1])


# ----------------------------------------------------------------------------- the build and the driver


@dataclass
class Build:
    root: str
    lib_dir: str
    include_dirs: List[str]
    quantize: Optional[str]
    converter: Optional[str]
    version: Dict[str, Any] = field(default_factory=dict)


_LIB = {"linux": "libllama.so", "darwin": "libllama.dylib", "win32": "llama.dll"}


def find_build(path: Optional[str] = None) -> Build:
    root = path or os.environ.get("TRUESTAR_LLAMA_CPP")
    if not root:
        raise LlamaCppError("TS1310", "no llama.cpp build given: pass --llama-cpp DIR (a llama.cpp source tree "
                                      "built with cmake, or an install prefix) or set TRUESTAR_LLAMA_CPP")
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise LlamaCppError("TS1310", f"llama.cpp directory {root} does not exist")
    lib = _LIB.get(sys.platform, "libllama.so")
    lib_dir = next((d for d in (os.path.join(root, "build", "bin"), os.path.join(root, "build", "src"),
                                os.path.join(root, "lib"), os.path.join(root, "build", "lib"), root)
                    if os.path.isfile(os.path.join(d, lib))), None)
    if lib_dir is None:
        raise LlamaCppError("TS1310", f"{lib} not found under {root} (build/bin, build/src, lib); build llama.cpp "
                                      "with shared libraries (cmake default)")
    incs = [d for d in (os.path.join(root, "include"), os.path.join(root, "ggml", "include")) if os.path.isdir(d)]
    if not any(os.path.isfile(os.path.join(d, "llama.h")) for d in incs) or \
            not any(os.path.isfile(os.path.join(d, "ggml.h")) for d in incs):
        raise LlamaCppError("TS1310", f"llama.h and ggml.h not found under {root}/include and {root}/ggml/include; "
                                      "the driver is compiled against the build's own headers")
    exe = ".exe" if sys.platform == "win32" else ""
    quant = next((p for p in (os.path.join(lib_dir, "llama-quantize" + exe), os.path.join(root, "bin", "llama-quantize" + exe))
                  if os.path.isfile(p)), None)
    conv = os.path.join(root, "convert_hf_to_gguf.py")
    b = Build(root, lib_dir, incs, quant, conv if os.path.isfile(conv) else None)
    b.version = _version(b, lib)
    return b


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _version(b: Build, lib: str) -> Dict[str, Any]:
    v: Dict[str, Any] = {"root": b.root, "libllama_sha256": _sha256(os.path.join(b.lib_dir, lib))}
    if os.path.isdir(os.path.join(b.root, ".git")):
        try:
            v["git_describe"] = subprocess.run(["git", "-C", b.root, "describe", "--tags", "--always", "--dirty"],
                                               capture_output=True, text=True, errors="replace", timeout=20).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            pass
    for name in ("TRUESTAR_SOURCE", ".truestar-source"):  # a note written next to a tarball build, if any
        p = os.path.join(b.root, name)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                v["source_note"] = f.read().strip()[:400]
    return v


def _cache_dir() -> str:
    d = os.environ.get("TRUESTAR_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "truestar")
    d = os.path.join(d, "llamacpp")
    os.makedirs(d, exist_ok=True)
    return d


def build_driver(b: Build) -> str:
    """Compile Truestar's driver against this build's headers and libllama (cached by content)."""
    if sys.platform == "win32":
        raise LlamaCppError("TS1310", "the llama.cpp driver is compiled with a POSIX C++ compiler; building it with "
                                      "MSVC on Windows is not supported yet (use a Linux or macOS llama.cpp build)")
    src = os.path.join(os.path.dirname(__file__), "llamacpp_driver.cpp")
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        raise LlamaCppError("TS1310", "no C++ compiler found (set CXX) to build the llama.cpp driver")
    h = hashlib.sha256()
    for p in [src] + [os.path.join(d, n) for d in b.include_dirs for n in ("llama.h", "ggml.h") if os.path.isfile(os.path.join(d, n))]:
        with open(p, "rb") as f:
            h.update(f.read())
    h.update(b.lib_dir.encode())
    h.update(b.version.get("libllama_sha256", "").encode())
    out = os.path.join(_cache_dir(), f"driver-{h.hexdigest()[:16]}")
    if os.path.isfile(out):
        return out
    libs = [n for n in ("llama", "ggml", "ggml-base") if any(f.startswith(f"lib{n}.") for f in os.listdir(b.lib_dir))]
    cmd = [cxx, "-O2", "-std=c++17", src, "-o", out + ".tmp"] + [a for d in b.include_dirs for a in ("-I", d)] + \
          ["-L", b.lib_dir] + [f"-l{n}" for n in libs] + [f"-Wl,-rpath,{b.lib_dir}"]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=600)
    if r.returncode != 0:
        raise LlamaCppError("TS1310", "building the driver against this llama.cpp failed: "
                                      + (r.stderr.strip().splitlines() or ["no output"])[-1][:400])
    os.replace(out + ".tmp", out)
    return out


@dataclass
class Setting:
    threads: int
    batch: int
    flash: bool
    kv: str = "f16"
    rope_base: float = 0.0  # 0: the file's; anything else overrides it (a user setting, or a planted fault)

    @property
    def label(self) -> str:
        return (f"threads {self.threads}, batch {self.batch}, flash attention {'on' if self.flash else 'off'}, "
                f"kv {self.kv}" + (f", rope base override {self.rope_base:g}" if self.rope_base else ""))

    def as_dict(self) -> Dict[str, Any]:
        return {"threads": self.threads, "batch": self.batch, "ubatch": self.batch, "flash_attention": self.flash,
                "kv_cache": self.kv, "rope_freq_base_override": self.rope_base or None}


@dataclass
class EngineOutput:
    prompt_logits: List[np.ndarray]      # per prompt [T, V] float32
    greedy_tokens: List[List[int]]
    greedy_logits: List[np.ndarray]      # per prompt [NEW, V]
    info: Dict[str, Any] = field(default_factory=dict)


def parse_output(blob: bytes) -> Tuple[int, List[Tuple[int, List[int], np.ndarray]]]:
    if blob[:4] != b"TSLC":
        raise LlamaCppError("TS1310", "the driver wrote no logits file")
    ver, vocab, n = struct.unpack_from("<III", blob, 4)
    off = 16
    items = []
    for _ in range(n):
        kind, rows = struct.unpack_from("<II", blob, off)
        off += 8
        toks = list(struct.unpack_from(f"<{rows}i", blob, off))
        off += 4 * rows
        lg = np.frombuffer(blob, dtype="<f4", count=rows * vocab, offset=off).reshape(rows, vocab).copy()
        off += 4 * rows * vocab
        items.append((kind, toks, lg))
    return vocab, items


def run_driver(driver: str, gguf_path: str, s: Setting, prompts: Sequence[Sequence[int]], n_new: int,
               timeout: float = 600) -> EngineOutput:
    ctx = max(256, ((max(len(p) for p in prompts) + n_new + 8 + 255) // 256) * 256)
    lines = [f"threads {s.threads}", f"batch {s.batch}", f"ubatch {s.batch}", f"flash {int(s.flash)}",
             f"ctx {ctx}", f"kv {s.kv}", f"rope_base {s.rope_base!r}"]
    lines += [f"prompt {len(p)} " + " ".join(map(str, p)) for p in prompts]
    lines += [f"greedy {n_new} {len(p)} " + " ".join(map(str, p)) for p in prompts]
    with tempfile.TemporaryDirectory(prefix="truestar-llamacpp-") as d:
        req, out = os.path.join(d, "request.txt"), os.path.join(d, "out.bin")
        with open(req, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        r = subprocess.run([driver, gguf_path, req, out], capture_output=True, text=True, errors="replace", timeout=timeout)
        if r.returncode != 0 or not os.path.isfile(out):
            raise LlamaCppError("TS1310", f"llama.cpp failed on {os.path.basename(gguf_path)} ({s.label}): "
                                + (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1][:400])
        with open(out, "rb") as f:
            _, items = parse_output(f.read())
    info: Dict[str, Any] = {}
    for line in r.stdout.splitlines():
        key, _, rest = line.partition(" ")
        if key == "info":
            info.update(dict(kv.split("=", 1) for kv in rest.split() if "=" in kv))
        elif key == "ggml":
            info.update({"ggml_" + k: v for k, v in (kv.split("=", 1) for kv in rest.split() if "=" in kv)})
        elif key == "system":
            info["system"] = rest.strip()
    n = len(prompts)
    return EngineOutput([it[2] for it in items[:n]], [it[1] for it in items[n:]], [it[2] for it in items[n:]], info)


def convert(b: Build, hf_dir: str, workdir: str, formats: Sequence[str], python: Optional[str] = None,
            threads: int = 4) -> Dict[str, str]:
    """GGUF files made by llama.cpp's own converter (f32, f16, q8_0) and quantizer (q4_k_m, from f32)."""
    out: Dict[str, str] = {}
    base = os.path.basename(os.path.normpath(hf_dir))
    py = python or sys.executable
    for fmt in formats:
        if fmt not in FORMATS:
            raise LlamaCppError("TS1319", f"format {fmt!r} is not one of {', '.join(FORMATS)}")
    need_f32 = any(f in ("f32", "q4_k_m") for f in formats)
    for fmt in (["f32"] if need_f32 else []) + [f for f in formats if f in ("f16", "q8_0")]:
        if b.converter is None:
            raise LlamaCppError("TS1310", f"convert_hf_to_gguf.py not found in {b.root}")
        path = os.path.join(workdir, f"{base}-{fmt}.gguf")
        r = subprocess.run([py, b.converter, hf_dir, "--outtype", fmt, "--outfile", path],
                           capture_output=True, text=True, errors="replace", timeout=1800)
        if r.returncode != 0 or not os.path.isfile(path):
            raise LlamaCppError("TS1319", f"llama.cpp's converter failed for {fmt}: "
                                + (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1][:400])
        out[fmt] = path
    if "q4_k_m" in formats:
        if b.quantize is None:
            raise LlamaCppError("TS1310", f"llama-quantize not found in {b.lib_dir}")
        path = os.path.join(workdir, f"{base}-q4_k_m.gguf")
        r = subprocess.run([b.quantize, out["f32"], path, "Q4_K_M", str(threads)], capture_output=True, text=True, errors="replace",
                           timeout=1800)
        if r.returncode != 0 or not os.path.isfile(path):
            raise LlamaCppError("TS1319", "llama-quantize failed for Q4_K_M: "
                                + (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1][:400])
        out["q4_k_m"] = path
    return {f: out[f] for f in formats}


# ----------------------------------------------------------------------------- verdicts


def _f(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2e}"


def _num(x) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


@dataclass
class GreedyResult:
    prompt: int
    tokens: List[int]
    first_divergent: Optional[int] = None   # step where the engine's token differs from the float64 replay
    near_tie: bool = False
    gap: Optional[float] = None
    bound: Optional[float] = None
    ref_token: Optional[int] = None
    worst_ratio: float = 0.0               # engine logit error / tolerance over the steps
    quant_first: Optional[int] = None      # step where the source weights' replay picks another token

    def as_dict(self) -> Dict[str, Any]:
        return {"prompt": self.prompt, "tokens": self.tokens, "first_divergent_step": self.first_divergent,
                "near_tie": self.near_tie if self.first_divergent is not None else None, "gap": _num(self.gap),
                "bound": _num(self.bound), "replay_token": self.ref_token, "worst_ratio": _num(self.worst_ratio),
                "quantization_first_step": self.quant_first}


@dataclass
class RunResult:
    format: str
    setting: Setting
    positions: int = 0
    over: List[Dict[str, Any]] = field(default_factory=list)  # positions over tolerance
    worst_ratio: float = 0.0
    max_rel: float = 0.0
    flips: int = 0
    flips_near_tie: int = 0
    worst_abs_ratio: float = 0.0  # engine max abs deviation / (bound / FACTOR): how much of the per element bound it used
    greedy: List[GreedyResult] = field(default_factory=list)
    total_rel: Optional[float] = None       # engine against the source weights' replay (prompts)
    digest: str = ""
    info: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"format": self.format, "setting": self.setting.as_dict(), "positions": self.positions,
                "positions_over_tolerance": self.over[:20], "n_over": len(self.over),
                "worst_ratio": _num(self.worst_ratio), "max_rel_err": _num(self.max_rel), "flips": self.flips,
                "flips_near_tie": self.flips_near_tie, "worst_abs_ratio": _num(self.worst_abs_ratio),
                "greedy": [g.as_dict() for g in self.greedy],
                "total_rel_err_vs_source": _num(self.total_rel), "logits_sha256": self.digest, "engine": self.info}


@dataclass
class FormatResult:
    format: str
    path: str
    tensor_types: Dict[str, int]
    runs: List[RunResult] = field(default_factory=list)
    conversion: List[str] = field(default_factory=list)
    quantization: Optional[Dict[str, Any]] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    issues: List[Tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"format": self.format, "path": self.path, "tensor_types": self.tensor_types,
                "conversion_findings": self.conversion, "quantization": self.quantization,
                "settings": self.settings, "runs": [r.as_dict() for r in self.runs],
                "issues": [Issue(c, m).as_dict() for c, m in self.issues]}


@dataclass
class LlamaCppReport:
    target: str
    source: Optional[str]
    build: Dict[str, Any]
    prompts: List[List[int]]
    n_new: int
    formats: List[FormatResult] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def issues(self) -> List[Tuple[str, str]]:
        return [i for f in self.formats for i in f.issues]

    @property
    def codes(self) -> List[str]:
        return sorted({c for c, _ in self.issues})

    @property
    def finding(self) -> bool:
        return any(c in FINDINGS for c in self.codes)

    def as_dict(self) -> Dict[str, Any]:
        return {"target": self.target, "source": self.source, "llama_cpp": self.build, "prompts": self.prompts,
                "new_tokens": self.n_new, "formats": [f.as_dict() for f in self.formats],
                "seconds": round(self.seconds, 3)}

    def to_text(self) -> str:
        v = self.build
        out = [f"truestar conform llamacpp: {self.target}" + (f" (source {self.source})" if self.source else ""),
               f"llama.cpp: {v.get('git_describe') or v.get('source_note') or v.get('root')}; ggml "
               f"{v.get('ggml_version', '?')} commit {v.get('ggml_commit', '?')}; libllama sha256 "
               f"{str(v.get('libllama_sha256', ''))[:12]}",
               f"prompts: {len(self.prompts)} x {len(self.prompts[0]) if self.prompts else 0} tokens teacher forced, "
               f"then {self.n_new} greedy steps each; reference: float64 replay of the GGUF's decoded weights"]
        for f in self.formats:
            types = ", ".join(f"{n} {k}" for k, n in sorted(f.tensor_types.items()))
            out.append("")
            out.append(f"[{f.format}] {os.path.basename(f.path)}: {types}")
            if f.quantization:
                q = f.quantization
                out.append(f"  quantization (float64 replays only): rel err median {_f(q['rel_median'])}, max "
                           f"{_f(q['rel_max'])}; top 1 agreement {100 * q['top1_agreement']:.1f}% "
                           f"({q['flips']} flips, {q['flips_beyond_near_ties']} beyond near ties); "
                           f"greedy steps changed by quantization: {q['greedy_changed']}")
            for c in f.conversion:
                out.append(f"  CONVERSION: {c}")
            for r in f.runs:
                g_bad = [g for g in r.greedy if g.first_divergent is not None]
                gtxt = "greedy same as replay" if not g_bad else "greedy differs: " + "; ".join(
                    f"prompt {g.prompt} step {g.first_divergent} ({'near tie' if g.near_tie else 'real'}, gap "
                    f"{g.gap:.2e}, bound {g.bound:.2e})" for g in g_bad)
                state = "OK" if not r.over and not any(not g.near_tie for g in g_bad) else "DIVERGENT"
                out.append(f"  {state:9s} {r.setting.label}: worst error {r.worst_ratio:.2f} of tolerance "
                           f"(max rel {r.max_rel:.2e}); {len(r.over)} of {r.positions} positions over; "
                           f"flips {r.flips} ({r.flips_near_tie} near ties); {gtxt}")
            s = f.settings
            if s:
                out.append(f"  settings: {s['groups']} distinct result(s) over {s['runs']} settings"
                           + (f"; greedy tokens change with settings ({s['token_changes']})" if s.get("token_changes")
                              else "; greedy tokens identical across settings")
                           + (f"; max difference between settings {s['max_rel_between']:.2e}" if s["groups"] > 1 else ""))
            for c, m in f.issues:
                out.append(f"  {c}: {m}")
        status = ("DIVERGENT" if self.finding else "OK")
        out.append("")
        out.append(f"status: {status} ({self.seconds:.1f} s)")
        return "\n".join(out)


def _types(wt: Weights) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t in wt.types.values():
        out[t] = out.get(t, 0) + 1
    return out


def default_prompts(vocab: int, n: int = 3, length: int = 24, bos: Optional[int] = None) -> List[List[int]]:
    rng = np.random.default_rng(1310)
    low = 3 if vocab > 16 else 0
    ps = [[int(x) for x in rng.integers(low, vocab, size=length)] for _ in range(n)]
    if bos is not None and ps:
        ps[0][0] = int(bos)
    return ps


def default_settings(threads=DEFAULT_THREADS, batch=DEFAULT_BATCH, flash=DEFAULT_FLASH, kv="f16") -> List[Setting]:
    return [Setting(t, b, fl, kv) for t, b, fl in itertools.product(threads, batch, flash)]


Runner = Callable[[str, Setting, List[List[int]], int], EngineOutput]


def check_format(fmt: str, path: str, file_w: Weights, src_w: Optional[Weights], settings: Sequence[Setting],
                 prompts: List[List[int]], n_new: int, runner: Runner) -> FormatResult:
    fr = FormatResult(fmt, path, _types(file_w))
    cache: Dict[Tuple[Tuple[int, ...], str], Replay] = {}
    src_cache: Dict[Tuple[int, ...], torch.Tensor] = {}

    def rp(ids, kv):
        key = (tuple(ids), kv)
        if key not in cache:
            cache[key] = replay(file_w, ids, kv)
        return cache[key]

    def src_logits(ids):
        key = tuple(ids)
        if key not in src_cache:
            src_cache[key] = forward(src_w, ids)
        return src_cache[key]

    if src_w is not None:
        fr.conversion = _conversion(path, file_w, src_w)
        if fr.conversion:
            fr.issues.append(("TS1316", f"{fmt}: " + "; ".join(fr.conversion[:3])))
        fr.quantization = _quantization(file_w, src_w, prompts, rp, src_logits)
    tokens_by_setting: List[List[List[int]]] = []
    logits_by_setting: List[List[np.ndarray]] = []
    greedy_seen: Dict[Tuple, list] = {}
    for s in settings:
        eo = runner(path, s, prompts, n_new)
        rr = RunResult(fmt, s, info=dict(eo.info))
        h = hashlib.sha256()
        for i, p in enumerate(prompts):
            e = torch.from_numpy(eo.prompt_logits[i]).to(F64)
            ref = rp(p, s.kv)
            err = _rel_rows(e, ref.l64)
            ratio = err / ref.tau
            rr.positions += len(p)
            rr.worst_ratio = max(rr.worst_ratio, float(ratio.max()))
            rr.max_rel = max(rr.max_rel, float(err.max()))
            for j in torch.nonzero(ratio > 1).flatten().tolist():
                rr.over.append({"prompt": i, "position": j, "rel_err": float(err[j]), "tolerance": float(ref.tau[j])})
            flip = e.argmax(-1) != ref.l64.argmax(-1)
            near = ref.gap <= 2 * ref.bound
            rr.flips += int(flip.sum())
            rr.flips_near_tie += int((flip & near).sum())
            rr.worst_abs_ratio = max(rr.worst_abs_ratio, float(((e - ref.l64).abs().amax(-1) * FACTOR / ref.bound).max()))
            if src_w is not None:
                rr.total_rel = max(rr.total_rel or 0.0, float(_rel_rows(e, src_logits(p)).max()))
            h.update(eo.prompt_logits[i].tobytes())
        for i, p in enumerate(prompts):
            toks = list(eo.greedy_tokens[i])
            gl = torch.from_numpy(eo.greedy_logits[i]).to(F64)
            seq = list(p) + toks[:-1]
            ref = rp(seq, s.kv)
            rows = slice(len(p) - 1, len(p) - 1 + len(toks))
            l64, tau, bound, gap = ref.l64[rows], ref.tau[rows], ref.bound[rows], ref.gap[rows]
            g = GreedyResult(i, toks, worst_ratio=float((_rel_rows(gl, l64) / tau).max()))
            pick = l64.argmax(-1).tolist()
            for j, t in enumerate(toks):
                if t != pick[j]:
                    g.first_divergent, g.ref_token = j, pick[j]
                    g.gap, g.bound = float(gap[j]), float(bound[j])
                    g.near_tie = bool(gap[j] <= 2 * bound[j])
                    break
            if src_w is not None:
                sp = src_logits(seq)[rows].argmax(-1).tolist()
                g.quant_first = next((j for j, t in enumerate(toks) if t != sp[j]), None)
            rr.worst_ratio = max(rr.worst_ratio, g.worst_ratio)
            rr.worst_abs_ratio = max(rr.worst_abs_ratio, float(((gl - l64).abs().amax(-1) * FACTOR / bound).max()))
            rr.greedy.append(g)
            h.update(eo.greedy_logits[i].tobytes())
            h.update(np.asarray(toks, dtype="<i4").tobytes())
        rr.digest = h.hexdigest()
        tokens_by_setting.append([g.tokens for g in rr.greedy])
        logits_by_setting.append(eo.prompt_logits)
        fr.runs.append(rr)
        if rr.over:
            w = max(rr.over, key=lambda o: o["rel_err"] / o["tolerance"])
            fr.issues.append(("TS1311", f"{fmt}, {s.label}: {len(rr.over)} of {rr.positions} positions over the derived "
                                        f"tolerance; worst prompt {w['prompt']} position {w['position']}, rel err "
                                        f"{w['rel_err']:.2e} vs {w['tolerance']:.2e}"))
        for g in rr.greedy:
            if g.first_divergent is None:
                continue
            key = (g.prompt, g.first_divergent, g.tokens[g.first_divergent], g.ref_token, g.near_tie)
            greedy_seen.setdefault(key, [g, []])[1].append(s.label)
    for (pi, step, tok, ref_tok, near), (g, labels) in greedy_seen.items():
        what = (f"{fmt}: greedy prompt {pi} step {step}: llama.cpp picks token {tok}, the float64 replay {ref_tok}; "
                f"top two gap {g.gap:.2e}, rounding bound {g.bound:.2e}; in {len(labels)} of {len(settings)} settings"
                + ("" if len(labels) == len(settings) else " (" + "; ".join(labels[:4]) + ")"))
        fr.issues.append(("TS1313", what + " (near tie: ambiguous, not a bug)") if near else ("TS1312", what))
    fr.settings = _settings_summary(fr.runs, tokens_by_setting, prompts, rp, logits_by_setting)
    if fr.settings.get("token_changes"):
        fr.issues.append(("TS1317", f"{fmt}: greedy tokens change with engine settings: {fr.settings['token_changes']}"))
    elif fr.settings.get("groups", 1) > 1:
        fr.issues.append(("TS1318", f"{fmt}: logits bits differ between settings ({fr.settings['groups']} distinct "
                                    f"results over {len(fr.runs)} settings, max relative difference "
                                    f"{fr.settings['max_rel_between']:.2e}); greedy tokens identical"))
    if fr.quantization is not None:
        q = fr.quantization
        # with a conversion fault the file's float64 math differs from the source for that reason too
        what = ("the file against the source in float64 (quantization plus the conversion fault above)"
                if fr.conversion else "quantization alone (float64, decoded weights against source)")
        q["includes_conversion_fault"] = bool(fr.conversion)
        fr.issues.append(("TS1314", f"{fmt}: {what}: rel err max {q['rel_max']:.2e}, top 1 agreement "
                                    f"{100 * q['top1_agreement']:.1f}%"))
        if q["flips_beyond_near_ties"] or q["greedy_changed"]:
            fr.issues.append(("TS1315", f"{fmt}: {what.split(' (')[0]} changes decisions: {q['flips_beyond_near_ties']} "
                                        f"teacher forced flips beyond near ties, greedy steps changed "
                                        f"{q['greedy_changed']}"))
    return fr


def _conversion(path: str, file_w: Weights, src_w: Weights) -> List[str]:
    from ..engines.engine_port import _gguf_weights, match_weights

    out = []
    a, b = file_w.cfg, src_w.cfg
    for name in ("vocab", "dim", "n_layers", "n_heads", "n_kv_heads", "ffn", "head_dim", "tied"):
        if getattr(a, name) != getattr(b, name):
            out.append(f"{name} is {getattr(a, name)} in the GGUF, {getattr(b, name)} in the source")
    for name, tol in (("eps", 1e-6), ("theta", 1e-6), ("rope_scale", 1e-6)):
        x, y = float(getattr(a, name)), float(getattr(b, name))
        if abs(x - y) > tol * max(abs(y), 1e-30):
            out.append(f"{name} is {x:g} in the GGUF, {y:g} in the source")
    if out:
        return out
    G = _gguf_weights(gguf.read(path))
    ref = {k: v.to(F64) for k, v in src_w.raw.items() if not k.endswith(".bias")}
    edges, unmatched, _, _ = match_weights(G, ref, src_w.cfg, file_w.arch)
    out += [e.sentence() for e in edges if e.verdict != "ok"]
    out += [f"source tensor {n} has no GGUF tensor" for n in unmatched]
    from ..engines.engine_port import TYPE_BOUND
    for role, bt in src_w.B.items():
        fb = file_w.B.get(role)
        if fb is None:
            out.append(f"bias {role} is missing from the GGUF")
        elif float((fb - bt).norm() / max(float(bt.norm()), 1e-300)) > TYPE_BOUND["F32"]:
            out.append(f"bias {role} differs from the source")
    return out


def _quantization(file_w: Weights, src_w: Weights, prompts, rp, src_logits) -> Dict[str, Any]:
    rels, flips, beyond, n = [], 0, 0, 0
    for p in prompts:
        ref = rp(p, "f32")
        s = src_logits(p)
        rels += _rel_rows(ref.l64, s).tolist()
        top = s.topk(2, dim=-1).values
        tie = (top[:, 0] - top[:, 1]) < QUANT_TIE * s.std(dim=-1)  # as `truestar port` counts near ties
        f = ref.l64.argmax(-1) != s.argmax(-1)
        n += len(p)
        flips += int(f.sum())
        beyond += int((f & ~tie).sum())
    greedy_changed = 0
    for p in prompts:  # the source replay's own greedy continuation against the file's replay of it
        seq = list(p)
        for _ in range(8):
            seq.append(int(src_logits(seq)[-1].argmax()))
        lf = rp(seq[:-1], "f32").l64[len(p) - 1:].argmax(-1).tolist()
        greedy_changed += int(any(a != b for a, b in zip(lf, seq[len(p):])))
    return {"rel_median": float(np.median(rels)), "rel_max": float(np.max(rels)), "positions": n,
            "top1_agreement": (n - flips) / max(n, 1), "flips": flips, "flips_beyond_near_ties": beyond,
            "greedy_changed": greedy_changed}


def _settings_summary(runs: List[RunResult], toks: List[List[List[int]]], prompts, rp,
                      logits: List[List[np.ndarray]]) -> Dict[str, Any]:
    if not runs:
        return {}
    groups: Dict[str, List[str]] = {}
    for r in runs:
        groups.setdefault(r.digest, []).append(r.setting.label)
    changes = []
    base = toks[0]
    for r, t in zip(runs[1:], toks[1:]):
        for i, (a, b) in enumerate(zip(base, t)):
            j = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if j is not None:
                ref = rp(list(prompts[i]) + a[:-1], r.setting.kv)
                row = len(prompts[i]) - 1 + j
                near = bool(ref.gap[row] <= 2 * ref.bound[row])
                changes.append(f"{r.setting.label} vs {runs[0].setting.label}: prompt {i} step {j} "
                               f"({'near tie' if near else 'not a near tie'})")
    max_between = 0.0  # teacher forced logits of every setting against the first setting's
    for other in logits[1:]:
        for a, b in zip(logits[0], other):
            max_between = max(max_between, float(_rel_rows(torch.from_numpy(b), torch.from_numpy(a)).max()))
    return {"runs": len(runs), "groups": len(groups), "identical_groups": list(groups.values()),
            "token_changes": "; ".join(changes[:6]) if changes else "", "n_token_changes": len(changes),
            "max_rel_between": max_between}


def conform_llamacpp(target: str, *, build: Optional[Build] = None, llama_cpp: Optional[str] = None,
                     formats: Optional[Sequence[str]] = None, settings: Optional[Sequence[Setting]] = None,
                     prompts: Optional[List[List[int]]] = None, n_new: int = 16, source: Optional[str] = None,
                     workdir: Optional[str] = None, runner: Optional[Runner] = None,
                     python: Optional[str] = None) -> LlamaCppReport:
    """Run llama.cpp on `target` (a GGUF, or a Hugging Face directory converted by llama.cpp's own
    tools to every format in `formats`) and judge it. `runner` replaces the driver (tests)."""
    t0 = time.perf_counter()
    is_dir = os.path.isdir(target)
    if not is_dir and not os.path.isfile(target):
        raise LlamaCppError("TS1319", f"{target} not found")
    src_dir = target if is_dir else source
    src_w = load_hf_dir(src_dir) if src_dir else None
    if runner is None or is_dir:
        build = build or find_build(llama_cpp)
    bver = dict(build.version) if build else {}
    if runner is None:
        drv = build_driver(build)

        def runner(path, s, ps, n):  # noqa: E306
            return run_driver(drv, path, s, ps, n)
    if is_dir:
        wd = workdir or tempfile.mkdtemp(prefix="truestar-llamacpp-")
        files = convert(build, target, wd, formats or FORMATS, python=python)
    else:
        files = {"gguf": target}
    settings = list(settings or default_settings())
    rep = None
    for fmt, path in files.items():
        fw = load_gguf(path)
        if prompts is None:
            bos = gguf.read(path).metadata.get("tokenizer.ggml.bos_token_id")
            prompts = default_prompts(fw.cfg.vocab, bos=bos)
        if rep is None:
            rep = LlamaCppReport(target, src_dir, bver, prompts, n_new)
        fr = check_format(fmt, path, fw, src_w, settings, prompts, n_new, runner)
        rep.formats.append(fr)
        for r in fr.runs:  # the engine's own version line, once
            for k in ("ggml_version", "ggml_commit", "system"):
                if k in r.info and k not in rep.build:
                    rep.build[k] = r.info[k]
    rep.seconds = time.perf_counter() - t0
    return rep
