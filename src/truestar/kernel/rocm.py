"""AMD GPUs (ROCm, HIP) for the kernel layer: targets, detection and ahead of time builds.

The GPU backend emits Triton, and Triton's AMD backend lowers the same source to AMDGPU code. Two
ways in:

* On a machine with a ROCm build of torch (``torch.version.hip`` set) the device is still torch
  ``"cuda"``. ``K.compile(kernel, "rocm")`` (or ``"cuda"`` there) compiles and launches through
  Triton's HIP driver; ``current_target()`` reads the arch, wave size, CU count and LDS of the
  attached GPU and the launch defaults follow them, as they follow the SM count on NVIDIA.
* Without an AMD GPU, ``build(kernel, "gfx942", shapes)`` compiles every Triton function of one
  launch to an AMDGPU code object for a named target (``TARGETS``), with that target's wave size,
  CU count and LDS, and checks its resources (LDS, scratch) without running it. Nothing is run, so
  nothing is verified against float64 either: every build carries TS885 (compiled, not run).

Codes: TS880 unknown target, TS881 compile failed, TS882 LDS over the target's limit, TS883 no ROCm
here, TS884 TF32 asked on a target without it (float32 stays strict), TS885 compiled and not run.
Spilled registers over the target's scratch limit raise TS700, as on NVIDIA. See docs/ROCM.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import ir

GiB = 1024 ** 3

# Largest scratch (private memory) one wave can own: the WAVESIZE field of the scratch setup, in
# LLVM's GCNSubtarget::getMaxWaveScratchSize. Before GFX11 a 13 bit field in units of 256 dwords,
# on GFX11 a 15 bit field in units of 64 dwords (llvm PR #77625).
MAX_WAVE_SCRATCH_GFX9 = 256 * 4 * ((1 << 13) - 1)    # 8387584 bytes
MAX_WAVE_SCRATCH_GFX11 = 64 * 4 * ((1 << 15) - 1)    # 8388352 bytes
LDS_PER_WORKGROUP = 64 * 1024   # bytes a workgroup can address on gfx9 and gfx11 (gfx950: 160 KiB)
LANES_PER_CU = 2048   # resident lanes per CU assumed for the scratch reservation when no GPU is attached


@dataclass(frozen=True)
class AmdTarget:
    """One AMD GPU target. ``cus`` and ``memory`` describe one product with that arch (the table's
    values; on hardware the attached GPU's own)."""

    arch: str            # LLVM target, "gfx942"
    product: str         # what the CU count and memory belong to
    wave: int            # lanes per wavefront Triton compiles for (64 on CDNA, 32 on RDNA)
    cus: int             # compute units, the analogue of NVIDIA SMs for launch defaults
    lds: int             # bytes of LDS (shared memory) one workgroup may use
    memory: int          # bytes of device memory
    matrix: str          # "mfma" (CDNA), "wmma" (RDNA3) or "none"
    xf32: bool           # float32 dots may run as TF32 (xf32 MFMA); Triton allows it on gfx942 only
    lanes_per_cu: int = LANES_PER_CU
    notes: Tuple[str, ...] = ()

    @property
    def max_wave_scratch(self) -> int:
        return MAX_WAVE_SCRATCH_GFX11 if _major(self.arch) >= 11 else MAX_WAVE_SCRATCH_GFX9

    @property
    def scratch_per_lane(self) -> int:
        """Largest scratch (spilled registers, private arrays) one lane may use."""
        return self.max_wave_scratch // self.wave

    def as_dict(self) -> Dict:
        return dict(arch=self.arch, product=self.product, wave=self.wave, cus=self.cus,
                    lds=self.lds, memory=self.memory, matrix=self.matrix, xf32=self.xf32,
                    scratch_per_lane=self.scratch_per_lane, notes=list(self.notes))


# CU counts, wave sizes and memory from the ROCm GPU hardware specifications table. RDNA3 has 128
# KiB of LDS per WGP, but one workgroup addresses 64 KiB, which is what a Triton program gets.
TARGETS: Dict[str, AmdTarget] = {
    "gfx942": AmdTarget("gfx942", "AMD Instinct MI300X (CDNA3)", 64, 304, LDS_PER_WORKGROUP, 192 * GiB,
                        "mfma", True),
    "gfx1100": AmdTarget("gfx1100", "AMD Radeon RX 7900 XTX (RDNA3)", 32, 96, LDS_PER_WORKGROUP, 24 * GiB,
                         "wmma", False),
    "gfx1103": AmdTarget("gfx1103", "AMD Radeon 780M iGPU (RDNA3)", 32, 12, LDS_PER_WORKGROUP, 8 * GiB,
                         "wmma", False,
                         notes=("memory is shared with the CPU; 8 GiB is an assumed carve out",)),
}

DOT_SHAPES = ("contraction", "attention", "conv")


class RocmError(RuntimeError):
    code = "TS881"


class UnknownTarget(RocmError, ValueError):
    """TS880: an AMD target name this table does not know."""
    code = "TS880"


class RocmCompileError(RocmError):
    """TS881: Triton's AMD backend refused a kernel for a target. The message names the kernel, its
    loop shape, the launch config and the target."""
    code = "TS881"


class RocmUnavailable(RocmError):
    """TS883: ROCm was asked for on a machine whose torch has no HIP build or no AMD GPU."""
    code = "TS883"


def _major(arch: str) -> int:
    m = re.match(r"gfx(\d+?)(\w\w)$", arch)
    return int(m.group(1)) if m else 0


def target(arch) -> AmdTarget:
    """The table's target for ``arch`` ("gfx942", "gfx1100", "gfx1103"), or an AmdTarget as is."""
    if isinstance(arch, AmdTarget):
        return arch
    a = str(arch).split(":")[0].strip().lower()
    if a not in TARGETS:
        raise UnknownTarget(f"[TS880] unknown AMD target {arch!r}; known: {', '.join(TARGETS)}. "
                            f"On a machine with that GPU, current_target() reads it from the device")
    return TARGETS[a]


# ----------------------------------------------------------------------------- the attached GPU

def hip_version() -> Optional[str]:
    import torch
    return getattr(torch.version, "hip", None)


def hip_available() -> Tuple[bool, str]:
    """(True, "") when torch is a ROCm build with an AMD GPU it can use, else (False, why)."""
    import torch
    if not hip_version():
        return False, (f"torch {torch.__version__} is not a ROCm build (torch.version.hip is None); "
                       f"install a ROCm wheel of torch on a machine with an AMD GPU")
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return False, f"torch is a ROCm build (HIP {hip_version()}) but sees no AMD GPU"
    except Exception as e:  # pragma: no cover - driver specific
        return False, f"torch is a ROCm build but asking for a GPU failed: {e}"
    return True, ""


def require_hip(what: str) -> None:
    ok, why = hip_available()
    if not ok:
        raise RocmUnavailable(f"[TS883] {what} needs ROCm: {why}. Nothing falls back to another "
                              f"device; ahead of time builds without a GPU: truestar.kernel.rocm.build")


_CURRENT: Dict[int, AmdTarget] = {}


def current_target(device=None) -> AmdTarget:
    """The attached AMD GPU as an AmdTarget, read once per device from torch's properties: arch
    (``gcnArchName``), wave size, CUs, LDS per workgroup, memory and resident lanes per CU. An arch
    the table does not know is still described from the device."""
    import torch
    require_hip("current_target")
    dev = torch.cuda.current_device() if device is None else torch.device(device).index or 0
    t = _CURRENT.get(dev)
    if t is not None:
        return t
    pr = torch.cuda.get_device_properties(dev)
    arch = str(getattr(pr, "gcnArchName", "") or "").split(":")[0] or "gfx_unknown"
    known = TARGETS.get(arch)
    wave = int(getattr(pr, "warp_size", 0) or (known.wave if known else (32 if _major(arch) >= 10 else 64)))
    lds = int(getattr(pr, "shared_memory_per_block", 0) or LDS_PER_WORKGROUP)
    matrix = known.matrix if known else ("mfma" if _major(arch) == 9 else "wmma" if _major(arch) >= 11 else "none")
    t = AmdTarget(arch, pr.name, wave, int(pr.multi_processor_count), lds, int(pr.total_memory), matrix,
                  known.xf32 if known else False,
                  int(getattr(pr, "max_threads_per_multi_processor", 0) or LANES_PER_CU))
    _CURRENT[dev] = t
    return t


def launch_options(t: AmdTarget, plan_kind: str) -> Dict:
    """Compile options for Triton's AMD backend for one launch. Dot kernels on MFMA targets use
    the 16 x 16 instruction (ROCm's Triton tuning guide: mfma_16x16 usually beats 32 x 32 on
    MI300X GEMMs); WMMA on RDNA3 has only 16 x 16, which Triton picks itself."""
    if t.matrix == "mfma" and plan_kind in DOT_SHAPES:
        return {"matrix_instr_nonkdim": 16}
    return {}


def kernel_opts(t: AmdTarget, opts: Dict) -> Tuple[Dict, List[str]]:
    """The kernel compiler's options for target t, and TS884 notes. TF32 is a permission, not a
    requirement: on a target without xf32 float32 dots stay strict IEEE (more exact, never less)."""
    o = dict(opts)
    notes = []
    if o.get("allow_tf32") and not t.xf32:
        o.pop("allow_tf32")
        notes.append(f"[TS884] TF32 was allowed but {t.arch} has no TF32 (xf32) dot in Triton; "
                     f"float32 dots run as strict IEEE float32")
    o["amd_arch"] = t.arch
    return o, notes


# ----------------------------------------------------------------------------- ahead of time builds

def compile_available() -> Tuple[bool, str]:
    """(True, "") when this Python's Triton has its AMD backend, so ``build`` can run without a
    GPU, else (False, why)."""
    try:
        import triton  # noqa: F401
        from triton.backends import backends
    except Exception as e:
        return False, f"Triton is not importable here ({type(e).__name__}: {e})"
    if "amd" not in backends:
        return False, f"Triton {triton.__version__} here has no AMD backend (backends: {', '.join(backends)})"
    try:
        from triton.backends.compiler import GPUTarget  # noqa: F401
        from triton.compiler import ASTSource  # noqa: F401
    except Exception as e:
        return False, f"Triton {triton.__version__} lacks the ahead of time compile API ({e})"
    return True, ""


_PTR = {"float16": "*fp16", "bfloat16": "*bf16", "float32": "*fp32", "float64": "*fp64"}
_OPTION_KEYS = ("num_warps", "num_stages")


@dataclass
class FunctionBuild:
    """One Triton function compiled to an AMDGPU code object, with what its metadata says."""
    name: str
    num_warps: int
    lanes: int            # lanes per program (num_warps times the wave size)
    lds: int              # bytes of LDS the program allocates
    vgprs: int
    sgprs: int
    vgpr_spills: int
    sgpr_spills: int
    scratch: int          # bytes of scratch per lane
    hsaco_bytes: int
    options: Dict = field(default_factory=dict)


@dataclass
class AmdBuild:
    """A launch compiled for an AMD target without running it (``build``)."""
    kernel: str
    target: AmdTarget
    plan_kind: str
    config_kind: str
    config: Dict
    functions: List[FunctionBuild]
    tf32: bool
    notes: List[str]
    source: str

    def as_dict(self) -> Dict:
        return dict(kernel=self.kernel, target=self.target.arch, plan_kind=self.plan_kind,
                    config_kind=self.config_kind, config=self.config, tf32=self.tf32,
                    notes=list(self.notes),
                    functions=[dict(vars(f)) for f in self.functions])


def _int_field(text: str, key: str) -> int:
    m = re.findall(rf"\.{key}:\s+(\d+)", text)
    return max((int(x) for x in m), default=0)


def _resources(t: AmdTarget, fb: FunctionBuild, cfg: Dict) -> Optional[Tuple[str, str]]:
    """(code, why) when the compiled function cannot run on target t, else None."""
    if fb.lds > t.lds:
        return "TS882", (f"function {fb.name} with launch config {cfg} needs {fb.lds} bytes of LDS, "
                         f"above the {t.lds} one workgroup has on {t.arch}")
    if fb.scratch > t.scratch_per_lane:
        return "TS700", (f"function {fb.name} with launch config {cfg} needs {fb.scratch} bytes of scratch per "
                         f"lane (registers spilled), above the {t.scratch_per_lane} {t.arch} allows")
    total = fb.scratch * t.lanes_per_cu * t.cus
    if fb.scratch and total > t.memory // 4:
        return "TS700", (f"function {fb.name} with launch config {cfg} needs {fb.scratch} bytes of scratch per "
                         f"lane, {total / 2 ** 20:.0f} MiB for every resident lane of {t.cus} CUs, above 1/4 "
                         f"of {t.memory / GiB:.0f} GiB")
    return None


class LdsError(ValueError):
    """TS882: a compiled launch config needs more LDS than one workgroup has on the target."""
    code = "TS882"


def _compile_fn(fn, t: AmdTarget, ck: ir.Kernel, meta: Dict, amd: Dict):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    outs = {o.name: o.dtype for o in ck.outputs}
    ins = {b.name: b.dtype for b in ck.inputs}
    sig, consts, attrs = {}, {}, {}
    for i, prm in enumerate(fn.params):
        n = prm.name
        if prm.is_constexpr:
            sig[n] = "constexpr"
            consts[n] = meta[n]
            continue
        if n.startswith("p_"):
            base = n[2:]
            if base == "ws":
                sig[n] = "*fp32"
            elif base == "chk":
                sig[n] = "*i32"
            else:
                sig[n] = _PTR[ins.get(base) or outs[base]]
            attrs[(i,)] = [["tt.divisibility", 16]]    # torch allocations are 16 byte aligned
        else:
            sig[n] = "i32"   # strides, sizes and row counts; offsets are 32 bit (KERNEL_IR.md, limits)
    options = {k: meta[k] for k in _OPTION_KEYS if k in meta}
    options.update(amd)
    src = ASTSource(fn, sig, constexprs=consts, attrs=attrs)
    return triton.compile(src, target=GPUTarget("hip", t.arch, t.wave), options=options), options


def build(k: ir.Kernel, arch, shapes: Dict[str, Sequence[int]], *, allow_tf32: bool = False,
          deterministic: bool = False, checks=None, config: Optional[Dict] = None) -> AmdBuild:
    """Compile kernel k for AMD target ``arch`` at input ``shapes`` ({input name: shape}) without
    a GPU: plan the launch with the target's CUs and LDS (``config`` forces one), compile every
    Triton function of it (the split K second pass too) to an AMDGPU code object, and check its
    LDS and scratch. Raises UnknownTarget (TS880), triton_backend.Unschedulable, RocmCompileError
    (TS881), LdsError (TS882) or triton_backend.ConfigResourceError (TS700)."""
    from . import checks as checks_mod
    from . import compiler, reference
    from . import triton_backend as tb

    t = target(arch)
    ok, why = compile_available()
    if not ok:
        raise RocmCompileError(f"[TS881] cannot compile for {t.arch}: {why}")
    opts = {}
    if allow_tf32:
        opts["allow_tf32"] = True
    if deterministic:
        opts["deterministic"] = True
    chk = checks_mod.text(checks_mod.parse(checks))
    if chk:
        opts["checks"] = chk
    opts, notes = kernel_opts(t, opts)
    ck, in_map, _ = ir.canonicalize(k)
    key = compiler.kernel_key(k, "triton", opts)
    with compiler._lock:
        art = compiler._memory.get(key)
        if art is None:
            art = compiler._build(ck, "triton", opts, key)
            compiler._memory[key] = art
    p = art.plan
    env_user = reference.bind_sizes(k, {n: tuple(s) for n, s in shapes.items()})
    env = {}
    for b in k.inputs:
        cb = next(x for x in ck.inputs if x.name == in_map[b.name])
        for su, sc in zip(b.shape, cb.shape):
            if isinstance(sc, ir.Sym):
                env[sc.name] = env_user[su.name]
    with tb.target_limits(t.cus, t.lds, t.memory):
        cfg = tb.default_config(p, env) if config is None else dict(config)
        kind = tb.config_kind(p, env) if p.kind not in ("conv", "window") else p.kind
    try:
        fbs = _build_launch(k, t, ck, art, p, env, cfg)
        relieve = config is None and any(f.scratch or f.vgpr_spills for f in fbs)
    except tb.ConfigResourceError:
        if config is not None:
            raise
        fbs, relieve = None, True
    if relieve:
        # the default spills (typical on wave32 RDNA3, where a program has fewer lanes for the
        # same tile than on wave64 CDNA): walk the same ladder the hardware path walks
        start = cfg
        best = None if fbs is None else (max(f.scratch for f in fbs), cfg, fbs)
        for c in relief_ladder(p, env, cfg, t):
            try:
                got = _build_launch(k, t, ck, art, p, env, c)
            except (tb.ConfigResourceError, LdsError, RocmCompileError):
                continue
            worst = max(f.scratch for f in got)
            if best is None or worst < best[0]:
                best = (worst, c, got)
            if worst == 0 and not any(f.vgpr_spills for f in got):
                break
        if best is None:
            raise tb.ConfigResourceError(f"[TS700] kernel {k.name} on {t.arch}: the default config {start} and "
                                         f"every config of its relief ladder need more scratch than {t.arch} allows")
        _, cfg, fbs = best
        if cfg != start:
            notes.append(f"[TS886] the default config {start} spills registers on {t.arch}; "
                         f"{cfg} from the relief ladder "
                         + ("does not" if best[0] == 0 else f"still needs {best[0]} bytes of scratch per lane"))
    notes.append(f"[TS885] compiled for {t.arch} and not run: no float64 verification until it runs on the GPU")
    return AmdBuild(k.name, t, p.kind, kind, cfg, fbs, bool(getattr(p, "tf32", False)), notes, art.source)


def _build_launch(k, t: AmdTarget, ck, art, p, env, cfg) -> List[FunctionBuild]:
    """Compile every Triton function of one launch with cfg for target t and check its resources."""
    from . import triton_backend as tb
    with tb.target_limits(t.cus, t.lds, t.memory):
        fn, _grid, _extra, meta = tb.prepare(art.module, ck, p, env, config=cfg)
        post = tb.split_k_epilogue(art.module, p, env, cfg)
    amd = launch_options(t, p.kind)
    launches = [(fn, meta)] + ([(post[0], post[2])] if post is not None else [])
    fbs = []
    for f, m in launches:
        try:
            ck_bin, used = _compile_fn(f, t, ck, m, amd)
        except Exception as e:
            raise RocmCompileError(f"[TS881] kernel {k.name} ({p.kind}, function {f.__name__}) did not compile for "
                                   f"{t.arch} with launch config {cfg}: {type(e).__name__}: {e}") from e
        asm = ck_bin.asm.get("amdgcn", "")
        md = ck_bin.metadata
        nw = int(getattr(md, "num_warps", used.get("num_warps", 4)))
        fb = FunctionBuild(
            name=f.__name__, num_warps=nw, lanes=nw * t.wave,
            lds=max(int(getattr(md, "shared", 0) or 0), _int_field(asm, "group_segment_fixed_size")),
            vgprs=_int_field(asm, "vgpr_count"), sgprs=_int_field(asm, "sgpr_count"),
            vgpr_spills=_int_field(asm, "vgpr_spill_count"), sgpr_spills=_int_field(asm, "sgpr_spill_count"),
            scratch=_int_field(asm, "private_segment_fixed_size"),
            hsaco_bytes=len(ck_bin.asm.get("hsaco", b"")), options=used)
        bad = _resources(t, fb, cfg)
        if bad is not None:
            code, msg = bad
            if code == "TS882":
                raise LdsError(f"[TS882] kernel {k.name} on {t.arch}: {msg}")
            raise tb.ConfigResourceError(f"[TS700] kernel {k.name} on {t.arch}: {msg}")
        fbs.append(fb)
    return fbs


def relief_ladder(p, env: Dict[str, int], cfg: Dict, t: AmdTarget) -> List[Dict]:
    """Configs to try, in order, when cfg spills registers on target t: each step keeps the last
    one's changes and adds one (more waves per program up to 1024 lanes, fewer pipeline stages, a
    shorter BK, narrower then shorter tiles). SPLIT_K, and so the summation order of a
    deterministic kernel, never changes. Tile and window kernels have no ladder."""
    from . import triton_backend as tb
    if p.kind in ("contraction", "conv"):
        steps = [("warps",), ("stages", 2), ("BK",), ("stages", 1), ("BK",), ("BN",), ("BM",)]
    elif p.kind == "attention":
        steps = [("warps",), ("stages", 1), ("BN",), ("BM",)]
    else:
        return []
    out, c = [], dict(cfg)
    for st in steps:
        if st[0] == "warps":
            if c["num_warps"] * 2 * t.wave <= 1024 and c["num_warps"] * 2 <= 16:
                c["num_warps"] *= 2
        elif st[0] == "stages":
            c["num_stages"] = min(c["num_stages"], st[1])
        else:
            c[st[0]] = max(16, c[st[0]] // 2)
        if c != cfg and c not in out and tb.check_config(p, env, c) is None:
            out.append(dict(c))
    return out


class RocmNote(UserWarning):
    """An informational note of the AMD path (TS884: TF32 not available, float32 stays strict)."""


def build_candidates(k: ir.Kernel, arch, shapes: Dict[str, Sequence[int]], **kw) -> List[Tuple[Dict, object]]:
    """Every tuning candidate of the launch (default first), each built for ``arch``: a list of
    (config, AmdBuild or the exception that refused it). The tuner on hardware skips the refused
    ones; here they are reported."""
    from . import reference
    from . import triton_backend as tb

    t = target(arch)
    o, _ = kernel_opts(t, {"allow_tf32": True} if kw.get("allow_tf32") else {})
    if kw.get("deterministic"):
        o["deterministic"] = True
    ck, in_map, _ = ir.canonicalize(k)
    p = tb.generate(ck, o)[1]
    env_user = reference.bind_sizes(k, {n: tuple(s) for n, s in shapes.items()})
    env = {}
    for b in k.inputs:
        cb = next(x for x in ck.inputs if x.name == in_map[b.name])
        for su, sc in zip(b.shape, cb.shape):
            if isinstance(sc, ir.Sym):
                env[sc.name] = env_user[su.name]
    with tb.target_limits(t.cus, t.lds, t.memory):
        cands = tb.candidates(p, env)
    out = []
    for c in cands:
        try:
            out.append((c, build(k, t, shapes, config=c, **kw)))
        except (RocmCompileError, LdsError, tb.ConfigResourceError, ValueError) as e:
            out.append((c, e))
    return out
