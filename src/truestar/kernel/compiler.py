"""Compile a kernel for a backend, with a cache keyed on the kernel's canonical IR and verification.

Cache layers, fastest first:

1. In process: canonical IR hash -> loaded module. A second compile of the same kernel, or of a
   kernel that differs only in names, costs one hash.
2. On disk: the generated module is written to `<cache>/<backend>/<hash>.py`. Triton and numba
   key their own binary caches on that source, so a new process that compiles the same kernel
   skips code generation and the backend compiler.

The cache key covers the canonical IR, the backend, compile options and CODEGEN_VERSION, so an
edit to one layer of a model recompiles only the kernels whose IR changed.

GPU launch configs (block sizes, warps, stages) are autotuned per shape bucket and stored next to
the generated modules; see tune.py.
"""

from __future__ import annotations

import contextvars
import hashlib
import importlib.util
import os
import re
import sys
import threading
import warnings
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import checks as checks_mod
from . import ir, reference, tune, verify

CODEGEN_VERSION = "7"  # 3: contraction tiles and split K (main); 4: strided loads, conv and window shapes; 5: sentinels in the conv and window shapes; 6: batch axes folded into contraction rows; 7: deterministic K chunks (det_k_chunk)

_BACKENDS = {"cuda": "triton", "gpu": "triton", "triton": "triton", "rocm": "triton", "hip": "triton",
             "cpu": "numba", "numba": "numba", "metal": "metal"}
_DEVICE_OF = {"triton": "cuda", "numba": "cpu", "metal": "metal"}
_NUMBA_PURPOSE = "The kernel layer's 'cpu' backend"


def cache_dir() -> Path:
    d = os.environ.get("TRUESTAR_CACHE_DIR")
    return Path(d) if d else Path.home() / ".cache" / "truestar" / "kernels"


@dataclass
class CacheStats:
    memory_hits: int = 0
    disk_hits: int = 0
    misses: int = 0

    def reset(self):
        self.memory_hits = self.disk_hits = self.misses = 0


stats = CacheStats()
_memory: Dict[str, "_Artifact"] = {}
_verified: set = set()  # (artifact key, input signature) already checked against the reference
# A verification group (compiler/dynamic.py, shape buckets): while one is set, a kernel is checked
# once per (group, binary) instead of once per shape signature. The binary is everything Triton
# compiles a variant for (function, constexpr meta, launch config, the 1 and divisible by 16
# specialization of strides and the 16 byte alignment of pointers), so a shape that runs another
# binary in the same group is still checked on its own. None (the default) keeps per shape checks.
verify_group: contextvars.ContextVar = contextvars.ContextVar("truestar_verify_group", default=None)
group_skips = [0]   # checks skipped because the group had verified the same binary (for reports)


def _spec_bits(v) -> tuple:
    """What Triton specializes an argument on by default: pointers on 16 byte alignment, ints on
    being 1 and on divisibility by 16."""
    if hasattr(v, "data_ptr"):
        return ("p", v.data_ptr() % 16 == 0)
    if isinstance(v, (bool, float)) or v is None:
        return ("c",)
    return ("i", v == 1, v % 16 == 0)


def _binary_id(L, got, backend: str) -> tuple:
    """Identity of the compiled binary one launch runs (see verify_group)."""
    if backend != "triton":         # numba and Metal compile once per dtype signature, never per size
        return (backend,)
    params = getattr(L.fn, "params", None)
    if params is None:          # not a JIT function (a library call): nothing to specialize
        return ("lib", L.lib is not None)
    plain = [q.name for q in params if not q.is_constexpr]
    end = len(plain) - (1 if L.has_ws else 0)
    int_names = plain[end - len(L.int_args):end]
    skip = set(getattr(L.fn, "do_not_specialize", ()) or ())
    ints = tuple(("u",) if n in skip else _spec_bits(v) for n, v in zip(int_names, L.int_args))
    ptrs = tuple(_spec_bits(got[n]) for n in L.in_order)
    post = None if L.post is None else (getattr(L.post[0], "__name__", ""), tuple(sorted(L.post[2].items())))
    return (getattr(L.fn, "__name__", ""), tuple(sorted(L.meta.items())), ptrs, ints, post,
            L.lib is not None)


def _vsig(L, got, backend: str):
    g = verify_group.get()
    return L.shape_sig if g is None else ("group", g, _binary_id(L, got, backend))


_lock = threading.Lock()


@dataclass
class _Artifact:
    key: str
    backend: str
    module: object
    plan: object
    path: Path
    source: str


def _vkey(key: str, L, sig, got, backend: str) -> tuple:
    """What one first call check covers: the kernel, the shapes (or, inside a verification group,
    the group and the binary) and dtypes and the launch config (a different config is a different
    binary)."""
    cfg = tuple(sorted(L.config.items())) if L.config else ()
    return (key, _vsig(L, got, backend), tuple(str(x[2]) for x in sig), cfg)


def _vkey_text(vkey: tuple) -> str:
    return repr(vkey)


_tls = threading.local()


class collect_checks:
    """Within the block, record every kernel launch's check key (seen, as text). With
    defer=True the first call checks are not run but listed in pending: the caller
    (compile_cache, through CompiledProgram) either accepts them with accept() or leaves them
    for the next verified call."""

    def __init__(self, defer: bool = False):
        self.defer = defer
        self.seen: set = set()
        self.pending: list = []

    def __enter__(self):
        self._prev = getattr(_tls, "collect", None)
        _tls.collect = self
        return self

    def __exit__(self, *exc):
        _tls.collect = self._prev

    def accept(self) -> None:
        for vkey, L in self.pending:
            _verified.add(vkey)
            L.verified = True
        self.pending = []


def clear_memory_cache():
    """Forget loaded kernels in this process (the disk cache stays)."""
    _memory.clear()
    _verified.clear()


def _opts_text(opts: Dict) -> str:
    parts = []
    for k in sorted(opts):
        v = opts[k]
        if k == "_break":  # test hook: deliberately broken codegen rules
            v = {op: f("x") if op in ir.UNARY_OPS else f("x", "y") for op, f in sorted(v.items())}
        parts.append(f"{k}={v!r}")
    return ";".join(parts)


def kernel_key(k: ir.Kernel, backend: str, opts: Optional[Dict] = None) -> str:
    ck, _, _ = ir.canonicalize(k)
    text = f"v{CODEGEN_VERSION}\n{_BACKENDS[backend]}\n{_opts_text(opts or {})}\n{ck.to_text()}"
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build(ck: ir.Kernel, backend: str, opts: Dict, key: str) -> _Artifact:
    if backend == "triton":
        from . import triton_backend as tb
        src, plan = tb.generate(ck, opts)
    elif backend == "metal":
        from . import metal_backend as mb
        src, plan = mb.generate(ck, opts)
    else:
        from . import cpu_backend as cb
        src, plan = cb.generate(ck, opts), None
    d = cache_dir() / backend
    d.mkdir(parents=True, exist_ok=True)
    path = d / (f"k_{key}.metal" if backend == "metal" else f"k_{key}.py")
    if backend == "triton":
        from .. import compile_cache
        compile_cache.install_triton_dir()   # Triton's binaries under the Truestar cache
    if path.exists() and path.read_text(encoding="utf-8") == src:
        stats.disk_hits += 1
        try:
            os.utime(path)                   # least recently used order (compile_cache.trim)
        except OSError:
            pass
    else:
        stats.misses += 1
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(src, encoding="utf-8")
        os.replace(tmp, path)
    if backend == "metal":   # the library compiled by torch.mps.compile_shader, and its kernel
        from . import metal_backend as mb
        return _Artifact(key, backend, mb.load(src, ck.name, str(path)), plan, path, src)
    try:
        mod = _load_module(path, f"truestar_kernel_{backend}_{key}")
    except ModuleNotFoundError as e:
        if backend == "numba" and e.name == "numba":  # a generated file imports numba first
            from ..deps import MissingDependencyError
            raise MissingDependencyError("numba", _NUMBA_PURPOSE) from None
        raise
    return _Artifact(key, backend, mod, plan, path, src)


class CompiledKernel:
    """A kernel compiled for one backend. Call it with the input tensors (by name or position)."""

    def __init__(self, k: ir.Kernel, backend: str, art: _Artifact, ck: ir.Kernel,
                 in_map: Dict[str, str], out_map: Dict[str, str], opts: Dict,
                 tune_mode: Optional[bool] = None, force_config: Optional[dict] = None,
                 library: bool = False):
        self.kernel = k
        self.name = k.name
        self.backend = backend
        self.art = art
        self.ck = ck
        self.in_map = in_map
        self.out_map = out_map
        self.opts = opts
        self.verify_seconds = 0.0
        self.tune_seconds = 0.0
        self._tuned: Dict[str, dict] = {}   # tune key -> config chosen in this process
        self.tune_mode = tune_mode
        self.force_config = force_config
        self.library = bool(library)   # may run as a cuBLAS call when measured faster (library.py)
        self._fast: Dict[tuple, _Launch] = {}

    @property
    def key(self) -> str:
        return self.art.key

    @property
    def source(self) -> str:
        return self.art.source

    @property
    def plan_kind(self) -> str:
        return "loops" if self.art.plan is None else self.art.plan.kind

    @property
    def tune_base_key(self) -> str:
        """Key of the kernel's structure for the tuning cache: the cache key with every float
        constant blanked, so editing a scale or an epsilon reuses the tuned launch configs (the
        new binary is still verified on its first call)."""
        text = _FLOAT_RE.sub("C", self.ck.to_text())
        # compiled in checks add flag stores, not another loop shape: they share tuned configs
        opts = {k: v for k, v in self.opts.items() if k not in ("checks", "check_loads")}
        text = f"v{CODEGEN_VERSION}\n{self.art.backend}\n{_opts_text(opts)}\n{text}"
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def launch_configs(self) -> list:
        """[(shape signature, launch config, source)] for every input signature seen so far;
        source is 'default', 'tuned' (searched in this process), 'order tuned' (deterministic and
        its bits follow its config: searched or read over the parameters that keep the summation
        order only, triton_backend.ORDER_FREE), 'cache' (read from disk),
        'shape cache' (tuned for another kernel with the same operands and sizes), 'reused' (a
        config this kernel already compiled for another size, see _reuse), 'pinned' (the default,
        because the kernel is compiled deterministic and its bits depend on the config, see
        triton_backend.bits_follow_config) or 'forced' (the _config test hook)."""
        return [(L.shape_sig, L.config, L.config_source) for L in self._fast.values()
                if L.config is not None]

    def _gather(self, args, kwargs) -> Dict[str, object]:
        names = [b.name for b in self.kernel.inputs]
        if len(args) > len(names):
            raise TypeError(f"kernel {self.name} takes {len(names)} inputs ({', '.join(names)}), "
                            f"got {len(args)}")
        got = dict(zip(names, args))
        for kname, v in kwargs.items():
            if kname not in names:
                raise TypeError(f"kernel {self.name} has no input {kname!r}; inputs: "
                                f"{', '.join(names)}")
            got[kname] = v
        missing = [n for n in names if n not in got]
        if missing:
            raise TypeError(f"kernel {self.name} is missing inputs: {', '.join(missing)}")
        return got

    def _signature(self, got) -> tuple:
        sig = []
        for b in self.kernel.inputs:
            v = got[b.name]
            if hasattr(v, "stride") and callable(v.stride):
                sig.append((tuple(v.shape), v.stride(), v.dtype, v.device))
            else:
                sig.append((np.shape(v), getattr(v, "strides", None), getattr(v, "dtype", None)))
        return tuple(sig)

    def _prepare(self, got, sig) -> "_Launch":
        """First call with a new input signature: bind sizes, check inputs, plan the launch."""
        import torch
        shapes = {n: tuple(v.shape) for n, v in got.items()}
        env_user = reference.bind_sizes(self.kernel, shapes)
        env = {}  # canonical symbol name -> size
        for b in self.kernel.inputs:
            cb = next(x for x in self.ck.inputs if x.name == self.in_map[b.name])
            for su, sc in zip(b.shape, cb.shape):
                if isinstance(sc, ir.Sym):
                    env[sc.name] = env_user[su.name]
        out_specs = []
        for o, co in zip(self.kernel.outputs, self.ck.outputs):
            shape = tuple(a.size if isinstance(a.size, int) else env[a.size.name] for a in co.axes)
            out_specs.append((o.name, co.name, shape, o.dtype))
        canon_to_user = {c: u for u, c in self.in_map.items()}
        in_order = [canon_to_user[b.name] for b in sorted(self.ck.inputs, key=lambda b: b.name)]
        sym_vals = [env[s.name] for s in sorted(self.ck.symbols(), key=lambda s: s.name)]
        L = _Launch(env_user=env_user, env=env, out_specs=out_specs, in_order=in_order,
                    sym_vals=sym_vals, shape_sig=tuple((n, shapes[n]) for n in sorted(shapes)))
        if self.art.backend == "triton":
            from . import triton_backend as tb
            dev = None
            for b in self.kernel.inputs:
                t = got[b.name]
                if not isinstance(t, torch.Tensor) or t.device.type != "cuda":
                    raise TypeError(f"kernel {self.name} on cuda: input {b.name} must be a CUDA "
                                    f"tensor")
                if str(t.dtype).replace("torch.", "") != b.dtype:
                    raise TypeError(f"kernel {self.name}: input {b.name} is {t.dtype}, the kernel "
                                    f"was built for {b.dtype}")
                if dev is not None and t.device != dev:
                    raise TypeError(f"kernel {self.name}: inputs are on different devices")
                dev = t.device
            # a kernel with no tensor input (its value folds constants: sqrt(1.0)) runs on the
            # current CUDA device; before, its launch had no device and failed at the first call
            L.device = dev if dev is not None else torch.device("cuda", torch.cuda.current_device())
            strides = []
            for n in in_order:
                strides += list(got[n].stride())
            for _, _, shape, _ in out_specs:
                strides += _contiguous_strides(shape)
            L.torch_dtypes = [getattr(torch, dt) for *_, dt in out_specs]
            # default configs follow the SM count and memory of the device the inputs are on
            with torch.cuda.device(dev):
                cfg, L.config_source = self._choose_config(got, L, strides + sym_vals, sig)
                if cfg is not None and cfg.get("LIB"):
                    if self._lib_form is None:
                        raise ValueError(f"kernel {self.name}: launch config {{'LIB': 1}} needs a contraction "
                                         f"cuBLAS can compute (library.py)")
                    L.config, L.lib = dict(cfg), self._lib_form
                    L.lib_names = sorted(b.name for b in self.ck.inputs)
                    self._fast[sig] = L
                    return L
            amd = "amd_arch" in self.opts       # an AMD GPU (kernel/rocm.py)
            relieve = amd and L.config_source == "default"
            try:
                kernels = self._set_launch(got, L, env, cfg, strides + sym_vals)
            except tb.ConfigResourceError:
                if not relieve:
                    raise
                kernels = None
            if relieve and (kernels is None or _spills(kernels)):
                self._relieve(got, L, env, strides + sym_vals, None if kernels is None else _spills(kernels))
        elif self.art.backend == "metal":
            self._prepare_metal(got, L, shapes)
        self._fast[sig] = L
        return L

    @property
    def _lib_form(self):
        """library.LibForm of this kernel (cuBLAS can compute it), or None."""
        if not hasattr(self, "_lib_cache"):
            from . import library
            p = self.art.plan
            self._lib_cache = None
            if self.art.backend == "triton" and p is not None and p.kind == "contraction" \
                    and not self.checks:
                self._lib_cache = library.library_form(self.ck, p)
        return self._lib_cache

    def _prepare_metal(self, got, L: "_Launch", shapes):
        import torch
        from . import metal_backend as mb
        for b in self.kernel.inputs:
            t = got[b.name]
            if not isinstance(t, torch.Tensor) or t.device.type != "mps":
                raise TypeError(f"kernel {self.name} on metal: input {b.name} must be an MPS tensor")
            if str(t.dtype).replace("torch.", "") != b.dtype:
                raise TypeError(f"kernel {self.name}: input {b.name} is {t.dtype}, the kernel "
                                f"was built for {b.dtype}")
        L.device = torch.device("mps")
        L.torch_dtypes = [getattr(torch, dt) for *_, dt in L.out_specs]
        p = self.art.plan
        cin = {self.in_map[n]: s for n, s in shapes.items()}
        cout = {cname: shape for _, cname, shape, _ in L.out_specs}
        vals = mb.params(self.ck, L.env, cin, cout)
        threads, group, work = mb.grid(p, self.ck, L.env)
        L.config = p.config()
        L.config_source = "default"
        L.fn = self.art.module[1]
        L.grid = (threads, group)
        L.int_args = vals
        L.meta = {"work": work, "params": torch.tensor(vals, dtype=torch.int32, device="mps")}

    def _set_launch(self, got, L: "_Launch", env, cfg, ints) -> list:
        """Plan the launch for cfg (None: the default), compile it and check its resources.
        Returns the compiled Triton kernels."""
        import torch
        from . import triton_backend as tb
        with torch.cuda.device(L.device):
            fn, grid, extra, meta = tb.prepare(self.art.module, self.ck, self.art.plan, env, config=cfg)
            L.config = tb.default_config(self.art.plan, env) if cfg is None else cfg
            if "amd_arch" in self.opts:
                from . import rocm
                meta = dict(meta, **rocm.launch_options(rocm.current_target(L.device), self.art.plan.kind))
        L.fn, L.grid, L.meta = fn, grid, meta
        L.int_args = list(ints) + extra
        L.has_ws = self.art.plan.kind in ("contraction", "attention")
        L.post = tb.split_k_epilogue(self.art.module, self.art.plan, env, L.config)
        if L.post is not None:
            L.ws_numel = tb.launch_workspace_elems(self.art.plan, env, L.config)
            return self._build_split(got, L)
        return self._build_one(got, L)

    def _relieve(self, got, L: "_Launch", env, ints, spilled=None):
        """AMD only: the default config spilled registers (or was refused for it), so walk
        rocm.relief_ladder and keep the first config that compiles without spills, else the one
        that spills least (the default too, when it compiled: ``spilled`` is its spill). Every config
        is still verified at its first call."""
        from . import rocm
        from . import triton_backend as tb
        start = dict(L.config)
        best = None if spilled is None else (spilled, start)
        for c in rocm.relief_ladder(self.art.plan, env, start, rocm.current_target(L.device)):
            try:
                ks = self._set_launch(got, L, env, c, ints)
            except tb.ConfigResourceError:
                continue
            n = _spills(ks)
            if n == 0:
                L.config_source = "default, spill relief"
                return
            if best is None or n < best[0]:
                best = (n, c)
        if best is None:
            raise tb.ConfigResourceError(f"[TS700] kernel {self.name}: the default config {start} and every "
                                         f"config of its relief ladder need more scratch than this GPU allows")
        self._set_launch(got, L, env, best[1], ints)
        L.config_source = "default, spill relief (still spills)"

    def _choose_config(self, got, L: "_Launch", int_prefix: list, sig) -> tuple:
        """(launch config or None for the default, source): the Triton config, then, for a
        contraction cuBLAS can compute, the faster of it and the cuBLAS call (library.py),
        measured once per shape bucket and cached with the tuned configs."""
        cfg, src = self._choose_triton_config(got, L, int_prefix, sig)
        form = self._lib_form
        if form is None or not self.library or self.force_config is not None or tune.mode(self.tune_mode) == "off" \
                or library_disabled():
            return cfg, src
        from . import library
        from . import triton_backend as tb
        import torch
        p, env = self.art.plan, L.env
        if not library.torch_precision_is(bool(getattr(p, "tf32", False))):
            return cfg, src
        names = sorted(b.name for b in self.ck.inputs)
        canon = dict(zip(names, [got[n] for n in L.in_order]))
        mats = library.views(form, canon)
        if mats is None:
            return cfg, src
        dev_name = torch.cuda.get_device_name(L.device)
        dtypes = [str(x[2]) for x in sig]
        key = tune.tune_key(self.tune_base_key + ":library", env, dtypes, dev_name,
                            ignore=tb.batch_symbols(p), coarse=tb.coarse_symbols(p))

        def check(c):
            ok = isinstance(c, dict) and set(c) == {"LIB"} and c["LIB"] in (0, 1)
            return None if ok else "not a library choice"
        choice = self._tuned.get(key) or tune.load(key, check)
        if choice is None:
            choice, rec = self._duel(got, L, int_prefix, cfg, form, canon, mats)
            rec.update(kernel=self.name, kernel_key=self.key, plan="contraction library",
                       sizes=dict(env), device=dev_name)
            tune.stats.log.append(rec)
            tune.store(key, choice, rec)
        self._tuned[key] = choice
        if choice.get("LIB"):
            return {"LIB": 1}, "library"
        return cfg, src

    def _duel(self, got, L, int_prefix, cfg, form, canon, mats):
        """Time the Triton launch against the cuBLAS call on these inputs: ({"LIB": 0 or 1}, record)."""
        from . import library
        from . import triton_backend as tb
        import torch
        p, env = self.art.plan, L.env
        ins = [got[n] for n in L.in_order]
        outs = [torch.empty(shape, dtype=dt, device=L.device)
                for (_, _, shape, _), dt in zip(L.out_specs, L.torch_dtypes)]
        c = tb.default_config(p, env) if cfg is None else cfg
        fn, grid, extra, meta = tb.prepare(self.art.module, self.ck, p, env, config=c)
        post = tb.split_k_epilogue(self.art.module, p, env, c)
        ws = (torch.empty(tb.k_slots(p, env, c) * tb.workspace_elems(p, env), dtype=torch.float32,
                          device=L.device) if post is not None else outs[0])
        ints = int_prefix + extra

        def run_triton():
            fn[grid](*ins, *outs, *ints, ws, **meta)
            if post is not None:
                post[0][post[1]](*ins, *outs, *ints, ws, **post[2])

        def run_lib():
            library.run(form, canon, outs[0], mats)
        t0 = time.perf_counter()
        with torch.cuda.device(L.device):
            run_triton()          # compiles the Triton binary (needed anyway when it wins)
            run_lib()
            torch.cuda.synchronize()
            # the fastest of interleaved rounds: on a shared GPU a median can hold another job's time
            t_tri = t_lib = float("inf")
            for _ in range(3):
                t_tri = min(t_tri, tune._bench(run_triton, 5, "min"))
                t_lib = min(t_lib, tune._bench(run_lib, 5, "min"))
        lib = t_lib < t_tri
        rec = {"triton_ms": t_tri, "library_ms": t_lib, "chosen": "library" if lib else "triton",
               "tune_seconds": time.perf_counter() - t0, "config": c}
        self.tune_seconds += rec["tune_seconds"]
        return {"LIB": int(lib)}, rec

    def _choose_triton_config(self, got, L: "_Launch", int_prefix: list, sig) -> tuple:
        """(launch config or None for the default, source) for a new input signature."""
        from . import triton_backend as tb
        import torch
        p, env = self.art.plan, L.env
        if self.force_config is not None:
            return dict(self.force_config), "forced"
        # deterministic and the bits follow the config: only the parameters that keep the summation
        # order are tuned (tb.ORDER_FREE), so every choice gives the default's bits on any GPU
        ordered = bool(self.opts.get("deterministic")) and tb.bits_follow_config(p, env)
        if ordered and len(tb.order_candidates(p, env)) < 2:
            return None, "pinned"
        m = tune.mode(self.tune_mode)
        if m == "off":
            return None, "pinned" if ordered else "default"
        # Batch like symbols are not part of the key: one tuned config serves every batch size, so a
        # new batch size never compiles a new Triton variant. A config tuned at a large batch is
        # therefore also used by later small launches of the same kernel. The row count of a
        # contraction is bucketed by powers of four (tb.coarse_symbols), so nearby batch sizes
        # share one search.
        batch = tb.batch_symbols(p)
        coarse = tb.coarse_symbols(p)
        worth = m != "auto" or tb.worth_tuning(p, env)
        if not worth and not batch and not coarse:
            return None, "pinned" if ordered else "default"
        dev_name = torch.cuda.get_device_name(L.device)
        dtypes = [str(x[2]) for x in sig]
        key = tune.tune_key(self.tune_base_key, env, dtypes, dev_name, ignore=batch, coarse=coarse)
        okey = tb.order_key(p, env, tb.default_config(p, env)) if ordered else None

        def check(c):
            why = tb.check_config(p, env, c)
            if why is None and ordered and tb.order_key(p, env, c) != okey:
                why = "changes the summation order of a deterministic kernel"
            return why
        cfg = self._tuned.get(key)
        if cfg is not None and check(cfg) is None:
            return cfg, "order tuned" if ordered else "cache"
        cfg = tune.load(key, check)
        if cfg is not None:
            self._tuned[key] = cfg
            return cfg, "cache"
        # a config tuned for another kernel with the same operands and sizes (only the epilogue
        # differs): the operand loads and the tile loop are what the timing measured
        shape_key = self.tune_shape_key(env, dtypes, dev_name, batch, coarse)
        if shape_key is not None:
            cfg = tune.load(shape_key, check)
            if cfg is not None:
                self._tuned[key] = cfg
                return cfg, "shape cache"
        if not worth:
            return (None, "pinned") if ordered else self._reuse(p, env, L.device)
        ins = [got[n] for n in L.in_order]
        outs = [torch.empty(shape, dtype=dt, device=L.device)
                for (_, _, shape, _), dt in zip(L.out_specs, L.torch_dtypes)]
        if self.checks:   # a scratch flag buffer: tuning launches never report
            outs.append(torch.zeros(checks_mod.WORDS, dtype=torch.int32, device=L.device))

        def make_launch(c):
            fn, grid, extra, meta = tb.prepare(self.art.module, self.ck, p, env, config=c)
            ints = int_prefix + extra
            post = tb.split_k_epilogue(self.art.module, p, env, c)
            tail = []
            if p.kind in ("contraction", "attention"):
                tail = [torch.empty(tb.launch_workspace_elems(p, env, c), dtype=torch.float32,
                                    device=L.device) if post is not None else outs[0]]

            def run():
                fn[grid](*ins, *outs, *ints, *tail, **meta)
                if post is not None:
                    post[0][post[1]](*ins, *outs, *ints, *tail, **post[2])

            def build():
                ks = [fn.warmup(*ins, *outs, *ints, *tail, grid=grid, **meta)]
                if post is not None:
                    ks.append(post[0].warmup(*ins, *outs, *ints, *tail, grid=post[1], **post[2]))
                why = tb.check_resources(ks, c)
                if why is not None:     # the search records the reason and skips the config
                    raise tb.ConfigResourceError(f"[TS700] {why}")
            return run, build

        with torch.cuda.device(L.device):
            best, rec = tune.search(make_launch, tb.order_candidates(p, env) if ordered else tb.candidates(p, env))
            if ordered and best != tb.default_config(p, env):
                # the order argument checked on these inputs: the choice must give the default's bits
                outs_ref = []
                for c in (tb.default_config(p, env), best):
                    make_launch(c)[0]()
                    outs_ref.append([o.clone() for o in outs[:len(L.out_specs)]])
                if not all(torch.equal(a, b) for a, b in zip(*outs_ref)):
                    rec["order_check"] = "failed: kept the default"
                    best = tb.default_config(p, env)
                else:
                    rec["order_check"] = "bitwise equal to the default"
        self.tune_seconds += rec["tune_seconds"]
        rec.update(kernel=self.name, kernel_key=self.key, plan=p.kind, sizes=dict(env),
                   bucket=tune.bucket(env), device=dev_name)
        tune.stats.log.append(rec)
        tune.store(key, best, rec)
        if shape_key is not None:
            tune.store(shape_key, best, rec)
        self._tuned[key] = best
        return best, "order tuned" if ordered else "tuned"

    def tune_shape_key(self, env, dtypes, dev_name, batch, coarse) -> Optional[str]:
        """Tuning cache key of a contraction's operands and sizes, without its epilogue, or None
        for other loop shapes (their best config depends on the whole expression)."""
        p = self.art.plan
        if p is None or p.kind != "contraction":
            return None
        sizes = [a.size if isinstance(a.size, int) else a.size.name
                 for a in (p.m, p.n, p.k) + tuple(p.batch) + tuple(getattr(p, "fold", ()))]
        text = "\n".join([f"v{CODEGEN_VERSION}", "contraction operands", _opts_text({k: v for k, v in self.opts.items() if k not in ("checks", "check_loads")}),
                          _FLOAT_RE.sub("C", ir.to_text(p.a)), _FLOAT_RE.sub("C", ir.to_text(p.b)),
                          repr(sizes)])
        base = hashlib.sha256(text.encode()).hexdigest()[:32]
        return tune.tune_key(base, env, dtypes, dev_name, ignore=batch, coarse=coarse)

    def _reuse(self, p, env, device) -> tuple:
        """Config for a launch not worth tuning: the default, unless the default would compile a new
        Triton variant while one this kernel already compiled is about as good here (within
        tb.REUSE_SLACK by tb.estimate) and the launch is small (below tb.REUSE_MAX_WORK). A new batch size then costs no Triton compile. Every config is still
        verified at every new shape signature."""
        from . import triton_backend as tb
        import torch
        d = tb.default_config(p, env)
        have = []
        for L in self._fast.values():
            c = L.config
            if c is not None and c not in have and tb.check_config(p, env, c) is None:
                have.append(c)
        if not have or d in have or p.kind != "contraction" or tb.work(p, env) >= tb.REUSE_MAX_WORK:
            return None, "default"
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        best = min(have, key=lambda c: tb.estimate(p, env, c, sms))
        if tb.estimate(p, env, best, sms) <= tb.REUSE_SLACK * tb.estimate(p, env, d, sms):
            return dict(best), "reused"
        return None, "default"

    @property
    def checks(self) -> frozenset:
        """Check classes compiled into this kernel (kernel/checks.py)."""
        return frozenset(c for c in str(self.opts.get("checks", "")).split(",") if c)

    def __call__(self, *args, _chk=None, **kwargs):
        """``_chk``: an int32 tensor of ``checks.WORDS`` flag words on the kernel's device, required
        when the kernel was compiled with checks; a word becomes 1 when its class fired."""
        got = self._gather(args, kwargs)
        sig = self._signature(got)
        L = self._fast.get(sig)
        if L is None:
            L = self._prepare(got, sig)
        if self.checks and _chk is None:
            raise TypeError(f"kernel {self.name} was compiled with checks ({checks_mod.text(self.checks)}); "
                            f"pass the flag buffer as _chk")
        # without a flag buffer the run methods are called as before (tests wrap them)
        if self.art.backend == "triton":
            outs = self._run_triton(got, L) if _chk is None else self._run_triton(got, L, chk=_chk)
        elif self.art.backend == "metal":
            outs = self._run_metal(got, L)
        else:
            outs = self._run_cpu(got, L) if _chk is None else self._run_cpu(got, L, _chk)
        mode = verify.verify_mode()
        col = getattr(_tls, "collect", None)
        if col is not None:
            vkey = _vkey(self.key, L, sig, got, self.art.backend)
            col.seen.add(_vkey_text(vkey))
            if col.defer and mode == "first" and not L.verified:
                if vkey not in _verified:
                    col.pending.append((vkey, L))   # accepted from the program record, or checked
                else:
                    L.verified = True
                mode = "off"
        if mode == "all" or (mode == "first" and not L.verified):
            # a different launch config is a different binary, so it is verified on its own
            vkey = _vkey(self.key, L, sig, got, self.art.backend)
            if mode != "all" and vkey in _verified and verify_group.get() is not None:
                group_skips[0] += 1
            if mode == "all" or vkey not in _verified:
                t = time.perf_counter()
                self.check(got, outs, L.env_user)
                self.verify_seconds += time.perf_counter() - t
                _verified.add(vkey)
            L.verified = True
        if len(outs) == 1:
            return next(iter(outs.values()))
        return outs

    def call_into(self, out, *args, _chk=None):
        """Like ``__call__`` for a one output Triton kernel, but the result is written into ``out``,
        a contiguous tensor of the output's shape and dtype. ``out`` may be the storage of one of the
        inputs: the compiler's memory planner (docs/MEMORY.md) does that only for a kernel it proved
        elementwise in that input (no reduction, every load of it at the output's own index), where
        each element is read by the program that writes it, before it writes it. When the kernel
        layer's check runs on this call, it is given a copy of the inputs taken before the launch."""
        import torch
        got = self._gather(args, {})
        sig = self._signature(got)
        L = self._fast.get(sig)
        if L is None:
            L = self._prepare(got, sig)
        if self.art.backend != "triton" or len(L.out_specs) != 1:
            raise TypeError(f"kernel {self.name}: call_into needs a one output Triton kernel")
        shape, dt = L.out_specs[0][2], L.torch_dtypes[0]
        if tuple(out.shape) != tuple(shape) or out.dtype != dt or not out.is_contiguous() \
                or out.device != L.device:
            raise TypeError(f"kernel {self.name}: call_into needs a contiguous {dt} tensor of shape "
                            f"{tuple(shape)} on {L.device}, got {out.dtype} {tuple(out.shape)} on {out.device}")
        mode = verify.verify_mode()
        vkey = _vkey(self.key, L, sig, got, self.art.backend)
        col = getattr(_tls, "collect", None)
        if col is not None:
            col.seen.add(_vkey_text(vkey))
            if col.defer and mode == "first" and not L.verified:
                if vkey not in _verified:
                    col.pending.append((vkey, L))
                else:
                    L.verified = True
                mode = "off"
        check = mode == "all" or (mode == "first" and not L.verified and vkey not in _verified)
        if mode == "first" and not L.verified and not check:
            group_skips[0] += 1
        seen = got
        if check:   # the launch overwrites an input: the check reads the values it had before
            base = out.untyped_storage().data_ptr()
            seen = {n: (v.clone() if isinstance(v, torch.Tensor) and v.untyped_storage().data_ptr() == base
                        else v) for n, v in got.items()}
        if self.checks and _chk is None:
            raise TypeError(f"kernel {self.name} was compiled with checks; pass the flag buffer as _chk")
        outs = self._run_triton(got, L, [out]) if _chk is None else self._run_triton(got, L, [out], _chk)
        if mode == "all" or (mode == "first" and not L.verified):
            if check:
                t = time.perf_counter()
                self.check(seen, outs, L.env_user)
                self.verify_seconds += time.perf_counter() - t
                _verified.add(vkey)
            L.verified = True
        return out

    def tolerance(self, env: Dict[str, int]) -> float:
        return verify.tolerance(self.kernel, env, allow_tf32=bool(self.opts.get("allow_tf32")))

    def check(self, inputs: Dict, outs: Dict, env: Optional[Dict[str, int]] = None):
        """Compare outputs with the float64 reference; raise KernelMismatchError on mismatch."""
        if env is None:
            env = reference.bind_sizes(self.kernel, {n: tuple(v.shape) for n, v in inputs.items()})
        dev = _reference_device(inputs, outs)
        if dev is not None and verify.verify_device() == "auto" and                 reference.work_estimate(self.kernel, env) < reference.GPU_MIN_WORK:
            dev = None  # small: the NumPy reference is faster than the GPU's fixed costs
        if dev is not None:
            # float64 reference on the kernel's own GPU; same math and tolerance as on the CPU
            refs = reference.run_reference_torch(self.kernel, inputs, dev)

            def mag():
                mk = verify.magnitude_kernel(self.kernel)
                return None if mk is None else reference.run_reference_torch(mk, inputs, dev)
            def rounded():
                return _derived_runs(self.kernel, {n: _to_f64(v) for n, v in inputs.items()},
                                     {o.name: refs[o.name] for o in self.kernel.outputs})
            verify.check_outputs(self.kernel, self.backend, dict(outs), refs, self.tolerance(env), mag,
                                 bool(self.opts.get("allow_tf32")), rounded)
            return
        np_in = {n: _to_f64(v) for n, v in inputs.items()}
        refs = reference.run_reference(self.kernel, np_in)
        np_out = {n: _to_f64(v) for n, v in outs.items()}

        def mag_np():
            mk = verify.magnitude_kernel(self.kernel)
            return None if mk is None else reference.run_reference(mk, np_in)
        def rounded_np():
            return _derived_runs(self.kernel, np_in, refs)
        verify.check_outputs(self.kernel, self.backend, np_out, refs, self.tolerance(env), mag_np,
                             bool(self.opts.get("allow_tf32")), rounded_np)

    def _run_triton(self, got, L: "_Launch", outs=None, chk=None):
        import torch
        if outs is None:
            outs = [torch.empty(shape, dtype=dt, device=L.device)
                    for (_, _, shape, _), dt in zip(L.out_specs, L.torch_dtypes)]
        if L.lib is not None:
            from . import library
            if library.torch_precision_is(bool(getattr(self.art.plan, "tf32", False))):
                canon = dict(zip(L.lib_names, [got[n] for n in L.in_order]))
                with torch.cuda.device(L.device):
                    library.run(L.lib, canon, outs[0])
                return {spec[0]: t for spec, t in zip(L.out_specs, outs)}
            L = self._alt(got, L)   # PyTorch's precision is not the kernel's now: the Triton launch
        ins = [got[n] for n in L.in_order]
        ptrs = outs if chk is None else outs + [chk]
        if L.device.index is not None and L.device.index != torch.cuda.current_device():
            with torch.cuda.device(L.device):
                self._launch(L, ins, ptrs)
        else:
            self._launch(L, ins, ptrs)
        return {spec[0]: t for spec, t in zip(L.out_specs, outs)}

    def _alt(self, got, L: "_Launch") -> "_Launch":
        """The Triton launch of a signature whose chosen config is the cuBLAS call, built once."""
        if L.alt is None:
            from . import triton_backend as tb
            A = _Launch(env_user=L.env_user, env=L.env, out_specs=L.out_specs, in_order=L.in_order,
                        sym_vals=L.sym_vals, shape_sig=L.shape_sig, device=L.device,
                        torch_dtypes=L.torch_dtypes)
            strides = []
            for n in L.in_order:
                strides += list(got[n].stride())
            for _, _, shape, _ in L.out_specs:
                strides += _contiguous_strides(shape)
            fn, grid, extra, meta = tb.prepare(self.art.module, self.ck, self.art.plan, L.env)
            if "amd_arch" in self.opts:   # the same AMD launch options as _set_launch
                from . import rocm
                meta = dict(meta, **rocm.launch_options(rocm.current_target(L.device), self.art.plan.kind))
            A.config = tb.default_config(self.art.plan, L.env)
            A.config_source = "default (PyTorch's precision is not the library choice's)"
            A.fn, A.grid, A.meta = fn, grid, meta
            A.int_args = strides + L.sym_vals + extra
            A.has_ws = True
            A.post = tb.split_k_epilogue(self.art.module, self.art.plan, L.env, A.config)
            if A.post is not None:
                A.ws_numel = tb.k_slots(self.art.plan, L.env, A.config) * tb.workspace_elems(self.art.plan, L.env)
            L.alt = A
        return L.alt

    def _run_metal(self, got, L: "_Launch"):
        import torch
        outs = [torch.empty(shape, dtype=dt, device=L.device)
                for (_, _, shape, _), dt in zip(L.out_specs, L.torch_dtypes)]
        if L.meta["work"] > 0:
            ins = []
            for n in L.in_order:
                t = got[n]
                if not t.is_contiguous():
                    t = t.contiguous()
                elif t.storage_offset() != 0:   # the kernel indexes from the start of the buffer
                    t = t.clone()
                ins.append(t)
            L.fn(*ins, *outs, L.meta["params"], threads=L.grid[0], group_size=L.grid[1])
        return {spec[0]: t for spec, t in zip(L.out_specs, outs)}

    def _build_split(self, got, L: "_Launch"):
        """Compile both kernels of a split K or split KV launch at once (Triton releases the GIL
        for most of a compile), instead of one after the other at the first launch."""
        import torch
        from concurrent.futures import ThreadPoolExecutor
        ins = [got[n] for n in L.in_order]
        outs = [torch.empty(shape, dtype=dt, device=L.device)
                for (_, _, shape, _), dt in zip(L.out_specs, L.torch_dtypes)]
        if self.checks:   # a scratch flag buffer for the compile launches
            outs.append(torch.zeros(checks_mod.WORDS, dtype=torch.int32, device=L.device))
        ws = torch.empty(L.ws_numel, dtype=torch.float32, device=L.device)
        fn2, grid2, meta2 = L.post
        with torch.cuda.device(L.device), ThreadPoolExecutor(2) as ex:
            jobs = [ex.submit(L.fn.warmup, *ins, *outs, *L.int_args, ws, grid=L.grid, **L.meta),
                    ex.submit(fn2.warmup, *ins, *outs, *L.int_args, ws, grid=grid2, **meta2)]
            kernels = [j.result() for j in jobs]
        self._check_resources(kernels, L)
        return kernels

    def _build_one(self, got, L: "_Launch"):
        """Compile the launch's kernel before its first run, so its resources are checked."""
        import torch
        ins = [got[n] for n in L.in_order]
        outs = [torch.empty(shape, dtype=dt, device=L.device)
                for (_, _, shape, _), dt in zip(L.out_specs, L.torch_dtypes)]
        tail = [outs[0]] if L.has_ws else []    # no split: the workspace pointer is not read
        if self.checks:   # the flag buffer follows the outputs, as in every launch; scratch, never read
            outs.append(torch.zeros(checks_mod.WORDS, dtype=torch.int32, device=L.device))
        with torch.cuda.device(L.device):
            kernels = [L.fn.warmup(*ins, *outs, *L.int_args, *tail, grid=L.grid, **L.meta)]
        self._check_resources(kernels, L)
        return kernels

    def _check_resources(self, kernels, L: "_Launch"):
        from . import triton_backend as tb
        why = tb.check_resources(kernels, L.config)
        if why is not None:
            raise tb.ConfigResourceError(f"[TS700] kernel {self.name}: {why}")

    @staticmethod
    def _launch(L: "_Launch", ins, outs):
        if not L.has_ws:
            L.fn[L.grid](*ins, *outs, *L.int_args, **L.meta)
            return
        import torch
        # contraction and attention: the split K or split KV workspace (any pointer when there is
        # no split, it is not read)
        ws = outs[0] if L.post is None else torch.empty(L.ws_numel, dtype=torch.float32,
                                                        device=L.device)
        L.fn[L.grid](*ins, *outs, *L.int_args, ws, **L.meta)
        if L.post is not None:
            fn, grid, meta = L.post
            fn[grid](*ins, *outs, *L.int_args, ws, **meta)

    def _run_cpu(self, got, L: "_Launch", chk=None):
        env = L.env
        import torch
        arrays = {}
        for b in self.kernel.inputs:
            v = got[b.name]
            if isinstance(v, torch.Tensor):
                if v.device.type != "cpu":
                    raise TypeError(f"kernel {self.name} on cpu: input {b.name} is on {v.device}")
                v = v.detach()
                if v.dtype in (torch.float16, torch.bfloat16):
                    v = v.float()
                v = v.numpy()
            else:
                v = np.asarray(v)
                if v.dtype == np.float16:
                    v = v.astype(np.float32)
            arrays[self.in_map[b.name]] = v
        outs_np = {}
        for co, o in zip(self.ck.outputs, self.kernel.outputs):
            shape = tuple(a.size if isinstance(a.size, int) else env[a.size.name] for a in co.axes)
            dt = np.float64 if o.dtype == "float64" else np.float32
            outs_np[co.name] = np.empty(shape if shape else (1,), dtype=dt)
        args = [arrays[b.name] for b in sorted(self.ck.inputs, key=lambda b: b.name)]
        args += [outs_np[o.name] for o in self.ck.outputs]
        args += [env[s.name] for s in sorted(self.ck.symbols(), key=lambda s: s.name)]
        words = None
        if chk is not None:
            words = np.zeros(checks_mod.WORDS, dtype=np.int32)
            args.append(words)
        self.art.module.kern(*args)
        res = {}
        for o, co in zip(self.kernel.outputs, self.ck.outputs):
            arr = outs_np[co.name]
            if not co.axes:
                arr = arr.reshape(())
            res[o.name] = torch.from_numpy(arr).to(getattr(torch, o.dtype))
            if words is not None and "overflow" in self.checks and o.dtype in ("float16", "bfloat16"):
                # numba stores float32; the rounding to half precision happens here
                if bool((torch.isinf(res[o.name]) & torch.isfinite(torch.from_numpy(arr))).any()):
                    words[checks_mod.SLOT["overflow"]] = 1
        if words is not None:
            chk.copy_(torch.maximum(chk, torch.from_numpy(words).to(chk.device)))
        return res


@dataclass
class _Launch:
    env_user: Dict[str, int]
    env: Dict[str, int]
    out_specs: list
    in_order: list
    sym_vals: list
    shape_sig: tuple
    verified: bool = False
    device: object = None
    fn: object = None
    grid: tuple = ()
    meta: dict = field(default_factory=dict)
    int_args: list = field(default_factory=list)
    torch_dtypes: list = field(default_factory=list)
    config: Optional[dict] = None
    config_source: str = ""
    has_ws: bool = False        # contraction and attention kernels take a workspace pointer
    post: Optional[tuple] = None  # split K second pass (function, grid, meta)
    ws_numel: int = 0
    lib: object = None          # library.LibForm when the launch runs as a cuBLAS call
    lib_names: list = field(default_factory=list)   # canonical input names in in_order
    alt: Optional["_Launch"] = None   # the Triton launch, when PyTorch's precision is not the kernel's


_FLOAT_RE = re.compile(r"(?<![\w.])(?:\d+\.\d*(?:e[-+]?\d+)?|\d+e[-+]?\d+)")


def library_disabled() -> bool:
    """TRUESTAR_LIBRARY=0 keeps every contraction on Truestar's own kernels (for measurement)."""
    return os.environ.get("TRUESTAR_LIBRARY", "1").strip().lower() in ("0", "off", "false", "no")


def _spills(kernels) -> int:
    """Bytes of local memory (spilled registers, scratch on AMD) per thread of the worst kernel."""
    return max((int(getattr(k, "n_spills", 0) or 0) * 4 for k in kernels), default=0)


def _contiguous_strides(shape) -> list:
    st, acc = [], 1
    for x in reversed(shape):
        st.append(acc)
        acc *= x
    return list(reversed(st))


def _reference_device(inputs: Dict, outs: Dict):
    """The CUDA device to run the float64 reference on, or None for the NumPy reference: every
    input and output must be a tensor on the same CUDA device, and TRUESTAR_VERIFY_DEVICE must
    not be 'cpu'. The size rule of the default 'auto' mode is applied by the caller."""
    if verify.verify_device() == "cpu":
        return None
    try:
        import torch
    except ImportError:
        return None
    dev = None
    for v in list(inputs.values()) + list(outs.values()):
        if not isinstance(v, torch.Tensor) or v.device.type != "cuda":
            return None
        if dev is not None and v.device != dev:
            return None
        dev = v.device
    return dev


def _to_f64(v) -> np.ndarray:
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return verify.to_host_f64(v).numpy()
    except ImportError:
        pass
    return np.asarray(v, dtype=np.float64)


def compile(k: ir.Kernel, backend: str = "cuda", allow_tf32: bool = False,
            tune: Optional[bool] = None, deterministic: bool = False, checks=None,
            check_loads: Optional[Dict[str, tuple]] = None, library: bool = False,
            order_free: bool = False, math: str = "default", portable_tensor_cores: bool = False,
            **hidden) -> CompiledKernel:
    """Compile kernel k for backend 'cuda' (Triton), 'rocm' (Triton on an AMD GPU), 'cpu'
    (numba) or 'metal' (Metal Shading Language compiled at run time on macOS, docs/METAL.md).

    On a ROCm build of torch the GPU is torch "cuda", so 'cuda' and 'rocm' both compile for the
    attached AMD GPU; 'rocm' without ROCm raises RocmUnavailable (TS883), never another device.
    AMD builds follow the GPU's arch, wave size, CUs and LDS (kernel/rocm.py, docs/ROCM.md); TF32
    on an arch without it runs strict float32 (TS884).

    allow_tf32 lets float32 dot products use TF32 tensor cores (faster, about 1e-3 relative error);
    it is off by default so float32 stays float32. Raises triton_backend.Unschedulable when the
    kernel does not fit a GPU loop shape: there is no silent fallback to another backend. Raises
    truestar.deps.MissingDependencyError (TS022) for 'cpu' when numba is not installed.

    deterministic: contraction results must not depend on the launch config, and so not on the
    batch size: split K is never used (see docs/KERNEL_IR.md, determinism). Every kernel is
    deterministic run to run in either mode (no atomics; split K adds partials in a fixed order).

    tune: True autotunes every GPU launch config, False never does (fixed heuristic), None
    follows TRUESTAR_TUNE (default auto: tune launches big enough to matter). Tuned configs are
    cached on disk per kernel, shape bucket, dtypes and device (see tune.py).

    checks: compiled in runtime checks (kernel/checks.py): None or "" for none, else classes such
    as "nan,div". A kernel with checks is a different binary and takes the ``_chk`` flag buffer.
    check_loads: {input name: classes} also checked where the kernel loads that input (GPU only).

    order_free: every sum of a row reduction kernel becomes an order free sum (osum,
    docs/ORDER_FREE_SUM.md): the same float32 bits for any block size, order, split or SM count,
    so such a kernel is tuned even when deterministic. Contractions and attention keep their sums
    (tensor core inner loops; the result's order_free dict says which way a kernel went).

    math: "default" (native exp, log, tanh, erf, rsqrt, sigmoid) or "round_once": each of those is
    evaluated in float64 by Truestar's own correctly rounded operation sequence and rounded once to
    float32, so its bits are the same on every device (docs/ROUND_ONCE.md). Triton launches of such
    kernels run with enable_fp_fusion=False. Refused, never silently downgraded: Metal (TS1089), an
    unknown mode (TS1090), float64 buffers (TS1091), ops without a round once form (TS1092) and the
    attention loop shape (TS1093).

    portable_tensor_cores: float16, bfloat16 and TF32 dot products (contraction, convolution and
    attention kernels) run as exact tensor core blocks (kernel/tcexact.py, docs/PORTABLE_TC.md):
    every tensor core instruction sums integers exactly, and all rounding happens in a fixed order
    in float32, so the bits are the same on every NVIDIA generation from sm_80 on. Implies
    deterministic. Refused, never downgraded: a backend other than CUDA Triton (TS1330), ROCm or a
    GPU below sm_80 (TS1331), a dot product K block outside 16 to 256 (TS1332).
    """
    of_note = None
    if order_free:
        k, of_note = _order_free_kernel(k)
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; use 'cuda', 'rocm', 'cpu' or 'metal'")
    be = _BACKENDS[backend]
    from . import roundonce
    roundonce.check_mode(math)
    if math == "round_once":
        roundonce.check_kernel(k, be)
    amd = None
    if backend in ("rocm", "hip"):
        from . import rocm
        rocm.require_hip(f"K.compile(..., {backend!r})")
        amd = rocm.current_target()
    elif be == "triton":
        from . import rocm
        if rocm.hip_version() and rocm.hip_available()[0]:
            amd = rocm.current_target()
    if be == "numba":
        from ..deps import require
        require("numba", _NUMBA_PURPOSE)
    opts = {}
    if be == "metal":
        from . import metal_backend as mb
        mb.require("the kernel layer's 'metal' backend")
        if checks_mod.text(checks_mod.parse(checks)):
            raise mb.MetalUnsupported("TS842", f"kernel {k.name}: compiled in checks (sentinels) are not "
                                               f"generated for Metal kernels yet; compile without checks")
        opts["simdgroup"] = mb.simdgroup_enabled()
        opts["attention"] = mb.attention_option()
        if mb.mm_option():
            opts["mm"] = mb.mm_option()
    if allow_tf32:
        opts["allow_tf32"] = True
    if math == "round_once":
        opts["math"] = "round_once"
    if portable_tensor_cores:
        from . import tcexact
        if be != "triton":
            raise tcexact.PortableTensorCoreError(
                "TS1330", f"kernel {k.name}: portable tensor cores are generated for the CUDA Triton "
                          f"backend only, not {backend!r}")
        if amd is not None:
            raise tcexact.PortableTensorCoreError(
                "TS1331", f"kernel {k.name}: portable tensor cores are proven for NVIDIA sm_80 and "
                          f"newer only; AMD matrix cores were not measured")
        tcexact.check_device()
        opts["portable_tc"] = True
        deterministic = True
    if deterministic:
        opts["deterministic"] = True
    chk = checks_mod.text(checks_mod.parse(checks))
    if chk:
        opts["checks"] = chk
    if amd is not None:
        from . import rocm
        opts, notes = rocm.kernel_opts(amd, opts)
        for n in notes:
            warnings.warn(n, rocm.RocmNote, stacklevel=2)
    if "_break" in hidden:
        opts["_break"] = hidden.pop("_break")
    if "_sched" in hidden:  # measurement hook: override code generation choices
        opts["_sched"] = dict(hidden.pop("_sched"))
    force = hidden.pop("_config", None)  # test hook: launch with this config (not tuned)
    if hidden:
        raise TypeError(f"unknown compile options: {', '.join(hidden)}")
    ck, in_map, out_map = ir.canonicalize(k)
    if check_loads and chk and be == "triton":
        # input buffer name -> classes checked on every load of it (the GPU backend only)
        parts = sorted(f"{in_map[n]}:{'+'.join(c for c in checks_mod.CLASSES if c in set(cls))}"
                       for n, cls in check_loads.items() if n in in_map and cls)
        if parts:
            opts["check_loads"] = ";".join(parts)
    text = f"v{CODEGEN_VERSION}\n{be}\n{_opts_text(opts)}\n{ck.to_text()}"
    key = hashlib.sha256(text.encode()).hexdigest()[:32]
    with _lock:
        art = _memory.get(key)
        if art is not None:
            stats.memory_hits += 1
        else:
            art = _build(ck, be, opts, key)
            _memory[key] = art
    out = CompiledKernel(k, _DEVICE_OF[be], art, ck, in_map, out_map, opts,
                         tune_mode=tune, force_config=force, library=library)
    out.order_free = of_note
    return out


def _order_free_kernel(k: ir.Kernel):
    """(kernel, note): k with its sums made order free when it is a row reduction kernel."""
    from . import triton_backend as tb
    sums = sum(1 for r in k.reduces() if r.op == "sum")
    if not sums:
        return k, {"applied": False, "sums": 0, "reason": "no sum to rewrite"}
    try:
        kind = tb.plan(k).kind
    except tb.Unschedulable as e:
        kind = f"unschedulable ({e})"
    if kind != "tile":
        return k, {"applied": False, "sums": sums,
                   "reason": f"loop shape {kind}: only row reductions take order free sums"}
    new = ir.Kernel(k.name, [ir.Output(o.name, o.axes, ir.to_order_free(o.expr), o.dtype)
                             for o in k.outputs])
    return new, {"applied": True, "sums": sums, "reason": ""}


def _derived_runs(k, np_in, refs, samples: int = 8):
    """{output: (rounded run, sigma)} of kernel k on these inputs (NumPy, float64): the run with every
    node rounded to float32 and the spread of ``samples`` error model runs around the float64 result
    (reference.RoundedEvaluator, reference.ErrorModelEvaluator; docs/TOLERANCE.md, Kernels)."""
    import numpy as np
    from . import verify as _v
    rr = reference.run_reference(k, np_in, reference.RoundedEvaluator)
    r64 = {n: _v.to_host_f64(v).numpy() if hasattr(v, "detach") else np.asarray(v, dtype=np.float64)
           for n, v in refs.items()}
    acc = {n: np.zeros_like(r64[n]) for n in r64}
    for s in range(samples):
        runs = reference.run_reference(k, np_in, lambda inp, env, s=s: reference.ErrorModelEvaluator(inp, env, s))
        for n in acc:
            d = np.where(np.isfinite(r64[n]), runs[n] - r64[n], 0.0)
            acc[n] += np.nan_to_num(d) ** 2
    return {n: (rr[n], np.sqrt(acc[n] / samples)) for n in acc}
