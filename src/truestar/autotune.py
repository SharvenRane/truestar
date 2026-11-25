"""`@truestar.verified_autotune`: Triton autotuning that checks every config's answer, not only its
time (R7).

`@triton.autotune` times each config and keeps the fastest. A config that computes the wrong
answer (a block size that does not divide a dimension the kernel assumes it divides, a miscompiled
unroll, a dropped last tile) is kept if it is fastest, and nothing says so. Here, on the first call
for each autotune key:

1. the tolerance is derived once from the reference function in float64 (truestar.tolerance: a per
   element bound from a rounded run and an error model of the dtype, no rtol or atol);
2. each config runs on the real arguments with every output prefilled with NaN (and the
   `reset_to_zero` arguments zeroed), and its outputs are checked against that tolerance; a config
   that fails is removed from the tuning set with a named reason (TS207: which outputs, how many
   elements, where they sit, the error in ulps) and a VerifiedAutotuneWarning;
3. only the configs that pass are timed (triton.testing.do_bench, as triton.autotune does) and the
   fastest of them is used and cached for the key.

If no config passes, the call raises VerifiedAutotuneError listing every reason: a wrong result is
never returned. Usage, with the same arguments as triton.autotune plus the reference:

    @truestar.verified_autotune(configs=[...], key=["N"], reference=lambda x: torch.softmax(x, -1),
                                inputs=["x_ptr"], outputs=["y_ptr"])
    @triton.jit
    def softmax_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr): ...

    softmax_kernel[grid](x, y, N)

`reference` takes the tensors named in `inputs`, in that order, and returns the tensors named in
`outputs` (one tensor, or a tuple in that order). `.reports` maps each key to a TuneReport.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .tolerance import clear_cache, derive


class VerifiedAutotuneError(RuntimeError):
    """No autotune config computes the right answer for these arguments."""


class VerifiedAutotuneWarning(UserWarning):
    """An autotune config was removed because its answer is wrong (TS207)."""


@dataclass
class ConfigResult:
    config: str
    accepted: bool
    ms: Optional[float] = None
    reason: str = ""
    code: str = ""
    nondeterministic: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {"config": self.config, "accepted": self.accepted, "ms": self.ms,
                "reason": self.reason, "code": self.code,
                "nondeterministic": self.nondeterministic}


@dataclass
class TuneReport:
    key: tuple
    results: List[ConfigResult] = field(default_factory=list)
    best: Optional[str] = None
    derive_seconds: float = 0.0
    check_seconds: float = 0.0
    bench_seconds: float = 0.0

    @property
    def rejected(self) -> List[ConfigResult]:
        return [r for r in self.results if not r.accepted]

    def as_dict(self) -> Dict[str, Any]:
        return {"key": [repr(k) for k in self.key], "best": self.best,
                "results": [r.as_dict() for r in self.results],
                "derive_seconds": self.derive_seconds, "check_seconds": self.check_seconds,
                "bench_seconds": self.bench_seconds}


def _cfg_kwargs(cfg) -> Dict[str, Any]:
    kw = dict(cfg.all_kwargs()) if hasattr(cfg, "all_kwargs") else dict(
        cfg.kwargs, num_warps=cfg.num_warps, num_stages=cfg.num_stages)
    return {k: v for k, v in kw.items() if v is not None}


def _cfg_name(cfg) -> str:
    return ", ".join(f"{k}={v}" for k, v in _cfg_kwargs(cfg).items())


class VerifiedAutotuner:
    def __init__(self, fn, configs: Sequence, key: Sequence[str], reference: Callable,
                 inputs: Sequence[str], outputs: Sequence[str],
                 reset_to_zero: Sequence[str] = (), warmup: int = 25, rep: int = 100,
                 check_determinism: bool = True):
        self.fn = fn
        self.configs = list(configs)
        self.key = list(key)
        self.reference = reference
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.reset_to_zero = list(reset_to_zero)
        self.warmup, self.rep = warmup, rep
        self.check_determinism = check_determinism
        self.arg_names = list(fn.arg_names)
        for n in self.key + self.inputs + self.outputs + self.reset_to_zero:
            if n not in self.arg_names:
                raise ValueError(f"{n!r} is not an argument of {fn.__name__} "
                                 f"({', '.join(self.arg_names)})")
        self.cache: Dict[tuple, Any] = {}
        self.reports: Dict[tuple, TuneReport] = {}
        self.__name__ = getattr(fn, "__name__", "kernel")

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(grid, args, kwargs)

    # ------------------------------------------------------------------ helpers

    def _named(self, args, kwargs) -> Dict[str, Any]:
        named = dict(zip(self.arg_names, args))
        named.update(kwargs)
        return named

    def _key(self, named) -> tuple:
        k = [named.get(n) for n in self.key]
        k += [str(v.dtype) for v in named.values() if isinstance(v, torch.Tensor)]
        return tuple(k)

    def _launch(self, grid, args, kwargs, cfg, named):
        current = dict(kwargs, **_cfg_kwargs(cfg))
        g = grid(dict(named, **current)) if callable(grid) else grid
        for n in self.reset_to_zero:
            named[n].zero_()
        if getattr(cfg, "pre_hook", None):
            cfg.pre_hook(dict(named, **current))
        self.fn[g](*args, **current)

    # ------------------------------------------------------------------ tuning

    def _tune(self, grid, args, kwargs, named, key) -> Any:
        rep = TuneReport(key)
        t0 = time.perf_counter()
        ins = tuple(named[n] for n in self.inputs)
        outs = [named[n] for n in self.outputs]
        clear_cache()
        tol = derive(self.reference, ins, like=outs, device=outs[0].device, cache=False)
        rep.derive_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        ok_cfgs = []
        for cfg in self.configs:
            name = _cfg_name(cfg)
            for o in outs:
                o.fill_(float("nan")) if o.is_floating_point() else o.fill_(-1)
            try:
                self._launch(grid, args, kwargs, cfg, named)
                torch.cuda.synchronize(outs[0].device)
            except Exception as exc:
                rep.results.append(ConfigResult(name, False, reason=f"failed to compile or run: "
                                                                    f"{type(exc).__name__}: "
                                                                    f"{str(exc)[:200]}"))
                continue
            got = [o.clone() for o in outs]
            res = tol.check(got, explain=False)
            if not res.ok:
                why = res.message()
                rep.results.append(ConfigResult(name, False, reason=why, code="TS207"))
                warnings.warn(f"[TS207] {self.__name__}: config {name} removed from autotuning, its "
                              f"answer is wrong: {why}", VerifiedAutotuneWarning, stacklevel=4)
                continue
            nondet = False
            if self.check_determinism:
                for o in outs:
                    o.fill_(float("nan")) if o.is_floating_point() else o.fill_(-1)
                self._launch(grid, args, kwargs, cfg, named)
                torch.cuda.synchronize(outs[0].device)
                nondet = any(not torch.equal(torch.nan_to_num(a), torch.nan_to_num(b))
                             for a, b in zip(got, outs))
            ok_cfgs.append(cfg)
            rep.results.append(ConfigResult(name, True, nondeterministic=nondet))
        rep.check_seconds = time.perf_counter() - t1
        if not ok_cfgs:
            self.reports[key] = rep
            raise VerifiedAutotuneError(
                f"{self.__name__}: no autotune config gives the right answer for key {key}: "
                + "; ".join(f"[{r.config}] {r.reason}" for r in rep.results))
        from triton.testing import do_bench

        t2 = time.perf_counter()
        times = {}
        for cfg in ok_cfgs:
            try:
                times[id(cfg)] = do_bench(lambda c=cfg: self._launch(grid, args, kwargs, c, named),
                                          warmup=self.warmup, rep=self.rep)
            except Exception:
                times[id(cfg)] = float("inf")
        rep.bench_seconds = time.perf_counter() - t2
        best = min(ok_cfgs, key=lambda c: times[id(c)])
        for r, cfg in zip([r for r in rep.results if r.accepted], ok_cfgs):
            r.ms = times[id(cfg)]
        rep.best = _cfg_name(best)
        self.reports[key] = rep
        return best

    def run(self, grid, args, kwargs):
        named = self._named(args, kwargs)
        key = self._key(named)
        if key not in self.cache:
            self.cache[key] = self._tune(grid, args, kwargs, named, key)
        cfg = self.cache[key]
        self._launch(grid, args, kwargs, cfg, named)

    @property
    def best_config(self):
        return next(iter(self.cache.values()), None)


def verified_autotune(configs: Sequence, key: Sequence[str], reference: Callable,
                      inputs: Sequence[str], outputs: Sequence[str],
                      reset_to_zero: Sequence[str] = (), warmup: int = 25, rep: int = 100,
                      check_determinism: bool = True):
    """Decorator for a @triton.jit kernel: autotune over `configs` like triton.autotune, but every
    config's outputs are first checked against `reference` in float64 with a derived tolerance,
    and wrong configs are removed (see the module docstring)."""

    def deco(fn):
        return VerifiedAutotuner(fn, configs, key, reference, inputs, outputs, reset_to_zero,
                                 warmup, rep, check_determinism)

    return deco
