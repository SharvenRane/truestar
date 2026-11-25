"""CUDA graph capture for any executor: record a sequence of kernel launches once, replay it with
one launch.

An executor is any callable that runs a program on CUDA tensors: a kernel layer
`CompiledKernel` or `CompiledGraph`, the compiled executor of `truestar.compiler` when it lands,
or plain PyTorch code. `capture(fn, example_args)` returns a `CapturedFunction` that

* keeps static input buffers, copies each call's inputs into them, replays the graph and returns
  the static outputs (cloned by default, so a returned tensor never changes behind the caller);
* captures one graph per shape bucket when a dim is declared dynamic, lazily on first use;
* checks at capture time that the replay is bitwise equal to the eager run on the same inputs;
* refuses loudly (an exception naming the cause) for host syncs, inputs that are not CUDA
  tensors, functions that write to their inputs, and shapes outside the declared buckets.

See docs/RUNTIME.md for the measurements behind this layer and its memory cost.
"""

from __future__ import annotations

import bisect
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

Executor = Callable[..., Any]
"""Anything callable with CUDA tensors (positionally, or by name when example_args is a dict)
that returns a tensor, or a tuple, list or dict of tensors."""


class CaptureError(RuntimeError):
    """The function cannot be recorded into a CUDA graph, or a call does not fit the capture."""


class HostSyncError(CaptureError):
    """The function waits for the GPU on the host (.item(), .cpu(), data dependent shapes...)."""


class InputMutationError(CaptureError):
    """The function writes to its inputs; a replay would write to the static copies instead."""


class BucketError(CaptureError, ValueError):
    """A call's shapes, dtypes or devices are outside what was captured."""


class CaptureMismatchError(CaptureError, AssertionError):
    """A replay is not bitwise equal to the eager run on the same inputs."""


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same shape, dtype and bytes (NaN equals NaN with the same payload, -0.0 differs from 0.0)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.numel() == 0:
        return True
    return torch.equal(a.detach().reshape(-1).view(torch.uint8),
                       b.detach().reshape(-1).view(torch.uint8))


# ----------------------------------------------------------------------------- output trees


def _flatten(out) -> Tuple[List[torch.Tensor], tuple]:
    if isinstance(out, torch.Tensor):
        return [out], ("tensor",)
    if isinstance(out, (tuple, list)):
        _check_leaves(out)
        return list(out), ("seq", type(out), len(out))
    if isinstance(out, dict):
        _check_leaves(out.values())
        keys = list(out)
        return [out[k] for k in keys], ("dict", keys)
    raise CaptureError(f"the function returned {type(out).__name__}; a captured function must "
                       f"return a tensor, or a tuple, list or dict of tensors")


def _check_leaves(xs):
    for x in xs:
        if not isinstance(x, torch.Tensor):
            raise CaptureError(f"the function returned a {type(x).__name__} inside its outputs; "
                               f"only tensors can be static graph outputs (nested containers and "
                               f"Python numbers are not supported)")


def _unflatten(spec: tuple, flat: List[torch.Tensor]):
    if spec[0] == "tensor":
        return flat[0]
    if spec[0] == "seq":
        return spec[1](flat)
    return dict(zip(spec[1], flat))


# ----------------------------------------------------------------------------- capture


@dataclass
class _Entry:
    """One captured graph: its static buffers and what it cost."""
    size: Optional[int]                 # bucket size of the dynamic dim, None when fully static
    graph: torch.cuda.CUDAGraph
    static_in: List[torch.Tensor]
    static_out: List[torch.Tensor]
    out_spec: tuple
    out_slice: List[bool]               # pad mode: output is sliced along out_dim to the call size
    allocated_bytes: int                # live bytes held by this graph (static inputs and outputs)
    reserved_bytes: int                 # growth of the caching allocator's reservation
    checked_sizes: set = field(default_factory=set)
    last_size: Optional[int] = None


class CapturedFunction:
    """Result of `capture`. Call it like the original function."""

    def __init__(self, fn: Executor, example_args, dynamic: Optional[Dict] = None,
                 buckets: Optional[Sequence[int]] = None, pad: bool = False, out_dim: int = 0,
                 outputs: str = "copy", warmup: int = 2, check: bool = True,
                 share_pool: bool = True, static: Sequence[Union[int, str]] = ()):
        if not torch.cuda.is_available():
            raise CaptureError("CUDA graphs need a CUDA device; none is available")
        if outputs not in ("copy", "static"):
            raise ValueError("outputs must be 'copy' (fresh tensors) or 'static' (graph buffers, "
                             "overwritten by the next call)")
        if warmup < 2:
            raise ValueError("warmup must be at least 2: the first run may compile and verify on "
                             "the host, the second is checked for host syncs")
        self.fn = fn
        self.outputs = outputs
        self.warmup = warmup
        self.check = check
        self.pad = pad
        self.out_dim = out_dim

        if isinstance(example_args, dict):
            self.names: Optional[List[str]] = list(example_args)
            ex = list(example_args.values())
        elif isinstance(example_args, torch.Tensor):
            self.names, ex = None, [example_args]
        else:
            self.names, ex = None, list(example_args)
        if not ex:
            raise CaptureError("capture needs at least one example input tensor")
        for i, t in enumerate(ex):
            label = self._label(i)
            if not isinstance(t, torch.Tensor):
                raise CaptureError(f"input {label} is a {type(t).__name__}; only tensors can be "
                                   f"graph inputs. Bake constants into the function "
                                   f"(functools.partial or a closure) before capturing")
            if t.device.type != "cuda":
                raise CaptureError(f"input {label} is on {t.device}; a CUDA graph records GPU work "
                                   f"only. Move it to cuda first")
            if t.requires_grad:
                raise CaptureError(f"input {label} requires grad; capture records inference "
                                   f"launches only. Pass a detached tensor")
        devs = {t.device for t in ex}
        if len(devs) > 1:
            raise CaptureError(f"inputs are on several devices ({', '.join(map(str, devs))})")
        self.device = ex[0].device
        self._example = [(tuple(t.shape), t.stride(), t.dtype) for t in ex]

        # dynamic dims: {position: dim}, all sharing one size (batch or tokens)
        self.dyn: Dict[int, int] = {}
        if dynamic:
            for k, d in dynamic.items():
                p = self._position(k)
                nd = len(self._example[p][0])
                if not -nd <= d < nd:
                    raise ValueError(f"dynamic dim {d} is out of range for input {self._label(p)} "
                                     f"with {nd} dims")
                self.dyn[p] = d % nd
            sizes = {self._example[p][0][d] for p, d in self.dyn.items()}
            if len(sizes) > 1:
                raise ValueError(f"dynamic dims must share one size in the example, got {sizes}")
            if not buckets:
                raise ValueError("dynamic dims need buckets: the sizes to capture one graph each")
            self.buckets = sorted(set(int(b) for b in buckets))
            if self.buckets[0] < 1:
                raise ValueError("bucket sizes must be positive")
        else:
            if buckets:
                raise ValueError("buckets given without dynamic dims; say which dims vary with "
                                 "dynamic={input: dim}")
            if pad:
                raise ValueError("pad=True needs dynamic dims and buckets")
            self.buckets = []
        self.bound = set()
        for k in static:
            p = self._position(k)
            if p in self.dyn:
                raise ValueError(f"input {self._label(p)} cannot be both dynamic and static")
            self.bound.add(p)
        self._bound_tensors = {p: ex[p] for p in self.bound}
        self._dyn_items = list(self.dyn.items())
        self._first_dyn = self._dyn_items[0] if self._dyn_items else None
        self._pool = torch.cuda.graph_pool_handle() if share_pool else None
        self._entries: Dict[Optional[int], _Entry] = {}
        self._n_in = len(ex)
        # the example inputs are used for the first capture, so a bad function fails here
        self._entry_for(ex)

    # ------------------------------------------------------------------ helpers

    def _label(self, i: int) -> str:
        return repr(self.names[i]) if self.names else f"#{i}"

    def _position(self, k) -> int:
        if isinstance(k, str):
            if not self.names or k not in self.names:
                raise ValueError(f"dynamic names input {k!r}, which is not an input name")
            return self.names.index(k)
        if not 0 <= k < len(self._example):
            raise ValueError(f"dynamic names input #{k}; there are {len(self._example)} inputs")
        return k

    def _run(self, vals: List[torch.Tensor]):
        if self.names is not None:
            return self.fn(**dict(zip(self.names, vals)))
        return self.fn(*vals)

    def _collect(self, args, kwargs) -> List[torch.Tensor]:
        if self.names is None:
            if kwargs or len(args) != self._n_in:
                raise TypeError(f"captured function takes {self._n_in} positional tensors")
            return list(args)
        if args:
            if len(args) > self._n_in:
                raise TypeError(f"captured function takes {self._n_in} inputs")
            got = dict(zip(self.names, args))
            got.update(kwargs)
        else:
            got = kwargs
        try:
            return [got[n] for n in self.names]
        except KeyError as e:
            raise TypeError(f"captured function is missing input {e.args[0]!r}") from None

    def _call_size(self, vals) -> Optional[int]:
        if self._first_dyn is None:
            return None
        p0, d0 = self._first_dyn
        n = vals[p0].shape[d0]
        for p, d in self._dyn_items:
            if vals[p].shape[d] != n:
                raise BucketError(f"dynamic dims disagree: input {self._label(p0)} dim {d0} is "
                                  f"{n}, input {self._label(p)} dim {d} is {vals[p].shape[d]}")
        return n

    def _bucket(self, n: Optional[int]) -> Optional[int]:
        if n is None:
            return None
        if self.pad:
            i = bisect.bisect_left(self.buckets, n)
            if i == len(self.buckets) or n < 1:
                raise BucketError(f"dynamic size {n} is outside the buckets {self.buckets}; the "
                                  f"largest bucket is {self.buckets[-1]}. Add a bucket or run the "
                                  f"function eagerly")
            return self.buckets[i]
        if n not in self.buckets:
            raise BucketError(f"dynamic size {n} is not one of the captured buckets "
                              f"{self.buckets}. Add it, or use pad=True to round up to the next "
                              f"bucket")
        return n

    def _static_shape(self, i: int, size: Optional[int]) -> tuple:
        shape = list(self._example[i][0])
        if i in self.dyn:
            shape[self.dyn[i]] = size
        return tuple(shape)

    # ------------------------------------------------------------------ capture one bucket

    def _entry_for(self, vals) -> _Entry:
        n = self._call_size(vals)
        size = self._bucket(n)
        e = self._entries.get(size)
        if e is None:
            e = self._capture(size, vals, n)
            self._entries[size] = e
        return e

    def _capture(self, size: Optional[int], vals, n: Optional[int]) -> _Entry:
        dev = self.device
        torch.cuda.synchronize(dev)
        alloc0 = torch.cuda.memory_allocated(dev)
        res0 = torch.cuda.memory_reserved(dev)
        static_in = []
        for i, (shape, stride, dtype) in enumerate(self._example):
            if i in self.bound:
                s = self._bound_tensors[i]
            elif size is None:
                s = torch.empty_strided(shape, stride, dtype=dtype, device=dev)
            else:
                s = torch.zeros(self._static_shape(i, size), dtype=dtype, device=dev)
            static_in.append(s)
        e = _Entry(size, None, static_in, [], (), [], 0, 0)
        self._validate(e, vals, n)
        self._load(e, vals, n)

        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            self._first_run_checked(static_in)  # may compile and verify, may sync
            prev = torch.cuda.get_sync_debug_mode()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # "prototype feature" notice
                torch.cuda.set_sync_debug_mode("error")
            try:
                for _ in range(self.warmup - 1):
                    out = self._run(static_in)
            except RuntimeError as err:
                if "synchroniz" in str(err).lower():
                    raise HostSyncError(
                        f"the function synchronizes with the GPU on the host ({err}). Graph "
                        f"replay cannot run host code: remove .item(), .cpu(), printing of CUDA "
                        f"values and data dependent shapes (nonzero, boolean masks). With "
                        f"Truestar kernels, TRUESTAR_VERIFY=all also checks every call on the "
                        f"host; use the default 'first'") from err
                raise
            finally:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.current_stream(dev).wait_stream(side)
        eager_flat, _ = _flatten(out)
        for t in eager_flat:
            if t.device != dev:
                raise CaptureError(f"the function returned a tensor on {t.device}; graph outputs "
                                   f"must stay on {dev}")

        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g, pool=self._pool):
                gout = self._run(static_in)
        except Exception as err:
            raise CaptureError(f"recording the function into a CUDA graph failed: {err}. Common "
                               f"causes: host syncs, CPU tensors created inside the function, "
                               f"calls into libraries that do not support capture") from err
        static_out, spec = _flatten(gout)
        e.graph, e.static_out, e.out_spec = g, static_out, spec
        if self.pad:
            e.out_slice = [len(t.shape) > self.out_dim and t.shape[self.out_dim] == size
                           for t in static_out]
        else:
            e.out_slice = [False] * len(static_out)
        g.replay()
        for k, (a, b) in enumerate(zip(static_out, eager_flat)):
            if not bitwise_equal(a, b):
                raise CaptureMismatchError(
                    f"output {k} of the replay differs from the eager run on the same inputs. The "
                    f"function's result depends on something the graph froze (host state, "
                    f"Python randomness, a pointer kept from a previous call)")
        if n is not None and self.pad:
            e.checked_sizes.add(size)  # the capture itself checked the padded size
        del out, eager_flat
        torch.cuda.synchronize(dev)
        e.allocated_bytes = torch.cuda.memory_allocated(dev) - alloc0
        e.reserved_bytes = torch.cuda.memory_reserved(dev) - res0
        return e

    def _first_run_checked(self, static_in):
        """First eager run; refuse functions that write to their inputs."""
        before = [s.clone() for s in static_in]
        self._run(static_in)
        for i, (s, b) in enumerate(zip(static_in, before)):
            if not bitwise_equal(s, b):
                raise InputMutationError(
                    f"the function writes to input {self._label(i)}. A replay would write to "
                    f"the graph's static copy, not to the caller's tensor. Return the new "
                    f"value as an output instead")

    # ------------------------------------------------------------------ replay

    def _validate(self, e: _Entry, vals, n):
        for i, (v, s) in enumerate(zip(vals, e.static_in)):
            if not isinstance(v, torch.Tensor):
                raise BucketError(f"input {self._label(i)} is a {type(v).__name__}, not a tensor")
            if v.dtype != s.dtype or v.device != s.device:
                raise BucketError(f"input {self._label(i)} is {v.dtype} on {v.device}; the graph "
                                  f"was captured for {s.dtype} on {s.device}")
            if i in self.bound:
                if v.data_ptr() != s.data_ptr() or v.shape != s.shape or v.stride() != s.stride():
                    raise BucketError(f"input {self._label(i)} is bound by address (static=...), "
                                      f"but the call passed a different tensor. Pass the same "
                                      f"tensor and update it in place, or capture without static")
                continue
            if v.shape != s.shape:
                want = list(s.shape)
                if self.pad and i in self.dyn:
                    d = self.dyn[i]
                    want_n = list(want)
                    want_n[d] = n
                    if list(v.shape) == want_n:
                        continue
                    want = want_n
                raise BucketError(f"input {self._label(i)} has shape {list(v.shape)}; the graph "
                                  f"was captured for {want}. Only the declared dynamic dims may "
                                  f"change")

    def _load(self, e: _Entry, vals, n):
        """Copy the call's inputs into the static buffers, all in one launch when there are
        several. Skipped for the static buffers themselves and for inputs bound by address."""
        dst, src = [], []
        padded = self.pad and n != e.size
        for i, (v, s) in enumerate(zip(vals, e.static_in)):
            if v is s or i in self.bound:
                continue
            if padded and i in self.dyn:
                d = self.dyn[i]
                if e.last_size != n:
                    s.narrow(d, n, e.size - n).zero_()
                s = s.narrow(d, 0, n)
            dst.append(s)
            src.append(v)
        if len(dst) == 1:
            dst[0].copy_(src[0])
        elif dst:
            torch._foreach_copy_(dst, src)
        e.last_size = n

    def __call__(self, *args, **kwargs):
        vals = self._collect(args, kwargs)
        n = self._call_size(vals)
        e = self._entries.get(self._bucket(n))
        if e is None:
            e = self._entry_for(vals)
        self._validate(e, vals, n)
        self._load(e, vals, n)
        e.graph.replay()
        outs = e.static_out
        if self.pad and n != e.size:
            d = self.out_dim
            outs = [t.narrow(d, 0, n) if sl else t for t, sl in zip(outs, e.out_slice)]
        if self.outputs == "copy":
            if len(outs) == 1:
                outs = [outs[0].clone()]
            else:
                fresh = [torch.empty_like(t) for t in outs]
                torch._foreach_copy_(fresh, outs)
                outs = fresh
        if self.pad and self.check and n not in e.checked_sizes:
            self._check_padded(e, vals, n, outs)
        return _unflatten(e.out_spec, outs)

    def _check_padded(self, e: _Entry, vals, n, outs):
        """First call at a padded size: compare with the eager run on the unpadded inputs."""
        ref, _ = _flatten(self._run(list(vals)))
        for k, (a, b) in enumerate(zip(outs, ref)):
            if not bitwise_equal(a, b):
                raise CaptureMismatchError(
                    f"output {k} at dynamic size {n}, padded to bucket {e.size}, differs from the "
                    f"eager run at size {n}. The function is not independent along the padded "
                    f"dim (a reduction or normalization over it, or a block size that depends on "
                    f"it). Use pad=False with exact buckets")
        e.checked_sizes.add(n)

    # ------------------------------------------------------------------ introspection

    def capture_all(self):
        """Capture every bucket now, largest first, instead of lazily on first use."""
        if not self.buckets:
            return self
        for size in reversed(self.buckets):
            if size in self._entries:
                continue
            vals = [self._bound_tensors[i] if i in self.bound else
                    torch.zeros(self._static_shape(i, size), dtype=dt, device=self.device)
                    for i, (_, _, dt) in enumerate(self._example)]
            self._entries[size] = self._capture(size, vals, size)
        return self

    @property
    def num_graphs(self) -> int:
        return len(self._entries)

    def static_inputs(self, size: Optional[int] = None) -> List[torch.Tensor]:
        """The static input buffers of a bucket. Writing new values into them and passing them
        back skips the input copies."""
        return self._entries[size].static_in

    def memory(self) -> Dict[Optional[int], Dict[str, int]]:
        """Per bucket: bytes held by static buffers and growth of the allocator reservation."""
        return {k: {"allocated": e.allocated_bytes, "reserved": e.reserved_bytes}
                for k, e in self._entries.items()}

    def describe(self) -> str:
        lines = []
        for k, e in sorted(self._entries.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
            lines.append(f"bucket {k if k is not None else 'static'}: {len(e.static_in)} inputs, "
                         f"{len(e.static_out)} outputs, {e.allocated_bytes / 2**20:.2f} MiB held, "
                         f"{e.reserved_bytes / 2**20:.2f} MiB reserved")
        return "\n".join(lines)


def capture(fn: Executor, example_args, *, dynamic: Optional[Dict[Union[int, str], int]] = None,
            buckets: Optional[Sequence[int]] = None, pad: bool = False, out_dim: int = 0,
            outputs: str = "copy", warmup: int = 2, check: bool = True,
            share_pool: bool = True, static: Sequence[Union[int, str]] = ()
            ) -> CapturedFunction:
    """Record fn's kernel launches into CUDA graphs and return a function that replays them.

    example_args: a tensor, a tuple or list of tensors (passed positionally), or a dict of
        tensors (passed by name). All must be CUDA tensors on one device.
    dynamic: {input position or name: dim} for dims whose size changes between calls. All of
        them share one size per call (batch or tokens). Needs buckets.
    buckets: sizes of the dynamic dims. One graph is captured per bucket on first use (or all
        at once with `.capture_all()`). A call at a size that is not a bucket is refused, unless
        pad=True.
    pad: round the dynamic size up to the next bucket, zero fill the rest and slice outputs along
        out_dim. Only valid when rows along that dim are computed independently; the first call at
        each new size is compared bitwise with the eager run and refused if it differs (check).
    outputs: 'copy' returns fresh tensors (one copy kernel per output); 'static' returns the
        graph's own buffers, which the next call of this function (any bucket) overwrites.
    share_pool: buckets share one memory pool for intermediates (sequential use on one stream).
    static: inputs bound by address (weights, caches): never copied; every call must pass the
        same tensor, whose values may change in place between calls.

    Raises HostSyncError, InputMutationError, CaptureError or CaptureMismatchError when fn cannot
    be captured faithfully, and BucketError when a later call does not fit what was captured.
    """
    return CapturedFunction(fn, example_args, dynamic=dynamic, buckets=buckets, pad=pad,
                            out_dim=out_dim, outputs=outputs, warmup=warmup, check=check,
                            share_pool=share_pool, static=static)
