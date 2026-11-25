"""Run a GGUF model in llama.cpp and read every intermediate tensor, through its C API.

The engine is driven through the ctypes bindings that the `llama-cpp-python` wheel ships (its
`llama.dll` / `libllama.so` and the ggml libraries next to it). Intermediate tensors come from the
scheduler's evaluation callback (`cb_eval` in the context parameters): ggml calls it for each graph
node, first asking whether the node is wanted, then with the node computed. Each wanted node is
copied out with `ggml_backend_tensor_get`, which works whatever backend holds it.

The callback forces the scheduler to stop after every wanted node, which can change fusion on GPU
backends. So the verdict is never taken from a tapped run alone: `run(..., tap=False)` gives the
untouched engine's logits and `observer_effect` compares the two (never introduce item 2).
"""

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

GGML_MAX_DIMS = 4
GGML_MAX_SRC = 10
GGML_MAX_NAME = 64
GGML_MAX_OP_PARAMS = 64


class _Tensor(ctypes.Structure):
    """`struct ggml_tensor` as declared in ggml.h; checked against the library's own accessors on
    first use (`_layout_ok`), so a changed layout is refused instead of misread."""
    _fields_ = [
        ("type", ctypes.c_int),
        ("buffer", ctypes.c_void_p),
        ("ne", ctypes.c_int64 * GGML_MAX_DIMS),
        ("nb", ctypes.c_size_t * GGML_MAX_DIMS),
        ("op", ctypes.c_int),
        ("op_params", ctypes.c_int32 * (GGML_MAX_OP_PARAMS // 4)),
        ("flags", ctypes.c_int32),
        ("src", ctypes.c_void_p * GGML_MAX_SRC),
        ("view_src", ctypes.c_void_p),
        ("view_offs", ctypes.c_size_t),
        ("data", ctypes.c_void_p),
        ("name", ctypes.c_char * GGML_MAX_NAME),
        ("extra", ctypes.c_void_p),
        ("padding", ctypes.c_char * 8),
    ]


class EngineError(RuntimeError):
    """llama.cpp is missing, failed to load the model, or its tensors could not be read."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {}


def _libs():
    if "llama" in _STATE:
        return _STATE["llama"], _STATE["ggml"]
    try:
        import llama_cpp  # the ctypes bindings and the shared libraries
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise EngineError("TS647", f"llama.cpp is not importable ({type(exc).__name__}: {exc}); "
                                   "pip install llama-cpp-python (a CPU wheel needs no compiler)") from exc
    libdir = os.path.join(os.path.dirname(llama_cpp.__file__), "lib")
    name = {"nt": "ggml-base.dll"}.get(os.name, "libggml-base.so")
    path = os.path.join(libdir, name)
    if not os.path.exists(path):
        alt = [f for f in os.listdir(libdir) if f.startswith(("ggml-base", "libggml-base"))] if os.path.isdir(libdir) else []
        if not alt:
            raise EngineError("TS647", f"ggml base library not found next to llama_cpp ({libdir})")
        path = os.path.join(libdir, alt[0])
    g = ctypes.CDLL(path)
    vp = ctypes.c_void_p
    for fn, res, args in [("ggml_get_name", ctypes.c_char_p, [vp]), ("ggml_nbytes", ctypes.c_size_t, [vp]),
                          ("ggml_nelements", ctypes.c_int64, [vp]), ("ggml_is_contiguous", ctypes.c_bool, [vp]),
                          ("ggml_op_desc", ctypes.c_char_p, [vp]), ("ggml_type_name", ctypes.c_char_p, [ctypes.c_int]),
                          ("ggml_backend_tensor_get", None, [vp, vp, ctypes.c_size_t, ctypes.c_size_t])]:
        f = getattr(g, fn)
        f.restype = res
        f.argtypes = args
    llama_cpp.llama_backend_init()
    _STATE["llama"], _STATE["ggml"] = llama_cpp, g
    # keep the engine quiet; its messages are collected and shown on failure
    _STATE["log"] = []

    @llama_cpp.llama_log_callback
    def _log(level, text, user):
        try:
            _STATE["log"].append(text.decode("utf-8", errors="replace"))
            del _STATE["log"][:-200]
        except Exception:
            pass
    _STATE["log_cb"] = _log
    llama_cpp.llama_log_set(_log, ctypes.c_void_p(0))
    return llama_cpp, g


def available() -> Optional[str]:
    """None when llama.cpp can run here, else the reason."""
    try:
        _libs()
    except EngineError as exc:
        return str(exc)
    return None


def engine_version() -> str:
    try:
        from importlib.metadata import version

        return f"llama-cpp-python {version('llama-cpp-python')}"
    except Exception:
        return "unknown"


def devices() -> List[str]:
    """Backend device names ggml reports (for example CPU, CUDA0, Vulkan0)."""
    L, _ = _libs()
    out = []
    try:
        g = _STATE["ggml"]
        g.ggml_backend_dev_count.restype = ctypes.c_size_t
        g.ggml_backend_dev_get.restype = ctypes.c_void_p
        g.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
        g.ggml_backend_dev_name.restype = ctypes.c_char_p
        g.ggml_backend_dev_name.argtypes = [ctypes.c_void_p]
        for i in range(g.ggml_backend_dev_count()):
            out.append(g.ggml_backend_dev_name(g.ggml_backend_dev_get(i)).decode())
    except AttributeError:
        out = ["CPU"]
    return out


@dataclass
class Tap:
    """One computed graph node."""
    index: int
    name: str
    op: str
    type_name: str
    ne: Tuple[int, ...]
    op_params: Tuple[int, ...]
    data: Optional[np.ndarray] = None  # float64, torch layout (reversed ne), None if not readable

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(x) for x in reversed(self.ne))

    def op_float(self, i: int) -> float:
        return float(np.array([self.op_params[i]], dtype=np.int32).view(np.float32)[0])


@dataclass
class EngineRun:
    logits: np.ndarray  # [T, vocab] float64 (from float32)
    taps: List[Tap] = field(default_factory=list)
    seconds: float = 0.0
    settings: Dict[str, Any] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)

    def named(self, name: str) -> List[Tap]:
        return [t for t in self.taps if t.name == name]


GGML_TYPE = {"f32": 0, "f16": 1}


def run(path: str, ids: Sequence[int], *, threads: int = 4, kv_type: str = "f32", gpu_layers: int = 0,
        tap: bool = True, mutate: Optional[Callable[[Tap, np.ndarray], Optional[np.ndarray]]] = None,
        ) -> EngineRun:
    """One teacher forced pass: every token of `ids` in one batch, logits for every position.

    `mutate(tap, array)` is a test hook: it may return a changed array, which is written back into
    the engine's node so every later node computes from it (an injected engine fault). It works
    only for host memory (the CPU backend)."""
    import time

    L, g = _libs()
    ids = [int(i) for i in ids]
    T = len(ids)
    with _LOCK:
        _STATE["log"].clear()
        mp = L.llama_model_default_params()
        mp.n_gpu_layers = int(gpu_layers)
        model = L.llama_model_load_from_file(path.encode("utf-8"), mp)
        if not model:
            raise EngineError("TS647", f"llama.cpp could not load {path}: " + "".join(_STATE["log"][-8:]).strip())
        ctx = None
        batch = None
        taps: List[Tap] = []
        failure: List[str] = []
        try:
            cp = L.llama_context_default_params()
            n_ctx = max(64, -(-T // 64) * 64)
            cp.n_ctx = n_ctx
            cp.n_batch = n_ctx
            cp.n_ubatch = n_ctx
            cp.n_threads = threads
            cp.n_threads_batch = threads
            cp.type_k = GGML_TYPE[kv_type]
            cp.type_v = GGML_TYPE[kv_type]
            cp.flash_attn_type = 0  # disabled: the attention scores stay separate graph nodes
            cp.no_perf = True
            if tap:
                checked = []

                def _cb(t, ask, user):
                    if ask:
                        return True
                    try:
                        _record(g, t, taps, checked, mutate)
                    except Exception as exc:  # never let an exception cross into C
                        failure.append(f"{type(exc).__name__}: {exc}")
                        return False
                    return True
                cb = L.ggml_backend_sched_eval_callback(_cb)
                cp.cb_eval = cb
                cp.cb_eval_user_data = None
            ctx = L.llama_init_from_model(model, cp)
            if not ctx:
                raise EngineError("TS647", "llama.cpp could not create a context: " + "".join(_STATE["log"][-8:]).strip())
            batch = L.llama_batch_init(T, 0, 1)
            for i, tok in enumerate(ids):
                batch.token[i] = tok
                batch.pos[i] = i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = 1
            batch.n_tokens = T
            t0 = time.perf_counter()
            rc = L.llama_decode(ctx, batch)
            secs = time.perf_counter() - t0
            if failure:
                raise EngineError("TS648", "reading an engine tensor failed: " + failure[0])
            if rc != 0:
                raise EngineError("TS647", f"llama_decode returned {rc}: " + "".join(_STATE["log"][-8:]).strip())
            vocab = L.llama_vocab_n_tokens(L.llama_model_get_vocab(model))
            ptr = L.llama_get_logits(ctx)
            logits = np.ctypeslib.as_array(ptr, shape=(T * vocab,)).astype(np.float64).reshape(T, vocab)
            settings = {"threads": threads, "kv_type": kv_type, "gpu_layers": gpu_layers,
                        "flash_attention": False, "n_ctx": n_ctx, "batch": T}
            return EngineRun(logits=logits, taps=taps, seconds=secs, settings=settings, log=list(_STATE["log"]))
        finally:
            if batch is not None:
                L.llama_batch_free(batch)
            if ctx:
                L.llama_free(ctx)
            L.llama_model_free(model)


def _record(g, t, taps: List[Tap], checked: List[bool], mutate) -> None:
    s = _Tensor.from_address(t)
    name = g.ggml_get_name(t).decode("utf-8", errors="replace")
    if not checked:
        n = 1
        for x in s.ne:
            n *= int(x)
        tn = g.ggml_type_name(s.type)
        if s.name.decode("utf-8", errors="replace") != name or n != g.ggml_nelements(t) or not tn:
            raise EngineError("TS648", "struct ggml_tensor in this llama.cpp build has another layout than "
                                       "Truestar reads; tensors cannot be tapped")
        checked.append(True)
    tname = g.ggml_type_name(s.type).decode()
    ne = tuple(int(x) for x in s.ne)
    op = g.ggml_op_desc(t).decode()
    tp = Tap(len(taps), name, op, tname, ne, tuple(int(x) for x in s.op_params))
    if tname == "f32" and g.ggml_is_contiguous(t):
        nbytes = g.ggml_nbytes(t)
        buf = np.empty(nbytes // 4, dtype=np.float32)
        g.ggml_backend_tensor_get(t, buf.ctypes.data, 0, nbytes)
        arr = buf.reshape(tuple(reversed(ne)))
        if mutate is not None:
            new = mutate(tp, arr.astype(np.float64))
            if new is not None:
                if not s.data:
                    raise EngineError("TS648", f"cannot inject into {name}: not in host memory")
                src = np.ascontiguousarray(np.asarray(new, dtype=np.float32).reshape(arr.shape))
                ctypes.memmove(s.data, src.ctypes.data, nbytes)
                arr = src
        tp.data = arr.astype(np.float64)
    taps.append(tp)


def quantize(src: str, dst: str, ftype: str, threads: int = 4) -> None:
    """The engine's own quantizer (`llama_model_quantize`), for example ftype "Q8_0" or "Q4_K_M"."""
    L, _ = _libs()
    code = getattr(L, f"LLAMA_FTYPE_MOSTLY_{ftype}", None)
    if code is None:
        raise EngineError("TS651", f"llama.cpp has no quantization type {ftype}")
    p = L.llama_model_quantize_default_params()
    p.nthread = threads
    p.ftype = code
    with _LOCK:
        rc = L.llama_model_quantize(src.encode("utf-8"), dst.encode("utf-8"), ctypes.byref(p))
    if rc != 0:
        raise EngineError("TS647", f"llama_model_quantize returned {rc}: " + "".join(_STATE["log"][-8:]).strip())


def tokenize(path: str, text: str, add_special: bool = True) -> List[int]:
    """The engine's own tokenization of `text` with the vocabulary in the GGUF."""
    L, _ = _libs()
    with _LOCK:
        mp = L.llama_model_default_params()
        mp.vocab_only = True
        model = L.llama_model_load_from_file(path.encode("utf-8"), mp)
        if not model:
            raise EngineError("TS647", f"llama.cpp could not load the vocabulary of {path}")
        try:
            vocab = L.llama_model_get_vocab(model)
            b = text.encode("utf-8")
            n = len(b) + 8
            out = (L.llama_token * n)()
            k = L.llama_tokenize(vocab, b, len(b), out, n, add_special, False)
            if k < 0:
                raise EngineError("TS647", "llama_tokenize needed more room than expected")
            return [int(out[i]) for i in range(k)]
        finally:
            L.llama_model_free(model)
