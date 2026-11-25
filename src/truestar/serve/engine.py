"""Continuous batching engine of `truestar serve`: KV blocks, prefix cache, scheduler, keyed sampler.

The engine runs one step program (serve/model.py) in which every token is a row. Each step it takes
one row for every running decode and fills the rest of its token budget with prefill rows (whole
prompts, or chunks of them with chunked prefill), pads the row count up to a power of two with rows
that write to scratch slots, and calls the program once. A row's arithmetic does not depend on the
other rows, so a request gets the same bits whatever it is batched with, whether its prefix came from
the cache, and however its prompt was chunked. `check_invariance` (serve/check.py) verifies exactly
that, bit for bit.

Sampling is keyed: the uniform draw for a request's i-th generated token is
``uniform01(seed, i, 0)`` (the IR's ``multinomial`` scheme, docs/LLM.md), so it depends on the
request's seed and the token index only, never on the batch slot, the step count or the chip.
"""

from __future__ import annotations

import hashlib
import itertools
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..ir.reference import uniform01
from ..kernel import verify as KV
from .model import ServeConfig, TinyLM, import_token_step


class ServeError(ValueError):
    """A request the engine refuses; ``code`` is a TS14xx code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# ----------------------------------------------------------------------------- sampling


def logprobs_f64(logits: np.ndarray) -> np.ndarray:
    """log softmax in float64 with a fixed summation order (a sequential cumulative sum), so the same
    logits give the same bits on any host."""
    x = logits.astype(np.float64)
    m = x.max()
    z = np.exp(x - m)
    return x - m - np.log(np.cumsum(z)[-1])


def sample(logits: np.ndarray, temperature: float, top_p: float, seed: int, index: int) -> int:
    """Greedy (temperature 0: the first index of the largest logit) or keyed inverse CDF sampling."""
    if temperature <= 0.0:
        return int(np.argmax(logits))
    x = logits.astype(np.float64) / temperature
    w = np.exp(x - x.max())
    if top_p < 1.0:
        order = np.argsort(-w, kind="stable")
        cum = np.cumsum(w[order])
        keep = cum - w[order] < top_p * cum[-1]        # tokens before the one that crosses p, and it
        mask = np.zeros_like(w, dtype=bool)
        mask[order[keep]] = True
        w = np.where(mask, w, 0.0)
    cum = np.cumsum(w)
    u = uniform01(seed, index, 0) * cum[-1]
    k = int(np.searchsorted(cum, u, side="right"))
    last = int(np.flatnonzero(w > 0)[-1])
    return min(k, last)


# ----------------------------------------------------------------------------- byte tokenizer


def encode(text: str) -> List[int]:
    """UTF-8 bytes are tokens 0..255 (no tokenizer download; ids above 255 are the model's own)."""
    return list(text.encode("utf-8"))


def text_of(tokens: Sequence[int]) -> bytes:
    """The bytes of tokens below 256; other ids are written as <|id|>."""
    out = bytearray()
    for t in tokens:
        out += bytes([t]) if t < 256 else f"<|{t}|>".encode()
    return bytes(out)


# ----------------------------------------------------------------------------- requests


@dataclass
class Request:
    prompt: List[int]
    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    stop_ids: Tuple[int, ...] = ()
    stop_bytes: Tuple[bytes, ...] = ()   # stop strings, on the byte text of the generated tokens
    top_logprobs: int = 0
    rid: str = ""


@dataclass
class Seq:
    req: Request
    blocks: List[int] = field(default_factory=list)
    tokens: List[int] = field(default_factory=list)     # prompt then generated
    computed: int = 0                                   # positions whose K and V are in the pool
    cached: int = 0                                     # positions taken from the prefix cache
    out: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    top: List[List[Tuple[int, float]]] = field(default_factory=list)
    finish: Optional[str] = None
    done: threading.Event = field(default_factory=threading.Event)
    listeners: List[Callable] = field(default_factory=list)
    hashes: List[bytes] = field(default_factory=list)   # chained hash of every full block registered

    @property
    def prefilling(self) -> bool:
        return self.computed < len(self.req.prompt)


# ----------------------------------------------------------------------------- KV blocks and the prefix cache


class Blocks:
    """Block allocator with a prefix cache: a full block is keyed by the chained hash of all tokens up
    to its end. A block held by no sequence stays readable until it is reallocated (least recently
    released first)."""

    def __init__(self, n_blocks: int, prefix_cache: bool):
        self.free = deque(range(n_blocks))
        self.ref = [0] * n_blocks
        self.by_hash: Dict[bytes, int] = {}
        self.hash_of: Dict[int, bytes] = {}
        self.idle: "OrderedDict[int, None]" = OrderedDict()   # cached blocks with no holder
        self.prefix_cache = prefix_cache
        self.hits = 0

    def available(self) -> int:
        return len(self.free) + len(self.idle)

    def take(self) -> int:
        if self.free:
            b = self.free.popleft()
        elif self.idle:
            b, _ = self.idle.popitem(last=False)
            h = self.hash_of.pop(b)
            del self.by_hash[h]
        else:
            raise ServeError("TS1403", "the KV block pool is full")
        self.ref[b] = 1
        return b

    def hold(self, b: int) -> None:
        if self.ref[b] == 0:
            self.idle.pop(b, None)
        self.ref[b] += 1

    def release(self, b: int) -> None:
        self.ref[b] -= 1
        if self.ref[b] == 0:
            if b in self.hash_of:
                self.idle[b] = None
            else:
                self.free.append(b)

    def lookup(self, h: bytes) -> Optional[int]:
        return self.by_hash.get(h) if self.prefix_cache else None

    def register(self, h: bytes, b: int) -> None:
        if self.prefix_cache and h not in self.by_hash and b not in self.hash_of:
            self.by_hash[h] = b
            self.hash_of[b] = h


def chain(prev: bytes, toks: Sequence[int]) -> bytes:
    return hashlib.blake2b(prev + np.asarray(toks, dtype=np.int64).tobytes(), digest_size=16).digest()


# ----------------------------------------------------------------------------- the step runner


class StepRunner:
    """Calls the token step on one executor: ``cpu`` or ``cuda`` (Truestar compiled, float32,
    deterministic), ``reference`` (the float64 IR interpreter), ``eager`` or ``eager-cuda`` (the same
    module in eager PyTorch float32; a baseline, no promise)."""

    def __init__(self, model: TinyLM, executor: str = "cpu", deterministic: bool = True):
        self.c = model.c
        self.deterministic = deterministic
        self.executor = executor
        self.fallbacks = 0
        if executor in ("cpu", "cuda"):
            from ..compiler import compile_program

            self.prog = import_token_step(model)
            self.compiled = compile_program(self.prog, executor, "float32", deterministic=deterministic)
            self.fallbacks = len(self.compiled.report.fallbacks)
            self.device, self.dtype = executor, torch.float32
        elif executor == "reference":
            self.prog = import_token_step(model)
            self.device, self.dtype = "cpu", torch.float64
        elif executor in ("eager", "eager-cuda"):
            self.device = "cuda" if executor == "eager-cuda" else "cpu"
            self.dtype = torch.float32
            import copy

            self.module = copy.deepcopy(model).float().to(self.device)
        else:
            raise ValueError(f"unknown executor {executor!r}")

    def pools(self, n_slots: int):
        c = self.c
        kp = torch.zeros(c.layers, n_slots, c.kv_heads, c.head_dim, dtype=self.dtype, device=self.device)
        return kp, torch.zeros_like(kp)

    def __call__(self, ids, pos, slots, table, kp, vp):
        args = [t.to(self.device) for t in (ids, pos, slots, table)]
        if self.executor in ("cpu", "cuda"):
            return self.compiled(args + [kp, vp])
        if self.executor == "reference":
            from .. import ir

            return ir.run(self.prog, *args, kp, vp)
        with torch.no_grad():
            return self.module.token_step(*args, kp, vp)


# ----------------------------------------------------------------------------- the engine


def bucket(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


class Engine:
    """``max_batch`` running sequences at most; ``chunk`` prefill tokens per step (None: whole prompts,
    no chunked prefill); ``max_rows`` rows per step (decodes plus prefill)."""

    def __init__(self, model: TinyLM, executor: str = "cpu", *, runner: Optional[StepRunner] = None,
                 max_batch: int = 32, chunk: Optional[int] = None, prefix_cache: bool = True,
                 max_rows: int = 512):
        self.c: ServeConfig = model.c
        self.runner = runner or StepRunner(model, executor)
        self.max_batch, self.chunk, self.max_rows = max_batch, chunk, max_rows
        self.blocks = Blocks(self.c.n_blocks, prefix_cache)
        self.pad_base = self.c.n_blocks * self.c.block        # scratch slots for padding rows
        self.kp, self.vp = self.runner.pools(self.pad_base + max_rows)
        self.waiting: deque = deque()
        self.running: List[Seq] = []
        self.steps = 0
        self.rows_run = 0
        self.lock = threading.Condition()
        self._ids = itertools.count()
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self._batch_width = False

    # -- requests
    def validate(self, r: Request) -> None:
        c = self.c
        if not r.prompt:
            raise ServeError("TS1404", "the prompt is empty")
        if any((not isinstance(t, int)) or t < 0 or t >= c.vocab for t in r.prompt):
            raise ServeError("TS1404", f"a prompt token is outside the vocabulary 0..{c.vocab - 1}")
        if r.max_tokens < 1:
            raise ServeError("TS1404", "max_tokens must be at least 1")
        if len(r.prompt) + r.max_tokens > c.max_len:
            raise ServeError("TS1403", f"prompt ({len(r.prompt)}) plus max_tokens ({r.max_tokens}) exceeds the "
                                       f"model length {c.max_len}")
        if -(-(len(r.prompt) + r.max_tokens) // c.block) > c.n_blocks:
            raise ServeError("TS1403", "the request needs more KV blocks than the pool has")
        if self.chunk is None and len(r.prompt) > self.max_rows:
            raise ServeError("TS1403", f"a prompt of {len(r.prompt)} tokens exceeds max_rows {self.max_rows} "
                                       "without chunked prefill")
        if not (0.0 < r.top_p <= 1.0) or r.temperature < 0:
            raise ServeError("TS1404", "temperature must be >= 0 and top_p in (0, 1]")

    def submit(self, r: Request) -> Seq:
        return self.submit_with(r, None)

    def submit_with(self, r: Request, listener: Optional[Callable]) -> Seq:
        """Submit; ``listener(seq, token)`` is called after every generated token and with None at the end."""
        self.validate(r)
        if not r.rid:
            r.rid = f"cmpl-{next(self._ids)}"
        s = Seq(req=r, tokens=list(r.prompt))
        if listener is not None:
            s.listeners.append(listener)
        with self.lock:
            self.waiting.append(s)
            self.lock.notify_all()
        return s

    # -- scheduling
    def _admit(self) -> None:
        c = self.c
        while self.waiting and len(self.running) < self.max_batch:
            s = self.waiting[0]
            r = s.req
            need = -(-(len(r.prompt) + r.max_tokens) // c.block)
            # prefix cache: full blocks of the prompt, leaving at least one prompt token to compute
            hits, h = [], b""
            for i in range((len(r.prompt) - 1) // c.block):
                h = chain(h, r.prompt[i * c.block:(i + 1) * c.block])
                b = self.blocks.lookup(h)
                if b is None:
                    break
                hits.append((b, h))
            fresh = need - len(hits)
            idle_hits = sum(1 for b, _ in hits if self.blocks.ref[b] == 0)
            if self.blocks.available() - idle_hits < fresh:
                return                                   # wait for blocks; first come first served
            self.waiting.popleft()
            for b, hh in hits:
                self.blocks.hold(b)
                s.blocks.append(b)
                s.hashes.append(hh)
            s.cached = s.computed = len(hits) * c.block
            self.blocks.hits += len(hits)
            for _ in range(fresh):
                s.blocks.append(self.blocks.take())
            self.running.append(s)

    def _plan(self) -> List[Tuple[Seq, int, int]]:
        """(sequence, first position, count) of this step's rows: every decode, then prefills."""
        work, rows = [], 0
        for s in self.running:
            if not s.prefilling:
                work.append((s, len(s.tokens) - 1, 1))
                rows += 1
        budget = self.max_rows - rows
        if self.chunk is not None:
            budget = min(budget, self.chunk)
        for s in self.running:
            if s.prefilling and budget > 0:
                left = len(s.req.prompt) - s.computed
                if self.chunk is None and left > budget:
                    continue                             # whole prompts only; wait for room
                n = min(left, budget)
                work.append((s, s.computed, n))
                budget -= n
                rows += n
        return work

    def step(self) -> int:
        """Run one step; returns the number of real rows (0: nothing to do)."""
        with self.lock:
            self._admit()
            work = self._plan()
        if not work:
            return 0
        c = self.c
        ids, pos, slots, tables, ends = [], [], [], [], []
        # every table row is c.width blocks wide, so a row attends over the same slots in any batch;
        # _batch_width (tests only) narrows it to the widest sequence of the step, as engines that size
        # attention to the batch do
        width = max(len(s.blocks) for s, _, _ in work) if self._batch_width else c.width
        for s, p0, n in work:
            t = s.blocks + [s.blocks[0]] * (width - len(s.blocks))
            for p in range(p0, p0 + n):
                ids.append(s.tokens[p])
                pos.append(p)
                slots.append(s.blocks[p // c.block] * c.block + p % c.block)
                tables.append(t)
            ends.append(len(ids) - 1)
        real = len(ids)
        nb = bucket(real)
        for i in range(nb - real):                       # padding rows: scratch slots, position 0
            ids.append(0)
            pos.append(0)
            slots.append(self.pad_base + i)
            tables.append([0] * width)
        col = lambda v: torch.tensor(v, dtype=torch.int64).reshape(-1, 1)  # noqa: E731
        logits, self.kp, self.vp = self.runner(col(ids), col(pos), col(slots),
                                               torch.tensor(tables, dtype=torch.int64), self.kp, self.vp)
        lg = KV.to_host_f64(logits[:real]).numpy()
        self.steps += 1
        self.rows_run += real
        finished = []
        for (s, p0, n), e in zip(work, ends):
            s.computed = p0 + n
            self._register_blocks(s)
            if s.computed < len(s.tokens):
                continue                                 # a prefill chunk that does not end the prompt
            r = s.req
            row = lg[e]
            tok = sample(row, r.temperature, r.top_p, r.seed, len(s.out))
            lp = logprobs_f64(row)
            s.out.append(tok)
            s.tokens.append(tok)
            s.logprobs.append(float(lp[tok]))
            if r.top_logprobs:
                order = np.argsort(-lp, kind="stable")[:r.top_logprobs]
                s.top.append([(int(k), float(lp[k])) for k in order])
            if tok in r.stop_ids or (r.stop_bytes and any(text_of(s.out).endswith(b) for b in r.stop_bytes)):
                s.finish = "stop"
            elif len(s.out) >= r.max_tokens:
                s.finish = "length"
            for f in s.listeners:
                f(s, tok)
            if s.finish:
                finished.append(s)
        with self.lock:
            for s in finished:
                self.running.remove(s)
                for b in s.blocks:
                    self.blocks.release(b)
                s.blocks = []
        for s in finished:
            s.done.set()
            for f in s.listeners:
                f(s, None)
        return real

    def _register_blocks(self, s: Seq) -> None:
        c = self.c
        while (len(s.hashes) + 1) * c.block <= s.computed:
            i = len(s.hashes)
            h = chain(s.hashes[-1] if s.hashes else b"", s.tokens[i * c.block:(i + 1) * c.block])
            s.hashes.append(h)
            self.blocks.register(h, s.blocks[i])

    def run_until_idle(self) -> None:
        while self.step() or self.waiting:
            pass

    # -- background loop for the HTTP server
    def start(self) -> None:
        def loop():
            while not self._stop:
                with self.lock:
                    while not self._stop and not self.waiting and not self.running:
                        self.lock.wait(0.5)
                if self._stop:
                    return
                self.step()

        self._thread = threading.Thread(target=loop, name="truestar-serve-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        with self.lock:
            self.lock.notify_all()
        if self._thread:
            self._thread.join(5)

    def warmup(self) -> None:
        """Run every row bucket once (each new row count is a new shape for the compiled program's first
        call checks), then drop what the warm up cached."""
        n = 1
        while n <= self.max_rows:
            ids = torch.zeros(n, 1, dtype=torch.int64)
            slots = (self.pad_base + torch.arange(n)).reshape(n, 1)
            table = torch.zeros(n, self.c.width, dtype=torch.int64)
            _, self.kp, self.vp = self.runner(ids, torch.zeros(n, 1, dtype=torch.int64), slots, table,
                                              self.kp, self.vp)
            n *= 2
