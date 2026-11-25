"""Decode a Llama style model compiled by Truestar with an opt in quantized KV cache.

Run: ``python examples/llama_kvquant_decode.py`` (needs a CUDA GPU)

``make_decode_step(model, kv_cache=None)`` is the dense decode step of examples/llama_paged_decode.py,
unchanged. ``kv_cache="int8"`` or ``"fp8"`` (never automatic) gives a step that keeps, per layer,

* a quantized prefix of whole groups of ``group`` tokens: data int8 (or fp8 as its uint8 bits)
  [L, B, HKV, N, HD] and one float32 scale per head and group [L, B, HKV, N / group], and
* a float tail of the newest tokens [L, B, HKV, group, HD] with its live length ``tlen``.

The step writes the new token's K and V into the tail (``index_copy``) and attends with
``torch.ops.truestar.kvq_attention``, which the importer turns into the IR op ``kvq_attention`` and
the compiled executor runs through truestar.kvquant's split KV kernel (dequantized in registers,
float32 accumulation, fixed order merge, grouped query heads read their KV head). On the host,
``QuantCache.after_step`` quantizes a full tail exactly once, from its float values, and appends it
to the prefix, so no value is ever quantized twice. The compiled program checks the step against the
float64 reference of the IR (attention over the dequantized cache) on its first call, like every
library step. ``layer_error_bounds`` reports truestar.kvquant's proven bound against the
unquantized cache per layer.
"""

from __future__ import annotations

import copy
import os
import sys
from typing import List, Optional

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llama_paged_decode as P  # noqa: E402
from truestar import kvquant as KQ  # noqa: E402,F401  (registers torch.ops.truestar.kvq_attention)

FORMATS = (None, "int8", "fp8", "int4", "nvfp4")


class QuantDecodeStep(nn.Module):
    """One decode token per sequence (T = 1) over a quantized prefix and a float tail."""

    def __init__(self, model: P.Llama, fmt: str, group: int = 32, return_q: bool = False, fixed: bool = False,
                 rot: bool = False):
        super().__init__()
        if fmt not in KQ.FORMATS + KQ.LOW_FORMATS:
            raise KQ.KVQuantError(f"[TS701] kv_cache must be one of {KQ.FORMATS + KQ.LOW_FORMATS} or None, "
                                  f"got {fmt!r}")
        if rot and fmt not in KQ.LOW_FORMATS:
            raise KQ.KVQuantError("[TS701] rot=True is for the 4 bit formats (int4, nvfp4)")
        self.model, self.fmt, self.group, self.return_q, self.fixed = model, fmt, group, return_q, fixed
        self.rot = rot

    def forward(self, ids, pos, kd, ks, vd, vs, tk, tv, tlen, plen=None):
        """``plen`` (with ``fixed``): live keys of a prefix allocated to its capacity, int64 [1]."""
        return self._step(ids, pos, kd, ks, vd, vs, tk, tv, tlen, plen, None, None)

    def _step(self, ids, pos, kd, ks, vd, vs, tk, tv, tlen, plen, kb, vb):
        m = self.model
        cos, sin = m.rope(pos)
        x = m.embed(ids)
        nk, nv, qs = [], [], []
        for i, blk in enumerate(m.blocks):
            q, k, v = m.qkv(blk, x, cos, sin)
            tki = tk[i].index_copy(2, tlen, k)
            tvi = tv[i].index_copy(2, tlen, v)
            a = torch.ops.truestar.kvq_attention(q, kd[i], ks[i], vd[i], vs[i], tki, tvi, tlen + 1,
                                                 self.fmt, self.group, self.group, False, 0.0,
                                                 plen if self.fixed else None,
                                                 None if kb is None else kb[i], None if vb is None else vb[i],
                                                 self.rot)
            x = m.finish(blk, x, a)
            nk.append(tki)
            nv.append(tvi)
            qs.append(q)
        out = (m.head(x), torch.stack(nk), torch.stack(nv))
        return out + (torch.stack(qs),) if self.return_q else out


class LowQuantDecodeStep(QuantDecodeStep):
    """The step over a 4 bit prefix (``int4``, ``nvfp4``, truestar.kvquant.low): two more inputs after
    ``tlen``, the K and V E4M3 block scales as uint8 bits [L, B, HKV, N, HD / 16]; data is uint8
    [L, B, HKV, N, HD / 2]. ``rot``: the prefix holds x H; the float tail stays in the natural basis."""

    def forward(self, ids, pos, kd, ks, vd, vs, tk, tv, tlen, kb, vb, plen=None):
        return self._step(ids, pos, kd, ks, vd, vs, tk, tv, tlen, plen, kb, vb)


def make_decode_step(model: P.Llama, kv_cache: Optional[str] = None, group: int = 32,
                     return_q: bool = False, rot: bool = False) -> nn.Module:
    """The decode step module: kv_cache None is P.DenseStep exactly; "int8", "fp8", "int4" or
    "nvfp4" opts in."""
    if kv_cache is None:
        return P.DenseStep(model)
    if kv_cache in KQ.LOW_FORMATS:
        return LowQuantDecodeStep(model, kv_cache, group, return_q, rot=rot)
    return QuantDecodeStep(model, kv_cache, group, return_q)


def import_quant_step(model: P.Llama, fmt: str, group: int = 32, batch: int = 2, return_q: bool = False,
                      fixed: bool = False, rot: bool = False):
    """The quantized step exported with symbolic batch and prefix length and imported (float64).
    ``fixed``: the prefix is allocated to a capacity and its live length is the input ``plen``, so the
    step's shapes stay the same as the cache fills (one CUDA graph for the whole decode)."""
    from torch.export import Dim

    from truestar import ir

    c = model.c
    low = fmt in KQ.LOW_FORMATS
    step = (LowQuantDecodeStep if low else QuantDecodeStep)(copy.deepcopy(model).double(), fmt, group, return_q,
                                                            fixed, rot)
    B, N = Dim("B", min=1, max=64), Dim("N", min=2, max=1 << 20)
    n0 = 2 * group
    dt = torch.int8 if fmt == "int8" else torch.uint8
    kd = torch.zeros(c.layers, batch, c.kv_heads, n0, c.head_dim // 2 if low else c.head_dim, dtype=dt)
    # float32 scales: a mixed precision island, stored in float32 even in a float16 program
    ks = torch.ones(c.layers, batch, c.kv_heads, n0 // group, dtype=torch.float32)
    tk = torch.zeros(c.layers, batch, c.kv_heads, group, c.head_dim, dtype=torch.float64)
    args = (torch.zeros(batch, 1, dtype=torch.int64), torch.tensor([n0]), kd, ks, kd.clone(), ks.clone(),
            tk, tk.clone(), torch.tensor([0]))
    NG = Dim("NG", min=1, max=1 << 16)          # scale groups: N / group, a symbol of its own
    dyn = ({0: B}, None, {1: B, 3: N}, {1: B, 3: NG}, {1: B, 3: N}, {1: B, 3: NG}, {1: B}, {1: B}, None)
    if low:      # E4M3 block scales as uint8 bits (0x38 is 1.0), one per token and 16 channels
        kb = torch.full((c.layers, batch, c.kv_heads, n0, c.head_dim // 16), 0x38, dtype=torch.uint8)
        args = args + (kb, kb.clone())
        dyn = dyn + ({1: B, 3: N}, {1: B, 3: N})
    if fixed:
        args = args + (torch.tensor([n0]),)
        dyn = dyn + (None,)
    return ir.import_module(step, args, dynamic_shapes=dyn,
                            name=f"quant_step_{fmt}" + ("_rot" if rot else "") + ("_fixed" if fixed else ""))


class QuantCache:
    """The host side of a quantized cache: prefix (data, scales), float tails and their length."""

    def __init__(self, fmt: str, group: int, kd, ks, vd, vs, tk, tv, tlen: int, plen: Optional[int] = None,
                 kb=None, vb=None, rot: bool = False):
        self.fmt, self.group, self.kb, self.vb, self.rot = fmt, group, kb, vb, rot
        self.kd, self.ks, self.vd, self.vs, self.tk, self.tv, self.tlen = kd, ks, vd, vs, tk, tv, tlen
        self.plen = plen          # live keys of a fixed capacity prefix, or None (prefix grows)
        self.k_orig: List[torch.Tensor] = []   # unquantized groups, kept only when bounds are wanted
        self.v_orig: List[torch.Tensor] = []
        self.keep_orig = False

    @staticmethod
    def _q(x: torch.Tensor, fmt: str, group: int, rot: bool = False):
        """Quantize [L, B, H, n, D] per layer; fp8 data as uint8 bits (the program's input dtype).
        Returns (data, scale, the 4 bit formats' block scales as uint8 bits or None)."""
        low = fmt in KQ.LOW_FORMATS
        parts = [KQ.quantize(x[i].half(), fmt, group, **({"rot": rot} if low else {})) for i in range(x.shape[0])]
        data = torch.stack([p.data for p in parts])
        if fmt == "fp8":
            data = data.view(torch.uint8)
        bs = torch.stack([p.bscale.view(torch.uint8) for p in parts]) if low else None
        return data, torch.stack([p.scale for p in parts]), bs

    @classmethod
    def from_dense(cls, kc: torch.Tensor, vc: torch.Tensor, length: int, fmt: str, group: int = 32,
                   keep_orig: bool = False, capacity: Optional[int] = None, rot: bool = False):
        """From a dense float cache [L, B, HKV, S, HD] holding ``length`` tokens (after a prefill):
        whole groups go to the prefix, the rest to the tail. ``capacity`` (a multiple of ``group``)
        allocates the prefix to that many tokens, for a step imported with ``fixed=True``."""
        full = length // group * group
        kd, ks, kb = cls._q(kc[:, :, :, :full].contiguous(), fmt, group, rot)
        vd, vs, vb = cls._q(vc[:, :, :, :full].contiguous(), fmt, group, rot)
        L, B, H, _, D = kc.shape
        plen = None
        if capacity is not None:
            if capacity % group or capacity < full:
                raise KQ.KVQuantError(f"[TS701] capacity {capacity} must be a multiple of {group} of at least {full}")
            def grow(d, sc):
                d2 = torch.zeros(L, B, H, capacity, d.shape[-1], dtype=d.dtype, device=d.device)
                s2 = torch.ones(L, B, H, capacity // group, dtype=sc.dtype, device=sc.device)
                d2[:, :, :, :full], s2[:, :, :, : full // group] = d, sc
                return d2, s2
            kd, ks = grow(kd, ks)
            vd, vs = grow(vd, vs)
            if kb is not None:
                kb = grow(kb, ks[:, :, :, : full // group])[0]
                vb = grow(vb, vs[:, :, :, : full // group])[0]
            plen = full
        tk = torch.zeros(L, B, H, group, D, dtype=kc.dtype, device=kc.device)
        tv = torch.zeros_like(tk)
        tk[:, :, :, : length - full] = kc[:, :, :, full:length]
        tv[:, :, :, : length - full] = vc[:, :, :, full:length]
        qc = cls(fmt, group, kd, ks, vd, vs, tk, tv, length - full, plen, kb, vb, rot)
        qc.keep_orig = keep_orig
        if keep_orig:
            qc.k_orig.append(kc[:, :, :, :full].clone())
            qc.v_orig.append(vc[:, :, :, :full].clone())
        return qc

    def inputs(self, device):
        ins = [self.kd, self.ks, self.vd, self.vs, self.tk, self.tv,
               torch.tensor([self.tlen], dtype=torch.int64, device=device)]
        if self.kb is not None:
            ins += [self.kb, self.vb]
        if self.plen is not None:
            ins.append(torch.tensor([self.plen], dtype=torch.int64, device=device))
        return ins

    def after_step(self, tk, tv):
        """Take the step's tails; quantize a full one into the prefix (each group exactly once)."""
        self.tk, self.tv, self.tlen = tk, tv, self.tlen + 1
        if self.tlen == self.group:
            kd, ks, kb = self._q(tk, self.fmt, self.group, self.rot)
            vd, vs, vb = self._q(tv, self.fmt, self.group, self.rot)
            if self.keep_orig:
                self.k_orig.append(tk.clone())
                self.v_orig.append(tv.clone())
            if self.plen is not None:     # in place at the live end: same tensors, same shapes
                p, g = self.plen, self.group
                if p + g > self.kd.shape[3]:
                    raise KQ.KVQuantError(f"[TS701] the quantized cache is full ({self.kd.shape[3]} tokens)")
                self.kd[:, :, :, p:p + g], self.ks[:, :, :, p // g:p // g + 1] = kd, ks
                self.vd[:, :, :, p:p + g], self.vs[:, :, :, p // g:p // g + 1] = vd, vs
                if kb is not None:
                    self.kb[:, :, :, p:p + g], self.vb[:, :, :, p:p + g] = kb, vb
                self.plen = p + g
            else:
                self.kd, self.ks = torch.cat([self.kd, kd], 3), torch.cat([self.ks, ks], 3)
                self.vd, self.vs = torch.cat([self.vd, vd], 3), torch.cat([self.vs, vs], 3)
                if kb is not None:
                    self.kb, self.vb = torch.cat([self.kb, kb], 3), torch.cat([self.vb, vb], 3)
            tk.zero_()       # in place: a graphed step binds the tails by address (update=)
            tv.zero_()
            self.tlen = 0

    def kv_bytes_per_token(self) -> float:
        L, _, H, _, _ = self.kd.shape
        return L * KQ.bytes_per_token(self.fmt, self.tk.shape[-1], self.group, H)


def layer_error_bounds(q_layers: torch.Tensor, qc: QuantCache) -> List[float]:
    """Per layer, the largest proven bound (truestar.kvquant.error_bound) on the attention output's
    deviation from attention over the unquantized cache, for the step whose q's are given. Needs a
    cache built with ``keep_orig=True``; the live tail is exact and enters both sides alike."""
    out = []
    for i in range(q_layers.shape[0]):
        k = torch.cat([o[i] for o in qc.k_orig], 2).half()
        v = torch.cat([o[i] for o in qc.v_orig], 2).half()
        fmt = qc.fmt
        kd = qc.kd[i].view(torch.float8_e4m3fn) if fmt == "fp8" else qc.kd[i]
        vd = qc.vd[i].view(torch.float8_e4m3fn) if fmt == "fp8" else qc.vd[i]
        kc = KQ.QuantKV(kd, qc.ks[i].float(), fmt, qc.group)
        vc = KQ.QuantKV(vd, qc.vs[i].float(), fmt, qc.group)
        n = qc.tlen
        tail = (qc.tk[i][:, :, :n].half(), qc.tv[i][:, :, :n].half())
        b = KQ.error_bound_values(q_layers[i].half(), torch.cat([kc.dequantize(), tail[0].double()], 2),
                                  torch.cat([vc.dequantize(), tail[1].double()], 2),
                                  torch.cat([k, tail[0]], 2), torch.cat([v, tail[1]], 2))
        out.append(b.max().item())
    return out


def main():
    torch.manual_seed(0)
    if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        raise SystemExit("needs CUDA")
    from truestar.compiler import compile_program

    c = P.Cfg(vocab=256, dim=128, heads=8, kv_heads=2, head_dim=16, layers=2, ffn=256, block=16)
    model = P.Llama(c).eval()
    prompt_len, new, group, b = 70, 40, 16, 2
    dense, _ = P.import_steps(model, capacity=prompt_len + new)
    cd = compile_program(dense, "cuda", "float32")
    prompt = torch.randint(0, c.vocab, (b, prompt_len), generator=torch.Generator().manual_seed(1))
    toks_ref, _ = P.generate_dense(lambda *a: cd(list(a)), prompt, new, c, prompt_len + new,
                                   device="cuda", dtype=torch.float32)
    for fmt in ("int8", "fp8"):
        cq = compile_program(import_quant_step(model, fmt, group), "cuda", "float32")
        kc = torch.zeros(c.layers, b, c.kv_heads, prompt_len, c.head_dim, device="cuda")
        logits, kc, vc = cd([prompt.cuda(), torch.arange(prompt_len, device="cuda"), kc, kc.clone()])
        qc = QuantCache.from_dense(kc, vc, prompt_len, fmt, group)
        ids, toks = logits[:, -1].argmax(-1)[:, None], [logits[:, -1].argmax(-1)]
        for s in range(new - 1):
            lg, tk, tv = cq([ids, torch.tensor([prompt_len + s], device="cuda")] + qc.inputs("cuda"))
            qc.after_step(tk, tv)
            ids = lg[:, -1].argmax(-1)[:, None]
            toks.append(ids[:, 0])
        agree = (torch.stack(toks, 1).cpu() == toks_ref.cpu()).float().mean().item()
        print(f"{fmt}: {agree:.0%} of {new} greedy tokens equal the float32 cache; library steps "
              f"{len(cq.report.library)}, fallbacks {len(cq.report.fallbacks)}; KV bytes per token "
              f"{qc.kv_bytes_per_token():.0f} (float16 {c.layers * KQ.bytes_per_token('float16', c.head_dim, group, c.kv_heads):.0f})")


if __name__ == "__main__":
    main()
