"""Cross chip check at scale: a 1.24B parameter Llama style decoder, compiled deterministically.

    PYTHONPATH=src python scripts/xchip_scale.py --json out/scale.json [--dtypes float32,bfloat16]

The shapes are Llama 3.2 1B's (dim 2048, 16 layers, 32 query heads, 8 KV heads, head dim 64,
SwiGLU hidden 8192, vocab 128256, tied embeddings, rotary base 500000 without the long context
rescaling). Weights are random, drawn on the CPU with the portable counter hash the fingerprint uses
(splitmix64 in numpy uint64, ``fingerprint.models.portable_uniform``, here chunked), so every machine
starts from the same bits; their sha256 is recorded and compared before any output is.

Two programs, each compiled with ``deterministic=True`` and TF32 off, torch deterministic algorithms on:

* prefill: token ids [1, 64] and positions -> logits [1, 64, vocab] and the K, V cache of every layer
* decode: one token id, its position and the 32 cache tensors from the prefill -> logits [1, vocab]
  and the grown caches

Recorded per program: sha256 of every output on the first and on a second call, the compiled
program's own end to end error against its float64 reference interpreter, and an independent
error against float64: the same module run in float64 by PyTorch on the GPU, one layer at a time.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from truestar.fingerprint import data as D
from truestar.fingerprint.models import _M64, _splitmix
from truestar.fingerprint.run import environment, global_settings, torch_settings

DIM, LAYERS, HEADS, KV, HD, HIDDEN, VOCAB, THETA, EPS = 2048, 16, 32, 8, 64, 8192, 128256, 500000.0, 1e-5
T = 64
SEED = 2026


class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(DIM))

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))


def rope(x, cos, sin):
    half = x.shape[-1] // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rot * sin


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.q = nn.Linear(DIM, HEADS * HD, bias=False)
        self.k = nn.Linear(DIM, KV * HD, bias=False)
        self.v = nn.Linear(DIM, KV * HD, bias=False)
        self.o = nn.Linear(HEADS * HD, DIM, bias=False)
        self.mlp_norm = RMSNorm()
        self.gate = nn.Linear(DIM, HIDDEN, bias=False)
        self.up = nn.Linear(DIM, HIDDEN, bias=False)
        self.down = nn.Linear(HIDDEN, DIM, bias=False)

    def forward(self, x, cos, sin, past_k=None, past_v=None):
        B, S, _ = x.shape
        h = self.attn_norm(x)
        q = rope(self.q(h).view(B, S, HEADS, HD).transpose(1, 2), cos, sin)
        k = rope(self.k(h).view(B, S, KV, HD).transpose(1, 2), cos, sin)
        v = self.v(h).view(B, S, KV, HD).transpose(1, 2)
        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            att = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        else:
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + self.o(att.transpose(1, 2).reshape(B, S, HEADS * HD))
        h = self.mlp_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h)), k, v


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = RMSNorm()
        inv = 1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
        self.register_buffer("inv_freq", torch.from_numpy(inv.astype(np.float32)))

    def angles(self, pos):
        ang = pos.to(self.inv_freq.dtype)[:, None] * self.inv_freq[None, :]
        ang = torch.cat([ang, ang], dim=-1)
        return ang.cos(), ang.sin()

    def head(self, x):
        return F.linear(self.norm(x), self.embed.weight)


class Prefill(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, ids, pos):
        cos, sin = self.dec.angles(pos)
        x = self.dec.embed(ids)
        caches = []
        for blk in self.dec.blocks:
            x, k, v = blk(x, cos, sin)
            caches += [k, v]
        return (self.dec.head(x),) + tuple(caches)


class Decode(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, ids, pos, *caches):
        cos, sin = self.dec.angles(pos)
        x = self.dec.embed(ids)
        out = []
        for i, blk in enumerate(self.dec.blocks):
            x, k, v = blk(x, cos, sin, caches[2 * i], caches[2 * i + 1])
            out += [k, v]
        return (self.dec.head(x)[:, -1],) + tuple(out)


def _key(seed: int, stream: int):
    return np.uint64(((seed * 1_000_003 + stream) * 0x2545F4914F6CDD1D) & _M64)


def uniform_pm1(seed: int, stream: int, n: int, chunk: int = 1 << 24) -> np.ndarray:
    """2 U - 1 in float32 with U = portable_uniform(seed, stream, n), computed in chunks."""
    key, out = _key(seed, stream), np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        with np.errstate(over="ignore"):
            x = _splitmix(np.arange(s, e, dtype=np.uint64) ^ key)
        out[s:e] = ((x >> np.uint64(40)).astype(np.float64) / float(1 << 24) * 2.0 - 1.0).astype(np.float32)
    return out


def build() -> Decoder:
    with torch.device("meta"):
        dec = Decoder()
    dec = dec.to_empty(device="cpu")
    inv = 1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
    dec.inv_freq.copy_(torch.from_numpy(inv.astype(np.float32)))
    with torch.no_grad():
        for i, (name, p) in enumerate(dec.named_parameters()):
            u = uniform_pm1(SEED, 1000 + i, p.numel()).reshape(tuple(p.shape))
            if p.dim() >= 2:
                u *= np.float32(1.0 / np.sqrt(p.shape[1]))
            else:
                u = np.float32(1.0) + np.float32(0.1) * u
            p.copy_(torch.from_numpy(u))
    return dec.eval()


def portable_ids(stream: int, n: int) -> torch.Tensor:
    u = (uniform_pm1(SEED, stream, n) + np.float32(1.0)) / np.float32(2.0)
    return torch.from_numpy(np.minimum((u.astype(np.float64) * VOCAB).astype(np.int64), VOCAB - 1))


def rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    g, w = got.detach().double(), want.detach().double().to(got.device)
    scale = float(w.abs().max())
    err = float((g - w).abs().max())
    return float("inf") if err != err else (err / scale if scale > 0 else err)


def outputs_sha(outs) -> str:
    h = hashlib.sha256()
    for o in outs:
        h.update(D.tensor_sha(o).encode())
    return h.hexdigest()


@torch.no_grad()
def reference(dec: Decoder, ids, pos, caches=None):
    """float64 forward on the GPU, one block at a time (the float64 weights never all live at once)."""
    dev = torch.device("cuda")
    pos = pos.to(dev)
    small = copy.deepcopy(nn.ModuleDict({"norm": dec.norm})).to(dev, torch.float64)
    inv = dec.inv_freq.to(dev, torch.float64)
    ang = pos.double()[:, None] * inv[None, :]
    ang = torch.cat([ang, ang], dim=-1)
    cos, sin = ang.cos(), ang.sin()
    emb = dec.embed.weight.to(dev, torch.float64)
    x = emb[ids.to(dev)]
    out = []
    for i, blk in enumerate(dec.blocks):
        b = copy.deepcopy(blk).to(dev, torch.float64)
        if caches is None:
            x, k, v = b(x, cos, sin)
        else:
            x, k, v = b(x, cos, sin, caches[2 * i], caches[2 * i + 1])
        out += [k, v]
        del b
    logits = F.linear(small["norm"](x), emb)
    if caches is not None:
        logits = logits[:, -1]
    del emb
    return [logits] + out


def run_program(name: str, module: nn.Module, example, args, dtype: str, log) -> dict:
    from truestar import ir
    from truestar.compiler import compile_program

    rec = {"program": name, "dtype": dtype}
    t0 = time.perf_counter()
    prog = ir.import_module(module, tuple(example), name=name)
    rec["import_s"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    cp = compile_program(prog, "cuda", dtype, allow_tf32=False, deterministic=True)
    rec["compile_s"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    first = [o.detach().clone() for o in cp(list(args))]
    torch.cuda.synchronize()
    rec["first_call_s"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    again = [o.detach().clone() for o in cp(list(args))]
    torch.cuda.synchronize()
    rec["second_call_s"] = round(time.perf_counter() - t0, 3)
    rec["logits_sha256"] = D.tensor_sha(first[0])
    rec["outputs_sha256"] = outputs_sha(first)
    rec["again_outputs_sha256"] = outputs_sha(again)
    rec["run_to_run_equal"] = rec["outputs_sha256"] == rec["again_outputs_sha256"]
    rec["per_output_sha256"] = [D.tensor_sha(o) for o in first]
    rec["own_check_max_error_vs_float64"] = getattr(cp, "max_error", None)
    rec["end_to_end_device"] = getattr(cp, "end_to_end_device", None)
    rep = cp.report
    rec["kernels"] = len(rep.kernels)
    rec["library_ops"] = sorted({op for e in rep.library for op in e.get("ops", [])})
    rec["fallback_ops"] = sorted({op for e in rep.fallbacks for op in e.get("ops", [])})
    cfgs = []
    for step in cp.plan.steps:
        ck = getattr(step, "ck", None)
        if ck is None or not hasattr(ck, "launch_configs"):
            continue
        for sig, cfg, src in ck.launch_configs():
            cfgs.append([step.name, [[n, list(s)] for n, s in sig], dict(cfg), src])
    rec["config_sources"] = sorted({c[3] for c in cfgs})
    rec["configs_sha256"] = hashlib.sha256(json.dumps([c[:3] for c in cfgs], sort_keys=True,
                                                      default=str).encode()).hexdigest()
    rec["configs"] = cfgs
    rec["peak_gpu_mib"] = torch.cuda.max_memory_allocated() // (1 << 20)
    log(f"{name} {dtype}: logits {rec['logits_sha256'][:12]} all outputs {rec['outputs_sha256'][:12]} "
        f"run to run {'equal' if rec['run_to_run_equal'] else 'DIFFERENT'}, own check {rec['own_check_max_error_vs_float64']}, "
        f"compile {rec['compile_s']} s, first call {rec['first_call_s']} s, second {rec['second_call_s']} s")
    del cp, prog
    torch.cuda.empty_cache()
    return rec, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--dtypes", default="float32")
    a = ap.parse_args()

    def log(s):
        print(s, flush=True)

    if not torch.cuda.is_available():
        sys.exit("CUDA is required")
    doc = {"model": f"Llama style decoder, Llama 3.2 1B shapes: dim {DIM}, {LAYERS} layers, {HEADS} heads, "
                    f"{KV} KV heads, head dim {HD}, hidden {HIDDEN}, vocab {VOCAB}, tied embeddings, "
                    f"rotary base {THETA:g}; prompt {T} tokens, batch 1",
           "env": environment(), "settings": global_settings(), "programs": []}
    t0 = time.perf_counter()
    dec = build()
    doc["parameters"] = sum(p.numel() for p in dec.parameters())
    doc["weights_sha256"] = D.state_sha(dec.state_dict())
    doc["build_s"] = round(time.perf_counter() - t0, 1)
    ids = portable_ids(1, T).reshape(1, T)
    pos = torch.arange(T, dtype=torch.int64)
    nxt = portable_ids(2, 1).reshape(1, 1)
    npos = torch.tensor([T], dtype=torch.int64)
    doc["inputs_sha256"] = D.inputs_sha((ids, pos, nxt, npos))
    log(f"{doc['parameters'] / 1e9:.3f}B parameters, weights {doc['weights_sha256'][:16]}, inputs "
        f"{doc['inputs_sha256'][:16]}, built in {doc['build_s']} s")

    ref_pre = reference(dec, ids, pos)
    ref_dec = reference(dec, nxt, npos, caches=ref_pre[1:])
    ref_pre_logits, ref_dec_logits = ref_pre[0], ref_dec[0]
    del ref_pre, ref_dec
    torch.cuda.empty_cache()

    def save():
        doc["seconds"] = round(time.perf_counter() - t0, 1)
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1, default=str)

    for dtype in a.dtypes.split(","):
        try:
            one_dtype(doc, dec, dtype, ids, pos, nxt, npos, ref_pre_logits, ref_dec_logits, log)
        except Exception as exc:          # recorded, never hidden; the other dtypes still run
            import traceback

            doc["programs"].append({"dtype": dtype, "error": f"{type(exc).__name__}: {exc}"[:3000],
                                    "traceback": traceback.format_exc()[-4000:]})
            log(f"{dtype}: ERROR {type(exc).__name__}: {str(exc)[:500]}")
        torch.cuda.empty_cache()
        save()
    log(f"wrote {a.json} in {doc['seconds']} s")


def one_dtype(doc, dec, dtype, ids, pos, nxt, npos, ref_pre_logits, ref_dec_logits, log):
    if True:
        td = getattr(torch, dtype)
        model = dec if dtype == "float32" else copy.deepcopy(dec).to(td)
        with torch_settings(True, False):
            pre_rec, pre_out = run_program("llama1b_prefill", Prefill(model), (ids, pos),
                                           [ids.cuda(), pos.cuda()], dtype, log)
            pre_rec["max_rel_error_vs_float64_logits"] = rel_err(pre_out[0], ref_pre_logits)
            pre_rec["top1_equal_float64"] = float((pre_out[0].argmax(-1).cpu() == ref_pre_logits.argmax(-1).cpu())
                                                  .double().mean())
            caches = [c for c in pre_out[1:]]
            del pre_out
            dec_rec, dec_out = run_program("llama1b_decode", Decode(model), (nxt, npos, *[c.cpu() for c in caches]),
                                           [nxt.cuda(), npos.cuda()] + caches, dtype, log)
            dec_rec["decode_inputs_sha256"] = outputs_sha(caches)
            dec_rec["max_rel_error_vs_float64_logits"] = rel_err(dec_out[0], ref_dec_logits)
            dec_rec["top1_equal_float64"] = bool(int(dec_out[0].argmax(-1)) == int(ref_dec_logits.argmax(-1)))
            del dec_out, caches
        log(f"{dtype}: error vs float64 prefill {pre_rec['max_rel_error_vs_float64_logits']:.3g}, decode "
            f"{dec_rec['max_rel_error_vs_float64_logits']:.3g}")
        doc["programs"] += [pre_rec, dec_rec]
        if model is not dec:
            del model


if __name__ == "__main__":
    main()
