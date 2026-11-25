"""Prefix cache check on Truestar's own caches: a cached prefix run against one full prefill.

    PYTHONPATH=src python scripts/prefix_check_llama.py --json out.json

A Llama style decoder (examples/llama_paged_decode.py, random weights) compiled in deterministic mode
runs a sequence of ``--length`` tokens two ways for each split point and cache:

* full: one prefill of all tokens (dense step).
* cached, by cache:
  * ``dense``: prefill of the first ``split`` tokens, then the rest as one chunk against that cache.
  * ``paged``: the same through the paged step, the pool's blocks in a shuffled order.
  * ``dense-decode``: the prefix prefill, then the rest one token at a time (teacher forced).
  * ``int8``, ``fp8``: the prefix prefill, its cache quantized (examples/llama_kvquant_decode.py), then
    the rest one token at a time.

Each pair goes through ``truestar.prefixcheck.compare`` with the K and V caches of every layer and a
float64 reference (the eager model in float64); one row per cache and split.
"""

import argparse
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples"))


def run(length=160, splits=(1, 37, 64, 120), caches=("dense", "paged", "dense-decode", "int8", "fp8"),
        dtype="float32", batch=2, seed=0, group=16):
    import torch
    import llama_paged_decode as P
    import llama_kvquant_decode as Q
    from truestar import prefixcheck as PC
    from truestar.compiler import compile_program

    torch.manual_seed(seed)
    c = P.Cfg(vocab=256, dim=128, heads=4, kv_heads=2, head_dim=32, layers=3, ffn=256, block=16)
    model = P.Llama(c).eval()
    dense_ir, paged_ir = P.import_steps(model, capacity=length, n_blocks=4 * (length // c.block + 1),
                                        width=length // c.block + 1, max_capacity=max(4096, 2 * length),
                                        max_tokens=max(512, length))
    cd = compile_program(dense_ir, "cuda", dtype, deterministic=True)
    cp = compile_program(paged_ir, "cuda", dtype, deterministic=True)
    tdt = getattr(torch, dtype)
    ids = torch.randint(0, c.vocab, (batch, length), generator=torch.Generator().manual_seed(seed + 1)).cuda()
    z = torch.zeros(c.layers, batch, c.kv_heads, length, c.head_dim, device="cuda", dtype=tdt)
    full, kf, vf = cd([ids, torch.arange(length, device="cuda"), z, z.clone()])
    m64 = P.DenseStep(copy.deepcopy(model)).double().cuda()   # a copy: import_quant_step below traces the CPU model
    with torch.no_grad():
        ref, _, _ = m64(ids, torch.arange(length, device="cuda"), z.double(), z.double())
    qsteps = {f: compile_program(Q.import_quant_step(model, f, group), "cuda", dtype, deterministic=True)
              for f in ("int8", "fp8") if f in caches}
    rows = []
    for split in splits:
        pre, ka, va = cd([ids[:, :split], torch.arange(split, device="cuda"), z, z.clone()])
        for cache in caches:
            if cache == "dense":
                lg, kc, vc = cd([ids[:, split:], torch.arange(split, length, device="cuda"), ka, va])
                layers = lambda kk, vv: {f"layer{l}.k": kk[l] for l in range(c.layers)}  # noqa: E731
                fl = {**{f"layer{l}.k": kf[l] for l in range(c.layers)}, **{f"layer{l}.v": vf[l] for l in range(c.layers)}}
                cl = {**{f"layer{l}.k": kc[l] for l in range(c.layers)}, **{f"layer{l}.v": vc[l] for l in range(c.layers)}}
                del layers
            elif cache == "paged":
                width = length // c.block + 1
                pool = P.BlockPool(4 * width, c.block, order=torch.randperm(4 * width,
                                   generator=torch.Generator().manual_seed(7)).tolist())
                seqs = [pool.add_sequence() for _ in range(batch)]
                kp = torch.zeros(c.layers, 4 * width * c.block, c.kv_heads, c.head_dim, device="cuda", dtype=tdt)
                vp = torch.zeros_like(kp)

                def call(lo, hi, kp, vp):
                    for q in seqs:
                        pool.reserve(q, hi)
                    pos = torch.tensor([list(range(lo, hi))] * batch, device="cuda")
                    slots = torch.tensor([pool.slots(q, list(range(lo, hi))) for q in seqs], device="cuda")
                    return cp([ids[:, lo:hi], pos, slots, pool.table(seqs, width).cuda(), kp, vp])
                _, kp, vp = call(0, split, kp, vp)
                lg, kp, vp = call(split, length, kp, vp)
                read = (pool.table(seqs, width).cuda()[:, :, None] * c.block
                        + torch.arange(c.block, device="cuda")).reshape(batch, -1)[:, :length]
                kc = kp[:, read].permute(0, 1, 3, 2, 4)      # [L, B, HKV, S, HD]
                vc = vp[:, read].permute(0, 1, 3, 2, 4)
                fl = {**{f"layer{l}.k": kf[l] for l in range(c.layers)}, **{f"layer{l}.v": vf[l] for l in range(c.layers)}}
                cl = {**{f"layer{l}.k": kc[l] for l in range(c.layers)}, **{f"layer{l}.v": vc[l] for l in range(c.layers)}}
            elif cache == "dense-decode":
                kc, vc, out = ka, va, []
                for t in range(split, length):
                    lg1, kc, vc = cd([ids[:, t:t + 1], torch.tensor([t], device="cuda"), kc, vc])
                    out.append(lg1)
                lg = torch.cat(out, 1)
                fl = {**{f"layer{l}.k": kf[l] for l in range(c.layers)}, **{f"layer{l}.v": vf[l] for l in range(c.layers)}}
                cl = {**{f"layer{l}.k": kc[l] for l in range(c.layers)}, **{f"layer{l}.v": vc[l] for l in range(c.layers)}}
            else:
                if split < group:    # the quantized step needs one full group in its prefix
                    print(f"{cache:12} split {split:4}: skipped (prefix shorter than one group of {group})")
                    continue
                qc = Q.QuantCache.from_dense(ka[:, :, :, :split], va[:, :, :, :split], split, cache, group)
                out = []
                for t in range(split, length):
                    lg1, tk, tv = qsteps[cache]([ids[:, t:t + 1], torch.tensor([t], device="cuda")] + qc.inputs("cuda"))
                    qc.after_step(tk, tv)
                    out.append(lg1)
                lg = torch.cat(out, 1)
                fl = cl = None
            rep = PC.compare(full[:, split:], lg, fl, cl, ref[:, split:])
            row = {"cache": cache, "split": split, "dtype": dtype, **rep.to_dict()}
            rows.append(row)
            print(f"{cache:12} split {split:4}: {'BITWISE' if rep.bitwise else 'DIVERGENT'}"
                  + ("" if rep.bitwise else f" first token {rep.first_token}, max diff {rep.max_abs:.2e}"
                     + (f", first layer {rep.first_layer}" if rep.first_layer else ""))
                  + f"; vs float64: full {rep.error_full:.2e}, cached {rep.error_cached:.2e}; {' '.join(rep.codes)}",
                  flush=True)
    return rows


def attention_case(keys=2200, rows=100, dtype="float16", seed=0):
    """The fused attention kernel alone, deterministic: the last query row computed inside a prefill
    of ``rows`` rows (no split: a prefill has a program per query block) and as a one row decode over
    the same keys (split KV in chunks of 256 from 2048 keys). Shows where a decode can differ from the
    prefill it continues, as the Llama steps above do not (their attention runs unfused)."""
    import torch
    from truestar import kernel as K
    from truestar import prefixcheck as PC
    from truestar.kernel import Graph
    g = Graph()
    T, N = K.sym("T"), K.sym("N")
    q = g.input("q", (1, 4, T, 64), dtype)
    k, v = (g.input(n, (1, 4, N, 64), dtype) for n in "kv")
    (_, kern), = g.fuse({"o": g.softmax((q @ k.T) * 0.125) @ v})
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dt = getattr(torch, dtype)
    qs = torch.randn(1, 4, rows, 64, device="cuda", generator=gen).to(dt)
    ks, vs = (torch.randn(1, 4, keys, 64, device="cuda", generator=gen).to(dt) for _ in range(2))
    ck = K.compile(kern, "cuda", tune=False, deterministic=True)
    pre = ck(q=qs, k=ks, v=vs)[:, :, -1:]
    dec = ck(q=qs[:, :, -1:].contiguous(), k=ks, v=vs)
    chunks = [c.get("KV_CHUNK") for _, c, _ in ck.launch_configs()]
    s = (qs[:, :, -1:].double() @ ks.double().transpose(-1, -2)) * 0.125
    ref = torch.softmax(s, -1) @ vs.double()
    rep = PC.compare(pre[0, 0], dec[0, 0], reference=ref[0, 0])
    print(f"attention kernel, {keys} keys, prefill row vs decode (KV_CHUNK {chunks}): "
          f"{'BITWISE' if rep.bitwise else 'DIVERGENT'}, max diff {rep.max_abs:.2e}; vs float64: prefill "
          f"{rep.error_full:.2e}, decode {rep.error_cached:.2e}; {' '.join(rep.codes)}")
    return {"case": "attention_kernel", "keys": keys, "kv_chunks": chunks, **rep.to_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=160)
    ap.add_argument("--splits", default="1,37,64,120")
    ap.add_argument("--caches", default="dense,paged,dense-decode,int8,fp8")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = run(a.length, tuple(int(s) for s in a.splits.split(",")), tuple(a.caches.split(",")), a.dtype)
    rows.append(attention_case())
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
