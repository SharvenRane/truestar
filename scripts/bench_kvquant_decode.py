"""End to end Llama decode compiled by Truestar: float16 cache against an opt in int8 or fp8 cache.

    PYTHONPATH=src python scripts/bench_kvquant_decode.py --json out.json

Model: the configuration of scripts/bench_decode.py (vocab 8192, dim 512, 4 layers, 8 query and 2 KV
heads of dim 64, ffn 1408), random weights. For each batch and context N the cache holds N tokens of
random K and V (decode speed does not depend on their values) and 32 tokens are decoded:

* ``float32``: examples/llama_paged_decode.py's dense step (``kv_cache=None``) compiled in float32,
  capacity N + 32. Its step returns a new cache (value semantics), so every step also copies the
  whole cache: part of the gap below is that copy, not the cache format. (Compiled in float16 the
  dense step failed Truestar's own end to end check on these random weights, relative error 0.085
  against the tolerance 0.066 after one step, and returned nothing; so every program here is
  float32, and the float16 cache memory is reported as the figure a float16 cache would take.)
* ``int8``, ``fp8``: examples/llama_kvquant_decode.py's step compiled in float32 (its attention runs
  in float16 inside the kernel), groups of 32, including the host's quantization of the tail when it
  fills (once in the 32 steps).

Each step is timed with CUDA events (host work in between included); reported: the median and the
minimum step time, tokens per second (batch / median), and KV memory per token and in total.
Provisional numbers on a laptop GPU.
"""

import argparse
import json
import os
import statistics
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--contexts", default="2048,8192")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                    help="program precision; float16 needs --rope-f32 to pass Truestar's checks")
    ap.add_argument("--rope-f32", action="store_true", help="float32 rotary angles (mixed precision islands)")
    ap.add_argument("--mode", default="plain", choices=["plain", "inplace", "graphed"],
                    help="plain: each step returns new caches; inplace: update= writes them in place; "
                         "graphed: cp.graphed() with the same in place inputs (one CUDA graph per signature)")
    ap.add_argument("--flat", action="store_true",
                    help="the float cache through examples/llama_paged_decode.py's FlatStep (one buffer, "
                         "written in place) instead of the dense step")
    ap.add_argument("--fixed", action="store_true",
                    help="the quantized steps with a fixed capacity prefix and a device live length "
                         "(graphed: one graph for the whole decode, the prefix bound by address)")
    ap.add_argument("--formats", default="int8,fp8",
                    help="quantized formats: int8, fp8, int4, nvfp4, int4_rot, nvfp4_rot (4 bit, Hadamard rotated)")
    ap.add_argument("--json")
    a = ap.parse_args()
    os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_bench_kvqd_"))
    os.environ.setdefault("TRUESTAR_VERIFY_DEVICE", "cpu")

    import torch
    import llama_kvquant_decode as Q
    import llama_paged_decode as P
    from truestar import kvquant as KQ
    from truestar.compiler import compile_program

    torch.manual_seed(0)
    c = P.Cfg(vocab=8192, dim=512, heads=8, kv_heads=2, head_dim=64, layers=4, ffn=1408, block=16,
              rope_f32=a.rope_f32)
    pdt = a.dtype
    tdt = getattr(torch, pdt)
    model = P.Llama(c).eval()
    dense_prog = (P.import_flat_step(model, capacity=64) if a.flat
                  else P.import_steps(model, capacity=64, max_capacity=1 << 15)[0])
    progs = {pdt: compile_program(dense_prog, "cuda", pdt)}
    fmts = a.formats.split(",")
    spec = {f: (f.split("_")[0], f.endswith("_rot")) for f in fmts}
    for f, (fmt, rot) in spec.items():
        progs[f] = compile_program(Q.import_quant_step(model, fmt, a.group, fixed=a.fixed, rot=rot), "cuda", pdt)
    upd = {pdt: {3: 1, 4: 2} if a.flat else {2: 1, 3: 2}}   # float cache: K and V written in place
    upd.update({f: {6: 1, 7: 2} for f in fmts})          # quantized: the tails written in place
    bound = {f: (2, 3, 4, 5, 9, 10) if spec[f][0] in KQ.LOW_FORMATS else (2, 3, 4, 5) for f in fmts}
    if a.mode == "graphed":
        calls = {n: p.graphed(update=upd[n], static=bound[n] if (a.fixed and n in fmts) else ())
                 for n, p in progs.items()}
    elif a.mode == "inplace":
        calls = {n: (lambda ins, p=p, u=upd[n]: p(ins, update=u)) for n, p in progs.items()}
    else:
        calls = dict(progs)
    rows = []
    for B in (int(x) for x in a.batches.split(",")):
        for N in (int(x) for x in a.contexts.split(",")):
            gen = torch.Generator(device="cuda").manual_seed(N + B)
            shape = (c.layers, B, c.kv_heads, N, c.head_dim)
            k = torch.randn(*shape, device="cuda", generator=gen)
            v = torch.randn(*shape, device="cuda", generator=gen)
            ids0 = torch.randint(0, c.vocab, (B, 1), device="cuda", generator=gen)
            row = {"B": B, "N": N, "steps": a.steps,
                   "float16_cache_bytes_per_token": c.layers * KQ.bytes_per_token("float16", c.head_dim, a.group, c.kv_heads)}
            for name in progs:
                prog = calls[name]
                times = []
                ids = ids0
                if name == pdt:
                    cap = N + a.steps + 3          # 3 warm up steps, then the timed ones
                    kc = torch.zeros(c.layers, B, c.kv_heads, cap, c.head_dim, device="cuda", dtype=tdt)
                    vc = torch.zeros_like(kc)
                    kc[:, :, :, :N], vc[:, :, :, :N] = k, v
                    if a.flat:                    # [L * cap, B, HKV, HD], layer major
                        kc = kc.permute(0, 3, 1, 2, 4).reshape(c.layers * cap, B, c.kv_heads, c.head_dim).contiguous()
                        vc = vc.permute(0, 3, 1, 2, 4).reshape(c.layers * cap, B, c.kv_heads, c.head_dim).contiguous()
                    mem = c.layers * 2 * c.kv_heads * c.head_dim * float(tdt.itemsize)
                    for s in range(a.steps + 3):
                        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        st.record()
                        pos_t = torch.tensor([N + s], device="cuda")
                        if a.flat:
                            lg, kc, vc = prog([ids, pos_t, P.flat_rows(c, pos_t, cap), kc, vc])
                        else:
                            lg, kc, vc = prog([ids, pos_t, kc, vc])
                        ids = lg[:, -1].argmax(-1)[:, None]
                        en.record()
                        torch.cuda.synchronize()
                        if s >= 3:                      # the first calls compile and verify
                            times.append(st.elapsed_time(en) * 1e3)
                    total = kc.numel() * tdt.itemsize * 2
                else:
                    cap = -(-(N + a.steps + 3) // a.group) * a.group if a.fixed else None
                    qc = Q.QuantCache.from_dense(k, v, N, spec[name][0], a.group, capacity=cap, rot=spec[name][1])
                    qc.tk, qc.tv = qc.tk.to(tdt), qc.tv.to(tdt)
                    mem = qc.kv_bytes_per_token()
                    for s in range(a.steps + 3):
                        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        st.record()
                        lg, tk, tv = prog([ids, torch.tensor([N + s], device="cuda")] + qc.inputs("cuda"))
                        qc.after_step(tk, tv)
                        ids = lg[:, -1].argmax(-1)[:, None]
                        en.record()
                        torch.cuda.synchronize()
                        if s >= 3:
                            times.append(st.elapsed_time(en) * 1e3)
                    total = sum(t.numel() * t.element_size() for t in (qc.kd, qc.ks, qc.vd, qc.vs, qc.tk, qc.tv,
                                                                       qc.kb, qc.vb) if t is not None)
                    del qc
                med = statistics.median(times)
                row[name] = {"median_us": med, "min_us": min(times), "tokens_per_s": B / med * 1e6,
                             "kv_bytes_per_token": mem, "kv_total_mb": total / 2 ** 20, "mode": a.mode,
                             "captures": getattr(prog, "captures", None)}
                print(f"B{B} N{N} {name:8}: {med:8.1f} us median, {min(times):8.1f} min, "
                      f"{B / med * 1e6:8.0f} tokens/s, KV {mem:.0f} B/token, {total / 2 ** 20:.1f} MiB", flush=True)
                torch.cuda.empty_cache()
            rows.append(row)
            del k, v
    if a.json:
        props = torch.cuda.get_device_properties(0)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"gpu": props.name, "sms": props.multi_processor_count, "torch": torch.__version__,
                       "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
