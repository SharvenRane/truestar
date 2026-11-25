"""Decode attention (one new token per sequence): Truestar against FlashInfer.

    PYTHONPATH=src python scripts/bench_decode_flashinfer.py --json out.json
    PYTHONPATH=src python scripts/bench_decode_flashinfer.py --batches 1 --contexts 4096 --heads 32,8

Needs a CUDA GPU and ``flashinfer-python`` (Linux only; installed in the cloud image, not a
Truestar dependency). Cases: q [B, Hq, D] against a cache of N keys per sequence with Hkv KV heads
(``--heads Hq,Hkv``; D ``--dim``). Tools, per cache format:

* float16 cache
  * ``ts``: the kernel layer's generated attention, default launch config (tuning off). With
    Hq > Hkv the query heads that share a KV head are folded into the query rows (q viewed as
    [B, Hkv, Hq / Hkv, D], a view, no copy), so each KV head is read once.
  * ``ts_bcast`` (only Hq > Hkv): the same kernel with q [B, Hkv, n, 1, D] against k, v
    [B, Hkv, 1, N, D] (the importer's grouped query lowering: a size 1 broadcast dim).
  * ``fi``: FlashInfer ``BatchDecodeWithPagedKVCacheWrapper`` (HND layout, 16 token pages, CUDA
    graph mode), with ``use_tensor_cores`` off and on (``fi_tc``).
  * ``sdpa``: ``F.scaled_dot_product_attention`` with ``enable_gqa`` (a reference point).
* int8 cache: ``ts_int8`` (truestar.kvquant, groups of 32 tokens). FlashInfer has no int8 KV cache.
* fp8 (e4m3) cache: ``ts_fp8`` (truestar.kvquant, a scale per head and group of 32 tokens) and
  ``fi_fp8`` (FlashInfer, one scale per tensor for K and one for V).

Every tool runs as a CUDA graph of 20 calls (GPU time per call, median over interleaved rounds).
Errors are max relative errors against float64 attention over the float16 cache, so the quantized
rows include their quantization error.
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--contexts", default="1024,4096,16384")
    ap.add_argument("--heads", default="32,8;8,8", help="Hq,Hkv pairs separated by ;")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--tools", default="ts,ts_bcast,fi,fi_tc,sdpa,ts_int8,ts_fp8,fi_fp8")
    ap.add_argument("--json")
    a = ap.parse_args()
    os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="ts_bench_fi_"))

    import torch
    import torch.nn.functional as F
    import triton.testing
    from truestar import kernel as K
    from truestar import kvquant as KQ
    from truestar.kernel import Graph

    try:
        import flashinfer
        fi_version = getattr(flashinfer, "__version__", "?")
    except Exception as e:  # recorded, never hidden
        flashinfer, fi_version = None, f"import failed: {type(e).__name__}: {e}"
    props = torch.cuda.get_device_properties(0)
    print(f"gpu {props.name}, SMs {props.multi_processor_count}, torch {torch.__version__}, "
          f"flashinfer {fi_version}", flush=True)
    tools = a.tools.split(",")
    D = a.dim
    PAGE = 16
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    kern_cache = {}

    def ts_kernel(shape_q, shape_kv):
        key = (shape_q, shape_kv)
        if key not in kern_cache:
            g = Graph()
            q = g.input("q", shape_q, "float16")
            k = g.input("k", shape_kv, "float16")
            v = g.input("v", shape_kv, "float16")
            (_, kern), = g.fuse({"o": g.softmax((q @ k.T) * (D ** -0.5)) @ v})
            kern_cache[key] = K.compile(kern, "cuda", tune=False)
        return kern_cache[key]

    def graph_of(fn, calls=20):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(calls):
                fn()
        return g

    def fi_wrapper(B, Hq, Hkv, N, kv_dtype, tc):
        pages = -(-N // PAGE)
        indptr = torch.arange(0, B + 1, dtype=torch.int32, device="cuda") * pages
        indices = torch.arange(B * pages, dtype=torch.int32, device="cuda")
        last = torch.full((B,), N - (pages - 1) * PAGE, dtype=torch.int32, device="cuda")
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "HND", use_cuda_graph=True, use_tensor_cores=tc,
            paged_kv_indptr_buffer=indptr, paged_kv_indices_buffer=indices,
            paged_kv_last_page_len_buffer=last)
        w.plan(indptr, indices, last, Hq, Hkv, D, PAGE, pos_encoding_mode="NONE",
               q_data_type=torch.float16, kv_data_type=kv_dtype)
        return w

    def paged(x, B, Hkv, N):
        """[B, Hkv, N, D] -> FlashInfer HND pages [B * pages, Hkv, PAGE, D] (one copy, untimed)."""
        pages = -(-N // PAGE)
        pad = pages * PAGE - N
        if pad:
            x = torch.cat([x, x.new_zeros(B, Hkv, pad, D)], 2)
        return x.view(B, Hkv, pages, PAGE, D).permute(0, 2, 1, 3, 4).contiguous().view(B * pages, Hkv, PAGE, D)

    rows = []
    for hp in a.heads.split(";"):
        Hq, Hkv = (int(v) for v in hp.split(","))
        n = Hq // Hkv
        for B in (int(x) for x in a.batches.split(",")):
            for N in (int(x) for x in a.contexts.split(",")):
                gen = torch.Generator(device="cuda").manual_seed(0)
                q = torch.randn(B, Hq, D, device="cuda", generator=gen).half()
                k = torch.randn(B, Hkv, N, D, device="cuda", generator=gen).half()
                v = torch.randn(B, Hkv, N, D, device="cuda", generator=gen).half()
                qd = q.double().view(B, Hkv, n, D)
                s = (qd @ k.double().transpose(-1, -2)) * D ** -0.5
                ref = (torch.softmax(s, -1) @ v.double()).view(B, Hq, D)
                del s, qd
                fns, errs, notes = {}, {}, {}

                def add(name, make):
                    if name not in tools:
                        return
                    try:
                        fn = make()
                        out = fn()
                        torch.cuda.synchronize()
                        errs[name] = ((out.double().reshape(B, Hq, D) - ref).abs().max() / ref.abs().max()).item()
                        fns[name] = fn
                    except Exception as e:  # recorded, never hidden
                        notes[name] = f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
                        if os.environ.get("TS_BENCH_TRACE"):
                            traceback.print_exc()

                if Hq == Hkv:
                    q4 = q.view(B, Hq, 1, D)
                    add("ts", lambda: (lambda ck=ts_kernel((B, Hq, K.sym("T"), D), (B, Hkv, K.sym("N"), D)):
                                       lambda: ck(q=q4, k=k, v=v))())
                else:
                    qr = q.view(B, Hkv, n, D)
                    add("ts", lambda: (lambda ck=ts_kernel((B, Hkv, K.sym("T"), D), (B, Hkv, K.sym("N"), D)):
                                       lambda: ck(q=qr, k=k, v=v))())
                    q5 = q.view(B, Hkv, n, 1, D)
                    k5, v5 = k.view(B, Hkv, 1, N, D), v.view(B, Hkv, 1, N, D)
                    add("ts_bcast", lambda: (lambda ck=ts_kernel((B, Hkv, n, K.sym("T"), D), (B, Hkv, 1, K.sym("N"), D)):
                                             lambda: ck(q=q5, k=k5, v=v5))())
                q4s = q.view(B, Hq, 1, D)
                add("sdpa", lambda: (lambda: F.scaled_dot_product_attention(q4s, k, v, enable_gqa=Hq != Hkv)))
                for fmt in ("int8", "fp8"):
                    def make_ts_q(fmt=fmt):
                        kc, vc = KQ.quantize(k, fmt, group=32), KQ.quantize(v, fmt, group=32)
                        return lambda: KQ.decode_attention(q4s, kc, vc)
                    add(f"ts_{fmt}", make_ts_q)
                if flashinfer is not None:
                    kp, vp = paged(k, B, Hkv, N), paged(v, B, Hkv, N)
                    for name, tc in (("fi", False), ("fi_tc", True)):
                        def make_fi(tc=tc):
                            w = fi_wrapper(B, Hq, Hkv, N, torch.float16, tc)
                            return lambda: w.run(q, (kp, vp))
                        add(name, make_fi)

                    def make_fi8():
                        ks = (k.float().abs().amax() / 448.0).item()
                        vs = (v.float().abs().amax() / 448.0).item()
                        k8 = (kp.float() / ks).to(torch.float8_e4m3fn)
                        v8 = (vp.float() / vs).to(torch.float8_e4m3fn)
                        w = fi_wrapper(B, Hq, Hkv, N, torch.float8_e4m3fn, False)
                        return lambda: w.run(q, (k8, v8), k_scale=ks, v_scale=vs)
                    add("fi_fp8", make_fi8)
                graphs = {}
                for t, fn in fns.items():
                    try:
                        graphs[t] = graph_of(fn)
                    except Exception as e:  # recorded, never hidden
                        notes[t] = f"graph capture: {type(e).__name__}: {str(e)[:160]}"
                times = {t: [] for t in graphs}
                for _ in range(a.rounds):
                    for t, g in graphs.items():
                        times[t].append(triton.testing.do_bench(g.replay, warmup=3, rep=20,
                                                                return_mode="median") / 20 * 1e3)
                med = {t: statistics.median(x) for t, x in times.items()}
                row = dict(Hq=Hq, Hkv=Hkv, D=D, B=B, N=N, median_us=med,
                           min_us={t: min(x) for t, x in times.items()}, error=errs, notes=notes)
                rows.append(row)
                kv_bytes = 2 * B * Hkv * N * D * 2
                line = f"Hq{Hq} Hkv{Hkv} B{B} N{N}: " + "  ".join(
                    f"{t} {med[t]:.1f}us({kv_bytes / med[t] / 1e3:.0f}GB/s fp16eq, err {errs[t]:.1e})" for t in med)
                print(line, flush=True)
                for t, msg in notes.items():
                    print(f"    {t}: {msg}", flush=True)
                del graphs, fns
                torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"gpu": props.name, "sms": props.multi_processor_count, "torch": torch.__version__,
                       "flashinfer": fi_version, "rows": rows}, f, indent=1)
    if "--print-json" in sys.argv:
        print(json.dumps(rows))


if __name__ == "__main__":
    main()
